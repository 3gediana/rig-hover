#!/usr/bin/env python
"""demo.py -- RIG-Hover 分层决策中枢端到端演示。

跑的是真链路，只有"上层的思考时间"和"外部世界"是造出来的：

    人做了什么（合成观测，50Hz 虚拟时钟 + 实时节拍）
      -> CfC 输出 + 势函数      （真算）
      -> 升级判定（门 + 路由）  （真算，打印判据与置信度）
      -> 派发到 L2/L3           （真线程、真 JSON、真超时；后端可真可 mock）
      -> 上层回一条结构化指令    （真解析、真校验）
      -> 机器人按 50Hz 播放      （真执行器，落到 CfC 的动作空间）
      -> 交还 L0

用法::

    python demo.py                 # 有 DEEPSEEK_API_KEY 就真调，没有就 mock
    python demo.py --mock          # 强制 mock
    python demo.py --probe-net     # 额外测一次云端传输延迟（不需要 key）
    python demo.py --scenes S3,S4  # 只跑指定场景

整个 demo ~18 秒（按 50Hz 实时走；延迟数字必须是实时的才有意义）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

try:  # 控制台编码不是 UTF-8 时也不要因为一个表情符号炸掉
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

from hub import (
    CfcOut,
    DecisionHub,
    HubConfig,
    HubObs,
    LEVEL_NAMES,
    POTENTIALS,
    Stats,
    percentile,
)
from llm_backend import (
    CloudLLMBackend,
    MockAgentBackend,
    MockCloudBackend,
    load_api_key,
    probe_cloud_transport,
)

BAR = "=" * 78
SUB = "-" * 66


# --------------------------------------------------------------------------
# 场景：唯一被"造"出来的东西 —— 外部世界
# --------------------------------------------------------------------------


def _osc(t: float, f: float, amp: float) -> float:
    return amp * math.sin(2.0 * math.pi * f * t)


def scene_track(t: float) -> HubObs:
    """人站在 1.2m 处小幅左右移动。CfC 完全够用，应该一直是 L0。"""
    theta = _osc(t, 0.35, 12.0)
    return HubObs(
        t_now=t,
        cfc=CfcOut(head_des=theta * 0.8, vyaw_des=0.0, vx_des=0.0, mode=1, emote=0,
                   action_conf=0.94),
        person_conf=0.94, feat_age_ms=22.0, theta_img=theta, dist_est=1.18,
    )


def scene_wave(t: float) -> HubObs:
    """人挥手：本地 ACK（转头 + 表情），仍然不该升级。"""
    o = scene_track(t)
    if t > 0.45:
        o.cfc.mode = 2
        o.cfc.emote = 1
        o.cfc.action_conf = 0.90
    o.dist_est = 1.05
    return o


def scene_point(t: float) -> HubObs:
    """人指着桌上的杯子。本地没有物体识别与指代消解 -> L2（有帧、有算力）。"""
    o = scene_track(t)
    o.theta_img = 9.0
    o.cfc.head_des = 9.0
    if t > 0.70:
        o.pointing = 1.0
        o.point_dir_deg = 32.0
        o.cfc.action_conf = 0.55
    return o


def scene_speech_plan(t: float) -> HubObs:
    """一句话里两个动作。需要 ASR + 多步规划 -> L3。"""
    o = scene_track(t)
    o.theta_img = 4.0
    o.cfc.head_des = 4.0
    if t > 0.60:
        o.speech = 1.0
        o.instr_clauses = 2
        o.cfc.action_conf = 0.45
        o.cfc.mode = 1
    return o


def scene_speech_simple(t: float) -> HubObs:
    """一句话一个动作，但得先听懂人话。L3 最好、L2 也够用 —— 云端挂了就退 L2。"""
    o = scene_track(t)
    o.theta_img = 4.0
    o.cfc.head_des = 4.0
    if t > 0.60:
        o.speech = 1.0
        o.instr_clauses = 1
        o.cfc.action_conf = 0.5
        o.cfc.mode = 1
    return o


def scene_blind(t: float) -> HubObs:
    """光太暗：置信度塌了、特征大量出域 -> 本地扫描重捕获，别去打扰云端。"""
    o = scene_track(t)
    o.person_conf = 0.14
    o.clip_ratio = 0.62
    o.cfc.head_des = 0.0
    o.cfc.action_conf = 0.35
    o.cfc.mode = 1
    return o


def scene_lost(t: float) -> HubObs:
    """人走出视野 ~1s：本地重捕获（扫一圈），别急着找大脑。"""
    o = scene_track(t)
    o.seen = 0
    o.person_conf = 0.0
    o.feat_age_ms = 900.0 + 100.0 * t
    o.cfc.head_des = 0.0
    o.cfc.vx_des = 0.0
    o.cfc.mode = 1
    o.cfc.action_conf = 0.6
    return o


# --------------------------------------------------------------------------
# 打印
# --------------------------------------------------------------------------


class Recorder:
    """收集一个场景的事件与轨迹，并把关键事件立刻打印出来。"""

    def __init__(self, name: str, desc: str, show_json: bool = False):
        self.name = name
        self.desc = desc
        self.show_json = show_json
        self.levels: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.max_level = 0
        self.exec_levels: set[int] = set()
        self.ticks = 0
        self.last_action: dict | None = None
        self.t_escalated: float | None = None
        self.t_first_cmd: float | None = None
        self._last_level = 0
        self._key: tuple | None = None
        self._json_shown = False

    def _line(self, t: float, text: str) -> None:
        print(f"  {t:5.2f}s  {text}")

    def feed(self, t: float, out) -> None:
        self.ticks += 1
        self.levels[out.level] = self.levels.get(out.level, 0) + 1
        self.last_action = out.action.as_dict()
        p = out.payload
        status = p.get("status", "")
        d = out.decision

        # ---- 层级切换：这一行就是"为什么升级"的证据 ----
        if out.level != self._last_level:
            self.max_level = max(self.max_level, out.level)
            if out.level > 0:
                self.t_escalated = t if self.t_escalated is None else self.t_escalated
                pfix = " ".join(f"L{k}:{v:.2f}" for k, v in (d.pfix.items() if d else []))
                why = p.get("why") or (d.reason_text() if d else "")
                verb = "升级" if out.level > self._last_level else "转交"
                self._line(t, f"{verb} L{self._last_level} -> L{out.level} "
                              f"({LEVEL_NAMES[out.level]})  "
                              f"p_esc={p.get('p_esc', 0):.3f}  P_fix={{{pfix}}}")
                self._line(t, f"        判据 {why}")
                if d and d.top_family:
                    self._line(t, f"        风险族 {d.top_family}"
                                  f"（决定本地技能与路由方向），"
                                  f"U(L)=" + " ".join(
                                      f"L{k}:{v:+.2f}" for k, v in d.utility.items() if k))
                if out.level == 1:
                    self._line(t, "        本地技能：零网络、当拍开始")
            else:
                self._line(t, f"交还 L0  ({p.get('status') or p.get('gated_by') or 'idle'})"
                              + (f"  技能共 {p.get('skill_ms', 0):.0f}ms"
                                 if p.get("skill_ms") else ""))
            self._last_level = out.level

        # ---- 状态行去重 ----
        key = (out.level, status, p.get("step", ""))
        if status == "waiting" and key != self._key:
            self._key = key
            self._line(t, f"        -> 派发 {p.get('backend')} ttl={p.get('ttl_ms')}ms "
                          f"（等待期仍 50Hz 出指令：车轮清零、头继续跟人）")
            if self.show_json and not self._json_shown:
                req = dict(p.get("request") or {})
                req.pop("phi", None)   # 势函数已单独打印成判据，这里不重复
                js = json.dumps(req, ensure_ascii=False)
                self._line(t, f"        -> 上行 JSON {js[:300]}"
                              + (" ...}" if len(js) > 300 else ""))
        elif status == "executing" and key != self._key:
            self._key = key
            self.exec_levels.add(out.level)
            if self.t_first_cmd is None:
                self.t_first_cmd = t
            ins = p.get("instruction") or {}
            prog = " → ".join(
                s["op"] + (f" {s.get('deg', s.get('dps', s.get('mps', s.get('id'))))}"
                           if s["op"] in ("head", "yaw", "vx", "emote") else "")
                + f"/{s.get('ms', 0):.0f}ms"
                for s in ins.get("prog", [])
            )
            self._line(t, f"        执行 [{p.get('step')}] {p.get('intent')} "
                          f"<- {p.get('source')}   {prog}"
                          + (f"   say={p.get('say')!r}" if p.get("say") else ""))
            if self.show_json and not self._json_shown:
                self._json_shown = True
                js = json.dumps(ins, ensure_ascii=False)
                self._line(t, f"        <- 下行 JSON {js[:260]}"
                              + (" ...}" if len(js) > 260 else ""))
        elif status == "exec_done" and key != self._key:
            self._key = key
            self._line(t, f"        尝试链 {[a['level'] for a in p.get('attempts', [])]}  "
                          f"指令生效时间 {p.get('skill_ms', 0):.0f}ms")
        elif out.level == 0 and p.get("gated_by", "").startswith("veto") and key != self._key:
            self._key = key
            self._line(t, f"        安全否决 {p['gated_by']}：不升级，只本地处置")

    def summary(self) -> str:
        hist = " ".join(f"L{k}:{v}" for k, v in self.levels.items() if v)
        ran = max(self.exec_levels) if self.exec_levels else 0
        return f"ticks={self.ticks} 实际执行最高=L{ran} 触达最高=L{self.max_level} 分布[{hist}]"


def run_scene(hub: DecisionHub, rec: Recorder, scene, duration: float) -> Recorder:
    """按 50Hz 跑一个场景。时间轴用虚拟时钟步进，节拍跟真实墙钟对齐。"""
    print(f"\n[{rec.name}] {rec.desc}")
    t0 = time.perf_counter()
    step = 0
    while step * 0.02 < duration:
        t = step * 0.02
        out = hub.tick(scene(t), 0.02)
        rec.feed(t, out)
        step += 1
        gap = t0 + step * 0.02 - time.perf_counter()
        if gap > 0:
            time.sleep(gap)
    print(f"  {SUB}")
    print(f"  小结: {rec.summary()}")
    return rec


# --------------------------------------------------------------------------
# 统计与报告
# --------------------------------------------------------------------------


def merge_stats(hubs: list[DecisionHub]) -> Stats:
    """把每个场景各自一份的统计合成一份（场景之间不共享状态机）。"""
    m = Stats()
    for h in hubs:
        s = h.stats
        m.ticks += s.ticks
        m.escalations += s.escalations
        m.local_us.extend(s.local_us)
        for lvl in (1, 2, 3):
            m.dispatch_ms[lvl].extend(s.dispatch_ms[lvl])
            m.skill_ms[lvl].extend(s.skill_ms[lvl])
        m.timeouts += s.timeouts
        m.failures += s.failures
        m.fallbacks += s.fallbacks
        m.invalid_input += s.invalid_input
        m.invalid_instruction += s.invalid_instruction
        m.stale_reply += s.stale_reply
        m.vetoed += s.vetoed
        for k, v in s.by_level.items():
            m.by_level[k] = m.by_level.get(k, 0) + v
    return m


def hot_loop_us(n: int = 3000, scene=scene_track) -> list[float]:
    """纯计算路径的热循环耗时（不 sleep、不睡醒），用来区分"算法成本"和
    "实时循环里被操作系统/CPU 电源状态加上的开销"。"""
    h = DecisionHub(HubConfig())
    ts: list[float] = []
    for i in range(n):
        obs = scene(1.0)
        obs.t_now = 1.0 + i * 0.02
        a = time.perf_counter()
        h.tick(obs, 0.02)
        ts.append((time.perf_counter() - a) * 1e6)
    h.close()
    return ts


def print_budget(master: Stats, probe: dict | None, real_llm: bool, mock_l2: float,
                 mock_l3: float, hot_us: list[float] | None = None) -> None:
    """延迟预算表。每一行标注来源，不把建模值混进实测值。"""
    us = master.local_us
    d2, d3 = master.dispatch_ms[2], master.dispatch_ms[3]
    s1 = master.skill_ms[1]

    rows = [
        ("L0", "决策链：势函数+门+路由+清洗（纯 Python）",
         f"{percentile(us, 0.5):.1f}us", f"{percentile(us, 0.95):.1f}us",
         f"{max(us):.0f}us" if us else "-", str(len(us)), "实测（50Hz 实时循环内）"),
    ]
    if hot_us:
        rows.append(("L0", "同上但热循环（无 sleep，纯计算成本）",
                     f"{percentile(hot_us, 0.5):.1f}us",
                     f"{percentile(hot_us, 0.95):.1f}us",
                     f"{max(hot_us):.0f}us", str(len(hot_us)), "实测（热循环）"))
    rows += [
        ("L0", "CfC 推理（ESP32-S3, hidden=64, 32k MAC）",
         "~0.6ms", "~1.5ms", "-", "-", "引用 TECH-PLAN §6.4"),
        ("L1", "本地技能播放（零网络；这里是程序时长）",
         f"{percentile(s1, 0.5):.0f}ms" if s1 else "-",
         f"{percentile(s1, 0.95):.0f}ms" if s1 else "-", "-", str(len(s1)),
         "实测（程序时长）"),
        ("L2", f"pc_agent 往返（含建模思考 {mock_l2 * 1000:.0f}ms）",
         f"{percentile(d2, 0.5):.1f}ms" if d2 else "-",
         f"{percentile(d2, 0.95):.1f}ms" if d2 else "-",
         f"{max(d2):.1f}ms" if d2 else "-", str(len(d2)), "实测（线程+JSON 序列化）"),
        ("L3", f"cloud 往返（含建模思考 {mock_l3 * 1000:.0f}ms）",
         f"{percentile(d3, 0.5):.1f}ms" if d3 else "-",
         f"{percentile(d3, 0.95):.1f}ms" if d3 else "-",
         f"{max(d3):.1f}ms" if d3 else "-", str(len(d3)), "实测（线程+JSON 序列化）"),
    ]
    if probe is not None:
        if probe.get("ok"):
            rows.append(("L3", f"云端 TLS+HTTP 传输下界（无 key 探测，HTTP {probe.get('status')}）",
                         f"{probe['ms']:.0f}ms", "-", "-", "1", "实测（真实网络）"))
        else:
            rows.append(("L3", "云端传输探测", "失败", "-", "-", "1",
                         str(probe.get("note"))[:26]))

    print(f"\n{BAR}")
    print("延迟预算表  （『实测』= 本机真实测量；『建模』= sleep 出来的假思考时间）")
    print(BAR)
    print(f"{'级别':<3} {'环节':<38} {'p50':>9} {'p95':>9} {'max':>8} {'n':>5}  来源")
    for lvl, what, p50, p95, mx, n, src in rows:
        print(f"{lvl:<3} {what:<38} {p50:>9} {p95:>9} {mx:>8} {n:>5}  {src}")
    print(BAR)
    print(f"上层后端: {'真实 DeepSeek API' if real_llm else 'mock（未发现 DEEPSEEK_API_KEY）'}"
          "  |  门控/路由权重: 先验（无标注数据；fit_gate / fit_router 可训练）")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


SCENES = [
    ("S1", "人站着小幅走动：本地够用，不该升级", scene_track, 1.0, "healthy", 0),
    ("S2", "人挥手：本地 ACK（转头+表情），仍然不升级", scene_wave, 1.1, "healthy", 0),
    ("S3", "人指着桌上的杯子：本地没物体识别 -> L2 取帧 -> 回 look_at", scene_point, 2.6, "healthy", 2),
    ("S4", "一句话两个动作：本地没 ASR/规划 -> L3 -> 回多步程序", scene_speech_plan, 3.2, "healthy", 3),
    ("S5", "云端 600ms 未回（超出预算）：L3 超时 -> L2 接住", scene_speech_simple, 3.6, "cloud_timeout", 2),
    ("S6", "云端和 agent 全挂：退到 L1 本地技能，机器人不僵住", scene_speech_plan, 2.6, "all_down", 1),
    ("S7", "光太暗 + 特征出域：问云端没用，本地扫描重捕获", scene_blind, 1.8, "healthy", 1),
    ("S8", "人走出视野 ~1s：本地重捕获，别急着找大脑", scene_lost, 1.8, "healthy", 1),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RIG-Hover 分层决策中枢演示")
    ap.add_argument("--mock", action="store_true", help="强制使用 mock 后端")
    ap.add_argument("--probe-net", action="store_true", help="额外做一次云端传输探测")
    ap.add_argument("--scenes", default="", help="只跑指定场景，逗号分隔，如 S3,S4")
    args = ap.parse_args(argv)

    key = "" if args.mock else load_api_key()
    real_llm = bool(key)
    mock_l2, mock_l3 = 0.085, 0.85

    print(BAR)
    print("RIG-Hover 分层决策中枢 · 端到端演示")
    print("L0 反射(本地 CfC) | L1 技能(本地) | L2 PC agent | L3 云端 LLM")
    print(BAR)
    print(f"python={sys.version.split()[0]}  DEEPSEEK_API_KEY={'有' if key else '无'}"
          f"  → 上层用 {'真实 API' if real_llm else 'mock'}")
    print("势函数 K=%d: %s" % (len(POTENTIALS), " ".join(p.name for p in POTENTIALS)))

    want = {s.strip().upper() for s in args.scenes.split(",") if s.strip()}
    picked = [s for s in SCENES if not want or s[0] in want]
    hubs: list[DecisionHub] = []
    recs: list[Recorder] = []

    for tag, desc, fn, dur, mode, expect_lvl in picked:
        backends: dict = {2: MockAgentBackend(latency_s=mock_l2)}
        backends[3] = CloudLLMBackend(api_key=key) if real_llm else MockCloudBackend(latency_s=mock_l3)
        cfg = HubConfig()
        if mode == "cloud_timeout":
            backends[3] = MockCloudBackend(latency_s=0.90, fail_mode="timeout")
            cfg.request_ttl_ms = {1: 0, 2: 900, 3: 600}   # 云端预算压到 600ms
        elif mode == "all_down":
            backends[3] = MockCloudBackend(fail_mode="down")
            backends[2] = MockAgentBackend(fail_mode="down")
        hub = DecisionHub(cfg, backends=backends)
        hubs.append(hub)
        rec = Recorder(f"{tag}", desc, show_json=(tag in ("S3", "S4")))
        run_scene(hub, rec, fn, dur)
        recs.append(rec)
        hub.close()

    master = merge_stats(hubs)
    probe = None
    if args.probe_net:
        print("\n探测云端传输延迟（不带 key，预期 401）...")
        probe = probe_cloud_transport()
        print(f"  -> {probe}")

    print_budget(master, probe, real_llm, mock_l2, mock_l3, hot_loop_us())

    print("\n场景判定（期望的『实际执行层级』由设计给出，不是事后改的）")
    ok_all = True
    for rec, (tag, desc, fn, dur, mode, expect_lvl) in zip(recs, picked):
        got = max(rec.exec_levels) if rec.exec_levels else 0
        ok = got == expect_lvl
        ok_all &= ok
        print(f"  [{'OK  ' if ok else 'MISS'}] {rec.name} 期望执行 L{expect_lvl}，实际 L{got}"
              f"   {rec.summary()}")

    print(f"\n{BAR}")
    print("汇总")
    print(BAR)
    print(f"L0 决策 p50={percentile(master.local_us, 0.5):.1f}us "
          f"p95={percentile(master.local_us, 0.95):.1f}us max={max(master.local_us):.0f}us "
          f"样本={len(master.local_us)}")
    print(f"升级={master.escalations} 超时={master.timeouts} 失败={master.failures} "
          f"降级={master.fallbacks} 非法输入={master.invalid_input} "
          f"非法指令={master.invalid_instruction} 迟到回复={master.stale_reply} "
          f"安全否决={master.vetoed}")
    print("层级执行分布: " + " ".join(f"{LEVEL_NAMES[k]}:{v}"
                                       for k, v in sorted(master.by_level.items())))
    print(BAR)
    print("全部场景按预期分层：PASS" if ok_all else "有场景未按预期分层：见上")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
