#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_real_backend.py -- 真实 L2（agent_bridge）的验收测试。

    python test_real_backend.py

覆盖三块（对应任务 A/B/C）：

A. **接得上**：``LocalAgentBackend`` 把 esc/1 请求翻成 agent_bridge 的意图，
   agent 的真实执行结果再翻回通过 ``parse_instruction`` 校验的指令；每个动作的
   detail 都用本机事实独立核对（时间对表、磁盘对账、日志对行），占位符过不了。
B. **端到端**：DecisionHub 在真实配置（dwell=150ms / EMA 0.15）下判定升级 ->
   真的在 PC 上写了一条日志 -> 结构化指令回到 L0 播放。对照组（L2 不可用时）
   证明这条副作用确实来自 L2。
C. **对照实验**（每种都实测延迟分布，不是"会回退"四个字）：
     1) 正常        L2 可用           -> 升级成功
     2) agent 未启动 L2 不可达         -> 同拍落 L1；零调用、零失败、零崩溃
     3) agent 超时   任务端点延迟 1.5s  -> 在 ttl 预算内回退 L1

D. **失败模式**：HTTP 404 / 非 JSON 响应 / 服务中途被杀，全部映射成
   ``BackendError`` / ``BackendUnavailable``，交给 hub 已有的降级路径。

测试会真的拉起 ``agent_bridge/server.py`` 子进程（随机空闲端口），用完收掉；
超时场景用 ``slow_bridge.py``（import 同一份 server.py，只给 ``/agent/task``
注入延迟），**不修改** agent_bridge 的源码。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # 控制台编码不是 UTF-8 时也不要因为一个汉字炸掉
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

from bridge_proc import BridgeProcess, SLOW_PY, free_port
from demo_real import scene_point
from detail_check import check_detail, notes_path
from hub import (
    DecisionHub,
    HubConfig,
    HubObs,
    POTENTIAL_NAMES,
    percentile,
)
from llm_backend import (
    BackendError,
    BackendUnavailable,
    build_escalation_request,
)
from local_agent import ACTION_PROG, FAMILY_INTENT, LocalAgentBackend

BAR = "=" * 78
PASS = 0
FAIL = 0
FAILED: list[str] = []

#: 环境里没有 key 时 agent 走规则 planner，动作可预测；有 key 时不强求同一动作。
NO_KEY = not (os.environ.get("DEEPSEEK_API_KEY") or "").strip()


def check(name: str, cond: bool, detail: str = "") -> bool:
    """记一条断言（不抛，跑完汇总）。"""
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def expect_raises(name: str, exc, fn, *a, **k) -> None:
    """断言 fn 抛指定异常类型。"""
    try:
        fn(*a, **k)
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:70]}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"抛了 {type(e).__name__}: {e}")
    else:
        check(name, False, "没有抛异常")


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def fast_cfg(**kw) -> HubConfig:
    """测试用配置：关掉触发驻留与离散量平滑，让升级在当拍发生。"""
    kw.setdefault("dwell_ms", 0)
    kw.setdefault("ema_tau", {n: 0.0 for n in POTENTIAL_NAMES})
    return HubConfig(**kw)


def esc_request(potential: str, *, esc_id: int = 7, p_esc: float = 0.80,
                ttl_ms: int = 900) -> dict:
    """造一条 esc/1 请求：first 归因是该势函数（决定风险族与意图）。"""
    return build_escalation_request(
        esc_id=esc_id,
        level=2,
        p_esc=p_esc,
        phi={n: 0.0 for n in POTENTIAL_NAMES},
        contributions=[(potential, 2.0), ("u_conf", 0.1)],
        obs={"seen": 1, "person_conf": 0.90, "feat_age_ms": 20.0},
        ttl_ms=ttl_ms,
    )


def scene_escalate(t: float) -> HubObs:
    """把 demo 的指物场景右移 0.62s：第一个 tick 就已经越过触发点。"""
    return scene_point(t + 0.62)


def run_trial(scene, be: LocalAgentBackend | None, cfg: HubConfig,
              *, max_s: float = 2.0, pace: float = 0.002) -> dict:
    """跑一次"决策 -> 派发 -> 拿到指令（或回退）"，把延迟和计数都记下来。

    Returns:
        含 crash / levels / t_dec / t_exec / dispatch_ms / waiting_ticks /
        timeouts / failures / fallbacks / stats 的字典。
    """
    backends = {2: be} if be is not None else {}
    hub = DecisionHub(cfg, backends=backends)
    t0 = time.perf_counter()
    rec: dict = {
        "crash": "", "levels": set(), "exec": False, "level_exec": None,
        "t_dec": None, "t_exec": None, "waiting_ticks": 0, "bad_wait": 0,
        "dispatch_ms": 0.0, "timeouts": 0, "failures": 0, "fallbacks": 0,
    }
    try:
        while True:
            t = time.perf_counter() - t0
            if t > max_s:
                break
            try:
                out = hub.tick(scene(t), 0.02)
            except Exception as exc:  # noqa: BLE001 - 这里就是要证明它不会抛
                rec["crash"] = f"{type(exc).__name__}: {exc}"
                break
            rec["levels"].add(out.level)
            d, p = out.decision, out.payload
            if rec["t_dec"] is None and d is not None and d.level > 0:
                rec["t_dec"] = t
            if p.get("status") == "waiting":
                rec["waiting_ticks"] += 1
                a = out.action
                if a.vx_des != 0.0 or a.vyaw_des != 0.0:
                    rec["bad_wait"] += 1
            if p.get("status") == "executing" and rec["t_exec"] is None:
                rec["t_exec"] = t
                rec["exec"] = True
                rec["level_exec"] = out.level
                rec["instruction"] = p.get("instruction") or {}
                break
            time.sleep(pace)
        rec["dispatch_ms"] = (hub.stats.dispatch_ms[2][-1]
                              if hub.stats.dispatch_ms[2] else 0.0)
        rec["timeouts"] = hub.stats.timeouts
        rec["failures"] = hub.stats.failures
        rec["fallbacks"] = hub.stats.fallbacks
        rec["stats"] = hub.stats.summary()
    finally:
        hub.close()
    return rec


def _dist(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs):<3} p50={percentile(xs, 0.5):7.1f}ms "
            f"p95={percentile(xs, 0.95):7.1f}ms max={max(xs):7.1f}ms")


# --------------------------------------------------------------------------
# A. 协议翻译 + 真实执行
# --------------------------------------------------------------------------


def test_translation(be: LocalAgentBackend) -> dict:
    print("\n[A] 协议翻译 + 真实执行（每条 detail 用本机事实核对）")
    be.probe()
    check("健康检查：agent 可达且 ok=1", be.available(),
          f"actions={be.actions} llm={be.llm}")
    check("白名单覆盖 clock/disk/note/screenshot/system_status",
          set(be.actions) == {"clock", "disk", "note", "screenshot", "system_status"},
          str(be.actions))

    cases = [
        # (风险族, 首要势函数, 期望动作)
        ("language", "m_point", "note"),
        ("unknown", "u_ood", "system_status"),
        ("stakes", "s_approach", "disk"),
        ("context", "m_stale_sem", "clock"),
    ]
    first_detail: dict = {}
    for family, potential, want in cases:
        req = esc_request(potential, p_esc=0.8)
        before = _note_lines()
        ins = be.ask(req, 0.9)
        action = str(ins.args.get("action", ""))
        if NO_KEY:
            check(f"风险族 {family}（{potential}）-> agent 自己选 {want}",
                  action == want, f"action={action} planner={ins.args.get('planner')}")
        else:
            check(f"风险族 {family}（{potential}）-> agent 返回白名单动作",
                  action in ACTION_PROG, f"action={action} planner={ins.args.get('planner')}")
        ok, ev = check_detail(action, ins.args.get("detail", ""))
        if action == "note":
            after = _note_lines()
            ok = ok and after == before + 1
            ev += f"；robot_notes.log {before} -> {after} 行"
        check(f"{action} 的 detail 是真实内容（非占位符）", ok, ev)
        check(f"{action} 的指令通过 parse_instruction 且 id 对得上",
              ins.level == 2 and ins.esc_id == req["id"] and bool(ins.prog)
              and ins.args.get("roundtrip_ms", 0) > 0,
              f"intent={ins.intent} prog={len(ins.prog)}步 say={ins.say!r}")
        first_detail.setdefault(action, ins.args.get("detail", ""))

    # 意图本身带归因，LLM planner 才有上下文
    intent = be.intent_for(esc_request("m_point"))
    check("意图里带归因与 p_esc（给 LLM planner 的上下文）",
          "m_point" in intent and "p_esc" in intent, intent[:60])
    check("风险族 -> 意图表覆盖全部 6 个族",
          set(FAMILY_INTENT) == {"blind", "unknown", "language", "planning",
                                 "context", "stakes"}, str(sorted(FAMILY_INTENT)))

    # 显式 action hint：跳过 planner，走 bridge 的 explicit 分支
    forced = LocalAgentBackend(be.base_url, action_hint="clock", start_health_thread=False)
    ins = forced.ask(esc_request("u_ood"), 0.9)
    check("action_hint 走 explicit 分支（planner=explicit, action=clock）",
          ins.args.get("planner") == "explicit" and ins.args.get("action") == "clock",
          f"planner={ins.args.get('planner')} action={ins.args.get('action')}")
    check("forced 指令同样通过校验", ins.esc_id == 7 and ins.level == 2)
    forced.close()

    check("一次真实回包原文已留存（demo 打印用）",
          isinstance(be.last_raw, str) and be.last_raw.startswith("{"), be.last_raw[:60])
    return first_detail


# --------------------------------------------------------------------------
# B. 端到端：决策 -> 真实副作用 -> 结构化指令
# --------------------------------------------------------------------------


def test_end_to_end(be: LocalAgentBackend) -> dict:
    print("\n[B] 端到端：DecisionHub 判定升级 -> PC 上真写一条日志 -> 指令回到 L0")
    before = _note_lines()

    # 对照组：没有 L2 后端时，同一场景只能落 L1，日志不该多一行
    ctrl = run_trial(scene_point, None, HubConfig(), max_s=1.4, pace=0.02)
    check("对照组（无 L2）：落 L1 且没有写日志",
          ctrl["exec"] and ctrl["level_exec"] == 1 and _note_lines() == before,
          f"exec=L{ctrl['level_exec']} notes={_note_lines() - before:+d}")

    # 实验组：真实配置（dwell=150ms / EMA 0.15）+ 真实 agent
    t0 = time.perf_counter()
    hub = DecisionHub(HubConfig(), backends={2: be})
    outs = []
    instruction: dict = {}
    raw = ""
    detail = ""
    action = ""
    t_dec = t_exec = None
    notes_before = _note_lines()
    try:
        while time.perf_counter() - t0 < 3.0:
            t = time.perf_counter() - t0
            out = hub.tick(scene_point(t), 0.02)
            outs.append(out)
            if t_dec is None and out.decision is not None and out.decision.level > 0:
                t_dec = t
            if out.payload.get("status") == "executing" and not instruction:
                instruction = out.payload.get("instruction") or {}
                t_exec = t
                raw = be.last_raw
                action = str(instruction.get("args", {}).get("action", ""))
                detail = str(instruction.get("args", {}).get("detail", ""))
            if out.payload.get("status") == "exec_done":
                break
            time.sleep(0.02)
        stats = hub.stats.summary()
    finally:
        hub.close()

    after = _note_lines()
    check("升级真的发生并被派发到 L2", stats["escalations"] >= 1
          and len(hub.stats.dispatch_ms[2]) >= 1,
          f"escalations={stats['escalations']} L2 样本={len(hub.stats.dispatch_ms[2])}")
    check("拿到的是 L2 指令（source=local-agent），不是本地技能",
          instruction.get("source") == "local-agent" and instruction.get("level") == 2,
          f"source={instruction.get('source')} level={instruction.get('level')}")
    check("agent 执行了 note，日志真的多了一行（真实副作用）",
          action == "note" and after == notes_before + 1,
          f"action={action} notes {notes_before} -> {after}")
    ok, ev = check_detail(action, detail)
    check("这次真实执行的 detail 经独立核对", ok, ev)
    check("等待期仍按 50Hz 出指令且车轮清零",
          sum(1 for o in outs if o.payload.get("status") == "waiting") >= 1
          and all(o.action.vx_des == 0.0 and o.action.vyaw_des == 0.0
                  for o in outs if o.payload.get("status") == "waiting"),
          f"等待拍={sum(1 for o in outs if o.payload.get('status') == 'waiting')}")
    check("场景按 50Hz 跑满且拿到了指令", len(outs) > 10 and bool(instruction),
          f"ticks={len(outs)}")
    lat = (t_exec - t_dec) * 1000.0 if (t_dec is not None and t_exec is not None) else -1.0
    print(f"     真实回包原文: {raw}")
    print(f"     结构化指令  : {json.dumps(instruction, ensure_ascii=False)[:220]}...")
    print(f"     判据超线 -> 第一条 L2 指令: {lat:.1f}ms（含 150ms 驻留）")
    return {"lat": lat, "raw": raw, "detail": detail, "action": action}


def _note_lines() -> int:
    try:
        return len([ln for ln in notes_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()])
    except OSError:
        return 0


# --------------------------------------------------------------------------
# C. 对照实验
# --------------------------------------------------------------------------


def test_normal(be: LocalAgentBackend, trials: int = 15) -> dict:
    print(f"\n[C1] 正常：L2 可用（{trials} 次升级，每次全新 hub）")
    be.ask(esc_request("m_point"), 0.9)          # 预热：第一次请求含连接建立
    be.roundtrip_ms.clear()
    runs = [run_trial(scene_escalate, be, fast_cfg()) for _ in range(trials)]
    disp = [r["dispatch_ms"] for r in runs]
    rt = list(be.roundtrip_ms)
    ok_all = all(r["exec"] and r["level_exec"] == 2 and not r["crash"] for r in runs)
    check(f"全部 {trials} 次都升级到 L2 并拿到指令", ok_all,
          f"levels={[sorted(r['levels']) for r in runs][:3]}...")
    check("hub 观测延迟样本齐全", len(disp) == trials, f"n={len(disp)}")
    check("正常路径没有超时/失败", all(r["timeouts"] == 0 and r["failures"] == 0
                                   for r in runs))
    check("每一次都落在 900ms 预算内", max(disp) < 900.0, _dist(disp))
    print(f"      hub 观测 (50Hz tick 粒度): {_dist(disp)}")
    print(f"      客户端 HTTP 往返        : {_dist(rt)}")
    return {"dispatch": disp, "roundtrip": rt, "runs": runs}


def test_agent_down(trials: int = 10) -> dict:
    print(f"\n[C2] agent 没启动：{trials} 次升级，每次都要求同拍落 L1 且不崩")
    dead = LocalAgentBackend(f"http://127.0.0.1:{free_port()}", name="dead-agent",
                             start_health_thread=False)
    check("不可达的 agent -> available() 为假", dead.available() is False,
          dead.last_error[:60])
    runs = [run_trial(scene_escalate, dead, fast_cfg()) for _ in range(trials)]
    lat = [((r["t_exec"] or 0.0) - (r["t_dec"] or 0.0)) * 1000.0 for r in runs]
    check(f"全部 {trials} 次都执行了 L1（机器人不僵住）",
          all(r["exec"] and r["level_exec"] == 1 for r in runs))
    check("L2 一次都没被调用（不是失败回退，是根本不进候选）", dead.calls == 0,
          f"calls={dead.calls}")
    check("没有超时/失败/崩溃", all(r["timeouts"] == 0 and r["failures"] == 0
                               and not r["crash"] for r in runs))
    at_l2 = [r for r in runs if 2 in r["levels"]]
    check("L2 从未出现在输出层级里", not at_l2, f"触达 L2 的 trial={len(at_l2)}")
    check("回退是当拍完成的（0ms 额外等待）", max(lat) < 20.0, _dist(lat))
    print(f"      判据超线 -> 第一条 L1 指令: {_dist(lat)}")
    dead.close()
    return {"lat": lat, "runs": runs}


def test_timeout(slow_url: str, trials: int = 10, ttl_ms: int = 300) -> dict:
    print(f"\n[C3] agent 超时：任务端点延迟 1500ms，ttl={ttl_ms}ms（{trials} 次）")
    be = LocalAgentBackend(slow_url, name="slow-agent")
    try:
        check("慢 agent 的健康检查仍然正常（延迟只作用于 /agent/task）",
              be.wait_healthy(10.0), f"probe={[round(x, 1) for x in be.probe_ms[-2:]]}")
        cfg = fast_cfg(request_ttl_ms={1: 0, 2: ttl_ms, 3: 2500}, total_budget_ms=3500)
        runs = [run_trial(scene_escalate, be, cfg, max_s=ttl_ms / 1000.0 + 1.2)
                for _ in range(trials)]
        disp = [r["dispatch_ms"] for r in runs]
        check(f"全部 {trials} 次都在预算内回退到 L1",
              all(r["exec"] and r["level_exec"] == 1 for r in runs))
        check("超时被计数（不是静默失败）", all(r["timeouts"] >= 1 for r in runs),
              f"timeouts={[r['timeouts'] for r in runs][:5]}")
        check("降级被计数", all(r["fallbacks"] >= 1 for r in runs))
        check("零崩溃", all(not r["crash"] for r in runs),
              str([r["crash"] for r in runs if r["crash"]])[:80])
        budget_ok = max(disp) < ttl_ms * 1.6
        check(f"实测回退延迟没有超出 ttl 预算的量级（max < {ttl_ms * 1.6:.0f}ms）",
              budget_ok, _dist(disp))
        wait = [r["waiting_ticks"] for r in runs]
        check("等待期间 L0 一直在出指令（车轮清零）",
              min(wait) > 10 and all(r["bad_wait"] == 0 for r in runs),
              f"最少等待拍={min(wait)}")
        check("慢 agent 的 socket 超时 = 预算 × 0.9", abs(
            be._socket_timeout(ttl_ms / 1000.0) - ttl_ms / 1000.0 * 0.9) < 1e-6,
            f"{be._socket_timeout(ttl_ms / 1000.0) * 1000:.0f}ms")
        print(f"      hub 观测（=回退时刻）: {_dist(disp)}")
        return {"dispatch": disp, "runs": runs}
    finally:
        be.close()


def test_timeout_production_ttl(slow_url: str, trials: int = 3) -> dict:
    print(f"\n[C3b] 同一个慢 agent，换成生产默认 ttl=900ms（{trials} 次）")
    be = LocalAgentBackend(slow_url, name="slow-agent-900")
    try:
        be.wait_healthy(10.0)
        cfg = fast_cfg(request_ttl_ms={1: 0, 2: 900, 3: 2500}, total_budget_ms=3500)
        runs = [run_trial(scene_escalate, be, cfg, max_s=2.0) for _ in range(trials)]
        disp = [r["dispatch_ms"] for r in runs]
        check("900ms 预算下同样在预算内回退", all(
            r["exec"] and r["level_exec"] == 1 and r["timeouts"] >= 1 for r in runs))
        check("回退延迟 < 1000ms", max(disp) < 1000.0, _dist(disp))
        print(f"      hub 观测: {_dist(disp)}")
        return {"dispatch": disp, "runs": runs}
    finally:
        be.close()


# --------------------------------------------------------------------------
# D. 失败模式
# --------------------------------------------------------------------------

_BAD_HTML = b"<html><body>not json</body></html>"


class _BadJsonHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", "0") or 0)
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(_BAD_HTML)))
        self.end_headers()
        self.wfile.write(_BAD_HTML)

    def do_GET(self):  # noqa: N802
        self.do_POST()

    def log_message(self, fmt, *a):  # 静音
        return


def test_failure_modes(be: LocalAgentBackend) -> None:
    print("\n[D] 失败模式：错误必须映射成 hub 已知的失败类型，不能穿透成异常")
    # 1) 404：路径写错 / 服务换了契约
    wrong = LocalAgentBackend(be.base_url, task_path="/agent/nope", name="wrong-path",
                              start_health_thread=False)
    expect_raises("404 -> BackendError", BackendError, wrong.ask,
                  esc_request("m_point"), 0.5)
    check("404 是应用层错误，不把这一级判成不可达（可用性只反映可达性）",
          wrong.available() is True, wrong.last_error[:60])
    wrong.close()

    # 2) 200 但不是 JSON
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _BadJsonHandler)
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        bad = LocalAgentBackend(f"http://127.0.0.1:{port}", name="bad-json",
                                start_health_thread=False)
        expect_raises("200 + 非 JSON -> BackendError", BackendError,
                      bad.ask, esc_request("m_point"), 0.5)
        bad.close()
    finally:
        srv.shutdown()
        srv.server_close()

    # 3) 服务中途被杀：真实请求失败 -> 立刻判不可用 -> hub 落 L1
    print("     杀掉 agent 子进程，验证运行期掉线")
    be.probe()
    check("掉线前可用", be.available() is True)


def test_killed_midflight(be: LocalAgentBackend, bridge: BridgeProcess) -> dict:
    print("\n[D2] 运行期 agent 被杀：请求失败 -> 回退 L1 -> 不崩溃")
    check("杀之前 available() 仍为真（缓存还没过期）", be.available() is True)
    logs = bridge.stop()                       # 真的把它杀掉
    check("子进程已退出", not bridge.alive(),
          logs.strip().splitlines()[-1] if logs.strip() else "")

    # 第一次：hub 仍会把它派发到 L2（缓存还没过期），在建连上失败，然后回退 L1
    first = run_trial(scene_escalate, be, fast_cfg(), max_s=1.5)
    lat_first = ((first["t_exec"] or 0.0) - (first["t_dec"] or 0.0)) * 1000.0
    check("失败类型是 BackendUnavailable（建连阶段，不是超时）",
          be.last_error.startswith("BackendUnavailable"), be.last_error[:80])
    check("第一次掉线：真的先试了 L2 再回退 L1",
          first["failures"] >= 1 and first["level_exec"] == 1,
          f"failures={first['failures']} exec=L{first['level_exec']}")
    check("回退延迟 ≈ 建连超时，远小于 0.9×ttl=810ms", lat_first < 400.0,
          f"实测 {lat_first:.0f}ms（dispatch 记录 {first['dispatch_ms']:.0f}ms）")
    check("失败一次后立刻判不可用（不等探测周期）", be.available() is False,
          be.last_error[:60])

    # 之后：L2 已经不在候选里，回退不该再花时间
    later = [run_trial(scene_escalate, be, fast_cfg()) for _ in range(3)]
    lat_later = [((r["t_exec"] or 0.0) - (r["t_dec"] or 0.0)) * 1000.0 for r in later]
    check("后续 3 次升级直接落 L1 且无崩溃",
          all(r["exec"] and r["level_exec"] == 1 and not r["crash"] for r in later),
          f"levels={[sorted(r['levels']) for r in later]}")
    check("后续回退不再付建连超时", max(lat_later) < 20.0, _dist(lat_later))
    print(f"      第一次掉线 {lat_first:.0f}ms；后续 {_dist(lat_later)}")
    return {"lat": [lat_first], "later": lat_later}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    t_start = time.perf_counter()
    print(BAR)
    print("test_real_backend.py -- 真实 L2（agent_bridge）验收")
    print(BAR)
    print(f"python={sys.version.split()[0]}  DEEPSEEK_API_KEY={'有' if not NO_KEY else '无'}"
          f"（{'LLM planner' if not NO_KEY else '规则 planner，动作可预测'}）")

    bridge = BridgeProcess()
    slow = BridgeProcess(script=SLOW_PY, extra_args=("--delay-ms", "1500"))
    be = None
    normal: dict = {"dispatch": [], "runs": []}
    down: dict = {"lat": [], "runs": []}
    t300: dict = {"dispatch": [], "runs": []}
    t900: dict = {"dispatch": [], "runs": []}
    killed: dict = {"lat": []}
    try:
        print(f"\n拉起真实 agent_bridge: {bridge.script} --port {bridge.port}")
        bridge.start()
        be = LocalAgentBackend(bridge.url)
        if not be.wait_healthy(15.0):
            print(f"[FAIL] agent_bridge 起不来: {be.last_error}")
            print(bridge.stop()[-800:])
            return 1
        print(f"  就绪: {bridge.url}  actions={be.actions}")

        test_translation(be)
        test_end_to_end(be)
        normal = test_normal(be)
        down = test_agent_down()

        print(f"\n拉起慢 agent（延迟注入版，同一份 server.py）: {SLOW_PY.name}")
        slow.start()
        probe = LocalAgentBackend(slow.url, name="slow-probe")
        ok = probe.wait_healthy(15.0)
        probe.close()
        if not ok:
            print(f"[FAIL] slow_bridge 起不来: {probe.last_error}")
            print(slow.stop()[-800:])
            return 1
        t300 = test_timeout(slow.url)
        t900 = test_timeout_production_ttl(slow.url)

        test_failure_modes(be)
        killed = test_killed_midflight(be, bridge)
    finally:
        if slow.proc is not None:
            slow.stop()
        if be is not None:
            be.close()
        if bridge.proc is not None:
            bridge.stop()

    # -- 延迟分布总表 -------------------------------------------------------
    print(f"\n{BAR}")
    print("对照实验：三种场景的实测延迟分布（全部本机真实测量）")
    print(BAR)
    print(f"{'场景':<26}{'n':>3}  {'p50':>9}{'p95':>9}{'max':>9}   行为")
    rows = [
        ("正常：L2 可用", normal["dispatch"], "升级成功，detail 经独立核对"),
        ("agent 未启动", down["lat"], "同拍落 L1；L2 零调用、零失败"),
        ("agent 超时 ttl=300", t300["dispatch"], "270ms socket 超时 -> 回退 L1"),
        ("agent 超时 ttl=900", t900["dispatch"], "生产默认预算内回退 L1"),
        ("运行期被杀", killed["lat"], "首次 L2 建连超时 -> 回退 L1；之后直接 L1"),
    ]
    for name, xs, note in rows:
        if not xs:
            continue
        print(f"{name:<26}{len(xs):>3}  {percentile(xs, 0.5):>8.1f}ms"
              f"{percentile(xs, 0.95):>8.1f}ms{max(xs):>8.1f}ms   {note}")
    print(BAR)

    dt = time.perf_counter() - t_start
    total = PASS + FAIL
    print(f"\n结果: {PASS}/{total} PASS, {FAIL} FAIL   用时 {dt:.2f}s")
    if FAILED:
        print("失败项:")
        for n in FAILED:
            print(f"  - {n}")
    print("*** PASS ***" if FAIL == 0 else "*** FAIL ***")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
