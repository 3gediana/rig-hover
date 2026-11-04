#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""demo_real.py -- 真实 L2：机器人升级 -> PC 上真的执行了一个动作 -> 结果回到上层。

和 ``demo.py`` 的区别只有一处，但它是关键的一处：**L2 不是 mock**。
``LocalAgentBackend`` 通过 HTTP 调 ``../agent_bridge/server.py``（自动拉起子进程），
agent 真的在主机上执行了 ``clock`` / ``disk`` / ``note`` / ``system_status``，
返回的 detail 是真时间、真磁盘剩余、真主机名。demo 用 ``detail_check`` 独立核对
每一条，核对不通过就判 FAIL。

跑通的链路（每一段都有实测耗时）::

    合成观测序列（50Hz 实时）
      -> DecisionHub 判定要升级：打印 p_esc、归因、P_fix、U(L)
      -> LocalAgentBackend 把升级翻译成 intent，POST /agent/task
      -> agent_bridge 用规则 planner（无 key）挑动作并真实执行
      -> 真实 detail 回到 hub，经 parse_instruction 校验
      -> 变成机器人可执行的动作程序，本机 50Hz 播放后交还 L0

用法::

    python demo_real.py                          # 自动拉起 agent_bridge（随机空闲端口）
    python demo_real.py --port 8899              # 指定端口
    python demo_real.py --no-spawn --url http://127.0.0.1:8899   # 用已经在跑的服务
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:  # 控制台编码不是 UTF-8 时也不要因为一个汉字炸掉
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

from bridge_proc import BridgeProcess, SERVER_PY, free_port
from detail_check import check_detail, notes_path
from hub import CfcOut, DecisionHub, HubConfig, HubObs, LEVEL_NAMES, percentile
from local_agent import LocalAgentBackend

BAR = "=" * 78
SUB = "-" * 70

DEFAULT_PORT = 8899


# --------------------------------------------------------------------------
# 观测序列：唯一被"造"出来的东西
# --------------------------------------------------------------------------


def _base(t: float) -> HubObs:
    """看得见、置信度高的人。三个场景都从这一帧开始，再叠加各自的异常。"""
    return HubObs(
        t_now=t,
        cfc=CfcOut(head_des=3.0, vyaw_des=0.0, vx_des=0.0, mode=1, emote=0,
                   action_conf=0.94),
        person_conf=0.94, feat_age_ms=20.0, dist_est=1.18,
    )


def scene_point(t: float) -> HubObs:
    """人指着桌上的杯子：本地没有物体识别/指代消解 -> L2（风险族 language）。"""
    o = _base(t)
    o.theta_img = 9.0
    o.cfc.head_des = 9.0
    if t > 0.60:
        o.pointing = 1.0
        o.point_dir_deg = 32.0
        o.cfc.action_conf = 0.55
    return o


def scene_ood(t: float) -> HubObs:
    """场景滑出训练分布（马氏距离 6σ）：本地不知道该怎么办 -> L2（风险族 unknown）。"""
    o = _base(t)
    if t > 0.60:
        o.novelty = 6.0
        o.cfc.action_conf = 0.60
    return o


def scene_approach(t: float) -> HubObs:
    """人很近（0.55m）而且机器人没把握还要往前：社会性不可逆 -> L2（风险族 stakes）。"""
    o = _base(t)
    if t > 0.60:
        o.dist_est = 0.55
        o.cfc.vx_des = 0.24
        o.cfc.action_conf = 0.30
    return o


# --------------------------------------------------------------------------
# 跑一个场景
# --------------------------------------------------------------------------


def _prog_text(prog: list[dict]) -> str:
    parts = []
    for s in prog:
        arg = ""
        for k in ("deg", "dps", "mps", "id"):
            if k in s:
                arg = f" {s[k]}"
                break
        parts.append(f"{s.get('op')}{arg}/{s.get('ms', 0):.0f}ms")
    return " → ".join(parts)


def run_scene(tag: str, title: str, scene, duration: float, be: LocalAgentBackend,
              cfg: HubConfig | None = None, pace: float = 0.02,
              expect_l2: bool = True) -> dict:
    """在一个新 hub 上跑一个场景，打印判据、耗时、真实返回，最后核对 detail。"""
    hub = DecisionHub(cfg or HubConfig(), backends={2: be} if be is not None else {})
    print(f"\n[{tag}] {title}")
    t0 = time.perf_counter()
    rec: dict = {
        "tag": tag, "t_dec": None, "t_wait": None, "t_exec": None, "t_done": None,
        "decision": None, "instruction": None, "raw": "", "detail": "", "action": "",
        "waiting_ticks": 0, "bad_wait_actions": 0, "levels": {}, "evidence": "",
        "ok": False, "crash": "",
    }
    notes_before = _note_lines()
    deadline = t0 + duration
    while time.perf_counter() < deadline:
        t = time.perf_counter() - t0
        try:
            out = hub.tick(scene(t), 0.02)
        except Exception as exc:  # noqa: BLE001 - demo 要把它打出来，而不是崩掉
            rec["crash"] = f"{type(exc).__name__}: {exc}"
            print(f"  {t:5.2f}s  !! tick 抛异常 {rec['crash']}")
            break

        rec["levels"][out.level] = rec["levels"].get(out.level, 0) + 1
        d, p = out.decision, out.payload

        # -- 判据首次超线（真正的升级判据在派发那一拍打印） -------------------
        if rec["t_dec"] is None and d is not None and d.level > 0:
            rec["t_dec"] = t
            print(f"  {t:5.2f}s  判据首次超线 p_esc={d.p_esc:.3f} -> L{d.level}"
                  f"({LEVEL_NAMES[d.level]})  风险族={d.top_family or '-'}"
                  f"  （要连续在上面停 {hub.cfg.dwell_ms}ms 才派发）")

        # -- 派发：这一拍的 decision 才是真正做路由的那一个 -------------------
        if p.get("status") == "waiting":
            rec["waiting_ticks"] += 1
            a = out.action
            if a.vx_des != 0.0 or a.vyaw_des != 0.0:
                rec["bad_wait_actions"] += 1
            if rec["t_wait"] is None:
                rec["t_wait"] = t
                dec = out.decision or d
                rec["decision"] = dec
                pfix = " ".join(f"L{k}:{v:.2f}" for k, v in dec.pfix.items())
                util = " ".join(f"L{k}:{v:+.2f}" for k, v in dec.utility.items() if k)
                print(f"  {t:5.2f}s  派发 L{p.get('level')} -> {p.get('backend')} "
                      f"ttl={p.get('ttl_ms')}ms  p_esc={p.get('p_esc')}")
                print(f"         归因 {dec.reason_text()}")
                print(f"         P_fix={{{pfix}}}  U(L)={{{util}}}  可选层级={dec.allowed}")
                print("         等待期每拍仍出指令：车轮清零、头继续跟人")

        # -- 真实回包落地（L2） / 本地技能当拍开始（L1） ----------------------
        if p.get("status") == "executing" and rec["t_exec"] is None:
            rec["t_exec"] = t
            rec["instruction"] = p.get("instruction") or {}
            if out.level == 2 and p.get("source") == "local-agent":
                rec["raw"] = be.last_raw
                res = be.last_result if isinstance(be.last_result, dict) else {}
                rec["detail"] = str(res.get("detail", ""))
                rec["action"] = str(res.get("action", ""))
                rt = be.roundtrip_ms[-1] if be.roundtrip_ms else 0.0
                print(f"  {t:5.2f}s  agent 回包：HTTP 往返 {rt:.1f}ms，agent 内部 "
                      f"{res.get('elapsed_ms')}ms，planner={res.get('planner')}，"
                      f"action={rec['action']}")
                print(f"         下行意图 {be.last_task.get('intent', '')!r}")
                print(f"         原始回包 {rec['raw']}")
                print(f"         结构化指令 {json.dumps(rec['instruction'], ensure_ascii=False)}")
            else:
                print(f"  {t:5.2f}s  L{out.level} 本地技能当拍开始（零网络）："
                      f"{p.get('intent')}")
                print(f"         结构化指令 {json.dumps(rec['instruction'], ensure_ascii=False)}")
            print(f"         执行 {_prog_text(rec['instruction'].get('prog', []))}  "
                  f"say={rec['instruction'].get('say')!r}")

        # -- 程序播放完，交还 L0 --------------------------------------------
        if p.get("status") == "exec_done" and rec["t_exec"] is not None:
            rec["t_done"] = t
            print(f"  {t:5.2f}s  交还 L0（本地播放 {p.get('skill_ms', 0):.0f}ms，不占网络）")
            break

        time.sleep(pace)

    # -- 账本 ---------------------------------------------------------------
    lat = {
        "dec_to_wait": _ms(rec["t_dec"], rec["t_wait"]),
        "wait_to_exec": _ms(rec["t_wait"], rec["t_exec"]),
        "dec_to_exec": _ms(rec["t_dec"], rec["t_exec"]),
        "dec_to_done": _ms(rec["t_dec"], rec["t_done"]),
        "http": be.roundtrip_ms[-1] if (expect_l2 and be.roundtrip_ms) else 0.0,
        "agent": float((be.last_result or {}).get("elapsed_ms") or 0.0) if expect_l2 else 0.0,
        "hub_dispatch": hub.stats.dispatch_ms[2][-1] if hub.stats.dispatch_ms[2] else 0.0,
        "l0_p50": percentile(hub.stats.local_us, 0.5),
    }
    rec["lat"] = lat
    rec["stats"] = hub.stats.summary()

    notes_after = _note_lines()
    if expect_l2:
        ok, ev = (check_detail(rec["action"], rec["detail"]) if rec["action"]
                  else (False, "没有执行"))
        if rec["action"] == "note":
            ev += f"；robot_notes.log {notes_before} -> {notes_after} 行"
            ok = ok and notes_after == notes_before + 1
    else:  # 反证场景：不核对 detail，核对"真的退到了 L1 且一次都没碰网络"
        ok = (rec["levels"].get(1, 0) > 0 and rec["levels"].get(2, 0) == 0
              and be.calls == 0 and not rec["crash"])
        ev = (f"L1 执行 {rec['levels'].get(1, 0)} 拍，L2 调用 {be.calls} 次，"
              f"tick 异常 {rec['crash'] or '无'}")
    rec["ok"], rec["evidence"] = ok, ev

    print(f"  {SUB}")
    print(f"  账本  判据超线->指令 {lat['dec_to_exec']:.1f}ms"
          f"（判据->派发 {lat['dec_to_wait']:.1f} + 派发->指令 {lat['wait_to_exec']:.1f}）"
          f"  HTTP 往返 {lat['http']:.1f}ms  agent 内部 {lat['agent']:.1f}ms"
          f"  hub 观测 {lat['hub_dispatch']:.1f}ms  L0 决策 p50 {lat['l0_p50']:.0f}µs")
    print(f"  等待期 {rec['waiting_ticks']} 拍（车轮清零违规 {rec['bad_wait_actions']} 拍）"
          f"  层级分布 {rec['levels']}")
    print(f"  核对 [{'PASS' if rec['ok'] else 'FAIL'}] "
          + ("真实执行 " if expect_l2 else "回退 ") + rec["evidence"])
    hub.close()
    return rec


def _ms(t1: float | None, t2: float | None) -> float:
    if t1 is None or t2 is None:
        return 0.0
    return (t2 - t1) * 1000.0


def _note_lines() -> int:
    """robot_notes.log 当前行数（不存在算 0）。"""
    try:
        return len([ln for ln in notes_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
    except OSError:
        return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _usage_hint(port: int) -> str:
    return (
        f"    手动启动:  \"{sys.executable}\" \"{SERVER_PY}\" --port {port}\n"
        f"    然后运行:  python demo_real.py --no-spawn --url http://127.0.0.1:{port}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RIG-Hover 真实 L2（agent_bridge）演示")
    ap.add_argument("--port", type=int, default=0, help="agent_bridge 端口；0 = 随机空闲端口")
    ap.add_argument("--url", default="", help="已有 agent_bridge 的地址（配 --no-spawn）")
    ap.add_argument("--no-spawn", action="store_true", help="不自动拉起子进程")
    args = ap.parse_args(argv)

    print(BAR)
    print("RIG-Hover · 真实 L2 演示：升级 -> PC 上真的执行 -> 结果回到上层")
    print("L0 反射(本地) | L1 技能(本地) | L2 PC agent(真实 HTTP) | L3 云端(无 key，不可用)")
    print(BAR)
    print(f"python={sys.version.split()[0]}  cwd={Path.cwd()}")

    bridge: BridgeProcess | None = None
    url = args.url.strip()
    if args.no_spawn:
        port = int(url.rsplit(":", 1)[-1].split("/")[0]) if ":" in url else DEFAULT_PORT
        url = url or f"http://127.0.0.1:{DEFAULT_PORT}"
        print(f"使用已有 agent_bridge: {url}")
    else:
        bridge = BridgeProcess(port=args.port or None)
        try:
            bridge.start()
        except (OSError, RuntimeError) as exc:
            print(f"\n[!] 拉不起 agent_bridge：{exc}\n{_usage_hint(args.port or DEFAULT_PORT)}")
            return 1
        port = bridge.port
        url = bridge.url
        print(f"已拉起 agent_bridge 子进程：{SERVER_PY}")
        print(f"  --port {port}  ->  {url}")

    be = LocalAgentBackend(url)
    try:
        # 自己拉子进程要等它启动（实测 1.5s 上下）；用已有服务就不该干等 10s。
        if not be.wait_healthy(3.0 if args.no_spawn else 10.0):
            print(f"\n[!] agent_bridge 没就绪: {be.last_error}")
            if bridge is not None:
                logs = bridge.stop()
                if logs.strip():
                    print("  子进程输出：\n" + "\n".join("    " + ln for ln in logs.strip().splitlines()[-8:]))
            else:
                print(_usage_hint(port))
            return 1
        print(f"  健康检查 ok，动作白名单 {be.actions}，LLM planner={'有 key' if be.llm else '无 key（用规则 planner）'}")

        scenes = [
            ("1/4", "指物：本地没物体识别 -> L2 -> 让 PC 记一笔（真写 robot_notes.log）",
             scene_point, 2.4, "language"),
            ("2/4", "场景出域 6σ：本地不知道怎么办 -> L2 -> 问 PC 系统状态（真主机名/磁盘/时间）",
             scene_ood, 2.4, "unknown"),
            ("3/4", "太近且没把握还要往前：不可逆 -> L2 -> 问 PC 磁盘剩余（真剩余空间）",
             scene_approach, 2.4, "stakes"),
        ]
        runs = []
        for tag, title, scene, dur, _fam in scenes:
            runs.append(run_scene(tag, title, scene, dur, be))

        # 反证：L2 不在时不能崩，只是退到 L1
        dead = LocalAgentBackend(f"http://127.0.0.1:{free_port()}", name="dead-agent",
                                 probe_on_start=True, start_health_thread=False)
        print(f"\n[反证] agent 没起来（{dead.base_url} 无人监听）")
        print(f"  available()={dead.available()}  预期：hub 根本不把 L2 放进候选，直接 L1")
        ref = run_scene("4/4", "同一个指物场景，L2 不可用 -> L1 本地技能，机器人不僵住",
                        scene_point, 1.6, dead, expect_l2=False)
        dead.close()

        # -- 汇总 -----------------------------------------------------------
        print(f"\n{BAR}")
        print("实测汇总（全部是本机真实测量；mock 的 99.9ms 不在这张表里）")
        print(BAR)
        print(f"{'场景':<5}{'L2 动作':<14}{'agent内部':>10}{'HTTP往返':>10}{'hub观测':>9}"
              f"{'判据->指令':>11}  核对")
        for r in runs:
            lat = r["lat"]
            print(f"{r['tag']:<5}{r['action'] or '-':<14}{lat['agent']:>9.1f}ms"
                  f"{lat['http']:>9.1f}ms{lat['hub_dispatch']:>8.1f}ms"
                  f"{lat['dec_to_exec']:>10.1f}ms  {'PASS' if r['ok'] else 'FAIL'}")
        print(BAR)
        print("层级账本")
        l0_p50 = [r["lat"]["l0_p50"] for r in runs]
        print(f"  L0 决策链      p50={percentile(l0_p50, 0.5):.0f}µs p95={percentile(l0_p50, 0.95):.0f}µs"
              f"（50Hz 实时循环内，纯 Python）")
        print("  L1 本地技能    4/4 兜底：本地程序播放 %.0fms，零网络；成功的三个场景不需要它"
              % ref["stats"]["levels"]["1"]["skill_p50_ms"])
        print("  L2 真实 HTTP   " + "  ".join(
            f"{r['action']}:{r['lat']['http']:.1f}ms" for r in runs)
            + "（127.0.0.1，含 agent 真实执行时间）")
        print("  L3 云端        不可用：没有 DEEPSEEK_API_KEY（L3 后端 available()=False，"
              "路由器自动不选它）")
        print(BAR)
        print("注：hub 观测 = 50Hz tick 采样粒度（20ms），所以它 >= HTTP 往返；"
              "真实 HTTP 延迟看『HTTP往返』那列。")

        all_ok = (all(r["ok"] and r["action"] for r in runs)
                  and not any(r["crash"] for r in runs)
                  and ref["levels"].get(2, 0) == 0
                  and ref["levels"].get(1, 0) > 0
                  and dead.calls == 0)
        print("结论：升级链路产生了真实副作用，detail 经独立核对："
              + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1
    finally:
        be.close()
        if bridge is not None:
            logs = bridge.stop()
            if logs.strip():
                print(f"\n[agent_bridge] 子进程输出：{logs.strip().splitlines()[-1]}")
            print("[agent_bridge] 已停止")


if __name__ == "__main__":
    raise SystemExit(main())
