#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all_tests.py -- one command to verify every claim in the report.

Each suite is run in its own process with the project interpreter; a suite
counts as passing when it exits 0 AND prints no FAIL marker. Output is
summarised rather than streamed, so a long run stays readable.

Usage:
    python run_all_tests.py              # everything
    python run_all_tests.py --quick      # skip the slow simulation sweep
    python run_all_tests.py --only sim   # one suite
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

PY = r"D:\smartcar\.venv\Scripts\python.exe"
W = r"D:\robot_work"

# (id, display name, cwd, argv, slow)
SUITES = [
    ("cfc_host",  "CfC 3-way numeric parity (ncps/numpy/C)", rf"{W}\pc\cfc_host",
     ["test_parity.py"], False),
    ("train",     "train + blob export + firmware C end-to-end", rf"{W}\pc\train",
     ["test_e2e.py"], False),
    # 场景 v2 的观测组装：106 -> 116 维，前 106 维必须与 v1 逐元素一致；
    # v2a 锁存目标 / v2b 恒 0 是"CfC 是否承担记忆"的对照开关。
    ("obsv2",     "observation v2 assembly (v2a/v2b)", rf"{W}\pc\train",
     ["obs_v2.py"], False),
    ("precision", "weight precision study (bf16/fp16/int8/mixed)", rf"{W}\pc\precision",
     ["study.py"], False),
    ("perception","perception chain (YOLO features + gesture prims)", rf"{W}\pc\perception",
     ["test_perception.py"], False),
    # 场景 v2 的信号层：人脸方位/尺度、拳头方位、腕部归一化高度、敲两下的时序判定。
    # 纯 numpy，不加载 torch，几秒跑完。
    ("scenesig",  "scene v2 signals (face/fist/knock)", rf"{W}\pc\perception",
     ["test_scene_signals.py"], False),
    # 场景 v2 的决策层：1–2 米门、头优先/轮子兜底、敲击锁定目标、手离开后仍走过去、
    # 到达后停留再回看人脸。纯 numpy，秒级。
    ("scenariov2","scene v2 state machine (gaze/lock/approach)", rf"{W}\pc\perception",
     ["test_scenario_v2.py"], False),
    # 取流结构：单帧轮询 vs 持久 MJPEG。用本地 mock 服务端验正确性（帧边界、
    # 长连接不漏帧、上层能直接吃）。**不验延迟收益**——那要真机。
    ("framesrc",  "frame sources (HTTP poll vs MJPEG stream)", rf"{W}\pc\perception",
     ["test_frame_source.py"], False),
    # 多人时选谁：用户明确"谁靠着近就看谁"。按框高排（不受画面边缘裁剪影响），
    # 不是按面积。桩对象喂 _pick_person，不需要 GPU。
    ("pickperson","nearest-person selection", rf"{W}\pc\perception",
     ["test_pick_person.py"], False),
    # 桌沿检测：地面单应（俯仰角 + 相机高度 + 焦距）把图像行换算成"离桌沿多少米"。
    # 自平衡车前进时整车前倾，摄像头自然低头看路——这是唯一能看见桌沿的时刻。
    ("tableedge", "table-edge detection during approach", rf"{W}\pc\perception",
     ["table_edge.py"], False),
    ("gesture",   "gesture classifier + event segmentation", rf"{W}\pc\gesture",
     ["test_gesture.py"], False),
    ("teacher",   "teacher controller (onset driven)", rf"{W}\pc\teacher",
     ["test_teacher.py"], False),
    ("hub",       "hierarchical decision hub", rf"{W}\pc\decision_hub",
     ["test_hub.py"], False),
    ("escalate",  "CfC escalation head (temporal evidence)", rf"{W}\pc\escalate",
     ["test_escalate.py"], False),
    ("bridge",    "local agent bridge (real side effects)", rf"{W}\pc\agent_bridge",
     ["test_bridge.py"], False),
    ("sim",       "closed-loop sim, 4 controllers", rf"{W}\pc\sim",
     ["eval.py", "--seeds", "3"], True),
    ("realimg",   "end-to-end gesture on real photos (domain gap)", rf"{W}\pc\bench",
     ["real_image_gap.py", "--n-per-class", "20"], True),
    # Runs the firmware's own C engine over real perception and compares it to
    # the teacher. Gates on "chain runs, outputs finite and in range" -- action
    # quality is a separate bar this policy does not yet meet.
    ("e2ereplay", "offline replay: firmware engine vs teacher", rf"{W}\pc\bench",
     ["e2e_replay.py", "--n", "40"], True),
    # The real quality bar: held-out sessions never seen in training, scored
    # against the always-zero baseline so "did it learn anything" is answered.
    ("evalsession", "held-out session eval (real-feature policy)", rf"{W}\pc\bench",
     ["eval_session.py",
      "--blob", rf"{W}\pc\train\out_nofeat\m8_inert.blob",
      "--sessions", rf"{W}\pc\train\data_pan_holdout2"], True),
    # The only suite that scores the policy on frames from the robot's own
    # camera against that recording's teacher labels. The candidate is the
    # no-feature model; the baseline is the synthetic-feature model it replaced.
    ("deploygate", "deployability on recorded robot frames", rf"{W}\pc\bench",
     ["deploy_gate.py", "--blob", rf"{W}\pc\train\out_nofeat\m8_inert.blob"], True),
    # Behavioural check, not an error metric: does the learned policy keep the
    # three-tier structure (head share must fall as the bearing error grows)?
    ("threetier", "three-tier allocation behaviour", rf"{W}\pc\bench",
     ["three_tier_check.py",
      "--blob", rf"{W}\pc\train\out_nofeat\m8_inert.blob",
      "--sessions", rf"{W}\pc\train\data_pan_holdout2"], True),
    # The firmware's safety shell compiled for the host and driven through each
    # of its seven defences. Until this existed the code that stops the robot
    # hurting itself had only ever run on hardware.
    ("safety", "firmware safety shell (7 defences)", rf"{W}\pc\train\arbtest",
     ["build_and_run.py"], True),
]

FAIL_LINE_RE = re.compile(r"(?m)^\s*\[?(?:FAIL|FAILED)\b")
# Suites print their tally in two different orders:
#   "87 PASS / 0 FAIL"   and   "PASS 16 / FAIL 0"
# A single regex cannot tell which number belongs to FAIL, so check both
# adjacencies: digits *immediately* before FAIL, and digits after FAIL.
# Requiring "immediately" is what stops "PASS 16 / FAIL 0" from being read as
# "16 failures".
N_FAIL_BEFORE = re.compile(r"(\d+)\s*FAIL\b")
N_FAIL_AFTER = re.compile(r"FAIL\s*[:=]?\s*(\d+)")

# Diagnostics answer a question but do not gate: they are reported, not counted.
# obsdomain is here because its reference distribution is wrong. It feeds raw
# 512x512 HaGRID hand close-ups into the perception chain and measures how far a
# policy's input envelope sits from that -- but the robot camera produces 240x240
# room views, not hand close-ups. Passing --framing robot to match the robot made
# the ratio worse, which is the tell that the reference, not the policy, was the
# thing being measured. deploygate replaces it: same question, recorded robot
# frames, and a behavioural gate (teacher-ACK recall) instead of a statistic.
DIAGNOSTICS = [
    ("obsdomain", "policy input stats vs HaGRID photos (wrong domain)", rf"{W}\pc\bench",
     ["obs_domain_check.py", "--n", "40"], True),
]


def has_failure(out: str) -> bool:
    """A suite fails when a line starts with FAIL, or a tally reports a
    non-zero failure count in either print order."""
    if FAIL_LINE_RE.search(out):
        return True
    for rx in (N_FAIL_BEFORE, N_FAIL_AFTER):
        for m in rx.finditer(out):
            if int(m.group(1)) > 0:
                return True
    return False


def run_one(suite, timeout=900):
    sid, name, cwd, argv, _slow = suite
    if not os.path.isdir(cwd):
        return sid, name, False, 0.0, "cwd missing", ""
    t0 = time.perf_counter()
    try:
        p = subprocess.run([PY] + argv, cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return sid, name, False, time.perf_counter() - t0, f"超时 >{timeout}s", ""
    dt = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join([ln for ln in out.strip().splitlines() if ln.strip()][-6:])

    has_fail = has_failure(out)
    ok = (p.returncode == 0) and not has_fail
    note = f"exit={p.returncode}" + (", output has FAIL" if has_fail else "")
    return sid, name, ok, dt, note, tail


def run_build(timeout=900):
    """Compile the firmware. Several suites edit C sources under rig_omni; a
    broken build would otherwise only surface at flash time, which is the
    expensive moment to discover it."""
    # IDF_TOOLS_PATH must be in the child's environment: embedding `set` inside
    # the cmd /c quoting silently loses it, and export.bat then leaves idf.py
    # off PATH while redirecting its own error to nul.
    cmd = ('call G:\\esp\\esp-idf\\export.bat >nul 2>&1 && '
           'cd /d D:\\robot_work\\rig_omni && idf.py build')
    env = dict(os.environ, IDF_TOOLS_PATH=r"G:\esp\.espressif")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout,
                           env=env)
    except subprocess.TimeoutExpired:
        return "firmware", "firmware build (idf.py)", False, \
               time.perf_counter() - t0, f"timeout >{timeout}s", ""
    dt = time.perf_counter() - t0
    out = (p.stdout or "") + (p.stderr or "")
    ok = "Project build complete" in out
    tail = "\n".join([ln for ln in out.strip().splitlines() if ln.strip()][-5:])
    return "firmware", "firmware build (idf.py)", ok, dt, \
           ("ok" if ok else "build failed"), ("" if ok else tail)


def main():
    # The Windows console defaults to GBK; child suites emit UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip slow suites")
    ap.add_argument("--only", default=None, help="run one suite id")
    ap.add_argument("--build", action="store_true",
                    help="also compile the ESP-IDF firmware (~60 s incremental)")
    a = ap.parse_args()

    sel = [s for s in SUITES
           if (a.only is None or s[0] == a.only)
           and not (a.quick and s[4])]
    dsel = [s for s in DIAGNOSTICS
            if (a.only is None or s[0] == a.only)
            and not (a.quick and s[4])]

    if not os.path.exists(PY):
        print(f"interpreter not found: {PY}")
        return 2

    tag = f" + {len(dsel)} diagnostic(s)" if dsel else ""
    print(f"running {len(sel)} gate suite(s){tag} with {os.path.basename(PY)}\n")
    results = []
    for s in sel:
        print(f"  {s[0]:<11} {s[1]} ...", end="", flush=True)
        sid, name, ok, dt, note, tail = run_one(s)
        results.append((sid, name, ok, dt, note, tail))
        print(f" {'PASS' if ok else 'FAIL'}  ({dt:.1f}s)")

    if a.build:
        print(f"  {'firmware':<11} firmware build (idf.py) ...", end="", flush=True)
        rb = run_build()
        results.append(rb)
        print(f" {'PASS' if rb[2] else 'FAIL'}  ({rb[3]:.1f}s)")

    diag = []
    for s in dsel:
        print(f"  {s[0]:<11} {s[1]} ...", end="", flush=True)
        sid, name, ok, dt, note, tail = run_one(s)
        diag.append((sid, name, ok, dt, note, tail))
        print(f" {'PASS' if ok else 'FAIL'}  ({dt:.1f}s, diagnostic)")

    npass = sum(1 for r in results if r[2])
    print("\n" + "=" * 78)
    print(f"{'suite':<12}{'result':<8}{'time':>9}   note")
    print("-" * 78)
    for sid, name, ok, dt, note, _t in results:
        print(f"{sid:<12}{'PASS' if ok else 'FAIL':<8}{dt:>8.1f}s   {note}")
    for sid, name, ok, dt, note, _t in diag:
        print(f"{sid:<12}{'PASS' if ok else 'FAIL':<8}{dt:>8.1f}s   {note}"
              f" (diagnostic)")
    print("=" * 78)

    for sid, name, ok, dt, note, tail in results:
        if not ok and tail:
            safe = tail.encode("ascii", "replace").decode("ascii")
            print(f"\n--- {sid} failure tail ---\n{safe}")
    for sid, name, ok, dt, note, tail in diag:
        if not ok and tail:
            safe = tail.encode("ascii", "replace").decode("ascii")
            print(f"\n--- {sid} (diagnostic) tail ---\n{safe}")

    print(f"\nGATES: {npass}/{len(results)} passed")
    if diag:
        print(f"DIAGNOSTICS: {sum(1 for r in diag if r[2])}/{len(diag)} green "
              f"(not counted; obsdomain is red by design -- see its entry)")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
