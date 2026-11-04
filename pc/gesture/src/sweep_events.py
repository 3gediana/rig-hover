# -*- coding: utf-8 -*-
"""sweep_events.py — 画出「触发延迟 ↔ 误报率」的取舍曲线

`calibrate_events.py` 只输出一个最优解，但**选哪一点是产品决策**：1 s 的反应
延迟对人机交互太慢，而 3 次/分钟的未知手势误报也不可接受。这里把曲线打出来，
让人看着数字选，而不是让代价函数替人决定。

用法::

    python src/sweep_events.py                 # 默认网格
    python src/sweep_events.py --set 0.65 0.15 0.15   # 选定后写回 YAML
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from calibrate_events import (CFG_PATH, _variant, build_suite,  # noqa: E402
                              load_bank, score)
from gesture_classifier import MODEL_PATH, GestureClassifier  # noqa: E402

DT = 1.0 / 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enter", type=float, nargs="+",
                    default=[0.55, 0.60, 0.65, 0.70, 0.75])
    ap.add_argument("--pending", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--release", type=float, nargs="+", default=[0.15])
    ap.add_argument("--set", type=float, nargs=3, default=None,
                    metavar=("ENTER", "PENDING", "RELEASE"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        cur = yaml.safe_load(fh)
    bank = load_bank("val")
    episodes = build_suite(bank)
    clf = GestureClassifier.load(MODEL_PATH, device=args.device)
    suite = []
    for ep in episodes:
        clf.reset()
        suite.append((ep, clf.run_episode(ep.feat, ep.crop, dt=DT)))
    print("[sweep] %d 条序列（含 %.1f 分钟静止、%.1f 分钟未见手势）"
          % (len(episodes),
             sum(len(ep.feat) for ep in episodes if ep.name.startswith("idle"))
             * DT / 60.0,
             sum(len(ep.feat) for ep in episodes
                 if ep.name.startswith("unseen")) * DT / 60.0), flush=True)

    if args.set:
        cfg = _variant(cur, *args.set)
        sc = score(cfg, suite)
        _write(cur, args.set)
        print("[sweep] 已写回 YAML: enter %.2f pending %.2f release %.2f"
              % tuple(args.set), flush=True)
        print("[sweep] 读回校验 延迟 %.0f ms  静止误报 %.2f/min  "
              "未见手势 %.2f/min" % (sc["wave"]["latency_mean_ms"],
                                  sc["fa_idle_per_min"],
                                  sc["fa_unseen_per_min"]), flush=True)
        return

    rows = []
    for enter in args.enter:
        for pending in args.pending:
            for release in args.release:
                sc = score(_variant(cur, enter, pending, release), suite)
                rows.append({
                    "enter": enter, "pending": pending, "release": release,
                    "latency_ms": sc["wave"]["latency_mean_ms"],
                    "miss": sc["wave"]["miss_rate"],
                    "cont": sc["cont_single_onset_rate"],
                    "static": sc["static_one_onset_rate"],
                    "split": sc["two_wave_split_rate"],
                    "fa_idle": sc["fa_idle_per_min"],
                    "fa_unseen": sc["fa_unseen_per_min"],
                })
                print("  enter %.2f pending %.2f  → 延迟 %5.0f ms  漏检 %.2f  "
                      "单onset %.2f  静态 %.2f  拆分 %.2f  "
                      "静止误报 %.2f/min  未见 %.2f/min"
                      % (enter, pending, rows[-1]["latency_ms"] or -1,
                         rows[-1]["miss"], rows[-1]["cont"],
                         rows[-1]["static"], rows[-1]["split"],
                         rows[-1]["fa_idle"], rows[-1]["fa_unseen"]),
                      flush=True)
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "sweep.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print("[sweep] → reports/sweep.json", flush=True)


def _write(cur, triple):
    enter, pending, release = triple
    out = dict(cur)
    out["enter_thresh"] = float(enter)
    out["exit_thresh"] = round(float(enter) - 0.23, 3)
    out["timing_s"] = dict(cur["timing_s"])
    out["timing_s"]["pending_hold"] = float(pending)
    out["timing_s"]["release"] = float(release)
    per = {}
    for k in (cur.get("per_class") or {}):
        per[k] = {"enter": float(enter), "exit": round(float(enter) - 0.23, 3)}
    if per:
        out["per_class"] = per
    with open(CFG_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
