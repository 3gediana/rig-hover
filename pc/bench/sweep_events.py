#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_events.py -- tune the gesture event machine against two objectives.

Motivation: on real photos the per-frame classifier reaches 75.8% but the
event machine only 72.1%; for `fist` it loses 22 points (67% -> 45%). The
machine requires the same class to hold above `enter_thresh` for
`pending_hold` seconds, so a class that is right ~2/3 of the time almost never
produces a clean run. Loosening the threshold should recover accuracy -- but it
must not break the wave event, which is what the machine was designed for.

So every candidate is scored on BOTH:
  * real-photo accuracy through the full chain
  * onset count on synthetic probability sequences (exactly one onset per
    wave burst, none when idle)

Run:  python sweep_events.py --n-per-class 25
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GESTURE = r"D:\robot_work\pc\gesture"
PERCEPTION = r"D:\robot_work\pc\perception"
MANIFEST = os.path.join(GESTURE, "data", "manifest.csv")
BASE_CFG = os.path.join(GESTURE, "configs", "gesture_events.yaml")

sys.path.insert(0, PERCEPTION)
sys.path.insert(0, GESTURE)

HAGRID_TO_ID = {"no_gesture": 0, "palm": 2, "like": 3, "point": 4, "fist": 5}


def sample_rows(n_per_class, seed=0):
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["hagrid_class"] in HAGRID_TO_ID:
                rows.append(r)
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["hagrid_class"], []).append(r)
    out = []
    for cls, rs in sorted(by.items()):
        rng.shuffle(rs)
        out += [(cls, r) for r in rs[:n_per_class]]
    return out


def wave_sanity(cfg: dict, dt: float) -> tuple[int, int, int]:
    """Feed three probability sequences through a fresh event machine at a
    given frame interval.

    Returns (onsets during a 4 s wave burst, onsets during 4 s of idle,
    onsets during 8 s of single-frame flicker).

    `dt` matters: the debounce parameter `pending_hold` is in seconds, so the
    same value means very different things at 66 Hz (evaluation throughput) and
    8 Hz (the deployed perception rate measured end to end). Every candidate
    must therefore satisfy the contract at BOTH rates -- a value that is only
    safe at the evaluation rate is not safe on the robot.
    """
    from gesture_events import GestureEventMachine  # noqa: E402

    def run(seq, dt_):
        em = GestureEventMachine(copy.deepcopy(cfg))
        n = 0
        for p in seq:
            out = em.update(np.asarray(p, np.float64), dt_)
            n += int(getattr(out, "gesture_onset", 0))
        return n

    idle_p = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    wave_p = np.array([0.05, 0.90, 0.015, 0.015, 0.015, 0.005])
    idle_seq = [idle_p] * max(1, int(4.0 / dt))
    wave_seq = ([idle_p] * max(1, int(0.5 / dt)) + [wave_p] * max(1, int(4.0 / dt))
                + [idle_p] * max(1, int(2.0 / dt)))

    # Flicker: every 10th frame one gesture class spikes to 0.85 for a single
    # frame, otherwise idle dominates. Nothing here is a real gesture.
    rng = np.random.default_rng(0)
    flick = []
    for i in range(max(1, int(8.0 / dt))):
        p = idle_p.copy()
        if i % 10 == 5:
            k = int(rng.integers(1, 6))
            p = np.array([0.05, 0.02, 0.02, 0.02, 0.02, 0.02])
            p[k] = 0.85
        flick.append(p)
    return run(wave_seq, dt), run(idle_seq, dt), run(flick, dt)


def eval_variant(name, cfg, sample, tile, device):
    from run_perception import PerceptionPipeline  # noqa: E402
    pipe = PerceptionPipeline(gesture_config=cfg, device=device)
    per_cls = {}
    for cls, r in sample:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        out = None
        for _ in range(tile):
            out = pipe(img)
        d = per_cls.setdefault(cls, [0, 0])
        d[0] += 1
        d[1] += int(int(getattr(out, "gesture_id", 0)) == HAGRID_TO_ID[cls])
    n = sum(v[0] for v in per_cls.values())
    ok = sum(v[1] for v in per_cls.values())
    c_eval = wave_sanity(cfg, 1.0 / 30.0)     # evaluation throughput
    c_dep = wave_sanity(cfg, 0.125)           # deployed ~8 Hz perception
    contract_ok = (c_eval[0] == 1 and c_dep[0] == 1
                   and c_eval[1] == 0 and c_dep[1] == 0
                   and c_eval[2] == 0 and c_dep[2] == 0)
    return {
        "name": name, "n": n, "acc": ok / max(n, 1),
        "per_class": {c: v[1] / max(v[0], 1) for c, v in per_cls.items()},
        "wave_onsets": c_dep[0], "idle_onsets": c_dep[1],
        "flick_onsets": c_dep[2], "contract_ok": contract_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--tile", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0,
                    help="sample seed; run a second seed to check that the "
                         "ranking generalises instead of fitting the sample")
    a = ap.parse_args()

    with open(BASE_CFG, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    variants = []
    for enter in (0.75, 0.65, 0.55, 0.45):
        for hold in (0.10, 0.05):
            cfg = copy.deepcopy(base)
            cfg.setdefault("per_class", {}).setdefault("fist", {})["enter"] = enter
            cfg.setdefault("timing_s", {})["pending_hold"] = hold
            variants.append((f"fist_enter={enter:.2f} hold={hold:.2f}", cfg))

    sample = sample_rows(a.n_per_class, seed=a.seed)
    print(f"evaluating {len(variants)} variants on {len(sample)} real photos "
          f"(tile={a.tile})\n")

    res = []
    for name, cfg in variants:
        r = eval_variant(name, cfg, sample, a.tile, a.device)
        res.append(r)
        print(f"{name:<30} acc {100*r['acc']:5.1f}%  "
              f"fist {100*r['per_class'].get('fist',0):4.0f}%  "
              f"wave={r['wave_onsets']} idle={r['idle_onsets']} "
              f"flicker={r['flick_onsets']}")

    print("\n" + "=" * 92)
    print(f"{'variant':<30}{'overall':>9}{'fist':>7}{'palm':>7}{'like':>7}"
          f"{'wave':>6}{'idle':>6}{'flick':>7}")
    print("-" * 92)
    best = None
    for r in res:
        ok_contract = r["contract_ok"]
        print(f"{r['name']:<30}{100*r['acc']:>8.1f}%"
              f"{100*r['per_class'].get('fist',0):>6.0f}%"
              f"{100*r['per_class'].get('palm',0):>6.0f}%"
              f"{100*r['per_class'].get('like',0):>6.0f}%"
              f"{r['wave_onsets']:>6}{r['idle_onsets']:>6}{r['flick_onsets']:>7}"
              + ("" if ok_contract else "  <-- breaks event contract"))
        # Only variants that keep the event contract are eligible.
        if ok_contract and (best is None or r["acc"] > best["acc"]):
            best = r
    print("=" * 92)
    if best is None:
        print("no variant kept the event contract -- keep the current config")
        return 1
    base_r = next(r for r in res if r["name"].startswith("fist_enter=0.75 hold=0.10"))
    print(f"\nrecommended: {best['name']}")
    print(f"  overall {100*base_r['acc']:.1f}% -> {100*best['acc']:.1f}% "
          f"({100*(best['acc']-base_r['acc']):+.1f} points)")
    print(f"  fist    {100*base_r['per_class'].get('fist',0):.0f}% -> "
          f"{100*best['per_class'].get('fist',0):.0f}%")
    print(f"  contract: wave {best['wave_onsets']}/1  idle {best['idle_onsets']}/0"
          f"  flicker {best['flick_onsets']}/0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
