#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""real_image_gap.py -- run the deployed perception chain on real HaGRID photos.

All gesture numbers reported so far come from *ground-truth* skeletons inside
the gesture module's own split. Deployment uses YOLO26n-pose keypoints, which
are noisier. This script closes that gap: it feeds real photographs through the
exact chain the robot uses (YOLO26n-pose -> skeleton -> classifier -> event
machine) and reports the resulting agreement with the HaGRID label.

Run:
    python real_image_gap.py --n-per-class 40
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GESTURE = r"D:\robot_work\pc\gesture"
PERCEPTION = r"D:\robot_work\pc\perception"
MANIFEST = os.path.join(GESTURE, "data", "manifest.csv")

sys.path.insert(0, PERCEPTION)
sys.path.insert(0, GESTURE)

# HaGRID label -> the class id the deployed pipeline should report.
# `wave` has no HaGRID counterpart (it is a motion, synthesised from palm poses),
# so it is excluded from the agreement table rather than counted as wrong.
HAGRID_TO_ID = {
    "no_gesture": 0,   # idle
    "palm": 2,         # open_palm
    "like": 3,         # thumbs_up
    "point": 4,        # pointing
    "fist": 5,         # fist
}
ID_TO_NAME = {0: "idle", 1: "wave", 2: "open_palm", 3: "thumbs_up",
              4: "pointing", 5: "fist"}


def load_rows():
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["hagrid_class"] in HAGRID_TO_ID:
                rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--tile", type=int, default=12,
                    help="repeat each frame to give the event machine a window")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_rows()
    print(f"manifest: {len(rows)} usable rows across "
          f"{len(set(r['hagrid_class'] for r in rows))} classes")

    rng = random.Random(a.seed)
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["hagrid_class"], []).append(r)
    sample = []
    for cls, rs in sorted(by_class.items()):
        rng.shuffle(rs)
        sample += [(cls, r) for r in rs[:a.n_per_class]]
    print(f"sampled {len(sample)} images\n")

    from run_perception import PerceptionPipeline  # noqa: E402

    pipe = PerceptionPipeline()
    print("perception pipeline ready\n")

    stats = {c: {"n": 0, "seen": 0, "hand": 0, "correct": 0, "correct_raw": 0,
                 "conf": [], "person_h": [], "pred": {}} for c in by_class}
    for i, (cls, r) in enumerate(sample):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        # The event machine needs a window; repeating the same frame is the
        # "holding the gesture still" case, which is what deployment sees for a
        # static gesture anyway.
        out = None
        for _ in range(a.tile):
            out = pipe(img)
        s = stats[cls]
        s["n"] += 1
        s["seen"] += int(out.seen)
        s["hand"] += int(bool(getattr(out, "hand_up_L", 0) or
                              getattr(out, "hand_up_R", 0)))
        s["conf"].append(float(out.person_conf))
        s["person_h"].append(float(out.person_h))
        pred = int(getattr(out, "gesture_id", 0))
        s["pred"][pred] = s["pred"].get(pred, 0) + 1
        want = HAGRID_TO_ID[cls]
        s["correct"] += int(pred == want)
        # Per-frame classifier argmax, before the event machine's confirmation
        # logic. Comparing the two accuracies separates "the classifier is
        # wrong" from "the event machine failed to confirm a correct frame".
        s["correct_raw"] += int(int(getattr(out, "gesture_raw_id", 0)) == want)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(sample)}")

    print("\n" + "=" * 96)
    print(f"{'HaGRID class':<12}{'n':>4}{'seen%':>7}{'hand%':>7}"
          f"{'mean conf':>11}{'per-frame':>11}{'end-to-end':>12}   predictions")
    print("-" * 96)
    tot_n = tot_ok = tot_raw = 0
    for cls in sorted(by_class):
        s = stats[cls]
        if not s["n"]:
            continue
        n = s["n"]
        preds = " ".join(f"{ID_TO_NAME.get(k, k)}={v}"
                         for k, v in sorted(s["pred"].items(), key=lambda x: -x[1]))
        print(f"{cls:<12}{n:>4}{100*s['seen']/n:>6.0f}%{100*s['hand']/n:>6.0f}%"
              f"{np.mean(s['conf']):>11.3f}{100*s['correct_raw']/n:>10.0f}%"
              f"{100*s['correct']/n:>11.0f}%   {preds}")
        tot_n += n
        tot_ok += s["correct"]
        tot_raw += s["correct_raw"]
    print("=" * 96)
    print(f"per-frame classifier top-1 (predicted keypoints): "
          f"{tot_raw}/{tot_n} = {100*tot_raw/max(tot_n,1):.1f}%")
    print(f"end-to-end through the event machine          : "
          f"{tot_ok}/{tot_n} = {100*tot_ok/max(tot_n,1):.1f}%")
    print(f"=> the event machine costs "
          f"{(tot_raw - tot_ok)/max(tot_n,1)*100:.1f} points of top-1 accuracy")

    ok = tot_ok / max(tot_n, 1)
    print(f"\n{'PASS' if ok > 0.30 else 'FAIL'} (gate: >30% top-1 on real photos)")
    return 0 if ok > 0.30 else 1


if __name__ == "__main__":
    sys.exit(main())
