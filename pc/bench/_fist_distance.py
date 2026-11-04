# -*- coding: utf-8 -*-
"""_fist_distance.py -- at what apparent hand size does fist stop being readable?

The scenario needs "person makes a fist" to be detected while the person stands
1-4 m away, not while a hand fills the frame. The deployed gesture classifier
sees a 32x32 hand crop, and the report already showed what that costs: `pointing`
scores F1 0.574 because COCO-17 has no finger keypoints and the hand is about
20 px across at 240x240. `fist` scores 0.846 today, but today's numbers come from
photos where the hand fills the frame.

This measures the degradation curve: shrink a HaGRID frame on a 240x240 canvas so
the person occupies a realistic fraction of the view, run the deployed chain, and
report top-1 per class and the pipeline's own `person_h`. That curve decides
whether "use a stronger visual representation" is a real requirement or a
solution looking for a problem.

Run:  python _fist_distance.py --n-per-class 20
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import real_image_gap as RIG  # noqa: E402

SCALES = (1.00, 0.75, 0.55, 0.40, 0.30, 0.22)
CLASSES = ("fist", "palm", "like", "no_gesture")
FRAME = 240


def fit_on_canvas(img, scale, rng):
    """Shrink the photo and place it on a 240x240 frame, like a distant subject.

    `scale` is relative to "the photo fills the frame", so 1.0 means the hand
    close-up fills the view and 0.22 means the subject occupies about a fifth of
    it -- the far end of the requested 1-4 m working range.
    """
    h, w = img.shape[:2]
    s = scale * (FRAME / max(h, w, 1))
    nw, nh = max(int(round(w * s)), 8), max(int(round(h * s)), 8)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # a mid-grey background reads as "not a person" to the detector, which is
    # what the extra border should look like
    canvas = np.full((FRAME, FRAME, 3), 110, np.uint8)
    y0, x0 = (FRAME - nh) // 2, (FRAME - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--tile", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = RIG.load_rows()
    rng = random.Random(a.seed)
    by_class: dict[str, list] = {}
    for r in rows:
        if r["hagrid_class"] in CLASSES:
            by_class.setdefault(r["hagrid_class"], []).append(r)
    missing = [c for c in CLASSES if c not in by_class]
    if missing:
        print(f"classes not in manifest: {missing}")
        return 2
    sample = []
    for cls in CLASSES:
        rs = by_class[cls][:]
        rng.shuffle(rs)
        sample += [(cls, r) for r in rs[:a.n_per_class]]
    print(f"sampled {len(sample)} images across {len(CLASSES)} classes")

    from run_perception import PerceptionPipeline  # noqa: E402
    pipe = PerceptionPipeline()
    print("perception pipeline ready\n")

    # stats[scale][class] = [n, raw_correct, event_correct, person_h list]
    stats = {s: {c: [0, 0, 0, []] for c in CLASSES} for s in SCALES}
    for cls, r in sample:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        want = RIG.HAGRID_TO_ID[cls]
        for s in SCALES:
            frame = fit_on_canvas(img, s, rng)
            out = None
            for _ in range(a.tile):
                out = pipe(frame)
            st = stats[s][cls]
            st[0] += 1
            st[1] += int(int(getattr(out, "gesture_raw_id", 0)) == want)
            st[2] += int(int(getattr(out, "gesture_id", 0)) == want)
            st[3].append(float(out.person_h))

    print("=" * 92)
    print("top-1 accuracy after the deployed chain (event machine confirmed), "
          "and the raw per-frame classifier")
    hdr = f"{'class':<12}" + "".join(f"{s:>13.2f}" for s in SCALES)
    print(hdr)
    print(f"{'scale':<12}" + "".join(f"{'':>13}" for _ in SCALES))
    print(f"{'person_h':<12}" + "".join(
        f"{np.mean([v for c in CLASSES for v in stats[s][c][3]]):>13.3f}"
        for s in SCALES))
    print("-" * len(hdr))
    for cls in CLASSES:
        row = f"{cls:<12}"
        for s in SCALES:
            n, raw, ev, _ph = stats[s][cls]
            row += f"{100*(raw/max(n,1)):>7.1f}%/{100*(ev/max(n,1)):>4.0f}%"
        print(row)
    print("-" * len(hdr))
    row = f"{'MEAN raw':<12}"
    for s in SCALES:
        num = sum(stats[s][c][1] for c in CLASSES)
        den = sum(stats[s][c][0] for c in CLASSES)
        row += f"{100*num/max(den,1):>12.1f}%"
    print(row)
    print("=" * 92)
    print("Read: the last column is roughly the hand size the robot sees when the")
    print("person is at the far end of the requested 1-4 m range. If 'fist' still")
    print("reads there, the scenario needs no stronger visual representation; if")
    print("it collapses, that IS the case for collecting deployment-domain data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
