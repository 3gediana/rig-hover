"""Cut the hand region the gesture classifier actually sees, grouped by prediction.

A confusion matrix needs ground truth. When the operator's timing is unknown,
the next best evidence is the pixels: crop the wrist neighbourhood out of frames
the classifier labelled `fist` and frames it labelled `thumbs_up`, blow them up,
and put them side by side. If both boxes hold the same pose, the classifier is
inventing the difference.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"D:\robot_work\evidence\live\gesture_2")
AN = Path(r"D:\robot_work\evidence\live\gesture_2_analysis")
OUT = AN / "hand_by_prediction"
NAMES = {0: "none", 1: "wave", 2: "open_palm", 3: "thumbs_up", 4: "pointing", 5: "fist"}
CROP = 44          # half-size of the wrist crop, in source pixels
SIZE = 160         # display size of each crop


def hand_crop(img, wrists):
    """Union crop around every confident wrist, resized to SIZE."""
    h, w = img.shape[:2]
    if not wrists:
        return None
    xs = [p[0] for p in wrists]
    ys = [p[1] for p in wrists]
    cx = int(round(float(np.mean(xs))))
    cy = int(round(float(np.mean(ys))))
    x1, x2 = max(0, cx - CROP), min(w, cx + CROP)
    y1, y2 = max(0, cy - CROP), min(h, cy + CROP)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(AN / "detections.csv", encoding="utf-8")))
    by_class: dict[int, list] = {}
    for r in rows:
        try:
            wrists = ast.literal_eval(r["wrists"])
        except (ValueError, SyntaxError):
            continue
        if not wrists:
            continue
        by_class.setdefault(int(r["gesture_raw"]), []).append((int(r["idx"]), r, wrists))

    print("frames with a confident wrist, by predicted class:")
    for k in sorted(by_class):
        print(f"  {k} {NAMES.get(k,'?'):<10} {len(by_class[k]):3d}")

    cols = 6
    for cls in sorted(by_class):
        picks = by_class[cls][:cols]
        if not picks:
            continue
        tiles = []
        for idx, r, wrists in picks:
            img = cv2.imread(str(SRC / "frames" / f"{idx:05d}.jpg"))
            if img is None:
                continue
            # Draw every wrist the pipeline reported so the crop is auditable.
            vis = img.copy()
            for (x, y, c) in wrists:
                cv2.circle(vis, (int(x), int(y)), 3, (255), -1)
            crop = hand_crop(img, wrists)
            if crop is None:
                continue
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            for (x, y, c) in wrists:
                cv2.circle(crop, (int((x - (np.mean([w[0] for w in wrists])) + CROP)) * SIZE // (2 * CROP),
                                  int((y - (np.mean([w[1] for w in wrists])) + CROP)) * SIZE // (2 * CROP)),
                           4, (0, 0, 255), -1)
            label = f"{idx} {NAMES.get(cls,'?')} c={float(r['gesture_conf']):.2f}"
            cv2.putText(crop, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
            tiles.append(crop)
            full = cv2.resize(vis, (SIZE, SIZE))
            cv2.putText(full, f"{idx} full", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
            tiles.append(full)
        if not tiles:
            continue
        n = len(tiles)
        rowsn = (n + cols - 1) // cols
        sheet = np.full((rowsn * SIZE, cols * SIZE, 3), 24, np.uint8)
        for i, t in enumerate(tiles):
            r0, c0 = (i // cols) * SIZE, (i % cols) * SIZE
            sheet[r0:r0 + SIZE, c0:c0 + SIZE] = t
        path = OUT / f"pred_{cls}_{NAMES.get(cls,'?')}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
