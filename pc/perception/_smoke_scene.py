# -*- coding: utf-8 -*-
"""_smoke_scene.py -- 新信号在真实照片上的直观看数。

不是单元测试（那在 `test_scene_signals.py`），而是把完整感知链跑在几张真实
HaGRID 照片上，确认 `theta_face` / `face_scale` / `theta_fist` / `wrist_y_norm`
确实被填上了合理的值，而不是恒零。

Run:  python _smoke_scene.py
"""
from __future__ import annotations

import os
import random
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, "..", "bench"))
sys.path.insert(0, HERE)
sys.path.insert(0, BENCH)

import real_image_gap as RIG  # noqa: E402


def main() -> int:
    rows = RIG.load_rows()
    rng = random.Random(0)
    by_class: dict[str, list] = {}
    for r in rows:
        by_class.setdefault(r["hagrid_class"], []).append(r)

    from run_perception import PerceptionPipeline  # noqa: E402
    pipe = PerceptionPipeline(verbose=False)

    hdr = (f"{'class':<11}{'seen':>5}{'f_valid':>8}{'theta_face':>12}"
           f"{'face_scale':>12}{'theta_fist':>12}{'wrist_y':>9}{'side':>6}")
    print(hdr)
    print("-" * len(hdr))
    n_nonzero = 0
    n_total = 0
    for cls in ("fist", "palm", "no_gesture"):
        rs = by_class.get(cls)
        if not rs:
            continue
        rs = rs[:]
        rng.shuffle(rs)
        for r in rs[:3]:
            img = cv2.imread(r["path"])
            if img is None:
                continue
            out = None
            for _ in range(12):
                out = pipe(img)
            n_total += 1
            n_nonzero += int(out.face_valid or out.hand_side != "")
            print(f"{cls:<11}{out.seen:>5}{out.face_valid:>8}"
                  f"{out.theta_face:>12.2f}{out.face_scale:>12.4f}"
                  f"{out.theta_fist:>12.2f}{out.wrist_y_norm:>9.3f}"
                  f"{out.hand_side:>6}")
    print("-" * len(hdr))
    print(f"{n_nonzero}/{n_total} 帧至少给出了一条新信号")
    return 0 if n_nonzero else 1


if __name__ == "__main__":
    sys.exit(main())
