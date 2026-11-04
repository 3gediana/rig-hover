# -*- coding: utf-8 -*-
"""_make_nofeat.py -- copy a training set with the 64-d visual feature block zeroed.

Why: every quantity the teacher's behaviour depends on already reaches the
policy through other channels -- obs[64] is theta_img, 65-69 are the person
geometry, 70-77 are the gesture primitives, and those are computed on the PC and
transmitted, so they cross the domain gap intact. The 64-d YOLO feature block is
the one input whose real population is nowhere near the training population
(mean-profile correlation -0.06), and it was measured to be non-load-bearing once
the normalisation is fixed: zeroing it changes recorded-robot results by nothing.

So train without it. A policy that cannot read the feature block cannot depend on
it, and the exported blob gets mean=0/std=1 on those dims so real features arrive
as their own small values rather than as a saturated sign.

Run:  python _make_nofeat.py --src data_pan_all --dst data_nofeat
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

FEAT = slice(0, 64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\robot_work\pc\train\data_pan_all")
    ap.add_argument("--dst", default=r"D:\robot_work\pc\train\data_nofeat")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.npz")))
    if not files:
        print(f"no npz under {a.src}")
        return 2
    os.makedirs(a.dst, exist_ok=True)

    n_frames = 0
    for p in files:
        z = np.load(p, allow_pickle=True)
        out = {k: np.array(z[k]) for k in z.files}
        obs = out["obs"]
        if obs.ndim == 3:
            obs = obs.reshape(-1, obs.shape[-1])
        if obs.shape[1] != 106:
            print(f"skip {os.path.basename(p)}: dim {obs.shape[1]}")
            continue
        obs = obs.copy()
        obs[:, FEAT] = 0.0
        out["obs"] = obs
        n_frames += obs.shape[0]
        np.savez(os.path.join(a.dst, os.path.basename(p)), **out)

    print(f"wrote {len(files)} sessions / {n_frames} frames to {a.dst}")
    print("feat block (obs[:, 0:64]) is identically zero in every frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
