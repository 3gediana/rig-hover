# -*- coding: utf-8 -*-
"""_real_domain.py -- does a trained policy's input envelope cover REAL ROBOT frames?

obs_domain_check.py measures the gap on raw HaGRID photos, which are 512x512 hand
close-ups -- a framing the robot camera never produces. The robot camera produces
a 240x240 view of a room. The honest domain test is therefore the recorded robot
observations themselves.

Reads one or more policy.npz and one or more recorded session .npz and reports the
per-group clip ratio, using the same clip_sigma the firmware uses.

Run:  python _real_domain.py --sessions data_realtest/real_probe.npz \
                             --policies policy.npz out_sw2_b256_80e/policy.npz
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

FEAT_DIMS = 64
CAT_NAMES = ("feat", "target", "gesture", "proprio", "feedback", "clock")
CAT_SLICES = (slice(0, 64), slice(64, 70), slice(70, 78),
              slice(78, 100), slice(100, 104), slice(104, 106))


def load_sessions(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=True)
            if "obs" not in z.files:
                continue
            o = np.asarray(z["obs"], np.float32)
            if o.ndim == 3:
                o = o.reshape(-1, o.shape[-1])
            if o.shape[1] != 106:
                print(f"  skip {os.path.basename(p)}: dim {o.shape[1]}")
                continue
            rows.append((p, o))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    a = ap.parse_args()

    sess = load_sessions(a.sessions)
    if not sess:
        print("no usable sessions")
        return 2
    total = sum(o.shape[0] for _, o in sess)
    print(f"sessions: {len(sess)} file(s), {total} frames total")
    for p, o in sess:
        print(f"  {os.path.basename(p):24s} {o.shape[0]:5d} frames")

    print()
    hdr = (f"{'policy':<28}{'sigma':>6}{'ALL':>8}"
           + "".join(f"{n:>9}" for n in CAT_NAMES) + f"{'worst64':>9}")
    print(hdr)
    print("-" * len(hdr))

    for pol_path in a.policies:
        if not os.path.exists(pol_path):
            print(f"{os.path.basename(pol_path):<28} MISSING")
            continue
        pol = np.load(pol_path)
        mean = pol["input_mean"].astype(np.float32)
        std = np.maximum(pol["input_std"].astype(np.float32), 1e-6)
        sigma = float(pol["clip_sigma"]) if "clip_sigma" in pol.files else 4.0

        all_clip = []
        cat_clip = [[] for _ in CAT_NAMES]
        worst = []
        for _, o in sess:
            z = np.abs((o - mean) / std) > sigma
            all_clip.append(z)
            for k, sl in enumerate(CAT_SLICES):
                cat_clip[k].append(z[:, sl])
            if z.shape[1] >= FEAT_DIMS:
                worst.append((z[:, :FEAT_DIMS].mean(axis=0) > 0.5).sum())
        allz = np.concatenate(all_clip, axis=0)
        line = f"{os.path.basename(pol_path):<28}{sigma:>6.1f}{100*allz.mean():>7.1f}%"
        for k in range(len(CAT_NAMES)):
            c = np.concatenate(cat_clip[k], axis=0)
            line += f"{100*c.mean():>8.1f}%"
        line += f"{int(np.sum(worst)):>9d}"
        print(line)

        # per-dim detail for the worst feature dims
        fr = allz[:, :FEAT_DIMS].mean(axis=0)
        order = np.argsort(-fr)[:6]
        print(f"    worst feat dims: "
              + ", ".join(f"f{i}={100*fr[i]:.0f}%" for i in order))
        # a dim that clips 100% of the time is a dead input, not a saturated one
        dead = int((fr > 0.99).sum())
        print(f"    feat dims clipping >99%: {dead}/{FEAT_DIMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
