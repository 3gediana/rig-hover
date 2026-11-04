# -*- coding: utf-8 -*-
"""Inspect real-robot recordings: what channels are populated, what domains look like.

ASCII-only output so it is safe to run in a GBK console.
"""
from __future__ import annotations

import sys
import numpy as np

OBS_NAMES = (
    [f"feat{i}" for i in range(64)]
    + [f"tgt{i}" for i in range(6)]
    + [f"gest{i}" for i in range(8)]
    + [f"prop{i}" for i in range(22)]
    + [f"fb{i}" for i in range(4)]
    + [f"clk{i}" for i in range(2)]
)


def describe(path):
    z = np.load(path, allow_pickle=True)
    print("=" * 78)
    print("FILE", path)
    for k in z.files:
        a = z[k]
        print(f"  {k:18s} shape={str(a.shape):14s} dtype={a.dtype}")
    return z


def main():
    paths = sys.argv[1:]
    for p in paths:
        z = describe(p)
        obs = None
        for key in ("obs", "x", "X"):
            if key in z.files:
                obs = np.asarray(z[key], dtype=np.float32)
                break
        if obs is None:
            print("  (no obs array)")
            continue
        if obs.ndim == 3:
            obs = obs.reshape(-1, obs.shape[-1])
        n, d = obs.shape
        print(f"  frames={n}  dim={d}")
        if d != 106:
            print("  !! dim != 106, skipping stats")
            continue

        # per-channel min/max on selected interesting channels
        print("\n  channel sanity (min / mean / max / std):")
        groups = [
            ("feat 0..63", range(0, 64)),
            ("target 64..69", range(64, 70)),
            ("gesture 70..77", range(70, 78)),
            ("proprio 78..99", range(78, 100)),
            ("feedback 100..103", range(100, 104)),
            ("clock 104..105", range(104, 106)),
        ]
        for name, rng in groups:
            sub = obs[:, list(rng)]
            print(f"    {name:16s} min={sub.min():9.4f} mean={sub.mean():9.4f} "
                  f"max={sub.max():9.4f} std={sub.std():9.4f} "
                  f"zeros={(sub == 0).mean() * 100:5.1f}%")

        # which channels are all-zero
        dead = np.where(np.abs(obs).max(axis=0) < 1e-9)[0]
        print(f"\n  dead channels ({len(dead)}): {dead[:40].tolist()}"
              f"{' ...' if len(dead) > 40 else ''}")

        # clip rate under +-8 (the policy's typical normalized range)
        outside = (np.abs(obs) > 8.0).mean()
        print(f"  |obs|>8 rate: {outside * 100:.2f}%")
        for i in np.argsort(-(np.abs(obs) > 8.0).mean(axis=0))[:8]:
            print(f"    ch{i:3d} {OBS_NAMES[i]:10s} outside={((np.abs(obs[:, i]) > 8.0).mean() * 100):5.1f}% "
                  f"min={obs[:, i].min():8.3f} max={obs[:, i].max():8.3f}")


if __name__ == "__main__":
    main()
