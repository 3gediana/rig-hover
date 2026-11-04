#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""three_tier_check.py -- did the learned policy reproduce the three-tier
allocation the teacher was designed around?

The teacher allocates a bearing error across three regimes:
    |eps| small   -> move the HEAD only      (vyaw stays ~0)
    |eps| medium  -> head + slow body
    |eps| large   -> body turns, head counter-rotates

That structure is the point of the whole controller, and it is checkable from
held-out recordings: bucket frames by |eps_body| and look at what the policy
actually commanded on each channel.

Reports the policy next to the teacher on the same buckets, so "it learned the
structure" is a comparison, not a claim.

Run:
    python three_tier_check.py --blob ..\\train\\out_big\\m.blob ^
        --sessions ..\\train\\data_pan_holdout2
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\robot_work\pc\train")

from eval_session import build_host_engine, run_engine  # noqa: E402

BUCKETS = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 90)]
LABELS = ["<3° (tiny)", "3-8° (small)", "8-15° (medium)",
          "15-25° (large)", ">25° (huge)"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.sessions, "*.npz")))
    if not files:
        print(f"no npz in {a.sessions}")
        return 2

    import tempfile
    tmp = tempfile.mkdtemp(prefix="three_tier_")
    exe = build_host_engine(tmp)

    eps_all, p_head, p_vyaw, t_head, t_vyaw = [], [], [], [], []
    for f in files:
        z = np.load(f)
        if "eps_body" not in z.files:
            continue
        O = z["obs"].astype(np.float32)
        c = run_engine(exe, a.blob, O, tmp)
        # The first frames of a session have no meaningful eps (filter warm-up);
        # drop them so the buckets describe steady-state behaviour.
        e = np.abs(z["eps_body"].astype(np.float32))[10:]
        eps_all.append(e)
        p_head.append(np.abs(c["head"])[10:])
        p_vyaw.append(np.abs(c["vyaw"])[10:])
        t_head.append(np.abs(z["head_deg"].astype(np.float32))[10:])
        t_vyaw.append(np.abs(z["vyaw_dps"].astype(np.float32))[10:])

    if not eps_all:
        print("sessions carry no eps_body meta -- cannot bucket")
        return 2

    E = np.concatenate(eps_all)
    PH, PV = np.concatenate(p_head), np.concatenate(p_vyaw)
    TH, TV = np.concatenate(t_head), np.concatenate(t_vyaw)

    print(f"frames: {len(E)}   (from {len(files)} held-out sessions)\n")
    print(f"{'|eps_body|':<16}{'frames':>8}{'head(deg)':>18}{'vyaw(deg/s)':>22}")
    print(f"{'':16}{'':8}{'policy  teacher':>18}{'policy   teacher':>22}")
    print("-" * 66)
    rows = []
    for (lo, hi), lab in zip(BUCKETS, LABELS):
        m = (E >= lo) & (E < hi)
        if m.sum() < 20:
            continue
        rows.append((lab, int(m.sum()), PH[m].mean(), TH[m].mean(),
                     PV[m].mean(), TV[m].mean()))
        print(f"{lab:<16}{int(m.sum()):>8}{PH[m].mean():>9.2f}{TH[m].mean():>9.2f}"
              f"{PV[m].mean():>11.2f}{TV[m].mean():>11.2f}")
    print("-" * 66)

    # ---- the actual three-tier question ----------------------------------
    # Tier 1 must be "head moves, body mostly stays". Measure how much of the
    # policy's rotation happens on the head in the smallest bucket, and how
    # many of those frames keep the body quiet.
    print("\nthree-tier behaviour check")
    tiny = E < 3.0
    small = (E >= 3.0) & (E < 8.0)
    large = E >= 15.0

    def ratio(mask):
        h, v = PH[mask].mean(), PV[mask].mean()
        tot = h + v * (1.0 / 8.0)      # 8 deg/s is roughly one head-degree/sec
        return h / tot if tot > 1e-6 else float("nan")

    def tr(mask):
        h, v = TH[mask].mean(), TV[mask].mean()
        tot = h + v * (1.0 / 8.0)
        return h / tot if tot > 1e-6 else float("nan")

    print(f"  tier 1 (|eps|<3°)   head share of rotation : {100*ratio(tiny):5.1f}%"
          f"   frames body quiet (<2 °/s): {100*float((PV[tiny] < 2.0).mean()):5.1f}%")
    print(f"  tier 2 (3-8°)       head share of rotation : {100*ratio(small):5.1f}%")
    print(f"  tier 3 (|eps|≥15°)  head share of rotation : {100*ratio(large):5.1f}%")
    print(f"  teacher, same tiers -> tier1 {100*tr(tiny):.1f}%  "
          f"tier2 {100*tr(small):.1f}%  tier3 {100*tr(large):.1f}%")

    mono = (ratio(large) < ratio(small) < ratio(tiny))
    print(f"\n  head share decreases as the error grows (the defining property "
          f"of the three-tier design): {mono}")
    print("\nPASS -- the learned policy shows the three-tier allocation"
          if mono else "\nFAIL -- the allocation is not monotone in the error")
    return 0 if mono else 1


if __name__ == "__main__":
    sys.exit(main())
