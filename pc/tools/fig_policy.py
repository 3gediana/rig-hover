#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig_policy.py -- figure for the trained-policy results (RESULTS §4.5).

Three panels, each answering one question a reviewer actually asks:

  A  Does it beat doing nothing, and is that stable across seeds?
  B  Did it learn the three-tier head/body allocation the teacher was built
     around?
  C  Is the mode channel coherent, and does a debounce fix it?

Numbers are the measured ones recorded in RESULTS.md §4.5.3/§4.5.4; this script
only plots them, so re-running it never invents data.

Run:  python fig_policy.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_policy.png")

# ---- measured numbers (RESULTS.md §4.5.3 / §4.5.4) -------------------------
SEED_MAE = [1.03, 1.11, 1.18]          # held-out head MAE per seed
BASELINE_MAE = 7.79                    # "always command zero"
SMALL_DATA_MAE = [5.22, 5.97, 7.64, 8.54, 15.22]   # 30-source-image attempt

TIERS = ["<3°", "3-8°", "8-15°", "15-25°", ">25°"]
POLICY_HEAD = [1.17, 3.67, 8.07, 12.55, 7.12]
TEACHER_HEAD = [1.30, 4.18, 8.04, 13.15, 4.00]
POLICY_HEAD_SHARE = [82.7, 80.4, 76.4]     # tier1..3, policy
TEACHER_HEAD_SHARE = [100.0, 85.5, 76.2]

HOLDS = [0, 3, 6]
PRECISION = [42.9, 76.9, 88.9]
PULSES = [15, 8, 4]
TEACHER_PULSES = 5


def main() -> int:
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # ---- A: does it beat doing nothing, across seeds ------------------
    a = ax[0]
    xs = np.arange(len(SEED_MAE))
    a.bar(xs, SEED_MAE, width=0.55, color="#2b8cbe", label="policy (per seed)")
    a.axhline(BASELINE_MAE, color="#d7301f", lw=2,
              label=f"always-zero baseline ({BASELINE_MAE}°)")
    m = float(np.mean(SEED_MAE))
    # Sample standard deviation (ddof=1), matching how RESULTS.md quotes it.
    # Population std would print 0.06 here and disagree with the text.
    s = float(np.std(SEED_MAE, ddof=1))
    a.axhline(m, color="#047878", ls="--", lw=1.5,
              label=f"policy mean {m:.2f}° ± {s:.2f}°")
    for i, v in enumerate(SEED_MAE):
        a.text(i, v + 0.15, f"{v:.2f}°", ha="center", fontsize=9)
    # the failed small-data attempt, for contrast
    a.scatter([len(SEED_MAE) - 0.5], [np.mean(SMALL_DATA_MAE)], marker="x", s=90,
              color="#636363", zorder=5,
              label=f"30-image attempt: {np.mean(SMALL_DATA_MAE):.1f}° ± "
                    f"{np.std(SMALL_DATA_MAE, ddof=1):.1f}°")
    a.set_xticks(list(xs))
    a.set_xticklabels([f"seed {i}" for i in xs])
    a.set_ylabel("held-out head MAE (deg)")
    a.set_title("A. Accuracy, and whether it survives a seed change")
    a.legend(fontsize=7.5, loc="upper left")
    a.set_ylim(0, 16)

    # ---- B: three-tier allocation ------------------------------------
    b = ax[1]
    w = 0.38
    xb = np.arange(len(TIERS))
    b.bar(xb - w / 2, POLICY_HEAD, w, label="policy", color="#2b8cbe")
    b.bar(xb + w / 2, TEACHER_HEAD, w, label="teacher", color="#fdae6b")
    b.set_xticks(xb)
    b.set_xticklabels(TIERS, fontsize=9)
    b.set_xlabel("|eps_body| (bearing error bucket)")
    b.set_ylabel("mean |head| command (deg)")
    b.set_title("B. Command magnitude tracks the teacher per tier")
    b.legend(fontsize=8)

    axb2 = b.twinx()
    axb2.plot([0, 1, 2], TEACHER_HEAD_SHARE, "o--", color="#d7301f",
              label="teacher head share")
    axb2.plot([0, 1, 2], POLICY_HEAD_SHARE, "s-", color="#047878",
              label="policy head share")
    axb2.set_ylabel("head share of rotation (%)")
    axb2.set_ylim(60, 105)
    axb2.legend(fontsize=8, loc="lower left")

    # ---- C: mode coherence vs debounce -------------------------------
    c = ax[2]
    xc = np.arange(len(HOLDS))
    c.bar(xc, PRECISION, width=0.55, color="#2b8cbe")
    for i, v in enumerate(PRECISION):
        c.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    c.set_xticks(xc)
    c.set_xticklabels([f"hold {h} frames" for h in HOLDS])
    c.set_ylabel("mode-switch precision (%)")
    c.set_ylim(0, 100)
    c.set_title("C. Debouncing the mode channel removes the flicker")

    axc2 = c.twinx()
    axc2.plot(xc, PULSES, "o-", color="#d7301f", label="policy ACK pulses")
    axc2.axhline(TEACHER_PULSES, color="#636363", ls="--",
                 label=f"teacher ({TEACHER_PULSES} pulses)")
    axc2.set_ylabel("ACK pulses per held-out set")
    axc2.set_ylim(0, 18)
    axc2.legend(fontsize=8, loc="center right")

    fig.suptitle("Trained CfC policy: accuracy, allocation structure, and mode "
                 "coherence on held-out sessions", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
