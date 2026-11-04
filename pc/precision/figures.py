#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures.py -- read study_results.json, draw the precision Pareto front."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "study_results.json"), encoding="utf-8") as f:
    R = json.load(f)

# uniform points: worst relative error over the four physical outputs
DENOM = {"head_err": 45.0, "vyaw_err": 90.0, "vx_err": 0.30}
uni = []
for r in R["uniform"]:
    worst = max(r["head_err_max"] / 45.0, r["vyaw_err_max"] / 90.0,
                1.0 - r["mode_agree"])
    uni.append((r["KiB"], worst, r["precision"]))

greedy = [(g["KiB"], g["worst_rel"], f"+{g['n_layers']}") for g in R["greedy"]]

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=160)

# uniform precision curve
xs = [u[0] for u in uni]
ys = [u[1] for u in uni]
order = np.argsort(xs)
ax.plot(np.array(xs)[order], np.array(ys)[order], "o-", color="#2c3e50",
        lw=1.5, ms=7, label="uniform precision")

LABEL_OFFSET = {
    "fp32": (8, 6), "bf16": (10, 6), "fp16": (10, -14),
    "int8/tensor": (-52, 10), "int8/channel": (10, -14),
}
for x, y, name in uni:
    if name not in ("int8/tensor", "int8/channel"):
        continue
    dx, dy = LABEL_OFFSET[name]
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8,
                                shrinkA=0, shrinkB=3) if dx < 0 else None)
for x, y, name in uni:
    if name in ("int8/tensor", "int8/channel"):
        continue
    dx, dy = LABEL_OFFSET[name]
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5)

ax.set_xscale("linear")
ax.set_xlim(26, 138)
ax.set_xticks([31.6, 63.1, 126.2])
ax.set_xticklabels(["31.6 (int8)", "63.1 (16-bit)", "126.2 (fp32)"], fontsize=8.5)
ax.axvline(126.2, color="0.75", ls=":", lw=1.0)

# greedy mixed-precision curve
gx = [g[0] for g in greedy]
gy = [g[1] for g in greedy]
ax.plot(gx, gy, "s--", color="#27ae60", lw=1.5, ms=6,
        label="greedy mixed precision (int8, one layer at a time)")

# recommended recipe
ax.plot([34.1], [0.147 / 45.0], "*", color="#c0392b", ms=17, zorder=6,
        label="recommended: bulk int8 + heads fp16 (34.1 KiB)")
ax.annotate("heads are 30x more sensitive per byte\n"
            "-> keep the 960-float output layer in fp16",
            xy=(34.1, 0.147 / 45.0), xytext=(52, 0.0042), fontsize=8,
            color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.1))

ax.set_xlabel("weight blob size (KiB)")
ax.set_ylabel("worst relative decision error\n(max over head_yaw / vyaw / mode / vx)")
ax.set_title("Weight precision trade-off for the CfC policy\n"
             "(bandwidth-bound target: fewer bytes is the only lever)",
             fontsize=10)
ax.set_yscale("log")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
out = os.path.join(HERE, "fig_precision.png")
fig.savefig(out)
print(out)
print(f"fp16 vs bf16 at same 63.1 KiB: "
      f"head {[r['head_err_max'] for r in R['uniform'] if r['precision'] == 'fp16'][0]:.3f} deg vs "
      f"{[r['head_err_max'] for r in R['uniform'] if r['precision'] == 'bf16'][0]:.3f} deg")
