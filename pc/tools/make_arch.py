#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_arch.py -- system architecture figure for TECHNICAL-REPORT.md.

Deliberately sparse: two lines of text per block. Dense diagrams read as
cluttered; the numbers live in the tables, the figure carries the topology.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
EDGE = "#34495e"


def block(ax, x, y, w, h, title, sub, fc, tfs=9.6, sfs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc=fc, ec=EDGE, lw=1.3, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#1b2631", zorder=4)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sfs, color="#2c3e50", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#1b2631", zorder=4)


def arrow(ax, p0, p1, label="", color="#2c3e50", ls="-", fs=7.6, off=0.012):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.35, color=color, linestyle=ls,
                                 shrinkA=3, shrinkB=3, zorder=2))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + off, label,
                ha="center", va="bottom", fontsize=fs, color=color, zorder=5)


fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=165)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ---- column headers -------------------------------------------------------
for x, t in ((0.16, "PC"), (0.475, "Wi-Fi 2.4 GHz"), (0.775, "ESP32-S3")):
    ax.text(x, 0.965, t, ha="center", fontsize=11.5, fontweight="bold", color="#1b2631")
ax.text(0.16, 0.925, "RTX 5060 Laptop", ha="center", fontsize=8.2, color="#7f8c8d")
ax.text(0.775, 0.925, "240 MHz · no NPU · 64 KiB L1", ha="center", fontsize=8.2, color="#7f8c8d")

# ---- PC column ------------------------------------------------------------
block(ax, 0.035, 0.700, 0.250, 0.175, "GC0308 240x240", "10 Hz JPEG, on the head", "#e8f0fb")
block(ax, 0.035, 0.505, 0.250, 0.175, "YOLO26n-pose", "9.3 ms/frame on GPU", "#d6e6f7")
block(ax, 0.035, 0.310, 0.250, 0.175, "FeatureAdapter", "548-d -> 64-d (+ gesture events)", "#c5d9f1")

# ---- link column ----------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.360, 0.310), 0.230, 0.565,
                            boxstyle="round,pad=0.010,rounding_size=0.018",
                            fc="#f4f6f7", ec="#95a5a6", lw=1.1, ls="--", zorder=2))
rowy = [0.800, 0.695, 0.590, 0.485, 0.395]
rows = [("/capture", "10 Hz JPEG down"),
        ("/feat", "64-d features up, 10 Hz"),
        ("/model", "weights upload, 133 KB"),
        ("/cfg  /mode", "22 params, source switch"),
        ("", "553 KB/s · RTT 2 ms")]
for (a, b), y in zip(rows, rowy):
    if a:
        ax.text(0.475, y + 0.020, a, ha="center", fontsize=8.6, fontweight="bold", color="#1b2631")
        ax.text(0.475, y - 0.010, b, ha="center", fontsize=7.6, color="#2c3e50")
    else:
        ax.text(0.475, y, b, ha="center", fontsize=7.6, color="#7f8c8d", style="italic")

# ---- MCU column -----------------------------------------------------------
block(ax, 0.610, 0.700, 0.360, 0.175, "cfc_task · 50 Hz",
      "CfC forward: 4.04 ms fp32 / 1.40 ms int8", "#dff0d8")
block(ax, 0.610, 0.505, 0.360, 0.175, "control_arbiter · 6 ms",
      "safety shell + anti-windup + rate limit", "#fcf3cf")
block(ax, 0.610, 0.310, 0.360, 0.175, "robot_control · 167 Hz",
      "LQR balance K=[1300,5,25,0.4] + yaw P", "#fdebd0")
block(ax, 0.610, 0.115, 0.360, 0.175, "actuators",
      "2 hub motors · head servo +-45 deg · LCD", "#e8daef")

# ---- decision hub ---------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.035, 0.030), 0.555, 0.245,
                            boxstyle="round,pad=0.010,rounding_size=0.018",
                            fc="#fdf0e3", ec="#b9770e", lw=1.3, zorder=2))
ax.text(0.312, 0.245, "Decision hub — where should this be handled?",
        ha="center", fontsize=10.2, fontweight="bold", color="#7e5109", zorder=4)
ax.text(0.312, 0.212, "p_esc = sigmoid(b + sum_k w_k phi_k)   ·   logit = sum of contributions (<1e-9)",
        ha="center", fontsize=7.9, color="#7e5109", zorder=4)
for i, (name, lat, col) in enumerate([
        ("L0 reflex", "38-64 us", "#f9e79f"),
        ("L1 skills", "no net", "#f8c471"),
        ("L2 PC agent", "99.9 ms", "#f0b27a"),
        ("L3 cloud LLM", "740-860 ms", "#e59866")]):
    x = 0.048 + i * 0.136
    ax.add_patch(FancyBboxPatch((x, 0.070), 0.122, 0.110,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                fc=col, ec="#9c640c", lw=1.0, zorder=3))
    ax.text(x + 0.061, 0.140, name, ha="center", fontsize=8.4, fontweight="bold", zorder=4)
    ax.text(x + 0.061, 0.100, lat, ha="center", fontsize=8.4, color="#6e2c00", zorder=4)
    if i < 3:
        arrow(ax, (x + 0.123, 0.125), (x + 0.134, 0.125), color="#9c640c")

# ---- arrows ---------------------------------------------------------------
arrow(ax, (0.160, 0.695), (0.160, 0.648))          # capture -> yolo
arrow(ax, (0.160, 0.500), (0.160, 0.453))          # yolo -> adapter
arrow(ax, (0.286, 0.845), (0.358, 0.845), "", fs=7.0)
arrow(ax, (0.286, 0.400), (0.358, 0.400), "", fs=7.0)
ax.text(0.322, 0.860, "/capture", ha="center", fontsize=7.4, color="#2c3e50")
ax.text(0.322, 0.415, "/feat", ha="center", fontsize=7.4, color="#2c3e50")
arrow(ax, (0.592, 0.845), (0.608, 0.845), "")
arrow(ax, (0.790, 0.698), (0.790, 0.682))          # cfc -> arbiter
arrow(ax, (0.790, 0.503), (0.790, 0.487))          # arbiter -> control
arrow(ax, (0.790, 0.308), (0.790, 0.292))          # control -> actuators
arrow(ax, (0.690, 0.155), (0.592, 0.155), "", color="#b9770e", ls="--")
ax.text(0.641, 0.168, "L2/L3 calls", ha="center", fontsize=7.2, color="#b9770e")

ax.text(0.995, 0.012, "every number measured on this unit — see RESULTS.md",
        ha="right", fontsize=7.0, color="#95a5a6", style="italic")

fig.tight_layout()
p = os.path.join(OUT, "fig_architecture.png")
fig.savefig(p, bbox_inches="tight")
print(p)
