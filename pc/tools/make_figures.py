#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_figures.py -- figures for TECHNICAL-REPORT.md.

All numbers are measured on the target hardware (ESP32-S3 @240 MHz) or on the
held-out test split. Nothing here is synthetic.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- measurement data -------------------------------------------------------
# (weights KiB, ms/step, saturated?)  Saturated = the 200 Hz request cap was hit,
# so the true cost is only bounded above and the point is drawn as an upper arrow.
ROOFLINE = [
    (59.6, 5.00, True),
    (89.9, 4.98, True),
    (130.2, 4.04, False),   # trained policy, read from /status cfc.us
    (180.6, 5.56, False),
    (240.9, 6.90, False),
]

# Fit only the two largest unsaturated models; the 130 KiB trained policy is
# then a held-out point, so the agreement below is a real validation rather
# than a re-fit of the same data.
fit_pts = [(a, b) for a, b, sat in ROOFLINE if not sat and a > 150]
xs = np.array([p[0] for p in fit_pts])
ys = np.array([p[1] for p in fit_pts])
slope, intercept = np.polyfit(xs, ys, 1)
bw_mbs = 1024.0 / slope / 1000.0            # KiB/ms -> MB/s


def fig_roofline():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)

    xx = np.linspace(40, 260, 200)
    ax.plot(xx, intercept + slope * xx, "--", color="#c0392b", lw=1.6,
            label=f"fit: t = {intercept:.2f} ms + bytes / {bw_mbs:.0f} MB/s")

    for kib, ms, sat in ROOFLINE:
        if sat:
            ax.annotate("", xy=(kib, ms), xytext=(kib, ms + 1.05),
                        arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.4))
            ax.plot([kib], [ms], "o", mfc="white", mec="#7f8c8d", ms=7, mew=1.6)
        else:
            ax.plot([kib], [ms], "o", color="#2c3e50", ms=8)

    # highlight the trained policy
    ax.plot([130.2], [4.04], "o", color="#e67e22", ms=11, zorder=5)
    ax.annotate("trained policy\n(130 KiB, 4.04 ms)", xy=(130.2, 4.04),
                xytext=(150, 3.1), fontsize=8.5, color="#e67e22",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.1))

    ax.annotate("arrow = upper bound only\n(200 Hz request cap hit)",
                xy=(59.6, 6.0), fontsize=7.5, color="#7f8c8d", ha="left")

    # pure-compute floor annotation
    ax.axhline(intercept, color="#2980b9", lw=1.0, ls=":")
    ax.text(245, intercept + 0.12, f"compute floor {intercept:.2f} ms",
            fontsize=7.5, color="#2980b9", ha="right")

    ax.set_xlabel("weight blob size (KiB)")
    ax.set_ylabel("CfC forward pass (ms/step)")
    ax.set_title("CfC inference on ESP32-S3 @240 MHz is bandwidth-bound, not compute-bound",
                 fontsize=10)
    ax.set_xlim(40, 260)
    ax.set_ylim(0, 8.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_roofline.png")
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_gesture():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)

    # --- left: threshold-based vs event-based ------------------------------
    labels = ["trigger latency\n(ms)", "missed gestures\n(%)"]
    old = [3567, 90]
    new = [837, 0]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax1.bar(x - w / 2, old, w, label="threshold + cooldown", color="#c0392b")
    b2 = ax1.bar(x + w / 2, new, w, label="event-based (ours)", color="#27ae60")
    for bars, vals in ((b1, old), (b2, new)):
        for r, v in zip(bars, vals):
            ax1.text(r.get_x() + r.get_width() / 2, r.get_height(),
                     f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yscale("symlog", linthresh=10)
    ax1.set_ylabel("value (symlog)", fontsize=9)
    ax1.set_title("Gesture response: restructuring the signal\n"
                  "(measured, 1 Hz slow wave)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # --- right: per-class F1, skeleton-only vs +appearance ------------------
    # Only pointing and fist were measured in the skeleton-only ablation; the
    # other classes are left blank rather than invented.
    classes = ["idle", "wave", "open_palm", "thumbs_up", "pointing", "fist"]
    skel = [None, None, None, None, 0.000, 0.033]
    full = [0.884, 0.970, 0.822, 0.735, 0.574, 0.846]
    y = np.arange(len(classes))
    h = 0.36
    for i, s in enumerate(skel):
        if s is not None:
            ax2.barh(y[i] + h / 2, max(s, 0.004), h, color="#95a5a6")
            ax2.text(max(s, 0.012) + 0.01, i + h / 2, f"{s:.3f}", va="center",
                     fontsize=8, color="#c0392b", fontweight="bold")
    ax2.barh(y - h / 2, full, h, label="skeleton + 32x32 hand crop", color="#2980b9")
    for i, f in enumerate(full):
        ax2.text(f + 0.015, i - h / 2, f"{f:.3f}", va="center", fontsize=7.5)
    ax2.barh([], [], color="#95a5a6", label="skeleton only (COCO-17)\n(measured for 2 classes)")
    ax2.set_yticks(y)
    ax2.set_yticklabels(classes, fontsize=9)
    ax2.set_xlim(0, 1.45)
    ax2.set_xlabel("F1 (held-out users)", fontsize=9)
    ax2.set_title("COCO-17 has no finger keypoints, so pure skeleton\n"
                  "cannot separate pointing from open_palm", fontsize=10)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.grid(axis="x", alpha=0.25)
    ax2.invert_yaxis()

    fig.tight_layout()
    p = os.path.join(OUT, "fig_gesture.png")
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("fit: slope=%.6f ms/KiB  intercept=%.3f ms  ->  %.1f MB/s"
          % (slope, intercept, bw_mbs))
    print("predicted h=64 (130.2 KiB): %.2f ms  (measured 4.04 ms)"
          % (intercept + slope * 130.2))
    print(fig_roofline())
    print(fig_gesture())
