"""Generate the report figures from runs/*/probes_all.json + runs/*/metrics.json.

Figures produced in reports/figures/:
  fig_gap_curve.png        recall vs occlusion duration (H1)  — per model, one panel per seed-avg
  fig_capacity_curve.png   recall vs number of simultaneous objects (H7)
  fig_crosstalk.png        crosstalk matrices (H3/H4)
  fig_robust.png           metric retention under stream perturbations (D)
  fig_prefix.png           causal online prefix accuracy (E)
  fig_event_gap.png        post-event memory vs gap (C)
  fig_main_bars.png        main metric per dataset with 95% CI
  fig_shuffled_drop.png    temporal-shuffle control (is the "memory" real?)
  fig_params_latency.png   params vs latency scatter

Usage: python scripts/make_figures.py [--datasets ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common import contract  # noqa: E402

FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["mlp", "gru", "lstm", "cfc", "cfc__widebackbone", "mamba", "mamba__tdt",
               "deltanet", "dsm", "dsm__ntok", "dsm_slot", "dsm_slot__ntok",
               "dsm__nodt", "dsm__no_decay", "dsm__h_by_index"]
COLORS = {}


def model_label(rid: str) -> str:
    parts = rid.split("__")
    ds, rest = parts[0], "__".join(parts[1:])
    name = rest.split("__s")[0]
    return name


def short(m: str) -> str:
    return m.replace("__", "-")


def color_for(name: str):
    if name not in COLORS:
        cmap = plt.get_cmap("tab20")
        COLORS[name] = cmap(len(COLORS) % 20)
    return COLORS[name]


def load_runs(metric_filter=None) -> Dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "*" / "metrics.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") != "complete":
            continue
        out[d["run_id"]] = d
    return out


def load_probes() -> Dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(str(ROOT / "runs" / "*" / "probes_all.json"))):
        try:
            out[Path(p).parent.name] = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
    return out


def group_by_dataset(runs) -> Dict[str, Dict[str, List[dict]]]:
    g: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for rid, d in runs.items():
        g[d.get("dataset", "?")][model_label(rid)].append(d)
    return g


def _metric_of(d, split="test"):
    mm = d.get("main_metric", "cos")
    return (d.get("best", {}) or {}).get(split, {}).get(mm)


def _save(fig, name):
    p = FIGS / name
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p.name}")


# --------------------------------------------------------------------------- #
def fig_gap_curve(probes):
    """model -> gap_s -> cos, from the *_gap*.*s manifests."""
    curves: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rid, pr in probes.items():
        name = model_label(rid)
        for man, r in pr.items():
            if "_gap" not in man:
                continue
            g = r.get("gap_s")
            if g is None:
                continue
            c = (r.get("metrics", {}) or {}).get("main", {}).get("cos")
            if c is not None:
                curves[name][float(g)].append(float(c))
    if not curves:
        print("  [skip] gap curve: no probes")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in sorted(curves, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n)):
        xs = sorted(curves[name])
        ys = [float(np.mean(curves[name][x])) for x in xs]
        ax.plot(xs, ys, "o-", label=short(name), color=color_for(name), lw=2, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("occlusion / absence gap (s, log scale)")
    ax.set_ylabel("recall cosine on hidden steps")
    ax.set_title("H1 | long-term recall vs occlusion duration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig_gap_curve.png")


def fig_capacity_curve(probes):
    curves: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rid, pr in probes.items():
        name = model_label(rid)
        for man, r in pr.items():
            k = r.get("capacity_k")
            if k is None:
                continue
            c = (r.get("metrics", {}) or {}).get("main", {}).get("cos")
            if c is not None:
                curves[name][int(k)].append(float(c))
    if not curves:
        print("  [skip] capacity: no probes")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in sorted(curves, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n)):
        xs = sorted(curves[name])
        ys = [float(np.mean(curves[name][x])) for x in xs]
        ax.plot(xs, ys, "o-", label=short(name), color=color_for(name), lw=2, ms=5)
    ax.set_xlabel("number of simultaneously tracked objects K")
    ax.set_ylabel("recall cosine")
    ax.set_title("H7 | effective memory capacity (recall vs K)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig_capacity_curve.png")


def fig_crosstalk(probes):
    mats = {}
    for rid, pr in probes.items():
        for man, r in pr.items():
            m = (r.get("metrics", {}) or {}).get("crosstalk_matrix")
            if m:
                mats[model_label(rid)] = np.array([[np.nan if v is None else v for v in row]
                                                   for row in m], dtype=float)
    if not mats:
        print("  [skip] crosstalk: no probes")
        return
    names = sorted(mats, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))[:6]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0][j]
        M = mats[name]
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(1e-6, np.nanmax(M)))
        ax.set_title(f"{short(name)}\noffdiag={np.nanmean(M - np.diag(np.diag(M))):.3f}", fontsize=8)
        ax.set_xlabel("updated obj", fontsize=7)
        ax.set_ylabel("read obj", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("H3/H4 | memory cross-talk (should be diagonal-dominant)", fontsize=10)
    _save(fig, "fig_crosstalk.png")


def fig_robust(runs, datasets=None):
    g = group_by_dataset(runs)
    levels = ["stride2", "stride4", "drop30", "drop50", "drop70", "burst", "repeat",
              "jitter", "shuffled", "mixed"]
    for ds in sorted(g):
        if datasets and ds not in datasets:
            continue
        if not ds.startswith("synth"):
            continue          # real-video tasks only have their own robustness
        fig, ax = plt.subplots(figsize=(8, 4.5))
        names = sorted(g[ds], key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            vals = []
            for lv in levels:
                xs = []
                for d in g[ds][name]:
                    dense = (d.get("best", {}).get("robust", {}) or {}).get("dense", {})
                    pert = (d.get("best", {}).get("robust", {}) or {}).get(lv) or {}
                    mm = d.get("main_metric", "cos")
                    a, b = pert.get(mm), dense.get(mm)
                    if a is not None and b is not None:
                        xs.append(a - b)
                vals.append(float(np.mean(xs)) if xs else 0.0)
            ax.bar(np.arange(len(levels)) + i * width, vals, width, label=short(name),
                   color=color_for(name))
        ax.set_xticks(np.arange(len(levels)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels(levels, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("metric drop vs dense")
        ax.set_title(f"{ds} | robustness (lower drop = better)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25, axis="y")
        _save(fig, f"fig_robust_{ds}.png")


def fig_prefix(runs):
    g = group_by_dataset(runs)
    for ds in sorted(g):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        drew = False
        for name in sorted(g[ds], key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n)):
            keys = None
            curves = []
            for d in g[ds][name]:
                pref = (d.get("best", {}) or {}).get("prefix") or {}
                if not pref:
                    continue
                if keys is None:
                    keys = sorted(pref, key=lambda k: float(k.split("@")[-1]))
                ys = [pref.get(k) for k in keys]
                if all(v is None for v in ys):
                    continue
                curves.append([np.nan if v is None else v for v in ys])
            if curves:
                ax.plot([float(k.split("@")[-1]) for k in keys], np.nanmean(curves, 0),
                        "o-", label=short(name), color=color_for(name))
                drew = True
        if not drew:
            plt.close(fig)
            continue
        ax.set_xlabel("fraction of the causal prefix observed")
        ax.set_ylabel("metric")
        ax.set_title(f"{ds} | causal online prefix accuracy (E)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _save(fig, f"fig_prefix_{ds}.png")


def fig_main_bars(runs):
    g = group_by_dataset(runs)
    for ds in sorted(g):
        names = sorted(g[ds], key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))
        means, los, his = [], [], []
        for name in names:
            vals = [_metric_of(d) for d in g[ds][name]]
            vals = [v for v in vals if v is not None]
            if not vals:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            from src.common.metrics import bootstrap_ci
            ci = bootstrap_ci(np.array(vals))
            means.append(ci["mean"]); los.append(ci["lo"]); his.append(ci["hi"])
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4.2))
        xs = np.arange(len(names))
        ax.bar(xs, means, color=[color_for(n) for n in names])
        ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los),
                                     np.array(his) - np.array(means)],
                    fmt="none", ecolor="k", capsize=3, lw=1)
        mm = g[ds][names[0]][0].get("main_metric", "cos")
        ax.set_xticks(xs)
        ax.set_xticklabels([short(n) for n in names], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(mm)
        ax.set_title(f"{ds} | test {mm} (mean + 95% bootstrap CI)")
        ax.grid(alpha=0.25, axis="y")
        _save(fig, f"fig_main_{ds}.png")


def fig_shuffled(probes):
    drops = defaultdict(list)
    for rid, pr in probes.items():
        name = model_label(rid)
        base = None
        shuf = None
        for man, r in pr.items():
            c = (r.get("metrics", {}) or {}).get("main", {}).get("cos")
            if c is None:
                continue
            if man.endswith("_shuffled"):
                shuf = c
            elif "gap" not in man and "capacity" not in man:
                base = c
        if base is not None and shuf is not None:
            drops[name].append(base - shuf)
    if not drops:
        print("  [skip] shuffled: no probes")
        return
    names = sorted(drops, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4.2))
    xs = np.arange(len(names))
    ax.bar(xs, [float(np.mean(drops[n])) for n in names],
           color=[color_for(n) for n in names])
    for i, n in enumerate(names):
        ax.text(i, float(np.mean(drops[n])), f"{np.mean(drops[n]):.2f}", ha="center",
                va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([short(n) for n in names], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("recall drop when observations are time-shuffled")
    ax.set_title("Control | a real memory model must DROP on shuffled observations")
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "fig_shuffled_drop.png")


def fig_params_latency(runs):
    pts = {}
    for rid, d in runs.items():
        b = d.get("best", {}) or {}
        lat = b.get("latency_ms_per_step")
        p = d.get("params")
        if lat is None or not p:
            continue
        pts[model_label(rid)] = (float(p), float(lat))
    if not pts:
        print("  [skip] params/latency")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, (p, l) in sorted(pts.items()):
        ax.scatter(p, l, s=70, color=color_for(name))
        ax.annotate(short(name), (p, l), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("parameters")
    ax.set_ylabel("ms / step (streaming)")
    ax.set_title("cost | params vs single-step latency")
    ax.grid(alpha=0.3)
    _save(fig, "fig_params_latency.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None)
    args = ap.parse_args()
    ds = set(args.datasets.split(",")) if args.datasets else None
    print("[figures]")
    runs = load_runs()
    probes = load_probes()
    print(f"  {len(runs)} completed runs, {len(probes)} with probes")
    fig_gap_curve(probes)
    fig_capacity_curve(probes)
    fig_crosstalk(probes)
    fig_robust(runs, ds)
    fig_prefix(runs)
    fig_main_bars(runs)
    fig_shuffled(probes)
    fig_params_latency(runs)
    print(f"[figures] -> {FIGS}")


if __name__ == "__main__":
    main()
