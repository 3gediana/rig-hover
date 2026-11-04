"""Aggregate the pixel-to-event matrix: mean/std over seeds, paired bootstrap,
stress breakdown, and the figures used by the report.

Statistics follow the protocol already established in this project: models are
compared **per video** with a paired bootstrap over the test videos (the same
video is scored by every model), 4000 resamples.  Only videos present in both
runs are paired, so a partial matrix still yields an honest comparison.

    python scripts/aggregate_pixel.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
OUT = ROOT / "runs_pixel"
REP = ROOT / "reports"
FIGS = REP / "figures"

ORDER = ["pixel_event", "pixel_event(hazard)", "frame_cnn", "gru", "lstm", "tcn",
         "transformer", "ssm", "frozen_dsm"]
METRICS = [("map", "mAP", True),
           ("phase_macro_f1", "phase macro-F1", True),
           ("boundary_f1", "onset F1", True),
           ("boundary_latency", "onset latency (frames, +late)", False),
           ("count_exact", "exact count acc", True),
           ("count_mae", "count MAE", False),
           ("ece", "ECE", False),
           ("brier", "Brier", False)]


def flat(e: dict) -> dict:
    t = e.get("test") or {}
    if not t:
        return {}
    ph = t.get("phase") or {}
    bd = t.get("boundary") or {}
    ct = t.get("count") or {}
    cal = t.get("calibration") or {}
    return {"map": ph.get("map"), "phase_macro_f1": ph.get("macro_f1"),
            "phase_precision": ph.get("precision"),
            "phase_recall": ph.get("recall"),
            "boundary_precision": bd.get("precision"),
            "boundary_recall": bd.get("recall"),
            "boundary_f1": bd.get("f1"),
            "boundary_latency": bd.get("lat"),
            "count_exact": ct.get("exact_accuracy"),
            "count_mae": ct.get("mae"),
            "count_over": ct.get("over_rate"),
            "count_under": ct.get("under_rate"),
            "ece": cal.get("ece"), "brier": cal.get("brier"), "mce": cal.get("mce")}


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 0):
    """Paired bootstrap over units (videos): mean(a-b), 95% CI, two-sided p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 8:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p = 2 * min(float((m <= 0).mean()), float((m >= 0).mean()))
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0)), len(d)


def main() -> int:
    runs = defaultdict(list)          # model -> [eval dict]
    for p in sorted(OUT.glob("*/eval.json")):
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = e.get("model") or p.parent.name.split("__")[1]
        # a run id may carry a trailing tag (e.g. __hazard); keep those as separate
        # rows so the event-head ablation shows up as its own line rather than
        # silently merging into the model's mean
        m = re.search(r"__s\d+__(.+)$", p.parent.name)
        if m:
            model = f"{model}({m.group(1)})"
        runs[model].append(e)

    models = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    rows = {}
    for m in models:
        vals = defaultdict(list)
        for e in runs[m]:
            for k, v in flat(e).items():
                if v is not None and np.isfinite(v):
                    vals[k].append(float(v))
        rows[m] = {k: (float(np.mean(v)), float(np.std(v)), len(v))
                   for k, v in vals.items()}

    # ---- main table ----
    lines = ["| model | n_seeds | mAP | phase macro-F1 | onset F1 | onset lat | count exact | count MAE | ECE | Brier |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = rows[m]
        g = lambda k: (f"{r[k][0]:.4f} ± {r[k][1]:.4f}" if k in r
                       else "—")
        n = max((r[k][2] for k in r), default=0)
        lines.append(f"| **{m}** | {n} | {g('map')} | {g('phase_macro_f1')} | {g('boundary_f1')} | "
                     f"{g('boundary_latency')} | {g('count_exact')} | {g('count_mae')} | "
                     f"{g('ece')} | {g('brier')} |")
    table = "\n".join(lines)

    # ---- paired bootstrap vs the proposed model ----
    boot_lines = []
    ref = runs.get("pixel_event") or []
    ref_map = {}
    for e in ref:
        for d in ((e.get("test") or {}).get("per_seq") or []):
            ref_map.setdefault(d["seq_id"], []).append(d["f1"])
    ref_f1 = {k: float(np.mean(v)) for k, v in ref_map.items()}
    boot = {}
    if ref_f1:
        boot_lines = ["| model | Δ phase macro-F1 vs pixel_event | 95% CI | p | n videos |",
                      "|---|---|---|---|---|"]
        for m in models:
            if m == "pixel_event":
                continue
            cur = {}
            for e in runs[m]:
                for d in ((e.get("test") or {}).get("per_seq") or []):
                    cur.setdefault(d["seq_id"], []).append(d["f1"])
            cur_f1 = {k: float(np.mean(v)) for k, v in cur.items()}
            keys = sorted(set(ref_f1) & set(cur_f1))
            a = np.array([ref_f1[k] for k in keys])
            b = np.array([cur_f1[k] for k in keys])
            mu, lo, hi, pv, n = paired_boot(a, b)
            boot[m] = {"delta": mu, "lo": lo, "hi": hi, "p": pv, "n": n}
            flag = "✅" if (np.isfinite(lo) and lo > 0) else ("❌" if (np.isfinite(hi) and hi < 0) else "—")
            boot_lines.append(f"| {m} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                              f"{pv:.1e} | {n} | {flag}")

    # ---- stress breakdown ----
    stress = {}
    for m in models:
        acc = defaultdict(list)
        for e in runs[m]:
            for k, v in (e.get("stress") or {}).items():
                f1 = ((v.get("phase") or {}).get("macro_f1"))
                if f1 is not None and np.isfinite(f1):
                    acc[k].append(float(f1))
        stress[m] = {k: float(np.mean(v)) for k, v in acc.items()}
    keys = sorted({k for m in stress for k in stress[m]})
    st_lines = ["| condition | " + " | ".join(models) + " |",
                "|---" * (len(models) + 1) + "|"]
    for k in keys:
        cells = [f"{stress[m][k]:.4f}" if k in stress[m] else "—" for m in models]
        st_lines.append(f"| {k} | " + " | ".join(cells) + " |")

    data = {"models": models, "rows": rows, "bootstrap": boot, "stress": stress,
            "table_md": table, "boot_md": "\n".join(boot_lines),
            "stress_md": "\n".join(st_lines),
            "n_runs": {m: len(runs[m]) for m in models}}
    (REP / "pixel_results.json").write_text(
        json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(table)
    print()
    print("\n".join(boot_lines))
    print()
    print("\n".join(st_lines))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        for ax, (k, lab, _hi) in zip(axes, [("map", "mAP", True),
                                            ("boundary_f1", "onset F1", True),
                                            ("count_mae", "count MAE", False)]):
            ms = [m for m in models if k in rows[m]]
            ys = [rows[m][k][0] for m in ms]
            es = [rows[m][k][1] for m in ms]
            ax.bar(range(len(ms)), ys, yerr=es, capsize=3,
                   color=["#d62728" if m == "pixel_event" else "#4c72b0" for m in ms])
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8)
            ax.set_title(lab)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("pixel-to-event: real Charades pixels, causal, same backbone "
                     "(mean ± std over seeds)")
        fig.tight_layout()
        fig.savefig(FIGS / "fig_pixel_main.png", dpi=140)
        print(f"\nwrote {FIGS / 'fig_pixel_main.png'}")
    except Exception as exc:                       # pragma: no cover
        print(f"[figure] skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
