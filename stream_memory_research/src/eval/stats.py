"""Multi-seed aggregation + paired bootstrap vs a reference model (DESIGN §4).

The Lead pastes the output of :func:`comparison_table` into reports, so this
module is intentionally file-light: read ``runs/<run_id>/{metrics.json,
per_source.json}``, aggregate over seeds, and pair models *by source_id* (the
resampling unit required by DESIGN §4, n ~ number of videos) with
``src.common.metrics.paired_bootstrap_diff``.

Usage
-----
    from src.eval.stats import scan_runs, comparison_table, format_markdown
    runs = scan_runs()                                    # all finished runs
    rows = comparison_table(runs, metric="r2", ref="cfc")
    print(format_markdown(rows))
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.common import metrics as M
from src.common.contract import RUNS, read_json

LOWER_IS_BETTER = ("mae", "mse", "rmse", "loss", "mae_macro")


def higher_is_better(name: str) -> bool:
    return str(name) not in LOWER_IS_BETTER


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def parse_run_id(run_id: str) -> Dict[str, Any]:
    parts = str(run_id).split("__")
    seed = 0
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
    dataset = parts[0] if parts else ""
    model = parts[1] if len(parts) > 1 else ""
    variant = parts[2] if len(parts) >= 4 else "base"
    return {"run_id": run_id, "dataset": dataset, "model": model,
            "variant": variant, "seed": seed}


def load_run(run_dir) -> Optional[dict]:
    run_dir = Path(run_dir)
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return None
    try:
        m = read_json(mp)
    except Exception:
        return None
    ps = {}
    pp = run_dir / "per_source.json"
    if pp.exists():
        try:
            ps = read_json(pp)
        except Exception:
            ps = {}
    info = parse_run_id(m.get("run_id") or run_dir.name)
    info.update({"run_dir": str(run_dir), "metrics": m, "per_source": ps})
    return info


def scan_runs(root=RUNS, include_smoke: bool = False,
              status: Sequence[str] = ("complete",)) -> List[dict]:
    runs = []
    for d in sorted(Path(root).iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_") or d.name == "checkpoints":
            continue
        if (not include_smoke) and d.name.endswith("__smoke"):
            continue
        r = load_run(d)
        if r is None:
            continue
        if status and r["metrics"].get("status") not in status:
            continue
        runs.append(r)
    return runs


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def run_value(run: dict, split: str = "test", metric: Optional[str] = None) -> Optional[float]:
    m = run.get("metrics") or {}
    best = m.get("best") or {}
    metric = metric or m.get("main_metric") or "acc"
    v = (best.get(split) or {}).get(metric)
    return float(v) if v is not None and np.isfinite(v) else None


def per_source_values(run: dict, split: str = "test",
                      metric: Optional[str] = None) -> Dict[str, float]:
    ps = run.get("per_source") or {}
    metric = metric or (run.get("metrics") or {}).get("main_metric") or "acc"
    block = ps.get(split) or {}
    vals = block.get(metric) if isinstance(block, dict) else None
    if not isinstance(vals, dict):
        # per_source.json stores {"val": {metric: {...}}, "test": {...}}
        block2 = ps.get(split) or {}
        if isinstance(block2, dict) and metric in block2 and isinstance(block2[metric], dict):
            vals = block2[metric]
    return {str(k): float(v) for k, v in (vals or {}).items()
            if v is not None and np.isfinite(v)}


def group_runs(runs: Iterable[dict]) -> Dict[Tuple[str, str, str], List[dict]]:
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in runs:
        key = (r["dataset"], r["model"], r["variant"])
        groups.setdefault(key, []).append(r)
    for v in groups.values():
        v.sort(key=lambda r: r["seed"])
    return groups


def aggregate_seeds(runs: Iterable[dict], metric: Optional[str] = None,
                    split: str = "test") -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Per (dataset, model, variant): mean/std/bootstrap-CI over seeds."""
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, rs in group_runs(runs).items():
        vals = np.array([v for v in (run_value(r, split, metric) for r in rs) if v is not None],
                        dtype=np.float64)
        entry = {"n_seeds": int(vals.size), "seeds": [r["seed"] for r in rs],
                 "values": vals.tolist()}
        if vals.size:
            entry.update(M.bootstrap_ci(vals))
        else:
            entry.update({"mean": float("nan"), "std": float("nan"),
                          "lo": float("nan"), "hi": float("nan")})
        out[key] = entry
    return out


# --------------------------------------------------------------------------- #
# paired comparison vs a reference model
# --------------------------------------------------------------------------- #
def _paired_source_arrays(ref_run: dict, run: dict, split: str,
                          metric: str) -> Tuple[np.ndarray, np.ndarray, int]:
    a = per_source_values(ref_run, split, metric)
    b = per_source_values(run, split, metric)
    common = sorted(set(a) & set(b))
    if common:
        return (np.array([a[k] for k in common]), np.array([b[k] for k in common]),
                len(common))
    # fall back to seed-level pairing (only 3 points -> very weak CI)
    ra = run_value(ref_run, split, metric)
    rb = run_value(run, split, metric)
    if ra is None or rb is None:
        return np.zeros(0), np.zeros(0), 0
    return np.array([ra]), np.array([rb]), 1


def comparison_table(runs: Iterable[dict], metric: Optional[str] = None,
                     ref: str = "cfc", split: str = "test", n_boot: int = 1000,
                     seed: int = 0, dataset: Optional[str] = None) -> List[dict]:
    """One row per (dataset, model, variant) with a paired diff CI vs ``ref``."""
    runs = list(runs)
    groups = group_runs(runs)
    ref_by_ds: Dict[Tuple[str, str], List[dict]] = {}
    for (ds, model, variant), rs in groups.items():
        ref_by_ds.setdefault((ds, variant), []).append((model, rs))
    rows: List[dict] = []
    for (ds, model, variant), rs in sorted(groups.items()):
        if dataset and ds != dataset:
            continue
        m = metric or (rs[0]["metrics"].get("main_metric") or "acc")
        agg = aggregate_seeds(rs, m, split)[(ds, model, variant)]
        ref_runs = [x for mm, rr in ref_by_ds.get((ds, variant), []) if mm == ref for x in rr]
        ref_agg = (aggregate_seeds(ref_runs, m, split)[(ds, ref, variant)]
                   if ref_runs else None)
        row = {
            "dataset": ds, "model": model, "variant": variant, "metric": m,
            "split": split, "n_seeds": agg["n_seeds"],
            "mean": agg.get("mean"), "std": agg.get("std"),
            "lo": agg.get("lo"), "hi": agg.get("hi"),
            "ref": ref, "ref_mean": (ref_agg or {}).get("mean"),
            "diff": None, "diff_lo": None, "diff_hi": None, "p": None,
            "significant": None, "cohens_d": None, "rel_gain": None,
            "n_sources": 0, "higher_is_better": higher_is_better(m),
        }
        if ref_runs and model != ref:
            a_all, b_all, n_src = [], [], 0
            for rr in ref_runs:
                for ur in rs:
                    if rr["seed"] != ur["seed"]:
                        continue
                    a, b, n = _paired_source_arrays(rr, ur, split, m)
                    if n:
                        a_all.append(a)
                        b_all.append(b)
                        n_src += n
            if n_src:
                a = np.concatenate(a_all)
                b = np.concatenate(b_all)
                # paired_bootstrap_diff(a, b) = mean(a-b); we report new - ref
                # a,b are (ref, new), so swap
                d = M.paired_bootstrap_diff(b, a, n_boot=n_boot, seed=seed)
                row.update({"diff": d["diff"], "diff_lo": d["lo"], "diff_hi": d["hi"],
                            "p": d["p"], "n_sources": int(d["n"]),
                            "significant": bool(np.isfinite(d["lo"]) and np.isfinite(d["hi"])
                                                and (d["lo"] > 0 or d["hi"] < 0)),
                            "cohens_d": M.cohens_d(b, a),
                            "rel_gain": M.relative_gain(float(np.mean(b)), float(np.mean(a)))})
        rows.append(row)
    return rows


def format_markdown(rows: Sequence[dict]) -> str:
    """Compact markdown table the Lead can paste into a report."""
    head = ("| dataset | model | variant | metric | seeds | mean +/- std | 95% CI | "
            "vs ref | diff [95% CI] | p | sig | n_src |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        def f(v, fmt="{:.4f}"):
            return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)
        ci = f"[{f(r['lo'])}, {f(r['hi'])}]".replace("'", "")
        dci = (f"[{f(r['diff_lo'])}, {f(r['diff_hi'])}]" if r["diff"] is not None else "-")
        lines.append("| " + " | ".join([
            str(r["dataset"]), str(r["model"]), str(r["variant"]), str(r["metric"]),
            str(r["n_seeds"]), f"{f(r['mean'])} +/- {f(r['std'])}", ci,
            f"{r['ref']}={f(r['ref_mean'])}", dci, f(r["p"]), 
            ("YES" if r["significant"] else ("no" if r["significant"] is False else "-")),
            str(r["n_sources"]),
        ]) + " |")
    return "\n".join(lines)


def write_table(path, rows: Sequence[dict], metric_title: str = "") -> None:
    from src.common.contract import write_json
    write_json(path, {"title": metric_title, "n_rows": len(rows), "rows": list(rows),
                      "markdown": format_markdown(rows)})


def summarize_robustness(runs: Iterable[dict], split: str = "test") -> Dict[str, Dict[str, Any]]:
    """Mean +/- std of the robustness drop vs dense, grouped by model."""
    groups = group_runs(runs)
    out: Dict[str, Dict[str, Any]] = {}
    for (ds, model, variant), rs in sorted(groups.items()):
        m = (rs[0]["metrics"].get("main_metric") or "acc")
        per_level: Dict[str, List[float]] = {}
        for r in rs:
            best = (r["metrics"].get("best") or {})
            drop = best.get("robust_drop") or {}
            for lv, dv in drop.items():
                if dv is None:
                    continue
                per_level.setdefault(lv, []).append(float(dv))
        if per_level:
            out[f"{ds}::{model}::{variant}"] = {
                lv: {"mean_drop": float(np.mean(v)), "std": float(np.std(v)),
                     "n": len(v)}
                for lv, v in sorted(per_level.items())}
    return out
