"""Worst-N sequence dump with a one-line diagnosis (DESIGN §3/§4.1).

Writes BOTH
  * ``failure_cases/<run_id>.json``   (project-level, Design §5)
  * ``runs/<run_id>/failure_cases.json`` (per-run artifact)
so the Lead can analyse failures without re-running anything.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.common.contract import FAILURES, now_str, write_json
from src.eval.evaluator import per_seq_main_values


def _pct(values: Sequence[float], q: float) -> float:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.percentile(v, q)) if v else float("nan")


def _diagnose(kind: str, rec: dict, head: str, value: float,
              higher_better: bool, t_p10: float, t_p90: float) -> Tuple[str, dict]:
    """Short rule-based diagnosis + the numbers behind it."""
    tips: List[str] = []
    details: Dict[str, Any] = {}
    h = rec["heads"].get(head)
    if h is None:
        return "no_head_data", details
    if not np.isfinite(value):
        return "no_supervised_steps", {"value": None}
    T = int(rec.get("T") or 0)
    details["T"] = T
    if np.isfinite(t_p10) and T <= t_p10:
        tips.append("short_sequence")
    dt_raw = rec.get("dt")
    dt = np.asarray(dt_raw if dt_raw is not None else [], dtype=np.float64)
    if dt.size > 2 and dt[1:].mean() > 1e-9:
        cv = float(dt[1:].std() / dt[1:].mean())
        details["dt_cv"] = round(cv, 4)
        details["dt_mean"] = round(float(dt[1:].mean()), 6)
        if cv > 0.5:
            tips.append("irregular_sampling")
    pred = np.asarray(h["pred"])
    target = np.asarray(h["target"])
    valid = np.asarray(h["valid"], dtype=bool)
    n_valid = int(valid.sum())
    details["n_valid"] = n_valid
    if n_valid == 0:
        return "no_supervised_steps", details
    if kind == "ce":
        p = pred[valid].astype(np.int64)
        t = target[valid].astype(np.int64)
        vals, cnts = np.unique(p, return_counts=True)
        top = int(vals[int(cnts.argmax())])
        share = float(cnts.max() / cnts.sum())
        details["pred_top_class"] = top
        details["pred_top_share"] = round(share, 4)
        if share > 0.9:
            tips.append("collapsed_prediction")
        tv, tc = np.unique(t, return_counts=True)
        details["n_true_classes"] = int(tv.size)
        details["true_top_class"] = int(tv[int(tc.argmax())])
        details["seq_majority_baseline"] = round(float(tc.max() / tc.sum()), 4)
        if value + 1e-9 < details["seq_majority_baseline"]:
            tips.append("below_majority_baseline")
        details["acc"] = round(float(value), 4)
    elif kind == "mse":
        p = pred[valid].astype(np.float64)
        t = target[valid].astype(np.float64)
        err = np.abs(p - t)
        details["mae"] = round(float(err.mean()), 6)
        details["err_p95"] = round(float(np.percentile(err, 95)), 6)
        details["target_std"] = round(float(t.std()), 6)
        if t.std() < 1e-6:
            tips.append("degenerate_target")
        if float(err.max()) > 5.0 * max(float(err.mean()), 1e-6):
            tips.append("heavy_tail_errors")
    elif kind == "bce":
        p = pred[valid].astype(np.float64)
        t = (target[valid].astype(np.float64) >= 0.5)
        pred_b = p >= 0.5
        tp = int((pred_b & t).sum())
        fp = int((pred_b & ~t).sum())
        fn = int((~pred_b & t).sum())
        details["pos_rate"] = round(float(t.mean()), 4) if t.size else None
        details["tp"], details["fp"], details["fn"] = tp, fp, fn
        if t.size and t.mean() < 0.02:
            tips.append("rare_positive")
        if tp == 0 and fn > 0:
            tips.append("all_onsets_missed")
        if fp > 2 * max(tp, 1):
            tips.append("over_firing")
    if not tips and not higher_better:
        tips.append("highest_error")
    return ";".join(tips) or "no_rule_triggered", details


def _write_both(run_id: str, payload: dict, run_dir: Optional[Path]) -> str:
    FAILURES.mkdir(parents=True, exist_ok=True)
    project_path = FAILURES / f"{run_id}.json"
    write_json(project_path, payload)
    if run_dir is not None:
        write_json(Path(run_dir) / "failure_cases.json", payload)
    return str(project_path)


def dump_failure_cases(run_id: str, per_seq: Sequence[dict], metrics: dict,
                       main_metric: str, higher_better: bool = True, k: int = 10,
                       run_dir=None, extra: Optional[dict] = None) -> dict:
    """Write the worst-k / best-k sequences for the main metric. Never raises."""
    extra = dict(extra or {})
    payload: Dict[str, Any] = {
        "run_id": run_id, "generated_at": now_str(), "split": extra.get("split", "test"),
        "main_metric": main_metric, "higher_is_better": bool(higher_better),
        "n_sequences": len(per_seq or []), "global": dict(metrics or {}),
        "head": None, "worst": [], "best": [], "diagnosis_counts": {},
        "extra": extra,
    }
    if not per_seq:
        payload["global"] = dict(metrics or {})
        _write_both(run_id, payload, run_dir)
        return payload
    head = extra.get("head") or next(iter(per_seq[0]["heads"]))
    kind = str(per_seq[0]["heads"].get(head, {}).get("kind", "ce"))
    payload["head"] = head
    payload["kind"] = kind
    values = per_seq_main_values(per_seq, head, main_metric)
    by_id = {rec["seq_id"]: rec for rec in per_seq}
    Ts = [float(rec.get("T") or 0) for rec in per_seq]
    t_p10, t_p90 = _pct(Ts, 10), _pct(Ts, 90)

    items = []
    for seq_id, rec in by_id.items():
        v = values.get(seq_id, float("nan"))
        items.append((seq_id, rec, float(v) if v is not None else float("nan")))

    def _sort_key(item, worst: bool):
        seq_id, _rec, v = item
        if not np.isfinite(v):
            return (2, 0.0, seq_id)              # invalid last
        sign = 1.0 if (higher_better == worst) else -1.0
        # higher_better & worst -> ascending ; lower_better & worst -> descending
        return (0, sign * v, seq_id)

    worst_items = sorted(items, key=lambda it: _sort_key(it, True))[:max(1, int(k))]
    best_items = sorted(items, key=lambda it: _sort_key(it, False))[:max(1, int(k))]

    counts: Dict[str, int] = {}
    for label, seq_items in (("worst", worst_items), ("best", best_items)):
        block = []
        for rank, (seq_id, rec, v) in enumerate(seq_items, 1):
            diag, details = _diagnose(kind, rec, head, v, higher_better, t_p10, t_p90)
            for d in diag.split(";"):
                counts[d] = counts.get(d, 0) + 1
            block.append({
                "rank": rank, "seq_id": seq_id,
                "source_id": rec.get("source_id"),
                "value": (None if not np.isfinite(v) else round(float(v), 6)),
                "T": int(rec.get("T") or 0),
                "diagnosis": diag, "details": details,
            })
        payload[label] = block
    payload["diagnosis_counts"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    payload["value_stats"] = {
        "mean": (float(np.nanmean([v for _, _, v in items])) if items else None),
        "p10": _pct([v for _, _, v in items], 10),
        "p90": _pct([v for _, _, v in items], 90),
    }
    payload["path"] = f"failure_cases/{run_id}.json"
    _write_both(run_id, payload, run_dir)
    return payload
