"""Unified evaluation engine (DESIGN §4, §4.1).

One function is the entry point:

    evaluate(model, loader, task, cfg, splits=("val","test"), robustness=None,
             light=False, return_seq=False) -> dict

* ``loader`` may be a single DataLoader (one split) or ``{split: DataLoader}``.
* The returned dict follows the unified structure of DESIGN §4 for a single
  split, or ``{split: unified, "meta": ...}`` for several splits.
* The robustness suite (dense / stride2 / stride4 / drop30 / drop50 / drop70 /
  burst / repeat / jitter / mixed / shuffled) is evaluated at INFERENCE time on
  the test split; perturbations are seeded per (seq_id, level) so repeated
  evaluation of the same checkpoint is bit-identical.
* No train split sequence can ever be evaluated: asserted per split and per
  batch.

Metric conventions (pinned by the Lead)
---------------------------------------
* classification (ce): acc, f1, f1_macro, precision, recall (macro where it
  makes sense); honours ``ignore=-1``.
* multilabel (bce): per-frame micro F1 ("f1"), macro F1 over classes,
  precision, recall, acc; plus onset-based event metrics from
  ``metrics.event_report`` (tolerance + dedup) when ``t`` is available.
* regression (mse): r2 (per-sequence then macro-averaged, over lab_mask-selected
  entries; if ``lab_mask`` is absent, non-zero target entries are the selected
  ones), mae, mse, rmse, cos (per-sequence then macro).
* per_prefix: metric on the first f of each sequence's valid steps, averaged
  over sequences; per_gap: metric bucketed by time since the target last
  changed (or by ``lab_gap``/``lab_age`` when the dataset provides it -- that is
  the recall-by-gap curve for synth tasks); rollout: held prediction at
  horizons 1/3/5/10 steps.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.common import metrics as M
from src.common.contract import move_batch
from src.train import data as tdata
from src.train.factory import head_specs
from src.train.objectives import aux_statistics

EPS = 1e-9

CLASSIF_METRICS = ("acc", "f1", "f1_macro", "precision", "recall")
REGRESS_METRICS = ("r2", "mae", "mse", "rmse", "cos")
LOWER_IS_BETTER = ("mae", "mse", "rmse", "loss", "mae_macro")
PREFIX_FRACS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
GAP_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DT_EDGES = (0.0, 0.02, 0.0334, 0.05, 0.1, 0.2, 0.5, 1.0)
ROLLOUT_HORIZONS = (1, 3, 5, 10)
GAP_KEYS = ("age", "gap", "gap_s")   # lab_age (preferred) / lab_gap / lab_gap_s
MASK_KEYS = ("mask",)           # optional label array: lab_mask (PER-SLOT, state head only)


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def higher_is_better(name: str) -> bool:
    return str(name) not in LOWER_IS_BETTER


def metric_family(name: str) -> str:
    if name in CLASSIF_METRICS:
        return "classif"
    if name in REGRESS_METRICS:
        return "regress"
    return "auto"


def pick_main_head(heads_cfg: Dict[str, Dict[str, Any]],
                   main_metric: Optional[str] = None) -> str:
    """Head that owns the main metric (family match, then config order)."""
    names = list(heads_cfg)
    if not names:
        raise ValueError("no heads configured")
    if len(names) == 1 or not main_metric:
        return names[0]
    fam = metric_family(main_metric)
    if fam != "auto":
        want = ("ce", "bce") if fam == "classif" else ("mse",)
        for n in names:
            if str(heads_cfg[n].get("kind")) in want:
                return n
    return names[0]


def _as_element_mask(m: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Align a label mask to the target's element shape.

    Handles the three layouts the datasets actually use:
      * per-element mask, same shape as target
      * per-STEP mask [T] for a trailing-dim target (broadcast)
      * per-SLOT mask [T, K] for a flattened target [T, K*V] (repeat V times),
        which is what the synthetic multi-object tasks emit (lab_mask [T,16] for
        lab_state [T, 16*16]).
    """
    m = np.asarray(m)
    shape = tuple(int(s) for s in shape)
    if shape == m.shape:
        return m.astype(bool)
    if m.ndim == 0:
        return np.full(shape, bool(m))
    # per-slot mask -> per-element mask (repeat the last axis)
    if (m.ndim == len(shape) and m.shape[:-1] == shape[:-1]
            and m.shape[-1] > 0 and shape[-1] != m.shape[-1]
            and shape[-1] % m.shape[-1] == 0):
        m = np.repeat(m, shape[-1] // m.shape[-1], axis=-1)
        if m.shape == shape:
            return m.astype(bool)
    # per-step (or lower-rank) mask -> broadcast over trailing dims
    if m.ndim < len(shape):
        try:
            return np.broadcast_to(
                m.reshape(m.shape + (1,) * (len(shape) - m.ndim)).astype(bool),
                shape).copy()
        except ValueError:
            pass
    try:
        return np.broadcast_to(m.astype(bool), shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"cannot align label mask of shape {m.shape} to target shape {shape}: "
            f"a per-slot mask needs equal leading dims and a last dim dividing "
            f"the target's last dim (got {m.shape} vs {shape})") from exc


def valid_entries(kind: str, target: np.ndarray, lab_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean element mask of supervised entries for one head.

    Pinned semantics (Lead):
      * ``ce``  : supervised iff ``target != ignore_index`` (-1).  ``lab_mask`` is
                  a PER-SLOT mask for the ``state`` head only and is IGNORED here
                  (synth emits lab_class=-1 on unsupervised steps).
      * ``bce`` : supervised iff the target is finite.  ``lab_mask`` ignored.
      * ``mse`` : ``lab_mask`` is per-SLOT: target is [.., D] flattened from K
                  slots x V dims and lab_mask is [.., K]; require D % K == 0 and
                  expand with np.repeat each slot V = D/K times.  Without a mask,
                  non-zero finite target entries are the supervised ones.
    """
    target = np.asarray(target)
    if kind == "ce":
        return (target != -1)
    if kind == "bce":
        return np.isfinite(target)
    finite = np.isfinite(target)
    if lab_mask is None:
        return (target != 0) & finite
    m = np.asarray(lab_mask)
    if m.shape == target.shape:
        v = m.astype(bool)
    elif target.ndim <= 1:
        v = np.ones(target.shape, dtype=bool)
    elif m.ndim == target.ndim - 1:
        # per-step mask -> broadcast over the trailing dim(s)
        v = np.broadcast_to(m.reshape(m.shape + (1,)), target.shape).astype(bool)
    elif m.ndim == target.ndim:
        D = int(target.shape[-1])
        K = int(m.shape[-1])
        if K <= 0 or D % K != 0:
            raise ValueError(
                f"lab_mask last dim {K} does not divide target last dim {D} "
                f"(target {target.shape}, mask {m.shape}); a per-slot mask must "
                f"describe K slots x V=D/K values")
        v = np.broadcast_to(np.repeat(m, D // K, axis=-1), target.shape).astype(bool)
    else:
        raise ValueError(
            f"cannot align lab_mask {m.shape} with target {target.shape}: "
            f"expected same rank, one rank less (per-step), or same leading dims "
            f"with a last dim dividing the target's last dim")
    return v & finite


# --------------------------------------------------------------------------- #
# per-head pooled metrics
# --------------------------------------------------------------------------- #
def _f1_from_confusion(pred: np.ndarray, tgt: np.ndarray, n_classes: int) -> Tuple[float, List[float]]:
    f1s: List[float] = []
    for c in range(int(n_classes)):
        tp = int(((pred == c) & (tgt == c)).sum())
        fp = int(((pred == c) & (tgt != c)).sum())
        fn = int(((pred != c) & (tgt == c)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        f1s.append(2 * prec * rec / (prec + rec + EPS))
    return (float(np.mean(f1s)) if f1s else float("nan")), f1s


def pooled_metrics(kind: str, pred: np.ndarray, target: np.ndarray,
                   valid: np.ndarray, n_classes: Optional[int] = None) -> Dict[str, float]:
    """Pooled (micro over all entries) metrics for one head."""
    pred = np.asarray(pred)
    target = np.asarray(target)
    valid = np.asarray(valid, dtype=bool)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0}
    if kind == "ce":
        p = pred[valid].astype(np.int64)
        t = target[valid].astype(np.int64)
        nc = int(n_classes or (max(int(t.max()), int(p.max())) + 1))
        acc = float((p == t).mean())
        f1m, f1s = _f1_from_confusion(p, t, nc)
        # precision / recall macro over classes present
        precs, recs = [], []
        for c in range(nc):
            tp = int(((p == c) & (t == c)).sum())
            fp = int(((p == c) & (t != c)).sum())
            fn = int(((p != c) & (t == c)).sum())
            if tp + fn + fp == 0:
                continue
            precs.append(tp / (tp + fp + EPS))
            recs.append(tp / (tp + fn + EPS))
        return {"acc": acc, "f1": f1m, "f1_macro": f1m,
                "precision": float(np.mean(precs)) if precs else float("nan"),
                "recall": float(np.mean(recs)) if recs else float("nan"),
                "n": n}

    if kind == "bce":
        p = pred[valid].astype(np.float64)
        t = (target[valid].astype(np.float64) >= 0.5).astype(np.float64)
        pb = (p >= 0.5).astype(np.float64)
        tp = float((pb * t).sum())
        fp = float((pb * (1 - t)).sum())
        fn = float(((1 - pb) * t).sum())
        tn = float(((1 - pb) * (1 - t)).sum())
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        f1 = 2 * prec * rec / (prec + rec + EPS)
        prec0 = tn / (tn + fn + EPS)
        rec0 = tn / (tn + fp + EPS)
        f1_0 = 2 * prec0 * rec0 / (prec0 + rec0 + EPS)
        return {"f1": float(f1), "f1_macro": float(0.5 * (f1 + f1_0)),
                "precision": float(prec), "recall": float(rec), "acc": float((pb == t).mean()),
                "n": n}

    # regression
    p = pred[valid].astype(np.float64)
    t = target[valid].astype(np.float64)
    err = p - t
    mse = float((err ** 2).mean())
    return {"mae": float(np.abs(err).mean()), "mse": mse,
            "rmse": float(math.sqrt(mse)), "n": n}


def _regression_seq_stats(pred: np.ndarray, target: np.ndarray,
                          valid: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred)[valid].astype(np.float64)
    t = np.asarray(target)[valid].astype(np.float64)
    out = {"mae": float("nan"), "mse": float("nan"), "r2": float("nan"),
           "cos": float("nan"), "n": int(p.size)}
    if p.size == 0:
        return out
    err = p - t
    out["mae"] = float(np.abs(err).mean())
    out["mse"] = float((err ** 2).mean())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    if p.size >= 2 and ss_tot > 1e-12:
        out["r2"] = float(1.0 - float((err ** 2).sum()) / ss_tot)
    np_, nt = float(np.linalg.norm(p)), float(np.linalg.norm(t))
    if np_ > 1e-12 and nt > 1e-12:
        out["cos"] = float(np.dot(p, t) / (np_ * nt))
    return out


def _seq_metric(kind: str, metric: str, pred: np.ndarray, target: np.ndarray,
                valid: np.ndarray, n_classes: Optional[int] = None) -> float:
    """One sequence's value of one metric (nan when not defined)."""
    valid = np.asarray(valid, dtype=bool)
    if valid.sum() == 0:
        return float("nan")
    if kind == "ce":
        p = np.asarray(pred)[valid].astype(np.int64)
        t = np.asarray(target)[valid].astype(np.int64)
        if metric == "acc":
            return float((p == t).mean())
        if metric in ("f1", "f1_macro"):
            nc = int(n_classes or (max(int(t.max()), int(p.max())) + 1))
            return _f1_from_confusion(p, t, nc)[0]
        return pooled_metrics(kind, pred, target, valid, n_classes).get(metric, float("nan"))
    if kind == "bce":
        return pooled_metrics(kind, pred, target, valid).get(metric, float("nan"))
    st = _regression_seq_stats(pred, target, valid)
    return float(st.get(metric, float("nan")))


def _pooled_over_seqs(per_seq: Sequence[dict], head: str, kind: str,
                      n_classes: Optional[int] = None) -> Dict[str, float]:
    preds, targets, valid = [], [], []
    for rec in per_seq:
        h = rec["heads"].get(head)
        if h is None:
            continue
        preds.append(np.asarray(h["pred"]).reshape(-1))
        targets.append(np.asarray(h["target"]).reshape(-1))
        valid.append(np.asarray(h["valid"]).reshape(-1))
    if not preds:
        return {"n": 0}
    return pooled_metrics(kind, np.concatenate(preds), np.concatenate(targets),
                          np.concatenate(valid), n_classes)


def _macro_over_seqs(per_seq: Sequence[dict], head: str, kind: str, metric: str,
                     n_classes: Optional[int] = None) -> float:
    vals = []
    for rec in per_seq:
        h = rec["heads"].get(head)
        if h is None:
            continue
        v = _seq_metric(kind, metric, h["pred"], h["target"], h["valid"], n_classes)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _per_seq_values(per_seq: Sequence[dict], head: str, kind: str, metric: str,
                    n_classes: Optional[int] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rec in per_seq:
        h = rec["heads"].get(head)
        if h is None:
            continue
        v = _seq_metric(kind, metric, h["pred"], h["target"], h["valid"], n_classes)
        if np.isfinite(v):
            out[rec["seq_id"]] = float(v)
    return out


# --------------------------------------------------------------------------- #
# heads metrics container
# --------------------------------------------------------------------------- #
def head_metrics(per_seq: Sequence[dict], heads_cfg: Dict[str, Dict[str, Any]],
                 main_head: str, main_metric: str) -> Dict[str, Any]:
    """Merged metrics dict (primary head plain, secondary heads prefixed)."""
    metrics: Dict[str, Any] = {}
    per_seq_main: Dict[str, float] = {}
    per_source: Dict[str, Dict[str, float]] = {}
    for name, spec in heads_cfg.items():
        kind = str(spec.get("kind", "ce"))
        n_classes = spec.get("n_classes")
        pooled = _pooled_over_seqs(per_seq, name, kind, n_classes)
        if kind == "mse":
            pooled["r2"] = _macro_over_seqs(per_seq, name, kind, "r2")
            pooled["cos"] = _macro_over_seqs(per_seq, name, kind, "cos")
            pooled["mae_macro"] = _macro_over_seqs(per_seq, name, kind, "mae")
        pfx = "" if name == main_head else f"{name}_"
        for k, v in pooled.items():
            if k == "n":
                metrics[pfx + "n"] = int(v)
            else:
                metrics[pfx + k] = float(v) if v is not None else None
        # per-sequence value of this head's own main-family metric
        head_metric = main_metric if name == main_head else (
            "acc" if kind == "ce" else ("f1" if kind == "bce" else "r2"))
        vals = _per_seq_values(per_seq, name, kind, head_metric, n_classes)
        if name == main_head:
            per_seq_main = vals
        # per-source aggregation (for paired bootstrap over source_id)
        if name == main_head:
            by_src: Dict[str, List[float]] = {}
            for rec in per_seq:
                v = vals.get(rec["seq_id"])
                if v is None or not np.isfinite(v):
                    continue
                by_src.setdefault(rec["source_id"], []).append(v)
            per_source[head_metric] = {s: float(np.mean(v)) for s, v in by_src.items()}
    metrics["main_metric"] = main_metric
    metrics["main_head"] = main_head
    return {"metrics": metrics, "per_seq_main": per_seq_main,
            "per_source": per_source}


# --------------------------------------------------------------------------- #
# curves: prefix / gap / dt / rollout
# --------------------------------------------------------------------------- #
def prefix_curve(per_seq: Sequence[dict], head: str, kind: str, metric: str,
                 n_classes: Optional[int] = None,
                 fracs: Sequence[float] = PREFIX_FRACS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for f in fracs:
        vals = []
        for rec in per_seq:
            h = rec["heads"].get(head)
            if h is None:
                continue
            n = int(np.asarray(h["pred"]).shape[0])
            k = max(1, int(round(f * n)))
            v = _seq_metric(kind, metric, np.asarray(h["pred"])[:k],
                            np.asarray(h["target"])[:k], np.asarray(h["valid"])[:k], n_classes)
            if np.isfinite(v):
                vals.append(v)
        key = f"p@{f:.3f}".rstrip("0")
        if key.endswith("."):
            key += "0"          # "p@1.0", "p@0.5", "p@0.25" (DESIGN §4.1 example)
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _fallback_age(rec: dict, head: str) -> Optional[np.ndarray]:
    """Time since the target last changed (per step) -- the pre-lab_age fallback."""
    h = rec["heads"].get(head)
    if h is None:
        return None
    v = np.asarray(h["valid"])
    tgt = np.asarray(h["target"])
    n = int(np.asarray(rec["t"]).shape[0])
    t = np.asarray(rec["t"], dtype=np.float64)
    flat_valid = v.reshape(n, -1).any(1) if v.ndim > 1 else v.astype(bool)
    flat_tgt = tgt.reshape(n, -1) if tgt.ndim > 1 else tgt.reshape(n, 1)
    gap = np.full(n, np.nan)
    last_change = None
    prev = None
    for i in range(n):
        cur = flat_tgt[i]
        changed = prev is None or not np.array_equal(cur, prev)
        if changed:
            last_change = t[i]
        prev = cur
        if flat_valid[i] and last_change is not None:
            gap[i] = max(0.0, t[i] - last_change)
    return gap


def _seq_age_elements(rec: dict, head: str, kind: str) -> Optional[np.ndarray]:
    """Age axis for one sequence, aligned to the head's element shape.

    Datasets emit ``lab_age`` EITHER per step [T] OR per slot [T, K] (synth
    emits [T,16] for a [T,256] flattened state target).  For the mse head the
    per-slot age is expanded exactly like the per-slot lab_mask (repeat V=D/K
    times), so the recall-vs-occlusion-age table is per element.  For ce/bce the
    per-step age is the age at the asked slot (lab_mask==1) or the slot mean.
    Falls back to age-since-label-change when the dataset provides no age.
    """
    h = rec["heads"].get(head)
    if h is None:
        return None
    tgt = np.asarray(h["target"])
    n = int(tgt.shape[0])
    raw = None
    for k in GAP_KEYS:
        g = (rec.get("gap_arrays") or {}).get(k)
        if g is not None:
            raw = np.asarray(g, dtype=np.float64)
            break
    if raw is None:
        fb = _fallback_age(rec, head)
        if fb is None:
            return None
        raw = fb
    if raw.ndim == 0:
        return np.full(tgt.shape, float(raw))
    raw = raw[:n]
    if kind == "mse":
        if raw.ndim == 0 or raw.shape == tgt.shape:
            return raw if raw.ndim else np.full(tgt.shape, float(raw))
        if raw.ndim == 1:
            return np.broadcast_to(raw.reshape((n,) + (1,) * (tgt.ndim - 1)),
                                   tgt.shape).astype(np.float64)
        # per-slot age [T,K] -> per-element [T,D]
        if raw.ndim == tgt.ndim and raw.shape[:-1] == tgt.shape[:-1]:
            D, K = int(tgt.shape[-1]), int(raw.shape[-1])
            if K > 0 and D % K == 0 and D != K:
                raw = np.repeat(raw, D // K, axis=-1)
            return np.broadcast_to(raw, tgt.shape).astype(np.float64)
        # last resort: mean age over trailing dims, broadcast
        return np.broadcast_to(raw.reshape((n,) + (1,) * (tgt.ndim - 1)),
                               tgt.shape).astype(np.float64)
    # ce / bce: reduce to one age per step
    if raw.ndim == 1:
        return raw.astype(np.float64)
    lm = h.get("lab_mask")
    if lm is not None:
        lm2 = np.asarray(lm)[:n]
        while lm2.ndim < raw.ndim:
            lm2 = lm2[..., None]
        asked = np.argmax((np.broadcast_to(lm2, raw.shape) > 0).astype(np.int64), axis=-1)
        taken = np.take_along_axis(raw, asked[..., None], axis=-1)[..., 0]
        any_asked = (np.asarray(lm2).reshape(n, -1) > 0).any(-1)
        mean_age = np.nanmean(raw.reshape(n, -1), axis=-1)
        return np.where(any_asked, taken, mean_age)
    return np.nanmean(raw.reshape(n, -1), axis=-1)


def _seq_gap(rec: dict, kind: str) -> Optional[np.ndarray]:
    """Backward-compatible shim: per-step age for the sequence's main head."""
    head = next(iter(rec["heads"].keys()), None)
    if head is None:
        return None
    age = _seq_age_elements(rec, head, kind)
    return None if age is None else np.asarray(age, dtype=np.float64).reshape(-1)[:len(rec["t"])]


def _macro_regression_extras(slices: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                             ) -> Dict[str, float]:
    """Per-sequence r2 / cos / mae on the given (pred, target, valid) slices, then macro."""
    r2s: List[float] = []
    coss: List[float] = []
    maes: List[float] = []
    for pred, tgt, val in slices:
        p = np.asarray(pred)[np.asarray(val, dtype=bool)].reshape(-1).astype(np.float64)
        t = np.asarray(tgt)[np.asarray(val, dtype=bool)].reshape(-1).astype(np.float64)
        if p.size == 0:
            continue
        maes.append(float(np.abs(p - t).mean()))
        if p.size < 2:
            continue
        ss_tot = float(((t - t.mean()) ** 2).sum())
        if ss_tot > 1e-12:
            r2s.append(float(1.0 - float(((p - t) ** 2).sum()) / ss_tot))
        nv, nt = float(np.linalg.norm(p)), float(np.linalg.norm(t))
        if nv > 1e-12 and nt > 1e-12:
            coss.append(float(np.dot(p, t) / (nv * nt)))
    return {
        "r2": float(np.mean(r2s)) if r2s else float("nan"),
        "cos": float(np.mean(coss)) if coss else float("nan"),
        "mae_macro": float(np.mean(maes)) if maes else float("nan"),
    }


def _pooled_plus_macro(kind: str, slices: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                       n_classes: Optional[int] = None) -> Dict[str, float]:
    """Pooled metrics over concatenated slices + per-sequence macro extras (mse)."""
    if not slices:
        return {"n": 0}
    p = np.concatenate([np.asarray(s[0]).reshape(-1) for s in slices])
    t = np.concatenate([np.asarray(s[1]).reshape(-1) for s in slices])
    v = np.concatenate([np.asarray(s[2]).reshape(-1) for s in slices])
    d = pooled_metrics(kind, p, t, v, n_classes)
    if kind == "mse":
        d.update(_macro_regression_extras(slices))
    return d


def gap_curve(per_seq: Sequence[dict], head: str, kind: str, metric: str,
              n_classes: Optional[int] = None,
              edges: Sequence[float] = GAP_EDGES) -> Dict[str, Dict[str, Any]]:
    """Metric bucketed by the age axis -> ``{bucket: {metric, n, ...}}``.

    The age axis is ``lab_age`` (seconds since the last information-bearing
    observation of the supervised target) when the dataset provides it -- that IS
    the recall-vs-occlusion-duration table.  Otherwise it falls back to
    time-since-label-change (long-horizon retention).
    """
    buckets: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
        [] for _ in range(len(edges))]
    for rec in per_seq:
        h = rec["heads"].get(head)
        if h is None:
            continue
        pred = np.asarray(h["pred"])
        tgt = np.asarray(h["target"])
        val = np.asarray(h["valid"], dtype=bool)
        age = _seq_age_elements(rec, head, kind)
        if age is None:
            continue
        try:
            age_b = np.broadcast_to(age, tgt.shape)
        except ValueError:
            continue
        fa = age_b.reshape(-1)
        fp, ft, fv = pred.reshape(-1), tgt.reshape(-1), val.reshape(-1)
        for b, lo in enumerate(edges):
            hi = edges[b + 1] if b + 1 < len(edges) else None
            sel = fv & np.isfinite(fa) & (fa >= lo - 1e-9)
            if hi is not None:
                sel &= fa < hi - 1e-9
            if not sel.any():
                continue
            buckets[b].append((fp[sel], ft[sel], fv[sel]))
    out: Dict[str, Dict[str, Any]] = {}
    for b, lo in enumerate(edges):
        hi = edges[b + 1] if b + 1 < len(edges) else None
        key = f"{lo:g}-{hi:g}" if hi is not None else f">={lo:g}"
        if not buckets[b]:
            out[key] = {"n": 0, metric: float("nan")}
            continue
        d = _pooled_plus_macro(kind, buckets[b], n_classes)
        entry: Dict[str, Any] = {"n": int(d.get("n", 0))}
        for k, v in d.items():
            if k == "n":
                continue
            entry[k] = float(v) if v is not None else None
        entry[metric] = float(d.get(metric, float("nan")))
        out[key] = entry
    return out


def dt_curve(per_seq: Sequence[dict], head: str, kind: str, metric: str,
             n_classes: Optional[int] = None,
             edges: Sequence[float] = DT_EDGES) -> Dict[str, Dict[str, Any]]:
    """Metric bucketed by the input dt (irregular-sampling robustness)."""
    out: Dict[str, Dict[str, Any]] = {}
    for b, lo in enumerate(edges):
        hi = edges[b + 1] if b + 1 < len(edges) else None
        slices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rec in per_seq:
            h = rec["heads"].get(head)
            dt = rec.get("dt")
            if h is None or dt is None:
                continue
            n = min(int(np.asarray(h["target"]).shape[0]),
                    int(np.asarray(dt).reshape(-1).shape[0]))
            if n <= 0:
                continue
            tgt_s = np.asarray(h["target"])[:n]
            pred_s = np.asarray(h["pred"])[:n]
            val_s = np.asarray(h["valid"], dtype=bool)[:n]
            dt_v = np.asarray(dt, dtype=np.float64).reshape(-1)[:n]
            try:
                dt_b = np.broadcast_to(
                    dt_v.reshape((n,) + (1,) * (tgt_s.ndim - 1)), tgt_s.shape)
            except ValueError:
                continue
            fd = dt_b.reshape(-1)
            fp, ft, fv = pred_s.reshape(-1), tgt_s.reshape(-1), val_s.reshape(-1)
            sel = fv & np.isfinite(fd) & (fd >= lo - 1e-9)
            if hi is not None:
                sel &= fd < hi - 1e-9
            if not sel.any():
                continue
            slices.append((fp[sel], ft[sel], fv[sel]))
        key = f"{lo:g}-{hi:g}" if hi is not None else f">={lo:g}"
        if not slices:
            out[key] = {"n": 0, metric: float("nan")}
            continue
        d = _pooled_plus_macro(kind, slices, n_classes)
        entry: Dict[str, Any] = {"n": int(d.get("n", 0))}
        for k, v in d.items():
            if k == "n":
                continue
            entry[k] = float(v) if v is not None else None
        entry[metric] = float(d.get(metric, float("nan")))
        out[key] = entry
    return out


def rollout_curve(per_seq: Sequence[dict], head: str, kind: str, metric: str,
                  n_classes: Optional[int] = None,
                  horizons: Sequence[int] = ROLLOUT_HORIZONS
                  ) -> Optional[Dict[str, Dict[str, float]]]:
    """Held-prediction rollout: predict at t, score against the target at t+h.

    This is the multi-step metric the Lead pinned: a model that only interpolates
    one step degrades with h, whereas a model carrying state holds its score.
    For synth tasks the t+h targets are exactly the hidden-phase labels, i.e. the
    latent-overshooting / hidden-gap rollout (horizons 1/3/5/10).
    """
    out: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        slices: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for rec in per_seq:
            hd = rec["heads"].get(head)
            if hd is None:
                continue
            pred = np.asarray(hd["pred"])
            tgt = np.asarray(hd["target"])
            val = np.asarray(hd["valid"], dtype=bool)
            n = int(pred.shape[0])
            if n <= h:
                continue
            slices.append((pred[:n - h], tgt[h:n], val[h:n]))
        key = str(int(h))
        if not slices:
            out[key] = {"n": 0, metric: float("nan")}
            continue
        d = _pooled_plus_macro(kind, slices, n_classes)
        entry: Dict[str, Any] = {"n": int(d.get("n", 0))}
        for k, v in d.items():
            if k == "n":
                continue
            entry[k] = float(v) if v is not None else None
        entry[metric] = float(d.get(metric, float("nan")))
        out[key] = {k: float(v) for k, v in entry.items() if v is not None}
        out[key]["n"] = int(d.get("n", 0))
    return out or None


# --------------------------------------------------------------------------- #
# crosstalk / aux stats
# --------------------------------------------------------------------------- #
def slot_crosstalk(route: torch.Tensor, mask: torch.Tensor) -> Optional[np.ndarray]:
    """N x N slot co-activation matrix from route [B,T,K,N].

    Off-diagonal = two slots written/used at the same step; diagonal = usage.
    A memory with isolated slots has small off-diagonal mass.
    """
    if route is None or route.dim() != 4:
        return None
    R = route.detach().float()
    R = R / R.sum(-1, keepdim=True).clamp_min(1e-6)
    R = R.mean(-2)                                          # [B,T,N]
    m = mask.detach().float().unsqueeze(-1)                 # [B,T,1]
    denom = float(m.sum().clamp_min(1.0))
    Mx = torch.einsum("btn,btm->nm", R * m, R) / denom      # [N,N]
    return Mx.cpu().numpy()


def _merge_aux_stats(acc: Dict[str, float], add: Dict[str, float], w: float) -> None:
    for k, v in add.items():
        if v is None:
            continue
        acc[k] = acc.get(k, 0.0) + float(v) * w


# --------------------------------------------------------------------------- #
# performance measurement
# --------------------------------------------------------------------------- #
def measure_perf(model, cfg: dict, feature_dim: int, tok_dim: int = 0, n_tok: int = 0,
                 roi_dim: int = 0, n_roi: int = 0) -> Dict[str, Any]:
    """params / FLOPs / latency for a single forward step (Lead: time 128 steps)."""
    ecfg = dict((cfg or {}).get("eval") or {})
    out: Dict[str, Any] = {"params": int(model.n_params)}
    device = next(model.parameters()).device
    T = int(ecfg.get("perf_T", 128))
    try:
        batch = tdata.make_dummy_batch(1, T, feature_dim, tok_dim=tok_dim, n_tok=n_tok,
                                       roi_dim=roi_dim, n_roi=n_roi, device=device)
    except Exception:                                        # pragma: no cover
        return out
    model.eval()
    if ecfg.get("flops", True):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with torch.no_grad(), FlopCounterMode(display=False) as fc:
                model(batch)
            total = int(fc.get_total_flops())
            out["flops"] = total
            out["flops_per_step"] = total // max(1, T)
        except Exception:
            out["flops"] = None
            out["flops_per_step"] = None
    if ecfg.get("latency", True):
        cuda = device.type == "cuda"
        reps = int(ecfg.get("latency_reps", 3))
        try:
            with torch.no_grad():
                for _ in range(max(1, int(ecfg.get("latency_warmup", 2)))):
                    model(batch)
                if cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(max(1, reps)):
                        model(batch)
                if cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            out["latency_ms_per_step"] = 1000.0 * dt / (max(1, reps) * T)
            out["latency_reps"] = reps
        except Exception:
            out["latency_ms_per_step"] = None
    return out


# --------------------------------------------------------------------------- #
# split evaluation
# --------------------------------------------------------------------------- #
def _split_of_loader(loader) -> str:
    ds = getattr(loader, "dataset", None)
    recs = getattr(ds, "records", None)
    if not recs:
        return "unknown"
    splits = {r.split for r in recs}
    if len(splits) != 1:
        raise ValueError(f"loader mixes splits {splits}; build one loader per split")
    return next(iter(splits))


def _assert_batch_no_train(batch: dict, where: str = "eval") -> None:
    for meta in batch.get("meta") or []:
        if meta.get("split") == "train":
            raise RuntimeError(
                f"{where}: refusing to evaluate TRAIN sequence {meta.get('seq_id')!r} "
                f"(data leakage is not allowed)")


def eval_split(model, loader, cfg: dict, task: Optional[str] = None,
               light: bool = False) -> Tuple[dict, List[dict]]:
    """Evaluate one checkpoint on one split -> (unified dict, per-sequence data)."""
    device = next(model.parameters()).device
    cfg = dict(cfg or {})
    heads_cfg = head_specs(cfg)
    main_metric = str(cfg.get("main_metric") or "")
    if not main_metric:
        main_metric = ("acc" if str(next(iter(heads_cfg.values()))["kind"]) == "ce"
                       else "f1" if str(next(iter(heads_cfg.values()))["kind"]) == "bce"
                       else "r2")
    main_head = pick_main_head(heads_cfg, main_metric)
    main_kind = str(heads_cfg[main_head].get("kind", "ce"))
    split = _split_of_loader(loader)

    model.eval()
    per_seq: List[dict] = []
    aux_acc: Dict[str, float] = {}
    aux_w = 0.0
    route_sum: Optional[np.ndarray] = None
    n_steps = 0
    n_seq_seen = 0

    for batch in loader:
        _assert_batch_no_train(batch, where=f"eval[{split}]")
        batch = move_batch(batch, device)
        with torch.no_grad():
            logits, aux, _out = model(batch)
        mask_np = batch["mask"].detach().cpu().numpy()
        B = int(mask_np.shape[0])
        n_seq_seen += B
        n_steps += int(mask_np.sum())
        stats = aux_statistics(aux, batch["mask"])
        _merge_aux_stats(aux_acc, stats, float(B))
        aux_w += float(B)
        if not light:
            route = None
            try:
                from src.train.objectives import aux_get
                route = aux_get(aux, "route")
            except Exception:
                route = None
            if route is not None:
                Mx = slot_crosstalk(route, batch["mask"])
                if Mx is not None:
                    route_sum = Mx if route_sum is None else route_sum + Mx

        t_np = batch["t"].detach().cpu().numpy()
        dt_np = batch["dt"].detach().cpu().numpy()
        labels = batch.get("labels") or {}
        lab_mask = {}
        for k in MASK_KEYS:
            v = labels.get(k)
            if v is not None:
                lab_mask[k] = v.detach().cpu().numpy()
        gap_arrays = {}
        for k in GAP_KEYS:
            v = labels.get(k)
            if v is not None:
                gap_arrays[k] = v.detach().cpu().numpy()
        preds, targets = {}, {}
        for name, spec in heads_cfg.items():
            lg = logits.get(name)
            if lg is None:
                continue
            kind = str(spec.get("kind", "ce"))
            if kind == "ce":
                preds[name] = lg.argmax(-1).detach().cpu().numpy().astype(np.int64)
            elif kind == "bce":
                preds[name] = torch.sigmoid(lg.float()).detach().cpu().numpy()
            else:
                preds[name] = lg.float().detach().cpu().numpy()
            tgt = labels.get(name)
            targets[name] = (tgt.detach().cpu().numpy() if tgt is not None else None)

        metas = batch.get("meta") or [{} for _ in range(B)]
        for i in range(B):
            n = int(mask_np[i].sum())
            rec = {
                "seq_id": metas[i].get("seq_id", f"{split}_{n_seq_seen - B + i}"),
                "source_id": metas[i].get("source_id", metas[i].get("seq_id", "")),
                "split": metas[i].get("split", split),
                "T": n,
                "t": np.asarray(t_np[i][:n], dtype=np.float64),
                "dt": np.asarray(dt_np[i][:n], dtype=np.float64),
                "heads": {},
                "gap_arrays": {k: np.asarray(v[i][:n]) for k, v in gap_arrays.items()},
            }
            for name, spec in heads_cfg.items():
                if name not in preds or targets[name] is None:
                    continue
                kind = str(spec.get("kind", "ce"))
                pred = np.asarray(preds[name][i][:n])
                target = np.asarray(targets[name][i][:n])
                lm = lab_mask.get("mask")
                lm_i = lm[i][:n] if lm is not None else None
                valid = valid_entries(kind, target, lm_i)
                rec["heads"][name] = {"kind": kind, "pred": pred, "target": target,
                                      "valid": valid, "lab_mask": lm_i}
            rec["main_metric"] = main_metric
            per_seq.append(rec)

    hm = head_metrics(per_seq, heads_cfg, main_head, main_metric)
    metrics = dict(hm["metrics"])
    n_classes = heads_cfg[main_head].get("n_classes")
    per_prefix = prefix_curve(per_seq, main_head, main_kind, main_metric, n_classes)
    out: Dict[str, Any] = {
        "task": task or cfg.get("task"),
        "split": split,
        "metrics": metrics,
        "main_head": main_head,
        "main_metric": main_metric,
        "main_value": metrics.get(main_metric),
        "higher_is_better": higher_is_better(main_metric),
        "n_seqs": len(per_seq),
        "n_steps": int(n_steps),
        "per_prefix": per_prefix,
        "per_gap": None,
        "per_dt": None,
        "rollout": None,
        "crosstalk_matrix": None,
        "aux_stats": ({k: float(v / aux_w) for k, v in aux_acc.items()} if aux_w > 0 else {}),
        "per_source": hm["per_source"],
        "per_seq": hm["per_seq_main"],
        "params": int(model.n_params),
        "flops": None,
        "flops_per_step": None,
        "latency_ms_per_step": None,
    }
    if not light:
        out["per_gap"] = gap_curve(per_seq, main_head, main_kind, main_metric, n_classes)
        out["per_dt"] = dt_curve(per_seq, main_head, main_kind, main_metric, n_classes)
        out["rollout"] = rollout_curve(per_seq, main_head, main_kind, main_metric, n_classes)
        if route_sum is not None:
            out["crosstalk_matrix"] = route_sum.tolist()
        # event-onset metrics when a bce head exists and t is available
        if main_kind == "bce":
            out["event"] = _event_metrics(per_seq, main_head)
        # NOTE: feature_dim here is the RAW `feat` width; StreamModel.prepare()
        # applies the shared token fusion, so the dummy batch must carry tokens
        # too (tok_dim/n_tok) and the core then sees the resolved in_dim.
        raw_dim = int(cfg.get("_feature_dim")
                      or getattr(getattr(loader, "dataset", None), "feature_dim", 0) or 1)
        perf = measure_perf(model, cfg, feature_dim=raw_dim,
                            tok_dim=int(cfg.get("_tok_dim", 0) or 0),
                            n_tok=int(cfg.get("_n_tok", 0) or 0),
                            roi_dim=int(cfg.get("_roi_dim", 0) or 0),
                            n_roi=int(cfg.get("_n_roi", 0) or 0))
        out["params"] = int(perf.get("params", out["params"]))
        for k in ("flops", "flops_per_step", "latency_ms_per_step"):
            out[k] = perf.get(k)
    return out, per_seq


def batch_to_per_seq(batch: dict, logits: Dict[str, torch.Tensor],
                     heads_cfg: Dict[str, Dict[str, Any]]) -> List[dict]:
    """Turn one batch's logits into per-sequence records (shared with probes)."""
    mask_np = batch["mask"].detach().cpu().numpy()
    B = int(mask_np.shape[0])
    t_np = batch["t"].detach().cpu().numpy()
    dt_np = batch["dt"].detach().cpu().numpy()
    labels = batch.get("labels") or {}
    lab_mask = {k: labels[k].detach().cpu().numpy() for k in MASK_KEYS if labels.get(k) is not None}
    gap_arrays = {k: labels[k].detach().cpu().numpy() for k in GAP_KEYS if labels.get(k) is not None}
    preds, targets = {}, {}
    for name, spec in heads_cfg.items():
        lg = logits.get(name)
        if lg is None:
            continue
        kind = str(spec.get("kind", "ce"))
        if kind == "ce":
            preds[name] = lg.argmax(-1).detach().cpu().numpy().astype(np.int64)
        elif kind == "bce":
            preds[name] = torch.sigmoid(lg.float()).detach().cpu().numpy()
        else:
            preds[name] = lg.float().detach().cpu().numpy()
        tgt = labels.get(name)
        targets[name] = (tgt.detach().cpu().numpy() if tgt is not None else None)
    metas = batch.get("meta") or [{} for _ in range(B)]
    out: List[dict] = []
    for i in range(B):
        n = int(mask_np[i].sum())
        rec = {
            "seq_id": metas[i].get("seq_id", f"seq{i}"),
            "source_id": metas[i].get("source_id", metas[i].get("seq_id", "")),
            "split": metas[i].get("split", ""),
            "T": n,
            "t": np.asarray(t_np[i][:n], dtype=np.float64),
            "dt": np.asarray(dt_np[i][:n], dtype=np.float64),
            "heads": {},
            "gap_arrays": {k: np.asarray(v[i][:n]) for k, v in gap_arrays.items()},
        }
        for name, spec in heads_cfg.items():
            if name not in preds or targets[name] is None:
                continue
            kind = str(spec.get("kind", "ce"))
            pred = np.asarray(preds[name][i][:n])
            target = np.asarray(targets[name][i][:n])
            lm = lab_mask.get("mask")
            lm_i = lm[i][:n] if lm is not None else None
            rec["heads"][name] = {"kind": kind, "pred": pred, "target": target,
                                  "valid": valid_entries(kind, target, lm_i),
                                  "lab_mask": lm_i}
        out.append(rec)
    return out


def per_seq_main_values(per_seq: Sequence[dict], head: Optional[str] = None,
                        metric: Optional[str] = None) -> Dict[str, float]:
    """{seq_id: value} of one head/metric (public helper for failure cases)."""
    if not per_seq:
        return {}
    head = head or next(iter(per_seq[0]["heads"]))
    kind = str(per_seq[0]["heads"][head].get("kind", "ce"))
    metric = metric or ("acc" if kind == "ce" else ("f1" if kind == "bce" else "r2"))
    return _per_seq_values(per_seq, head, kind, metric)


def _event_metrics(per_seq: Sequence[dict], head: str) -> Dict[str, float]:
    """Onset-based event metrics reusing src.common.metrics.event_report."""
    reports = []
    for rec in per_seq:
        h = rec["heads"].get(head)
        if h is None:
            continue
        score = np.asarray(h["pred"], dtype=np.float64)
        target = np.asarray(h["target"], dtype=np.float64)
        t = np.asarray(rec["t"], dtype=np.float64)
        n = min(score.shape[0], target.shape[0], t.shape[0])
        if n < 2:
            continue
        score, target, t = score[:n], target[:n], t[:n]
        if score.ndim > 1:
            score = score.max(-1)
        if target.ndim > 1:
            target = target.max(-1)
        try:
            reports.append(M.event_report(score, target, t))
        except Exception:
            continue
    if not reports:
        return {}
    agg = M.aggregate_event_reports(reports)
    return {f"event_{k}": float(v) for k, v in agg.items()
            if isinstance(v, (int, float)) and k in
            ("precision", "recall", "f1", "mean_delay", "dup_rate", "false_per_min", "tp", "fp", "fn")}


# --------------------------------------------------------------------------- #
# robustness suite
# --------------------------------------------------------------------------- #
def resolve_robustness(cfg: dict, robustness=None) -> List[str]:
    """Normalise the robustness request to an ordered level list."""
    if robustness is False:
        return []
    if robustness is True:
        return list(tdata.ROBUST_LEVELS)
    if isinstance(robustness, (list, tuple)):
        levels = [str(x) for x in robustness]
    else:
        ecfg = dict((cfg or {}).get("eval") or {})
        req = ecfg.get("robustness", None)
        if req is False or req is None:
            return []
        if req is True:
            return list(tdata.ROBUST_LEVELS)
        levels = [str(x) for x in req]
    seen, out = set(), []
    for lv in ["dense"] + levels:
        if lv in seen:
            continue
        seen.add(lv)
        out.append(lv)
    return out


def _robust_metrics(result: dict) -> Dict[str, float]:
    """The subset of metrics the sweep and reports read."""
    return {k: v for k, v in (result.get("metrics") or {}).items()
            if isinstance(v, (int, float)) and v is not None}


def run_robustness(model, loader, task: Optional[str], cfg: dict,
                   levels: Optional[Sequence[str]] = None,
                   max_seqs: int = 0) -> Dict[str, Dict[str, float]]:
    """Evaluate the SAME checkpoint under every robustness level (inference only)."""
    cfg = dict(cfg or {})
    levels = list(levels) if levels else resolve_robustness(cfg)
    if not levels:
        return {}
    ds = getattr(loader, "dataset", None)
    recs = list(getattr(ds, "records", []) or [])
    if not recs:
        return {}
    tdata.assert_no_train(recs, where="robustness suite")
    ecfg = dict(cfg.get("eval") or {})
    cap = int(max_seqs or ecfg.get("robust_max_seqs", 0) or 0)
    if cap > 0:
        recs = tdata.subsample(recs, cap)
    feature_dim = int(getattr(ds, "feature_dim", 0) or cfg.get("_feature_dim", 1))
    cfg2 = dict(cfg)
    cfg2["_feature_dim"] = feature_dim
    out: Dict[str, Dict[str, float]] = {}
    for level in levels:
        if level == "dense":
            continue
        rl = tdata.make_eval_loader(cfg2, recs, level, feature_dim=feature_dim,
                                    load_tok=bool(getattr(ds, "load_tok", False)),
                                    load_roi=bool(getattr(ds, "load_roi", False)))
        res, _ = eval_split(model, rl, cfg2, task, light=True)
        out[level] = _robust_metrics(res)
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def _normalize_loader(loader, splits) -> Dict[str, Any]:
    if isinstance(loader, dict):
        loaders = {str(k): v for k, v in loader.items() if v is not None}
    else:
        loaders = {_split_of_loader(loader): loader}
    order = [str(s) for s in (splits or list(loaders.keys()))]
    order = [s for s in order if s in loaders]
    if not order:
        raise ValueError(f"no requested split in loaders (requested={splits}, "
                         f"available={list(loaders)})")
    return {s: loaders[s] for s in order}


def evaluate(model, loader, task: Optional[str] = None, cfg: Optional[dict] = None,
             splits: Sequence[str] = ("val", "test"), robustness=None,
             light: bool = False, return_seq: bool = False):
    """Unified evaluation (DESIGN §4).

    Parameters
    ----------
    model : nn.Module   constructed via src.train.factory.build_model
    loader : DataLoader | {split: DataLoader}
    task : str          "action" | "event" | "state" | "track" | "synth"
    cfg : dict          the run config (main_metric, eval.*, heads, ...)
    splits : tuple      which splits to evaluate (train is refused)
    robustness : None|bool|list   None -> cfg["eval"]["robustness"]
    light : bool        skip robustness/curves/perf (used during training)
    return_seq : bool   also return per-sequence records (for failure cases)

    Returns
    -------
    dict  unified dict for one split, or ``{split: unified, "meta": ...}``
    tuple (dict, {split: per_seq}) when ``return_seq=True``
    """
    cfg = dict(cfg or {})
    loaders = _normalize_loader(loader, splits)
    results: Dict[str, dict] = {}
    seqs: Dict[str, List[dict]] = {}
    for split, ld in loaders.items():
        res, per_seq = eval_split(model, ld, cfg, task, light=light)
        results[split] = res
        seqs[split] = per_seq

    rob_split = "test" if "test" in results else list(results)[-1]
    levels = [] if light else resolve_robustness(cfg, robustness)
    if levels:
        rob = run_robustness(model, loaders[rob_split], task, cfg, levels,
                             max_seqs=int((cfg.get("eval") or {}).get("robust_max_seqs", 0) or 0))
        rob["dense"] = _robust_metrics(results[rob_split])
        main_m = results[rob_split]["main_metric"]
        dense_v = rob["dense"].get(main_m)
        drop = {}
        for lv, mm in rob.items():
            v = mm.get(main_m)
            drop[lv] = (float(dense_v - v) if (dense_v is not None and v is not None) else None)
        results[rob_split]["robustness"] = rob
        results[rob_split]["robust_drop"] = drop
        results[rob_split]["robust_split"] = rob_split

    meta = {
        "task": task or cfg.get("task"),
        "splits": list(results),
        "robust_split": rob_split if levels else None,
        "levels": levels,
        "main_metric": results[list(results)[0]]["main_metric"],
        "higher_is_better": results[list(results)[0]]["higher_is_better"],
        "n_seqs": {s: results[s]["n_seqs"] for s in results},
    }
    if len(results) == 1:
        out = results[list(results)[0]]
        out["meta"] = meta
    else:
        out = dict(results)
        out["meta"] = meta
    if return_seq:
        return out, seqs
    return out


def deterministic_view(result: dict) -> dict:
    """The bit-reproducible part of an evaluation result (timings excluded)."""
    keys = ("task", "split", "metrics", "per_prefix", "per_gap", "per_dt", "rollout",
            "crosstalk_matrix", "aux_stats", "per_source", "per_seq", "n_seqs",
            "n_steps", "event")
    if "meta" in result or any(k in result for k in ("val", "test")):
        out = {}
        for k, v in result.items():
            if isinstance(v, dict) and k != "meta":
                out[k] = deterministic_view(v)
        return out
    return {k: result.get(k) for k in keys if k in result}
