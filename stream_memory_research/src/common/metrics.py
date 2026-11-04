"""Metrics + statistics for the stream-memory research project.

All metric functions accept numpy arrays and an optional boolean mask.
Nothing here touches files.  Every function is pure.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-9


# --------------------------------------------------------------------------- #
# classification (per-frame / per-step)
# --------------------------------------------------------------------------- #
def masked_accuracy(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None,
                    ignore: int = -1) -> float:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if mask is None:
        mask = np.ones(target.shape, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)
    ok = mask & (target != ignore)
    if ok.sum() == 0:
        return float("nan")
    return float((pred[ok] == target[ok]).mean())


def classification_report(pred: np.ndarray, target: np.ndarray,
                          mask: Optional[np.ndarray] = None, ignore: int = -1,
                          n_classes: Optional[int] = None) -> Dict[str, float]:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if mask is None:
        mask = np.ones(target.shape, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)
    ok = mask & (target != ignore)
    if ok.sum() == 0:
        return {"acc": float("nan"), "f1_macro": float("nan"), "n": 0}
    p, t = pred[ok], target[ok]
    if n_classes is None:
        n_classes = int(max(t.max(), p.max())) + 1
    f1s = []
    for c in range(n_classes):
        tp = int(((p == c) & (t == c)).sum())
        fp = int(((p == c) & (t != c)).sum())
        fn = int(((p != c) & (t == c)).sum())
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        f1s.append(2 * prec * rec / (prec + rec + EPS))
    return {
        "acc": float((p == t).mean()),
        "f1_macro": float(np.mean(f1s)) if f1s else float("nan"),
        "n": int(ok.sum()),
    }


def prefix_accuracy(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                    n_points: int = 8, ignore: int = -1) -> Dict[str, float]:
    """Accuracy restricted to the last prefix fraction (causal online evaluation).

    Returns {"p@0.25": acc_on_first_25pct_of_valid_steps, ...} using *prefix means*:
    for each sequence the prefix fraction f of its valid steps is scored, then
    averaged across sequences.
    """
    B, T = target.shape
    fracs = np.linspace(1.0 / n_points, 1.0, n_points)
    out: Dict[str, float] = {}
    accs = {f: [] for f in fracs}
    n_v = np.asarray(mask.sum(1))
    for b in range(B):
        if n_v[b] < 2:
            continue
        valid_idx = np.flatnonzero(mask[b])
        for f in fracs:
            k = max(1, int(round(f * valid_idx.size)))
            idx = valid_idx[:k]
            t = target[b][idx]
            p = pred[b][idx]
            ok = t != ignore
            if ok.sum() == 0:
                continue
            accs[f].append(float((p[ok] == t[ok]).mean()))
    for f in fracs:
        out[f"p@{f:.2f}".rstrip("0").rstrip(".")] = float(np.mean(accs[f])) if accs[f] else float("nan")
    return out


# --------------------------------------------------------------------------- #
# event detection (onset-based, tolerance window, dedup)
# --------------------------------------------------------------------------- #
def detect_onsets(score: np.ndarray, t: np.ndarray, thr: float = 0.5,
                  min_gap: float = 0.2) -> np.ndarray:
    """Convert a per-step event score into discrete onset times."""
    s = np.asarray(score, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    on = s >= thr
    if not on.any():
        return np.zeros(0)
    edges = np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))
    times = tt[edges]
    if times.size > 1:
        keep = [0]
        for i in range(1, times.size):
            if times[i] - times[keep[-1]] >= min_gap:
                keep.append(i)
        times = times[keep]
    return times


def event_report(score: np.ndarray, target: np.ndarray, t: np.ndarray,
                 thr: float = 0.5, tol: float = 0.5, mask: Optional[np.ndarray] = None,
                 min_gap: float = 0.2) -> Dict[str, float]:
    """Score-based event detection metrics for ONE sequence.

    onset tolerance `tol` seconds; duplicate predictions inside `min_gap` are merged.
    """
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        score, target, tt = score[m], target[m], tt[m]
    if tt.size == 0:
        return dict(tp=0, fp=0, fn=0, precision=float("nan"), recall=float("nan"),
                    f1=float("nan"), mean_delay=float("nan"), dup_rate=0.0, n_gt=0)

    gt_on = detect_onsets(target, tt, thr=0.5, min_gap=min_gap)
    pr_on = detect_onsets(score, tt, thr=thr, min_gap=min_gap)

    used = np.zeros(gt_on.size, dtype=bool)
    tp, delays, dup = 0, [], 0
    for p in pr_on:
        if gt_on.size == 0:
            continue
        d = np.abs(gt_on - p)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            tp += 1
            delays.append(p - gt_on[j])
        elif d[j] <= tol and used[j]:
            dup += 1
    fp = int(pr_on.size - tp - dup)
    fn = int(gt_on.size - used.sum())
    prec = tp / (tp + fp + EPS)
    rec = tp / (gt_on.size + EPS)
    dur_min = max((tt[-1] - tt[0]) / 60.0, EPS)
    return dict(
        tp=int(tp), fp=int(fp), fn=int(fn), n_gt=int(gt_on.size),
        precision=float(prec), recall=float(rec),
        f1=float(2 * prec * rec / (prec + rec + EPS)),
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        dup_rate=float(dup / max(pr_on.size, 1)),
        false_per_min=float(fp / dur_min),
    )


def aggregate_event_reports(reports: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Micro-average event metrics over sequences."""
    if not reports:
        return {}
    tp = sum(r["tp"] for r in reports)
    fp = sum(r["fp"] for r in reports)
    fn = sum(r["fn"] for r in reports)
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    delays = [r["mean_delay"] for r in reports if np.isfinite(r.get("mean_delay", np.nan))]
    dups = [r["dup_rate"] for r in reports if np.isfinite(r.get("dup_rate", np.nan))]
    fpm = [r["false_per_min"] for r in reports if np.isfinite(r.get("false_per_min", np.nan))]
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec + EPS),
                mean_delay=float(np.mean(delays)) if delays else float("nan"),
                dup_rate=float(np.mean(dups)) if dups else float("nan"),
                false_per_min=float(np.mean(fpm)) if fpm else float("nan"))


def best_event_threshold(score: np.ndarray, target: np.ndarray, t: np.ndarray,
                         tol: float = 0.5, grid: Optional[Iterable[float]] = None) -> Dict[str, float]:
    """Sweep the score threshold on VAL and return best-F1 config."""
    if grid is None:
        grid = np.arange(0.1, 0.95, 0.05)
    best = None
    for thr in grid:
        r = event_report(score, target, t, thr=float(thr), tol=tol)
        if best is None or r["f1"] > best["f1"]:
            best = dict(r, thr=float(thr))
    return best or {"f1": float("nan"), "thr": 0.5}


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def masked_mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is None:
        mask = np.ones(target.shape[:-1] if target.ndim > pred.ndim else target.shape, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    while m.ndim < pred.ndim:
        m = m[..., None]
    ok = np.broadcast_to(m, pred.shape) & np.isfinite(target)
    if ok.sum() == 0:
        return float("nan")
    return float(np.abs(pred[ok] - target[ok]).mean())


# --------------------------------------------------------------------------- #
# tracking / memory-specific metrics
# --------------------------------------------------------------------------- #
def recall_after_gap(scores: Dict[int, np.ndarray], pred: Dict[int, np.ndarray]) -> Dict[str, float]:
    """scores/pred keyed by gap length -> arrays of correct(1/0). Reports recall per gap."""
    out = {}
    for g in sorted(scores):
        v = np.asarray(scores[g], dtype=np.float64)
        out[f"gap{g}"] = float(np.nanmean(v)) if v.size else float("nan")
    return out


def crosstalk_matrix(readouts: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """Given slot readouts before/after updating target j, measure change in slot i.

    readouts: [K, D] memory readout per slot before the update
    changed:  [K, D] after updating exactly one slot
    Returns [K, K] matrix  M[i,j] = relative change of slot i when slot j was written.
    """
    before = np.asarray(readouts, dtype=np.float64)
    after = np.asarray(changed, dtype=np.float64)
    # here `changed` should be [K_update, K, D]: update k -> new readouts
    if after.ndim == 3:
        K_up, K, D = after.shape
        M = np.zeros((K, K_up))
        denom = np.linalg.norm(before, axis=-1) + EPS
        for u in range(K_up):
            delta = np.linalg.norm(after[u] - before, axis=-1)
            M[:, u] = delta / denom
        return M
    delta = np.linalg.norm(after - before, axis=-1)
    denom = np.linalg.norm(before, axis=-1) + EPS
    return (delta / denom)[None, :]


def id_metrics(pred_ids: np.ndarray, gt_ids: np.ndarray,
               visible: np.ndarray) -> Dict[str, float]:
    """Identity preservation metrics for a per-frame per-slot id assignment.

    pred_ids, gt_ids, visible: [T, N] (gt id -1 = ignore).  Uses the best
    one-to-one matching over the whole sequence (greedy by frequency) to map
    predicted slot ids to ground-truth ids, then reports:
      id_recall  -- fraction of visible GT identities correctly recovered
      id_switch  -- number of identity changes for a matched GT track
    """
    pred_ids = np.asarray(pred_ids)
    gt_ids = np.asarray(gt_ids)
    vis = np.asarray(visible).astype(bool)
    pairs: Dict[Tuple[int, int], int] = {}
    for t in range(gt_ids.shape[0]):
        for n in range(gt_ids.shape[1]):
            if not vis[t, n] or gt_ids[t, n] < 0:
                continue
            key = (int(pred_ids[t, n]), int(gt_ids[t, n]))
            pairs[key] = pairs.get(key, 0) + 1
    # greedy one-to-one mapping, highest co-occurrence first
    mapping: Dict[int, int] = {}
    used_gt = set()
    for (p, g), c in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if p in mapping or g in used_gt:
            continue
        mapping[p] = g
        used_gt.add(g)
    correct = total = 0
    for t in range(gt_ids.shape[0]):
        for n in range(gt_ids.shape[1]):
            if not vis[t, n] or gt_ids[t, n] < 0:
                continue
            total += 1
            correct += int(mapping.get(int(pred_ids[t, n]), -99) == int(gt_ids[t, n]))
    # id switches: per GT track, count changes of assigned prediction
    switches = 0
    tracks = 0
    gt_tracks = {}
    for t in range(gt_ids.shape[0]):
        for n in range(gt_ids.shape[1]):
            if vis[t, n] and gt_ids[t, n] >= 0:
                gt_tracks.setdefault(int(gt_ids[t, n]), []).append(int(pred_ids[t, n]))
    for g, ps in gt_tracks.items():
        tracks += 1
        switches += sum(1 for a, b in zip(ps[:-1], ps[1:]) if a != b)
    return {
        "id_recall": correct / (total + EPS),
        "id_switch": float(switches),
        "id_switch_per_track": float(switches / max(tracks, 1)),
        "n_gt_tracks": tracks,
    }


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), std=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(1)
    return dict(
        mean=float(x.mean()), std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
        lo=float(np.quantile(means, alpha / 2)), hi=float(np.quantile(means, 1 - alpha / 2)),
        n=int(x.size),
    )


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Mean(a-b) with a bootstrap CI.  a and b must be aligned (same seeds/units)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(diff=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(1)
    # two-sided bootstrap p value for diff != 0
    p = float(2 * min((means <= 0).mean(), (means >= 0).mean()))
    return dict(diff=float(d.mean()), lo=float(np.quantile(means, alpha / 2)),
                hi=float(np.quantile(means, 1 - alpha / 2)), p=p, n=int(d.size))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + EPS))


def relative_gain(new: float, base: float) -> float:
    if not np.isfinite(new) or not np.isfinite(base) or abs(base) < EPS:
        return float("nan")
    return float((new - base) / abs(base))


def summarize_by_seed(runs: Sequence[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed metric dicts into mean/std/ci."""
    out = {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in runs], dtype=np.float64)
        if np.all(np.isnan(vals)):
            out[k] = dict(mean=float("nan"), std=float("nan"), lo=float("nan"),
                          hi=float("nan"), n=0)
        else:
            out[k] = bootstrap_ci(vals)
    return out
