"""Model-agnostic probes for the controlled synthetic memory benchmark.

All probes take *predictions* + *ground-truth npz dictionaries* and never look
at a model's internal state.  That keeps them fair for MLP / GRU / CfC / Mamba /
DeltaNet / DSM: every baseline is judged on its per-step read-out only
(DESIGN §1.6 "串扰在预测上测").

Prediction dict accepted by every probe
---------------------------------------
    {"state": array [T, 256],          # flattened 16 slots x 16 dims
     "class": array [T] or [T, 16],    # optional; ints, or logits (argmaxed)
     "event": array [T] or [T, 1]}     # optional; event probability / logit

`state` is the only required entry; probes that need class/event fall back to
decoding the state (`layout.decode_code`) or score nan when unavailable.

Task -> main metric (Lead's uniform head set)
---------------------------------------------
recall_single / recall_multi / overwrite / change_detect / capacity_scan -> cos
vanish_reappear                                                         -> acc
event_seq                                                               -> f1

Reference oracles (in this module)
----------------------------------
PerfectMemoryOracle  streams the observation, writes on every fully observed
                     token, holds otherwise -> scores 1.0 on every memory metric
ZeroMemoryOracle     outputs the current observation only (no state)
                     -> scores ~0 on hidden steps, i.e. exactly chance for the
                        identity task and 0 for cosine
RandomOracle         outputs random values -> defines "chance"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..common import metrics as M
from . import layout as L
from .pattern import (PATTERN_TASKS, is_pattern_task, pattern_class_from_state,
                      pattern_n_classes)

EPS = 1e-8
ROLLOUT_STEPS: Tuple[int, ...] = (1, 3, 5, 10, 20, 40)
GAP_EDGES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
POST_EVENT_STEPS: Tuple[int, ...] = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _to_np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def exists_mask(truth: Dict[str, np.ndarray]) -> np.ndarray:
    """[T, KMAX] bool: slot k exists in this episode."""
    T = int(truth["t"].shape[0])
    K = int(np.asarray(truth["meta_k"]).reshape(-1)[0])
    out = np.zeros((T, L.KMAX), dtype=bool)
    out[:, :max(0, min(K, L.KMAX))] = True
    return out


def observed_mask(truth: Dict[str, np.ndarray]) -> np.ndarray:
    """[T, KMAX] bool: slot k's token is fully observed at this step."""
    tok = np.asarray(truth["tok"], dtype=np.float32)
    vis = tok[..., L.V] > 0.5
    id_seen = np.abs(tok[..., :L.VC]).sum(-1) > 0.5   # identity dims present
    return vis & id_seen


def visible_mask(truth: Dict[str, np.ndarray]) -> np.ndarray:
    """[T, KMAX] bool: slot k is detected (token present), even if not identifiable."""
    return np.asarray(truth["tok"], dtype=np.float32)[..., L.V] > 0.5


def asked_mask(truth: Dict[str, np.ndarray]) -> np.ndarray:
    """[T, KMAX] bool: the memory-only evaluation mask (lab_mask)."""
    return np.asarray(truth["lab_mask"], dtype=np.float32) > 0.5


def truth_state(truth: Dict[str, np.ndarray]) -> np.ndarray:
    return L.unflatten_state(np.asarray(truth["lab_state"], dtype=np.float32))


def pred_state(pred: Dict[str, np.ndarray], T: int) -> np.ndarray:
    st = _to_np(pred.get("state"))
    if st is None:
        raise KeyError("prediction dict must contain 'state' [T, 256]")
    st = np.asarray(st, dtype=np.float32).reshape(int(T), L.KMAX * L.V)
    return L.unflatten_state(st)


def pred_class(truth: Dict[str, np.ndarray], pred: Dict[str, np.ndarray],
               in_state: np.ndarray) -> np.ndarray:
    """Predicted class per step: use the class head when present, else decode state."""
    cls = _to_np(pred.get("class"))
    if cls is not None:
        cls = np.asarray(cls)
        if cls.ndim == 2:
            cls = cls.argmax(-1)
        return cls.astype(np.int64).reshape(-1)
    return L.decode_code(in_state[:, 0, :])


def pred_event(pred: Dict[str, np.ndarray], T: int) -> Optional[np.ndarray]:
    ev = _to_np(pred.get("event"))
    if ev is None:
        return None
    ev = np.asarray(ev, dtype=np.float64).reshape(int(T), -1)
    if ev.shape[1] == 1:
        return ev[:, 0]
    return ev.max(1)


def _phase_names(truth: Dict[str, np.ndarray], task: str) -> np.ndarray:
    from .phases import PHASES
    table = PHASES.get(task, {})
    ph = np.asarray(truth["meta_phase"], dtype=np.int64)
    names = np.empty(ph.shape[0], dtype=object)
    for i, v in enumerate(ph):
        names[i] = table.get(int(v), str(int(v)))
    return names


def _cos_matrix(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-(step, slot) cosine similarity; 0 where the prediction is all-zero."""
    pn = np.linalg.norm(pred, axis=-1)
    tn = np.linalg.norm(true, axis=-1)
    dot = (pred * true).sum(-1)
    denom = pn * tn
    out = np.zeros_like(dot, dtype=np.float64)
    ok = (pn > EPS) & (tn > EPS)
    out[ok] = dot[ok] / (denom[ok] + EPS)
    out[tn <= EPS] = np.nan
    return out


def _cos_vec(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar cosine similarity between two vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        return 0.0 if nb > EPS else float("nan")
    return float(np.dot(a, b) / (na * nb + EPS))


def _runs(mask2d: np.ndarray) -> List[Tuple[int, int, int]]:
    """Maximal runs of True per column: list of (start, end_exclusive, col).

    Runs that touch the first or last frame are also returned (the array is
    padded on BOTH sides with zeros before diffing).
    """
    m = np.asarray(mask2d, dtype=np.int8)
    T, K = m.shape
    if T == 0:
        return []
    zero = np.zeros((1, K), np.int8)
    padded = np.vstack([zero, m, zero])
    d = np.diff(padded, axis=0)                      # length T+1
    starts = np.argwhere(d == 1)
    ends = np.argwhere(d == -1)
    runs: List[Tuple[int, int, int]] = []
    by_col: Dict[int, List[int]] = {}
    by_col_end: Dict[int, List[int]] = {}
    for st, c in starts:
        by_col.setdefault(int(c), []).append(int(st))
    for en, c in ends:
        by_col_end.setdefault(int(c), []).append(int(en))
    for c, ss in by_col.items():
        for st, en in zip(ss, by_col_end.get(c, [])):
            runs.append((st, en, c))
    runs.sort()
    return runs


def _masked_cos_values(truth: Dict[str, np.ndarray], pred: Dict[str, np.ndarray],
                       task: str) -> Dict[str, Any]:
    T = int(truth["t"].shape[0])
    pst = pred_state(pred, T)
    tst = truth_state(truth)
    cos = _cos_matrix(pst, tst)
    mask = asked_mask(truth) & exists_mask(truth)
    vis = visible_mask(truth)
    vals = cos[mask]
    vals = vals[np.isfinite(vals)]
    hidden = mask & ~vis
    vals_h = cos[hidden]
    vals_h = vals_h[np.isfinite(vals_h)]
    return {"cos": vals, "cos_hidden": vals_h, "mask": mask, "cos_full": cos,
            "pred": pst, "true": tst, "vis": vis}


def _mean(a) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


# --------------------------------------------------------------------------- #
# 1) main metric: cosine on lab_state
# --------------------------------------------------------------------------- #
def masked_cosine(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """Masked cosine on lab_state (the main metric of the recall tasks)."""
    all_v, hid_v = [], []
    by_phase: Dict[str, List[float]] = {}
    for s in seqs:
        d = _masked_cos_values(s.truth, s.pred, task)
        all_v.append(d["cos"])
        hid_v.append(d["cos_hidden"])
        if task:
            names = _phase_names(s.truth, task)
            cosf = d["cos_full"]
            m = d["mask"]
            for nm in np.unique(names):
                sel = m & (names == nm)[:, None]
                v = cosf[sel]
                v = v[np.isfinite(v)]
                if v.size:
                    by_phase.setdefault(str(nm), []).extend(v.tolist())
    all_c = np.concatenate(all_v) if all_v else np.zeros(0)
    hid_c = np.concatenate(hid_v) if hid_v else np.zeros(0)
    return {
        "cos": _mean(all_c),
        "cos_hidden": _mean(hid_c),
        "cos_by_phase": {k: _mean(v) for k, v in by_phase.items()},
        "n_masked": int(np.isfinite(all_c).sum()),
    }


# --------------------------------------------------------------------------- #
# 2) recall vs gap length (and rollout at fixed hidden-step offsets)
# --------------------------------------------------------------------------- #
def recall_by_gap(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """Cosine as a function of how long the value has been out of sight.

    The age of a slot's value is `t - t[last fully observed frame]`, measured in
    units of the nominal interval Δ0 and bucketed into 1x, 2x, 4x, ... 128x.
    For the eval-only `*_gap{0.1..3.2}s` datasets this is exactly the DESIGN
    §1.6 gap sweep.
    """
    buckets: Dict[int, List[float]] = {}
    overall: List[float] = []
    for s in seqs:
        d = _masked_cos_values(s.truth, s.pred, task)
        t = np.asarray(s.truth["t"], dtype=np.float64)
        dt0 = float(np.asarray(s.truth["meta_dt0"]).reshape(-1)[0])
        full = observed_mask(s.truth)
        T, K = full.shape
        if "lab_age" in s.truth:
            # exact age shipped with the dataset (same field the evaluator uses)
            age_steps = np.asarray(s.truth["lab_age"], dtype=np.float64) / dt0
        else:  # pragma: no cover - fallback for old npz
            idx = np.where(full, np.arange(T)[:, None], -1)
            last = np.maximum.accumulate(idx, axis=0)
            last_t = np.where(last >= 0, t[np.clip(last, 0, max(T - 1, 0))], np.nan)
            age_steps = np.where(np.isnan(last_t), np.inf,
                                 (t[:, None] - last_t) / dt0)
        cos = d["cos_full"]
        # step-quantise the age: timestamps are float32, so a 1-step age can
        # come out as 1.0000002 and would otherwise fall into the next bucket
        age_steps = np.round(age_steps, 3)
        m = d["mask"] & (age_steps > 0.5)
        # bucket = the SMALLEST power-of-two edge >= age (first match wins)
        up = np.full_like(age_steps, np.inf)
        for e in GAP_EDGES:
            up = np.where((age_steps <= e) & np.isinf(up), float(e), up)
        for edge in GAP_EDGES:
            sel = m & (up == float(edge))
            v = cos[sel]
            v = v[np.isfinite(v)]
            if v.size:
                buckets.setdefault(int(edge), []).extend(v.tolist())
        v = cos[d["mask"]]
        overall.extend(v[np.isfinite(v)].tolist())
    return {
        "by_gap": {f"{k}x": _mean(v) for k, v in sorted(buckets.items())},
        "by_gap_n": {f"{k}x": int(len(v)) for k, v in sorted(buckets.items())},
        "all": _mean(overall),
    }


def rollout(seqs: Sequence["ProbeSeq"], task: str = "",
            steps: Tuple[int, ...] = ROLLOUT_STEPS) -> Dict[str, Any]:
    """Score as a function of the hidden-step horizon {1,3,5,10,...}.

    For every maximal run of consecutive hidden-but-recoverable steps, score the
    prediction at the d-th hidden step.  A model that only interpolates between
    two observed frames collapses immediately; a real memory holds.
    """
    vals: Dict[int, List[float]] = {int(d): [] for d in steps}
    for s in seqs:
        d = _masked_cos_values(s.truth, s.pred, task)
        cos = d["cos_full"]
        mask = d["mask"] & exists_mask(s.truth)
        for st, en, k in _runs(mask):
            for dd in steps:
                idx = st + int(dd) - 1
                if idx < en:
                    v = cos[idx, k]
                    if np.isfinite(v):
                        vals[int(dd)].append(float(v))
    out = {f"{dd}": _mean(vals[int(dd)]) for dd in steps}
    out_n = {f"{dd}": int(len(vals[int(dd)])) for dd in steps}
    return {"rollout": out, "rollout_n": out_n}


# --------------------------------------------------------------------------- #
# 3) crosstalk matrix (measured on PREDICTIONS, per DESIGN §1.6)
# --------------------------------------------------------------------------- #
def crosstalk_matrix(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """(updated j -> read i) relative read-out movement + isolation score.

    For every recorded crosstalk protocol triple (t_prev, t_write, t_post,
    writer j, reader i):

        Delta_i = || pred_i(t_post) - pred_i(t_prev) ||  /  || true_i(t_prev) ||

    - correct behaviour: Delta_reader ~ 0 (memory of the untouched object is
      undisturbed) while Delta_writer / diag is large (the update did happen).
    - a model that never updates anything has diag ~ 0 (judged as failure).
    """
    num = np.zeros((L.KMAX, L.KMAX), dtype=np.float64)
    raw = np.zeros((L.KMAX, L.KMAX), dtype=np.float64)
    cnt = np.zeros((L.KMAX, L.KMAX), dtype=np.int64)
    valid = 0
    total = 0
    for s in seqs:
        cross = _to_np(s.truth.get("meta_cross"))
        if cross is None or cross.size == 0:
            continue
        cross = np.asarray(cross, dtype=np.float64).reshape(-1, 5)
        T = int(s.truth["t"].shape[0])
        t = np.asarray(s.truth["t"], dtype=np.float64)
        pst = pred_state(s.pred, T)
        tst = truth_state(s.truth)
        mask = asked_mask(s.truth)
        for row in cross:
            if row[0] < 0:
                continue
            j, i = int(round(row[3])), int(round(row[4]))
            if not (0 <= j < L.KMAX and 0 <= i < L.KMAX):
                continue
            total += 1
            p = int(np.abs(t - row[0]).argmin())
            q = int(np.abs(t - row[2]).argmin())
            if not (p < q):
                continue
            if not (mask[p, i] and mask[q, i]):
                # the reader is not in the memory-only window (e.g. dropped by
                # an irregular-sampling perturbation) -> skip but count
                continue
            d = float(np.linalg.norm(pst[q, i] - pst[p, i]))
            scale = float(np.linalg.norm(tst[p, i])) + EPS
            num[i, j] += d / scale
            raw[i, j] += d
            cnt[i, j] += 1
            valid += 1
    mat = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    rawm = np.where(cnt > 0, raw / np.maximum(cnt, 1), np.nan)
    diag = np.diag(mat)
    off = mat[~np.eye(L.KMAX, dtype=bool)]
    return {
        "crosstalk_matrix": [[None if not np.isfinite(v) else float(v) for v in row]
                             for row in mat],
        "crosstalk_raw": [[None if not np.isfinite(v) else float(v) for v in row]
                          for row in rawm],
        "crosstalk_diag_mean": _mean(diag),
        "crosstalk_offdiag_mean": _mean(off),
        "crosstalk_isolation": 1.0 - _mean(off) / (_mean(diag) + EPS),
        "n_cross_events": int(cnt.sum()),
        "n_cross_specs": int(total),
    }


# --------------------------------------------------------------------------- #
# 4) capacity curve
# --------------------------------------------------------------------------- #
def capacity_curve(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """Masked cosine grouped by the number of simultaneously tracked objects K.

    §1.6: effective capacity K* = largest K whose score is still >= 0.9 x the
    K=1 score.
    """
    per_k: Dict[int, List[float]] = {}
    for s in seqs:
        d = _masked_cos_values(s.truth, s.pred, task)
        K = int(np.asarray(s.truth["meta_k"]).reshape(-1)[0])
        v = d["cos"][np.isfinite(d["cos"])]
        per_k.setdefault(K, []).extend(v.tolist())
    curve = {int(k): _mean(v) for k, v in sorted(per_k.items())}
    base_k = 1 if 1 in curve else (min(curve) if curve else 1)
    base = curve.get(base_k, float("nan"))
    norm = {int(k): (float(v) / base if np.isfinite(base) and abs(base) > EPS
                     else float("nan")) for k, v in curve.items()}
    k_star = 0
    for k in sorted(curve):
        if np.isfinite(norm[k]) and norm[k] >= 0.9:
            k_star = int(k)
    return {
        "capacity_curve": {str(k): float(v) for k, v in curve.items()},
        "capacity_norm": {str(k): float(v) for k, v in norm.items()},
        "capacity_base_k": int(base_k),
        "K_star": int(k_star),
        "n_by_k": {str(k): int(len(v)) for k, v in sorted(per_k.items())},
    }


# --------------------------------------------------------------------------- #
# 5) update latency + false-update rate
# --------------------------------------------------------------------------- #
def update_latency(seqs: Sequence["ProbeSeq"], task: str = "",
                   horizons: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                   cos_thr: float = 0.9) -> Dict[str, Any]:
    """Latency (seconds / samples) from a true observed change to the read-out.

    A change is "observed" at a step where the token is fully visible and the
    true state differs from the previous step.  Latency is measured from that
    step to the first later *masked* (memory-only) step whose predicted state
    matches the new value (cos >= thr).  Perfect memory: 1 sample; a model that
    never updates: never -> update_success = 0.
    """
    lat_s: List[float] = []
    lat_n: List[float] = []
    success_flags: List[int] = []
    succ: Dict[int, List[int]] = {int(h): [] for h in horizons}
    n_events = 0
    for s in seqs:
        T = int(s.truth["t"].shape[0])
        t = np.asarray(s.truth["t"], dtype=np.float64)
        pst = pred_state(s.pred, T)
        tst = truth_state(s.truth)
        mask = asked_mask(s.truth) & exists_mask(s.truth)
        full = observed_mask(s.truth)
        if T < 2:
            continue
        known = np.abs(tst).sum(-1) > 0
        changed = np.zeros((T, L.KMAX), dtype=bool)
        changed[1:] = (known[1:] & known[:-1]
                       & np.any(tst[1:] != tst[:-1], axis=-1))
        changed &= full       # the change must be observable to be learnable
        for w, k in np.argwhere(changed):
            n_events += 1
            new = tst[w, k]
            first_found = None
            for h in range(1, int(max(horizons)) + 1):
                f = w + h
                if f >= T:
                    break
                if mask[f, k]:
                    v = _cos_vec(pst[f, k], new)
                    if np.isfinite(v) and v >= cos_thr:
                        first_found = h
                        break
            success_flags.append(int(first_found is not None))
            for h in horizons:
                succ[int(h)].append(int(first_found is not None and first_found <= h))
            if first_found is not None:
                lat_n.append(float(first_found))
                lat_s.append(float(t[w + int(first_found)] - t[w]))
    return {
        "update_latency_s": _mean(lat_s),
        "update_latency_samples": _mean(lat_n),
        "update_success": _mean(success_flags),
        "update_success_by_h": {f"{h}": _mean(succ[int(h)]) for h in horizons},
        "n_change_events": int(n_events),
    }


def false_update_rate(seqs: Sequence["ProbeSeq"], task: str = "",
                      hold_phases: Sequence[str] = ("hold",)) -> Dict[str, Any]:
    """Over-updating during a stable period (§1.6: variance ratio / total var).

    - `false_update_rate`   = mean temporal variance of the prediction during
      the stable hold, divided by the total temporal variance of the true state
      across the scored frames of the dataset.
    - `false_update_step_rate` = fraction of consecutive masked hold steps whose
      prediction moved by more than `tol`.
    """
    pred_var: List[float] = []
    step_flags: List[float] = []
    hold_cos: List[float] = []
    true_vals: List[np.ndarray] = []
    names_cache = None
    for s in seqs:
        T = int(s.truth["t"].shape[0])
        pst = pred_state(s.pred, T)
        tst = truth_state(s.truth)
        mask = asked_mask(s.truth) & exists_mask(s.truth)
        names = _phase_names(s.truth, task)
        sel = mask.copy()
        if task:
            keep = np.isin(names, list(hold_phases))
            sel = sel & keep[:, None]
        cos = _cos_matrix(pst, tst)
        v = cos[sel]
        v = v[np.isfinite(v)]
        hold_cos.extend(v.tolist())
        true_vals.append(tst[sel])
        # temporal variance inside the selected frames, per slot
        pv = []
        for k in range(L.KMAX):
            rows = np.flatnonzero(sel[:, k])
            if rows.size >= 2:
                x = pst[rows, k, :]
                pv.append(float(x.var(axis=0).mean()))
        if pv:
            pred_var.append(float(np.mean(pv)))
        # consecutive-step movement
        for k in range(L.KMAX):
            rows = np.flatnonzero(sel[:, k])
            if rows.size >= 2:
                step = np.linalg.norm(np.diff(pst[rows, k, :], axis=0), axis=-1)
                step_flags.extend((step > 0.5).astype(float).tolist())
    tv = np.concatenate(true_vals) if true_vals else np.zeros((0, L.V))
    total_var = float(tv.var(axis=0).mean()) if tv.size else float("nan")
    rate = (float(np.mean(pred_var)) / (total_var + EPS)) if pred_var else float("nan")
    return {
        "false_update_rate": rate,
        "false_update_step_rate": _mean(step_flags),
        "hold_cos": _mean(hold_cos),
        "total_state_var": total_var,
        "n_hold_steps": int(len(step_flags)),
    }


# --------------------------------------------------------------------------- #
# 6) identity accuracy (vanish_reappear)
# --------------------------------------------------------------------------- #
def identity_accuracy(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """Accuracy of the predicted identity code at the memory-only steps.

    §1.6: identity accuracy 1/3/5 steps after the object re-appears (during the
    window in which it is detected but not identifiable, so a memoryless model
    can only guess).
    """
    all_hit: List[float] = []
    absence_hit: List[float] = []
    reentry_hit: List[float] = []
    offset_hit: Dict[int, List[float]] = {1: [], 3: [], 5: []}
    n_frames = 0
    for s in seqs:
        T = int(s.truth["t"].shape[0])
        pst = pred_state(s.pred, T)
        cls_gt = np.asarray(s.truth["lab_class"], dtype=np.int64)
        cls_pr = pred_class(s.truth, s.pred, pst)
        scored = cls_gt >= 0
        if not scored.any():
            continue
        flags = np.asarray(s.truth["meta_flags"], dtype=np.int64)
        idmask = (flags & L.FLAG_IDMASK) > 0
        hit = (cls_pr == cls_gt).astype(float)
        all_hit.extend(hit[scored].tolist())
        n_frames += int(scored.sum())
        absence_hit.extend(hit[scored & ~idmask].tolist())
        reentry_hit.extend(hit[scored & idmask].tolist())
        # offsets 1/3/5 steps after the start of each identity-masked run
        for st, en, _k in _runs(idmask[:, None]):
            for off in (1, 3, 5):
                f = st + off - 1
                if f < en and scored[f]:
                    offset_hit[off].append(float(hit[f]))
    return {
        "acc": _mean(all_hit),
        "acc_absence": _mean(absence_hit),
        "acc_reentry": _mean(reentry_hit),
        "acc_reentry_1step": _mean(offset_hit[1]),
        "acc_reentry_3step": _mean(offset_hit[3]),
        "acc_reentry_5step": _mean(offset_hit[5]),
        "n_class_frames": int(n_frames),
    }


# --------------------------------------------------------------------------- #
# 7) event report (event_seq)
# --------------------------------------------------------------------------- #
def event_report(seqs: Sequence["ProbeSeq"], task: str = "",
                 thr: float = 0.5, tol: float = 0.2,
                 min_gap: float = 0.2) -> Dict[str, Any]:
    """Event F1 / onset delay / duplicate-trigger rate + post-event state score.

    §1.6 event_seq: sparse knock pulses, distractors that must NOT lock the
    target, duplicate triggers, and post-event state accuracy at 1x/2x/4x/8x.
    """
    reports = []
    post_cos: Dict[int, List[float]] = {d: [] for d in POST_EVENT_STEPS}
    disturb_cos: Dict[int, List[float]] = {d: [] for d in POST_EVENT_STEPS}
    type_hit: List[float] = []
    n_seqs_with_events = 0
    for s in seqs:
        T = int(s.truth["t"].shape[0])
        t = np.asarray(s.truth["t"], dtype=np.float64)
        target = np.asarray(s.truth["lab_event"], dtype=np.float64)
        score = pred_event(s.pred, T)
        if score is None:
            continue
        r = M.event_report(score, target, t, thr=thr, tol=tol, min_gap=min_gap)
        reports.append(r)
        pst = pred_state(s.pred, T)
        tst = truth_state(s.truth)
        cos = _cos_matrix(pst[:, 0, :], tst[:, 0, :])
        mask = asked_mask(s.truth)[:, 0]
        gt_type = np.asarray(s.truth["lab_class"], dtype=np.int64)
        cls_pr = pred_class(s.truth, s.pred, pst)
        ev_frames = np.flatnonzero(target > 0)
        if ev_frames.size:
            n_seqs_with_events += 1
        for w in ev_frames:
            typ = int(gt_type[w]) if (w < gt_type.size and gt_type[w] >= 0) else -1
            if typ >= 0:
                type_hit.append(float(cls_pr[w] == typ))
            bucket = post_cos if typ == 1 else (disturb_cos if typ == 2 else None)
            if bucket is None:
                continue
            for d in POST_EVENT_STEPS:
                f = int(w) + int(d)
                if f >= T:
                    break
                # no other event in between
                if np.any(target[int(w) + 1:f + 1] > 0):
                    break
                if not mask[f]:
                    continue
                v = cos[f]
                if np.isfinite(v):
                    bucket[d].append(float(v))
    agg = M.aggregate_event_reports(reports) if reports else {}
    return {
        "f1": float(agg.get("f1", float("nan"))),
        "event_precision": float(agg.get("precision", float("nan"))),
        "event_recall": float(agg.get("recall", float("nan"))),
        "event_mean_delay_s": float(agg.get("mean_delay", float("nan"))),
        "event_dup_rate": float(agg.get("dup_rate", float("nan"))),
        "event_false_per_min": float(agg.get("false_per_min", float("nan"))),
        "event_tp": int(agg.get("tp", 0)),
        "event_fp": int(agg.get("fp", 0)),
        "event_fn": int(agg.get("fn", 0)),
        "post_event_cos": {f"{d}x": _mean(post_cos[d]) for d in POST_EVENT_STEPS},
        "post_distractor_cos": {f"{d}x": _mean(disturb_cos[d])
                                for d in POST_EVENT_STEPS},
        "event_type_acc": _mean(type_hit),
        "n_event_seqs": int(n_seqs_with_events),
    }


# --------------------------------------------------------------------------- #
# 8) temporal-pattern report (pattern_*)
# --------------------------------------------------------------------------- #
def pattern_report(seqs: Sequence["ProbeSeq"], task: str = "") -> Dict[str, Any]:
    """Causal whole-window classification metrics for the `pattern_*` family.

    - `acc` / `balanced_acc` on `lab_mask` (evidence-complete steps only)
    - per-class accuracy + full confusion matrix
    - accuracy as a function of how many frames have passed since the last
      target pulse (does the answer survive after the event is over?)
    - pulse-detection F1 (aux task; the pulses themselves are visible)
    """
    hits: List[float] = []
    per_class_hit: Dict[int, List[float]] = {}
    conf = np.zeros((L.N_CODES, L.N_CODES), dtype=np.int64)
    n_classes = 2
    by_age: Dict[str, List[float]] = {"0-0.5s": [], "0.5-1s": [], "1-2s": [],
                                      ">2s": []}
    pulse_reports: List[Dict[str, float]] = []
    n_scored = 0
    for s in seqs:
        T = int(s.truth["t"].shape[0])
        pst = pred_state(s.pred, T)
        nc = pattern_n_classes(s.truth)
        n_classes = max(n_classes, nc)
        cls_gt = np.asarray(s.truth["lab_class"], dtype=np.int64)
        cls_pr = pred_class(s.truth, s.pred, pst)
        if _to_np(s.pred.get("class")) is None:
            # regression-only read-out: decode the one-hot from slot-0 state
            cls_pr = pattern_class_from_state(pst[:, 0, :], nc)
        scored = (np.asarray(s.truth["lab_mask"], dtype=np.float32)[:, 0] > 0.5)
        scored &= cls_gt >= 0
        if not scored.any():
            continue
        n_scored += int(scored.sum())
        hit = (cls_pr == cls_gt).astype(float)
        hits.extend(hit[scored].tolist())
        for c in range(nc):
            sel = scored & (cls_gt == c)
            if sel.any():
                per_class_hit.setdefault(c, []).extend(hit[sel].tolist())
            selp = scored & (cls_pr == c)
            gt = cls_gt[sel]
            if gt.size:
                for g in gt:
                    conf[int(c), int(g)] += 1
        # accuracy vs frames since the last target pulse
        age = np.asarray(s.truth["lab_age"], dtype=np.float64)[:, 0]
        for name, (lo, hi) in (("0-0.5s", (0.0, 0.5)), ("0.5-1s", (0.5, 1.0)),
                               ("1-2s", (1.0, 2.0)), (">2s", (2.0, 1e9))):
            sel = scored & (age >= lo) & (age < hi)
            if sel.any():
                by_age[name].extend(hit[sel].tolist())
        # pulse detection (aux): visible event pulses
        score = pred_event(s.pred, T)
        if score is not None:
            t = np.asarray(s.truth["t"], dtype=np.float64)
            target = np.asarray(s.truth["lab_event"], dtype=np.float64)
            pulse_reports.append(M.event_report(score, target, t, thr=0.5,
                                                tol=0.15, min_gap=0.12))
    acc = _mean(hits)
    per_class = {str(c): _mean(v) for c, v in sorted(per_class_hit.items())}
    bal = _mean(list(per_class.values()))
    agg = M.aggregate_event_reports(pulse_reports) if pulse_reports else {}
    return {
        "acc": acc,
        "balanced_acc": bal,
        "acc_per_class": per_class,
        "confusion": conf[:n_classes, :n_classes].tolist(),
        "acc_by_age": {k: _mean(v) for k, v in by_age.items()},
        "pulse_f1": float(agg.get("f1", float("nan"))),
        "pulse_mean_delay_s": float(agg.get("mean_delay", float("nan"))),
        "n_scored": int(n_scored),
        "chance": 1.0 / float(n_classes),
        "n_classes": int(n_classes),
    }


# --------------------------------------------------------------------------- #
# ProbeSeq container + task dispatch
# --------------------------------------------------------------------------- #
@dataclass
class ProbeSeq:
    truth: Dict[str, np.ndarray]
    pred: Dict[str, np.ndarray]

    @property
    def T(self) -> int:
        return int(self.truth["t"].shape[0])


def make_probe_seqs(truths: Sequence[Dict[str, np.ndarray]],
                    preds: Sequence[Dict[str, np.ndarray]]) -> List[ProbeSeq]:
    if len(truths) != len(preds):
        raise ValueError(f"{len(truths)} truths vs {len(preds)} predictions")
    return [ProbeSeq(truth=t, pred=p) for t, p in zip(truths, preds)]


def load_truths(paths: Sequence[str]) -> List[Dict[str, np.ndarray]]:
    from .base import load_seq
    return [load_seq(p) for p in paths]


def main_scores(task: str, seqs: Sequence[ProbeSeq]) -> Dict[str, Any]:
    """The single number the Lead's summary compares (main_metric per task)."""
    if is_pattern_task(task):
        r = pattern_report(seqs, task)
        return {"acc": r["acc"], "balanced_acc": r["balanced_acc"],
                "acc_by_age": r["acc_by_age"], "chance": r["chance"]}
    if task == "vanish_reappear":
        r = identity_accuracy(seqs, task)
        return {"acc": r["acc"], "acc_reentry": r["acc_reentry"],
                "acc_absence": r["acc_absence"]}
    if task == "event_seq":
        r = event_report(seqs, task)
        return {"f1": r["f1"], "post_event_cos": r["post_event_cos"]}
    r = masked_cosine(seqs, task)
    return {"cos": r["cos"], "cos_hidden": r["cos_hidden"]}


def probe_all(task: str, seqs: Sequence[ProbeSeq]) -> Dict[str, Any]:
    """Run every probe that applies to `task` (superset of the main metric)."""
    out: Dict[str, Any] = {}
    if is_pattern_task(task):
        out.update(pattern_report(seqs, task))
        out["main"] = main_scores(task, seqs)
        return out
    out.update(masked_cosine(seqs, task))
    out.update(recall_by_gap(seqs, task))
    out.update(rollout(seqs, task))
    out["main"] = main_scores(task, seqs)
    if task == "recall_multi":
        out.update(crosstalk_matrix(seqs, task))
    if task == "capacity_scan":
        out.update(capacity_curve(seqs, task))
    if task in ("change_detect", "overwrite"):
        out.update(false_update_rate(seqs, task, hold_phases=("hold",)))
    if task in ("change_detect", "overwrite", "recall_multi"):
        out.update(update_latency(seqs, task))
    if task == "vanish_reappear":
        out.update(identity_accuracy(seqs, task))
        out.update(false_update_rate(seqs, task, hold_phases=("absence",)))
    if task == "event_seq":
        out.update(event_report(seqs, task))
        out.update(false_update_rate(seqs, task, hold_phases=("hold",)))
    return out


# --------------------------------------------------------------------------- #
# reference oracles
# --------------------------------------------------------------------------- #
def _event_type_from_feat(feat: np.ndarray) -> np.ndarray:
    """Decode the event type purely from the observation (feat channel 5)."""
    pulse = np.asarray(feat[:, L.CH_EVENT], dtype=np.float64) > 0.5
    norm = np.asarray(feat[:, L.CH_EVENT_TYPE], dtype=np.float64)
    typ = np.rint(norm * max(L.C_EVENT - 1, 1)).astype(np.int64)
    return np.where(pulse, typ, -1)


class OracleBase:
    """Prediction interface shared by the reference oracles."""

    name = "oracle"

    def __init__(self, task: str = "") -> None:
        self.task = str(task)

    # -- interface ---------------------------------------------------------
    def predict_seq(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def predict(self, truths: Sequence[Dict[str, np.ndarray]]
                ) -> List[Dict[str, np.ndarray]]:
        return [self.predict_seq(t) for t in truths]

    # -- helpers -----------------------------------------------------------
    def _class_from_memory(self, truth: Dict[str, np.ndarray],
                           mem: np.ndarray) -> np.ndarray:
        if self.task == "event_seq":
            return _event_type_from_feat(np.asarray(truth["feat"], dtype=np.float32))
        return L.decode_code(mem[:, 0, :])


class PerfectMemoryOracle(OracleBase):
    """An unbounded, perfectly faithful memory that reads ONLY the observation.

    On every step whose token is present it writes the token value into the
    matching slot (the identity dims are left untouched when the token carries a
    zeroed identity, i.e. a detected-but-not-identifiable re-entry); on every
    other step it holds.  It streams causally, so it can only "know" what the
    observation really contains.
    """

    name = "perfect"

    def predict_seq(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feat = np.asarray(truth["feat"], dtype=np.float32)
        hist = self._memory_history(truth)          # [T, KMAX, V]
        T = int(hist.shape[0])
        out_state = hist.reshape(T, L.KMAX * L.V).astype(np.float32)
        if self.task == "event_seq":
            cls = _event_type_from_feat(feat)
        else:
            cls = L.decode_code(hist[:, 0, :])
        return {
            "state": out_state,
            "class": cls,
            "event": feat[:, L.CH_EVENT].astype(np.float32),
        }

    @staticmethod
    def _memory_history(truth: Dict[str, np.ndarray]) -> np.ndarray:
        """Replay the observation stream and record the per-step memory."""
        tok = np.asarray(truth["tok"], dtype=np.float32)
        T = int(tok.shape[0])
        mem = np.zeros((L.KMAX, L.V), dtype=np.float32)
        hist = np.zeros((T, L.KMAX, L.V), dtype=np.float32)
        for t in range(T):
            vis = tok[t, :, L.V] > 0.5
            vals = tok[t, :, :L.V]
            rows = np.flatnonzero(vis)
            if rows.size:
                id_present = np.abs(vals[rows, :L.VC]).sum(-1) > 0.5
                full_rows = rows[id_present]
                if full_rows.size:
                    mem[full_rows] = vals[full_rows]
                part_rows = rows[~id_present]
                if part_rows.size:
                    mem[part_rows, L.VC:] = vals[part_rows, L.VC:]
            hist[t] = mem
        return hist


class PerfectPatternOracle(OracleBase):
    """Unbounded faithful memory for the temporal-pattern family.

    Reads ONLY the observation: target pulses (`feat[4]`), their type
    (`feat[5]`) and their physical timestamps.  At every step it answers from
    the pulses observed **so far** (strictly causal):

    - count : number of pulses so far (1/2/3 -> class 0/1/2)
    - order : first two pulse types -> A->B vs B->A
    - rate  : interval between the first two pulses vs the mid-point between the
              fast/slow intervals (scaled by the actual sampling rate, so the
              2x/4x stretch datasets stay answerable)
    - mixed : 1 pulse / 3+ pulses / 2 fast / 2 slow
    """

    name = "perfect"

    @staticmethod
    def _kind(task: str) -> str:
        t = str(task)
        if t.startswith("pattern_order"):
            return "order"
        if t.startswith("pattern_rate"):
            return "rate"
        if t.startswith("pattern_count_rate_mixed"):
            return "mixed"
        return "count"

    def predict_seq(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feat = np.asarray(truth["feat"], dtype=np.float32)
        t = np.asarray(truth["t"], dtype=np.float64)
        T = int(t.shape[0])
        nc = pattern_n_classes(truth)
        kind = self._kind(self.task)
        pulse = feat[:, L.CH_EVENT] > 0.5
        ptype = (feat[:, L.CH_EVENT_TYPE] > 0.5).astype(np.int64)
        dt_med = float(np.median(np.diff(t))) if T > 1 else 0.1
        thr = 0.40 * (dt_med / 0.10)          # mid of 0.2 s (fast) / 0.6 s (slow)
        cls = np.zeros(T, dtype=np.int64)
        seen_t: List[float] = []
        seen_typ: List[int] = []
        for i in range(T):
            if pulse[i]:
                seen_t.append(float(t[i]))
                seen_typ.append(int(ptype[i]))
            n = len(seen_t)
            if kind == "count":
                cls[i] = min(max(n - 1, 0), max(nc - 1, 0))
            elif kind == "order":
                cls[i] = 0 if (n < 2 or seen_typ[0] == 0) else 1
            elif kind == "rate":
                gap = (seen_t[1] - seen_t[0]) if n >= 2 else None
                cls[i] = 0 if (gap is None or gap < thr) else 1
            else:  # mixed
                if n <= 1:
                    cls[i] = 0
                elif n >= 3:
                    cls[i] = 3
                else:
                    cls[i] = 1 if (seen_t[1] - seen_t[0]) < thr else 2
        state = np.zeros((T, L.KMAX * L.V), dtype=np.float32)
        state[np.arange(T), cls] = 1.0
        return {"state": state, "class": cls,
                "event": pulse.astype(np.float32)}


class ZeroMemoryOracle(OracleBase):
    """Memoryless reference: output exactly what is observed RIGHT NOW."""

    name = "memoryless"

    def predict_seq(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        tok = np.asarray(truth["tok"], dtype=np.float32)
        feat = np.asarray(truth["feat"], dtype=np.float32)
        T = int(tok.shape[0])
        if is_pattern_task(self.task):
            # nothing class-relevant is visible at the scored steps -> constant
            return {"state": np.zeros((T, L.KMAX * L.V), dtype=np.float32),
                    "class": np.zeros(T, dtype=np.int64),
                    "event": feat[:, L.CH_EVENT].astype(np.float32)}
        vals = tok[:, :, :L.V] * (tok[:, :, L.V:L.V + 1] > 0.5)
        out_state = vals.reshape(T, -1).astype(np.float32)
        if self.task == "event_seq":
            cls = _event_type_from_feat(feat)
        else:
            cls = L.decode_code(np.asarray(vals, dtype=np.float32)[:, 0, :])
        return {
            "state": out_state,
            "class": cls,
            "event": feat[:, L.CH_EVENT].astype(np.float32),
        }


class RandomOracle(OracleBase):
    """Chance reference: random state, random class, random event score."""

    name = "chance"

    def __init__(self, task: str = "", seed: int = 0) -> None:
        super().__init__(task)
        self.seed = int(seed)

    def predict_seq(self, truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        T = int(truth["t"].shape[0])
        rng = np.random.default_rng(self.seed + 7919 * T)
        state = rng.standard_normal((T, L.KMAX * L.V)).astype(np.float32)
        if is_pattern_task(self.task):
            nc = pattern_n_classes(truth)
            cls = rng.integers(0, nc, size=T)
        else:
            cls = rng.integers(0, L.N_CODES, size=T)
        ev = rng.random(T)
        return {"state": state, "class": cls, "event": ev}


def make_oracle(name: str, task: str, seed: int = 0) -> OracleBase:
    name = str(name)
    if name in ("perfect", "PerfectMemoryOracle"):
        if is_pattern_task(task):
            return PerfectPatternOracle(task)
        return PerfectMemoryOracle(task)
    if name in ("memoryless", "zero", "ZeroMemoryOracle"):
        return ZeroMemoryOracle(task)
    if name in ("chance", "random", "RandomOracle"):
        return RandomOracle(task, seed=seed)
    raise KeyError(f"unknown oracle {name!r}")


def reference_scores(truths: Sequence[Dict[str, np.ndarray]], task: str,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """perfect / memoryless / chance scores for the Lead's sanity bounds."""
    out: Dict[str, Dict[str, float]] = {}
    for name in ("perfect", "memoryless", "chance"):
        oracle = make_oracle(name, task, seed=seed)
        preds = oracle.predict(truths)
        seqs = make_probe_seqs(truths, preds)
        out[name] = main_scores(task, seqs)
    return out


def oracle_probe_report(truths: Sequence[Dict[str, np.ndarray]], task: str,
                        seed: int = 0) -> Dict[str, Any]:
    """Full probe dict for every oracle — used by the unit tests and summary."""
    out: Dict[str, Any] = {}
    for name in ("perfect", "memoryless", "chance"):
        oracle = make_oracle(name, task, seed=seed)
        seqs = make_probe_seqs(truths, oracle.predict(truths))
        out[name] = probe_all(task, seqs)
    return out


# --------------------------------------------------------------------------- #
# integration helpers for the evaluator (trainer-eval)
# --------------------------------------------------------------------------- #
def jsonable(obj):
    """Recursively convert numpy scalars / arrays to plain python objects."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def evaluate_manifest(manifest_path, preds: Dict[str, Dict[str, Any]],
                      split: str = "test", task: Optional[str] = None,
                      cap: Optional[int] = None) -> Dict[str, Any]:
    """Run every applicable probe for one split of a synthetic dataset.

    Parameters
    ----------
    manifest_path : contract.Manifest json (or G:/.../index.json entry path)
    preds : {seq_id: {"state": [T,256], "class": [T] or [T,16],
                      "event": [T] or [T,1]}}
            Predictions must be per *sequence*, aligned with the npz rows.
            Missing sequences are skipped.
    split : which manifest split to score (default "test")
    task  : override the task id (default: manifest.extra["task_id"])

    Returns a JSON-serialisable dict with the main metric plus every probe that
    applies to the task (recall_by_gap, rollout, crosstalk, capacity, latency,
    false updates, identity, event report).
    """
    from ..common.contract import Manifest
    man = Manifest.load(manifest_path)
    task = task or man.extra.get("task_id")
    recs = man.split(split) or list(man.seqs)
    recs = [r for r in recs if r.seq_id in preds]
    recs = sorted(recs, key=lambda r: r.seq_id)
    if cap:
        recs = recs[: int(cap)]
    if not recs:
        raise ValueError(f"no predictions matched split {split!r} of {man.dataset}")
    truths = [load_truths([r.path])[0] for r in recs]
    seqs = make_probe_seqs(truths, [preds[r.seq_id] for r in recs])
    metrics = probe_all(task, seqs)
    main = main_scores(task, seqs)
    return jsonable({
        "dataset": man.dataset,
        "task": task,
        "split": split,
        "main_metric": man.extra.get("main_metric"),
        "n_seqs": len(recs),
        "gap_s": man.extra.get("gap_s"),
        "capacity_k": man.extra.get("capacity_k"),
        "metrics": {**metrics, "main": main},
    })


def evaluate_paths(task: str, paths: Sequence[str],
                   preds: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Lower-level helper: explicit npz paths + aligned predictions."""
    truths = load_truths(paths)
    seqs = make_probe_seqs(truths, preds)
    return jsonable({"task": task, "n_seqs": len(paths),
                     "metrics": probe_all(task, seqs),
                     "main": main_scores(task, seqs)})
