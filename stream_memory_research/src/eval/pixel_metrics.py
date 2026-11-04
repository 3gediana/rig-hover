"""Pixel-to-event evaluation metrics (DESIGN-PIXEL-EVENT-NETWORK.md §9).

Seven pure-numpy metric functions that score a causal pixel-to-event model on
the tasks of the design spec: per-frame phase, event onset/completion
boundaries, event counts, person presence, readiness calibration, abstention
coverage and lag-stratified curves.

There is **no file IO, no torch and no sklearn** in this module: every function
accepts array-likes (list / tuple / numpy array) and returns plain Python
scalars, lists or ``np.ndarray`` values.  The module can therefore be imported
by the training loop, by ``src/eval/evaluator.py`` and by report scripts alike
and is deterministic (no randomness, no global state).

Input conventions
-----------------
* label / index arrays are flattened to 1-D and cast to ``int64``.  Integer
  valued floats (``2.0``) are accepted; ``2.5``, NaN and inf are rejected so a
  silently truncated index can never corrupt a metric.
* probability / confidence arrays are flattened to 1-D ``float64`` and must be
  finite; values within ``1e-6`` of ``[0, 1]`` are clipped into range, values
  further out raise ``ValueError`` (a common symptom of logits being passed by
  mistake).
* arrays that are scored pairwise must have the same number of elements.

Missing-value convention (uniform across the module)
----------------------------------------------------
* A metric that would average over **zero** elements is reported as ``nan`` --
  never as ``0.0`` -- so "no data" cannot be silently averaged in as a perfect
  or terrible score.  Aggregators should use ``np.nanmean``.
* Count-like outputs (``confusion``, ``bin_count``, ``n_valid``, ...) are ``0``.
* ``identity_switch_rate_placeholder`` is always ``None`` (no identity labels
  are available yet, see :func:`presence_metrics`).

Sign conventions used consistently below
----------------------------------------
* ``TP`` / ``FP`` / ``FN`` always refer to an event, class or candidate pair
  being matched.
* "latency" and "delay" are **signed**: ``pred - true``.  Positive means the
  prediction happened *later* than the ground truth ("late"), negative means it
  fired early.
* A one-to-one match means no predicted index and no true index is used twice.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

__all__ = [
    "phase_prf",
    "boundary_prf",
    "count_metrics",
    "presence_metrics",
    "calibration_metrics",
    "abstention_coverage",
    "lag_curve",
]

EPS = 1e-12
_PROB_TOL = 1e-6
_PRESENCE_STATES = (0, 1, 2, 3)  # 0=absent, 1=entering, 2=present, 3=leaving


# --------------------------------------------------------------------------- #
# input normalisation helpers
# --------------------------------------------------------------------------- #
def _to_array(x, name: str) -> np.ndarray:
    try:
        return np.asarray(x)
    except Exception as exc:  # ragged nested lists, exotic objects, ...
        raise TypeError(f"{name} must be array-like, got {type(x)!r}") from exc


def _as_1d(x, name: str) -> np.ndarray:
    arr = _to_array(x, name)
    return arr.reshape(-1)


def _as_1d_int(x, name: str) -> np.ndarray:
    """Flatten to 1-D int64, accepting integer-valued floats but nothing lossy."""
    arr = _as_1d(x, name)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    kind = arr.dtype.kind
    if kind in "iu":
        return arr.astype(np.int64, copy=False)
    if kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values (nan/inf)")
        if not np.all(arr == np.floor(arr)):
            raise ValueError(f"{name} must be integer-valued, got {arr.dtype}")
        return arr.astype(np.int64)
    raise TypeError(f"{name} must be an integer array, got dtype {arr.dtype!r}")


def _as_1d_float(x, name: str) -> np.ndarray:
    arr = _as_1d(x, name)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    kind = arr.dtype.kind
    if kind not in "iuf":
        raise TypeError(f"{name} must be a numeric array, got dtype {arr.dtype!r}")
    out = arr.astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (nan/inf)")
    return out


def _as_1d_bool(x, name: str) -> np.ndarray:
    """Flatten to 1-D bool, accepting bool / 0-1 integers / 0.0-1.0 floats."""
    arr = _as_1d(x, name)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    if arr.dtype.kind == "b":
        return arr.astype(bool)
    vals = _as_1d_float(x, name)
    if not np.all((vals == 0.0) | (vals == 1.0)):
        raise ValueError(f"{name} must contain only 0/1 (or bool) values")
    return vals.astype(bool)


def _as_int_scalar(x, name: str, minimum: int | None = None) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(x)!r}")
    out = int(x)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _require_same_length(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"{name_a} and {name_b} must have the same length, "
            f"got {a.shape[0]} vs {b.shape[0]}"
        )


def _confusion_and_f1(pred: np.ndarray, true: np.ndarray,
                      n_classes: int) -> Tuple[np.ndarray, List[float], float]:
    """Confusion matrix (rows=truth, cols=pred) + per-class F1 + macro-F1.

    Macro-F1 is averaged over classes that occur in the truth *or* in the
    prediction.  A class with frames in the truth but no prediction gets F1=0;
    a class with predictions but no true frame gets F1=0 and is counted in the
    macro average (per spec).  Classes that occur nowhere are reported as
    ``nan`` and excluded, so a perfect single-class prediction scores 1.0.
    """
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    if pred.size:
        np.add.at(conf, (true, pred), 1)
    tp = np.diag(conf).astype(np.float64)
    support = conf.sum(axis=1).astype(np.float64)   # per-class true frames
    n_pred = conf.sum(axis=0).astype(np.float64)    # per-class predicted frames
    per_class: List[float] = []
    macro_terms: List[float] = []
    for c in range(n_classes):
        if support[c] == 0.0 and n_pred[c] == 0.0:
            per_class.append(float("nan"))
            continue
        # 2TP / (2TP + FP + FN) == 2TP / (support + n_pred), denominator > 0
        f1 = 2.0 * tp[c] / (support[c] + n_pred[c])
        per_class.append(float(f1))
        macro_terms.append(float(f1))
    macro = float(np.mean(macro_terms)) if macro_terms else float("nan")
    return conf, per_class, macro


def _match_switches(true_switches: Sequence[Tuple[int, int]],
                    pred_switches: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Greedy one-to-one match of same-direction switches, nearest first.

    ``switches`` are ``(frame_index, direction)`` tuples with direction +1 for
    "starts being present" and -1 for "stops being present".  Candidate pairs
    of equal direction are sorted by ``|delay|`` (ties: true index then pred
    index) and consumed greedily, so no switch is used twice.
    """
    candidates: List[Tuple[int, int, int]] = []
    for ti, (t_frame, t_dir) in enumerate(true_switches):
        for pi, (p_frame, p_dir) in enumerate(pred_switches):
            if t_dir == p_dir:
                candidates.append((abs(p_frame - t_frame), ti, pi))
    candidates.sort()
    used_true: set[int] = set()
    used_pred: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for _delay, ti, pi in candidates:
        if ti in used_true or pi in used_pred:
            continue
        used_true.add(ti)
        used_pred.add(pi)
        pairs.append((true_switches[ti][0], pred_switches[pi][0]))
    return pairs


# --------------------------------------------------------------------------- #
# 1. per-frame phase
# --------------------------------------------------------------------------- #
def phase_prf(phase_pred, phase_true, n_classes: int,
              ignore_index: int = -1) -> Dict[str, object]:
    """Per-frame phase classification: macro-F1, per-class F1, accuracy, confusion.

    Parameters
    ----------
    phase_pred : array-like of int, shape [T]
        Predicted phase label per frame.  Values must lie in ``[0, n_classes)``
        on every frame whose ``phase_true`` is not ``ignore_index``.  Abstention
        ("no decision") is *not* representable here; score it with
        :func:`abstention_coverage` instead.
    phase_true : array-like of int, shape [T]
        Ground-truth phase label per frame, either in ``[0, n_classes)`` or
        equal to ``ignore_index`` for frames that must be skipped (ambiguous
        boundaries, padded frames, ...).
    n_classes : int
        Number of phase classes ``C`` (>= 1).  Fixes the shape of the
        confusion matrix and the length of ``per_class_f1``.
    ignore_index : int, default -1
        Value in ``phase_true`` marking a frame to exclude from *every*
        statistic (accuracy, F1, confusion).

    Returns
    -------
    dict with fields
        ``macro_f1`` : float
            Mean F1 over classes that occur in the truth or in the prediction.
            ``nan`` when no frame is scored.  A class that is predicted but has
            no true frame contributes F1 = 0 to the average (per spec).
        ``per_class_f1`` : list[float] of length ``C``
            ``f1[c] = 2*TP / (2*TP + FP + FN)`` in ``[0, 1]``; never-observed
            classes are ``nan`` and are excluded from ``macro_f1``.
        ``accuracy`` : float in [0, 1]
            Fraction of scored frames where ``phase_pred == phase_true``;
            ``nan`` when no frame is scored.
        ``confusion`` : np.ndarray[int64], shape [C, C]
            ``confusion[t, p]`` = number of scored frames with true class ``t``
            and predicted class ``p`` (rows = truth, columns = prediction).
        ``n_valid`` : int
            Number of scored frames (``T`` minus ignored frames).

    Degenerate inputs
    -----------------
    * ``T == 0`` or every frame ignored: ``macro_f1 = accuracy = nan``,
      ``per_class_f1 = [nan] * C``, ``confusion = zeros([C, C])``,
      ``n_valid = 0``.
    * single class only (e.g. all frames class 2 and correct): the class is the
      only macro term, so ``macro_f1 = 1.0`` and ``accuracy = 1.0``.
    * mismatched lengths, out-of-range predictions / truth, or a non-integer
      label raise ``ValueError``.
    """
    pred = _as_1d_int(phase_pred, "phase_pred")
    true = _as_1d_int(phase_true, "phase_true")
    _require_same_length(pred, true, "phase_pred", "phase_true")
    n = _as_int_scalar(n_classes, "n_classes", minimum=1)
    ignore = _as_int_scalar(ignore_index, "ignore_index")

    valid = true != ignore
    n_valid = int(valid.sum())
    if n_valid:
        pred_v = pred[valid]
        true_v = true[valid]
        if int(pred_v.min()) < 0 or int(pred_v.max()) >= n:
            raise ValueError(
                f"phase_pred values on scored frames must lie in [0, {n}), "
                f"got range [{int(pred_v.min())}, {int(pred_v.max())}]"
            )
        if int(true_v.min()) < 0 or int(true_v.max()) >= n:
            raise ValueError(
                f"phase_true values must be in [0, {n}) or equal ignore_index="
                f"{ignore}, got range [{int(true_v.min())}, {int(true_v.max())}]"
            )
    else:
        pred_v = np.zeros(0, dtype=np.int64)
        true_v = np.zeros(0, dtype=np.int64)

    confusion, per_class, macro = _confusion_and_f1(pred_v, true_v, n)
    accuracy = float(np.mean(pred_v == true_v)) if n_valid else float("nan")
    return {
        "macro_f1": float(macro),
        "per_class_f1": per_class,
        "accuracy": accuracy,
        "confusion": confusion,
        "n_valid": n_valid,
    }


# --------------------------------------------------------------------------- #
# 2. event boundaries (onsets / completions) with one-to-one greedy matching
# --------------------------------------------------------------------------- #
def boundary_prf(pred_idx, true_idx, tolerance_frames: int) -> Dict[str, float]:
    """Event onset/completion precision, recall, F1 and latency.

    Greedy one-to-one matching in frame-index space: enumerate every candidate
    pair ``(pred, true)`` with ``|pred - true| <= tolerance_frames``, sort the
    candidates by distance ascending (ties broken by predicted index, then true
    index), and consume them while neither index has been used yet.  A
    predicted event is therefore matched to at most one true event and vice
    versa.  Unmatched predictions are FP, unmatched truths are FN.

    Parameters
    ----------
    pred_idx : array-like of int, shape [M]
        Frame indices of predicted events (onsets or completions, one call per
        event type).  Any integer range is allowed; only distances matter.
    true_idx : array-like of int, shape [K]
        Frame indices of ground-truth events.
    tolerance_frames : int >= 0
        Maximum ``|pred - true|`` in frames for a match.  ``0`` requires exact
        frame agreement; a negative value raises ``ValueError``.

    Returns
    -------
    dict with fields
        ``precision`` : float in [0, 1]
            ``matched / M``; with ``M == 0`` it is ``1.0`` when there is also
            nothing to detect and ``0.0`` otherwise (see Degenerate inputs).
        ``recall`` : float in [0, 1]
            ``matched / K``.
        ``f1`` : float in [0, 1]
            ``2PR / (P + R)``; ``0.0`` when ``P + R == 0``.
        ``latency_mean_frames`` : float
            Mean **absolute** delay ``mean(|pred - true|)`` over matched pairs
            (``0.0`` for a perfect detector).  ``nan`` if nothing matched.
        ``latency_signed_mean`` : float
            Mean **signed** delay ``mean(pred - true)`` over matched pairs.
            Positive = predictions are late, negative = early.  ``nan`` if
            nothing matched.
        ``n_pred``, ``n_true``, ``n_matched`` : int
            Raw counts behind precision / recall.

    Degenerate inputs
    -----------------
    * ``M == 0, K == 0``: ``precision = recall = f1 = 1.0``,
      latencies ``nan`` (perfect detection of an empty event sequence).
    * ``M == 0, K > 0``: ``precision = recall = f1 = 0.0`` (all events missed).
    * ``K == 0, M > 0``: ``precision = recall = f1 = 0.0`` (all predictions are
      false alarms on a sequence without events).
    * a single matched pair gives latency equal to that pair's delay.
    """
    pred = _as_1d_int(pred_idx, "pred_idx")
    true = _as_1d_int(true_idx, "true_idx")
    tol = _as_int_scalar(tolerance_frames, "tolerance_frames", minimum=0)

    n_pred = int(pred.size)
    n_true = int(true.size)
    matched: List[Tuple[int, int, int]] = []  # (pred index, true index, |dist|)
    if n_pred and n_true:
        # range-query candidates instead of materialising an M x K matrix
        order = np.argsort(true, kind="stable")
        true_sorted = true[order]
        candidates: List[Tuple[int, int, int]] = []
        for i in range(n_pred):
            p = int(pred[i])
            lo = int(np.searchsorted(true_sorted, p - tol, side="left"))
            hi = int(np.searchsorted(true_sorted, p + tol, side="right"))
            for pos in range(lo, hi):
                j = int(order[pos])
                candidates.append((abs(p - int(true[j])), i, j))
        candidates.sort()
        used_pred: set[int] = set()
        used_true: set[int] = set()
        for dist, i, j in candidates:
            if i in used_pred or j in used_true:
                continue
            used_pred.add(i)
            used_true.add(j)
            matched.append((i, j, dist))

    n_matched = len(matched)
    if n_pred:
        precision = n_matched / n_pred
    else:
        precision = 1.0 if n_true == 0 else 0.0
    if n_true:
        recall = n_matched / n_true
    else:
        recall = 1.0 if n_pred == 0 else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    if n_matched:
        latencies = np.array([float(pred[i]) - float(true[j]) for i, j, _ in matched],
                             dtype=np.float64)
        latency_mean = float(np.mean(np.abs(latencies)))
        latency_signed = float(np.mean(latencies))
    else:
        latency_mean = float("nan")
        latency_signed = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "latency_mean_frames": latency_mean,
        "latency_signed_mean": latency_signed,
        "n_pred": n_pred,
        "n_true": n_true,
        "n_matched": n_matched,
    }


# --------------------------------------------------------------------------- #
# 3. exact count
# --------------------------------------------------------------------------- #
def count_metrics(count_pred, count_true) -> Dict[str, float]:
    """Exact-count accuracy, MAE and over/under-count rates.

    Parameters
    ----------
    count_pred : array-like of int, shape [N]
        Predicted event count per sequence / clip.  Counts are assumed to be
        non-negative, but negative integers are accepted (and compared
        numerically) so a broken head is visible rather than silently clipped.
    count_true : array-like of int, shape [N]
        Ground-truth event count per sequence / clip.

    Returns
    -------
    dict with fields
        ``exact_accuracy`` : float in [0, 1]
            Fraction of samples with ``count_pred == count_true``.
        ``mae`` : float >= 0
            Mean absolute error ``mean(|pred - true|)``.
        ``over_rate`` : float in [0, 1]
            Fraction of samples with ``count_pred > count_true``.
        ``under_rate`` : float in [0, 1]
            Fraction of samples with ``count_pred < count_true``.
        ``mean_signed_error`` : float
            ``mean(pred - true)``; positive = systematic over-counting.

    Degenerate inputs
    -----------------
    * ``N == 0``: every metric is ``nan`` (no data to average).
    * ``count_pred`` all zero: no special case -- scores follow from the
      formulas above (e.g. if every true count is also 0 then
      ``exact_accuracy = 1.0, mae = 0.0, over_rate = under_rate = 0.0,
      mean_signed_error = 0.0``).
    * mismatched lengths raise ``ValueError``; non-integer counts raise
      ``TypeError`` / ``ValueError``.
    """
    pred = _as_1d_int(count_pred, "count_pred")
    true = _as_1d_int(count_true, "count_true")
    _require_same_length(pred, true, "count_pred", "count_true")
    n = int(pred.size)
    if n == 0:
        nan = float("nan")
        return {
            "exact_accuracy": nan,
            "mae": nan,
            "over_rate": nan,
            "under_rate": nan,
            "mean_signed_error": nan,
            "n": 0,
        }
    diff = pred - true
    return {
        "exact_accuracy": float(np.mean(diff == 0)),
        "mae": float(np.mean(np.abs(diff))),
        "over_rate": float(np.mean(diff > 0)),
        "under_rate": float(np.mean(diff < 0)),
        "mean_signed_error": float(np.mean(diff)),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# 4. person presence (departure / return)
# --------------------------------------------------------------------------- #
def presence_metrics(state_pred, state_true) -> Dict[str, object]:
    """Person presence / departure / return metrics for the 4-state encoding.

    State encoding: ``0 = absent``, ``1 = entering``, ``2 = present``,
    ``3 = leaving``.  "Presence" is the binary signal ``state != 0``; the
    prediction is credited with detecting presence whenever it predicts a
    non-absent state.

    Parameters
    ----------
    state_pred : array-like of int, shape [T]
        Predicted presence state per frame; values must be in ``{0, 1, 2, 3}``.
    state_true : array-like of int, shape [T]
        Ground-truth presence state per frame; same value range.  There is no
        ignore index: an unknown frame should be excluded from both arrays by
        the caller.

    Returns
    -------
    dict with fields
        ``macro_f1`` : float
            Macro-F1 over the 4-state encoding, using the same convention as
            :func:`phase_prf` (classes observed in truth or prediction;
            predicted-but-never-true classes contribute F1 = 0; classes that
            occur nowhere are ``nan`` and excluded).  ``nan`` when ``T == 0``.
        ``presence_precision`` : float in [0, 1]
            Binary precision of ``state != 0``:
            ``TP / (TP + FP)``, where TP = both non-absent.  Convention: when
            neither truth nor prediction contains a present frame the score is
            ``1.0`` (vacuously perfect); when only one side has present frames
            the missing side contributes 0.  ``nan`` when ``T == 0``.
        ``presence_recall`` : float in [0, 1]
            Binary recall of ``state != 0``: ``TP / (TP + FN)``, same
            conventions as above.
        ``mean_switch_delay_frames`` : float
            Signed mean detection delay in frames over matched presence
            transitions, ``mean(t_pred_switch - t_true_switch)``; positive =
            the predicted switch is late.  A presence transition is an
            adjacent-frame boundary where ``(state == 0)`` flips to/from
            ``(state != 0)`` -- for 2-state sequences that use only ``0`` and
            ``2`` this is exactly the ``0 <-> 2`` transition.  Predicted and
            true transitions are matched one-to-one only within the same
            direction (both entering or both leaving), nearest first.
            ``nan`` when there is no true transition or none was matched.
        ``identity_switch_rate_placeholder`` : None
            Always ``None``.  DESIGN §9 asks for an identity-switch rate, but
            the 4-state labels carry no identity, so the slot is reserved until
            identity annotations exist.
        ``n_switch_true``, ``n_switch_matched`` : int
            Number of true presence transitions and how many were matched to a
            predicted transition (the denominator behind the delay).

    Degenerate inputs
    -----------------
    * ``T == 0``: ``macro_f1 = nan``, ``presence_precision = nan``,
      ``presence_recall = nan``, ``mean_switch_delay_frames = nan``,
      ``identity_switch_rate_placeholder = None``.
    * ``T == 1``: no transitions, delay ``nan``; the classification metrics are
      computed normally.
    * all frames absent and correctly predicted: ``macro_f1 = 1.0`` (only class
      0 is observed), ``presence_precision = presence_recall = 1.0``, delay
      ``nan``.
    * single state only (e.g. all ``2`` and correct): ``macro_f1 = 1.0``.
    * values outside ``{0, 1, 2, 3}`` or mismatched lengths raise
      ``ValueError``.
    """
    pred = _as_1d_int(state_pred, "state_pred")
    true = _as_1d_int(state_true, "state_true")
    _require_same_length(pred, true, "state_pred", "state_true")
    n_frames = int(pred.size)

    allowed = np.array(_PRESENCE_STATES, dtype=np.int64)
    if n_frames:
        for name, arr in (("state_pred", pred), ("state_true", true)):
            bad = arr[~np.isin(arr, allowed)]
            if bad.size:
                raise ValueError(
                    f"{name} values must be in {set(_PRESENCE_STATES)}, "
                    f"got {sorted(set(int(v) for v in bad))[:8]}"
                )

    confusion, per_class, macro = _confusion_and_f1(pred, true, len(_PRESENCE_STATES))

    # ---- binary presence: positive class is "not absent" ---------------- #
    true_pos = true != 0
    pred_pos = pred != 0
    tp = int(np.sum(true_pos & pred_pos))
    n_true_pos = int(np.sum(true_pos))
    n_pred_pos = int(np.sum(pred_pos))
    if n_frames == 0:
        precision = float("nan")
        recall = float("nan")
    elif n_true_pos == 0 and n_pred_pos == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = tp / n_pred_pos if n_pred_pos else 0.0
        recall = tp / n_true_pos if n_true_pos else 0.0

    # ---- presence transitions (0 <-> present, incl. entering/leaving) --- #
    def _switches(flags: np.ndarray) -> List[Tuple[int, int]]:
        if flags.size < 2:
            return []
        change = flags[1:] != flags[:-1]
        idx = np.flatnonzero(change) + 1
        return [(int(t), 1 if bool(flags[t]) else -1) for t in idx]

    true_sw = _switches(true_pos)
    pred_sw = _switches(pred_pos)
    pairs = _match_switches(true_sw, pred_sw)
    if pairs:
        delays = np.array([float(p_frame) - float(t_frame) for t_frame, p_frame in pairs],
                          dtype=np.float64)
        switch_delay = float(np.mean(delays))
    else:
        switch_delay = float("nan")

    return {
        "macro_f1": float(macro),
        "presence_precision": float(precision),
        "presence_recall": float(recall),
        "mean_switch_delay_frames": switch_delay,
        "identity_switch_rate_placeholder": None,
        "n_switch_true": len(true_sw),
        "n_switch_matched": len(pairs),
        "n_valid": n_frames,
    }


# --------------------------------------------------------------------------- #
# 5. readiness calibration
# --------------------------------------------------------------------------- #
def calibration_metrics(p, y, n_bins: int = 15) -> Dict[str, object]:
    """Brier score, expected/maximum calibration error and the reliability table.

    Binning is **equal width** over ``[0, 1]``: ``n_bins`` bins with edges
    ``bin_edges = linspace(0, 1, n_bins + 1)``.  A prediction ``p`` falls in
    bin ``i`` when ``bin_edges[i] <= p < bin_edges[i + 1]``; the outermost
    right edge is closed so ``p == 1.0`` lands in the last bin (and
    ``p == 0.0`` in the first).

    Parameters
    ----------
    p : array-like of float, shape [N]
        Predicted positive-class probability / confidence per sample.
        Values must be finite and in ``[0, 1]`` (tiny numerical overshoot up to
        ``1e-6`` is clipped; anything further out raises ``ValueError``).
    y : array-like of int or bool, shape [N]
        Ground-truth binary label per sample, in ``{0, 1}``.
    n_bins : int, default 15
        Number of equal-width bins (>= 1).

    Returns
    -------
    dict with fields
        ``brier`` : float in [0, 1]
            Mean squared error ``mean((p - y)^2)``; lower is better.
            ``nan`` when ``N == 0``.
        ``ece`` : float in [0, 1]
            Expected calibration error: ``sum over non-empty bins of
            (n_bin / N) * |acc_bin - conf_bin|``.  Empty bins are skipped
            entirely (they are never treated as a contribution of 0 to the
            sum's weight).  ``nan`` when ``N == 0``.
        ``mce`` : float in [0, 1]
            Maximum over non-empty bins of ``|acc_bin - conf_bin|``; ``nan``
            when there is no non-empty bin.
        ``reliability`` : dict
            ``bin_edges`` : list[float] of length ``n_bins + 1``
                Bin boundaries, ascending from 0.0 to 1.0.
            ``bin_count`` : list[int] of length ``n_bins``
                Number of samples per bin (sums to ``N``).
            ``bin_acc`` : list[float] of length ``n_bins``
                Mean label per bin in ``[0, 1]``; ``nan`` for an empty bin.
            ``bin_conf`` : list[float] of length ``n_bins``
                Mean prediction per bin in ``[0, 1]``; ``nan`` for an empty bin.

    Degenerate inputs
    -----------------
    * ``N == 0``: ``brier = ece = mce = nan``; ``bin_count`` all ``0``;
      ``bin_acc`` and ``bin_conf`` all ``nan``; ``bin_edges`` still has
      ``n_bins + 1`` entries.
    * ``n_bins == 1``: one bin over ``[0, 1]``; ECE equals ``|mean(y) -
      mean(p)|`` and MCE equals the same value.
    * a perfectly calibrated predictor has ``ece == mce == 0.0`` (up to float
      round-off), see the self-test invariant.
    * labels outside ``{0, 1}``, probabilities outside ``[0, 1]``, non-integer
      ``n_bins`` or mismatched lengths raise ``ValueError`` / ``TypeError``.
    """
    prob = _as_1d_float(p, "p")
    labels = _as_1d_float(y, "y")
    _require_same_length(prob, labels, "p", "y")
    nb = _as_int_scalar(n_bins, "n_bins", minimum=1)
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("y must contain only 0/1 labels")
    n = int(prob.size)
    if n:
        if np.any(prob < -_PROB_TOL) or np.any(prob > 1.0 + _PROB_TOL):
            raise ValueError(
                f"p must be probabilities in [0, 1], got range "
                f"[{float(prob.min())}, {float(prob.max())}]"
            )
        prob = np.clip(prob, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, nb + 1)
    counts = np.zeros(nb, dtype=np.int64)
    sums = np.zeros(nb, dtype=np.float64)
    pos = np.zeros(nb, dtype=np.float64)
    if n:
        # inner edges: p == 1.0 lands in the last (right-closed) bin
        bin_idx = np.digitize(prob, edges[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, nb - 1)
        counts = np.bincount(bin_idx, minlength=nb).astype(np.int64)
        sums = np.bincount(bin_idx, weights=prob, minlength=nb)
        pos = np.bincount(bin_idx, weights=labels, minlength=nb)

    denom = np.maximum(counts, 1).astype(np.float64)
    bin_conf = sums / denom
    bin_acc = pos / denom
    nonempty = counts > 0
    bin_conf = np.where(nonempty, bin_conf, np.nan)
    bin_acc = np.where(nonempty, bin_acc, np.nan)

    if n == 0 or not nonempty.any():
        brier = float("nan")
        ece = float("nan")
        mce = float("nan")
    else:
        brier = float(np.mean((prob - labels) ** 2))
        gaps = np.abs(bin_acc[nonempty] - bin_conf[nonempty])
        ece = float(np.sum((counts[nonempty].astype(np.float64) / n) * gaps))
        mce = float(np.max(gaps))

    return {
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "reliability": {
            "bin_edges": [float(e) for e in edges],
            "bin_count": [int(c) for c in counts],
            "bin_acc": [float(a) for a in bin_acc],
            "bin_conf": [float(c) for c in bin_conf],
        },
    }


# --------------------------------------------------------------------------- #
# 6. abstention / selective prediction
# --------------------------------------------------------------------------- #
def abstention_coverage(confidence, correct,
                        thresholds: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)
                        ) -> Dict[str, list]:
    """Selective-prediction coverage and accuracy at each confidence threshold.

    A sample is *accepted* when ``confidence >= threshold`` (inclusive, so a
    threshold equal to a confidence value accepts it).  Coverage is the
    accepted fraction; selective accuracy is the accuracy on accepted samples
    only.  This is the coverage/accuracy trade-off curve for abstention.

    Parameters
    ----------
    confidence : array-like of float, shape [N]
        Confidence of the readiness prediction per sample, in ``[0, 1]``
        (same validation as :func:`calibration_metrics`).
    correct : array-like of bool or 0/1, shape [N]
        Whether the prediction was correct (``1``/``True``) or not.
    thresholds : sequence of float, default ``(0.5, 0.6, 0.7, 0.8, 0.9)``
        Acceptance thresholds, reported in the given order.  Any finite values
        are allowed (outside ``[0, 1]`` just means accept-all / accept-none).

    Returns
    -------
    dict with fields
        ``thresholds`` : list[float]
            The thresholds, in the input order.
        ``coverage`` : list[float] of length ``len(thresholds)``
            ``n_accepted / N`` at each threshold; ``0.0`` when ``N == 0``.
        ``selective_accuracy`` : list[float]
            ``mean(correct[accepted])`` at each threshold; ``nan`` when that
            threshold accepts no sample (including ``N == 0``).

    Degenerate inputs
    -----------------
    * ``N == 0``: every ``coverage`` is ``0.0`` and every
      ``selective_accuracy`` is ``nan``.
    * a threshold above every confidence: ``coverage = 0.0``,
      ``selective_accuracy = nan`` for that threshold.
    * an empty ``thresholds`` sequence returns three empty lists.
    * confidence outside ``[0, 1]``, ``correct`` values outside ``{0, 1}`` or
      mismatched lengths raise ``ValueError``.
    """
    conf = _as_1d_float(confidence, "confidence")
    hits = _as_1d_bool(correct, "correct")
    _require_same_length(conf, hits, "confidence", "correct")
    n = int(conf.size)
    if n:
        if np.any(conf < -_PROB_TOL) or np.any(conf > 1.0 + _PROB_TOL):
            raise ValueError(
                f"confidence must be in [0, 1], got range "
                f"[{float(conf.min())}, {float(conf.max())}]"
            )
        conf = np.clip(conf, 0.0, 1.0)

    thr_list = [float(t) for t in thresholds]
    coverage: List[float] = []
    selective: List[float] = []
    for thr in thr_list:
        if n == 0:
            coverage.append(0.0)
            selective.append(float("nan"))
            continue
        accepted = conf >= thr
        n_acc = int(accepted.sum())
        coverage.append(n_acc / n)
        selective.append(float(np.mean(hits[accepted])) if n_acc else float("nan"))
    return {
        "thresholds": thr_list,
        "coverage": coverage,
        "selective_accuracy": selective,
    }


# --------------------------------------------------------------------------- #
# 7. lag-stratified curves
# --------------------------------------------------------------------------- #
def lag_curve(score_by_lag: Mapping[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Sort a ``{lag: score}`` mapping into aligned ``(lags, values)`` arrays.

    Used for the "report performance by ..." requirement of DESIGN §9: build a
    dict keyed by occlusion length / frame rate / hand pixel size / pause
    length and get back two aligned arrays ready for plotting or table
    generation.

    Parameters
    ----------
    score_by_lag : mapping of int -> float
        ``lag`` (must be an ``int`` / ``np.integer``, not a bool) is the x-axis
        bucket, e.g. occlusion length in frames; ``score`` is any finite metric
        value, e.g. macro-F1.  Values are returned unchanged, just cast to
        ``float64``.

    Returns
    -------
    (lags, values) : tuple of np.ndarray
        ``lags`` : int64 array of shape [L], sorted ascending.
        ``values`` : float64 array of shape [L], the matching input values.
        An empty mapping returns two empty arrays.

    Degenerate inputs
    -----------------
    * ``{}`` -> ``(array([], dtype=int64), array([], dtype=float64))``.
    * a single key -> arrays of length 1.
    * non-integer keys (including bools and strings) raise ``TypeError``;
      non-numeric values raise ``TypeError`` / ``ValueError``.
    """
    if not isinstance(score_by_lag, Mapping):
        raise TypeError(
            f"score_by_lag must be a mapping, got {type(score_by_lag)!r}"
        )
    items: List[Tuple[int, float]] = []
    for key, value in score_by_lag.items():
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
            raise TypeError(
                f"score_by_lag keys must be integers, got {type(key)!r}"
            )
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"score_by_lag[{key!r}] must be a number, got {type(value)!r}"
            ) from exc
        items.append((int(key), val))
    items.sort(key=lambda kv: kv[0])
    lags = np.array([k for k, _ in items], dtype=np.int64)
    values = np.array([v for _, v in items], dtype=np.float64)
    return lags, values

# --------------------------------------------------------------------------- #
# average precision (the standard metric for sparse multi-label event detection;
# a single-threshold macro-F1 is dominated by the operating point, and this
# project should not report only that)
# --------------------------------------------------------------------------- #
def average_precision(score, label) -> float:
    """All-points-interpolated AP for one class.

    `score [N]` float in [0,1], `label [N]` in {0,1}.  Returns 0.0 when the
    class has no positive label (AP undefined), and 0.0 when no score is finite.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    label = np.asarray(label).ravel().astype(np.int64)
    if score.size == 0 or label.sum() == 0:
        return 0.0
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if score.size == 0 or label.sum() == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    label = label[order]
    score = score[order]
    tp = np.cumsum(label)
    fp = np.cumsum(1 - label)
    # one point per distinct score
    keep = np.r_[np.where(np.diff(score))[0], score.size - 1]
    tp, fp = tp[keep], fp[keep]
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(1.0, float(label.sum()))
    # all-points interpolation: precision envelope from the right
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rec = np.r_[0.0, rec]
    prec = np.r_[prec[0], prec]
    return float(np.sum(np.diff(rec) * prec[1:]))


def mean_average_precision(probs, labels, present=None) -> dict:
    """Macro mAP over the classes that actually occur.

    `probs` / `labels`: sequences of `[T, C]` arrays (one per video).
    """
    probs = [np.asarray(p, dtype=np.float64) for p in probs]
    labels = [np.asarray(y) for y in labels]
    if not probs:
        return {"map": float("nan"), "ap_per_class": []}
    C = probs[0].shape[1]
    P = np.concatenate([p for p in probs], 0)
    Y = np.concatenate([y for y in labels], 0)
    aps = [average_precision(P[:, j], (Y[:, j] > 0.5).astype(np.int64)) for j in range(C)]
    if present is None:
        present = np.array([(Y[:, j] > 0.5).any() for j in range(C)])
    vals = [aps[j] for j in range(C) if present[j]]
    return {"map": float(np.mean(vals)) if vals else float("nan"),
            "ap_per_class": aps}
