#!/usr/bin/env python
"""Self-tests for ``src/eval/pixel_metrics.py`` (DESIGN-PIXEL-EVENT-NETWORK.md §9).

Run from anywhere with the project interpreter:

    & 'D:\smartcar\.venv\Scripts\python.exe' tests\test_pixel_metrics.py

The script uses plain ``assert``-style checks (no pytest/sklearn/torch): every
section either raises ``AssertionError`` with a readable message or increments a
global counter, and the last printed line is ``ALL OK``.

Coverage:
  * hand-computed values for each of the 7 metric functions,
  * degenerate inputs (T=0, empty arrays, all-ignore, single class,
    count_pred all zero, empty event lists),
  * invariants (perfect phase prediction => macro_f1 == 1.0; perfect boundary
    prediction => precision == recall == f1 == 1.0 and zero latency; perfectly
    calibrated probabilities => ECE/MCE ~ 0; perfect counts => zero error),
  * independent reference implementations cross-checked on fixed RNG seeds.
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# import the module under test regardless of the current working directory
# --------------------------------------------------------------------------- #
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import pixel_metrics as pm  # noqa: E402

TOL = 1e-12
CHECKS = 0


# --------------------------------------------------------------------------- #
# tiny assertion helpers
# --------------------------------------------------------------------------- #
def _fail(msg: str) -> None:
    raise AssertionError(msg)


def ok(msg: str) -> None:
    global CHECKS
    CHECKS += 1


def check(cond, msg: str) -> None:
    if not cond:
        _fail(msg)
    ok(msg)


def is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def check_close(actual, expected, msg: str, atol: float = TOL) -> None:
    if is_nan(expected):
        if not is_nan(actual):
            _fail(f"{msg}: expected nan, got {actual!r}")
    else:
        if is_nan(actual) or abs(float(actual) - float(expected)) > atol:
            _fail(f"{msg}: expected {expected!r} (+-{atol:g}), got {actual!r}")
    ok(msg)


def check_array(actual, expected, msg: str, atol: float = TOL) -> None:
    try:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
            rtol=0.0,
            atol=atol,
            equal_nan=True,
        )
    except AssertionError as exc:
        _fail(f"{msg}: {exc}")
    ok(msg)


def check_confusion(actual, expected, msg: str) -> None:
    a = np.asarray(actual)
    e = np.asarray(expected)
    check(a.shape == e.shape, f"{msg}: shape {a.shape} != {e.shape}")
    check(np.array_equal(a, e), f"{msg}: {a.tolist()} != {e.tolist()}")


def check_raises(fn, exc_type, msg: str) -> None:
    try:
        fn()
    except exc_type:
        ok(msg)
        return
    except Exception as exc:  # noqa: BLE001
        _fail(f"{msg}: raised {type(exc).__name__}({exc}) instead of {exc_type.__name__}")
    _fail(f"{msg}: no exception raised")


# --------------------------------------------------------------------------- #
# independent reference implementations (used only to cross-check the module)
# --------------------------------------------------------------------------- #
def ref_confusion(pred, true, n_classes: int, ignore: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for p, t in zip(pred, true):
        if int(t) == ignore:
            continue
        conf[int(t), int(p)] += 1
    return conf


def ref_macro_f1(conf: np.ndarray):
    f1s = []
    for c in range(conf.shape[0]):
        tp = float(conf[c, c])
        support = float(conf[c, :].sum())
        n_pred = float(conf[:, c].sum())
        if support == 0.0 and n_pred == 0.0:
            continue
        f1s.append(0.0 if tp == 0.0 else 2.0 * tp / (support + n_pred))
    return (float(np.mean(f1s)) if f1s else float("nan")), f1s


def ref_boundary(pred, true, tol: int):
    candidates = []
    for i, p in enumerate(pred):
        for j, t in enumerate(true):
            if abs(int(p) - int(t)) <= tol:
                candidates.append((abs(int(p) - int(t)), i, j))
    candidates.sort()
    used_p, used_t, matched = set(), set(), []
    for dist, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matched.append((i, j, dist))
    return matched


def ref_calibration(p, y, n_bins: int):
    """Independent equal-width binning: bin i covers edges[i] <= p < edges[i+1],
    last bin right-closed.  Returns (counts, accs, confs, ece, mce, brier)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    pos = np.zeros(n_bins, dtype=np.float64)
    conf_sum = np.zeros(n_bins, dtype=np.float64)
    for prob, label in zip(p, y):
        idx = n_bins - 1
        for b in range(n_bins):
            if edges[b] <= prob < edges[b + 1]:
                idx = b
                break
        counts[idx] += 1
        pos[idx] += label
        conf_sum[idx] += prob
    n = p.size
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    nonempty = counts > 0
    accs[nonempty] = pos[nonempty] / counts[nonempty]
    confs[nonempty] = conf_sum[nonempty] / counts[nonempty]
    if n and nonempty.any():
        gaps = np.abs(accs[nonempty] - confs[nonempty])
        ece = float(np.sum(counts[nonempty] / n * gaps))
        mce = float(np.max(gaps))
        brier = float(np.mean((p - y) ** 2))
    else:
        ece = mce = brier = float("nan")
    return counts, accs, confs, ece, mce, brier


# --------------------------------------------------------------------------- #
# 1. phase_prf
# --------------------------------------------------------------------------- #
def test_phase_prf() -> None:
    # ---- hand-computed confusion / F1 / accuracy ------------------------- #
    pred = [0, 1, 1, 2, 0]
    true = [0, 1, 2, 0, 0]
    r = pm.phase_prf(pred, true, 3)
    check_confusion(
        r["confusion"],
        [[2, 0, 1], [0, 1, 0], [0, 1, 0]],
        "phase_prf hand confusion (rows=true, cols=pred)",
    )
    # class0: tp=2 support=3 pred=2 -> f1=4/5;  class1: tp=1 support=1 pred=2 -> 2/3
    # class2: tp=0 support=1 pred=1 -> 0 (counted in macro)
    check_array(r["per_class_f1"], [0.8, 2.0 / 3.0, 0.0], "phase_prf hand per_class_f1")
    check_close(r["macro_f1"], (0.8 + 2.0 / 3.0 + 0.0) / 3.0, "phase_prf hand macro_f1")
    check_close(r["accuracy"], 3.0 / 5.0, "phase_prf hand accuracy")
    check(r["n_valid"] == 5, "phase_prf hand n_valid == 5")
    check(r["confusion"].dtype == np.int64, "phase_prf confusion dtype int64")

    # ---- ignore_index frames are excluded from every statistic ---------- #
    r = pm.phase_prf([0, 2, 1, 2], [0, -1, 1, -1], 3)
    check(r["n_valid"] == 2, "phase_prf ignore: n_valid == 2")
    check_confusion(
        r["confusion"],
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
        "phase_prf ignore: prediction on ignored frame creates no confusion count",
    )
    check_close(r["accuracy"], 1.0, "phase_prf ignore: accuracy == 1.0")
    check_close(r["macro_f1"], 1.0, "phase_prf ignore: macro_f1 == 1.0")
    check(is_nan(r["per_class_f1"][2]),
          "phase_prf ignore: never-observed class 2 F1 is nan (excluded from macro)")

    # ---- predicted-but-never-true class contributes F1 = 0 to macro ----- #
    r = pm.phase_prf([0, 1, 1], [0, 0, 0], 2)
    check_array(r["per_class_f1"], [0.5, 0.0], "phase_prf predicted-only class F1")
    check_close(r["macro_f1"], 0.25, "phase_prf predicted-only class macro == 0.25")
    check_close(r["accuracy"], 1.0 / 3.0, "phase_prf predicted-only accuracy")

    # ---- single class only ---------------------------------------------- #
    r = pm.phase_prf([2] * 7, [2] * 7, 3)
    check_array(r["per_class_f1"], [np.nan, np.nan, 1.0], "phase_prf single-class per_class")
    check_close(r["macro_f1"], 1.0, "phase_prf single-class macro_f1 == 1.0")
    check_close(r["accuracy"], 1.0, "phase_prf single-class accuracy == 1.0")

    # ---- n_classes == 1 -------------------------------------------------- #
    r = pm.phase_prf([0, 0, 0], [0, 0, 0], 1)
    check_confusion(r["confusion"], [[3]], "phase_prf n_classes=1 confusion [[3]]")
    check_close(r["macro_f1"], 1.0, "phase_prf n_classes=1 macro_f1 == 1.0")

    # ---- inputs are not mutated ----------------------------------------- #
    pred_arr = np.array([0, 1, 2])
    true_arr = np.array([0, 1, 1])
    pm.phase_prf(pred_arr, true_arr, 3)
    check(np.array_equal(pred_arr, [0, 1, 2]) and np.array_equal(true_arr, [0, 1, 1]),
          "phase_prf does not mutate its inputs")

    # ---- invariant: perfect prediction => macro_f1 == 1.0, accuracy 1.0 - #
    rng = np.random.default_rng(0)
    for _ in range(5):
        t = rng.integers(0, 5, size=137)
        r = pm.phase_prf(t.copy(), t.copy(), 5)
        check_close(r["macro_f1"], 1.0, "phase_prf perfect macro_f1 == 1.0")
        check_close(r["accuracy"], 1.0, "phase_prf perfect accuracy == 1.0")
        check(np.array_equal(r["confusion"], np.diag(np.diag(r["confusion"]))),
              "phase_prf perfect confusion is diagonal")

    # ---- T == 0 ---------------------------------------------------------- #
    r = pm.phase_prf([], [], 3)
    check(is_nan(r["macro_f1"]), "phase_prf T=0: macro_f1 nan")
    check(is_nan(r["accuracy"]), "phase_prf T=0: accuracy nan")
    check_array(r["per_class_f1"], [np.nan, np.nan, np.nan], "phase_prf T=0: per_class all nan")
    check(r["confusion"].shape == (3, 3), "phase_prf T=0: confusion shape [3,3]")
    check(int(r["confusion"].sum()) == 0, "phase_prf T=0: confusion all zeros")
    check(r["n_valid"] == 0, "phase_prf T=0: n_valid == 0")

    # ---- all frames ignored --------------------------------------------- #
    r = pm.phase_prf([0, 1, 2], [-1, -1, -1], 3)
    check(is_nan(r["macro_f1"]), "phase_prf all-ignore: macro_f1 nan")
    check(is_nan(r["accuracy"]), "phase_prf all-ignore: accuracy nan")
    check(int(r["confusion"].sum()) == 0, "phase_prf all-ignore: confusion zeros")
    check(r["n_valid"] == 0, "phase_prf all-ignore: n_valid == 0")

    # ---- input normalisation and validation ----------------------------- #
    r_list = pm.phase_prf([0, 1, 2, 0], [0, 1, 2, 0], 3)
    r_np = pm.phase_prf(np.array([0, 1, 2, 0]), np.array([0.0, 1.0, 2.0, 0.0]), 3)
    check_array(r_list["per_class_f1"], r_np["per_class_f1"], "phase_prf list/int-float arrays agree")
    check_close(r_np["macro_f1"], 1.0, "phase_prf integer-valued float labels accepted")

    check_raises(lambda: pm.phase_prf([0, 1], [0], 2), ValueError, "phase_prf length mismatch raises")
    check_raises(lambda: pm.phase_prf([0, 3], [0, 1], 3), ValueError, "phase_prf out-of-range pred raises")
    check_raises(lambda: pm.phase_prf([0, 1.5], [0, 1], 3), ValueError, "phase_prf non-integer float pred raises")
    check_raises(lambda: pm.phase_prf([0, np.nan], [0, 1], 3), ValueError, "phase_prf nan pred raises")
    check_raises(lambda: pm.phase_prf([0, 1], [0, 1], 0), ValueError, "phase_prf n_classes=0 raises")
    check_raises(lambda: pm.phase_prf([0, 1], [0, 1], 2.0), TypeError, "phase_prf non-int n_classes raises")

    # ---- independent reference cross-check on seeded random data -------- #
    rng = np.random.default_rng(1)
    for trial in range(20):
        n_frames = int(rng.integers(0, 40))
        n_classes = int(rng.integers(1, 5))
        t = rng.integers(0, n_classes, size=n_frames)
        p = rng.integers(0, n_classes, size=n_frames)
        keep = rng.random(n_frames) > 0.2
        t_ignored = t.copy()
        t_ignored[~keep] = -1
        r = pm.phase_prf(p, t_ignored, n_classes)
        conf = ref_confusion(p, t_ignored, n_classes, -1)
        macro, _ = ref_macro_f1(conf)
        check_confusion(r["confusion"], conf, f"phase_prf ref confusion trial {trial}")
        if is_nan(macro):
            check(is_nan(r["macro_f1"]), f"phase_prf ref macro nan trial {trial}")
        else:
            check_close(r["macro_f1"], macro, f"phase_prf ref macro trial {trial}")
        n_valid = int(keep.sum())
        exp_acc = float(np.mean(p[keep] == t[keep])) if n_valid else float("nan")
        check_close(r["accuracy"], exp_acc, f"phase_prf ref accuracy trial {trial}")


# --------------------------------------------------------------------------- #
# 2. boundary_prf
# --------------------------------------------------------------------------- #
def test_boundary_prf() -> None:
    # ---- hand case: all three matched, |delay| == 1 --------------------- #
    r = pm.boundary_prf([11, 19, 31], [10, 20, 30], 2)
    check_close(r["precision"], 1.0, "boundary_prf hand precision == 1.0")
    check_close(r["recall"], 1.0, "boundary_prf hand recall == 1.0")
    check_close(r["f1"], 1.0, "boundary_prf hand f1 == 1.0")
    check_close(r["latency_mean_frames"], 1.0, "boundary_prf hand mean |latency| == 1")
    check_close(r["latency_signed_mean"], 1.0 / 3.0, "boundary_prf hand signed latency == 1/3")
    check(r["n_matched"] == 3 and r["n_pred"] == 3 and r["n_true"] == 3,
          "boundary_prf hand match counts")

    # ---- greedy tie-break: equidistant candidates, smaller true index --- #
    r = pm.boundary_prf([11], [10, 12], 2)
    check_close(r["precision"], 1.0, "boundary_prf tie precision == 1.0")
    check_close(r["recall"], 0.5, "boundary_prf tie recall == 0.5")
    check_close(r["f1"], 2.0 / 3.0, "boundary_prf tie f1 == 2/3")
    check_close(r["latency_signed_mean"], 1.0,
                "boundary_prf tie matches true index 10 first (delay +1)")

    # ---- false positives and false negatives ---------------------------- #
    r = pm.boundary_prf([10, 15, 25], [10, 20], 2)
    check_close(r["precision"], 1.0 / 3.0, "boundary_prf FP precision == 1/3")
    check_close(r["recall"], 0.5, "boundary_prf FN recall == 0.5")
    check_close(r["f1"], 0.4, "boundary_prf FP/FN f1 == 0.4")
    check_close(r["latency_mean_frames"], 0.0, "boundary_prf FP/FN matched latency == 0")
    check_close(r["latency_signed_mean"], 0.0, "boundary_prf FP/FN signed latency == 0")

    # ---- one-to-one: a single prediction cannot match two truths -------- #
    r = pm.boundary_prf([5], [5, 6], 1)
    check_close(r["precision"], 1.0, "boundary_prf one-to-one precision")
    check_close(r["recall"], 0.5, "boundary_prf one-to-one recall")
    check_close(r["f1"], 2.0 / 3.0, "boundary_prf one-to-one f1")

    # ---- duplicate indices on both sides still match one-to-one --------- #
    r = pm.boundary_prf([0, 0], [0, 0], 0)
    check_close(r["precision"], 1.0, "boundary_prf duplicate-index precision")
    check_close(r["recall"], 1.0, "boundary_prf duplicate-index recall")
    check(r["n_matched"] == 2, "boundary_prf duplicate-index n_matched == 2")

    # ---- no match within tolerance -------------------------------------- #
    r = pm.boundary_prf([20], [10], 2)
    check_close(r["precision"], 0.0, "boundary_prf no-match precision == 0")
    check_close(r["recall"], 0.0, "boundary_prf no-match recall == 0")
    check_close(r["f1"], 0.0, "boundary_prf no-match f1 == 0")
    check(is_nan(r["latency_mean_frames"]), "boundary_prf no-match abs latency nan")
    check(is_nan(r["latency_signed_mean"]), "boundary_prf no-match signed latency nan")

    # ---- tolerance == 0 means exact frame agreement --------------------- #
    r = pm.boundary_prf([4, 9, 15], [4, 8, 15], 0)
    check_close(r["precision"], 2.0 / 3.0, "boundary_prf tol=0 precision == 2/3")
    check_close(r["recall"], 2.0 / 3.0, "boundary_prf tol=0 recall == 2/3")

    # ---- invariant: perfect prediction => P=R=F1=1 and zero latency ----- #
    rng = np.random.default_rng(2)
    for _ in range(5):
        idx = np.sort(rng.choice(60, size=9, replace=False))
        r = pm.boundary_prf(idx, idx, 3)
        check_close(r["precision"], 1.0, "boundary_prf perfect precision == 1.0")
        check_close(r["recall"], 1.0, "boundary_prf perfect recall == 1.0")
        check_close(r["f1"], 1.0, "boundary_prf perfect f1 == 1.0")
        check_close(r["latency_mean_frames"], 0.0, "boundary_prf perfect abs latency == 0")
        check_close(r["latency_signed_mean"], 0.0, "boundary_prf perfect signed latency == 0")

    # ---- degenerate empty inputs ---------------------------------------- #
    r = pm.boundary_prf([], [], 2)
    check_close(r["precision"], 1.0, "boundary_prf empty/empty precision == 1.0")
    check_close(r["recall"], 1.0, "boundary_prf empty/empty recall == 1.0")
    check_close(r["f1"], 1.0, "boundary_prf empty/empty f1 == 1.0")
    check(is_nan(r["latency_mean_frames"]), "boundary_prf empty/empty latency nan")
    check(r["n_matched"] == 0, "boundary_prf empty/empty n_matched == 0")

    r = pm.boundary_prf([], [4, 9], 2)
    check_close(r["precision"], 0.0, "boundary_prf empty pred precision == 0")
    check_close(r["recall"], 0.0, "boundary_prf empty pred recall == 0")
    check_close(r["f1"], 0.0, "boundary_prf empty pred f1 == 0")

    r = pm.boundary_prf([4, 9], [], 2)
    check_close(r["precision"], 0.0, "boundary_prf empty true precision == 0")
    check_close(r["recall"], 0.0, "boundary_prf empty true recall == 0")
    check_close(r["f1"], 0.0, "boundary_prf empty true f1 == 0")

    # empty float arrays (a common padding artefact) are accepted
    r = pm.boundary_prf(np.array([], dtype=float), np.array([], dtype=float), 2)
    check_close(r["precision"], 1.0, "boundary_prf empty float arrays accepted")

    # ---- validation ------------------------------------------------------ #
    check_raises(lambda: pm.boundary_prf([1], [1], -1), ValueError, "boundary_prf negative tolerance raises")
    check_raises(lambda: pm.boundary_prf([1], [1], 2.5), TypeError, "boundary_prf non-integer tolerance raises")
    check_raises(lambda: pm.boundary_prf([1.5], [1], 1), ValueError, "boundary_prf non-integer index raises")

    # ---- independent reference cross-check on seeded random data -------- #
    rng = np.random.default_rng(3)
    for trial in range(60):
        m = int(rng.integers(0, 9))
        k = int(rng.integers(0, 9))
        pred = np.sort(rng.integers(0, 40, size=m))
        true = np.sort(rng.integers(0, 40, size=k))
        tol = int(rng.integers(0, 6))
        r = pm.boundary_prf(pred, true, tol)
        ref = ref_boundary(pred, true, tol)
        check(r["n_matched"] == len(ref), f"boundary_prf ref n_matched trial {trial}")
        if ref:
            exp_p = len(ref) / m if m else (1.0 if k == 0 else 0.0)
            exp_r = len(ref) / k if k else (1.0 if m == 0 else 0.0)
            delays = np.array([int(pred[i]) - int(true[j]) for i, j, _ in ref], dtype=float)
            check_close(r["precision"], exp_p, f"boundary_prf ref precision trial {trial}")
            check_close(r["recall"], exp_r, f"boundary_prf ref recall trial {trial}")
            check_close(r["latency_mean_frames"], float(np.mean(np.abs(delays))),
                        f"boundary_prf ref abs latency trial {trial}")
            check_close(r["latency_signed_mean"], float(np.mean(delays)),
                        f"boundary_prf ref signed latency trial {trial}")
        else:
            exp_p = 1.0 if m == 0 and k == 0 else 0.0
            exp_r = 1.0 if m == 0 and k == 0 else 0.0
            check_close(r["precision"], exp_p, f"boundary_prf ref empty precision trial {trial}")
            check_close(r["recall"], exp_r, f"boundary_prf ref empty recall trial {trial}")


# --------------------------------------------------------------------------- #
# 3. count_metrics
# --------------------------------------------------------------------------- #
def test_count_metrics() -> None:
    r = pm.count_metrics([2, 0, 1, 3], [2, 1, 1, 1])
    check_close(r["exact_accuracy"], 0.5, "count_metrics hand exact_accuracy == 0.5")
    check_close(r["mae"], 0.75, "count_metrics hand mae == 0.75")
    check_close(r["over_rate"], 0.25, "count_metrics hand over_rate == 0.25")
    check_close(r["under_rate"], 0.25, "count_metrics hand under_rate == 0.25")
    check_close(r["mean_signed_error"], 0.25, "count_metrics hand mean_signed_error == 0.25")

    # ---- invariant: perfect counts => no error of any kind --------------- #
    counts = np.array([0, 1, 2, 3, 3])
    r = pm.count_metrics(counts, counts)
    check_close(r["exact_accuracy"], 1.0, "count_metrics perfect exact_accuracy == 1.0")
    check_close(r["mae"], 0.0, "count_metrics perfect mae == 0.0")
    check_close(r["over_rate"], 0.0, "count_metrics perfect over_rate == 0.0")
    check_close(r["under_rate"], 0.0, "count_metrics perfect under_rate == 0.0")
    check_close(r["mean_signed_error"], 0.0, "count_metrics perfect signed error == 0.0")

    # ---- degenerate: count_pred all zero -------------------------------- #
    r = pm.count_metrics([0, 0, 0], [0, 1, 2])
    check_close(r["exact_accuracy"], 1.0 / 3.0, "count_metrics all-zero pred exact == 1/3")
    check_close(r["mae"], 1.0, "count_metrics all-zero pred mae == 1.0")
    check_close(r["over_rate"], 0.0, "count_metrics all-zero pred over_rate == 0")
    check_close(r["under_rate"], 2.0 / 3.0, "count_metrics all-zero pred under_rate == 2/3")
    check_close(r["mean_signed_error"], -1.0, "count_metrics all-zero pred signed == -1.0")

    # ---- degenerate: all true zero --------------------------------------- #
    r = pm.count_metrics([0, 1, 0], [0, 0, 0])
    check_close(r["over_rate"], 1.0 / 3.0, "count_metrics over-count rate == 1/3")
    check_close(r["mae"], 1.0 / 3.0, "count_metrics mae over zeros == 1/3")

    # ---- degenerate: N == 0 ---------------------------------------------- #
    r = pm.count_metrics([], [])
    for key in ("exact_accuracy", "mae", "over_rate", "under_rate", "mean_signed_error"):
        check(is_nan(r[key]), f"count_metrics N=0: {key} nan")

    # ---- validation ------------------------------------------------------ #
    check_raises(lambda: pm.count_metrics([1, 2], [1]), ValueError, "count_metrics length mismatch raises")
    check_raises(lambda: pm.count_metrics([1.5], [1]), ValueError, "count_metrics non-integer count raises")
    r = pm.count_metrics(np.array([2.0, 0.0]), np.array([2, 0]))
    check_close(r["exact_accuracy"], 1.0, "count_metrics integer-valued floats accepted")


# --------------------------------------------------------------------------- #
# 4. presence_metrics
# --------------------------------------------------------------------------- #
def test_presence_metrics() -> None:
    # ---- perfect 4-state sequence, two presence transitions -------------- #
    seq = [0, 0, 1, 2, 2, 3, 0, 0]
    r = pm.presence_metrics(seq, seq)
    check_close(r["macro_f1"], 1.0, "presence_metrics perfect macro_f1 == 1.0")
    check_close(r["presence_precision"], 1.0, "presence_metrics perfect presence_precision == 1.0")
    check_close(r["presence_recall"], 1.0, "presence_metrics perfect presence_recall == 1.0")
    check_close(r["mean_switch_delay_frames"], 0.0, "presence_metrics perfect switch delay == 0")
    check(r["n_switch_true"] == 2, "presence_metrics perfect n_switch_true == 2")
    check(r["n_switch_matched"] == 2, "presence_metrics perfect n_switch_matched == 2")
    check(r["identity_switch_rate_placeholder"] is None,
          "presence_metrics identity_switch_rate_placeholder is None")

    # ---- hand-computed imperfect case ----------------------------------- #
    # true: 0,0,2,0 ; pred: 0,2,2,0
    # confusion rows=true: [2,0,1,0], [0,0,0,0], [0,0,1,0], [0,0,0,0]
    # class0 f1=2*2/(3+2)=0.8 ; class2 f1=2*1/(1+2)=2/3 ; macro=0.7333...
    # presence: TP=1 (t2), FP=1 (t1), FN=0 -> P=0.5, R=1.0
    # switches: true +1@2, -1@3 ; pred +1@1, -1@3 -> delays -1, 0 -> mean -0.5
    r = pm.presence_metrics([0, 2, 2, 0], [0, 0, 2, 0])
    check_close(r["macro_f1"], (0.8 + 2.0 / 3.0) / 2.0, "presence_metrics hand macro_f1")
    check_close(r["presence_precision"], 0.5, "presence_metrics hand presence_precision == 0.5")
    check_close(r["presence_recall"], 1.0, "presence_metrics hand presence_recall == 1.0")
    check_close(r["mean_switch_delay_frames"], -0.5, "presence_metrics hand switch delay == -0.5")

    # ---- second hand case: equal P/R and -1 mean delay ------------------- #
    # true: 0,0,2,2,0 ; pred: 0,2,2,0,0
    # class0 f1=4/(3+3)=2/3 ; class2 f1=2/(2+2)=0.5 ; macro=0.58333...
    # TP=1, FP=1, FN=1 -> P=R=0.5 ; switches delays -1, -1 -> mean -1.0
    r = pm.presence_metrics([0, 2, 2, 0, 0], [0, 0, 2, 2, 0])
    check_close(r["macro_f1"], (2.0 / 3.0 + 0.5) / 2.0, "presence_metrics 2nd hand macro_f1")
    check_close(r["presence_precision"], 0.5, "presence_metrics 2nd hand precision == 0.5")
    check_close(r["presence_recall"], 0.5, "presence_metrics 2nd hand recall == 0.5")
    check_close(r["mean_switch_delay_frames"], -1.0, "presence_metrics 2nd hand switch delay == -1")

    # ---- late entering switch, missed leaving switch --------------------- #
    # true: 0,1,2,3,0 ; pred: 0,0,1,2,3
    # true b switches +1@1, -1@4 ; pred b switch +1@2 only -> matched delay +1
    r = pm.presence_metrics([0, 0, 1, 2, 3], [0, 1, 2, 3, 0])
    check_close(r["mean_switch_delay_frames"], 1.0,
                "presence_metrics delayed entering switch delay == +1")
    check(r["n_switch_true"] == 2, "presence_metrics delayed entering n_switch_true == 2")
    check(r["n_switch_matched"] == 1, "presence_metrics delayed entering n_switch_matched == 1")
    check_close(r["presence_precision"], 2.0 / 3.0, "presence_metrics shifted presence precision")
    check_close(r["presence_recall"], 2.0 / 3.0, "presence_metrics shifted presence recall")

    # ---- degenerate: all absent and correct ------------------------------ #
    r = pm.presence_metrics([0] * 6, [0] * 6)
    check_close(r["macro_f1"], 1.0, "presence_metrics all-absent macro_f1 == 1.0")
    check_close(r["presence_precision"], 1.0, "presence_metrics all-absent presence_precision == 1.0")
    check_close(r["presence_recall"], 1.0, "presence_metrics all-absent presence_recall == 1.0")
    check(is_nan(r["mean_switch_delay_frames"]), "presence_metrics all-absent switch delay nan")

    # ---- degenerate: single class only ----------------------------------- #
    r = pm.presence_metrics([2] * 6, [2] * 6)
    check_close(r["macro_f1"], 1.0, "presence_metrics single-class macro_f1 == 1.0")
    check_close(r["presence_precision"], 1.0, "presence_metrics single-class precision == 1.0")
    check_close(r["presence_recall"], 1.0, "presence_metrics single-class recall == 1.0")
    check(is_nan(r["mean_switch_delay_frames"]), "presence_metrics single-class switch delay nan")

    # ---- degenerate: T == 0 ---------------------------------------------- #
    r = pm.presence_metrics([], [])
    check(is_nan(r["macro_f1"]), "presence_metrics T=0: macro_f1 nan")
    check(is_nan(r["presence_precision"]), "presence_metrics T=0: presence_precision nan")
    check(is_nan(r["presence_recall"]), "presence_metrics T=0: presence_recall nan")
    check(is_nan(r["mean_switch_delay_frames"]), "presence_metrics T=0: switch delay nan")
    check(r["identity_switch_rate_placeholder"] is None, "presence_metrics T=0: placeholder None")

    # ---- degenerate: T == 1 (no transitions possible) ------------------- #
    r = pm.presence_metrics([2], [2])
    check_close(r["macro_f1"], 1.0, "presence_metrics T=1 macro_f1 == 1.0")
    check(is_nan(r["mean_switch_delay_frames"]), "presence_metrics T=1 switch delay nan")

    # ---- validation ------------------------------------------------------ #
    check_raises(lambda: pm.presence_metrics([4], [4]), ValueError, "presence_metrics state 4 raises")
    check_raises(lambda: pm.presence_metrics([-1], [-1]), ValueError, "presence_metrics state -1 raises")
    check_raises(lambda: pm.presence_metrics([0, 1], [0]), ValueError, "presence_metrics length mismatch raises")

    # ---- property: counts and directions are consistent ----------------- #
    rng = np.random.default_rng(4)
    for trial in range(20):
        n_frames = int(rng.integers(0, 30))
        true = rng.integers(0, 4, size=n_frames)
        pred = rng.integers(0, 4, size=n_frames)
        r = pm.presence_metrics(pred, true)
        check(0.0 <= r["presence_precision"] <= 1.0,
              f"presence_metrics precision in [0,1] trial {trial}")
        check(0.0 <= r["presence_recall"] <= 1.0,
              f"presence_metrics recall in [0,1] trial {trial}")
        check(r["n_switch_matched"] <= r["n_switch_true"],
              f"presence_metrics matched <= true switches trial {trial}")


# --------------------------------------------------------------------------- #
# 5. calibration_metrics
# --------------------------------------------------------------------------- #
def test_calibration_metrics() -> None:
    # ---- hand case ------------------------------------------------------ #
    r = pm.calibration_metrics([0.9, 0.1], [1, 0])
    check_close(r["brier"], 0.01, "calibration_metrics hand brier == 0.01")
    check_close(r["ece"], 0.1, "calibration_metrics hand ece == 0.1")
    check_close(r["mce"], 0.1, "calibration_metrics hand mce == 0.1")
    rel = r["reliability"]
    check(len(rel["bin_edges"]) == 16, "calibration_metrics reliability bin_edges length 16")
    check(sum(rel["bin_count"]) == 2, "calibration_metrics reliability bin_count sums to N")
    check(rel["bin_count"][1] == 1 and rel["bin_count"][13] == 1,
          "calibration_metrics hand bin assignment (0.1 -> bin 1, 0.9 -> bin 13)")
    check_close(rel["bin_acc"][1], 0.0, "calibration_metrics hand bin1 acc == 0")
    check_close(rel["bin_conf"][1], 0.1, "calibration_metrics hand bin1 conf == 0.1")
    check_close(rel["bin_acc"][13], 1.0, "calibration_metrics hand bin13 acc == 1")
    check_close(rel["bin_conf"][13], 0.9, "calibration_metrics hand bin13 conf == 0.9")
    check(is_nan(rel["bin_acc"][0]), "calibration_metrics hand empty bin acc is nan")
    check(set(rel.keys()) == {"bin_edges", "bin_count", "bin_acc", "bin_conf"},
          "calibration_metrics reliability keys exactly the 4 documented ones")

    # ---- invariant: perfectly calibrated uniform predictions => ECE ~ 0 -- #
    n_bins = 10
    probs, labels = [], []
    per_bin_pos, per_bin_p = [], []
    for b in range(n_bins):
        p = (b + 0.5) / n_bins
        pos = int(round(p * 20))
        per_bin_p.append(p)
        per_bin_pos.append(pos)
        probs.extend([p] * 20)
        labels.extend([1] * pos + [0] * (20 - pos))
    probs = np.array(probs)
    labels = np.array(labels)
    r = pm.calibration_metrics(probs, labels, n_bins=n_bins)
    check(abs(r["ece"]) < 1e-12, "calibration invariant: perfectly calibrated ECE ~ 0")
    check(abs(r["mce"]) < 1e-12, "calibration invariant: perfectly calibrated MCE ~ 0")
    expected_sq_err = []
    for b in range(n_bins):
        p = per_bin_p[b]
        pos = per_bin_pos[b]
        expected_sq_err.extend([(1.0 - p) ** 2] * pos + [p ** 2] * (20 - pos))
    expected_brier = float(np.mean(expected_sq_err))
    check_close(r["brier"], expected_brier, "calibration invariant: perfectly calibrated brier hand value")
    expected_ece = sum(
        20.0 / len(probs) * abs(per_bin_pos[b] / 20.0 - per_bin_p[b])
        for b in range(n_bins)
    )
    check_close(r["ece"], expected_ece, "calibration invariant: ECE matches per-bin hand formula")
    check_close(r["mce"], 0.0, "calibration invariant: every bin gap is 0")

    # ---- edge probabilities p == 0.0 and p == 1.0 ----------------------- #
    r = pm.calibration_metrics([0.0, 1.0], [0, 1], n_bins=15)
    check_close(r["ece"], 0.0, "calibration_metrics p=0/p=1 ECE == 0")
    check_close(r["mce"], 0.0, "calibration_metrics p=0/p=1 MCE == 0")
    check_close(r["brier"], 0.0, "calibration_metrics p=0/p=1 brier == 0")
    check(r["reliability"]["bin_count"][0] == 1 and r["reliability"]["bin_count"][-1] == 1,
          "calibration_metrics p=0 -> first bin, p=1 -> last bin")

    # ---- badly calibrated predictor ------------------------------------- #
    r = pm.calibration_metrics([0.9, 0.9], [0, 0])
    check_close(r["ece"], 0.9, "calibration_metrics overconfident ECE == 0.9")
    check_close(r["mce"], 0.9, "calibration_metrics overconfident MCE == 0.9")
    check_close(r["brier"], 0.81, "calibration_metrics overconfident brier == 0.81")

    # ---- n_bins == 1 ----------------------------------------------------- #
    r = pm.calibration_metrics([0.2, 0.8], [0, 1], n_bins=1)
    check(len(r["reliability"]["bin_edges"]) == 2, "calibration_metrics n_bins=1: 2 edges")
    check_close(r["ece"], 0.0, "calibration_metrics n_bins=1 ECE == 0 (means agree)")
    check_close(r["mce"], 0.0, "calibration_metrics n_bins=1 MCE == 0")
    check_close(r["brier"], 0.04, "calibration_metrics n_bins=1 brier == 0.04")

    # ---- exact bin-edge value goes to the upper bin --------------------- #
    # n_bins=2 -> edges [0, 0.5, 1]; p == 0.5 belongs to the second bin,
    # so both samples land there -> acc = conf = 0.5 -> ECE = 0.
    r = pm.calibration_metrics([0.5, 0.5], [1, 0], n_bins=2)
    check(r["reliability"]["bin_count"] == [0, 2],
          "calibration_metrics p == inner edge goes to upper bin")
    check_close(r["reliability"]["bin_acc"][1], 0.5, "calibration_metrics edge bin acc == 0.5")
    check_close(r["reliability"]["bin_conf"][1], 0.5, "calibration_metrics edge bin conf == 0.5")
    check_close(r["ece"], 0.0, "calibration_metrics edge bin ECE == 0")
    check_close(r["brier"], 0.25, "calibration_metrics edge bin brier == 0.25")

    # ---- degenerate: N == 0 ---------------------------------------------- #
    r = pm.calibration_metrics([], [], n_bins=15)
    check(is_nan(r["brier"]), "calibration_metrics N=0: brier nan")
    check(is_nan(r["ece"]), "calibration_metrics N=0: ece nan")
    check(is_nan(r["mce"]), "calibration_metrics N=0: mce nan")
    check(len(r["reliability"]["bin_edges"]) == 16, "calibration_metrics N=0: 16 edges")
    check(sum(r["reliability"]["bin_count"]) == 0, "calibration_metrics N=0: counts all zero")
    check(all(is_nan(v) for v in r["reliability"]["bin_acc"]),
          "calibration_metrics N=0: bin_acc all nan")
    check(all(is_nan(v) for v in r["reliability"]["bin_conf"]),
          "calibration_metrics N=0: bin_conf all nan")

    # ---- validation ------------------------------------------------------ #
    check_raises(lambda: pm.calibration_metrics([1.5], [1]), ValueError,
                 "calibration_metrics p>1 raises")
    check_raises(lambda: pm.calibration_metrics([-0.1], [1]), ValueError,
                 "calibration_metrics p<0 raises")
    check_raises(lambda: pm.calibration_metrics([0.5], [2]), ValueError,
                 "calibration_metrics y=2 raises")
    check_raises(lambda: pm.calibration_metrics([0.5], [0], n_bins=0), ValueError,
                 "calibration_metrics n_bins=0 raises")
    check_raises(lambda: pm.calibration_metrics([0.5, 0.6], [0]), ValueError,
                 "calibration_metrics length mismatch raises")
    check_raises(lambda: pm.calibration_metrics([np.nan], [0]), ValueError,
                 "calibration_metrics nan probability raises")
    r = pm.calibration_metrics([1.0 + 1e-9], [1])
    check_close(r["brier"], 0.0, "calibration_metrics tiny numerical overshoot is clipped")

    prob_arr = np.array([0.9, 0.1])
    pm.calibration_metrics(prob_arr, np.array([1, 0]))
    check(np.array_equal(prob_arr, [0.9, 0.1]),
          "calibration_metrics does not mutate its inputs")

    # ---- independent reference cross-check on seeded random data -------- #
    rng = np.random.default_rng(5)
    for trial in range(10):
        n = int(rng.integers(1, 60))
        p = rng.random(n)
        y = rng.integers(0, 2, size=n)
        nb = int(rng.integers(1, 8))
        r = pm.calibration_metrics(p, y, n_bins=nb)
        counts, accs, confs, ece, mce, brier = ref_calibration(p, y, nb)
        check(np.array_equal(np.array(r["reliability"]["bin_count"]), counts),
              f"calibration_metrics ref bin_count trial {trial}")
        check_array(r["reliability"]["bin_acc"], accs,
                    f"calibration_metrics ref bin_acc trial {trial}")
        check_array(r["reliability"]["bin_conf"], confs,
                    f"calibration_metrics ref bin_conf trial {trial}")
        check_close(r["ece"], ece, f"calibration_metrics ref ece trial {trial}")
        check_close(r["mce"], mce, f"calibration_metrics ref mce trial {trial}")
        check_close(r["brier"], brier, f"calibration_metrics ref brier trial {trial}")


# --------------------------------------------------------------------------- #
# 6. abstention_coverage
# --------------------------------------------------------------------------- #
def test_abstention_coverage() -> None:
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [1, 1, 0, 0, 1]
    r = pm.abstention_coverage(conf, correct)
    check_array(r["thresholds"], [0.5, 0.6, 0.7, 0.8, 0.9], "abstention thresholds default order")
    check_array(r["coverage"], [1.0, 0.8, 0.6, 0.4, 0.2], "abstention coverage hand values")
    check_array(
        r["selective_accuracy"],
        [3.0 / 5.0, 2.0 / 4.0, 2.0 / 3.0, 1.0, 1.0],
        "abstention selective_accuracy hand values",
    )

    # ---- inclusive threshold comparison --------------------------------- #
    r = pm.abstention_coverage([0.5], [True], thresholds=(0.5,))
    check_close(r["coverage"][0], 1.0, "abstention confidence == threshold is accepted")
    check_close(r["selective_accuracy"][0], 1.0, "abstention accepted sample accuracy == 1")

    # ---- custom thresholds ---------------------------------------------- #
    r = pm.abstention_coverage(conf, correct, thresholds=(1.0, 0.0))
    check_array(r["thresholds"], [1.0, 0.0], "abstention custom thresholds kept in order")
    check_close(r["coverage"][0], 0.0, "abstention threshold 1.0: coverage 0")
    check(is_nan(r["selective_accuracy"][0]), "abstention threshold 1.0: selective acc nan")
    check_close(r["coverage"][1], 1.0, "abstention threshold 0.0: coverage 1")
    check_close(r["selective_accuracy"][1], 3.0 / 5.0, "abstention threshold 0.0: overall accuracy")

    # ---- empty thresholds ------------------------------------------------ #
    r = pm.abstention_coverage(conf, correct, thresholds=())
    check(r["thresholds"] == [] and r["coverage"] == [] and r["selective_accuracy"] == [],
          "abstention empty thresholds -> three empty lists")

    # ---- no sample accepted --------------------------------------------- #
    r = pm.abstention_coverage([0.1, 0.2], [1, 0])
    check_array(r["coverage"], [0.0, 0.0, 0.0, 0.0, 0.0],
                "abstention all thresholds above confidence: coverage all 0")
    check(all(is_nan(v) for v in r["selective_accuracy"]),
          "abstention all thresholds above confidence: selective acc all nan")

    # ---- degenerate: N == 0 ---------------------------------------------- #
    r = pm.abstention_coverage([], [])
    check_array(r["coverage"], [0.0] * 5, "abstention N=0: coverage all 0")
    check(all(is_nan(v) for v in r["selective_accuracy"]),
          "abstention N=0: selective acc all nan")

    # ---- validation ------------------------------------------------------ #
    check_raises(lambda: pm.abstention_coverage([1.5], [1]), ValueError,
                 "abstention confidence > 1 raises")
    check_raises(lambda: pm.abstention_coverage([0.5], [2]), ValueError,
                 "abstention correct=2 raises")
    check_raises(lambda: pm.abstention_coverage([0.5], [1, 0]), ValueError,
                 "abstention length mismatch raises")

    # ---- property: coverage is monotone non-increasing in the threshold - #
    rng = np.random.default_rng(6)
    conf_rand = rng.random(200)
    correct_rand = rng.integers(0, 2, size=200)
    thr = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    r = pm.abstention_coverage(conf_rand, correct_rand, thresholds=thr)
    for i in range(1, len(thr)):
        check(r["coverage"][i] <= r["coverage"][i - 1],
              f"abstention coverage monotone at threshold {thr[i]}")
    check_close(r["coverage"][0], 1.0, "abstention threshold 0.0 accepts everything")
    overall = float(np.mean(correct_rand))
    check_close(r["selective_accuracy"][0], overall,
                "abstention accept-all accuracy equals overall accuracy")


# --------------------------------------------------------------------------- #
# 7. lag_curve
# --------------------------------------------------------------------------- #
def test_lag_curve() -> None:
    lags, values = pm.lag_curve({30: 0.5, 10: 0.2, 20: 0.4})
    check(lags.dtype == np.int64, "lag_curve lags dtype int64")
    check(values.dtype == np.float64, "lag_curve values dtype float64")
    check_array(lags, [10, 20, 30], "lag_curve lags sorted ascending")
    check_array(values, [0.2, 0.4, 0.5], "lag_curve values follow sorted lags")

    lags, values = pm.lag_curve({})
    check(lags.shape == (0,) and values.shape == (0,), "lag_curve empty mapping -> empty arrays")
    check(lags.dtype == np.int64 and values.dtype == np.float64, "lag_curve empty arrays keep dtypes")

    lags, values = pm.lag_curve({0: -1.5})
    check_array(lags, [0], "lag_curve single key lags")
    check_array(values, [-1.5], "lag_curve single negative value preserved")

    lags, values = pm.lag_curve({-5: 0.1, 5: 0.9, 0: 0.5})
    check_array(lags, [-5, 0, 5], "lag_curve negative lags sorted correctly")
    check_array(values, [0.1, 0.5, 0.9], "lag_curve negative lags values aligned")

    lags, values = pm.lag_curve({np.int64(7): np.float64(0.25)})
    check_array(lags, [7], "lag_curve np.integer key accepted")
    check_close(float(values[0]), 0.25, "lag_curve np.float value preserved")

    check_raises(lambda: pm.lag_curve({True: 0.1}), TypeError, "lag_curve bool key raises")
    check_raises(lambda: pm.lag_curve({"10": 0.1}), TypeError, "lag_curve string key raises")
    check_raises(lambda: pm.lag_curve({1: "x"}), TypeError, "lag_curve non-numeric value raises")
    check_raises(lambda: pm.lag_curve([(1, 0.5)]), TypeError, "lag_curve non-mapping raises")


# --------------------------------------------------------------------------- #
# degenerate-input summary (printed so the caller can quote actual values)
# --------------------------------------------------------------------------- #
def _brief(value) -> str:
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_brief(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        if len(value) <= 8:
            return "[" + ", ".join(_brief(v) for v in value) + "]"
        return f"list(len={len(value)})"
    if isinstance(value, float):
        return f"{value!r}"
    return repr(value)


def print_degenerate_summary() -> None:
    print("\n[DEGENERATE INPUTS -- actual returned values]")
    print("phase_prf T=0                 :", _brief(pm.phase_prf([], [], 3)))
    print("phase_prf all-ignore          :", _brief(pm.phase_prf([0, 1], [-1, -1], 3)))
    print("boundary_prf empty/empty      :", _brief(pm.boundary_prf([], [], 2)))
    print("boundary_prf empty pred       :", _brief(pm.boundary_prf([], [4, 9], 2)))
    print("boundary_prf empty true       :", _brief(pm.boundary_prf([4, 9], [], 2)))
    print("count_metrics N=0             :", _brief(pm.count_metrics([], [])))
    print("count_metrics pred all zero   :", _brief(pm.count_metrics([0, 0, 0], [0, 1, 2])))
    print("presence_metrics T=0          :", _brief(pm.presence_metrics([], [])))
    print("presence_metrics all-absent   :", _brief(pm.presence_metrics([0, 0], [0, 0])))
    print("calibration_metrics N=0       :", _brief(pm.calibration_metrics([], [])))
    print("abstention_coverage N=0       :", _brief(pm.abstention_coverage([], [])))
    print("lag_curve empty               :", _brief(pm.lag_curve({})))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    test_phase_prf()
    test_boundary_prf()
    test_count_metrics()
    test_presence_metrics()
    test_calibration_metrics()
    test_abstention_coverage()
    test_lag_curve()
    print_degenerate_summary()
    print(f"\n{CHECKS} checks passed")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
