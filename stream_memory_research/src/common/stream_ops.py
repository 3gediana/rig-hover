"""Stream perturbations on cached feature sequences (physics-time aware).

Every function takes a `seq` dict with keys
    feat [T,Dg], t [T] (seconds, increasing), optional tok/roi/roi_valid/roi_meta,
    lab_* label arrays  [T,...]
and returns a NEW dict with consistent `t` (dt is derived, never stored).

Design rule: dropping/repeating frames must NOT renumber time.  A dropped frame
simply means the next observation is later -> dt grows.  This is what makes the
down-sampling tests physically meaningful.

All randomness comes from a caller-supplied numpy Generator so seeds are explicit.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

SEQ_KEYS = ("feat", "t", "tok", "roi", "roi_valid", "roi_meta")


def _keys(seq) -> List[str]:
    return [k for k in seq.keys()]


def _take(seq: dict, idx: np.ndarray) -> dict:
    out = {}
    for k, v in seq.items():
        if isinstance(v, np.ndarray) and v.shape[0] == seq["t"].shape[0]:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def _concat(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if isinstance(a[k], np.ndarray) and k != "t":
            out[k] = np.concatenate([a[k], b[k]], axis=0)
        else:
            out[k] = a[k]
    out["t"] = np.concatenate([a["t"], b["t"]])
    return out


def normalize_time(seq: dict) -> dict:
    seq = dict(seq)
    t = np.asarray(seq["t"], dtype=np.float32)
    seq["t"] = (t - t[0]).astype(np.float32)
    return seq


def subseq(seq: dict, start: int, length: int) -> dict:
    T = seq["t"].shape[0]
    s = max(0, min(int(start), max(0, T - 1)))
    e = min(T, s + int(length))
    out = _take(seq, np.arange(s, e))
    return normalize_time(out)


def stride(seq: dict, factor: int, offset: int = 0) -> dict:
    """Keep every `factor`-th frame (uniform decimation, preserves t)."""
    factor = max(1, int(factor))
    T = seq["t"].shape[0]
    idx = np.arange(offset % factor, T, factor)
    if idx.size < 2:
        idx = np.array([0, max(1, T - 1)])
    return normalize_time(_take(seq, idx))


def drop_random(seq: dict, keep_prob: float, rng: np.random.Generator,
                guarantee_first: bool = True) -> dict:
    T = seq["t"].shape[0]
    keep = rng.random(T) < float(keep_prob)
    if guarantee_first:
        keep[0] = True
    if keep.sum() < 2:
        keep[:min(2, T)] = True
    return normalize_time(_take(seq, np.flatnonzero(keep)))


def drop_burst(seq: dict, rng: np.random.Generator, n_bursts: int = 2,
               burst_frac: float = 0.2) -> dict:
    """Drop contiguous chunks (models loss of a whole observation window)."""
    T = seq["t"].shape[0]
    keep = np.ones(T, dtype=bool)
    for _ in range(int(n_bursts)):
        L = max(1, int(round(burst_frac * T)))
        s = int(rng.integers(1, max(2, T - L)))
        keep[s:s + L] = False
    keep[0] = True
    if keep.sum() < 2:
        keep[:] = True
    return normalize_time(_take(seq, np.flatnonzero(keep)))


def repeat_frames(seq: dict, rng: np.random.Generator, prob: float = 0.3) -> dict:
    """Duplicate some observations; duplicated frame is placed slightly later in time."""
    T = seq["t"].shape[0]
    t = np.asarray(seq["t"], dtype=np.float64)
    if T < 2:
        return normalize_time(dict(seq))
    # nominal frame period
    period = float(np.median(np.diff(t)))
    dup = rng.random(T - 1) < float(prob)
    idx_list = [0]
    for i in range(1, T):
        idx_list.append(i)
        if dup[i - 1]:
            idx_list.append(i)
    idx = np.asarray(idx_list)
    out = _take(seq, idx)
    tt = np.asarray(out["t"], dtype=np.float64).copy()
    # bump duplicated timestamps halfway toward the next distinct time
    for j in range(1, len(tt)):
        if idx[j] == idx[j - 1]:
            tt[j] = tt[j - 1] + 0.5 * period
    tt = np.maximum.accumulate(tt + 1e-6 * np.arange(len(tt)))
    out["t"] = tt.astype(np.float32)
    return normalize_time(out)


def jitter_time(seq: dict, rng: np.random.Generator, sigma: float = 0.25,
                absolute_sigma_s: Optional[float] = None) -> dict:
    """Perturb inter-observation intervals (log-normal multiplicative noise)."""
    t = np.asarray(seq["t"], dtype=np.float64)
    if t.shape[0] < 2:
        return normalize_time(dict(seq))
    dts = np.diff(t)
    noise = rng.normal(0.0, float(sigma), size=dts.shape)
    if absolute_sigma_s:
        dts2 = dts + rng.normal(0.0, float(absolute_sigma_s), size=dts.shape)
        dts2 = np.maximum(dts2, 0.2 * np.median(dts))
    else:
        dts2 = dts * np.exp(noise)
    dts2 = np.maximum(dts2, 1e-4)
    tt = np.concatenate([[0.0], np.cumsum(dts2)])
    out = dict(seq)
    out["t"] = tt.astype(np.float32)
    return out


def occlude_roi(seq: dict, rng: np.random.Generator, frac: float = 0.3,
                dur_frac: float = 0.3, which: str = "random") -> dict:
    """Simulate target occlusion by invalidating ROI observations for an interval.

    `which`: 'random' | 'all' (all ROIs vanish) | int (a single ROI index).
    """
    if "roi_valid" not in seq:
        return dict(seq)
    out = dict(seq)
    rv = np.array(seq["roi_valid"], copy=True)
    T, N = rv.shape
    L = max(1, int(round(dur_frac * T)))
    s = int(rng.integers(0, max(1, T - L)))
    if which == "all":
        cols = np.arange(N)
    elif which == "random":
        cols = np.flatnonzero(rng.random(N) < frac)
        if cols.size == 0:
            cols = np.array([int(rng.integers(0, N))])
    else:
        cols = np.array([int(which) % N])
    for c in cols:
        rv[s:s + L, c] = 0
    out["roi_valid"] = rv
    return out


def zero_tokens(seq: dict, rng: np.random.Generator, frac: float = 0.25) -> dict:
    """Black out a random subset of spatial tokens (simulated partial visibility)."""
    if "tok" not in seq:
        return dict(seq)
    out = dict(seq)
    tok = np.array(seq["tok"], copy=True)
    T, K, _ = tok.shape
    k = max(1, int(round(frac * K)))
    cols = rng.choice(K, size=k, replace=False)
    tok[:, cols, :] = 0.0
    out["tok"] = tok
    return out


# --------------------------------------------------------------------------- #
# composite curricula
# --------------------------------------------------------------------------- #
def compose(*fns: Callable[[dict], dict]) -> Callable[[dict], dict]:
    def _f(seq):
        for fn in fns:
            seq = fn(seq)
        return seq
    return _f


def make_course_fn(level: str, rng_seed: int) -> Callable[[dict, np.random.Generator], dict]:
    """Config-driven augmentation preset.  Used by training and robustness eval.

    level: none | mild | medium | hard | drop30 | drop50 | drop70 | burst
           | repeat | jitter | stride2 | stride4 | mixed
    """
    def _fn(seq, rng):
        if level in ("none", "", None):
            return seq
        if level == "mild":
            seq = drop_random(seq, 0.9, rng)
            seq = jitter_time(seq, rng, 0.1)
        elif level == "medium":
            seq = drop_random(seq, 0.8, rng)
            seq = jitter_time(seq, rng, 0.2)
            seq = repeat_frames(seq, rng, 0.1)
        elif level == "hard":
            seq = drop_random(seq, 0.6, rng)
            seq = jitter_time(seq, rng, 0.35)
            seq = repeat_frames(seq, rng, 0.2)
        elif level == "drop30":
            seq = drop_random(seq, 0.7, rng)
        elif level == "drop50":
            seq = drop_random(seq, 0.5, rng)
        elif level == "drop70":
            seq = drop_random(seq, 0.3, rng)
        elif level == "burst":
            seq = drop_burst(seq, rng, n_bursts=2, burst_frac=0.25)
        elif level == "repeat":
            seq = repeat_frames(seq, rng, 0.4)
        elif level == "jitter":
            seq = jitter_time(seq, rng, 0.4)
        elif level == "stride2":
            seq = stride(seq, 2)
        elif level == "stride4":
            seq = stride(seq, 4)
        elif level == "mixed":
            seq = drop_random(seq, 0.7, rng)
            seq = drop_burst(seq, rng, 1, 0.15)
            seq = repeat_frames(seq, rng, 0.15)
            seq = jitter_time(seq, rng, 0.25)
        else:
            raise ValueError(f"unknown perturbation level {level!r}")
        return seq
    return _fn


def augment_training(seq: dict, rng: np.random.Generator, p: float = 0.7,
                     keep_range=(0.6, 1.0)) -> dict:
    """Random training-time stream augmentation (simulates real streaming)."""
    if rng.random() > p:
        return seq
    kind = rng.integers(0, 5)
    if kind == 0:
        seq = drop_random(seq, float(rng.uniform(*keep_range)), rng)
    elif kind == 1:
        seq = jitter_time(seq, rng, float(rng.uniform(0.05, 0.3)))
    elif kind == 2:
        seq = repeat_frames(seq, rng, float(rng.uniform(0.05, 0.3)))
    elif kind == 3:
        seq = drop_burst(seq, rng, n_bursts=int(rng.integers(1, 3)),
                         burst_frac=float(rng.uniform(0.1, 0.25)))
    else:
        seq = stride(seq, int(rng.integers(2, 3)))
    return seq


# --------------------------------------------------------------------------- #
# multi-rate pairs (for the time-consistency loss / cross-rate metric)
# --------------------------------------------------------------------------- #
def rate_variants(seq: dict, factors=(1, 2, 4)) -> Dict[int, dict]:
    """Same physical process observed at several sampling rates.

    IMPORTANT: this is only meaningful on an UNDISTURBED sequence (dense, regular
    or natively irregular).  It is a *simulated re-sampling of the stream*: the
    decimated version simply lacks the intermediate observations, which is what a
    lower camera frame rate would produce.
    """
    return {int(k): (dict(seq) if k == 1 else stride(seq, int(k))) for k in factors}


def shared_timestamps(t_a: np.ndarray, t_b: np.ndarray, tol: float = 1e-4
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of timestamps present in both sequences within `tol` seconds."""
    t_a = np.asarray(t_a, dtype=np.float64)
    t_b = np.asarray(t_b, dtype=np.float64)
    ia: List[int] = []
    ib: List[int] = []
    j = 0
    for i in range(t_a.size):
        while j < t_b.size and t_b[j] < t_a[i] - tol:
            j += 1
        if j < t_b.size and abs(t_b[j] - t_a[i]) <= tol:
            ia.append(i)
            ib.append(j)
            j += 1
    return np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)


def resample_consistency_pairs(seq: dict, ref: int = 1, factors=(2, 4)
                               ) -> List[dict]:
    """Return a list of {"idx_ref", "idx_alt", "seq_ref", "seq_alt"} for the
    cross-rate consistency objective / metric.

    The loss is `mean |out_ref[idx_ref] - out_alt[idx_alt]|`, plus the same
    in the other direction is NOT used (the decimated stream is information-poor,
    so it must not be treated as an equally valid target for the dense one).
    """
    base = seq if ref == 1 else stride(seq, ref)
    out = []
    for f in factors:
        if f == ref:
            continue
        alt = stride(seq, f)
        ia, ib = shared_timestamps(base["t"], alt["t"])
        if ia.size >= 2:
            out.append({"seq_ref": base, "seq_alt": alt,
                        "idx_ref": ia, "idx_alt": ib, "factor": f})
    return out

