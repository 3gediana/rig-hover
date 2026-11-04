"""Temporal-pattern task family (`pattern_*`).

The point of these tasks: no single frame is sufficient.  The class is a
property of the **whole pulse train** (how many pulses, in which order, how far
apart), and every informative channel is silent at the steps where the class is
scored.  Only a model that integrates the window (a counter / a leaky
integrator / a multi-timescale filter) can answer.

Design rules that make this honest
----------------------------------
1. **The last target pulse is time-aligned across classes** (`anchor_s`).
   Therefore, at the scored steps the amplitude envelope, its derivative, the
   elapsed clock and every distractor channel have the *same distribution for
   every class*.  A memoryless model sees a class-independent input and can
   only guess (uniform chance).  A perfect memory reads the earlier pulses from
   its history and answers exactly.
2. The amplitude envelope **resets** at each pulse, so the trailing envelope
   only reflects the last pulse -- earlier pulses leave no residue that could
   leak the count/order/interval into a single frame.
3. Target pulses are visible while they happen (`feat[4]`, also in `tok`), which
   is exactly what a recurrent model has to remember.  Noise events live on
   separate distractor channels and never change the class.
4. Scoring only happens **after the evidence is complete** (`lab_class >= 0`
   from `anchor + score_margin_s` to the end), so the per-frame label is always
   causally determinable.

Sub-tasks
---------
pattern_count               1 / 2 / 3 target pulses                 (3 classes)
pattern_order               A->B vs B->A                          (2 classes)
pattern_rate                same 2 pulses, fast (0.2 s) vs slow (0.6 s) (2 classes)
pattern_count_irregular     count task + lognormal dt, 30/50/70 % dropout,
                            burst drops, repeat frames, per-episode stretch x1/2/4
pattern_count_rate_mixed    1 pulse / 2 fast / 2 slow / 3 pulses  (4 classes)

Observation layout is byte-identical to every other synthetic dataset
(`feat [T,64] f16`, `tok [T,16,17] f16`, `lab_*`), so the existing model heads
and the evaluator work unchanged.  The pattern-family channel semantics are
documented in `manifest.extra["pattern_channel_layout"]`:

feat channels
    0 bias = 1.0
    1 0.0                      (unused in this family)
    2 0.0                      (unused)
    3 query = 1.0 on scored steps
    4 target pulse = 1.0 at every target-event frame
    5 event type   = 0.0 (A) / 1.0 (B) at target-event frames, 0 otherwise
    6 elapsed physical time / 60 s
    7 seconds since the last NOISE event / 60 s
    8 0.0                      (unused)
    9 background amplitude     (smooth, class-independent)
    10 amplitude envelope      (decay from the most recent target pulse, jittered)
    11 noise-event pulse = 1.0 (distractor, never changes the class)
    12-15 unused (0.0)
    14 0.0                     (unused)
    15 0.0                     (unused)

tok slot 0 = [amplitude, background, target pulse, noise pulse, event type,
              0, 0, ..., 0] / visibility flag = 1.0 (the signal channel is always
              observed); other slots are zero.

lab_state [T,256] at scored steps: slot 0 dims 0..7 = one-hot class
(`n_classes <= 4`), dims 8..15 = (n_pulses/3, span_s/1.0, last_type, 0...);
lab_mask = 1.0 on slot 0 at scored steps only.
lab_class = class at scored steps, -1 elsewhere; lab_event = 1.0 at target
pulses; lab_age[slot 0] = seconds since the last target pulse.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import layout as L
from .base import _enforce_strictly_increasing, stable_hash_int

PATTERN_TASKS: Tuple[str, ...] = (
    "pattern_count",
    "pattern_order",
    "pattern_rate",
    "pattern_count_irregular",
    "pattern_count_rate_mixed",
)

# kind ids stored in meta_pattern
KIND_IDS = {"count": 0, "order": 1, "rate": 2, "mixed": 3}
KIND_NAMES = {v: k for k, v in KIND_IDS.items()}

DEFAULTS: Dict[str, Any] = {
    "dt": 0.1,
    # ---- pattern schedule -------------------------------------------------
    "n_classes": 3,
    "anchor_s": 1.7,          # time of the LAST target pulse (aligned classes)
    "anchor_jitter_s": 0.10,
    "tail_s": 0.90,           # episode ends anchor + tail_s
    "score_margin_s": 0.30,   # first scored step = anchor + margin
    "gap_range_s": [0.35, 0.70],   # order-task inter-pulse gap
    "fast_s": 0.20,           # rate task: fast interval
    "slow_s": 0.60,           # rate task: slow interval
    "early_min_s": 0.30,      # earliest extra pulse (count task)
    # ---- signal -----------------------------------------------------------
    "decay_tau_s": 0.25,
    "pulse_amp": [0.8, 1.2],
    "amp_jitter": 0.04,
    "noise_rate_hz": 0.9,
    "noise_amp": 0.12,
    "bg_amp": 0.25,
    # ---- anti-shortcut perturbations for the BASE datasets ----------------
    "p_repeat": 0.05,
    # ---- irregular / stretch variants -------------------------------------
    "irregular": None,        # None | "mixed" | "logdt" | "drop50" | "burst"
    "drop_choices": [0.3, 0.5, 0.7],
    "logdt_cv": 0.40,
    "stretch": 1.0,           # physical-time stretch of the whole episode
}


def pattern_defaults(task: str) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    if task in ("pattern_count", "pattern_count_irregular",
                "pattern_count_stretch2", "pattern_count_stretch4"):
        out.update({"n_classes": 3, "kind": "count"})
    elif task == "pattern_order":
        out.update({"n_classes": 2, "kind": "order"})
    elif task in ("pattern_rate", "pattern_rate_stretch2", "pattern_rate_stretch4"):
        out.update({"n_classes": 2, "kind": "rate"})
    elif task == "pattern_count_rate_mixed":
        out.update({"n_classes": 4, "kind": "mixed"})
    else:
        raise KeyError(f"unknown pattern task {task!r}")
    return out


def is_pattern_task(task: str) -> bool:
    t = str(task)
    return t in PATTERN_TASKS or t.startswith("pattern_")


def _kind(task: str, params: Dict[str, Any]) -> str:
    k = params.get("kind")
    if k:
        return str(k)
    return pattern_defaults(task).get("kind", "count")


# --------------------------------------------------------------------------- #
# balanced class assignment (deterministic, per split)
# --------------------------------------------------------------------------- #
def class_permutation(seed_base: int, dataset_rng_name: str, split: str,
                      n_classes: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed_base) & 0xFFFFFFFF,
        stable_hash_int("pattern-class:" + str(dataset_rng_name)),
        stable_hash_int("split:" + str(split)),
    ]))
    return rng.permutation(int(n_classes))


def class_for_episode(cfg: Dict[str, Any], spec, split: str,
                      episode: int) -> int:
    """Balanced class id for one episode (round-robin over a fixed permutation).

    Every dataset derived from the same base pattern dataset (irregular,
    stretch, shuffled) uses the same mapping, so classes stay balanced and the
    episodes are paired across variants.
    """
    params = spec.params
    n_classes = int(params.get("n_classes", 3))
    lo = int(cfg["split_ranges"][split][0])
    rel = int(episode) - lo
    perm = class_permutation(int(cfg["seed_base"]), spec.rng_name or spec.name,
                             split, n_classes)
    return int(perm[rel % n_classes])


# --------------------------------------------------------------------------- #
# schedule sampling
# --------------------------------------------------------------------------- #
def _sample_schedule(rng: np.random.Generator, params: Dict[str, Any],
                     class_id: int) -> Dict[str, Any]:
    kind = str(params.get("kind", "count"))
    ndt = float(params["dt"])
    anchor = float(params["anchor_s"]) + float(
        rng.uniform(-params["anchor_jitter_s"], params["anchor_jitter_s"]))
    anchor = max(anchor, 0.6)

    times: List[float] = []
    types: List[int] = []
    if kind == "count":
        n = int(class_id) + 1                      # 1..3
        # extra pulses BEFORE the aligned anchor pulse (spacing ~0.4-0.7 s);
        # the anchor is NOT adjusted per class, otherwise the trailing envelope
        # would leak the class into a single frame
        offsets = []
        for i in range(n - 1):
            off = 0.42 + 0.22 * i + float(rng.uniform(-0.05, 0.05))
            offsets.append(off)
        offsets = sorted(offsets, reverse=True)
        for off in offsets:
            times.append(anchor - off)
        times.append(anchor)
        types = [0] * n
    elif kind == "order":
        gap = float(rng.uniform(*params["gap_range_s"]))
        times = [anchor - gap, anchor]
        types = [0, 1] if int(class_id) == 0 else [1, 0]   # A->B or B->A
    elif kind == "rate":
        delta = float(params["fast_s"]) if int(class_id) == 0 else float(params["slow_s"])
        times = [anchor - delta, anchor]
        types = [0, 0]
    elif kind == "mixed":
        c = int(class_id)
        if c == 0:                                   # 1 pulse
            times = [anchor]
            types = [0]
        elif c == 1:                                 # 2 pulses, fast
            times = [anchor - float(params["fast_s"]), anchor]
            types = [0, 0]
        elif c == 2:                                 # 2 pulses, slow
            times = [anchor - float(params["slow_s"]), anchor]
            types = [0, 0]
        else:                                        # 3 pulses
            times = [anchor - 0.85, anchor - 0.42, anchor]
            types = [0, 0, 0]
    else:  # pragma: no cover
        raise KeyError(f"unknown pattern kind {kind!r}")

    pulse_times = np.asarray(sorted(times), dtype=np.float64)
    # types follow the (already sorted) time order: recompute order for order-task
    if kind == "order":
        # `times` was already ascending for both classes
        types_sorted = list(types)
    else:
        types_sorted = [0] * len(pulse_times)
    t_first = float(pulse_times.min())
    t_last = float(pulse_times.max())
    t_end = t_last + float(params["tail_s"])
    amps = rng.uniform(params["pulse_amp"][0], params["pulse_amp"][1],
                       size=pulse_times.size).astype(np.float64)
    # noise events: independent Poisson process over [0, t_end].  They are NOT
    # filtered near target pulses on purpose: any class-dependent rejection
    # would correlate the distractor count with the class (a weak leak).
    rate = float(params["noise_rate_hz"])
    n_noise = int(rng.poisson(rate * t_end))
    noise_times = sorted(float(rng.uniform(0.0, t_end)) for _ in range(n_noise))
    return {
        "class_id": int(class_id),
        "kind": kind,
        "n_pulses": int(pulse_times.size),
        "pulse_times": pulse_times,
        "pulse_types": np.asarray(types_sorted, dtype=np.int64),
        "pulse_amps": amps,
        "anchor": float(anchor),
        "t_first": t_first,
        "t_last": t_last,
        "span_s": float(t_last - t_first),
        "t_end": float(t_end),
        "noise_times": np.asarray(sorted(noise_times), dtype=np.float64),
        "bg_phase": float(rng.uniform(0, 2 * math.pi)),
    }


# --------------------------------------------------------------------------- #
# time grids + irregular sampling
# --------------------------------------------------------------------------- #
def _regular_times(t_end: float, dt: float) -> np.ndarray:
    n = int(math.ceil(t_end / dt)) + 1
    t = np.arange(n, dtype=np.float64) * dt
    return t


def _duplicate_times(times: np.ndarray, protected: np.ndarray,
                     rng: np.random.Generator, p: float) -> np.ndarray:
    """Duplicate a random subset of UNPROTECTED frames (in place, bumped t)."""
    T = times.size
    if T < 3 or p <= 0:
        return times
    out: List[float] = [float(times[0])]
    for i in range(1, T):
        out.append(float(times[i]))
        if not protected[i] and rng.random() < p:
            out.append(float(times[i]) + 0.5 * (times[i] - times[i - 1]))
    return np.asarray(out, dtype=np.float64)


def _drop_times(times: np.ndarray, protected: np.ndarray,
                rng: np.random.Generator, drop_frac: float) -> np.ndarray:
    keep = rng.random(times.size) >= drop_frac
    keep |= protected
    if int(keep.sum()) < max(6, int(0.3 * times.size)):
        keep = protected.copy()
    return times[keep]


def _burst_drop(times: np.ndarray, protected: np.ndarray,
                rng: np.random.Generator, n_bursts: int = 2,
                frac: float = 0.2) -> np.ndarray:
    T = times.size
    keep = np.ones(T, dtype=bool)
    for _ in range(n_bursts):
        Lb = max(1, int(round(frac * T)))
        s = int(rng.integers(1, max(2, T - Lb)))
        keep[s:s + Lb] = False
    keep |= protected
    if int(keep.sum()) < max(6, int(0.3 * T)):
        keep = protected.copy()
    return times[keep]


def _protect_mask(times: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    """Frames that must survive irregular sampling: pulses + scoring region."""
    prot = np.zeros(times.size, dtype=bool)
    for pt in spec["pulse_times"]:
        prot[int(np.abs(times - pt).argmin())] = True
    prot |= times >= spec["anchor"] - 1e-9
    return prot


def _irregular_times(times: np.ndarray, rng: np.random.Generator,
                     params: Dict[str, Any], spec: Dict[str, Any]) -> np.ndarray:
    """Native irregular sampling: lognormal dt -> dropout -> burst drops.

    Target pulses and the whole scoring region are protected (they are strong
    signals that a real sensor would not miss); the *history* frames are dropped
    so the model must integrate an irregular stream.
    """
    mode = str(params.get("irregular") or "")
    if not mode:
        return times
    dt = float(params["dt"])
    if mode in ("mixed", "logdt"):
        cv = float(params["logdt_cv"])
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        dts = np.diff(times) * np.exp(rng.normal(0.0, sigma, size=times.size - 1))
        dts = np.clip(dts, 0.3 * dt, 3.0 * dt)
        tt = np.concatenate([[times[0]], times[0] + np.cumsum(dts)])
        # re-insert the exact pulse times so every knock has its own frame
        tt = np.union1d(tt, spec["pulse_times"])
        times = _enforce_strictly_increasing(tt)
    if mode in ("mixed", "drop30", "drop50", "drop70"):
        frac = {"drop30": 0.3, "drop50": 0.5, "drop70": 0.7}.get(mode, 0.0)
        if mode == "mixed":
            frac = float(rng.choice(list(params["drop_choices"])))
        times = _drop_times(times, _protect_mask(times, spec), rng, frac)
    if mode in ("mixed", "burst"):
        times = _burst_drop(times, _protect_mask(times, spec), rng,
                            n_bursts=2, frac=0.18)
    return np.asarray(times, dtype=np.float64)


def _build_times(spec: Dict[str, Any], params: Dict[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
    """Physical observation times for the whole episode (strictly increasing)."""
    dt = float(params["dt"])
    times = _regular_times(spec["t_end"], dt)
    # always guarantee a frame exactly at every target pulse: the event itself is
    # never lost to sampling (a knock is a strong signal, not a sampling artifact)
    times = np.union1d(times, spec["pulse_times"])
    times = _enforce_strictly_increasing(times)

    if params.get("irregular"):
        times = _irregular_times(times, rng, params, spec)
        times = _enforce_strictly_increasing(times)
    else:
        protected = _protect_mask(times, spec)
        times = _duplicate_times(times, protected, rng,
                                 float(params.get("p_repeat", 0.0)))
        times = _enforce_strictly_increasing(times)

    # NOTE: the clock is NOT re-zeroed here: the schedule (pulse times, anchor,
    # noise events) is expressed in the same absolute physical clock and every
    # perturbation is applied to both, so they must stay in sync.
    stretch = float(params.get("stretch", 1.0) or 1.0)
    if stretch != 1.0:
        times = _enforce_strictly_increasing(times * stretch).astype(np.float64)
    return np.asarray(times, dtype=np.float64)


# --------------------------------------------------------------------------- #
# rendering: physical times + schedule -> observation / label tensors
# --------------------------------------------------------------------------- #
def _nearest_idx(times: np.ndarray, t: float) -> int:
    return int(np.abs(times - t).argmin())


def render_pattern_seq(times: np.ndarray, spec: Dict[str, Any],
                       params: Dict[str, Any], rng: np.random.Generator,
                       stretch: float = 1.0) -> Dict[str, np.ndarray]:
    """Observation + label tensors for one pattern episode."""
    t = np.asarray(times, dtype=np.float64)
    T = t.size
    dt = float(params["dt"])
    dt_eff = float(np.median(np.diff(t))) if T > 1 else dt
    K = L.KMAX
    V = L.V

    # ---- pulse frames (physical times are preserved by every perturbation) --
    # all schedule times are expressed in the SAME stretched physical clock as
    # the observation times (stretch=1 for the base datasets)
    pt_phys = np.asarray(spec["pulse_times"], dtype=np.float64) * float(stretch)
    pulse_frame = np.full(pt_phys.size, -1, dtype=np.int64)
    for i, pt in enumerate(pt_phys):
        pulse_frame[i] = _nearest_idx(t, pt)
    pulse_frame = np.unique(pulse_frame)          # collisions cannot happen
    n_seen = pulse_frame.size
    # event type per pulse frame: derive from the schedule, matched by time
    type_at = np.zeros(T, dtype=np.float32)
    for i, pf in enumerate(pulse_frame):
        pt = t[pf]
        j = int(np.abs(pt_phys - pt).argmin())
        type_at[pf] = float(spec["pulse_types"][j])

    # ---- amplitude envelope: reset at each pulse (no leakage of earlier ones)
    amp = np.zeros(T, dtype=np.float64)
    pulse_t_eff = t[pulse_frame]
    pulse_a_eff = spec["pulse_amps"][
        [int(np.abs(pt_phys - x).argmin()) for x in pulse_t_eff]]
    tau = float(params["decay_tau_s"]) * stretch
    j = np.searchsorted(pulse_t_eff, t, side="right") - 1
    ok = j >= 0
    amp[ok] = pulse_a_eff[j[ok]] * np.exp(-(t[ok] - pulse_t_eff[j[ok]]) / tau)
    # distractor noise events (small bumps, class-independent)
    noise_times = np.asarray(spec["noise_times"], dtype=np.float64) * stretch
    noise_frame = np.zeros(T, dtype=bool)
    noise_amp = float(params["noise_amp"])
    tau_n = 0.15 * stretch
    for tn in noise_times:
        idx = _nearest_idx(t, tn)
        noise_frame[idx] = True
        tn_eff = t[idx]
        active = t >= tn_eff
        amp[active] += noise_amp * np.exp(-(t[active] - tn_eff) / tau_n)
    # smooth class-independent background + per-frame amplitude jitter
    ph = float(spec["bg_phase"])
    bg = float(params["bg_amp"]) * np.abs(np.sin(1.9 * t / max(stretch, 1e-9) + ph))
    amp_obs = amp + 0.25 * bg
    amp_obs = amp_obs * (1.0 + float(params["amp_jitter"])
                         * rng.standard_normal(T))
    amp_obs = np.clip(amp_obs, 0.0, None)

    # ---- scored region: only after the evidence is complete -----------------
    anchor = float(spec["anchor"]) * stretch
    margin = float(params["score_margin_s"]) * stretch
    scored = t >= anchor + margin - 1e-9
    if scored.sum() == 0:  # pragma: no cover - defensive
        scored[-1] = True
    lab_mask = np.zeros((T, K), dtype=np.float32)
    lab_mask[:, 0] = scored.astype(np.float32)
    lab_class = np.full(T, -1, dtype=np.int64)
    lab_class[scored] = int(spec["class_id"])

    # ---- state (slot 0 = class one-hot + schedule summary) ------------------
    n_classes = int(params["n_classes"])
    lab_state = np.zeros((T, K * V), dtype=np.float32)
    st = np.zeros(V, dtype=np.float32)
    if n_classes > 8:  # pragma: no cover - never happens
        raise ValueError("pattern class one-hot needs <= 8 dims")
    st[int(spec["class_id"])] = 1.0
    st[8] = float(spec["n_pulses"]) / 3.0
    st[9] = float(spec["span_s"]) / 1.0
    st[10] = float(np.sum(spec["pulse_types"]))       # last/second type marker
    lab_state[scored, :V] = st

    # ---- feat ---------------------------------------------------------------
    feat = np.zeros((T, L.DG), dtype=np.float32)
    feat[:, L.CH_BIAS] = 1.0
    feat[:, L.CH_QUERY] = scored.astype(np.float32)
    feat[pulse_frame, L.CH_EVENT] = 1.0
    feat[pulse_frame, L.CH_EVENT_TYPE] = type_at[pulse_frame]
    feat[:, L.CH_ELAPSED] = (t / 60.0).astype(np.float32)
    # noise-event age
    last_noise = np.maximum.accumulate(
        np.where(noise_frame, np.arange(T), -1).astype(np.int64))
    noise_age = np.where(last_noise >= 0,
                         t - t[np.clip(last_noise, 0, T - 1)], 0.0)
    feat[:, L.CH_EVENT_AGE] = (noise_age / 60.0).astype(np.float32)
    feat[:, L.CH_HIDDEN_ASKED] = bg.astype(np.float32)             # ch9: bg amp
    feat[:, 10] = amp_obs.astype(np.float32)                       # envelope
    feat[:, 11] = noise_frame.astype(np.float32)                   # noise pulse
    # ch12..15 stay unused on purpose: a finite-difference derivative blows up
    # on near-duplicate timestamps and would give a (class-independent but)
    # exploitable tail, so we do not publish it as an observation channel.

    # ---- tok: the signal channel (always observed) -------------------------
    tok = np.zeros((T, K, L.TOK_DIM), dtype=np.float16)
    tok[:, 0, 0] = amp_obs.astype(np.float16)
    tok[:, 0, 1] = bg.astype(np.float16)
    tok[:, 0, 2] = (np.zeros(T) + feat[:, L.CH_EVENT]).astype(np.float16)
    tok[:, 0, 3] = noise_frame.astype(np.float16)
    tok[:, 0, 4] = type_at.astype(np.float16)
    tok[:, 0, L.V] = 1.0

    # ---- lab_event / lab_age / lab_gap_s -----------------------------------
    lab_event = feat[:, L.CH_EVENT].astype(np.float32)
    lab_age = np.zeros((T, K), dtype=np.float32)
    lj = np.searchsorted(pulse_t_eff, t, side="right") - 1
    age = np.where(lj >= 0, t - pulse_t_eff[np.clip(lj, 0, max(T - 1, 0))], 0.0)
    lab_age[:, 0] = age.astype(np.float32)          # physical seconds
    span_phys = float(spec["span_s"]) * float(stretch)
    lab_gap_s = np.full(T, span_phys, dtype=np.float32)

    # ---- meta --------------------------------------------------------------
    phase = np.where(scored, 1, 0).astype(np.int8)
    flags = np.where(scored, np.int8(L.FLAG_QUERY), np.int8(0))
    flags = flags | (noise_frame.astype(np.int8) * np.int8(L.FLAG_EVENT))
    n_classes = int(params["n_classes"])
    kind = str(params.get("kind", "count"))
    meta_pattern = np.asarray([[n_classes, float(KIND_IDS.get(kind, 0)),
                                float(stretch),
                                1.0 if params.get("irregular") else 0.0]],
                              dtype=np.float32)
    mix = np.asarray([L.V, L.VC, L.N_CODES, L.C_EVENT, L.KMAX, L.LAYOUT_VERSION],
                     dtype=np.float32).reshape(1, 6)
    return {
        "feat": feat.astype(np.float16),
        "t": t.astype(np.float32),
        "tok": tok,
        "lab_state": lab_state,
        "lab_mask": lab_mask,
        "lab_class": lab_class,
        "lab_event": lab_event,
        "lab_age": lab_age,
        "lab_gap_s": lab_gap_s,
        "meta_phase": phase,
        "meta_flags": flags,
        "meta_k": np.full(T, 1, dtype=np.int8),
        "meta_gap_s": np.full(T, span_phys, dtype=np.float32),
        "meta_cross": np.full((1, 5), -1.0, dtype=np.float32),
        "meta_mix": mix,
        "meta_dt0": np.asarray([float(dt_eff)], dtype=np.float32),
        "meta_pattern": meta_pattern,
    }


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def build_pattern_seq(rng: np.random.Generator, params: Dict[str, Any],
                      class_id: int) -> Dict[str, np.ndarray]:
    """One complete pattern episode as an npz-compatible dict."""
    spec = _sample_schedule(rng, params, int(class_id))
    times = _build_times(spec, params, rng)
    return render_pattern_seq(times, spec, params, rng,
                              stretch=float(params.get("stretch", 1.0) or 1.0))


def pattern_channel_layout() -> Dict[str, Any]:
    """Documentation of the pattern-family channel semantics (for manifests)."""
    return {
        "family": "pattern_*",
        "rule": "no single frame is sufficient; the class is a property of the "
                "whole pulse train; the last target pulse is time-aligned across "
                "classes so every non-memory channel is class-independent at the "
                "scored steps",
        "feat": {
            "shape": "[T, 64] float16 (same tensor shape as all synth datasets)",
            "0": "bias = 1.0",
            "1": "0.0 (unused in this family)",
            "2": "0.0 (unused)",
            "3": "query = 1.0 on scored steps (evidence complete)",
            "4": "target pulse = 1.0 at target-event frames",
            "5": "event type = 0.0 (A) / 1.0 (B) at target pulses (pattern_order)",
            "6": "elapsed physical time / 60 s",
            "7": "seconds since the last NOISE event / 60 s",
            "8": "0.0 (unused)",
            "9": "background amplitude (smooth, class-independent)",
            "10": "amplitude envelope (decay from the most recent target pulse)",
            "11": "noise-event pulse = 1.0 (distractor, never changes the class)",
            "12-15": "0.0 (unused; no finite-difference channels, they would give "
                     "an exploitable tail on near-duplicate timestamps)",
        },
        "tok": {
            "slot0": "[amplitude, background, target_pulse, noise_pulse, "
                     "event_type, 0 x 11], visibility flag = 1.0 (always observed)",
            "slots1..15": "zero (unused)",
        },
        "labels": {
            "lab_state": "slot0 dims 0..7 = one-hot class; dims 8..15 = "
                         "(n_pulses/3, span_s, last_type_marker, 0...)",
            "lab_mask": "1.0 on slot 0 at scored steps only",
            "lab_class": "class id at scored steps, -1 elsewhere (main metric acc)",
            "lab_event": "1.0 at target pulse frames",
            "lab_age": "slot0 = physical seconds since the last target pulse",
            "lab_gap_s": "pattern span (last - first pulse), physical seconds",
        },
        "meta_pattern": "[1,4] float: (n_classes, kind_id, stretch, is_irregular)",
        "kinds": KIND_NAMES,
    }


def pattern_class_from_state(state: np.ndarray, n_classes: int) -> np.ndarray:
    """Decode a class id from the slot-0 state vector (one-hot in dims 0..7)."""
    s = np.asarray(state, dtype=np.float32)
    flat = s.reshape(s.shape[0], -1)[:, :max(int(n_classes), 2)]
    return flat.argmax(-1).astype(np.int64)


def pattern_n_classes(seq: Dict[str, np.ndarray]) -> int:
    mp = seq.get("meta_pattern")
    if mp is not None and np.asarray(mp).size >= 1:
        return int(round(float(np.asarray(mp).reshape(-1)[0])))
    cls = np.asarray(seq["lab_class"])
    return int(cls.max()) + 1 if cls.max() >= 0 else 2
