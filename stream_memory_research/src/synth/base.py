"""Sequence assembly for the synthetic memory benchmark.

Pipeline
--------
    task.build(rng, episode, params) -> RawEpisode          # latent truth only
    raw_to_seq(raw)                  -> dict of npz arrays # feat/tok/labels/meta
    (optional) perturb_raw(raw, ...) -> RawEpisode          # irregular sampling
    save_seq(dict, path)                                    # npz

Design rule that makes the benchmark honest (DESIGN §1.6): the observation
tensors only ever contain what is visible at the CURRENT step; `lab_state`
always contains the true latent state, and `lab_mask` marks the steps where
the true state is recoverable from the observation history but absent from the
current observation.  A memoryless model therefore scores ~0 on exactly the
masked steps and ~1.0 nowhere.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import layout as L


# --------------------------------------------------------------------------- #
# raw episode (latent truth, before observation encoding)
# --------------------------------------------------------------------------- #
@dataclass
class RawEpisode:
    t: np.ndarray            # [T] float32, strictly increasing (starts at 0)
    vis: np.ndarray          # [T, KMAX] bool   token observed
    lat: np.ndarray          # [T, KMAX, V] float32  true latent state
    idmask: np.ndarray       # [T, KMAX] bool   identity dims not observable
    query: np.ndarray        # [T] bool
    event: np.ndarray        # [T] int8, 0 = none, 1..C_EVENT-1 = type
    phase: np.ndarray        # [T] int8
    cls: np.ndarray          # [T] int64, -1 ignore
    K: int                   # number of existing object slots
    gap_s: float = -1.0      # designated hidden-gap length (sweep datasets)
    cross: Optional[np.ndarray] = None   # [Nc, 5] float32 crosstalk spec
    meta: Dict[str, Any] = None          # free-form generator metadata

    def copy(self) -> "RawEpisode":
        return RawEpisode(
            t=self.t.copy(), vis=self.vis.copy(), lat=self.lat.copy(),
            idmask=self.idmask.copy(), query=self.query.copy(),
            event=self.event.copy(), phase=self.phase.copy(), cls=self.cls.copy(),
            K=int(self.K), gap_s=float(self.gap_s),
            cross=None if self.cross is None else self.cross.copy(),
            meta=dict(self.meta or {}),
        )

    def __len__(self) -> int:
        return int(self.t.shape[0])


class Timeline:
    """Append-only helper that keeps the latent state for task generators.

    Usage::

        tl = Timeline(K=2, V=16, rng=rng)
        tl.set_random(0); tl.set_random(1)
        tl.append(4, vis=[0, 1], phase=1)     # write phase, both observed
        tl.append(8, vis=[], phase=2)         # hidden gap
        tl.set_random(0)                      # update slot 0 while visible
        tl.append(3, vis=[0], phase=3)
        raw = tl.finish()
    """

    def __init__(self, K: int, rng: np.random.Generator, dt: float = 0.1,
                 V: int = L.V):
        self.K = int(K)
        self.rng = rng
        self.dt = float(dt)
        self.V = int(V)
        self._t: List[float] = []
        self._vis: List[np.ndarray] = []
        self._lat: List[np.ndarray] = []
        self._idmask: List[np.ndarray] = []
        self._query: List[bool] = []
        self._event: List[int] = []
        self._phase: List[int] = []
        self._cls: List[int] = []
        self._cross: List[Tuple[float, float, float, int, int]] = []
        self.value = np.zeros((L.KMAX, self.V), dtype=np.float32)
        self._has_value = np.zeros(L.KMAX, dtype=bool)

    # ---- state manipulation ------------------------------------------------
    def set_state(self, k: int, state: np.ndarray) -> np.ndarray:
        st = np.asarray(state, dtype=np.float32).reshape(self.V)
        self.value[k] = st
        self._has_value[k] = True
        return st

    def set_random(self, k: int) -> np.ndarray:
        st = L.random_state(self.rng)
        return self.set_state(k, st)

    def set_code(self, k: int, code: int) -> np.ndarray:
        st = self.value[k].copy()
        st[:L.VC] = L.code_to_vec(code)
        st[L.VC:] = self.rng.standard_normal(L.VA).astype(np.float32)
        return self.set_state(k, st)

    def current(self, k: int) -> np.ndarray:
        return self.value[k].copy()

    # ---- frame appending ---------------------------------------------------
    def append(self, n: int, vis: Sequence[int] = (), phase: int = 0,
               query: bool = False, event: int = 0,
               idmask: Sequence[int] = (), t0: Optional[float] = None,
               dt: Optional[float] = None,
               cls: int = -1) -> np.ndarray:
        """Append `n` frames; returns the per-frame times that were appended."""
        n = int(n)
        dt = float(self.dt if dt is None else dt)
        vis_set = set(int(v) for v in vis)
        mask_set = set(int(v) for v in idmask)
        times = np.empty(n, dtype=np.float64)
        start = self.t_next if t0 is None else float(t0)
        for i in range(n):
            tt = start + i * dt
            times[i] = tt
            vrow = np.zeros(L.KMAX, dtype=bool)
            mrow = np.zeros(L.KMAX, dtype=bool)
            for k in vis_set:
                vrow[k] = True
            for k in mask_set:
                mrow[k] = True
            self._t.append(float(tt))
            self._vis.append(vrow)
            self._lat.append(self.value.copy())
            self._idmask.append(mrow)
            self._query.append(bool(query))
            self._event.append(int(event))
            self._phase.append(int(phase))
            self._cls.append(int(cls))
        return times.astype(np.float32)

    def append_frames(self, n: int, phase: int, vis_for_frame: Callable[[int], Sequence[int]],
                      query_for_frame: Optional[Callable[[int], bool]] = None,
                      event_for_frame: Optional[Callable[[int], int]] = None,
                      idmask_for_frame: Optional[Callable[[int], Sequence[int]]] = None,
                      cls_for_frame: Optional[Callable[[int], int]] = None) -> None:
        """Append frames whose visibility/query/... are computed per frame index."""
        n = int(n)
        for i in range(n):
            self.append(
                1, vis=list(vis_for_frame(i)), phase=phase,
                query=bool(query_for_frame(i)) if query_for_frame else False,
                event=int(event_for_frame(i)) if event_for_frame else 0,
                idmask=list(idmask_for_frame(i)) if idmask_for_frame else (),
                cls=int(cls_for_frame(i)) if cls_for_frame else -1,
            )

    @property
    def t_next(self) -> float:
        if not self._t:
            return 0.0
        return float(self._t[-1] + self.dt)

    @property
    def n_frames(self) -> int:
        return len(self._t)

    def t_at(self, frame_idx: int) -> float:
        return float(self._t[int(frame_idx)])

    def add_cross(self, t_prev: float, t_write: float, t_post: float,
                  writer: int, reader: int) -> None:
        self._cross.append((float(t_prev), float(t_write), float(t_post),
                            int(writer), int(reader)))

    # ---- finalise ----------------------------------------------------------
    def finish(self, gap_s: float = -1.0, meta: Optional[Dict[str, Any]] = None,
               cls_override: Optional[Callable[[int], int]] = None) -> RawEpisode:
        T = len(self._t)
        t = np.asarray(self._t, dtype=np.float32)
        # strictly increasing (guard against any generator slip)
        if T > 1 and not np.all(np.diff(t) > 0):
            t = _enforce_strictly_increasing(t)
        cls = np.asarray(self._cls, dtype=np.int64)
        if cls_override is not None:
            cls = np.asarray([cls_override(i) for i in range(T)], dtype=np.int64)
        cross = (np.asarray(self._cross, dtype=np.float32)
                 if self._cross else None)
        return RawEpisode(
            t=t,
            vis=np.stack(self._vis) if T else np.zeros((0, L.KMAX), bool),
            lat=np.stack(self._lat) if T else np.zeros((0, L.KMAX, self.V), np.float32),
            idmask=np.stack(self._idmask) if T else np.zeros((0, L.KMAX), bool),
            query=np.asarray(self._query, dtype=bool),
            event=np.asarray(self._event, dtype=np.int8),
            phase=np.asarray(self._phase, dtype=np.int8),
            cls=cls, K=int(self.K), gap_s=float(gap_s), cross=cross,
            meta=dict(meta or {}),
        )


def _enforce_strictly_increasing(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).copy()
    for i in range(1, t.size):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-4
    return t.astype(np.float32)


# --------------------------------------------------------------------------- #
# observability
# --------------------------------------------------------------------------- #
def compute_observability(raw: RawEpisode) -> Dict[str, np.ndarray]:
    """Replay the observation stream to find what a perfect memory could know.

    Returns
    -------
    observable : [T, KMAX] uint8     value of slot k at step t is recoverable
                                     from observations up to and including t
    full_obs   : [T, KMAX] bool      slot k is fully observed AT step t
    mask       : [T, KMAX] uint8     1 iff observable AND not fully observed now
                                     (== the "asked for / memory-only" mask)
    reveal     : [T, KMAX] uint8     1 at steps where a new value first becomes
                                     observable (write / first sight / change)
    """
    T = int(raw.t.shape[0])
    K = L.KMAX
    vis = np.asarray(raw.vis, dtype=bool)
    idmask = np.asarray(raw.idmask, dtype=bool)
    full = vis & ~idmask

    # last fully-observed frame index per (t, k)
    idx = np.where(full, np.arange(T, dtype=np.int64)[:, None], np.int64(-1))
    last = np.maximum.accumulate(idx, axis=0)          # [T, K]
    observable = last >= 0
    safe = np.clip(last, 0, max(T - 1, 0))
    if T > 0:
        seen = np.take_along_axis(raw.lat, safe[..., None], axis=0)   # [T,K,V]
        observable &= np.all(seen == raw.lat, axis=-1)
    else:  # pragma: no cover
        observable = np.zeros((0, K), bool)

    # value changed vs the value that was known at the previous step
    prev_last = np.full((T, K), -1, dtype=np.int64)
    if T > 1:
        prev_last[1:] = last[:-1]
    prev_safe = np.clip(prev_last, 0, max(T - 1, 0))
    if T > 0:
        prev_seen = np.take_along_axis(raw.lat, prev_safe[..., None], axis=0)
        changed = np.any(raw.lat != prev_seen, axis=-1)
    else:  # pragma: no cover
        changed = np.zeros((0, K), bool)
    reveal = full & (prev_last < 0) | (full & (prev_last >= 0) & changed)
    reveal = np.asarray(reveal, dtype=bool)

    mask = observable & ~full
    return {
        "observable": observable.astype(np.uint8),
        "full_obs": full,
        "mask": mask,
        "reveal": reveal.astype(np.uint8),
    }


# --------------------------------------------------------------------------- #
# observation encoding
# --------------------------------------------------------------------------- #
def encode_observations(raw: RawEpisode, info: Dict[str, np.ndarray],
                        dt0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build feat [T,64] f16 and tok [T,16,17] f16 from a RawEpisode."""
    T = int(raw.t.shape[0])
    K = L.KMAX
    vis = np.asarray(raw.vis, dtype=bool)
    idmask = np.asarray(raw.idmask, dtype=bool)
    full = np.asarray(info["full_obs"], dtype=bool)
    observable = np.asarray(info["observable"], dtype=bool)
    mask = np.asarray(info["mask"], dtype=bool)
    t = np.asarray(raw.t, dtype=np.float64)

    # ---- tok: observed value (identity dims zeroed when masked) + vis flag --
    obs_val = raw.lat.astype(np.float32).copy()
    obs_val[..., :L.VC] = np.where(idmask[..., None], np.float32(0.0),
                                   obs_val[..., :L.VC])
    obs_val = obs_val * vis[..., None].astype(np.float32)
    tok = np.zeros((T, K, L.TOK_DIM), dtype=np.float16)
    if T:
        tok[..., :L.V] = obs_val.astype(np.float16)
        tok[..., L.V] = vis.astype(np.float16)

    # ---- feat: coarse global summary --------------------------------------
    feat = np.zeros((T, L.DG), dtype=np.float32)
    if T:
        n_vis = vis.sum(1).astype(np.float32)
        n_exist = np.minimum(raw.K, K)
        feat[:, L.CH_BIAS] = 1.0
        feat[:, L.CH_VIS_FRAC] = n_vis / float(K)
        existing = np.zeros(K, dtype=bool)
        existing[:int(raw.K)] = True
        feat[:, L.CH_ANY_HIDDEN] = (np.any(existing[None, :] & ~vis, axis=1)).astype(np.float32)
        feat[:, L.CH_QUERY] = raw.query.astype(np.float32)
        feat[:, L.CH_EVENT] = (raw.event > 0).astype(np.float32)
        feat[:, L.CH_EVENT_TYPE] = (np.maximum(raw.event, 0).astype(np.float32)
                                    / max(L.C_EVENT - 1, 1))
        feat[:, L.CH_ELAPSED] = (t / 60.0).astype(np.float32)
        feat[:, L.CH_N_EXIST] = float(n_exist) / float(K)
        # event age
        last_ev = np.maximum.accumulate(
            np.where(raw.event > 0, np.arange(T), -1).astype(np.int64))
        age = np.where(last_ev >= 0, t - t[np.clip(last_ev, 0, T - 1)], 0.0)
        feat[:, L.CH_EVENT_AGE] = (age / 60.0).astype(np.float32)
        # time since any observed token value changed
        change_now = np.zeros(T, dtype=bool)
        if T > 1:
            both = full[1:] & full[:-1]
            diff = np.any(obs_val[1:] != obs_val[:-1], axis=-1)
            change_now[1:] = np.any(both & diff, axis=1)
        last_ch = np.maximum.accumulate(
            np.where(change_now, np.arange(T), -1).astype(np.int64))
        since = np.where(last_ch >= 0, t - t[np.clip(last_ch, 0, T - 1)], 0.0)
        feat[:, L.CH_SINCE_CHANGE] = (since / 60.0).astype(np.float32)
        feat[:, L.CH_HIDDEN_ASKED] = mask.sum(1).astype(np.float32) / float(K)
        n_obs_norm = np.linalg.norm(obs_val, axis=-1) * vis.astype(np.float32)
        feat[:, L.CH_MEAN_VIS_NORM] = (n_obs_norm.sum(1)
                                       / np.maximum(n_vis, 1.0)).astype(np.float32)
    return feat.astype(np.float16), tok


def raw_to_seq(raw: RawEpisode, dt0: float = 0.1) -> Dict[str, np.ndarray]:
    """RawEpisode -> complete npz array dict (observations + labels + meta)."""
    T = int(raw.t.shape[0])
    info = compute_observability(raw)
    feat, tok = encode_observations(raw, info, dt0)

    observable = np.asarray(info["observable"], dtype=bool)
    mask_b = np.asarray(info["mask"], dtype=bool)
    existing = np.zeros(L.KMAX, dtype=bool)
    existing[:int(raw.K)] = True
    known = observable & existing[None, :]
    ask = mask_b & existing[None, :]

    lab_state = np.zeros((T, L.KMAX * L.V), dtype=np.float32)
    if T:
        truth = raw.lat.astype(np.float32) * known[..., None].astype(np.float32)
        lab_state = L.flatten_state(truth)
    lab_mask = ask.astype(np.float32)

    lab_class = np.asarray(raw.cls, dtype=np.int64).copy()
    lab_event = (np.asarray(raw.event) > 0).astype(np.float32)

    flags = np.zeros(T, dtype=np.int8)
    if T:
        flags |= (np.any(raw.idmask, axis=1)).astype(np.int8) * L.FLAG_IDMASK
        flags |= raw.query.astype(np.int8) * L.FLAG_QUERY
        flags |= (raw.event > 0).astype(np.int8) * L.FLAG_EVENT
        if raw.cross is not None and raw.cross.size:
            for row in raw.cross.reshape(-1, 5):
                if row[0] < 0:
                    continue
                f = int(np.searchsorted(raw.t.astype(np.float64), row[0]))
                f = min(max(f, 0), T - 1)
                flags[f] |= L.FLAG_CROSS

    if raw.cross is not None and raw.cross.size:
        cross = np.asarray(raw.cross, dtype=np.float32).reshape(-1, 5)
    else:
        cross = np.full((1, 5), -1.0, dtype=np.float32)

    meta_k = np.full(T, int(raw.K), dtype=np.int8)
    meta_gap = np.full(T, float(raw.gap_s), dtype=np.float32)
    # lab_age[t, k] = seconds since slot k's last information-bearing
    # observation (0 on any step where the token is present, then grows by the
    # actual dt of each hidden step).  For event_seq this is the age since the
    # last *target* event, because only target events carry a token.
    lab_age = np.zeros((T, L.KMAX), dtype=np.float32)
    lab_gap_s = np.full(T, float(raw.gap_s), dtype=np.float32)
    if T:
        vis_i = np.asarray(raw.vis, dtype=bool)
        t64 = np.asarray(raw.t, dtype=np.float64)
        last_vis_idx = np.maximum.accumulate(
            np.where(vis_i, np.arange(T, dtype=np.int64)[:, None], np.int64(-1)),
            axis=0)
        last_vis_t = np.where(last_vis_idx >= 0,
                              t64[np.clip(last_vis_idx, 0, T - 1)], 0.0)
        age = np.where(last_vis_idx >= 0, t64[:, None] - last_vis_t, 0.0)
        lab_age = age.astype(np.float32)
    mix = np.asarray([L.V, L.VC, L.N_CODES, L.C_EVENT, L.KMAX, L.LAYOUT_VERSION],
                     dtype=np.float32).reshape(1, 6)
    return {
        "feat": feat,
        "t": raw.t.astype(np.float32),
        "tok": tok,
        "lab_state": lab_state,
        "lab_mask": lab_mask,
        "lab_class": lab_class.astype(np.int64),
        "lab_event": lab_event,
        "lab_age": lab_age,
        "lab_gap_s": lab_gap_s,
        "meta_phase": raw.phase.astype(np.int8),
        "meta_flags": flags,
        "meta_k": meta_k,
        "meta_gap_s": meta_gap,
        "meta_cross": cross,
        "meta_mix": mix,
        "meta_dt0": np.asarray([float(dt0)], dtype=np.float32),
    }


# --------------------------------------------------------------------------- #
# npz io
# --------------------------------------------------------------------------- #
def save_seq(seq: Dict[str, np.ndarray], path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in seq.items()})
    return str(path)


def load_seq(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# determinism helpers
# --------------------------------------------------------------------------- #
def stable_hash_int(text: str, mod: int = 2 ** 31 - 1) -> int:
    """Platform-independent integer hash (Python's hash() is randomised)."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod


def episode_rng(seed_base: int, dataset: str, episode: int) -> np.random.Generator:
    """Deterministic RNG for one generated episode."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed_base) & 0xFFFFFFFF,
                                stable_hash_int(dataset, 2 ** 31 - 1),
                                int(episode) & 0xFFFFFFFF]))


def seed_of_episode(seed_base: int, dataset: str, episode: int) -> List[int]:
    """The seed triple recorded in the manifest (fully reproducible)."""
    return [int(seed_base) & 0xFFFFFFFF,
            stable_hash_int(dataset, 2 ** 31 - 1),
            int(episode) & 0xFFFFFFFF]


# --------------------------------------------------------------------------- #
# temporal shuffle control
# --------------------------------------------------------------------------- #
_ROW_KEYS = ("feat", "tok", "lab_state", "lab_mask", "lab_class", "lab_event",
             "lab_age", "lab_gap_s", "meta_phase", "meta_flags", "meta_k",
             "meta_gap_s")


def temporal_shuffle(seq: Dict[str, np.ndarray], rng: np.random.Generator,
                     min_step: int = 3) -> Dict[str, np.ndarray]:
    """Temporal-shuffle control: permute the ROWS (obs + labels) in time.

    Timestamps stay unchanged and increasing; every (observation row, label
    row) pair is kept intact.  A frame-copying model is unaffected, whereas a
    model that exploits temporal continuity / writes-then-holds is destroyed.
    This is the control requested for `reports/LIT-BRIEF.md` (one-step temporal
    tasks can be passed by emitting a blurred previous frame).
    """
    T = int(seq["t"].shape[0])
    if T <= 2:
        return {k: np.asarray(v).copy() for k, v in seq.items()}
    perm = np.arange(T)
    tries = 0
    while tries < 50:
        perm = rng.permutation(T)
        # reject near-identity permutations so the control is actually different
        if np.abs(perm - np.arange(T)).mean() >= min_step * 0.5:
            break
        tries += 1
    out = {}
    for k, v in seq.items():
        arr = np.asarray(v)
        if k in _ROW_KEYS and arr.shape[0] == T:
            out[k] = arr[perm].copy()
        else:
            out[k] = arr.copy()
    return out


# --------------------------------------------------------------------------- #
# irregular-sampling perturbations (operate on RawEpisode via stream_ops)
# --------------------------------------------------------------------------- #
_RAW_ROW_KEYS = ("vis", "lat", "idmask", "query", "event", "phase", "cls")


def _raw_to_dict(raw: RawEpisode) -> Dict[str, Any]:
    return {
        "t": raw.t.astype(np.float32),
        "vis": raw.vis,
        "lat": raw.lat,
        "idmask": raw.idmask,
        "query": raw.query,
        "event": raw.event,
        "phase": raw.phase,
        "cls": raw.cls,
        "meta_mix": np.asarray([L.KMAX], dtype=np.float32),
    }


def _dict_to_raw(d: Dict[str, Any], K: int, gap_s: float,
                 cross: Optional[np.ndarray], meta: Dict[str, Any]) -> RawEpisode:
    return RawEpisode(
        t=np.asarray(d["t"], dtype=np.float32),
        vis=np.asarray(d["vis"], dtype=bool),
        lat=np.asarray(d["lat"], dtype=np.float32),
        idmask=np.asarray(d["idmask"], dtype=bool),
        query=np.asarray(d["query"], dtype=bool),
        event=np.asarray(d["event"], dtype=np.int8),
        phase=np.asarray(d["phase"], dtype=np.int8),
        cls=np.asarray(d["cls"], dtype=np.int64),
        K=int(K), gap_s=float(gap_s),
        cross=cross, meta=dict(meta),
    )


def perturb_raw(raw: RawEpisode, variant: str, rng: np.random.Generator,
                nominal_dt: float = 0.1, params: Optional[Dict[str, Any]] = None
                ) -> RawEpisode:
    """Apply one irregular-sampling variant to a RawEpisode.

    Variants (DESIGN §1.6 (a)-(e))::

        logdt     lognormal Δt with configurable CV (mean preserved)
        drop30/50/70   random frame dropout of 30/50/70 %
        repeat    duplicate a random subset of frames
        burst     drop two contiguous bursts
        stretch2/4/8    time stretch of the physical clock
        mixed     drop50 + burst + repeat + logdt

    Physical time is preserved: dropping a frame makes the next dt larger, it
    does not renumber the clock (see src/common/stream_ops.py).
    """
    from ..common import stream_ops as so

    params = dict(params or {})
    d = _raw_to_dict(raw)
    variant = str(variant)

    if variant in ("none", "", None):
        pass
    elif variant == "logdt":
        cv = float(params.get("cv", 0.5))
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        d = so.jitter_time(d, rng, sigma=sigma)
    elif variant in ("drop30", "drop50", "drop70"):
        keep = {"drop30": 0.7, "drop50": 0.5, "drop70": 0.3}[variant]
        d = so.drop_random(d, keep, rng)
    elif variant == "repeat":
        d = so.repeat_frames(d, rng, prob=float(params.get("prob", 0.3)))
    elif variant == "burst":
        d = so.drop_burst(d, rng, n_bursts=int(params.get("n_bursts", 2)),
                          burst_frac=float(params.get("burst_frac", 0.25)))
    elif variant.startswith("stretch"):
        factor = float(variant[len("stretch"):])
        d = dict(d)
        d["t"] = (np.asarray(d["t"], dtype=np.float64) * factor).astype(np.float32)
    elif variant == "mixed":
        d = so.drop_random(d, 0.5, rng)
        d = so.drop_burst(d, rng, n_bursts=1, burst_frac=0.12)
        d = so.repeat_frames(d, rng, prob=0.15)
        cv = float(params.get("cv", 0.25))
        d = so.jitter_time(d, rng, sigma=math.sqrt(math.log(1.0 + cv * cv)))
    else:  # pragma: no cover
        raise ValueError(f"unknown irregular variant {variant!r}")

    t = np.asarray(d["t"], dtype=np.float64)
    t = t - t[0]
    t = _enforce_strictly_increasing(t)
    d["t"] = t.astype(np.float32)
    # a heavy dropout can shred short episodes into unusable stubs: fall back to
    # a milder dropout (deterministic, still seeded) so every sequence keeps
    # enough masked steps to be probed.
    if d["t"].shape[0] < 6:
        d0 = _raw_to_dict(raw)
        d0 = so.drop_random(d0, 0.7, rng)
        t0 = np.asarray(d0["t"], dtype=np.float64)
        t0 = _enforce_strictly_increasing(t0 - t0[0])
        d0["t"] = t0.astype(np.float32)
        d = d0
    # cross-spec times scale with the physical clock for stretching
    cross = None
    if raw.cross is not None and raw.cross.size:
        cross = np.asarray(raw.cross, dtype=np.float32).copy()
        if variant.startswith("stretch"):
            factor = float(variant[len("stretch"):])
            cross[:, 0:3] *= factor
    new_meta = dict(raw.meta or {})
    new_meta["irregular"] = variant
    return _dict_to_raw(d, raw.K, raw.gap_s, cross, new_meta)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_seq(seq: Dict[str, np.ndarray], strict: bool = True) -> None:
    """Shape / dtype / monotonicity / range checks for one generated sequence."""
    T = int(seq["t"].shape[0])
    assert T >= 2, "sequence too short"
    assert seq["feat"].shape == (T, L.DG), seq["feat"].shape
    assert seq["feat"].dtype == np.float16
    assert seq["tok"].shape == (T, L.KMAX, L.TOK_DIM), seq["tok"].shape
    assert seq["tok"].dtype == np.float16
    assert seq["lab_state"].shape == (T, L.KMAX * L.V)
    assert seq["lab_state"].dtype == np.float32
    assert seq["lab_mask"].shape == (T, L.KMAX)
    assert seq["lab_class"].shape == (T,)
    assert seq["lab_class"].dtype == np.int64
    assert seq["lab_event"].shape == (T,)
    assert seq["lab_age"].shape == (T, L.KMAX), seq["lab_age"].shape
    assert seq["lab_age"].dtype == np.float32
    assert seq["lab_gap_s"].shape == (T,), seq["lab_gap_s"].shape
    assert seq["t"].dtype == np.float32
    assert np.all(np.diff(seq["t"]) > 0), "t must be strictly increasing"
    for k in ("feat", "tok", "lab_state", "lab_mask", "lab_event", "lab_class",
              "lab_age", "lab_gap_s"):
        assert np.all(np.isfinite(seq[k])), f"non-finite values in {k}"
    assert np.all(seq["lab_age"] >= -1e-6), "lab_age must be >= 0"
    if strict:
        st = seq["lab_mask"] > 0.5
        # every masked step must have a non-zero true state (it is asked for)
        asked = L.unflatten_state(seq["lab_state"])[st]
        assert np.all(np.any(asked != 0, axis=-1)), "lab_mask set on a zero state"
        # tok vis flag must be 0/1
        visv = seq["tok"][..., L.V]
        assert np.all((visv == 0) | (visv == 1))
        # class labels only where an id exists or event fired
        assert np.all((seq["lab_class"] >= -1) & (seq["lab_class"] < L.N_CODES))
