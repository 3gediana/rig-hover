"""Concrete controlled-memory task generators (DESIGN §1.6).

Every task returns a :class:`~src.synth.base.RawEpisode`: the *latent truth*
plus the visibility/idmask/query/event/phase schedule.  The observation tensors
are derived from it by :func:`src.synth.base.raw_to_seq`, so the invariant
"the input carries no information about a hidden object" is structurally
guaranteed: a hidden slot contributes nothing to `feat`/`tok`.

Tasks
-----
recall_single    visible -> hidden(G) -> visible, hold one value across the gap
recall_multi     K objects; explicit crosstalk protocol: hide B over a window,
                 update only A in the middle, measure read-out of B and A
capacity_scan    K in {1,2,4,8,12,16}: present K slots one by one, hide all,
                 then ask for all K (effective-capacity measurement)
overwrite        same slot written v1 -> v2 -> v1 returns
vanish_reappear  object leaves; on re-entry it is detected but NOT identifiable
                 for a few frames -> identity must come from memory
change_detect    long stable hold, then ONE true change -> hold vs update-latency
event_seq        sparse knock impulses; target events carry a payload state,
                 distractor impulses carry none; after an event the state must
                 be held for a long time
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from . import layout as L
from .base import RawEpisode, Timeline


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _new_state_diff(rng: np.random.Generator, old: Optional[np.ndarray],
                    task: str = "") -> np.ndarray:
    """A fresh random state whose identity code differs from `old`."""
    old_code = -1 if old is None else int(L.decode_code(old))
    for _ in range(64):
        st = L.random_state(rng)
        if int(L.decode_code(st)) != old_code:
            return st
    return st  # pragma: no cover


def _merge(defaults: Dict[str, Any], params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(defaults)
    out.update(params or {})
    return out


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
class TaskBase:
    name: str = "base"
    defaults: Dict[str, Any] = {"dt": 0.1}

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge(self.defaults, params)

    def build(self, rng: np.random.Generator) -> RawEpisode:  # pragma: no cover
        raise NotImplementedError

    # ---- convenience -------------------------------------------------------
    def make(self, rng: np.random.Generator) -> RawEpisode:
        raw = self.build(rng)
        if not raw.meta:
            raw.meta = {}
        raw.meta.setdefault("task", self.name)
        return raw


# --------------------------------------------------------------------------- #
# 1) recall_single
# --------------------------------------------------------------------------- #
class RecallSingle(TaskBase):
    """One object: write -> hidden gap G seconds -> query/re-verify.

    §1.6: only the hidden segment is scored (`lab_mask==1`).
    """

    name = "recall_single"
    defaults = {
        "dt": 0.1,
        "pre_frames": 2,
        "write_frames": 4,
        "query_frames": 2,
        "verify_frames": 3,
        "gap_s": None,          # fixed gap for the eval-only sweep datasets
        "gap_min_s": 0.1,
        "gap_max_s": 3.2,       # = 32 steps at dt=0.1 (DESIGN §1.6 sweep)
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        if p.get("gap_s") is not None:
            gap_steps = max(1, int(round(float(p["gap_s"]) / dt)))
        else:
            gs = float(np.exp(rng.uniform(np.log(float(p["gap_min_s"])),
                                         np.log(float(p["gap_max_s"])))))
            gap_steps = max(1, int(round(gs / dt)))
        gap_s = gap_steps * dt

        tl = Timeline(K=1, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)     # nothing known yet
        tl.set_random(0)                                     # the value to remember
        tl.append(int(p["write_frames"]), vis=[0], phase=1)  # visible write
        tl.append(gap_steps, vis=[], phase=2)                # hidden gap (scored)
        tl.append(int(p["query_frames"]), vis=[], phase=3, query=True)
        tl.append(int(p["verify_frames"]), vis=[0], phase=4)  # reappears: sanity
        return tl.finish(gap_s=gap_s, meta={"gap_steps": gap_steps})


# --------------------------------------------------------------------------- #
# 2) recall_multi (+ crosstalk protocol)
# --------------------------------------------------------------------------- #
class RecallMulti(TaskBase):
    """K objects; explicit crosstalk protocol per block.

    Block = [refresh: all K visible] + [window: all K hidden except the writer,
    which is visible at the middle frame and receives a new value].
    For every block we record, for every reader i (0..K-1, i == writer is the
    diagonal), the probe spec (t_prev = 0.2 W, t_write = 0.5 W, t_post = 0.9 W).
    §1.6 crosstalk:  Delta_i = |pred_i(t_post) - pred_i(t_prev)|.
    """

    name = "recall_multi"
    defaults = {
        "dt": 0.1,
        "k_choices": [2, 3, 4, 6, 8],
        "n_blocks": [3, 5],
        "pre_frames": 2,
        "refresh_frames": 2,
        "window_steps": [6, 12],
        "p_refresh_update": 0.5,
        "gap_steps_fixed": None,   # eval-only gap sweep: plain K-object recall
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        K = int(rng.choice(list(p["k_choices"])))
        if p.get("gap_steps_fixed") is not None:
            # eval-only gap sweep: refresh all K slots, hide ALL of them for a
            # fixed number of steps, then ask for all K.  Same observation /
            # label layout as the main dataset, so one checkpoint can be run.
            G = max(1, int(p["gap_steps_fixed"]))
            tl = Timeline(K=K, rng=rng, dt=dt)
            tl.append(int(p["pre_frames"]), vis=[], phase=0)
            for k in range(K):
                tl.set_random(k)
            tl.append(int(p["refresh_frames"]), vis=list(range(K)), phase=1)
            tl.append(G, vis=[], phase=2, query=True)
            tl.append(2, vis=[], phase=2, query=True)
            tl.append(int(p["refresh_frames"]), vis=list(range(K)), phase=1)
            return tl.finish(gap_s=G * dt,
                             meta={"K": K, "gap_steps": G, "sweep": "multi_gap"})
        n_blocks = int(rng.integers(int(p["n_blocks"][0]), int(p["n_blocks"][1]) + 1))
        refresh = int(p["refresh_frames"])
        p_upd = float(p["p_refresh_update"])

        tl = Timeline(K=K, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        for k in range(K):
            tl.set_random(k)
        tl.append(refresh, vis=list(range(K)), phase=1)      # initial write

        for _ in range(n_blocks):
            W = int(rng.integers(int(p["window_steps"][0]), int(p["window_steps"][1]) + 1))
            writer = int(rng.integers(0, K))
            # refresh: all slots visible; some get a fresh value at frame 0
            for i in range(refresh):
                if i == 0:
                    for k in range(K):
                        if rng.random() < p_upd:
                            tl.set_random(k)
                tl.append(1, vis=list(range(K)), phase=1)
            # hidden window
            start = tl.n_frames
            w_off = W // 2
            for i in range(W):
                if i == w_off:
                    tl.set_random(writer)          # the ONLY update in the window
                    tl.append(1, vis=[writer], phase=2, query=True)
                else:
                    tl.append(1, vis=[], phase=2, query=True)
            prev_off = max(1, int(round(0.2 * W)))
            post_off = max(w_off + 1, int(round(0.9 * W)))
            post_off = min(post_off, W - 1)
            t_prev = tl.t_at(start + prev_off)
            t_write = tl.t_at(start + w_off)
            t_post = tl.t_at(start + post_off)
            for reader in range(K):
                tl.add_cross(t_prev, t_write, t_post, writer, reader)

        return tl.finish(meta={"K": K, "n_blocks": n_blocks})


# --------------------------------------------------------------------------- #
# 3) capacity_scan
# --------------------------------------------------------------------------- #
class CapacityScan(TaskBase):
    """Present K slots one by one, hide ALL of them, then ask for all K.

    §1.6: K in {1,2,4,8,12,16}, "全部隐藏 G 秒后一次性预测"; metric = capacity
    curve and the saturation point K* (>= 0.9 x the K=1 score).
    """

    name = "capacity_scan"
    defaults = {
        "dt": 0.1,
        "k": None,                       # fixed K for the eval-only sweep sets
        "k_choices": [1, 2, 4, 8, 12, 16],
        "pre_frames": 2,
        "present_frames": 2,
        "retention_steps": [6, 10],
        "query_frames": 3,
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        K = int(p["k"]) if p.get("k") is not None else int(rng.choice(list(p["k_choices"])))
        K = min(K, L.KMAX)
        pres = int(p["present_frames"])
        tl = Timeline(K=K, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        for k in range(K):
            tl.set_random(k)
            tl.append(pres, vis=[k], phase=0)
        gap = int(rng.integers(int(p["retention_steps"][0]),
                               int(p["retention_steps"][1]) + 1))
        tl.append(gap, vis=[], phase=1, query=True)
        tl.append(int(p["query_frames"]), vis=[], phase=2, query=True)
        return tl.finish(gap_s=gap * dt, meta={"K": K, "retention_steps": gap})


# --------------------------------------------------------------------------- #
# 4) overwrite
# --------------------------------------------------------------------------- #
class Overwrite(TaskBase):
    """Same slot: write v1 -> write v2 (v1 gone) -> v1 returns.

    §1.6: accuracy of reading v2 right after the overwrite, and of reading v1
    again after it returns.  Tests "overwrite when you should, restore when the
    original comes back" (rules out both stubborn-no-write and never-restore).
    """

    name = "overwrite"
    defaults = {
        "dt": 0.1,
        "pre_frames": 2,
        "write_frames": 4,
        "gap_after_v1": [4, 6],
        "gap_after_v2": [4, 6],
        "return_frames": 4,
        "gap_after_return": [4, 8],
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        tl = Timeline(K=1, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        v1 = tl.set_random(0)
        tl.append(int(p["write_frames"]), vis=[0], phase=1)
        g1 = int(rng.integers(p["gap_after_v1"][0], p["gap_after_v1"][1] + 1))
        tl.append(g1, vis=[], phase=2, query=True)
        v2 = tl.set_state(0, _new_state_diff(rng, v1))       # overwrite identity
        tl.append(int(p["write_frames"]), vis=[0], phase=3)
        g2 = int(rng.integers(p["gap_after_v2"][0], p["gap_after_v2"][1] + 1))
        tl.append(g2, vis=[], phase=4, query=True)           # must read v2 here
        tl.set_state(0, v1)                                  # v1 returns unchanged
        tl.append(int(p["return_frames"]), vis=[0], phase=5)
        g3 = int(rng.integers(p["gap_after_return"][0],
                              p["gap_after_return"][1] + 1))
        tl.append(g3, vis=[], phase=6, query=True)           # must read v1 again
        return tl.finish(meta={"v1_code": int(L.decode_code(v1)),
                               "v2_code": int(L.decode_code(v2))})


# --------------------------------------------------------------------------- #
# 5) vanish_reappear
# --------------------------------------------------------------------------- #
class VanishReappear(TaskBase):
    """Object leaves the frame, then re-enters detected but not identifiable.

    On re-entry the token carries the analogue state but the identity dims are
    zeroed for `reappear_masked` frames, so the identity has to come from
    memory.  §1.6 metric: identity accuracy at 1/3/5 steps after re-entry.
    """

    name = "vanish_reappear"
    defaults = {
        "dt": 0.1,
        "pre_frames": 2,
        "write_frames": 4,
        "absence_steps": [6, 30],
        "reappear_masked": 5,
        "normal_frames": 4,
        "final_hidden": 4,
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        K = 1
        tl = Timeline(K=K, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        code = int(rng.integers(0, L.N_CODES))
        st = L.make_state(code, rng.standard_normal(L.VA).astype(np.float32))
        tl.set_state(0, st)
        tl.append(int(p["write_frames"]), vis=[0], phase=1)
        g = int(rng.integers(p["absence_steps"][0], p["absence_steps"][1] + 1))
        tl.append(g, vis=[], phase=2, query=True, cls=code)   # absent: ask id
        R = int(p["reappear_masked"])
        tl.append(R, vis=[0], phase=3, idmask=[0], query=True, cls=code)
        tl.append(int(p["normal_frames"]), vis=[0], phase=4)  # now identifiable
        tl.append(int(p["final_hidden"]), vis=[], phase=5, query=True, cls=code)
        return tl.finish(gap_s=g * dt,
                         meta={"identity": code, "absence_steps": g,
                               "reappear_masked": R})


# --------------------------------------------------------------------------- #
# 6) change_detect
# --------------------------------------------------------------------------- #
class ChangeDetect(TaskBase):
    """Long stable hold (false-update test), then exactly ONE true change.

    §1.6: the target does not change during `hold`, then changes once while
    visible; metrics are false_update_rate on the hold and update_latency_s
    from the change frame to the first hidden frame whose prediction matches.
    """

    name = "change_detect"
    defaults = {
        "dt": 0.1,
        "pre_frames": 2,
        "write_frames": 4,
        "hold_steps": [12, 30],
        "change_frames": 2,
        "post_steps": [10, 20],
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        tl = Timeline(K=1, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        v0 = tl.set_random(0)
        tl.append(int(p["write_frames"]), vis=[0], phase=1)
        hold = int(rng.integers(p["hold_steps"][0], p["hold_steps"][1] + 1))
        tl.append(hold, vis=[], phase=2, query=True)          # stable hold
        v1 = tl.set_state(0, _new_state_diff(rng, v0))        # the ONE change
        tl.append(int(p["change_frames"]), vis=[0], phase=3)
        post = int(rng.integers(p["post_steps"][0], p["post_steps"][1] + 1))
        tl.append(post, vis=[], phase=4, query=True)          # must hold new value
        return tl.finish(gap_s=hold * dt,
                         meta={"hold_steps": hold, "post_steps": post,
                               "v0_code": int(L.decode_code(v0)),
                               "v1_code": int(L.decode_code(v1))})


# --------------------------------------------------------------------------- #
# 7) event_seq
# --------------------------------------------------------------------------- #
class EventSeq(TaskBase):
    """Sparse knock-like impulses; a target event carries a payload state.

    Target events (type 1) deliver a new 16-dim payload at the impulse frame;
    distractor impulses (type 2) are visually similar (event pulse, event-type
    channel) but carry NO payload, so the state must NOT change.  Knocks can
    arrive in bursts (duplicate-trigger test).  Between events the target is
    not observed and the payload must be held (§1.6: post-event 1x/2x/4x/8x
    state accuracy; event F1 with 0.2 s tolerance and 0.2 s dedup).
    """

    name = "event_seq"
    defaults = {
        "dt": 0.1,
        "pre_frames": 2,
        "frames": [60, 90],
        "min_gap": 12,
        "max_gap": 25,
        "p_target": 0.6,
        "p_burst": 0.35,
        "fixed_gap_steps": None,   # eval-only spacing sweep: constant quiet gap
    }

    def build(self, rng: np.random.Generator) -> RawEpisode:
        p = self.params
        dt = float(p["dt"])
        T_total = int(rng.integers(int(p["frames"][0]), int(p["frames"][1]) + 1))
        fixed_gap = p.get("fixed_gap_steps")
        fixed_gap = None if fixed_gap is None else max(1, int(fixed_gap))
        tl = Timeline(K=1, rng=rng, dt=dt)
        tl.append(int(p["pre_frames"]), vis=[], phase=0)
        state: Optional[np.ndarray] = None
        n_target = n_distractor = 0
        while tl.n_frames < T_total:
            if fixed_gap is not None:
                gap = fixed_gap
            else:
                gap = int(rng.integers(int(p["min_gap"]), int(p["max_gap"]) + 1))
            gap = min(gap, T_total - tl.n_frames)
            if gap <= 0:
                break
            tl.append(gap, vis=[], phase=1)                   # quiet hold
            if tl.n_frames >= T_total:
                break
            burst = 1
            if rng.random() < float(p["p_burst"]):
                burst += int(rng.integers(0, 3))              # 1..2 extra impulses
            burst = min(burst, max(1, T_total - tl.n_frames))
            if rng.random() < float(p["p_target"]):
                state = _new_state_diff(rng, state)
                tl.set_state(0, state)
                tl.append(1, vis=[0], phase=2, event=1, cls=1)   # payload visible
                if burst > 1:
                    tl.append(burst - 1, vis=[], phase=2, event=1, cls=1)
                n_target += 1
            else:
                tl.append(burst, vis=[], phase=2, event=2, cls=2)  # no payload
                n_distractor += 1
        gap_s = (-1.0 if fixed_gap is None else float(fixed_gap) * dt)
        return tl.finish(gap_s=gap_s,
                         meta={"n_target": n_target,
                               "n_distractor": n_distractor,
                               "fixed_gap_steps": fixed_gap})


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
TASKS: Dict[str, type] = {
    "recall_single": RecallSingle,
    "recall_multi": RecallMulti,
    "capacity_scan": CapacityScan,
    "overwrite": Overwrite,
    "vanish_reappear": VanishReappear,
    "change_detect": ChangeDetect,
    "event_seq": EventSeq,
}


def build_task(name: str, params: Optional[Dict[str, Any]] = None) -> TaskBase:
    if name not in TASKS:
        raise KeyError(f"unknown synth task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name](params)


def task_defaults(name: str) -> Dict[str, Any]:
    return dict(TASKS[name].defaults)
