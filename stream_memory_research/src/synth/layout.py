"""Tensor / channel layout for the controlled synthetic memory benchmark.

This module is the single source of truth for

* the **observation** tensors written to every synthetic npz,
* the **label** tensors, and
* the identity *code* used inside the 16-dim per-object state.

It implements the FINAL layout agreed with the Lead (see DESIGN.md §1.6 and the
Lead's final label/head message), namely one uniform tensor set for every task
so that a single trained model (one head set) can be evaluated at any K:

npz keys (every synthetic sequence carries all of them)
-------------------------------------------------------
feat       [T, 64]   float16   coarse GLOBAL summary (channels 0..15 meaningful,
                             16..63 reserved zeros).  No per-object values!
tok        [T, 16, 17] float16 per-object tokens: channel 0..15 = that object's
                             current 16-dim value *as observed now* (identity
                             dims zeroed when the object is detected but not
                             identifiable), channel 16 = visibility flag
                             (1.0 iff the object is observed at this step;
                              when it is not observed the WHOLE token is zero).
t          [T]      float32   physical seconds, strictly increasing, t[0]=0
lab_state  [T, 256] float32   16 slots x 16 state dims, flattened
                             (slot k -> columns 16k .. 16k+15)
lab_mask   [T, 16]  float32   1.0 iff slot k is ASKED FOR here, i.e. the state
                             is recoverable from the observation history but is
                             NOT present in the current observation
                             (hidden / identity-masked steps).  Memory-only.
lab_class  [T]      int64     -1 = ignore.  vanish_reappear: identity id
                             (0..15).  event_seq: event type (0..C-1).
lab_event  [T]      float32   0/1 event indicator (single channel)
meta_phase [T]      int8      task phase id (names in manifest.extra.phases)
meta_flags [T]      int8      bit 1 = identity-masked step, bit 2 = query step,
                             bit 4 = event step, bit 8 = crosstalk probe step
meta_k     [T]      int8      number of existing object slots in this episode
meta_gap_s [T]      float32   designated hidden-gap length (seconds) of the
                             episode, -1.0 where the generator did not fix one
meta_cross [Nc, 5]  float32   crosstalk probe spec, padded with -1:
                             (t_prev, t_write, t_post, writer j, reader i)
meta_mix   [1, 6]   float32   (V, VC, N_CODES, C_EVENT, KMAX, LAYOUT_VERSION)
meta_dt0   [1]      float32   nominal inter-sample interval Δ0 (seconds)

Identity code (inside each 16-dim state vector)
-----------------------------------------------
dims 0..VC-1 (VC=8) carry a *signed one-hot* code: exactly one of them is
+1 or -1, the rest are 0.  That gives N_CODES = 2*VC = 16 distinguishable
identities and is exactly the number of classes of the shared `class` head.
dims VC..V-1 (VA=8) carry the continuous analogue state ~ N(0, 1).

`decode_code` turns any 16-dim vector into a code in 0..N_CODES-1, so a
regression-only model can still be scored with identity accuracy.

Numeric range: state = code (±1) + N(0,1) analogue -> entries ~O(1); feat
summary channels are in [-1, 1] or small positive counts.  Documented here so
the report can state exactly what each channel means.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
KMAX = 16          # number of object slots visible to the model at all times
V = 16             # state dims per object
VC = 8             # identity-code dims (signed one-hot -> 16 codes)
VA = V - VC        # analogue dims
N_CODES = 2 * VC   # 16 identities == class head size
TOK_DIM = V + 1    # 16 value channels + 1 visibility flag
DG = 64            # feat width (contract / DESIGN default)
FEAT_SUMMARY_DIM = 16
C_EVENT = 4        # event "type" classes for event_seq (type 0 == no event)
LAYOUT_VERSION = 1

# feat channel semantics (channels 0..15; 16..63 reserved zeros)
FEAT_CHANNELS: List[Dict[str, Any]] = [
    {"idx": 0, "name": "bias", "meaning": "constant 1.0"},
    {"idx": 1, "name": "vis_frac",
     "meaning": "number of currently observed object tokens / 16"},
    {"idx": 2, "name": "any_hidden",
     "meaning": "1.0 if at least one existing object is currently hidden"},
    {"idx": 3, "name": "query",
     "meaning": "1.0 on steps flagged as a recall query"},
    {"idx": 4, "name": "event_pulse",
     "meaning": "1.0 on an event step (knock/wave impulse)"},
    {"idx": 5, "name": "event_type_norm",
     "meaning": "event type / (C_EVENT-1) in [0,1]; 0 when no event"},
    {"idx": 6, "name": "elapsed_s_60",
     "meaning": "physical elapsed time / 60 s (causal clock)"},
    {"idx": 7, "name": "event_age_s_60",
     "meaning": "seconds since the last event / 60 s (0 if none yet)"},
    {"idx": 8, "name": "since_change_s_60",
     "meaning": "seconds since any visible token value last changed / 60 s"},
    {"idx": 9, "name": "hidden_asked_frac",
     "meaning": "number of object-slots that are existing+hidden / 16"},
    {"idx": 10, "name": "mean_vis_norm",
     "meaning": "mean L2 norm of the currently observed token values"},
    {"idx": 11, "name": "n_existing_frac",
     "meaning": "number of existing object slots / 16"},
    {"idx": 12, "name": "reserved", "meaning": "0.0"},
    {"idx": 13, "name": "reserved", "meaning": "0.0"},
    {"idx": 14, "name": "reserved", "meaning": "0.0"},
    {"idx": 15, "name": "reserved", "meaning": "0.0"},
]

CH_BIAS = 0
CH_VIS_FRAC = 1
CH_ANY_HIDDEN = 2
CH_QUERY = 3
CH_EVENT = 4
CH_EVENT_TYPE = 5
CH_ELAPSED = 6
CH_EVENT_AGE = 7
CH_SINCE_CHANGE = 8
CH_HIDDEN_ASKED = 9
CH_MEAN_VIS_NORM = 10
CH_N_EXIST = 11

# meta_flags bits
FLAG_IDMASK = 1
FLAG_QUERY = 2
FLAG_EVENT = 4
FLAG_CROSS = 8


# --------------------------------------------------------------------------- #
# identity code
# --------------------------------------------------------------------------- #
def code_to_vec(codes) -> np.ndarray:
    """code(s) in [0, N_CODES) -> signed one-hot [..., V] float32.

    code c: dim c//2 gets (+1 if c even else -1).  This is canonical, so any
    model can be scored by `decode_code` without storing anything.
    """
    c = np.asarray(codes, dtype=np.int64)
    vec = np.zeros(c.shape + (V,), dtype=np.float32)
    dim = c // 2
    sign = np.where(c % 2 == 0, 1.0, -1.0).astype(np.float32)
    flat = vec.reshape(-1, V)
    flat[np.arange(flat.shape[0]), dim.reshape(-1)] = sign.reshape(-1)
    return vec


def decode_code(state: np.ndarray) -> np.ndarray:
    """16-dim state vector(s) -> code(s) in [0, N_CODES).

    Argmax over |code dims|; sign selects +/-.  An all-zero vector (object not
    known / not observed) decodes deterministically to 0, which is exactly the
    uniform-chance behaviour of a memoryless model, so oracle tests can assert
    "memoryless == chance == 1/N_CODES".
    """
    s = np.asarray(state, dtype=np.float32)
    code_part = s[..., :VC]
    dim = np.argmax(np.abs(code_part), axis=-1)
    picked = np.take_along_axis(code_part, dim[..., None], axis=-1)[..., 0]
    return (2 * dim + (picked < 0)).astype(np.int64)


def make_state(code: int, analog: np.ndarray) -> np.ndarray:
    """Build one V-dim state vector from an identity code + analogue vector."""
    v = np.zeros(V, dtype=np.float32)
    v[:] = code_to_vec(code)
    a = np.asarray(analog, dtype=np.float32).reshape(-1)
    if a.size != VA:
        raise ValueError(f"analogue must have {VA} dims, got {a.size}")
    v[VC:] = a
    return v


def random_state(rng: np.random.Generator) -> np.ndarray:
    """A fresh random latent state (random identity + N(0,1) analogue)."""
    return make_state(int(rng.integers(0, N_CODES)),
                      rng.standard_normal(VA).astype(np.float32))


# --------------------------------------------------------------------------- #
# derived layout used by probes
# --------------------------------------------------------------------------- #
def apply_idmask(state: np.ndarray, idmask: np.ndarray) -> np.ndarray:
    """Zero the identity dims on masked rows (observation-side masking)."""
    out = np.array(state, copy=True)
    out[..., :VC] = np.where(np.asarray(idmask, dtype=bool)[..., None],
                             np.float32(0.0), out[..., :VC])
    return out


def flatten_state(states: np.ndarray) -> np.ndarray:
    """[T, KMAX, V] -> [T, KMAX*V] (column order slot-major)."""
    st = np.asarray(states, dtype=np.float32)
    T, K, VV = st.shape
    return st.reshape(T, K * VV)


def unflatten_state(flat: np.ndarray) -> np.ndarray:
    """[T, KMAX*V] -> [T, KMAX, V]."""
    f = np.asarray(flat, dtype=np.float32)
    T = f.shape[0]
    return f.reshape(T, KMAX, V)


def label_specs() -> Dict[str, Dict[str, Any]]:
    """Head spec shared by every synthetic dataset (matches DESIGN §1.3)."""
    return {
        "state": {"kind": "mse", "dim": KMAX * V},
        "class": {"kind": "ce", "dim": 1, "n_classes": N_CODES, "ignore": -1},
        "event": {"kind": "bce", "dim": 1, "n_classes": 1},
    }


def layout_dict() -> Dict[str, Any]:
    return {
        "layout_version": LAYOUT_VERSION,
        "KMAX": KMAX, "V": V, "VC": VC, "VA": VA, "N_CODES": N_CODES,
        "tok_dim": TOK_DIM, "feat_dim": DG, "feat_summary_dim": FEAT_SUMMARY_DIM,
        "C_EVENT": C_EVENT,
    }


def layout_signature() -> str:
    """Stable hash of the observation/label layout.

    Identical for every synthetic dataset (base, sweeps, shuffled, irregular),
    so the evaluator can assert head compatibility mechanically.
    """
    import hashlib
    import json
    payload = json.dumps({"layout": layout_dict(), "labels": label_specs(),
                          "channels": channel_layout_dict()},
                         sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def channel_layout_dict() -> Dict[str, Any]:
    return {
        "feat": {
            "shape": "[T, 64] float16",
            "summary_channels": FEAT_CHANNELS,
            "reserved": "channels 16..63 are always 0.0 (reserved)",
        },
        "tok": {
            "shape": f"[T, {KMAX}, {TOK_DIM}] float16",
            "channels_0_15": "that object's current 16-dim value (identity dims "
                             "zeroed when visible-but-not-identifiable)",
            "channel_16": "visibility flag: 1.0 iff the object is observed now; "
                          "when not observed the entire token is 0",
        },
        "labels": {
            "lab_state": f"[T, {KMAX * V}] float32 = 16 slots x {V} state dims; "
                         "true state scaled by lab_mask-independent knowability "
                         "(0 where the slot is unused or its value is unknowable)",
            "lab_mask": f"[T, {KMAX}] float32 = 1.0 iff the slot is asked for "
                        "(knowable but absent from the current observation); "
                        "all metrics are computed on lab_mask==1",
            "lab_class": "[T] int64 = -1 ignore; identity id (vanish_reappear) "
                         "or event type (event_seq)",
            "lab_event": "[T] float32 0/1 event indicator",
        },
        "identity_code": f"state dims 0..{VC - 1}: signed one-hot "
                         f"({N_CODES} identities); dims {VC}..{V - 1}: N(0,1) analogue",
    }
