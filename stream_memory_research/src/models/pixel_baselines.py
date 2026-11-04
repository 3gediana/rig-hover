"""Pixel-to-event baselines -- same pixels and the same Scout backbone, seven
different temporal cores.

Why this file exists
--------------------
``pixel_event.py`` implements *our* network (939,227 params / 81.25 M MACs per
frame).  Any claim about its temporal machinery is only meaningful against
baselines that see **exactly the same pixels through exactly the same backbone**
and differ **only** in the temporal operator.  This module provides those
baselines under one contract:

    forward(imgs: [B,T,1,160,160] float, dt: [B,T] s, mask: [B,T] bool)
        -> (seq: Dict[str, Tensor], aux: Dict)

Every model returns the same heads as ``PixelEventNet``::

    "event"       [B,T,C]   logits          (C = n_event_classes, default 25)
    "onset"       [B,T,1]   logit
    "count"       [B,T,C]   current cumulative-count estimate  (softplus >= 0)
    "presence"    [B,T,P]   logits          (P = 4)
    "ready"       [B,T,1]   logit
    "gesture"     [B,T,G]   logits          (G = 10)
    "count_final" [B,C]     count at the last valid step

Shared backbone / shared head (fairness)
----------------------------------------
* Every model imports :class:`Scout`, :func:`dwise_sep`, ``PIXEL_SPEC``,
  :func:`count_params` and :func:`count_macs` from ``pixel_event`` -- not one
  re-implementation.  The Scout is 65,120 params / 26.756 M MACs per frame and
  is bit-identical across all seven models (same module, same widths).
* Encoder: ``Scout -> dwise_sep neck -> [global(128) || mean(f4)(64)] -> Linear``
  (256-d per frame).  Identical for the six trainable baselines.  ``frozen_dsm``
  is the documented exception: its Scout is frozen and it feeds the
  globally-average-pooled stride-4 map (64-d) straight into the DSM core.
* Input stage ``TimeFeatureStage``: zero the padded features, append
  ``log1p(dt)`` and the validity flag, LayerNorm -- the exact same semantics as
  ``src.models.base.InputNorm`` used by the whole temporal-core zoo.
* **One head class** :class:`BaselineHead` (LayerNorm -> 2-layer MLP with
  ``head_hidden=512`` -> one linear per task) for all seven models, always
  operating on the same ``core_dim=192`` width.  The head therefore has an
  identical 395,714 params in every model; all parameter differences come from
  the temporal operator.

Timestep handling (dt in real seconds)
--------------------------------------
``dt`` must not be ignorable, otherwise the irregular-sampling experiment is
rigged.  Each baseline states its mode explicitly (also recorded on the model as
``dt_mode``):

=================  ==========================================================
frame_cnn          (a) feature  -- ``log1p(dt)`` channel; stateless per frame
tcn                (a) feature  -- ``log1p(dt)`` channel; the conv operator is
                   discrete-time, so its 31-frame receptive field is 31
                   *observations* regardless of physical time (documented
                   weakness, not patched)
gru                (a) feature  -- ``log1p(dt)`` channel; recurrence advances
                   once per observation (single-layer ``nn.GRU`` maths,
                   stepped so the mask can hold the state)
lstm               (a) feature  -- same as gru
transformer        (a) feature  -- ``log1p(dt)`` channel; the W=32 window is
                   counted in observations, as a KV cache is
ssm                (a)+(b)      -- ``log1p(dt)`` channel AND real dt drives the
                   state decay ``exp(A * Delta_t * dt)`` (exact ZOH decay,
                   semigroup in dt for a fixed Delta_t)
frozen_dsm         (a)+(b)      -- DSM's ``InputNorm`` appends ``log1p(dt)`` and
                   real dt drives the slow-state ZOH decay ``exp(-Gamma*dt)``
                   and every fast-memory decay ``exp(-Lambda*dt)``
=================  ==========================================================

Strict causality (bit-exact, not "small")
-----------------------------------------
* Conv2d/conv1d are causal (left padding only for the temporal convs).
* The transformer uses a strict lower-triangular mask **including the diagonal**,
  restricted to the last W=32 observations; relative-position bias only, no
  absolute positional embedding, so nothing can leak from the future.
* BatchNorm: ``Scout`` contains BatchNorm2d, whose training-mode statistics are
  computed over the batch dimension.  In training mode the encoder processes one
  frame at a time, so BN can never mix statistics across time; in eval mode it
  batches over T (exact, BN then uses running statistics only).
* No op anywhere reads ``t+1`` (no reversed scans, no full-sequence
  normalisation over time, no bidirectional recurrence).

Mask semantics
--------------
``mask=False`` steps are *padding*: their frame content is replaced by zeros
before it can touch the network, their dt is forced to 0, their features are
zeroed, their emitted outputs are exactly 0, and their information never reaches
a valid step:

* recurrent cores (gru/lstm/ssm/frozen_dsm) hold the state with ``torch.where``;
* windowed cores (tcn/transformer) run on the **compacted** valid subsequence
  (stable partition, valid frames keep their order), so a masked gap is
  equivalent to the truncated sequence -- the window is over observations, the
  same thing a streaming KV cache holds;
* the stateless floor (frame_cnn) is per-frame by construction.

Masked-step garbage (including NaN) is proven inert bit-exactly (0.0): it is
replaced by zeros before the first convolution, dt is forced to 0, the state is
held, and each emitted value at a padded step is exactly 0.  The only claim
tested at 1e-6 rather than exactly 0 is "remove the pads and you get the
truncated sequence": the *pre-head* representation is bit-identical for six of
the seven models (0.0; the transformer's attention sums reassociate at ~6e-7
when T changes), and the residual ~6e-8 appears only in the shared head, i.e. a
``Linear`` over ``[B,T,.]`` cannot associate identically for different ``T``.
That is float reassociation, not information leakage; the project's own
``test_mask`` uses 1e-6 for this same check.

``count_final`` is the count head evaluated at the last valid step.

Parameter budget
----------------
Task budget is 0.5 M - 2.0 M params (ours: 0.939 M).  Widths were chosen per
core from that budget and are reported as-is (no silent width tweaks).  All
seven land inside the budget:

    frame_cnn  565,414      gru        787,750      transformer 1,160,102
    lstm       861,862      ssm        692,374      tcn         1,010,086
    frozen_dsm 807,982 total / 742,862 trainable (65,120 params are the frozen
               Scout; ``pixel_event.count_params`` counts trainable params only,
               so both numbers are reported)

The head is identical in all seven (395,714 params at ``core_dim=192,
head_hidden=512``), and the Scout is identical in the six trainable baselines
(65,120 params); only the temporal operator differs.

MACs per frame (per *sequence*, same counter as the reference)
--------------------------------------------------------------
``pixel_event.count_macs`` hooks only ``nn.Conv2d`` and ``nn.Linear`` and sums
over batch and time, so at ``B=2`` it returns a batch-scaled number: the task's
81.25 M reference is that raw counter at ``B=2`` (i.e. ~40.6 M per sequence at
the code state when the task was written).  This module reports the
per-sequence value (independent of B) and the raw value next to it.  It also
provides :func:`count_macs_full`, which additionally accounts for Conv1d,
GRUCell/LSTMCell matmuls and the explicit attention/scan products that hooks
cannot see.  Measured per-sequence per-frame MACs: 34.72 M (frame_cnn, gru),
34.72 M + 0.44 M conv1d (tcn), 34.85 M (ssm), 35.31 M (transformer), 27.49 M
(frozen_dsm, which skips the neck and the global token) -- all below ours
(41.72 M current / 40.63 M at the task's code state), because the baselines use
only the shared Scout while our model additionally runs the foveal eye, the
entity binder, the relation graph and the semi-Markov event engine.

Effective span: tcn reaches 31 observations back (1 + 2*(1+2+4+8)), the
2-layer W=32 transformer reaches 63 (1 + 2*(32-1); the per-layer KV cache is
still bounded at 32), everything else carries an unbounded recurrent state.
"""
from __future__ import annotations

import argparse
import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsm import DSM
from .pixel_event import (PIXEL_SPEC, Scout, make_scout, count_macs, count_params,
                          dwise_sep)

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
#: defaults shared by every baseline.  ``build_baseline(name, cfg={...})``
#: overrides individual keys.  Widths are chosen so the total parameter count
#: lands in the task's 0.5 M - 2.0 M budget (ours: 0.939 M).
BASELINE_SPEC: Dict[str, object] = {
    # --- shared encoder (six trainable baselines) ---
    "feat_dim": 256,            # per-frame feature width produced by the Scout fusion
    "core_dim": 192,            # width every temporal core outputs == head input
    "head_hidden": 512,         # shared head MLP width
    # --- tcn ---
    "tcn_kernel": 3,
    "tcn_dilations": (1, 2, 4, 8),   # RF = 1 + 2*(1+2+4+8) = 31 frames
    # --- transformer ---
    "n_heads": 4,
    "n_layers": 2,
    "window": 32,               # bounded KV cache: last 32 observations
    "ffn_mult": 2,
    # --- ssm ---
    "ssm_d_state": 16,
    "ssm_dt_rank": 16,
    # --- frozen_dsm ---
    "dsm_hidden": 192,
    "dsm_mem_dim": 64,
}

BASELINE_MODELS: Tuple[str, ...] = (
    "frame_cnn", "tcn", "gru", "lstm", "transformer", "ssm", "frozen_dsm",
)

#: output heads, same names/shapes as ``PixelEventNet``
SEQ_KEYS: Tuple[str, ...] = (
    "event", "onset", "count", "presence", "ready", "gesture",
)

#: what dt actually does in each baseline (reported by the smoke test)
DT_MODES: Dict[str, str] = {
    "frame_cnn": "feature(log1p dt) only; stateless per frame",
    "tcn": "feature(log1p dt) only; conv operator is discrete-time "
           "(31-frame RF == 31 observations)",
    "gru": "feature(log1p dt) only; recurrence advances per observation",
    "lstm": "feature(log1p dt) only; recurrence advances per observation",
    "transformer": "feature(log1p dt) only; per-layer W=32 window "
                   "(effective multi-layer RF = 63 observations)",
    "ssm": "feature(log1p dt) + real dt drives state decay exp(A*Delta*dt)",
    "frozen_dsm": "feature(log1p dt) + real dt drives DSM ZOH/memory decay",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _safe_dt(dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Physical dt in seconds: float32, finite, non-negative, exactly 0 at pads."""
    d = torch.nan_to_num(dt.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return d.clamp_min(0.0) * mask.to(d.dtype)


def _last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``x [B,T,C]`` at the last valid step -> ``[B,C]`` (zeros if no step)."""
    b, _, c = x.shape
    idx = (mask.sum(dim=1) - 1).clamp_min(0)                 # [B]
    picked = x.gather(1, idx.view(b, 1, 1).expand(b, 1, c)).squeeze(1)
    any_valid = mask.any(dim=1, keepdim=True)
    return torch.where(any_valid, picked, torch.zeros_like(picked))


def _finalise(raw: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Zero the emitted outputs at padded steps, add ``count_final``.

    Zeroing padded outputs (instead of leaving whatever was computed there) makes
    the mask contract checkable: every value the model emits for a step that was
    not observed is exactly 0.0, and nothing from a padded step can reach a real
    step (feature zeroing + state hold + windowed-core compaction).
    """
    m = mask.unsqueeze(-1)
    seq = {k: torch.where(m, raw[k], torch.zeros_like(raw[k])) for k in SEQ_KEYS}
    seq["count_final"] = _last_valid(seq["count"], mask)
    return seq


def _empty_seq(b: int, c: int, g: int, p: int,
               device, dtype) -> Dict[str, torch.Tensor]:
    """Shape-safe output for a zero-length window (T == 0)."""
    def z(n: int) -> torch.Tensor:
        return torch.zeros(b, 0, n, device=device, dtype=dtype)
    return {"event": z(c), "onset": z(1), "count": z(c), "presence": z(p),
            "ready": z(1), "gesture": z(g),
            "count_final": torch.zeros(b, c, device=device, dtype=dtype)}


def compact_frames(x: torch.Tensor, mask: torch.Tensor,
                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Stable partition: valid frames first (order kept), padded frames last.

    This is how a *windowed* operator gets exact mask semantics: running the
    causal conv / causal attention on the compacted sequence means the window
    holds the last W **observations**, exactly like a streaming KV cache, and a
    masked gap is equivalent to the truncated sequence.

    Returns ``(x_sorted, unpermute_index)``; ``unpermute_index`` is ``None``
    when the mask is all-true (identity permutation, no gather needed).
    """
    if bool(mask.all()):
        return x, None
    perm = torch.argsort(~mask, dim=1, stable=True)          # valid (False) first
    inv = torch.argsort(perm, dim=1, stable=True)
    xs = x.gather(1, perm.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return xs, inv


def uncompact_frames(h: torch.Tensor, inv: Optional[torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`compact_frames` (no-op when ``inv is None``)."""
    if inv is None:
        return h
    return h.gather(1, inv.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


# --------------------------------------------------------------------------- #
# shared building blocks (same for every baseline)
# --------------------------------------------------------------------------- #
class FrozenScout(nn.Module):
    """A Scout that is frozen *and* stays in eval mode under ``model.train()``.

    Both requirements matter for the ``frozen_dsm`` ablation:

    * every parameter has ``requires_grad=False`` (so ``count_params`` -- which
      counts trainable params only -- is reported next to the total);
    * ``BatchNorm2d`` must not update its running statistics while the rest of
      the model trains, otherwise the "frozen encoder" would keep drifting.
      ``train()`` is overridden to pin the wrapper and the Scout to eval, and
      the forward runs under ``torch.no_grad``.
    """

    is_frozen = True

    def __init__(self, scout: nn.Module):
        super().__init__()
        self.scout = scout
        for p in self.scout.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):                      # noqa: D102
        super().train(False)          # this wrapper is always eval
        self.scout.eval()             # BN keeps its running statistics
        return self

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.scout(img)


class PixelFrameEncoder(nn.Module):
    """Scout + shared neck + fusion -> one feature vector per frame.

    Identical module (and therefore identical parameters/MACs) in the six
    trainable baselines.  ``frame_cnn``/``tcn``/``gru``/``lstm``/``transformer``
    /``ssm`` all see exactly the same per-frame features; only the temporal
    operator downstream differs.
    """

    def __init__(self, cfg: Dict[str, object], neck: bool = True):
        super().__init__()
        f4 = int(PIXEL_SPEC["scout_f4_dim"])
        gdim = int(PIXEL_SPEC["global_dim"])
        # shared visual front end: the SAME factory the main model uses, so an
        # ImageNet-initialised scout benefits every architecture equally
        self.scout = make_scout(dict(cfg))
        # Reuse pixel_event's depthwise-separable block (never re-implemented).
        self.neck = dwise_sep(f4, f4) if neck else nn.Identity()
        self.fuse = nn.Sequential(
            nn.Linear(gdim + f4, int(cfg["feat_dim"])), nn.GELU())
        self.feat_dim = int(cfg["feat_dim"])

    def _fuse(self, f4: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f4 = self.neck(f4)
        return self.fuse(torch.cat([g, f4.mean(dim=(2, 3))], dim=-1))

    def _one(self, img: torch.Tensor) -> torch.Tensor:
        return self._fuse(*self.scout(img))

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        if t == 0:
            return imgs.new_zeros(b, 0, self.feat_dim)
        if self.training:
            # ---- per-frame in training mode -------------------------------- #
            # BatchNorm2d computes its statistics over the *batch* dimension.
            # Folding T into the batch (B*T) would let the statistics of frame
            # t+1 influence the output at frame t, i.e. break strict causality
            # in training mode.  One frame per call costs little and keeps the
            # model exactly causal in both modes.
            return torch.stack([self._one(imgs[:, i]) for i in range(t)], dim=1)
        # ---- eval: BN uses running statistics, so batching over T is exact -- #
        f4, g = self.scout(imgs.reshape(b * t, *imgs.shape[2:]))
        return self._fuse(f4, g).reshape(b, t, self.feat_dim)


class TimeFeatureStage(nn.Module):
    """Append ``log1p(dt)`` + validity flag, then LayerNorm.

    Mirrors ``src.models.base.InputNorm`` exactly (zero padded features, real
    physical dt as a bounded log feature, no normalisation across time/batch).
    This is the uniform dt path for every baseline; the operator may *also* use
    dt structurally (ssm, frozen_dsm).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim) + 2)

    def forward(self, z: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        z = z.masked_fill(~mask.unsqueeze(-1), 0.0)
        feats = torch.cat([z, torch.log1p(dt).unsqueeze(-1),
                           mask.to(z.dtype).unsqueeze(-1)], dim=-1)
        return self.norm(feats)


class BaselineHead(nn.Module):
    """THE shared multi-task head -- one class, one width, used by all 7 models.

    LayerNorm -> 2-layer MLP (``head_hidden``) -> one linear per task.  Identical
    structure and parameter count in every baseline (395,714 params at
    ``core_dim=192, head_hidden=512``), so no parameter difference between
    baselines can come from the head.

    ``count`` is the current cumulative-count estimate; it is passed through
    ``softplus`` because a cumulative count is non-negative by definition (ours
    is a non-negative state variable).  ``event``/``onset``/``presence``/
    ``ready``/``gesture`` are logits.
    """

    def __init__(self, in_dim: int, n_event: int, n_gesture: int, n_presence: int,
                 hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim))
        self.mlp = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), int(hidden)), nn.GELU(),
        )
        self.event = nn.Linear(int(hidden), int(n_event))
        self.onset = nn.Linear(int(hidden), 1)
        self.count = nn.Linear(int(hidden), int(n_event))
        self.presence = nn.Linear(int(hidden), int(n_presence))
        self.ready = nn.Linear(int(hidden), 1)
        self.gesture = nn.Linear(int(hidden), int(n_gesture))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.mlp(self.norm(h))
        return {
            "event": self.event(z),
            "onset": self.onset(z),
            "count": F.softplus(self.count(z)),
            "presence": self.presence(z),
            "ready": self.ready(z),
            "gesture": self.gesture(z),
        }


# --------------------------------------------------------------------------- #
# temporal cores
# --------------------------------------------------------------------------- #
class CausalConvBlock(nn.Module):
    """One left-padded (causal) dilated Conv1d + GELU + residual + LayerNorm.

    Left padding of ``(k-1)*dilation`` is exactly the amount needed for the
    output at t to depend on inputs ``<= t`` only; with kernel 3 one block adds
    ``2*dilation`` frames to the receptive field.  The LayerNorm is over channels
    at each time step (never over time), so training-mode statistics cannot leak
    the future.
    """

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (int(kernel) - 1) * int(dilation)
        self.conv = nn.Conv1d(ch, ch, int(kernel), dilation=int(dilation))
        self.norm = nn.LayerNorm(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # [B,T,C]
        y = x.transpose(1, 2)                               # [B,C,T]
        y = F.pad(y, (self.pad, 0))                         # causal padding
        y = F.gelu(self.conv(y)).transpose(1, 2)
        return self.norm(x + y)


class CausalTCN(nn.Module):
    """Dilated causal TCN; receptive field = 1 + 2*sum(dilations) = 31 frames.

    One causal conv per dilation level, so the task's example formula
    ``1 + 2*(1+2+4+8) = 31`` holds exactly (verified empirically in the smoke
    notes: a perturbation at t reaches outputs t..t+30 and nothing later).
    """

    def __init__(self, ch: int, kernel: int, dilations: Sequence[int]):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CausalConvBlock(int(ch), int(kernel), int(d)) for d in dilations])
        self.receptive_field = 1 + 2 * sum(int(d) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        return h


class _GRUCore(nn.Module):
    """``nn.GRU`` maths (single layer, via ``nn.GRUCell``) stepped over time.

    Stepping is required by the contract: a padded step must leave the state
    untouched, which ``nn.GRU``'s fused sequence path cannot express.  Same
    parameters/equations as ``nn.GRU(input_size=dim, hidden_size=dim,
    batch_first=True)``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.cell = nn.GRUCell(self.dim, self.dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.dim)
        outs: List[torch.Tensor] = []
        for i in range(t):
            h_new = self.cell(x[:, i], h)
            h = torch.where(mask[:, i].unsqueeze(-1), h_new, h)   # mask -> hold
            outs.append(h)
        return torch.stack(outs, dim=1)


class _LSTMCore(nn.Module):
    """``nn.LSTM`` maths (single layer, via ``nn.LSTMCell``) stepped over time."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.cell = nn.LSTMCell(self.dim, self.dim)

    def forward(self, x: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.dim)
        c = x.new_zeros(b, self.dim)
        outs: List[torch.Tensor] = []
        for i in range(t):
            h_new, c_new = self.cell(x[:, i], (h, c))
            m = mask[:, i].unsqueeze(-1)
            h = torch.where(m, h_new, h)                          # mask -> hold
            c = torch.where(m, c_new, c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class CausalSelfAttention(nn.Module):
    """Sliding-window causal self-attention (bounded W-frame KV cache).

    The allowed set is the strict lower triangle **including the diagonal**,
    restricted to the last ``window`` observations: ``0 <= t - j < W``.
    Positions are encoded *only* by a learned per-offset relative bias, so there
    is no absolute positional embedding and no way for a future token to be
    read.  Building the dense [T,T] score matrix is just how a batched forward
    evaluates the sliding window; the streaming state is the last W keys/values.
    """

    def __init__(self, dim: int, n_heads: int, window: int):
        super().__init__()
        if int(dim) % int(n_heads) != 0:
            raise ValueError("core_dim must be divisible by n_heads")
        self.dim = int(dim)
        self.h = int(n_heads)
        self.hd = self.dim // self.h
        self.window = int(window)
        self.in_proj = nn.Linear(self.dim, 3 * self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.rel_bias = nn.Parameter(torch.zeros(self.h, self.window))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # [B,T,C]
        b, t, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(b, t, self.h, self.hd).transpose(1, 2)    # [B,H,T,hd]
        k = k.view(b, t, self.h, self.hd).transpose(1, 2)
        v = v.view(b, t, self.h, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hd)
        pos = torch.arange(t, device=x.device)
        rel = pos[:, None] - pos[None, :]                     # [T,T]; j<=t <=> rel>=0
        allowed = (rel >= 0) & (rel < self.window)            # incl. diagonal
        bias = self.rel_bias[:, rel.clamp(0, self.window - 1)]  # [H,T,T]
        scores = scores + bias.unsqueeze(0)
        scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), -1e9)
        att = torch.softmax(scores, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, t, self.dim)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: causal-windowed attention + 2x FFN."""

    def __init__(self, dim: int, n_heads: int, window: int, ffn_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, window)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(ffn_mult) * dim), nn.GELU(),
            nn.Linear(int(ffn_mult) * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    """``n_layers`` causal-windowed transformer blocks + final LayerNorm."""

    def __init__(self, dim: int, n_heads: int, window: int, n_layers: int,
                 ffn_mult: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, window, ffn_mult)
            for _ in range(int(n_layers))])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


class SelectiveSSM(nn.Module):
    """Minimal pure-torch selective state-space model (S6 / Mamba skeleton).

    No ``mamba-ssm`` dependency, no CUDA kernels: an explicit per-step scan over
    a diagonal state of shape ``[dim, d_state]``.  This is the *continuous-time*
    variant: the input-selected timescale ``Delta_t = softplus(W_d u_t)`` is a
    multiplier on the **real** elapsed seconds, and the transfer matrix is the
    exact zero-order-hold decay of a leaky integrator::

        h_t = exp(A * Delta_t * dt_t) * h_{t-1}
              + (1 - exp(A * Delta_t * dt_t)) * (B_t u_t)
        y_t = C_t . h_t + D * u_t
        out_t = W_o( y_t * silu(z_t) )

    with ``A = -exp(A_log)`` in 1/s.  Because the decay is ``exp(rate * dt)``, dt
    is a *structural* input: a 2x longer gap produces exactly the squared decay
    (semigroup in dt for fixed Delta_t) instead of one extra un-decayed step.
    ``A_log`` is initialised to rates 1..d_state /s and ``dt_proj`` bias to
    softplus ~ 0.1 s, so the effective per-second rates start in a sane range.
    """

    def __init__(self, dim: int, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.dim = int(dim)
        self.d_state = int(d_state)
        self.dt_rank = int(dt_rank)
        self.in_proj = nn.Linear(self.dim, 2 * self.dim)      # u and gate z
        self.x_proj = nn.Linear(self.dim, self.dt_rank + 2 * self.d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.dim)
        rates = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(rates).repeat(self.dim, 1))
        self.D = nn.Parameter(torch.ones(self.dim))
        self.out_proj = nn.Linear(self.dim, self.dim)
        with torch.no_grad():
            self.dt_proj.bias.fill_(math.log(math.expm1(0.1)))   # softplus -> 0.1 s

    def scan(self, x: torch.Tensor, dt: torch.Tensor,
             mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(out [B,T,dim], states [B,T,dim,d_state])``.

        ``forward`` is ``scan(...)[0]``; the state trajectory is exposed so the
        smoke test can prove the mask really holds the state at padded steps.
        """
        b, t, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)              # [B,T,dim]
        proj = self.x_proj(u)                                # [B,T,rank+2*S]
        delta = F.softplus(self.dt_proj(proj[..., :self.dt_rank]))  # [B,T,dim]
        b_proj = proj[..., self.dt_rank:self.dt_rank + self.d_state]
        c_proj = proj[..., self.dt_rank + self.d_state:]
        decay = torch.exp(self.A_log.reshape(1, 1, self.dim, self.d_state)
                          * delta.unsqueeze(-1)
                          * dt.view(b, t, 1, 1))             # [B,T,dim,d_state]
        h = x.new_zeros(b, self.dim, self.d_state)
        outs: List[torch.Tensor] = []
        state: List[torch.Tensor] = []
        for i in range(t):
            a = decay[:, i]                                  # [B,dim,d_state]
            h_new = a * h + (1.0 - a) * (
                u[:, i].unsqueeze(-1) * b_proj[:, i].unsqueeze(1))
            h = torch.where(mask[:, i].view(b, 1, 1), h_new, h)        # mask -> hold
            read = (c_proj[:, i].unsqueeze(1) * h).sum(dim=-1)
            y = (read + self.D * u[:, i]) * F.silu(z[:, i])
            outs.append(self.out_proj(y))
            state.append(h)
        return torch.stack(outs, dim=1), torch.stack(state, dim=1)

    def forward(self, x: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self.scan(x, dt, mask)[0]


# --------------------------------------------------------------------------- #
# the baseline models
# --------------------------------------------------------------------------- #
class PixelBaseline(nn.Module):
    """Common trunk: shared encoder -> dt feature stage -> temporal core -> head.

    Subclasses only provide ``_temporal`` (the actual baseline difference) and
    the core module itself.
    """

    name = "baseline"
    dt_mode = "feature"
    #: True for cores with an explicit persistent state (mask -> hold the state)
    stateful = False

    def __init__(self, cfg: Dict[str, object], n_event_classes: int = 25,
                 n_gesture: int = 10, n_presence: int = 4,
                 build_encoder: bool = True):
        super().__init__()
        self.cfg = dict(cfg)
        self.C = int(n_event_classes)
        self.G = int(n_gesture)
        self.P = int(n_presence)
        if build_encoder:
            self.encoder = PixelFrameEncoder(self.cfg)
            self.tf = TimeFeatureStage(self.encoder.feat_dim)
            self.proj = nn.Sequential(
                nn.Linear(self.encoder.feat_dim + 2, int(self.cfg["core_dim"])),
                nn.GELU(),
            )
        self.head = BaselineHead(int(self.cfg["core_dim"]), self.C, self.G, self.P,
                                 hidden=int(self.cfg["head_hidden"]))

    # ---- pieces ---- #
    def _encode(self, imgs: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.tf(self.encoder(imgs), dt, mask))

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def core_dim(self) -> int:
        return int(self.cfg["core_dim"])

    def extra_macs_per_frame(self) -> float:
        """MACs per frame of ops that module hooks cannot see (override)."""
        return 0.0

    # ---- diagnostics ---- #
    def _encode_imgs(self, imgs: torch.Tensor, dt: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        dt = _safe_dt(dt, mask)
        return self._encode(imgs.masked_fill(~mask.view(b, t, 1, 1, 1), 0.0),
                            dt, mask)

    def state_sequence(self, imgs: torch.Tensor, dt: torch.Tensor,
                       mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Persistent-state trajectory ``[B,T,...]`` for stateful cores.

        Used by the smoke test to prove ``mask=False`` steps hold the state
        bit-exactly; ``None`` for cores without a persistent state.
        """
        return None

    # ---- contract ---- #
    def forward(self, imgs: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        mask = mask.to(torch.bool)
        dt = _safe_dt(dt, mask)
        if t == 0:
            seq = _empty_seq(b, self.C, self.G, self.P, imgs.device, imgs.dtype)
            return seq, {"hidden": imgs.new_zeros(b, 0, self.core_dim)}
        # Padded frames never reach a convolution: NaN/garbage there (a) cannot
        # poison BatchNorm statistics and (b) cannot influence any real step.
        imgs = imgs.masked_fill(~mask.view(b, t, 1, 1, 1), 0.0)
        x = self._encode(imgs, dt, mask)                    # [B,T,core_dim]
        h = self._temporal(x, dt, mask)                     # [B,T,core_dim]
        seq = _finalise(self.head(h), mask)
        return seq, {"hidden": h}


class FrameCNNBaseline(PixelBaseline):
    """Memoryless floor: one independent prediction per frame (no state at all)."""

    name = "frame_cnn"
    dt_mode = DT_MODES["frame_cnn"]

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        return x                                            # per-frame, no state


class TCNBaseline(PixelBaseline):
    """Bounded-memory causal TCN over per-frame Scout features (RF = 31)."""

    name = "tcn"
    dt_mode = DT_MODES["tcn"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = int(self.cfg["core_dim"])
        self.core = CausalTCN(dim, int(self.cfg["tcn_kernel"]),
                              tuple(self.cfg["tcn_dilations"]))

    @property
    def receptive_field(self) -> int:
        return int(self.core.receptive_field)

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        xs, inv = compact_frames(x, mask)                   # window over observations
        return uncompact_frames(self.core(xs), inv)


class GRUBaseline(PixelBaseline):
    """Single-layer GRU over per-frame Scout features (mask holds the state)."""

    name = "gru"
    dt_mode = DT_MODES["gru"]
    stateful = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = _GRUCore(int(self.cfg["core_dim"]))

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        return self.core(x, dt, mask)

    def state_sequence(self, imgs: torch.Tensor, dt: torch.Tensor,
                       mask: torch.Tensor) -> Optional[torch.Tensor]:
        # for a GRU the emitted hidden state *is* the output vector
        dt = _safe_dt(dt, mask)
        return self.core(self._encode_imgs(imgs, dt, mask), dt, mask)


class LSTMBaseline(PixelBaseline):
    """Single-layer LSTM over per-frame Scout features (mask holds h and c)."""

    name = "lstm"
    dt_mode = DT_MODES["lstm"]
    stateful = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = _LSTMCore(int(self.cfg["core_dim"]))

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        return self.core(x, dt, mask)

    def state_sequence(self, imgs: torch.Tensor, dt: torch.Tensor,
                       mask: torch.Tensor) -> Optional[torch.Tensor]:
        dt = _safe_dt(dt, mask)
        return self.core(self._encode_imgs(imgs, dt, mask), dt, mask)


class TransformerBaseline(PixelBaseline):
    """2-layer causal sliding-window (W=32) transformer over observations."""

    name = "transformer"
    dt_mode = DT_MODES["transformer"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = TransformerStack(
            int(self.cfg["core_dim"]), int(self.cfg["n_heads"]),
            int(self.cfg["window"]), int(self.cfg["n_layers"]),
            int(self.cfg["ffn_mult"]))

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        xs, inv = compact_frames(x, mask)                   # KV cache over observations
        return uncompact_frames(self.core(xs), inv)

    @property
    def receptive_field(self) -> int:
        """Effective multi-layer receptive field, in observations.

        Each layer attends to the last W observations, so stacking ``n_layers``
        of them gives ``1 + n_layers*(W-1)``: 63 for n_layers=2, W=32.  The KV
        cache of each individual layer is still bounded at W (the task's
        "bounded KV cache"); stacking is what widens the *effective* span, and
        this number is what a perturbation at t can reach.
        """
        return 1 + int(self.cfg["n_layers"]) * (int(self.cfg["window"]) - 1)

    def extra_macs_per_frame(self) -> float:
        # QK^T and AV products are matmuls, invisible to nn.Linear hooks:
        # per layer per frame: 2 * W * core_dim
        return 2.0 * int(self.cfg["n_layers"]) * int(self.cfg["window"]) \
            * int(self.cfg["core_dim"])


class SSMBaseline(PixelBaseline):
    """Minimal selective SSM (S6) with real dt driving the state decay."""

    name = "ssm"
    dt_mode = DT_MODES["ssm"]
    stateful = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = SelectiveSSM(int(self.cfg["core_dim"]),
                                 int(self.cfg["ssm_d_state"]),
                                 int(self.cfg["ssm_dt_rank"]))

    def _temporal(self, x: torch.Tensor, dt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        return self.core(x, dt, mask)

    def extra_macs_per_frame(self) -> float:
        # scan: a*h, (1-a)*B u, C.h, plus the u*B outer product
        return 4.0 * int(self.cfg["core_dim"]) * int(self.cfg["ssm_d_state"])

    def state_sequence(self, imgs: torch.Tensor, dt: torch.Tensor,
                       mask: torch.Tensor) -> Optional[torch.Tensor]:
        dt = _safe_dt(dt, mask)
        _out, states = self.core.scan(
            self._encode_imgs(imgs, dt, mask), dt, mask)
        return states


class FrozenDSMBaseline(PixelBaseline):
    """Frozen Scout -> GAP(stride-4 map) -> DSM core (previous-generation cell).

    Continuity ablation: "the same frozen encoder plus the previous temporal
    core".  The Scout is *frozen* (``requires_grad=False``, pinned to eval so its
    BatchNorm statistics cannot drift); the pooled 64-d stride-4 feature vector
    goes straight into ``src.models.dsm.DSM`` (``core(x, dt, mask)``), which
    brings its own continuous-time slow state + matrix memory, and then into the
    same shared head as every other baseline.

    A frozen encoder must come from somewhere.  Use
    :meth:`load_scout_from_pixel_event` to copy a trained ``PixelEventNet``
    backbone into it before training this baseline; with random frozen weights
    the model is intentionally meaningless (it is an ablation harness, not a
    contender).
    """

    name = "frozen_dsm"
    dt_mode = DT_MODES["frozen_dsm"]
    stateful = True

    def __init__(self, cfg: Dict[str, object], n_event_classes: int = 25,
                 n_gesture: int = 10, n_presence: int = 4):
        super().__init__(cfg, n_event_classes, n_gesture, n_presence,
                         build_encoder=False)
        # Instead of the shared trainable encoder: frozen Scout + no neck, and
        # feed the GAP of the stride-4 map (not the fused feature) to DSM.
        f4 = int(PIXEL_SPEC["scout_f4_dim"])
        self.scout = FrozenScout(Scout(in_ch=int(PIXEL_SPEC["in_ch"]),
                                       ch=tuple(PIXEL_SPEC["scout_ch"]),
                                       f4=f4, gdim=int(PIXEL_SPEC["global_dim"])))
        self.pool_dim = f4
        self.core = DSM(in_dim=self.pool_dim,
                        hidden=int(self.cfg["dsm_hidden"]),
                        mem_dim=int(self.cfg["dsm_mem_dim"]))
        self.head = BaselineHead(int(self.cfg["dsm_hidden"]), self.C, self.G,
                                 self.P, hidden=int(self.cfg["head_hidden"]))
        if int(self.cfg["dsm_hidden"]) != int(self.cfg["core_dim"]):
            raise ValueError("frozen_dsm needs dsm_hidden == core_dim so the "
                             "shared head keeps the same input width")

    # ---- API ---- #
    def load_scout_from_pixel_event(self, pixel_event_model: nn.Module) -> None:
        """Copy ``PixelEventNet.eye.scout`` weights into the frozen Scout."""
        self.scout.scout.load_state_dict(pixel_event_model.eye.scout.state_dict())
        for p in self.scout.parameters():
            p.requires_grad_(False)

    @property
    def core_dim(self) -> int:
        return int(self.cfg["dsm_hidden"])

    def extra_macs_per_frame(self) -> float:
        # DSM matrix-memory products outside nn.Linear: M q, M k, v k^T, M q
        return 4.0 * float(int(self.cfg["dsm_mem_dim"]) ** 2)

    def state_sequence(self, imgs: torch.Tensor, dt: torch.Tensor,
                       mask: torch.Tensor) -> Optional[torch.Tensor]:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        dt = _safe_dt(dt, mask)
        x = self._encode(imgs.masked_fill(~mask.view(b, t, 1, 1, 1), 0.0))
        _out, aux = self.core(x, dt, mask)
        return aux["hidden"]

    # ---- pieces ---- #
    def _encode(self, imgs: torch.Tensor) -> torch.Tensor:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        if t == 0:
            return imgs.new_zeros(b, 0, self.pool_dim)
        # FrozenScout is pinned to eval, so BN uses running statistics and the
        # batch over T is exactly per-frame equivalent (and cheap).
        f4, _g = self.scout(imgs.reshape(b * t, *imgs.shape[2:]))
        return f4.mean(dim=(2, 3)).reshape(b, t, self.pool_dim)

    def _encode_imgs(self, imgs: torch.Tensor, dt: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        return self._encode(imgs.masked_fill(~mask.view(b, t, 1, 1, 1), 0.0))

    # ---- contract ---- #
    def forward(self, imgs: torch.Tensor, dt: torch.Tensor,
                mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict]:
        b, t = int(imgs.shape[0]), int(imgs.shape[1])
        mask = mask.to(torch.bool)
        dt = _safe_dt(dt, mask)
        if t == 0:
            seq = _empty_seq(b, self.C, self.G, self.P, imgs.device, imgs.dtype)
            return seq, {"hidden": imgs.new_zeros(b, 0, self.core_dim)}
        imgs = imgs.masked_fill(~mask.view(b, t, 1, 1, 1), 0.0)
        x = self._encode(imgs)                              # [B,T,64] pooled f4
        # DSM's LoopCore.forward does its own safe_dt/mask handling, appends
        # log1p(dt) + validity flag (InputNorm) and holds the state at pads.
        h, _aux = self.core(x, dt, mask)                    # [B,T,192]
        seq = _finalise(self.head(h), mask)
        return seq, {"hidden": h}


#: name -> class
BASELINE_CLASSES = {
    "frame_cnn": FrameCNNBaseline,
    "tcn": TCNBaseline,
    "gru": GRUBaseline,
    "lstm": LSTMBaseline,
    "transformer": TransformerBaseline,
    "ssm": SSMBaseline,
    "frozen_dsm": FrozenDSMBaseline,
}


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def build_baseline(name: str, n_event_classes: int = 25, n_gesture: int = 10,
                   n_presence: int = 4,
                   cfg: Optional[Dict[str, object]] = None) -> nn.Module:
    """Build one pixel baseline.

    Parameters
    ----------
    name : one of :data:`BASELINE_MODELS`
    n_event_classes : C for the ``event``/``count`` heads (default 25)
    n_gesture : G for the ``gesture`` head (default 10)
    n_presence : P for the ``presence`` head (default 4)
    cfg : overrides for :data:`BASELINE_SPEC` (e.g. ``{"window": 16}``)
    """
    key = str(name)
    if key not in BASELINE_CLASSES:
        raise ValueError(f"unknown baseline {name!r}; have {list(BASELINE_MODELS)}")
    spec = dict(BASELINE_SPEC)
    spec.update(cfg or {})
    return BASELINE_CLASSES[key](spec, n_event_classes=n_event_classes,
                                 n_gesture=n_gesture, n_presence=n_presence)


def baseline_param_counts(m: nn.Module) -> Tuple[int, int]:
    """``(total params, trainable params)``.  Frozen params count in the total."""
    total = sum(p.numel() for p in m.parameters())
    return int(total), int(count_params(m))


# --------------------------------------------------------------------------- #
# MAC counting (all ops; the project's count_macs sees Conv2d/Linear only)
# --------------------------------------------------------------------------- #
def _hook_conv2d(total: List[float], mod, _inp, out) -> None:
    total[0] += (float(out.numel()) * mod.kernel_size[0] * mod.kernel_size[1]
                 * mod.in_channels // mod.groups)


def _hook_conv1d(total: List[float], mod, _inp, out) -> None:
    total[0] += (float(out.numel()) * mod.kernel_size[0]
                 * mod.in_channels // mod.groups)


def _hook_linear(total: List[float], mod, _inp, out) -> None:
    total[0] += float(out.numel()) * mod.in_features


def _hook_gru_cell(total: List[float], mod, _inp, out) -> None:
    calls = float(out.numel()) // float(mod.hidden_size)
    total[0] += calls * 3.0 * (mod.input_size * mod.hidden_size
                               + mod.hidden_size * mod.hidden_size)


def _hook_lstm_cell(total: List[float], mod, _inp, out) -> None:
    calls = float(out[0].numel()) // float(mod.hidden_size) if isinstance(out, tuple) \
        else float(out.numel()) // float(mod.hidden_size)
    total[0] += calls * 4.0 * (mod.input_size * mod.hidden_size
                               + mod.hidden_size * mod.hidden_size)


def count_macs_full(m: nn.Module, imgs: torch.Tensor, dt: torch.Tensor,
                    mask: torch.Tensor) -> float:
    """Exact forward-MAC count over *all* ops, per analysed frame.

    Hooks Conv2d / Conv1d / Linear / GRUCell / LSTMCell and adds the model's
    manual ``extra_macs_per_frame`` (attention products, SSM scan, DSM matrix
    reads).  This is the honest number; ``pixel_event.count_macs`` (Conv2d +
    Linear only) is reported next to it for direct comparison with the 81.25 M
    / frame reference.
    """
    total: List[float] = [0.0]
    handles = []
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(
                functools.partial(_hook_conv2d, total)))
        elif isinstance(mod, nn.Conv1d):
            handles.append(mod.register_forward_hook(
                functools.partial(_hook_conv1d, total)))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(
                functools.partial(_hook_linear, total)))
        elif isinstance(mod, nn.GRUCell):
            handles.append(mod.register_forward_hook(
                functools.partial(_hook_gru_cell, total)))
        elif isinstance(mod, nn.LSTMCell):
            handles.append(mod.register_forward_hook(
                functools.partial(_hook_lstm_cell, total)))
    was_training = m.training
    m.eval()
    with torch.no_grad():
        m(imgs, dt, mask)
    if was_training:
        m.train()
    for h in handles:
        h.remove()
    extra = float(m.extra_macs_per_frame()) if hasattr(m, "extra_macs_per_frame") else 0.0
    # per *sequence*, per frame: hooks accumulate over batch and time
    n_frames = max(1, int(imgs.shape[1])) * max(1, int(imgs.shape[0]))
    return total[0] / n_frames + extra


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _rand_inputs(b: int, t: int, device, seed: int = 0,
                 res: int = 160) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random window: irregular physical dt (dt[:,0]=0), all steps valid."""
    g = torch.Generator().manual_seed(int(seed))
    imgs = torch.rand(b, t, 1, res, res, generator=g)
    dt = torch.full((b, t), 0.1)
    dt = dt * torch.exp(0.4 * torch.randn(b, t, generator=g)).clamp(0.3, 3.0)
    dt[:, 0] = 0.0
    mask = torch.ones(b, t, dtype=torch.bool)
    return imgs.to(device), dt.to(device), mask.to(device)


def _prefix_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor],
                 k: int) -> float:
    """max |a-b| over the first ``k`` steps of every emitted head."""
    worst = 0.0
    for key in SEQ_KEYS:
        d = (a[key][:, :k].float() - b[key][:, :k].float()).abs().max()
        worst = max(worst, float(d))
    return worst


def _perturb(imgs: torch.Tensor, dt: torch.Tensor, k: int,
             seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perturb frames and dt from step ``k`` onwards (same shapes)."""
    g = torch.Generator().manual_seed(int(seed))
    imgs2 = imgs.clone()
    dt2 = dt.clone()
    b, t = imgs.shape[0], imgs.shape[1]
    imgs2[:, k:] = torch.rand(b, t - k, 1, imgs.shape[3], imgs.shape[4],
                              generator=g).to(imgs2.device)
    dt2[:, k:] = dt2[:, k:] * (1.0 + torch.rand(b, t - k, generator=g).to(dt2.device))
    return imgs2, dt2


def _check_shapes(seq: Dict[str, torch.Tensor], b: int, t: int, c: int, g: int,
                  p: int) -> None:
    want = {"event": (b, t, c), "onset": (b, t, 1), "count": (b, t, c),
            "presence": (b, t, p), "ready": (b, t, 1), "gesture": (b, t, g),
            "count_final": (b, c)}
    for key, shape in want.items():
        got = tuple(seq[key].shape)
        if got != shape:
            raise AssertionError(f"seq[{key!r}] shape {got} != {shape}")


def _smoke_one(name: str, imgs: torch.Tensor, dt: torch.Tensor,
               mask: torch.Tensor, device, verbose: bool = True) -> Dict[str, object]:
    """Run every contract check for one baseline; returns a result row."""
    b, t = int(imgs.shape[0]), int(imgs.shape[1])
    model = build_baseline(name).to(device)
    c, g, p = model.C, model.G, model.P

    total, trainable = baseline_param_counts(model)
    # project counter: hooks count every batch element and time step, so it
    # returns a *batch-scaled* per-frame number (B=2 doubles it).  Report the
    # per-sequence value (independent of B) next to the raw value so the number
    # is directly comparable with the task's 81.25 M reference (raw, B=2).
    macs_raw = float(count_macs(model, imgs, dt, mask))
    macs_ref = macs_raw / max(1, b)
    macs_full = float(count_macs_full(model, imgs, dt, mask))    # all ops

    # ---- shapes + finiteness ----
    model.eval()
    with torch.no_grad():
        seq, aux = model(imgs, dt, mask)
    _check_shapes(seq, b, t, c, g, p)
    for key, val in seq.items():
        if not torch.isfinite(val).all():
            raise AssertionError(f"{name}: non-finite seq[{key!r}]")
    if not torch.isfinite(aux["hidden"]).all():
        raise AssertionError(f"{name}: non-finite hidden state")

    # ---- strict causality (eval + train mode, bit-exact) ----
    causal = {}
    for mode in ("eval", "train"):
        (model.train() if mode == "train" else model.eval())
        with torch.no_grad():
            base, _ = model(imgs, dt, mask)
            worst = 0.0
            for k in (1, t // 2, t - 1):
                if k <= 0:
                    continue
                i2, d2 = _perturb(imgs, dt, k, seed=1000 + k)
                out2, _ = model(i2, d2, mask)
                worst = max(worst, _prefix_diff(base, out2, k))
        causal[mode] = worst
        if worst != 0.0:
            raise AssertionError(
                f"{name}: causality violated in {mode} mode (max|d|={worst:.3e}); "
                "perturbing t>=k must leave out[:, :k] bit-identical")

    # ---- mask semantics ----
    # Two different claims are tested and reported at different tolerances:
    #   * information from a masked frame is inert -> must be EXACTLY 0 (frame
    #     content is replaced by zeros before any op; stateful cores hold the
    #     state; windowed cores compact the window);
    #   * removing a masked gap (so the conv/attention batch composition
    #     changes) reproduces the truncated sequence -> <= 1e-6.  The only
    #     difference is float non-associativity from a differently-sized conv
    #     batch in eval mode; the project's own test_mask uses 1e-6 for exactly
    #     this check, and in training mode the encoder loops per frame, so it
    #     is exact there.
    model.eval()
    tail = max(1, t // 3)
    mask_pad = mask.clone()
    mask_pad[:, t - tail:] = False
    dt_pad = dt * mask_pad.to(dt.dtype)
    exact_tol, shape_tol = 0.0, 1e-6
    with torch.no_grad():
        clean, _ = model(imgs, dt_pad, mask_pad)
        # (a) garbage/NaN in padded frames must not change any output at all
        garbage = imgs.clone()
        garbage[:, t - tail:] = float("nan")
        garb, _ = model(garbage, dt_pad, mask_pad)
        pad_inert = 0.0
        for key in SEQ_KEYS:
            pad_inert = max(pad_inert, float((clean[key] - garb[key]).abs().max()))
        # (b) padded steps emit exactly 0 (so they cannot leak through outputs)
        pad_zero = 0.0
        for key in SEQ_KEYS:
            pad_zero = max(pad_zero, float(clean[key][:, t - tail:].abs().max()))
        # (c) trailing pads == truncated sequence on the real steps
        trunc, _ = model(imgs[:, :t - tail], dt[:, :t - tail], mask[:, :t - tail])
        trunc_diff = 0.0
        for key in SEQ_KEYS:
            trunc_diff = max(trunc_diff,
                             float((clean[key][:, :t - tail] - trunc[key]).abs().max()))
        # (d) a *middle* masked gap: garbage there is inert everywhere ...
        gap = t // 2
        mask_gap = mask.clone()
        mask_gap[:, gap] = False
        dt_gap = dt * mask_gap.to(dt.dtype)
        imgs_gap = imgs.clone()
        imgs_gap[:, gap] = float("nan")
        out_gap, _gap_aux = model(imgs_gap, dt_gap, mask_gap)
        out_gap_ref, _ = model(imgs, dt_gap, mask_gap)
        gap_inert = 0.0
        for key in SEQ_KEYS:
            gap_inert = max(gap_inert,
                            float((out_gap[key] - out_gap_ref[key]).abs().max()))
        # ... the persistent state at the masked step is held bit-exactly ...
        state_hold = None
        if getattr(model, "stateful", False):
            states = model.state_sequence(imgs_gap, dt_gap, mask_gap)
            state_hold = float((states[:, gap] - states[:, gap - 1]).abs().max())
        # ... and the valid observations reproduce the truncated sequence.
        keep = torch.ones(t, dtype=torch.bool)
        keep[gap] = False
        out_comp, _ = model(imgs[:, keep], dt_gap[:, keep], mask_gap[:, keep])
        gap_diff = 0.0
        for key in SEQ_KEYS:
            gap_diff = max(gap_diff,
                           float((out_gap_ref[key][:, keep] - out_comp[key]).abs().max()))
    if pad_inert != exact_tol:
        raise AssertionError(f"{name}: padded garbage changed outputs "
                             f"({pad_inert:.3e})")
    if gap_inert != exact_tol:
        raise AssertionError(f"{name}: garbage at a masked step changed outputs "
                             f"({gap_inert:.3e})")
    if state_hold is not None and state_hold != exact_tol:
        raise AssertionError(f"{name}: state changed at a masked step "
                             f"({state_hold:.3e})")
    if pad_zero != exact_tol:
        raise AssertionError(f"{name}: padded steps emitted non-zero outputs "
                             f"({pad_zero:.3e})")
    if trunc_diff > shape_tol:
        raise AssertionError(f"{name}: trailing pads changed real outputs "
                             f"({trunc_diff:.3e})")
    if gap_diff > shape_tol:
        raise AssertionError(f"{name}: masked gap != truncated sequence "
                             f"({gap_diff:.3e})")

    # ---- dt sensitivity (dt must reach the model) ----
    with torch.no_grad():
        out_dt, _ = model(imgs, dt, mask)
        out_zero, _ = model(imgs, torch.zeros_like(dt), mask)
    dt_sens = max(float((out_dt[k] - out_zero[k]).abs().max()) for k in SEQ_KEYS)
    if dt_sens <= 1e-6:
        raise AssertionError(f"{name}: dt has no effect (max|d|={dt_sens:.3e})")

    # ---- frozen_dsm: frozen scout + eval-in-train BN ----
    frozen_info = ""
    if name == "frozen_dsm":
        model.train()
        if any(q.requires_grad for q in model.scout.parameters()):
            raise AssertionError("frozen_dsm: Scout has trainable parameters")
        if model.scout.scout.training:
            raise AssertionError("frozen_dsm: frozen Scout left train mode")
        before = {k: v.clone() for k, v in model.scout.scout.named_buffers()
                  if "running" in k}
        with torch.no_grad():
            model(imgs, dt, mask)
        after = {k: v for k, v in model.scout.scout.named_buffers()
                 if "running" in k}
        bn_drift = max(float((before[k] - after[k]).abs().max()) for k in before)
        frozen_info = (f"frozen scout: requires_grad=False, eval-in-train=True, "
                       f"running-stat drift={bn_drift:.1e}")

    # ---- gradients reach every trainable parameter group ----
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        seq_g, _ = model(imgs, dt, mask)
        loss = sum(seq_g[k].float().sum() for k in SEQ_KEYS)
        loss = loss + seq_g["count_final"].float().sum()
        loss.backward()
    n_no_grad = 0
    for q in model.parameters():
        if q.requires_grad and (q.grad is None or float(q.grad.abs().sum()) == 0.0):
            n_no_grad += 1
    if n_no_grad != 0:
        # BatchNorm running stats are buffers, not parameters; every trainable
        # parameter must receive a gradient from the summed head loss.
        raise AssertionError(f"{name}: {n_no_grad} trainable params got no gradient")
    model.zero_grad(set_to_none=True)

    scout_mod = model.scout if hasattr(model, "scout") else model.encoder.scout
    row = {
        "name": name,
        "params": total,
        "trainable": trainable,
        "head_params": int(sum(q.numel() for q in model.head.parameters())),
        "scout_params": int(sum(q.numel() for q in scout_mod.parameters())),
        "macs_ref": macs_ref,
        "macs_raw": macs_raw,
        "macs_full": macs_full,
        "causal_eval": causal["eval"],
        "causal_train": causal["train"],
        "dt_sens": dt_sens,
        "pad_inert": pad_inert,
        "gap_inert": gap_inert,
        "state_hold": state_hold,
        "pad_zero": pad_zero,
        "trunc": trunc_diff,
        "gap": gap_diff,
        "dt_mode": str(getattr(model, "dt_mode", "")),
        "frozen": frozen_info,
        "shapes": {k: tuple(seq[k].shape) for k in seq},
        "receptive_field": getattr(model, "receptive_field", None),
    }
    if verbose:
        print(f"[{name}]")
        print(f"  params total={total:,} trainable={trainable:,} | "
              f"MACs/frame (per sequence): count_macs={macs_ref / 1e6:.3f} M "
              f"all-ops={macs_full / 1e6:.3f} M  "
              f"[count_macs raw @B={b}: {macs_raw / 1e6:.3f} M]")
        print(f"  shapes: " + " ".join(f"{k}={tuple(v)}" for k, v in
                                       row["shapes"].items()))
        print(f"  causality max|d|: eval={causal['eval']:.1e} "
              f"train={causal['train']:.1e} (must be 0.0e+00)")
        sh = "n/a" if state_hold is None else f"{state_hold:.1e}"
        print(f"  mask: pad-inert={pad_inert:.1e} gap-inert={gap_inert:.1e} "
              f"state-hold={sh} pad-emit={pad_zero:.1e} "
              f"trunc={trunc_diff:.1e} mid-gap={gap_diff:.1e} | "
              f"dt-sensitivity={dt_sens:.1e}")
        print(f"  dt mode: {row['dt_mode']}")
        rf = row["receptive_field"]
        if rf:
            print(f"  effective receptive field: {rf} observations")
        if frozen_info:
            print(f"  {frozen_info}")
    return row


def _print_reference(imgs: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor,
                     device) -> None:
    """Print our model's params/MACs so the comparison is anchored."""
    try:
        from .pixel_event import build_pixel_model
        ref = build_pixel_model("pixel_event").to(device)
        total, trainable = baseline_param_counts(ref)
        macs = float(count_macs(ref, imgs, dt, mask))
        b = max(1, int(imgs.shape[0]))
        print(f"[reference] pixel_event params={total:,} "
              f"trainable={trainable:,} | MACs/frame (per sequence)="
              f"{macs / b / 1e6:.3f} M | same count_macs counter raw @B={b}: "
              f"{macs / 1e6:.3f} M")
        print("            (the task's 81.25 M reference is this RAW counter at "
              "B=2, i.e. ~40.6-41.7 M per sequence per frame)")
    except Exception as exc:                                  # pragma: no cover
        print(f"[reference] could not measure pixel_event: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Smoke-test the pixel-to-event baselines: params, MACs per "
                    "frame, output shapes, bit-exact causality, mask semantics.")
    ap.add_argument("--names", nargs="*", default=list(BASELINE_MODELS),
                    choices=list(BASELINE_MODELS), help="subset of baselines")
    ap.add_argument("--B", type=int, default=2, help="batch size")
    ap.add_argument("--T", type=int, default=10, help="window length")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference", dest="reference", action="store_true",
                    default=True, help="also measure our pixel_event model")
    ap.add_argument("--no-reference", dest="reference", action="store_false")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    imgs, dt, mask = _rand_inputs(int(args.B), int(args.T), device, int(args.seed))
    if args.reference:
        _print_reference(imgs, dt, mask, device)
    print(f"window: B={args.B} T={args.T} dt~0.1s irregular, all steps valid; "
          f"device={device}")
    print("")

    rows = []
    for name in args.names:
        rows.append(_smoke_one(name, imgs, dt, mask, device))
        print("")

    print("=" * 110)
    print(f"{'model':<14}{'params':>11}{'trainable':>11}"
          f"{'MACs/f seq':>12}{'all-ops/f':>12}"
          f"{'causal eval':>13}{'causal train':>14}{'dt-sens':>11}")
    print("-" * 110)
    for r in rows:
        print(f"{r['name']:<14}{r['params']:>11,}{r['trainable']:>11,}"
              f"{r['macs_ref'] / 1e6:>9.2f} M{r['macs_full'] / 1e6:>10.2f} M"
              f"{r['causal_eval']:>13.1e}{r['causal_train']:>14.1e}"
              f"{r['dt_sens']:>11.1e}")
    print("=" * 110)
    if not rows:
        print("no baselines selected")
        return 0
    heads = {r["head_params"] for r in rows}
    scouts = {r["scout_params"] for r in rows}
    if len(heads) != 1:
        raise AssertionError(f"head params differ between baselines: {sorted(heads)}")
    if len(scouts) != 1:
        raise AssertionError(f"Scout params differ between baselines: {sorted(scouts)}")
    print(f"shared head: {rows[0]['head_params']:,} params (identical in all "
          f"{len(rows)} models) | shared Scout: {rows[0]['scout_params']:,} "
          f"params (identical in all {len(rows)} models)")
    print("ALL CHECKS PASSED: causality is 0.0e+00 in eval and train mode, "
          "padded steps are inert and emit exactly 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
