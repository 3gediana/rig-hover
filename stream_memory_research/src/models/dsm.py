"""DSM -- Dual-State Memory (this project's model): slow state + fast memory.

Three variants live here, all sharing the same slow-state operator and the same
per-frame contract:

``dsm``        slow state ``h`` + matrix fast memory ``M in R^{d_v x d_k}``
``dsm_slot``   slow state ``h`` + ``N`` content-addressed slots (key_j, val_j, t_last_j)
``window``     ``dsm_slot`` with a hard per-object route and a hard time window
               (the "only the object in the window is updated" ablation)

Equations (per frame, ``Delta`` = physical seconds, clamped to ``dt_clamp_max``)--

Slow state, exact zero-order-hold in physical time (same ``Delta`` as everything
else, so the whole system has a continuous-time limit)::

    g_h   = exp(-Gamma_h * Delta)                     # Gamma per-dim, tau in [0.2, 2] s
    h^-   = g_h * h
    g     = sigmoid(W_g [z, h])
    h     = h^- + g * (tanh(W_u [z, h^-]) - h^-)      # gated write onto the decayed state

Fast memory, multi-timescale decay + residual (Householder) write::

    Lambda = softplus(raw)[group(d_v)]                # G groups, rates [8,2,0.5,0.02] /s
    M^-   = M * exp(-Lambda * Delta)                  # decay is a pure function of Delta
    q,k,v = W_q[z,h], unit(W_k z), W_v[z,h]
    r     = read(M^-, q)                              # linear: M q ; bilinear: 2-hop
    e     = v - r
    beta  = sigmoid(w.[z,h,r,log1p Delta] + w_s log1p(||e||))     # strength <- surprise
    M     = M^- - beta (M^- k) k^T + beta v k^T       # == (I - beta k k^T) M^- + beta v k^T
    o     = read(M, q)

    out   = LayerNorm(MLP([h, o, z]))                 # (+ tok_pool, not used yet)

Why each choice (see reports/LIT-BRIEF.md, which the Lead red-teamed this
against):

* **Residual (Householder) write, not additive**: the delta rule removes the old
  value of the addressed key before writing, so a key written many times
  overwrites instead of accumulating interference.  The additive
  ``write_rule="outer"`` exists *only* so that capacity can be attributed in the
  ablation -- the published ablation for this family is literally "decay hurts
  memory retention" (residual vs decay-but-no-residual: 98.8 vs 30.4 at 8K).
* **Multi-timescale decay, one ~non-decaying group**: a single rate cannot both
  suppress cross-talk (must be fast) and survive a multi-second occlusion (must
  be slow).  Rates are anchored in physical units (1/s) so they cannot be
  rescaled away by the optimiser -- otherwise learned decay would silently
  cancel ``Delta`` and the cross-sampling-rate experiment would measure nothing.
* **Decay and write strength are separate**: decay is a pure function of
  ``Delta``; write strength comes from surprise ``||v - r||`` times a content
  gate.  Sharing one gate would couple forgetting to writing.
* **Every time constant is a rate in 1/s times physical seconds**; nothing is
  indexed by observation count unless ``h_by_index=True``, which exists as the
  "no continuous-time limit" ablation.

``h_by_index=True``  slow state advances with ``Delta = 1`` per observation.
``mem_off=True``     only the slow state (out = MLP([h, z])) -- "no fast memory".
``slow_off=True``    only the fast memory (out = MLP([o, z])) -- "no slow state".
``read_mode``        "linear" (capacity ~ d_k) or "bilinear" (second-order read).
``hard_write``       straight-through binary write mask (``beta -> 1[beta>delta]``;
                     the slot variant instead uses ``u_j = 1[||v - val_j|| > delta]``
                     so that "only change what changed" is bit-exactly true).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.contract import apply_mask
from .base import (COMMON_SPEC, GroupedDecay, InputNorm, LoopCore, TimeDecay,
                   hard_mask_st, hard_st, unpack_cfg)
from .deltanet import MemRead, unit_norm

DSM_SPEC = {
    **COMMON_SPEC,
    # --- memory / write rule ---
    "write_rule": "delta",        # "delta" (Householder residual) | "outer"
    "read_mode": "linear",        # "linear" | "bilinear"
    "mem_dim": 64,                # d_k = d_v
    "decay": True,                # False -> Lambda == 0
    "n_decay_groups": 4,          # multi-timescale groups over d_v
    # --- slowed / ablation switches ---
    "mem_off": False,             # dsm only: drop the fast memory
    "slow_off": False,            # drop the slow state
    "h_by_index": False,          # slow state per observation index, not per Delta
    "hard_write": False,          # straight-through binary write mask
    "write_threshold": 0.5,       # delta for hard_write
    "beta_exp": False,            # stability switch only: beta -> 1 - exp(-beta)
    "dt_clamp_max": 10.0,         # seconds, clamped BEFORE any exp (not Lambda*Delta)
    # --- slow state ---
    "tau_h_min": 0.2,             # seconds
    "tau_h_max": 2.0,             # seconds
    # --- read ---
    "hop_scale": 1.0,             # bilinear read: weight of the second hop
    # --- object tokens (optional; off = bit-identical to the tokenless model) ---
    "use_tok": False,             # address the fast memory by object tokens
    "tok_dim": 17,                # per-token feature width (17 for the synth tasks)
    "tok_vis_channel": 16,        # token channel holding visibility (>0 => valid)
    "tok_read_pool": "mean_max",  # "mean_max" | "mean"
}

SLOT_SPEC = {
    **{k: v for k, v in DSM_SPEC.items()
       if k not in ("read_mode", "n_decay_groups", "mem_off")},
    "n_slots": 12,
    "slot_dim": None,             # None -> use mem_dim
    "slot_hard_route": False,
    "route_mode": "content",      # "content" | "positional" | "none"
    "tau_age_init": 4.0,          # seconds, age prior in the routing score
    "window_s": 0.0,              # >0 : only slots younger than this are candidates
}


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #
class SlowState(nn.Module):
    """Per-dimension continuous-time slow state with a gated write.

    ``h^- = exp(-Gamma * Delta) * h``  then  ``h = h^- + g*(tanh(W_u[z,h^-]) - h^-)``.

    ``Gamma`` is a per-dimension rate (``tau_h`` log-spaced in [tau_h_min,
    tau_h_max]) and ``Delta`` is physical seconds, so the decay is an exact
    semigroup and the cell has a proper continuous-time limit: as ``Delta -> 0``
    the state change goes to zero proportionally to ``Delta``.
    """

    def __init__(self, hidden: int, tau_h_min: float = 0.2, tau_h_max: float = 2.0):
        super().__init__()
        self.H = int(hidden)
        lo, hi = math.log10(max(tau_h_min, 1e-2)), math.log10(max(tau_h_max, tau_h_min * 1.001))
        tau = torch.logspace(lo, hi, self.H)
        self.dec = TimeDecay(self.H, init_tau=tau)
        self.g_z = nn.Linear(self.H, self.H)
        self.g_h = nn.Linear(self.H, self.H, bias=False)
        self.u_z = nn.Linear(self.H, self.H)
        self.u_h = nn.Linear(self.H, self.H, bias=False)

    def prep(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"g_z": self.g_z(z), "u_z": self.u_z(z)}

    def step(self, t: int, pre, h_prev, dts_t, mask_t, by_index: bool = False):
        if by_index:                                  # "no continuous-time limit"
            d = torch.ones_like(dts_t)
        else:
            d = dts_t
        g_h = self.dec.step_factor(d)                 # [B,H] in (0,1]
        h_pre = h_prev * g_h                          # ZOH decay
        g = torch.sigmoid(pre["g_z"][:, t] + self.g_h(h_prev))
        h_new = h_pre + g * (torch.tanh(pre["u_z"][:, t] + self.u_h(h_pre)) - h_pre)
        return apply_mask(h_new, h_prev, mask_t), g, h_pre


class MatrixMemory(nn.Module):
    """``M in R^{d_v x d_k}`` with multi-timescale decay and a delta-rule write.

    Read modes (operator shared with ``deltanet`` via :class:`MemRead`, so the two
    models differ in their *state*, not in their read):
      ``linear``   r = M q                -- capacity ~ d_k separable associations
      ``bilinear`` o = diag(U M V)        -- d_k*d_v independent bilinear functionals,
                   the capacity-theory operator (query-independent by design)
      ``two_hop``  r = M q + hop*M u(M^T M q) -- a third, quadratic-in-M operator;
                   never to be reported as "bilinear".
    Writes are identical across read modes, so a linear-vs-bilinear comparison
    isolates read capacity.

    ``write_rule="outer"`` (additive ``M + beta v k^T``) exists purely as the
    capacity-attribution ablation; ``"delta"`` (default) is the residual write
    that makes repeated writes of one key overwrite instead of interfere.
    """

    def __init__(self, hidden: int, mem_dim: int, n_decay_groups: int = 4,
                 write_rule: str = "delta", read_mode: str = "linear",
                 decay: bool = True, beta_exp: bool = False,
                 hard_write: bool = False, write_threshold: float = 0.5,
                 hop_scale: float = 1.0, eps: float = 1e-6,
                 use_tok: bool = False, tok_dim: int = 17, tok_vis_channel: int = 16,
                 read_pool: str = "mean_max"):
        super().__init__()
        if write_rule not in ("delta", "outer"):
            raise ValueError(f"write_rule must be 'delta'|'outer', got {write_rule!r}")
        self.H, self.dk, self.dv = int(hidden), int(mem_dim), int(mem_dim)
        self.write_rule, self.read_mode = write_rule, read_mode
        self.beta_exp, self.hard_write = bool(beta_exp), bool(hard_write)
        self.write_threshold, self.eps = float(write_threshold), float(eps)
        self.q_z = nn.Linear(self.H, self.dk)
        self.q_h = nn.Linear(self.H, self.dk, bias=False)
        self.k_z = nn.Linear(self.H, self.dk)
        self.v_z = nn.Linear(self.H, self.dv)
        self.v_h = nn.Linear(self.H, self.dv, bias=False)
        self.beta_z = nn.Linear(self.H, 1)
        self.beta_h = nn.Linear(self.H, 1, bias=False)
        self.beta_r = nn.Linear(self.dv, 1, bias=False)
        self.beta_t = nn.Parameter(torch.zeros(1))
        self.beta_s = nn.Parameter(torch.zeros(1))
        self.beta_b = nn.Parameter(torch.zeros(1))
        self.dec = GroupedDecay(n_decay_groups, self.dv,
                                learnable=bool(decay), decay=bool(decay))
        self.readmod = MemRead(self.dk, self.dv, mode=read_mode,
                               hop_scale=float(hop_scale), eps=self.eps)
        # --- object tokens: the fast memory is addressed by object, not by frame ---
        self.use_tok = bool(use_tok)
        self.tok_dim = int(tok_dim)
        self.tok_vis = tok_vis_channel
        self.read_pool = str(read_pool)
        self.extra_out = 0
        if self.use_tok:
            self.tok_k = nn.Linear(self.tok_dim, self.dk)     # k_j = unit(W_k tok_j)
            self.tok_v = nn.Linear(self.tok_dim, self.dv)     # v_j = W_v tok_j
            self.tok_beta_c = nn.Linear(self.H, 1)            # content term (shared)
            self.tok_beta_r = nn.Linear(self.dv, 1, bias=False)   # surprise term
            self.tok_beta_s = nn.Parameter(torch.zeros(1))
            self.tok_beta_b = nn.Parameter(torch.zeros(1))
            self.extra_out = self.dv * (2 if self.read_pool == "mean_max" else 1)

    # ---- state ---- #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {"M": torch.zeros(B, self.dv, self.dk, device=device)}

    # ---- state-independent projections (hoisted out of the loop) ---- #
    def prep(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"q_z": self.q_z(z), "k_z": unit_norm(self.k_z(z)),
                "v_z": self.v_z(z), "beta_z": self.beta_z(z)}

    # ---- read (identical operator to deltanet: see MemRead) ---- #
    def read(self, M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.readmod(M, q)

    # ---- object tokens ---- #
    def prep_tok(self, tok, tok_valid=None, mask=None):
        """``tok [B,T,K,Dt]`` -> per-object unit keys, values and validity.

        ``valid_j`` is ``tok_j[..., tok_vis_channel] > 0`` (the synth tasks put
        visibility there); an explicit ``tok_valid [B,T,K]`` overrides it, and a
        padded frame (``mask=False``) invalidates all of its objects.
        """
        if tok is None:
            return None
        if tok.shape[-1] != self.tok_dim:
            raise ValueError(f"use_tok=True expects tok_dim={self.tok_dim} "
                             f"(cfg), got {tok.shape[-1]}")
        if tok_valid is not None:
            valid = tok_valid.to(torch.bool)
        elif self.tok_vis is not None and self.tok_vis < tok.shape[-1]:
            valid = tok[..., self.tok_vis] > 0
        else:
            valid = torch.ones(tok.shape[:3], dtype=torch.bool, device=tok.device)
        if mask is not None:
            valid = valid & mask.unsqueeze(-1)
        tv = torch.where(valid.unsqueeze(-1), tok, torch.zeros_like(tok))
        return {"kt": unit_norm(self.tok_k(tv), self.eps), "vt": self.tok_v(tv),
                "valid": valid}

    def _pool(self, o_j: torch.Tensor, valid: torch.Tensor):
        """Mean (and optionally max) over the valid objects only."""
        v = valid.to(o_j.dtype).unsqueeze(-1)
        cnt = v.sum(dim=1).clamp_min(1.0)
        pooled = [(o_j * v).sum(dim=1) / cnt]
        if self.read_pool == "mean_max":
            mx = o_j.masked_fill(v <= 0, -1e4).amax(dim=1)
            has = (v.sum(dim=1) > 0.5).to(o_j.dtype)
            pooled.append(mx * has)                     # no valid object -> 0
        return pooled

    # ---- one step ---- #
    def step(self, t: int, pre, h: torch.Tensor, M_prev: torch.Tensor,
             dts_t: torch.Tensor, logdt_t: torch.Tensor, mask_t: torch.Tensor,
             tok_pre=None):
        q = pre["q_z"][:, t] + self.q_h(h)
        k = pre["k_z"][:, t]                                  # already unit-norm
        v = pre["v_z"][:, t] + self.v_h(h)
        f = self.dec.step_factor(dts_t)                       # [B,dv] in (0,1]
        M_dec = M_prev * f.unsqueeze(-1)                      # decay the rows of M
        r = self.read(M_dec, q)
        e = v - r
        surprise = e.norm(dim=-1, keepdim=True)               # [B,1]  ||v - r||
        beta = torch.sigmoid(pre["beta_z"][:, t] + self.beta_h(h) + self.beta_r(r)
                             + self.beta_t * logdt_t
                             + self.beta_s * torch.log1p(surprise) + self.beta_b)
        if self.beta_exp:
            beta = 1.0 - torch.exp(-beta)                     # stability switch only
        if self.hard_write:
            beta = hard_st(beta, self.write_threshold)
        vk = v.unsqueeze(-1) * k.unsqueeze(1)                 # [B,dv,dk] = v k^T
        if self.write_rule == "delta":                        # exact Householder delta
            Mk = torch.einsum("bvd,bd->bv", M_dec, k).unsqueeze(-1)
            M_new = M_dec - beta.unsqueeze(-1) * Mk * k.unsqueeze(1) \
                + beta.unsqueeze(-1) * vk
        else:                                                 # additive outer product
            M_new = M_dec + beta.unsqueeze(-1) * vk
        # ---- object tokens: one Householder delta per valid object, applied in
        # parallel against the same post-frame memory (order-independent; an
        # invalid object contributes an exactly-zero rank-1 term) ---- #
        if tok_pre is not None:
            kt, vt = tok_pre["kt"][:, t], tok_pre["vt"][:, t]      # [B,K,dk] [B,K,dv]
            val = tok_pre["valid"][:, t].to(M_new.dtype)           # [B,K]
            Mk = torch.einsum("bvd,bkd->bkv", M_new, kt)           # [B,K,dv] = M k_j
            sj = (vt - Mk).norm(dim=-1, keepdim=True)              # [B,K,1]
            bj = torch.sigmoid(
                pre["beta_z"][:, t].unsqueeze(1) + self.tok_beta_c(h).unsqueeze(1)
                + self.tok_beta_r(Mk)
                + self.tok_beta_s * torch.log1p(sj) + self.tok_beta_b) * val.unsqueeze(-1)
            M_new = M_new + torch.einsum("bk,bkv,bkd->bvd",
                                         bj.squeeze(-1), vt - Mk, kt)
        M = apply_mask(M_new, M_prev, mask_t)
        o = self.read(M, q)
        aux = {
            "read": o,
            "write_strength": beta.squeeze(-1),
            "write_gate": beta.squeeze(-1),
            "surprise": surprise.squeeze(-1),
            "mem_norm": M.flatten(1).norm(dim=-1),
        }
        if tok_pre is not None:
            o_j = torch.einsum("bvd,bkd->bkv", M, kt)              # [B,K,dv] per object
            pooled = self._pool(o_j, tok_pre["valid"][:, t])
            aux["slot_read"] = o_j                                 # [B,K,dv]
            aux["tok_write_gate"] = bj.squeeze(-1)                 # [B,K]
            aux["tok_pool_mean"] = pooled[0]
            if len(pooled) > 1:
                aux["tok_pool_max"] = pooled[1]
        return o, M, aux

    def stats(self) -> Dict[str, float]:
        out = {f"lam_{k}": v for k, v in self.dec.stats().items()}
        out["lam_mean"] = float(self.dec.lam().detach().mean())
        return out


class SlotMemory(nn.Module):
    """``N`` content-addressed slots ``(key_j, val_j, t_last_j, valid_j)``.

    Read:  ``w_j = softmax_j(scale * <q, unit(k_j)> / sqrt(D) + log age_prior_j)``,
    ``r = sum_j w_j val_j``, with ``log age_prior_j = -age_j / tau_age``
    (``age_j`` in **physical seconds**, from the real dt) so that slots which have
    not been written for a long time lose the competition for the read.

    Write: gated delta onto the routed slots,
    ``val_j += w_j * beta * u_j * (v - val_j)`` with
    ``u_j = 1[||v - val_j|| > delta]`` (straight-through) when ``hard_write``.
    Because ``w_j`` is *exactly* 0 for un-routed slots (hard route) and ``u_j`` is
    *exactly* 0 for unchanged slots, all state updates are exactly-zero additive
    terms: unselected slots are bit-identical before/after (unit-tested).

    Routing ablations: ``route_mode`` = "content" (content address), "positional"
    (round-robin by observation index), "none" (uniform).  ``slot_hard_route``
    replaces the softmax by a straight-through top-1.
    """

    def __init__(self, hidden: int, n_slots: int = 12, slot_dim: int = 64,
                 route_mode: str = "content", slot_hard_route: bool = False,
                 hard_write: bool = False, write_threshold: float = 0.5,
                 beta_exp: bool = False, tau_age_init: float = 4.0,
                 write_rule: str = "delta", decay: bool = True,
                 window_s: float = 0.0, eps: float = 1e-6,
                 use_tok: bool = False, tok_dim: int = 17, tok_vis_channel: int = 16,
                 read_pool: str = "mean_max"):
        super().__init__()
        if route_mode not in ("content", "positional", "none"):
            raise ValueError(f"route_mode must be content|positional|none, got {route_mode!r}")
        if write_rule not in ("delta", "outer"):
            raise ValueError(f"write_rule must be 'delta'|'outer', got {write_rule!r}")
        self.H, self.N, self.D = int(hidden), int(n_slots), int(slot_dim)
        self.route_mode = route_mode
        self.hard_route = bool(slot_hard_route)
        self.hard_write = bool(hard_write)
        self.write_threshold = float(write_threshold)
        self.write_rule = str(write_rule)
        self.beta_exp = bool(beta_exp)
        self.decay = bool(decay)          # False -> no recency term in the routing score
        self.window_s = float(window_s)
        self.eps = 1e-6
        self.q_z = nn.Linear(self.H, self.D)
        self.q_h = nn.Linear(self.H, self.D, bias=False)
        self.v_z = nn.Linear(self.H, self.D)
        self.v_h = nn.Linear(self.H, self.D, bias=False)
        self.beta_z = nn.Linear(self.H, 1)
        self.beta_h = nn.Linear(self.H, 1, bias=False)
        self.beta_r = nn.Linear(self.D, 1, bias=False)
        self.beta_t = nn.Parameter(torch.zeros(1))
        self.beta_s = nn.Parameter(torch.zeros(1))
        self.beta_b = nn.Parameter(torch.zeros(1))
        # learned anchor keys; the *state* keys start from these and can drift
        self.keys0 = nn.Parameter(torch.randn(self.N, self.D) * 0.1)
        self.key_gate = nn.Parameter(torch.tensor(-4.0))      # sigma ~ 0.018: keys barely move
        self.score_scale = nn.Parameter(torch.ones(1))
        self.empty_bias = nn.Parameter(torch.tensor(-2.0))
        self.age = TimeDecay(1, init_tau=float(tau_age_init))  # scalar tau_age
        # --- object tokens (optional; off = bit-identical to the tokenless model) ---
        self.use_tok = bool(use_tok)
        self.tok_dim = int(tok_dim)
        self.tok_vis = tok_vis_channel
        self.read_pool = str(read_pool)
        self.extra_out = 0
        if self.use_tok:
            self.tok_k = nn.Linear(self.tok_dim, self.D)       # k_j = unit(W_k tok_j)
            self.tok_v = nn.Linear(self.tok_dim, self.D)       # v_j = W_v tok_j
            self.tok_beta_c = nn.Linear(self.H, 1)
            self.tok_beta_r = nn.Linear(self.D, 1, bias=False)
            self.tok_beta_s = nn.Parameter(torch.zeros(1))
            self.tok_beta_b = nn.Parameter(torch.zeros(1))
            self.extra_out = self.D * (2 if self.read_pool == "mean_max" else 1)

    # ---- object tokens ---- #
    def prep_tok(self, tok, tok_valid=None, mask=None):
        """``tok [B,T,K,Dt]`` -> per-object unit keys, values and validity."""
        if tok is None:
            return None
        if tok.shape[-1] != self.tok_dim:
            raise ValueError(f"use_tok=True expects tok_dim={self.tok_dim} "
                             f"(cfg), got {tok.shape[-1]}")
        if tok_valid is not None:
            valid = tok_valid.to(torch.bool)
        elif self.tok_vis is not None and self.tok_vis < tok.shape[-1]:
            valid = tok[..., self.tok_vis] > 0
        else:
            valid = torch.ones(tok.shape[:3], dtype=torch.bool, device=tok.device)
        if mask is not None:
            valid = valid & mask.unsqueeze(-1)
        tv = torch.where(valid.unsqueeze(-1), tok, torch.zeros_like(tok))
        return {"kt": unit_norm(self.tok_k(tv), self.eps), "vt": self.tok_v(tv),
                "valid": valid}

    def _pool(self, o_j: torch.Tensor, valid: torch.Tensor):
        v = valid.to(o_j.dtype).unsqueeze(-1)
        cnt = v.sum(dim=1).clamp_min(1.0)
        pooled = [(o_j * v).sum(dim=1) / cnt]
        if self.read_pool == "mean_max":
            mx = o_j.masked_fill(v <= 0, -1e4).amax(dim=1)
            has = (v.sum(dim=1) > 0.5).to(o_j.dtype)
            pooled.append(mx * has)
        return pooled

    # ---- state ---- #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {
            "keys": self.keys0.unsqueeze(0).expand(B, -1, -1).clone().to(device),
            "vals": torch.zeros(B, self.N, self.D, device=device),
            "t_last": torch.zeros(B, self.N, device=device),
            "valid": torch.zeros(B, self.N, device=device),
            # cumulative physical time, carried in the state so that the streaming
            # path (one frame per call) sees the same clock as the batched path
            "t_cur": torch.zeros(B, 1, device=device),
        }

    def prep(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"q_z": self.q_z(z), "v_z": self.v_z(z), "beta_z": self.beta_z(z)}

    # ---- routing ---- #
    def route(self, t: int, pre, q, state, t_cur_t):
        """Routing weights ``w [B,N]`` (rows sum to 1) and per-slot ages ``age [B,N]``."""
        B, dev = pre["q_z"].shape[0], pre["q_z"].device
        age = (t_cur_t.unsqueeze(1) - state["t_last"]).clamp_min(0.0)      # [B,N] seconds
        valid = state["valid"]                                            # [B,N]
        zeros = torch.zeros_like(age)
        fresh = -age / self.age.tau().reshape(()) if self.decay else zeros
        log_prior = torch.where(valid > 0.5, fresh, self.empty_bias.expand_as(age))

        if self.route_mode == "positional":                               # by observation index
            idx = torch.full((B,), t % self.N, dtype=torch.long, device=dev)
            return F.one_hot(idx, self.N).to(pre["q_z"].dtype), age
        if self.route_mode == "none":                                     # uniform
            return torch.full((B, self.N), 1.0 / self.N, device=dev,
                              dtype=pre["q_z"].dtype), age

        keys = unit_norm(state["keys"], self.eps)                         # [B,N,D]
        scores = self.score_scale * torch.einsum("bnd,bd->bn", keys, q) / math.sqrt(self.D)
        scores = scores + log_prior
        if self.window_s > 0:                                             # hard time window
            in_win = (age <= self.window_s) & (valid > 0.5)
            empty = valid <= 0.5
            scores = torch.where(in_win, scores, torch.full_like(scores, -1e9))
            scores = torch.where(empty, log_prior, scores)                # empty slots stay candidates
        w = torch.softmax(scores, dim=-1)
        if self.hard_route:                                               # straight-through top-1
            onehot = F.one_hot(w.argmax(dim=-1), self.N).to(w.dtype)
            w = w + (onehot - w).detach()
        return w, age

    # ---- one step ---- #
    def step(self, t: int, pre, h, state, dts_t, t_cur_t, logdt_t, mask_t):
        q = pre["q_z"][:, t] + self.q_h(h)                     # [B,D]
        v = pre["v_z"][:, t] + self.v_h(h)                     # [B,D]
        w, age = self.route(t, pre, q, state, t_cur_t)         # [B,N]
        vals = state["vals"]                                   # [B,N,D]
        r = torch.einsum("bn,bnd->bd", w, vals)                # soft content read
        e = v.unsqueeze(1) - vals                              # [B,N,D] per-slot residual
        surprise_j = e.norm(dim=-1)                            # [B,N]
        surprise = (w * surprise_j).sum(dim=-1, keepdim=True)  # [B,1] routed surprise
        beta = torch.sigmoid(pre["beta_z"][:, t] + self.beta_h(h) + self.beta_r(r)
                             + self.beta_t * logdt_t
                             + self.beta_s * torch.log1p(surprise) + self.beta_b)  # [B,1]
        if self.beta_exp:
            beta = 1.0 - torch.exp(-beta)
        # "only change what changed": per-slot hard mask; forward is exactly 0/1 so
        # an unselected slot receives an exactly-zero additive update.
        if self.hard_write:
            u = hard_mask_st(surprise_j, self.write_threshold,
                             temp=max(0.5 * self.write_threshold, 0.05))
        else:
            u = torch.ones_like(surprise_j)
        mass = w * beta * u                                    # [B,N] write strength
        src = e if self.write_rule == "delta" else v.unsqueeze(1).expand_as(vals)
        vals_new = vals + mass.unsqueeze(-1) * src              # gated delta / outer write
        # keys drift toward the query only where written
        q_hat = unit_norm(q, self.eps)
        keys_new = state["keys"] + torch.sigmoid(self.key_gate) * mass.unsqueeze(-1) * \
            (q_hat.unsqueeze(1) - state["keys"])
        t_last_new = state["t_last"] + mass * (t_cur_t.unsqueeze(1) - state["t_last"])
        valid_new = (state["valid"] + mass * (1.0 - state["valid"])).clamp(0.0, 1.0)
        # masked hold: a padded step leaves the whole slot bank untouched
        m3 = mask_t.unsqueeze(-1).unsqueeze(-1)
        vals_new = torch.where(m3, vals_new, vals)
        keys_new = torch.where(m3, keys_new, state["keys"])
        t_last_new = apply_mask(t_last_new, state["t_last"], mask_t)
        valid_new = apply_mask(valid_new, state["valid"], mask_t)
        state_new = {"keys": keys_new, "vals": vals_new,
                     "t_last": t_last_new, "valid": valid_new,
                     "vals_prev": vals}      # last step's pre-write values (probe only)
        o = torch.einsum("bn,bnd->bd", w, vals_new)            # read after write
        aux = {
            "read": o,
            "write_gate": mass,                                # [B,N]
            "write_strength": beta.squeeze(-1),                # [B]
            "route": w.unsqueeze(1),                           # [B,1,N] -> stacked [B,T,1,N]
            "slot_read": w.unsqueeze(-1) * vals_new,           # [B,N,D] per-slot readout
            "surprise": surprise.squeeze(-1),                  # [B]
            "write_mask": u,                                   # [B,N]
            "mem_norm": vals_new.flatten(1).norm(dim=-1),
            "slot_age": age,
        }
        return o, state_new, aux

    # ---- token path: the object bank is addressed by tokens, not by the frame ---- #
    def step_tok(self, t: int, pre, h, state, dts_t, t_cur_t, logdt_t, mask_t, tok_pre):
        """Write/read the slot bank from **object tokens**.

        ``route_mode="content"``  each token competes for slots through a softmax
            over ``<k_j, unit(key_n)>`` (so slots become object identity slots);
        ``route_mode="none"``/``"positional"``  a fixed ``j -> j % N`` assignment,
            which is the "no multi-object routing" ablation.

        Every per-object quantity is masked by token validity with an exactly-zero
        factor, so an invalid token performs a bit-exactly zero update.
        """
        kt = tok_pre["kt"][:, t]                       # [B,K,D]
        vt = tok_pre["vt"][:, t]                       # [B,K,D]
        B, K, D = kt.shape
        val = tok_pre["valid"][:, t].to(vt.dtype)      # [B,K]
        keys, vals = state["keys"], state["vals"]      # [B,N,D]
        N = vals.shape[1]
        age = (t_cur_t.unsqueeze(1) - state["t_last"]).clamp_min(0.0)      # [B,N]
        slot_valid = state["valid"]
        zeros = torch.zeros_like(age)
        fresh = -age / self.age.tau().reshape(()) if self.decay else zeros
        log_prior = torch.where(slot_valid > 0.5, fresh, self.empty_bias.expand_as(age))

        if self.route_mode == "content":
            kn = unit_norm(keys, self.eps)                                 # [B,N,D]
            scores = self.score_scale * torch.einsum("bkd,bnd->bkn", kt, kn) / math.sqrt(D)
            scores = scores + log_prior.unsqueeze(1)                       # [B,K,N]
            if self.window_s > 0:                                          # hard time window
                in_win = ((age <= self.window_s) & (slot_valid > 0.5)).unsqueeze(1)
                scores = torch.where(in_win, scores, torch.full_like(scores, -1e9))
                empty = (slot_valid <= 0.5).unsqueeze(1)
                scores = torch.where(empty, log_prior.unsqueeze(1), scores)
            w = torch.softmax(scores, dim=-1)
            if self.hard_route:
                onehot = F.one_hot(w.argmax(dim=-1), N).to(w.dtype)
                w = w + (onehot - w).detach()
        else:                                                              # fixed j -> j%N
            idx = torch.arange(K, device=kt.device) % N
            w = F.one_hot(idx, N).to(kt.dtype).unsqueeze(0).expand(B, K, N).contiguous()

        rj = torch.einsum("bkn,bnd->bkd", w, vals)                         # [B,K,D]
        sj = (vt - rj).norm(dim=-1)                                        # [B,K]
        base = (pre["beta_z"][:, t] + self.tok_beta_c(h)).unsqueeze(1)     # [B,1,1]
        beta = torch.sigmoid(base + self.tok_beta_r(rj)
                             + self.tok_beta_s * torch.log1p(sj).unsqueeze(-1)
                             + self.tok_beta_b)                            # [B,K,1]
        if self.beta_exp:
            beta = 1.0 - torch.exp(-beta)
        if self.hard_write:                                                # per (token,slot)
            d2 = (vt.pow(2).sum(-1).unsqueeze(-1) + vals.pow(2).sum(-1).unsqueeze(1)
                  - 2.0 * torch.einsum("bkd,bnd->bkn", vt, vals)).clamp_min(0.0)
            u = hard_mask_st(d2.sqrt(), self.write_threshold,
                             temp=max(0.5 * self.write_threshold, 0.05))   # [B,K,N]
        else:
            u = torch.ones(B, K, N, device=vals.device, dtype=vals.dtype)
        mass = w * beta * u * val.unsqueeze(-1)        # [B,K,N]; exactly 0 when invalid
        tot = mass.sum(dim=1)                          # [B,N] per-slot write mass
        vbar = torch.einsum("bkn,bkd->bnd", mass, vt)
        vals_new = vals * (1.0 - tot).unsqueeze(-1) + vbar if self.write_rule == "delta" \
            else vals + vbar
        kbar = torch.einsum("bkn,bkd->bnd", mass, kt)
        keys_new = keys + torch.sigmoid(self.key_gate) * (kbar - keys * tot.unsqueeze(-1))
        t_last_new = state["t_last"] + tot * (t_cur_t.unsqueeze(1) - state["t_last"])
        valid_new = (slot_valid + tot * (1.0 - slot_valid)).clamp(0.0, 1.0)
        m3 = mask_t.unsqueeze(-1).unsqueeze(-1)
        vals_new = torch.where(m3, vals_new, vals)
        keys_new = torch.where(m3, keys_new, keys)
        t_last_new = apply_mask(t_last_new, state["t_last"], mask_t)
        valid_new = apply_mask(valid_new, slot_valid, mask_t)
        state_new = {"keys": keys_new, "vals": vals_new, "t_last": t_last_new,
                     "valid": valid_new, "vals_prev": vals}
        o_j = torch.einsum("bkn,bnd->bkd", w, vals_new)                    # [B,K,D]
        pooled = self._pool(o_j, tok_pre["valid"][:, t])
        aux = {
            "read": pooled[0],
            "write_gate": tot,                          # [B,N] aggregated over tokens
            "write_strength": beta.squeeze(-1).mean(dim=-1),
            "route": w,                                 # [B,K,N]
            "slot_read": o_j,                           # [B,K,D]
            "tok_write_gate": mass.sum(dim=-1),         # [B,K]
            "write_mask": u,
            "surprise": (sj * val).sum(-1) / val.sum(-1).clamp_min(1.0),
            "mem_norm": vals_new.flatten(1).norm(dim=-1),
            "slot_age": age,
            "tok_pool_mean": pooled[0],
        }
        if len(pooled) > 1:
            aux["tok_pool_max"] = pooled[1]
        return pooled[0], state_new, aux


class OutMLP(nn.Module):
    """``out = LN(MLP([h?, o?, z, tok_pool?]))`` -- the identical output head."""

    def __init__(self, hidden: int, mem_dim: int, use_slow: bool, use_mem: bool,
                 extra: int = 0):
        super().__init__()
        parts = ((hidden if use_slow else 0) + (mem_dim if use_mem else 0)
                 + hidden + int(extra))
        self.parts = parts
        self.mlp = nn.Sequential(nn.Linear(parts, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, parts: List[torch.Tensor]) -> torch.Tensor:
        return self.norm(self.mlp(torch.cat(parts, dim=-1)))


# --------------------------------------------------------------------------- #
# DSM
# --------------------------------------------------------------------------- #
class DSM(LoopCore):
    """Dual-state memory: slow state + matrix fast memory."""

    name = "dsm"
    _spec = DSM_SPEC

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.dt_scale = float(v["dt_scale"])
        self.dt_clamp_max = float(v["dt_clamp_max"])
        self.H = int(hidden)
        self.out_dim = int(hidden)
        self.mem_dim = int(v["mem_dim"])
        self.slow_on = not bool(v["slow_off"])
        self.mem_on = not bool(v["mem_off"])
        self.h_by_index = bool(v["h_by_index"])
        d = self.inp.d

        # nonlinear input embedding z (shared by every sub-module)
        self.z = nn.Linear(d, self.H)
        self.z_norm = nn.LayerNorm(self.H)
        self.slow = SlowState(self.H, v["tau_h_min"], v["tau_h_max"]) if self.slow_on else None
        self.mem = MatrixMemory(
            self.H, self.mem_dim, n_decay_groups=int(v["n_decay_groups"]),
            write_rule=str(v["write_rule"]), read_mode=str(v["read_mode"]),
            decay=bool(v["decay"]), beta_exp=bool(v["beta_exp"]),
            hard_write=bool(v["hard_write"]),
            write_threshold=float(v["write_threshold"]),
            hop_scale=float(v["hop_scale"]),
            use_tok=bool(v["use_tok"]), tok_dim=int(v["tok_dim"]),
            tok_vis_channel=v["tok_vis_channel"],
            read_pool=str(v["tok_read_pool"])) if self.mem_on else None
        self.use_tok = bool(v["use_tok"]) and self.mem_on
        self.tok_pool_n = 2 if str(v["tok_read_pool"]) == "mean_max" else 1
        self.out = OutMLP(self.H, self.mem_dim, self.slow_on, self.mem_on,
                          extra=(self.mem.extra_out if self.use_tok else 0))
        self.drop = nn.Dropout(dropout)

        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": self.H,
            "inp_dim": int(d), "mem_dim_eff": self.mem_dim if self.mem_on else 0,
            "decay_rates_init": list(self.mem.dec.rates) if self.mem_on else [],
        }
        self.cfg_unused = sorted(unused)

    # ------------------------------------------------------------------ #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        st: Dict[str, torch.Tensor] = {}
        if self.slow_on:
            st["h"] = torch.zeros(B, self.H, device=device)
        if self.mem_on:
            st.update(self.mem.init_state(B, device))
        return st

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None):
        xn = self.inp(x, dt, mask)                            # real dt -> feature channel
        z = self.z_norm(F.gelu(self.z(xn)))                   # [B,T,H]
        ctx = {"z": z, "mask": mask, "dts": dts,
               "logdt": torch.log1p(dts).unsqueeze(-1),       # [B,T,1]
               "t_cur": torch.cumsum(dts, dim=1)}             # physical time, [B,T]
        if self.slow_on:
            ctx["slow"] = self.slow.prep(z)
        else:                                                 # zero stand-in for [z,h] terms
            ctx["h0"] = torch.zeros(z.shape[0], self.H, device=z.device, dtype=z.dtype)
        if self.mem_on:
            ctx["mem"] = self.mem.prep(z)
            if self.use_tok and tok is not None:
                ctx["tok"] = self.mem.prep_tok(tok, tok_valid, mask)
        return ctx

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        m_t = ctx["mask"][:, t]
        h = state["h"] if "h" in state else ctx["h0"]
        g = None
        if self.slow_on:
            h, g, _ = self.slow.step(t, ctx["slow"], h, ctx["dts"][:, t], m_t,
                                     by_index=self.h_by_index)
            state["h"] = h
        o, aux = None, {}
        pooled = []
        if self.mem_on:
            o, M, aux = self.mem.step(t, ctx["mem"], h, state["M"],
                                      ctx["dts"][:, t], ctx["logdt"][:, t], m_t,
                                      tok_pre=ctx.get("tok"))
            state["M"] = M
            if "tok_pool_mean" in aux:
                pooled.append(aux.pop("tok_pool_mean"))
                if "tok_pool_max" in aux:
                    pooled.append(aux.pop("tok_pool_max"))
        parts = []
        if self.slow_on:
            parts.append(h)
        if self.mem_on:
            parts.append(o)
        parts.append(ctx["z"][:, t])
        if self.use_tok and not pooled:
            # tokens absent for this call: degrade gracefully to the tokenless
            # computation by feeding zeros (never crash on a missing tok tensor)
            z0 = ctx["z"].new_zeros(ctx["z"].shape[0], self.mem_dim)
            pooled = [z0] * self.tok_pool_n
        parts.extend(pooled)
        out = self.drop(self.out(parts))
        if self.slow_on:
            aux["hidden"] = h
            if g is not None:
                aux["slow_gate"] = g
        return out, state, aux

    def _final_aux(self, state) -> Dict[str, Any]:
        stats: Dict[str, float] = {}
        if self.slow_on:
            stats["tau_h_mean"] = float(self.slow.dec.tau().detach().mean())
        if self.mem_on:
            stats.update(self.mem.stats())
        return {"decay_stats": stats}


# --------------------------------------------------------------------------- #
# DSM with slots
# --------------------------------------------------------------------------- #
class DSMSlot(LoopCore):
    """Dual-state memory with content-addressed slots instead of a matrix memory."""

    name = "dsm_slot"
    _spec = SLOT_SPEC

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.dt_scale = float(v["dt_scale"])
        self.dt_clamp_max = float(v["dt_clamp_max"])
        self.H = int(hidden)
        self.out_dim = int(hidden)
        self.slow_on = not bool(v["slow_off"])
        self.h_by_index = bool(v["h_by_index"])
        self.slot_dim = int(v["slot_dim"] or v["mem_dim"])
        d = self.inp.d

        self.z = nn.Linear(d, self.H)
        self.z_norm = nn.LayerNorm(self.H)
        self.slow = SlowState(self.H, v["tau_h_min"], v["tau_h_max"]) if self.slow_on else None
        self.mem = SlotMemory(
            self.H, n_slots=int(v["n_slots"]), slot_dim=self.slot_dim,
            route_mode=str(v["route_mode"]),
            slot_hard_route=bool(v["slot_hard_route"]),
            hard_write=bool(v["hard_write"]),
            write_threshold=float(v["write_threshold"]),
            beta_exp=bool(v["beta_exp"]), tau_age_init=float(v["tau_age_init"]),
            write_rule=str(v["write_rule"]), decay=bool(v["decay"]),
            window_s=float(v["window_s"]),
            use_tok=bool(v["use_tok"]), tok_dim=int(v["tok_dim"]),
            tok_vis_channel=v["tok_vis_channel"], read_pool=str(v["tok_read_pool"]))
        self.use_tok = bool(v["use_tok"])
        self.tok_pool_n = 2 if str(v["tok_read_pool"]) == "mean_max" else 1
        self.out = OutMLP(self.H, self.slot_dim, self.slow_on, True,
                          extra=(self.mem.extra_out if self.use_tok else 0))
        self.drop = nn.Dropout(dropout)

        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": self.H,
            "inp_dim": int(d), "slot_dim_eff": self.slot_dim,
        }
        self.cfg_unused = sorted(unused)

    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        st = {}
        if self.slow_on:
            st["h"] = torch.zeros(B, self.H, device=device)
        st.update(self.mem.init_state(B, device))
        return st

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None):
        xn = self.inp(x, dt, mask)
        z = self.z_norm(F.gelu(self.z(xn)))
        ctx = {"z": z, "mask": mask, "dts": dts,
               "logdt": torch.log1p(dts).unsqueeze(-1),
               "mem": self.mem.prep(z)}
        if self.use_tok and tok is not None:
            ctx["tok"] = self.mem.prep_tok(tok, tok_valid, mask)
        if self.slow_on:
            ctx["slow"] = self.slow.prep(z)
        else:
            ctx["h0"] = torch.zeros(z.shape[0], self.H, device=z.device, dtype=z.dtype)
        return ctx

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        m_t = ctx["mask"][:, t]
        h = state["h"] if "h" in state else ctx["h0"]
        g = None
        if self.slow_on:
            h, g, _ = self.slow.step(t, ctx["slow"], h, ctx["dts"][:, t], m_t,
                                     by_index=self.h_by_index)
            state["h"] = h
        # cumulative physical clock lives in the state (dts is 0 at padded steps)
        t_cur_t = state["t_cur"].squeeze(-1) + ctx["dts"][:, t]
        state["t_cur"] = t_cur_t.unsqueeze(-1)
        if ctx.get("tok") is not None:                 # object-addressed mode
            o, new_st, aux = self.mem.step_tok(t, ctx["mem"], h, state,
                                               ctx["dts"][:, t], t_cur_t,
                                               ctx["logdt"][:, t], m_t, ctx["tok"])
        else:                                          # frame-addressed mode (default)
            o, new_st, aux = self.mem.step(t, ctx["mem"], h, state, ctx["dts"][:, t],
                                           t_cur_t, ctx["logdt"][:, t], m_t)
        state.update(new_st)
        pooled = []
        if "tok_pool_mean" in aux:
            pooled.append(aux.pop("tok_pool_mean"))
            if "tok_pool_max" in aux:
                pooled.append(aux.pop("tok_pool_max"))
        if self.use_tok and not pooled:                # tokens absent: graceful degradation
            z0 = ctx["z"].new_zeros(ctx["z"].shape[0], self.slot_dim)
            pooled = [z0] * self.tok_pool_n
        parts = ([h] if self.slow_on else []) + [o, ctx["z"][:, t]] + pooled
        out = self.drop(self.out(parts))
        if g is not None:
            aux["slow_gate"] = g
        if self.slow_on:
            aux["hidden"] = h
        return out, state, aux

    def _final_aux(self, state) -> Dict[str, Any]:
        stats: Dict[str, float] = {}
        if self.slow_on:
            stats["tau_h_mean"] = float(self.slow.dec.tau().detach().mean())
        stats["tau_age"] = float(self.mem.age.tau().detach().reshape(()))
        return {"decay_stats": stats,
                "slot_usage": state["valid"].detach().mean(0),           # [N]
                "slot_state_after": state["vals"].detach(),
                "slot_state_before": state.get("vals_prev", state["vals"]).detach(),
                "slot_t_last": state["t_last"].detach()}


class WindowDSM(DSMSlot):
    """``dsm_slot`` with a hard per-object route *and* a hard time window.

    "Only the object currently inside the window is updated": routing is top-1
    (straight-through) and slots older than ``window_s`` seconds are removed from
    the candidate set, so an object that leaves the window cannot be written and
    its slot is not disturbed by other objects.  This is the ablation for the
    claim that *hard* per-object isolation (rather than a soft attention) is what
    prevents cross-talk.
    """

    name = "window"

    def __init__(self, in_dim: int, hidden: int, **cfg):
        cfg = dict(cfg)
        cfg.setdefault("slot_hard_route", True)
        cfg.setdefault("route_mode", "content")
        cfg.setdefault("window_s", 2.0)
        super().__init__(in_dim, hidden, **cfg)
        self.cfg_effective["model"] = self.name
