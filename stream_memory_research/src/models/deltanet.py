"""DeltaNet -- fast-weight matrix memory only, no slow state (strong baseline).

State is a single matrix ``M in R^{d_v x d_k}`` (``d_k = d_v = mem_dim``) written
with the **delta / Householder rule**, which is the residual write::

    M^-   = M * exp(-Lambda * dt)                 # multi-timescale, grouped
    r     = read(M^-, q)                          # content-addressed read
    k     = W_k z,  unit-normalised
    beta  = sigmoid(w.[z, r, log1p(dt)] + w_s.log1p(||v - r||))
    M     = M^- - beta (M^- k) k^T + beta v k^T   # == (I - beta k k^T) M^- + beta v k^T
    o     = read(M, q)

The Householder form is *the* memory mechanism: it removes the old value of the
addressed key before writing the new one, so repeated writes of the same key
overwrite instead of accumulating interference.  The ablation in the Gated
DeltaNet paper ("Decay hurts memory retention", DeltaNet 98.8 vs Mamba2 with
decay-but-no-residual 30.4 at 8K) is precisely this distinction, so
``write_rule="outer"`` (plain additive ``M + beta v k^T``) exists **only** to
attribute capacity in the ablation -- it is never the default.

``read_mode``:
  ``"linear"``    r = M q            -- capacity O(d_k) associations;
  ``"bilinear"``  r = M q + hop * M (M^T M q)/||.||  -- a second, quadratic hop.
                  The write is unchanged; only the readout is second order, which
                  sharpens retrieval when keys are nearly parallel (this is the
                  "read is part of the capacity" control).  Every capacity number
                  must state which mode produced it.

No slow state: ``out = Linear([o, x_n])``.  This makes the model a clean test of
"can a pure fast-weight memory solve the task without a persistent summary".
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.contract import apply_mask
from .base import (COMMON_SPEC, GroupedDecay, InputNorm, LoopCore, hard_st,
                   unpack_cfg)


def unit_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise the last dimension (the delta rule needs ||k|| = 1)."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MemRead(nn.Module):
    """Read operator for a matrix memory ``M in R^{d_v x d_k}`` (shared by
    ``deltanet`` and ``dsm``), so the two models differ in their state, not in
    their read.

    ``mode="linear"``   ``r = M q``.  The memory exposes a ``d_k``-dimensional
    slice of itself, so it can hold on the order of ``d_k`` separable
    associations.  This is the standard DeltaNet read.

    ``mode="bilinear"`` ``o = diag(U M V)``, i.e. each of the ``d_out = d_v``
    output dimensions has *its own* learned query/key pair
    (``u_j in R^{d_k}``, ``v_j in R^{d_v}``)::

        o_j = u_j^T M v_j

    There are ``d_k * d_v`` independent bilinear functionals of ``M``, so this
    read can in principle separate ~``d_v d_k`` associations instead of ``d_k``
    -- the published capacity result this A/B exists to test.  Note it is
    **query-independent by construction** (the query is learned, not
    observation-dependent); the write rule is identical in both modes, so a
    linear-vs-bilinear comparison isolates read capacity only.

    ``mode="two_hop"``  ``r = M q + hop * M unit(M^T M q)`` -- a third,
    genuinely different operator (quadratic in ``M``, query-dependent); kept
    because it may help retrieval, but it must never be reported as "bilinear".
    """

    MODES = ("linear", "bilinear", "two_hop")

    def __init__(self, dk: int, dv: int, mode: str = "linear",
                 hop_scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"read_mode must be one of {self.MODES}, got {mode!r}")
        self.mode, self.eps = mode, float(eps)
        self.dk, self.dv = int(dk), int(dv)
        self.U = self.V = self.hop = None
        if mode == "bilinear":                      # d_out = d_v
            self.U = nn.Parameter(torch.empty(self.dv, self.dk))   # [d_out, d_k]
            self.V = nn.Parameter(torch.empty(self.dv, self.dv))   # [d_v, d_out]
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        elif mode == "two_hop":
            self.hop = nn.Parameter(torch.tensor(float(hop_scale)))

    def forward(self, M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return torch.einsum("bvd,bd->bv", M, q)
        if self.mode == "two_hop":
            r = torch.einsum("bvd,bd->bv", M, q)
            kq = unit_norm(torch.einsum("bvd,bv->bd", M, r), self.eps)
            return r + self.hop * torch.einsum("bvd,bd->bv", M, kq)
        # bilinear: P[b,v,o] = sum_d M[b,v,d] U[o,d] ; o[b,j] = sum_v P[b,v,j] V[v,j]
        P = torch.einsum("bvd,od->bvo", M, self.U)
        return (P * self.V.unsqueeze(0)).sum(dim=1)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, dk={self.dk}, dv={self.dv}"


class DeltaNet(LoopCore):
    """Fast-weight only: matrix memory + delta rule."""

    name = "deltanet"
    _spec = {
        **COMMON_SPEC,
        "mem_dim": 64,             # d_k = d_v
        "decay": True,             # False -> Lambda == 0
        "n_decay_groups": 4,       # multi-timescale groups
        "write_rule": "delta",     # "delta" (Householder residual) | "outer" (additive)
        "read_mode": "linear",     # "linear" | "bilinear" (capacity A/B) | "two_hop"
        "beta_exp": False,         # stability switch only: beta -> 1 - exp(-beta)
        "dt_clamp_max": 10.0,      # seconds; clamp dt BEFORE any exp
        "hard_write": False,       # straight-through binary write mask
        "write_threshold": 0.5,    # delta for hard_write
        "hop_scale": 1.0,          # bilinear read: initial weight of the 2nd hop
    }

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        if v["write_rule"] not in ("delta", "outer"):
            raise ValueError(f"{self.name}: bad write_rule={v['write_rule']!r}")
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.dt_scale = float(v["dt_scale"])
        self.dt_clamp_max = float(v["dt_clamp_max"])
        self.H = int(hidden)
        self.out_dim = int(hidden)
        self.dk = self.dv = int(v["mem_dim"])
        self.write_rule = str(v["write_rule"])
        self.read_mode = str(v["read_mode"])
        self.beta_exp = bool(v["beta_exp"])
        self.hard_write = bool(v["hard_write"])
        self.write_threshold = float(v["write_threshold"])
        self.eps = 1e-6
        d = self.inp.d

        # --- input bottleneck, identical in role to dsm's z ------------------- #
        # Every core first projects the (wide) feature vector into the model width
        # and then does its temporal thing.  DeltaNet must do the same: projecting
        # q/k/v straight from a 770-dim input would (a) give it a parameter floor
        # of ~3*d*mem_dim that no other core has and (b) make dsm-vs-deltanet a
        # comparison of input projections rather than of memory mechanisms.
        self.z = nn.Linear(d, self.H)
        self.z_norm = nn.LayerNorm(self.H)
        # --- memory projections (all state-independent -> hoisted) ----------- #
        self.q_proj = nn.Linear(self.H, self.dk)
        self.k_proj = nn.Linear(self.H, self.dk)
        self.v_proj = nn.Linear(self.H, self.dv)
        # write-strength logit: content term + read term + dt term + surprise term
        self.beta_x = nn.Linear(self.H, 1)
        self.beta_r = nn.Linear(self.dv, 1, bias=False)
        self.beta_t = nn.Parameter(torch.zeros(1))
        self.beta_s = nn.Parameter(torch.zeros(1))
        self.beta_b = nn.Parameter(torch.zeros(1))
        # --- time --- #
        self.dec = GroupedDecay(v["n_decay_groups"], self.dv,
                               learnable=bool(v["decay"]), decay=bool(v["decay"]))
        # --- read (shared operator; bilinear = capacity-theory A/B) --- #
        self.readmod = MemRead(self.dk, self.dv, mode=self.read_mode,
                               hop_scale=float(v["hop_scale"]), eps=self.eps)
        # --- output: [readout, latent] -> hidden (no slow state) --- #
        self.out = nn.Linear(self.dv + self.H, self.H)
        self.drop = nn.Dropout(dropout)

        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": self.H,
            "inp_dim": int(d), "dk": self.dk, "dv": self.dv,
            "decay_rates_init": list(self.dec.rates),
        }
        self.cfg_unused = sorted(unused)

    # ------------------------------------------------------------------ #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {"M": torch.zeros(B, self.dv, self.dk, device=device)}

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None):
        xn = self.inp(x, dt, mask)                        # real dt in the feature channel
        z = self.z_norm(F.gelu(self.z(xn)))               # [B,T,H] latent bottleneck
        return {
            "z": z,
            "dts": dts,                                   # dt that drives the dynamics
            "q": self.q_proj(z),                          # [B,T,dk]
            "k": unit_norm(self.k_proj(z)),                # [B,T,dk] unit keys
            "v": self.v_proj(z),                          # [B,T,dv]
            "beta_x": self.beta_x(z),                     # [B,T,1]
            "logdt": torch.log1p(dts).unsqueeze(-1),      # [B,T,1]
            "mask": mask,
        }

    def _read(self, M: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.readmod(M, q)

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        m_t = ctx["mask"][:, t]
        M_prev = state["M"]
        q = ctx["q"][:, t]
        k = ctx["k"][:, t]
        v = ctx["v"][:, t]
        f = self.dec.step_factor(ctx["dts"][:, t])         # [B,dv] decay factor (<=1)
        M_dec = M_prev * f.unsqueeze(-1)                  # decay rows of M (<= 1)
        r = self._read(M_dec, q)
        e = v - r
        surprise = e.norm(dim=-1, keepdim=True)           # [B,1]
        beta = torch.sigmoid(
            ctx["beta_x"][:, t] + self.beta_r(r)
            + self.beta_t * ctx["logdt"][:, t]
            + self.beta_s * torch.log1p(surprise) + self.beta_b)
        if self.beta_exp:
            beta = 1.0 - torch.exp(-beta)                 # stability only
        if self.hard_write:
            beta = hard_st(beta, self.write_threshold)
        vk = v.unsqueeze(-1) * k.unsqueeze(1)             # [B,dv,dk]  v k^T
        if self.write_rule == "delta":                    # exact Householder delta
            Mk = torch.einsum("bvd,bd->bv", M_dec, k).unsqueeze(-1)   # [B,dv,1]
            M_new = M_dec - beta.unsqueeze(-1) * Mk * k.unsqueeze(1) + beta.unsqueeze(-1) * vk
        else:                                             # additive outer product
            M_new = M_dec + beta.unsqueeze(-1) * vk
        M = apply_mask(M_new, M_prev, m_t)
        o = self._read(M, q)
        out = self.drop(self.out(torch.cat([o, ctx["z"][:, t]], dim=-1)))
        aux = {
            "read": o,
            "write_gate": beta.squeeze(-1),
            "surprise": surprise.squeeze(-1),
            "mem_norm": M.flatten(1).norm(dim=-1),
        }
        return out, {"M": M}, aux

    def _final_aux(self, state) -> Dict[str, Any]:
        return {"decay_stats": self.dec.stats(), "lam_rate": self.dec.lam().detach()}
