"""Mamba -- S6-style selective state-space model in pure PyTorch (no mamba_ssm/triton).

Equation (one SSM layer, diagonal ``A``, per-step scalar timescale)::

    dt_ssm_t = softplus( dt_mlp(x_t) [+ tdt_scale * log1p(dt_t)] )   # input-selective
    a_t      = exp(dt_ssm_t * A)                     # A < 0 diagonal, so a_t in (0,1]
    b_t      = dt_ssm_t * B(x_t) * u_t               # outer product over the state dim
    h_t      = a_t * h_{t-1} + b_t                   # linear recurrence
    y_t      = <C(x_t), h_t> + D * u_t

``B``, ``C``, ``dt`` and ``u`` all come from linear projections of the input, which
is the whole "selective" idea: the dynamics are a function of the *content*.

Why a scan is legitimate here
-----------------------------
``A`` is diagonal, so the recurrence decouples into ``d_state`` independent
first-order linear recurrences, each of which has the closed form

    h_t = sum_{j<=t} exp(LA_t - LA_j) * b_j ,      LA = cumsum(log a)

That is a prefix sum with weights, and the weights are ``<= 1`` (decay), so it can
be evaluated **stably** without any log-sum-exp trick: sequences are cut into
chunks of ``chunk`` steps, inside a chunk the (t, j) weight matrix is built
explicitly and multiplied as one batched einsum, and the carry between chunks is
``h_{end} * exp(LA_t - LA_{end})``.  Nothing ever overflows: every exponential
argument is clamped to ``<= 0``.

``_scan_naive`` is the literal per-step loop; ``tests_models.test_mamba_scan``
asserts the two agree to 1e-5 (and ``Mamba(scan="naive")`` is available for the
slow reference run).

Masking: at a padded step ``dt_ssm_t`` is forced to 0, so ``a_t = 1`` and
``b_t = 0`` -- the step is exactly the identity, which is the required
"``mask=False`` must not change the state" semantics.

Time awareness (ablation ``tdt``)
---------------------------------
By default the internal timestep is a pure function of the *content*
(``mamba``).  With ``tdt=True`` the real physical ``log1p(dt)`` is added to the
``softplus`` logit (``mamba+time``): the model can then slow down / speed up its
integration when the sampling interval changes.
"""
from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.contract import TemporalCore, count_params
from .base import (COMMON_SPEC, InputNorm, dt_struct, inv_softplus, safe_dt,
                   scale_dt, unpack_cfg)


class Mamba(TemporalCore):
    """Selective SSM layer.  ``scan`` in {"fast", "naive"}."""

    name = "mamba"
    _spec = {
        **COMMON_SPEC,
        "d_state": 16,        # SSM state width (diagonal A, one rate per state dim)
        "d_conv": 4,          # depthwise causal conv on the input stream (Mamba default)
        "dt_rank": None,      # bottleneck width of the dt MLP (None -> max(8, hidden//16))
        "tdt": False,         # True -> add the REAL log1p(dt) to the dt_ssm logit
        "chunk": 64,          # chunk length of the fast scan
        "scan": "fast",       # "fast" | "naive" (identical results, unit-tested)
        "dt_init": 0.03,      # initial softplus(dt) in seconds
    }

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        d_state, d_conv = int(v["d_state"]), int(v["d_conv"])
        dt_rank, tdt = v["dt_rank"], bool(v["tdt"])
        chunk, scan, dt_init = int(v["chunk"]), str(v["scan"]), float(v["dt_init"])
        if scan not in ("fast", "naive"):
            raise ValueError(f"scan must be 'fast' or 'naive', got {scan!r}")
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.dt_scale = float(v["dt_scale"])
        self.H = int(hidden)
        self.S = d_state
        self.out_dim = int(hidden)
        self.chunk = chunk
        self.scan = scan
        self.tdt = tdt
        self.d_conv = d_conv
        d = self.inp.d

        self.x_proj = nn.Linear(d, self.H)                 # u_t (value stream)
        self.b_proj = nn.Linear(d, self.S)                 # B_t
        self.c_proj = nn.Linear(d, self.S)                 # C_t
        dt_rank = int(dt_rank or max(8, self.H // 16))
        self.dt_mlp = nn.Sequential(
            nn.Linear(d, dt_rank), nn.GELU(), nn.Linear(dt_rank, 1)
        )
        with torch.no_grad():                              # start at dt_ssm ~ dt_init
            self.dt_mlp[-1].weight.mul_(0.1)
            self.dt_mlp[-1].bias.fill_(inv_softplus(dt_init))
        self.dt_bias = nn.Parameter(torch.zeros(1))
        # A = -exp(A_log) with A_log = log(1..S): decay rates 1/s ... S/s
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.S + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.tdt_scale = nn.Parameter(torch.tensor(1.0 if tdt else 0.0))
        self.drop = nn.Dropout(dropout)
        # standard Mamba block also has a SiLU gate on the output and an out_proj,
        # which is what keeps its parameter budget comparable at a given width
        self.gate_proj = nn.Linear(d, self.H)
        self.out_proj = nn.Linear(self.H, self.H)

        if self.d_conv and self.d_conv > 1:                # depthwise causal conv
            self.conv_w = nn.Parameter(torch.empty(self.H, 1, self.d_conv))
            self.conv_b = nn.Parameter(torch.zeros(self.H))
            nn.init.kaiming_uniform_(self.conv_w, a=math.sqrt(5))
        else:
            self.conv_w = None
            self.conv_b = None

        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": self.H,
            "inp_dim": int(d), "d_state_eff": self.S,
            "dt_init_eff": dt_init,
            "variant": "mamba+tdt" if tdt else "mamba",
            "mamba_block": "gated+out_proj (standard Mamba block)",
        }
        self.cfg_unused = sorted(unused)

    # ------------------------------------------------------------------ #
    # projections (shared by forward / step)
    # ------------------------------------------------------------------ #
    def _prep(self, x, dt, dts, mask, use_conv: bool = True) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        xn = self.inp(x, dt, mask)                         # real dt -> feature channel
        u = self.x_proj(xn)                                # [B,T,H]
        if use_conv and self.conv_w is not None:            # causal depthwise conv
            uu = F.pad(u.transpose(1, 2), (self.d_conv - 1, 0))
            u = F.conv1d(uu, self.conv_w, self.conv_b, groups=self.H).transpose(1, 2)
        Bt = self.b_proj(xn)                               # [B,T,S]
        Ct = self.c_proj(xn)                               # [B,T,S]
        logit = self.dt_mlp(xn).squeeze(-1)                # [B,T]
        if self.tdt:                                       # real physical time enters
            logit = logit + self.tdt_scale * torch.log1p(dts)
        # mask -> dt_ssm = 0 -> a = 1, b = 0 -> exact identity step
        dt_ssm = F.softplus(logit + self.dt_bias) * mask.to(logit.dtype)   # [B,T]
        A = -torch.exp(self.A_log)                         # [S]  (< 0)
        log_a = dt_ssm.unsqueeze(-1) * A                   # [B,T,S]  (<= 0)
        upd = dt_ssm[:, :, None, None] * Bt[:, :, None, :] * u[:, :, :, None]
        return {"u": u, "Bt": Bt, "Ct": Ct, "log_a": log_a, "upd": upd,
                "dt_ssm": dt_ssm, "gate": self.gate_proj(xn)}

    def _out_from_h(self, h, u, Ct, gate):
        """``y = <C, h> + D*u``, SiLU-gated and projected (standard Mamba block)."""
        y = (h * Ct.unsqueeze(2)).sum(-1) + self.D * u
        return self.out_proj(y * F.silu(gate))

    # ------------------------------------------------------------------ #
    # the scan
    # ------------------------------------------------------------------ #
    def _scan_naive(self, log_a, upd):
        """Literal per-step loop -- the reference the fast path is checked against."""
        B, T, S = log_a.shape
        H = upd.shape[2]
        h = upd.new_zeros((B, H, S))
        out = []
        for t in range(T):
            a = torch.exp(log_a[:, t])                     # [B,S]
            h = a.unsqueeze(1) * h + upd[:, t]             # [B,H,S]
            out.append(h)
        return torch.stack(out, dim=1)

    def _scan_fast(self, log_a, upd):
        """Chunked, fully vectorised-in-T prefix scan (see module docstring)."""
        B, T, S = log_a.shape
        H = upd.shape[2]
        C = max(1, min(self.chunk, T))
        LA = torch.cumsum(log_a, dim=1)                    # [B,T,S], non-increasing
        idx = torch.arange(C, device=log_a.device)
        causal = (idx.view(C, 1) >= idx.view(1, C)).to(log_a.dtype)   # [C,C]
        y = upd.new_zeros((B, H, S))
        outs = []
        for s in range(0, T, C):
            e = min(T, s + C)
            c = e - s
            la = LA[:, s:e]                                # [B,c,S]
            uu = upd[:, s:e]                               # [B,c,H,S]
            # W[i,j] = exp(LA_i - LA_j) for j <= i   (all arguments <= 0)
            diff = (la.unsqueeze(2) - la.unsqueeze(1)).clamp_max(0.0)   # [B,c,c,S]
            W = torch.exp(diff) * causal[:c, :c].view(1, c, c, 1)
            yc = torch.einsum("bijs,bjhs->bihs", W, uu)    # intra-chunk contributions
            if s > 0:                                      # carry from earlier chunks
                carry = torch.exp((la - LA[:, s - 1:s]).clamp_max(0.0))   # [B,c,S]
                yc = yc + y.unsqueeze(1) * carry.unsqueeze(2)             # [B,c,H,S]
            y = yc[:, -1]
            outs.append(yc)
        return torch.cat(outs, dim=1)

    def _scan(self, log_a, upd):
        return self._scan_fast(log_a, upd) if self.scan == "fast" else self._scan_naive(log_a, upd)

    # ------------------------------------------------------------------ #
    def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None, roi_valid=None,
                roi_meta=None):
        B, T, _ = x.shape
        dt = scale_dt(safe_dt(dt, mask), self.dt_scale)
        if T == 0:
            return x.new_zeros(B, 0, self.out_dim), {}
        dts = dt_struct(dt, mask, self.use_dt_struct, self.dt_const)
        p = self._prep(x, dt, dts, mask)
        h = self._scan(p["log_a"], p["upd"])               # [B,T,H,S]
        out = self._out_from_h(h, p["u"], p["Ct"], p["gate"])
        snorm = h.flatten(2).norm(dim=-1)                  # [B,T]
        aux = {"dt_ssm": p["dt_ssm"], "ssm_state_norm": snorm,
               "hidden": h.mean(-1)}                       # coarse readout of h
        return out, aux

    # ------------------------------------------------------------------ #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        st = {"h": torch.zeros(B, self.H, self.S, device=device)}
        if self.conv_w is not None:
            # history buffer so that step() reproduces the causal conv exactly
            st["conv_buf"] = torch.zeros(B, self.d_conv - 1, self.H, device=device)
        return st

    def step(self, x_t, dt_t, mask_t=None, state=None, tok_t=None, roi_t=None,
             roi_valid_t=None, roi_meta_t=None):
        """One frame: the naive recurrence evaluated once, plus the conv history.

        With ``d_conv > 1`` the streaming path needs the last ``d_conv-1`` input
        projections; they live in ``state["conv_buf"]``, so this is bit-for-bit
        the same operator as ``forward`` (unit-tested).
        """
        B = x_t.shape[0]
        if state is None:
            state = self.init_state(B, x_t.device)
        if mask_t is None:
            mask_t = torch.ones(B, dtype=torch.bool, device=x_t.device)
        mask_t = mask_t.to(torch.bool).reshape(B)
        dt2 = scale_dt(safe_dt(dt_t.reshape(B, 1), mask_t.unsqueeze(1)), self.dt_scale)
        dts2 = dt_struct(dt2, mask_t.unsqueeze(1), self.use_dt_struct, self.dt_const)
        p = self._prep(x_t.unsqueeze(1), dt2, dts2, mask_t.unsqueeze(1),
                       use_conv=False)
        u_raw = p["u"]                                     # [B,1,H] pre-conv
        if self.conv_w is not None:
            seq = torch.cat([state["conv_buf"], u_raw], dim=1)     # [B,d_conv,H]
            u = F.conv1d(seq.transpose(1, 2), self.conv_w, self.conv_b,
                         groups=self.H).transpose(1, 2)            # [B,1,H]
            conv_buf = seq[:, 1:, :]
        else:
            u = u_raw
            conv_buf = state.get("conv_buf")
        # recompute the additive update with the *conv* output (p["upd"] used the
        # pre-conv u, which is only correct for the batched path)
        upd = p["dt_ssm"][:, :, None, None] * p["Bt"][:, :, None, :] * u[:, :, :, None]
        a = torch.exp(p["log_a"][:, 0])                    # [B,S]
        h = a.unsqueeze(1) * state["h"] + upd[:, 0]        # [B,H,S]
        y = self._out_from_h(h.unsqueeze(1), u, p["Ct"], p["gate"])[:, 0]
        aux = {"dt_ssm": p["dt_ssm"], "ssm_state_norm": h.flatten(1).norm(dim=-1,
                                                                        keepdim=True),
               "hidden": h.mean(-1).unsqueeze(1)}          # [B,1], [B,1,H] for parity
        new_state = {"h": h}
        if conv_buf is not None:
            new_state["conv_buf"] = conv_buf
        return y, aux, new_state

    @property
    def n_params(self) -> int:
        return count_params(self)
