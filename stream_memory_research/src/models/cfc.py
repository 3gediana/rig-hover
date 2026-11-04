"""CfC -- the ncps-semantics closed-form continuous-time RNN (main baseline).

The operator is copied element for element from ``D:/robot_work/pc/cfc_host/cfc_ref.py``
(itself verified against ``ncps.torch.CfCCell(mode="default",
backbone_activation="lecun_tanh")`` and against the deployed C engine)::

    x     = concat(inp, h)                       # (input_dim + hidden_dim)
    z     = lecun(bb_w @ x + bb_b)               # backbone, lecun_tanh
    ff1   = tanh(ff1_w @ z + ff1_b)
    ff2   = tanh(ff2_w @ z + ff2_b)
    ta    = ta_w @ z + ta_b
    tb    = tb_w @ z + tb_b
    t     = sigmoid(ta * ts + tb)                # ts = REAL dt in seconds
    h_new = ff1 * (1 - t) + t * ff2              # NOTE: no h self-loop term

Three details that must not be "fixed":

1. ``lecun_tanh(v) = 1.7159 * tanh(0.666 * v)``.  ``0.666`` is the truncated
   constant in the ncps source, not ``2/3``.
2. The timespan multiplies ``ta`` and only then is ``tb`` added (``ta*ts + tb``).
   That ordering is what makes the cell a *time-constant* interpolation: with
   ``ts = 0`` the cell still writes ``sigmoid(tb)`` instead of holding its state.
3. There is no ``h`` self-loop, so dt == 0 does **not** mean "state unchanged".
   Padding therefore requires the explicit ``apply_mask`` hold below.

Optional local extension (from the deployed firmware model, off by default):
``retention_gate`` adds ``h_out = h_new + sigmoid(ret)·(h − h_new)`` with ``ret``
initialised to −3, so it starts almost identical to plain CfC.

``dt == 0`` is safe: ``t = sigmoid(tb)`` is finite and nothing divides by dt.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.contract import apply_mask
from .base import COMMON_SPEC, InputNorm, LoopCore, unpack_cfg

LECUN_A = 1.7159
LECUN_B = 0.666          # truncated in the ncps source -- do not use 2/3


def lecun_tanh(v: torch.Tensor) -> torch.Tensor:
    """LeCun tanh, identical to ``cfc_ref.lecun``."""
    return LECUN_A * torch.tanh(LECUN_B * v)


class CfC(LoopCore):
    """Closed-form continuous-time cell (ncps ``CfCCell`` semantics)."""

    name = "cfc"
    state_as_out = True          # out_t IS the hidden state -> aux["hidden"] == out
    _spec = {**COMMON_SPEC, "backbone_units": 64, "retention_gate": False,
             "dt_scale": 1.0}

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.H = int(hidden)
        self.out_dim = int(hidden)
        self.backbone_units = int(v["backbone_units"])
        self.dt_scale = float(v["dt_scale"])
        d = self.inp.d

        # backbone (ncps backbone_layers=1); 0 -> no backbone, z is the raw concat
        # (ncps backbone_layers=0 / AutoNCP shape)
        if self.backbone_units > 0:
            self.bb_w = nn.Parameter(torch.empty(self.backbone_units, d + self.H))
            self.bb_b = nn.Parameter(torch.zeros(self.backbone_units))
            nn.init.xavier_uniform_(self.bb_w)
            z_units = self.backbone_units
        else:
            self.bb_w = None
            self.bb_b = None
            z_units = d + self.H

        def _lin(out_dim: int) -> nn.Linear:
            lin = nn.Linear(z_units, out_dim)
            nn.init.xavier_uniform_(lin.weight)
            return lin

        self.ff1 = _lin(self.H)
        self.ff2 = _lin(self.H)
        self.ta = _lin(self.H)
        self.tb = _lin(self.H)
        # ncps inits these like any Linear; a small std keeps the initial cell from
        # being saturated by the timespan (ta*ts is an exponent of a sigmoid).
        nn.init.normal_(self.ta.weight, std=0.1)
        nn.init.normal_(self.tb.weight, std=0.1)
        nn.init.zeros_(self.tb.bias)

        self.ret = None
        if v["retention_gate"]:
            self.ret = nn.Parameter(torch.full((self.H,), -3.0))
        self.drop = nn.Dropout(dropout)
        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": int(self.H),
            "inp_dim": int(self.inp.d), "z_units": int(z_units),
        }
        self.cfg_unused = sorted(unused)

    # ------------------------------------------------------------------ #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {"h": torch.zeros(B, self.H, device=device)}

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None):
        # CfC is fully input-driven: nothing can be hoisted out of the loop.
        # NOTE: the timespan-unit normalisation (dt_scale) is applied centrally in
        # LoopCore.forward/step, *before* this, so `dts` is already scaled.
        return {"xn": self.inp(x, dt, mask), "mask": mask,
                "ts": dts.unsqueeze(-1)}                          # [B,T,1] seconds

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        h_prev = state["h"]
        x = torch.cat([ctx["xn"][:, t], h_prev], dim=-1)          # (batch, d+H)
        if self.bb_w is not None:
            z = lecun_tanh(F.linear(x, self.bb_w, self.bb_b))
        else:
            z = x
        ff1 = torch.tanh(self.ff1(z))
        ff2 = torch.tanh(self.ff2(z))
        tt = torch.sigmoid(self.ta(z) * ctx["ts"][:, t] + self.tb(z))
        h_new = ff1 * (1.0 - tt) + tt * ff2                       # no h self-loop
        if self.ret is not None:
            lam = torch.sigmoid(self.ret)
            h_new = h_new + lam * (h_prev - h_new)
        h = apply_mask(self.drop(h_new), h_prev, ctx["mask"][:, t])
        return h, {"h": h}, {"time_gate": tt}
