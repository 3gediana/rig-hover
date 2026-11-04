"""Standard gated recurrent baselines (``gru``, ``lstm``).

``torch.nn.GRUCell`` / ``LSTMCell`` are the reference operators, but they must be
driven one step at a time: a GRU given an all-zero input still moves its state, so
padding *cannot* be handled by zeroing the input (that would corrupt the state at
every padded step and leak into later real steps).  The contract therefore
requires an explicit ``state = where(mask_t, state_new, state_prev)`` per step,
which is what :class:`~src.models.base.LoopCore` + ``apply_mask`` do.

Time enters only through the shared input stage (``log1p(dt)`` appended to the
input vector); these cells have no structural notion of time -- which is exactly
the weakness the D-dimension (irregular sampling) of the study is meant to expose.
``use_dt_struct`` therefore has *nothing to switch off* for these models: with
``use_dt_feature=False`` and ``use_dt_struct=False`` a GRU becomes fully
time-blind, which is the honest discrete-RNN ablation.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..common.contract import apply_mask
from .base import COMMON_SPEC, InputNorm, LoopCore, unpack_cfg


class _RecurrentBase(LoopCore):
    """Shared plumbing for the two torch cell baselines."""

    state_as_out = True          # out_t IS h_t -> aux["hidden"] aliases the output
    _spec = dict(COMMON_SPEC)

    def __init__(self, in_dim: int, hidden: int, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.use_dt_struct = bool(v["use_dt_struct"])
        self.dt_const = float(v["dt_const"])
        self.dt_scale = float(v["dt_scale"])
        self.H = int(hidden)
        self.out_dim = int(hidden)
        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": int(self.H),
            "inp_dim": int(self.inp.d),
        }
        self.cfg_unused = sorted(unused)

    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {"h": torch.zeros(B, self.H, device=device)}

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None):
        return {"xn": self.inp(x, dt, mask), "mask": mask}

    def _final_aux(self, state):
        return {}


class GRU(_RecurrentBase):
    """``h_t = GRUCell(x_t, h_{t-1})`` with an explicit masked state hold."""

    name = "gru"

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__(in_dim, hidden, **cfg)
        self.cell = nn.GRUCell(self.inp.d, self.H)
        self.drop = nn.Dropout(dropout)

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        h_prev = state["h"]
        h_new = self.cell(ctx["xn"][:, t], h_prev)
        h = apply_mask(self.drop(h_new), h_prev, ctx["mask"][:, t])
        return h, {"h": h}, {}


class LSTM(_RecurrentBase):
    """``h_t, c_t = LSTMCell(x_t, (h_{t-1}, c_{t-1}))`` with a masked state hold."""

    name = "lstm"

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__(in_dim, hidden, **cfg)
        self.cell = nn.LSTMCell(self.inp.d, self.H)
        self.drop = nn.Dropout(dropout)

    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {"h": torch.zeros(B, self.H, device=device),
                "c": torch.zeros(B, self.H, device=device)}

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        m = ctx["mask"][:, t]
        h_prev, c_prev = state["h"], state["c"]
        h_new, c_new = self.cell(ctx["xn"][:, t], (h_prev, c_prev))
        h = apply_mask(self.drop(h_new), h_prev, m)
        c = apply_mask(c_new, c_prev, m)
        return h, {"h": h, "c": c}, {}
