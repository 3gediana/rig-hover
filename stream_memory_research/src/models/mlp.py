"""Memoryless floor: a per-frame MLP.

No state at all, so it is trivially causal and the mask changes nothing except
suppressing the padded input.  Whatever a task scores above this floor is memory.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..common.contract import TemporalCore, count_params
from .base import COMMON_SPEC, InputNorm, safe_dt, scale_dt, unpack_cfg


class MLP(TemporalCore):
    """``out_t = MLP(LN(x_t [+ log1p(dt)]))`` -- independent for every frame."""

    name = "mlp"
    _spec = {**COMMON_SPEC, "n_layers": 2}

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.0, **cfg):
        super().__init__()
        v, unused = unpack_cfg(cfg, self._spec)
        self.inp = InputNorm(in_dim, v["use_dt_feature"], v["use_mask_feature"],
                             v["input_norm"])
        self.hidden = int(hidden)
        self.out_dim = int(hidden)
        self.dt_scale = float(v.get("dt_scale", 1.0))
        layers = [nn.Linear(self.inp.d, self.hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, int(v["n_layers"]) - 1)):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.cfg_effective = {
            **{k: v[k] for k in self._spec},
            "model": self.name, "in_dim": int(in_dim), "hidden": self.hidden,
            "inp_dim": int(self.inp.d),
            # a memoryless core has no state transition for dt to act on
            "use_dt_struct": "n/a (no state)",
        }
        self.cfg_unused = sorted(unused)

    def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None, roi_valid=None,
                roi_meta=None):
        dt = scale_dt(safe_dt(dt, mask), self.dt_scale)
        out = self.mlp(self.inp(x, dt, mask))
        return out, {}

    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        return {}

    def step(self, x_t, dt_t, mask_t=None, state=None, tok_t=None, tok_valid_t=None,
             roi_t=None, roi_valid_t=None, roi_meta_t=None):
        """Stateless: one frame in, one frame out, the state is always ``{}``."""
        B = x_t.shape[0]
        if mask_t is None:
            mask_t = torch.ones(B, dtype=torch.bool, device=x_t.device)
        mask_t = mask_t.to(torch.bool).reshape(B)
        out, _ = self.forward(x_t.unsqueeze(1), dt_t.reshape(B, 1), mask_t.unsqueeze(1))
        return out[:, 0], {}, {}

    @property
    def n_params(self) -> int:
        return count_params(self)
