"""`dual` — the redesigned dual-memory core (ROUND 8 of the research loop).

Motivation (measured, see reports/中文总结报告.md §5.3 and §7 conclusion 3)
-------------------------------------------------------------------------
The single-bank DSM has to choose ONE decay policy, and the data show the right
choice is opposite for two kinds of temporal task:

* counting / holding  ("how many knocks?")  -> decay destroys the history:
  `pattern_count`: 4-group decay 0.553, **no decay 0.889**, single timescale 0.333 (=chance)
* order / direction   ("which way does time run?") -> decay is essential:
  `ucf101_order`: 4-group decay **0.943**, single timescale 0.500 (=chance)

So the honest redesign is to stop choosing: run **two fast-memory banks in
parallel on the same input and slow state**,

* **retention bank**  -- `decay=False`: delta-rule memory with no time decay.
  Keeps an arbitrarily long history; this is what counting/recall needs.
* **order bank**      -- `decay=True, n_decay_groups=4`: multi-timescale decay
  (tau = 0.125 / 0.5 / 2 / 50 s). Acts as a bank of leaky integrators, i.e. a
  time-domain filter bank; this is what direction/rhythm discrimination needs.

and fuse the two readouts with a small MLP.  Both banks see exactly the same
observations and timestamps, so nothing extra is peeking.

This core is deliberately a *composition of already-tested parts*: it instantiates
two `DSM` cores from `dsm.py`, which carry the causality / masking / dt-property
tests of the zoo.  The only new parameters are the fusion head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..common.contract import TemporalCore, count_params


def _clean_cfg(cfg: dict) -> dict:
    """Drop keys this wrapper sets per-bank so sub-cores cannot be mis-configured."""
    out = dict(cfg)
    for k in ("decay", "n_decay_groups", "mem_dim", "mem_off", "slow_off"):
        out.pop(k, None)
    return out


class DualMemory(TemporalCore):
    """slow state (shared input) + retention bank + multi-timescale order bank."""

    name = "dual"
    out_dim = 0

    def __init__(self, in_dim: int, hidden: int, mem_dim: int = 128,
                 hidden_ret: Optional[int] = None, hidden_ord: Optional[int] = None,
                 n_decay_groups: int = 4, fusion: str = "mlp",
                 ret_kind: str = "delta", **cfg):
        super().__init__()
        from .dsm import DSM
        from .recurrent import GRU

        sub_cfg = _clean_cfg(cfg)
        self.ret_kind = str(ret_kind)
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.h_ret = int(hidden_ret or hidden)
        self.h_ord = int(hidden_ord or hidden)
        self.mem_dim = int(mem_dim)
        self.fusion = str(fusion)

        if self.ret_kind == "gru":
            # dense recurrent state as the retention bank: a GRU overwrites a
            # full-rank state every step, which is exactly what the measured
            # failure of rank-1 delta writes on 16-object recall points to.
            self.ret = GRU(in_dim=int(in_dim), hidden=self.h_ret, **cfg)
        elif self.ret_kind == "delta":
            self.ret = DSM(in_dim=int(in_dim), hidden=self.h_ret, mem_dim=int(mem_dim),
                           decay=False, **sub_cfg)
        else:
            raise ValueError(f"ret_kind={ret_kind!r} not in {{'delta','gru'}}")
        self.ord = DSM(in_dim=int(in_dim), hidden=self.h_ord, mem_dim=int(mem_dim),
                       decay=True, n_decay_groups=int(n_decay_groups), **sub_cfg)

        fused_dim = self.h_ret + self.h_ord
        if self.fusion == "mlp":
            self.fuse = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, int(hidden)), nn.GELU(),
                nn.Linear(int(hidden), int(hidden)),
            )
        elif self.fusion == "linear":
            self.fuse = nn.Linear(fused_dim, int(hidden))
        else:
            raise ValueError(f"fusion={fusion!r} not in {{'mlp','linear'}}")
        self.out_dim = int(hidden)
        self.cfg_effective = {
            "model": "dual", "in_dim": self.in_dim, "hidden": self.hidden,
            "mem_dim": self.mem_dim, "hidden_ret": self.h_ret, "hidden_ord": self.h_ord,
            "n_decay_groups": int(n_decay_groups), "fusion": self.fusion,
            "banks": f"retention({self.ret_kind}) + order(multi-timescale decay delta)",
            "ret_kind": self.ret_kind,
            **{f"ret_{k}": v for k, v in (self.ret.cfg_effective or {}).items()},
            **{f"ord_{k}": v for k, v in (self.ord.cfg_effective or {}).items()},
        }
        self.cfg_unused = {}

    # ------------------------------------------------------------------ #
    def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None,
                roi_valid=None, roi_meta=None):
        o_ret, aux_ret = self.ret(x, dt, mask, tok=tok)
        o_ord, aux_ord = self.ord(x, dt, mask, tok=tok)
        out = self.fuse(torch.cat([o_ret, o_ord], dim=-1))
        aux: Dict[str, torch.Tensor] = {}
        for k, v in (aux_ret or {}).items():
            aux[f"ret_{k}"] = v
        for k, v in (aux_ord or {}).items():
            aux[f"ord_{k}"] = v
        # expose the fused stream under the names the trainers' probes expect
        aux["hidden"] = out
        aux["h"] = out
        aux["slot_read"] = aux.get("ord_slot_read", aux.get("ret_slot_read"))
        aux["route"] = aux.get("ord_route", aux.get("ret_route"))
        aux["read"] = torch.cat([o_ret, o_ord], dim=-1)
        return out, aux

    def init_state(self, B: int, device) -> dict:
        return {"ret": self.ret.init_state(B, device) if hasattr(self.ret, "init_state") else {},
                "ord": self.ord.init_state(B, device) if hasattr(self.ord, "init_state") else {}}

    @property
    def n_params(self) -> int:
        return count_params(self)
