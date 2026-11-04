"""Temporal-core zoo: one registry, one contract, every model comparable.

All cores implement the shared contract (DESIGN.md §2.1, ``src/common/contract.py``)::

    class TemporalCore(nn.Module):
        name: str
        def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None,
                    roi_valid=None, roi_meta=None) -> (out [B,T,H], aux: dict)

with these invariants, each enforced by ``tests_models.py``:

* **causal** -- ``out[:, :k]`` depends only on steps ``<= k``;
* **mask**  -- ``mask=False`` steps leave the state untouched;
* **dt**    -- physical seconds, ``dt[:, 0] == 0``, never normalised across a batch;
* **aux**   -- probes for auxiliary losses (see below).

Public API
----------
``build_core(name, in_dim, hidden, **cfg)``          -> TemporalCore
``build_core_matched(name, in_dim, target_params, **cfg)`` -> TemporalCore
    Searches the hidden width so the built core lands within 1.25x of
    ``target_params`` (mandatory for fair comparisons, DESIGN §2.2).
``list_models()``                                    -> list[str]
``default_cfg(name)``                                -> dict of cfg defaults for that core
``CORE_CLASSES``                                     -> name -> class

Models
------
===========  ==================================================================
``mlp``      per-frame MLP, no state (memoryless floor)
``gru``      GRUCell, masked state hold
``lstm``     LSTMCell, masked state hold
``cfc``      ncps-semantics CfC (``cfc_ref.py`` operator, timespans = real dt)
``mamba``    S6-style selective SSM, pure torch, chunked parallel scan
             (``"mamba+time"`` alias = ``tdt=True``: real dt drives the internal step)
``deltanet`` fast-weight matrix memory only, delta rule (strong baseline)
``dsm``      **ours**: slow state + matrix fast memory (multi-timescale decay)
``dsm_slot`` **ours**: slow state + N content-addressed slots
``window``   ``dsm_slot`` with a hard per-object route + hard time window
===========  ==================================================================

``aux`` dict (float tensors, same device, [B,T,...]; padded rows are meaningless,
mask them with ``batch["mask"]``)
------------------------------------------------------------------------------
``read`` / ``mem_read``      [B,T,Dr]     fast-memory readout o_t
``hidden`` / ``h`` / ``slow`` / ``h_traj`` [B,T,H]  slow state h_t
``write_gate``               [B,T] or [B,T,N]  write strength beta_t (per slot)
``write_strength``           [B,T]        same value, un-reduced (pre-broadcast)
``surprise``                 [B,T]        ||v - r||, the drive of the write gate
``mem_norm``                 [B,T]        ||M|| (or ||vals||) per step
``route``                    [B,T,1,N]    slot routing weights (rows sum to 1)
``slot_read``                [B,T,N,Ds]   per-slot readout w_j*val_j
``slot_age``                 [B,T,N]      per-slot age in physical seconds
``write_mask``               [B,T,N]      straight-through binary per-slot write mask
``dt_ssm``                   [B,T]        mamba's internal (input-selected) timestep
``time_gate``                [B,T,H]      cfc's interpolation gate
``decay_stats``              dict of floats (NOT a tensor - skip in losses)
``slot_usage``/``slot_state_after``/``slot_t_last``   diagnostics from the final state

Ablation flags (accepted by **every** core; meaningful ones are recorded in
``core.cfg_effective``, meaningless ones in ``core.cfg_unused``)
--------------------------------------------------------------
common: ``use_dt_feature`` (T), ``use_mask_feature`` (T), ``input_norm`` (T),
``use_dt_struct`` (T), ``dt_const`` (0.1)
dsm/dsm_slot/deltanet: ``write_rule`` ("delta"|"outer"), ``read_mode``
("linear"|"bilinear"|"two_hop"), ``decay`` (T), ``n_decay_groups`` (4),
``beta_exp`` (F), ``dt_clamp_max`` (10.0), ``mem_dim`` (64), ``mem_off`` (F),
``slow_off`` (F), ``h_by_index`` (F), ``hard_write`` (F), ``write_threshold`` (0.5)
dsm_slot/window: ``n_slots`` (12), ``slot_dim`` (=mem_dim), ``slot_hard_route`` (F),
``route_mode`` ("content"|"positional"|"none"), ``tau_age_init`` (4.0), ``window_s`` (0.0)
"""
from __future__ import annotations

import math
from typing import Dict, List, Type

import torch
import torch.nn as nn

from ..common.contract import TemporalCore, count_params
from .base import COMMON_SPEC, DECAY_RATES, GroupedDecay, InputNorm, LoopCore, TimeDecay
from .cfc import CfC
from .deltanet import DeltaNet, MemRead
from .dsm import DSM, SLOT_SPEC, DSMSlot, WindowDSM
from .dual import DualMemory
from .mlp import MLP
from .recurrent import GRU, LSTM
from .ssm import Mamba

__all__ = [
    "build_core", "build_core_matched", "list_models", "default_cfg",
    "param_count", "CORE_CLASSES", "ALIASES",
    "TemporalCore", "MLP", "GRU", "LSTM", "CfC", "Mamba", "DeltaNet",
    "DSM", "DSMSlot", "WindowDSM",
]

CORE_CLASSES: Dict[str, Type[nn.Module]] = {
    "dual": DualMemory,
    "mlp": MLP,
    "gru": GRU,
    "lstm": LSTM,
    "cfc": CfC,
    "mamba": Mamba,
    "deltanet": DeltaNet,
    "dsm": DSM,
    "dsm_slot": DSMSlot,
    "window": WindowDSM,
}

#: model-name aliases -> (registered name, cfg forced on top of the user's cfg)
ALIASES: Dict[str, tuple] = {
    "mamba+time": ("mamba", {"tdt": True}),
    "mamba_tdt": ("mamba", {"tdt": True}),
}


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def build_core(name: str, in_dim: int, hidden: int, **cfg) -> TemporalCore:
    """Build one temporal core.

    ``build_core("dsm", in_dim=768, hidden=128, mem_dim=64, ...)``

    Every core accepts the full suite config; keys it does not use are listed in
    ``core.cfg_unused`` instead of raising, so one ablation file can drive the
    whole zoo.  Keys that *do* change behaviour are echoed in
    ``core.cfg_effective`` (recorded into metrics.json).
    """
    key = str(name)
    forced: Dict[str, object] = {}
    if key in ALIASES:
        key, forced = ALIASES[name]
    if key not in CORE_CLASSES:
        raise KeyError(f"unknown core {name!r}; available: {list_models()}")
    kwargs = {**forced, **cfg}
    return CORE_CLASSES[key](in_dim=int(in_dim), hidden=int(hidden), **kwargs)


def default_cfg(name: str) -> Dict[str, object]:
    """Default cfg of a core (what it would honour if built with no kwargs)."""
    key, forced = ALIASES.get(name, (name, {}))
    cls = CORE_CLASSES[key]
    spec = dict(getattr(cls, "_spec", COMMON_SPEC))
    spec.update(forced)
    return spec


def list_models(include_aliases: bool = False) -> List[str]:
    """Registered core names (sorted), optionally with the alias entries."""
    names = sorted(CORE_CLASSES)
    if include_aliases:
        names += sorted(ALIASES)
    return names


def param_count(name: str, in_dim: int, hidden: int, **cfg) -> int:
    """Parameter count of ``build_core(name, in_dim, hidden, **cfg)``."""
    return int(count_params(build_core(name, in_dim=in_dim, hidden=hidden, **cfg)))


def build_core_matched(name: str, in_dim: int, target_params: int,
                       **cfg) -> TemporalCore:
    """Build a core whose parameter count is within 1.25x of ``target_params``.

    Parameter matching is a hard fairness requirement (DESIGN §2.2): the only
    thing that may differ between compared models is the temporal operator, so
    the hidden width is tuned until the *total* parameter count lands close to a
    budget.  Monotone bisection on ``hidden``; the returned core carries
    ``matched_hidden / matched_params / matched_target / matched_ratio``.
    """
    target = float(target_params)
    if target <= 0:
        raise ValueError("target_params must be positive")
    best = None      # (abs log ratio, hidden, params)
    lo, hi = 4, 4096
    while lo <= hi:
        mid = (lo + hi) // 2
        n = param_count(name, in_dim, mid, **cfg)
        ratio = abs(math.log(max(n, 1) / target))
        if best is None or ratio < best[0]:
            best = (ratio, mid, n)
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    # refine around the bracket: params(hidden) is monotone but not smooth
    _, h0, _ = best
    for h in range(max(4, h0 - 6), h0 + 7):
        n = param_count(name, in_dim, h, **cfg)
        ratio = abs(math.log(max(n, 1) / target))
        if ratio < best[0]:
            best = (ratio, h, n)
    _, hidden, n_best = best
    core = build_core(name, in_dim=in_dim, hidden=hidden, **cfg)
    exact = int(count_params(core))
    ratio = exact / target
    core.matched_hidden = int(hidden)
    core.matched_params = int(exact)
    core.matched_target = int(round(target))
    core.matched_ratio = float(ratio)
    if not (1.0 / 1.25 <= ratio <= 1.25):
        raise RuntimeError(
            f"{name}: cannot match {target_params} params within 1.25x "
            f"(best hidden={hidden} -> {exact} params, ratio {ratio:.3f})")
    return core
