"""Controlled synthetic memory tasks + probes (owner: synth-lab).

Public API
----------
layout      : tensor/channel contract (feat 64, tok 16x17, lab_state 256,
              lab_mask 16, lab_class, lab_event) + identity code codec
base        : RawEpisode, Timeline, raw_to_seq, npz io, irregular perturbations
tasks       : recall_single / recall_multi / capacity_scan / overwrite /
              vanish_reappear / change_detect / event_seq generators
pattern     : temporal-pattern family (pattern_count / pattern_order /
              pattern_rate / pattern_count_irregular /
              pattern_count_rate_mixed) + stretch / shuffle variants
probes      : model-agnostic probes + Perfect / Zero / Random reference oracles
config      : config schema + dataset plan (base, gap sweeps, capacity sweeps,
              temporal-shuffle controls, irregular variants, pattern family)
generate    : dataset writer (npz + contract.Manifest + summary.json)
make_all    : CLI (`python -m src.synth.make_all --out ... --tiny`)
tests_synth : unit tests / oracle sanity checks
"""
from __future__ import annotations

from . import layout

__all__ = ["layout", "base", "tasks", "pattern", "probes", "config", "generate"]
