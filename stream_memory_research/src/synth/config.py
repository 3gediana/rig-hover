"""Configuration for the synthetic memory benchmark.

A single JSON config fully determines the suite.  `synth_v1.json` is the
canonical one; `load_config` deep-merges it over the defaults below so the Lead
can override any single knob without re-specifying the whole file.

Plan structure
--------------
base tasks        train/val/test datasets (trainable)
gap sweeps        eval-only test-split datasets, one fixed hidden gap each
                  (`--eval-on` targets, DESIGN §1.6 sweep 0.1 .. 3.2 s)
capacity sweeps   eval-only test-split datasets, one fixed K each (1,2,4,8,12,16)
shuffled control  eval-only copy of each base task's test split whose rows are
                  permuted in time (LIT-BRIEF temporal-shuffle control)
irregular         the same base episodes regenerated under lognormal Δt, 30/50/
                  70 % dropout, repeated frames, burst drops, 2x/4x/8x time
                  stretch and a mixed variant (DESIGN §1.6 (a)-(e))
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pattern import is_pattern_task, pattern_defaults
from .tasks import task_defaults

TINY_SUBDIR = "tiny"


DEFAULTS: Dict[str, Any] = {
    "version": "v1",
    "seed_base": 20261103,
    "nominal_dt": 0.1,
    # split by EPISODE GENERATOR SEED GROUP, never by frame
    "split_ranges": {"train": [0, 800], "val": [800, 1000], "test": [1000, 1200]},
    "counts": {"train": 200, "val": 20, "test": 20},
    "tiny_counts": {"train": 12, "val": 4, "test": 4},
    "eval_count": 40,            # per eval-only sweep dataset (test split)
    "tiny_eval_count": 3,
    "shuffle_count": 30,         # rows copied from the base task's test split
    "tiny_shuffle_count": 3,
    "irregular_counts": {"train": 60, "val": 8, "test": 8},
    "tiny_irregular_counts": {"train": 2, "val": 1, "test": 1},
    "tasks": {},                 # per-task generator overrides (merged over task_defaults)
    "plan": {
        "base": ["recall_single", "recall_multi", "overwrite",
                 "vanish_reappear", "change_detect", "event_seq"],
        # eval-only fixed-hidden-gap sweeps.  `params_extra` may contain the
        # placeholder "__GAP_STEPS__" which is replaced by round(gap/Δ0).
        "gap_sweeps": [
            {"task": "recall_single", "base_dataset": "synth_recall_single",
             "gaps_s": [0.1, 0.2, 0.4, 0.8, 1.6, 3.2], "params_extra": {},
             "name_fmt": "synth_recall_single_gap{gap:g}s", "role": "single_gap"},
            {"task": "recall_multi", "base_dataset": "synth_recall_multi",
             "gaps_s": [0.1, 0.2, 0.4, 0.8, 1.6, 3.2],
             "params_extra": {"gap_steps_fixed": "__GAP_STEPS__",
                              "k_choices": [8]},
             "name_fmt": "synth_recall_multi_gap{gap:g}s", "role": "multi_gap"},
            {"task": "event_seq", "base_dataset": "synth_event_seq",
             "gaps_s": [0.2, 0.4, 0.8, 1.6, 3.2, 6.4],
             "params_extra": {"fixed_gap_steps": "__GAP_STEPS__"},
             "name_fmt": "synth_event_seq_gap{gap:g}s", "role": "event_gap"},
        ],
        "capacity": {
            "task": "capacity_scan",
            "base_dataset": "synth_recall_multi",
            "k_values": [1, 2, 4, 8, 12, 16],
            "name_fmt": "synth_capacity_K{k}",
        },
        "shuffled": {"enabled": True, "suffix": "_shuffled"},
        # ---- temporal-pattern family (core task candidate) -----------------
        "pattern": {
            "enabled": True,
            "shuffled": {"enabled": True, "suffix": "_shuffled", "count": 20},
            "datasets": [
                # counts are exact multiples of n_classes => perfectly balanced
                {"name": "synth_pattern_count", "task": "pattern_count",
                 "counts": {"train": 201, "val": 21, "test": 21}},
                {"name": "synth_pattern_order", "task": "pattern_order",
                 "counts": {"train": 200, "val": 20, "test": 20}},
                {"name": "synth_pattern_rate", "task": "pattern_rate",
                 "counts": {"train": 200, "val": 20, "test": 20}},
                {"name": "synth_pattern_count_irregular", "task": "pattern_count",
                 "params": {"irregular": "mixed"},
                 "counts": {"train": 201, "val": 21, "test": 21}},
                {"name": "synth_pattern_count_rate_mixed",
                 "task": "pattern_count_rate_mixed",
                 "counts": {"train": 200, "val": 20, "test": 20}},
            ],
            # eval-only physics-time tests: identical episodes, clock scaled
            "stretch": [
                {"base": "synth_pattern_count", "factor": 2.0, "count": 21},
                {"base": "synth_pattern_count", "factor": 4.0, "count": 21},
                {"base": "synth_pattern_rate", "factor": 2.0, "count": 20},
                {"base": "synth_pattern_rate", "factor": 4.0, "count": 20},
            ],
        },
        "irregular": {
            "bases": ["recall_single", "recall_multi"],
            "variants": [
                {"name": "logdt", "op": "logdt", "params": {"cv": 0.5}},
                {"name": "drop30", "op": "drop30", "params": {}},
                {"name": "drop50", "op": "drop50", "params": {}},
                {"name": "drop70", "op": "drop70", "params": {}},
                {"name": "repeat", "op": "repeat", "params": {"prob": 0.3}},
                {"name": "burst", "op": "burst", "params": {"n_bursts": 2,
                                                             "burst_frac": 0.25}},
                {"name": "stretch2", "op": "stretch2", "params": {}},
                {"name": "stretch4", "op": "stretch4", "params": {}},
                {"name": "stretch8", "op": "stretch8", "params": {}},
                {"name": "mixed", "op": "mixed", "params": {"cv": 0.25}},
            ],
            "name_fmt": "synth_irregular_{base}_{variant}",
        },
    },
}


@dataclass
class DatasetSpec:
    """One generated dataset (a directory of npz + one manifest)."""

    name: str
    task: str
    params: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    splits: List[str] = field(default_factory=list)
    eval_only: bool = False
    base_dataset: Optional[str] = None
    rng_name: Optional[str] = None          # dataset name used to seed the RNG
    irregular: Optional[str] = None         # irregular variant op
    irregular_params: Dict[str, Any] = field(default_factory=dict)
    control: Optional[str] = None           # "temporal_shuffle"
    eval_manifests: List[str] = field(default_factory=list)
    sweep_gap_s: Optional[float] = None     # designated hidden gap (seconds)
    capacity_k: Optional[int] = None        # designated K for capacity sweeps
    role: str = ""                          # "single_gap" / "multi_gap" / ...
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "task": self.task, "params": self.params,
            "counts": self.counts, "splits": list(self.splits),
            "eval_only": bool(self.eval_only),
            "base_dataset": self.base_dataset,
            "irregular": self.irregular,
            "control": self.control,
            "eval_manifests": list(self.eval_manifests),
            "sweep_gap_s": self.sweep_gap_s,
            "capacity_k": self.capacity_k,
            "role": self.role,
        }


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg = deep_merge(cfg, user)
    # resolve task params = generator defaults <- per-task overrides
    params: Dict[str, Dict[str, Any]] = {}
    for task in _all_tasks_in_plan(cfg):
        defaults = (pattern_defaults(task) if is_pattern_task(task)
                    else task_defaults(task))
        params[task] = deep_merge(defaults, cfg.get("tasks", {}).get(task, {}))
        params[task]["dt"] = float(cfg["nominal_dt"])
    cfg["resolved_task_params"] = params
    return cfg


def _all_tasks_in_plan(cfg: Dict[str, Any]) -> List[str]:
    plan = cfg["plan"]
    tasks = list(plan.get("base", []))
    for sweep in plan.get("gap_sweeps", []) or []:
        t = sweep.get("task")
        if t and t not in tasks:
            tasks.append(t)
    cap = plan.get("capacity", {}).get("task")
    if cap and cap not in tasks:
        tasks.append(cap)
    for b in plan.get("irregular", {}).get("bases", []):
        if b not in tasks:
            tasks.append(b)
    # temporal-pattern family (base datasets + stretch variants)
    pat = plan.get("pattern", {}) or {}
    for d in pat.get("datasets", []) or []:
        t = d.get("task")
        if t and t not in tasks:
            tasks.append(t)
    base_task_of: Dict[str, str] = {
        d.get("name"): d.get("task") for d in pat.get("datasets", []) or []}
    for st in pat.get("stretch", []) or []:
        t = base_task_of.get(st.get("base"))
        if t and t not in tasks:
            tasks.append(t)
    return tasks


def _balanced(n: int, n_classes: int) -> int:
    """Largest count <= n that is an exact multiple of n_classes (>= 1 each)."""
    n_classes = max(int(n_classes), 1)
    return max(n_classes, (int(n) // n_classes) * n_classes)


def _tiny_pattern_counts(n_classes: int, tiny_counts: Dict[str, int]) -> Dict[str, int]:
    return {split: _balanced(int(tiny_counts.get(split, 0)), n_classes)
            for split in ("train", "val", "test")}


def build_plan(cfg: Dict[str, Any], tiny: bool = False) -> List[DatasetSpec]:
    """Expand the config into the full ordered list of dataset specs."""
    plan = cfg["plan"]
    base_tasks: List[str] = list(plan.get("base", []))
    base_datasets = {f"synth_{t}": t for t in base_tasks}

    counts = dict(cfg["tiny_counts"] if tiny else cfg["counts"])
    eval_count = int(cfg["tiny_eval_count"] if tiny else cfg["eval_count"])
    shuffle_count = int(cfg["tiny_shuffle_count"] if tiny else cfg["shuffle_count"])
    irr_counts = dict(cfg["tiny_irregular_counts"] if tiny
                      else cfg["irregular_counts"])
    splits = ["train", "val", "test"]

    specs: List[DatasetSpec] = []

    # ---- base trainable datasets -----------------------------------------
    for task in base_tasks:
        name = f"synth_{task}"
        specs.append(DatasetSpec(
            name=name, task=task, params=dict(cfg["resolved_task_params"][task]),
            counts=dict(counts), splits=list(splits), rng_name=name,
            notes=f"base {task} dataset (train/val/test, split by episode group)",
        ))

    # ---- eval-only gap sweep ---------------------------------------------
    dt0 = float(cfg["nominal_dt"])
    for sweep in plan.get("gap_sweeps", []) or []:
        task = sweep.get("task")
        if not task:
            continue
        base_ds = sweep.get("base_dataset") or f"synth_{task}"
        for g in sweep.get("gaps_s", []):
            g = float(g)
            name = sweep.get("name_fmt", "synth_{task}_gap{gap:g}s").format(
                task=task, gap=g)
            params = dict(cfg["resolved_task_params"][task])
            params["gap_s"] = g
            g_steps = max(1, int(round(g / dt0)))
            for k, v in (sweep.get("params_extra") or {}).items():
                params[k] = g_steps if v == "__GAP_STEPS__" else v
            specs.append(DatasetSpec(
                name=name, task=task, params=params,
                counts={"test": eval_count}, splits=["test"],
                eval_only=True, base_dataset=base_ds, rng_name=base_ds,
                sweep_gap_s=g, role=sweep.get("role", "gap"),
                notes=f"eval-only fixed hidden gap {g:g} s "
                      f"(= {g_steps} steps at Δ0={dt0}) — DESIGN §1.6 sweep",
            ))

    # ---- eval-only capacity sweep ----------------------------------------
    cs = plan.get("capacity", {})
    if cs and cs.get("task"):
        task = cs["task"]
        base_ds = cs.get("base_dataset") or f"synth_{task}"
        for k in cs.get("k_values", []):
            name = cs.get("name_fmt", "synth_capacity_K{k}").format(k=int(k))
            params = dict(cfg["resolved_task_params"][task])
            params["k"] = int(k)
            specs.append(DatasetSpec(
                name=name, task=task, params=params,
                counts={"test": eval_count}, splits=["test"],
                eval_only=True, base_dataset=base_ds, capacity_k=int(k),
                role="capacity",
                notes=f"eval-only capacity scan K={k} (DESIGN §1.6)",
            ))

    # ---- temporal-shuffle controls (eval-only) ---------------------------
    sh = plan.get("shuffled", {})
    if sh.get("enabled"):
        for task in base_tasks:
            base_ds = f"synth_{task}"
            name = base_ds + sh.get("suffix", "_shuffled")
            specs.append(DatasetSpec(
                name=name, task=task, params=dict(cfg["resolved_task_params"][task]),
                counts={"test": shuffle_count}, splits=["test"],
                eval_only=True, base_dataset=base_ds,
                control="temporal_shuffle",
                notes="LIT-BRIEF temporal-shuffle control: rows permuted in time, "
                      "timestamps unchanged and labels kept per row",
            ))

    # ---- irregular-sampling datasets -------------------------------------
    irr = plan.get("irregular", {})
    for base_task in irr.get("bases", []):
        base_ds = f"synth_{base_task}"
        for var in irr.get("variants", []):
            name = irr.get("name_fmt", "synth_irregular_{base}_{variant}").format(
                base=base_task, variant=var["name"])
            specs.append(DatasetSpec(
                name=name, task=base_task,
                params=dict(cfg["resolved_task_params"][base_task]),
                counts=dict(irr_counts), splits=list(splits),
                base_dataset=base_ds, rng_name=base_ds,
                irregular=var.get("op", var["name"]),
                irregular_params=dict(var.get("params", {})),
                notes=f"DESIGN §1.6 irregular variant '{var['name']}'",
            ))

    # ---- temporal-pattern family (core task candidate) -------------------
    pat = plan.get("pattern", {}) or {}
    if pat.get("enabled", False):
        pat_specs: List[DatasetSpec] = []
        for d in pat.get("datasets", []) or []:
            task = str(d["task"])
            name = str(d.get("name") or f"synth_{task}")
            params = deep_merge(pattern_defaults(task),
                                cfg.get("tasks", {}).get(task, {}))
            params = deep_merge(params, d.get("params", {}))
            params["dt"] = dt0
            n_classes = int(params.get("n_classes", 3))
            if tiny:
                c = _tiny_pattern_counts(n_classes, cfg["tiny_counts"])
            else:
                c = dict(d.get("counts") or counts)
            for split in c:
                if c[split] % n_classes != 0:
                    raise ValueError(
                        f"{name}: split {split} count {c[split]} is not a "
                        f"multiple of n_classes={n_classes} (classes must be "
                        f"balanced)")
            spec = DatasetSpec(
                name=name, task=task, params=params, counts=c,
                splits=list(splits), rng_name=name, role="pattern_base",
                notes=f"temporal-pattern task '{task}' (balanced classes, "
                      f"episode-generator split groups)",
            )
            specs.append(spec)
            pat_specs.append(spec)
        pat_by_name = {s.name: s for s in pat_specs}
        # eval-only physical-clock stretch variants (paired with the base)
        for st in pat.get("stretch", []) or []:
            base = pat_by_name.get(str(st.get("base")))
            if base is None:
                continue
            factor = float(st.get("factor", 1.0))
            name = f"{base.name}_stretch{factor:g}"
            params = dict(base.params)
            params["stretch"] = factor
            count = (max(int(base.params.get("n_classes", 3)),
                         int(_balanced(int(cfg["tiny_eval_count"]),
                                       int(base.params.get("n_classes", 3)))))
                     if tiny else int(st.get("count", eval_count)))
            specs.append(DatasetSpec(
                name=name, task=base.task, params=params,
                counts={"test": count}, splits=["test"],
                eval_only=True, base_dataset=base.name, rng_name=base.name,
                role="pattern_stretch",
                notes=f"eval-only physics-time stretch x{factor:g}; same "
                      f"episodes as {base.name}, only the clock is scaled",
            ))
        # temporal-shuffle controls for the pattern tasks
        psh = pat.get("shuffled", {}) or {}
        if psh.get("enabled", False):
            scount = int(psh.get("count", 20))
            for base in pat_specs:
                specs.append(DatasetSpec(
                    name=base.name + str(psh.get("suffix", "_shuffled")),
                    task=base.task, params=dict(base.params),
                    counts={"test": min(scount, base.counts.get("test", scount))},
                    splits=["test"], eval_only=True, base_dataset=base.name,
                    control="temporal_shuffle",
                    role="pattern_shuffled",
                    notes="temporal-shuffle control: rows permuted in time, "
                          "timestamps unchanged, labels kept per row",
                ))

    # ---- cross-links: main.extra.eval_manifests --------------------------
    by_base: Dict[str, List[str]] = {}
    for spec in specs:
        if spec.eval_only and spec.base_dataset:
            by_base.setdefault(spec.base_dataset, []).append(spec.name)
    for spec in specs:
        if not spec.eval_only and (spec.name in base_datasets
                                   or spec.role == "pattern_base"):
            spec.eval_manifests = list(by_base.get(spec.name, []))
            if spec.eval_manifests:
                spec.notes += "; eval-only companions: " + ", ".join(
                    spec.eval_manifests[:3]) + (" ..." if len(spec.eval_manifests) > 3 else "")

    return specs


def spec_by_name(specs: List[DatasetSpec]) -> Dict[str, DatasetSpec]:
    return {s.name: s for s in specs}
