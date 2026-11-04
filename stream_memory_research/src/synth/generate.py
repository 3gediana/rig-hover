"""Generate the whole controlled synthetic memory suite.

Entry point used by :mod:`src.synth.make_all`::

    generate_suite(out_dir, cfg, tiny=False, only=None, overwrite=False)

For every dataset spec it writes

    <out>/<dataset>/<dataset>_<split>_<episode:06d>.npz
    <out>/manifests/<dataset>.json          (contract.Manifest)
    <out>/summary.json                      (all datasets + oracle references)
    <out>/index.json                        (machine readable dataset index)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..common.contract import Manifest, SeqRecord, read_json, write_json
from . import layout as L
from . import probes as P
from .base import (RawEpisode, episode_rng, load_seq, perturb_raw, raw_to_seq,
                   save_seq, seed_of_episode, stable_hash_int, temporal_shuffle,
                   validate_seq)
from .config import DatasetSpec, load_config
from .pattern import (build_pattern_seq, is_pattern_task, pattern_channel_layout)
from .phases import MAIN_METRIC, PHASES
from .tasks import build_task

REFERENCE_SEQS_CAP = 40
SUITE_FORMAT = "synth-suite-v1"


# --------------------------------------------------------------------------- #
# output store: generate into a fast local staging dir, then bulk-copy
# --------------------------------------------------------------------------- #
class _Store:
    """Maps final output paths <-> local write paths.

    G:\\stream_memory_data is often busy (video downloads/feature extraction by
    the data-video teammate: we measured ~100 ms per small-file write during
    contention vs ~1.4 ms on D:).  We therefore generate into a local staging
    directory and bulk-copy the finished tree with Robocopy /MT afterwards.
    """

    def __init__(self, out_dir, staging_dir=None):
        self.out = Path(out_dir)
        self.root = Path(staging_dir) if staging_dir else Path(out_dir)
        self.staged = self.root.resolve() != self.out.resolve()

    def local(self, final_path) -> Path:
        final_path = Path(final_path)
        if not self.staged:
            return final_path
        try:
            rel = final_path.resolve().relative_to(self.out.resolve())
        except ValueError:  # pragma: no cover - defensive
            return self.root / final_path.name
        return self.root / rel

    def manifest_path(self, name: str) -> Path:
        return self.root / "manifests" / f"{name}.json"

    def copy_to_out(self, quiet: bool = False) -> float:
        if not self.staged:
            return 0.0
        t0 = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        if not quiet:
            n = sum(1 for _ in self.root.rglob("*") if _.is_file())
            print(f"[make_all] copying {n} files to {self.out} "
                  f"(staged on {self.root}) ...", flush=True)
        ok = False
        try:
            import subprocess
            res = subprocess.run(
                ["robocopy", str(self.root), str(self.out), "/E", "/MT:32",
                 "/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:2", "/W:1"],
                capture_output=True, text=True)
            ok = res.returncode in (0, 1, 2, 3, 4, 5, 6, 7)
        except Exception:
            ok = False
        if not ok:  # fallback (slower) path
            import shutil
            for src in self.root.rglob("*"):
                if src.is_file():
                    dst = self.out / src.relative_to(self.root)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                        shutil.copyfile(src, dst)
        dt = time.time() - t0
        if not quiet:
            print(f"[make_all] copy done in {dt:.1f}s", flush=True)
        return dt


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _episode_seed(cfg: Dict[str, Any], spec: DatasetSpec, episode: int) -> List[int]:
    return seed_of_episode(int(cfg["seed_base"]),
                           spec.rng_name or spec.name, int(episode))


def _split_bounds(cfg: Dict[str, Any], split: str) -> Tuple[int, int]:
    lo, hi = cfg["split_ranges"][split]
    return int(lo), int(hi)


def _seq_id(spec: DatasetSpec, split: str, episode: int) -> str:
    return f"{spec.name}_{split}_{episode:06d}"


def _build_one(cfg: Dict[str, Any], spec: DatasetSpec, split: str, episode: int,
               seq_path: Path, final_path: Optional[Path] = None) -> SeqRecord:
    dt0 = float(cfg["nominal_dt"])
    seed_triple = _episode_seed(cfg, spec, episode)
    rng = np.random.default_rng(np.random.SeedSequence(seed_triple))
    if is_pattern_task(spec.task):
        from .pattern import class_for_episode
        class_id = class_for_episode(cfg, spec, split, episode)
        seq = build_pattern_seq(rng, spec.params, class_id)
        validate_seq(seq)
        save_seq(seq, seq_path)
        T = int(seq["t"].shape[0])
        meta = {
            "episode": int(episode),
            "seed": [int(x) for x in seed_triple],
            "split_group": list(cfg["split_ranges"][split]),
            "class_id": int(class_id),
            "kind": str(spec.params.get("kind", "count")),
            "n_classes": int(spec.params.get("n_classes", 3)),
            "stretch": float(spec.params.get("stretch", 1.0)),
            "irregular": spec.params.get("irregular"),
            "T": T,
            "same_raw_as": spec.base_dataset or spec.name,
        }
        return SeqRecord(
            seq_id=_seq_id(spec, split, episode),
            source_id=f"ep{episode:06d}",
            split=split,
            T=T,
            path=str(final_path if final_path is not None else seq_path),
            labels=["state", "mask", "class", "event", "age", "gap_s"],
            meta=meta,
        )
    task = build_task(spec.task, spec.params)
    raw = task.make(rng)
    if spec.irregular:
        rng_p = np.random.default_rng(np.random.SeedSequence(
            seed_triple + [stable_hash_int("perturb:" + spec.irregular, 999983)]))
        raw = perturb_raw(raw, spec.irregular, rng_p, nominal_dt=dt0,
                          params=spec.irregular_params)
    seq = raw_to_seq(raw, dt0=dt0)
    validate_seq(seq)
    save_seq(seq, seq_path)
    T = int(seq["t"].shape[0])
    meta = {
        "episode": int(episode),
        "seed": [int(x) for x in seed_triple],
        "split_group": list(cfg["split_ranges"][split]),
        "gap_s": float(raw.gap_s),
        "K": int(raw.K),
        "T": T,
        "same_raw_as": spec.base_dataset or spec.name,
    }
    if raw.meta:
        for k in ("hold_steps", "post_steps", "absence_steps", "reappear_masked",
                  "n_target", "n_distractor", "n_blocks", "retention_steps",
                  "irregular", "gap_steps", "v1_code", "v2_code", "identity"):
            if k in raw.meta:
                meta[k] = raw.meta[k]
    return SeqRecord(
        seq_id=_seq_id(spec, split, episode),
        source_id=f"ep{episode:06d}",
        split=split,
        T=T,
        path=str(final_path if final_path is not None else seq_path),
        labels=["state", "mask", "class", "event", "age", "gap_s"],
        meta=meta,
    )


def _build_shuffled(cfg: Dict[str, Any], spec: DatasetSpec, episode: int,
                    base_seq_path: Path, seq_path: Path,
                    final_path: Optional[Path] = None) -> SeqRecord:
    truth = load_seq(base_seq_path)
    rng = np.random.default_rng(np.random.SeedSequence(
        _episode_seed(cfg, spec, episode) + [424242]))
    shuffled = temporal_shuffle(truth, rng)
    validate_seq(shuffled)
    save_seq(shuffled, seq_path)
    T = int(shuffled["t"].shape[0])
    return SeqRecord(
        seq_id=_seq_id(spec, "test", episode),
        source_id=f"ep{episode:06d}",
        split="test",
        T=T,
        path=str(final_path if final_path is not None else seq_path),
        labels=["state", "mask", "class", "event", "age", "gap_s"],
        meta={"episode": int(episode), "T": T, "control": "temporal_shuffle",
              "base_seq": str(base_seq_path), "labels_kept_per_row": True},
    )


# --------------------------------------------------------------------------- #
# manifest / dataset
# --------------------------------------------------------------------------- #
def _manifest(spec: DatasetSpec, cfg: Dict[str, Any], records: List[SeqRecord],
              out_dir: Path) -> Manifest:
    extra = {
        "generator": "src/synth",
        "suite_format": SUITE_FORMAT,
        "task_id": spec.task,
        "seed_base": int(cfg["seed_base"]),
        "split_policy": (
            "split by episode generator seed group: train=[%d,%d), val=[%d,%d), "
            "test=[%d,%d); source_id = 'ep<index>' (never split by frame); "
            "test episodes are generated independently"
            % (cfg["split_ranges"]["train"][0], cfg["split_ranges"]["train"][1],
               cfg["split_ranges"]["val"][0], cfg["split_ranges"]["val"][1],
               cfg["split_ranges"]["test"][0], cfg["split_ranges"]["test"][1])),
        "split_ranges": {k: list(v) for k, v in cfg["split_ranges"].items()},
        "nominal_dt": float(cfg["nominal_dt"]),
        "layout": L.layout_dict(),
        "channel_layout": L.channel_layout_dict(),
        "phases": {name: int(pid) for pid, name in PHASES.get(spec.task, {}).items()},
        "main_metric": MAIN_METRIC.get(
            "capacity_scan" if spec.task == "capacity_scan" else spec.task, "cos"),
        "task_params": {k: v for k, v in spec.params.items() if k != "dt"},
        "eval_only": bool(spec.eval_only),
        "base_dataset": spec.base_dataset,
        "base_task": (spec.base_dataset or "").replace("synth_", "")
                      if spec.base_dataset else spec.task,
        "sweep_gap_s": spec.sweep_gap_s,
        "gap_s": spec.sweep_gap_s,
        "capacity_k": spec.capacity_k,
        "sweep_role": spec.role,
        "layout_signature": L.layout_signature(),
        "irregular": spec.irregular,
        "irregular_params": dict(spec.irregular_params),
        "control": spec.control,
        "eval_manifests": list(spec.eval_manifests),
        "eval_manifest_paths": [
            str(out_dir / "manifests" / f"{n}.json") for n in spec.eval_manifests],
        "notes": spec.notes,
        "oracle_reference_definition": {
            "perfect": "unbounded faithful memory reading only the observation "
                       "(PerfectMemoryOracle) -> sanity upper bound 1.0",
            "memoryless": "outputs the current observation only "
                          "(ZeroMemoryOracle) -> memoryless bound",
            "chance": "random read-out (RandomOracle) -> theoretical chance",
        },
    }
    if is_pattern_task(spec.task):
        n_classes = int(spec.params.get("n_classes", 3))
        extra.update({
            "pattern_family": True,
            "n_classes": n_classes,
            "chance_level_acc": 1.0 / float(n_classes),
            "pattern_kind": spec.params.get("kind", "count"),
            "stretch": float(spec.params.get("stretch", 1.0)),
            "pattern_channel_layout": pattern_channel_layout(),
            "balanced_classes": True,
        })
    if spec.control == "temporal_shuffle":
        extra["main_metric"] = MAIN_METRIC.get(spec.task, "cos")
    return Manifest(
        dataset=spec.name,
        task="synth_mem",
        feature_dim=L.DG,
        seqs=records,
        tok_dim=L.TOK_DIM,
        encoder={"name": "synthetic_controlled", "version": SUITE_FORMAT,
                 "task": spec.task},
        split_key="source_id",
        label_specs=L.label_specs(),
        extra=extra,
    )


def generate_dataset(cfg: Dict[str, Any], spec: DatasetSpec, out_dir: Path,
                     store: Optional[_Store] = None, overwrite: bool = False,
                     quiet: bool = True,
                     base_specs: Optional[Dict[str, DatasetSpec]] = None
                     ) -> Tuple[Manifest, Dict[str, Any]]:
    """Generate one dataset; returns (manifest, timing stats)."""
    t0 = time.time()
    out_dir = Path(out_dir)
    store = store or _Store(out_dir)
    ds_dir = store.local(out_dir / spec.name)          # where bytes are written
    ds_dir.mkdir(parents=True, exist_ok=True)
    records: List[SeqRecord] = []
    n_written = 0

    if spec.control == "temporal_shuffle":
        if not spec.base_dataset:
            raise ValueError(f"{spec.name}: shuffled control needs a base dataset")
        base_dir = store.local(out_dir / spec.base_dataset)
        if not base_dir.exists():
            raise FileNotFoundError(
                f"{spec.name}: base dataset dir missing ({base_dir}); generate "
                f"the base datasets before the shuffled controls")
        base_manifest_path = store.manifest_path(spec.base_dataset)
        if base_manifest_path.exists():
            base_manifest = Manifest.load(base_manifest_path)
            test_recs = base_manifest.split("test")
        else:  # fall back to a directory scan
            paths = sorted(base_dir.glob("*_test_*.npz"))
            test_recs = [SeqRecord(seq_id=p.stem, source_id=p.stem, split="test",
                                   T=0, path=str(p)) for p in paths]
        n = int(spec.counts.get("test", 0))
        chosen = sorted(test_recs, key=lambda r: r.seq_id)[:n]
        for rec in chosen:
            episode = int(rec.seq_id.rsplit("_", 1)[-1])
            final_path = out_dir / spec.name / f"{_seq_id(spec, 'test', episode)}.npz"
            seq_path = store.local(final_path)
            base_local = store.local(Path(rec.path))
            if not seq_path.exists() or overwrite:
                records.append(_build_shuffled(cfg, spec, episode,
                                               base_local, seq_path, final_path))
                n_written += 1
            else:
                records.append(SeqRecord(
                    seq_id=_seq_id(spec, "test", episode),
                    source_id=f"ep{episode:06d}", split="test",
                    T=int(load_seq(seq_path)["t"].shape[0]),
                    path=str(final_path),
                    labels=["state", "mask", "class", "event", "age", "gap_s"],
                    meta={"episode": episode, "control": "temporal_shuffle"}))
    else:
        for split in spec.splits:
            lo, hi = _split_bounds(cfg, split)
            n = int(spec.counts.get(split, 0))
            if n > (hi - lo):
                raise ValueError(
                    f"{spec.name}: {split} count {n} exceeds seed group "
                    f"[{lo},{hi})")
            for i in range(n):
                episode = lo + i
                final_path = out_dir / spec.name / f"{_seq_id(spec, split, episode)}.npz"
                seq_path = store.local(final_path)
                if seq_path.exists() and not overwrite:
                    truth = load_seq(seq_path)
                    T = int(truth["t"].shape[0])
                    records.append(SeqRecord(
                        seq_id=_seq_id(spec, split, episode),
                        source_id=f"ep{episode:06d}", split=split, T=T,
                        path=str(final_path),
                        labels=["state", "mask", "class", "event", "age", "gap_s"],
                        meta={"episode": episode, "seed":
                              _episode_seed(cfg, spec, episode)}))
                else:
                    records.append(_build_one(cfg, spec, split, episode,
                                              seq_path, final_path))
                    n_written += 1

    manifest = _manifest(spec, cfg, records, out_dir)
    manifest.check_split_disjoint()
    man_path = store.manifest_path(spec.name)
    manifest.save(man_path)
    final_man_path = out_dir / "manifests" / f"{spec.name}.json"
    final_ds_dir = out_dir / spec.name
    counts = manifest.counts()
    stats = {
        "dataset": spec.name,
        "task": spec.task,
        "main_metric": manifest.extra["main_metric"],
        "manifest": str(final_man_path),
        "path": str(final_ds_dir),
        "n_seqs": len(records),
        "splits": counts,
        "n_written": n_written,
        "T_range": _t_range(records),
        "feature_dim": manifest.feature_dim,
        "tok_dim": manifest.tok_dim,
        "label_dims": {k: v.get("dim", 1) for k, v in manifest.label_specs.items()},
        "layout_signature": L.layout_signature(),
        "eval_only": bool(spec.eval_only),
        "base_dataset": spec.base_dataset,
        "gap_s": spec.sweep_gap_s,
        "capacity_k": spec.capacity_k,
        "sweep_role": spec.role,
        "control": spec.control,
        "irregular": spec.irregular,
        "generation_seconds": round(time.time() - t0, 3),
    }
    return manifest, stats


def _t_range(records: Sequence[SeqRecord]) -> List[int]:
    Ts = [int(r.T) for r in records if r.T]
    return [min(Ts), max(Ts)] if Ts else [0, 0]


# --------------------------------------------------------------------------- #
# oracle references for summary.json
# --------------------------------------------------------------------------- #
def reference_for_manifest(manifest: Manifest, task: str,
                           cap: int = REFERENCE_SEQS_CAP,
                           store: Optional[_Store] = None) -> Dict[str, Any]:
    """perfect / memoryless / chance scores of the main metric (oracles only)."""
    recs = list(manifest.seqs)
    test = [r for r in recs if r.split == "test"] or recs
    test = sorted(test, key=lambda r: r.seq_id)[:cap]
    truths = []
    for r in test:
        p = store.local(r.path) if store is not None else r.path
        truths.append(load_seq(p))
    refs = P.reference_scores(truths, task, seed=0)
    chance_level = {
        "cos": 0.0,
        "acc": 1.0 / float(L.N_CODES),
        "f1": float(refs.get("chance", {}).get("f1", float("nan"))),
    }[manifest.extra["main_metric"]]
    return {
        "n_reference_seqs": len(truths),
        "main_metric": manifest.extra["main_metric"],
        "chance_level": chance_level,
        "perfect": refs["perfect"],
        "memoryless": refs["memoryless"],
        "chance": refs["chance"],
    }


# --------------------------------------------------------------------------- #
# suite
# --------------------------------------------------------------------------- #
def _summary_markdown(summary: Dict[str, Any]) -> str:
    """Human-readable reference table for the report (ASCII only on purpose:
    PowerShell's Get-Content mangles UTF-8 on this box)."""
    lines = [
        "# Synthetic memory benchmark - oracle reference points",
        "",
        f"suite `{summary.get('suite_format')}` version {summary.get('version')}"
        f" | {summary['n_datasets']} datasets"
        f" | {summary['n_sequences_total']} sequences"
        f" | Delta0 = {summary.get('nominal_dt')} s"
        f" | layout signature `{summary.get('layout_signature')}`",
        "",
        "`perfect` = unbounded faithful memory reading only the observations "
        "(upper bound); `memoryless` = current observation only (a model with no "
        "state); `chance` = random read-out.  Scores are the main metric on the "
        "test split.",
        "",
        "| dataset | task | metric | n | T range | gap s | perfect | memoryless | chance |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in summary["datasets"]:
        ref = ds.get("reference", {})
        mm = ds.get("main_metric", "cos")
        def _v(block):
            if isinstance(block, dict) and mm in block:
                try:
                    return f"{float(block[mm]):.3f}"
                except (TypeError, ValueError):
                    return "nan"
            return "nan"
        gap = ds.get("gap_s")
        gap_s = "-" if gap is None else f"{float(gap):g}"
        lines.append(
            f"| {ds['dataset']} | {ds['task']} | {mm} | {ds['n_seqs']} | "
            f"{ds['T_range'][0]}-{ds['T_range'][1]} | {gap_s} | "
            f"{_v(ref.get('perfect', {}))} | {_v(ref.get('memoryless', {}))} | "
            f"{_v(ref.get('chance', {}))} |")
    lines.append("")
    return "\n".join(lines)


def generate_suite(out_dir, cfg: Dict[str, Any], tiny: bool = False,
                   only: Optional[Sequence[str]] = None,
                   overwrite: bool = False, references: bool = True,
                   quiet: bool = False, staging_dir=None,
                   append: bool = False) -> Dict[str, Any]:
    from .config import build_plan, spec_by_name

    out_dir = Path(out_dir)
    if tiny:
        out_dir = out_dir / "tiny"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    store = _Store(out_dir, staging_dir)
    store.root.mkdir(parents=True, exist_ok=True)
    (store.root / "manifests").mkdir(parents=True, exist_ok=True)
    write_json(store.root / "config_used.json", cfg)

    specs = build_plan(cfg, tiny=tiny)
    if only:
        wanted = [s for s in str(only).split(",") if s]
        keep = [s for s in specs if s.name in wanted]
        missing = set(wanted) - {s.name for s in keep}
        if missing:
            raise KeyError(f"unknown dataset(s): {sorted(missing)}")
        specs = keep
    base_specs = spec_by_name(build_plan(cfg, tiny=tiny))

    t_all = time.time()
    manifests: List[Manifest] = []
    stats_list: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        manifest, stats = generate_dataset(cfg, spec, out_dir, store=store,
                                           overwrite=overwrite, quiet=quiet,
                                           base_specs=base_specs)
        manifests.append(manifest)
        stats_list.append(stats)
        if not quiet:
            print(f"[{i:2d}/{len(specs)}] {spec.name:42s} "
                  f"n={stats['n_seqs']:4d} splits={stats['splits']} "
                  f"T={stats['T_range']} {time.time() - t0:6.2f}s"
                  f"{' [eval-only]' if spec.eval_only else ''}")

    # ---- summary / index ------------------------------------------------
    summary: Dict[str, Any] = {
        "suite_format": SUITE_FORMAT,
        "version": cfg.get("version", "v1"),
        "tiny": bool(tiny),
        "out_dir": str(out_dir),
        "seed_base": int(cfg["seed_base"]),
        "nominal_dt": float(cfg["nominal_dt"]),
        "split_policy": (
            "episode-generator seed groups; train=[%d,%d) val=[%d,%d) "
            "test=[%d,%d); source_id='ep<index>'; split by episode, never by frame"
            % (cfg["split_ranges"]["train"][0], cfg["split_ranges"]["train"][1],
               cfg["split_ranges"]["val"][0], cfg["split_ranges"]["val"][1],
               cfg["split_ranges"]["test"][0], cfg["split_ranges"]["test"][1])),
        "datasets": stats_list,
        "n_datasets": len(stats_list),
        "n_sequences_total": int(sum(s["n_seqs"] for s in stats_list)),
        "layout": L.layout_dict(),
        "label_specs": L.label_specs(),
        "layout_signature": L.layout_signature(),
        "channel_layout": L.channel_layout_dict(),
        "generation_seconds": round(time.time() - t_all, 2),
    }
    index = {
        "suite_format": SUITE_FORMAT,
        "tiny": bool(tiny),
        "out_dir": str(out_dir),
        "datasets": [
            {"dataset": s["dataset"], "task": s["task"],
             "manifest": s["manifest"], "path": s["path"],
             "main_metric": s["main_metric"], "eval_only": s["eval_only"],
             "splits": s["splits"]}
            for s in stats_list
        ],
    }
    # ---- oracle reference points ---------------------------------------
    if references:
        t_ref = time.time()
        for manifest, stats in zip(manifests, stats_list):
            try:
                stats["reference"] = reference_for_manifest(
                    manifest, manifest.extra["task_id"], store=store)
            except Exception as exc:  # pragma: no cover - defensive
                stats["reference"] = {"error": repr(exc)}
            if not quiet:
                ref = stats["reference"]
                print(f"  ref {stats['dataset']:42s} "
                      f"perfect={_fmt(ref['perfect'])} "
                      f"memoryless={_fmt(ref['memoryless'])} "
                      f"chance={_fmt(ref['chance'])}")
        summary["reference_seconds"] = round(time.time() - t_ref, 2)
        summary["generation_seconds"] = round(time.time() - t_all, 2)

    # ---- append mode: merge with an existing summary (e.g. --only runs) ----
    if append:
        old_path = out_dir / "summary.json"
        if old_path.exists():
            old = read_json(old_path)
            by_name = {d["dataset"]: d for d in old.get("datasets", [])}
            for s in stats_list:
                by_name[s["dataset"]] = s
            from .config import build_plan as _bp
            order = [sp.name for sp in _bp(cfg, tiny=tiny)]
            merged = [by_name[n] for n in order if n in by_name]
            merged += [v for k, v in by_name.items() if k not in set(order)]
            stats_list = merged
            summary["datasets"] = stats_list
            summary["n_datasets"] = len(stats_list)
            summary["n_sequences_total"] = int(
                sum(int(s.get("n_seqs", 0)) for s in stats_list))
            index["datasets"] = [
                {"dataset": s["dataset"], "task": s["task"],
                 "manifest": s["manifest"], "path": s["path"],
                 "main_metric": s["main_metric"], "eval_only": s["eval_only"],
                 "splits": s["splits"]}
                for s in stats_list]

    write_json(store.root / "summary.json", summary)
    write_json(store.root / "index.json", index)
    (store.root / "summary.md").write_text(_summary_markdown(summary),
                                           encoding="utf-8")
    summary["copy_seconds"] = round(store.copy_to_out(quiet=quiet), 2)
    # re-write the summary at the final location so the timing is complete
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "index.json", index)
    (out_dir / "summary.md").write_text(_summary_markdown(summary),
                                       encoding="utf-8")
    return summary


def _fmt(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict):
        return str(d)
    keys = [k for k in ("cos", "acc", "f1") if k in d]
    if not keys and "post_event_cos" in d:
        return f"f1={d.get('f1', float('nan')):.3f}"
    return " ".join(f"{k}={float(d[k]):.3f}" if np.isfinite(float(d[k]))
                    else f"{k}=nan" for k in keys)


def rebuild_summary(out_dir, cfg: Optional[Dict[str, Any]] = None,
                    tiny: bool = False, references: bool = True,
                    quiet: bool = False) -> Dict[str, Any]:
    """Recompute summary.json / index.json / summary.md from existing manifests.

    Useful after moving the suite to another root, or after a run whose staging
    directory was already deleted.  Never touches the npz files.
    """
    from .config import build_plan, load_config
    out_dir = Path(out_dir)
    if tiny:
        out_dir = out_dir / "tiny"
    cfg = cfg or load_config(None)
    stats_list: List[Dict[str, Any]] = []
    for spec in build_plan(cfg, tiny=tiny):
        man_path = out_dir / "manifests" / f"{spec.name}.json"
        if not man_path.exists():
            continue
        man = Manifest.load(man_path)
        stats: Dict[str, Any] = {
            "dataset": spec.name,
            "task": man.extra.get("task_id", spec.task),
            "main_metric": man.extra.get("main_metric", "cos"),
            "manifest": str(man_path),
            "path": str(out_dir / spec.name),
            "n_seqs": len(man.seqs),
            "splits": man.counts(),
            "n_written": 0,
            "T_range": _t_range(man.seqs),
            "feature_dim": man.feature_dim,
            "tok_dim": man.tok_dim,
            "label_dims": {k: v.get("dim", 1)
                           for k, v in man.label_specs.items()},
            "layout_signature": man.extra.get("layout_signature"),
            "eval_only": bool(man.extra.get("eval_only")),
            "base_dataset": man.extra.get("base_dataset"),
            "gap_s": man.extra.get("gap_s"),
            "capacity_k": man.extra.get("capacity_k"),
            "sweep_role": man.extra.get("sweep_role"),
            "control": man.extra.get("control"),
            "irregular": man.extra.get("irregular"),
            "generation_seconds": None,
        }
        if references:
            try:
                stats["reference"] = reference_for_manifest(
                    man, man.extra["task_id"])
            except Exception as exc:  # pragma: no cover
                stats["reference"] = {"error": repr(exc)}
        stats_list.append(stats)
        if not quiet and references:
            print(f"  ref {spec.name:42s} "
                  f"perfect={_fmt(stats['reference']['perfect'])} "
                  f"memoryless={_fmt(stats['reference']['memoryless'])}",
                  flush=True)
    summary = {
        "suite_format": SUITE_FORMAT,
        "version": cfg.get("version", "v1"),
        "tiny": bool(tiny),
        "out_dir": str(out_dir),
        "seed_base": int(cfg["seed_base"]),
        "nominal_dt": float(cfg["nominal_dt"]),
        "split_policy": (
            "episode-generator seed groups; split by episode, never by frame"),
        "datasets": stats_list,
        "n_datasets": len(stats_list),
        "n_sequences_total": int(sum(s["n_seqs"] for s in stats_list)),
        "layout": L.layout_dict(),
        "label_specs": L.label_specs(),
        "layout_signature": L.layout_signature(),
        "channel_layout": L.channel_layout_dict(),
        "generation_seconds": None,
        "rebuilt": True,
    }
    index = {
        "suite_format": SUITE_FORMAT, "tiny": bool(tiny),
        "out_dir": str(out_dir),
        "datasets": [{"dataset": s["dataset"], "task": s["task"],
                      "manifest": s["manifest"], "path": s["path"],
                      "main_metric": s["main_metric"],
                      "eval_only": s["eval_only"], "splits": s["splits"]}
                     for s in stats_list],
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "index.json", index)
    (out_dir / "summary.md").write_text(_summary_markdown(summary),
                                       encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# verification helper (oracle sanity on generated test data)
# --------------------------------------------------------------------------- #
def validate_suite(out_dir, splits: Optional[Sequence[str]] = ("test",),
                   check_npz: bool = True, quiet: bool = False
                   ) -> Dict[str, Any]:
    """Structural validation of an existing suite against the contract.

    Checks every manifest (`Manifest.load`, `check_split_disjoint`, identical
    layout signature / feature_dim / tok_dim / label_specs) and, when
    `check_npz`, every npz of the requested splits (shape, dtype, strictly
    increasing t, no NaN, T matches the manifest record).
    """
    out_dir = Path(out_dir)
    man_dir = out_dir / "manifests"
    problems: List[str] = []
    n_npz = 0
    sig = L.layout_signature()
    man_paths = sorted(man_dir.glob("*.json"))
    for mp in man_paths:
        try:
            man = Manifest.load(mp)
        except Exception as exc:
            problems.append(f"{mp.name}: manifest load failed: {exc!r}")
            continue
        try:
            man.check_split_disjoint()
        except Exception as exc:
            problems.append(f"{mp.name}: split leak: {exc!r}")
        if man.feature_dim != L.DG or man.tok_dim != L.TOK_DIM:
            problems.append(f"{mp.name}: dims {man.feature_dim}/{man.tok_dim}")
        if man.label_specs != L.label_specs():
            problems.append(f"{mp.name}: label_specs differ")
        if man.extra.get("layout_signature") != sig:
            problems.append(f"{mp.name}: layout signature differs")
        for rec in man.seqs:
            if splits and rec.split not in tuple(splits):
                continue
            if not Path(rec.path).exists():
                problems.append(f"{mp.name}: missing {rec.path}")
                continue
            if not check_npz:
                continue
            try:
                arr = load_seq(rec.path)
                validate_seq(arr)
                if arr["t"].shape[0] != rec.T:
                    problems.append(f"{rec.seq_id}: T mismatch {arr['t'].shape[0]}"
                                    f" != {rec.T}")
                n_npz += 1
            except Exception as exc:
                problems.append(f"{rec.seq_id}: npz invalid: {exc!r}")
        if not quiet:
            print(f"  checked {man.dataset:44s} n={len(man.seqs):4d} "
                  f"splits={man.counts()}", flush=True)
    res = {"n_manifests": len(man_paths), "n_npz": n_npz,
           "n_problems": len(problems), "problems": problems,
           "ok": not problems}
    return res


def verify_suite(out_dir, cfg: Optional[Dict[str, Any]] = None,
                 tiny: bool = False, tasks: Optional[Sequence[str]] = None,
                 write: bool = True) -> Dict[str, Any]:
    """Run all oracles + probes on every generated test split and report.

    Writes ``<out>/reference_probes.json`` (per-dataset perfect / memoryless /
    chance probe dicts) so the report can quote per-probe sanity bounds.
    """
    out_dir = Path(out_dir)
    if tiny:
        out_dir = out_dir / "tiny"
    summary = read_json(out_dir / "summary.json")
    out: Dict[str, Any] = {}
    for ds in summary["datasets"]:
        man = Manifest.load(ds["manifest"])
        task = man.extra["task_id"]
        if tasks and task not in tasks:
            continue
        test = [r for r in man.seqs if r.split == "test"] or list(man.seqs)
        test = sorted(test, key=lambda r: r.seq_id)[:REFERENCE_SEQS_CAP]
        truths = [load_seq(r.path) for r in test]
        rep = P.oracle_probe_report(truths, task, seed=0)
        out[ds["dataset"]] = {"task": task,
                              "main_metric": man.extra["main_metric"],
                              "manifest": ds["manifest"],
                              "gap_s": man.extra.get("gap_s"),
                              "capacity_k": man.extra.get("capacity_k"),
                              "n_seqs": len(test),
                              "report": rep}
    if write:
        write_json(out_dir / "reference_probes.json", out)
    return out
