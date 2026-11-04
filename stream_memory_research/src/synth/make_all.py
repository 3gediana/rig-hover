"""CLI: build the controlled synthetic memory benchmark.

Examples
--------
    # full suite (~3 min budget on CPU)
    python -m src.synth.make_all --out G:\\stream_memory_data\\synth \
        --config configs/synth_v1.json

    # tiny smoke-test fixture (<= 30 s), written to <out>/tiny
    python -m src.synth.make_all --out G:\\stream_memory_data\\synth --tiny

    # print the perfect / memoryless / chance oracle scores
    python -m src.synth.make_all --out G:\\stream_memory_data\\synth --verify

    # regenerate a subset
    python -m src.synth.make_all --out G:\\stream_memory_data\\synth \
        --only synth_recall_single,synth_recall_multi --overwrite

The command writes, per dataset, ``<out>/<dataset>/*.npz`` +
``<out>/manifests/<dataset>.json`` (a :class:`contract.Manifest`), and finally
``<out>/summary.json`` (per-dataset stats + oracle reference points) and
``<out>/index.json`` (machine readable index for the sweep driver).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_config
from .generate import generate_suite, verify_suite

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "synth_v1.json"


def _print_summary(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 100)
    print(f"synthetic suite: {summary['n_datasets']} datasets / "
          f"{summary['n_sequences_total']} sequences / "
          f"{summary['generation_seconds']}s / out={summary['out_dir']}")
    print("=" * 100)
    print(f"{'dataset':44s} {'task':16s} {'metric':6s} {'n':>5s}  "
          f"{'T':>9s} {'gap_s':>6s}  {'perfect':>8s} {'memless':>8s} {'chance':>8s}")
    for ds in summary["datasets"]:
        ref = ds.get("reference", {})
        def _main(block):
            mm = ds.get("main_metric", "cos")
            if isinstance(block, dict) and mm in block:
                try:
                    return float(block[mm])
                except (TypeError, ValueError):
                    return float("nan")
            return float("nan")
        gap = ds.get("gap_s")
        gap_s = "   -  " if gap is None else f"{float(gap):5.1f} "
        print(f"{ds['dataset']:44s} {ds['task']:16s} {ds['main_metric']:6s} "
              f"{ds['n_seqs']:5d}  {str(ds['T_range']):>9s} {gap_s}  "
              f"{_main(ref.get('perfect', {})):8.3f} "
              f"{_main(ref.get('memoryless', {})):8.3f} "
              f"{_main(ref.get('chance', {})):8.3f}"
              f"{'  [eval-only]' if ds['eval_only'] else ''}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.synth.make_all",
        description="Build the controlled synthetic memory benchmark.")
    ap.add_argument("--out", default=r"G:\stream_memory_data\synth",
                    help="output root (default: G:\\stream_memory_data\\synth)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="config json (default: configs/synth_v1.json)")
    ap.add_argument("--tiny", action="store_true",
                    help="build the tiny smoke-test fixture into <out>/tiny")
    ap.add_argument("--only", default=None,
                    help="comma separated dataset names to (re)generate")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite existing npz files")
    ap.add_argument("--no-references", action="store_true",
                    help="skip the oracle reference points in summary.json")
    ap.add_argument("--verify", action="store_true",
                    help="after building, run all oracles+probes on the test "
                         "splits and print the sanity table")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip generation, only verify an existing suite")
    ap.add_argument("--resummary", action="store_true",
                    help="rebuild summary.json / index.json / summary.md from the "
                         "existing manifests (no npz regeneration, no copy)")
    ap.add_argument("--validate", action="store_true",
                    help="structurally validate the existing suite (manifests + "
                         "npz of --validate-splits) and exit")
    ap.add_argument("--validate-splits", default="test",
                    help="comma separated splits checked by --validate "
                         "(default: test; use 'all' for everything)")
    ap.add_argument("--staging-dir", default=None,
                    help="fast local staging dir; the finished tree is bulk-"
                         "copied to --out at the end (recommended while "
                         "G:\\stream_memory_data is busy). Default: system temp")
    ap.add_argument("--no-stage", action="store_true",
                    help="write the npz files directly into --out")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path) if cfg_path.exists() else None)
    if not cfg_path.exists() and not args.quiet:
        print(f"[make_all] config {cfg_path} not found -> using built-in defaults")

    staging = None if args.no_stage else (
        args.staging_dir or str(Path(tempfile.gettempdir()) / "dsh_synth_stage"))
    if staging:
        if Path(staging).exists():
            shutil.rmtree(staging, ignore_errors=True)

    if args.resummary:
        from .generate import rebuild_summary
        out_dir = Path(args.out) / ("tiny" if args.tiny else "")
        summary = rebuild_summary(out_dir, cfg, tiny=False,
                                  references=not args.no_references,
                                  quiet=args.quiet)
        _print_summary(summary)
        return 0

    if args.validate:
        from .generate import validate_suite
        splits = (None if args.validate_splits.strip() == "all"
                  else [s.strip() for s in args.validate_splits.split(",") if s])
        out_dir = Path(args.out) / ("tiny" if args.tiny else "")
        res = validate_suite(out_dir, splits=splits, quiet=args.quiet)
        print(f"[validate] {res['n_manifests']} manifests, {res['n_npz']} npz "
              f"checked, {res['n_problems']} problems")
        for p in res["problems"][:20]:
            print("  PROBLEM:", p)
        return 0 if res["ok"] else 1

    if not args.verify_only:
        summary = generate_suite(args.out, cfg, tiny=args.tiny, only=args.only,
                                 overwrite=args.overwrite,
                                 references=not args.no_references,
                                 quiet=args.quiet, staging_dir=staging,
                                 append=bool(args.only))
        _print_summary(summary)
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

    if args.verify or args.verify_only:
        out_dir = Path(args.out) / ("tiny" if args.tiny else "")
        rep = verify_suite(out_dir, tiny=False)
        print()
        print("=" * 100)
        print("oracle sanity on generated test splits "
              "(perfect should be ~1.0; memoryless ~chance for memory metrics)")
        print("=" * 100)
        for name, r in rep.items():
            task = r["task"]
            mm = r["main_metric"]
            row = {o: r["report"][o].get("main", {}).get(mm, float("nan"))
                   for o in ("perfect", "memoryless", "chance")}
            extra = ""
            if task == "event_seq":
                pe = r["report"]["perfect"].get("post_event_cos", {})
                me = r["report"]["memoryless"].get("post_event_cos", {})
                extra = (f" post_event_cos(perfect 1x/4x)="
                         f"{pe.get('1x', float('nan')):.2f}/{pe.get('4x', float('nan')):.2f}"
                         f" memless 1x/4x={me.get('1x', float('nan')):.2f}/"
                         f"{me.get('4x', float('nan')):.2f}")
            print(f"{name:44s} {task:16s} {mm:4s}  "
                  f"perfect={row['perfect']:6.3f} "
                  f"memoryless={row['memoryless']:6.3f} "
                  f"chance={row['chance']:6.3f}{extra}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
