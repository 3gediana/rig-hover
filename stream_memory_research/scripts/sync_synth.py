"""Copy synthetic-suite manifests into data_manifests/ (the contract location).

synth-lab writes the whole suite under G:\\stream_memory_data\\synth\\<suite>\\:
    manifests/<dataset>.json , <dataset>/<seq>.npz , summary.json, index.json
The training/eval contract expects manifests at data_manifests/<dataset>.json,
so this script mirrors them and adds an `eval_index.json` listing the eval-only
sweeps (gap sweeps, capacity scans, shuffled controls, irregular variants) so a
trained checkpoint can be pointed at them without touching the trainer.

Usage:
    python scripts/sync_synth.py --suite tiny      # tiny smoke fixture
    python scripts/sync_synth.py --suite full
    python scripts/sync_synth.py --suite full --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402

SYNTH_ROOT = contract.SYNTH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="full", help="subdir of G:\\stream_memory_data\\synth")
    ap.add_argument("--all", action="store_true", help="sync every suite subdir found")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        roots = [d for d in sorted(SYNTH_ROOT.iterdir()) if (d / "manifests").is_dir()]
    else:
        roots = [SYNTH_ROOT / args.suite if args.suite != "full" else SYNTH_ROOT]
        roots = [r for r in roots if (r / "manifests").is_dir()]
    if not roots:
        print(f"no suite with manifests/ found under {SYNTH_ROOT}")
        return

    eval_index = {}
    n_copied = 0
    for r in roots:
        print(f"[suite] {r}")
        for man in sorted((r / "manifests").glob("*.json")):
            d = contract.read_json(man)
            name = d.get("dataset") or man.stem
            extra = d.get("extra", {}) or {}
            eval_only = bool(extra.get("eval_only"))
            dst = contract.MANIFESTS / f"{name}.json"
            if not args.dry_run:
                shutil.copyfile(man, dst)
            n_copied += 1
            counts = {}
            for s in d["seqs"]:
                counts[s["split"]] = counts.get(s["split"], 0) + 1
            eval_index[name] = {
                "manifest": str(dst).replace("\\", "/"),
                "source": str(man).replace("\\", "/"),
                "task": d.get("task", ""),
                "eval_only": eval_only,
                "base_dataset": extra.get("base_dataset"),
                "task_id": extra.get("task_id"),
                "main_metric": extra.get("main_metric", "cos"),
                "counts": counts,
                "n_seqs": len(d["seqs"]),
                "feature_dim": d.get("feature_dim"),
                "tok_dim": d.get("tok_dim"),
            }
            flag = " [eval-only]" if eval_only else ""
            print(f"   {name:42s} {str(counts):34s} Dg={d.get('feature_dim')} "
                  f"tok={d.get('tok_dim')}{flag}")

    if not args.dry_run:
        contract.write_json(contract.MANIFESTS / "_synth_eval_index.json", eval_index)
        print(f"\ncopied {n_copied} manifests -> {contract.MANIFESTS}")
        print(f"wrote {contract.MANIFESTS / '_synth_eval_index.json'} "
              f"({sum(1 for v in eval_index.values() if v['eval_only'])} eval-only sweeps)")

    # sanity: every non-eval-only manifest must load and have disjoint splits
    if not args.dry_run:
        bad = 0
        for name, info in eval_index.items():
            try:
                m = contract.Manifest.load(info["manifest"])
                m.check_split_disjoint()
                if info["task"] != "synth_mem" and not info["eval_only"]:
                    pass
            except Exception as e:
                bad += 1
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        print(f"loaded+checked {len(eval_index) - bad}/{len(eval_index)} manifests, {bad} failures")


if __name__ == "__main__":
    main()
