"""Run the synthetic-suite probes on a trained checkpoint.

This is the bridge that turns a training run into evidence for the scientific
questions: recall-vs-gap (H1), crosstalk matrix (H3/H4), capacity curve (H7),
update latency / false update rate (H5), event report (H12), rollout (H5b).

It reuses the checkpoint of a trained model but evaluates it on ARBITRARY synth
manifests (the eval-only sweeps), which is how gap/capacity/permutation curves are
produced without retraining.

Usage
-----
  # base test split of the run's own dataset
  python scripts/probe_synth.py --run synth_recall_single__dsm__s0

  # the same checkpoint on an eval-only sweep
  python scripts/probe_synth.py --run synth_recall_single__dsm__s0 \
      --manifest data_manifests/synth_recall_single_gap3.2s.json

  # several eval manifests at once + aggregated comparison later
  python scripts/probe_synth.py --run <id> --eval-index data_manifests/_synth_eval_index.json \
      --match synth_recall_single --out-tag gapsweep

Output: runs/<run_id>/probes_<tag>.json  (+ merged into metrics.json best.probes_<tag>)
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest  # noqa: E402


# --------------------------------------------------------------------------- #
def load_seq_raw(path: str) -> dict:
    """Load one npz into the raw dict that `predict_stream` expects."""
    with np.load(path, allow_pickle=False) as z:
        seq = {"feat": np.asarray(z["feat"], dtype=np.float32),
               "t": np.asarray(z["t"], dtype=np.float32),
               "seq_id": Path(path).stem}
        for k in ("tok", "roi", "roi_valid", "roi_meta"):
            if k in z.files:
                seq[k] = np.asarray(z[k])
    return seq


def build_model_from_run(run_id: str, device: str = "cuda"):
    from src.train import factory
    run_dir = ROOT / "runs" / run_id
    cfg = contract.read_json(run_dir / "config.json")
    man = Manifest.load(ROOT / cfg["manifest"])
    cfg["_feature_dim"] = int(man.feature_dim)
    cfg["_tok_dim"] = int(man.tok_dim or 0)
    cfg["_n_tok"] = int(man.extra.get("layout", {}).get("KMAX", 0)) if man.tok_dim else 0
    tshape = (int(cfg["_n_tok"]), int(cfg["_tok_dim"])) if cfg.get("_tok_dim") else None
    model = factory.build_model(cfg, int(man.feature_dim), tok_shape=tshape)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu",
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg, man, ck


# --------------------------------------------------------------------------- #
def probe_checkpoint(run_id: str, manifest_path: str, split: str = "test",
                     device: str = "cuda", limit: int = 0, verbose: bool = True) -> dict:
    from src.eval.predict import StreamPredictor
    from src.synth import probes as SP

    model, cfg, base_man, ck = build_model_from_run(run_id, device)
    man = Manifest.load(manifest_path)
    man.check_split_disjoint()
    recs = [r for r in man.seqs if r.split == split]
    if not recs:
        raise SystemExit(f"no '{split}' sequences in {manifest_path}")
    if limit:
        recs = recs[:limit]

    task_id = (man.extra or {}).get("task_id") or (base_man.extra or {}).get("task_id") or ""

    pred = StreamPredictor(model, cfg, device=device)
    preds: List[Dict[str, np.ndarray]] = []
    t0 = time.time()
    for i, r in enumerate(recs):
        seq = load_seq_raw(r.path)
        out = pred.run_sequence(seq)
        preds.append({k: np.asarray(v, dtype=np.float32) for k, v in out["pred"].items()})
        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(recs)}] {time.time() - t0:.1f}s", flush=True)
    truths = SP.load_truths([r.path for r in recs])
    seqs = SP.make_probe_seqs(truths, preds)
    res = SP.probe_all(task_id, seqs)
    res["_meta"] = {
        "run_id": run_id, "manifest": str(manifest_path), "split": split,
        "task_id": task_id, "n_seqs": len(recs),
        "eval_sec": round(time.time() - t0, 2),
        "params": int(sum(p.numel() for p in model.parameters())),
        "main_metric": (man.extra or {}).get("main_metric"),
        "gap_s": (man.extra or {}).get("gap_s"),
    }
    return res


# --------------------------------------------------------------------------- #
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_id (dir under runs/)")
    ap.add_argument("--manifest", default=None, help="evaluate on this manifest instead")
    ap.add_argument("--eval-index", default=None,
                    help="use data_manifests/_synth_eval_index.json, filter with --match")
    ap.add_argument("--match", default=None, help="substring filter on dataset names")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / args.run
    if not (run_dir / "checkpoints" / "best.pt").exists():
        raise SystemExit(f"no best checkpoint in {run_dir}")

    jobs: List[tuple] = []          # (manifest_path, tag)
    if args.eval_index:
        idx = contract.read_json(ROOT / args.eval_index)
        for name, info in idx.items():
            if args.match and args.match not in name:
                continue
            jobs.append((ROOT / info["manifest"], name))
    elif args.manifest:
        p = Path(args.manifest)
        p = p if p.is_absolute() else ROOT / p
        jobs.append((p, Path(args.manifest).stem))
    else:
        cfg = contract.read_json(run_dir / "config.json")
        jobs.append((ROOT / cfg["manifest"], Path(cfg["manifest"]).stem))

    all_out = {}
    for man_path, tag in jobs:
        tag = args.out_tag or tag
        print(f"[probe] {args.run} on {Path(man_path).name}", flush=True)
        res = probe_checkpoint(args.run, str(man_path), split=args.split,
                               device=args.device, limit=args.limit)
        res = jsonable(res)
        outp = run_dir / f"probes_{tag}.json"
        contract.write_json(outp, res)
        m = res.get("main", {})
        print(f"   -> {outp.name}  main={m}  "
              f"offdiag={res.get('crosstalk_offdiag_mean')}  "
              f"n={res['_meta']['n_seqs']} {res['_meta']['eval_sec']}s", flush=True)
        all_out[tag] = res

    if len(all_out) > 1:
        contract.write_json(run_dir / f"probes_{args.out_tag or 'all'}_merged.json", all_out)
        # compact curve summary across the sweeps
        rows = []
        for tag, r in all_out.items():
            rows.append({
                "tag": tag,
                "gap_s": r["_meta"].get("gap_s"),
                "main": r.get("main", {}),
                "offdiag": r.get("crosstalk_offdiag_mean"),
                "isolation": r.get("crosstalk_isolation"),
            })
        print("\n== sweep summary ==")
        for r in rows:
            print(f"  {r['tag']:42s} gap={r['gap_s']}  {r['main']}  offdiag={r['offdiag']}")


if __name__ == "__main__":
    main()
