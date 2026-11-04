"""Fast BATCHED probe pass: run a trained checkpoint over a synthetic eval sweep.

`probe_synth.py` streams one frame at a time (correct but slow); this script does
batched teacher-forced forward passes instead, which is ~100x faster and therefore
lets us build gap / capacity / crosstalk curves for *every* run.

The two agree because the parallel scan is exact (see src/models/tests_models.py:
"forward == step-loop" < 1e-5), and because at evaluation time we are scoring
teacher-forced predictions, not free-running rollouts.

Usage
-----
  python scripts/probe_synth_fast.py --run synth_recall_multi__dsm__s0
  python scripts/probe_synth_fast.py --all --datasets synth_recall_multi,synth_recall_single
  python scripts/probe_synth_fast.py --all --kind sweeps     # all linked eval sweeps
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, collate_stream, move_batch  # noqa: E402


def load_seq_rows(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        row = {"feat": torch.from_numpy(np.asarray(z["feat"], dtype=np.float32)),
               "t": torch.from_numpy(np.asarray(z["t"], dtype=np.float32)),
               "labels": {}}
        for k in ("tok", "roi", "roi_meta"):
            if k in z.files:
                row[k] = torch.from_numpy(np.asarray(z[k], dtype=np.float32))
        if "roi_valid" in z.files:
            row["roi_valid"] = torch.from_numpy(np.asarray(z["roi_valid"], dtype=np.bool_))
    return row


@torch.no_grad()
def predict_manifest(model, man: Manifest, split="test", batch=64, device="cuda"):
    from src.train import factory
    recs = [r for r in man.seqs if r.split == split]
    recs = sorted(recs, key=lambda r: r.seq_id)
    preds: Dict[str, Dict[str, np.ndarray]] = {}
    heads_cfg = factory.head_specs(
        {k: v for k, v in [("_heads_cache", None)]}) if False else None  # unused
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        rows = [load_seq_rows(r.path) for r in chunk]
        lens = [int(x["feat"].shape[0]) for x in rows]
        rows = [{"feat": x["feat"], "t": x["t"], "labels": {},
                 "seq_id": chunk[j].seq_id, "source_id": chunk[j].source_id,
                 "split": chunk[j].split, **{k: x[k] for k in ("tok", "roi", "roi_meta", "roi_valid") if k in x}}
                for j, x in enumerate(rows)]
        b = move_batch(collate_stream(rows), device)
        res = model(b)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        for j, rec in enumerate(chunk):
            n = lens[j]
            out = {}
            for name, lg in logits.items():
                a = lg[j, :n].detach().float().cpu().numpy()
                kind = (model.head_kinds or {}).get(name, "ce") if hasattr(model, "head_kinds") else None
                if kind is None:
                    shp = a.shape
                    kind = "mse" if (len(shp) == 2 and shp[-1] > 16) else "ce"
                if kind == "ce":
                    out[name] = a.argmax(-1)
                elif kind == "bce":
                    out[name] = a.reshape(n, -1)
                    out[name] = (1.0 / (1.0 + np.exp(-out[name])))
                else:
                    out[name] = a.reshape(n, -1)
            preds[rec.seq_id] = out
    return preds


def probe_run(run_id: str, manifests: List[Path], device="cuda", quiet=False):
    from src.synth import probes as SP
    from src.train import factory
    run_dir = ROOT / "runs" / run_id
    cfg = contract.read_json(run_dir / "config.json")
    base_man = Manifest.load(ROOT / cfg["manifest"])
    cfg["_feature_dim"] = int(base_man.feature_dim)
    cfg["_tok_dim"] = int(base_man.tok_dim or 0)
    cfg["_n_tok"] = 16 if base_man.tok_dim else 0
    tshape = (int(cfg["_n_tok"]), int(cfg["_tok_dim"])) if cfg.get("_tok_dim") else None
    model = factory.build_model(cfg, int(base_man.feature_dim), tok_shape=tshape)
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    # head kinds from the run config
    model.head_kinds = {k: str(v.get("kind", "ce")) for k, v in (cfg.get("heads") or {}).items()}

    out = {}
    for mpath in manifests:
        man = Manifest.load(mpath)
        if int(man.feature_dim) != int(base_man.feature_dim):
            continue
        t0 = time.time()
        preds = predict_manifest(model, man, device=device)
        try:
            r = SP.evaluate_manifest(str(mpath), preds, split="test")
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}"}
        r["_meta"] = {"run_id": run_id, "manifest": str(mpath), "n_seqs": len(preds),
                      "sec": round(time.time() - t0, 2)}
        out[Path(mpath).stem] = r
        m = (r.get("main") or {})
        if not quiet:
            print(f"   {Path(mpath).stem:44s} {m}  ({r['_meta']['sec']}s)", flush=True)
    contract.write_json(run_dir / "probes_all.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--datasets", default=None, help="comma filter on dataset name")
    ap.add_argument("--kind", default="base+sweeps",
                    choices=["base", "sweeps", "base+sweeps", "shuffled"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-runs", type=int, default=0)
    args = ap.parse_args()

    if args.run:
        run_ids = [args.run]
    else:
        run_ids = []
        for p in sorted(glob.glob("runs/*/metrics.json")):
            d = contract.read_json(p)
            if d.get("status") != "complete":
                continue
            ds = d.get("dataset", "")
            if args.datasets and ds not in args.datasets.split(","):
                continue
            run_ids.append(d["run_id"])
        if args.limit_runs:
            run_ids = run_ids[:args.limit_runs]
    print(f"[probe-fast] {len(run_ids)} runs")
    for k, rid in enumerate(run_ids, 1):
        run_dir = ROOT / "runs" / rid
        if not (run_dir / "checkpoints" / "best.pt").exists():
            print(f"[{k}] {rid}: no checkpoint, skip")
            continue
        cfg = contract.read_json(run_dir / "config.json")
        base = ROOT / cfg["manifest"]
        man = Manifest.load(base)
        mpaths: List[Path] = []
        kind = args.kind
        if kind in ("base", "base+sweeps"):
            mpaths.append(base)
        evs = (man.extra or {}).get("eval_manifests") or []
        if kind in ("sweeps", "base+sweeps"):
            for e in evs:
                name = Path(str(e)).stem
                p = ROOT / "data_manifests" / f"{name}.json"
                if p.exists():
                    mpaths.append(p)
        if kind == "shuffled":
            for e in evs:
                if "shuffled" in str(e):
                    p = ROOT / "data_manifests" / f"{Path(str(e)).stem}.json"
                    if p.exists():
                        mpaths.append(p)
        if not mpaths:
            print(f"[{k}/{len(run_ids)}] {rid}: no manifests")
            continue
        print(f"[{k}/{len(run_ids)}] {rid}  ({len(mpaths)} manifests)", flush=True)
        try:
            probe_run(rid, mpaths, device=args.device)
        except Exception as e:  # noqa: BLE001
            print(f"   FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("[probe-fast] done")


if __name__ == "__main__":
    main()
