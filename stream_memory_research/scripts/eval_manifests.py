"""Evaluate a trained checkpoint on ANY manifest (cross-task / time-rescale tests).

The trainer only evaluates the manifest it trained on.  The scientifically
interesting questions are cross-manifest:
  * train at 1x speed -> test at 2x / 4x speed (does the model carry *physical*
    time, or does it just count samples?)
  * temporal-shuffle controls
  * capacity / gap sweeps

This script rebuilds the model from runs/<run_id>/config.json, loads best.pt and
scores it on one or more manifests with batched forward passes.

Usage
-----
  python scripts/eval_manifests.py --run synth_pattern_rate__dsm__s0 \
      --match synth_pattern_rate
  python scripts/eval_manifests.py --all --match synth_pattern --limit 40
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, collate_stream, move_batch  # noqa: E402


def load_rows(recs):
    rows = []
    for r in recs:
        with np.load(r.path, allow_pickle=False) as z:
            row = {"feat": torch.from_numpy(np.asarray(z["feat"], dtype=np.float32)),
                   "t": torch.from_numpy(np.asarray(z["t"], dtype=np.float32)),
                   "labels": {}, "seq_id": r.seq_id, "source_id": r.source_id,
                   "split": r.split}
            for k in ("tok", "roi", "roi_meta"):
                if k in z.files:
                    row[k] = torch.from_numpy(np.asarray(z[k], dtype=np.float32))
            if "roi_valid" in z.files:
                row["roi_valid"] = torch.from_numpy(np.asarray(z["roi_valid"], dtype=np.bool_))
            for k in z.files:
                if k.startswith("lab_"):
                    v = np.asarray(z[k])
                    name = k[4:]
                    row["labels"][name] = torch.from_numpy(
                        v.astype(np.int64) if np.issubdtype(v.dtype, np.integer)
                        else v.astype(np.float32))
        rows.append((r, row))
    return rows


@torch.no_grad()
def score(model, man: Manifest, split="test", batch=64, device="cuda") -> Dict[str, float]:
    recs = sorted([r for r in man.seqs if r.split == split], key=lambda r: r.seq_id)
    if not recs:
        return {}
    heads = getattr(model, "head_kinds", {})
    accs: Dict[str, List[float]] = {}
    maes: Dict[str, List[float]] = {}
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        pairs = load_rows(chunk)
        b = move_batch(collate_stream([p[1] for p in pairs]), device)
        res = model(b)
        logits = res[0] if isinstance(res, (tuple, list)) else res
        for j, (rec, row) in enumerate(pairs):
            n = int(row["feat"].shape[0])
            for name, lg in logits.items():
                a = lg[j, :n].detach().float().cpu().numpy()
                kind = heads.get(name, "ce")
                lab = row["labels"].get(name)
                if lab is None:
                    continue
                lab = lab[:n]
                if kind == "ce":
                    tgt = lab.reshape(n, -1)[:, 0] if lab.dim() > 1 else lab
                    ok = tgt.numpy() != -1
                    if ok.sum():
                        pred = a.argmax(-1)
                        accs.setdefault(name, []).append(
                            float((pred[ok] == tgt.numpy()[ok]).mean()))
                elif kind == "mse":
                    t = lab.numpy().reshape(n, -1)
                    tgt = t
                    m = np.isfinite(t)
                    if m.sum():
                        ae = np.abs(a.reshape(n, -1) - np.where(m, t, 0.0))
                        maes.setdefault(name, []).append(float(ae[m].mean()))
                else:  # bce
                    t = lab.numpy().reshape(n, -1)
                    p = 1.0 / (1.0 + np.exp(-a.reshape(n, -1)))
                    accs.setdefault(name, []).append(
                        float(((p > 0.5) == (t > 0.5)).mean()))
    out = {}
    for k, v in accs.items():
        out[f"{k}_acc"] = float(np.mean(v))
    for k, v in maes.items():
        out[f"{k}_mae"] = float(np.mean(v))
    out["n_seqs"] = len(recs)
    return out


#: 2026-11-04: the per-run checkpoints were moved off D: to free ~4.3 GB
#: (D: is the small system disk).  The run directories keep config.json /
#: metrics.json / curves.json, so only the weight files live on the mirror.
CKPT_MIRROR = Path(r"G:\stream_memory_data\research_checkpoints")


def best_ckpt(run_id: str) -> Path:
    """Path to a run's best.pt, following the checkpoint offload to G:."""
    local = ROOT / "runs" / run_id / "checkpoints" / "best.pt"
    if local.exists():
        return local
    return CKPT_MIRROR / run_id / "checkpoints" / "best.pt"


def build(run_id: str, device="cuda"):
    from src.train import factory
    run_dir = ROOT / "runs" / run_id
    cfg = contract.read_json(run_dir / "config.json")
    man = Manifest.load(ROOT / cfg["manifest"])
    cfg["_feature_dim"] = int(man.feature_dim)
    cfg["_tok_dim"] = int(man.tok_dim or 0)
    cfg["_n_tok"] = int((man.extra.get("layout") or {}).get("KMAX", 16)) if man.tok_dim else 0
    tshape = (int(cfg["_n_tok"]), int(cfg["_tok_dim"])) if cfg["_tok_dim"] else None
    model = factory.build_model(cfg, int(man.feature_dim), tok_shape=tshape)
    ck = torch.load(best_ckpt(run_id), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    model.head_kinds = {k: str(v.get("kind", "ce")) for k, v in (cfg.get("heads") or {}).items()}
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--match", default=None, help="substring filter on manifest file names")
    ap.add_argument("--datasets", default=None, help="comma filter on the run's dataset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.run:
        run_ids = [args.run]
    else:
        run_ids = []
        for p in sorted(glob.glob("runs/*/metrics.json")):
            try:
                d = contract.read_json(p)
            except Exception:
                continue
            if d.get("status") != "complete":
                continue
            rid = d["run_id"]
            if rid.endswith("__smoke") or rid.startswith("fixture"):
                continue
            if args.datasets and d.get("dataset") not in args.datasets.split(","):
                continue
            run_ids.append(rid)

    mans = sorted(glob.glob(str(ROOT / "data_manifests" / "*.json")))
    if args.match:
        mans = [m for m in mans if args.match in Path(m).name]
    mans = [Path(m) for m in mans if not Path(m).name.startswith("_")]

    out = {}
    for k, rid in enumerate(run_ids, 1):
        if not best_ckpt(rid).exists():
            continue
        try:
            model, cfg = build(rid, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}] {rid}: build failed {type(e).__name__}: {e}")
            continue
        out[rid] = {}
        for mp in mans:
            try:
                man = Manifest.load(mp)
            except Exception:
                continue
            if int(man.feature_dim) != int(cfg["_feature_dim"]):
                continue
            try:
                r = score(model, man, device=args.device)
            except Exception as e:  # noqa: BLE001
                r = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            if r:
                out[rid][Path(mp).stem] = r
        got = {k2: round(v, 4) for k2, v in list(out[rid].get(Path(mans[0]).stem, {}).items())} if mans else {}
        print(f"[{k}/{len(run_ids)}] {rid}: {len(out[rid])} manifests", flush=True)
        contract.write_json(ROOT / "runs" / rid / "cross_eval.json", out[rid])
    if args.out:
        contract.write_json(args.out, out)
    print("done")


if __name__ == "__main__":
    main()
