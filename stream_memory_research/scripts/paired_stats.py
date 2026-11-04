"""Sequence-level paired bootstrap: is model A really better than model B?

The goal's acceptance criterion is "the 95% CI of the advantage must not cross 0".
With only 2-3 seeds a seed-level CI is useless, but every run stores
`best.per_seq_test[metric] = {seq_id: value}` with ~1200 sequences, so we can pair
A and B **on the same test sequences** and bootstrap over sequences.  That is the
statistically meaningful test.

Usage
-----
  python scripts/paired_stats.py --dataset ucf101_order --ref cfc
  python scripts/paired_stats.py --dataset ucf101_order --ref gru
  python scripts/paired_stats.py --all --ref cfc
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.metrics import paired_bootstrap_diff, cohens_d  # noqa: E402


def model_of(run_id: str) -> str:
    parts = run_id.split("__")
    return "__".join(parts[1:-1]) if parts[-1].startswith("s") else "__".join(parts[1:])


def collect(dataset: str) -> Dict[str, List[dict]]:
    """model -> list of per-run dicts {seed, per_seq: {seq:val}, value: float}."""
    out: Dict[str, List[dict]] = defaultdict(list)
    for p in sorted(glob.glob(str(ROOT / "runs" / "*" / "metrics.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("status") != "complete" or d.get("dataset") != dataset:
            continue
        b = d.get("best") or {}
        mm = d.get("main_metric", "acc")
        ps = (b.get("per_seq_test") or {}).get(mm)
        val = (b.get("test") or {}).get(mm)
        if ps is None or val is None:
            continue
        out[model_of(d["run_id"])].append({"seed": d.get("seed"), "per_seq": ps,
                                           "value": float(val), "metric": mm})
    return out


def pair(a: List[dict], b: List[dict], seed: int = 0):
    """Pair run-by-run (same seed first), return pooled diffs over sequences."""
    by_seed_b = {r["seed"]: r for r in b}
    diffs: List[np.ndarray] = []
    n_common = 0
    for ra in a:
        rb = by_seed_b.get(ra["seed"])
        if rb is None:
            continue
        common = sorted(set(ra["per_seq"]) & set(rb["per_seq"]))
        if len(common) < 10:
            continue
        va = np.array([ra["per_seq"][s] for s in common], dtype=np.float64)
        vb = np.array([rb["per_seq"][s] for s in common], dtype=np.float64)
        diffs.append(va - vb)
        n_common = len(common)
    if not diffs:
        return None
    d = np.concatenate(diffs) if len(diffs) > 1 else diffs[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(4000, d.size))
    means = d[idx].mean(1)
    return {
        "diff": float(d.mean()), "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "p": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
        "n_pairs": int(d.size), "seeds_paired": len(diffs),
        "a_mean": float(np.mean([r["value"] for r in a])),
        "b_mean": float(np.mean([r["value"] for r in b])),
        "a_seeds": len(a), "b_seeds": len(b),
        "d": float(cohens_d(d, np.zeros_like(d))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ref", default="cfc")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    datasets = (sorted({json.load(open(p, encoding="utf-8")).get("dataset")
                        for p in glob.glob(str(ROOT / "runs" / "*" / "metrics.json"))
                        if '"complete"' in open(p, encoding="utf-8").read()})
                if args.all else [args.dataset])
    result = {}
    for ds in datasets:
        if not ds:
            continue
        data = collect(ds)
        if len(data) < 2:
            continue
        ref = args.ref if args.ref in data else None
        if ref is None:
            # fall back: strongest baseline that exists
            for cand in ("cfc", "gru", "mlp"):
                if cand in data:
                    ref = cand
                    break
        print(f"\n===== {ds}   (reference = {ref}) =====")
        hdr = f"{'model':24s} {'mean':>7s} {'n_seeds':>7s}  | Δ vs {ref}  95% CI            p      n_pairs"
        print(hdr)
        rows = {}
        for m in sorted(data, key=lambda k: -np.mean([r["value"] for r in data[k]])):
            mean = float(np.mean([r["value"] for r in data[m]]))
            if m == ref:
                print(f"{m:24s} {mean:7.4f} {len(data[m]):7d}  | (reference)")
                continue
            r = pair(data[m], data[ref])
            if r is None:
                print(f"{m:24s} {mean:7.4f} {len(data[m]):7d}  | (no paired seeds)")
                continue
            flag = "" if (r["lo"] > 0 or r["hi"] < 0) else "   <- CI crosses 0"
            print(f"{m:24s} {mean:7.4f} {len(data[m]):7d}  | {r['diff']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['p']:.2e} {r['n_pairs']:6d}{flag}")
            rows[m] = r
        result[ds] = {"reference": ref, "rows": rows}

    if args.json_out:
        contract.write_json(args.json_out, result)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
