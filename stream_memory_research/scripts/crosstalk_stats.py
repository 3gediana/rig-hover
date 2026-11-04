"""Per-seed crosstalk statistics from runs/*/probes_all.json.

The aggregate `crosstalk_isolation = 1 - offdiag/diag` penalises a model that updates
conservatively (small `diag`) even when its ABSOLUTE disturbance of non-target
objects (`offdiag`) is smaller.  The user's stated dimension-B requirement is
"非目标状态变化幅度显著更小" (the non-target state changes significantly less),
i.e. it is about the ABSOLUTE off-diagonal change, not the ratio.

This script reports, per model, per seed:
    diag    -- mean |change| of the object that WAS updated   (should be large)
    offdiag -- mean |change| of objects that were NOT updated (should be small)
    ratio   -- offdiag/diag (the commonly quoted "isolation" error)
and a seed-level mean±std so we can say whether the absolute cross-talk is
significantly lower than a baseline's.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = r"D:\robot_work\stream_memory_research"


def model_of(rid: str) -> str:
    parts = rid.split("__")
    return "__".join(parts[1:-1]) if parts[-1].startswith("s") else "__".join(parts[1:])


def summarize(A: np.ndarray):
    n = A.shape[0]
    off = ~np.eye(n, dtype=bool)
    diag = np.nanmean(np.diag(A))
    od = np.nanmean(A[off])
    return diag, od, od / (diag + 1e-9)


def main():
    per = defaultdict(lambda: defaultdict(list))   # model -> seed -> [(diag, off, ratio)]
    for p in glob.glob(os.path.join(ROOT, "runs", "*", "probes_all.json")):
        rid = os.path.basename(os.path.dirname(p))
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        cm = None
        for man, r in d.items():
            mt = r.get("metrics") or {}
            if mt.get("crosstalk_matrix"):
                cm = mt["crosstalk_matrix"]
                break
        if not cm:
            continue
        A = np.array([[np.nan if v is None else v for v in row] for row in cm], dtype=float)
        m = model_of(rid)
        seed = rid.split("__")[-1]
        per[m][seed].append(summarize(A))

    if not per:
        print("no crosstalk data; run scripts/probe_synth_fast.py --datasets synth_recall_multi first")
        return

    print(f"{'model':18s} {'n_seeds':>7s} {'diag':>16s} {'offdiag (absolute)':>22s} {'ratio':>16s}")
    rows = []
    for m in sorted(per):
        seeds = per[m]
        ds = [np.mean([t[0] for t in v]) for v in seeds.values()]
        os_ = [np.mean([t[1] for t in v]) for v in seeds.values()]
        rs = [np.mean([t[2] for t in v]) for v in seeds.values()]
        rows.append((m, len(seeds), ds, os_, rs))
    # sort by absolute off-diagonal (the dimension-B quantity), ascending = better
    for m, n, ds, os_, rs in sorted(rows, key=lambda r: np.mean(r[3])):
        print(f"{m:18s} {n:7d} "
              f"{np.mean(ds):8.3f}±{np.std(ds):.3f} "
              f"{np.mean(os_):10.4f}±{np.std(os_):.4f} "
              f"{np.mean(rs):8.3f}±{np.std(rs):.3f}")

    # paired-by-seed comparison of absolute offdiag, ours vs each baseline
    print("\n=== absolute off-diagonal change, ours vs baselines (paired by seed) ===")
    ours = [m for m in per if m.startswith("dsm") or m.startswith("dual")]
    for o in ours:
        for b in ("gru", "lstm", "mamba", "deltanet", "cfc"):
            if b not in per:
                continue
            common = sorted(set(per[o]) & set(per[b]))
            if not common:
                continue
            a = np.array([np.mean([t[1] for t in per[o][s]]) for s in common])
            c = np.array([np.mean([t[1] for t in per[b][s]]) for s in common])
            d = a - c
            print(f"  {o:16s} vs {b:9s}: Δoffdiag={d.mean():+.4f} "
                  f"(seeds: {[round(float(x), 4) for x in d]})  "
                  f"{'LOWER (better)' if d.mean() < 0 else 'higher (worse)'}")


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
