"""Aggregate the synthetic probes into the answers for the 10 scientific questions.

Reads runs/*/probes_all.json (written by scripts/probe_synth_fast.py) and prints:
  * Q3 crosstalk matrix summary (diag vs off-diagonal)
  * Q7 effective capacity curve (recall vs K)
  * gap curve (recall vs occlusion duration)
  * shuffled-control drop
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = r"D:\robot_work\stream_memory_research"


def model_of(rid: str) -> str:
    parts = rid.split("__")
    return "__".join(parts[1:-1]) if parts[-1].startswith("s") else "__".join(parts[1:])


def main():
    cap = defaultdict(dict)
    gap = defaultdict(dict)
    cross = {}
    shuffle = defaultdict(dict)
    for p in glob.glob(os.path.join(ROOT, "runs", "*", "probes_all.json")):
        rid = os.path.basename(os.path.dirname(p))
        m = model_of(rid)
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for man, r in d.items():
            mt = r.get("metrics") or {}
            main = mt.get("main") or {}
            v = main.get("cos")
            if v is None:
                v = mt.get("cos")
            if "_K" in man:
                try:
                    k = int(man.split("_K")[-1])
                except ValueError:
                    continue
                if v is not None:
                    cap[m].setdefault(k, []).append(float(v))
            elif "_gap" in man:
                g = r.get("gap_s")
                if g is not None and v is not None:
                    gap[m].setdefault(float(g), []).append(float(v))
            elif man.endswith("_shuffled"):
                if v is not None:
                    shuffle[m].setdefault("shuf", []).append(float(v))
            else:
                if v is not None:
                    shuffle[m].setdefault("base", []).append(float(v))
            if man == "synth_recall_multi":
                cm = mt.get("crosstalk_matrix")
                if cm:
                    cross[m] = cm

    def order(names):
        pref = ["gru", "lstm", "mamba", "deltanet", "dsm", "dsm_slot", "cfc", "mlp"]
        return sorted(names, key=lambda x: (pref.index(x) if x in pref else 99, x))

    if cap:
        ks = sorted({k for d in cap.values() for k in d})
        print("=== Q7: EFFECTIVE CAPACITY (recall cos vs number of simultaneous objects K) ===")
        print("  %-16s" % "model" + "".join(f"  K{k:<2d}" for k in ks))
        for m in order(set(cap)):
            cells = []
            for k in ks:
                vs = cap[m].get(k)
                cells.append(f"{np.mean(vs):5.2f}" if vs else "    -")
            print("  %-16s" % m + " ".join(cells))
        print()

    if cross:
        print("=== Q3: CROSSTALK (|readout change| when another object is updated) ===")
        for m in order(set(cross)):
            A = np.array([[np.nan if v is None else v for v in row] for row in cross[m]],
                         dtype=float)
            n = A.shape[0]
            off = ~np.eye(n, dtype=bool)
            diag = np.nanmean(np.diag(A))
            od = np.nanmean(A[off])
            iso = 1.0 - od / (diag + 1e-9)
            print(f"  {m:16s} diag={diag:5.3f}  offdiag={od:7.4f}  isolation={iso:5.3f}")
        print()

    if gap:
        gs = sorted({g for d in gap.values() for g in d})
        print("=== gap sweep: recall cos vs occlusion duration (s) ===")
        print("  %-16s" % "model" + "".join(f"  {g:4.1f}s" for g in gs))
        for m in order(set(gap)):
            cells = []
            for g in gs:
                vs = gap[m].get(g)
                cells.append(f"{np.mean(vs):5.2f}" if vs else "    -")
            print("  %-16s" % m + " ".join(cells))
        print()

    if shuffle:
        print("=== shuffled control (real memory must drop) ===")
        for m in order(set(shuffle)):
            b = shuffle[m].get("base")
            s = shuffle[m].get("shuf")
            if b and s:
                print(f"  {m:16s} base={np.mean(b):5.3f}  shuffled={np.mean(s):5.3f}  "
                      f"drop={np.mean(b) - np.mean(s):+5.3f}")


if __name__ == "__main__":
    main()
