# -*- coding: utf-8 -*-
"""Quick inspection of recorded real-robot npz files.

Prints every key, shape and dtype so we know exactly what is already stored
and whether raw frames are available for offline re-processing / VLM labelling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DIRS = [
    r"D:\robot_work\pc\train\data_real",
    r"D:\robot_work\pc\train\data_realtest",
]


def show(path: Path) -> None:
    z = np.load(path, allow_pickle=True)
    print("=" * 78)
    print(path.name, "  bytes =", path.stat().st_size)
    print("-" * 78)
    for k in sorted(z.files):
        a = z[k]
        try:
            shape = a.shape
            dtype = a.dtype
        except Exception:  # noqa: BLE001 - object arrays
            shape, dtype = "?", "?"
        extra = ""
        if a.dtype != object and a.size and np.issubdtype(a.dtype, np.number):
            flat = np.asarray(a, dtype=np.float64).ravel()
            if flat.size:
                extra = "min=%.4g max=%.4g mean=%.4g" % (
                    np.nanmin(flat), np.nanmax(flat), np.nanmean(flat))
        print("  %-28s %-16s %-10s %s" % (k, str(shape), str(dtype), extra))


def main() -> int:
    for d in DIRS:
        p = Path(d)
        if not p.is_dir():
            print("missing:", p)
            continue
        for f in sorted(p.glob("*.npz"))[:2]:
            show(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
