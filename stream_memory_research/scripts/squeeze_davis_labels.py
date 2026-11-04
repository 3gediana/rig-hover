"""Squeeze DAVIS labels from [T,1,4]/[T,1] to [T,4]/[T] so the shared loss accepts them.

The extraction kept a singleton ROI axis (`lab_track [T,1,4]`, `lab_present [T,1]`)
because the tracker is written for N ROIs.  With N=1 for DAVIS that axis is pure
overhead and the generic MSE/BCE losses see a rank mismatch, so re-emit the label
arrays squeezed.  Features/timestamps are hard-linked by copying the arrays
unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SRC = contract.MANIFESTS / "davis_vos2.json"
DST = "davis_vos3"
OUT = contract.FEATURES / DST


def main():
    man = Manifest.load(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = []
    for r in man.seqs:
        with np.load(r.path, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        if arrays["lab_track"].ndim == 3:
            arrays["lab_track"] = arrays["lab_track"].reshape(arrays["lab_track"].shape[0], -1)
        if arrays.get("lab_present") is not None and arrays["lab_present"].ndim == 2:
            arrays["lab_present"] = arrays["lab_present"].reshape(-1)
        if arrays.get("lab_age") is not None and arrays["lab_age"].ndim == 2:
            arrays["lab_age"] = arrays["lab_age"].reshape(-1)
        dst = OUT / Path(r.path).name
        np.savez_compressed(dst, **arrays)
        seqs.append(SeqRecord(seq_id=r.seq_id, source_id=r.source_id, split=r.split,
                              T=r.T, path=str(dst), labels=list(r.labels), meta=dict(r.meta)))
    new = Manifest(dataset=DST, task="track", feature_dim=man.feature_dim,
                   tok_dim=man.tok_dim, roi_dim=man.roi_dim, n_roi=man.n_roi,
                   encoder=man.encoder,
                   label_specs={"track": {"kind": "mse", "dim": 4}},
                   extra=dict(man.extra), seqs=seqs)
    new.extra["labels_squeezed"] = ("lab_track [T,1,4]->[T,4], lab_present/lab_age [T,1]->[T] "
                                    "so the shared generic losses accept them; features unchanged")
    p = new.save(contract.MANIFESTS / f"{DST}.json")
    new.check_split_disjoint()
    print("wrote", p, new.counts())
    z = np.load(seqs[0].path)
    for k in z.files:
        print("  ", k, z[k].shape, z[k].dtype)


if __name__ == "__main__":
    main()
