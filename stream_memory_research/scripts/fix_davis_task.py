"""Correct the DAVIS task from the extracted features (no re-extraction).

Two problems with `davis_vos.json` as delivered:
  1. splits are official train(60)/val(30); we need train/val/test so that model
     selection never sees the test set.
  2. there is no `lab_age`, so the evaluator cannot bucket the box error by
     "seconds since the object was last observed" -- which IS the occlusion-recovery
     measurement we care about (the task is trivial while the ROI is visible).

This script re-emits the dataset as `davis_vos2`:
  * train = official train minus a deterministic val slice, val = that slice,
    test = official val (never used for selection)
  * adds `lab_age` [T,1] float32 = seconds since roi_valid was last 1 (0 on visible
    steps), so per_gap gives "bbox error vs time since last observation"
  * copies feat/tok/t/roi/roi_valid/roi_meta/lab_track/lab_present unchanged
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SRC = contract.MANIFESTS / "davis_vos.json"
DST_NAME = "davis_vos2"
OUT = contract.FEATURES / DST_NAME


def main():
    man = Manifest.load(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    train = [s for s in man.seqs if s.split == "train"]
    other = [s for s in man.seqs if s.split != "train"]
    train = sorted(train, key=lambda r: r.seq_id)
    # deterministic val slice: every 6th official-train sequence
    val_idx = set(range(5, len(train), 6))
    seqs = []
    for i, r in enumerate(train):
        new_split = "val" if i in val_idx else "train"
        seqs.append(_remap(r, new_split))
    for r in sorted(other, key=lambda r: r.seq_id):
        seqs.append(_remap(r, "test"))
    newman = Manifest(dataset=DST_NAME, task=man.task, feature_dim=man.feature_dim,
                      tok_dim=man.tok_dim, roi_dim=man.roi_dim, n_roi=man.n_roi,
                      encoder=man.encoder, label_specs=dict(man.label_specs),
                      extra=dict(man.extra), seqs=seqs)
    newman.extra["split"] = ("train = official train minus deterministic val slice "
                             "(every 6th sorted seq, offset 5); val = that slice; "
                             "test = official val (never used for model selection)")
    newman.extra["added_lab_age"] = ("seconds since roi_valid was last 1; 0 on visible "
                                     "steps. per_gap then measures bbox error vs time "
                                     "since the object was last observed.")
    p = newman.save(contract.MANIFESTS / f"{DST_NAME}.json")
    newman.check_split_disjoint()
    print(f"wrote {p}")
    print("counts:", newman.counts())


def _remap(rec: SeqRecord, split: str) -> SeqRecord:
    src = Path(rec.path)
    dst = OUT / src.name
    with np.load(src, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    t = np.asarray(arrays["t"], dtype=np.float64)
    rv = np.asarray(arrays["roi_valid"], dtype=np.float32).reshape(-1)
    age = np.zeros(t.shape[0], dtype=np.float32)
    last = None
    for i in range(t.shape[0]):
        if rv[i] > 0.5:
            last = i
            age[i] = 0.0
        else:
            age[i] = (t[i] - t[last]) if last is not None else t[i]
    arrays["lab_age"] = age.reshape(-1, 1).astype(np.float32)
    np.savez_compressed(dst, **arrays)
    return SeqRecord(seq_id=rec.seq_id, source_id=rec.source_id, split=split,
                     T=rec.T, path=str(dst),
                     labels=list(rec.labels) + ["age"], meta=dict(rec.meta))


if __name__ == "__main__":
    main()
