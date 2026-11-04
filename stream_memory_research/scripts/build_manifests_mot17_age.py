"""Derived view: add per-slot `lab_age` to the MOT17 tracking task.

Motivation (same reasoning the Lead applied to DAVIS in ``fix_davis_task.py``):
the tracking task is trivial while a slot's ROI is visible, so the measurement
that matters is **box error as a function of how long the object has been
unobserved**.  ``lab_age [T,N] float32`` = seconds since ``roi_valid`` was last
1 for that slot (0 on visible steps, ``t[i] - t[0]`` for a slot never yet seen).

This does NOT modify the delivered `mot17_track` npz files -- it re-emits them
into ``G:\\stream_memory_data\\features\\mot17_track_age`` as a separate dataset
(90 % of the bytes are already f16 features, so it is a straight copy plus one
extra array).  Use `mot17_track_age.json` when you want per-gap curves; keep
using `mot17_track.json` otherwise.

    python scripts/build_manifests_mot17_age.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SRC_NAME = "mot17_track"
DST_NAME = "mot17_track_age"
OUT = contract.FEATURES / DST_NAME


def slot_age(t: np.ndarray, roi_valid: np.ndarray) -> np.ndarray:
    """[T,N] seconds since each slot's ROI was last observed."""
    T, N = roi_valid.shape
    age = np.zeros((T, N), np.float32)
    last = np.full(N, -1, np.int64)
    for i in range(T):
        for j in range(N):
            if roi_valid[i, j]:
                last[j] = i
                age[i, j] = 0.0
            else:
                age[i, j] = float(t[i] - t[last[j]]) if last[j] >= 0 \
                    else float(t[i] - t[0])
    return age


def main() -> int:
    src = contract.MANIFESTS / f"{SRC_NAME}.json"
    man = Manifest.load(src)
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = []
    for rec in man.seqs:
        sp = Path(rec.path)
        with np.load(sp, allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files}
        t = np.asarray(arrays["t"], dtype=np.float64).astype(np.float32)
        rv = np.asarray(arrays["roi_valid"]).astype(np.uint8)
        arrays["lab_age"] = slot_age(t, rv.astype(bool)).astype(np.float32)
        dst = OUT / sp.name
        np.savez(dst, **arrays)
        rec2 = SeqRecord(seq_id=rec.seq_id, source_id=rec.source_id,
                         split=rec.split, T=rec.T, path=str(dst),
                         labels=list(rec.labels) + ["age"], meta=dict(rec.meta))
        rec2.meta["age_semantics"] = ("seconds since this slot's roi_valid was last 1; "
                                      "0 on visible steps")
        seqs.append(rec2)

    specs = dict(man.label_specs)
    specs["lab_age"] = {"kind": "reg", "dim": 1,
                        "note": "not a training target by default -- grouping key "
                                "for per-gap box-error evaluation"}
    newman = Manifest(dataset=DST_NAME, task=man.task,
                      feature_dim=man.feature_dim, seqs=seqs,
                      tok_dim=man.tok_dim, roi_dim=man.roi_dim, n_roi=man.n_roi,
                      encoder=man.encoder, split_key=man.split_key,
                      label_specs=specs, extra=dict(man.extra))
    newman.extra["derived_from"] = SRC_NAME
    newman.extra["added_lab_age"] = (
        "lab_age [T,N] float32, seconds since that ROI slot was last observed. "
        "Bucketing box error by lab_age gives 'bbox error vs occlusion length' "
        "for the multi-object case, analogous to davis_vos2.")
    newman.check_split_disjoint()
    p = newman.save(contract.MANIFESTS / f"{DST_NAME}.json")
    print(f"[age] wrote {p}  counts={newman.counts()}")
    # quick sanity print
    for rec in seqs:
        with np.load(rec.path, allow_pickle=False) as z:
            a, rv = z["lab_age"], z["roi_valid"]
        print(f"  {rec.seq_id:<16} age max={a.max():5.2f}s  "
              f"mean_age_when_invalid={a[rv == 0].mean() if (rv == 0).any() else 0:5.2f}s"
              f"  invalid={100.0 * (rv == 0).mean():4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
