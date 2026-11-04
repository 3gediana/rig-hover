"""Build a REAL-VIDEO temporal-order task from cached UCF101 features (no GPU needed).

Why
---
Recognition of "which way time runs" is the cheapest honest test that a model uses
*temporal order* rather than per-frame appearance: the two classes contain the
SAME frames in a different order, so any model that pools frames (or only looks at
the current frame) is at chance by construction.  It is a standard benchmark task
(arrow-of-time / temporal order verification) and it is exactly the capability our
robot scenario needs ("two knocks" vs "one knock", "push then pull").

Construction
------------
For every UCF101 sequence we emit two sequences from the SAME cached features:
  * label 0 -> frames in their original order
  * label 1 -> frames reversed in time (timestamps stay increasing)
`tok` is reversed the same way.  Everything else (features, timestamps, split,
source_id) is unchanged, so the split stays leakage-free by construction: both
order variants of a clip live in the SAME split.

Because the label is constant per sequence and the only difference is order, the
causal online prefix curve is meaningful: a temporal model must wait long enough
to see the asymmetry.

Output: data_manifests/ucf101_order.json + G:/stream_memory_data/features/ucf101_order/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SRC_NAME = "ucf101_action"
DST_NAME = "ucf101_order"
OUT = contract.FEATURES / DST_NAME


def main():
    man = Manifest.load(contract.MANIFESTS / f"{SRC_NAME}.json")
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = []
    # subsample deterministically: keep every 2nd clip (both order variants of a clip
    # stay in the same split, so this cannot leak).  48 frames is plenty for order.
    keep = sorted(man.seqs, key=lambda r: r.seq_id)[::2]
    NF = 48
    for r in keep:
        with np.load(r.path, allow_pickle=False) as z:
            feat = np.asarray(z["feat"])[:NF]
            tok = np.asarray(z["tok"])[:NF] if "tok" in z.files else None
            t = np.asarray(z["t"], dtype=np.float32)[:NF]
        for rev in (0, 1):
            tag = "rev" if rev else "fwd"
            seq_id = f"{r.seq_id}__{tag}"
            dst = OUT / f"{seq_id}.npz"
            arrays = {
                "feat": feat[::-1] if rev else feat,
                "t": t.copy(),                       # timestamps still increase
                "lab_order": np.full(t.shape[0], rev, dtype=np.int64),
            }
            if tok is not None:
                arrays["tok"] = tok[::-1] if rev else tok
            np.savez_compressed(dst, **arrays)
            seqs.append(SeqRecord(
                seq_id=seq_id, source_id=r.source_id, split=r.split, T=int(t.shape[0]),
                path=str(dst), labels=["order"],
                meta={**r.meta, "order_label": rev, "origin": r.seq_id},
            ))
    newman = Manifest(
        dataset=DST_NAME, task="order", feature_dim=int(man.feature_dim),
        tok_dim=int(man.tok_dim), seqs=seqs, encoder=man.encoder,
        label_specs={"order": {"kind": "ce", "ignore": -1, "dim": 2, "n_classes": 2}},
        extra={
            "note": ("arrow-of-time task: two sequences per clip, original vs reversed "
                     "frame order. Same frames, same split, same source_id -> a model "
                     "that ignores temporal order is at chance by construction."),
            "classes": ["forward", "reversed"],
            "source_dataset": SRC_NAME,
            "construction": "feat and tok reversed in time; t unchanged (increasing); 48 frames; every 2nd clip",
            "main_metric": "acc",
            "eval_manifests": [],
        },
    )
    p = newman.save(contract.MANIFESTS / f"{DST_NAME}.json")
    newman.check_split_disjoint()
    print(f"wrote {p}")
    print("counts:", newman.counts(), "n_seqs:", len(seqs))


if __name__ == "__main__":
    main()
