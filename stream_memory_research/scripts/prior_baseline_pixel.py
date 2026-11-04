"""The no-image reference for the pixel event task.

A macro-F1 number means nothing without the score you get **without looking at the
pixels at all**.  With a constant score per frame the only non-trivial predictor
is "always fire" (the threshold is irrelevant because every frame gets the same
score), so:

    all-negative : F1 = 0
    all-positive : F1 = 2p / (1 + p)   per class, p = that class's positive rate

Both are reported.  An earlier version of this script ranked frames by their own
label to pick "the best k frames" and reported 0.78 -- that is an oracle leak
(it reads the answer), and it is exactly the kind of number this project must not
publish, so the ranking is gone.

    python scripts/prior_baseline_pixel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
REP = ROOT / "reports"


def load(name: str):
    idx = json.loads((ROOT / "data_manifests" / f"{name}_index.json")
                     .read_text(encoding="utf-8"))
    out = {}
    for sp in ("train", "val", "test"):
        rows = [s for s in idx["seqs"] if s["split"] == sp]
        Y = np.stack([np.load(s["path"])["event"].astype(np.float32) for s in rows])
        out[sp] = Y.reshape(-1, len(idx["classes"]))
    return idx["classes"], out


def const_f1(Y: np.ndarray, positive: bool) -> float:
    """Macro-F1 of a constant predictor (all-positive or all-negative)."""
    n, C = Y.shape
    f1s = []
    for j in range(C):
        y = Y[:, j] > 0
        tp = float(y.sum()) if positive else 0.0
        fp = float(n - y.sum()) if positive else 0.0
        fn = 0.0 if positive else float(y.sum())
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
    return float(np.mean(f1s))


def main() -> int:
    res = {}
    for name in ("charades_pixel", "charades_pixel10"):
        p = ROOT / "data_manifests" / f"{name}_index.json"
        if not p.exists():
            continue
        classes, d = load(name)
        tr, va, te = d["train"], d["val"], d["test"]
        res[name] = {
            "n_classes": len(classes),
            "train_positive_rate": float(tr.mean()),
            "prior_f1_val_allpos": const_f1(va, True),
            "prior_f1_test_allpos": const_f1(te, True),
            "prior_f1_val_allneg": const_f1(va, False),
            "prior_f1_test_allneg": const_f1(te, False),
            "per_class_test_rate": [float(te[:, j].mean()) for j in range(len(classes))],
        }
        r = res[name]
        print(f"{name}: C={len(classes)} pos_rate={tr.mean() * 100:.2f}%")
        print(f"   all-negative macro-F1  val={r['prior_f1_val_allneg']:.4f} "
              f"test={r['prior_f1_test_allneg']:.4f}")
        print(f"   all-positive macro-F1  val={r['prior_f1_val_allpos']:.4f} "
              f"test={r['prior_f1_test_allpos']:.4f}")
    (REP / "pixel_prior_baseline.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {REP / 'pixel_prior_baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
