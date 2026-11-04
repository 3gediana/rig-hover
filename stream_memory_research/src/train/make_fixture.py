"""Tiny dev fixture: a 20-sequence action-classification dataset.

Kept under ``runs/_fixture/`` on purpose (NOT in data_manifests/, which belongs to
data-video), so the training/evaluation pipeline is testable before the real
feature caches land.

    python -m src.train.make_fixture                 # -> runs/_fixture/
    python -m src.train.make_fixture --corrupt 2     # + 2 truncated npz files

The features are synthetic but *learnable*: a class one-hot plus noise, so a
smoke run must reach above-chance accuracy (i.e. a broken pipeline is visible).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.common.contract import ROOT, Manifest, SeqRecord, write_json


def make_fixture(out_dir=None, n: int = 20, seed: int = 0, feature_dim: int = 32,
                 n_classes: int = 5, corrupt: int = 0, noise: float = 0.6,
                 name: str = "fixture_action") -> Path:
    out = Path(out_dir or (ROOT / "runs" / "_fixture"))
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    splits = (["train"] * n_train + ["val"] * n_val
              + ["test"] * max(1, n - n_train - n_val))
    splits = splits[:n]

    recs = []
    for i, split in enumerate(splits):
        T = int(rng.integers(24, 41))
        cls = int(i % n_classes)
        period = 1.0 / 30.0
        dt = period * (1.0 + 0.05 * rng.standard_normal(T))
        dt = np.maximum(dt, period * 0.2)
        t = np.concatenate([[0.0], np.cumsum(dt[1:])]).astype(np.float32)
        feat = np.zeros((T, feature_dim), dtype=np.float32)
        onehot = np.zeros(feature_dim, dtype=np.float32)
        onehot[cls % feature_dim] = 1.0
        drift = np.linspace(0.0, 0.5, T, dtype=np.float32)[:, None]
        feat = onehot[None, :] + noise * rng.standard_normal((T, feature_dim)).astype(np.float32)
        feat[:, :n_classes] += drift * 0.1
        lab = np.full(T, cls, dtype=np.int64)
        # a few ignore frames at the start (tests ignore semantics end to end)
        if i % 3 == 0:
            lab[:2] = -1
        seq_id = f"{name}_{i:03d}"
        path = feat_dir / f"{seq_id}.npz"
        np.savez_compressed(path, feat=feat.astype(np.float16), t=t, lab_action=lab)
        recs.append(SeqRecord(seq_id=seq_id, source_id=f"src_{i:03d}", split=split,
                              T=T, path=str(path), labels=["action"],
                              meta={"class": cls, "fixture": True}))
        if corrupt and i < corrupt:
            with open(path, "wb") as fcorrupt:
                fcorrupt.write(b"PK\x03\x04 this is not a valid npz")
            recs[-1].meta["corrupt"] = True

    man = Manifest(dataset=name, task="action", feature_dim=feature_dim,
                   seqs=recs, label_specs={"lab_action": {"kind": "ce", "ignore": -1}},
                   extra={"fixture": True, "seed": seed, "n_classes": n_classes})
    mpath = out / "manifest.json"
    man.save(mpath)
    print(f"[fixture] wrote {len(recs)} sequences to {out} "
          f"(counts: {man.counts()}, feature_dim={feature_dim}, classes={n_classes})")
    return mpath


def make_synth_fixture(out_dir=None, n: int = 16, seed: int = 0, feature_dim: int = 24,
                       n_slots: int = 8, slot_dim: int = 8, n_classes: int = 6,
                       name: str = "fixture_synth") -> Path:
    """Synthetic-style fixture with the REAL synth label layout.

    Mirrors what synth-lab emits (and what broke the Lead's first gate):
      * ``lab_state``  [T, n_slots*slot_dim] regression target (flattened slots)
      * ``lab_mask``   [T, n_slots]           PER-SLOT mask (state head only)
      * ``lab_age``    [T]                    seconds since last informative obs
      * ``lab_class``  [T] int64 with -1 on unsupervised steps (ce head)
    Features: a class one-hot + a random "memory value" that must be recalled
    after a hidden stretch, so the smoke run is meaningful.
    """
    out = Path(out_dir or (ROOT / "runs" / "_fixture_synth"))
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    splits = (["train"] * n_train + ["val"] * n_val
              + ["test"] * max(1, n - n_train - n_val))[:n]
    D = int(n_slots) * int(slot_dim)
    recs = []
    for i, split in enumerate(splits):
        T = int(rng.integers(32, 49))
        cls = int(i % n_classes)
        period = 1.0 / 20.0
        dt = period * (1.0 + 0.08 * rng.standard_normal(T))
        dt = np.maximum(dt, period * 0.2)
        t = np.concatenate([[0.0], np.cumsum(dt[1:])]).astype(np.float32)
        # hidden phase in the middle: target is invisible, state must be recalled
        h0 = int(T * 0.35)
        h1 = int(T * 0.75)
        hidden = np.zeros(T, dtype=bool)
        hidden[h0:h1] = True
        feat = 0.6 * rng.standard_normal((T, feature_dim)).astype(np.float32)
        cls_vec = np.zeros(feature_dim, dtype=np.float32)
        cls_vec[cls % feature_dim] = 1.0
        feat += cls_vec[None, :]
        # a random latent value appears before the hidden phase and is queried after
        latent = rng.standard_normal(D).astype(np.float32)
        state = np.zeros((T, D), dtype=np.float32)
        state[h0:h1] = latent[None, :]
        lab_state = state.copy()
        lab_mask = np.zeros((T, n_slots), dtype=np.float32)
        lab_mask[h0:h1] = 1.0
        lab_age = np.zeros(T, dtype=np.float32)
        lab_age[h0:h1] = np.arange(1, h1 - h0 + 1, dtype=np.float32) * period
        lab_class = np.full(T, -1, dtype=np.int64)
        lab_class[~hidden] = cls
        seq_id = f"{name}_{i:03d}"
        path = feat_dir / f"{seq_id}.npz"
        np.savez_compressed(path, feat=feat.astype(np.float16), t=t,
                            lab_state=lab_state.astype(np.float16),
                            lab_mask=lab_mask, lab_age=lab_age, lab_class=lab_class)
        recs.append(SeqRecord(seq_id=seq_id, source_id=f"src_{i:03d}", split=split,
                              T=T, path=str(path), labels=["state", "class", "mask", "age"],
                              meta={"class": cls, "fixture": True}))
    man = Manifest(dataset=name, task="synth", feature_dim=feature_dim, seqs=recs,
                   label_specs={"lab_state": {"kind": "mse"},
                                "lab_class": {"kind": "ce", "ignore": -1}},
                   extra={"fixture": True, "seed": seed, "n_slots": n_slots,
                          "slot_dim": slot_dim})
    mpath = out / "manifest.json"
    man.save(mpath)
    print(f"[fixture] wrote {len(recs)} synth-style sequences to {out} "
          f"(counts: {man.counts()}, feature_dim={feature_dim}, "
          f"lab_state=[T,{D}], lab_mask=[T,{n_slots}])")
    return mpath


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-dim", type=int, default=32)
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--corrupt", type=int, default=0)
    args = ap.parse_args(argv)
    make_fixture(args.out, n=args.n, seed=args.seed, feature_dim=args.feature_dim,
                 n_classes=args.classes, corrupt=args.corrupt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
