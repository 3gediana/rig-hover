# -*- coding: utf-8 -*-
"""_feat_info.py -- do the 64-d activation features actually carry information?

My earlier claim ("the feature block is a net liability") collapsed two different
questions:

  (a) do the activation features encode scene information at all?
  (b) does that information help *this* policy?

Only (b) was measured, and it came out negative because the teacher's labels are
computed from quantities that are already in the observation -- theta_img is
obs[64], person_h is obs[66]. A policy holding those needs nothing from a feature
block to reproduce the teacher. That says the block is redundant for this task,
not that the block is empty.

This script measures (a) directly, and separates "the block is empty" from "the
block points the wrong way in the deployment domain":

  in-domain   fit on data_pan_all, score on data_pan_holdout2   (same generator)
  cross       fit on data_pan_all, score on recorded robot frames

Targets:
  theta_img   the bearing angle, obs[64] -- present in the observation, so a
              policy that has it needs no feature block for it. A high in-domain
              R^2 here proves the block encodes geometry.
  emote       the teacher's expression head, NOT present anywhere in the 106-d
              observation. If features predict it better than the other channels
              do, the block carries information the policy cannot otherwise see.

Run:  python _feat_info.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"D:\robot_work\pc\train"

FEAT = slice(0, 64)
REST = slice(64, 106)


def load(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=True)
            if "obs" not in z.files:
                continue
            o = np.asarray(z["obs"], np.float32)
            if o.ndim == 3:
                o = o.reshape(-1, o.shape[-1])
            if o.shape[1] != 106:
                continue
            rows.append({
                "name": os.path.basename(p),
                "O": o,
                "emote": (np.asarray(z["emote"], np.int64)
                          if "emote" in z.files else None),
                "head": np.asarray(z["head_deg"], np.float32),
                "mode": np.asarray(z["mode"], np.int64),
            })
    return rows


def stack(rows, key=None):
    O = np.concatenate([r["O"] for r in rows], axis=0)
    if key == "emote":
        parts = [r["emote"] for r in rows if r["emote"] is not None]
        return O, (np.concatenate(parts).reshape(-1) if parts else None)
    return O, None


def ridge_fit(X, y, lam):
    """Closed-form ridge with an intercept, via the normal equations."""
    Xm = X.mean(axis=0)
    ym = y.mean(axis=0)
    Xc = X - Xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ (y - ym))
    return Xm, ym, W


def ridge_pred(model, X):
    Xm, ym, W = model
    return (X - Xm) @ W + ym


def r2(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> int:
    train = load([os.path.join(TRAIN, "data_pan_all", "*.npz")])
    indom = load([os.path.join(TRAIN, "data_pan_holdout2", "*.npz")])
    real = load([os.path.join(TRAIN, "data_real", "*.npz"),
                 os.path.join(TRAIN, "data_realtest", "*.npz")])
    if not (train and indom and real):
        print("missing data")
        return 2

    Otr, _ = stack(train)
    Oin, _ = stack(indom)
    Ore, _ = stack(real)
    print(f"train {Otr.shape[0]} frames | in-domain held-out {Oin.shape[0]} | "
          f"real robot {Ore.shape[0]}")

    lam = 1.0
    print("\n=== does the feature block encode the bearing angle (obs[64])? ===")
    y = Otr[:, 64]
    for label, Xtr, Xte, yte in (
            ("feat only", Otr[:, FEAT], Oin[:, FEAT], Oin[:, 64]),
            ("non-feat only", Otr[:, REST], Oin[:, REST], Oin[:, 64])):
        m = ridge_fit(Xtr, y, lam)
        pin = ridge_pred(m, Xte)
        pre = ridge_pred(m, Ore[:, FEAT] if Xtr.shape[1] == 64 else Ore[:, REST])
        print(f"  {label:<16} in-domain R2 {r2(yte, pin):+.3f}   "
              f"cross-domain R2 {r2(Ore[:, 64], pre):+.3f}")

    emo_tr = np.concatenate([r["emote"] for r in train if r["emote"] is not None])
    emo_in = np.concatenate([r["emote"] for r in indom if r["emote"] is not None])
    if emo_tr.size and emo_in.size:
        print("\n=== does the feature block predict emote (NOT in the 106-d obs)? ===")
        print(f"  emote classes present: train {sorted(set(emo_tr.tolist()))}, "
              f"in-domain {sorted(set(emo_in.tolist()))}")
        base = float((emo_in == np.bincount(emo_tr).argmax()).mean())
        print(f"  majority-class baseline accuracy: {100*base:.1f}%")
        for label, sl in (("feat only", FEAT), ("non-feat only", REST),
                          ("all 106", slice(0, 106))):
            m = ridge_fit(Otr[:, sl], emo_tr.astype(np.float32), lam)
            pred = np.rint(ridge_pred(m, Oin[:, sl])).astype(np.int64)
            acc = float((pred == emo_in).mean())
            print(f"  {label:<16} in-domain accuracy {100*acc:.1f}%")

    print("\n=== within-domain vs cross-domain, per feature dimension ===")
    A, B, C = Otr[:, FEAT], Oin[:, FEAT], Ore[:, FEAT]
    muA, sdA = A.mean(0), A.std(0) + 1e-6
    muB, sdB = B.mean(0), B.std(0) + 1e-6
    muC, sdC = C.mean(0), C.std(0) + 1e-6
    print(f"  train vs in-domain held-out : |dmu|/sd median "
          f"{np.median(np.abs(muB - muA) / sdA):.3f}   sd ratio median "
          f"{np.median(sdB / sdA):.3f}")
    print(f"  train vs real robot frames  : |dmu|/sd median "
          f"{np.median(np.abs(muC - muA) / sdA):.3f}   sd ratio median "
          f"{np.median(sdC / sdA):.3f}")
    print("  -> the first line being near zero is what a matched distribution")
    print("     looks like; the second is the domain gap that broke deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
