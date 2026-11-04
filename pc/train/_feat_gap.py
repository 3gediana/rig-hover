# -*- coding: utf-8 -*-
"""_feat_gap.py -- is the visual-feature domain gap a shift/scale difference or a
structural one?

If the 64-d feature distribution differs between the training set and real robot
frames only in per-dimension mean and scale, per-dimension standardisation fixes
it. If it differs structurally, standardisation cannot help and the honest answer
is that real data is required.

Reports, per feature dim: training mean/std, real mean/std, the mean shift in
training-standard-deviation units, the scale ratio, and the clip ratio that would
remain after standardising the real stream by its OWN statistics (that residual is
the part standardisation cannot remove).

Run:  python _feat_gap.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(HERE, "data_pan_all")
REAL = [os.path.join(HERE, "data_realtest", "real_probe.npz")]
REAL += sorted(glob.glob(os.path.join(HERE, "data_real", "real_*.npz")))
FEAT = 64


def load_obs(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=True)
            if "obs" not in z.files:
                continue
            o = np.asarray(z["obs"], np.float32)
            if o.ndim == 3:
                o = o.reshape(-1, o.shape[-1])
            if o.shape[1] == 106:
                out.append(o)
    return np.concatenate(out, axis=0) if out else None


def main() -> int:
    tr = load_obs([os.path.join(TRAIN_DIR, "*.npz")])
    re_ = load_obs(REAL)
    if tr is None or re_ is None:
        print("missing data")
        return 2
    print(f"train frames {tr.shape[0]}   real frames {re_.shape[0]}")

    A = tr[:, :FEAT]
    B = re_[:, :FEAT]
    muA, sdA = A.mean(0), A.std(0) + 1e-6
    muB, sdB = B.mean(0), B.std(0) + 1e-6

    shift = (muB - muA) / sdA          # in training-std units
    ratio = sdB / sdA

    print(f"\nmean shift |(mu_real-mu_train)/std_train|: "
          f"median {np.median(np.abs(shift)):.2f}  max {np.abs(shift).max():.2f}")
    print(f"scale ratio std_real/std_train:            "
          f"median {np.median(ratio):.2f}  min {ratio.min():.2f}  max {ratio.max():.2f}")

    n_big_shift = int((np.abs(shift) > 2).sum())
    n_big_scale = int(((ratio > 3) | (ratio < 1 / 3)).sum())
    print(f"dims with |shift|>2 sigma : {n_big_shift}/{FEAT}")
    print(f"dims with scale ratio out of [1/3,3] : {n_big_scale}/{FEAT}")

    # residual after standardising the real stream by its own per-dim stats:
    # this measures structure that a shift/scale fix cannot remove. Compare the
    # training stream's own tails for the reference number.
    zB = (B - muB) / sdB
    zA = (A - muA) / sdA
    for sig in (3.0, 4.0):
        rb = (np.abs(zB) > sig).mean()
        ra = (np.abs(zA) > sig).mean()
        print(f"clip ratio at +-{sig:.0f}sigma  train {100*ra:5.2f}%  "
              f"real(after own standardisation) {100*rb:5.2f}%")

    # correlation between train and real per-dim profile: if the feature
    # SEMANTICS matched, the two mean vectors would be proportional-ish. A near
    # zero correlation says the dims are not measuring the same thing.
    c = np.corrcoef(muA, muB)[0, 1]
    print(f"\ncorr(mean_train, mean_real) over 64 dims : {c:+.3f}")
    c2 = np.corrcoef(sdA, np.maximum(sdB, 1e-6))[0, 1]
    print(f"corr(std_train, std_real)  over 64 dims : {c2:+.3f}")

    print(f"\n{'dim':>4}{'train_mu':>10}{'train_sd':>10}{'real_mu':>10}"
          f"{'real_sd':>10}{'shift':>8}{'ratio':>8}")
    order = np.argsort(-np.abs(shift))[:12]
    for i in order:
        print(f"{i:>4}{muA[i]:>10.3f}{sdA[i]:>10.3f}{muB[i]:>10.3f}"
              f"{sdB[i]:>10.3f}{shift[i]:>8.2f}{ratio[i]:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
