"""No-model reference floors for the pixel event task.

A model comparison is only readable if the reader knows what "no learning" and
"no architecture" score.  Three references are computed here, all from the cached
pixel labels with no network:

1. **prior-only** per-frame macro-F1: a constant score means the only non-trivial
   predictor is "always fire" (F1 = 2p/(1+p)) or "never fire" (0).
2. **trivial onset detector**: the same two constants for the boundary task.
3. **linear probe**: ridge least squares from 16x16 mean-pooled frames to the
   per-frame label.  This is the strongest thing you can get *without any temporal
   model and without any trained front end*, i.e. an upper bound on what the
   per-frame appearance signal alone supports.

    python scripts/reference_floors_pixel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
REP = ROOT / "reports"
sys.path.insert(0, str(ROOT))

from src.eval.pixel_metrics import mean_average_precision  # noqa: E402


def pool(a: np.ndarray, n: int = 16) -> np.ndarray:
    T, H, W = a.shape
    s = H // n
    return (a.reshape(T, n, s, n, s).mean((2, 4)).reshape(T, -1) / 255.0).astype(np.float32)


def main() -> int:
    idx = json.loads((ROOT / "data_manifests" / "charades_pixel_index.json")
                     .read_text(encoding="utf-8"))
    C = len(idx["classes"])
    out = {"n_classes": C}
    frames = {}
    for sp in ("train", "test"):
        X, E, O = [], [], []
        for s in idx["seqs"]:
            if s["split"] != sp:
                continue
            z = np.load(s["path"])
            X.append(pool(np.asarray(z["frames"]).astype(np.float32)))
            E.append(np.asarray(z["event"], np.float32))
            O.append(np.asarray(z["onset"], np.float32))
        frames[sp] = (np.concatenate(X, 0), np.concatenate(E, 0), np.concatenate(O, 0))

    for sp in ("train", "test"):
        _X, Y, O = frames[sp]
        p = float(Y.mean())
        q = float(O.mean())
        out[sp] = {
            "n_frames": int(Y.shape[0]),
            "event_positive_rate": p,
            "prior_f1_allpos": 2 * p / (1 + p),
            "prior_f1_allneg": 0.0,
            "onset_positive_rate": q,
            "onset_f1_allpos": 2 * q / (1 + q),
            "onset_f1_allneg": 0.0,
        }

    # ---- linear probe (ridge least squares) on 16x16 pooled pixels ----
    Xtr, Ytr, _Otr = frames["train"]
    Xte, Yte, _Ote = frames["test"]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = np.c_[((Xtr - mu) / sd), np.ones(len(Xtr))]
    W = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Ytr)
    Ste = np.c_[((Xte - mu) / sd), np.ones(len(Xte))] @ W
    present = Yte.sum(0) > 0
    mp = mean_average_precision([Ste], [Yte], present)
    out["linear_probe"] = {"input": "16x16 mean-pooled grey frame",
                           "regressor": "ridge least squares, lambda=1.0",
                           "map": mp["map"], "ap_per_class": mp["ap_per_class"]}
    out["maps"] = {"prior": 0.0, "linear_probe": mp["map"]}

    (REP / "pixel_reference_floors.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("n_classes", "train", "test", "linear_probe")},
                     indent=1, ensure_ascii=False)[:1400])
    print(f"wrote {REP / 'pixel_reference_floors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
