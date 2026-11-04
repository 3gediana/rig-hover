# -*- coding: utf-8 -*-
"""_recalib_test.py -- is the real-robot failure only a normalisation mismatch?

The firmware normalises every input as xn = clip((x - mean)/std) using mean/std
fitted on synthetic training data. On real robot frames the feat block sits far
outside that envelope, so it saturates at +-clip_sigma and the network sees a
near-binary input.

If mean/std were the whole story, feeding an observation pre-warped so that the
engine's own normalisation reproduces the real-normalised distribution would
restore sane behaviour. The warp is

    x' = (x_real - mu_real)/sd_real * sd_syn + mu_syn

which makes the engine compute xn' = (x_real - mu_real)/sd_real.

Conditions:
  raw        -- ship as-is
  realnorm   -- per-dim normalisation fitted on the real robot frames
  trainnorm  -- per-dim normalisation fitted on the training set (control: this
                is what the blob already uses, so it must match `raw`)

Reported against the always-zero baseline, because on this data a policy that
outputs a constant zero is a serious competitor.

Run:  python _recalib_test.py --blob ..\\train\\out_sw2_b256_80e\\m8.blob
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"D:\robot_work\pc\train"
sys.path.insert(0, HERE)
sys.path.insert(0, TRAIN)

import eval_session as ES  # noqa: E402

FEAT = slice(0, 64)


def load_sessions(patterns):
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=True)
            if "obs" not in z.files or "head_deg" not in z.files:
                continue
            O = np.asarray(z["obs"], np.float32)
            if O.ndim == 3:
                O = O.reshape(-1, O.shape[-1])
            if O.shape[1] != 106:
                continue
            out.append({
                "name": os.path.basename(p),
                "O": O,
                "head": np.asarray(z["head_deg"], np.float32),
                "vyaw": np.asarray(z["vyaw_dps"], np.float32),
                "mode": np.asarray(z["mode"], np.int64),
            })
    return out


def score(sessions, exe, blob, tmp, warp, mu_syn, sd_syn):
    eh, ev, agr, ack_rec, ack_n = 0.0, 0.0, 0.0, 0.0, 0
    z_head = z_vyaw = z_mode = 0.0
    N = 0
    for s in sessions:
        O = s["O"].copy()
        if warp is not None:
            mu_r, sd_r = warp
            O[:, FEAT] = (s["O"][:, FEAT] - mu_r) / sd_r * sd_syn + mu_syn
        c = ES.run_engine(exe, blob, O, tmp)
        n = len(O)
        eh += np.abs(c["head"] - s["head"]).mean() * n
        ev += np.abs(c["vyaw"] - s["vyaw"]).mean() * n
        agr += (c["mode"] == s["mode"]).mean() * n
        z_head += np.abs(s["head"]).mean() * n
        z_vyaw += np.abs(s["vyaw"]).mean() * n
        z_mode += (s["mode"] == 0).mean() * n
        tm = s["mode"] == 2
        if tm.any():
            ack_rec += (c["mode"][tm] == 2).sum()
            ack_n += int(tm.sum())
        N += n
    return {
        "n": N,
        "head": eh / N,
        "vyaw": ev / N,
        "agr": agr / N,
        "ack_rec": (ack_rec / ack_n) if ack_n else float("nan"),
        "ack_n": ack_n,
        "z_head": z_head / N,
        "z_vyaw": z_vyaw / N,
        "z_mode": z_mode / N,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", nargs="+",
                    default=[os.path.join(TRAIN, "data_real", "*.npz"),
                             os.path.join(TRAIN, "data_realtest", "*.npz")])
    a = ap.parse_args()

    sessions = load_sessions(a.sessions)
    if not sessions:
        print("no sessions")
        return 2
    allreal = np.concatenate([s["O"][:, FEAT] for s in sessions], axis=0)
    mu_r, sd_r = allreal.mean(0), np.maximum(allreal.std(0), 1e-6)

    tr = []
    for p in sorted(glob.glob(os.path.join(TRAIN, "data_pan_all", "*.npz")))[:80]:
        tr.append(np.load(p)["obs"][:, FEAT])
    tr = np.concatenate(tr, axis=0).astype(np.float32)
    mu_t, sd_t = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    del tr

    tmp = tempfile.mkdtemp(prefix="recalib_")
    exe = ES.build_host_engine(tmp)

    rows = []
    rows.append(("raw", score(sessions, exe, a.blob, tmp, None, mu_t, sd_t)))
    rows.append(("trainnorm", score(sessions, exe, a.blob, tmp,
                                    (mu_t, sd_t), mu_t, sd_t)))
    rows.append(("realnorm", score(sessions, exe, a.blob, tmp,
                                   (mu_r, sd_r), mu_t, sd_t)))

    print(f"sessions {len(sessions)}  frames {rows[0][1]['n']}  "
          f"teacher ACK frames {rows[0][1]['ack_n']}")
    print(f"always-zero baseline: head {rows[0][1]['z_head']:.2f}deg  "
          f"vyaw {rows[0][1]['z_vyaw']:.2f}  mode {100*rows[0][1]['z_mode']:.1f}%")
    print()
    hdr = (f"{'condition':<12}{'head MAE':>10}{'vyaw MAE':>10}{'mode agr':>10}"
           f"{'ACK rec':>9}{'vs zero':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows:
        ok = "better" if r["head"] < r["z_head"] else "WORSE"
        print(f"{name:<12}{r['head']:>9.2f}°{r['vyaw']:>10.2f}"
              f"{100*r['agr']:>9.1f}%{100*r['ack_rec']:>8.1f}%{ok:>10}")
    print("-" * len(hdr))
    print("trainnorm is the control: it must reproduce `raw`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
