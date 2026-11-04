# -*- coding: utf-8 -*-
"""_recalib_cv.py -- leave-one-session-out validation of the normalisation refit.

_recalib_test.py fits the per-dim normalisation on all 715 real robot frames and
scores on the same frames. That is optimistic. This script fits on N-1 sessions
and scores on the held-out one, so every reported number comes from frames the
fit never saw.

Also reports how much real data the fit needs: the same procedure with the fit
restricted to the first K frames of the training sessions.

Run:  python _recalib_cv.py --blob ..\\train\\out_sw2_b256_80e\\m8.blob
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


def score(sessions, exe, blob, tmp, mu_w, sd_w, mu_s, sd_s):
    eh = ev = agr = z_head = z_vyaw = z_mode = 0.0
    ack_hit = ack_n = 0
    N = 0
    for s in sessions:
        O = s["O"].copy()
        if mu_w is not None:
            O[:, FEAT] = (s["O"][:, FEAT] - mu_w) / sd_w * sd_s + mu_s
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
            ack_hit += int((c["mode"][tm] == 2).sum())
            ack_n += int(tm.sum())
        N += n
    return {"head": eh / N, "vyaw": ev / N, "agr": agr / N,
            "ack_rec": (ack_hit / ack_n) if ack_n else float("nan"),
            "ack_n": ack_n, "z_head": z_head / N, "z_vyaw": z_vyaw / N,
            "z_mode": z_mode / N, "n": N}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", nargs="+",
                    default=[os.path.join(TRAIN, "data_real", "*.npz"),
                             os.path.join(TRAIN, "data_realtest", "*.npz")])
    a = ap.parse_args()

    sessions = load_sessions(a.sessions)
    tr = []
    for p in sorted(glob.glob(os.path.join(TRAIN, "data_pan_all", "*.npz")))[:80]:
        tr.append(np.load(p)["obs"][:, FEAT])
    tr = np.concatenate(tr, axis=0).astype(np.float32)
    mu_t, sd_t = tr.mean(0), np.maximum(tr.std(0), 1e-6)
    del tr

    tmp = tempfile.mkdtemp(prefix="recalib_cv_")
    exe = ES.build_host_engine(tmp)

    base = score(sessions, exe, a.blob, tmp, None, None, mu_t, sd_t)
    print(f"frames {base['n']}  teacher ACK frames {base['ack_n']}")
    print(f"always-zero baseline: head {base['z_head']:.2f}deg  "
          f"vyaw {base['z_vyaw']:.2f}  mode {100*base['z_mode']:.1f}%")
    print(f"as shipped (raw)   : head {base['head']:.2f}deg  "
          f"vyaw {base['vyaw']:.2f}  ACK rec {100*base['ack_rec']:.1f}%\n")

    print("leave-one-session-out (fit on the other 5, score on the held-out one)")
    print("-" * 78)
    print(f"{'held-out session':<22}{'fit frames':>11}{'head MAE':>10}"
          f"{'vyaw':>8}{'ACK rec':>9}{'teacher ACK':>12}")
    tot_hit = tot_ack = 0
    heads = []
    for i, held in enumerate(sessions):
        others = [s for j, s in enumerate(sessions) if j != i]
        X = np.concatenate([s["O"][:, FEAT] for s in others], axis=0)
        mu_r, sd_r = X.mean(0), np.maximum(X.std(0), 1e-6)
        r = score([held], exe, a.blob, tmp, mu_r, sd_r, mu_t, sd_t)
        tot_hit += int(round(r["ack_rec"] * r["ack_n"])) if r["ack_n"] else 0
        tot_ack += r["ack_n"]
        heads.append(r["head"] * r["n"])
        n = sum(len(s["O"]) for s in others)
        print(f"{held['name']:<22}{n:>11}{r['head']:>9.2f}°{r['vyaw']:>8.2f}"
              f"{100*r['ack_rec'] if r['ack_n'] else float('nan'):>8.1f}%"
              f"{r['ack_n']:>12d}")
    N = sum(r["n"] for r in [base])
    pooled_h = sum(heads) / sum(s["O"].shape[0] for s in sessions)
    print("-" * 78)
    print(f"pooled held-out head MAE {pooled_h:.2f}°   "
          f"ACK recall {100*tot_hit/max(tot_ack,1):.1f}% "
          f"({tot_hit}/{tot_ack})")

    print("\nhow much real data does the fit need?")
    print("-" * 78)
    print(f"{'fit frames':>12}{'head MAE':>10}{'vyaw':>8}{'ACK rec':>9}")
    pool = np.concatenate([s["O"][:, FEAT] for s in sessions], axis=0)
    for k in (30, 60, 120, 240, 480, len(pool)):
        X = pool[:k]
        mu_r, sd_r = X.mean(0), np.maximum(X.std(0), 1e-6)
        r = score(sessions, exe, a.blob, tmp, mu_r, sd_r, mu_t, sd_t)
        print(f"{k:>12}{r['head']:>9.2f}°{r['vyaw']:>8.2f}"
              f"{100*r['ack_rec']:>8.1f}%")
    print("(the last row is in-sample; the earlier rows are still optimistic "
          "because the fit and the score share the frame pool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
