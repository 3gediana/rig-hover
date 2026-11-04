# -*- coding: utf-8 -*-
"""_feat_ablation.py -- how much does the policy actually depend on the 64-d
visual feature block?

The visual feature block is the one input whose real-world distribution is known
to be far outside the training envelope (measured: median 2.7 training-sigma mean
shift, 3.3x scale ratio, mean-profile correlation -0.06 against real robot
frames). Before deciding what that costs, measure whether the policy uses the
block at all.

Each condition rewrites obs[:, 0:64] and re-runs the firmware engine. Everything
else -- target, gesture, proprio, feedback, clock -- is left untouched, so the
delta is attributable to the feature block alone.

Run:  python _feat_ablation.py --blob ..\\train\\out_sw2_b256_80e\\m8.blob
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


def load_real_feat():
    """Real robot feature frames, used as the replacement population."""
    pats = [os.path.join(TRAIN, "data_realtest", "real_probe.npz")]
    pats += sorted(glob.glob(os.path.join(TRAIN, "data_real", "real_*.npz")))
    out = []
    for p in pats:
        z = np.load(p, allow_pickle=True)
        if "obs" not in z.files:
            continue
        o = np.asarray(z["obs"], np.float32)
        if o.ndim == 3:
            o = o.reshape(-1, o.shape[-1])
        if o.shape[1] == 106:
            out.append(o[:, :64])
    R = np.concatenate(out, axis=0) if out else np.zeros((1, 64), np.float32)
    return R


def cond_clean(O, rng, real, stats):
    return O.copy()


def cond_zero(O, rng, real, stats):
    O = O.copy()
    O[:, FEAT] = 0.0
    return O


def cond_frame_shuffle(O, rng, real, stats):
    """Same contents, temporal order destroyed: tests whether the block's
    usefulness is in its dynamics or in its per-frame content."""
    O = O.copy()
    idx = rng.permutation(len(O))
    O[:, FEAT] = O[idx][:, FEAT]
    return O


def cond_dim_shuffle(O, rng, real, stats):
    """Per-frame contents destroyed, marginals preserved: a strict null."""
    O = O.copy()
    idx = rng.permutation(64)
    O[:, FEAT] = O[:, idx][:, FEAT]
    return O


def cond_real_replace(O, rng, real, stats):
    """Swap in real robot features drawn from the recorded population."""
    O = O.copy()
    pick = rng.integers(0, len(real), size=len(O))
    O[:, FEAT] = real[pick]
    return O


def cond_affine_to_real(O, rng, real, stats):
    """Keep the temporal information, move the distribution onto the real one.

    Per-dim affine so the stream has the real mean and standard deviation while
    its within-dim temporal structure (and therefore the information the policy
    could read) is preserved. This is the 'what if we only had to fix the input
    statistics' condition.
    """
    O = O.copy()
    f = O[:, FEAT]
    mu_t, sd_t = stats["mu_t"], stats["sd_t"]
    mu_r, sd_r = stats["mu_r"], stats["sd_r"]
    O[:, FEAT] = (f - mu_t) / sd_t * sd_r + mu_r
    return O


def cond_noise(O, rng, real, stats):
    """Information-free substitute with the right statistics."""
    O = O.copy()
    mu_r, sd_r = stats["mu_r"], stats["sd_r"]
    O[:, FEAT] = rng.standard_normal((len(O), 64)).astype(np.float32) * sd_r + mu_r
    return O


CONDITIONS = [
    ("clean", cond_clean, "as recorded"),
    ("feat_zero", cond_zero, "feature block zeroed"),
    ("feat_dimshuf", cond_dim_shuffle, "per-dim shuffle (strict null)"),
    ("feat_frameshuf", cond_frame_shuffle, "frame shuffle (keeps per-frame content)"),
    ("feat_real", cond_real_replace, "replaced by real robot features"),
    ("feat_affine", cond_affine_to_real, "affine-mapped onto real statistics"),
    ("feat_noise", cond_noise, "gaussian noise with real statistics"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", default=os.path.join(TRAIN, "data_pan_holdout2"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.sessions, "*.npz")))
    if not files:
        print(f"no npz in {a.sessions}")
        return 2

    real = load_real_feat()
    print(f"real feature frames available: {real.shape[0]}")

    # statistics needed by the affine/noise conditions
    tr = []
    for p in sorted(glob.glob(os.path.join(TRAIN, "data_pan_all", "*.npz")))[:60]:
        z = np.load(p)
        tr.append(z["obs"][:, :64])
    tr = np.concatenate(tr, axis=0).astype(np.float32)
    stats = {
        "mu_t": tr.mean(0),
        "sd_t": np.maximum(tr.std(0), 1e-6),
        "mu_r": real.mean(0),
        "sd_r": np.maximum(real.std(0), 1e-6),
    }
    del tr

    tmp = tempfile.mkdtemp(prefix="feat_abl_")
    exe = ES.build_host_engine(tmp)

    sessions = []
    for f in files:
        z = np.load(f)
        sessions.append({
            "O": z["obs"].astype(np.float32),
            "head": z["head_deg"].astype(np.float32),
            "vyaw": z["vyaw_dps"].astype(np.float32),
            "mode": z["mode"].astype(np.int64),
        })
    nframes = sum(len(s["O"]) for s in sessions)
    print(f"sessions {len(sessions)}  frames {nframes}\n")

    hdr = (f"{'condition':<16}{'head MAE':>10}{'vyaw MAE':>10}"
           f"{'mode agr':>10}{'ACK rec':>9}{'ACK prec':>10}{'  transitions':>13}")
    print(hdr)
    print("-" * len(hdr))

    base = None
    for name, fn, _desc in CONDITIONS:
        rng = np.random.default_rng(a.seed)
        eh = []
        ev = []
        agr = []
        ack_t_rec = []
        ack_t_prec = []
        ntrans = 0
        for s in sessions:
            O = fn(s["O"], rng, real, stats)
            c = ES.run_engine(exe, a.blob, O, tmp)
            eh.append(np.abs(c["head"] - s["head"]).mean() * len(O))
            ev.append(np.abs(c["vyaw"] - s["vyaw"]).mean() * len(O))
            agr.append((c["mode"] == s["mode"]).mean() * len(O))
            # ACK recall: fraction of teacher ACK frames the policy also calls ACK
            tm = s["mode"] == 2
            if tm.any():
                ack_t_rec.append(((c["mode"][tm] == 2).mean(), int(tm.sum())))
                ack_t_prec.append((c["mode"] == 2))
            ntrans += int((np.diff(c["mode"].astype(np.int64)) != 0).sum())
        N = max(nframes, 1)
        h = sum(eh) / N
        v = sum(ev) / N
        g = sum(agr) / N
        rec = sum(r * n for r, n in ack_t_rec) / max(sum(n for _, n in ack_t_rec), 1)
        pred_ack = np.concatenate(ack_t_prec) if ack_t_prec else np.array([False])
        # precision is not well defined without a matched-teacher notion here;
        # report the policy's own ACK rate instead, which is what flicker shows up in
        own_ack = pred_ack.mean()
        if base is None:
            base = (h, v, g, rec)
        print(f"{name:<16}{h:>9.2f}°{v:>8.2f} {100*g:>9.1f}%{100*rec:>8.1f}%"
              f"{100*own_ack:>9.1f}%{ntrans:>13d}")

    print("-" * len(hdr))
    print("ACK rec = recall on teacher ACK frames; the last column is the policy's")
    print("own ACK frame rate (teacher ACK frame rate is fixed across conditions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
