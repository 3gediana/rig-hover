#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deploy_gate.py -- can this blob be deployed to the robot?

Every other suite in this repository scores the policy on synthetic or
HaGRID-derived sessions. None of them answers the question that decides whether
the robot actually works: what does the network do on frames from *its own
camera*, against the teacher's labels for those frames?

This gate answers it with recorded robot sessions, in three conditions:

  shipped     the blob exactly as exported
  calibrated  the blob with its 64 feature-dim normalisation refitted on the
              other recordings (``pc/train/calibrate_inputs.py``)
  zero        the trivial policy, reported as the floor a real policy must beat

The refit is scored **leave-one-session-out**: the normalisation used on a
session never saw that session, so the calibrated column is a held-out number.

Why only the feature dims are refitted: their real population is a property of
the camera and the room, so a lying-down robot samples it faithfully. The other
42 dims (proprioception, feedback, clock) depend on the robot standing, which is
not available while it is on its side.

Exit status is 0 when the calibrated blob beats the shipped one on both pooled
head MAE and teacher-ACK recall; 1 otherwise. The always-zero comparison is
printed but does not gate -- on mostly-stationary teacher labels a constant zero
scores well on head MAE while being useless for ACK.

Run:  python deploy_gate.py --blob ..\\train\\out_sw2_b256_80e\\m8.blob
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
import export_blob as EB  # noqa: E402

OBS_DIM = 106
FEAT_DIMS = 64


def glob_sessions(patterns):
    files = []
    for s in patterns:
        if os.path.isdir(s):
            files.extend(sorted(glob.glob(os.path.join(s, "*.npz"))))
        elif any(c in s for c in "*?["):
            files.extend(sorted(glob.glob(s)))
        else:
            files.append(s)
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "obs" not in z.files or "head_deg" not in z.files:
            continue
        O = np.asarray(z["obs"], np.float32)
        if O.ndim == 3:
            O = O.reshape(-1, O.shape[-1])
        if O.shape[1] != OBS_DIM:
            continue
        out.append({
            "name": os.path.basename(f),
            "path": f,
            "O": O,
            "head": np.asarray(z["head_deg"], np.float32),
            "vyaw": np.asarray(z["vyaw_dps"], np.float32),
            "mode": np.asarray(z["mode"], np.int64),
        })
    return out


def patch_blob(blob, feat_mean, feat_std):
    b = bytearray(blob)
    om = EB.HDR_FIELD_OFFSETS["input_mean"]
    os_ = EB.HDR_FIELD_OFFSETS["input_std"]
    mean = np.frombuffer(blob[om:om + 1024], dtype="<f4", count=256).copy()
    std = np.frombuffer(blob[os_:os_ + 1024], dtype="<f4", count=256).copy()
    mean[:FEAT_DIMS] = feat_mean
    std[:FEAT_DIMS] = feat_std
    b[om:om + 1024] = mean.astype("<f4").tobytes()
    b[os_:os_ + 1024] = std.astype("<f4").tobytes()
    oc = EB.HDR_FIELD_OFFSETS["crc32"]
    b[oc:oc + 4] = b"\x00\x00\x00\x00"
    b[oc:oc + 4] = np.uint32(EB.blob_crc32(bytes(b))).tobytes()
    return bytes(b)


def run_folds(sessions, exe, blob_bytes, tmp, folds, mode_hold=0):
    """Score `folds` (list of (held_session, fit_sessions)) and pool."""
    acc = {"head": 0.0, "vyaw": 0.0, "agr": 0.0, "ack_hit": 0, "ack_n": 0,
           "z_head": 0.0, "z_vyaw": 0.0, "z_mode": 0.0, "n": 0}
    per = []
    for held, fit in folds:
        if fit:
            F = np.concatenate([s["O"][:, :FEAT_DIMS] for s in fit], axis=0)
            mu = F.mean(axis=0)
            sd = np.maximum(F.std(axis=0), 1e-6)
            blob_path = os.path.join(tmp, "f.blob")
            with open(blob_path, "wb") as f:
                f.write(patch_blob(blob_bytes, mu, sd))
            fit_n = int(F.shape[0])
        else:
            blob_path = os.path.join(tmp, "shipped.blob")
            with open(blob_path, "wb") as f:
                f.write(blob_bytes)
            fit_n = 0
        O = held["O"]
        c = ES.run_engine(exe, blob_path, O, tmp)
        if mode_hold > 1:
            c["mode"] = ES.debounce_mode(c["mode"], mode_hold)
        n = len(O)
        eh = np.abs(c["head"] - held["head"]).mean()
        ev = np.abs(c["vyaw"] - held["vyaw"]).mean()
        ag = float((c["mode"] == held["mode"]).mean())
        tm = held["mode"] == 2
        hit = int((c["mode"][tm] == 2).sum()) if tm.any() else 0
        acc["head"] += eh * n
        acc["vyaw"] += ev * n
        acc["agr"] += ag * n
        acc["z_head"] += np.abs(held["head"]).mean() * n
        acc["z_vyaw"] += np.abs(held["vyaw"]).mean() * n
        acc["z_mode"] += float((held["mode"] == 0).mean()) * n
        acc["ack_hit"] += hit
        acc["ack_n"] += int(tm.sum())
        acc["n"] += n
        per.append((held["name"], fit_n, n, eh, ag, hit, int(tm.sum())))
    N = max(acc["n"], 1)
    return {
        "head": acc["head"] / N, "vyaw": acc["vyaw"] / N,
        "agr": acc["agr"] / N,
        "ack_rec": (acc["ack_hit"] / acc["ack_n"]) if acc["ack_n"] else None,
        "z_head": acc["z_head"] / N, "z_vyaw": acc["z_vyaw"] / N,
        "z_mode": acc["z_mode"] / N,
        "ack_hit": acc["ack_hit"], "ack_n": acc["ack_n"], "n": acc["n"],
    }, per


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", required=True,
                    help="candidate artifact, scored exactly as it would ship")
    ap.add_argument("--baseline",
                    default=os.path.join(TRAIN, "out_sw2_b256_80e", "m8.blob"),
                    help="the synthetic-feature model the candidate must beat")
    ap.add_argument("--sessions", nargs="+",
                    default=[os.path.join(TRAIN, "data_real"),
                             os.path.join(TRAIN, "data_realtest")])
    ap.add_argument("--mode-hold", type=int, default=0,
                    help="debounce the mode channel for N frames, as the "
                         "arbiter would; fixes the 'policy fires many 1-frame "
                         "ACK pulses' failure that per-frame scoring hides")
    ap.add_argument("--no-refit", action="store_true",
                    help="skip the leave-one-session-out normalisation refit "
                         "column (it is only informative for a blob whose "
                         "feature block was trained on synthetic data)")
    a = ap.parse_args()

    sessions = glob_sessions(a.sessions)
    if len(sessions) < 2:
        print("need at least 2 recorded sessions")
        return 2
    cand_blob = open(a.blob, "rb").read()
    base_blob = open(a.baseline, "rb").read()
    print(f"candidate {a.blob}  {len(cand_blob)} bytes")
    print(f"baseline  {a.baseline}  {len(base_blob)} bytes")
    print(f"recorded sessions: {len(sessions)}")
    for s in sessions:
        print(f"  {s['name']:24s} {len(s['O']):5d} frames  "
              f"teacher ACK frames {int((s['mode'] == 2).sum()):3d}")

    tmp = tempfile.mkdtemp(prefix="deploy_gate_")
    exe = ES.build_host_engine(tmp)

    shipped_folds = [(s, []) for s in sessions]
    refit_folds = [(s, [o for o in sessions if o is not s]) for s in sessions]

    base, base_per = run_folds(sessions, exe, base_blob, tmp, shipped_folds,
                               a.mode_hold)
    cand, cand_per = run_folds(sessions, exe, cand_blob, tmp, shipped_folds,
                               a.mode_hold)
    refit = None
    if not a.no_refit:
        refit, _ = run_folds(sessions, exe, base_blob, tmp, refit_folds,
                             a.mode_hold)

    print(f"\nframes {cand['n']}   teacher ACK frames {cand['ack_n']}"
          + (f"   mode-hold {a.mode_hold}" if a.mode_hold > 1 else ""))
    print("=" * 94)
    print(f"{'held-out session':<24}{'frames':>8}{'baseline':>10}"
          f"{'candidate':>11}{'zero':>8}{'ACK hit base/cand':>20}")
    print("-" * 94)
    for (nm, _fn, n, eh, _ag, hit, tan), c in zip(base_per, cand_per):
        z = np.abs(next(s for s in sessions if s["name"] == nm)["head"]).mean()
        print(f"{nm:<24}{n:>8}{eh:>9.2f}°{c[3]:>10.2f}°{z:>7.2f}°"
              f"{hit:>13d}/{tan}  {c[5]}/{c[6]}")
    print("=" * 94)
    cols = [("POOLED head MAE", "head"),
            ("vyaw MAE", "vyaw"),
            ("mode agreement", "agr")]
    print(f"{'':<24}{'frames':>8}{'baseline':>10}{'candidate':>11}"
          + (f"{'base+refit':>12}" if refit else ""))
    for label, key in cols:
        b = base[key]
        c = cand[key]
        btxt = f"{100*b:>10.1f}%" if key == "agr" else f"{b:>9.2f} "
        ctxt = f"{100*c:>10.1f}%" if key == "agr" else f"{c:>9.2f} "
        line = f"{label:<24}{cand['n']:>8}{btxt}{ctxt}"
        if refit:
            r = refit[key]
            line += f"{100*r:>10.1f}%" if key == "agr" else f"{r:>11.2f} "
        print(line)
    bt = "always-zero"
    bv = f"{base['z_head']:.2f}° head, {100*base['z_mode']:.1f}% mode"
    print(f"{bt:<24}{cand['n']:>8}{'':>10}{'':>11}   {bv}")
    print(f"{'ACK hits':<24}{cand['n']:>8}"
          f"{base['ack_hit']:>7d}/{base['ack_n']:<3d}"
          f"{cand['ack_hit']:>8d}/{cand['ack_n']:<3d}", end="")
    if refit:
        print(f"{refit['ack_hit']:>9d}/{refit['ack_n']:<3d}", end="")
    print()
    print("=" * 94)

    ok_head = cand["head"] < base["head"]
    ok_ack = cand["ack_hit"] > base["ack_hit"]
    ok_mode = cand["agr"] > base["agr"]
    print(f"head MAE improves on the baseline : {ok_head} "
          f"({base['head']:.2f}° -> {cand['head']:.2f}°)")
    print(f"teacher-ACK recall improves       : {ok_ack} "
          f"({base['ack_hit']}/{base['ack_n']} -> "
          f"{cand['ack_hit']}/{cand['ack_n']})")
    print(f"mode agreement improves           : {ok_mode} "
          f"({100*base['agr']:.1f}% -> {100*cand['agr']:.1f}%)")
    print(f"candidate head MAE vs always-zero : {cand['head']:.2f}° vs "
          f"{cand['z_head']:.2f}° "
          f"({'better' if cand['head'] < cand['z_head'] else 'still worse'})")
    print("NOTE the always-zero baseline is a serious competitor on this data:")
    print("     the teacher commands near-zero head most of the time, so a "
          "policy\n     that moves at all is charged for it. ACK recall and "
          "pulse structure\n     are the informative columns.")
    if not (ok_head and ok_ack and ok_mode):
        print("\nFAIL -- the candidate does not beat the baseline on held-out "
              "robot sessions")
        return 1
    print("\nPASS -- the candidate beats the baseline on every recorded robot "
          "session\n       (nothing in the candidate was fitted on this data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
