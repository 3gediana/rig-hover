# -*- coding: utf-8 -*-
"""_real_feat_control.py -- on REAL robot frames, does the feature block carry the
behaviour the calibrated blob shows, or is it inert?

The calibration experiment recovers teacher-ACK recall from 0/24 to 14/24 on
recorded robot sessions. That number only means something if the features are
what the network is reading. Two conditions separate the cases:

  as-is     the calibrated blob
  no-feat   the same blob with obs[:, 0:64] zeroed before the engine

If no-feat holds the ACK recall, the recovery came from removing the saturation
that was poisoning the input vector, not from the network decoding real YOLO
features -- which matters, because the alternative reading would claim a
capability that was never demonstrated.

Also reports the same pair on the shipped (uncalibrated) blob, and a condition
that zeroes everything except the feature block, which shows how much the
feature block carries on its own.

Run:  python _real_feat_control.py --blob ..\\train\\out_sw2_b256_80e\\m8.blob
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
import deploy_gate as DG  # noqa: E402

FEAT = slice(0, 64)
REST = slice(64, 106)


def score(sessions, exe, blob_path, tmp, xform, mode_hold=0):
    acc = {"head": 0.0, "vyaw": 0.0, "agr": 0.0, "hit": 0, "ack_n": 0,
           "z_head": 0.0, "z_mode": 0.0, "n": 0, "ntr": 0}
    for s in sessions:
        O = s["O"].copy()
        xform(O)
        c = ES.run_engine(exe, blob_path, O, tmp)
        if mode_hold > 1:
            c["mode"] = ES.debounce_mode(c["mode"], mode_hold)
        n = len(O)
        acc["head"] += np.abs(c["head"] - s["head"]).mean() * n
        acc["vyaw"] += np.abs(c["vyaw"] - s["vyaw"]).mean() * n
        acc["agr"] += float((c["mode"] == s["mode"]).mean()) * n
        acc["z_head"] += np.abs(s["head"]).mean() * n
        acc["z_mode"] += float((s["mode"] == 0).mean()) * n
        tm = s["mode"] == 2
        if tm.any():
            acc["hit"] += int((c["mode"][tm] == 2).sum())
            acc["ack_n"] += int(tm.sum())
        acc["ntr"] += int((np.diff(c["mode"].astype(np.int64)) != 0).sum())
        acc["n"] += n
    N = max(acc["n"], 1)
    return {"head": acc["head"] / N, "vyaw": acc["vyaw"] / N,
            "agr": acc["agr"] / N,
            "ack": acc["hit"] / acc["ack_n"] if acc["ack_n"] else float("nan"),
            "hit": acc["hit"], "ack_n": acc["ack_n"],
            "z_head": acc["z_head"] / N, "ntr": acc["ntr"]}


def noop(O):
    pass


def zero_feat(O):
    O[:, FEAT] = 0.0


def only_feat(O):
    O[:, REST] = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", nargs="+",
                    default=[os.path.join(TRAIN, "data_real"),
                             os.path.join(TRAIN, "data_realtest")])
    a = ap.parse_args()

    sessions = DG.glob_sessions(a.sessions)
    blob = open(a.blob, "rb").read()
    tmp = tempfile.mkdtemp(prefix="real_feat_ctl_")
    exe = ES.build_host_engine(tmp)

    # leave-one-session-out calibration of the feature dims, as deploy_gate does
    blob_paths = {}
    for i, held in enumerate(sessions):
        F = np.concatenate([s["O"][:, :64] for j, s in enumerate(sessions)
                            if j != i], axis=0)
        mu, sd = F.mean(0), np.maximum(F.std(0), 1e-6)
        p = os.path.join(tmp, f"cal_{i}.blob")
        with open(p, "wb") as f:
            f.write(DG.patch_blob(blob, mu, sd))
        blob_paths[held["name"]] = p

    print(f"sessions {len(sessions)}  "
          f"frames {sum(len(s['O']) for s in sessions)}")

    for label, patched in (("shipped blob", False), ("calibrated (LOSO)", True)):
        bpaths = (blob_paths if patched
                  else {s["name"]: a.blob for s in sessions})
        print(f"\n=== {label} ===")
        hdr = (f"{'condition':<12}{'head MAE':>10}{'vyaw':>8}{'mode agr':>10}"
               f"{'ACK rec':>9}{'ACK hit':>10}{'transitions':>13}")
        print(hdr)
        print("-" * len(hdr))
        for name, xf in (("as-is", noop), ("no-feat", zero_feat),
                         ("only-feat", only_feat)):
            # per-session patched blob, so run manually rather than pooled
            acc = {"head": 0.0, "vyaw": 0.0, "agr": 0.0, "hit": 0, "ack_n": 0,
                   "n": 0, "ntr": 0, "z_head": 0.0}
            for s in sessions:
                O = s["O"].copy()
                xf(O)
                c = ES.run_engine(exe, bpaths[s["name"]], O, tmp)
                n = len(O)
                acc["head"] += np.abs(c["head"] - s["head"]).mean() * n
                acc["vyaw"] += np.abs(c["vyaw"] - s["vyaw"]).mean() * n
                acc["agr"] += float((c["mode"] == s["mode"]).mean()) * n
                acc["z_head"] += np.abs(s["head"]).mean() * n
                tm = s["mode"] == 2
                if tm.any():
                    acc["hit"] += int((c["mode"][tm] == 2).sum())
                    acc["ack_n"] += int(tm.sum())
                acc["ntr"] += int((np.diff(c["mode"].astype(np.int64)) != 0).sum())
                acc["n"] += n
            N = max(acc["n"], 1)
            ack = (acc["hit"] / acc["ack_n"]) if acc["ack_n"] else float("nan")
            print(f"{name:<12}{acc['head']/N:>9.2f}°{acc['vyaw']/N:>8.2f}"
                  f"{100*acc['agr']/N:>9.1f}%{100*ack:>8.1f}%"
                  f"{acc['hit']:>6d}/{acc['ack_n']:<4d}{acc['ntr']:>12d}")
        print(f"always-zero head MAE {acc['z_head']/N:.2f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
