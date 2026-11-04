# -*- coding: utf-8 -*-
"""_compare_blobs.py -- one table, every candidate blob, on the recorded robot frames.

Each blob is scored exactly as it would ship: no fitting on the evaluation data,
except for the one row that is explicitly the fitted variant and says so.

The columns that decide deployability are, in order: teacher-ACK recall and pulse
structure, then switch precision/recall, then head MAE. Head MAE alone is
misleading here because the teacher commands near-zero head most of the time, so
a constant zero scores well while doing nothing.

Run:  python _compare_blobs.py
"""
from __future__ import annotations

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

CANDIDATES = [
    ("m8 shipped (synthetic feat)", os.path.join(TRAIN, "out_sw2_b256_80e", "m8.blob")),
    ("m8 + r/w feat refit", os.path.join(TRAIN, "out_sw2_b256_80e", "m8_cal.blob")),
    ("no-feat raw stats", os.path.join(TRAIN, "out_nofeat", "m8.blob")),
    ("no-feat inert stats", os.path.join(TRAIN, "out_nofeat", "m8_inert.blob")),
]


def run_lengths(seq, value):
    m = (seq == value).astype(np.int64)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    return (np.where(d == -1)[0] - np.where(d == 1)[0]).tolist()


def score(sessions, exe, blob_path, tmp):
    acc = {"head": 0.0, "vyaw": 0.0, "agr": 0.0, "n": 0,
           "z_head": 0.0, "z_vyaw": 0.0, "z_mode": 0.0}
    ack_hit = ack_n = 0
    pulses = []
    tt = tp = 0
    matched_t = matched_p = 0
    for s in sessions:
        c = ES.run_engine(exe, blob_path, s["O"], tmp)
        n = len(s["O"])
        acc["head"] += np.abs(c["head"] - s["head"]).mean() * n
        acc["vyaw"] += np.abs(c["vyaw"] - s["vyaw"]).mean() * n
        acc["agr"] += float((c["mode"] == s["mode"]).mean()) * n
        acc["z_head"] += np.abs(s["head"]).mean() * n
        acc["z_vyaw"] += np.abs(s["vyaw"]).mean() * n
        acc["z_mode"] += float((s["mode"] == 0).mean()) * n
        acc["n"] += n
        tm = s["mode"] == 2
        if tm.any():
            ack_hit += int((c["mode"][tm] == 2).sum())
            ack_n += int(tm.sum())
        pulses.extend(run_lengths(c["mode"].astype(np.int64), 2))
        a_tt = np.where(np.diff(s["mode"].astype(np.int64)) != 0)[0] + 1
        a_tp = np.where(np.diff(c["mode"].astype(np.int64)) != 0)[0] + 1
        tt += len(a_tt)
        tp += len(a_tp)
        used = np.zeros(len(a_tp), bool)
        for st in a_tt:
            cand = np.where((~used) & (np.abs(a_tp - st) <= 2))[0]
            if len(cand):
                used[cand[0]] = True
                matched_t += 1
                matched_p += 1
    N = max(acc["n"], 1)
    return {
        "head": acc["head"] / N, "vyaw": acc["vyaw"] / N,
        "agr": acc["agr"] / N, "z_head": acc["z_head"] / N,
        "ack_rec": (ack_hit / ack_n) if ack_n else float("nan"),
        "ack_hit": ack_hit, "ack_n": ack_n,
        "pulses": len(pulses),
        "pulse_med": int(np.median(pulses)) if pulses else 0,
        "prec": matched_p / max(tp, 1), "rec": matched_t / max(tt, 1),
        "n": acc["n"],
    }


def main() -> int:
    sessions = DG.glob_sessions([os.path.join(TRAIN, "data_real"),
                                 os.path.join(TRAIN, "data_realtest")])
    print(f"sessions {len(sessions)}  frames {sum(len(s['O']) for s in sessions)}  "
          f"teacher ACK frames {sum(int((s['mode'] == 2).sum()) for s in sessions)}")
    # the teacher's own pulse structure, for reference
    tpul = []
    for s in sessions:
        tpul.extend(run_lengths(s["mode"].astype(np.int64), 2))
    print(f"teacher ACK pulses: {len(tpul)} "
          f"(median {int(np.median(tpul)) if tpul else 0} frames)")

    tmp = tempfile.mkdtemp(prefix="cmp_blobs_")
    exe = ES.build_host_engine(tmp)

    hdr = (f"{'blob':<30}{'head MAE':>10}{'vyaw':>8}{'mode':>8}"
           f"{'ACK rec':>10}{'pulses':>9}{'med':>5}{'sw P':>7}{'sw R':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    base = None
    for name, path in CANDIDATES:
        if not os.path.exists(path):
            print(f"{name:<30} MISSING")
            continue
        r = score(sessions, exe, path, tmp)
        if base is None:
            base = r
        ack = "" if np.isnan(r["ack_rec"]) else f"{100*r['ack_rec']:.0f}%"
        print(f"{name:<30}{r['head']:>9.2f}°{r['vyaw']:>8.2f}"
              f"{100*r['agr']:>7.1f}%{ack:>10}"
              f"{r['pulses']:>9}{r['pulse_med']:>5}"
              f"{100*r['prec']:>6.0f}%{100*r['rec']:>6.0f}%")
    print("-" * len(hdr))
    print(f"{'always-zero baseline':<30}{base['z_head']:>9.2f}°"
          f"{'':>8}{100*base['z_mode'] if 'z_mode' in base else 0:>7.1f}%")
    print("\nAll rows except 'm8 + r/w feat refit' are scored as shipped: nothing in")
    print("them was fitted on these recordings. The refit row exists only to show")
    print("what the calibration buys, and it was fitted leave-one-session-out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
