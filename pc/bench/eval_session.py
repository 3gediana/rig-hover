#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_session.py -- score the firmware C engine against the teacher's labels
on held-out recorded sessions.

A session npz already contains the 106-d observations and the teacher's actions
for every frame, so evaluation needs no perception at all: feed the observations
through the **firmware's own C engine** and compare.

Two baselines are reported alongside, because a low error is meaningless without
them:
  * `always_zero`  -- the trivial policy (head 0, vyaw 0, IDLE everywhere)
  * `teacher`      -- the labels themselves (error 0 by construction)

Run:
    python eval_session.py --blob ..\\train\\out_pan\\model_pan.blob ^
        --sessions ..\\train\\data_pan_holdout
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"D:\robot_work\pc\train"
sys.path.insert(0, TRAIN)


def build_host_engine(tmpdir):
    import ctools
    import export_blob
    srcs = [export_blob.FIRMWARE_DIR / "cfc_engine.c",
            export_blob.FIRMWARE_DIR / "cfc_model.c",
            os.path.join(TRAIN, "host_shim.c"),
            os.path.join(TRAIN, "host_blob_main.c")]
    os.makedirs(tmpdir, exist_ok=True)
    exe, _warns = ctools.compile_exe(
        srcs, "host_blob.exe",
        include_dirs=[os.path.join(TRAIN, "hoststub"),
                      export_blob.FIRMWARE_DIR],
        out_dir=tmpdir)
    return exe


def run_engine(exe, blob, O, tmpdir):
    import export_blob
    fp = os.path.join(tmpdir, "f.bin")
    op = os.path.join(tmpdir, "o.bin")
    O.astype(np.float32).tofile(fp)
    r = subprocess.run([str(exe), "run", blob, fp, op], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError((r.stdout or "") + (r.stderr or ""))
    n_heads = len(export_blob.HEADS)
    C = np.fromfile(op, dtype=np.float32).reshape(len(O), n_heads * 16)
    idx = {name: i for i, (name, _d, _a) in enumerate(export_blob.HEADS)}
    return {
        "head": C[:, idx["head_yaw"] * 16] * 45.0,
        "vyaw": C[:, idx["vyaw"] * 16] * 90.0,
        "vx": C[:, idx["vx"] * 16] * 0.30,
        "mode": np.argmax(C[:, idx["mode"] * 16: idx["mode"] * 16 + 4], axis=1),
    }


def debounce_mode(seq, hold_frames):
    """Accept a new mode only after it has been the raw argmax for `hold_frames`
    consecutive frames; otherwise keep the previously accepted mode.

    Same idea as the gesture layer's `min_gesture`, applied to the control mode
    channel. A smooth network crosses the argmax boundary near ties, which is
    why the raw policy emits 15 three-frame ACK pulses where the teacher emits
    5 twelve-frame ones. This is the post-process the firmware arbiter would
    apply in the same place it already applies the rate limiters.
    """
    if hold_frames <= 1 or len(seq) == 0:
        return np.asarray(seq, np.int64).copy()
    out = np.empty(len(seq), np.int64)
    out[0] = seq[0]
    cur = int(seq[0])
    run_val, run_len = int(seq[0]), 1
    for i in range(1, len(seq)):
        v = int(seq[i])
        if v == run_val:
            run_len += 1
        else:
            run_val, run_len = v, 1
        if run_len >= hold_frames:
            cur = v
        out[i] = cur
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blob", required=True)
    ap.add_argument("--sessions", required=True,
                    help="directory of held-out session npz files")
    ap.add_argument("--mode-hold", type=int, default=0,
                    help="debounce the mode channel for N frames (0 = raw). "
                         "With 0 the policy is scored exactly as the firmware "
                         "would run it today; >0 shows what a minimum-duration "
                         "hold in the arbiter would buy.")
    ap.add_argument("--switch-tol", type=int, default=None,
                    help="frames of slack when matching a predicted switch to a "
                         "teacher switch. Defaults to 2, widened to mode-hold+2 "
                         "when a hold is applied -- otherwise the filter's own "
                         "delay is charged against it as a false switch.")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.sessions, "*.npz")))
    if not files:
        print(f"no npz in {a.sessions}")
        return 2

    tmp = tempfile.mkdtemp(prefix="eval_session_")
    exe = build_host_engine(tmp)

    tot = {"n": 0, "head": 0.0, "vyaw": 0.0, "vx": 0.0, "mode": 0,
           "z_head": 0.0, "z_vyaw": 0.0, "z_mode": 0}
    per_session = []
    cm_true, cm_pred = [], []
    for f in files:
        z = np.load(f)
        O = z["obs"].astype(np.float32)
        t_head = z["head_deg"].astype(np.float32)
        t_vyaw = z["vyaw_dps"].astype(np.float32)
        t_vx = z["vx_mps"].astype(np.float32)
        t_mode = z["mode"].astype(np.int64)

        c = run_engine(exe, a.blob, O, tmp)
        if a.mode_hold > 1:
            c["mode"] = debounce_mode(c["mode"], a.mode_hold)
        e_head = np.abs(c["head"] - t_head).mean()
        e_vyaw = np.abs(c["vyaw"] - t_vyaw).mean()
        e_vx = np.abs(c["vx"] - t_vx).mean()
        agr = float((c["mode"] == t_mode).mean())

        # trivial baseline: always command zero / IDLE
        z_head = np.abs(0.0 - t_head).mean()
        z_vyaw = np.abs(0.0 - t_vyaw).mean()
        z_mode = float((np.zeros_like(t_mode) == t_mode).mean())

        n = len(O)
        tot["n"] += n
        tot["head"] += e_head * n
        tot["vyaw"] += e_vyaw * n
        tot["vx"] += e_vx * n
        tot["mode"] += agr * n
        tot["z_head"] += z_head * n
        tot["z_vyaw"] += z_vyaw * n
        tot["z_mode"] += z_mode * n
        per_session.append((os.path.basename(f), n, e_head, z_head, agr, z_mode))
        cm_true.append(t_mode)
        cm_pred.append(c["mode"])

    N = max(tot["n"], 1)
    print(f"\nheld-out sessions: {len(files)}   frames: {tot['n']}")
    print("=" * 84)
    print(f"{'session':<22}{'frames':>8}{'head MAE':>11}{'  vs zero':>11}"
          f"{'mode agr':>11}{'  vs zero':>11}")
    print("-" * 84)
    for name, n, eh, zh, agr, zm in per_session[:12]:
        print(f"{name:<22}{n:>8}{eh:>10.2f}°{zh:>10.2f}°"
              f"{100*agr:>10.1f}%{100*zm:>10.1f}%")
    print("=" * 84)
    print(f"{'POOLED':<22}{tot['n']:>8}{tot['head']/N:>10.2f}°"
          f"{tot['z_head']/N:>10.2f}°{100*tot['mode']/N:>10.1f}%"
          f"{100*tot['z_mode']/N:>10.1f}%")
    print(f"{'':22}{'':8}{tot['vyaw']/N:>10.2f} {tot['z_vyaw']/N:>10.2f}"
          f"   (vyaw deg/s)")
    print("=" * 84)

    # ---- mode-SWITCH metrics --------------------------------------------
    cm_true = np.concatenate(cm_true) if len(cm_true) else np.zeros(0, np.int64)
    cm_pred = np.concatenate(cm_pred) if len(cm_pred) else np.zeros(0, np.int64)
    # Per-frame agreement hides flicker: a policy can hit 70% of ACK frames and
    # still flip TRACK/ACK/TRACK every few frames, which is unusable on a robot.
    # What matters operationally is whether the *transitions* line up and how
    # long a confirmed pulse lasts.
    def transitions(seq):
        return np.where(np.diff(seq.astype(np.int64)) != 0)[0] + 1

    tt, tp = transitions(cm_true), transitions(cm_pred)
    tol = a.switch_tol if a.switch_tol is not None else (2 + max(a.mode_hold, 0))
    matched_t = 0
    used = np.zeros(len(tp), bool)
    for s in tt:
        cand = np.where((~used) & (np.abs(tp - s) <= tol))[0]
        if len(cand):
            used[cand[0]] = True
            matched_t += 1
    prec = matched_t / max(len(tp), 1)
    rec = matched_t / max(len(tt), 1)

    def run_lengths(seq, value):
        m = (seq == value).astype(np.int64)
        if not m.any():
            return []
        d = np.diff(np.concatenate([[0], m, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        return (ends - starts).tolist()

    pol_ack = run_lengths(cm_pred, 2)
    tea_ack = run_lengths(cm_true, 2)
    print("mode-switch metrics")
    print(f"  teacher transitions {len(tt):4d}   policy transitions {len(tp):4d}")
    print(f"  switch precision {prec:5.1%}   switch recall {rec:5.1%}   "
          f"(match within +-{tol} frames)")
    print(f"  ACK pulses: teacher {len(tea_ack)} "
          f"(median {int(np.median(tea_ack)) if tea_ack else 0} frames) | "
          f"policy {len(pol_ack)} "
          f"(median {int(np.median(pol_ack)) if pol_ack else 0} frames)")
    if len(pol_ack) > 3 * max(len(tea_ack), 1):
        print("  !! the policy fires far more ACK pulses than the teacher -- "
              "flicker, not detection")
    print("=" * 84)
    better = (tot["head"] / N) < (tot["z_head"] / N)
    better_vy = (tot["vyaw"] / N) < (tot["z_vyaw"] / N)
    print(f"beats the always-zero baseline on head MAE : {better}")
    print(f"beats the always-zero baseline on vyaw MAE : {better_vy}")

    # ---- per-mode recall -------------------------------------------------
    # The pooled "mode agreement" is dominated by whatever class is most common
    # (TRACK is ~98% of frames here), so a policy that always answers TRACK
    # scores ~98% while never detecting a single gesture. Recall per class is
    # the only number that answers "did it learn the rare behaviour".
    # (cm_true / cm_pred were concatenated above for the switch metrics)

    print("\nper-mode recall on held-out data")
    print(f"{'mode':<10}{'frames':>8}{'recall':>9}{'predicted-as':>40}")
    names = {0: "IDLE", 1: "TRACK", 2: "ACK", 3: "REFRACTORY"}
    for m in sorted(np.unique(np.concatenate([cm_true, cm_pred]))):
        mask = cm_true == m
        n_m = int(mask.sum())
        if n_m == 0:
            continue
        rec = float((cm_pred[mask] == m).mean())
        dist = {}
        for p in cm_pred[mask]:
            dist[int(p)] = dist.get(int(p), 0) + 1
        txt = " ".join(f"{names.get(k, k)}={v}" for k, v in
                       sorted(dist.items(), key=lambda x: -x[1]))
        print(f"{names.get(m, m):<10}{n_m:>8}{100*rec:>8.1f}%   {txt}")
    print("=" * 84)
    print("\n" + ("PASS -- the policy learned something beyond 'do nothing'"
                  if better and better_vy
                  else "FAIL -- the policy is not better than doing nothing"))
    return 0 if (better and better_vy) else 1


if __name__ == "__main__":
    sys.exit(main())
