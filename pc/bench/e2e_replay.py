#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e2e_replay.py -- offline replay benchmark: CfC vs teacher on REAL perception.

The simulation already compares the two, but on synthetic observations. This
runs the comparison through the real perception chain and through the **actual
firmware C inference engine** (compiled for the host), so it exercises the
deployed code path end to end without the robot:

    real photo -> YOLO26n-pose -> FeatureAdapter -> 106-d obs
               -> firmware cfc_model_forward()  ->  action
               -> teacher.step()                ->  reference action

Reports per-head error, mode agreement, and the fraction of frames where the
teacher would have moved the head while the CfC asked for zero body rotation
(the "head-only" behaviour the project cares about).

Run:  python e2e_replay.py --n 200
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = r"D:\robot_work\pc\perception"
TRAIN = r"D:\robot_work\pc\train"
TEACHER = r"D:\robot_work\pc\teacher"
GESTURE = r"D:\robot_work\pc\gesture"
MANIFEST = os.path.join(GESTURE, "data", "manifest.csv")

sys.path.insert(0, PERCEPTION)
sys.path.insert(0, TRAIN)
sys.path.insert(0, TEACHER)


def build_host_engine(tmpdir):
    """Compile the firmware's own C engine for the host."""
    import ctools
    import export_blob
    if not ctools.CL_EXE.exists():
        raise SystemExit(f"cl.exe not found: {ctools.CL_EXE}")
    srcs = [export_blob.FIRMWARE_DIR / "cfc_engine.c",
            export_blob.FIRMWARE_DIR / "cfc_model.c",
            os.path.join(TRAIN, "host_shim.c"),
            os.path.join(TRAIN, "host_blob_main.c")]
    out_dir = os.path.join(tmpdir)
    os.makedirs(out_dir, exist_ok=True)
    # hoststub/ supplies the ESP-IDF headers the firmware sources include
    # (esp_heap_caps.h and friends); without it cl.exe fails on the first line.
    exe, warns = ctools.compile_exe(
        srcs, "host_blob.exe",
        include_dirs=[os.path.join(TRAIN, "hoststub"),
                      export_blob.FIRMWARE_DIR],
        out_dir=out_dir)
    return exe


def static_proprio():
    return {
        "roll": 0.0, "pitch": -2.0, "pit_gyro": 0.0, "dq": 0.0, "yaw": 0.0,
        "q_head": 0.0, "target_head_pos": 0.0, "wheel1_vel": 0.0,
        "wheel2_vel": 0.0, "wheel_x": 0.0, "stable_pos": 0.0,
        "stable_yaw": 0.0, "robot_state": 1, "vx": 0.0, "vyaw": 0.0,
        "control_mode": 0, "feature_age_ms": 0.0, "gaze_danger": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--blob", default=os.path.join(TRAIN, "model_slot0.blob"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import cv2
    import obs as obsmod
    import export_blob
    from run_perception import PerceptionPipeline
    from teacher import TeacherController, TeacherObs, load_config

    if not os.path.exists(a.blob):
        print(f"blob not found: {a.blob}")
        return 2

    tmp = tempfile.mkdtemp(prefix="e2e_replay_")
    print(f"[1] compiling firmware C engine -> {tmp}")
    exe = build_host_engine(tmp)
    print(f"    {exe}")

    info = subprocess.run([str(exe), "info", a.blob], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    print("[2] blob:\n" + (info.stdout or info.stderr).strip()[:400])

    # ---- perception over real photos ----
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r["path"])
    rng = random.Random(0)
    rng.shuffle(rows)
    paths = rows[:a.n]

    pipe = PerceptionPipeline(device=a.device)
    tcfg = load_config(os.path.join(TEACHER, "config.yaml"))
    teacher = TeacherController(tcfg)
    teacher.reset()
    dist_est = obsmod.DistEstimator()
    prop = static_proprio()

    print(f"\n[3] running perception on {len(paths)} real photos")
    obs_list, t_actions, perc_log = [], [], []
    prev_act = None
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        o = pipe(img)
        perc = {
            "feat_64": o.feat_64, "theta_img": o.theta_img,
            "person_w": o.person_w, "person_h": o.person_h,
            "person_conf": o.person_conf, "seen": o.seen,
            "hand_up_L": o.hand_up_L, "hand_up_R": o.hand_up_R,
            "wave_sin": o.wave_sin, "wave_cos": o.wave_cos,
            "wave_amp": o.wave_amp,
            "gesture_conf": o.wave_gesture_conf,
        }
        d_est = dist_est.step(o.seen, o.person_h) if hasattr(dist_est, "step") else 0.0
        v = obsmod.assemble(perc, prop, prev_act, float(i) / 8.0, d_est)
        obs_list.append(np.asarray(v, np.float32))
        act = teacher.step(TeacherObs(
            theta_img=o.theta_img, q_head=prop["q_head"], seen=o.seen,
            person_h=o.person_h, hand_up_L=o.hand_up_L, hand_up_R=o.hand_up_R,
            t_now=float(i) / 8.0, gesture_id=o.gesture_id,
            gesture_onset=o.gesture_onset, gesture_conf=o.gesture_conf,
            gesture_gap_s=o.gesture_gap_s))
        t_actions.append((act.head_des, act.vyaw_des, act.mode, act.vx_des))
        perc_log.append({"seen": o.seen, "theta": o.theta_img})
        prev_act = act
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(paths)}")

    O = np.stack(obs_list).astype(np.float32)
    T = len(O)
    print(f"    {T} frames, obs {O.shape}")

    # ---- firmware C engine ----
    frames_path = os.path.join(tmp, "frames.bin")
    out_path = os.path.join(tmp, "out.bin")
    O.tofile(frames_path)
    print("\n[4] running the firmware C engine")
    r = subprocess.run([str(exe), "run", a.blob, frames_path, out_path],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        print("engine failed:\n" + (r.stdout or "") + (r.stderr or ""))
        return 2
    outs = np.fromfile(out_path, dtype=np.float32)
    n_heads = len(export_blob.HEADS)
    col = n_heads * 16
    C = outs.reshape(T, col)

    # ---- compare ----
    head_max = float(export_blob.HEADS[0][2]) if False else None  # noqa: F841
    SCALE = {"head_yaw": 45.0, "vyaw": 90.0, "vx": 0.30}
    idx = {name: i for i, (name, _d, _a) in enumerate(export_blob.HEADS)}

    cfc_head = C[:, idx["head_yaw"] * 16 + 0] * SCALE["head_yaw"]
    cfc_vyaw = C[:, idx["vyaw"] * 16 + 0] * SCALE["vyaw"]
    cfc_vx = C[:, idx.get("vx", 4) * 16 + 0] * SCALE["vx"]
    cfc_mode = np.argmax(C[:, idx["mode"] * 16: idx["mode"] * 16 + 4], axis=1)

    ta = np.asarray(t_actions, np.float32)
    t_head, t_vyaw, t_mode, t_vx = ta[:, 0], ta[:, 1], ta[:, 2].astype(int), ta[:, 3]

    def stats(x, y, name, unit):
        d = np.abs(x - y)
        print(f"  {name:<12} MAE {d.mean():8.2f} {unit}   "
              f"p95 {np.percentile(d, 95):8.2f}   max {d.max():8.2f}")

    print("\n[5] CfC (firmware C engine) vs teacher, same real observations")
    stats(cfc_head, t_head, "head_yaw", "deg")
    stats(cfc_vyaw, t_vyaw, "vyaw", "deg/s")
    stats(cfc_vx, t_vx, "vx", "m/s")
    agree = float((cfc_mode == t_mode).mean())
    print(f"  mode agreement {100*agree:.1f}%")

    seen = np.array([p["seen"] for p in perc_log], bool)
    if seen.any():
        both_still = ((np.abs(t_vyaw) < 1.0) & (np.abs(cfc_vyaw) < 1.0))
        print(f"\n  frames with a detected person      : {int(seen.sum())}/{T}")
        print(f"  both ask for no body rotation      : {100*both_still.mean():.1f}%")
        moved = ((np.abs(t_head) > 2.0) & (np.abs(cfc_head) > 2.0))
        print(f"  both move the head (>2 deg)        : {100*moved.mean():.1f}%")

    # Gate: this is a *plumbing* check, not a quality bar. A freshly trained
    # policy that saw none of this distribution cannot be expected to match the
    # teacher; what must hold is that the chain runs and produces finite,
    # in-range actions rather than diverging.
    finite = np.isfinite(cfc_head).all() and np.isfinite(cfc_vyaw).all()
    in_range = (np.abs(cfc_head) <= 45.001).all() and (np.abs(cfc_vyaw) <= 90.001).all()
    print(f"\n  finite outputs: {finite}   within actuator range: {in_range}")
    print("\nPASS (chain runs; action quality is a separate, not-yet-met bar)"
          if finite and in_range else "\nFAIL")
    return 0 if (finite and in_range) else 1


if __name__ == "__main__":
    sys.exit(main())

