#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""session_from_realpan.py -- build training sessions from real photos by panning.

The blocker is not the perception chain (it is validated on real images) but the
fact that `synth_feat` fabricates its 64-d feature vector, so nothing the policy
learns transfers. This script attacks that directly: take a real photo of a
person, place it on a wider canvas, and slide a 240x240 crop across it. The
perception chain then sees **real features of a real person at real horizontal
angles**, in a temporal order that mimics someone walking past the robot.

What is real: the pixels, the YOLO keypoints, the 548->64, features, the
resulting theta_img.  What is synthetic: the scene ordering and the teacher's
reaction to it. That is strictly more realistic than the current synthetic
feature generator, and it is testable: re-run e2e_replay.py afterwards and see
whether the policy closes the gap.

Run:
    python session_from_realpan.py --n-images 40 --out data_pan
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = r"D:\robot_work\pc\perception"
TEACHER = r"D:\robot_work\pc\teacher"
GESTURE = r"D:\robot_work\pc\gesture"
MANIFEST = os.path.join(GESTURE, "data", "manifest.csv")

sys.path.insert(0, HERE)
sys.path.insert(0, PERCEPTION)
sys.path.insert(0, TEACHER)

CANVAS_H = 240
VIEW = 240


def static_proprio():
    return {
        "roll": 0.0, "pitch": -2.0, "pit_gyro": 0.0, "dq": 0.0, "yaw": 0.0,
        "q_head": 0.0, "target_head_pos": 0.0, "wheel1_vel": 0.0,
        "wheel2_vel": 0.0, "wheel_x": 0.0, "stable_pos": 0.0,
        "stable_yaw": 0.0, "robot_state": 1, "vx": 0.0, "vyaw": 0.0,
        "control_mode": 0, "feature_age_ms": 0.0, "gaze_danger": 0,
    }


def make_canvas(img, rng):
    """Put a real photo on a canvas wide enough to pan across."""
    h, w = img.shape[:2]
    scale = CANVAS_H / float(h)
    w2 = max(int(round(w * scale)), VIEW + 40)
    img = cv2.resize(img, (w2, CANVAS_H), interpolation=cv2.INTER_AREA)
    # background: replicate and darken the border so the pan does not expose
    # pure black, which the detector treats differently from a real scene
    left = max(VIEW // 2, 40)
    canvas = np.zeros((CANVAS_H, w2 + 2 * left, 3), np.uint8)
    border = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    tile = cv2.resize(border, (left, CANVAS_H))
    canvas[:, :left] = (tile[:, :1] * 0.85).astype(np.uint8)
    canvas[:, left:left + w2] = img
    canvas[:, left + w2:] = (tile[:, -1:] * 0.85).astype(np.uint8)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--step", type=int, default=6, help="pan step in pixels")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="depth oscillation amplitude. NOTE: leave at 0 for "
                         "HaGRID -- those are hand close-ups where the person "
                         "already fills the frame (measured person_h 0.73-0.84, "
                         "barely moving with zoom up to 0.9), so the teacher's "
                         "distance deadband (+-0.15 m) is never crossed and vx "
                         "stays 0. Real robot frames are needed for the "
                         "forward tier.")
    ap.add_argument("--out", default=os.path.join(HERE, "data_pan"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import obs as obsmod
    from run_perception import PerceptionPipeline
    from teacher import TeacherController, TeacherObs, load_config
    from synth import Session

    os.makedirs(a.out, exist_ok=True)
    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r["path"])
    rng = random.Random(a.seed)
    rng.shuffle(rows)

    pipe = PerceptionPipeline(device=a.device)
    tcfg = load_config(os.path.join(TEACHER, "config.yaml"))
    prop = static_proprio()

    done = 0
    for path in rows:
        if done >= a.n_images:
            break
        img = cv2.imread(path)
        if img is None:
            continue
        canvas = make_canvas(img, rng)
        max_x = canvas.shape[1] - VIEW
        if max_x <= a.step * 4:
            continue
        teacher = TeacherController(tcfg)
        teacher.reset()
        dist_est = obsmod.DistEstimator()
        obs_rows, head, vyaw, vx, mode, emote = [], [], [], [], [], []
        meta = {k: [] for k in ("theta_img", "eps_body", "dist_est",
                                "wave_score", "person_h", "seen", "wave_sin",
                                "q_head", "target_head_pos", "yaw", "t_s")}
        prev_act = None
        i = 0
        # Sweep left->right then right->left: two passes, like a person walking
        # past and back, which also gives the teacher both directions to label.
        xs = list(range(0, max_x + 1, a.step)) + \
             list(range(max_x, -1, -a.step))
        n_fr = max(len(xs), 1)
        for k, x in enumerate(xs):
            # Depth variation. Without it the teacher never commands vx (the
            # distance never changes), so the "forward" tier of the three-tier
            # allocation is unlearnable no matter how long we train. Scaling the
            # crop changes person_h, which is what dist_est reads.
            phase = 2.0 * math.pi * k / n_fr
            zoom = 1.0 + a.zoom * math.sin(phase)
            w = int(round(VIEW / max(zoom, 0.4)))
            cx = min(max(x + VIEW // 2, w // 2), canvas.shape[1] - w // 2)
            x0 = cx - w // 2
            crop = canvas[:, x0:x0 + w]
            if w != VIEW:
                crop = cv2.resize(crop, (VIEW, VIEW), interpolation=cv2.INTER_AREA)
            o = pipe(crop)
            t_s = i / 8.0
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
            obs_rows.append(obsmod.assemble(perc, prop, prev_act, t_s, d_est))
            act = teacher.step(TeacherObs(
                theta_img=o.theta_img, q_head=prop["q_head"], seen=o.seen,
                person_h=o.person_h, hand_up_L=o.hand_up_L,
                hand_up_R=o.hand_up_R, t_now=t_s, gesture_id=o.gesture_id,
                gesture_onset=o.gesture_onset, gesture_conf=o.gesture_conf,
                gesture_gap_s=o.gesture_gap_s))
            head.append(act.head_des); vyaw.append(act.vyaw_des)
            vx.append(act.vx_des); mode.append(act.mode); emote.append(act.emote)
            prev_act = act
            meta["theta_img"].append(o.theta_img)
            meta["eps_body"].append(act.debug.get("eps_body", 0.0))
            meta["dist_est"].append(d_est)
            meta["wave_score"].append(o.wave_gesture_conf)
            meta["person_h"].append(o.person_h)
            meta["seen"].append(float(o.seen))
            meta["wave_sin"].append(o.wave_sin)
            meta["q_head"].append(prop["q_head"])
            meta["target_head_pos"].append(prop["target_head_pos"])
            meta["yaw"].append(prop["yaw"])
            meta["t_s"].append(t_s)
            i += 1

        if len(obs_rows) < 64:
            continue
        sess = Session(
            obs=np.stack(obs_rows).astype(np.float32),
            head_deg=np.asarray(head, np.float32),
            vyaw_dps=np.asarray(vyaw, np.float32),
            vx_mps=np.asarray(vx, np.float32),
            mode=np.asarray(mode, np.int8),
            emote=np.asarray(emote, np.int8),
            meta={k: np.asarray(v, np.float32) for k, v in meta.items()},
        )
        sid = f"pan_{done:03d}"
        sess.save(os.path.join(a.out, f"{sid}.npz"))
        th = sess.meta["theta_img"]
        print(f"  {sid}: {len(sess):3d} frames  "
              f"theta [{th.min():+.1f}, {th.max():+.1f}] deg  "
              f"seen {100*float(sess.meta['seen'].mean()):.0f}%")
        done += 1

    print(f"\nwrote {done} panning sessions to {a.out}")
    if done == 0:
        print("nothing produced -- try --step smaller or a different source")
        return 2
    print("\nNext:\n"
          f"  .venv\\Scripts\\python.exe train_cfc.py --data-dir {a.out} "
          f"--out-dir out_pan --epochs 40\n"
          "  then re-run pc\\bench\\e2e_replay.py with the new blob to see "
          "whether mode agreement / MAE improved")
    return 0


if __name__ == "__main__":
    sys.exit(main())

