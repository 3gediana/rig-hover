#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""obs_domain_check.py -- will the trained policy's input statistics survive
real perception?

The report carries an unresolved risk: `input_mean/std` in the exported blob
were fitted on **synthetic** observations. The firmware clips every input to
mean +- clip_sigma*std before the CfC sees it, so if real observations land
outside that envelope, dimensions saturate and the policy silently runs on
clipped inputs.

This script measures it directly: run the deployed perception chain on real
HaGRID photos, assemble the same 106-dim observation the firmware builds, and
report how much of it falls outside the trained envelope.

Run:  python obs_domain_check.py --n 200
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = r"D:\robot_work\pc\perception"
TRAIN = r"D:\robot_work\pc\train"
GESTURE = r"D:\robot_work\pc\gesture"
MANIFEST = os.path.join(GESTURE, "data", "manifest.csv")

sys.path.insert(0, PERCEPTION)
sys.path.insert(0, TRAIN)


def static_proprio():
    """A robot lying still: pitch ~90 deg, everything else near zero.

    Proprioception is not what this check is about -- the synthetic set already
    covers it. Holding it fixed isolates the *perception* domain gap, which is
    the part that was never validated against real imagery.
    """
    return {
        "roll": -33.0, "pitch": 90.0, "pit_gyro": 0.0, "dq": 0.0,
        "yaw": 0.0, "q_head": 0.0, "target_head_pos": 0.0,
        "wheel1_vel": 0.0, "wheel2_vel": 0.0, "wheel_x": 0.0,
        "stable_pos": 0.0, "stable_yaw": 0.0, "robot_state": 0,
        "vx": 0.0, "vyaw": 0.0, "control_mode": 0,
        "feature_age_ms": 0.0, "gaze_danger": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--policy", default=os.path.join(TRAIN, "policy.npz"),
                    help="policy.npz to read input_mean/std from; point this at a "
                         "real-data-trained policy to see the envelope close")
    ap.add_argument("--framing", choices=("raw", "robot"), default="raw",
                    help="'raw' feeds the source photo as-is; 'robot' first "
                         "resizes and center-crops to 240x240 like the robot's "
                         "camera. Which one is right depends on what the policy "
                         "was trained on -- that is the point of offering both.")
    a = ap.parse_args()

    import obs as obsmod  # noqa: E402
    from run_perception import PerceptionPipeline  # noqa: E402

    pol = np.load(a.policy)
    mean = pol["input_mean"].astype(np.float32)
    std = pol["input_std"].astype(np.float32)
    sigma = float(pol["clip_sigma"]) if "clip_sigma" in pol.files else 4.0
    print(f"policy: {a.policy}")
    print(f"policy stats: {len(mean)} dims, clip_sigma={sigma}")

    rows = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r["path"])
    rng = random.Random(0)
    rng.shuffle(rows)
    paths = rows[:a.n]
    print(f"sampling {len(paths)} real photos\n")

    pipe = PerceptionPipeline(device=a.device)
    prop = static_proprio()
    obs = []
    n_ok = 0
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        if img is None:
            continue
        if a.framing == "robot":
            # Mimic the robot camera: 240x240 square, short side filled.
            h, w = img.shape[:2]
            s = 240.0 / max(min(h, w), 1)
            img = cv2.resize(img, (max(int(round(w * s)), 240),
                                   max(int(round(h * s)), 240)),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            y0, x0 = (h - 240) // 2, (w - 240) // 2
            img = img[y0:y0 + 240, x0:x0 + 240]
        out = pipe(img)
        perc = {
            "feat_64": out.feat_64,
            "theta_img": out.theta_img,
            "person_w": out.person_w,
            "person_h": out.person_h,
            "person_conf": out.person_conf,
            "seen": out.seen,
            "hand_up_L": out.hand_up_L,
            "hand_up_R": out.hand_up_R,
            "wave_sin": out.wave_sin,
            "wave_cos": out.wave_cos,
            "wave_amp": out.wave_amp,
            "gesture_conf": out.wave_gesture_conf,
        }
        try:
            v = obsmod.assemble(perc, prop, None, 0.0, 0.0)
        except Exception as exc:  # noqa: BLE001
            if n_ok == 0:
                print(f"assemble failed on first frame: {exc}")
                return 2
            continue
        obs.append(np.asarray(v, np.float32))
        n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}")

    if not obs:
        print("no observations assembled")
        return 2
    O = np.stack(obs)
    print(f"\nassembled {len(O)} observations of {O.shape[1]} dims")

    safe_std = np.maximum(std, 1e-6)
    z = (O - mean) / safe_std
    clipped = np.abs(z) > sigma
    ratio = clipped.mean(axis=0)
    overall = float(clipped.mean())

    print(f"\noverall clip ratio (fraction of inputs outside "
          f"mean+-{sigma}sigma): {100*overall:.1f}%")

    # Which dimensions are worst, by name
    try:
        names = obsmod.OBS_NAMES
    except AttributeError:
        names = [f"dim{i}" for i in range(O.shape[1])]
    order = np.argsort(-ratio)
    print(f"\n{'dim':>4} {'name':<24}{'clip%':>8}{'mean':>12}{'std':>10}"
          f"{'real mean':>12}{'real range':>22}")
    print("-" * 96)
    for k in order[:15]:
        rng_txt = f"[{O[:, k].min():.3f}, {O[:, k].max():.3f}]"
        print(f"{k:>4} {names[k]:<24}{100*ratio[k]:>7.1f}%{mean[k]:>12.3f}"
              f"{std[k]:>10.3f}{O[:, k].mean():>12.3f}{rng_txt:>22}")

    n_bad = int((ratio > 0.5).sum())
    n_warn = int((ratio > 0.05).sum())
    print("-" * 96)

    # Split the verdict by group. The proprio dims below are driven by the
    # static "robot is lying down" dict this script feeds in -- they would
    # saturate for a standing robot too only if the training set never saw a
    # lying one. The feature dims are the real question, because the synthetic
    # trainer FABRICATED them (`synth.synth_feat` is a random projection of six
    # scene variables, not YOLO output), so their statistics are unrelated to
    # what the perception chain actually produces.
    FEAT_DIMS = 64
    feat_ratio = float(ratio[:FEAT_DIMS].mean())
    rest_ratio = float(ratio[FEAT_DIMS:].mean())
    print(f"clip ratio over the 64 visual-feature dims : {100*feat_ratio:5.1f}%")
    print(f"clip ratio over the 42 proprio/gesture dims: {100*rest_ratio:5.1f}%"
          f"   (partly an artefact of the fixed lying-down proprio)")
    n_feat_bad = int((ratio[:FEAT_DIMS] > 0.5).sum())
    print(f"feature dims clipping >50% of frames      : {n_feat_bad}/64")

    # The gate is on the *fraction of inputs clipped*, not on the count of
    # heavy-tailed dimensions. Each dimension contributes 1/64 of the input, so
    # what matters for the policy is how much of its input vector is saturated;
    # a couple of naturally heavy-tailed dims clipping is not the same problem
    # as a third of the vector doing so.
    verdict = ("OK -- envelope holds" if feat_ratio < 0.05
               else "RISK -- the policy cannot be deployed as trained")
    print(f"\n{verdict}   (gate: feature clip ratio < 5%; "
          f"worst-dim count {n_feat_bad}/64 is a diagnostic, not the gate)")
    if verdict.startswith("RISK"):
        print("""
Why: pc/train/synth.py::synth_feat builds the 64-d "visual feature" as a fixed
random projection of six scene variables followed by tanh. That satisfies the
interface contract (bounded, non-constant) but does NOT reproduce the
statistics of YOLO's internal features. Consequence measured above: the real
perception chain lands far outside the trained envelope.

NOTE ON THIS CHECK'S REFERENCE. It feeds raw 512x512 HaGRID hand close-ups to
the perception chain, but the robot camera produces 240x240 room views. Matching
the robot's framing with --framing robot makes the ratio WORSE, which is the
tell: this script measures a policy envelope against the wrong population. It is
therefore a diagnostic, not a gate. deploy_gate.py asks the same question on
recorded robot frames and gates on behaviour (teacher-ACK recall) rather than on
a statistic.

All three options below were measured afterwards; numbers from
TECHNICAL-REPORT.md section 4.3c:

  1. Refit input_mean/std alone -- no labels, 600 unlabeled robot frames,
     leave-one-session-out: head MAE 5.63deg -> 2.37deg, teacher-ACK recall
     0/24 -> 14/24. Works and costs nothing; implemented in
     pc/train/calibrate_inputs.py. It does NOT make the feature block usable:
     zeroing the features afterwards changes nothing, so the recovered
     behaviour comes from the transmitted channels (target, gesture).
  2. Retrain on recorded robot features -- still the only way to make the 64-d
     block carry information. Needs frames from the robot's own camera.
  3. Calibrate synth_feat to the real distribution -- measured as the affine
     condition in _feat_ablation.py: teacher-ACK recall 75.9% -> 20.7%. It moves
     the statistics without adding information, so it is the worst of the three
     by a wide margin.
""")
    return 0 if verdict.startswith("OK") else 1


if __name__ == "__main__":
    sys.exit(main())

