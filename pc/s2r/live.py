# -*- coding: utf-8 -*-
"""s2r/live.py - lean real-robot runtime for RIG-Hover (scenario 1 + click-to-go), built on measured hardware facts.

Modes
  --mode dry      perception + control computed, NOTHING sent (safe with the robot lying / hand-held)
  --mode gaze     face/fist following.  --gaze hybrid (default: head centers fast, body turns to face you)
                                         --gaze v25    (Devin's head-first servo, for A/B)
  --mode s1       gaze + double-knock -> drive to the knock point (max travel --max-travel) -> dwell -> face
  --mode click    live camera window: left-click a point on the TABLE -> drive there (flat-table projection)
Safety (always on): pitch deviation / roll abort, firmware lead caps (yaw 18 deg, pos 5 cm), speed cap 5 cm/s,
travel cap, command watchdog (firmware 600 ms + thread 0.5 s), safe stop in `finally`, log flushed every frame.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (HERE, REPO / "pc" / "end_side", REPO / "pc" / "train", REPO / "pc" / "perception"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from link import F_PX_PER_DEG, Robot  # noqa: E402
from controllers import GotoPoint, HeadServo, HeadingServo, HybridGaze, HybridGazeV2, pixel_to_table, wrap  # noqa: E402
from fist import FistDetector  # noqa: E402

YUNET = REPO / "pc" / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"
POSE = REPO / "pc" / "perception" / "assets" / "yolo26n-pose.pt"
YOLOE = REPO / "pc" / "perception" / "assets" / "yoloe-26n-seg.pt"
KNOCK = REPO / "runs_knock" / "double_knock_fpsaug2_20261115" / "double_knock_deploy.pt"
CAM_TILT_DEG = float(os.environ.get("S2R_CAM_TILT_DEG", "nan"))   # optical axis below horizon; nan -> use IMU pitch (unverified)


class Safety:
    def __init__(self, pitch_dev: float = 9.0, roll_max: float = 14.0):
        self.pitch_dev, self.roll_max = pitch_dev, roll_max
        self.p0 = None
        self.reason = ""

    def ok(self, tel) -> bool:
        if tel is None:
            self.reason = "no telemetry"; return False
        if self.p0 is None:
            self.p0 = float(tel.pitch)
        if abs(tel.pitch - self.p0) > self.pitch_dev:
            self.reason = f"pitch {tel.pitch:.1f} vs {self.p0:.1f}"; return False
        if abs(tel.roll) > self.roll_max:
            self.reason = f"roll {tel.roll:.1f}"; return False
        return True


def build_perception():
    from real_robot_bridge import FusedHandFacePerception
    return FusedHandFacePerception(yunet_onnx=YUNET, yolo_pose_pt=POSE, yoloe_pt=YOLOE, enable_yoloe_closeup=True, yoloe_stride=2)


def build_knock():
    if not KNOCK.exists():
        return None
    import torch
    from train_double_knock import DoubleKnockDetector, DoubleKnockNet
    kb = torch.load(KNOCK, map_location="cpu", weights_only=False)
    km = DoubleKnockNet(in_ch=2, kin_dim=8, hidden_dim=64, num_classes=4).eval()
    km.load_state_dict(kb["state_dict"])
    return DoubleKnockDetector(model=km, device="cpu", window=int(kb.get("window", 48)), size=int(kb.get("size", 96)),
                               imp_threshold=0.38, double_threshold=0.45, sem_latch_s=2.0,
                               single_confirm_thr=0.72, single_lookback_s=1.5)


def cam_tilt(tel) -> float:
    return CAM_TILT_DEG if math.isfinite(CAM_TILT_DEG) else float(tel.pitch)


def main():
    # ultralytics YOLOE.set_classes() looks for mobileclip2_b.ts in the CWD and silently downloads 242 MB otherwise
    # (that is how three copies ended up in the repo); the canonical copy lives in the repo root.
    os.chdir(REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.137.50")
    ap.add_argument("--mode", default="dry", choices=["dry", "gaze", "s1", "click"])
    ap.add_argument("--gaze", default="hybrid", choices=["hybrid", "v25", "body"])
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--max-travel", type=float, default=0.30)
    ap.add_argument("--log", default=str(REPO / "runs_end_side" / "s2r_live" / time.strftime("%Y%m%d_%H%M%S")))
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--no-knock", action="store_true")
    ap.add_argument("--fist-recall", action="store_true", help="use s2r/fist recall-first detector instead of the afternoon raised-fist gate")
    a = ap.parse_args()
    out = pathlib.Path(a.log); out.mkdir(parents=True, exist_ok=True)
    flog = open(out / "frames.jsonl", "w", encoding="utf-8")
    send = a.mode != "dry"

    perc = build_perception()
    knock = None if (a.no_knock or a.mode in ("click",)) else build_knock()
    R = Robot(a.base)
    if not R.wait_ready(8):
        print("robot not reachable", flush=True); R.close(); return
    tel = R.udp.latest()
    print(f"ready: udp={R.udp.n} fps={R.frames.fps:.1f} pitch={tel.pitch:.1f} roll={tel.roll:.1f} state={tel.robot_state}", flush=True)
    if send and (tel.robot_state != 1 or abs(tel.roll) > 12):
        print("robot is not standing/balancing -> refusing to actuate (use --mode dry)", flush=True); R.close(); return
    bias0 = R.rest_bias(2.5)
    print(f"rest gyro bias {bias0:+.2f} dps", flush=True)
    safety = Safety()
    hyb, head_only, body_only = HybridGazeV2(), HeadServo(), HeadingServo()
    fistdet = None
    if a.fist_recall:
        from ultralytics import YOLO as _YOLO, YOLOE as _YOLOE
        fistdet = FistDetector(_YOLO(str(POSE)), _YOLOE(str(YOLOE)))
    from controllers import BodyGaze
    bgaze = BodyGaze()
    goto = GotoPoint()
    if send:
        R.act.cfg.update({"yaw_lead_max": 18.0, "pos_lead_max": 0.05, "vx_max": 0.08, "vyaw_max": 45.0})
        print("cfg", R.act.enable(init_bias_dps=bias0), flush=True)

    mode = "IDLE"; last_idx = 0; t0 = time.perf_counter(); last_t = t0
    ox = oy = 0.0; last_xm = None
    start_xy = (0.0, 0.0); dwell_until = 0.0; click = {"uv": None}
    if a.mode == "click":
        win = "s2r click-to-go (left click on the TABLE; q quits)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, lambda ev, x, y, f, p: click.__setitem__("uv", (x / 3.0, y / 3.0)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    n = 0; stats = {"face": 0, "fist": 0, "knock": 0, "frames": 0, "abort": ""}
    try:
        while time.perf_counter() - t0 < a.secs:
            fr = R.frames.get(after_idx=last_idx, wait=0.5)
            if fr is None:
                continue
            gray, bgr, t_exp, tel_cap, last_idx = fr
            now = time.perf_counter(); dt = now - last_t; last_t = now
            tel = R.udp.latest()
            if tel is None or tel_cap is None:
                continue
            if send and not safety.ok(tel):
                stats["abort"] = safety.reason; print("SAFETY ABORT:", safety.reason, flush=True); break
            # wheel odometry (drift free heading from wheel difference)
            xm = tel.x_m
            if last_xm is not None:
                ds = xm - last_xm; h = math.radians(tel.body_yaw_wheel)
                ox += ds * math.cos(h); oy += ds * math.sin(h)
            last_xm = xm
            lead_yaw = float(tel.stable_yaw) - tel.body_yaw_imu
            lead_pos = float(tel.stable_pos) - xm
            P = perc.detect(bgr, now, mode in ("IDLE", "FACE", "FIST"))
            face = P["faces"][0] if P["faces"] else None
            fist = P["fist"]
            if fistdet is not None:
                fb = None
                if face is not None:
                    cxp, cyp, wp, hp = face["cx"]*240, face["cy"]*240, face["w"]*240, face["h"]*240
                    fb = (cxp-wp/2, cyp-hp/2, cxp+wp/2, cyp+hp/2)
                d = fistdet.detect(bgr, now, face_box_px=fb)
                fist = d if (d is not None and not d.get("coasted")) else (fist if fist is not None else d)
            kout = {"fired": 0}
            if knock is not None:
                g96 = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
                kout = knock.step(g96, wrist_box=P["wrist_box"], scenario_state=None, timestamp=t_exp - t0)
            vx = vy = 0.0; head = 0.0; dbg = {}
            if a.mode == "click":
                if click["uv"] is not None:
                    u, v = click["uv"]; click["uv"] = None
                    pr = pixel_to_table(u, v, cam_tilt(tel_cap))
                    if pr is not None:
                        d, lat = pr
                        bear = tel_cap.q + math.degrees(math.atan2(lat, d))
                        gx = ox + d * math.cos(math.radians(tel.body_yaw_wheel + bear))
                        gy = oy + d * math.sin(math.radians(tel.body_yaw_wheel + bear))
                        if math.hypot(gx - ox, gy - oy) <= a.max_travel:
                            goto.set_goal(gx, gy); start_xy = (ox, oy); mode = "APPROACH"
                            print(f"CLICK uv=({u:.0f},{v:.0f}) -> table d={d:.3f}m lat={lat:+.3f}m goal=({gx:.3f},{gy:.3f})", flush=True)
                        else:
                            print(f"CLICK rejected: {d:.2f} m > max travel", flush=True)
                    else:
                        print("CLICK rejected: above horizon / not on the table plane", flush=True)
            elif knock is not None and kout.get("fired") and mode in ("FACE", "FIST", "IDLE") and P["wrist_box"] is not None:
                wb = P["wrist_box"]; u = 0.5 * (wb[0] + wb[2]); v = min(236.0, wb[3])
                pr = pixel_to_table(u, v, cam_tilt(tel_cap))
                stats["knock"] += 1
                if pr is not None and a.mode == "s1":
                    d, lat = pr
                    bear = tel_cap.q + math.degrees(math.atan2(lat, d))
                    d = min(d, a.max_travel)
                    gx = ox + d * math.cos(math.radians(tel.body_yaw_wheel + bear)); gy = oy + d * math.sin(math.radians(tel.body_yaw_wheel + bear))
                    goto.set_goal(gx, gy); start_xy = (ox, oy); mode = "APPROACH"
                    print(f"KNOCK -> table d={d:.3f} lat={lat:+.3f} bearing={bear:+.1f}", flush=True)
            if mode == "APPROACH":
                vx, vy, dbg = goto.step(now, ox, oy, tel.body_yaw_wheel, lead_yaw, lead_pos)
                head = 0.0
                if math.hypot(ox - start_xy[0], oy - start_xy[1]) > a.max_travel + 0.02:
                    goto.phase = "done"; print("travel cap reached", flush=True)
                if goto.phase == "done":
                    mode = "DWELL"; dwell_until = now + 2.0
            elif mode == "DWELL":
                vx, vy, _ = goto.step(now, ox, oy, tel.body_yaw_wheel, lead_yaw, lead_pos)
                if now > dwell_until:
                    mode = "IDLE"
            else:
                tgt = fist if fist is not None else face
                mode = "FIST" if fist is not None else ("FACE" if face is not None else "IDLE")
                cx = tgt["cx"] * 240.0 if tgt is not None else None
                det = (cx, tel_cap.body_yaw_wheel, tel_cap.q)
                if a.gaze == "hybrid":
                    vy, head = hyb.step(now, det, tel.body_yaw_wheel, lead_yaw, tel.q, dt, cx is not None)
                elif a.gaze == "v25":
                    head = head_only.step(now, cx, tel_cap.q, tel.q, dt) if cx is not None else head_only.hold
                    vy = 0.0
                else:
                    bgaze.update(now, cx, tel_cap.body_yaw_wheel, tel_cap.q, tel.q, tel.body_yaw_wheel)
                    vy, _ = body_only.step(now, tel.body_yaw_wheel, lead_yaw, bgaze.target)
                vx = float(np.clip(-2.0 * lead_pos, -0.02, 0.02))
            if send:
                R.act.set(vx=vx, vyaw=vy, head=head)
            stats["frames"] += 1; stats["face"] += face is not None; stats["fist"] += fist is not None
            rec = {"t": round(now - t0, 3), "lat_ms": round((now - t_exp) * 1e3, 1), "mode": mode,
                   "face_cx": None if face is None else round(face["cx"] * 240, 1), "fist_cx": None if fist is None else round(fist["cx"] * 240, 1),
                   "q": round(tel.q, 1), "psi_w": round(tel.body_yaw_wheel, 2), "lead_yaw": round(lead_yaw, 2), "lead_pos": round(lead_pos, 4),
                   "odo": [round(ox, 3), round(oy, 3)], "pitch": round(tel.pitch, 2), "roll": round(tel.roll, 2), "bias": round(R.act.bias_dps, 3),
                   "cmd": [round(vx, 4), round(vy, 2), round(head, 1)], "knock": int(kout.get("fired", 0)), "p_dbl": round(float(kout.get("p_double", 0) or 0), 3), "dbg": dbg}
            flog.write(json.dumps(rec) + "\n"); flog.flush()
            if n % a.save_every == 0:
                cv2.imwrite(str(out / f"f{n:05d}.jpg"), bgr)
            if a.mode == "click":
                vis = cv2.resize(bgr, (720, 720), interpolation=cv2.INTER_NEAREST)
                for vv in range(130, 240, 15):
                    pr = pixel_to_table(120, vv, cam_tilt(tel))
                    if pr: cv2.putText(vis, f"{pr[0]*100:.0f}cm", (5, vv * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(vis, f"{mode} odo=({ox:+.2f},{oy:+.2f}) psi={tel.body_yaw_wheel:+.1f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imshow(win, vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            n += 1
    finally:
        R.close()
        flog.close()
    # summary
    L = [json.loads(l) for l in open(out / "frames.jsonl", encoding="utf-8")]
    fx = np.array([r["face_cx"] for r in L if r["face_cx"] is not None and r["mode"] == "FACE"], dtype=float)
    lat = np.array([r["lat_ms"] for r in L]) if L else np.array([0.0])
    summ = {"mode": a.mode, "gaze": a.gaze, "frames": len(L), "fps": round(len(L) / max(1e-3, (L[-1]["t"] if L else 1)), 1),
            "loop_latency_ms_p50": round(float(np.median(lat)), 1), "loop_latency_ms_p90": round(float(np.percentile(lat, 90)), 1),
            "face_frames": int(len(fx)), "abort": stats["abort"], "knock_fired": stats["knock"],
            "face_err_deg_med": round(float(np.median(abs(fx - 120) / F_PX_PER_DEG)), 2) if len(fx) else None,
            "face_err_deg_p90": round(float(np.percentile(abs(fx - 120) / F_PX_PER_DEG, 90)), 2) if len(fx) else None,
            "face_within5deg_pct": round(float(np.mean(abs(fx - 120) / F_PX_PER_DEG < 5) * 100), 1) if len(fx) else None,
            "body_travel_m": round(float(max((math.hypot(*r["odo"]) for r in L), default=0.0)), 3)}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
