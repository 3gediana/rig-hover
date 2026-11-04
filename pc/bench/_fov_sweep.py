"""Ruler-free horizontal-FOV calibration using the head servo as the reference.

Why this works
==============

`f_px` (focal length in pixels) is the only unknown that turns a detector box
into a distance. Measuring it normally needs a target of known physical size at
a measured distance, which needs a tape measure and a person standing still.

The robot already carries a calibrated angular actuator: the head yaw servo.
Rotate the head by a known angle and a *fixed* object in the room moves across
the image by exactly that angle, so

    x_px(theta) = f_px * tan(theta_obj - theta_head) + cx

Fitting that over a sweep of head angles yields `f_px` directly, with no ruler
and no cooperation beyond "sit still". `theta_head` is read back from the
firmware (`head.q` for the servo, `imu.yaw` for the head's world yaw -- both are
recorded and the fit reports which one actually explains the image motion,
because the balance loop counter-rotates the body against the head).

Safety: the sweep is deliberately small (+-15 deg, 5 deg steps) and between
steps the robot is checked for a fall. If pitch/roll leave the standing band the
sweep aborts and returns the head to 0.

Usage:
    python bench/_fov_sweep.py --angles=-15,-10,-5,0,5,10,15
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = "192.168.137.50"
PC = Path(__file__).resolve().parents[1]
YUNET = PC / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"


class Robot:
    def __init__(self, host: str = BASE) -> None:
        self.host = host
        self.conn = http.client.HTTPConnection(host, 80, timeout=10)

    def _get(self, path: str) -> bytes:
        self.conn.request("GET", path)
        resp = self.conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status}")
        return data

    def status(self) -> dict:
        return json.loads(self._get("/status"))

    def capture(self) -> np.ndarray:
        jpg = self._get("/capture?q=80")
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)

    def head(self, deg: float) -> None:
        self._get(f"/cmd?head={deg:.1f}")


def pose_ok(st: dict, pitch_band=25.0, roll_band=30.0) -> bool:
    return (abs(st["imu"]["pitch"]) < pitch_band
            and abs(st["imu"]["roll"]) < roll_band)


def face_cx(img: np.ndarray, det) -> float | None:
    # YuNet's first conv expects 3 channels; the wire format is grayscale.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _rc, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # Faces come back as x,y,w,h, 10 landmark coords, score at index 14.
    best = max(faces, key=lambda f: f[14])
    return float(best[0] + best[2] / 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", default="-15,-10,-5,0,5,10,15")
    ap.add_argument("--frames", type=int, default=6, help="frames per angle")
    ap.add_argument("--settle", type=float, default=1.2, help="seconds after a move")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2]
                                        / "evidence" / "live" / "fov_sweep"))
    ap.add_argument("--host", default=BASE)
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    det = cv2.FaceDetectorYN_create(str(YUNET), "", (240, 240), 0.45, 0.3, 20)
    rb = Robot(args.host)

    st0 = rb.status()
    print(f"start: pitch={st0['imu']['pitch']:.1f} roll={st0['imu']['roll']:.1f} "
          f"head.q={st0['head']['q']} up={st0['t']/1000:.0f}s")
    if not pose_ok(st0):
        print("refusing to sweep: robot is not standing", file=sys.stderr)
        return 2

    samples = []
    aborted = False
    for deg in angles:
        rb.head(deg)
        time.sleep(args.settle)
        st = rb.status()
        if not pose_ok(st):
            print(f"ABORT at head={deg:+.1f}: pitch={st['imu']['pitch']:.1f} "
                  f"roll={st['imu']['roll']:.1f}")
            aborted = True
            break
        xs, ys = [], []
        for k in range(args.frames):
            img = rb.capture()
            cx = face_cx(img, det)
            if cx is not None:
                xs.append(cx)
            ys.append(st["imu"]["yaw"])
            cv2.imwrite(str(out / "frames" / f"h{deg:+05.1f}_{k:02d}.jpg"), img)
            time.sleep(0.05)
        q = float(st["head"]["q"])
        yaw = float(st["imu"]["yaw"])
        if xs:
            samples.append({"commanded": deg, "q_head": q, "yaw": yaw,
                            "cx": float(np.median(xs)), "n_face": len(xs)})
            print(f"head cmd={deg:+6.1f}  q={q:+6.1f}  yaw={yaw:+7.1f}  "
                  f"face_cx={np.median(xs):6.1f} px  ({len(xs)}/{args.frames})")
        else:
            print(f"head cmd={deg:+6.1f}  q={q:+6.1f}  yaw={yaw:+7.1f}  "
                  f"no face detected")

    rb.head(0.0)
    print("head returned to 0")

    with open(out / "samples.json", "w", encoding="utf-8") as fh:
        json.dump({"aborted": aborted, "samples": samples}, fh, indent=1)

    if len(samples) < 4:
        print(f"\nnot enough samples ({len(samples)}) to fit f_px")
        return 3

    print("\n" + "=" * 70)
    print("Fit  x_px = f*(theta_obj - theta_head)  (small-angle, per degree)")
    print("=" * 70)
    for key, label in (("q_head", "servo readback head.q"), ("yaw", "IMU world yaw")):
        ang = np.array([s[key] for s in samples], float)
        cx = np.array([s["cx"] for s in samples], float)
        # dx/dtheta in px per degree -> f_px = -dx/dtheta * 180/pi
        A = np.vstack([ang, np.ones_like(ang)]).T
        slope, icpt = np.linalg.lstsq(A, cx, rcond=None)[0]
        resid = cx - (slope * ang + icpt)
        f_px = abs(slope) * 180.0 / np.pi
        fov = 2.0 * np.degrees(np.arctan(120.0 / f_px))
        print(f"{label:<24} slope={slope:8.3f} px/deg  ->  f_px={f_px:7.1f}"
              f"  FOV_H={fov:5.1f} deg  resid_rms={resid.std():5.2f} px")
    print()
    print("lower residual = the variable that really moves the camera. If both")
    print("are large, the head did not actually turn (check servo power).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
