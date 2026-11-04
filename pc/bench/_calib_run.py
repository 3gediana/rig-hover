"""Continuous burst recorder for human-in-the-loop calibration.

Records every frame for a fixed duration at a fixed rate and writes a sidecar
CSV with wall-clock time and robot posture per frame. The operator performs a
timed protocol (A4 sheet at known distances, then face/fist at known
distances); the analysis segments the recording by what it sees and by order,
so no signalling between operator and recorder is required.

Usage:
    python bench/_calib_run.py --seconds 300 --fps 5 --out evidence/live/calib_1
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import sys
import time
from pathlib import Path

BASE = "192.168.137.50"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=BASE)
    ap.add_argument("--status-every", type=int, default=10,
                    help="sample /status every N frames")
    args = ap.parse_args()

    out = Path(args.out)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)

    cam = http.client.HTTPConnection(args.host, 80, timeout=10)
    st = http.client.HTTPConnection(args.host, 80, timeout=10)

    def get(conn, path):
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"{path} -> HTTP {resp.status}")
        return data

    period = 1.0 / args.fps
    t0 = time.time()
    idx = 0
    saved = 0
    failures = 0
    posture: dict[str, float] = {}

    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "wall", "elapsed_s", "bytes", "pitch", "roll",
                    "v1", "v2", "heap", "ok"])
        while True:
            elapsed = time.time() - t0
            if elapsed >= args.seconds:
                break
            target = t0 + idx * period
            sleep_for = target - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            try:
                jpg = get(cam, "/capture")
            except Exception as exc:  # noqa: BLE001 - keep recording
                failures += 1
                print(f"[{elapsed:6.1f}s] capture failed: {exc}", flush=True)
                cam = http.client.HTTPConnection(args.host, 80, timeout=10)
                idx += 1
                continue
            if idx % args.status_every == 0:
                try:
                    posture = json.loads(get(st, "/status"))
                except Exception:  # noqa: BLE001 - posture is optional
                    st = http.client.HTTPConnection(args.host, 80, timeout=10)
            (frames / f"{idx:05d}.jpg").write_bytes(jpg)
            imu = (posture or {}).get("imu", {})
            wheel = (posture or {}).get("wheel", {})
            w.writerow([idx, f"{time.time():.3f}", f"{elapsed:.3f}", len(jpg),
                        round(imu.get("pitch", 0.0), 2),
                        round(imu.get("roll", 0.0), 2),
                        round(wheel.get("v1", 0.0), 2),
                        round(wheel.get("v2", 0.0), 2),
                        (posture or {}).get("heap", 0), 1])
            saved += 1
            if idx % 50 == 0:
                print(f"[{elapsed:6.1f}s] {saved} frames, {failures} failures",
                      flush=True)
                fh.flush()
            idx += 1

    cam.close()
    st.close()
    print(f"done: {saved} frames -> {out}  ({failures} failures)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
