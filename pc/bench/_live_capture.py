"""Capture a labelled burst of frames from the robot and keep them on disk.

Used for the physical-only measurements that need a human in the loop: A4-sheet
FOV calibration, face/fist size-vs-distance, and accuracy spot checks. Every
frame is kept because the analysis happens offline and a discarded frame is
unrecoverable.

Usage:
    python bench/_live_capture.py --label a4_1.0m --n 15
    python bench/_live_capture.py --label fist_1.3m --n 15 --root evidence/live
"""

from __future__ import annotations

import argparse
import http.client
import statistics
import sys
import time
from pathlib import Path

import numpy as np

BASE = "192.168.137.50"
ROOT = Path(__file__).resolve().parents[2] / "evidence" / "live"


def connect(host: str = BASE, port: int = 80) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(host, port, timeout=10)


def http_get(conn: http.client.HTTPConnection, path: str) -> bytes:
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read()
    if resp.status != 200:
        raise RuntimeError(f"{path} -> HTTP {resp.status}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="folder name for this burst")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--interval", type=float, default=0.12,
                    help="seconds between frames")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--host", default=BASE)
    args = ap.parse_args()

    import cv2  # noqa: PLC0415 - only needed for the summary

    out = Path(args.root) / args.label
    out.mkdir(parents=True, exist_ok=True)

    conn = connect(args.host)
    try:
        http_get(conn, "/status")
    except Exception as exc:  # noqa: BLE001
        print(f"robot unreachable: {exc}", file=sys.stderr)
        return 1

    sizes: list[int] = []
    means: list[float] = []
    for i in range(args.n):
        jpg = http_get(conn, "/capture")
        path = out / f"{i:03d}.jpg"
        path.write_bytes(jpg)
        sizes.append(len(jpg))
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            means.append(float(img.mean()))
        time.sleep(args.interval)
    conn.close()

    print(f"saved {args.n} frames -> {out}")
    print(f"jpeg size  mean {statistics.mean(sizes):.0f} B  "
          f"min {min(sizes)}  max {max(sizes)}")
    if means:
        print(f"brightness mean {statistics.mean(means):.1f}  "
              f"min {min(means):.1f}  max {max(means):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
