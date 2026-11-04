#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mock_robot.py -- stand-in for the RIG-Hover HTTP endpoints, for rehearsal.

The morning's critical path is `session_from_robot.py --source robot`, but that
code path (HTTP client, JPEG decode, /status parsing, proprio mapping) can only
be exercised against a live robot. This server reproduces just enough of the
contract to run it end to end while the robot is off, so the recording run does
not discover a bug in its own plumbing at 8 a.m.

It serves:
    GET /capture   a real JPEG, cycled from a folder of real photos
    GET /status    the JSON fields `proprio_from_status` reads

Run:  python mock_robot.py --port 8099
Then: python session_from_robot.py --source robot --host 127.0.0.1:8099 ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_DIR = os.path.join(r"D:\robot_work\pc\gesture\data\raw\hagrid",
                           r"hagrid-sample-30k-384p\hagrid_30k\train_val_palm")


class State:
    def __init__(self, folder: str, fps: float):
        self.frames = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                self.frames.append(os.path.join(folder, name))
        if not self.frames:
            raise SystemExit(f"no images under {folder}")
        self.i = 0
        self.t0 = time.time()
        self.fps = fps
        self.lock = threading.Lock()

    def next_jpeg(self) -> bytes:
        with self.lock:
            path = self.frames[self.i % len(self.frames)]
            self.i += 1
        with open(path, "rb") as f:
            return f.read()

    def status(self) -> dict:
        # Values drift a little so the observation is not accidentally constant,
        # which would make the rehearsal pass for the wrong reason.
        t = time.time() - self.t0
        return {
            "imu": {"roll": 0.4 * (t % 3.0) - 0.6, "pitch": -2.1 + 0.3 * (t % 2.0),
                    "pit_gyro": 1.7 * (t % 1.1) - 0.9, "dq": 0.0, "yaw": 0.2 * (t % 5.0)},
            "wheel": {"v1": 1.3 * (t % 1.7) - 1.1, "v2": 1.1 * (t % 1.9) - 1.0},
            "head": {"q": 18.0 * (t % 4.0) / 4.0 - 9.0, "target": 0.5},
            "cmd": {"pos": 0.12, "yaw_target": 0.0, "vx": 0.04, "vyaw": -3.2, "mode": 1},
            "cfc": {"pos": 0.12, "feat_age_ms": 37, "danger": 0},
        }


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the rehearsal output clean
            pass

        def _send(self, code, body: bytes, ctype="application/octet-stream"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/capture"):
                self._send(200, state.next_jpeg(), "image/jpeg")
            elif self.path.startswith("/status"):
                body = json.dumps(state.status()).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--folder", default=DEFAULT_DIR)
    ap.add_argument("--fps", type=float, default=8.0,
                    help="throttle /capture to this rate, like the real board")
    a = ap.parse_args()

    state = State(a.folder, a.fps)
    print(f"mock robot on http://127.0.0.1:{a.port}  "
          f"({len(state.frames)} photos from {os.path.basename(a.folder)})")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
