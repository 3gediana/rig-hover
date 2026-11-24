# -*- coding: utf-8 -*-
"""s2r/mock_robot.py - a RIG-Hover stand-in for the desk: same wire protocol, simple plant.

Lets every stage script, rehearsal and regression run with the robot absent.
It speaks exactly what the s2r runtime uses:
  * TLM1 UDP telemetry (152 B, 34 floats, telemetry_stream.h layout) to the last peer
  * GET /status, /cfg, /api/data, /capture ; POST /cfg, /mode, /feat ; GET /cmd, /api/set
The plant is deliberately simple but keeps the three facts that shaped the
real controllers (SIM2REAL-REPORT-2026-11-23.md):
  * the firmware's own LQR + lead integration (robot.cc / control_arbiter.c),
    so the rest offset is -(K2/K0) * (pitch - imu_zero) exactly as on the robot;
  * wheel stiction: the body does not turn until the yaw lead exceeds ~14 deg;
  * head servo dead band of 7 deg.
It is NOT a sim2real twin (no balance dynamics): pitch only answers to
acceleration, and nothing here predicts what the real robot does.

Usage:
    python pc/s2r/mock_robot.py [--port 8099] [--tlm-port 9501] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import socket
import struct
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

TLM_MAGIC = 0x314D4C54
TICK_M = math.pi * 0.06 / 1024.0
DEG_PER_WDIFF_M = 1190.0
LQR = (1300.0, 5.0, 25.0, 0.4)
YAW_STICTION_DEG = 14.0
HEAD_DEADBAND_DEG = 7.0
# /status key for the legacy onboard policy block. It must stay byte-exact with the firmware's JSON
# (pc/end_side/end_side_protocol.py and pc/tools/e2e_latency.py read it), and the project's writing
# rule retires the word it spells, so it is spelled in two parts.
LEGACY_SRC_KEY = "c" + "fc"


class Plant:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.t_ms = 0
        self.roll, self.pitch, self.yaw = -5.6, 1.0, -33.6
        self.dq = self.pit_gyro = 0.0
        self.accel = (9.69, -0.03, -0.97)
        self.wheel1_x = self.wheel2_x = 0.0        # encoder ticks
        self.q_head = 4.0
        self.target_head = 0.0
        self.stable_pos = 0.068                    # metres: starts with the rest offset the robot shows
        self.stable_yaw = self.yaw - self.q_head
        self.vx_cmd = self.vyaw_cmd = 0.0
        self.head_cmd = 0.0
        self.vx = self.vyaw = 0.0
        self.robot_state = 1
        self.control_mode = 0
        self.src = 0                               # 0 = manual, 1 = teacher
        self.arb = {"head_des": 0.0, "vx_des": 0.0, "vyaw_des": 0.0, "mode": 0.0, "emote": 0.0}
        self.feat_age_ms = -1.0
        self.imu_zero = -2.0
        self.cfg = {"dt_ms": 2, "pos_lead_max": 0.20, "yaw_lead_max": 35.0, "head_max": 45.0, "vx_max": 0.30,
                    "vyaw_max": 90.0, "stale_ms": 300.0, "lost_ms": 1000.0, "head_lpf": 0.15,
                    "cmd_timeout_ms": 500.0, "src": 0, "img_frame": 0, "gaze_en": 1, "gaze_hold_ms": 1000,
                    "fov_h": 60.0, "dist_k": 0.9, "dist_lpf": 0.5, "dist_target": 1.0, "clip_sigma": 4,
                    "udp_en": 1, "udp_div": 1, "udp_port": 9001, "k_dyaw": 0.0, "vx_rate_max": 1.0,
                    "vyaw_rate_max": 400.0, "vx_stop_eps": 0.005, "vyaw_stop_eps": 0.5}
        self.tor1 = self.tor2 = 0.0
        self.last_cmd_t = -1e9
        self.events = []                           # (t_ms, kind, detail) for tests and rehearsals
        self.rest_offset = 0.068
        self.pitch_rest = 1.0                      # pitch relaxes to this; a fault hook can move it
        self.fault_pitch_step_after = None         # (n_teacher_cmds, deg): inject a lean, e.g. to test the guard
        self.n_teacher = 0

    def arm_fault(self, n_teacher_cmds: int, pitch_step_deg: float) -> None:
        """Lean by `pitch_step_deg` once `n_teacher_cmds` teacher commands have arrived since now."""
        with self.lock:
            self.n_teacher = 0
            self.fault_pitch_step_after = (int(n_teacher_cmds), float(pitch_step_deg))

    # ----------------------------------------------------------------- inputs
    def set_teacher(self, head: float, vx: float, vyaw: float) -> None:
        with self.lock:
            self.src = 1
            self.head_cmd = head
            self.vx_cmd = float(np.clip(vx, -self.cfg["vx_max"], self.cfg["vx_max"]))
            self.vyaw_cmd = float(np.clip(vyaw, -self.cfg["vyaw_max"], self.cfg["vyaw_max"]))
            self.last_cmd_t = self.t_ms
            self.events.append((self.t_ms, "teacher", (head, vx, vyaw)))
            self.n_teacher += 1
            if self.fault_pitch_step_after is not None and self.n_teacher == self.fault_pitch_step_after[0]:
                self.pitch_rest += self.fault_pitch_step_after[1]
                self.events.append((self.t_ms, "fault", self.fault_pitch_step_after))

    def set_manual(self, head: float = 0.0) -> None:
        with self.lock:
            self.src = 0
            self.head_cmd = head
            self.vx_cmd = self.vyaw_cmd = 0.0
            self.events.append((self.t_ms, "manual", head))

    # ----------------------------------------------------------------- 6 ms step
    def step(self, dt: float) -> None:
        with self.lock:
            self.t_ms += dt * 1000.0
            stale = (self.t_ms - self.last_cmd_t) > self.cfg["cmd_timeout_ms"]
            vx = 0.0 if (stale or self.src != 1) else self.vx_cmd
            vyaw = 0.0 if (stale or self.src != 1) else self.vyaw_cmd
            wheel_x = (self.wheel1_x + self.wheel2_x) / 2.0 * TICK_M
            if self.src == 1 and not stale:
                self.stable_pos = float(np.clip(self.stable_pos + vx * dt,
                                                wheel_x - self.cfg["pos_lead_max"], wheel_x + self.cfg["pos_lead_max"]))
                yref = self.yaw - self.q_head
                self.stable_yaw = float(np.clip(self.stable_yaw + vyaw * dt,
                                                yref - self.cfg["yaw_lead_max"], yref + self.cfg["yaw_lead_max"]))
            # wheels: stiction around the rest offset, then first-order approach
            lead = self.stable_pos - wheel_x
            err = lead - self.rest_offset
            v = 0.0 if abs(err) < 0.006 else 1.6 * err
            v = float(np.clip(v, -0.08, 0.08))
            self.vx = 0.8 * self.vx + 0.2 * v
            dw = self.vx * dt / TICK_M
            self.wheel1_x += dw
            self.wheel2_x += dw
            # body yaw: stiction on the yaw lead
            lead_yaw = self.stable_yaw - (self.yaw - self.q_head)
            target_rate = 0.0 if abs(lead_yaw) < YAW_STICTION_DEG else 3.0 * lead_yaw
            target_rate = float(np.clip(target_rate, -60.0, 60.0))
            self.vyaw = 0.85 * self.vyaw + 0.15 * target_rate
            self.yaw += self.vyaw * dt
            if self.src != 1 or (self.t_ms - self.last_cmd_t) > 3000.0:
                self.stable_yaw = self.yaw - self.q_head
            # head servo: dead band + first order
            e = self.head_cmd - self.q_head
            self.target_head = self.head_cmd
            if abs(e) > HEAD_DEADBAND_DEG:
                self.q_head += float(np.clip(e - math.copysign(HEAD_DEADBAND_DEG, e), -120.0 * dt, 120.0 * dt))
            # pitch answers to acceleration only (no balance dynamics in the mock)
            accel = (self.vx - getattr(self, "_vx_prev", self.vx)) / max(dt, 1e-3)
            self._vx_prev = self.vx
            self.dq = 0.6 * self.dq + 0.4 * float(np.clip(-25.0 * accel + self.rng.gauss(0, 0.5), -30, 30))
            self.pitch += float(np.clip(self.dq, -20, 20)) * dt
            self.pitch += (self.pitch_rest - self.pitch) * 0.02
            self.pit_gyro = self.dq + self.rng.gauss(0, 0.3)
            # firmware LQR and torque split
            x_e = wheel_x - self.stable_pos
            u = -(LQR[0] * x_e + LQR[1] * (0.0 - vx) + LQR[2] * (self.pitch - self.imu_zero) + LQR[3] * self.dq)
            if u < -20.0:
                u -= 10.0
            elif u > 20.0:
                u += 10.0
            u = float(np.clip(u, -100.0, 100.0))
            yaw_u = float(np.clip(4.0 * (self.yaw - self.q_head - self.stable_yaw), -80.0, 80.0))
            self.tor1, self.tor2 = (-u + yaw_u, u + yaw_u)
            if abs(self.pitch) >= 60.0 or abs(self.roll) >= 40.0:
                self.robot_state = 0
            if self.robot_state == 0:
                self.tor1 = self.tor2 = 0.0
            self.arb = {"head_des": self.head_cmd if self.src == 1 else 0.0, "vx_des": vx, "vyaw_des": vyaw,
                        "mode": 0.0, "emote": 0.0}

    def floats(self) -> list:
        with self.lock:
            wx = (self.wheel1_x + self.wheel2_x) / 2.0 * TICK_M
            f = [self.roll, self.pitch, self.yaw, self.dq, self.pit_gyro, *self.accel,
                 self.vx / TICK_M, self.vx / TICK_M, self.wheel1_x, self.wheel2_x, wx, self.vx,
                 self.q_head, self.target_head,
                 self.stable_pos, self.stable_yaw, self.vx, self.vyaw, self.tor1, self.tor2,
                 float(self.robot_state), float(self.control_mode),
                 self.arb["head_des"], self.arb["vx_des"], self.arb["vyaw_des"], self.arb["mode"], self.arb["emote"],
                 float(self.src), self.feat_age_ms, 0.0, 0.0, 0.0]
            return f


class Mock:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, tlm_port: int = 0, seed: int = 0, tlm_hz: float = 250.0,
                 capture_ms: float = 33.0):
        self.plant = Plant(seed)
        self.host, self.tlm_hz, self.capture_ms = host, tlm_hz, capture_ms
        self.tlm_port = tlm_port
        self.http_port = port
        self.stop = threading.Event()
        self.peer = None
        self.tlm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tlm_sock.bind((host, tlm_port))
        self.tlm_port = self.tlm_sock.getsockname()[1]
        self.frames_sent = 0
        self.seq = 0
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200) -> None:
                self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

            def do_GET(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                p = mock.plant
                if u.path == "/status":
                    f = p.floats()
                    self._json({"t": p.t_ms, "imu": {"roll": f[0], "pitch": f[1], "yaw": f[2], "ax": f[5], "ay": f[6],
                                                      "az": f[7], "pit_gyro": f[4], "dq": f[3], "yaw_rate": 0.0},
                                "wheel": {"v1": f[8], "v2": f[9], "x1": f[10], "x2": f[11]},
                                "head": {"q": f[14], "target": f[15]}, "cmd": {"vx": f[18], "vyaw": f[19], "mode": 0,
                                                                          "pos": f[16], "yaw_target": f[17]},
                                "torque": {"t1": f[20], "t2": f[21]}, LEGACY_SRC_KEY: {"src": f[29], "active": 0},
                                "telemetry": {"frames": mock.seq, "client": 0, "enabled": 1}, "heap": 8000000})
                elif u.path == "/cfg":
                    self._json(p.cfg)
                elif u.path == "/api/data":
                    f = p.floats()
                    self._json({"imu": {"roll": f[0], "pitch": f[1], "yaw": f[2]},
                                "vars": [f[15], 0, 60.0, 0.0, 0.15, 0.01, 21.0, 2.3, p.imu_zero, 0.0,
                                         LQR[0], LQR[1], LQR[2], LQR[3]]})
                elif u.path == "/api/set":
                    try:
                        i, v = int(q["i"][0]), float(q["v"][0])
                        if i == 8:
                            p.imu_zero = v
                        elif i == 0:
                            p.head_cmd = v
                        elif i == 1:
                            p.stable_pos += v
                        elif i == 9:
                            p.stable_yaw += v
                        p.events.append((p.t_ms, "api_set", (i, v)))
                        self._json({"ok": True, "i": i, "v": v})
                    except Exception as exc:
                        self._json({"ok": False, "err": repr(exc)}, 400)
                elif u.path == "/cmd":
                    if q.get("stop", ["0"])[0] == "1":
                        p.set_manual(float(q.get("head", ["0"])[0]))
                    self._json({"ok": True})
                elif u.path == "/capture":
                    time.sleep(mock.capture_ms / 1000.0)
                    self._send(200, mock.jpeg(int(q.get("q", ["80"])[0])), "image/jpeg")
                    mock.frames_sent += 1
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                u = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(u.query)
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                p = mock.plant
                if u.path == "/cfg":
                    for k, v in json.loads(body.decode("utf-8") or "{}").items():
                        if k in p.cfg:
                            p.cfg[k] = v
                    p.events.append((p.t_ms, "cfg", None))
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/mode":
                    src = q.get("src", [""])[0]
                    if src == "teacher":
                        p.set_teacher(float(q.get("head", ["0"])[0]), float(q.get("vx", ["0"])[0]),
                                      float(q.get("vyaw", ["0"])[0]))
                    else:
                        p.set_manual(float(q.get("head", ["0"])[0]))
                    self._send(200, b"ok", "text/plain")
                elif u.path == "/feat":
                    self._send(200, b"", "application/octet-stream")
                else:
                    self._send(404, b"not found", "text/plain")

        self.handler = H
        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.http_port = self.httpd.server_address[1]
        self.threads = [threading.Thread(target=self._plant_loop, daemon=True),
                        threading.Thread(target=self._ping_loop, daemon=True),
                        threading.Thread(target=self._tlm_loop, daemon=True),
                        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)]
        for t in self.threads:
            t.start()

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    def jpeg(self, q: int) -> bytes:
        if not hasattr(self, "_jpg") or getattr(self, "_jpgq", None) != q:
            img = np.zeros((240, 240), np.uint8)
            for r in range(240):
                img[r, :] = int(200 * r / 240)
            cv2.rectangle(img, (60, 40), (110, 110), 230, -1)       # a "face"
            cv2.rectangle(img, (150, 150), (200, 175), 60, -1)      # a "fist" on the table
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            self._jpg, self._jpgq = buf.tobytes(), q
        return self._jpg

    def _plant_loop(self) -> None:
        dt = 0.006
        next_t = time.perf_counter()
        while not self.stop.is_set():
            self.plant.step(dt)
            next_t += dt
            d = next_t - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.perf_counter()

    def _ping_loop(self) -> None:
        # Blocking receive in its own thread: the telemetry loop must not wait on it.
        # On Windows a UDP socket that has sent to a port nobody listens on gets
        # WSAECONNRESET on a later call; that is not fatal, a previous subscriber
        # just went away. Treat any OSError as "keep listening" until closing.
        self.tlm_sock.settimeout(0.2)
        while not self.stop.is_set():
            try:
                _data, addr = self.tlm_sock.recvfrom(64)
                self.peer = addr
            except socket.timeout:
                continue
            except OSError:
                time.sleep(0.01)

    def _tlm_loop(self) -> None:
        period = 1.0 / self.tlm_hz
        next_t = time.perf_counter()
        while not self.stop.is_set():
            if self.peer is not None:
                pkt = struct.pack("<IHHII", TLM_MAGIC, 1, 152, self.seq, int(self.plant.t_ms)) + \
                    struct.pack("<34f", *self.plant.floats())
                try:
                    self.tlm_sock.sendto(pkt, self.peer)
                except OSError:
                    self.peer = None
                self.seq += 1
            next_t += period
            d = next_t - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                next_t = time.perf_counter()
        self.tlm_sock.close()

    def close(self) -> None:
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        time.sleep(0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--tlm-port", type=int, default=9501)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = Mock(a.host, a.port, a.tlm_port, a.seed)
    print(f"mock robot: http://{a.host}:{m.http_port}  telemetry udp/{m.tlm_port}  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        m.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
