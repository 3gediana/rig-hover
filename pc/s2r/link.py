# -*- coding: utf-8 -*-
"""s2r/link.py - lean, measured real-robot I/O for RIG-Hover (sim2real layer, 2026-11-23).

Every constant here was measured on the real robot (upright, on the desk) - see HW_PROFILE.md.
  * ~490 Hz TLM1 UDP telemetry (incl. per-wheel encoder ticks) - no /status polling on the hot path.
  * frame fetch thread (/capture) with exposure-time telemetry alignment.
  * command thread: firmware ARB_SRC_TEACHER (167 Hz LQR + heading loop) + /feat heartbeat.
  * body heading from WHEEL ENCODERS (drift-free; vision-checked 11.9 deg per cm of wheel difference),
    used to estimate the QMI8658C yaw-gyro bias online, which is fed forward so the firmware heading
    loop stops "turning by itself".
"""
from __future__ import annotations

import collections
import math
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

_PC = Path(__file__).resolve().parents[1]
if str(_PC) not in sys.path:
    sys.path.insert(0, str(_PC))
import robot_truth  # noqa: E402

# ---------------- measured hardware profile ----------------
# Read from pc/robot_truth.json, where each number carries its method, date and evidence. None of them is typed
# in here: this runtime went on the real robot with exactly these values and must keep using them.
F_PX_PER_DEG = robot_truth.PX_PER_DEG   # image px per IMU-degree of camera yaw at the image centre (teacher-mode body turns + head steps)
IMG_W = robot_truth.IMG_W
TICK_M = robot_truth.TICK_M             # firmware's own wheel_x scale (stable_pos/vx use the same unit)
DEG_PER_WDIFF_M = -robot_truth.DEG_PER_WDIFF_M   # body yaw (deg) per metre of (wheel1_x - wheel2_x) in TICK_M units; sign: yaw = -k*wdiff
HEAD_DEADBAND_DEG = robot_truth.HEAD_DEADBAND_DEG   # EM3 servo backlash/stiction measured upright (cmd 12 -> q 5.3, cmd 0 -> q 4.0)
H_CAM_M = robot_truth.CAM_HEIGHT_M

TLM_MAGIC = 0x314D4C54
TLM_BYTES = 152
FEAT_HEARTBEAT = struct.pack("<HHIHHf", 0xFE01, 0, 100, 0, 0, 1.0 / 32767.0)


class Tel:
    __slots__ = ("t", "roll", "pitch", "yaw", "x1", "x2", "wheel_x", "vx_w", "q", "q_tgt", "stable_pos",
                 "stable_yaw", "robot_state", "src", "feat_age", "danger", "tor1", "tor2")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, 0.0))

    @property
    def body_yaw_imu(self) -> float:
        return float(self.yaw) - float(self.q)

    @property
    def wdiff_m(self) -> float:
        return (float(self.x1) - float(self.x2)) * TICK_M

    @property
    def body_yaw_wheel(self) -> float:
        return -DEG_PER_WDIFF_M * self.wdiff_m

    @property
    def x_m(self) -> float:
        return float(self.wheel_x)   # firmware wheel_x, metres (TLM f[12]); same unit as stable_pos


def parse_tlm(pkt: bytes) -> Tel | None:
    if len(pkt) < TLM_BYTES:
        return None
    magic, ver = struct.unpack_from("<IH", pkt, 0)
    if magic != TLM_MAGIC:
        return None
    f = struct.unpack_from("<34f", pkt, 16)
    return Tel(t=time.perf_counter(), roll=f[0], pitch=f[1], yaw=f[2], x1=f[10], x2=f[11],
               wheel_x=f[12], vx_w=f[13], q=f[14], q_tgt=f[15], stable_pos=f[16], stable_yaw=f[17],
               tor1=f[20], tor2=f[21], robot_state=int(round(f[22])), src=int(round(f[29])), feat_age=f[30], danger=f[33])


class UdpTel:
    def __init__(self, host: str, port: int = 9001, keep_s: float = 3.0):
        self.host, self.port = host, port
        self.buf: collections.deque[Tel] = collections.deque(maxlen=int(500 * keep_s))
        self.lock = threading.Lock()
        self.stop = False
        self.n = 0
        self.th = threading.Thread(target=self._loop, daemon=True, name="s2r-udp")
        self.th.start()

    def _loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        last_ping = 0.0
        while not self.stop:
            now = time.perf_counter()
            if now - last_ping > 1.5:
                try:
                    s.sendto(b"hi", (self.host, self.port))
                except Exception:
                    pass
                last_ping = now
            try:
                pkt, _ = s.recvfrom(256)
            except Exception:
                continue
            t = parse_tlm(pkt)
            if t is not None:
                with self.lock:
                    self.buf.append(t)
                    self.n += 1
        s.close()

    def latest(self, max_age: float = 0.3) -> Tel | None:
        with self.lock:
            if not self.buf:
                return None
            t = self.buf[-1]
        return t if time.perf_counter() - t.t <= max_age else None

    def at(self, tq: float) -> Tel | None:
        with self.lock:
            if not self.buf:
                return None
            best = None
            for t in reversed(self.buf):
                best = t
                if t.t <= tq:
                    break
            return best

    def window(self, t0: float, t1: float) -> list[Tel]:
        with self.lock:
            return [t for t in self.buf if t0 <= t.t <= t1]


class FrameGrabber:
    """Continuously pulls /capture; keeps latest frame + exposure-time telemetry."""

    def __init__(self, base: str, udp: UdpTel, q: int = 60, on_frame=None):
        self.base, self.udp, self.q = base, udp, q
        self.s = requests.Session(); self.s.trust_env = False
        self.lock = threading.Lock()
        self.latest = None   # (gray240, bgr240, t_exp, tel_exp, idx)
        self.idx = 0
        self.stop = False
        self.on_frame = on_frame
        self.fps_hist: collections.deque[float] = collections.deque(maxlen=40)
        self.th = threading.Thread(target=self._loop, daemon=True, name="s2r-frames")
        self.th.start()

    def _loop(self):
        url = f"{self.base}/capture?q={self.q}"
        while not self.stop:
            try:
                t_req = time.perf_counter()
                r = self.s.get(url, timeout=1.5)
                t_done = time.perf_counter()
                bgr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                # measured: /capture waits ~10 ms for the next sensor frame, then ~21 ms JPEG + ~9 ms wire
                t_exp = t_req + 0.010
                tel = self.udp.at(t_exp)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                with self.lock:
                    self.idx += 1
                    self.latest = (gray, bgr, t_exp, tel, self.idx)
                self.fps_hist.append(t_done)
                if self.on_frame is not None:
                    try:
                        self.on_frame(gray, bgr, t_exp, tel)
                    except Exception:
                        pass
            except Exception:
                time.sleep(0.05)

    def get(self, after_idx: int = 0, wait: float = 0.3):
        dl = time.perf_counter() + wait
        while time.perf_counter() < dl:
            with self.lock:
                L = self.latest
            if L is not None and L[4] > after_idx:
                return L
            time.sleep(0.002)
        return None

    @property
    def fps(self) -> float:
        h = list(self.fps_hist)
        return (len(h) - 1) / (h[-1] - h[0]) if len(h) > 2 else 0.0


class GyroBias:
    """Online yaw-gyro bias = d(IMU body yaw)/dt - d(wheel body yaw)/dt over a sliding window.
    Valid while the wheels do not slip (in-place turns at <=60 deg/s on the desk mat: checked)."""

    def __init__(self, win_s: float = 4.0):
        self.win_s = win_s
        self.bias = None
        self.hist: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=4000)

    def update(self, tel: Tel) -> float | None:
        self.hist.append((tel.t, tel.body_yaw_imu, tel.body_yaw_wheel))
        t0 = tel.t - self.win_s
        while self.hist and self.hist[0][0] < t0 - 0.5:
            self.hist.popleft()
        if len(self.hist) < 50 or self.hist[-1][0] - self.hist[0][0] < 0.6 * self.win_s:
            return self.bias
        a, b = self.hist[0], self.hist[-1]
        dt = b[0] - a[0]
        est = ((b[1] - a[1]) - (b[2] - a[2])) / dt
        if abs(est) > 15.0:
            return self.bias
        self.bias = est if self.bias is None else 0.9 * self.bias + 0.1 * est
        return self.bias


class Actuator:
    """Teacher-mode command thread (25 Hz) with heartbeat, bias feed-forward and guaranteed safe stop."""

    def __init__(self, base: str, udp: UdpTel, rate_hz: float = 25.0, k_dyaw: float = -0.16):
        self.base, self.udp = base, udp
        self.s = requests.Session(); self.s.trust_env = False
        self.cmd = (0.0, 0.0, 0.0)  # vx m/s (TICK_M units), vyaw deg/s (body, excl. bias), head deg
        self.bias = GyroBias()
        self.bias_dps = 0.0
        self.lock = threading.Lock()
        self.stop = False
        self.rate = rate_hz
        self.enabled = False
        self.sent = 0
        self.errors = 0
        self.last_t_cmd = 0.0
        self.cfg = {"stale_ms": 1400.0, "lost_ms": 2800.0, "cmd_timeout_ms": 600.0, "head_lpf": 0.28,
                    "vyaw_stop_eps": 0.3, "vx_stop_eps": 0.002, "k_dyaw": k_dyaw, "vyaw_rate_max": 300.0}
        self.th = threading.Thread(target=self._loop, daemon=True, name="s2r-act")
        self.th.start()

    def set(self, vx: float = 0.0, vyaw: float = 0.0, head: float = 0.0):
        with self.lock:
            self.cmd = (float(np.clip(vx, -0.12, 0.12)), float(np.clip(vyaw, -70.0, 70.0)), float(np.clip(head, -45.0, 45.0)))
            self.last_t_cmd = time.perf_counter()

    def enable(self, init_bias_dps: float | None = None):
        if init_bias_dps is not None:
            self.bias_dps = float(init_bias_dps)
        r = self.s.post(f"{self.base}/cfg", json=self.cfg, timeout=1.0)
        self.enabled = True
        return r.text

    def _loop(self):
        last_hb = 0.0
        period = 1.0 / self.rate
        while not self.stop:
            t0 = time.perf_counter()
            tel = self.udp.latest()
            if tel is not None:
                b = self.bias.update(tel)
                if b is not None:
                    self.bias_dps = b
            if self.enabled:
                with self.lock:
                    vx, vy, hd = self.cmd
                    stale = (t0 - self.last_t_cmd) > 0.5
                if stale:
                    vx, vy = 0.0, 0.0
                try:
                    if t0 - last_hb > 0.2:
                        self.s.post(f"{self.base}/feat", data=FEAT_HEARTBEAT, timeout=0.4)
                        last_hb = t0
                    self.s.post(f"{self.base}/mode?src=teacher&head={hd:.1f}&vx={vx:.4f}&vyaw={vy + self.bias_dps:.2f}&md=1&em=0",
                                data=b"", timeout=0.4)
                    self.sent += 1
                except Exception:
                    self.errors += 1
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, period - dt))

    def safe_stop(self):
        self.enabled = False
        for _ in range(3):
            try:
                self.s.post(f"{self.base}/mode?src=manual", data=b"", timeout=0.5)
                self.s.get(f"{self.base}/cmd?stop=1&head=0", timeout=0.5)
                return True
            except Exception:
                time.sleep(0.2)
        return False

    def close(self):
        self.safe_stop()
        self.stop = True


class Robot:
    def __init__(self, base: str = "http://192.168.137.50", jpg_q: int = 60, on_frame=None):
        host = base.split("//")[-1].split(":")[0].split("/")[0]
        self.base = base
        self.udp = UdpTel(host)
        self.frames = FrameGrabber(base, self.udp, q=jpg_q, on_frame=on_frame)
        self.act = Actuator(base, self.udp)

    def wait_ready(self, timeout: float = 5.0) -> bool:
        dl = time.perf_counter() + timeout
        while time.perf_counter() < dl:
            if self.udp.latest() is not None and self.frames.get(wait=0.05) is not None:
                return True
            time.sleep(0.05)
        return False

    def rest_pos_lead(self) -> float:
        # balance equilibrium offset: firmware holds stable_pos ahead/behind the wheels at rest (measured 4.65 cm).
        t = self.udp.latest()
        return 0.0 if t is None else float(t.stable_pos) - t.x_m

    def drive_cfg(self, headroom_m: float = 0.05) -> dict:
        # pos_lead cap must sit ON TOP of the rest offset, else the cap is consumed and vx cannot move the robot.
        lead0 = abs(self.rest_pos_lead())
        return {"yaw_lead_max": 18.0, "pos_lead_max": float(min(0.14, lead0 + headroom_m)), "vx_max": 0.08, "vyaw_max": 45.0}

    def rest_bias(self, secs: float = 3.0) -> float:
        """Bias from a stationary window using wheel truth (works whatever mode the robot is in)."""
        t0 = time.perf_counter(); time.sleep(secs); t1 = time.perf_counter()
        w = self.udp.window(t0, t1)
        if len(w) < 20:
            return 0.0
        a, b = w[0], w[-1]
        return ((b.body_yaw_imu - a.body_yaw_imu) - (b.body_yaw_wheel - a.body_yaw_wheel)) / (b.t - a.t)

    def close(self):
        self.act.close()
        self.frames.stop = True
        self.udp.stop = True
