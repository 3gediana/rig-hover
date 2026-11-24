# -*- coding: utf-8 -*-
"""s2r/tellog.py - full-rate TLM1 telemetry: keep every float, stream to CSV.

`link.UdpTel` keeps a 3 s ring of 17 derived values; identifying a plant or
reconstructing an incident needs all 34 floats for the whole session. The
firmware streams to the *last* client that sent to the telemetry port, so a
session must own exactly one socket: build the Robot around `RawTel` (see
`guarded_session.make_robot`) instead of letting `link.Robot` open its own.

Layout is the one in rig_omni/main/boards/hover/telemetry_stream.h.
"""
from __future__ import annotations

import collections
import socket
import struct
import threading
import time

from link import TLM_BYTES, TLM_MAGIC, Tel, parse_tlm

FIELDS = [
    "roll", "pitch", "yaw", "dq", "pit_gyro", "accel_x", "accel_y", "accel_z",
    "wheel1_vel", "wheel2_vel", "wheel1_x", "wheel2_x", "wheel_x", "wheel_vx",
    "q_head", "target_head_pos",
    "stable_pos", "stable_yaw", "vx", "vyaw", "tor1", "tor2",
    "robot_state", "control_mode",
    "head_des", "vx_des", "vyaw_des", "arb_mode", "emote", "src",
    "feat_age_ms", "n_obj", "theta_img", "danger",
]
NF = len(FIELDS)


class RawTel:
    """Same interface as link.UdpTel (latest/at/window/buf/lock/n) plus every raw packet."""

    def __init__(self, host: str, port: int = 9001, keep_s: float = 60.0, on_pkt=None):
        self.host, self.port = host, port
        self.buf: collections.deque[Tel] = collections.deque(maxlen=int(500 * keep_s))
        self.raw: collections.deque[tuple] = collections.deque(maxlen=int(500 * keep_s))
        self.lock = threading.Lock()
        self.stop = False
        self.n = 0
        self.on_pkt = on_pkt          # called in the receive thread with (recv_t, seq, t_ms, f34)
        self.th = threading.Thread(target=self._loop, daemon=True, name="s2r-rawtel")
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
            if len(pkt) < TLM_BYTES:
                continue
            magic, _ver, _ln, seq, t_ms = struct.unpack_from("<IHHII", pkt, 0)
            if magic != TLM_MAGIC:
                continue
            f = struct.unpack_from("<34f", pkt, 16)
            t = Tel(t=now, roll=f[0], pitch=f[1], yaw=f[2], x1=f[10], x2=f[11], wheel_x=f[12], vx_w=f[13],
                    q=f[14], q_tgt=f[15], stable_pos=f[16], stable_yaw=f[17], tor1=f[20], tor2=f[21],
                    robot_state=int(round(f[22])), src=int(round(f[29])), feat_age=f[30], danger=f[33])
            with self.lock:
                self.buf.append(t)
                self.raw.append((now, int(seq), int(t_ms), f))
                self.n += 1
            if self.on_pkt is not None:
                self.on_pkt(now, int(seq), int(t_ms), f)
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


class TelCsv:
    """Streaming CSV writer fed from the receive thread; flushed once a second."""

    def __init__(self, path: str):
        self.fh = open(path, "w", encoding="utf-8", newline="\n")
        self.fh.write("recv_s,seq,uptime_ms," + ",".join(FIELDS) + "\n")
        self.n = 0
        self._last_flush = 0.0
        self.t0 = time.perf_counter()

    def __call__(self, recv_t: float, seq: int, t_ms: int, f) -> None:
        self.fh.write(f"{recv_t - self.t0:.6f},{seq},{t_ms}," + ",".join(f"{v:.7g}" for v in f) + "\n")
        self.n += 1
        if recv_t - self._last_flush > 1.0:
            self.fh.flush()
            self._last_flush = recv_t

    def close(self) -> None:
        self.fh.flush()
        self.fh.close()


def read_tel_csv(path: str) -> dict[str, list]:
    """Load a TelCsv file into column lists (keys: recv_s, seq, uptime_ms, FIELDS...)."""
    cols: dict[str, list] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        cols = {k: [] for k in header}
        for line in fh:
            v = line.rstrip("\n").split(",")
            if len(v) != len(header):
                continue
            for k, s in zip(header, v):
                cols[k].append(float(s) if k not in ("seq", "uptime_ms") else int(s))
    return cols
