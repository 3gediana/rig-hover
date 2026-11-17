# -*- coding: utf-8 -*-
"""end_side_protocol.py — RIG-Hover 端侧（PC / Android）HTTP 控制协议与 800ms 看门狗守护。

严格对齐 G:/robot_app/PROTOCOL.md：
1. GET /status  -> 28 维本体遥测 (IMU, 轮速, 里程计, 头部角度, 当前指令, 力矩, PID, 堆内存)
2. GET /capture -> 240x240 JPEG 图像帧
3. GET /cmd     -> 控制指令 vx (±100), vyaw (±100), head (±75°), stop=1
4. 800ms 看门狗保护：客户端必须以 ≥2Hz（建议 10-20Hz）刷新 vx/vyaw；
   若端侧推理超时或失联超过 timeout_s (默认 0.40s)，立即将 vx/vyaw 归零。
"""
from __future__ import annotations

import http.client
import json
import socket
import time
from dataclasses import dataclass, field


def _clamp(v: float, lo: float, hi: float) -> float:
    if v != v:  # NaN
        return 0.0
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class DriveCommand:
    vx: float = 0.0
    vyaw: float = 0.0
    head: float = 0.0
    stop: bool = False

    def sanitized(self) -> DriveCommand:
        if self.stop:
            return DriveCommand(vx=0.0, vyaw=0.0, head=_clamp(float(self.head), -75.0, 75.0), stop=True)
        return DriveCommand(
            vx=_clamp(float(self.vx), -100.0, 100.0),
            vyaw=_clamp(float(self.vyaw), -100.0, 100.0),
            head=_clamp(float(self.head), -75.0, 75.0),
            stop=False,
        )

    def to_query_path(self) -> str:
        c = self.sanitized()
        if c.stop:
            return f"/cmd?stop=1&head={c.head:.1f}"
        return f"/cmd?vx={c.vx:.1f}&vyaw={c.vyaw:.1f}&head={c.head:.1f}"


@dataclass
class RobotTelemetry:
    t_ms: int = 0
    # IMU (8)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    pit_gyro: float = 0.0
    dq: float = 0.0
    # Wheel (4)
    v1: float = 0.0
    v2: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    # Head (3)
    head_q: float = 0.0
    head_target: float = 0.0
    gimbal_target: float = 0.0
    # Cmd (5)
    cmd_vx: float = 0.0
    cmd_vyaw: float = 0.0
    cmd_mode: int = 0
    cmd_pos: float = 0.0
    cmd_yaw_target: float = 0.0
    # Torque (2)
    t1: float = 0.0
    t2: float = 0.0
    # PID (6)
    pos_e: float = 0.0
    pos_u: float = 0.0
    vel_e: float = 0.0
    vel_u: float = 0.0
    pit_e: float = 0.0
    pit_u: float = 0.0
    heap: int = 0

    @property
    def wheel_x_avg(self) -> float:
        return 0.5 * (self.x1 + self.x2)


def parse_status_json(data: dict) -> RobotTelemetry:
    imu = data.get("imu") or {}
    wheel = data.get("wheel") or {}
    head = data.get("head") or {}
    cmd = data.get("cmd") or {}
    torque = data.get("torque") or {}
    pid = data.get("pid") or {}
    return RobotTelemetry(
        t_ms=int(data.get("t", 0)),
        roll=float(imu.get("roll", 0.0)),
        pitch=float(imu.get("pitch", 0.0)),
        yaw=float(imu.get("yaw", 0.0)),
        ax=float(imu.get("ax", 0.0)),
        ay=float(imu.get("ay", 0.0)),
        az=float(imu.get("az", 0.0)),
        pit_gyro=float(imu.get("pit_gyro", 0.0)),
        dq=float(imu.get("dq", 0.0)),
        v1=float(wheel.get("v1", 0.0)),
        v2=float(wheel.get("v2", 0.0)),
        x1=float(wheel.get("x1", 0.0)),
        x2=float(wheel.get("x2", 0.0)),
        head_q=float(head.get("q", 0.0)),
        head_target=float(head.get("target", 0.0)),
        gimbal_target=float(head.get("gimbal_target", 0.0)),
        cmd_vx=float(cmd.get("vx", 0.0)),
        cmd_vyaw=float(cmd.get("vyaw", 0.0)),
        cmd_mode=int(cmd.get("mode", 0)),
        cmd_pos=float(cmd.get("pos", 0.0)),
        cmd_yaw_target=float(cmd.get("yaw_target", 0.0)),
        t1=float(torque.get("t1", 0.0)),
        t2=float(torque.get("t2", 0.0)),
        pos_e=float(pid.get("pos_e", 0.0)),
        pos_u=float(pid.get("pos_u", 0.0)),
        vel_e=float(pid.get("vel_e", 0.0)),
        vel_u=float(pid.get("vel_u", 0.0)),
        pit_e=float(pid.get("pit_e", 0.0)),
        pit_u=float(pid.get("pit_u", 0.0)),
        heap=int(data.get("heap", 0)),
    )


class EndSideWatchdog:
    """端侧 800ms 看门狗守护器。

    固件要求至少每 800ms 收到一次 vx/vyaw 指令，否则自动归零。
    同时若端侧推理模块超过 timeout_s (默认 0.40s) 未刷新指令，端侧主动将 vx/vyaw 归零，
    防止端侧视觉卡顿时机器人保持旧速度冲出桌面。
    """

    def __init__(self, timeout_s: float = 0.40):
        self.timeout_s = float(timeout_s)
        self.last_cmd = DriveCommand()
        self.last_update_s = -1e9
        self.estop_locked = False

    def update_command(self, cmd: DriveCommand, now_s: float | None = None) -> None:
        t = time.monotonic() if now_s is None else float(now_s)
        self.last_cmd = cmd.sanitized()
        self.last_update_s = t

    def trigger_estop(self, lock: bool = True) -> DriveCommand:
        self.estop_locked = bool(lock)
        self.last_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=self.last_cmd.head, stop=True)
        return self.last_cmd

    def get_safe_command(self, now_s: float | None = None) -> DriveCommand:
        t = time.monotonic() if now_s is None else float(now_s)
        if self.estop_locked:
            return DriveCommand(vx=0.0, vyaw=0.0, head=self.last_cmd.head, stop=True)
        if t - self.last_update_s > self.timeout_s:
            return DriveCommand(vx=0.0, vyaw=0.0, head=self.last_cmd.head, stop=False)
        return self.last_cmd


class HttpRobotClient:
    """长连接 HTTP/1.1 + TCP_NODELAY 机器人端侧客户端（严禁走系统代理）。"""

    def __init__(self, host: str = "192.168.137.50", port: int = 80, timeout: float = 0.5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._conn: http.client.HTTPConnection | None = None

    def _ensure_conn(self) -> http.client.HTTPConnection:
        if self._conn is None:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            conn.connect()
            if conn.sock is not None:
                conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _request(self, method: str, path: str) -> bytes:
        for attempt in range(2):
            try:
                conn = self._ensure_conn()
                conn.request(method, path, headers={"Connection": "keep-alive"})
                resp = conn.getresponse()
                body = resp.read()
                if resp.status == 200:
                    return body
                raise RuntimeError(f"HTTP {resp.status} on {path}")
            except Exception:
                self.close()
                if attempt == 1:
                    raise
        return b""

    def get_status(self) -> RobotTelemetry:
        raw = self._request("GET", "/status")
        return parse_status_json(json.loads(raw.decode("utf-8")))

    def get_capture(self) -> bytes:
        return self._request("GET", "/capture")

    def send_cmd(self, cmd: DriveCommand) -> bool:
        raw = self._request("GET", cmd.to_query_path())
        data = json.loads(raw.decode("utf-8"))
        return bool(data.get("ok", False))
