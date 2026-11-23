# -*- coding: utf-8 -*-
"""s2r/controllers.py - stiction-aware, torque-bounded controllers identified on the real robot (2026-11-23).

Why the old loops failed on hardware (measured, see SIM2REAL-REPORT-2026-11-23.md):
  * firmware heading loop is P-only (k_yaw=4) against wheel/mat stiction: the body does not move until the
    heading lead reaches ~14 deg, then breaks free -> integrating PC loops pile up lead and overshoot
    (20 deg command -> 33.7 deg on the robot, then a hard reversal; the robot had to be knocked over).
  * the lead also saturates yaw torque (yaw_u clip 80) which steals torque from balance.
Design rules here:
  1. never integrate error into the firmware reference blindly: control the *lead* L = stable_yaw - yaw_imu
     (bias-invariant) towards a bounded desired lead;
  2. friction is beaten by a bounded, slowly growing offset that resets the moment the body moves;
  3. firmware yaw_lead_max / pos_lead_max are cut at runtime (16 deg / 5 cm) so torque stays bounded
     even if this code misbehaves.
"""
from __future__ import annotations

import math

import numpy as np

F_PX_PER_DEG = 2.705


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class HeadingServo:
    """Drives body heading (wheel-odometry frame, drift free) to a target; output = vyaw for the firmware."""

    def __init__(self, k_lead: float = 4.0, vmax: float = 40.0, tol: float = 1.5, c0: float = 3.0,
                 c_rate: float = 8.0, c_max: float = 10.0, lead_cap: float = 16.0, acc_max: float = 160.0):
        self.k_lead, self.vmax, self.tol = k_lead, vmax, tol
        self.c0, self.c_rate, self.c_max, self.lead_cap, self.acc_max = c0, c_rate, c_max, lead_cap, acc_max
        self.reset()

    def reset(self):
        self.c = 0.0
        self.sgn = 0
        self.prev_psi = None
        self.prev_t = None
        self.v_out = 0.0
        self.target = None

    def step(self, t: float, psi_wheel: float, lead_fw: float, target: float | None) -> tuple[float, dict]:
        """psi_wheel: body heading from wheels (deg); lead_fw: stable_yaw - (yaw_imu - q_head) (deg)."""
        dt = 0.04 if self.prev_t is None else max(1e-3, t - self.prev_t)
        rate = 0.0 if self.prev_psi is None else (psi_wheel - self.prev_psi) / dt
        self.prev_psi, self.prev_t = psi_wheel, t
        if target is None:
            want_lead = 0.0
            e = 0.0
            self.c = 0.0
        else:
            e = wrap(target - psi_wheel)
            if abs(e) <= self.tol:
                self.c = 0.0
                want_lead = e
            else:
                s = 1 if e > 0 else -1
                if s != self.sgn:
                    self.c = self.c0
                self.sgn = s
                if abs(rate) < 2.0:
                    self.c = min(self.c_max, self.c + self.c_rate * dt)
                else:
                    self.c = max(self.c0, self.c * 0.9)
                want_lead = e + s * min(self.c, abs(e) + self.c0)
            want_lead = float(np.clip(want_lead, -self.lead_cap, self.lead_cap))
        v = float(np.clip(self.k_lead * (want_lead - lead_fw), -self.vmax, self.vmax))
        dv = float(np.clip(v - self.v_out, -self.acc_max * dt, self.acc_max * dt))
        self.v_out += dv
        return self.v_out, {"e": round(e, 2), "lead": round(lead_fw, 2), "want_lead": round(want_lead, 2), "c": round(self.c, 2)}


class BodyGaze:
    """Face/fist centering by turning the BODY (the head servo has a measured +-7 deg dead band and its
    reaction torque spins the body).  Absolute bearing at exposure time -> latency does not cause hunting."""

    def __init__(self, deadband_deg: float = 2.2, hold_lost_s: float = 1.2):
        self.deadband, self.hold_lost_s = deadband_deg, hold_lost_s
        self.hist: list[float] = []
        self.target: float | None = None
        self.last_seen = -1e9

    def update(self, t: float, cx_px: float | None, psi_wheel_cap: float, q_cap: float, q_now: float,
               psi_wheel_now: float) -> float | None:
        if cx_px is not None:
            err = (float(cx_px) - 120.0) / F_PX_PER_DEG
            face_dir = psi_wheel_cap + q_cap + err          # camera = body + head
            self.hist = (self.hist + [face_dir])[-3:]
            fd = float(np.median(self.hist))
            want = fd - q_now
            if self.target is None or abs(wrap(want - self.target)) > self.deadband or abs(wrap(want - psi_wheel_now)) > self.deadband:
                self.target = want
            self.last_seen = t
        elif t - self.last_seen > self.hold_lost_s:
            self.hist = []
            self.target = psi_wheel_now  # freeze where we are; never chase stale bearings
        return self.target


class GotoPoint:
    """Turn-then-drive to a point in the wheel-odometry frame, bounded speed, stops short by `standoff`."""

    def __init__(self, v_max: float = 0.05, standoff: float = 0.03, turn_tol: float = 5.0, pos_tol: float = 0.02,
                 lead_cap: float = 0.04):
        self.v_max, self.standoff, self.turn_tol, self.pos_tol, self.lead_cap = v_max, standoff, turn_tol, pos_tol, lead_cap
        self.heading = HeadingServo()
        self.phase = "idle"
        self.goal = None
        self.c = 0.0

    def set_goal(self, x: float, y: float):
        self.goal = (float(x), float(y)); self.phase = "turn"; self.heading.reset(); self.c = 0.0

    def step(self, t: float, x: float, y: float, psi_w: float, lead_yaw: float, lead_pos: float) -> tuple[float, float, dict]:
        """x,y: odometry position (m); lead_pos = stable_pos - wheel_x (m). Returns (vx m/s, vyaw deg/s, dbg)."""
        if self.goal is None or self.phase in ("idle", "done"):
            vy, _ = self.heading.step(t, psi_w, lead_yaw, None)
            return float(np.clip(-2.0 * lead_pos, -0.02, 0.02)), vy, {"phase": self.phase}
        gx, gy = self.goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        bear = math.degrees(math.atan2(dy, dx))
        along = dist - self.standoff
        e_head = wrap(bear - psi_w)
        if self.phase == "turn":
            vy, hd = self.heading.step(t, psi_w, lead_yaw, bear)
            vx = float(np.clip(-2.0 * lead_pos, -0.02, 0.02))
            if abs(e_head) < self.turn_tol and abs(lead_yaw) < 6.0:
                self.phase = "drive"
            return vx, vy, {"phase": "turn", "e_head": round(e_head, 1), "dist": round(dist, 3)}
        # drive: hold heading toward the goal (re-aim only while far), bounded position lead
        hold = bear if dist > 0.08 else self.heading.target
        vy, _ = self.heading.step(t, psi_w, lead_yaw, hold)
        if along <= self.pos_tol or abs(e_head) > 60.0:
            self.phase = "done"
            return float(np.clip(-2.0 * lead_pos, -0.02, 0.02)), vy, {"phase": "done", "dist": round(dist, 3)}
        s = 1.0
        if abs(lead_pos) < 0.004 and along > self.pos_tol:
            self.c = min(0.02, self.c + 0.01 * 0.04)
        want = min(along, 0.03) * math.cos(math.radians(e_head)) + self.c
        want = float(np.clip(want, -self.lead_cap, self.lead_cap))
        vx = float(np.clip(1.5 * (want - lead_pos), -self.v_max, self.v_max))
        return vx, vy, {"phase": "drive", "dist": round(dist, 3), "along": round(along, 3), "want_lead": round(want, 3)}


def pixel_to_table(u: float, v: float, pitch_cam_deg: float, h_cam: float = 0.135, f_px: float = F_PX_PER_DEG * 57.2958,
                   cx: float = 120.0, cy: float = 120.0) -> tuple[float, float] | None:
    """Flat-table projection (camera frame: x fwd, y left). Returns None above the horizon/too far.
    pitch_cam_deg: optical-axis tilt below horizontal (positive = looking down)."""
    ang_down = math.radians(pitch_cam_deg) + math.atan2(v - cy, f_px)
    if ang_down <= math.radians(4.0):
        return None
    d = h_cam / math.tan(ang_down)
    lat = -(u - cx) / f_px * math.hypot(d, h_cam)
    return d, lat


class HeadServo:
    """Port of real_robot_bridge.CalibratedGazeServo head path (stiction preload + absolute exposure-time goal),
    with the measured FOV (75.5 deg, not 62.5). Body turning is delegated to HybridGaze."""

    def __init__(self, fov_h_deg: float = 240.0 / F_PX_PER_DEG, head_max: float = 44.0, center_tol: float = 0.022):
        self.fov, self.head_max, self.center_tol = fov_h_deg, head_max, center_tol
        self.gh: list[float] = []; self.qf = None; self.qi = 0.0; self.ls = 0; self.hold = 0.0
        self.qp = None; self.tp = None; self.dq = 0.0

    def vel(self, q: float, t: float) -> float:
        if self.qp is not None and t - self.tp >= 0.004:
            self.dq = 0.62 * self.dq + 0.38 * float(np.clip((q - self.qp) / (t - self.tp), -220, 220))
        self.qp, self.tp = q, t
        return self.dq

    def step(self, t: float, cx_px: float, q_cap: float, q_now: float, dt: float) -> float:
        self.vel(q_now, t)
        cx_err = cx_px / 240.0 - 0.5
        qg = q_cap + cx_err * self.fov
        self.gh = (self.gh + [qg])[-3:]; qm = float(np.median(self.gh))
        a = 0.62 if (self.qf is not None and abs(qm - self.qf) > 6) else 0.34
        self.qf = qm if self.qf is None else (1 - a) * self.qf + a * qm
        q_des = float(np.clip(self.qf, -self.head_max, self.head_max))
        if abs(cx_err) <= self.center_tol and abs(self.dq) < 7.5:
            return self.hold
        eq = q_des - q_now
        if abs(eq) > 1.8:
            cs = 1 if eq > 0 else -1
            if self.ls != 0 and cs != self.ls:
                self.qi *= 0.2
            self.ls = cs
        if abs(eq) > 0.7:
            if abs(self.dq) < 6.5:
                self.qi = float(np.clip(self.qi + eq * 4.2 * float(np.clip(dt, 0.01, 0.15)), -13, 13))
            self.hold = float(np.clip(q_des + 6.2 * math.tanh((eq - 0.14 * self.dq) / 3.2) + self.qi, -45, 45))
        return self.hold


class HybridGaze:
    """Head centers the target fast; the body then turns (bounded torque, stiction aware) so the robot ends up
    FACING the person with the head back near 0 - instead of staring sideways with a twisted neck."""

    def __init__(self, face_after_deg: float = 7.0, body_vmax: float = 25.0):
        self.head = HeadServo(); self.heading = HeadingServo(vmax=body_vmax)
        self.face_after = face_after_deg
        self.body_target = None
        self.last_seen = -1e9
        self.head_cmd = 0.0

    def step(self, t: float, det, psi_w: float, lead_yaw: float, q_now: float, dt: float, new_det: bool):
        """det = (cx_px or None, psi_w_cap, q_cap). Returns (vyaw, head_cmd)."""
        if det is not None and det[0] is not None and new_det:
            cx, psi_cap, q_cap = det
            self.head_cmd = self.head.step(t, cx, q_cap, q_now, dt)
            cam_dir = psi_cap + q_cap + (cx - 120.0) / F_PX_PER_DEG
            want_body = cam_dir                           # face the person with head at 0
            if self.body_target is None:
                self.body_target = psi_w
            if abs(wrap(want_body - psi_w)) > self.face_after:
                self.body_target = want_body
            elif abs(wrap(want_body - self.body_target)) > self.face_after:
                self.body_target = want_body
            self.last_seen = t
        elif t - self.last_seen > 1.2:
            self.body_target = psi_w
        vy, _ = self.heading.step(t, psi_w, lead_yaw, self.body_target)
        return vy, self.head_cmd
