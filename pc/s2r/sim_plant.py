# -*- coding: utf-8 -*-
"""s2r/sim_plant.py - stick-slip heading plant identified from the real robot (teacher-mode turn log,
runs_end_side/hw_char_upright/teacher). Used to stress-test controllers before they ever touch the robot.
Fit (RMS 2.1 deg on the +-30 deg steps): u = 4*(stable_yaw - yaw_body) - 0.8*w ; static friction 15,
kinetic 12, viscous 2, inertia 0.1 (firmware torque units). Swept over Fs 15..56, J 0.1..0.8 for robustness."""
from __future__ import annotations
import math, sys, itertools
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
import controllers as C

F = C.F_PX_PER_DEG

class Plant:
    def __init__(s, Fs=15, Fk=12, b=2, J=0.1, kd=0.8, drift=-3.7, ff_err=0.0, lead_max=16.0, head_db=7.0):
        s.Fs, s.Fk, s.b, s.J, s.kd, s.drift, s.ff_err, s.lead_max, s.head_db = Fs, Fk, b, J, kd, drift, ff_err, lead_max, head_db
        s.psi = s.w = s.S = s.t = s.drift_acc = s.q = s.qcmd = s.umax = 0.0
    @property
    def imu(s):
        return s.psi + s.drift_acc + s.q
    def step(s, vyaw_cmd, dt=0.002):
        body_imu = s.imu - s.q
        s.S += (vyaw_cmd + s.drift + s.ff_err) * dt
        s.S = float(np.clip(s.S, body_imu - s.lead_max, body_imu + s.lead_max))
        u = float(np.clip(4.0 * (s.S - body_imu) - s.kd * s.w, -80, 80)); s.umax = max(s.umax, abs(u))
        if abs(s.w) < 0.5:
            a = 0.0 if abs(u) < s.Fs else (u - np.sign(u) * s.Fk - s.b * s.w) / s.J
            if abs(u) < s.Fs: s.w = 0.0
        else:
            a = (u - np.sign(s.w) * s.Fk - s.b * s.w) / s.J
        s.w += a * dt; s.psi += s.w * dt; s.drift_acc += s.drift * dt
        e = s.qcmd - s.q
        if abs(e) > s.head_db:
            dqh = np.sign(e) * min(abs(e) - s.head_db, 150 * dt)
            s.q += dqh; s.psi -= 0.115 * dqh
        s.t += dt

def face_traj(kind):
    if kind == "steps": return lambda t: 0.0 if t < 1 else (18.0 if t < 6 else (-12.0 if t < 11 else 6.0))
    if kind == "sway": return lambda t: 8 * math.sin(2 * math.pi * 0.15 * t) + 3 * math.sin(2 * math.pi * 0.4 * t)
    return lambda t: 4.0

def run_gaze(controller, face, Tend=16.0, plant_kw=None, seed=0, fps=14.0, lat=0.075, cmd_dt=0.04, noise_px=1.5, cmd_lat=0.03):
    rng = np.random.default_rng(seed); P = Plant(**(plant_kw or {}))
    next_frame = next_cmd = 0.0; dets = []; cmdq = []; vy = 0.0; errs = []; rates = []; k = 0
    while P.t < Tend:
        t = P.t
        if t >= next_frame:
            cx = 120 + F * (face(t) - (P.psi + P.q)) + rng.normal(0, noise_px)
            dets.append((t + lat, cx if 0 <= cx <= 240 else None, P.psi, P.q)); next_frame += 1.0 / fps
        if t >= next_cmd:
            avail = [d for d in dets if d[0] <= t]; det = max(avail, key=lambda d: d[0]) if avail else None
            dets = ([det] if det else []) + [d for d in dets if d[0] > t]
            cmdq.append((t + cmd_lat, controller(t, det, P, rng))); next_cmd += cmd_dt
        while cmdq and cmdq[0][0] <= t:
            vy, P.qcmd = cmdq.pop(0)[1]
        P.step(vy); k += 1
        if k % 10 == 0:
            errs.append(face(t) - (P.psi + P.q)); rates.append(P.w)
    errs = np.array(errs[int(len(errs) * 0.12):]); rates = np.array(rates); body_face = abs(face(Tend) - P.psi)
    mv = rates[np.abs(rates) > 3]
    return dict(med=float(np.median(abs(errs))), p90=float(np.percentile(abs(errs), 90)), within3=float(np.mean(abs(errs) < 3) * 100),
                within5=float(np.mean(abs(errs) < 5) * 100), body_off_end=float(body_face), reversals=int(np.sum(np.diff(np.sign(mv)) != 0)), umax=P.umax)

def make_s2r():
    G = C.BodyGaze(); H = C.HeadingServo(); st = {"used": None}
    def ctrl(t, det, P, rng):
        psi_w = P.psi + rng.normal(0, 0.1)
        if det is not None and det is not st["used"]:
            st["used"] = det; G.update(t, det[1], det[2], det[3], P.q, psi_w)
        elif det is None:
            G.update(t, None, psi_w, P.q, P.q, psi_w)
        v, _ = H.step(t, psi_w, P.S - (P.imu - P.q), G.target)
        return (v, 0.0)
    return ctrl

def make_devin_v25():
    """Faithful port of real_robot_bridge.CalibratedGazeServo (FOV 62.5, head-first, body only past 34 deg)."""
    s = {"gh": [], "qf": None, "qi": 0.0, "ls": 0, "hold": 0.0, "qp": None, "tp": None, "dq": 0.0, "used": None, "vy": 0.0, "hc": 0.0}
    def ctrl(t, det, P, rng):
        if s["qp"] is not None and t - s["tp"] >= 0.004:
            s["dq"] = 0.62 * s["dq"] + 0.38 * float(np.clip((P.q - s["qp"]) / (t - s["tp"]), -220, 220))
        s["qp"], s["tp"] = P.q, t
        if det is None or det is s["used"] or det[1] is None:
            return (s["vy"], s["hc"])
        s["used"] = det
        cx_err = det[1] / 240.0 - 0.5; qg = det[3] + cx_err * 62.5
        s["gh"] = (s["gh"] + [qg])[-3:]; qm = float(np.median(s["gh"]))
        s["qf"] = qm if s["qf"] is None else (1 - (0.62 if abs(qm - s["qf"]) > 6 else 0.34)) * s["qf"] + (0.62 if abs(qm - s["qf"]) > 6 else 0.34) * qm
        q_des = float(np.clip(s["qf"], -44, 44)); vyaw = 0.0
        if abs(s["qf"]) > 34:
            vyaw = math.copysign(float(np.clip(1.85 * (abs(s["qf"]) - 32) + 10, 10, 58)), s["qf"])
        if abs(cx_err) <= 0.026 and abs(s["dq"]) < 7.5 and vyaw == 0.0:
            s["vy"] = 0.0; s["hc"] = s["hold"]; return (0.0, s["hold"])
        eq = q_des - P.q
        if abs(eq) > 1.8:
            cs = 1 if eq > 0 else -1
            if s["ls"] != 0 and cs != s["ls"]: s["qi"] *= 0.2
            s["ls"] = cs
        if abs(eq) > 0.7:
            if abs(s["dq"]) < 6.5: s["qi"] = float(np.clip(s["qi"] + eq * 4.2 * 0.07, -13, 13))
            s["hold"] = float(np.clip(q_des + 6.2 * math.tanh((eq - 0.14 * s["dq"]) / 3.2) + s["qi"], -45, 45))
        s["vy"], s["hc"] = vyaw, s["hold"]
        return (s["vy"], s["hc"])
    return ctrl

def make_hybrid():
    Hy = C.HybridGaze(); st = {"used": None, "tp": None}
    def ctrl(t, det, P, rng):
        psi_w = P.psi + rng.normal(0, 0.1)
        new = det is not None and det is not st["used"]
        if new: st["used"] = det
        dt = 0.04 if st["tp"] is None else t - st["tp"]; st["tp"] = t
        d = (det[1], det[2], det[3]) if det is not None else None
        return Hy.step(t, d, psi_w, P.S - (P.imu - P.q), P.q, dt, new)
    return ctrl

def make_hybrid2():
    Hy = C.HybridGazeV2(); st = {"used": None, "tp": None}
    def ctrl(t, det, P, rng):
        psi_w = P.psi + rng.normal(0, 0.1)
        new = det is not None and det is not st["used"]
        if new: st["used"] = det
        dt = 0.04 if st["tp"] is None else t - st["tp"]; st["tp"] = t
        d = (det[1], det[2], det[3]) if det is not None else None
        return Hy.step(t, d, psi_w, P.S - (P.imu - P.q), P.q, dt, new)
    return ctrl

PLANTS = [(0, {}), (1, dict(Fs=40, Fk=14, J=0.3)), (2, dict(Fs=56, Fk=8, J=0.8, b=1)), (3, dict(ff_err=1.0)), (4, dict(head_db=10)), (5, dict(ff_err=-1.5, Fs=30))]

def heading_sweep():
    """Torque/overshoot of the heading servo over a wide friction sweep (targets 20/8/3 deg)."""
    out = []
    for Fs, Fk, J, b, kd in itertools.product([15, 35, 56], [8, 14, 22], [0.1, 0.3, 0.8], [1, 2, 4], [0.0, 0.8]):
        if Fk > Fs: continue
        for tgt in (20.0, 8.0, 3.0):
            P = Plant(Fs=Fs, Fk=Fk, J=J, b=b, kd=kd); H = C.HeadingServo(); t = 0.0; nc = 0.0; mx = -1e9; hist = []
            v = 0.0
            while P.t < 7.0:
                if P.t >= nc:
                    v, _ = H.step(P.t, P.psi, P.S - (P.imu - P.q), tgt); nc += 0.04
                P.step(v); mx = max(mx, P.psi); hist.append(P.psi)
            out.append((tgt, mx - tgt, abs(np.mean(hist[-250:]) - tgt), P.umax))
    a = np.array(out)
    return dict(overshoot_p95=float(np.percentile(a[:, 1], 95)), overshoot_max=float(a[:, 1].max()), final_err_p50=float(np.median(a[:, 2])),
                final_err_p90=float(np.percentile(a[:, 2], 90)), torque_max=float(a[:, 3].max()), n=len(a))

if __name__ == "__main__":
    import json
    res = {"heading_sweep": heading_sweep(), "gaze": {}}
    print("HEADING", json.dumps(res["heading_sweep"]), flush=True)
    for kind in ("still", "steps", "sway"):
        for name, mk in (("devin_v25", make_devin_v25), ("s2r_body", make_s2r), ("s2r_hybrid", make_hybrid), ("s2r_hybrid2", make_hybrid2)):
            rs = [run_gaze(mk(), face_traj(kind), seed=sd, plant_kw=pk) for sd, pk in PLANTS]
            agg = {k: (float(np.median([r[k] for r in rs])) if k != "umax" else float(max(r[k] for r in rs))) for k in rs[0]}
            res["gaze"][f"{kind}/{name}"] = agg
            print(f"{kind:6s} {name:10s} |err| med {agg['med']:.1f} p90 {agg['p90']:.1f}  <3deg {agg['within3']:.0f}%  <5deg {agg['within5']:.0f}%  reversals {agg['reversals']:.0f}  body-vs-face at end {agg['body_off_end']:.1f}deg  torque_max {agg['umax']:.0f}", flush=True)
    json.dump(res, open(__file__.rsplit("\\", 1)[0] + "\\sim_results.json", "w"), indent=1)
