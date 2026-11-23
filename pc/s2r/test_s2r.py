# -*- coding: utf-8 -*-
"""Gate tests for s2r controllers (no robot needed): python -m pytest pc/s2r/test_s2r.py -q"""
import math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import controllers as C
import sim_plant as S


def _run_heading(tgt, **pk):
    P = S.Plant(**pk); H = C.HeadingServo(); nc = 0.0; v = 0.0; hist = []
    while P.t < 7.0:
        if P.t >= nc:
            v, _ = H.step(P.t, P.psi, P.S - (P.imu - P.q), tgt); nc += 0.04
        P.step(v); hist.append(P.psi)
    return np.array(hist), P.umax


def test_heading_no_big_overshoot_and_bounded_torque():
    for pk in ({}, dict(Fs=40, Fk=14, J=0.3), dict(Fs=56, Fk=8, J=0.8, b=1)):
        for tgt in (20.0, 8.0, -15.0):
            h, umax = _run_heading(tgt, **pk)
            over = (h.max() - tgt) if tgt > 0 else (tgt - h.min())
            assert over < 6.0, (pk, tgt, over)
            assert umax <= 66.0, (pk, tgt, umax)     # lead cap 16 deg * k_yaw 4 (+damping)
            assert abs(h[-200:].mean() - tgt) < 4.0, (pk, tgt, h[-200:].mean())


def test_old_style_loop_reproduces_fall_mechanism():
    """The integrating P-loop used before the fall saturates yaw torque (80) and overshoots - keep as a regression fact."""
    P = S.Plant(Fs=56, Fk=8, J=0.8, b=1, lead_max=35.0); nc = 0.0; v = 0.0; hist = []
    while P.t < 7.0:
        if P.t >= nc:
            e = 20.0 - P.psi; v = 0.0 if abs(e) < 0.8 else float(np.clip(3.0 * e, -45, 45)); nc += 0.04
        P.step(v); hist.append(P.psi)
    assert P.umax >= 79.0 and max(hist) - 20.0 > 6.0


def test_body_gaze_faces_person():
    r = S.run_gaze(S.make_hybrid(), S.face_traj("steps"), plant_kw={})
    assert r["body_off_end"] < 4.0 and r["umax"] <= 66.0


def test_pixel_to_table_geometry():
    f = C.F_PX_PER_DEG * 57.2958
    d, lat = C.pixel_to_table(120, 120 + f * math.tan(math.atan2(0.135, 0.30)), 0.0)
    assert abs(d - 0.30) < 1e-3 and abs(lat) < 1e-6
    assert C.pixel_to_table(120, 100, 0.0) is None           # above the horizon


def test_goto_reaches_and_stops():
    # kinematic toy: firmware tracks stable_pos with lag; checks phase logic + stop
    G = C.GotoPoint(); G.set_goal(0.10, 0.0); x = 0.0; sp = 0.0; psi = 0.0; S_y = 0.0
    for k in range(600):
        vx, vy, d = G.step(k * 0.04, x, 0.0, psi, S_y - psi, sp - x)
        sp += vx * 0.04; x += 0.5 * (sp - x); S_y += vy * 0.04; psi += 0.5 * (S_y - psi)
        if G.phase == "done":
            break
    assert G.phase == "done" and abs(x - (0.10 - G.standoff)) < 0.03
