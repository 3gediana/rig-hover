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


def test_fist_detector_face_overlap_and_band():
    import numpy as np
    class _KP:
        def __init__(self, xy, cf): self.xy=_T(xy); self.conf=_T(cf); self.n=len(xy)
        def __len__(self): return self.n
    class _T:
        def __init__(self, a): self.a=np.array(a)
        def cpu(self): return self
        def numpy(self): return self.a
    class _Boxes:
        def __init__(self, n): self.n=n
        def __len__(self): return self.n
    class _R:
        def __init__(self, xy, cf): self.keypoints=_KP([xy],[cf]); self.boxes=_Boxes(1)
    class _Pose:
        def __init__(self, xy, cf): self.xy, self.cf = xy, cf
        def predict(self, *a, **k): return [_R(self.xy, self.cf)]
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("fist", str(pathlib.Path(__file__).resolve().parent / "fist.py"))
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
    # 17 COCO keypoints; wrist R=10 at (184,128) high conf, elbow R=8 present
    xy = [[0,0]]*17; cf = [0.0]*17
    xy[8]=[170,110]; cf[8]=0.5; xy[10]=[184,128]; cf[10]=0.64
    det = F.FistDetector(_Pose(xy, cf), None)
    bgr = np.zeros((240,240,3), np.uint8)
    d = det.detect(bgr, 0.0, face_box_px=None)
    assert d is not None and d["src"] == "wrist"       # desk-height fist kept (old gate dropped it)
    det2 = F.FistDetector(_Pose(xy, cf), None)
    d2 = det2.detect(bgr, 0.0, face_box_px=(150,100,220,170))  # wrist inside face box -> rejected
    assert d2 is None


def test_click_to_go_converges_and_stops():
    import importlib.util, pathlib, numpy as np
    spec = importlib.util.spec_from_file_location("ectg", str(pathlib.Path(__file__).resolve().parent / "eval_click_to_go.py"))
    E = importlib.util.module_from_spec(spec)
    import sys as _s; _s.argv = ["x"]  # prevent __main__ block (guarded) 
    spec.loader.exec_module(E)
    for g in [(0.15, 0.0), (0.22, -0.06), (0.25, 0.10)]:
        for pk in [{}, dict(Fs=56, Fk=8, J=0.8, b=1), dict(drift=-3.7)]:
            r = E.run_click(g, pk)
            assert r["done"], (g, pk, r)
            assert r["along_err_cm"] < 4.0, (g, pk, r)      # along-track, standoff removed
            assert r["lateral_max_cm"] < 12.0, (g, pk, r)


def test_gaze_turns_body_when_face_beyond_head_reach():
    # real failure 2026-11-24: face 56 px right, head pinned at +32 deg, body never turned
    H = C.HybridGazeV2()
    vy = 0.0
    for k in range(40):
        vy, hd = H.step(k * 0.07, (176.0, 0.0, 32.4), 0.0, 0.0, 32.4, 0.07, True)
    assert H.body_target is not None and H.body_target > 30.0, H.body_target
    assert vy > 0.0
    assert abs(hd) <= 45.0


def test_no_hold_term_drives_robot_at_rest_offset():
    # stable_pos rests ~4.6 cm ahead of the wheels; idle/turn must not command vx from that offset
    G = C.GotoPoint()
    vx, vy, d = G.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.046)
    assert vx == 0.0
    G.set_goal(0.0, 0.10)
    vx, vy, d = G.step(0.04, 0.0, 0.0, 0.0, 0.0, 0.046)
    assert d["phase"] == "turn" and vx == 0.0
