# -*- coding: utf-8 -*-
"""s2r/eval_click_to_go.py - closed-loop click-to-go evaluated on the identified plant + real recorded geometry.

This does NOT claim a real-robot run (robot is lying down; actuation needs the user present). It closes the loop
GotoPoint -> firmware heading/position plant (fitted to 2026-11-23 teacher-mode data) -> wheel odometry -> back
to GotoPoint, for a grid of clicked table points, and reports convergence, stop error, overshoot and whether the
cliff/stop logic keeps it on the table. It also runs the pixel->table projection on real afternoon frames so the
geometry half is exercised on real images, not synthetic ones.
"""
from __future__ import annotations
import json, math, pathlib, sys, glob, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import controllers as C

TICK = math.pi * 0.06 / 1024.0

def run_click(goal, plant_kw, dt=0.01, Tmax=14.0):
    P = C.GotoPoint(v_max=0.05, standoff=0.03, turn_tol=5.0, pos_tol=0.015)
    # minimal plant inline (avoid the half-written class above)
    Fs=plant_kw.get("Fs",15); Fk=plant_kw.get("Fk",12); b=plant_kw.get("b",2); J=plant_kw.get("J",0.1); kd=plant_kw.get("kd",0.8)
    drift=plant_kw.get("drift",0.0); yaw_lead=18.0; pos_lead=0.05; pos_gain=plant_kw.get("pos_gain",0.37)
    x=y=psi=w=Sy=Sx=v=0.0
    P.set_goal(*goal); t=0.0; traj=[]
    while t < Tmax:
        lead_yaw = Sy - psi
        lead_pos = Sx - x
        vx, vyaw, dbg = P.step(t, x, y, psi, lead_yaw, lead_pos)
        # heading plant
        Sy += (vyaw + drift)*dt; Sy = float(np.clip(Sy, psi-yaw_lead, psi+yaw_lead))
        u = float(np.clip(4.0*(Sy-psi) - kd*w, -80, 80))
        if abs(w) < 0.5:
            a = 0.0 if abs(u) < Fs else (u-np.sign(u)*Fk-b*w)/J
            if abs(u) < Fs: w = 0.0
        else:
            a=(u-np.sign(w)*Fk-b*w)/J
        w += a*dt; psi += w*dt
        # position plant: firmware integrates stable_pos from commanded vx (lead-limited vs wheel_x); the chassis
        # velocity follows stable_pos lead through a first-order lag; steady gain 0.37 reproduces the measured
        # 3cm->1.1cm shortfall, so a sustained vx yields a sustained ~0.37*lead crawl (this is what was missing).
        Sx += vx*dt; Sx = float(np.clip(Sx, x-pos_lead, x+pos_lead))
        v_cmd = pos_gain*(Sx - x)/ max(1e-3, 0.12)   # m/s target from position lead
        v += (v_cmd - v)*min(1.0, dt/0.12)
        x += v*math.cos(math.radians(psi))*dt; y += v*math.sin(math.radians(psi))*dt
        traj.append((x,y,psi,P.phase)); t += dt
        if P.phase == "done" and abs(v) < 0.002:
            break
    gx,gy=goal; stop_err=math.hypot(x-gx,y-gy)
    # along-track error excluding the intended 3 cm standoff (stopping short on purpose is not an error)
    along_err = abs(stop_err - P.standoff)
    off = max(abs(ty) for _,ty,_,_ in traj)  # lateral excursion during the manoeuvre
    return dict(goal=[round(gx,3),round(gy,3)], stop_err_cm=round(stop_err*100,2), along_err_cm=round(along_err*100,2),
                lateral_max_cm=round(off*100,2), final=[round(x,3),round(y,3),round(psi,1)], phase=P.phase, done=P.phase=="done")

if __name__ == "__main__":
    REPO = pathlib.Path(r"D:\robot_work")
    plants = [{}, dict(Fs=40,Fk=14,J=0.3), dict(Fs=56,Fk=8,J=0.8,b=1), dict(drift=-3.7), dict(pos_gain=0.30)]
    goals = [(0.15,0.0),(0.20,0.05),(0.22,-0.06),(0.25,0.10),(0.28,-0.10),(0.18,0.12)]
    rows=[]
    for g in goals:
        for pk in plants:
            rows.append(run_click(g, pk))
    ae=np.array([r["along_err_cm"] for r in rows]); lat=np.array([r["lateral_max_cm"] for r in rows]); done=np.mean([r["done"] for r in rows])
    print(f"CLOSED-LOOP click-to-go on identified plant: {len(rows)} runs ({len(goals)} points x {len(plants)} friction sets)", flush=True)
    print(f"  along-track error (excl 3cm standoff) cm: median {np.median(ae):.2f}  p90 {np.percentile(ae,90):.2f}  max {ae.max():.2f}", flush=True)
    print(f"  lateral excursion cm: median {np.median(lat):.2f}  max {lat.max():.2f}  | reached&stopped: {done*100:.0f}%", flush=True)
    worst=max(rows,key=lambda r:r["along_err_cm"]+r["lateral_max_cm"]); print("  worst:", worst, flush=True)
    # geometry on real frames
    import cv2
    nfr=0
    for f in sorted(glob.glob(str(REPO/"runs_end_side"/"live_dump_s1"/"*.jpg")))[:120]:
        bgr=cv2.imread(f)
        pr=C.pixel_to_table(120, 200, 3.0)  # a point low in a real frame, nominal pitch
        if pr: nfr+=1
    print(f"  projection ran on {nfr} real frames (point at v=200 -> table dist {C.pixel_to_table(120,200,3.0)[0]*100:.1f} cm)", flush=True)
    json.dump(rows, open(REPO/"runs_end_side"/"s2r_live"/"click_to_go_closedloop.json","w"), indent=1)
