# -*- coding: utf-8 -*-
"""s2r/morning_check.py - staged, self-aborting real-robot acceptance for the s2r stack.

Run with the robot STANDING and balancing on the desk, >=20 cm of free table in front, you sitting in front of
it (normal posture is fine - you do not need to do anything):
    D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\s2r\morning_check.py
Stages (each aborts on pitch deviation > 8 deg / |roll| > 12 deg and always ends in a safe stop):
  1. heading servo  +8 / 0 / -8 / 0 / +15 / 0 deg  (overshoot, final error, peak lead)
  2. gaze A/B       40 s hybrid vs 40 s v25 head-servo (face centering error, body facing)
  3. go-to-point    6 cm forward, back to start (odometry error)
Results: runs_end_side/s2r_morning/<stamp>/report.json + report.md
"""
from __future__ import annotations
import json, math, pathlib, subprocess, sys, time
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import numpy as np
from link import Robot
from controllers import GotoPoint, HeadingServo

OUT = REPO / "runs_end_side" / "s2r_morning" / time.strftime("%Y%m%d_%H%M%S"); OUT.mkdir(parents=True, exist_ok=True)
report: dict = {"stamp": OUT.name}

def save():
    json.dump(report, open(OUT / "report.json", "w"), indent=1)

def stage_motion():
    R = Robot()
    log = []
    try:
        if not R.wait_ready(8):
            report["abort"] = "robot unreachable"; return False
        tel = R.udp.latest()
        if tel.robot_state != 1 or abs(tel.roll) > 8:
            report["abort"] = f"robot not standing (state={tel.robot_state}, roll={tel.roll:.1f})"; return False
        p0 = tel.pitch
        b = R.rest_bias(3.0); report["rest_bias_dps"] = round(b, 3)
        R.act.cfg.update(R.drive_cfg()); print("drive cfg", R.act.cfg, flush=True)
        R.act.enable(init_bias_dps=b)
        H = HeadingServo()
        psi0 = R.udp.latest().body_yaw_wheel
        res = []
        def safe(t):
            return abs(t.pitch - p0) <= 8.0 and abs(t.roll) <= 12.0
        prev_tgt = 0.0
        for tgt, dur in ((8, 4), (0, 4), (-8, 4), (0, 4), (15, 5), (0, 5)):
            H.reset(); te = time.perf_counter() + dur; peak = 0.0; seg = []
            while time.perf_counter() < te:
                t = R.udp.latest()
                if t is None: time.sleep(0.02); continue
                if not safe(t):
                    report["abort"] = f"tilt during heading {tgt}: pitch {t.pitch:.1f} roll {t.roll:.1f}"; return False
                lead = float(t.stable_yaw) - t.body_yaw_imu
                vy, dbg = H.step(time.perf_counter(), t.body_yaw_wheel - psi0, lead, float(tgt))
                R.act.set(vx=0.0, vyaw=vy, head=0.0)
                rel = t.body_yaw_wheel - psi0; seg.append(rel); peak = max(peak, abs(lead))
                log.append([round(time.perf_counter(), 3), tgt, round(rel, 2), round(lead, 2), round(vy, 1), round(t.pitch, 2)])
                time.sleep(0.03)
            fin = float(np.mean(seg[-10:]))
            step = tgt - prev_tgt; prev_tgt = tgt
            ov = (max(seg) - tgt) if step > 0 else ((tgt - min(seg)) if step < 0 else 0.0)
            res.append({"target": tgt, "final": round(fin, 2), "final_err": round(abs(fin - tgt), 2), "overshoot": round(ov, 2), "peak_lead": round(peak, 1)})
            print("heading", res[-1], flush=True)
        report["heading"] = res; save()
        # go-to-point: 6 cm forward then back
        G = GotoPoint(v_max=0.04, standoff=0.0, pos_tol=0.012)
        t = R.udp.latest(); x0 = t.x_m; ox = oy = 0.0; lx = x0
        gres = []
        for gx in (0.06, 0.0):
            G.set_goal(gx, 0.0); te = time.perf_counter() + 8.0
            while time.perf_counter() < te and G.phase != "done":
                t = R.udp.latest()
                if t is None: time.sleep(0.02); continue
                if not safe(t):
                    report["abort"] = f"tilt during goto: pitch {t.pitch:.1f}"; return False
                ds = t.x_m - lx; lx = t.x_m; h = math.radians(t.body_yaw_wheel - psi0)
                ox += ds * math.cos(h); oy += ds * math.sin(h)
                vx, vy, dbg = G.step(time.perf_counter(), ox, oy, t.body_yaw_wheel - psi0, float(t.stable_yaw) - t.body_yaw_imu, float(t.stable_pos) - t.x_m)
                R.act.set(vx=vx, vyaw=vy, head=0.0)
                log.append([round(time.perf_counter(), 3), f"goto{gx}", round(ox, 4), round(oy, 4), round(vx, 4), round(t.pitch, 2)])
                time.sleep(0.03)
            time.sleep(1.0)
            gres.append({"goal_x": gx, "reached_x": round(ox, 4), "err_cm": round(abs(ox - gx) * 100, 2), "lateral_cm": round(oy * 100, 2), "phase": G.phase})
            print("goto", gres[-1], flush=True)
        report["goto"] = gres
        return True
    finally:
        R.close()
        json.dump(log, open(OUT / "motion_log.json", "w"))
        save()

def stage_gaze():
    py = sys.executable
    out = {}
    for g in ("hybrid", "v25"):
        d = OUT / f"gaze_{g}"
        r = subprocess.run([py, str(HERE / "live.py"), "--mode", "gaze", "--gaze", g, "--secs", "40", "--log", str(d), "--no-knock"],
                           capture_output=True, text=True, timeout=180)
        try:
            out[g] = json.load(open(d / "summary.json"))
        except Exception:
            out[g] = {"error": r.stdout[-600:] + r.stderr[-600:]}
        print("gaze", g, out[g], flush=True)
        if out[g].get("abort"):
            break
    report["gaze"] = out; save()

if __name__ == "__main__":
    ok = stage_motion()
    if ok:
        stage_gaze()
    md = ["# s2r morning check " + OUT.name, "", "```json", json.dumps(report, indent=1, ensure_ascii=False), "```"]
    open(OUT / "report.md", "w", encoding="utf-8").write("\n".join(md))
    print("REPORT", OUT / "report.md", flush=True)
