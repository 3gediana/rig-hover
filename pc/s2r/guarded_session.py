# -*- coding: utf-8 -*-
"""s2r/guarded_session.py - real-robot identification session behind a hard guard.

Built for the rule "do not knock the robot off the table". Every actuated
stage runs inside a guard that is evaluated on every telemetry packet (500 Hz)
plus a watchdog, and trips into the same safe stop the s2r runtime always uses
(`/mode?src=manual` + `/cmd?stop=1`). All 34 telemetry floats are streamed to
CSV for the whole session, so anything that happens can be reconstructed.

Guard (hard limits, independent of the controllers under test):
  * pitch deviation from the rest median          (--pitch-dev,  default 6 deg)
  * roll: absolute, and deviation from rest       (--roll-abs 12, --roll-dev 5 deg)
  * position box around the rest position (wheel odometry)      (--pos-box, default 7 cm)
  * raw wheel_x travel from rest                                (--wheel-box, default 9 cm)
  * body heading excursion (in-place turns only)                (--psi-box,  default 30 deg)
  * speed: wheel speed (0.10 m/s) / body yaw rate (70 deg/s), 0.1 s windows
  * robot_state != 1, telemetry older than 0.6 s
An empty file named ABORT in the output directory also trips it.

Stages (one per run keeps each experiment clean):
  rest      no actuation; rest statistics only
  head      teacher mode, head servo stepped through --head-targets, body heading held, frames saved
  heading   in-place heading servo steps (+8/0/-8/0/+15/0), the morning_check protocol
  ramp      quasi-static balance-angle identification: imu_zero is ramped (GET /api/set?i=8)
            while the firmware's own position offset is watched against the LQR prediction
  drive     signed straight-line distance (vx integrated by the firmware), heading held

Usage:
    python pc/s2r/guarded_session.py --stage rest --secs 60
    python pc/s2r/guarded_session.py --stage head
    python pc/s2r/guarded_session.py --stage heading
    python pc/s2r/guarded_session.py --stage ramp --imu-zero-to 1.0 --ramp-rate 0.1
    python pc/s2r/guarded_session.py --stage drive --dist 0.03 --back
Output: runs_end_side/s2r_live/session_<stamp>_<stage>/{tel.csv,ctrl.jsonl,summary.json,frames/}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import requests  # noqa: E402
from controllers import HeadingServo  # noqa: E402
from link import DEG_PER_WDIFF_M, TICK_M, Robot  # noqa: E402
from tellog import FIELDS, RawTel, TelCsv  # noqa: E402

OUT_ROOT = REPO / "runs_end_side" / "s2r_live"
K0, K2 = 1300.0, 25.0                    # firmware lqr_k[0], lqr_k[2] (robot.cc InitializeController)
IMU_ZERO_VAR_INDEX = 8                   # /api/set?i=8  -> imu_zero (runtime only, resets on reboot)

DEFAULT_LIMITS = {"pitch_dev": 6.0, "roll_abs": 12.0, "roll_dev": 5.0, "pos_box": 0.07, "wheel_box": 0.09,
                  "psi_box": 30.0, "wheel_vmax": 0.10, "yaw_rate_max": 70.0, "stale_s": 0.6,
                  "debounce_angle": 30, "debounce_other": 10}     # packets at 500 Hz: 60 ms / 20 ms


class Guard:
    """Packet-rate safety envelope. `__call__` is the RawTel on_pkt hook."""

    def __init__(self, on_abort, abort_file: pathlib.Path, **lim):
        self.lim = dict(DEFAULT_LIMITS)
        self.lim.update(lim)
        self.on_abort = on_abort
        self.abort_file = abort_file
        self.armed = False
        self.tripped = ""
        self.trip_at = None
        self.p0 = self.r0 = 0.0
        self.o0 = (0.0, 0.0)
        self.w0 = 0.0
        self.psi0 = 0.0
        self.ox = self.oy = 0.0                  # odometry since the first packet
        self.lx = None
        self.last_pkt_t = 0.0
        self.n_pkt = 0
        self.xhist = collections.deque(maxlen=50)   # (recv_t, wheel_x): 0.1 s at 500 Hz
        self.psihist = collections.deque(maxlen=50)  # (recv_t, psi_wheel_deg)
        self.max_pos = 0.0
        self.max_pitch_dev = 0.0
        self.peak_speed = 0.0
        self._why_kind, self._why_n = "", 0
        self.angle_spikes = {"roll": 0, "pitch": 0}   # single-sample jumps > 5 deg (raw-accelerometer artefacts)
        self._prev_f = None

    @staticmethod
    def psi_wheel(f) -> float:
        return -DEG_PER_WDIFF_M * (f[10] - f[11]) * TICK_M

    def arm(self, p0: float, r0: float, f) -> None:
        self.p0, self.r0 = p0, r0
        self.o0 = (self.ox, self.oy)
        self.w0 = f[12]
        self.psi0 = self.psi_wheel(f)
        self.armed = True

    def __call__(self, recv_t: float, seq: int, t_ms: int, f) -> None:
        self.last_pkt_t = recv_t
        self.n_pkt += 1
        psi = self.psi_wheel(f)
        h = math.radians(psi)
        if self.lx is not None:
            ds = f[12] - self.lx
            self.ox += ds * math.cos(h)
            self.oy += ds * math.sin(h)
        self.lx = f[12]
        self.xhist.append((recv_t, f[12]))
        self.psihist.append((recv_t, psi))
        if not self.armed or self.tripped:
            return
        pd = abs(f[1] - self.p0)
        self.max_pitch_dev = max(self.max_pitch_dev, pd)
        if self._prev_f is not None:
            for i, name in ((0, "roll"), (1, "pitch")):
                if abs(f[i] - self._prev_f[i]) > 5.0:
                    self.angle_spikes[name] += 1
        self._prev_f = f
        pos = math.hypot(self.ox - self.o0[0], self.oy - self.o0[1])
        self.max_pos = max(self.max_pos, pos)
        state = int(round(f[22]))
        speed = abs(f[12] - self.xhist[0][1]) / max(1e-3, recv_t - self.xhist[0][0])
        yaw_rate = abs(psi - self.psihist[0][1]) / max(1e-3, recv_t - self.psihist[0][0])
        if len(self.xhist) == self.xhist.maxlen:
            self.peak_speed = max(self.peak_speed, speed)
        why = ""
        if state != 1:
            why = f"robot_state={state}"
        elif pd > self.lim["pitch_dev"]:
            why = f"pitch {f[1]:.2f} vs rest {self.p0:.2f}"
        elif abs(f[0]) > self.lim["roll_abs"]:
            why = f"roll {f[0]:.2f} abs"
        elif abs(f[0] - self.r0) > self.lim["roll_dev"]:
            why = f"roll {f[0]:.2f} vs rest {self.r0:.2f}"
        elif pos > self.lim["pos_box"]:
            why = f"position box {pos * 100:.1f} cm"
        elif abs(f[12] - self.w0) > self.lim["wheel_box"]:
            why = f"wheel_x travel {(f[12] - self.w0) * 100:.1f} cm"
        elif abs(psi - self.psi0) > self.lim["psi_box"]:
            why = f"heading excursion {psi - self.psi0:+.1f} deg"
        elif len(self.xhist) == self.xhist.maxlen and speed > self.lim["wheel_vmax"]:
            why = f"wheel speed {speed:.3f} m/s"
        elif len(self.psihist) == self.psihist.maxlen and yaw_rate > self.lim["yaw_rate_max"]:
            why = f"yaw rate {yaw_rate:.0f} deg/s"
        if why:
            # Debounce. The firmware's `roll` is the raw accelerometer angle,
            # unfiltered: the head servo starting shakes the IMU on the head and
            # single samples jump by 10 deg (measured 2026-11-24 23:46). A real
            # tip develops over hundreds of ms, so angle limits must persist.
            kind = why.split(" ")[0]
            need = self.lim["debounce_angle"] if kind in ("pitch", "roll") else self.lim["debounce_other"]
            if self._why_kind == kind:
                self._why_n += 1
            else:
                self._why_kind, self._why_n = kind, 1
            if self._why_n >= need:
                self._trip(why, recv_t)
        else:
            self._why_kind, self._why_n = "", 0

    def _trip(self, why: str, t: float) -> None:
        if self.tripped:
            return
        self.tripped = why
        self.trip_at = t
        threading.Thread(target=self.on_abort, args=(why,), daemon=True, name="s2r-abort").start()

    def watchdog(self, t_now: float) -> None:
        if self.armed and not self.tripped:
            if t_now - self.last_pkt_t > self.lim["stale_s"]:
                self._trip("telemetry stale", t_now)
            elif self.abort_file.exists():
                self._trip("ABORT file", t_now)


def make_robot(base: str, out: pathlib.Path, guard: Guard, frames: bool, keep_s: float, jpg_q: int = 60,
               tlm_port: int = 9001):
    """link.Robot, but with the single telemetry socket owned by tellog.RawTel.

    The firmware streams to the *last* client that sent to the telemetry port,
    so a session must own exactly one socket. `frames=False` leaves the frame
    grabber out: each /capture holds the single-threaded httpd for ~33 ms and
    would add jitter to the command path, and the stage does not need pictures.
    """
    csv = TelCsv(str(out / "tel.csv"))
    host = base.split("//")[-1].split(":")[0].split("/")[0]
    R = Robot.__new__(Robot)
    R.base = base
    R.udp = RawTel(host, port=tlm_port, keep_s=keep_s, on_pkt=lambda rt, sq, tm, f: (csv(rt, sq, tm, f), guard(rt, sq, tm, f)))
    R.frames = None
    if frames:
        from link import FrameGrabber
        R.frames = FrameGrabber(base, R.udp, q=jpg_q)
    R.act = None
    return R, csv


def api_get(base: str, path: str, timeout: float = 0.6):
    s = requests.Session()
    s.trust_env = False          # the dev box has a system proxy; never use it for the robot
    return s.get(base + path, timeout=timeout).text


def get_imu_zero(base: str) -> float:
    return float(json.loads(api_get(base, "/api/data"))["vars"][IMU_ZERO_VAR_INDEX])


def set_imu_zero(base: str, v: float) -> None:
    api_get(base, f"/api/set?i={IMU_ZERO_VAR_INDEX}&v={v:.3f}")


def stat(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}


class Session:
    def __init__(self, a):
        self.a = a
        self.base = f"http://{a.host}"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out = OUT_ROOT / f"session_{stamp}_{a.stage}"
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        lim = {"pos_box": 0.085, "wheel_box": 0.10} if a.stage == "ramp" else {}
        for k in ("pos_box", "wheel_box", "pitch_dev", "roll_dev", "psi_box"):
            v = getattr(a, k, None)
            if v:
                lim[k] = float(v)
        self.guard = Guard(self._on_abort, self.out / "ABORT", **lim)
        self.R, self.csv = make_robot(self.base, self.out, self.guard, frames=(a.stage == "head"),
                                      keep_s=max(60.0, a.secs + 5.0), tlm_port=a.tlm_port)
        self.ctrl = open(self.out / "ctrl.jsonl", "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.report: dict = {"stage": a.stage, "out": str(self.out), "args": vars(a), "abort": "",
                             "t0_perf": {"tel": self.csv.t0, "session": self.t0}}   # to align tel.csv and ctrl.jsonl
        self.actuated = False
        self.imu_zero_changed = False

    def log(self, **rec) -> None:
        rec["t"] = round(time.perf_counter() - self.t0, 3)
        self.ctrl.write(json.dumps(rec) + "\n")

    def _safe_stop(self) -> bool:
        """Zero, then manual, then verify from telemetry that the firmware is really in manual.

        Order matters. A teacher-mode request can be delayed by a TCP
        retransmit and land after the switch back to manual, which would put
        the firmware in teacher mode with whatever velocities that request
        carried. So the last teacher requests are sent with zeros first; then
        the source is switched (twice); then the arbiter source in the
        telemetry stream is checked, and the switch is retried if it did not
        take.
        """
        act = self.R.act
        # Re-assert the zero command for ~160 ms. On a guard trip the control
        # loop may be mid-iteration and issue one more `set()` right after the
        # first zero; re-asserting makes sure the requests still in flight at
        # the end of the window all carry zeros.
        t = time.perf_counter()
        while time.perf_counter() - t < 0.16:
            act.set(0.0, 0.0, 0.0)
            time.sleep(0.02)
        act.enabled = False
        time.sleep(0.12)                           # let any in-flight POST land before the switch
        ok = True
        for _ in range(4):
            ok = bool(act.safe_stop()) and ok
            time.sleep(0.25)
            tel = self.R.udp.latest(max_age=1.0)
            if tel is not None and tel.src == 0:
                return ok
        self.report["safe_stop_unverified"] = True
        return False

    def _on_abort(self, why: str) -> None:
        # Runs in its own thread so telemetry logging is not held up by the HTTP calls.
        ok = self._safe_stop()
        self.report["abort"] = why
        self.report["safe_stop_ok"] = ok
        print(f"GUARD TRIPPED: {why} -> safe stop {'ok' if ok else 'FAILED'}", flush=True)

    # ------------------------------------------------------------------ setup
    def preflight(self) -> bool:
        R = self.R
        dl = time.perf_counter() + 8.0
        while time.perf_counter() < dl and (R.udp.latest() is None or (R.frames is not None and R.frames.get(wait=0.05) is None)):
            time.sleep(0.05)
        t = R.udp.latest()
        if t is None:
            print("robot unreachable (no telemetry)", flush=True)
            self.report["preflight"] = "unreachable"
            return False
        self.report["imu_zero_start"] = get_imu_zero(self.base)
        b = R.rest_bias(2.5)
        raw = [r[3] for r in R.udp.raw][-1200:]
        p0 = float(np.median([f[1] for f in raw]))
        r0 = float(np.median([f[0] for f in raw]))
        self.guard.arm(p0, r0, R.udp.raw[-1][3])
        self.report.update({"rest_bias_dps": round(b, 3), "rest_pitch": round(p0, 3), "rest_roll": round(r0, 3),
                            "rest_lead_cm": round((t.stable_pos - t.x_m) * 100, 2),
                            "cfg_before": json.loads(api_get(self.base, "/cfg"))})
        print(f"ready: udp={R.udp.n} pitch={t.pitch:+.2f} roll={t.roll:+.2f} state={t.robot_state} "
              f"imu_zero={self.report['imu_zero_start']:+.2f}", flush=True)
        if self.a.stage == "rest":
            return True
        if t.robot_state != 1 or abs(t.roll) > 9:
            print("not standing/balancing (or roll > 9 deg) -> refusing to actuate", flush=True)
            self.report["preflight"] = "refused"
            return False
        print(f"rest bias {b:+.2f} dps; guard armed {self.guard.lim}", flush=True)
        return True

    def enable_drive(self, vx_max: float) -> None:
        from link import Actuator
        R = self.R
        R.act = Actuator(self.base, R.udp)
        cfg = R.drive_cfg()
        cfg.update({"vx_max": vx_max, "vyaw_max": 45.0})
        R.act.cfg.update(cfg)
        self.report["act_cfg"] = dict(R.act.cfg)
        print("drive cfg", R.act.cfg, flush=True)
        print("cfg ->", R.act.enable(init_bias_dps=self.report["rest_bias_dps"]), flush=True)
        self.actuated = True
        self.report["cfg_during"] = json.loads(api_get(self.base, "/cfg"))

    def finish(self) -> None:
        R = self.R
        if self.actuated:
            self.report["final_safe_stop_ok"] = self._safe_stop()
        if self.imu_zero_changed:
            try:
                self.report["imu_zero_end"] = get_imu_zero(self.base)
            except Exception as exc:
                self.report["imu_zero_end"] = f"read failed: {exc!r}"
        if R.frames is not None:
            R.frames.stop = True
        if R.act is not None:
            R.act.stop = True
        R.udp.stop = True
        time.sleep(0.3)
        self.ctrl.close()
        self.csv.close()
        try:
            self.report["cfg_after"] = json.loads(api_get(self.base, "/cfg"))   # RAM-only; resets on reboot
        except Exception as exc:
            self.report["cfg_after"] = f"read failed: {exc!r}"
        g = self.guard
        self.report["guard"] = {"tripped": g.tripped, "max_pos_cm": round(g.max_pos * 100, 2),
                                "max_pitch_dev": round(g.max_pitch_dev, 3), "peak_speed_mps": round(g.peak_speed, 4),
                                "packets": g.n_pkt, "limits": g.lim, "angle_spikes": g.angle_spikes}
        json.dump(self.report, open(self.out / "summary.json", "w", encoding="utf-8"), indent=1)
        print("REPORT", self.out / "summary.json", flush=True)

    # ------------------------------------------------------------------ helpers
    def ok(self) -> bool:
        """One watchdog tick; False once the guard has tripped."""
        self.guard.watchdog(time.perf_counter())
        return not self.guard.tripped

    def hold_heading(self, H: HeadingServo, psi0: float, head: float, tag: str, secs: float) -> bool:
        """Hold the body heading where it was; command the head. Returns False if aborted."""
        te = time.perf_counter() + secs
        while time.perf_counter() < te:
            if not self.ok():
                return False
            t = self.R.udp.latest()
            if t is None:
                time.sleep(0.02)
                continue
            self.step_heading(H, psi0, head, tag)
            time.sleep(0.03)
        return True

    def step_heading(self, H: HeadingServo, psi0: float, head: float, tag: str, vx: float = 0.0, **extra) -> None:
        t = self.R.udp.latest()
        if t is None:
            return
        lead = float(t.stable_yaw) - t.body_yaw_imu
        vy, _ = H.step(time.perf_counter(), t.body_yaw_wheel - psi0, lead, 0.0)
        self.R.act.set(vx=vx, vyaw=vy, head=head)
        self.log(stage=tag, head_cmd=head, q=float(t.q), q_tgt=float(t.q_tgt), psi_rel=t.body_yaw_wheel - psi0,
                 lead_yaw=lead, vx=vx, vyaw=vy, pitch=float(t.pitch), roll=float(t.roll),
                 lead_pos=float(t.stable_pos) - t.x_m, wheel_x=float(t.x_m), stable_pos=float(t.stable_pos),
                 tor1=float(t.tor1), tor2=float(t.tor2), **extra)

    # ------------------------------------------------------------------ stages
    def run_rest(self) -> None:
        t_mark = time.perf_counter()
        te = t_mark + self.a.secs
        while time.perf_counter() < te:
            if not self.ok():
                return
            time.sleep(0.05)
        with self.R.udp.lock:                     # only this stage's packets, not the preflight window
            pk = [r for r in self.R.udp.raw if r[0] >= t_mark]
        F = np.array([r[3] for r in pk], dtype=float)
        up = np.array([r[2] for r in pk], dtype=float) / 1000.0
        dur = float(up[-1] - up[0])
        st = {n: stat(F[:, i]) for i, n in enumerate(FIELDS)}
        self.report["rest"] = {"duration_s": round(dur, 2), "packets": int(len(F)),
                               "rate_hz": round(len(F) / dur, 1) if dur > 0 else 0.0,
                               "robot_state": sorted(set(np.round(F[:, 22]).astype(int).tolist())),
                               "stats": st}
        lead = F[:, 16] - F[:, 12]
        self.report["rest"]["lead_cm"] = stat(lead * 100)
        print(f"rest {dur:.1f} s / {len(F)} pkts: pitch {st['pitch']['mean']:+.3f}+-{st['pitch']['std']:.3f} "
              f"roll {st['roll']['mean']:+.3f}+-{st['roll']['std']:.3f} tor1 {st['tor1']['mean']:+.2f}+-"
              f"{st['tor1']['std']:.2f} lead {lead.mean() * 100:+.2f}+-{lead.std() * 100:.3f} cm "
              f"wheel_x range {(F[:, 12].max() - F[:, 12].min()) * 1000:.2f} mm", flush=True)

    def run_head(self) -> None:
        import cv2
        self.enable_drive(vx_max=0.02)
        H = HeadingServo()
        psi0 = self.R.udp.latest().body_yaw_wheel
        self.hold_heading(H, psi0, 0.0, "head_settle0", 2.0)
        for tgt in self.a.head_targets:
            if not self.hold_heading(H, psi0, float(tgt), f"head_{tgt:+.0f}", self.a.hold):
                break
            fr = self.R.frames.get(wait=0.3)
            if fr is not None:
                gray, bgr, t_exp, tel_exp, idx = fr
                name = f"hold_{tgt:+.0f}_{int((t_exp - self.t0) * 1000):06d}.jpg"
                cv2.imwrite(str(self.out / "frames" / name), bgr)
                self.log(event="hold_frame", frame=name, target=tgt, q_exp=None if tel_exp is None else float(tel_exp.q))
        self.hold_heading(H, psi0, 0.0, "head_settle1", 2.0)
        self.report["head"] = {"targets": self.a.head_targets, "hold_s": self.a.hold}
        self.report["head_note"] = "steady state per hold comes out of session_analyze.py (tel.csv + ctrl.jsonl)"

    def run_heading(self) -> None:
        self.enable_drive(vx_max=0.02)
        psi0 = self.R.udp.latest().body_yaw_wheel
        self.hold_heading(HeadingServo(), psi0, 0.0, "heading_settle0", 2.0 * self.a.time_scale)
        res = []
        prev_tgt = 0.0
        ts = self.a.time_scale
        for rnd in range(self.a.rounds):
            for tgt, dur in ((8, 4 * ts), (0, 4 * ts), (-8, 4 * ts), (0, 4 * ts), (15, 5 * ts), (0, 5 * ts)):
                H = HeadingServo()
                te = time.perf_counter() + dur
                seg, peak = [], 0.0
                while time.perf_counter() < te:
                    if not self.ok():
                        break
                    t = self.R.udp.latest()
                    if t is None:
                        time.sleep(0.02)
                        continue
                    lead = float(t.stable_yaw) - t.body_yaw_imu
                    vy, _ = H.step(time.perf_counter(), t.body_yaw_wheel - psi0, lead, float(tgt))
                    self.R.act.set(vx=0.0, vyaw=vy, head=0.0)
                    rel = t.body_yaw_wheel - psi0
                    seg.append(rel)
                    peak = max(peak, abs(lead))
                    self.log(stage=f"heading_{tgt:+d}", target=tgt, psi_rel=rel, lead_yaw=lead, vyaw=vy,
                             pitch=float(t.pitch), roll=float(t.roll), wheel_x=float(t.x_m))
                    time.sleep(0.03)
                if self.guard.tripped:
                    break
                fin = float(np.mean(seg[-10:]))
                step = tgt - prev_tgt
                prev_tgt = float(tgt)
                ov = (max(seg) - tgt) if step > 0 else ((tgt - min(seg)) if step < 0 else 0.0)
                r = {"round": rnd, "target": tgt, "final": round(fin, 2), "final_err": round(abs(fin - tgt), 2),
                     "overshoot": round(ov, 2), "peak_lead": round(peak, 1)}
                res.append(r)
                print("heading", r, flush=True)
            if self.guard.tripped:
                break
        self.report["heading"] = res
        if not self.guard.tripped:
            self.hold_heading(HeadingServo(), psi0, 0.0, "heading_settle1", 2.0 * self.a.time_scale)

    def run_ramp(self) -> None:
        a = self.a
        self.enable_drive(vx_max=0.02)
        H = HeadingServo()
        psi0 = self.R.udp.latest().body_yaw_wheel
        self.hold_heading(H, psi0, 0.0, "ramp_settle0", 2.0)
        z0, z1, rate = self.report["imu_zero_start"], float(a.imu_zero_to), float(a.ramp_rate)
        self.report["ramp"] = {"imu_zero_from": z0, "imu_zero_to": z1, "rate_dps": rate, "completed": False,
                               "slope_prediction_cm_per_deg": round(-K2 / K0 * 100, 3)}

        def ramp(z_from: float, z_to: float, tag: str) -> bool:
            t_start = time.perf_counter()
            span = abs(z_to - z_from)
            z_sent = None
            while True:
                if not self.ok():
                    return False
                done = rate * (time.perf_counter() - t_start) >= span
                z = z_to if done else z_from + math.copysign(rate * (time.perf_counter() - t_start), z_to - z_from)
                if z_sent is None or abs(z - z_sent) >= 0.02 or (done and z_sent != z_to):
                    set_imu_zero(self.base, z)
                    self.imu_zero_changed = True
                    z_sent = z
                self.step_heading(H, psi0, 0.0, tag, imu_zero=round(z, 3))
                if done:
                    return True
                time.sleep(0.03)

        ok = ramp(z0, z1, "ramp_up")
        if ok:
            ok = self.hold_heading(H, psi0, 0.0, "ramp_hold_top", a.hold)
        if ok and not a.no_return:
            ok = ramp(z1, z0, "ramp_down")
        if ok:
            ok = self.hold_heading(H, psi0, 0.0, "ramp_hold_end", a.hold)
        self.report["ramp"]["completed"] = bool(ok)

    def grab(self, name: str) -> None:
        """One frame on demand (the continuous grabber stays off on purpose: /capture holds the httpd ~33 ms)."""
        import cv2
        try:
            s = requests.Session()
            s.trust_env = False
            t_req = time.perf_counter()
            r = s.get(self.base + "/capture?q=80", timeout=1.5)
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_GRAYSCALE)
            fname = f"{name}_{int((t_req - self.t0) * 1000):06d}.jpg"
            if img is not None:
                cv2.imwrite(str(self.out / "frames" / fname), img)
                t = self.R.udp.latest()
                self.log(event="frame", frame=fname, t_req=round(t_req - self.t0, 4),
                         wheel_x=None if t is None else float(t.x_m), q=None if t is None else float(t.q),
                         pitch=None if t is None else float(t.pitch))
        except Exception as exc:
            self.log(event="frame_failed", name=name, err=repr(exc))

    def run_pulse(self) -> None:
        """Open-loop micro pulse: a fixed `vx` for a fixed time, then zero. Finds out which way is forward.

        No feedback on position here on purpose: the sign of wheel_x versus the
        commanded velocity is exactly what this stage measures, so a closed loop
        built on that sign could run away before the measurement is made.
        """
        a = self.a
        self.enable_drive(vx_max=abs(a.vx) + 0.002)
        H = HeadingServo()
        psi0 = self.R.udp.latest().body_yaw_wheel
        self.hold_heading(H, psi0, 0.0, "pulse_settle0", 2.0)
        self.grab("pulse_start")
        t = self.R.udp.latest()
        x0, p0, sp0 = t.x_m, t.pitch, t.stable_pos
        te = time.perf_counter() + a.dur
        while time.perf_counter() < te:
            if not self.ok():
                break
            self.step_heading(H, psi0, 0.0, "pulse_on", vx=float(a.vx))
            time.sleep(0.03)
        if not self.guard.tripped:
            self.hold_heading(H, psi0, 0.0, "pulse_off", a.post)
        self.grab("pulse_end")
        t = self.R.udp.latest()
        r = {"vx_cmd": a.vx, "dur_s": a.dur, "wheel_x_change_mm": round((t.x_m - x0) * 1000, 2),
             "stable_pos_change_mm": round((t.stable_pos - sp0) * 1000, 2),
             "pitch_start": round(float(p0), 2), "pitch_end": round(float(t.pitch), 2)}
        # image shift of the desk region between the two frames (positive dy = scene moved down = robot moved forward)
        import cv2
        fs = sorted((self.out / "frames").glob("pulse_*.jpg"))
        if len(fs) >= 2:
            a_img = cv2.imread(str(fs[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32)[120:, :]
            b_img = cv2.imread(str(fs[-1]), cv2.IMREAD_GRAYSCALE).astype(np.float32)[120:, :]
            (dx, dy), resp = cv2.phaseCorrelate(a_img, b_img)
            r["image_shift_px"] = {"dx": round(dx, 3), "dy": round(-dy, 3), "response": round(resp, 3)}
        self.report["pulse"] = r
        print("pulse", r, flush=True)

    def run_drive(self) -> None:
        a = self.a
        self.enable_drive(vx_max=float(a.vmax))
        H = HeadingServo()
        psi0 = self.R.udp.latest().body_yaw_wheel
        self.hold_heading(H, psi0, 0.0, "drive_settle0", 2.0)
        x_start = self.R.udp.latest().x_m
        self.grab("drive_start")
        goals = [float(a.dist)] + ([0.0] if a.back else [])
        res = []
        for gi, goal in enumerate(goals):
            te = time.perf_counter() + a.timeout
            done_since = None
            t_shot = time.perf_counter()
            while time.perf_counter() < te:
                if not self.ok():
                    break
                t = self.R.udp.latest()
                if t is None:
                    time.sleep(0.02)
                    continue
                if time.perf_counter() - t_shot > 1.0:
                    t_shot = time.perf_counter()
                    self.grab(f"drive_{gi}_mid")
                rem = goal - (t.x_m - x_start)
                if abs(rem) <= a.tol:
                    done_since = done_since or time.perf_counter()
                    if time.perf_counter() - done_since > 1.0:
                        break
                else:
                    done_since = None
                vx = math.copysign(a.vmax * float(np.clip(abs(rem) / 0.05, 0.3, 1.0)), rem)
                self.step_heading(H, psi0, 0.0, f"drive_{gi}", vx=vx, goal=goal, rem=rem)
                time.sleep(0.03)
            if self.guard.tripped:
                break
            self.hold_heading(H, psi0, 0.0, f"drive_settle_{gi}", 1.5)
            self.grab(f"drive_end_{gi}")
            t = self.R.udp.latest()
            r = {"leg": gi, "goal_cm": round(goal * 100, 2), "reached_cm": round((t.x_m - x_start) * 100, 2),
                 "err_cm": round(abs(t.x_m - x_start - goal) * 100, 2), "timed_out": done_since is None,
                 "rest_lead_cm": round((t.stable_pos - t.x_m) * 100, 2), "rest_pitch": round(float(t.pitch), 2)}
            res.append(r)
            print("drive", r, flush=True)
        self.report["drive"] = res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.137.50")
    ap.add_argument("--tlm-port", type=int, default=9001, help="telemetry UDP port (mock_robot uses a random one)")
    ap.add_argument("--stage", required=True, choices=["rest", "head", "heading", "ramp", "drive", "pulse"])
    ap.add_argument("--vx", type=float, default=0.02, help="pulse stage: velocity command (m/s), sign tells front from back")
    ap.add_argument("--dur", type=float, default=0.25, help="pulse stage: how long the velocity command lasts (s)")
    ap.add_argument("--post", type=float, default=3.0, help="pulse stage: zero-command wait before the end frame (s)")
    ap.add_argument("--pos-box", type=float, default=None, help="guard: tighten the position box (m)")
    ap.add_argument("--wheel-box", type=float, default=None, help="guard: tighten the raw wheel_x travel box (m)")
    ap.add_argument("--pitch-dev", type=float, default=None, help="guard: tighten the pitch deviation limit (deg)")
    ap.add_argument("--roll-dev", type=float, default=None, help="guard: tighten the roll deviation limit (deg)")
    ap.add_argument("--psi-box", type=float, default=None, help="guard: tighten the heading excursion limit (deg)")
    ap.add_argument("--secs", type=float, default=60.0, help="rest stage duration")
    ap.add_argument("--hold", type=float, default=3.0, help="ramp stage hold seconds at each end")
    ap.add_argument("--head-targets", type=float, nargs="*", default=[10, 0, -10, 0, 20, 0, -20, 0, 30, 0, -30, 0])
    ap.add_argument("--rounds", type=int, default=1, help="heading stage rounds")
    ap.add_argument("--imu-zero-to", type=float, default=1.0, help="ramp stage: target imu_zero (deg)")
    ap.add_argument("--ramp-rate", type=float, default=0.1, help="ramp stage: deg per second")
    ap.add_argument("--no-return", action="store_true", help="ramp stage: do not ramp back to the start value")
    ap.add_argument("--dist", type=float, default=0.03, help="drive stage: distance (m)")
    ap.add_argument("--back", action="store_true", help="drive stage: come back to the start after reaching")
    ap.add_argument("--vmax", type=float, default=0.03, help="drive stage: speed cap (m/s)")
    ap.add_argument("--tol", type=float, default=0.006, help="drive stage: arrival tolerance (m)")
    ap.add_argument("--timeout", type=float, default=20.0, help="drive stage: per-leg timeout (s)")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="scales the heading-step durations; for rehearsals against mock_robot only")
    a = ap.parse_args()

    s = Session(a)
    try:
        if s.preflight():
            {"rest": s.run_rest, "head": s.run_head, "heading": s.run_heading, "ramp": s.run_ramp,
             "drive": s.run_drive, "pulse": s.run_pulse}[a.stage]()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        s.report["exception"] = repr(exc)
        if s.actuated:
            s.report["exception_safe_stop_ok"] = s._safe_stop()
    finally:
        s.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
