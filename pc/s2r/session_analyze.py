# -*- coding: utf-8 -*-
"""s2r/session_analyze.py - turn a guarded_session log into numbers.

Reads what guarded_session.py wrote (tel.csv at 500 Hz, ctrl.jsonl at 30 Hz,
summary.json) and prints per-stage metrics. Everything here is real-robot
evidence (T4); nothing is simulated.

Usage:
    python pc/s2r/session_analyze.py <session_dir> [--json out.json]
    python pc/s2r/session_analyze.py runs_end_side/s2r_live --latest head
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tellog import read_tel_csv  # noqa: E402

K0, K2 = 1300.0, 25.0


def load(d: pathlib.Path):
    rep = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    tel = read_tel_csv(str(d / "tel.csv"))
    ctrl = [json.loads(line) for line in open(d / "ctrl.jsonl", encoding="utf-8") if line.strip()]
    return rep, tel, ctrl


def tel_arrays(tel, rep=None):
    """Telemetry columns with `recv_s` re-zeroed to the session clock (ctrl.jsonl `t`)."""
    off = 0.0
    z = (rep or {}).get("t0_perf")
    if z:
        off = float(z["tel"]) - float(z["session"])      # recv_s is relative to z["tel"]; session time = recv_s - off
    t = np.array(tel["recv_s"]) - off
    col = {k: np.array(v) for k, v in tel.items()}
    return t, col


def window(t, a, t0, t1):
    m = (t >= t0) & (t < t1)
    return a[m]


def ctrl_windows(ctrl, prefix, exclude=("settle",)):
    """Split the 30 Hz control log into per-tag runs (a tag restarts after another tag)."""
    runs = []
    cur = None
    for r in ctrl:
        tag = r.get("stage", "")
        if not tag.startswith(prefix) or any(e in tag for e in exclude):
            if cur:
                runs.append(cur)
                cur = None
            continue
        if cur is None or cur["tag"] != tag:
            if cur:
                runs.append(cur)
            cur = {"tag": tag, "rows": []}
        cur["rows"].append(r)
    if cur:
        runs.append(cur)
    return runs


def spikes(t, a, thr=5.0):
    if len(a) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(a)) > thr))


def analyze_head(rep, tel, ctrl, out):
    t, c = tel_arrays(tel, rep)
    res = []
    # ctrl `t` and tel `recv_s` share the perf_counter epoch up to a few ms
    # (two zero points taken at session start); windows are >= 1 s wide.
    for run in ctrl_windows(ctrl, "head_"):
        rows = run["rows"]
        t0, t1 = rows[0]["t"], rows[-1]["t"]
        tail = max(t0, t1 - 1.0)
        m = (t >= tail) & (t <= t1)
        q_tail = float(np.mean(c["q_head"][m])) if m.any() else float("nan")
        qt_tail = float(np.mean(c["target_head_pos"][m])) if m.any() else float("nan")
        mfull = (t >= t0) & (t <= t1)
        cmd = float(np.mean([r["head_cmd"] for r in rows[-20:]]))
        r0 = {
            "tag": run["tag"], "cmd_deg": cmd, "q_ss_deg": round(q_tail, 2), "q_target_ss_deg": round(qt_tail, 2),
            "err_deg": round(q_tail - cmd, 2),
            "body_yaw_change_deg": round(float(c["yaw"][mfull][-1] - c["yaw"][mfull][0]) - float(c["q_head"][mfull][-1] - c["q_head"][mfull][0]), 2) if mfull.any() else None,
            "wheel_x_change_mm": round(float(c["wheel_x"][mfull][-1] - c["wheel_x"][mfull][0]) * 1000, 2) if mfull.any() else None,
            "pitch_mean": round(float(np.mean(c["pitch"][mfull])), 2) if mfull.any() else None,
            "pitch_absmax_dev": round(float(np.max(np.abs(c["pitch"][mfull] - c["pitch"][mfull][0]))), 2) if mfull.any() else None,
            "roll_mean": round(float(np.mean(c["roll"][mfull])), 2) if mfull.any() else None,
            "roll_spikes": int(spikes(t[mfull], c["roll"][mfull])) if mfull.any() else None,
            "temp_u_mean": round(float(np.mean(0.5 * (c["tor2"][mfull] - c["tor1"][mfull]))), 2) if mfull.any() else None,
            "yaw_u_absmax": round(float(np.max(np.abs(0.5 * (c["tor1"][mfull] + c["tor2"][mfull])))), 2) if mfull.any() else None,
            "tor_absmax": round(float(np.max(np.abs(np.r_[c["tor1"][mfull], c["tor2"][mfull]]))), 1) if mfull.any() else None,
            "frames": [r["frame"] for r in rows if r.get("frame")],
        }
        res.append(r0)
    out["head"] = res
    print(f"{'tag':>10} {'cmd':>6} {'q_ss':>7} {'err':>6} {'yaw_ch':>7} {'wheel_mm':>9} {'pitch':>6} {'roll':>6} {'|tor|max':>8}")
    for r in res:
        print(f"{r['tag']:>10} {r['cmd_deg']:+6.1f} {r['q_ss_deg']:+7.2f} {r['err_deg']:+6.2f} "
              f"{r['body_yaw_change_deg']:+7.2f} {r['wheel_x_change_mm']:+9.2f} {r['pitch_mean']:+6.2f} "
              f"{r['roll_mean']:+6.2f} {r['tor_absmax']:8.1f}")
    # dead band estimate: holds at cmd 0 after an excursion, by where it came from
    zeros = [r for r in res if abs(r["cmd_deg"]) < 0.5]
    if zeros:
        print("holds at cmd 0 after an excursion: " + ", ".join(f"{r['tag']} -> q_ss {r['q_ss_deg']:+.2f}" for r in zeros))
    return out


def analyze_heading(rep, tel, ctrl, out):
    t, c = tel_arrays(tel, rep)
    res = []
    runs = ctrl_windows(ctrl, "heading_", exclude=("settle",))
    psi_abs = -0.2190 * (c["wheel1_x"] - c["wheel2_x"])     # body heading from the wheel difference, deg
    # the controller's reference is the heading at stage start; recover it from the first window
    first = runs[0]["rows"][0]["t"] if runs else 0.0
    psi0 = float(psi_abs[np.argmin(np.abs(t - first))])
    for run in runs:
        rows = run["rows"]
        t0, t1 = rows[0]["t"], rows[-1]["t"]
        tgt = float(rows[0]["target"])
        m = (t >= t0) & (t <= t1)
        psi = psi_abs[m] - psi0
        lead = c["stable_yaw"][m] - (c["yaw"][m] - c["q_head"][m])
        fin = float(np.mean(psi[-500:]))
        # time to come within 2 deg of target and stay
        inside = np.abs(psi - tgt) < 2.0
        settle_t = None
        for i in range(len(psi)):
            if inside[i:].all():
                settle_t = float(t[m][i] - t0)
                break
        r = {"tag": run["tag"], "target": tgt, "final_deg": round(fin, 2), "final_err_deg": round(abs(fin - tgt), 2),
             "overshoot_deg": round(float(max(psi.max() - tgt, tgt - psi.min(), 0.0)), 2),
             "peak_lead_deg": round(float(np.max(np.abs(lead))), 1), "settle_within2deg_s": None if settle_t is None else round(settle_t, 2),
             "pitch_absmax": round(float(np.max(np.abs(c["pitch"][m]))), 2),
             "temp_u_absmax": round(float(np.max(np.abs(0.5 * (c["tor2"][m] - c["tor1"][m])))), 1),
             "yaw_u_absmax": round(float(np.max(np.abs(0.5 * (c["tor1"][m] + c["tor2"][m])))), 1),
             "wheel_x_change_mm": round(float(c["wheel_x"][m][-1] - c["wheel_x"][m][0]) * 1000, 2),
             "roll_spikes": int(spikes(t[m], c["roll"][m]))}
        res.append(r)
    out["heading"] = res
    print(f"{'tag':>12} {'target':>6} {'final':>7} {'err':>5} {'overshoot':>9} {'peaklead':>8} {'settle_s':>8} "
          f"{'|pitch|':>7} {'|temp_u|':>8} {'|yaw_u|':>7} {'wheel_mm':>9} {'spikes':>6}")
    for r in res:
        s = r["settle_within2deg_s"]
        print(f"{r['tag']:>12} {r['target']:+6.0f} {r['final_deg']:+7.2f} {r['final_err_deg']:5.2f} "
              f"{r['overshoot_deg']:9.2f} {r['peak_lead_deg']:8.1f} {('  n/a' if s is None else f'{s:8.2f}')} "
              f"{r['pitch_absmax']:7.2f} {r['temp_u_absmax']:8.1f} {r['yaw_u_absmax']:7.1f} "
              f"{r['wheel_x_change_mm']:+9.2f} {r['roll_spikes']:6d}")
    return out


def analyze_ramp(rep, tel, ctrl, out):
    t, c = tel_arrays(tel, rep)
    rows = [r for r in ctrl if str(r.get("stage", "")).startswith("ramp_") and r.get("imu_zero") is not None]
    if not rows:
        return out
    z = np.array([r["imu_zero"] for r in rows])
    lead = np.array([r["lead_pos"] for r in rows]) * 100
    pitch = np.array([r["pitch"] for r in rows])
    wx = np.array([r["wheel_x"] for r in rows])
    tr = np.array([r["t"] for r in rows])
    up = [i for i in range(1, len(z)) if z[i] > z[i - 1]]
    dn = [i for i in range(1, len(z)) if z[i] < z[i - 1]]
    res = {}
    for name, idx in (("up", up), ("down", dn)):
        if len(idx) < 20:
            continue
        zz, ll, pp, ww = z[idx], lead[idx], pitch[idx], wx[idx]
        # 1 s block medians: quasi-static points
        nb = max(2, len(zz) // int(max(1.0, (tr[idx][-1] - tr[idx][0]))))
        blocks = []
        for b in range(nb):
            s = slice(b * len(zz) // nb, (b + 1) * len(zz) // nb)
            blocks.append((float(np.median(zz[s])), float(np.median(ll[s])), float(np.median(pp[s]))))
        B = np.array(blocks)
        k_lead = float(np.polyfit(B[:, 0], B[:, 1], 1)[0]) if len(B) > 2 else float("nan")
        res[name] = {"slope_cm_per_deg_lead": round(k_lead, 3), "slope_cm_per_deg_wheel": round(float(np.polyfit(zz, ww * 100, 1)[0]), 3),
                     "slope_deg_per_deg_pitch": round(float(np.polyfit(zz, pp, 1)[0]), 3),
                     "blocks_z_lead_pitch": [[round(a, 3), round(b, 2), round(c_, 3)] for a, b, c_ in B],
                     "wheel_travel_cm": round(float(ww[-1] - ww[0]) * 100, 2), "imu_zero_range": [round(float(zz[0]), 3), round(float(zz[-1]), 3)],
                     "lead_range_cm": [round(float(ll.min()), 2), round(float(ll.max()), 2)]}
    out["ramp"] = res
    print(f"{'dir':>5} {'slope lead/deg (cm)':>20} {'slope pitch/deg':>16} {'wheel travel cm':>15} {'z range':>16} {'lead range cm':>16}")
    for name, r in res.items():
        print(f"{name:>5} {r['slope_cm_per_deg_lead']:+20.3f} {r['slope_deg_per_deg_pitch']:+16.3f} "
              f"{r['wheel_travel_cm']:+15.2f} {str(r['imu_zero_range']):>16} {str(r['lead_range_cm']):>16}")
    print(f"prediction: lead slope = -K2/K0 = {-K2 / K0 * 100:+.3f} cm/deg (firmware LQR), pitch slope 0 if the wheels are free, "
          f"-1 if the wheels stay stuck")
    return out


def analyze_drive(rep, tel, ctrl, out):
    t, c = tel_arrays(tel, rep)
    res = []
    for run in ctrl_windows(ctrl, "drive_", exclude=("settle",)):
        rows = run["rows"]
        t0, t1 = rows[0]["t"], rows[-1]["t"]
        m = (t >= t0) & (t <= t1)
        x = c["wheel_x"][m]
        vx = c["vx"][m]
        goal = rows[0].get("goal")
        moved = np.abs(x - x[0]) > 0.003
        t_move = float(t[m][np.argmax(moved)] - t0) if moved.any() else None
        r = {"tag": run["tag"], "goal_cm": None if goal is None else round(goal * 100, 2),
             "wheel_travel_cm": round(float(x[-1] - x[0]) * 100, 2),
             "first_motion_s": None if t_move is None else round(t_move, 2),
             "peak_speed_mps": round(float(np.max(np.abs(np.diff(x) / np.maximum(1e-3, np.diff(t[m]))))), 4) if len(x) > 2 else None,
             "lead_cm_range": [round(float(np.min(c["stable_pos"][m] - x)) * 100, 2), round(float(np.max(c["stable_pos"][m] - x)) * 100, 2)],
             "pitch_range": [round(float(c["pitch"][m].min()), 2), round(float(c["pitch"][m].max()), 2)],
             "temp_u_absmax": round(float(np.max(np.abs(0.5 * (c["tor2"][m] - c["tor1"][m])))), 1),
             "roll_spikes": int(spikes(t[m], c["roll"][m])), "duration_s": round(t1 - t0, 2)}
        res.append(r)
    out["drive"] = res
    for r in res:
        print(r)
    return out


def analyze_rest(rep, tel, ctrl, out):
    t, c = tel_arrays(tel, rep)
    rest = rep.get("rest", {})
    out["rest_summary"] = {k: rest.get(k) for k in ("duration_s", "packets", "rate_hz", "seq_lost", "lead_cm")}
    print(json.dumps(out["rest_summary"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="session dir, or runs_end_side/s2r_live with --latest")
    ap.add_argument("--latest", default=None, help="stage name: analyze the newest session_*_<stage>")
    ap.add_argument("--json", default=None, help="write the analysis next to the log as analysis.json")
    a = ap.parse_args()
    root = pathlib.Path(a.path)
    if a.latest:
        cands = sorted(root.glob(f"session_*_{a.latest}"))
        if not cands:
            print(f"no session_*_{a.latest} under {root}")
            return 2
        d = cands[-1]
    else:
        d = root
    rep, tel, ctrl = load(d)
    out = {"session": str(d), "stage": rep.get("stage"), "abort": rep.get("abort"), "guard": rep.get("guard")}
    print(f"session {d.name}  stage {out['stage']}  abort '{out['abort']}'  "
          f"max_pos {out['guard']['max_pos_cm'] if out['guard'] else None} cm  spikes {out['guard'].get('angle_spikes') if out['guard'] else None}")
    fn = {"head": analyze_head, "heading": analyze_heading, "ramp": analyze_ramp, "drive": analyze_drive, "rest": analyze_rest}
    if rep.get("stage") in fn:
        fn[rep["stage"]](rep, tel, ctrl, out)
    dst = pathlib.Path(a.json) if a.json else d / "analysis.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("analysis ->", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
