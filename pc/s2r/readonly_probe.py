# -*- coding: utf-8 -*-
"""s2r/readonly_probe.py -- strictly read-only probe of a powered-on RIG-Hover.

Sends nothing that can actuate:
  * subscribes to the TLM1 UDP telemetry stream (a 2-byte "hi" datagram, the
    same subscription `link.UdpTel` always does),
  * GET /status, GET /cfg, GET /api/data, a handful of GET /capture frames.
No POST, no /mode, no /cmd, no serial port. Use it to answer "what state is
the robot in right now" before anything else, and to bank identification logs
for the digital twin (every one of the 34 telemetry floats is kept).

Usage:
    python pc/s2r/readonly_probe.py [--host 192.168.137.50] [--secs 10] [--frames 5] [--tag name]
Output:
    runs_end_side/s2r_live/probe_<stamp>[_<tag>]/{tel.csv,summary.json,status.json,cfg.json,api_data.json,frame_XX.jpg}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import requests

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tellog import FIELDS, RawTel, TelCsv  # noqa: E402

OUT_ROOT = HERE.parent.parent / "runs_end_side" / "s2r_live"


def _stat(a: np.ndarray) -> dict:
    if a.size == 0:
        return {}
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95))}


def summarize(F: np.ndarray, up_s: np.ndarray, seq: np.ndarray, lost: int) -> dict:
    """Rest statistics from an (N, 34) telemetry block. Robot-uptime based, so Wi-Fi jitter does not count."""
    col = {n: F[:, i] for i, n in enumerate(FIELDS)}
    dur = float(up_s[-1] - up_s[0])

    def rate(x: np.ndarray) -> tuple[float, float]:
        if dur < 0.5:
            return 0.0, 0.0
        k, b = np.polyfit(up_s, x, 1)
        return float(k), float(np.std(x - (k * up_s + b)))

    body_imu = col["yaw"] - col["q_head"]
    k_imu, r_imu = rate(body_imu)
    k_wx, r_wx = rate(col["wheel_x"])
    lead = col["stable_pos"] - col["wheel_x"]
    temp_u = 0.5 * (col["tor2"] - col["tor1"])       # tor1 = -temp_u + yaw_u, tor2 = +temp_u + yaw_u
    return {
        "duration_robot_s": dur, "packets": int(len(F)), "rate_hz": float(len(F) / dur) if dur > 0 else 0.0,
        "seq_lost": lost,
        "robot_state_values": sorted(set(np.round(col["robot_state"]).astype(int).tolist())),
        "src_values": sorted(set(np.round(col["src"]).astype(int).tolist())),
        "control_mode_values": sorted(set(np.round(col["control_mode"]).astype(int).tolist())),
        "arb_mode_values": sorted(set(np.round(col["arb_mode"]).astype(int).tolist())),
        "rest_lead_cm": _stat(lead * 100), "temp_u": _stat(temp_u), "yaw_u": _stat(0.5 * (col["tor1"] + col["tor2"])),
        "heading": {"imu_yaw_deg_start": float(col["yaw"][0]), "imu_yaw_deg_end": float(col["yaw"][-1]),
                    "body_yaw_imu_rate_dps": k_imu, "body_yaw_imu_fit_resid_deg": r_imu,
                    "wheel_x_rate_mps": k_wx, "wheel_x_fit_resid_m": r_wx},
        "stats": {n: _stat(col[n]) for n in FIELDS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="192.168.137.50")
    ap.add_argument("--secs", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"probe_{stamp}{('_' + a.tag) if a.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://{a.host}"
    s = requests.Session()
    s.trust_env = False          # the dev box has a system proxy; never use it for the robot

    csv = TelCsv(str(out / "tel.csv"))
    tel = RawTel(a.host, on_pkt=csv)
    t0 = time.perf_counter()
    frame_bytes = []
    try:
        for name, path in (("status.json", "/status"), ("cfg.json", "/cfg"), ("api_data.json", "/api/data")):
            try:
                with open(out / name, "wb") as fh:
                    fh.write(s.get(base + path, timeout=2.0).content)
            except Exception as exc:  # the probe must still bank what it can
                (out / name).write_text(f"GET {path} failed: {exc!r}\n", encoding="utf-8")
        for i in range(a.frames):
            if i:
                time.sleep(max(0.0, a.secs - 1.0) / max(1, a.frames - 1))
            try:
                r = s.get(base + "/capture?q=80", timeout=2.0)
                (out / f"frame_{i:02d}.jpg").write_bytes(r.content)
                frame_bytes.append(len(r.content))
            except Exception as exc:
                frame_bytes.append(-1)
                print(f"frame {i}: {exc!r}")
        while time.perf_counter() - t0 < a.secs:
            time.sleep(0.1)
    finally:
        tel.stop = True
        time.sleep(0.3)
        csv.close()

    if len(tel.raw) < 20:
        summary = {"error": "no telemetry packets received", "out": str(out)}
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(json.dumps(summary))
        return 1
    F = np.array([r[3] for r in tel.raw], dtype=float)
    up_s = np.array([r[2] for r in tel.raw], dtype=float) / 1000.0
    seq = np.array([r[1] for r in tel.raw], dtype=np.int64)
    lost = int(max(0, int(seq[-1]) - int(seq[0]) + 1 - len(seq)))
    summary = summarize(F, up_s, seq, lost)
    summary.update({"out": str(out), "host": a.host, "frame_bytes": frame_bytes})
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"out: {out}")
    print(f"packets {len(F)} over {summary['duration_robot_s']:.2f} s robot time = {summary['rate_hz']:.1f} Hz, "
          f"seq lost {lost}")
    print(f"robot_state {summary['robot_state_values']}  src {summary['src_values']}  "
          f"control_mode {summary['control_mode_values']}  arb_mode {summary['arb_mode_values']}")
    st = summary["stats"]
    print(f"rest lead (stable_pos - wheel_x): {summary['rest_lead_cm']['mean']:+.2f} cm "
          f"(std {summary['rest_lead_cm']['std']:.2f}); temp_u {summary['temp_u']['mean']:+.2f} "
          f"+-{summary['temp_u']['std']:.2f}; yaw_u {summary['yaw_u']['mean']:+.2f}")
    print(f"body yaw (IMU, head-corrected) rate: {summary['heading']['body_yaw_imu_rate_dps']:+.3f} deg/s "
          f"(resid {summary['heading']['body_yaw_imu_fit_resid_deg']:.3f} deg); "
          f"wheel_x rate {summary['heading']['wheel_x_rate_mps'] * 100:+.3f} cm/s")
    for n in ("roll", "pitch", "dq", "pit_gyro", "tor1", "tor2", "q_head", "stable_yaw", "feat_age_ms", "danger"):
        v = st[n]
        print(f"  {n:14s} mean {v['mean']:+9.3f}  std {v['std']:7.3f}  min {v['min']:+9.3f}  max {v['max']:+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
