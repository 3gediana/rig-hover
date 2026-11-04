#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e2e_latency.py -- measure the real perception->robot latency, end to end.

TECH-PLAN-v1 §15-V6 lists this as a must-measure and it was never run: every
number reported so far is a component, not the chain. This drives the actual
loop against the live robot:

    GET /capture  ->  decode  ->  YOLO26n-pose + 548-d features + adapter  ->  POST /feat

and reports per-stage and total percentiles, plus the robot's own view of the
frame age so the two clocks can be compared.

Run:
    python e2e_latency.py --n 100
    python e2e_latency.py --n 50 --no-upload     # capture+perceive only
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import socket
import statistics
import struct
import sys
import time

import numpy as np

sys.path.insert(0, r"D:\robot_work\pc\perception")

BASE = "http://192.168.137.50"
_HOST = "192.168.137.50"
_PORT = 80
MAGIC = 0xFE01


def http_get(path: str, timeout: float = 10.0) -> bytes:
    return _request("GET", path, None, timeout)


def http_post(path: str, body: bytes, timeout: float = 10.0) -> bytes:
    return _request("POST", path, body, timeout)


# A persistent connection matters more than it looks: a fresh TCP handshake per
# request costs tens of ms on this link, which swamps the actual payload cost
# (a /feat frame is ~200 bytes). Measuring with urllib's per-request connection
# would attribute connection setup to the stage and hide the real budget.
_CONN: dict = {"conn": None}


def _connect(timeout: float):
    if _CONN["conn"] is None:
        c = http.client.HTTPConnection(_HOST, _PORT, timeout=timeout)
        c.connect()
        # Small request+response pairs are exactly the case Nagle's algorithm
        # and delayed ACK punish: the client's body write waits for an ACK of
        # the header write, which the peer delays by tens of milliseconds. That
        # shows up as "a 200-byte POST took 56 ms" and has nothing to do with
        # how fast the robot parses it.
        try:
            c.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        _CONN["conn"] = c
    return _CONN["conn"]


def _request(method: str, path: str, body: bytes | None,
             timeout: float) -> bytes:
    for attempt in (0, 1):
        conn = _connect(timeout)
        try:
            conn.request(method, path, body=body,
                         headers={"Content-Type": "application/octet-stream"}
                         if body is not None else {})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return data
        except Exception:
            # One transparent reconnect: an abandoned keep-alive slot on the
            # robot side must not be counted as a request failure.
            try:
                conn.close()
            except Exception:
                pass
            _CONN["conn"] = None
            if attempt:
                raise
    raise RuntimeError("unreachable")


def http_step(body: bytes, timeout: float = 10.0) -> tuple[bytes, str]:
    """POST /step -> (jpeg, X-Cfc json). One round trip for observe+reply."""
    for attempt in (0, 1):
        conn = _connect(timeout)
        try:
            conn.request("POST", "/step", body=body,
                         headers={"Content-Type": "application/octet-stream"})
            resp = conn.getresponse()
            data = resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return data, resp.getheader("X-Cfc") or "{}"
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _CONN["conn"] = None
            if attempt:
                raise
    raise RuntimeError("unreachable")


def build_feat_frame(perc, t_ms: int) -> bytes:
    """TECH-PLAN-v1 §5.2 wire format. Kept byte-for-byte in sync with the
    firmware parser; any drift shows up as a rejected frame on /feat."""
    feat = np.asarray(perc.feat_64, dtype=np.float32).reshape(-1)
    n = int(feat.size)
    scale = 1.0 / 32767.0
    q = np.clip(np.round(feat / scale), -32767, 32767).astype(np.int16)

    seen = int(getattr(perc, "seen", 0))
    n_obj = 1 if seen else 0
    out = bytearray()
    out += struct.pack("<HHI", MAGIC, n_obj, t_ms & 0xFFFFFFFF)
    out += struct.pack("<HH", n, 0)
    if n_obj:
        flags = (int(getattr(perc, "hand_up_L", 0) & 1)
                 | ((int(getattr(perc, "hand_up_R", 0)) & 1) << 1))
        out += struct.pack(
            "<hhhhBBBBhh",
            int(round(float(perc.person_cx) * 1000.0)),
            int(round(0.5 * 1000.0)),
            int(round(float(perc.person_w) * 1000.0)),
            int(round(float(perc.person_h) * 1000.0)),
            int(round(float(perc.person_conf) * 255.0)) & 0xFF,
            0,
            flags,
            0,
            int(round(float(getattr(perc, "wave_sin", 0.0)) * 1000.0)),
            int(round(float(getattr(perc, "wave_cos", 0.0)) * 1000.0)),
        )
    out += struct.pack("<f", scale)
    out += q.tobytes()
    return bytes(out)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p / 100.0))]


def main() -> int:
    global BASE, _HOST, _PORT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--step", action="store_true",
                    help="benchmark POST /step (observation out, JPEG back in "
                         "one round trip) instead of GET /capture + POST /feat")
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    _rest = BASE.split("://", 1)[-1]
    _HOST, _, _p = _rest.partition(":")
    _PORT = int(_p or 80)

    try:
        st = json.loads(http_get("/status"))
    except Exception as exc:  # noqa: BLE001
        print(f"robot unreachable: {exc}")
        return 2
    print(f"robot up {st['t']/1000:.0f}s  heap {st['heap']}  "
          f"cfc.src={st['cfc']['src']}\n")

    from run_perception import PerceptionPipeline  # noqa: E402

    pipe = PerceptionPipeline()
    import cv2
    print("warmup...")
    seed_frame = None
    for _ in range(a.warmup):
        img = cv2.imdecode(np.frombuffer(http_get("/capture"), np.uint8),
                           cv2.IMREAD_COLOR)
        seed_frame = build_feat_frame(pipe(img),
                                      int(time.time() * 1000) & 0xFFFFFFFF)

    rows = []
    print(f"measuring {a.n} frames\n")
    prev_frame = seed_frame
    for i in range(a.n):
        if a.step:
            # Pipelined: the observation computed for frame N-1 goes out with the
            # request that fetches frame N, so one round trip covers both.
            t0 = time.perf_counter()
            jpg, xcfc = http_step(prev_frame)
            t1 = time.perf_counter()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            t2 = time.perf_counter()
            perc = pipe(img)
            t3 = time.perf_counter()
            prev_frame = build_feat_frame(perc,
                                          int(time.time() * 1000) & 0xFFFFFFFF)
            try:
                reply = json.loads(xcfc)
            except Exception:  # noqa: BLE001 - header is advisory
                reply = {"ok": 0}
            rows.append({
                "step_ms": (t1 - t0) * 1000,
                "decode_ms": (t2 - t1) * 1000,
                "perceive_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
                "jpeg_bytes": len(jpg),
                "feat_ok": 1 if reply.get("ok", 0) else 0,
                "age_ms": reply.get("age_ms"),
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n}")
            continue

        t0 = time.perf_counter()
        jpg = http_get("/capture")
        t1 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        t2 = time.perf_counter()
        perc = pipe(img)
        t3 = time.perf_counter()
        t4, t5, reply = t3, t3, None
        if not a.no_upload:
            frame = build_feat_frame(perc, int(time.time() * 1000) & 0xFFFFFFFF)
            try:
                t4 = time.perf_counter()
                r = http_post("/feat", frame)
                t5 = time.perf_counter()
                reply = json.loads(r)
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": 0, "err": str(exc)[:80]}
                t5 = time.perf_counter()
        rows.append({
            "capture_ms": (t1 - t0) * 1000,
            "decode_ms": (t2 - t1) * 1000,
            "perceive_ms": (t3 - t2) * 1000,
            "upload_ms": (t5 - t4) * 1000 if not a.no_upload else 0.0,
            "total_ms": (t5 - t0) * 1000 if not a.no_upload else (t3 - t0) * 1000,
            "jpeg_bytes": len(jpg),
            "feat_ok": 1 if (a.no_upload or (reply or {}).get("ok", 0)) else 0,
            "age_ms": (reply or {}).get("age_ms") if reply else None,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{a.n}")

    print("\n" + "=" * 74)
    print(f"{'stage':<14}{'p50':>9}{'p90':>9}{'p99':>9}{'mean':>9}")
    print("-" * 74)
    if a.step:
        stages = (("step_ms", "POST /step"), ("decode_ms", "jpeg decode"),
                  ("perceive_ms", "YOLO+features"))
    else:
        stages = (("capture_ms", "GET /capture"), ("decode_ms", "jpeg decode"),
                  ("perceive_ms", "YOLO+features"), ("upload_ms", "POST /feat"))
    for key, label in stages:
        xs = [r[key] for r in rows]
        print(f"{label:<14}{pct(xs,50):>9.1f}{pct(xs,90):>9.1f}"
              f"{pct(xs,99):>9.1f}{statistics.mean(xs):>9.1f}")

    tot = [r["total_ms"] for r in rows]
    print("-" * 74)
    print(f"{'TOTAL':<14}{pct(tot,50):>9.1f}{pct(tot,90):>9.1f}"
          f"{pct(tot,99):>9.1f}{statistics.mean(tot):>9.1f}")
    print("=" * 74)
    print(f"achievable rate  {1000.0/statistics.mean(tot):.1f} Hz "
          f"(sensor-to-decision, one frame in flight)")
    print(f"jpeg size        mean {statistics.mean([r['jpeg_bytes'] for r in rows])/1024:.1f} KiB")

    if not a.no_upload:
        ok = sum(r["feat_ok"] for r in rows)
        ages = [r["age_ms"] for r in rows if r["age_ms"] is not None]
        print(f"{'/step' if a.step else '/feat'} accepted   {ok}/{len(rows)}")
        print(f"robot-side frame age (age_ms)  "
              f"{('p50 %.0f' % pct(ages,50)) if ages else 'n/a'}")
        if ages and abs(pct(ages, 50)) > 5000:
            print("  note: age_ms compares the PC wall clock against robot uptime; "
                  "the two domains are unrelated, so only the spread is meaningful")

    # Regression gates from TECH-PLAN-v1 §12.1
    print()
    budget = 150.0
    ok_total = pct(tot, 95) < budget
    print(f"V1 端到端 P95 < {budget:.0f} ms : "
          f"{'PASS' if ok_total else 'FAIL'}  (P95={pct(tot,95):.0f} ms)")
    return 0 if ok_total else 1


if __name__ == "__main__":
    sys.exit(main())
