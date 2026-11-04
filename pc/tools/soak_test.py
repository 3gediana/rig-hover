#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""soak_test.py -- sustained-load reliability check against the robot.

The httpd socket-exhaustion bug (see TECHNICAL-REPORT.md §5 #2) was found by
accident: the device stayed pingable while every HTTP request was reset, which
reads like a hang. This script is the regression evidence for that fix --
sustained mixed traffic plus a heap trend, so a leak or a socket leak shows up
as a slope rather than as a failure at an unpredictable moment.

Run:
    python soak_test.py --seconds 600
    python soak_test.py --seconds 60 --rps 20      # quick smoke
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

BASE = "http://192.168.137.50"
# (weight, path). /status is the heaviest handler; /cfg exercises the
# parameter path; /model reads the blob metadata.
MIX = [("/status", 5), ("/cfg", 3), ("/model", 2)]


def get(path: str, timeout: float = 6.0):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            body = r.read()
        return True, (time.perf_counter() - t0) * 1000.0, body
    except Exception as exc:  # noqa: BLE001 - the reason is what we report
        return False, (time.perf_counter() - t0) * 1000.0, repr(exc).encode()


def pick(i: int) -> str:
    total = sum(w for _p, w in MIX)
    k = i % total
    for p, w in MIX:
        if k < w:
            return p
        k -= w
    return MIX[0][0]


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--rps", type=float, default=8.0)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    BASE = a.base.rstrip("/")

    ok, lat, body = get("/status")
    if not ok:
        print(f"robot unreachable at {BASE}: {body.decode(errors='replace')}")
        return 2
    try:
        heap0 = json.loads(body)["heap"]
        uptime0 = json.loads(body)["t"]
    except Exception:
        print("unexpected /status payload; is the firmware the patched build?")
        return 2
    print(f"robot up {uptime0/1000:.0f}s, heap {heap0} B -- starting "
          f"{a.seconds}s soak at {a.rps} req/s\n")

    interval = 1.0 / a.rps
    latencies: list[float] = []
    failures: list[str] = []
    heap_samples: list[tuple[float, int]] = []
    n = 0
    t_end = time.perf_counter() + a.seconds
    next_heap = time.perf_counter()

    while time.perf_counter() < t_end:
        target = time.perf_counter() + interval
        path = pick(n)
        good, ms, payload = get(path)
        n += 1
        if good:
            latencies.append(ms)
        else:
            failures.append(f"{path}: {payload.decode(errors='replace')[:90]}")

        if time.perf_counter() >= next_heap:
            g, _ms, b = get("/status")
            if g:
                try:
                    j = json.loads(b)
                    heap_samples.append((time.perf_counter(), j["heap"]))
                except Exception:
                    pass
            next_heap = time.perf_counter() + max(5.0, a.seconds / 20.0)

        sleep = target - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

    def pct(xs, p):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p / 100.0))]

    bad = len(failures)
    print("=" * 70)
    print(f"requests      {n}   ({n/a.seconds:.1f}/s achieved)")
    print(f"failures      {bad}   ({100.0*bad/max(n,1):.2f}%)")
    if latencies:
        print(f"latency ms    p50 {pct(latencies,50):.0f}  p90 {pct(latencies,90):.0f}  "
              f"p99 {pct(latencies,99):.0f}  max {max(latencies):.0f}")
    if len(heap_samples) >= 2:
        hs = [h for _t, h in heap_samples]
        drift = hs[-1] - hs[0]
        per_min = drift / max(a.seconds / 60.0, 1e-9)
        print(f"heap B        first {hs[0]}  last {hs[-1]}  "
              f"drift {drift:+d}  ({per_min:+.0f} B/min)")
        print(f"heap range    min {min(hs)}  max {max(hs)}")
        verdict = "stable" if abs(per_min) < 5000 else "DRIFTING"
        print(f"heap verdict  {verdict}")
    print("=" * 70)
    if failures:
        print("\nthe first 5 failures:")
        for f in failures[:5]:
            print("  " + f)
    print("\nPASS" if bad == 0 else "\nFAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
