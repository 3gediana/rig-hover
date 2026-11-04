# -*- coding: utf-8 -*-
"""robot_watch.py -- wait for the robot to come back on WiFi, then say so.

The robot is on USB only right now: the hotspot adapter and ICS are both up, but
nothing answers on the 192.168.137.0/24 subnet, so its station never associated
(or it is powered off and charging). Polling is the only way to notice it return,
and it will otherwise be noticed only when someone happens to try.

Exits as soon as the HTTP API answers, or after --minutes. Prints one line per
state change, so the log stays readable when read hours later.

Run:  python robot_watch.py --ip 192.168.137.50 --minutes 240
"""
from __future__ import annotations

import argparse
import datetime
import json
import socket
import sys
import time
import urllib.request


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def http_json(ip: str, path: str, timeout: float) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://{ip}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def port_open(ip: str, port: int, timeout: float) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.137.50")
    ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--period", type=float, default=45,
                    help="seconds between probes")
    a = ap.parse_args()

    deadline = time.time() + a.minutes * 60
    print(f"[{now()}] watching {a.ip} for up to {a.minutes:.0f} min "
          f"(period {a.period:.0f}s)", flush=True)
    was_open = False
    n = 0
    while time.time() < deadline:
        n += 1
        open80 = port_open(a.ip, 80, 1.2)
        if open80 != was_open:
            print(f"[{now()}] port 80 {'OPEN' if open80 else 'closed'}", flush=True)
            was_open = open80
        if open80:
            st = http_json(a.ip, "/status", 2.5)
            if st is not None:
                up = st.get("uptime_s") or st.get("uptime") or st.get("up")
                print(f"[{now()}] ROBOT IS BACK -- /status answered "
                      f"({len(json.dumps(st))} bytes, uptime={up})", flush=True)
                print("STATUS_JSON " + json.dumps(st)[:1500], flush=True)
                return 0
            print(f"[{now()}] port 80 open but /status did not parse", flush=True)
        elif n % 40 == 0:
            print(f"[{now()}] still nothing after {n} probes", flush=True)
        time.sleep(a.period)

    print(f"[{now()}] gave up after {a.minutes:.0f} min", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
