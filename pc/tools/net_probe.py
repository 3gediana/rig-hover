# -*- coding: utf-8 -*-
"""net_probe.py -- is the robot reachable, and if not, why?

Checks the three things that have to hold for the robot's HTTP API to answer,
in the order they fail:

  1. the hotspot adapter still owns 192.168.137.1
  2. Internet Connection Sharing is running (it is what hands out DHCP leases)
  3. a host actually answers on the subnet

Quiet by design: one ARP-ish sweep with a short timeout, then an HTTP probe of
the recorded address. Nothing is written to the device.

Run:  python net_probe.py [--ip 192.168.137.50]
"""
from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SUBNET = "192.168.137."


def tcp_probe(ip: str, port: int, timeout: float) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def sweep(timeout: float = 0.4, workers: int = 128):
    """Return hosts with the HTTP port open."""
    def one(i: int):
        ip = f"{SUBNET}{i}"
        return ip if tcp_probe(ip, 80, timeout) else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [r for r in ex.map(one, range(1, 255)) if r]


def ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=25)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except subprocess.TimeoutExpired:
        return "(timeout)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=f"{SUBNET}50")
    a = ap.parse_args()

    print("== hotspot adapter ==")
    print(ps("Get-NetIPAddress -AddressFamily IPv4 | "
             "Where-Object {$_.IPAddress -like '192.168.137.*'} | "
             "Select-Object -ExpandProperty IPAddress"))

    print("\n== sharing services ==")
    for svc in ("SharedAccess", "icssvc", "WlanSvc"):
        out = ps(f"(Get-Service -Name {svc} -ErrorAction SilentlyContinue)."
                 f"Status -join ''")
        print(f"  {svc:12s} {out or '(missing)'}")

    print("\n== mobile hotspot ==")
    print(ps("try { $ns = [Windows.Networking.NetworkOperators."
             "NetworkOperatorTetheringManager,Windows.Networking.NetworkOperators,"
             "ContentType=WindowsRuntime]; 'winrt-ok' } catch { 'winrt-unavailable' }"))

    print(f"\n== tcp/80 sweep of {SUBNET}0/24 ==")
    t0 = time.perf_counter()
    live = sweep()
    print(f"  hosts answering on :80 -> {live if live else '(none)'} "
          f"({time.perf_counter() - t0:.1f}s)")

    print(f"\n== direct probe of {a.ip} ==")
    print(f"  tcp/80  {'OPEN' if tcp_probe(a.ip, 80, 1.5) else 'closed'}")
    print(f"  tcp/9001 {'OPEN' if tcp_probe(a.ip, 9001, 0.8) else 'closed'}")
    return 0 if live else 1


if __name__ == "__main__":
    sys.exit(main())
