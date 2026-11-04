# -*- coding: utf-8 -*-
"""reboot_and_capture.py -- one reset, then read the console.

The firmware logs at WARN level (CONFIG_LOG_DEFAULT_LEVEL=2), so a running robot
is silent: the only messages it emits are at boot. The WARN-level lines that do
exist at boot are `dbg_boot: internal_free=...` and, once the station associates,
`WiFi IP: <addr>` -- so a single reset answers both "is the chip running" and
"did it get on the network".

Reset uses the CH343 control lines through the usual two-transistor ESP auto-reset
circuit: EN is pulled low when RTS is asserted and DTR is not, and released when
RTS drops. IO0 is left high, so the chip boots into the application, not the ROM
bootloader. Nothing is written to the board.

Side effect: the robot reboots once. On this hardware a boot briefly drives the
motors, which is audible. Nothing else changes.

Run:  python reboot_and_capture.py --port COM3 --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--no-reset", action="store_true",
                    help="only listen, do not pulse the reset line")
    a = ap.parse_args()

    sp = serial.Serial()
    sp.port = a.port
    sp.baudrate = a.baud
    sp.timeout = 0.3
    sp.dtr = False
    sp.rts = False
    sp.open()

    if not a.no_reset:
        # EN low via RTS, then release. IO0 stays high -> normal boot.
        sp.rts = True
        sp.dtr = False
        time.sleep(0.15)
        sp.rts = False
        print(f"pulsed reset on {a.port}; capturing {a.seconds:.0f}s ...",
              flush=True)
    else:
        print(f"listening {a.seconds:.0f}s without reset ...", flush=True)

    t_end = time.time() + a.seconds
    buf = bytearray()
    last = time.time()
    while time.time() < t_end:
        chunk = sp.read(4096)
        if chunk:
            buf.extend(chunk)
            last = time.time()
        elif time.time() - last > 8.0 and buf:
            # keep going: WiFi association can take 20 s with no output
            pass
    sp.close()

    print(f"bytes received: {len(buf)}")
    if not buf:
        print("NO OUTPUT. Either the board is not powered, or the reset line is")
        print("not wired the way the two-transistor circuit assumes.")
        return 2
    for line in buf.decode("utf-8", "replace").splitlines():
        line = line.rstrip()
        if line:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
