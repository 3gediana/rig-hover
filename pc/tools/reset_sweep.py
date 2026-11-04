# -*- coding: utf-8 -*-
"""reset_sweep.py -- find a reset polarity that makes the board talk.

A reset was pulsed as (RTS asserted, DTR clear) and produced nothing, so either
that is not how EN is wired here, or the chip never sees the pulse. Try every
DTR/RTS combination, hold each for a moment with IO0 high, and report which of
them yields any bytes. Each attempt leaves the board in the normal boot state.

Nothing is written to the flash. If no combination produces output, the UART RX
path itself is not carrying the console.

Run:  python reset_sweep.py --port COM3 --baud 115200
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time

import serial


def attempt(port: str, baud: int, dtr: bool, rts: bool,
            listen: float = 2.5) -> tuple[int, bytes]:
    sp = serial.Serial()
    sp.port = port
    sp.baudrate = baud
    sp.timeout = 0.2
    sp.dtr = False
    sp.rts = False
    sp.open()
    # hold the pair, then release both to the run state
    sp.dtr = dtr
    sp.rts = rts
    time.sleep(0.12)
    sp.dtr = False
    sp.rts = False
    t_end = time.time() + listen
    buf = bytearray()
    while time.time() < t_end:
        buf.extend(sp.read(4096))
    sp.close()
    return len(buf), bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=115200)
    a = ap.parse_args()

    combos = list(itertools.product((False, True), (False, True)))
    hits = []
    for dtr, rts in combos:
        n, data = attempt(a.port, a.baud, dtr, rts)
        print(f"  dtr={int(dtr)} rts={int(rts)} -> {n} bytes", flush=True)
        if n:
            hits.append((dtr, rts, data))
        time.sleep(0.4)

    if not hits:
        print("\nno combination produced output at this baud")
        # the console could be at a different rate; the ROM always uses 115200,
        # so a mismatch here points at the wiring, not the firmware
        return 2

    dtr, rts, data = hits[0]
    print(f"\nfirst producing combo dtr={int(dtr)} rts={int(rts)}:")
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.rstrip()
        if line:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
