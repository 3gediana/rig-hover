"""Sweep GET /capture?q=NN and report JPEG latency, size and encoder split.

The robot logs its own readout / convert / encode breakdown on the UART, so
this tool reads the serial console in a background thread while it drives the
HTTP requests. That is the only way to tell whether the remaining frame budget
sits in the camera readout or in the software JPEG encoder.

Usage:
    python tools/quality_sweep.py --base http://192.168.137.50
    python tools/quality_sweep.py --serial COM3 --baud 921600 --n 20
    python tools/quality_sweep.py --no-serial
"""

from __future__ import annotations

import argparse
import http.client
import re
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

LOG_RE = re.compile(
    r"cap q=(?P<q>\d+) readout=(?P<readout>\d+) conv=(?P<conv>\d+) "
    r"encode=(?P<encode>\d+) us out=(?P<out>\d+)"
)


class SerialTap:
    """Background reader that keeps every `cap q=...` line the robot emits."""

    def __init__(self, port: str, baud: int) -> None:
        self.lines: list[str] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._port = port
        self._baud = baud

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import serial  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment issue
            self.error = f"pyserial missing: {exc}"
            return
        try:
            # Opening COM3 on this board asserts RTS -> EN and reboots the
            # ESP32, so both lines are preset low before open(). A context
            # manager would open first and defeat the point.
            port = serial.Serial()
            port.port = self._port
            port.baudrate = self._baud
            port.timeout = 0.2
            port.dtr = False
            port.rts = False
            port.open()
            try:
                while not self._stop.is_set():
                    chunk = port.read(4096)
                    if chunk:
                        self.lines.append(chunk.decode("utf-8", "replace"))
            finally:
                port.close()
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self.error = f"serial {self._port}: {exc}"

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def samples(self) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for match in LOG_RE.finditer("".join(self.lines)):
            out.append({k: int(v) for k, v in match.groupdict().items()})
        return out


def connect(host) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(host.hostname, host.port or 80, timeout=10)


def fetch(conn: http.client.HTTPConnection, path: str) -> tuple[float, int]:
    """One request; returns (elapsed_ms, body_bytes)."""
    start = time.perf_counter()
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    elapsed = (time.perf_counter() - start) * 1000.0
    if resp.status != 200:
        raise RuntimeError(f"{path} -> HTTP {resp.status}")
    return elapsed, len(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://192.168.137.50")
    ap.add_argument("--serial", default="COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--no-serial", action="store_true")
    ap.add_argument("--qualities", default="80,70,60,50,40,30,20")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    host = urlparse(args.base)
    conn = connect(host)

    # Wait for the board: the static-IP fallback fires at ~12 s after boot and
    # the httpd only starts after that. A failed attempt leaves the connection
    # unusable, so a fresh one is made every time.
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            fetch(conn, "/status")
            break
        except Exception:  # noqa: BLE001 - boot race, retried below
            conn.close()
            conn = connect(host)
            time.sleep(1.0)
    else:
        print("robot never answered /status", file=sys.stderr)
        return 1
    print(f"robot up: {args.base}")

    tap: SerialTap | None = None
    if not args.no_serial:
        tap = SerialTap(args.serial, args.baud)
        tap.start()
        time.sleep(0.3)
        if tap.error:
            print(f"[warn] {tap.error}")
            tap = None

    for _ in range(args.warmup):
        fetch(conn, "/capture")

    qualities = [int(q) for q in args.qualities.split(",") if q.strip()]
    rows: list[tuple[int, float, float, float, float]] = []
    print(f"\n{'q':>4} {'p50_ms':>8} {'p90_ms':>8} {'bytes':>8} {'ratio':>7}")
    print("-" * 42)
    for q in qualities:
        times: list[float] = []
        sizes: list[int] = []
        for _ in range(args.n):
            elapsed, size = fetch(conn, f"/capture?q={q}")
            times.append(elapsed)
            sizes.append(size)
        p50 = statistics.median(times)
        p90 = sorted(times)[min(len(times) - 1, int(len(times) * 0.9))]
        size = statistics.median(sizes)
        rows.append((q, p50, p90, size, 0.0))
        print(f"{q:>4} {p50:>8.1f} {p90:>8.1f} {size:>8.0f}")

    baseline = rows[0][3] or 1
    print("\n" + "=" * 42)
    a = rows[0]
    for q, p50, p90, size, _ in rows:
        saved = a[1] - p50
        print(
            f"q={q:<3} p50={p50:6.1f} ms  ({saved:+5.1f} vs q={a[0]})  "
            f"{size:5.0f} B ({100.0 * size / baseline:5.1f}% size)"
        )

    if tap is not None:
        tap.stop()
        samples = tap.samples()
        print("\nrobot-side split (from UART), grouped by q:")
        by_q: dict[int, list[dict[str, int]]] = {}
        for s in samples:
            by_q.setdefault(s["q"], []).append(s)
        if not by_q:
            print("  no 'cap q=' lines seen -- wrong --baud, or logging sampled out")
        for q in sorted(by_q):
            group = by_q[q]
            readout = statistics.median([s["readout"] for s in group]) / 1000.0
            conv = statistics.median([s["conv"] for s in group]) / 1000.0
            encode = statistics.median([s["encode"] for s in group]) / 1000.0
            print(
                f"  q={q:<3} n={len(group):<3} readout={readout:5.1f} ms  "
                f"conv={conv:5.1f} ms  encode={encode:5.1f} ms  "
                f"sum={readout + conv + encode:5.1f} ms"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
