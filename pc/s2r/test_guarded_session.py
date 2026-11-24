# -*- coding: utf-8 -*-
"""s2r/test_guarded_session.py - prove the guard before it is trusted on the table.

The guard is the last line between a controller bug and a robot on the floor,
so each limit is shown to trip on a replayed stream, and the real standing
log is shown to stay silent. The fixture is 4 s of real telemetry (100 Hz)
recorded standing on the desk on 2026-11-24; it contains angles and torques
only.

Run:  python -m pytest test_guarded_session.py -q
"""
import csv
import math
import pathlib
import threading
import time

import numpy as np
import pytest

import guarded_session as gs
from tellog import FIELDS, RawTel, TelCsv, read_tel_csv

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "testdata" / "rest_tlm_20261124.csv"


def load_fixture():
    cols = read_tel_csv(str(FIXTURE))
    n = len(cols["recv_s"])
    return [(1.0 + cols["recv_s"][i], cols["seq"][i], cols["uptime_ms"][i],
             [cols[k][i] for k in FIELDS]) for i in range(n)]


def make_guard(tmp_path, **lim):
    calls = []
    g = gs.Guard(lambda why: calls.append(why), tmp_path / "ABORT", **lim)
    return g, calls


def arm_on(g, pkts, n=200):
    f = np.array([p[3] for p in pkts[:n]], dtype=float)
    g.arm(float(np.median(f[:, 1])), float(np.median(f[:, 0])), pkts[0][3])


def replay(g, pkts):
    """Feed packets until the guard trips or the stream ends; return the trip reason."""
    for p in pkts:
        g(*p)
        if g.tripped:
            break
    deadline = time.perf_counter() + 2.0
    while not g.tripped and time.perf_counter() < deadline:
        time.sleep(0.002)            # a late trip means a limit did not fire at all
    return g.tripped


def wait_calls(calls, want=True):
    for _ in range(200):
        if bool(calls) == want:
            return
        time.sleep(0.005)


def test_guard_is_silent_on_the_real_standing_log(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    assert replay(g, pkts) == ""
    wait_calls(calls, want=False)
    assert calls == []
    assert g.max_pos < 0.001 and g.max_pitch_dev < 1.0


def test_guard_trips_on_pitch(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    for p in pkts[250:]:
        p[3][1] += 7.0
    assert "pitch" in replay(g, pkts)
    wait_calls(calls)
    assert len(calls) == 1


def test_guard_trips_on_roll_absolute_and_deviation(tmp_path):
    for add, needle in ((13.0 - (-5.6), "abs"), (6.0, "vs rest")):
        pkts = load_fixture()
        g, calls = make_guard(tmp_path)
        arm_on(g, pkts)
        for p in pkts[250:]:
            p[3][0] += add
        assert needle in replay(g, pkts)


def test_guard_trips_on_position_box_before_raw_wheel_box(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts, n=50)                     # 3.5 s of stream left after arming
    x0 = pkts[50][3][12]
    for i, p in enumerate(pkts[50:], start=0):
        p[3][12] = x0 + 0.03 * 0.01 * i        # 3 cm/s (100 Hz rows): below the speed limit, crosses 7 cm at 2.3 s
    assert "position box" in replay(g, pkts)


def test_guard_trips_on_raw_wheel_travel(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path, pos_box=10.0)
    arm_on(g, pkts, n=50)
    x0 = pkts[50][3][12]
    for i, p in enumerate(pkts[50:], start=0):
        p[3][12] = x0 + 0.05 * 0.01 * i        # 5 cm/s: crosses the 9 cm raw box at 1.8 s
    assert "wheel_x travel" in replay(g, pkts)


def test_guard_trips_on_heading_excursion(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts, n=50)
    w10 = pkts[50][3][10]
    for i, p in enumerate(pkts[50:], start=0):
        p[3][10] = w10 + 0.5 * i               # encoder ticks: 0.219 deg per tick -> 11 deg/s, 38 deg after 3.5 s
    assert "heading excursion" in replay(g, pkts)


def test_guard_trips_on_speed(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path, pos_box=10.0, wheel_box=10.0)
    arm_on(g, pkts)
    x0 = pkts[250][3][12]
    for i, p in enumerate(pkts[250:], start=0):
        p[3][12] = x0 + 0.30 * 0.01 * i        # 30 cm/s
    assert "wheel speed" in replay(g, pkts)


def test_guard_trips_on_yaw_rate(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path, psi_box=360.0)
    arm_on(g, pkts)
    w10 = pkts[250][3][10]
    for i, p in enumerate(pkts[250:], start=0):
        p[3][10] = w10 + 4.0 * i               # 4 ticks per 10 ms = 87.6 deg/s
    assert "yaw rate" in replay(g, pkts)


def test_guard_trips_when_not_balancing(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    for p in pkts[250:]:
        p[3][22] = 0.0
    assert "robot_state=0" in replay(g, pkts)


def test_guard_is_deaf_before_armed(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    for p in pkts:
        p[3][1] += 30.0
    assert replay(g, pkts) == ""
    wait_calls(calls, want=False)
    assert calls == []


def test_watchdog_trips_on_stale_telemetry_and_on_abort_file(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    replay(g, pkts)
    g.watchdog(g.last_pkt_t + 1.0)
    assert g.tripped == "telemetry stale"
    wait_calls(calls)

    g2, calls2 = make_guard(tmp_path)
    arm_on(g2, pkts)
    replay(g2, pkts)
    (tmp_path / "ABORT").touch()
    g2.watchdog(g2.last_pkt_t + 0.01)
    assert g2.tripped == "ABORT file"
    wait_calls(calls2)


def test_trip_fires_exactly_once(tmp_path):
    pkts = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    for p in pkts[250:]:
        p[3][1] += 40.0
    replay(g, pkts)
    g._trip("again", 0.0)
    wait_calls(calls)
    assert len(calls) == 1 and calls[0].startswith("pitch ")


def test_real_head_start_spike_does_not_trip_the_debounced_guard(tmp_path):
    """The telemetry of the false trip on 2026-11-24 23:46 (head servo start, roll jumped
    -4.9 -> +6.3 deg for exactly two samples). The debounced guard must stay silent on it."""
    cols = read_tel_csv(str(HERE / "testdata" / "head_start_spike_tlm_20261124.csv"))
    pkts = [(1.0 + cols["recv_s"][i], cols["seq"][i], cols["uptime_ms"][i], [cols[k][i] for k in FIELDS])
            for i in range(len(cols["recv_s"]))]
    assert max(p[3][0] for p in pkts) > 6.0 and min(p[3][0] for p in pkts) < -4.0
    g, calls = make_guard(tmp_path)
    arm_on(g, pkts)
    assert replay(g, pkts) == ""
    wait_calls(calls, want=False)
    assert calls == [] and g.angle_spikes["roll"] >= 1 and g.angle_spikes["roll"] <= 4   # up, down: a few samples


def test_angle_limits_must_persist(tmp_path):
    pkts = load_fixture()
    spike = load_fixture()
    g, calls = make_guard(tmp_path)
    arm_on(g, spike)
    for p in spike[250:253]:                 # 3 rows = 30 ms at 100 Hz
        p[3][0] += 8.0
    assert replay(g, spike) == ""
    for p in pkts[250:]:                      # sustained
        p[3][0] += 8.0
    assert "roll" in replay(g, pkts)
    g1, calls1 = make_guard(tmp_path, debounce_angle=1)   # without debounce the same spike fires
    arm_on(g1, spike)
    assert "roll" in replay(g1, spike)


def test_firmware_lqr_identity_on_real_rest_log():
    """temp_u read back from the two wheel torques equals the firmware's own LQR expression.

    tor1 = -temp_u + yaw_u, tor2 = +temp_u + yaw_u (robot.cc), so the common-mode
    torque is (tor2 - tor1)/2. On the standing log it must reproduce
    -(K0*x_e + K1*(v - vx) + K2*(pitch - imu_zero) + K3*dq) with imu_zero = -2 deg,
    which is what makes the rest offset predictable: x_e = -K2/K0 * (pitch - imu_zero).
    """
    c = read_tel_csv(str(FIXTURE))
    tor1, tor2 = np.array(c["tor1"]), np.array(c["tor2"])
    temp_u = 0.5 * (tor2 - tor1)
    yaw_u = 0.5 * (tor1 + tor2)
    x_e = np.array(c["wheel_x"]) - np.array(c["stable_pos"])
    v = np.array(c["wheel_vx"]) - np.array(c["vx"])
    dq = np.array(c["dq"])
    u_pred = -(1300.0 * x_e + 5.0 * v + 25.0 * (np.array(c["pitch"]) - (-2.0)) + 0.4 * dq)
    assert np.max(np.abs(u_pred)) < 20.0            # outside the +-10 friction bump zone
    assert float(np.sqrt(np.mean((temp_u - u_pred) ** 2))) < 0.6
    assert float(np.abs(yaw_u).max()) < 0.5          # no yaw torque while standing still
    # the steady-state offset line: K0*x_e + K2*(pitch - imu_zero) + temp_u ~= 0
    resid = 1300.0 * x_e + 25.0 * (np.array(c["pitch"]) + 2.0) + temp_u
    assert float(np.abs(resid).max()) < 0.5


def test_rest_offset_slope_is_k2_over_k0():
    slope_m_per_deg = -gs.K2 / gs.K0
    assert math.isclose(slope_m_per_deg * 100, -1.9231, abs_tol=1e-3)


def test_telcsv_roundtrip(tmp_path):
    p = tmp_path / "t.csv"
    w = TelCsv(str(p))
    f = [float(i) for i in range(34)]
    w(0.0, 0, 1000, f)
    w(0.002, 1, 1002, f)
    w.close()
    c = read_tel_csv(str(p))
    assert list(c.keys())[:3] == ["recv_s", "seq", "uptime_ms"]
    assert c.keys() - {"recv_s", "seq", "uptime_ms"} == set(FIELDS)
    assert c["seq"] == [0, 1] and c["uptime_ms"] == [1000, 1002]
    assert c["roll"] == [0.0, 0.0] and c["danger"] == [33.0, 33.0]


def test_rawtel_receives_tlm1_over_a_local_socket():
    """Real receive loop against a local UDP fake of the firmware stream."""
    import socket
    import struct

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.bind(("127.0.0.1", 0))
    srv.settimeout(3.0)
    port = srv.getsockname()[1]
    got = []
    rt = RawTel("127.0.0.1", port=port, on_pkt=lambda a, b, c, f: got.append((b, c, f)))
    try:
        hi, peer = srv.recvfrom(64)          # the subscription ping tells us where to stream
        assert hi == b"hi"
        for seq in range(3):
            f = [float(i + seq) for i in range(34)]
            f[22] = 1.0                      # robot_state
            pkt = struct.pack("<IHHII", 0x314D4C54, 1, 152, seq, 1000 + 2 * seq) + struct.pack("<34f", *f)
            srv.sendto(pkt, peer)
        for _ in range(300):
            if len(got) >= 3:
                break
            time.sleep(0.01)
        assert [(b, c) for b, c, _ in got] == [(0, 1000), (1, 1002), (2, 1004)]
        assert got[0][2][22] == 1.0 and got[2][2][33] == 35.0
        t = rt.latest(max_age=10.0)
        assert t is not None and t.robot_state == 1 and abs(t.x_m - 14.0) < 1e-6   # f[12] = 12 + seq
    finally:
        rt.stop = True
        srv.close()
