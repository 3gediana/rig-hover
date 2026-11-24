# -*- coding: utf-8 -*-
"""s2r/test_session_on_mock.py - every guarded_session stage, end to end, with no robot.

Runs the real session code (telemetry socket, HTTP command path, guard, safe
stop, logging) against mock_robot.py. What this proves: the stage scripts
complete, always end in manual mode with a zero command, write their logs,
and that a guard trip raised through the real path really stops the robot.
What it does not prove: anything about the physical robot.

Run:  python -m pytest test_session_on_mock.py -q
"""
import json
import pathlib
import sys
import time

import pytest

import guarded_session as gs
from mock_robot import Mock
from tellog import FIELDS, read_tel_csv


@pytest.fixture(scope="module")
def mock():
    m = Mock("127.0.0.1", 0, 0)
    time.sleep(0.3)
    yield m
    m.close()


def run_stage(mock, tmp_path, monkeypatch, argv):
    out_root = tmp_path / "s2r_live"
    monkeypatch.setattr(gs, "OUT_ROOT", out_root)
    monkeypatch.setattr(sys, "argv", ["guarded_session.py", "--host", f"127.0.0.1:{mock.http_port}",
                                      "--tlm-port", str(mock.tlm_port)] + argv)
    n_ev = len(mock.plant.events)
    rc = gs.main()
    dirs = sorted(out_root.glob("session_*"))
    assert rc == 0 and len(dirs) == 1, "exactly one session directory expected"
    rep = json.loads((dirs[0] / "summary.json").read_text(encoding="utf-8"))
    return rep, dirs[0], mock.plant.events[n_ev:]


def assert_ended_safe(rep, d, events):
    assert rep.get("exception") is None, rep.get("exception")
    assert rep.get("final_safe_stop_ok") is True or rep.get("safe_stop_ok") is True
    kinds = [e[1] for e in events]
    assert "manual" in kinds and kinds[-1] == "manual", "last request must be the manual/stop one"
    tel = read_tel_csv(str(d / "tel.csv"))
    assert len(tel["seq"]) > 100 and tel["robot_state"][-1] == 1.0
    assert set(tel.keys()) >= set(FIELDS)


def test_rest_stage_never_actuates(mock, tmp_path, monkeypatch):
    rep, d, ev = run_stage(mock, tmp_path, monkeypatch, ["--stage", "rest", "--secs", "1.5"])
    assert "rest" in rep and rep["rest"]["packets"] > 100
    assert rep["rest"]["robot_state"] == [1]
    assert all(e[1] not in ("teacher", "manual", "cfg") for e in ev), "rest stage must send nothing that actuates"


def test_head_stage(mock, tmp_path, monkeypatch):
    rep, d, ev = run_stage(mock, tmp_path, monkeypatch,
                           ["--stage", "head", "--hold", "0.5", "--head-targets", "10", "-10"])
    assert rep.get("abort") == ""
    frames = list((d / "frames").glob("hold_*.jpg"))
    assert len(frames) == 2
    heads = [e[2][0] for e in ev if e[1] == "teacher"]
    assert max(heads) == pytest.approx(10.0) and min(heads) == pytest.approx(-10.0)
    assert_ended_safe(rep, d, ev)


def test_heading_stage_runs_the_six_steps_and_stops(mock, tmp_path, monkeypatch):
    rep, d, ev = run_stage(mock, tmp_path, monkeypatch, ["--stage", "heading", "--time-scale", "0.1"])
    assert rep.get("abort") == ""
    assert [r["target"] for r in rep["heading"]] == [8, 0, -8, 0, 15, 0]
    assert all(r["timed_out"] is False if "timed_out" in r else True for r in rep["heading"])
    assert all(e[2][1] == 0.0 for e in ev if e[1] == "teacher"), "heading stage never commands vx"
    assert_ended_safe(rep, d, ev)


def test_ramp_stage_returns_imu_zero_and_stops(mock, tmp_path, monkeypatch):
    rep, d, ev = run_stage(mock, tmp_path, monkeypatch,
                           ["--stage", "ramp", "--imu-zero-to", "0.0", "--ramp-rate", "4", "--hold", "0.4"])
    assert rep.get("abort") == "" and rep["ramp"]["completed"] is True
    assert rep["imu_zero_start"] == pytest.approx(-2.0) and rep["imu_zero_end"] == pytest.approx(-2.0)
    sets = [e[2] for e in ev if e[1] == "api_set"]
    assert sets and max(v for _, v in sets) == pytest.approx(0.0, abs=1e-3) and sets[-1][1] == pytest.approx(-2.0)
    assert_ended_safe(rep, d, ev)


def test_drive_stage_goes_and_comes_back(mock, tmp_path, monkeypatch):
    rep, d, ev = run_stage(mock, tmp_path, monkeypatch,
                           ["--stage", "drive", "--dist", "0.02", "--vmax", "0.03", "--back", "--timeout", "8"])
    assert rep.get("abort") == ""
    assert [r["goal_cm"] for r in rep["drive"]] == [2.0, 0.0]
    assert all(r["timed_out"] is False for r in rep["drive"])
    assert rep["drive"][0]["err_cm"] < 0.6 and rep["drive"][1]["err_cm"] < 0.6
    assert_ended_safe(rep, d, ev)


def test_guard_trip_through_the_real_path_stops_the_robot(mock, tmp_path, monkeypatch):
    # the mock starts leaning 8 deg once it has received 40 teacher commands
    mock.plant.arm_fault(40, 8.0)
    try:
        rep, d, ev = run_stage(mock, tmp_path, monkeypatch, ["--stage", "drive", "--dist", "0.02", "--timeout", "8"])
    finally:
        mock.plant.fault_pitch_step_after = None
        mock.plant.pitch_rest = 1.0
    assert rep["abort"].startswith("pitch"), rep
    assert rep.get("safe_stop_ok") is True
    kinds = [e[1] for e in ev]
    assert "fault" in kinds and kinds[-1] == "manual"
    last_manual = len(kinds) - 1 - kinds[::-1].index("manual")
    assert "teacher" not in kinds[last_manual + 1:], "no teacher command may follow the safe stop"
    assert rep["guard"]["tripped"].startswith("pitch")
    assert rep.get("safe_stop_unverified") is None
    zeros = [e[2] for e in ev if e[1] == "teacher"][-3:]
    assert all(v == (0.0, 0.0, 0.0) for v in zeros), "last teacher requests carry zeros"
