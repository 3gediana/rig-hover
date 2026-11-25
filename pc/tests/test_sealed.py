# -*- coding: utf-8 -*-
"""Tests for the sealed-data doorway and pre-registered split manifests."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pc" / "pack"))

import paths  # noqa: E402
import sealed  # noqa: E402

PY = sys.executable


def test_sealed_classifier_covers_all_three_sources():
    assert sealed.is_sealed_path("dataset/takes/robot_20261115_151733_cb8455/take.mp4")
    assert sealed.is_sealed_path("evidence/live/fov_sweep/frames/x.jpg")
    assert sealed.is_sealed_path("evidence/sessions/empty_d0.5_front_night_182201/frames/x.jpg")
    assert not sealed.is_sealed_path("dataset/takes/robot_20261115_150045_65a54d/take.mp4")
    assert not sealed.is_sealed_path("evidence/sessions/knock_d0.5_front_night_182129/frames/x.jpg")


def test_require_refuses_without_final_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PACK_ROOT", tmp_path)
    monkeypatch.delenv("ROBOT_PACK_FINAL_EVAL", raising=False)
    with pytest.raises(sealed.SealedAccessError):
        sealed.require_final_eval(["dataset/takes/robot_20261115_151733_cb8455/take.mp4"], "unit test", argv=[])
    assert not (tmp_path / "sealed_access.log").exists()


def test_final_eval_logs_exactly_the_sealed_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "PACK_ROOT", tmp_path)
    got = sealed.require_final_eval(
        [
            "dataset/takes/robot_20261115_151733_cb8455/take.mp4",
            "dataset/takes/robot_20261115_150045_65a54d/take.mp4",
        ],
        "unit test final evaluation",
        argv=["--final-eval"],
    )
    assert got == ["dataset/takes/robot_20261115_151733_cb8455/take.mp4"]
    rows = [json.loads(x) for x in (tmp_path / "sealed_access.log").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["purpose"] == "unit test final evaluation"
    assert rows[0]["paths"] == got


def test_cli_tripwire_returns_2_without_final_eval():
    p = subprocess.run(
        [PY, str(ROOT / "pc" / "pack" / "sealed.py"), "--require", "dataset/takes/robot_20261115_151733_cb8455/take.mp4"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
    )
    assert p.returncode == 2
    assert "sealed data access refused" in p.stderr


def test_training_entry_points_have_sealed_tripwires():
    expected = {
        ROOT / "pc" / "end_side" / "train_hover_geonet_240.py": "train_hover_geonet_240 final held-out evaluation",
        ROOT / "pc" / "end_side" / "train_dense_splat_student.py": "train_dense_splat_student final held-out evaluation",
        ROOT / "pc" / "end_side" / "train_mono_depth.py": "train_mono_depth final held-out evaluation",
        ROOT / "pc" / "train" / "double_knock_data.py": "double-knock dataset final held-out evaluation",
    }
    for path, marker in expected.items():
        src = path.read_text(encoding="utf-8")
        assert "sealed.require_final_eval" in src, path
        assert marker in src, path


def test_split_manifests_exist_and_never_mix_units():
    names = ["take_splits.json", "evidence_splits.json", "live_session_splits.json", "preregistered_rules.json"]
    for name in names:
        assert (paths.SPLITS / name).exists(), name
    takes = json.loads((paths.SPLITS / "take_splits.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r["split"] for r in takes["rows"]}
    assert by_id["robot_20261115_151733_cb8455"] == "sealed"
    assert by_id["robot_20261115_150045_65a54d"] == "dev"
    assert len(by_id) == 6
    ev = json.loads((paths.SPLITS / "evidence_splits.json").read_text(encoding="utf-8"))
    ev_by_id = {r["id"]: r["split"] for r in ev["rows"]}
    assert ev_by_id["empty_d0.5_front_night_182201"] == "sealed"
    assert ev_by_id["fov_sweep"] == "sealed"
    for row in takes["rows"] + ev["rows"]:
        assert row["n_files"] == len(row["files"])
        assert row["bytes"] == sum(f["bytes"] for f in row["files"])
        assert all(len(f["sha256"]) == 64 for f in row["files"])


def test_materialized_mirror_manifest_matches_current_tree():
    ok, problems = sealed.verify_sealed()
    assert ok, problems
