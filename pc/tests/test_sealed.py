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


def _geonet_trainer():
    endside = ROOT / "pc" / "end_side"
    if str(endside) not in sys.path:
        sys.path.insert(0, str(endside))
    import train_hover_geonet_240 as mod  # noqa: PLC0415
    return mod


def test_geonet_dev_eval_split_is_dev_only_and_named():
    mod = _geonet_trainer()
    spec = mod.HELDOUT_SPLIT_SPECS["dev_geometry_v1"]
    assert not spec["sealed"]
    assert "dev_geometry_v1" in spec["cache_file"]
    assert all(not sealed.is_sealed_path(src[1]) for src in spec["sources"])
    assert not sealed.is_sealed_path(mod._heldout_cache_path("dev_geometry_v1"))


def test_geonet_sealed_final_split_is_guarded_and_named():
    mod = _geonet_trainer()
    spec = mod.HELDOUT_SPLIT_SPECS["sealed_final_v1"]
    assert spec["sealed"]
    assert "sealed_final_v1" in spec["cache_file"]
    assert sealed.is_sealed_path(mod._heldout_cache_path("sealed_final_v1"))
    with pytest.raises(sealed.SealedAccessError):
        mod.load_heldout_split("sealed_final_v1", final_eval=False)


def test_geonet_validation_and_dev_eval_video_windows_are_disjoint():
    mod = _geonet_trainer()
    take = ROOT / "dataset" / "takes" / "robot_20261115_150045_65a54d" / "take.mp4"
    assert take.exists()
    val = mod._sample_video_pairs(take, 8, start_frac=mod.VAL_TAKE_WINDOW[0], end_frac=mod.VAL_TAKE_WINDOW[1])
    dev = mod._sample_video_pairs(take, 6, start_frac=mod.DEV_TAKE_WINDOW[0], end_frac=mod.DEV_TAKE_WINDOW[1])
    val_idx = [int(x[2].rsplit(":", 1)[1]) for x in val]
    dev_idx = [int(x[2].rsplit(":", 1)[1]) for x in dev]
    assert val_idx and dev_idx
    assert max(val_idx) < min(dev_idx)


def test_geonet_trainer_drops_legacy_cache_and_writes_measured_camera_dir():
    mod = _geonet_trainer()
    src = (ROOT / "pc" / "end_side" / "train_hover_geonet_240.py").read_text(encoding="utf-8")
    assert "test_heldout_80.npz" not in src
    assert "out_dir = OUT_DIR" in src
    assert 'score = dev_m["abs_rel"]' in src
    assert '"selection_split": "dev_geometry_v1"' in src
    assert mod.OUT_DIR.name == "hover_geonet_240_h135_f75.5"


def test_legacy_geonet_heldout_cache_is_not_active():
    cache_root = ROOT / "pc" / "bench" / "_teacher_cache"
    active = list(cache_root.glob("geonet240_h*/test_heldout_80.npz"))
    assert active == []
