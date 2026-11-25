# -*- coding: utf-8 -*-
"""tests for pc/robot_truth.py and pc/pack/const_scan.py.

The truth file is what every other constant is checked against, so it is
validated itself: every fact says how, when and where it was measured, the
derived quantities follow from the measured ones, and the numbers the s2r
runtime was built on (2.705 px/deg, 0.135 m) come out of the file unchanged.

Run from the repository root:
    python -m pytest pc/tests/test_robot_truth.py -q
"""
import math
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pc"))
sys.path.insert(0, str(ROOT / "pc" / "pack"))

import robot_truth as rt  # noqa: E402


def test_every_fact_is_fully_attributed():
    for name, f in rt.DATA["facts"].items():
        for key in rt.REQUIRED:
            measured = f["level"] != "untested"
            assert f.get(key) not in (None, "") or (key == "value" and not measured), f"{name}: missing {key}"
        assert f["level"] in rt.LEVELS, name
        # a fact nobody has measured is recorded as such (value null) and may not claim a level
        assert f["value"] is not None or f["level"] == "untested", name


def test_an_unmeasured_fact_cannot_claim_a_level(tmp_path):
    bad = {"facts": {"pitch": {"value": None, "method": "not calibrated", "date": "2026-11-23", "source": "-", "confidence": "none",
                               "level": "T2"}}}
    p = tmp_path / "t.json"
    p.write_text(__import__("json").dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        rt.load(p)


def test_unknown_fact_names_the_available_ones():
    with pytest.raises(KeyError) as e:
        rt.fact("camera.fov_x_deg")
    assert "camera.fov_h_deg" in str(e.value)


def test_derived_values_follow_from_the_measured_field_of_view():
    assert rt.FOV_H_DEG == pytest.approx(75.5)
    assert rt.F_PX == pytest.approx(154.99, abs=0.05)
    assert rt.PX_PER_DEG == pytest.approx(2.705, abs=0.002)
    assert rt.BLIND_ZONE_M == pytest.approx(0.174, abs=0.002)
    assert rt.TICK_M == pytest.approx(math.pi * 0.06 / 1024)
    assert rt.CAM_HEIGHT_M == pytest.approx(0.135)
    assert rt.REST_OFFSET_M_PER_DEG * 100 == pytest.approx(-1.923, abs=0.002)
    assert rt.FRICTION_LEAD_M * 100 == pytest.approx(2.31, abs=0.05)


def test_deprecated_values_are_listed_as_superseded():
    assert 60 in rt.fact_meta("camera.fov_h_deg")["supersedes"]
    assert 62.5 in rt.fact_meta("camera.fov_h_deg")["supersedes"]
    assert 0.115 in rt.fact_meta("camera.height_m")["supersedes"]


def test_hand_size_correction_is_consistent():
    # a 9 cm hand at 1 m: f_px * 0.09 / 1.0
    assert rt.fact("camera.hand_9cm_px_per_m") == pytest.approx(rt.F_PX * 0.09, abs=0.5)


def test_s2r_runtime_uses_the_truth_file():
    src = (ROOT / "pc" / "s2r" / "link.py").read_text(encoding="utf-8")
    assert "robot_truth" in src and "F_PX_PER_DEG = 2.705" not in src
    src2 = (ROOT / "pc" / "s2r" / "controllers.py").read_text(encoding="utf-8")
    assert "robot_truth" in src2 and "F_PX_PER_DEG = 2.705" not in src2


def test_s2r_runtime_constants_equal_the_truth():
    """The names the runtime was built on still exist, and they are the truth file's numbers."""
    sys.path.insert(0, str(ROOT / "pc" / "s2r"))
    import controllers as C  # noqa: E402
    import link as L  # noqa: E402

    assert L.F_PX_PER_DEG == rt.PX_PER_DEG == C.F_PX_PER_DEG
    assert L.IMG_W == rt.IMG_W
    assert L.TICK_M == rt.TICK_M
    assert L.DEG_PER_WDIFF_M == -rt.DEG_PER_WDIFF_M                       # link's own sign convention: body_yaw_wheel = -k * wdiff
    assert L.DEG_PER_WDIFF_M == pytest.approx(1190.0, rel=1e-3)            # the value the runtime was built on (0.219 deg per tick)
    assert L.HEAD_DEADBAND_DEG == rt.HEAD_DEADBAND_DEG
    assert L.H_CAM_M == rt.CAM_HEIGHT_M
    assert C.pixel_to_table.__defaults__[:2] == (rt.CAM_HEIGHT_M, rt.F_PX)


def test_pixel_to_angle_is_the_measured_scale_at_the_centre_and_the_fov_at_the_edge():
    sys.path.insert(0, str(ROOT / "pc" / "s2r"))
    import controllers as C  # noqa: E402

    assert C.px_to_deg(2.705) == pytest.approx(1.0, abs=0.001)            # the measured centre scale
    assert C.px_to_deg(120.0) == pytest.approx(rt.FOV_H_DEG / 2, abs=0.01)  # the measured field of view
    for x in (-118.0, -60.0, -7.0, 0.0, 7.0, 60.0, 118.0):
        assert C.deg_to_px(C.px_to_deg(x)) == pytest.approx(x, abs=1e-9)
    # the linear scale that HeadServo used to extrapolate (240 px / 2.705 px per deg) is 17 deg too wide
    assert 240.0 / rt.PX_PER_DEG > rt.FOV_H_DEG + 10


# ---------------------------------------------------------------------------------------------
# Who may use which camera. Three kinds of camera exist in this repository and they must not be mixed:
#   robot_truth            the measured camera: real-robot chains, general models (MoGe-2), new training
#   the artifact's own     recorded in the released weights / manifests: whatever loads those weights
#   a synthetic camera     a renderer's own setting for a model trained on that renderer
# ---------------------------------------------------------------------------------------------
ENDSIDE = ROOT / "pc" / "end_side"
RUNS = ROOT / "runs_end_side"


def _import_end_side(name: str):
    for p in (ENDSIDE, ROOT / "pc"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return __import__(name)


def test_released_weights_record_the_camera_they_were_trained_with():
    """The constants the loaders fall back on equal what the released files actually say."""
    import torch

    for p in (ENDSIDE, ROOT / "pc"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from hd_moge_contour_engine import GEONET_CKPT_FOV_H_DEG, GEONET_CKPT_H_CAM_M
    from dense_splat_student import STUDENT_V3_FOV_H_DEG, STUDENT_V3_H_CAM_M
    from mono_depth_model import RELEASED_V1_FOV_H_DEG, RELEASED_V1_H_CAM_M

    geonet = sorted(RUNS.glob("hover_geonet_240/*_fp16.pt")) + sorted(RUNS.glob("hover_geonet_vit_240/*_fp16.pt"))
    if not geonet:
        pytest.skip("released GeoNet weights not on this machine")
    for ck in geonet:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        assert (blob.get("h_cam_m"), blob.get("fov_h_deg")) == (GEONET_CKPT_H_CAM_M, GEONET_CKPT_FOV_H_DEG), ck.name
    manifests = {
        RUNS / "mono_depth" / "hover_mono_depth_v1.json": (RELEASED_V1_H_CAM_M, RELEASED_V1_FOV_H_DEG),
        RUNS / "neural_3dgs" / "hover_gaussian_splat_v2.json": (RELEASED_V1_H_CAM_M, RELEASED_V1_FOV_H_DEG),
        RUNS / "dense_splat_v3" / "model.json": (STUDENT_V3_H_CAM_M, STUDENT_V3_FOV_H_DEG),
    }
    for path, (h, fov) in manifests.items():
        if not path.exists():
            pytest.skip(f"{path.name} not on this machine")
        cam = __import__("json").loads(path.read_text(encoding="utf-8"))["camera"]
        assert (cam["heightM"], cam["fovHDeg"]) == (h, fov), path.name
    # none of the released weights was trained with the measured camera - that gap is real and is recorded as such
    assert {GEONET_CKPT_FOV_H_DEG, RELEASED_V1_FOV_H_DEG, STUDENT_V3_FOV_H_DEG}.isdisjoint({rt.FOV_H_DEG})


def test_real_robot_chains_read_the_measured_camera():
    for p in (ENDSIDE, ROOT / "pc"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import real_robot_bridge
    import hd_moge_contour_engine as moge
    import hd_dynamic_contour_slam as slam
    import mono_depth_model
    import inspect

    assert real_robot_bridge.CALIB_FOV_H_DEG == rt.FOV_H_DEG and real_robot_bridge.CALIB_H_CAM_M == rt.CAM_HEIGHT_M
    assert moge.REAL_FOV_H_DEG == rt.FOV_H_DEG and moge.REAL_H_CAM_M == rt.CAM_HEIGHT_M
    assert mono_depth_model.DEFAULT_FOV_H_DEG == rt.FOV_H_DEG and mono_depth_model.DEFAULT_H_CAM_M == rt.CAM_HEIGHT_M
    # MoGe-2 is a general model that takes the camera as an input, so it gets the measured one
    moge_hint = inspect.signature(moge.MoGe2DesktopReconstructor.__init__).parameters["fov_x_deg"].default
    moge_fe = inspect.signature(slam.MoGe2Frontend.__init__).parameters
    assert moge_hint == rt.FOV_H_DEG
    assert (moge_fe["h_cam_m"].default, moge_fe["fov_h_deg"].default) == (rt.CAM_HEIGHT_M, rt.FOV_H_DEG)
    assert real_robot_bridge.CALIB_FOV_H_DEG != moge.GEONET_CKPT_FOV_H_DEG                       # the two cameras are distinct things


TRAINERS = ["train_hover_geonet_240.py", "train_hover_geonet_mv_selfsup_v4.py", "train_mono_depth.py",
            "train_neural_gaussian_splat.py", "train_dense_splat_student.py"]
ARTIFACT_CAMERAS = ("GEONET_CKPT_", "RELEASED_V1_", "STUDENT_V3_")


def test_new_training_starts_from_the_measured_camera():
    """No trainer may start from a released model's camera. A released camera may only be *compared against*
    (e.g. to pick the output folder); it must never be the camera a trainer renders or saves with."""
    artifact = {n for pre in ARTIFACT_CAMERAS for n in (pre + "FOV_H_DEG", pre + "H_CAM_M")}
    for name in TRAINERS:
        tree = __import__("ast").parse((ENDSIDE / name).read_text(encoding="utf-8"), filename=name)
        inside_compare = {id(n) for cmp_ in __import__("ast").walk(tree) if isinstance(cmp_, __import__("ast").Compare)
                          for n in __import__("ast").walk(cmp_)}
        used_as_camera = sorted({n.id for n in __import__("ast").walk(tree)
                                 if isinstance(n, __import__("ast").Name) and n.id in artifact and id(n) not in inside_compare})
        assert not used_as_camera, f"{name} trains with a released model's camera: {used_as_camera}"


def test_scripts_that_load_released_weights_do_not_hand_them_the_measured_camera():
    import re

    keepers = [p for p in ENDSIDE.glob("*.py")
               if "HoverGeoNet240Frontend(" in p.read_text(encoding="utf-8") and p.name not in TRAINERS
               and p.name not in ("hd_dynamic_contour_slam.py", "rigorous_audit_geo_gt.py")]
    assert len(keepers) >= 12
    for p in keepers:
        src = p.read_text(encoding="utf-8")
        for call in re.finditer(r"HoverGeoNet240Frontend\((.*?)\)", src, flags=re.S):
            assert "REAL_FOV_H_DEG" not in call.group(1), f"{p.name}: hands the released weights the measured camera"


def test_load_rejects_an_unattributed_fact(tmp_path):
    bad = {"facts": {"x": {"value": 1, "method": "guess"}}}
    p = tmp_path / "t.json"
    p.write_text(__import__("json").dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        rt.load(p)


def test_const_scan_flags_conflicts_and_passes_when_triaged():
    """The scan must report a 60 deg field of view as a conflict; with --check it must exit 1
    while anything is untriaged, and 0 once every conflict has a disposition."""
    env = {"PACK_ROOT": str(ROOT / "runs_end_side" / "_const_scan_tmp"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "C:\\Windows")}
    r = subprocess.run([sys.executable, str(ROOT / "pc" / "pack" / "const_scan.py"), "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, env={**__import__("os").environ, **env}, timeout=300)
    m = re.search(r"untriaged (\d+), stale dispositions (\d+)", r.stdout)
    assert m, r.stdout + r.stderr
    n_untriaged, n_stale = int(m.group(1)), int(m.group(2))
    assert r.returncode == (1 if (n_untriaged or n_stale) else 0), r.stdout
    report = pathlib.Path(r.stdout.split("report: ")[1].strip())
    text = report.read_text(encoding="utf-8")
    assert "60" in text and "75.5" in text          # the firmware default and the truth both show up


def test_the_repository_currently_has_no_untriaged_camera_constant():
    """The gate itself: every field of view and camera height in pc/ and the firmware has a disposition."""
    env = {"PACK_ROOT": str(ROOT / "runs_end_side" / "_const_scan_tmp"), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "C:\\Windows")}
    r = subprocess.run([sys.executable, str(ROOT / "pc" / "pack" / "const_scan.py"), "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, env={**__import__("os").environ, **env}, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "untriaged 0, stale dispositions 0" in r.stdout
