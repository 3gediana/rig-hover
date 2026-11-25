# -*- coding: utf-8 -*-
"""pc/pack/const_scan.py - find every hard-coded field of view / camera height and give each a disposition.

The same camera had at least eight different fields of view written into this
repository (50, 52, 56, 60, 62.5, 65, 72 deg) and three camera heights
(0.115, 0.135, 0.2 m); the measured values are 75.5 deg and about 0.135 m
(robot_truth.json). This scan is what stops a ninth value from appearing
unnoticed.

Each hit is reported with a status:
  ALIGNED     the written value matches the measured truth within tolerance
  TRIAGED     a conflict, with a recorded disposition in DISPOSITIONS below
              (fixed / annotated / kept, and why)
  UNTRIAGED   a conflict nobody has looked at yet  -> exit code 1

Usage:
    python pc/pack/const_scan.py            # scan, write PACK_ROOT/findings/CONST_SCAN.md
    python pc/pack/const_scan.py --check    # same, exit 1 if anything is UNTRIAGED (gate)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402
import robot_truth as rt  # noqa: E402

SCAN_ROOTS = ["pc", "rig_omni/main"]
SKIP_DIRS = {"__pycache__", "build", "managed_components", "data", "synth", "raw", "models", "_teacher_cache"}
SCAN_EXT = (".py", ".c", ".cc", ".h", ".cpp")

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
# The keyword may start the name (fov_h_deg, cam_z) or sit inside it (REAL_FOV_H_DEG): the prefix is optional.
FOV_NAME = re.compile(rf"\b((?:{_NAME})?(?:FOV|Fov|fov|F_PX|PX_PER_DEG|[fF]ocal|f_px)[A-Za-z0-9_]*)"
                      r"\s*(?::\s*[\w\[\], .]+)?=\s*(-?\d+(?:\.\d+)?)(?:f)?\b")
H_NAME = re.compile(rf"\b((?:{_NAME})?(?:CAM_H|H_CAM|cam_h|h_cam|CAM_HEIGHT|cam_height|CAM_Z|cam_z)[A-Za-z0-9_]*)"
                    r"\s*(?::\s*[\w\[\], .]+)?=\s*(-?\d+(?:\.\d+)?)(?:f)?\b")
KW_FOV = re.compile(r"\b([A-Za-z_]*(?:fov|ocal|f_px|px_per_deg)[A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\b")
KW_H = re.compile(r"\b([A-Za-z_]*(?:cam_h|h_cam|cam_height|cam_z)[A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)\b")

FOV_TOL_DEG = 0.3
H_TOL_M = 0.0025
FOV_PLAUSIBLE = (40.0, 130.0)      # outside this a "fov" number is a pixel count, a flag, etc.
H_PLAUSIBLE = (0.05, 0.5)          # outside this a "cam height" number is something else

# Sites that used to carry a wrong camera and now read it from robot_truth (or from the weights they load), so the
# scan no longer sees a number there. Kept in the report so the fix stays auditable.
FIXED = [
    ("pc/s2r/link.py, pc/s2r/controllers.py", "2.705 px/deg, 240 px, 0.135 m, 7 deg typed in",
     "robot_truth; pixel <-> angle is now the exact pinhole form of the measured focal length"),
    ("pc/end_side/real_robot_bridge.py", "62.5 deg", "robot_truth (75.5 deg)"),
    ("pc/end_side/hd_moge_contour_engine.py", "REAL_FOV_H_DEG = 65.0", "robot_truth for MoGe-2 (it takes the camera as input); the GeoNet experiment family now uses GEONET_CKPT_FOV_H_DEG"),
    ("pc/end_side/hover_geonet_240.py", "REAL_FOV_H_DEG = 65.0", "robot_truth: new training starts from the measured camera"),
    ("pc/end_side/mono_depth_model.py", "0.115 m / 60 deg", "robot_truth for generic geometry and new training; the released v1 bundles keep their own (RELEASED_V1_*)"),
    ("pc/end_side/train_dense_splat_student.py, dense_scene_pipeline.py, dense_splat_student.py", "72 deg / 0.115 m as training default",
     "robot_truth for new training, caches and output tagged with the camera; the released v3 bundle keeps 72 deg"),
    ("pc/end_side/jarvis_4d_holo_engine_v10.py, scenario2_visual_nav_closed_loop.py", "hard-coded 'FOV=65.0' in a label and a comment that called it calibrated",
     "prints the constant it actually uses"),
    ("test_hover_geonet_240.py, test_mono_depth.py, test_neural_gaussian_splat.py, test_real_frame_3dgs.py", "released weights evaluated through module defaults",
     "each gate reads the camera recorded in the weights / model.json it loads"),
]

DENSE_V3 = ("kept", "training camera of the released dense-splat v3 bundle (runs_end_side/dense_splat_v3/model.json: "
             "0.115 m / 72 deg). Everything that loads that bundle keeps this camera; new training starts from the "
             "measured one (train_dense_splat_student.py). On real frames its ground ranges are off until it is retrained.")
ORBIT = ("annotated", "not the robot's camera: `fov_vertical_deg` belongs to OrbitCamera3D, the virtual overhead camera "
                      "that renders the memory map for viewing.")
LEGACY_ONBOARD = ("kept", "convention of the legacy onboard policy, not a claim about the lens: theta_img = (cx - 0.5) * "
                          "fov_h is part of the 116-dim observation its released blobs were trained with, at 60 deg, "
                          "paired with the firmware default (control_arbiter.c). Changing it means retraining those blobs "
                          "and reflashing. The end-side chain does not use it.")

# file (repo-relative, forward slashes) or file::name -> (status, why). "file::name" wins over "file".
# status: fixed (value now comes from robot_truth or from the weights), kept (deliberately different, reason given),
#         dormant (code path cannot run on the robot as shipped), annotated (not a statement about the robot's camera).
DISPOSITIONS: dict[str, tuple[str, str]] = {
    # ---- the camera a released model was trained with (recorded in the weights or the manifest) ----
    "pc/end_side/dense_splat_student.py": DENSE_V3,
    "pc/end_side/robust_contour_slam.py": DENSE_V3,
    "pc/end_side/proto_3d_edl_traversable.py": DENSE_V3,
    "pc/end_side/proto_static_vr_scene.py": DENSE_V3,
    "pc/end_side/verify_balancing_rocking_survival.py": DENSE_V3,
    "pc/end_side/test_dense_scene_v3.py": DENSE_V3,
    "pc/end_side/test_robust_contour_slam.py": DENSE_V3,
    "pc/end_side/hd_moge_contour_engine.py::GEONET_CKPT_FOV_H_DEG": (
        "kept", "training camera of the released GeoNet weights (v3, MV-v4, ViT-B; each .pt records 0.135 m / 65 deg). "
                "The experiment scripts built around those weights use it so they keep measuring what they measured. "
                "Not the measured camera: on real frames those weights put the ground about 22 % too far (AUD-015) "
                "until they are retrained at the measured one."),
    "pc/end_side/mono_depth_model.py": (
        "kept", "training camera of the released v1 bundles (mono depth v1, gaussian splat v2; both model.json record "
                "0.115 m / 60 deg, the Android copies too). Same weights on one synthetic scene: AbsRel 0.0455 at this "
                "camera, 0.2342 at the measured one (AUD-016). Retraining at the measured camera retires this."),
    "pc/end_side/train_room360_self_corrector_v12.py": (
        "kept", "input definition of the room-360 model: 70 deg pinhole views cut from 360 panoramas. A synthetic camera "
                "the model's ray channels are built with; it has nothing to do with the robot's lens."),
    "pc/end_side/room360_self_correcting_v12.py": (
        "kept", "docstring of the room-360 model's 70 deg panorama pinhole views (synthetic input definition, see "
                "train_room360_self_corrector_v12.py)."),
    # ---- deliberately miscalibrated, or a value quoted from elsewhere ----
    "pc/end_side/sim2real_torture_and_bridge_v16.py": (
        "kept", "stress arenas that are meant to be miscalibrated: S3 renders at 56 deg while the brain assumes the "
                "released GeoNet camera (65 deg), to test its boot-time self-calibration; the docstring quotes the "
                "firmware's uncalibrated 60 deg. Neither number claims to be the measured camera."),
    "pc/end_side/sandbox_arena.py": (
        "annotated", "synthetic scenario parameter: a face-visibility half angle (72 deg for the S5 wide-angle case), "
                     "used as a gate on the target bearing. Not a camera calibration."),
    # ---- not the robot's camera at all ----
    "pc/end_side/neural_gaussian_splat_model.py": ORBIT,
    "pc/end_side/train_neural_gaussian_splat.py": ORBIT,
    "pc/end_side/real_frame_3dgs_eval.py": ORBIT,
    "pc/perception/test_perception.py": ("annotated", "unit test of the fov_h setter with arbitrary values (60, 120)."),
    # ---- legacy onboard policy ----
    "pc/perception/run_perception.py": LEGACY_ONBOARD,
    "pc/perception/scene_signals.py": LEGACY_ONBOARD,
    "pc/sim/params.py": LEGACY_ONBOARD,
    "pc/train/arbtest/control_arbiter_under_test.c": (
        "kept", "host-test copy of the firmware default; must stay identical to control_arbiter.c (see that file)."),
    "rig_omni/main/boards/hover/control_arbiter.c": (
        "kept", "firmware runtime default (`/cfg` key fov_h, clamped 10..180 deg). " + LEGACY_ONBOARD[1]),
    # ---- dormant ----
    "pc/perception/table_edge.py": (
        "dormant", "table-edge detector the user decided not to use (AGENTS.md section 3.1); its 0.20 m camera height "
                   "was a design value for another mounting. Not on the robot's path."),
}


def iter_files():
    me = Path(__file__).resolve()
    for root in SCAN_ROOTS:
        base = paths.REPO / root
        for p in base.rglob("*"):
            if p.suffix.lower() not in SCAN_EXT or p.resolve() == me:   # this file quotes the history it documents
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def scan_file(p: Path, truth_fov: float, truth_h: float) -> list[dict]:
    hits = []
    rel = p.relative_to(paths.REPO).as_posix()
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return hits
    for i, line in enumerate(lines, 1):
        s = line.split("#")[0] if p.suffix == ".py" else line
        s = re.sub(r"/\*.*?\*/", " ", s)
        s = re.sub(r"//.*$", " ", s)
        for kind, rx, truth, tol, plausible in (("fov", FOV_NAME, truth_fov, FOV_TOL_DEG, FOV_PLAUSIBLE),
                                                ("fov", KW_FOV, truth_fov, FOV_TOL_DEG, FOV_PLAUSIBLE),
                                                ("cam_h", H_NAME, truth_h, H_TOL_M, H_PLAUSIBLE),
                                                ("cam_h", KW_H, truth_h, H_TOL_M, H_PLAUSIBLE)):
            for m in rx.finditer(s):
                name, val = m.group(1), float(m.group(2))
                if not (plausible[0] <= val <= plausible[1]):
                    continue
                aligned = abs(val - truth) <= tol
                hits.append({"file": rel, "line": i, "kind": kind, "name": name, "value": val,
                             "aligned": aligned, "text": line.strip()[:140]})
    # de-duplicate identical (kind,name,value,line)
    seen, out = set(), []
    for h in hits:
        k = (h["file"], h["line"], h["kind"], h["name"], h["value"])
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    truth_fov, truth_h = rt.FOV_H_DEG, rt.CAM_HEIGHT_M
    all_hits = []
    for p in iter_files():
        all_hits += scan_file(p, truth_fov, truth_h)
    for h in all_hits:
        d = DISPOSITIONS.get(h["file"] + "::" + h["name"]) or DISPOSITIONS.get(h["file"])
        h["status"] = "ALIGNED" if h["aligned"] else (f"TRIAGED:{d[0]}" if d else "UNTRIAGED")
        h["why"] = "" if h["aligned"] else (d[1] if d else "")
    n_aligned = sum(1 for h in all_hits if h["aligned"])
    conflicts = [h for h in all_hits if not h["aligned"]]
    untriaged = [h for h in conflicts if h["status"] == "UNTRIAGED"]
    stale = []
    for k in DISPOSITIONS:
        f, _, name = k.partition("::")
        if not any(h["file"] == f and (not name or h["name"] == name) for h in conflicts):
            stale.append(k)
    values = sorted({h["value"] for h in all_hits if h["kind"] == "fov"})
    hvalues = sorted({h["value"] for h in all_hits if h["kind"] == "cam_h"})
    paths.FINDINGS.mkdir(parents=True, exist_ok=True)
    L = [f"# Constant scan: field of view and camera height ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"Truth: FOV_H = {truth_fov} deg, camera height = {truth_h} m (`pc/robot_truth.json`).",
         "Three kinds of camera exist in this repository and the scan tells them apart: the measured one "
         "(`robot_truth`: real-robot chains, general models, new training), the camera a released model was trained "
         "with (kept where that model is loaded), and synthetic or virtual cameras (annotated).", "",
         f"Hits: {len(all_hits)}; aligned with truth: {n_aligned}; conflicts: {len(conflicts)} "
         f"(triaged {len(conflicts) - len(untriaged)}, **untriaged {len(untriaged)}**).", "",
         f"Distinct FOV values written in code: {values}", "",
         f"Distinct camera-height values written in code: {hvalues}", "",
         "Status meanings: ALIGNED = matches the measured value; TRIAGED:kept = deliberately different, reason given; "
         "TRIAGED:annotated = not a statement about the robot's camera; TRIAGED:dormant = cannot run on the robot as "
         "shipped; UNTRIAGED = nobody has looked at it.", ""]
    L += [f"## Fixed since the first scan on 2026-11-25 ({len(FIXED)})", "",
          "These sites no longer appear as hits because the number is gone from the source.", "",
          "| where | what was written | now |", "|---|---|---|"]
    L += [f"| `{a}` | {b} | {c} |" for a, b, c in FIXED]
    if stale:
        L += ["", "Dispositions that match nothing any more (remove them): " + ", ".join(f"`{s}`" for s in stale)]
    L.append("")
    for status in ("UNTRIAGED", "TRIAGED:kept", "TRIAGED:annotated", "TRIAGED:dormant"):
        rows = [h for h in conflicts if h["status"] == status]
        if not rows:
            continue
        L += [f"## {status} ({len(rows)})", "", "| file:line | kind | name | value | why |", "|---|---|---|---:|---|"]
        for h in rows:
            L.append(f"| `{h['file']}:{h['line']}` | {h['kind']} | `{h['name']}` | {h['value']:g} | {h['why']} |")
        L.append("")
    L += ["## ALIGNED", "", "| file:line | kind | name | value |", "|---|---|---|---:|"]
    for h in all_hits:
        if h["aligned"]:
            L.append(f"| `{h['file']}:{h['line']}` | {h['kind']} | `{h['name']}` | {h['value']:g} |")
    (paths.FINDINGS / "CONST_SCAN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (paths.FINDINGS / "const_scan.json").write_text(json.dumps(all_hits, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"hits {len(all_hits)}, aligned {n_aligned}, conflicts {len(conflicts)}, untriaged {len(untriaged)}, "
          f"stale dispositions {len(stale)}")
    for k in stale:
        print(f"  stale disposition (nothing matches it any more, remove it): {k}")
    print("FOV values in code:", values)
    print("camera-height values in code:", hvalues)
    print("report:", paths.FINDINGS / "CONST_SCAN.md")
    return 1 if (a.check and (untriaged or stale)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
