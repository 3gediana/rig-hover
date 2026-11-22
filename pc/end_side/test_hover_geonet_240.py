# -*- coding: utf-8 -*-
"""test_hover_geonet_240.py — 本地化 HoverGeoNet-240、2mm Dynablox 零残影与 Scenario 2 闭环重定位硬门禁单测。"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hd_dynamic_contour_slam import (  # noqa: E402
    HdDynamicContourSlamEngine,
    HoverGeoNet240Frontend,
)
from hover_geonet_240 import (  # noqa: E402
    GEO_H,
    GEO_W,
    REAL_FOV_H_DEG,
    REAL_H_CAM_M,
    compute_desk_prior_from_rays,
    decode_geonet_to_world_cloud,
)
from mono_depth_model import make_camera_ray_grid  # noqa: E402

CKPT_FP16 = REPO / "runs_end_side" / "hover_geonet_240" / "hover_geonet_240_fp16.pt"
TRAIN_REPORT = REPO / "runs_end_side" / "hover_geonet_240" / "train_eval_report.json"
SCEN2_SUMMARY = REPO / "runs_end_side" / "scenario2_closed_loop_v7" / "scenario2_summary.json"
SCEN1_E2E_REPORT = REPO / "runs_end_side" / "scenario1_own_models_e2e" / "scenario1_e2e_report.json"
SCEN2_OWN_REPORT = REPO / "runs_end_side" / "scenario2_own_models_e2e" / "scenario2_own_models_report.json"
VIT_CKPT = REPO / "runs_end_side" / "hover_geonet_vit_240" / "hover_geonet_vit_b1_fp16.pt"
EMPTY_DIR = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"


def test_1_footprint_latency_and_desk_prior_identity() -> None:
    assert CKPT_FP16.exists(), f"Missing trained FP16 checkpoint: {CKPT_FP16}"
    mb = CKPT_FP16.stat().st_size / 1048576.0
    assert mb < 7.5, f"FP16 model too large: {mb:.2f} MB (expected < 7.5 MB)"

    # 解析桌面先验 Z=0 精确恒等验证 (H_cam = 0.135m)
    rays = make_camera_ray_grid(GEO_H, GEO_W, fov_h_deg=REAL_FOV_H_DEG, pitch_deg=4.2, roll_deg=-3.8)
    inv_p = compute_desk_prior_from_rays(rays, h_cam_m=REAL_H_CAM_M)
    out5_zero = np.zeros((5, GEO_H, GEO_W), dtype=np.float32)
    out5_zero[3] = 1.0  # nz = +1
    out5_zero[4] = 5.0  # valid logit high
    dec = decode_geonet_to_world_cloud(out5_zero, rays, inv_p, h_cam_m=REAL_H_CAM_M, auto_level=False)
    desk_rays = (rays[..., 2] < -0.10) & dec["valid_grid"]
    max_z_err_mm = float(np.max(np.abs(dec["pts_grid"][desk_rays, 2])) * 1000.0)
    assert max_z_err_mm < 0.05, f"Desk prior identity violated: max |Z| = {max_z_err_mm:.4f} mm"

    # 实测推理速度与实景输出有效性
    fe = HoverGeoNet240Frontend(CKPT_FP16, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
    jpgs = sorted(EMPTY_DIR.glob("*.jpg"))
    bgr0 = cv2.imread(str(jpgs[10]))
    bgr1 = cv2.imread(str(jpgs[12]))
    times = []
    for _ in range(15):
        out = fe.infer_frame(bgr1, bgr0, pitch_deg=3.0, roll_deg=0.0, auto_level=True)
        times.append(out["infer_ms"])
    med_ms = float(np.median(times))
    assert med_ms < 16.0, f"HoverGeoNet-240 GPU latency too slow: {med_ms:.2f} ms"
    assert len(out["pts"]) > 25000, f"Too few dense points from 240x240 model: {len(out['pts'])}"
    print(f"[PASS] test_1: FP16={mb:.2f}MB (<7.5MB), GPU_p50={med_ms:.2f}ms, desk_prior_err={max_z_err_mm:.4f}mm, pts={len(out['pts']):,}", flush=True)


def test_2_strictly_heldout_zero_shot_metrics() -> None:
    assert TRAIN_REPORT.exists(), f"Missing train report: {TRAIN_REPORT}"
    rep = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    for key in ("heldout_empty_d05_with_imu", "heldout_cb8455_with_imu"):
        m = rep[key]
        assert m["abs_rel"] < 0.18, f"{key} abs_rel too high: {m['abs_rel']}"
        assert m["desk_z_mad_mm"] < 5.0, f"{key} desk_z_mad_mm too high: {m['desk_z_mad_mm']}"
        assert m["normal_cos"] > 0.89, f"{key} normal_cos too low: {m['normal_cos']}"
    print(
        f"[PASS] test_2: Held-out empty_d0.5 absRel={rep['heldout_empty_d05_with_imu']['abs_rel']}, "
        f"deskMAD={rep['heldout_empty_d05_with_imu']['desk_z_mad_mm']}mm, nrmCos={rep['heldout_empty_d05_with_imu']['normal_cos']} | "
        f"cb8455 absRel={rep['heldout_cb8455_with_imu']['abs_rel']}, "
        f"deskMAD={rep['heldout_cb8455_with_imu']['desk_z_mad_mm']}mm, nrmCos={rep['heldout_cb8455_with_imu']['normal_cos']}",
        flush=True,
    )


def test_3_hd_dynablox_zero_ghost_and_static_preservation() -> None:
    """验证 2.0 mm 超密建图 + Dynablox Occupied-Protected Ever-Free 在入侵物移动与短暂停留时 0 静态残留，同时保护静态水杯。"""
    slam = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=REAL_H_CAM_M)
    rays = make_camera_ray_grid(120, 120, fov_h_deg=REAL_FOV_H_DEG, pitch_deg=6.0, roll_deg=0.0)
    cam_o = np.array([0.0, 0.0, REAL_H_CAM_M], dtype=np.float32)

    def make_scene(intruder_y: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = np.where(rays[..., 2] < -1e-4, -cam_o[2] / np.minimum(rays[..., 2], -1e-4), 1.40)
        t = np.minimum(t, 1.40)
        n = np.zeros((120, 120, 3), dtype=np.float32)
        n[..., 2] = 1.0
        n[t >= 1.39] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        gray = np.full((120, 120), 115, dtype=np.uint8)

        # 永久静态水杯在 (x=0.68, y=+0.18, z=0..0.14)
        ox, oy = cam_o[0] - 0.68, cam_o[1] - 0.18
        a = rays[..., 0] ** 2 + rays[..., 1] ** 2
        b = 2.0 * (ox * rays[..., 0] + oy * rays[..., 1])
        c = ox * ox + oy * oy - 0.045 ** 2
        disc = b * b - 4.0 * a * c
        ok = (disc > 0) & (a > 1e-6)
        tc = np.where(ok, (-b - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(2 * a, 1e-6), 3.0)
        zc = cam_o[2] + tc * rays[..., 2]
        hc = ok & (tc > 0.12) & (tc < t) & (zc >= 0.0) & (zc <= 0.14)
        t[hc] = tc[hc]
        n[hc] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        gray[hc] = 185

        # 动态入侵方块 (模拟挥动并悬停的人手) 在 x=0.44, y=intruder_y, z=0.06..0.18
        if intruder_y is not None:
            tx = (0.44 - cam_o[0]) / np.maximum(rays[..., 0], 1e-4)
            py = cam_o[1] + tx * rays[..., 1]
            pz = cam_o[2] + tx * rays[..., 2]
            hi = (tx > 0.12) & (tx < t) & (np.abs(py - intruder_y) < 0.055) & (pz >= 0.06) & (pz <= 0.18)
            t[hi] = tx[hi]
            n[hi] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            gray[hi] = 230

        pts = cam_o[None, None, :] + t[..., None] * rays
        cr = np.zeros((120, 120), dtype=np.float32)
        valid = (t > 0.14) & (t < 1.55) & (pts[..., 2] > -0.02) & (pts[..., 2] < 0.50)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr, pts.astype(np.float32), n, cr, valid

    # 1) 4 帧干净空桌建立 Ever-Free 空域与已确认静态水杯
    for _ in range(4):
        bgr, pts, n, cr, val = make_scene(intruder_y=None)
        slam.process_frame(bgr, pts, n, cr, val, align=False)

    # 2) 动态手进入空域并悬停静止 3 帧！
    for iy in (-0.08, -0.02, -0.02, -0.02):
        bgr, pts, n, cr, val = make_scene(intruder_y=iy)
        rec = slam.process_frame(bgr, pts, n, cr, val, align=False)
        assert rec["n_dynamic_frame"] > 80, f"Dynamic intruder not routed to DynamicLayer: {rec}"

    # 3) 动态手离开后检查静态地图：入侵区应 0 残留，静态水杯应完整保留！
    bgr, pts, n, cr, val = make_scene(intruder_y=None)
    slam.process_frame(bgr, pts, n, cr, val, align=False)
    st_pts, _, _ = slam.get_confirmed_static_cloud(min_wt=1.0)

    leaked = int(np.sum(
        (st_pts[:, 0] > 0.36) & (st_pts[:, 0] < 0.52)
        & (st_pts[:, 1] > -0.16) & (st_pts[:, 1] < 0.08)
        & (st_pts[:, 2] > 0.045)
    ))
    cup_pts = int(np.sum(
        (st_pts[:, 0] > 0.60) & (st_pts[:, 0] < 0.76)
        & (st_pts[:, 1] > 0.10) & (st_pts[:, 1] < 0.26)
        & (st_pts[:, 2] > 0.03)
    ))
    assert leaked == 0, f"Static map leaked {leaked} ghost points from paused dynamic intruder!"
    assert cup_pts > 150, f"Static cup was eroded too aggressively: only {cup_pts} pts left!"
    assert len(slam.dyn_pts) == 0, f"Dynamic layer did not clear after intruder left: {len(slam.dyn_pts)}"
    print(f"[PASS] test_3: Dynablox 2mm zero-ghost passed (leaked={leaked}, cup_preserved={cup_pts}, dyn_after_leave={len(slam.dyn_pts)})", flush=True)


def test_4_scenario2_closed_loop_visual_relocalization() -> None:
    assert SCEN2_SUMMARY.exists(), f"Missing Scenario 2 summary: {SCEN2_SUMMARY}"
    s = json.loads(SCEN2_SUMMARY.read_text(encoding="utf-8"))
    blind = s["mode_a_blind_odometry"]
    vis = s["mode_b_visual_3d_reloc"]
    assert not vis["collision"], f"Visual-3D reloc collided with obstacle! clearance={vis['min_obstacle_clearance_cm']}cm"
    assert vis["final_true_err_to_target_cm"] < 5.0, f"Visual-3D reloc final error too high: {vis['final_true_err_to_target_cm']} cm"
    assert s["improvement_factor"] >= 3.0, f"Expected >= 3x improvement over drifted odometry, got {s['improvement_factor']}x"
    print(
        f"[PASS] test_4: Scenario 2 Closed-Loop Visual-3D Reloc={vis['final_true_err_to_target_cm']}cm "
        f"(vs Blind Odom={blind['final_true_err_to_target_cm']}cm, {s['improvement_factor']}x better, "
        f"clearance={vis['min_obstacle_clearance_cm']}cm, avg_net={s['hover_geonet_avg_ms']}ms)",
        flush=True,
    )


def test_5_own_vit240_scenario1_and_scenario2_e2e() -> None:
    """硬门禁验证：100% 自研模型独立跑通场景一（全5阶段+真录像双敲3D升维）与场景二（真留出重定位+自研PPO闭环）。"""
    assert VIT_CKPT.exists(), f"Missing HoverGeoNet-ViT-240 FP16 checkpoint: {VIT_CKPT}"
    vit_mb = VIT_CKPT.stat().st_size / 1048576.0
    assert vit_mb < 30.0, f"HoverGeoNet-ViT-240 FP16 too large: {vit_mb:.2f} MB"

    assert SCEN1_E2E_REPORT.exists(), f"Missing Scenario 1 E2E report: {SCEN1_E2E_REPORT}"
    r1 = json.loads(SCEN1_E2E_REPORT.read_text(encoding="utf-8"))
    pa = r1["part_a_real_cb8455"]
    pb = r1["part_b_closed_loop_scenario1"]
    assert pa["double_knock_fist_gated"]["fp"] == 0, f"Expected 0 FP on fist-gated cb8455 LOO: {pa}"
    assert pa["double_knock_fist_gated"]["hit"] >= 10, f"Too few real knocks detected on cb8455: {pa}"
    assert pb["knock_3d_lift_err_cm"] < 3.0, f"3D knock lift error too high: {pb['knock_3d_lift_err_cm']} cm"
    assert pb["own_models"]["all_5_phases_completed"] is True, f"Scenario 1 did not complete all 5 phases: {pb['own_models']}"
    assert pb["own_models"]["collision"] is False, f"Scenario 1 collided: {pb['own_models']}"
    assert pb["own_models"]["final_err_to_true_knock_cm"] < 4.0, f"Scenario 1 knock arrival error too high: {pb['own_models']}"
    assert pb["own_models"]["lookback_face_err_deg"] < 5.0, f"Scenario 1 lookback face error too high: {pb['own_models']}"

    assert SCEN2_OWN_REPORT.exists(), f"Missing Scenario 2 own-models report: {SCEN2_OWN_REPORT}"
    r2 = json.loads(SCEN2_OWN_REPORT.read_text(encoding="utf-8"))
    sc_e = r2["part_ab_real_heldout"]["scenes"]["empty_d05"]["real_frame_reloc"]
    sc_c = r2["part_ab_real_heldout"]["scenes"]["cb8455"]["real_frame_reloc"]
    assert sc_e["ok"] and sc_e["target_rel_err_mm"] < 20.0 and sc_e["recovered_rot_err_deg"] < 2.0, f"empty_d05 real reloc failed: {sc_e}"
    assert sc_c["ok"] and sc_c["target_rel_err_mm"] < 20.0 and sc_c["recovered_rot_err_deg"] < 2.0, f"cb8455 real reloc failed: {sc_c}"
    cl = r2["part_c_closed_loop_own_models"]
    assert cl["own_models"]["collision"] is False, f"Scenario 2 own-models collided: {cl['own_models']}"
    assert cl["own_models"]["final_err_cm"] < 5.5, f"Scenario 2 own-models arrival error too high: {cl['own_models']}"
    assert cl["improvement_factor"] >= 4.0, f"Expected >= 4x vs drifted odom, got {cl['improvement_factor']}x"
    print(
        f"[PASS] test_5: 100% Own Models E2E | ViT={vit_mb:.1f}MB | "
        f"Scen1 5-phase err={pb['own_models']['final_err_to_true_knock_cm']}cm lookback={pb['own_models']['lookback_face_err_deg']}deg | "
        f"Scen2 real_reloc={sc_e['target_rel_err_mm']}/{sc_c['target_rel_err_mm']}mm closed_loop={cl['own_models']['final_err_cm']}cm ({cl['improvement_factor']}x)",
        flush=True,
    )


if __name__ == "__main__":
    test_1_footprint_latency_and_desk_prior_identity()
    test_2_strictly_heldout_zero_shot_metrics()
    test_3_hd_dynablox_zero_ghost_and_static_preservation()
    test_4_scenario2_closed_loop_visual_relocalization()
    test_5_own_vit240_scenario1_and_scenario2_e2e()
    print("=== ALL 5/5 HOVER-GEONET & SCENARIO-1/2 OWN-MODEL GATES PASSED ===", flush=True)
