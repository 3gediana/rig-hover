# -*- coding: utf-8 -*-
"""test_robust_contour_slam.py — 开源 SLAM 工业机制（KISS-ICP / Dynablox / ElasticFusion / Unblur-SLAM / EDL 全绿轮廓）五项硬性验收测试。"""
from __future__ import annotations

import math
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mono_depth_model import DEFAULT_H_CAM_M, make_camera_ray_grid  # noqa: E402
from neural_gaussian_splat_model import OrbitCamera3D  # noqa: E402
from robust_contour_slam import (  # noqa: E402
    KissIcpStaticAligner,
    RobustContourSlamPipeline,
    RobustDualLayerContourMap,
    render_green_contour_edl,
)

TFLITE_PATH = REPO / "runs_end_side" / "dense_splat_v3" / "model.tflite"
# O-003/D-010: regression gates must not burn the sealed set; use the non-sealed dev take 65a54d.
DEV_65A54D = REPO / "dataset" / "takes" / "robot_20261115_150045_65a54d" / "take.mp4"


def _make_sharp_texture(h: int = 160, w: int = 160) -> np.ndarray:
    """高对比度棋盘格纹理（用于测试 Unblur-SLAM 清晰度门控）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    pat = (((xx // 8) + (yy // 8)) % 2) * 180 + 35
    return np.stack([pat, pat, pat], axis=-1).astype(np.uint8)


def _synth_desk_and_box(
    rays: np.ndarray,
    with_intruder: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """合成带桌面 (Z=0)、远端 L 型静态墙 (0.95m) 及可选近处入侵物 (0.42m) 的世界点云。"""
    h, w, _ = rays.shape
    depth = np.full((h, w), 0.95, dtype=np.float32)
    # 下半区桌面平面 Z = 0
    desk_mask = rays[..., 2] < -0.10
    depth_desk = np.where(desk_mask, -DEFAULT_H_CAM_M / np.minimum(rays[..., 2], -1e-4), 0.95)
    use_desk = desk_mask & (depth_desk > 0.18) & (depth_desk < 0.95)
    depth[use_desk] = depth_desk[use_desk]

    # 右侧固定立柱（用于约束偏航 yaw 与平移）
    depth[50:130, 128:156] = 0.65

    if with_intruder:
        # 位于原本空域中的人手/前景障碍（即便连续静止多帧，也不应进入 StaticMap）
        depth[55:105, 58:102] = 0.40

    xyz = rays * depth[..., None]
    xyz[..., 2] += DEFAULT_H_CAM_M

    # 网格差分法向并朝向相机
    dx = np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)
    dy = np.roll(xyz, -1, axis=0) - np.roll(xyz, 1, axis=0)
    nrm = np.cross(dy, dx)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    flip = np.sum(nrm * rays, axis=-1, keepdims=True) > 0
    nrm = np.where(flip, -nrm, nrm)

    valid = np.ones((h, w), dtype=bool)
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False
    return xyz[valid].astype(np.float32), nrm[valid].astype(np.float32), depth


def test_1_dynablox_ever_free_blocks_paused_intruder() -> dict:
    """Dynablox Ever-Free：人手进入曾为空域的桌面并停顿 5 帧，静态层零泄漏、离开后零残影。"""
    h, w = 160, 160
    rays = make_camera_ray_grid(h, w, fov_h_deg=72.0, pitch_deg=0.0, roll_deg=0.0)
    sharp = _make_sharp_texture(h, w)
    eye3 = np.eye(3, dtype=np.float32)

    pts_empty, nrm_empty, d_empty = _synth_desk_and_box(rays, with_intruder=False)
    pts_hand, nrm_hand, d_hand = _synth_desk_and_box(rays, with_intruder=True)

    cmap = RobustDualLayerContourMap(vox_size=0.0085, fov_h_deg=72.0, confirm_hits=2)
    # 先跑 3 帧干净空桌面 -> 建立已确认静态骨架 + 桌面以上 Ever-Free 历史空域
    for _ in range(3):
        cmap.update(sharp, pts_empty, nrm_empty, d_empty, eye3)

    ef_count = cmap.ever_free.confirmed_voxel_count
    assert ef_count > 500, f"ever_free grid failed to establish empty air: {ef_count}"

    # 入侵人手在 0.40m 处静止停留 5 帧
    dyn_counts = []
    for _ in range(5):
        cmap.update(sharp, pts_hand, nrm_hand, d_hand, eye3)
        dyn_counts.append(len(cmap.d_pos))

    # 检查静态地图在入侵区域 (x∈[0.34,0.46], y∈[-0.18,0.08], z>0.04) 内是否零泄漏
    s_pos, _ = cmap.get_static_confirmed()
    leaked_static = int(
        np.sum(
            (s_pos[:, 0] > 0.34)
            & (s_pos[:, 0] < 0.46)
            & (s_pos[:, 1] > -0.18)
            & (s_pos[:, 1] < 0.08)
            & (s_pos[:, 2] > 0.045)
        )
    )
    assert min(dyn_counts) > 100, f"intruder was not routed to DynamicLayer: {dyn_counts}"
    assert leaked_static == 0, f"paused intruder leaked into StaticMap: {leaked_static} voxels"

    # 人手离开后 1 帧，DynamicLayer 立即归零，静态骨架零幽灵
    cmap.update(sharp, pts_empty, nrm_empty, d_empty, eye3)
    assert len(cmap.d_pos) == 0, f"DynamicLayer did not clear after intruder left: {len(cmap.d_pos)}"
    return {
        "ever_free_voxels": ef_count,
        "dyn_routed_per_frame": dyn_counts,
        "static_leaked_voxels": leaked_static,
        "dyn_after_leave": len(cmap.d_pos),
    }


def test_2_kiss_icp_recovers_pose_under_outliers() -> dict:
    """KISS-ICP 自适应 Geman-McClure Point-to-Plane 配准：在 25% 外点下精确纠回 +5.5°俯仰 / -4.5°偏航 / 2.5cm偏移。"""
    rng = np.random.default_rng(20261122)
    # 构造静态 L 型场景：桌面 (Z=0) + 正前墙 (X=0.85) + 侧墙 (Y=0.45)
    gx, gy = np.meshgrid(np.linspace(0.25, 0.85, 36), np.linspace(-0.45, 0.45, 36))
    desk_p = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1)
    desk_n = np.tile(np.array([0.0, 0.0, 1.0]), (len(desk_p), 1))

    wy, wz = np.meshgrid(np.linspace(-0.45, 0.45, 32), np.linspace(0.02, 0.42, 20))
    wall1_p = np.stack([np.full(wy.size, 0.85), wy.ravel(), wz.ravel()], axis=-1)
    wall1_n = np.tile(np.array([-1.0, 0.0, 0.0]), (len(wall1_p), 1))

    wx, wz2 = np.meshgrid(np.linspace(0.25, 0.85, 24), np.linspace(0.02, 0.42, 20))
    wall2_p = np.stack([wx.ravel(), np.full(wx.size, 0.45), wz2.ravel()], axis=-1)
    wall2_n = np.tile(np.array([0.0, -1.0, 0.0]), (len(wall2_p), 1))

    static_p = np.concatenate([desk_p, wall1_p, wall2_p], axis=0).astype(np.float32)
    static_n = np.concatenate([desk_n, wall1_n, wall2_n], axis=0).astype(np.float32)

    cam_pos = np.array([0.0, 0.0, DEFAULT_H_CAM_M], dtype=np.float32)
    # 注入扰动：pitch +5.5 deg, yaw -4.5 deg, roll +2.0 deg, trans (+0.020, -0.015, +0.010)
    rvec_err = np.array([math.radians(2.0), math.radians(5.5), math.radians(-4.5)], dtype=np.float64)
    r_err, _ = cv2.Rodrigues(rvec_err)
    t_err = np.array([0.020, -0.015, 0.010], dtype=np.float64)
    init_rot_deg = math.degrees(float(np.linalg.norm(rvec_err)))
    init_trans_m = float(np.linalg.norm(t_err))

    src_p = ((static_p - cam_pos[None, :]) @ r_err.T + cam_pos[None, :] + t_err[None, :]).astype(np.float32)
    src_n = (static_n @ r_err.T).astype(np.float32)

    # 额外掺入 25% 恶意动态外点簇（模拟手部突然闯入）
    n_out = len(src_p) // 3
    out_p = np.stack(
        [
            rng.uniform(0.30, 0.55, n_out),
            rng.uniform(-0.20, 0.15, n_out),
            rng.uniform(0.08, 0.30, n_out),
        ],
        axis=-1,
    ).astype(np.float32)
    out_n = np.tile(np.array([-1.0, 0.0, 0.0], np.float32), (n_out, 1))
    src_all_p = np.concatenate([src_p, out_p], axis=0)
    src_all_n = np.concatenate([src_n, out_n], axis=0)

    aligner = KissIcpStaticAligner(init_sigma_m=0.10, max_iters=10)
    res = aligner.align_to_static(src_all_p, src_all_n, static_p, static_n, cam_pos)
    assert res["converged"], f"KISS-ICP failed to converge: {res}"

    # 验证组合误差 R_rem = dR @ r_err 接近单位阵，t_rem = dR @ t_err + dt 接近 0
    r_rem = res["dR"].astype(np.float64) @ r_err
    rv_rem, _ = cv2.Rodrigues(r_rem)
    rem_rot_deg = math.degrees(float(np.linalg.norm(rv_rem)))
    t_rem = res["dR"].astype(np.float64) @ t_err + res["dt"].astype(np.float64)
    rem_trans_m = float(np.linalg.norm(t_rem))

    assert rem_rot_deg < 0.75, f"residual rotation too large: {rem_rot_deg:.2f} deg (init={init_rot_deg:.2f})"
    assert rem_trans_m < 0.006, f"residual translation too large: {rem_trans_m*1000:.1f} mm (init={init_trans_m*1000:.1f})"
    return {
        "init_rot_deg": round(init_rot_deg, 2),
        "rem_rot_deg": round(rem_rot_deg, 3),
        "init_trans_mm": round(init_trans_m * 1000.0, 1),
        "rem_trans_mm": round(rem_trans_m * 1000.0, 2),
        "inlier_ratio": round(res["inlier_ratio"], 3),
    }


def test_3_disturbance_gate_blocks_blur_and_shock() -> dict:
    """Unblur-SLAM 扰动门控：拦截严重运动模糊帧与 +8.5° 俯仰冲击帧，静态骨架 0 污染。"""
    h, w = 160, 160
    rays = make_camera_ray_grid(h, w, fov_h_deg=72.0, pitch_deg=0.0, roll_deg=0.0)
    sharp = _make_sharp_texture(h, w)
    blurry = cv2.GaussianBlur(sharp, (19, 19), 6.0)
    eye3 = np.eye(3, dtype=np.float32)

    pts, nrm, depth = _synth_desk_and_box(rays, with_intruder=False)
    cmap = RobustDualLayerContourMap(vox_size=0.0085, fov_h_deg=72.0)

    for _ in range(4):
        st = cmap.update(sharp, pts, nrm, depth, eye3, pitch_deg=0.5)
        assert not st.is_disturbed, f"clean frame wrongly rejected: {st}"

    static_before = cmap.s_pos.copy()
    updates_before = cmap.static_updates

    # 注入运动模糊坏帧（携带故意偏移 15cm 的坏点云）
    st_blur = cmap.update(blurry, pts + np.array([0.0, 0.0, 0.15], np.float32), nrm, depth, eye3, pitch_deg=0.6)
    assert st_blur.is_disturbed and "motion_blur" in st_blur.reason, f"blur frame not blocked: {st_blur}"
    assert len(cmap.d_pos) > 0, "live dynamic preview should still render during blur"

    # 注入剧烈俯仰冲击帧 (+8.5 deg 跳变)
    st_shock = cmap.update(sharp, pts + np.array([0.0, 0.0, 0.20], np.float32), nrm, depth, eye3, pitch_deg=9.2)
    assert st_shock.is_disturbed and "pitch_shock" in st_shock.reason, f"pitch shock not blocked: {st_shock}"

    assert cmap.static_updates == updates_before, "static map was mutated during disturbance"
    assert np.array_equal(cmap.s_pos, static_before), "static coordinates mutated during disturbance"
    return {
        "blur_reason": st_blur.reason,
        "blur_ratio": round(st_blur.blur_ratio, 3),
        "shock_reason": st_shock.reason,
        "blocked_total": cmap.disturbed_frames_blocked,
        "static_unchanged": True,
    }


def test_4_green_contour_reveals_pitch_black_object() -> dict:
    """Potree EDL 纯绿色轮廓渲染：即使物体反照率近乎全黑，纯绿轮廓仍清晰高对比且严格绿通道主导。"""
    # 在暗桌面上放置一个圆柱物体（无任何 RGB 输入，纯几何渲染）
    gx, gy = np.meshgrid(np.linspace(0.20, 1.00, 60), np.linspace(-0.40, 0.40, 60))
    desk_p = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1).astype(np.float32)
    desk_n = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(desk_p), 1))

    ang = np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
    zz = np.linspace(0.01, 0.22, 28)
    aa, zzg = np.meshgrid(ang, zz)
    cx, cy, rad = 0.52, 0.0, 0.075
    cyl_p = np.stack([cx + rad * np.cos(aa).ravel(), cy + rad * np.sin(aa).ravel(), zzg.ravel()], axis=-1).astype(np.float32)
    cyl_n = np.stack([np.cos(aa).ravel(), np.sin(aa).ravel(), np.zeros(aa.size)], axis=-1).astype(np.float32)

    pts = np.concatenate([desk_p, cyl_p], axis=0)
    nrm = np.concatenate([desk_n, cyl_n], axis=0)
    cam = OrbitCamera3D(azimuth_deg=-150.0, elevationDeg=35.0, distance_m=0.95, target_x=0.50, target_y=0.0, target_z=0.08) \
        if False else OrbitCamera3D(azimuth_deg=-150.0, elevation_deg=35.0, distance_m=0.95, target_x=0.50, target_y=0.0, target_z=0.08)

    img_bgr = render_green_contour_edl(pts, nrm, cam, w=480, h=360, draw_trav_overlay=False)
    b_ch = img_bgr[..., 0].astype(np.float32)
    g_ch = img_bgr[..., 1].astype(np.float32)
    r_ch = img_bgr[..., 2].astype(np.float32)

    lit = g_ch > 40.0
    assert int(np.sum(lit)) > 3000, f"too few lit green contour pixels: {int(np.sum(lit))}"
    mean_g = float(np.mean(g_ch[lit]))
    mean_r = float(np.mean(r_ch[lit]))
    mean_b = float(np.mean(b_ch[lit]))
    peak_g = float(np.max(g_ch))
    assert peak_g >= 200.0, f"green silhouette contour not bright enough: peak_g={peak_g}"
    assert mean_g > mean_r * 2.5 and mean_g > mean_b * 1.4, f"palette not pure green dominant: R={mean_r:.1f} G={mean_g:.1f} B={mean_b:.1f}"
    return {
        "lit_pixels": int(np.sum(lit)),
        "peak_green": round(peak_g, 1),
        "mean_rgb_lit": [round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)],
    }


def test_5_real_dev_65a54d_end_to_end() -> dict:
    """真机开发集视频 65a54d 端到端压测（纯 .tflite 学生推理 + 开源双层建图）。"""
    assert TFLITE_PATH.exists(), f"missing tflite: {TFLITE_PATH}"
    assert DEV_65A54D.exists(), f"missing video: {DEV_65A54D}"

    cap = cv2.VideoCapture(str(DEV_65A54D))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    frames = []
    for _ in range(10):
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    assert len(frames) == 10, f"failed to read 10 frames from {DEV_65A54D}"

    pipe = RobustContourSlamPipeline(TFLITE_PATH)
    t0 = time.perf_counter()
    for bgr in frames:
        pipe.process_frame(bgr)
    ms_per_frame = (time.perf_counter() - t0) * 1000.0 / len(frames)

    flat = pipe.map.desk_flatness_metrics()
    s_pos, _ = pipe.map.get_static_confirmed()
    n_green = int(pipe.grid.reach.sum())
    assert len(s_pos) > 1000, f"too few confirmed static points: {len(s_pos)}"
    assert flat["z_mad_mm"] < 18.0, f"static desk not flat enough: {flat}"
    assert n_green > 15, f"too few reachable green cells: {n_green}"
    return {
        "ms_per_frame": round(ms_per_frame, 1),
        "static_confirmed_pts": int(len(s_pos)),
        "ever_free_voxels": pipe.map.ever_free.confirmed_voxel_count,
        "desk_z_mad_mm": round(flat["z_mad_mm"], 2),
        "green_cells": n_green,
    }


def main() -> None:
    tests = [
        ("1. Dynablox Ever-Free blocks paused dynamic intruder", test_1_dynablox_ever_free_blocks_paused_intruder),
        ("2. KISS-ICP Geman-McClure recovers 6-DoF pose under 25% outliers", test_2_kiss_icp_recovers_pose_under_outliers),
        ("3. Unblur-SLAM gate blocks motion blur & pitch shock", test_3_disturbance_gate_blocks_blur_and_shock),
        ("4. Potree EDL all-green contour reveals pitch-black geometry", test_4_green_contour_reveals_pitch_black_object),
        ("5. Real dev 65a54d end-to-end .tflite SLAM", test_5_real_dev_65a54d_end_to_end),
    ]
    print("=== test_robust_contour_slam ===", flush=True)
    failed = False
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            res = fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.2f}s) {res}", flush=True)
        except Exception as exc:
            failed = True
            print(f"FAIL {name} ({time.perf_counter() - t0:.2f}s) {exc}", flush=True)
    if failed:
        sys.exit(1)
    print("all 5 robust contour SLAM assertions passed", flush=True)


if __name__ == "__main__":
    main()
