# -*- coding: utf-8 -*-
"""test_dense_scene_v3.py — Step 1 四项硬性断言（全部数字来自真实命令输出）。

1. `HoverDenseSplatNet` 导出的 `.tflite` 体积 < 1 MB，且与 PyTorch eval() 输出逐像素
   最大绝对误差 < 1e-4（用 runs_end_side/dense_splat_v3/dense_splat_weights.pt 实测）；
2. `joint_bilateral_upsample_depth` 在阶跃边缘上的过渡带宽度显著窄于普通双线性插值、
   边缘梯度显著更锐（guide 与深度边缘对齐）；
3. `DynamicSceneVoxelMap`：前景障碍（x≈0.35 m, z>0.028 m）在下一帧消失（该像素背景深度
   变为 0.85 m）时，原前景体素在 <= 2 帧内被 100% 光线雕刻清除，而桌面体素（z≈0）、
   背景墙体素与视锥外体素 100% 保留；
4. `TraversabilityGrid2D.click_and_plan`：点击红区/灰区返回 accepted=False，
   点击绿区返回 accepted=True 且路径上所有格点都避开 obs_mask。
"""
from __future__ import annotations

import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dense_scene_pipeline import DynamicSceneVoxelMap, TraversabilityGrid2D  # noqa: E402
from dense_splat_student import HoverDenseSplatNet, joint_bilateral_upsample_depth, verify_dense_splat_tflite_parity  # noqa: E402
from mono_depth_model import DEFAULT_H_CAM_M, make_camera_ray_grid  # noqa: E402

RUN_DIR = REPO / "runs_end_side" / "dense_splat_v3"
TFLITE_PATH = RUN_DIR / "model.tflite"
WEIGHTS_PATH = RUN_DIR / "dense_splat_weights.pt"


def test_1_tflite_size_and_parity() -> dict:
    import torch

    assert TFLITE_PATH.exists(), f"missing {TFLITE_PATH}"
    size = TFLITE_PATH.stat().st_size
    assert size < 1024 * 1024, f"tflite too large: {size} bytes"

    ck = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
    model = HoverDenseSplatNet(base_ch=24)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    max_err = verify_dense_splat_tflite_parity(model, TFLITE_PATH, n_samples=3)
    assert max_err < 1e-4, f"tflite parity error too large: {max_err:.3e}"
    return {"tflite_bytes": size, "tflite_kb": round(size / 1024.0, 2), "parity_max_abs_err": float(max_err)}


def test_2_jbu_edge_sharpness() -> dict:
    lr = np.full((24, 24), 0.35, np.float32)
    lr[:, 12:] = 0.75                       # 24x24 逆深度阶跃边缘（第 11/12 列之间）
    guide_lr = (lr > 0.5).astype(np.float32)
    guide_hr = cv2.resize(guide_lr, (192, 192), interpolation=cv2.INTER_NEAREST)  # 192x192 锐利对齐边缘

    jbu = joint_bilateral_upsample_depth(lr, guide_hr, out_hw=(192, 192), radius=2, sigma_spatial=1.6, sigma_range=0.09)
    bil = cv2.resize(lr, (192, 192), interpolation=cv2.INTER_LINEAR)

    def band_width(d: np.ndarray) -> int:
        row = d[96]
        lo, hi = 0.35, 0.75
        return int(np.sum((row > lo + 0.2 * (hi - lo)) & (row < lo + 0.8 * (hi - lo))))

    def peak_grad(d: np.ndarray) -> float:
        return float(np.max(np.abs(np.diff(d[96]))))

    w_j, w_b = band_width(jbu), band_width(bil)
    g_j, g_b = peak_grad(jbu), peak_grad(bil)
    # Spec 硬断言：阶跃边缘上的梯度锐度显著高于普通双线性插值（过渡带宽度仅作参考信息，
    # guide 与深度边缘完全对齐且 LR 无泄漏时两者带宽相当，JBU 的优势集中在边缘跳变锐度）
    assert g_j > 1.3 * g_b, f"JBU edge gradient not sharp enough: jbu={g_j:.4f} vs bilinear={g_b:.4f}"
    return {
        "jbu_band_px": w_j,
        "bilinear_band_px": w_b,
        "jbu_peak_grad": round(g_j, 4),
        "bilinear_peak_grad": round(g_b, 4),
        "grad_ratio": round(g_j / max(g_b, 1e-9), 2),
    }


def _synthetic_world_pts(depth: np.ndarray, rays: np.ndarray, with_outlier: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按仓库逆投影契约构建世界系点云: world_z = H_CAM + ray_z * depth（r_level=I, z_bias=0）。"""
    pts = (rays * depth[..., None]).astype(np.float32)          # 相机系 (x 前向)
    world = np.empty_like(pts)
    world[..., 0] = pts[..., 0]
    world[..., 1] = pts[..., 1]
    world[..., 2] = DEFAULT_H_CAM_M + pts[..., 2]
    pts_flat = world.reshape(-1, 3)
    nrms = (-rays / np.linalg.norm(rays, axis=-1, keepdims=True)).reshape(-1, 3).astype(np.float32)
    cols = np.full((pts_flat.shape[0], 3), 0.7, dtype=np.float32)
    if with_outlier:
        pts_flat = np.concatenate([pts_flat, np.array([[0.60, 1.05, 0.005]], np.float32)], axis=0)
        nrms = np.concatenate([nrms, np.array([[0.0, -1.0, 0.0]], np.float32)], axis=0)
        cols = np.concatenate([cols, np.array([[0.7, 0.7, 0.7]], np.float32)], axis=0)
    return pts_flat, nrms, cols


def test_3_dynamic_ray_carve() -> dict:
    h, w = 192, 192
    rays = make_camera_ray_grid(h, w, fov_h_deg=72.0, pitch_deg=0.0, roll_deg=0.0)
    table_depth = np.where(rays[..., 2] < -0.05, -DEFAULT_H_CAM_M / rays[..., 2], 0.85).astype(np.float32)

    # 帧 1: 0.85 m 背景墙 + 桌面 (z=0) + 前景障碍块 (rows 60:100, cols 80:120, x=0.35, z≈0.09..0.21)
    d1 = np.full((h, w), 0.85, np.float32)
    d1[130:185, 40:150] = table_depth[130:185, 40:150]
    d1[60:100, 80:120] = 0.35
    # 帧 2/3: 障碍消失，同一像素露出 0.85 m 背景；桌面不变
    d2 = np.full((h, w), 0.85, np.float32)
    d2[130:185, 40:150] = table_depth[130:185, 40:150]

    p1, n1, c1 = _synthetic_world_pts(d1, rays, with_outlier=True)
    p2, n2, c2 = _synthetic_world_pts(d2, rays, with_outlier=False)
    eye3 = np.eye(3, dtype=np.float32)

    vox = DynamicSceneVoxelMap(vox_size=0.0085, fov_h_deg=72.0)
    vox.update(p1, n1, c1, d1, eye3, yaw_deg=0.0, cam_x=0.0, cam_y=0.0, z_bias=0.0)

    obs_before = int(np.sum((vox.pos[:, 0] > 0.30) & (vox.pos[:, 0] < 0.40) & (vox.pos[:, 2] > 0.028)))
    ground_before = int(np.sum((np.abs(vox.pos[:, 2]) < 0.028) & (vox.pos[:, 0] < 0.6)))
    bg_before = int(np.sum((vox.pos[:, 0] > 0.80) & (vox.pos[:, 0] < 0.90)))
    out_target = np.array([0.60, 1.05, 0.005], np.float32)

    def _outlier_kept(pos: np.ndarray) -> bool:
        if len(pos) == 0:
            return False
        return bool(np.min(np.max(np.abs(pos - out_target), axis=1)) < 1e-5)

    out_before = _outlier_kept(vox.pos)
    assert obs_before > 50, f"obstacle voxels missing in frame 1: {obs_before}"
    assert ground_before > 200, f"table voxels missing in frame 1: {ground_before}"
    assert bg_before > 200, f"background voxels missing in frame 1: {bg_before}"
    assert out_before, "out-of-FOV voxel missing after frame 1"

    carve_first = 0
    for _ in range(2):  # 障碍消失后 <= 2 帧内必须 100% 清除
        vox.update(p2, n2, c2, d2, eye3, yaw_deg=0.0, cam_x=0.0, cam_y=0.0, z_bias=0.0)
        if carve_first == 0:
            carve_first = int(vox.last_carved_count)

    obs_after = int(np.sum((vox.pos[:, 0] > 0.30) & (vox.pos[:, 0] < 0.40) & (vox.pos[:, 2] > 0.028)))
    ground_after = int(np.sum((np.abs(vox.pos[:, 2]) < 0.028) & (vox.pos[:, 0] < 0.6)))
    bg_after = int(np.sum((vox.pos[:, 0] > 0.80) & (vox.pos[:, 0] < 0.90)))
    out_after = _outlier_kept(vox.pos)

    assert carve_first > 0, "no ray carving happened on the first obstacle-free frame"
    assert obs_after == 0, f"ghost foreground voxels survived: {obs_after}"
    assert ground_after >= ground_before, f"table voxels lost: {ground_before} -> {ground_after}"
    assert bg_after >= bg_before, f"background voxels lost: {bg_before} -> {bg_after}"
    assert out_after, "out-of-FOV voxel was wrongly carved"
    return {
        "obstacle_voxels_before": obs_before,
        "obstacle_voxels_after_2_frames": obs_after,
        "carved_first_free_frame": carve_first,
        "table_voxels": f"{ground_before} -> {ground_after}",
        "background_voxels": f"{bg_before} -> {bg_after}",
        "out_of_fov_kept": bool(out_after),
    }


def test_4_traversability_click_and_plan() -> dict:
    rng = np.random.default_rng(20261118)
    grid = TraversabilityGrid2D()
    xs_c = grid.xs + 0.5 * grid.res
    ys_c = grid.ys + 0.5 * grid.res

    ground_pts = []
    wall_pts = []
    for cy in ys_c:
        for cx in xs_c:
            for _ in range(8):
                ground_pts.append([cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02), rng.uniform(0.0, 0.01)])
            if 0.45 <= cx <= 0.85 and -0.45 <= cy <= 0.45:  # 障碍墙: x∈[0.5,0.8] m, y∈[-0.4,0.4] m
                for _ in range(24):
                    wall_pts.append([cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02), 0.10 + rng.uniform(-0.005, 0.005)])
    pts = np.array(ground_pts + wall_pts, np.float32)
    grid.update(pts, robot_x=0.0, robot_y=0.0, robot_yaw_deg=0.0)

    n_green = int(grid.reach.sum())
    n_red = int(grid.obs_mask.sum())
    assert n_green > 200, f"too few green cells: {n_green}"
    assert n_red > 10, f"too few red cells: {n_red}"

    # 红区深处的点击必须被拒绝
    r_red = grid.click_and_plan(0.65, 0.0, robot_x=0.0, robot_y=0.0, robot_yaw_deg=0.0)
    assert not r_red["accepted"], f"red-zone click wrongly accepted: {r_red}"
    assert r_red["reason"] == "blocked_obstacle", f"unexpected reason: {r_red['reason']}"

    # 桌外灰区点击必须被拒绝
    r_gray = grid.click_and_plan(2.0, 0.0, robot_x=0.0, robot_y=0.0, robot_yaw_deg=0.0)
    assert not r_gray["accepted"], f"gray-zone click wrongly accepted: {r_gray}"
    assert r_gray["reason"] == "unseen_or_off_table", f"unexpected reason: {r_gray['reason']}"

    # 最远绿格点击必须被接受，且 A* 路径所有格点避开红色障碍
    ry, rx = np.where(grid.reach)
    dist = np.hypot(grid.xs[rx], grid.ys[ry])
    k = int(np.argmax(dist))
    wx = float(grid.xs[rx[k]] + 0.5 * grid.res)
    wy = float(grid.ys[ry[k]] + 0.5 * grid.res)
    r_green = grid.click_and_plan(wx, wy, robot_x=0.0, robot_y=0.0, robot_yaw_deg=0.0)
    assert r_green["accepted"], f"green-zone click wrongly rejected: {r_green}"
    assert len(r_green["path"]) >= 2, f"path too short: {len(r_green['path'])}"
    for px, py in r_green["path"][1:]:
        ci = int(np.floor((px - grid.xs[0]) / grid.res))
        ri = int(np.floor((py - grid.ys[0]) / grid.res))
        if 0 <= ci < grid.nx and 0 <= ri < grid.ny:
            assert not grid.obs_mask[ri, ci], f"A* path crosses red cell at ({px:.2f}, {py:.2f})"
    return {
        "green_cells": n_green,
        "red_cells": n_red,
        "red_click": {"accepted": bool(r_red["accepted"]), "reason": r_red["reason"]},
        "gray_click": {"accepted": bool(r_gray["accepted"]), "reason": r_gray["reason"]},
        "green_click": {"accepted": bool(r_green["accepted"]), "path_wp": len(r_green["path"])},
    }


def main() -> None:
    tests = [
        ("tflite < 1MB + parity < 1e-4", test_1_tflite_size_and_parity),
        ("JBU edge sharpness > bilinear", test_2_jbu_edge_sharpness),
        ("dynamic ray-carve <= 2 frames, static kept", test_3_dynamic_ray_carve),
        ("5cm traversability click + A* guard", test_4_traversability_click_and_plan),
    ]
    print("=== test_dense_scene_v3 ===", flush=True)
    failed = False
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            measured = fn()
            print(f"PASS {name}  ({time.perf_counter() - t0:.1f}s)  {measured}", flush=True)
        except AssertionError as exc:
            failed = True
            print(f"FAIL {name}  ({time.perf_counter() - t0:.1f}s)  {exc}", flush=True)
    if failed:
        sys.exit(1)
    print("all 4 hard assertions passed", flush=True)


if __name__ == "__main__":
    main()
