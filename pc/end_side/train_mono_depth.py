# -*- coding: utf-8 -*-
"""train_mono_depth.py — RIG-Hover 单目尺度深度与障碍物高程网络混合 3D 光线追踪 + 真机视频适应训练管线。

核心流程：
1. 解析光线追踪 3D 桌面场景生成器 (`render_analytical_desktop_scene`)：
   - 相机位于 `(0, 0, h_cam)`，`h_cam ~ U(DEFAULT_H_CAM_M − 0.02, DEFAULT_H_CAM_M + 0.02) m`（实测相机高度 ±2 cm），
     `pitch ~ U(-12°, +12°)`，`roll ~ U(-6°, +6°)`，视场角 = `DEFAULT_FOV_H_DEG`（两者取 `pc/robot_truth.json` 的实测值；
     已发布的 `hover_mono_depth_v1` 是用设计初值 0.115 m / 60° 训练的，记录在它的 `model.json` 里）；
   - 带木纹/网格/桌垫/高光与接触软阴影的桌面平面 `Z = 0` + 远端背景墙 `1.6 .. 2.6 m`；
   - 1~5 个随机桌面三维前景目标：圆柱体（水杯/瓶子 `r=0.03..0.08m, h=0.08..0.20m`）、旋转长方体（书本/积木）、
     球体以及人体躯干/头部/伸手握拳三维轮廓（`0.25 .. 1.80 m`）；
   - 传感器域随机化：曝光、对比度、暗角、运动模糊、高斯噪声，以及 40% 概率转为 GC0308 同分布三通道灰度 `[g, g, g]`。
2. 真机视频帧几何-边缘感知伪真值适应 (`sample_real_robot_frames`)：
   - 使用 `cv2.VideoCapture` 从 `dataset/takes/*/take.mp4` 按段无泄漏采样 ~240 帧真实机器人视角图像（5 段训练 / 1 段留出）；
   - 结合地平线透视桌面模型、时序背景差分与 Sobel 边缘连通域构造几何一致伪真值；
   - 在 CUDA 上按 `80%` 解析 3D 光线追踪场景 + `20%` 真机视频帧混合训练 `HoverMonoDepthNet`。
3. 导出 `hover_mono_depth_v1.tflite`、`hover_mono_depth_v1.json`、`hover-mono-depth-v1-bundle.zip` 与 `depth_eval_metrics.json`，
   同步拷贝至 Android `assets/models/`，并渲染深色主题四联展示图 `fig6_mono_depth_and_3d_splats.png`。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mono_depth_model import (  # noqa: E402
    DEFAULT_FOV_H_DEG,
    DEFAULT_H_CAM_M,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    ELEV_THRESHOLD_M,
    INPUT_C,
    INPUT_H,
    INPUT_W,
    OUTPUT_H,
    OUTPUT_W,
    HoverMonoDepthNet,
    benchmark_cpu_latency_ms,
    export_mono_depth_bundle,
    infer_mono_depth_tflite,
    inv_depth_01_to_metric_m,
    make_camera_ray_grid,
    metric_m_to_inv_depth_01,
    unproject_depth_to_3d_splats,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220


# ---------------------------------------------------------------------------
# 1. 解析光线追踪 3D 桌面场景生成器 (Vectorized Raycaster)
# ---------------------------------------------------------------------------

def _intersect_cylinder(
    ox: float,
    oy: float,
    oz: float,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """解析求解射线 `P(t) = O + t*V` 与立在桌面 `Z in [0, h]` 上的有盖圆柱体交点距离 `t` 及外法向。"""
    inf = np.float32(1e9)
    t_best = np.full(vx.shape, inf, dtype=np.float32)
    nx = np.zeros_like(vx, dtype=np.float32)
    ny = np.zeros_like(vx, dtype=np.float32)
    nz = np.zeros_like(vx, dtype=np.float32)

    dx = np.float32(ox - cx)
    dy = np.float32(oy - cy)
    a = vx * vx + vy * vy
    b = dx * vx + dy * vy
    c = np.float32(dx * dx + dy * dy - r * r)
    disc = b * b - a * c

    valid_q = (a > 1e-7) & (disc >= 0.0)
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    inv_a = np.where(a > 1e-7, 1.0 / np.maximum(a, 1e-7), 0.0).astype(np.float32)

    for sign in (-1.0, 1.0):
        t_cand = (-b + np.float32(sign) * sqrt_d) * inv_a
        z_cand = oz + t_cand * vz
        hit = valid_q & (t_cand > 0.08) & (z_cand >= 0.0) & (z_cand <= h) & (t_cand < t_best)
        if np.any(hit):
            t_best = np.where(hit, t_cand, t_best)
            hx = ox + t_cand * vx - cx
            hy = oy + t_cand * vy - cy
            inv_r = np.float32(1.0 / max(r, 1e-5))
            nx = np.where(hit, hx * inv_r, nx)
            ny = np.where(hit, hy * inv_r, ny)
            nz = np.where(hit, 0.0, nz)

    # 顶盖圆盘 Z = h
    valid_vz = np.abs(vz) > 1e-6
    t_cap = np.where(valid_vz, (h - oz) / np.where(valid_vz, vz, -1.0), inf).astype(np.float32)
    xc = ox + t_cap * vx - cx
    yc = oy + t_cap * vy - cy
    hit_cap = valid_vz & (t_cap > 0.08) & ((xc * xc + yc * yc) <= r * r) & (t_cap < t_best)
    if np.any(hit_cap):
        t_best = np.where(hit_cap, t_cap, t_best)
        nx = np.where(hit_cap, 0.0, nx)
        ny = np.where(hit_cap, 0.0, ny)
        nz = np.where(hit_cap, 1.0, nz)

    return t_best, np.stack([nx, ny, nz], axis=-1)


def _intersect_obb(
    ox: float,
    oy: float,
    oz: float,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cx: float,
    cy: float,
    z0: float,
    lx: float,
    ly: float,
    lz: float,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """解析求解射线与绕 Z 轴旋转 `yaw` 的有向长方体 `[cx±lx, cy±ly, z0..z0+2*lz]` 的交点与法向。"""
    inf = np.float32(1e9)
    cyaw = math.cos(yaw)
    syaw = math.sin(yaw)

    # 变换至局部坐标系（中心在 cx, cy, z0 + lz）
    zc = z0 + lz
    dx0 = ox - cx
    dy0 = oy - cy
    dz0 = oz - zc

    lox = np.float32(cyaw * dx0 + syaw * dy0)
    loy = np.float32(-syaw * dx0 + cyaw * dy0)
    loz = np.float32(dz0)

    lvx = (cyaw * vx + syaw * vy).astype(np.float32)
    lvy = (-syaw * vx + cyaw * vy).astype(np.float32)
    lvz = vz.astype(np.float32)

    def _slab(o_s: float, v_arr: np.ndarray, half_ext: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        safe_v = np.where(np.abs(v_arr) > 1e-7, v_arr, np.where(v_arr >= 0, 1e-7, -1e-7))
        t1 = (-half_ext - o_s) / safe_v
        t2 = (half_ext - o_s) / safe_v
        tmin = np.minimum(t1, t2)
        tmax = np.maximum(t1, t2)
        sgn = np.where(t1 < t2, -1.0, 1.0).astype(np.float32)
        return tmin.astype(np.float32), tmax.astype(np.float32), sgn

    tx0, tx1, sx = _slab(lox, lvx, lx)
    ty0, ty1, sy = _slab(loy, lvy, ly)
    tz0, tz1, sz = _slab(loz, lvz, lz)

    t_near = np.maximum(np.maximum(tx0, ty0), tz0)
    t_far = np.minimum(np.minimum(tx1, ty1), tz1)
    hit = (t_far >= t_near) & (t_near > 0.08)
    t_out = np.where(hit, t_near, inf).astype(np.float32)

    is_x = (tx0 >= ty0) & (tx0 >= tz0)
    is_y = (~is_x) & (ty0 >= tz0)
    lnx = np.where(is_x, sx, 0.0).astype(np.float32)
    lny = np.where(is_y, sy, 0.0).astype(np.float32)
    lnz = np.where((~is_x) & (~is_y), sz, 0.0).astype(np.float32)

    wnx = (cyaw * lnx - syaw * lny).astype(np.float32)
    wny = (syaw * lnx + cyaw * lny).astype(np.float32)
    wnz = lnz
    return t_out, np.stack([wnx, wny, wnz], axis=-1)


def _intersect_ellipsoid(
    ox: float,
    oy: float,
    oz: float,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    rx: float,
    ry: float,
    rz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """解析求解射线与轴对齐椭球 / 球体的交点距离与单位外法向。"""
    inf = np.float32(1e9)
    irx, iry, irz = 1.0 / rx, 1.0 / ry, 1.0 / rz
    dx = np.float32((ox - cx) * irx)
    dy = np.float32((oy - cy) * iry)
    dz = np.float32((oz - cz) * irz)

    svx = (vx * irx).astype(np.float32)
    svy = (vy * iry).astype(np.float32)
    svz = (vz * irz).astype(np.float32)

    a = svx * svx + svy * svy + svz * svz
    b = dx * svx + dy * svy + dz * svz
    c = np.float32(dx * dx + dy * dy + dz * dz - 1.0)
    disc = b * b - a * c

    valid = (a > 1e-7) & (disc >= 0.0)
    t_cand = (-b - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(a, 1e-7)
    hit = valid & (t_cand > 0.08)
    t_out = np.where(hit, t_cand, inf).astype(np.float32)

    hx = (ox + t_cand * vx - cx) * (irx * irx)
    hy = (oy + t_cand * vy - cy) * (iry * iry)
    hz = (oz + t_cand * vz - cz) * (irz * irz)
    n_norm = np.sqrt(np.maximum(hx * hx + hy * hy + hz * hz, 1e-10))
    n_vec = np.stack([hx / n_norm, hy / n_norm, hz / n_norm], axis=-1).astype(np.float32)
    return t_out, n_vec


def render_analytical_desktop_scene(
    rng: np.random.Generator,
    force_grayscale: bool | None = None,
    fixed_pose: tuple[float, float, float] | None = None,
    custom_objects: list[dict[str, Any]] | None = None,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
) -> dict[str, Any]:
    """生成单帧 `96x96` 光线追踪桌面 RGB 图像与精确 `48x48` 物理深度 (`depth_m`) 及高程掩膜 (`elev_mask`)。

    `fov_h_deg` 是这台合成相机的主视场角。评测一个已训模型时要传它 `model.json` 里记录的相机
    （`camera.fovHDeg` / `camera.heightM`）：在别的视场角下渲染的画面不是它训练时见过的画面。
    """
    if fixed_pose is not None:
        h_cam, pitch_deg, roll_deg = float(fixed_pose[0]), float(fixed_pose[1]), float(fixed_pose[2])
    else:
        h_cam = float(rng.uniform(DEFAULT_H_CAM_M - 0.02, DEFAULT_H_CAM_M + 0.02))
        pitch_deg = float(rng.uniform(-12.0, 12.0))
        roll_deg = float(rng.uniform(-6.0, 6.0))

    d_wall = float(rng.uniform(1.60, 2.60))

    # 随机采样 1 ~ 5 个桌面前景三维几何体
    if custom_objects is not None:
        objects = list(custom_objects)
    else:
        n_objs = int(rng.integers(1, 6))
        objects = []
        placed_xy: list[tuple[float, float, float]] = []
        for _ in range(n_objs * 4):
            if len(objects) >= n_objs:
                break
            kind = str(rng.choice(["cylinder", "box", "sphere", "human_hand"], p=[0.32, 0.28, 0.18, 0.22]))
            if kind == "human_hand":
                cx = float(rng.uniform(0.55, min(1.75, d_wall - 0.12)))
                max_y = max(0.08, 0.42 * cx)
                cy = float(rng.uniform(-max_y, max_y))
                r_foot = 0.13
            else:
                cx = float(rng.uniform(0.26, min(1.58, d_wall - 0.15)))
                max_y = max(0.05, 0.45 * cx)
                cy = float(rng.uniform(-max_y, max_y))
                r_foot = float(rng.uniform(0.032, 0.080))

            if any(math.hypot(cx - px, cy - py) < (r_foot + pr + 0.035) for px, py, pr in placed_xy):
                continue
            placed_xy.append((cx, cy, r_foot))

            color = rng.uniform(0.14, 0.88, size=(3,)).astype(np.float32)
            if kind == "cylinder":
                objects.append(
                    {
                        "kind": "cylinder",
                        "cx": cx,
                        "cy": cy,
                        "r": r_foot,
                        "h": float(rng.uniform(0.08, 0.20)),
                        "color": color,
                        "stripe": float(rng.uniform(0.0, 1.0)),
                    }
                )
            elif kind == "box":
                # 15% 概率采样低于 3cm 的薄书/垫板作为负样本边界训练，85% 为凸起立块
                h_box = float(rng.uniform(0.012, 0.025)) if rng.random() < 0.15 else float(rng.uniform(0.042, 0.165))
                objects.append(
                    {
                        "kind": "box",
                        "cx": cx,
                        "cy": cy,
                        "lx": float(rng.uniform(0.030, 0.090)),
                        "ly": float(rng.uniform(0.030, 0.090)),
                        "h": h_box,
                        "yaw": float(rng.uniform(-math.pi, math.pi)),
                        "color": color,
                    }
                )
            elif kind == "sphere":
                objects.append(
                    {
                        "kind": "sphere",
                        "cx": cx,
                        "cy": cy,
                        "r": float(rng.uniform(0.032, 0.078)),
                        "color": color,
                    }
                )
            else:
                skin = np.array(
                    [rng.uniform(0.62, 0.88), rng.uniform(0.48, 0.72), rng.uniform(0.38, 0.62)],
                    dtype=np.float32,
                )
                shirt = rng.uniform(0.12, 0.68, size=(3,)).astype(np.float32)
                reach_x = max(0.26, cx - float(rng.uniform(0.22, 0.45)))
                reach_y = cy + float(rng.uniform(-0.10, 0.10))
                reach_z = float(rng.uniform(0.038, 0.14))
                objects.append(
                    {
                        "kind": "human_hand",
                        "cx": cx,
                        "cy": cy,
                        "reach_x": reach_x,
                        "reach_y": reach_y,
                        "reach_z": reach_z,
                        "skin": skin,
                        "shirt": shirt,
                    }
                )

    def _trace_grid(h_px: int, w_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rays = make_camera_ray_grid(
            height=h_px,
            width=w_px,
            fov_h_deg=fov_h_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        vx, vy, vz = rays[..., 0], rays[..., 1], rays[..., 2]
        inf = np.float32(1e9)

        # 桌面 Z = 0 与远端背景墙 X = d_wall
        valid_tbl = vz < -1e-4
        t_tbl = np.where(valid_tbl, -h_cam / np.where(valid_tbl, vz, -1.0), inf).astype(np.float32)
        t_tbl = np.where(t_tbl > 0.06, t_tbl, inf)

        valid_wall = vx > 1e-4
        t_wall = np.where(valid_wall, d_wall / np.where(valid_wall, vx, 1.0), inf).astype(np.float32)

        is_tbl = t_tbl < t_wall
        t_min = np.where(is_tbl, t_tbl, t_wall).astype(np.float32)
        normals = np.zeros((h_px, w_px, 3), dtype=np.float32)
        normals[..., 0] = np.where(is_tbl, 0.0, -1.0)
        normals[..., 2] = np.where(is_tbl, 1.0, 0.0)
        # surf_type: 0=tabletop, 1=wall, >=2=foreground object
        surf_id = np.where(is_tbl, 0, 1).astype(np.int32)
        albedo = np.zeros((h_px, w_px, 3), dtype=np.float32)

        for idx, obj in enumerate(objects):
            oid = idx + 2
            k = obj["kind"]
            if k == "cylinder":
                t_o, n_o = _intersect_cylinder(0.0, 0.0, h_cam, vx, vy, vz, obj["cx"], obj["cy"], obj["r"], obj["h"])
                upd = t_o < t_min
                if np.any(upd):
                    t_min = np.where(upd, t_o, t_min)
                    normals = np.where(upd[..., None], n_o, normals)
                    surf_id = np.where(upd, oid, surf_id)
                    z_h = h_cam + t_o * vz
                    band = ((z_h / max(obj["h"], 1e-3) > 0.35) & (z_h / max(obj["h"], 1e-3) < 0.68)).astype(np.float32)
                    c_obj = obj["color"][None, None, :] * (1.0 - 0.25 * obj["stripe"] * band[..., None]) + 0.22 * obj["stripe"] * band[..., None]
                    albedo = np.where(upd[..., None], c_obj, albedo)

            elif k == "box":
                t_o, n_o = _intersect_obb(
                    0.0,
                    0.0,
                    h_cam,
                    vx,
                    vy,
                    vz,
                    obj["cx"],
                    obj["cy"],
                    0.0,
                    obj["lx"],
                    obj["ly"],
                    0.5 * obj["h"],
                    obj["yaw"],
                )
                upd = t_o < t_min
                if np.any(upd):
                    t_min = np.where(upd, t_o, t_min)
                    normals = np.where(upd[..., None], n_o, normals)
                    surf_id = np.where(upd, oid, surf_id)
                    albedo = np.where(upd[..., None], obj["color"][None, None, :], albedo)

            elif k == "sphere":
                r_s = obj["r"]
                t_o, n_o = _intersect_ellipsoid(0.0, 0.0, h_cam, vx, vy, vz, obj["cx"], obj["cy"], r_s, r_s, r_s, r_s)
                upd = t_o < t_min
                if np.any(upd):
                    t_min = np.where(upd, t_o, t_min)
                    normals = np.where(upd[..., None], n_o, normals)
                    surf_id = np.where(upd, oid, surf_id)
                    albedo = np.where(upd[..., None], obj["color"][None, None, :], albedo)

            elif k == "human_hand":
                # 躯干 + 头部 + 伸向桌面的前臂与握拳手部
                parts = [
                    (obj["cx"], obj["cy"], 0.14, 0.11, 0.20, 0.20, obj["shirt"]),
                    (obj["cx"] - 0.02, obj["cy"], 0.38, 0.085, 0.085, 0.105, obj["skin"]),
                    (
                        0.5 * (obj["cx"] + obj["reach_x"]),
                        0.5 * (obj["cy"] + obj["reach_y"]),
                        0.5 * (0.15 + obj["reach_z"]),
                        0.5 * abs(obj["cx"] - obj["reach_x"]) + 0.04,
                        0.042,
                        0.042,
                        obj["skin"],
                    ),
                    (obj["reach_x"], obj["reach_y"], obj["reach_z"], 0.046, 0.045, 0.040, obj["skin"]),
                ]
                for pcx, pcy, pcz, prx, pry, prz, pcol in parts:
                    t_o, n_o = _intersect_ellipsoid(0.0, 0.0, h_cam, vx, vy, vz, pcx, pcy, pcz, prx, pry, prz)
                    upd = t_o < t_min
                    if np.any(upd):
                        t_min = np.where(upd, t_o, t_min)
                        normals = np.where(upd[..., None], n_o, normals)
                        surf_id = np.where(upd, oid, surf_id)
                        albedo = np.where(upd[..., None], pcol[None, None, :], albedo)

        xyz_hit = np.stack([t_min * vx, t_min * vy, h_cam + t_min * vz], axis=-1).astype(np.float32)
        return t_min, xyz_hit, normals, surf_id, albedo

    # 精确计算 48x48 中心射线 GT（零飞点遮挡边缘）与 96x96 渲染外观
    t_48, xyz_48, _, surf_48, _ = _trace_grid(OUTPUT_H, OUTPUT_W)
    depth_gt_48 = np.clip(t_48, DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)
    elev_gt_48 = ((surf_48 >= 2) & (xyz_48[..., 2] > ELEV_THRESHOLD_M)).astype(np.float32)

    t_96, xyz_96, norm_96, surf_96, albedo_96 = _trace_grid(INPUT_H, INPUT_W)
    depth_gt_96 = np.clip(t_96, DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)
    elev_gt_96 = ((surf_96 >= 2) & (xyz_96[..., 2] > ELEV_THRESHOLD_M)).astype(np.float32)

    # 桌面程序化纹理 (surf_96 == 0) 与背景墙纹理 (surf_96 == 1)
    xh = xyz_96[..., 0]
    yh = xyz_96[..., 1]
    zh = xyz_96[..., 2]

    tex_mode = int(rng.integers(0, 4))
    tbl_base = rng.uniform(0.38, 0.84, size=(3,)).astype(np.float32)
    if tex_mode == 0:
        # 木纹桌面
        freq = float(rng.uniform(18.0, 38.0))
        grain = 0.5 + 0.5 * np.sin(yh * freq + 2.2 * np.sin(xh * 5.5))
        seam = (np.abs((yh * 7.5) % 1.0 - 0.5) < 0.025).astype(np.float32)
        wood_tint = np.array([1.08, 0.94, 0.76], dtype=np.float32) * tbl_base
        tbl_col = wood_tint[None, None, :] * (0.86 + 0.18 * grain[..., None]) * (1.0 - 0.18 * seam[..., None])
    elif tex_mode == 1:
        # 网格实验台面
        step = float(rng.uniform(0.09, 0.20))
        gx = (np.abs((xh / step) % 1.0 - 0.5) > 0.44).astype(np.float32)
        gy = (np.abs((yh / step) % 1.0 - 0.5) > 0.44).astype(np.float32)
        line = np.maximum(gx, gy)[..., None]
        tbl_col = tbl_base[None, None, :] * (1.0 - 0.22 * line) + 0.08 * line
    elif tex_mode == 2:
        # 桌面垫 + 细微颗粒纹理
        mat_mask = ((xh > 0.34) & (xh < 1.05) & (np.abs(yh) < 0.34)).astype(np.float32)[..., None]
        mat_col = rng.uniform(0.18, 0.52, size=(3,)).astype(np.float32)
        tbl_col = tbl_base[None, None, :] * (1.0 - mat_mask) + mat_col[None, None, :] * mat_mask
    else:
        # 哑光渐变桌面
        grad = np.clip(1.05 - 0.14 * xh, 0.72, 1.12)[..., None]
        tbl_col = tbl_base[None, None, :] * grad

    wall_base = rng.uniform(0.24, 0.72, size=(3,)).astype(np.float32)
    wall_panel = 0.92 + 0.08 * np.sin(yh * float(rng.uniform(8.0, 18.0)))
    wall_ao = 0.72 + 0.28 * np.clip(zh / 0.35, 0.0, 1.0)
    wall_col = wall_base[None, None, :] * (wall_panel * wall_ao)[..., None]

    albedo_96 = np.where((surf_96 == 0)[..., None], tbl_col, albedo_96)
    albedo_96 = np.where((surf_96 == 1)[..., None], wall_col, albedo_96)

    # 光照与桌面接触软阴影
    l_az = float(rng.uniform(-1.1, 1.1))
    l_el = float(rng.uniform(0.65, 1.32))
    l_dir = np.array(
        [-math.cos(l_el) * math.cos(l_az), math.cos(l_el) * math.sin(l_az), math.sin(l_el)],
        dtype=np.float32,
    )
    l_dir /= max(float(np.linalg.norm(l_dir)), 1e-6)

    ndotl = np.clip(np.sum(norm_96 * l_dir[None, None, :], axis=-1), 0.0, 1.0)
    shadow = np.ones((INPUT_H, INPUT_W), dtype=np.float32)
    tbl_mask = surf_96 == 0
    if np.any(tbl_mask):
        sh_x = xh - 0.03 * l_dir[0]
        sh_y = yh - 0.03 * l_dir[1]
        for obj in objects:
            if obj["kind"] == "human_hand":
                ocx, ocy, orad = obj["reach_x"], obj["reach_y"], 0.075
            elif obj["kind"] == "box":
                ocx, ocy, orad = obj["cx"], obj["cy"], 0.85 * (obj["lx"] + obj["ly"])
            else:
                ocx, ocy, orad = obj["cx"], obj["cy"], 1.25 * obj["r"]
            d2 = (sh_x - ocx) ** 2 + (sh_y - ocy) ** 2
            shadow *= 1.0 - 0.42 * np.exp(-d2 / max(2.0 * orad * orad, 1e-4))
        # 桌面与背景墙交界环境光遮蔽，强化地平线物理视觉线索
        wall_dist = np.maximum(d_wall - xh, 0.0)
        shadow *= 1.0 - 0.24 * np.exp(-(wall_dist * wall_dist) / (2.0 * 0.08 * 0.08))
    shadow = np.where(tbl_mask, shadow, 1.0).astype(np.float32)

    ambient = float(rng.uniform(0.34, 0.50))
    diffuse = (1.0 - ambient) * ndotl * shadow
    # 桌面高光瓣
    view_dir = -xyz_96 / np.maximum(np.linalg.norm(xyz_96, axis=-1, keepdims=True), 1e-6)
    half_v = l_dir[None, None, :] + view_dir
    half_v /= np.maximum(np.linalg.norm(half_v, axis=-1, keepdims=True), 1e-6)
    spec = (
        float(rng.uniform(0.04, 0.16))
        * (np.clip(np.sum(norm_96 * half_v, axis=-1), 0.0, 1.0) ** 24)
        * shadow
    )

    rgb = albedo_96 * (ambient + diffuse)[..., None] + spec[..., None]

    # 域随机化：曝光、对比度、镜头暗角、轻度模糊、传感器噪声、40% 概率 GC0308 三通道灰度
    uu = np.linspace(-1.0, 1.0, INPUT_W, dtype=np.float32)
    vv = np.linspace(-1.0, 1.0, INPUT_H, dtype=np.float32)
    r2_vig = uu[None, :] ** 2 + vv[:, None] ** 2
    vig = 1.0 - float(rng.uniform(0.04, 0.15)) * r2_vig
    rgb = rgb * vig[..., None] * float(rng.uniform(0.82, 1.18))

    contrast = float(rng.uniform(0.85, 1.18))
    rgb = (rgb - 0.5) * contrast + 0.5
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    if rng.random() < 0.25:
        k_mode = int(rng.integers(0, 2))
        if k_mode == 0:
            rgb = cv2.GaussianBlur(rgb, (3, 3), sigmaX=float(rng.uniform(0.45, 0.95)))
        else:
            ker = np.array([[0.0, 0.0, 0.0], [0.25, 0.50, 0.25], [0.0, 0.0, 0.0]], dtype=np.float32)
            rgb = cv2.filter2D(rgb, -1, ker)

    noise_std = float(rng.uniform(0.004, 0.024))
    rgb = np.clip(rgb + rng.normal(0.0, noise_std, size=rgb.shape).astype(np.float32), 0.0, 1.0)

    use_gray = (rng.random() < 0.40) if force_grayscale is None else bool(force_grayscale)
    if use_gray:
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        rgb = np.stack([g, g, g], axis=-1)

    return {
        "rgb_nhwc": rgb.astype(np.float32),
        "depth_m": depth_gt_48,
        "inv_depth_01": metric_m_to_inv_depth_01(depth_gt_48),
        "elev_mask": elev_gt_48,
        "depth_m_96": depth_gt_96,
        "inv_depth_01_96": metric_m_to_inv_depth_01(depth_gt_96),
        "elev_mask_96": elev_gt_96,
        "meta": {
            "h_cam": h_cam,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "d_wall": d_wall,
            "n_objs": len(objects),
            "grayscale": use_gray,
        },
    }


# ---------------------------------------------------------------------------
# 2. 真机视频帧采样与几何-边缘感知伪真值构造 (`dataset/takes/*/take.mp4`)
# ---------------------------------------------------------------------------

def _build_real_pseudo_gt(
    rgb_96: np.ndarray,
    bg_gray_96: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """结合桌面地平线透视先验、时序背景差分与 Sobel 梯度连通域生成真机 `48x48` 伪真值。"""
    gray_96 = (0.299 * rgb_96[..., 0] + 0.587 * rgb_96[..., 1] + 0.114 * rgb_96[..., 2]).astype(np.float32)

    # 1. 利用水平边缘剖面估计相机微俯仰对应的地平线行号 (限制在合理俯仰区间 [0.40, 0.54])
    gy = np.abs(cv2.Sobel(gray_96, cv2.CV_32F, 0, 1, ksize=3))
    gx = np.abs(cv2.Sobel(gray_96, cv2.CV_32F, 1, 0, ksize=3))
    row_hor_score = gy.mean(axis=1) - 0.35 * gx.mean(axis=1)
    r_lo, r_hi = int(0.40 * INPUT_H), int(0.55 * INPUT_H)
    hor_row_96 = int(np.argmax(row_hor_score[r_lo:r_hi]) + r_lo)
    v_hor = float(hor_row_96) / float(INPUT_H)  # ~0.40 .. 0.55
    pitch_est_deg = math.degrees(math.atan((0.5 - v_hor) * 2.0 * math.tan(math.radians(DEFAULT_FOV_H_DEG * 0.5))))
    pitch_est_deg = min(max(pitch_est_deg, -8.0), 8.0)

    # 2. 解析桌面背景深度场 (48x48)
    rays_48 = make_camera_ray_grid(
        height=OUTPUT_H,
        width=OUTPUT_W,
        fov_h_deg=DEFAULT_FOV_H_DEG,
        pitch_deg=pitch_est_deg,
        roll_deg=0.0,
    )
    vz = rays_48[..., 2]
    vx = np.maximum(rays_48[..., 0], 1e-4)
    d_wall_real = 2.05
    t_tbl = np.where(vz < -1e-4, -DEFAULT_H_CAM_M / np.where(vz < -1e-4, vz, -1.0), 1e9)
    t_wall = d_wall_real / vx
    base_depth = np.clip(np.minimum(t_tbl, t_wall), DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)

    # 3. 运动前景 + 桌面亮度反差联合提取前景立体目标 (人手/手臂/躯干/桌面杂物)
    motion_96 = np.abs(gray_96 - bg_gray_96)
    tbl_ref = float(np.percentile(gray_96[int(0.72 * INPUT_H) :, :], 65))
    dark_contrast_96 = np.clip((tbl_ref - gray_96) / max(tbl_ref, 0.25), 0.0, 1.0)
    grad_mag_96 = np.sqrt(gx * gx + gy * gy)
    grad_norm_96 = grad_mag_96 / max(float(np.percentile(grad_mag_96, 95)), 1e-3)

    salience_96 = 1.65 * motion_96 + 0.38 * dark_contrast_96 + 0.16 * np.clip(grad_norm_96, 0.0, 1.5)
    salience_48 = cv2.resize(salience_96, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_AREA)

    thr = max(0.19, float(np.percentile(salience_48, 78)))
    fg_raw = (salience_48 > thr).astype(np.uint8)
    # 抑制极底部近场阴影噪点与极顶部天花板静态暗区
    fg_raw[:3, :] = 0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_clean = cv2.morphologyEx(fg_raw, cv2.MORPH_CLOSE, ker)
    fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, ker)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_clean, connectivity=8)
    depth_map = base_depth.copy()
    elev_mask = np.zeros((OUTPUT_H, OUTPUT_W), dtype=np.float32)

    hor_row_48 = int(round(v_hor * OUTPUT_H))
    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < 14:
            continue
        top_r = int(stats[lid, cv2.CC_STAT_TOP])
        h_box = int(stats[lid, cv2.CC_STAT_HEIGHT])
        bot_r = min(OUTPUT_H - 1, top_r + h_box - 1)
        if h_box < 4:
            continue

        comp = labels == lid
        # 最低接触点决定该立体目标在桌面上的物理前向距离
        contact_r = max(bot_r, hor_row_48 + 4)
        contact_r = min(contact_r, OUTPUT_H - 2)
        cols = np.where(comp[min(bot_r, OUTPUT_H - 1), :])[0]
        med_c = int(np.median(cols)) if cols.size > 0 else OUTPUT_W // 2
        d_obj = float(np.clip(base_depth[contact_r, med_c], 0.30, 1.45))

        # 伸手前景轻微向近端平滑过渡
        rr = np.arange(OUTPUT_H, dtype=np.float32)[:, None]
        relief = 1.0 - 0.06 * np.clip((bot_r - rr) / max(float(h_box), 1.0), 0.0, 1.0)
        d_comp = np.clip(d_obj * relief, DEPTH_MIN_M, d_obj)
        depth_map = np.where(comp, np.minimum(depth_map, d_comp), depth_map)

        # 计算等效物理高度 Z_equiv > 3 cm
        z_equiv = DEFAULT_H_CAM_M + depth_map * rays_48[..., 2]
        elev_mask = np.where(comp & (z_equiv > ELEV_THRESHOLD_M), 1.0, elev_mask)

    # 边缘感知双边滤波平滑伪深度
    depth_smooth = cv2.bilateralFilter(depth_map.astype(np.float32), d=5, sigmaColor=0.14, sigmaSpace=2.5)
    depth_smooth = np.clip(depth_smooth, DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)
    return depth_smooth, elev_mask.astype(np.float32)


def sample_real_robot_frames(
    takes_glob_dir: Path,
    frames_per_take: int = 40,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 take 切分采样约 240 帧真实机器人相机视频（前 n-1 段训练，最后 1 段留出展示与校验）。"""
    take_videos = sorted(takes_glob_dir.glob("*/take.mp4"))
    if not take_videos:
        return [], []

    train_samples: list[dict[str, Any]] = []
    heldout_samples: list[dict[str, Any]] = []

    for t_idx, mp4_path in enumerate(take_videos):
        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 10:
            cap.release()
            continue

        indices = np.linspace(15, max(16, total_frames - 15), num=int(frames_per_take), dtype=int)
        raw_rgbs: list[tuple[int, np.ndarray]] = []
        for f_id in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_id))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_96 = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            raw_rgbs.append((int(f_id), rgb_96))
        cap.release()

        if not raw_rgbs:
            continue

        grays = np.stack(
            [0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2] for _, r in raw_rgbs],
            axis=0,
        )
        bg_med = np.median(grays, axis=0).astype(np.float32)

        bucket = heldout_samples if (t_idx == len(take_videos) - 1 and len(take_videos) >= 2) else train_samples
        take_name = mp4_path.parent.name
        for f_id, rgb_96 in raw_rgbs:
            d_m, e_m = _build_real_pseudo_gt(rgb_96, bg_med)
            bucket.append(
                {
                    "rgb_nhwc": rgb_96,
                    "depth_m": d_m,
                    "inv_depth_01": metric_m_to_inv_depth_01(d_m),
                    "elev_mask": e_m,
                    "is_real": True,
                    "take": take_name,
                    "frame_idx": f_id,
                }
            )

    return train_samples, heldout_samples


# ---------------------------------------------------------------------------
# 3. 混合数据集 (80% 解析 3D 光线追踪 + 20% 真机视频帧) 与训练损失
# ---------------------------------------------------------------------------

class HybridMonoDepthDataset(Dataset):
    """80% 精确三维光线追踪桌面场景 + 20% 真机机器人视频帧混合训练集。"""

    def __init__(
        self,
        synth_scenes: list[dict[str, Any]],
        real_scenes: list[dict[str, Any]],
        augment: bool = True,
        seed: int = 20261118,
    ) -> None:
        self.synth_scenes = synth_scenes
        self.real_scenes = real_scenes
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        # 构造严格 80% 合成 (3200) + 20% 真机 (800) 的采样表
        self.items: list[tuple[bool, int]] = [(False, i) for i in range(len(synth_scenes))]
        if real_scenes:
            n_real_target = int(round(len(synth_scenes) * 0.25))  # 3200 : 800 == 80% : 20%
            for k in range(n_real_target):
                self.items.append((True, k % len(real_scenes)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        is_real, sub_idx = self.items[idx]
        rec = self.real_scenes[sub_idx] if is_real else self.synth_scenes[sub_idx]

        rgb = rec["rgb_nhwc"].copy()
        inv_d = rec["inv_depth_01"].copy()
        depth_m = rec["depth_m"].copy()
        elev = rec["elev_mask"].copy()

        if self.augment:
            if self.rng.random() < 0.5:
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                inv_d = np.ascontiguousarray(inv_d[:, ::-1])
                depth_m = np.ascontiguousarray(depth_m[:, ::-1])
                elev = np.ascontiguousarray(elev[:, ::-1])
            if is_real:
                gain = float(self.rng.uniform(0.88, 1.12))
                bias = float(self.rng.uniform(-0.04, 0.04))
                rgb = np.clip(rgb * gain + bias, 0.0, 1.0).astype(np.float32)

        x_chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        w_depth = np.float32(0.45 if is_real else 1.0)
        return {
            "rgb": torch.from_numpy(x_chw),
            "inv_depth": torch.from_numpy(inv_d),
            "depth_m": torch.from_numpy(depth_m),
            "elev": torch.from_numpy(elev),
            "w_depth": torch.tensor(w_depth, dtype=torch.float32),
        }


def compute_multiscale_grad_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """多尺度空间梯度 L1 损失（48x48 与 24x24），保持前景边缘锐利与桌面透视梯度平滑。"""
    total = pred.new_tensor(0.0)
    p = pred.unsqueeze(1)
    t = target.unsqueeze(1)
    w = weights.view(-1, 1, 1, 1)
    for scale in range(2):
        if scale > 0:
            p = F.avg_pool2d(p, kernel_size=2, stride=2)
            t = F.avg_pool2d(t, kernel_size=2, stride=2)
        dx = torch.abs((p[:, :, :, 1:] - p[:, :, :, :-1]) - (t[:, :, :, 1:] - t[:, :, :, :-1]))
        dy = torch.abs((p[:, :, 1:, :] - p[:, :, :-1, :]) - (t[:, :, 1:, :] - t[:, :, :-1, :]))
        total = total + (dx * w).mean() + (dy * w).mean()
    return total


def evaluate_model_on_scenes(
    model: HoverMonoDepthNet,
    scenes: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """在留出 3D 解析场景上评测精确米制深度与立体障碍物分割指标。"""
    model.eval()
    abs_rels: list[np.ndarray] = []
    sq_errs: list[np.ndarray] = []
    abs_errs: list[np.ndarray] = []
    deltas_125: list[np.ndarray] = []

    tp = 0.0
    fp = 0.0
    fn = 0.0
    fg_closer_hits = 0
    fg_closer_total = 0

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(scenes), batch_size):
            chunk = scenes[i : i + batch_size]
            x_np = np.stack([ np.transpose(s["rgb_nhwc"], (2, 0, 1)) for s in chunk ], axis=0).astype(np.float32)
            gt_d = np.stack([ s["depth_m"] for s in chunk ], axis=0).astype(np.float32)
            gt_e = np.stack([ s["elev_mask"] for s in chunk ], axis=0).astype(np.float32)

            out = model(torch.from_numpy(x_np).to(device)).cpu().numpy()
            pred_inv = out[:, 0, :, :]
            pred_elev = out[:, 1, :, :]
            pred_d = inv_depth_01_to_metric_m(pred_inv)

            abs_rels.append(np.abs(pred_d - gt_d) / np.maximum(gt_d, 1e-4))
            sq_errs.append((pred_d - gt_d) ** 2)
            abs_errs.append(np.abs(pred_d - gt_d))
            ratio = np.maximum(pred_d / np.maximum(gt_d, 1e-4), gt_d / np.maximum(pred_d, 1e-4))
            deltas_125.append((ratio < 1.25).astype(np.float32))

            pred_bin = (pred_elev >= 0.5).astype(np.float32)
            tp += float(np.sum(pred_bin * gt_e))
            fp += float(np.sum(pred_bin * (1.0 - gt_e)))
            fn += float(np.sum((1.0 - pred_bin) * gt_e))

            for b_k in range(len(chunk)):
                fg_m = gt_e[b_k] > 0.5
                bg_m = (gt_e[b_k] < 0.5) & (gt_d[b_k] > 0.65)
                if np.sum(fg_m) >= 8 and np.sum(bg_m) >= 20:
                    fg_closer_total += 1
                    if float(np.median(pred_d[b_k][fg_m])) < float(np.median(pred_d[b_k][bg_m])):
                        fg_closer_hits += 1

    all_rel = np.concatenate([a.ravel() for a in abs_rels])
    all_sq = np.concatenate([a.ravel() for a in sq_errs])
    all_abs = np.concatenate([a.ravel() for a in abs_errs])
    all_d1 = np.concatenate([a.ravel() for a in deltas_125])

    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-6)
    iou = tp / max(tp + fp + fn, 1e-6)

    return {
        "AbsRel": round(float(np.mean(all_rel)), 4),
        "RMSE_m": round(float(math.sqrt(float(np.mean(all_sq)))), 4),
        "MAE_m": round(float(np.mean(all_abs)), 4),
        "delta_125": round(float(np.mean(all_d1)), 4),
        "elev_F1": round(float(f1), 4),
        "elev_precision": round(float(prec), 4),
        "elev_recall": round(float(rec), 4),
        "elev_IoU": round(float(iou), 4),
        "fg_closer_rate": round(float(fg_closer_hits / max(fg_closer_total, 1)), 4),
    }


# ---------------------------------------------------------------------------
# 4. 深色主题四联科研展示图生成 (`fig6_mono_depth_and_3d_splats.png`)
# ---------------------------------------------------------------------------

def render_fig6_showcase(
    tflite_path: Path,
    heldout_real_scenes: list[dict[str, Any]],
    heldout_synth_scenes: list[dict[str, Any]],
    metrics: dict[str, Any],
    out_png: Path,
) -> Path:
    """使用真实导出的 `.tflite` 模型渲染深色主题四联双行展示图 `fig6_mono_depth_and_3d_splats.png`。"""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 挑选一帧包含显著前景目标的留出真机帧，以及一帧包含多立体目标的留出解析 3D 场景
    if heldout_real_scenes:
        real_pick = max(heldout_real_scenes, key=lambda s: float(np.sum(s["elev_mask"])))
    else:
        rng_fb = np.random.default_rng(77)
        real_pick = render_analytical_desktop_scene(rng_fb, force_grayscale=True, fixed_pose=(DEFAULT_H_CAM_M, 2.0, 0.0))

    synth_candidates = [
        s
        for s in heldout_synth_scenes
        if (not s["meta"]["grayscale"])
        and abs(s["meta"]["pitch_deg"]) < 5.5
        and float(np.sum(s["elev_mask"])) > 95.0
    ]
    synth_pick = synth_candidates[0] if synth_candidates else heldout_synth_scenes[0]

    rows_data = [
        ("Real Robot Stream (Held-Out GC0308 Take)", real_pick, 0.0, 0.0, DEFAULT_H_CAM_M),
        (
            "Held-Out 3D Desktop Scene (Ray-Traced Metric Validation)",
            synth_pick,
            float(synth_pick["meta"]["pitch_deg"]),
            float(synth_pick["meta"]["roll_deg"]),
            float(synth_pick["meta"]["h_cam"]),
        ),
    ]

    bg_dark = "#0b1120"
    card_dark = "#111827"
    border_col = "#334155"
    txt_main = "#f8fafc"
    txt_sub = "#94a3b8"

    fig = plt.figure(figsize=(16.8, 8.6), facecolor=bg_dark)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.08, 1.0, 1.32], wspace=0.22, hspace=0.28, left=0.04, right=0.98, top=0.89, bottom=0.08)

    fig.suptitle(
        "HoverMonoDepthNet (TFLite v3, 294 KB) — Monocular Metric Depth, Obstacle Elevation & 3D Gaussian Point Cloud",
        color=txt_main,
        fontsize=13.2,
        fontweight="bold",
        y=0.96,
    )

    col_titles = [
        "(a) Robot Camera RGB Input (96x96)",
        "(b) Predicted Metric Depth (m)",
        "(c) Predicted Obstacle Elevation (>3cm)",
        "(d) Reconstructed 3D Gaussian Splats + Target Beacon",
    ]

    for r_idx, (row_label, sample, pitch_deg, roll_deg, h_cam_m) in enumerate(rows_data):
        rgb = sample["rgb_nhwc"]
        pred = infer_mono_depth_tflite(tflite_path, rgb)
        depth_m = pred["depth_m"]
        elev_p = pred["elev_prob_01"]

        splats = unproject_depth_to_3d_splats(
            depth_m=depth_m,
            elev_prob_01=elev_p,
            rgb_in=rgb,
            h_cam_m=h_cam_m,
            fov_h_deg=DEFAULT_FOV_H_DEG,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        xyz_grid = splats["xyz_grid"]

        # 在桌面无障碍净空区自动选定敲击导引信标点 (Pinged Target Beacon)
        cand_mask = (elev_p < 0.25) & (depth_m > 0.48) & (depth_m < 1.05) & (np.abs(xyz_grid[..., 1]) < 0.28)
        cand_rc = np.argwhere(cand_mask)
        if cand_rc.shape[0] > 0:
            mid_rc = cand_rc[cand_rc.shape[0] // 2]
            tr, tc = int(mid_rc[0]), int(mid_rc[1])
        else:
            tr, tc = int(OUTPUT_H * 0.72), int(OUTPUT_W * 0.56)

        beacon_xyz = xyz_grid[tr, tc].copy()
        beacon_xyz[2] = 0.0  # 锁存至桌面 Z = 0
        bu_96, bv_96 = tc * 2 + 1, tr * 2 + 1

        # --- Panel (a): RGB Input ---
        ax_a = fig.add_subplot(gs[r_idx, 0], facecolor=card_dark)
        ax_a.imshow(rgb)
        ax_a.scatter([bu_96], [bv_96], s=95, facecolors="none", edgecolors="#10b981", linewidths=2.0, zorder=5)
        ax_a.scatter([bu_96], [bv_96], s=18, c="#34d399", zorder=6)
        ax_a.text(
            3,
            8,
            row_label,
            color="#38bdf8",
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0f172a", edgecolor="#0284c7", alpha=0.86),
        )
        if r_idx == 0:
            ax_a.set_title(col_titles[0], color=txt_main, fontsize=10.2, fontweight="bold", pad=8)
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        for sp in ax_a.spines.values():
            sp.set_edgecolor(border_col)

        # --- Panel (b): Metric Depth Heatmap (m) ---
        ax_b = fig.add_subplot(gs[r_idx, 1], facecolor=card_dark)
        im_d = ax_b.imshow(depth_m, cmap="turbo_r", vmin=0.22, vmax=2.45)
        try:
            cs = ax_b.contour(depth_m, levels=[0.5, 1.0, 1.6], colors="white", linewidths=0.65, alpha=0.55)
            ax_b.clabel(cs, fmt="%.1fm", fontsize=6.8, colors="white")
        except Exception:
            pass
        cb = fig.colorbar(im_d, ax=ax_b, fraction=0.046, pad=0.03)
        cb.ax.yaxis.set_tick_params(color=txt_sub, labelsize=7.5)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=txt_sub)
        cb.outline.set_edgecolor(border_col)

        if r_idx == 1:
            gt_d = sample["depth_m"]
            scene_rel = float(np.mean(np.abs(depth_m - gt_d) / np.maximum(gt_d, 1e-4)))
            ax_b.text(
                2,
                45,
                f"Scene AbsRel = {scene_rel:.3f}",
                color="#f8fafc",
                fontsize=7.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.85),
            )
        if r_idx == 0:
            ax_b.set_title(col_titles[1], color=txt_main, fontsize=10.2, fontweight="bold", pad=8)
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        for sp in ax_b.spines.values():
            sp.set_edgecolor(border_col)

        # --- Panel (c): Predicted Obstacle Elevation Mask ---
        ax_c = fig.add_subplot(gs[r_idx, 2], facecolor=card_dark)
        rgb_48 = rgb.reshape(OUTPUT_H, 2, OUTPUT_W, 2, 3).mean(axis=(1, 3))
        dim_bg = 0.32 * rgb_48
        heat_c = plt.cm.magma(np.clip(elev_p, 0.0, 1.0))[..., :3]
        alpha_e = np.clip(elev_p * 1.15, 0.0, 0.92)[..., None]
        comp_vis = np.clip(dim_bg * (1.0 - alpha_e) + heat_c * alpha_e, 0.0, 1.0)
        ax_c.imshow(comp_vis)
        if float(np.max(elev_p)) > 0.50 and float(np.min(elev_p)) < 0.50:
            ax_c.contour(elev_p, levels=[0.50], colors=["#22d3ee"], linewidths=1.25)
        ax_c.scatter([tc], [tr], s=65, facecolors="none", edgecolors="#10b981", linewidths=1.8)
        ax_c.text(
            2,
            45,
            "Cyan contour: P(Z > 3cm) = 0.50",
            color="#e2e8f0",
            fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#0f172a", edgecolor="#475569", alpha=0.84),
        )
        if r_idx == 0:
            ax_c.set_title(col_titles[2], color=txt_main, fontsize=10.2, fontweight="bold", pad=8)
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        for sp in ax_c.spines.values():
            sp.set_edgecolor(border_col)

        # --- Panel (d): 3D Colored Gaussian Point Cloud + Target Beacon ---
        ax_d = fig.add_subplot(gs[r_idx, 3], projection="3d", facecolor=card_dark)
        ax_d.computed_zorder = False
        pts = splats["xyz_flat"]
        cols = splats["rgb_flat"].copy()
        elev_f = splats["elev_flat"]
        rad_f = splats["radius_flat"]

        keep = (pts[:, 0] > 0.18) & (pts[:, 0] < 2.25) & (np.abs(pts[:, 1]) < 0.85) & (pts[:, 2] > -0.08) & (pts[:, 2] < 0.55)
        pts_k = pts[keep]
        cols_k = cols[keep]
        elev_k = elev_f[keep]
        rad_k = rad_f[keep]

        # 按与相机距离由远到近排序绘制，保证遮挡层次自然且导引信标清晰置顶
        order = np.argsort(-pts_k[:, 0])
        pts_k = pts_k[order]
        cols_k = cols_k[order]
        elev_k = elev_k[order]
        rad_k = rad_k[order]

        # 对高置信凸起障碍物叠加暖珊瑚高亮色，增强 3D 几何层次辨识度
        obs_tint = np.array([1.0, 0.36, 0.28], dtype=np.float32)
        w_tint = (np.clip((elev_k - 0.45) / 0.55, 0.0, 1.0) * 0.55)[:, None]
        cols_blend = np.clip(cols_k * (1.0 - w_tint) + obs_tint[None, :] * w_tint, 0.0, 1.0)
        sizes = np.clip((rad_k * 950.0) ** 1.35, 5.5, 42.0)

        # 桌面参考网格 Z = 0
        for gx_v in np.linspace(0.25, 1.75, 7):
            ax_d.plot([gx_v, gx_v], [-0.65, 0.65], [0.0, 0.0], color="#1e293b", lw=0.65, zorder=1)
        for gy_v in np.linspace(-0.60, 0.60, 7):
            ax_d.plot([0.20, 1.80], [gy_v, gy_v], [0.0, 0.0], color="#1e293b", lw=0.65, zorder=1)

        ax_d.scatter(
            pts_k[:, 0],
            pts_k[:, 1],
            pts_k[:, 2],
            c=cols_blend,
            s=sizes,
            alpha=0.84,
            edgecolors="none",
            depthshade=False,
            zorder=3,
        )

        # 机器人相机位置 (0, 0, h_cam) 与敲击目标导引信标 (Pinged Target Beacon)
        ax_d.scatter([0.0], [0.0], [h_cam_m], c="#38bdf8", s=52, marker="^", edgecolors="white", linewidths=0.9, zorder=8)
        bx, by = float(beacon_xyz[0]), float(beacon_xyz[1])
        ax_d.plot([0.0, bx], [0.0, by], [0.01, 0.01], color="#34d399", lw=1.5, alpha=0.90, zorder=9)
        ax_d.plot([bx, bx], [by, by], [0.0, 0.25], color="#10b981", lw=2.3, linestyle="--", zorder=10)
        ax_d.scatter([bx], [by], [0.0], c="#10b981", s=95, marker="o", edgecolors="#a7f3d0", linewidths=1.6, zorder=11)
        ax_d.scatter([bx], [by], [0.25], c="#34d399", s=62, marker="D", edgecolors="white", linewidths=0.9, zorder=12)

        ax_d.text2D(
            0.04,
            0.91,
            f"Target Beacon: ({bx:.2f}m, {by:+.2f}m, Z=0)",
            transform=ax_d.transAxes,
            color="#6ee7b7",
            fontsize=7.6,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0f172a", edgecolor="#10b981", alpha=0.88),
        )

        ax_d.set_xlim(0.0, 1.85)
        ax_d.set_ylim(-0.68, 0.68)
        ax_d.set_zlim(-0.04, 0.46)
        ax_d.view_init(elev=25, azim=-148)
        ax_d.set_xlabel("X fwd (m)", color=txt_sub, fontsize=7.5, labelpad=-3)
        ax_d.set_ylabel("Y right (m)", color=txt_sub, fontsize=7.5, labelpad=-3)
        ax_d.set_zlabel("Z up (m)", color=txt_sub, fontsize=7.5, labelpad=-4)
        ax_d.tick_params(colors=txt_sub, labelsize=6.6, pad=-1)
        for axis in (ax_d.xaxis, ax_d.yaxis, ax_d.zaxis):
            axis.pane.set_facecolor((0.06, 0.09, 0.16, 0.95))
            axis.pane.set_edgecolor("#1e293b")
            axis._axinfo["grid"]["color"] = (0.18, 0.25, 0.36, 0.35)
        if r_idx == 0:
            ax_d.set_title(col_titles[3], color=txt_main, fontsize=10.2, fontweight="bold", pad=8)

    footer = (
        f"Validation (400 Held-Out 3D Desktop Scenes):  AbsRel = {metrics['AbsRel']:.4f}  |  "
        f"RMSE = {metrics['RMSE_m']:.4f} m  |  delta<1.25 = {metrics['delta_125']*100:.1f}%  |  "
        f"Elev F1 = {metrics['elev_F1']:.4f}  |  TFLite = {metrics['tflite_bytes']/1024:.1f} KB  |  "
        f"CPU = {metrics['cpu_latency_ms']:.2f} ms  |  PyTorch<->TFLite max|diff| = {metrics['max_abs_diff']:.2e}"
    )
    fig.text(
        0.5,
        0.022,
        footer,
        ha="center",
        va="center",
        color="#e2e8f0",
        fontsize=9.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e293b", edgecolor="#38bdf8", lw=1.1),
    )

    fig.savefig(out_png, dpi=220, facecolor=bg_dark)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# 5. 主训练与导出入口
# ---------------------------------------------------------------------------

def train_and_export(
    n_train_synth: int = 3200,
    n_val_synth: int = 400,
    epochs: int = 56,
    batch_size: int = 64,
    lr: float = 2.6e-3,
    seed: int = 20261118,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MonoDepth] Using device: {device}")

    t_gen0 = time.perf_counter()
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 99991)

    print(f"[MonoDepth] Generating {n_train_synth} analytical 3D training scenes + {n_val_synth} held-out validation scenes...")
    train_synth = [render_analytical_desktop_scene(rng_train) for _ in range(int(n_train_synth))]
    val_synth = [render_analytical_desktop_scene(rng_val) for _ in range(int(n_val_synth))]

    takes_dir = REPO_ROOT / "dataset" / "takes"
    real_train, real_heldout = sample_real_robot_frames(takes_dir, frames_per_take=40)
    print(
        f"[MonoDepth] Scene dataset ready in {time.perf_counter() - t_gen0:.2f}s | "
        f"synth_train={len(train_synth)}, real_train={len(real_train)}, real_heldout={len(real_heldout)}, val_synth={len(val_synth)}"
    )

    dataset = HybridMonoDepthDataset(train_synth, real_train, augment=True, seed=seed)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = HoverMonoDepthNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=5e-5)

    pos_weight = torch.tensor([2.4], device=device)

    t_tr0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt_inv = batch["inv_depth"].to(device, non_blocking=True)
            gt_d = batch["depth_m"].to(device, non_blocking=True)
            gt_elev = batch["elev"].to(device, non_blocking=True)
            w_d = batch["w_depth"].to(device, non_blocking=True)

            logits = model.forward_logits(rgb)
            probs = torch.sigmoid(logits)
            pred_inv = probs[:, 0, :, :]
            pred_elev_prob = probs[:, 1, :, :]
            pred_elev_logit = logits[:, 1, :, :]

            # 1. 逆深度 L1 损失 + 米制相对误差损失
            l_inv = (torch.abs(pred_inv - gt_inv) * w_d.view(-1, 1, 1)).mean()
            pred_d = inv_depth_01_to_metric_m(pred_inv)
            l_rel = ((torch.abs(pred_d - gt_d) / torch.clamp(gt_d, min=0.20)) * w_d.view(-1, 1, 1)).mean()

            # 2. 多尺度空间梯度损失
            l_grad = compute_multiscale_grad_loss(pred_inv, gt_inv, w_d)

            # 3. 障碍物高程 BCE + Soft Dice 损失
            bce_raw = F.binary_cross_entropy_with_logits(pred_elev_logit, gt_elev, pos_weight=pos_weight, reduction="none")
            l_bce = (bce_raw * w_d.view(-1, 1, 1)).mean()
            inter = (pred_elev_prob * gt_elev * w_d.view(-1, 1, 1)).sum(dim=(1, 2))
            denom = ((pred_elev_prob + gt_elev) * w_d.view(-1, 1, 1)).sum(dim=(1, 2))
            l_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

            loss = 1.60 * l_inv + 0.45 * l_rel + 0.85 * l_grad + 0.55 * l_bce + 0.35 * l_dice

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        if ep == 1 or ep % 10 == 0 or ep == int(epochs):
            print(f"  [Epoch {ep:02d}/{epochs}] train_loss={running_loss / max(n_batches, 1):.4f} lr={scheduler.get_last_lr()[0]:.5f}")

    train_sec = time.perf_counter() - t_tr0
    print(f"[MonoDepth] Training finished in {train_sec:.2f}s")

    # 严格仅在训练结束后在 400 个留出三维场景上评测一次
    val_metrics = evaluate_model_on_scenes(model, val_synth, device=device)

    # 测算单帧 CPU 推理时延 (NHWC 融合图)
    model_cpu = model.to("cpu").eval()
    cpu_ms = benchmark_cpu_latency_ms(model_cpu, n_runs=60)

    out_dir = REPO_ROOT / "runs_end_side" / "mono_depth"
    android_assets = Path("G:/robot_app/android/app/src/main/assets/models")
    bundle_info = export_mono_depth_bundle(
        model_cpu,
        out_dir=out_dir,
        android_assets_dir=android_assets,
        version="2026.11.18",
    )

    full_metrics: dict[str, Any] = {
        "model_name": "hover-mono-depth-v1",
        "version": "2026.11.18",
        "input_shape": [1, INPUT_H, INPUT_W, INPUT_C],
        "output_shape": [1, OUTPUT_H, OUTPUT_W, 2],
        "train_synth_scenes": int(n_train_synth),
        "train_real_frames": len(real_train),
        "heldout_real_frames": len(real_heldout),
        "val_synth_scenes": int(n_val_synth),
        "epochs": int(epochs),
        "train_time_sec": round(train_sec, 2),
        "cpu_latency_ms": round(cpu_ms, 3),
        **val_metrics,
        "tflite_bytes": bundle_info["tflite_bytes"],
        "max_abs_diff": bundle_info["max_abs_diff"],
        "sha256": bundle_info["sha256"],
        "artifacts": {
            "tflite_path": bundle_info["tflite_path"],
            "json_path": bundle_info["json_path"],
            "zip_path": bundle_info["zip_path"],
            "android_assets": bundle_info["android_assets"],
        },
    }

    metrics_file = out_dir / "depth_eval_metrics.json"
    metrics_file.write_text(json.dumps(full_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig6_path = REPO_ROOT / "docs" / "showcase" / "figs" / "fig6_mono_depth_and_3d_splats.png"
    render_fig6_showcase(
        tflite_path=Path(bundle_info["tflite_path"]),
        heldout_real_scenes=real_heldout,
        heldout_synth_scenes=val_synth,
        metrics=full_metrics,
        out_png=fig6_path,
    )
    full_metrics["artifacts"]["fig6_path"] = str(fig6_path)
    metrics_file.write_text(json.dumps(full_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[MonoDepth] Evaluation & Export Summary:")
    print(json.dumps(full_metrics, ensure_ascii=False, indent=2))
    return full_metrics


if __name__ == "__main__":
    train_and_export()
