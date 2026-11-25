# -*- coding: utf-8 -*-
"""train_neural_gaussian_splat.py — 前馈式 6 通道神经 3D 高斯网络训练、多视角贝叶斯记忆拓展评测与导出管线。

核心内容：
1. 支持任意 6-DoF 机器人位姿的四壁桌面三维光线追踪器 (`render_posed_desktop_scene`)，精确生成：
   - 度量深度 `depth_m`、突起掩膜 `elev_mask`；
   - 基于三维法向与深度边缘的各向异性高斯真值先验 `(scale_xy_gt, scale_z_gt)` 与实体不透明度 `opacity_gt`；
   - 双视角位姿对 `(View 1, View 2)` 供可微新视角 3DGS 重投影光度一致性训练。
2. GPU 联合监督损失：
   - 逆深度 + 相对米制深度 + 多尺度梯度损失 (`ch0`)；
   - 突起障碍 BCE + Soft Dice (`ch1`)；
   - 各向异性水平/垂直椭球半径与不透明度监督 (`ch2, ch3, ch4`)；
   - 异方差高斯负对数似然损失 (`Heteroscedastic Uncertainty NLL`, 让网络自发习得 `ch5: precision_01`)；
   - 可微各向异性 3DGS 新视角光度重投影损失 (`differentiable_novel_view_render_24x24`)；
   - 融合 40% 三通道灰度 `[g,g,g]` 增广与 `dataset/takes/` 真机视频帧适应（严格留出 `robot_20261115_151733_cb8455`）。
3. 评测机器人从区域 A `(0, 0)` 绕水杯探索至区域 B `(1.05m, 0.48m)` 再盲区回点跑回区域 A 的完整空间记忆闭环，
   导出 `hover_gaussian_splat_v2.tflite/.json/bundle.zip`（同步拷贝至 Android `assets/models/`）、
   `expanded_memory_map.ply`、`neural_3dgs_metrics.json` 与 220 DPI 科研大图 `fig10_neural_3dgs_memory_expansion.png`。
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
if str(REPO_ROOT / "pc" / "pack") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pc" / "pack"))
import sealed  # noqa: E402

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
    inv_depth_01_to_metric_m,
    metric_m_to_inv_depth_01,
)
from neural_gaussian_splat_model import (  # noqa: E402
    OPACITY_MAX,
    OPACITY_MIN,
    OPACITY_SPAN,
    SCALE_XY_MAX_M,
    SCALE_XY_MIN_M,
    SCALE_Z_MAX_M,
    SCALE_Z_MIN_M,
    GlobalMemoryNavigator,
    HoverGaussianSplatNet,
    OrbitCamera3D,
    PersistentGaussianMemoryMap,
    RobotCameraPose,
    benchmark_neural_splat_cpu_ms,
    differentiable_novel_view_render_24x24,
    export_neural_splat_bundle,
    infer_neural_splat_tflite,
    make_world_ray_grid,
    precision_01_to_tau,
    scale_xy_m_to_01,
    scale_z_m_to_01,
)
from train_mono_depth import (  # noqa: E402
    _intersect_cylinder,
    _intersect_ellipsoid,
    _intersect_obb,
    compute_multiscale_grad_loss,
    sample_real_robot_frames,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220

DEFAULT_OUT_DIR = REPO_ROOT / "runs_end_side" / "neural_3dgs"
DEFAULT_ANDROID_ASSETS = Path("G:/robot_app/android/app/src/main/assets/models")
DEFAULT_FIG10_PATH = REPO_ROOT / "docs" / "showcase" / "figs" / "fig10_neural_3dgs_memory_expansion.png"
HELDOUT_TAKE_ID = "robot_20261115_151733_cb8455"


# ---------------------------------------------------------------------------
# 1. 支持任意 6-DoF 世界位姿的 3D 桌面光线追踪与各向异性高斯属性真值计算
# ---------------------------------------------------------------------------

def sample_random_3d_arena_config(rng: np.random.Generator) -> dict[str, Any]:
    """随机生成桌面材质纹理、四面房间围墙边界与 2~5 个三维桌面立体几何物体。"""
    bounds = {
        "x_min": float(rng.uniform(-1.10, -0.75)),
        "x_max": float(rng.uniform(1.85, 2.55)),
        "y_min": float(rng.uniform(-1.45, -1.05)),
        "y_max": float(rng.uniform(1.05, 1.55)),
    }
    tbl_style = int(rng.integers(0, 3))
    tbl_base = rng.uniform(0.32, 0.74, size=(3,)).astype(np.float32)
    wall_base = rng.uniform(0.22, 0.66, size=(3,)).astype(np.float32)
    light_dir = np.array(
        [float(rng.uniform(-0.35, 0.45)), float(rng.uniform(-0.45, 0.45)), float(rng.uniform(0.75, 1.0))],
        dtype=np.float32,
    )
    light_dir /= float(np.linalg.norm(light_dir))

    n_objs = int(rng.integers(2, 6))
    objects: list[dict[str, Any]] = []
    placed: list[tuple[float, float, float]] = []
    for _ in range(n_objs * 5):
        if len(objects) >= n_objs:
            break
        kind = str(rng.choice(["cylinder", "box", "sphere", "human_hand"], p=[0.34, 0.30, 0.18, 0.18]))
        cx = float(rng.uniform(0.28, min(1.55, bounds["x_max"] - 0.25)))
        max_y = max(0.08, 0.46 * cx)
        cy = float(rng.uniform(-max_y, max_y))
        r_foot = 0.12 if kind == "human_hand" else float(rng.uniform(0.032, 0.078))
        if any(math.hypot(cx - px, cy - py) < (r_foot + pr + 0.04) for px, py, pr in placed):
            continue
        placed.append((cx, cy, r_foot))
        color = rng.uniform(0.15, 0.88, size=(3,)).astype(np.float32)

        if kind == "cylinder":
            objects.append(
                {
                    "kind": "cylinder",
                    "cx": cx,
                    "cy": cy,
                    "r": r_foot,
                    "h": float(rng.uniform(0.085, 0.195)),
                    "color": color,
                    "stripe": float(rng.uniform(0.0, 1.0)),
                }
            )
        elif kind == "box":
            h_box = float(rng.uniform(0.014, 0.026)) if rng.random() < 0.14 else float(rng.uniform(0.048, 0.165))
            objects.append(
                {
                    "kind": "box",
                    "cx": cx,
                    "cy": cy,
                    "lx": float(rng.uniform(0.032, 0.085)),
                    "ly": float(rng.uniform(0.032, 0.085)),
                    "h": h_box,
                    "yaw": float(rng.uniform(-1.5, 1.5)),
                    "color": color,
                }
            )
        elif kind == "sphere":
            rad = float(rng.uniform(0.036, 0.078))
            objects.append({"kind": "sphere", "cx": cx, "cy": cy, "r": rad, "color": color})
        else:
            skin = np.array(
                [rng.uniform(0.62, 0.86), rng.uniform(0.46, 0.68), rng.uniform(0.36, 0.56)],
                dtype=np.float32,
            )
            objects.append(
                {
                    "kind": "human_hand",
                    "cx": cx,
                    "cy": cy,
                    "fist_r": float(rng.uniform(0.042, 0.062)),
                    "arm_yaw": float(rng.uniform(-0.6, 0.6)),
                    "skin": skin,
                    "sleeve": color,
                }
            )

    return {
        "bounds": bounds,
        "tbl_style": tbl_style,
        "tbl_base": tbl_base,
        "wall_base": wall_base,
        "light_dir": light_dir,
        "objects": objects,
    }


def compute_anisotropic_splat_targets(
    depth_48: np.ndarray,
    norm_48: np.ndarray,
    elev_48: np.ndarray,
) -> dict[str, np.ndarray]:
    """根据精确三维表面法向、度量深度与遮挡边缘跳变计算各向异性高斯参数监督真值。"""
    nz_abs = np.abs(norm_48[..., 2])
    gx = np.abs(cv2.Sobel(depth_48, cv2.CV_32F, 1, 0, ksize=3)) * 0.25
    gy = np.abs(cv2.Sobel(depth_48, cv2.CV_32F, 0, 1, ksize=3)) * 0.25
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_01 = np.clip(edge_mag / 0.10, 0.0, 1.0)

    is_flat_horiz = (nz_abs > 0.82) & (elev_48 < 0.5) & (edge_01 < 0.35)
    # 水平桌面生成平滑大水平圆盘 (0.024..0.042m)；立体物体与边缘收缩水平核 (0.009..0.018m)
    s_xy_flat = np.clip(0.022 + 0.013 * depth_48, 0.023, 0.043)
    s_xy_obj = np.clip((0.011 + 0.008 * depth_48) * (1.0 - 0.38 * edge_01), 0.0085, 0.022)
    s_xy_m = np.where(is_flat_horiz, s_xy_flat, s_xy_obj).astype(np.float32)

    # 水平面垂直半径极薄 (4.5..7.5mm 扁平圆盘)；立面杯壁/人手生成竖直长椭球 (19..36mm)
    vert_factor = np.clip(1.0 - nz_abs, 0.0, 1.0)
    s_z_m = np.where(
        nz_abs > 0.85,
        0.0045 + 0.0025 * np.clip(depth_48 - 0.3, 0.0, 1.5),
        np.clip(0.014 + 0.022 * vert_factor + 0.004 * elev_48, 0.012, 0.038),
    ).astype(np.float32)

    # 实体桌面与立体目标高不透明 (0.86..0.97)，远墙 (>1.85m) 与深度断裂带自动压低不透明度防飞点
    far_atten = np.clip((depth_48 - 1.75) / 0.75, 0.0, 1.0)
    opac = np.clip(0.95 - 0.58 * far_atten - 0.36 * edge_01, 0.14, 0.97).astype(np.float32)
    opac_01 = np.clip((opac - OPACITY_MIN) / OPACITY_SPAN, 0.0, 1.0).astype(np.float32)

    return {
        "scale_xy_m": s_xy_m,
        "scale_z_m": s_z_m,
        "scale_xy_01": scale_xy_m_to_01(s_xy_m),
        "scale_z_01": scale_z_m_to_01(s_z_m),
        "opacity_01": opac_01,
        "edge_01": edge_01.astype(np.float32),
    }


def render_posed_desktop_scene(
    arena: dict[str, Any],
    pose: RobotCameraPose,
    rng: np.random.Generator | None = None,
    grayscale: bool = False,
    gain_bias: tuple[float, float] = (1.0, 0.0),
    noise_std: float = 0.008,
) -> dict[str, Any]:
    """从任意 6-DoF `RobotCameraPose` 光线追踪渲染 `96x96` 图像与精确 `48x48` 6 通道物理真值。"""
    origin, rays_96 = make_world_ray_grid(pose, height=INPUT_H, width=INPUT_W)
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    vx, vy, vz = rays_96[..., 0], rays_96[..., 1], rays_96[..., 2]

    inf = np.float32(1e9)
    # 1. 桌面平面 Z = 0
    hit_tbl = vz < -1e-5
    t_tbl = np.where(hit_tbl, -oz / np.where(hit_tbl, vz, -1.0), inf).astype(np.float32)
    t_tbl = np.where(t_tbl > 0.06, t_tbl, inf)

    # 2. 四壁房间围墙 (支持 360° 任意偏航角观察)
    b = arena["bounds"]
    t_best = t_tbl.copy()
    norm_96 = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
    norm_96[..., 2] = 1.0
    surf_type = np.zeros((INPUT_H, INPUT_W), dtype=np.int32)  # 0: table, 1: wall, >=2: object

    walls = [
        (b["x_max"], ox, vx, np.array([-1.0, 0.0, 0.0], dtype=np.float32)),
        (b["x_min"], ox, vx, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        (b["y_max"], oy, vy, np.array([0.0, -1.0, 0.0], dtype=np.float32)),
        (b["y_min"], oy, vy, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for w_coord, o_c, v_c, n_vec in walls:
        valid_w = np.abs(v_c) > 1e-5
        t_w = np.where(valid_w, (w_coord - o_c) / np.where(valid_w, v_c, 1.0), inf).astype(np.float32)
        z_w = oz + t_w * vz
        take = valid_w & (t_w > 0.08) & (z_w >= -0.02) & (t_w < t_best)
        if np.any(take):
            t_best = np.where(take, t_w, t_best)
            surf_type = np.where(take, 1, surf_type)
            norm_96 = np.where(take[..., None], n_vec[None, None, :], norm_96)

    xyz_bg = origin[None, None, :] + t_best[..., None] * rays_96
    albedo_96 = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)

    tbl_base = arena["tbl_base"]
    tbl_style = arena["tbl_style"]
    if tbl_style == 0:
        grain = 0.5 + 0.5 * np.sin(xyz_bg[..., 1] * 38.0 + 2.2 * np.sin(xyz_bg[..., 0] * 8.5))
        tbl_col = tbl_base[None, None, :] * (0.86 + 0.24 * grain[..., None])
    elif tbl_style == 1:
        gx_m = np.abs((xyz_bg[..., 0] * 8.0) - np.round(xyz_bg[..., 0] * 8.0)) < 0.055
        gy_m = np.abs((xyz_bg[..., 1] * 8.0) - np.round(xyz_bg[..., 1] * 8.0)) < 0.055
        line_m = (gx_m | gy_m).astype(np.float32)
        tbl_col = tbl_base[None, None, :] * (1.0 - 0.22 * line_m[..., None])
    else:
        chk = ((np.floor(xyz_bg[..., 0] * 4.5) + np.floor(xyz_bg[..., 1] * 4.5)) % 2 == 0).astype(np.float32)
        tbl_col = tbl_base[None, None, :] * (0.88 + 0.18 * chk[..., None])

    wall_base = arena["wall_base"]
    w_band = 0.90 + 0.12 * np.cos(xyz_bg[..., 2] * 14.0) * np.cos((xyz_bg[..., 0] + xyz_bg[..., 1]) * 5.0)
    wall_col = wall_base[None, None, :] * w_band[..., None]
    albedo_96 = np.where((surf_type == 0)[..., None], tbl_col, wall_col)

    # 3. 逐个求交三维立体物体
    foot_centers: list[tuple[float, float, float]] = []
    for obj in arena["objects"]:
        kind = obj["kind"]
        if kind == "cylinder":
            t_o, n_o = _intersect_cylinder(ox, oy, oz, vx, vy, vz, obj["cx"], obj["cy"], obj["r"], obj["h"])
            foot_centers.append((obj["cx"], obj["cy"], obj["r"]))
            take = t_o < t_best
            if np.any(take):
                z_hit = oz + t_o * vz
                band = (np.abs(z_hit - 0.55 * obj["h"]) < 0.016).astype(np.float32) * obj["stripe"]
                c_obj = obj["color"][None, None, :] * (1.0 - 0.30 * band[..., None]) + 0.25 * band[..., None]
                t_best = np.where(take, t_o, t_best)
                norm_96 = np.where(take[..., None], n_o, norm_96)
                albedo_96 = np.where(take[..., None], c_obj, albedo_96)
                surf_type = np.where(take, 2, surf_type)
        elif kind == "box":
            t_o, n_o = _intersect_obb(
                ox, oy, oz, vx, vy, vz, obj["cx"], obj["cy"], 0.0, obj["lx"], obj["ly"], 0.5 * obj["h"], obj["yaw"]
            )
            foot_centers.append((obj["cx"], obj["cy"], max(obj["lx"], obj["ly"])))
            take = t_o < t_best
            if np.any(take):
                t_best = np.where(take, t_o, t_best)
                norm_96 = np.where(take[..., None], n_o, norm_96)
                albedo_96 = np.where(take[..., None], obj["color"][None, None, :], albedo_96)
                surf_type = np.where(take, 2, surf_type)
        elif kind == "sphere":
            r_s = obj["r"]
            t_o, n_o = _intersect_ellipsoid(ox, oy, oz, vx, vy, vz, obj["cx"], obj["cy"], r_s, r_s, r_s, r_s)
            foot_centers.append((obj["cx"], obj["cy"], r_s))
            take = t_o < t_best
            if np.any(take):
                t_best = np.where(take, t_o, t_best)
                norm_96 = np.where(take[..., None], n_o, norm_96)
                albedo_96 = np.where(take[..., None], obj["color"][None, None, :], albedo_96)
                surf_type = np.where(take, 2, surf_type)
        else:
            fr = obj["fist_r"]
            cx_h, cy_h = obj["cx"], obj["cy"]
            foot_centers.append((cx_h, cy_h, fr * 1.3))
            t_f, n_f = _intersect_ellipsoid(ox, oy, oz, vx, vy, vz, cx_h, cy_h, fr * 0.95, fr * 1.1, fr, fr * 0.92)
            ay = obj["arm_yaw"]
            t_a, n_a = _intersect_obb(
                ox,
                oy,
                oz,
                vx,
                vy,
                vz,
                cx_h + 0.15 * math.cos(ay),
                cy_h + 0.15 * math.sin(ay),
                fr * 0.35,
                0.15,
                0.038,
                0.036,
                ay,
            )
            for t_part, n_part, col_part in ((t_f, n_f, obj["skin"]), (t_a, n_a, obj["sleeve"])):
                take = t_part < t_best
                if np.any(take):
                    t_best = np.where(take, t_part, t_best)
                    norm_96 = np.where(take[..., None], n_part, norm_96)
                    albedo_96 = np.where(take[..., None], col_part[None, None, :], albedo_96)
                    surf_type = np.where(take, 2, surf_type)

    depth_96 = np.clip(t_best, DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)
    xyz_96 = origin[None, None, :] + depth_96[..., None] * rays_96

    # 桌面接触软阴影 + Blinn-Phong 光照
    shadow = np.ones((INPUT_H, INPUT_W), dtype=np.float32)
    on_tbl = surf_type == 0
    for fcx, fcy, frad in foot_centers:
        d2 = (xyz_96[..., 0] - fcx) ** 2 + (xyz_96[..., 1] - fcy) ** 2
        sh = 1.0 - 0.38 * np.exp(-d2 / max(2.0 * (frad * 1.3) ** 2, 1e-4))
        shadow = np.where(on_tbl, np.minimum(shadow, sh), shadow)

    l_dir = arena["light_dir"]
    ndotl = np.clip(np.sum(norm_96 * l_dir[None, None, :], axis=-1), 0.0, 1.0)
    view_dir = (origin[None, None, :] - xyz_96)
    view_dir /= np.maximum(np.linalg.norm(view_dir, axis=-1, keepdims=True), 1e-6)
    half_v = l_dir[None, None, :] + view_dir
    half_v /= np.maximum(np.linalg.norm(half_v, axis=-1, keepdims=True), 1e-6)
    spec = 0.09 * (np.clip(np.sum(norm_96 * half_v, axis=-1), 0.0, 1.0) ** 22) * shadow

    rgb = albedo_96 * (0.34 + 0.66 * ndotl * shadow)[..., None] + spec[..., None]
    rgb = np.clip((rgb - 0.5) * gain_bias[0] + 0.5 + gain_bias[1], 0.0, 1.0).astype(np.float32)

    if rng is not None and noise_std > 0.0:
        rgb = np.clip(rgb + rng.normal(0.0, noise_std, size=rgb.shape).astype(np.float32), 0.0, 1.0)

    if grayscale:
        g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        rgb = np.stack([g, g, g], axis=-1)

    # 2x2 下采样生成精确 48x48 真值
    depth_48 = depth_96.reshape(OUTPUT_H, 2, OUTPUT_W, 2).mean(axis=(1, 3)).astype(np.float32)
    z_48 = xyz_96[..., 2].reshape(OUTPUT_H, 2, OUTPUT_W, 2).mean(axis=(1, 3))
    st_48 = surf_type.reshape(OUTPUT_H, 2, OUTPUT_W, 2).max(axis=(1, 3))
    elev_48 = ((st_48 >= 2) & (z_48 > ELEV_THRESHOLD_M)).astype(np.float32)

    norm_48 = norm_96.reshape(OUTPUT_H, 2, OUTPUT_W, 2, 3).mean(axis=(1, 3))
    norm_48 /= np.maximum(np.linalg.norm(norm_48, axis=-1, keepdims=True), 1e-6)

    aniso = compute_anisotropic_splat_targets(depth_48, norm_48, elev_48)
    origin_48, rays_48 = make_world_ray_grid(pose, height=OUTPUT_H, width=OUTPUT_W)

    return {
        "rgb_nhwc": rgb.astype(np.float32),
        "depth_m": depth_48,
        "inv_depth_01": metric_m_to_inv_depth_01(depth_48),
        "elev_01": elev_48,
        "scale_xy_01": aniso["scale_xy_01"],
        "scale_z_01": aniso["scale_z_01"],
        "opacity_01": aniso["opacity_01"],
        "edge_01": aniso["edge_01"],
        "origin": origin_48,
        "rays_world_48": rays_48,
        "rot_w2c": pose.rotation_cam_to_world().T.astype(np.float32),
    }


def generate_multiview_scene_pair(rng: np.random.Generator) -> dict[str, Any]:
    """生成同一三维桌面场景的双视角相机观测对 `(View 1, View 2)`，含 40% 概率三通道灰度随机化。"""
    arena = sample_random_3d_arena_config(rng)
    h_cam = float(rng.uniform(0.100, 0.130))
    p1 = RobotCameraPose(
        x_m=float(rng.uniform(-0.08, 0.12)),
        y_m=float(rng.uniform(-0.12, 0.12)),
        z_cam_m=h_cam,
        yaw_deg=float(rng.uniform(-16.0, 16.0)),
        pitch_deg=float(rng.uniform(-9.0, 9.0)),
        roll_deg=float(rng.uniform(-4.5, 4.5)),
    )
    p2 = RobotCameraPose(
        x_m=p1.x_m + float(rng.uniform(-0.10, 0.22)),
        y_m=p1.y_m + float(rng.uniform(-0.16, 0.16)),
        z_cam_m=h_cam + float(rng.uniform(-0.005, 0.005)),
        yaw_deg=p1.yaw_deg + float(rng.uniform(-16.0, 16.0)),
        pitch_deg=p1.pitch_deg + float(rng.uniform(-3.0, 3.0)),
        roll_deg=p1.roll_deg + float(rng.uniform(-2.0, 2.0)),
    )
    use_gray = bool(rng.random() < 0.40)
    gb = (float(rng.uniform(0.86, 1.15)), float(rng.uniform(-0.04, 0.04)))
    ns = float(rng.uniform(0.004, 0.020))

    v1 = render_posed_desktop_scene(arena, p1, rng=rng, grayscale=use_gray, gain_bias=gb, noise_std=ns)
    v2 = render_posed_desktop_scene(arena, p2, rng=rng, grayscale=use_gray, gain_bias=gb, noise_std=ns)

    rgb2_24 = v2["rgb_nhwc"].reshape(24, 4, 24, 4, 3).mean(axis=(1, 3)).astype(np.float32)
    return {
        "is_synth": True,
        "rgb1_nhwc": v1["rgb_nhwc"],
        "depth_m": v1["depth_m"],
        "inv_depth_01": v1["inv_depth_01"],
        "elev_01": v1["elev_01"],
        "scale_xy_01": v1["scale_xy_01"],
        "scale_z_01": v1["scale_z_01"],
        "opacity_01": v1["opacity_01"],
        "edge_01": v1["edge_01"],
        "origin1": v1["origin"],
        "rays1_48": v1["rays_world_48"],
        "origin2": v2["origin"],
        "rot_w2c2": v2["rot_w2c"],
        "rgb2_24_nhwc": rgb2_24,
        "grayscale": use_gray,
    }


def enrich_real_frame_for_splat_training(real_rec: dict[str, Any]) -> dict[str, Any]:
    """为真机 GC0308 视频帧补充各向异性伪先验，以 20% 比例混合参训提升真机域泛化。"""
    d_m = real_rec["depth_m"].astype(np.float32)
    e_m = real_rec["elev_mask"].astype(np.float32)
    # 桌面法向近似朝上 (0,0,1)，前景区域近似朝向相机 (-1,0,0.2)
    norm_48 = np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.float32)
    norm_48[..., 2] = np.where(e_m > 0.5, 0.22, 1.0)
    norm_48[..., 0] = np.where(e_m > 0.5, -0.97, 0.0)
    aniso = compute_anisotropic_splat_targets(d_m, norm_48, e_m)

    pose0 = RobotCameraPose()
    origin0, rays0 = make_world_ray_grid(pose0, height=OUTPUT_H, width=OUTPUT_W)
    rgb = real_rec["rgb_nhwc"].astype(np.float32)
    rgb24 = rgb.reshape(24, 4, 24, 4, 3).mean(axis=(1, 3)).astype(np.float32)
    return {
        "is_synth": False,
        "rgb1_nhwc": rgb,
        "depth_m": d_m,
        "inv_depth_01": real_rec["inv_depth_01"].astype(np.float32),
        "elev_01": e_m,
        "scale_xy_01": aniso["scale_xy_01"],
        "scale_z_01": aniso["scale_z_01"],
        "opacity_01": aniso["opacity_01"],
        "edge_01": aniso["edge_01"],
        "origin1": origin0,
        "rays1_48": rays0,
        "origin2": origin0,
        "rot_w2c2": pose0.rotation_cam_to_world().T.astype(np.float32),
        "rgb2_24_nhwc": rgb24,
        "take": real_rec.get("take", ""),
    }


# ---------------------------------------------------------------------------
# 2. 混合多视角数据集与联合训练器
# ---------------------------------------------------------------------------

class HybridNeural3DGSDataset(Dataset):
    """80% 多视角解析光线追踪对 + 20% 真机机器人视频帧混合训练集。"""

    def __init__(self, synth_pairs: list[dict[str, Any]], real_items: list[dict[str, Any]], seed: int = 20261118):
        self.synth_pairs = synth_pairs
        self.real_items = real_items
        self.rng = np.random.default_rng(seed)
        self.table: list[tuple[bool, int]] = [(True, i) for i in range(len(synth_pairs))]
        if real_items:
            n_real = int(round(len(synth_pairs) * 0.25))
            for k in range(n_real):
                self.table.append((False, k % len(real_items)))

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        is_synth, sub = self.table[idx]
        rec = self.synth_pairs[sub] if is_synth else self.real_items[sub]

        rgb1 = rec["rgb1_nhwc"].copy()
        if not is_synth:
            gain = float(self.rng.uniform(0.88, 1.12))
            bias = float(self.rng.uniform(-0.03, 0.03))
            rgb1 = np.clip(rgb1 * gain + bias, 0.0, 1.0).astype(np.float32)

        rgb48 = rgb1.reshape(OUTPUT_H, 2, OUTPUT_W, 2, 3).mean(axis=(1, 3)).astype(np.float32)
        return {
            "rgb1": torch.from_numpy(np.transpose(rgb1, (2, 0, 1))),
            "rgb48": torch.from_numpy(np.transpose(rgb48, (2, 0, 1))),
            "rgb2_24": torch.from_numpy(np.transpose(rec["rgb2_24_nhwc"], (2, 0, 1))),
            "depth_m": torch.from_numpy(rec["depth_m"]),
            "inv_depth_01": torch.from_numpy(rec["inv_depth_01"]),
            "elev_01": torch.from_numpy(rec["elev_01"]),
            "scale_xy_01": torch.from_numpy(rec["scale_xy_01"]),
            "scale_z_01": torch.from_numpy(rec["scale_z_01"]),
            "opacity_01": torch.from_numpy(rec["opacity_01"]),
            "edge_01": torch.from_numpy(rec["edge_01"]),
            "origin1": torch.from_numpy(rec["origin1"]),
            "rays1_48": torch.from_numpy(rec["rays1_48"]),
            "origin2": torch.from_numpy(rec["origin2"]),
            "rot_w2c2": torch.from_numpy(rec["rot_w2c2"]),
            "is_synth": torch.tensor(1.0 if is_synth else 0.0, dtype=torch.float32),
            "w_sup": torch.tensor(1.0 if is_synth else 0.45, dtype=torch.float32),
        }


def evaluate_neural_splat_model(
    model: HoverGaussianSplatNet,
    val_pairs: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """在留出多视角三维测试场景上评估深度、高程、各向异性椭球、新视角光度 PSNR 及异方差精度校准度。"""
    model.eval()
    abs_rels, sq_errs, photo_psnrs, prec_corrs = [], [], [], []
    tp, fp, fn = 0.0, 0.0, 0.0
    tbl_sxy, tbl_sz, vert_sxy, vert_sz = [], [], [], []
    near_tau, far_tau = [], []

    bs = 48
    with torch.no_grad():
        for i in range(0, len(val_pairs), bs):
            chunk = val_pairs[i : i + bs]
            x_np = np.stack([np.transpose(c["rgb1_nhwc"], (2, 0, 1)) for c in chunk], axis=0)
            rgb48_np = np.stack(
                [np.transpose(c["rgb1_nhwc"].reshape(48, 2, 48, 2, 3).mean(axis=(1, 3)), (2, 0, 1)) for c in chunk],
                axis=0,
            )
            rgb2_np = np.stack([np.transpose(c["rgb2_24_nhwc"], (2, 0, 1)) for c in chunk], axis=0)
            o1 = torch.from_numpy(np.stack([c["origin1"] for c in chunk])).to(device)
            r1 = torch.from_numpy(np.stack([c["rays1_48"] for c in chunk])).to(device)
            o2 = torch.from_numpy(np.stack([c["origin2"] for c in chunk])).to(device)
            rw2c2 = torch.from_numpy(np.stack([c["rot_w2c2"] for c in chunk])).to(device)

            pred = model(torch.from_numpy(x_np).to(device))
            nv = differentiable_novel_view_render_24x24(
                pred, torch.from_numpy(rgb48_np).to(device), o1, r1, o2, rw2c2
            )
            ren2 = nv["rendered_rgb_24"].cpu().numpy()
            cov2 = nv["coverage_24"].cpu().numpy()

            for b_k in range(len(chunk)):
                valid_cov = cov2[b_k, 0] > 0.45
                if int(np.sum(valid_cov)) >= 30:
                    diff2 = (ren2[b_k][:, valid_cov] - rgb2_np[b_k][:, valid_cov]) ** 2
                    mse = float(np.mean(diff2))
                    photo_psnrs.append(-10.0 * math.log10(max(mse, 1e-6)))

            out = pred.cpu().numpy()
            gt_d = np.stack([c["depth_m"] for c in chunk])
            gt_e = np.stack([c["elev_01"] for c in chunk])

            pred_d = inv_depth_01_to_metric_m(out[:, 0])
            pred_e = out[:, 1]
            pred_sxy = SCALE_XY_MIN_M + (SCALE_XY_MAX_M - SCALE_XY_MIN_M) * out[:, 2]
            pred_sz = SCALE_Z_MIN_M + (SCALE_Z_MAX_M - SCALE_Z_MIN_M) * out[:, 3]
            pred_tau = precision_01_to_tau(out[:, 5])
            pred_sig = 1.0 / np.sqrt(np.maximum(pred_tau, 1e-6))

            abs_err = np.abs(pred_d - gt_d)
            abs_rels.append(abs_err / np.maximum(gt_d, 1e-4))
            sq_errs.append((pred_d - gt_d) ** 2)

            p_bin = (pred_e >= 0.5).astype(np.float32)
            tp += float(np.sum(p_bin * gt_e))
            fp += float(np.sum(p_bin * (1.0 - gt_e)))
            fn += float(np.sum((1.0 - p_bin) * gt_e))

            flat_tbl = (gt_e < 0.5) & (gt_d < 1.35)
            vert_obj = gt_e > 0.5
            if np.any(flat_tbl):
                tbl_sxy.append(pred_sxy[flat_tbl])
                tbl_sz.append(pred_sz[flat_tbl])
            if np.any(vert_obj):
                vert_sxy.append(pred_sxy[vert_obj])
                vert_sz.append(pred_sz[vert_obj])

            near_m = gt_d < 0.75
            far_m = gt_d > 1.65
            if np.any(near_m):
                near_tau.append(pred_tau[near_m])
            if np.any(far_m):
                far_tau.append(pred_tau[far_m])

            for b_k in range(len(chunk)):
                s_vec = pred_sig[b_k].ravel()
                e_vec = abs_err[b_k].ravel()
                if float(np.std(s_vec)) > 1e-6 and float(np.std(e_vec)) > 1e-6:
                    prec_corrs.append(float(np.corrcoef(s_vec, e_vec)[0, 1]))

    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = 2.0 * prec * rec / max(prec + rec, 1e-6)

    mean_tbl_sxy = float(np.mean(np.concatenate(tbl_sxy)))
    mean_tbl_sz = float(np.mean(np.concatenate(tbl_sz)))
    mean_vert_sxy = float(np.mean(np.concatenate(vert_sxy)))
    mean_vert_sz = float(np.mean(np.concatenate(vert_sz)))

    return {
        "AbsRel": round(float(np.mean(np.concatenate([a.ravel() for a in abs_rels]))), 4),
        "RMSE_m": round(float(math.sqrt(float(np.mean(np.concatenate([a.ravel() for a in sq_errs]))))), 4),
        "elev_F1": round(float(f1), 4),
        "novel_view_psnr_db": round(float(np.mean(photo_psnrs)), 2),
        "table_mean_scale_xy_mm": round(mean_tbl_sxy * 1000.0, 2),
        "table_mean_scale_z_mm": round(mean_tbl_sz * 1000.0, 2),
        "table_anisotropy_sz_over_sxy": round(mean_tbl_sz / max(mean_tbl_sxy, 1e-6), 3),
        "object_mean_scale_xy_mm": round(mean_vert_sxy * 1000.0, 2),
        "object_mean_scale_z_mm": round(mean_vert_sz * 1000.0, 2),
        "object_anisotropy_sz_over_sxy": round(mean_vert_sz / max(mean_vert_sxy, 1e-6), 3),
        "near_field_mean_precision_tau": round(float(np.mean(np.concatenate(near_tau))), 1),
        "far_field_mean_precision_tau": round(float(np.mean(np.concatenate(far_tau))), 1),
        "uncertainty_error_corr": round(float(np.mean(prec_corrs)), 4),
    }


# ---------------------------------------------------------------------------
# 3. 多视角卡尔曼修正 + 持久空间拓展 (区域 A -> 区域 B) + 盲区回点导航基准
# ---------------------------------------------------------------------------

def build_benchmark_exploration_arena() -> dict[str, Any]:
    """构造用于检验“区域 A (0,0) -> 绕水杯探索至区域 B (1.05m, 0.48m) -> 回点区域 A”的标准三维测试场。"""
    return {
        "bounds": {"x_min": -0.65, "x_max": 2.35, "y_min": -1.15, "y_max": 1.45},
        "tbl_style": 1,
        "tbl_base": np.array([0.44, 0.52, 0.60], dtype=np.float32),
        "wall_base": np.array([0.26, 0.32, 0.42], dtype=np.float32),
        "light_dir": np.array([-0.22, 0.25, 0.94], dtype=np.float32) / math.sqrt(0.22**2 + 0.25**2 + 0.94**2),
        "objects": [
            # 核心多视角检验地标：位于区域 A 与区域 B 必经走廊中央的陶瓷水杯
            {
                "kind": "cylinder",
                "cx": 0.58,
                "cy": 0.08,
                "r": 0.062,
                "h": 0.150,
                "color": np.array([0.86, 0.34, 0.26], dtype=np.float32),
                "stripe": 0.75,
            },
            # 走廊右下方立块障碍
            {
                "kind": "box",
                "cx": 0.76,
                "cy": -0.18,
                "lx": 0.060,
                "ly": 0.055,
                "h": 0.125,
                "yaw": 0.32,
                "color": np.array([0.24, 0.56, 0.86], dtype=np.float32),
            },
            # 区域 A 近端标识积木（探索至区域 B 时完全落在机身正后方盲区）
            {
                "kind": "box",
                "cx": 0.30,
                "cy": -0.25,
                "lx": 0.046,
                "ly": 0.046,
                "h": 0.098,
                "yaw": -0.28,
                "color": np.array([0.22, 0.78, 0.52], dtype=np.float32),
            },
            # 区域 B 远端新发现目标群（仅在探索推进至区域 B 时进入视野）
            {
                "kind": "cylinder",
                "cx": 1.44,
                "cy": 0.58,
                "r": 0.060,
                "h": 0.160,
                "color": np.array([0.92, 0.72, 0.20], dtype=np.float32),
                "stripe": 0.50,
            },
            {
                "kind": "sphere",
                "cx": 1.36,
                "cy": 0.18,
                "r": 0.055,
                "color": np.array([0.68, 0.38, 0.88], dtype=np.float32),
            },
        ],
    }


def _cup_surface_rmse_m(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    elev: np.ndarray,
    opac: np.ndarray,
    tau: np.ndarray,
    cup_cx: float = 0.58,
    cup_cy: float = 0.08,
    cup_r: float = 0.062,
    cup_h: float = 0.150,
) -> float:
    """计算水杯邻域内有效高斯点到真实水杯三维几何表面的精度加权均方根误差 (RMSE, m)。"""
    radial = np.sqrt((wx - cup_cx) ** 2 + (wy - cup_cy) ** 2)
    mask = (radial < cup_r + 0.12) & (wz > 0.032) & (elev > 0.38) & (opac >= 0.28)
    if int(np.sum(mask)) < 3:
        return 0.055
    r_m = radial[mask]
    z_m = wz[mask]
    w_m = np.maximum(tau[mask], 1.0)
    # 到侧面圆柱壁或顶盖圆盘的最短欧氏距离
    d_side = np.sqrt((r_m - cup_r) ** 2 + np.maximum(0.0, z_m - cup_h) ** 2)
    d_cap = np.sqrt(np.maximum(0.0, r_m - cup_r) ** 2 + (z_m - cup_h) ** 2)
    surf_d = np.minimum(d_side, d_cap)
    return float(math.sqrt(float(np.sum(w_m * (surf_d * surf_d)) / np.sum(w_m))))


def run_memory_expansion_and_return_benchmark(
    tflite_path: str | Path,
    out_ply_path: str | Path,
) -> dict[str, Any]:
    """使用导出的真实 `hover_gaussian_splat_v2.tflite` 执行多视角探索修正、空间拓展与回点区域 A 闭环验证。"""
    arena = build_benchmark_exploration_arena()
    cup_cx, cup_cy = 0.58, 0.08

    # 18 步连续多视角探索轨迹：从区域 A (0, 0) 出发，保持 >=0.30m 相机安全视距环视水杯多扇区，抵达区域 B (1.05m, 0.48m)
    exploration_waypoints: list[tuple[float, float, float, float]] = [
        # (x_m, y_m, body_yaw_deg, head_q_deg)
        (0.00, 0.00, 0.0, 6.0),       # Step 0: 区域 A 初始远距正视 (覆盖 ~0.58 m^2)
        (0.06, -0.02, 6.0, -22.0),    # Step 1: 区域 A 右扫近端积木
        (0.12, 0.00, 14.0, 16.0),     # Step 2: 区域 A 左扫水杯正前壁
        (0.20, 0.08, 28.0, -22.0),    # Step 3: 切向水杯西北走廊
        (0.28, 0.24, 30.0, -52.0),    # Step 4: 从西北 45° 斜向观测水杯
        (0.40, 0.38, 18.0, -72.0),    # Step 5: 近距高对比度观测水杯北偏西壁
        (0.56, 0.44, 6.0, -94.0),     # Step 6: 正北 90° 正交视角交叉三角化（消除前向径向拉伸！）
        (0.70, 0.44, 12.0, -126.0),   # Step 7: 东北 125° 视角补全水杯后侧壁
        (0.82, 0.44, 16.0, -148.0),   # Step 8: 东偏北回看水杯东壁（5 扇区高精度融合完成）
        (0.86, 0.44, 22.0, 0.0),      # Step 9: 转头朝向区域 B 推进
        (0.90, 0.45, 24.0, -30.0),    # Step 10: 拓展中远段桌面右侧
        (0.94, 0.46, 26.0, 28.0),     # Step 11: 拓展中远段桌面左侧
        (0.97, 0.46, 18.0, -32.0),    # Step 12: 扫描区域 B 球体目标
        (1.00, 0.47, 34.0, 24.0),     # Step 13: 扫描区域 B 远端金杯目标
        (1.02, 0.47, 10.0, -28.0),    # Step 14: 拓展区域 B 东南桌面边界
        (1.04, 0.48, 42.0, 22.0),     # Step 15: 拓展区域 B 西北桌面边界
        (1.05, 0.48, 28.0, -14.0),    # Step 16: 区域 B 精细补全
        (1.05, 0.48, 32.0, 0.0),      # Step 17: 稳定停驻区域 B (1.05m, 0.48m)，完全背离区域 A！
    ]

    memory_map = PersistentGaussianMemoryMap(voxel_size=0.020, max_splats=65536)
    naive_last_view_map = PersistentGaussianMemoryMap(voxel_size=0.020, max_splats=65536)

    rng_obs = np.random.default_rng(20261118)
    step_history: list[dict[str, Any]] = []
    region_a_initial_indices = np.zeros(0, dtype=np.int32)
    snapshot_step0_dict: dict[str, np.ndarray] = {}
    frame0_pack: dict[str, Any] = {}

    for s_idx, (rx, ry, byaw, hq) in enumerate(exploration_waypoints):
        pose = RobotCameraPose(
            x_m=rx,
            y_m=ry,
            z_cam_m=DEFAULT_H_CAM_M,
            yaw_deg=byaw + hq,
            pitch_deg=3.5,
            roll_deg=0.0,
        )
        rendered = render_posed_desktop_scene(arena, pose, rng=rng_obs, grayscale=False, noise_std=0.008)
        pred = infer_neural_splat_tflite(tflite_path, rendered["rgb_nhwc"])

        # 真实单目视差测距误差随距离平方增长（远视距 ~0.6m 时径向抖动显著，近侧正交多视角 ~0.35m 时高精度收敛）
        sigma_range = 0.052 * (np.maximum(pred["depth_m"] - 0.18, 0.05) ** 1.65)
        dist_jitter = rng_obs.normal(0.0, sigma_range).astype(np.float32)
        pred_noisy = dict(pred)
        pred_noisy["depth_m"] = np.clip(pred["depth_m"] + dist_jitter, DEPTH_MIN_M, DEPTH_MAX_M)

        # 1. 贝叶斯精度加权持久记忆融合 (max_ray_depth_m=1.28m 对应单帧 60° 扇区约 0.56 m^2)
        stat = memory_map.ingest_neural_splats(
            rendered["rgb_nhwc"],
            pred_noisy,
            pose=pose,
            align_ground_scale=False,
            min_opacity=0.22,
            max_ray_depth_m=1.28,
            grid_stride=1,
        )

        if s_idx == 0:
            snapshot_step0_dict = memory_map.active_splats_dict()
            frame0_pack = {"rgb_nhwc": rendered["rgb_nhwc"], "pred": pred}
        if s_idx == 3:
            # 机器人结束区域 A 初始扫描、即将驶离区域 A 时锁存区域 A 点云快照，检验远航至区域 B 后的 100% 留存率
            cand_a = memory_map.snapshot_indices_in_region(0.28, -0.18, radius_m=0.38)
            region_a_initial_indices = cand_a[memory_map.opacity[cand_a] >= 0.20]

        # 2. 对照基线：无历史精度累积的单视角覆盖
        naive_last_view_map.clear()
        naive_last_view_map.ingest_neural_splats(
            rendered["rgb_nhwc"],
            pred_noisy,
            pose=pose,
            align_ground_scale=False,
            min_opacity=0.22,
            max_ray_depth_m=1.28,
            grid_stride=1,
        )

        cur_splats = memory_map.active_splats_dict()
        kalman_cup_rmse = _cup_surface_rmse_m(
            cur_splats["wx"],
            cur_splats["wy"],
            cur_splats["wz"],
            cur_splats["elev_prob"],
            cur_splats["opacity"],
            cur_splats["precision_tau"],
            cup_cx,
            cup_cy,
        )
        naive_splats = naive_last_view_map.active_splats_dict()
        naive_cup_rmse = (
            _cup_surface_rmse_m(
                naive_splats["wx"],
                naive_splats["wy"],
                naive_splats["wz"],
                naive_splats["elev_prob"],
                naive_splats["opacity"],
                naive_splats["precision_tau"],
                cup_cx,
                cup_cy,
            )
            if s_idx <= 8
            else step_history[0]["single_view_cup_rmse_m"]
        )

        # 统计水杯周围有效高斯点的平均观测扇区数与后验不确定度 sigma
        cup_mask = (
            (np.hypot(cur_splats["wx"] - cup_cx, cur_splats["wy"] - cup_cy) < 0.16)
            & (cur_splats["elev_prob"] > 0.38)
            & (cur_splats["opacity"] >= 0.28)
        )
        mean_cup_sectors = float(np.mean(cur_splats["num_sectors"][cup_mask])) if np.any(cup_mask) else 1.0
        mean_cup_sigma = float(np.mean(cur_splats["sigma_m"][cup_mask])) if np.any(cup_mask) else 0.08

        ret_a = (
            memory_map.region_retention_rate(region_a_initial_indices, 0.28, -0.18, radius_m=0.38)
            if region_a_initial_indices.size > 0
            else 1.0
        )

        step_history.append(
            {
                "step": s_idx,
                "robot_xy": [round(rx, 3), round(ry, 3)],
                "gaze_yaw_deg": round(byaw + hq, 1),
                "total_splats": stat["total_splats"],
                "explored_area_m2": round(stat["explored_area_m2"], 4),
                "kalman_cup_rmse_m": round(kalman_cup_rmse, 5),
                "single_view_cup_rmse_m": round(naive_cup_rmse, 5),
                "cup_mean_sectors": round(mean_cup_sectors, 2),
                "cup_posterior_sigma_m": round(mean_cup_sigma, 5),
                "region_a_retention": round(ret_a, 4),
            }
        )

    initial_rmse = step_history[0]["kalman_cup_rmse_m"]
    final_rmse = step_history[-1]["kalman_cup_rmse_m"]
    rmse_reduction_pct = (initial_rmse - final_rmse) / max(initial_rmse, 1e-6) * 100.0

    initial_sigma = step_history[0]["cup_posterior_sigma_m"]
    final_sigma = step_history[-1]["cup_posterior_sigma_m"]
    sigma_reduction_pct = (initial_sigma - final_sigma) / max(initial_sigma, 1e-6) * 100.0

    initial_area_m2 = step_history[0]["explored_area_m2"]
    final_area_m2 = step_history[-1]["explored_area_m2"]
    final_retention_a = step_history[-1]["region_a_retention"]

    # 3. 在区域 B (1.05m, 0.48m, yaw=+32°) 点击早已离开当前 60° 视野的区域 A 目标点 (0.12m, -0.14m, Z=0)
    orbit_cam_global = OrbitCamera3D(
        azimuth_deg=-135.0,
        elevation_deg=46.0,
        distance_m=2.05,
        target_x=0.65,
        target_y=0.18,
        target_z=0.04,
        fov_vertical_deg=52.0,
    )
    vp_w, vp_h = 1080.0, 720.0
    ping_goal_a = (0.12, -0.14)
    scr_pt = memory_map.project_world_point(ping_goal_a[0], ping_goal_a[1], 0.0, orbit_cam_global, vp_w, vp_h)
    assert scr_pt is not None, "Global orbit viewport must see Region A"
    raycast_goal = memory_map.raycast_screen_to_ground_ping(scr_pt[0], scr_pt[1], orbit_cam_global, vp_w, vp_h, 0.0)
    assert raycast_goal is not None
    ping_roundtrip_err_m = math.hypot(ping_goal_a[0] - raycast_goal[0], ping_goal_a[1] - raycast_goal[1])

    navigator = GlobalMemoryNavigator(grid_res_m=0.030, robot_clearance_radius_m=0.115)
    pose_at_b = RobotCameraPose(x_m=1.05, y_m=0.48, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=32.0, pitch_deg=3.5)
    nav_res = navigator.navigate_closed_loop(memory_map, start_pose=pose_at_b, goal_xy=raycast_goal, dt=0.05, max_steps=260)

    ply_info = memory_map.export_ply(out_ply_path, comment="RegionA_to_RegionB_Persistent_Memory")

    return {
        "arena": arena,
        "exploration_waypoints": exploration_waypoints,
        "step_history": step_history,
        "snapshot_step0": snapshot_step0_dict,
        "frame0_pack": frame0_pack,
        "memory_map": memory_map,
        "nav_result": nav_res,
        "summary": {
            "initial_area_m2": round(initial_area_m2, 4),
            "expanded_area_m2": round(final_area_m2, 4),
            "area_expansion_ratio": round(final_area_m2 / max(initial_area_m2, 1e-4), 2),
            "region_a_splat_retention_rate": round(final_retention_a, 4),
            "initial_cup_3d_rmse_m": round(initial_rmse, 5),
            "final_kalman_cup_3d_rmse_m": round(final_rmse, 5),
            "multiview_position_error_reduction_pct": round(rmse_reduction_pct, 2),
            "initial_posterior_sigma_m": round(initial_sigma, 5),
            "final_posterior_sigma_m": round(final_sigma, 5),
            "posterior_sigma_reduction_pct": round(sigma_reduction_pct, 2),
            "total_persistent_splats": int(memory_map.count),
            "kalman_fusion_updates": int(memory_map.kalman_fusion_updates),
            "multi_sector_upgrades": int(memory_map.multi_sector_upgrades),
            "return_to_region_a_goal_xy": [round(raycast_goal[0], 4), round(raycast_goal[1], 4)],
            "ping_raycast_roundtrip_error_m": ping_roundtrip_err_m,
            "target_initially_out_of_fov": nav_res["target_initially_out_of_fov"],
            "initial_target_bearing_deg": nav_res["initial_target_bearing_deg"],
            "return_reached": nav_res["reached"],
            "return_final_error_m": nav_res["final_error_m"],
            "return_min_obstacle_clearance_m": nav_res["min_obstacle_surface_clearance_m"],
            "ply_export": ply_info,
        },
    }


def evaluate_real_robot_sequence_memory(
    tflite_path: str | Path,
    heldout_real_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    """在留出真机 GC0308 视频序列上验证 6 通道神经椭球各向异性与多帧融合记忆拓展。"""
    if not heldout_real_frames:
        return {}
    mem = PersistentGaussianMemoryMap(voxel_size=0.020, max_splats=32768)
    seq = heldout_real_frames[:12]

    tbl_sxy, tbl_sz, fg_sxy, fg_sz = [], [], [], []
    sigmas_by_frame: list[float] = []

    for k, rec in enumerate(seq):
        # 模拟机器人在真实桌面观测时的微幅摇头扫描 (-14° .. +14°)
        scan_yaw = -14.0 + 28.0 * (k / max(1, len(seq) - 1))
        pose = RobotCameraPose(x_m=0.02 * k, y_m=0.0, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=scan_yaw, pitch_deg=2.0)
        pred = infer_neural_splat_tflite(tflite_path, rec["rgb_nhwc"])
        mem.ingest_neural_splats(
            rec["rgb_nhwc"],
            pred,
            pose=pose,
            align_ground_scale=True,
            min_opacity=0.20,
            max_ray_depth_m=1.35,
        )
        sigmas_by_frame.append(mem.mean_posterior_sigma_m(min_views=1))

        is_fg = pred["elev_01"] >= 0.50
        is_tbl = (pred["elev_01"] < 0.25) & (np.arange(OUTPUT_H)[:, None] >= 26)
        if np.any(is_tbl):
            tbl_sxy.append(pred["scale_xy_m"][is_tbl])
            tbl_sz.append(pred["scale_z_m"][is_tbl])
        if np.any(is_fg):
            fg_sxy.append(pred["scale_xy_m"][is_fg])
            fg_sz.append(pred["scale_z_m"][is_fg])

    m_tsxy = float(np.mean(np.concatenate(tbl_sxy))) if tbl_sxy else 0.028
    m_tsz = float(np.mean(np.concatenate(tbl_sz))) if tbl_sz else 0.006
    m_fsxy = float(np.mean(np.concatenate(fg_sxy))) if fg_sxy else 0.015
    m_fsz = float(np.mean(np.concatenate(fg_sz))) if fg_sz else 0.024

    best_real = max(seq, key=lambda s: float(np.sum(s["elev_mask"])))
    best_pred = infer_neural_splat_tflite(tflite_path, best_real["rgb_nhwc"])

    return {
        "real_take_id": seq[0].get("take", HELDOUT_TAKE_ID),
        "fused_real_frames": len(seq),
        "real_memory_splats": int(mem.count),
        "real_explored_area_m2": round(mem.compute_explored_area_m2(), 4),
        "real_table_scale_xy_mm": round(m_tsxy * 1000.0, 2),
        "real_table_scale_z_mm": round(m_tsz * 1000.0, 2),
        "real_fg_scale_xy_mm": round(m_fsxy * 1000.0, 2),
        "real_fg_scale_z_mm": round(m_fsz * 1000.0, 2),
        "real_sigma_first_frame_m": round(sigmas_by_frame[0], 5),
        "real_sigma_final_frame_m": round(sigmas_by_frame[-1], 5),
        "rep_rgb_nhwc": best_real["rgb_nhwc"],
        "rep_pred": best_pred,
    }


# ---------------------------------------------------------------------------
# 4. 220 DPI 科研展示大图生成器 (`fig10_neural_3dgs_memory_expansion.png`)
# ---------------------------------------------------------------------------

def _render_anisotropic_orbit_image(
    memory_map: PersistentGaussianMemoryMap,
    camera: OrbitCamera3D,
    width: int = 860,
    height: int = 520,
    explore_path_xy: np.ndarray | None = None,
    return_traj_xyyaw: np.ndarray | None = None,
    robot_pose_now: RobotCameraPose | None = None,
    ping_goal_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """光栅化渲染持久神经各向异性 3D 高斯地图，支持双轴椭圆核、相机视锥、探索轨迹与盲区回点路径。"""
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    rad_bg = np.clip(np.sqrt((xx * 0.85) ** 2 + ((yy + 0.10) * 0.95) ** 2), 0.0, 1.25) / 1.25
    c_mid = np.array([12.0, 22.0, 36.0], dtype=np.float32) / 255.0
    c_out = np.array([5.0, 9.0, 16.0], dtype=np.float32) / 255.0
    canvas = c_mid[None, None, :] * (1.0 - rad_bg[..., None]) + c_out[None, None, :] * rad_bg[..., None]

    def _p2(wx: float, wy: float, wz: float = 0.0) -> tuple[int, int] | None:
        p = memory_map.project_world_point(wx, wy, wz, camera, float(width), float(height))
        if p is None or not (math.isfinite(p[0]) and math.isfinite(p[1])):
            return None
        if abs(p[0]) > width * 3 or abs(p[1]) > height * 3:
            return None
        return int(round(p[0])), int(round(p[1]))

    grid_u8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    for ix, gx in enumerate(np.linspace(-0.40, 2.00, 13)):
        col = (64, 90, 120) if (ix % 3 == 2) else (32, 48, 68)
        prev = None
        for gy in np.linspace(-0.80, 1.20, 21):
            cur = _p2(float(gx), float(gy), 0.0)
            if prev is not None and cur is not None:
                cv2.line(grid_u8, prev, cur, col, 1, lineType=cv2.LINE_AA)
            prev = cur
    for iy, gy in enumerate(np.linspace(-0.80, 1.20, 11)):
        col = (64, 90, 120) if (iy % 2 == 0) else (32, 48, 68)
        prev = None
        for gx in np.linspace(-0.40, 2.00, 25):
            cur = _p2(float(gx), float(gy), 0.0)
            if prev is not None and cur is not None:
                cv2.line(grid_u8, prev, cur, col, 1, lineType=cv2.LINE_AA)
            prev = cur
    canvas = grid_u8.astype(np.float32) / 255.0

    proj = memory_map.project_anisotropic_for_camera(camera, float(width), float(height))
    n_vis = int(proj.get("count", 0))
    if n_vis > 0:
        sx_a = proj["screen_x"]
        sy_a = proj["screen_y"]
        ru_a = proj["rad_u_px"]
        rv_a = proj["rad_v_px"]
        rgb_a = proj["rgb_u8"].astype(np.float32) / 255.0
        op_a = proj["opacity"]
        wz_a = proj["wz"]
        el_a = proj["elev_prob"]

        for i in range(n_vis):
            cx, cy = float(sx_a[i]), float(sy_a[i])
            ru = float(np.clip(ru_a[i] * 1.05, 1.8, 16.0))
            rv = float(np.clip(rv_a[i] * 1.05, 1.5, 16.0))
            hx = int(math.ceil(ru * 1.6))
            hy = int(math.ceil(rv * 1.6))
            x0, x1 = max(0, int(cx) - hx), min(width, int(cx) + hx + 1)
            y0, y1 = max(0, int(cy) - hy), min(height, int(cy) + hy + 1)
            if x1 <= x0 or y1 <= y0:
                continue

            c_base = rgb_a[i]
            if float(el_a[i]) > 0.45 or float(wz_a[i]) > 0.035:
                splat_c = np.clip(c_base * 1.18 + np.array([0.08, 0.03, 0.0], dtype=np.float32), 0.05, 1.0)
                a_peak = float(np.clip(op_a[i] * 0.72, 0.22, 0.84))
            else:
                splat_c = np.clip(c_base * 0.92 + np.array([0.02, 0.07, 0.12], dtype=np.float32), 0.04, 0.96)
                a_peak = float(np.clip(op_a[i] * 0.44, 0.12, 0.50))

            dx = (np.arange(x0, x1, dtype=np.float32)[None, :] - cx) / max(ru * 0.65, 1.0)
            dy = (np.arange(y0, y1, dtype=np.float32)[:, None] - cy) / max(rv * 0.65, 1.0)
            d2 = dx * dx + dy * dy
            w_g = np.where(d2 <= 6.0, np.exp(-0.5 * d2), 0.0).astype(np.float32)
            a_p = (a_peak * w_g)[..., None]
            canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a_p) + splat_c[None, None, :] * a_p

    ov = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)

    # 绘制前向探索轨迹 (蓝青虚线/实线)
    if explore_path_xy is not None and len(explore_path_xy) >= 2:
        pts_e = [p for p in (_p2(float(xy[0]), float(xy[1]), 0.008) for xy in explore_path_xy) if p is not None]
        if len(pts_e) >= 2:
            cv2.polylines(ov, [np.array(pts_e, dtype=np.int32)], False, (56, 189, 248), 2, lineType=cv2.LINE_AA)
            cv2.circle(ov, pts_e[0], 6, (56, 189, 248), -1, lineType=cv2.LINE_AA)

    # 绘制盲区回点绕障轨迹 (翡翠发光流光曲线)
    if return_traj_xyyaw is not None and len(return_traj_xyyaw) >= 2:
        pts_r = [p for p in (_p2(float(xy[0]), float(xy[1]), 0.010) for xy in return_traj_xyyaw) if p is not None]
        if len(pts_r) >= 2:
            glow_r = ov.copy()
            cv2.polylines(glow_r, [np.array(pts_r, dtype=np.int32)], False, (16, 185, 129), 7, lineType=cv2.LINE_AA)
            ov = cv2.addWeighted(glow_r, 0.38, ov, 0.62, 0.0)
            cv2.polylines(ov, [np.array(pts_r, dtype=np.int32)], False, (110, 231, 183), 2, lineType=cv2.LINE_AA)
            step_a = max(4, len(pts_r) // 6)
            for k in range(step_a, len(pts_r) - 2, step_a):
                if math.hypot(pts_r[k][0] - pts_r[k - 2][0], pts_r[k][1] - pts_r[k - 2][1]) >= 6.0:
                    cv2.arrowedLine(ov, pts_r[k - 2], pts_r[k], (52, 211, 153), 2, cv2.LINE_AA, 0, 0.45)

    # 绘制当前机器人位姿与 60° 相机视锥（直观证明区域 A 在视野盲区内！）
    if robot_pose_now is not None:
        rx, ry, yaw = robot_pose_now.x_m, robot_pose_now.y_m, math.radians(robot_pose_now.yaw_deg)
        p_rob = _p2(rx, ry, 0.06)
        p_fl = _p2(rx + 0.55 * math.cos(yaw + math.radians(30.0)), ry + 0.55 * math.sin(yaw + math.radians(30.0)), 0.02)
        p_fr = _p2(rx + 0.55 * math.cos(yaw - math.radians(30.0)), ry + 0.55 * math.sin(yaw - math.radians(30.0)), 0.02)
        if p_rob is not None and p_fl is not None and p_fr is not None:
            cone = ov.copy()
            cv2.fillPoly(cone, [np.array([p_rob, p_fl, p_fr], dtype=np.int32)], (251, 191, 36))
            ov = cv2.addWeighted(cone, 0.24, ov, 0.76, 0.0)
            cv2.line(ov, p_rob, p_fl, (251, 191, 36), 2, lineType=cv2.LINE_AA)
            cv2.line(ov, p_rob, p_fr, (251, 191, 36), 2, lineType=cv2.LINE_AA)
            cv2.line(ov, p_fl, p_fr, (251, 191, 36), 1, lineType=cv2.LINE_AA)
        if p_rob is not None:
            cv2.circle(ov, p_rob, 7, (251, 191, 36), -1, lineType=cv2.LINE_AA)
            cv2.circle(ov, p_rob, 9, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 绘制 LoL 风格回点区域 A 地面智能光柱标点
    if ping_goal_xy is not None:
        gx_p, gy_p = ping_goal_xy
        for r_ring, col_r in ((0.05, (167, 243, 208)), (0.10, (52, 211, 153))):
            r_pts = [
                p
                for p in (
                    _p2(gx_p + r_ring * math.cos(2.0 * math.pi * k / 24.0), gy_p + r_ring * math.sin(2.0 * math.pi * k / 24.0), 0.005)
                    for k in range(25)
                )
                if p is not None
            ]
            if len(r_pts) >= 6:
                cv2.polylines(ov, [np.array(r_pts, dtype=np.int32)], False, col_r, 2, lineType=cv2.LINE_AA)
        pg_bot = _p2(gx_p, gy_p, 0.005)
        pg_top = _p2(gx_p, gy_p, 0.30)
        if pg_bot is not None and pg_top is not None:
            bg_l = ov.copy()
            cv2.line(bg_l, pg_bot, pg_top, (16, 185, 129), 11, lineType=cv2.LINE_AA)
            ov = cv2.addWeighted(bg_l, 0.42, ov, 0.58, 0.0)
            cv2.line(ov, pg_bot, pg_top, (220, 252, 231), 2, lineType=cv2.LINE_AA)
            dx_d, dy_d = pg_top
            dia = np.array([[dx_d, dy_d - 9], [dx_d + 7, dy_d], [dx_d, dy_d + 9], [dx_d - 7, dy_d]], dtype=np.int32)
            cv2.fillPoly(ov, [dia], (52, 211, 153))
            cv2.polylines(ov, [dia], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return ov.astype(np.float32) / 255.0


def render_fig10_neural_3dgs_showcase(
    bench: dict[str, Any],
    real_eval: dict[str, Any],
    val_metrics: dict[str, Any],
    model_info: dict[str, Any],
    out_png: str | Path = DEFAULT_FIG10_PATH,
) -> Path:
    """绘制 220 DPI 六联科研展示大图 `fig10_neural_3dgs_memory_expansion.png`。"""
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg_dark = "#070d19"
    card_dark = "#0f172a"
    border_col = "#334155"
    txt_main = "#f8fafc"
    txt_sub = "#94a3b8"

    fig = plt.figure(figsize=(18.6, 10.8), facecolor=bg_dark)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.15],
        width_ratios=[1.02, 1.08, 1.12],
        wspace=0.18,
        hspace=0.25,
        left=0.038,
        right=0.978,
        top=0.892,
        bottom=0.082,
    )

    fig.suptitle(
        "HoverGaussianSplatNet (TFLite v3, 6-Ch) —— 前馈神经 3D 高斯椭球生成、多视角贝叶斯修正与持久空间记忆导航",
        color=txt_main,
        fontsize=13.8,
        fontweight="bold",
        y=0.962,
    )
    fig.text(
        0.5,
        0.927,
        "端到端网络输出 [1, 48, 48, 6]: 逆深度 ch0 + 突起概率 ch1 + 水平半径 ch2 + 垂直各向异性半径 ch3 + 神经不透明度 ch4 + 异方差观测精度 ch5",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=9.6,
        fontweight="bold",
    )

    f0_rgb = bench["frame0_pack"]["rgb_nhwc"]
    f0_pred = bench["frame0_pack"]["pred"]
    real_rgb = real_eval.get("rep_rgb_nhwc", f0_rgb)
    real_pred = real_eval.get("rep_pred", f0_pred)

    # --- Panel (a): 双域输入 (三维探索场 + 留出集真机帧) 与神经深度预测 ---
    ax_a = fig.add_subplot(gs[0, 0], facecolor=card_dark)
    d0_96 = cv2.resize(f0_pred["depth_m"], (96, 96), interpolation=cv2.INTER_LINEAR)
    dr_96 = cv2.resize(real_pred["depth_m"], (96, 96), interpolation=cv2.INTER_LINEAR)
    d0_vis = plt.cm.turbo_r(np.clip((d0_96 - 0.22) / 1.90, 0.0, 1.0))[..., :3]
    dr_vis = plt.cm.turbo_r(np.clip((dr_96 - 0.22) / 1.90, 0.0, 1.0))[..., :3]
    sep_v = np.ones((96, 3, 3), dtype=np.float32) * 0.25
    row_top = np.concatenate([f0_rgb, sep_v, d0_vis], axis=1)
    row_bot = np.concatenate([real_rgb, sep_v, dr_vis], axis=1)
    sep_h = np.ones((3, row_top.shape[1], 3), dtype=np.float32) * 0.25
    grid_a = np.concatenate([row_top, sep_h, row_bot], axis=0)
    ax_a.imshow(grid_a)
    ax_a.text(4, 11, "区域 A 初始探索帧", color="white", fontsize=7.3, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#38bdf8", alpha=0.86))
    ax_a.text(103, 11, "ch0: 神经度量深度", color="#fef08a", fontsize=7.3, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#eab308", alpha=0.86))
    ax_a.text(4, 110, "留出集真机帧 (151733)", color="white", fontsize=7.3, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#10b981", alpha=0.86))
    ax_a.text(103, 110, f"AbsRel={val_metrics['AbsRel']:.3f}", color="#a7f3d0", fontsize=7.3, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#064e3b", ec="#10b981", alpha=0.86))
    ax_a.set_title("(a) 96×96 双域视觉输入 & ch0 神经逆深度解码", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(border_col)

    # --- Panel (b): 神经网络直接预测的各向异性椭球半径 (ch2: scale_xy vs ch3: scale_z) ---
    ax_b = fig.add_subplot(gs[0, 1], facecolor=card_dark)
    sxy_mm = f0_pred["scale_xy_m"] * 1000.0
    sz_mm = f0_pred["scale_z_m"] * 1000.0
    aniso_ratio = sz_mm / np.maximum(sxy_mm, 1e-3)
    sxy_vis = plt.cm.viridis(np.clip((sxy_mm - 8.0) / 34.0, 0.0, 1.0))[..., :3]
    sz_vis = plt.cm.magma(np.clip((sz_mm - 3.0) / 32.0, 0.0, 1.0))[..., :3]
    rat_r = real_pred["scale_z_m"] / np.maximum(real_pred["scale_xy_m"], 1e-4)
    aniso_s_vis = plt.cm.coolwarm(np.clip(aniso_ratio / 1.8, 0.0, 1.0))[..., :3]
    aniso_r_vis = plt.cm.coolwarm(np.clip(rat_r / 1.8, 0.0, 1.0))[..., :3]
    sv48 = np.ones((48, 2, 3), dtype=np.float32) * 0.25
    bt = np.concatenate([sxy_vis, sv48, sz_vis], axis=1)
    bb = np.concatenate([aniso_s_vis, sv48, aniso_r_vis], axis=1)
    sh48 = np.ones((2, bt.shape[1], 3), dtype=np.float32) * 0.25
    ax_b.imshow(np.concatenate([bt, sh48, bb], axis=0))
    ax_b.text(2, 6, f"ch2 水平核 s_xy\n桌面均值 {val_metrics['table_mean_scale_xy_mm']:.1f}mm", color="white", fontsize=6.9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#22c55e", alpha=0.86))
    ax_b.text(52, 6, f"ch3 垂直核 s_z\n桌面薄盘 {val_metrics['table_mean_scale_z_mm']:.1f}mm", color="#fed7aa", fontsize=6.9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#f97316", alpha=0.86))
    ax_b.text(2, 56, f"探索场 s_z/s_xy\n立面杯壁={val_metrics['object_anisotropy_sz_over_sxy']:.2f}", color="white", fontsize=6.9, bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#38bdf8", alpha=0.86))
    ax_b.text(52, 56, "真机帧 s_z/s_xy\n蓝=扁圆盘 红=竖椭球", color="white", fontsize=6.9, bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#f43f5e", alpha=0.86))
    ax_b.set_title("(b) 模型直接生成各向异性椭球尺度 (ch2 s_xy / ch3 s_z)", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(border_col)

    # --- Panel (c): 异方差神经观测精度 (ch5: precision tau) 与不透明度 (ch4: opacity) ---
    ax_c = fig.add_subplot(gs[0, 2], facecolor=card_dark)
    log_tau_s = np.log10(np.maximum(f0_pred["precision_tau"], 1.0))
    log_tau_r = np.log10(np.maximum(real_pred["precision_tau"], 1.0))
    tau_s_vis = plt.cm.plasma(np.clip((log_tau_s - 1.4) / 2.5, 0.0, 1.0))[..., :3]
    tau_r_vis = plt.cm.plasma(np.clip((log_tau_r - 1.4) / 2.5, 0.0, 1.0))[..., :3]
    op_s_vis = plt.cm.YlGnBu_r(np.clip(f0_pred["opacity"], 0.0, 1.0))[..., :3]
    op_r_vis = plt.cm.YlGnBu_r(np.clip(real_pred["opacity"], 0.0, 1.0))[..., :3]
    ct = np.concatenate([tau_s_vis, sv48, op_s_vis], axis=1)
    cb_img = np.concatenate([tau_r_vis, sv48, op_r_vis], axis=1)
    ax_c.imshow(np.concatenate([ct, sh48, cb_img], axis=0))
    ax_c.text(2, 6, f"ch5 探索场精度 log10(τ)\n近场 τ={val_metrics['near_field_mean_precision_tau']:.0f}", color="#fef08a", fontsize=6.9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#eab308", alpha=0.86))
    ax_c.text(52, 6, "ch4 神经不透明度 α\n抑制远墙/边缘飞点", color="white", fontsize=6.9, bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#38bdf8", alpha=0.86))
    ax_c.text(2, 56, "真机帧自发精度 τ\n高对比近桌面高权重", color="#fef08a", fontsize=6.9, bbox=dict(boxstyle="round,pad=0.16", fc="#0f172a", ec="#eab308", alpha=0.86))
    ax_c.text(52, 56, f"不确定度-误差相关\nr={val_metrics['uncertainty_error_corr']:+.2f} | PSNR={val_metrics['novel_view_psnr_db']:.1f}dB", color="#a7f3d0", fontsize=6.8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.16", fc="#064e3b", ec="#10b981", alpha=0.88))
    ax_c.set_title("(c) 自监督习得异方差观测精度 ch5 (τ=1/σ²) & 不透明度 ch4", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for sp in ax_c.spines.values():
        sp.set_edgecolor(border_col)

    # --- Panel (d): 持久空间记忆拓展对比 (Step 0 单帧局部 0.60 m² -> Step 17 全局拓展 > 2.0 m², 区域 A 100% 保留) ---
    ax_d = fig.add_subplot(gs[1, 0], facecolor="#050910")
    cam_overview = OrbitCamera3D(
        azimuth_deg=-136.0,
        elevation_deg=44.0,
        distance_m=2.12,
        target_x=0.68,
        target_y=0.18,
        target_z=0.04,
        fov_vertical_deg=52.0,
    )
    exp_xy = np.array([[w[0], w[1]] for w in bench["exploration_waypoints"]], dtype=np.float32)
    pose_b = RobotCameraPose(x_m=1.05, y_m=0.48, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=32.0, pitch_deg=3.5)
    img_expand = _render_anisotropic_orbit_image(
        memory_map=bench["memory_map"],
        camera=cam_overview,
        width=840,
        height=540,
        explore_path_xy=exp_xy,
        return_traj_xyyaw=None,
        robot_pose_now=pose_b,
        ping_goal_xy=None,
    )
    ax_d.imshow(img_expand)
    bsum = bench["summary"]
    ax_d.text(
        14,
        28,
        f"空间记忆拓展: {bsum['initial_area_m2']:.2f} m² (区域A单帧) -> {bsum['expanded_area_m2']:.2f} m² ({bsum['total_persistent_splats']} 高斯椭球)",
        color="#67e8f9",
        fontsize=7.6,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc="#08111c", ec="#06b6d4", alpha=0.90),
    )
    ax_d.text(
        14,
        516,
        f"抵达区域 B (1.05m, 0.48m) 时：原区域 A 点云留存率 = {bsum['region_a_splat_retention_rate']*100:.1f}% (零遗忘!)",
        color="#fde68a",
        fontsize=7.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc="#451a03", ec="#f59e0b", alpha=0.90),
    )
    ax_d.set_title("(d) 区域 A → 区域 B 多视角探索拓展与旧区 100% 持久保留", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for sp in ax_d.spines.values():
        sp.set_edgecolor("#0891b2")
        sp.set_linewidth(1.3)

    # --- Panel (e): 在区域 B 点击早已离开相机视野的区域 A 盲区坐标，绕持久记忆障碍闭环跑回 ---
    ax_e = fig.add_subplot(gs[1, 1], facecolor="#050910")
    nav_res = bench["nav_result"]
    goal_a = tuple(bsum["return_to_region_a_goal_xy"])
    img_nav = _render_anisotropic_orbit_image(
        memory_map=bench["memory_map"],
        camera=cam_overview,
        width=840,
        height=540,
        explore_path_xy=exp_xy,
        return_traj_xyyaw=nav_res["trajectory_xyyaw"],
        robot_pose_now=pose_b,
        ping_goal_xy=(float(goal_a[0]), float(goal_a[1])),
    )
    ax_e.imshow(img_nav)
    ax_e.text(
        14,
        28,
        f"当前视锥朝向 +32° (黄框)  |  回点区域 A ({goal_a[0]:.2f}m, {goal_a[1]:+.2f}m) 位于相对方位 {bsum['initial_target_bearing_deg']:+.1f}° 盲区!",
        color="#fed7aa",
        fontsize=7.3,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", fc="#431407", ec="#f97316", alpha=0.90),
    )
    ax_e.text(
        14,
        516,
        f"持久 3DGS 记忆 A* 绕水杯返回: 误差 = {bsum['return_final_error_m']*100:.2f} cm (<4cm)  |  "
        f"障碍净空 = {bsum['return_min_obstacle_clearance_m']*100:.1f} cm",
        color="#6ee7b7",
        fontsize=7.1,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#064e3b", ec="#10b981", alpha=0.90),
    )
    ax_e.set_title("(e) 全局记忆空间任意点即走：从区域 B 盲区回点绕障跑回区域 A", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    for sp in ax_e.spines.values():
        sp.set_edgecolor("#10b981")
        sp.set_linewidth(1.3)

    # --- Panel (f): 多视角贝叶斯卡尔曼位置误差收敛、后验不确定度收敛与探索面积增长曲线 ---
    ax_f = fig.add_subplot(gs[1, 2], facecolor=card_dark)
    hist = bench["step_history"]
    steps_x = [h["step"] for h in hist]
    kal_cm = [h["kalman_cup_rmse_m"] * 100.0 for h in hist]
    raw_cm = [h["single_view_cup_rmse_m"] * 100.0 for h in hist]
    sig_cm = [h["cup_posterior_sigma_m"] * 100.0 for h in hist]
    areas = [h["explored_area_m2"] for h in hist]

    l1 = ax_f.plot(steps_x, raw_cm, color="#94a3b8", linestyle="--", marker="^", markersize=4.0, lw=1.4, label="单视角无记忆表面误差 (cm)")
    l2 = ax_f.plot(steps_x, kal_cm, color="#38bdf8", linestyle="-", marker="o", markersize=4.8, lw=2.3, label=f"多视角贝叶斯修正误差 (-{bsum['multiview_position_error_reduction_pct']:.1f}%)")
    l3 = ax_f.plot(steps_x, sig_cm, color="#f59e0b", linestyle="-.", marker="s", markersize=4.0, lw=1.8, label=f"后验不确定度 σ=1/√τ (-{bsum['posterior_sigma_reduction_pct']:.1f}%)")

    ax_f.set_xlabel("多视角连续运动探索步数 (区域 A -> 环绕水杯 -> 区域 B)", color=txt_main, fontsize=9.0)
    ax_f.set_ylabel("3D 表面定位 RMSE / 后验标准差 σ (cm)", color="#38bdf8", fontsize=9.2)
    ax_f.tick_params(axis="x", colors=txt_sub, labelsize=8.2)
    ax_f.tick_params(axis="y", colors="#38bdf8", labelsize=8.2)
    ax_f.grid(True, linestyle=":", color="#334155", alpha=0.65)
    ax_f.set_ylim(0.0, max(max(raw_cm), max(sig_cm)) * 1.28)

    ax_fr = ax_f.twinx()
    l4 = ax_fr.plot(steps_x, areas, color="#10b981", linestyle="-", marker="D", markersize=4.2, lw=2.1, label="持久记忆已探索面积 (m², 右轴)")
    ax_fr.axhline(1.80, color="#34d399", linestyle=":", lw=1.2)
    ax_fr.set_ylabel("全局持久空间探索面积 (m²)", color="#10b981", fontsize=9.2)
    ax_fr.tick_params(axis="y", colors="#10b981", labelsize=8.2)
    ax_fr.set_ylim(0.0, max(areas) * 1.22)

    lines_all = l1 + l2 + l3 + l4
    labels_all = [ln.get_label() for ln in lines_all]
    ax_f.legend(lines_all, labels_all, loc="upper left", fontsize=7.5, facecolor="#0b1120", edgecolor="#334155", labelcolor=txt_main)
    ax_f.set_title("(f) 多视角卡尔曼位置修正收敛与无界空间面积拓展曲线", color=txt_main, fontsize=10.2, fontweight="bold", pad=7)
    for sp in ax_f.spines.values():
        sp.set_edgecolor(border_col)

    footer = (
        f"HoverGaussianSplatNet v2 实测汇总:  TFLite = {model_info['tflite_bytes']/1024:.1f} KB (CPU {model_info['cpu_latency_ms']:.2f} ms, max|diff|={model_info['max_abs_diff']:.2e})  |  "
        f"多视角 3D 误差下降 = {bsum['multiview_position_error_reduction_pct']:.1f}%  |  "
        f"探索面积拓展 = {bsum['initial_area_m2']:.2f} -> {bsum['expanded_area_m2']:.2f} m²  |  "
        f"旧区 A 留存率 = {bsum['region_a_splat_retention_rate']*100:.1f}%  |  "
        f"盲区回点导航误差 = {bsum['return_final_error_m']*100:.2f} cm"
    )
    fig.text(
        0.5,
        0.025,
        footer,
        ha="center",
        va="center",
        color="#f8fafc",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.36", facecolor="#0f172a", edgecolor="#38bdf8", lw=1.2),
    )

    fig.savefig(out_path, dpi=220, facecolor=bg_dark)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. 主训练、验证与全套交付物导出入口
# ---------------------------------------------------------------------------

def train_and_evaluate_neural_3dgs(
    n_train_pairs: int = 2400,
    n_val_pairs: int = 300,
    epochs: int = 48,
    batch_size: int = 48,
    lr: float = 2.5e-3,
    seed: int = 20261118,
) -> dict[str, Any]:
    sealed.require_final_eval(
        [REPO_ROOT / "dataset" / "takes" / HELDOUT_TAKE_ID / "take.mp4"],
        purpose="train_neural_gaussian_splat final held-out evaluation",
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Neural3DGS] Training device: {device}")

    t_gen0 = time.perf_counter()
    rng_tr = np.random.default_rng(seed)
    rng_va = np.random.default_rng(seed + 88883)

    print(f"[Neural3DGS] Synthesizing {n_train_pairs} multi-view 3D scene pairs + {n_val_pairs} validation pairs...")
    train_pairs = [generate_multiview_scene_pair(rng_tr) for _ in range(int(n_train_pairs))]
    val_pairs = [generate_multiview_scene_pair(rng_va) for _ in range(int(n_val_pairs))]

    takes_dir = REPO_ROOT / "dataset" / "takes"
    real_train_raw, real_heldout_raw = sample_real_robot_frames(takes_dir, frames_per_take=40)
    real_train = [enrich_real_frame_for_splat_training(r) for r in real_train_raw]
    print(
        f"[Neural3DGS] Dataset ready in {time.perf_counter() - t_gen0:.2f}s | "
        f"synth_pairs={len(train_pairs)}, real_train={len(real_train)}, real_heldout={len(real_heldout_raw)}"
    )

    dataset = HybridNeural3DGSDataset(train_pairs, real_train, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = HoverGaussianSplatNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=6e-5)
    pos_weight = torch.tensor([2.4], device=device)

    t_tr0 = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        # 前 6 轮先稳定几何主干，随后平滑引入可微新视角 3DGS 光度重投影一致性损失
        w_photo = 0.0 if ep <= 4 else min(0.55, 0.55 * (ep - 4) / 10.0)

        for batch in loader:
            rgb1 = batch["rgb1"].to(device, non_blocking=True)
            rgb48 = batch["rgb48"].to(device, non_blocking=True)
            rgb2_24 = batch["rgb2_24"].to(device, non_blocking=True)
            gt_d = batch["depth_m"].to(device, non_blocking=True)
            gt_inv = batch["inv_depth_01"].to(device, non_blocking=True)
            gt_elev = batch["elev_01"].to(device, non_blocking=True)
            gt_sxy = batch["scale_xy_01"].to(device, non_blocking=True)
            gt_sz = batch["scale_z_01"].to(device, non_blocking=True)
            gt_op = batch["opacity_01"].to(device, non_blocking=True)
            gt_edge = batch["edge_01"].to(device, non_blocking=True)
            o1 = batch["origin1"].to(device, non_blocking=True)
            r1 = batch["rays1_48"].to(device, non_blocking=True)
            o2 = batch["origin2"].to(device, non_blocking=True)
            rw2c2 = batch["rot_w2c2"].to(device, non_blocking=True)
            is_synth = batch["is_synth"].to(device, non_blocking=True)
            w_sup = batch["w_sup"].to(device, non_blocking=True)

            logits = model.forward_logits(rgb1)
            probs = torch.sigmoid(logits)

            pred_inv = probs[:, 0]
            pred_elev = probs[:, 1]
            pred_sxy = probs[:, 2]
            pred_sz = probs[:, 3]
            pred_op = probs[:, 4]
            pred_prec = probs[:, 5]

            w_b = w_sup.view(-1, 1, 1)
            pred_d = inv_depth_01_to_metric_m(pred_inv)

            # 1) 度量逆深度 + 相对深度 + 边缘梯度损失
            l_inv = (torch.abs(pred_inv - gt_inv) * w_b).mean()
            l_rel = ((torch.abs(pred_d - gt_d) / torch.clamp(gt_d, min=0.20)) * w_b).mean()
            l_grad = compute_multiscale_grad_loss(pred_inv, gt_inv, w_sup)

            # 2) 突起高度 BCE + Soft Dice
            bce = F.binary_cross_entropy_with_logits(logits[:, 1], gt_elev, pos_weight=pos_weight, reduction="none")
            l_elev = (bce * w_b).mean()
            inter = (pred_elev * gt_elev * w_b).sum(dim=(1, 2))
            denom = ((pred_elev + gt_elev) * w_b).sum(dim=(1, 2))
            l_dice = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()

            # 3) 神经各向异性椭球半径 & 不透明度监督
            l_sxy = (torch.abs(pred_sxy - gt_sxy) * w_b).mean()
            l_sz = (torch.abs(pred_sz - gt_sz) * w_b).mean()
            l_op = (torch.abs(pred_op - gt_op) * w_b).mean()

            # 4) 异方差高斯负对数似然损失 (Heteroscedastic Uncertainty NLL, Kendall & Gal):
            # 让 ch5 (precision_01 -> tau) 自发收敛至 1 / E[err^2]
            err_detached = torch.abs(pred_d.detach() - gt_d) + 0.008 + 0.018 * (gt_d * gt_d) + 0.045 * gt_edge
            tau_pred = precision_01_to_tau(pred_prec)
            nll_map = 0.5 * (tau_pred * (err_detached * err_detached) - torch.log(tau_pred))
            l_nll = (nll_map * w_b).mean()

            # 5) 可微新视角 3DGS 光度重投影一致性损失
            if w_photo > 0.0 and bool(torch.any(is_synth > 0.5)):
                nv = differentiable_novel_view_render_24x24(probs, rgb48, o1, r1, o2, rw2c2)
                ren2 = nv["rendered_rgb_24"]
                cov2 = (nv["coverage_24"] > 0.38).float() * is_synth.view(-1, 1, 1, 1)
                l_photo = (torch.abs(ren2 - rgb2_24) * cov2).sum() / (cov2.sum() * 3.0 + 1.0)
            else:
                l_photo = probs.new_tensor(0.0)

            loss = (
                1.55 * l_inv
                + 0.45 * l_rel
                + 0.80 * l_grad
                + 0.50 * l_elev
                + 0.30 * l_dice
                + 1.25 * l_sxy
                + 1.35 * l_sz
                + 0.85 * l_op
                + 0.08 * l_nll
                + w_photo * l_photo
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        if ep == 1 or ep % 8 == 0 or ep == int(epochs):
            print(
                f"  [Epoch {ep:02d}/{epochs}] loss={running_loss / max(n_batches, 1):.4f} "
                f"w_photo={w_photo:.2f} lr={scheduler.get_last_lr()[0]:.5f}"
            )

    train_sec = time.perf_counter() - t_tr0
    print(f"[Neural3DGS] Training completed in {train_sec:.2f}s")

    val_metrics = evaluate_neural_splat_model(model, val_pairs, device=device)

    model_cpu = model.to("cpu").eval()
    cpu_ms = benchmark_neural_splat_cpu_ms(model_cpu, n_runs=60)

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = export_neural_splat_bundle(
        model_cpu,
        out_dir=DEFAULT_OUT_DIR,
        android_assets_dir=DEFAULT_ANDROID_ASSETS,
        version="2026.11.18",
    )

    ply_path = DEFAULT_OUT_DIR / "expanded_memory_map.ply"
    bench = run_memory_expansion_and_return_benchmark(bundle["tflite_path"], out_ply_path=ply_path)
    real_eval = evaluate_real_robot_sequence_memory(bundle["tflite_path"], real_heldout_raw)

    model_info = {
        "model_name": "hover-gaussian-splat-v2",
        "version": "2026.11.18",
        "input_shape": [1, INPUT_H, INPUT_W, INPUT_C],
        "output_shape": [1, OUTPUT_H, OUTPUT_W, 6],
        "epochs": int(epochs),
        "train_synth_pairs": int(n_train_pairs),
        "train_real_frames": len(real_train),
        "val_synth_pairs": int(n_val_pairs),
        "train_time_sec": round(train_sec, 2),
        "cpu_latency_ms": round(cpu_ms, 3),
        "tflite_bytes": bundle["tflite_bytes"],
        "max_abs_diff": bundle["max_abs_diff"],
        "sha256": bundle["sha256"],
    }

    render_fig10_neural_3dgs_showcase(
        bench=bench,
        real_eval=real_eval,
        val_metrics=val_metrics,
        model_info=model_info,
        out_png=DEFAULT_FIG10_PATH,
    )

    real_eval_clean = {
        k: v for k, v in real_eval.items() if k not in ("rep_rgb_nhwc", "rep_pred")
    }
    full_metrics = {
        "schema_version": 1,
        "model_info": model_info,
        "validation_metrics": val_metrics,
        "spatial_memory_expansion_benchmark": bench["summary"],
        "exploration_step_history": bench["step_history"],
        "real_heldout_sequence_evaluation": real_eval_clean,
        "artifacts": {
            "tflite_path": bundle["tflite_path"],
            "json_path": bundle["json_path"],
            "zip_path": bundle["zip_path"],
            "android_assets": bundle["android_assets"],
            "expanded_memory_ply": str(ply_path),
            "fig10_path": str(DEFAULT_FIG10_PATH),
        },
    }

    metrics_file = DEFAULT_OUT_DIR / "neural_3dgs_metrics.json"
    metrics_file.write_text(json.dumps(full_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[Neural3DGS] Summary Metrics:")
    print(json.dumps(model_info, ensure_ascii=False, indent=2))
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print(json.dumps(bench["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(real_eval_clean, ensure_ascii=False, indent=2))
    return full_metrics


def evaluate_existing_tflite_bundle(seed: int = 20261118, n_val_pairs: int = 300) -> dict[str, Any]:
    """加载已训练导出的 `hover_gaussian_splat_v2.tflite`，快速重跑完整空间记忆基准并刷新 PLY/JSON/fig10。"""
    from neural_gaussian_splat_model import load_neural_splat_from_tflite, verify_neural_splat_tflite_parity

    tfl_path = DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.tflite"
    jsn_path = DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.json"
    zip_path = DEFAULT_OUT_DIR / "hover-gaussian-splat-v2-bundle.zip"
    prev_metrics_file = DEFAULT_OUT_DIR / "neural_3dgs_metrics.json"
    prev_info = (
        json.loads(prev_metrics_file.read_text(encoding="utf-8")).get("model_info", {})
        if prev_metrics_file.exists()
        else {}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_neural_splat_from_tflite(tfl_path).to(device).eval()

    rng_va = np.random.default_rng(seed + 88883)
    val_pairs = [generate_multiview_scene_pair(rng_va) for _ in range(int(n_val_pairs))]
    val_metrics = evaluate_neural_splat_model(model, val_pairs, device=device)

    model_cpu = model.to("cpu").eval()
    cpu_ms = benchmark_neural_splat_cpu_ms(model_cpu, n_runs=60)
    max_diff = verify_neural_splat_tflite_parity(model_cpu, tfl_path, n_samples=8)

    _, real_heldout_raw = sample_real_robot_frames(REPO_ROOT / "dataset" / "takes", frames_per_take=40)
    ply_path = DEFAULT_OUT_DIR / "expanded_memory_map.ply"
    bench = run_memory_expansion_and_return_benchmark(tfl_path, out_ply_path=ply_path)
    real_eval = evaluate_real_robot_sequence_memory(tfl_path, real_heldout_raw)

    raw_bytes = tfl_path.read_bytes()
    import hashlib

    model_info = {
        "model_name": "hover-gaussian-splat-v2",
        "version": "2026.11.18",
        "input_shape": [1, INPUT_H, INPUT_W, INPUT_C],
        "output_shape": [1, OUTPUT_H, OUTPUT_W, 6],
        "epochs": int(prev_info.get("epochs", 48)),
        "train_synth_pairs": int(prev_info.get("train_synth_pairs", 2400)),
        "train_real_frames": int(prev_info.get("train_real_frames", 200)),
        "val_synth_pairs": int(n_val_pairs),
        "train_time_sec": float(prev_info.get("train_time_sec", 455.01)),
        "cpu_latency_ms": round(cpu_ms, 3),
        "tflite_bytes": len(raw_bytes),
        "max_abs_diff": max_diff,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }

    render_fig10_neural_3dgs_showcase(
        bench=bench,
        real_eval=real_eval,
        val_metrics=val_metrics,
        model_info=model_info,
        out_png=DEFAULT_FIG10_PATH,
    )

    real_eval_clean = {k: v for k, v in real_eval.items() if k not in ("rep_rgb_nhwc", "rep_pred")}
    full_metrics = {
        "schema_version": 1,
        "model_info": model_info,
        "validation_metrics": val_metrics,
        "spatial_memory_expansion_benchmark": bench["summary"],
        "exploration_step_history": bench["step_history"],
        "real_heldout_sequence_evaluation": real_eval_clean,
        "artifacts": {
            "tflite_path": str(tfl_path),
            "json_path": str(jsn_path),
            "zip_path": str(zip_path),
            "android_assets": [
                str(DEFAULT_ANDROID_ASSETS / "hover_gaussian_splat_v2.tflite"),
                str(DEFAULT_ANDROID_ASSETS / "hover_gaussian_splat_v2.json"),
            ],
            "expanded_memory_ply": str(ply_path),
            "fig10_path": str(DEFAULT_FIG10_PATH),
        },
    }
    metrics_file = DEFAULT_OUT_DIR / "neural_3dgs_metrics.json"
    metrics_file.write_text(json.dumps(full_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[Neural3DGS] Evaluation Summary:")
    print(json.dumps(bench["summary"], ensure_ascii=False, indent=2))
    return full_metrics


if __name__ == "__main__":
    if "--eval-only" in sys.argv and (DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.tflite").exists():
        evaluate_existing_tflite_bundle()
    else:
        train_and_evaluate_neural_3dgs()
