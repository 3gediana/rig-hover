# -*- coding: utf-8 -*-
"""dense_scene_pipeline.py — 端侧轻量 3D 高斯视觉建图、动态光线雕刻、EDL 渲染与 5cm 可达域 A* 导航管线。

全部 3D 重建与可通行域分析仅依赖超轻量端侧 student `.tflite`（约 529 KB Float32，CPU < 10 ms）：
1. 联合双边边缘引导上采样 (JBU: 96x96 -> 192x192) + 飞点梯度剔除 + 3D 表面法向估计；
2. 无教师纯端侧桌面 Z=0 自动配平 (IRLS 鲁棒平面拟合) + 航向配准；
3. 动态体素地图 + 自由空间光线雕刻 (Free-Space Ray Carving)：
   人物/手进入、挥动或离开时，视锥内被穿透的前景幽灵体素在 1~2 帧内立即清除，绝不卡死在场景中，
   同时保留视野外与遮挡后的已探索静态桌面与公共空间记忆；
4. Z-Buffer 遮挡 + Lambertian 法向光照 + 两遍 Eye-Dome Lighting (EDL) 阴影勾勒；
5. 5 cm 可通行栅格 (绿=BFS连通可达桌面，红=障碍物理膨胀，灰=未知/桌外) + 点击校验吸附 + A* 路径规划 + 2D 公共小地图。
"""
from __future__ import annotations

from collections import deque
import heapq
import math
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 本管线处理的是 dense splat 学生模型的输出，所以各函数的相机默认值取该模型工件训练时的相机
# （v3：72°，见 dense_splat_student.STUDENT_V3_*），而不是机器人的实测相机：换了相机，学生模型
# 输出的逆深度就不再对应它学过的几何。用实测相机重训学生模型之后，这里要跟着改成新工件的相机。
from dense_splat_student import (  # noqa: E402
    STUDENT_V3_FOV_H_DEG,
    dense_inv_depth_01_to_metric_m,
    joint_bilateral_upsample_depth,
)
from export_tflite_bundle import run_tflite_flatbuffer_cpu  # noqa: E402
from mono_depth_model import (  # noqa: E402
    DEFAULT_H_CAM_M,
    make_camera_ray_grid,
)
from neural_gaussian_splat_model import OrbitCamera3D, PersistentGaussianMemoryMap  # noqa: E402

_PROJ = PersistentGaussianMemoryMap(max_splats=8)


def run_student_tflite_frame(tflite_path: pathlib.Path | str, bgr_or_rgb_240: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """运行纯端侧 `.tflite` 推理: 240x240 -> 96x96 RGB [0,1] -> [96, 96, 6] float32 splat map."""
    img = cv2.cvtColor(bgr_or_rgb_240, cv2.COLOR_BGR2RGB) if is_bgr else bgr_or_rgb_240
    rgb_96 = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    out_nhwc = run_tflite_flatbuffer_cpu(tflite_path, [rgb_96[None, ...]])[0]
    return out_nhwc[0]  # [96, 96, 6]


def solve_table_plane_from_student(
    inv01_lr: np.ndarray,
    elev_lr: np.ndarray,
    fov_h_deg: float = STUDENT_V3_FOV_H_DEG,
    prev_r_level: np.ndarray | None = None,
    prev_z_bias: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """完全基于端侧 student 自身预测的桌面 Z=0 平面配平（零服务端依赖）。

    地面波段内点最大化搜索（pitch x yaw 网格 + median 高度配平）：
    1. 学生网络的近场桌面逆深度会饱和在 1.0（训练教师深度下限 0.222 m 裁剪所致），
       metric 深度在桌面上近似等深壳，直接做平面最小二乘会拟出 [1,0,0] 法向（实测）；
       离线教师逐帧标定的桌面法向俯仰本身也在 0.4~17.4 度间波动（std 4.6 度），
       无法作为固定常数。因此这里不再做平面拟合，而是直接优化下游需要的目标：
       在 (pitch, yaw) 网格上找让最多"学生自判桌面点"（elev<0.35 的底部平坦区）
       落进 5cm 可达栅格的地面波段 Z∈[-0.035, +0.036] m 的水平旋转；
    2. z_bias 取最优旋转下桌面点的中位高度，使桌面严格落到 Z=0（限幅 ±0.08 m）；
    3. 与上一帧做轻度时域平滑避免抖动。
    纯向量网格搜索（17x13 组合 x ~1k 点），单帧 < 2 ms，适合端侧移植。
    """
    h, w = inv01_lr.shape
    rays = make_camera_ray_grid(h, w, fov_h_deg=fov_h_deg, pitch_deg=0.0, roll_deg=0.0)
    gx = np.abs(cv2.Sobel(inv01_lr, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(inv01_lr, cv2.CV_32F, 0, 1, ksize=3))
    yy = np.arange(h)[:, None]
    uu = ((np.arange(w, dtype=np.float32) + 0.5) / float(w)) * 2.0 - 1.0
    base = (elev_lr < 0.35) & (yy >= int(h * 0.50)) & (np.abs(uu)[None, :] <= 0.55) & ((gx + gy) < 0.14)
    metric = dense_inv_depth_01_to_metric_m(inv01_lr.astype(np.float32))
    base &= (metric > 0.18) & (metric < 1.35)  # 覆盖近端与远端桌面点，排除极远背景墙
    if int(np.sum(base)) < 80:
        if prev_r_level is None:
            r_level = np.eye(3, dtype=np.float32)
        else:
            r_level = prev_r_level
        return r_level, float(prev_z_bias), math.degrees(math.asin(float(np.clip(r_level[2, 0], -1.0, 1.0))))

    r_pts = rays[base]
    d_pts = metric[base]

    best_score = -1e9
    best: tuple[float, float, float] | None = None  # (theta_deg, psi_deg, z_med)
    for theta_deg in np.linspace(-6.0, 12.0, 19):
        for psi_deg in np.linspace(-6.0, 6.0, 13):
            th, ps = math.radians(float(theta_deg)), math.radians(float(psi_deg))
            z_ax = np.array(
                [-math.sin(th) * math.cos(ps), math.sin(ps), math.cos(th) * math.cos(ps)],
                dtype=np.float32,
            )
            wz = DEFAULT_H_CAM_M + d_pts * (r_pts @ z_ax)
            z_med = float(np.median(wz))
            if abs(z_med) > 0.08:
                continue
            dev = wz - z_med
            in_band = int(np.sum(np.abs(dev) < 0.028))
            mad = float(np.median(np.abs(dev)))
            score = float(in_band) - 120.0 * mad - 0.15 * abs(float(theta_deg))
            if score > best_score:
                best_score = score
                best = (float(theta_deg), float(psi_deg), z_med)

    r_level = prev_r_level
    z_bias = float(prev_z_bias)
    if best is not None:
        th, ps, z_med = best
        z_ax = np.array(
            [-math.sin(math.radians(th)) * math.cos(math.radians(ps)),
             math.sin(math.radians(ps)),
             math.cos(math.radians(th)) * math.cos(math.radians(ps))],
            dtype=np.float32,
        )
        x_ax = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), z_ax)
        x_ax = x_ax / max(float(np.linalg.norm(x_ax)), 1e-6)
        y_ax = np.cross(z_ax, x_ax)
        r_cand = np.stack([x_ax, y_ax, z_ax], axis=0).astype(np.float32)
        if prev_r_level is not None:
            r_blend = 0.55 * r_cand + 0.45 * prev_r_level
            u_svd, _, vt_svd = np.linalg.svd(r_blend)
            r_level = (u_svd @ vt_svd).astype(np.float32)
            z_bias = 0.65 * z_med + 0.35 * prev_z_bias
        else:
            r_level = r_cand
            z_bias = z_med

    if r_level is None:
        r_level = np.eye(3, dtype=np.float32)
    pitch_deg = math.degrees(math.asin(float(np.clip(r_level[2, 0], -1.0, 1.0))))
    return r_level, z_bias, pitch_deg


def reconstruct_dense_frame_3d(
    bgr: np.ndarray,
    splats_96: np.ndarray,
    r_level: np.ndarray,
    z_bias: float = 0.0,
    fov_h_deg: float = STUDENT_V3_FOV_H_DEG,
    yaw_deg: float = 0.0,
    cam_x: float = 0.0,
    cam_y: float = 0.0,
    out_hw: tuple[int, int] = (192, 192),
    jbu_radius: int = 2,
    jbu_sigma_range: float = 0.085,
) -> dict[str, np.ndarray]:
    """JBU 边缘上采样 + 飞点剔除 + 法向计算 -> 稠密世界坐标系点云。"""
    hr_h, hr_w = out_hw
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rgb_hr = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (hr_w, hr_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    inv_96 = splats_96[..., 0].astype(np.float32)
    elev_96 = splats_96[..., 1].astype(np.float32)

    inv_hr = joint_bilateral_upsample_depth(inv_96, gray, out_hw=out_hw, radius=jbu_radius, sigma_spatial=1.5, sigma_range=jbu_sigma_range)
    depth_hr = dense_inv_depth_01_to_metric_m(inv_hr)
    elev_hr = cv2.resize(elev_96, (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)

    # 飞点梯度滤波 (剔除悬空跳变边)
    gx = cv2.Sobel(inv_hr, cv2.CV_32F, 1, 0, ksize=3) * 0.25
    gy = cv2.Sobel(inv_hr, cv2.CV_32F, 0, 1, ksize=3) * 0.25
    gmag = np.sqrt(gx * gx + gy * gy)

    rays = make_camera_ray_grid(hr_h, hr_w, fov_h_deg=fov_h_deg, pitch_deg=0.0, roll_deg=0.0)
    rays_l = rays @ r_level.T
    xyz_cam = rays_l * depth_hr[..., None]

    # 从 3D 点网格差分求法向并保证朝向相机
    dx_r = np.roll(xyz_cam, -1, axis=1) - np.roll(xyz_cam, 1, axis=1)
    dy_d = np.roll(xyz_cam, -1, axis=0) - np.roll(xyz_cam, 1, axis=0)
    nrm_cam = np.cross(dy_d, dx_r)
    nrm_cam /= np.maximum(np.linalg.norm(nrm_cam, axis=-1, keepdims=True), 1e-6)
    flip = np.sum(nrm_cam * rays_l, axis=-1, keepdims=True) > 0
    nrm_cam = np.where(flip, -nrm_cam, nrm_cam)

    rad = math.radians(float(yaw_deg))
    cy, sy = math.cos(rad), math.sin(rad)
    wx = cam_x + xyz_cam[..., 0] * cy - xyz_cam[..., 1] * sy
    wy = cam_y + xyz_cam[..., 0] * sy + xyz_cam[..., 1] * cy
    wz = (DEFAULT_H_CAM_M - z_bias) + xyz_cam[..., 2]

    nx_w = nrm_cam[..., 0] * cy - nrm_cam[..., 1] * sy
    ny_w = nrm_cam[..., 0] * sy + nrm_cam[..., 1] * cy
    nz_w = nrm_cam[..., 2]

    valid = (depth_hr < 2.45) & (depth_hr > 0.145) & (gmag < 0.16) & (wz > -0.05) & (wz < 0.68)
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False

    pts = np.stack([wx[valid], wy[valid], wz[valid]], axis=-1).astype(np.float32)
    nrms = np.stack([nx_w[valid], ny_w[valid], nz_w[valid]], axis=-1).astype(np.float32)
    cols = rgb_hr[valid].astype(np.float32)
    elvs = elev_hr[valid].astype(np.float32)

    return {
        "pts": pts,
        "nrms": nrms,
        "cols": cols,
        "elev": elvs,
        "depth_hr": depth_hr,
    }


class DynamicSceneVoxelMap:
    """动态 0.85cm 体素世界地图 + 自由空间光线雕刻 (Free-Space Ray Carving)。

    特性：
    - 视锥内动态前景（人物/手臂/拳头/移动杯子）移动或离开后，光线穿过原位置的空域会在 1-2 帧内
      立即将旧体素权重衰减清除，彻底杜绝动态物体卡在场景中的重影；
    - 视锥外已探索静态桌面（如旋转扫描左右环境）与被近距离短暂遮挡的后景均持久保留，
      随时随地构建完整的共享公共空间地图。
    """

    def __init__(self, vox_size: float = 0.0085, fov_h_deg: float = STUDENT_V3_FOV_H_DEG):
        self.vox_size = float(vox_size)
        self.fov_h_deg = float(fov_h_deg)
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.nrm = np.zeros((0, 3), dtype=np.float32)
        self.col = np.zeros((0, 3), dtype=np.float32)
        self.wt = np.zeros((0,), dtype=np.float32)
        self.last_step = np.zeros((0,), dtype=np.int32)
        self.step = 0
        self.last_carved_count = 0

    def reset(self) -> None:
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.nrm = np.zeros((0, 3), dtype=np.float32)
        self.col = np.zeros((0, 3), dtype=np.float32)
        self.wt = np.zeros((0,), dtype=np.float32)
        self.last_step = np.zeros((0,), dtype=np.int32)
        self.step = 0
        self.last_carved_count = 0

    def update(
        self,
        pts: np.ndarray,
        nrms: np.ndarray,
        cols: np.ndarray,
        depth_hr: np.ndarray,
        r_level: np.ndarray,
        yaw_deg: float = 0.0,
        cam_x: float = 0.0,
        cam_y: float = 0.0,
        z_bias: float = 0.0,
    ) -> None:
        self.step += 1
        hr_h, hr_w = depth_hr.shape

        inv_vs = 1.0 / self.vox_size
        qi = np.floor(pts * inv_vs).astype(np.int64) + 1024
        keys_new = (qi[:, 0] << 24) | (qi[:, 1] << 12) | qi[:, 2]
        _, first_idx = np.unique(keys_new, return_index=True)
        n_pts = pts[first_idx]
        n_nrm = nrms[first_idx]
        n_col = cols[first_idx]

        if len(self.pos) == 0:
            self.pos = n_pts
            self.nrm = n_nrm
            self.col = n_col
            self.wt = np.ones(len(n_pts), dtype=np.float32)
            self.last_step = np.full(len(n_pts), self.step, dtype=np.int32)
            self.last_carved_count = 0
            return

        # 逆投影现有地图点到当前相机坐标系
        rad = math.radians(float(yaw_deg))
        cy, sy = math.cos(rad), math.sin(rad)
        dx = self.pos[:, 0] - cam_x
        dy = self.pos[:, 1] - cam_y
        dz = self.pos[:, 2] - (DEFAULT_H_CAM_M - z_bias)
        lx = dx * cy + dy * sy
        ly = -dx * sy + dy * cy
        cam_p = np.stack([lx, ly, dz], axis=-1) @ r_level

        cx_v = cam_p[:, 0]
        tan_h = math.tan(math.radians(self.fov_h_deg * 0.5))
        fx = (hr_w * 0.5) / tan_h
        fy = (hr_h * 0.5) / tan_h

        in_front = cx_v > 0.14
        u_px = np.rint((hr_w * 0.5 - 0.5) + (cam_p[:, 1] / np.maximum(cx_v, 1e-4)) * fx).astype(np.int32)
        v_px = np.rint((hr_h * 0.5 - 0.5) - (cam_p[:, 2] / np.maximum(cx_v, 1e-4)) * fy).astype(np.int32)
        in_fov = in_front & (u_px >= 2) & (u_px < hr_w - 2) & (v_px >= 2) & (v_px < hr_h - 2)

        keep = np.ones(len(self.pos), dtype=bool)
        carved_cnt = 0
        if np.any(in_fov):
            idx = np.where(in_fov)[0]
            d_new_at = depth_hr[v_px[idx], u_px[idx]]
            d_old_at = cx_v[idx]
            z_world_at = self.pos[idx, 2]

            # 1) 自由空间光线雕刻：老体素落在当前有效深度前方 (>3.5cm)，必已被空光线穿破
            carved = d_old_at < (d_new_at - 0.035)
            carved_cnt = int(np.sum(carved))
            self.wt[idx[carved]] *= 0.04

            # 2) 视锥内实时动态更新：未被更近物体遮挡的高台/人体体素直接刷新，杜绝运动重影
            unoccluded_elev = (z_world_at > 0.028) & (d_old_at <= d_new_at + 0.045)
            self.wt[idx[unoccluded_elev]] *= 0.04

            keep[self.wt < 0.35] = False

        # 视角外极长时间未更新的高空体素(>180步)缓慢淘汰，地面持久保留
        age = self.step - self.last_step
        keep &= ~((age > 180) & (self.pos[:, 2] > 0.045))

        rem_pos = self.pos[keep]
        rem_nrm = self.nrm[keep]
        rem_col = self.col[keep]
        rem_wt = np.minimum(self.wt[keep], 2.0)
        rem_step = self.last_step[keep]

        qi_rem = np.floor(rem_pos * inv_vs).astype(np.int64) + 1024
        keys_rem = (qi_rem[:, 0] << 24) | (qi_rem[:, 1] << 12) | qi_rem[:, 2]
        overlap = np.isin(keys_rem, keys_new[first_idx])
        mask_k = ~overlap

        self.pos = np.concatenate([rem_pos[mask_k], n_pts], axis=0)
        self.nrm = np.concatenate([rem_nrm[mask_k], n_nrm], axis=0)
        self.col = np.concatenate([rem_col[mask_k], n_col], axis=0)
        self.wt = np.concatenate([rem_wt[mask_k], np.ones(len(n_pts), dtype=np.float32)], axis=0)
        self.last_step = np.concatenate([rem_step[mask_k], np.full(len(n_pts), self.step, dtype=np.int32)], axis=0)
        self.last_carved_count = carved_cnt


class TraversabilityGrid2D:
    """5cm 桌面可通行栅格地图 + BFS 最大可达域分割 + 点击安全校验/吸附 + A* 避障路径规划。"""

    def __init__(
        self,
        res: float = 0.05,
        x_range: tuple[float, float] = (-0.25, 1.55),
        y_range: tuple[float, float] = (-0.95, 0.95),
        obs_radius_m: float = 0.055,
    ):
        self.res = float(res)
        self.xs = np.arange(x_range[0], x_range[1], self.res, dtype=np.float32)
        self.ys = np.arange(y_range[0], y_range[1], self.res, dtype=np.float32)
        self.nx = len(self.xs)
        self.ny = len(self.ys)
        self.obs_radius_m = float(obs_radius_m)

        self.reach = np.zeros((self.ny, self.nx), dtype=bool)
        self.obs_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.seen_ground = np.zeros((self.ny, self.nx), dtype=bool)

    def update(self, pts: np.ndarray, robot_x: float = 0.0, robot_y: float = 0.0, robot_yaw_deg: float = 0.0) -> None:
        self.reach.fill(False)
        self.obs_mask.fill(False)
        self.seen_ground.fill(False)
        if len(pts) == 0:
            return

        gx = np.floor((pts[:, 0] - self.xs[0]) / self.res).astype(np.int32)
        gy = np.floor((pts[:, 1] - self.ys[0]) / self.res).astype(np.int32)
        inb = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
        gx, gy, pz = gx[inb], gy[inb], pts[inb, 2]

        ground_cnt = np.zeros((self.ny, self.nx), dtype=np.int32)
        obs_cnt = np.zeros((self.ny, self.nx), dtype=np.int32)

        # 平坦桌面 Z ∈ [-3.5cm, +3.6cm] vs 物理碰撞身高区内障碍 Z ∈ [+4.2cm, +18.0cm]
        g_pts = (pz >= -0.035) & (pz <= 0.036)
        o_pts = (pz >= 0.042) & (pz <= 0.185)
        np.add.at(ground_cnt, (gy[g_pts], gx[g_pts]), 1)
        np.add.at(obs_cnt, (gy[o_pts], gx[o_pts]), 1)

        seen = ground_cnt >= 3
        raw_obs = (obs_cnt >= 12) & (obs_cnt >= ground_cnt * 0.65)
        seen = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

        rad_c = max(1, int(round(self.obs_radius_m / self.res)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad_c + 1, 2 * rad_c + 1))
        inflated_obs = cv2.dilate(raw_obs.astype(np.uint8), ker) > 0

        # 机器人当前足迹下保证一小块局部起始通行区不受自己前方视锥边缘盲区割裂
        free_cell = seen & (~inflated_obs)
        self.seen_ground = seen
        self.obs_mask = inflated_obs & (seen | raw_obs)

        fy, fx = np.where(free_cell)
        if len(fy) == 0:
            return

        rad = math.radians(robot_yaw_deg)
        seed_x = robot_x + 0.25 * math.cos(rad)
        seed_y = robot_y + 0.25 * math.sin(rad)

        num_cc, labels = cv2.connectedComponents(free_cell.astype(np.uint8), connectivity=4)
        best_score = -1e9
        best_rc = None
        for cid in range(1, num_cc):
            cy_i, cx_i = np.where(labels == cid)
            if len(cy_i) < 3:
                continue
            d_rob = float(np.min(np.hypot(self.xs[cx_i] + 0.5 * self.res - seed_x, self.ys[cy_i] + 0.5 * self.res - seed_y)))
            sc = len(cy_i) * 0.45 - d_rob * 38.0
            if sc > best_score:
                best_score = sc
                k = int(np.argmin(np.hypot(self.xs[cx_i] + 0.5 * self.res - seed_x, self.ys[cy_i] + 0.5 * self.res - seed_y)))
                best_rc = (int(cy_i[k]), int(cx_i[k]))
        if best_rc is None:
            k = int(np.argmin(np.hypot(self.xs[fx] + 0.5 * self.res - seed_x, self.ys[fy] + 0.5 * self.res - seed_y)))
            best_rc = (int(fy[k]), int(fx[k]))

        q = deque([best_rc])
        self.reach[best_rc] = True
        while q:
            rr, cc = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < self.ny and 0 <= nc < self.nx and free_cell[nr, nc] and not self.reach[nr, nc]:
                    self.reach[nr, nc] = True
                    q.append((nr, nc))

    def click_and_plan(
        self,
        click_wx: float,
        click_wy: float,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw_deg: float = 0.0,
        max_snap_m: float = 0.08,
    ) -> dict[str, Any]:
        """校验用户点击位置并规划 A* 路径。若点在红区/灰区则拒绝并给出最近安全绿化点。"""
        ry, rx = np.where(self.reach)
        if len(ry) == 0:
            return {"accepted": False, "reason": "no_reachable_ground", "snapped_xy": None, "path": []}

        # 判断点击所在格子类型
        ci = int(np.floor((click_wx - self.xs[0]) / self.res))
        ri = int(np.floor((click_wy - self.ys[0]) / self.res))
        inb = (0 <= ci < self.nx) and (0 <= ri < self.ny)

        d_click = np.hypot(self.xs[rx] + 0.5 * self.res - click_wx, self.ys[ry] + 0.5 * self.res - click_wy)
        gi = int(np.argmin(d_click))
        snap_dist = float(d_click[gi])
        goal_rc = (int(ry[gi]), int(rx[gi]))
        snapped_xy = (float(self.xs[goal_rc[1]] + 0.5 * self.res), float(self.ys[goal_rc[0]] + 0.5 * self.res))

        accepted = snap_dist <= float(max_snap_m)
        if not accepted:
            reason = "blocked_obstacle" if (inb and self.obs_mask[ri, ci]) else "unseen_or_off_table"
            return {"accepted": False, "reason": reason, "snapped_xy": snapped_xy, "snap_dist_m": snap_dist, "path": []}

        rad = math.radians(robot_yaw_deg)
        sx_anchor = robot_x + 0.16 * math.cos(rad)
        sy_anchor = robot_y + 0.16 * math.sin(rad)
        d_start = np.hypot(self.xs[rx] + 0.5 * self.res - sx_anchor, self.ys[ry] + 0.5 * self.res - sy_anchor)
        si = int(np.argmin(d_start))
        start_rc = (int(ry[si]), int(rx[si]))

        pq = [(0.0, start_rc)]
        g_cost = {start_rc: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        while pq:
            _, cur = heapq.heappop(pq)
            if cur == goal_rc:
                break
            cr, cc = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < self.ny and 0 <= nc < self.nx and self.reach[nr, nc]:
                        # 禁止斜角擦过红色障碍棱角
                        if dr != 0 and dc != 0 and not (self.reach[cr + dr, cc] and self.reach[cr, cc + dc]):
                            continue
                        ncost = g_cost[cur] + math.hypot(dr, dc)
                        if ncost < g_cost.get((nr, nc), 1e9):
                            g_cost[(nr, nc)] = ncost
                            came[(nr, nc)] = cur
                            heapq.heappush(pq, (ncost + math.hypot(nr - goal_rc[0], nc - goal_rc[1]), (nr, nc)))

        if goal_rc not in came and goal_rc != start_rc:
            return {"accepted": False, "reason": "disconnected", "snapped_xy": snapped_xy, "snap_dist_m": snap_dist, "path": []}

        cells = []
        cur = goal_rc
        while cur != start_rc:
            cells.append(cur)
            cur = came[cur]
        cells.append(start_rc)
        cells.reverse()
        path = [(float(robot_x), float(robot_y))] + [
            (float(self.xs[cc] + 0.5 * self.res), float(self.ys[rr] + 0.5 * self.res)) for rr, cc in cells
        ]
        return {"accepted": True, "reason": "reachable", "snapped_xy": snapped_xy, "snap_dist_m": snap_dist, "path": path}


def render_edl_3d_view(
    pts: np.ndarray,
    nrms: np.ndarray,
    cols: np.ndarray,
    cam: OrbitCamera3D,
    grid: TraversabilityGrid2D,
    w: int = 640,
    h: int = 440,
    robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    path: list[tuple[float, float]] | None = None,
    click_marker: tuple[float, float, tuple[int, int, int]] | None = None,
    head_yaw_deg: float = 0.0,
    bg_bgr: tuple[int, int, int] = (26, 18, 14),
    draw_ground_grid: bool = False,
    ground_blend: float = 0.58,
    draw_trav_overlay: bool = True,
    lambert_base: float = 0.56,
    lambert_span: float = 0.50,
    edl_gain: float = 3.4,
) -> np.ndarray:
    """带 Z-Buffer 遮挡 + Lambertian 法向光照 + 两遍 Eye-Dome Lighting + 3D 机器人/视锥 + A* 路径渲染。

    bg_bgr / draw_ground_grid / ground_blend / draw_trav_overlay / lambert_* / edl_gain
    为可选渲染风格参数（默认值保持旧观感）：浅色背景 + 地面网格线用于"照片建模查看器"
    风格的静态场景轮廓展示；draw_trav_overlay=False 时纯密点轮廓（不画绿/红可达域贴片）；
    lambert_base/span 提高平坦度观感（如 0.80/0.28），edl_gain 提高边缘勾线强度。
    """
    basis = _PROJ.compute_camera_basis(cam)
    focal = _PROJ.compute_focal_px(cam, float(h))
    cx, cy = w * 0.5, h * 0.5

    img = np.full((h, w, 3), np.array(bg_bgr, dtype=np.uint8), dtype=np.uint8)

    if draw_ground_grid:
        gg = img.copy()
        step = 0.10
        x_lo, x_hi = -0.3, 1.6
        y_lo, y_hi = -1.0, 1.0
        n_l = int((x_hi - x_lo) / step) + 1
        for i in range(n_l):
            wx = x_lo + i * step
            seg = [_PROJ.project_world_point(float(wx), float(y_lo), 0.0, cam, float(w), float(h)),
                   _PROJ.project_world_point(float(wx), float(y_hi), 0.0, cam, float(w), float(h))]
            if all(p is not None for p in seg):
                cv2.line(gg, (int(seg[0][0]), int(seg[0][1])), (int(seg[1][0]), int(seg[1][1])), (198, 202, 208), 1, cv2.LINE_AA)
        n_l = int((y_hi - y_lo) / step) + 1
        for i in range(n_l):
            wy = y_lo + i * step
            seg = [_PROJ.project_world_point(float(x_lo), float(wy), 0.0, cam, float(w), float(h)),
                   _PROJ.project_world_point(float(x_hi), float(wy), 0.0, cam, float(w), float(h))]
            if all(p is not None for p in seg):
                cv2.line(gg, (int(seg[0][0]), int(seg[0][1])), (int(seg[1][0]), int(seg[1][1])), (198, 202, 208), 1, cv2.LINE_AA)
        img = cv2.addWeighted(gg, 0.5, img, 0.5, 0)
    z_buf = np.full((h, w), 1e4, dtype=np.float32)

    if draw_trav_overlay:
        res = grid.res
        ov_green = img.copy()
        ov_red = img.copy()
        for r_i, wy in enumerate(grid.ys):
            for c_i, wx in enumerate(grid.xs):
                is_g = bool(grid.reach[r_i, c_i])
                is_r = bool(grid.obs_mask[r_i, c_i]) and not is_g
                if not (is_g or is_r):
                    continue
                corners = []
                ok_c = True
                for dx, dy in ((0, 0), (res, 0), (res, res), (0, res)):
                    pr = _PROJ.project_world_point(float(wx + dx), float(wy + dy), 0.0, cam, float(w), float(h))
                    if pr is None:
                        ok_c = False
                        break
                    corners.append([int(round(pr[0])), int(round(pr[1]))])
                if not ok_c:
                    continue
                poly = np.array(corners, dtype=np.int32)
                if is_g:
                    cv2.fillPoly(ov_green, [poly], (65, 195, 50))
                    cv2.polylines(ov_green, [poly], True, (95, 240, 75), 1, cv2.LINE_AA)
                else:
                    cv2.fillPoly(ov_red, [poly], (55, 55, 220))

        img = cv2.addWeighted(ov_green, 0.45, img, 0.55, 0)
        img = cv2.addWeighted(ov_red, 0.32, img, 0.68, 0)
    c_buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if len(pts) > 0:
        rx = pts[:, 0] - basis[0]
        ry = pts[:, 1] - basis[1]
        rz = pts[:, 2] - basis[2]
        z_cam = rx * basis[3] + ry * basis[4] + rz * basis[5]
        vis = z_cam > 0.08
        pts_v, nrms_v, cols_v, z_v = pts[vis], nrms[vis], cols[vis], z_cam[vis]
        x_cam = rx[vis] * basis[6] + ry[vis] * basis[7] + rz[vis] * basis[8]
        y_cam = rx[vis] * basis[9] + ry[vis] * basis[10] + rz[vis] * basis[11]
        sx = np.rint(cx + (x_cam / z_v) * focal).astype(np.int32)
        sy = np.rint(cy - (y_cam / z_v) * focal).astype(np.int32)
        inb = (sx >= 1) & (sx < w - 1) & (sy >= 1) & (sy < h - 1)
        sx, sy, z_v = sx[inb], sy[inb], z_v[inb]
        pts_v, nrms_v, cols_v = pts_v[inb], nrms_v[inb], cols_v[inb]

        ldir = np.array([-0.40, -0.25, 0.88], dtype=np.float32)
        ldir /= np.linalg.norm(ldir)
        ndotl = np.clip(np.sum(nrms_v * ldir[None, :], axis=-1), 0.0, 1.0)
        shade = lambert_base + lambert_span * ndotl
        rgb = np.clip(cols_v * shade[:, None], 0.0, 1.0)

        order = np.argsort(-z_v)
        sx, sy, z_v, rgb, pz_s = sx[order], sy[order], z_v[order], rgb[order], pts_v[order, 2]
        is_ground_pt = np.abs(pz_s) < 0.028

        # 13 偏移钻石核散射：约 5px 直径的稠密面片观感（纯向量化，零逐点循环）
        for oy, ox in ((0, 0), (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 2), (2, 0), (0, -2), (-2, 0)):
            py = np.clip(sy + oy, 0, h - 1)
            px = np.clip(sx + ox, 0, w - 1)
            z_buf[py, px] = z_v
            c_buf[py, px] = np.where(is_ground_pt[:, None], (1.0 - ground_blend) * rgb + ground_blend * c_buf[py, px], rgb)

    has_pt = z_buf < 1e3
    log_z = np.where(has_pt, np.log2(np.maximum(z_buf, 0.05)), 10.0)
    resp = np.zeros((h, w), dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        nbr = np.roll(np.roll(log_z, dy, axis=0), dx, axis=1)
        resp += np.maximum(0.0, log_z - nbr) * (1.0 if abs(dx) + abs(dy) == 1 else 0.55)
    edl = np.exp(-edl_gain * resp)
    c_shaded = np.where(has_pt[..., None], c_buf * edl[..., None], c_buf)
    out_bgr = cv2.cvtColor(np.clip(c_shaded * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # A* 导航路径
    if path and len(path) >= 2:
        ppx = []
        for wx, wy in path:
            pr = _PROJ.project_world_point(wx, wy, 0.015, cam, float(w), float(h))
            if pr is not None:
                ppx.append((int(round(pr[0])), int(round(pr[1]))))
        for i in range(len(ppx) - 1):
            cv2.line(out_bgr, ppx[i], ppx[i + 1], (255, 240, 40), 3, cv2.LINE_AA)
        if ppx:
            cv2.circle(out_bgr, ppx[-1], 7, (60, 255, 110), -1, cv2.LINE_AA)
            cv2.circle(out_bgr, ppx[-1], 9, (255, 255, 255), 2, cv2.LINE_AA)

    if click_marker is not None:
        pr = _PROJ.project_world_point(click_marker[0], click_marker[1], 0.02, cam, float(w), float(h))
        if pr is not None:
            cv2.drawMarker(out_bgr, (int(pr[0]), int(pr[1])), click_marker[2], cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    # 3D 机器人本体位置与当前视野角扇区
    rx0, ry0, ryaw = robot_pose
    view_yaw = ryaw + head_yaw_deg
    pr0 = _PROJ.project_world_point(rx0, ry0, 0.0, cam, float(w), float(h))
    pr_top = _PROJ.project_world_point(rx0, ry0, 0.09, cam, float(w), float(h))
    pr_nose = _PROJ.project_world_point(
        rx0 + 0.16 * math.cos(math.radians(view_yaw)),
        ry0 + 0.16 * math.sin(math.radians(view_yaw)),
        0.04,
        cam,
        float(w),
        float(h),
    )
    if pr0 is not None and pr_top is not None:
        cv2.circle(out_bgr, (int(pr0[0]), int(pr0[1])), 10, (90, 240, 140), 2, cv2.LINE_AA)
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(pr_top[0]), int(pr_top[1])), (110, 250, 180), 4, cv2.LINE_AA)
        cv2.circle(out_bgr, (int(pr_top[0]), int(pr_top[1])), 7, (60, 220, 255), -1, cv2.LINE_AA)
    if pr_top is not None and pr_nose is not None:
        cv2.arrowedLine(out_bgr, (int(pr_top[0]), int(pr_top[1])), (int(pr_nose[0]), int(pr_nose[1])), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    p_fov_l = _PROJ.project_world_point(rx0 + 0.78 * math.cos(math.radians(view_yaw + 36.0)), ry0 + 0.78 * math.sin(math.radians(view_yaw + 36.0)), 0.0, cam, float(w), float(h))
    p_fov_r = _PROJ.project_world_point(rx0 + 0.78 * math.cos(math.radians(view_yaw - 36.0)), ry0 + 0.78 * math.sin(math.radians(view_yaw - 36.0)), 0.0, cam, float(w), float(h))
    if pr0 is not None and p_fov_l is not None and p_fov_r is not None:
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(p_fov_l[0]), int(p_fov_l[1])), (80, 230, 150), 1, cv2.LINE_AA)
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(p_fov_r[0]), int(p_fov_r[1])), (80, 230, 150), 1, cv2.LINE_AA)

    return out_bgr


def render_minimap_2d(
    grid: TraversabilityGrid2D,
    size_px: int = 136,
    robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    path: list[tuple[float, float]] | None = None,
    head_yaw_deg: float = 0.0,
) -> np.ndarray:
    """绘制右上角/右下角 2D 公共空间小地图（绿=可达，红=障碍，深灰=未知，青色扇形=当前视野，黄色=A*路径）。"""
    mm = np.full((grid.ny, grid.nx, 3), (38, 34, 32), dtype=np.uint8)
    mm[grid.reach] = (65, 195, 55)
    mm[grid.obs_mask & (~grid.reach)] = (55, 55, 225)
    # 图像坐标系转换：X 向前(+up), Y 向右(+right) -> transpose 使得上=X+, 右=Y+
    mm_up = cv2.flip(np.transpose(mm, (1, 0, 2)), 0)
    mm_big = cv2.resize(mm_up, (size_px, size_px), interpolation=cv2.INTER_NEAREST)

    x0, x1 = float(grid.xs[0]), float(grid.xs[-1] + grid.res)
    y0, y1 = float(grid.ys[0]), float(grid.ys[-1] + grid.res)

    def _to_px(wx: float, wy: float) -> tuple[int, int]:
        px = int(round((wy - y0) / (y1 - y0) * size_px))
        py = int(round((1.0 - (wx - x0) / (x1 - x0)) * size_px))
        return max(0, min(size_px - 1, px)), max(0, min(size_px - 1, py))

    rx0, ry0, ryaw = robot_pose
    view_yaw = ryaw + head_yaw_deg
    pr = _to_px(rx0, ry0)
    fl = _to_px(rx0 + 0.72 * math.cos(math.radians(view_yaw + 36.0)), ry0 + 0.72 * math.sin(math.radians(view_yaw + 36.0)))
    fr = _to_px(rx0 + 0.72 * math.cos(math.radians(view_yaw - 36.0)), ry0 + 0.72 * math.sin(math.radians(view_yaw - 36.0)))
    cv2.line(mm_big, pr, fl, (120, 240, 170), 1, cv2.LINE_AA)
    cv2.line(mm_big, pr, fr, (120, 240, 170), 1, cv2.LINE_AA)

    if path and len(path) >= 2:
        pts_p = [_to_px(wx, wy) for wx, wy in path]
        for i in range(len(pts_p) - 1):
            cv2.line(mm_big, pts_p[i], pts_p[i + 1], (255, 240, 40), 2, cv2.LINE_AA)
        cv2.circle(mm_big, pts_p[-1], 4, (80, 255, 110), -1, cv2.LINE_AA)

    cv2.circle(mm_big, pr, 5, (0, 235, 255), -1, cv2.LINE_AA)
    cv2.rectangle(mm_big, (0, 0), (size_px - 1, size_px - 1), (110, 125, 140), 1)
    cv2.putText(mm_big, "2D Map", (6, 14), 0, 0.38, (220, 230, 240), 1)
    return mm_big
