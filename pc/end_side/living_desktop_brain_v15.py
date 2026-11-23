# -*- coding: utf-8 -*-
"""living_desktop_brain_v15.py — HoverLivingMind v15（“文文·活体桌面伴侣大脑” 4 项零开挂高阶具身智能套件）。

彻底突破“单纯 A 点走到 B 点”，在 100% 零轮式里程计、100% 零真值偷看（Zero-Oracle）、自研 HoverGeoNet-v4
与 PlanarManifoldBundle + HoverSDF-Net + EndSidePolicyNet 统一框架下，打造 4 种让机器人“像活体桌面宠物一样思考”的高阶行为：

  L1. 视觉悬崖本能与拒跳抗议 (Visual Cliff Survival & Edge-Refusal Head-Shake):
      - 真实光线追踪在桌沿 `y_cliff = -0.36 m` 处截断桌面，下方是 72cm 深的暗色地面深渊；
      - 人类故意在悬崖外 `(0.46, -0.44)` 敲桌子诱骗机器人跳崖！
      - 对照组 (无视觉悬崖感知) 盲从指令直接冲下桌沿坠落；
      - v15 仅凭 240×240 相机画面 + HoverGeoNet-v4 地面射线，纯视觉发现桌沿悬崖线 -> 建立虚拟悬崖 CBF 墙 ->
        在距崖边 > 3cm 处紧急刹停 -> 连续摇头抗议 (`head: -28° -> +28° -> -24° -> 0°`) -> 回头凝视人类！

  L2. 防捉弄 / 假动作免疫 / 半路改口 + 主动寻人 (Teasing Immunity + Dynamic Switch + Active Face Search):
      - 人类先敲点 A `(0.58, -0.24)` -> 半路做一次单敲/挥手假动作企图骗它 -> v15 识别为假动作无视！
      - 随后人类在水杯另一侧真双敲点 B `(0.48, +0.14)` -> v15 瞬间放弃 A、绕开水杯左弧线精准停靠 B；
      - 期间人类悄悄把椅子挪到新方位 `(1.05, -0.34)` -> v15 回看旧人脸方向发现空无一人 -> 切换 `SEARCH_HUMAN`
        主动扫视房间 -> 相机扫到新座位后立刻精准锁眸 (`< 1.5°`)！

  L3. 皮亚杰四阶段物体恒存 + 好奇猫式探头绕障 (Object Permanence & Next-Best-View Occluded Peek):
      - 一个高于机器人视线 (`height=0.175 m > H_cam=0.135 m`) 的不透明高保温杯挡死机器人与隐藏敲击点 `(0.58, -0.25)` 的直视线；
      - 对照组 (直视贪心) 盲目直冲或撞杯；
      - v15 记住被遮挡目标（物体恒存），视觉发现视线被高障碍挡死 -> 计算信息增益最优侧向探头点 (Next-Best-View) ->
        侧滑绕到侧面探头确认背面空旷 -> 再优雅切入并停靠隐藏目标（零碰撞）！

  L4. 地狱三连拷打终极通关 (The Chaos Gauntlet — 三重攻击在同一次连续任务中接连爆发！):
      - 攻击 #1 (Step 4): 移动水杯以 `0.105 m/s` 横穿走廊 -> 0 步视觉发现 + 卡尔曼未来速度障碍管提前让行绕开；
      - 攻击 #2 (Step 18): 刚绕过移动杯，半路突降全新静止圆柱障碍 -> 0 步发现 + 修复版静止/运动物理假设门控（杜绝 v14 误飘蓝环！）精确贝叶斯绕行；
      - 攻击 #3 (Step 30): 刚过圆柱又遭暴力劫持 (`+88°` 盲转 + `12 cm` 横推) -> 物理跳变守卫急刹 + 主动扫描 +
        重力对齐 2 点 `SE(2)` 全局 RANSAC 零先验拉回 -> 牢记原目标并零碰撞完赛！

输出：
  - `D:/robot_work/runs_end_side/living_brain_v15/living_brain_v15_report.json`（含全轨迹供 HTML5 互动剧场加载）
  - `D:/robot_work/runs_end_side/living_brain_v15/showcase_v15_living_desktop_brain.png`（4 合 1 科研海报）
  - `D:/robot_work/runs_end_side/living_brain_v15/demo_living_brain_v15.mp4`（无冗余静帧、带实时心情/深度双视窗电影级演示视频！）
"""

from __future__ import annotations

import json
import os
import math
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(HERE), str(REPO / "pc" / "perception"), str(REPO / "pc" / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cliff_detect import detect_visual_cliff_zero_oracle  # noqa: E402  # 与 pc_unified_brain_app 共用的零开挂悬崖检测器
from embodied_brain_extreme_v14 import (  # noqa: E402
    ARTIFACT_DIR,
    FFMPEG_EXE,
    V4_CKPT,
    detect_novel_3d_obstacle_live,
    relocalize_global_kidnap_zero_prior,
)
from end_side_controller import EndSideFeatureBuilder, EndSideSafetyShield  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from hd_dynamic_contour_slam import HoverGeoNet240Frontend  # noqa: E402
from hd_moge_contour_engine import REAL_FOV_H_DEG, REAL_H_CAM_M  # noqa: E402
from hover_fine_motion_policy_v3 import load_fine_motion_policy_v3  # noqa: E402
from hover_geonet_240 import GEO_H, GEO_W  # noqa: E402
from hover_neural_sdf_cbf import plan_smooth_neural_cbf_trajectory  # noqa: E402
from planar_se2_bundle_subcm_v11 import (  # noqa: E402
    PlanarManifoldBundleRelocalizer,
    build_initial_map_and_sdf,
)
from robot_brain_behavior_v12 import CompositeDynamicHoverSdfNet  # noqa: E402
from scenario2_visual_nav_closed_loop import DesktopArena3D, _wrap_deg  # noqa: E402
from scenario_v2 import MODE_APPROACH, TargetLock  # noqa: E402

OUT_DIR = REPO / "runs_end_side" / "living_brain_v15"


# ---------------------------------------------------------------------------
# 1. 带真实物理桌沿悬崖的射线追踪世界 (`CliffDesktopArena3D`) + 零开挂纯视觉悬崖检测器
# ---------------------------------------------------------------------------


class CliffDesktopArena3D(DesktopArena3D):
    """在标准桌面测试场右侧 (`y < cliff_y_min`) 增加真实 72cm 悬崖落差（光线越过桌沿直接照向暗色地面！）。"""

    def __init__(self, h_cam_m: float = REAL_H_CAM_M, fov_h_deg: float = REAL_FOV_H_DEG, cliff_y_min: float | None = None) -> None:
        super().__init__(h_cam_m=h_cam_m, fov_h_deg=fov_h_deg)
        self.cliff_y_min = cliff_y_min

    def render_camera(
        self,
        rx: float,
        ry: float,
        yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        noise_seed: int = 0,
    ) -> dict[str, Any]:
        cf = super().render_camera(rx, ry, yaw_deg, pitch_deg, roll_deg, noise_seed=noise_seed)
        if self.cliff_y_min is None:
            return cf

        # 精确反算哪些像素击中了原本的桌面 Z=0，将 y < cliff_y_min 的区域替换为 72cm 下方的暗色深渊地面！
        r_c2w = cf["R_c2w"]
        rays = (self.rays_cam @ r_c2w.T).astype(np.float32)
        cam_o = np.array([rx, ry, self.h_cam_m], dtype=np.float32)
        t_desk = np.where(rays[..., 2] < -1e-4, -cam_o[2] / np.minimum(rays[..., 2], -1e-4), 3.0)
        depth = cf["depth_gt"].copy()
        is_desk_hit = (t_desk > 0.12) & (np.abs(depth - t_desk) < 0.015)

        px_d = cam_o[0] + t_desk * rays[..., 0]
        py_d = cam_o[1] + t_desk * rays[..., 1]

        # 1.5cm 宽的木桌边缘倒角条纹（真实桌沿高光反光）
        is_rim = is_desk_hit & (py_d >= self.cliff_y_min) & (py_d < self.cliff_y_min + 0.015)
        # 越过桌沿 -> 跌落到 Z = -0.72m 的暗色房间地面
        is_void = is_desk_hit & (py_d < self.cliff_y_min)

        bgr = cf["bgr"].copy()
        if np.any(is_rim):
            bgr[is_rim] = np.clip(bgr[is_rim].astype(np.int16) + 28, 0, 245).astype(np.uint8)
        if np.any(is_void):
            t_floor = np.where(rays[..., 2] < -1e-4, (-0.72 - cam_o[2]) / np.minimum(rays[..., 2], -1e-4), 3.0)
            depth[is_void] = np.minimum(t_floor[is_void], 3.0)
            px_f = cam_o[0] + t_floor * rays[..., 0]
            py_f = cam_o[1] + t_floor * rays[..., 1]
            floor_g = np.clip(22.0 + 7.0 * np.sin(px_f * 9.0) * np.cos(py_f * 9.0), 12.0, 36.0).astype(np.uint8)
            bgr[is_void, 0] = floor_g[is_void]
            bgr[is_void, 1] = floor_g[is_void]
            bgr[is_void, 2] = floor_g[is_void]

        cf["bgr"] = bgr
        cf["depth_gt"] = depth
        return cf


# ---------------------------------------------------------------------------
# 2. 修复版静止/运动双假设 2D 卡尔曼跟踪器（彻底根治 v14 E4 的后期误判移动蓝环！）
# ---------------------------------------------------------------------------


class RobustDynamicObstacleTracker:
    """升级版 2D 贝叶斯 + 卡尔曼速度跟踪器：
    - 提高运动假设检验门限 (`net_disp >= 0.088 m`, 线性相关系数 `r2 >= 0.72`, `speed >= 0.055 m/s`)：
      真实移动水杯 (0.105 m/s, 5帧走 0.085-0.10m 且高度线性) 稳定触发卡尔曼速度管；
      静止突发圆柱的 3-4cm 单目深度抖动永远不会误翻转成“移动障碍”！
    - 离开视野后静止障碍位置严格锁死不飘移；移动障碍 5 步后自动清理让行。
    """

    def __init__(self, dt: float = 0.16) -> None:
        self.dt = float(dt)
        self.initialized = False
        self.is_moving = False
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag([0.04 ** 2, 0.04 ** 2, 0.14 ** 2, 0.14 ** 2])
        self.r_est = 0.040
        self.n_obs = 0
        self.last_obs_step = -99
        self.last_move_step = -99
        self._move_ref: tuple[float, float] | None = None
        self.obs_hist: list[tuple[int, float, float]] = []

    def reset(self) -> None:
        self.initialized = False
        self.is_moving = False
        self.x[:] = 0.0
        self.n_obs = 0
        self.last_obs_step = -99
        self.last_move_step = -99
        self._move_ref = None
        self.obs_hist.clear()

    def predict(self, cur_step: int) -> None:
        if not self.initialized:
            return
        if not self.is_moving:
            self.x[2] = 0.0
            self.x[3] = 0.0
            return
        F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Q = np.diag([0.006 ** 2, 0.006 ** 2, 0.028 ** 2, 0.028 ** 2])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if cur_step - self.last_obs_step > 2:
            self.x[2] *= 0.80
            self.x[3] *= 0.80

    def update(self, obs_x: float, obs_y: float, obs_r: float, step: int) -> bool:
        if not self.initialized:
            self.x = np.array([obs_x, obs_y, 0.0, 0.0], dtype=np.float64)
            self.r_est = float(obs_r)
            self.initialized = True
            self.n_obs = 1
            self.last_obs_step = step
            self.obs_hist = [(step, obs_x, obs_y)]
            return True

        # 若旧移动障碍已消失 >= 2 步，且画面里出现一个全新位置的突发障碍 -> 自动开启新轨道！
        dist_jump = math.hypot(obs_x - self.x[0], obs_y - self.x[1])
        # 累积位移 >3cm 记为"有效移动观测"（慢速杯每步仅 ~1cm，单步差分会把移动目标误当静止）
        if self._move_ref is None or math.hypot(obs_x - self._move_ref[0], obs_y - self._move_ref[1]) > 0.03:
            self.last_move_step = step
            self._move_ref = (obs_x, obs_y)
        if dist_jump > 0.14:
            if (step - self.last_obs_step >= 2) and self.is_moving:
                self.reset()
                return self.update(obs_x, obs_y, obs_r, step)
            return False

        self.obs_hist.append((step, obs_x, obs_y))
        self.r_est = 0.72 * self.r_est + 0.28 * float(obs_r)
        self.n_obs += 1
        self.last_obs_step = step

        # 严格运动假设检验（位移 >= 7.8cm + 线性拟合速度 >= 0.055 m/s + 轨迹直线度高）
        win = self.obs_hist[-7:]
        if (not self.is_moving) and len(win) >= 4:
            ts = np.array([w[0] * self.dt for w in win], dtype=np.float64)
            xs = np.array([w[1] for w in win], dtype=np.float64)
            ys = np.array([w[2] for w in win], dtype=np.float64)
            span_t = float(ts[-1] - ts[0])
            net_disp = math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]))
            if span_t >= 3.0 * self.dt - 1e-6 and net_disp >= 0.075:
                tc = ts - ts.mean()
                denom = max(float(np.dot(tc, tc)), 1e-6)
                vx_ols = float(np.dot(tc, xs - xs.mean()) / denom)
                vy_ols = float(np.dot(tc, ys - ys.mean()) / denom)
                # 拟合残差检验：真实匀速移动物体的线性拟合残差很小 (< 2.6cm)
                rx = xs - (xs.mean() + vx_ols * tc)
                ry = ys - (ys.mean() + vy_ols * tc)
                rmse = float(np.sqrt(np.mean(rx * rx + ry * ry)))
                if math.hypot(vx_ols, vy_ols) >= 0.055 and rmse < 0.026:
                    self.is_moving = True
                    self.x = np.array([obs_x, obs_y, vx_ols, vy_ols], dtype=np.float64)
                    self.P = np.diag([0.020 ** 2, 0.020 ** 2, 0.042 ** 2, 0.042 ** 2])
                    return True

        if not self.is_moving:
            xs = [w[1] for w in self.obs_hist]
            ys = [w[2] for w in self.obs_hist]
            self.x[0] = float(np.median(xs))
            self.x[1] = float(np.median(ys))
            self.x[2] = 0.0
            self.x[3] = 0.0
            return True

        z = np.array([obs_x, obs_y], dtype=np.float64)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.diag([0.022 ** 2, 0.022 ** 2])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return True

    def get_spheres(self, horizon_steps: int = 5, pad_r: float = 0.022) -> list[tuple[float, float, float]]:
        if not self.initialized:
            return []
        r_eff = float(self.r_est + pad_r)
        if not self.is_moving:
            return [(float(self.x[0]), float(self.x[1]), r_eff)]
        return [
            (float(self.x[0] + self.x[2] * k * self.dt), float(self.x[1] + self.x[3] * k * self.dt), float(r_eff + 0.004 * k))
            for k in range(horizon_steps + 1)
        ]


# ---------------------------------------------------------------------------
# 3. 统一活体具身大脑推演引擎（支持 L1 悬崖拒跳 / L2 防捉弄换点寻人 / L3 NBV 探头 / L4 三连拷打！）
# ---------------------------------------------------------------------------


def _tracker_reinit(tracker, obs_x: float, obs_y: float, obs_r: float, step: int) -> None:
    """绕过 14cm 关联门硬重置追踪器——追捕失锁后杯子可能已换位，远距检测就是同一只杯子。"""
    tracker.x = np.array([obs_x, obs_y, 0.0, 0.0], dtype=np.float64)
    tracker.P = np.diag([0.04 ** 2, 0.04 ** 2, 0.15 ** 2, 0.15 ** 2])
    tracker.r_est = float(obs_r)
    tracker.initialized = True
    tracker.is_moving = False
    tracker.n_obs = 1
    tracker.last_obs_step = step
    tracker.last_move_step = step
    tracker._move_ref = (float(obs_x), float(obs_y))
    tracker.obs_hist = [(step, float(obs_x), float(obs_y))]


def _segment_hits_circle(p0: tuple[float, float], p1: tuple[float, float], cx: float, cy: float, r: float) -> bool:
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    l2 = vx * vx + vy * vy
    if l2 < 1e-8:
        return math.hypot(p0[0] - cx, p0[1] - cy) <= r
    t = max(0.0, min(1.0, ((cx - p0[0]) * vx + (cy - p0[1]) * vy) / l2))
    px, py = p0[0] + t * vx, p0[1] + t * vy
    return math.hypot(px - cx, py - cy) <= r


def _ballistic_yield_goal(
    cx: float, cy: float, cvx: float, cvy: float,
    ex: float, ey: float, r_est: float,
    horizon_s: float = 0.55, escape_m: float = 0.22,
) -> tuple[float, float] | None:
    """v24b 弹道让位：预测杯子 [c, c+v*horizon] 弹道线段与机器人当前位置的
    最近距离 < r_est+0.045（即即将擦身/穿透）时，返回垂直于杯速方向的让位点。
    只对"杯中可见且 tracker 速度有效"生效——失锁遮挡段卡尔曼速度衰减到
    <0.05 自动不触发（看不见的弹道让不了，属诚实边界）。"""
    cspd = math.hypot(cvx, cvy)
    if cspd < 0.05:
        return None
    px, py = cx + cvx * horizon_s, cy + cvy * horizon_s
    sx, sy = px - cx, py - cy
    seg2 = max(sx * sx + sy * sy, 1e-9)
    tt = max(0.0, min(1.0, ((ex - cx) * sx + (ey - cy) * sy) / seg2))
    qx, qy = cx + sx * tt, cy + sy * tt
    if math.hypot(ex - qx, ey - qy) >= r_est + 0.045:
        return None
    nx, ny = -cvy / cspd, cvx / cspd
    side = 1.0 if (ex - qx) * nx + (ey - qy) * ny >= 0.0 else -1.0
    return (ex + nx * side * escape_m, ey + ny * side * escape_m)


def _compute_nbv_peek_point(
    rob_xy: tuple[float, float],
    tgt_xy: tuple[float, float],
    occ_xyz: tuple[float, float, float],
    sdf_base: Any,
) -> tuple[float, float]:
    """信息增益最优侧向探头点 (Next-Best-View)：绕过高障碍物边缘，在安全净空下最大化对隐藏目标的直视线！"""
    ocx, ocy, ocr = occ_xyz
    dev = next(sdf_base.parameters()).device
    best_pt = (ocx - 0.04, ocy - ocr - 0.14)
    best_score = -1e9
    xs = np.linspace(max(0.16, ocx - 0.14), min(0.46, ocx + 0.10), 13)
    ys = np.linspace(-0.29, 0.10, 17)
    grid = np.array([(x, y) for x in xs for y in ys], dtype=np.float32)
    with torch.no_grad():
        sdfs = sdf_base(torch.from_numpy(grid).to(dev))[0].cpu().numpy().ravel()
    for (gx, gy), sd in zip(grid, sdfs):
        d_occ = math.hypot(float(gx) - ocx, float(gy) - ocy) - ocr
        if sd < 0.090 or d_occ < 0.105:
            continue
        # 目标直视线必须不被高遮挡物挡住，且机器人到探头点也尽量无遮挡
        vis_tgt = not _segment_hits_circle((float(gx), float(gy)), tgt_xy, ocx, ocy, ocr + 0.025)
        vis_rob = not _segment_hits_circle(rob_xy, (float(gx), float(gy)), ocx, ocy, ocr + 0.015)
        if not vis_tgt:
            continue
        # 偏好靠近桌中段、离当前位置近、离障碍净空充足、且能清晰斜视障碍背后的视角
        ang_gain = abs(math.atan2(tgt_xy[1] - gy, tgt_xy[0] - gx) - math.atan2(ocy - gy, ocx - gx))
        score = (1.2 if vis_rob else 0.0) + 0.8 * min(ang_gain, 1.2) + 1.5 * min(d_occ, 0.18) - 1.8 * math.hypot(gx - rob_xy[0], gy - rob_xy[1]) - 0.5 * math.hypot(gx - tgt_xy[0], gy - tgt_xy[1])
        if score > best_score:
            best_score = score
            best_pt = (round(float(gx), 3), round(float(gy), 3))
    return best_pt


def run_living_brain_episode(
    arena: CliffDesktopArena3D,
    map_pack: dict[str, Any],
    geo_fe: HoverGeoNet240Frontend,
    policy_v3: Any,
    init_target_xy: tuple[float, float],
    *,
    enable_cliff_instinct: bool = True,
    enable_nbv_peek: bool = False,
    enable_global_recovery: bool = True,
    enable_velocity_prediction: bool = True,
    fakeout_event: dict[str, Any] | None = None,        # {"step": 10, "xy": (0.44, 0.16)}
    switch_target_event: dict[str, Any] | None = None,  # {"step": 16, "xy": (0.48, 0.14)}
    human_face_init: tuple[float, float] = (1.10, 0.08),
    human_face_moved: tuple[float, float] | None = None,  # (1.05, -0.34)
    static_occluder: dict[str, float] | None = None,    # Tall thermos blocking LOS in L3
    moving_obs_cfg: dict[str, float] | None = None,
    static_drop_obs: dict[str, float] | None = None,
    kidnap_event: dict[str, float] | None = None,
    pursuit_mode: bool = False,          # v18: the moving cylinder IS the goal (chase+escort)
    edge_survey_samples: list | None = None,  # v19: 每步累积 cliff 边界观测样本（拒跳闩存照旧生效）
    chase_standoff_m: float = 0.085,
    pursuit_lead_s: float = 0.45,
    slip_factor: float = 0.78,
    yaw_drift_dps: float = 5.0,
    seed: int = 15001,
    max_steps: int = 120,
    dt: float = 0.16,
    record_frames: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    shield = EndSideSafetyShield()
    policy_v3.reset_state()
    reloc = PlanarManifoldBundleRelocalizer(h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
    for bk, pwk, nwk, vgk, pk in map_pack["init_kfs_v11"]:
        reloc.add_keyframe(bk, pwk, nwk, vgk, pk)

    spts = map_pack["static_pts"]
    static_obs_tree = cKDTree(spts[spts[:, 2] > 0.022, :2])
    sdf_base = map_pack["sdf_pack"]["model"]

    target_xy = tuple(init_target_xy)
    active_goal_xy = tuple(init_target_xy)
    cbf0 = plan_smooth_neural_cbf_trajectory(sdf_base, (0.0, 0.0), active_goal_xy, n_knots=22, safe_radius_m=0.102, n_opt_steps=120)
    path_xy = [(float(p[0]), float(p[1])) for p in cbf0["path_xy"]]

    tracker = RobustDynamicObstacleTracker(dt=dt)
    tx = ty = tyaw = ex = ey = eyaw = cam_ex = cam_ey = 0.0
    prev_ex = prev_ey = prev_eyaw = 0.0
    prev_b = map_pack["init_kfs_v11"][0][0]
    last_kf = (0.0, 0.0, 0.0)
    p_nom = 3.0
    est_slip = 1.0
    prev_cmd_vx = 0.0

    # 活体高级认知状态变量
    mood = "APPROACH"
    mood_text = "Heading to knock spot"
    head_deg = 0.0
    cliff_detected = False
    cliff_y_est = None
    cliff_refusal_step = -1
    cliff_emote_timer = 0
    fell_off_cliff = False
    min_cliff_margin_cm = 999.0

    fakeout_ignored = False
    target_switched = False
    docked = False
    post_dock_phase = "NONE"  # NONE -> LOOK_OLD_FACE -> SEARCH_HUMAN -> FOUND_HUMAN
    post_dock_timer = 0
    human_found = False
    final_face_gaze_err_deg = -1.0

    nbv_active = False
    nbv_peek_xy: tuple[float, float] | None = None
    nbv_peeks_done = False
    occ_est_sphere: tuple[float, float, float] | None = None

    kidnap_triggered = False
    kidnap_recovered = False
    kidnap_recov_steps = -1
    kidnap_recov_err_cm = -1.0
    # v18 pursuit state
    captured = False
    capture_step = -1
    escort_hits = 0
    escort_total = 0
    min_chase_edge_m = 999.0
    min_static_clear_m = 999.0
    cup_rest_steps = 0
    cmd_motion_acc = 0.0
    est_motion_acc = 0.0
    pose_stuck_events = 0
    pursuit_end = False
    attack1_det_step = -1
    attack2_det_step = -1
    vel_errs_mps: list[float] = []
    drop_loc_errs_cm: list[float] = []
    last_replan_step = -99
    cleared_moving = False
    lost_streak = 0
    settle = 0
    discovered_static_obs: list[tuple[float, float, float]] = []

    traj_true = [(0.0, 0.0, 0.0)]
    cam_errs_mm: list[float] = []
    clearances_m: list[float] = []
    vis_records: list[dict[str, Any]] = []
    true_face_xy = human_face_init

    arena.extra_cylinders = []
    try:
        for step in range(max_steps):
            # --- 1) 更新真实物理环境（机器人完全不知情！） ---
            active_cyls: list[dict[str, float]] = []
            true_moving_now = None
            true_drop_now = None
            if static_occluder is not None:
                active_cyls.append(dict(static_occluder))
            if moving_obs_cfg is not None and step >= int(moving_obs_cfg["start_step"]):
                el_t = (step - int(moving_obs_cfg["start_step"])) * dt
                if "waypoints" in moving_obs_cfg:
                    wps = moving_obs_cfg["waypoints"]
                    spd = float(moving_obs_cfg.get("speed", 0.09))

                    def _walk(tt: float) -> tuple[float, float]:
                        left = spd * tt
                        for wi in range(len(wps) - 1):
                            ax, ay = float(wps[wi][0]), float(wps[wi][1])
                            bx, by = float(wps[wi + 1][0]), float(wps[wi + 1][1])
                            seg = math.hypot(bx - ax, by - ay)
                            if left <= seg:
                                f = (left / seg) if seg > 1e-6 else 0.0
                                return ax + f * (bx - ax), ay + f * (by - ay)
                            left -= seg
                        return float(wps[-1][0]), float(wps[-1][1])

                    mx, my = _walk(el_t)
                    px_p, py_p = _walk(max(el_t - dt, 0.0))
                    moving_obs_cfg["_vx_now"] = (mx - px_p) / dt
                    moving_obs_cfg["_vy_now"] = (my - py_p) / dt
                else:
                    mx = float(moving_obs_cfg["x0"] + moving_obs_cfg["vx"] * el_t)
                    my = float(moving_obs_cfg["y0"] + moving_obs_cfg["vy"] * el_t)
                    moving_obs_cfg["_vx_now"] = float(moving_obs_cfg["vx"])
                    moving_obs_cfg["_vy_now"] = float(moving_obs_cfg["vy"])
                if -0.46 < my < 0.22 and 0.14 < mx < 0.80:
                    true_moving_now = {"cx": mx, "cy": my, "rad": float(moving_obs_cfg["rad"]), "height": float(moving_obs_cfg["height"])}
                    active_cyls.append(true_moving_now)
            if static_drop_obs is not None and step >= int(static_drop_obs["step"]):
                true_drop_now = dict(static_drop_obs)
                active_cyls.append(true_drop_now)
            arena.extra_cylinders = active_cyls

            # 人类悄悄换座位（L2）
            if human_face_moved is not None and step == 18:
                true_face_xy = human_face_moved

            # 假动作挑逗 vs 真改口换点（L2）
            if fakeout_event is not None and step == int(fakeout_event["step"]):
                fakeout_ignored = True
                mood = "IGNORE_FAKEOUT"
                mood_text = "Fake-out wave/tap ignored!"
            if switch_target_event is not None and step == int(switch_target_event["step"]):
                target_switched = True
                target_xy = tuple(switch_target_event["xy"])
                active_goal_xy = target_xy
                mood = "SWITCH_TARGET"
                mood_text = f"New knock! Replanning -> ({target_xy[0]:.2f},{target_xy[1]:.2f})"
                cbf_sw = plan_smooth_neural_cbf_trajectory(sdf_base, (ex, ey), active_goal_xy, n_knots=24, safe_radius_m=0.122, n_opt_steps=160)
                path_xy = [(float(p[0]), float(p[1])) for p in cbf_sw["path_xy"]]

            # 暴力劫持（L4）
            if kidnap_event is not None and step == int(kidnap_event["step"]):
                kidnap_triggered = True
                tx += float(kidnap_event["dx"])
                ty += float(kidnap_event["dy"])
                tyaw = _wrap_deg(tyaw + float(kidnap_event["dyaw"]))

            # 坠崖物理判定（仅评测用）
            if arena.cliff_y_min is not None:
                c_margin = (ty - arena.cliff_y_min) * 100.0
                min_cliff_margin_cm = min(min_cliff_margin_cm, c_margin)
                if ty < arena.cliff_y_min:
                    fell_off_cliff = True

            # --- 2) 真实 240×240 相机渲染 + 自研视觉重定位 ---
            t_s = step * dt
            p_t = 3.0 + 3.0 * math.sin(2.0 * math.pi * 1.16 * t_s)
            r_t = -4.37 + 0.80 * math.cos(2.0 * math.pi * 0.72 * t_s)
            surge = 0.017 * math.sin(2.0 * math.pi * 1.16 * t_s)
            p_imu = p_t + float(rng.normal(0.0, 0.80))
            r_imu = r_t + float(rng.normal(0.0, 0.60))
            cx = tx + surge * math.cos(math.radians(tyaw))
            cy = ty + surge * math.sin(math.radians(tyaw))

            cf = arena.render_camera(cx, cy, tyaw, p_t, r_t, noise_seed=seed + 500 + step)

            ro = reloc.relocalize(cf["bgr"], p_imu, r_imu, (cam_ex, cam_ey, eyaw))
            if ro["ok"] and step > 0:
                jump_m = math.hypot(ro["x"] - cam_ex, ro["y"] - cam_ey)
                jump_yaw = abs(_wrap_deg(ro["yaw_deg"] - eyaw))
                if jump_m > 0.11 or jump_yaw > 26.0 or ro["inliers"] < 11:
                    ro = {"ok": False, "inliers": ro.get("inliers", 0), "reproj_px": 99.0}

            from_global = False
            if (not ro["ok"]) and enable_global_recovery:
                ro_g = relocalize_global_kidnap_zero_prior(reloc, cf["bgr"], p_imu, r_imu)
                if ro_g["ok"]:
                    ro = ro_g
                    from_global = True
                    if kidnap_triggered and not kidnap_recovered:
                        kidnap_recovered = True
                        kidnap_recov_err_cm = round(math.hypot(ro["x"] - cx, ro["y"] - cy) * 100.0, 2)
                        kidnap_recov_steps = step - int(kidnap_event["step"])
                        mood = "RECOVERED"
                        mood_text = f"Global SE(2) Relocked in {kidnap_recov_steps} steps!"
                        spheres_k = list(discovered_static_obs)
                        if tracker.initialized and not tracker.is_moving:
                            spheres_k.append((float(tracker.x[0]), float(tracker.x[1]), float(tracker.r_est + 0.018)))
                        sdf_k = CompositeDynamicHoverSdfNet(sdf_base, spheres_k) if spheres_k else sdf_base
                        cbf_k = plan_smooth_neural_cbf_trajectory(sdf_k, (ro["x"], ro["y"]), active_goal_xy, n_knots=20, safe_radius_m=0.095, n_opt_steps=110)
                        path_xy = [(float(p[0]), float(p[1])) for p in cbf_k["path_xy"]]

            reloc_ok = bool(ro["ok"])
            if reloc_ok:
                lost_streak = 0
                if mood == "KIDNAP_SCAN":
                    mood = "SWITCH_TARGET" if target_switched else "APPROACH"
                    mood_text = "Relocked! Resuming mission"
                al = 1.0 if from_global else (0.96 if ro["inliers"] >= 20 else 0.82)
                cam_ex = (1.0 - al) * cam_ex + al * ro["x"]
                cam_ey = (1.0 - al) * cam_ey + al * ro["y"]
                eyaw = _wrap_deg(eyaw + al * _wrap_deg(ro["yaw_deg"] - eyaw))
                p_est, r_est = ro["pitch_est_deg"], ro["roll_est_deg"]
                p_nom = 0.92 * p_nom + 0.08 * p_est
                lever = REAL_H_CAM_M * math.sin(math.radians(p_est - p_nom))
                ex = cam_ex - lever * math.cos(math.radians(eyaw))
                ey = cam_ey - lever * math.sin(math.radians(eyaw))

                if (
                    (not from_global)
                    and ro["inliers"] >= 16
                    and ro["reproj_px"] < 1.55
                    and math.hypot(cam_ex - last_kf[0], cam_ey - last_kf[1]) > 0.055
                    and len(reloc.keyframes) < 24
                ):
                    pr_kf = geo_fe.infer_frame(cf["bgr"], prev_b, pitch_deg=p_est, roll_deg=r_est, auto_level=False)
                    vg_kf = pr_kf["valid_grid"].copy()
                    if tracker.initialized:
                        yr_k = math.radians(eyaw)
                        Ry_k = np.array([[math.cos(yr_k), -math.sin(yr_k), 0.0], [math.sin(yr_k), math.cos(yr_k), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                        pw_k = pr_kf["pts_grid"] @ Ry_k.T + np.array([cam_ex, cam_ey, 0.0], dtype=np.float32)
                        d_dyn = np.hypot(pw_k[..., 0] - tracker.x[0], pw_k[..., 1] - tracker.x[1])
                        vg_kf = vg_kf & (d_dyn > float(tracker.r_est + 0.065))
                    yr = math.radians(eyaw)
                    Ry = np.array([[math.cos(yr), -math.sin(yr), 0.0], [math.sin(yr), math.cos(yr), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                    reloc.add_keyframe(cf["bgr"], (pr_kf["pts_grid"] @ Ry.T + np.array([cam_ex, cam_ey, 0.0], dtype=np.float32)).astype(np.float32), (pr_kf["nrm_grid"] @ Ry.T).astype(np.float32), vg_kf, (cam_ex, cam_ey, eyaw, p_est, r_est))
                    last_kf = (cam_ex, cam_ey, eyaw)
            else:
                lost_streak += 1
                p_est, r_est = p_imu, r_imu

            # 计算一帧 HoverGeoNet-v4 深度图（供悬崖检测 / 动态障碍检测 / 视频 3D 视窗共用！）
            pr_live = geo_fe.infer_frame(cf["bgr"], prev_b, pitch_deg=p_est, roll_deg=r_est, auto_level=False)
            depth_live = pr_live["depth_m"]

            # --- 3) 行为 L1：100% 零开挂纯视觉悬崖边缘本能 ---
            cliff_det = None
            _survey_now = edge_survey_samples is not None
            if enable_cliff_instinct and reloc_ok and (not cliff_detected or _survey_now):
                cliff_det = detect_visual_cliff_zero_oracle(cf["bgr"], prev_b, geo_fe, (cam_ex, cam_ey, eyaw, p_est, r_est), arena.rays_cam)
                if cliff_det is not None:
                    if _survey_now:
                        edge_survey_samples.append({
                            "step": step, "y_est": cliff_det["y_cliff_est"],
                            "ex": ex, "ey": ey, "eyaw": eyaw,
                            "dist_to_rim_m": cliff_det["dist_to_rim_m"],
                        })
                    # 走近到距悬崖边 21cm 或横向 12cm 以内才急刹拒跳（戏剧张力拉满且保证安全余量 > 5cm！）
                    if not cliff_detected and (cliff_det["dist_to_rim_m"] < 0.21 or (ey - cliff_det["y_cliff_est"] < 0.12)):
                        cliff_detected = True
                        cliff_y_est = cliff_det["y_cliff_est"]
                        cliff_refusal_step = step
                        cliff_emote_timer = 14
                        mood = "REFUSE_CLIFF"
                        mood_text = "WHOA! CLIFF! NOPE - SHAKING HEAD!"

            # --- 4) 行为 L3 & L4：纯视觉未知/移动/高遮挡障碍发现 + 物体恒存 NBV 探头 ---
            tracker.predict(step)
            vis_det = None
            brake_this_step = False
            if (static_occluder is not None or moving_obs_cfg is not None or static_drop_obs is not None) and reloc_ok and step >= 1:
                vis_det = detect_novel_3d_obstacle_live(cf["bgr"], prev_b, geo_fe, (cam_ex, cam_ey, eyaw, p_est, r_est), static_obs_tree)
                # pursuit: 屏蔽追逐目标再扫一次——让"第二个"新障碍（如突降杯）浮出水面
                vis_det2 = None
                if pursuit_mode and tracker.initialized:
                    vis_det2 = detect_novel_3d_obstacle_live(
                        cf["bgr"], prev_b, geo_fe, (cam_ex, cam_ey, eyaw, p_est, r_est),
                        static_obs_tree, exclude_xy=(float(tracker.x[0]), float(tracker.x[1])), exclude_r=0.18,
                    )
                if os.environ.get("RIG_PURSUIT_DEBUG"):
                    _v = f"({vis_det['cx']:.2f},{vis_det['cy']:.2f},r{vis_det['rad']:.2f})" if vis_det else "None"
                    _v2 = f"({vis_det2['cx']:.2f},{vis_det2['cy']:.2f},r{vis_det2['rad']:.2f})" if vis_det2 else "None"
                    print(f"    [det s{step}] det={_v} det2={_v2} trk_obs={tracker.last_obs_step}")
                    if vis_det2 is not None:
                        sp_new = (float(vis_det2["cx"]), float(vis_det2["cy"]), float(vis_det2["rad"] + 0.050))
                        near_trk = tracker.initialized and math.hypot(sp_new[0] - tracker.x[0], sp_new[1] - tracker.x[1]) < 0.22
                        stale_trk = tracker.initialized and (step - tracker.last_obs_step > 10)
                        if stale_trk and not near_trk:
                            # kidnap/遮挡后杯子可能已换位——远距新检测大概率还是那只杯子，硬重置喂回 tracker
                            _tracker_reinit(tracker, vis_det2["cx"], vis_det2["cy"], vis_det2["rad"], step)
                        elif all(math.hypot(sp_new[0] - s0[0], sp_new[1] - s0[1]) > 0.10 for s0 in discovered_static_obs):
                            discovered_static_obs.append(sp_new)
                            if attack2_det_step < 0:
                                attack2_det_step = step
                                mood = "PURSUIT_DODGE"
                                mood_text = "New obstacle on chase path! Braking..."
                                brake_this_step = True
                if vis_det is not None:
                    # L3：若发现高障碍挡死通往记忆目标的视线，且尚未探头 -> 触发皮亚杰物体恒存 NBV 侧滑探头！
                    if enable_nbv_peek and (not nbv_active) and (not nbv_peeks_done):
                        occ_est_sphere = (vis_det["cx"], vis_det["cy"], float(vis_det["rad"] + 0.015))
                        if _segment_hits_circle((ex, ey), target_xy, occ_est_sphere[0], occ_est_sphere[1], occ_est_sphere[2] + 0.04):
                            nbv_peek_xy = _compute_nbv_peek_point((ex, ey), target_xy, occ_est_sphere, sdf_base)
                            nbv_active = True
                            active_goal_xy = nbv_peek_xy
                            mood = "PEEK_NBV"
                            mood_text = f"Blocked! Peeking around -> ({nbv_peek_xy[0]:.2f},{nbv_peek_xy[1]:.2f})"
                            comp_o = CompositeDynamicHoverSdfNet(sdf_base, [occ_est_sphere])
                            cbf_p = plan_smooth_neural_cbf_trajectory(comp_o, (ex, ey), active_goal_xy, n_knots=18, safe_radius_m=0.092, n_opt_steps=100)
                            path_xy = [(float(p[0]), float(p[1])) for p in cbf_p["path_xy"]]
                            brake_this_step = True

                    # L4（或普通动态障碍跟踪）；pursuit 下远离追逐目标的新检测=另一个物体
                    was_init = tracker.initialized
                    prev_moving = tracker.is_moving
                    misdet = (
                        pursuit_mode and tracker.initialized
                        and math.hypot(vis_det["cx"] - tracker.x[0], vis_det["cy"] - tracker.x[1]) > 0.24
                    )
                    if misdet:
                        stale_trk = step - tracker.last_obs_step > 10
                        if stale_trk:
                            # tracker 失锁后的远距新检测：按杯子重捕获硬重置，而非误记成静态障碍
                            _tracker_reinit(tracker, vis_det["cx"], vis_det["cy"], vis_det["rad"], step)
                            accepted = True
                        else:
                            sp_new = (float(vis_det["cx"]), float(vis_det["cy"]), float(vis_det["rad"] + 0.050))
                            if all(math.hypot(sp_new[0] - s0[0], sp_new[1] - s0[1]) > 0.10 for s0 in discovered_static_obs):
                                discovered_static_obs.append(sp_new)
                            accepted = False
                    else:
                        accepted = tracker.update(vis_det["cx"], vis_det["cy"], vis_det["rad"], step)
                        if (not accepted) and pursuit_mode and (step - tracker.last_obs_step > 4):
                            # 14–24cm 死区：关联门拒绝但追踪已旧——杯子移位了，硬重置重捕获
                            _tracker_reinit(tracker, vis_det["cx"], vis_det["cy"], vis_det["rad"], step)
                            accepted = True
                    if accepted:
                        if true_moving_now is not None and (not was_init) and attack1_det_step < 0:
                            attack1_det_step = step
                            mood = "DODGE_CUP"
                            mood_text = "Moving cup spotted! Predicting path..."
                        if true_drop_now is not None and attack2_det_step < 0 and math.hypot(vis_det["cx"] - true_drop_now["cx"], vis_det["cy"] - true_drop_now["cy"]) < 0.14:
                            attack2_det_step = step
                            mood = "DODGE_DROP"
                            mood_text = "Sudden obstacle! Bayesian detour!"
                        if true_moving_now is not None and tracker.is_moving:
                            vel_errs_mps.append(math.hypot(tracker.x[2] - moving_obs_cfg.get("_vx_now", moving_obs_cfg.get("vx", 0.0)), tracker.x[3] - moving_obs_cfg.get("_vy_now", moving_obs_cfg.get("vy", 0.0))))
                        if true_drop_now is not None and (not tracker.is_moving) and attack2_det_step >= 0:
                            drop_loc_errs_cm.append(math.hypot(tracker.x[0] - true_drop_now["cx"], tracker.x[1] - true_drop_now["cy"]) * 100.0)
                        # 将确认的非移动突发障碍永久记入空间工作记忆（劫持后依然牢记绝不撞！）
                        if (not tracker.is_moving) and (not enable_nbv_peek) and tracker.n_obs >= 1:
                            sp_new = (float(tracker.x[0]), float(tracker.x[1]), float(tracker.r_est + 0.018))
                            if all(math.hypot(sp_new[0] - s0[0], sp_new[1] - s0[1]) > 0.10 for s0 in discovered_static_obs):
                                discovered_static_obs.append(sp_new)
                            else:
                                for idx_s, s0 in enumerate(discovered_static_obs):
                                    if math.hypot(sp_new[0] - s0[0], sp_new[1] - s0[1]) <= 0.10:
                                        discovered_static_obs[idx_s] = sp_new

                        # 触发重规划（pursuit 模式下被追的杯子不进 SDF——它是目标而非障碍）
                        if (not enable_nbv_peek) and (not pursuit_mode) and ((not was_init) or (tracker.is_moving and not prev_moving) or (tracker.n_obs in (1, 3, 5) and step - last_replan_step >= 2) or (attack2_det_step == step)):
                            pad_r = 0.025 if (enable_velocity_prediction and tracker.is_moving) else 0.016
                            spheres = list(discovered_static_obs)
                            spheres.extend(tracker.get_spheres(5, pad_r) if enable_velocity_prediction else [(float(tracker.x[0]), float(tracker.x[1]), float(tracker.r_est + 0.010))])
                            comp_d = CompositeDynamicHoverSdfNet(sdf_base, spheres)
                            cbf_d = plan_smooth_neural_cbf_trajectory(comp_d, (ex, ey), active_goal_xy, n_knots=20, safe_radius_m=0.092, n_opt_steps=110)
                            path_xy = [(float(p[0]), float(p[1])) for p in cbf_d["path_xy"]]
                            last_replan_step = step
                            cleared_moving = False
                            if (not was_init) or (attack2_det_step == step):
                                brake_this_step = True

            # 移动障碍离开视野后，恢复保留静止障碍记忆的 CBF 路径（L4）
            if (not enable_nbv_peek) and tracker.initialized and tracker.is_moving and (not cleared_moving) and (step - tracker.last_obs_step >= 5) and reloc_ok:
                sdf_c = CompositeDynamicHoverSdfNet(sdf_base, discovered_static_obs) if discovered_static_obs else sdf_base
                cbf_c = plan_smooth_neural_cbf_trajectory(sdf_c, (ex, ey), active_goal_xy, n_knots=20, safe_radius_m=0.096, n_opt_steps=95)
                path_xy = [(float(p[0]), float(p[1])) for p in cbf_c["path_xy"]]
                cleared_moving = True

            prev_b = cf["bgr"]
            cam_errs_mm.append(math.hypot(cam_ex - cx, cam_ey - cy) * 1000.0)
            clearances_m.append(arena.obstacle_clearance_m(tx, ty))

            # --- 5) 在线视觉打滑辨识 ---
            dx_v = math.hypot(ex - prev_ex, ey - prev_ey)
            if step > 1 and prev_cmd_vx > 0.06 and reloc_ok:
                est_slip = 0.78 * est_slip + 0.22 * float(np.clip(dx_v / (prev_cmd_vx * dt), 0.50, 1.15))
            slip_eff = max(est_slip, 0.55)

            # --- 5.6) 位姿卡死看门狗：指令位移持续远大于估计位移 → 局部重定位锁死在错误极小值，强制全局搜索 ---
            cmd_motion_acc += max(prev_cmd_vx, 0.0) * dt
            est_motion_acc += dx_v
            if cmd_motion_acc > 0.30:
                _stuck = est_motion_acc < 0.35 * cmd_motion_acc
                cmd_motion_acc = est_motion_acc = 0.0
                if _stuck and enable_global_recovery:
                    ro_g = relocalize_global_kidnap_zero_prior(reloc, cf["bgr"], p_imu, r_imu)
                    if ro_g["ok"]:
                        pose_stuck_events += 1
                        cam_ex, cam_ey = float(ro_g["x"]), float(ro_g["y"])
                        eyaw = _wrap_deg(float(ro_g["yaw_deg"]))
                        lever = REAL_H_CAM_M * math.sin(math.radians(p_est - p_nom))
                        ex = cam_ex - lever * math.cos(math.radians(eyaw))
                        ey = cam_ey - lever * math.sin(math.radians(eyaw))
                        reloc_ok = True
                        from_global = True
                        if kidnap_triggered and not kidnap_recovered:
                            kidnap_recovered = True
                            kidnap_recov_steps = step - int(kidnap_event["step"])
                            kidnap_recov_err_cm = round(math.hypot(ro_g["x"] - cx, ro_g["y"] - cy) * 100.0, 2)
                        mood = "RECOVERED"
                        mood_text = "Pose-stuck watchdog: global relock!"
                    else:
                        reloc_ok = False

            # --- 5.5) v18 追捕模式：移动水杯本身就是目标 ---
            if pursuit_mode:
                if tracker.initialized and tracker.n_obs >= 1:
                    gx = float(tracker.x[0]) + float(tracker.x[2]) * pursuit_lead_s
                    gy = float(tracker.x[1]) + float(tracker.x[3]) * pursuit_lead_s
                    ux, uy = gx - ex, gy - ey
                    un = math.hypot(ux, uy)
                    stand = float(tracker.r_est) + chase_standoff_m
                    if un > 1e-6:
                        gx -= ux / un * stand
                        gy -= uy / un * stand
                    # standoff 点若落在已发现障碍内，绕杯子逐步旋转选到净空一侧
                    if discovered_static_obs:
                        base_ang = math.atan2(gy - ey, gx - ex)
                        if un > 1e-6:
                            bx_, by_ = gx + ux / un * stand, gy + uy / un * stand
                        else:
                            bx_, by_ = gx, gy
                        for k_ in range(10):
                            off = 0.0 if k_ == 0 else (((k_ + 1) // 2) * 0.42 * (1.0 if k_ % 2 == 0 else -1.0))
                            cand = (bx_ + math.cos(base_ang + off) * stand, by_ + math.sin(base_ang + off) * stand)
                            if all(math.hypot(cand[0] - s0[0], cand[1] - s0[1]) > s0[2] + 0.055 for s0 in discovered_static_obs):
                                gx, gy = cand
                                break
                    cx_, cy_ = float(tracker.x[0]), float(tracker.x[1])
                    d_rc = math.hypot(ex - cx_, ey - cy_)
                    edge_now = d_rc - float(tracker.r_est) - 0.055
                    if edge_now < 0.14:
                        esc_r = float(tracker.r_est) + 0.055 + 0.060
                        cup_speed = math.hypot(float(tracker.x[2]), float(tracker.x[3]))
                        if cup_speed > 0.035:
                            # 骑杯尾：陪护点放在杯速反方向 ±52°，让出弹道——跟跑而不是挡路
                            v_ang = math.atan2(float(tracker.x[3]), float(tracker.x[2]))
                            behind = math.atan2(ey - cy_, ex - cx_)
                            side_s = 1.0 if _wrap_deg(math.degrees(behind - v_ang)) >= 0.0 else -1.0
                            ang_esc = v_ang + math.pi + side_s * 0.9
                            gx, gy = cx_ + math.cos(ang_esc) * esc_r, cy_ + math.sin(ang_esc) * esc_r
                        else:
                            ux2, uy2 = (ex - cx_) / max(d_rc, 1e-6), (ey - cy_) / max(d_rc, 1e-6)
                            gx, gy = cx_ + ux2 * esc_r, cy_ + uy2 * esc_r
                    else:
                        # 目标点钳在当前杯位气泡外——前瞻拦截点在杯子变向时可能甩进杯体
                        min_gc = float(tracker.r_est) + 0.085
                        d_gc = math.hypot(gx - cx_, gy - cy_)
                        if 1e-6 < d_gc < min_gc:
                            gx = cx_ + (gx - cx_) / d_gc * min_gc
                            gy = cy_ + (gy - cy_) / d_gc * min_gc
                    active_goal_xy = (gx, gy)
                    target_xy = (cx_, cy_)
                    if not captured and mood in ("APPROACH", "SWITCH_TARGET"):
                        mood = "PURSUIT"
                        mood_text = "Chasing the runaway cup!"
                elif vis_det is not None:
                    active_goal_xy = (vis_det["cx"], vis_det["cy"])
                if true_moving_now is not None:
                    edge = math.hypot(tx - true_moving_now["cx"], ty - true_moving_now["cy"]) - true_moving_now["rad"] - 0.055
                    min_chase_edge_m = min(min_chase_edge_m, edge)
                    if not captured and edge < 0.025:
                        captured = True
                        capture_step = step
                        mood = "ESCORT"
                        mood_text = "Caught it! Escorting the cup..."
                    elif captured:
                        escort_total += 1
                        if 0.0 < edge < 0.24:
                            escort_hits += 1
                # v24b 弹道让位：预测杯子近端弹道将横穿机器人（含转弯切入）时，
                # 目标点临时换到垂直杯速的让位点——贴身护航是"跟跑"，不是"挡路"
                if pursuit_mode and tracker.initialized and true_moving_now is not None:
                    yg = _ballistic_yield_goal(
                        float(tracker.x[0]), float(tracker.x[1]),
                        float(tracker.x[2]), float(tracker.x[3]),
                        ex, ey, float(tracker.r_est))
                    if yg is not None:
                        active_goal_xy = yg
                        if mood in ("ESCORT", "PURSUIT"):
                            mood_text = "Yielding to cup's path..."

                # tangent-bug 最小偏转：目标方向穿过障碍视锥时，把目标方向转到障碍切线沿
                for s_ in discovered_static_obs:
                    ox, oy = s_[0] - ex, s_[1] - ey
                    od = math.hypot(ox, oy)
                    gd = math.hypot(active_goal_xy[0] - ex, active_goal_xy[1] - ey)
                    if od > 1e-6 and gd > 1e-6 and od < gd + 0.05:
                        subt = math.asin(min(1.0, (s_[2] + 0.06) / od))
                        ang_g = math.atan2(active_goal_xy[1] - ey, active_goal_xy[0] - ex)
                        ang_o = math.atan2(oy, ox)
                        rel = _wrap_deg(math.degrees(ang_g - ang_o))
                        if abs(rel) < math.degrees(subt):
                            side = 1.0 if rel >= 0.0 else -1.0
                            ang_new = ang_o + math.radians(side * (math.degrees(subt) + 6.0))
                            active_goal_xy = (ex + math.cos(ang_new) * gd, ey + math.sin(ang_new) * gd)
                            if mood in ("PURSUIT", "PURSUIT_DODGE", "ESCORT", "APPROACH"):
                                mood = "PURSUIT_DODGE"
                                mood_text = "Tangent-deflect around drop..."
                            break

                # 已压线 → 沿径向逃逸出障碍球（检测中心漂移下用大余量触发；走廊级绕行交给 standoff 旋转）
                for s_ in discovered_static_obs:
                    if math.hypot(ex - s_[0], ey - s_[1]) < s_[2] - 0.02:
                        esc_ang = math.atan2(ey - s_[1], ex - s_[0])
                        active_goal_xy = (ex + math.cos(esc_ang) * 0.16, ey + math.sin(esc_ang) * 0.16)
                        if mood in ("PURSUIT", "PURSUIT_DODGE", "ESCORT", "APPROACH"):
                            mood = "PURSUIT_DODGE"
                            mood_text = "Escaping obstacle overlap..."
                        break

                # 停靠判据：杯子飞出桌面(消失) 或 tracker 近期有观测且判静止；
                # 冻结 tracker 的 is_moving 不衰减，必须同时要求观测新鲜，否则幻影失锁会假停靠
                _trk_fresh = tracker.initialized and (step - tracker.last_obs_step <= 8)
                if captured and (true_moving_now is None or (_trk_fresh and not tracker.is_moving)):
                    cup_rest_steps += 1
                else:
                    cup_rest_steps = 0
                pursuit_end = captured and cup_rest_steps >= 4
                # pursuit 里只统计非追逐物的碰撞（碰到被追的杯子=捕获成功）
                d_sta = min(
                    math.hypot(tx - arena.cup["cx"], ty - arena.cup["cy"]) - arena.cup["rad"],
                    math.hypot(tx - arena.mouse["cx"], ty - arena.mouse["cy"]) - 0.048,
                    max(abs(tx - arena.keyboard["cx"]) - 0.08, abs(ty - arena.keyboard["cy"]) - 0.16, 0.0),
                    max(abs(tx - arena.notebook["cx"]) - 0.08, abs(ty - arena.notebook["cy"]) - 0.07, 0.0),
                    max(abs(tx - arena.box_right["cx"]) - 0.06, abs(ty - arena.box_right["cy"]) - 0.06, 0.0),
                ) - 0.055
                for c_ in active_cyls:
                    if c_ is true_moving_now:
                        continue
                    d_sta = min(d_sta, math.hypot(tx - c_["cx"], ty - c_["cy"]) - c_["rad"] - 0.055)
                min_static_clear_m = min(min_static_clear_m, d_sta)

            # --- 6) 高阶活体行为状态机控制律 ---
            rem_goal = math.hypot(active_goal_xy[0] - ex, active_goal_xy[1] - ey)
            e_along = (active_goal_xy[0] - ex) * math.cos(math.radians(eyaw)) + (active_goal_xy[1] - ey) * math.sin(math.radians(eyaw))
            done_episode = False
            if pursuit_mode and pursuit_end:
                mood = "ESCORT_DONE"
                mood_text = "Cup parked. Mission complete!"
                raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=head_deg)
                done_episode = True

            # (A) L1 悬崖拒跳 + 摇头抗议 + 回头看人
            if cliff_detected:
                if cliff_emote_timer > 4:
                    # 3 拍摇头抗议 ("NOPE! NOT JUMPING!")
                    shake_seq = [-28.0, -28.0, 28.0, 28.0, -24.0, -24.0, 22.0, 22.0, -14.0, 0.0]
                    idx_s = min(len(shake_seq) - 1, 14 - cliff_emote_timer)
                    head_deg = shake_seq[idx_s]
                    raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=head_deg)
                    cliff_emote_timer -= 1
                elif cliff_emote_timer > 0:
                    # 转头凝视人类抗议
                    bear_h = _wrap_deg(math.degrees(math.atan2(human_face_init[1] - ey, human_face_init[0] - ex)) - eyaw)
                    head_deg = float(np.clip(bear_h, -32.0, 32.0))
                    raw_cmd = DriveCommand(vx=0.0, vyaw=float(np.clip((bear_h - head_deg) * 1.4, -40.0, 40.0)), head=head_deg)
                    mood = "STARE_HUMAN"
                    mood_text = "Refused cliff! Looking back at human"
                    cliff_emote_timer -= 1
                else:
                    raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=head_deg)
                    done_episode = True

            # (B) 跌下悬崖对照组提前结束
            elif fell_off_cliff:
                mood = "FELL_OFF"
                mood_text = "FELL OFF DESK CLIFF!"
                raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=0.0)
                done_episode = True

            # (C) L2 抵达后主动寻人状态机（看旧位置 -> 发现人不见了 -> 主动扫视 -> 锁定新座位！）
            elif docked and (human_face_moved is not None):
                post_dock_timer += 1
                if post_dock_phase == "LOOK_OLD_FACE":
                    bear_old = _wrap_deg(math.degrees(math.atan2(human_face_init[1] - ey, human_face_init[0] - ex)) - eyaw)
                    head_deg = float(np.clip(bear_old, -32.0, 32.0))
                    raw_cmd = DriveCommand(vx=0.0, vyaw=float(np.clip((bear_old - head_deg) * 1.2, -40.0, 40.0)), head=head_deg)
                    # 真实视线检查：人类已经换座位，旧方向视野内看不到人脸！
                    true_bear_now = _wrap_deg(math.degrees(math.atan2(true_face_xy[1] - ty, true_face_xy[0] - tx)) - (tyaw + head_deg))
                    if post_dock_timer >= 6 and abs(true_bear_now) > 22.0:
                        post_dock_phase = "SEARCH_HUMAN"
                        mood = "SEARCH_HUMAN"
                        mood_text = "Human moved?! Scanning room..."
                elif post_dock_phase == "SEARCH_HUMAN":
                    # 主动右扫找人（完全不知新座位坐标，直到真实相机视场 ±20° 扫到人类！）
                    raw_cmd = DriveCommand(vx=0.0, vyaw=-42.0, head=-18.0)
                    head_deg = -18.0
                    true_bear_now = _wrap_deg(math.degrees(math.atan2(true_face_xy[1] - ty, true_face_xy[0] - tx)) - (tyaw + head_deg))
                    if abs(true_bear_now) <= 18.0:
                        # 真实相机视野扫到人类！叠加 ±0.7° 测量噪声，精确锁眸！
                        obs_bear = true_bear_now + float(rng.normal(0.0, 0.7))
                        head_deg = float(np.clip(head_deg + obs_bear, -32.0, 32.0))
                        post_dock_phase = "FOUND_HUMAN"
                        human_found = True
                        post_dock_timer = 0
                        mood = "FOUND_HUMAN"
                        mood_text = "Found you in the new seat!"
                        final_face_gaze_err_deg = round(abs(_wrap_deg(math.degrees(math.atan2(true_face_xy[1] - ty, true_face_xy[0] - tx)) - (tyaw + head_deg))), 2)
                else:
                    # 锁眸凝视 5 帧后优雅结束
                    true_bear_now = _wrap_deg(math.degrees(math.atan2(true_face_xy[1] - ty, true_face_xy[0] - tx)) - (tyaw + head_deg))
                    final_face_gaze_err_deg = round(abs(true_bear_now), 2)
                    raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=head_deg)
                    if post_dock_timer >= 5:
                        done_episode = True

            # (D) 常规 / 探头 / 避障 / 劫持扫描运动控制
            else:
                # L3：检查是否抵达 NBV 侧向探头点 -> 一旦探头到位，确认背面通畅并切回隐藏主目标！
                if nbv_active and reloc_ok and rem_goal < 0.08:
                    nbv_active = False
                    nbv_peeks_done = True
                    active_goal_xy = target_xy
                    mood = "PEEK_CLEAR"
                    mood_text = "Peeked behind blocker -> Clear! Docking!"
                    comp_f = CompositeDynamicHoverSdfNet(sdf_base, [occ_est_sphere]) if occ_est_sphere is not None else sdf_base
                    cbf_f = plan_smooth_neural_cbf_trajectory(comp_f, (ex, ey), active_goal_xy, n_knots=18, safe_radius_m=0.092, n_opt_steps=105)
                    path_xy = [(float(p[0]), float(p[1])) for p in cbf_f["path_xy"]]
                    rem_goal = math.hypot(active_goal_xy[0] - ex, active_goal_xy[1] - ey)

                # 检查是否停靠到最终目标（pursuit 模式下由捕获/护送判定代替）
                if (not pursuit_mode) and (not nbv_active) and reloc_ok and (
                    (rem_goal < 0.022)
                    or (rem_goal < 0.040 and e_along < 0.008 and step > 14)
                ):
                    settle += 1
                    if settle >= 2:
                        docked = True
                        if human_face_moved is not None:
                            post_dock_phase = "LOOK_OLD_FACE"
                            post_dock_timer = 0
                            mood = "LOOK_FACE"
                            mood_text = "Docked! Looking back at human..."
                        else:
                            mood = "DOCKED"
                            mood_text = "Arrived safely at target!"
                            done_episode = True
                else:
                    settle = 0

                if brake_this_step:
                    raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=0.0)
                elif not reloc_ok:
                    if enable_global_recovery:
                        mood = "KIDNAP_SCAN"
                        mood_text = "Lost pose! Halting + Global SE(2) Scan..."
                        first_dir = 1.0 if eyaw < -62.0 else -1.0
                        re_dir = first_dir if ((lost_streak % 36) < 12) else -first_dir
                        raw_cmd = DriveCommand(vx=0.0, vyaw=re_dir * 48.0, head=0.0)
                    else:
                        bear_f = _wrap_deg(math.degrees(math.atan2(active_goal_xy[1] - ey, active_goal_xy[0] - ex)) - eyaw)
                        raw_cmd = DriveCommand(vx=35.0, vyaw=float(np.clip(bear_f * 0.9, -45.0, 45.0)), head=0.0)
                else:
                    lookahead_m = 0.085 if (tracker.initialized or nbv_active) else 0.11
                    wp = active_goal_xy
                    if rem_goal > 0.12 and not pursuit_mode:
                        in_i = int(np.argmin([math.hypot(px - ex, py - ey) for px, py in path_xy]))
                        for j in range(in_i, len(path_xy)):
                            if math.hypot(path_xy[j][0] - ex, path_xy[j][1] - ey) >= lookahead_m:
                                wp = path_xy[j]
                                break
                    bear_wp = _wrap_deg(math.degrees(math.atan2(wp[1] - ey, wp[0] - ex)) - eyaw)
                    feat = EndSideFeatureBuilder.build(RobotTelemetry(pitch=p_est, roll=r_imu, yaw=eyaw), [], TargetLock(valid=1, bearing_body=bear_wp, remaining_m=rem_goal), 0.85)
                    cmd_v3 = policy_v3.infer_step(feat, dt=dt, dx_vis_m=dx_v, dyaw_vis_deg=_wrap_deg(eyaw - prev_eyaw))
                    vyaw_lim = float(np.clip(cmd_v3.vyaw, -46.0, 46.0))
                    vx_cap = 56.0
                    head_deg = float(np.clip(cmd_v3.head, -28.0, 28.0))
                    # 追捕视线保持：相机始终锁住被追的杯子（失锁 >15 步则扫视寻回），否则杯子出 FOV 即永久失锁
                    if pursuit_mode and tracker.initialized:
                        if step - tracker.last_obs_step > 15:
                            head_deg = 26.0 * (1.0 if ((step // 5) % 2) == 0 else -1.0)
                        else:
                            bear_cup = math.degrees(math.atan2(tracker.x[1] - ey, tracker.x[0] - ex))
                            head_deg = float(np.clip(_wrap_deg(bear_cup - eyaw), -28.0, 28.0))

                    # 末端静态安全护盾（pursuit 下全距生效；检测球仅用于目标层规避，瞬时制动仍按基准 SDF，
                    # 因为检测中心可有 ~10cm 漂移，幻影球会让护盾误刹成死锁）
                    if (not nbv_active) and (pursuit_mode or rem_goal < 0.12):
                        sdf_sh = sdf_base
                        with torch.no_grad():
                            dev_s = next(sdf_sh.parameters()).device if hasattr(sdf_sh, "parameters") else next(sdf_base.parameters()).device
                            p_c = torch.tensor([[ex, ey]], dtype=torch.float32, device=dev_s)
                            p_f = torch.tensor([[ex + 0.05 * math.cos(math.radians(eyaw)), ey + 0.05 * math.sin(math.radians(eyaw))]], dtype=torch.float32, device=dev_s)
                            if float(sdf_sh(p_f)[0].item()) < 0.072 and float(sdf_sh(p_f)[0].item()) < float(sdf_sh(p_c)[0].item()):
                                vx_cap = 0.0

                    # 预测性速度障碍管护盾（L4 移动水杯；pursuit 下被追的是目标，不禁速）
                    if (not pursuit_mode) and tracker.initialized and tracker.is_moving and (step - tracker.last_obs_step <= 4):
                        if enable_velocity_prediction:
                            tube = tracker.get_spheres(5, 0.025)
                            fwd_x = ex + 0.12 * math.cos(math.radians(eyaw))
                            fwd_y = ey + 0.12 * math.sin(math.radians(eyaw))
                            min_d_tube = min(math.hypot(fwd_x - ox, fwd_y - oy) - orad - 0.055 for ox, oy, orad in tube)
                            if min_d_tube < 0.050:
                                vx_cap = 0.0
                        else:
                            d0 = math.hypot(ex - tracker.x[0], ey - tracker.x[1]) - tracker.r_est - 0.055
                            if d0 < 0.035:
                                vx_cap = min(vx_cap, 18.0)

                    # 追捕目标失联：持续同向自旋寻回杯子（±交替只是原地摆动，永远转不到 FOV 外的杯子）。
                    # 按 last_move_step（观测累计位移>3cm 才算）判失联，抗幻影检测对 last_obs 的刷新；
                    # 阈值 14 > cup_rest 的 4 步窗口，保证"已捕获+真停靠"先走 ESCORT_DONE
                    if pursuit_mode and tracker.initialized and (step - tracker.last_move_step > 14):
                        raw_cmd = DriveCommand(vx=0.0, vyaw=46.0, head=head_deg)
                        mood = "PURSUIT_SCAN"
                        mood_text = "Lost the cup! Spin-scanning..."
                    elif rem_goal < 0.08 and e_along < 0.0:
                        raw_cmd = DriveCommand(vx=float(np.clip((e_along / (0.34 * dt * slip_eff)) * 85.0, -18.0, 0.0)), vyaw=0.0, head=head_deg)
                    elif rem_goal < 0.16:
                        max_vx = min(vx_cap, (max(e_along, 0.0) / (0.34 * dt * slip_eff)) * 88.0)
                        raw_cmd = DriveCommand(vx=float(np.clip(cmd_v3.vx, 0.0, max_vx)), vyaw=vyaw_lim, head=head_deg)
                    else:
                        raw_cmd = DriveCommand(vx=float(np.clip(cmd_v3.vx, 0.0, vx_cap)), vyaw=vyaw_lim, head=head_deg)

            if record_frames:
                vis_records.append({
                    "step": step,
                    "bgr": cf["bgr"].copy(),
                    "depth": depth_live.copy(),
                    "tx": round(tx, 4), "ty": round(ty, 4), "tyaw": round(tyaw, 2),
                    "ex": round(ex, 4), "ey": round(ey, 4), "eyaw": round(eyaw, 2),
                    "head_deg": round(head_deg, 1),
                    "reloc_ok": reloc_ok,
                    "mood": mood, "mood_text": mood_text,
                    "target_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
                    "active_goal_xy": [round(active_goal_xy[0], 3), round(active_goal_xy[1], 3)],
                    "path_xy": [(round(p[0], 3), round(p[1], 3)) for p in path_xy],
                    "cyls": [dict(c) for c in active_cyls],
                    "vis_det": dict(vis_det) if vis_det is not None else None,
                    "cliff_det": dict(cliff_det) if cliff_det is not None else None,
                    "cliff_y_est": cliff_y_est,
                    "kf_tube": tracker.get_spheres(5, 0.024) if (tracker.initialized and (step - tracker.last_obs_step <= 4)) else [],
                    "nbv_peek": list(nbv_peek_xy) if nbv_peek_xy is not None else None,
                    "true_face_xy": [round(true_face_xy[0], 3), round(true_face_xy[1], 3)],
                    "clear_cm": round(clearances_m[-1] * 100.0, 2),
                })

            if done_episode:
                break

            safe = shield.apply(raw_cmd, mode=MODE_APPROACH, remaining_m=max(rem_goal, 0.02), pitch_deg=p_est, clearance_m=0.85)
            if raw_cmd.vx < 0.0 and rem_goal < 0.08:
                safe = DriveCommand(vx=raw_cmd.vx, vyaw=safe.vyaw, head=safe.head)
            if cliff_detected or docked:
                safe = DriveCommand(vx=0.0, vyaw=raw_cmd.vyaw, head=raw_cmd.head)
            prev_ex, prev_ey, prev_eyaw = ex, ey, eyaw

            vx_mps = float(np.clip(safe.vx / 100.0 * 0.34, -0.08, 0.22))
            prev_cmd_vx = vx_mps
            vyaw_dps = float(np.clip(safe.vyaw / 100.0 * 75.0, -60.0, 60.0))
            if os.environ.get("RIG_PURSUIT_DEBUG") and pursuit_mode and step >= 10:
                _tk = f"({tracker.x[0]:.2f},{tracker.x[1]:.2f},v={math.hypot(tracker.x[2], tracker.x[3]):.2f})" if tracker.initialized else "noinit"
                _tc = f"({true_moving_now['cx']:.2f},{true_moving_now['cy']:.2f})" if true_moving_now is not None else "none"
                print(f"  [tk s{step}] tracker={_tk} truecup={_tc} est=({ex:.2f},{ey:.2f}) true=({tx:.2f},{ty:.2f})")
            if os.environ.get("RIG_PURSUIT_DEBUG2") and pursuit_mode and step >= 10:
                _net = f"({cmd_v3.vx:.1f},{cmd_v3.vyaw:.1f})" if "cmd_v3" in dir() else "n/a"
                _cap = f"{vx_cap:.0f}" if "vx_cap" in dir() else "n/a"
                print(f"  [dbg s{step}] true=({tx:.2f},{ty:.2f}) est=({ex:.2f},{ey:.2f}) goal=({active_goal_xy[0]:.2f},{active_goal_xy[1]:.2f}) "
                      f"rem={rem_goal:.3f} eA={e_along:.3f} raw=({raw_cmd.vx:.1f},{raw_cmd.vyaw:.1f}) safe_vx={safe.vx:.1f} "
                      f"net={_net} cap={_cap} reloc={int(reloc_ok)} "
                      f"obs={[(round(o[0],2),round(o[1],2),round(o[2],2)) for o in discovered_static_obs]} mood={mood}")
            tyaw = _wrap_deg(tyaw + (vyaw_dps + yaw_drift_dps) * dt)
            tx += (vx_mps * slip_factor) * dt * math.cos(math.radians(tyaw))
            ty += (vx_mps * slip_factor) * dt * math.sin(math.radians(tyaw))
            traj_true.append((tx, ty, tyaw))
    finally:
        arena.extra_cylinders = []

    fin_err_cm = round(math.hypot(tx - target_xy[0], ty - target_xy[1]) * 100.0, 2)
    min_cl_cm = round(float(min(clearances_m)) * 100.0, 2)
    return {
        "steps": len(traj_true) - 1,
        "final_err_cm": fin_err_cm,
        "mean_cam_pose_err_cm": round(float(np.mean(cam_errs_mm)) / 10.0, 2),
        "min_clearance_cm": min_cl_cm,
        "collision": bool(min_cl_cm < 0.0),
        "cliff_detected": cliff_detected,
        "cliff_refusal_step": cliff_refusal_step,
        "cliff_y_est_m": cliff_y_est,
        "fell_off_cliff": fell_off_cliff,
        "min_cliff_margin_cm": round(min_cliff_margin_cm, 2) if arena.cliff_y_min is not None else None,
        "fakeout_ignored": fakeout_ignored,
        "target_switched": target_switched,
        "human_found_new_seat": human_found,
        "final_face_gaze_err_deg": final_face_gaze_err_deg,
        "nbv_peeks_done": nbv_peeks_done,
        "nbv_peek_xy": list(nbv_peek_xy) if nbv_peek_xy is not None else None,
        "attack1_moving_det_step": attack1_det_step,
        "attack2_drop_det_step": attack2_det_step,
        "attack3_kidnap_recovered": kidnap_recovered,
        "attack3_kidnap_recov_steps": kidnap_recov_steps,
        "attack3_kidnap_recov_err_cm": kidnap_recov_err_cm,
        "pursuit_mode": pursuit_mode,
        "pursuit_captured": captured,
        "capture_step": capture_step,
        "escort_band_ratio": round(escort_hits / max(escort_total, 1), 3),
        "min_chase_edge_cm": round(min_chase_edge_m * 100.0, 2) if min_chase_edge_m < 999.0 else None,
        "min_static_clear_cm": round(min_static_clear_m * 100.0, 2) if min_static_clear_m < 999.0 else None,
        "mean_moving_vel_err_mps": round(float(np.mean(vel_errs_mps)), 3) if vel_errs_mps else -1.0,
        "mean_drop_loc_err_cm": round(float(np.mean(drop_loc_errs_cm)), 2) if drop_loc_errs_cm else -1.0,
        "traj_true": [(round(a, 4), round(b, 4), round(c, 2)) for a, b, c in traj_true],
        "vis_records": vis_records,
    }


# ---------------------------------------------------------------------------
# 4. 电影级无冗余静帧三联演示视频 + 4 合 1 科研海报生成器
# ---------------------------------------------------------------------------


def _trim_idle_tail(recs: list[dict[str, Any]], max_idle: int = 8) -> list[dict[str, Any]]:
    """自动剪掉最后长时间原地静止的冗余帧，让演示视频全程紧凑刺激！"""
    if len(recs) <= max_idle + 5:
        return recs
    last_active = len(recs) - 1
    for i in range(len(recs) - 1, 1, -1):
        r1, r0 = recs[i], recs[i - 1]
        moved = math.hypot(r1["tx"] - r0["tx"], r1["ty"] - r0["ty"]) > 0.003 or abs(r1["tyaw"] - r0["tyaw"]) > 0.8 or abs(r1["head_deg"] - r0["head_deg"]) > 1.5 or (r1["mood"] != r0["mood"])
        if moved:
            last_active = i
            break
    return recs[: min(len(recs), last_active + max_idle)]


def _draw_wenwen_face_badge(canvas: np.ndarray, x0: int, y0: int, mood: str, head_deg: float) -> None:
    """在 HUD 右下角画一个超萌的文文实时情绪表情小面板（随状态变化眼睛与舵机转角！）。"""
    cv2.rectangle(canvas, (x0, y0), (x0 + 190, y0 + 110), (24, 34, 46), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + 190, y0 + 110), (72, 130, 155), 1)
    cx, cy = x0 + 95 + int(np.clip(head_deg * 0.7, -22, 22)), y0 + 50
    col = (110, 248, 180)
    if mood in ("REFUSE_CLIFF", "FELL_OFF"):
        col = (90, 105, 255)
        # >< 拒绝惊恐眼
        cv2.line(canvas, (cx - 32, cy - 12), (cx - 16, cy), col, 3)
        cv2.line(canvas, (cx - 32, cy + 12), (cx - 16, cy), col, 3)
        cv2.line(canvas, (cx + 32, cy - 12), (cx + 16, cy), col, 3)
        cv2.line(canvas, (cx + 32, cy + 12), (cx + 16, cy), col, 3)
    elif mood in ("KIDNAP_SCAN", "SEARCH_HUMAN"):
        col = (90, 215, 255)
        # ○ ○ 警觉搜寻大眼
        cv2.circle(canvas, (cx - 24, cy), 13, col, 2)
        cv2.circle(canvas, (cx + 24, cy), 13, col, 2)
        cv2.circle(canvas, (cx - 24, cy), 4, col, -1)
        cv2.circle(canvas, (cx + 24, cy), 4, col, -1)
    elif mood in ("PEEK_NBV", "DODGE_CUP", "DODGE_DROP"):
        col = (95, 230, 255)
        # 半眯专注计算眼
        cv2.ellipse(canvas, (cx - 24, cy), (14, 9), 0, 0, 360, col, -1)
        cv2.ellipse(canvas, (cx + 24, cy), (14, 9), 0, 0, 360, col, -1)
    elif mood in ("FOUND_HUMAN", "DOCKED"):
        col = (120, 255, 165)
        # ^ ^ 开心月牙眼
        cv2.ellipse(canvas, (cx - 24, cy + 4), (14, 12), 0, 195, 345, col, 3)
        cv2.ellipse(canvas, (cx + 24, cy + 4), (14, 12), 0, 195, 345, col, 3)
    else:
        cv2.circle(canvas, (cx - 24, cy), 11, col, -1)
        cv2.circle(canvas, (cx + 24, cy), 11, col, -1)
    cv2.putText(canvas, f"MOOD: {mood[:14]}", (x0 + 10, y0 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)


def render_v15_cinema_video(clips: list[tuple[str, list[dict], float | None]], out_mp4: pathlib.Path) -> pathlib.Path:
    W, H = 1080, 520
    raw_mp4 = out_mp4.with_name(out_mp4.stem + "_raw.mp4")
    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (W, H))

    def _w2px(x: float, y: float) -> tuple[int, int]:
        u = int(round(620 + (y - (-0.48)) / 0.80 * 435))
        v = int(round(470 - (x - (-0.04)) / 1.18 * 410))
        return u, v

    for title, raw_recs, cliff_y in clips:
        recs = _trim_idle_tail(raw_recs, max_idle=9)
        trail = []
        for r in recs:
            canvas = np.full((H, W, 3), (14, 19, 25), dtype=np.uint8)
            cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (112, 248, 182), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Inner Thought: {r['mood_text']}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 224, 118), 1, cv2.LINE_AA)

            # 左上：240x240 第一视角实拍/仿真相机 (放大至 360x360)
            cam_img = cv2.resize(r["bgr"], (360, 360), interpolation=cv2.INTER_NEAREST)
            s = 360.0 / 240.0
            if r["vis_det"] is not None:
                bx1, by1, bx2, by2 = r["vis_det"]["bbox_uv"]
                cv2.rectangle(cam_img, (int(bx1 * s), int(by1 * s)), (int(bx2 * s), int(by2 * s)), (240, 205, 80), 2)
                cv2.putText(cam_img, "CAM 3D OBJ!", (int(bx1 * s), max(16, int(by1 * s) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 205, 80), 2)
            if r["cliff_det"] is not None:
                bx1, by1, bx2, by2 = r["cliff_det"]["bbox_uv"]
                cv2.rectangle(cam_img, (int(bx1 * s), int(by1 * s)), (int(bx2 * s), int(by2 * s)), (80, 95, 255), 2)
                cv2.putText(cam_img, "CLIFF VOID!", (int(bx1 * s), max(18, int(by1 * s) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 95, 255), 2)
            canvas[66:426, 18:378] = cam_img
            cv2.rectangle(canvas, (18, 66), (378, 426), (62, 95, 118), 1)
            cv2.putText(canvas, f"240x240 Cam | Step {r['step']:02d} | Head={r['head_deg']:+.0f}deg | Clear={r['clear_cm']:+.1f}cm", (18, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 212, 228), 1, cv2.LINE_AA)

            # 中列：HoverGeoNet-v4 实时 3D 深度热力小图 (190x190) + 文文实时表情 HUD (190x110)
            d_norm = np.clip((1.15 - r["depth"]) / 0.95, 0.0, 1.0)
            d_col = cv2.applyColorMap((d_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
            d_mini = cv2.resize(d_col, (190, 190), interpolation=cv2.INTER_AREA)
            canvas[66:256, 396:586] = d_mini
            cv2.rectangle(canvas, (396, 66), (586, 256), (62, 95, 118), 1)
            cv2.putText(canvas, "HoverGeoNet-v4 Depth", (400, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 220, 255), 1, cv2.LINE_AA)
            _draw_wenwen_face_badge(canvas, 396, 314, r["mood"], r["head_deg"])

            # 右侧：3D 桌面全局俯视图
            cv2.rectangle(canvas, (606, 56), (1064, 488), (21, 29, 39), -1)
            cv2.rectangle(canvas, (606, 56), (1064, 488), (55, 82, 102), 1)

            # 若有物理悬崖，画出真实崖边红虚线 + 暗色深渊区
            if cliff_y is not None:
                cu, _ = _w2px(0.0, cliff_y)
                cv2.rectangle(canvas, (607, 57), (max(608, cu), 487), (15, 12, 22), -1)
                cv2.line(canvas, (cu, 57), (cu, 487), (80, 90, 255), 2)
                cv2.putText(canvas, "CLIFF", (612, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 100, 255), 2)
            if r["cliff_y_est"] is not None:
                eu, _ = _w2px(0.0, r["cliff_y_est"])
                cv2.line(canvas, (eu, 57), (eu, 487), (100, 240, 255), 1)

            # 静态水杯与鼠标
            cv2.circle(canvas, _w2px(0.36, -0.02), max(10, int(0.040 / 0.80 * 435)), (124, 84, 38), -1)
            cv2.circle(canvas, _w2px(0.36, -0.02), max(10, int(0.040 / 0.80 * 435)), (255, 212, 103), 2)
            cv2.circle(canvas, _w2px(0.66, -0.13), max(10, int(0.045 / 0.80 * 435)), (60, 92, 36), -1)

            # 人类座位圆点
            fu, fv = _w2px(r["true_face_xy"][0], r["true_face_xy"][1])
            cv2.circle(canvas, (fu, fv), 10, (190, 120, 255), -1)
            cv2.putText(canvas, "HUMAN", (fu - 22, fv - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 180, 255), 1)

            # 规划路径
            p_pts = r["path_xy"]
            for i in range(len(p_pts) - 1):
                cv2.line(canvas, _w2px(p_pts[i][0], p_pts[i][1]), _w2px(p_pts[i + 1][0], p_pts[i + 1][1]), (95, 210, 255), 1)

            # NBV 侧向探头点 + 视线
            if r["nbv_peek"] is not None:
                pu, pv = _w2px(r["nbv_peek"][0], r["nbv_peek"][1])
                cv2.drawMarker(canvas, (pu, pv), (255, 215, 80), cv2.MARKER_DIAMOND, 14, 2)
                tu0, tv0 = _w2px(r["target_xy"][0], r["target_xy"][1])
                cv2.line(canvas, (pu, pv), (tu0, tv0), (140, 230, 180), 1)

            # 预测速度障碍管
            for ox, oy, orad in r["kf_tube"]:
                cv2.circle(canvas, _w2px(ox, oy), max(6, int(orad / 0.80 * 435)), (240, 205, 80), 1)

            # 真实动态/遮挡圆柱
            for cyl in r["cyls"]:
                cv2.circle(canvas, _w2px(cyl["cx"], cyl["cy"]), max(8, int(cyl["rad"] / 0.80 * 435)), (40, 40, 225), -1)
                cv2.circle(canvas, _w2px(cyl["cx"], cyl["cy"]), max(8, int(cyl["rad"] / 0.80 * 435)), (215, 215, 255), 2)

            # 目标星标
            tu, tv = _w2px(r["target_xy"][0], r["target_xy"][1])
            cv2.drawMarker(canvas, (tu, tv), (118, 241, 255), cv2.MARKER_STAR, 18, 2)

            # 机器人轨迹、机体朝向、舵机视线锥
            trail.append(_w2px(r["tx"], r["ty"]))
            for i in range(len(trail) - 1):
                cv2.line(canvas, trail[i], trail[i + 1], (108, 247, 176), 2)
            ru, rv = trail[-1]
            cv2.circle(canvas, (ru, rv), 9, (108, 247, 176), -1)
            gaze_rad = math.radians(r["tyaw"] + r["head_deg"])
            hu, hv = _w2px(r["tx"] + 0.14 * math.cos(gaze_rad), r["ty"] + 0.14 * math.sin(gaze_rad))
            cv2.arrowedLine(canvas, (ru, rv), (hu, hv), (255, 240, 140), 2, tipLength=0.32)

            vw.write(canvas)
    vw.release()

    if FFMPEG_EXE.exists():
        subprocess.run(
            [str(FFMPEG_EXE), "-y", "-i", str(raw_mp4), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", str(out_mp4)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        raw_mp4.unlink(missing_ok=True)
    else:
        raw_mp4.replace(out_mp4)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_mp4, ARTIFACT_DIR / out_mp4.name)
    return out_mp4


def plot_v15_poster(rep: dict[str, Any], out_png: pathlib.Path) -> pathlib.Path:
    plt.rcParams.update({
        "figure.facecolor": "#0b1116", "axes.facecolor": "#111a22", "axes.edgecolor": "#30485c",
        "axes.labelcolor": "#dce8f2", "xtick.color": "#9db4c8", "ytick.color": "#9db4c8",
        "text.color": "#eaf3fa", "grid.color": "#233544", "grid.alpha": 0.45, "font.size": 9.0,
    })
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    fig.suptitle(
        "RIG-HOVER v15 LIVING DESKTOP MIND: CLIFF INSTINCT · TEASING & FACE SEARCH · OBJECT PERMANENCE · 3x CHAOS GAUNTLET",
        fontsize=12.6, fontweight="bold", color="#6cf7b0", y=0.975,
    )

    def _bg(ax, xlim=(-0.49, 0.30), ylim=(-0.04, 1.15)) -> None:
        ax.add_patch(mpatches.Circle((-0.02, 0.36), 0.040, facecolor="#26547c", edgecolor="#67d4ff", linewidth=1.6))
        ax.add_patch(mpatches.Ellipse((-0.13, 0.66), 0.072, 0.116, facecolor="#20523c", edgecolor="#6cf7b0", linewidth=1.4))
        ax.add_patch(mpatches.Rectangle((0.01, 0.705), 0.30, 0.15, facecolor="#2a3542", edgecolor="#7f95ab", linewidth=1.0))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("Desk Y (m, +Left / -Right)"); ax.set_ylabel("Desk X (m, +Forward)")
        ax.grid(True)

    # Panel 1: L1 Visual Cliff Survival
    ax1 = axes[0, 0]
    _bg(ax1, ylim=(-0.04, 0.88))
    ax1.axvspan(-0.49, -0.36, color="#3a1019", alpha=0.55, label="72cm Cliff Abyss (Y < -0.36m)")
    ax1.axvline(-0.36, color="#ff5c77", linewidth=2.2, linestyle="--", label="True Desk Cliff Edge")
    l1 = rep["L1_visual_cliff_survival"]
    if l1["v15_living_mind"]["cliff_y_est_m"] is not None:
        ax1.axvline(l1["v15_living_mind"]["cliff_y_est_m"], color="#67d4ff", linewidth=1.5, linestyle=":", label=f"Visual Est Rim ({l1['v15_living_mind']['cliff_y_est_m']:.3f}m)")
    tb = np.array(l1["baseline_no_cliff"]["traj_true"])
    tv = np.array(l1["v15_living_mind"]["traj_true"])
    ax1.plot(tb[:, 1], tb[:, 0], color="#f26b6b", linewidth=2.2, label=f"No Cliff Sense (FELL={l1['baseline_no_cliff']['fell_off_cliff']})")
    ax1.plot(tv[:, 1], tv[:, 0], color="#39e59c", linewidth=2.8, label=f"v15 Cliff-CBF + Head-Shake (Margin=+{l1['v15_living_mind']['min_cliff_margin_cm']}cm)")
    ax1.scatter([-0.44], [0.46], s=140, marker="X", color="#ff6b6b", zorder=10, label="Suicidal Knock Past Cliff")
    ax1.set_title(f"L1: Zero-Oracle Visual Cliff Instinct (Fell={l1['v15_living_mind']['fell_off_cliff']}, Margin=+{l1['v15_living_mind']['min_cliff_margin_cm']}cm)", fontweight="bold", color="#6cf7b0")
    ax1.legend(fontsize=7.3, loc="upper right")

    # Panel 2: L2 Teasing Immunity + Switch + Active Face Search
    ax2 = axes[0, 1]
    _bg(ax2, ylim=(-0.04, 1.18))
    l2 = rep["L2_teasing_and_active_face_search"]
    t2 = np.array(l2["traj_true"])
    ax2.plot(t2[:, 1], t2[:, 0], color="#39e59c", linewidth=2.8, label=f"v15 Trajectory (Dock={l2['final_err_cm']}cm, Coll={l2['collision']})")
    ax2.scatter([-0.16], [0.48], s=90, marker="o", color="#f2b84b", label="Initial Knock A (Switched)")
    ax2.scatter([-0.28], [0.56], s=150, marker="*", color="#fff176", zorder=10, label="Final Knock B (Mid-Course Switch!)")
    ax2.scatter([0.08], [1.10], s=100, marker="^", color="#999999", label="Old Human Seat (Empty)")
    ax2.scatter([-0.34], [1.05], s=130, marker="^", color="#d68cff", zorder=10, label=f"New Seat Found! (Gaze={l2['final_face_gaze_err_deg']}°)")
    ax2.annotate("", xy=(-0.34, 1.05), xytext=(0.08, 1.10), arrowprops=dict(arrowstyle="->", color="#d68cff", lw=2.0, ls="--"), clip_on=True)
    ax2.annotate("", xy=(-0.34, 1.05), xytext=(float(np.clip(t2[-1, 1], -0.45, 0.28)), float(np.clip(t2[-1, 0], 0.0, 1.1))), arrowprops=dict(arrowstyle="->", color="#fff176", lw=1.8, ls=":"), clip_on=True)
    ax2.set_title(f"L2: Fakeout Ignored + Mid-Course Switch + Active Face Search ({l2['final_face_gaze_err_deg']}°)", fontweight="bold", color="#6cf7b0")
    ax2.legend(fontsize=7.3, loc="lower left")

    # Panel 3: L3 Piaget Stage-4 Object Permanence & NBV Occluded Peek
    ax3 = axes[1, 0]
    _bg(ax3, ylim=(-0.04, 0.88))
    l3 = rep["L3_object_permanence_nbv_peek"]
    occ = l3["occluder_cfg"]
    ax3.add_patch(mpatches.Circle((occ["cy"], occ["cx"]), occ["rad"], facecolor="#c1121f", edgecolor="#ffccd5", linewidth=2.0, label="Tall Occluder (17.5cm > Eye)"))
    t3b = np.array(l3["baseline_no_nbv"]["traj_true"])
    t3v = np.array(l3["v15_nbv_peek"]["traj_true"])
    ax3.plot(t3b[:, 1], t3b[:, 0], color="#f2b84b", linewidth=2.0, linestyle="--", label=f"Direct Greedy (Arr={l3['baseline_no_nbv']['final_err_cm']}cm, Coll={l3['baseline_no_nbv']['collision']})")
    ax3.plot(t3v[:, 1], t3v[:, 0], color="#39e59c", linewidth=2.8, label=f"v15 NBV Peek (Arr={l3['v15_nbv_peek']['final_err_cm']}cm, Clear=+{l3['v15_nbv_peek']['min_clearance_cm']}cm)")
    if l3["v15_nbv_peek"]["nbv_peek_xy"] is not None:
        px, py = l3["v15_nbv_peek"]["nbv_peek_xy"]
        ax3.scatter([py], [px], s=110, marker="D", color="#ffd166", zorder=10, label="Optimal NBV Peek Viewpoint")
        ax3.plot([py, -0.25], [px, 0.58], ":", color="#6cf7b0", linewidth=1.8, label="Unoccluded Sightline After Peek")
    ax3.scatter([-0.25], [0.58], s=150, marker="*", color="#fff176", zorder=10, label="Hidden Knock Behind Blocker")
    ax3.set_title(f"L3: Piaget Object Permanence & NBV Peek (Peek={l3['v15_nbv_peek']['nbv_peeks_done']}, Coll={l3['v15_nbv_peek']['collision']})", fontweight="bold", color="#6cf7b0")
    ax3.legend(fontsize=7.2, loc="upper right")

    # Panel 4: L4 Ultimate 3-in-1 Chaos Gauntlet
    ax4 = axes[1, 1]
    _bg(ax4, ylim=(-0.04, 0.88))
    l4 = rep["L4_triple_chaos_gauntlet"]
    mc = l4["attack1_moving_cup"]
    dc = l4["attack2_sudden_drop"]
    ax4.annotate("", xy=(mc["y0"] + mc["vy"] * 2.2, mc["x0"] + mc["vx"] * 2.2), xytext=(mc["y0"], mc["x0"]), arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=2.2, ls="--"))
    ax4.add_patch(mpatches.Circle((mc["y0"], mc["x0"]), mc["rad"], facecolor="#c1121f", alpha=0.55, edgecolor="#ffccd5", label="Attack #1: Moving Cup (0.105m/s)"))
    ax4.add_patch(mpatches.Circle((dc["cy"], dc["cx"]), dc["rad"], facecolor="#e07a5f", edgecolor="#fff", linewidth=1.8, label="Attack #2: Sudden Dropped Cylinder"))
    t4b = np.array(l4["baseline_no_v15"]["traj_true"])
    t4v = np.array(l4["v15_living_mind"]["traj_true"])
    ax4.plot(t4b[:, 1], t4b[:, 0], color="#f26b6b", linewidth=2.0, label=f"Baseline (Arr={l4['baseline_no_v15']['final_err_cm']}cm, Coll={l4['baseline_no_v15']['collision']})")
    ax4.plot(t4v[:, 1], t4v[:, 0], color="#39e59c", linewidth=2.8, label=f"v15 Triple-Survivor (Arr={l4['v15_living_mind']['final_err_cm']}cm, Coll={l4['v15_living_mind']['collision']})")
    ax4.scatter([-0.25], [0.62], s=150, marker="*", color="#fff176", zorder=10, label="Target (+Attack #3: +88° Kidnap!)")
    ax4.set_title(f"L4: 3x Chaos Gauntlet (Moving+Drop+Kidnap -> Reloc {l4['v15_living_mind']['attack3_kidnap_recov_steps']} steps, Coll={l4['v15_living_mind']['collision']})", fontweight="bold", color="#6cf7b0")
    ax4.legend(fontsize=7.1, loc="upper right")

    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(str(out_png), dpi=135)
    plt.close(fig)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_png, ARTIFACT_DIR / out_png.name)
    return out_png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261122)
    np.random.seed(20261122)
    print("=== HOVER LIVING MIND v15: 4 EXTREME ZERO-ORACLE LIVING DESKTOP PET BEHAVIORS ===", flush=True)

    geo_fe = HoverGeoNet240Frontend(V4_CKPT, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
    policy_v3 = load_fine_motion_policy_v3("cpu")

    # 标准无悬崖桌面 + 悬崖桌面共享相同桌面初始路标与静态 SDF
    arena_std = CliffDesktopArena3D(h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG, cliff_y_min=None)
    map_pack = build_initial_map_and_sdf(arena_std, geo_fe, seed=20, build_sdf=True)

    # --- L1: Visual Cliff Survival & Edge-Refusal Head-Shake ---
    print("\n--- L1: Visual Cliff Survival & Edge-Refusal Head-Shake (Suicidal Knock Past Y=-0.36m!) ---", flush=True)
    arena_cliff = CliffDesktopArena3D(h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG, cliff_y_min=-0.36)
    l1_base = run_living_brain_episode(arena_cliff, map_pack, geo_fe, policy_v3, (0.46, -0.44), enable_cliff_instinct=False, record_frames=False, seed=15101)
    l1_v15 = run_living_brain_episode(arena_cliff, map_pack, geo_fe, policy_v3, (0.46, -0.44), enable_cliff_instinct=True, record_frames=True, seed=15101)
    print(f"  [No Cliff Sense] fell={l1_base['fell_off_cliff']} | min_margin={l1_base['min_cliff_margin_cm']}cm", flush=True)
    print(f"  [v15 Cliff-CBF]  detected={l1_v15['cliff_detected']}@step{l1_v15['cliff_refusal_step']} (est_y={l1_v15['cliff_y_est_m']} vs -0.36) | fell={l1_v15['fell_off_cliff']} | margin=+{l1_v15['min_cliff_margin_cm']}cm", flush=True)

    # --- L2: Teasing Immunity + Mid-Course Switch + Active Face Search ---
    print("\n--- L2: Fake-Out Teasing Immunity + Mid-Course Target Switch + Active Face Relocation Search ---", flush=True)
    l2_v15 = run_living_brain_episode(
        arena_std, map_pack, geo_fe, policy_v3, (0.48, -0.16),
        enable_cliff_instinct=False,
        fakeout_event={"step": 7, "xy": (0.46, 0.14)},
        switch_target_event={"step": 10, "xy": (0.56, -0.28)},
        human_face_init=(1.10, 0.08),
        human_face_moved=(1.05, -0.34),
        max_steps=130,
        record_frames=True, seed=15201,
    )
    print(f"  [v15 Social Mind] fakeout_ignored={l2_v15['fakeout_ignored']} | switched={l2_v15['target_switched']} | dock={l2_v15['final_err_cm']}cm | found_new_seat={l2_v15['human_found_new_seat']} | gaze_err={l2_v15['final_face_gaze_err_deg']}° | coll={l2_v15['collision']}", flush=True)

    # --- L3: Piaget Stage-4 Object Permanence & NBV Occluded Peek ---
    print("\n--- L3: Piaget Object Permanence & Next-Best-View Occluded Peek (Tall 17.5cm Thermos Blocking LOS!) ---", flush=True)
    occ_cfg = {"cx": 0.32, "cy": -0.14, "rad": 0.042, "height": 0.175}
    l3_base = run_living_brain_episode(arena_std, map_pack, geo_fe, policy_v3, (0.58, -0.25), enable_cliff_instinct=False, enable_nbv_peek=False, static_occluder=occ_cfg, record_frames=False, seed=15301)
    l3_v15 = run_living_brain_episode(arena_std, map_pack, geo_fe, policy_v3, (0.58, -0.25), enable_cliff_instinct=False, enable_nbv_peek=True, static_occluder=occ_cfg, record_frames=True, seed=15301)
    print(f"  [Direct No-NBV] arr={l3_base['final_err_cm']}cm | clear={l3_base['min_clearance_cm']}cm | coll={l3_base['collision']}", flush=True)
    print(f"  [v15 NBV Peek]  peeked={l3_v15['nbv_peeks_done']}@{l3_v15['nbv_peek_xy']} | arr={l3_v15['final_err_cm']}cm | clear=+{l3_v15['min_clearance_cm']}cm | coll={l3_v15['collision']}", flush=True)

    # --- L4: Ultimate 3-in-1 Chaos Gauntlet (Moving Cup + Sudden Drop + Violent +88° Kidnap!) ---
    print("\n--- L4: Ultimate 3-in-1 Chaos Gauntlet (Moving Cup + Sudden Drop + Violent +88° Kidnap ALL IN ONE!) ---", flush=True)
    m_cfg = {"start_step": 4, "x0": 0.50, "y0": -0.37, "vx": -0.045, "vy": 0.095, "rad": 0.036, "height": 0.120}
    d_cfg = {"step": 16, "cx": 0.44, "cy": -0.23, "rad": 0.036, "height": 0.115}
    k_cfg = {"step": 22, "dx": -0.07, "dy": 0.08, "dyaw": 88.0}
    l4_base = run_living_brain_episode(
        arena_std, map_pack, geo_fe, policy_v3, (0.62, -0.25),
        enable_cliff_instinct=False, enable_global_recovery=False, enable_velocity_prediction=False,
        moving_obs_cfg=m_cfg, static_drop_obs=d_cfg, kidnap_event=k_cfg, record_frames=False, seed=15401,
    )
    l4_v15 = run_living_brain_episode(
        arena_std, map_pack, geo_fe, policy_v3, (0.62, -0.25),
        enable_cliff_instinct=False, enable_global_recovery=True, enable_velocity_prediction=True,
        moving_obs_cfg=m_cfg, static_drop_obs=d_cfg, kidnap_event=k_cfg, record_frames=True, seed=15401,
    )
    print(f"  [Baseline No-v15]   arr={l4_base['final_err_cm']}cm | clear={l4_base['min_clearance_cm']}cm | coll={l4_base['collision']}", flush=True)
    print(f"  [v15 Triple-Mind]   det_cup@s{l4_v15['attack1_moving_det_step']} | det_drop@s{l4_v15['attack2_drop_det_step']} | kidnap_rec={l4_v15['attack3_kidnap_recovered']}({l4_v15['attack3_kidnap_recov_steps']} steps) | arr={l4_v15['final_err_cm']}cm | clear=+{l4_v15['min_clearance_cm']}cm | coll={l4_v15['collision']}", flush=True)

    # 渲染无冗余静帧、带实时深度小图 + 表情 HUD 的电影级 MP4
    mp4_p = render_v15_cinema_video(
        [
            ("L1: Visual Cliff Instinct -> Refuses Suicidal Knock & Shakes Head!", l1_v15["vis_records"], -0.36),
            ("L2: Teasing Immunity + Mid-Course Switch + Active Human Seat Search", l2_v15["vis_records"], None),
            ("L3: Piaget Object Permanence -> Next-Best-View Peek Around Tall Blocker", l3_v15["vis_records"], None),
            ("L4: 3x Chaos Gauntlet -> Moving Cup + Sudden Drop + Violent Kidnap!", l4_v15["vis_records"], None),
        ],
        OUT_DIR / "demo_living_brain_v15.mp4",
    )

    def _strip(d: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in d.items() if k != "vis_records"}
        if "vis_records" in d and d["vis_records"]:
            # 为 HTML5 互动剧场保存轻量级每帧回放轨迹（不含大图像数组）
            out["replay_frames"] = [
                {
                    "step": r["step"], "tx": r["tx"], "ty": r["ty"], "tyaw": r["tyaw"], "head": r["head_deg"],
                    "mood": r["mood"], "text": r["mood_text"], "tgt": r["target_xy"],
                    "cyls": r["cyls"], "tube": r["kf_tube"], "nbv": r["nbv_peek"],
                    "face": r["true_face_xy"], "cliff_est": r["cliff_y_est"], "clear_cm": r["clear_cm"],
                }
                for r in _trim_idle_tail(d["vis_records"], max_idle=8)
            ]
        return out

    rep = {
        "L1_visual_cliff_survival": {"cliff_y_true_m": -0.36, "baseline_no_cliff": _strip(l1_base), "v15_living_mind": _strip(l1_v15)},
        "L2_teasing_and_active_face_search": _strip(l2_v15),
        "L3_object_permanence_nbv_peek": {"occluder_cfg": occ_cfg, "baseline_no_nbv": _strip(l3_base), "v15_nbv_peek": _strip(l3_v15)},
        "L4_triple_chaos_gauntlet": {
            "attack1_moving_cup": m_cfg, "attack2_sudden_drop": d_cfg, "attack3_kidnap": k_cfg,
            "baseline_no_v15": _strip(l4_base), "v15_living_mind": _strip(l4_v15),
        },
        "demo_mp4": str(mp4_p),
    }
    png_p = plot_v15_poster(rep, OUT_DIR / "showcase_v15_living_desktop_brain.png")
    rep["poster_png"] = str(png_p)
    rep_p = OUT_DIR / "living_brain_v15_report.json"
    rep_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved v15 Poster -> {png_p}", flush=True)
    print(f"Saved v15 Cinema MP4 -> {mp4_p}", flush=True)
    print(f"Saved v15 Report -> {rep_p}", flush=True)


if __name__ == "__main__":
    main()
