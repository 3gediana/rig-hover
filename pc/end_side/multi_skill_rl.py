# -*- coding: utf-8 -*-
"""multi_skill_rl.py — RIG-Hover 多技能具身强化学习动物园 (Multi-Skill Embodied RL Zoo)。

严格遵循端侧 `[1, 39] -> [1, 3]` (`[vx, vyaw, head]`) 统一架构契约
(`EndSidePolicyNet(feature_dim=39, hidden_dim=128, action_dim=3, use_residual=True)`)，
在校准后的 167Hz LQR 轮式倒立摆动力学 (`sim.Plant`) + 二维平面运动学闭环仿真器中，
通过「Stage 1 行为克隆预热 (BC Warm-Start) + Stage 2 非对称特权闭环 PPO (CUDA)」训练并导出三大全新具身技能：

1. `hover-skill-orbit-inspector-v1` (`orbit_inspector_v1.tflite`) — 360° 环绕检视哨兵：
   - 围绕视觉检测目标 (`slot0`) 或空间锁定目标 (`slot1`) 以期望半径 r* = 0.48 m 平滑 360° 环绕；
   - 协调前向弧线线速度 `vx` (~28..45%)、向心偏航角速度 `vyaw` 与向内注视云台角 `head` (±38°)，全程 `|pitch| < 8°`。
2. `hover-skill-social-proxemics-v1` (`social_proxemics_v1.tflite`) — 弹性社交距离伴随 (Come-Closer / Back-Off)：
   - 维持舒适社交距离 d* = 0.75 m (`slot0` 框高 h* = 0.28)：人靠太近 (`h > 0.34`) 时带俯仰阻尼平滑后退 (`vx < 0`，至 `-45%`)；
   - 人退远 (`0.10 < h < 0.22`) 时温和跟进 (`vx > 0`)，同时通过 `head + vyaw` 保持视线居中锁定。
3. `hover-skill-figure8-patrol-v1` (`figure8_patrol_v1.tflite`) — 自平衡律动巡检与即时凝视捕获：
   - 无目标 (`feat[38] == 0`) 时利用内部 IMU/轮速/云台相位反馈 (`roll/pitch/v1/v2/head_q/cmd_vyaw`) 自激产生零漂移
     Bessel 8 字形主动巡检与正弦扫视；一旦目标出现 (`feat[38] == 1`)，立即刹停轮速 (`vx -> 0`) 并平滑捕获视线。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SIM_DIR = HERE.parent / "sim"
PERC_DIR = HERE.parent / "perception"
for _p in (str(HERE), str(SIM_DIR), str(PERC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params as P  # noqa: E402
import sim  # noqa: E402
from end_side_controller import EndSideFeatureBuilder, EndSidePolicyNet, TargetLock  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry, _clamp  # noqa: E402
from export_tflite_bundle import export_policy_bundle, verify_tflite_numerical  # noqa: E402
from spatial_memory import wrap180  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220

SKILL_SPECS: dict[str, dict[str, str]] = {
    "orbit_inspector": {
        "model_name": "hover-skill-orbit-inspector-v1",
        "asset_stem": "orbit_inspector_v1",
        "version": "2026.11.18-zoo-v1",
        "title_zh": "360° 环绕检视哨兵 (Orbit Inspector)",
        "notes": "360-deg Orbit Inspector around detected/locked target (r*=0.48m, vx 28..45%, inward head +-38 deg, |pitch|<8 deg).",
    },
    "social_proxemics": {
        "model_name": "hover-skill-social-proxemics-v1",
        "asset_stem": "social_proxemics_v1",
        "version": "2026.11.18-zoo-v1",
        "title_zh": "弹性社交距离伴随 (Social Proxemics)",
        "notes": "Elastic Social Proxemics companion maintaining d*=0.75m (h*=0.28) with pitch-damped back-off (vx<0) & approach (vx>0).",
    },
    "figure8_patrol": {
        "model_name": "hover-skill-figure8-patrol-v1",
        "asset_stem": "figure8_patrol_v1",
        "version": "2026.11.18-zoo-v1",
        "title_zh": "8 字律动巡检与凝视捕获 (Figure-8 Patrol)",
        "notes": "Self-sustaining Figure-8 active search via internal IMU/wheel/head phase feedback + instant brake & gaze capture on target.",
    },
}


def build_skill_projection_matrix() -> np.ndarray:
    """构造 39x39 姿态/坐标不变性投影矩阵 P (训练后精确折叠进 fc1.weight = fc1.weight @ P)。

    严格屏蔽绝对世界航向与累积里程计 (yaw[2], x1[10], x2[11], cmd_pos[18], cmd_yaw_target[19])，
    保留全部内部相位反馈通道 (roll, pitch, pit_gyro, dq, v1, v2, head_q, head_target, gimbal_target, cmd_vx, cmd_vyaw)
    并对双目标槽做高灵敏度零中心化映射。
    """
    P_mat = np.zeros((39, 39), dtype=np.float32)
    P_mat[0, 0] = 6.0             # roll / 30 deg
    P_mat[1, 1] = 6.0             # pitch / 30 deg
    P_mat[3, 3] = 1.0             # ax / 20
    P_mat[4, 4] = 1.0             # ay / 20
    P_mat[6, 6] = 1.0             # pit_gyro
    P_mat[7, 7] = 1.0             # dq
    P_mat[8, 8] = 1.0             # v1 / 100
    P_mat[9, 9] = 1.0             # v2 / 100
    P_mat[12, 12] = 75.0 / 45.0   # head_q / 45 deg
    P_mat[13, 13] = 75.0 / 45.0   # head_target / 45 deg
    P_mat[14, 14] = 1.0           # gimbal_target / 45 deg
    P_mat[15, 15] = 1.0           # cmd_vx / 100
    P_mat[16, 16] = 1.0           # cmd_vyaw / 100
    P_mat[20, 20] = 1.0           # t1 / 100
    P_mat[21, 21] = 1.0           # t2 / 100
    P_mat[22, 22] = 1.0           # pos_e / 100
    P_mat[24, 24] = 1.0           # vel_e / 100
    P_mat[26, 26] = 1.0           # pit_e / 100

    # Slot 0 (视觉检测主槽)
    P_mat[28, 28] = 2.0           # 2 * cx
    P_mat[29, 29] = 1.0           # cy
    P_mat[30, 30] = 2.5           # 2.5 * w
    P_mat[31, 31] = 2.5           # 2.5 * h (h*=0.28 -> 0.70)
    P_mat[32, 32] = 1.0           # score

    # Slot 1 (SE(2) 空间锁定槽 / 次视觉槽)
    P_mat[33, 33] = 2.0
    P_mat[33, 37] = -1.0          # lock_valid==1 时精确为 (bearing_body - head_q) / 30 deg
    P_mat[34, 34] = 2.0
    P_mat[34, 37] = -1.0          # lock_valid==1 时精确为 bearing_body / 90 deg (覆盖 +-180 deg)
    P_mat[35, 35] = 2.0           # remaining_m / 0.75m
    P_mat[36, 36] = 1.0           # clearance_m / 1.0m
    P_mat[37, 37] = 1.0           # lock_valid / slot1_conf
    P_mat[38, 38] = 1.0           # target_present (0/1)
    return P_mat


SKILL_PROJ_39 = build_skill_projection_matrix()


def mirror_feat39_batch(x: torch.Tensor) -> torch.Tensor:
    """对批量 39 维端侧特征执行左右镜像变换 M_s(x)。"""
    m = x.clone()
    for idx in (0, 12, 13, 14, 16, 24):
        m[:, idx] = -x[:, idx]
    m[:, 8] = x[:, 9]
    m[:, 9] = x[:, 8]
    m[:, 20] = x[:, 21]
    m[:, 21] = x[:, 20]
    has_s0 = (x[:, 32] > 0.05).float()
    m[:, 28] = has_s0 * (1.0 - x[:, 28]) + (1.0 - has_s0) * x[:, 28]
    has_s1 = (x[:, 37] > 0.05).float()
    m[:, 33] = has_s1 * (1.0 - x[:, 33]) + (1.0 - has_s1) * x[:, 33]
    m[:, 34] = has_s1 * (1.0 - x[:, 34]) + (1.0 - has_s1) * x[:, 34]
    return m


def mirror_feat39_np(feat_39: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(feat_39, dtype=np.float32).reshape(1, 39))
    return mirror_feat39_batch(t)[0].numpy().astype(np.float32)


# =========================================================================== #
# 三大具身技能解析专家控制律 (用于 Stage 1 BC 预热与 Stage 2 PPO 锚定)        #
# =========================================================================== #

def orbit_expert_action(feat_39: np.ndarray, desired_radius_m: float = 0.48) -> np.ndarray:
    """Skill 1 专家：Lyapunov 极限环向量场 360° 环绕检视控制律 (r* = 0.48m, vx 28..45%, head +-38°)。"""
    q_head_deg = float(feat_39[12]) * 75.0
    pitch_deg = float(feat_39[1]) * 180.0
    cmd_vx_pct = float(feat_39[15]) * 100.0
    has_lock = float(feat_39[37]) > 0.90
    has_s0 = float(feat_39[32]) >= 0.35

    if has_lock:
        bearing_deg = (float(feat_39[34]) - 0.5) * 180.0
        r_m = max(0.18, float(feat_39[35]) * 1.5)
    elif has_s0:
        theta_img_deg = (float(feat_39[28]) - 0.5) * 60.0
        bearing_deg = q_head_deg + theta_img_deg
        h_box = max(float(feat_39[31]), 0.10)
        r_m = float(_clamp(0.230 / h_box, 0.20, 1.35))
    elif abs(q_head_deg) > 18.0 and cmd_vx_pct > 15.0:
        # 环绕途中瞬时丢帧：依靠内部云台角与前向速度维持向心环绕并将目标拉回视野
        sgn_hold = 1.0 if q_head_deg >= 0.0 else -1.0
        return np.array([34.0, sgn_hold * 24.0, sgn_hold * 38.0], dtype=np.float32)
    else:
        return np.array([0.0, 0.0, 0.65 * q_head_deg], dtype=np.float32)

    if abs(bearing_deg) >= 4.5:
        sgn = 1.0 if bearing_deg >= 0.0 else -1.0
    elif abs(q_head_deg) >= 4.0:
        sgn = 1.0 if q_head_deg >= 0.0 else -1.0
    else:
        sgn = 1.0  # 打破正前方 0° 鞍点进入逆时针平滑入轨

    beta_des = sgn * (90.0 - math.degrees(math.atan(4.2 * (r_m - desired_radius_m))))
    err_beta = wrap180(bearing_deg - beta_des)

    vx_nom = 38.5 - 8.5 * min(1.0, abs(err_beta) / 75.0)
    pitch_dev = abs(pitch_deg + 2.0)
    pitch_scale = _clamp(1.0 - max(0.0, pitch_dev - 3.8) / 7.5, 0.40, 1.0)
    vx_pct = float(_clamp(vx_nom * pitch_scale, 28.5, 44.0))

    v_mps = (vx_pct / 100.0) * 0.30
    w_ff_dps = math.degrees((v_mps / max(r_m, 0.26)) * math.sin(math.radians(bearing_deg)))
    vyaw_dps = float(_clamp(w_ff_dps + 1.65 * err_beta, -62.0, 62.0))
    vyaw_pct = (vyaw_dps / 90.0) * 100.0

    # 云台始终朝轨道圆心内侧凝视 (限幅 +-38°，确保相机尽可能指向被检视目标)
    head_deg = float(_clamp(bearing_deg, -38.0, 38.0))
    return np.array([vx_pct, vyaw_pct, head_deg], dtype=np.float32)


def proxemics_expert_action(feat_39: np.ndarray) -> np.ndarray:
    """Skill 2 专家：弹性社交距离伴随控制律 (d* = 0.75m <-> h* = 0.28；近退远进 + 俯仰阻尼刹车)。"""
    q_head_deg = float(feat_39[12]) * 75.0
    pitch_deg = float(feat_39[1]) * 180.0
    pit_gyro_dps = float(feat_39[6]) * 100.0
    v_avg_mps = 0.5 * (float(feat_39[8]) + float(feat_39[9]))
    has_s0 = float(feat_39[32]) >= 0.35
    has_lock = float(feat_39[37]) > 0.90

    if not has_s0 and not has_lock:
        vx_brake = float(_clamp(-18.0 * v_avg_mps, -15.0, 15.0)) if abs(v_avg_mps) > 0.02 else 0.0
        return np.array([vx_brake, 0.0, 0.70 * q_head_deg], dtype=np.float32)

    if has_s0:
        theta_img_deg = (float(feat_39[28]) - 0.5) * 60.0
        eps_deg = q_head_deg + theta_img_deg
        h_box = float(_clamp(float(feat_39[31]), 0.06, 0.95))
        d_est = 0.210 / h_box  # h* = 0.28 <-> d* = 0.75m
    else:
        eps_deg = (float(feat_39[34]) - 0.5) * 180.0
        d_est = max(0.20, float(feat_39[35]) * 1.5)
        h_box = 0.210 / d_est

    # 弹性距离律：h > 0.295 (尤其 h > 0.34) 平滑后退至 -45%；0.10 < h < 0.265 温和跟进；舒适区主动阻尼刹停
    if h_box > 0.295:
        vx_raw = -min(45.0, 210.0 * (h_box - 0.295) + 55.0 * max(0.0, 0.712 - d_est))
    elif h_box < 0.265:
        vx_raw = min(43.0, 82.0 * (d_est - 0.792))
    else:
        vx_raw = -24.0 * v_avg_mps

    turn_gate = max(0.30, math.cos(math.radians(min(72.0, abs(eps_deg) * 1.15))))
    pitch_scale = _clamp(1.0 - max(0.0, abs(pitch_deg + 2.0) - 3.8) / 7.5, 0.25, 1.0)
    pitch_rate_damp = -0.06 * pit_gyro_dps if abs(vx_raw) > 5.0 else 0.0
    vx_pct = float(_clamp(vx_raw * turn_gate * pitch_scale + pitch_rate_damp, -45.0, 43.0))
    if 0.268 <= h_box <= 0.292 and abs(v_avg_mps) < 0.015:
        vx_pct = 0.0

    # 头身协同居中凝视
    head_deg = 0.0 if abs(eps_deg) < 0.8 else float(_clamp(0.88 * eps_deg, -40.0, 40.0))
    vyaw_dps = 0.0 if abs(eps_deg) < 2.2 else float(_clamp(1.95 * eps_deg, -58.0, 58.0))
    vyaw_pct = (vyaw_dps / 90.0) * 100.0
    return np.array([vx_pct, vyaw_pct, head_deg], dtype=np.float32)


def figure8_expert_action(feat_39: np.ndarray) -> np.ndarray:
    """Skill 3 专家：无目标时基于内部相位反馈自激 Bessel 8 字巡检；见目标瞬间刹停轮速并捕获视线。"""
    q_head_deg = float(feat_39[12]) * 75.0
    pitch_deg = float(feat_39[1]) * 180.0
    roll_deg = float(feat_39[0]) * 180.0
    v1_mps = float(feat_39[8])
    v2_mps = float(feat_39[9])
    v_avg_mps = 0.5 * (v1_mps + v2_mps)
    cmd_vyaw_pct = float(feat_39[16]) * 100.0
    target_visible = float(feat_39[38]) > 0.5

    # 分支 A: 一旦视野出现任何目标 (feat[38] == 1)，立即制动前向巡航 (vx -> 0) 并快速平滑锁定视线！
    if target_visible:
        if float(feat_39[32]) >= 0.35:
            theta_img_deg = (float(feat_39[28]) - 0.5) * 60.0
            eps_deg = q_head_deg + theta_img_deg
        elif float(feat_39[37]) > 0.5:
            eps_deg = (float(feat_39[34]) - 0.5) * 180.0
        else:
            eps_deg = q_head_deg

        head_deg = 0.0 if abs(eps_deg) < 0.8 else float(_clamp(0.90 * eps_deg, -42.0, 42.0))
        vyaw_dps = 0.0 if abs(eps_deg) < 2.0 else float(_clamp(2.25 * eps_deg, -64.0, 64.0))
        vyaw_pct = (vyaw_dps / 90.0) * 100.0
        return np.array([0.0, vyaw_pct, head_deg], dtype=np.float32)

    # 分支 B: 无目标 (feat[38] == 0) -> C^inf 光滑 Hopf 相平面振荡器生成零漂移 Bessel 8 字轨迹
    u1 = q_head_deg / 34.5
    u2 = cmd_vyaw_pct / 88.0
    r_sq = u1 * u1 + u2 * u2
    r_reg = math.sqrt(r_sq + 0.04 * 0.04)
    g_kick = math.exp(-r_sq / (2.0 * 0.18 * 0.18))

    # 原点微扰平滑破缺 + 单位圆吸引子旋转推进 (dphi = 0.026 rad/step -> 航向摆幅 +-139°, J0(2.405)=0)
    tu1 = (1.0 - g_kick) * (u1 / r_reg) + g_kick * 0.16
    tu2 = (1.0 - g_kick) * (u2 / r_reg) + g_kick * 0.98
    tr = math.sqrt(tu1 * tu1 + tu2 * tu2 + 1e-6)
    hu1, hu2 = tu1 / tr, tu2 / tr

    dphi = 0.026
    lead_h = dphi + 0.09
    vyaw_pct = 88.0 * (hu1 * math.sin(dphi) + hu2 * math.cos(dphi))
    head_deg = 34.5 * (hu1 * math.cos(lead_h) - hu2 * math.sin(lead_h))

    pitch_dev = abs(pitch_deg + 2.0) + 0.25 * abs(roll_deg) + 2.0 * abs(v_avg_mps - 0.075)
    pitch_scale = _clamp(1.0 - max(0.0, pitch_dev - 4.2) / 8.5, 0.45, 1.0)
    vx_pct = 25.5 * pitch_scale
    return np.array([float(vx_pct), float(vyaw_pct), float(head_deg)], dtype=np.float32)


def expert_action_for_skill(skill: str, feat_39: np.ndarray) -> np.ndarray:
    if skill == "orbit_inspector":
        return orbit_expert_action(feat_39)
    if skill == "social_proxemics":
        return proxemics_expert_action(feat_39)
    if skill == "figure8_patrol":
        return figure8_expert_action(feat_39)
    raise ValueError(f"Unknown skill {skill!r}")


# =========================================================================== #
# 校准 167Hz sim.Plant + 二维刚体运动学多技能闭环训练与评测沙箱                #
# =========================================================================== #

class MultiSkillArenaEnv:
    """20.8Hz (48ms = 8x6ms) 端侧策略 × 166.7Hz `sim.Plant` 轮式倒立摆二维多技能物理环境。"""

    CTRL_SUBSTEPS = 8
    ORBIT_RADIUS_M = 0.48
    SOCIAL_DIST_M = 0.75

    def __init__(
        self,
        skill: str,
        seed: int = 42,
        domain_rand: bool = False,
        max_steps: int = 260,
        lock_mode: bool | None = None,
    ):
        if skill not in SKILL_SPECS:
            raise ValueError(f"Unknown skill {skill!r}")
        self.skill = skill
        self.base_seed = int(seed)
        self.domain_rand = bool(domain_rand)
        self.max_steps = int(max_steps)
        self.forced_lock_mode = lock_mode

        self.base_sp = sim.calibrated(P.SimParams(), verbose=False)
        self.dt_inner = self.base_sp.fw.ctrl_dt_ms * 1e-3
        self.dt_step = self.dt_inner * self.CTRL_SUBSTEPS
        self.reset(seed=seed)

    def _sample_sim_params(self, rng: np.random.Generator) -> P.SimParams:
        if not self.domain_rand:
            return self.base_sp
        pp = self.base_sp.plant
        rand_pp = replace(
            pp,
            mass_body_kg=float(pp.mass_body_kg * rng.uniform(0.87, 1.13)),
            wheel_rolling_damping=float(pp.wheel_rolling_damping * rng.uniform(0.85, 1.15)),
            yaw_damping=float(pp.yaw_damping * rng.uniform(0.84, 1.18)),
            k_yaw_accel=float(pp.k_yaw_accel * rng.uniform(0.88, 1.12)),
            actuator_delay_ticks=int(rng.choice([1, 2])),
        )
        return replace(self.base_sp, plant=rand_pp)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed * 6151 + 97)
        self.sp = self._sample_sim_params(self.rng)
        self.plant = sim.Plant(self.sp, self.dt_inner)
        self.plant.reset(pitch0_deg=float(self.rng.uniform(-1.0, 1.0)), seed=self.base_seed)
        self.imu = sim.Imu(self.sp, self.rng if self.domain_rand else None)
        self.fw = sim.FirmwareControl(self.sp)
        self.fw.k_dyaw = -0.52
        self.fw.head_rate_comp = 1.0
        self.arb = sim.Arbiter(self.sp, 2)
        self.sv = sim.SensorView()

        self.t_s = 0.0
        self.step_idx = 0
        self.rx_m = 0.0
        self.ry_m = 0.0
        self.slip_scale = float(self.rng.uniform(0.94, 1.06)) if self.domain_rand else 1.0
        self.trail_xy: list[tuple[float, float]] = [(0.0, 0.0)]

        sgn = 1.0 if (self.base_seed % 2 == 0) else -1.0
        if self.skill == "orbit_inspector":
            r0 = float(self.rng.uniform(0.36, 0.70))
            ang0 = sgn * float(self.rng.uniform(8.0, 52.0))
            self.target_xy = (r0 * math.cos(math.radians(ang0)), r0 * math.sin(math.radians(ang0)))
            if self.forced_lock_mode is not None:
                self.use_target_lock = bool(self.forced_lock_mode)
            else:
                self.use_target_lock = (self.base_seed % 3 != 2)  # 覆盖 slot1 锁定与纯 slot0 广角检测两种输入路径
            self.appear_step = 0
        elif self.skill == "social_proxemics":
            # 人物初始站在偏远或偏近位置，并在回合中途主动逼近/后退 + 左右摆动
            self.p_r0 = float(self.rng.uniform(1.08, 1.38)) if (self.base_seed % 2 == 0) else float(self.rng.uniform(0.42, 0.56))
            self.p_ang0 = sgn * float(self.rng.uniform(5.0, 22.0))
            self.p_sway_amp = sgn * float(self.rng.uniform(0.14, 0.24))
            self.p_sway_freq = float(self.rng.uniform(0.42, 0.65))
            self.target_xy = (
                self.p_r0 * math.cos(math.radians(self.p_ang0)),
                self.p_r0 * math.sin(math.radians(self.p_ang0)),
            )
            self.use_target_lock = False
            self.appear_step = 0
        else:  # figure8_patrol
            # 前 65% 时间无目标做自激 8 字巡检，随后人物突然出现在视野内检验急停与凝视捕获！
            self.appear_step = int(self.max_steps * float(self.rng.uniform(0.62, 0.72)))
            self.capture_rel_bearing = sgn * float(self.rng.uniform(16.0, 26.0))
            self.capture_dist_m = float(self.rng.uniform(0.68, 0.95))
            self.target_xy = (1.0, 0.0)
            self.target_spawned = False
            self.use_target_lock = False

        self.prev_orbit_alpha: float | None = None
        self.cum_orbit_deg: float = 0.0
        self._sync_sensors(0.0)
        return self._build_obs()

    def _sync_sensors(self, dt: float) -> None:
        q_meas = round(self.plant.st.q_head / self.sp.plant.head_quant_deg) * self.sp.plant.head_quant_deg
        if dt > 0.0:
            self.imu.update(self.plant, q_meas, dt)
        self.sv.roll, self.sv.pitch = self.imu.roll, self.imu.pitch
        self.sv.pit_gyro, self.sv.dq, self.sv.yaw = self.imu.pit_gyro, self.imu.dq, self.imu.yaw
        self.sv.yaw_rate = self.imu.yaw_rate
        self.sv.q_head = q_meas
        self.sv.qd_head = self.plant.st.qd_head
        self.sv.wheel1_vel = self.sv.wheel2_vel = self.plant.st.v_w
        self.sv.wheel_x = self.plant.st.x_w

    def _world_to_body(self, wx: float, wy: float) -> tuple[float, float, float, float]:
        dx = wx - self.rx_m
        dy = wy - self.ry_m
        psi = math.radians(self.plant.st.yaw_body)
        xb = math.cos(psi) * dx + math.sin(psi) * dy
        yb = -math.sin(psi) * dx + math.cos(psi) * dy
        dist = math.hypot(xb, yb)
        bearing_deg = math.degrees(math.atan2(yb, xb))
        return xb, yb, dist, bearing_deg

    def _update_dynamic_target(self) -> None:
        if self.skill == "social_proxemics":
            # 分段测试：前段保持初始远/近距离 -> 中段反向逼近或后撤 -> 后段稳定交互
            half_t = 0.42 * self.max_steps * self.dt_step
            if self.p_r0 > 0.90:
                # 先远 (1.25m, 诱发走近) -> 再逼近至 x=0.82m (由于机器人已前进至 ~0.50m，间距骤缩至 ~0.32m, h>0.55，诱发倒车！)
                if self.t_s < half_t:
                    px = self.p_r0
                else:
                    px = max(0.78, self.p_r0 - 0.28 * (self.t_s - half_t))
            else:
                # 先近 (0.48m, h=0.44>0.34 诱发倒车至 -0.27m) -> 再退远至 x=0.98m (间距 >1.1m, h<0.19 诱发走近)
                if self.t_s < half_t:
                    px = self.p_r0
                else:
                    px = min(1.02, self.p_r0 + 0.26 * (self.t_s - half_t))
            py = self.p_sway_amp * math.sin(self.p_sway_freq * self.t_s)
            self.target_xy = (px, py)
        elif self.skill == "figure8_patrol":
            if self.step_idx >= self.appear_step and not self.target_spawned:
                self.target_spawned = True
                gaze_world_rad = math.radians(self.plant.st.yaw_body + self.sv.q_head + self.capture_rel_bearing)
                self.target_xy = (
                    self.rx_m + self.capture_dist_m * math.cos(gaze_world_rad),
                    self.ry_m + self.capture_dist_m * math.sin(gaze_world_rad),
                )

    def _build_telemetry(self) -> RobotTelemetry:
        sv = self.sv
        body_yaw_rate_dps = sv.yaw_rate - sv.qd_head
        pos_err_m = sv.wheel_x - sv.stable_pos
        half_diff = 0.08 * body_yaw_rate_dps
        return RobotTelemetry(
            t_ms=int(round(self.t_s * 1000.0)),
            roll=float(sv.roll),
            pitch=float(sv.pitch),
            yaw=float(sv.yaw),
            ax=0.0,
            ay=0.0,
            az=9.81,
            pit_gyro=float(sv.pit_gyro / 10.0),
            dq=float(sv.dq / 10.0),
            v1=float(sv.wheel1_vel * 100.0 + half_diff),
            v2=float(sv.wheel2_vel * 100.0 - half_diff),
            x1=float(sv.wheel_x),
            x2=float(sv.wheel_x),
            head_q=float(sv.q_head),
            head_target=float(sv.target_head_pos),
            gimbal_target=float(sv.q_head - sv.target_head_pos),
            cmd_vx=float((sv.vx / 0.30) * 100.0),
            cmd_vyaw=float((sv.vyaw / 90.0) * 100.0),
            cmd_mode=int(sv.control_mode),
            cmd_pos=float(sv.stable_pos),
            cmd_yaw_target=float(sv.stable_yaw),
            t1=float(-self.fw.temp_u + self.fw.yaw_u),
            t2=float(self.fw.temp_u + self.fw.yaw_u),
            pos_e=float(pos_err_m * 100.0),
            pos_u=0.0,
            vel_e=float(np.clip(body_yaw_rate_dps / 60.0, -1.0, 1.0) * 100.0),
            vel_u=0.0,
            pit_e=float(sv.pitch),
            pit_u=0.0,
            heap=120000,
        )

    def _build_obs(self) -> dict[str, Any]:
        self._update_dynamic_target()
        telem = self._build_telemetry()
        _, _, dist_m, bearing_deg = self._world_to_body(*self.target_xy)
        theta_img_deg = wrap180(bearing_deg - telem.head_q)

        dets: list[dict[str, float]] = []
        t_lock: TargetLock | None = None

        if self.skill == "orbit_inspector":
            if self.use_target_lock:
                t_lock = TargetLock(valid=1, bearing_body=bearing_deg, dist_m=dist_m, remaining_m=dist_m)
                if abs(theta_img_deg) <= 32.0:
                    cx = float(_clamp(0.5 + theta_img_deg / 60.0, 0.03, 0.97))
                    h = float(_clamp(0.230 / max(dist_m, 0.18), 0.10, 0.80))
                    dets.append({"cx": cx, "cy": 0.48, "w": h * 0.85, "h": h, "score": 0.92, "cls": 0})
            else:
                # 广角/全景视觉追踪框输入 slot0 (测试纯 slot0 路径)
                if abs(theta_img_deg) <= 56.0:
                    cx = float(_clamp(0.5 + theta_img_deg / 60.0, 0.02, 0.98))
                    h = float(_clamp(0.230 / max(dist_m, 0.18), 0.10, 0.80))
                    dets.append({"cx": cx, "cy": 0.48, "w": h * 0.85, "h": h, "score": 0.92, "cls": 0})

        elif self.skill == "social_proxemics":
            if abs(theta_img_deg) <= 38.0:
                cx = float(_clamp(0.5 + theta_img_deg / 60.0, 0.02, 0.98))
                h = float(_clamp(0.210 / max(dist_m, 0.22), 0.08, 0.88))
                dets.append({"cx": cx, "cy": 0.36, "w": h * 0.78, "h": h, "score": 0.93, "cls": 0})

        else:  # figure8_patrol
            if self.step_idx >= self.appear_step:
                if abs(theta_img_deg) <= 42.0:
                    cx = float(_clamp(0.5 + theta_img_deg / 60.0, 0.02, 0.98))
                    h = float(_clamp(0.210 / max(dist_m, 0.25), 0.12, 0.65))
                    dets.append({"cx": cx, "cy": 0.38, "w": h * 0.80, "h": h, "score": 0.94, "cls": 0})
                else:
                    t_lock = TargetLock(valid=1, bearing_body=bearing_deg, dist_m=dist_m, remaining_m=dist_m)

        feat_39 = EndSideFeatureBuilder.build(telem, dets, target_lock=t_lock, clearance_m=1.0)
        return {
            "telemetry": telem,
            "detections": dets,
            "target_lock": t_lock,
            "feat_39": feat_39,
            "true_dist_m": dist_m,
            "true_bearing_deg": bearing_deg,
            "theta_img_deg": theta_img_deg,
        }

    def build_privileged_critic_obs(self, feat_39: np.ndarray) -> np.ndarray:
        """构建 47 维非对称特权 Critic 状态 (39维部署观测 + 8维世界几何与真实动力学状态)。"""
        xb, yb, dist_m, bear_deg = self._world_to_body(*self.target_xy)
        target_active = 1.0 if (self.skill != "figure8_patrol" or self.step_idx >= self.appear_step) else 0.0
        priv_8 = np.array(
            [
                _clamp(xb / 1.2, -1.5, 1.5),
                _clamp(yb / 1.2, -1.5, 1.5),
                _clamp(dist_m / 1.5, 0.0, 1.5),
                _clamp(bear_deg / 180.0, -1.0, 1.0),
                _clamp(self.plant.pitch_true_deg / 15.0, -2.0, 2.0),
                _clamp(self.plant.st.yaw_rate / 90.0, -2.0, 2.0),
                _clamp(self.plant.st.v_w / 0.30, -1.5, 1.5),
                target_active,
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat_39.astype(np.float32), priv_8], axis=0)

    def step(self, cmd: DriveCommand) -> tuple[dict[str, Any], dict[str, Any]]:
        c = cmd.sanitized()
        vx_mps = float((c.vx / 100.0) * 0.30)
        vyaw_dps = float((c.vyaw / 100.0) * 90.0)
        head_deg = float(_clamp(c.head, -45.0, 45.0))

        for _ in range(self.CTRL_SUBSTEPS):
            prev_xw = self.plant.st.x_w
            prev_yaw = self.plant.st.yaw_body
            self.arb.post_raw(sim.CtrlOut(head_des=head_deg, vx_des=vx_mps, vyaw_des=vyaw_dps, mode=1), self.t_s)
            self.arb.note_feature(self.t_s)
            self.fw.update_state(self.sv, self.t_s)
            self.arb.tick(self.sv, self.fw, self.t_s)
            _, _, h_cmd = self.fw.control(self.sv, self.t_s, do_update_state=False)
            self.plant.step(self.fw.temp_u, self.fw.yaw_u, h_cmd)
            self.t_s += self.dt_inner
            self._sync_sensors(self.dt_inner)

            ds = (self.plant.st.x_w - prev_xw) * self.slip_scale
            mid_rad = math.radians(0.5 * (prev_yaw + self.plant.st.yaw_body))
            self.rx_m += ds * math.cos(mid_rad)
            self.ry_m += ds * math.sin(mid_rad)

        self.step_idx += 1
        self.trail_xy.append((round(self.rx_m, 4), round(self.ry_m, 4)))

        # 统计围绕目标扫过的累计圆心角
        alpha_deg = math.degrees(math.atan2(self.ry_m - self.target_xy[1], self.rx_m - self.target_xy[0]))
        d_alpha = 0.0
        if self.prev_orbit_alpha is not None:
            d_alpha = wrap180(alpha_deg - self.prev_orbit_alpha)
            self.cum_orbit_deg += d_alpha
        self.prev_orbit_alpha = alpha_deg

        obs = self._build_obs()
        info = {
            "t_s": round(self.t_s, 3),
            "step": self.step_idx,
            "robot_xy": (self.rx_m, self.ry_m),
            "target_xy": self.target_xy,
            "robot_yaw_deg": float(self.plant.st.yaw_body),
            "head_q_deg": float(self.sv.q_head),
            "pitch_deg": float(self.plant.pitch_true_deg),
            "v_w_mps": float(self.plant.st.v_w),
            "yaw_rate_dps": float(self.plant.st.yaw_rate),
            "true_dist_m": obs["true_dist_m"],
            "true_bearing_deg": obs["true_bearing_deg"],
            "theta_img_deg": obs["theta_img_deg"],
            "d_orbit_deg": d_alpha,
            "cum_orbit_deg": abs(self.cum_orbit_deg),
            "target_visible": float(obs["feat_39"][38]) > 0.5,
            "done": self.step_idx >= self.max_steps or abs(self.plant.pitch_true_deg) > 35.0,
        }
        return obs, info


# =========================================================================== #
# 非对称特权 Actor-Critic 网络 (Actor 严格对齐 EndSidePolicyNet(39,128,3,Res)) #
# =========================================================================== #

class SkillActorCritic(nn.Module):
    """部署侧 Actor 100% 同构于 EndSidePolicyNet(39, 128, 3, use_residual=True)，支持首层无损折叠。"""

    def __init__(self, hidden_dim: int = 128, critic_dim: int = 160):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor = EndSidePolicyNet(
            feature_dim=39,
            hidden_dim=hidden_dim,
            action_dim=3,
            use_residual=True,
        )
        self.critic = nn.Sequential(
            nn.Linear(47, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-2.45, -2.35, -2.35], dtype=torch.float32))
        self.register_buffer("proj_mat", torch.from_numpy(SKILL_PROJ_39))

    def proj_obs(self, feat_39: torch.Tensor) -> torch.Tensor:
        return feat_39 @ self.proj_mat.T

    def forward_actor_norm(self, feat_39: torch.Tensor) -> torch.Tensor:
        return self.actor.forward_normalized(self.proj_obs(feat_39))

    def forward_value(self, priv_47: torch.Tensor) -> torch.Tensor:
        p39 = self.proj_obs(priv_47[:, :39])
        return self.critic(torch.cat([p39, priv_47[:, 39:]], dim=-1)).squeeze(-1)

    def symmetry_loss(self, feat_39: torch.Tensor) -> torch.Tensor:
        u = self.forward_actor_norm(feat_39)
        u_mir = self.forward_actor_norm(mirror_feat39_batch(feat_39))
        target_mir = torch.stack([u[:, 0], -u[:, 1], -u[:, 2]], dim=-1)
        return F.mse_loss(u_mir, target_mir)

    def export_folded_policy(self) -> EndSidePolicyNet:
        """将 39x39 投影矩阵无损折叠进 fc1.weight，产出标准 EndSidePolicyNet(39, 128, 3, use_residual=True)。"""
        folded = EndSidePolicyNet(
            feature_dim=39,
            hidden_dim=self.hidden_dim,
            action_dim=3,
            use_residual=True,
        )
        with torch.no_grad():
            w1_folded = self.actor.fc1.weight.cpu() @ self.proj_mat.cpu()
            folded.fc1.weight.copy_(w1_folded)
            folded.fc1.bias.copy_(self.actor.fc1.bias.cpu())
            folded.fc2.load_state_dict(self.actor.fc2.state_dict())
            assert folded.fc_res is not None and self.actor.fc_res is not None
            folded.fc_res.load_state_dict(self.actor.fc_res.state_dict())
            folded.fc3.load_state_dict(self.actor.fc3.state_dict())
        folded.eval()
        return folded


# =========================================================================== #
# Stage 1: 闭环专家采样 + 相空间稠密覆盖增强 + BC 监督预热                      #
# =========================================================================== #

def collect_skill_expert_dataset(
    skill: str,
    n_episodes: int = 36,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """采集校准倒立摆闭环专家轨迹 + 边界/相平面稠密增强样本。"""
    rng = np.random.default_rng(seed)
    feats_list: list[np.ndarray] = []
    acts_list: list[np.ndarray] = []

    max_steps = 280 if skill == "orbit_inspector" else 240
    for ep in range(n_episodes):
        env = MultiSkillArenaEnv(skill=skill, seed=seed + ep, domain_rand=(ep % 2 == 1), max_steps=max_steps)
        obs = env.reset()
        for _ in range(env.max_steps):
            f39 = obs["feat_39"]
            act = expert_action_for_skill(skill, f39)
            feats_list.append(f39)
            acts_list.append(act)

            # 对称镜像增强 (orbit 仅在明显偏离正前鞍点时镜像；proxemics 全量镜像；figure8 在见目标捕获段镜像)
            if skill == "social_proxemics":
                feats_list.append(mirror_feat39_np(f39))
                acts_list.append(np.array([act[0], -act[1], -act[2]], dtype=np.float32))
            elif skill == "orbit_inspector" and abs(obs["true_bearing_deg"]) > 8.0:
                feats_list.append(mirror_feat39_np(f39))
                acts_list.append(np.array([act[0], -act[1], -act[2]], dtype=np.float32))
            elif skill == "figure8_patrol" and float(f39[38]) > 0.5:
                feats_list.append(mirror_feat39_np(f39))
                acts_list.append(np.array([act[0], -act[1], -act[2]], dtype=np.float32))

            # 注入小幅探索扰动使专家覆盖恢复流形
            noisy_cmd = DriveCommand(
                vx=float(act[0] + rng.normal(0.0, 2.2)),
                vyaw=float(act[1] + rng.normal(0.0, 3.0)),
                head=float(act[2] + rng.normal(0.0, 1.8)),
            )
            obs, info = env.step(noisy_cmd)
            if info["done"]:
                break

    # 针对各技能补充关键边界流形合成锚点，确保全状态空间零死角泛化
    for _ in range(1800):
        st = RobotTelemetry(
            roll=float(rng.uniform(-3.0, 3.0)),
            pitch=float(rng.uniform(-7.0, 5.0)),
            pit_gyro=float(rng.uniform(-1.5, 1.5)),
            dq=float(rng.uniform(-1.5, 1.5)),
            v1=float(rng.uniform(-14.0, 14.0)),
            v2=float(rng.uniform(-14.0, 14.0)),
            head_q=float(rng.uniform(-40.0, 40.0)),
            head_target=float(rng.uniform(-40.0, 40.0)),
            cmd_vx=float(rng.uniform(-45.0, 45.0)),
            cmd_vyaw=float(rng.uniform(-90.0, 90.0)),
            cmd_mode=1,
        )
        dets: list[dict[str, float]] = []
        t_lock: TargetLock | None = None

        if skill == "orbit_inspector":
            mode_pick = int(rng.integers(0, 3))
            bear = float(rng.uniform(-135.0, 135.0))
            r_m = float(rng.uniform(0.24, 0.95))
            if mode_pick == 0:
                t_lock = TargetLock(valid=1, bearing_body=bear, dist_m=r_m, remaining_m=r_m)
            elif mode_pick == 1:
                th = float(_clamp(bear - st.head_q, -29.0, 29.0))
                h = float(_clamp(0.230 / r_m, 0.12, 0.78))
                dets.append({"cx": 0.5 + th / 60.0, "cy": 0.48, "w": 0.85 * h, "h": h, "score": 0.92, "cls": 0})
            # mode_pick == 2: 模拟无目标或瞬时丢帧保持
        elif skill == "social_proxemics":
            if rng.random() < 0.88:
                cx = float(rng.uniform(0.06, 0.94))
                h = float(rng.uniform(0.10, 0.62))
                dets.append({"cx": cx, "cy": 0.36, "w": 0.78 * h, "h": h, "score": float(rng.uniform(0.55, 0.98)), "cls": 0})
        else:  # figure8_patrol
            if rng.random() < 0.45:
                # 覆盖见目标急停捕获
                cx = float(rng.uniform(0.05, 0.95))
                h = float(rng.uniform(0.14, 0.52))
                dets.append({"cx": cx, "cy": 0.38, "w": 0.80 * h, "h": h, "score": float(rng.uniform(0.55, 0.98)), "cls": 0})
            elif rng.random() < 0.18:
                # 专门覆盖从零静止起点自激起振
                st.head_q = float(rng.uniform(-1.5, 1.5))
                st.cmd_vyaw = float(rng.uniform(-2.0, 2.0))
                st.v1 = st.v2 = 0.0

        f_syn = EndSideFeatureBuilder.build(st, dets, target_lock=t_lock, clearance_m=1.0)
        a_syn = expert_action_for_skill(skill, f_syn)
        feats_list.append(f_syn)
        acts_list.append(a_syn)

    return np.stack(feats_list, axis=0), np.stack(acts_list, axis=0)


def pretrain_skill_bc(
    skill: str,
    ac: SkillActorCritic,
    feats: np.ndarray,
    acts: np.ndarray,
    epochs: int = 38,
    batch_size: int = 256,
    lr: float = 2.2e-3,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Stage 1: 行为克隆预热 (CUDA) + 对称等变约束。"""
    ac.to(device)
    opt = torch.optim.AdamW(ac.actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(20261118)

    scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    y_norm = np.clip(acts / scale, -1.0, 1.0).astype(np.float32)

    losses: list[float] = []
    N = len(feats)
    for _ in range(epochs):
        perm = rng.permutation(N)
        ep_loss = 0.0
        n_b = 0
        for s in range(0, N, batch_size):
            idx = perm[s : s + batch_size]
            bx = torch.from_numpy(feats[idx]).to(device)
            by = torch.from_numpy(y_norm[idx]).to(device)
            opt.zero_grad()
            pred = ac.forward_actor_norm(bx)
            loss = F.mse_loss(pred, by)
            if skill == "social_proxemics":
                loss = loss + 0.30 * ac.symmetry_loss(bx)
            elif skill == "figure8_patrol":
                vis_mask = bx[:, 38] > 0.5
                if torch.any(vis_mask):
                    loss = loss + 0.25 * ac.symmetry_loss(bx[vis_mask])
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        sched.step()
        losses.append(round(ep_loss / max(1, n_b), 6))
    return losses


# =========================================================================== #
# Stage 2: 闭环多技能物理奖励与非对称特权 PPO 强化微调 (CUDA)                  #
# =========================================================================== #

def compute_skill_step_reward(
    skill: str,
    feat_39: np.ndarray,
    u_norm: np.ndarray,
    prev_u_norm: np.ndarray,
    info: dict[str, Any],
) -> float:
    """针对三大具身技能物理目标定制的闭环奖励函数。"""
    act_phys = u_norm * np.array([100.0, 100.0, 75.0], dtype=np.float32)
    vx_pct, vyaw_pct, head_deg = float(act_phys[0]), float(act_phys[1]), float(act_phys[2])
    pitch_deg = float(info["pitch_deg"])
    dist_m = float(info["true_dist_m"])
    bear_deg = float(info["true_bearing_deg"])
    th_img = float(info["theta_img_deg"])

    r_pitch = -0.35 * (pitch_deg / 8.0) ** 2 - (4.0 if abs(pitch_deg) > 8.0 else 0.0)
    r_jerk = -0.18 * float(np.sum((u_norm - prev_u_norm) ** 2))

    if skill == "orbit_inspector":
        r_rad = 1.45 * math.exp(-20.0 * (dist_m - 0.48) ** 2)
        in_band = 1.0 if (26.0 <= vx_pct <= 46.0) else 0.25
        r_orbit = 0.32 * abs(float(info["d_orbit_deg"])) * in_band
        des_inward_head = _clamp(bear_deg, -38.0, 38.0)
        r_gaze = 0.65 - 0.45 * ((info["head_q_deg"] - des_inward_head) / 22.0) ** 2
        return float(r_rad + r_orbit + r_gaze + r_pitch + r_jerk)

    if skill == "social_proxemics":
        r_dist = 1.65 * math.exp(-16.0 * (dist_m - 0.75) ** 2)
        r_gaze = 0.85 - 0.60 * (th_img / 18.0) ** 2
        # 靠太近时奖励及时倒车退让，在舒适区奖励静稳零速
        r_vel = 0.0
        if dist_m < 0.58 and vx_pct < -8.0:
            r_vel += 0.35
        elif 0.71 <= dist_m <= 0.79:
            r_vel += 0.30 * math.exp(-0.02 * vx_pct * vx_pct)
        return float(r_dist + r_gaze + r_vel + r_pitch + r_jerk)

    # figure8_patrol
    if bool(info["target_visible"]):
        # 目标出现：重奖瞬间零轮速制动 + 视线快速居中捕获！
        r_stop = 1.35 * math.exp(-0.04 * vx_pct * vx_pct) - 1.2 * abs(float(info["v_w_mps"]))
        r_snap = 1.15 * math.exp(-0.012 * th_img * th_img)
        return float(r_stop + r_snap + r_pitch + r_jerk)
    else:
        # 无目标巡检：奖励沿着零漂移 Bessel 8 字 Hopf 极限环切向推进 + 有界桌面包络
        rx, ry = info["robot_xy"]
        ref_act = figure8_expert_action(feat_39)
        ref_u = ref_act / np.array([100.0, 100.0, 75.0], dtype=np.float32)
        r_bound = 0.55 if (abs(rx) < 0.36 and abs(ry) < 0.42) else -1.5
        r_hopf = 1.15 * math.exp(-12.0 * float(np.mean((u_norm - ref_u) ** 2)))
        return float(r_bound + r_hopf + r_pitch + r_jerk)


def rollout_skill_ppo_episode(
    skill: str,
    ac: SkillActorCritic,
    seed: int,
    domain_rand: bool,
    device: torch.device,
) -> dict[str, Any]:
    max_steps = 250 if skill == "orbit_inspector" else 210
    env = MultiSkillArenaEnv(skill=skill, seed=seed, domain_rand=domain_rand, max_steps=max_steps)
    obs = env.reset()

    obs_buf, priv_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], [], []
    prev_u_norm = np.zeros(3, dtype=np.float32)
    act_scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    std = torch.exp(ac.log_std).to(device)

    for _ in range(env.max_steps):
        f39 = obs["feat_39"]
        p47 = env.build_privileged_critic_obs(f39)
        with torch.no_grad():
            ot = torch.from_numpy(f39).unsqueeze(0).to(device)
            pt = torch.from_numpy(p47).unsqueeze(0).to(device)
            mu = ac.forward_actor_norm(ot)[0]
            val = ac.forward_value(pt)[0]
            dist_n = torch.distributions.Normal(mu, std)
            u_t = torch.clamp(dist_n.sample(), -1.0, 1.0)
            logp_t = dist_n.log_prob(u_t).sum()

        u_norm = u_t.cpu().numpy()
        act_phys = u_norm * act_scale
        obs_next, info = env.step(DriveCommand(vx=float(act_phys[0]), vyaw=float(act_phys[1]), head=float(act_phys[2])))
        rew = compute_skill_step_reward(skill, f39, u_norm, prev_u_norm, info)

        obs_buf.append(f39)
        priv_buf.append(p47)
        act_buf.append(u_norm)
        logp_buf.append(float(logp_t.item()))
        rew_buf.append(rew)
        val_buf.append(float(val.item()))

        prev_u_norm = u_norm.copy()
        obs = obs_next
        if info["done"]:
            break

    return {
        "obs": np.stack(obs_buf, axis=0),
        "priv": np.stack(priv_buf, axis=0),
        "act": np.stack(act_buf, axis=0),
        "logp": np.array(logp_buf, dtype=np.float32),
        "rew": np.array(rew_buf, dtype=np.float32),
        "val": np.array(val_buf, dtype=np.float32),
    }


def train_skill_ppo(
    skill: str,
    ac: SkillActorCritic,
    expert_feats: np.ndarray,
    expert_acts: np.ndarray,
    n_iters: int = 10,
    episodes_per_iter: int = 6,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    lr_actor: float = 1.5e-4,
    lr_critic: float = 9.0e-4,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, Any]]:
    """Stage 2: 闭环 PPO 强化微调 + 特权 Critic 价值预热 + 物理平衡点锚定。"""
    ac.to(device)
    opt_actor = torch.optim.AdamW([p for n, p in ac.named_parameters() if "critic" not in n], lr=lr_actor)
    opt_critic = torch.optim.AdamW(ac.critic.parameters(), lr=lr_critic)
    scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    exp_y_norm = np.clip(expert_acts / scale, -1.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(20261118)

    # Critic 预热 5 轮，消除初始优势方差
    warm_trajs = [rollout_skill_ppo_episode(skill, ac, 400 + i, True, device) for i in range(6)]
    for _ in range(5):
        for tr in warm_trajs:
            rew = tr["rew"]
            ret = np.zeros_like(rew)
            g = 0.0
            for t in reversed(range(len(rew))):
                g = rew[t] + 0.99 * g
                ret[t] = g
            opt_critic.zero_grad()
            F.mse_loss(ac.forward_value(torch.from_numpy(tr["priv"]).to(device)), torch.from_numpy(ret).to(device)).backward()
            opt_critic.step()

    # 预构造 figure8 全相位 Hopf 极限环与捕获平衡点锚定张量 (避免内层循环重复构建)
    fig8_rx_t: torch.Tensor | None = None
    fig8_ry_t: torch.Tensor | None = None
    if skill == "figure8_patrol":
        ring_rows: list[np.ndarray] = []
        ring_tgts: list[np.ndarray] = []
        for h_v in (0.18, 0.26, 0.34, 0.44):
            st_c = RobotTelemetry(pitch=-2.0, v1=0.0, v2=0.0, head_q=0.0, cmd_mode=1)
            f_c = EndSideFeatureBuilder.build(
                st_c,
                [{"cx": 0.5, "cy": 0.38, "w": 0.8 * h_v, "h": h_v, "score": 0.92, "cls": 0}],
            )
            ring_rows.append(f_c)
            ring_tgts.append(np.zeros(3, dtype=np.float32))
        for p_var, r_var in ((-2.0, 0.0), (-3.6, 0.5), (-1.0, -0.5)):
            for k_ph in range(36):
                phi_k = (2.0 * math.pi * k_ph) / 36.0
                rad_k = 1.0 if k_ph < 33 else 0.0
                hq = 34.5 * rad_k * math.cos(phi_k)
                cvy = 88.0 * rad_k * math.sin(phi_k)
                half_d = 0.08 * (cvy * 0.90)
                st_p = RobotTelemetry(
                    roll=r_var,
                    pitch=p_var,
                    v1=7.6 + half_d,
                    v2=7.6 - half_d,
                    head_q=hq,
                    head_target=hq,
                    cmd_vx=25.5,
                    cmd_vyaw=cvy,
                    cmd_mode=1,
                )
                f_p = EndSideFeatureBuilder.build(st_p, [], target_lock=None)
                ring_rows.append(f_p)
                ring_tgts.append(np.clip(figure8_expert_action(f_p) / scale, -1.0, 1.0))
        fig8_rx_t = torch.from_numpy(np.stack(ring_rows, axis=0)).to(device)
        fig8_ry_t = torch.from_numpy(np.stack(ring_tgts, axis=0)).to(device)

    history: list[dict[str, Any]] = []
    for it in range(n_iters):
        with torch.no_grad():
            ac.log_std.copy_(torch.tensor([-2.35 - 0.10 * it, -2.25 - 0.10 * it, -2.25 - 0.10 * it], device=device))

        trajs = [
            rollout_skill_ppo_episode(skill, ac, seed=2000 + it * 20 + ep, domain_rand=(it >= 2), device=device)
            for ep in range(episodes_per_iter)
        ]

        obs_all, priv_all, act_all, logp_all, adv_all, ret_all = [], [], [], [], [], []
        ep_returns: list[float] = []
        for tr in trajs:
            rew, val = tr["rew"], tr["val"]
            ep_returns.append(float(np.sum(rew)))
            T = len(rew)
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                next_v = val[t + 1] if t + 1 < T else val[t]
                delta = rew[t] + 0.99 * next_v - val[t]
                gae = delta + 0.99 * 0.95 * gae
                adv[t] = gae
            obs_all.append(tr["obs"])
            priv_all.append(tr["priv"])
            act_all.append(tr["act"])
            logp_all.append(tr["logp"])
            adv_all.append(adv)
            ret_all.append(adv + val)

        obs_b = torch.from_numpy(np.concatenate(obs_all, axis=0)).to(device)
        priv_b = torch.from_numpy(np.concatenate(priv_all, axis=0)).to(device)
        act_b = torch.from_numpy(np.concatenate(act_all, axis=0)).to(device)
        old_logp_b = torch.from_numpy(np.concatenate(logp_all, axis=0)).to(device)
        adv_b = torch.from_numpy(np.concatenate(adv_all, axis=0)).to(device)
        ret_b = torch.from_numpy(np.concatenate(ret_all, axis=0)).to(device)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-6)

        N = len(obs_b)
        for _ in range(ppo_epochs):
            perm = rng.permutation(N)
            for s in range(0, N, 256):
                idx = perm[s : s + 256]
                std = torch.exp(ac.log_std)
                mu = ac.forward_actor_norm(obs_b[idx])
                dist_n = torch.distributions.Normal(mu, std)
                new_logp = dist_n.log_prob(act_b[idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_b[idx])
                surr1 = ratio * adv_b[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b[idx]
                loss_ppo = -torch.min(surr1, surr2).mean()

                exp_idx = rng.choice(len(expert_feats), size=min(256, len(expert_feats)), replace=False)
                bx_exp = torch.from_numpy(expert_feats[exp_idx]).to(device)
                by_exp = torch.from_numpy(exp_y_norm[exp_idx]).to(device)
                loss_anchor = F.mse_loss(ac.forward_actor_norm(bx_exp), by_exp)

                loss_sym = torch.tensor(0.0, device=device)
                loss_eq = torch.tensor(0.0, device=device)
                if skill == "social_proxemics":
                    loss_sym = ac.symmetry_loss(obs_b[idx]) + ac.symmetry_loss(bx_exp)
                    # 舒适社交距离平衡点 (cx=0.5, h*=0.28) 严格零动作偏置锚定
                    eq_x = torch.zeros((4, 39), dtype=torch.float32, device=device)
                    for i_eq, sc_v in enumerate((0.0, 0.75, 0.90, 0.96)):
                        if sc_v > 0.0:
                            eq_x[i_eq, 28] = 0.5
                            eq_x[i_eq, 29] = 0.36
                            eq_x[i_eq, 30] = 0.22
                            eq_x[i_eq, 31] = 0.28
                            eq_x[i_eq, 32] = sc_v
                            eq_x[i_eq, 38] = 1.0
                    loss_eq = (ac.forward_actor_norm(eq_x) ** 2).mean()
                elif skill == "figure8_patrol" and fig8_rx_t is not None and fig8_ry_t is not None:
                    loss_eq = F.mse_loss(ac.forward_actor_norm(fig8_rx_t), fig8_ry_t)

                opt_actor.zero_grad()
                (0.32 * loss_ppo + 0.88 * loss_anchor + 0.50 * loss_sym + 3.0 * loss_eq).backward()
                nn.utils.clip_grad_norm_(ac.actor.parameters(), 1.0)
                opt_actor.step()

                opt_critic.zero_grad()
                F.mse_loss(ac.forward_value(priv_b[idx]), ret_b[idx]).backward()
                opt_critic.step()

        mean_ret = float(np.mean(ep_returns))
        history.append({"iter": it + 1, "mean_return": round(mean_ret, 3)})
        print(f"  [{skill} PPO {it + 1:02d}/{n_iters:02d}] mean_return = {mean_ret:7.2f}", flush=True)
    return history


# =========================================================================== #
# 闭环基准评测与高分辨率展示图 fig7_multi_skill_zoo.png 绘制                  #
# =========================================================================== #

def benchmark_orbit_skill(policy: EndSidePolicyNet, n_eval: int = 8) -> dict[str, Any]:
    radial_rmses, sweeps, max_pitches, mean_vxs, steady_heads = [], [], [], [], []
    sample_trace: dict[str, Any] = {}

    for ep in range(n_eval):
        env = MultiSkillArenaEnv(
            skill="orbit_inspector",
            seed=900 + ep,
            domain_rand=(ep % 2 == 1),
            max_steps=560,
            lock_mode=(ep != n_eval - 1),  # 同时检验锁定与纯视觉 slot0
        )
        obs = env.reset()
        t_arr, rx_arr, ry_arr, r_arr, p_arr, vx_arr, vy_arr, hd_arr = [], [], [], [], [], [], [], []
        for _ in range(env.max_steps):
            cmd = policy.infer_numpy(obs["feat_39"])
            obs, info = env.step(cmd)
            t_arr.append(info["t_s"])
            rx_arr.append(info["robot_xy"][0])
            ry_arr.append(info["robot_xy"][1])
            r_arr.append(info["true_dist_m"])
            p_arr.append(info["pitch_deg"])
            vx_arr.append(cmd.vx)
            vy_arr.append(cmd.vyaw)
            hd_arr.append(info["head_q_deg"])

        steady = slice(160, len(r_arr))
        r_st = np.asarray(r_arr[steady], dtype=np.float64)
        radial_rmses.append(float(math.sqrt(np.mean((r_st - 0.48) ** 2)) * 100.0))
        sweeps.append(float(info["cum_orbit_deg"]))
        max_pitches.append(float(np.max(np.abs(p_arr))))
        mean_vxs.append(float(np.mean(vx_arr[steady])))
        steady_heads.append(float(np.mean(np.abs(hd_arr[steady]))))

        if ep == 0:
            sample_trace = {
                "t": t_arr,
                "rx": rx_arr,
                "ry": ry_arr,
                "target_xy": list(env.target_xy),
                "radius_m": r_arr,
                "pitch_deg": p_arr,
                "vx_pct": vx_arr,
                "vyaw_pct": vy_arr,
                "head_q_deg": hd_arr,
            }

    return {
        "desired_radius_m": 0.48,
        "radial_rmse_cm_mean": round(float(np.mean(radial_rmses)), 2),
        "orbit_sweep_deg_mean": round(float(np.mean(sweeps)), 1),
        "mean_vx_pct": round(float(np.mean(mean_vxs)), 2),
        "steady_abs_head_deg": round(float(np.mean(steady_heads)), 2),
        "max_abs_pitch_deg": round(float(np.max(max_pitches)), 2),
        "sample_trace": sample_trace,
    }


def benchmark_proxemics_skill(policy: EndSidePolicyNet, n_eval: int = 8) -> dict[str, Any]:
    final_dists, min_vxs, max_vxs, gaze_rmses, max_pitches = [], [], [], [], []
    sample_trace: dict[str, Any] = {}

    for ep in range(n_eval):
        env = MultiSkillArenaEnv(
            skill="social_proxemics",
            seed=950 + ep,
            domain_rand=(ep % 2 == 1),
            max_steps=380,
        )
        obs = env.reset()
        t_arr, d_arr, rx_arr, ry_arr, px_arr, py_arr, vx_arr, p_arr, g_arr = [], [], [], [], [], [], [], [], []
        for _ in range(env.max_steps):
            cmd = policy.infer_numpy(obs["feat_39"])
            obs, info = env.step(cmd)
            t_arr.append(info["t_s"])
            d_arr.append(info["true_dist_m"])
            rx_arr.append(info["robot_xy"][0])
            ry_arr.append(info["robot_xy"][1])
            px_arr.append(info["target_xy"][0])
            py_arr.append(info["target_xy"][1])
            vx_arr.append(cmd.vx)
            p_arr.append(info["pitch_deg"])
            g_arr.append(info["theta_img_deg"])

        final_dists.append(float(np.mean(d_arr[-45:])))
        min_vxs.append(float(np.min(vx_arr)))
        max_vxs.append(float(np.max(vx_arr)))
        gaze_rmses.append(float(math.sqrt(np.mean(np.asarray(g_arr) ** 2))))
        max_pitches.append(float(np.max(np.abs(p_arr))))

        if ep == 0:
            sample_trace = {
                "t": t_arr,
                "dist_m": d_arr,
                "rx": rx_arr,
                "ry": ry_arr,
                "px": px_arr,
                "py": py_arr,
                "vx_pct": vx_arr,
                "pitch_deg": p_arr,
                "gaze_err_deg": g_arr,
            }

    return {
        "desired_dist_m": 0.75,
        "steady_dist_m_mean": round(float(np.mean(final_dists)), 3),
        "steady_dist_err_cm": round(abs(float(np.mean(final_dists)) - 0.75) * 100.0, 2),
        "min_backoff_vx_pct": round(float(np.min(min_vxs)), 2),
        "max_approach_vx_pct": round(float(np.max(max_vxs)), 2),
        "gaze_rmse_deg": round(float(np.mean(gaze_rmses)), 2),
        "max_abs_pitch_deg": round(float(np.max(max_pitches)), 2),
        "sample_trace": sample_trace,
    }


def benchmark_figure8_skill(policy: EndSidePolicyNet, n_eval: int = 8) -> dict[str, Any]:
    patrol_spans_x, patrol_spans_y, head_swings, post_vxs, snap_errs, max_pitches = [], [], [], [], [], []
    sample_trace: dict[str, Any] = {}

    for ep in range(n_eval):
        max_st = 620 if ep == 0 else 440
        env = MultiSkillArenaEnv(
            skill="figure8_patrol",
            seed=980 + ep,
            domain_rand=(ep % 2 == 1),
            max_steps=max_st,
        )
        obs = env.reset()
        if ep == 0:
            env.appear_step = 460  # 首条展示轨迹完整呈现整圈 8 字后再触发凝视捕获

        t_arr, rx_arr, ry_arr, vx_arr, vy_arr, hd_arr, p_arr, vis_arr, g_arr = [], [], [], [], [], [], [], [], []
        for _ in range(env.max_steps):
            cmd = policy.infer_numpy(obs["feat_39"])
            obs, info = env.step(cmd)
            t_arr.append(info["t_s"])
            rx_arr.append(info["robot_xy"][0])
            ry_arr.append(info["robot_xy"][1])
            vx_arr.append(cmd.vx)
            vy_arr.append(cmd.vyaw)
            hd_arr.append(info["head_q_deg"])
            p_arr.append(info["pitch_deg"])
            vis_arr.append(1 if info["target_visible"] else 0)
            g_arr.append(info["theta_img_deg"])

        ap = env.appear_step
        pxs = rx_arr[:ap]
        pys = ry_arr[:ap]
        patrol_spans_x.append(float(max(pxs) - min(pxs)))
        patrol_spans_y.append(float(max(pys) - min(pys)))
        head_swings.append(float(max(hd_arr[:ap]) - min(hd_arr[:ap])))
        post_vxs.append(float(np.mean(np.abs(vx_arr[ap + 8 :]))))
        snap_errs.append(float(np.mean(np.abs(g_arr[-30:]))))
        max_pitches.append(float(np.max(np.abs(p_arr))))

        if ep == 0:
            sample_trace = {
                "t": t_arr,
                "appear_t_s": round(ap * env.dt_step, 3),
                "appear_idx": ap,
                "rx": rx_arr,
                "ry": ry_arr,
                "target_xy": list(env.target_xy),
                "vx_pct": vx_arr,
                "vyaw_pct": vy_arr,
                "head_q_deg": hd_arr,
                "pitch_deg": p_arr,
                "target_visible": vis_arr,
                "gaze_err_deg": g_arr,
            }

    return {
        "patrol_bbox_x_m_mean": round(float(np.mean(patrol_spans_x)), 3),
        "patrol_bbox_y_m_mean": round(float(np.mean(patrol_spans_y)), 3),
        "head_sweep_p2p_deg_mean": round(float(np.mean(head_swings)), 1),
        "post_capture_abs_vx_pct": round(float(np.mean(post_vxs)), 2),
        "captured_gaze_mae_deg": round(float(np.mean(snap_errs)), 2),
        "max_abs_pitch_deg": round(float(np.max(max_pitches)), 2),
        "sample_trace": sample_trace,
    }


def plot_fig7_multi_skill_zoo(
    metrics: dict[str, Any],
    out_file: Path,
) -> Path:
    """绘制 2x3 高分辨率多技能动物园综合展示图 (2D 空间轨迹 + 倒立摆俯仰/跟踪时序曲线)。"""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))

    orb = metrics["skills"]["orbit_inspector"]["benchmark"]
    prx = metrics["skills"]["social_proxemics"]["benchmark"]
    fg8 = metrics["skills"]["figure8_patrol"]["benchmark"]

    tr_o = orb["sample_trace"]
    tr_p = prx["sample_trace"]
    tr_f = fg8["sample_trace"]

    # --- (a) Orbit Inspector 2D 环绕轨迹 ---
    ax = axes[0, 0]
    tx, ty = tr_o["target_xy"]
    circ = patches.Circle((tx, ty), 0.48, fill=False, linestyle="--", linewidth=1.8, edgecolor="#dc2626", label="理想环绕圆 r* = 0.48 m")
    ax.add_patch(circ)
    ax.plot(tr_o["rx"], tr_o["ry"], color="#2563eb", linewidth=2.3, label=f"RL 环绕轨迹 (扫角 {orb['orbit_sweep_deg_mean']:.0f}°)")
    ax.scatter([tr_o["rx"][0]], [tr_o["ry"][0]], color="#0f172a", s=55, zorder=5, label="起始位置 (0, 0)")
    ax.scatter([tx], [ty], color="#dc2626", marker="*", s=160, zorder=6, label="桌面检视目标")
    # 绘制向心注视射线箭头
    for k_idx in range(90, len(tr_o["rx"]), 75):
        rx_k, ry_k = tr_o["rx"][k_idx], tr_o["ry"][k_idx]
        dx_k, dy_k = (tx - rx_k) * 0.24, (ty - ry_k) * 0.24
        ax.arrow(rx_k, ry_k, dx_k, dy_k, head_width=0.025, head_length=0.03, fc="#7c3aed", ec="#7c3aed", alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("世界 X (m)", fontsize=9.8)
    ax.set_ylabel("世界 Y (m)", fontsize=9.8)
    ax.set_title(
        f"(a) Skill 1: 360° 环绕检视哨兵 2D 轨迹\n径向 RMSE = {orb['radial_rmse_cm_mean']:.2f} cm | 内视头角 ±{orb['steady_abs_head_deg']:.1f}°",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.0)

    # --- (b) Social Proxemics 距离伴随包络 ---
    ax = axes[0, 1]
    tp = np.asarray(tr_p["t"])
    ax.axhline(0.75, color="#059669", linestyle="--", linewidth=1.9, label="舒适社交距离 d* = 0.75 m (h* = 0.28)")
    ax.axhspan(0.68, 0.82, color="#d1fae5", alpha=0.55, label="舒适伴随缓冲带")
    ax.plot(tp, tr_p["px"], color="#94a3b8", linestyle=":", linewidth=1.7, label="人体世界 X 轨迹 (远立 -> 逼近)")
    ax.plot(tp, tr_p["rx"], color="#8b5cf6", linestyle="-.", linewidth=1.8, label="机器人纵向位移 X (走近 -> 退让)")
    ax.plot(tp, tr_p["dist_m"], color="#2563eb", linewidth=2.5, label=f"实际人机间距 d(t) (稳态 {prx['steady_dist_m_mean']:.2f} m)")
    ax.set_xlabel("时间 t (s)", fontsize=9.8)
    ax.set_ylabel("纵向坐标 / 人机距离 (m)", fontsize=9.8)
    ax.set_title(
        f"(b) Skill 2: 弹性社交距离伴随 (Come-Closer / Back-Off)\n稳态误差 {prx['steady_dist_err_cm']:.1f} cm | 视线误差 {prx['gaze_rmse_deg']:.2f}°",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=7.9)

    # --- (c) Figure-8 Patrol 2D 有界 8 字轨迹与捕获点 ---
    ax = axes[0, 2]
    ap_i = int(tr_f["appear_idx"])
    ax.plot(tr_f["rx"][:ap_i], tr_f["ry"][:ap_i], color="#0284c7", linewidth=2.3, label="无目标自激 Bessel 8 字巡检轨迹")
    ax.plot(tr_f["rx"][ap_i:], tr_f["ry"][ap_i:], color="#dc2626", linewidth=2.8, label="见目标后原地驻停捕获段")
    ax.scatter([tr_f["rx"][0]], [tr_f["ry"][0]], color="#0f172a", s=50, zorder=5, label="巡检起点")
    ax.scatter([tr_f["rx"][ap_i]], [tr_f["ry"][ap_i]], color="#f59e0b", marker="D", s=70, zorder=6, label=f"t={tr_f['appear_t_s']:.1f}s 目标突现急停点")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("世界 X (m)", fontsize=9.8)
    ax.set_ylabel("世界 Y (m)", fontsize=9.8)
    ax.set_title(
        f"(c) Skill 3: 有界 8 字巡检与即时凝视捕获\n巡检包络 {fg8['patrol_bbox_x_m_mean']*100:.0f}×{fg8['patrol_bbox_y_m_mean']*100:.0f} cm | 捕获残速 {fg8['post_capture_abs_vx_pct']:.2f}%",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=7.9)

    # --- (d) Orbit Inspector 半径收敛、动作与俯仰稳定性 ---
    ax = axes[1, 0]
    to = np.asarray(tr_o["t"])
    l1 = ax.plot(to, np.asarray(tr_o["radius_m"]) * 100.0, color="#2563eb", linewidth=2.2, label="环绕半径 r(t) (cm, 左轴)")
    ax.axhline(48.0, color="#dc2626", linestyle="--", linewidth=1.4, label="期望半径 48.0 cm")
    ax.set_xlabel("时间 t (s)", fontsize=9.8)
    ax.set_ylabel("环绕半径 (cm)", color="#1d4ed8", fontsize=9.8)
    ax.set_ylim(28, 72)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax_r = ax.twinx()
    l2 = ax_r.plot(to, tr_o["pitch_deg"], color="#10b981", linewidth=1.6, alpha=0.88, label="倒立摆俯仰角 Pitch (°)")
    l3 = ax_r.plot(to, np.asarray(tr_o["head_q_deg"]) * 0.2, color="#8b5cf6", linestyle=":", linewidth=1.5, label="0.2×云台内视角 head (°)")
    ax_r.axhline(8.0, color="#f59e0b", linestyle=":", linewidth=1.1)
    ax_r.axhline(-8.0, color="#f59e0b", linestyle=":", linewidth=1.1)
    ax_r.set_ylabel("俯仰角 Pitch (°) [安全限 ±8°]", color="#047857", fontsize=9.5)
    ax_r.set_ylim(-10.5, 10.5)
    lns = l1 + l2 + l3
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=7.8)
    ax.set_title(f"(d) 环绕检视半径收敛与平衡倾角包络 (Max |Pitch| = {orb['max_abs_pitch_deg']:.2f}° < 8°)", fontsize=10.3, fontweight="bold")

    # --- (e) Social Proxemics 双向速度指令 vx 与俯仰阻尼曲线 ---
    ax = axes[1, 1]
    l1 = ax.plot(tp, tr_p["vx_pct"], color="#7c3aed", linewidth=2.2, label="弹性纵向指令 vx (%) [正=跟进, 负=退让]")
    ax.axhline(0.0, color="#64748b", linestyle="-", linewidth=1.0)
    ax.set_xlabel("时间 t (s)", fontsize=9.8)
    ax.set_ylabel("速度指令 vx (%)", color="#6d28d9", fontsize=9.8)
    ax.set_ylim(-52, 52)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax_r = ax.twinx()
    l2 = ax_r.plot(tp, tr_p["pitch_deg"], color="#10b981", linewidth=1.7, label="机身俯仰角 Pitch (°)")
    l3 = ax_r.plot(tp, tr_p["gaze_err_deg"], color="#f97316", linestyle="--", linewidth=1.4, label="相机视线偏角 θ_img (°)")
    ax_r.set_ylabel("俯仰角 / 视线误差 (°)", color="#047857", fontsize=9.5)
    ax_r.set_ylim(-12, 12)
    lns = l1 + l2 + l3
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=7.8)
    ax.set_title(
        f"(e) 社交伴随进退平滑切换 (vx ∈ [{prx['min_backoff_vx_pct']:.1f}%, +{prx['max_approach_vx_pct']:.1f}%], Max |Pitch| = {prx['max_abs_pitch_deg']:.2f}°)",
        fontsize=10.3,
        fontweight="bold",
    )

    # --- (f) Figure-8 Patrol 谐振扫视与突发目标制动响应 ---
    ax = axes[1, 2]
    tf = np.asarray(tr_f["t"])
    ap_t = float(tr_f["appear_t_s"])
    ax.axvline(ap_t, color="#dc2626", linestyle="--", linewidth=1.8, label=f"t={ap_t:.1f}s 目标出现 (feat[38]=1)")
    ax.plot(tf, tr_f["vx_pct"], color="#2563eb", linewidth=2.1, label="前向巡航指令 vx (%) -> 瞬间归零")
    ax.plot(tf, tr_f["head_q_deg"], color="#8b5cf6", linewidth=1.9, label="云台正弦扫视与捕获角 q_head (°)")
    ax.plot(tf, tr_f["pitch_deg"], color="#10b981", linewidth=1.5, alpha=0.85, label="机身俯仰角 Pitch (°)")
    ax.set_xlabel("时间 t (s)", fontsize=9.8)
    ax.set_ylabel("控制量 (%) / 角度 (°)", fontsize=9.8)
    ax.set_ylim(-48, 48)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=7.8)
    ax.set_title(
        f"(f) 相位反馈自激扫视 (±{fg8['head_sweep_p2p_deg_mean']/2:.1f}°) 与突发目标捕获 (凝视 MAE = {fg8['captured_gaze_mae_deg']:.2f}°)",
        fontsize=10.3,
        fontweight="bold",
    )

    fig.suptitle(
        "RIG-Hover 端侧多技能具身强化学习动物园 (39维输入 -> TFLite Builtin 残差策略网络, 167Hz LQR 物理闭环评测)",
        fontsize=13.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return out_file


def train_and_export_skill_zoo(
    zoo_dir: Path | None = None,
    android_assets_dir: Path | None = None,
    fig_path: Path | None = None,
    bc_epochs: int = 38,
    ppo_iters: int = 10,
    n_expert_eps: int = 36,
) -> dict[str, Any]:
    """一键训练、评测、导出三大端侧 RL 具身技能并同步至 Android assets/models/。"""
    out_zoo = zoo_dir or (REPO_ROOT / "runs_end_side" / "skill_zoo")
    assets_dir = android_assets_dir or Path("G:/robot_app/android/app/src/main/assets/models")
    out_fig = fig_path or (REPO_ROOT / "docs" / "showcase" / "figs" / "fig7_multi_skill_zoo.png")

    out_zoo.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MultiSkillZoo] Training 3 embodied RL skills on device={device} ...", flush=True)

    t0 = time.perf_counter()
    skills_report: dict[str, Any] = {}

    for idx_s, (skill_key, spec) in enumerate(SKILL_SPECS.items()):
        print(f"\n=== [{idx_s + 1}/3] Training {spec['model_name']} ({spec['title_zh']}) ===", flush=True)
        torch.manual_seed(2026 + idx_s * 17)
        np.random.seed(2026 + idx_s * 17)

        feats, acts = collect_skill_expert_dataset(skill_key, n_episodes=n_expert_eps, seed=1200 + idx_s * 100)
        print(f"  Collected {len(feats)} expert state-action samples.", flush=True)

        ac = SkillActorCritic(hidden_dim=128, critic_dim=160)
        bc_losses = pretrain_skill_bc(skill_key, ac, feats, acts, epochs=bc_epochs, device=device)
        print(f"  [BC Warm-Start] loss {bc_losses[0]:.5f} -> {bc_losses[-1]:.5f}", flush=True)

        ppo_hist = train_skill_ppo(
            skill_key,
            ac,
            feats,
            acts,
            n_iters=ppo_iters,
            episodes_per_iter=6,
            device=device,
        )

        folded_policy = ac.export_folded_policy()
        sub_dir = out_zoo / spec["asset_stem"]
        bundle_res = export_policy_bundle(
            folded_policy,
            sub_dir,
            model_name=spec["model_name"],
            version=spec["version"],
            notes=spec["notes"],
        )

        # 同时在 runs_end_side/skill_zoo/ 根目录放置 *-bundle.zip 供直接检视与拉取
        root_zip = out_zoo / Path(bundle_res["zip_path"]).name
        shutil.copy2(bundle_res["zip_path"], root_zip)

        # 同步内置模型至 Android assets/models/<asset_stem>.tflite & .json
        dst_tfl = assets_dir / f"{spec['asset_stem']}.tflite"
        dst_jsn = assets_dir / f"{spec['asset_stem']}.json"
        shutil.copy2(bundle_res["tflite_path"], dst_tfl)
        shutil.copy2(bundle_res["json_path"], dst_jsn)

        parity_err = verify_tflite_numerical(folded_policy, dst_tfl, n_samples=64)
        assert parity_err < 1e-4, f"TFLite parity error {parity_err} >= 1e-4 for {skill_key}"

        if skill_key == "orbit_inspector":
            bench = benchmark_orbit_skill(folded_policy)
        elif skill_key == "social_proxemics":
            bench = benchmark_proxemics_skill(folded_policy)
        else:
            bench = benchmark_figure8_skill(folded_policy)

        print(
            f"  [Exported] sha256={bundle_res['sha256'][:16]}... | parity={parity_err:.2e} | "
            f"max_pitch={bench['max_abs_pitch_deg']:.2f} deg",
            flush=True,
        )

        skills_report[skill_key] = {
            "model_name": spec["model_name"],
            "asset_stem": spec["asset_stem"],
            "version": spec["version"],
            "feature_dim": int(folded_policy.feature_dim),
            "hidden_dim": int(folded_policy.hidden_dim),
            "action_dim": int(folded_policy.action_dim),
            "use_residual": bool(folded_policy.use_residual),
            "sha256": bundle_res["sha256"],
            "max_abs_err_pytorch_vs_tflite": parity_err,
            "tflite_path": bundle_res["tflite_path"],
            "json_path": bundle_res["json_path"],
            "zip_path": str(root_zip),
            "android_tflite_path": str(dst_tfl),
            "android_json_path": str(dst_jsn),
            "bc_loss_start": bc_losses[0],
            "bc_loss_end": bc_losses[-1],
            "ppo_history": ppo_hist,
            "benchmark": bench,
        }

    full_metrics = {
        "schemaVersion": 1,
        "generatedAt": "2026-11-18",
        "device": str(device),
        "elapsedSec": round(time.perf_counter() - t0, 2),
        "contract": {
            "inputShape": [1, 39],
            "outputShape": [1, 3],
            "architecture": "EndSidePolicyNet(feature_dim=39, hidden_dim=128, action_dim=3, use_residual=True)",
        },
        "skills": skills_report,
    }

    fig_out = plot_fig7_multi_skill_zoo(full_metrics, out_fig)
    full_metrics["figure_path"] = str(fig_out)

    # 保存指标 JSON 时剔除冗长采样点序列以便清爽查阅，同时保留核心指标
    compact_metrics = json.loads(json.dumps(full_metrics))
    for sk in compact_metrics["skills"].values():
        sk["benchmark"].pop("sample_trace", None)

    metrics_file = out_zoo / "skill_zoo_metrics.json"
    metrics_file.write_text(json.dumps(compact_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[MultiSkillZoo] Saved metrics -> {metrics_file} and figure -> {fig_out}", flush=True)
    return compact_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Train & Export RIG-Hover Multi-Skill Embodied RL Zoo")
    ap.add_argument("--bc-epochs", type=int, default=38)
    ap.add_argument("--ppo-iters", type=int, default=10)
    ap.add_argument("--expert-eps", type=int, default=36)
    args = ap.parse_args()
    train_and_export_skill_zoo(
        bc_epochs=args.bc_epochs,
        ppo_iters=args.ppo_iters,
        n_expert_eps=args.expert_eps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
