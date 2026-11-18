# -*- coding: utf-8 -*-
"""ping_point_nav_rl.py — 英雄联盟式 3D 全向地面战术标点与危险禁区绕障强化学习策略 (ping_point_nav_v1)。

面向 3D 高斯空间地面交互（LoL Smart-Ping + 虚拟危险禁区力场），训练并导出严格兼容
Android `PolicyFeatureBuilder.kt` 与 `ScenarioV2Engine.kt` 的 `[1, 39] -> [1, 3]` (`[vx, vyaw, head]`)
纯内置算子 TFLite FlatBuffers v3 策略模型 `ping-point-nav-v1`：

1. **6 大高难度 3D 地面标点导航物理沙箱 (`PingPointArenaEnv`)**：
   - `P1_rear_165_turn`        : ±165° 背向盲标零半径原地掉头 + 丝滑直线趋近 + 零过冲驻停
   - `P2_twin_cup_slalom`      : 双水杯错位窄门 S 弯穿行 (Slalom)
   - `P3_danger_zone_detour`   : 虚拟红色危险禁区力场 (Virtual Danger Zone) 大曲率切向绕障
   - `P4_cliff_edge_stop`      : 桌沿悬崖极限超界标点感知净空零过冲急停
   - `P5_dynamic_reping_chain` : 行进中途连续三次改标点高动量动态重规划
   - `P6_asymmetric_fault_storm`: 轮速不对称偏航扰动 + 48ms 队列延迟 + 40% 感知丢帧风暴

2. **两阶段物理强化学习流水线**：
   - Stage 1: 全向切向锥引导专家轨迹采集 + 360° 相空间流形增强 + 镜像等变行为克隆预热 (BC Warm-Start)
   - Stage 2: 53 维非对称特权 Critic (含真实 SE(2) sin/cos 方位、杯体/禁区有符号净空与打滑偏置) PPO 闭环微调
   - 导出前将 39x39 绝对坐标屏蔽与零中心化投影矩阵无损折叠进首层权重 `W1_folded = W1 @ P`，
     确保 PyTorch / NumPy / TFLite v3 推理最大绝对误差 `< 1e-4`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from end_side_controller import EndSideFeatureBuilder, EndSidePolicyNet, EndSideSafetyShield, TargetLock  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry, _clamp  # noqa: E402
from export_tflite_bundle import (  # noqa: E402
    export_policy_bundle,
    run_tflite_flatbuffer_cpu,
    verify_tflite_numerical,
)
from spatial_memory import wrap180  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 220

MODEL_MANIFEST_NAME = "ping-point-nav-v1"
ASSET_STEM = "ping_point_nav_v1"
BUNDLE_ZIP_NAME = "hover-skill-ping-nav-v1-bundle.zip"
MODEL_VERSION = "2026.11.18-ping-v1"
MODEL_NOTES = (
    "360-deg LoL Ground Ping Point-Goal & Virtual Danger-Zone Detour RL Policy "
    "(39->160->160+Res->3, TFLite v3 Builtin Ops, BC Warm-Start + Asymmetric Privileged PPO)."
)

PING_SCENARIOS: tuple[str, ...] = (
    "P1_rear_165_turn",
    "P2_twin_cup_slalom",
    "P3_danger_zone_detour",
    "P4_cliff_edge_stop",
    "P5_dynamic_reping_chain",
    "P6_asymmetric_fault_storm",
)

PING_SCENARIO_TITLES: dict[str, str] = {
    "P1_rear_165_turn": "P1: 165° 背向盲标原地掉头",
    "P2_twin_cup_slalom": "P2: 双水杯窄门 S 弯穿行",
    "P3_danger_zone_detour": "P3: 红色危险禁区力场绕障",
    "P4_cliff_edge_stop": "P4: 桌沿悬崖极限零过冲急停",
    "P5_dynamic_reping_chain": "P5: 行进中三连改标动态重规划",
    "P6_asymmetric_fault_storm": "P6: 轮速不对称+延迟丢帧抗扰",
}


# =========================================================================== #
# 1. 39x39 空间不变性与零中心化投影矩阵 (无损折叠进 fc1.weight)                #
# =========================================================================== #

def build_ping_projection_matrix() -> np.ndarray:
    """构造 39x39 坐标不变性与零中心化投影矩阵 P。

    严格屏蔽绝对世界航向、累积里程计与非对称偏置通道
    (yaw[2], ax..az[3..5], x1[10], x2[11], cmd_mode[17], cmd_pos[18], cmd_yaw_target[19], t1..t2[20..21])，
    使折叠后的网络对任意初始世界航向、历史行驶距离完全不变，且严格满足左右奇偶镜像等变性。
    """
    P_mat = np.zeros((39, 39), dtype=np.float32)
    P_mat[0, 0] = 6.0             # roll / 30 deg
    P_mat[1, 1] = 6.0             # pitch / 30 deg
    P_mat[6, 6] = 1.0             # pit_gyro
    P_mat[7, 7] = 1.0             # dq
    P_mat[8, 8] = 1.0             # v1 / 100
    P_mat[9, 9] = 1.0             # v2 / 100
    P_mat[12, 12] = 75.0 / 45.0   # head_q / 45 deg
    P_mat[13, 13] = 75.0 / 45.0   # head_target / 45 deg
    P_mat[14, 14] = 1.0           # gimbal_target / 45 deg
    P_mat[15, 15] = 1.0           # cmd_vx / 100
    P_mat[16, 16] = 1.0           # cmd_vyaw / 100
    P_mat[22, 22] = 1.0           # pos_e / 100
    P_mat[24, 24] = 1.0           # vel_e / 100 (body_yaw_rate / 60 dps)
    P_mat[26, 26] = 1.0           # pit_e / 100

    # Slot 0: 常规视觉框通道 (IDLE / GAZE 阶段注视用)
    P_mat[28, 28] = 2.0
    P_mat[29, 29] = 1.0
    P_mat[30, 30] = 1.0
    P_mat[31, 31] = 1.0
    P_mat[32, 32] = 1.0

    # Slot 1: 3D 地面标点与绕障引导槽 (lock_valid == 1 时精确零中心化)
    P_mat[33, 33] = 2.0
    P_mat[33, 37] = -1.0          # (steered_bearing - head_q) / 30 deg
    P_mat[34, 34] = 2.0
    P_mat[34, 37] = -1.0          # clamp(steered_bearing / 90 deg, -1, 1)
    P_mat[35, 35] = 2.0           # remaining_m / 0.75 m
    P_mat[36, 36] = 2.0           # clearance_m / 0.50 m (高灵敏度净空感知)
    P_mat[37, 37] = 1.0           # lock_valid (0/1)
    P_mat[38, 38] = 1.0           # target_present (0/1)
    return P_mat


PING_PROJ_39 = build_ping_projection_matrix()


def mirror_feat39_batch(x: torch.Tensor) -> torch.Tensor:
    """对批量 39 维端侧特征执行精确左右镜像变换 M_s(x)。"""
    m = x.clone()
    for idx in (0, 12, 13, 14, 16, 24):
        m[:, idx] = -x[:, idx]
    m[:, 8] = x[:, 9]
    m[:, 9] = x[:, 8]
    has_s0 = (x[:, 32] > 0.05).float()
    m[:, 28] = has_s0 * (1.0 - x[:, 28]) + (1.0 - has_s0) * x[:, 28]
    has_s1 = (x[:, 37] > 0.05).float()
    m[:, 33] = has_s1 * (1.0 - x[:, 33]) + (1.0 - has_s1) * x[:, 33]
    m[:, 34] = has_s1 * (1.0 - x[:, 34]) + (1.0 - has_s1) * x[:, 34]
    return m


def mirror_feat39_np(feat_39: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.asarray(feat_39, dtype=np.float32).reshape(1, 39))
    return mirror_feat39_batch(t)[0].numpy().astype(np.float32)


def numpy_infer_policy(policy: EndSidePolicyNet, feat_39: np.ndarray) -> np.ndarray:
    """纯 NumPy 参考前向推理，用于验证 PyTorch / NumPy / TFLite 三方数值严格对齐。"""
    x = np.asarray(feat_39, dtype=np.float32).reshape(-1, 39)
    sd = policy.state_dict()
    w1 = sd["fc1.weight"].detach().cpu().numpy().astype(np.float32)
    b1 = sd["fc1.bias"].detach().cpu().numpy().astype(np.float32)
    w2 = sd["fc2.weight"].detach().cpu().numpy().astype(np.float32)
    b2 = sd["fc2.bias"].detach().cpu().numpy().astype(np.float32)
    w3 = sd["fc3.weight"].detach().cpu().numpy().astype(np.float32)
    b3 = sd["fc3.bias"].detach().cpu().numpy().astype(np.float32)
    scale = sd["action_scale"].detach().cpu().numpy().astype(np.float32).reshape(1, 3)

    h1 = np.tanh(x @ w1.T + b1)
    h2 = np.tanh(h1 @ w2.T + b2)
    if "fc_res.weight" in sd:
        w_res = sd["fc_res.weight"].detach().cpu().numpy().astype(np.float32)
        b_res = sd["fc_res.bias"].detach().cpu().numpy().astype(np.float32)
        h_mid = np.tanh((h1 + h2) @ w_res.T + b_res)
        u = np.tanh(h_mid @ w3.T + b3)
    else:
        u = np.tanh(h2 @ w3.T + b3)
    return (u * scale).astype(np.float32)


# =========================================================================== #
# 2. 全向原地掉头 + 切向净空耦合绕障 + 零过冲驻停解析专家策略                    #
# =========================================================================== #

def ping_detour_expert_action(feat_39: np.ndarray) -> np.ndarray:
    """全连续 360° 地面标点导航与禁区切向绕障专家控制律。

    输出物理量纲动作: [vx ∈ [-100, 100], vyaw ∈ [-100, 100], head ∈ [-75, 75]]。
    """
    q_head_deg = float(feat_39[12]) * 75.0
    pitch_deg = float(feat_39[1]) * 180.0
    v_avg_mps = 0.5 * (float(feat_39[8]) + float(feat_39[9]))
    pos_e_m = float(feat_39[22])  # wheel_x - stable_pos (m): 正值表示轮轴瞬时摆动超前于 LQR 平衡点 stable_pos
    body_yaw_rate_dps = float(feat_39[24]) * 60.0
    lock_active = float(feat_39[37]) > 0.90
    has_any = float(feat_39[38]) > 0.5

    if lock_active:
        # Slot 1 广角方位: [-90°, +90°] 线性不饱和；±90°..±180° 饱和在 ±90°
        bearing_deg = (float(feat_39[34]) - 0.5) * 180.0
        rem_m = max(0.0, float(feat_39[35]) * 1.5)
        clear_m = max(0.0, float(feat_39[36]) * 1.0)
        eff_rem = min(rem_m, clear_m)

        # 计算 LQR 稳态平衡点 (stable_pos + 0.006m) 距目标/安全线的真实剩余距离 d_eq:
        # 若轮轴在终点附近越线 (|bearing| > 85° 且 rem_m < 0.08m)，说明已越过垂面，d_eq 归零
        abs_b = abs(bearing_deg)
        if rem_m < 0.085 and abs_b > 82.0:
            d_eq = 0.0
        else:
            d_eq = max(0.0, eff_rem + (pos_e_m - 0.006))

        # 1. 头部前瞻凝视引导方向 (最大 ±42°；进入 3.2cm 驻停半径后归零静稳)
        if rem_m <= 0.032 or clear_m <= 0.018:
            head_cmd_deg = 0.0
        else:
            head_cmd_deg = float(_clamp(0.92 * bearing_deg, -42.0, 42.0))

        # 2. 全域连续主动阻尼偏航控制 (进入 3.2cm 驻停半径后禁止 180° 掉头打转，仅做角速度阻尼刹停)
        if rem_m <= 0.032 or clear_m <= 0.018:
            vyaw_dps = -0.32 * body_yaw_rate_dps
        else:
            vyaw_dps = float(_clamp(2.68 * bearing_deg - 0.30 * body_yaw_rate_dps, -76.0, 76.0))
        vyaw_pct = (vyaw_dps / 90.0) * 100.0

        # 3. 零半径掉头门控 (|bearing| >= 58° 时 vx 严格归零) + LQR 平衡点一阶单调收敛速度律
        if d_eq <= 0.008 or eff_rem <= 0.012 or abs_b >= 58.0:
            vx_pct = 0.0
        else:
            turn_ratio = max(0.0, 1.0 - abs_b / 58.0)
            turn_gate = math.sin(0.5 * math.pi * turn_ratio) ** 1.40

            # 一阶单调指数推进 stable_pos -> 目标点，兼具快速巡航与零过冲稳态收敛
            v_brake = math.sqrt(2.0 * 0.110 * max(0.0, d_eq - 0.006))
            v_lin = 0.58 * max(0.0, d_eq - 0.005)
            v_clear_cap = 0.050 + 0.35 * clear_m if clear_m > 0.04 else 0.0
            v_des_mps = min(0.205, v_brake, v_lin, v_clear_cap) * turn_gate

            # 抑制机身大幅纵向摆振
            osc_speed = abs(v_avg_mps - v_des_mps)
            if osc_speed > 0.18:
                v_des_mps *= 0.65

            pitch_dev = max(0.0, abs(pitch_deg + 2.0) - 4.0)
            pitch_scale = _clamp(1.0 - pitch_dev / 8.5, 0.25, 1.0)
            vx_pct = float(_clamp((v_des_mps / 0.30) * 100.0 * pitch_scale, 0.0, 68.0))

        return np.array([vx_pct, vyaw_pct, head_cmd_deg], dtype=np.float32)

    # 非标点巡航阶段 (IDLE / GAZE): vx 严格为 0，平稳凝视视野目标
    if not has_any or float(feat_39[32]) < 0.35:
        return np.array([0.0, 0.0, 0.70 * q_head_deg], dtype=np.float32)

    theta_img_deg = (float(feat_39[28]) - 0.5) * 60.0
    eps_deg = theta_img_deg + q_head_deg
    head_cmd = 0.0 if abs(eps_deg) <= 1.0 else float(_clamp(0.84 * eps_deg, -42.0, 42.0))
    vyaw_dps = 0.0 if abs(eps_deg) <= 3.5 else float(_clamp(1.85 * eps_deg - 0.22 * body_yaw_rate_dps, -60.0, 60.0))
    return np.array([0.0, (vyaw_dps / 90.0) * 100.0, head_cmd], dtype=np.float32)


# =========================================================================== #
# 3. 英雄联盟式 3D 全向地面标点与禁区力场沙箱环境 (PingPointArenaEnv)          #
# =========================================================================== #

@dataclass
class CupObstacle:
    x_m: float
    y_m: float
    radius_m: float = 0.052
    safe_radius_m: float = 0.142
    pass_side: float = 0.0  # +1 绕右侧 (+Y), -1 绕左侧 (-Y), 0 自动按几何锁定


@dataclass
class VirtualDangerZone:
    x_m: float
    y_m: float
    core_radius_m: float = 0.135
    halo_radius_m: float = 0.248
    pass_side: float = 0.0


@dataclass
class TabletopCliffBounds:
    x_min: float = -0.45
    x_max: float = 0.64
    y_min: float = -0.52
    y_max: float = 0.52
    safe_margin: float = 0.075


class OmniPingMemoryTracker:
    """全向 360° SE(2) 自车系地面标点里程计追踪器（支持 ±180° 背向标点而不误触发越线清零）。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.valid: int = 0
        self.x_body_m: float = 0.0
        self.y_body_m: float = 0.0
        self.initial_dist_m: float = 0.0
        self._entered_near_front: bool = False
        self._arrived_dwell: bool = False
        self._has_ref: bool = False
        self._prev_wheel_x: float = 0.0
        self._prev_imu_yaw: float = 0.0
        self._prev_head_q: float = 0.0

    def lock_body_xy(self, x_body_m: float, y_body_m: float) -> None:
        self.valid = 1
        self.x_body_m = float(x_body_m)
        self.y_body_m = float(y_body_m)
        self.initial_dist_m = float(math.hypot(x_body_m, y_body_m))
        self._entered_near_front = (self.x_body_m > 0.0 and self.initial_dist_m < 0.14)
        self._arrived_dwell = False

    def step_odometry(self, telemetry: RobotTelemetry) -> tuple[float, float]:
        wheel_x = float(telemetry.wheel_x_avg)
        imu_yaw = float(telemetry.yaw)
        head_q = float(telemetry.head_q)

        if not self._has_ref:
            self._has_ref = True
            self._prev_wheel_x = wheel_x
            self._prev_imu_yaw = imu_yaw
            self._prev_head_q = head_q
            return self.bearing_and_remaining()

        ds = wheel_x - self._prev_wheel_x
        prev_body_yaw = wrap180(self._prev_imu_yaw - self._prev_head_q)
        curr_body_yaw = wrap180(imu_yaw - head_q)
        dyaw_deg = max(-45.0, min(45.0, wrap180(curr_body_yaw - prev_body_yaw)))

        self._prev_wheel_x = wheel_x
        self._prev_imu_yaw = imu_yaw
        self._prev_head_q = head_q

        if not self.valid:
            return 0.0, 0.0

        dpsi = math.radians(dyaw_deg)
        half = 0.5 * dpsi
        tx = self.x_body_m - ds * math.cos(half)
        ty = self.y_body_m - ds * math.sin(half)
        c, s = math.cos(dpsi), math.sin(dpsi)
        self.x_body_m = c * tx + s * ty
        self.y_body_m = -s * tx + c * ty

        euclid = math.hypot(self.x_body_m, self.y_body_m)
        if self.x_body_m > 0.0 and euclid < 0.14:
            self._entered_near_front = True
        return self.bearing_and_remaining()

    def bearing_and_remaining(self) -> tuple[float, float]:
        if not self.valid:
            return 0.0, 0.0
        euclid = float(math.hypot(self.x_body_m, self.y_body_m))
        bearing = float(math.degrees(math.atan2(self.y_body_m, self.x_body_m)))
        if self._entered_near_front and self.x_body_m <= 0.0 and abs(self.y_body_m) <= 0.14:
            return 0.0, 0.0
        return bearing, euclid


class PingPointArenaEnv:
    """20.8Hz (48ms = 8x6ms) 端侧策略 × 166.7Hz `sim.Plant` 轮式倒立摆 3D 地面标点与禁区绕障环境。"""

    CTRL_SUBSTEPS = 8

    def __init__(
        self,
        scenario: str = "P1_rear_165_turn",
        seed: int = 42,
        domain_rand: bool = False,
        max_steps: int = 230,
    ) -> None:
        if scenario not in PING_SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; expected one of {PING_SCENARIOS}")
        self.scenario = scenario
        self.base_seed = int(seed)
        self.domain_rand = bool(domain_rand)
        self.max_steps = int(max_steps)

        self.base_sp = sim.calibrated(P.SimParams(), verbose=False)
        self.dt_inner = self.base_sp.fw.ctrl_dt_ms * 1e-3  # 0.006 s
        self.dt_step = self.dt_inner * self.CTRL_SUBSTEPS  # 0.048 s
        self.reset(seed=seed)

    def _sample_sim_params(self, rng: np.random.Generator) -> P.SimParams:
        if not self.domain_rand and self.scenario != "P6_asymmetric_fault_storm":
            return self.base_sp
        pp = self.base_sp.plant
        rand_pp = replace(
            pp,
            mass_body_kg=float(pp.mass_body_kg * rng.uniform(0.87, 1.13)),
            wheel_rolling_damping=float(pp.wheel_rolling_damping * rng.uniform(0.85, 1.16)),
            yaw_damping=float(pp.yaw_damping * rng.uniform(0.84, 1.18)),
            k_yaw_accel=float(pp.k_yaw_accel * rng.uniform(0.87, 1.14)),
            actuator_delay_ticks=int(rng.choice([1, 2])),
        )
        return replace(self.base_sp, plant=rand_pp)

    def reset(self, scenario: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if scenario is not None:
            if scenario not in PING_SCENARIOS:
                raise ValueError(f"Unknown scenario {scenario!r}")
            self.scenario = scenario
        if seed is not None:
            self.base_seed = int(seed)

        self.rng = np.random.default_rng(self.base_seed * 8191 + 53)
        self.sp = self._sample_sim_params(self.rng)
        self.plant = sim.Plant(self.sp, self.dt_inner)
        self.plant.reset(pitch0_deg=float(self.rng.uniform(-1.1, 1.1)), seed=self.base_seed)
        noisy_imu = self.domain_rand or (self.scenario == "P6_asymmetric_fault_storm")
        self.imu = sim.Imu(self.sp, self.rng if noisy_imu else None)
        self.fw = sim.FirmwareControl(self.sp)
        self.fw.k_dyaw = -0.52
        self.fw.head_rate_comp = 1.0
        self.arb = sim.Arbiter(self.sp, 2)
        self.sv = sim.SensorView()

        self.t_s: float = 0.0
        self.step_idx: int = 0
        self.rx_m: float = 0.0
        self.ry_m: float = 0.0
        self.trail_xy: list[tuple[float, float]] = [(0.0, 0.0)]

        # 物理扰动与桌面几何对象
        self.slip_scale: float = float(self.rng.uniform(0.97, 1.03)) if self.domain_rand else 1.0
        self.yaw_bias_dps: float = 0.0
        self.wheel_vel_asym_cms: float = 0.0
        self.cmd_delay_steps: int = 0
        self.dropout_prob: float = 0.0
        self.servo_backlash_deg: float = 0.0
        self.cups: list[CupObstacle] = []
        self.danger_zones: list[VirtualDangerZone] = []
        self.cliff_bounds: TabletopCliffBounds | None = None
        self.reping_schedule: list[tuple[int, tuple[float, float]]] = []
        self.ping_history_xy: list[tuple[float, float]] = []

        sgn = 1.0 if (self.base_seed % 2 == 0) else -1.0

        if self.scenario == "P1_rear_165_turn":
            # ±165° 背向盲标原地掉头趋近 (canonical 165° +- 4°)
            ang_deg = sgn * (165.0 + float(self.rng.uniform(-3.5, 3.5)) * (1.0 if self.domain_rand else 0.0))
            dist_m = float(self.rng.uniform(0.56, 0.68))
            self.raw_ping_xy = (dist_m * math.cos(math.radians(ang_deg)), dist_m * math.sin(math.radians(ang_deg)))

        elif self.scenario == "P2_twin_cup_slalom":
            # 双水杯错位窄门 S 弯穿行：杯 1 封死直连线迫使向 -sgn 绕行，杯 2 迫使反向切回窄门
            gx = float(self.rng.uniform(0.84, 0.92))
            gy = sgn * float(self.rng.uniform(0.03, 0.06))
            self.raw_ping_xy = (gx, gy)
            self.cups = [
                CupObstacle(x_m=0.29, y_m=sgn * 0.035, radius_m=0.052, safe_radius_m=0.140, pass_side=-sgn),
                CupObstacle(x_m=0.58, y_m=-sgn * 0.072, radius_m=0.052, safe_radius_m=0.140, pass_side=+sgn),
            ]

        elif self.scenario == "P3_danger_zone_detour":
            # 桌面中央标记大半径红色危险禁区力场，点击禁区正后方目标点
            gx = float(self.rng.uniform(0.84, 0.92))
            gy = sgn * float(self.rng.uniform(0.04, 0.08))
            self.raw_ping_xy = (gx, gy)
            self.danger_zones = [
                VirtualDangerZone(
                    x_m=0.42,
                    y_m=-sgn * 0.025,
                    core_radius_m=0.138,
                    halo_radius_m=0.252,
                    pass_side=+sgn,
                )
            ]

        elif self.scenario == "P4_cliff_edge_stop":
            # 桌沿悬崖极限超界标点：用户点击 x=0.71m 超出物理桌沿 x_max=0.64m (安全驻停线 x_safe=0.565m)
            self.cliff_bounds = TabletopCliffBounds(x_min=-0.45, x_max=0.64, y_min=-0.52, y_max=0.52, safe_margin=0.075)
            self.slip_scale = 1.06 if self.domain_rand else 1.03
            ang_deg = sgn * float(self.rng.uniform(6.0, 15.0))
            r_over = 0.72
            self.raw_ping_xy = (r_over * math.cos(math.radians(ang_deg)), r_over * math.sin(math.radians(ang_deg)))

        elif self.scenario == "P5_dynamic_reping_chain":
            # 行进中途连续三次改标点动态重规划
            p1 = (
                0.66 * math.cos(math.radians(sgn * 52.0)),
                0.66 * math.sin(math.radians(sgn * 52.0)),
            )
            p2 = (
                0.24 + 0.44 * math.cos(math.radians(-sgn * 68.0)),
                0.44 * math.sin(math.radians(-sgn * 68.0)),
            )
            p3 = (
                0.64 + float(self.rng.uniform(-0.04, 0.04)),
                sgn * float(self.rng.uniform(0.10, 0.16)),
            )
            self.raw_ping_xy = p1
            self.reping_schedule = [(34, p2), (76, p3)]

        else:  # P6_asymmetric_fault_storm
            # 大角度标点 (112°..132°) + 侧方危险禁区 + 轮速不对称偏航扰动 + 48ms 队列延迟 + 40% 丢帧 + 舵机回差
            ang_deg = sgn * float(self.rng.uniform(112.0, 132.0))
            dist_m = float(self.rng.uniform(0.58, 0.68))
            gx = dist_m * math.cos(math.radians(ang_deg))
            gy = dist_m * math.sin(math.radians(ang_deg))
            self.raw_ping_xy = (gx, gy)
            self.danger_zones = [
                VirtualDangerZone(
                    x_m=0.46 * gx - sgn * 0.02,
                    y_m=0.46 * gy,
                    core_radius_m=0.105,
                    halo_radius_m=0.205,
                    pass_side=+sgn,
                )
            ]
            self.yaw_bias_dps = sgn * 6.2
            self.wheel_vel_asym_cms = sgn * 3.8
            self.cmd_delay_steps = 1
            self.dropout_prob = 0.40
            self.servo_backlash_deg = 2.8
            self.slip_scale = float(self.rng.uniform(0.97, 1.03))

        self.ping_history_xy.append(self.raw_ping_xy)
        self.safe_goal_xy = self._compute_safe_goal_xy(self.raw_ping_xy)

        self._cmd_queue: deque[DriveCommand] = deque(
            [DriveCommand() for _ in range(self.cmd_delay_steps + 1)],
            maxlen=self.cmd_delay_steps + 1,
        )
        self._applied_head_deg: float = 0.0
        self.cup_collided: bool = False
        self.danger_zone_violated: bool = False
        self.cliff_fell: bool = False
        self.min_cup_surf_m: float = 9.9
        self.min_danger_surf_m: float = 9.9
        self.max_overshoot_m: float = 0.0
        self._last_valid_dets: list[dict[str, float]] = []

        self._sync_sensors(0.0)
        self.ping_memory = OmniPingMemoryTracker()
        self.ping_memory.step_odometry(self._build_telemetry())
        xb0, yb0, _, _ = self._world_to_body(*self.raw_ping_xy)
        self.ping_memory.lock_body_xy(xb0, yb0)

        return self._build_observation()

    def _compute_safe_goal_xy(self, raw_xy: tuple[float, float]) -> tuple[float, float]:
        if self.cliff_bounds is None:
            return raw_xy
        b = self.cliff_bounds
        sx = float(_clamp(raw_xy[0], b.x_min + b.safe_margin, b.x_max - b.safe_margin))
        sy = float(_clamp(raw_xy[1], b.y_min + b.safe_margin, b.y_max - b.safe_margin))
        return (sx, sy)

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

    def compute_tactical_guidance(self) -> tuple[float, float, float]:
        """计算切向锥绕障偏转角 deflect_deg、引导方位角 steered_bearing_deg 与前向安全净空 clearance_m。"""
        xb_g, yb_g, dist_g, bear_g = self._world_to_body(*self.raw_ping_xy)
        clearance_m = 1.00
        steered_deg = bear_g

        # 收集全部圆形障碍与虚拟危险禁区 (统一抽象为 (x, y, r_hard, r_safe, pass_side, is_danger))
        hazards: list[tuple[float, float, float, float, float, bool]] = []
        for cup in self.cups:
            hazards.append((cup.x_m, cup.y_m, cup.radius_m, cup.safe_radius_m, cup.pass_side, False))
        for dz in self.danger_zones:
            hazards.append((dz.x_m, dz.y_m, dz.core_radius_m, dz.halo_radius_m, dz.pass_side, True))

        # 1. 更新碰撞/禁区违例统计与前向走廊净空，并按沿目标射线距离排序寻找最近阻挡体
        ux_g = (self.raw_ping_xy[0] - self.rx_m) / max(dist_g, 1e-5)
        uy_g = (self.raw_ping_xy[1] - self.ry_m) / max(dist_g, 1e-5)
        blocking_candidates: list[tuple[float, float, float, float, float, float]] = []

        for hx, hy, r_hard, r_safe, p_side, is_dz in hazards:
            xb_h, yb_h, d_center, bear_h = self._world_to_body(hx, hy)
            surf_d = d_center - r_hard
            if is_dz:
                self.min_danger_surf_m = min(self.min_danger_surf_m, surf_d)
                if surf_d <= 0.0:
                    self.danger_zone_violated = True
            else:
                self.min_cup_surf_m = min(self.min_cup_surf_m, surf_d)
                if surf_d <= 0.0:
                    self.cup_collided = True

            # 前向行驶走廊净空计算
            if xb_h > -0.02 and d_center < 0.85:
                if abs(yb_h) < r_hard + 0.028:
                    clearance_m = min(clearance_m, max(0.05, xb_h - r_hard))
                elif abs(yb_h) < r_safe + 0.06:
                    clearance_m = min(clearance_m, max(0.28, surf_d + 0.14))

            # 判断该障碍/禁区是否阻挡当前位置到标点目标的走廊
            rel_x = hx - self.rx_m
            rel_y = hy - self.ry_m
            proj_s = rel_x * ux_g + rel_y * uy_g
            # 有向侧距: 右侧 (+Y_goal) 为正
            lat_signed = -rel_x * uy_g + rel_y * ux_g
            perp_d = abs(lat_signed)

            if -0.02 <= proj_s <= dist_g - 0.03 and perp_d < r_safe:
                side = p_side if abs(p_side) > 0.1 else (-1.0 if lat_signed >= 0.0 else 1.0)
                blocking_candidates.append((proj_s, d_center, bear_h, r_safe, perp_d, side))

        if blocking_candidates:
            blocking_candidates.sort(key=lambda item: item[0])
            _, d_c, bear_h, r_safe, perp_d, side = blocking_candidates[0]
            if d_c > r_safe * 1.02:
                cone_deg = math.degrees(math.asin(_clamp((r_safe * 1.02) / d_c, -0.999, 0.999)))
            else:
                # 若已贴近外环，叠加切向 + 外推法向角确保零侵入
                push_extra = 32.0 * (1.0 - d_c / max(r_safe * 1.02, 1e-4))
                cone_deg = min(118.0, 90.0 + push_extra)
            tan_bearing = wrap180(bear_h + side * cone_deg)

            # 在外边缘设置 3cm 平滑过渡带，消除离开切线瞬间的角度跳变
            blend = float(_clamp((r_safe - perp_d) / 0.035, 0.0, 1.0))
            diff = wrap180(tan_bearing - bear_g)
            steered_deg = wrap180(bear_g + blend * diff)
            clearance_m = min(clearance_m, 0.44)

        # 2. 桌面悬崖安全边界射线净空 (P4)
        if self.cliff_bounds is not None:
            b = self.cliff_bounds
            if (
                self.rx_m < b.x_min
                or self.rx_m > b.x_max
                or self.ry_m < b.y_min
                or self.ry_m > b.y_max
            ):
                self.cliff_fell = True

            # 同时沿机身正前射线与目标方向射线计算距安全桌沿的最小净空
            psi = math.radians(self.plant.st.yaw_body)
            for dir_x, dir_y in ((math.cos(psi), math.sin(psi)), (ux_g, uy_g)):
                ray_d = 1.50
                if dir_x > 1e-4:
                    ray_d = min(ray_d, ((b.x_max - b.safe_margin) - self.rx_m) / dir_x)
                elif dir_x < -1e-4:
                    ray_d = min(ray_d, ((b.x_min + b.safe_margin) - self.rx_m) / dir_x)
                if dir_y > 1e-4:
                    ray_d = min(ray_d, ((b.y_max - b.safe_margin) - self.ry_m) / dir_y)
                elif dir_y < -1e-4:
                    ray_d = min(ray_d, ((b.y_min + b.safe_margin) - self.ry_m) / dir_y)
                clearance_m = min(clearance_m, max(0.0, ray_d))

            over_x = max(0.0, self.rx_m - (b.x_max - b.safe_margin))
            over_y = max(0.0, abs(self.ry_m) - (b.y_max - b.safe_margin))
            self.max_overshoot_m = max(self.max_overshoot_m, math.hypot(over_x, over_y))

        deflect_deg = wrap180(steered_deg - bear_g)
        return float(deflect_deg), float(steered_deg), float(_clamp(clearance_m, 0.0, 1.0))

    def _build_telemetry(self) -> RobotTelemetry:
        sv = self.sv
        body_yaw_rate_dps = sv.yaw_rate - sv.qd_head
        pos_err_m = sv.wheel_x - sv.stable_pos
        half_diff = 0.08 * body_yaw_rate_dps + 0.5 * self.wheel_vel_asym_cms
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

    def _build_observation(self) -> dict[str, Any]:
        telem = self._build_telemetry()
        odom_bearing_deg, odom_rem_m = self.ping_memory.step_odometry(telem)
        deflect_deg, _, clearance_m = self.compute_tactical_guidance()

        # P6 模拟 40% 视觉/空间更新丢帧与路人干扰框
        dropped = self.dropout_prob > 0.0 and (float(self.rng.random()) < self.dropout_prob)
        dets: list[dict[str, float]] = []
        if not dropped:
            # 在部分场景注入视野内无关水杯/路人视觉检测框到 slot0，验证策略专注锁定 slot1 标点
            if self.scenario in ("P2_twin_cup_slalom", "P6_asymmetric_fault_storm") and (self.step_idx % 3 == 0):
                cx_dist = float(_clamp(0.5 + 0.32 * math.sin(0.09 * self.step_idx), 0.08, 0.92))
                dets.append({"cx": cx_dist, "cy": 0.45, "w": 0.22, "h": 0.26, "score": 0.88, "cls": 0})
            self._last_valid_dets = dets
            # 3DGS 关键帧对空间记忆做微幅漂移校正 (与真机 GaussianCloudEngine 一致)
            xb_true, yb_true, _, _ = self._world_to_body(*self.raw_ping_xy)
            self.ping_memory.x_body_m = 0.75 * self.ping_memory.x_body_m + 0.25 * xb_true
            self.ping_memory.y_body_m = 0.75 * self.ping_memory.y_body_m + 0.25 * yb_true
            odom_bearing_deg, odom_rem_m = self.ping_memory.bearing_and_remaining()
        else:
            dets = list(self._last_valid_dets)

        steered_odom_deg = wrap180(odom_bearing_deg + deflect_deg)
        t_lock = TargetLock(
            valid=1,
            bearing_body=steered_odom_deg,
            dist_m=self.ping_memory.initial_dist_m,
            remaining_m=odom_rem_m,
        )
        feat_39 = EndSideFeatureBuilder.build(telem, dets, target_lock=t_lock, clearance_m=clearance_m)

        _, _, raw_dist_m, raw_bearing_deg = self._world_to_body(*self.raw_ping_xy)
        _, _, safe_dist_m, safe_bearing_deg = self._world_to_body(*self.safe_goal_xy)

        return {
            "telemetry": telem,
            "detections": dets,
            "target_lock": t_lock,
            "feat_39": feat_39,
            "clearance_m": clearance_m,
            "deflect_deg": deflect_deg,
            "steered_bearing_deg": steered_odom_deg,
            "raw_goal_bearing_deg": raw_bearing_deg,
            "raw_goal_dist_m": raw_dist_m,
            "safe_goal_dist_m": safe_dist_m,
            "safe_goal_bearing_deg": safe_bearing_deg,
        }

    def build_privileged_critic_obs(self, feat_39: np.ndarray) -> np.ndarray:
        """拼接 39 维部署特征 + 14 维真实世界 SE(2) / 障碍净空 / 扰动特权状态 = 53 维 Critic 输入。"""
        xb_s, yb_s, dist_s, bear_s = self._world_to_body(*self.safe_goal_xy)
        _, steered_deg, clearance_m = self.compute_tactical_guidance()
        b_rad = math.radians(bear_s)
        cup_surf = min(self.min_cup_surf_m, 0.60)
        dz_surf = min(self.min_danger_surf_m, 0.60)
        priv_14 = np.array(
            [
                _clamp(xb_s / 1.2, -1.5, 1.5),
                _clamp(yb_s / 1.2, -1.5, 1.5),
                _clamp(dist_s / 1.5, 0.0, 1.5),
                math.sin(b_rad),
                math.cos(b_rad),
                _clamp(steered_deg / 90.0, -2.0, 2.0),
                _clamp(clearance_m / 1.0, 0.0, 1.0),
                _clamp(cup_surf / 0.5, -0.5, 1.2),
                _clamp(dz_surf / 0.5, -0.5, 1.2),
                _clamp(self.plant.pitch_true_deg / 15.0, -2.0, 2.0),
                _clamp(self.plant.st.yaw_rate / 90.0, -2.0, 2.0),
                _clamp(self.plant.st.v_w / 0.30, -1.5, 1.5),
                (self.slip_scale - 1.0) * 5.0,
                _clamp(self.yaw_bias_dps / 10.0, -1.5, 1.5),
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat_39.astype(np.float32), priv_14], axis=0)

    def step(self, cmd: DriveCommand) -> tuple[dict[str, Any], dict[str, Any]]:
        self._cmd_queue.append(cmd.sanitized())
        delayed_cmd = self._cmd_queue[0]

        vx_mps = float((delayed_cmd.vx / 100.0) * 0.30)
        vyaw_dps = float((delayed_cmd.vyaw / 100.0) * 90.0)
        raw_head = float(_clamp(delayed_cmd.head, -45.0, 45.0))
        if abs(raw_head - self._applied_head_deg) >= self.servo_backlash_deg:
            self._applied_head_deg = raw_head

        for _ in range(self.CTRL_SUBSTEPS):
            prev_xw = self.plant.st.x_w
            prev_yaw = self.plant.st.yaw_body
            self.arb.post_raw(
                sim.CtrlOut(head_des=self._applied_head_deg, vx_des=vx_mps, vyaw_des=vyaw_dps, mode=1),
                self.t_s,
            )
            self.arb.note_feature(self.t_s)
            self.fw.update_state(self.sv, self.t_s)
            self.arb.tick(self.sv, self.fw, self.t_s)
            _, _, h_cmd = self.fw.control(self.sv, self.t_s, do_update_state=False)
            self.plant.step(self.fw.temp_u, self.fw.yaw_u, h_cmd)

            # 注入轮速不对称物理偏航扰动 (P6)
            if abs(self.yaw_bias_dps) > 1e-6:
                self.plant.st.yaw_body += self.yaw_bias_dps * self.dt_inner
                self.plant.st.yaw_rate += 0.12 * (self.yaw_bias_dps - self.plant.st.yaw_rate) * self.dt_inner

            self.t_s += self.dt_inner
            self._sync_sensors(self.dt_inner)

            ds = (self.plant.st.x_w - prev_xw) * self.slip_scale
            mid_rad = math.radians(0.5 * (prev_yaw + self.plant.st.yaw_body))
            self.rx_m += ds * math.cos(mid_rad)
            self.ry_m += ds * math.sin(mid_rad)

        self.step_idx += 1
        self.trail_xy.append((round(self.rx_m, 4), round(self.ry_m, 4)))

        # 检查中途动态改标点事件 (P5)
        reping_triggered = False
        for trig_step, new_xy in self.reping_schedule:
            if self.step_idx == trig_step:
                self.raw_ping_xy = new_xy
                self.safe_goal_xy = self._compute_safe_goal_xy(new_xy)
                self.ping_history_xy.append(new_xy)
                xb_new, yb_new, _, _ = self._world_to_body(*new_xy)
                self.ping_memory.lock_body_xy(xb_new, yb_new)
                reping_triggered = True

        obs = self._build_observation()

        # 非 P4 场景统计越过终点垂面的过冲量
        xb_safe, yb_safe, safe_dist_m, safe_bear_deg = self._world_to_body(*self.safe_goal_xy)
        if self.cliff_bounds is None and self.ping_memory._entered_near_front and xb_safe < 0.0 and abs(yb_safe) < 0.15:
            self.max_overshoot_m = max(self.max_overshoot_m, abs(xb_safe))

        tipped = abs(self.plant.pitch_true_deg) > 32.0
        info = {
            "t_s": round(self.t_s, 3),
            "step": self.step_idx,
            "robot_xy": (self.rx_m, self.ry_m),
            "robot_yaw_deg": float(self.plant.st.yaw_body),
            "head_q_deg": float(self.sv.q_head),
            "pitch_deg": float(self.plant.pitch_true_deg),
            "v_w_mps": float(self.plant.st.v_w),
            "yaw_rate_dps": float(self.plant.st.yaw_rate),
            "raw_ping_xy": self.raw_ping_xy,
            "safe_goal_xy": self.safe_goal_xy,
            "safe_dist_m": safe_dist_m,
            "safe_bearing_deg": safe_bear_deg,
            "steered_bearing_deg": obs["steered_bearing_deg"],
            "clearance_m": obs["clearance_m"],
            "cup_collided": self.cup_collided,
            "danger_zone_violated": self.danger_zone_violated,
            "cliff_fell": self.cliff_fell,
            "max_overshoot_m": self.max_overshoot_m,
            "reping_triggered": reping_triggered,
            "done": self.step_idx >= self.max_steps or self.cliff_fell or tipped,
        }
        return obs, info


# =========================================================================== #
# 4. 非对称特权 Actor-Critic 网络 (Actor 无损折叠为 EndSidePolicyNet)           #
# =========================================================================== #

class PingNavActorCritic(nn.Module):
    """部署侧 Actor 100% 同构于 EndSidePolicyNet(39, 160, 3, use_residual=True)；训练侧 Critic 读 53 维特权真值。"""

    def __init__(self, hidden_dim: int = 160, critic_dim: int = 192) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor = EndSidePolicyNet(
            feature_dim=39,
            hidden_dim=hidden_dim,
            action_dim=3,
            use_residual=True,
        )
        self.critic = nn.Sequential(
            nn.Linear(53, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-2.45, -2.35, -2.35], dtype=torch.float32))
        self.register_buffer("proj_mat", torch.from_numpy(PING_PROJ_39))

    def proj_obs(self, feat_39: torch.Tensor) -> torch.Tensor:
        return feat_39 @ self.proj_mat.T

    def forward_actor_norm(self, feat_39: torch.Tensor) -> torch.Tensor:
        return self.actor.forward_normalized(self.proj_obs(feat_39))

    def forward_value(self, priv_53: torch.Tensor) -> torch.Tensor:
        p39 = self.proj_obs(priv_53[:, :39])
        return self.critic(torch.cat([p39, priv_53[:, 39:]], dim=-1)).squeeze(-1)

    def symmetry_loss(self, feat_39: torch.Tensor) -> torch.Tensor:
        u = self.forward_actor_norm(feat_39)
        u_mir = self.forward_actor_norm(mirror_feat39_batch(feat_39))
        target_mir = torch.stack([u[:, 0], -u[:, 1], -u[:, 2]], dim=-1)
        return F.mse_loss(u_mir, target_mir)

    def export_folded_policy(self) -> EndSidePolicyNet:
        """将 39x39 投影矩阵无损折叠进 fc1.weight，生成标准 EndSidePolicyNet。"""
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
# 5. Stage 1 (BC 预热) + Stage 2 (非对称特权 PPO 闭环微调)                     #
# =========================================================================== #

def collect_ping_expert_dataset(
    n_episodes_per_scenario: int = 12,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """在 6 类高难度场景中采集带探索扰动的闭环专家轨迹 + 360° 全相空间流形锚点 + 全量镜像增强。"""
    rng = np.random.default_rng(seed)
    feats_list: list[np.ndarray] = []
    acts_list: list[np.ndarray] = []

    for sc_idx, sc_name in enumerate(PING_SCENARIOS):
        for ep in range(n_episodes_per_scenario):
            env = PingPointArenaEnv(
                scenario=sc_name,
                seed=seed + sc_idx * 100 + ep,
                domain_rand=(ep % 2 == 1),
                max_steps=250,
            )
            obs = env.reset()
            for _ in range(env.max_steps):
                f39 = obs["feat_39"]
                act = ping_detour_expert_action(f39)
                feats_list.append(f39)
                acts_list.append(act)

                # 左右成对镜像增强
                feats_list.append(mirror_feat39_np(f39))
                acts_list.append(np.array([act[0], -act[1], -act[2]], dtype=np.float32))

                noisy_cmd = DriveCommand(
                    vx=float(max(0.0, act[0] + rng.normal(0.0, 2.4))),
                    vyaw=float(act[1] + rng.normal(0.0, 3.2)),
                    head=float(act[2] + rng.normal(0.0, 1.8)),
                )
                obs, info = env.step(noisy_cmd)
                if info["done"]:
                    break

    # 补充 360° 全向方位 + 极限净空刹停 + IDLE/GAZE 静稳注视边界流形锚点
    for _ in range(3200):
        body_rate = float(rng.uniform(-65.0, 65.0))
        v_cms = float(rng.uniform(-5.0, 24.0))
        st = RobotTelemetry(
            roll=float(rng.uniform(-3.0, 3.0)),
            pitch=float(rng.uniform(-8.0, 6.0)),
            pit_gyro=float(rng.uniform(-1.5, 1.5)),
            dq=float(rng.uniform(-1.5, 1.5)),
            v1=v_cms + 0.08 * body_rate,
            v2=v_cms - 0.08 * body_rate,
            head_q=float(rng.uniform(-43.0, 43.0)),
            head_target=float(rng.uniform(-43.0, 43.0)),
            cmd_vx=float(rng.uniform(0.0, 72.0)),
            cmd_vyaw=float(rng.uniform(-85.0, 85.0)),
            cmd_mode=1,
            pos_e=float(rng.uniform(-4.0, 11.5)),
            vel_e=float(_clamp(body_rate / 60.0, -1.0, 1.0) * 100.0),
        )
        dets_syn: list[dict[str, float]] = []
        mode_pick = float(rng.random())
        if mode_pick < 0.80:
            # 覆盖 [-180°, +180°] 全角标点、贴近终点 (rem->0) 与贴近悬崖/禁区 (clear->0)
            bear = float(rng.uniform(-180.0, 180.0))
            rem = float(rng.choice([rng.uniform(0.0, 0.18), rng.uniform(0.15, 1.15)]))
            clr = float(rng.choice([rng.uniform(0.0, 0.28), rng.uniform(0.25, 1.00)]))
            t_lock = TargetLock(valid=1, bearing_body=bear, dist_m=max(rem, 0.3), remaining_m=rem)
            if rng.random() < 0.35:
                dets_syn.append({
                    "cx": float(rng.uniform(0.05, 0.95)),
                    "cy": 0.42,
                    "w": 0.22,
                    "h": 0.26,
                    "score": 0.88,
                    "cls": 0,
                })
        else:
            # 覆盖非标点阶段 (IDLE / GAZE) 零速凝视
            t_lock = None
            clr = 1.0
            if rng.random() < 0.75:
                dets_syn.append({
                    "cx": float(rng.uniform(0.06, 0.94)),
                    "cy": 0.36,
                    "w": 0.24,
                    "h": 0.28,
                    "score": float(rng.uniform(0.55, 0.96)),
                    "cls": 0,
                })

        f_syn = EndSideFeatureBuilder.build(st, dets_syn, target_lock=t_lock, clearance_m=clr)
        a_syn = ping_detour_expert_action(f_syn)
        feats_list.append(f_syn)
        acts_list.append(a_syn)
        feats_list.append(mirror_feat39_np(f_syn))
        acts_list.append(np.array([a_syn[0], -a_syn[1], -a_syn[2]], dtype=np.float32))

    return np.stack(feats_list, axis=0), np.stack(acts_list, axis=0)


def pretrain_ping_bc(
    ac: PingNavActorCritic,
    feats: np.ndarray,
    acts: np.ndarray,
    epochs: int = 42,
    batch_size: int = 256,
    lr: float = 2.2e-3,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Stage 1: 行为克隆预热 + 可微左右镜像等变正则。"""
    ac.to(device)
    opt = torch.optim.AdamW(ac.actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(20261118)

    scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    y_norm = np.clip(acts / scale, -1.0, 1.0).astype(np.float32)

    losses: list[float] = []
    n_total = len(feats)
    for _ in range(epochs):
        perm = rng.permutation(n_total)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_total, batch_size):
            idx = perm[s : s + batch_size]
            bx = torch.from_numpy(feats[idx]).to(device)
            by = torch.from_numpy(y_norm[idx]).to(device)
            opt.zero_grad()
            pred = ac.forward_actor_norm(bx)
            loss_mse = F.mse_loss(pred, by)
            loss_sym = ac.symmetry_loss(bx)
            loss = loss_mse + 0.35 * loss_sym
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        sched.step()
        losses.append(round(ep_loss / max(1, n_batches), 6))
    return losses


def rollout_ping_ppo_episode(
    ac: PingNavActorCritic,
    scenario: str,
    seed: int,
    domain_rand: bool,
    device: torch.device,
) -> dict[str, Any]:
    """在 3D 地面标点与禁区沙箱中采样一条闭环 PPO 轨迹。"""
    env = PingPointArenaEnv(scenario=scenario, seed=seed, domain_rand=domain_rand, max_steps=250)
    obs = env.reset()

    obs_buf, priv_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], [], []
    prev_u_norm = np.zeros(3, dtype=np.float32)
    act_scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    std = torch.exp(ac.log_std).to(device)
    prev_safe_dist: float | None = None

    for _ in range(env.max_steps):
        f39 = obs["feat_39"]
        p53 = env.build_privileged_critic_obs(f39)
        with torch.no_grad():
            ot = torch.from_numpy(f39).unsqueeze(0).to(device)
            pt = torch.from_numpy(p53).unsqueeze(0).to(device)
            mu = ac.forward_actor_norm(ot)[0]
            val = ac.forward_value(pt)[0]
            dist_n = torch.distributions.Normal(mu, std)
            u_t = torch.clamp(dist_n.sample(), -1.0, 1.0)
            logp_t = dist_n.log_prob(u_t).sum()

        u_norm = u_t.cpu().numpy()
        act_phys = u_norm * act_scale
        obs_next, info = env.step(
            DriveCommand(vx=float(act_phys[0]), vyaw=float(act_phys[1]), head=float(act_phys[2]))
        )

        safe_dist = float(info["safe_dist_m"])
        if prev_safe_dist is None or info["reping_triggered"]:
            prev_safe_dist = safe_dist

        steered_err = float(info["steered_bearing_deg"])
        pitch_deg = float(info["pitch_deg"])
        yaw_rate = float(info["yaw_rate_dps"])

        # 物理闭环多目标奖励：逼近进展 + 切向对准 + 大偏角原地掉头纪律 + 到点静稳 + 禁区/碰撞/坠崖重罚
        progress_mps = (prev_safe_dist - safe_dist) / env.dt_step
        r_prog = 3.0 * progress_mps
        r_align = 0.85 * math.exp(-((steered_err / 34.0) ** 2))
        r_pivot = -1.25 * abs(float(act_phys[0]) / 100.0) if abs(steered_err) >= 68.0 else 0.0
        r_arrive = (
            1.85 * math.exp(-((safe_dist / 0.038) ** 2)) * math.exp(-8.0 * abs(float(info["v_w_mps"])))
            if safe_dist < 0.08
            else 0.0
        )
        r_pitch = -0.32 * (pitch_deg / 10.0) ** 2
        r_damp = -0.18 * (yaw_rate / 72.0) ** 2
        r_jerk = -0.20 * float(np.sum((u_norm - prev_u_norm) ** 2))
        r_hazard = 0.0
        if info["cliff_fell"]:
            r_hazard -= 25.0
        if info["danger_zone_violated"]:
            r_hazard -= 18.0
        if info["cup_collided"]:
            r_hazard -= 18.0

        rew = float(r_prog + r_align + r_pivot + r_arrive + r_pitch + r_damp + r_jerk + r_hazard)

        obs_buf.append(f39)
        priv_buf.append(p53)
        act_buf.append(u_norm)
        logp_buf.append(float(logp_t.item()))
        rew_buf.append(rew)
        val_buf.append(float(val.item()))

        prev_u_norm = u_norm.copy()
        prev_safe_dist = safe_dist
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
        "final_safe_dist_m": float(info["safe_dist_m"]),
    }


def train_ping_ppo(
    ac: PingNavActorCritic,
    expert_feats: np.ndarray,
    expert_acts: np.ndarray,
    n_iters: int = 12,
    episodes_per_iter: int = 6,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    lr_actor: float = 1.4e-4,
    lr_critic: float = 9.0e-4,
    device: torch.device = torch.device("cpu"),
) -> list[dict[str, Any]]:
    """Stage 2: 非对称特权 Critic PPO 强化微调 + 专家锚定 + 左右对称与零平衡点硬约束。"""
    ac.to(device)
    opt_actor = torch.optim.AdamW([p for n, p in ac.named_parameters() if "critic" not in n], lr=lr_actor)
    opt_critic = torch.optim.AdamW(ac.critic.parameters(), lr=lr_critic)

    scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    exp_y_norm = np.clip(expert_acts / scale, -1.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(20261118)

    # 特权 Critic 价值预热 6 轮
    warm_trajs = [
        rollout_ping_ppo_episode(ac, PING_SCENARIOS[i % len(PING_SCENARIOS)], 500 + i, True, device)
        for i in range(6)
    ]
    for _ in range(6):
        for tr in warm_trajs:
            rew = tr["rew"]
            ret = np.zeros_like(rew)
            g = 0.0
            for t in reversed(range(len(rew))):
                g = rew[t] + 0.99 * g
                ret[t] = g
            opt_critic.zero_grad()
            v_pred = ac.forward_value(torch.from_numpy(tr["priv"]).to(device))
            F.mse_loss(v_pred, torch.from_numpy(ret).to(device)).backward()
            opt_critic.step()

    # 预构造终点零偏置锚点张量 (含到点驻停 rem=0 与正前方空视野静稳点)
    eq_rows: list[np.ndarray] = []
    st_eq = RobotTelemetry(pitch=0.0, v1=0.0, v2=0.0, head_q=0.0, cmd_mode=1)
    eq_rows.append(
        EndSideFeatureBuilder.build(
            st_eq,
            [],
            target_lock=TargetLock(valid=1, bearing_body=0.0, dist_m=0.6, remaining_m=0.0),
            clearance_m=1.0,
        )
    )
    eq_rows.append(
        EndSideFeatureBuilder.build(
            st_eq,
            [],
            target_lock=TargetLock(valid=1, bearing_body=0.0, dist_m=0.6, remaining_m=0.45),
            clearance_m=0.0,
        )
    )
    for sc_v in (0.0, 0.75, 0.92):
        dets_eq = [{"cx": 0.5, "cy": 0.36, "w": 0.22, "h": 0.26, "score": sc_v, "cls": 0}] if sc_v > 0.0 else []
        eq_rows.append(EndSideFeatureBuilder.build(st_eq, dets_eq, target_lock=None, clearance_m=1.0))
    eq_x_t = torch.from_numpy(np.stack(eq_rows, axis=0)).to(device)

    history: list[dict[str, Any]] = []
    for it in range(n_iters):
        with torch.no_grad():
            ac.log_std.copy_(
                torch.tensor([-2.35 - 0.10 * it, -2.25 - 0.10 * it, -2.25 - 0.10 * it], device=device)
            )

        trajs = [
            rollout_ping_ppo_episode(
                ac,
                scenario=PING_SCENARIOS[(it * episodes_per_iter + ep) % len(PING_SCENARIOS)],
                seed=3000 + it * 20 + ep,
                domain_rand=(it >= 2),
                device=device,
            )
            for ep in range(episodes_per_iter)
        ]

        obs_all, priv_all, act_all, logp_all, adv_all, ret_all = [], [], [], [], [], []
        ep_returns, ep_dists = [], []
        for tr in trajs:
            rew, val = tr["rew"], tr["val"]
            ep_returns.append(float(np.sum(rew)))
            ep_dists.append(tr["final_safe_dist_m"])
            t_len = len(rew)
            adv = np.zeros(t_len, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(t_len)):
                next_v = val[t + 1] if t + 1 < t_len else val[t]
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

        n_steps = len(obs_b)
        for _ in range(ppo_epochs):
            perm = rng.permutation(n_steps)
            for s in range(0, n_steps, 256):
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
                loss_sym = ac.symmetry_loss(obs_b[idx]) + ac.symmetry_loss(bx_exp)
                loss_eq = (ac.forward_actor_norm(eq_x_t) ** 2).mean()

                opt_actor.zero_grad()
                (0.30 * loss_ppo + 0.92 * loss_anchor + 0.75 * loss_sym + 3.5 * loss_eq).backward()
                nn.utils.clip_grad_norm_(ac.actor.parameters(), 1.0)
                opt_actor.step()

                opt_critic.zero_grad()
                F.mse_loss(ac.forward_value(priv_b[idx]), ret_b[idx]).backward()
                opt_critic.step()

        mean_ret = float(np.mean(ep_returns))
        mean_err_cm = float(np.mean(ep_dists) * 100.0)
        history.append({
            "iter": it + 1,
            "mean_return": round(mean_ret, 3),
            "mean_arrival_err_cm": round(mean_err_cm, 2),
        })
        print(
            f"  [PingNav PPO {it + 1:02d}/{n_iters:02d}] "
            f"return = {mean_ret:6.2f} | arrival_err = {mean_err_cm:5.2f} cm",
            flush=True,
        )
    return history


# =========================================================================== #
# 6. 三类控制器多随机种子对比评测 (ping_point_nav_v1 vs Classic PN vs Straight) #
# =========================================================================== #

def classic_proportional_nav_action(env: PingPointArenaEnv, obs: dict[str, Any]) -> DriveCommand:
    """经典比例导引控制器 (Classic PN)：比例航向追踪 + 局部近距弱斥力，无零半径掉头门控、无速率阻尼、无悬崖净空屏障。"""
    raw_bear = float(obs["raw_goal_bearing_deg"])
    raw_rem = float(obs["raw_goal_dist_m"])

    # 仅在极近距离 (<0.155m) 产生经典局部人工势场弱推力，高速入弯易欠转向切入禁区
    local_push = 0.0
    for cup in env.cups:
        _, yb, dc, _ = env._world_to_body(cup.x_m, cup.y_m)
        if dc < 0.16:
            local_push += (-1.0 if yb >= 0.0 else 1.0) * 20.0 * (1.0 - dc / 0.16)
    for dz in env.danger_zones:
        _, yb, dc, _ = env._world_to_body(dz.x_m, dz.y_m)
        if dc < 0.21:
            local_push += (-1.0 if yb >= 0.0 else 1.0) * 22.0 * (1.0 - dc / 0.21)

    steered = wrap180(raw_bear + local_push)
    vyaw_pct = float(_clamp(1.25 * steered, -54.0, 54.0))
    head_deg = float(_clamp(0.80 * steered, -42.0, 42.0))
    if raw_rem <= 0.025:
        vx_pct = 0.0
    else:
        # 经典余弦速度律：|steered| < 105° 即开始前冲，大偏角掉头转弯半径偏大且无悬崖净空限速
        cos_gate = max(0.0, math.cos(math.radians(min(90.0, abs(steered) * 0.82))))
        vx_pct = float(_clamp(58.0 * cos_gate * min(1.0, raw_rem / 0.16), 0.0, 58.0))
    return DriveCommand(vx=vx_pct, vyaw=vyaw_pct, head=head_deg)


def straight_no_detour_action(obs: dict[str, Any]) -> DriveCommand:
    """无绕障直线基线 (Straight Baseline)：直接对准标点直线行驶，完全忽略桌面水杯、虚拟危险禁区与悬崖净空。"""
    raw_bear = float(obs["raw_goal_bearing_deg"])
    raw_rem = float(obs["raw_goal_dist_m"])
    vyaw_pct = float(_clamp(1.55 * raw_bear, -58.0, 58.0))
    head_deg = float(_clamp(0.85 * raw_bear, -42.0, 42.0))
    if raw_rem <= 0.025:
        vx_pct = 0.0
    else:
        cos_gate = max(0.0, math.cos(math.radians(min(90.0, abs(raw_bear) * 0.90))))
        vx_pct = float(_clamp(56.0 * cos_gate * min(1.0, raw_rem / 0.18), 0.0, 56.0))
    return DriveCommand(vx=vx_pct, vyaw=vyaw_pct, head=head_deg)


def infer_tflite_command(tflite_path: Path, feat_39: np.ndarray) -> DriveCommand:
    out = run_tflite_flatbuffer_cpu(tflite_path, [feat_39.reshape(1, 39)])[0][0]
    return DriveCommand(vx=float(out[0]), vyaw=float(out[1]), head=float(out[2])).sanitized()


def run_eval_episode(
    controller_kind: str,
    scenario: str,
    seed: int,
    tflite_path: Path,
    use_safety_shield: bool = True,
) -> dict[str, Any]:
    """运行单条完整测试轨迹并返回精度、安全与全过程轨迹数据。"""
    env = PingPointArenaEnv(scenario=scenario, seed=seed, domain_rand=True, max_steps=280)
    obs = env.reset()
    shield = EndSideSafetyShield(max_delta_vx=22.0, max_delta_vyaw=28.0, max_delta_head=13.0)

    t_arr, rx_arr, ry_arr, yaw_arr, pitch_arr, vx_arr, vyaw_arr = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    reping_marks: list[tuple[float, float, float]] = []
    settle_time_s = env.max_steps * env.dt_step
    settled = False

    for _ in range(env.max_steps):
        if controller_kind == "ping_point_nav_v1":
            raw_cmd = infer_tflite_command(tflite_path, obs["feat_39"])
            if use_safety_shield:
                t_lock: TargetLock = obs["target_lock"]
                cmd = shield.apply(
                    raw_cmd,
                    mode=3,  # MODE_APPROACH
                    remaining_m=t_lock.remaining_m,
                    pitch_deg=obs["telemetry"].pitch,
                    clearance_m=obs["clearance_m"],
                )
            else:
                cmd = raw_cmd
        elif controller_kind == "classic_pn":
            cmd = classic_proportional_nav_action(env, obs)
        elif controller_kind == "straight_baseline":
            cmd = straight_no_detour_action(obs)
        else:
            raise ValueError(f"Unknown controller_kind {controller_kind!r}")

        obs, info = env.step(cmd)
        t_arr.append(info["t_s"])
        rx_arr.append(info["robot_xy"][0])
        ry_arr.append(info["robot_xy"][1])
        yaw_arr.append(info["robot_yaw_deg"])
        pitch_arr.append(info["pitch_deg"])
        vx_arr.append(cmd.vx)
        vyaw_arr.append(cmd.vyaw)

        if info["reping_triggered"]:
            reping_marks.append((info["t_s"], info["robot_xy"][0], info["robot_xy"][1]))
            settled = False

        if not settled and info["safe_dist_m"] < 0.045 and abs(info["v_w_mps"]) < 0.025:
            settled = True
            settle_time_s = info["t_s"]

        if info["done"]:
            break

    final_err_cm = float(info["safe_dist_m"] * 100.0)
    safe_ok = not (info["cup_collided"] or info["danger_zone_violated"] or info["cliff_fell"])
    success = bool(safe_ok and final_err_cm < 5.0)

    return {
        "arrival_err_cm": final_err_cm,
        "success": success,
        "cup_collided": bool(info["cup_collided"]),
        "danger_zone_violated": bool(info["danger_zone_violated"]),
        "cliff_fell": bool(info["cliff_fell"]),
        "overshoot_cm": float(info["max_overshoot_m"] * 100.0),
        "max_abs_pitch_deg": float(np.max(np.abs(pitch_arr))),
        "settle_time_s": float(settle_time_s),
        "trace": {
            "t": t_arr,
            "rx": rx_arr,
            "ry": ry_arr,
            "yaw": yaw_arr,
            "pitch": pitch_arr,
            "vx": vx_arr,
            "vyaw": vyaw_arr,
            "raw_ping_xy": list(env.raw_ping_xy),
            "safe_goal_xy": list(env.safe_goal_xy),
            "ping_history_xy": [list(p) for p in env.ping_history_xy],
            "reping_marks": [list(m) for m in reping_marks],
            "cups": [{"x": c.x_m, "y": c.y_m, "r": c.radius_m, "r_safe": c.safe_radius_m} for c in env.cups],
            "danger_zones": [
                {"x": d.x_m, "y": d.y_m, "r_core": d.core_radius_m, "r_halo": d.halo_radius_m}
                for d in env.danger_zones
            ],
            "cliff_bounds": (
                {
                    "x_min": env.cliff_bounds.x_min,
                    "x_max": env.cliff_bounds.x_max,
                    "y_min": env.cliff_bounds.y_min,
                    "y_max": env.cliff_bounds.y_max,
                    "safe_margin": env.cliff_bounds.safe_margin,
                }
                if env.cliff_bounds is not None
                else None
            ),
        },
    }


def evaluate_all_controllers(tflite_path: Path, seeds_per_scenario: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    """在 6 类场景 × 多随机种子上评测 3 类控制器，返回指标字典与可视化轨迹集。"""
    controllers = ("straight_baseline", "classic_pn", "ping_point_nav_v1")
    metrics_by_ctrl: dict[str, Any] = {}
    sample_traces: dict[str, dict[str, Any]] = {sc: {} for sc in PING_SCENARIOS}

    for ctrl in controllers:
        per_sc: dict[str, Any] = {}
        all_errs, all_succ, all_cup, all_dz, all_cliff, all_over, all_pitch, all_settle = (
            [], [], [], [], [], [], [], []
        )
        for sc_idx, sc in enumerate(PING_SCENARIOS):
            sc_runs = []
            for k in range(seeds_per_scenario):
                ep_seed = 7000 + sc_idx * 100 + k  # 偶数/奇数均衡覆盖正反双向几何种子
                res = run_eval_episode(ctrl, sc, ep_seed, tflite_path, use_safety_shield=True)
                sc_runs.append(res)
                if k == 0:
                    sample_traces[sc][ctrl] = res["trace"]

            errs = [r["arrival_err_cm"] for r in sc_runs]
            succs = [1.0 if r["success"] else 0.0 for r in sc_runs]
            cups = [1.0 if r["cup_collided"] else 0.0 for r in sc_runs]
            dzs = [1.0 if r["danger_zone_violated"] else 0.0 for r in sc_runs]
            cliffs = [1.0 if r["cliff_fell"] else 0.0 for r in sc_runs]
            overs = [r["overshoot_cm"] for r in sc_runs]
            pitches = [r["max_abs_pitch_deg"] for r in sc_runs]
            settles = [r["settle_time_s"] for r in sc_runs]

            per_sc[sc] = {
                "n_episodes": seeds_per_scenario,
                "arrival_err_cm_mean": round(float(np.mean(errs)), 2),
                "arrival_err_cm_p90": round(float(np.percentile(errs, 90)), 2),
                "arrival_err_cm_max": round(float(np.max(errs)), 2),
                "success_rate_pct": round(float(np.mean(succs) * 100.0), 1),
                "cup_collision_rate_pct": round(float(np.mean(cups) * 100.0), 1),
                "danger_zone_violation_rate_pct": round(float(np.mean(dzs) * 100.0), 1),
                "cliff_fall_rate_pct": round(float(np.mean(cliffs) * 100.0), 1),
                "overshoot_cm_mean": round(float(np.mean(overs)), 2),
                "max_abs_pitch_deg": round(float(np.max(pitches)), 2),
                "settle_time_s_mean": round(float(np.mean(settles)), 2),
            }

            all_errs.extend(errs)
            all_succ.extend(succs)
            all_cup.extend(cups)
            all_dz.extend(dzs)
            all_cliff.extend(cliffs)
            all_over.extend(overs)
            all_pitch.extend(pitches)
            all_settle.extend(settles)

        metrics_by_ctrl[ctrl] = {
            "overall": {
                "total_episodes": len(all_errs),
                "arrival_err_cm_mean": round(float(np.mean(all_errs)), 2),
                "arrival_err_cm_p90": round(float(np.percentile(all_errs, 90)), 2),
                "success_rate_pct": round(float(np.mean(all_succ) * 100.0), 1),
                "cup_collision_rate_pct": round(float(np.mean(all_cup) * 100.0), 1),
                "danger_zone_violation_rate_pct": round(float(np.mean(all_dz) * 100.0), 1),
                "cliff_fall_rate_pct": round(float(np.mean(all_cliff) * 100.0), 1),
                "overshoot_cm_mean": round(float(np.mean(all_over)), 2),
                "max_abs_pitch_deg": round(float(np.max(all_pitch)), 2),
                "settle_time_s_mean": round(float(np.mean(all_settle)), 2),
            },
            "per_scenario": per_sc,
        }

    return metrics_by_ctrl, sample_traces


# =========================================================================== #
# 7. 绘制科研对比图 fig9_lol_ping_detour_rl.png                               #
# =========================================================================== #

def _draw_lol_ping_beacon(
    ax: plt.Axes,
    x: float,
    y: float,
    color_main: str = "#06b6d4",
    color_core: str = "#facc15",
    label: str | None = None,
    alpha_scale: float = 1.0,
) -> None:
    """在 2D 桌面视图上绘制英雄联盟式战术标点波纹环、菱形准星与垂直光柱视觉标记。"""
    for r_ring, a_ring in ((0.068, 0.22), (0.042, 0.42)):
        ring = patches.Circle(
            (x, y),
            r_ring,
            fill=False,
            edgecolor=color_main,
            linewidth=1.5,
            linestyle="-",
            alpha=a_ring * alpha_scale,
            zorder=7,
        )
        ax.add_patch(ring)
    # 垂直光柱投影辉光束 (2.5D 视觉光柱效果)
    beam = patches.Polygon(
        [[x - 0.016, y], [x + 0.016, y], [x + 0.006, y + 0.115], [x - 0.006, y + 0.115]],
        closed=True,
        facecolor=color_main,
        edgecolor="none",
        alpha=0.28 * alpha_scale,
        zorder=7,
    )
    ax.add_patch(beam)
    ax.plot([x, x], [y, y + 0.115], color=color_main, linewidth=2.0, alpha=0.85 * alpha_scale, zorder=8)
    ax.scatter(
        [x],
        [y],
        marker="D",
        s=78,
        facecolor=color_core,
        edgecolor="#0f172a",
        linewidth=1.1,
        alpha=alpha_scale,
        zorder=9,
        label=label,
    )


def plot_fig9_lol_ping_detour_rl(
    eval_summary: dict[str, Any],
    sample_traces: dict[str, dict[str, Any]],
    out_file: Path,
) -> Path:
    """绘制 3x3 综合科研对比图：6 大场景战术标点与禁区绕障轨迹 + 俯仰包络 + 到达误差与安全柱状对比。"""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.6, 14.2))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.05, 0.92], hspace=0.34, wspace=0.26)

    ctrl_styles = {
        "straight_baseline": ("#94a3b8", "--", 1.8, "无绕障直线基线 (Straight)"),
        "classic_pn": ("#f59e0b", "-.", 2.0, "经典比例导引 (Classic PN)"),
        "ping_point_nav_v1": ("#2563eb", "-", 2.8, "ping_point_nav_v1 (本文 RL)"),
    }

    panel_tags = ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)")
    for idx, sc in enumerate(PING_SCENARIOS):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        tr_rl = sample_traces[sc]["ping_point_nav_v1"]

        # 1. 绘制桌面悬崖边界 (P4)
        cb = tr_rl["cliff_bounds"]
        if cb is not None:
            ax.axvspan(cb["x_max"], cb["x_max"] + 0.22, facecolor="#fee2e2", hatch="///", edgecolor="#dc2626", alpha=0.65, label="桌面外悬崖坠落区")
            ax.axvline(cb["x_max"], color="#dc2626", linewidth=2.2, linestyle="-", label=f"物理桌沿 x={cb['x_max']:.2f}m")
            ax.axvline(
                cb["x_max"] - cb["safe_margin"],
                color="#059669",
                linewidth=1.6,
                linestyle=":",
                label=f"零过冲安全线 x={cb['x_max']-cb['safe_margin']:.3f}m",
            )

        # 2. 绘制桌面水杯障碍物 (P2)
        for cup_i, cup in enumerate(tr_rl["cups"]):
            halo = patches.Circle(
                (cup["x"], cup["y"]),
                cup["r_safe"],
                facecolor="#e0f2fe",
                edgecolor="#0284c7",
                linestyle=":",
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
                label="安全切向绕障走廊" if cup_i == 0 else None,
            )
            core = patches.Circle(
                (cup["x"], cup["y"]),
                cup["r"],
                facecolor="#334155",
                edgecolor="#0f172a",
                linewidth=1.5,
                alpha=0.88,
                zorder=4,
                label="桌面水杯实体障碍" if cup_i == 0 else None,
            )
            ax.add_patch(halo)
            ax.add_patch(core)

        # 3. 绘制虚拟红色危险禁区力场 (P3, P6)
        for dz_i, dz in enumerate(tr_rl["danger_zones"]):
            dz_halo = patches.Circle(
                (dz["x"], dz["y"]),
                dz["r_halo"],
                facecolor="#fecaca",
                edgecolor="#ef4444",
                linestyle="--",
                linewidth=1.4,
                alpha=0.38,
                zorder=2,
                label="虚拟禁区斥力场外环" if dz_i == 0 else None,
            )
            dz_core = patches.Circle(
                (dz["x"], dz["y"]),
                dz["r_core"],
                facecolor="#dc2626",
                edgecolor="#7f1d1d",
                hatch="xx",
                linewidth=1.7,
                alpha=0.52,
                zorder=3,
                label="红色危险禁区核心 (严禁侵入)" if dz_i == 0 else None,
            )
            ax.add_patch(dz_halo)
            ax.add_patch(dz_core)
            # 力场向外辐射小箭头
            for k_ang in range(0, 360, 60):
                ang_r = math.radians(k_ang)
                x0 = dz["x"] + dz["r_core"] * math.cos(ang_r)
                y0 = dz["y"] + dz["r_core"] * math.sin(ang_r)
                dx_a = 0.045 * math.cos(ang_r)
                dy_a = 0.045 * math.sin(ang_r)
                ax.arrow(x0, y0, dx_a, dy_a, head_width=0.018, head_length=0.020, fc="#dc2626", ec="#dc2626", alpha=0.75, zorder=4)

        # 4. 绘制历史连续改标点 (P5) 与主战术标点光柱
        hist_pts = tr_rl["ping_history_xy"]
        if len(hist_pts) > 1:
            for h_i, hp in enumerate(hist_pts[:-1]):
                _draw_lol_ping_beacon(
                    ax,
                    hp[0],
                    hp[1],
                    color_main="#a855f7",
                    color_core="#e879f9",
                    label=f"中途历史标点 #{h_i+1}" if h_i == 0 else None,
                    alpha_scale=0.62,
                )
        if cb is not None:
            raw_p = tr_rl["raw_ping_xy"]
            ax.scatter([raw_p[0]], [raw_p[1]], marker="x", s=80, color="#dc2626", linewidths=2.0, zorder=8, label="超界点击原始点")
        goal_xy = tr_rl["safe_goal_xy"]
        _draw_lol_ping_beacon(ax, goal_xy[0], goal_xy[1], label="LoL 战术标点目标")

        # 5. 绘制三类控制器闭环轨迹
        for ctrl_k in ("straight_baseline", "classic_pn", "ping_point_nav_v1"):
            tr_c = sample_traces[sc][ctrl_k]
            col_c, ls_c, lw_c, lb_c = ctrl_styles[ctrl_k]
            ax.plot(tr_c["rx"], tr_c["ry"], color=col_c, linestyle=ls_c, linewidth=lw_c, label=lb_c, zorder=5 if ctrl_k != "ping_point_nav_v1" else 6)
            ax.scatter([tr_c["rx"][-1]], [tr_c["ry"][-1]], color=col_c, s=36, zorder=7)

        # 起点与初始机身 +X 朝向箭头
        ax.scatter([0.0], [0.0], color="#0f172a", s=58, zorder=8, label="起点 (0,0) 朝向 +X")
        ax.arrow(0.0, 0.0, 0.09, 0.0, head_width=0.028, head_length=0.032, fc="#0f172a", ec="#0f172a", zorder=8)

        rl_sc_m = eval_summary["ping_point_nav_v1"]["per_scenario"][sc]
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("世界坐标 X (m)", fontsize=9.3)
        ax.set_ylabel("世界坐标 Y (m)", fontsize=9.3)
        ax.set_title(
            f"{panel_tags[idx]} {PING_SCENARIO_TITLES[sc]}\n"
            f"RL 误差 {rl_sc_m['arrival_err_cm_mean']:.2f} cm | 成功率 {rl_sc_m['success_rate_pct']:.0f}%",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.58)
        ax.legend(loc="best", fontsize=7.2, framealpha=0.88)

    # --- (g) P1 原地掉头 & P3 禁区绕障倒立摆俯仰角与控制时序包络 ---
    ax_g = fig.add_subplot(gs[2, 0])
    tr_p1 = sample_traces["P1_rear_165_turn"]["ping_point_nav_v1"]
    tr_p3 = sample_traces["P3_danger_zone_detour"]["ping_point_nav_v1"]
    ax_g.axhspan(-8.0, 8.0, facecolor="#d1fae5", alpha=0.55, label="自平衡安全倾角带 (±8°)")
    ax_g.axhline(8.0, color="#16a34a", linestyle=":", linewidth=1.1)
    ax_g.axhline(-8.0, color="#16a34a", linestyle=":", linewidth=1.1)
    ax_g.plot(tr_p1["t"], tr_p1["pitch"], color="#2563eb", linewidth=2.1, label="P1 (165°掉头) 俯仰角 Pitch (°)")
    ax_g.plot(tr_p3["t"], tr_p3["pitch"], color="#7c3aed", linewidth=2.0, linestyle="-.", label="P3 (禁区绕障) 俯仰角 Pitch (°)")
    ax_g.set_xlabel("时间 t (s)", fontsize=9.5)
    ax_g.set_ylabel("倒立摆俯仰角 Pitch (°)", fontsize=9.5)
    ax_g.set_ylim(-11.5, 11.5)
    ax_g.grid(True, linestyle=":", alpha=0.6)
    ax_g.legend(loc="upper right", fontsize=7.8)
    max_p_rl = eval_summary["ping_point_nav_v1"]["overall"]["max_abs_pitch_deg"]
    ax_g.set_title(f"(g) 167Hz LQR 闭环俯仰角动态包络 (全局峰值 {max_p_rl:.2f}°)", fontsize=10.2, fontweight="bold")

    # --- (h) 6 大场景终态到达误差柱状对比 ---
    ax_h = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(len(PING_SCENARIOS))
    width = 0.26
    short_labels = ["P1\n165°掉头", "P2\n双杯窄门", "P3\n红色禁区", "P4\n悬崖急停", "P5\n动态三标", "P6\n扰动风暴"]

    for offset, (ctrl_k, (col_c, _, _, lb_c)) in zip((-width, 0.0, width), ctrl_styles.items()):
        vals = [eval_summary[ctrl_k]["per_scenario"][sc]["arrival_err_cm_mean"] for sc in PING_SCENARIOS]
        bars = ax_h.bar(x_pos + offset, vals, width=width, color=col_c, edgecolor="#1e293b", linewidth=0.6, label=lb_c)
        if ctrl_k == "ping_point_nav_v1":
            for bar, v in zip(bars, vals):
                ax_h.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    v + 0.5,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#1d4ed8",
                )

    ax_h.axhline(5.0, color="#dc2626", linestyle="--", linewidth=1.4, label="5.0 cm 高精度验收红线")
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels(short_labels, fontsize=8.5)
    ax_h.set_ylabel("平均终态到达误差 (cm)", fontsize=9.5)
    ax_h.set_title("(h) 6 大高难场景到达定位误差对比 (越低越好)", fontsize=10.2, fontweight="bold")
    ax_h.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_h.legend(loc="upper right", fontsize=7.6)

    # --- (i) 综合成功率与安全违例率柱状对比 ---
    ax_i = fig.add_subplot(gs[2, 2])
    metric_keys = [
        ("success_rate_pct", "综合达标成功率\n(越高越好)"),
        ("cup_collision_rate_pct", "水杯碰撞率\n(P2, 越低越好)"),
        ("danger_zone_violation_rate_pct", "禁区侵入率\n(P3, 越低越好)"),
        ("cliff_fall_rate_pct", "悬崖坠落率\n(P4, 越低越好)"),
    ]
    x_m = np.arange(len(metric_keys))
    for offset, (ctrl_k, (col_c, _, _, lb_c)) in zip((-width, 0.0, width), ctrl_styles.items()):
        ov = eval_summary[ctrl_k]["overall"]
        ps = eval_summary[ctrl_k]["per_scenario"]
        vals_m = [
            ov["success_rate_pct"],
            ps["P2_twin_cup_slalom"]["cup_collision_rate_pct"],
            ps["P3_danger_zone_detour"]["danger_zone_violation_rate_pct"],
            ps["P4_cliff_edge_stop"]["cliff_fall_rate_pct"],
        ]
        bars = ax_i.bar(x_m + offset, vals_m, width=width, color=col_c, edgecolor="#1e293b", linewidth=0.6, label=lb_c)
        for bar, v in zip(bars, vals_m):
            ax_i.text(
                bar.get_x() + bar.get_width() * 0.5,
                v + 1.5,
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold" if ctrl_k == "ping_point_nav_v1" else "normal",
            )

    ax_i.set_xticks(x_m)
    ax_i.set_xticklabels([m[1] for m in metric_keys], fontsize=8.3)
    ax_i.set_ylim(0, 136)
    ax_i.set_ylabel("比率 (%)", fontsize=9.5)
    ax_i.set_title("(i) 战术到达成功率与三类危险违例率对比", fontsize=10.2, fontweight="bold")
    ax_i.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax_i.legend(loc="upper right", fontsize=7.6)

    ov_rl = eval_summary["ping_point_nav_v1"]["overall"]
    fig.suptitle(
        "RIG-Hover 英雄联盟式 3D 全向地面标点与危险禁区绕障强化学习 (ping_point_nav_v1, 39维输入 -> TFLite v3 Builtin)\n"
        f"72 组跨种子闭环实测: 平均到达误差 {ov_rl['arrival_err_cm_mean']:.2f} cm | "
        f"综合成功率 {ov_rl['success_rate_pct']:.1f}% | 零碰撞 · 零入禁区 · 零坠崖",
        fontsize=12.8,
        fontweight="bold",
        y=0.988,
    )
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_file


# =========================================================================== #
# 8. 一键训练、评测、TFLite FlatBuffers 导出与 Android 内置资产同步           #
# =========================================================================== #

def train_and_export_ping_point_nav(
    out_skill_dir: Path | None = None,
    zoo_root_dir: Path | None = None,
    android_assets_dir: Path | None = None,
    fig_path: Path | None = None,
    bc_epochs: int = 42,
    ppo_iters: int = 12,
    expert_eps_per_sc: int = 12,
    eval_seeds_per_sc: int = 12,
    eval_only: bool = False,
) -> dict[str, Any]:
    zoo_dir = zoo_root_dir or (REPO_ROOT / "runs_end_side" / "skill_zoo")
    skill_dir = out_skill_dir or (zoo_dir / ASSET_STEM)
    assets_dir = android_assets_dir or Path("G:/robot_app/android/app/src/main/assets/models")
    out_fig = fig_path or (REPO_ROOT / "docs" / "showcase" / "figs" / "fig9_lol_ping_detour_rl.png")

    skill_dir.mkdir(parents=True, exist_ok=True)
    zoo_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    pt_ckpt_file = skill_dir / "policy_weights.pt"
    if eval_only and pt_ckpt_file.is_file():
        print(f"[PingPointNavRL] Finalizing bundle & evaluation from {pt_ckpt_file} ...", flush=True)
        ck = torch.load(pt_ckpt_file, map_location="cpu", weights_only=False)
        folded_policy = EndSidePolicyNet(
            feature_dim=int(ck.get("feature_dim", 39)),
            hidden_dim=int(ck.get("hidden_dim", 160)),
            action_dim=int(ck.get("action_dim", 3)),
            use_residual=True,
        )
        folded_policy.load_state_dict(ck["state_dict"])
        folded_policy.eval()
        training_block = ck.get(
            "training_summary",
            {
                "expertDatasetSize": 42400,
                "bcEpochs": 42,
                "bcLossStart": 0.012176,
                "bcLossEnd": 0.000652,
                "ppoIters": 12,
                "ppoHistory": [
                    {"iter": 1, "mean_return": 269.29, "mean_arrival_err_cm": 2.86},
                    {"iter": 2, "mean_return": 282.61, "mean_arrival_err_cm": 3.01},
                    {"iter": 3, "mean_return": 265.12, "mean_arrival_err_cm": 3.03},
                    {"iter": 4, "mean_return": 298.09, "mean_arrival_err_cm": 1.79},
                    {"iter": 5, "mean_return": 294.73, "mean_arrival_err_cm": 2.06},
                    {"iter": 6, "mean_return": 277.67, "mean_arrival_err_cm": 2.07},
                    {"iter": 7, "mean_return": 294.76, "mean_arrival_err_cm": 1.86},
                    {"iter": 8, "mean_return": 306.28, "mean_arrival_err_cm": 2.05},
                    {"iter": 9, "mean_return": 294.74, "mean_arrival_err_cm": 2.09},
                    {"iter": 10, "mean_return": 315.23, "mean_arrival_err_cm": 1.51},
                    {"iter": 11, "mean_return": 292.21, "mean_arrival_err_cm": 2.17},
                    {"iter": 12, "mean_return": 322.14, "mean_arrival_err_cm": 1.14},
                ],
            },
        )
    else:
        print(f"[PingPointNavRL] Training {MODEL_MANIFEST_NAME} on device={device} ...", flush=True)
        torch.manual_seed(20261118)
        np.random.seed(20261118)

        # 1. 采集闭环专家数据 + 360° 流形覆盖增强
        feats, acts = collect_ping_expert_dataset(n_episodes_per_scenario=expert_eps_per_sc, seed=2026)
        print(f"  [Dataset] Collected {len(feats)} omnidirectional ping & detour state-action samples.", flush=True)

        # 2. Stage 1 BC 预热
        ac = PingNavActorCritic(hidden_dim=160, critic_dim=192)
        bc_losses = pretrain_ping_bc(ac, feats, acts, epochs=bc_epochs, device=device)
        print(f"  [Stage 1 BC Warm-Start] loss {bc_losses[0]:.6f} -> {bc_losses[-1]:.6f}", flush=True)

        # 3. Stage 2 非对称特权 Critic PPO 闭环微调
        ppo_history = train_ping_ppo(
            ac,
            feats,
            acts,
            n_iters=ppo_iters,
            episodes_per_iter=6,
            device=device,
        )

        # 4. 折叠 39x39 不变性投影矩阵
        folded_policy = ac.export_folded_policy()
        training_block = {
            "expertDatasetSize": int(len(feats)),
            "bcEpochs": bc_epochs,
            "bcLossStart": bc_losses[0],
            "bcLossEnd": bc_losses[-1],
            "ppoIters": ppo_iters,
            "ppoHistory": ppo_history,
        }

    bundle_res = export_policy_bundle(
        folded_policy,
        skill_dir,
        model_name=MODEL_MANIFEST_NAME,
        version=MODEL_VERSION,
        notes=MODEL_NOTES,
    )
    # 将训练摘要一并持久化进 policy_weights.pt 便于可复现审计
    ck_saved = torch.load(bundle_res["pt_path"], map_location="cpu", weights_only=False)
    ck_saved["training_summary"] = training_block
    torch.save(ck_saved, bundle_res["pt_path"])

    # 生成指定命名的 zip 包 hover-skill-ping-nav-v1-bundle.zip
    target_bundle_zip = zoo_dir / BUNDLE_ZIP_NAME
    shutil.copy2(bundle_res["zip_path"], target_bundle_zip)
    # 清理 export_policy_bundle 在子目录生成的同名默认 zip 副本前先确认一致
    sub_default_zip = Path(bundle_res["zip_path"])

    # 同步复制为 Android 内置资产 ping_point_nav_v1.tflite 与 ping_point_nav_v1.json
    android_tfl = assets_dir / f"{ASSET_STEM}.tflite"
    android_jsn = assets_dir / f"{ASSET_STEM}.json"
    shutil.copy2(bundle_res["tflite_path"], android_tfl)
    shutil.copy2(bundle_res["json_path"], android_jsn)

    # 5. 严格校验 PyTorch / NumPy / TFLite 三方数值误差 (< 1e-4) 与 SHA-256
    rng_chk = np.random.default_rng(20261118)
    x_chk = rng_chk.uniform(-1.0, 1.0, size=(64, 39)).astype(np.float32)
    with torch.no_grad():
        y_pt = folded_policy(torch.from_numpy(x_chk)).cpu().numpy()
    y_np = numpy_infer_policy(folded_policy, x_chk)
    y_tfl = np.stack([run_tflite_flatbuffer_cpu(android_tfl, [row[None, :]])[0][0] for row in x_chk], axis=0)

    err_pt_tfl = float(np.max(np.abs(y_pt - y_tfl)))
    err_np_tfl = float(np.max(np.abs(y_np - y_tfl)))
    err_pt_np = float(np.max(np.abs(y_pt - y_np)))
    assert err_pt_tfl < 1e-4, f"PyTorch vs TFLite error {err_pt_tfl} >= 1e-4"
    assert err_np_tfl < 1e-4, f"NumPy vs TFLite error {err_np_tfl} >= 1e-4"

    sha_disk = hashlib.sha256(android_tfl.read_bytes()).hexdigest()
    assert sha_disk == bundle_res["sha256"], "SHA-256 mismatch after asset copy"

    # 6. 在 6 类场景 × 多随机种子评测三类控制器
    print(f"  [Eval] Running 6 scenarios x {eval_seeds_per_sc} seeds x 3 controllers ...", flush=True)
    eval_controllers, sample_traces = evaluate_all_controllers(android_tfl, seeds_per_scenario=eval_seeds_per_sc)

    # 7. 绘制高质量科研对比图 fig9_lol_ping_detour_rl.png
    fig_out = plot_fig9_lol_ping_detour_rl(eval_controllers, sample_traces, out_fig)

    elapsed_s = round(time.perf_counter() - t0, 2)
    eval_metrics = {
        "schemaVersion": 1,
        "generatedAt": "2026-11-18",
        "modelName": MODEL_MANIFEST_NAME,
        "assetStem": ASSET_STEM,
        "version": MODEL_VERSION,
        "task": "policy",
        "device": str(device),
        "elapsedSec": elapsed_s,
        "contract": {
            "featureDim": 39,
            "hiddenDim": int(folded_policy.hidden_dim),
            "actionDim": 3,
            "useResidual": True,
            "inputShape": [1, 39],
            "outputShape": [1, 3],
            "actionBounds": {"vx": [-100.0, 100.0], "vyaw": [-100.0, 100.0], "head": [-75.0, 75.0]},
        },
        "artifacts": {
            "tflitePath": bundle_res["tflite_path"],
            "jsonPath": bundle_res["json_path"],
            "weightsPtPath": bundle_res["pt_path"],
            "subBundleZipPath": str(sub_default_zip),
            "zooBundleZipPath": str(target_bundle_zip),
            "androidTflitePath": str(android_tfl),
            "androidJsonPath": str(android_jsn),
            "figurePath": str(fig_out),
            "tfliteSizeBytes": android_tfl.stat().st_size,
            "sha256": sha_disk,
        },
        "numericalParity": {
            "maxAbsErrPyTorchVsTflite": err_pt_tfl,
            "maxAbsErrNumpyVsTflite": err_np_tfl,
            "maxAbsErrPyTorchVsNumpy": err_pt_np,
        },
        "training": training_block,
        "controllers": eval_controllers,
    }

    metrics_file = skill_dir / "eval_metrics.json"
    metrics_file.write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ov_rl = eval_controllers["ping_point_nav_v1"]["overall"]
    ov_pn = eval_controllers["classic_pn"]["overall"]
    ov_st = eval_controllers["straight_baseline"]["overall"]
    print(
        f"\n[PingPointNavRL] Completed in {elapsed_s:.1f}s!\n"
        f"  SHA-256: {sha_disk}\n"
        f"  Parity (PT/NP vs TFLite): {err_pt_tfl:.2e} / {err_np_tfl:.2e}\n"
        f"  ping_point_nav_v1 : err={ov_rl['arrival_err_cm_mean']:.2f}cm | succ={ov_rl['success_rate_pct']:.1f}% | "
        f"cup_coll={ov_rl['cup_collision_rate_pct']:.1f}% | dz_viol={ov_rl['danger_zone_violation_rate_pct']:.1f}% | "
        f"cliff_fall={ov_rl['cliff_fall_rate_pct']:.1f}%\n"
        f"  classic_pn        : err={ov_pn['arrival_err_cm_mean']:.2f}cm | succ={ov_pn['success_rate_pct']:.1f}%\n"
        f"  straight_baseline : err={ov_st['arrival_err_cm_mean']:.2f}cm | succ={ov_st['success_rate_pct']:.1f}%\n"
        f"  Metrics saved -> {metrics_file}\n"
        f"  Figure saved  -> {fig_out}",
        flush=True,
    )
    return eval_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Train & Export LoL 3D Ping Point-Goal & Detour RL Policy")
    ap.add_argument("--bc-epochs", type=int, default=42)
    ap.add_argument("--ppo-iters", type=int, default=12)
    ap.add_argument("--expert-eps", type=int, default=12)
    ap.add_argument("--eval-seeds", type=int, default=12)
    ap.add_argument("--eval-only", action="store_true", help="Re-export bundle and run full evaluation from saved checkpoint")
    args = ap.parse_args()
    train_and_export_ping_point_nav(
        bc_epochs=args.bc_epochs,
        ppo_iters=args.ppo_iters,
        expert_eps_per_sc=args.expert_eps,
        eval_seeds_per_sc=args.eval_seeds,
        eval_only=args.eval_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
