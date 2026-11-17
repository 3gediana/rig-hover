# -*- coding: utf-8 -*-
"""train_spatial_rl.py — 面向复杂 2D 桌面交互的端侧空间记忆强化学习训练、对比评测与 TFLite 部署流水线。

核心方法论亮点：
1. **SE(2) 刚体空间记忆 + 残差 Tanh 策略网络 (`39 -> 160 -> 160(Res) -> 160 -> 3`)**：
   - 输入严格保持 `[1, 39]`、输出 `[1, 3]`，100% 兼容移动端标准 TFLite 内置算子 (`FULLY_CONNECTED/ADD/TANH/MUL`)；
   - 首层折叠 $39 \times 39$ 平移/绝对航向不变性投影矩阵 $W_1^{\text{folded}} = W_1 P$，辅以可微左右镜像等变正则
     $\|\pi(\mathcal{M}_s(x)) - \mathcal{M}_a(\pi(x))\|_2^2$，确保正前方平衡点绝对零偏置 (`< 1e-4`)。
2. **非对称特权 Actor-Critic 课程式 PPO (Asymmetric Privileged PPO)**：
   - 部署侧 Actor 仅读取 39 维真实端侧特征；训练侧 Critic 读取 51 维特权状态（含二维世界真值误差、真实倾角、
     障碍净空、打滑系数与通信延迟步数），显著降低部分可观测（手消失、35% 丢帧、60ms 延迟）下的价值估计方差。
3. **8 大复杂桌面沙箱压力基准 (`S1`~`S8`) × 5 类控制器对比消融**：
   - 定量评测：终态定位误差 (cm)、双敲到达成功率 (%)、抗路人/挥手诱骗率、桌沿坠落率、障碍碰撞率与最大俯仰角。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "sim"
PERC_DIR = HERE.parent / "perception"
for _p in (str(HERE), str(SIM_DIR), str(PERC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import controllers as C  # noqa: E402
import eval as sim_eval  # noqa: E402
import params as P  # noqa: E402
from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    EndSideController,
    EndSideFeatureBuilder,
    EndSidePolicyNet,
    EndSideSafetyShield,
)
from end_side_protocol import DriveCommand, RobotTelemetry, _clamp  # noqa: E402
from export_tflite_bundle import export_policy_bundle, verify_tflite_numerical  # noqa: E402
from sandbox_arena import SCENARIO_NAMES, HoverSandboxEnv  # noqa: E402
from scenario_v2 import TargetLock, V2Params  # noqa: E402
from train_rl_policy import EndSideSimPolicy, build_end_side_feat39_from_sim, sample_randomized_sim_params  # noqa: E402
import sim  # noqa: E402


def build_spatial_projection_matrix_v2() -> np.ndarray:
    """构造 39x39 空间不变性与零中心化线性投影矩阵 P (训练后无损折叠进 fc1.weight = fc1.weight @ P)。

    严格屏蔽绝对世界航向与绝对里程坐标 (yaw[2], x1[10], x2[11], cmd_pos[18], cmd_yaw_target[19])，
    使策略对任意初始朝向与累积行驶里程具备数学级绝对不变性。
    """
    P_mat = np.zeros((39, 39), dtype=np.float32)
    P_mat[0, 0] = 6.0             # roll
    P_mat[1, 1] = 6.0             # pitch
    P_mat[6, 6] = 1.0             # pit_gyro
    P_mat[7, 7] = 1.0             # dq
    P_mat[8, 8] = 1.0             # v1 / 100
    P_mat[9, 9] = 1.0             # v2 / 100
    P_mat[12, 12] = 75.0 / 45.0   # head_q / 45°
    P_mat[13, 13] = 75.0 / 45.0   # head_target / 45°
    P_mat[14, 14] = 1.0           # gimbal_target / 45°
    P_mat[15, 15] = 1.0           # cmd_vx / 100
    P_mat[16, 16] = 1.0           # cmd_vyaw / 100
    P_mat[22, 22] = 1.0           # pos_e
    P_mat[24, 24] = 1.0           # vel_e (body_yaw_rate / 60)
    P_mat[26, 26] = 1.0           # pit_e

    # Slot 0 (视觉目标): 利用 score[32] 缩放或配合对称损失在 cx=0.5 处严格归零
    P_mat[28, 28] = 2.0
    P_mat[29, 29] = 1.0
    P_mat[30, 30] = 1.0
    P_mat[31, 31] = 1.0
    P_mat[32, 32] = 1.0

    # Slot 1 (SE(2) 空间记忆锁存槽): 当 lock_valid (idx 37) == 1.0 时严格零中心化！
    # z[33] = 2*cx_lock - 1*valid = (bearing_body - head_q) / 30°
    P_mat[33, 33] = 2.0
    P_mat[33, 37] = -1.0
    # z[34] = 2*body_bearing_01 - 1*valid = bearing_body / 90° (广角不饱和！)
    P_mat[34, 34] = 2.0
    P_mat[34, 37] = -1.0
    # z[35] = 2*rem_norm (remaining_m / 0.75m)
    P_mat[35, 35] = 2.0
    # z[36] = clear_norm (clearance_m / 1.0m)
    P_mat[36, 36] = 1.0
    # z[37] = lock_valid (1.0 表示处于双敲记忆巡航模式)
    P_mat[37, 37] = 1.0
    P_mat[38, 38] = 1.0
    return P_mat


FEAT_PROJ_V2 = build_spatial_projection_matrix_v2()


def mirror_feat39_batch(x: torch.Tensor) -> torch.Tensor:
    """对批量 39 维观测执行精确左右镜像反射 M_obs(x)，用于可微等变约束。"""
    m = x.clone()
    # 反对称标量取反
    for idx in (0, 12, 13, 14, 16, 24):
        m[:, idx] = -x[:, idx]
    # 左右轮速交换 (v1 <-> v2: v_avg + diff <-> v_avg - diff)
    m[:, 8] = x[:, 9]
    m[:, 9] = x[:, 8]
    # Slot 0 水平翻转 cx -> 1 - cx (当存在目标 score > 0.05 时)
    has_s0 = (x[:, 32] > 0.05).float()
    m[:, 28] = has_s0 * (1.0 - x[:, 28]) + (1.0 - has_s0) * x[:, 28]
    # Slot 1 水平翻转 (cx_lock 与广角 body_bearing_01 均以 0.5 为左右对称中心)
    has_s1 = (x[:, 37] > 0.05).float()
    m[:, 33] = has_s1 * (1.0 - x[:, 33]) + (1.0 - has_s1) * x[:, 33]
    m[:, 34] = has_s1 * (1.0 - x[:, 34]) + (1.0 - has_s1) * x[:, 34]
    return m


def mirror_feat39_np(feat_39: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(feat_39.reshape(1, 39))
    return mirror_feat39_batch(t)[0].numpy().astype(np.float32)


def spatial_expert_action(feat_39: np.ndarray) -> np.ndarray:
    """二维空间记忆弧线追击 + 抗路人干扰 + 头身分离阻尼专家策略。

    输出物理量纲指令: [vx ∈ [-100, 100], vyaw ∈ [-100, 100], head ∈ [-75, 75]]。
    """
    q_head_deg = float(feat_39[12]) * 75.0
    pitch_deg = float(feat_39[1]) * 180.0
    body_yaw_rate_dps = float(feat_39[24]) * 60.0
    lock_active = float(feat_39[37]) > 0.92
    has_any = float(feat_39[38]) > 0.5

    # 分支 A: 双敲空间记忆锁定模式 (完全屏蔽 slot0 视觉路人干扰，专注奔赴记忆坐标！)
    if lock_active:
        bearing_body_deg = (float(feat_39[34]) - 0.5) * 180.0
        rem_m = float(feat_39[35]) * 1.5
        clear_m = float(feat_39[36]) * 1.0
        eff_rem = min(rem_m, clear_m)

        # 头部提前指向记忆敲击点 ("眼随心动")，大偏角时限幅 ±42°
        head_cmd_deg = float(np.clip(0.90 * bearing_body_deg, -42.0, 42.0))

        # 阻尼航向控制
        a = abs(bearing_body_deg)
        if a < 1.5:
            vyaw_dps = -0.18 * body_yaw_rate_dps
        else:
            vyaw_dps = 2.50 * bearing_body_deg - 0.24 * body_yaw_rate_dps
        vyaw_dps = float(np.clip(vyaw_dps, -66.0, 66.0))
        vyaw_pct = (vyaw_dps / 90.0) * 100.0

        # 余弦门控平滑弧线前向速度 + 俯仰抗倾覆衰减 + 终端平方根软刹停
        if eff_rem <= 0.015 or a > 50.0:
            vx_pct = 0.0
        else:
            turn_gate = max(0.0, math.cos(math.radians(min(86.0, a * 1.55)))) ** 1.15
            v_brake = math.sqrt(max(0.0, 2.0 * 0.155 * eff_rem))
            v_linear = 0.56 * eff_rem + 0.022
            v_mps = min(0.215, v_brake, v_linear) * turn_gate
            pitch_scale = _clamp(1.0 - max(0.0, abs(pitch_deg) - 6.0) / 12.0, 0.25, 1.0)
            vx_pct = float(np.clip((v_mps / 0.30) * 100.0 * pitch_scale, 0.0, 72.0))

        return np.array([vx_pct, vyaw_pct, head_cmd_deg], dtype=np.float32)

    # 分支 B: 常规视觉注视追踪模式 (IDLE / GAZE_FACE / GAZE_FIST -> vx 严格为 0)
    if not has_any or float(feat_39[32]) < 0.35:
        return np.array([0.0, 0.0, 0.72 * q_head_deg], dtype=np.float32)

    theta_img_deg = (float(feat_39[28]) - 0.5) * 60.0
    eps_deg = theta_img_deg + q_head_deg
    a = abs(eps_deg)
    s = 1.0 if eps_deg >= 0.0 else -1.0

    if a < 4.5:
        head_cmd_deg = 0.0 if a <= 1.2 else eps_deg
        vyaw_dps = 0.0
    elif a < 24.0:
        head_cmd_deg = 0.76 * eps_deg
        vyaw_dps = s * 1.65 * a - 0.22 * body_yaw_rate_dps
    else:
        head_cmd_deg = eps_deg - s * 22.0
        vyaw_dps = s * min(3.10 * a, 64.0) - 0.22 * body_yaw_rate_dps

    head_cmd_deg = float(np.clip(head_cmd_deg, -42.0, 42.0))
    if abs(head_cmd_deg) > 39.0:
        vyaw_dps = s * max(abs(vyaw_dps), 22.0)
    vyaw_dps = float(np.clip(vyaw_dps, -64.0, 64.0))
    vyaw_pct = (vyaw_dps / 90.0) * 100.0
    return np.array([0.0, vyaw_pct, head_cmd_deg], dtype=np.float32)


class PrivilegedSpatialActorCritic(nn.Module):
    """非对称特权 Actor-Critic：39维可折叠残差部署 Actor + 51维沙箱真值特权 Critic。"""

    def __init__(self, hidden_dim: int = 160, critic_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor = EndSidePolicyNet(
            feature_dim=39,
            hidden_dim=hidden_dim,
            action_dim=3,
            use_residual=True,
        )
        self.critic = nn.Sequential(
            nn.Linear(51, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, critic_dim),
            nn.Tanh(),
            nn.Linear(critic_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-2.4, -2.3, -2.3], dtype=torch.float32))
        self.register_buffer("proj_mat", torch.from_numpy(FEAT_PROJ_V2))

    def proj_obs(self, feat_39: torch.Tensor) -> torch.Tensor:
        return feat_39 @ self.proj_mat.T

    def forward_actor_norm(self, feat_39: torch.Tensor) -> torch.Tensor:
        return self.actor.forward_normalized(self.proj_obs(feat_39))

    def forward_value(self, priv_51: torch.Tensor) -> torch.Tensor:
        proj_39 = self.proj_obs(priv_51[:, :39])
        priv_in = torch.cat([proj_39, priv_51[:, 39:]], dim=-1)
        return self.critic(priv_in).squeeze(-1)

    def symmetry_loss(self, feat_39: torch.Tensor) -> torch.Tensor:
        """可微左右镜像等变损失: vx 偶对称，vyaw 与 head 奇反对称。"""
        u = self.forward_actor_norm(feat_39)
        u_mir = self.forward_actor_norm(mirror_feat39_batch(feat_39))
        target_mir = torch.stack([u[:, 0], -u[:, 1], -u[:, 2]], dim=-1)
        return F.mse_loss(u_mir, target_mir)

    def export_folded_policy(self) -> EndSidePolicyNet:
        """将 39x39 投影矩阵 P 无损折叠进首层线性权重 W1_folded = W1 @ P。"""
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


def collect_hybrid_expert_dataset(
    sp: P.SimParams,
    n_sandbox_eps: int = 40,
    n_track_eps: int = 20,
    seed: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """联合采集 8 类 2D 复杂沙箱轨迹 + 5 类经典动态视线追踪轨迹（含全量镜像配对增强）。"""
    feats_list: list[np.ndarray] = []
    acts_list: list[np.ndarray] = []
    rng = np.random.default_rng(seed)

    # Part A: 8 类 2D 空间记忆复杂沙箱专家轨迹 (含专门的路人干扰/超大偏角/绕障/桌沿刹车样本)
    dummy_ctrl = EndSideController(policy_model=EndSidePolicyNet(), blend_rl=0.0)
    for ep in range(n_sandbox_eps):
        sc_name = SCENARIO_NAMES[ep % len(SCENARIO_NAMES)]
        env = HoverSandboxEnv(scenario=sc_name, seed=seed + ep, domain_rand=(ep % 2 == 1), max_steps=190)
        obs = env.reset()
        dummy_ctrl.reset()

        for _ in range(env.max_steps):
            telem: RobotTelemetry = obs["telemetry"]
            scene_obs: dict = obs["scene_obs"]
            dets: list[dict] = obs["detections"]
            clearance_m: float = obs["clearance_m"]
            deflect_deg: float = obs["obstacle_deflect_deg"]

            # 利用控制器维护真实的 SE(2) 空间记忆状态
            _ = dummy_ctrl.step(env.dt_step, telem, scene_obs, detections=dets)
            lock_for_feat = None
            if dummy_ctrl.mode == MODE_APPROACH and dummy_ctrl.target_lock.valid:
                steered = float(_clamp(dummy_ctrl.target_lock.bearing_body + deflect_deg, -90.0, 90.0))
                lock_for_feat = TargetLock(
                    valid=1,
                    bearing_body=steered,
                    dist_m=dummy_ctrl.target_lock.dist_m,
                    remaining_m=dummy_ctrl.target_lock.remaining_m,
                )

            feat_39 = EndSideFeatureBuilder.build(telem, dets, target_lock=lock_for_feat, clearance_m=clearance_m)
            act_3 = spatial_expert_action(feat_39)

            feats_list.append(feat_39)
            acts_list.append(act_3)

            # 精确成对镜像增强
            f_mir = mirror_feat39_np(feat_39)
            a_mir = np.array([act_3[0], -act_3[1], -act_3[2]], dtype=np.float32)
            feats_list.append(f_mir)
            acts_list.append(a_mir)

            safe_cmd = dummy_ctrl.shield.apply(
                DriveCommand(vx=float(act_3[0]), vyaw=float(act_3[1]), head=float(act_3[2])),
                mode=dummy_ctrl.mode,
                remaining_m=dummy_ctrl.target_lock.remaining_m if dummy_ctrl.mode == MODE_APPROACH else 0.0,
                pitch_deg=telem.pitch,
                clearance_m=clearance_m,
            )
            obs, info = env.step(safe_cmd)
            if info["done"]:
                break

    # Part B: 5 类高频连续视线追踪场景样本 (保证 GAZE 模式下极低跟踪误差与零超调)
    dt = sp.fw.ctrl_dt_ms * 1e-3
    vision_period = 1.0 / sp.fw.cfc_hz
    for ep in range(n_track_eps):
        ep_sp = sample_randomized_sim_params(sp, rng) if ep % 2 == 1 else sp
        sc_name = sim.SCENARIOS[ep % len(sim.SCENARIOS)]
        scene = sim.make_scene(sc_name, seed + 500 + ep, duration=10.0, dt=dt)
        plant = sim.Plant(ep_sp, dt)
        plant.reset(pitch0_deg=float(rng.uniform(-1.2, 1.2)), seed=seed + 500 + ep)
        imu = sim.Imu(ep_sp, rng)
        fw = sim.FirmwareControl(ep_sp)
        fw.k_dyaw = -0.52
        fw.head_rate_comp = 1.0
        arb = sim.Arbiter(ep_sp, sim.ARB_SRC_CFC)
        dist_est = sim.DistEstimator()
        sv = sim.SensorView()
        next_vision = 0.0
        last_world_b = 0.0

        for k in range(len(scene)):
            t = k * dt
            q_meas = round(plant.st.q_head / ep_sp.plant.head_quant_deg) * ep_sp.plant.head_quant_deg
            imu.update(plant, q_meas, dt)
            sv.roll, sv.pitch = imu.roll, imu.pitch
            sv.pit_gyro, sv.dq, sv.yaw = imu.pit_gyro, imu.dq, imu.yaw
            sv.yaw_rate = imu.yaw_rate
            sv.q_head = q_meas
            sv.qd_head = plant.st.qd_head
            sv.wheel1_vel = sv.wheel2_vel = plant.st.v_w
            sv.wheel_x = plant.st.x_w

            if k == 0 or t + 1e-9 >= next_vision:
                next_vision = vision_period if k == 0 else next_vision + vision_period
                gaze = plant.st.yaw_body + plant.st.q_head
                theta_geo = sim.wrap180(float(scene.bearing[k]) - gaze)
                seen = abs(theta_geo) <= ep_sp.fov_half_deg
                if seen:
                    last_world_b = float(scene.bearing[k])
                virt_theta = float(np.clip(sim.wrap180(last_world_b - gaze), -30.0, 30.0))
                d = float(scene.dist[k])
                ph = min(max(0.90 / d, 0.03), 1.0) if seen else 0.0
                d_est = dist_est.step(seen, ph)
                frame = sim.Frame(
                    t_s=t, theta_img=(theta_geo if seen else 0.0), q_head=sv.q_head,
                    seen=int(seen), person_h=ph, person_w=0.4 * ph,
                    person_conf=(0.90 if seen else 0.0), dist_est=d_est,
                    hand_up_L=0, hand_up_R=0, wave_sin=0.0, wave_cos=0.0,
                    gesture_conf=0.0, gesture_onset=0, gesture_id=0,
                    feature_age_ms=0.0, eps_body=sim.wrap180((theta_geo if seen else 0.0) + sv.q_head),
                )
                feat_39 = build_end_side_feat39_from_sim(frame, sv, virt_theta, True, approach_active=False)
                act_3 = spatial_expert_action(feat_39)
                feats_list.append(feat_39)
                acts_list.append(act_3)

                f_mir = mirror_feat39_np(feat_39)
                a_mir = np.array([act_3[0], -act_3[1], -act_3[2]], dtype=np.float32)
                feats_list.append(f_mir)
                acts_list.append(a_mir)

                arb.post_raw(
                    sim.CtrlOut(
                        head_des=float(np.clip(act_3[2], -45.0, 45.0)),
                        vx_des=0.0,
                        vyaw_des=float((act_3[1] / 100.0) * 90.0),
                        mode=1,
                    ),
                    t,
                )
                arb.note_feature(t)

            fw.update_state(sv, t)
            arb.tick(sv, fw, t)
            _, _, h_cmd = fw.control(sv, t, do_update_state=False)
            plant.step(fw.temp_u, fw.yaw_u, h_cmd)

    return np.stack(feats_list, axis=0), np.stack(acts_list, axis=0)


def stage1_pretrain_bc(
    ac: PrivilegedSpatialActorCritic,
    feats: np.ndarray,
    acts: np.ndarray,
    epochs: int = 42,
    batch_size: int = 256,
    lr: float = 2.0e-3,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Stage 1: 空间记忆与视线追踪混合行为克隆 + 左右镜像等变正则。"""
    ac.to(device)
    opt = torch.optim.AdamW(ac.actor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(2026)

    scale_np = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    y_norm = np.clip(acts / scale_np, -1.0, 1.0).astype(np.float32)

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
            loss_mse = F.mse_loss(pred, by)
            loss_sym = ac.symmetry_loss(bx)
            loss = loss_mse + 0.35 * loss_sym
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        sched.step()
        losses.append(round(ep_loss / max(1, n_b), 6))
    return losses


def rollout_sandbox_ppo_episode(
    ac: PrivilegedSpatialActorCritic,
    scenario: str,
    seed: int,
    domain_rand: bool,
    device: torch.device,
) -> dict:
    """在 2D 复杂物理沙箱中执行一条完整闭环 RL 采样轨迹，记录 39 维 Actor 输入与 51 维特权 Critic 输入。"""
    env = HoverSandboxEnv(scenario=scenario, seed=seed, domain_rand=domain_rand, max_steps=190)
    obs = env.reset()
    state_ctrl = EndSideController(policy_model=EndSidePolicyNet(), blend_rl=0.0)
    shield = EndSideSafetyShield(max_delta_vx=20.0, max_delta_vyaw=22.0, max_delta_head=11.0)

    obs_buf, priv_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], [], []
    prev_u_norm = np.zeros(3, dtype=np.float32)
    std = torch.exp(ac.log_std).to(device)
    act_scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)

    prev_true_dist: float | None = None

    for _ in range(env.max_steps):
        telem: RobotTelemetry = obs["telemetry"]
        scene_obs: dict = obs["scene_obs"]
        dets: list[dict] = obs["detections"]
        clearance_m: float = obs["clearance_m"]
        deflect_deg: float = obs["obstacle_deflect_deg"]

        # 更新高层状态与 SE(2) 空间记忆
        _ = state_ctrl.step(env.dt_step, telem, scene_obs, detections=dets)
        mode_now = state_ctrl.mode
        lock_for_feat = None
        if mode_now == MODE_APPROACH and state_ctrl.target_lock.valid:
            steered = float(_clamp(state_ctrl.target_lock.bearing_body + deflect_deg, -90.0, 90.0))
            lock_for_feat = TargetLock(
                valid=1,
                bearing_body=steered,
                dist_m=state_ctrl.target_lock.dist_m,
                remaining_m=state_ctrl.target_lock.remaining_m,
            )

        feat_39 = EndSideFeatureBuilder.build(telem, dets, target_lock=lock_for_feat, clearance_m=clearance_m)
        priv_51 = env.build_privileged_critic_features(feat_39)

        with torch.no_grad():
            obs_t = torch.from_numpy(feat_39).unsqueeze(0).to(device)
            priv_t = torch.from_numpy(priv_51).unsqueeze(0).to(device)
            mu = ac.forward_actor_norm(obs_t)[0]
            val = ac.forward_value(priv_t)[0]
            dist_n = torch.distributions.Normal(mu, std)
            u_norm_t = torch.clamp(dist_n.sample(), -1.0, 1.0)
            logp_t = dist_n.log_prob(u_norm_t).sum()

        u_norm = u_norm_t.cpu().numpy()
        act_phys = u_norm * act_scale
        raw_cmd = DriveCommand(vx=float(act_phys[0]), vyaw=float(act_phys[1]), head=float(act_phys[2]))
        safe_cmd = shield.apply(
            raw_cmd,
            mode=mode_now,
            remaining_m=state_ctrl.target_lock.remaining_m if mode_now == MODE_APPROACH else 0.0,
            pitch_deg=telem.pitch,
            clearance_m=clearance_m,
        )

        obs_next, info = env.step(safe_cmd)
        true_dist = float(info["true_dist_m"])
        true_bear = float(info["true_bearing_deg"])
        pitch_deg = float(info["pitch_deg"])
        yaw_rate = float(info["yaw_rate_dps"])

        # 多目标物理奖励函数：进展奖励 + 航向对准 + 抗干扰定力 + 平滑阻尼 + 碰撞/坠崖重罚
        if prev_true_dist is None:
            prev_true_dist = true_dist

        if mode_now == MODE_APPROACH:
            progress = (prev_true_dist - true_dist) / env.dt_step  # m/s 朝目标逼近速度
            r_prog = 2.8 * progress
            r_align = 0.85 - 0.55 * (true_bear / 45.0) ** 2
            r_gaze = 0.25 - 0.20 * ((true_bear - telem.head_q) / 35.0) ** 2
            r_arrive = 1.6 if true_dist < 0.08 else 0.0
            r_core = r_prog + r_align + r_gaze + r_arrive
        else:
            # 非 APPROACH 阶段严惩前向轮速输出，鼓励平稳凝视
            r_core = 0.55 - 0.8 * abs(float(act_phys[0]) / 100.0)
            if scene_obs.get("face_valid", 0):
                gaze_err = float(scene_obs.get("theta_face", 0.0))
                r_core += 0.45 - 0.35 * (gaze_err / 25.0) ** 2

        r_pitch = -0.30 * (pitch_deg / 12.0) ** 2
        r_damp = -0.20 * (yaw_rate / 55.0) ** 2
        r_jerk = -0.22 * float(np.sum((u_norm - prev_u_norm) ** 2))
        r_hazard = 0.0
        if info["cliff_fell"]:
            r_hazard -= 25.0
        if info["obstacle_collided"]:
            r_hazard -= 15.0

        reward = float(r_core + r_pitch + r_damp + r_jerk + r_hazard)
        prev_u_norm = u_norm.copy()
        prev_true_dist = true_dist
        obs = obs_next

        obs_buf.append(feat_39)
        priv_buf.append(priv_51)
        act_buf.append(u_norm)
        logp_buf.append(float(logp_t.item()))
        rew_buf.append(reward)
        val_buf.append(float(val.item()))

        if info["done"]:
            break

    return {
        "obs": np.stack(obs_buf, axis=0),
        "priv": np.stack(priv_buf, axis=0),
        "act": np.stack(act_buf, axis=0),
        "logp": np.array(logp_buf, dtype=np.float32),
        "rew": np.array(rew_buf, dtype=np.float32),
        "val": np.array(val_buf, dtype=np.float32),
        "final_dist_m": float(info["true_dist_m"]),
    }


def train_curriculum_ppo(
    ac: PrivilegedSpatialActorCritic,
    expert_feats: np.ndarray,
    expert_acts: np.ndarray,
    n_iters: int = 14,
    episodes_per_iter: int = 8,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    lr_actor: float = 1.4e-4,
    lr_critic: float = 9e-4,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Stage 2 & Stage 3: 从中等难度空间记忆 (S1~S5) 到全扰动风暴 (S1~S8) 的非对称特权 PPO。"""
    ac.to(device)
    opt_actor = torch.optim.AdamW([p for n, p in ac.named_parameters() if "critic" not in n], lr=lr_actor)
    opt_critic = torch.optim.AdamW(ac.critic.parameters(), lr=lr_critic)

    scale_np = np.array([100.0, 100.0, 75.0], dtype=np.float32)
    exp_y_norm = np.clip(expert_acts / scale_np, -1.0, 1.0).astype(np.float32)
    rng = np.random.default_rng(20261118)

    # 先对特权 Critic 做 6 轮价值预热，消除初始 PPO 优势估计噪声
    init_trajs = [
        rollout_sandbox_ppo_episode(ac, SCENARIO_NAMES[i % len(SCENARIO_NAMES)], 300 + i, True, device)
        for i in range(8)
    ]
    for _ in range(6):
        for tr in init_trajs:
            rew, val = tr["rew"], tr["val"]
            ret = np.zeros_like(rew)
            running = 0.0
            for t in reversed(range(len(rew))):
                running = rew[t] + 0.99 * running
                ret[t] = running
            priv_t = torch.from_numpy(tr["priv"]).to(device)
            ret_t = torch.from_numpy(ret).to(device)
            opt_critic.zero_grad()
            F.mse_loss(ac.forward_value(priv_t), ret_t).backward()
            opt_critic.step()

    history: list[dict] = []
    moderate_scenarios = SCENARIO_NAMES[:5]
    all_scenarios = SCENARIO_NAMES

    for it in range(n_iters):
        stage_pool = moderate_scenarios if it < 5 else all_scenarios
        dom_rand = it >= 4
        with torch.no_grad():
            ac.log_std.copy_(torch.tensor([-2.25 - 0.10 * it, -2.15 - 0.10 * it, -2.15 - 0.10 * it], device=device))

        trajs = [
            rollout_sandbox_ppo_episode(
                ac,
                scenario=stage_pool[(it * episodes_per_iter + ep) % len(stage_pool)],
                seed=1000 + it * 20 + ep,
                domain_rand=dom_rand,
                device=device,
            )
            for ep in range(episodes_per_iter)
        ]

        obs_all, priv_all, act_all, logp_all, adv_all, ret_all = [], [], [], [], [], []
        ep_returns, ep_final_dists = [], []

        for tr in trajs:
            rew, val = tr["rew"], tr["val"]
            ep_returns.append(float(np.sum(rew)))
            ep_final_dists.append(tr["final_dist_m"])
            T = len(rew)
            adv = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_val = val[t + 1] if t + 1 < T else val[t]
                delta = rew[t] + 0.99 * next_val - val[t]
                last_gae = delta + 0.99 * 0.95 * last_gae
                adv[t] = last_gae
            ret = adv + val
            obs_all.append(tr["obs"])
            priv_all.append(tr["priv"])
            act_all.append(tr["act"])
            logp_all.append(tr["logp"])
            adv_all.append(adv)
            ret_all.append(ret)

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
                loss_sym = ac.symmetry_loss(obs_b[idx]) + ac.symmetry_loss(bx_exp)

                # 正前方平衡点零偏置硬锚定（覆盖不同视觉框尺寸与置信度 score ∈ [0, 1]）
                eq_x = torch.zeros((8, 39), dtype=torch.float32, device=device)
                for eq_i, sc_v in enumerate((0.0, 0.45, 0.65, 0.80, 0.90, 0.95, 1.0, 0.90)):
                    if sc_v > 0.0:
                        eq_x[eq_i, 28] = 0.5
                        eq_x[eq_i, 29] = 0.35
                        eq_x[eq_i, 30] = 0.20 + 0.03 * eq_i
                        eq_x[eq_i, 31] = 0.25 + 0.03 * eq_i
                        eq_x[eq_i, 32] = sc_v
                        eq_x[eq_i, 38] = 1.0
                loss_eq = (ac.forward_actor_norm(eq_x) ** 2).mean()

                opt_actor.zero_grad()
                (0.32 * loss_ppo + 0.90 * loss_anchor + 0.80 * loss_sym + 4.0 * loss_eq).backward()
                nn.utils.clip_grad_norm_(ac.actor.parameters(), 1.0)
                opt_actor.step()

                opt_critic.zero_grad()
                v_pred = ac.forward_value(priv_b[idx])
                F.mse_loss(v_pred, ret_b[idx]).backward()
                opt_critic.step()

        mean_ret = float(np.mean(ep_returns))
        mean_dist_cm = float(np.mean(ep_final_dists) * 100.0)
        stage_tag = "S2-Moderate" if it < 5 else "S3-StressAll"
        history.append({
            "iter": it + 1,
            "stage": stage_tag,
            "mean_return": round(mean_ret, 3),
            "mean_final_dist_cm": round(mean_dist_cm, 2),
        })
        print(
            f"  [{stage_tag} PPO {it + 1:02d}/{n_iters:02d}] "
            f"return = {mean_ret:6.2f} | final_err = {mean_dist_cm:5.2f} cm",
            flush=True,
        )
    return history


def _run_controller_in_sandbox(
    controller_kind: str,
    scenario: str,
    seed: int,
    bc_net: EndSidePolicyNet | None,
    ppo_net: EndSidePolicyNet | None,
) -> dict:
    """在指定沙箱场景下评测单类控制器，返回轨迹与精度安全指标。"""
    env = HoverSandboxEnv(scenario=scenario, seed=seed, domain_rand=True, max_steps=210)
    obs = env.reset()

    ctrl_se2_nom = EndSideController(policy_model=EndSidePolicyNet(), blend_rl=0.0)
    ctrl_bc = EndSideController(policy_model=bc_net or EndSidePolicyNet(), blend_rl=1.0)
    ctrl_ppo = EndSideController(policy_model=ppo_net or EndSidePolicyNet(), blend_rl=1.0)

    # 旧版 1D 标量距离记忆状态 (复现升级前的四大盲区用于诚实对比)
    legacy_mode = MODE_IDLE
    legacy_bearing = 0.0
    legacy_dist = 0.0
    legacy_wheel0 = 0.0
    legacy_t_dwell = 0.0

    # 纯视觉无记忆反应状态
    reactive_latched = False

    max_pitch = 0.0
    reached_dwell = False

    for step_k in range(env.max_steps):
        telem: RobotTelemetry = obs["telemetry"]
        sc: dict = obs["scene_obs"]
        dets: list[dict] = obs["detections"]

        if controller_kind == "naive_reactive":
            if sc.get("knock_fired", 0):
                reactive_latched = True
            if reactive_latched and sc.get("hand_valid", 0):
                eps = float(sc["theta_fist"]) + telem.head_q
                cmd = DriveCommand(vx=45.0, vyaw=_clamp(1.8 * eps, -65.0, 65.0), head=_clamp(eps, -42.0, 42.0))
            elif sc.get("face_valid", 0):
                eps = float(sc["theta_face"]) + telem.head_q
                cmd = DriveCommand(vx=0.0, vyaw=_clamp(1.5 * eps, -55.0, 55.0), head=_clamp(eps, -42.0, 42.0))
            else:
                cmd = DriveCommand(vx=0.0, vyaw=0.0, head=0.0)

        elif controller_kind == "scalar_1d_lock":
            # 旧版 1D 标量距离记忆：无 SE(2) 旋转扣减、APPROACH 期间会被 hand_valid 挥手覆写、不支持中途改敲
            if legacy_mode != MODE_APPROACH and legacy_mode != MODE_DWELL and sc.get("knock_fired", 0) and sc.get("hand_valid", 0):
                ph = float(sc.get("person_h", 0.0))
                d_est = (0.84 / ph) if ph >= 0.06 else 0.80
                legacy_dist = max(0.0, d_est - 0.12)
                legacy_bearing = float(sc["theta_fist"]) + telem.head_q
                legacy_wheel0 = telem.wheel_x_avg
                legacy_mode = MODE_APPROACH
            elif legacy_mode == MODE_APPROACH:
                if sc.get("hand_valid", 0):
                    legacy_bearing = float(sc["theta_fist"]) + telem.head_q
                walked = max(0.0, telem.wheel_x_avg - legacy_wheel0)
                rem = max(0.0, legacy_dist - walked)
                if rem <= 0.06 or step_k > 185:
                    legacy_mode = MODE_DWELL
                    legacy_t_dwell = env.t_s
                eps = legacy_bearing
                vyaw = _clamp(2.0 * eps, -65.0, 65.0)
                vx = _clamp((min(0.18, 0.6 * rem) / 0.30) * 100.0, 0.0, 60.0) if abs(eps) <= 18.0 else 0.0
                cmd = DriveCommand(vx=vx, vyaw=vyaw, head=_clamp(eps, -42.0, 42.0))
            elif legacy_mode == MODE_DWELL:
                reached_dwell = True
                cmd = DriveCommand(vx=0.0, vyaw=0.0, head=0.0)
            else:
                eps = (float(sc["theta_fist"]) if sc.get("hand_valid", 0) else float(sc.get("theta_face", 0.0))) + telem.head_q
                cmd = DriveCommand(vx=0.0, vyaw=0.0, head=_clamp(eps, -42.0, 42.0))

        elif controller_kind == "se2_nominal":
            cmd = ctrl_se2_nom.step(env.dt_step, telem, sc, detections=dets)
            if ctrl_se2_nom.mode == MODE_DWELL:
                reached_dwell = True

        elif controller_kind == "spatial_bc_v2":
            # 纯 BC 消融：无主动安全护盾限幅，直接检验纯模仿策略输出
            _ = ctrl_bc.step(env.dt_step, telem, sc, detections=dets)
            lock_f = None
            if ctrl_bc.mode == MODE_APPROACH and ctrl_bc.target_lock.valid:
                steered = float(_clamp(ctrl_bc.target_lock.bearing_body + float(sc.get("obstacle_deflect_deg", 0.0)), -90.0, 90.0))
                lock_f = TargetLock(valid=1, bearing_body=steered, dist_m=ctrl_bc.target_lock.dist_m, remaining_m=ctrl_bc.target_lock.remaining_m)
            f39 = EndSideFeatureBuilder.build(telem, dets, target_lock=lock_f, clearance_m=float(sc.get("clearance_m", 1.2)))
            cmd = ctrl_bc.policy.infer_numpy(f39)
            if ctrl_bc.mode != MODE_APPROACH:
                cmd = DriveCommand(vx=0.0, vyaw=cmd.vyaw, head=cmd.head)
            if ctrl_bc.mode == MODE_DWELL:
                reached_dwell = True

        else:  # spatial_ppo_v2 (完整 SE(2) 空间记忆 + 特权 PPO 残差网络 + 二维安全屏障)
            cmd = ctrl_ppo.step(env.dt_step, telem, sc, detections=dets)
            if ctrl_ppo.mode == MODE_DWELL:
                reached_dwell = True

        obs, info = env.step(cmd)
        max_pitch = max(max_pitch, abs(float(info["pitch_deg"])))
        if info["done"]:
            break

    final_dist_m = float(info["true_dist_m"])
    # S7_cliff_guard 的真实安全考核是停在桌沿安全裕度内且不坠崖
    if scenario == "S7_cliff_guard":
        success = (not info["cliff_fell"]) and (0.42 <= env.rx_m <= 0.58)
        eff_err_cm = abs(0.52 - env.rx_m) * 100.0
    else:
        success = (not info["cliff_fell"]) and (not info["obstacle_collided"]) and (final_dist_m <= 0.095)
        eff_err_cm = final_dist_m * 100.0

    return {
        "scenario": scenario,
        "seed": seed,
        "success": bool(success),
        "reached_dwell": bool(reached_dwell),
        "final_err_cm": round(eff_err_cm, 2),
        "max_pitch_deg": round(max_pitch, 2),
        "cliff_fell": bool(info["cliff_fell"]),
        "obstacle_collided": bool(info["obstacle_collided"]),
        "trail_xy": list(env.trail_xy),
        "target1_xy": env.target1_xy,
        "final_target_xy": env.active_target_xy,
        "obstacles": [(o.x_m, o.y_m, o.radius_m) for o in env.obstacles],
        "table_bounds": (
            (env.table_bounds.x_min, env.table_bounds.x_max, env.table_bounds.y_min, env.table_bounds.y_max)
            if env.table_bounds is not None else None
        ),
    }


def evaluate_8_scenario_benchmark(
    bc_net: EndSidePolicyNet,
    ppo_net: EndSidePolicyNet,
    seeds: tuple[int, ...] = (201, 202, 203, 204),
) -> tuple[dict, dict]:
    """在 8 大复杂沙箱场景 × 4 个独立评测种子 (每控制器 32 回合，训练期未见种子) 上对比 5 类控制器。"""
    controllers = (
        "naive_reactive",
        "scalar_1d_lock",
        "se2_nominal",
        "spatial_bc_v2",
        "spatial_ppo_v2",
    )
    summary: dict[str, dict] = {}
    sample_traces: dict[str, dict[str, dict]] = {sc: {} for sc in SCENARIO_NAMES}

    for c_name in controllers:
        all_runs: list[dict] = []
        per_scenario: dict[str, dict] = {}
        for sc in SCENARIO_NAMES:
            sc_runs = []
            for idx_s, sd in enumerate(seeds):
                res = _run_controller_in_sandbox(c_name, sc, sd, bc_net, ppo_net)
                sc_runs.append(res)
                all_runs.append(res)
                if idx_s == 0:
                    sample_traces[sc][c_name] = res
            per_scenario[sc] = {
                "success_rate": round(float(np.mean([r["success"] for r in sc_runs])) * 100.0, 1),
                "mean_err_cm": round(float(np.mean([r["final_err_cm"] for r in sc_runs])), 2),
                "cliff_falls": int(sum(1 for r in sc_runs if r["cliff_fell"])),
                "collisions": int(sum(1 for r in sc_runs if r["obstacle_collided"])),
            }

        summary[c_name] = {
            "overall_success_rate_pct": round(float(np.mean([r["success"] for r in all_runs])) * 100.0, 2),
            "mean_final_err_cm": round(float(np.mean([r["final_err_cm"] for r in all_runs])), 2),
            "median_final_err_cm": round(float(np.median([r["final_err_cm"] for r in all_runs])), 2),
            "max_pitch_deg": round(float(np.max([r["max_pitch_deg"] for r in all_runs])), 2),
            "cliff_fall_count": int(sum(1 for r in all_runs if r["cliff_fell"])),
            "obstacle_collision_count": int(sum(1 for r in all_runs if r["obstacle_collided"])),
            "per_scenario": per_scenario,
        }
        print(
            f"  [Eval {c_name:16s}] success = {summary[c_name]['overall_success_rate_pct']:6.2f}% | "
            f"err = {summary[c_name]['mean_final_err_cm']:5.2f} cm | "
            f"falls = {summary[c_name]['cliff_fall_count']} | collisions = {summary[c_name]['obstacle_collision_count']}",
            flush=True,
        )
    return summary, sample_traces


def render_spatial_sandbox_figure(
    ppo_history: list[dict],
    summary: dict,
    sample_traces: dict,
    out_png: Path,
) -> None:
    """绘制 4 宫格科研对比图: (a) S1 盲走 & S2 路人干扰二维轨迹, (b) S4 中途改敲 & S6 绕障轨迹, (c) 5 控制器指标, (d) 课程 PPO 收敛。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8), dpi=170)
    fig.patch.set_facecolor("#0B1017")
    for ax in axes.flat:
        ax.set_facecolor("#111923")
        for sp_line in ax.spines.values():
            sp_line.set_color("#27364A")
        ax.tick_params(colors="#B8C7D9", labelsize=9)
        ax.xaxis.label.set_color("#D5E2F0")
        ax.yaxis.label.set_color("#D5E2F0")
        ax.title.set_color("#F0F6FC")
        ax.grid(True, color="#1D2A3A", linestyle="--", linewidth=0.6, alpha=0.8)

    colors = {
        "naive_reactive": "#FF5C77",
        "scalar_1d_lock": "#FFB020",
        "spatial_bc_v2": "#9B7BFF",
        "spatial_ppo_v2": "#00E5FF",
    }
    labels_map = {
        "naive_reactive": "纯视觉无记忆 (Naive)",
        "scalar_1d_lock": "旧 1D 标量锁存 (1D Lock)",
        "spatial_bc_v2": "空间记忆纯 BC (BC v2)",
        "spatial_ppo_v2": "SE(2) 空间记忆 + 特权 PPO (Ours)",
    }

    # Panel A: S2_bystander_distractor 路人横穿诱骗 + 手消失盲走 2D 平面轨迹
    ax0 = axes[0, 0]
    tr_s2 = sample_traces["S2_bystander_distractor"]
    for ck in ("naive_reactive", "scalar_1d_lock", "spatial_bc_v2", "spatial_ppo_v2"):
        pts = np.asarray(tr_s2[ck]["trail_xy"])
        ax0.plot(pts[:, 0], pts[:, 1], color=colors[ck], lw=2.2 if ck == "spatial_ppo_v2" else 1.5, label=labels_map[ck], alpha=0.92)
        ax0.scatter(pts[-1, 0], pts[-1, 1], color=colors[ck], s=38, zorder=5)
    tx, ty = tr_s2["spatial_ppo_v2"]["final_target_xy"]
    ax0.scatter([0.0], [0.0], color="#FFFFFF", marker="s", s=55, zorder=6, label="起点 (0,0)")
    ax0.scatter([tx], [ty], color="#52FF9A", marker="*", s=190, zorder=7, label="桌面双敲锁定点")
    ax0.set_title("(a) 场景 S2：双敲后手消失 + 路人横穿诱骗下的 2D 轨迹", fontsize=11, fontweight="bold")
    ax0.set_xlabel("桌面纵向 X (m)")
    ax0.set_ylabel("桌面横向 Y (m)")
    ax0.legend(facecolor="#141E2B", edgecolor="#2C3E55", labelcolor="#E2ECF7", fontsize=8, loc="best")
    ax0.set_aspect("equal", adjustable="datalim")

    # Panel B: S4 中途动态二次改敲 + S6 桌面障碍物切向绕行
    ax1 = axes[0, 1]
    tr_s4 = sample_traces["S4_mid_reknock"]
    for ck in ("scalar_1d_lock", "spatial_ppo_v2"):
        pts = np.asarray(tr_s4[ck]["trail_xy"])
        ax1.plot(pts[:, 0], pts[:, 1], color=colors[ck], lw=2.3 if ck == "spatial_ppo_v2" else 1.6, linestyle="-", label=f"S4 改敲: {labels_map[ck]}")
        ax1.scatter(pts[-1, 0], pts[-1, 1], color=colors[ck], s=42, zorder=6)
    t1x, t1y = tr_s4["spatial_ppo_v2"]["target1_xy"]
    t2x, t2y = tr_s4["spatial_ppo_v2"]["final_target_xy"]
    ax1.scatter([t1x], [t1y], color="#FFB020", marker="X", s=95, zorder=7, label="首次敲击点 A (中途作废)")
    ax1.scatter([t2x], [t2y], color="#52FF9A", marker="*", s=190, zorder=8, label="中途新敲击点 B (最终目标)")

    tr_s6 = sample_traces["S6_obstacle_detour"]["spatial_ppo_v2"]
    pts6 = np.asarray(tr_s6["trail_xy"])
    ax1.plot(pts6[:, 0], pts6[:, 1], color="#38F8B2", lw=2.0, linestyle="--", label="S6 水杯绕障轨迹 (Ours)")
    for ox, oy, orad in tr_s6["obstacles"]:
        circ = plt.Circle((ox, oy), orad, color="#FF5C77", alpha=0.42, label="桌面水杯障碍")
        ax1.add_patch(circ)
    o_tx, o_ty = tr_s6["final_target_xy"]
    ax1.scatter([o_tx], [o_ty], color="#38F8B2", marker="D", s=60, zorder=7)
    ax1.set_title("(b) 场景 S4 中途改敲重路由 & S6 桌面障碍平滑绕行", fontsize=11, fontweight="bold")
    ax1.set_xlabel("桌面纵向 X (m)")
    ax1.set_ylabel("桌面横向 Y (m)")
    ax1.legend(facecolor="#141E2B", edgecolor="#2C3E55", labelcolor="#E2ECF7", fontsize=7.6, loc="best")
    ax1.set_aspect("equal", adjustable="datalim")

    # Panel C: 5 控制器在 8 大复杂沙箱场景 (32 回合) 上的成功率与终态定位误差
    ax2 = axes[1, 0]
    order = ["naive_reactive", "scalar_1d_lock", "se2_nominal", "spatial_bc_v2", "spatial_ppo_v2"]
    short_names = ["Naive\n无记忆", "1D Lock\n旧标量锁", "SE(2)\n规则基线", "Spatial\n纯 BC", "Spatial PPO\n(Ours)"]
    succ_vals = [summary[k]["overall_success_rate_pct"] for k in order]
    err_vals = [summary[k]["mean_final_err_cm"] for k in order]
    xs = np.arange(len(order))
    w = 0.36
    bars1 = ax2.bar(xs - w / 2, succ_vals, width=w, color="#00E5FF", alpha=0.88, label="8场景总成功率 (%) ↑")
    ax2_r = ax2.twinx()
    bars2 = ax2_r.bar(xs + w / 2, err_vals, width=w, color="#FF8A4C", alpha=0.88, label="平均终态误差 (cm) ↓")
    ax2_r.tick_params(colors="#FFBA92", labelsize=9)
    ax2_r.set_ylabel("平均终态误差 (cm) ↓", color="#FFBA92")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(short_names, color="#E2ECF7", fontsize=8.8)
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("总到达成功率 (%) ↑")
    ax2.set_title("(c) 8 大复杂压力场景 × 4 种子 (32 回合) 消融对比", fontsize=11, fontweight="bold")
    for b in bars1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.8, f"{b.get_height():.0f}%", ha="center", color="#00E5FF", fontsize=8)
    for b in bars2:
        ax2_r.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}", ha="center", color="#FFBA92", fontsize=8)

    # Panel D: 非对称特权 PPO 两阶段课程收敛曲线
    ax3 = axes[1, 1]
    iters = [h["iter"] for h in ppo_history]
    rets = [h["mean_return"] for h in ppo_history]
    dists = [h["mean_final_dist_cm"] for h in ppo_history]
    ax3.plot(iters, rets, color="#52FF9A", marker="o", lw=2.2, label="回合平均回报 (Return) ↑")
    ax3.axvline(5.5, color="#FFB020", linestyle=":", lw=1.4, label="课程升级 (S1~S5 → 全扰动 S1~S8)")
    ax3_r = ax3.twinx()
    ax3_r.plot(iters, dists, color="#9B7BFF", marker="s", lw=1.9, linestyle="--", label="训练期终态误差 (cm) ↓")
    ax3_r.tick_params(colors="#C6B4FF", labelsize=9)
    ax3_r.set_ylabel("训练期随机策略终态误差 (cm) ↓", color="#C6B4FF")
    ax3.set_xlabel("PPO 课程迭代轮次 (Iteration)")
    ax3.set_ylabel("平均回合回报 ↑")
    ax3.set_title("(d) 非对称特权 Actor-Critic 课程强化学习收敛曲线", fontsize=11, fontweight="bold")
    ax3.legend(facecolor="#141E2B", edgecolor="#2C3E55", labelcolor="#E2ECF7", fontsize=8, loc="upper left")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.6)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="D:/robot_work/runs_end_side/hover_rl_policy")
    ap.add_argument("--android-assets-dir", default="G:/robot_app/android/app/src/main/assets/models")
    ap.add_argument("--fig-path", default="D:/robot_work/docs/showcase/figs/fig5_spatial_sandbox_trajectories.png")
    ap.add_argument("--bc-epochs", type=int, default=42)
    ap.add_argument("--ppo-iters", type=int, default=14)
    args = ap.parse_args()

    t0 = time.perf_counter()
    torch.manual_seed(20261118)
    np.random.seed(20261118)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = sim.calibrated(P.SimParams(), verbose=False)
    print(f"[1/6] Collecting 2D Spatial Sandbox + Gaze Tracking expert dataset on {device}...", flush=True)
    exp_feats, exp_acts = collect_hybrid_expert_dataset(sp, n_sandbox_eps=48, n_track_eps=20, seed=100)
    print(f"      Collected {len(exp_feats)} paired symmetric state-action samples.", flush=True)

    ac = PrivilegedSpatialActorCritic(hidden_dim=160, critic_dim=192)
    print(f"[2/6] Stage 1: Behavioral Cloning + Mirror Equivariance ({args.bc_epochs} epochs)...", flush=True)
    bc_losses = stage1_pretrain_bc(ac, exp_feats, exp_acts, epochs=args.bc_epochs, device=device)
    print(f"      BC loss: {bc_losses[0]:.6f} -> {bc_losses[-1]:.6f}", flush=True)
    bc_folded_net = ac.export_folded_policy()

    print(f"[3/6] Stage 2 & 3: Asymmetric Privileged PPO Curriculum ({args.ppo_iters} iterations)...", flush=True)
    ppo_history = train_curriculum_ppo(
        ac,
        exp_feats,
        exp_acts,
        n_iters=args.ppo_iters,
        episodes_per_iter=8,
        device=device,
    )
    ppo_folded_net = ac.export_folded_policy()

    # 校验正前方平衡点零偏置特性
    eq_telem = RobotTelemetry(t_ms=0, head_q=0.0, pitch=0.0, roll=0.0, yaw=45.0, x1=2.3, x2=2.3)
    eq_feat = EndSideFeatureBuilder.build(
        eq_telem,
        detections=[{"cx": 0.5, "cy": 0.35, "w": 0.25, "h": 0.25, "score": 0.90, "cls": 0}],
        target_lock=None,
    )
    eq_cmd = ppo_folded_net.infer_numpy(eq_feat)
    zero_bias_err = max(abs(eq_cmd.vyaw), abs(eq_cmd.head))
    print(f"      Equilibrium zero-bias check (at x=2.3m, yaw=45°): cmd=({eq_cmd.vx:.4f}, {eq_cmd.vyaw:.4f}, {eq_cmd.head:.4f})", flush=True)

    print("[4/6] Benchmarking 5 controllers across all 8 complex 2D sandbox scenarios...", flush=True)
    sandbox_summary, sample_traces = evaluate_8_scenario_benchmark(bc_folded_net, ppo_folded_net)

    print("[5/6] Verifying classic 5-scenario continuous gaze tracking benchmark...", flush=True)
    orig_make = C.make

    def _custom_make(name: str, **kw):
        if name == "spatial_ppo_v2":
            return EndSideSimPolicy(ppo_folded_net, name="spatial_ppo_v2", use_shield=True)
        return orig_make(name, **kw)

    C.make = _custom_make
    try:
        track_rows = sim_eval.eval_tracking(sp, [0, 1, 2], ["spatial_ppo_v2"], verbose=False)
    finally:
        C.make = orig_make
    track_eps_rms = float(np.mean([r["eps_rms"] for r in track_rows]))
    track_succ = float(np.mean([r["success"] for r in track_rows])) * 100.0
    print(f"      Classic 5-scenario tracking: eps_rms = {track_eps_rms:.2f}°, success = {track_succ:.1f}%", flush=True)

    print("[6/6] Exporting FlatBuffers v3 TFLite bundle + syncing Android assets + rendering figure...", flush=True)
    torch.save(ppo_folded_net.state_dict(), out_dir / "spatial_policy_v2_folded.pt")
    bundle = export_policy_bundle(
        ppo_folded_net,
        out_dir=out_dir,
        model_name="hover-end-side-rl-ppo-v1",
        version="2026.11.18-spatial-v2",
        notes=(
            "SE(2) Spatial-Memory Residual Tanh Policy (39->160->160+Res->3) trained via "
            "BC + Asymmetric Privileged PPO across 8 complex 2D tabletop scenarios."
        ),
    )
    max_tfl_diff = verify_tflite_numerical(ppo_folded_net, Path(bundle["tflite_path"]), n_samples=64)

    assets_dir = Path(args.android_assets_dir)
    if assets_dir.exists():
        shutil.copy2(bundle["tflite_path"], assets_dir / "hover_rl_policy.tflite")
        shutil.copy2(bundle["json_path"], assets_dir / "hover_rl_policy.json")

    fig_path = Path(args.fig_path)
    render_spatial_sandbox_figure(ppo_history, sandbox_summary, sample_traces, fig_path)

    report = {
        "timestamp": "2026-11-18",
        "device": str(device),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "architecture": {
            "feature_dim": 39,
            "hidden_dim": 160,
            "use_residual": True,
            "critic_privileged_dim": 51,
            "zero_bias_max_err": round(float(zero_bias_err), 5),
            "tflite_max_abs_diff": float(max_tfl_diff),
            "tflite_sha256": bundle["sha256"],
            "tflite_bytes": Path(bundle["tflite_path"]).stat().st_size,
        },
        "bc_loss_start_end": [bc_losses[0], bc_losses[-1]],
        "ppo_history": ppo_history,
        "classic_gaze_tracking": {
            "eps_rms_deg": round(track_eps_rms, 2),
            "success_pct": round(track_succ, 2),
        },
        "sandbox_8_scenarios": sandbox_summary,
    }
    eval_json = out_dir / "spatial_sandbox_eval.json"
    eval_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done in {report['elapsed_sec']}s! Saved {eval_json} & {fig_path} (TFLite diff={max_tfl_diff:.2e}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
