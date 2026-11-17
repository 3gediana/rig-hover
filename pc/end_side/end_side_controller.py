# -*- coding: utf-8 -*-
"""end_side_controller.py — RIG-Hover 端侧（PC / Android）感知-记忆-策略-安全壳统一闭环控制器。

核心架构：
1. EndSideFeatureBuilder: 构建与 Android App (PolicyFeatureBuilder.kt) 100% 逐维对齐的 39 维特征向量；
   - 0..27 : 28 维归一化本体遥测 (IMU, 轮速, 里程计, 头角, 指令, 力矩, PID)
   - 28..32: 目标槽 0 (cx, cy, w, h, conf) -> 当前注视主目标 (人脸/拳头)
   - 33..37: 目标槽 1 (cx, cy, w, h, conf) -> 次目标，或在 APPROACH 阶段编码锁定的敲击点 (TargetLock)
   - 38    : 目标存在标志 (0/1)
2. EndSidePolicyNet: 39 -> 128 -> 128 -> 3 (Tanh MLP 策略网络，支持 BC 预热与 PPO 强化微调，
   100% 兼容标准 TFLite FULLY_CONNECTED + TANH 算子)；
3. EndSideSafetyShield:
   - 非对称桌沿零过冲屏障：当 remaining_m <= 0 时强制 vx <= 0，杜绝冲出桌面；
   - 倒立摆大俯仰角降速保护：|pitch| > 12° 时按比例衰减 vx/vyaw；
   - 动作变化率平滑 (Jerk Limiter)：抑制欠阻尼航向环 (ζ≈0.08) 激振。
4. EndSideController: 完整融合工作记忆状态机 (IDLE -> GAZE_FACE -> GAZE_FIST -> APPROACH -> DWELL)
   与端侧策略网络，支持真机 HTTP 闭环与物理仿真器闭环。
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

from end_side_protocol import DriveCommand, EndSideWatchdog, RobotTelemetry, _clamp  # noqa: E402
from scenario_v2 import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    MODE_NAMES,
    TargetLock,
    V2Params,
)
from spatial_memory import SpatialMemoryTracker  # noqa: E402


class EndSideFeatureBuilder:
    """与 Android 端 com.luwu.hovercontroller.ml.PolicyFeatureBuilder 完全对齐的 39 维特征构建器。"""

    FEATURE_DIM = 39
    MIN_SCORE = 0.35

    @classmethod
    def build(
        cls,
        status: RobotTelemetry,
        detections: list[dict] | None = None,
        target_lock: TargetLock | None = None,
        clearance_m: float = 1.0,
    ) -> np.ndarray:
        out = np.zeros(cls.FEATURE_DIM, dtype=np.float32)

        # 0-2 姿态
        out[0] = status.roll / 180.0
        out[1] = status.pitch / 180.0
        out[2] = status.yaw / 180.0

        # 3-5 加速度
        out[3] = _clamp(status.ax / 20.0, -1.0, 1.0)
        out[4] = _clamp(status.ay / 20.0, -1.0, 1.0)
        out[5] = _clamp(status.az / 20.0, -1.0, 1.0)

        # 6-7 角速度
        out[6] = _clamp(status.pit_gyro / 10.0, -1.0, 1.0)
        out[7] = _clamp(status.dq / 10.0, -1.0, 1.0)

        # 8-9 轮速
        out[8] = status.v1 / 100.0
        out[9] = status.v2 / 100.0

        # 10-11 里程计
        out[10] = _clamp(status.x1 / 5.0, -1.0, 1.0)
        out[11] = _clamp(status.x2 / 5.0, -1.0, 1.0)

        # 12-14 头部
        out[12] = status.head_q / 75.0
        out[13] = status.head_target / 75.0
        out[14] = status.gimbal_target / 45.0

        # 15-19 当前指令
        out[15] = status.cmd_vx / 100.0
        out[16] = status.cmd_vyaw / 100.0
        out[17] = 0.0 if status.cmd_mode == 0 else 1.0
        out[18] = _clamp(status.cmd_pos / 2.0, -1.0, 1.0)
        out[19] = status.cmd_yaw_target / 180.0

        # 20-21 力矩
        out[20] = _clamp(status.t1 / 100.0, -1.0, 1.0)
        out[21] = _clamp(status.t2 / 100.0, -1.0, 1.0)

        # 22-27 PID
        out[22] = _clamp(status.pos_e / 100.0, -1.0, 1.0)
        out[23] = _clamp(status.pos_u / 100.0, -1.0, 1.0)
        out[24] = _clamp(status.vel_e / 100.0, -1.0, 1.0)
        out[25] = _clamp(status.vel_u / 100.0, -1.0, 1.0)
        out[26] = _clamp(status.pit_e / 100.0, -1.0, 1.0)
        out[27] = _clamp(status.pit_u / 100.0, -1.0, 1.0)

        # 28-37 两个目标槽 (按面积从大到小)
        dets = detections or []
        valid_dets = [
            d for d in dets
            if float(d.get("score", 0.0)) >= cls.MIN_SCORE
            and float(d.get("w", 0.0)) > 0.0
            and float(d.get("h", 0.0)) > 0.0
        ]
        valid_dets.sort(key=lambda d: float(d["w"]) * float(d["h"]), reverse=True)
        top2 = valid_dets[:2]

        for slot in range(2):
            base = 28 + slot * 5
            if slot < len(top2):
                d = top2[slot]
                out[base + 0] = _clamp(float(d["cx"]), 0.0, 1.0)
                out[base + 1] = _clamp(float(d["cy"]), 0.0, 1.0)
                out[base + 2] = _clamp(float(d["w"]), 0.0, 1.0)
                out[base + 3] = _clamp(float(d["h"]), 0.0, 1.0)
                raw_sc = _clamp(float(d["score"]), 0.0, 1.0)
                # 槽 1 普通视觉框置信度封顶 0.85，保留 1.0 专用于双敲空间记忆激活门控
                out[base + 4] = min(raw_sc, 0.85) if slot == 1 else raw_sc

        # 若处于双敲锁定巡航 (target_lock.valid == 1)，将 SE(2) 空间记忆编码进槽 1 (33..37)
        if target_lock is not None and target_lock.valid:
            rel_theta = target_lock.bearing_body - status.head_q
            cx_lock = _clamp(0.5 + rel_theta / 60.0, 0.0, 1.0)
            body_bearing_01 = _clamp(0.5 + target_lock.bearing_body / 180.0, 0.0, 1.0)
            rem_norm = _clamp(target_lock.remaining_m / 1.5, 0.0, 1.0)
            clear_norm = _clamp(float(clearance_m) / 1.0, 0.0, 1.0)
            out[33] = cx_lock          # 精细视线相对角 (±30° 高清区)
            out[34] = body_bearing_01  # 广角机身相对角 (±90° 线性不饱和区)
            out[35] = rem_norm         # 欧氏剩余距离 (÷1.5m)
            out[36] = clear_norm       # 前向安全净空裕度 (÷1.0m)
            out[37] = 1.0              # 双敲空间记忆锁定标志

        out[38] = 1.0 if (len(top2) > 0 or (target_lock is not None and target_lock.valid)) else 0.0
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


class EndSidePolicyNet(nn.Module):
    """端侧 39 维输入 -> 3 维控制输出 (vx, vyaw, head) 策略网络。

    支持标准三层 Tanh MLP 与残差增强四层结构 (use_residual=True)，
    导出 TFLite 仅含标准 FULLY_CONNECTED / ADD / TANH / MUL 内置算子。
    """

    def __init__(
        self,
        feature_dim: int = 39,
        hidden_dim: int = 128,
        action_dim: int = 3,
        use_residual: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_residual = bool(use_residual)

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_res = nn.Linear(hidden_dim, hidden_dim) if self.use_residual else None
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 动作缩放比例：与协议 vx ∈ [-100,100], vyaw ∈ [-100,100], head ∈ [-75,75] 对齐
        self.register_buffer("action_scale", torch.tensor([100.0, 100.0, 75.0], dtype=torch.float32))

        self._init_weights()

    def _init_weights(self) -> None:
        layers = [self.fc1, self.fc2] + ([self.fc_res] if self.fc_res is not None else [])
        for m in layers:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        # 输出层小权重初始化，保证初始策略平稳近零
        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        if self.fc_res is not None:
            h_mid = torch.tanh(self.fc_res(h1 + h2))
            return torch.tanh(self.fc3(h_mid))
        return torch.tanh(self.fc3(h2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回物理量纲指令: [vx (±100), vyaw (±100), head (±75°)]。"""
        u = self.forward_normalized(x)
        return u * self.action_scale

    @torch.no_grad()
    def infer_numpy(self, feat_39: np.ndarray) -> DriveCommand:
        x = torch.from_numpy(np.asarray(feat_39, dtype=np.float32)).reshape(1, -1)
        out = self.forward(x)[0].cpu().numpy()
        return DriveCommand(vx=float(out[0]), vyaw=float(out[1]), head=float(out[2])).sanitized()


class EndSideSafetyShield:
    """端侧確定性安全屏障（Safety Shield）。

    保证任何神经网络输出都绝对满足：
    1. 桌沿零过冲约束 (Zero-Overshoot Barrier)：APPROACH 阶段一旦 remaining_m <= 0，强制 vx <= 0；
       在接近目标最后 0.20m 内按平方根包络平滑限速 vx <= k_brake * sqrt(remaining_m)；
    2. 倒立摆俯仰保护 (Pitch Stability Envelope)：|pitch| > 12° 时线性压低 vx/vyaw；
    3. 欠阻尼航向与舵机平滑限幅 (Jerk / Slew-Rate Limiter)：限制单步最大指令跳变。
    """

    def __init__(
        self,
        max_delta_vx: float = 18.0,
        max_delta_vyaw: float = 28.0,
        max_delta_head: float = 12.0,
        pitch_warn_deg: float = 12.0,
        pitch_crit_deg: float = 22.0,
    ):
        self.max_delta_vx = float(max_delta_vx)
        self.max_delta_vyaw = float(max_delta_vyaw)
        self.max_delta_head = float(max_delta_head)
        self.pitch_warn_deg = float(pitch_warn_deg)
        self.pitch_crit_deg = float(pitch_crit_deg)
        self.prev_cmd = DriveCommand()

    def reset(self) -> None:
        self.prev_cmd = DriveCommand()

    def apply(
        self,
        raw_cmd: DriveCommand,
        mode: int,
        remaining_m: float = 0.0,
        pitch_deg: float = 0.0,
        clearance_m: float = 1.2,
    ) -> DriveCommand:
        vx = float(raw_cmd.vx)
        vyaw = float(raw_cmd.vyaw)
        head = float(raw_cmd.head)
        eff_rem = min(float(remaining_m), float(clearance_m))

        # 1. 非 APPROACH 模式下禁止前向巡航 (防止注视阶段误动轮子纵向漂移)
        if mode != MODE_APPROACH:
            vx = 0.0
        else:
            # 桌沿与障碍物零过冲硬屏障
            if eff_rem <= 0.0:
                vx = 0.0
            elif eff_rem < 0.25:
                # 软刹停包络：距离越近上限越低
                v_cap = 100.0 * math.sqrt(max(eff_rem, 0.0) / 0.25)
                vx = min(vx, v_cap)
            vx = max(0.0, vx)  # APPROACH 阶段不倒车

        # 2. 倒立摆俯仰稳定包络
        abs_p = abs(float(pitch_deg))
        if abs_p > self.pitch_warn_deg:
            scale = max(0.0, 1.0 - (abs_p - self.pitch_warn_deg) / max(self.pitch_crit_deg - self.pitch_warn_deg, 1.0))
            vx *= scale * 0.25
            vyaw *= scale * 0.50

        # 3. 速率限幅 (Jerk Limiter)
        vx = _clamp(vx, self.prev_cmd.vx - self.max_delta_vx, self.prev_cmd.vx + self.max_delta_vx)
        vyaw = _clamp(vyaw, self.prev_cmd.vyaw - self.max_delta_vyaw, self.prev_cmd.vyaw + self.max_delta_vyaw)
        head = _clamp(head, self.prev_cmd.head - self.max_delta_head, self.prev_cmd.head + self.max_delta_head)

        # 若目标已到点或触及净空边界，vx 直接清零不受下限拖尾影响
        if mode != MODE_APPROACH or eff_rem <= 0.0:
            vx = 0.0

        safe = DriveCommand(vx=vx, vyaw=vyaw, head=head, stop=raw_cmd.stop).sanitized()
        self.prev_cmd = safe
        return safe


class EndSideController:
    """端侧完整闭环控制器：SE(2) 二维空间记忆 (SpatialMemoryTracker) + 端侧 RL 策略网络 + 二维安全屏障。"""

    def __init__(
        self,
        policy_model: EndSidePolicyNet | None = None,
        params: V2Params | None = None,
        blend_rl: float = 1.0,
    ):
        self.p = params or V2Params(arrive_tol_m=0.032)
        self.policy = policy_model or EndSidePolicyNet()
        self.policy.eval()
        self.blend_rl = float(_clamp(blend_rl, 0.0, 1.0))
        self.spatial_memory = SpatialMemoryTracker(reknock_refractory_s=0.65)
        self.shield = EndSideSafetyShield()
        self.watchdog = EndSideWatchdog(timeout_s=0.40)
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.mode = MODE_IDLE
        self.target_lock = TargetLock()
        self._t_mode = 0.0
        self._t_fist_lost = 0.0
        self._t_approach = 0.0
        self.spatial_memory.reset()
        self.shield.reset()
        self.history: list[tuple[float, int]] = []

    def _enter(self, new_mode: int) -> None:
        if new_mode != self.mode:
            self.mode = new_mode
            self._t_mode = self.t
            self.history.append((round(self.t, 3), new_mode))

    def _person_dist(self, person_h: float) -> float:
        if person_h < self.p.dist_person_h_min:
            return 0.0
        return float(self.p.dist_k / person_h)

    def _build_virtual_detections(self, scene_obs: dict, q_head: float) -> list[dict]:
        dets = []
        if scene_obs.get("face_valid", 0):
            cx_f = _clamp(0.5 + float(scene_obs.get("theta_face", 0.0)) / 60.0, 0.05, 0.95)
            s_f = _clamp(float(scene_obs.get("face_scale", 0.10)) * 3.0, 0.10, 0.80)
            dets.append({"cx": cx_f, "cy": 0.35, "w": s_f, "h": s_f, "score": 0.90, "cls": 0})
        if scene_obs.get("hand_valid", 0):
            cx_h = _clamp(0.5 + float(scene_obs.get("theta_fist", 0.0)) / 60.0, 0.05, 0.95)
            dets.append({"cx": cx_h, "cy": 0.65, "w": 0.18, "h": 0.18, "score": max(0.45, min(0.85, float(scene_obs.get("fist_prob", 0.5)))), "cls": 1})
        return dets

    def step(
        self,
        dt: float,
        telemetry: RobotTelemetry,
        scene_obs: dict,
        detections: list[dict] | None = None,
    ) -> DriveCommand:
        """单步端侧决策：SE(2) 里程计空间记忆更新 -> 组装 39 维特征 -> RL 策略推理 -> 二维安全屏障滤波。"""
        dt = float(max(dt, 0.0))
        self.t += dt
        p = self.p

        q_head = telemetry.head_q
        mem_state = self.spatial_memory.step_odometry(dt, telemetry)

        face_valid = int(scene_obs.get("face_valid", 0))
        theta_face = float(scene_obs.get("theta_face", 0.0))
        face_scale = float(scene_obs.get("face_scale", 0.0))
        person_h = float(scene_obs.get("person_h", 0.0))
        hand_valid = int(scene_obs.get("hand_valid", 0))
        fist_prob = float(scene_obs.get("fist_prob", 0.0))
        theta_fist = float(scene_obs.get("theta_fist", 0.0))
        knock_fired = int(scene_obs.get("knock_fired", 0))
        knock_x = float(scene_obs.get("knock_x", 0.5))
        knock_y = float(scene_obs.get("knock_y", 0.72))
        clearance_m = float(scene_obs.get("clearance_m", 1.2))
        obstacle_deflect_deg = float(scene_obs.get("obstacle_deflect_deg", 0.0))

        face_in_range = bool(face_valid and (p.face_scale_min <= face_scale <= p.face_scale_max))
        fist_on = bool(hand_valid and fist_prob >= p.fist_prob_on)
        fist_hold = bool(hand_valid and fist_prob >= p.fist_prob_off)

        # 1. 高层状态转移与 SE(2) 空间记忆锁定（支持 IDLE/GAZE 首发锁定 + APPROACH/DWELL 冷却期后动态改敲重锁定）
        if knock_fired and hand_valid and self.spatial_memory.can_accept_knock(self.t):
            dist = self._person_dist(person_h)
            if dist <= 0.0 and face_scale > 1e-3:
                dist = p.dist_k / max(face_scale * 5.0, 0.2)
            if dist <= 0.0:
                dist = 0.80
            travel = max(0.0, dist - p.overshoot_margin_m)
            locked = self.spatial_memory.lock_polar(
                now_s=self.t,
                bearing_body_deg=float(theta_fist + q_head),
                dist_m=travel,
                x_px=knock_x,
                y_px=knock_y,
            )
            self.target_lock = locked.to_target_lock()
            self._t_approach = 0.0
            self._enter(MODE_APPROACH)
        elif self.mode == MODE_IDLE:
            if fist_on:
                self._t_fist_lost = 0.0
                self._enter(MODE_GAZE_FIST)
            elif face_in_range:
                self._enter(MODE_GAZE_FACE)
        elif self.mode == MODE_GAZE_FACE:
            if fist_on:
                self._t_fist_lost = 0.0
                self._enter(MODE_GAZE_FIST)
            elif not face_in_range:
                self._enter(MODE_IDLE)
        elif self.mode == MODE_GAZE_FIST:
            if fist_hold:
                self._t_fist_lost = 0.0
            else:
                self._t_fist_lost += dt
                if self._t_fist_lost >= p.fist_lost_s:
                    self._enter(MODE_GAZE_FACE if face_in_range else MODE_IDLE)
        elif self.mode == MODE_APPROACH:
            self._t_approach += dt
            # 严禁在未发生新双敲时用普通 hand_valid 覆写桌面锁定点（彻底免疫挥手假动作 S3）
            self.target_lock = mem_state.to_target_lock()
            too_close = bool(face_valid and face_scale >= p.stop_if_face_scale)
            cliff_or_blocked = clearance_m <= 0.02
            if (
                self.target_lock.remaining_m <= p.arrive_tol_m
                or self._t_approach >= p.approach_timeout_s
                or too_close
                or cliff_or_blocked
            ):
                self._enter(MODE_DWELL)
        elif self.mode == MODE_DWELL:
            self.target_lock = mem_state.to_target_lock()
            if (self.t - self._t_mode) >= p.dwell_s:
                self.spatial_memory.clear_lock()
                self.target_lock = TargetLock()
                self._enter(MODE_GAZE_FACE if face_in_range else MODE_IDLE)

        # 2. 组装 39 维端侧观测（局部绕障时在引导方位上叠加切向偏转建议角）
        dets = detections if detections is not None else self._build_virtual_detections(scene_obs, q_head)
        lock_for_feat = None
        if self.mode == MODE_APPROACH and self.target_lock.valid:
            steered_bearing = float(_clamp(self.target_lock.bearing_body + obstacle_deflect_deg, -90.0, 90.0))
            lock_for_feat = TargetLock(
                valid=1,
                t_lock=self.target_lock.t_lock,
                x_px=self.target_lock.x_px,
                y_px=self.target_lock.y_px,
                bearing_body=steered_bearing,
                dist_m=self.target_lock.dist_m,
                remaining_m=self.target_lock.remaining_m,
                last_seen_t=self.target_lock.last_seen_t,
                updates=self.target_lock.updates,
            )

        feat_39 = EndSideFeatureBuilder.build(
            telemetry,
            dets,
            target_lock=lock_for_feat,
            clearance_m=clearance_m,
        )

        # 3. 策略推理 + 引导基准混合
        rl_cmd = self.policy.infer_numpy(feat_39)
        nominal_cmd = self._nominal_reference(telemetry, scene_obs, obstacle_deflect_deg=obstacle_deflect_deg)

        raw_cmd = DriveCommand(
            vx=self.blend_rl * rl_cmd.vx + (1.0 - self.blend_rl) * nominal_cmd.vx,
            vyaw=self.blend_rl * rl_cmd.vyaw + (1.0 - self.blend_rl) * nominal_cmd.vyaw,
            head=self.blend_rl * rl_cmd.head + (1.0 - self.blend_rl) * nominal_cmd.head,
        )

        # 4. 端侧二维安全屏障滤波 + 看门狗刷新
        safe_cmd = self.shield.apply(
            raw_cmd,
            mode=self.mode,
            remaining_m=self.target_lock.remaining_m if self.mode == MODE_APPROACH else 0.0,
            pitch_deg=telemetry.pitch if abs(telemetry.pitch) < 45.0 else 0.0,
            clearance_m=clearance_m,
        )
        self.watchdog.update_command(safe_cmd, now_s=self.t)
        return safe_cmd

    def _nominal_reference(
        self,
        telemetry: RobotTelemetry,
        scene_obs: dict,
        obstacle_deflect_deg: float = 0.0,
    ) -> DriveCommand:
        """物理阻尼增强型二维弧线追击基线控制律（用于安全兜底与混合参考）。"""
        p = self.p
        q_head = telemetry.head_q
        body_yaw_rate = float(telemetry.vel_e) * 0.60
        if self.mode == MODE_GAZE_FACE:
            eps = float(scene_obs.get("theta_face", 0.0)) + q_head
            head = _clamp(eps, -p.head_max_deg, p.head_max_deg)
            over = abs(eps) - (p.head_max_deg - p.head_limit_margin_deg)
            vyaw = (math.copysign(min(p.k_fast * over, p.vyaw_max_dps), eps) - 0.18 * body_yaw_rate) if over > 0 else 0.0
            return DriveCommand(vx=0.0, vyaw=(vyaw / 90.0) * 100.0, head=head)
        if self.mode == MODE_GAZE_FIST:
            eps = float(scene_obs.get("theta_fist", 0.0)) + q_head
            head = _clamp(eps, -p.head_max_deg, p.head_max_deg)
            over = abs(eps) - (p.head_max_deg - p.head_limit_margin_deg)
            vyaw = (math.copysign(min(p.k_fast * over, p.vyaw_max_dps), eps) - 0.18 * body_yaw_rate) if over > 0 else 0.0
            return DriveCommand(vx=0.0, vyaw=(vyaw / 90.0) * 100.0, head=head)
        if self.mode == MODE_APPROACH:
            eps = self.target_lock.bearing_body + obstacle_deflect_deg
            head = _clamp(eps, -p.head_max_deg, p.head_max_deg)
            vyaw_dps = _clamp(2.4 * eps - 0.22 * body_yaw_rate, -68.0, 68.0)
            rem = self.target_lock.remaining_m
            if rem > p.arrive_tol_m and abs(eps) < 52.0:
                turn_gate = max(0.0, math.cos(math.radians(min(85.0, abs(eps) * 1.4))))
                v_ms = min(0.22, math.sqrt(max(0.0, 2.0 * 0.16 * rem))) * turn_gate
                vx_pct = (v_ms / 0.30) * 100.0
            else:
                vx_pct = 0.0
            return DriveCommand(vx=vx_pct, vyaw=(vyaw_dps / 90.0) * 100.0, head=head)
        if self.mode == MODE_DWELL:
            return DriveCommand(vx=0.0, vyaw=0.0, head=_clamp(self.target_lock.bearing_body, -p.head_max_deg, p.head_max_deg))
        return DriveCommand(vx=0.0, vyaw=0.0, head=0.0)
