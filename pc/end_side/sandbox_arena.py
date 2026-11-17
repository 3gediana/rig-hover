# -*- coding: utf-8 -*-
"""sandbox_arena.py — RIG-Hover 2D 桌面自平衡多情景复杂物理沙箱 (HoverSandboxEnv)。

将 `pc/sim/sim.py` 的 167Hz LQR 轮式倒立摆精确离散动力学与二维桌面平面刚体运动学
(x_world, y_world, yaw_body) 耦合，构建 8 类高难度现实干扰与边界测试场景：
  1. S1_blind_vanish          : 双敲锁定后手与人立即消失，纯靠 SE(2) 空间记忆曲线盲走入位
  2. S2_bystander_distractor  : 双敲后路人人脸横穿视野试图劫持视觉主槽 slot0
  3. S3_waving_decoy          : 双敲后手部抬起持续大幅挥动（非双敲假动作），检验记忆不被拖走
  4. S4_mid_reknock           : 行进中途在另一侧触发二次双敲，检验动态重锁定与平滑变轨
  5. S5_extreme_off_axis      : 大偏角 (±64°..±70°) 双敲，检验头身协同广角转身与弧线接近
  6. S6_obstacle_detour       : 路径侧方存在桌面水杯障碍物，检验净空感知与切向平滑绕障
  7. S7_cliff_guard           : 桌沿附近双敲叠加车轮滑移，检验桌面边界零坠落硬防护
  8. S8_latency_dropout_storm : 60ms 通信队列延迟 + 35% 视觉丢帧 + ±3.5° 舵机回差 + 陀螺漂移
"""
from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SIM_DIR = HERE.parent / "sim"
PERC_DIR = HERE.parent / "perception"
for _p in (str(HERE), str(SIM_DIR), str(PERC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import params as P  # noqa: E402
import sim  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from spatial_memory import wrap180  # noqa: E402

SCENARIO_NAMES: tuple[str, ...] = (
    "S1_blind_vanish",
    "S2_bystander_distractor",
    "S3_waving_decoy",
    "S4_mid_reknock",
    "S5_extreme_off_axis",
    "S6_obstacle_detour",
    "S7_cliff_guard",
    "S8_latency_dropout_storm",
)


@dataclass
class CircleObstacle:
    x_m: float
    y_m: float
    radius_m: float = 0.075
    margin_m: float = 0.085


@dataclass
class TableBounds:
    x_min: float = -0.35
    x_max: float = 0.92
    y_min: float = -0.52
    y_max: float = 0.52
    safe_margin: float = 0.07


class HoverSandboxEnv:
    """20Hz 端侧决策节拍 × 166.7Hz (6ms) 底层 LQR 自平衡耦合二维桌面沙箱。"""

    CTRL_SUBSTEPS = 8  # 8 * 6ms = 48ms (~20.8Hz 端侧闭环周期)

    def __init__(
        self,
        scenario: str = "S1_blind_vanish",
        seed: int = 42,
        domain_rand: bool = False,
        max_steps: int = 210,
    ):
        if scenario not in SCENARIO_NAMES:
            raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIO_NAMES}")
        self.scenario = scenario
        self.base_seed = int(seed)
        self.domain_rand = bool(domain_rand)
        self.max_steps = int(max_steps)

        self.base_sp = sim.calibrated(P.SimParams(), verbose=False)
        self.dt_inner = self.base_sp.fw.ctrl_dt_ms * 1e-3  # 0.006s
        self.dt_step = self.dt_inner * self.CTRL_SUBSTEPS  # 0.048s

        self.reset(seed=seed)

    def _sample_params(self, rng: np.random.Generator) -> P.SimParams:
        if not self.domain_rand and self.scenario != "S8_latency_dropout_storm":
            return self.base_sp
        pp = self.base_sp.plant
        rand_pp = replace(
            pp,
            mass_body_kg=float(pp.mass_body_kg * rng.uniform(0.86, 1.14)),
            wheel_rolling_damping=float(pp.wheel_rolling_damping * rng.uniform(0.85, 1.18)),
            yaw_damping=float(pp.yaw_damping * rng.uniform(0.82, 1.20)),
            k_yaw_accel=float(pp.k_yaw_accel * rng.uniform(0.86, 1.15)),
            actuator_delay_ticks=int(rng.choice([1, 2])),
        )
        return replace(self.base_sp, plant=rand_pp)

    def reset(self, scenario: str | None = None, seed: int | None = None) -> dict[str, Any]:
        if scenario is not None:
            if scenario not in SCENARIO_NAMES:
                raise ValueError(f"Unknown scenario {scenario!r}")
            self.scenario = scenario
        if seed is not None:
            self.base_seed = int(seed)

        self.rng = np.random.default_rng(self.base_seed * 7919 + 31)
        self.sp = self._sample_params(self.rng)
        self.plant = sim.Plant(self.sp, self.dt_inner)
        self.plant.reset(pitch0_deg=float(self.rng.uniform(-1.2, 1.2)), seed=self.base_seed)
        self.imu = sim.Imu(self.sp, self.rng if (self.domain_rand or self.scenario == "S8_latency_dropout_storm") else None)
        self.fw = sim.FirmwareControl(self.sp)
        self.fw.k_dyaw = -0.52
        self.fw.head_rate_comp = 1.0
        self.arb = sim.Arbiter(self.sp, sim.ARB_SRC_CFC)
        self.sv = sim.SensorView()

        self.t_s: float = 0.0
        self.step_idx: int = 0
        self.rx_m: float = 0.0
        self.ry_m: float = 0.0
        self.trail_xy: list[tuple[float, float]] = [(0.0, 0.0)]

        # 故障与环境参数
        self.slip_scale: float = 1.0
        self.dropout_prob: float = 0.0
        self.servo_backlash_deg: float = 0.0
        self.yaw_drift_dps: float = 0.0
        self.cmd_delay_steps: int = 0
        self.table_bounds: TableBounds | None = None
        self.obstacles: list[CircleObstacle] = []

        # 配置各场景几何目标 (世界系 +X 正前, +Y 右侧)
        sgn = 1.0 if (self.base_seed % 2 == 0) else -1.0
        self._knock1_step = 14  # ~0.67s
        self._knock2_step = -1
        self._knock1_fired = False
        self._knock2_fired = False

        if self.scenario == "S1_blind_vanish":
            ang1 = sgn * float(self.rng.uniform(24.0, 42.0))
            dist1 = float(self.rng.uniform(0.55, 0.78))
        elif self.scenario == "S2_bystander_distractor":
            ang1 = sgn * float(self.rng.uniform(26.0, 38.0))
            dist1 = float(self.rng.uniform(0.58, 0.76))
        elif self.scenario == "S3_waving_decoy":
            ang1 = sgn * float(self.rng.uniform(22.0, 36.0))
            dist1 = float(self.rng.uniform(0.55, 0.74))
        elif self.scenario == "S4_mid_reknock":
            ang1 = sgn * float(self.rng.uniform(28.0, 40.0))
            dist1 = float(self.rng.uniform(0.66, 0.80))
            self._knock2_step = 42  # ~2.0s 中途第二次敲击
        elif self.scenario == "S5_extreme_off_axis":
            ang1 = sgn * float(self.rng.uniform(58.0, 68.0))
            dist1 = float(self.rng.uniform(0.56, 0.74))
            self._knock1_step = 18  # 留出先看拳头转头入视野时间
        elif self.scenario == "S6_obstacle_detour":
            ang1 = sgn * float(self.rng.uniform(12.0, 24.0))
            dist1 = float(self.rng.uniform(0.68, 0.82))
            obs_r = 0.50 * dist1
            obs_ang = math.radians(ang1) + sgn * 0.14  # 偏在直连线一侧约 7cm，触发切向平滑绕障
            self.obstacles = [
                CircleObstacle(
                    x_m=obs_r * math.cos(obs_ang),
                    y_m=obs_r * math.sin(obs_ang),
                    radius_m=0.065,
                    margin_m=0.075,
                )
            ]
        elif self.scenario == "S7_cliff_guard":
            self.table_bounds = TableBounds(x_min=-0.32, x_max=0.62, y_min=-0.48, y_max=0.48, safe_margin=0.08)
            self.slip_scale = 1.15
            ang1 = sgn * float(self.rng.uniform(8.0, 20.0))
            dist1 = 0.72  # 敲击估计点超出安全桌沿 x_max - safe_margin = 0.54m！
        else:  # S8_latency_dropout_storm
            ang1 = sgn * float(self.rng.uniform(22.0, 40.0))
            dist1 = float(self.rng.uniform(0.54, 0.74))
            self.cmd_delay_steps = 1  # ~50ms 队列延迟
            self.dropout_prob = 0.35
            self.servo_backlash_deg = 3.2
            self.yaw_drift_dps = sgn * 1.2
            self.slip_scale = float(self.rng.uniform(0.90, 1.12))

        # 注意: 真实敲击桌面点距离人体约差 overshoot_margin (0.12m)，这里定义 p_knock 为期望停驻点，
        # 观测中给控制器暴露的 person_h 对应 dist_person = dist_stop + 0.12m，使理想几何刚好落在该桌面敲击点上。
        self.target1_xy = (
            dist1 * math.cos(math.radians(ang1)),
            dist1 * math.sin(math.radians(ang1)),
        )
        ang2 = -sgn * float(self.rng.uniform(30.0, 44.0))
        dist2 = float(self.rng.uniform(0.46, 0.62))
        self.target2_rel = (dist2, ang2)
        self.active_target_xy: tuple[float, float] = self.target1_xy
        self.person_xy: tuple[float, float] = (
            (dist1 + 0.12) * math.cos(math.radians(ang1)),
            (dist1 + 0.12) * math.sin(math.radians(ang1)),
        )

        self._cmd_queue: deque[DriveCommand] = deque(
            [DriveCommand() for _ in range(self.cmd_delay_steps + 1)],
            maxlen=self.cmd_delay_steps + 1,
        )
        self._applied_head_deg: float = 0.0
        self.min_obstacle_dist_m: float = 9.9
        self.cliff_fell: bool = False
        self.obstacle_collided: bool = False

        self._sync_sensors(0.0)
        return self._build_observation(knock_fired=0)

    def _sync_sensors(self, dt: float) -> None:
        q_raw = self.plant.st.q_head
        q_meas = round(q_raw / self.sp.plant.head_quant_deg) * self.sp.plant.head_quant_deg
        if dt > 0.0:
            self.imu.update(self.plant, q_meas, dt)
            if abs(self.yaw_drift_dps) > 1e-6:
                self.imu.yaw = wrap180(self.imu.yaw + self.yaw_drift_dps * dt)
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

    def compute_clearance_and_deflection(self) -> tuple[float, float]:
        """计算机身前向安全净空 clearance_m (0..1.2m) 与局部切向绕障偏转建议角 deflect_deg。"""
        clearance = 1.20
        deflect_deg = 0.0

        # 1. 圆形桌面障碍物净空与切向绕行势场
        for obs in self.obstacles:
            xb, yb, center_d, bearing = self._world_to_body(obs.x_m, obs.y_m)
            surf_d = max(0.0, center_d - obs.radius_m)
            self.min_obstacle_dist_m = min(self.min_obstacle_dist_m, surf_d)
            if surf_d <= 0.005:
                self.obstacle_collided = True
            safe_r = obs.radius_m + obs.margin_m
            if xb > -0.04 and abs(yb) < safe_r + 0.14 and center_d < 0.75:
                # 若障碍物挡在前向走廊内，压低净空并给出远离障碍中心的切向偏转角
                lat_ratio = max(0.0, 1.0 - abs(yb) / (safe_r + 0.14))
                eff_clear = max(0.05, surf_d - obs.margin_m * 0.5)
                if abs(yb) < safe_r * 0.75:
                    clearance = min(clearance, eff_clear)
                # 目标在障碍后方时产生平滑绕行偏角 (yb > 0 表示障碍在右侧 -> 向左偏转 < 0)
                push_sign = -1.0 if yb >= 0.0 else 1.0
                deflect_deg += push_sign * 28.0 * lat_ratio * max(0.0, 1.0 - center_d / 0.75)

        # 2. 桌面悬崖边界净空 (沿当前机身正前方 +X 射线求距安全桌沿边界的距离)
        if self.table_bounds is not None:
            b = self.table_bounds
            if (
                self.rx_m < b.x_min
                or self.rx_m > b.x_max
                or self.ry_m < b.y_min
                or self.ry_m > b.y_max
            ):
                self.cliff_fell = True
            psi = math.radians(self.plant.st.yaw_body)
            cx, cy = math.cos(psi), math.sin(psi)
            ray_d = 1.50
            if cx > 1e-4:
                ray_d = min(ray_d, ((b.x_max - b.safe_margin) - self.rx_m) / cx)
            elif cx < -1e-4:
                ray_d = min(ray_d, ((b.x_min + b.safe_margin) - self.rx_m) / cx)
            if cy > 1e-4:
                ray_d = min(ray_d, ((b.y_max - b.safe_margin) - self.ry_m) / cy)
            elif cy < -1e-4:
                ray_d = min(ray_d, ((b.y_min + b.safe_margin) - self.ry_m) / cy)
            clearance = min(clearance, max(0.0, ray_d))

        return float(max(0.0, min(1.20, clearance))), float(max(-38.0, min(38.0, deflect_deg)))

    def _build_telemetry(self) -> RobotTelemetry:
        sv = self.sv
        body_yaw_rate_dps = -(sv.yaw_rate - sv.qd_head)
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
            heap=118000,
        )

    def _build_observation(self, knock_fired: int) -> dict[str, Any]:
        telem = self._build_telemetry()
        clearance_m, deflect_deg = self.compute_clearance_and_deflection()
        gaze_yaw = self.plant.st.yaw_body + telem.head_q
        fov_half = 32.0

        # 计算当前主人物/敲击点的视线角
        _, _, t_dist, t_bearing_body = self._world_to_body(*self.active_target_xy)
        theta_target_img = wrap180(t_bearing_body - telem.head_q)
        person_dist = max(0.28, t_dist + 0.12)
        person_h_true = min(2.20, max(0.08, 0.84 / person_dist))
        face_scale_true = min(0.22, max(0.055, 0.168 / person_dist))

        face_valid = 0
        theta_face = 0.0
        face_scale = 0.0
        person_h = 0.0
        hand_valid = 0
        fist_prob = 0.0
        theta_fist = 0.0

        dropped = (
            knock_fired == 0
            and self.dropout_prob > 0.0
            and float(self.rng.random()) < self.dropout_prob
        )

        if not dropped:
            # 敲击前与敲击当拍：展示人脸与拳头（S5 大偏角时允许余光检测手势引导先转头）
            pre_knock = self.step_idx <= self._knock1_step
            is_reknock_window = (
                self._knock2_step > 0
                and (self._knock2_step - 3) <= self.step_idx <= self._knock2_step
            )

            if pre_knock or is_reknock_window or knock_fired:
                eff_fov = 72.0 if (self.scenario == "S5_extreme_off_axis" and pre_knock) else fov_half
                if abs(theta_target_img) <= eff_fov or knock_fired:
                    face_valid = 1
                    theta_face = float(np.clip(theta_target_img, -30.0, 30.0))
                    face_scale = face_scale_true
                    person_h = person_h_true
                    hand_valid = 1
                    fist_prob = 0.90
                    theta_fist = float(theta_target_img)  # 敲击瞬间记录真实相对头角
            else:
                # 敲击完成后的复杂干扰情景分支
                if self.scenario == "S1_blind_vanish":
                    # 手和人完全离开画面 -> 100% 考验空间记忆盲走！
                    pass
                elif self.scenario == "S2_bystander_distractor":
                    # 路人人脸从另一侧横穿视野，试图把 slot0 带偏；到达后期主人物重现供 DWELL->GAZE_FACE 回看
                    if t_dist > 0.11 and self.step_idx < 130:
                        sweep = -38.0 * math.sin(0.07 * (self.step_idx - self._knock1_step))
                        dist_bystander_img = wrap180((sweep - self.plant.st.yaw_body) - telem.head_q)
                        if abs(dist_bystander_img) <= fov_half:
                            face_valid = 1
                            theta_face = float(dist_bystander_img)
                            face_scale = 0.13
                            person_h = 0.65
                    elif abs(theta_target_img) <= fov_half:
                        face_valid = 1
                        theta_face = float(theta_target_img)
                        face_scale = 0.14
                        person_h = 0.70
                elif self.scenario == "S3_waving_decoy":
                    # 手抬起大幅左右挥舞（非双敲 knock_fired=0），验证控制器绝不篡改已锁定的桌面坐标
                    wave_world = 42.0 * math.sin(0.18 * (self.step_idx - self._knock1_step))
                    wave_img = wrap180((wave_world - self.plant.st.yaw_body) - telem.head_q)
                    if abs(wave_img) <= fov_half:
                        hand_valid = 1
                        fist_prob = 0.86
                        theta_fist = float(wave_img)
                    if abs(theta_target_img) <= fov_half:
                        face_valid = 1
                        theta_face = float(theta_target_img)
                        face_scale = 0.13
                        person_h = 0.68
                else:
                    # S4/S5/S6/S7/S8：手敲完立刻撤离 (hand_valid=0)，仅当机器人接近后人脸在视野内才可见
                    if t_dist < 0.14 and abs(theta_target_img) <= fov_half:
                        face_valid = 1
                        theta_face = float(theta_target_img)
                        face_scale = 0.14
                        person_h = 0.70

        detections: list[dict[str, float]] = []
        if face_valid:
            cx_f = float(np.clip(0.5 + theta_face / 60.0, 0.04, 0.96))
            s_f = float(np.clip(face_scale * 2.8, 0.10, 0.75))
            detections.append({"cx": cx_f, "cy": 0.34, "w": s_f, "h": s_f, "score": 0.92, "cls": 0})
        if hand_valid:
            cx_h = float(np.clip(0.5 + theta_fist / 60.0, 0.04, 0.96))
            detections.append({"cx": cx_h, "cy": 0.68, "w": 0.16, "h": 0.16, "score": 0.84, "cls": 1})

        scene_obs = {
            "face_valid": face_valid,
            "theta_face": theta_face,
            "face_scale": face_scale,
            "person_h": person_h,
            "hand_valid": hand_valid,
            "fist_prob": fist_prob,
            "theta_fist": theta_fist,
            "knock_fired": int(knock_fired),
            "knock_x": float(np.clip(0.5 + theta_fist / 60.0, 0.05, 0.95)),
            "knock_y": 0.72,
            "clearance_m": clearance_m,
            "obstacle_deflect_deg": deflect_deg,
        }
        return {
            "telemetry": telem,
            "scene_obs": scene_obs,
            "detections": detections,
            "clearance_m": clearance_m,
            "obstacle_deflect_deg": deflect_deg,
        }

    def step(self, cmd: DriveCommand) -> tuple[dict[str, Any], dict[str, Any]]:
        """执行一步端侧闭环控制 (48ms = 8 × 6ms 倒立摆子步)，返回新观测与真实世界评测指标。"""
        self._cmd_queue.append(cmd.sanitized())
        delayed_cmd = self._cmd_queue[0]

        vx_mps = float((delayed_cmd.vx / 100.0) * 0.30)
        vyaw_dps = float((delayed_cmd.vyaw / 100.0) * 90.0)
        raw_head = float(np.clip(delayed_cmd.head, -45.0, 45.0))
        if abs(raw_head - self._applied_head_deg) >= self.servo_backlash_deg:
            self._applied_head_deg = raw_head

        for _ in range(self.CTRL_SUBSTEPS):
            prev_xw = self.plant.st.x_w
            prev_yaw = self.plant.st.yaw_body
            self.arb.post_raw(
                sim.CtrlOut(
                    head_des=self._applied_head_deg,
                    vx_des=vx_mps,
                    vyaw_des=vyaw_dps,
                    mode=1,
                ),
                self.t_s,
            )
            self.arb.note_feature(self.t_s)
            self.fw.update_state(self.sv, self.t_s)
            self.arb.tick(self.sv, self.fw, self.t_s)
            _, _, h_cmd = self.fw.control(self.sv, self.t_s, do_update_state=False)
            self.plant.step(self.fw.temp_u, self.fw.yaw_u, h_cmd)
            self.t_s += self.dt_inner
            self._sync_sensors(self.dt_inner)

            # 2D 平面刚体位姿积分 (含车轮打滑比例)
            ds = (self.plant.st.x_w - prev_xw) * self.slip_scale
            mid_yaw_rad = math.radians(0.5 * (prev_yaw + self.plant.st.yaw_body))
            self.rx_m += ds * math.cos(mid_yaw_rad)
            self.ry_m += ds * math.sin(mid_yaw_rad)

        self.step_idx += 1
        self.trail_xy.append((round(self.rx_m, 4), round(self.ry_m, 4)))

        # 触发第一次双敲或中途第二次双敲重锁定
        knock_fired = 0
        if self.step_idx == self._knock1_step and not self._knock1_fired:
            self._knock1_fired = True
            knock_fired = 1
        elif self._knock2_step > 0 and self.step_idx == self._knock2_step and not self._knock2_fired:
            self._knock2_fired = True
            d2, ang2 = self.target2_rel
            world_ang2 = math.radians(self.plant.st.yaw_body + ang2)
            self.active_target_xy = (
                self.rx_m + d2 * math.cos(world_ang2),
                self.ry_m + d2 * math.sin(world_ang2),
            )
            knock_fired = 1

        obs = self._build_observation(knock_fired=knock_fired)
        xb_true, yb_true, dist_true, bearing_true = self._world_to_body(*self.active_target_xy)
        info = {
            "t_s": round(self.t_s, 3),
            "step": self.step_idx,
            "robot_xy": (self.rx_m, self.ry_m),
            "robot_yaw_deg": float(self.plant.st.yaw_body),
            "target_xy": self.active_target_xy,
            "true_x_body_m": xb_true,
            "true_y_body_m": yb_true,
            "true_dist_m": dist_true,
            "true_bearing_deg": bearing_true,
            "pitch_deg": float(self.plant.pitch_true_deg),
            "yaw_rate_dps": float(self.plant.st.yaw_rate),
            "cliff_fell": self.cliff_fell,
            "obstacle_collided": self.obstacle_collided,
            "min_obstacle_dist_m": self.min_obstacle_dist_m,
            "done": self.step_idx >= self.max_steps or self.cliff_fell,
        }
        return obs, info

    def build_privileged_critic_features(self, feat_39: np.ndarray) -> np.ndarray:
        """拼接 39 维部署观测 + 12 维沙箱真值特权特征 = 51 维，仅供训练期非对称 Critic 使用。"""
        xb_t, yb_t, dist_t, bear_t = self._world_to_body(*self.active_target_xy)
        clearance_m, deflect_deg = self.compute_clearance_and_deflection()
        priv = np.array(
            [
                np.clip(xb_t / 1.2, -1.5, 1.5),
                np.clip(yb_t / 1.2, -1.5, 1.5),
                np.clip(dist_t / 1.2, 0.0, 1.5),
                np.clip(bear_t / 90.0, -2.0, 2.0),
                np.clip(self.plant.pitch_true_deg / 20.0, -2.0, 2.0),
                np.clip(self.plant.st.yaw_rate / 90.0, -2.0, 2.0),
                np.clip(self.plant.st.v_w / 0.30, -1.5, 1.5),
                np.clip(clearance_m / 1.2, 0.0, 1.0),
                np.clip(deflect_deg / 40.0, -1.0, 1.0),
                float(self.slip_scale - 1.0) * 4.0,
                float(self.cmd_delay_steps) * 0.5,
                1.0 if self._knock1_fired else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([feat_39.astype(np.float32), priv], axis=0)
