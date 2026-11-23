# -*- coding: utf-8 -*-
"""spatial_memory.py — SE(2) 自车系二维刚体里程计空间记忆追踪器 (SpatialMemoryTracker)。

解决原有 1D 标量距离记忆在真实复杂场景下的四大结构性问题：
1. **盲走旋转守恒**：双敲锁定偏轴目标后哪怕手与人立刻离开视野，机身旋转 Δψ 时利用精确 SE(2) 刚体逆变换
   实时旋转自车系坐标 (x_body, y_body)，杜绝方位角冻结导致的原地打转；
2. **二维曲线路径积分**：沿任意圆弧 (vx, vyaw) 行驶或绕障时，使用半角刚体弦向位移更新真实欧氏距离；
3. **防非敲击手势诱骗**：锁定期间忽略未触发双敲的挥手/路人框，仅在超过防抖窗口的新双敲事件到来时执行重锁定；
4. **越过目标垂面即停**：当纵向坐标 x_body <= 0 且处于横向走廊内时立即将有效剩余距离归零，配合安全屏障实现零过冲。
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

from end_side_protocol import RobotTelemetry  # noqa: E402
from scenario_v2 import TargetLock  # noqa: E402


def wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


@dataclass
class SpatialTargetState:
    valid: int = 0
    x_body_m: float = 0.0
    y_body_m: float = 0.0
    bearing_body_deg: float = 0.0
    initial_dist_m: float = 0.0
    euclidean_dist_m: float = 0.0
    remaining_m: float = 0.0
    t_lock: float = 0.0
    x_px: float = 0.5
    y_px: float = 0.72
    reknock_count: int = 0

    def to_target_lock(self) -> TargetLock:
        if not self.valid:
            return TargetLock()
        return TargetLock(
            valid=1,
            t_lock=round(self.t_lock, 3),
            x_px=float(self.x_px),
            y_px=float(self.y_px),
            bearing_body=float(self.bearing_body_deg),
            dist_m=round(float(self.initial_dist_m), 4),
            remaining_m=round(float(self.remaining_m), 4),
            last_seen_t=round(self.t_lock, 3),
            updates=int(max(1, self.reknock_count)),
        )


class SpatialMemoryTracker:
    """维护自车坐标系 (+X 正前, +Y 右侧) 下的桌面敲击点二维记忆。"""

    def __init__(
        self,
        reknock_refractory_s: float = 0.65,
        pass_corridor_y_m: float = 0.25,
    ):
        self.reknock_refractory_s = float(reknock_refractory_s)
        self.pass_corridor_y_m = float(pass_corridor_y_m)
        self.reset()

    def reset(self) -> None:
        self._valid: bool = False
        self._x_body: float = 0.0
        self._y_body: float = 0.0
        self._initial_dist: float = 0.0
        self._t_lock: float = -999.0
        self._x_px: float = 0.5
        self._y_px: float = 0.72
        self._reknock_count: int = 0

        self._has_odom_ref: bool = False
        self._prev_wheel_x: float = 0.0
        self._prev_cmd_pos: float = 0.0
        self._prev_imu_yaw: float = 0.0
        self._prev_head_q: float = 0.0

    def clear_lock(self) -> None:
        self._valid = False
        self._x_body = 0.0
        self._y_body = 0.0
        self._initial_dist = 0.0

    def can_accept_knock(self, now_s: float) -> bool:
        if not self._valid:
            return True
        return (float(now_s) - self._t_lock) >= self.reknock_refractory_s

    def lock_xy(
        self,
        now_s: float,
        x_body_m: float,
        y_body_m: float,
        x_px: float = 0.5,
        y_px: float = 0.72,
    ) -> SpatialTargetState:
        self._valid = True
        self._x_body = float(x_body_m)
        self._y_body = float(y_body_m)
        self._initial_dist = float(math.hypot(self._x_body, self._y_body))
        self._t_lock = float(now_s)
        self._x_px = float(x_px)
        self._y_px = float(y_px)
        self._reknock_count += 1
        return self.state

    def lock_polar(
        self,
        now_s: float,
        bearing_body_deg: float,
        dist_m: float,
        x_px: float = 0.5,
        y_px: float = 0.72,
    ) -> SpatialTargetState:
        d = max(0.0, float(dist_m))
        rad = math.radians(float(bearing_body_deg))
        xb = d * math.cos(rad)
        yb = d * math.sin(rad)
        return self.lock_xy(now_s=now_s, x_body_m=xb, y_body_m=yb, x_px=x_px, y_px=y_px)

    def step_odometry(self, dt_s: float, telemetry: RobotTelemetry) -> SpatialTargetState:
        """根据双轮编码器位移与头部 IMU 减舵机角解算的纯机身偏航角增量，更新自车系目标位置。"""
        wheel_x = float(getattr(telemetry, "wheel_x_m", telemetry.wheel_x_avg))
        cmd_pos = float(telemetry.cmd_pos)
        imu_yaw = float(telemetry.yaw)
        head_q = float(telemetry.head_q)

        if not self._has_odom_ref:
            self._has_odom_ref = True
            self._prev_wheel_x = wheel_x
            self._prev_cmd_pos = cmd_pos
            self._prev_imu_yaw = imu_yaw
            self._prev_head_q = head_q
            return self.state

        # 1. 前向弧长增量 Δs (优先采用真实轮编码器均值，若编码器恒零而 cmd_pos 推进则回退 cmd_pos)
        ds_wheel = wheel_x - self._prev_wheel_x
        ds_cmd = cmd_pos - self._prev_cmd_pos
        if abs(ds_wheel) >= 1e-6:
            ds = ds_wheel
        elif abs(ds_cmd) >= 1e-6 and abs(wheel_x) < 1e-6:
            ds = ds_cmd
        else:
            ds = 0.0

        # 2. 纯机身偏航角增量 Δψ_body (RIG-Hover 的 IMU 装在头部: imu_yaw = yaw_body + head_q)
        #    兼容单元测试省略 imu 字段 (imu_yaw 恒为 0 且无差速/转向指令) 的精简遥测包
        imu_unpopulated = (
            abs(imu_yaw) < 1e-6
            and abs(self._prev_imu_yaw) < 1e-6
            and abs(telemetry.v1 - telemetry.v2) < 1e-5
            and abs(telemetry.cmd_vyaw) < 1e-5
        )
        if imu_unpopulated:
            dyaw_deg = 0.0
        else:
            prev_body_yaw = wrap180(self._prev_imu_yaw - self._prev_head_q)
            curr_body_yaw = wrap180(imu_yaw - head_q)
            dyaw_deg = wrap180(curr_body_yaw - prev_body_yaw)
            # 单步异常跳变保护
            dyaw_deg = max(-45.0, min(45.0, dyaw_deg))

        self._prev_wheel_x = wheel_x
        self._prev_cmd_pos = cmd_pos
        self._prev_imu_yaw = imu_yaw
        self._prev_head_q = head_q

        if not self._valid:
            return self.state

        # 3. SE(2) 半角刚体逆变换: p_new = R(-Δψ) @ (p_old - [Δs cos(Δψ/2), Δs sin(Δψ/2)]^T)
        dpsi = math.radians(dyaw_deg)
        half = 0.5 * dpsi
        tx = self._x_body - ds * math.cos(half)
        ty = self._y_body - ds * math.sin(half)
        c = math.cos(dpsi)
        s = math.sin(dpsi)
        self._x_body = c * tx + s * ty
        self._y_body = -s * tx + c * ty
        return self.state

    @property
    def state(self) -> SpatialTargetState:
        if not self._valid:
            return SpatialTargetState()
        euclid = float(math.hypot(self._x_body, self._y_body))
        bearing = float(math.degrees(math.atan2(self._y_body, self._x_body)))
        # 一旦纵向越过目标垂面 (x_body <= 0) 且处于横向到达走廊内，有效剩余距离立即置 0
        if self._x_body <= 0.0 and abs(self._y_body) <= self.pass_corridor_y_m:
            eff_rem = 0.0
        else:
            eff_rem = euclid
        return SpatialTargetState(
            valid=1,
            x_body_m=self._x_body,
            y_body_m=self._y_body,
            bearing_body_deg=bearing,
            initial_dist_m=self._initial_dist,
            euclidean_dist_m=euclid,
            remaining_m=eff_rem,
            t_lock=self._t_lock,
            x_px=self._x_px,
            y_px=self._y_px,
            reknock_count=self._reknock_count,
        )
