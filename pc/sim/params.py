# -*- coding: utf-8 -*-
"""params.py — 仿真用全部参数，逐项标注来源（固件文件:行号 / 接口字段 / 估计）。

三条纪律：

1. **能读到源码的一律抄源码**，不猜。每条带 ``SRC`` 字段，写清出处。
2. 拿不到真值的（质量、轮距、舵机带宽、力矩常数）标 ``EST``，并在
   :func:`provenance_table` 里输出成 REPORT 的参数出处表 —— 不许藏在代码里。
3. 标定的量（``K_TORQUE`` / ``K_YAW``）用 :func:`calibrate` 从"固件增益能闭环稳定"
   这个判据反推，见 :mod:`sim` 的 ``calibrate()``。标定值是**仿真自洽**的结果，
   不是真机实测值，REPORT 里必须这么写。

单位约定（与固件一致）：

* 角度：``deg``，速度 ``deg/s``
* 位移：``m``，线速度 ``m/s``
* ``pitch`` / ``dq`` / ``yaw`` / ``q_head``：来自头部 IMU 与头部舵机
* 轮子力矩指令：``tor1/tor2`` 是 int16 伺服指令（无量纲计数），乘 ``K_TORQUE``
  得 N·m —— ``K_TORQUE`` 是标定量
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

F32 = float

DEG = math.pi / 180.0


# =========================================================================== #
# 1. 直接抄固件的量（出处写死在这里，改了会有人发现）                          #
# =========================================================================== #
@dataclass(frozen=True)
class FirmwareParams:
    """逐字段对应固件的一个常量。``src`` 就是出处。"""

    # --- 控制律 -------------------------------------------------------- #
    # robot.cc:89-92 InitializeController()
    lqr_k0: float = 1300.0      # 位置误差增益 (torque-count / m)
    lqr_k1: float = 5.0         # 速度误差增益 (torque-count / (m/s))
    lqr_k2: float = 25.0        # 俯仰误差增益 (torque-count / deg)
    lqr_k3: float = 0.4         # 俯仰角速度增益 (torque-count / (deg/s))
    # robot.cc:87
    imu_zero: float = -2.0      # 平衡点俯仰基准 (deg)
    # robot.cc:55 k_yaw = 4.0
    k_yaw: float = 4.0          # 航向 P 增益 (torque-count / deg)
    # robot.cc:284
    yaw_u_clip: float = 80.0    # yaw_u 限幅 (torque-count)
    # control_arbiter.c arb_cfg_t.k_dyaw（默认 0 = 纯 P，与旧固件逐位一致）
    # robot.cc:航向环 yaw_u = k_yaw*(yaw-q_head-stable_yaw) - k_dyaw*yaw_rate
    k_dyaw: float = 0.0         # 航向 D 增益 (torque-count / (deg/s))
    # control_arbiter.c 安全壳的命令速率限幅。**固件默认是 1.0 / 400**
    # （control_arbiter.c cfg_defaults），这里默认 0（关）是为了让 eval.py 逐位
    # 复现 REPORT 的基线；要跑"固件口径"就显式传 1.0 / 400（scan_dyaw.py 与
    # diag_sep.py 都这么做），或 POST /cfg 覆盖。
    vx_rate_max: float = 0.0    # m/s^2，0 = 关闭
    vyaw_rate_max: float = 0.0  # deg/s^2，0 = 关闭
    vx_stop_eps: float = 0.005  # m/s
    vyaw_stop_eps: float = 0.5  # deg/s
    # robot.cc:275-279：|temp_u|>20 时额外 ±10（脱困用的死区补偿）
    hyst_thresh: float = 20.0
    hyst_bump: float = 10.0
    # robot.h PID 结构顺序 + robot.cc:34 pid_pit 初始化 {…, fpUMax=1000, …}
    torque_cmd_max: float = 1000.0   # temp_u 的绝对限幅 (torque-count)
    # robot.cc:295/302/315 三次 vTaskDelay(2) → 6 ms
    ctrl_dt_ms: float = 6.0
    # robot.cc:121  wheel_x = (w1+w2)/2/1024*3.1415*0.06 → 直径 0.06 m
    wheel_counts_per_rev: float = 1024.0
    wheel_diameter_m: float = 0.06
    # robot.cc:294 WriteByte_P_V(3, 1500 - target_head_pos*10, 1300)
    head_pwm_center: float = 1500.0
    head_counts_per_deg: float = 10.0
    head_pwm_limit: float = 450.0    # 1500±450 ↔ ±45°，见 control_arbiter.c:99 head_max=45
    # control_arbiter.c:99
    head_max_deg: float = 45.0
    # control_arbiter.c:98
    vx_max: float = 0.30
    # control_arbiter.c:99 vyaw_max = 90.0
    vyaw_max: float = 90.0
    # control_arbiter.c:95-96
    pos_lead_max: float = 0.20
    yaw_lead_max: float = 35.0
    # control_arbiter.c:101-102
    stale_ms: float = 300.0
    lost_ms: float = 1000.0
    # control_arbiter.c:102 head_lpf = 0.15（每 2 ms tick 一次一阶低通）
    head_lpf: float = 0.15
    # control_arbiter.c:100 cmd_timeout_ms
    cmd_timeout_ms: float = 500.0
    # control_arbiter.c:108
    cfc_hz: float = 50.0
    # cfc_task.c:457-472 publish()：头/机身/前进都是 tanh 头 × 满量程
    # (head_max / vyaw_max / vx_max)，与 export_blob 的 HEAD_MAX 一致
    head_full_scale: float = 45.0
    vyaw_full_scale: float = 90.0
    vx_full_scale: float = 0.30
    # imu.cc:251 dq = 0.6*dq + 0.4*pit_gyro（在 IMU 读取率上，motor_rx 5 ms）
    dq_lpf_keep: float = 0.6
    # imu.cc:252 gyro_yaw = 0.5*gyro_yaw + 0.5*gyro_y（同上）
    yaw_lpf_keep: float = 0.5
    # imu.cc:264-268 静止死区，|gyro|<1.5 dps 视为零
    yaw_deadzone_dps: float = 1.5
    # imu.cc:254-256 互补滤波：pit_freq = 1/(|pit_gyro|+200)
    pit_freq_k: float = 200.0
    # board_config.h:24 MOTOR_RX_TASK_INTERVAL_MS = 5 → IMU 采样 200 Hz
    imu_dt_ms: float = 5.0
    # robot.cc:143-146 站立/摔倒判定
    stand_hold_ms: float = 1500.0
    fall_hold_ms: float = 500.0
    upright_pitch_deg: float = 7.0
    fallen_pitch_deg: float = 60.0
    # 视觉视场（control_arbiter.c:109 fov_h = 60）
    fov_h_deg: float = 60.0

    #: LQR 增益的三种口径，见 :func:`gains`。默认 ``sign_flip`` —— 不是"调参"，
    #: 是**修正一个符号错误**，理由见 :meth:`gains`。
    lqr_mode: str = "sign_flip"

    def gains(self, mode: str | None = None):
        """返回 LQR 的 4 个增益 ``(K0, K1, K2, K3)``，单位见 :mod:`params` 模块文档。

        ============ ==========================================================
        mode         含义
        ============ ==========================================================
        ``firmware`` 源码原样 ``[1300, 5, 25, 0.4]``（robot.cc:89-92）。
        ``sign_flip`` 只把位置/速度两个通道取负：``[-1300, -5, 25, 0.4]``。
        ``care``     用 ``scipy.linalg.solve_continuous_are`` 对仿真机体重算的
                     最优增益（Q=diag(10,1,10,1)，R=1），作为"本该是多少"的参考。
        ============ ==========================================================

        为什么 ``sign_flip`` 才是对的（不是凑出来的）：把
        ``u = -(K0·x + K1·v + K2·p + K3·dq)`` 写成物理口径，令 ``p = -57.3·θ``
        （θ 为前倾角），则

            平衡环（快）：  θ̈ = |B3|·g·M·r·(1 − c)·θ，  c = 57.3·K2·k_t/(M·r·g)
            位置环（慢，θ̈≈0 的慢流形 θ = ẍ/g）：ẍ = −k_t·K0·Δx /(M·r·(1−c))

        θ 环稳定要求 **c > 1**，位置环稳定要求 **c < 1** —— 两者互补，且
        ``c`` 的临界值 ``M·r·g/(57.3·K2)`` 与机体质量/惯量/质心高度**无关**。
        所以只要 ``K0 > 0``，这个闭环对**任何**力矩标定都不可稳定；把 K0/K1 取负
        后 c>1 同时满足两式。`:mod:`sim.calibrate` 对两种 mode 都验证了这一点。

        Returns:
            ``(K0, K1, K2, K3)``
        """
        m = self.lqr_mode if mode is None else mode
        base = (self.lqr_k0, self.lqr_k1, self.lqr_k2, self.lqr_k3)
        if m == "firmware":
            return base
        if m == "sign_flip":
            return (-base[0], -base[1], base[2], base[3])
        if m == "care":
            # scipy CARE(Q=diag(10,1,10,1), R=1) 给出 u = -K·[x,v,θ,θ̇] 的
            # K = [-3.16, -1.94, -7.50, -1.25]（θ 为弧度、前倾为正）。
            # 换算到固件口径（p = -57.3θ，单位度）：
            #   u = 3.16x + 1.94v + 7.50θ + 1.25θ̇ = -K0x -K1v +57.3K2θ + 57.3K3θ̇
            return (-3.16, -1.94, 7.50 / 57.29578, 1.25 / 57.29578)
        raise ValueError("未知 lqr_mode %r（firmware / sign_flip / care）" % (m,))

    def src(self, name: str) -> str:
        """给 ``name`` 找一条出处字符串（REPORT 的参数表直接用它）。"""
        return _SRC.get(name, "")


#: 出处表：字段名 → "文件:行号 说明"。REPORT 的出处表就是这张表的投影。
_SRC = {
    "lqr_k0": "robot.cc:89 InitializeController() lqr_k[0]=1300",
    "lqr_k1": "robot.cc:90 lqr_k[1]=5",
    "lqr_k2": "robot.cc:91 lqr_k[2]=25",
    "lqr_k3": "robot.cc:92 lqr_k[3]=0.4",
    "imu_zero": "robot.cc:87 imu_zero=-2.0",
    "k_yaw": "robot.cc:55 k_yaw=4.0",
    "yaw_u_clip": "robot.cc:284 Clip(yaw_u,-80,80)",
    "k_dyaw": "control_arbiter.c arb_cfg_t.k_dyaw（默认 0）+ robot.cc 航向环 D 项",
    "vx_rate_max": "control_arbiter.c s_cfg.vx_rate_max=1.0 m/s^2（仿真默认 0=关，见字段注释）",
    "vyaw_rate_max": "control_arbiter.c s_cfg.vyaw_rate_max=400 deg/s^2（与教师 config.yaml 同值）",
    "vx_stop_eps": "control_arbiter.c s_cfg.vx_stop_eps=0.005 m/s",
    "vyaw_stop_eps": "control_arbiter.c s_cfg.vyaw_stop_eps=0.5 deg/s",
    "hyst_thresh": "robot.cc:275-279 temp_u 死区补偿 ±20/±10",
    "hyst_bump": "robot.cc:275-279",
    "torque_cmd_max": "robot.h PID 结构第 9 项 fpUMax；robot.cc:34 pid_pit 初值 1000",
    "ctrl_dt_ms": "robot.cc:295/302/315 三次 vTaskDelay(2ms)",
    "wheel_counts_per_rev": "robot.cc:121 wheel_x=…/1024.0*3.1415*0.06",
    "wheel_diameter_m": "robot.cc:121 圆周换算里的 0.06（米）",
    "head_pwm_center": "robot.cc:294 WriteByte_P_V(3,1500-head*10,1300)",
    "head_counts_per_deg": "robot.cc:294 的 ×10；反向 motor.cc:57 (1500-pos)/10",
    "head_pwm_limit": "robot.cc:294 与 control_arbiter.c:99 head_max=45 → ±450",
    "head_max_deg": "control_arbiter.c:99 s_cfg.head_max=45.0",
    "vx_max": "control_arbiter.c:98 s_cfg.vx_max=0.30",
    "vyaw_max": "control_arbiter.c:99 s_cfg.vyaw_max=90.0",
    "pos_lead_max": "control_arbiter.c:95 s_cfg.pos_lead_max=0.20",
    "yaw_lead_max": "control_arbiter.c:96 s_cfg.yaw_lead_max=35.0",
    "stale_ms": "control_arbiter.c:101 s_cfg.stale_ms=300",
    "lost_ms": "control_arbiter.c:102 s_cfg.lost_ms=1000",
    "head_lpf": "control_arbiter.c:102 s_cfg.head_lpf=0.15（每 tick）",
    "cmd_timeout_ms": "control_arbiter.c:103 cmd_timeout_ms=500",
    "cfc_hz": "control_arbiter.c:108 s_cfg.cfc_hz=50",
    "head_full_scale": "cfc_task.c:457 head_des=out[0]*cfg->head_max(=45)",
    "vyaw_full_scale": "cfc_task.c:458 vyaw_des=out[1]*cfg->vyaw_max(=90)",
    "vx_full_scale": "cfc_task.c:472 vx_des=out[4]*cfg->vx_max(=0.30)",
    "dq_lpf_keep": "imu.cc:251 dq=0.6*dq+0.4*pit_gyro",
    "yaw_lpf_keep": "imu.cc:252 gyro_yaw=0.5*gyro_yaw+0.5*gyro_y",
    "yaw_deadzone_dps": "imu.cc:262-268 kYawGyroDeadZoneDps=1.5",
    "pit_freq_k": "imu.cc:254 pit_freq=1/(|pit_gyro|+200)",
    "imu_dt_ms": "board_config.h:24 MOTOR_RX_TASK_INTERVAL_MS=5",
    "stand_hold_ms": "robot.cc:144 kStandHoldUs=1.5e6",
    "fall_hold_ms": "robot.cc:143 kFallHoldUs=0.5e6",
    "upright_pitch_deg": "robot.cc:146 upright_pose |pitch|<7",
    "fallen_pitch_deg": "robot.cc:145 fallen_pose |pitch|>=60",
    "fov_h_deg": "control_arbiter.c:109 s_cfg.fov_h=60",
}


# =========================================================================== #
# 2. 估计量（真机拿不到 / 只有经验值）                                        #
# =========================================================================== #
@dataclass(frozen=True)
class PlantParams:
    """机体物理参数。除轮半径外**全部是估计**，见 ``est`` 标记。"""

    # --- 机构 ---------------------------------------------------------- #
    wheel_radius_m: float = 0.03        # 固件可推：直径 0.06（robot.cc:121）
    wheel_track_m: float = 0.13         # EST：两轮中心距，整机照片目测
    # --- 质量 / 惯量（EST）--------------------------------------------- #
    mass_body_kg: float = 0.32          # EST：电池+主控+云台，不含两轮
    mass_wheel_kg: float = 0.035        # EST：单个 KP4012 轮组
    inertia_wheel_kgm2: float = 0.5 * 0.035 * 0.03 ** 2   # 实心盘，EST
    inertia_body_kgm2: float = 0.0012    # EST：绕轮轴
    com_height_m: float = 0.055         # EST：质心到轮轴高度
    yaw_inertia_kgm2: float = 0.0016     # EST：绕竖直轴
    # --- 摩擦（EST）----------------------------------------------------- #
    yaw_damping: float = 1.2            # 1/s，机身偏航黏性阻尼
    wheel_rolling_damping: float = 0.8  # 1/s，轮子滚动阻尼（等效到线速度）
    # --- 执行器 --------------------------------------------------------- #
    #: torque-count → N·m。**标定量**（:func:`sim.calibrate` 反推）。
    k_torque: float = 4.0e-4
    #: 从"算出 temp_u"到"力矩真正作用"的延迟，按整拍（6 ms）算（robot.cc:302
    #: 第二次 vTaskDelay 之后才 sendWheelTor）。EST，保守。
    actuator_delay_ticks: int = 1
    #: 舵机：二阶临界阻尼 + 速率限幅。bandwidth 与 rate 都是 EST。
    head_bw_rad: float = 14.0           # rad/s（≈2.2 Hz）
    head_zeta: float = 0.9
    head_rate_max_dps: float = 420.0
    #: 头部质量偏心引起的反作用力矩（头部加减速 → 机身俯仰扰动）。EST，很小。
    head_reaction_gain: float = 2.0e-5  # N·m per (deg/s²)
    # --- 航向 ----------------------------------------------------------- #
    #: yaw_u → 机身角加速度的系数，(deg/s²)/torque-count。**标定量**。
    k_yaw_accel: float = 1.6
    # --- 传感器噪声（EST，取 IMU 手册量级）------------------------------ #
    gyro_noise_dps: float = 0.25
    accel_pitch_noise_deg: float = 0.30
    wheel_vel_noise_mps: float = 0.02
    #: 头部实测角的位置分辨率（伺服反馈 1024 计数/300°，motor.cc:57）
    head_quant_deg: float = 300.0 / 1024.0

    #: 哪些字段是估计值（REPORT 里必须列出来）
    EST = (
        "wheel_track_m", "mass_body_kg", "mass_wheel_kg", "inertia_wheel_kgm2",
        "inertia_body_kgm2", "com_height_m", "yaw_inertia_kgm2", "yaw_damping",
        "wheel_rolling_damping", "k_torque", "actuator_delay_ticks", "head_bw_rad",
        "head_zeta", "head_rate_max_dps", "head_reaction_gain", "k_yaw_accel",
        "gyro_noise_dps", "accel_pitch_noise_deg", "wheel_vel_noise_mps",
    )


# =========================================================================== #
# 3. 符号约定（有一步是"为了让固件增益能闭环稳定"而定的，必须写明）            #
# =========================================================================== #
@dataclass(frozen=True)
class SignConvention:
    """三个符号约定。固件源码里**没有**写死它们，真机也没验证过。

    * ``pitch_sign``: 机体前倾时 ``pitch`` 读数是正还是负。
      ``-1`` 表示前倾 → ``pitch`` 变负。选 ``-1`` 的理由：只有这个极性下
      ``lqr_k2=+25`` 才构成负反馈（见 REPORT §局限）。
    * ``torque_sign``: ``temp_u>0`` 时轮子把机器人往前推还是往后推。``+1``。
    * ``yaw_sign``: ``yaw_u>0`` 时机身 yaw 增还是减。``-1``。
      固件 ``yaw_u = 4*(yaw - q_head - stable_yaw) = 4*(ψ_body - ψ_ref)``，即
      "误差的负"，所以只有 ``yaw_sign=-1`` 才构成负反馈；``+1`` 会让航向环
      正反馈发散（``sim.calibrate`` 与 ``closed_loop_eigs`` 都验证这一点）。
    """

    pitch_sign: int = -1
    torque_sign: int = 1
    yaw_sign: int = -1


# =========================================================================== #
# 4. 汇总                                                                     #
# =========================================================================== #
@dataclass(frozen=True)
class SimParams:
    """一份完整的仿真参数（固件 + 机体 + 符号 + 噪声 + 场景）。"""

    fw: FirmwareParams = field(default_factory=FirmwareParams)
    plant: PlantParams = field(default_factory=PlantParams)
    sign: SignConvention = field(default_factory=SignConvention)
    #: 关闭全部噪声（标定/阶跃实验用）
    deterministic: bool = False
    #: 视觉有效视场的一半 = fov_h/2（control_arbiter.c:109）
    fov_half_deg: float = 30.0
    #: 头部/机身耦合：pitch 测量按 cos(q_head) 缩放（固件自己的补偿模型
    #: ``pitch_ref = imu_zero*cos(q_head)``，robot.cc:264）
    cos_coupling: bool = True
    #: 机身 yaw = yaw_imu - q_head（robot.cc:139 / TECHNICAL-REPORT §3.3）
    channel_separation: bool = True


def provenance_table():
    """REPORT §参数出处表 的行：``(参数, 值, 出处, 是否估计)``。"""
    fw = FirmwareParams()
    pl = PlantParams()
    rows = []
    for name in _SRC:
        rows.append((name, getattr(fw, name), _SRC[name], False))
    for name in ("wheel_track_m", "mass_body_kg", "mass_wheel_kg", "com_height_m",
                 "inertia_body_kgm2", "inertia_wheel_kgm2", "yaw_inertia_kgm2",
                 "yaw_damping", "wheel_rolling_damping", "k_torque",
                 "head_bw_rad", "head_zeta", "head_rate_max_dps",
                 "head_reaction_gain", "k_yaw_accel", "actuator_delay_ticks",
                 "gyro_noise_dps", "accel_pitch_noise_deg", "wheel_vel_noise_mps"):
        rows.append((name, getattr(pl, name), "估计（无真机实测）", True))
    rows.append(("pitch_sign / torque_sign / yaw_sign",
                 "-1 / +1 / -1", "约定，为让 lqr_k[2]=+25 构成负反馈而选", True))
    return rows


if __name__ == "__main__":
    w = max(len(r[0]) for r in provenance_table())
    for name, val, src, est in provenance_table():
        tag = "EST" if est else "SRC"
        print("%-4s %-*s %-12s %s" % (tag, w, name, ("%.6g" % val), src))
