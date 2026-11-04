# -*- coding: utf-8 -*-
"""sim.py — RIG-Hover 闭环仿真：机体动力学 + 传感器模型 + 固件控制律副本。

设计目标只有一个：**能反映闭环本质**，不是高保真。四个维度各自独立建模：

=========== ==================================================== ==============
维度        模型                                                  出处 / 依据
=========== ==================================================== ==============
平衡        轮式倒立摆（线性化，小车-摆两自由度）               物理，参数见 params
航向        双积分 + 黏性阻尼                                   纯 P 控双积分 → 欠阻尼
头部        二阶临界阻尼 + 速率限幅 + ±45° 硬限位               舵机 EST
视觉        头部相机，FOV 60°，50 Hz，含丢帧                     control_arbiter.c:109
=========== ==================================================== ==============

**固件部分逐行复刻**，不是重新设计：

* ``robot_control()``：:class:`FirmwareControl`，对应 robot.cc:192-326
  （LQR + yaw P + 死区补偿 + Clip，以及 tor1 = -temp_u + yaw_u / tor2 = temp_u + yaw_u）
* ``update_state()``：:meth:`FirmwareControl.update_state`，对应 robot.cc:120-190
  （``stable_pos`` / ``stable_yaw`` 引用积分、``robot_state`` 站立/摔倒状态机）
* ``control_arbiter.c``：:class:`Arbiter`，对应 arbiter_tick_2ms + safety_shell
  （head 一阶低通、pos/yaw 提前量钳位、视觉时效三档、失联看门狗）
* IMU：:class:`Imu`，对应 imu.cc:234-271（互补滤波、dq 低通、yaw 死区）

时间步严格 6 ms（robot.cc:295/302/315 三次 ``vTaskDelay(2)``），
视觉 50 Hz（control_arbiter.c:108 ``cfc_hz``），控制 166.7 Hz。

**离散化**：平衡维与航向维都是线性系统 ``ż = Az + Bτ``，用矩阵指数做精确 ZOH
离散（:func:`_zoh`），不是欧拉积分 —— 6 ms 步长下欧拉积分会让不稳定极点的
增长率偏差几个百分点，而这里正好只有一个不稳定极点。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from params import DEG, FirmwareParams, PlantParams, SimParams

RAD2DEG = 180.0 / math.pi
G = 9.81

ARB_SRC_MANUAL, ARB_SRC_TEACHER, ARB_SRC_CFC = 0, 1, 2


def wrap180(deg):
    """把角度折到 [-180, 180)。"""
    return (float(deg) + 180.0) % 360.0 - 180.0


# =========================================================================== #
# ZOH 离散化                                                                  #
# =========================================================================== #
def _expm(M, terms=20):
    """缩放-平方 + Taylor 级数（一个小矩阵，不引入 scipy 依赖）。"""
    norm = float(np.abs(M).max())
    s = 0
    while norm > 0.25:
        norm *= 0.5
        s += 1
    Ms = M / (2.0 ** s)
    out = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms + 1):
        term = term @ Ms / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def _zoh(A, B, dt):
    """线性系统的精确零阶保持离散化。

    Args:
        A: ``(n, n)``。
        B: ``(n, m)``。
        dt: 步长（秒）。

    Returns:
        ``(Ad, Bd)``，满足 ``z[k+1] = Ad z[k] + Bd u[k]``。
    """
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = _expm(M * dt)
    return E[:n, :n], E[:n, n:]


# =========================================================================== #
# 机体动力学                                                                  #
# =========================================================================== #
@dataclass
class PlantState:
    """机体的完整真实状态（仿真内部真值，传感器只能看到其中一部分）。"""

    x_w: float = 0.0          # 轮轴位移 (m)
    v_w: float = 0.0          # (m/s)
    theta: float = 0.0        # 前倾角 (rad)，正 = 前倾
    theta_dot: float = 0.0    # (rad/s)
    yaw_body: float = 0.0     # 机身朝向 (deg)
    yaw_rate: float = 0.0     # (deg/s)
    q_head: float = 0.0       # 头相对机身角 (deg)
    qd_head: float = 0.0      # (deg/s)
    qdd_head: float = 0.0     # (deg/s²)


class Plant:
    """轮式倒立摆 + 航向双积分 + 头部伺服。

    平衡维连续方程（拉格朗日，见 REPORT §方法）::

        M·ẍ + m_b·l·θ̈ = τ/r − c_v·ẋ
        m_b·l·ẍ + I·θ̈ = m_b·g·l·θ

    ⇒ ``ẍ = (I/D)(τ/r) − (m_b²l²g/D)θ``，``θ̈ = (m_b·g·l·M/D)θ − (m_b·l/(D·r))τ``，
    其中 ``M = m_b + 2m_w + 2I_w/r²``，``I = I_b + m_b·l²``，``D = M·I − (m_b·l)²``。

    头部伺服：二阶 ``q̈ = ω²(q_des − q) − 2ζω·q̇``，带速率限幅与 ±45° 限位。
    头部反作用力矩进航向维（纯 yaw 反力，不进俯仰）。

    Args:
        sp: 仿真参数。
        dt: 控制步长（秒），默认取固件的 6 ms。
    """

    def __init__(self, sp: SimParams, dt: float | None = None):
        self.sp = sp
        self.fw: FirmwareParams = sp.fw
        self.pp: PlantParams = sp.plant
        self.dt = (self.fw.ctrl_dt_ms * 1e-3) if dt is None else float(dt)

        r = self.pp.wheel_radius_m
        mb = self.pp.mass_body_kg
        mw = self.pp.mass_wheel_kg
        Iw = self.pp.inertia_wheel_kgm2
        Ib = self.pp.inertia_body_kgm2
        l = self.pp.com_height_m

        M = mb + 2.0 * mw + 2.0 * Iw / (r * r)
        I = Ib + mb * l * l
        D = M * I - (mb * l) ** 2
        if D <= 1e-12:
            raise ValueError("倒立摆参数退化：D<=0，检查质量/惯量")

        #: 状态顺序 [x, v, θ, θ̇]
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[1, 1] = -self.pp.wheel_rolling_damping
        A[1, 2] = -(mb * mb * l * l * G) / D
        A[2, 3] = 1.0
        A[3, 2] = (mb * G * l * M) / D
        B = np.zeros((4, 1))
        B[1, 0] = (I / D) / r
        B[3, 0] = -(mb * l) / (D * r)

        self.M, self.I, self.D = M, I, D
        self.mb, self.l = mb, l
        #: 开环不稳定极点 σ = sqrt(m_b·g·l·M/D)（1/s）
        self.sigma_open_loop = math.sqrt(A[3, 2])
        self.Ad, self.Bd = _zoh(A, B, self.dt)

        # 航向维：[ψ, ψ̇]，输入 yaw_u（torque-count）
        Ay = np.array([[0.0, 1.0], [0.0, -self.pp.yaw_damping]])
        By = np.array([[0.0], [sp.sign.yaw_sign * self.pp.k_yaw_accel]])
        self.Ay, self.By = _zoh(Ay, By, self.dt)

        # 头部伺服
        w = self.pp.head_bw_rad
        Ah = np.array([[0.0, 1.0], [-w * w, -2.0 * self.pp.head_zeta * w]])
        Bh = np.array([[0.0], [w * w]])
        self.Ah, self.Bh = _zoh(Ah, Bh, self.dt)

        self.st = PlantState()

    # ------------------------------------------------------------------ #
    def reset(self, pitch0_deg=1.0, seed=0):
        """初始状态：直立 + ``pitch0_deg`` 的前倾扰动，头部回中。"""
        self.st = PlantState(theta=math.radians(pitch0_deg), theta_dot=0.0)

    # ------------------------------------------------------------------ #
    def step(self, torque_cmd, yaw_cmd, head_cmd_deg):
        """推进一个控制拍。

        Args:
            torque_cmd: 差模力矩指令（固件的 ``temp_u``，int16 计数）。
            yaw_cmd: 共模力矩指令（固件的 ``yaw_u``，int16 计数）。
            head_cmd_deg: 头部目标角（度），已经过 arbiter 的低通。

        Returns:
            ``(qdd_head_prev, qdd_head_now)``：头部角加速度（deg/s²）。
        """
        sp = self.sp
        st = self.st
        tau = sp.sign.torque_sign * self.pp.k_torque * float(torque_cmd)   # N·m

        z = np.array([st.x_w, st.v_w, st.theta, st.theta_dot])
        z = self.Ad @ z + self.Bd[:, 0] * tau
        st.x_w, st.v_w, st.theta, st.theta_dot = (
            float(z[0]), float(z[1]), float(z[2]), float(z[3]))

        # ---- 头部伺服 ----
        qdd_prev = st.qdd_head
        h = np.array([st.q_head, st.qd_head])
        h = self.Ah @ h + self.Bh[:, 0] * float(head_cmd_deg)
        q, qd = float(h[0]), float(h[1])
        lim = self.pp.head_rate_max_dps
        qd = max(-lim, min(lim, qd))
        lim_deg = self.fw.head_max_deg
        if q > lim_deg:
            q, qd = lim_deg, min(qd, 0.0)
        elif q < -lim_deg:
            q, qd = -lim_deg, max(qd, 0.0)
        st.qdd_head = (qd - st.qd_head) / self.dt
        st.q_head, st.qd_head = q, qd

        # ---- 航向：偏航力矩 + 头部反作用 ----
        y = np.array([st.yaw_body, st.yaw_rate])
        y = self.Ay @ y + self.By[:, 0] * float(yaw_cmd)
        st.yaw_body = wrap180(float(y[0]))
        # 头部绕竖直轴加减速 → 机身反向力矩（EST，params.plant.head_reaction_gain）
        st.yaw_rate = float(y[1]) - self.pp.head_reaction_gain * st.qdd_head
        return qdd_prev, st.qdd_head

    # ------------------------------------------------------------------ #
    @property
    def pitch_true_deg(self):
        """真实俯仰角（IMU 口径，°）：``pitch_sign`` 把前倾映射成读数符号。"""
        return self.sp.sign.pitch_sign * self.st.theta * RAD2DEG

    @property
    def pitch_rate_true_dps(self):
        return self.sp.sign.pitch_sign * self.st.theta_dot * RAD2DEG


# =========================================================================== #
# 传感器                                                                      #
# =========================================================================== #
@dataclass
class SensorView:
    """一帧传感器读数（= 固件里同名的全局变量）。"""

    roll: float = 0.0
    pitch: float = 0.0
    pit_gyro: float = 0.0
    dq: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0      # d(yaw)/dt (deg/s)，imu.cc 新增的全局 yaw_rate
    q_head: float = 0.0
    qd_head: float = 0.0       # 头部角速度 (deg/s)，只给"用机身角速度做 D"的对照用
    target_head_pos: float = 0.0
    wheel1_vel: float = 0.0
    wheel2_vel: float = 0.0
    wheel_x: float = 0.0
    stable_pos: float = 0.0
    stable_yaw: float = 0.0
    robot_state: int = 1
    vx: float = 0.0
    vyaw: float = 0.0
    control_mode: int = 2
    feature_age_ms: float = 0.0
    gaze_danger: int = 0
    #: 轮速低通的一拍延迟状态（固件里是 wheel1_vel 全局量本身）
    _w1: float = 0.0
    _w2: float = 0.0


class Imu:
    """头部 IMU 模型，复刻 imu.cc:234-271。

    两个固件细节必须保留，否则闭环不是同一个系统：

    1. **头部角度进测量轴**：``pit_gyro`` / ``pit_acc`` 都先用 ``q_head`` 旋转
       （imu.cc:240-245）。线性化后等价于按 ``cos(q_head)`` 缩放俯仰信号 ——
       正是固件用 ``pitch_ref = imu_zero*cos(q_head)`` 补偿的那一项。
    2. **yaw 纯陀螺积分 + 1.5 °/s 死区**（imu.cc:264-270）：低速转向时航向
       估计根本不动，闭环里表现为"慢转迟钝"。

    Args:
        sp: 仿真参数。
        rng: ``np.random.Generator``；``None`` 或无噪声模式下不注入噪声。
    """

    def __init__(self, sp: SimParams, rng=None):
        self.sp = sp
        self.rng = rng
        self.pitch = 0.0
        self.pit_gyro = 0.0
        self.dq = 0.0
        self.yaw = 0.0
        self.gyro_yaw = 0.0
        self.yaw_rate = 0.0
        self.roll = 0.0

    def _n(self, sigma):
        if self.rng is None or self.sp.deterministic:
            return 0.0
        return float(self.rng.normal(0.0, sigma))

    def reset(self):
        self.pitch = self.pit_gyro = self.dq = self.yaw = self.gyro_yaw = 0.0
        self.yaw_rate = 0.0

    def update(self, plant: Plant, q_head_meas: float, dt: float):
        """推进一个采样周期，更新 ``pitch / dq / yaw``。

        Args:
            plant: 机体真值。
            q_head_meas: 头部实测角（度），固件 ``q_head`` 来自伺服回读。
            dt: 采样间隔（秒）。
        """
        fw = self.sp.fw
        cpl = math.cos(q_head_meas * DEG) if self.sp.cos_coupling else 1.0
        self.pit_gyro = plant.pitch_rate_true_dps * cpl + self._n(
            self.sp.plant.gyro_noise_dps)
        pit_acc = plant.pitch_true_deg * cpl + self._n(
            self.sp.plant.accel_pitch_noise_deg)

        # dq = 0.6*dq + 0.4*pit_gyro（200 Hz 上标定，按步长折算）
        a = 1.0 - fw.dq_lpf_keep ** (dt * 1000.0 / fw.imu_dt_ms)
        self.dq += a * (self.pit_gyro - self.dq)

        # 互补滤波 pitch = f*pit_acc + (1-f)*(pitch + pit_gyro*dt)，f=1/(|ω|+200)
        f = 1.0 / (abs(self.pit_gyro) + fw.pit_freq_k)
        if abs(pit_acc) < 100.0 and abs(self.pit_gyro) < 1000.0:
            self.pitch = f * pit_acc + (1.0 - f) * (self.pitch + self.pit_gyro * dt)

        # yaw：头部 IMU 绝对 yaw = 机身 yaw + q_head（所以 yaw − q_head = 机身）
        gyro_y = -(plant.st.yaw_rate + plant.st.qd_head) + self._n(
            self.sp.plant.gyro_noise_dps)
        a2 = 1.0 - fw.yaw_lpf_keep ** (dt * 1000.0 / fw.imu_dt_ms)
        self.gyro_yaw += a2 * (gyro_y - self.gyro_yaw)
        g = 0.0 if abs(self.gyro_yaw) < fw.yaw_deadzone_dps else self.gyro_yaw
        self.yaw = wrap180(self.yaw - g * dt)
        # imu.cc 新增的全局：yaw = yaw - g*dt ⇒ d(yaw)/dt = -g（含死区、含头部
        # 角速度：imu 装在头上，所以这是"视线"角速度，不是纯机身角速度）
        self.yaw_rate = -g
        self.roll = 0.6 * math.sin(0.7 * plant.st.x_w) + self._n(0.2)


# =========================================================================== #
# 固件控制律副本                                                              #
# =========================================================================== #
class FirmwareControl:
    """``robot.cc`` 的 ``update_state()`` + ``robot_control()`` 逐行副本。

    这里**没有**"改进"：Clip 的写法、``±20/±10`` 的死区补偿、
    ``tor1 = -temp_u + yaw_u`` 的符号都照抄。唯一的可选项是
    ``params.FirmwareParams.lqr_mode``（LQR 增益口径），默认不是源码原值，
    原因见 :func:`params.FirmwareParams.gains` 与 REPORT §固件稳定性。
    """

    def __init__(self, sp: SimParams):
        self.sp = sp
        self.fw = sp.fw
        #: LQR 的 4 个增益，口径见 ``params.FirmwareParams.gains``
        self.k = sp.fw.gains()
        self.stable_pos = 0.0
        self.stable_yaw = 0.0
        self.robot_state = 1        # 仿真从"已被扶正并保持 1.5 s"开始
        self.fall_since = None
        self.upright_since = None
        self.pick_up_count = 0
        self.yaw_ctrl_time = 0.0
        self.temp_u = 0.0
        self.yaw_u = 0.0
        #: 航向环 D 增益。默认取 ``FirmwareParams.k_dyaw``（=0，纯 P）；
        #: ``controllers.CfCDamped`` 之类的控制器可以在重置后覆盖它。
        self.k_dyaw = sp.fw.k_dyaw
        #: 0 = D 项用固件现有的 ``yaw_rate``（= 视线角速度 = 机身 + 头部）；
        #: 1 = 先减掉头部角速度，只留机身角速度。**默认 0 是当前固件的真实
        #: 行为**；1 是 REPORT_DYAW §4 那条"头部角速度污染"的对照实验，固件
        #: 要做这一步得先把 q_head 回读差分出来。
        self.head_rate_comp = 0.0

    # ------------------------------------------------------------------ #
    def update_state(self, sv: SensorView, t_s: float):
        """robot.cc:120-190：``robot_state`` 站立/摔倒状态机。

        Args:
            sv: 本拍传感器视图（就地更新 ``robot_state``）。
            t_s: 单调时间（秒）。
        """
        fw = self.fw
        fallen = abs(sv.pitch) >= fw.fallen_pitch_deg or abs(sv.roll) >= 40.0
        upright = abs(sv.pitch) < fw.upright_pitch_deg and abs(sv.roll) < 7.0
        if fallen:
            self.fall_since = t_s if self.fall_since is None else self.fall_since
            self.upright_since = None
            if t_s - self.fall_since >= fw.fall_hold_ms * 1e-3:
                self.robot_state = 0
        else:
            self.fall_since = None
            if upright:
                self.upright_since = (t_s if self.upright_since is None
                                      else self.upright_since)
                if t_s - self.upright_since >= fw.stand_hold_ms * 1e-3:
                    self.robot_state = 1
            else:
                self.upright_since = None

        if abs(sv.wheel1_vel) > 350.0 or abs(sv.wheel2_vel) > 350.0:
            self.pick_up_count += 1
            if self.pick_up_count > 8:
                self.robot_state = 0
        else:
            self.pick_up_count = 0
        sv.robot_state = self.robot_state

    # ------------------------------------------------------------------ #
    def control(self, sv: SensorView, t_s: float, do_update_state: bool = True):
        """robot.cc:192-326 的力矩部分。

        Args:
            sv: 本拍传感器视图。
            t_s: 单调时间（秒）。
            do_update_state: 调用方已经跑过 ``update_state()`` 时传 ``False``
                （固件里它在 ``arbiter_tick_2ms`` 之前、每拍只跑一次）。

        Returns:
            ``(tor1, tor2, head_cmd_deg)``。
        """
        fw = self.fw
        if do_update_state:
            self.update_state(sv, t_s)

        k0, k1, k2, k3 = self.k
        # robot.cc:263-273
        pitch_ref = fw.imu_zero * math.cos(sv.q_head * DEG)
        lqr_x = sv.wheel_x - self.stable_pos
        lqr_vx = 0.5 * (sv.wheel1_vel + sv.wheel2_vel) - sv.vx
        lqr_q = sv.pitch - pitch_ref
        lqr_dq = sv.dq
        temp_u = -(k0 * lqr_x + k1 * lqr_vx + k2 * lqr_q + k3 * lqr_dq)
        # robot.cc:275-279 死区补偿
        if temp_u < -fw.hyst_thresh:
            temp_u -= fw.hyst_bump
        elif temp_u > fw.hyst_thresh:
            temp_u += fw.hyst_bump
        temp_u = max(-fw.torque_cmd_max, min(fw.torque_cmd_max, temp_u))

        # robot.cc:283-284 航向 P + D
        yaw_rate_d = sv.yaw_rate - self.head_rate_comp * sv.qd_head
        yaw_u = (fw.k_yaw * (sv.yaw - sv.q_head - self.stable_yaw)
                 - self.k_dyaw * yaw_rate_d)
        yaw_u = max(-fw.yaw_u_clip, min(fw.yaw_u_clip, yaw_u))

        # robot.cc:285-291 差/共模分配
        if self.robot_state == 0:
            tor1 = tor2 = 0.0
        else:
            tor1 = -temp_u + yaw_u
            tor2 = temp_u + yaw_u
        self.temp_u, self.yaw_u = temp_u, yaw_u
        return tor1, tor2, max(-fw.head_max_deg,
                               min(fw.head_max_deg, sv.target_head_pos))


# =========================================================================== #
# arbiter 副本                                                                #
# =========================================================================== #
@dataclass
class CtrlOut:
    """``ctrl_out_t``（control_arbiter.h:37-43）。"""

    head_des: float = 0.0
    vx_des: float = 0.0
    vyaw_des: float = 0.0
    mode: int = 0
    emote: int = 0


class Arbiter:
    """``control_arbiter.c`` 的 ``arbiter_tick_2ms()`` + ``safety_shell()`` 副本。

    只实现影响闭环行为的部分：视觉时效三档、硬钳位、头部一阶低通、
    ``stable_pos``/``stable_yaw`` 的提前量积分（pos_lead_max / yaw_lead_max）。
    """

    def __init__(self, sp: SimParams, src: int = ARB_SRC_CFC):
        self.sp = sp
        self.fw = sp.fw
        self.src = src
        self.head_filt = 0.0
        self.head_init = False
        self.blocked = 0
        self.raw = CtrlOut()
        self.pub = CtrlOut()
        self.raw_ms = -1e9
        self.feat_ms = -1e9
        #: 命令速率限幅状态（None = 还没初始化）。默认 ``vx_rate_max=0`` 关闭，
        #: 所以原有的 4 个控制器结果不变；``scan_dyaw.py`` 会显式打开它。
        self.vx_prev = None
        self.vyaw_prev = None

    @staticmethod
    def _rate_limit(prev, target, max_step, stop_eps, snap):
        """``control_arbiter.c`` 的 ``rate_limit()`` 副本。

        吸附判据用 ``|target| <= stop_eps``，不是 ``target <= -limit + eps``
        —— 后者在允许负速度（倒车脱困）时会把全速倒车吸附成 0
        （smartcar/code/lnn/action_shaper.py 的 ``stop_speed`` 注释）。
        """
        if max_step <= 0.0:
            return target
        if snap or prev is None:
            return 0.0 if snap else target
        d = max(-max_step, min(max_step, target - prev))
        out = prev + d
        if abs(target) <= stop_eps:
            out = 0.0
        return out

    def post_raw(self, out: CtrlOut, t_s: float):
        """``arbiter_post_raw()``：非有限值丢弃（control_arbiter.c:291-318）。"""
        if not all(math.isfinite(v) for v in
                   (out.head_des, out.vx_des, out.vyaw_des)):
            out = CtrlOut()
        self.raw = out
        self.raw_ms = t_s * 1000.0

    def note_feature(self, t_s: float):
        """``arbiter_note_feat()``：记一帧视觉新鲜度。"""
        self.feat_ms = t_s * 1000.0

    def feature_age_ms(self, t_s: float):
        if self.feat_ms < 0:
            return -1.0
        return t_s * 1000.0 - self.feat_ms

    def tick(self, sv: SensorView, ctrl: FirmwareControl, t_s: float):
        """``arbiter_tick_2ms()``：安全壳 → head 低通 → 引用积分。"""
        fw = self.fw
        age = self.feature_age_ms(t_s)
        o = CtrlOut(**vars(self.raw))
        block = 0

        if sv.robot_state == 0:                      # 摔倒/力矩关
            o = CtrlOut()
            block |= 2
        if age < 0 or age > fw.lost_ms:              # 视觉丢失
            o.vx_des = o.vyaw_des = o.head_des = 0.0
            block |= 8
        elif age > fw.stale_ms:                      # 视觉过期：冻结头部
            o.vx_des = o.vyaw_des = 0.0
            block |= 8
        if self.src != ARB_SRC_MANUAL:               # 指令看门狗
            if t_s * 1000.0 - self.raw_ms > fw.cmd_timeout_ms:
                o.vx_des = o.vyaw_des = 0.0
                if self.src == ARB_SRC_TEACHER:
                    o.head_des = 0.0
                block |= 16
        o.head_des = max(-fw.head_max_deg, min(fw.head_max_deg, o.head_des))
        o.vx_des = max(-fw.vx_max, min(fw.vx_max, o.vx_des))
        o.vyaw_des = max(-fw.vyaw_max, min(fw.vyaw_max, o.vyaw_des))

        # 命令速率限幅（control_arbiter.c 安全壳第 7 步）。放在所有钳位之后，
        # 且在任何保护动作触发的 tick 上"吸附"而不是斜坡——停车不许排队。
        dt = fw.ctrl_dt_ms * 1e-3
        snap = block != 0
        self.vx_prev = self._rate_limit(self.vx_prev, o.vx_des,
                                        fw.vx_rate_max * dt, fw.vx_stop_eps, snap)
        self.vyaw_prev = self._rate_limit(self.vyaw_prev, o.vyaw_des,
                                          fw.vyaw_rate_max * dt, fw.vyaw_stop_eps,
                                          snap)
        o.vx_des, o.vyaw_des = self.vx_prev, self.vyaw_prev

        # 头部一阶低通（control_arbiter.c:484-494），每 tick 一次
        if not self.head_init:
            self.head_filt = o.head_des
            self.head_init = True
        else:
            self.head_filt += fw.head_lpf * (o.head_des - self.head_filt)
        o.head_des = max(-fw.head_max_deg, min(fw.head_max_deg, self.head_filt))
        self.pub = o
        self.blocked = block

        # 引用积分（control_arbiter.c:505-540）
        yref = sv.yaw - sv.q_head
        if 0.0 <= age <= fw.stale_ms and math.isfinite(sv.wheel_x):
            ctrl.stable_pos = min(sv.wheel_x + fw.pos_lead_max,
                                  max(sv.wheel_x - fw.pos_lead_max,
                                      ctrl.stable_pos + o.vx_des * dt))
            ctrl.stable_yaw = min(yref + fw.yaw_lead_max,
                                  max(yref - fw.yaw_lead_max,
                                      ctrl.stable_yaw + o.vyaw_des * dt))
        # 刷新 yaw_ctrl_time，否则 robot.cc:138-140 的 3 s 冻结会取消引用
        ctrl.yaw_ctrl_time = t_s * 1000.0
        sv.target_head_pos = o.head_des
        sv.vx = o.vx_des
        sv.vyaw = o.vyaw_des
        sv.stable_pos = ctrl.stable_pos
        sv.stable_yaw = ctrl.stable_yaw


# =========================================================================== #
# 场景（人）                                                                  #
# =========================================================================== #
SCENARIOS = ("orbit", "approach", "fov_exit", "wave", "mixed")

#: 场景说明（REPORT 里直接用）
SCENARIO_ZH = {
    "orbit": "人绕机器人画弧，方位 ±50°，距离 1.2~1.8 m",
    "approach": "人前后移动，距离 2.6→0.95→2.6 m，方位小幅漂移",
    "fov_exit": "人走出视野（|方位|>30°）再回来",
    "wave": "人站定挥手（触发 gesture_onset），方位固定",
    "mixed": "上面四段的串联（长轨迹）",
}


@dataclass
class Scene:
    """一条人的运动脚本，已按控制步长采样成数组。

    Attributes:
        name: 场景名。
        t: 时间（秒）。
        bearing: 人的绝对方位（度）。
        dist: 与机器人的距离（米）。
        visible: 人是否"在场"（走出房间 = 0；只是转出视野仍为 1）。
        hand_up / waving: 手势基元。
        onset: ``gesture_onset`` 边沿（一段动作只有一个 1）。
    """

    name: str
    t: np.ndarray
    bearing: np.ndarray
    dist: np.ndarray
    visible: np.ndarray
    hand_up: np.ndarray
    waving: np.ndarray
    onset: np.ndarray

    def __len__(self):
        return int(self.t.shape[0])


def make_scene(name: str, seed: int, duration: float, dt: float) -> Scene:
    """按场景名生成人的轨迹。

    Args:
        name: ``SCENARIOS`` 之一。
        seed: 随机种子（决定抖动）。
        duration: 时长（秒）。
        dt: 采样步长（秒）。

    Returns:
        :class:`Scene`
    """
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    n = int(round(duration / dt)) + 1
    t = np.arange(n, dtype=np.float64) * dt
    T = float(duration)
    j = lambda a: float(rng.uniform(-a, a))  # noqa: E731

    if name == "orbit":
        br = np.interp(t, [0, 0.15 * T, 0.75 * T, T],
                       [-20 + j(4), -45 + j(6), 45 + j(6), 50 + j(6)])
        br = br + 4.0 * np.sin(2 * math.pi * 0.55 * t + j(1.0))
        d = 1.5 + 0.3 * np.sin(2 * math.pi * 0.23 * t + j(1.0))
        hu = np.zeros(n, dtype=bool)
        wv = np.zeros(n, dtype=bool)
    elif name == "approach":
        br = np.interp(t, [0, 0.3 * T, 0.8 * T, T],
                       [-8 + j(5), 4 + j(5), -5 + j(5), 2 + j(5)])
        d = np.interp(t, [0, 0.35 * T, 0.55 * T, 0.78 * T, T],
                      [2.6, 0.95, 0.95, 2.4, 2.6])
        d = d + 0.05 * np.sin(2 * math.pi * 0.4 * t)
        hu = np.zeros(n, dtype=bool)
        wv = np.zeros(n, dtype=bool)
    elif name == "fov_exit":
        br = np.interp(t, [0, 0.2 * T, 0.42 * T, 0.62 * T, 0.8 * T, T],
                       [j(4), 10 + j(4), 78 + j(8), 78 + j(8), -12 + j(5), j(4)])
        d = 1.6 + 0.15 * np.sin(2 * math.pi * 0.2 * t)
        hu = np.zeros(n, dtype=bool)
        wv = np.zeros(n, dtype=bool)
    elif name == "wave":
        br = np.full(n, 6.0 + j(6))
        d = np.full(n, 1.3 + j(0.2))
        t0 = 0.30 * T
        hu = (t >= t0 - 0.2) & (t < t0 + 4.3)
        wv = (t >= t0) & (t < t0 + 2.8)
    elif name == "mixed":
        br = np.interp(t, [0, 0.16 * T, 0.38 * T, 0.5 * T, 0.62 * T, 0.72 * T,
                           0.88 * T, T],
                       [-18 + j(6), -6 + j(6), 44 + j(10), 44 + j(10), 24 + j(6),
                        78 + j(10), -24 + j(6), j(4)])
        d = np.interp(t, [0, 0.16 * T, 0.27 * T, 0.38 * T, 0.5 * T, 0.62 * T,
                          0.72 * T, 0.88 * T, T],
                      [2.6, 1.05, 1.75, 1.10, 1.10, 1.45, 2.5, 1.6, 1.35])
        t0 = 0.42 * T
        hu = (t >= t0 - 0.2) & (t < t0 + 4.3)
        wv = (t >= t0) & (t < t0 + 2.8)
    else:
        raise ValueError("未知场景 %r，可选 %s" % (name, SCENARIOS))

    hu = np.asarray(hu, dtype=bool)
    wv = np.asarray(wv, dtype=bool)
    prev = np.roll(t < 0, 1)          # 全 False 的左移一格的初值
    onset = wv & ~np.concatenate([prev[:1], wv[:-1]])
    return Scene(name=name, t=t, bearing=np.asarray(br, dtype=np.float64),
                 dist=np.clip(np.asarray(d, dtype=np.float64), 0.45, 4.0),
                 visible=np.ones(n), hand_up=hu, waving=wv, onset=onset)


# =========================================================================== #
# 观测组装（对齐 pc/train/obs.py / cfc_task.c build_obs）                     #
# =========================================================================== #
@dataclass
class Frame:
    """一帧感知 + 本体感知：控制器能看到的全部信息。"""

    t_s: float
    theta_img: float
    q_head: float
    seen: int
    person_h: float
    person_w: float
    person_conf: float
    dist_est: float
    hand_up_L: int
    hand_up_R: int
    wave_sin: float
    wave_cos: float
    gesture_conf: float
    gesture_onset: int
    gesture_id: int
    feature_age_ms: float
    eps_body: float
    #: 64 维视觉特征（固件 ``f->feat``）
    feat: np.ndarray = field(default=None, repr=False)
    #: 106 维观测（CfC / MLP 吃这个）
    obs: np.ndarray = field(default=None, repr=False)


def feat_projection():
    """``feat_64`` 的合成投影（与 ``pc/train/synth.py:70`` 同一个固定种子）。

    真机上是 YOLO+FeatureAdapter 的输出；闭环仿真里它只是"视觉特征"的占位，
    必须与训练时同分布，否则 CfC 会落到域外（那是另一个问题，REPORT §局限）。
    """
    return np.random.default_rng(1234).normal(0.0, 1.0, size=(6, 64)).astype(np.float32)


_FEAT_PROJ = feat_projection()


def synth_feat(theta_img, dist, seen, hand_up, waving, conf, noise=None):
    """与 ``pc/train/synth.py:207`` 逐位一致的 64 维特征合成。"""
    scene = np.array([
        math.sin(float(theta_img) / 30.0),
        math.cos(float(theta_img) / 30.0),
        float(dist) / 2.0,
        float(seen),
        float(bool(hand_up) and bool(waving)),
        float(conf),
    ], dtype=np.float32)
    if noise is not None:
        scene = scene + np.asarray(noise, dtype=np.float32)
    return np.tanh(scene @ _FEAT_PROJ).astype(np.float32)


class DistEstimator:
    """复刻 cfc_task.c:347-362 的距离低通（与 pc/train/obs.py 同构）。"""

    def __init__(self, dist_k=0.90, tau=0.50, target=1.0, hz=50.0):
        self.dist_k, self.tau, self.target, self.hz = dist_k, tau, target, hz
        self.dist = 0.0

    def reset(self):
        self.dist = 0.0

    def step(self, seen, person_h):
        """推进一帧，返回送进 ``obs[67]`` 的距离（米）。"""
        dt = 1.0 / self.hz
        a = min(max(dt / (self.tau + dt), 0.0), 1.0)
        raw = (self.dist_k / person_h) if (seen and person_h > 0.02) else self.dist
        if self.dist <= 0.0:
            self.dist = self.target
        self.dist += a * (raw - self.dist)
        self.dist = min(max(self.dist, 0.05), 20.0)
        return float(self.dist)


def build_obs(frame: Frame, sv: SensorView, prev_act):
    """组装 106 维观测（``cfc_task.c:326 build_obs()`` 的等价实现）。

    Args:
        frame: 本帧感知。
        sv: 本帧传感器视图。
        prev_act: 上一帧动作 ``(head, vyaw, vx, mode)`` 或 ``None``。

    Returns:
        ``(106,) float32``
    """
    o = np.zeros(106, dtype=np.float32)
    if frame.feat is not None:
        o[0:64] = frame.feat[:64]
    o[64] = frame.theta_img
    o[65] = frame.person_w
    o[66] = frame.person_h
    o[67] = frame.dist_est
    o[68] = frame.person_conf
    o[69] = float(frame.seen)
    o[70] = float(frame.hand_up_L)
    o[71] = float(frame.hand_up_R)
    o[72] = frame.wave_sin
    o[73] = frame.wave_cos
    o[74] = min(math.hypot(frame.wave_sin, frame.wave_cos), 1.0)
    o[75] = frame.gesture_conf
    # 76/77 固件写死 0
    o[78] = sv.roll
    o[79] = sv.pitch
    o[80] = sv.pit_gyro
    o[81] = sv.dq
    o[82] = sv.yaw
    o[83] = sv.q_head
    o[84] = sv.target_head_pos
    o[85] = sv.q_head - sv.target_head_pos
    o[86] = sv.wheel1_vel
    o[87] = sv.wheel2_vel
    o[88] = sv.wheel1_vel - sv.wheel2_vel
    o[89] = sv.wheel_x
    o[90] = sv.stable_pos
    o[91] = sv.wheel_x - sv.stable_pos
    o[92] = sv.yaw - sv.q_head
    o[93] = (sv.yaw - sv.q_head) - sv.stable_yaw
    o[94] = float(sv.robot_state)
    o[95] = sv.vx
    o[96] = sv.vyaw
    o[97] = float(sv.control_mode)
    o[98] = -1.0 if frame.feature_age_ms < 0 else frame.feature_age_ms / 1000.0
    o[99] = float(sv.gaze_danger)
    ph, pv, pvy, pm = prev_act if prev_act is not None else (0.0, 0.0, 0.0, 0)
    o[100] = ph / 45.0
    o[101] = pv / 90.0
    o[102] = pm / 3.0
    o[103] = pvy / 0.30
    ang = 2.0 * math.pi * frame.t_s / 10.0
    o[104] = math.sin(ang)
    o[105] = math.cos(ang)
    return o


# =========================================================================== #
# 闭环运行器                                                                  #
# =========================================================================== #
@dataclass
class Trace:
    """一条闭环轨迹，所有数组等长、按控制拍（6 ms）采样。"""

    t: np.ndarray
    eps_true: np.ndarray           # 真实 ε_body = 人绝对方位 − 机身朝向 (deg)
    eps_meas: np.ndarray           # 观测到的 ε_body = theta_img + q_head
    theta_img: np.ndarray
    seen: np.ndarray
    in_fov: np.ndarray
    q_head: np.ndarray             # 头部实测角 (deg)
    q_head_des: np.ndarray         # arbiter 输出的头部目标 (deg)
    body_yaw: np.ndarray
    body_rate: np.ndarray
    pitch: np.ndarray
    vyaw_des: np.ndarray           # 发布的角速度指令 (deg/s)
    vx_des: np.ndarray
    head_cmd_action: np.ndarray    # 控制器原始输出（未过 arbiter 低通）
    vyaw_cmd_action: np.ndarray
    vx_cmd_action: np.ndarray
    mode: np.ndarray
    temp_u: np.ndarray
    yaw_u: np.ndarray
    robot_state: np.ndarray
    dist: np.ndarray
    x_w: np.ndarray                # 轮轴位移 (m)，看"有没有漂走"
    warm: np.ndarray
    fell: bool = False


class ClosedLoop:
    """把 Plant / Imu / Arbiter / FirmwareControl / 策略 串成闭环。

    一拍内的顺序严格按固件（robot.cc:192-326）：

    1. 读传感器（IMU + 伺服回读）
    2. （每 20 ms）新视觉帧 → 策略出 head/vyaw/vx → ``arbiter_post_raw()``
    3. ``update_state()`` → ``robot_state``
    4. ``arbiter_tick_2ms()`` → 安全壳 + head LPF + 引用积分
    5. LQR + yaw P → tor1/tor2/head_cmd
    6. 机体积分（力矩延迟 1 拍，头部指令当拍生效）

    Args:
        sp: 仿真参数。
        policy: 实现 ``reset()`` 与 ``step(frame, prev_act, sv) -> CtrlOut`` 的对象。
        scene: 人的轨迹。
        seed: 噪声种子。
        warmup_s: 前多少秒不计入指标。
        perturbation_deg: 初始俯仰扰动。
    """

    def __init__(self, sp: SimParams, policy, scene: Scene, seed: int = 0,
                 warmup_s: float = 1.5, perturbation_deg: float = 1.0):
        self.sp = sp
        self.policy = policy
        self.scene = scene
        self.seed = int(seed)
        self.warmup_s = float(warmup_s)
        self.perturbation_deg = float(perturbation_deg)
        self.dt = sp.fw.ctrl_dt_ms * 1e-3
        self.vision_period = 1.0 / sp.fw.cfc_hz

    # ------------------------------------------------------------------ #
    def run(self) -> Trace:
        sp = self.sp
        dt = self.dt
        n = len(self.scene)
        rng = np.random.default_rng(self.seed * 104729 + 7)
        feat_rng = np.random.default_rng(self.seed * 33 + 5)

        plant = Plant(sp, dt)
        plant.reset(self.perturbation_deg, self.seed)
        imu = Imu(sp, None if sp.deterministic else rng)
        fw = FirmwareControl(sp)
        # 控制器可以声明自己的航向 D 增益（``controllers.CfCDamped``）；没声明的
        # 就沿用固件参数 ``FirmwareParams.k_dyaw``（默认 0）。
        if getattr(self.policy, "k_dyaw", None) is not None:
            fw.k_dyaw = float(self.policy.k_dyaw)
        if getattr(self.policy, "head_rate_comp", None) is not None:
            fw.head_rate_comp = float(self.policy.head_rate_comp)
        arb = Arbiter(sp, ARB_SRC_CFC)
        dist_est = DistEstimator()
        self.policy.reset()
        sv = SensorView()

        prev_act = None
        next_vision = 0.0
        t_last_seen = None
        ndelay = max(1, sp.plant.actuator_delay_ticks)
        u_delay = [0.0] * ndelay
        y_delay = [0.0] * ndelay

        phase = amp = score = prev_sin = 0.0
        prev_onset = False

        keys = ("eps_true", "eps_meas", "theta_img", "seen", "in_fov", "q_head",
                "q_head_des", "body_yaw", "body_rate", "pitch", "vyaw_des",
                "vx_des", "head_cmd_action", "vyaw_cmd_action", "vx_cmd_action",
                "mode", "temp_u", "yaw_u", "robot_state", "dist", "x_w")
        cols = {k: np.zeros(n, dtype=np.float64) for k in keys}
        fell = False
        cur = dict(action=(0.0, 0.0, 0.0, 0), theta=0.0, seen=0, in_fov=0,
                   eps=0.0, dist=0.0)

        for k in range(n):
            t = k * dt
            # ---------------- 1) 传感器 ----------------
            q_meas = round(plant.st.q_head / sp.plant.head_quant_deg) * \
                sp.plant.head_quant_deg
            imu.update(plant, q_meas, dt)
            sv.roll, sv.pitch = imu.roll, imu.pitch
            sv.pit_gyro, sv.dq, sv.yaw = imu.pit_gyro, imu.dq, imu.yaw
            sv.yaw_rate = imu.yaw_rate
            sv.q_head = q_meas
            sv.qd_head = plant.st.qd_head
            wv = plant.st.v_w + (0.0 if sp.deterministic else
                                 float(rng.normal(0, sp.plant.wheel_vel_noise_mps)))
            a = sp.fw.dq_lpf_keep ** (dt * 1000.0 / sp.fw.imu_dt_ms)
            sv.wheel1_vel = a * sv._w1 + (1.0 - a) * wv
            sv.wheel2_vel = a * sv._w2 + (1.0 - a) * wv
            sv._w1, sv._w2 = sv.wheel1_vel, sv.wheel2_vel
            sv.wheel_x = plant.st.x_w

            # ---------------- 2) 视觉帧 ----------------
            # 相位累加而不是 "t - t_last >= 20ms"：6 ms 整拍下后者只能落在
            # 24 ms（41.7 Hz），视觉帧率会系统性低 20%，而 CfC 是按 50 Hz 训的。
            if k == 0 or t + 1e-9 >= next_vision:
                next_vision = (self.vision_period if k == 0
                               else next_vision + self.vision_period)
                fr = self._perceive(t, plant, k, dist_est, feat_rng, t_last_seen,
                                    phase, amp, score, prev_sin, prev_onset)
                phase, amp, score, prev_sin, prev_onset = fr["gs"]
                if fr["seen"]:
                    t_last_seen = t
                frame = Frame(
                    t_s=t, theta_img=fr["theta_img"], q_head=sv.q_head,
                    seen=int(fr["seen"]), person_h=fr["person_h"],
                    person_w=fr["person_w"], person_conf=fr["conf"],
                    dist_est=fr["dist_est"], hand_up_L=0,
                    hand_up_R=int(fr["hand_up"]), wave_sin=fr["wave_sin"],
                    wave_cos=fr["wave_cos"], gesture_conf=fr["gesture_conf"],
                    gesture_onset=int(fr["onset"]), gesture_id=1,
                    # 与 pc/train/synth.py:370 同口径：从未见过 → 0，不是 -1
                    # （-1 落在训练分布外，CfC 在首个视觉帧之前就会跑飞）
                    feature_age_ms=(0.0 if t_last_seen is None
                                    else (t - t_last_seen) * 1000.0),
                    eps_body=fr["eps_meas"], feat=fr["feat"])
                frame.obs = build_obs(frame, sv, prev_act)
                out = self.policy.step(frame, prev_act, sv)
                arb.post_raw(out, t)
                arb.note_feature(t)
                prev_act = (out.head_des, out.vyaw_des, out.vx_des, out.mode)
                cur["action"] = prev_act
                cur["theta"] = fr["theta_img"]
                cur["seen"] = int(fr["seen"])
                cur["in_fov"] = int(fr["in_fov"])
                cur["eps"] = fr["eps_meas"]
                cur["dist"] = fr["dist"]

            # ---------------- 3) + 4) ----------------
            fw.update_state(sv, t)
            arb.tick(sv, fw, t)

            # ---------------- 5) 控制律 ----------------
            tor1, tor2, head_cmd = fw.control(sv, t, do_update_state=False)
            u_used = u_delay.pop(0)
            y_used = y_delay.pop(0)
            u_delay.append(fw.temp_u)
            y_delay.append(fw.yaw_u)
            fell = fell or bool(sv.robot_state == 0)

            # ---------------- 6) 机体 ----------------
            plant.step(u_used, y_used, head_cmd)

            # ---------------- 记录 ----------------
            cols["eps_true"][k] = wrap180(float(self.scene.bearing[k])
                                          - plant.st.yaw_body)
            cols["eps_meas"][k] = cur["eps"]
            cols["theta_img"][k] = cur["theta"]
            cols["seen"][k] = float(cur["seen"])
            cols["in_fov"][k] = float(cur["in_fov"])
            cols["q_head"][k] = plant.st.q_head
            cols["q_head_des"][k] = sv.target_head_pos
            cols["body_yaw"][k] = plant.st.yaw_body
            cols["body_rate"][k] = plant.st.yaw_rate
            cols["pitch"][k] = sv.pitch
            cols["vyaw_des"][k] = sv.vyaw
            cols["vx_des"][k] = sv.vx
            cols["head_cmd_action"][k] = cur["action"][0]
            cols["vyaw_cmd_action"][k] = cur["action"][1]
            cols["vx_cmd_action"][k] = cur["action"][2]
            cols["mode"][k] = cur["action"][3]
            cols["temp_u"][k] = fw.temp_u
            cols["yaw_u"][k] = fw.yaw_u
            cols["robot_state"][k] = sv.robot_state
            cols["dist"][k] = cur["dist"]
            cols["x_w"][k] = plant.st.x_w

        t = np.arange(n, dtype=np.float64) * dt
        return Trace(t=t, warm=(t >= self.warmup_s).astype(np.float64),
                     fell=fell, **cols)

    # ------------------------------------------------------------------ #
    def _perceive(self, t, plant, k, dist_est, rng, t_last_seen, phase, amp,
                  score, prev_sin, prev_onset):
        """把几何真值变成一帧感知（含 FOV、丢帧、手势基元）。"""
        sp = self.sp
        sc = self.scene
        gaze = plant.st.yaw_body + plant.st.q_head      # §3.3 的 ψ_gaze
        theta_geo = wrap180(float(sc.bearing[k]) - gaze)
        in_fov = abs(theta_geo) <= sp.fov_half_deg
        seen = bool(sc.visible[k] > 0.5 and in_fov)
        # 没检测到时 theta_img 无意义 → 0（真机语义）。
        # synth.py 训练时喂的是几何值，这是训练/部署口径差异，REPORT §局限单列。
        theta_img = theta_geo if seen else 0.0

        d = float(sc.dist[k])
        person_h = min(max(0.90 / d, 0.03), 1.0) if seen else 0.0
        conf = float(np.clip(0.55 + 0.40 * math.exp(-d / 2.5), 0.0, 1.0)) if seen else 0.0

        dt = self.dt
        hand_up = bool(sc.hand_up[k])
        waving = bool(sc.waving[k]) and seen
        target_amp = 0.62 if waving else (0.25 if hand_up else 0.0)
        amp += (target_amp - amp) * min(1.0, dt / 0.08)
        if waving:
            phase += 2.0 * math.pi * 2.00 * dt
        elif hand_up:
            phase += 2.0 * math.pi * 0.20 * dt
        wave_sin = amp * math.sin(phase)
        wave_cos = amp * math.cos(phase)
        if hand_up and amp > 0.35:
            if (prev_sin <= 0.0 < wave_sin) or (prev_sin >= 0.0 > wave_sin):
                score += 1.0
        prev_sin = wave_sin
        score *= math.exp(-dt / 2.0)
        gesture_conf = float(conf * min(1.0, score / 4.0))
        onset = bool(score >= 4.0 and not prev_onset)

        d_est = dist_est.step(seen, person_h)
        noise = (None if sp.deterministic
                 else rng.normal(0, 0.05, size=6).astype(np.float32))
        feat = synth_feat(theta_img, d, seen, hand_up, waving, conf, noise)
        return {
            "theta_img": theta_img, "in_fov": in_fov, "seen": seen,
            "person_h": person_h, "person_w": float(min(max(0.40 * person_h, 0.0), 1.0)),
            "conf": conf, "dist_est": d_est, "feat": feat,
            "hand_up": hand_up, "wave_sin": wave_sin, "wave_cos": wave_cos,
            "gesture_conf": gesture_conf, "onset": onset, "dist": d,
            "eps_meas": wrap180(theta_img + plant.st.q_head),
            "gs": (phase, amp, score, prev_sin, onset),
        }


# =========================================================================== #
# 标定 / 稳定性分析                                                           #
# =========================================================================== #
def closed_loop_eigs(sp: SimParams, k_torque=None, k_yaw_accel=None,
                     k_dyaw=None):
    """平衡维闭环极点（连续时间，不含采样与执行器延迟）。

    Args:
        sp: 仿真参数。
        k_torque: 覆盖 ``plant.k_torque``。
        k_yaw_accel: 覆盖 ``plant.k_yaw_accel``。
        k_dyaw: 覆盖 ``fw.k_dyaw``（航向环 D 增益）。

    Returns:
        ``(eigs_pitch, eigs_yaw)``：两个复数数组。
    """
    pp = sp.plant
    plant = Plant(sp)
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 1] = -pp.wheel_rolling_damping
    A[1, 2] = -(plant.mb ** 2 * plant.l ** 2 * G) / plant.D
    A[2, 3] = 1.0
    A[3, 2] = plant.mb * G * plant.l * plant.M / plant.D
    B = np.array([0.0, (plant.I / plant.D) / pp.wheel_radius_m, 0.0,
                  -(plant.mb * plant.l) / (plant.D * pp.wheel_radius_m)])
    # 状态 [x, v, θ(rad), θ̇(rad/s)]；控制器输入用 (m, m/s, deg, deg/s)
    sg = sp.sign.pitch_sign * RAD2DEG
    k0, k1, k2, k3 = sp.fw.gains()
    K = np.array([k0, k1, k2 * sg, k3 * sg])
    kt = pp.k_torque if k_torque is None else float(k_torque)
    Acl = A - np.outer(B * (sp.sign.torque_sign * kt), K)
    e_p = np.linalg.eigvals(Acl)

    ky = pp.k_yaw_accel if k_yaw_accel is None else float(k_yaw_accel)
    # ψ̈ = -d·ψ̇ + c·yaw_u，yaw_u = k_yaw·(ψ − ψ_ref) − k_dyaw·ψ̇
    # ⇒ [[0, 1], [c·k_yaw, −d − c·k_dyaw]]。c<0（yaw_u>0 使 ψ̈<0），所以阻尼
    # 需要 c·k_dyaw>0，即 **k_dyaw<0** 时 zeta 变大。
    c = sp.sign.yaw_sign * ky
    kd = sp.fw.k_dyaw if k_dyaw is None else float(k_dyaw)
    Acl_y = np.array([[0.0, 1.0],
                      [c * sp.fw.k_yaw, -pp.yaw_damping - c * kd]])
    return e_p, np.linalg.eigvals(Acl_y)


def damping_ratio(lam):
    """极点 → 阻尼比（实极点返回 1，正实部返回负值）。"""
    lam = complex(lam)
    w = abs(lam)
    if w < 1e-12:
        return 1.0
    if abs(lam.imag) < 1e-9:
        return 1.0 if lam.real < 0 else -1.0
    return -lam.real / w


def calibrate(sp: SimParams, verbose=True, wn_min=15.0, settle_target=0.35):
    """标定 ``k_torque`` 与 ``k_yaw_accel``。

    判据（都写进 REPORT）：

    1. **平衡维**：扫 ``k_torque``，要求 4 个闭环极点全在左半平面、俯仰模态
       （频率最高那对复极点）阻尼 ``ζ >= 0.25``、带宽 ``ωn >= wn_min``；
       满足条件的点里取**最小** ``k_torque``（力矩最保守）。
    2. **航向维**：扫 ``k_yaw_accel``，让机身 yaw 对 60° 参考阶跃的 90% 上升
       时间 = ``settle_target``（含 ``yaw_u`` 的 ±80 饱和），取满足的最小增益。

    这两个量是**仿真自洽**的标定值，不是真机实测：``k_torque`` 是
    "torque-count → N·m"，``k_yaw_accel`` 是 "yaw_u → deg/s²"。真机上一个都
    没测过（REPORT §局限）。

    Args:
        sp: 仿真参数。
        verbose: 打印标定过程。
        wn_min: 俯仰模态带宽下界（rad/s）。
        settle_target: 航向阶跃 90% 上升时间目标（秒）。

    Returns:
        ``(k_torque, k_yaw_accel, info)``。

    Raises:
        RuntimeError: 该 ``lqr_mode`` 下无解（这本身就是结论）。
    """
    # ---- 平衡维 ----
    best, scan = None, []
    for kt in np.logspace(-6, 0, 121):
        e_p, _ = closed_loop_eigs(sp, k_torque=kt)
        if np.any(e_p.real >= 0):
            scan.append((kt, float("nan"), float("nan"), False))
            continue
        cplx = sorted([l for l in e_p if abs(l.imag) > 1e-9],
                      key=lambda l: -abs(l.imag))
        zeta = damping_ratio(cplx[0]) if cplx else 1.0
        wn = abs(cplx[0]) if cplx else float(np.max(np.abs(e_p)))
        ok = zeta >= 0.25 and wn >= wn_min
        scan.append((kt, float(zeta), float(wn), ok))
        if ok and best is None:
            best = (kt, float(zeta), float(wn))
    if best is None:
        raise RuntimeError(
            "标定失败：lqr_mode=%r 下没有任何 k_torque 能让平衡环稳定并达到 "
            "ζ>=0.25 / ωn>=%.0f。这条信息本身就是结论：见 REPORT §固件稳定性。"
            % (sp.fw.lqr_mode, wn_min))
    k_torque, zeta_p, wn_p = best
    e_p, _ = closed_loop_eigs(sp, k_torque=k_torque)

    # ---- 航向维 ----
    def step_settle(ky):
        sp2 = replace(sp, plant=replace(sp.plant, k_yaw_accel=ky))
        plant = Plant(sp2)
        y = np.zeros(2)              # [ψ, ψ̇]，ψ 从 0 起，参考跳到 60°
        ref, t = 60.0, 0.0
        while t < 6.0:
            yaw_u = sp2.fw.k_yaw * (y[0] - ref)
            yaw_u = max(-sp2.fw.yaw_u_clip, min(sp2.fw.yaw_u_clip, yaw_u))
            y = plant.Ay @ y + plant.By[:, 0] * yaw_u
            t += plant.dt
            if y[0] >= 0.9 * ref:
                return t
        return float("inf")

    lo, hi = 0.005, 500.0
    if step_settle(lo) <= settle_target:
        hi = lo
    elif step_settle(hi) > settle_target:
        raise RuntimeError("航向维标定无解：k_yaw_accel=%.1f 仍追不上 60°/s 阶跃" % hi)
    else:
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            # settle 随 ky 单调下降 → 找"刚好够快"的最小 ky
            if step_settle(mid) < settle_target:
                hi = mid
            else:
                lo = mid
    k_yaw_accel = hi
    _, e_y = closed_loop_eigs(sp, k_torque=k_torque, k_yaw_accel=k_yaw_accel)
    oks = [s[0] for s in scan if s[3]]
    info = {
        "lqr_mode": sp.fw.lqr_mode,
        "k_torque": k_torque, "zeta_pitch": zeta_p, "wn_pitch": wn_p,
        "poles_pitch": e_p, "poles_yaw": e_y,
        "k_yaw_accel": k_yaw_accel, "yaw_settle_s": step_settle(k_yaw_accel),
        "stability_window": (min(oks), max(oks)) if oks else (float("nan"),) * 2,
        "scan": scan,
    }
    if verbose:
        print("[calib] mode=%s  k_torque=%.4e  (ζ_pitch=%.3f, ω_n=%.2f rad/s)"
              % (sp.fw.lqr_mode, k_torque, zeta_p, wn_p))
        print("[calib]   平衡维极点 %s" % np.array2string(e_p, precision=3))
        print("[calib]   可行域        k_torque ∈ [%.2e, %.2e]"
              % info["stability_window"])
        print("[calib]   k_yaw_accel=%.4f  (60° 阶跃 90%% 上升 %.3f s)"
              % (k_yaw_accel, info["yaw_settle_s"]))
        print("[calib]   航向维极点 %s" % np.array2string(e_y, precision=3))
    return k_torque, k_yaw_accel, info


def calibrated(sp: SimParams, verbose=False, wn_min=15.0):
    """返回把两个标定量代进去的 :class:`SimParams`。"""
    kt, ky, _ = calibrate(sp, verbose=verbose, wn_min=wn_min)
    return replace(sp, plant=replace(sp.plant, k_torque=kt, k_yaw_accel=ky))


# =========================================================================== #
# 阶跃实验                                                                    #
# =========================================================================== #
def step_scene(step_deg=25.0, t0=2.0, duration=9.0, bearing0=0.0,
               dt=0.006) -> Scene:
    """人方位 0° → ``step_deg`` 的阶跃（测收敛时间与超调）。

    **为什么默认 25° 而不是 40°**：摄像头 FOV 只有 60°（±30°，
    ``control_arbiter.c:109``），人一步跨到 40° 就直接落在视野外 ——
    此时所有控制器拿到的都是 ``seen=0``，测出来的是"丢失目标后怎么办"，
    不是"跟踪阶跃"。40° 的版本单独用 :func:`reacquire_metrics` 量。
    """
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    bearing = np.where(t < t0, bearing0, bearing0 + step_deg)
    return Scene(name="step", t=t, bearing=bearing,
                 dist=np.full(n, 1.4), visible=np.ones(n),
                 hand_up=np.zeros(n, dtype=bool),
                 waving=np.zeros(n, dtype=bool),
                 onset=np.zeros(n, dtype=bool))


def reacquire_metrics(tr: Trace, t0=2.0, band=15.0):
    """大幅阶跃（>FOV 半角）之后多久重新看见人并压回 ``band`` 以内。

    Args:
        tr: :class:`Trace`。
        t0: 阶跃时刻（秒）。
        band: 重新捕获的误差界（度）。

    Returns:
        dict：``reacq_s``（首次满足 视野内且 |ε|<band；不满足为 NaN）、
        ``lost_frac``（阶跃后 ``seen=0`` 的帧比例）、``eps_peak``。
    """
    m = tr.t >= t0
    tt = tr.t[m] - t0
    seen = tr.seen[m] > 0.5
    ok = seen & (np.abs(tr.eps_true[m]) < band)
    reacq = float(tt[np.argmax(ok)]) if np.any(ok) else float("nan")
    return {"reacq_s": reacq, "lost_frac": float(np.mean(~seen)),
            "eps_peak": float(np.max(np.abs(tr.eps_true[m])))}


def step_metrics(tr: Trace, step_deg=40.0, t0=2.0):
    """从阶跃轨迹里量上升时间 / 超调 / 调节时间。

    Args:
        tr: :class:`Trace`。
        step_deg: 阶跃幅度（度）。
        t0: 阶跃时刻（秒）。

    Returns:
        dict：``rise_s`` / ``settle_s`` / ``overshoot_pct`` / ``ss_err`` /
        ``peak_err``。
    """
    t = tr.t
    m = t >= t0
    tt, ee = t[m] - t0, tr.eps_true[m]
    if tt.size < 2:
        return {"rise_s": float("nan"), "settle_s": float("nan"),
                "overshoot_pct": float("nan"), "ss_err": float("nan"),
                "peak_err": float("nan")}
    band10, band5 = 0.10 * step_deg, 0.05 * step_deg
    rise = float("nan")
    idx = np.where(np.abs(ee) <= band10)[0]
    if idx.size:
        rise = float(tt[idx[0]])
    # 超调：ε 反向穿零后的最大幅度 / 阶跃幅度
    sign0 = np.sign(ee[0]) or 1.0
    cross = np.where(np.sign(ee) != sign0)[0]
    overshoot = 0.0
    if cross.size:
        after = ee[cross[0]:]
        overshoot = float(max(0.0, -np.min(after * sign0)) / step_deg) * 100.0
    outside = np.where(np.abs(ee) > band5)[0]
    settle = float("nan")
    if outside.size and outside[-1] + 1 < tt.size:
        settle = float(tt[outside[-1] + 1])
    tail_n = max(1, int(0.5 / max(tt[1] - tt[0], 1e-6)))
    tail = ee[-tail_n:]
    return {"rise_s": rise, "settle_s": settle, "overshoot_pct": overshoot,
            "ss_err": float(np.mean(np.abs(tail))),
            "peak_err": float(np.max(np.abs(ee)))}
