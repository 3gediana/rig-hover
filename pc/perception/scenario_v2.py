# -*- coding: utf-8 -*-
"""scenario_v2.py -- 场景 v2 的决策层：盯着人脸 → 盯拳头 → 敲两下 → 走过去 → 停一会儿再看人脸。

`SCENARIO-v2.md` 的三个细节决定了这一层的形状：

1. **工作距离 1–2 米**。太近整屏都是脸、太远没必要。所以 ``face_scale`` 是**双端
   门限**（``scale_min`` / ``scale_max``），不是"越大越好"。
2. **敲击位置要记住、要盯着走过去**。所以目标不是一个瞬间事件，而是一个
   :class:`TargetLock`：敲击那一帧锁定，之后每帧只要还能看见手就**用观测修正**，
   看不见就靠里程计走完剩下的距离。这就是"哪怕拳头离开"的实现方式。
3. **看人的时候优先动人头，头够不到才动轮子**。这一段直接复用教师控制器
   （`pc/teacher/teacher.py::_allocate` §7.2）的三档分配律：
   ``|eps| < body_engage_deg`` 时轮子绝对不转，只有头在动；头到限位才强制机身转。
   本模块重写了一份结构相同的实现，是为了让它能脱离教师单独跑和单测，
   参数含义与教师逐条对应。

本模块是纯 numpy、无 torch、无网络，可直接单测。

坐标约定（与教师一致）：``eps_body = theta + q_head``，即图像方位角加上当前头角
换算到机身坐标系；正 = 目标在右。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

__all__ = [
    "MODE_IDLE", "MODE_GAZE_FACE", "MODE_GAZE_FIST", "MODE_APPROACH", "MODE_DWELL",
    "MODE_NAMES", "V2Params", "TargetLock", "V2Action", "ScenarioV2",
]

MODE_IDLE = 0
MODE_GAZE_FACE = 1
MODE_GAZE_FIST = 2
MODE_APPROACH = 3
MODE_DWELL = 4
MODE_NAMES = {
    MODE_IDLE: "IDLE",
    MODE_GAZE_FACE: "GAZE_FACE",
    MODE_GAZE_FIST: "GAZE_FIST",
    MODE_APPROACH: "APPROACH",
    MODE_DWELL: "DWELL",
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class V2Params:
    """场景参数。

    三档分配的几个量**与教师控制器同名同义**，默认值取自教师配置量级；
    其余是场景新增的。**全部尚未在真机标定**——标定时改这里。
    """
    # ---- 人脸门（1–2 米，双端）----
    #: 人脸尺度下限：低于它说明太远，不盯
    face_scale_min: float = 0.045
    #: 人脸尺度上限：高于它说明太近（整屏是脸），不盯
    face_scale_max: float = 0.230

    # ---- 注视分配（相机在头上，头对准 == 目标在画面中心）----
    head_max_deg: float = 45.0
    #: 死区：目标已经足够居中时不动，避免抖动。**不要调大**——它直接等于
    #: "允许的斜眼角度"，用户明确要求目标尽量在画面中心。
    head_deadband_deg: float = 2.0
    #: 头距限位还有这么多度时就开始让机身帮转，避免顶死
    head_limit_margin_deg: float = 2.0
    k_fast: float = 12.0
    vyaw_max_dps: float = 90.0
    vyaw_floor_dps: float = 12.0

    # ---- 拳头 ----
    #: 判定为"人手握拳、该优先盯拳头"的概率阈值
    fist_prob_on: float = 0.55
    fist_prob_off: float = 0.30
    #: 拳头消失这么久才退回看脸（秒）
    fist_lost_s: float = 0.40

    # ---- 前进 ----
    vx_max: float = 0.15
    #: 转向对准的容差（度）：大于它就先原地转，不走
    turn_tol_deg: float = 6.0
    k_turn: float = 8.0
    #: 距离误差 → 速度
    k_v: float = 1.2
    #: 到点判定（米）
    arrive_tol_m: float = 0.06
    #: **过冲保险**（米）。从估计距离里预先扣掉这么多，
    #: 保证**任何距离估计误差都不会导致冲过头**。
    #:
    #: 用户 2026-11-03 决定不做桌沿检测，理由是"敲击点在桌上，精准停在
    #: 那里就不可能掉下去"。这个推理成立，但**只在不过冲的前提下成立**：
    #: 停在敲击点附近是安全的，冲过头一次在桌沿就是硬件损失。
    #: 所以代价是**故意留一个稳定的欠调量**——少走 0.12 米只是位置差一点，
    #: 多走 0.12 米可能是摔下去。两边代价不对称，所以必须偏保守。
    #: 想把这个值调小，前提是先把距离估计做准（标定 + 用手部尺度而不是
    #: 人体尺度估距），而不是直接改小这个数。
    overshoot_margin_m: float = 0.12
    #: 走过去最多给多久（秒），超时就停（防呆）。
    #: 要大于 最远距离/vx_max；1.7 m / 0.15 m/s ≈ 11 s，留一倍余量。
    approach_timeout_s: float = 12.0
    #: 人比这个近就不要往前走了
    stop_if_face_scale: float = 0.260

    # ---- 停住之后 ----
    dwell_s: float = 2.5

    # ---- 距离估计 ----
    #: person_h(=框高/画面高) → 距离(米) 的比例常数：d ≈ dist_k / person_h。
    #: 0.84 对应 person_h=0.70 时约 1.2 米，是**先验猜测**，必须在真机标定。
    dist_k: float = 0.84
    #: person_h 小于它就不给距离估计
    dist_person_h_min: float = 0.06


@dataclass
class TargetLock:
    """锁定的敲击点。

    ``bearing_body`` 是机身坐标系里的方位角（度），``dist_m`` 是距离（米）。
    只要手还在视野里就每帧用观测修正 ``bearing_body``，手没了就保持最后的值，
    靠里程计走完 ``remaining_m``。
    """
    valid: int = 0
    t_lock: float = 0.0
    #: 锁定瞬间的图像像素坐标（溯源用）
    x_px: float = 0.0
    y_px: float = 0.0
    bearing_body: float = 0.0
    dist_m: float = 0.0
    #: 还需要走多远（米）
    remaining_m: float = 0.0
    #: 锁定后是否还看见过手（用于判断是"盯着走"还是"盲走"）
    last_seen_t: float = -1.0
    updates: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class V2Action:
    head_des: float = 0.0
    vyaw_des: float = 0.0
    vx_des: float = 0.0
    mode: int = MODE_IDLE
    tier: int = 0
    debug: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["mode_name"] = MODE_NAMES.get(self.mode, "?")
        return d


class ScenarioV2:
    """场景 v2 状态机。

    每帧一次 :meth:`step`，输入感知信号与本体状态，输出三路指令与当前模式。
    """

    def __init__(self, params: V2Params | None = None):
        self.p = params or V2Params()
        self.reset()

    def reset(self):
        self.t = 0.0
        self.mode = MODE_IDLE
        self.target = TargetLock()
        self._t_mode = 0.0
        self._t_fist_lost = 0.0
        self._t_approach = 0.0
        self._wheel_x0 = None
        self._arrived = False
        self.history: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    def _enter(self, mode: int):
        if mode != self.mode:
            self.mode = mode
            self._t_mode = self.t
            self.history.append((round(self.t, 3), mode))

    def _allocate(self, eps: float) -> tuple[float, float, int]:
        """注视分配律：**头一次性对准目标，机身只在头顶到限位时才转**。

        与教师控制器 ``teacher._allocate`` 的**关键区别**：教师在中/大误差档会给
        头留 30% 余量（``head = eps*0.7``）或让头反向补偿（``head = eps - 25``），
        那是为了让机身有个"驱动源"。但相机装在头上，头一旦没对准，目标就偏离
        画面中心——**看起来就是在斜眼看人**。用户明确不要这个。

        所以这里的规则是：

        * ``|eps| <= head_max``：``head = eps``，目标**精确落在画面中心**，
          机身不动（"脖子够得到就别动轮子"）。
        * ``|eps| > head_max``：头只能顶到 ±45°，剩下的角度必须由机身转出来。
          一旦机身转到 ``|eps| <= head_max``，头立刻重新精确对准。

        **头没有死区**：只要目标偏了，头就把它转回中心。把头压在一个死区里
        不动，等于允许一个永久偏角，那正是"斜眼"。抖动交给下游的头部低通
        （固件 `head_lpf`）去滤，不由这里承担。``head_deadband_deg`` 只用于
        在 debug 里报告"已经足够居中"，不参与指令。

        Returns:
            (head_cmd, vyaw_cmd, tier)；``tier`` 0=已居中 1=头独任 2=头顶限位
        """
        p = self.p
        a = abs(eps)
        s = 1.0 if eps >= 0.0 else -1.0
        # 头满量程对准：相机在头上，头对准 == 目标在画面中心
        head = _clamp(eps, -p.head_max_deg, p.head_max_deg)
        head_reach = p.head_max_deg - p.head_limit_margin_deg
        if a <= head_reach:
            vyaw = 0.0
            tier = 0 if a <= p.head_deadband_deg else 1
        else:
            # 头顶死了，只能靠机身把这个角度吃掉；转速按超出量给，给个下限防卡死
            over = a - head_reach
            vyaw = s * max(p.vyaw_floor_dps,
                           min(p.k_fast * over, p.vyaw_max_dps))
            tier = 2
        vyaw = _clamp(vyaw, -p.vyaw_max_dps, p.vyaw_max_dps)
        return head, vyaw, tier

    @staticmethod
    def off_center_deg(eps: float, head_cmd: float) -> float:
        """注视残差 = 相机指向与目标方向的夹角（度）。

        这是"斜不斜眼"的直接量化：**0 表示目标精确落在画面中心**。
        非零只应出现在头已经顶到 ±45° 限位的时候。
        """
        return float(eps) - float(head_cmd)

    def _person_dist(self, person_h: float) -> float:
        p = self.p
        if person_h < p.dist_person_h_min:
            return 0.0
        return float(p.dist_k / person_h)

    # ------------------------------------------------------------------
    def step(self, dt: float, *, theta_face: float = 0.0, face_scale: float = 0.0,
             face_valid: int = 0, theta_fist: float = 0.0,
             wrist_y_norm: float = 0.0, fist_prob: float = 0.0,
             knock_fired: int = 0, knock_x: float = 0.0, knock_y: float = 0.0,
             q_head: float = 0.0, wheel_x: float = 0.0,
             person_h: float = 0.0, hand_valid: int = 0) -> V2Action:
        """推进一帧。

        Args:
            dt: 秒。
            theta_face: 人脸方位角（度，正=右，头部相机系）。
            face_scale: 瞳距/图高。
            face_valid: 0/1。
            theta_fist: 主导手方位角（度）。
            wrist_y_norm: 腕部归一化高度（保留，供上层判断"手在桌面高度"）。
            fist_prob: 分类器给出的 fist 概率。
            knock_fired: 敲击检测器的触发位。
            knock_x / knock_y: 敲击点图像坐标。
            q_head: 当前头角（度）。
            wheel_x: 里程计累计位移（米）。
            person_h: 人框高/图高，用于距离估计。
            hand_valid: 0/1，腕部本帧是否可用。**不能用 theta_fist 是否等于 0
                来判断**——0 度是一个合法方位角。

        Returns:
            V2Action
        """
        p = self.p
        self.t += max(0.0, float(dt))
        dt = max(1e-6, float(dt))

        in_range = bool(face_valid) and (
            p.face_scale_min <= face_scale <= p.face_scale_max)

        # ---------------- 模式切换 ----------------
        if not face_valid:
            self._enter(MODE_IDLE)
            self.target.valid = 0
        elif self.mode == MODE_DWELL:
            # 停在原地看一会儿脸，时间到了再回到正常跟随
            if (self.t - self._t_mode) >= p.dwell_s:
                self._enter(MODE_GAZE_FACE if in_range else MODE_IDLE)
        elif not in_range and self.mode in (MODE_IDLE, MODE_GAZE_FACE,
                                            MODE_GAZE_FIST):
            # 太近（整屏是脸，看不到）或太远（没必要追）：不跟。
            # APPROACH 不在这里中断——目标已经锁定，走完它；
            # "人贴太近就停"的保护在 _approach 里单独判。
            self._enter(MODE_IDLE)
        elif self.mode == MODE_IDLE and in_range:
            self._enter(MODE_GAZE_FACE)

        if self.mode == MODE_GAZE_FACE and fist_prob >= p.fist_prob_on:
            self._enter(MODE_GAZE_FIST)
            self._t_fist_lost = self.t
        elif self.mode == MODE_GAZE_FIST:
            if fist_prob >= p.fist_prob_off:
                self._t_fist_lost = self.t
            elif (self.t - self._t_fist_lost) > p.fist_lost_s:
                self._enter(MODE_GAZE_FACE)

        # 敲击只在"正盯着拳头"时才算数
        if knock_fired and self.mode == MODE_GAZE_FIST:
            self._lock_target(knock_x, knock_y, theta_fist, q_head, person_h)
            self._enter(MODE_APPROACH)
            self._t_approach = self.t
            self._wheel_x0 = float(wheel_x)

        # ---------------- 各模式输出 ----------------
        if self.mode == MODE_GAZE_FACE:
            eps = theta_face + q_head
            head, vyaw, tier = self._allocate(eps)
            return V2Action(head, vyaw, 0.0, self.mode, tier,
                            {"eps_body": eps, "target": "face",
                             "off_center_deg": self.off_center_deg(eps, head)})

        if self.mode == MODE_GAZE_FIST:
            eps = theta_fist + q_head
            head, vyaw, tier = self._allocate(eps)
            return V2Action(head, vyaw, 0.0, self.mode, tier,
                            {"eps_body": eps, "target": "fist",
                             "wrist_y_norm": wrist_y_norm,
                             "off_center_deg": self.off_center_deg(eps, head)})

        if self.mode == MODE_APPROACH:
            return self._approach(dt, theta_fist, q_head, wheel_x, person_h,
                                  face_scale, hand_valid)

        if self.mode == MODE_DWELL:
            eps = theta_face + q_head
            head, vyaw, tier = self._allocate(eps)
            return V2Action(head, vyaw, 0.0, self.mode, tier,
                            {"eps_body": eps, "target": "face", "dwell": True,
                             "off_center_deg": self.off_center_deg(eps, head)})

        # IDLE
        return V2Action(0.0, 0.0, 0.0, MODE_IDLE, 0, {"reason": "no person"})

    # ------------------------------------------------------------------
    def _lock_target(self, x_px, y_px, theta_fist, q_head, person_h):
        """敲击那一帧锁定目标。**此后不依赖拳头可见性**。"""
        d = self._person_dist(person_h)
        bearing = float(theta_fist) + float(q_head)
        self._arrived = False
        self.target = TargetLock(
            valid=1, t_lock=self.t, x_px=float(x_px), y_px=float(y_px),
            bearing_body=bearing, dist_m=d, remaining_m=d,
            last_seen_t=self.t, updates=0)

    def _approach(self, dt, theta_fist, q_head, wheel_x, person_h, face_scale,
                  hand_valid):
        p = self.p
        tgt = self.target

        # 手还在视野里就**用观测修正方位**（"盯着那个位置走过去"）；
        # 手没了就保持最后的值，靠里程计走完剩下的距离（"哪怕拳头离开"）。
        if hand_valid:
            tgt.bearing_body = float(theta_fist) + float(q_head)
            tgt.last_seen_t = self.t
            tgt.updates += 1

        # 里程计扣减剩余距离（盲走阶段全靠它）。
        # 目标距离先扣掉过冲保险：宁可少走，绝不多走。
        drive_m = max(0.0, tgt.dist_m - p.overshoot_margin_m)
        if self._wheel_x0 is not None:
            travelled = abs(float(wheel_x) - self._wheel_x0)
            remaining = max(0.0, drive_m - travelled)
        else:
            remaining = tgt.remaining_m
        tgt.remaining_m = remaining

        # 一旦到达就**锁死**：此后任何情况下都不再前进。
        # 没有这个锁，目标方位被观测修正时可能让 vx 重新变正，形成推拉振荡。
        if self._arrived:
            self._enter(MODE_DWELL)
            return V2Action(0.0, 0.0, 0.0, MODE_DWELL, 0,
                            {"phase": "latched", "target": tgt.as_dict()})

        # 安全：人贴太近就不走了
        if face_scale >= p.stop_if_face_scale:
            self._arrived = True
            self._enter(MODE_DWELL)
            return V2Action(0.0, 0.0, 0.0, MODE_DWELL, 0,
                            {"reason": "person_too_close",
                             "target": tgt.as_dict()})

        # 先对准再走：方位误差大于容差时原地转
        eps = tgt.bearing_body
        if abs(eps) > p.turn_tol_deg:
            vyaw = _clamp(p.k_turn * eps, -p.vyaw_max_dps, p.vyaw_max_dps)
            head, _vy2, tier = self._allocate(eps)
            return V2Action(head, vyaw, 0.0, MODE_APPROACH, tier,
                            {"phase": "turn", "eps_body": eps,
                             "off_center_deg": self.off_center_deg(eps, head),
                             "target": tgt.as_dict()})

        # 对准了再前进；速度随剩余距离收敛，避免冲过头
        vx = _clamp(p.k_v * remaining, 0.0, p.vx_max)
        head, vyaw, tier = self._allocate(eps)
        arrived = remaining <= p.arrive_tol_m
        timeout = (self.t - self._t_approach) > p.approach_timeout_s
        if arrived or timeout:
            self._arrived = True
            self._enter(MODE_DWELL)
            return V2Action(0.0, 0.0, 0.0, MODE_DWELL, tier,
                            {"phase": "arrived" if arrived else "timeout",
                             "remaining_m": remaining,
                             "target": tgt.as_dict()})
        return V2Action(head, vyaw, vx, MODE_APPROACH, tier,
                        {"phase": "drive", "remaining_m": remaining,
                         "eps_body": eps,
                         "off_center_deg": self.off_center_deg(eps, head),
                         "target": tgt.as_dict()})
