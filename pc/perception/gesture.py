# -*- coding: utf-8 -*-
"""gesture.py — 从 COCO-17 关键点算手势基元（工作流 A）**已废弃**

.. deprecated::
    手势判定已经换成 ``pc/gesture`` 的「分类器 + 事件机」，入口在
    :meth:`run_perception.PerceptionPipeline`。本模块**只保留向后兼容**：

    * ``hand_up_L / hand_up_R``：教师控制器的 ``require_hand_up`` 还在用，有效；
    * ``wave_sin / wave_cos / wave_amp``：事件化后没有语义，只为旧字段而留；
    * ``wave_score`` 与 ``waving``：**不要再拿它做触发判定**
      （``wave_score > 4`` 在 1 Hz 慢挥上稳态只有 2.17，实测 90% 漏检，
      见 ``pc/gesture/reports/events.md``）。

    新代码请用::

        from gesture_classifier import GestureClassifier
        from gesture_events import GestureEventMachine

    事件机输出的 ``gesture_onset`` 是边沿，一段连续动作只有一个，
    所以下游不需要冷却期。字段语义见 ``GESTURE_INTEGRATION.md``。

--- 以下是废弃实现的原始说明（保留，因为测试还在测它）---------------

契约 §3.2 / TECH-PLAN §4.1(a) 的落地实现：

    shoulder_width = |kp[5].x - kp[6].x|              # 左肩、右肩 x 差（像素）
    x_rel = (wrist.x - shoulder_mid.x) / shoulder_width
    y_rel = (wrist.y - shoulder_mid.y) / shoulder_width
    hand_up_L = y_rel_L < 0 且关键点置信度够
    wave_score *= exp(-dt / 2.0)                      # 每帧衰减
    手腕横向过零 → wave_score += 1                    # 累积，不用瞬时阈值
    wave_score > 4 → 触发（判定权在教师控制器，这里只给累积量）

**为什么必须按肩宽归一化**：人走近，所有像素尺寸一起变大；用像素阈值判定
「手举起来没有 / 摆幅够不够」，距离一变全错。除以肩宽后量纲消失，同一个人在
0.5m 和 2m 给出同一个数（测试里有两个尺度的等价性断言）。

两处**对契约字面做法的偏离**，都是为了让它在真人身上成立（已在回报里说明）：

1. **过零是相对「慢速中心」的过零，不是相对 x_rel=0 的过零。**
   契约写的是「手腕横向位置过零」。真人挥手时手在体侧，``x_rel`` 在
   [0.6, 1.4] 之间摆，**永远不穿过 0**，字面实现会一次都不计数。这里用
   ``x_rel - EMA(x_rel, τ=1.5s)`` 去趋势后再判过零：中心真的在 0 附近时，
   两者完全等价；中心偏在一边时，前者仍然工作。
2. **过零要求「上一个半周摆出过超过 ``wave_amp_thresh``（默认 0.05 肩宽）的幅值」。**
   注意判据不能用上一帧的值：过零那一帧的 ``|d|`` 本来就在 0 附近，用它永远判不出
   过零（第一版就是这么写的，3 秒挥手 0 分）。所以记的是半周期内的峰值 ``_peak``。
   不加幅值门槛则相反：静止时关键点抖动会在中心附近疯狂翻转，几秒就刷到触发。

肩宽 < 5 像素时按契约「算不出来就别瞎算」：``hand_up_*=0``、不更新挥手状态，
只保留衰减。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["GesturePrimitives", "GestureEstimator", "COCO17",
           "WAVE_TRIGGER_SCORE", "SHOULDER_MIN_PX"]

#: COCO-17 关键点索引（只用上肢 + 肩）
COCO17 = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

#: 手腕横向过零累积到这个分数以上算「在挥手」（契约 §3.2：> 4）。
#: **已废弃**：这个阈值就是被删掉的旧判定，只为 ``GesturePrimitives.waving`` 保留。
WAVE_TRIGGER_SCORE = 4.0

#: 肩宽小于这个像素数就放弃判定（契约 §3.2）
SHOULDER_MIN_PX = 5.0

_L_SH, _R_SH = COCO17["left_shoulder"], COCO17["right_shoulder"]
_L_WR, _R_WR = COCO17["left_wrist"], COCO17["right_wrist"]


@dataclass
class GesturePrimitives:
    """一帧的手势基元。字段与 ``PerceptionOut`` / ``TeacherObs`` 对得上。

    **已废弃**：除 ``hand_up_L/R`` 外，``wave_*`` 与 ``gesture_conf`` 都不再参与
    手势判定（判定看 :attr:`PerceptionOut.gesture_onset`）。
    """

    hand_up_L: int = 0
    hand_up_R: int = 0
    wave_score: float = 0.0
    wave_sin: float = 0.0
    wave_cos: float = 1.0
    wave_amp: float = 0.0
    gesture_conf: float = 0.0
    debug: dict = field(default_factory=dict)

    @property
    def hand_up_any(self):
        return int(bool(self.hand_up_L) or bool(self.hand_up_R))

    @property
    def waving(self):
        """**已废弃**：累积量是否过阈值。事件化后触发看 ``gesture_onset`` 边沿。"""
        return self.wave_score > WAVE_TRIGGER_SCORE


class GestureEstimator:
    """有状态的挥手/举手估计器：逐帧喂关键点，返回基元。

    **已废弃（除 ``hand_up_L/R``）**：``wave_score`` 的过零累积不再用于触发，
    手势判定改用 ``pc/gesture`` 的分类器 + 事件机。保留这个类是因为
    ``PerceptionOut`` 的 ``hand_up_*`` / ``wave_*`` 旧字段还由它产出。

    用法：

        >>> est = GestureEstimator()
        >>> g = est.update(kp_xy, kp_conf, dt=1 / 30.0)   # kp_xy (17,2) 像素
        >>> g.hand_up_L, g.wave_score
        (0, 0.0)

    Args:
        hand_up_conf: 手腕/肩关键点的最低置信度（TECH-PLAN §4.1 取 0.3）。
        shoulder_min_px: 肩宽下限（契约 §3.2：< 5 像素直接判 0）。
        wave_amp_thresh: 过零所需的最小归一化幅值（相对肩宽）。
        score_decay_tau: ``wave_score`` 的衰减时间常数（秒），契约取 2.0。
        center_tau: 慢速中心 EMA 时间常数（秒），必须明显大于挥手周期。
        amp_tau: 幅值 EMA 时间常数（秒）。
        freq_min/freq_max: 频率估计的钳位范围（Hz）。
        trigger_score: 触发阈值，只用于 ``gesture_conf`` 与 ``waving`` 属性。
    """

    def __init__(self, hand_up_conf=0.3, shoulder_min_px=SHOULDER_MIN_PX,
                 wave_amp_thresh=0.05, score_decay_tau=2.0, center_tau=1.5,
                 amp_tau=0.4, freq_min=0.4, freq_max=6.0,
                 trigger_score=WAVE_TRIGGER_SCORE):
        self.hand_up_conf = float(hand_up_conf)
        self.shoulder_min_px = float(shoulder_min_px)
        self.wave_amp_thresh = float(wave_amp_thresh)
        self.score_decay_tau = float(score_decay_tau)
        self.center_tau = float(center_tau)
        self.amp_tau = float(amp_tau)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.trigger_score = float(trigger_score)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """清空全部时序状态（换人、重连、长时间丢失后调用）。"""
        self.wave_score = 0.0
        self.wave_amp = 0.0
        self.phase = 0.0
        self.freq = 0.0
        self.time = 0.0
        self.crossings = 0
        self._active = None          # "L" / "R" / None
        self._center = None          # x_rel 的慢速中心
        self._prev_sign = 0
        self._peak = 0.0             # 当前半周期内 |d| 的峰值
        self._last_cross_t = {1: None, -1: None}
        self._since_cross = 0.0

    # ------------------------------------------------------------------
    def update(self, kp_xy, kp_conf, dt=0.0):
        """喂一帧关键点，返回 ``GesturePrimitives``。

        Args:
            kp_xy: (17, 2) 关键点像素坐标（原图坐标系）；无目标时传 ``None``。
            kp_conf: (17,) 每个关键点的置信度；``None`` 视为全 1。
            dt: 距上一帧的秒数（首帧传 0）。
        """
        dt = float(dt) if dt is not None else 0.0
        if not math.isfinite(dt) or dt < 0.0:
            dt = 0.0
        dt = min(dt, 1.0)
        self.time += dt

        # 1) 先衰减（契约顺序：先衰减后累加）
        self.wave_score *= math.exp(-dt / self.score_decay_tau)
        self._since_cross += dt
        if self._since_cross > 1.0:      # 1 秒没有过零，频率慢慢归零
            self.freq *= math.exp(-dt / 0.5)
        # 2) 相位积分（sin/cos 由频率 f 累积得到，契约 §3.2）
        if self.freq > 0.0:
            self.phase = (self.phase + 2.0 * math.pi * self.freq * dt) % (2.0 * math.pi)

        dbg = {
            "dt": dt, "shoulder_width_px": 0.0, "x_rel": 0.0, "y_rel_L": 0.0,
            "y_rel_R": 0.0, "active_hand": None, "crossings": self.crossings,
            "freq_hz": self.freq, "wave_score": self.wave_score,
            "amp_thresh": self.wave_amp_thresh, "reason": "",
        }

        # 3) 无人：保留 wave_score 衰减（短暂丢失不该清空手势记忆），只清过零状态
        if kp_xy is None:
            self._set_active(None)
            dbg["reason"] = "no_person"
            return self._pack(0, 0, dbg)

        kp = np.asarray(kp_xy, dtype=np.float32)
        conf = (np.ones(17, dtype=np.float32) if kp_conf is None
                else np.asarray(kp_conf, dtype=np.float32).reshape(-1))
        if kp.shape != (17, 2) or conf.shape != (17,):
            raise ValueError("关键点形状应为 (17,2)/(17,)，收到 %r/%r"
                             % (kp.shape, conf.shape))

        # 4) 肩宽归一化（契约 §3.2）
        sw = float(abs(kp[_L_SH, 0] - kp[_R_SH, 0]))
        dbg["shoulder_width_px"] = sw
        if sw < self.shoulder_min_px:
            self._set_active(None)
            dbg["reason"] = "shoulder_too_small"
            return self._pack(0, 0, dbg)

        mid = 0.5 * (kp[_L_SH] + kp[_R_SH])
        rel = (kp - mid) / sw                       # (17,2)，量纲消除
        x_rel_L, y_rel_L = float(rel[_L_WR, 0]), float(rel[_L_WR, 1])
        x_rel_R, y_rel_R = float(rel[_R_WR, 0]), float(rel[_R_WR, 1])

        # 5) 举手：手腕在肩线以上（图像 y 向下，所以 y_rel<0）+ 置信度
        up_L = int(y_rel_L < 0.0 and
                   min(conf[_L_WR], conf[_L_SH]) >= self.hand_up_conf)
        up_R = int(y_rel_R < 0.0 and
                   min(conf[_R_WR], conf[_R_SH]) >= self.hand_up_conf)

        # 6) 选「正在挥的那只手」：举着的手里横向偏离更大的那只
        if up_L or up_R:
            if up_L and up_R:
                hand, x_rel, c = ("L", x_rel_L, conf[_L_WR]) if abs(x_rel_L) >= abs(x_rel_R) \
                    else ("R", x_rel_R, conf[_R_WR])
            elif up_L:
                hand, x_rel, c = "L", x_rel_L, conf[_L_WR]
            else:
                hand, x_rel, c = "R", x_rel_R, conf[_R_WR]
        else:
            hand, x_rel, c = None, 0.0, 0.0
        self._set_active(hand)

        dbg.update(x_rel=x_rel, y_rel_L=y_rel_L, y_rel_R=y_rel_R,
                   active_hand=hand)

        if hand is None:
            # 手放下了：幅值衰减到 0，但 wave_score 继续按时间衰减（有记忆）
            self.wave_amp *= math.exp(-dt / self.amp_tau)
            dbg["reason"] = "no_hand_up"
            return self._pack(up_L, up_R, dbg)

        # 7) 去趋势 + 过零计数
        if self._center is None:
            self._center = x_rel
        elif dt > 0.0:
            a = 1.0 - math.exp(-dt / self.center_tau)
            self._center += a * (x_rel - self._center)
        d = x_rel - self._center
        self.wave_amp += (1.0 - math.exp(-dt / self.amp_tau)) * (abs(d) - self.wave_amp)

        # 过零判定：符号翻转算一次，但要求「上一侧确实摆出过幅度」
        # （用半周期内的峰值 _peak，不能用上一帧的值——过零那一帧的
        #   |d| 本来就接近 0，用它会永远判不出过零）
        s = 1 if d > 0.0 else -1
        if self._prev_sign == 0:
            self._prev_sign = s
            self._peak = abs(d)
        elif s != self._prev_sign:
            if self._peak > self.wave_amp_thresh:
                self._on_crossing(s, dt, dbg)
            self._prev_sign = s
            self._peak = abs(d)
        else:
            self._peak = max(self._peak, abs(d))
        dbg["d_rel"] = d
        dbg["peak_rel"] = self._peak

        # 8) 手势置信度：举手 × 累积量饱和 × 关键点置信度
        gconf = (min(1.0, self.wave_score / max(self.trigger_score, 1e-6))
                 * float(c) * (1.0 if (up_L or up_R) else 0.0))
        return self._pack(up_L, up_R, dbg, gesture_conf=gconf)

    # ------------------------------------------------------------------
    def _on_crossing(self, sign, dt, dbg):
        """处理一次有效过零：加 1 分、更新频率估计、推进半个相位。"""
        self.crossings += 1
        self.wave_score += 1.0
        self._since_cross = 0.0
        prev_t = self._last_cross_t.get(sign)
        if prev_t is not None and 0.08 <= (self.time - prev_t) <= 4.0:
            f_inst = 1.0 / (self.time - prev_t)       # 同向过零 = 一个完整周期
            self.freq = min(self.freq_max,
                            max(self.freq_min, 0.6 * f_inst + 0.4 * self.freq))
        self._last_cross_t[sign] = self.time
        dbg["crossings"] = self.crossings

    def _set_active(self, hand):
        """切换活动手时清掉过零状态（两只手的中心不是同一个）。"""
        if hand == self._active:
            return
        self._active = hand
        self._center = None
        self._prev_sign = 0
        self._peak = 0.0
        self._last_cross_t = {1: None, -1: None}

    def _pack(self, up_L, up_R, dbg, gesture_conf=0.0):
        return GesturePrimitives(
            hand_up_L=int(up_L), hand_up_R=int(up_R),
            wave_score=float(self.wave_score),
            wave_sin=float(math.sin(self.phase)),
            wave_cos=float(math.cos(self.phase)),
            wave_amp=max(0.0, float(self.wave_amp)),
            gesture_conf=float(np.clip(gesture_conf, 0.0, 1.0)),
            debug=dbg,
        )
