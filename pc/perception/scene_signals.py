# -*- coding: utf-8 -*-
"""scene_signals.py -- 场景 v2 需要的人脸 / 拳头 / 敲击信号。

`SCENARIO-v2.md` 第一阶段：**不新增任何模型**。场景要的三条主信号
（人脸方位、拳头方位、敲击位置）全部可以从已有的 COCO-17 关键点算出来，
"握没握拳"用已有的手部外观分类器（`gesture_id == 5`）。

COCO-17 索引::

    0 鼻  1 左眼  2 右眼  3 左耳  4 右耳
    5 左肩 6 右肩 7 左肘 8 右肘
    9 左腕 10 右腕
    11 左髋 12 右髋 13 左膝 14 右膝 15 左踝 16 右踝

三条信号的设计要点：

* ``theta_face`` 用鼻+双眼的加权中心，**不用检测框中心**——框中心随人体姿态
  漂移，而"盯着人脸看"要的是脸的方位。
* ``face_scale`` 用瞳距（像素）归一化到画面高度。它与人脸大小的关系是线性的，
  直接就是距离的代理量，符合"人脸在一定大小内才盯"这个门限需求。
* ``wrist_y_norm`` 把腕部的竖直位置按**肩到髋的体长**归一化，于是它与距离无关——
  否则同一个敲击动作在 1 米和 3 米处的像素幅度差三倍，阈值没法统一。
  这与 ``wave_amp`` 按肩宽归一化是同一个思路。

敲击判定放在 :class:`KnockDetector`：腕部归一化高度的一次"下探—回弹"算一次敲击，
两次敲击落在 ``window_s`` 内且间隔不小于 ``min_gap_s`` 才触发。
触发后进入不应期，避免一次动作被判成多次。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

__all__ = [
    "COCO", "image_angle", "face_signals", "hand_signals", "dominant_hand",
    "KnockParams", "KnockOut", "KnockDetector",
]

#: COCO-17 关键点索引
COCO = {
    "nose": 0, "eye_l": 1, "eye_r": 2, "ear_l": 3, "ear_r": 4,
    "sh_l": 5, "sh_r": 6, "el_l": 7, "el_r": 8,
    "wr_l": 9, "wr_r": 10,
    "hip_l": 11, "hip_r": 12, "kn_l": 13, "kn_r": 14, "an_l": 15, "an_r": 16,
}

FOV_H_DEFAULT = 60.0


def image_angle(cx_norm, fov_h: float = FOV_H_DEFAULT) -> float:
    """归一化水平位置 → 相对视线中心的水平角（度，正 = 右）。

    与 ``run_perception.image_angle`` 同一个定义；这里重复一份是为了让本模块
    能脱离 torch / ultralytics 单独导入和单测。
    """
    return (float(cx_norm) - 0.5) * float(fov_h)


def _pt(kxy, kconf, idx: int, min_conf: float):
    """取一个关键点；置信度不足或坐标非有限值返回 None。"""
    if kxy is None or kconf is None:
        return None
    kxy = np.asarray(kxy, dtype=np.float32).reshape(-1, 2)
    kconf = np.asarray(kconf, dtype=np.float32).reshape(-1)
    if idx >= kxy.shape[0] or idx >= kconf.shape[0]:
        return None
    x, y = float(kxy[idx, 0]), float(kxy[idx, 1])
    c = float(kconf[idx])
    if c < min_conf or not (np.isfinite(x) and np.isfinite(y)):
        return None
    return x, y, c


def _mid(a, b):
    if a is None or b is None:
        return None
    return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))


def _dist(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def face_signals(kxy, kconf, img_w: float, img_h: float,
                 fov_h: float = FOV_H_DEFAULT, min_conf: float = 0.30) -> dict:
    """人脸方位与尺度。

    Returns:
        ``theta_face``  度，正 = 右；人脸不可用时为 0.0
        ``face_scale``  瞳距 / 画面高度，∈[0,1]；不可用时 0.0
        ``face_px``     瞳距像素数（诊断用）
        ``face_conf``   所用关键点里最低的那个置信度
        ``face_valid``  0/1
    """
    nose = _pt(kxy, kconf, COCO["nose"], min_conf)
    eye_l = _pt(kxy, kconf, COCO["eye_l"], min_conf)
    eye_r = _pt(kxy, kconf, COCO["eye_r"], min_conf)

    used = [p for p in (nose, eye_l, eye_r) if p is not None]
    if not used:
        return {"theta_face": 0.0, "face_scale": 0.0, "face_px": 0.0,
                "face_conf": 0.0, "face_valid": 0}

    # 置信度加权中心：眼睛比鼻子稳（鼻子在侧脸时漂）
    wsum = sum(p[2] for p in used)
    cx = sum(p[0] * p[2] for p in used) / max(wsum, 1e-6)
    theta = image_angle(np.clip(cx / max(img_w, 1.0), 0.0, 1.0), fov_h)

    face_px = _dist(eye_l, eye_r)
    scale = float(np.clip(face_px / max(img_h, 1.0), 0.0, 1.0))
    return {
        "theta_face": float(theta),
        "face_scale": scale,
        "face_px": float(face_px),
        "face_conf": float(min(p[2] for p in used)),
        "face_valid": 1,
    }


def _body_scale(kxy, kconf, min_conf: float, img_h: float):
    """肩—髋距离（像素），作为与人体尺度成正比的量；退化时用肩宽/框高。"""
    sh_l = _pt(kxy, kconf, COCO["sh_l"], min_conf)
    sh_r = _pt(kxy, kconf, COCO["sh_r"], min_conf)
    hip_l = _pt(kxy, kconf, COCO["hip_l"], min_conf)
    hip_r = _pt(kxy, kconf, COCO["hip_r"], min_conf)
    sh_mid, hip_mid = _mid(sh_l, sh_r), _mid(hip_l, hip_r)
    d = _dist(sh_mid, hip_mid)
    if d > 1e-3:
        return d, "shoulder-hip"
    d = _dist(sh_l, sh_r)
    if d > 1e-3:
        return d, "shoulder-width"
    return max(img_h * 0.25, 1e-3), "fallback"


def hand_signals(kxy, kconf, img_w: float, img_h: float,
                 fov_h: float = FOV_H_DEFAULT, min_conf: float = 0.30) -> dict:
    """双腕的方位与归一化竖直位置。

    归一化竖直位置定义::

        wrist_y_norm = (wrist_y - shoulder_mid_y) / body_scale

    肩中点以下为正（屏幕坐标向下为正）。与距离无关，因此同一个敲击动作
    在 1 米和 3 米处得到同一个数值。

    Returns:
        ``hands``: ``{"L": {...}, "R": {...}}``，每只手含
        ``theta_fist`` / ``wrist_y_norm`` / ``wrist_x`` / ``wrist_y`` / ``conf`` / ``valid``
        ``body_scale_px`` / ``body_scale_src``
    """
    sh_l = _pt(kxy, kconf, COCO["sh_l"], min_conf)
    sh_r = _pt(kxy, kconf, COCO["sh_r"], min_conf)
    sh_mid = _mid(sh_l, sh_r)
    scale, src = _body_scale(kxy, kconf, min_conf, img_h)

    out = {
        "hands": {},
        "body_scale_px": float(scale),
        "body_scale_src": src,
    }
    for side, key in (("L", "wr_l"), ("R", "wr_r")):
        wr = _pt(kxy, kconf, COCO[key], min_conf)
        if wr is None or sh_mid is None:
            out["hands"][side] = {
                "theta_fist": 0.0, "wrist_y_norm": 0.0,
                "wrist_x": 0.0, "wrist_y": 0.0, "conf": 0.0, "valid": 0,
            }
            continue
        x, y, c = wr
        out["hands"][side] = {
            "theta_fist": float(image_angle(
                np.clip(x / max(img_w, 1.0), 0.0, 1.0), fov_h)),
            "wrist_y_norm": float((y - sh_mid[1]) / max(scale, 1e-6)),
            "wrist_x": float(x),
            "wrist_y": float(y),
            "conf": float(c),
            "valid": 1,
        }
    return out


def dominant_hand(hands: dict, fist_probs: dict | None = None) -> tuple[str, dict]:
    """选当前该盯的那只手。

    优先级：**被判成 fist 且置信度高的手** > 关键点置信度高的一只 > 左手。
    这里只做"选哪只手"，"是不是拳头"由手势分类器给。
    """
    probs = fist_probs or {}
    best_side, best_score, best = "L", -1.0, None
    for side in ("L", "R"):
        h = hands.get(side) or {}
        if not h.get("valid"):
            continue
        score = float(probs.get(side, 0.0)) * 2.0 + float(h.get("conf", 0.0))
        if score > best_score:
            best_side, best_score, best = side, score, h
    if best is None:
        return "L", {"valid": 0, "theta_fist": 0.0, "wrist_y_norm": 0.0,
                     "wrist_x": 0.0, "wrist_y": 0.0, "conf": 0.0}
    return best_side, best


@dataclass
class KnockParams:
    """敲击检测参数。

    默认值来自场景先验（拳头在桌面上抬落），**尚未在真机标定**；
    标定时改这里或换成配置加载，不要在下游写死。
    """
    #: 腕部归一化高度超过这个值才算"手放到了桌面高度"
    y_norm_min: float = 0.55
    #: 一次下探的有效深度（归一化单位）
    depth: float = 0.18
    #: 回弹幅度：峰值落下这么多才算一次完整敲击
    rebound: float = 0.10
    #: 两次敲击的最大间隔（秒）
    window_s: float = 1.50
    #: 两次敲击的最小间隔（秒），防止一次抖动算两次
    min_gap_s: float = 0.12
    #: 触发后的不应期（秒）
    refractory_s: float = 1.20
    #: 相邻帧最大时间差；超过就重置峰谷状态（断流保护）
    max_dt_s: float = 0.35


@dataclass
class KnockOut:
    """一帧的敲击状态。"""
    fired: int = 0
    count: int = 0
    #: 触发时的敲击点（图像像素坐标），供目标锁定使用
    x_px: float = 0.0
    y_px: float = 0.0
    #: 累计完成的有效敲击次数（每帧递增，便于调试）
    hits: int = 0
    phase: str = "idle"

    def as_dict(self) -> dict:
        return asdict(self)


class KnockDetector:
    """腕部归一化高度的"下探—回弹"双峰检测器。

    状态机::

        idle --下探超过 depth--> down --回弹 rebound--> idle（记一次敲击）
        两次敲击在 window_s 内且间隔 >= min_gap_s --> fired

    只在"手确实在桌面高度"（``y_norm > y_norm_min``）时才计数，避免
    手在胸前摆动被误判成敲击。
    """

    def __init__(self, params: KnockParams | None = None):
        self.p = params or KnockParams()
        self.reset()

    def reset(self):
        self._t_prev: float | None = None
        self._phase = "idle"
        self._peak_y = -1e9
        self._t_peak = 0.0
        self._last_hit_t = -1e9
        self._first_hit_t: float | None = None
        self._n_in_window = 0
        self._t_last_fire = -1e9
        self.hits = 0

    def update(self, t: float, wrist_y_norm: float, wrist_x: float,
               wrist_y: float, valid: int = 1) -> KnockOut:
        """喂一帧。

        Args:
            t: 秒，单调递增
            wrist_y_norm: :func:`hand_signals` 给的归一化竖直位置
            wrist_x / wrist_y: 像素坐标，触发时原样上报作为敲击点
            valid: 该帧腕部是否可用

        Returns:
            KnockOut
        """
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        if self._t_prev is not None and dt > self.p.max_dt_s:
            # 断流：峰谷状态作废，但保留已经攒下的敲击次数
            self._phase = "idle"
            self._peak_y = -1e9
        self._t_prev = t

        out = KnockOut(hits=self.hits)
        if not valid:
            self._phase = "idle"
            self._peak_y = -1e9
            out.phase = "no_hand"
            return out

        y = float(wrist_y_norm)

        if self._phase == "idle":
            if y >= self.p.y_norm_min:
                self._phase = "down"
                self._peak_y = y
                self._t_peak = t
        elif self._phase == "down":
            if y > self._peak_y:
                self._peak_y = y
                self._t_peak = t
            # 回弹：从峰值回落足够多 = 一次敲击完成
            if self._peak_y - y >= self.p.depth * 0.5 + self.p.rebound:
                self._register_hit(self._t_peak, wrist_x, wrist_y)
                self._phase = "idle"
                self._peak_y = -1e9

        out.hits = self.hits
        out.phase = self._phase

        # 两次敲击都在窗口内才触发
        if (self._first_hit_t is not None
                and (t - self._first_hit_t) <= self.p.window_s
                and self._n_in_window >= 2
                and (t - self._t_last_fire) >= self.p.refractory_s):
            out.fired = 1
            out.count = self._n_in_window
            self._t_last_fire = t
            out.x_px, out.y_px = self._last_hit_xy
            # 触发后清窗口，避免同一动作重复触发
            self._first_hit_t = None
            self._n_in_window = 0
        return out

    def _register_hit(self, t_hit: float, x: float, y: float):
        if (t_hit - self._last_hit_t) >= self.p.min_gap_s:
            self.hits += 1
            self._last_hit_t = t_hit
            self._last_hit_xy = (float(x), float(y))
            if self._first_hit_t is None:
                self._first_hit_t = t_hit
            self._n_in_window += 1
            # 窗口过期就重开
            if (t_hit - self._first_hit_t) > self.p.window_s:
                self._first_hit_t = t_hit
                self._n_in_window = 1
