# -*- coding: utf-8 -*-
"""gesture_events.py — 把连续分类输出变成**结构化事件**（感知端事件化）

## 为什么要这一层

手势是**连续动作**，网络输出是**逐帧概率**。如果把概率当电平用，下游就得自己
从电平里推断「一次动作从哪儿开始、到哪儿算结束」。而 ``ncps`` 的 CfC 没有 h
自环项，实测记忆时标只有 ~2.2 s（见 ``D:\\smartcar\\code\\lnn\\reports\\cfc_memory_20261030.md``），
让网络做长时程的事件边界推断，它结构上就做不好。所以边界推断放在这里，
用确定性的状态机做，下游只学「事件 → 动作」。

## 状态机

::

    IDLE ──候选概率≥enter──▶ PENDING ──持续 pending_hold──▶ GESTURE ──低于 exit release──▶ RETURN
     ▲                          │                                                        │
     └─────────────── 候选消失 ─┘◀──────────── idle 主导 return_clear ────────────────────┘

* **IDLE**：没有候选动作。
* **PENDING**：正在积累证据，还没确认。这一段时间是「确认窗口」，不是冷却。
* **GESTURE**：已确认。``gesture_onset`` 只在**进入这一帧**是 1，之后一直是 0
  ——这是**边沿**，不是电平。下游看到 1 就是「这里发生了一次手势」。
* **RETURN**：动作结束、手在放下。这段时间**不再产生 onset**，好让「一次连续
  动作」和「两次分开的动作」区分开：连续挥手时手不会真的放下来，所以进不了
  RETURN；两次挥手之间手会落下，RETURN 会清空，下一次才有新的边沿。
  如果 RETURN 里同一动作又升回 ``enter`` 阈值之上（概率中途掉了一下、手并没
  放下），事件**恢复**而不是发新 onset —— 这是「一段举手不动不能算两次」的
  实现方式。

## 三个时间参数为什么不是冷却

* ``pending_hold``：需要「同一类连续成立」多久才确认——去抖。
* ``release``：概率掉下去多久才算动作结束——防抖，也是「同一动作别拆成两个」的闸门。
* ``min_gap``：两个 onset 的最小间隔——**只**用来滤掉两帧连着的重复边沿，
  取值 200~300 ms，量级与人手再做一个动作的时间（>400 ms）差一个数量级，
  所以它没有能力「忘记」任何东西。
* ``min_gesture``：单个事件的最短持续。连续挥手时概率可能瞬间掉一下，
  没有这条，一段挥手会被拆成两个 onset。

## 输出（每帧，字段名与验收脚本一致）

===============  =========================================================
gesture_id       0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist
gesture_onset    边沿脉冲：进入 GESTURE 那一帧 1，其余 0
gesture_phase    0=IDLE 1=PENDING 2=GESTURE 3=RETURN
gesture_conf     0..1，当前候选/已确认类的平滑概率
gesture_gap_s    距上次 GESTURE 结束的秒数（连续量，上限 gap_cap_s）
===============  =========================================================

所有参数都在 ``configs/gesture_events.yaml``，代码里不留魔法数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:                                     # PyYAML 是硬依赖，但别让 import 顺序坑人
    import yaml
except Exception:                        # pragma: no cover
    yaml = None

__all__ = [
    "GestureEventOut", "GestureEventMachine", "PHASE_IDLE", "PHASE_PENDING",
    "PHASE_GESTURE", "PHASE_RETURN", "DEFAULT_CONFIG", "load_config",
]

PHASE_IDLE = 0
PHASE_PENDING = 1
PHASE_GESTURE = 2
PHASE_RETURN = 3

PHASE_NAMES = {PHASE_IDLE: "IDLE", PHASE_PENDING: "PENDING",
               PHASE_GESTURE: "GESTURE", PHASE_RETURN: "RETURN"}

DEFAULT_CLASSES = ("idle", "wave", "open_palm", "thumbs_up", "pointing", "fist")

DEFAULT_CONFIG = {
    "version": 1,
    "classes": list(DEFAULT_CLASSES),
    "prob_smooth_tau_s": 0.10,
    "enter_thresh": 0.55,
    "exit_thresh": 0.32,
    "swap_margin": 0.15,
    "idle_dominant": 0.42,
    "per_class": {},
    "timing_s": {
        "pending_hold": 0.18,
        "release": 0.25,
        "min_gesture": 0.45,
        "return_clear": 0.35,
        "min_gap": 0.25,
        "return_timeout": 2.00,
        "max_dt_s": 0.25,
    },
    "gap_cap_s": 3600.0,
}


def _deep_update(base, over):
    """把 ``over`` 递归盖到 ``base`` 上（不修改入参）。"""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg=None):
    """读配置：``dict`` 直接用，``str`` 当 YAML 路径，``None`` 用默认值。

    Args:
        cfg: dict / 路径 / None。

    Returns:
        dict，已与 :data:`DEFAULT_CONFIG` 合并。
    """
    if cfg is None:
        return dict(DEFAULT_CONFIG)
    if isinstance(cfg, str):
        if yaml is None:                 # pragma: no cover
            raise RuntimeError("需要 PyYAML 才能读 YAML 配置")
        with open(cfg, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return _deep_update(DEFAULT_CONFIG, data)
    if isinstance(cfg, dict):
        return _deep_update(DEFAULT_CONFIG, cfg)
    raise TypeError("配置应为 dict/路径/None，收到 %r" % type(cfg))


@dataclass
class GestureEventOut:
    """一帧的事件化输出（字段名与验收脚本、下游契约一致）。"""

    gesture_id: int = 0
    gesture_onset: int = 0
    gesture_phase: int = PHASE_IDLE
    gesture_conf: float = 0.0
    gesture_gap_s: float = 0.0
    debug: dict = field(default_factory=dict)

    def as_dict(self):
        """展开成普通 dict（写 CSV / 喂下游用）。"""
        return {"gesture_id": self.gesture_id,
                "gesture_onset": self.gesture_onset,
                "gesture_phase": self.gesture_phase,
                "gesture_conf": self.gesture_conf,
                "gesture_gap_s": self.gesture_gap_s}

    @property
    def phase_name(self):
        return PHASE_NAMES[self.gesture_phase]


class GestureEventMachine:
    """逐帧喂概率，吐事件。有状态、可 reset。

    用法::

        >>> m = GestureEventMachine()          # 用默认参数
        >>> out = m.update([0.9, 0.02, 0.02, 0.02, 0.02, 0.02], dt=1/30)
        >>> out.phase_name
        'IDLE'

    Args:
        config: dict 或 YAML 路径或 None（默认参数）。
        classes: 覆盖类别名列表（必须与概率向量同序）。
    """

    def __init__(self, config=None, classes=None):
        self.cfg = load_config(config)
        self.classes = tuple(classes or self.cfg["classes"])
        n = len(self.classes)
        self.n_classes = n
        if n < 2:
            raise ValueError("至少要有 idle 和一个手势类")
        self.idle_id = 0

        pc = self.cfg.get("per_class") or {}
        self.enter = np.full(n, float(self.cfg["enter_thresh"]), np.float64)
        self.exit = np.full(n, float(self.cfg["exit_thresh"]), np.float64)
        for name, spec in pc.items():
            if name not in self.classes:
                raise ValueError("per_class 里的 %r 不在 classes 里" % name)
            i = self.classes.index(name)
            if "enter" in spec:
                self.enter[i] = float(spec["enter"])
            if "exit" in spec:
                self.exit[i] = float(spec["exit"])
        # idle 的 enter/exit 不参与候选判定，但保留值便于调试
        self.enter[self.idle_id] = float(self.cfg["enter_thresh"])
        self.exit[self.idle_id] = float(self.cfg["exit_thresh"])

        t = self.cfg["timing_s"]
        self.tau = float(self.cfg["prob_smooth_tau_s"])
        self.pending_hold = float(t["pending_hold"])
        self.release = float(t["release"])
        self.min_gesture = float(t["min_gesture"])
        self.return_clear = float(t["return_clear"])
        self.min_gap = float(t["min_gap"])
        self.return_timeout = float(t["return_timeout"])
        self.max_dt = float(t["max_dt_s"])
        self.swap_margin = float(self.cfg["swap_margin"])
        self.idle_dominant = float(self.cfg["idle_dominant"])
        self.gap_cap = float(self.cfg["gap_cap_s"])
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        """清空全部时序状态（换人、断流重连后调用）。"""
        self.t = 0.0
        self.p = np.zeros(self.n_classes, dtype=np.float64)
        self._has_p = False
        self.phase = PHASE_IDLE
        self.gid = 0
        self.pend_cls = 0
        self.pend_t = 0.0
        self.t_gesture = -1e9
        self.t_last_end = 0.0
        self.low_run = 0.0
        self.idle_run = 0.0
        self.cand_run = 0.0
        self.ret_t = 0.0
        self.n_onsets = 0
        self.last_onset_t = -1e9
        self.last_gid = 0

    # ------------------------------------------------------------------
    def update(self, probs, dt, t=None):
        """喂一帧分类概率，返回 :class:`GestureEventOut`。

        Args:
            probs: (n_classes,) 概率（顺序同 ``classes``）。允许未归一化，
                但必须是有限数；NaN/Inf 直接抛错（宁可炸也不静默输出垃圾）。
            dt: 距上一帧的秒数；内部会钳到 ``[0, max_dt_s]``。
            t: 可选外部时间戳（秒）。给了就覆盖内部时钟，用于和视频帧对齐。

        Returns:
            GestureEventOut
        """
        p_raw = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p_raw.shape[0] != self.n_classes:
            raise ValueError("probs 长度应为 %d，收到 %d"
                             % (self.n_classes, p_raw.shape[0]))
        if not np.all(np.isfinite(p_raw)):
            raise ValueError("probs 含 NaN/Inf：%r" % (p_raw,))
        p_raw = np.clip(p_raw, 0.0, 1.0)

        dt = float(dt) if dt is not None else 0.0
        if not math.isfinite(dt):
            raise ValueError("dt 必须是有限数，收到 %r" % (dt,))
        dt = min(max(dt, 0.0), self.max_dt)
        if t is not None:
            self.t = float(t)
        else:
            self.t += dt

        # ---- 1) 低通 ----
        if not self._has_p or dt <= 0.0:
            self.p = p_raw.copy()
            self._has_p = True
        else:
            a = 1.0 - math.exp(-dt / max(self.tau, 1e-6))
            self.p += a * (p_raw - self.p)
        p = self.p

        # ---- 2) 候选：非 idle 里最强的那个 ----
        body = p[1:]
        cand = int(np.argmax(body)) + 1 if body.size else self.idle_id
        p_cand = float(p[cand])
        p_idle = float(p[self.idle_id])

        onset = 0
        if self.phase == PHASE_IDLE:
            if p_cand >= self.enter[cand] and p_cand > p_idle:
                self.phase = PHASE_PENDING
                self.pend_cls = cand
                self.pend_t = 0.0
            self.low_run = 0.0

        elif self.phase == PHASE_PENDING:
            k = self.pend_cls
            if p[k] < self.exit[k]:
                # 候选塌了：退回 IDLE，不在弱证据上硬撑
                self.phase = PHASE_IDLE
                self.pend_t = 0.0
                self.low_run = 0.0
            elif cand != k and p_cand > p[k] + self.swap_margin:
                # 换类：重新计时（两类似是而非时不要提前触发）
                self.pend_cls = cand
                k = cand
                self.pend_t = 0.0
            else:
                # 只有**仍然高于进入阈值**的帧才算证据。第一版这里错写成
                # 「不低于退出阈值就累加」，结果候选掉到 0.25、idle 已经 0.71
                # 的时候照样确认，静止段里凭空冒出一个 onset。
                if p[k] >= self.enter[k]:
                    self.pend_t += dt
                else:
                    self.pend_t = max(0.0, self.pend_t - dt)
                if (self.pend_t >= self.pending_hold
                        and self.t - self.t_last_end >= self.min_gap):
                    self.phase = PHASE_GESTURE
                    self.gid = k
                    self.t_gesture = self.t
                    self.last_onset_t = self.t
                    self.n_onsets += 1
                    self.low_run = 0.0
                    onset = 1

        elif self.phase == PHASE_GESTURE:
            k = self.gid
            if p[k] < self.exit[k]:
                self.low_run += dt
                if (self.low_run >= self.release
                        and self.t - self.t_gesture >= self.min_gesture):
                    self.phase = PHASE_RETURN
                    self.last_gid = k
                    self.t_last_end = self.t
                    self.ret_t = 0.0
                    self.idle_run = 0.0
                    self.cand_run = 0.0
                    self.gid = 0
            else:
                self.low_run = 0.0

        elif self.phase == PHASE_RETURN:
            self.ret_t += dt
            if p_idle >= self.idle_dominant and p_cand < self.enter[cand]:
                self.idle_run += dt
                self.cand_run = 0.0
            else:
                self.idle_run = 0.0
                if p_cand >= self.enter[cand]:
                    self.cand_run += dt
                    if self.cand_run >= self.pending_hold:
                        # 手其实没放下（同一段连续动作的延续，只是概率中途掉了一下）：
                        # **恢复**事件，不产生新 onset。没有这一条，一个「举手不动」
                        # 会在概率抖动时被拆成两个事件——恰恰是本设计要避免的事。
                        self.phase = PHASE_GESTURE
                        self.gid = cand
                        self.last_gid = cand
                        self.t_gesture = self.t - self.min_gesture
                        self.low_run = 0.0
                        self.cand_run = 0.0
                else:
                    self.cand_run = 0.0
            if (self.phase == PHASE_RETURN
                    and (self.idle_run >= self.return_clear
                         or self.ret_t >= self.return_timeout)):
                self.phase = PHASE_IDLE

        # ---- 3) 输出 ----
        if self.phase == PHASE_GESTURE:
            gid, conf = self.gid, float(p[self.gid])
        elif self.phase == PHASE_PENDING:
            gid, conf = 0, float(p[self.pend_cls])
        else:
            gid, conf = 0, 0.0
        gap = min(self.t - self.t_last_end, self.gap_cap)
        gap = max(0.0, gap)

        return GestureEventOut(
            gesture_id=int(gid),
            gesture_onset=int(onset),
            gesture_phase=int(self.phase),
            gesture_conf=float(np.clip(conf, 0.0, 1.0)),
            gesture_gap_s=float(gap),
            debug={
                "phase": PHASE_NAMES[self.phase],
                "p_smooth": [float(v) for v in p],
                "p_raw": [float(v) for v in p_raw],
                "cand": int(cand),
                "pending_t": float(self.pend_t),
                "low_run": float(self.low_run),
                "idle_run": float(self.idle_run),
                "cand_run": float(self.cand_run),
                "ret_t": float(self.ret_t),
                "n_onsets": int(self.n_onsets),
                "last_gid": int(self.last_gid),
                "t": float(self.t),
            },
        )

    # ------------------------------------------------------------------
    def run_sequence(self, probs_seq, dt=1.0 / 30.0):
        """把一段概率序列跑完，返回 :class:`GestureEventOut` 列表。"""
        probs_seq = np.asarray(probs_seq, dtype=np.float64)
        return [self.update(probs_seq[i], dt=dt, t=None)
                for i in range(len(probs_seq))]

    def __repr__(self):
        return ("GestureEventMachine(classes=%r, tau=%.3f, enter=%.2f, "
                "exit=%.2f, pending=%.3f, release=%.3f, min_gap=%.3f)"
                % (self.classes, self.tau, float(self.enter[1]),
                   float(self.exit[1]), self.pending_hold, self.release,
                   self.min_gap))
