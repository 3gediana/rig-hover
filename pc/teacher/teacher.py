"""RIG-Hover 教师控制器（规则版）。

定位：不是最终产品，是**数据标注器**（TECH-PLAN §8.1）。输入视觉角度、头部角、
bbox 高度、手势累积量，输出 ``(head_des, vyaw_des, vx_des, mode, emote)``，
用来监督 CfC 训练。所以它保守、平滑、可解释，不追求聪明。

数学口径全部来自 TECH-PLAN-v1.md：

* §7.1 / §7.1b   ``eps_body = theta_img + q_head``（head_cam）或 ``theta_img``（body_cam）
* §7.2           三档分配律：小误差只动头 / 中误差头对准 0.7 + 机身慢跟 / 大误差机身快转
                 + 头顶到限位强制执行机身转
* §7.2b          距离通道：``dist = K / person_h`` → 一阶低通 → 死区 → vx，
                 且 ``|eps_body| > 15°`` 时 vx 必须为 0，人太近时后退减半
* §8.3           全部阈值外置到 YAML，代码里不留数字

状态机（§6.3 / §8.2）：IDLE ⇄ TRACK ⇄ ACK。

手势判定用 A 侧的**事件化输出**：``gesture_onset`` 是边沿（一段连续动作只有
进入 GESTURE 那一帧是 1），所以「一次动作一次 ACK」是**结构保证**，不需要
冷却期。旧实现的 ``wave_score > 4`` 阈值累积与 ``refractory_s`` 冷却已删除；
留下的 ``gesture_gap_s`` 门限是**手势层去抖**（200~300 ms），语义与冷却不同，
见 ``config.yaml`` 的 gesture 组注释。

用法::

    from teacher import TeacherController, TeacherObs, load_config
    ctrl = TeacherController(load_config("config.yaml"))
    act = ctrl.step(TeacherObs(theta_img=3.0, q_head=0.0, seen=1,
                               person_h=0.65, gesture_onset=0, t_now=0.0))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --- mode（CONTRACT §4.1：0=IDLE 1=TRACK 2=ACK 3=REFRACTORY）------------------
# MODE_REFRACTORY 是契约/固件的线格式常量（cfc_task.c 的 obs[102] = mode/3），
# 事件化后**不再进入**这个状态：ACK 直接回 TRACK，重入由 onset 边沿结构性地挡住。
MODE_IDLE = 0
MODE_TRACK = 1
MODE_ACK = 2
MODE_REFRACTORY = 3
MODE_NAMES = ("IDLE", "TRACK", "ACK", "REFRACTORY")

# --- 三档分配律命中档位（只进 debug，不出口）--------------------------------
TIER_NONE = 0      # 死区内，什么都不动
TIER_SMALL = 1     # 小误差：只动头
TIER_MID = 2       # 中误差
TIER_LARGE = 3     # 大误差
TIER_NAMES = ("none", "small", "mid", "large")

# --- 视觉时效档位（§5.4 / §8.3 lost 组）-------------------------------------
DEG_OK = "ok"
DEG_STALE = "stale"
DEG_LOST = "lost"

# --- 表情（CONTRACT §4.1：0..7）----------------------------------------------
EMOTE_NONE = 0
EMOTE_NOD = 1      # ACK 的「点头确认」，具体 EAF 动画由固件映射

CONFIG_PATH_DEFAULT = Path(__file__).with_name("config.yaml")


def clamp(x: float, lo: float, hi: float) -> float:
    """把 x 钳到 [lo, hi]。"""
    return lo if x < lo else (hi if x > hi else x)


def _move_toward(x: float, target: float, max_step: float) -> float:
    """以不超过 max_step 的步长把 x 推向 target。"""
    d = target - x
    if d > max_step:
        return x + max_step
    if d < -max_step:
        return x - max_step
    return target


@dataclass
class TeacherObs:
    """教师控制器的单帧观测（字段与 CONTRACT §4.1 一致）。

    Attributes:
        theta_img: 人相对「视线」的水平角，度，正=右。
        q_head: 头部当前角（相对机身），度。
        seen: 本帧是否检测到人，0/1。
        person_h: bbox 高 / 图高，∈[0,1]（CONTRACT §3.1）。
        hand_up_L: 左手抬起 0/1。
        hand_up_R: 右手抬起 0/1。
        wave_score: **已废弃**，A 侧旧累积量。事件化后不参与任何判定，
            只为旧调用方（``pc/train/synth.py``）还能构造观测而保留。
        t_now: 单调时间戳，秒。
        gesture_onset: A 侧事件机的**边沿**：进入 GESTURE 那一帧为 1，其余 0。
            ACK 的唯一触发源。
        gesture_id: 0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist。
        gesture_conf: 事件机给该手势的置信度 ∈[0,1]。
        gesture_gap_s: 距上次 GESTURE 结束的秒数。``None`` = 未知（不抑制）。
            只在小于 ``ack_min_gap_s`` 时抑制一次响应，语义见 config.yaml。
    """

    theta_img: float = 0.0
    q_head: float = 0.0
    seen: int = 0
    person_h: float = 0.0
    hand_up_L: int = 0
    hand_up_R: int = 0
    wave_score: float = 0.0        # 已废弃：不参与判定
    t_now: float = 0.0
    gesture_onset: int = 0
    gesture_id: int = 0
    gesture_conf: float = 0.0
    gesture_gap_s: float | None = None


@dataclass
class TeacherAction:
    """教师控制器的单帧动作（字段与 CONTRACT §4.1 一致）。

    Attributes:
        head_des: 头部目标角，度，∈[-head_max_deg, head_max_deg]。
        vx_des: 前进速度，米/秒，正=朝人走。
        vyaw_des: 机身角速度，度/秒。符号与 eps_body 一致：人偏右（eps>0）时为正。
            固件里 ``stable_yaw += vyaw_des*dt`` 的实际正方向尚未核实（§15），
            若现场发现转向相反，只在这一处取反即可。
        mode: 0=IDLE 1=TRACK 2=ACK 3=REFRACTORY。
        emote: 0..7，ACK 时为 EMOTE_NOD。
        debug: 中间量，键见 ``TeacherController`` 类文档。
    """

    head_des: float = 0.0
    vx_des: float = 0.0
    vyaw_des: float = 0.0
    mode: int = MODE_IDLE
    emote: int = EMOTE_NONE
    debug: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """给录制器用的扁平字典（debug 不展平）。"""
        return {
            "head_des": self.head_des,
            "vx_des": self.vx_des,
            "vyaw_des": self.vyaw_des,
            "mode": self.mode,
            "emote": self.emote,
        }


def load_config(path: str | Path = CONFIG_PATH_DEFAULT) -> dict:
    """读 YAML 配置。

    Args:
        path: 配置文件路径。

    Returns:
        配置字典。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: YAML 顶层不是 mapping。
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件顶层必须是 mapping：{p}")
    return cfg


def _section(cfg: dict, name: str) -> dict:
    if name not in cfg:
        raise KeyError(f"config 缺少分组 '{name}'（TECH-PLAN §8.3）")
    sec = cfg[name]
    if not isinstance(sec, dict):
        raise ValueError(f"config 分组 '{name}' 必须是 mapping，实际 {type(sec).__name__}")
    return sec


def _num(sec: dict, name: str, key: str) -> float:
    if key not in sec:
        raise KeyError(f"config 缺字段 '{name}.{key}'")
    try:
        return float(sec[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config 字段 '{name}.{key}' 不是数：{sec[key]!r}") from exc


def _bool(sec: dict, name: str, key: str) -> bool:
    if key not in sec:
        raise KeyError(f"config 缺字段 '{name}.{key}'")
    return bool(sec[key])


def _ints(sec: dict, name: str, key: str) -> frozenset[int]:
    """读一个整数集合（可响应的手势 id）。允许写成单个整数或列表。"""
    if key not in sec:
        raise KeyError(f"config 缺字段 '{name}.{key}'")
    val = sec[key]
    if isinstance(val, (int, float)):
        val = [val]
    if not isinstance(val, (list, tuple)) or not val:
        raise ValueError(f"config 字段 '{name}.{key}' 必须是非空列表，实际 {val!r}")
    try:
        return frozenset(int(v) for v in val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config 字段 '{name}.{key}' 里不是整数：{val!r}") from exc


class TeacherController:
    """规则教师：视觉误差 → 头/机身/前进三通道动作。

    输入输出见 CONTRACT §4.1。``step()`` 是纯函数式的（除了内部滤波器与状态机状态），
    所以可以逐帧喂、也可以离线重放。

    debug 字典固定给出：

    ==================== ====================================================
    键                   含义
    ==================== ====================================================
    eps_body             机身朝向误差（度），§7.1 核心公式
    mode / mode_name     状态机当前态
    gesture_onset/id/conf/gap_s
                         A 侧事件化输出的原值（判定依据）
    ack_count            本条序列累计 ACK 次数
    hit_ack_onset/conf/id/gap/hand
                         ACK 五道判据各自的命中情况（排查「为什么没响应」）
    dist_est / dist_raw  低通后 / 原始距离估计（米）
    tier / tier_name     三档分配律命中档位
    hit_*                各档位/保护规则是否命中
    degrade              ok / stale / lost
    wave_score           **已废弃**，A 侧旧累积量原值（仅透传入档）
    ==================== ====================================================
    """

    def __init__(self, cfg: dict):
        """构造。

        Args:
            cfg: ``config.yaml`` 读出来的字典（见 ``load_config``）。

        Raises:
            KeyError: 缺分组或缺字段（配置错误一律炸掉，不静默取默认值）。
            ValueError: 字段类型/范围不合法。
        """
        self.cfg = cfg
        tr = _section(cfg, "tracking")
        gs = _section(cfg, "gesture")
        sm = _section(cfg, "smoothing")
        lo = _section(cfg, "lost")
        di = _section(cfg, "distance")

        if "img_frame" not in cfg:
            raise KeyError("config 缺字段 'img_frame'（0=head_cam 1=body_cam）")
        img_frame = int(cfg["img_frame"])
        if img_frame not in (0, 1):
            raise ValueError(f"img_frame 只能是 0 或 1，实际 {img_frame}")
        self.img_frame = img_frame

        # tracking（§8.3）
        self.head_deadband_deg = _num(tr, "tracking", "head_deadband_deg")
        self.body_engage_deg = _num(tr, "tracking", "body_engage_deg")
        self.head_full_deg = _num(tr, "tracking", "head_full_deg")
        self.head_share_mid = _num(tr, "tracking", "head_share_mid")
        self.k_med = _num(tr, "tracking", "k_med")
        self.k_fast = _num(tr, "tracking", "k_fast")
        self.vyaw_max_dps = _num(tr, "tracking", "vyaw_max_dps")
        self.head_max_deg = _num(tr, "tracking", "head_max_deg")
        self.head_limit_margin_deg = _num(tr, "tracking", "head_limit_margin_deg")
        self.vyaw_floor_dps = _num(tr, "tracking", "vyaw_floor_dps")

        # gesture（§8.3 / §8.4，事件化后）
        # 触发条件只有一条：gesture_onset 边沿 + 置信度 + 可响应集合 + 去抖门。
        # 旧实现的 wave_cycles_needed / integrate_window_s / refractory_s 已删除：
        #   * wave_score 阈值累积删掉（A 侧不再输出这个电平）；
        #   * refractory_s 冷却删掉（一段动作只产生一个 onset 是结构保证）；
        #   * 替代物是 ack_min_gap_s，它挡的是「同一次动作被拆成两个边沿」，
        #     不是「两次动作隔得太近」，取值 200~300 ms（人手再做一次动作 >400 ms）。
        self.ack_hold_s = _num(gs, "gesture", "ack_hold_s")
        self.ack_conf_thresh = _num(gs, "gesture", "ack_conf_thresh")
        self.ack_min_gap_s = _num(gs, "gesture", "ack_min_gap_s")
        self.ack_gesture_ids = _ints(gs, "gesture", "ack_gesture_ids")
        self.require_hand_up = _bool(gs, "gesture", "require_hand_up")
        # 这两个是 A 侧量化关键点/过零用的门限：CONTRACT §3.1 的 hand_up_* 到这里
        # 已经是 0/1。教师侧只读进来保证"参数只有一份来源"，不重复判定。
        self.hand_up_conf = _num(gs, "gesture", "hand_up_conf")
        self.wave_amp_thresh = _num(gs, "gesture", "wave_amp_thresh")

        # smoothing（§8.3）
        self.head_lpf_alpha = _num(sm, "smoothing", "head_lpf_alpha")
        self.vyaw_lpf_alpha = _num(sm, "smoothing", "vyaw_lpf_alpha")
        self.head_rate_limit_dps = _num(sm, "smoothing", "head_rate_limit_dps")
        self.vyaw_rate_limit_dps2 = _num(sm, "smoothing", "vyaw_rate_limit_dps2")
        self.ctrl_fps = _num(sm, "smoothing", "ctrl_fps")
        self.dt_max = _num(sm, "smoothing", "dt_max")
        self.vx_rate_limit_mps2 = _num(sm, "smoothing", "vx_rate_limit_mps2")

        # lost（§8.3 / §5.4）
        self.stale_ms = _num(lo, "lost", "stale_ms")
        self.lost_ms = _num(lo, "lost", "lost_ms")
        self.recenter_ms = _num(lo, "lost", "recenter_ms")

        # distance（§7.2b）
        self.dist_target = _num(di, "distance", "dist_target")
        self.dist_dead = _num(di, "distance", "dist_dead")
        self.k_dist = _num(di, "distance", "k_dist")
        self.vx_max = _num(di, "distance", "vx_max")
        self.vx_max_desktop = _num(di, "distance", "vx_max_desktop")
        self.desktop_mode = _bool(di, "distance", "desktop_mode")
        self.dist_k_coef = _num(di, "distance", "dist_k_coef")
        self.dist_lpf_tau = _num(di, "distance", "dist_lpf_tau")
        self.dist_min = _num(di, "distance", "dist_min")
        self.dist_max = _num(di, "distance", "dist_max")
        self.person_h_min = _num(di, "distance", "person_h_min")
        self.vx_align_deg = _num(di, "distance", "vx_align_deg")
        self.vx_backward_scale = _num(di, "distance", "vx_backward_scale")

        self._validate()

        self.dt_nominal = 1.0 / self.ctrl_fps
        self.vx_limit = min(self.vx_max, self.vx_max_desktop) if self.desktop_mode else self.vx_max
        self.head_recenter_dps = self.head_max_deg / (self.recenter_ms / 1000.0)
        self.reset()

    # ------------------------------------------------------------------ 校验
    def _validate(self) -> None:
        """范围检查。配置写错就在这里炸，不留到运行时。"""
        checks = [
            ("tracking.head_max_deg", self.head_max_deg, 0.0, 45.0),
            ("tracking.vyaw_max_dps", self.vyaw_max_dps, 0.0, 180.0),
            ("tracking.head_share_mid", self.head_share_mid, 0.0, 1.0),
            ("smoothing.head_lpf_alpha", self.head_lpf_alpha, 0.0, 1.0),
            ("smoothing.vyaw_lpf_alpha", self.vyaw_lpf_alpha, 0.0, 1.0),
            ("smoothing.ctrl_fps", self.ctrl_fps, 1.0, 500.0),
            ("distance.dist_target", self.dist_target, 0.0, 10.0),
            ("distance.dist_lpf_tau", self.dist_lpf_tau, 1e-6, 10.0),
            ("distance.vx_max", self.vx_max, 0.0, 1.0),
            ("gesture.ack_conf_thresh", self.ack_conf_thresh, 0.0, 1.0),
            ("gesture.ack_min_gap_s", self.ack_min_gap_s, 0.0, 2.0),
            ("gesture.ack_hold_s", self.ack_hold_s, 0.05, 20.0),
        ]
        for name, val, lo, hi in checks:
            if not (lo <= val <= hi):
                raise ValueError(f"config 字段 {name}={val} 越界 [{lo}, {hi}]")
        bad_ids = sorted(i for i in self.ack_gesture_ids if not 1 <= i <= 5)
        if bad_ids:
            raise ValueError(
                f"config 字段 gesture.ack_gesture_ids 里有非法手势 id：{bad_ids}"
                "（0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist）")
        if self.body_engage_deg > self.head_full_deg:
            raise ValueError("tracking.body_engage_deg 必须 <= tracking.head_full_deg")
        if not (self.stale_ms <= self.lost_ms <= self.recenter_ms):
            raise ValueError("lost 组必须满足 stale_ms <= lost_ms <= recenter_ms")
        if self.recenter_ms <= 0.0:
            raise ValueError("lost.recenter_ms 必须 > 0")

    # ------------------------------------------------------------------ 状态
    def reset(self) -> None:
        """清空状态机、滤波器、计时器（每条录制序列开始处必须调用）。"""
        self._mode = MODE_IDLE
        self._t_last: float | None = None
        self._t_last_seen: float | None = None
        self._eps_last = 0.0
        self._t_ack = -math.inf
        self._n_acks = 0
        self._head_sm: float | None = None
        self._head_out: float | None = None
        self._vyaw_sm = 0.0
        self._vyaw_out = 0.0
        self._vx_out = 0.0
        self._dist_sm: float | None = None

    @property
    def mode(self) -> int:
        """当前状态机状态（0..3）。"""
        return self._mode

    # ------------------------------------------------------------------ 主步
    def step(self, obs: TeacherObs) -> TeacherAction:
        """推进一帧。

        Args:
            obs: 本帧观测（CONTRACT §4.1）。

        Returns:
            本帧动作（CONTRACT §4.1），``debug`` 带全部中间量。
        """
        t = float(obs.t_now)
        seen = bool(int(obs.seen))
        dt = self._advance_time(t)

        if seen:
            self._t_last_seen = t
        age_ms = math.inf if self._t_last_seen is None else max(0.0, (t - self._t_last_seen) * 1000.0)

        # §7.1 / §7.1b：核心公式
        theta_img = float(obs.theta_img)
        q_head = float(obs.q_head)
        eps_meas = theta_img + q_head if self.img_frame == 0 else theta_img
        if seen:
            self._eps_last = eps_meas
            eps_body = eps_meas
        else:
            # 没看见人的帧里 theta_img 没有意义，沿用最后一次有效值
            eps_body = self._eps_last

        degrade = self._degrade_level(seen, age_ms)
        ack_gate = self._update_state(t, seen, age_ms, degrade, obs)
        mode = self._mode

        # §7.2：三档分配律 + 限位保护
        attention = mode == MODE_ACK  # ACK 时全力看人，不再留 30%
        head_cmd, vyaw_cmd, tier, hits = self._allocate(eps_body, attention)

        # §7.2b：距离通道
        vx_cmd, dist_raw, dist_est, vx_hits = self._plan_vx(obs, seen, dt, eps_body)

        # §5.4 / §8.3 lost 组：视觉失效时冻结/回中
        head_target, vyaw_target, vx_target, head_mode = self._degrade_targets(
            degrade, mode, head_cmd, vyaw_cmd, vx_cmd
        )

        # §8.3 smoothing：一阶低通 + 速率限幅
        head_des = self._filter_head(head_target, dt, obs, head_mode)
        vyaw_des = self._filter_vyaw(vyaw_target, dt)
        vx_des = self._filter_vx(vx_target, dt)

        # ACK 的「点头 + 表情」，其它状态清零
        emote = EMOTE_NOD if mode == MODE_ACK else EMOTE_NONE

        # 硬钳位 + NaN 兜底（§7.5 安全壳的教师侧等价物）
        head_des, vyaw_des, vx_des = self._sanitize(head_des, vyaw_des, vx_des)

        dbg = {
            "eps_body": eps_body,
            "eps_held": not seen,
            "theta_img": theta_img,
            "q_head": q_head,
            "img_frame": self.img_frame,
            "mode": mode,
            "mode_name": MODE_NAMES[mode],
            "tier": tier,
            "tier_name": TIER_NAMES[tier],
            "head_cmd_raw": head_cmd,
            "vyaw_cmd_raw": vyaw_cmd,
            "vx_cmd_raw": vx_cmd,
            "head_des": head_des,
            "vyaw_des": vyaw_des,
            "vx_des": vx_des,
            "head_share": 1.0 if attention else self.head_share_mid,
            "wave_score": float(obs.wave_score),      # 已废弃，仅透传记录
            "gesture_onset": int(obs.gesture_onset),
            "gesture_id": int(obs.gesture_id),
            "gesture_conf": float(obs.gesture_conf),
            "gesture_gap_s": (None if obs.gesture_gap_s is None
                              else float(obs.gesture_gap_s)),
            "ack_conf_thresh": self.ack_conf_thresh,
            "ack_min_gap_s": self.ack_min_gap_s,
            "ack_count": self._n_acks,
            "dist_raw": dist_raw,
            "dist_est": dist_est,
            "e_dist": (dist_est - self.dist_target) if dist_est is not None else 0.0,
            "degrade": degrade,
            "seen_age_ms": age_ms,
            "ack_left_s": max(0.0, self.ack_hold_s - (t - self._t_ack)) if mode == MODE_ACK else 0.0,
            "dt": dt,
            # 各档位命中情况
            "hit_deadband": hits["hit_deadband"],
            "hit_small": hits["hit_small"],
            "hit_mid": hits["hit_mid"],
            "hit_large": hits["hit_large"],
            "hit_head_limit": hits["hit_head_limit"],
            "hit_vx_dead": vx_hits["hit_vx_dead"],
            "hit_vx_align": vx_hits["hit_vx_align"],
            "hit_vx_backoff": vx_hits["hit_vx_backoff"],
            "hit_stale": degrade == DEG_STALE,
            "hit_lost": degrade == DEG_LOST,
        }
        dbg.update(ack_gate)      # hit_ack_onset / conf / id / gap / hand
        return TeacherAction(
            head_des=head_des,
            vx_des=vx_des,
            vyaw_des=vyaw_des,
            mode=mode,
            emote=emote,
            debug=dbg,
        )

    # ------------------------------------------------------------- 内部实现
    def _advance_time(self, t: float) -> float:
        """算本帧 dt（秒），并推进内部时间轴。"""
        if self._t_last is None:
            dt = self.dt_nominal
        else:
            dt = t - self._t_last
        self._t_last = t
        if not math.isfinite(dt) or dt < 0.0:
            dt = 0.0
        return min(dt, self.dt_max)

    def _degrade_level(self, seen: bool, age_ms: float) -> str:
        """视觉时效三档。"""
        if seen:
            return DEG_OK
        if age_ms > self.lost_ms:
            return DEG_LOST
        if age_ms > self.stale_ms:
            return DEG_STALE
        return DEG_OK

    def _update_state(self, t: float, seen: bool, age_ms: float, degrade: str, obs: TeacherObs) -> dict:
        """状态机 + 手势判定（§6.3 / §8.4）。

        ACK 的触发条件是**一条边沿**，不是电平::

            gesture_onset == 1
            and gesture_conf >= ack_conf_thresh
            and gesture_id in ack_gesture_ids
            and gesture_gap_s >= ack_min_gap_s   # 手势层去抖，不是冷却
            and (hand_up 或 not require_hand_up)
            and degrade == ok and mode == TRACK

        ACK 结束后直接回 TRACK（或 IDLE）：没有 REFRACTORY 冷却。一段连续动作
        只产生一个 onset 是事件机的结构保证，重入不可能发生。

        Returns:
            本帧 ACK 各判据的命中情况（``hit_ack_*``），直接并进 ``debug``。
        """
        # 1) 纯计时的转换
        if self._mode == MODE_ACK:
            if t - self._t_ack >= self.ack_hold_s:
                self._mode = MODE_TRACK if seen else MODE_IDLE
        elif self._mode == MODE_IDLE and seen:
            self._mode = MODE_TRACK

        # 2) 彻底丢失 → IDLE（§6.3：seen=0 持续 2s）
        if degrade == DEG_LOST and age_ms >= self.recenter_ms:
            self._mode = MODE_IDLE
            self._t_ack = -math.inf

        # 3) 手势：onset 边沿（事件化）
        hand_up = bool(int(obs.hand_up_L)) or bool(int(obs.hand_up_R))
        hand_ok = hand_up or not self.require_hand_up

        gap = obs.gesture_gap_s
        # None / NaN = 生产者没给（或数据坏了）：不抑制，宁可响应也不要静默失效
        if gap is None:
            gap_ok = True
        else:
            g = float(gap)
            gap_ok = (not math.isfinite(g)) or g >= self.ack_min_gap_s

        conf_ok = float(obs.gesture_conf) >= self.ack_conf_thresh
        id_ok = int(obs.gesture_id) in self.ack_gesture_ids
        onset = int(obs.gesture_onset) == 1
        ack_ok = (degrade == DEG_OK and self._mode == MODE_TRACK and onset
                  and conf_ok and id_ok and gap_ok and hand_ok)

        gate = {
            "hit_ack_onset": onset,
            "hit_ack_conf": conf_ok,
            "hit_ack_id": id_ok,
            "hit_ack_gap": not gap_ok,
            "hit_ack_hand": not hand_ok,
        }
        if ack_ok:
            self._mode = MODE_ACK
            self._t_ack = t
            self._n_acks += 1
        return gate

    def _allocate(self, eps: float, attention: bool) -> tuple[float, float, int, dict]:
        """§7.2 三档分配律。

        Args:
            eps: eps_body，度。
            attention: ACK 时为 True，头部不再留 30%。

        Returns:
            (head_cmd, vyaw_cmd, tier, hits)
        """
        a = abs(eps)
        s = 1.0 if eps >= 0.0 else -1.0
        hits = {
            "hit_deadband": False,
            "hit_small": False,
            "hit_mid": False,
            "hit_large": False,
            "hit_head_limit": False,
        }

        if a < self.body_engage_deg:
            # 小误差：只动头，轮子绝对不转（用户的核心要求）
            if a <= self.head_deadband_deg:
                tier = TIER_NONE
                hits["hit_deadband"] = True
                head_cmd = 0.0
            else:
                tier = TIER_SMALL
                hits["hit_small"] = True
                head_cmd = eps
            vyaw_cmd = 0.0
        elif a < self.head_full_deg:
            # 中误差：头先对准（0.7），机身慢慢跟上，留下 30% 当驱动源
            tier = TIER_MID
            hits["hit_mid"] = True
            head_cmd = eps * self.head_share_mid
            vyaw_cmd = s * self.k_med * a
        else:
            # 大误差：机身快转，头反向补偿保持注视
            tier = TIER_LARGE
            hits["hit_large"] = True
            head_cmd = eps - s * self.head_full_deg
            vyaw_cmd = s * min(self.k_fast * a, self.vyaw_max_dps)

        if attention:
            head_cmd = eps  # ACK：全力对准人

        head_cmd = clamp(head_cmd, -self.head_max_deg, self.head_max_deg)

        # 硬件限位保护：头顶死了必须让机身转（优先于「小误差只动头」）
        if abs(head_cmd) > self.head_max_deg - self.head_limit_margin_deg:
            hits["hit_head_limit"] = True
            vyaw_cmd = s * max(abs(vyaw_cmd), self.vyaw_floor_dps)

        vyaw_cmd = clamp(vyaw_cmd, -self.vyaw_max_dps, self.vyaw_max_dps)
        return head_cmd, vyaw_cmd, tier, hits

    def _plan_vx(self, obs: TeacherObs, seen: bool, dt: float, eps: float) -> tuple[float, float | None, float | None, dict]:
        """§7.2b 前进通道。

        Returns:
            (vx_cmd, dist_raw, dist_est, hits)
        """
        hits = {"hit_vx_dead": False, "hit_vx_align": False, "hit_vx_backoff": False}
        dist_raw: float | None = None

        if seen and float(obs.person_h) > self.person_h_min:
            dist_raw = clamp(self.dist_k_coef / float(obs.person_h), self.dist_min, self.dist_max)
            if self._dist_sm is None:
                self._dist_sm = dist_raw
            else:
                a = 1.0 - math.exp(-dt / self.dist_lpf_tau) if dt > 0.0 else 0.0
                self._dist_sm += a * (dist_raw - self._dist_sm)

        if self._dist_sm is None:
            return 0.0, dist_raw, None, hits

        dist_est = self._dist_sm
        e_dist = dist_est - self.dist_target
        vx_cmd = 0.0
        if abs(e_dist) < self.dist_dead:
            hits["hit_vx_dead"] = True
        else:
            vx_cmd = clamp(self.k_dist * e_dist, -self.vx_limit, self.vx_limit)

        # 安全约束 1：没对准绝不前进（否则斜着冲过去撞人）
        if abs(eps) > self.vx_align_deg:
            hits["hit_vx_align"] = True
            vx_cmd = 0.0

        # 安全约束 3：人太近时后退减半（后退时人看不见，反应更慢）
        if e_dist < -self.dist_dead:
            hits["hit_vx_backoff"] = True
            vx_cmd *= self.vx_backward_scale

        return vx_cmd, dist_raw, dist_est, hits

    def _degrade_targets(
        self, degrade: str, mode: int, head_cmd: float, vyaw_cmd: float, vx_cmd: float
    ) -> tuple[float, float, float, str]:
        """视觉失效时的目标覆盖（§5.4 三档：stale 冻结 / lost 回中 / idle 全零）。

        Returns:
            (head_target, vyaw_target, vx_target, head_mode)，
            head_mode ∈ {"filter", "freeze", "recenter"}，决定头部走低通还是直接回中。
        """
        if degrade == DEG_STALE:
            # 冻结头部参考，机身和前进归零（让现有 LQR 把机器人稳在原地）
            return head_cmd, 0.0, 0.0, "freeze"
        if degrade == DEG_LOST:
            # 头部按 head_max/recenter_ms 的速率回中
            return 0.0, 0.0, 0.0, "recenter"
        if mode == MODE_IDLE:
            return 0.0, 0.0, 0.0, "filter"
        return head_cmd, vyaw_cmd, vx_cmd, "filter"

    def _alpha(self, alpha_base: float, dt: float) -> float:
        """把按 ctrl_fps 标定的 alpha 换算到本帧 dt。"""
        if dt <= 0.0:
            return 0.0
        return 1.0 - (1.0 - alpha_base) ** (dt / self.dt_nominal)

    def _filter_head(self, target: float, dt: float, obs: TeacherObs, mode: str = "filter") -> float:
        """头部命令：一阶低通 + 速率限幅。首帧从当前头角起步，不跳变。

        Args:
            target: 目标角（度）。
            dt: 本帧时长（秒）。
            obs: 本帧观测，仅首帧用来取 ``q_head`` 当初始值。
            mode: ``"filter"`` 正常低通；``"freeze"`` 原地保持（stale）；
                ``"recenter"`` 按 ``head_max/recenter_ms`` 的速率直接回中（lost）。
        """
        if self._head_sm is None:
            start = clamp(float(obs.q_head), -self.head_max_deg, self.head_max_deg)
            self._head_sm = start
            self._head_out = start
            return start
        if mode == "freeze":
            self._head_sm = self._head_out
            return self._head_out
        if mode == "recenter":
            self._head_sm = _move_toward(self._head_out, target, self.head_recenter_dps * dt)
            self._head_out = self._head_sm
            return self._head_out
        self._head_sm += self._alpha(self.head_lpf_alpha, dt) * (target - self._head_sm)
        step = self.head_rate_limit_dps * dt
        self._head_out = _move_toward(self._head_out, self._head_sm, step)
        return self._head_out

    def _filter_vyaw(self, target: float, dt: float) -> float:
        """机身角速度：一阶低通 + 加速度限幅。"""
        self._vyaw_sm += self._alpha(self.vyaw_lpf_alpha, dt) * (target - self._vyaw_sm)
        step = self.vyaw_rate_limit_dps2 * dt
        self._vyaw_out = _move_toward(self._vyaw_out, self._vyaw_sm, step)
        if abs(target) < 1e-9 and abs(self._vyaw_out) < step:
            self._vyaw_out = 0.0
            self._vyaw_sm = 0.0
        return self._vyaw_out

    def _filter_vx(self, target: float, dt: float) -> float:
        """前进：只有速率限幅（距离通道本身已有 dist_lpf_tau 的低通）。"""
        step = self.vx_rate_limit_mps2 * dt
        self._vx_out = _move_toward(self._vx_out, target, step)
        if abs(target) < 1e-9 and abs(self._vx_out) < step:
            self._vx_out = 0.0
        return self._vx_out

    def _sanitize(self, head: float, vyaw: float, vx: float) -> tuple[float, float, float]:
        """NaN/Inf 兜底 + 硬钳位（§7.5 安全壳）。"""
        if not (math.isfinite(head) and math.isfinite(vyaw) and math.isfinite(vx)):
            self._head_sm = self._head_out = None
            self._vyaw_sm = self._vyaw_out = 0.0
            self._vx_out = 0.0
            self._dist_sm = None
            return 0.0, 0.0, 0.0
        return (
            clamp(head, -self.head_max_deg, self.head_max_deg),
            clamp(vyaw, -self.vyaw_max_dps, self.vyaw_max_dps),
            clamp(vx, -self.vx_limit, self.vx_limit),
        )
