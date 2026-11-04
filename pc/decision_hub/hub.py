"""hub.py -- DecisionHub：把 CfC 从"控制策略"升级成"决策路由器"。

四层结构（design.md §2）::

    L0 反射   机器人本地 CfC，50Hz，直接驱动 LQR          —— 延迟 ~1.5ms，始终在跑
    L1 技能   机器人自己的动作原语程序（本地播放，零网络）  —— 延迟 = 程序时长
    L2 agent  PC 上的 agent（有帧、有算力、有工具）         —— 延迟 ~0.1s
    L3 云端   语言理解 / 开放域 / 外部工具                  —— 延迟 ~1s+

本模块负责的**只有一件事**：每个 tick 决定"这一刻归谁"。这个问题必须在本地、
在毫秒级回答完，因为它是唯一一个"等不起"的决策——等云端告诉你"这事儿我该
管"，黄花菜都凉了。

三个不变量（测试逐条覆盖）：

1. **L0 永不等待**。升级请求发出去之后，本 tick 仍然输出一条可执行指令
   （等待期策略：车轮清零、头继续跟人）。上层超时只是"没拿到指令"，
   不是"机器人停摆"。
2. **没有校验过的指令绝不执行**。上层的 JSON 走 ``parse_instruction``，
   任何字段非法 -> 丢弃 -> 降级。LQR 前的最后一道闸门是 ``_sanitize_action``。
3. **降级有兜底**。L3 超时 -> L2 -> L1 本地技能 -> L0。每一级失败都记一条
   attempt，demo 会把它打出来。

升级判据的形式（design.md §3 的推导，这里是唯一实现）：

    z      = b + Σ_k w_k · φ_k(obs)                      # 广义可加 logit
    p_esc  = σ(z)
    P_fix  = softmax_L (b_L + Σ_k W_Lk · φ_k)            # 哪一级真能修好
    U(L)   = p_esc · P_fix(L) − C_L                      # 单位：任务成功=1
    level  = argmax_L U(L)，且仅当 max U > 0 才升级

φ_k 是有名字、有量纲、有上界的**风险势函数**（12 个，见 ``POTENTIALS``），
每个都是 C¹ 平滑的分段多项式，没有 if。所以：

* 可解释：logit 精确等于各项贡献之和，"为什么升级"就是贡献排序，不是事后编故事。
* 可训练：13 个参数的凸问题（逻辑回归），比 64→1 的 MLP 少两个数量级的数据。
* 可部署：和 CfC 的 head 一样是一个点积，固件侧 ~50 MAC（FIRMWARE_CHANGES.md §4）。
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

from llm_backend import (
    Backend,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
    Instruction,
    InvalidInstruction,
    MAX_TTL_MS,
    build_escalation_request,
    make_instruction,
)

# --------------------------------------------------------------------------
# 层级
# --------------------------------------------------------------------------

LEVEL_REFLEX = 0
LEVEL_SKILL = 1
LEVEL_AGENT = 2
LEVEL_CLOUD = 3

LEVEL_NAMES = {0: "reflex", 1: "skill", 2: "pc_agent", 3: "cloud"}

EMOTE_NONE = 0
EMOTE_NOD = 1
EMOTE_THINK = 4
EMOTE_ASK = 5

LIM_HEAD_DEG = 45.0
LIM_VYAW_DPS = 90.0
LIM_VX_MPS = 0.30


# --------------------------------------------------------------------------
# 势函数：升级判据的唯一输入
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PotentialSpec:
    """一个风险势函数。

    Attributes:
        name: 唯一名字，同时是 JSON 字段名和固件侧数组下标。
        family: 风险族，决定"缺的是什么资源"，也是 L1 本地技能的选择依据。
        doc: 量纲、单调方向、上下界。design.md §3.2 的表格从这里生成。
        gate_w: 升级头的先验权重（logit 每单位 φ）。
        fix_w: 层级路由头的先验权重 (L1, L2, L3)，含义是"这一族风险该由谁修"。
        ema_tau: >0 时对输入做时间常数 ema_tau 秒的一阶平滑（只给离散检测量用）。
    """

    name: str
    family: str
    doc: str
    gate_w: float
    fix_w: tuple[float, float, float]
    ema_tau: float = 0.0


#: 12 个势函数。顺序即固件侧 ``esc_phi[K]`` 的下标顺序，改动必须同步固件。
#:
#: 权重是**先验**，不是训练结果（当前没有标注数据）。先验的推导见 design.md §3.3：
#: 每个权重量级由"单一信号满值时必须跨过 L2 的代价比 C/(C+V)"反推，不是拍脑袋。
POTENTIALS: tuple[PotentialSpec, ...] = (
    PotentialSpec(
        "u_conf", "blind",
        "视觉置信度不足：φ=1−smoothstep(person_conf; 0.55→0.15)。看不见就没法自己判断。",
        gate_w=4.2, fix_w=(3.0, -1.0, -2.0),
    ),
    PotentialSpec(
        "u_lost", "blind",
        "目标丢失时长：smoothstep(feat_age_ms; 250→1200)，seen=1 时为 0。"
        "丢 250ms 是噪声，丢 1.2s 是「人真走了」。",
        gate_w=4.6, fix_w=(2.6, -1.0, -2.0),
    ),
    PotentialSpec(
        "u_clip", "unknown",
        "特征域外比例：smoothstep(clip_ratio; 0.15→0.50)。50% 是 TECH-PLAN §7.5 的"
        "「视觉不可信」线，这里是它的连续版本。",
        gate_w=3.0, fix_w=(0.2, 1.2, -0.4),
    ),
    PotentialSpec(
        "u_ood", "unknown",
        "场景不在训练分布内：smoothstep(novelty; 2.5σ→5.0σ)，novelty = feat_64 到"
        "训练集均值的马氏距离（PC 侧或固件侧算，随 blob 冻结统计量）。",
        gate_w=3.6, fix_w=(0.2, 1.8, 0.4),
    ),
    PotentialSpec(
        "m_speech", "language",
        "人在说话：0/1（PC 侧 VAD/说话人检测）。本地没有 ASR，这一条本身就是"
        "「本地必须有罪推定」的证据。",
        gate_w=5.0, fix_w=(-0.8, 0.4, 2.2), ema_tau=0.15,
    ),
    PotentialSpec(
        "m_point", "language",
        "指着/举着东西：max(pointing, object_held)，0/1。需要物体识别与指代消解。",
        gate_w=4.4, fix_w=(-1.0, 1.6, 0.4), ema_tau=0.15,
    ),
    PotentialSpec(
        "m_stale_sem", "context",
        "语义通道过期：smoothstep(sem_age_ms; 400→1500)。注意它路由到 L1 而不是 L3——"
        "「我不知道他刚才是不是在说话」不能靠问云端解决，只能本地摆出请再说一遍的姿态。",
        gate_w=2.0, fix_w=(1.6, -0.8, -1.6),
    ),
    PotentialSpec(
        "p_multi", "planning",
        "一句话里多个动作：smoothstep(instr_clauses; 1→3)。多步规划超出 2.2s 记忆。",
        gate_w=5.2, fix_w=(-0.6, -0.4, 2.4),
    ),
    PotentialSpec(
        "p_lookup", "planning",
        "需要查资料/外部工具：0/1（PC 侧语义分类器给出的 need_lookup）。",
        gate_w=4.8, fix_w=(-1.6, -0.8, 2.8),
    ),
    PotentialSpec(
        "s_irrev", "stakes",
        "动作大且没把握：(1−action_conf)·smoothstep(max(|head|/45,|vyaw|/90,|vx|/0.3); 0.25→0.90)。"
        "这是显式写出来的交互项，不让网络自己学。",
        gate_w=2.6, fix_w=(0.2, 1.4, 0.8),
    ),
    PotentialSpec(
        "s_approach", "stakes",
        "要不要靠近：(1−action_conf)·smoothstep(vx_des; 0.03→0.15)·smoothstep(1.1−dist; 0→0.6)。"
        "接近人在社会性上不可逆，值得多问一句。",
        gate_w=3.2, fix_w=(0.4, 1.2, 0.6),
    ),
    PotentialSpec(
        "c_local_ok", "context",
        "反证项：本地确实看得很清楚且没有任何外部语义需求。权重为负，负责把基率压回去，"
        "否则一堆中庸的风险相加就会不停误触发。",
        gate_w=-1.0, fix_w=(0.0, 0.0, 0.0),
    ),
)

POTENTIAL_NAMES: tuple[str, ...] = tuple(p.name for p in POTENTIALS)
K_POTENTIALS: int = len(POTENTIALS)
POTENTIAL_INDEX: dict[str, int] = {n: i for i, n in enumerate(POTENTIAL_NAMES)}

#: 层级代价，单位 = 一次任务成功（V=1 归一化）。这是决策论里的 C_L，
#: 不是计时器：L1 打断跟踪约 0.3s（可能跟丢人），L2 花 CPU+一次 WiFi 往返，
#: L3 花 token + 秒级延迟 + 隐私。design.md §3.4 有推导。
DEFAULT_LEVEL_COST: dict[int, float] = {1: 0.05, 2: 0.12, 3: 0.35}

#: 路由头的层级先验偏置：朝便宜的层级偏，让"代价"和"能力"分开建模。
DEFAULT_ROUTER_BIAS: dict[int, float] = {1: -1.5, 2: 0.0, 3: -0.6}


def smoothstep(x: float, lo: float, hi: float) -> float:
    """C¹ 平滑斜坡：x=lo 处 0，x=hi 处 1，中间 3t²−2t³。

    允许 lo > hi（此时单调递减），范围外精确钳到 0/1。用平滑替代硬阈值是
    "判据可训练"的前提：硬阈值处处导数为 0，梯度根本流不回去。

    Args:
        x: 输入。
        lo: 起点（含）。
        hi: 终点（含）。

    Returns:
        [0, 1] 内的浮点数。x 为 NaN 时返回 0.0（保守：不当作风险）。
    """
    if not math.isfinite(x):
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    t = (x - lo) / (hi - lo)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _safe_float(v: Any, default: float = 0.0) -> float:
    """把任何东西变成有限 float。None / 字符串 / NaN / Inf 一律回默认值。"""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# --------------------------------------------------------------------------
# 观测与动作
# --------------------------------------------------------------------------


@dataclass
class CfcOut:
    """CfC 本帧的输出（与 ctrl_out_t 同构，多一个置信度）。

    Attributes:
        head_des: 头部目标角，度。
        vyaw_des: 机身角速度，度/秒。
        vx_des: 前进速度，米/秒。
        mode: 0=IDLE 1=TRACK 2=ACK 3=REFRACTORY。
        emote: 0..7。
        action_conf: 动作置信度 ∈[0,1]（不要用 softmax 原始概率——分层 softmax
            的 top-1 概率普遍偏高，建议用 mode 头的 (p1−p2) 或对温度标定后的值）。
    """

    head_des: float = 0.0
    vyaw_des: float = 0.0
    vx_des: float = 0.0
    mode: int = 0
    emote: int = 0
    action_conf: float = 1.0

    def as_dict(self) -> dict:
        return {
            "head_des": round(self.head_des, 2),
            "vyaw_des": round(self.vyaw_des, 2),
            "vx_des": round(self.vx_des, 3),
            "mode": int(self.mode),
            "emote": int(self.emote),
            "action_conf": round(self.action_conf, 3),
        }


@dataclass
class HubObs:
    """一帧决策观测：CfC 的输入摘要 + CfC 的输出 + 语义通道。

    Attributes:
        t_now: 单调时间戳，秒。DecisionHub 的所有时序都用它，方便测试注入。
        cfc: CfC 本帧输出。
        seen: 本帧是否看到人，0/1。
        person_conf: 检测置信度。
        clip_ratio: 观测被域外守卫裁掉的比例（固件 clip_ratio()）。
        novelty: feat_64 到训练分布的标准化距离，0 = 典型。
        feat_age_ms: 视觉帧年龄，毫秒。
        dist_est: 距离估计，米。
        sem_age_ms: 语义通道年龄，毫秒；sem_expected=0 时忽略。
        sem_present: 语义通道是否存在过。
        speech: 检测到说话，0/1。
        pointing: 检测到指向，0/1。
        object_held: 检测到举着东西，0/1。
        point_dir_deg: 指向方向（相对图像中心），度。
        instr_clauses: 已识别指令里的动作子句数。
        lookup: 需要查外部资料，0/1。
        gaze_danger: 桌沿危险标志（安全壳最高优先级）。
        robot_state: 1=正常，0=摔倒/被抱起。
        budget_left_ms: 当前任务还剩多少时间预算；默认无穷（不限制）。
    """

    t_now: float = 0.0
    cfc: CfcOut = field(default_factory=CfcOut)

    seen: int = 1
    person_conf: float = 0.9
    clip_ratio: float = 0.0
    novelty: float = 0.0
    feat_age_ms: float = 0.0
    dist_est: float = 1.0
    #: 人相对视线的水平角（度），上层要算"往哪看"就必须有它。
    theta_img: float = 0.0
    #: 头部当前角（度），相对机身。
    q_head: float = 0.0

    sem_expected: int = 1
    sem_present: int = 1
    sem_age_ms: float = 0.0
    speech: float = 0.0
    pointing: float = 0.0
    object_held: float = 0.0
    point_dir_deg: float = 0.0
    instr_clauses: int = 0
    lookup: float = 0.0

    gaze_danger: int = 0
    robot_state: int = 1
    budget_left_ms: float = float("inf")


@dataclass
class CtrlAction:
    """送给仲裁器的一条指令（= CfC 的动作空间，不多一个自由度）。"""

    head_des: float = 0.0
    vyaw_des: float = 0.0
    vx_des: float = 0.0
    mode: int = 0
    emote: int = 0

    def clamped(self) -> "CtrlAction":
        return CtrlAction(
            head_des=_clamp(_safe_float(self.head_des), -LIM_HEAD_DEG, LIM_HEAD_DEG),
            vyaw_des=_clamp(_safe_float(self.vyaw_des), -LIM_VYAW_DPS, LIM_VYAW_DPS),
            vx_des=_clamp(_safe_float(self.vx_des), -LIM_VX_MPS, LIM_VX_MPS),
            mode=int(_clamp(_safe_float(self.mode), 0, 3)),
            emote=int(_clamp(_safe_float(self.emote), 0, 7)),
        )

    def as_dict(self) -> dict:
        c = self.clamped()
        return {
            "head_des": round(c.head_des, 2),
            "vyaw_des": round(c.vyaw_des, 2),
            "vx_des": round(c.vx_des, 3),
            "mode": c.mode,
            "emote": c.emote,
        }


# --------------------------------------------------------------------------
# 决策结果
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """一次路由决策的完整账本（纯函数结果，不含任何状态机副作用）。"""

    level: int
    p_esc: float
    logit: float
    phi: dict[str, float]
    contributions: list[tuple[str, float]]
    pfix: dict[int, float]
    utility: dict[int, float]
    allowed: list[int]
    veto: str = ""
    gated_by: str = ""
    top_family: str = ""

    def reasons(self, k: int = 4) -> list[dict]:
        return [{"k": n, "c": round(c, 4)} for n, c in self.contributions[:k]]

    def reason_text(self, k: int = 4) -> str:
        """人读的判据串：``m_speech=0.94(+4.70) s_approach=0.61(+1.95) ...``"""
        parts = []
        for name, c in self.contributions[:k]:
            parts.append(f"{name}={self.phi.get(name, 0.0):.2f}({c:+.2f})")
        return " ".join(parts) if parts else "no-risk"

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": LEVEL_NAMES.get(self.level, "?"),
            "p_esc": round(self.p_esc, 4),
            "logit": round(self.logit, 4),
            "phi": {k: round(v, 4) for k, v in self.phi.items()},
            "reasons": self.reasons(),
            "pfix": {k: round(v, 4) for k, v in self.pfix.items()},
            "utility": {k: round(v, 4) for k, v in self.utility.items()},
            "allowed": list(self.allowed),
            "veto": self.veto,
            "gated_by": self.gated_by,
            "top_family": self.top_family,
        }


@dataclass
class HubOut:
    """tick() 的返回值 = 机器人这一刻该执行什么。"""

    level: int
    action: CtrlAction
    payload: dict = field(default_factory=dict)
    decision: Decision | None = None
    t_us: float = 0.0

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, "?")

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "action": self.action.as_dict(),
            "t_us": round(self.t_us, 2),
            "payload": self.payload,
        }


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


@dataclass
class HubConfig:
    """DecisionHub 的全部可调项。没有埋在代码里的魔法数字。"""

    gate_bias: float = -3.0
    gate_w: dict[str, float] = field(
        default_factory=lambda: {p.name: p.gate_w for p in POTENTIALS}
    )
    router_bias: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_ROUTER_BIAS))
    router_w: dict[int, dict[str, float]] = field(
        default_factory=lambda: {
            lvl: {p.name: p.fix_w[lvl - 1] for p in POTENTIALS} for lvl in (1, 2, 3)
        }
    )
    level_cost: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_LEVEL_COST))

    #: 每一级允许等待的时长（毫秒）。L1 本地执行，不需要等。
    request_ttl_ms: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 900, 3: 2500})
    #: 一次升级从发起到彻底放弃的总预算（毫秒）。L3+L2 依次超时也不会超过它。
    total_budget_ms: int = 3500
    #: 升级结束后多久内不再升级（冷却），以及同一理由的重复触发需要更高的效用。
    cooldown_ms: int = 600
    refractory_ms: int = 1200
    hyst: float = 0.05
    #: 事件闩锁的复位阈值：p_esc 掉回这个值以下才算"这件事过去了"，才允许为
    #: 同一族理由再次升级。没有它，一个人一直指着杯子，机器人就会每 0.6s 问一次。
    rearm_p: float = 0.30
    #: 触发前必须在门限之上停留多久（毫秒）。让离散检测量的平滑收敛，
    #: 避免"刚过线就按第一版估计选层级"。
    dwell_ms: int = 150

    max_level: int = 3
    enable_skill_level: bool = True
    #: 是否把 P_fix 在"当前可用的层级"里重新归一化。关掉它，大模型一挂就
    #: 什么都不能做了；打开它，机器人会退而求其次做点力所能及的。
    renormalize_pfix: bool = True

    #: 语义通道：多久算"过期"、多久算"彻底没有"。sem_expected=0 时整条忽略。
    sem_stale_ms: float = 400.0
    sem_lost_ms: float = 1500.0
    #: 视觉丢失超过这个值就干脆不升级了（没有信号可问，问了也白问），
    #: 与 control_arbiter.c 的 lost_ms 默认值对齐。
    body_lost_ms: float = 1500.0

    #: 等待上层的 tick 里是否把车轮清零（头继续跟人）。
    hold_vx_while_waiting: bool = True
    #: 离散检测量（speech/pointing/lookup）的一阶平滑时间常数，秒。
    ema_tau: dict[str, float] = field(
        default_factory=lambda: {p.name: p.ema_tau for p in POTENTIALS}
    )

    def validate(self) -> None:
        """配置自检。写错就炸，不留到运行时。

        Raises:
            ValueError: 权重缺项、代价非正、层级越界、预算自相矛盾。
        """
        for name in POTENTIAL_NAMES:
            if name not in self.gate_w:
                raise ValueError(f"gate_w 缺势函数 '{name}'")
        for lvl in (1, 2, 3):
            if lvl not in self.router_w or set(self.router_w[lvl]) != set(POTENTIAL_NAMES):
                raise ValueError(f"router_w[{lvl}] 必须覆盖全部 {K_POTENTIALS} 个势函数")
            if self.level_cost[lvl] <= 0.0:
                raise ValueError(f"level_cost[{lvl}] 必须为正")
            if self.request_ttl_ms[lvl] < 0:
                raise ValueError(f"request_ttl_ms[{lvl}] 不能为负")
        if self.max_level not in (0, 1, 2, 3):
            raise ValueError(f"max_level={self.max_level} 越界")
        if not (0.0 <= self.max_level and sum(self.level_cost.values()) < 10.0):
            raise ValueError("level_cost 量级不对（单位是任务成功，V=1）")
        if self.request_ttl_ms[3] > self.total_budget_ms:
            raise ValueError("request_ttl_ms[3] 超过 total_budget_ms，L3 永远等不到")


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------


def percentile(xs: Sequence[float], q: float) -> float:
    """线性插值分位数（不需要 numpy）。空序列返回 0。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = _clamp(q, 0.0, 1.0) * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


@dataclass
class Stats:
    """延迟与失败计数。demo 直接打印，test 断言。"""

    ticks: int = 0
    escalations: int = 0
    local_us: list[float] = field(default_factory=list)
    dispatch_ms: dict[int, list[float]] = field(
        default_factory=lambda: {1: [], 2: [], 3: []}
    )
    skill_ms: dict[int, list[float]] = field(
        default_factory=lambda: {1: [], 2: [], 3: []}
    )
    timeouts: int = 0
    failures: int = 0
    fallbacks: int = 0
    invalid_input: int = 0
    invalid_instruction: int = 0
    stale_reply: int = 0
    vetoed: int = 0
    by_level: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})

    def record_local(self, us: float) -> None:
        self.ticks += 1
        self.local_us.append(us)
        if len(self.local_us) > 20000:
            del self.local_us[:10000]

    def summary(self) -> dict:
        """扁平统计，含每级 p50/p95/max 延迟。"""
        per: dict[str, dict] = {}
        for lvl in (1, 2, 3):
            d = self.dispatch_ms[lvl]
            s = self.skill_ms[lvl]
            per[str(lvl)] = {
                "name": LEVEL_NAMES[lvl],
                "n": len(d),
                "dispatch_p50_ms": round(percentile(d, 0.5), 3),
                "dispatch_p95_ms": round(percentile(d, 0.95), 3),
                "dispatch_max_ms": round(max(d), 3) if d else 0.0,
                "skill_p50_ms": round(percentile(s, 0.5), 3),
                "skill_p95_ms": round(percentile(s, 0.95), 3),
            }
        return {
            "ticks": self.ticks,
            "escalations": self.escalations,
            "local_us_p50": round(percentile(self.local_us, 0.5), 3),
            "local_us_p95": round(percentile(self.local_us, 0.95), 3),
            "local_us_max": round(max(self.local_us), 3) if self.local_us else 0.0,
            "timeouts": self.timeouts,
            "failures": self.failures,
            "fallbacks": self.fallbacks,
            "invalid_input": self.invalid_input,
            "invalid_instruction": self.invalid_instruction,
            "stale_reply": self.stale_reply,
            "vetoed": self.vetoed,
            "by_level": {LEVEL_NAMES[k]: v for k, v in self.by_level.items()},
            "levels": per,
        }


# --------------------------------------------------------------------------
# 本地技能（L1）
# --------------------------------------------------------------------------

#: L1 技能库：本地程序，零网络。刻意做得很小——它存在的意义是"在等大脑的时候
#: 别僵在那里"，不是"假装自己很聪明"。
LOCAL_SKILLS: dict[str, tuple[str, list[dict]]] = {
    "blind": (
        "reacquire_scan",
        [
            {"op": "head", "deg": 28.0, "ms": 320},
            {"op": "head", "deg": -28.0, "ms": 620},
            {"op": "head", "deg": 0.0, "ms": 320},
            {"op": "track", "ms": 0},
        ],
    ),
    "attention": (
        "ask_attention",
        [{"op": "emote", "id": EMOTE_ASK, "ms": 800}, {"op": "hold", "ms": 300}],
    ),
    "stakes": ("hold_position", [{"op": "hold", "ms": 700}]),
    "unknown": ("hold_position", [{"op": "hold", "ms": 600}]),
    "planning": ("hold_position", [{"op": "hold", "ms": 600}]),
    "language": ("ask_attention", [{"op": "emote", "id": EMOTE_THINK, "ms": 700}]),
    "context": ("ask_attention", [{"op": "emote", "id": EMOTE_ASK, "ms": 700}]),
}


class SkillExecutor:
    """把一个动作程序按时间展开成 50Hz 的 ``CtrlAction`` 流。

    程序里的每个 op 都直接映射到 CfC 的输出通道，所以上层指令天然落在
    CfC 的动作空间里，不需要第二套动作定义（design.md §5.3）。
    """

    def __init__(self, ins: Instruction, start: CtrlAction):
        self.ins = ins
        self.steps = list(ins.prog)
        self.i = 0
        self.t_in = 0.0
        self.t_total = 0.0
        self.head_cur = float(start.head_des)
        self.mode = int(start.mode) if start.mode else 1
        self.cur_index = 0
        self._step_head0 = self.head_cur
        self._entered = False
        self.done = not self.steps

    # -- 内部 ---------------------------------------------------------
    def _enter(self) -> None:
        self._step_head0 = self.head_cur
        self._entered = True

    def _advance(self) -> None:
        """步进到下一步；``track`` 立即结束程序。"""
        self.i += 1
        self.t_in = 0.0
        self._entered = False
        while self.i < len(self.steps) and self.steps[self.i]["op"] == "track":
            self.done = True
            return

    def _dur(self, step: dict) -> float:
        return max(0.0, float(step.get("ms", 0.0))) / 1000.0

    # -- 对外 ---------------------------------------------------------
    def step(self, dt: float) -> CtrlAction | None:
        """推进 dt 秒并返回这一刻的指令；程序结束返回 None。"""
        if self.done or self.i >= len(self.steps):
            return None
        self.cur_index = self.i
        step = self.steps[self.i]
        if not self._entered:
            self._enter()
        dur = self._dur(step)
        op = step["op"]

        a = CtrlAction(head_des=self.head_cur, vyaw_des=0.0, vx_des=0.0,
                        mode=self.mode, emote=EMOTE_NONE)

        if op == "head":
            # 用 t_in + dt 算步进量，保证最后一步真的落在目标角上（不是差一步）
            frac = 1.0 if dur <= 0.0 else _clamp((self.t_in + dt) / dur, 0.0, 1.0)
            a.head_des = self._step_head0 + (float(step["deg"]) - self._step_head0) * frac
        elif op == "yaw":
            a.vyaw_des = float(step["dps"])
            if step.get("compensate", True):
                # 机身转多少，头就反向补多少，把视线钉在原地（§7.3 的同一件事）
                a.head_des = self._step_head0 - float(step["dps"]) * (self.t_in / 1000.0)
        elif op == "vx":
            a.vx_des = float(step["mps"])
        elif op == "emote":
            a.emote = int(step["id"])
        elif op == "say":
            a.emote = int(step.get("id", EMOTE_NOD))
        elif op in ("hold", "stop"):
            a.head_des = self.head_cur
        elif op == "track":
            self.done = True
            return None
        else:  # 校验层已经挡掉了未知 op，这里只是防御
            self.done = True
            return None

        self.t_in += dt
        self.t_total += dt
        self.head_cur = _clamp(a.head_des, -LIM_HEAD_DEG, LIM_HEAD_DEG)

        if dur <= 0.0 or self.t_in >= dur - 1e-9:
            over = self.t_in - dur
            self._advance()
            if self.done:
                return a.clamped()
            self.t_in = max(0.0, over)
        return a.clamped()

    @property
    def elapsed_ms(self) -> float:
        return self.t_total * 1000.0


# --------------------------------------------------------------------------
# DecisionHub
# --------------------------------------------------------------------------


class DecisionHub:
    """四层决策中枢。``tick()`` 是唯一入口，非阻塞。

    Args:
        cfg: 配置；None 用默认值。
        backends: {2: Backend, 3: Backend}。缺失的层级视为不可用，
            路由器会自动跳过（这就是"云端不可用降级"的机制，不是特判）。
        clock: 单调时钟，测试可注入。
        pool_workers: 后端线程池大小。
    """

    STATE_IDLE = "idle"
    STATE_WAIT = "wait"      # 已派发，等上层
    STATE_EXEC = "exec"      # 正在播放动作程序
    STATE_COOLDOWN = "cooldown"

    def __init__(
        self,
        cfg: HubConfig | None = None,
        backends: dict[int, Backend] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        pool_workers: int = 4,
    ):
        self.cfg = cfg or HubConfig()
        self.cfg.validate()
        self.backends: dict[int, Backend] = dict(backends or {})
        self._clock = clock
        self._pool = ThreadPoolExecutor(max_workers=pool_workers, thread_name_prefix="esc")
        self.stats = Stats()
        self._lock = threading.Lock()

        # 状态机
        self.state = self.STATE_IDLE
        self._exec: SkillExecutor | None = None
        self._ins_dict: dict = {}
        self._future: Future | None = None
        self._pending: dict | None = None       # 当前 attempt 的上下文
        self._attempts: list[dict] = []
        self._t_dispatch = 0.0
        self._t_start = 0.0
        self._t_state = 0.0
        self._t_last = 0.0
        self._esc_id = 0
        self._tried: set[int] = set()
        self._last_decision: Decision | None = None
        self._last_top = ""
        self._last_family = ""
        self._last_level_req = 0
        self._t_above: float | None = None
        self._armed = True
        self._last_esc_end = -1e9

        # 势函数的平滑状态
        self._phi_ema: dict[str, float] = {n: 0.0 for n in POTENTIAL_NAMES}
        self._phi: dict[str, float] = {n: 0.0 for n in POTENTIAL_NAMES}

    # ------------------------------------------------------------ 生命周期
    def close(self, wait: bool = False) -> None:
        """关闭线程池。未完成的请求被丢弃，不影响已发布的指令。"""
        with self._lock:
            self.state = self.STATE_IDLE
            self._future = None
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> "DecisionHub":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ 势函数
    def compute_phi(self, obs: HubObs, dt: float) -> dict[str, float]:
        """算 12 个势函数（含离散量的一阶平滑）。

        Args:
            obs: 已 sanitize 的观测。
            dt: 距上一个 tick 的秒数，用于平滑。

        Returns:
            {势函数名: [0,1]}，顺序与 ``POTENTIALS`` 一致。
        """
        c = obs.cfc
        seen = 1.0 if _safe_float(obs.seen) >= 0.5 else 0.0
        age = max(0.0, _safe_float(obs.feat_age_ms))
        conf = _clamp(_safe_float(obs.person_conf), 0.0, 1.0)
        sem_exp = _safe_float(obs.sem_expected) >= 0.5
        sem_age = _safe_float(obs.sem_age_ms, 1e6)
        speech = 1.0 if _safe_float(obs.speech) >= 0.5 else 0.0
        point = 1.0 if _safe_float(obs.pointing) >= 0.5 else 0.0
        held = 1.0 if _safe_float(obs.object_held) >= 0.5 else 0.0
        lookup = 1.0 if _safe_float(obs.lookup) >= 0.5 else 0.0
        conf_act = _clamp(_safe_float(c.action_conf, 1.0), 0.0, 1.0)

        mag = max(
            abs(_safe_float(c.head_des)) / LIM_HEAD_DEG,
            abs(_safe_float(c.vyaw_des)) / LIM_VYAW_DPS,
            abs(_safe_float(c.vx_des)) / LIM_VX_MPS,
        )
        mag = _clamp(mag, 0.0, 1.0)
        dist = _safe_float(obs.dist_est, 1.0)

        raw = {
            "u_conf": smoothstep(conf, 0.55, 0.15) if seen else 0.0,
            "u_lost": 0.0 if seen else smoothstep(age, 250.0, 1200.0),
            "u_clip": smoothstep(_clamp(_safe_float(obs.clip_ratio), 0.0, 1.0), 0.15, 0.50),
            "u_ood": smoothstep(max(0.0, _safe_float(obs.novelty)), 2.5, 5.0),
            "m_speech": speech,
            "m_point": max(point, held),
            "m_stale_sem": (
                smoothstep(sem_age, self.cfg.sem_stale_ms, self.cfg.sem_lost_ms)
                if sem_exp else 0.0
            ),
            "p_multi": smoothstep(float(max(0, int(_safe_float(obs.instr_clauses)))), 1.0, 3.0),
            "p_lookup": lookup,
            "s_irrev": (1.0 - conf_act) * smoothstep(mag, 0.25, 0.90),
            "s_approach": (1.0 - conf_act)
            * smoothstep(_safe_float(c.vx_des), 0.03, 0.15)
            * smoothstep(1.1 - dist, 0.0, 0.6),
            "c_local_ok": seen
            * smoothstep(conf, 0.60, 0.85)
            * (1.0 - max(speech, max(point, held), smoothstep(max(0.0, _safe_float(obs.novelty)), 2.5, 5.0))),
        }

        # 只对离散检测量做一阶平滑：硬 0/1 抖动会让 p_esc 在同一帧里跳一个 logit
        phi: dict[str, float] = {}
        for spec in POTENTIALS:
            v = _clamp(raw[spec.name], 0.0, 1.0)
            tau = self.cfg.ema_tau.get(spec.name, 0.0)
            if tau > 0.0:
                a = 1.0 if dt <= 0.0 else _clamp(dt / (tau + dt), 0.0, 1.0)
                self._phi_ema[spec.name] += a * (v - self._phi_ema[spec.name])
                v = self._phi_ema[spec.name]
            phi[spec.name] = v
        self._phi = phi
        return phi

    # ------------------------------------------------------------ 决策
    def decide(self, phi: dict[str, float], obs: HubObs | None = None) -> Decision:
        """纯函数：势函数 -> 层级。没有副作用，可以离线批量跑。

        Args:
            phi: ``compute_phi`` 的输出（也接受任意 dict，缺项按 0 处理）。
            obs: 用于安全否决（桌沿/摔倒/视觉彻底丢失）。None 时不做否决。

        Returns:
            ``Decision``。
        """
        cfg = self.cfg
        logit = cfg.gate_bias
        contrib: list[tuple[str, float]] = []
        for name in POTENTIAL_NAMES:
            v = _clamp(_safe_float(phi.get(name, 0.0)), 0.0, 1.0)
            c = cfg.gate_w[name] * v
            logit += c
            contrib.append((name, c))
        p_esc = 1.0 / (1.0 + math.exp(-_clamp(logit, -30.0, 30.0)))
        contrib.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_family = next(
            (p.family for p in POTENTIALS if p.name == contrib[0][0]), ""
        ) if contrib else ""

        # 安全否决：这些条件下"升级"没有意义，只有本地停车有意义。
        veto = ""
        if obs is not None:
            if _safe_float(obs.robot_state, 1.0) < 0.5:
                veto = "robot_off"
            elif _safe_float(obs.gaze_danger) >= 0.5:
                veto = "gaze_danger"
            elif _safe_float(obs.feat_age_ms, 1.0) > cfg.body_lost_ms:
                veto = "vision_lost"

        # 路由：P_fix(L) = softmax(b_L + Σ_k W_Lk φ_k)
        logits: dict[int, float] = {}
        for lvl in (1, 2, 3):
            z = cfg.router_bias[lvl]
            for name in POTENTIAL_NAMES:
                z += cfg.router_w[lvl][name] * _clamp(_safe_float(phi.get(name, 0.0)), 0.0, 1.0)
            logits[lvl] = z
        zmax = max(logits.values())
        exps = {lvl: math.exp(z - zmax) for lvl, z in logits.items()}
        zsum = sum(exps.values()) or 1.0
        pfix = {lvl: exps[lvl] / zsum for lvl in exps}

        # 可用性：不可用的层级不参与竞争（这是降级的第一层机制）
        allowed = [0]
        if cfg.enable_skill_level:
            allowed.append(1)
        for lvl in (2, 3):
            if lvl > cfg.max_level:
                continue
            be = self.backends.get(lvl)
            if be is not None and be.available():
                allowed.append(lvl)

        # 只在"还能选的资源"里比较：大模型全挂时 L1 的条件概率就是 1，
        # 于是"做点力所能及的"自动变成唯一选项，而不是被 0.001 的概率压死。
        live = [l for l in (1, 2, 3) if l in allowed]
        zlive = sum(pfix[l] for l in live) or 1.0
        pfix_eff = {l: (pfix[l] / zlive if l in live else 0.0) for l in (1, 2, 3)}
        if cfg.renormalize_pfix:
            pfix = pfix_eff

        utility: dict[int, float] = {0: 0.0}
        for lvl in allowed:
            if lvl == 0:
                continue
            utility[lvl] = p_esc * pfix[lvl] - cfg.level_cost[lvl]

        best_lvl, best_u = 0, 0.0
        for lvl, u in utility.items():
            if lvl != 0 and u > best_u:
                best_lvl, best_u = lvl, u

        gated_by = ""
        if veto:
            best_lvl, best_u, gated_by = 0, 0.0, f"veto:{veto}"
        elif best_lvl == 0:
            # 只在"本来能升级"的情况下区分原因，方便现场诊断
            if [l for l in (1, 2, 3) if l in allowed] == []:
                gated_by = "unavailable"
            elif max(utility.values(), default=0.0) <= 0.0:
                gated_by = "below_cost"

        return Decision(
            level=best_lvl,
            p_esc=p_esc,
            logit=logit,
            phi={n: _clamp(_safe_float(phi.get(n, 0.0)), 0.0, 1.0) for n in POTENTIAL_NAMES},
            contributions=contrib,
            pfix=pfix,
            utility=utility,
            allowed=allowed,
            veto=veto,
            gated_by=gated_by,
            top_family=top_family,
        )

    # ------------------------------------------------------------ tick
    def tick(self, obs: HubObs, dt: float | None = None) -> HubOut:
        """推进一帧，返回这一刻该执行的动作。**永不阻塞**。

        Args:
            obs: 本帧观测。字段非法（None/NaN/Inf/超范围）会被就地清洗并计数，
                不会抛异常——机器人上没有一个字段是可信的。
            dt: 距上一拍秒数；None 时由 obs.t_now 差分得到。

        Returns:
            ``HubOut``：level + action + payload（含判据、尝试记录、耗时）。
        """
        t0 = self._clock()
        obs = self._sanitize_obs(obs)
        t = _safe_float(obs.t_now, 0.0)
        if dt is None:
            dt = 0.0 if self._t_last <= 0.0 else t - self._t_last
        dt = _clamp(_safe_float(dt, 0.02), 0.0, 0.2)
        if dt <= 0.0:
            dt = 1.0 / 50.0
        self._t_last = t

        phi = self.compute_phi(obs, dt)

        out: HubOut | None = None
        if self.state == self.STATE_WAIT:
            out = self._poll(t, obs)
        elif self.state == self.STATE_EXEC:
            out = self._exec_tick(dt, obs, t)
        elif self.state == self.STATE_COOLDOWN and (t - self._t_state) * 1000.0 < self.cfg.cooldown_ms:
            out = self._reflex(obs, phi, t, note=f"cooldown {self.cfg.cooldown_ms - (t - self._t_state) * 1000.0:.0f}ms left")
        else:
            self.state = self.STATE_IDLE

        if out is None:  # IDLE（或冷却到期）：正常决策
            dec = self.decide(phi, obs)
            self._last_decision = dec
            if dec.p_esc < self.cfg.rearm_p:
                self._armed = True  # 这件事过去了，重新武装
            if dec.level > 0:
                if self._t_above is None:
                    self._t_above = t
            else:
                self._t_above = None

            # 事件闩锁：同一件事（层级不升反降）不重复问；层级升高说明问题变大，
            # 可以立刻再升级。
            latched = (not self._armed) and dec.level <= self._last_level_req
            dwelled = (self._t_above is not None
                       and (t - self._t_above) * 1000.0 >= self.cfg.dwell_ms)

            if dec.level > 0 and not latched and dwelled:
                out = self._start_escalation(dec, obs, phi, t)
            elif dec.level > 0:
                self.stats.by_level[0] += 1
                out = HubOut(
                    level=0,
                    action=self._reflex_action(obs),
                    payload={"p_esc": round(dec.p_esc, 4), "reasons": dec.reasons(),
                             "why": dec.reason_text(),
                             "gated_by": "latched" if latched else "dwell",
                             "note": (f"已就 {dec.top_family} 升级过，事件未结束"
                                      if latched else
                                      f"等待判据稳定 {self.cfg.dwell_ms}ms")},
                    decision=dec,
                )
            else:
                self.stats.by_level[0] += 1
                out = HubOut(
                    level=0,
                    action=self._reflex_action(obs),
                    payload={"p_esc": round(dec.p_esc, 4), "reasons": dec.reasons(),
                             "why": dec.reason_text(), "gated_by": dec.gated_by},
                    decision=dec,
                )

        out.t_us = (self._clock() - t0) * 1e6
        if out.decision is not None and out.decision.veto:
            self.stats.vetoed += 1
        self.stats.record_local(out.t_us)
        return out

    # ------------------------------------------------------------ 内部：L0
    def _reflex_action(self, obs: HubObs) -> CtrlAction:
        c = obs.cfc
        return CtrlAction(c.head_des, c.vyaw_des, c.vx_des, c.mode, c.emote).clamped()

    def _hold_action(self, obs: HubObs, emote: int = EMOTE_THINK) -> CtrlAction:
        """等待上层时的策略：车轮清零、头继续跟人、脸上挂"在想"。

        为什么不是"保持上一帧"：以 0.3 m/s 等 2.5s 就是 0.75m 的盲走。
        为什么不是"全停"：头是唯一还在采集信息的通道，停下来就真的瞎了。
        """
        a = self._reflex_action(obs)
        if self.cfg.hold_vx_while_waiting:
            a.vx_des = 0.0
            a.vyaw_des = 0.0
        a.emote = emote
        return a.clamped()

    def _reflex(self, obs: HubObs, phi: dict[str, float], t: float, note: str = "") -> HubOut:
        dec = self.decide(phi, obs)
        return HubOut(
            level=0,
            action=self._reflex_action(obs),
            payload={"p_esc": round(dec.p_esc, 4), "why": dec.reason_text(),
                     "gated_by": dec.gated_by, "note": note},
            decision=dec,
        )

    # ------------------------------------------------------------ 内部：升级
    def _start_escalation(
        self, dec: Decision, obs: HubObs, phi: dict[str, float], t: float
    ) -> HubOut:
        """按决策结果发起升级。返回本 tick 的指令（通常是"等待"策略）。"""
        lvl = dec.level
        # 冷却期内同理由重复触发需要更高的效用（防抖，不是防升级）
        if (t - self._last_esc_end) * 1000.0 < self.cfg.refractory_ms:
            top = dec.contributions[0][0] if dec.contributions else ""
            if top == self._last_top and max(dec.utility.values(), default=0.0) < self.cfg.hyst:
                self.stats.by_level[0] += 1
                return HubOut(
                    level=0, action=self._reflex_action(obs),
                    payload={"p_esc": round(dec.p_esc, 4), "why": dec.reason_text(),
                             "gated_by": "refractory"},
                    decision=dec,
                )

        self._esc_id += 1
        self._t_start = t
        self._tried = set()
        self._attempts = []
        self._armed = False
        self._last_family = dec.top_family
        self._last_level_req = dec.level
        self.stats.escalations += 1
        return self._dispatch(lvl, dec, obs, phi, t)

    def _dispatch(
        self, lvl: int, dec: Decision, obs: HubObs, phi: dict[str, float], t: float
    ) -> HubOut:
        """把请求交给第 lvl 级。"""
        self._tried.add(lvl)
        self._t_state = t

        # ---- L1：本地技能，不需要网络，当拍就能开始 ----
        if lvl == 1:
            key = dec.top_family or "unknown"
            intent, prog = LOCAL_SKILLS.get(key, LOCAL_SKILLS["unknown"])
            try:
                ins = make_instruction(
                    level=1, intent=intent, prog=prog, ttl_ms=2000,
                    source="local-skill", esc_id=self._esc_id, confidence=1.0,
                )
            except InvalidInstruction as exc:  # 技能库写错是我们的 bug，不该发生
                self.stats.invalid_instruction += 1
                self.state = self.STATE_IDLE
                return HubOut(level=0, action=self._reflex_action(obs),
                              payload={"error": f"local skill invalid: {exc}"}, decision=dec)
            self._attempts.append({"level": 1, "backend": "local-skill", "ok": True,
                                   "latency_ms": 0.0, "intent": ins.intent})
            self._begin_exec(ins, obs, t)
            return self._exec_tick(0.0, obs, t, first=True)

        # ---- L2 / L3：异步派发，绝不阻塞 ----
        be = self.backends.get(lvl)
        if be is None or not be.available():
            self._attempts.append({"level": lvl, "backend": "-", "ok": False,
                                   "err": "unavailable", "latency_ms": 0.0})
            self.stats.failures += 1
            return self._step_down(dec, obs, phi, t, why="unavailable")

        ttl = min(int(self.cfg.request_ttl_ms.get(lvl, 1000)), MAX_TTL_MS)
        request = self._build_request(lvl, dec, obs, phi, ttl)
        self._pending = {"level": lvl, "backend": be.name, "ttl_ms": ttl,
                         "request": request, "dec": dec, "phi": phi}
        self._t_dispatch = self._clock()
        self.state = self.STATE_WAIT
        try:
            self._future = self._pool.submit(be.ask, request, ttl / 1000.0)
        except RuntimeError as exc:  # 线程池已关闭：当成这一级不可用
            self.stats.failures += 1
            self._pending = None
            self._future = None
            self.state = self.STATE_IDLE
            self._attempts.append({"level": lvl, "backend": be.name, "ok": False,
                                   "latency_ms": 0.0, "err": f"pool: {exc}"})
            return self._step_down(dec, obs, phi, t, why="pool")
        return HubOut(
            level=lvl,
            action=self._hold_action(obs),
            payload={
                "esc_id": self._esc_id,
                "status": "waiting",
                "level": lvl,
                "backend": be.name,
                "ttl_ms": ttl,
                "p_esc": round(dec.p_esc, 4),
                "reasons": dec.reasons(),
                "why": dec.reason_text(),
                "request": request,
            },
            decision=dec,
        )

    def _build_request(
        self, lvl: int, dec: Decision, obs: HubObs, phi: dict[str, float], ttl: int
    ) -> dict:
        c = obs.cfc
        return build_escalation_request(
            esc_id=self._esc_id,
            level=lvl,
            p_esc=dec.p_esc,
            phi=phi,
            contributions=dec.contributions,
            obs={
                "seen": int(_safe_float(obs.seen) >= 0.5),
                "person_conf": _safe_float(obs.person_conf),
                "person_h": 0.0,   # 由上层按需 GET /capture 取，不放这里
                "theta_img": _safe_float(obs.theta_img),
                "q_head": _safe_float(obs.q_head),
                "clip_ratio": _safe_float(obs.clip_ratio),
                "novelty": _safe_float(obs.novelty),
                "feat_age_ms": _safe_float(obs.feat_age_ms),
                "dist_est": _safe_float(obs.dist_est),
                "speech": int(_safe_float(obs.speech) >= 0.5),
                "pointing": int(_safe_float(obs.pointing) >= 0.5),
                "object_held": int(_safe_float(obs.object_held) >= 0.5),
                "point_dir_deg": _safe_float(obs.point_dir_deg),
                "instr_clauses": int(_safe_float(obs.instr_clauses)),
                "lookup": int(_safe_float(obs.lookup) >= 0.5),
                "mode": int(_safe_float(c.mode)),
                "action": c.as_dict(),
                "gaze_danger": int(_safe_float(obs.gaze_danger) >= 0.5),
            },
            ttl_ms=ttl,
            capture_hint="GET http://<robot>/capture  # 240x240 JPEG",
        )

    def _poll(self, t: float, obs: HubObs) -> HubOut | None:
        """等待期：看有没有结果 / 有没有超预算。返回 None 表示"继续等"。"""
        p = self._pending
        fut = self._future
        if p is None or fut is None:
            self.state = self.STATE_IDLE
            return None

        dec: Decision = p["dec"]
        lvl = p["level"]

        if fut.done():
            latency_ms = (self._clock() - self._t_dispatch) * 1000.0
            self.stats.dispatch_ms[lvl].append(latency_ms)
            try:
                ins = fut.result(timeout=0.0)
            except (BackendUnavailable, BackendTimeout, BackendError, InvalidInstruction) as exc:
                kind = type(exc).__name__
                self._attempts.append({"level": lvl, "backend": p["backend"], "ok": False,
                                       "latency_ms": round(latency_ms, 2), "err": f"{kind}: {exc}"})
                if isinstance(exc, BackendTimeout):
                    self.stats.timeouts += 1
                elif isinstance(exc, InvalidInstruction):
                    self.stats.invalid_instruction += 1
                else:
                    self.stats.failures += 1
                self._pending = None
                self._future = None
                return self._step_down(dec, obs, self._phi, t, why=kind)
            except Exception as exc:  # 后端实现里的意外异常不能拖垮控制循环
                self._attempts.append({"level": lvl, "backend": p["backend"], "ok": False,
                                       "latency_ms": round(latency_ms, 2),
                                       "err": f"{type(exc).__name__}: {exc}"})
                self.stats.failures += 1
                self._pending = None
                self._future = None
                return self._step_down(dec, obs, self._phi, t, why="exception")

            # 迟到的回复（上一次升级的）绝不允许执行
            if ins.esc_id >= 0 and ins.esc_id != self._esc_id:
                self._attempts.append({"level": lvl, "backend": p["backend"], "ok": False,
                                       "latency_ms": round(latency_ms, 2), "err": "stale_id"})
                self.stats.stale_reply += 1
                self._pending = None
                self._future = None
                return self._step_down(dec, obs, self._phi, t, why="stale_id")

            self._attempts.append({"level": lvl, "backend": p["backend"], "ok": True,
                                   "latency_ms": round(latency_ms, 2), "intent": ins.intent})
            self._pending = None
            self._future = None
            ins.level = ins.level or lvl
            self._begin_exec(ins, obs, t)
            return self._exec_tick(0.0, obs, t, first=True)

        waited_ms = (t - self._t_state) * 1000.0          # 本级已经等了多久
        total_ms = (t - self._t_start) * 1000.0            # 整条升级链已经耗了多久
        if waited_ms > p["ttl_ms"] or total_ms > self.cfg.total_budget_ms:
            self.stats.timeouts += 1
            # 超时也是一个真实的延迟样本：我们花了这么久，什么也没拿到。
            self.stats.dispatch_ms[lvl].append(waited_ms)
            self._attempts.append({"level": lvl, "backend": p["backend"], "ok": False,
                                   "latency_ms": round(waited_ms, 2), "err": "timeout"})
            try:
                fut.cancel()
            except Exception:  # noqa: BLE001 - cancel 只影响未开始的 future
                pass
            self._pending = None
            self._future = None
            return self._step_down(dec, obs, self._phi, t, why="timeout")

        # 还在等：继续输出"等待"策略，绝不阻塞
        return HubOut(
            level=lvl,
            action=self._hold_action(obs),
            payload={
                "esc_id": self._esc_id, "status": "waiting", "level": lvl,
                "backend": p["backend"], "waited_ms": round(waited_ms, 1),
                "ttl_ms": p["ttl_ms"], "why": dec.reason_text(),
                "reasons": dec.reasons(), "p_esc": round(dec.p_esc, 4),
                "attempts": self._attempts,
            },
            decision=dec,
        )

    def _step_down(
        self, dec: Decision, obs: HubObs, phi: dict[str, float], t: float, why: str
    ) -> HubOut:
        """当前级失败：在还没试过的层级里挑一个仍然划算的，最后落到 L1 本地技能。

        "划算"用的是发起升级时算好的 U(L)：失败不改变"哪一级有能力解决"的估计，
        只改变"还剩哪些资源可试"。所以这里不是无脑降一级，而是重新挑一次。
        """
        cur = max(self._tried) if self._tried else 3
        budget_used = (t - self._t_start) * 1000.0
        best: tuple[int, float] | None = None
        for nxt in range(cur - 1, 0, -1):
            if nxt in self._tried:
                continue
            if nxt > self.cfg.max_level:
                continue
            if nxt > 1:
                be = self.backends.get(nxt)
                if be is None or not be.available():
                    continue
                # 剩余预算装不下这一级的等待时间就别试了
                if budget_used + self.cfg.request_ttl_ms.get(nxt, 0) > self.cfg.total_budget_ms:
                    continue
                if dec.utility.get(nxt, 0.0) <= 0.0:
                    continue
            u = dec.utility.get(nxt, 0.0) if nxt > 1 else 0.0
            if best is None or u > best[1]:
                best = (nxt, u)
        if best is not None:
            self.stats.fallbacks += 1
            return self._dispatch(best[0], dec, obs, phi, t)

        # 全部失败 -> L1 兜底技能（本地，一定成功）
        self.stats.fallbacks += 1
        key = dec.top_family or "unknown"
        intent, prog = LOCAL_SKILLS.get(key, LOCAL_SKILLS["unknown"])
        ins = make_instruction(
            level=1, intent="fallback_" + intent, prog=prog, ttl_ms=2000,
            source="local-fallback", esc_id=self._esc_id, confidence=1.0,
        )
        self._attempts.append({"level": 1, "backend": "local-fallback", "ok": True,
                               "latency_ms": 0.0, "err_prev": why, "intent": ins.intent})
        self._begin_exec(ins, obs, t)
        return self._exec_tick(0.0, obs, t, first=True)

    def _begin_exec(self, ins: Instruction, obs: HubObs, t: float) -> None:
        self._exec = SkillExecutor(ins, self._reflex_action(obs))
        self._ins_dict = ins.as_dict()   # 每 tick 不再重复深拷贝
        self.state = self.STATE_EXEC
        self._t_state = t
        self.stats.by_level[ins.level] += 1

    def _exec_tick(self, dt: float, obs: HubObs, t: float, first: bool = False) -> HubOut:
        ex = self._exec
        if ex is None:
            self.state = self.STATE_IDLE
            return HubOut(level=0, action=self._reflex_action(obs), payload={"status": "exec_missing"})
        a = ex.step(0.0 if first else dt)
        if a is None:
            self._finish_exec(t, ex)
            return HubOut(
                level=0, action=self._reflex_action(obs),
                payload={"status": "exec_done", "esc_id": self._esc_id,
                         "skill_ms": round(ex.elapsed_ms, 1),
                         "attempts": self._attempts},
                decision=self._last_decision,
            )
        return HubOut(
            level=ex.ins.level,
            action=a.clamped(),
            payload={
                "esc_id": self._esc_id,
                "status": "executing",
                "intent": ex.ins.intent,
                "source": ex.ins.source,
                "step": f"{ex.cur_index + 1}/{len(ex.steps)}",
                "say": ex.ins.say,
                "instruction": self._ins_dict,
                "attempts": self._attempts,
                "p_esc": round(self._last_decision.p_esc, 4) if self._last_decision else 0.0,
                "why": self._last_decision.reason_text() if self._last_decision else "",
                "reasons": self._last_decision.reasons() if self._last_decision else [],
            },
            decision=self._last_decision,
        )

    def _finish_exec(self, t: float, ex: SkillExecutor) -> None:
        self.stats.skill_ms[ex.ins.level].append(ex.elapsed_ms)
        self._exec = None
        self.state = self.STATE_COOLDOWN
        self._t_state = t
        self._last_esc_end = t
        if self._last_decision and self._last_decision.contributions:
            self._last_top = self._last_decision.contributions[0][0]

    # ------------------------------------------------------------ 清洗
    def _sanitize_obs(self, obs: HubObs) -> HubObs:
        """把一个可能很脏的观测洗成可用的。**永不抛异常**。"""
        if not isinstance(obs, HubObs):
            self.stats.invalid_input += 1
            obs = HubObs()
        c = obs.cfc if isinstance(obs.cfc, CfcOut) else CfcOut()
        bad = 0

        def num(v: Any, lo: float, hi: float, default: float) -> float:
            nonlocal bad
            f = _safe_float(v, default)
            if f != v and not isinstance(v, (int, float)):
                bad += 1
            if not math.isfinite(_safe_float(v, float("nan"))):
                bad += 1
            return _clamp(f, lo, hi)

        clean_cfc = CfcOut(
            head_des=num(c.head_des, -LIM_HEAD_DEG * 2, LIM_HEAD_DEG * 2, 0.0),
            vyaw_des=num(c.vyaw_des, -LIM_VYAW_DPS * 2, LIM_VYAW_DPS * 2, 0.0),
            vx_des=num(c.vx_des, -LIM_VX_MPS * 2, LIM_VX_MPS * 2, 0.0),
            mode=int(num(c.mode, 0, 3, 0)),
            emote=int(num(c.emote, 0, 7, 0)),
            action_conf=num(c.action_conf, 0.0, 1.0, 1.0),
        )
        clean = replace(
            obs,
            t_now=num(obs.t_now, 0.0, 1e9, 0.0),
            cfc=clean_cfc,
            seen=int(num(obs.seen, 0.0, 1.0, 1.0) >= 0.5),
            person_conf=num(obs.person_conf, 0.0, 1.0, 0.0),
            clip_ratio=num(obs.clip_ratio, 0.0, 1.0, 0.0),
            novelty=num(obs.novelty, 0.0, 10.0, 0.0),
            feat_age_ms=num(obs.feat_age_ms, 0.0, 60000.0, 0.0),
            dist_est=num(obs.dist_est, 0.05, 10.0, 1.0),
            theta_img=num(obs.theta_img, -180.0, 180.0, 0.0),
            q_head=num(obs.q_head, -90.0, 90.0, 0.0),
            sem_expected=int(num(obs.sem_expected, 0.0, 1.0, 1.0) >= 0.5),
            sem_present=int(num(obs.sem_present, 0.0, 1.0, 1.0) >= 0.5),
            sem_age_ms=num(obs.sem_age_ms, 0.0, 60000.0, 1e6),
            speech=num(obs.speech, 0.0, 1.0, 0.0),
            pointing=num(obs.pointing, 0.0, 1.0, 0.0),
            object_held=num(obs.object_held, 0.0, 1.0, 0.0),
            point_dir_deg=num(obs.point_dir_deg, -180.0, 180.0, 0.0),
            instr_clauses=int(num(obs.instr_clauses, 0, 16, 0)),
            lookup=num(obs.lookup, 0.0, 1.0, 0.0),
            gaze_danger=int(num(obs.gaze_danger, 0.0, 1.0, 0.0) >= 0.5),
            robot_state=int(num(obs.robot_state, 0.0, 1.0, 1.0) >= 0.5),
            budget_left_ms=_safe_float(obs.budget_left_ms, float("inf")),
        )
        if bad:
            self.stats.invalid_input += 1
        return clean

    # ------------------------------------------------------------ 统计
    def stats_summary(self) -> dict:
        """``Stats.summary()`` 的别名（demo/test 用）。"""
        return self.stats.summary()

    def format_stats(self) -> str:
        """人能看懂的一段延迟表。"""
        s = self.stats.summary()
        lines = [
            f"ticks={s['ticks']} escalations={s['escalations']} "
            f"timeouts={s['timeouts']} failures={s['failures']} fallbacks={s['fallbacks']} "
            f"invalid_in={s['invalid_input']} invalid_ins={s['invalid_instruction']}",
            f"L0 decision path: p50={s['local_us_p50']:.1f}us "
            f"p95={s['local_us_p95']:.1f}us max={s['local_us_max']:.1f}us",
        ]
        for lvl in ("1", "2", "3"):
            d = s["levels"][lvl]
            if not d["n"]:
                continue
            lines.append(
                f"L{lvl} {d['name']:<9} n={d['n']:<3} dispatch p50={d['dispatch_p50_ms']:.2f}ms "
                f"p95={d['dispatch_p95_ms']:.2f}ms max={d['dispatch_max_ms']:.2f}ms | "
                f"skill p50={d['skill_p50_ms']:.0f}ms"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 训练（可训练性的证据，不是可训练性的声明）
# --------------------------------------------------------------------------


def fit_gate(
    samples: Sequence[tuple[dict[str, float], int]],
    *,
    init: HubConfig | None = None,
    epochs: int = 600,
    lr: float = 0.35,
    l2: float = 1e-3,
) -> tuple[dict[str, float], float, dict]:
    """用标注数据拟合升级头（13 参数的逻辑回归，凸问题）。

    这一层必须可训练，否则它就是一堆手调阈值。选择逻辑回归而不是 MLP 的理由
    见 design.md §3.3：参数少 3 个数量级、凸、权重直接读作 log-odds。

    Args:
        samples: [(φ 字典, label)]，label=1 表示"本地确实处理不了"。
        init: 初始配置，默认用先验权重（热启动比随机初始化收敛快得多）。
        epochs: 全批量梯度下降轮数。
        lr: 学习率。
        l2: L2 正则（防共线势函数把权重推到 ±∞）。

    Returns:
        (权重字典, 偏置, 训练信息)。

    Raises:
        ValueError: 样本为空或标签不是 0/1。
    """
    if not samples:
        raise ValueError("fit_gate: 样本为空")
    try:
        import torch
    except ImportError as exc:  # 训练是离线步骤，部署链不依赖 torch
        raise RuntimeError("需要 torch 才能训练升级头") from exc

    cfg = init or HubConfig()
    x = torch.tensor(
        [[_clamp(_safe_float(phi.get(n, 0.0)), 0.0, 1.0) for n in POTENTIAL_NAMES] for phi, _ in samples],
        dtype=torch.float32,
    )
    y = torch.tensor([float(lbl) for _, lbl in samples], dtype=torch.float32)
    if float(y.min()) < 0.0 or float(y.max()) > 1.0:
        raise ValueError("fit_gate: 标签必须是 0/1")

    w = torch.tensor([cfg.gate_w[n] for n in POTENTIAL_NAMES], dtype=torch.float32, requires_grad=True)
    b = torch.tensor(float(cfg.gate_bias), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    hist: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        logits = x @ w + b
        loss = loss_fn(logits, y) + l2 * (w * w).sum()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    with torch.no_grad():
        probs = torch.sigmoid(x @ w + b)
        pred = (probs > 0.5).float()
        acc = float((pred == y).float().mean())
        bce = float(loss_fn(x @ w + b, y))
    info = {
        "samples": len(samples),
        "epochs": epochs,
        "final_loss": hist[-1],
        "bce": bce,
        "accuracy": acc,
        "pos_rate": float(y.mean()),
    }
    return ({n: float(v) for n, v in zip(POTENTIAL_NAMES, w.detach())}, float(b.detach()), info)


def fit_router(
    samples: Sequence[tuple[dict[str, float], int]],
    *,
    init: HubConfig | None = None,
    epochs: int = 600,
    lr: float = 0.35,
    l2: float = 1e-3,
) -> tuple[dict[int, dict[str, float]], dict[int, float], dict]:
    """用"哪一级真的修好了"的标签拟合层级路由头（多项逻辑回归，也是凸的）。

    Args:
        samples: [(φ, level)]，level ∈ {1,2,3}。
        init: 初始配置（默认先验）。
        epochs: 轮数。
        lr: 学习率。
        l2: L2 正则。

    Returns:
        (权重, 偏置, 训练信息)。

    Raises:
        ValueError: 样本为空或 level 不在 1..3。
    """
    if not samples:
        raise ValueError("fit_router: 样本为空")
    levels = (1, 2, 3)
    for _, lvl in samples:
        if lvl not in levels:
            raise ValueError(f"fit_router: level={lvl} 不在 {levels}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("需要 torch 才能训练路由头") from exc

    cfg = init or HubConfig()
    x = torch.tensor(
        [[_clamp(_safe_float(phi.get(n, 0.0)), 0.0, 1.0) for n in POTENTIAL_NAMES] for phi, _ in samples],
        dtype=torch.float32,
    )
    y = torch.tensor([levels.index(lbl) for _, lbl in samples], dtype=torch.long)
    w = torch.tensor(
        [[cfg.router_w[lvl][n] for n in POTENTIAL_NAMES] for lvl in levels],
        dtype=torch.float32, requires_grad=True,
    )
    b = torch.tensor([cfg.router_bias[lvl] for lvl in levels], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    hist: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(x @ w.t() + b, y) + l2 * (w * w).sum()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    with torch.no_grad():
        acc = float((torch.argmax(x @ w.t() + b, dim=1) == y).float().mean())
    info = {"samples": len(samples), "epochs": epochs, "final_loss": hist[-1], "accuracy": acc}
    wd = {lvl: {n: float(w[i, j].detach()) for j, n in enumerate(POTENTIAL_NAMES)}
          for i, lvl in enumerate(levels)}
    bd = {lvl: float(b[i].detach()) for i, lvl in enumerate(levels)}
    return wd, bd, info
