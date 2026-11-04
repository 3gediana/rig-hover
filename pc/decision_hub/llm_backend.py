"""llm_backend.py -- 上层（Level 2 / Level 3）后端与上下行消息契约。

这个模块只做两件事，两件事都不依赖 hub.py：

1. **消息契约**：``build_escalation_request()`` 组装机器人发给上层的 JSON，
   ``parse_instruction()`` 校验上层回来的 JSON。校验是硬的——上层是任意的
   LLM/agent，它的输出必须当成不可信输入（FIRMWARE_CHANGES.md §2 定义了
   同一份契约的固件侧版本）。
2. **后端**：Level 2 = PC 上的 agent，Level 3 = 云端 LLM。两种都实现真调和
   mock 两条路：

   ======================  ==================================================
   类                      用途
   ======================  ==================================================
   MockAgentBackend        L2 的确定性 mock：按 request 里的场景字段推出指令
   MockCloudBackend        L3 的确定性 mock，同上但更贵更慢
   CloudLLMBackend         L3 真调 DeepSeek（urllib，无新依赖），无 key 即不可用
   HttpAgentBackend        L2 真调一个本地/远程 HTTP agent（同一份契约）
   ======================  ==================================================

   mock 不是"返回期望答案"的假实现：它只读 request 里它有权看到的字段
   （``obs``/``potentials``），用几何和规则算出动作，然后走**和真 LLM 完全
   一样的** ``parse_instruction()`` 校验路径。所以 mock 通 ≠ 真 LLM 通，但
   mock 能证明链路、超时、降级、校验都活着。

关键约定（与 design.md §6 一致）：

* ``level`` 只描述"这条指令该由谁执行"，不描述"谁产生的"。
* ``prog`` 是**动作原语序列**，固件在本地按 50Hz 播放（不依赖网络）。
* ``ttl_ms`` 是上层的预算，超时由机器人侧 unilateral 放弃，不需要上层配合。
* ``fallback`` 是上层自己给的兜底程序；机器人侧还有一层兜底（L1 技能），
  所以 ``fallback`` 缺失不是错误。

用法::

    be = CloudLLMBackend()          # available() 为 False 时 hub 会跳过它
    ins = be.ask(request, timeout_s=2.5)
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# 契约常量
# --------------------------------------------------------------------------

SCHEMA = "esc/1"
CONTRACT_VERSION = 1

#: 一次升级请求最多等多久的硬上限（机器人侧，毫秒）。上层给的 ttl 不能超。
MAX_TTL_MS = 5000
#: 一个动作程序最多几步。防止上层返回 200 步把本地队列塞满。
MAX_PROG_STEPS = 16

EMOTE_NONE = 0
EMOTE_NOD = 1
EMOTE_THINK = 4
EMOTE_ASK = 5
EMOTE_NAMES = {
    0: "none",
    1: "nod",
    2: "happy",
    3: "sad",
    4: "think",
    5: "ask",
    6: "surprise",
    7: "sleep",
}

#: op -> 允许的字段与取值范围。原语本身就是 CfC 的输出空间
#: （head_des / vyaw_des / vx_des / mode / emote），所以上层指令天然落在
#: CfC 的动作空间里，不需要第二套动作定义。
OP_SPEC: dict[str, dict[str, tuple[float, float]]] = {
    "head": {"deg": (-45.0, 45.0), "ms": (0.0, 4000.0)},
    "yaw": {"dps": (-90.0, 90.0), "ms": (0.0, 4000.0)},
    "vx": {"mps": (-0.30, 0.30), "ms": (0.0, 4000.0)},
    "emote": {"id": (0.0, 7.0), "ms": (0.0, 8000.0)},
    "hold": {"ms": (0.0, 8000.0)},
    "stop": {"ms": (0.0, 8000.0)},
    "track": {"ms": (0.0, 1000.0)},
    "say": {"ms": (0.0, 8000.0)},
}

#: 动作空间的硬边界，和 control_arbiter.c 的 cfg 默认值一致。
LIM_HEAD_DEG = 45.0
LIM_VYAW_DPS = 90.0
LIM_VX_MPS = 0.30


class InvalidInstruction(ValueError):
    """上层回来的 JSON 不合法（未知 op / 超范围 / 缺字段 / 非数）。"""


class BackendUnavailable(RuntimeError):
    """这一级当前不可用（没 key、没配置、健康检查失败）。"""


class BackendTimeout(RuntimeError):
    """这一级在预算内没有返回。"""


class BackendError(RuntimeError):
    """这一级返回了错误（HTTP 5xx、解析失败、内部异常）。"""


# --------------------------------------------------------------------------
# 上下行消息
# --------------------------------------------------------------------------


def _f(x: Any, nd: int = 4) -> float:
    """JSON 安全的 float：去掉 NaN/Inf（allow_nan=False 时它们会炸）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return round(v, nd)


def build_escalation_request(
    *,
    esc_id: int,
    level: int,
    p_esc: float,
    phi: dict[str, float],
    contributions: list[tuple[str, float]],
    obs: dict[str, Any],
    ttl_ms: int,
    fallback: list[dict] | None = None,
    capture_hint: str | None = None,
) -> dict:
    """机器人 -> 上层的一次升级请求（GET /esc 的响应体就是它）。

    Args:
        esc_id: 机器人侧单调递增的升级编号，回复必须原样带回。
        level: 机器人请求的层级（路由器给出的最省资源且够用的那一级）。
        p_esc: 升级头的概率输出，[0,1]。
        phi: 12 个势函数的取值，给上层和离线训练用。
        contributions: [(势函数名, 对 logit 的贡献)]，已按绝对值降序，取前 5 条。
        obs: 观测快照（只放上层真的需要的字段，不放 64 维特征——那个太大，
            上层要看就自己 GET /capture 取帧）。
        ttl_ms: 机器人愿意等多久。超时机器人自己放弃，不需要回复。
        fallback: 上层自己建议的兜底程序，可为 None。
        capture_hint: 提示上层去哪里取当前帧。

    Returns:
        可直接 ``json.dumps(..., allow_nan=False)`` 的字典。
    """
    return {
        "schema": SCHEMA,
        "v": CONTRACT_VERSION,
        "id": int(esc_id),
        "level": int(level),
        "p_esc": _f(p_esc),
        "ttl_ms": int(ttl_ms),
        "phi": {k: _f(v) for k, v in phi.items()},
        "why": [{"k": k, "c": _f(c)} for k, c in contributions[:5]],
        "obs": {k: (_f(v) if isinstance(v, (int, float)) else v) for k, v in obs.items()},
        "fallback": fallback or [],
        "capture": capture_hint or "",
    }


def _need_dict(obj: Any, what: str) -> dict:
    if not isinstance(obj, dict):
        raise InvalidInstruction(f"{what} 必须是 JSON 对象，实际 {type(obj).__name__}")
    return obj


def _num(d: dict, key: str, lo: float, hi: float, where: str) -> float:
    if key not in d:
        raise InvalidInstruction(f"{where} 缺字段 '{key}'")
    try:
        v = float(d[key])
    except (TypeError, ValueError) as exc:
        raise InvalidInstruction(f"{where}.{key} 不是数：{d[key]!r}") from exc
    if not math.isfinite(v):
        raise InvalidInstruction(f"{where}.{key} 不是有限数：{d[key]!r}")
    if v < lo or v > hi:
        raise InvalidInstruction(f"{where}.{key}={v} 越界 [{lo}, {hi}]")
    return v


def parse_prog(prog: Any, where: str = "prog") -> list[dict]:
    """校验并规范化一个动作程序。

    Args:
        prog: 上层给的列表，每项是 ``{"op": str, ...}``。
        where: 报错时用的字段路径。

    Returns:
        规范化后的列表，每项含 ``op`` 和该 op 允许的字段（float）。

    Raises:
        InvalidInstruction: 不是列表、太长、op 未知、字段越界。
    """
    if prog is None:
        return []
    if not isinstance(prog, list):
        raise InvalidInstruction(f"{where} 必须是数组，实际 {type(prog).__name__}")
    if len(prog) > MAX_PROG_STEPS:
        raise InvalidInstruction(f"{where} 有 {len(prog)} 步，上限 {MAX_PROG_STEPS}")
    out: list[dict] = []
    for i, step in enumerate(prog):
        s = _need_dict(step, f"{where}[{i}]")
        op = s.get("op")
        if not isinstance(op, str) or op not in OP_SPEC:
            raise InvalidInstruction(
                f"{where}[{i}].op={op!r} 未知，允许 {sorted(OP_SPEC)}"
            )
        spec = OP_SPEC[op]
        item: dict[str, Any] = {"op": op}
        for key, (lo, hi) in spec.items():
            item[key] = _num(s, key, lo, hi, f"{where}[{i}]({op})")
        if op == "yaw":
            item["compensate"] = bool(s.get("compensate", True))
        out.append(item)
    return out


@dataclass
class Instruction:
    """一条通过校验、可以被机器人执行的上层指令。"""

    level: int
    intent: str
    prog: list[dict]
    ttl_ms: int = 1500
    fallback: list[dict] = field(default_factory=list)
    args: dict = field(default_factory=dict)
    say: str | None = None
    confidence: float = 0.0
    source: str = ""
    esc_id: int = -1
    raw: str = ""

    def as_dict(self) -> dict:
        """序列化成交付/日志用的字典（不含 raw 以免日志爆炸）。"""
        d = asdict(self)
        d.pop("raw", None)
        return d

    def describe(self) -> str:
        """人读的一行摘要，demo/test 打印用。"""
        ops = "+".join(s["op"] for s in self.prog) or "-"
        return f"L{self.level} {self.intent} [{ops}] ttl={self.ttl_ms}ms src={self.source}"


def parse_instruction(obj: Any, *, source: str = "", raw: str = "") -> Instruction:
    """校验上层的回复。

    Args:
        obj: 已 json 解析的对象。
        source: 后端名，进日志。
        raw: 原始文本，出错时截断进异常信息。

    Returns:
        ``Instruction``。

    Raises:
        InvalidInstruction: 任何字段非法。宁可拒绝一条指令，也不能把
            未校验的动作灌进 LQR——这是本模块最重要的一条不变量。
    """
    d = _need_dict(obj, "instruction")

    # id：允许缺失（mock 场景），出现就必须是整数；不匹配由 hub 侧判定。
    esc_id = -1
    if "id" in d:
        try:
            esc_id = int(d["id"])
        except (TypeError, ValueError) as exc:
            raise InvalidInstruction(f"id 不是整数：{d['id']!r}") from exc

    level = 0
    if "level" in d:
        try:
            level = int(d["level"])
        except (TypeError, ValueError) as exc:
            raise InvalidInstruction(f"level 不是整数：{d['level']!r}") from exc
        if level < 1 or level > 3:
            raise InvalidInstruction(f"level={level} 必须是 1/2/3（0 由机器人自己产生）")

    intent = d.get("intent", "unknown")
    if not isinstance(intent, str):
        raise InvalidInstruction(f"intent 必须是字符串，实际 {type(intent).__name__}")
    intent = intent.strip()[:64]
    if not intent:
        raise InvalidInstruction("intent 为空")

    ttl_ms = 1500
    if "ttl_ms" in d:
        ttl_ms = int(_num(d, "ttl_ms", 1.0, float(MAX_TTL_MS), "instruction"))

    prog = parse_prog(d.get("prog"), "prog")
    fallback = parse_prog(d.get("fallback"), "fallback")

    conf = 0.0
    if "confidence" in d:
        conf = _num(d, "confidence", 0.0, 1.0, "instruction")

    say = d.get("say")
    if say is not None and not isinstance(say, str):
        raise InvalidInstruction(f"say 必须是字符串或 null，实际 {type(say).__name__}")
    if isinstance(say, str):
        say = say.strip()[:200] or None

    args = d.get("args", {})
    if args is not None and not isinstance(args, dict):
        raise InvalidInstruction(f"args 必须是对象，实际 {type(args).__name__}")

    return Instruction(
        level=level,
        intent=intent,
        prog=prog,
        ttl_ms=ttl_ms,
        fallback=fallback,
        args=args or {},
        say=say,
        confidence=conf,
        source=source,
        esc_id=esc_id,
        raw=raw[:2000],
    )


def make_instruction(
    *, level: int, intent: str, prog: list[dict], ttl_ms: int = 1500,
    say: str | None = None, source: str = "local", esc_id: int = -1,
    confidence: float = 1.0, args: dict | None = None,
) -> Instruction:
    """本地构造一条指令（L1 技能库用），走和上层一样的校验路径。"""
    return parse_instruction(
        {
            "id": esc_id,
            "level": level,
            "intent": intent,
            "prog": prog,
            "ttl_ms": ttl_ms,
            "say": say,
            "confidence": confidence,
            "args": args or {},
        },
        source=source,
    )


# --------------------------------------------------------------------------
# 后端基类
# --------------------------------------------------------------------------


class Backend:
    """上层后端的接口。``ask`` 是阻塞的，hub 会把它丢到线程池里。"""

    name: str = "backend"
    level: int = 2

    def available(self) -> bool:
        """这一级现在能不能用。False 时 hub 不会把请求路由过来。"""
        return True

    def ask(self, request: dict, timeout_s: float) -> Instruction:
        """把请求转成一个经过校验的指令。

        Args:
            request: ``build_escalation_request`` 的产物。
            timeout_s: 这一级的时间预算（秒）。

        Returns:
            ``Instruction``。

        Raises:
            BackendUnavailable: 不可用。
            BackendTimeout: 超预算。
            BackendError: 后端报错。
            InvalidInstruction: 输出不合法。
        """
        raise NotImplementedError


# --------------------------------------------------------------------------
# Level 2 / Level 3 的 mock：确定性脚本 agent
# --------------------------------------------------------------------------


#: mock 的"思考耗时"（秒）。这是**建模值**不是实测值，demo 的延迟表里
#: 一律标 modelled，不允许冒充实测。
MOCK_LATENCY = {2: 0.085, 3: 0.850}


class _ScriptedBackend(Backend):
    """按 request.obs 里的场景字段推一条指令的确定性 agent。

    fail_mode:
        ``none``     正常
        ``timeout``  sleep 超过调用方预算，然后抛 BackendTimeout
        ``down``     抛 BackendUnavailable
        ``error``    抛 BackendError
        ``garbage``  返回非法 JSON（校验层必须拒）
    """

    def __init__(
        self,
        level: int,
        name: str,
        latency_s: float | None = None,
        fail_mode: str = "none",
        jitter_s: float = 0.0,
    ):
        self.level = int(level)
        self.name = name
        self.latency_s = MOCK_LATENCY.get(self.level, 0.1) if latency_s is None else latency_s
        self.fail_mode = fail_mode
        self.jitter_s = jitter_s
        self.calls = 0

    def available(self) -> bool:
        return self.fail_mode != "down"

    def ask(self, request: dict, timeout_s: float) -> Instruction:
        self.calls += 1
        delay = self.latency_s
        if self.jitter_s > 0.0:
            # 确定性抖动：不用 random，测试可复现
            delay += self.jitter_s * ((self.calls * 37) % 11) / 10.0
        if self.fail_mode == "timeout":
            time.sleep(min(timeout_s + 0.05, 2.0))
            raise BackendTimeout(f"{self.name}: 预算 {timeout_s * 1000:.0f}ms 内没回来")
        if delay > 0.0:
            time.sleep(min(delay, timeout_s) if self.fail_mode != "none" else delay)
        if self.fail_mode == "down":
            raise BackendUnavailable(f"{self.name}: 不可用")
        if self.fail_mode == "error":
            raise BackendError(f"{self.name}: 上游 503")
        if self.fail_mode == "garbage":
            return parse_instruction(
                {"level": self.level, "intent": "x", "prog": [{"op": "teleport"}]},
                source=self.name,
                raw="{not json",
            )
        obj = self._respond(request)
        return parse_instruction(obj, source=self.name, raw=json.dumps(obj, ensure_ascii=False))

    # -- 场景推理（只读 request 里上层真的能看到的字段）--------------------
    def _respond(self, request: dict) -> dict:
        obs = request.get("obs") or {}
        phi = request.get("phi") or {}
        esc_id = request.get("id", -1)
        seen = float(obs.get("seen", 0.0)) >= 0.5
        conf = float(obs.get("person_conf", 0.0))
        theta = float(obs.get("theta_img", 0.0))
        speech = float(obs.get("speech", 0.0)) > 0.5
        pointing = float(obs.get("pointing", 0.0)) > 0.5
        obj_held = float(obs.get("object_held", 0.0)) > 0.5
        clauses = int(obs.get("instr_clauses", 0) or 0)
        lookup = float(obs.get("lookup", 0.0)) > 0.5
        point_dir = float(obs.get("point_dir_deg", 0.0))

        base = {"id": esc_id, "level": self.level, "ttl_ms": 1500}

        # (a) 说话了：两个动作需要规划（多步程序）；一个动作只需要听懂（单步）
        if speech and clauses >= 2:
            return dict(
                base,
                intent="follow_spoken_plan",
                confidence=0.72,
                say="先看你，再点头。",
                prog=[{"op": "head", "deg": 0.0, "ms": 350},
                      {"op": "emote", "id": 1, "ms": 500},
                      {"op": "track", "ms": 0}],
                args={"plan": ["look_at_speaker", "acknowledge"]},
            )
        if speech:
            return dict(
                base,
                intent="acknowledge_speech",
                confidence=0.6,
                say="听到了。",
                prog=[{"op": "head", "deg": 0.0, "ms": 300},
                      {"op": "emote", "id": EMOTE_NOD, "ms": 400},
                      {"op": "track", "ms": 0}],
                args={"plan": ["acknowledge"]},
            )
        # (b) 指着东西 / 举着东西 -> 看过去（L2 有帧，不需要云）
        if (pointing or obj_held) and seen:
            deg = max(-40.0, min(40.0, theta + point_dir))
            return dict(
                base,
                intent="look_at_target",
                confidence=0.66,
                prog=[{"op": "head", "deg": round(deg, 1), "ms": 300},
                      {"op": "hold", "ms": 250}],
                args={"target": "pointed_object", "yaw_delta_deg": round(deg, 1)},
            )
        # (c) 看不清 -> 上层也只能说"我看不清"，给一个稳住的动作
        if phi.get("u_conf", 0.0) > 0.5 or not seen:
            return dict(
                base,
                intent="cannot_see",
                confidence=0.3,
                prog=[{"op": "hold", "ms": 800}, {"op": "emote", "id": EMOTE_ASK, "ms": 600}],
                say="画面太暗，我没把握。",
            )
        # (d) 有疑问 -> 保守后退观察
        if phi.get("s_approach", 0.0) > 0.4:
            return dict(
                base,
                intent="back_off_and_look",
                confidence=0.5,
                prog=[{"op": "vx", "mps": -0.06, "ms": 500},
                      {"op": "hold", "ms": 400}],
            )
        # (e) 兜底：什么都不做，交回 L0
        return dict(base, intent="noop", confidence=0.2, prog=[{"op": "track", "ms": 0}])

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} L{self.level} fail={self.fail_mode}>"


class MockAgentBackend(_ScriptedBackend):
    """Level 2：PC 上的 agent（本机进程/本机 HTTP 服务）。"""

    def __init__(self, latency_s: float | None = None, fail_mode: str = "none",
                 jitter_s: float = 0.0, name: str = "mock-agent"):
        super().__init__(2, name, latency_s=latency_s, fail_mode=fail_mode, jitter_s=jitter_s)


class MockCloudBackend(_ScriptedBackend):
    """Level 3：云端 LLM。"""

    def __init__(self, latency_s: float | None = None, fail_mode: str = "none",
                 jitter_s: float = 0.0, name: str = "mock-cloud"):
        super().__init__(3, name, latency_s=latency_s, fail_mode=fail_mode, jitter_s=jitter_s)


# --------------------------------------------------------------------------
# Level 3 真调：DeepSeek（OpenAI 兼容 /chat/completions）
# --------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """你是 RIG-Hover 桌面机器人的上层决策者。机器人只有四个动作原语：
head(deg -45..45, ms 0..4000)、yaw(dps -90..90, ms 0..4000)、vx(mps -0.30..0.30, ms 0..4000)、
emote(id 0..7, ms 0..8000)、hold(ms)、stop(ms)、track(ms)、say(ms)。
你收到一个 escalation 请求，必须只输出一个 JSON 对象，字段：
{"id": <原样带回>, "level": 2 或 3, "intent": "<短语>", "prog": [ ... ],
 "ttl_ms": <100..5000>, "say": <字符串或 null>, "confidence": <0..1>,
 "args": {<可选的解释字段>}}
规则：
1. prog 最多 8 步，只用上面列出的 op 和字段名，不要发明新的 op。
2. 动作要保守：不确定就 hold 或 track，不要让人形机器人朝人前进。
3. 你看不到图。需要看的请把 intents 写成需要取帧的描述，不要编造视觉细节。
4. 只输出 JSON，不要 markdown 代码块，不要解释。"""


def load_api_key() -> str:
    """按 环境变量 -> 仓库根 .env -> 当前目录 .env 的顺序找 key。

    ../../.env 是 D:\\robot_work\\.env（CONTRACT §0 的工作根）。

    Returns:
        key 字符串；找不到返回空串（调用方据此判断不可用）。
    """
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    here = Path(__file__).resolve()
    for cand in (here.parent / ".env", here.parent.parent.parent / ".env",
                 Path.cwd() / ".env"):
        try:
            if not cand.is_file():
                continue
            for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, val = line.partition("=")
                if k.strip() in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
                    val = val.strip().strip('"').strip("'")
                    if val:
                        return val
        except OSError:
            continue
    return ""


class CloudLLMBackend(Backend):
    """Level 3：真调 DeepSeek。没有 key 时 ``available()`` 为 False。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        name: str = "deepseek",
    ):
        self.level = 3
        self.name = name
        self.api_key = api_key if api_key is not None else load_api_key()
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.calls = 0
        self.last_error = ""

    def available(self) -> bool:
        return bool(self.api_key)

    def ask(self, request: dict, timeout_s: float) -> Instruction:
        if not self.api_key:
            raise BackendUnavailable("没有 DEEPSEEK_API_KEY")
        self.calls += 1
        body = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            self.last_error = f"HTTP {exc.code}"
            if exc.code in (401, 403, 404):
                raise BackendUnavailable(f"{self.name}: HTTP {exc.code}") from exc
            raise BackendError(f"{self.name}: HTTP {exc.code}") from exc
        except TimeoutError as exc:
            self.last_error = "timeout"
            raise BackendTimeout(f"{self.name}: {timeout_s:.1f}s 超时") from exc
        except urllib.error.URLError as exc:
            self.last_error = str(exc.reason)
            raise BackendError(f"{self.name}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise BackendError(f"{self.name}: 响应不是 JSON") from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"{self.name}: 响应结构不对") from exc
        try:
            obj = json.loads(_strip_fence(text))
        except json.JSONDecodeError as exc:
            raise InvalidInstruction(f"{self.name}: 返回的不是 JSON：{text[:120]!r}") from exc
        return parse_instruction(obj, source=self.name, raw=text)


class HttpAgentBackend(Backend):
    """Level 2：把同一份契约发给一个 HTTP agent（本机或局域网）。

    没配置 url 时 ``available()`` 为 False，hub 会自动跳过这一级——这就是
    "云端/agent 不可用 -> 降级" 的入口。
    """

    def __init__(self, url: str | None = None, name: str = "http-agent", level: int = 2):
        self.level = level
        self.name = name
        self.url = (url or os.environ.get("ROBOT_AGENT_URL") or "").strip()
        self.calls = 0

    def available(self) -> bool:
        return bool(self.url)

    def ask(self, request: dict, timeout_s: float) -> Instruction:
        if not self.url:
            raise BackendUnavailable(f"{self.name}: 没有配置 URL")
        self.calls += 1
        data = json.dumps(request, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                text = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise BackendError(f"{self.name}: HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise BackendTimeout(f"{self.name}: {timeout_s:.1f}s 超时") from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"{self.name}: {exc.reason}") from exc
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidInstruction(f"{self.name}: 不是 JSON：{text[:120]!r}") from exc
        return parse_instruction(obj, source=self.name, raw=text)


def _strip_fence(text: str) -> str:
    """剥掉 ```json ... ``` 围栏。模型的常见违约，剥比报错划算。"""
    t = text.strip()
    if not t.startswith("```"):
        return t
    t = t.split("\n", 1)[1] if "\n" in t else t
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def probe_cloud_transport(base_url: str | None = None, timeout_s: float = 3.0) -> dict:
    """测一次到云端端点的 TLS+HTTP 往返（不带 key，预期 401）。

    这是 L3 的**传输延迟下界**实测值：它包含 DNS + TCP + TLS + 请求/响应，
    不含模型思考。没有网络时返回 ``{"ok": False, ...}``，不抛异常。

    Args:
        base_url: 端点，默认 DeepSeek 官方。
        timeout_s: 超时。

    Returns:
        ``{"ok": bool, "ms": float, "status": int|None, "note": str}``。
    """
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps({"model": DEFAULT_MODEL, "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
        return {"ok": True, "ms": (time.perf_counter() - t0) * 1000.0,
                "status": 200, "note": "意外：无 key 也 200"}
    except urllib.error.HTTPError as exc:
        return {"ok": True, "ms": (time.perf_counter() - t0) * 1000.0,
                "status": exc.code, "note": "无 key 探测（预期 401）"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "ms": (time.perf_counter() - t0) * 1000.0,
                "status": None, "note": f"不可达：{exc}"}
