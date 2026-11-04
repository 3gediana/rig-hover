#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""local_agent.py -- 真实 L2 backend：把 DecisionHub 的升级请求交给 agent_bridge。

`llm_backend.HttpAgentBackend` 已经能把 esc/1 请求 POST 给一个任意 URL，但
agent_bridge 说的是另一套协议（``POST /agent/task {"intent", "level"}`` ->
``{"ok", "action", "detail", ...}``），而且它的返回值是**真实执行的输出**，
不是可以直接执行的机器人动作程序。本模块只做翻译和失败处理：

1. **下行**：esc/1 请求 -> 自然语言意图。意图由归因（``request["why"][0]["k"]``）
   的风险族决定，**不直接指定 action**——让 agent 自己的 planner（有 key 用
   LLM，没 key 用规则）挑动作，才是在验证"PC agent 真的会干活"。
2. **上行**：agent 的执行结果 -> 通过 ``parse_instruction`` 校验的 ``Instruction``。
   真实 detail 原文进 ``args.detail``，人读摘要进 ``say``；``prog`` 是一段合法的
   动作原语程序（机器人的动作空间里没有"显示文本"这个自由度，把 detail 塞进
   prog 就是编造）。
3. **失败**：建连失败 / 建连超时 / 等回包超时 / HTTP 错误 / 非 JSON / ``ok=0``
   一律映射成 ``BackendUnavailable`` / ``BackendTimeout`` / ``BackendError``，
   交给 hub 已有的降级路径（design.md §5）。这里**绝不**吞异常返回一条假指令：
   一条未经验证的指令比没有指令更危险。

``available()`` 每拍都会被 hub 调用（``decide()`` 里），所以它只读一个缓存布尔，
网络探测在后台线程和维护性的 ``probe()`` 里做。L0 的 50Hz 循环不会被 HTTP 阻塞。

用法::

    be = LocalAgentBackend("http://127.0.0.1:8899")
    hub = DecisionHub(backends={2: be})
    ...
    be.close()
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from hub import POTENTIALS, percentile
from llm_backend import (
    EMOTE_NOD,
    EMOTE_THINK,
    Backend,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
    Instruction,
    parse_instruction,
)

__all__ = [
    "ACTION_PROG",
    "DEFAULT_BASE_URL",
    "FAMILY_INTENT",
    "LocalAgentBackend",
]

#: agent_bridge 的默认监听地址（server.py --port 8899）。
DEFAULT_BASE_URL = "http://127.0.0.1:8899"
HEALTH_PATH = "/agent/health"
TASK_PATH = "/agent/task"

#: socket 超时 = 预算 × 这个比例。留 10% 给"回包正在路上"：
#: 让失败发生在 hub 自己的 ttl 判定之前，失败原因更准，线程也不会滞留在
#: 一个已经超时的连接上（hub 的 future.cancel() 杀不掉已阻塞的 socket）。
DEFAULT_TIMEOUT_RATIO = 0.9
MIN_SOCKET_TIMEOUT_S = 0.02

#: **建连**超时（秒），和"等回包"的预算分开。实测（design.md §5.1）：
#: 服务进程被杀之后，Windows 不会对发往该端口的 SYN 回 RST，而是静默丢弃，
#: 于是 urllib 的单超时会把"掉线"误判成"超时"——TTL=900ms 时要等 810ms 才
#: 回退。把建连单独限制在 150ms，掉线回退就从 810ms 降到 150ms。
#: 同机部署用 0.15s 足够；如果 agent 在局域网远端，按网络 RTT 调大。
DEFAULT_CONNECT_TIMEOUT_S = 0.15

#: 健康探测超时 / 间隔（秒）。只在构造和后台线程里用，绝不进 tick。
DEFAULT_HEALTH_TIMEOUT_S = 0.25
DEFAULT_HEALTH_TTL_S = 1.0


# --------------------------------------------------------------------------
# 翻译表
# --------------------------------------------------------------------------

#: 势函数 -> 风险族（与 hub.POTENTIALS 同源，不复制一份手工表）。
_FAMILY: dict[str, str] = {p.name: p.family for p in POTENTIALS}

#: 风险族 -> 交给 PC agent 的自然语言意图。
#:
#: 每一句都刻意命中 agent_bridge 规则 planner 的关键词，从而落到一个**真会
#: 产生副作用**的动作上；有 DEEPSEEK_API_KEY 时则交给 LLM planner 自己判。
#: 右侧注释是规则 planner 的实测路由（test_real_backend.py 逐条断言）。
FAMILY_INTENT: dict[str, str] = {
    # "看看" -> screenshot：写 screen.png。实测 ~360ms，是全表最慢的动作。
    "blind": "看看 PC 屏幕上现在是什么情况。",
    # 无关键词 -> system_status：真主机名 / 平台 / 磁盘 / 时间。
    "unknown": "报告 PC 的系统状态。",
    # "记" -> note：往 robot_notes.log 追加一行（唯一写文件的动作）。
    "language": "记下这次升级：机器人听到声音但没听懂。",
    "planning": "记下这次升级：机器人需要多步规划，本地做不了。",
    # "几点" -> clock：真时间。
    "context": "现在几点了？机器人要跟 PC 校时。",
    # "磁盘"/"空间" -> disk：真剩余空间。
    "stakes": "PC 上各磁盘还剩多少空间？",
}

#: 归因缺失或风险族未知时的意图（规则 planner 会落到 system_status）。
DEFAULT_INTENT = "报告 PC 的系统状态。"

#: 动作 -> 机器人能执行的动作原语程序（按 50Hz 本地播放，最后一步交还 L0）。
#:
#: detail 不进 prog：动作空间里只有 head/yaw/vx/mode/emote，"把查到的磁盘空间
#: 变成一个动作"没有对应的自由度，硬编就是假执行。机器人在拿到结果后的动作
#: 是一次确认（点头/思考），内容通过 say 和 args 交给上层。
ACTION_PROG: dict[str, list[dict]] = {
    "clock": [{"op": "emote", "id": EMOTE_NOD, "ms": 350},
              {"op": "track", "ms": 0}],
    "system_status": [{"op": "emote", "id": EMOTE_THINK, "ms": 400},
                      {"op": "track", "ms": 0}],
    "disk": [{"op": "emote", "id": EMOTE_THINK, "ms": 400},
             {"op": "track", "ms": 0}],
    "note": [{"op": "emote", "id": EMOTE_NOD, "ms": 300},
             {"op": "track", "ms": 0}],
    "screenshot": [{"op": "head", "deg": 0.0, "ms": 250},
                   {"op": "track", "ms": 0}],
}

#: 未知动作（bridge 未来扩了白名单）：仍然给一条合法、保守的程序。
_GENERIC_PROG: list[dict] = [{"op": "hold", "ms": 300}, {"op": "track", "ms": 0}]

#: 动作 -> say 模板（人读摘要，真实 detail 原样带进去）。
_SAY: dict[str, str] = {
    "clock": "PC 时间 {detail}",
    "system_status": "PC 状态 {detail}",
    "disk": "磁盘 {detail}",
    "note": "已记下：{detail}",
    "screenshot": "已截图：{detail}",
}

#: planner 来源 -> 指令置信度。规则匹配不理解语义，给不了高于 0.6 的置信度。
_PLANNER_CONF: dict[str, float] = {"llm": 0.75, "rule": 0.55, "explicit": 0.65}


def _finite(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


# --------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------


class LocalAgentBackend(Backend):
    """Level 2：通过 HTTP 调 ``agent_bridge`` 真实执行，结果翻回 esc/1 指令。

    Args:
        base_url: agent_bridge 根地址（默认 ``http://127.0.0.1:8899``）。
        task_path: 任务端点路径；改成别的路径可以测 404 -> ``BackendError``。
        health_path: 健康端点路径。
        name: 后端名，进 hub 的 attempt 记录。
        action_hint: 强制指定动作，跳过 agent 自己的 planner。默认 None：
            让 agent 按意图自己选。只有调用方确知要哪个动作时才该用。
        timeout_ratio: socket 超时占预算的比例，见 ``DEFAULT_TIMEOUT_RATIO``。
        connect_timeout_s: 建连超时（秒），与等回包的预算分开；见
            ``DEFAULT_CONNECT_TIMEOUT_S``。掉线检测靠它。
        health_timeout_s: 单次健康探测超时。
        health_ttl_s: 后台健康探测间隔；``available()`` 的缓存最多陈旧这么久。
        fail_threshold: 连续多少次探测失败才把这一级判为不可用。1 次成功立即恢复。
            设成 1 就是"每次失败立刻下线"，实测会在主机抖动时误判（§5.1）。
        start_health_thread: 是否启动后台探测线程。关掉后只能手动 ``probe()``。
        probe_on_start: 构造时同步探测一次（有界，最多 health_timeout_s）。
    """

    level = 2

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        task_path: str = TASK_PATH,
        health_path: str = HEALTH_PATH,
        name: str = "local-agent",
        action_hint: str | None = None,
        timeout_ratio: float = DEFAULT_TIMEOUT_RATIO,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
        health_ttl_s: float = DEFAULT_HEALTH_TTL_S,
        fail_threshold: int = 2,
        start_health_thread: bool = True,
        probe_on_start: bool = True,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.task_url = self.base_url + task_path
        self.health_url = self.base_url + health_path
        parts = urlsplit(self.base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"base_url 必须是 http(s)://host[:port]：{base_url!r}")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.task_path = task_path
        self.action_hint = action_hint
        self.timeout_ratio = min(1.0, max(0.1, float(timeout_ratio)))
        self.connect_timeout_s = max(0.02, float(connect_timeout_s))
        self.health_timeout_s = max(0.01, float(health_timeout_s))
        self.health_ttl_s = max(0.05, float(health_ttl_s))

        # available() 只读这个字段：hub 每拍问一次，不能在这里做网络 IO。
        self._healthy = False
        # 健康状态的两道保护（实测出来的，见 design.md §5.1）：
        #  1. 单次探测超时不能把 L2 从候选里踢掉——主机在冷启动/杀毒扫描时会
        #     偶发 250ms 级抖动；连续 fail_threshold 次失败才判不可用，一次成功
        #     立刻恢复。
        #  2. 探测可能重叠（后台线程 + wait_healthy + 手动 probe）。用"开始时间"
        #     做单调闸门，晚到的旧结果不允许覆盖新结果。
        self._health_stamp = 0.0
        self._health_fail = 0
        self._health_lock = threading.Lock()
        self.fail_threshold = max(1, int(fail_threshold))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # 统计（demo/test 打印；不算进 hub 的 Stats）
        self.calls = 0
        self.ok = 0
        self.timeouts = 0
        self.errors = 0
        self.unavailable = 0
        self.last_error = ""
        self.last_raw = ""
        self.last_task: dict[str, Any] = {}
        self.last_result: dict[str, Any] = {}
        self.roundtrip_ms: list[float] = []
        self.probe_ms: list[float] = []
        self.actions: list[str] = []
        self.llm = False

        if probe_on_start:
            self.probe(self.health_timeout_s)
        if start_health_thread:
            self._thread = threading.Thread(
                target=self._health_loop, name=f"{self.name}-health", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------ 生命周期
    def close(self, timeout_s: float = 1.0) -> None:
        """停掉后台探测线程。可重复调用。"""
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=timeout_s)
        self._thread = None

    def __enter__(self) -> "LocalAgentBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ 可用性
    def available(self) -> bool:
        """这一级当前能不能用。**非阻塞**：只读后台探测的缓存结果。"""
        return self._healthy

    def probe(self, timeout_s: float | None = None) -> bool:
        """同步做一次健康检查并更新缓存。

        Args:
            timeout_s: 本次探测的连接/读超时，默认 ``health_timeout_s``。

        Returns:
            服务可达且 ``ok=1`` 时为 True；任何异常都返回 False（不抛）。
        """
        t0 = time.perf_counter()
        ok = False
        info: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(
                self.health_url, timeout=timeout_s or self.health_timeout_s
            ) as resp:
                info = json.loads(resp.read().decode("utf-8", "replace"))
            ok = int(info.get("ok", 0) or 0) == 1
        except Exception as exc:  # noqa: BLE001 - 探测失败不是错误，是"不可用"
            self.last_error = f"probe: {type(exc).__name__}: {exc}"
        finally:
            self._remember(self.probe_ms, (time.perf_counter() - t0) * 1000.0)
        self._apply_health(ok, t0)
        if ok:
            self.actions = [str(a) for a in (info.get("actions") or [])]
            self.llm = bool(info.get("llm"))
        return ok

    def _apply_health(self, ok: bool, started: float) -> None:
        """合并一次探测结果（带单调闸门与失败阈值，见 ``__init__`` 的注释）。"""
        with self._health_lock:
            if started < self._health_stamp:
                return  # 更早开始的探测晚到了：它的结论已经过期
            self._health_stamp = started
            if ok:
                self._health_fail = 0
                self._healthy = True
            else:
                self._health_fail += 1
                if self._health_fail >= self.fail_threshold:
                    self._healthy = False

    def _mark_down(self) -> None:
        """一次真实请求失败：立刻判不可用，不等探测阈值。"""
        with self._health_lock:
            self._health_stamp = time.perf_counter()
            self._health_fail = self.fail_threshold
            self._healthy = False

    def wait_healthy(self, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
        """轮询到可用或超时（demo 等子进程起来用）。

        Args:
            timeout_s: 最长等待。
            poll_s: 轮询间隔。

        Returns:
            在超时前可用则为 True。
        """
        deadline = time.perf_counter() + max(0.0, timeout_s)
        while True:
            if self.probe(min(self.health_timeout_s, max(0.02, deadline - time.perf_counter()))):
                return True
            if time.perf_counter() >= deadline:
                return False
            time.sleep(poll_s)

    def _health_loop(self) -> None:
        while not self._stop.wait(self.health_ttl_s):
            self.probe(self.health_timeout_s)

    # ------------------------------------------------------------ 请求翻译
    def reason_family(self, request: dict) -> tuple[str, str]:
        """从归因里取 (首要势函数, 风险族)。取不到就是空串。"""
        why = request.get("why")
        top = ""
        if isinstance(why, list) and why and isinstance(why[0], dict):
            top = str(why[0].get("k", "") or "")
        return top, _FAMILY.get(top, "")

    def intent_for(self, request: dict) -> str:
        """把一次升级翻译成给 PC agent 的自然语言意图（带归因上下文）。"""
        top, family = self.reason_family(request)
        base = FAMILY_INTENT.get(family, DEFAULT_INTENT)
        p_esc = _finite(request.get("p_esc"), 0.0)
        return f"{base}（判据 {top or 'n/a'}，p_esc={p_esc:.2f}）"

    def build_task_request(self, request: dict) -> dict:
        """esc/1 请求 -> agent_bridge 的 ``/agent/task`` 请求体。"""
        intent = self.intent_for(request)[:400]
        body: dict[str, Any] = {"intent": intent, "level": self.level}
        if self.action_hint:
            body["action"] = self.action_hint
            # note 动作读 args.text；其他动作忽略它。
            body["args"] = {"text": intent}
        return body

    # ------------------------------------------------------------ 结果翻译
    def to_instruction(
        self, request: dict, result: dict, roundtrip_ms: float
    ) -> Instruction:
        """agent 的执行结果 -> 通过校验的 ``Instruction``。

        Raises:
            BackendError: 信封不是对象 / ``ok != 1`` / 缺 action。
        """
        if not isinstance(result, dict):
            raise BackendError(f"{self.name}: 响应不是 JSON 对象")
        action = str(result.get("action", "") or "").strip()
        detail = str(result.get("detail", "") or "")
        if int(_finite(result.get("ok"), 0.0)) != 1:
            raise BackendError(f"{self.name}: agent 执行失败 [{action or '?'}] {detail[:120]}")
        if not action:
            raise BackendError(f"{self.name}: 响应缺 action 字段")

        planner = str(result.get("planner", "") or "?")
        say = _SAY.get(action, "PC 返回 {detail}").format(detail=detail)[:200]
        ttl = int(_finite(request.get("ttl_ms"), 1500.0))
        obj = {
            "id": int(_finite(request.get("id"), -1.0)),
            "level": self.level,
            "intent": f"pc_{action}"[:64],
            "prog": [dict(s) for s in ACTION_PROG.get(action, _GENERIC_PROG)],
            "ttl_ms": max(100, min(1500, ttl)),
            "say": say,
            "confidence": _PLANNER_CONF.get(planner, 0.5),
            "args": {
                "action": action,
                "detail": detail,              # 真实执行输出原文
                "planner": planner,
                "agent_ms": _finite(result.get("elapsed_ms"), 0.0),
                "roundtrip_ms": round(roundtrip_ms, 2),
                "backend": self.name,
                "url": self.task_url,
            },
        }
        return parse_instruction(obj, source=self.name, raw=self.last_raw)

    # ------------------------------------------------------------ 执行
    def ask(self, request: dict, timeout_s: float) -> Instruction:
        """把请求交给 agent_bridge 执行。

        Args:
            request: ``build_escalation_request`` 的产物。
            timeout_s: hub 给的预算（秒）。实际 socket 超时是它的 ``timeout_ratio``。

        Returns:
            通过 ``parse_instruction`` 校验的指令。

        Raises:
            BackendUnavailable: 建连失败/建连超时/连接中途断开（进程死了走这条）。
            BackendTimeout: 连上了但预算内没回包。
            BackendError: HTTP 非 200 / 非 JSON / ``ok=0``。
            InvalidInstruction: 翻译出的指令不合法（不该发生，是真 bug）。
        """
        self.calls += 1
        t0 = time.perf_counter()
        self.last_task = self.build_task_request(request)
        try:
            text = self._post(self.last_task, self._socket_timeout(timeout_s))
        except (BackendError, BackendTimeout, BackendUnavailable) as exc:
            self._count_failure(exc, t0)
            raise

        roundtrip = self._remember(self.roundtrip_ms, (time.perf_counter() - t0) * 1000.0)
        self.last_raw = text
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            self.errors += 1
            self.last_error = f"bad json: {text[:120]!r}"
            raise BackendError(f"{self.name}: 响应不是 JSON：{text[:120]!r}") from exc

        self.last_result = result if isinstance(result, dict) else {}
        try:
            ins = self.to_instruction(request, result, roundtrip)
        except BackendError as exc:
            self._count_failure(exc, t0)
            raise
        self.ok += 1
        return ins

    # ------------------------------------------------------------ 内部
    def _socket_timeout(self, timeout_s: float) -> float:
        return max(MIN_SOCKET_TIMEOUT_S, _finite(timeout_s, 1.0) * self.timeout_ratio)

    def _post(self, payload: dict, read_timeout_s: float) -> str:
        """POST 一次 ``/agent/task``，把两类失败分开。

        建连用 ``connect_timeout_s``，之后把 socket 超时切到 ``read_timeout_s``：
        "连不上"（进程死了）= ``BackendUnavailable``，快；
        "连上了但没回包"（agent 卡住）= ``BackendTimeout``，受预算约束。
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        conn = cls(self.host, self.port, timeout=self.connect_timeout_s)
        try:
            try:
                conn.connect()
            except TimeoutError as exc:
                self._mark_down()
                raise BackendUnavailable(
                    f"{self.name}: 连接 {self.host}:{self.port} "
                    f"{self.connect_timeout_s * 1000.0:.0f}ms 超时（进程可能已退出）"
                ) from exc
            except OSError as exc:
                self._mark_down()
                raise BackendUnavailable(
                    f"{self.name}: 连接 {self.host}:{self.port} 失败：{exc}"
                ) from exc
            if conn.sock is not None:
                conn.sock.settimeout(read_timeout_s)
            try:
                conn.request("POST", self.task_path, body=data,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                raw = resp.read()
            except TimeoutError as exc:
                raise BackendTimeout(
                    f"{self.name}: {read_timeout_s * 1000.0:.0f}ms 内没回包"
                ) from exc
            except (ConnectionError, OSError) as exc:
                self._mark_down()
                raise BackendUnavailable(
                    f"{self.name}: 连接中断 {type(exc).__name__}: {exc}"
                ) from exc
            if resp.status != 200:
                raise BackendError(f"{self.name}: HTTP {resp.status} {self.task_path}")
            return raw.decode("utf-8", "replace")
        finally:
            conn.close()

    def _count_failure(self, exc: BaseException, t0: float) -> None:
        if isinstance(exc, BackendTimeout):
            self.timeouts += 1
        elif isinstance(exc, BackendUnavailable):
            self.unavailable += 1
        else:
            self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        self._remember(self.roundtrip_ms, (time.perf_counter() - t0) * 1000.0)

    @staticmethod
    def _remember(xs: list[float], v: float) -> float:
        xs.append(v)
        if len(xs) > 500:
            del xs[:250]
        return v

    # ------------------------------------------------------------ 统计
    def summary(self) -> dict:
        """一次调用账本：计数 + 往返延迟分位数（毫秒）。"""
        return {
            "name": self.name,
            "url": self.task_url,
            "healthy": self._healthy,
            "actions": list(self.actions),
            "llm": self.llm,
            "calls": self.calls,
            "ok": self.ok,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "unavailable": self.unavailable,
            "roundtrip_n": len(self.roundtrip_ms),
            "roundtrip_p50_ms": round(percentile(self.roundtrip_ms, 0.5), 2),
            "roundtrip_p95_ms": round(percentile(self.roundtrip_ms, 0.95), 2),
            "roundtrip_max_ms": round(max(self.roundtrip_ms), 2) if self.roundtrip_ms else 0.0,
            "last_error": self.last_error,
        }
