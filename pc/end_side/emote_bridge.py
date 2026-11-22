# -*- coding: utf-8 -*-
"""emote_bridge.py — 把 PC 端大脑状态实时映射到 RIG-Hover 的 LCD 表情。

固件已暴露 `POST /emote?name=<emote>`（见 `hover_debug_server.cc` 的
`emote_post_handler`），本模块只做一件事：在大脑状态切换或瞬时事件
（双敲触发 / 3D 锁定 / 悬崖拒走）发生时，把对应的表情名丢给后台线程
去 POST。感知环永不被 HTTP 阻塞：请求在线程里执行，丢帧、超时、断网
都静默吞掉（机器人离线时零副作用）。

用法：
    bridge = EmoteBridge("http://192.168.137.50")
    bridge.update_mode("APPROACH")          # 状态驱动，去重 + 限速
    bridge.event("knock")                    # 瞬时事件，插队显示 ~1.4s
    bridge.close()

离线 / 仿真模式下不要构造 bridge（或构造后机器人离线也安全）。
"""
from __future__ import annotations

import queue
import threading
import time
import urllib.parse
import urllib.request

# 大脑状态 -> LCD 表情名（emote.json 里的 key）。
# brain_* 三个自定义表情由 pc/end_side/make_brain_emotes_v17.py 生成。
MODE_TO_EMOTE: dict[str, str] = {
    "IDLE": "neutral",
    "GAZE_FACE": "loving",
    "GAZE_FIST": "scanning",
    "APPROACH": "thinking",
    "DWELL": "winking",
    "FALLBACK": "confused",
}

# 瞬时事件 -> (表情名, 驻留秒数)。事件期间屏蔽状态更新。
EVENT_EMOTE: dict[str, tuple[str, float]] = {
    "knock": ("brain_knock", 1.6),
    "lock": ("brain_lock", 1.4),
    "cliff": ("brain_cliff_alert", 2.2),
    "kidnap": ("confused", 1.5),
}


class EmoteBridge:
    """线程安全的 LCD 表情桥；所有 HTTP 都在后台线程执行。"""

    def __init__(self, base_url: str, min_interval_s: float = 0.5,
                 timeout_s: float = 0.8, verbose: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.verbose = verbose
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._last_sent: str | None = None
        self._last_sent_t = 0.0
        self._event_until = 0.0
        self._closed = False
        self.sent = 0
        self.failed = 0
        self._th = threading.Thread(target=self._worker, daemon=True, name="emote-bridge")
        self._th.start()

    # ---------- public API (never blocks) ----------
    def update_mode(self, mode_name: str) -> None:
        """状态机每帧可调用；内部去重 + 事件驻留屏蔽。"""
        if self._closed or time.perf_counter() < self._event_until:
            return
        self._offer(MODE_TO_EMOTE.get(mode_name, "neutral"))

    def event(self, name: str) -> None:
        """瞬时事件：knock / lock / cliff / kidnap。"""
        if self._closed:
            return
        entry = EVENT_EMOTE.get(name)
        if entry is None:
            return
        emote, hold_s = entry
        self._event_until = time.perf_counter() + hold_s
        self._offer(emote, force=True)

    def close(self) -> None:
        self._closed = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._th.join(timeout=1.5)

    # ---------- internals ----------
    def _offer(self, emote: str, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and emote == self._last_sent:
            return
        if not force and now - self._last_sent_t < self.min_interval_s:
            return
        try:
            self._q.put_nowait(emote)
        except queue.Full:
            pass  # 机器人断联时队列会满，直接丢

    def _worker(self) -> None:
        while True:
            emote = self._q.get()
            if emote is None:
                return
            url = f"{self.base_url}/emote?name={urllib.parse.quote(emote)}"
            req = urllib.request.Request(url, method="POST", data=b"")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    r.read()
                self.sent += 1
                self._last_sent = emote
                self._last_sent_t = time.perf_counter()
                if self.verbose:
                    print(f"[emote] -> {emote}", flush=True)
            except Exception:
                self.failed += 1
