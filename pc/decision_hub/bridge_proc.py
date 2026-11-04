#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bridge_proc.py -- 把 agent_bridge（或其延迟注入版）作为子进程拉起/收掉。

demo 和测试都要"真的启动一个 agent 服务"，这段逻辑只留一份：

* ``BridgeProcess()``               -> ``../agent_bridge/server.py --port N``
* ``BridgeProcess(script=SLOW_PY,
   extra_args=["--delay-ms", "1500"])`` -> ``slow_bridge.py``（同一份服务代码 + 延迟）

只读 ``agent_bridge/server.py``，不修改它：``--port`` 就是它自己的 CLI 契约。
子进程用 ``-u`` 启动，输出走 PIPE，``stop()`` 时一起收回来当诊断信息。
"""
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

__all__ = ["BridgeProcess", "SERVER_PY", "SLOW_PY", "free_port"]

#: 真实服务（只读、不修改）。
SERVER_PY = Path(__file__).resolve().parent.parent / "agent_bridge" / "server.py"
#: 同目录的延迟注入封装（import server.py，只覆盖 do_POST）。
SLOW_PY = Path(__file__).resolve().parent / "slow_bridge.py"


def free_port() -> int:
    """向内核要一个当前空闲的本地端口（bind 完立刻释放，仅供子进程使用）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class BridgeProcess:
    """agent_bridge 子进程。

    Args:
        port: 监听端口；None 用随机空闲端口。
        script: 要跑的脚本，默认真实 server.py。
        extra_args: 追加给脚本的参数（如 ``--delay-ms 1500``）。
        python: 解释器，默认当前解释器。
    """

    def __init__(self, port: int | None = None, script: Path = SERVER_PY,
                 extra_args: tuple[str, ...] = (), python: str | None = None):
        self.port = int(port or free_port())
        self.script = Path(script)
        self.extra_args = tuple(str(a) for a in extra_args)
        self.python = python or sys.executable
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "BridgeProcess":
        if not self.script.is_file():
            raise RuntimeError(f"找不到 agent_bridge: {self.script}")
        self.proc = subprocess.Popen(
            [self.python, "-u", str(self.script), "--port", str(self.port),
             *self.extra_args],
            cwd=str(self.script.parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return self

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> str:
        """终止子进程并返回它打印过的内容（拿不到就返回空串）。"""
        proc, self.proc = self.proc, None
        if proc is None:
            return ""
        if proc.poll() is None:
            proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                out, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out = ""
        return out or ""

    def __enter__(self) -> "BridgeProcess":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
