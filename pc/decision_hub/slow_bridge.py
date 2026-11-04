#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""slow_bridge.py -- 给 agent_bridge 注入可控延迟的测试夹具。

对照实验需要在**同一条 HTTP 契约、同一份服务端代码**上让 agent 变慢，而不改
``agent_bridge/server.py``（那份服务要保持独立可测、默认行为不变）。做法是把
server.py 当模块加载，用子类只覆盖 ``do_POST``：先按 ``--delay-ms`` 睡，再走
原实现。健康检查 ``GET /agent/health`` 不打延迟——否则客户端连"这一级是否
存在"都探测不到，测的就不是超时而是不可达了。

用法::

    python slow_bridge.py --port 8899 --delay-ms 1500

``--delay-ms 0`` 时行为与原服务完全一致（默认值）。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

#: 被包装的真实服务。只读，不修改。
SERVER_PATH = Path(__file__).resolve().parent.parent / "agent_bridge" / "server.py"


def load_server(path: Path = SERVER_PATH):
    """按文件路径加载 agent_bridge 的 server 模块（同目录里没有同名包冲突）。"""
    if not path.is_file():
        raise SystemExit(f"找不到 agent_bridge: {path}")
    spec = importlib.util.spec_from_file_location("agent_bridge_server", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_handler(server_mod, delay_s: float):
    """造一个只对 ``/agent/task`` 延迟的 Handler 子类。"""

    class DelayHandler(server_mod.Handler):
        def do_POST(self):  # noqa: N802 - stdlib 命名
            if delay_s > 0.0 and self.path.startswith("/agent/task"):
                time.sleep(delay_s)
            super().do_POST()

    return DelayHandler


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # 客户端在延迟期间超时断开后，原 handler 还会往一个关闭的 socket 写回包。
        # 这是本夹具的预期行为，不该污染输出。
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def serve(port: int, delay_ms: float) -> None:
    mod = load_server()
    handler = make_handler(mod, delay_ms / 1000.0)
    srv = _Server(("0.0.0.0", port), handler)
    print(f"[slow_bridge] listening on :{port}  delay={delay_ms:.0f}ms "
          f"actions={sorted(mod.ACTIONS)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[slow_bridge] stopped", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--delay-ms", type=float, default=0.0,
                    help="只对 /agent/task 注入的延迟（毫秒）；0 = 与原服务一致")
    a = ap.parse_args(argv)
    serve(a.port, max(0.0, a.delay_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
