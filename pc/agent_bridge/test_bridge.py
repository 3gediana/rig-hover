#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_bridge.py -- acceptance tests for the local agent bridge.

Starts the server on a scratch port, drives it over real HTTP, and checks both
the happy path and the ways an untrusted caller could misbehave. The bridge is
the level-2 executor the robot escalates to, so "it returns JSON" is not the
bar -- it must also refuse anything outside its whitelist.

Run:  python test_bridge.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "server.py")
PORT = 8931
BASE = f"http://127.0.0.1:{PORT}"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def post(path: str, payload, raw: bytes | None = None, timeout: float = 20.0):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except Exception:
            return exc.code, {"_raw": body[:120].decode("utf-8", "replace")}


def get(path: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def wait_port(host: str, port: int, seconds: float = 12.0) -> bool:
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            with socket.create_connection((host, port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.15)
    return False


def main() -> int:
    proc = subprocess.Popen([sys.executable, SERVER, "--port", str(PORT)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if not wait_port("127.0.0.1", PORT):
            print("server failed to start")
            out, err = proc.communicate(timeout=5)
            print((out + err).decode("utf-8", "replace")[:800])
            return 2

        # ---- health -------------------------------------------------------
        st, h = get("/agent/health")
        check("GET /agent/health 返回 200", st == 200, f"status={st}")
        check("health 列出全部动作",
              set(h.get("actions", [])) == {"clock", "disk", "note",
                                            "screenshot", "system_status"},
              str(h.get("actions")))
        check("health 报告 llm 可用性", "llm" in h, f"llm={h.get('llm')}")

        # ---- every whitelisted action really runs -------------------------
        for act in ("clock", "disk", "system_status"):
            st, r = post("/agent/task", {"intent": "", "action": act, "level": 2})
            check(f"显式动作 {act} 执行成功",
                  st == 200 and r.get("ok") == 1 and bool(r.get("detail")),
                  f"detail={str(r.get('detail'))[:60]}")

        # ---- planner routes by intent -------------------------------------
        st, r = post("/agent/task", {"intent": "what time is it", "level": 2})
        check("规则 planner 把 clock 意图路由对",
              r.get("action") == "clock" and r.get("planner") == "rule",
              f"action={r.get('action')} planner={r.get('planner')}")
        st, r = post("/agent/task", {"intent": "how much disk space", "level": 3})
        check("未知精确动作时回落到 planner", r.get("action") == "disk",
              f"action={r.get('action')}")
        check("回包带上 level 字段", r.get("level") == 3, f"level={r.get('level')}")

        # ---- untrusted-input handling -------------------------------------
        st, r = post("/agent/task", {"intent": "drop table", "level": 2})
        check("未知意图不会崩，走兜底", st == 200 and r.get("ok") == 1,
              f"action={r.get('action')}")

        st, r = post("/agent/task", {"action": "../../etc/passwd"})
        check("非白名单动作被 planner 接管而不是执行",
              st == 200 and r.get("planner") != "explicit",
              f"action={r.get('action')} planner={r.get('planner')}")

        st, r = post("/agent/task", None, raw=b"{not json")
        check("坏 JSON 返回 400", st == 400, f"status={st} err={r.get('err')}")

        st, r = post("/agent/task", None, raw=b"x" * 9000)
        check("超大 body 被拒", st == 400, f"status={st} err={r.get('err')}")

        st, r = post("/agent/unknown", {"intent": "x"})
        check("未知路径返回 404", st == 404, f"status={st} err={r.get('err')}")

        # ---- note action really writes ------------------------------------
        marker = f"test_bridge {time.time():.3f}"
        st, r = post("/agent/task", {"intent": marker, "action": "note",
                                     "args": {"text": marker}, "level": 2})
        log = os.path.join(HERE, "robot_notes.log")
        written = os.path.exists(log) and marker in open(log, encoding="utf-8").read()
        check("note 动作真的写盘了", st == 200 and written, f"log={log}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    ok = sum(1 for _n, p, _d in results if p)
    total = len(results)
    print("\n" + "=" * 68)
    print(f"{total} 项，{ok} PASS / {total - ok} FAIL")
    print("PASS" if ok == total else "FAIL")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
