#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server.py -- a real local agent the robot can escalate to.

The decision hub already decides *when* to escalate; this is the level-2
executor it talks to. Unlike the mock backends in `pc/decision_hub`, every
action here has a real side effect on the host, so an escalation demo proves
the whole path instead of printing a placeholder.

Design constraints:
  * No new dependencies -- stdlib only (http.server + json + subprocess).
  * Whitelisted actions only. The robot is an untrusted input source: never
    let it run an arbitrary command string.
  * Every action returns the same envelope so the hub does not branch on shape:
        {"ok": 1, "action": str, "detail": str, "elapsed_ms": float}
  * An LLM backend is used when a key exists; otherwise a deterministic
    rule-based planner maps intent -> action, so the demo always runs.

Run:  python server.py --port 8899
Test: python server.py --selftest      (runs every action, prints a table)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST_NAME = "rig-hover"
MAX_BODY = 8192

# --------------------------------------------------------------------------
# actions: name -> (description, callable(args: dict) -> str)
# Every callable must be safe for an untrusted caller: no shell interpolation
# of caller-supplied strings, no arbitrary paths, no unbounded work.
# --------------------------------------------------------------------------


def _act_system_status(args: dict) -> str:
    import shutil as _sh
    du = _sh.disk_usage(os.path.abspath(os.sep))
    return json.dumps({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "disk_free_gb": round(du.free / 1e9, 1),
        "time": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False)


def _act_clock(args: dict) -> str:
    return datetime.now().strftime("%H:%M:%S")


def _act_note(args: dict) -> str:
    """Append a timestamped line to a fixed log file. The caller's text is
    written verbatim but never interpreted."""
    text = str(args.get("text", ""))[:400]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robot_notes.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{text}\n")
    return f"noted {len(text)} chars"


def _act_screenshot(args: dict) -> str:
    """Grab the primary screen through .NET so no extra package is needed.

    The script is written to a temp .ps1 and executed by path: passing a long
    multi-statement command through argv mangles the quoting on Windows.
    """
    import tempfile
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen.png")
    ps = f"""
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$b = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Left, $b.Top, 0, 0, $bmp.Size)
$bmp.Save('{out}')
Write-Output "$($bmp.Width)x$($bmp.Height)"
"""
    fd, path = tempfile.mkstemp(suffix=".ps1")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(ps)
        r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                            "-File", path],
                           capture_output=True, text=True, timeout=25)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if r.returncode != 0 or not os.path.exists(out):
        return "screenshot failed: " + ((r.stderr or r.stdout).strip()[:160] or "unknown")
    dims = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
    return f"saved {os.path.basename(out)} ({dims})"


def _act_disk(args: dict) -> str:
    drives = []
    for d in ("C:\\", "D:\\", "E:\\", "G:\\"):
        if os.path.exists(d):
            u = shutil.disk_usage(d)
            drives.append(f"{d[:-1]} {u.free / 1e9:.0f}G free")
    return "; ".join(drives)


ACTIONS = {
    "system_status": ("report host name, platform, free disk and time", _act_system_status),
    "clock":         ("report the current time", _act_clock),
    "note":          ("append a line to the robot notes log", _act_note),
    "screenshot":    ("capture the primary screen to screen.png", _act_screenshot),
    "disk":          ("report free space per drive", _act_disk),
}


# --------------------------------------------------------------------------
# intent -> action planning
# --------------------------------------------------------------------------

def _rule_plan(intent: str) -> tuple[str, dict]:
    """Deterministic fallback planner. Keeps the demo runnable with no API key
    while still exercising the real execute path."""
    t = intent.lower()
    if any(k in t for k in ("time", "clock", "几点", "时间")):
        return "clock", {}
    if any(k in t for k in ("screen", "capture", "截图", "看看")):
        return "screenshot", {}
    if any(k in t for k in ("disk", "space", "存储", "空间")):
        return "disk", {}
    if any(k in t for k in ("note", "remember", "记", "记下")):
        return "note", {"text": intent[:400]}
    return "system_status", {}


def _llm_plan(intent: str) -> tuple[str, dict] | None:
    """Use the LLM only to pick an action name; never to emit code.

    The model output is validated against the whitelist, so a malformed or
    adversarial completion can at worst fall back to the rule planner.
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        import urllib.request
        names = ", ".join(sorted(ACTIONS))
        body = json.dumps({
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": (f"Pick exactly one action for the request below.\n"
                            f"Allowed actions: {names}\n"
                            f"Reply with the action name only.\n"
                            f"Request: {intent[:300]}"),
            }],
            "temperature": 0,
            "max_tokens": 16,
        }).encode()
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read())
        name = (data["choices"][0]["message"]["content"] or "").strip().split()[0]
        name = name.strip(".,`'\"")
        if name in ACTIONS:
            return name, {"text": intent[:400]}
    except Exception:
        return None
    return None


def plan(intent: str) -> dict:
    t0 = time.perf_counter()
    picked = _llm_plan(intent)
    source = "llm"
    if picked is None:
        picked = _rule_plan(intent)
        source = "rule"
    name, args = picked
    desc, fn = ACTIONS[name]
    try:
        detail = fn(args)
        ok = 1
    except Exception as exc:  # noqa: BLE001 - the reason is reported to the robot
        detail = f"{type(exc).__name__}: {exc}"
        ok = 0
    return {"ok": ok, "action": name, "planner": source, "detail": detail,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2)}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "RigAgentBridge/1.0"

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - stdlib naming
        if self.path.startswith("/agent/health"):
            self._send(200, {"ok": 1, "host": HOST_NAME,
                             "actions": sorted(ACTIONS),
                             "llm": bool(os.environ.get("DEEPSEEK_API_KEY"))})
        else:
            self._send(404, {"ok": 0, "err": "not found"})

    def do_POST(self):  # noqa: N802 - stdlib naming
        if not self.path.startswith("/agent/task"):
            self._send(404, {"ok": 0, "err": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0 or n > MAX_BODY:
            self._send(400, {"ok": 0, "err": "bad length"})
            return
        try:
            req = json.loads(self.rfile.read(n))
        except Exception:
            self._send(400, {"ok": 0, "err": "bad json"})
            return
        intent = str(req.get("intent", ""))[:400]
        level = int(req.get("level", 2))
        if req.get("action") in ACTIONS:
            desc, fn = ACTIONS[req["action"]]
            try:
                detail = fn(req.get("args", {}) or {})
                res = {"ok": 1, "action": req["action"], "planner": "explicit",
                       "detail": detail, "elapsed_ms": 0.0}
            except Exception as exc:  # noqa: BLE001
                res = {"ok": 0, "action": req["action"], "planner": "explicit",
                       "detail": f"{type(exc).__name__}: {exc}", "elapsed_ms": 0.0}
        else:
            res = plan(intent)
        res["level"] = level
        self._send(200, res)

    def log_message(self, fmt, *a):  # quiet by default
        if os.environ.get("RIG_AGENT_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def serve(port: int) -> None:
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[agent_bridge] listening on :{port}  actions={sorted(ACTIONS)}  "
          f"llm={bool(os.environ.get('DEEPSEEK_API_KEY'))}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[agent_bridge] stopped")


def selftest() -> int:
    print(f"{'action':<15}{'ok':<4}{'ms':>8}  detail")
    fails = 0
    for name in sorted(ACTIONS):
        t0 = time.perf_counter()
        try:
            detail = ACTIONS[name][1]({})
            ok = 1
        except Exception as exc:  # noqa: BLE001
            detail, ok = f"{type(exc).__name__}: {exc}", 0
            fails += 1
        ms = (time.perf_counter() - t0) * 1000
        print(f"{name:<15}{ok:<4}{ms:>8.1f}  {detail[:80]}")
    print("\n-- planner routing --")
    for intent in ("what time is it", "take a screenshot",
                   "how much disk space", "note that the person waved"):
        r = plan(intent)
        print(f"  {intent!r:<34} -> {r['action']:<14}{r['planner']:<7}{r['elapsed_ms']}ms")
    print(f"\n{'PASS' if fails == 0 else 'FAIL'} ({fails} failing actions)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    serve(a.port)
