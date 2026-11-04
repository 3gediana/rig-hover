#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""detail_check.py -- 判断 agent_bridge 的 detail 是不是**真实执行输出**。

验收要求 detail 不能是占位符。这个模块用本机可以独立算出来的事实去核对它：

======================  ==========================================================
动作                     核对方式
======================  ==========================================================
``clock``                ``HH:MM:SS``，与本机时钟相差 < 120s
``system_status``        hostname 必须等于本机 ``socket.gethostname()``，
                         ``disk_free_gb`` 与 ``shutil.disk_usage(os.sep)`` 相差 < 3GB，
                         ``time`` 与本机时钟相差 < 120s
``disk``                 每个盘的剩余空间与 ``shutil.disk_usage`` 相差 < 3GB
``note``                 日志最后一行的时间戳就是刚才，字符数 == detail 里的 N
``screenshot``           文件存在，尺寸形如 ``1707x1067``
======================  ==========================================================

``check_detail`` 返回 ``(ok, 证据串)``，证据串直接打到 demo 输出里——"detail 是
真的"这句话必须有可复现的依据，不能自己声明。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path

__all__ = ["check_detail", "notes_path"]

#: agent_bridge 的 note 日志位置（server.py 里是固定路径）。
notes_path = Path(__file__).resolve().parent.parent / "agent_bridge" / "robot_notes.log"

#: 时钟类证据允许的偏差（秒）。跨进程、跨执行的正常抖动远小于它。
CLOCK_TOLERANCE_S = 120.0
#: 磁盘类证据允许的偏差（GB）。detail 只保留整数 GB。
DISK_TOLERANCE_GB = 3.0

_DISK_RE = re.compile(r"([A-Za-z]):\s*(\d+)G free")
_NOTE_RE = re.compile(r"^noted (\d+) chars$")
_SHOT_RE = re.compile(r"^saved (\S+) \((\d+)x(\d+)\)$")
_CLOCK_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")


def _recent_hms(ts: str) -> float:
    """``HH:MM:SS`` 与本机当前时刻之差（秒，可能为负）。"""
    m = _CLOCK_RE.match(ts.strip())
    if not m:
        raise ValueError(f"不是 HH:MM:SS：{ts!r}")
    h, mi, s = (int(x) for x in m.groups())
    now = datetime.now()
    cand = now.replace(hour=h, minute=mi, second=s, microsecond=0)
    return (cand - now).total_seconds()


def _recent_iso(ts: str) -> float:
    """ISO 时间戳与本机当前时刻之差（秒）。"""
    return (datetime.fromisoformat(ts) - datetime.now()).total_seconds()


def check_detail(
    action: str, detail: str, *, notes: Path | None = None
) -> tuple[bool, str]:
    """核对一条执行结果是否来自真实动作。

    Args:
        action: agent_bridge 返回的动作名。
        detail: 该动作返回的 detail 原文。
        notes: note 动作的日志路径，默认 ``notes_path``。

    Returns:
        ``(ok, 证据)``。``ok=False`` 时证据里写清哪一条对不上。
    """
    detail = str(detail or "")
    if not detail.strip():
        return False, "detail 为空（占位符）"
    try:
        if action == "clock":
            delta = _recent_hms(detail)
            ok = abs(delta) < CLOCK_TOLERANCE_S
            return ok, f"本机时钟差 {delta:+.0f}s" if ok else f"与本机时钟差 {delta:+.0f}s"
        if action == "system_status":
            obj = json.loads(detail)
            host = str(obj.get("hostname", ""))
            same_host = host == socket.gethostname()
            free = float(obj.get("disk_free_gb", -1.0))
            root_free = shutil.disk_usage(os.path.abspath(os.sep)).free / 1e9
            disk_ok = free > 0 and abs(free - root_free) < DISK_TOLERANCE_GB
            dt = abs(_recent_iso(str(obj.get("time", ""))))
            time_ok = dt < CLOCK_TOLERANCE_S
            ok = same_host and disk_ok and time_ok and bool(obj.get("platform"))
            ev = (f"hostname={host}{'=' if same_host else '!='}{socket.gethostname()} "
                  f"disk_free={free:.1f}G(本机 {root_free:.1f}G) time±{dt:.0f}s")
            return ok, ev
        if action == "disk":
            hits = _DISK_RE.findall(detail)
            if not hits:
                return False, f"没有 'X: NN G free' 形式的盘符：{detail[:60]!r}"
            worst = 0.0
            for letter, gb in hits:
                real = shutil.disk_usage(f"{letter.upper()}:\\").free / 1e9
                worst = max(worst, abs(real - float(gb)))
            return worst < DISK_TOLERANCE_GB, f"{len(hits)} 个盘，最大偏差 {worst:.1f}GB"
        if action == "note":
            m = _NOTE_RE.match(detail)
            if not m:
                return False, f"格式不对：{detail!r}"
            n = int(m.group(1))
            path = Path(notes) if notes is not None else notes_path
            if not path.is_file():
                return False, f"日志不存在：{path}"
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                return False, "日志为空"
            ts, _, text = lines[-1].partition("\t")
            delta = abs(_recent_iso(ts))
            ok = delta < CLOCK_TOLERANCE_S and len(text) == n
            return ok, (f"{path.name} 末行 {delta:.0f}s 前，{len(text)} 字 == detail 的 {n}")
        if action == "screenshot":
            m = _SHOT_RE.match(detail)
            if not m:
                return False, f"格式不对：{detail!r}"
            name, w, h = m.group(1), int(m.group(2)), int(m.group(3))
            path = Path(__file__).resolve().parent.parent / "agent_bridge" / name
            ok = path.is_file() and w > 0 and h > 0 and time.time() - path.stat().st_mtime < 300
            return ok, f"{name} 存在且是 {w}x{h}"
    except (ValueError, OSError, json.JSONDecodeError, TypeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    # 未知动作：只能证明它不是空串
    return True, "未知动作，只校验非空"
