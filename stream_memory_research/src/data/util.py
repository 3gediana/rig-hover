"""Tiny shared utilities for the data-video pipeline (logging + status notes)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


class Tee:
    """Duplicate stdout+stderr into a utf-8 log file (line buffered)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        finally:
            sys.stdout = self.stdout
            sys.stderr = self.stderr


def attach_log(path):
    """Redirect this process' stdout/stderr to ``path`` as well as the console."""
    t = Tee(path)
    sys.stdout = t
    sys.stderr = t
    return t


def append_status(msg: str, path="logs/data_video_status.md") -> None:
    """Append a timestamped bullet to the decision log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"- `{stamp}` {msg}\n")


def dir_size_gb(path) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / 1e9
