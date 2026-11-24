# -*- coding: utf-8 -*-
"""benign.py -- run a command at BelowNormal priority.

Project rule (user's hard requirement): every simulation and every training
run goes at BelowNormal priority, one process at a time. The machine has been
frozen by parallel heavy runs before. Windows children inherit the parent's
priority class when no class flag is passed to CreateProcess, so lowering this
process before spawning lowers the whole tree (the venv shim included).

Usage:
    python pc/pack/benign.py <command> [args ...]
    python pc/pack/benign.py -- python train.py --epochs 3
"""
from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes

BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def set_below_normal() -> bool:
    """Return True if the priority class of this process is now BelowNormal."""
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # HANDLE must be a pointer-sized type: the pseudo-handle is -1, and a
    # 32-bit int passed where a 64-bit HANDLE is expected reaches the kernel
    # as an invalid handle (first version failed exactly this way).
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
        return False
    return kernel32.GetPriorityClass(handle) == BELOW_NORMAL_PRIORITY_CLASS


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    lowered = set_below_normal()
    if sys.platform == "win32" and not lowered:
        print("benign: could not set BelowNormal priority; refusing to run "
              "(project rule)", file=sys.stderr)
        return 3
    rc = subprocess.call(argv)
    return rc if rc >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
