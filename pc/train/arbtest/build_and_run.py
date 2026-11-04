#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_and_run.py -- compile the firmware safety shell on the host and run
its scenario tests.

control_arbiter.c is the code that keeps the robot from hurting itself, and it
had only ever executed on hardware. This builds it against host stubs and drives
each of its seven defences.

Note on the copy: `control_arbiter.c` uses quoted includes ("robot.h"), and
MSVC resolves those against the including file's own directory before any -I
path. Compiling it in place therefore pulls the firmware's real robot.h (which
drags in motor.h and driver/uart.h). Copying the source next to the stubs makes
the stubs win; the copy is refreshed on every run, so it never goes stale.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.dirname(HERE)
sys.path.insert(0, TRAIN)

import ctools  # noqa: E402
import export_blob  # noqa: E402

FW = export_blob.FIRMWARE_DIR


def main() -> int:
    local_copy = os.path.join(HERE, "control_arbiter_under_test.c")
    shutil.copyfile(FW / "control_arbiter.c", local_copy)

    srcs = [local_copy,
            os.path.join(HERE, "arb_shim.c"),
            os.path.join(HERE, "arb_main.c")]
    exe, warns = ctools.compile_exe(
        srcs, "arbiter_test.exe",
        include_dirs=[HERE, os.path.join(TRAIN, "hoststub"), FW],
        out_dir=os.path.join(HERE, "build"))
    own = [w for w in warns if "control_arbiter" not in w]
    for w in own[:10]:
        print("  warn:", w)
    print(f"built {exe}")
    r = subprocess.run([str(exe)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(r.stdout or "")
    if r.stderr.strip():
        print("[stderr]")
        print(r.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
