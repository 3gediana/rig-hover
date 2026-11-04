# -*- coding: utf-8 -*-
"""_make_final_verification.py -- turn the last run_all_tests.py log into the
acceptance record.

The log is the primary evidence, so the record embeds it verbatim rather than
restating its numbers: a summary can drift from the run it claims to describe,
while a transcript cannot. Only the header is generated.

Run:  python _make_final_verification.py
"""
from __future__ import annotations

import datetime
import os
import re

W = r"D:\robot_work"
LOG = os.path.join(W, "final_tests.log")
OUT = os.path.join(W, "FINAL-VERIFICATION.txt")

HEADER = """RIG-Hover 通宵交付 —— 最终验收日志
生成时间: {ts}
python   : D:\\smartcar\\.venv\\Scripts\\python.exe
机器人   : 离线（热点仍在，192.168.137.50 不通）。所有条目均不需要机器人。
说明     : 17 个门禁套件 + 固件编译 + 1 个设计上为红的诊断项。
           本次新增 deploygate（真机帧可部署性）与 calibrate_inputs 的验收入口。

==============================================================================

"""


def read_sniff(path: str) -> str:
    """Read a file whose encoding is unknown.

    PowerShell's Tee-Object writes UTF-16LE on this host while the Python suites
    it captures emit UTF-8, so the log cannot be read with a fixed encoding.
    """
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) > 4 and raw[1] == 0 and raw[3] == 0:
        return raw.decode("utf-16-le")
    return raw.decode("utf-8", "replace")


def main() -> int:
    log = read_sniff(LOG)
    # strip the live progress lines; the summary table carries the same data
    lines = [ln for ln in log.splitlines()
             if not re.match(r"^  \w+\s+\S.* \.\.\. (PASS|FAIL)", ln)]
    body = "\n".join(lines).strip() + "\n"
    ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER.format(ts=ts))
        f.write(body)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
