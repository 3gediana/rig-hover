# -*- coding: utf-8 -*-
"""_check_encoding.py -- verify text deliverables are clean UTF-8 with no BOM.

The project lost three source files and one document to a GBK round trip through
PowerShell, so every deliverable that carries Chinese gets checked before it is
called done. Reports per file: byte size, line count, whether a BOM is present,
whether the bytes decode as UTF-8 at all, and how many private-use or
replacement characters survived.

Run:  python _check_encoding.py [file ...]
"""
from __future__ import annotations

import glob
import sys

BOM = b"\xef\xbb\xbf"


def check(path: str) -> bool:
    raw = open(path, "rb").read()
    has_bom = raw[:3] == BOM
    try:
        text = raw.decode("utf-8")
        decodes = True
    except UnicodeDecodeError:
        text = raw.decode("utf-8", "replace")
        decodes = False
    pua = sum(1 for c in text if "\ue000" <= c <= "\uf8ff")
    bad = text.count("\ufffd")
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    lines = text.count("\n")
    clean = decodes and not has_bom and pua == 0 and bad == 0
    print(f"{'OK  ' if clean else 'BAD '}{path}")
    print(f"     {len(raw)} bytes  {lines} lines  CJK {cjk}  "
          f"utf8={decodes}  bom={has_bom}  pua={pua}  u+fffd={bad}")
    return clean


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = [r"D:\robot_work\*.md",
                r"D:\robot_work\*.txt"]
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a])
    ok = True
    for p in paths:
        try:
            ok &= check(p)
        except OSError as exc:
            print(f"BAD  {p}: {exc}")
            ok = False
    print("\nall clean" if ok else "\nsome files need attention")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
