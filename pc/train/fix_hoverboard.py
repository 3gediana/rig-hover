#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_hoverboard.py -- repair hover_board.cc after a GBK round-trip.

Same failure mode as before: PowerShell decoded the UTF-8 source as GBK and
re-encoded it, so multi-byte chars became mojibake and any <0x40 byte that
followed one (newline, quote, space) was swallowed and replaced with '?'.

Repair rules, applied until the file's brace/quote balance parses:
  1. a comment line that swallowed the statement after it  (#...?STMT)
  2. a string literal that lost its closing quote          ("...?")

Verification here cannot use `compile()` (this is C++), so the acceptance test
is a clean rebuild, which the caller runs.
"""
import re
import sys

P = r"D:\robot_work\rig_omni\main\boards\hover\hover_board.cc"
# '?' followed by something that looks like the start of a statement
STMT = re.compile(r"\?(?=\s*(?:ESP_LOG|esp_|xTask|vTask|const\s|static\s|if\s*\(|for\s*\(|return\s|\w+\s*=\s|\}))")


def main() -> int:
    write = "--write" in sys.argv
    lines = open(P, encoding="utf-8-sig").read().splitlines()

    out, fixed = [], []
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith("//") or l.lstrip().startswith("#"):
            m = STMT.search(l)
            if m:
                cut = m.start()
                out.append(l[:cut].rstrip())
                out.append(l[cut + 1:].rstrip())
                fixed.append((i, "comment/code split"))
                continue
        out.append(l)

    # second pass: a line whose double-quote count is odd and which has a '?'
    # right after a non-ASCII char lost its closing quote
    fixed2 = []
    for i, l in enumerate(out, 1):
        code = l.split("//", 1)[0]
        if code.count('"') % 2 == 1 and "?" in code:
            idx = code.rfind("?")
            if idx > 0 and ord(code[idx - 1]) > 127:
                out[i - 1] = l[:idx] + '"' + l[idx + 1:]
                fixed2.append(i)

    print(f"lines: {len(lines)} -> {len(out)}")
    print(f"comment/code merges split: {len(fixed)}")
    for i, why in fixed[:20]:
        print(f"  L{i}: {why}")
    print(f"quotes restored: {len(fixed2)} -> {fixed2}")

    if write:
        open(P, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
        print(f"written {P}")
    else:
        print("dry run; pass --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
