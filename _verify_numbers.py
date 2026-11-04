# -*- coding: utf-8 -*-
"""_verify_numbers.py -- every number in the rebuilt RESULTS.md must be traceable.

RESULTS.md was rebuilt from a partially recovered original, so the prose is new
text. Numbers are the part that must not move. This extracts every numeric token
from the rebuilt document and reports which ones do not appear in the recovered
original, so a fabricated figure cannot hide in fluent prose.

Derived values are expected to be absent and are reported for manual review
rather than treated as failures.

Run:  python _verify_numbers.py
"""
from __future__ import annotations

import re
import sys

NEW = r"D:\robot_work\RESULTS.md"
OLD = r"D:\robot_work\RESULTS.recovered.md"

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def norm(tok: str) -> str:
    """Strip thousands separators so 2,971,968 and 2971968 compare equal."""
    return tok.replace(",", "")


def collect(text: str) -> set[str]:
    return {norm(m.group(0)) for m in NUM.finditer(text)}


def main() -> int:
    new_text = open(NEW, encoding="utf-8").read()
    old_text = open(OLD, encoding="utf-8", errors="replace").read()

    new_nums = collect(new_text)
    old_nums = collect(old_text)
    # long literals such as sha fragments are noise for this purpose
    missing = sorted((n for n in new_nums if n not in old_nums),
                     key=lambda s: (-len(s), s))

    print(f"numeric tokens: rebuilt {len(new_nums)}, original {len(old_nums)}")
    print(f"not present in the original: {len(missing)}\n")
    for tok in missing:
        # show the line it came from so the reader can judge
        for line in new_text.splitlines():
            if re.search(r"(?<!\d)" + re.escape(tok) + r"(?!\d)", line.replace(",", "")):
                snippet = line.strip()[:110]
                break
        else:
            snippet = ""
        print(f"  {tok:<16} {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
