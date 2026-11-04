# -*- coding: utf-8 -*-
"""_recover_results.py -- undo the UTF-8-read-as-GBK corruption in RESULTS.md.

The file was written by a pipeline that read UTF-8 bytes with a GBK decoder and
re-encoded the result as UTF-8:

    original text T --utf-8--> bytes B --gbk-decode--> mojibake M --utf-8--> file

so the inverse is T = M.encode('gb18030').decode('utf-8'). gb18030 is required
instead of gbk: the PUA characters in the mojibake come from GBK's user-defined
regions, which plain 'gbk' refuses to encode.

Two kinds of line survive the corruption:

* lines with no CJK at all, where the round trip is the identity;
* mojibake lines, where the round trip either inverts exactly or -- when the GBK
  decoder substituted '?' for an undecodable byte pair -- inverts up to each loss
  point, which becomes U+FFFD.

A line counts as mojibake when its CJK characters belong to the corruption's
repertoire: the set of characters obtained by decoding valid UTF-8 CJK byte
pairs as GBK. That repertoire is disjoint from ordinary Chinese text, so
membership separates the two cases far more reliably than counting characters --
1.5 mojibake characters come out of every 1.5 original ones, so counts do not
discriminate.

Writes a sibling file; never overwrites the input.
"""
from __future__ import annotations

import argparse


def cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def pua(ch: str) -> bool:
    return "\ue000" <= ch <= "\uf8ff"


def build_repertoire() -> set[str]:
    """Characters produced by decoding UTF-8-encoded CJK as GBK."""
    rep: set[str] = set()
    for cp in range(0x4E00, 0xA000):
        b = chr(cp).encode("utf-8")
        for i in range(0, len(b) - 1):
            try:
                rep.add(b[i:i + 2].decode("gbk"))
            except UnicodeDecodeError:
                pass
    return rep


def recover_line(line: str) -> str:
    """Best-effort inverse of the GBK mojibake for one line."""
    try:
        return line.encode("gb18030").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    try:
        raw = line.encode("gb18030", "replace")
    except UnicodeEncodeError:
        return line
    return raw.decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\robot_work\RESULTS.md")
    ap.add_argument("--dst", default=r"D:\robot_work\RESULTS.recovered.md")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fraction of a line's CJK that must be in the mojibake "
                         "repertoire for the line to count as corrupted")
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    print(f"src {a.src}  {len(raw)} bytes  BOM={has_bom}")
    text = raw.decode("utf-8-sig", "replace")

    rep = build_repertoire()
    print(f"mojibake repertoire: {len(rep)} characters")

    out = []
    n_clean = n_partial = n_ascii = n_kept = 0
    for line in text.splitlines():
        chars = [c for c in line if cjk(c) or pua(c)]
        if not chars:
            n_ascii += 1
            out.append(line)
            continue
        hit = sum(1 for c in chars if c in rep or pua(c))
        frac = hit / len(chars)
        if frac < a.threshold:
            n_kept += 1
            out.append(line)
            continue
        rec = recover_line(line)
        if rec == line:
            n_kept += 1
            out.append(line)
            continue
        if "\ufffd" in rec:
            n_partial += 1
        else:
            n_clean += 1
        out.append(rec)

    body = "\n".join(out) + "\n"
    with open(a.dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)

    n_cjk = sum(1 for ch in body if cjk(ch))
    n_pua = sum(1 for ch in body if pua(ch))
    print(f"lines: clean {n_clean}  partial {n_partial}  ascii {n_ascii}  "
          f"kept-as-is {n_kept}")
    print(f"dst {a.dst}: {len(body)} chars, real CJK {n_cjk}, PUA {n_pua}, "
          f"U+FFFD {body.count(chr(0xFFFD))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
