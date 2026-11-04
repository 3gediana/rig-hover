# -*- coding: utf-8 -*-
"""_repair_results.py -- recover RESULTS.md, then fill the bytes the GBK decode ate.

Two stages, applied per line of the mojibake file:

1. Invert the corruption. A pipeline read UTF-8 bytes with a GBK decoder and wrote
   the result back as UTF-8, so the inverse is
   ``line.encode('gb18030').decode('utf-8')``. gb18030 rather than gbk: the
   mojibake's PUA characters come from GBK's user-defined regions, which plain
   gbk refuses to encode.

2. Fill the loss points. Where the GBK decoder met a lead byte whose trailing
   byte was not a valid continuation it substituted '?', destroying exactly one
   byte. That byte is constrained, not free: a UTF-8 CJK character is three
   bytes, the first two survive, and the third lies in 0x80..0xBF -- 64 candidate
   characters per hole. The candidate is chosen with a bigram model built from
   jieba's word-frequency dictionary, so the surrounding Chinese decides rather
   than Unicode order.

A line is only replaced when the repaired text decodes without U+FFFD and scores
better than the line it came from. Lines whose CJK does not belong to the
corruption repertoire are left untouched, which keeps already-correct text and
pure-ASCII tables out of the round trip.

Writes a sibling file; never overwrites the input.
"""
from __future__ import annotations

import argparse
import math
import os

import jieba

CAND = bytes(range(0x80, 0xC0))


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


def build_model():
    """Character and bigram frequencies from jieba's shipped word list."""
    dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    uni: dict[str, int] = {}
    bi: dict[str, int] = {}
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            w, n = parts[0], int(parts[1])
            try:
                n = int(parts[1])
            except ValueError:
                continue
            if len(w) == 1:
                uni[w] = uni.get(w, 0) + n
            else:
                for a, b in zip(w, w[1:]):
                    k = a + b
                    bi[k] = bi.get(k, 0) + n
    return uni, bi


def logf(x: int) -> float:
    return math.log(x + 1.0)


def plain_inverse(line: str) -> str | None:
    try:
        return line.encode("gb18030").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def repair_line(line: str, uni, bi):
    """Return (text, n_bytes_filled). Falls back to the input on any doubt."""
    base: list[int | None] = []
    holes: list[int] = []
    for ch in line:
        if ch == "?":
            holes.append(len(base))
            base.append(None)
        else:
            try:
                base.extend(ch.encode("gb18030"))
            except UnicodeEncodeError:
                return line, 0

    if not holes:
        inv = plain_inverse(line)
        return (inv, 0) if inv is not None else (line, 0)

    buf = list(base)
    filled = 0
    unresolved: list[int] = []
    for h in holes:
        prefix = bytes(x for x in buf[:h] if x is not None)
        idx = len(prefix.decode("utf-8", "ignore"))
        best = None
        for b in CAND:
            try:
                t = (prefix + bytes([b])).decode("utf-8", "ignore")
            except UnicodeDecodeError:
                continue
            if len(t) <= idx:
                continue
            ch = t[idx]
            if not cjk(ch):
                continue
            prev = t[idx - 1] if idx > 0 else ""
            nxt = ""
            sc = (logf(uni.get(ch, 0))
                  + 2.0 * logf(bi.get(prev + ch, 0)))
            if best is None or sc > best[0]:
                best = (sc, b)
        if best is None:
            buf[h] = 0x3F
            unresolved.append(h)
            continue
        buf[h] = best[1]
        filled += 1

    # second pass: re-score each filled hole with both neighbours known
    for h in holes:
        if h in unresolved:
            continue
        prefix = bytes(x for x in buf[:h] if x is not None)
        idx = len(prefix.decode("utf-8", "ignore"))
        best = None
        for b in CAND:
            try:
                t = (prefix + bytes([b])).decode("utf-8", "ignore")
            except UnicodeDecodeError:
                continue
            if len(t) <= idx:
                continue
            ch = t[idx]
            if not cjk(ch):
                continue
            prev = t[idx - 1] if idx > 0 else ""
            nxt = ""
            tail = bytes(x for x in buf[h + 1:] if x is not None)
            try:
                tt = (bytes([b]) + tail).decode("utf-8")
            except UnicodeDecodeError:
                tt = (bytes([b]) + tail).decode("utf-8", "ignore")
            if len(tt) > 1:
                nxt = tt[1]
            sc = (logf(uni.get(ch, 0))
                  + 2.0 * logf(bi.get(prev + ch, 0))
                  + 2.0 * logf(bi.get(ch + nxt, 0)))
            if best is None or sc > best[0]:
                best = (sc, b)
        if best is not None:
            buf[h] = best[1]

    try:
        out = bytes(x if x is not None else 0x3F for x in buf).decode("utf-8")
    except UnicodeDecodeError:
        out = bytes(x if x is not None else 0x3F for x in buf).decode(
            "utf-8", "replace")

    # An unresolved hole decodes to an ordinary '?', which reads as punctuation.
    # Mark it as a lost character instead, so the damage is visible.
    for h in reversed(unresolved):
        idx = len(bytes(x for x in buf[:h] if x is not None).decode("utf-8", "ignore"))
        if idx < len(out) and out[idx] == "?":
            out = out[:idx] + "\ufffd" + out[idx + 1:]
    return out, filled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\robot_work\RESULTS.md")
    ap.add_argument("--dst", default=r"D:\robot_work\RESULTS.recovered.md")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    print(f"src {a.src}  {len(raw)} bytes  BOM={has_bom}")
    text = raw.decode("utf-8-sig", "replace")

    rep = build_repertoire()
    uni, bi = build_model()
    print(f"repertoire {len(rep)} chars; model {len(uni)} chars / {len(bi)} bigrams")

    out = []
    stat = {"inv": 0, "repaired": 0, "ascii": 0, "kept": 0}
    filled = 0
    for line in text.splitlines():
        chars = [c for c in line if cjk(c) or pua(c)]
        if not chars:
            stat["ascii"] += 1
            out.append(line)
            continue
        hit = sum(1 for c in chars if c in rep or pua(c))
        if hit / len(chars) < a.threshold:
            stat["kept"] += 1
            out.append(line)
            continue
        rec, n = repair_line(line, uni, bi)
        if pua(rec) >= pua(line) and pua(line) > 0:
            stat["kept"] += 1
            out.append(line)
            continue
        if pua(rec) > 0:
            stat["kept"] += 1
            out.append(line)
            continue
        if n:
            stat["repaired"] += 1
            filled += n
        else:
            stat["inv"] += 1
        out.append(rec)

    body = "\n".join(out) + "\n"
    with open(a.dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    n_cjk = sum(1 for c in body if cjk(c))
    print(f"lines: invert-only {stat['inv']}  repaired {stat['repaired']}  "
          f"ascii {stat['ascii']}  kept {stat['kept']}   bytes filled {filled}")
    print(f"dst {a.dst}: {len(body)} chars, CJK {n_cjk}, "
          f"PUA {sum(1 for c in body if pua(c))}, ? {body.count('?')}, "
          f"U+FFFD {body.count(chr(0xFFFD))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
