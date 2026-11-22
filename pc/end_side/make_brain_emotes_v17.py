"""Generate custom ESP-Emote .eaf animations for the RIG-Hover LCD face.

The firmware ships esp_emote_gfx which decodes a proprietary container:
    header : 0x89 "EAF" + total_frames u32 + checksum u32 + table_len u32
    table  : total_frames x {frame_size u32, frame_offset u32}
    frames : each begins 0x5A5A then "_S" + version(6) + bit_depth u8
             + width u16 + height u16 + blocks u16 + block_height u16
             + block_len u32 * blocks + palette (1<<bit_depth) x 4 BGRA
             + per-block data: enc byte + payload
    block b covers rows [b*block_height, min(height,(b+1)*block_height))
    encodings: 0=RLE (count u8, value u8 pairs), 5=RAW, trailer block "ZZ_C"
    palette  : per entry B,G,R,A; entry all-zero = transparent
    checksum : uint32 sum of every byte in [16 : 16 + table_len]

This generator renders three new brain-state emotes at 240x240:
    brain_knock        - two sonar ripples (double-knock detected)
    brain_lock         - contracting reticle lock (3D target pinned)
    brain_cliff_alert  - pulsing hazard band + '!' (visual-cliff refusal)

Verification: every written file is re-parsed and fully RLE-decoded in
process; decode must reproduce the rendered indexed frames bit-exactly.

Outputs:
    rig_omni/main/boards/hover/emoji/brain_{knock,lock,cliff_alert}.eaf
    runs_end_side/brain_emotes_v17/{preview.mp4,preview.html,report.json}
"""

from __future__ import annotations

import json
import math
import pathlib
import struct
import sys
from typing import Any

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
EMOJI_DIR = REPO / "rig_omni" / "main" / "boards" / "hover" / "emoji"
OUT_DIR = REPO / "runs_end_side" / "brain_emotes_v17"

W = H = 240
BLOCKS = 8
BLOCK_H = 32
VERSION = b"\x00\x00\x00\x00\x00\x01"

# BGR tuples rendered into frames; palette index assigned in order.
PALETTE_RGB: list[tuple[int, int, int]] = [
    (8, 10, 26),      # 0 bg deep navy
    (96, 210, 255),   # 1 cyan
    (255, 255, 255),  # 2 white
    (80, 255, 160),   # 3 green
    (255, 60, 70),    # 4 red
    (255, 200, 40),   # 5 amber
    (30, 40, 70),     # 6 dim steel
    (150, 120, 255),  # 7 violet
]
IDX = {c: i for i, c in enumerate(PALETTE_RGB)}
BG, CYAN, WHITE, GREEN, RED, AMBER, DIM, VIOLET = (IDX[c] for c in PALETTE_RGB)


def _new_frame() -> np.ndarray:
    f = np.full((H, W), BG, dtype=np.uint8)
    return f


def render_brain_knock(n: int = 20) -> list[np.ndarray]:
    """Two expanding ripples from the lower-center contact point."""
    frames = []
    cx, cy = 120, 168
    for i in range(n):
        f = _new_frame()
        phase = (i / n) * 2 * math.pi
        for k, lag in enumerate((0.0, math.pi * 0.62)):
            p = (phase + lag) % (2 * math.pi)
            r = 14 + 86 * (p / (2 * math.pi))
            fade = 1.0 - p / (2 * math.pi)
            col = CYAN if fade > 0.55 else (DIM if fade > 0.3 else 6)
            cv2.circle(f, (cx, cy), int(r), int(col), thickness=3 if k == 0 else 2)
        if i % n < n * 0.35:
            cv2.circle(f, (cx, cy), 9, WHITE, -1)
        else:
            cv2.circle(f, (cx, cy), 6, CYAN, -1)
        if i in (0, 1, 10, 11):  # flash marks on each tap
            cv2.line(f, (cx - 26, cy - 30), (cx - 12, cy - 14), AMBER, 3)
            cv2.line(f, (cx + 26, cy - 30), (cx + 12, cy - 14), AMBER, 3)
        frames.append(f)
    return frames


def render_brain_lock(n: int = 18) -> list[np.ndarray]:
    """Contracting reticle onto a center dot, then a green locked flash."""
    frames = []
    cx, cy = 120, 120
    for i in range(n):
        f = _new_frame()
        t = i / (n - 1)
        r = int(88 - 62 * min(1.0, t * 1.35))
        col = CYAN if t < 0.72 else GREEN
        cv2.circle(f, (cx, cy), r, col, 2)
        gap = max(6, int(14 * (1 - t)))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            x1, y1 = cx + dx * (r + gap), cy + dy * (r + gap)
            x2, y2 = cx + dx * (r + gap + 14), cy + dy * (r + gap + 14)
            cv2.line(f, (x1, y1), (x2, y2), col, 2)
        if t >= 0.72:
            cv2.circle(f, (cx, cy), 10, GREEN if (i // 2) % 2 == 0 else WHITE, -1)
        else:
            cv2.circle(f, (cx, cy), 4, WHITE, -1)
        frames.append(f)
    return frames


def render_brain_cliff_alert(n: int = 14) -> list[np.ndarray]:
    """Pulsing amber hazard band + red '!' warning."""
    frames = []
    for i in range(n):
        f = _new_frame()
        on = (i // 2) % 2 == 0
        y0 = 188
        for x in range(-40, W + 40, 32):
            pts = np.array([[x, H], [x + 16, H], [x + 40, y0], [x + 24, y0]], np.int32)
            cv2.fillPoly(f, [pts], AMBER if on else DIM)
        cv2.rectangle(f, (0, y0), (W, H), AMBER if on else DIM, 3)
        cx = 120
        pts = np.array([[cx, 60], [cx - 34, 128], [cx + 34, 128]], np.int32)
        cv2.fillPoly(f, [pts], RED)
        cv2.polylines(f, [pts], True, WHITE, 3)
        cv2.rectangle(f, (cx - 5, 80), (cx + 5, 104), WHITE, -1)
        cv2.circle(f, (cx, 116), 6, WHITE, -1)
        frames.append(f)
    return frames


def rle_encode(band: np.ndarray) -> bytes:
    flat = band.reshape(-1)
    out = bytearray()
    i = 0
    n = len(flat)
    while i < n:
        v = int(flat[i])
        run = 1
        while i + run < n and flat[i + run] == v and run < 255:
            run += 1
        out += bytes((run, v))
        i += run
    return bytes(out)


def pack_frame(indexed: np.ndarray, palette_bgra: bytes) -> bytes:
    assert indexed.shape == (H, W) and indexed.dtype == np.uint8
    blocks = []
    for b in range(BLOCKS):
        y0 = b * BLOCK_H
        band = indexed[y0 : min(H, y0 + BLOCK_H)]
        enc = b"\x00" + rle_encode(band)  # 0 = RLE
        blocks.append(enc)
    hdr = struct.pack("<H", 0x5A5A) + b"_S\x00" + VERSION + bytes((8,))
    hdr += struct.pack("<HHHH", W, H, BLOCKS, BLOCK_H)
    hdr += b"".join(struct.pack("<I", len(b)) for b in blocks)
    hdr += palette_bgra
    return hdr + b"".join(blocks)


def make_eaf(frames: list[np.ndarray], meta_name: str) -> bytes:
    pal = bytearray()
    for i in range(256):
        if i < len(PALETTE_RGB):
            r, g, b = PALETTE_RGB[i]
            pal += bytes((b, g, r, 255))
        else:
            pal += b"\x00" * 4
    payloads = [pack_frame(f, bytes(pal)) for f in frames]
    meta = b"ZZ" + b"_C" + b"\x00" * 9 + struct.pack("<I", len(meta_name) + 8)
    meta += meta_name.encode() + b" eaf-rle pygen"
    payloads.append(meta)

    total = len(payloads)
    offs, pos = [], 0
    for p in payloads:
        offs.append((len(p), pos))
        pos += len(p)
    table = b"".join(struct.pack("<II", s, o) for s, o in offs)
    data = b"".join(payloads)
    table_len = len(table) + len(data)
    checksum = (sum(bytearray(table + data))) & 0xFFFFFFFF
    hdr = b"\x89EAF" + struct.pack("<III", total, checksum, table_len)
    return hdr + table + data


def verify(path: pathlib.Path, expect: list[np.ndarray]) -> dict[str, Any]:
    d = path.read_bytes()
    assert d[0] == 0x89 and d[1:4] == b"EAF"
    total, chk, tlen = struct.unpack_from("<III", d, 4)
    assert (sum(d[16 : 16 + tlen]) & 0xFFFFFFFF) == chk, "checksum mismatch"
    ok_frames = 0
    for i in range(total - 1):
        sz, off = struct.unpack_from("<II", d, 16 + i * 8)
        f = d[16 + total * 8 + off : 16 + total * 8 + off + sz]
        assert struct.unpack_from("<H", f, 0)[0] == 0x5A5A and f[2:4] == b"_S"
        bd = f[11]
        w, h, nb, bh = struct.unpack_from("<HHHH", f, 12)
        lens = struct.unpack_from("<%dI" % nb, f, 20)
        ncolors = (1 << bd) if bd != 24 else 0
        cur = 20 + nb * 4 + ncolors * 4
        decoded = np.zeros((h, w), np.uint8)
        for bi, bl in enumerate(lens):
            enc = f[cur]
            payload = f[cur + 1 : cur + bl]
            assert enc == 0
            out = bytearray()
            j = 0
            while j < len(payload):
                out += bytes((payload[j + 1],)) * payload[j]
                j += 2
            y0 = bi * bh
            rows = min(h, y0 + bh) - y0
            decoded[y0 : y0 + rows] = np.frombuffer(bytes(out), np.uint8).reshape(rows, w)
            cur += bl
        assert np.array_equal(decoded, expect[i]), f"frame {i} roundtrip mismatch"
        ok_frames += 1
    return {"file": str(path), "frames_checked": ok_frames, "bytes": len(d)}


EMOTES = {
    "brain_knock": render_brain_knock,
    "brain_lock": render_brain_lock,
    "brain_cliff_alert": render_brain_cliff_alert,
}

# Brain-state -> LCD emote schedule (usable live via POST /emote?name=...)
EMOTE_SCHEDULE = {
    "IDLE": "neutral",
    "GAZE_FACE": "loving",
    "GAZE_FIST": "scanning",
    "KNOCK_TRIGGER": "brain_knock",
    "TARGET_LOCKED": "brain_lock",
    "APPROACH": "thinking",
    "DWELL_LOOKBACK": "winking",
    "CLIFF_REFUSAL": "brain_cliff_alert",
    "KIDNAP_RECOVERING": "confused",
    "SEARCH_FACE": "scanning",
    "LOW_BATTERY": "sleepy",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks = []
    strip_rows = []
    for name, fn in EMOTES.items():
        frames = fn()
        blob = make_eaf(frames, f"wenwen {name} v17")
        out = EMOJI_DIR / f"{name}.eaf"
        out.write_bytes(blob)
        res = verify(out, frames)
        res["emote"] = name
        checks.append(res)
        print(f"[emote] {name}: {res['frames_checked']}f RLE-verified, {res['bytes']} B", flush=True)
        # contact strip for the preview
        strip = np.hstack([frames[i] for i in range(0, len(frames), max(1, len(frames) // 6))][:6])
        strip_rows.append(strip)

    # animated preview mp4 (palette -> BGR)
    lut = np.zeros((256, 3), np.uint8)
    for i, (r, g, b) in enumerate(PALETTE_RGB):
        lut[i] = (b, g, r)
    vw = cv2.VideoWriter(str(OUT_DIR / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W * 3, H + 46))
    n = max(len(fn()) for fn in EMOTES.values())
    seqs = {k: fn() for k, fn in EMOTES.items()}
    for i in range(n):
        row = []
        for k, seq in seqs.items():
            fr = cv2.cvtColor(lut[seq[i % len(seq)]], cv2.COLOR_BGR2RGB)
            cv2.putText(fr, k, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            row.append(fr)
        tile = np.hstack(row)
        cap = np.full((tile.shape[0] + 46, tile.shape[1], 3), (12, 14, 24), np.uint8)
        cap[46:] = tile
        cv2.putText(cap, "Wenwen brain-state emotes (POST /emote?name=...)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (160, 220, 255), 1)
        vw.write(cap)
    vw.release()

    report = {
        "generator": "pc/end_side/make_brain_emotes_v17.py",
        "format": "0x89EAF / 240x240 / bit_depth=8 palette / 8x32-row RLE blocks",
        "verification": checks,
        "emote_schedule": EMOTE_SCHEDULE,
        "deploy": "files dropped into rig_omni/main/boards/hover/emoji/ + "
                  "emote.json entries; robot endpoint POST /emote?name=brain_knock",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # simple html preview embedding the mp4 + schedule table
    rows = "".join(f"<tr><td>{s}</td><td>{e}</td></tr>" for s, e in EMOTE_SCHEDULE.items())
    (OUT_DIR / "preview.html").write_text(
        "<meta charset=utf-8><title>Wenwen brain emotes</title>"
        "<body style='background:#0b1020;color:#eee;font-family:system-ui'>"
        "<h2>Wenwen brain-state emotes v17</h2>"
        "<video src=preview.mp4 controls loop autoplay muted width=720></video>"
        "<h3>Brain-state -> LCD emote schedule (POST /emote?name=&lt;emote&gt;)</h3>"
        f"<table border=1 cellpadding=6 style='border-collapse:collapse'>{rows}</table>"
        "</body>", encoding="utf-8")
    print(f"[emote] preview -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
