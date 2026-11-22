# -*- coding: utf-8 -*-
"""Regression gate for v17 brain-state emote assets + live emote bridge."""
from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from emote_bridge import EVENT_EMOTE, MODE_TO_EMOTE, EmoteBridge  # noqa: E402

EMOJI_DIR = REPO / "rig_omni" / "main" / "boards" / "hover" / "emoji"
EMOTE_JSON = REPO / "rig_omni" / "main" / "boards" / "hover" / "240_240" / "emote.json"
BRAIN_EMOTES = ("brain_knock", "brain_lock", "brain_cliff_alert")


def _parse_eaf(d: bytes) -> dict:
    assert d[0] == 0x89 and d[1:4] == b"EAF"
    total, chk, tlen = struct.unpack_from("<III", d, 4)
    assert (sum(d[16 : 16 + tlen]) & 0xFFFFFFFF) == chk
    frames = []
    for i in range(total):
        sz, off = struct.unpack_from("<II", d, 16 + i * 8)
        frames.append(d[16 + total * 8 + off : 16 + total * 8 + off + sz])
    return {"total": total, "frames": frames}


def _parse_frame(f: bytes) -> dict:
    assert struct.unpack_from("<H", f, 0)[0] == 0x5A5A
    assert f[2:4] == b"_S"
    bd = f[11]
    w, h, nb, bh = struct.unpack_from("<HHHH", f, 12)
    lens = struct.unpack_from("<%dI" % nb, f, 20)
    cur = 20 + nb * 4 + ((1 << bd) if bd != 24 else 0) * 4
    blocks = []
    for bl in lens:
        enc = f[cur]
        payload = f[cur + 1 : cur + bl]
        blocks.append((enc, payload))
        cur += bl
    return {"bd": bd, "w": w, "h": h, "nb": nb, "bh": bh, "blocks": blocks}


def test_eaf_files_structurally_valid():
    import json
    meta = {e["emote"]: e for e in json.loads(EMOTE_JSON.read_text(encoding="utf-8"))}
    for name in BRAIN_EMOTES:
        assert name in meta, f"{name} missing from emote.json"
        p = EMOJI_DIR / meta[name]["src"]
        assert p.exists(), f"{p} missing"
        parsed = _parse_eaf(p.read_bytes())
        n_img = parsed["total"] - 1  # last = ZZ_C trailer
        assert n_img >= 8, f"{name}: too few frames"
        for i in range(n_img):
            fr = _parse_frame(parsed["frames"][i])
            assert fr["w"] == 240 and fr["h"] == 240, f"{name} f{i}: bad dims"
            assert fr["bd"] == 8 and fr["nb"] == 8 and fr["bh"] == 32
            for bi, (enc, payload) in enumerate(fr["blocks"]):
                assert enc in (0, 5), f"{name} f{i}: unknown encoding {enc}"
                if enc == 0:
                    out = 0
                    for j in range(0, len(payload), 2):
                        out += payload[j]
                    rows = min(240, (bi + 1) * 32) - bi * 32  # last block is short: 8x32=256 > 240
                    assert out == 240 * rows, f"{name} f{i} b{bi}: RLE decodes to {out}px"


def test_rle_roundtrip_bitexact():
    sys.path.insert(0, str(HERE))
    import make_brain_emotes_v17 as gen  # noqa: E402

    rng = np.random.default_rng(7)
    img = np.full((240, 240), 0, np.uint8)
    img[40:60, 30:200] = 3
    img[100:220] = rng.integers(0, 6, (120, 240), dtype=np.uint8)
    enc = gen.rle_encode(img.reshape(-1))
    out = np.zeros(240 * 240, np.uint8)
    pos = 0
    for j in range(0, len(enc), 2):
        out[pos : pos + enc[j]] = enc[j + 1]
        pos += enc[j]
    assert pos == 240 * 240
    assert np.array_equal(out.reshape(240, 240), img)


def test_emote_bridge_dedup_and_events():
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    got: list[str] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    import time

    b = EmoteBridge(base, min_interval_s=0.05)
    b.update_mode("GAZE_FACE")
    time.sleep(0.12)
    b.update_mode("GAZE_FACE")  # dedup: no repost
    time.sleep(0.12)
    b.event("knock")
    time.sleep(0.12)
    b.update_mode("DWELL")  # suppressed during event hold
    time.sleep(0.12)
    b.close()
    srv.shutdown()

    assert "/emote?name=loving" in got
    assert got.count("/emote?name=loving") == 1, "dedup failed"
    assert "/emote?name=brain_knock" in got
    assert "/emote?name=winking" not in got  # still inside 1.6s event hold


def test_mode_map_covers_brain_states():
    assert set(EVENT_EMOTE) >= {"knock", "lock", "cliff"}
    assert MODE_TO_EMOTE["GAZE_FIST"] == "scanning"
    # every referenced emote must exist in emote.json (custom brain_* or stock)
    import json
    names = {e["emote"] for e in json.loads(EMOTE_JSON.read_text(encoding="utf-8"))}
    for em in list(MODE_TO_EMOTE.values()) + [e for e, _ in EVENT_EMOTE.values()]:
        assert em in names, f"emote '{em}' not registered in emote.json"


if __name__ == "__main__":
    test_eaf_files_structurally_valid()
    test_rle_roundtrip_bitexact()
    test_emote_bridge_dedup_and_events()
    test_mode_map_covers_brain_states()
    print("ALL 4 TESTS PASSED")
