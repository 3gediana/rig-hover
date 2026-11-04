#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mktest_blob.py -- build a synthetic CfC weight blob for the RIG-Hover firmware.

Purpose: validate the whole POST /model -> flash -> load -> inference path
without waiting for a trained policy. The weights are random; only the layout,
the CRC and the dimension bookkeeping are under test.

The layout is taken from main/boards/hover/cfc_model.h (the authority) and is
verified against the C parser by test_blob_upload.py.
"""
import argparse
import struct
import sys
import zlib

import numpy as np

MAGIC = 0x4D434643  # 'CFCM'
VER = 1
HDR_BYTES = 2240
MAX_HEADS = 8
MAX_INPUT = 256

# layer ids from cfc_model.h
L_BB_W, L_BB_B = 0, 1
L_FF1_W, L_FF1_B = 2, 3
L_FF2_W, L_FF2_B = 4, 5
L_TA_W, L_TA_B = 6, 7
L_TB_W, L_TB_B = 8, 9
L_RET = 10
L_PROJ_W, L_PROJ_B = 11, 12
L_HEADS = 13


def build_blob(input_dim=106, cell_in_dim=64, hidden_dim=64, backbone_units=64,
               head_dims=(1, 1, 4, 8, 1), head_acts=(1, 1, 2, 2, 1),
               has_proj=True, has_ret=True, clip_sigma=4.0, ts=1.0,
               seed=0):
    """Assemble one blob. Returns (bytes, meta dict)."""
    rng = np.random.default_rng(seed)
    n_heads = len(head_dims)
    assert n_heads <= MAX_HEADS, "too many heads"
    assert input_dim <= MAX_INPUT, "input_dim exceeds CFC_MAX_INPUT_DIM"

    cin = cell_in_dim if has_proj else input_dim
    zu = backbone_units if backbone_units > 0 else cin + hidden_dim

    def mat(r, c):
        # xavier-ish, same scale the training code uses
        lim = np.sqrt(6.0 / (r + c))
        return rng.uniform(-lim, lim, size=r * c).astype(np.float32)

    layers = {}
    if backbone_units > 0:
        layers[L_BB_W] = mat(backbone_units, cin + hidden_dim)
        layers[L_BB_B] = np.zeros(backbone_units, np.float32)
    layers[L_FF1_W] = mat(hidden_dim, zu)
    layers[L_FF1_B] = np.zeros(hidden_dim, np.float32)
    for lid_w, lid_b in ((L_FF2_W, L_FF2_B), (L_TA_W, L_TA_B), (L_TB_W, L_TB_B)):
        layers[lid_w] = mat(hidden_dim, zu)
        layers[lid_b] = np.zeros(hidden_dim, np.float32)
    if has_ret:
        layers[L_RET] = np.full(hidden_dim, -3.0, np.float32)  # sigmoid(-3) ~ 0.047
    if has_proj:
        layers[L_PROJ_W] = mat(cell_in_dim, input_dim)
        layers[L_PROJ_B] = np.zeros(cell_in_dim, np.float32)

    total_head = int(sum(head_dims))
    layers[L_HEADS] = np.concatenate([
        mat(hidden_dim, total_head),
        np.zeros(total_head, np.float32),
    ]).astype(np.float32)

    # ---- place payload, 4-byte aligned ----
    off = HDR_BYTES
    layer_off = [0] * 16
    layer_len = [0] * 16
    chunks = []
    for lid in sorted(layers):
        arr = np.ascontiguousarray(layers[lid], dtype=np.float32)
        data = arr.tobytes()
        assert off % 4 == 0
        layer_off[lid] = off
        layer_len[lid] = len(data)
        chunks.append(data)
        off += len(data)
    payload = b"".join(chunks)

    # ---- header ----
    flags = (1 if has_proj else 0) | (2 if has_ret else 0)
    h = bytearray(HDR_BYTES)
    struct.pack_into("<IHH", h, 0, MAGIC, VER, flags)
    struct.pack_into("<HHHHHH", h, 8, input_dim, cin, hidden_dim,
                     backbone_units, n_heads, 0)
    for i in range(MAX_HEADS):
        struct.pack_into("<H", h, 20 + 2 * i, head_dims[i] if i < n_heads else 0)
        struct.pack_into("<H", h, 36 + 2 * i, head_acts[i] if i < n_heads else 0)
    for i in range(16):
        struct.pack_into("<I", h, 52 + 4 * i, layer_off[i])
        struct.pack_into("<I", h, 116 + 4 * i, layer_len[i])

    mean = np.zeros(MAX_INPUT, np.float32)
    std = np.ones(MAX_INPUT, np.float32)
    h[180:180 + 1024] = mean.tobytes()
    h[1204:1204 + 1024] = std.tobytes()
    struct.pack_into("<fff", h, 2228, clip_sigma, ts, 0.0)

    blob = bytes(h) + payload
    # CRC32 over the whole blob with the crc32 field itself zeroed
    crc = zlib.crc32(blob) & 0xFFFFFFFF
    blob = blob[:2236] + struct.pack("<I", crc) + blob[2240:]

    return blob, {
        "input_dim": input_dim, "cell_in_dim": cin, "hidden_dim": hidden_dim,
        "backbone_units": backbone_units, "n_heads": n_heads,
        "head_dims": list(head_dims), "head_acts": list(head_acts),
        "has_proj": has_proj, "has_ret": has_ret,
        "header_bytes": HDR_BYTES, "payload_bytes": len(payload),
        "crc32": crc, "total_bytes": len(blob),
        "layer_off": layer_off, "layer_len": layer_len,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="model_slot0.blob")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--backbone", type=int, default=64)
    ap.add_argument("--input-dim", type=int, default=106)
    ap.add_argument("--no-ret", action="store_true")
    ap.add_argument("--no-proj", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    blob, meta = build_blob(input_dim=a.input_dim, hidden_dim=a.hidden,
                            backbone_units=a.backbone,
                            has_ret=not a.no_ret, has_proj=not a.no_proj,
                            seed=a.seed)
    with open(a.out, "wb") as f:
        f.write(blob)

    print(f"wrote {a.out}: {meta['total_bytes']} bytes "
          f"(header {meta['header_bytes']} + payload {meta['payload_bytes']})")
    print(f"  dims in={meta['input_dim']} cell_in={meta['cell_in_dim']} "
          f"hidden={meta['hidden_dim']} bb={meta['backbone_units']} "
          f"heads={meta['head_dims']} proj={meta['has_proj']} ret={meta['has_ret']}")
    print(f"  crc32={meta['crc32']:#010x}")
    for i in range(16):
        if meta["layer_len"][i]:
            print(f"  layer {i:2d}: off={meta['layer_off'][i]:6d} "
                  f"len={meta['layer_len'][i]:6d} "
                  f"({meta['layer_len'][i] // 4} floats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
