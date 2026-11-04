# -*- coding: utf-8 -*-
"""export_escalate.py — ``checkpoint_esc.pt`` → ``policy_esc.npz`` → 固件 blob（6 头）。

blob 布局与 ``pc/train/export_blob.py`` **逐字节同构**（头 2240 字节 + 按层 id
顺序紧密排列的 payload），只多了第 6 个头：

    n_heads   = 6
    head_dim  = [1, 1, 4, 8, 1, 1]
    head_act  = [1, 1, 2, 2, 1, 2]      0=linear 1=tanh 2=softmax(留 logits)

第 6 个头用 ``head_act=2``（沿用"激活交给消费方"的既有约定），**不是**新增
``act=3``。理由见 ``FIRMWARE_CHANGES.md`` §2：``cfc_model.c`` 现在对
``a > 2`` 直接 ``bad head_act`` 拒绝整个 blob，新增枚举会让**新 blob 在老固件上
一字节都加载不了**；而复用 ``act=2`` 时老固件照常加载（只是不读第 6 个头），
消费方多做一次 sigmoid。两种编码在本文件里都实现了（``--act-3``），
:func:`forward_np` 会证明它们给出**逐位相同**的 ``p_esc``。

本文件同时是**唯一的 numpy 参考实现**：:func:`parse_blob` 把 blob 解回权重，
:func:`forward_np` 逐帧递推，用来做 torch ↔ numpy 对拍（判据 max_abs_err < 1e-5）。
它只依赖 numpy —— 这是"固件加载的就是这个 blob"的唯一可执行证据。

用法::

    python export_escalate.py --checkpoint checkpoint_esc.pt --out model_slot0_esc.blob
    python export_escalate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

F32 = np.float32
HERE = Path(__file__).resolve().parent

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# --------------------------------------------------------------------------- #
# 固件常量（cfc_model.h / cfc_model.c 的 H_*）
#
# ⚠ 头长在本工程开工后被并行任务改过一次：2240 -> 2256，在末尾追加了
# ``layer_prec[16]``（每个层 1 字节精度，0=float32 1=int8，见 cfc_model.h
# "PER-LAYER PRECISION"）。**原有 18 个字段的偏移一个都没动**，所以本文件
# 只多了 ``layer_prec`` 一项、payload 基址从 2240 变成 2256。
# 本工程的 6 个头全是 float32，``layer_prec`` 恒为全 0。
# --------------------------------------------------------------------------- #
CFC_BLOB_MAGIC = 0x4D434643      # 'CFCM'
CFC_BLOB_VER = 1
CFC_HDR_BYTES = 2256
CFC_MAX_LAYERS = 16
CFC_MAX_HEADS = 8
CFC_MAX_HEAD_DIM = 16
CFC_MAX_INPUT_DIM = 256
CFC_MAX_HIDDEN_DIM = 128
CFC_MAX_BACKBONE = 256
CFC_SLOT_MAX_BYTES = 256 * 1024
FLAG_PROJ = 0x0001
FLAG_RET = 0x0002
FLAG_INT8 = 0x0004
#: 头部带 layer_prec[16] 表。pc/train/export_blob.py 现在**总是**置这一位
#: （表全 0 表示全 float32），所以这里必须跟着置，否则两份打包器只有 flags
#: 和它连带的 CRC 不同，字节级对拍会失败。
FLAG_LAYERPREC = 0x0008

HDR_FIELDS = (
    ("magic", 0, 4, "<I"),
    ("ver", 4, 2, "<H"),
    ("flags", 6, 2, "<H"),
    ("input_dim", 8, 2, "<H"),
    ("cell_in_dim", 10, 2, "<H"),
    ("hidden_dim", 12, 2, "<H"),
    ("backbone_units", 14, 2, "<H"),
    ("n_heads", 16, 2, "<H"),
    ("reserved", 18, 2, "<H"),
    ("head_dim", 20, 16, "<8H"),
    ("head_act", 36, 16, "<8H"),
    ("layer_off", 52, 64, "<16I"),
    ("layer_len", 116, 64, "<16I"),
    ("input_mean", 180, 1024, "<256f"),
    ("input_std", 1204, 1024, "<256f"),
    ("clip_sigma", 2228, 4, "<f"),
    ("ts", 2232, 4, "<f"),
    ("crc32", 2236, 4, "<I"),
    ("layer_prec", 2240, 16, "<16B"),
)
HDR_FIELD_OFFSETS = {name: off for name, off, _s, _f in HDR_FIELDS}

#: 层 id 顺序 = cfc_model.c 的 enum { L_BB_W ... L_COUNT }，顺序不能改
LAYER_IDS = ("bb_w", "bb_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b",
             "ta_w", "ta_b", "tb_w", "tb_b", "ret", "proj_w", "proj_b", "heads")

#: 头激活编码。3 = sigmoid（本工程**不**使用，见模块文档）
HEAD_ACT = {"linear": 0, "tanh": 1, "softmax": 2, "sigmoid": 3}

#: 6 个头的定义（顺序 = 固件 publish() 的下标）
HEADS = (
    ("head_yaw", 1, "tanh"),
    ("vyaw", 1, "tanh"),
    ("mode", 4, "softmax"),
    ("emote", 8, "softmax"),
    ("vx", 1, "tanh"),
    ("escalate", 1, "softmax"),      # act=2 = "激活交给消费方"，dim=1 时消费方算 sigmoid
)
N_HEADS = len(HEADS)
HEAD_DIM = tuple(d for _n, d, _a in HEADS)
HEAD_ACT_CODES = tuple(HEAD_ACT[a] for _n, _d, a in HEADS)


# --------------------------------------------------------------------------- #
# 收集 / 打包
# --------------------------------------------------------------------------- #
def _np(t):
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(t, dtype=F32))


def collect_policy(model, input_mean, input_std, ts=1.0, clip_sigma=4.0):
    """``EscalatePolicy`` -> 与 ``export_blob.collect_from_torch`` 同键的 dict。

    唯一的差别是 ``n_heads=6`` / ``head_dim`` / ``head_act`` 多一项。
    """
    cell = model.cell
    heads_w, heads_b = [], []
    for head in model.heads:
        heads_w.append(_np(head.weight))
        heads_b.append(_np(head.bias))
    p = {
        "bb_w": _np(cell.backbone[0].weight),
        "bb_b": _np(cell.backbone[0].bias),
        "ff1_w": _np(cell.ff1.weight),
        "ff1_b": _np(cell.ff1.bias),
        "ff2_w": _np(cell.ff2.weight),
        "ff2_b": _np(cell.ff2.bias),
        "ta_w": _np(cell.time_a.weight),
        "ta_b": _np(cell.time_a.bias),
        "tb_w": _np(cell.time_b.weight),
        "tb_b": _np(cell.time_b.bias),
        "ret": _np(model.ret),
        "proj_w": _np(model.proj.weight),
        "proj_b": _np(model.proj.bias),
        "head_w": np.concatenate(heads_w, axis=0),
        "head_b": np.concatenate(heads_b, axis=0),
        "sparsity_mask": np.zeros((0, 0), dtype=F32),
        "input_dim": int(model.proj.in_features),
        "cell_in_dim": int(model.proj.out_features),
        "hidden_dim": int(cell.hidden_size),
        "backbone_units": int(cell.backbone[0].weight.shape[0]),
        "n_heads": N_HEADS,
        "head_dim": np.asarray(HEAD_DIM, dtype=np.int32),
        "head_act": np.asarray(HEAD_ACT_CODES, dtype=np.int32),
        "ts": float(ts),
        "clip_sigma": float(clip_sigma),
        "input_mean": _np(input_mean).reshape(-1),
        "input_std": _np(input_std).reshape(-1),
    }
    return p


def save_policy_npz(path, policy):
    np.savez(path, **{k: (np.asarray(v) if v is not None else np.zeros(0, F32))
                      for k, v in policy.items()})
    return Path(path)


def load_policy_npz(path):
    z = np.load(path, allow_pickle=False)
    p = {k: z[k] for k in z.files}
    p["input_mean"] = _np(p["input_mean"]).reshape(-1)
    p["input_std"] = _np(p["input_std"]).reshape(-1)
    for k in ("bb_w", "bb_b"):
        if np.size(p[k]) == 0:
            p[k] = None
    return p


def pack_header(**kw):
    """按 :data:`HDR_FIELDS` 逐字段打包 ``CFC_HDR_BYTES`` 字节头（不存在结构体填充问题）。"""
    prec = kw.get("layer_prec") or (0,) * CFC_MAX_LAYERS
    vals = {
        "magic": CFC_BLOB_MAGIC, "ver": CFC_BLOB_VER, "flags": kw["flags"],
        "input_dim": kw["input_dim"], "cell_in_dim": kw["cell_in_dim"],
        "hidden_dim": kw["hidden_dim"], "backbone_units": kw["backbone_units"],
        "n_heads": kw["n_heads"], "reserved": 0,
        "head_dim": tuple(kw["head_dim"]) + (0,) * (CFC_MAX_HEADS - len(kw["head_dim"])),
        "head_act": tuple(kw["head_act"]) + (0,) * (CFC_MAX_HEADS - len(kw["head_act"])),
        "layer_off": tuple(kw["layer_off"]) + (0,) * (CFC_MAX_LAYERS - len(kw["layer_off"])),
        "layer_len": tuple(kw["layer_len"]) + (0,) * (CFC_MAX_LAYERS - len(kw["layer_len"])),
        "input_mean": tuple(float(v) for v in kw["input_mean"])
                      + (0.0,) * (CFC_MAX_INPUT_DIM - len(kw["input_mean"])),
        "input_std": tuple(float(v) for v in kw["input_std"])
                     + (1.0,) * (CFC_MAX_INPUT_DIM - len(kw["input_std"])),
        "clip_sigma": float(kw["clip_sigma"]), "ts": float(kw["ts"]),
        "crc32": 0,
        # 全 float32 -> 0。int8 层由 pc/train 那边的量化路径负责，本工程不做。
        "layer_prec": tuple(int(v) for v in prec) + (0,) * (CFC_MAX_LAYERS - len(prec)),
    }
    hdr = bytearray(CFC_HDR_BYTES)
    for name, off, size, fmt in HDR_FIELDS:
        v = vals[name]
        packed = struct.pack(fmt, *v) if isinstance(v, tuple) else struct.pack(fmt, v)
        if len(packed) != size:
            raise AssertionError("字段 %s 被打成 %d 字节，应为 %d" % (name, len(packed), size))
        hdr[off:off + size] = packed
    return hdr


def blob_crc32(blob):
    """固件口径 CRC32：整个 blob，其中 crc32 那 4 字节当 0（zlib/IEEE-802.3）。"""
    b = bytearray(blob)
    off = HDR_FIELD_OFFSETS["crc32"]
    b[off:off + 4] = b"\x00\x00\x00\x00"
    return zlib.crc32(bytes(b)) & 0xFFFFFFFF


def std_with_floor(std, floor=1e-2):
    s = _np(std).reshape(-1)
    if not np.isfinite(s).all():
        raise ValueError("input_std 里有 NaN/Inf")
    return np.maximum(s, max(float(floor), 1e-6)).astype(F32)


def build_blob(policy, *, clip_sigma=None, ts=None, std_floor=1e-2, act3=False):
    """把 policy 打成固件 blob（2240 字节头 + 按层 id 紧密排列的 payload）。

    Args:
        policy: :func:`collect_policy` 的结果。
        clip_sigma / ts: 覆盖；固件只接受 ``(0.5,16)`` / ``(1e-6,1e6)``。
        std_floor: ``input_std`` 下限（固件对 ``sd < 1e-6`` 直接拒绝整个模型）。
        act3: 为 True 时把 escalate 头的 ``head_act`` 写成 3（sigmoid 由模型算）。
            blob 其余字节完全相同 —— 用它证明两种编码在数值上等价。

    Returns:
        ``bytes``
    """
    input_dim = int(policy["input_dim"])
    cell_in = int(policy["cell_in_dim"])
    hidden = int(policy["hidden_dim"])
    backbone = int(policy["backbone_units"])
    n_heads = int(policy["n_heads"])
    head_dim = [int(v) for v in np.asarray(policy["head_dim"]).reshape(-1)]
    head_act = [int(v) for v in np.asarray(policy["head_act"]).reshape(-1)]
    if act3:
        if n_heads < 6:
            raise ValueError("act3 只对 6 头 blob 有意义")
        head_act[5] = 3
    clip_sigma = 4.0 if clip_sigma is None else float(clip_sigma)
    ts = 1.0 if ts is None else float(ts)

    if not (0 < input_dim <= CFC_MAX_INPUT_DIM):
        raise ValueError("input_dim %d 越界" % input_dim)
    if not (0 < n_heads <= CFC_MAX_HEADS) or len(head_dim) != n_heads:
        raise ValueError("n_heads/head_dim 不匹配")
    if any(not (0 < d <= CFC_MAX_HEAD_DIM) for d in head_dim):
        raise ValueError("head_dim 越界: %s" % head_dim)

    zu = backbone if backbone > 0 else (cell_in + hidden)
    has_proj = np.size(policy.get("proj_w", np.zeros(0))) > 0
    has_ret = np.size(policy.get("ret", np.zeros(0))) > 0

    layers = [None] * len(LAYER_IDS)

    def put(name, arr, expect):
        idx = LAYER_IDS.index(name)
        a = _np(arr).reshape(-1)
        if a.size != expect:
            raise ValueError("层 %s 应有 %d 个 float，实际 %d" % (name, expect, a.size))
        layers[idx] = a

    put("bb_w", policy["bb_w"], backbone * (cell_in + hidden))
    put("bb_b", policy["bb_b"], backbone)
    put("ff1_w", policy["ff1_w"], hidden * zu)
    put("ff1_b", policy["ff1_b"], hidden)
    put("ff2_w", policy["ff2_w"], hidden * zu)
    put("ff2_b", policy["ff2_b"], hidden)
    put("ta_w", policy["ta_w"], hidden * zu)
    put("ta_b", policy["ta_b"], hidden)
    put("tb_w", policy["tb_w"], hidden * zu)
    put("tb_b", policy["tb_b"], hidden)
    if has_ret:
        put("ret", policy["ret"], hidden)
    if has_proj:
        put("proj_w", policy["proj_w"], cell_in * input_dim)
        put("proj_b", policy["proj_b"], cell_in)
    put("heads", np.concatenate([_np(policy["head_w"]).reshape(-1),
                                 _np(policy["head_b"]).reshape(-1)]),
        sum(d * hidden for d in head_dim) + sum(head_dim))

    payload = bytearray()
    layer_off = [0] * len(LAYER_IDS)
    layer_len = [0] * len(LAYER_IDS)
    base = CFC_HDR_BYTES
    for i, name in enumerate(LAYER_IDS):
        if layers[i] is None:
            continue
        layer_off[i] = base + len(payload)
        data = np.ascontiguousarray(layers[i], dtype="<f4").tobytes()
        layer_len[i] = len(data)
        payload += data
    if len(payload) % 4:
        raise AssertionError("payload 不是 4 的倍数（固件要求 layer_off % 4 == 0）")

    flags = ((FLAG_PROJ if has_proj else 0) | (FLAG_RET if has_ret else 0)
             | FLAG_LAYERPREC)
    hdr = pack_header(input_dim=input_dim, cell_in_dim=cell_in, hidden_dim=hidden,
                      backbone_units=backbone, n_heads=n_heads, head_dim=head_dim,
                      head_act=head_act, layer_off=layer_off, layer_len=layer_len,
                      input_mean=_np(policy["input_mean"]).reshape(-1),
                      input_std=std_with_floor(policy["input_std"], std_floor),
                      clip_sigma=clip_sigma, ts=ts, flags=flags)
    blob = bytearray(hdr + payload)
    crc = blob_crc32(blob)
    blob[HDR_FIELD_OFFSETS["crc32"]:HDR_FIELD_OFFSETS["crc32"] + 4] = struct.pack("<I", crc)
    if len(blob) > CFC_SLOT_MAX_BYTES:
        raise ValueError("blob %d 字节超过槽上限 %d" % (len(blob), CFC_SLOT_MAX_BYTES))
    return bytes(blob)


def manifest_of(blob):
    """把 blob 头解析回来（自检 / 落盘 manifest 用）。"""
    hdr = blob[:CFC_HDR_BYTES]
    out = {}
    for name, off, _size, fmt in HDR_FIELDS:
        vals = struct.unpack_from(fmt, hdr, off)
        out[name] = vals[0] if len(vals) == 1 else list(vals)
    out["blob_len"] = len(blob)
    out["layer_names"] = list(LAYER_IDS)
    out["crc32_recomputed"] = blob_crc32(blob)
    out["crc32_ok"] = out["crc32_recomputed"] == out["crc32"]
    return out


# --------------------------------------------------------------------------- #
# numpy 参考实现（从 blob 里读权重，逐帧递推）
# --------------------------------------------------------------------------- #
def _lecun(v):
    """``1.7159 * tanh(0.666 * v)``（ncps 的 backbone 激活，0.666 不是 2/3）。"""
    return F32(1.7159) * np.tanh(F32(0.666) * v)


def _sigmoid(v):
    with np.errstate(over="ignore"):
        return F32(1.0) / (F32(1.0) + np.exp(-v))


def parse_blob(blob):
    """blob -> dict（header 字段 + 每个层的 (out,in) 权重视图）。

    Returns:
        dict：``input_dim/cell_in_dim/hidden_dim/backbone_units/n_heads/head_dim/
        head_act/input_mean/input_std/clip_sigma/ts`` 以及
        ``bb_w/bb_b/ff1_w/.../head_w[i]/head_b[i]``。
    """
    m = manifest_of(blob)
    if m["magic"] != CFC_BLOB_MAGIC:
        raise ValueError("magic 不对: 0x%08X" % m["magic"])
    if not m["crc32_ok"]:
        raise ValueError("CRC32 不对: 0x%08X != 0x%08X"
                         % (m["crc32"], m["crc32_recomputed"]))
    h = m["hidden_dim"]
    ci = m["cell_in_dim"]
    bb = m["backbone_units"]
    n_in = m["input_dim"]
    n_heads = m["n_heads"]
    head_dim = list(m["head_dim"])[:n_heads]
    head_act = list(m["head_act"])[:n_heads]
    zu = bb if bb > 0 else (ci + h)

    def layer(name):
        i = LAYER_IDS.index(name)
        off, ln = m["layer_off"][i], m["layer_len"][i]
        if ln == 0:
            return None
        return np.frombuffer(blob, dtype="<f4", count=ln // 4, offset=off)

    p = {
        "input_dim": n_in, "cell_in_dim": ci, "hidden_dim": h,
        "backbone_units": bb, "n_heads": n_heads, "head_dim": head_dim,
        "head_act": head_act,
        "input_mean": np.asarray(m["input_mean"][:n_in], dtype=F32),
        "input_std": np.asarray(m["input_std"][:n_in], dtype=F32),
        "clip_sigma": float(m["clip_sigma"]), "ts": float(m["ts"]),
    }
    p["bb_w"] = layer("bb_w").reshape(bb, ci + h) if bb > 0 else None
    p["bb_b"] = layer("bb_b") if bb > 0 else None
    for name, rows, cols in (("ff1", h, zu), ("ff2", h, zu),
                             ("ta", h, zu), ("tb", h, zu)):
        p[name + "_w"] = layer(name + "_w").reshape(rows, cols)
        p[name + "_b"] = layer(name + "_b")
    p["ret"] = layer("ret")
    p["proj_w"] = layer("proj_w").reshape(ci, n_in) if layer("proj_w") is not None else None
    p["proj_b"] = layer("proj_b")
    heads = layer("heads")
    hw = sum(d * h for d in head_dim)
    p["head_w"] = []
    p["head_b"] = []
    o = 0
    for d in head_dim:
        p["head_w"].append(heads[o:o + d * h].reshape(d, h))
        o += d * h
    for d in head_dim:
        p["head_b"].append(heads[o:o + d])
        o += d
    if o != heads.size:
        raise ValueError("heads 层长度不符：用了 %d，实际 %d" % (o, heads.size))
    return p


def apply_head_act(act, values):
    """模型侧的激活（与 ``cfc_model_forward`` 第 3 步同口径）。

    ``0`` linear、``1`` tanh、``2`` 保留原始值（softmax / sigmoid 由消费方算）、
    ``3`` sigmoid（**本工程不用**，只为证明两种编码等价）。
    """
    a = int(act)
    if a == 1:
        return np.tanh(values).astype(F32)
    if a == 3:
        return _sigmoid(values).astype(F32)
    return values.astype(F32)


def consumer_act(act, dim, values):
    """消费方对 ``head_act=2`` 的头做的归一化。

    与 ``cfc_task.c::publish`` 的既有约定一致：``dim>1`` 走 softmax，
    ``dim==1`` 走 **sigmoid**（softmax 归一化到 1 类恒等于 1.0，没有信息）。
    """
    v = np.asarray(values, dtype=F32)
    if int(act) == 2:
        if int(dim) > 1:
            e = np.exp(v - v.max())
            return (e / e.sum()).astype(F32)
        return _sigmoid(v)
    return v


def forward_np(p, obs_raw, h0=None):
    """逐帧前向（与 ``cfc_model_forward`` 同序），返回 ``(T, n_heads, MAX_HEAD_DIM)``。

    Args:
        p: :func:`parse_blob` 的输出。
        obs_raw: ``(T, input_dim)`` **原始**观测（未归一化）。
        h0: ``(hidden_dim,)`` 初值；None = 清零。

    Returns:
        ``out``：``(T, n_heads, CFC_MAX_HEAD_DIM)``，激活已按 ``head_act`` 套好
        （``act=2`` 保留 logits），未用到的槽位为 0 —— 与固件的 ``out`` 缓冲
        布局逐位对应。
    """
    o = np.asarray(obs_raw, dtype=F32)
    if o.ndim == 1:
        o = o[None, :]
    t_len = o.shape[0]
    n_in = p["input_dim"]
    h = np.zeros(p["hidden_dim"], dtype=F32) if h0 is None else np.asarray(h0, F32).copy()
    mu = p["input_mean"]
    sd = p["input_std"]
    lo = mu - F32(p["clip_sigma"]) * sd
    hi = mu + F32(p["clip_sigma"]) * sd
    ts = F32(p["ts"])
    out = np.zeros((t_len, p["n_heads"], CFC_MAX_HEAD_DIM), dtype=F32)
    lam = _sigmoid(p["ret"]) if p["ret"] is not None else None

    for k in range(t_len):
        v = o[k, :n_in]
        v = np.where(np.isfinite(v), v, mu)
        v = np.clip(v, lo, hi)
        x = ((v - mu) / sd).astype(F32)
        if p["proj_w"] is not None:
            zp = _silu(p["proj_w"] @ x + p["proj_b"]).astype(F32)
        else:
            zp = x
        # cfc_step
        zc = np.concatenate([zp, h]).astype(F32)
        if p["bb_w"] is not None:
            zb = _lecun(p["bb_w"] @ zc + p["bb_b"]).astype(F32)
        else:
            zb = zc
        ff1 = np.tanh(p["ff1_w"] @ zb + p["ff1_b"]).astype(F32)
        ff2 = np.tanh(p["ff2_w"] @ zb + p["ff2_b"]).astype(F32)
        ta = p["ta_w"] @ zb + p["ta_b"]
        tb = p["tb_w"] @ zb + p["tb_b"]
        tt = _sigmoid(ta * ts + tb).astype(F32)
        h_new = (ff1 * (F32(1.0) - tt) + tt * ff2).astype(F32)
        if lam is not None:
            h_new = np.where(lam != F32(0.0),
                             h_new + lam * (h - h_new), h_new).astype(F32)
        h = h_new
        for i in range(p["n_heads"]):
            d = p["head_dim"][i]
            acc = p["head_w"][i] @ h + p["head_b"][i]
            acc = apply_head_act(p["head_act"][i], acc)
            out[k, i, :d] = acc
    return out


def _silu(v):
    """``v * sigmoid(v)``（固件 ``silu()``）。"""
    return v * _sigmoid(v)


def p_esc_from_forward(out, p, head=5):
    """从 :func:`forward_np` 的输出里取升级概率（按 blob 声明的 ``head_act``）。

    ``act=2`` + ``dim=1`` → 消费方做 sigmoid；``act=3`` → 模型已经做过，原样读。
    """
    d = int(p["head_dim"][head])
    if d != 1:
        raise ValueError("升级头维度必须是 1，blob 里是 %d" % d)
    act = int(p["head_act"][head])
    return np.asarray(consumer_act(act, d, out[:, head, 0]), dtype=F32).reshape(-1)


# --------------------------------------------------------------------------- #
# 与 export_blob 的字节级对拍（可选，存在才跑）
# --------------------------------------------------------------------------- #
def cross_check_export_blob(policy, verbose=True):
    """用 ``pc/train/export_blob.py`` 打包同一份 policy，逐字节比较。

    这是"本文件的打包代码没有偷偷改布局"的最强证据：两份代码独立实现同一套
    偏移表，产出必须完全一致（包括 CRC32）。目录不存在时返回 ``None``。

    Returns:
        ``(ok, msg)`` 或 ``(None, msg)``
    """
    from pathlib import Path
    train_dir = Path(__file__).resolve().parent.parent / "train"
    if not (train_dir / "export_blob.py").exists():
        return None, "pc/train/export_blob.py 不存在（跳过交叉对拍）"
    import importlib.util

    spec = importlib.util.spec_from_file_location("_train_export_blob",
                                                  train_dir / "export_blob.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    theirs = mod.build_blob(policy, clip_sigma=float(policy["clip_sigma"]),
                            ts=float(policy["ts"]))
    ours = build_blob(policy)
    if theirs == ours:
        return True, ("与 pc/train/export_blob.py 逐字节一致（%d 字节，CRC 0x%08X）"
                      % (len(ours), manifest_of(ours)["crc32"]))
    diff = [i for i in range(min(len(theirs), len(ours))) if theirs[i] != ours[i]]
    return False, ("与 export_blob 有 %d 字节不同（首处 offset=%s，长度 %d vs %d）"
                   % (len(diff), diff[:1], len(theirs), len(ours)))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def describe(blob):
    m = manifest_of(blob)
    print("blob 总字节数 %d（头 %d + payload %d），CRC32 = 0x%08X（重算一致: %s）"
          % (m["blob_len"], CFC_HDR_BYTES, m["blob_len"] - CFC_HDR_BYTES,
             m["crc32"], m["crc32_ok"]))
    print("  input_dim=%d cell_in_dim=%d hidden_dim=%d backbone_units=%d n_heads=%d"
          % (m["input_dim"], m["cell_in_dim"], m["hidden_dim"],
             m["backbone_units"], m["n_heads"]))
    print("  head_dim=%s head_act=%s"
          % (m["head_dim"][:m["n_heads"]], m["head_act"][:m["n_heads"]]))
    for i, name in enumerate(LAYER_IDS):
        if m["layer_len"][i]:
            print("    layer[%2d] %-7s off=%6d len=%6d"
                  % (i, name, m["layer_off"][i], m["layer_len"][i]))


def _self_test():
    """不依赖 torch 的打包/解析自检。"""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               ("  " + detail) if detail else ""))

    rng = np.random.default_rng(0)
    hidden, ci, bb, n_in = 64, 64, 64, 106
    zu = bb
    p = {
        "bb_w": rng.normal(0, .05, (bb, ci + hidden)).astype(F32),
        "bb_b": rng.normal(0, .05, bb).astype(F32),
        "ff1_w": rng.normal(0, .05, (hidden, zu)).astype(F32),
        "ff1_b": rng.normal(0, .05, hidden).astype(F32),
        "ff2_w": rng.normal(0, .05, (hidden, zu)).astype(F32),
        "ff2_b": rng.normal(0, .05, hidden).astype(F32),
        "ta_w": rng.normal(0, .05, (hidden, zu)).astype(F32),
        "ta_b": rng.normal(0, .05, hidden).astype(F32),
        "tb_w": rng.normal(0, .05, (hidden, zu)).astype(F32),
        "tb_b": rng.normal(0, .05, hidden).astype(F32),
        "ret": np.full(hidden, -3.0, F32),
        "proj_w": rng.normal(0, .05, (ci, n_in)).astype(F32),
        "proj_b": rng.normal(0, .05, ci).astype(F32),
        "head_w": np.concatenate([rng.normal(0, .1, (d, hidden)).astype(F32)
                                  for d in HEAD_DIM]),
        "head_b": np.concatenate([rng.normal(0, .1, d).astype(F32) for d in HEAD_DIM]),
        "input_dim": n_in, "cell_in_dim": ci, "hidden_dim": hidden,
        "backbone_units": bb, "n_heads": N_HEADS,
        "head_dim": np.asarray(HEAD_DIM, np.int32),
        "head_act": np.asarray(HEAD_ACT_CODES, np.int32),
        "clip_sigma": 4.0, "ts": 1.0,
        "input_mean": np.zeros(n_in, F32), "input_std": np.ones(n_in, F32),
    }
    blob = build_blob(p)
    m = manifest_of(blob)
    ck("blob 长度 = 头 %d + payload" % CFC_HDR_BYTES,
       m["blob_len"] == len(blob) and m["blob_len"] > CFC_HDR_BYTES,
       "%d 字节" % m["blob_len"])
    ck("CRC32 自洽", m["crc32_ok"])
    ck("n_heads=6", m["n_heads"] == 6)
    ck("head_dim=[1,1,4,8,1,1]", m["head_dim"][:6] == [1, 1, 4, 8, 1, 1],
       str(m["head_dim"][:6]))
    ck("head_act=[1,1,2,2,1,2]", m["head_act"][:6] == [1, 1, 2, 2, 1, 2],
       str(m["head_act"][:6]))
    parsed = parse_blob(blob)
    ck("解析出的权重与打包前一致",
       np.allclose(parsed["bb_w"], p["bb_w"], atol=0) and
       np.allclose(parsed["head_w"][5], p["head_w"][-1], atol=0))
    blob3 = build_blob(p, act3=True)
    ck("act3 版本只改 head_act[5] 与 CRC",
       manifest_of(blob3)["head_act"][5] == 3 and
       len(blob3) == len(blob))
    out = forward_np(parsed, np.zeros((5, n_in), F32))
    ck("numpy 前向输出形状 (T,6,16)", out.shape == (5, 6, CFC_MAX_HEAD_DIM), str(out.shape))
    ck("escalate 列恒 0 输入下非 NaN", np.isfinite(out[:, 5, 0]).all())
    out3 = forward_np(parse_blob(blob3), np.zeros((5, n_in), F32))
    ck("act=2/act=3 两个 blob 的 p_esc 逐位相同",
       np.array_equal(p_esc_from_forward(out, parsed),
                      p_esc_from_forward(out3, parse_blob(blob3))))
    pe = p_esc_from_forward(out, parsed)
    ck("p_esc ∈ (0,1)", bool(((pe > 0) & (pe < 1)).all()))
    ok_ck, msg = cross_check_export_blob(p)
    if ok_ck is None:
        print("  [SKIP] 与 pc/train/export_blob.py 交叉对拍：%s" % msg)
    else:
        ck("与 pc/train/export_blob.py 逐字节一致", ok_ck, msg)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="导出 6 头 CfC blob（含 numpy 参考实现）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", default=str(HERE / "checkpoint_esc.pt"))
    ap.add_argument("--npz-out", default=str(HERE / "policy_esc.npz"))
    ap.add_argument("--out", default=str(HERE / "model_slot0_esc.blob"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()

    import torch

    import esc_model as M

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = M.EscalatePolicy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                             hidden=cfg["hidden"], backbone_units=cfg["backbone_units"],
                             ret_init=cfg.get("ret_init", -3.0))
    model.load_state_dict(ck["model"])
    model.eval()
    pol = collect_policy(model, ck["input_mean"], ck["input_std"])
    save_policy_npz(args.npz_out, pol)
    blob = build_blob(pol)
    Path(args.out).write_bytes(blob)
    m = manifest_of(blob)
    (HERE / "blob_manifest_esc.json").write_text(
        json.dumps({k: v for k, v in m.items() if k != "layer_names"}, indent=2),
        encoding="utf-8")
    print("[export] %s（%d 字节，CRC 0x%08X）" % (args.out, len(blob), m["crc32"]))
    describe(blob)
    ok_ck, msg = cross_check_export_blob(pol)
    print("[export] 交叉对拍: %s" % (msg if ok_ck is None else
                                   ("PASS " + msg if ok_ck else "FAIL " + msg)))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
