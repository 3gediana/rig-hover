# -*- coding: utf-8 -*-
"""export_blob.py — ``policy.npz`` → 固件 ``model_slot0.blob``（唯一权威：cfc_model.h）。

blob 的字节布局**不是** C 结构体，而是 ``cfc_model.c`` 里一组 ``#define H_*`` 的
硬偏移（``blob_parse()`` 用 ``rd_u32(blob + H_MAGIC)`` 这样读）。所以本文件：

* 用 :data:`HDR_FIELDS` 显式写下每个字段的 (offset, size, type)，不靠结构体对齐；
* ``--check-offsets`` 会同时核对三份来源，任何一份不一致就 FAIL：

  1. ``dump_offsets.c`` 里 ``offsetof(struct model_hdr_packed, ...)`` 的真实值
     （用 C 编译器算，不是猜的）；
  2. 固件 ``cfc_model.c`` 里的 ``#define H_*``（解析源码，固件改了这里会立刻炸）；
  3. 本文件的 :data:`HDR_FIELDS`（打包代码实际用的）。

CRC32 必须是固件那一份：``cfc_model.c`` 的 ``cfc_crc32()`` 是 zlib/IEEE-802.3
口径（poly 0xEDB88320，初值取反、末尾取反），且**先把它自己那 4 字节当 0** 再算。
Python 的 ``zlib.crc32`` 与它逐位一致。

用法::

    python export_blob.py --npz policy.npz --out model_slot0.blob
    python export_blob.py --check-offsets
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流没有 reconfigure
        pass

HERE = Path(__file__).resolve().parent
FIRMWARE_DIR = HERE.parent.parent / "rig_omni" / "main" / "boards" / "hover"
CFC_MODEL_C = FIRMWARE_DIR / "cfc_model.c"
CFC_MODEL_H = FIRMWARE_DIR / "cfc_model.h"

F32 = np.float32

# --------------------------------------------------------------------------- #
# 固件常量（cfc_model.h）
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
#: bit2：至少一个权重矩阵是 int8。逐层精度看 `layer_prec[]`（见 cfc_model.h）。
FLAG_INT8 = 0x0004
#: bit3：头里带 layer_prec[] 表。**必须置位**：老 blob 的那 16 个字节是 bb_w
#: 载荷的开头，不是全 0 的表，固件靠这一位决定要不要读它。
FLAG_LAYERPREC = 0x0008

#: layer_prec[] 的取值。
PREC_F32 = 0
PREC_INT8 = 1

#: 可以被存成 int8 的层（= cfc_model.c 的 layer_can_be_int8()）。偏置、ret 和
#: heads 永远 float32：heads 只占总参数 3%，int8 化它却让头误差涨 2.8 倍。
INT8_CAPABLE = ("bb_w", "ff1_w", "ff2_w", "ta_w", "tb_w", "proj_w")

#: ``--int8`` 的默认配方：六个权重矩阵 int8，其余 float32。这是 cfc_engine.c
#: 的 W8A8 内核路径（权重 int8、激活单 int8、输出层 fp32）。
INT8_RECIPE = {name: PREC_INT8 for name in INT8_CAPABLE}

#: 头里每个字段的 (名字, 字节偏移, 字节长度, 打包格式)。
#: 偏移直接抄 cfc_model.h 的字节表，并由 ``--check-offsets`` 与编译器/固件三方核对。
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
HDR_FIELD_OFFSETS = {name: off for name, off, _size, _fmt in HDR_FIELDS}

#: 层 id 顺序 = cfc_model.c 的 enum { L_BB_W ... L_COUNT }，**顺序不能改**。
LAYER_IDS = (
    "bb_w", "bb_b",
    "ff1_w", "ff1_b",
    "ff2_w", "ff2_b",
    "ta_w", "ta_b",
    "tb_w", "tb_b",
    "ret", "proj_w", "proj_b", "heads",
)
assert len(LAYER_IDS) == 14 <= CFC_MAX_LAYERS

#: 头部激活：0=linear 1=tanh 2=softmax（cfc_model.h）
HEAD_ACT = {"linear": 0, "tanh": 1, "softmax": 2}
#: 五个头的定义（顺序 = 固件 publish() 的下标）：(名字, 维度, 激活)
HEADS = (
    ("head_yaw", 1, "tanh"),     # 0  -> tanh, 调用方乘 head_max 度
    ("vyaw", 1, "tanh"),         # 1  -> tanh, 调用方乘 vyaw_max 度/秒
    ("mode", 4, "softmax"),      # 2  -> softmax 4 类，调用方 argmax
    ("emote", 8, "softmax"),     # 3  -> softmax 8 类
    ("vx", 1, "tanh"),           # 4  -> tanh, 调用方乘 vx_max 米/秒
)
N_HEADS = len(HEADS)

#: policy.npz 里必须有的 cell 权重键（与 cfc_engine.h 的 cfc_cell_t 一一对应）
CELL_KEYS = ("bb_w", "bb_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b",
             "ta_w", "ta_b", "tb_w", "tb_b", "ret")


# --------------------------------------------------------------------------- #
# 头打包
# --------------------------------------------------------------------------- #
def pack_header(*, input_dim, cell_in_dim, hidden_dim, backbone_units, n_heads,
                head_dim, head_act, layer_off, layer_len, layer_prec, input_mean,
                input_std, clip_sigma, ts, flags):
    """按 :data:`HDR_FIELDS` 逐字段打包 2240 字节头。

    所有字段都是**显式** struct.pack，不存在 C 结构体填充问题；
    ``--check-offsets`` 再拿 C 编译器算出来的 ``offsetof`` 兜一层。

    Returns:
        ``bytearray``，长度恒为 :data:`CFC_HDR_BYTES`。
    """
    hdr = bytearray(CFC_HDR_BYTES)
    vals = {
        "magic": CFC_BLOB_MAGIC,
        "ver": CFC_BLOB_VER,
        "flags": flags,
        "input_dim": input_dim,
        "cell_in_dim": cell_in_dim,
        "hidden_dim": hidden_dim,
        "backbone_units": backbone_units,
        "n_heads": n_heads,
        "reserved": 0,
        "head_dim": tuple(head_dim) + (0,) * (CFC_MAX_HEADS - len(head_dim)),
        "head_act": tuple(head_act) + (0,) * (CFC_MAX_HEADS - len(head_act)),
        "layer_off": tuple(layer_off) + (0,) * (CFC_MAX_LAYERS - len(layer_off)),
        "layer_len": tuple(layer_len) + (0,) * (CFC_MAX_LAYERS - len(layer_len)),
        "layer_prec": tuple(layer_prec) + (0,) * (CFC_MAX_LAYERS - len(layer_prec)),
        "input_mean": tuple(float(v) for v in input_mean)
                      + (0.0,) * (CFC_MAX_INPUT_DIM - len(input_mean)),
        "input_std": tuple(float(v) for v in input_std)
                     + (1.0,) * (CFC_MAX_INPUT_DIM - len(input_std)),
        "clip_sigma": float(clip_sigma),
        "ts": float(ts),
        "crc32": 0,          # 先写 0，算完 CRC 再回填
    }
    for name, off, size, fmt in HDR_FIELDS:
        val = vals[name]
        packed = struct.pack(fmt, *val) if isinstance(val, tuple) else struct.pack(fmt, val)
        if len(packed) != size:
            raise AssertionError("字段 %s 被打成 %d 字节，应为 %d" % (name, len(packed), size))
        hdr[off:off + size] = packed
    return hdr


def blob_crc32(blob):
    """固件口径的 CRC32：**整个 blob**，其中 crc32 那 4 字节当 0。

    与 ``cfc_model.c`` 的 ``blob_crc()`` 逐步同构：前半段 + 4 个 0 + 后半段。
    """
    b = bytearray(blob)
    off = HDR_FIELD_OFFSETS["crc32"]
    if len(b) >= off + 4:
        b[off:off + 4] = b"\x00\x00\x00\x00"
    return zlib.crc32(bytes(b)) & 0xFFFFFFFF


# --------------------------------------------------------------------------- #
# 权重收集（torch → npz）
# --------------------------------------------------------------------------- #
def _np(t):
    """torch tensor（或 array-like）→ contiguous float32 numpy。"""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(t, dtype=F32))


def fold_sparsity(w, mask):
    """把 ncps 的 ``sparsity_mask`` 折进权重（**只折 ff1/ff2**）。

    照抄 ``cfc_host/test_parity_ncps.py:params_from_masked_cell`` —— 那份配方
    已经被 ``test_parity.py [2b]`` 用三方差 0.6 的精度验证过：

    * ``CfCCell`` 入参 mask 形状是 ``(in, out)``，内部 ``mask.T`` 后与
      ``weight(out, in)`` 逐元素相乘，所以这里乘的 ``mask`` 必须是 ``(out, in)``；
    * ``time_a``/``time_b`` **不受 mask 影响**，把它们也乘上会让隐状态错 ~0.6，
      **而三方对拍仍然自洽**（ncps/numpy/C 都错得一样），所以只能照配方来。

    Args:
        w: ``(out, in)`` 权重。
        mask: ``(out, in)`` 掩码或 None/空。

    Returns:
        ``(out, in)`` float32。
    """
    w = _np(w)
    if mask is None or np.size(mask) == 0:
        return w
    m = _np(mask)
    if m.shape != w.shape:
        raise ValueError("mask 形状 %s 与权重 %s 不一致（mask 必须是 (out, in)，"
                         "如果拿到的是 (in, out) 要先转置）" % (m.shape, w.shape))
    return np.ascontiguousarray(w * m, dtype=F32)


def collect_from_torch(model, mask=None):
    """把 ``train_cfc.Policy`` 的权重收成 npz 用的 dict。

    Args:
        model: 有 ``proj``/``cell``/``ret``/``heads`` 属性的 ``nn.Module``。
        mask: 可选 ``(cell_in+hidden, hidden)`` 的 ncps mask（(in, out) 朝向）。

    Returns:
        dict（全是 np.ndarray/标量），键即 :data:`CELL_KEYS` + ``proj_w``… ``head_w``。
    """
    cell = model.cell
    m = None
    if mask is not None and np.size(mask) > 0:
        m = _np(mask).T                   # (in, out) -> (out, in)，权重同形
    out = {
        "bb_w": _np(cell.backbone[0].weight) if len(cell.backbone) > 0 else None,
        "bb_b": _np(cell.backbone[0].bias) if len(cell.backbone) > 0 else None,
        "ff1_w": fold_sparsity(cell.ff1.weight, m),
        "ff1_b": _np(cell.ff1.bias),
        "ff2_w": fold_sparsity(cell.ff2.weight, m),
        "ff2_b": _np(cell.ff2.bias),
        "ta_w": _np(cell.time_a.weight),     # 稠密，绝不折 mask
        "ta_b": _np(cell.time_a.bias),
        "tb_w": _np(cell.time_b.weight),
        "tb_b": _np(cell.time_b.bias),
        "ret": _np(model.ret),
        "proj_w": _np(model.proj.weight),
        "proj_b": _np(model.proj.bias),
    }
    heads_w, heads_b = [], []
    for head in model.heads:
        heads_w.append(_np(head.weight))
        heads_b.append(_np(head.bias))
    out["head_w"] = np.concatenate(heads_w, axis=0)
    out["head_b"] = np.concatenate(heads_b, axis=0)
    out["sparsity_mask"] = np.zeros((0, 0), dtype=F32) if m is None else m
    out["input_dim"] = int(model.proj.in_features)
    out["cell_in_dim"] = int(model.proj.out_features)
    out["hidden_dim"] = int(cell.hidden_size)
    out["backbone_units"] = int(out["bb_w"].shape[0]) if out["bb_w"] is not None else 0
    out["n_heads"] = len(HEADS)
    out["head_dim"] = np.asarray([d for _n, d, _a in HEADS], dtype=np.int32)
    out["head_act"] = np.asarray([HEAD_ACT[a] for _n, _d, a in HEADS], dtype=np.int32)
    out["ts"] = 1.0
    out["clip_sigma"] = 4.0
    return out


def save_policy_npz(path, policy, input_mean, input_std):
    """写 ``policy.npz``（导出脚本与训练脚本共用这一份键名定义）。"""
    p = dict(policy)
    p["input_mean"] = _np(input_mean)
    p["input_std"] = _np(input_std)
    np.savez(path, **{k: (np.asarray(v) if v is not None else np.zeros(0, F32))
                      for k, v in p.items()})
    return Path(path)


def load_policy_npz(path):
    """读 ``policy.npz`` → dict（空数组还原成 None 交给下游判断）。"""
    z = np.load(path, allow_pickle=False)
    p = {k: z[k] for k in z.files}
    p["input_mean"] = _np(p["input_mean"]).reshape(-1)
    p["input_std"] = _np(p["input_std"]).reshape(-1)
    for k in ("bb_w", "bb_b"):
        if p[k].size == 0:
            p[k] = None
    return p


# --------------------------------------------------------------------------- #
# 组 blob
# --------------------------------------------------------------------------- #
def std_with_floor(std, floor=1e-2):
    """给 ``input_std`` 加下限。

    固件 ``blob_parse()`` 对 ``sd < 1e-6`` 直接 "bad input statistics" 拒绝整个
    模型；而观测里有几维天然恒定（``cfc_task.c`` 写死 0 的 76/77、常量的
    robot_state/control_mode/gaze_danger）。这些维在部署时也恒为常数，归一化后
    恒等于 0，加下限不影响行为，只让加载器接受。
    """
    s = _np(std).reshape(-1)
    if not np.isfinite(s).all():
        raise ValueError("input_std 里有 NaN/Inf")
    floor = float(floor)
    if floor < 1e-6:
        floor = 1e-6
    return np.maximum(s, floor).astype(F32)


def _present(v):
    """npz 里"缺席"的层存成空数组，:func:`build_blob` 用这个判断。"""
    return v is not None and np.size(v) > 0


def quantize_per_channel(w):
    """逐输出通道对称量化：``W[i][:] ~= q[i][:] * scale[i]``。

    ``scale[i] = max|W[i][:]| / 127``，``q`` 是 int8 且夹在 [-127, 127]（不留
    -128，这样累加起来不会因为某一项爆发而溢出 int32）。四舍五入按"远离零"，
    与固件 ``cfc_engine.c`` 的 ``q8_quantize2()`` 逐位一致；全零行给 scale=1.0
    且 q 恒 0，解量化出来仍是 0，固件侧也不会除零。

    Args:
        w: ``(out_dim, k)`` float32。

    Returns:
        ``(q, scale)``：``q`` int8 ``(out_dim, k)``，``scale`` float32
        ``(out_dim,)``。
    """
    w = np.asarray(w, dtype=F32)
    if w.ndim != 2:
        raise ValueError("quantize_per_channel 需要 2-D 权重，收到 %s" % (w.shape,))
    amax = np.max(np.abs(w), axis=1)
    scale = np.where(amax > 0.0, amax / 127.0, 1.0).astype(F32)
    r = w / scale[:, None]
    q = np.trunc(r + np.copysign(np.float32(0.5), r))
    q = np.clip(q, -127.0, 127.0).astype(np.int8)
    return q, scale


def dequantize_per_channel(q, scale):
    """:func:`quantize_per_channel` 的逆运算（自检/误差报告用）。"""
    return (np.asarray(q, dtype=F32).reshape(len(scale), -1)
            * np.asarray(scale, dtype=F32)[:, None])


def int8_out_dim(name, *, input_dim, cell_in, hidden, backbone):
    """每个 int8 层有多少"输出通道"（= 有多少个 scale）。"""
    if name == "bb_w":
        return backbone
    if name == "proj_w":
        return cell_in
    return hidden


def int8_layer_bytes(elements):
    """int8 权重段长度：元素数向上取 4 的倍数（固件要求 layer_off/len 都 %4）。"""
    return (int(elements) + 3) & ~3


def resolve_prec(policy_names, int8, prec):
    """决定每一层的存储精度（``layer_prec[]``）。

    Args:
        policy_names: 这份 blob 里真实存在的层 id 列表。
        int8: ``--int8``：对 :data:`INT8_RECIPE`（六个权重矩阵）置 int8。
        prec: 显式覆盖，``{层名: PREC_F32|PREC_INT8}``；用于逐层精度实验。

    Returns:
        ``(layer_prec, flags_bit2)``。
    """
    table = {name: PREC_F32 for name in LAYER_IDS}
    if int8:
        table.update(INT8_RECIPE)
    if prec:
        for name, value in prec.items():
            if name not in LAYER_IDS:
                raise ValueError("未知层 %r（可用: %s）" % (name, ", ".join(LAYER_IDS)))
            if int(value) not in (PREC_F32, PREC_INT8):
                raise ValueError("层 %s 的精度只能是 0(float32) 或 1(int8)" % name)
            table[name] = int(value)
    for name, value in table.items():
        if value == PREC_INT8:
            if name not in INT8_CAPABLE:
                raise ValueError("层 %s 不能存成 int8（固件只允许 %s）"
                                 % (name, ", ".join(INT8_CAPABLE)))
            if name not in policy_names:
                raise ValueError("层 %s 标了 int8 但这份 policy 里没有" % name)
    return [table[name] for name in LAYER_IDS], int(any(table.values()))


def build_blob(policy, *, clip_sigma=None, ts=None, std_floor=1e-2, int8=False,
               prec=None):
    """把一份 policy（dict）打成固件 blob。

    Args:
        policy: :func:`collect_from_torch` 或 :func:`load_policy_npz` 的结果。
        clip_sigma: 覆盖 npz 里的值；固件只接受 (0.5, 16)，默认 4.0。
        ts: CfC 时间步长；固件只接受 (1e-6, 1e6)，默认 1.0。
        std_floor: ``input_std`` 下限，见 :func:`std_with_floor`。
        int8: True 时把 :data:`INT8_RECIPE` 六个权重矩阵量化成 int8 + 逐通道
            float32 scale，并在 ``layer_prec`` 里逐层记下、flags 置
            :data:`FLAG_INT8`；偏置、ret 和 heads 仍然 float32。
        prec: 逐层覆盖精度（见 :func:`resolve_prec`），用于精度实验；和
            ``int8`` 同时给时以 ``prec`` 为准。

    Returns:
        ``bytes``：2256 字节头 + 按层 id 顺序排布的权重（int8 层带 scale 尾巴）。
    """
    input_dim = int(policy["input_dim"])
    cell_in = int(policy["cell_in_dim"])
    hidden = int(policy["hidden_dim"])
    backbone = int(policy["backbone_units"])
    n_heads = int(policy["n_heads"])
    head_dim = [int(v) for v in np.asarray(policy["head_dim"]).reshape(-1)]
    head_act = [int(v) for v in np.asarray(policy["head_act"]).reshape(-1)]
    clip_sigma = 4.0 if clip_sigma is None else float(clip_sigma)
    ts = 1.0 if ts is None else float(ts)

    if not (0 < input_dim <= CFC_MAX_INPUT_DIM):
        raise ValueError("input_dim %d 越界" % input_dim)
    if not (0 < cell_in <= CFC_MAX_INPUT_DIM):
        raise ValueError("cell_in_dim %d 越界" % cell_in)
    if not (0 < hidden <= CFC_MAX_HIDDEN_DIM):
        raise ValueError("hidden_dim %d 越界" % hidden)
    if not (0 <= backbone <= CFC_MAX_BACKBONE):
        raise ValueError("backbone_units %d 越界" % backbone)
    if not (0 < n_heads <= CFC_MAX_HEADS) or len(head_dim) != n_heads:
        raise ValueError("n_heads/head_dim 不匹配")
    if any(not (0 < d <= CFC_MAX_HEAD_DIM) for d in head_dim):
        raise ValueError("head_dim 越界: %s" % head_dim)

    zu = backbone if backbone > 0 else (cell_in + hidden)
    has_proj = _present(policy.get("proj_w"))    # 我们总是带投影：106 -> 64 + SiLU
    has_ret = _present(policy.get("ret"))

    # ---- 每一层的 float32 数据，按层 id 顺序 ----
    layers = [None] * len(LAYER_IDS)

    def put(name, arr, expect):
        idx = LAYER_IDS.index(name)
        a = _np(arr).reshape(-1)
        if a.size != expect:
            raise ValueError("层 %s 应有 %d 个 float，实际 %d（形状 %s）"
                             % (name, expect, a.size, np.asarray(arr).shape))
        layers[idx] = a

    if backbone > 0:
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
    head_w = _np(policy["head_w"]).reshape(-1)
    head_b = _np(policy["head_b"]).reshape(-1)
    put("heads", np.concatenate([head_w, head_b]),
        sum(d * hidden for d in head_dim) + sum(head_dim))

    # ---- 逐层精度 ----
    present = [name for i, name in enumerate(LAYER_IDS) if layers[i] is not None]
    layer_prec, any_int8 = resolve_prec(present, int8, prec)

    # ---- 排布 payload：按层 id 顺序紧密排列 ----
    # int8 层：int8 权重（长度 = 元素数，%4 对齐）紧跟 float32 scale[out_dim]。
    payload = bytearray()
    layer_off = [0] * len(LAYER_IDS)
    layer_len = [0] * len(LAYER_IDS)
    base = CFC_HDR_BYTES
    for i, name in enumerate(LAYER_IDS):
        if layers[i] is None:
            continue
        layer_off[i] = base + len(payload)
        if layer_prec[i] == PREC_INT8:
            n_out = int8_out_dim(name, input_dim=input_dim, cell_in=cell_in,
                                 hidden=hidden, backbone=backbone)
            arr = layers[i]
            if arr.size % n_out:
                raise ValueError("层 %s 的 %d 个元素除不尽 %d 个输出通道"
                                 % (name, arr.size, n_out))
            q, sc = quantize_per_channel(arr.reshape(n_out, -1))
            wbytes = q.tobytes()
            layer_len[i] = int8_layer_bytes(len(wbytes))
            payload += wbytes + b"\x00" * (layer_len[i] - len(wbytes))
            payload += np.ascontiguousarray(sc, dtype="<f4").tobytes()
        else:
            data = np.ascontiguousarray(layers[i], dtype="<f4").tobytes()
            layer_len[i] = len(data)
            payload += data
    if len(payload) % 4:
        raise AssertionError("payload 不是 4 的倍数（固件要求 layer_off % 4 == 0）")

    flags = ((FLAG_PROJ if has_proj else 0) | (FLAG_RET if has_ret else 0)
             | (FLAG_INT8 if any_int8 else 0) | FLAG_LAYERPREC)

    hdr = pack_header(
        input_dim=input_dim, cell_in_dim=cell_in, hidden_dim=hidden,
        backbone_units=backbone, n_heads=n_heads,
        head_dim=head_dim, head_act=head_act,
        layer_off=layer_off, layer_len=layer_len, layer_prec=layer_prec,
        input_mean=_np(policy["input_mean"]).reshape(-1),
        input_std=std_with_floor(policy["input_std"], std_floor),
        clip_sigma=clip_sigma, ts=ts, flags=flags)

    blob = bytearray(hdr + payload)
    crc = blob_crc32(blob)
    blob[HDR_FIELD_OFFSETS["crc32"]:HDR_FIELD_OFFSETS["crc32"] + 4] = \
        struct.pack("<I", crc)
    if len(blob) > CFC_SLOT_MAX_BYTES:
        raise ValueError("blob %d 字节超过槽上限 %d" % (len(blob), CFC_SLOT_MAX_BYTES))
    return bytes(blob)


def manifest_of(blob):
    """把 blob 头解析回来（自检用；与固件解析器互相印证）。

    ``layer_prec`` 是**解析后**的逐层精度：不带 layer_prec 表的老 blob 按固件的
    同一条规则补成"六个权重矩阵 int8"，并把 ``legacy_int8`` 置 True。
    """
    hdr = blob[:CFC_HDR_BYTES]
    out = {}
    for name, off, _size, fmt in HDR_FIELDS:
        vals = struct.unpack_from(fmt, hdr, off)
        out[name] = vals[0] if len(vals) == 1 else list(vals)
    out["blob_len"] = len(blob)
    out["layer_off"] = [int(v) for v in out["layer_off"]]
    out["layer_len"] = [int(v) for v in out["layer_len"]]
    out["layer_names"] = list(LAYER_IDS)
    out["int8"] = (out["flags"] & FLAG_INT8) != 0
    out["has_prec_table"] = (out["flags"] & FLAG_LAYERPREC) != 0
    if out["has_prec_table"]:
        prec = [int(v) for v in out["layer_prec"]]
    else:
        # 老 blob：偏移 2240 处是载荷，不是表
        prec = [0] * CFC_MAX_LAYERS
        if out["int8"]:
            for name in INT8_CAPABLE:
                prec[LAYER_IDS.index(name)] = PREC_INT8
    out["layer_prec"] = prec
    out["legacy_int8"] = bool(out["int8"] and not out["has_prec_table"])
    out["int8_layers"] = [name for i, name in enumerate(LAYER_IDS)
                          if prec[i] == PREC_INT8 and out["layer_len"][i]]
    out["crc32_recomputed"] = blob_crc32(blob)
    out["crc32_ok"] = out["crc32_recomputed"] == out["crc32"]
    return out


def describe(blob):
    """打印一段人能核对的摘要（字节数、头大小、每个字段的偏移）。"""
    m = manifest_of(blob)
    print("blob 总字节数 %d（头 %d + payload %d），CRC32 = 0x%08X（重算一致: %s）"
          % (m["blob_len"], CFC_HDR_BYTES, m["blob_len"] - CFC_HDR_BYTES,
             m["crc32"], m["crc32_ok"]))
    print("  magic=0x%08X ver=%d flags=0x%04X(WITH_PROJ=%d WITH_RET=%d WITH_INT8=%d)"
          % (m["magic"], m["ver"], m["flags"], m["flags"] & 1,
             (m["flags"] >> 1) & 1, (m["flags"] >> 2) & 1))
    print("  input_dim=%d cell_in_dim=%d hidden_dim=%d backbone_units=%d n_heads=%d"
          % (m["input_dim"], m["cell_in_dim"], m["hidden_dim"],
             m["backbone_units"], m["n_heads"]))
    print("  head_dim=%s head_act=%s"
          % (m["head_dim"][:m["n_heads"]], m["head_act"][:m["n_heads"]]))
    print("  clip_sigma=%.3f ts=%.3f" % (m["clip_sigma"], m["ts"]))
    print("  字段偏移: " + "  ".join("%s=%d" % (n, o) for n, o, _s, _f in HDR_FIELDS))
    for i, name in enumerate(LAYER_IDS):
        if m["layer_len"][i]:
            q8 = m["layer_prec"][i] == PREC_INT8
            unit = "int8+scale" if q8 else "float"
            print("    layer[%2d] %-7s off=%6d len=%6d (%s)"
                  % (i, name, m["layer_off"][i], m["layer_len"][i], unit))


# --------------------------------------------------------------------------- #
# 三方偏移核对
# --------------------------------------------------------------------------- #
def firmware_offsets(cfc_model_c=CFC_MODEL_C):
    """解析固件 ``cfc_model.c`` 的 ``#define H_*``，返回 {字段名: 偏移}。"""
    text = Path(cfc_model_c).read_text(encoding="utf-8", errors="replace")
    alias = {
        "MAGIC": "magic", "VER": "ver", "FLAGS": "flags", "IN_DIM": "input_dim",
        "CELL_IN": "cell_in_dim", "HID": "hidden_dim", "BB": "backbone_units",
        "NHEADS": "n_heads", "RESERVED": "reserved", "HEADDIM": "head_dim",
        "HEADACT": "head_act", "LAYEROFF": "layer_off", "LAYERLEN": "layer_len",
        "MEAN": "input_mean", "STD": "input_std", "CLIPSIG": "clip_sigma",
        "TS": "ts", "CRC": "crc32", "PREC": "layer_prec",
    }
    found = {}
    for name, val in re.findall(r"#define\s+H_([A-Z_]+)\s+(\d+)u?", text):
        if name in alias:
            found[alias[name]] = int(val)
    return found


def firmware_constants(cfc_model_h=CFC_MODEL_H):
    """解析 ``cfc_model.h`` 里的几个关键常量。"""
    text = Path(cfc_model_h).read_text(encoding="utf-8", errors="replace")
    out = {}
    for name in ("CFC_BLOB_MAGIC", "CFC_BLOB_VER", "CFC_HDR_BYTES",
                 "CFC_MAX_LAYERS", "CFC_MAX_HEADS", "CFC_MAX_HEAD_DIM",
                 "CFC_MAX_INPUT_DIM", "CFC_MAX_HIDDEN_DIM", "CFC_MAX_BACKBONE",
                 "CFC_SLOT_MAX_BYTES"):
        m = re.search(r"#define\s+%s\s+(.+?)\s*(?:/\*|$)" % name, text, re.M)
        if not m:
            continue
        expr = m.group(1).strip()
        # 去掉 C 的整数后缀（0x4D434643u）与注释，剩下的都是纯常量算式
        expr = re.sub(r"[uUlLfF]+\s*$", "", expr).strip()
        try:
            out[name] = int(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 — 只有常量算式
        except Exception:  # noqa: BLE001 — 表达式里出现未知宏就跳过，交给 dump_offsets 兜
            pass
    return out


def check_offsets(dump_offsets_exe=None, verbose=True):
    """三方核对头字段偏移：固件 ``H_*`` / 编译器 ``offsetof`` / 本文件。

    Returns:
        ``(ok, problems)``
    """
    problems = []
    fw = firmware_offsets()
    for name, off, size, _fmt in HDR_FIELDS:
        if name in fw and fw[name] != off:
            problems.append("固件 cfc_model.c 的 H_%s=%d，本文件用 %d"
                            % (name.upper(), fw[name], off))
        if off + size > CFC_HDR_BYTES:
            problems.append("字段 %s 越过 CFC_HDR_BYTES" % name)
    missing = [n for n, _o, _s, _f in HDR_FIELDS if n not in fw]
    if missing:
        problems.append("固件 cfc_model.c 里没解析到这些字段: %s" % missing)

    consts = firmware_constants()
    for name, ours in (("CFC_BLOB_MAGIC", CFC_BLOB_MAGIC), ("CFC_BLOB_VER", CFC_BLOB_VER),
                       ("CFC_HDR_BYTES", CFC_HDR_BYTES),
                       ("CFC_MAX_HEADS", CFC_MAX_HEADS),
                       ("CFC_MAX_HEAD_DIM", CFC_MAX_HEAD_DIM),
                       ("CFC_MAX_INPUT_DIM", CFC_MAX_INPUT_DIM),
                       ("CFC_MAX_HIDDEN_DIM", CFC_MAX_HIDDEN_DIM),
                       ("CFC_MAX_BACKBONE", CFC_MAX_BACKBONE),
                       ("CFC_MAX_LAYERS", CFC_MAX_LAYERS),
                       ("CFC_SLOT_MAX_BYTES", CFC_SLOT_MAX_BYTES)):
        if name in consts and consts[name] != ours:
            problems.append("固件 %s=%d，本文件写的是 %d" % (name, consts[name], ours))

    if dump_offsets_exe is not None:
        exe = Path(dump_offsets_exe)
        if not exe.exists():
            problems.append("找不到 dump_offsets 可执行文件: %s" % exe)
        else:
            out = subprocess.run([str(exe)], capture_output=True, text=True,
                                 encoding="mbcs" if sys.platform == "win32" else "utf-8",
                                 errors="replace")
            if out.returncode != 0:
                problems.append("dump_offsets 退出码 %d: %s"
                                % (out.returncode, (out.stdout or "") + (out.stderr or "")))
            dumped = {}
            natural = {}
            for line in (out.stdout or "").splitlines():
                m = re.match(r"\s*offsetof\s+([\w.]+)\s*=\s*(\d+)", line)
                if m:
                    if m.group(1).startswith("natural."):
                        natural[m.group(1).split(".", 1)[1]] = int(m.group(2))
                    else:
                        dumped[m.group(1)] = int(m.group(2))
                m = re.match(r"\s*sizeof\s+(\w+)\s*=\s*(\d+)", line)
                if m:
                    dumped["sizeof:" + m.group(1)] = int(m.group(2))
            for name, off, _size, _fmt in HDR_FIELDS:
                if name not in dumped:
                    problems.append("dump_offsets 没有输出字段 %s" % name)
                elif dumped[name] != off:
                    problems.append("offsetof(%s)=%d，本文件用 %d"
                                    % (name, dumped[name], off))
                if name not in natural:
                    problems.append("dump_offsets 没有输出自然对齐字段 natural.%s" % name)
                elif natural[name] != off:
                    problems.append("自然对齐 offsetof(%s)=%d 与打包形式 %d 不同（填充！）"
                                    % (name, natural[name], off))
            for key, want in (("sizeof:model_hdr_packed", CFC_HDR_BYTES),
                              ("sizeof:model_hdr_natural", CFC_HDR_BYTES)):
                if key not in dumped:
                    problems.append("dump_offsets 没有输出 %s" % key)
                elif dumped[key] != want:
                    problems.append("%s=%d，应为 %d" % (key, dumped[key], want))
    if verbose:
        if problems:
            for p in problems:
                print("  [FAIL] " + p)
        else:
            print("  [PASS] 头部偏移三方一致：固件 H_* / 编译器 offsetof / export_blob "
                  "共 %d 个字段，头长 %d 字节" % (len(HDR_FIELDS), CFC_HDR_BYTES))
    return (not problems), problems


def fake_policy(input_dim=106, cell_in=64, hidden=64, backbone=64, seed=0):
    """造一份随机 policy（不依赖 torch），给自检和冒烟用。"""
    rng = np.random.default_rng(seed)
    zu = backbone if backbone > 0 else cell_in + hidden
    p = {
        "input_dim": input_dim, "cell_in_dim": cell_in, "hidden_dim": hidden,
        "backbone_units": backbone, "n_heads": N_HEADS,
        "head_dim": np.asarray([d for _n, d, _a in HEADS], dtype=np.int32),
        "head_act": np.asarray([HEAD_ACT[a] for _n, _d, a in HEADS], dtype=np.int32),
        "bb_w": rng.normal(0, 0.1, (backbone, cell_in + hidden)).astype(F32),
        "bb_b": rng.normal(0, 0.1, backbone).astype(F32),
        "ff1_w": rng.normal(0, 0.1, (hidden, zu)).astype(F32),
        "ff1_b": rng.normal(0, 0.1, hidden).astype(F32),
        "ff2_w": rng.normal(0, 0.1, (hidden, zu)).astype(F32),
        "ff2_b": rng.normal(0, 0.1, hidden).astype(F32),
        "ta_w": rng.normal(0, 0.1, (hidden, zu)).astype(F32),
        "ta_b": rng.normal(0, 0.1, hidden).astype(F32),
        "tb_w": rng.normal(0, 0.1, (hidden, zu)).astype(F32),
        "tb_b": rng.normal(0, 0.1, hidden).astype(F32),
        "ret": np.full(hidden, -3.0, dtype=F32),
        "proj_w": rng.normal(0, 0.1, (cell_in, input_dim)).astype(F32),
        "proj_b": np.zeros(cell_in, dtype=F32),
        "head_w": rng.normal(0, 0.1, (sum(d for _n, d, _a in HEADS), hidden)).astype(F32),
        "head_b": np.zeros(sum(d for _n, d, _a in HEADS), dtype=F32),
        "input_mean": rng.normal(0, 1, input_dim).astype(F32),
        "input_std": np.abs(rng.normal(1, 0.2, input_dim)).astype(F32),
        "ts": 1.0, "clip_sigma": 4.0,
    }
    return p


def self_test():
    """不依赖 torch / 固件源码的自检：布局、CRC、mask 折叠、std 下限。"""
    ok = True

    def ck(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                               ("  " + detail) if detail else ""))

    # 1) 头字段不重叠、总长 = CFC_HDR_BYTES
    covered = [False] * CFC_HDR_BYTES
    overlap = []
    for name, off, size, _fmt in HDR_FIELDS:
        for i in range(off, off + size):
            if covered[i]:
                overlap.append(name)
            covered[i] = True
    ck("%d 个字段恰好铺满 %d 字节" % (len(HDR_FIELDS), CFC_HDR_BYTES),
       all(covered) and not overlap,
       "未覆盖 %d 字节，重叠 %s" % (covered.count(False), overlap or "无"))

    # 2) mask 只折 ff1/ff2
    w = np.ones((4, 6), F32)
    m = (np.random.default_rng(3).random((6, 4)) < 0.5).astype(F32)   # (in, out)
    folded = fold_sparsity(w, m.T)                                     # (out, in)
    ck("fold_sparsity 只折叠传入的权重", folded.shape == w.shape
       and np.allclose(folded, m.T), "零占比 %.0f%%" % (100 * np.mean(folded == 0)))
    try:
        fold_sparsity(w, m)
        shape_guard = False
    except ValueError:
        shape_guard = True
    ck("fold_sparsity 拒绝 (in,out) 朝向的 mask", shape_guard)

    # 3) std 下限
    s = std_with_floor(np.array([0.0, 1e-8, 0.5], F32), 1e-2)
    ck("std_with_floor 抬起小值且不动大值",
       abs(float(s[0]) - 1e-2) < 1e-9 and abs(float(s[2]) - 0.5) < 1e-6, str(s))

    # 4) 打包 -> 解析 往返
    pol = fake_policy()
    blob = build_blob(pol)
    man = manifest_of(blob)
    ck("blob = 头 + payload，长度一致",
       len(blob) == CFC_HDR_BYTES + sum(man["layer_len"]),
       "%d 字节" % len(blob))
    ck("magic / flags / dims 往返一致",
       man["magic"] == CFC_BLOB_MAGIC
       and man["flags"] == (FLAG_PROJ | FLAG_RET | FLAG_LAYERPREC)
       and man["input_dim"] == 106 and man["hidden_dim"] == 64
       and man["n_heads"] == N_HEADS)
    ck("CRC32 与 zlib 口径一致", man["crc32_ok"], "0x%08X" % man["crc32"])
    ck("层长度等于各自 float 数 × 4",
       man["layer_len"][LAYER_IDS.index("bb_w")] == 64 * (64 + 64) * 4
       and man["layer_len"][LAYER_IDS.index("heads")] == (15 * 64 + 15) * 4
       and man["layer_len"][LAYER_IDS.index("ret")] == 64 * 4)
    ck("所有 layer_off 4 字节对齐",
       all(o % 4 == 0 for o in man["layer_off"] if o))
    try:
        bad = dict(pol)
        bad["ff1_w"] = np.zeros((64, 63), F32)
        build_blob(bad)
        size_guard = False
    except ValueError:
        size_guard = True
    ck("层形状不对时 build_blob 报错", size_guard)

    # 5) int8 量化 + int8 blob
    rng = np.random.default_rng(7)
    w = rng.normal(0, 0.1, (6, 9)).astype(F32)
    q, sc = quantize_per_channel(w)
    back = dequantize_per_channel(q, sc)
    rel = float(np.max(np.abs(back - w)) / max(np.max(np.abs(w)), 1e-12))
    ck("quantize_per_channel 逐通道 round-trip 相对误差 < 1%", rel < 1e-2,
       "max=%.3e，scale 范围 %.3e..%.3e" % (rel, sc.min(), sc.max()))
    ck("quantize_per_channel 保住零行",
       np.all(dequantize_per_channel(*quantize_per_channel(np.zeros((2, 5), F32)))
              == 0.0))

    blob8 = build_blob(pol, int8=True)
    man8 = manifest_of(blob8)
    ck("int8 blob flags 带 bit2/bit3 且仍有 proj/ret",
       man8["flags"] == (FLAG_PROJ | FLAG_RET | FLAG_INT8 | FLAG_LAYERPREC),
       "flags=0x%04X" % man8["flags"])
    ck("int8 blob 通过 CRC 自洽", man8["crc32_ok"], "0x%08X" % man8["crc32"])
    ck("int8 blob 明显小于 float32 blob",
       len(blob8) < len(blob) * 0.45,
       "int8 %d 字节 vs float32 %d 字节（%.1f%%）"
       % (len(blob8), len(blob), 100.0 * len(blob8) / len(blob)))
    int8_lens_ok = True
    for name in INT8_CAPABLE:
        i = LAYER_IDS.index(name)
        n_out = int8_out_dim(name, input_dim=106, cell_in=64, hidden=64,
                             backbone=64)
        k = {"bb_w": 128, "proj_w": 106}.get(name, 64)
        want = int8_layer_bytes(n_out * k)
        if man8["layer_len"][i] != want:
            int8_lens_ok = False
            print("        %s: layer_len %d != %d" % (name, man8["layer_len"][i], want))
        if man8["layer_off"][i] + want + n_out * 4 > len(blob8):
            int8_lens_ok = False
            print("        %s: scale 越界" % name)
    ck("int8 层的 layer_len = 元素数（%4 对齐）且 scale 在 blob 内", int8_lens_ok)
    ck("int8 blob 的所有 layer_off 仍 4 字节对齐",
       all(o % 4 == 0 for o in man8["layer_off"] if o))
    ck("float32 blob 不带 bit2（默认路径没被改动）", (man["flags"] & FLAG_INT8) == 0,
       "flags=0x%04X" % man["flags"])

    # 6) 逐层精度表（layer_prec）
    ck("默认 blob 的 layer_prec 全 0",
       man["layer_prec"] == [0] * CFC_MAX_LAYERS, str(man["layer_prec"]))
    want8 = [1 if n in INT8_CAPABLE else 0 for n in LAYER_IDS]
    ck("int8 blob 的 layer_prec 逐层记下（heads/biases 仍是 0）",
       man8["layer_prec"][:len(LAYER_IDS)] == want8
       and man8["int8_layers"] == list(INT8_CAPABLE),
       "heads=%d ret=%d proj_b=%d" % (man8["layer_prec"][LAYER_IDS.index("heads")],
                                      man8["layer_prec"][LAYER_IDS.index("ret")],
                                      man8["layer_prec"][LAYER_IDS.index("proj_b")]))
    # 只有 ta 一层 int8：格式真的逐层，不是"全有全无"
    man_ta = manifest_of(build_blob(pol, prec={"ta_w": PREC_INT8}))
    ck("逐层可选：只标 ta_w 时其余层都是 float32",
       man_ta["int8_layers"] == ["ta_w"]
       and man_ta["layer_len"][LAYER_IDS.index("ta_w")] == int8_layer_bytes(64 * 64)
       and man_ta["layer_len"][LAYER_IDS.index("tb_w")] == 64 * 64 * 4,
       "flags=0x%04X" % man_ta["flags"])
    for bad_prec, why in (({"heads": PREC_INT8}, "heads 不允许 int8"),
                          ({"ret": PREC_INT8}, "ret 不是权重矩阵"),
                          ({"nope": PREC_INT8}, "未知层名")):
        try:
            build_blob(pol, prec=bad_prec)
            guarded = False
        except ValueError:
            guarded = True
        ck("build_blob 拒绝 %s" % why, guarded)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="policy.npz -> 固件 blob（逐字段对齐 cfc_model.h），并做三方偏移核对。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--npz", default=str(HERE / "policy.npz"), help="输入 policy.npz")
    ap.add_argument("--out", default=str(HERE / "model_slot0.blob"), help="输出 blob 路径")
    ap.add_argument("--manifest", default=str(HERE / "blob_manifest.json"),
                    help="输出 manifest（字段偏移/长度，给人核对）")
    ap.add_argument("--clip-sigma", type=float, default=None, help="覆盖 clip_sigma（默认 4.0）")
    ap.add_argument("--ts", type=float, default=None, help="覆盖 ts（默认 1.0）")
    ap.add_argument("--std-floor", type=float, default=1e-2, help="input_std 下限")
    ap.add_argument("--int8", action="store_true",
                    help="六个权重矩阵量化成 int8 + 逐通道 scale（heads 保持 float32，"
                         "逐层精度写进 layer_prec，flags bit2）")
    ap.add_argument("--int8-layers", default=None,
                    help="只把这些层（逗号分隔）量化成 int8，用于逐层精度实验；"
                         "给定时不套用 --int8 的默认配方")
    ap.add_argument("--check-offsets", action="store_true",
                    help="只做头字段偏移三方核对（需要先编 dump_offsets.exe）")
    ap.add_argument("--self-test", action="store_true",
                    help="不依赖 torch/固件源码的自检（布局/CRC/mask 折叠/std 下限）")
    ap.add_argument("--dump-offsets-exe", default=str(HERE / "build" / "dump_offsets.exe"),
                    help="dump_offsets 可执行文件路径")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.check_offsets:
        ok, _ = check_offsets(args.dump_offsets_exe)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    policy = load_policy_npz(args.npz)
    prec = None
    if args.int8_layers:
        prec = {n.strip(): PREC_INT8
                for n in args.int8_layers.split(",") if n.strip()}
    blob = build_blob(policy, clip_sigma=args.clip_sigma, ts=args.ts,
                      std_floor=args.std_floor,
                      int8=args.int8 and not prec, prec=prec)
    Path(args.out).write_bytes(blob)
    man = manifest_of(blob)
    man["source_npz"] = str(args.npz)
    Path(args.manifest).write_text(json.dumps(man, indent=2), encoding="utf-8")
    describe(blob)
    ok, _ = check_offsets(args.dump_offsets_exe)
    print("PASS" if ok and man["crc32_ok"] else "FAIL")
    return 0 if (ok and man["crc32_ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
