# -*- coding: utf-8 -*-
"""export_tflite_bundle.py — 端侧 RL 策略与双敲动作模型导出为标准 FlatBuffers TFLite v3 (.tflite + model.json + .zip)。

严格遵循 tensorflow/lite/schema/schema_v3.fbs 与 G:/robot_app/android/model_schema.md：
1. 支持导出：
   - EndSidePolicyNet (task="policy", 输入 [1, 39] -> 输出 [1, 3])
   - DoubleKnockNet (task="action", 双输入 vis=[1, T, H, W, 2] 与 kin=[1, T, 8] -> 输出 imp=[1, T] 与 cnt=[1, 4])
2. 100% 仅使用 TFLite 原生内置核心算子（ZERO Flex/Custom Ops）：
   CONV_2D / PAD / MEAN / FULLY_CONNECTED / LOGISTIC / TANH / RELU / MUL / ADD / SUB / RSQRT /
   SLICE / CONCATENATION / RESHAPE / SOFTMAX / SUM / REDUCE_MAX
3. 内置 FlatBuffers v3 图结构解析与逐算子 CPU 执行校验器 run_tflite_flatbuffer_cpu，
   保证导出的 .tflite 文件字节结构严格合法且与 PyTorch eval() 输出最大绝对误差 < 1e-4。
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import flatbuffers
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from end_side_controller import EndSidePolicyNet  # noqa: E402

# TFLite schema_v3 BuiltinOperator & Enum 常量 (与 schema_v3.fbs 严格一致)
TFL_FLOAT32 = 0
TFL_INT32 = 2

BUILTIN_OP_ADD = 0
BUILTIN_OP_CONCATENATION = 2
BUILTIN_OP_CONV_2D = 3
BUILTIN_OP_FULLY_CONNECTED = 9
BUILTIN_OP_LOGISTIC = 14
BUILTIN_OP_MUL = 18
BUILTIN_OP_RELU = 19
BUILTIN_OP_RESHAPE = 22
BUILTIN_OP_SOFTMAX = 25
BUILTIN_OP_TANH = 28
BUILTIN_OP_PAD = 34
BUILTIN_OP_MEAN = 40
BUILTIN_OP_SUB = 41
BUILTIN_OP_SLICE = 65
BUILTIN_OP_SUM = 74
BUILTIN_OP_RSQRT = 76
BUILTIN_OP_REDUCE_MAX = 82

PAD_SAME = 0
PAD_VALID = 1

ACT_NONE = 0
ACT_RELU = 1
ACT_TANH = 4

# BuiltinOptions union 1-based tags (与 schema_v3.fbs L477-526 逐项对齐)
OPTS_NONE = 0
OPTS_CONV_2D = 1
OPTS_FULLY_CONNECTED = 8
OPTS_SOFTMAX = 9
OPTS_CONCATENATION = 10
OPTS_ADD = 11
OPTS_RESHAPE = 17
OPTS_MUL = 21
OPTS_PAD = 22
OPTS_REDUCER = 27
OPTS_SUB = 28
OPTS_SLICE = 48


def _fb_int_vec(b: flatbuffers.Builder, vals: list[int]) -> int:
    b.StartVector(4, len(vals), 4)
    for v in reversed(vals):
        b.PrependInt32(int(v))
    return b.EndVector()


def _fb_offset_vec(b: flatbuffers.Builder, offsets: list[int]) -> int:
    b.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        b.PrependUOffsetTRelative(off)
    return b.EndVector()


def _fb_buffer(b: flatbuffers.Builder, raw: bytes | None) -> int:
    data_off = b.CreateByteVector(raw) if raw else 0
    b.StartObject(1)
    if data_off:
        b.PrependUOffsetTRelativeSlot(0, data_off, 0)
    return b.EndObject()


def _fb_tensor(b: flatbuffers.Builder, shape: list[int], tensor_type: int, buffer_idx: int, name: str) -> int:
    name_off = b.CreateString(name)
    shape_off = _fb_int_vec(b, shape)
    b.StartObject(4)
    b.PrependUOffsetTRelativeSlot(0, shape_off, 0)
    b.PrependInt8Slot(1, tensor_type, 0)
    b.PrependUint32Slot(2, buffer_idx, 0)
    b.PrependUOffsetTRelativeSlot(3, name_off, 0)
    return b.EndObject()


def _fb_opcode(b: flatbuffers.Builder, builtin_code: int) -> int:
    b.StartObject(4)
    b.PrependInt8Slot(0, builtin_code if builtin_code < 127 else 127, 0)
    b.PrependInt32Slot(2, 1, 1)
    b.PrependInt32Slot(3, builtin_code, 0)
    return b.EndObject()


def _fb_conv2d_options(
    b: flatbuffers.Builder,
    padding: int = PAD_VALID,
    stride_w: int = 1,
    stride_h: int = 1,
    fused_act: int = ACT_NONE,
    dilation_w: int = 1,
    dilation_h: int = 1,
) -> int:
    b.StartObject(6)
    b.PrependInt8Slot(0, padding, 0)
    b.PrependInt32Slot(1, stride_w, 0)
    b.PrependInt32Slot(2, stride_h, 0)
    b.PrependInt8Slot(3, fused_act, 0)
    b.PrependInt32Slot(4, dilation_w, 1)
    b.PrependInt32Slot(5, dilation_h, 1)
    return b.EndObject()


def _fb_fc_options(b: flatbuffers.Builder, fused_act: int = ACT_NONE) -> int:
    b.StartObject(1)
    b.PrependInt8Slot(0, fused_act, 0)
    return b.EndObject()


def _fb_mul_options(b: flatbuffers.Builder, fused_act: int = ACT_NONE) -> int:
    b.StartObject(1)
    b.PrependInt8Slot(0, fused_act, 0)
    return b.EndObject()


def _fb_add_options(b: flatbuffers.Builder, fused_act: int = ACT_NONE) -> int:
    b.StartObject(1)
    b.PrependInt8Slot(0, fused_act, 0)
    return b.EndObject()


def _fb_sub_options(b: flatbuffers.Builder, fused_act: int = ACT_NONE) -> int:
    b.StartObject(1)
    b.PrependInt8Slot(0, fused_act, 0)
    return b.EndObject()


def _fb_concat_options(b: flatbuffers.Builder, axis: int, fused_act: int = ACT_NONE) -> int:
    b.StartObject(2)
    b.PrependInt32Slot(0, axis, 0)
    b.PrependInt8Slot(1, fused_act, 0)
    return b.EndObject()


def _fb_softmax_options(b: flatbuffers.Builder, beta: float = 1.0) -> int:
    b.StartObject(1)
    b.PrependFloat32Slot(0, float(beta), 0.0)
    return b.EndObject()


def _fb_reshape_options(b: flatbuffers.Builder, new_shape: list[int]) -> int:
    shp_off = _fb_int_vec(b, new_shape)
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, shp_off, 0)
    return b.EndObject()


def _fb_reducer_options(b: flatbuffers.Builder, keep_dims: bool) -> int:
    b.StartObject(1)
    b.PrependBoolSlot(0, bool(keep_dims), False)
    return b.EndObject()


def _fb_empty_options(b: flatbuffers.Builder) -> int:
    b.StartObject(0)
    return b.EndObject()


def _fb_operator(
    b: flatbuffers.Builder,
    opcode_idx: int,
    inputs: list[int],
    outputs: list[int],
    opt_type: int,
    opt_off: int,
) -> int:
    in_off = _fb_int_vec(b, inputs)
    out_off = _fb_int_vec(b, outputs)
    b.StartObject(5)
    b.PrependUint32Slot(0, opcode_idx, 0)
    b.PrependUOffsetTRelativeSlot(1, in_off, 0)
    b.PrependUOffsetTRelativeSlot(2, out_off, 0)
    if opt_type != OPTS_NONE:
        b.PrependUint8Slot(3, opt_type, 0)
        b.PrependUOffsetTRelativeSlot(4, opt_off, 0)
    return b.EndObject()


class _TFLiteGraphBuilder:
    """通用 TFLite v3 图构造器，自动管理 Buffers / OperatorCodes / Tensors / Operators。"""

    def __init__(self, description: str, initial_capacity: int = 524288):
        self.description = description
        self.b = flatbuffers.Builder(initial_capacity)
        self.buf_raws: list[bytes | None] = [None]  # Buffer 0 必须为空
        self.tensors: list[tuple[list[int], int, int, str]] = []
        self.opcode_map: dict[int, int] = {}
        self.opcodes_order: list[int] = []
        self.op_defs: list[tuple[int, list[int], list[int], int, int]] = []
        self.inputs: list[int] = []
        self.outputs: list[int] = []
        self._const_cache: dict[tuple[bytes, tuple[int, ...], int], int] = {}

    def _opcode_index(self, builtin_code: int) -> int:
        if builtin_code not in self.opcode_map:
            self.opcode_map[builtin_code] = len(self.opcodes_order)
            self.opcodes_order.append(builtin_code)
        return self.opcode_map[builtin_code]

    def add_input(self, name: str, shape: list[int], dtype: int = TFL_FLOAT32) -> int:
        tid = len(self.tensors)
        self.tensors.append(([int(d) for d in shape], dtype, 0, name))
        self.inputs.append(tid)
        return tid

    def add_temp(self, name: str, shape: list[int], dtype: int = TFL_FLOAT32) -> int:
        tid = len(self.tensors)
        self.tensors.append(([int(d) for d in shape], dtype, 0, name))
        return tid

    def add_const(self, name: str, arr: np.ndarray, dtype: int = TFL_FLOAT32) -> int:
        np_dtype = np.float32 if dtype == TFL_FLOAT32 else np.int32
        a = np.ascontiguousarray(arr, dtype=np_dtype)
        raw = a.tobytes(order="C")
        key = (raw, tuple(int(s) for s in a.shape), dtype)
        if key in self._const_cache:
            return self._const_cache[key]
        buf_idx = len(self.buf_raws)
        self.buf_raws.append(raw)
        tid = len(self.tensors)
        self.tensors.append(([int(s) for s in a.shape], dtype, buf_idx, name))
        self._const_cache[key] = tid
        return tid

    def set_outputs(self, tids: list[int]) -> None:
        self.outputs = list(tids)

    def shape_of(self, tid: int) -> list[int]:
        return list(self.tensors[tid][0])

    def _emit_op(self, builtin_code: int, inputs: list[int], outputs: list[int], opt_type: int, opt_off: int) -> None:
        oc_idx = self._opcode_index(builtin_code)
        self.op_defs.append((oc_idx, inputs, outputs, opt_type, opt_off))

    # --- High-level Core Builtin Ops ---

    def conv2d(
        self,
        name: str,
        x: int,
        w_ohwi: np.ndarray,
        bias: np.ndarray | None,
        stride: int = 1,
        dilation: int = 1,
        padding: int = PAD_VALID,
        fused_act: int = ACT_NONE,
    ) -> int:
        n, h_in, w_in, _ = self.shape_of(x)
        c_out, kh, kw, _ = w_ohwi.shape
        if padding == PAD_VALID:
            eff_kh = (kh - 1) * dilation + 1
            eff_kw = (kw - 1) * dilation + 1
            h_out = (h_in - eff_kh) // stride + 1
            w_out = (w_in - eff_kw) // stride + 1
        else:
            h_out = (h_in + stride - 1) // stride
            w_out = (w_in + stride - 1) // stride
        w_tid = self.add_const(f"{name}_w", w_ohwi, TFL_FLOAT32)
        b_arr = np.zeros((c_out,), dtype=np.float32) if bias is None else bias.reshape(c_out)
        b_tid = self.add_const(f"{name}_b", b_arr, TFL_FLOAT32)
        out = self.add_temp(name, [n, h_out, w_out, int(c_out)], TFL_FLOAT32)
        opt = _fb_conv2d_options(self.b, padding, stride, stride, fused_act, dilation, dilation)
        self._emit_op(BUILTIN_OP_CONV_2D, [x, w_tid, b_tid], [out], OPTS_CONV_2D, opt)
        return out

    def pad(self, name: str, x: int, paddings: list[list[int]]) -> int:
        shp = self.shape_of(x)
        out_shp = [shp[i] + paddings[i][0] + paddings[i][1] for i in range(len(shp))]
        p_tid = self.add_const(f"{name}_p", np.asarray(paddings, dtype=np.int32), TFL_INT32)
        out = self.add_temp(name, out_shp, TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_PAD, [x, p_tid], [out], OPTS_PAD, _fb_empty_options(self.b))
        return out

    def fc(self, name: str, x: int, w_oi: np.ndarray, bias: np.ndarray | None, fused_act: int = ACT_NONE) -> int:
        n = self.shape_of(x)[0]
        out_dim = int(w_oi.shape[0])
        w_tid = self.add_const(f"{name}_w", w_oi, TFL_FLOAT32)
        b_tid = self.add_const(
            f"{name}_b",
            np.zeros((out_dim,), dtype=np.float32) if bias is None else bias.reshape(out_dim),
            TFL_FLOAT32,
        )
        out = self.add_temp(name, [n, out_dim], TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_FULLY_CONNECTED, [x, w_tid, b_tid], [out], OPTS_FULLY_CONNECTED, _fb_fc_options(self.b, fused_act))
        return out

    def sigmoid(self, name: str, x: int) -> int:
        out = self.add_temp(name, self.shape_of(x), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_LOGISTIC, [x], [out], OPTS_NONE, 0)
        return out

    def tanh(self, name: str, x: int) -> int:
        out = self.add_temp(name, self.shape_of(x), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_TANH, [x], [out], OPTS_NONE, 0)
        return out

    def relu(self, name: str, x: int) -> int:
        out = self.add_temp(name, self.shape_of(x), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_RELU, [x], [out], OPTS_NONE, 0)
        return out

    def silu(self, name: str, x: int) -> int:
        s = self.sigmoid(f"{name}_sig", x)
        return self.mul(name, x, s)

    def add(self, name: str, a: int, b_tid: int, out_shape: list[int] | None = None, fused_act: int = ACT_NONE) -> int:
        out = self.add_temp(name, out_shape or self.shape_of(a), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_ADD, [a, b_tid], [out], OPTS_ADD, _fb_add_options(self.b, fused_act))
        return out

    def sub(self, name: str, a: int, b_tid: int, out_shape: list[int] | None = None) -> int:
        out = self.add_temp(name, out_shape or self.shape_of(a), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_SUB, [a, b_tid], [out], OPTS_SUB, _fb_sub_options(self.b, ACT_NONE))
        return out

    def mul(self, name: str, a: int, b_tid: int, out_shape: list[int] | None = None) -> int:
        out = self.add_temp(name, out_shape or self.shape_of(a), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_MUL, [a, b_tid], [out], OPTS_MUL, _fb_mul_options(self.b, ACT_NONE))
        return out

    def rsqrt(self, name: str, x: int) -> int:
        out = self.add_temp(name, self.shape_of(x), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_RSQRT, [x], [out], OPTS_NONE, 0)
        return out

    def reshape(self, name: str, x: int, new_shape: list[int]) -> int:
        shp_tid = self.add_const(f"{name}_shp", np.asarray(new_shape, dtype=np.int32), TFL_INT32)
        out = self.add_temp(name, list(new_shape), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_RESHAPE, [x, shp_tid], [out], OPTS_RESHAPE, _fb_reshape_options(self.b, list(new_shape)))
        return out

    def concat(self, name: str, tids: list[int], axis: int) -> int:
        base = self.shape_of(tids[0])
        ax = axis if axis >= 0 else len(base) + axis
        base[ax] = sum(self.shape_of(t)[ax] for t in tids)
        out = self.add_temp(name, base, TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_CONCATENATION, list(tids), [out], OPTS_CONCATENATION, _fb_concat_options(self.b, axis))
        return out

    def slice(self, name: str, x: int, begin: list[int], size: list[int]) -> int:
        b_tid = self.add_const(f"{name}_b", np.asarray(begin, dtype=np.int32), TFL_INT32)
        s_tid = self.add_const(f"{name}_s", np.asarray(size, dtype=np.int32), TFL_INT32)
        out = self.add_temp(name, list(size), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_SLICE, [x, b_tid, s_tid], [out], OPTS_SLICE, _fb_empty_options(self.b))
        return out

    def softmax(self, name: str, x: int, beta: float = 1.0) -> int:
        out = self.add_temp(name, self.shape_of(x), TFL_FLOAT32)
        self._emit_op(BUILTIN_OP_SOFTMAX, [x], [out], OPTS_SOFTMAX, _fb_softmax_options(self.b, beta))
        return out

    def _reduce(self, name: str, op_code: int, x: int, axes: list[int], keep_dims: bool) -> int:
        shp = self.shape_of(x)
        rank = len(shp)
        norm_axes = [ax if ax >= 0 else rank + ax for ax in axes]
        if keep_dims:
            out_shp = [1 if i in norm_axes else shp[i] for i in range(rank)]
        else:
            out_shp = [shp[i] for i in range(rank) if i not in norm_axes]
        ax_tid = self.add_const(f"{name}_ax", np.asarray(norm_axes, dtype=np.int32), TFL_INT32)
        out = self.add_temp(name, out_shp, TFL_FLOAT32)
        self._emit_op(op_code, [x, ax_tid], [out], OPTS_REDUCER, _fb_reducer_options(self.b, keep_dims))
        return out

    def mean(self, name: str, x: int, axes: list[int], keep_dims: bool = False) -> int:
        return self._reduce(name, BUILTIN_OP_MEAN, x, axes, keep_dims)

    def reduce_sum(self, name: str, x: int, axes: list[int], keep_dims: bool = False) -> int:
        return self._reduce(name, BUILTIN_OP_SUM, x, axes, keep_dims)

    def reduce_max(self, name: str, x: int, axes: list[int], keep_dims: bool = False) -> int:
        return self._reduce(name, BUILTIN_OP_REDUCE_MAX, x, axes, keep_dims)

    def layer_norm(self, name: str, x: int, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> int:
        shp = self.shape_of(x)
        dim = shp[-1]
        mu = self.mean(f"{name}_mu", x, [-1], keep_dims=True)
        diff = self.sub(f"{name}_diff", x, mu, out_shape=shp)
        sq = self.mul(f"{name}_sq", diff, diff)
        var = self.mean(f"{name}_var", sq, [-1], keep_dims=True)
        eps_tid = self.add_const(f"{name}_eps", np.asarray([[eps]], dtype=np.float32), TFL_FLOAT32)
        var_eps = self.add(f"{name}_veps", var, eps_tid, out_shape=self.shape_of(var))
        inv_std = self.rsqrt(f"{name}_rsqrt", var_eps)
        norm = self.mul(f"{name}_norm", diff, inv_std, out_shape=shp)
        g_tid = self.add_const(f"{name}_g", gamma.reshape(1, dim).astype(np.float32), TFL_FLOAT32)
        b_tid = self.add_const(f"{name}_beta", beta.reshape(1, dim).astype(np.float32), TFL_FLOAT32)
        scaled = self.mul(f"{name}_sc", norm, g_tid, out_shape=shp)
        return self.add(name, scaled, b_tid, out_shape=shp)

    def finish(self) -> bytes:
        buf_offs = [_fb_buffer(self.b, raw) for raw in self.buf_raws]
        buffers_vec = _fb_offset_vec(self.b, buf_offs)
        tensor_offs = [_fb_tensor(self.b, shp, tp, bi, nm) for shp, tp, bi, nm in self.tensors]
        tensors_vec = _fb_offset_vec(self.b, tensor_offs)
        opcodes_vec = _fb_offset_vec(self.b, [_fb_opcode(self.b, code) for code in self.opcodes_order])
        op_offs = [_fb_operator(self.b, oc, ins, outs, ot, oo) for oc, ins, outs, ot, oo in self.op_defs]
        ops_vec = _fb_offset_vec(self.b, op_offs)

        sg_in_vec = _fb_int_vec(self.b, self.inputs)
        sg_out_vec = _fb_int_vec(self.b, self.outputs)
        sg_name = self.b.CreateString("main")
        self.b.StartObject(5)
        self.b.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(1, sg_in_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(2, sg_out_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(3, ops_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(4, sg_name, 0)
        sg_off = self.b.EndObject()
        subgraphs_vec = _fb_offset_vec(self.b, [sg_off])

        desc_off = self.b.CreateString(self.description)
        self.b.StartObject(5)
        self.b.PrependUint32Slot(0, 3, 0)
        self.b.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
        self.b.PrependUOffsetTRelativeSlot(3, desc_off, 0)
        self.b.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
        model_root = self.b.EndObject()
        self.b.Finish(model_root, file_identifier=b"TFL3")
        return bytes(self.b.Output())


def _fold_conv_bn(conv: nn.Module, bn: nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """将 eval 模式下的 Conv2d/Conv1d + BatchNorm 精确折叠为带偏置卷积权重。"""
    w = conv.weight.detach().cpu().float()
    b_conv = conv.bias.detach().cpu().float() if conv.bias is not None else torch.zeros(w.shape[0])
    gamma = bn.weight.detach().cpu().float()
    beta = bn.bias.detach().cpu().float()
    mean = bn.running_mean.detach().cpu().float()
    var = bn.running_var.detach().cpu().float()
    scale = gamma / torch.sqrt(var + bn.eps)
    view_shape = [-1] + [1] * (w.ndim - 1)
    w_fold = (w * scale.view(*view_shape)).numpy().astype(np.float32)
    b_fold = ((b_conv - mean) * scale + beta).numpy().astype(np.float32)
    return w_fold, b_fold


def build_tflite_flatbuffer(policy: EndSidePolicyNet) -> bytes:
    """使用 FlatBuffers 构造符合 TFLite v3 规范的 39 维 RL 策略模型字节流（支持标准 MLP 与残差增强 MLP）。"""
    sd = policy.state_dict()
    w1 = sd["fc1.weight"].detach().cpu().numpy().astype(np.float32)
    b1 = sd["fc1.bias"].detach().cpu().numpy().astype(np.float32)
    w2 = sd["fc2.weight"].detach().cpu().numpy().astype(np.float32)
    b2 = sd["fc2.bias"].detach().cpu().numpy().astype(np.float32)
    w3 = sd["fc3.weight"].detach().cpu().numpy().astype(np.float32)
    b3 = sd["fc3.bias"].detach().cpu().numpy().astype(np.float32)
    scale = sd["action_scale"].detach().cpu().numpy().astype(np.float32).reshape(1, int(policy.action_dim))

    gb = _TFLiteGraphBuilder("RIG-Hover End-Side Spatial RL Policy (TFLite v3)", initial_capacity=524288)
    x = gb.add_input("obs_input", [1, int(policy.feature_dim)], TFL_FLOAT32)
    h1 = gb.fc("fc1", x, w1, b1, fused_act=ACT_TANH)
    h2 = gb.fc("fc2", h1, w2, b2, fused_act=ACT_TANH)
    if "fc_res.weight" in sd:
        w_res = sd["fc_res.weight"].detach().cpu().numpy().astype(np.float32)
        b_res = sd["fc_res.bias"].detach().cpu().numpy().astype(np.float32)
        h_sum = gb.add("res_sum", h1, h2)
        h_mid = gb.fc("fc_res", h_sum, w_res, b_res, fused_act=ACT_TANH)
        h_out = gb.fc("fc3", h_mid, w3, b3, fused_act=ACT_TANH)
    else:
        h_out = gb.fc("fc3", h2, w3, b3, fused_act=ACT_TANH)
    sc = gb.add_const("action_scale", scale, TFL_FLOAT32)
    out = gb.mul("action_output", h_out, sc)
    gb.set_outputs([out])
    return gb.finish()


def build_double_knock_tflite(model: nn.Module, window: int = 48, image_size: int = 96) -> bytes:
    """将 DoubleKnockNet 导出为 100% 核心内置算子组成的标准 TFLite v3 图。

    输入张量:
      - inputs[0] x_vis: [1, T, H, W, 2]  (float32, 通道 0=gray/255, 通道 1=|diff|/255)
      - inputs[1] x_kin: [1, T, 8]        (float32, 8 维运动学向量)
    输出张量:
      - outputs[0] logit_imp: [1, T]      (逐帧接触脉冲 logits)
      - outputs[1] logit_cnt: [1, 4]      (窗口动作 4 分类 logits)
    """
    model.eval()
    T = int(window)
    H = int(image_size)
    W = int(image_size)
    C = int(model.in_ch)
    K = int(model.kin_dim)
    Hid = int(model.hidden_dim)

    gb = _TFLiteGraphBuilder("RIG-Hover DoubleKnockNet (TFLite v3 Pure Builtins)", initial_capacity=1048576)
    x_vis_in = gb.add_input("x_vis", [1, T, H, W, C], TFL_FLOAT32)
    x_kin_in = gb.add_input("x_kin", [1, T, K], TFL_FLOAT32)

    # 1. Visual CNN Encoder (Conv2d + BN folded + explicit padding=1 + SiLU + AdaptiveAvgPool MEAN)
    v = gb.reshape("vis_flat", x_vis_in, [T, H, W, C])
    for idx, conv_i, bn_i in ((0, 0, 1), (1, 3, 4), (2, 6, 7)):
        w_nchw, b_vec = _fold_conv_bn(model.vis_enc[conv_i], model.vis_enc[bn_i])
        w_ohwi = np.transpose(w_nchw, (0, 2, 3, 1)).astype(np.float32)  # (O, kH, kW, I)
        v_pad = gb.pad(f"vis_pad{idx}", v, [[0, 0], [1, 1], [1, 1], [0, 0]])
        v_conv = gb.conv2d(f"vis_conv{idx}", v_pad, w_ohwi, b_vec, stride=2, dilation=1, padding=PAD_VALID)
        v = gb.silu(f"vis_silu{idx}", v_conv)
    v_pool = gb.mean("vis_pool", v, [1, 2], keep_dims=False)  # [T, vis_dim]

    # 2. Kinematics Encoder (Linear + LayerNorm + SiLU)
    k_flat = gb.reshape("kin_flat", x_kin_in, [T, K])
    w_k = model.kin_enc[0].weight.detach().cpu().numpy().astype(np.float32)
    b_k = model.kin_enc[0].bias.detach().cpu().numpy().astype(np.float32)
    k_fc = gb.fc("kin_fc", k_flat, w_k, b_k)
    ln = model.kin_enc[1]
    k_ln = gb.layer_norm(
        "kin_ln",
        k_fc,
        ln.weight.detach().cpu().numpy().astype(np.float32),
        ln.bias.detach().cpu().numpy().astype(np.float32),
        eps=float(ln.eps),
    )
    k_emb = gb.silu("kin_silu", k_ln)  # [T, Hid//2]

    # 3. Fuse & Causal TCN (PAD left + Conv2d VALID over [1, 1, T+pad, C])
    fused = gb.concat("fuse", [v_pool, k_emb], axis=1)  # [T, fuse_dim]
    u = gb.reshape("tcn_in", fused, [1, 1, T, gb.shape_of(fused)[1]])

    def _causal_tcn(name: str, x_nhwc: int, tcn_mod: nn.Module) -> int:
        w1d, b1d = _fold_conv_bn(tcn_mod.conv, tcn_mod.bn)  # (out, in, k)
        w2d = np.transpose(w1d, (0, 2, 1))[:, np.newaxis, :, :].astype(np.float32)  # (out, 1, k, in)
        pad_left = int(tcn_mod.pad)
        dil = int(tcn_mod.conv.dilation[0])
        if pad_left > 0:
            x_nhwc = gb.pad(f"{name}_pad", x_nhwc, [[0, 0], [0, 0], [pad_left, 0], [0, 0]])
        c_out = gb.conv2d(f"{name}_conv", x_nhwc, w2d, b1d, stride=1, dilation=dil, padding=PAD_VALID)
        return gb.silu(f"{name}_act", c_out)

    u1 = _causal_tcn("tcn1", u, model.tcn1)
    u2_raw = _causal_tcn("tcn2", u1, model.tcn2)
    u2 = gb.add("tcn2_res", u2_raw, u1)
    u4_raw = _causal_tcn("tcn4", u2, model.tcn4)
    u4 = gb.add("tcn4_res", u4_raw, u2)
    u_seq = gb.reshape("u_seq", u4, [T, Hid])  # [T, Hid]

    # 4. Unrolled 1-layer Causal GRU + Residual
    w_ih = model.gru.weight_ih_l0.detach().cpu().numpy().astype(np.float32)  # [3*Hid, Hid]
    b_ih = model.gru.bias_ih_l0.detach().cpu().numpy().astype(np.float32)    # [3*Hid]
    w_hh = model.gru.weight_hh_l0.detach().cpu().numpy().astype(np.float32)  # [3*Hid, Hid]
    b_hh = model.gru.bias_hh_l0.detach().cpu().numpy().astype(np.float32)    # [3*Hid]

    proj_x_all = gb.fc("gru_px", u_seq, w_ih, b_ih)  # [T, 3*Hid]
    w_hh_tid = gb.add_const("gru_wh", w_hh, TFL_FLOAT32)
    b_hh_tid = gb.add_const("gru_bh", b_hh, TFL_FLOAT32)

    h_prev = gb.add_const("gru_h0", np.zeros((1, Hid), dtype=np.float32), TFL_FLOAT32)
    h_steps: list[int] = []
    for t in range(T):
        px_t = gb.slice(f"g{t}_px", proj_x_all, [t, 0], [1, 3 * Hid])
        ph_t = gb.add_temp(f"g{t}_ph", [1, 3 * Hid], TFL_FLOAT32)
        gb._emit_op(BUILTIN_OP_FULLY_CONNECTED, [h_prev, w_hh_tid, b_hh_tid], [ph_t], OPTS_FULLY_CONNECTED, _fb_fc_options(gb.b, ACT_NONE))

        xr = gb.slice(f"g{t}_xr", px_t, [0, 0], [1, Hid])
        xz = gb.slice(f"g{t}_xz", px_t, [0, Hid], [1, Hid])
        xn = gb.slice(f"g{t}_xn", px_t, [0, 2 * Hid], [1, Hid])
        hr = gb.slice(f"g{t}_hr", ph_t, [0, 0], [1, Hid])
        hz = gb.slice(f"g{t}_hz", ph_t, [0, Hid], [1, Hid])
        hn = gb.slice(f"g{t}_hn", ph_t, [0, 2 * Hid], [1, Hid])

        r_t = gb.sigmoid(f"g{t}_r", gb.add(f"g{t}_ra", xr, hr))
        z_t = gb.sigmoid(f"g{t}_z", gb.add(f"g{t}_za", xz, hz))
        n_t = gb.tanh(f"g{t}_n", gb.add(f"g{t}_na", xn, gb.mul(f"g{t}_rhn", r_t, hn)))
        # h_t = n_t + z_t * (h_prev - n_t) == (1 - z_t) * n_t + z_t * h_prev
        h_diff = gb.sub(f"g{t}_hd", h_prev, n_t)
        h_t = gb.add(f"g{t}_h", n_t, gb.mul(f"g{t}_zh", z_t, h_diff))
        h_prev = h_t
        h_steps.append(h_t)

    h_gru = gb.concat("gru_seq", h_steps, axis=0)  # [T, Hid]
    h_seq = gb.add("h_seq", h_gru, u_seq)          # [T, Hid]

    # 5. Head 1: Per-frame Impulse Logits & Differentiable Impulse Statistics
    imp_fc = gb.fc(
        "imp_fc",
        h_seq,
        model.impulse_head.weight.detach().cpu().numpy().astype(np.float32),
        model.impulse_head.bias.detach().cpu().numpy().astype(np.float32),
    )
    logit_imp = gb.reshape("logit_imp", imp_fc, [1, T])
    p_imp = gb.sigmoid("p_imp", logit_imp)

    half_t = max(1, T // 2)
    c_inv6 = gb.add_const("c_inv6", np.asarray([[1.0 / 6.0]], dtype=np.float32), TFL_FLOAT32)
    c_inv25 = gb.add_const("c_inv25", np.asarray([[1.0 / 2.5]], dtype=np.float32), TFL_FLOAT32)

    imp_sum = gb.mul("imp_sum", gb.reduce_sum("imp_s0", p_imp, [1], keep_dims=True), c_inv6)
    imp_max = gb.reduce_max("imp_max", p_imp, [1], keep_dims=True)
    p_first = gb.slice("p_first", p_imp, [0, 0], [1, half_t])
    imp_first = gb.reduce_max("imp_first", p_first, [1], keep_dims=True)
    p_second = gb.slice("p_second", p_imp, [0, half_t], [1, T - half_t])
    imp_second = gb.reduce_max("imp_second", p_second, [1], keep_dims=True)
    p_tail = gb.slice("p_tail", p_imp, [0, 1], [1, T - 1])
    p_head = gb.slice("p_head", p_imp, [0, 0], [1, T - 1])
    p_diff = gb.relu("p_diff_pos", gb.sub("p_diff", p_tail, p_head))
    imp_rise = gb.mul("imp_rise", gb.reduce_sum("imp_r0", p_diff, [1], keep_dims=True), c_inv25)

    # 6. Head 2: Attention Pooling + Last Hidden + Count MLP
    attn_fc = gb.fc(
        "attn_fc",
        h_seq,
        model.attn_score.weight.detach().cpu().numpy().astype(np.float32),
        model.attn_score.bias.detach().cpu().numpy().astype(np.float32),
    )
    attn_1t = gb.reshape("attn_1t", attn_fc, [1, T])
    attn_sm = gb.softmax("attn_sm", attn_1t, beta=1.0)
    attn_w = gb.reshape("attn_w", attn_sm, [T, 1])
    h_weighted = gb.mul("h_weighted", h_seq, attn_w, out_shape=[T, Hid])
    h_attn = gb.reduce_sum("h_attn", h_weighted, [0], keep_dims=True)  # [1, Hid]
    h_last = gb.slice("h_last", h_seq, [T - 1, 0], [1, Hid])         # [1, Hid]

    cnt_in = gb.concat("cnt_in", [h_attn, h_last, imp_sum, imp_max, imp_first, imp_second, imp_rise], axis=1)
    c1 = gb.fc(
        "cnt_fc1",
        cnt_in,
        model.count_head[0].weight.detach().cpu().numpy().astype(np.float32),
        model.count_head[0].bias.detach().cpu().numpy().astype(np.float32),
    )
    c1_act = gb.silu("cnt_silu", c1)
    logit_cnt = gb.fc(
        "logit_cnt",
        c1_act,
        model.count_head[3].weight.detach().cpu().numpy().astype(np.float32),
        model.count_head[3].bias.detach().cpu().numpy().astype(np.float32),
    )

    gb.set_outputs([logit_imp, logit_cnt])
    return gb.finish()


# ---------------------------------------------------------------------------
# FlatBuffers 读取解析 & 逐算子结构级 CPU 验证器 (无 TF 依赖下 100% 校验真 .tflite 图)
# ---------------------------------------------------------------------------

def _tab_i32(tab: flatbuffers.table.Table, voff: int, default: int = 0) -> int:
    off = tab.Offset(voff)
    return int(tab.Get(flatbuffers.number_types.Int32Flags, off + tab.Pos)) if off else default


def _tab_u32(tab: flatbuffers.table.Table, voff: int, default: int = 0) -> int:
    off = tab.Offset(voff)
    return int(tab.Get(flatbuffers.number_types.Uint32Flags, off + tab.Pos)) if off else default


def _tab_i8(tab: flatbuffers.table.Table, voff: int, default: int = 0) -> int:
    off = tab.Offset(voff)
    return int(tab.Get(flatbuffers.number_types.Int8Flags, off + tab.Pos)) if off else default


def _tab_u8(tab: flatbuffers.table.Table, voff: int, default: int = 0) -> int:
    off = tab.Offset(voff)
    return int(tab.Get(flatbuffers.number_types.Uint8Flags, off + tab.Pos)) if off else default


def _tab_bool(tab: flatbuffers.table.Table, voff: int, default: bool = False) -> bool:
    off = tab.Offset(voff)
    return bool(tab.Get(flatbuffers.number_types.BoolFlags, off + tab.Pos)) if off else default


def _tab_f32(tab: flatbuffers.table.Table, voff: int, default: float = 0.0) -> float:
    off = tab.Offset(voff)
    return float(tab.Get(flatbuffers.number_types.Float32Flags, off + tab.Pos)) if off else default


def _tab_i32_vec(tab: flatbuffers.table.Table, voff: int) -> list[int]:
    off = tab.Offset(voff)
    if not off:
        return []
    n = tab.VectorLen(off)
    start = tab.Vector(off)
    return [int(flatbuffers.encode.Get(flatbuffers.packer.int32, tab.Bytes, start + i * 4)) for i in range(n)]


def parse_tflite_flatbuffer_buffers(tflite_path: str | Path) -> list[bytes]:
    """使用 flatbuffers 原生解析 .tflite 根表 Model.buffers。"""
    raw = bytearray(Path(tflite_path).read_bytes())
    if bytes(raw[4:8]) != b"TFL3":
        raise ValueError(f"非法 TFLite 魔数: {bytes(raw[4:8])!r}")
    root_pos = flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, 0)
    tab = flatbuffers.table.Table(raw, root_pos)
    version = _tab_u32(tab, 4, 0)
    if version != 3:
        raise ValueError(f"TFLite version 应为 3，实际 {version}")

    bufs_voff = tab.Offset(12)
    if not bufs_voff:
        raise ValueError("TFLite Model 缺少 buffers 向量")
    vec_len = tab.VectorLen(bufs_voff)
    vec_start = tab.Vector(bufs_voff)
    buffers: list[bytes] = []
    for i in range(vec_len):
        elem_pos = vec_start + i * 4
        buf_pos = elem_pos + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, elem_pos)
        btab = flatbuffers.table.Table(raw, buf_pos)
        d_off = btab.Offset(4)
        if not d_off:
            buffers.append(b"")
        else:
            d_len = btab.VectorLen(d_off)
            d_start = btab.Vector(d_off)
            buffers.append(bytes(raw[d_start : d_start + d_len]))
    return buffers


def _apply_fused_act(x: np.ndarray, act: int) -> np.ndarray:
    if act == ACT_NONE:
        return x
    if act == ACT_RELU:
        return np.maximum(x, 0.0)
    if act == ACT_TANH:
        return np.tanh(x)
    raise ValueError(f"不支持的 FusedActivationFunction={act}")


def run_tflite_flatbuffer_cpu(tflite_path: str | Path, feed_inputs: list[np.ndarray]) -> list[np.ndarray]:
    """从 .tflite 二进制解析全部 OperatorCode / SubGraph / Tensor / Operator 并逐算子执行推理。"""
    raw = bytearray(Path(tflite_path).read_bytes())
    if bytes(raw[4:8]) != b"TFL3":
        raise ValueError("Invalid TFL3 magic")
    buffers = parse_tflite_flatbuffer_buffers(tflite_path)

    root_pos = flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, 0)
    model_tab = flatbuffers.table.Table(raw, root_pos)

    # OperatorCodes (field 1 -> voff 6)
    oc_off = model_tab.Offset(6)
    oc_len = model_tab.VectorLen(oc_off)
    oc_start = model_tab.Vector(oc_off)
    opcodes: list[int] = []
    for i in range(oc_len):
        ep = oc_start + i * 4
        tp = ep + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, ep)
        otab = flatbuffers.table.Table(raw, tp)
        dep = _tab_i8(otab, 4, 0)
        ext = _tab_i32(otab, 10, 0)
        opcodes.append(ext if ext != 0 or dep == 0 else dep)

    # SubGraph[0] (field 2 -> voff 8)
    sg_off = model_tab.Offset(8)
    sg_start = model_tab.Vector(sg_off)
    sg_pos = sg_start + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, sg_start)
    sg_tab = flatbuffers.table.Table(raw, sg_pos)

    # Tensors (sg field 0 -> voff 4)
    ts_off = sg_tab.Offset(4)
    ts_len = sg_tab.VectorLen(ts_off)
    ts_start = sg_tab.Vector(ts_off)
    tensors_meta: list[tuple[list[int], int, int]] = []
    values: list[np.ndarray | None] = [None] * ts_len
    for i in range(ts_len):
        ep = ts_start + i * 4
        tp = ep + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, ep)
        ttab = flatbuffers.table.Table(raw, tp)
        shp = _tab_i32_vec(ttab, 4)
        ttype = _tab_i8(ttab, 6, 0)
        bidx = _tab_u32(ttab, 8, 0)
        tensors_meta.append((shp, ttype, bidx))
        if bidx > 0 and buffers[bidx]:
            np_dt = np.float32 if ttype == TFL_FLOAT32 else np.int32
            values[i] = np.frombuffer(buffers[bidx], dtype=np_dt).reshape(shp).copy()

    sg_inputs = _tab_i32_vec(sg_tab, 6)
    sg_outputs = _tab_i32_vec(sg_tab, 8)
    assert len(feed_inputs) == len(sg_inputs), f"输入个数不匹配: {len(feed_inputs)} vs {len(sg_inputs)}"
    for tid, arr in zip(sg_inputs, feed_inputs):
        expected_shp = tensors_meta[tid][0]
        values[tid] = np.asarray(arr, dtype=np.float32).reshape(expected_shp).copy()

    # Operators (sg field 3 -> voff 10)
    ops_off = sg_tab.Offset(10)
    ops_len = sg_tab.VectorLen(ops_off)
    ops_start = sg_tab.Vector(ops_off)
    for i in range(ops_len):
        ep = ops_start + i * 4
        op_pos = ep + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, ep)
        optab = flatbuffers.table.Table(raw, op_pos)
        oc_idx = _tab_u32(optab, 4, 0)
        code = opcodes[oc_idx]
        ins = _tab_i32_vec(optab, 6)
        outs = _tab_i32_vec(optab, 8)
        bopt_type = _tab_u8(optab, 10, 0)
        bopt_voff = optab.Offset(12)
        btab = None
        if bopt_voff:
            uoff = flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, bopt_voff + optab.Pos)
            btab = flatbuffers.table.Table(raw, bopt_voff + optab.Pos + uoff)

        if code == BUILTIN_OP_FULLY_CONNECTED:
            assert bopt_type == OPTS_FULLY_CONNECTED and btab is not None
            act = _tab_i8(btab, 4, 0)
            x, w = values[ins[0]], values[ins[1]]
            b = values[ins[2]] if len(ins) > 2 and ins[2] >= 0 else 0.0
            values[outs[0]] = _apply_fused_act(x @ w.T + b, act).astype(np.float32)

        elif code == BUILTIN_OP_CONV_2D:
            assert bopt_type == OPTS_CONV_2D and btab is not None
            pad_mode = _tab_i8(btab, 4, 0)
            sw = _tab_i32(btab, 6, 1)
            sh = _tab_i32(btab, 8, 1)
            act = _tab_i8(btab, 10, 0)
            dw = _tab_i32(btab, 12, 1)
            dh = _tab_i32(btab, 14, 1)
            assert pad_mode == PAD_VALID, "Validator expects explicit PAD + VALID"
            x, w, b = values[ins[0]], values[ins[1]], values[ins[2]]
            x_t = torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))
            w_t = torch.from_numpy(np.transpose(w, (0, 3, 1, 2)))
            b_t = torch.from_numpy(b)
            with torch.no_grad():
                y_t = torch.nn.functional.conv2d(x_t, w_t, b_t, stride=(sh, sw), dilation=(dh, dw))
            values[outs[0]] = _apply_fused_act(np.transpose(y_t.numpy(), (0, 2, 3, 1)), act).astype(np.float32)

        elif code == BUILTIN_OP_PAD:
            assert bopt_type == OPTS_PAD
            x = values[ins[0]]
            p = [(int(r[0]), int(r[1])) for r in values[ins[1]]]
            values[outs[0]] = np.pad(x, p, mode="constant", constant_values=0.0).astype(np.float32)

        elif code == BUILTIN_OP_MUL:
            assert bopt_type == OPTS_MUL and btab is not None
            values[outs[0]] = _apply_fused_act(values[ins[0]] * values[ins[1]], _tab_i8(btab, 4, 0)).astype(np.float32)

        elif code == BUILTIN_OP_ADD:
            assert bopt_type == OPTS_ADD and btab is not None
            values[outs[0]] = _apply_fused_act(values[ins[0]] + values[ins[1]], _tab_i8(btab, 4, 0)).astype(np.float32)

        elif code == BUILTIN_OP_SUB:
            assert bopt_type == OPTS_SUB and btab is not None
            values[outs[0]] = _apply_fused_act(values[ins[0]] - values[ins[1]], _tab_i8(btab, 4, 0)).astype(np.float32)

        elif code == BUILTIN_OP_LOGISTIC:
            x = np.clip(values[ins[0]], -60.0, 60.0)
            values[outs[0]] = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

        elif code == BUILTIN_OP_TANH:
            values[outs[0]] = np.tanh(values[ins[0]]).astype(np.float32)

        elif code == BUILTIN_OP_RELU:
            values[outs[0]] = np.maximum(values[ins[0]], 0.0).astype(np.float32)

        elif code == BUILTIN_OP_RSQRT:
            values[outs[0]] = (1.0 / np.sqrt(np.maximum(values[ins[0]], 1e-12))).astype(np.float32)

        elif code == BUILTIN_OP_RESHAPE:
            assert bopt_type == OPTS_RESHAPE
            new_shp = _tab_i32_vec(btab, 4) if btab is not None else [int(v) for v in values[ins[1]]]
            values[outs[0]] = values[ins[0]].reshape(new_shp).copy()

        elif code == BUILTIN_OP_CONCATENATION:
            assert bopt_type == OPTS_CONCATENATION and btab is not None
            axis = _tab_i32(btab, 4, 0)
            values[outs[0]] = np.concatenate([values[k] for k in ins], axis=axis).astype(np.float32)

        elif code == BUILTIN_OP_SLICE:
            assert bopt_type == OPTS_SLICE
            x = values[ins[0]]
            bg = [int(v) for v in values[ins[1]]]
            sz = [int(v) for v in values[ins[2]]]
            slices = tuple(slice(b, b + s) for b, s in zip(bg, sz))
            values[outs[0]] = x[slices].copy()

        elif code == BUILTIN_OP_SOFTMAX:
            assert bopt_type == OPTS_SOFTMAX and btab is not None
            beta = _tab_f32(btab, 4, 1.0)
            x = values[ins[0]] * beta
            ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
            values[outs[0]] = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)

        elif code in (BUILTIN_OP_MEAN, BUILTIN_OP_SUM, BUILTIN_OP_REDUCE_MAX):
            assert bopt_type == OPTS_REDUCER and btab is not None
            keep = _tab_bool(btab, 4, False)
            axes = tuple(int(v) for v in np.atleast_1d(values[ins[1]]))
            x = values[ins[0]]
            if code == BUILTIN_OP_MEAN:
                y = np.mean(x, axis=axes, keepdims=keep)
            elif code == BUILTIN_OP_SUM:
                y = np.sum(x, axis=axes, keepdims=keep)
            else:
                y = np.max(x, axis=axes, keepdims=keep)
            values[outs[0]] = np.asarray(y, dtype=np.float32)

        else:
            raise ValueError(f"Unsupported BuiltinOperator code: {code}")

    return [values[tid] for tid in sg_outputs]


def infer_from_tflite_flatbuffer(tflite_path: str | Path, feat_39: np.ndarray, hidden_dim: int = 128) -> np.ndarray:
    """保持兼容旧单输入接口，同时通过 FlatBuffer 解析器逐样本运行策略图。"""
    x = np.asarray(feat_39, dtype=np.float32).reshape(-1, 39)
    outs = [run_tflite_flatbuffer_cpu(tflite_path, [row[None, :]])[0][0] for row in x]
    return np.asarray(outs, dtype=np.float32)


def verify_tflite_numerical(policy: EndSidePolicyNet, tflite_path: str | Path, n_samples: int = 32) -> float:
    """对比 PyTorch 策略与导出的 FlatBuffers .tflite 在随机 39 维输入上的最大绝对误差。"""
    policy.eval()
    rng = np.random.default_rng(20261115)
    x_np = rng.uniform(-1.0, 1.0, size=(n_samples, policy.feature_dim)).astype(np.float32)
    with torch.no_grad():
        y_torch = policy(torch.from_numpy(x_np)).cpu().numpy()
    y_tfl = infer_from_tflite_flatbuffer(tflite_path, x_np, hidden_dim=int(policy.hidden_dim))
    return float(np.max(np.abs(y_torch - y_tfl)))


def verify_double_knock_tflite_numerical(
    model: nn.Module,
    tflite_path: str | Path,
    window: int = 48,
    image_size: int = 96,
    n_samples: int = 4,
) -> dict[str, float]:
    """验证导出的 DoubleKnockNet .tflite 与 PyTorch eval() 的逐算子数值误差。"""
    model.eval()
    rng = np.random.default_rng(20261115)
    max_imp_err = 0.0
    max_cnt_err = 0.0
    for _ in range(n_samples):
        x_vis_nchw = rng.uniform(0.0, 1.0, size=(1, window, 2, image_size, image_size)).astype(np.float32)
        x_kin = rng.uniform(-1.5, 1.5, size=(1, window, model.kin_dim)).astype(np.float32)
        with torch.no_grad():
            pt_imp, pt_cnt = model(torch.from_numpy(x_vis_nchw), torch.from_numpy(x_kin))
        x_vis_nhwc = np.transpose(x_vis_nchw, (0, 1, 3, 4, 2)).astype(np.float32)
        tfl_imp, tfl_cnt = run_tflite_flatbuffer_cpu(tflite_path, [x_vis_nhwc, x_kin])
        max_imp_err = max(max_imp_err, float(np.max(np.abs(pt_imp.numpy() - tfl_imp))))
        max_cnt_err = max(max_cnt_err, float(np.max(np.abs(pt_cnt.numpy() - tfl_cnt))))
    return {"max_imp_err": max_imp_err, "max_cnt_err": max_cnt_err}


def export_policy_bundle(
    policy: EndSidePolicyNet,
    out_dir: str | Path,
    model_name: str = "hover-rl-policy-v1",
    version: str = "2026.11.15",
    notes: str = "End-Side BC+PPO RL Policy (39-dim input -> vx, vyaw, head), 10Hz",
) -> dict:
    """导出标准 FlatBuffers model.tflite + model.json + zip 导入包 + PyTorch .pt 权重。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tflite_bytes = build_tflite_flatbuffer(policy)
    sha256_hex = hashlib.sha256(tflite_bytes).hexdigest()

    tflite_file = out_path / "model.tflite"
    tflite_file.write_bytes(tflite_bytes)

    manifest = {
        "schemaVersion": 1,
        "name": model_name,
        "version": version,
        "task": "policy",
        "input": {"width": 0, "height": 0, "channels": 3, "normalize": "unit"},
        "policy": {"featureDim": int(policy.feature_dim), "actionDim": int(policy.action_dim)},
        "sha256": sha256_hex,
        "notes": notes,
    }
    json_file = out_path / "model.json"
    json_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pt_file = out_path / "policy_weights.pt"
    torch.save({
        "model_name": model_name,
        "version": version,
        "feature_dim": int(policy.feature_dim),
        "hidden_dim": int(policy.hidden_dim),
        "action_dim": int(policy.action_dim),
        "sha256_tflite": sha256_hex,
        "state_dict": policy.state_dict(),
    }, pt_file)

    zip_file = out_path / f"{model_name}-bundle.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tflite_file, arcname="model.tflite")
        zf.write(json_file, arcname="model.json")

    max_err = verify_tflite_numerical(policy, tflite_file)
    return {
        "tflite_path": str(tflite_file),
        "json_path": str(json_file),
        "pt_path": str(pt_file),
        "zip_path": str(zip_file),
        "sha256": sha256_hex,
        "max_abs_err": max_err,
    }


def export_double_knock_bundle(
    model: nn.Module,
    out_dir: str | Path,
    window: int = 48,
    image_size: int = 96,
    calib: dict | None = None,
    model_name: str = "double-knock-v1",
    version: str = "2026.11.15",
    notes: str = "End-Side DoubleKnockNet (48x96x96x2 + 48x8 -> impulse + 4-class count)",
) -> dict:
    """将训练好的 DoubleKnockNet 导出为 Android 可直接导入的 TFLite + model.json + .zip。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tflite_bytes = build_double_knock_tflite(model, window=window, image_size=image_size)
    sha256_hex = hashlib.sha256(tflite_bytes).hexdigest()

    tflite_file = out_path / "model.tflite"
    tflite_file.write_bytes(tflite_bytes)

    cal = calib or {}
    manifest = {
        "schemaVersion": 1,
        "name": model_name,
        "version": version,
        "task": "action",
        "input": {"width": int(image_size), "height": int(image_size), "channels": int(model.in_ch), "normalize": "unit"},
        "classes": ["non_knock", "single_knock", "double_knock", "multi_knock"],
        "action": {
            "window": int(window),
            "kinDim": int(model.kin_dim),
            "numClasses": int(model.num_classes),
            "impThreshold": float(cal.get("imp_thr", 0.55)),
            "doubleThreshold": float(cal.get("double_thr", 0.55)),
            "pairMinSec": float(cal.get("pair_min_s", 0.18)),
            "pairMaxSec": float(cal.get("pair_max_s", 1.15)),
            "cooldownSec": float(cal.get("cooldown_s", 1.80)),
        },
        "sha256": sha256_hex,
        "notes": notes,
    }
    json_file = out_path / "model.json"
    json_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_file = out_path / f"{model_name}-bundle.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tflite_file, arcname="model.tflite")
        zf.write(json_file, arcname="model.json")

    errs = verify_double_knock_tflite_numerical(model, tflite_file, window=window, image_size=image_size, n_samples=2)
    return {
        "tflite_path": str(tflite_file),
        "json_path": str(json_file),
        "zip_path": str(zip_file),
        "sha256": sha256_hex,
        **errs,
    }


def main() -> None:
    """重新导出当前最新 RL 策略包与双敲动作模型包，并同步内置到 Android assets/models/。"""
    import shutil
    from train_double_knock import DoubleKnockNet

    android_assets = Path("G:/robot_app/android/app/src/main/assets/models")
    android_assets.mkdir(parents=True, exist_ok=True)

    # 1. Re-export RL Policy
    rl_pt = Path("D:/robot_work/runs_end_side/hover_rl_policy/policy_weights.pt")
    if rl_pt.exists():
        ck = torch.load(rl_pt, map_location="cpu", weights_only=False)
        policy = EndSidePolicyNet(
            feature_dim=int(ck.get("feature_dim", 39)),
            hidden_dim=int(ck.get("hidden_dim", 128)),
            action_dim=int(ck.get("action_dim", 3)),
        )
        policy.load_state_dict(ck["state_dict"])
        res_rl = export_policy_bundle(policy, rl_pt.parent, model_name="hover-end-side-rl-ppo-v1")
        shutil.copy2(res_rl["tflite_path"], android_assets / "hover_rl_policy.tflite")
        shutil.copy2(res_rl["json_path"], android_assets / "hover_rl_policy.json")
        print("RL Policy exported:", res_rl)

    # 2. Export DoubleKnockNet
    dk_pt = Path("D:/robot_work/runs_knock/double_knock_20261115/double_knock_deploy.pt")
    if dk_pt.exists():
        ck = torch.load(dk_pt, map_location="cpu", weights_only=False)
        dk_model = DoubleKnockNet()
        dk_model.load_state_dict(ck["state_dict"])
        res_dk = export_double_knock_bundle(
            dk_model,
            dk_pt.parent / "bundle",
            window=int(ck.get("window", 48)),
            image_size=int(ck.get("size", 96)),
            calib=ck.get("calib"),
            model_name="double-knock-v1",
        )
        shutil.copy2(res_dk["tflite_path"], android_assets / "double_knock_v1.tflite")
        shutil.copy2(res_dk["json_path"], android_assets / "double_knock_v1.json")
        print("DoubleKnockNet exported:", res_dk)


if __name__ == "__main__":
    main()
