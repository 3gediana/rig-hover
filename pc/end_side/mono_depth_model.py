# -*- coding: utf-8 -*-
"""mono_depth_model.py — RIG-Hover 单目尺度深度与桌面凸起障碍物联合估计网络及 TFLite 导出器。

针对桌面自平衡机器人 RIG-Hover 单目头部相机（240x240 采集 -> 96x96 归一化输入，
相机离桌面高度 h_cam = 0.115 m，水平视场角 FOV_H = 60°）设计：
1. 输入张量 (`rgb_in`): `[1, 96, 96, 3]` float32，值域 `[0.0, 1.0]`（兼容彩色与 GC0308 三通道灰度流）。
2. 输出张量 (`depth_elev_out`): `[1, 48, 48, 2]` float32，值域 `(0.0, 1.0)`（末端 Sigmoid / LOGISTIC）：
   - 通道 0 (`inv_depth_01`)：归一化逆深度，映射为物理尺度深度（米）：
     `inv_d = 0.35 + 4.15 * inv_depth_01` (`0.35 .. 4.50 m^-1`)，`depth_m = 1.0 / inv_d` (`0.222 .. 2.857 m`)。
   - 通道 1 (`elev_prob_01`)：像素属于高出桌面平面 `Z = 0` 超过 `3 cm` 的前景立体物体/障碍物/人手概率。
3. 100% 纯 TFLite 核心内置算子图（仅使用 `PAD` + `CONV_2D` + `ADD` + `LOGISTIC`），无任何自定义算子，
   支持通过 `export_tflite_bundle._TFLiteGraphBuilder` 直接导出标准 FlatBuffers v3 `.tflite`，
   且与 PyTorch `.eval()` 逐像素数值误差 `< 1e-4`。
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import flatbuffers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from export_tflite_bundle import (  # noqa: E402
    ACT_NONE,
    ACT_RELU,
    PAD_VALID,
    TFL_FLOAT32,
    _fold_conv_bn,
    _tab_i8,
    _tab_i32_vec,
    _tab_u32,
    _TFLiteGraphBuilder,
    parse_tflite_flatbuffer_buffers,
    run_tflite_flatbuffer_cpu,
)

# ---------------------------------------------------------------------------
# 物理相机与逆深度参数化常量
# ---------------------------------------------------------------------------
INPUT_H: int = 96
INPUT_W: int = 96
INPUT_C: int = 3

OUTPUT_H: int = 48
OUTPUT_W: int = 48
OUTPUT_C: int = 2

INV_DEPTH_MIN: float = 0.35
INV_DEPTH_SPAN: float = 4.15
INV_DEPTH_MAX: float = INV_DEPTH_MIN + INV_DEPTH_SPAN  # 4.50 m^-1

DEPTH_MIN_M: float = 1.0 / INV_DEPTH_MAX  # ~0.222222 m
DEPTH_MAX_M: float = 1.0 / INV_DEPTH_MIN  # ~2.857143 m

DEFAULT_H_CAM_M: float = 0.115
DEFAULT_FOV_H_DEG: float = 60.0
ELEV_THRESHOLD_M: float = 0.03


def inv_depth_01_to_metric_m(inv_depth_01: torch.Tensor | np.ndarray | float) -> Any:
    """将 `[0, 1]` 归一化逆深度映射为物理米制深度 `depth_m` (`~0.222 .. 2.857 m`)。"""
    if isinstance(inv_depth_01, torch.Tensor):
        inv_c = torch.clamp(inv_depth_01, 0.0, 1.0)
        inv_d = INV_DEPTH_MIN + INV_DEPTH_SPAN * inv_c
        return 1.0 / inv_d
    if isinstance(inv_depth_01, np.ndarray):
        inv_c = np.clip(inv_depth_01, 0.0, 1.0).astype(np.float32)
        inv_d = np.float32(INV_DEPTH_MIN) + np.float32(INV_DEPTH_SPAN) * inv_c
        return (np.float32(1.0) / inv_d).astype(np.float32)
    inv_c = min(max(float(inv_depth_01), 0.0), 1.0)
    return 1.0 / (INV_DEPTH_MIN + INV_DEPTH_SPAN * inv_c)


def metric_m_to_inv_depth_01(depth_m: torch.Tensor | np.ndarray | float) -> Any:
    """将物理米制深度 `depth_m` 逆映射回 `[0, 1]` 归一化逆深度监督目标。"""
    if isinstance(depth_m, torch.Tensor):
        d_c = torch.clamp(depth_m, DEPTH_MIN_M, DEPTH_MAX_M)
        inv_d = 1.0 / d_c
        return torch.clamp((inv_d - INV_DEPTH_MIN) / INV_DEPTH_SPAN, 0.0, 1.0)
    if isinstance(depth_m, np.ndarray):
        d_c = np.clip(depth_m, DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)
        inv_d = np.float32(1.0) / d_c
        return np.clip((inv_d - np.float32(INV_DEPTH_MIN)) / np.float32(INV_DEPTH_SPAN), 0.0, 1.0).astype(np.float32)
    d_c = min(max(float(depth_m), DEPTH_MIN_M), DEPTH_MAX_M)
    inv_d = 1.0 / d_c
    return min(max((inv_d - INV_DEPTH_MIN) / INV_DEPTH_SPAN, 0.0), 1.0)


def camera_rotation_matrix(pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """构造自车桌面系 (+X 正前, +Y 右侧, +Z 竖直向上) 下的相机姿态旋转矩阵 R_cam_to_world。

    约定：
      - `pitch_deg > 0` 表示镜头向下俯视桌面（绕 +Y 轴右手正向旋转：`X_new = X*cos + Z*sin`, `Z_new = -X*sin + Z*cos`）；
      - `roll_deg > 0` 表示机身向右横滚（绕 +X 轴旋转）。
    """
    th = math.radians(float(pitch_deg))
    ph = math.radians(float(roll_deg))
    ct, st = math.cos(th), math.sin(th)
    cp, sp = math.cos(ph), math.sin(ph)

    r_pitch = np.array(
        [
            [ct, 0.0, st],
            [0.0, 1.0, 0.0],
            [-st, 0.0, ct],
        ],
        dtype=np.float32,
    )
    r_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    return (r_pitch @ r_roll).astype(np.float32)


def make_camera_ray_grid(
    height: int = OUTPUT_H,
    width: int = OUTPUT_W,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """生成 `[height, width, 3]` 相机射线向量 `v_world`，满足 `P(d) = [0, 0, h_cam] + d * v_world`，
    其中参数 `d` 严格等于针孔相机前向平面深度 `depth_m`。
    """
    tan_half = math.tan(math.radians(float(fov_h_deg) * 0.5))
    u = ((np.arange(width, dtype=np.float32) + 0.5) / float(width)) * 2.0 - 1.0  # [-1 左 .. +1 右]
    v = 1.0 - ((np.arange(height, dtype=np.float32) + 0.5) / float(height)) * 2.0  # [+1 上 .. -1 下]
    uu, vv = np.meshgrid(u, v)
    d_cam = np.stack(
        [
            np.ones_like(uu, dtype=np.float32),
            (uu * tan_half).astype(np.float32),
            (vv * tan_half).astype(np.float32),
        ],
        axis=-1,
    )  # [H, W, 3]
    rot = camera_rotation_matrix(pitch_deg=pitch_deg, roll_deg=roll_deg)
    return np.einsum("ij,hwj->hwi", rot, d_cam).astype(np.float32)


def unproject_depth_to_3d_splats(
    depth_m: np.ndarray,
    elev_prob_01: np.ndarray,
    rgb_in: np.ndarray | None = None,
    h_cam_m: float = DEFAULT_H_CAM_M,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """将 `[48, 48]` 预测深度图与高程概率图反投影为自车系 3D 彩色高斯点云 (`Z = 0` 为桌面)。"""
    d_map = np.asarray(depth_m, dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
    e_map = np.asarray(elev_prob_01, dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
    rays = make_camera_ray_grid(
        height=OUTPUT_H,
        width=OUTPUT_W,
        fov_h_deg=fov_h_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    cam_origin = np.array([0.0, 0.0, float(h_cam_m)], dtype=np.float32)
    xyz = cam_origin[None, None, :] + d_map[:, :, None] * rays  # [48, 48, 3]

    if rgb_in is not None:
        rgb_arr = np.asarray(rgb_in, dtype=np.float32)
        if rgb_arr.shape == (INPUT_H, INPUT_W, INPUT_C):
            # 2x2 均值下采样对齐 48x48 射线像素中心
            rgb_48 = rgb_arr.reshape(OUTPUT_H, 2, OUTPUT_W, 2, INPUT_C).mean(axis=(1, 3))
        elif rgb_arr.shape == (OUTPUT_H, OUTPUT_W, INPUT_C):
            rgb_48 = rgb_arr
        else:
            raise ValueError(f"Unexpected rgb_in shape {rgb_arr.shape}")
        rgb_48 = np.clip(rgb_48, 0.0, 1.0).astype(np.float32)
    else:
        rgb_48 = np.full((OUTPUT_H, OUTPUT_W, 3), 0.75, dtype=np.float32)

    px_ang = 2.0 * math.tan(math.radians(float(fov_h_deg) * 0.5)) / float(OUTPUT_W)
    splat_radius_m = (d_map * px_ang * 0.85).astype(np.float32)

    return {
        "xyz_grid": xyz,
        "xyz_flat": xyz.reshape(-1, 3),
        "rgb_flat": rgb_48.reshape(-1, 3),
        "elev_flat": e_map.reshape(-1),
        "depth_flat": d_map.reshape(-1),
        "radius_flat": splat_radius_m.reshape(-1),
    }


# ---------------------------------------------------------------------------
# HoverMonoDepthNet 神经网络架构 (严格对齐 TFLite 内置算子)
# ---------------------------------------------------------------------------

class DilatedResBlock(nn.Module):
    """空洞残差块：对称零填充 `pad=d` + 两层 `3x3` 空洞卷积 (`dilation=d`) + 残差相加 + ReLU。"""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        d = int(dilation)
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.dilation = d

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=d, dilation=d, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=d, dilation=d, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.proj: nn.Sequential | None = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x) if self.proj is not None else x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + skip)


class HoverMonoDepthNet(nn.Module):
    """轻量级多尺度空洞残差单目深度与障碍物高程网络。

    通道拓扑：
      `3 (96x96) -> 24 (s=2, 48x48) -> 32 (d=1 res) -> 32 (d=2 res) -> 32 (d=4 res)`
      `-> 32 (d=8 res) + skip(d=1) -> 24 (1x1 RELU) -> 2 (1x1) -> sigmoid`
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(INPUT_C, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(24)

        self.res_d1 = DilatedResBlock(24, 32, dilation=1)
        self.res_d2 = DilatedResBlock(32, 32, dilation=2)
        self.res_d4 = DilatedResBlock(32, 32, dilation=4)
        self.res_d8 = DilatedResBlock(32, 32, dilation=8)

        self.head_conv1 = nn.Conv2d(32, 24, kernel_size=1, stride=1, padding=0, bias=False)
        self.head_bn1 = nn.BatchNorm2d(24)
        self.head_out = nn.Conv2d(24, OUTPUT_C, kernel_size=1, stride=1, padding=0, bias=True)

        # 初始化末端偏置使初始逆深度居中 (~0.45 -> ~0.45 m)、初始障碍物概率 ~0.18
        nn.init.constant_(self.head_out.bias[0], -0.20)
        nn.init.constant_(self.head_out.bias[1], -1.50)

    def forward_logits(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """输入 `[B, 3, 96, 96]` (NCHW)，输出未过 Sigmoid 的 `[B, 2, 48, 48]` logits。"""
        s0 = F.relu(self.stem_bn(self.stem_conv(x_nchw)))
        b1 = self.res_d1(s0)
        b2 = self.res_d2(b1)
        b4 = self.res_d4(b2)
        b8 = self.res_d8(b4)
        ms = b8 + b1
        h = F.relu(self.head_bn1(self.head_conv1(ms)))
        return self.head_out(h)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        """输入 `[B, 3, 96, 96]` float32 (`[0, 1]`)，输出 `[B, 2, 48, 48]` float32 (`(0, 1)`)。"""
        return torch.sigmoid(self.forward_logits(x_nchw))

    @torch.no_grad()
    def infer_nhwc(self, rgb_nhwc: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """便捷 NHWC 推理接口：输入 `[96, 96, 3]` 或 `[B, 96, 96, 3]`，输出字典含米制深度。"""
        was_training = self.training
        self.eval()
        try:
            arr = rgb_nhwc.detach().cpu().numpy() if isinstance(rgb_nhwc, torch.Tensor) else np.asarray(rgb_nhwc)
            single = arr.ndim == 3
            if single:
                arr = arr[None, ...]
            if arr.shape[1:] != (INPUT_H, INPUT_W, INPUT_C):
                raise ValueError(f"Expected NHWC [..., 96, 96, 3], got {arr.shape}")
            dev = next(self.parameters()).device
            x_t = torch.from_numpy(np.transpose(arr.astype(np.float32), (0, 3, 1, 2))).to(dev)
            y_nchw = self.forward(x_t).cpu().numpy()
            y_nhwc = np.transpose(y_nchw, (0, 2, 3, 1)).astype(np.float32)
            inv_01 = y_nhwc[..., 0]
            elev_01 = y_nhwc[..., 1]
            depth_m = inv_depth_01_to_metric_m(inv_01)
            if single:
                return {
                    "out_nhwc": y_nhwc[0],
                    "inv_depth_01": inv_01[0],
                    "depth_m": depth_m[0],
                    "elev_prob_01": elev_01[0],
                }
            return {
                "out_nhwc": y_nhwc,
                "inv_depth_01": inv_01,
                "depth_m": depth_m,
                "elev_prob_01": elev_01,
            }
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# FlatBuffers TFLite v3 图构造、加载与数值一致性校验
# ---------------------------------------------------------------------------

def build_mono_depth_tflite(model: HoverMonoDepthNet) -> bytes:
    """将 `HoverMonoDepthNet` 折叠 BN 后导出为纯内置算子的标准 TFLite v3 FlatBuffers 字节流。

    图输入: `rgb_in` `[1, 96, 96, 3]` float32
    图输出: `depth_elev_out` `[1, 48, 48, 2]` float32
    """
    model.eval()
    gb = _TFLiteGraphBuilder("RIG-Hover HoverMonoDepthNet (TFLite v3 Pure Builtins)", initial_capacity=524288)
    rgb_in = gb.add_input("rgb_in", [1, INPUT_H, INPUT_W, INPUT_C], TFL_FLOAT32)

    def _emit_conv_bn(
        name: str,
        x_tid: int,
        conv: nn.Conv2d,
        bn: nn.BatchNorm2d,
        stride: int = 1,
        dilation: int = 1,
        pad_hw: int = 0,
        fused_act: int = ACT_NONE,
    ) -> int:
        w_nchw, b_vec = _fold_conv_bn(conv, bn)
        w_ohwi = np.transpose(w_nchw, (0, 2, 3, 1)).astype(np.float32)
        cur = x_tid
        if pad_hw > 0:
            cur = gb.pad(f"{name}_pad", cur, [[0, 0], [pad_hw, pad_hw], [pad_hw, pad_hw], [0, 0]])
        return gb.conv2d(
            name,
            cur,
            w_ohwi,
            b_vec,
            stride=stride,
            dilation=dilation,
            padding=PAD_VALID,
            fused_act=fused_act,
        )

    def _emit_res_block(name: str, x_tid: int, blk: DilatedResBlock) -> int:
        d = int(blk.dilation)
        if blk.proj is not None:
            skip_tid = _emit_conv_bn(
                f"{name}_proj",
                x_tid,
                blk.proj[0],
                blk.proj[1],
                stride=1,
                dilation=1,
                pad_hw=0,
                fused_act=ACT_NONE,
            )
        else:
            skip_tid = x_tid
        h1 = _emit_conv_bn(
            f"{name}_c1",
            x_tid,
            blk.conv1,
            blk.bn1,
            stride=1,
            dilation=d,
            pad_hw=d,
            fused_act=ACT_RELU,
        )
        h2 = _emit_conv_bn(
            f"{name}_c2",
            h1,
            blk.conv2,
            blk.bn2,
            stride=1,
            dilation=d,
            pad_hw=d,
            fused_act=ACT_NONE,
        )
        return gb.add(f"{name}_out", h2, skip_tid, fused_act=ACT_RELU)

    s0 = _emit_conv_bn("stem", rgb_in, model.stem_conv, model.stem_bn, stride=2, dilation=1, pad_hw=1, fused_act=ACT_RELU)
    b1 = _emit_res_block("res_d1", s0, model.res_d1)
    b2 = _emit_res_block("res_d2", b1, model.res_d2)
    b4 = _emit_res_block("res_d4", b2, model.res_d4)
    b8 = _emit_res_block("res_d8", b4, model.res_d8)
    ms = gb.add("ms_skip", b8, b1, fused_act=ACT_NONE)

    hc1 = _emit_conv_bn("head_c1", ms, model.head_conv1, model.head_bn1, stride=1, dilation=1, pad_hw=0, fused_act=ACT_RELU)
    w_out = np.transpose(model.head_out.weight.detach().cpu().numpy().astype(np.float32), (0, 2, 3, 1))
    b_out = model.head_out.bias.detach().cpu().numpy().astype(np.float32)
    logits = gb.conv2d("head_logits", hc1, w_out, b_out, stride=1, dilation=1, padding=PAD_VALID, fused_act=ACT_NONE)
    depth_elev_out = gb.sigmoid("depth_elev_out", logits)

    gb.set_outputs([depth_elev_out])
    return gb.finish()


def parse_named_tflite_constants(tflite_path: str | Path) -> dict[str, np.ndarray]:
    """从 `.tflite` 二进制提取所有命名常量张量，支持无额外 `.pt` 文件逆向恢复 PyTorch 模型。"""
    raw = bytearray(Path(tflite_path).read_bytes())
    if bytes(raw[4:8]) != b"TFL3":
        raise ValueError(f"Invalid TFL3 magic in {tflite_path}")
    buffers = parse_tflite_flatbuffer_buffers(tflite_path)

    root_pos = flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, 0)
    model_tab = flatbuffers.table.Table(raw, root_pos)
    sg_off = model_tab.Offset(8)
    sg_start = model_tab.Vector(sg_off)
    sg_pos = sg_start + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, sg_start)
    sg_tab = flatbuffers.table.Table(raw, sg_pos)

    ts_off = sg_tab.Offset(4)
    ts_len = sg_tab.VectorLen(ts_off)
    ts_start = sg_tab.Vector(ts_off)
    named: dict[str, np.ndarray] = {}
    for i in range(ts_len):
        ep = ts_start + i * 4
        tp = ep + flatbuffers.encode.Get(flatbuffers.packer.uoffset, raw, ep)
        ttab = flatbuffers.table.Table(raw, tp)
        shp = _tab_i32_vec(ttab, 4)
        ttype = _tab_i8(ttab, 6, 0)
        bidx = _tab_u32(ttab, 8, 0)
        nm_off = ttab.Offset(10)
        name = ttab.String(nm_off + ttab.Pos).decode("utf-8") if nm_off else f"tensor_{i}"
        if bidx > 0 and buffers[bidx]:
            np_dt = np.float32 if ttype == TFL_FLOAT32 else np.int32
            named[name] = np.frombuffer(buffers[bidx], dtype=np_dt).reshape(shp).copy()
    return named


def load_hover_mono_depth_from_tflite(tflite_path: str | Path) -> HoverMonoDepthNet:
    """直接从导出的 `hover_mono_depth_v1.tflite` 重建等价 PyTorch `HoverMonoDepthNet` 模型。"""
    consts = parse_named_tflite_constants(tflite_path)
    model = HoverMonoDepthNet()

    def _assign_folded(prefix: str, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        w_ohwi = consts[f"{prefix}_w"]
        b_vec = consts[f"{prefix}_b"]
        w_oihw = np.transpose(w_ohwi, (0, 3, 1, 2)).astype(np.float32)
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(w_oihw))
            bn.weight.fill_(1.0)
            bn.bias.copy_(torch.from_numpy(b_vec))
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0 - float(bn.eps))

    _assign_folded("stem", model.stem_conv, model.stem_bn)
    for name, blk in (
        ("res_d1", model.res_d1),
        ("res_d2", model.res_d2),
        ("res_d4", model.res_d4),
        ("res_d8", model.res_d8),
    ):
        if blk.proj is not None:
            _assign_folded(f"{name}_proj", blk.proj[0], blk.proj[1])
        _assign_folded(f"{name}_c1", blk.conv1, blk.bn1)
        _assign_folded(f"{name}_c2", blk.conv2, blk.bn2)

    _assign_folded("head_c1", model.head_conv1, model.head_bn1)
    with torch.no_grad():
        w_out = np.transpose(consts["head_logits_w"], (0, 3, 1, 2)).astype(np.float32)
        b_out = consts["head_logits_b"].astype(np.float32)
        model.head_out.weight.copy_(torch.from_numpy(w_out))
        model.head_out.bias.copy_(torch.from_numpy(b_out))

    model.eval()
    return model


def infer_mono_depth_tflite(tflite_path: str | Path, rgb_nhwc: np.ndarray) -> dict[str, np.ndarray]:
    """使用纯 FlatBuffers CPU 执行器运行 `hover_mono_depth_v1.tflite` 并解码米制深度与高程概率。"""
    arr = np.asarray(rgb_nhwc, dtype=np.float32)
    single = arr.ndim == 3
    if single:
        arr = arr[None, ...]
    if arr.shape[1:] != (INPUT_H, INPUT_W, INPUT_C):
        raise ValueError(f"Expected [B, 96, 96, 3], got {arr.shape}")
    outs = [run_tflite_flatbuffer_cpu(tflite_path, [arr[i : i + 1]])[0][0] for i in range(arr.shape[0])]
    y_nhwc = np.stack(outs, axis=0).astype(np.float32)
    inv_01 = y_nhwc[..., 0]
    elev_01 = y_nhwc[..., 1]
    depth_m = inv_depth_01_to_metric_m(inv_01)
    if single:
        return {
            "out_nhwc": y_nhwc[0],
            "inv_depth_01": inv_01[0],
            "depth_m": depth_m[0],
            "elev_prob_01": elev_01[0],
        }
    return {
        "out_nhwc": y_nhwc,
        "inv_depth_01": inv_01,
        "depth_m": depth_m,
        "elev_prob_01": elev_01,
    }


def verify_mono_depth_tflite_parity(
    model: HoverMonoDepthNet,
    tflite_path: str | Path,
    n_samples: int = 8,
    seed: int = 20261118,
) -> float:
    """校验 PyTorch `.eval()` 与 FlatBuffers `.tflite` 在随机 `[1, 96, 96, 3]` 输入上的最大绝对误差。"""
    model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(int(n_samples)):
        x_nhwc = rng.uniform(0.0, 1.0, size=(1, INPUT_H, INPUT_W, INPUT_C)).astype(np.float32)
        x_nchw = np.transpose(x_nhwc, (0, 3, 1, 2))
        with torch.no_grad():
            pt_nchw = model(torch.from_numpy(x_nchw).to(dev)).cpu().numpy()
        pt_nhwc = np.transpose(pt_nchw, (0, 2, 3, 1))
        tfl_nhwc = run_tflite_flatbuffer_cpu(tflite_path, [x_nhwc])[0]
        diff = float(np.max(np.abs(pt_nhwc - tfl_nhwc)))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def benchmark_cpu_latency_ms(model: HoverMonoDepthNet, n_runs: int = 60) -> float:
    """使用 NHWC (`channels_last`) 与算子融合优化测算单帧 CPU 推理时延（对齐 TFLite XNNPACK 融合行为）。"""
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(4, max(1, prev_threads)))
        m_cpu = HoverMonoDepthNet()
        m_cpu.load_state_dict(model.state_dict())
        m_cpu.eval().to(memory_format=torch.channels_last)
        x_sample = torch.rand(1, INPUT_C, INPUT_H, INPUT_W).to(memory_format=torch.channels_last)
        with torch.no_grad():
            traced = torch.jit.trace(m_cpu, x_sample)
            opt_mod = torch.jit.optimize_for_inference(traced)
            for _ in range(12):
                _ = opt_mod(x_sample)
            t0 = time.perf_counter()
            for _ in range(int(n_runs)):
                _ = opt_mod(x_sample)
            return (time.perf_counter() - t0) * 1000.0 / float(n_runs)
    finally:
        torch.set_num_threads(prev_threads)


def export_mono_depth_bundle(
    model: HoverMonoDepthNet,
    out_dir: str | Path,
    android_assets_dir: str | Path | None = None,
    version: str = "2026.11.18",
) -> dict[str, Any]:
    """导出 `hover_mono_depth_v1.tflite`, `hover_mono_depth_v1.json`, `hover-mono-depth-v1-bundle.zip`，
    并可选同步拷贝至 Android `assets/models/`。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tflite_bytes = build_mono_depth_tflite(model)
    sha256_hex = hashlib.sha256(tflite_bytes).hexdigest()

    tflite_file = out_path / "hover_mono_depth_v1.tflite"
    tflite_file.write_bytes(tflite_bytes)

    manifest = {
        "schemaVersion": 1,
        "name": "hover-mono-depth-v1",
        "version": version,
        "task": "depth",
        "input": {
            "width": INPUT_W,
            "height": INPUT_H,
            "channels": INPUT_C,
            "normalize": "unit",
        },
        "output": {
            "width": OUTPUT_W,
            "height": OUTPUT_H,
            "channels": OUTPUT_C,
            "invDepthMin": INV_DEPTH_MIN,
            "invDepthSpan": INV_DEPTH_SPAN,
            "depthMinM": round(DEPTH_MIN_M, 4),
            "depthMaxM": round(DEPTH_MAX_M, 4),
            "elevThresholdM": ELEV_THRESHOLD_M,
        },
        "camera": {
            "heightM": DEFAULT_H_CAM_M,
            "fovHDeg": DEFAULT_FOV_H_DEG,
        },
        "sha256": sha256_hex,
        "notes": "RIG-Hover Monocular Metric Depth & Obstacle Elevation Net (96x96x3 -> 48x48x2, TFLite v3 pure builtins)",
    }
    json_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    json_file = out_path / "hover_mono_depth_v1.json"
    json_file.write_text(json_text, encoding="utf-8")

    zip_file = out_path / "hover-mono-depth-v1-bundle.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tflite_file, arcname="model.tflite")
        zf.write(json_file, arcname="model.json")

    max_abs_diff = verify_mono_depth_tflite_parity(model, tflite_file, n_samples=8)

    copied_android: list[str] = []
    if android_assets_dir is not None:
        assets_path = Path(android_assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)
        dst_tfl = assets_path / "hover_mono_depth_v1.tflite"
        dst_jsn = assets_path / "hover_mono_depth_v1.json"
        shutil.copy2(tflite_file, dst_tfl)
        shutil.copy2(json_file, dst_jsn)
        copied_android = [str(dst_tfl), str(dst_jsn)]

    return {
        "tflite_path": str(tflite_file),
        "json_path": str(json_file),
        "zip_path": str(zip_file),
        "tflite_bytes": len(tflite_bytes),
        "sha256": sha256_hex,
        "max_abs_diff": max_abs_diff,
        "android_assets": copied_android,
    }
