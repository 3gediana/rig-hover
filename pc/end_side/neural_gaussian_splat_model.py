# -*- coding: utf-8 -*-
"""neural_gaussian_splat_model.py — 前馈式 6 通道神经 3D 高斯椭球生成网络与持久贝叶斯空间记忆地图引擎。

采用 Splatter-Image 前馈单目 3DGS 范式，将 `96x96` RGB/灰度相机帧端到端映射为 `48x48` 稠密神经 3D 高斯图：
  输入张量 (`rgb_in`): `[1, 96, 96, 3]` float32，归一化值域 `[0.0, 1.0]`
  输出张量 (`gaussian_splat_out`): `[1, 48, 48, 6]` float32，末端 Sigmoid 值域 `(0.0, 1.0)`：
    - `ch0` (`inv_depth_01`): 归一化逆深度 -> `depth_m = 1.0 / (0.35 + 4.15 * ch0)` (`0.222 .. 2.857 m`)
    - `ch1` (`elev_01`)     : 相对 `Z = 0` 桌面凸起概率 (`Z > 3 cm` 立体前景指示)
    - `ch2` (`scale_xy_01`) : 神经预测水平高斯半径 -> `0.008 .. 0.046 m`（平坦桌面大圆盘、边缘/手指精细小核）
    - `ch3` (`scale_z_01`)  : 神经预测垂直各向异性半径 -> `0.003 .. 0.042 m`（桌面薄圆盘、立面杯身/人手竖椭球）
    - `ch4` (`opacity_01`)  : 神经预测不透明度 `alpha` -> `0.02 .. 0.99`（实体表面高不透明、抑制远景/反光伪影）
    - `ch5` (`precision_01`): 神经预测异方差观测精度 `tau = sigma^-2` -> `25.0 .. 10000.0 m^-2`（驱动多视角卡尔曼修正）

同时实现具备无界空间拓展与 `100%` 历史区域保留能力的持久空间记忆引擎 (`PersistentGaussianMemoryMap`)
以及基于持久 3DGS 障碍记忆的全局任意点回溯规划导航器 (`GlobalMemoryNavigator`)。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PERCEPTION_DIR = HERE.parent / "perception"
for _p in (str(HERE), str(PERCEPTION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from end_side_controller import DriveCommand, EndSideSafetyShield, MODE_APPROACH  # noqa: E402
from export_tflite_bundle import (  # noqa: E402
    ACT_NONE,
    ACT_RELU,
    PAD_VALID,
    TFL_FLOAT32,
    _fold_conv_bn,
    _TFLiteGraphBuilder,
    run_tflite_flatbuffer_cpu,
)
from mono_depth_model import (  # noqa: E402
    DEFAULT_FOV_H_DEG,
    DEFAULT_H_CAM_M,
    ELEV_THRESHOLD_M,
    INPUT_C,
    INPUT_H,
    INPUT_W,
    INV_DEPTH_MIN,
    INV_DEPTH_SPAN,
    OUTPUT_H,
    OUTPUT_W,
    DilatedResBlock,
    camera_rotation_matrix,
    inv_depth_01_to_metric_m,
    make_camera_ray_grid,
    metric_m_to_inv_depth_01,
    parse_named_tflite_constants,
)

# ---------------------------------------------------------------------------
# 1. 神经 3D 高斯 6 通道物理参数化常量与互逆映射
# ---------------------------------------------------------------------------
SPLAT_OUT_C: int = 6

SCALE_XY_MIN_M: float = 0.008
SCALE_XY_MAX_M: float = 0.046
SCALE_XY_SPAN_M: float = SCALE_XY_MAX_M - SCALE_XY_MIN_M  # 0.038 m

SCALE_Z_MIN_M: float = 0.003
SCALE_Z_MAX_M: float = 0.042
SCALE_Z_SPAN_M: float = SCALE_Z_MAX_M - SCALE_Z_MIN_M  # 0.039 m

OPACITY_MIN: float = 0.02
OPACITY_MAX: float = 0.99
OPACITY_SPAN: float = OPACITY_MAX - OPACITY_MIN  # 0.97

TAU_MIN: float = 25.0      # sigma_max = 0.200 m
TAU_MAX: float = 10000.0   # sigma_min = 0.010 m
LOG_TAU_MIN: float = math.log(TAU_MIN)
LOG_TAU_MAX: float = math.log(TAU_MAX)
LOG_TAU_SPAN: float = LOG_TAU_MAX - LOG_TAU_MIN

CHANNEL_NAMES: tuple[str, ...] = (
    "inv_depth_01",
    "elev_01",
    "scale_xy_01",
    "scale_z_01",
    "opacity_01",
    "precision_01",
)


def scale_xy_01_to_m(v01: torch.Tensor | np.ndarray | float) -> Any:
    if isinstance(v01, torch.Tensor):
        return SCALE_XY_MIN_M + SCALE_XY_SPAN_M * torch.clamp(v01, 0.0, 1.0)
    if isinstance(v01, np.ndarray):
        return (np.float32(SCALE_XY_MIN_M) + np.float32(SCALE_XY_SPAN_M) * np.clip(v01, 0.0, 1.0)).astype(np.float32)
    return SCALE_XY_MIN_M + SCALE_XY_SPAN_M * min(max(float(v01), 0.0), 1.0)


def scale_xy_m_to_01(scale_m: torch.Tensor | np.ndarray | float) -> Any:
    if isinstance(scale_m, torch.Tensor):
        return torch.clamp((scale_m - SCALE_XY_MIN_M) / SCALE_XY_SPAN_M, 0.0, 1.0)
    if isinstance(scale_m, np.ndarray):
        return np.clip((scale_m - SCALE_XY_MIN_M) / SCALE_XY_SPAN_M, 0.0, 1.0).astype(np.float32)
    return min(max((float(scale_m) - SCALE_XY_MIN_M) / SCALE_XY_SPAN_M, 0.0), 1.0)


def scale_z_01_to_m(v01: torch.Tensor | np.ndarray | float) -> Any:
    if isinstance(v01, torch.Tensor):
        return SCALE_Z_MIN_M + SCALE_Z_SPAN_M * torch.clamp(v01, 0.0, 1.0)
    if isinstance(v01, np.ndarray):
        return (np.float32(SCALE_Z_MIN_M) + np.float32(SCALE_Z_SPAN_M) * np.clip(v01, 0.0, 1.0)).astype(np.float32)
    return SCALE_Z_MIN_M + SCALE_Z_SPAN_M * min(max(float(v01), 0.0), 1.0)


def scale_z_m_to_01(scale_m: torch.Tensor | np.ndarray | float) -> Any:
    if isinstance(scale_m, torch.Tensor):
        return torch.clamp((scale_m - SCALE_Z_MIN_M) / SCALE_Z_SPAN_M, 0.0, 1.0)
    if isinstance(scale_m, np.ndarray):
        return np.clip((scale_m - SCALE_Z_MIN_M) / SCALE_Z_SPAN_M, 0.0, 1.0).astype(np.float32)
    return min(max((float(scale_m) - SCALE_Z_MIN_M) / SCALE_Z_SPAN_M, 0.0), 1.0)


def opacity_01_to_alpha(v01: torch.Tensor | np.ndarray | float) -> Any:
    if isinstance(v01, torch.Tensor):
        return OPACITY_MIN + OPACITY_SPAN * torch.clamp(v01, 0.0, 1.0)
    if isinstance(v01, np.ndarray):
        return (np.float32(OPACITY_MIN) + np.float32(OPACITY_SPAN) * np.clip(v01, 0.0, 1.0)).astype(np.float32)
    return OPACITY_MIN + OPACITY_SPAN * min(max(float(v01), 0.0), 1.0)


def precision_01_to_tau(v01: torch.Tensor | np.ndarray | float) -> Any:
    """将 `[0, 1]` 神经精度通道映射为逆方差权重 `tau = sigma^-2` (`25 .. 10000 m^-2`)。"""
    if isinstance(v01, torch.Tensor):
        log_tau = LOG_TAU_MIN + LOG_TAU_SPAN * torch.clamp(v01, 0.0, 1.0)
        return torch.exp(log_tau)
    if isinstance(v01, np.ndarray):
        log_tau = np.float32(LOG_TAU_MIN) + np.float32(LOG_TAU_SPAN) * np.clip(v01, 0.0, 1.0).astype(np.float32)
        return np.exp(log_tau).astype(np.float32)
    log_tau = LOG_TAU_MIN + LOG_TAU_SPAN * min(max(float(v01), 0.0), 1.0)
    return math.exp(log_tau)


def decode_neural_splat_channels(y_nhwc: np.ndarray) -> dict[str, np.ndarray]:
    """将 `[..., 48, 48, 6]` 模型输出解码为完整物理度量属性字典。"""
    arr = np.asarray(y_nhwc, dtype=np.float32)
    inv_01 = np.clip(arr[..., 0], 0.0, 1.0)
    elev_01 = np.clip(arr[..., 1], 0.0, 1.0)
    s_xy_01 = np.clip(arr[..., 2], 0.0, 1.0)
    s_z_01 = np.clip(arr[..., 3], 0.0, 1.0)
    opac_01 = np.clip(arr[..., 4], 0.0, 1.0)
    prec_01 = np.clip(arr[..., 5], 0.0, 1.0)

    depth_m = inv_depth_01_to_metric_m(inv_01)
    scale_xy_m = scale_xy_01_to_m(s_xy_01)
    scale_z_m = scale_z_01_to_m(s_z_01)
    opacity = opacity_01_to_alpha(opac_01)
    tau = precision_01_to_tau(prec_01)
    sigma_m = (1.0 / np.sqrt(np.maximum(tau, 1e-6))).astype(np.float32)

    return {
        "raw_nhwc": arr,
        "inv_depth_01": inv_01,
        "elev_01": elev_01,
        "scale_xy_01": s_xy_01,
        "scale_z_01": s_z_01,
        "opacity_01": opac_01,
        "precision_01": prec_01,
        "depth_m": depth_m,
        "scale_xy_m": scale_xy_m,
        "scale_z_m": scale_z_m,
        "opacity": opacity,
        "precision_tau": tau,
        "sigma_m": sigma_m,
    }


# ---------------------------------------------------------------------------
# 2. HoverGaussianSplatNet 神经网络定义与纯内置算子 TFLite v3 导出器
# ---------------------------------------------------------------------------

class HoverGaussianSplatNet(nn.Module):
    """端到端 6 通道前馈神经 3D 高斯椭球预测器 (`[B, 3, 96, 96] -> [B, 6, 48, 48]`)。

    仅使用 `CONV_2D + PAD + ADD + RELU + LOGISTIC` 核心算子，100% 兼容 TFLite v3 FlatBuffers。
    多感受野拓扑：
      `3 (96x96) -> 32 (s=2, 48x48) -> 40 (d=1) -> 40 (d=2) -> 40 (d=4) -> 40 (d=8)`
      `-> 多尺度融合 (b1 + b4 + b8) -> 40 (d=1 局部精化) -> 32 (1x1) -> 6 (1x1) -> Sigmoid`
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(INPUT_C, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)

        self.res_d1 = DilatedResBlock(32, 40, dilation=1)
        self.res_d2 = DilatedResBlock(40, 40, dilation=2)
        self.res_d4 = DilatedResBlock(40, 40, dilation=4)
        self.res_d8 = DilatedResBlock(40, 40, dilation=8)
        self.res_refine = DilatedResBlock(40, 40, dilation=1)

        self.head_conv1 = nn.Conv2d(40, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.head_bn1 = nn.BatchNorm2d(32)
        self.head_out = nn.Conv2d(32, SPLAT_OUT_C, kernel_size=1, stride=1, padding=0, bias=True)

        # 合理物理先验偏置初始化：居中逆深度、低初始突起、适中水平半径、偏薄垂直半径、高不透明、中等精度
        with torch.no_grad():
            self.head_out.bias.copy_(
                torch.tensor([-0.20, -1.50, 0.10, -0.85, 1.40, 0.00], dtype=torch.float32)
            )

    def forward_logits(self, x_nchw: torch.Tensor) -> torch.Tensor:
        s0 = F.relu(self.stem_bn(self.stem_conv(x_nchw)))
        b1 = self.res_d1(s0)
        b2 = self.res_d2(b1)
        b4 = self.res_d4(b2)
        b8 = self.res_d8(b4)
        ms_mid = b8 + b4
        ms_all = ms_mid + b1
        ref = self.res_refine(ms_all)
        h = F.relu(self.head_bn1(self.head_conv1(ref)))
        return self.head_out(h)

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x_nchw))

    @torch.no_grad()
    def infer_nhwc(self, rgb_nhwc: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        was_training = self.training
        self.eval()
        try:
            arr = rgb_nhwc.detach().cpu().numpy() if isinstance(rgb_nhwc, torch.Tensor) else np.asarray(rgb_nhwc)
            single = arr.ndim == 3
            if single:
                arr = arr[None, ...]
            if arr.shape[1:] != (INPUT_H, INPUT_W, INPUT_C):
                raise ValueError(f"Expected [..., 96, 96, 3], got {arr.shape}")
            dev = next(self.parameters()).device
            x_t = torch.from_numpy(np.transpose(arr.astype(np.float32), (0, 3, 1, 2))).to(dev)
            y_nchw = self.forward(x_t).cpu().numpy()
            y_nhwc = np.transpose(y_nchw, (0, 2, 3, 1)).astype(np.float32)
            return decode_neural_splat_channels(y_nhwc[0] if single else y_nhwc)
        finally:
            if was_training:
                self.train()


def build_neural_splat_tflite(model: HoverGaussianSplatNet) -> bytes:
    """折叠 BatchNorm 并导出标准纯内置算子 TFLite v3 FlatBuffers (`[1,96,96,3] -> [1,48,48,6]`)。"""
    model.eval()
    gb = _TFLiteGraphBuilder("RIG-Hover HoverGaussianSplatNet v2 (96x96x3 -> 48x48x6)", initial_capacity=1048576)
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
            skip_tid = _emit_conv_bn(f"{name}_proj", x_tid, blk.proj[0], blk.proj[1], 1, 1, 0, ACT_NONE)
        else:
            skip_tid = x_tid
        h1 = _emit_conv_bn(f"{name}_c1", x_tid, blk.conv1, blk.bn1, 1, d, d, ACT_RELU)
        h2 = _emit_conv_bn(f"{name}_c2", h1, blk.conv2, blk.bn2, 1, d, d, ACT_NONE)
        return gb.add(f"{name}_out", h2, skip_tid, fused_act=ACT_RELU)

    s0 = _emit_conv_bn("stem", rgb_in, model.stem_conv, model.stem_bn, stride=2, dilation=1, pad_hw=1, fused_act=ACT_RELU)
    b1 = _emit_res_block("res_d1", s0, model.res_d1)
    b2 = _emit_res_block("res_d2", b1, model.res_d2)
    b4 = _emit_res_block("res_d4", b2, model.res_d4)
    b8 = _emit_res_block("res_d8", b4, model.res_d8)
    ms_mid = gb.add("ms_mid", b8, b4, fused_act=ACT_NONE)
    ms_all = gb.add("ms_all", ms_mid, b1, fused_act=ACT_NONE)
    ref = _emit_res_block("res_refine", ms_all, model.res_refine)

    hc1 = _emit_conv_bn("head_c1", ref, model.head_conv1, model.head_bn1, stride=1, dilation=1, pad_hw=0, fused_act=ACT_RELU)
    w_out = np.transpose(model.head_out.weight.detach().cpu().numpy().astype(np.float32), (0, 2, 3, 1))
    b_out = model.head_out.bias.detach().cpu().numpy().astype(np.float32)
    logits = gb.conv2d("head_logits", hc1, w_out, b_out, stride=1, dilation=1, padding=PAD_VALID, fused_act=ACT_NONE)
    splat_out = gb.sigmoid("gaussian_splat_out", logits)

    gb.set_outputs([splat_out])
    return gb.finish()


def load_neural_splat_from_tflite(tflite_path: str | Path) -> HoverGaussianSplatNet:
    """从导出的 `hover_gaussian_splat_v2.tflite` 直接无损恢复等价 PyTorch 模型。"""
    consts = parse_named_tflite_constants(tflite_path)
    model = HoverGaussianSplatNet()

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
        ("res_refine", model.res_refine),
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


def infer_neural_splat_tflite(tflite_path: str | Path, rgb_nhwc: np.ndarray) -> dict[str, np.ndarray]:
    """使用纯 FlatBuffers CPU 算子执行器推理 `hover_gaussian_splat_v2.tflite` 并解码 6 通道高斯参数。"""
    arr = np.asarray(rgb_nhwc, dtype=np.float32)
    single = arr.ndim == 3
    if single:
        arr = arr[None, ...]
    if arr.shape[1:] != (INPUT_H, INPUT_W, INPUT_C):
        raise ValueError(f"Expected [B, 96, 96, 3], got {arr.shape}")
    outs = [run_tflite_flatbuffer_cpu(tflite_path, [arr[i : i + 1]])[0][0] for i in range(arr.shape[0])]
    y_nhwc = np.stack(outs, axis=0).astype(np.float32)
    return decode_neural_splat_channels(y_nhwc[0] if single else y_nhwc)


def verify_neural_splat_tflite_parity(
    model: HoverGaussianSplatNet,
    tflite_path: str | Path,
    n_samples: int = 8,
    seed: int = 20261118,
) -> float:
    """校验 PyTorch `.eval()` 与 `.tflite` FlatBuffers 在 `[1, 96, 96, 3]` 随机输入上的最大误差。"""
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


def benchmark_neural_splat_cpu_ms(model: HoverGaussianSplatNet, n_runs: int = 60) -> float:
    """测算 NHWC (`channels_last`) 算子融合下的单帧 CPU 推理时延（ms）。"""
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(4, max(1, prev_threads)))
        m_cpu = HoverGaussianSplatNet()
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


def export_neural_splat_bundle(
    model: HoverGaussianSplatNet,
    out_dir: str | Path,
    android_assets_dir: str | Path | None = None,
    version: str = "2026.11.18",
) -> dict[str, Any]:
    """导出 `hover_gaussian_splat_v2.tflite`, `.json`, `hover-gaussian-splat-v2-bundle.zip` 并同步 Android assets。"""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tflite_bytes = build_neural_splat_tflite(model)
    sha256_hex = hashlib.sha256(tflite_bytes).hexdigest()

    tflite_file = out_path / "hover_gaussian_splat_v2.tflite"
    tflite_file.write_bytes(tflite_bytes)

    manifest = {
        "schemaVersion": 1,
        "name": "hover-gaussian-splat-v2",
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
            "channels": SPLAT_OUT_C,
            "channelNames": list(CHANNEL_NAMES),
            "invDepthMin": INV_DEPTH_MIN,
            "invDepthSpan": INV_DEPTH_SPAN,
            "depthMinM": round(1.0 / (INV_DEPTH_MIN + INV_DEPTH_SPAN), 4),
            "depthMaxM": round(1.0 / INV_DEPTH_MIN, 4),
            "elevThresholdM": ELEV_THRESHOLD_M,
            "scaleXyMinM": SCALE_XY_MIN_M,
            "scaleXyMaxM": SCALE_XY_MAX_M,
            "scaleZMinM": SCALE_Z_MIN_M,
            "scaleZMaxM": SCALE_Z_MAX_M,
            "opacityMin": OPACITY_MIN,
            "opacityMax": OPACITY_MAX,
            "precisionMin": TAU_MIN,
            "precisionMax": TAU_MAX,
        },
        "camera": {
            "heightM": DEFAULT_H_CAM_M,
            "fovHDeg": DEFAULT_FOV_H_DEG,
        },
        "sha256": sha256_hex,
        "notes": "RIG-Hover Feed-Forward 6-Channel Neural 3D Gaussian Splat Predictor (96x96x3 -> 48x48x6, TFLite v3 pure builtins)",
    }
    json_file = out_path / "hover_gaussian_splat_v2.json"
    json_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_file = out_path / "hover-gaussian-splat-v2-bundle.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tflite_file, arcname="model.tflite")
        zf.write(json_file, arcname="model.json")

    max_abs_diff = verify_neural_splat_tflite_parity(model, tflite_file, n_samples=8)

    copied_android: list[str] = []
    if android_assets_dir is not None:
        assets_path = Path(android_assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)
        dst_tfl = assets_path / "hover_gaussian_splat_v2.tflite"
        dst_jsn = assets_path / "hover_gaussian_splat_v2.json"
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


# ---------------------------------------------------------------------------
# 3. 世界系相机位姿投影与可微新视角 3DGS 重投影渲染器 (Splatter-Image 光度监督)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RobotCameraPose:
    """机器人在全局桌面坐标系 (`+X` 前, `+Y` 右, `+Z` 上, 桌面 `Z=0`) 下的 6-DoF 相机位姿。"""

    x_m: float = 0.0
    y_m: float = 0.0
    z_cam_m: float = DEFAULT_H_CAM_M
    yaw_deg: float = 0.0       # 机身航向 + 头部云台偏航角总和
    pitch_deg: float = 0.0     # >0 表示镜头向下微俯（与 mono_depth_model 约定一致）
    roll_deg: float = 0.0
    fov_h_deg: float = DEFAULT_FOV_H_DEG

    def rotation_cam_to_world(self) -> np.ndarray:
        """返回 `3x3` 旋转矩阵 `R_c2w`，将相机系射线 `[1, u*tan, v*tan]` 变换至世界系。"""
        r_level = camera_rotation_matrix(pitch_deg=self.pitch_deg, roll_deg=self.roll_deg)
        psi = math.radians(float(self.yaw_deg))
        cg, sg = math.cos(psi), math.sin(psi)
        r_yaw = np.array(
            [
                [cg, -sg, 0.0],
                [sg, cg, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return (r_yaw @ r_level).astype(np.float32)


def make_world_ray_grid(
    pose: RobotCameraPose,
    height: int = OUTPUT_H,
    width: int = OUTPUT_W,
) -> tuple[np.ndarray, np.ndarray]:
    """生成指定位姿下相机原点 `cam_origin [3]` 与世界系射线向量 `rays_world [H, W, 3]`。

    其中 `P_world(d) = cam_origin + d * rays_world` 的参数 `d` 严格等于前向平面深度 `depth_m`。
    """
    rays_level = make_camera_ray_grid(
        height=height,
        width=width,
        fov_h_deg=pose.fov_h_deg,
        pitch_deg=pose.pitch_deg,
        roll_deg=pose.roll_deg,
    )
    psi = math.radians(float(pose.yaw_deg))
    cg, sg = math.cos(psi), math.sin(psi)
    r_yaw = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rays_world = np.einsum("ij,hwj->hwi", r_yaw, rays_level).astype(np.float32)
    origin = np.array([pose.x_m, pose.y_m, pose.z_cam_m], dtype=np.float32)
    return origin, rays_world


def differentiable_novel_view_render_24x24(
    pred_nchw_1: torch.Tensor,
    rgb48_nchw_1: torch.Tensor,
    origin_1: torch.Tensor,
    rays48_world_1: torch.Tensor,
    origin_2: torch.Tensor,
    rot_w2c_2: torch.Tensor,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
) -> dict[str, torch.Tensor]:
    """纯 PyTorch 可微各向异性 3D 高斯新视角重投影渲染器 (`24x24` 目标网格)。

    将视角 1 神经网络预测的 `48x48` 个 3D 高斯经 `2x2` 精度加权池化为 `576` 个三维各向异性椭球，
    投影至视角 2 相机平面并按深度由近及远精确执行体积 Alpha 合成 (`T_k = prod_{j<k} (1 - w_j)`)，
    让光度重建梯度直接反向传播至 `inv_depth_01`, `scale_xy_01`, `scale_z_01`, `opacity_01`, `precision_01`。
    """
    inv_01 = pred_nchw_1[:, 0:1]
    s_xy_01 = pred_nchw_1[:, 2:3]
    s_z_01 = pred_nchw_1[:, 3:4]
    op_01 = pred_nchw_1[:, 4:5]
    pr_01 = pred_nchw_1[:, 5:6]

    depth_48 = inv_depth_01_to_metric_m(inv_01)  # [B, 1, 48, 48]
    s_xy_48 = scale_xy_01_to_m(s_xy_01)
    s_z_48 = scale_z_01_to_m(s_z_01)
    alpha_48 = opacity_01_to_alpha(op_01)
    tau_48 = precision_01_to_tau(pr_01)

    # 世界系 3D 坐标 [B, 3, 48, 48]
    rays_bchw = rays48_world_1.permute(0, 3, 1, 2)
    xyz_48 = origin_1.view(-1, 3, 1, 1) + depth_48 * rays_bchw

    # 2x2 精度加权下采样至 K = 24x24 = 576 个代表性神经高斯椭球，兼顾全视场覆盖与 GPU 吞吐
    w_pool = tau_48 / (F.avg_pool2d(tau_48, kernel_size=2, stride=2).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3) * 4.0 + 1e-6)
    xyz_24 = F.avg_pool2d(xyz_48 * w_pool, kernel_size=2, stride=2) * 4.0  # [B, 3, 24, 24]
    rgb_24 = F.avg_pool2d(rgb48_nchw_1 * w_pool, kernel_size=2, stride=2) * 4.0
    s_xy_24 = F.avg_pool2d(s_xy_48, kernel_size=2, stride=2) * 1.45
    s_z_24 = F.avg_pool2d(s_z_48, kernel_size=2, stride=2) * 1.45
    alpha_24 = torch.clamp(F.avg_pool2d(alpha_48, kernel_size=2, stride=2) * 1.15, 0.02, 0.99)

    bsz = pred_nchw_1.shape[0]
    pts_w = xyz_24.flatten(2).transpose(1, 2)      # [B, K=576, 3]
    cols = rgb_24.flatten(2).transpose(1, 2)       # [B, K, 3]
    s_xy = s_xy_24.flatten(1)                      # [B, K]
    s_z = s_z_24.flatten(1)                        # [B, K]
    alpha = alpha_24.flatten(1)                    # [B, K]

    # 变换到视角 2 相机坐标系 (+X 前向深度, +Y 右, +Z 上)
    rel = pts_w - origin_2.unsqueeze(1)
    p_cam2 = torch.bmm(rel, rot_w2c_2.transpose(1, 2))  # [B, K, 3]
    z_fwd = p_cam2[..., 0]
    y_right = p_cam2[..., 1]
    z_up = p_cam2[..., 2]

    valid = (z_fwd > 0.14) & (z_fwd < 2.95)
    z_safe = torch.clamp(z_fwd, min=0.14)

    tan_half = math.tan(math.radians(float(fov_h_deg) * 0.5))
    out_res = 24
    focal_px = (out_res * 0.5) / tan_half

    u_px = (out_res * 0.5) + (y_right / z_safe) * focal_px - 0.5
    v_px = (out_res * 0.5) - (z_up / z_safe) * focal_px - 0.5

    # 透视投影各向异性屏幕椭圆标准差 (sigma_u 取决于水平尺度 s_xy, sigma_v 耦合 s_z 与俯视投影角)
    cos_view = torch.clamp(z_safe / torch.sqrt(z_safe * z_safe + z_up * z_up + 1e-6), 0.25, 1.0)
    sin_view = torch.sqrt(torch.clamp(1.0 - cos_view * cos_view, min=0.0))
    eff_sv_m = torch.sqrt(s_z * s_z * (cos_view * cos_view) + s_xy * s_xy * (sin_view * sin_view) + 1e-8)

    sig_u = torch.clamp((s_xy / z_safe) * focal_px, min=0.55, max=6.5)
    sig_v = torch.clamp((eff_sv_m / z_safe) * focal_px, min=0.55, max=6.5)

    # 按视角 2 前向距离由近到远排序，精确计算体积透射率 T_k
    sort_key = torch.where(valid, z_fwd, torch.full_like(z_fwd, 99.0))
    order = torch.argsort(sort_key, dim=1)  # [B, K]

    u_s = torch.gather(u_px, 1, order)
    v_s = torch.gather(v_px, 1, order)
    su_s = torch.gather(sig_u, 1, order)
    sv_s = torch.gather(sig_v, 1, order)
    al_s = torch.gather(alpha * valid.float(), 1, order)
    col_s = torch.gather(cols, 1, order.unsqueeze(-1).expand(-1, -1, 3))

    gy, gx = torch.meshgrid(
        torch.arange(out_res, device=pred_nchw_1.device, dtype=torch.float32),
        torch.arange(out_res, device=pred_nchw_1.device, dtype=torch.float32),
        indexing="ij",
    )
    px_u = gx.reshape(1, -1, 1)  # [1, M=576, 1]
    px_v = gy.reshape(1, -1, 1)

    du2 = ((px_u - u_s.unsqueeze(1)) / su_s.unsqueeze(1)) ** 2
    dv2 = ((px_v - v_s.unsqueeze(1)) / sv_s.unsqueeze(1)) ** 2
    gauss = torch.exp(-0.5 * torch.clamp(du2 + dv2, max=16.0))
    w_mk = torch.clamp(al_s.unsqueeze(1) * gauss, 0.0, 0.99)  # [B, M, K]

    one_minus = 1.0 - w_mk + 1e-6
    trans_inclusive = torch.cumprod(one_minus, dim=-1)
    trans_exclusive = torch.cat([torch.ones_like(trans_inclusive[..., :1]), trans_inclusive[..., :-1]], dim=-1)

    contrib = trans_exclusive * w_mk  # [B, M, K]
    rendered_flat = torch.bmm(contrib, col_s)  # [B, M, 3]
    cov_mass = torch.clamp(contrib.sum(dim=-1), 0.0, 1.0)  # [B, M]

    rendered_nchw = rendered_flat.transpose(1, 2).reshape(bsz, 3, out_res, out_res)
    cov_nchw = cov_mass.reshape(bsz, 1, out_res, out_res)
    return {
        "rendered_rgb_24": rendered_nchw,
        "coverage_24": cov_nchw,
    }


# ---------------------------------------------------------------------------
# 4. 持久贝叶斯空间记忆引擎 (`PersistentGaussianMemoryMap`) —— 无界拓展 & 100% 保留旧区
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitCamera3D:
    """与展示端严格一致的 3D 轨道观察相机参数。"""

    azimuth_deg: float = -145.0
    elevation_deg: float = 36.0
    distance_m: float = 1.45
    target_x: float = 0.45
    target_y: float = 0.15
    target_z: float = 0.05
    fov_vertical_deg: float = 52.0


class PersistentGaussianMemoryMap:
    """基于神经网络异方差精度 `tau` 的多视角贝叶斯卡尔曼 3D 高斯持久空间记忆地图。

    核心特性：
      1. 100% 模型生成属性：位置、各向异性水平/垂直半径 `(scale_xy, scale_z)`、不透明度 `opacity`、
         初始观测精度 `tau` 均直接取自 `HoverGaussianSplatNet` 6 通道输出；
      2. 多视角贝叶斯卡尔曼更新：同一 `2 cm` 体素从新视角观测时，按卡尔曼增益 `K = tau_eff / (tau_post + tau_eff)`
         融合三维坐标与各向异性椭球尺寸，后验精度累加 `tau_post += tau_eff`，不确定度 `sigma` 单调收敛；
      3. 持久空间拓展与历史区 100% 保留：采用大容量动态结构体数组（默认上限 `65536` 体素，无环形覆盖淘汰），
         离开当前相机视锥的旧区域点云（如探索至区域 B 时的区域 A）永久锁存在全局地图中绝不丢失。
    """

    VOXEL_BIAS: int = 524_288
    VOXEL_MASK_20: int = 0xFFFFF

    def __init__(self, voxel_size: float = 0.020, max_splats: int = 65536) -> None:
        self.voxel_size = float(voxel_size)
        self.max_splats = int(max_splats)

        self.count: int = 0
        self.pos_x = np.zeros(self.max_splats, dtype=np.float32)
        self.pos_y = np.zeros(self.max_splats, dtype=np.float32)
        self.pos_z = np.zeros(self.max_splats, dtype=np.float32)
        self.scale_xy = np.zeros(self.max_splats, dtype=np.float32)
        self.scale_z = np.zeros(self.max_splats, dtype=np.float32)
        self.opacity = np.zeros(self.max_splats, dtype=np.float32)
        self.elev_prob = np.zeros(self.max_splats, dtype=np.float32)
        self.precision_tau = np.zeros(self.max_splats, dtype=np.float32)
        self.rgb_u8 = np.zeros((self.max_splats, 3), dtype=np.uint8)
        self.obs_count = np.zeros(self.max_splats, dtype=np.int32)
        self.view_sector_mask = np.zeros(self.max_splats, dtype=np.uint8)
        self.birth_step = np.zeros(self.max_splats, dtype=np.int32)

        self._voxel_to_idx: dict[int, int] = {}
        self.step_count: int = 0
        self.kalman_fusion_updates: int = 0
        self.multi_sector_upgrades: int = 0

    def clear(self) -> None:
        self.count = 0
        self.step_count = 0
        self.kalman_fusion_updates = 0
        self.multi_sector_upgrades = 0
        self._voxel_to_idx.clear()

    def _pack_voxel_coords(self, ix: int, iy: int, iz: int) -> int:
        bx = (ix + self.VOXEL_BIAS) & self.VOXEL_MASK_20
        by = (iy + self.VOXEL_BIAS) & self.VOXEL_MASK_20
        bz = (iz + self.VOXEL_BIAS) & self.VOXEL_MASK_20
        return (bx << 40) | (by << 20) | bz

    def _pack_voxel_key(self, wx: float, wy: float, wz: float) -> int:
        inv = 1.0 / self.voxel_size
        return self._pack_voxel_coords(
            int(math.floor(wx * inv)),
            int(math.floor(wy * inv)),
            int(math.floor(wz * inv)),
        )

    def _find_associated_splat(self, wx: float, wy: float, wz: float, el: float) -> tuple[int, int]:
        """在 3x3x3 邻近体素内检索同类表面（桌面 vs 凸起物）最近邻高斯点，避免多视角微偏差分裂体素。"""
        inv = 1.0 / self.voxel_size
        cx = int(math.floor(wx * inv))
        cy = int(math.floor(wy * inv))
        cz = int(math.floor(wz * inv))
        exact_key = self._pack_voxel_coords(cx, cy, cz)
        exact_idx = self._voxel_to_idx.get(exact_key, -1)
        if exact_idx >= 0 and self.opacity[exact_idx] > 0.12:
            if (float(self.elev_prob[exact_idx]) >= 0.38) == (el >= 0.38):
                return exact_idx, exact_key

        # 对凸起立体目标启用 3x3x3 邻域关联（半径 <= 2.1 * voxel_size），确保多角度射线融合至同一椭球
        if el < 0.38:
            return (-1, exact_key) if exact_idx < 0 else (exact_idx, exact_key)

        max_r2 = (self.voxel_size * 2.15) ** 2
        best_idx = -1
        best_key = exact_key
        best_d2 = max_r2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k_nb = self._pack_voxel_coords(cx + dx, cy + dy, cz + dz)
                    cand = self._voxel_to_idx.get(k_nb, -1)
                    if cand < 0 or self.opacity[cand] <= 0.12:
                        continue
                    if float(self.elev_prob[cand]) < 0.38:
                        continue
                    d2 = (
                        (float(self.pos_x[cand]) - wx) ** 2
                        + (float(self.pos_y[cand]) - wy) ** 2
                        + (float(self.pos_z[cand]) - wz) ** 2
                    )
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = cand
                        best_key = k_nb
        return best_idx, best_key

    @staticmethod
    def _sector_bit(cam_x: float, cam_y: float, wx: float, wy: float) -> int:
        ang = math.atan2(cam_y - wy, cam_x - wx)  # [-pi, pi]
        sec = int(math.floor((ang + math.pi) / (2.0 * math.pi) * 8.0)) % 8
        return 1 << sec

    @staticmethod
    def _popcount8(mask: int) -> int:
        return bin(int(mask) & 0xFF).count("1")

    def ingest_neural_splats(
        self,
        rgb_in_96_or_48: np.ndarray,
        decoded_pred: dict[str, np.ndarray],
        pose: RobotCameraPose,
        align_ground_scale: bool = False,
        min_opacity: float = 0.16,
        max_ray_depth_m: float = 2.15,
        grid_stride: int = 1,
    ) -> dict[str, Any]:
        """将单帧 `HoverGaussianSplatNet` 6 通道预测融合进全局持久 3DGS 记忆场。"""
        rgb_arr = np.asarray(rgb_in_96_or_48, dtype=np.float32)
        if rgb_arr.shape == (INPUT_H, INPUT_W, INPUT_C):
            rgb48_f = rgb_arr.reshape(OUTPUT_H, 2, OUTPUT_W, 2, INPUT_C).mean(axis=(1, 3))
        elif rgb_arr.shape == (OUTPUT_H, OUTPUT_W, INPUT_C):
            rgb48_f = rgb_arr
        else:
            raise ValueError(f"Expected rgb shape (96,96,3) or (48,48,3), got {rgb_arr.shape}")
        rgb48_u8 = np.clip(np.rint(np.clip(rgb48_f, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)

        depth_m = np.asarray(decoded_pred["depth_m"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
        elev_p = np.asarray(decoded_pred["elev_01"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
        s_xy_m = np.asarray(decoded_pred["scale_xy_m"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
        s_z_m = np.asarray(decoded_pred["scale_z_m"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
        opac = np.asarray(decoded_pred["opacity"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)
        tau_map = np.asarray(decoded_pred["precision_tau"], dtype=np.float32).reshape(OUTPUT_H, OUTPUT_W)

        origin, rays_world = make_world_ray_grid(pose, height=OUTPUT_H, width=OUTPUT_W)

        # 可选真机地平面中值尺度锚定（等比缩放深度与神经椭球尺寸，保留全部神经网络各向异性与精度结构）
        scale_factor = 1.0
        if align_ground_scale:
            rz = rays_world[..., 2]
            plane_d = np.where(rz < -1e-3, -pose.z_cam_m / np.where(rz < -1e-3, rz, -1.0), 9.9)
            rows_idx = np.arange(OUTPUT_H)[:, None]
            cols_idx = np.arange(OUTPUT_W)[None, :]
            inliers = (
                (rows_idx >= 24)
                & (cols_idx >= 4)
                & (cols_idx < 44)
                & (elev_p < 0.30)
                & (rz < -0.04)
                & (plane_d < 1.95)
            )
            if int(np.sum(inliers)) >= 8:
                scale_factor = float(np.clip(np.median((plane_d / np.maximum(depth_m, 1e-3))[inliers]), 0.65, 1.65))
                depth_m = depth_m * scale_factor
                s_xy_m = np.clip(s_xy_m * scale_factor, SCALE_XY_MIN_M, SCALE_XY_MAX_M)
                s_z_m = np.clip(s_z_m * scale_factor, SCALE_Z_MIN_M, SCALE_Z_MAX_M)

        step_id = self.step_count
        stride = max(1, int(grid_stride))
        inserted_new = 0
        fused_existing = 0

        # 自由空间射线一致性裁剪：仅针对当前相机视锥内部 (严禁触碰视锥外如区域 A 的任何历史点！)
        # 若早期单视角低置信度凸起噪点被当前正交视角清晰穿透 (d_prev < d_obs - 0.055m)，衰减其伪影不透明度
        if self.count > 0:
            n_pre = self.count
            cand_m = (self.obs_count[:n_pre] <= 1) & (self.elev_prob[:n_pre] >= 0.38) & (self.opacity[:n_pre] > 0.15)
            if np.any(cand_m):
                c_indices = np.where(cand_m)[0]
                rel_x = self.pos_x[c_indices] - float(origin[0])
                rel_y = self.pos_y[c_indices] - float(origin[1])
                rel_z = self.pos_z[c_indices] - float(origin[2])
                rot_w2c = pose.rotation_cam_to_world().T
                p_cam = np.stack([rel_x, rel_y, rel_z], axis=-1) @ rot_w2c.T
                zc = p_cam[:, 0]
                tan_h = math.tan(math.radians(pose.fov_h_deg * 0.5))
                in_z = (zc > 0.24) & (zc < float(max_ray_depth_m))
                if np.any(in_z):
                    u_n = 0.5 + 0.5 * (p_cam[:, 1] / np.maximum(zc, 0.1) / tan_h)
                    v_n = 0.5 - 0.5 * (p_cam[:, 2] / np.maximum(zc, 0.1) / tan_h)
                    in_fov = in_z & (u_n >= 0.06) & (u_n <= 0.94) & (v_n >= 0.06) & (v_n <= 0.94)
                    for local_k in np.where(in_fov)[0]:
                        g_idx = int(c_indices[local_k])
                        cc = min(OUTPUT_W - 1, max(0, int(u_n[local_k] * OUTPUT_W)))
                        rr = min(OUTPUT_H - 1, max(0, int(v_n[local_k] * OUTPUT_H)))
                        d_obs_rc = float(depth_m[rr, cc])
                        tau_ray = float(tau_map[rr, cc]) / max(0.30, d_obs_rc * d_obs_rc)
                        if (
                            float(opac[rr, cc]) > 0.55
                            and float(zc[local_k]) < d_obs_rc - 0.075
                            and tau_ray > float(self.precision_tau[g_idx]) * 1.25
                        ):
                            self.opacity[g_idx] *= 0.22
                            if self.opacity[g_idx] < 0.14:
                                old_k = self._pack_voxel_key(
                                    float(self.pos_x[g_idx]), float(self.pos_y[g_idx]), float(self.pos_z[g_idx])
                                )
                                if self._voxel_to_idx.get(old_k) == g_idx:
                                    self._voxel_to_idx.pop(old_k, None)

        for r in range(OUTPUT_H):
            for c in range(OUTPUT_W):
                d = float(depth_m[r, c])
                al = float(opac[r, c])
                el = float(elev_p[r, c])
                if d < 0.23 or d > float(max_ray_depth_m) or al < float(min_opacity):
                    continue
                if stride > 1 and ((r % stride != 0) or (c % stride != 0)) and el < 0.42:
                    continue

                ray = rays_world[r, c]
                wx = float(origin[0] + d * ray[0])
                wy = float(origin[1] + d * ray[1])
                wz = float(origin[2] + d * ray[2])

                # 过滤高仰角远景背景墙天花板射线，聚焦桌面与桌面立体交互空间
                if wz > 0.48 or wz < -0.06:
                    continue

                s_xy = float(s_xy_m[r, c])
                s_z = float(s_z_m[r, c])
                # 远距离射线的投影不确定度按距离平方自然衰减观测权重
                tau_obs = float(tau_map[r, c]) / max(0.30, d * d)
                sec_bit = self._sector_bit(float(origin[0]), float(origin[1]), wx, wy)
                px_rgb = rgb48_u8[r, c]

                idx, matched_key = self._find_associated_splat(wx, wy, wz, el)
                if idx >= 0:
                    old_mask = int(self.view_sector_mask[idx])
                    is_new_sector = (old_mask & sec_bit) == 0
                    tau_eff = tau_obs * (1.45 if is_new_sector else 1.0)
                    tau_prev = float(self.precision_tau[idx])
                    k_gain = tau_eff / max(tau_prev + tau_eff, 1e-6)

                    new_wx = (1.0 - k_gain) * float(self.pos_x[idx]) + k_gain * wx
                    new_wy = (1.0 - k_gain) * float(self.pos_y[idx]) + k_gain * wy
                    new_wz = (1.0 - k_gain) * float(self.pos_z[idx]) + k_gain * wz

                    # 若卡尔曼修正后跨过了体素边界且目标体素为空，同步迁移哈希键
                    new_key = self._pack_voxel_key(new_wx, new_wy, new_wz)
                    if new_key != matched_key and new_key not in self._voxel_to_idx:
                        if self._voxel_to_idx.get(matched_key) == idx:
                            self._voxel_to_idx.pop(matched_key, None)
                        self._voxel_to_idx[new_key] = idx

                    self.pos_x[idx] = new_wx
                    self.pos_y[idx] = new_wy
                    self.pos_z[idx] = new_wz
                    self.scale_xy[idx] = (1.0 - k_gain) * float(self.scale_xy[idx]) + k_gain * s_xy
                    self.scale_z[idx] = (1.0 - k_gain) * float(self.scale_z[idx]) + k_gain * s_z
                    self.elev_prob[idx] = (1.0 - k_gain) * float(self.elev_prob[idx]) + k_gain * el
                    self.opacity[idx] = min(
                        0.99,
                        (1.0 - k_gain) * float(self.opacity[idx]) + k_gain * al + (0.025 if is_new_sector else 0.0),
                    )
                    self.precision_tau[idx] = min(50000.0, tau_prev + tau_eff)

                    k_col = min(max(k_gain, 0.08), 0.50)
                    old_c = self.rgb_u8[idx]
                    self.rgb_u8[idx] = (
                        int(round((1.0 - k_col) * float(old_c[0]) + k_col * float(px_rgb[0]))),
                        int(round((1.0 - k_col) * float(old_c[1]) + k_col * float(px_rgb[1]))),
                        int(round((1.0 - k_col) * float(old_c[2]) + k_col * float(px_rgb[2]))),
                    )
                    self.obs_count[idx] += 1
                    self.view_sector_mask[idx] = np.uint8(old_mask | sec_bit)
                    if is_new_sector:
                        self.multi_sector_upgrades += 1
                    self.kalman_fusion_updates += 1
                    fused_existing += 1
                elif self.count < self.max_splats:
                    new_idx = self.count
                    self.count += 1
                    self.pos_x[new_idx] = wx
                    self.pos_y[new_idx] = wy
                    self.pos_z[new_idx] = wz
                    self.scale_xy[new_idx] = s_xy
                    self.scale_z[new_idx] = s_z
                    self.opacity[new_idx] = al
                    self.elev_prob[new_idx] = el
                    self.precision_tau[new_idx] = tau_obs
                    self.rgb_u8[new_idx] = px_rgb
                    self.obs_count[new_idx] = 1
                    self.view_sector_mask[new_idx] = np.uint8(sec_bit)
                    self.birth_step[new_idx] = step_id
                    self._voxel_to_idx[matched_key] = new_idx
                    inserted_new += 1

        self.step_count += 1
        return {
            "step": self.step_count,
            "total_splats": self.count,
            "inserted_new": inserted_new,
            "fused_existing": fused_existing,
            "scale_factor": scale_factor,
            "explored_area_m2": self.compute_explored_area_m2(),
        }

    def compute_explored_area_m2(self, cell_size_m: float = 0.05) -> float:
        """统计全局记忆点云在水平桌面 `(X, Y)` 上已覆盖的探索面积（m²，按 5cm 栅格计数）。"""
        if self.count == 0:
            return 0.0
        n = self.count
        valid = self.opacity[:n] >= 0.20
        if not np.any(valid):
            return 0.0
        gx = np.floor(self.pos_x[:n][valid] / cell_size_m).astype(np.int64)
        gy = np.floor(self.pos_y[:n][valid] / cell_size_m).astype(np.int64)
        packed = (gx << 32) ^ (gy & 0xFFFFFFFF)
        unique_cells = int(np.unique(packed).size)
        return float(unique_cells * (cell_size_m * cell_size_m))

    def snapshot_indices_in_region(
        self,
        cx: float,
        cy: float,
        radius_m: float,
    ) -> np.ndarray:
        """获取指定 `(cx, cy)` 半径内所有已建图高斯点的索引数组（用于检验探索远方后旧区留存率）。"""
        if self.count == 0:
            return np.zeros(0, dtype=np.int32)
        n = self.count
        d2 = (self.pos_x[:n] - float(cx)) ** 2 + (self.pos_y[:n] - float(cy)) ** 2
        return np.where(d2 <= float(radius_m) ** 2)[0].astype(np.int32)

    def region_retention_rate(
        self,
        initial_indices: np.ndarray,
        cx: float,
        cy: float,
        radius_m: float,
    ) -> float:
        """核验初始在某区域生成的高斯点索引在后续全局探索后仍完好保留在原区域的比例 (`0..1`)。"""
        if initial_indices.size == 0:
            return 1.0
        valid_idx = initial_indices[initial_indices < self.count]
        if valid_idx.size == 0:
            return 0.0
        # 允许卡尔曼微调带来的厘米级边界移动 (+3 cm 容差)
        tol_r2 = (float(radius_m) + 0.03) ** 2
        d2 = (self.pos_x[valid_idx] - float(cx)) ** 2 + (self.pos_y[valid_idx] - float(cy)) ** 2
        alive = (d2 <= tol_r2) & (self.opacity[valid_idx] > 0.15)
        return float(np.sum(alive)) / float(initial_indices.size)

    def mean_posterior_sigma_m(self, min_views: int = 1) -> float:
        """计算地图内满足最少观测次数的高斯点平均后验位置标准差 `sigma = tau^(-1/2)`（米）。"""
        if self.count == 0:
            return 0.20
        n = self.count
        m = self.obs_count[:n] >= int(min_views)
        if not np.any(m):
            return 0.20
        sigmas = 1.0 / np.sqrt(np.maximum(self.precision_tau[:n][m], 1e-6))
        return float(np.mean(sigmas))

    def active_splats_dict(self) -> dict[str, np.ndarray]:
        n = self.count
        tau = self.precision_tau[:n].copy()
        sigma = (1.0 / np.sqrt(np.maximum(tau, 1e-6))).astype(np.float32)
        sectors = np.array([self._popcount8(int(m)) for m in self.view_sector_mask[:n]], dtype=np.int32)
        return {
            "wx": self.pos_x[:n].copy(),
            "wy": self.pos_y[:n].copy(),
            "wz": self.pos_z[:n].copy(),
            "scale_xy": self.scale_xy[:n].copy(),
            "scale_z": self.scale_z[:n].copy(),
            "opacity": self.opacity[:n].copy(),
            "elev_prob": self.elev_prob[:n].copy(),
            "precision_tau": tau,
            "sigma_m": sigma,
            "r": self.rgb_u8[:n, 0].copy(),
            "g": self.rgb_u8[:n, 1].copy(),
            "b": self.rgb_u8[:n, 2].copy(),
            "obs_count": self.obs_count[:n].copy(),
            "num_sectors": sectors,
        }

    # --- 3D 轨道相机投影与 LoL 屏幕点击 Z = 0 解析射线标点 ---

    @staticmethod
    def compute_camera_basis(camera: OrbitCamera3D) -> np.ndarray:
        az = math.radians(float(camera.azimuth_deg))
        el = math.radians(min(max(float(camera.elevation_deg), -89.5), 89.5))
        dist = max(float(camera.distance_m), 0.08)

        cos_el, sin_el = math.cos(el), math.sin(el)
        cos_az, sin_az = math.cos(az), math.sin(az)

        cam_x = float(camera.target_x) + dist * cos_el * cos_az
        cam_y = float(camera.target_y) + dist * cos_el * sin_az
        cam_z = float(camera.target_z) + dist * sin_el

        dx = float(camera.target_x) - cam_x
        dy = float(camera.target_y) - cam_y
        dz = float(camera.target_z) - cam_z
        inv_fwd = 1.0 / max(1e-9, math.sqrt(dx * dx + dy * dy + dz * dz))
        fwd_x, fwd_y, fwd_z = dx * inv_fwd, dy * inv_fwd, dz * inv_fwd

        horiz = math.hypot(fwd_x, fwd_y)
        if horiz > 1e-6:
            right_x, right_y = -fwd_y / horiz, fwd_x / horiz
        else:
            right_x, right_y = sin_az, -cos_az
        right_z = 0.0

        up_x = fwd_y * right_z - fwd_z * right_y
        up_y = fwd_z * right_x - fwd_x * right_z
        up_z = fwd_x * right_y - fwd_y * right_x
        inv_up = 1.0 / max(1e-9, math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z))

        return np.array(
            [
                cam_x,
                cam_y,
                cam_z,
                fwd_x,
                fwd_y,
                fwd_z,
                right_x,
                right_y,
                right_z,
                up_x * inv_up,
                up_y * inv_up,
                up_z * inv_up,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def compute_focal_px(camera: OrbitCamera3D, viewport_height_px: float) -> float:
        fov_rad = math.radians(min(max(float(camera.fov_vertical_deg), 15.0), 120.0) * 0.5)
        return (float(viewport_height_px) * 0.5) / max(1e-5, math.tan(fov_rad))

    def project_world_point(
        self,
        wx: float,
        wy: float,
        wz: float,
        camera: OrbitCamera3D,
        viewport_width_px: float,
        viewport_height_px: float,
    ) -> tuple[float, float, float] | None:
        if viewport_width_px <= 1.0 or viewport_height_px <= 1.0:
            return None
        basis = self.compute_camera_basis(camera)
        rel_x, rel_y, rel_z = float(wx) - basis[0], float(wy) - basis[1], float(wz) - basis[2]
        z_cam = rel_x * basis[3] + rel_y * basis[4] + rel_z * basis[5]
        if z_cam <= 0.05:
            return None
        x_cam = rel_x * basis[6] + rel_y * basis[7] + rel_z * basis[8]
        y_cam = rel_x * basis[9] + rel_y * basis[10] + rel_z * basis[11]
        focal_px = self.compute_focal_px(camera, viewport_height_px)
        sx = float(viewport_width_px * 0.5 + (x_cam / z_cam) * focal_px)
        sy = float(viewport_height_px * 0.5 - (y_cam / z_cam) * focal_px)
        return sx, sy, float(z_cam)

    def raycast_screen_to_ground_ping(
        self,
        tap_x: float,
        tap_y: float,
        camera: OrbitCamera3D,
        viewport_width_px: float,
        viewport_height_px: float,
        ground_z_m: float = 0.0,
    ) -> tuple[float, float] | None:
        """在拓展后的全局 3D 视窗中点击任意像素，解析反投回 `Z = ground_z_m` 桌面世界坐标（无论是否在当前相机视野内）。"""
        if viewport_width_px <= 1.0 or viewport_height_px <= 1.0:
            return None
        basis = self.compute_camera_basis(camera)
        focal_px = self.compute_focal_px(camera, viewport_height_px)
        if focal_px <= 1e-6:
            return None

        ndc_x = (float(tap_x) - viewport_width_px * 0.5) / focal_px
        ndc_y = (viewport_height_px * 0.5 - float(tap_y)) / focal_px
        dx = basis[3] + ndc_x * basis[6] + ndc_y * basis[9]
        dy = basis[4] + ndc_x * basis[7] + ndc_y * basis[10]
        dz = basis[5] + ndc_x * basis[8] + ndc_y * basis[11]
        inv_len = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
        ray_x, ray_y, ray_z = dx * inv_len, dy * inv_len, dz * inv_len
        if ray_z >= -1e-4:
            return None
        t = (float(ground_z_m) - basis[2]) / ray_z
        if t <= 0.0:
            return None
        return float(basis[0] + t * ray_x), float(basis[1] + t * ray_y)

    def project_anisotropic_for_camera(
        self,
        camera: OrbitCamera3D,
        viewport_width_px: float,
        viewport_height_px: float,
    ) -> dict[str, np.ndarray]:
        """投影全部持久神经 3D 高斯椭球，保留各向异性屏幕长短轴 `(rad_u_px, rad_v_px)` 并由远及近排序。"""
        if self.count == 0:
            return {"count": 0}
        basis = self.compute_camera_basis(camera)
        focal_px = self.compute_focal_px(camera, viewport_height_px)
        half_w, half_h = viewport_width_px * 0.5, viewport_height_px * 0.5
        margin = 56.0

        n = self.count
        rx = self.pos_x[:n].astype(np.float64) - basis[0]
        ry = self.pos_y[:n].astype(np.float64) - basis[1]
        rz = self.pos_z[:n].astype(np.float64) - basis[2]

        z_cam = rx * basis[3] + ry * basis[4] + rz * basis[5]
        valid = z_cam > 0.06
        if not np.any(valid):
            return {"count": 0}

        inv_z = np.where(valid, 1.0 / np.maximum(z_cam, 0.06), 0.0)
        x_cam = rx * basis[6] + ry * basis[7] + rz * basis[8]
        y_cam = rx * basis[9] + ry * basis[10] + rz * basis[11]
        sx = half_w + x_cam * inv_z * focal_px
        sy = half_h - y_cam * inv_z * focal_px

        vis = (
            valid
            & (sx >= -margin)
            & (sx <= viewport_width_px + margin)
            & (sy >= -margin)
            & (sy <= viewport_height_px + margin)
        )
        if not np.any(vis):
            return {"count": 0}

        v_sx = sx[vis].astype(np.float32)
        v_sy = sy[vis].astype(np.float32)
        v_z = z_cam[vis].astype(np.float32)
        v_sxy = self.scale_xy[:n][vis]
        v_sz = self.scale_z[:n][vis]

        el_rad = math.radians(abs(float(camera.elevation_deg)))
        sin_el, cos_el = math.sin(el_rad), math.cos(el_rad)
        eff_sv = np.sqrt((v_sz * cos_el) ** 2 + (v_sxy * sin_el) ** 2)

        rad_u = np.clip(focal_px * v_sxy / v_z, 1.8, 22.0).astype(np.float32)
        rad_v = np.clip(focal_px * eff_sv / v_z, 1.5, 22.0).astype(np.float32)

        order = np.argsort(-v_z)  # 由远到近精确排序
        return {
            "count": int(order.size),
            "screen_x": v_sx[order],
            "screen_y": v_sy[order],
            "rad_u_px": rad_u[order],
            "rad_v_px": rad_v[order],
            "rgb_u8": self.rgb_u8[:n][vis][order],
            "opacity": self.opacity[:n][vis][order],
            "wz": self.pos_z[:n][vis][order],
            "elev_prob": self.elev_prob[:n][vis][order],
            "precision_tau": self.precision_tau[:n][vis][order],
            "depth_m": v_z[order],
        }

    def export_ply(self, ply_path: str | Path, comment: str = "") -> dict[str, Any]:
        """导出完整神经各向异性 3D 高斯记忆地图为标准 ASCII PLY v1.0 (`expanded_memory_map.ply`)。"""
        out_path = Path(ply_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        d = self.active_splats_dict()
        n = self.count

        header = [
            "ply",
            "format ascii 1.0",
            "comment RIG-Hover Persistent Neural 3D Gaussian Splatting Memory Map (v2)",
            f"comment voxel_size_m={self.voxel_size:.3f} steps={self.step_count} {comment}".strip(),
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float scale_xy",
            "property float scale_z",
            "property float opacity",
            "property float precision_tau",
            "property float sigma_m",
            "property int obs_count",
            "property int view_sectors",
            "end_header",
        ]
        rows = [
            f"{float(d['wx'][i]):.5f} {float(d['wy'][i]):.5f} {float(d['wz'][i]):.5f} "
            f"{int(d['r'][i])} {int(d['g'][i])} {int(d['b'][i])} "
            f"{float(d['scale_xy'][i]):.5f} {float(d['scale_z'][i]):.5f} "
            f"{float(d['opacity'][i]):.4f} {float(d['precision_tau'][i]):.2f} "
            f"{float(d['sigma_m'][i]):.5f} {int(d['obs_count'][i])} {int(d['num_sectors'][i])}"
            for i in range(n)
        ]
        out_path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")
        return {
            "ply_path": str(out_path),
            "vertex_count": int(n),
            "file_bytes": int(out_path.stat().st_size),
            "explored_area_m2": round(self.compute_explored_area_m2(), 4),
        }


# ---------------------------------------------------------------------------
# 5. 基于持久 3DGS 障碍记忆的全局任意点回溯规划与闭环导航器 (`GlobalMemoryNavigator`)
# ---------------------------------------------------------------------------

class GlobalMemoryNavigator:
    """利用 `PersistentGaussianMemoryMap` 中持久保存的全局凸起障碍高斯点云，规划并闭环驶向任意历史记忆坐标。"""

    def __init__(
        self,
        grid_res_m: float = 0.030,
        robot_clearance_radius_m: float = 0.115,
    ) -> None:
        self.grid_res_m = float(grid_res_m)
        self.robot_clearance_radius_m = float(robot_clearance_radius_m)

    def extract_elevated_obstacles_xyr(self, memory_map: PersistentGaussianMemoryMap) -> np.ndarray:
        """从持久 3DGS 地图提取高出桌面的立体障碍椭球 `(wx, wy, effective_radius)`。"""
        if memory_map.count == 0:
            return np.zeros((0, 3), dtype=np.float32)
        n = memory_map.count
        wz = memory_map.pos_z[:n]
        el = memory_map.elev_prob[:n]
        op = memory_map.opacity[:n]
        tau = memory_map.precision_tau[:n]

        is_obs = (wz >= 0.038) & (wz <= 0.32) & (el >= 0.42) & (op >= 0.45) & (tau >= 45.0)
        if not np.any(is_obs):
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(
            [
                memory_map.pos_x[:n][is_obs],
                memory_map.pos_y[:n][is_obs],
                memory_map.scale_xy[:n][is_obs],
            ],
            axis=-1,
        ).astype(np.float32)

    def plan_astar_path(
        self,
        memory_map: PersistentGaussianMemoryMap,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        bounds_xy: tuple[float, float, float, float] = (-0.45, 1.85, -0.85, 1.15),
    ) -> dict[str, Any]:
        """在持久记忆障碍栅格上执行 8 邻域软代价 A* 搜索与视线拉直平滑，返回无碰撞路径点序列。"""
        x_min, x_max, y_min, y_max = bounds_xy
        res = self.grid_res_m
        nx = int(math.ceil((x_max - x_min) / res)) + 1
        ny = int(math.ceil((y_max - y_min) / res)) + 1

        obs_xyr = self.extract_elevated_obstacles_xyr(memory_map)
        blocked = np.zeros((ny, nx), dtype=bool)
        soft_cost = np.zeros((ny, nx), dtype=np.float32)

        if obs_xyr.shape[0] > 0:
            gx_coords = x_min + np.arange(nx, dtype=np.float32)[None, :] * res
            gy_coords = y_min + np.arange(ny, dtype=np.float32)[:, None] * res
            # 下采样高密度同体素簇加速距离场生成
            for ox, oy, os_r in obs_xyr:
                hard_r = self.robot_clearance_radius_m + float(os_r)
                soft_r = hard_r + 0.09
                c0 = max(0, int((ox - soft_r - x_min) / res))
                c1 = min(nx, int((ox + soft_r - x_min) / res) + 2)
                r0 = max(0, int((oy - soft_r - y_min) / res))
                r1 = min(ny, int((oy + soft_r - y_min) / res) + 2)
                if c1 <= c0 or r1 <= r0:
                    continue
                dist = np.sqrt((gx_coords[:, c0:c1] - ox) ** 2 + (gy_coords[r0:r1, :] - oy) ** 2)
                blocked[r0:r1, c0:c1] |= dist <= hard_r
                pen = np.clip((soft_r - dist) / max(soft_r - hard_r, 1e-4), 0.0, 1.0) * 3.5
                soft_cost[r0:r1, c0:c1] = np.maximum(soft_cost[r0:r1, c0:c1], pen)

        def _to_rc(wx: float, wy: float) -> tuple[int, int]:
            c = int(np.clip(round((wx - x_min) / res), 0, nx - 1))
            r = int(np.clip(round((wy - y_min) / res), 0, ny - 1))
            return r, c

        def _to_xy(r: int, c: int) -> tuple[float, float]:
            return x_min + c * res, y_min + r * res

        sr, sc = _to_rc(start_xy[0], start_xy[1])
        gr, gc = _to_rc(goal_xy[0], goal_xy[1])
        blocked[sr, sc] = False
        blocked[gr, gc] = False

        # 8 邻域 A*
        inf = 1e12
        g_score = np.full((ny, nx), inf, dtype=np.float64)
        parent_r = np.full((ny, nx), -1, dtype=np.int32)
        parent_c = np.full((ny, nx), -1, dtype=np.int32)
        g_score[sr, sc] = 0.0

        pq: list[tuple[float, int, int]] = [(math.hypot(gr - sr, gc - sc), sr, sc)]
        nbrs = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]

        while pq:
            _, r, c = heapq.heappop(pq)
            if (r, c) == (gr, gc):
                break
            cur_g = g_score[r, c]
            for dr, dc, step_len in nbrs:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= ny or nc < 0 or nc >= nx or blocked[nr, nc]:
                    continue
                cand_g = cur_g + step_len * (1.0 + float(soft_cost[nr, nc]))
                if cand_g + 1e-9 < g_score[nr, nc]:
                    g_score[nr, nc] = cand_g
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    h_est = math.hypot(gr - nr, gc - nc)
                    heapq.heappush(pq, (cand_g + h_est, nr, nc))

        raw_rc: list[tuple[int, int]] = []
        if math.isfinite(g_score[gr, gc]) and g_score[gr, gc] < inf * 0.5:
            cr, cc = gr, gc
            while cr >= 0 and cc >= 0:
                raw_rc.append((cr, cc))
                if (cr, cc) == (sr, sc):
                    break
                pr, pc = int(parent_r[cr, cc]), int(parent_c[cr, cc])
                cr, cc = pr, pc
            raw_rc.reverse()
        else:
            raw_rc = [(sr, sc), (gr, gc)]

        # 视线无碰撞拉直平滑 (Line-of-Sight String Pulling)
        def _los_free(rc0: tuple[int, int], rc1: tuple[int, int]) -> bool:
            steps = max(2, int(math.hypot(rc1[0] - rc0[0], rc1[1] - rc0[1]) * 2.5))
            for k in range(steps + 1):
                t = k / float(steps)
                rr = int(round(rc0[0] + t * (rc1[0] - rc0[0])))
                cc = int(round(rc0[1] + t * (rc1[1] - rc0[1])))
                if blocked[rr, cc] or soft_cost[rr, cc] > 1.8:
                    return False
            return True

        smooth_rc: list[tuple[int, int]] = [raw_rc[0]]
        anchor = 0
        while anchor < len(raw_rc) - 1:
            nxt = anchor + 1
            for probe in range(len(raw_rc) - 1, anchor + 1, -1):
                if _los_free(raw_rc[anchor], raw_rc[probe]):
                    nxt = probe
                    break
            smooth_rc.append(raw_rc[nxt])
            anchor = nxt

        waypoints = [_to_xy(r, c) for r, c in smooth_rc]
        waypoints[0] = (float(start_xy[0]), float(start_xy[1]))
        waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return {
            "waypoints_xy": np.asarray(waypoints, dtype=np.float32),
            "raw_grid_path_xy": np.asarray([_to_xy(r, c) for r, c in raw_rc], dtype=np.float32),
            "obstacle_splat_count": int(obs_xyr.shape[0]),
            "occupancy_grid": blocked,
            "bounds_xy": bounds_xy,
        }

    def navigate_closed_loop(
        self,
        memory_map: PersistentGaussianMemoryMap,
        start_pose: RobotCameraPose,
        goal_xy: tuple[float, float],
        dt: float = 0.05,
        max_steps: int = 240,
    ) -> dict[str, Any]:
        """从当前位姿（如区域 B）规划并闭环追踪返回全局记忆中的任意目标点（如早已离开视野的区域 A）。"""
        plan = self.plan_astar_path(memory_map, (start_pose.x_m, start_pose.y_m), goal_xy)
        wps = plan["waypoints_xy"]
        obs_xyr = self.extract_elevated_obstacles_xyr(memory_map)

        # 检查目标点在起始时刻是否位于机器人当前 60° 相机视锥之外（盲区记忆验证）
        dx0 = float(goal_xy[0]) - float(start_pose.x_m)
        dy0 = float(goal_xy[1]) - float(start_pose.y_m)
        bearing0 = (math.degrees(math.atan2(dy0, dx0)) - float(start_pose.yaw_deg) + 180.0) % 360.0 - 180.0
        target_initially_out_of_fov = abs(bearing0) > (start_pose.fov_h_deg * 0.5 + 5.0)

        shield = EndSideSafetyShield(max_delta_vx=22.0, max_delta_vyaw=34.0)
        rx, ry, ryaw = float(start_pose.x_m), float(start_pose.y_m), float(start_pose.yaw_deg)
        v_max_mps = 0.42
        w_max_dps = 125.0

        wp_idx = 1 if len(wps) > 1 else 0
        traj_xyyaw: list[tuple[float, float, float]] = [(rx, ry, ryaw)]
        min_obs_surface_dist = float("inf")
        reached = False

        for _ in range(int(max_steps)):
            final_rem = math.hypot(float(goal_xy[0]) - rx, float(goal_xy[1]) - ry)
            if final_rem <= 0.022:
                reached = True
                break

            while wp_idx < len(wps) - 1:
                d_wp = math.hypot(float(wps[wp_idx, 0]) - rx, float(wps[wp_idx, 1]) - ry)
                if d_wp < 0.11:
                    wp_idx += 1
                else:
                    break

            tx, ty = float(wps[wp_idx, 0]), float(wps[wp_idx, 1])
            desired_yaw = math.degrees(math.atan2(ty - ry, tx - rx))
            yaw_err = (desired_yaw - ryaw + 180.0) % 360.0 - 180.0

            # 原地大角度转向阶段压低前向速度，避免背向目标起步外摆擦碰障碍
            if abs(yaw_err) > 52.0:
                raw_vx = 0.0
            else:
                align = max(0.15, math.cos(math.radians(abs(yaw_err))))
                raw_vx = min(88.0, 165.0 * final_rem) * align
            raw_vyaw = float(np.clip(2.4 * yaw_err, -92.0, 92.0))
            raw_head = float(np.clip(yaw_err, -50.0, 50.0))

            safe = shield.apply(
                raw_cmd=DriveCommand(vx=raw_vx, vyaw=raw_vyaw, head=raw_head),
                mode=MODE_APPROACH,
                remaining_m=final_rem,
                pitch_deg=1.0,
                clearance_m=1.2,
            )

            v_mps = (safe.vx / 100.0) * v_max_mps
            w_dps = (safe.vyaw / 100.0) * w_max_dps
            ryaw += w_dps * dt
            psi = math.radians(ryaw)
            rx += v_mps * math.cos(psi) * dt
            ry += v_mps * math.sin(psi) * dt
            traj_xyyaw.append((rx, ry, ryaw))

            if obs_xyr.shape[0] > 0:
                d_surf = np.sqrt((obs_xyr[:, 0] - rx) ** 2 + (obs_xyr[:, 1] - ry) ** 2) - obs_xyr[:, 2]
                cur_min = float(np.min(d_surf))
                if cur_min < min_obs_surface_dist:
                    min_obs_surface_dist = cur_min

        final_err_m = math.hypot(rx - float(goal_xy[0]), ry - float(goal_xy[1]))
        return {
            "reached": bool(reached or final_err_m < 0.040),
            "final_error_m": round(float(final_err_m), 5),
            "steps": len(traj_xyyaw) - 1,
            "target_initially_out_of_fov": bool(target_initially_out_of_fov),
            "initial_target_bearing_deg": round(float(bearing0), 2),
            "min_obstacle_surface_clearance_m": round(
                float(min_obs_surface_dist if math.isfinite(min_obs_surface_dist) else 1.0), 4
            ),
            "waypoints_xy": wps,
            "trajectory_xyyaw": np.asarray(traj_xyyaw, dtype=np.float32),
            "obstacle_splat_count": plan["obstacle_splat_count"],
        }
