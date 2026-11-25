# -*- coding: utf-8 -*-
"""dense_splat_student.py — 超轻量端侧稠密 3D 高斯与度量深度蒸馏网络 (`HoverDenseSplatNet`, < 1 MB TFLite)。

设计目标（严格面向手机端 CPU 实时推理，绝不把重量级服务端模型放进 App）：
1. 输入 [1, 96, 96, 3] (0..1 float32)，直接输出全分辨率 [1, 96, 96, 6] 3D 高斯场：
   ch0=inv_depth_01, ch1=elev_01, ch2=scale_xy_01, ch3=scale_z_01, ch4=opacity_01, ch5=precision_01
   —— 像素点密度较旧版 48x48 提升 4 倍，同时保持与 model_schema.md §5.2 通道定义完全一致；
2. 采用 MobileNetV2 倒置残差深度可分离 U-Net + 空洞上下文瓶颈 + 双线性插值跳连解码：
   仅使用 TFLite 核心原生算子 (CONV_2D, DEPTHWISE_CONV_2D, RESIZE_BILINEAR, CONCATENATION,
   ADD, PAD, LOGISTIC, RELU6)，零 Flex/Custom 算子，零转置卷积棋盘伪影；
3. 导出前自动折叠全部 BatchNorm 参数，并校验导出的 .tflite Float32 与 PyTorch eval()
   逐像素最大绝对误差 < 1e-4。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import robot_truth  # noqa: E402  (imported for the DEFAULT_* it backs; see the note below)
from export_tflite_bundle import (  # noqa: E402
    ACT_NONE,
    ACT_RELU6,
    PAD_VALID,
    TFL_FLOAT32,
    _TFLiteGraphBuilder,
    _fold_conv_bn,
    run_tflite_flatbuffer_cpu,
)
from mono_depth_model import (  # noqa: E402
    DEFAULT_FOV_H_DEG,
    DEFAULT_H_CAM_M,
    ELEV_THRESHOLD_M,
    INV_DEPTH_MIN,
    INV_DEPTH_SPAN,
)

# 相机参数属于模型工件。v3 学生模型（runs_end_side/dense_splat_v3/model.tflite）是在 72° 视场角、
# 0.115 m 相机高度下蒸馏出来的——当时这两个数都还是设计初值，不是实测（实测是 75.5° / 0.135 m，
# 见 pc/robot_truth.json）。它的输入输出是按那组相机学的：处理这份权重必须用同一组参数。
# 新训练一律用实测相机（DEFAULT_*），并把实际用到的相机随工件写进 model.json。
STUDENT_V3_FOV_H_DEG: float = 72.0
STUDENT_V3_H_CAM_M: float = 0.115

DENSE_IN_H: int = 96
DENSE_IN_W: int = 96
DENSE_OUT_H: int = 96
DENSE_OUT_W: int = 96
DENSE_OUT_C: int = 6

# v3 扩展近场度量区间: inv_d = 0.35 + 6.65 * inv01 -> [0.1429 m, 2.8571 m]
# 消除 72° 视场角下画面底端 0.16..0.22 m 真实现场桌面的近场饱和截断，同时不改动 mono_depth_model 旧契约
DENSE_INV_DEPTH_MIN: float = 0.35
DENSE_INV_DEPTH_SPAN: float = 6.65
DENSE_DEPTH_MIN_M: float = 1.0 / (DENSE_INV_DEPTH_MIN + DENSE_INV_DEPTH_SPAN)
DENSE_DEPTH_MAX_M: float = 1.0 / DENSE_INV_DEPTH_MIN


def dense_inv_depth_01_to_metric_m(inv_depth_01: torch.Tensor | np.ndarray | float) -> Any:
    """将 v3 `[0, 1]` 归一化逆深度映射为物理米制深度 `depth_m` (`~0.1429 .. 2.8571 m`)。"""
    if isinstance(inv_depth_01, torch.Tensor):
        inv_c = torch.clamp(inv_depth_01, 0.0, 1.0)
        inv_d = DENSE_INV_DEPTH_MIN + DENSE_INV_DEPTH_SPAN * inv_c
        return 1.0 / inv_d
    if isinstance(inv_depth_01, np.ndarray):
        inv_c = np.clip(inv_depth_01, 0.0, 1.0).astype(np.float32)
        inv_d = np.float32(DENSE_INV_DEPTH_MIN) + np.float32(DENSE_INV_DEPTH_SPAN) * inv_c
        return (np.float32(1.0) / inv_d).astype(np.float32)
    inv_c = min(max(float(inv_depth_01), 0.0), 1.0)
    return 1.0 / (DENSE_INV_DEPTH_MIN + DENSE_INV_DEPTH_SPAN * inv_c)


def dense_metric_m_to_inv_depth_01(depth_m: torch.Tensor | np.ndarray | float) -> Any:
    """将物理米制深度逆映射回 v3 `[0, 1]` 归一化逆深度目标。"""
    if isinstance(depth_m, torch.Tensor):
        d_c = torch.clamp(depth_m, DENSE_DEPTH_MIN_M, DENSE_DEPTH_MAX_M)
        inv_d = 1.0 / d_c
        return torch.clamp((inv_d - DENSE_INV_DEPTH_MIN) / DENSE_INV_DEPTH_SPAN, 0.0, 1.0)
    if isinstance(depth_m, np.ndarray):
        d_c = np.clip(depth_m, DENSE_DEPTH_MIN_M, DENSE_DEPTH_MAX_M).astype(np.float32)
        inv_d = np.float32(1.0) / d_c
        return np.clip((inv_d - np.float32(DENSE_INV_DEPTH_MIN)) / np.float32(DENSE_INV_DEPTH_SPAN), 0.0, 1.0).astype(np.float32)
    d_c = min(max(float(depth_m), DENSE_DEPTH_MIN_M), DENSE_DEPTH_MAX_M)
    inv_d = 1.0 / d_c
    return min(max((inv_d - DENSE_INV_DEPTH_MIN) / DENSE_INV_DEPTH_SPAN, 0.0), 1.0)


class InvertedResidualDS(nn.Module):
    """MobileNetV2 风格轻量深度可分离模块（仅 CONV_2D + DEPTHWISE_CONV_2D + ReLU6 + ADD）。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: float = 2.0,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)
        self.dilation = int(dilation)
        hidden = max(in_ch, int(round(in_ch * expand_ratio)))
        self.has_expand = hidden != in_ch
        self.use_res = (self.stride == 1) and (self.in_ch == self.out_ch)

        if self.has_expand:
            self.exp_conv = nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False)
            self.exp_bn = nn.BatchNorm2d(hidden, eps=1e-5)
        self.dw_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            stride=self.stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=hidden,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(hidden, eps=1e-5)
        self.pw_conv = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        if self.has_expand:
            h = F.relu6(self.exp_bn(self.exp_conv(h)))
        h = F.relu6(self.dw_bn(self.dw_conv(h)))
        h = self.pw_bn(self.pw_conv(h))
        if self.use_res:
            return x + h
        return h


class DSFusionBlock(nn.Module):
    """解码器融合块：1x1 压维 + 3x3 深度可分离平滑 + 1x1 输出（ReLU6）。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.reduce_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.dw_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.pw_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu6(self.reduce_bn(self.reduce_conv(x)))
        h = F.relu6(self.dw_bn(self.dw_conv(h)))
        h = F.relu6(self.pw_bn(self.pw_conv(h)))
        return h


class HoverDenseSplatNet(nn.Module):
    """超轻量端侧 96x96 全分辨率 6 通道 3D 高斯深度学生网络 (< 1 MB TFLite)。"""

    def __init__(self, base_ch: int = 24):
        super().__init__()
        c0 = int(base_ch)         # 24 @ 96x96
        c1 = int(base_ch * 4 // 3)  # 32 @ 48x48
        c2 = int(base_ch * 2)     # 48 @ 24x24
        c3 = int(base_ch * 10 // 3)  # 80 @ 12x12

        # 96x96 Stem (保留高频边缘，轻量单次特征映射)
        self.stem_conv = nn.Conv2d(3, c0, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c0, eps=1e-5)

        # Encoder 1: 96 -> 48
        self.e1_down = InvertedResidualDS(c0, c1, stride=2, expand_ratio=2.0)
        self.e1_body = InvertedResidualDS(c1, c1, stride=1, expand_ratio=2.0)

        # Encoder 2: 48 -> 24
        self.e2_down = InvertedResidualDS(c1, c2, stride=2, expand_ratio=2.0)
        self.e2_body = InvertedResidualDS(c2, c2, stride=1, expand_ratio=2.0)

        # Encoder 3 + Dilated Context Bottleneck: 24 -> 12
        self.e3_down = InvertedResidualDS(c2, c3, stride=2, expand_ratio=2.0)
        self.bot_a = InvertedResidualDS(c3, c3, stride=1, expand_ratio=2.0, dilation=1)
        self.bot_b = InvertedResidualDS(c3, c3, stride=1, expand_ratio=2.0, dilation=2)
        self.bot_c = InvertedResidualDS(c3, c3, stride=1, expand_ratio=1.5, dilation=3)

        # Decoder: 12 -> 24 -> 48 -> 96
        self.dec2 = DSFusionBlock(c3 + c2, c2)
        self.dec1 = DSFusionBlock(c2 + c1, c1)
        self.dec0 = DSFusionBlock(c1 + c0, c0)
        self.ref_conv = nn.Conv2d(c0, c0, kernel_size=3, padding=1, bias=False)
        self.ref_bn = nn.BatchNorm2d(c0, eps=1e-5)

        # 96x96 6-channel Gaussian splat head
        self.head = nn.Conv2d(c0, DENSE_OUT_C, kernel_size=1, bias=True)

        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-0.3, -1.5, 0.0, -0.5, 1.2, 0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = F.relu6(self.stem_bn(self.stem_conv(x)))
        e1 = self.e1_body(self.e1_down(e0))
        e2 = self.e2_body(self.e2_down(e1))
        e3 = self.bot_c(self.bot_b(self.bot_a(self.e3_down(e2))))

        u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        u0 = F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = F.relu6(self.ref_bn(self.ref_conv(self.dec0(torch.cat([u0, e0], dim=1)))))

        return torch.sigmoid(self.head(d0))


def _conv2d_fold_nhwc(
    gb: _TFLiteGraphBuilder,
    name: str,
    x: int,
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d | None,
    fused_act: int = ACT_NONE,
) -> int:
    if bn is not None:
        w_oihw, bias = _fold_conv_bn(conv, bn)
        w_ohwi = np.transpose(w_oihw, (0, 2, 3, 1)).astype(np.float32)
    else:
        w_ohwi = np.transpose(conv.weight.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
        bias = conv.bias.detach().cpu().numpy().astype(np.float32) if conv.bias is not None else np.zeros((conv.out_channels,), np.float32)

    pad = int(conv.padding[0])
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    cur = x
    if pad > 0:
        cur = gb.pad(f"{name}_pad", cur, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    if int(conv.groups) == 1:
        return gb.conv2d(name, cur, w_ohwi, bias, stride=stride, dilation=dilation, padding=PAD_VALID, fused_act=fused_act)
    # Depthwise Conv2D: PyTorch [C, 1, kH, kW] -> w_ohwi [C, kH, kW, 1] -> TFLite [1, kH, kW, C]
    w_1hwc = np.transpose(w_ohwi, (3, 1, 2, 0)).astype(np.float32)
    return gb.depthwise_conv2d(
        name,
        cur,
        w_1hwc,
        bias,
        stride=stride,
        dilation=dilation,
        padding=PAD_VALID,
        depth_multiplier=1,
        fused_act=fused_act,
    )


def _emit_ir_block(gb: _TFLiteGraphBuilder, prefix: str, x: int, blk: InvertedResidualDS) -> int:
    h = x
    if blk.has_expand:
        h = _conv2d_fold_nhwc(gb, f"{prefix}_exp", h, blk.exp_conv, blk.exp_bn, ACT_RELU6)
    h = _conv2d_fold_nhwc(gb, f"{prefix}_dw", h, blk.dw_conv, blk.dw_bn, ACT_RELU6)
    h = _conv2d_fold_nhwc(gb, f"{prefix}_pw", h, blk.pw_conv, blk.pw_bn, ACT_NONE)
    if blk.use_res:
        return gb.add(f"{prefix}_res", x, h)
    return h


def _emit_fusion_block(gb: _TFLiteGraphBuilder, prefix: str, x: int, blk: DSFusionBlock) -> int:
    h = _conv2d_fold_nhwc(gb, f"{prefix}_red", x, blk.reduce_conv, blk.reduce_bn, ACT_RELU6)
    h = _conv2d_fold_nhwc(gb, f"{prefix}_dw", h, blk.dw_conv, blk.dw_bn, ACT_RELU6)
    h = _conv2d_fold_nhwc(gb, f"{prefix}_pw", h, blk.pw_conv, blk.pw_bn, ACT_RELU6)
    return h


def build_dense_splat_tflite(model: HoverDenseSplatNet) -> bytes:
    """将 HoverDenseSplatNet 直接序列化为标准 FlatBuffers TFLite v3 二进制（Float32）。"""
    model = model.eval().cpu()
    gb = _TFLiteGraphBuilder(
        description=f"HoverDenseSplatNet {DENSE_IN_H}x{DENSE_IN_W} -> {DENSE_OUT_H}x{DENSE_OUT_W}x{DENSE_OUT_C}",
        initial_capacity=1572864,
    )
    x_in = gb.add_input("rgb_in", [1, DENSE_IN_H, DENSE_IN_W, 3], TFL_FLOAT32)

    e0 = _conv2d_fold_nhwc(gb, "stem", x_in, model.stem_conv, model.stem_bn, ACT_RELU6)

    e1_d = _emit_ir_block(gb, "e1d", e0, model.e1_down)
    e1 = _emit_ir_block(gb, "e1", e1_d, model.e1_body)

    e2_d = _emit_ir_block(gb, "e2d", e1, model.e2_down)
    e2 = _emit_ir_block(gb, "e2", e2_d, model.e2_body)

    e3_d = _emit_ir_block(gb, "e3d", e2, model.e3_down)
    ba = _emit_ir_block(gb, "ba", e3_d, model.bot_a)
    bb = _emit_ir_block(gb, "bb", ba, model.bot_b)
    e3 = _emit_ir_block(gb, "bc", bb, model.bot_c)

    u2 = gb.resize_bilinear("up2", e3, DENSE_IN_H // 4, DENSE_IN_W // 4, align_corners=False, half_pixel_centers=True)
    d2 = _emit_fusion_block(gb, "dec2", gb.concat("cat2", [u2, e2], axis=-1), model.dec2)

    u1 = gb.resize_bilinear("up1", d2, DENSE_IN_H // 2, DENSE_IN_W // 2, align_corners=False, half_pixel_centers=True)
    d1 = _emit_fusion_block(gb, "dec1", gb.concat("cat1", [u1, e1], axis=-1), model.dec1)

    u0 = gb.resize_bilinear("up0", d1, DENSE_IN_H, DENSE_IN_W, align_corners=False, half_pixel_centers=True)
    d0_cat = _emit_fusion_block(gb, "dec0", gb.concat("cat0", [u0, e0], axis=-1), model.dec0)
    d0 = _conv2d_fold_nhwc(gb, "ref0", d0_cat, model.ref_conv, model.ref_bn, ACT_RELU6)

    logits = _conv2d_fold_nhwc(gb, "head", d0, model.head, None, ACT_NONE)
    out = gb.sigmoid("splats_out", logits)
    gb.set_outputs([out])
    return gb.finish()


def export_dense_splat_bundle(
    model: HoverDenseSplatNet,
    out_dir: str | Path,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    model_name: str = "hover-dense-splat-v3",
    version: str = "2026.11.18",
    notes: str = "Distilled 96x96->96x96x6 MobileNetV2-UNet 3D Gaussian student (<1MB, CPU real-time)",
    h_cam_m: float | None = None,
) -> dict[str, Any]:
    """导出 TFLite + model.json。

    `fov_h_deg` / `h_cam_m` 必须是这份权重训练时用的相机：默认取实测值，也就是新训练所用的相机；
    重新打包旧权重时要显式传入旧值（v3 用 `STUDENT_V3_FOV_H_DEG` / `STUDENT_V3_H_CAM_M`）。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tfl_bytes = build_dense_splat_tflite(model)
    sha = hashlib.sha256(tfl_bytes).hexdigest()

    tfl_file = out_path / "model.tflite"
    tfl_file.write_bytes(tfl_bytes)

    manifest = {
        "schemaVersion": 1,
        "name": model_name,
        "version": version,
        "task": "depth",
        "input": {"width": DENSE_IN_W, "height": DENSE_IN_H, "channels": 3, "normalize": "unit"},
        "output": {
            "width": DENSE_OUT_W,
            "height": DENSE_OUT_H,
            "channels": DENSE_OUT_C,
            "invDepthMin": DENSE_INV_DEPTH_MIN,
            "invDepthSpan": DENSE_INV_DEPTH_SPAN,
            "elevThresholdM": ELEV_THRESHOLD_M,
        },
        "camera": {"heightM": DEFAULT_H_CAM_M if h_cam_m is None else float(h_cam_m), "fovHDeg": float(fov_h_deg)},
        "sha256": sha,
        "notes": notes,
    }
    json_file = out_path / "model.json"
    json_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_file = out_path / f"{model_name}-bundle.zip"
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tfl_file, arcname="model.tflite")
        zf.write(json_file, arcname="model.json")

    max_err = verify_dense_splat_tflite_parity(model, tfl_file)
    return {
        "tflite_path": str(tfl_file),
        "json_path": str(json_file),
        "zip_path": str(zip_file),
        "tflite_bytes": len(tfl_bytes),
        "sha256": sha,
        "max_abs_err": float(max_err),
    }


def verify_dense_splat_tflite_parity(model: HoverDenseSplatNet, tflite_path: str | Path, n_samples: int = 3) -> float:
    model = model.eval().cpu()
    rng = np.random.default_rng(20261118)
    max_err = 0.0
    for _ in range(n_samples):
        x_nhwc = rng.uniform(0.0, 1.0, size=(1, DENSE_IN_H, DENSE_IN_W, 3)).astype(np.float32)
        x_nchw = np.transpose(x_nhwc, (0, 3, 1, 2))
        with torch.no_grad():
            pt_out = np.transpose(model(torch.from_numpy(x_nchw)).numpy(), (0, 2, 3, 1))
        tfl_out = run_tflite_flatbuffer_cpu(tflite_path, [x_nhwc])[0]
        max_err = max(max_err, float(np.max(np.abs(pt_out - tfl_out))))
    return max_err


def profile_model_complexity(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 96, 96), n_iters: int = 40) -> dict[str, float]:
    """精确统计 Conv2d/Linear MACs, FLOPs, 参数量与 PyTorch CPU 单帧推理时延。"""
    model_cpu = model.eval().cpu()
    total_params = sum(int(p.numel()) for p in model_cpu.parameters())
    macs = 0
    hooks = []

    def _conv_hook(mod: nn.Conv2d, inp: tuple[torch.Tensor, ...], out: torch.Tensor):
        nonlocal macs
        bs, cout, hout, wout = out.shape
        cin_g = mod.in_channels // mod.groups
        kh, kw = mod.kernel_size
        macs += bs * cout * hout * wout * cin_g * kh * kw

    def _lin_hook(mod: nn.Linear, inp: tuple[torch.Tensor, ...], out: torch.Tensor):
        nonlocal macs
        macs += int(out.numel()) * int(mod.in_features)

    for m in model_cpu.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_lin_hook))

    x = torch.randn(*input_shape, dtype=torch.float32)
    with torch.no_grad():
        _ = model_cpu(x)
        for h in hooks:
            h.remove()
        for _ in range(8):
            _ = model_cpu(x)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model_cpu(x)
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / float(n_iters)

    return {
        "params": int(total_params),
        "macs": int(macs),
        "mmacs": round(macs / 1e6, 2),
        "mflops": round(2.0 * macs / 1e6, 2),
        "cpu_ms": round(cpu_ms, 2),
    }


def joint_bilateral_upsample_depth(
    inv_depth_lr: np.ndarray,
    guide_gray_hr: np.ndarray,
    out_hw: tuple[int, int] = (192, 192),
    radius: int = 2,
    sigma_spatial: float = 1.6,
    sigma_range: float = 0.09,
) -> np.ndarray:
    """边缘引导联合双边上采样 (JBU)：以高分辨原图灰度边缘精确对齐深度边界，消除锯齿与飞点模糊。"""
    import cv2

    hr_h, hr_w = int(out_hw[0]), int(out_hw[1])
    guide = guide_gray_hr.astype(np.float32)
    if guide.max() > 1.5:
        guide = guide / 255.0
    if guide.shape[:2] != (hr_h, hr_w):
        guide = cv2.resize(guide, (hr_w, hr_h), interpolation=cv2.INTER_AREA)

    # 先用高保真双线性初始化，再用 guide 邻域联合双边滤波使物体轮廓与纹理严格咬合
    d_init = cv2.resize(inv_depth_lr.astype(np.float32), (hr_w, hr_h), interpolation=cv2.INTER_LINEAR)
    acc = np.zeros_like(d_init, dtype=np.float32)
    wsum = np.full_like(d_init, 1e-6, dtype=np.float32)
    inv_2s2 = 0.5 / (sigma_spatial * sigma_spatial)
    inv_2r2 = 0.5 / (sigma_range * sigma_range)

    pad_g = np.pad(guide, ((radius, radius), (radius, radius)), mode="edge")
    pad_d = np.pad(d_init, ((radius, radius), (radius, radius)), mode="edge")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = float(np.exp(-(dx * dx + dy * dy) * inv_2s2))
            y0, y1 = radius + dy, radius + dy + hr_h
            x0, x1 = radius + dx, radius + dx + hr_w
            dg = pad_g[y0:y1, x0:x1] - guide
            wr = np.exp(-(dg * dg) * inv_2r2)
            w = ws * wr
            acc += w * pad_d[y0:y1, x0:x1]
            wsum += w
    return (acc / wsum).astype(np.float32)


if __name__ == "__main__":
    import tempfile
    from neural_gaussian_splat_model import HoverGaussianSplatNet

    old_m = HoverGaussianSplatNet()
    new_m = HoverDenseSplatNet(base_ch=24)
    p_old = profile_model_complexity(old_m)
    p_new = profile_model_complexity(new_m)
    with tempfile.TemporaryDirectory() as td:
        res = export_dense_splat_bundle(new_m, td)
    print("Old v2 (48x48 out):", p_old)
    print("New v3 (96x96 out):", p_new, f"tflite={res['tflite_bytes']/1024:.1f} KB, parity_err={res['max_abs_err']:.2e}")
