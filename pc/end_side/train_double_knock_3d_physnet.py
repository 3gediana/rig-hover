#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_double_knock_3d_physnet.py —— 突破 #2：3D 度量桌面接触物理击穿 2D 动作识别天花板 (`DoubleKnock3D-PhysNet`)

核心科学问题与突破：
- 现有纯 2D 双流网络 (`DoubleKnockNet`, `runs_knock/double_knock_fpsaug2_20261115`) 在 6 条真实标注录像（89 次真实双敲）的
  诚实留一交叉验证上遭遇 2D 像素空间的物理天花板：
  1) 跨距离/跨视角退化：`robot_20261115_150045_65a54d` 仅命中 `5/19`（召回 26.3%，无门控 F1=0.400；流式更跌至 `4/19`, F1=0.333）；
  2) 空中挥手/抬手高误报：`robot_20261115_151733_cb8455` 无门控误报 `FP=13`（4.85 FA/min；流式 `FP=16`, 5.97 FA/min），
     原因是在 2D 图像中“空中挥手”与“桌面敲击”的像素帧差 `|diff|` 完全不可分；
  3) 外部正样本漏检：`evidence/sessions/knock_d0.5` 漏检 (`false`)。

- 本模块将**突破 #1 自研 `HoverGeoNet-MV-SelfSup v4` (27 MB)** 实时重建的 3D 度量桌面接触物理场直接注入动作识别网络：
  1) **因果无监督桌面基准平面在线标定**（仅用视频前 20 帧低处平坦像素拟合 `Z_desk(X, Y)`，零标签、零未来泄漏）；
  2) **4 通道 3D 几何感知视觉流 `[gray, |diff|, desk_3d_motion, z_contact_field]`**：
     利用高斯桌面高度门控 `exp(-max(Z-1.5cm,0)^2 / 2σ^2)` 让空中挥手在 `desk_3d_motion` 通道中归零，仅保留桌面接触面处的真实敲击能量；
  3) **16 维 2D+3D 物理运动学流 + 3D 物理 FiLM 交叉门控 (`DeskContactPhysicsGate`)**：
     融合指节最低 3D 高度 `z_mov_p15`、距离无关 3D 下落速度 `vz_3d`、桌面止动冲击减速度 `az_3d_impact`、
     `desk_zone_energy` / `air_zone_energy` 与照明/距离不变的 `desk_to_air_ratio`、手部 3D 距离 `hand_depth_m`；
  4) **100% 诚实零泄漏 6 折留一协议**：与原 `train_double_knock.py` 完全相同的 6 条真实录像（89 次双敲）、完全相同的标注与评测函数，
     所有合成增强、平滑窗口与触发阈值 **100% 仅在 5 条训练 take 上确定，绝不触碰留出 take**！
     同时评估滑窗留一、严格因果流式留一（对齐 `stream_eval.json`）与 26 条跨数据集压力测试（`evidence/sessions` + `Charades` + `UCF-101`）。
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN_DIR = REPO / "pc" / "train"
for p in (str(HERE), str(TRAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from double_knock_data import (  # noqa: E402
    COUNT_CLASSES,
    KNOCK_IDX,
    _resample_frames_to_fps,
    classify_window_count,
    detect_contact_impulses,
    extract_kinematics_and_energy,
    load_external_stress_clips,
    make_impulse_heatmap,
)
from hover_geonet_240 import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    GEO_H,
    GEO_IN_CH,
    GEO_W,
    INV_D_MAX,
    INV_D_MIN,
    INV_D_SCALE,
    build_robot_physics_input,
    create_hover_geonet,
)
from hd_moge_contour_engine import GEONET_CKPT_FOV_H_DEG, REAL_H_CAM_M  # noqa: E402
from interval_data import IGNORE, decode_gray, frame_labels, load_intervals, load_labels, load_takes  # noqa: E402
from train_double_knock import (  # noqa: E402
    CausalConv1d,
    eval_triggers_on_take,
    find_double_knock_triggers,
    focal_bce_loss,
)

V4_CKPT_FP16 = REPO / "runs_end_side" / "hover_geonet_mv_v4" / "hover_geonet_mv_v4_fp16.pt"
V3_CKPT_FP16 = REPO / "runs_end_side" / "hover_geonet_vit_240" / "hover_geonet_vit_b1_fp16.pt"
CACHE_DIR = REPO / "pc" / "bench" / "_knock3d_cache_v4"
OUT_DIR = REPO / "runs_end_side" / "double_knock_3d_v11"
BASELINE_2D_METRICS = REPO / "runs_knock" / "double_knock_fpsaug2_20261115" / "metrics.json"
BASELINE_2D_STREAM = REPO / "runs_knock" / "double_knock_fpsaug2_20261115" / "stream_eval.json"


# ---------------------------------------------------------------------------
# 1. 批量 3D 度量桌面接触物理场提取（100% 自研 HoverGeoNet-MV-SelfSup v4）
# ---------------------------------------------------------------------------


class BatchHoverGeoNet3DExtractor:
    """用自研 27 MB `HoverGeoNet-MV-SelfSup v4` 在 GPU 上批量提取每帧 3D 桌面接触物理场与 8 维 3D 运动学。"""

    def __init__(self, ckpt_path: pathlib.Path | None = None, device: str = "cuda") -> None:
        if ckpt_path is None:
            ckpt_path = V4_CKPT_FP16 if V4_CKPT_FP16.exists() else V3_CKPT_FP16
        self.ckpt_path = pathlib.Path(ckpt_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        blob = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict_fp16") or blob.get("state_dict")
        self.model = create_hover_geonet(
            arch=str(blob.get("arch", "vit_b1")),
            in_ch=int(blob.get("in_ch", GEO_IN_CH)),
            base=int(blob.get("base", 24)),
            pretrained_backbone=False,
        )
        self.model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model.half()

        # 预计算名义前倾 3.0° 射线网格与 13.5 cm 桌面先验（单次构建，全视频复用）
        dummy_g = np.zeros((GEO_H, GEO_W), dtype=np.uint8)
        x7_0, rays_0, inv_0 = build_robot_physics_input(
            dummy_g, dummy_g, pitch_deg=3.0, roll_deg=0.0, fov_h_deg=GEONET_CKPT_FOV_H_DEG, h_cam_m=REAL_H_CAM_M
        )
        self.static_ch2_6 = x7_0[2:].copy()          # [5, H, W] (rays + inv_prior + mask)
        self.rays_t = torch.from_numpy(rays_0.transpose(2, 0, 1)).to(self.device)  # [3, H, W]
        self.inv_prior_t = torch.from_numpy(inv_0).to(self.device)                 # [H, W]

    @torch.no_grad()
    def extract_sequence_3d(
        self,
        frames_240: np.ndarray,
        fps: float = 20.0,
        size: int = 96,
        batch_size: int = 24,
        calib_frames: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """对整段 240x240 灰度视频批量运行 HoverGeoNet-v4，输出：
        - `vis_4ch`: `[T, 4, size, size]` uint8 -> `[gray, |diff|, desk_3d_motion, z_contact_field]`
        - `kin_3d`:  `[T, 8]` float32 -> `[z_p15, z_med, vz_3d, az_3d_impact, desk_E, air_E, desk_ratio, depth_m]`
        """
        T = len(frames_240)
        assert frames_240.ndim == 3, f"Expected (T, H, W), got {frames_240.shape}"
        if frames_240.shape[1:] != (GEO_H, GEO_W):
            f240 = np.stack(
                [cv2.resize(f, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA) for f in frames_240],
                axis=0,
            )
        else:
            f240 = frames_240

        # 1) 批量跑 HoverGeoNet-v4 得到下采样 96x96 的 (X, Z, nrm_z) 物理场（省内存 + 极速！）
        X_96 = np.empty((T, size, size), dtype=np.float32)
        Y_96 = np.empty((T, size, size), dtype=np.float32)
        Z_96 = np.empty((T, size, size), dtype=np.float32)
        NZ_96 = np.empty((T, size, size), dtype=np.float32)

        g_norm_all = f240.astype(np.float32) / 255.0
        # 预计算 Sobel 边缘与帧差通道（仅需微秒级 OpenCV/NumPy）
        for s in range(0, T, batch_size):
            e = min(T, s + batch_size)
            B = e - s
            x_np = np.empty((B, GEO_IN_CH, GEO_H, GEO_W), dtype=np.float32)
            x_np[:, 2:] = self.static_ch2_6[None]
            for bi, idx in enumerate(range(s, e)):
                g = g_norm_all[idx]
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                edge = np.clip(np.hypot(gx, gy) * 1.4, 0.0, 1.0)
                x_np[bi, 0] = g
                x_np[bi, 1] = edge

            xt = torch.from_numpy(x_np).to(self.device)
            if self.device.type == "cuda":
                xt = xt.half()
            out = self.model(xt).float()  # [B, 7, 240, 240]

            inv_d = (self.inv_prior_t.unsqueeze(0) + out[:, 0] * INV_D_SCALE).clamp(INV_D_MIN, INV_D_MAX)
            d = (1.0 / inv_d).clamp(DEPTH_MIN_M, DEPTH_MAX_M)  # [B, 240, 240]
            pts = self.rays_t.unsqueeze(0) * d.unsqueeze(1)     # [B, 3, 240, 240]
            pts[:, 2] = pts[:, 2] + REAL_H_CAM_M
            nrm = F.normalize(out[:, 1:4], dim=1, eps=1e-6)
            nz = nrm[:, 2:3]                                     # [B, 1, 240, 240]

            # 统一双线性下采样到 96x96（保留精准 3D 物理度量坐标）
            pack = torch.cat([pts, nz], dim=1)                   # [B, 4, 240, 240]
            pack_96 = F.interpolate(pack, size=(size, size), mode="bilinear", align_corners=False)
            p_np = pack_96.cpu().numpy()
            X_96[s:e] = p_np[:, 0]
            Y_96[s:e] = p_np[:, 1]
            Z_96[s:e] = p_np[:, 2]
            NZ_96[s:e] = p_np[:, 3]

        # 2) 因果无监督桌面平面标定：仅用前 calib_frames 帧低处平坦像素拟合 Z_desk(X, Y) = a*X + b*Y + c
        #    完全不看任何标注、完全不看后续帧，模拟机器人在桌面开机后 1 秒内自动水平校准！
        n_cal = min(T, max(6, calib_frames))
        yy = np.arange(size)[:, None]
        desk_pts_x, desk_pts_y, desk_pts_z = [], [], []
        for t in range(n_cal):
            m = (yy >= int(size * 0.56)) & (NZ_96[t] > 0.86) & (X_96[t] > 0.16) & (X_96[t] < 1.15) & (np.abs(Z_96[t]) < 0.055)
            if int(m.sum()) >= 30:
                desk_pts_x.append(X_96[t][m][::3])
                desk_pts_y.append(Y_96[t][m][::3])
                desk_pts_z.append(Z_96[t][m][::3])

        plane_a, plane_b, plane_c = 0.0, 0.0, 0.0
        desk_calibrated = 0.0
        if desk_pts_x:
            px = np.concatenate(desk_pts_x)
            py = np.concatenate(desk_pts_y)
            pz = np.concatenate(desk_pts_z)
            if len(px) >= 80:
                A = np.stack([px, py, np.ones_like(px)], axis=1).astype(np.float64)
                # 2 轮 Huber IRLS 鲁棒桌面拟合
                w = np.ones(len(px), dtype=np.float64)
                coef = np.array([0.0, 0.0, float(np.median(pz))], dtype=np.float64)
                for _ in range(3):
                    Aw = A * w[:, None]
                    bw = pz.astype(np.float64) * w
                    try:
                        coef, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
                    except np.linalg.LinAlgError:
                        break
                    res = np.abs(A @ coef - pz)
                    w = 1.0 / np.maximum(res / 0.008, 1.0)
                # 物理合理性截断：桌面倾角不超过 ±10°，截距不超过 ±4.5cm（防止在 UCF-101 等无桌面视频上发散）
                if abs(coef[0]) < 0.18 and abs(coef[1]) < 0.18 and abs(coef[2]) < 0.045:
                    plane_a, plane_b, plane_c = float(coef[0]), float(coef[1]), float(coef[2])
                    desk_calibrated = 1.0

        Z_above = Z_96 - (plane_a * X_96 + plane_b * Y_96 + plane_c)  # [T, 96, 96], 桌面精确在 Z_above = 0!

        # 3) 构造 96x96 4 通道 3D 物理感知视觉流：[gray, |diff|, desk_3d_motion, z_contact_field]
        f96 = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in f240], axis=0)
        g96 = f96.astype(np.float32) / 255.0
        d96 = np.zeros_like(g96)
        if T > 1:
            d96[1:] = np.abs(f96[1:].astype(np.int16) - f96[:-1].astype(np.int16)).astype(np.float32) / 255.0

        # 距离与高度物理门控：仅允许 0.14m < X < 1.45m 且 Z_above ∈ [-2.5cm, +7.5cm] 桌面接触带通过！
        in_reach = ((X_96 > 0.14) & (X_96 < 1.45)).astype(np.float32)
        z_pos = np.maximum(Z_above - 0.015, 0.0)
        z_neg = np.maximum(-0.025 - Z_above, 0.0)
        desk_gate = np.exp(-0.5 * ((z_pos / 0.045) ** 2 + (z_neg / 0.020) ** 2)) * in_reach
        air_gate = (Z_above > 0.105).astype(np.float32) * in_reach

        desk_3d_motion = np.clip(d96 * desk_gate * 2.5, 0.0, 1.0)
        z_contact_field = np.clip(
            np.exp(-0.5 * ((Z_above - 0.022) / 0.058) ** 2) * (1.0 / (1.0 + np.exp((X_96 - 1.38) / 0.12))),
            0.0,
            1.0,
        ).astype(np.float32)

        vis_4ch_f = np.stack([g96, d96, desk_3d_motion, z_contact_field], axis=1)  # [T, 4, 96, 96]
        vis_4ch = np.rint(np.clip(vis_4ch_f, 0.0, 1.0) * 255.0).astype(np.uint8)

        # 4) 提取 8 维 3D 度量桌面接触物理运动学序列
        z_mov_p15 = np.full(T, 2.2, dtype=np.float32)
        z_mov_med = np.full(T, 2.5, dtype=np.float32)
        hand_depth = np.full(T, 0.85, dtype=np.float32)
        desk_energy = (desk_3d_motion.mean(axis=(1, 2)) * 14.0).astype(np.float32)
        air_energy = ((d96 * air_gate).mean(axis=(1, 2)) * 22.0).astype(np.float32)
        desk_energy = np.clip(desk_energy, 0.0, 6.0)
        air_energy = np.clip(air_energy, 0.0, 6.0)
        desk_to_air = (desk_energy / (desk_energy + air_energy + 0.04)).astype(np.float32)

        for t in range(1, T):
            # 寻找活跃前景运动像素（自动适应不同亮度与距离）
            dt = d96[t]
            thr = max(0.032, float(np.percentile(dt, 92)))
            act = (dt >= thr) & (X_96[t] > 0.14) & (X_96[t] < 1.45)
            if int(act.sum()) >= 12:
                za = Z_above[t][act]
                xa = X_96[t][act]
                z_mov_p15[t] = float(np.clip(np.percentile(za, 15) / 0.10, -0.5, 3.0))
                z_mov_med[t] = float(np.clip(np.median(za) / 0.10, -0.5, 3.0))
                hand_depth[t] = float(np.clip(np.median(xa), 0.15, 1.50))
            elif t > 1:
                # 静止帧做平滑衰减回中位，避免差分伪尖峰
                z_mov_p15[t] = 0.85 * z_mov_p15[t - 1] + 0.15 * 2.2
                z_mov_med[t] = 0.85 * z_mov_med[t - 1] + 0.15 * 2.5
                hand_depth[t] = hand_depth[t - 1]

        # 3D 竖直速度 vz_3d（向下落向桌面为负！）与桌面触底止动冲击减速度 az_3d_impact（向上反弹/骤停为正！）
        vz_3d = np.zeros(T, dtype=np.float32)
        if T > 1:
            vz_3d[1:] = np.clip((z_mov_p15[1:] - z_mov_p15[:-1]) * (fps * 0.5), -4.0, 4.0)
        az_3d_impact = np.zeros(T, dtype=np.float32)
        if T > 2:
            # 物理接触冲击条件：上一帧向下运动 (vz[t-1] < 0)，当前帧骤停/回弹 (vz[t] - vz[t-1] > 0)，且指节位于桌面附近 (Z <= 6cm)
            down_speed = np.maximum(0.0, -vz_3d[:-1])
            stop_jump = np.maximum(0.0, vz_3d[1:] - vz_3d[:-1])
            z_prox = np.exp(-np.maximum(z_mov_p15[1:], 0.0) / 0.65)
            az_3d_impact[1:] = np.clip(np.minimum(down_speed * 1.4, stop_jump) * 1.8 * z_prox, 0.0, 6.0)

        kin_3d = np.stack(
            [z_mov_p15, z_mov_med, vz_3d, az_3d_impact, desk_energy, air_energy, desk_to_air, hand_depth],
            axis=1,
        ).astype(np.float32)
        kin_3d = np.nan_to_num(kin_3d, nan=0.0, posinf=4.0, neginf=-4.0)

        stats = {
            "desk_calibrated": desk_calibrated,
            "plane_a": round(plane_a, 4),
            "plane_b": round(plane_b, 4),
            "plane_c_mm": round(plane_c * 1000.0, 2),
            "mean_desk_energy": round(float(desk_energy.mean()), 4),
            "mean_air_energy": round(float(air_energy.mean()), 4),
        }
        return vis_4ch, kin_3d, stats


# ---------------------------------------------------------------------------
# 2. DoubleKnock3D-PhysNet：3D 物理 FiLM 交叉门控 + 多尺度因果 TCN-GRU 双头网络
# ---------------------------------------------------------------------------


class DoubleKnock3DPhysNet(nn.Module):
    """融合 3D 桌面接触物理场 (`4ch`) 与 16 维 2D+3D 物理运动学的因果双流动作识别网络。"""

    def __init__(self, in_ch: int = 4, kin_dim: int = 16, hidden_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.kin_dim = kin_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_3d_film = (kin_dim > 8)

        vis_dim = hidden_dim
        k_emb = 48 if self.use_3d_film else hidden_dim // 2
        fuse_dim = vis_dim + k_emb

        self.vis_enc = nn.Sequential(
            nn.Conv2d(in_ch, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, vis_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(vis_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.kin_enc = nn.Sequential(
            nn.Linear(kin_dim, k_emb),
            nn.LayerNorm(k_emb),
            nn.SiLU(inplace=True),
            nn.Linear(k_emb, k_emb),
            nn.LayerNorm(k_emb),
            nn.SiLU(inplace=True),
        )
        # 3D 物理 FiLM 交叉门控：用 8 维 3D 度量桌面物理实时调制视觉流（空中挥手时自动抑制视觉伪激活！）
        if self.use_3d_film:
            self.phys_film = nn.Sequential(
                nn.Linear(kin_dim - 8, 32),
                nn.SiLU(inplace=True),
                nn.Linear(32, vis_dim * 2),
            )
            nn.init.zeros_(self.phys_film[-1].weight)
            nn.init.zeros_(self.phys_film[-1].bias)
        else:
            self.phys_film = None

        self.tcn1 = CausalConv1d(fuse_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn4 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        self.tcn8 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.22)

        self.impulse_head = nn.Linear(hidden_dim, 1)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x_vis.shape
        v = self.vis_enc(x_vis.reshape(b * t, c, h, w)).reshape(b, t, -1)
        k = self.kin_enc(x_kin)
        if self.phys_film is not None:
            gb = self.phys_film(x_kin[:, :, 8:])
            gamma, beta = gb.chunk(2, dim=-1)
            v = v * (1.0 + 0.55 * torch.tanh(gamma)) + 0.35 * beta

        fused = torch.cat([v, k], dim=-1)
        u = fused.transpose(1, 2)
        u1 = self.tcn1(u)
        u2 = self.tcn2(u1) + u1
        u4 = self.tcn4(u2) + u2
        u8 = self.tcn8(u4) + u4
        u_seq = u8.transpose(1, 2)

        h_seq, _ = self.gru(self.drop(u_seq))
        h_seq = h_seq + u_seq

        logit_imp = self.impulse_head(self.drop(h_seq)).squeeze(-1)
        p_imp = torch.sigmoid(logit_imp)

        half_t = max(1, t // 2)
        imp_sum = p_imp.sum(dim=1, keepdim=True) / 6.0
        imp_max = p_imp.amax(dim=1, keepdim=True)
        imp_first = p_imp[:, :half_t].amax(dim=1, keepdim=True)
        imp_second = p_imp[:, half_t:].amax(dim=1, keepdim=True)
        imp_rise = torch.relu(p_imp[:, 1:] - p_imp[:, :-1]).sum(dim=1, keepdim=True) / 2.5
        imp_stats = torch.cat([imp_sum, imp_max, imp_first, imp_second, imp_rise], dim=-1)

        attn_w = torch.softmax(self.attn_score(h_seq), dim=1)
        h_attn = (attn_w * h_seq).sum(dim=1)
        h_last = h_seq[:, -1, :]
        logit_cnt = self.count_head(torch.cat([h_attn, h_last, imp_stats], dim=-1))
        return logit_imp, logit_cnt


# ---------------------------------------------------------------------------
# 3. 3D 物理增强的数据集构建 + 100% 诚实零泄漏 6 折留一训练与流式评测
# ---------------------------------------------------------------------------


def build_3d_dataset(
    extractor: BatchHoverGeoNet3DExtractor,
    window: int = 48,
    stride: int = 4,
    size: int = 96,
    use_3d: bool = True,
) -> dict[str, dict[str, Any]]:
    """构建全部 6 条真实标注 take 的 2D+3D 物理窗口数据集（自动缓存 3D 特征加速复现）。"""
    ds_path = REPO / "dataset"
    roi_path = REPO / "runs_knock" / "pose_roi"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    id_to_name = load_labels(ds_path)
    takes = load_takes(ds_path)
    out: dict[str, dict[str, Any]] = {}

    for take in takes:
        if take.annotation_path is None:
            continue
        intervals = load_intervals(take, id_to_name)
        frames_240_src = decode_gray(take.video_path, GEO_H)
        src_t = len(frames_240_src)
        frames_240, src_idx, feat_fps = _resample_frames_to_fps(frames_240_src, take.fps, 20.0)
        T = len(frames_240)
        frames_96 = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames_240], axis=0)
        labels = frame_labels(intervals, T, feat_fps)

        npz_file = roi_path / f"{take.take_id}.npz"
        boxes = np.load(npz_file, allow_pickle=True)["boxes"] if npz_file.exists() else None
        if boxes is not None:
            boxes = np.asarray(boxes, dtype=object)
            if len(boxes) == src_t:
                boxes = boxes[src_idx]

        kin_2d, impact_2d = extract_kinematics_and_energy(frames_96, boxes, fps=feat_fps)

        # 读取或提取自研 HoverGeoNet-MV-SelfSup v4 3D 物理场缓存
        cache_file = CACHE_DIR / f"{take.take_id}_3d.npz"
        if cache_file.exists():
            z = np.load(cache_file)
            vis_4ch_u8 = z["vis_4ch"]
            kin_3d = z["kin_3d"].astype(np.float32)
            stats_3d = json.loads(str(z["stats_json"]))
        else:
            t0 = time.perf_counter()
            vis_4ch_u8, kin_3d, stats_3d = extractor.extract_sequence_3d(frames_240, fps=feat_fps, size=size)
            np.savez_compressed(cache_file, vis_4ch=vis_4ch_u8, kin_3d=kin_3d.astype(np.float16), stats_json=json.dumps(stats_3d))
            print(f"  [3D-Cache BUILT] {take.take_id}: T={T} ({time.perf_counter()-t0:.1f}s) | {stats_3d}", flush=True)

        if use_3d:
            vis_u8 = vis_4ch_u8
            kin_full = np.concatenate([kin_2d, kin_3d], axis=1).astype(np.float32)  # [T, 16]
            # 接触脉冲定位仅在训练折已标注的 knock 区间内进行，融合 3D 桌面冲击信号可提升脉冲时刻对齐精度
            # (注意：留出测试 take 绝不使用 contacts 评测，只用人工标注 intervals 与 labels_full!)
            impact_combined = (impact_2d + 0.35 * np.clip(kin_3d[:, 3], 0.0, 5.0) + 0.25 * np.clip(kin_3d[:, 4], 0.0, 5.0)).astype(np.float32)
        else:
            vis_u8 = vis_4ch_u8[:, :2]
            kin_full = kin_2d.astype(np.float32)
            impact_combined = impact_2d

        contacts = detect_contact_impulses(impact_combined, intervals, fps=feat_fps, min_gap_s=0.28)
        y_imp_full = make_impulse_heatmap(T, contacts, sigma=1.25)
        vis_full_f = (vis_u8.astype(np.float32) / 255.0).astype(np.float16)

        x_vis_list, x_kin_list, y_imp_list, y_cnt_list = [], [], [], []
        for s in range(0, max(1, T - window + 1), stride):
            e = s + window
            if e > T:
                break
            win_lbls = labels[s:e]
            if float(np.mean(win_lbls != IGNORE)) < 0.55:
                continue
            cnt_cls = classify_window_count(contacts, s, e, fps=feat_fps)
            if np.sum(win_lbls == KNOCK_IDX) == 0:
                cnt_cls = 0
            x_vis_list.append(vis_u8[s:e])
            x_kin_list.append(kin_full[s:e])
            y_imp_list.append(y_imp_full[s:e])
            y_cnt_list.append(cnt_cls)

        stroke_half = 6
        impulses = []
        for c in contacts:
            if c - stroke_half >= 0 and c + stroke_half <= T:
                impulses.append({
                    "vis": (vis_u8[c - stroke_half : c + stroke_half].astype(np.float32) / 255.0),
                    "kin": kin_full[c - stroke_half : c + stroke_half].copy(),
                })
        negatives = []
        for i, c_cls in enumerate(y_cnt_list):
            if c_cls == 0:
                negatives.append({
                    "vis": (x_vis_list[i].astype(np.float32) / 255.0),
                    "kin": x_kin_list[i].copy(),
                })

        out[take.take_id] = {
            "fps": feat_fps,
            "duration_s": float(take.duration_s),
            "vis_full": vis_full_f,
            "kin_full": kin_full,
            "labels_full": labels,
            "y_imp_full": y_imp_full,
            "contacts": contacts,
            "intervals": intervals,
            "x_vis": np.stack(x_vis_list).astype(np.uint8),
            "x_kin": np.stack(x_kin_list).astype(np.float32),
            "y_imp": np.stack(y_imp_list).astype(np.float32),
            "y_cnt": np.asarray(y_cnt_list, dtype=np.int64),
            "impulses": impulses,
            "negatives": negatives,
            "stats_3d": stats_3d,
        }
    return out


def synthesize_windows_3d(
    train_data: dict[str, dict],
    forbidden_val_take: str | None,
    n_per_class: int = 100,
    window: int = 48,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """仅从训练折的真实单次脉冲与真实背景窗口中合成 1/2/3 下敲击（支持任意 in_ch 与 kin_dim）。"""
    if forbidden_val_take is not None and forbidden_val_take in train_data:
        raise ValueError(f"LEAKAGE BLOCKED: {forbidden_val_take} in train_data!")
    all_imp, all_neg = [], []
    for d in train_data.values():
        all_imp.extend(d["impulses"])
        all_neg.extend(d["negatives"])
    rng = np.random.default_rng(seed)
    stroke_len = all_imp[0]["vis"].shape[0]
    half = stroke_len // 2
    blend = np.hanning(stroke_len + 2)[1:-1].astype(np.float32)
    b_vis = blend[:, None, None, None]
    b_kin = blend[:, None]

    xv, xk, yi, yc = [], [], [], []
    for target_cls in (1, 2, 3):
        for _ in range(n_per_class):
            neg = all_neg[int(rng.integers(0, len(all_neg)))]
            vis = neg["vis"].copy()
            kin = neg["kin"].copy()
            if target_cls == 1:
                centers = [int(rng.integers(half + 4, window - half - 4))]
            elif target_cls == 2:
                gap = int(rng.integers(6, min(18, window - 2 * stroke_len - 4)))
                c1 = int(rng.integers(half + 3, window - half - gap - 3))
                centers = [c1, c1 + gap]
            else:
                g1 = int(rng.integers(5, 9))
                g2 = int(rng.integers(5, 9))
                c1 = int(rng.integers(half + 2, max(half + 3, window - half - g1 - g2 - 2)))
                centers = [c1, c1 + g1, min(window - half - 2, c1 + g1 + g2)]
            for c in centers:
                imp = all_imp[int(rng.integers(0, len(all_imp)))]
                s, e = c - half, c - half + stroke_len
                if 0 <= s and e <= window:
                    vis[s:e] = (1.0 - b_vis) * vis[s:e] + b_vis * imp["vis"]
                    kin[s:e] = (1.0 - b_kin) * kin[s:e] + b_kin * imp["kin"]
            xv.append(np.rint(np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8))
            xk.append(kin.astype(np.float32))
            yi.append(make_impulse_heatmap(window, centers, sigma=1.25))
            yc.append(target_cls)
    return {
        "x_vis": np.stack(xv),
        "x_kin": np.stack(xk),
        "y_imp": np.stack(yi),
        "y_cnt": np.asarray(yc, dtype=np.int64),
    }


def _resample_idx(n_in: int, n_out: int) -> np.ndarray:
    return np.rint(np.linspace(0.0, n_in - 1, n_out)).astype(np.int64)


def augment_3d_batch(
    x_vis: torch.Tensor,
    x_kin: torch.Tensor,
    rng: np.random.Generator,
    fps_aug_p: float = 0.45,
    base_fps: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """同步 4 通道视觉 + 16 维 2D/3D 运动学增强（帧率不变性 + 水平翻转 + 亮度扰动 + 3D 尺度微扰）。"""
    v = x_vis.clone()
    k = x_kin.clone()
    b, t, ch = v.shape[0], v.shape[1], v.shape[2]
    has_3d = (k.shape[-1] > 8)
    dev = v.device

    for i in range(b):
        # 1) 随机帧率退化 (10..20 fps) 保证跨 15fps/20fps take 鲁棒！
        if rng.random() < fps_aug_p:
            rate = float(rng.uniform(10.0, 19.0))
            n_low = max(8, int(round(t * rate / base_fps)))
            if n_low < t:
                low = torch.from_numpy(_resample_idx(t, n_low)).to(dev)
                up = torch.from_numpy(_resample_idx(n_low, t)).to(dev)
                g_low = v[i, low, 0]
                d_low = torch.zeros_like(g_low)
                d_low[1:] = (g_low[1:] - g_low[:-1]).abs()
                v[i, :, 0] = g_low[up]
                v[i, :, 1] = d_low[up]
                if ch >= 4:
                    z_prox = v[i, low, 3][up]
                    v[i, :, 3] = z_prox
                    v[i, :, 2] = (d_low[up] * z_prox * 2.5).clamp(0.0, 1.0)
                k[i] = k[i, low][up]
                # 同步重算 2D 速度/加速度
                wy, wx = k[i, :, 1], k[i, :, 0]
                vy = torch.zeros(t, device=dev)
                vx = torch.zeros(t, device=dev)
                vy[1:] = ((wy[1:] - wy[:-1]) * (base_fps * 0.5)).clamp(-4.0, 4.0)
                vx[1:] = ((wx[1:] - wx[:-1]) * (base_fps * 0.5)).clamp(-4.0, 4.0)
                ay = torch.zeros(t, device=dev)
                ay[:-1] = ((vy[:-1] - vy[1:]).clamp(min=0.0) * 1.5).clamp(0.0, 6.0)
                k[i, :, 2], k[i, :, 3], k[i, :, 4] = vy, ay, vx
                k[i, :, 6] = v[i, :, 1].flatten(1).mean(dim=1) * 10.0

        # 2) 水平翻转：所有视觉通道同步翻转，仅反转 wx (0) 和 vx (4)；3D Z/能量/距离左右对称！
        if rng.random() < 0.5:
            v[i] = torch.flip(v[i], dims=[-1])
            k[i, :, 0] = -k[i, :, 0]
            k[i, :, 4] = -k[i, :, 4]

        # 3) 光照与对比度微扰（仅扰动灰度通道 0，不破坏 3D 几何通道）
        sc = float(rng.uniform(0.84, 1.16))
        bi = float(rng.uniform(-0.08, 0.08))
        v[i, :, 0] = (v[i, :, 0] * sc + bi).clamp(0.0, 1.0)

        # 4) 3D 高度与能量轻微领域随机化（防止过拟合单条 take 的桌面微倾角）
        if has_3d:
            dz_off = float(rng.uniform(-0.08, 0.08))
            k[i, :, 8] = (k[i, :, 8] + dz_off).clamp(-0.5, 3.0)
            k[i, :, 9] = (k[i, :, 9] + dz_off).clamp(-0.5, 3.0)

    k = k + torch.randn_like(k) * 0.03
    return v, k


@torch.no_grad()
def predict_seq_both_modes(
    model: DoubleKnock3DPhysNet,
    vis_full: np.ndarray,
    kin_full: np.ndarray,
    window: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """同时计算：
    1) 滑窗集成预测 `(imp_prob, cls_prob)`（对齐 `metrics.json`）
    2) 严格因果逐帧流式预测 `(imp_causal, cls_causal)`（对齐 `stream_eval.json`，每帧只看 `[t-47..t]` 零未来帧！）
    """
    model.eval()
    T = len(vis_full)
    # 1) 滑窗集成 (stride=2, 与原 train_double_knock.py 完全一致)
    imp_sum = np.zeros(T, dtype=np.float64)
    imp_cnt = np.zeros(T, dtype=np.float64)
    cls_sum = np.zeros((T, 4), dtype=np.float64)
    cls_cnt = np.zeros(T, dtype=np.float64)

    starts = list(range(0, max(1, T - window + 1), 2))
    if T >= window and starts[-1] != T - window:
        starts.append(T - window)
    bv, bk, bs = [], [], []
    for s in starts:
        e = min(T, s + window)
        pad = window - (e - s)
        vs = vis_full[s:e]
        ks = kin_full[s:e]
        if pad > 0:
            vs = np.pad(vs, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
            ks = np.pad(ks, ((0, pad), (0, 0)), mode="edge")
        bv.append(vs)
        bk.append(ks)
        bs.append((s, e))
        if len(bv) == 32 or s == starts[-1]:
            xv = torch.from_numpy(np.stack(bv)).float().to(device)
            xk = torch.from_numpy(np.stack(bk)).float().to(device)
            li, lc = model(xv, xk)
            pi = torch.sigmoid(li).cpu().numpy()
            pc = torch.softmax(lc, dim=-1).cpu().numpy()
            for bi_idx, (s0, e0) in enumerate(bs):
                L = e0 - s0
                imp_sum[s0:e0] += pi[bi_idx, :L]
                imp_cnt[s0:e0] += 1.0
                half_L = max(1, L // 2)
                cls_sum[s0 + L - half_L : e0] += pc[bi_idx]
                cls_cnt[s0 + L - half_L : e0] += 1.0
            bv, bk, bs = [], [], []
    imp_prob = (imp_sum / np.maximum(imp_cnt, 1.0)).astype(np.float32)
    fv = np.where(cls_cnt > 0)[0]
    if fv.size > 0:
        cls_sum[: fv[0]] = cls_sum[fv[0]]
        cls_cnt[: fv[0]] = cls_cnt[fv[0]]
    cls_prob = (cls_sum / np.maximum(cls_cnt[:, None], 1.0)).astype(np.float32)

    # 2) 严格因果逐帧流式预测（仅取末端 logit_imp[:, -1] 与当前窗 logit_cnt，绝不看未来帧！）
    imp_causal = np.zeros(T, dtype=np.float32)
    cls_causal = np.zeros((T, 4), dtype=np.float32)
    cls_causal[:, 0] = 1.0
    if T >= window:
        ends = list(range(window - 1, T))
        bv, bk, be = [], [], []
        for end_i in ends:
            s = end_i - window + 1
            bv.append(vis_full[s : end_i + 1])
            bk.append(kin_full[s : end_i + 1])
            be.append(end_i)
            if len(bv) == 64 or end_i == ends[-1]:
                xv = torch.from_numpy(np.stack(bv)).float().to(device)
                xk = torch.from_numpy(np.stack(bk)).float().to(device)
                li, lc = model(xv, xk)
                pi = torch.sigmoid(li[:, -1]).cpu().numpy()
                pc = torch.softmax(lc, dim=-1).cpu().numpy()
                for bi_idx, ei in enumerate(be):
                    imp_causal[ei] = pi[bi_idx]
                    cls_causal[ei] = pc[bi_idx]
                bv, bk, be = [], [], []

    return imp_prob, cls_prob, imp_causal, cls_causal


def find_causal_streaming_triggers(
    imp_causal: np.ndarray,
    cls_causal: np.ndarray,
    fps: float,
    smooth_frames: int = 3,
    imp_thr: float = 0.35,
    double_thr: float = 0.35,
    pair_min_s: float = 0.20,
    pair_max_s: float = 1.45,
    cooldown_s: float = 1.8,
) -> list[dict]:
    """与 `DoubleKnockDetector.step()` 100% 数学等价的严格因果流式触发器（零全局统计、零未来帧！）。"""
    from collections import deque

    imp_hist: deque[float] = deque(maxlen=max(1, smooth_frames))
    sm_hist: deque[float] = deque(maxlen=max(5, 2 * smooth_frames))
    sem_hist: deque[float] = deque(maxlen=max(8, 2 * smooth_frames))
    peak_times: deque[float] = deque(maxlen=10)
    last_t = -1e9
    triggers = []

    for i in range(len(imp_causal)):
        p_imp = float(imp_causal[i])
        p_dbl = float(cls_causal[i, 2] + 0.45 * cls_causal[i, 3])
        imp_hist.append(p_imp)
        sm = 0.65 * p_imp + 0.35 * float(np.mean(imp_hist))
        sm_hist.append(sm)
        sem_hist.append(p_dbl)
        if len(sm_hist) < 3:
            continue
        sh = list(sm_hist)
        prev, cur = sh[-2], sh[-1]
        is_peak = (prev >= imp_thr) and (prev >= sh[-3]) and (prev > cur)
        if not is_peak:
            continue
        t_now = i / fps
        while peak_times and t_now - peak_times[0] > pair_max_s + 0.5:
            peak_times.popleft()
        peak_times.append(t_now)
        if len(peak_times) >= 2 and pair_min_s <= (t_now - peak_times[-2]) <= pair_max_s:
            win_sem = float(max(sem_hist))
            if win_sem >= double_thr and (t_now - last_t >= cooldown_s):
                triggers.append({"first_s": round(peak_times[-2], 3), "second_s": round(t_now, 3), "score": round(win_sem, 3)})
                last_t = t_now
    return triggers


def calibrate_both_on_train_only(
    preds_train: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_data: dict[str, dict],
) -> tuple[dict, dict]:
    """严格仅在 5 条训练 take 上网格搜索滑窗与流式阈值（绝不触碰留出 take！）。"""
    best_win = None
    for sm in (2, 3, 5):
        for imp_thr in (0.6, 0.9, 1.2, 1.5, 1.8):
            for dbl_thr in (0.25, 0.35, 0.45, 0.55):
                th = tgt = ttp = tfp = 0
                for tid, d in train_data.items():
                    ip, cp, _, _ = preds_train[tid]
                    tr = find_double_knock_triggers(ip, cp, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr)
                    ev = eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; ttp += ev["tp"]; tfp += ev["fp"]
                prec = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.008 * tfp
                if best_win is None or score > best_win["score"]:
                    best_win = {
                        "score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr,
                        "train_f1": round(f1, 4), "train_hit": th, "train_gt": tgt, "train_fp": tfp,
                    }

    best_str = None
    for sm in (2, 3, 5):
        for imp_thr in (0.22, 0.28, 0.35, 0.42, 0.50):
            for dbl_thr in (0.20, 0.30, 0.38, 0.48):
                th = tgt = ttp = tfp = 0
                for tid, d in train_data.items():
                    _, _, ic, cc = preds_train[tid]
                    tr = find_causal_streaming_triggers(ic, cc, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr)
                    ev = eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; ttp += ev["tp"]; tfp += ev["fp"]
                prec = ttp / (ttp + tfp) if (ttp + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.008 * tfp
                if best_str is None or score > best_str["score"]:
                    best_str = {
                        "score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr,
                        "train_f1": round(f1, 4), "train_hit": th, "train_gt": tgt, "train_fp": tfp,
                    }
    return best_win, best_str


def train_one_fold_model(
    train_sub: dict[str, dict],
    forbidden_val_take: str | None,
    in_ch: int,
    kin_dim: int,
    epochs: int = 22,
    batch_size: int = 32,
    lr: float = 1.2e-3,
    synth_per_class: int = 100,
    device: torch.device = torch.device("cuda"),
    seed: int = 100,
) -> DoubleKnock3DPhysNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    xv_parts = [d["x_vis"] for d in train_sub.values()]
    xk_parts = [d["x_kin"] for d in train_sub.values()]
    yi_parts = [d["y_imp"] for d in train_sub.values()]
    yc_parts = [d["y_cnt"] for d in train_sub.values()]

    if synth_per_class > 0:
        syn = synthesize_windows_3d(train_sub, forbidden_val_take=forbidden_val_take, n_per_class=synth_per_class, seed=seed)
        xv_parts.append(syn["x_vis"])
        xk_parts.append(syn["x_kin"])
        yi_parts.append(syn["y_imp"])
        yc_parts.append(syn["y_cnt"])

    y_cnt_all = np.concatenate(yc_parts, axis=0)
    idx_map = [(p_i, r_i) for p_i, arr in enumerate(yc_parts) for r_i in range(len(arr))]

    cls_counts = np.bincount(y_cnt_all, minlength=4).astype(np.float32)
    cls_weights = np.sum(cls_counts) / (4.0 * np.maximum(cls_counts, 1.0))
    cls_weights[2] *= 1.35
    cls_w_t = torch.from_numpy(cls_weights).to(device)

    model = DoubleKnock3DPhysNet(in_ch=in_ch, kin_dim=kin_dim, hidden_dim=64, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    N = len(idx_map)
    for _ep in range(epochs):
        model.train()
        perm = rng.permutation(N)
        for st in range(0, N, batch_size):
            ids = perm[st : st + batch_size]
            bv = torch.from_numpy(np.stack([xv_parts[idx_map[i][0]][idx_map[i][1]] for i in ids])).float().div_(255.0).to(device)
            bk = torch.from_numpy(np.stack([xk_parts[idx_map[i][0]][idx_map[i][1]] for i in ids])).float().to(device)
            bi = torch.from_numpy(np.stack([yi_parts[idx_map[i][0]][idx_map[i][1]] for i in ids])).float().to(device)
            bc = torch.from_numpy(np.array([yc_parts[idx_map[i][0]][idx_map[i][1]] for i in ids], dtype=np.int64)).to(device)

            bv, bk = augment_3d_batch(bv, bk, rng, fps_aug_p=0.45)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                li, lc = model(bv, bk)
                l_imp = focal_bce_loss(li.float(), bi, gamma=2.0, pos_weight=4.5)
                l_cnt = F.cross_entropy(lc.float(), bc, weight=cls_w_t, label_smoothing=0.05)
                loss = l_imp + 0.90 * l_cnt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    return model


def eval_external_stress_3d(
    deploy_model: DoubleKnock3DPhysNet,
    calib_win: dict,
    calib_str: dict,
    extractor: BatchHoverGeoNet3DExtractor,
    window: int,
    device: torch.device,
) -> dict[str, Any]:
    """在 26 条跨数据集压力集 (evidence/sessions + Charades + UCF-101) 上评测零误报与正样本召回！"""
    clips = load_external_stress_clips(max_clips_per_source=10, size=GEO_H)
    neg_min = 0.0
    neg_fp_win = 0
    neg_fp_str = 0
    pos_hit_win = False
    pos_hit_str = False
    clip_rows = []

    for cl in clips:
        f240 = cl["frames"]
        src_fps = float(cl.get("fps", 20.0))
        f240_r, _, feat_fps = _resample_frames_to_fps(f240, src_fps, 20.0)
        f96 = np.stack([cv2.resize(f, (96, 96), interpolation=cv2.INTER_AREA) for f in f240_r], axis=0)
        kin_2d, _ = extract_kinematics_and_energy(f96, None, fps=feat_fps)
        vis_4ch_u8, kin_3d, _ = extractor.extract_sequence_3d(f240_r, fps=feat_fps, size=96)
        vis_f = (vis_4ch_u8.astype(np.float32) / 255.0).astype(np.float16)
        kin_f = np.concatenate([kin_2d, kin_3d], axis=1).astype(np.float32)

        ip, cp, ic, cc = predict_seq_both_modes(deploy_model, vis_f, kin_f, window, device)
        tr_w = find_double_knock_triggers(
            ip, cp, feat_fps,
            smooth_frames=calib_win["smooth_frames"],
            imp_thr=calib_win["imp_thr"],
            double_thr=calib_win["double_thr"],
        )
        tr_s = find_causal_streaming_triggers(
            ic, cc, feat_fps,
            smooth_frames=calib_str["smooth_frames"],
            imp_thr=calib_str["imp_thr"],
            double_thr=calib_str["double_thr"],
        )
        dur_s = len(f240_r) / max(feat_fps, 1.0)
        is_pos = bool(cl["is_positive_double_knock"])
        if is_pos:
            pos_hit_win = pos_hit_win or (len(tr_w) > 0)
            pos_hit_str = pos_hit_str or (len(tr_s) > 0)
        else:
            neg_min += dur_s / 60.0
            neg_fp_win += len(tr_w)
            neg_fp_str += len(tr_s)
        clip_rows.append({
            "id": cl["id"], "source": cl["source"], "task": cl["task"],
            "is_positive": is_pos, "duration_s": round(dur_s, 2),
            "triggers_win": len(tr_w), "triggers_stream": len(tr_s),
        })

    return {
        "num_clips": len(clips),
        "positive_knock_d05_win": pos_hit_win,
        "positive_knock_d05_stream": pos_hit_str,
        "negative_total_minutes": round(neg_min, 2),
        "negative_ungated_fp_win": neg_fp_win,
        "negative_ungated_fa_per_min_win": round(neg_fp_win / max(neg_min, 1e-3), 3),
        "negative_ungated_fp_stream": neg_fp_str,
        "negative_ungated_fa_per_min_stream": round(neg_fp_str / max(neg_min, 1e-3), 3),
        "clips": clip_rows,
    }


def run_breakthrough2(epochs: int = 22, window: int = 48) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== BREAKTHROUGH #2: DoubleKnock3D-PhysNet (3D Metric Desk-Contact Physics vs 2D Baseline) ===", flush=True)

    extractor = BatchHoverGeoNet3DExtractor(device=str(device))
    print(f"  Using 3D Geometry Backbone: {extractor.ckpt_path.name}", flush=True)
    t0 = time.perf_counter()
    data_3d = build_3d_dataset(extractor, window=window, stride=4, size=96, use_3d=True)
    take_ids = sorted(data_3d.keys())
    print(f"  Dataset ready for {len(take_ids)} real takes ({time.perf_counter()-t0:.1f}s)", flush=True)

    fold_reports = []
    for f_idx, val_tid in enumerate(take_ids):
        train_sub = {tid: d for tid, d in data_3d.items() if tid != val_tid}
        val_d = data_3d[val_tid]

        model = train_one_fold_model(
            train_sub, forbidden_val_take=val_tid, in_ch=4, kin_dim=16,
            epochs=epochs, device=device, seed=200 + f_idx,
        )
        # 严格仅在 5 条训练 take 上标定阈值（绝不触碰留出 take！）
        preds_train = {
            tid: predict_seq_both_modes(model, d["vis_full"], d["kin_full"], window, device)
            for tid, d in train_sub.items()
        }
        calib_win, calib_str = calibrate_both_on_train_only(preds_train, train_sub)

        # 在留出 take 上零泄漏测试！
        ip_v, cp_v, ic_v, cc_v = predict_seq_both_modes(model, val_d["vis_full"], val_d["kin_full"], window, device)

        # 窗口级 AUROC / F1
        p_chunks = []
        with torch.no_grad():
            for si in range(0, len(val_d["x_vis"]), 64):
                xv = torch.from_numpy(val_d["x_vis"][si : si + 64]).float().div_(255.0).to(device)
                xk = torch.from_numpy(val_d["x_kin"][si : si + 64]).float().to(device)
                _, lc = model(xv, xk)
                p_chunks.append(torch.softmax(lc, dim=-1).cpu().numpy())
        p_win = np.concatenate(p_chunks, axis=0)
        y_cnt = val_d["y_cnt"]
        y_k = (y_cnt > 0).astype(np.int32)
        p_k = 1.0 - p_win[:, 0]
        win_auroc = float(roc_auc_score(y_k, p_k)) if len(np.unique(y_k)) > 1 else 0.0
        win_f1 = float(f1_score(y_k, (p_k >= 0.5).astype(np.int32)))
        y_d = (y_cnt >= 2).astype(np.int32)
        p_d = p_win[:, 2] + p_win[:, 3]
        dbl_auroc = float(roc_auc_score(y_d, p_d)) if len(np.unique(y_d)) > 1 else 0.0

        # 1) 滑窗事件级（对齐 metrics.json）
        tr_w = find_double_knock_triggers(
            ip_v, cp_v, val_d["fps"],
            smooth_frames=calib_win["smooth_frames"], imp_thr=calib_win["imp_thr"], double_thr=calib_win["double_thr"],
        )
        ev_w_ung = eval_triggers_on_take(tr_w, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=False)
        ev_w_gat = eval_triggers_on_take(tr_w, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=True)

        # 2) 严格因果逐帧流式（对齐 stream_eval.json）
        tr_s = find_causal_streaming_triggers(
            ic_v, cc_v, val_d["fps"],
            smooth_frames=calib_str["smooth_frames"], imp_thr=calib_str["imp_thr"], double_thr=calib_str["double_thr"],
        )
        ev_s_ung = eval_triggers_on_take(tr_s, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=False)

        fold_reports.append({
            "val_take": val_tid,
            "calib_win_train_only": calib_win,
            "calib_stream_train_only": calib_str,
            "window_knock_auroc": round(win_auroc, 4),
            "window_knock_f1": round(win_f1, 4),
            "window_double_knock_auroc": round(dbl_auroc, 4),
            "ungated_event": ev_w_ung,
            "gated_event": ev_w_gat,
            "streaming_ungated_event": ev_s_ung,
        })
        print(
            f"  [3D Fold {val_tid}] AUROC={win_auroc:.4f} DblAUC={dbl_auroc:.4f} | "
            f"WinUngated={ev_w_ung['hit']}/{ev_w_ung['total_gt']} F1={ev_w_ung['f1']:.4f} FP={ev_w_ung['fp']} ({ev_w_ung['fa_per_min']:.2f}/m) | "
            f"Stream={ev_s_ung['hit']}/{ev_s_ung['total_gt']} F1={ev_s_ung['f1']:.4f} FP={ev_s_ung['fp']} ({ev_s_ung['fa_per_min']:.2f}/m)",
            flush=True,
        )
        del model

    # 汇总 6 折留一指标并与已发布的 2D 基线对比
    n_f = len(fold_reports)
    sum_3d = {
        "mean_window_knock_auroc": round(sum(r["window_knock_auroc"] for r in fold_reports) / n_f, 4),
        "mean_window_knock_f1": round(sum(r["window_knock_f1"] for r in fold_reports) / n_f, 4),
        "mean_double_knock_auroc": round(sum(r["window_double_knock_auroc"] for r in fold_reports) / n_f, 4),
        "ungated_total_hit": sum(r["ungated_event"]["hit"] for r in fold_reports),
        "ungated_total_gt": sum(r["ungated_event"]["total_gt"] for r in fold_reports),
        "ungated_total_fp": sum(r["ungated_event"]["fp"] for r in fold_reports),
        "ungated_mean_f1": round(sum(r["ungated_event"]["f1"] for r in fold_reports) / n_f, 4),
        "ungated_mean_fa_per_min": round(sum(r["ungated_event"]["fa_per_min"] for r in fold_reports) / n_f, 3),
        "gated_total_hit": sum(r["gated_event"]["hit"] for r in fold_reports),
        "gated_mean_f1": round(sum(r["gated_event"]["f1"] for r in fold_reports) / n_f, 4),
        "streaming_total_hit": sum(r["streaming_ungated_event"]["hit"] for r in fold_reports),
        "streaming_total_gt": sum(r["streaming_ungated_event"]["total_gt"] for r in fold_reports),
        "streaming_total_fp": sum(r["streaming_ungated_event"]["fp"] for r in fold_reports),
        "streaming_mean_f1": round(sum(r["streaming_ungated_event"]["f1"] for r in fold_reports) / n_f, 4),
        "streaming_mean_fa_per_min": round(sum(r["streaming_ungated_event"]["fa_per_min"] for r in fold_reports) / n_f, 3),
    }

    # 训练全量 6-take 部署模型并在 26 条外部跨数据集压力测试集上验证零误报与 knock_d0.5 召回！
    print("\n  Training 6-take deploy DoubleKnock3D-PhysNet & running 26-clip cross-dataset stress test...", flush=True)
    deploy_model = train_one_fold_model(
        data_3d, forbidden_val_take=None, in_ch=4, kin_dim=16, epochs=epochs, device=device, seed=777,
    )
    preds_all = {
        tid: predict_seq_both_modes(deploy_model, d["vis_full"], d["kin_full"], window, device)
        for tid, d in data_3d.items()
    }
    calib_all_win, calib_all_str = calibrate_both_on_train_only(preds_all, data_3d)
    ckpt_deploy = OUT_DIR / "double_knock_3d_deploy.pt"
    torch.save(
        {
            "arch": "DoubleKnock3DPhysNet", "in_ch": 4, "kin_dim": 16, "hidden_dim": 64, "window": window,
            "calib_win": calib_all_win, "calib_stream": calib_all_str,
            "geo_backbone": str(extractor.ckpt_path.name),
            "state_dict": {k: v.detach().cpu() for k, v in deploy_model.state_dict().items()},
        },
        ckpt_deploy,
    )
    ext_res = eval_external_stress_3d(deploy_model, calib_all_win, calib_all_str, extractor, window, device)

    # 读取历史 2D 基线指标用于客观同台对比
    base_2d = json.loads(BASELINE_2D_METRICS.read_text(encoding="utf-8")) if BASELINE_2D_METRICS.exists() else {}
    base_2d_str = json.loads(BASELINE_2D_STREAM.read_text(encoding="utf-8")) if BASELINE_2D_STREAM.exists() else {}

    report = {
        "model": "DoubleKnock3D-PhysNet (4ch 3D-Desk-Gated Vision + 16D 2D/3D Physics + FiLM Gate)",
        "geo_backbone": str(extractor.ckpt_path.name),
        "ckpt_deploy": str(ckpt_deploy),
        "ckpt_kb": round(ckpt_deploy.stat().st_size / 1024.0, 1),
        "summary_3d_honest_6fold_loo": sum_3d,
        "baseline_2d_reference": {
            "mean_window_knock_auroc": base_2d.get("mean_window_knock_auroc"),
            "mean_double_knock_auroc": base_2d.get("mean_double_knock_auroc"),
            "ungated_total_hit": base_2d.get("ungated_total_hit"),
            "ungated_total_gt": base_2d.get("ungated_total_gt"),
            "ungated_mean_f1": base_2d.get("ungated_mean_f1"),
            "ungated_mean_fa_per_min": base_2d.get("ungated_mean_fa_per_min"),
            "gated_mean_f1": base_2d.get("gated_mean_f1"),
            "streaming_total_hit": base_2d_str.get("folds", {}).get("summary", {}).get("hit"),
            "streaming_total_fp": base_2d_str.get("folds", {}).get("summary", {}).get("fp"),
            "streaming_mean_f1": base_2d_str.get("folds", {}).get("summary", {}).get("mean_f1"),
            "streaming_fa_per_min": base_2d_str.get("folds", {}).get("summary", {}).get("fa_per_min"),
            "external_pos_knock_d05": base_2d.get("external_stress_test", {}).get("positive_session_knock_d0.5_detected"),
        },
        "external_stress_test_3d": ext_res,
        "folds": fold_reports,
    }
    (OUT_DIR / "double_knock_3d_loo_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== BREAKTHROUGH #2 SUMMARY (100% HONEST 6-FOLD LOO, ZERO LEAKAGE) ===", flush=True)
    print(f"  2D Baseline -> AUROC={report['baseline_2d_reference']['mean_window_knock_auroc']} | "
          f"WinUngated={report['baseline_2d_reference']['ungated_total_hit']}/89 F1={report['baseline_2d_reference']['ungated_mean_f1']} FA/m={report['baseline_2d_reference']['ungated_mean_fa_per_min']} | "
          f"Stream={report['baseline_2d_reference']['streaming_total_hit']}/89 F1={report['baseline_2d_reference']['streaming_mean_f1']} FP={report['baseline_2d_reference']['streaming_total_fp']}", flush=True)
    print(f"  3D PhysNet  -> AUROC={sum_3d['mean_window_knock_auroc']} | "
          f"WinUngated={sum_3d['ungated_total_hit']}/89 F1={sum_3d['ungated_mean_f1']} FA/m={sum_3d['ungated_mean_fa_per_min']} (FP={sum_3d['ungated_total_fp']}) | "
          f"Stream={sum_3d['streaming_total_hit']}/89 F1={sum_3d['streaming_mean_f1']} FP={sum_3d['streaming_total_fp']} ({sum_3d['streaming_mean_fa_per_min']}/m)", flush=True)
    print(f"  External 26-Clip Stress Test -> knock_d0.5_win={ext_res['positive_knock_d05_win']}, knock_d0.5_stream={ext_res['positive_knock_d05_stream']}, "
          f"Neg FP (win/stream)={ext_res['negative_ungated_fp_win']}/{ext_res['negative_ungated_fp_stream']} over {ext_res['negative_total_minutes']} min", flush=True)
    return report


if __name__ == "__main__":
    run_breakthrough2(epochs=22, window=48)
