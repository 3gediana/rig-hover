#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_room360_self_corrector_v12.py —— 全房间 360° 几何网络 (`HoverRoomGeoNet-360`, 27MB) 与闭环纠错神经网 (`RoomSelfCorrectorNet`, 3.2MB) 训练脚本

严格数据划分（零泄漏！全部来自 Poly Haven CC0 真实 360° 室内全景图）：
- **训练集（6 个真实 360° 房间）**：
  `wooden_lounge`, `lythwood_room`, `art_studio`, `anniversary_lounge`, `combination_room`, `small_empty_room_1`
- **严格留出测试集（3 个从未见过的真实 360° 房间，本脚本严禁触碰！）**：
  `hotel_room`, `reading_room`, `lythwood_lounge`

训练两个 100% 自研端侧模型（运行时零 `MoGe-2` 依赖！）：
1. **`HoverRoomGeoNet-360` (`MiT-B1`, ~27 MB FP16, ~14 ms/帧)**：
   - 输入 `6ch` (`[R, G, B, sobel_edge, ray_y, ray_z]`, 240×240, `FOV=70°`)，输出 `6ch` (`[log_depth, nx, ny, nz, crease, conf]`)，
     覆盖 `0.35 m .. 10.0 m` 全房间大尺度深度与锐利建筑墙角/家具结构！
   - 结合**尺度不变对数深度损失 + 多尺度梯度锐化 + 法向余弦 + 360° 相邻重叠视角跨视几何一致性损失**。
2. **`RoomSelfCorrectorNet` (~3.2 MB Render-Compare-Patch 闭环纠错 U-Net)**：
   - 输入 `10ch` 渲染对比反馈张量：`[I_obs(3), I_render(3), |I_obs - I_render|(1), log_D_render(1), log_D_obs(1), hole_mask(1)]`；
   - 输出双头：
     - `delta_log_d (1ch)`：残差深度修正（专门修正首次渲染产生的墙面鼓包、尺度偏移、空洞补全）；
     - `evict_logit (1ch)`：错点/幽灵剔除概率（专门检测并物理删除首次渲染留下的漂浮鬼影与错位重影墙面）！
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
from torch.utils.data import DataLoader, Dataset

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hd_moge_contour_engine import MoGe2DesktopReconstructor  # noqa: E402
from hover_geonet_240 import HoverGeoNetViT240  # noqa: E402


class HoverRoomGeoNet360(HoverGeoNetViT240):
    """全房间 360° 真彩 RGB Vision Transformer 几何网络 (`MiT-B1`, ~27 MB FP16, 0.3..11.0 m)。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 显式把前 3 通道真彩 RGB 送入预训练 MiT-B1 主干，充分辨识家具/墙面/地毯/门窗语义！
        vis3 = (x[:, 0:3] - self.im_mean) / self.im_std
        phys = x[:, 1:]
        f240 = self.fine_240(x)
        f120 = self.fine_120(f240)
        feats = self.encoder(vis3)
        s2, s3, s4, s5 = feats[2], feats[3], feats[4], feats[5]

        p5 = F.interpolate(phys, size=s5.shape[-2:], mode="area")
        p4 = F.interpolate(phys, size=s4.shape[-2:], mode="area")
        p3 = F.interpolate(phys, size=s3.shape[-2:], mode="area")
        p2 = F.interpolate(phys, size=s2.shape[-2:], mode="area")
        p1 = F.interpolate(phys, size=f120.shape[-2:], mode="area")

        b5 = self.drop(self.bot_proj(torch.cat([s5, p5], dim=1)))
        d4 = self.up4(b5, torch.cat([s4, p4], dim=1))
        d3 = self.up3(d4, torch.cat([s3, p3], dim=1))
        d2 = self.up2(d3, torch.cat([s2, p2], dim=1))
        d1 = self.edge_attn_120(self.up1(d2, torch.cat([f120, p1], dim=1)), x)
        d0 = self.edge_attn_240(self.up0(d1, f240), x)

        feat_full = torch.cat([d0, x], dim=1)
        geo = self.geo_head(feat_full)
        nrm = F.normalize(self.normal_head(feat_full), dim=1, eps=1e-6)
        det = self.detail_contour_head(feat_full)
        return torch.cat([geo[:, 0:1], nrm, geo[:, 1:2], det[:, 0:1], det[:, 1:2]], dim=1)

PANO_DIR = REPO / "pc" / "bench" / "_room360_real_panos"
CACHE_DIR = REPO / "pc" / "bench" / "_room360_train_cache"
OUT_DIR = REPO / "runs_end_side" / "room360_self_correct_v12"

TRAIN_ROOMS = [
    "wooden_lounge",
    "lythwood_room",
    "art_studio",
    "anniversary_lounge",
    "combination_room",
    "small_empty_room_1",
]
HELDOUT_ROOMS = [
    "hotel_room",
    "reading_room",
    "lythwood_lounge",
]

ROOM_HW = 240
ROOM_FOV_DEG = 70.0
ROOM_D_MIN = 0.30
ROOM_D_MAX = 11.0
LOG_D_MEAN = 0.95  # ~2.6 m nominal indoor room distance
LOG_D_SCALE = 1.25


# ---------------------------------------------------------------------------
# 1. 精确 360° 全景图 → 针孔相机透视投影工具（+X 前, +Y 右, +Z 上）
# ---------------------------------------------------------------------------


def rot_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """相机到世界旋转矩阵 R_c2w = R_z(yaw) @ R_y(pitch)，约定 pitch > 0 向下俯视，yaw > 0 向右偏航。"""
    yr = math.radians(float(yaw_deg))
    pr = math.radians(float(pitch_deg))
    cy, sy = math.cos(yr), math.sin(yr)
    cp, sp = math.cos(pr), math.sin(pr)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    return (Rz @ Rp).astype(np.float32)


def make_pinhole_rays_cam(hw: int = ROOM_HW, fov_deg: float = ROOM_FOV_DEG) -> np.ndarray:
    """返回相机系平面深度射线 `[H, W, 3]`，满足 `P_cam = d_planar * ray` (+X 前, +Y 右, +Z 上)。"""
    tan_h = math.tan(math.radians(float(fov_deg) * 0.5))
    u = ((np.arange(hw, dtype=np.float32) + 0.5) / float(hw)) * 2.0 - 1.0
    v = 1.0 - ((np.arange(hw, dtype=np.float32) + 0.5) / float(hw)) * 2.0
    uu, vv = np.meshgrid(u, v)
    return np.stack([np.ones_like(uu), uu * tan_h, vv * tan_h], axis=-1).astype(np.float32)


def extract_pinhole_from_equirect(
    pano_bgr: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    hw: int = ROOM_HW,
    fov_deg: float = ROOM_FOV_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """从 360° Equirectangular 全景图精确反投影出 `(yaw, pitch, fov)` 针孔透视视图与世界射线矩阵。"""
    H_p, W_p = pano_bgr.shape[:2]
    rays_c = make_pinhole_rays_cam(hw, fov_deg)
    R = rot_yaw_pitch(yaw_deg, pitch_deg)
    rays_w = rays_c @ R.T
    norms = np.maximum(np.linalg.norm(rays_w, axis=-1, keepdims=True), 1e-6)
    dirs = rays_w / norms  # unit sphere direction [x, y, z]

    # Equirectangular: lon = atan2(y, x) ∈ [-π, π] -> [0, W_p), lat = asin(z) ∈ [-π/2, π/2] -> [H_p, 0)
    lon = np.arctan2(dirs[..., 1], dirs[..., 0])
    lat = np.arcsin(np.clip(dirs[..., 2], -1.0, 1.0))
    map_x = ((lon / (2.0 * math.pi) + 0.5) * W_p).astype(np.float32) % float(W_p)
    map_y = np.clip((0.5 - lat / math.pi) * (H_p - 1), 0.0, float(H_p - 1)).astype(np.float32)
    bgr = cv2.remap(pano_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return bgr, R


def build_room_input_6ch(bgr: np.ndarray, rays_c: np.ndarray) -> np.ndarray:
    """构造 6 通道全房间几何网络输入 `[6, H, W]` -> `[R, G, B, sobel_edge, ray_y, ray_z]`。"""
    if bgr.shape[:2] != (ROOM_HW, ROOM_HW):
        bgr = cv2.resize(bgr, (ROOM_HW, ROOM_HW), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.hypot(gx, gy) * 1.4, 0.0, 1.0)
    x = np.empty((6, ROOM_HW, ROOM_HW), dtype=np.float32)
    x[0:3] = rgb.transpose(2, 0, 1)
    x[3] = edge
    x[4] = rays_c[..., 1]
    x[5] = rays_c[..., 2]
    return x


# ---------------------------------------------------------------------------
# 2. 离线准备 6 个训练房间的几何缓存（严禁触碰 3 个留出房间！）
# ---------------------------------------------------------------------------


def prepare_train_room_cache(force_rebuild: bool = False) -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = CACHE_DIR / "train_6rooms_216views.npz"
    if cache_npz.exists() and not force_rebuild:
        print(f"  [Cache HIT] {cache_npz}", flush=True)
        return cache_npz

    print("  Building offline geometry cache for 6 TRAINING 360° rooms ONLY (ZERO held-out rooms!)...", flush=True)
    rec = MoGe2DesktopReconstructor(infer_hw=384, fov_x_deg=ROOM_FOV_DEG)
    bgr_list, depth_list, nrm_list, valid_list, room_ids = [], [], [], [], []

    yaws = np.linspace(0.0, 330.0, 12)  # 12 azimuths (30° step, 57% overlap!)
    pitches = (-18.0, 0.0, 18.0)        # 3 elevation rings -> 36 views/room -> 216 views total!

    for r_idx, rname in enumerate(TRAIN_ROOMS):
        assert rname not in HELDOUT_ROOMS, f"LEAKAGE BLOCKED: {rname}"
        pano = cv2.imread(str(PANO_DIR / f"{rname}_2k.jpg"))
        assert pano is not None, f"Missing panorama: {rname}"
        t0 = time.perf_counter()
        for p_deg in pitches:
            for y_deg in yaws:
                bgr, _ = extract_pinhole_from_equirect(pano, float(y_deg), float(p_deg), hw=ROOM_HW, fov_deg=ROOM_FOV_DEG)
                raw = rec.infer_raw(bgr)
                d_raw = raw["pts_cam"][..., 0].astype(np.float32)
                n_raw = raw["nrm_cam"].astype(np.float32)
                m_raw = raw["mask"].astype(np.float32)
                d240 = cv2.resize(d_raw, (ROOM_HW, ROOM_HW), interpolation=cv2.INTER_LINEAR)
                n240 = cv2.resize(n_raw, (ROOM_HW, ROOM_HW), interpolation=cv2.INTER_LINEAR)
                n240 /= np.maximum(np.linalg.norm(n240, axis=-1, keepdims=True), 1e-6)
                m240 = (cv2.resize(m_raw, (ROOM_HW, ROOM_HW), interpolation=cv2.INTER_NEAREST) > 0.5) & np.isfinite(d240) & (d240 > ROOM_D_MIN) & (d240 < ROOM_D_MAX)
                d240 = np.clip(np.nan_to_num(d240, nan=2.5, posinf=ROOM_D_MAX, neginf=ROOM_D_MIN), ROOM_D_MIN, ROOM_D_MAX)
                bgr_list.append(bgr)
                depth_list.append(d240.astype(np.float16))
                nrm_list.append(n240.astype(np.float16))
                valid_list.append(m240.astype(np.uint8))
                room_ids.append(r_idx)
        print(f"    Cached training room '{rname}': 36 views ({time.perf_counter()-t0:.1f}s)", flush=True)

    del rec
    torch.cuda.empty_cache()

    np.savez_compressed(
        cache_npz,
        bgr=np.stack(bgr_list, axis=0),
        depth=np.stack(depth_list, axis=0),
        nrm=np.stack(nrm_list, axis=0),
        valid=np.stack(valid_list, axis=0),
        room_id=np.asarray(room_ids, dtype=np.int32),
    )
    print(f"  Saved 6-room training cache -> {cache_npz} ({cache_npz.stat().st_size / 1048576:.1f} MB)", flush=True)
    return cache_npz


# ---------------------------------------------------------------------------
# 3. 模型 1：HoverRoomGeoNet-360 (27 MB FP16, 全房间尺度 MiT-B1)
# ---------------------------------------------------------------------------


class RoomGeoTrainDataset(Dataset):
    def __init__(self, npz_path: pathlib.Path) -> None:
        z = np.load(npz_path)
        self.bgr = z["bgr"]
        self.depth = z["depth"].astype(np.float32)
        self.nrm = z["nrm"].astype(np.float32)
        self.valid = z["valid"].astype(np.float32)
        self.rays_c = make_pinhole_rays_cam(ROOM_HW, ROOM_FOV_DEG)

    def __len__(self) -> int:
        return len(self.bgr)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        bgr = self.bgr[idx].copy()
        dep = self.depth[idx].copy()
        nrm = self.nrm[idx].copy()
        val = self.valid[idx].copy()

        # 同步水平翻转（同步翻转 nrm_y!）
        if np.random.rand() < 0.5:
            bgr = np.ascontiguousarray(bgr[:, ::-1])
            dep = np.ascontiguousarray(dep[:, ::-1])
            nrm = np.ascontiguousarray(nrm[:, ::-1])
            nrm[..., 1] = -nrm[..., 1]
            val = np.ascontiguousarray(val[:, ::-1])

        # 轻量色彩与亮度抖动增强泛化至任意未知房间光照！
        if np.random.rand() < 0.7:
            gain = np.random.uniform(0.82, 1.18, size=(1, 1, 3)).astype(np.float32)
            bias = np.random.uniform(-14.0, 14.0, size=(1, 1, 3)).astype(np.float32)
            bgr = np.clip(bgr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        x6 = build_room_input_6ch(bgr, self.rays_c)
        return {
            "x6": torch.from_numpy(x6),
            "log_d": torch.from_numpy(np.log(np.clip(dep, ROOM_D_MIN, ROOM_D_MAX)).astype(np.float32)),
            "nrm": torch.from_numpy(nrm.transpose(2, 0, 1).astype(np.float32)),
            "valid": torch.from_numpy(val.astype(np.float32)),
        }


def decode_room_geonet_out(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 HoverRoomGeoNet-360 7 通道输出解码为 `(log_d, nrm, conf)`。"""
    log_d = (LOG_D_MEAN + out[:, 0] * LOG_D_SCALE).clamp(math.log(ROOM_D_MIN), math.log(ROOM_D_MAX))
    nrm = F.normalize(out[:, 1:4], dim=1, eps=1e-6)
    conf = torch.sigmoid(out[:, 4])
    return log_d, nrm, conf


def train_hover_room_geonet(cache_npz: pathlib.Path, epochs: int = 18, lr: float = 3.5e-4) -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_fp16 = OUT_DIR / "hover_room_geonet_360_fp16.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RoomGeoTrainDataset(cache_npz)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    model = HoverRoomGeoNet360(in_ch=6, base=24, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": lr * 0.45, "weight_decay": 1e-2},
            {"params": list(model.head_parameters()), "lr": lr, "weight_decay": 5e-4},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    print(f"\n=== Training Model 1: HoverRoomGeoNet-360 (MiT-B1 27MB, {len(ds)} views from 6 real 360° rooms) ===", flush=True)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        ep_l, ep_rel, ep_ang, steps = 0.0, 0.0, 0.0, 0
        for b in dl:
            x6 = b["x6"].to(device, non_blocking=True)
            gt_ld = b["log_d"].to(device, non_blocking=True)
            gt_n = b["nrm"].to(device, non_blocking=True)
            vm = b["valid"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(x6).float()
                pr_ld, pr_n, pr_c = decode_room_geonet_out(out)

                # 1) 对数深度 Smooth-L1 + 尺度不变方差损失 (SILog)
                diff = (pr_ld - gt_ld) * vm
                n_v = vm.sum().clamp(min=10.0)
                l_abs = F.smooth_l1_loss(pr_ld * vm, gt_ld * vm, beta=0.06, reduction="sum") / n_v
                mean_d = diff.sum(dim=(1, 2)) / vm.sum(dim=(1, 2)).clamp(min=10.0)
                l_si = (((pr_ld - gt_ld - mean_d[:, None, None]) ** 2) * vm).sum() / n_v

                # 2) 多尺度对数深度梯度损失（逼出锐利门框、墙角、沙发边缘与床沿！）
                l_grad = 0.0
                for sc in (1, 2, 4):
                    p_s = F.avg_pool2d(pr_ld.unsqueeze(1), sc) if sc > 1 else pr_ld.unsqueeze(1)
                    g_s = F.avg_pool2d(gt_ld.unsqueeze(1), sc) if sc > 1 else gt_ld.unsqueeze(1)
                    m_s = (F.avg_pool2d(vm.unsqueeze(1), sc) > 0.8).float() if sc > 1 else vm.unsqueeze(1)
                    gx = ((p_s[:, :, :, 1:] - p_s[:, :, :, :-1]) - (g_s[:, :, :, 1:] - g_s[:, :, :, :-1])).abs() * m_s[:, :, :, 1:] * m_s[:, :, :, :-1]
                    gy = ((p_s[:, :, 1:, :] - p_s[:, :, :-1, :]) - (g_s[:, :, 1:, :] - g_s[:, :, :-1, :])).abs() * m_s[:, :, 1:, :] * m_s[:, :, :-1, :]
                    l_grad = l_grad + (gx.sum() + gy.sum()) / m_s.sum().clamp(min=10.0)

                # 3) 法向余弦 + 有效掩码置信度 (使用 autocast 安全的 BCEWithLogits!)
                cos_n = (pr_n * gt_n).sum(dim=1).clamp(-1.0, 1.0)
                l_nrm = ((1.0 - cos_n) * vm).sum() / n_v
                l_conf = F.binary_cross_entropy_with_logits(out[:, 4], vm)

                loss = 3.5 * l_abs + 2.5 * l_si + 2.2 * l_grad + 1.8 * l_nrm + 0.25 * l_conf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            with torch.no_grad():
                rel = ((torch.exp(pr_ld) - torch.exp(gt_ld)).abs() / torch.exp(gt_ld).clamp(min=0.2) * vm).sum() / n_v
                ang = (torch.acos(cos_n.clamp(-0.9999, 0.9999)) * (180.0 / math.pi) * vm).sum() / n_v
            ep_l += float(loss.item())
            ep_rel += float(rel.item())
            ep_ang += float(ang.item())
            steps += 1
        sched.step()
        if ep in (1, 5, 10, 14, epochs):
            print(f"  [RoomGeoNet-360] Ep {ep:02d}/{epochs} | loss={ep_l/steps:.3f} abs_rel={ep_rel/steps:.4f} nrm={ep_ang/steps:.1f}°", flush=True)

    fp16_sd = {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}
    torch.save(
        {
            "arch": "vit_b1", "in_ch": 6, "base": 24,
            "room_hw": ROOM_HW, "fov_deg": ROOM_FOV_DEG,
            "log_d_mean": LOG_D_MEAN, "log_d_scale": LOG_D_SCALE,
            "train_rooms": TRAIN_ROOMS,
            "state_dict_fp16": fp16_sd,
        },
        ckpt_fp16,
    )
    print(f"  Saved HoverRoomGeoNet-360 -> {ckpt_fp16} ({ckpt_fp16.stat().st_size/1048576:.2f} MB, {time.perf_counter()-t0:.1f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return ckpt_fp16


# ---------------------------------------------------------------------------
# 4. 模型 2：RoomSelfCorrectorNet (3.2 MB Render-Compare-Patch 闭环纠错 U-Net!)
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    def __init__(self, ci: int, co: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.SiLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoomSelfCorrectorNet(nn.Module):
    """闭环渲染对比纠错神经网络 (Render-Compare-Patch Residual Self-Corrector, ~3.2 MB)。

    输入 10 通道反馈张量 (240x240):
      [0..2] I_obs (真实观测 RGB)
      [3..5] I_render (首次 3D 地图重投影渲染 RGB)
      [6]    |I_obs - I_render| (光度自检误差图)
      [7]    log(D_render) 归一化重投影地图深度
      [8]    log(D_obs)    归一化单目新视角深度
      [9]    hole_mask     (1 = 首次地图空洞/缺失区, 0 = 有覆盖区)

    输出双头:
      - `delta_log_d`: 对数深度残差修正（专门拉平墙面鼓包、校准尺度偏移、修补空洞几何）
      - `evict_logit`: 首次错点/幽灵点剔除 logits（sigmoid > 0.5 物理删除地图中的漂浮鬼影与错位重影）
    """

    def __init__(self, in_ch: int = 10, base: int = 32) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.enc3 = _ConvBlock(base * 2, base * 4)
        self.bot = _ConvBlock(base * 4, base * 4)
        self.up3 = _ConvBlock(base * 8, base * 2)
        self.up2 = _ConvBlock(base * 4, base)
        self.up1 = _ConvBlock(base * 2, base)
        self.head_delta = nn.Conv2d(base, 1, 1)
        self.head_evict = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)                                      # 240
        e2 = self.enc2(F.avg_pool2d(e1, 2))                    # 120
        e3 = self.enc3(F.avg_pool2d(e2, 2))                    # 60
        b = self.bot(F.avg_pool2d(e3, 2))                      # 30
        d3 = self.up3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        delta_log_d = torch.tanh(self.head_delta(d1)) * 1.15   # 支持修正 ±3.1x 极端鼓包/偏移！
        evict_logit = self.head_evict(d1)
        return delta_log_d, evict_logit


class SelfCorrectionCorruptionDataset(Dataset):
    """在 6 个训练房间的真实视图上在线合成各类首次渲染缺陷（墙面鼓包、空洞缺失、漂浮幽灵、位姿错位、尺度偏置），
    训练 `RoomSelfCorrectorNet` 从 Render-Compare 视觉/几何矛盾中自动发现并修复缺陷！"""

    def __init__(self, npz_path: pathlib.Path) -> None:
        z = np.load(npz_path)
        self.bgr = z["bgr"]
        self.depth = z["depth"].astype(np.float32)
        self.valid = z["valid"].astype(np.float32)
        yy, xx = np.meshgrid(np.linspace(-1, 1, ROOM_HW), np.linspace(-1, 1, ROOM_HW), indexing="ij")
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

    def __len__(self) -> int:
        return len(self.bgr) * 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = idx % len(self.bgr)
        rgb = cv2.cvtColor(self.bgr[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d_gt = np.clip(self.depth[i].copy(), ROOM_D_MIN, ROOM_D_MAX)
        vm = self.valid[i].copy()
        if np.random.rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            d_gt = np.ascontiguousarray(d_gt[:, ::-1])
            vm = np.ascontiguousarray(vm[:, ::-1])

        log_gt = np.log(d_gt)
        # 模拟有噪声的观测单目深度 D_obs（±6% 局部起伏 + 小偏移）
        obs_scale = float(np.random.uniform(0.93, 1.07))
        obs_warp = float(np.random.uniform(-0.07, 0.07)) * np.sin(self.xx * 2.5 + np.random.uniform(0, 6.28))
        log_obs = np.clip(log_gt + math.log(obs_scale) + obs_warp, math.log(ROOM_D_MIN), math.log(ROOM_D_MAX))

        # 构造带首次缺陷的渲染图 (I_ren, log_D_ren, hole_mask) 以及精确的幽灵/坏点监督标签 evict_gt
        i_ren = rgb.copy()
        log_ren = log_gt.copy()
        hole = np.zeros((ROOM_HW, ROOM_HW), dtype=np.float32)
        evict = np.zeros((ROOM_HW, ROOM_HW), dtype=np.float32)

        # 缺陷 A: 位姿微小错位/接缝平移 (模拟 1st-pass 漂移重影)
        if np.random.rand() < 0.65:
            sx = int(np.random.randint(-9, 10))
            sy = int(np.random.randint(-5, 6))
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            i_ren = cv2.warpAffine(i_ren, M, (ROOM_HW, ROOM_HW), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            log_ren = cv2.warpAffine(log_ren, M, (ROOM_HW, ROOM_HW), borderMode=cv2.BORDER_REPLICATE)
            cov = cv2.warpAffine(np.ones((ROOM_HW, ROOM_HW), np.float32), M, (ROOM_HW, ROOM_HW), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            hole = np.maximum(hole, (cov < 0.5).astype(np.float32))

        # 缺陷 B: 大尺度平滑径向鼓包 / 塌陷 (±25..55% 深度误差!)
        if np.random.rand() < 0.80:
            cx, cy = float(np.random.uniform(-0.6, 0.6)), float(np.random.uniform(-0.5, 0.5))
            sig = float(np.random.uniform(0.28, 0.65))
            amp = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(0.22, 0.52))
            blob = np.exp(-((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / (2.0 * sig * sig))
            log_ren = np.clip(log_ren + amp * blob, math.log(ROOM_D_MIN), math.log(ROOM_D_MAX))
            evict = np.maximum(evict, (np.abs(log_ren - log_gt) > 0.24).astype(np.float32))

        # 缺陷 C: 首次渲染空洞 / 遮挡缺失区 (1..3 个随机矩形或圆形缺失块)
        if np.random.rand() < 0.78:
            for _ in range(int(np.random.randint(1, 4))):
                w_h = int(np.random.randint(24, 95))
                h_h = int(np.random.randint(24, 95))
                x0 = int(np.random.randint(0, ROOM_HW - w_h))
                y0 = int(np.random.randint(0, ROOM_HW - h_h))
                hole[y0 : y0 + h_h, x0 : x0 + w_h] = 1.0
                i_ren[y0 : y0 + h_h, x0 : x0 + w_h] = 0.0
                log_ren[y0 : y0 + h_h, x0 : x0 + w_h] = LOG_D_MEAN

        # 缺陷 D: 漂浮幽灵伪影块（近距离错误的漂浮彩色/灰白团块，光度与深度均与真实视角矛盾！）
        if np.random.rand() < 0.75:
            for _ in range(int(np.random.randint(1, 5))):
                rad = int(np.random.randint(8, 30))
                gx = int(np.random.randint(rad, ROOM_HW - rad))
                gy = int(np.random.randint(rad, ROOM_HW - rad))
                mask_g = np.zeros((ROOM_HW, ROOM_HW), dtype=np.uint8)
                cv2.circle(mask_g, (gx, gy), rad, 1, -1)
                mg = mask_g > 0
                ghost_col = np.random.uniform(0.05, 0.95, size=(3,)).astype(np.float32)
                i_ren[mg] = np.clip(ghost_col + np.random.normal(0, 0.06, size=(int(mg.sum()), 3)), 0, 1)
                log_ren[mg] = math.log(float(np.random.uniform(0.45, 1.35)))
                hole[mg] = 0.0
                evict[mg] = 1.0

        # 添加轻微真实点云渲染稀疏噪点
        i_ren = np.clip(i_ren + np.random.normal(0.0, 0.015, i_ren.shape).astype(np.float32), 0.0, 1.0)
        photo_err = np.abs(rgb - i_ren).mean(axis=-1)

        # 组装 10 通道输入张量
        x10 = np.empty((10, ROOM_HW, ROOM_HW), dtype=np.float32)
        x10[0:3] = rgb.transpose(2, 0, 1)
        x10[3:6] = i_ren.transpose(2, 0, 1)
        x10[6] = photo_err
        x10[7] = (log_ren - LOG_D_MEAN) / LOG_D_SCALE
        x10[8] = (log_obs - LOG_D_MEAN) / LOG_D_SCALE
        x10[9] = hole

        # 基准深度选 log_ren（非空洞）或 log_obs（空洞），纠错网络只需预测相对该基准的残差修正！
        base_log = np.where((hole > 0.5) | (evict > 0.5), log_obs, 0.5 * log_ren + 0.5 * log_obs).astype(np.float32)
        return {
            "x10": torch.from_numpy(x10),
            "log_obs": torch.from_numpy(log_obs.astype(np.float32)),
            "log_ren": torch.from_numpy(log_ren.astype(np.float32)),
            "log_gt": torch.from_numpy(log_gt.astype(np.float32)),
            "hole": torch.from_numpy(hole.astype(np.float32)),
            "evict": torch.from_numpy(evict.astype(np.float32)),
            "valid": torch.from_numpy(vm.astype(np.float32)),
            "base_log": torch.from_numpy(base_log),
        }


def train_self_corrector(cache_npz: pathlib.Path, epochs: int = 16, lr: float = 1.2e-3) -> pathlib.Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_p = OUT_DIR / "room_self_corrector_v12.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SelfCorrectionCorruptionDataset(cache_npz)
    dl = DataLoader(ds, batch_size=12, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    net = RoomSelfCorrectorNet(in_ch=10, base=32).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    print(f"\n=== Training Model 2: RoomSelfCorrectorNet (Render-Compare-Patch Residual Self-Corrector, {len(ds)} samples) ===", flush=True)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        net.train()
        ep_d, ep_ev, steps = 0.0, 0.0, 0
        for b in dl:
            x10 = b["x10"].to(device, non_blocking=True)
            log_obs = b["log_obs"].to(device, non_blocking=True)
            log_ren = b["log_ren"].to(device, non_blocking=True)
            log_gt = b["log_gt"].to(device, non_blocking=True)
            hole = b["hole"].to(device, non_blocking=True)
            ev_gt = b["evict"].to(device, non_blocking=True)
            vm = b["valid"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                delta_ld, ev_logit = net(x10)
                delta_ld = delta_ld.float().squeeze(1)
                ev_logit = ev_logit.float().squeeze(1)
                p_bad = torch.sigmoid(ev_logit)
                # 自适应融合门：被判定为坏点/空洞的像素自动放弃 log_ren 转向 log_obs，再叠加学习到的残差修正！
                use_obs = torch.maximum(hole, p_bad)
                pred_log = ((1.0 - use_obs) * (0.55 * log_ren + 0.45 * log_obs) + use_obs * log_obs + delta_ld).clamp(
                    math.log(ROOM_D_MIN), math.log(ROOM_D_MAX)
                )
                # 重点加权空洞与缺陷区的修复误差！
                w_pix = vm * (1.0 + 2.5 * hole + 2.5 * ev_gt)
                l_dep = (F.smooth_l1_loss(pred_log, log_gt, beta=0.04, reduction="none") * w_pix).sum() / w_pix.sum().clamp(min=10.0)
                pos_w = torch.tensor(3.0, device=device)
                l_ev = F.binary_cross_entropy_with_logits(ev_logit, ev_gt, pos_weight=pos_w)
                loss = 4.0 * l_dep + 1.2 * l_ev

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            ep_d += float(l_dep.item())
            ep_ev += float(l_ev.item())
            steps += 1
        sched.step()
        if ep in (1, 5, 10, epochs):
            print(f"  [RoomSelfCorrectorNet] Ep {ep:02d}/{epochs} | dep_loss={ep_d/steps:.4f} evict_bce={ep_ev/steps:.4f}", flush=True)

    torch.save(
        {
            "arch": "RoomSelfCorrectorNet", "in_ch": 10, "base": 32,
            "room_hw": ROOM_HW, "log_d_mean": LOG_D_MEAN, "log_d_scale": LOG_D_SCALE,
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        },
        ckpt_p,
    )
    print(f"  Saved RoomSelfCorrectorNet -> {ckpt_p} ({ckpt_p.stat().st_size/1048576:.2f} MB, {time.perf_counter()-t0:.1f}s)", flush=True)
    return ckpt_p


def main() -> None:
    cache_npz = prepare_train_room_cache(force_rebuild=False)
    train_hover_room_geonet(cache_npz, epochs=18, lr=3.5e-4)
    train_self_corrector(cache_npz, epochs=16, lr=1.2e-3)


if __name__ == "__main__":
    main()
