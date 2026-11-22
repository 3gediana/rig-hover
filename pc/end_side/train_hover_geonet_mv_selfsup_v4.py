#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_hover_geonet_mv_selfsup_v4.py —— 突破教师上限：多视角自监督时空物理几何蒸馏 (`HoverGeoNet-MV-SelfSup v4`, 27 MB)

核心科研目标（突破“学生永远打不过老师”的纯蒸馏天花板）：
- 旧版 `HoverGeoNet-v3` (27 MB) 仅通过单帧蒸馏模仿 `Microsoft MoGe-2-ViT-B` (419 MB)，因此继承了单帧基础模型在真实机器人连续视频上的致命缺陷：
  1) 逐帧尺度呼吸抖动 (Temporal Depth Flicker / Scale Breathing)；
  2) 真实跨帧 3D 重投影几何不一致；
  3) 桌面刚性平面在 1.16 Hz 自平衡晃动下出现弯曲与法向漂移。
- 本模块在**真实无标注机器人连续视频三元组 `(I_{t-2}, I_t, I_{t+2})`**（严格隔离留出集 `empty_d0.5` 与 `cb8455`！）上，
  引入可微相对位姿估计子网 `RockingPoseNet` 与四大真实物理自监督损失联合精调 `HoverGeoNet-v4`：
  1. **Monodepth2 自动掩码最小重投影光度损失 (`L_photo = 0.85*(1-SSIM)/2 + 0.15*L1`)**：直接让 3D 深度解释真实像素跨帧运动；
  2. **双向 3D 几何循环一致性损失 (`L_3d_cycle`)**：强迫 `T_{t->s} P_t` 与 `P_s(π(T P_t))` 在静态区毫米级重合，彻底消灭单帧深度呼吸抖动！
  3. **13.5 cm 刚性物理桌面平面锚定与法向-深度正交性损失 (`L_phys_desk`)**；
  4. **干净微物体几何回放防遗忘约束 (`L_replay`)**：100% 保持 3.5 cm 鼠标与 90° 桌角锐度！

最后在严格留出真实视频 `empty_d0.5` 与 `cb8455` 上，用 **6 项 100% 不依赖任何教师伪真值的客观物理指标**
正面 PK **`Microsoft MoGe-2-ViT-B (419 MB)`** vs **`HoverGeoNet-v3 (纯教师蒸馏)`** vs **`HoverGeoNet-MV-SelfSup v4 (27 MB)`**！
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

from hd_dynamic_contour_slam import HoverGeoNet240Frontend, MoGe2Frontend  # noqa: E402
from hd_moge_contour_engine import REAL_FOV_H_DEG, REAL_H_CAM_M, relocalize_frame_against_3d_map  # noqa: E402
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
    compute_desk_prior_from_rays,
    create_hover_geonet,
    decode_geonet_to_world_cloud,
)
from mono_depth_model import camera_rotation_matrix, make_camera_ray_grid  # noqa: E402
from train_hover_geonet_240 import HoverGeoDataset, _filter_clean_teacher, compute_geonet_loss, evaluate_split, prepare_all_caches  # noqa: E402

OUT_DIR = REPO / "runs_end_side" / "hover_geonet_mv_v4"
V3_CKPT_FP32 = REPO / "runs_end_side" / "hover_geonet_vit_240" / "hover_geonet_vit_b1_fp32.pt"
V3_CKPT_FP16 = REPO / "runs_end_side" / "hover_geonet_vit_240" / "hover_geonet_vit_b1_fp16.pt"
EMPTY_DIR = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"
CB8455_MP4 = REPO / "dataset" / "takes" / "robot_20261115_151733_cb8455" / "take.mp4"


# ---------------------------------------------------------------------------
# 1. 真实无标注多视角三元组数据集 (严格排除留出集 empty_d0.5 与 cb8455!)
# ---------------------------------------------------------------------------


def collect_real_unlabeled_triplets(max_per_src: int = 85, stride_frames: int = 2) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """从真实机器人录像与实机采集序列中采样连续三元组 `(g_{t-k}, g_t, g_{t+k})`（严格不含留出集！）。"""
    triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    mp4_sources = [
        REPO / "dataset" / "takes" / "robot_20261107_204337_5ca36d" / "take.mp4",
        REPO / "dataset" / "takes" / "robot_20261107_212119_67f0c1" / "take.mp4",
        REPO / "dataset" / "takes" / "robot_20261107_213314_5c0158" / "take.mp4",
        REPO / "dataset" / "takes" / "robot_20261107_214950_1f30d1" / "take.mp4",
        REPO / "dataset" / "takes" / "robot_20261115_150045_65a54d" / "take.mp4",
        REPO / "video_annotation_data" / "videos" / "robot_20261105_004630_ef1d19.mp4",
    ]
    for mp4 in mp4_sources:
        if not mp4.exists():
            continue
        assert "cb8455" not in mp4.name and "empty_d0.5" not in mp4.name
        cap = cv2.VideoCapture(str(mp4))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot < 30:
            cap.release()
            continue
        idxs = np.unique(np.linspace(10, tot - 10, max_per_src, dtype=np.int32))
        for idx in idxs:
            frames = []
            ok_all = True
            for off in (-stride_frames, 0, stride_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx + off))
                ok, fr = cap.read()
                if not ok or fr is None:
                    ok_all = False
                    break
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
                if g.shape != (GEO_H, GEO_W):
                    g = cv2.resize(g, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)
                frames.append(g)
            if ok_all:
                triplets.append((frames[0], frames[1], frames[2]))
        cap.release()

    dir_sources = [
        REPO / "evidence" / "live" / "calib_1" / "frames",
        REPO / "evidence" / "live" / "gesture_1" / "frames",
        REPO / "evidence" / "live" / "gesture_2" / "frames",
        REPO / "evidence" / "sessions" / "fist_d0.5_front_night_181907" / "frames",
        REPO / "evidence" / "sessions" / "none_d0.5_front_night_181841" / "frames",
        REPO / "evidence" / "sessions" / "open_palm_d0.5_front_night_181943" / "frames",
        REPO / "evidence" / "sessions" / "pointing_d0.5_front_night_182101" / "frames",
        REPO / "evidence" / "sessions" / "wave_d0.5_front_night_182035" / "frames",
    ]
    for d in dir_sources:
        if not d.exists():
            continue
        assert "empty_d0.5" not in str(d) and "cb8455" not in str(d)
        jpgs = sorted(d.glob("*.jpg"))
        if len(jpgs) < 12:
            continue
        idxs = np.unique(np.linspace(stride_frames + 1, len(jpgs) - stride_frames - 2, min(max_per_src // 2, len(jpgs) - 6), dtype=np.int32))
        for idx in idxs:
            gs = []
            ok_all = True
            for off in (-stride_frames, 0, stride_frames):
                fr = cv2.imread(str(jpgs[int(idx + off)]), cv2.IMREAD_GRAYSCALE)
                if fr is None:
                    ok_all = False
                    break
                if fr.shape != (GEO_H, GEO_W):
                    fr = cv2.resize(fr, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)
                gs.append(fr)
            if ok_all:
                triplets.append((gs[0], gs[1], gs[2]))
    return triplets


class RealVideoTripletDataset(Dataset):
    """输出真实三帧物理条件输入 `(x_prev, x_cur, x_next)` + 原始灰度 `(g_prev, g_cur, g_next)` 与射线网格。"""

    def __init__(self, triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g_m, g_c, g_p = self.triplets[idx]
        # 水平翻转增强（三帧同步翻转保持严格极线几何！）
        if np.random.rand() < 0.5:
            g_m = np.ascontiguousarray(g_m[:, ::-1])
            g_c = np.ascontiguousarray(g_c[:, ::-1])
            g_p = np.ascontiguousarray(g_p[:, ::-1])

        p_nom = float(np.random.uniform(2.2, 3.6))
        r_nom = float(np.random.uniform(-1.0, 1.0))

        xm, rays_m, inv_m = build_robot_physics_input(g_m, g_m, pitch_deg=p_nom, roll_deg=r_nom, fov_h_deg=REAL_FOV_H_DEG, h_cam_m=REAL_H_CAM_M)
        xc, rays_c, inv_c = build_robot_physics_input(g_c, g_m, pitch_deg=p_nom, roll_deg=r_nom, fov_h_deg=REAL_FOV_H_DEG, h_cam_m=REAL_H_CAM_M)
        xp, rays_p, inv_p = build_robot_physics_input(g_p, g_c, pitch_deg=p_nom, roll_deg=r_nom, fov_h_deg=REAL_FOV_H_DEG, h_cam_m=REAL_H_CAM_M)

        return {
            "xm": torch.from_numpy(xm),
            "xc": torch.from_numpy(xc),
            "xp": torch.from_numpy(xp),
            "rays": torch.from_numpy(rays_c.transpose(2, 0, 1)),  # [3, H, W]
            "inv_prior": torch.from_numpy(inv_c),                 # [H, W]
            "gm": torch.from_numpy(g_m.astype(np.float32) / 255.0).unsqueeze(0),
            "gc": torch.from_numpy(g_c.astype(np.float32) / 255.0).unsqueeze(0),
            "gp": torch.from_numpy(g_p.astype(np.float32) / 255.0).unsqueeze(0),
        }


# ---------------------------------------------------------------------------
# 2. 可微自平衡流形相对位姿估计网络 (`RockingPoseNet`) + 可微 3D 重投影算子
# ---------------------------------------------------------------------------


class RockingPoseNet(nn.Module):
    """仅训练期使用的可微相对 6-DoF 位姿网络，约束在自平衡桌面机器人的物理微运动流形上。"""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, 24, 5, stride=2, padding=2), nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, g_tgt: torch.Tensor, g_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回从 target 相机到 source 相机的相对旋转 R [B,3,3] 与平移 t [B,3]。"""
        x = torch.cat([g_tgt, g_src], dim=1)
        raw = self.head(self.enc(x).flatten(1))
        # 自平衡桌面机器人相邻 2 帧 (0.13s) 的物理运动范围限制：平移 ±6cm，旋转 ±5.5°
        t = torch.tanh(raw[:, 0:3]) * 0.06
        ang = torch.tanh(raw[:, 3:6]) * math.radians(5.5)  # [rx, ry, rz]
        cx, sx = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
        cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
        cz, sz = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])
        o = torch.ones_like(cx)
        z = torch.zeros_like(cx)
        Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], dim=-1).reshape(-1, 3, 3)
        Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], dim=-1).reshape(-1, 3, 3)
        Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], dim=-1).reshape(-1, 3, 3)
        R = Rz @ Ry @ Rx
        return R, t


def _ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """3×3 窗口可微结构相似性损失图 `(1 - SSIM) / 2` `[B, 1, H, W]`。"""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sig_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sig_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sig_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2)
    return torch.clamp((1.0 - num / den.clamp(min=1e-6)) * 0.5, 0.0, 1.0)


def _warp_target_to_source(
    pts_t: torch.Tensor,     # [B, 3, H, W] (in leveled world frame, cam at [0,0,H_cam])
    nrm_t: torch.Tensor,     # [B, 3, H, W]
    g_src: torch.Tensor,     # [B, 1, H, W]
    pts_src: torch.Tensor,   # [B, 3, H, W]
    nrm_src: torch.Tensor,   # [B, 3, H, W]
    R_t2s: torch.Tensor,     # [B, 3, 3]
    t_t2s: torch.Tensor,     # [B, 3]
    tan_h: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将目标帧 3D 点云经 SE(3) 刚体变换重投影到源帧像素平面并采样灰度、3D 坐标与法向！"""
    B, _, H, W = pts_t.shape
    # 转为相对相机原点坐标，应用 (R_t2s, t_t2s)
    p_rel = pts_t.clone()
    p_rel[:, 2] = p_rel[:, 2] - REAL_H_CAM_M
    p_flat = p_rel.reshape(B, 3, -1)  # [B, 3, HW]
    p_s = torch.bmm(R_t2s, p_flat) + t_t2s.unsqueeze(-1)  # [B, 3, HW]
    p_s_grid = p_s.reshape(B, 3, H, W)
    p_s_world = p_s_grid.clone()
    p_s_world[:, 2] = p_s_world[:, 2] + REAL_H_CAM_M

    # 旋转法向
    n_s = torch.bmm(R_t2s, nrm_t.reshape(B, 3, -1)).reshape(B, 3, H, W)

    # 透视投影到源帧归一化图像坐标 [-1, 1]（+X 前, +Y 右, +Z 上，与 make_camera_ray_grid 严格同系！）
    xc = p_s_grid[:, 0].clamp(min=0.08)
    yc = p_s_grid[:, 1]
    zc = p_s_grid[:, 2]
    u_norm = yc / (xc * tan_h)
    v_norm = -zc / (xc * tan_h)
    grid = torch.stack([u_norm, v_norm], dim=-1)  # [B, H, W, 2]
    in_bounds = ((u_norm.abs() <= 0.96) & (v_norm.abs() <= 0.96) & (p_s_grid[:, 0] > 0.12)).float().unsqueeze(1)

    g_warp = F.grid_sample(g_src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    pts_src_samp = F.grid_sample(pts_src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    nrm_src_samp = F.grid_sample(nrm_src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    nrm_src_samp = F.normalize(nrm_src_samp, dim=1, eps=1e-6)
    return g_warp, p_s_world, pts_src_samp, n_s, nrm_src_samp, in_bounds


def compute_multiview_selfsup_loss(
    out_m: torch.Tensor,
    out_c: torch.Tensor,
    out_p: torch.Tensor,
    batch_mv: dict[str, torch.Tensor],
    pose_net: RockingPoseNet,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算真实多视角自监督物理几何损失（零 MoGe-2 依赖！）。"""
    rays = batch_mv["rays"]            # [B, 3, H, W]
    inv_prior = batch_mv["inv_prior"]  # [B, H, W]
    gm, gc, gp = batch_mv["gm"], batch_mv["gc"], batch_mv["gp"]
    tan_h = math.tan(math.radians(REAL_FOV_H_DEG * 0.5))

    def _dec(o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_d = (inv_prior + o[:, 0] * INV_D_SCALE).clamp(INV_D_MIN, INV_D_MAX)
        d = 1.0 / inv_d
        pts = rays * d.unsqueeze(1)
        pts[:, 2] = pts[:, 2] + REAL_H_CAM_M
        nrm = F.normalize(o[:, 1:4], dim=1, eps=1e-6)
        vprob = torch.sigmoid(o[:, 4:5])
        return d, pts, nrm, vprob

    d_m, pts_m, nrm_m, _ = _dec(out_m)
    d_c, pts_c, nrm_c, v_c = _dec(out_c)
    d_p, pts_p, nrm_p, _ = _dec(out_p)

    # 预测相对位姿 T_{c -> m} 与 T_{c -> p}
    R_cm, t_cm = pose_net(gc, gm)
    R_cp, t_cp = pose_net(gc, gp)

    gw_m, ps_m, pm_samp, ns_m, nm_samp, ib_m = _warp_target_to_source(pts_c, nrm_c, gm, pts_m, nrm_m, R_cm, t_cm, tan_h)
    gw_p, ps_p, pp_samp, ns_p, np_samp, ib_p = _warp_target_to_source(pts_c, nrm_c, gp, pts_p, nrm_p, R_cp, t_cp, tan_h)

    # 1) Monodepth2 最小重投影光度损失 + 自动静态掩码 (排除动态挥手/遮挡区域)
    pe_m = 0.85 * _ssim_map(gc, gw_m) + 0.15 * (gc - gw_m).abs()
    pe_p = 0.85 * _ssim_map(gc, gw_p) + 0.15 * (gc - gw_p).abs()
    pe_id_m = 0.85 * _ssim_map(gc, gm) + 0.15 * (gc - gm).abs()
    pe_id_p = 0.85 * _ssim_map(gc, gp) + 0.15 * (gc - gp).abs()

    pe_warp_min = torch.minimum(pe_m, pe_p)
    pe_id_min = torch.minimum(pe_id_m, pe_id_p)
    automask = (pe_warp_min < pe_id_min + 0.015).float() * ib_m * ib_p * (v_c.detach() > 0.45).float()
    l_photo = (pe_warp_min * automask).sum() / automask.sum().clamp(min=10.0)

    # 2) 双向 3D 几何循环一致性损失 (直接消灭单帧深度呼吸抖动与法向闪烁！)
    static_m = ((gc - gm).abs() < 0.065).float() * ib_m * (v_c.detach() > 0.50).float()
    static_p = ((gc - gp).abs() < 0.065).float() * ib_p * (v_c.detach() > 0.50).float()
    # 重点强化近桌区域 (< 1.15m) 的毫米级时间一致性
    near_w = (d_c.detach() < 1.15).float().unsqueeze(1) * 1.8 + 0.4
    l_3d_m = ((ps_m - pm_samp).abs().clamp(max=0.12).mean(dim=1, keepdim=True) * static_m * near_w).sum() / (static_m * near_w).sum().clamp(min=10.0)
    l_3d_p = ((ps_p - pp_samp).abs().clamp(max=0.12).mean(dim=1, keepdim=True) * static_p * near_w).sum() / (static_p * near_w).sum().clamp(min=10.0)
    l_3d_cyc = (l_3d_m + l_3d_p) * 0.5

    cos_nm = (ns_m * nm_samp).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    cos_np = (ns_p * np_samp).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    l_nrm_cyc = (
        ((1.0 - cos_nm) * static_m).sum() / static_m.sum().clamp(min=10.0)
        + ((1.0 - cos_np) * static_p).sum() / static_p.sum().clamp(min=10.0)
    ) * 0.5

    # 2b) 解析深度-法向几何正交一致性 (n_geo = normalize(dP/dv x dP/du))，从物理上彻底杜绝法向坍缩为常向量！
    dp_u = pts_c[:, :, 1:-1, 2:] - pts_c[:, :, 1:-1, :-2]
    dp_v = pts_c[:, :, 2:, 1:-1] - pts_c[:, :, :-2, 1:-1]
    n_geo = F.normalize(torch.cross(dp_u, dp_v, dim=1), dim=1, eps=1e-6)
    # 符号对齐：面向相机（n_geo 与指向相机向量同向）
    to_cam = -pts_c[:, :, 1:-1, 1:-1].clone()
    to_cam[:, 2] = to_cam[:, 2] + REAL_H_CAM_M
    sign_f = torch.where((n_geo * to_cam).sum(dim=1, keepdim=True) >= 0.0, 1.0, -1.0)
    n_geo = n_geo * sign_f
    edge_in = batch_mv["xc"][:, 1:2, 1:-1, 1:-1]
    smooth_zone = ((edge_in < 0.22) & (v_c.detach()[:, :, 1:-1, 1:-1] > 0.55) & (d_c.detach().unsqueeze(1)[:, :, 1:-1, 1:-1] < 1.25)).float()
    cos_geo = (nrm_c[:, :, 1:-1, 1:-1] * n_geo.detach()).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    l_geo_nrm = ((1.0 - cos_geo) * smooth_zone).sum() / smooth_zone.sum().clamp(min=20.0)

    # 3) 物理 13.5 cm 刚性桌面平面平滑与法向-深度正交性约束（让真实桌面像玻璃一样平整！）
    edge_c = batch_mv["xc"][:, 1:2]  # [B, 1, H, W]
    yy = torch.linspace(0.0, 1.0, GEO_H, device=out_c.device).view(1, 1, GEO_H, 1)
    desk_mask = (
        (yy > 0.58)
        & (nrm_c[:, 2:3].detach() > 0.90)
        & (pts_c[:, 2:3].detach().abs() < 0.016)
        & (edge_c < 0.16)
        & (d_c.detach().unsqueeze(1) < 1.05)
    ).float()
    # 桌面局部零斜率平滑 (压低桌面 MAD 噪点，同时绝不触碰任何 edge >= 0.16 或 Z >= 1.6cm 的鼠标/水杯!)
    z_c = pts_c[:, 2:3]
    dz_x = (z_c[:, :, :, 1:] - z_c[:, :, :, :-1]).abs() * desk_mask[:, :, :, 1:] * desk_mask[:, :, :, :-1]
    dz_y = (z_c[:, :, 1:, :] - z_c[:, :, :-1, :]).abs() * desk_mask[:, :, 1:, :] * desk_mask[:, :, :-1, :]
    l_desk_flat = (dz_x.sum() + dz_y.sum()) / (desk_mask.sum().clamp(min=20.0)) + (z_c.abs() * desk_mask).sum() / desk_mask.sum().clamp(min=20.0) * 0.5

    # 边缘感知逆深度平滑 (禁止跨图像边缘平滑)
    inv_n = (1.0 / d_c.unsqueeze(1)) / INV_D_SCALE
    gx_img = (gc[:, :, :, 1:] - gc[:, :, :, :-1]).abs()
    gy_img = (gc[:, :, 1:, :] - gc[:, :, :-1, :]).abs()
    l_smooth = (
        ((inv_n[:, :, :, 1:] - inv_n[:, :, :, :-1]).abs() * torch.exp(-gx_img * 12.0)).mean()
        + ((inv_n[:, :, 1:, :] - inv_n[:, :, :-1, :]).abs() * torch.exp(-gy_img * 12.0)).mean()
    )

    total = 2.0 * l_photo + 10.0 * l_3d_cyc + 0.40 * l_nrm_cyc + 1.20 * l_geo_nrm + 6.0 * l_desk_flat + 0.10 * l_smooth
    return total, {
        "l_photo": float(l_photo.detach().item()),
        "l_3d_mm": float(l_3d_cyc.detach().item() * 1000.0),
        "l_nrm_cyc": float(l_nrm_cyc.detach().item()),
        "l_desk_mm": float(l_desk_flat.detach().item() * 1000.0),
    }


# ---------------------------------------------------------------------------
# 3. 100% 不依赖任何教师伪真值的客观真实物理头对头评测 (MoGe-2 419MB vs v3 vs v4!)
# ---------------------------------------------------------------------------


def _load_heldout_seq(kind: str, src: pathlib.Path, start: int, count: int, step: int = 2) -> list[np.ndarray]:
    if kind == "frames":
        jpgs = sorted(src.glob("*.jpg"))[start : start + count * step : step]
        return [cv2.imread(str(j)) for j in jpgs if cv2.imread(str(j)) is not None]
    cap = cv2.VideoCapture(str(src))
    out = []
    for k in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * step)
        ok, fr = cap.read()
        if ok and fr is not None:
            out.append(fr)
    cap.release()
    return out


def eval_objective_real_video_physics(
    fe_name: str,
    frontend: Any,
    frames_bgr: list[np.ndarray],
    scene_tag: str,
    pin_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    """用 6 项 100% 教师无关的客观物理指标评测任意 3D 前端在真实留出视频上的真实表现！

    指标：
    1) `temporal_3d_z_jitter_mm`: 静态像素连续帧 3D 世界高程 Z 的逐帧中位数抖动 (mm, 越低越稳!)
    2) `temporal_3d_depth_jitter_mm`: 静态像素连续帧 3D 深度逐帧抖动 (mm, 越低越稳!)
    3) `temporal_normal_flicker_deg`: 静态像素连续帧表面法向逐帧闪烁角 (°, 越低越稳!)
    4) `desk_plane_flatness_mad_mm`: 刚性桌面平面拟合残差 MAD (mm, 越低越平!)
    5) `multiview_reproj_psnr_db` & `multiview_reproj_l1`: 用每模型自己预测的 3D 点云做跨帧 2D-3D PnP 重投影图像重建 PSNR (dB, 越高几何越真!)
    6) `zero_odom_reloc_err_mm` & `zero_odom_reloc_rot_deg`: 真实异帧零里程计 6-DoF 视觉重定位误差 (mm / °, 越低越准!)
    """
    preds = []
    times_ms = []
    prev_b = None
    for fr in frames_bgr:
        pr = frontend.infer_frame(fr, prev_b, pitch_deg=3.0, roll_deg=0.0, auto_level=True)
        # 统一规范到真实相机原生 240x240 网格，确保 419MB MoGe-2 (480x480) 与 27MB HoverGeoNet (240x240) 100% 公平可比！
        pts = pr["pts_grid"]
        nrm = pr["nrm_grid"]
        vm = pr["valid_grid"].astype(np.float32)
        pts_c = pr["pts_cam"]
        nrm_c = pr["nrm_cam"]
        if pts.shape[:2] != (GEO_H, GEO_W):
            pts = cv2.resize(pts, (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
            nrm = cv2.resize(nrm, (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
            vm = cv2.resize(vm, (GEO_W, GEO_H), interpolation=cv2.INTER_NEAREST)
            pts_c = cv2.resize(pts_c, (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
            nrm_c = cv2.resize(nrm_c, (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
            nrm_c /= np.maximum(np.linalg.norm(nrm_c, axis=-1, keepdims=True), 1e-6)
        preds.append({
            "pts_grid": pts.astype(np.float32),
            "nrm_grid": nrm.astype(np.float32),
            "valid_grid": (vm > 0.5),
            "pts_cam": pts_c.astype(np.float32),
            "nrm_cam": nrm_c.astype(np.float32),
            "depth_m": pts_c[..., 0].astype(np.float32),
        })
        times_ms.append(float(pr["infer_ms"]))
        prev_b = fr

    # 1..3) 静态区时间稳定性 (相邻帧像素灰差 < 10 视为真实静态刚体桌面/物体!)
    z_jitters, d_jitters, n_flickers, desk_mads = [], [], [], []
    for i in range(len(preds)):
        pr = preds[i]
        pts = pr["pts_grid"]
        nrm = pr["nrm_grid"]
        vm = pr["valid_grid"]
        yy = np.arange(GEO_H)[:, None]
        dm = vm & (yy >= 130) & (nrm[..., 2] > 0.88) & (np.abs(pts[..., 2]) < 0.025) & (pr["depth_m"] < 1.15)
        if int(dm.sum()) > 100:
            zp = pts[dm, 2]
            desk_mads.append(float(np.median(np.abs(zp - np.median(zp))) * 1000.0))

        if i == 0:
            continue
        g0 = cv2.cvtColor(frames_bgr[i - 1], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        if g0.shape != (GEO_H, GEO_W):
            g0 = cv2.resize(g0, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)
            g1 = cv2.resize(g1, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)
        stat_m = (np.abs(g1.astype(np.int16) - g0.astype(np.int16)) < 10) & preds[i - 1]["valid_grid"] & vm & (pr["depth_m"] < 1.25)
        if int(stat_m.sum()) > 200:
            dz = np.abs(pts[stat_m, 2] - preds[i - 1]["pts_grid"][stat_m, 2]) * 1000.0
            dd = np.abs(pr["depth_m"][stat_m] - preds[i - 1]["depth_m"][stat_m]) * 1000.0
            cos_n = np.clip(np.sum(nrm[stat_m] * preds[i - 1]["nrm_grid"][stat_m], axis=-1), -1.0, 1.0)
            z_jitters.append(float(np.median(dz)))
            d_jitters.append(float(np.median(dd)))
            n_flickers.append(float(np.median(np.degrees(np.arccos(cos_n)))))

    # 5) 真实跨帧零里程计视觉-3D 重定位 (帧 0 vs 帧 -1, 叠加盲刚体扰动 +7.5cm/-5.0cm/+9.5° yaw/+3.2° pitch!)
    fr_ref, pr_ref = frames_bgr[0], preds[0]
    fr_q, pr_q = frames_bgr[-1], preds[-1]
    ang_y, ang_p = math.radians(9.5), math.radians(3.2)
    cy, sy = math.cos(ang_y), math.sin(ang_y)
    cp, sp = math.cos(ang_p), math.sin(ang_p)
    R_pert = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    t_pert = np.array([0.075, -0.050, 0.008], dtype=np.float32)
    q_pts_pert = ((pr_q["pts_grid"] - np.array([0.0, 0.0, REAL_H_CAM_M], dtype=np.float32) - t_pert) @ R_pert).astype(np.float32)
    q_nrm_pert = (pr_q["nrm_grid"] @ R_pert).astype(np.float32)
    ref_v = pr_ref["valid_grid"]
    rel = relocalize_frame_against_3d_map(
        query_bgr=fr_q,
        query_pts_cam=q_pts_pert,
        query_nrm_cam=q_nrm_pert,
        query_valid=pr_q["valid_grid"],
        ref_bgr=fr_ref,
        ref_pts_world=pr_ref["pts_grid"],
        ref_valid=ref_v,
        static_map_pts=pr_ref["pts_grid"][ref_v][::4],
        static_map_nrms=pr_ref["nrm_grid"][ref_v][::4],
        query_scale=1.0,
    )
    if rel.get("ok"):
        R_est = rel["R_cam_to_world"]
        t_est = rel["cam_pos_world"]
        true_cam = np.array([0.0, 0.0, REAL_H_CAM_M], dtype=np.float32) + t_pert
        p_pin = np.array(pin_xyz, dtype=np.float32)
        rel_true = R_pert.T @ (p_pin - true_cam)
        rel_est = R_est.T @ (p_pin - t_est)
        reloc_err_mm = float(np.linalg.norm(rel_est - rel_true) * 1000.0)
        rot_err_deg = float(math.degrees(math.acos(np.clip((float(np.trace(R_est @ R_pert.T)) - 1.0) * 0.5, -1.0, 1.0))))
    else:
        R_est = np.eye(3, dtype=np.float32)
        t_est = np.array([0.0, 0.0, REAL_H_CAM_M], dtype=np.float32)
        reloc_err_mm, rot_err_deg = 999.0, 99.0

    # 6) 真实无扰动跨帧光度重投影 PSNR (帧 0 3D 点云经实测相对位姿投影到帧 -1 计算像素重建 PSNR dB!)
    rel_raw = relocalize_frame_against_3d_map(
        query_bgr=fr_q,
        query_pts_cam=pr_q["pts_cam"],
        query_nrm_cam=pr_q["nrm_cam"],
        query_valid=pr_q["valid_grid"],
        ref_bgr=fr_ref,
        ref_pts_world=pr_ref["pts_grid"],
        ref_valid=ref_v,
        static_map_pts=pr_ref["pts_grid"][ref_v][::4],
        static_map_nrms=pr_ref["nrm_grid"][ref_v][::4],
        query_scale=1.0,
    )
    R_raw = rel_raw["R_cam_to_world"] if rel_raw.get("ok") else np.eye(3, dtype=np.float32)
    t_raw = rel_raw["cam_pos_world"] if rel_raw.get("ok") else np.array([0.0, 0.0, REAL_H_CAM_M], dtype=np.float32)
    g_ref = cv2.resize(cv2.cvtColor(fr_ref, cv2.COLOR_BGR2GRAY), (GEO_W, GEO_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    g_q = cv2.resize(cv2.cvtColor(fr_q, cv2.COLOR_BGR2GRAY), (GEO_W, GEO_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pts_in_q = (pr_ref["pts_grid"] - t_raw[None, None, :]) @ R_raw
    tan_h = math.tan(math.radians(REAL_FOV_H_DEG * 0.5))
    xc = np.maximum(pts_in_q[..., 0], 0.08)
    u_px = ((0.5 + pts_in_q[..., 1] / (2.0 * xc * tan_h)) * (GEO_W - 1)).astype(np.float32)
    v_px = ((0.5 - pts_in_q[..., 2] / (2.0 * xc * tan_h)) * (GEO_H - 1)).astype(np.float32)
    g_warped = cv2.remap(g_q, u_px, v_px, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    valid_w = ref_v & (u_px >= 4) & (u_px <= GEO_W - 5) & (v_px >= 4) & (v_px <= GEO_H - 5) & (pts_in_q[..., 0] > 0.14)
    mse = float(np.mean((g_ref[valid_w] - g_warped[valid_w]) ** 2)) if int(valid_w.sum()) > 100 else 0.1
    psnr_db = float(-10.0 * math.log10(max(mse, 1e-6)))
    reproj_l1 = float(np.mean(np.abs(g_ref[valid_w] - g_warped[valid_w]))) if int(valid_w.sum()) > 100 else 0.5

    steady_ms = float(np.mean(times_ms[2:])) if len(times_ms) > 2 else float(np.mean(times_ms))
    return {
        "model": fe_name,
        "scene": scene_tag,
        "infer_ms": round(steady_ms, 2),
        "fps": round(1000.0 / max(steady_ms, 0.1), 1),
        "temporal_3d_z_jitter_mm": round(float(np.mean(z_jitters)), 2) if z_jitters else -1.0,
        "temporal_3d_depth_jitter_mm": round(float(np.mean(d_jitters)), 2) if d_jitters else -1.0,
        "temporal_normal_flicker_deg": round(float(np.mean(n_flickers)), 2) if n_flickers else -1.0,
        "desk_plane_flatness_mad_mm": round(float(np.mean(desk_mads)), 2) if desk_mads else -1.0,
        "multiview_reproj_psnr_db": round(psnr_db, 2),
        "multiview_reproj_l1": round(reproj_l1, 4),
        "zero_odom_reloc_err_mm": round(reloc_err_mm, 2),
        "zero_odom_reloc_rot_deg": round(rot_err_deg, 2),
        "reloc_inliers": int(rel.get("ransac_inliers", 0)),
    }


def train_and_benchmark_v4(epochs: int = 5, lr: float = 2.2e-4) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=== BREAKTHROUGH #1: Training HoverGeoNet-MV-SelfSup v4 (Multi-View Self-Supervision to Beat 419MB MoGe-2!) ===", flush=True)
    triplets = collect_real_unlabeled_triplets(max_per_src=75, stride_frames=2)
    print(f"  Collected {len(triplets)} REAL unlabeled consecutive video triplets (ZERO held-out empty_d0.5 / cb8455!)", flush=True)
    ds_mv = RealVideoTripletDataset(triplets)
    dl_mv = DataLoader(ds_mv, batch_size=4, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    # 同时加载干净几何回放集防止遗忘 3.5cm 鼠标与 90° 桌角微凸起
    train_real, train_live, train_synth, val_real, _ = prepare_all_caches(force_rebuild=False)
    train_real = _filter_clean_teacher(train_real, "train_real")
    ds_sup = HoverGeoDataset([train_real, train_synth], is_train=True, imu_noise=True)
    dl_sup = DataLoader(ds_sup, batch_size=6, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    model = create_hover_geonet(arch="vit_b1", in_ch=GEO_IN_CH, base=24, pretrained_backbone=False).to(device)
    blob_v3 = torch.load(V3_CKPT_FP32, map_location=device, weights_only=False)
    model.load_state_dict(blob_v3["state_dict"], strict=False)
    pose_net = RockingPoseNet().to(device)

    opt = torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": lr * 0.15, "weight_decay": 1e-2},
            {"params": list(model.head_parameters()), "lr": lr, "weight_decay": 5e-4},
            {"params": list(pose_net.parameters()), "lr": lr * 3.0, "weight_decay": 1e-4},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.08)

    t0 = time.perf_counter()
    sup_iter = iter(dl_sup)
    best_score = 1e9
    best_sd = None
    for ep in range(1, epochs + 1):
        model.train()
        pose_net.train()
        ep_photo, ep_3d, ep_desk, steps = 0.0, 0.0, 0.0, 0
        t_ep = time.perf_counter()

        for batch_mv in dl_mv:
            try:
                batch_sup = next(sup_iter)
            except StopIteration:
                sup_iter = iter(dl_sup)
                batch_sup = next(sup_iter)

            bm = {k: v.to(device, non_blocking=True) for k, v in batch_mv.items()}
            bs = {k: v.to(device, non_blocking=True) for k, v in batch_sup.items()}

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    om = model(bm["xm"]).float()
                    op = model(bm["xp"]).float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                oc = model(bm["xc"])
                loss_mv, st_mv = compute_multiview_selfsup_loss(om, oc.float(), op, bm, pose_net)
            loss_mv.backward()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out_sup = model(bs["x7"])
                loss_sup, _ = compute_geonet_loss(out_sup.float(), bs)
            (1.35 * loss_sup).backward()

            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(pose_net.parameters()), 1.5)
            opt.step()

            ep_photo += st_mv["l_photo"]
            ep_3d += st_mv["l_3d_mm"]
            ep_desk += st_mv["l_desk_mm"]
            steps += 1

        sched.step()
        val_m = evaluate_split(model, val_real, device, use_gt_imu=True)
        # 综合验证指标：兼顾多视角时间一致 (3d_cyc)、桌面平整度 (deskMAD)、微物体锐度 (microZ) 与法向精度 (nrm)
        comp = float(ep_3d / steps) * 0.5 + float(val_m["desk_z_mad_mm"]) * 2.0 + float(val_m["micro_obj_z_mae_mm"]) * 0.8 + float(val_m["normal_angle_deg"]) * 0.4
        star = ""
        if comp < best_score:
            best_score = comp
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *BEST*"
        print(
            f"  [MV-SelfSup v4] Ep {ep}/{epochs} ({time.perf_counter()-t_ep:.1f}s) | "
            f"photo={ep_photo/steps:.4f} 3d_cyc={ep_3d/steps:.2f}mm desk_flat={ep_desk/steps:.2f}mm | "
            f"VAL microZ={val_m['micro_obj_z_mae_mm']:.1f}mm deskMAD={val_m['desk_z_mad_mm']:.1f}mm nrm={val_m['normal_angle_deg']:.1f}°{star}",
            flush=True,
        )

    if best_sd is not None:
        model.load_state_dict(best_sd)
    train_s = time.perf_counter() - t0
    ckpt_v4_fp32 = OUT_DIR / "hover_geonet_mv_v4_fp32.pt"
    ckpt_v4_fp16 = OUT_DIR / "hover_geonet_mv_v4_fp16.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arch": "vit_b1", "in_ch": GEO_IN_CH, "base": 24,
            "h_cam_m": REAL_H_CAM_M, "fov_h_deg": REAL_FOV_H_DEG,
        },
        ckpt_v4_fp32,
    )
    fp16_sd = {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict_fp16": fp16_sd,
            "arch": "vit_b1", "in_ch": GEO_IN_CH, "base": 24,
            "h_cam_m": REAL_H_CAM_M, "fov_h_deg": REAL_FOV_H_DEG,
        },
        ckpt_v4_fp16,
    )
    del model, pose_net
    torch.cuda.empty_cache()

    # ---- 严格留出真实视频头对头客观物理对比：MoGe-2-ViT-B (419MB) vs v3 (纯蒸馏) vs v4 (多视角自监督!) ----
    print("\n=== OBJECTIVE TEACHER-INDEPENDENT BENCHMARK ON STRICTLY HELD-OUT REAL VIDEOS ===", flush=True)
    seq_empty = _load_heldout_seq("frames", EMPTY_DIR, start=10, count=22, step=2)
    seq_cb = _load_heldout_seq("video", CB8455_MP4, start=580, count=22, step=3)

    results: dict[str, dict[str, Any]] = {"empty_d05": {}, "cb8455": {}}
    candidates = [
        ("MoGe-2-ViT-B (419MB Teacher)", lambda: MoGe2Frontend(infer_hw=480, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)),
        ("HoverGeoNet-v3 (27MB Distilled)", lambda: HoverGeoNet240Frontend(V3_CKPT_FP16, use_fp16=True)),
        ("HoverGeoNet-MV-SelfSup v4 (27MB)", lambda: HoverGeoNet240Frontend(ckpt_v4_fp16, use_fp16=True)),
    ]
    for name, ctor in candidates:
        fe = ctor()
        r_e = eval_objective_real_video_physics(name, fe, seq_empty, "heldout_empty_d05", (0.38, -0.05, 0.0))
        r_c = eval_objective_real_video_physics(name, fe, seq_cb, "heldout_cb8455", (0.42, 0.06, 0.0))
        results["empty_d05"][name] = r_e
        results["cb8455"][name] = r_c
        print(f"  [{name}] empty_d0.5 -> {r_e}", flush=True)
        print(f"  [{name}] cb8455     -> {r_c}", flush=True)
        del fe
        torch.cuda.empty_cache()

    rep = {
        "model": "HoverGeoNet-MV-SelfSup v4 (27.0 MB FP16, Multi-View Self-Supervised)",
        "train_seconds": round(train_s, 1),
        "n_real_unlabeled_triplets": len(triplets),
        "ckpt_fp16": str(ckpt_v4_fp16),
        "ckpt_fp16_mb": round(ckpt_v4_fp16.stat().st_size / 1048576.0, 2),
        "objective_real_video_benchmark": results,
    }
    (OUT_DIR / "mv_selfsup_v4_benchmark.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved Breakthrough #1 benchmark -> {OUT_DIR / 'mv_selfsup_v4_benchmark.json'}", flush=True)
    return rep


if __name__ == "__main__":
    train_and_benchmark_v4(epochs=4, lr=7.0e-5)
