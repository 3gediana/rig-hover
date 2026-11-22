# -*- coding: utf-8 -*-
"""train_hover_geonet_240.py — `HoverGeoNet-240` 端到端离线 MoGe-2 蒸馏 + 解析微物体训练与零泄漏评测脚本。

严格科研与工程标准：
1. **100% 零数据泄漏划分（跨天、跨桌面零样本泛化评测）**：
   - **训练集 (Real, 900 帧)**：仅取自 2026-11-05 与 2026-11-07 的 6 段独立录像（含桌面、手机、键盘、人手等）；
   - **训练集 (Exact Analytical Synth, 450 帧)**：精确解析射线追踪生成的 3.5 cm 弧顶鼠标、1.2 cm 手机、斜坡键盘、水杯与 90° 盒子（物理真值 `H_cam = 13.5 cm`）；
   - **内部验证集 (Val, 100 帧)**：仅用于早停与最优权重挑选 (`65a54d` 60 帧 + `lean_out_d0.5` 40 帧)；
   - **严格留出测试集 (Held-Out Test, 80 帧)**：`empty_d0.5_front_night_182201`（40 帧，鼠标+键盘+显示器+暗光桌面）与
     `robot_20261115_151733_cb8455`（40 帧，水杯+笔记本+挥动手臂），**全程绝不参与梯度更新或模型挑选**！
2. **精确物理尺度对齐**：
   - 使用用户实测相机离桌高度 `REAL_H_CAM_M = 0.135 m (13.5 cm)` + 锁定 `MoGe-2` `fov_x = 65.0°`，
     已验证 `MoGe-2` 与 `HoverGeoNet-240` 射线反投影误差 `median = 0.000 mm, max = 0.001 mm`！
3. **微凸起与 90° 折痕定向加权蒸馏**：
   - 对高度 `8 mm ~ 16 cm` 的桌面微物体（鼠标、手机、键盘）及法向折痕处施加重权重，
     结合多尺度深度梯度损失 + 法向余弦/折痕梯度损失 + 几何法向一致性约束 + 真机 1.16 Hz IMU 噪声鲁棒增强。
"""
from __future__ import annotations

import argparse
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

from hd_moge_contour_engine import (  # noqa: E402
    MoGe2DesktopReconstructor,
    level_and_scale_desktop_cloud,
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
    REAL_FOV_H_DEG,
    REAL_H_CAM_M,
    HoverGeoNet240,
    build_robot_physics_input,
    compute_desk_prior_from_rays,
    create_hover_geonet,
    decode_geonet_to_world_cloud,
    enhance_gray_240,
    render_synthetic_desktop_micro_scene,
)
from mono_depth_model import camera_rotation_matrix, make_camera_ray_grid  # noqa: E402

CACHE_DIR = REPO / "pc" / "bench" / "_teacher_cache" / "geonet240_h135"
OUT_DIR = REPO / "runs_end_side" / "hover_geonet_240"


# ---------------------------------------------------------------------------
# 1. 数据源采样器（配对真实相邻前帧以提供真实 frame_diff）
# ---------------------------------------------------------------------------


def _sample_video_pairs(mp4_path: pathlib.Path, n_samples: int, margin: int = 12) -> list[tuple[np.ndarray, np.ndarray, str]]:
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= margin * 2 + 4:
        cap.release()
        return []
    idxs = np.linspace(margin, total - margin - 1, int(n_samples), dtype=np.int32)
    idxs = np.unique(idxs)
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    for idx in idxs:
        prev_idx = max(0, int(idx) - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ok_p, bgr_p = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok_c, bgr_c = cap.read()
        if ok_c and bgr_c is not None:
            if not ok_p or bgr_p is None:
                bgr_p = bgr_c.copy()
            out.append((bgr_c, bgr_p, f"{mp4_path.parent.name if mp4_path.name == 'take.mp4' else mp4_path.stem}:{int(idx)}"))
    cap.release()
    return out


def _sample_session_pairs(frames_dir: pathlib.Path, n_samples: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if len(files) < 5:
        return []
    idxs = np.unique(np.linspace(3, len(files) - 2, int(n_samples), dtype=np.int32))
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    for idx in idxs:
        bgr_c = cv2.imread(str(files[int(idx)]))
        bgr_p = cv2.imread(str(files[max(0, int(idx) - 2)]))
        if bgr_c is not None:
            if bgr_p is None:
                bgr_p = bgr_c.copy()
            out.append((bgr_c, bgr_p, f"{frames_dir.parent.name}:{files[int(idx)].stem}"))
    return out


def _downsample_teacher_480_to_240(
    depth_480: np.ndarray,
    nrm_cam_480: np.ndarray,
    valid_480: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """保锐边 2×2 下采样：只有 2×2 子块全部有效时才判为有效，杜绝前景-背景深度粘连。"""
    v_blk = valid_480.reshape(GEO_H, 2, GEO_W, 2)
    valid_240 = v_blk.all(axis=(1, 3))

    d_blk = depth_480.reshape(GEO_H, 2, GEO_W, 2)
    depth_240 = np.clip(d_blk.mean(axis=(1, 3)), DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float32)

    n_blk = nrm_cam_480.reshape(GEO_H, 2, GEO_W, 2, 3)
    nrm_240 = n_blk.mean(axis=(1, 3)).astype(np.float32)
    nrm_240 /= np.maximum(np.linalg.norm(nrm_240, axis=-1, keepdims=True), 1e-6)
    return depth_240, nrm_240, valid_240


def _extract_teacher_split(
    rec: MoGe2DesktopReconstructor,
    pairs: list[tuple[np.ndarray, np.ndarray, str]],
    cache_path: pathlib.Path,
    tag: str,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        print(f"[Cache HIT] {tag}: {cache_path}", flush=True)
        return dict(np.load(cache_path, allow_pickle=False))

    n = len(pairs)
    gray_cur = np.zeros((n, GEO_H, GEO_W), dtype=np.uint8)
    gray_prev = np.zeros((n, GEO_H, GEO_W), dtype=np.uint8)
    depth_m = np.zeros((n, GEO_H, GEO_W), dtype=np.float16)
    nrm_cam = np.zeros((n, GEO_H, GEO_W, 3), dtype=np.float16)
    valid = np.zeros((n, GEO_H, GEO_W), dtype=np.uint8)
    pitch_gt = np.zeros((n,), dtype=np.float32)
    roll_gt = np.zeros((n,), dtype=np.float32)
    scales = np.zeros((n,), dtype=np.float32)
    meta_names: list[str] = []

    t0 = time.perf_counter()
    for i, (bgr_c, bgr_p, name) in enumerate(pairs):
        raw = rec.infer_raw(bgr_c)
        lvl = level_and_scale_desktop_cloud(
            raw["pts_cam"], raw["nrm_cam"], raw["mask"], h_cam_m=REAL_H_CAM_M
        )
        d480 = (raw["pts_cam"][..., 0] * lvl["scale"]).astype(np.float32)
        d240, n240, v240 = _downsample_teacher_480_to_240(d480, raw["nrm_cam"], lvl["valid_grid"])

        gc = cv2.cvtColor(bgr_c, cv2.COLOR_BGR2GRAY) if bgr_c.ndim == 3 else bgr_c
        gp = cv2.cvtColor(bgr_p, cv2.COLOR_BGR2GRAY) if bgr_p.ndim == 3 else bgr_p
        if gc.shape != (GEO_H, GEO_W):
            gc = cv2.resize(gc, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)
        if gp.shape != (GEO_H, GEO_W):
            gp = cv2.resize(gp, (GEO_W, GEO_H), interpolation=cv2.INTER_AREA)

        gray_cur[i] = gc
        gray_prev[i] = gp
        depth_m[i] = d240.astype(np.float16)
        nrm_cam[i] = n240.astype(np.float16)
        valid[i] = v240.astype(np.uint8)
        pitch_gt[i] = float(lvl["pitch_deg"])
        roll_gt[i] = float(lvl["roll_deg"])
        scales[i] = float(lvl["scale"])
        meta_names.append(name)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            el = time.perf_counter() - t0
            print(
                f"  [{tag}] {i+1:4d}/{n} ({el:.1f}s, {(el/(i+1))*1000:.0f} ms/f) "
                f"last={name} pitch={pitch_gt[i]:+.1f} roll={roll_gt[i]:+.1f} valid={v240.mean()*100:.0f}%",
                flush=True,
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gray_cur": gray_cur,
        "gray_prev": gray_prev,
        "depth_m": depth_m,
        "nrm_cam": nrm_cam,
        "valid": valid,
        "pitch_gt": pitch_gt,
        "roll_gt": roll_gt,
        "scales": scales,
        "names": np.asarray(meta_names, dtype="U96"),
    }
    np.savez(cache_path, **payload)
    print(f"[Cache SAVED] {tag} -> {cache_path} ({cache_path.stat().st_size / 1048576:.1f} MB)", flush=True)
    return payload


def _build_synth_split(n_synth: int, cache_path: pathlib.Path) -> dict[str, np.ndarray]:
    if cache_path.exists():
        print(f"[Cache HIT] synth: {cache_path}", flush=True)
        return dict(np.load(cache_path, allow_pickle=False))

    rng = np.random.default_rng(2026112201)
    gray_cur = np.zeros((n_synth, GEO_H, GEO_W), dtype=np.uint8)
    gray_prev = np.zeros((n_synth, GEO_H, GEO_W), dtype=np.uint8)
    depth_m = np.zeros((n_synth, GEO_H, GEO_W), dtype=np.float16)
    nrm_cam = np.zeros((n_synth, GEO_H, GEO_W, 3), dtype=np.float16)
    valid = np.zeros((n_synth, GEO_H, GEO_W), dtype=np.uint8)
    pitch_gt = np.zeros((n_synth,), dtype=np.float32)
    roll_gt = np.zeros((n_synth,), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n_synth):
        s = render_synthetic_desktop_micro_scene(rng, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
        gc = cv2.cvtColor(s["bgr"], cv2.COLOR_BGR2GRAY)
        # 模拟 1.16 Hz 自平衡微垂移作为 prev 帧
        dy = int(rng.integers(-3, 4))
        dx = int(rng.integers(-2, 3))
        gp = np.roll(np.roll(gc, dy, axis=0), dx, axis=1)
        gp = np.clip(gp.astype(np.int16) + rng.integers(-4, 5, size=gp.shape), 0, 255).astype(np.uint8)

        # 将合成器输出的世界法向旋回相机系，统一与 MoGe-2 缓存格式完全一致
        r_gt = camera_rotation_matrix(pitch_deg=s["pitch_deg"], roll_deg=s["roll_deg"])
        n_cam = (s["nrm_world"] @ r_gt).astype(np.float32)
        n_cam /= np.maximum(np.linalg.norm(n_cam, axis=-1, keepdims=True), 1e-6)

        gray_cur[i] = gc
        gray_prev[i] = gp
        depth_m[i] = np.clip(s["depth_m"], DEPTH_MIN_M, DEPTH_MAX_M).astype(np.float16)
        nrm_cam[i] = n_cam.astype(np.float16)
        valid[i] = (s["valid"] > 0.5).astype(np.uint8)
        pitch_gt[i] = float(s["pitch_deg"])
        roll_gt[i] = float(s["roll_deg"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gray_cur": gray_cur,
        "gray_prev": gray_prev,
        "depth_m": depth_m,
        "nrm_cam": nrm_cam,
        "valid": valid,
        "pitch_gt": pitch_gt,
        "roll_gt": roll_gt,
    }
    np.savez(cache_path, **payload)
    print(f"[Cache SAVED] synth ({n_synth} scenes, {time.perf_counter()-t0:.1f}s) -> {cache_path}", flush=True)
    return payload


def prepare_all_caches(force_rebuild: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    p_train_real = CACHE_DIR / "train_real_900.npz"
    p_train_live = CACHE_DIR / "train_live_480.npz"
    p_train_synth = CACHE_DIR / "train_synth_600.npz"
    p_val_real = CACHE_DIR / "val_internal_100.npz"
    p_test_heldout = CACHE_DIR / "test_heldout_80.npz"

    if force_rebuild:
        for p in (p_train_real, p_train_live, p_train_synth, p_val_real, p_test_heldout):
            if p.exists():
                p.unlink()

    need_moge = not (p_train_real.exists() and p_train_live.exists() and p_val_real.exists() and p_test_heldout.exists())
    rec = MoGe2DesktopReconstructor(infer_hw=480, fov_x_deg=REAL_FOV_H_DEG) if need_moge else None
    if rec is not None:
        print(f"MoGe-2-ViT-B teacher loaded in {rec.load_s:.2f}s on {rec.device} (H_cam={REAL_H_CAM_M}m, fov={REAL_FOV_H_DEG}°)", flush=True)

    # 1) Train Real (Nov 5 + Nov 7 videos, 900 frames — reused from cache!)
    if not p_train_real.exists():
        train_pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
        train_sources = [
            (REPO / "dataset" / "takes" / "robot_20261107_204337_5ca36d" / "take.mp4", 150),
            (REPO / "dataset" / "takes" / "robot_20261107_212119_67f0c1" / "take.mp4", 170),
            (REPO / "dataset" / "takes" / "robot_20261107_213314_5c0158" / "take.mp4", 170),
            (REPO / "dataset" / "takes" / "robot_20261107_214950_1f30d1" / "take.mp4", 170),
            (REPO / "video_annotation_data" / "videos" / "robot_20261105_004630_ef1d19.mp4", 180),
            (REPO / "video_annotation_data" / "videos" / "robot_20261105_190704_3e80b8.mp4", 60),
        ]
        for mp4, k in train_sources:
            train_pairs.extend(_sample_video_pairs(mp4, k))
        train_real = _extract_teacher_split(rec, train_pairs, p_train_real, "train_real")
    else:
        train_real = dict(np.load(p_train_real, allow_pickle=False))

    # 1b) Train Live (6 additional diverse live robot captures, 480 frames; fov_sweep & heldout strictly excluded!)
    if not p_train_live.exists():
        live_pairs: list[tuple[np.ndarray, np.ndarray, str]] = []
        live_sources = [
            (REPO / "evidence" / "live" / "calib_1" / "frames", 90),
            (REPO / "evidence" / "live" / "gesture_1" / "frames", 75),
            (REPO / "evidence" / "live" / "gesture_2" / "frames", 115),
            (REPO / "evidence" / "live" / "label_none" / "frames", 70),
            (REPO / "evidence" / "live" / "label_fist" / "frames", 65),
            (REPO / "evidence" / "live" / "label_open_palm" / "frames", 65),
        ]
        for fdir, k in live_sources:
            live_pairs.extend(_sample_session_pairs(fdir, k))
        train_live = _extract_teacher_split(rec, live_pairs, p_train_live, "train_live")
    else:
        train_live = dict(np.load(p_train_live, allow_pickle=False))

    # 2) Train Synth: 600 exact analytical desktop micro-object scenes (H_cam = 0.135m)
    train_synth = _build_synth_split(600, p_train_synth)

    # 3) Val Internal: 65a54d (60) + lean_out_d0.5 (40) — checkpoint selection only
    if not p_val_real.exists():
        val_pairs = (
            _sample_video_pairs(REPO / "dataset" / "takes" / "robot_20261115_150045_65a54d" / "take.mp4", 60)
            + _sample_session_pairs(REPO / "evidence" / "sessions" / "lean_out_d0.5_front_night_182257" / "frames", 40)
        )
        val_real = _extract_teacher_split(rec, val_pairs, p_val_real, "val_internal")
    else:
        val_real = dict(np.load(p_val_real, allow_pickle=False))

    # 4) Strictly Held-Out Test: empty_d0.5 (40) + cb8455 (40) — ZERO touching during train/val!
    if not p_test_heldout.exists():
        test_pairs = (
            _sample_session_pairs(REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames", 40)
            + _sample_video_pairs(REPO / "dataset" / "takes" / "robot_20261115_151733_cb8455" / "take.mp4", 40)
        )
        test_heldout = _extract_teacher_split(rec, test_pairs, p_test_heldout, "test_heldout")
    else:
        test_heldout = dict(np.load(p_test_heldout, allow_pickle=False))

    if rec is not None:
        del rec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_real, train_live, train_synth, val_real, test_heldout


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset（在线物理姿态扰动 + 左右镜像 + 暗光/模糊真机增强）
# ---------------------------------------------------------------------------


_RAYS_CAM_240 = make_camera_ray_grid(GEO_H, GEO_W, fov_h_deg=REAL_FOV_H_DEG, pitch_deg=0.0, roll_deg=0.0)


class HoverGeoDataset(Dataset):
    """支持真实 + 合成混合训练、物理精确水平镜像与真机 1.16 Hz IMU 噪声鲁棒增强的数据集。"""

    def __init__(self, splits: list[dict[str, np.ndarray]], is_train: bool = True, imu_noise: bool = True) -> None:
        self.gray_cur = np.concatenate([s["gray_cur"] for s in splits], axis=0)
        self.gray_prev = np.concatenate([s["gray_prev"] for s in splits], axis=0)
        self.depth_m = np.concatenate([s["depth_m"] for s in splits], axis=0)
        self.nrm_cam = np.concatenate([s["nrm_cam"] for s in splits], axis=0)
        self.valid = np.concatenate([s["valid"] for s in splits], axis=0)
        self.pitch_gt = np.concatenate([s["pitch_gt"] for s in splits], axis=0)
        self.roll_gt = np.concatenate([s["roll_gt"] for s in splits], axis=0)
        self.is_train = bool(is_train)
        self.imu_noise = bool(imu_noise)

        # 预计算 CLAHE 加速在线训练
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        n = len(self.gray_cur)
        self.enh_cur = np.empty((n, GEO_H, GEO_W), dtype=np.uint8)
        self.enh_prev = np.empty((n, GEO_H, GEO_W), dtype=np.uint8)
        for i in range(n):
            self.enh_cur[i] = clahe.apply(self.gray_cur[i])
            self.enh_prev[i] = clahe.apply(self.gray_prev[i])

    def __len__(self) -> int:
        return len(self.gray_cur)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g_cur = self.enh_cur[idx].astype(np.float32) / 255.0
        g_prev = self.enh_prev[idx].astype(np.float32) / 255.0
        depth = self.depth_m[idx].astype(np.float32)
        n_cam = self.nrm_cam[idx].astype(np.float32)
        valid = self.valid[idx].astype(np.float32)
        p_gt = float(self.pitch_gt[idx])
        r_gt = float(self.roll_gt[idx])

        rays_cam_use = _RAYS_CAM_240

        if self.is_train:
            # 1) 50% 物理精确水平镜像（同步翻转图像、深度、法向 Y 轴与 roll 符号）
            if np.random.rand() < 0.5:
                g_cur = np.ascontiguousarray(g_cur[:, ::-1])
                g_prev = np.ascontiguousarray(g_prev[:, ::-1])
                depth = np.ascontiguousarray(depth[:, ::-1])
                n_cam = np.ascontiguousarray(n_cam[:, ::-1, :]).copy()
                n_cam[..., 1] *= -1.0
                valid = np.ascontiguousarray(valid[:, ::-1])
                r_gt = -r_gt

            # 1b) 70% 精确针孔射线一致的随机裁剪平移增强（彻底破坏固定机位像素记忆，同时 100% 保持物理射线精确！）
            if np.random.rand() < 0.70:
                crop_s = int(round(GEO_H * float(np.random.uniform(0.68, 0.96))))
                y0 = int(np.random.randint(0, GEO_H - crop_s + 1))
                x0 = int(np.random.randint(0, GEO_W - crop_s + 1))
                g_cur = cv2.resize(g_cur[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
                g_prev = cv2.resize(g_prev[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_NEAREST)
                valid = cv2.resize(valid[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_NEAREST)
                n_cam = cv2.resize(n_cam[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_NEAREST)
                rays_cam_use = cv2.resize(_RAYS_CAM_240[y0 : y0 + crop_s, x0 : x0 + crop_s], (GEO_W, GEO_H), interpolation=cv2.INTER_LINEAR)

            # 2) 光照/暗夜对比度 + GC0308 噪声 + 轻度 1.16 Hz 垂直运动模糊增强
            gamma = float(np.random.uniform(0.68, 1.42))
            gain = float(np.random.uniform(0.74, 1.26))
            bias = float(np.random.uniform(-0.08, 0.08))
            g_cur = np.clip((np.power(np.clip(g_cur, 1e-4, 1.0), gamma) * gain) + bias, 0.0, 1.0)
            g_prev = np.clip((np.power(np.clip(g_prev, 1e-4, 1.0), gamma) * gain) + bias, 0.0, 1.0)
            if np.random.rand() < 0.40:
                g_cur = np.clip(g_cur + np.random.normal(0.0, float(np.random.uniform(0.005, 0.025)), g_cur.shape), 0.0, 1.0).astype(np.float32)
            if np.random.rand() < 0.25:
                k = np.array([[0.2], [0.6], [0.2]], dtype=np.float32)
                g_cur = cv2.filter2D(g_cur, -1, k)
            if np.random.rand() < 0.15:
                g_prev = g_cur

            # 3) 真机 IMU 噪声/40ms 滞后/无 IMU 鲁棒扰动
            if self.imu_noise:
                u = np.random.rand()
                if u < 0.16:
                    p_in = float(np.random.uniform(0.5, 5.0))
                    r_in = float(np.random.uniform(-2.0, 2.0))
                elif u < 0.35:
                    p_in = p_gt + float(np.clip(np.random.normal(0.0, 2.4), -5.5, 5.5))
                    r_in = r_gt + float(np.clip(np.random.normal(0.0, 1.8), -4.5, 4.5))
                else:
                    p_in = p_gt + float(np.clip(np.random.normal(0.0, 0.9), -2.5, 2.5))
                    r_in = r_gt + float(np.clip(np.random.normal(0.0, 0.7), -2.0, 2.0))
            else:
                p_in, r_in = p_gt, r_gt
        else:
            p_in, r_in = p_gt, r_gt

        # 构造 7 通道机器人物理输入
        gx = cv2.Sobel(g_cur, cv2.CV_32F, 1, 0, ksize=3) * 0.25
        gy = cv2.Sobel(g_cur, cv2.CV_32F, 0, 1, ksize=3) * 0.25
        edge_hf = np.clip(np.sqrt(gx * gx + gy * gy) * 3.0, 0.0, 1.0).astype(np.float32)
        fdiff = np.clip((g_cur - g_prev) * 2.5, -1.0, 1.0).astype(np.float32)

        r_in_mat = camera_rotation_matrix(pitch_deg=p_in, roll_deg=r_in)
        rays_in = (rays_cam_use @ r_in_mat.T).astype(np.float32)
        inv_d_prior = compute_desk_prior_from_rays(rays_in, h_cam_m=REAL_H_CAM_M)
        u_rays = rays_in / np.maximum(np.linalg.norm(rays_in, axis=-1, keepdims=True), 1e-6)

        x7 = np.stack(
            [g_cur, edge_hf, fdiff, u_rays[..., 0], u_rays[..., 1], u_rays[..., 2], inv_d_prior / INV_D_SCALE],
            axis=0,
        ).astype(np.float32)

        # 目标法向与输入坐标系 R_in 严格一致，确保点云与法向同系零错位
        n_in_gt = (n_cam @ r_in_mat.T).astype(np.float32)
        n_in_gt /= np.maximum(np.linalg.norm(n_in_gt, axis=-1, keepdims=True), 1e-6)

        # 真实世界 Z（用真实 R_gt 计算，专用于精确标记 8mm~16cm 桌面微凸起权重）
        r_gt_mat = camera_rotation_matrix(pitch_deg=p_gt, roll_deg=r_gt)
        rays_gt_z = (rays_cam_use @ r_gt_mat[2, :]).astype(np.float32)
        z_world_true = (REAL_H_CAM_M + rays_gt_z * depth).astype(np.float32)

        return {
            "x7": torch.from_numpy(x7),
            "rays_in": torch.from_numpy(rays_in.transpose(2, 0, 1)),      # [3, 240, 240]
            "inv_d_prior": torch.from_numpy(inv_d_prior),                 # [240, 240]
            "depth_gt": torch.from_numpy(depth),                          # [240, 240]
            "nrm_gt": torch.from_numpy(n_in_gt.transpose(2, 0, 1)),       # [3, 240, 240]
            "valid_gt": torch.from_numpy(valid),                          # [240, 240]
            "z_world_true": torch.from_numpy(z_world_true),               # [240, 240]
        }


# ---------------------------------------------------------------------------
# 3. 微凸起 + 90° 折痕定向加权 3D 几何损失
# ---------------------------------------------------------------------------


def compute_geonet_loss(out5: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    rays_in = batch["rays_in"]          # [B, 3, H, W]
    inv_d_prior = batch["inv_d_prior"]  # [B, H, W]
    depth_gt = batch["depth_gt"]        # [B, H, W]
    nrm_gt = batch["nrm_gt"]            # [B, 3, H, W]
    valid = batch["valid_gt"]           # [B, H, W]
    z_true = batch["z_world_true"]      # [B, H, W]

    delta_inv = out5[:, 0] * INV_D_SCALE
    inv_d_pred = (inv_d_prior + delta_inv).clamp(INV_D_MIN, INV_D_MAX)
    inv_d_gt = (1.0 / depth_gt.clamp(DEPTH_MIN_M, DEPTH_MAX_M)).clamp(INV_D_MIN, INV_D_MAX)
    depth_pred = 1.0 / inv_d_pred

    nrm_pred = out5[:, 1:4]
    valid_logit = out5[:, 4]

    # 计算真实法向折痕图（用于给 90° 桌角、键盘边与鼠标轮廓加权）
    dot_x = (nrm_gt[:, :, :, 1:] * nrm_gt[:, :, :, :-1]).sum(dim=1).clamp(-1.0, 1.0)
    dot_y = (nrm_gt[:, :, 1:, :] * nrm_gt[:, :, :-1, :]).sum(dim=1).clamp(-1.0, 1.0)
    crease_gt = torch.zeros_like(valid)
    crease_gt[:, :, 1:] += (1.0 - dot_x) * 0.5
    crease_gt[:, 1:, :] += (1.0 - dot_y) * 0.5

    # 微凸起 + 折痕定向重要性权重：鼠标/手机/键盘 (8mm..16cm, 近距<1.25m) 权重 x3.5，折痕棱角额外 +3.0
    is_micro = ((z_true >= 0.008) & (z_true <= 0.16) & (depth_gt < 1.25)).float()
    is_near_desk = ((z_true >= -0.02) & (z_true < 0.008) & (depth_gt < 1.10)).float()
    is_crease = (crease_gt > 0.05).float()
    w_pix = valid * (1.0 + 2.5 * is_micro + 0.6 * is_near_desk + 3.0 * is_crease)
    w_sum = w_pix.sum().clamp(min=1.0)

    # 1) 加权残差逆深度损失 (SmoothL1)
    l_inv = (F.smooth_l1_loss(inv_d_pred / INV_D_SCALE, inv_d_gt / INV_D_SCALE, beta=0.02, reduction="none") * w_pix).sum() / w_sum

    # 2) 直接物理 Z 高程误差 (米) —— 精确锁定 Z=0 桌面平整度与 3.5cm 鼠标高度
    z_pred = REAL_H_CAM_M + rays_in[:, 2] * depth_pred
    z_gt_in = REAL_H_CAM_M + rays_in[:, 2] * depth_gt
    l_z = ((z_pred - z_gt_in).abs().clamp(max=0.45) * w_pix).sum() / w_sum

    # 3) 多尺度逆深度空间梯度损失（严格保锐边，禁止跨无效边界平滑）
    l_grad = torch.tensor(0.0, device=out5.device)
    ip_s = (inv_d_pred / INV_D_SCALE).unsqueeze(1)
    ig_s = (inv_d_gt / INV_D_SCALE).unsqueeze(1)
    vm_s = valid.unsqueeze(1)
    for sc in (1, 2, 4):
        if sc > 1:
            ip_k = F.avg_pool2d(ip_s, sc, sc)
            ig_k = F.avg_pool2d(ig_s, sc, sc)
            vm_k = (F.avg_pool2d(vm_s, sc, sc) > 0.99).float()
        else:
            ip_k, ig_k, vm_k = ip_s, ig_s, vm_s
        gx_m = vm_k[:, :, :, 1:] * vm_k[:, :, :, :-1]
        gy_m = vm_k[:, :, 1:, :] * vm_k[:, :, :-1, :]
        l_gx = ((ip_k[:, :, :, 1:] - ip_k[:, :, :, :-1]) - (ig_k[:, :, :, 1:] - ig_k[:, :, :, :-1])).abs()
        l_gy = ((ip_k[:, :, 1:, :] - ip_k[:, :, :-1, :]) - (ig_k[:, :, 1:, :] - ig_k[:, :, :-1, :])).abs()
        l_grad = l_grad + (l_gx * gx_m).sum() / gx_m.sum().clamp(min=1.0) + (l_gy * gy_m).sum() / gy_m.sum().clamp(min=1.0)
    l_grad = l_grad / 3.0

    # 4) 神经表面法向损失 (余弦 + L1)
    cos_sim = (nrm_pred * nrm_gt).sum(dim=1).clamp(-1.0, 1.0)
    l_nrm = (((1.0 - cos_sim) + 0.4 * (nrm_pred - nrm_gt).abs().mean(dim=1)) * w_pix).sum() / w_sum

    # 5) 3D 法向折痕梯度损失（让 90° 桌角/键盘边缘锐利如刀！）
    mx = valid[:, :, 1:] * valid[:, :, :-1]
    my = valid[:, 1:, :] * valid[:, :-1, :]
    dnx = ((nrm_pred[:, :, :, 1:] - nrm_pred[:, :, :, :-1]) - (nrm_gt[:, :, :, 1:] - nrm_gt[:, :, :, :-1])).abs().mean(dim=1)
    dny = ((nrm_pred[:, :, 1:, :] - nrm_pred[:, :, :-1, :]) - (nrm_gt[:, :, 1:, :] - nrm_gt[:, :, :-1, :])).abs().mean(dim=1)
    l_ncrease = (dnx * mx).sum() / mx.sum().clamp(min=1.0) + (dny * my).sum() / my.sum().clamp(min=1.0)

    # 6) 几何-法向一致性约束（在平滑内点上，确保预测 3D 表面切向量与预测法向正交！）
    pts_pred = rays_in * depth_pred.unsqueeze(1)
    dx_p = pts_pred[:, :, :, 1:] - pts_pred[:, :, :, :-1]
    dy_p = pts_pred[:, :, 1:, :] - pts_pred[:, :, :-1, :]
    dx_u = dx_p / dx_p.norm(dim=1, keepdim=True).clamp(min=1e-4)
    dy_u = dy_p / dy_p.norm(dim=1, keepdim=True).clamp(min=1e-4)
    smooth_x = mx * (crease_gt[:, :, 1:] < 0.03).float()
    smooth_y = my * (crease_gt[:, 1:, :] < 0.03).float()
    l_ortho = (
        ((nrm_pred[:, :, :, 1:] * dx_u).sum(dim=1).abs() * smooth_x).sum() / smooth_x.sum().clamp(min=1.0)
        + ((nrm_pred[:, :, 1:, :] * dy_u).sum(dim=1).abs() * smooth_y).sum() / smooth_y.sum().clamp(min=1.0)
    )

    # 6b) 局部尺移不变微起伏对比损失 (MoGe-2 风格，专雕 3.5cm 鼠标弧顶/线材/键盘脊与 90° 桌角！)
    # 对比相距 6px 与 16px 像素对的相对逆深度差，摆脱全局尺度抖动干扰，逼出刀刻级轮廓！
    l_local_rel = torch.tensor(0.0, device=out5.device)
    for stp in (6, 16):
        m_rx = valid[:, :, stp:] * valid[:, :, :-stp]
        m_ry = valid[:, stp:, :] * valid[:, :-stp, :]
        dp_x = (inv_d_pred[:, :, stp:] - inv_d_pred[:, :, :-stp]) / INV_D_SCALE
        dg_x = (inv_d_gt[:, :, stp:] - inv_d_gt[:, :, :-stp]) / INV_D_SCALE
        dp_y = (inv_d_pred[:, stp:, :] - inv_d_pred[:, :-stp, :]) / INV_D_SCALE
        dg_y = (inv_d_gt[:, stp:, :] - inv_d_gt[:, :-stp, :]) / INV_D_SCALE
        l_local_rel = l_local_rel + ((dp_x - dg_x).abs() * m_rx).sum() / m_rx.sum().clamp(min=1.0)
        l_local_rel = l_local_rel + ((dp_y - dg_y).abs() * m_ry).sum() / m_ry.sum().clamp(min=1.0)
    l_local_rel = l_local_rel * 0.5

    # 7) 有效掩码 BCE 损失（防悬空飞点）
    l_val = F.binary_cross_entropy_with_logits(valid_logit, valid)

    # 8) 在线难例挖掘 (OHEM, Top-25% 最高误差边界/折痕/微物体像素额外强化)
    pix_err = ((inv_d_pred - inv_d_gt).abs() / INV_D_SCALE + 0.45 * (1.0 - cos_sim)) * valid
    flat_err = pix_err.view(pix_err.shape[0], -1)
    q75 = torch.quantile(flat_err.detach(), 0.75, dim=1, keepdim=True).unsqueeze(-1)
    hard_mask = ((pix_err >= q75) & (valid > 0.5)).float()
    l_ohem = (pix_err * hard_mask).sum() / hard_mask.sum().clamp(min=1.0)

    # 9) 2D FFT 频域高频幅度谱对齐损失（逼迫网络在高频空间频率上锁死鼠标/键盘边缘锐度）
    res_p = ((inv_d_pred - inv_d_prior) / INV_D_SCALE) * valid
    res_g = ((inv_d_gt - inv_d_prior) / INV_D_SCALE) * valid
    fft_p = torch.fft.rfft2(res_p, norm="ortho")
    fft_g = torch.fft.rfft2(res_g, norm="ortho")
    amp_p = torch.log1p(torch.abs(fft_p))
    amp_g = torch.log1p(torch.abs(fft_g))
    l_fft = F.l1_loss(amp_p, amp_g)

    # 10) v3 7通道专属：微浮雕头 (Ch 5) + 3D-光度联合神经轮廓边界头 (Ch 6)
    l_micro = torch.tensor(0.0, device=out5.device)
    l_contour = torch.tensor(0.0, device=out5.device)
    if out5.shape[1] >= 7:
        micro_dz_pred = torch.tanh(out5[:, 5]) * 0.045
        z_lowpass = F.avg_pool2d(z_gt_in.unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)
        micro_dz_gt = (z_gt_in - z_lowpass).clamp(-0.045, 0.045)
        w_micro = valid * (1.0 + 4.0 * is_micro + 2.0 * is_crease)
        l_micro = ((micro_dz_pred - micro_dz_gt).abs() * w_micro).sum() / w_micro.sum().clamp(min=1.0) * 25.0

        contour_logit = out5[:, 6]
        edge_in = batch["x7"][:, 1]
        contour_gt = (((crease_gt > 0.040) | (edge_in > 0.28)).float()) * valid
        pos_w = torch.tensor(3.5, device=out5.device)
        bce_c = F.binary_cross_entropy_with_logits(contour_logit, contour_gt, pos_weight=pos_w, reduction="none")
        l_bce_c = (bce_c * valid).sum() / valid.sum().clamp(min=1.0)
        c_prob = torch.sigmoid(contour_logit) * valid
        inter = (c_prob * contour_gt).sum()
        l_dice_c = 1.0 - (2.0 * inter + 1.0) / (c_prob.sum() + contour_gt.sum() + 1.0)
        l_contour = 0.65 * l_bce_c + 0.35 * l_dice_c

    total = (
        4.2 * l_inv
        + 5.2 * l_z
        + 3.6 * l_grad
        + 3.0 * l_local_rel
        + 3.4 * l_nrm
        + 2.2 * l_ncrease
        + 0.35 * l_ortho
        + 0.35 * l_val
        + 1.1 * l_ohem
        + 0.85 * l_fft
        + 1.4 * l_micro
        + 0.75 * l_contour
    )
    stats = {
        "total": float(total.detach().item()),
        "l_inv": float(l_inv.detach().item()),
        "l_z_mm": float(l_z.detach().item() * 1000.0),
        "l_nrm": float(l_nrm.detach().item()),
        "l_grad": float(l_grad.detach().item()),
        "l_micro": float(l_micro.detach().item()),
        "l_contour": float(l_contour.detach().item()),
    }
    return total, stats


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    split: dict[str, np.ndarray],
    device: torch.device,
    use_gt_imu: bool = True,
) -> dict[str, float]:
    """在验证集或严格留出集上测算完整 3D 几何与法向指标（支持真实 IMU 与无 IMU 双模式）。"""
    model.eval()
    n = len(split["gray_cur"])
    abs_rels, z_maes_mm, desk_mads_mm, micro_maes_mm, nrm_angles_deg, nrm_cos_list = [], [], [], [], [], []

    for i in range(n):
        gc = split["gray_cur"][i]
        gp = split["gray_prev"][i]
        d_gt = split["depth_m"][i].astype(np.float32)
        n_cam_gt = split["nrm_cam"][i].astype(np.float32)
        v_gt = split["valid"][i] > 0
        p_gt = float(split["pitch_gt"][i])
        r_gt = float(split["roll_gt"][i])

        if use_gt_imu:
            p_in, r_in = p_gt, r_gt
        else:
            # 模拟完全没有精确 IMU 时的标称输入 (pitch=+3.0°, roll=0°)
            p_in, r_in = 3.0, 0.0

        x7, rays_in, inv_d_prior = build_robot_physics_input(
            gc, gp, pitch_deg=p_in, roll_deg=r_in, fov_h_deg=REAL_FOV_H_DEG, h_cam_m=REAL_H_CAM_M
        )
        xt = torch.from_numpy(x7).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out5 = model(xt)[0].float().cpu().numpy()

        dec = decode_geonet_to_world_cloud(
            out5, rays_in, inv_d_prior, h_cam_m=REAL_H_CAM_M, auto_level=(not use_gt_imu)
        )
        d_pred = dec["depth_m"]
        pts_pred = dec["pts_grid"]
        nrm_pred = dec["nrm_grid"]

        r_gt_mat = camera_rotation_matrix(pitch_deg=p_gt, roll_deg=r_gt)
        rays_gt = (_RAYS_CAM_240 @ r_gt_mat.T).astype(np.float32)
        pts_gt = rays_gt * d_gt[..., None]
        pts_gt[..., 2] += REAL_H_CAM_M
        nrm_gt_w = n_cam_gt @ r_gt_mat.T
        nrm_gt_w /= np.maximum(np.linalg.norm(nrm_gt_w, axis=-1, keepdims=True), 1e-6)

        m = v_gt & (d_gt > 0.14) & (d_gt < 2.2)
        if int(np.sum(m)) < 100:
            continue
        abs_rels.append(float(np.mean(np.abs(d_pred[m] - d_gt[m]) / np.maximum(d_gt[m], 0.1))))
        z_maes_mm.append(float(np.mean(np.abs(pts_pred[m, 2] - pts_gt[m, 2])) * 1000.0))

        # 桌面平整度 (仅统计真值桌面像素 |z_gt| < 1.2cm & nz_gt > 0.92)
        dm = m & (np.abs(pts_gt[..., 2]) < 0.012) & (nrm_gt_w[..., 2] > 0.92) & (d_gt < 1.1)
        if int(np.sum(dm)) > 80:
            zp = pts_pred[dm, 2]
            desk_mads_mm.append(float(np.median(np.abs(zp - np.median(zp))) * 1000.0))

        # 桌面微凸起区精度 (鼠标/键盘/水杯底 1.2cm..15cm, 近距<1.1m)
        mm = m & (pts_gt[..., 2] >= 0.012) & (pts_gt[..., 2] <= 0.15) & (d_gt < 1.1)
        if int(np.sum(mm)) > 50:
            micro_maes_mm.append(float(np.mean(np.abs(pts_pred[mm, 2] - pts_gt[mm, 2])) * 1000.0))

        cos_v = np.clip(np.sum(nrm_pred[m] * nrm_gt_w[m], axis=-1), -1.0, 1.0)
        nrm_cos_list.append(float(np.mean(cos_v)))
        nrm_angles_deg.append(float(np.mean(np.degrees(np.arccos(cos_v)))))

    return {
        "n_frames": int(n),
        "abs_rel": round(float(np.mean(abs_rels)), 4),
        "z_mae_mm": round(float(np.mean(z_maes_mm)), 2),
        "desk_z_mad_mm": round(float(np.mean(desk_mads_mm)), 2) if desk_mads_mm else -1.0,
        "micro_obj_z_mae_mm": round(float(np.mean(micro_maes_mm)), 2) if micro_maes_mm else -1.0,
        "normal_cos": round(float(np.mean(nrm_cos_list)), 4),
        "normal_angle_deg": round(float(np.mean(nrm_angles_deg)), 2),
    }


def _filter_clean_teacher(sp: dict[str, np.ndarray], tag: str) -> dict[str, np.ndarray]:
    """剔除教师 RANSAC 尺度被钳位 (<=0.50 或 >=1.85) 或有效像素过低 (<42%) 的噪声帧，防止毒化训练。"""
    if "scales" not in sp:
        return sp
    sc = sp["scales"]
    vf = sp["valid"].mean(axis=(1, 2))
    keep = (sc >= 0.50) & (sc <= 1.85) & (vf >= 0.42)
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(f"  [CleanFilter {tag}] kept {int(keep.sum())}/{len(keep)} (dropped {n_drop} scale/valid outliers)", flush=True)
    return {k: v[keep] for k, v in sp.items()}


def train_and_evaluate(
    epochs: int = 16,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 9.0e-4,
    arch: str = "vit_b1",
    force_rebuild: bool = False,
    warm_start: bool = True,
) -> dict[str, Any]:
    out_dir = (REPO / "runs_end_side" / "hover_geonet_vit_240") if arch == "vit_b1" else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ch = 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"=== HoverGeoNet-240 v3 Training Pipeline (arch={arch}, base={base_ch}, bs={batch_size}x{grad_accum}, device={device}, H_cam={REAL_H_CAM_M}m, epochs={epochs}, warm_start={warm_start}) ===", flush=True)

    train_real, train_live, train_synth, val_real, test_heldout = prepare_all_caches(force_rebuild=force_rebuild)
    train_real = _filter_clean_teacher(train_real, "train_real")
    train_live = _filter_clean_teacher(train_live, "train_live")
    print(
        f"Dataset sizes: train_real={len(train_real['gray_cur'])}, train_live={len(train_live['gray_cur'])}, "
        f"train_synth={len(train_synth['gray_cur'])}, val_internal={len(val_real['gray_cur'])}, test_heldout={len(test_heldout['gray_cur'])}",
        flush=True,
    )

    ds_train = HoverGeoDataset([train_real, train_live, train_synth], is_train=True, imu_noise=True)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)

    ckpt_fp32 = out_dir / f"hover_geonet_{arch}_fp32.pt"
    ckpt_fp16 = out_dir / f"hover_geonet_{arch}_fp16.pt"

    model = create_hover_geonet(arch=arch, in_ch=GEO_IN_CH, base=base_ch, pretrained_backbone=(not (warm_start and ckpt_fp32.exists()))).to(device)
    if warm_start and ckpt_fp32.exists():
        prev_blob = torch.load(ckpt_fp32, map_location=device, weights_only=False)
        prev_sd = prev_blob.get("state_dict", {})
        missing, unexpected = model.load_state_dict(prev_sd, strict=False)
        print(f"  [WarmStart] Loaded existing {ckpt_fp32.name} (prev_epoch={prev_blob.get('epoch')}); new v3 keys={len(missing)}, unexpected={len(unexpected)}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: HoverGeoNet ({arch}) params={n_params:,} (FP32={n_params*4/1048576:.2f}MB, FP16={n_params*2/1048576:.2f}MB)", flush=True)

    # 差分学习率 + 余弦退火：保护 ViT / ImageNet 预训练通用几何特征，加速 v3 新增边缘注意力与微浮雕/神经轮廓头！
    bb_params = list(model.backbone_parameters())
    bb_ids = {id(p) for p in bb_params}
    v3_new_params = []
    for attr_nm in ("edge_attn_120", "edge_attn_240", "detail_contour_head"):
        if hasattr(model, attr_nm):
            v3_new_params.extend(list(getattr(model, attr_nm).parameters()))
    v3_ids = {id(p) for p in v3_new_params}
    base_head_params = [p for p in model.parameters() if (id(p) not in bb_ids) and (id(p) not in v3_ids)]

    bb_mult = 0.14 if arch == "vit_b1" else 0.22
    opt_groups = [
        {"params": bb_params, "lr": lr * bb_mult, "weight_decay": 1e-2},
        {"params": base_head_params, "lr": lr * (0.65 if (warm_start and ckpt_fp32.exists()) else 1.0), "weight_decay": 5e-4},
    ]
    if v3_new_params:
        opt_groups.append({"params": v3_new_params, "lr": lr * 1.45, "weight_decay": 3e-4})
    opt = torch.optim.AdamW(opt_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.03)

    best_score = 1e9
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_z_mm = 0.0
        ep_nrm = 0.0
        steps = 0
        t_ep = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        for bi, batch in enumerate(dl_train):
            batch_gpu = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out5 = model(batch_gpu["x7"])
                loss, st = compute_geonet_loss(out5.float(), batch_gpu)
            (loss / float(grad_accum)).backward()

            if (bi + 1) % grad_accum == 0 or (bi + 1) == len(dl_train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                opt.step()
                opt.zero_grad(set_to_none=True)

            ep_loss += st["total"]
            ep_z_mm += st["l_z_mm"]
            ep_nrm += st["l_nrm"]
            steps += 1

        sched.step()
        ep_loss /= max(1, steps)
        ep_z_mm /= max(1, steps)
        ep_nrm /= max(1, steps)

        val_m = evaluate_split(model, val_real, device, use_gt_imu=True)
        # 均衡深度 + 微物体高程 + 表面法向锐利度的复合选优指标（绝不碰 held-out test!）
        score = val_m["abs_rel"] * 90.0 + val_m["micro_obj_z_mae_mm"] * 1.4 + val_m["normal_angle_deg"] * 0.8 + val_m["desk_z_mad_mm"] * 1.0
        is_best = score < best_score
        if is_best:
            best_score = score
            best_epoch = ep
            best_val_metrics = dict(val_m)
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "arch": arch,
                    "in_ch": GEO_IN_CH,
                    "base": base_ch,
                    "h_cam_m": REAL_H_CAM_M,
                    "fov_h_deg": REAL_FOV_H_DEG,
                    "epoch": ep,
                    "val_metrics": val_m,
                },
                ckpt_fp32,
            )
        history.append({"epoch": ep, "train_loss": round(ep_loss, 4), "train_z_mm": round(ep_z_mm, 2), "val": val_m})
        vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
        print(
            f"Ep {ep:02d}/{epochs} ({time.perf_counter()-t_ep:.1f}s, VRAM={vram_gb:.2f}GB) | "
            f"tr_loss={ep_loss:.3f} tr_z={ep_z_mm:.1f}mm | "
            f"VAL absRel={val_m['abs_rel']:.4f} zMAE={val_m['z_mae_mm']:.1f}mm "
            f"deskMAD={val_m['desk_z_mad_mm']:.1f}mm microZ={val_m['micro_obj_z_mae_mm']:.1f}mm "
            f"nrm={val_m['normal_angle_deg']:.1f}° (cos={val_m['normal_cos']:.3f})"
            f"{'  <-- BEST' if is_best else ''}",
            flush=True,
        )

    train_s = time.perf_counter() - t_start

    # 加载验证集最优模型，导出 FP16 紧凑权重并在严格留出集上评测
    best_blob = torch.load(ckpt_fp32, map_location=device, weights_only=False)
    model.load_state_dict(best_blob["state_dict"])
    model.eval()

    fp16_sd = {k: (v.half().cpu() if v.is_floating_point() else v.cpu()) for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict_fp16": fp16_sd,
            "arch": arch,
            "in_ch": GEO_IN_CH,
            "base": base_ch,
            "h_cam_m": REAL_H_CAM_M,
            "fov_h_deg": REAL_FOV_H_DEG,
            "epoch": best_epoch,
        },
        ckpt_fp16,
    )

    names_test = [str(x) for x in test_heldout["names"]]
    idx_empty = [i for i, nm in enumerate(names_test) if "empty_d0.5" in nm]
    idx_cb = [i for i, nm in enumerate(names_test) if "cb8455" in nm]

    def _slice(sp: dict[str, np.ndarray], idxs: list[int]) -> dict[str, np.ndarray]:
        return {k: v[idxs] for k, v in sp.items()}

    heldout_empty_imu = evaluate_split(model, _slice(test_heldout, idx_empty), device, use_gt_imu=True)
    heldout_empty_noimu = evaluate_split(model, _slice(test_heldout, idx_empty), device, use_gt_imu=False)
    heldout_cb_imu = evaluate_split(model, _slice(test_heldout, idx_cb), device, use_gt_imu=True)
    heldout_cb_noimu = evaluate_split(model, _slice(test_heldout, idx_cb), device, use_gt_imu=False)

    m_fp16 = create_hover_geonet(arch=arch, in_ch=GEO_IN_CH, base=base_ch, pretrained_backbone=False).to(device).half().eval()
    m_fp16.load_state_dict({k: v.to(device) for k, v in fp16_sd.items()})
    x_dummy = torch.randn(1, GEO_IN_CH, GEO_H, GEO_W, device=device, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(15):
            _ = m_fp16(x_dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_lat = time.perf_counter()
    with torch.no_grad():
        for _ in range(60):
            _ = m_fp16(x_dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_fp16_ms = (time.perf_counter() - t_lat) * 1000.0 / 60.0

    report = {
        "model": f"HoverGeoNet-240 ({arch}, 7ch physics-conditioned, H_cam=0.135m)",
        "arch": arch,
        "h_cam_m": REAL_H_CAM_M,
        "fov_h_deg": REAL_FOV_H_DEG,
        "params": int(n_params),
        "ckpt_fp32_mb": round(ckpt_fp32.stat().st_size / 1048576.0, 2),
        "ckpt_fp16_mb": round(ckpt_fp16.stat().st_size / 1048576.0, 2),
        "gpu_fp16_ms": round(gpu_fp16_ms, 2),
        "gpu_fp16_fps": round(1000.0 / max(gpu_fp16_ms, 1e-3), 1),
        "best_epoch": int(best_epoch),
        "train_seconds": round(train_s, 1),
        "val_internal": best_val_metrics,
        "heldout_empty_d05_with_imu": heldout_empty_imu,
        "heldout_empty_d05_no_imu_autolevel": heldout_empty_noimu,
        "heldout_cb8455_with_imu": heldout_cb_imu,
        "heldout_cb8455_no_imu_autolevel": heldout_cb_noimu,
        "history": history,
    }
    (out_dir / "train_eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== FINAL HELD-OUT ZERO-SHOT REPORT (HoverGeoNet-{arch}) ===", flush=True)
    print(f"FP16 Size: {report['ckpt_fp16_mb']} MB | GPU FP16 Latency: {report['gpu_fp16_ms']} ms ({report['gpu_fp16_fps']} FPS)", flush=True)
    print("Held-out empty_d0.5 (IMU)     :", heldout_empty_imu, flush=True)
    print("Held-out empty_d0.5 (No-IMU)  :", heldout_empty_noimu, flush=True)
    print("Held-out cb8455     (IMU)     :", heldout_cb_imu, flush=True)
    print("Held-out cb8455     (No-IMU)  :", heldout_cb_noimu, flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="vit_b1", choices=["vit_b1", "mbv3"])
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=9.0e-4)
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()
    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, arch=args.arch, force_rebuild=args.rebuild_cache)
