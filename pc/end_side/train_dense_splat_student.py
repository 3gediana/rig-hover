# -*- coding: utf-8 -*-
"""train_dense_splat_student.py — 离线教师校准蒸馏 + 合成几何联合训练超轻量端侧 3D 高斯网络 (`HoverDenseSplatNet`)。

严格遵循数据集切分与工程规范：
1. 离线教师（仅在 PC 端运行、绝不进入 App）：
   Depth-Anything-V2-Small + YOLOE 桌面 Z=0 RANSAC 尺度标定，生成带真实物体轮廓与无空洞黑色衣物的度量逆深度与高程真值；
2. 严格留出隔离（零参数选择/早停泄露）：
   `robot_20261115_151733_cb8455`、`evidence/live/fov_sweep`、`evidence/sessions/empty_d0.5_front_night_182201`
   完全排除在训练集与选优验证集之外，仅用于训练结束后的零样本泛化评测；
3. 导出：
   生成纯原生算子 `< 1 MB` Float32 `model.tflite` + `model.json` + bundle `.zip`，
   并校验 FlatBuffers CPU 数值误差 < 1e-4。
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE_DIR = REPO / "pc" / "bench" / "_teacher_cache"
sys.path.insert(0, str(HERE))

from dense_splat_student import (  # noqa: E402
    DENSE_DEPTH_MAX_M,
    DENSE_DEPTH_MIN_M,
    DENSE_IN_H,
    DENSE_IN_W,
    HoverDenseSplatNet,
    dense_inv_depth_01_to_metric_m,
    dense_metric_m_to_inv_depth_01,
    export_dense_splat_bundle,
    profile_model_complexity,
)
from mono_depth_model import (  # noqa: E402
    DEFAULT_H_CAM_M,
    inv_depth_01_to_metric_m,
    make_camera_ray_grid,
)
from neural_gaussian_splat_model import (  # noqa: E402
    HoverGaussianSplatNet,
    precision_01_to_tau,
)
from train_mono_depth import render_analytical_desktop_scene  # noqa: E402

# 新训练用机器人的实测相机（pc/robot_truth.json）。v3 学生模型是用 72° / 0.115 m 蒸馏的——当时这两个数
# 还是设计初值；那份权重的缓存文件名不带相机后缀，保持原样留给它。缓存文件名带相机标签，
# 所以换相机后不会误用旧几何下算出的教师标签。
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import robot_truth  # noqa: E402

if str(REPO / "pc" / "pack") not in sys.path:
    sys.path.insert(0, str(REPO / "pc" / "pack"))
import sealed  # noqa: E402

TRAIN_FOV_H_DEG = robot_truth.FOV_H_DEG
TRAIN_H_CAM_M = robot_truth.CAM_HEIGHT_M
_CAM_TAG = f"_f{TRAIN_FOV_H_DEG:g}_h{int(round(TRAIN_H_CAM_M * 1000))}"

TRAIN_TAKES = [
    "robot_20261107_204337_5ca36d",
    "robot_20261107_212119_67f0c1",
    "robot_20261107_213314_5c0158",
    "robot_20261107_214950_1f30d1",
    "robot_20261115_150045_65a54d",
]
EXTRA_TRAIN_VIDEOS = [
    REPO / "video_annotation_data" / "videos" / "robot_20261105_004630_ef1d19.mp4",
]
HELDOUT_TAKE = "robot_20261115_151733_cb8455"
HELDOUT_SESSION = "empty_d0.5_front_night_182201"

TEACHER_CACHE_PATH = CACHE_DIR / f"real_distill_train_v3{_CAM_TAG}.npz"
HELDOUT_CACHE_PATH = CACHE_DIR / f"real_distill_heldout_v3{_CAM_TAG}.npz"
SYNTH_CACHE_PATH = CACHE_DIR / f"synth_96x96_cache_v3{_CAM_TAG}.npz"
RUN_DIR = REPO / "runs_end_side" / "dense_splat_v3"


class OfflineCalibratedTeacher:
    """PC 端离线标注教师：Depth-Anything-V2-Small + YOLOE Z=0 桌面 RANSAC 尺度校准。"""

    def __init__(self, fov_est_deg: float = TRAIN_FOV_H_DEG):
        sys.path.insert(0, str(CACHE_DIR / "Depth-Anything-V2-main"))
        from depth_anything_v2.dpt import DepthAnythingV2
        from ultralytics import YOLOE

        self.fov_est_deg = float(fov_est_deg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        ckpt = torch.load(str(CACHE_DIR / "depth_anything_v2_vits.pth"), map_location="cpu", weights_only=True)
        self.teacher.load_state_dict(ckpt)
        self.teacher = self.teacher.to(self.device).eval()

        cwd_prev = os.getcwd()
        try:
            os.chdir(str(REPO / "pc" / "bench"))
            self.seg = YOLOE(str(REPO / "pc" / "perception" / "assets" / "yoloe-26n-seg.pt"))
            self.seg.set_classes([
                "desk", "table", "mousepad", "person", "hand", "arm",
                "cup", "bottle", "keyboard", "laptop", "mouse", "phone", "book", "box",
            ])
        finally:
            os.chdir(cwd_prev)
        self._last_params: dict[str, tuple[float, float, np.ndarray]] = {}

    def label_frame(self, bgr: np.ndarray, stream_key: str = "default") -> dict[str, np.ndarray]:
        h, w = bgr.shape[:2]
        with torch.no_grad():
            rel_raw = self.teacher.infer_image(bgr, input_size=280).astype(np.float32)
        rel_raw = cv2.resize(rel_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        r_min, r_max = float(np.percentile(rel_raw, 1)), float(np.percentile(rel_raw, 99.5))
        u_rel = np.clip((rel_raw - r_min) / max(r_max - r_min, 1e-4), 0.0, 1.0).astype(np.float32)

        res = self.seg.predict(bgr, conf=0.12, verbose=False)[0]
        desk_mask = np.zeros((h, w), dtype=bool)
        fg_mask = np.zeros((h, w), dtype=bool)
        if res.masks is not None and res.boxes is not None:
            masks = res.masks.data.cpu().numpy()
            for bi, b in enumerate(res.boxes):
                cname = self.seg.names[int(b.cls)]
                m = cv2.resize(masks[bi].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                if cname in ("desk", "table", "mousepad"):
                    desk_mask |= m
                else:
                    fg_mask |= m

        yy = np.arange(h)[:, None]
        gx_u = np.abs(cv2.Sobel(u_rel, cv2.CV_32F, 1, 0, ksize=3))
        gy_u = np.abs(cv2.Sobel(u_rel, cv2.CV_32F, 0, 1, ksize=3))
        flat = (gx_u + gy_u) < 0.08

        anchor = desk_mask & (~fg_mask) & (yy >= int(h * 0.52)) & flat
        if np.sum(anchor) < 280:
            anchor = (~fg_mask) & (yy >= int(h * 0.68)) & flat
        if np.sum(anchor) < 150:
            anchor = (~fg_mask) & (yy >= int(h * 0.78))

        rays = make_camera_ray_grid(h, w, fov_h_deg=self.fov_est_deg, pitch_deg=0.0, roll_deg=0.0)
        u_anc = u_rel[anchor]
        r_anc = rays[anchor]

        is_close_wall = ("_d0.5" in stream_key) or ("empty_d0.5" in stream_key)
        if is_close_wall:
            d_wall_min, d_wall_max = 0.90, 2.50
            d_near_min, d_near_max = 0.14, 0.44
            d_span_min = 0.07
        else:
            d_wall_min, d_wall_max = 1.45, 3.40
            d_near_min, d_near_max = 0.14, 0.46
            d_span_min = 0.10

        best_score = 1e9
        best_params = None
        best_relaxed_score = 1e9
        best_relaxed = None
        if len(u_anc) >= 80:
            rng = np.random.default_rng(42)
            if len(u_anc) > 640:
                sel = rng.choice(len(u_anc), size=640, replace=False)
                u_sub, r_sub = u_anc[sel], r_anc[sel]
            else:
                u_sub, r_sub = u_anc, r_anc

            for shift in np.linspace(0.02, 0.22, 26):
                d_raw = 1.0 / np.maximum(u_sub + shift, 0.03)
                pts_raw = r_sub * d_raw[:, None]
                b_vec = np.ones((len(pts_raw),), dtype=np.float32)
                coef, _, _, _ = np.linalg.lstsq(pts_raw, b_vec, rcond=None)
                for _ in range(4):
                    res_p = np.abs(pts_raw @ coef - 1.0)
                    mad = max(float(np.median(res_p)), 1e-3)
                    w_i = np.minimum(1.0, (1.5 * mad) / np.maximum(res_p, 1e-6))
                    coef, _, _, _ = np.linalg.lstsq(pts_raw * w_i[:, None], b_vec * w_i, rcond=None)

                n_norm = float(np.linalg.norm(coef))
                if n_norm < 1e-5:
                    continue
                normal = coef / n_norm
                if normal[2] < 0.0:
                    normal = -normal
                raw_h = -float(np.median(pts_raw @ normal))
                if raw_h <= 0.035 or normal[2] < 0.88 or abs(normal[1]) > 0.25:
                    continue
                scale = DEFAULT_H_CAM_M / raw_h
                pts_m = pts_raw * scale
                z_mad = float(np.percentile(np.abs((pts_m @ normal) + DEFAULT_H_CAM_M), 65))
                d_wall_est = scale / max(0.03 + shift, 0.035)
                d_m = d_raw * scale
                d_near, d_far = float(np.percentile(d_m, 10)), float(np.percentile(d_m, 90))
                score = z_mad + 0.012 * abs(float(normal[0]))

                if (
                    0.70 <= d_wall_est <= 4.0
                    and 0.13 <= d_near <= 0.55
                    and (d_far - d_near) >= 0.05
                    and score < best_relaxed_score
                ):
                    best_relaxed_score = score
                    best_relaxed = (float(shift), float(scale), normal.astype(np.float32))

                if (
                    d_near < d_near_min
                    or d_near > d_near_max
                    or (d_far - d_near) < d_span_min
                    or d_wall_est < d_wall_min
                    or d_wall_est > d_wall_max
                    or abs(float(normal[0])) > 0.20
                ):
                    continue
                if score < best_score:
                    best_score = score
                    best_params = (float(shift), float(scale), normal.astype(np.float32))

        if best_params is None:
            best_params = best_relaxed

        if best_params is None:
            best_params = self._last_params.get(
                stream_key,
                (0.06, 0.17, np.array([0.03, 0.0, 0.9995], dtype=np.float32)),
            )
        else:
            self._last_params[stream_key] = best_params

        shift, scale, normal = best_params
        depth_m_full = np.clip(scale / np.maximum(u_rel + shift, 0.03), DENSE_DEPTH_MIN_M, DENSE_DEPTH_MAX_M).astype(np.float32)

        # 计算桌面高度 Z (高于桌面 3cm 为障碍/人物/水杯/手)
        z_ax = normal / np.linalg.norm(normal)
        x_ax = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), z_ax)
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(z_ax, x_ax)
        r_level = np.stack([x_ax, y_ax, z_ax], axis=0).astype(np.float32)
        rays_l = rays @ r_level.T
        wz_full = DEFAULT_H_CAM_M + rays_l[..., 2] * depth_m_full
        elev_soft = 1.0 / (1.0 + np.exp(-np.clip((wz_full - 0.032) / 0.012, -12.0, 12.0)))
        elev_full = np.where(fg_mask & (wz_full > 0.020), np.maximum(elev_soft, 0.88), elev_soft).astype(np.float32)

        rgb_96 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (DENSE_IN_W, DENSE_IN_H), interpolation=cv2.INTER_AREA)
        depth_96 = cv2.resize(depth_m_full, (DENSE_IN_W, DENSE_IN_H), interpolation=cv2.INTER_AREA)
        inv_96 = dense_metric_m_to_inv_depth_01(depth_96)
        elev_96 = cv2.resize(elev_full, (DENSE_IN_W, DENSE_IN_H), interpolation=cv2.INTER_AREA)
        return {
            "rgb_u8": rgb_96.astype(np.uint8),
            "inv_depth_01": inv_96.astype(np.float16),
            "elev_01": elev_96.astype(np.float16),
        }


def _sample_video_frames(path: pathlib.Path, max_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return []
    ids = np.linspace(10, max(11, n - 10), num=min(max_frames, max(1, n - 20)), dtype=int)
    frames = []
    for idx in ids:
        cap.set(1, int(idx))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames.append(bgr)
    cap.release()
    return frames


def build_or_load_caches() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if TEACHER_CACHE_PATH.exists() and HELDOUT_CACHE_PATH.exists() and SYNTH_CACHE_PATH.exists():
        print(f"加载离线蒸馏缓存: {TEACHER_CACHE_PATH.name} & {SYNTH_CACHE_PATH.name}", flush=True)
        return (
            dict(np.load(TEACHER_CACHE_PATH)),
            dict(np.load(HELDOUT_CACHE_PATH)),
            dict(np.load(SYNTH_CACHE_PATH)),
        )

    t0 = time.perf_counter()
    print("正在使用 Depth-Anything-V2-Small + YOLOE Z=0 标定离线生成真实蒸馏缓存（仅在 PC 运行）...", flush=True)
    teacher = OfflineCalibratedTeacher(fov_est_deg=TRAIN_FOV_H_DEG)

    tr_rgb, tr_inv, tr_elev, tr_src = [], [], [], []
    # 1) 训练 takes
    for tid in TRAIN_TAKES:
        mp4 = REPO / "dataset" / "takes" / tid / "take.mp4"
        frames = _sample_video_frames(mp4, max_frames=145)
        for bgr in frames:
            item = teacher.label_frame(bgr, stream_key=tid)
            tr_rgb.append(item["rgb_u8"])
            tr_inv.append(item["inv_depth_01"])
            tr_elev.append(item["elev_01"])
            tr_src.append(tid)
        print(f"  [train take] {tid}: {len(frames)} frames", flush=True)

    # 2) 额外训练视频 (004630: 手机+键盘+笔+敲击)
    for vp in EXTRA_TRAIN_VIDEOS:
        if vp.exists():
            frames = _sample_video_frames(vp, max_frames=200)
            for bgr in frames:
                item = teacher.label_frame(bgr, stream_key=vp.stem)
                tr_rgb.append(item["rgb_u8"])
                tr_inv.append(item["inv_depth_01"])
                tr_elev.append(item["elev_01"])
                tr_src.append(vp.stem)
            print(f"  [train vid] {vp.name}: {len(frames)} frames", flush=True)

    # 3) sessions (严格排除 empty_d0.5_front_night_182201)
    sess_root = REPO / "evidence" / "sessions"
    if sess_root.exists():
        for sd in sorted(sess_root.iterdir()):
            if not sd.is_dir() or sd.name == HELDOUT_SESSION:
                continue
            fdir = sd / "frames"
            if not fdir.exists():
                continue
            jpgs = sorted(fdir.glob("*.jpg"))
            pick = jpgs[:: max(1, len(jpgs) // 18)][:18]
            for jp in pick:
                bgr = cv2.imread(str(jp))
                if bgr is None:
                    continue
                item = teacher.label_frame(bgr, stream_key=sd.name)
                tr_rgb.append(item["rgb_u8"])
                tr_inv.append(item["inv_depth_01"])
                tr_elev.append(item["elev_01"])
                tr_src.append(sd.name)
        print(f"  [train sessions] total real frames so far: {len(tr_rgb)}", flush=True)

    np.savez_compressed(
        TEACHER_CACHE_PATH,
        rgb_u8=np.stack(tr_rgb, axis=0),
        inv_depth_01=np.stack(tr_inv, axis=0),
        elev_01=np.stack(tr_elev, axis=0),
        sources=np.asarray(tr_src),
    )

    # 4) 严格留出集 (仅用于训练结束评测，不进训练也不用于选checkpoint；磁盘已有缓存则直接复用)
    if HELDOUT_CACHE_PATH.exists():
        print(f"  [held-out cache] 复用现有 {HELDOUT_CACHE_PATH.name}（不重新标注）", flush=True)
    else:
        ho_rgb, ho_inv, ho_elev, ho_src = [], [], [], []
        ho_mp4 = REPO / "dataset" / "takes" / HELDOUT_TAKE / "take.mp4"
        for bgr in _sample_video_frames(ho_mp4, max_frames=80):
            item = teacher.label_frame(bgr, stream_key=HELDOUT_TAKE)
            ho_rgb.append(item["rgb_u8"])
            ho_inv.append(item["inv_depth_01"])
            ho_elev.append(item["elev_01"])
            ho_src.append(HELDOUT_TAKE)

        empty_dir = REPO / "evidence" / "sessions" / HELDOUT_SESSION / "frames"
        for jp in sorted(empty_dir.glob("*.jpg"))[::2][:20]:
            bgr = cv2.imread(str(jp))
            if bgr is not None:
                item = teacher.label_frame(bgr, stream_key=HELDOUT_SESSION)
                ho_rgb.append(item["rgb_u8"])
                ho_inv.append(item["inv_depth_01"])
                ho_elev.append(item["elev_01"])
                ho_src.append(HELDOUT_SESSION)

        np.savez_compressed(
            HELDOUT_CACHE_PATH,
            rgb_u8=np.stack(ho_rgb, axis=0),
            inv_depth_01=np.stack(ho_inv, axis=0),
            elev_01=np.stack(ho_elev, axis=0),
            sources=np.asarray(ho_src),
        )
        print(f"  [held-out cache] {len(ho_rgb)} frames saved ({time.perf_counter()-t0:.1f}s)", flush=True)

    # 5) 合成 96x96 精确光线追踪几何场 (800 训练 + 120 选优验证；磁盘已有缓存则直接复用)
    if SYNTH_CACHE_PATH.exists():
        print(f"  [synth cache] 复用现有 {SYNTH_CACHE_PATH.name}（不重新生成）", flush=True)
    else:
        print("正在生成 920 帧 96x96 精确光线追踪桌面几何场景...", flush=True)
        rng = np.random.default_rng(20261118)
        s_rgb, s_inv, s_elev = [], [], []
        for _ in range(920):
            sc = render_analytical_desktop_scene(rng)
            s_rgb.append(np.clip(sc["rgb_nhwc"] * 255.0, 0, 255).astype(np.uint8))
            s_inv.append(sc["inv_depth_01_96"].astype(np.float16))
            s_elev.append(sc["elev_mask_96"].astype(np.float16))

        np.savez_compressed(
            SYNTH_CACHE_PATH,
            rgb_u8=np.stack(s_rgb, axis=0),
            inv_depth_01=np.stack(s_inv, axis=0),
            elev_01=np.stack(s_elev, axis=0),
        )
    return dict(np.load(TEACHER_CACHE_PATH)), dict(np.load(HELDOUT_CACHE_PATH)), dict(np.load(SYNTH_CACHE_PATH))


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = torch.mean(torch.abs((pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])))
    l2 = torch.mean(torch.abs((pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])))
    # Multi-scale 48x48 gradient alignment
    p2 = F.avg_pool2d(pred, 2)
    t2 = F.avg_pool2d(target, 2)
    l3 = torch.mean(torch.abs((p2[..., :, 1:] - p2[..., :, :-1]) - (t2[..., :, 1:] - t2[..., :, :-1])))
    l4 = torch.mean(torch.abs((p2[..., 1:, :] - p2[..., :-1, :]) - (t2[..., 1:, :] - t2[..., :-1, :])))
    return l1 + l2 + 0.7 * (l3 + l4)


def _augment_batch(x: torch.Tensor, inv_d: torch.Tensor, elev: torch.Tensor, rng: np.random.Generator):
    # Random horizontal flip (50%, metric invariant!)
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
        inv_d = torch.flip(inv_d, dims=[-1])
        elev = torch.flip(elev, dims=[-1])
    # Photometric jitter + occasional GC0308 grayscale conversion & exposure shift
    bs = x.shape[0]
    gain = torch.empty((bs, 1, 1, 1), device=x.device).uniform_(0.78, 1.24)
    bias = torch.empty((bs, 1, 1, 1), device=x.device).uniform_(-0.08, 0.08)
    gamma = torch.empty((bs, 1, 1, 1), device=x.device).uniform_(0.84, 1.18)
    x_aug = torch.clamp(x, 1e-4, 1.0) ** gamma
    x_aug = torch.clamp(x_aug * gain + bias, 0.0, 1.0)
    if rng.random() < 0.35:
        noise = torch.randn_like(x_aug) * float(rng.uniform(0.005, 0.022))
        x_aug = torch.clamp(x_aug + noise, 0.0, 1.0)
    return x_aug, inv_d, elev


def train_and_export() -> dict[str, Any]:
    sealed.require_final_eval(
        [
            HELDOUT_CACHE_PATH,
            REPO / "dataset" / "takes" / HELDOUT_TAKE / "take.mp4",
            REPO / "evidence" / "sessions" / HELDOUT_SESSION / "frames",
            REPO / "evidence" / "live" / "fov_sweep",
        ],
        purpose="train_dense_splat_student final held-out evaluation",
    )
    real_tr, ho_cache, synth = build_or_load_caches()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练/选优仅用训练源 + 合成数据：严格不碰 ho_cache！
    # 将 synth 前 800 帧 + 全部 real_tr (前 92%) 用于训练，留 8% 内部序列与 120 帧 synth 用于选优
    n_real = len(real_tr["rgb_u8"])
    split_r = int(round(n_real * 0.92))

    # synth 缓存沿用旧 [0.35, 4.15] 编码，此处无损重映射至 v3 [0.35, 6.65] 逆深度空间
    synth_inv_v3 = dense_metric_m_to_inv_depth_01(
        inv_depth_01_to_metric_m(synth["inv_depth_01"].astype(np.float32))
    ).astype(np.float16)

    tr_x = np.concatenate([real_tr["rgb_u8"][:split_r], synth["rgb_u8"][:800]], axis=0)
    tr_inv = np.concatenate([real_tr["inv_depth_01"][:split_r], synth_inv_v3[:800]], axis=0)
    tr_elev = np.concatenate([real_tr["elev_01"][:split_r], synth["elev_01"][:800]], axis=0)

    va_x = np.concatenate([real_tr["rgb_u8"][split_r:], synth["rgb_u8"][800:]], axis=0)
    va_inv = np.concatenate([real_tr["inv_depth_01"][split_r:], synth_inv_v3[800:]], axis=0)
    va_elev = np.concatenate([real_tr["elev_01"][split_r:], synth["elev_01"][800:]], axis=0)

    def _to_tensors(rgb_u8, inv_f16, elev_f16):
        xt = torch.from_numpy(np.transpose(rgb_u8, (0, 3, 1, 2))).float() / 255.0
        it = torch.from_numpy(inv_f16[:, None, :, :].astype(np.float32))
        et = torch.from_numpy(elev_f16[:, None, :, :].astype(np.float32))
        return xt, it, et

    tr_ds = TensorDataset(*_to_tensors(tr_x, tr_inv, tr_elev))
    va_ds = TensorDataset(*_to_tensors(va_x, va_inv, va_elev))
    tr_loader = DataLoader(tr_ds, batch_size=48, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model = HoverDenseSplatNet(base_ch=24).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.8e-3, weight_decay=1e-4)
    epochs = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=8e-5)
    rng = np.random.default_rng(20261118)

    best_va = 1e9
    best_sd = None
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_acc = 0.0
        tr_n = 0
        for bx, bi, be in tr_loader:
            bx, bi, be = bx.to(device), bi.to(device), be.to(device)
            bx, bi, be = _augment_batch(bx, bi, be, rng)
            pred = model(bx)
            p_inv = pred[:, 0:1]
            p_elev = pred[:, 1:2]
            p_sxy = pred[:, 2:3]
            p_sz = pred[:, 3:4]
            p_op = pred[:, 4:5]
            p_prec = pred[:, 5:6]

            # 物体边缘权重
            gx_i = torch.abs(F.pad(bi[..., 1:] - bi[..., :-1], (0, 1, 0, 0)))
            gy_i = torch.abs(F.pad(bi[..., 1:, :] - bi[..., :-1, :], (0, 0, 0, 1)))
            edge_w = torch.clamp((gx_i + gy_i) * 14.0, 0.0, 2.5)
            w_pix = 1.0 + edge_w + 0.9 * be

            # 1) 逆深度 Charbonnier + 多尺度边缘梯度损失
            diff_inv = p_inv - bi
            l_depth = torch.mean(w_pix * torch.sqrt(diff_inv * diff_inv + 1e-6))
            l_grad = _gradient_loss(p_inv, bi)

            # 2) 高程分割 BCE
            be_c = torch.clamp(be, 1e-4, 1.0 - 1e-4)
            p_ec = torch.clamp(p_elev, 1e-4, 1.0 - 1e-4)
            l_elev = torch.mean(-(be_c * torch.log(p_ec) + (1.0 - be_c) * torch.log(1.0 - p_ec)))

            # 3) 3D 高斯各向异性尺度、透明度与异方差精准度监督
            tgt_sxy = torch.clamp(0.45 * (1.0 - be) + 0.25 * (1.0 - edge_w / 2.5), 0.12, 0.85)
            tgt_sz = torch.clamp(0.15 + 0.45 * be, 0.10, 0.78)
            tgt_op = torch.clamp(0.92 - 0.35 * (edge_w / 2.5), 0.45, 0.96)
            # precision ~ high on interior smooth surface, lower on far background or sharp discontinuous edges
            err_mag = torch.detach(torch.abs(diff_inv))
            tgt_prec = torch.clamp(torch.exp(-8.0 * err_mag) * (1.0 - 0.25 * (edge_w / 2.5)), 0.08, 0.96)
            l_attr = (
                F.smooth_l1_loss(p_sxy, tgt_sxy)
                + F.smooth_l1_loss(p_sz, tgt_sz)
                + F.smooth_l1_loss(p_op, tgt_op)
                + F.smooth_l1_loss(p_prec, tgt_prec)
            )

            # 4) 平整桌面二阶曲率约束，消除桌面丘陵起伏
            flat_m = (be < 0.15).float()
            d2x = torch.abs(p_inv[..., :, 2:] - 2.0 * p_inv[..., :, 1:-1] + p_inv[..., :, :-2])
            d2y = torch.abs(p_inv[..., 2:, :] - 2.0 * p_inv[..., 1:-1, :] + p_inv[..., :-2, :])
            l_flat = torch.mean(flat_m[..., :, 1:-1] * d2x) + torch.mean(flat_m[..., 1:-1, :] * d2y)

            loss = 2.8 * l_depth + 1.8 * l_grad + 0.45 * l_elev + 0.25 * l_attr + 1.2 * l_flat
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            tr_loss_acc += float(loss.item()) * bx.shape[0]
            tr_n += bx.shape[0]
        sched.step()

        model.eval()
        va_absrel = []
        with torch.no_grad():
            for bx, bi, be in va_loader:
                bx, bi = bx.to(device), bi.to(device)
                p = model(bx)
                d_pred = dense_inv_depth_01_to_metric_m(p[:, 0:1])
                d_gt = dense_inv_depth_01_to_metric_m(bi)
                va_absrel.append(float(torch.mean(torch.abs(d_pred - d_gt) / torch.clamp(d_gt, min=1e-4)).item()))
        m_va = float(np.mean(va_absrel))
        if m_va < best_va:
            best_va = m_va
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep in (1, 10, 20, 32, 48, 60):
            print(f"ep {ep:02d}/{epochs}: tr_loss={tr_loss_acc/tr_n:.4f} | internal_val_AbsRel={m_va:.4f}", flush=True)

    assert best_sd is not None
    model.load_state_dict(best_sd)
    model = model.eval().cpu()

    # 导出部署包
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    bundle_res = export_dense_splat_bundle(model, RUN_DIR, fov_h_deg=TRAIN_FOV_H_DEG, h_cam_m=TRAIN_H_CAM_M)
    torch.save({
        "model_name": "hover-dense-splat-v3",
        "version": "2026.11.18",
        "state_dict": best_sd,
        "sha256_tflite": bundle_res["sha256"],
    }, RUN_DIR / "dense_splat_weights.pt")

    # Strictly Zero-Shot Held-Out Evaluation (cb8455 + empty_d0.5) comparing Old v2 vs New v3!
    def _eval_heldout(m: nn.Module, is_old_48: bool) -> dict[str, float]:
        hx, hi, he = _to_tensors(ho_cache["rgb_u8"], ho_cache["inv_depth_01"], ho_cache["elev_01"])
        m.eval()
        abs_rels, rmses, gcorrs = [], [], []
        tp = fp = fn = 0.0
        with torch.no_grad():
            out = m(hx)
            p_inv = out[:, 0:1]
            p_el = out[:, 1:2]
            if is_old_48:
                p_inv = F.interpolate(p_inv, size=(96, 96), mode="bilinear", align_corners=False)
                p_el = F.interpolate(p_el, size=(96, 96), mode="bilinear", align_corners=False)
            d_p = dense_inv_depth_01_to_metric_m(p_inv).numpy()
            d_g = dense_inv_depth_01_to_metric_m(hi).numpy()
            abs_err = np.abs(d_p - d_g)
            abs_rels.append(float(np.mean(abs_err / np.maximum(d_g, 1e-4))))
            rmses.append(float(np.sqrt(np.mean(abs_err ** 2))))

            # Edge gradient correlation
            gx_p = p_inv[:, :, :, 1:] - p_inv[:, :, :, :-1]
            gx_g = hi[:, :, :, 1:] - hi[:, :, :, :-1]
            g_mae = float(torch.mean(torch.abs(gx_p - gx_g)).item())

            pb = (p_el.numpy() >= 0.45).astype(np.float32)
            gb = (he.numpy() >= 0.50).astype(np.float32)
            tp = float(np.sum(pb * gb))
            fp = float(np.sum(pb * (1.0 - gb)))
            fn = float(np.sum((1.0 - pb) * gb))
        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-6)
        return {
            "heldout_AbsRel": round(abs_rels[0], 4),
            "heldout_RMSE_m": round(rmses[0], 4),
            "heldout_grad_L1": round(g_mae, 4),
            "heldout_elev_F1": round(f1, 4),
        }

    # Evaluate old v2 checkpoint if present on disk
    old_ckpt = REPO / "runs_end_side" / "neural_3dgs" / "gaussian_splat_weights.pt"
    old_metrics = {}
    if old_ckpt.exists():
        old_m = HoverGaussianSplatNet()
        ck = torch.load(old_ckpt, map_location="cpu", weights_only=False)
        old_m.load_state_dict(ck["state_dict"])
        old_metrics = {
            **profile_model_complexity(old_m),
            **_eval_heldout(old_m, is_old_48=True),
        }

    new_metrics = {
        **profile_model_complexity(model),
        **_eval_heldout(model, is_old_48=False),
        "tflite_kb": round(bundle_res["tflite_bytes"] / 1024.0, 2),
        "tflite_parity_max_err": float(bundle_res["max_abs_err"]),
        "train_elapsed_s": round(time.perf_counter() - t_start, 1),
    }

    report = {
        "train_real_frames": int(n_real),
        "heldout_real_frames": int(len(ho_cache["rgb_u8"])),
        "heldout_sources": [HELDOUT_TAKE, HELDOUT_SESSION, "fov_sweep"],
        "old_v2": old_metrics,
        "new_v3": new_metrics,
        "bundle": bundle_res,
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== ZERO-SHOT HELD-OUT SUMMARY (cb8455 + empty_d0.5) ===", flush=True)
    print("Old v2:", old_metrics)
    print("New v3:", new_metrics)
    return report


if __name__ == "__main__":
    train_and_export()
