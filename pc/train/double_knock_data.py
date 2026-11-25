# -*- coding: utf-8 -*-
"""double_knock_data.py — 多源接触脉冲定位与窗口级“敲击两下”数据集构建。

核心职责：
1. 融合 YOLO26n-pose 腕部竖直运动学 (wy, vy, ay_rebound) 与手桌局部帧差能量，
   在粗粒度 knock 区间内精确定位单次桌面接触脉冲时刻；
2. 构建双监督目标：逐帧接触脉冲热图 y_imp(t) ∈ [0,1] + 窗口动作计数类 y_cnt ∈ {0,1,2,3}
   (0=non_knock, 1=single_knock, 2=double_knock, 3=multi_knock)；
3. 仅在训练折内执行运动学一致的脉冲组合增强（严禁触碰留出测试 take）；
4. 挖掘本地真机录制 (evidence/sessions) 与公开视频数据集 (Charades 桌面动作 + UCF-101)
   构建跨数据集零误报压力测试集。
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_PACK_DIR = Path(__file__).resolve().parents[1] / "pack"
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

from cut_knock_candidates import contact_sheet  # noqa: E402
import sealed  # noqa: E402
from interval_data import (  # noqa: E402
    CLASS_TO_IDX,
    IGNORE,
    Interval,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
)

COUNT_CLASSES = ("non_knock", "single_knock", "double_knock", "multi_knock")
COUNT_TO_IDX = {name: i for i, name in enumerate(COUNT_CLASSES)}
KNOCK_IDX = CLASS_TO_IDX["knock"]


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, float(np.std(x)), 1e-5)
    return np.clip((x - med) / scale, -5.0, 8.0)


def extract_kinematics_and_energy(
    frames: np.ndarray,
    boxes: np.ndarray | list | None,
    fps: float = 20.0,
    orig_size: float = 240.0,
) -> tuple[np.ndarray, np.ndarray]:
    """提取 8 维距离无关姿态运动学特征与逐帧接触冲击得分。

    8 维运动学通道定义:
        0: wx_norm     腕部水平中心 ∈ [-1, 1]
        1: wy_norm     腕部竖直中心 ∈ [-1, 1]（向下为正）
        2: vy          竖直下落速度 (Δwy * fps / 2)，截断 ±4
        3: ay_rebound  触底回弹减速度 max(0, vy[t] - vy[t+1])
        4: vx          水平速度 (Δwx * fps / 2)，截断 ±4
        5: box_valid   当前帧是否有有效腕点 ROI 框 (0/1)
        6: global_diff 全局归一化帧差能量
        7: roi_diff    手桌局部 ROI 归一化帧差能量
    """
    T, H, W = frames.shape
    fps = float(max(fps, 1.0))
    wx = np.zeros(T, dtype=np.float32)
    wy = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=np.float32)

    if boxes is not None:
        for t in range(min(T, len(boxes))):
            b = boxes[t]
            if b is None:
                continue
            arr = np.asarray(b, dtype=np.float32).reshape(-1)
            if arr.size >= 4 and np.isfinite(arr[:4]).all():
                cx = 0.5 * (float(arr[0]) + float(arr[2])) / max(orig_size, 1.0)
                cy = 0.5 * (float(arr[1]) + float(arr[3])) / max(orig_size, 1.0)
                wx[t] = float(np.clip(cx * 2.0 - 1.0, -1.0, 1.0))
                wy[t] = float(np.clip(cy * 2.0 - 1.0, -1.0, 1.0))
                valid[t] = 1.0

    # 对短暂丢框做线性插值，避免速度差分出现伪尖峰
    valid_idx = np.where(valid > 0.5)[0]
    if valid_idx.size >= 2:
        all_idx = np.arange(T)
        wx = np.interp(all_idx, valid_idx, wx[valid_idx]).astype(np.float32)
        wy = np.interp(all_idx, valid_idx, wy[valid_idx]).astype(np.float32)
    elif valid_idx.size == 1:
        wx[:] = wx[valid_idx[0]]
        wy[:] = wy[valid_idx[0]]

    vy = np.zeros(T, dtype=np.float32)
    vx = np.zeros(T, dtype=np.float32)
    if T > 1:
        vy[1:] = np.clip((wy[1:] - wy[:-1]) * (fps * 0.5), -4.0, 4.0)
        vx[1:] = np.clip((wx[1:] - wx[:-1]) * (fps * 0.5), -4.0, 4.0)

    ay_rebound = np.zeros(T, dtype=np.float32)
    if T > 2:
        # 敲击触底瞬间：前一刻快速下探 (vy > 0)，下一刻停止或回弹 (vy 骤降或变负)
        ay_rebound[:-1] = np.clip(np.maximum(0.0, vy[:-1] - vy[1:]) * 1.5, 0.0, 6.0)

    global_diff = np.zeros(T, dtype=np.float32)
    roi_diff = np.zeros(T, dtype=np.float32)
    if T > 1:
        f_int = frames.astype(np.int16)
        adiff = np.abs(f_int[1:] - f_int[:-1]).astype(np.float32) / 255.0
        global_diff[1:] = adiff.mean(axis=(1, 2)) * 10.0
        for t in range(1, T):
            if valid[t] > 0.5:
                cx_px = int(round((wx[t] * 0.5 + 0.5) * W))
                cy_px = int(round((wy[t] * 0.5 + 0.5) * H))
                r = max(12, H // 4)
                x1, x2 = max(0, cx_px - r), min(W, cx_px + r)
                y1, y2 = max(0, cy_px - r), min(H, cy_px + r)
                patch = adiff[t - 1, y1:y2, x1:x2]
                roi_diff[t] = float(patch.mean() * 12.0) if patch.size else global_diff[t]
            else:
                # 无框时聚焦画面中下部桌面区域
                roi_diff[t] = float(adiff[t - 1, H // 3 :, W // 6 : 5 * W // 6].mean() * 12.0)

    kin = np.stack([wx, wy, vy, ay_rebound, vx, valid, global_diff, roi_diff], axis=1).astype(np.float32)
    kin = np.nan_to_num(kin, nan=0.0, posinf=4.0, neginf=-4.0)

    # 接触冲击综合得分：只奖励“向下落至桌面低点/触底回弹”瞬间，抑制向上抬手回抽 (vy < -0.25) 的伪峰
    downward_gate = np.where(vy >= -0.25, 1.0, 0.20).astype(np.float32)
    impact = (
        downward_gate * (0.45 * _robust_z(roi_diff) + 0.40 * _robust_z(ay_rebound))
        + 0.25 * np.maximum(0.0, _robust_z(wy))
    ).astype(np.float32)
    impact = np.nan_to_num(impact, nan=0.0, posinf=6.0, neginf=-3.0)
    return kin, impact


def detect_contact_impulses(
    impact_score: np.ndarray,
    intervals: list[Interval],
    fps: float = 20.0,
    min_gap_s: float = 0.28,
    k_mad: float = 1.05,
) -> list[int]:
    """在已标注的 knock 区间内检测单次桌面接触脉冲帧索引。"""
    T = len(impact_score)
    min_gap = max(2, int(round(min_gap_s * fps)))
    all_peaks: list[int] = []

    for iv in intervals:
        if iv.label not in ("knock", "敲击", "连续敲击"):
            continue
        a = max(1, int(round(iv.start_s * fps)))
        b = min(T - 1, int(round(iv.end_s * fps)))
        if b - a < 2:
            continue
        seg = impact_score[a:b]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med))) or 1e-5
        thr = max(med + k_mad * 1.4826 * mad, float(np.percentile(seg, 60)))

        local_peaks: list[int] = []
        for i in range(a, b):
            val = float(impact_score[i])
            if val < thr:
                continue
            prev_v = float(impact_score[i - 1]) if i > a else -1e9
            next_v = float(impact_score[i + 1]) if i + 1 < b else -1e9
            if val < prev_v or val < next_v:
                continue
            if local_peaks and i - local_peaks[-1] < min_gap:
                if val > impact_score[local_peaks[-1]]:
                    local_peaks[-1] = i
                continue
            local_peaks.append(i)

        # 若某段已标为 knock 但极其平滑未过自适应阈值，保底取区间内最大冲击点
        if not local_peaks and b > a:
            best_i = int(a + np.argmax(seg))
            local_peaks.append(best_i)

        all_peaks.extend(local_peaks)

    return sorted(set(all_peaks))


def classify_window_count(
    contact_frames: list[int] | np.ndarray,
    start_frame: int,
    end_frame: int,
    fps: float = 20.0,
    pair_min_s: float = 0.18,
    pair_max_s: float = 1.45,
) -> int:
    """根据窗口 [start_frame, end_frame) 内的接触脉冲分布判定动作类别 {0,1,2,3}。"""
    margin = 1
    in_win = sorted(int(p) for p in contact_frames if start_frame + margin <= p < end_frame - margin)
    n = len(in_win)
    if n == 0:
        return 0  # non_knock
    if n == 1:
        return 1  # single_knock
    if n == 2:
        gap_s = (in_win[1] - in_win[0]) / max(fps, 1.0)
        if pair_min_s <= gap_s <= pair_max_s:
            return 2  # double_knock
        return 1 if gap_s > pair_max_s else 2
    return 3  # multi_knock (3+)


def make_impulse_heatmap(num_frames: int, contact_frames: list[int], sigma: float = 1.25) -> np.ndarray:
    """将离散接触帧转化为高斯平滑的脉冲热图目标 y_imp ∈ [0, 1]^T。"""
    y = np.zeros(num_frames, dtype=np.float32)
    if not contact_frames:
        return y
    t_idx = np.arange(num_frames, dtype=np.float32)
    for c in contact_frames:
        y = np.maximum(y, np.exp(-0.5 * ((t_idx - float(c)) / sigma) ** 2))
    return y.astype(np.float32)


def _two_channel_vis(frames: np.ndarray) -> np.ndarray:
    """(T, H, W) uint8 -> (T, 2, H, W) float32 [gray, |diff|] ∈ [0, 1]。"""
    g = frames.astype(np.float32) / 255.0
    d = np.zeros_like(g)
    if len(frames) > 1:
        d[1:] = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).astype(np.float32) / 255.0
    return np.stack([g, d], axis=1)


def _resample_frames_to_fps(
    frames: np.ndarray,
    src_fps: float,
    dst_fps: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """把按 src_fps 解码的帧重采样到 dst_fps（最近邻），并返回新帧与原帧索引的映射。

    旧 4 条 take 是 20 fps，11-15 新增的 take 是 15 fps；统一到 20 Hz 后，
    窗口长度（48 帧 = 2.4 s）、速度特征（Δ位置 × fps/2）与合成脉冲间隔才在同一时间尺度上。
    """
    src_fps = float(max(src_fps, 1e-6))
    dst_fps = float(max(dst_fps, 1e-6))
    n_src = len(frames)
    if n_src == 0 or abs(src_fps - dst_fps) < 1e-3:
        return frames, np.arange(n_src, dtype=np.int64), dst_fps
    n_dst = max(1, int(round(n_src * dst_fps / src_fps)))
    idx = np.round(np.arange(n_dst, dtype=np.float64) * (src_fps / dst_fps)).astype(np.int64)
    idx = np.clip(idx, 0, n_src - 1)
    return frames[idx], idx, dst_fps


def build_window_dataset(
    dataset_dir: str | Path = "dataset",
    pose_roi_dir: str | Path = "runs_knock/pose_roi",
    window: int = 48,
    stride: int = 4,
    size: int = 96,
    contact_sheet_dir: str | Path | None = None,
) -> dict[str, dict]:
    """构建全部已标 take 的双流窗口数据集，并提取单次脉冲供训练折合成增强使用。"""
    repo_root = Path(__file__).resolve().parents[2]
    ds_path = Path(dataset_dir) if Path(dataset_dir).is_absolute() else repo_root / dataset_dir
    roi_path = Path(pose_roi_dir) if Path(pose_roi_dir).is_absolute() else repo_root / pose_roi_dir

    id_to_name = load_labels(ds_path)
    takes = load_takes(ds_path)
    sealed_paths = [t.video_path for t in takes if sealed.is_sealed_path(t.take_id)]
    if sealed_paths:
        sealed.require_final_eval(sealed_paths, purpose="double-knock dataset final held-out evaluation")
    out: dict[str, dict] = {}

    if contact_sheet_dir is not None:
        cs_dir = Path(contact_sheet_dir) if Path(contact_sheet_dir).is_absolute() else repo_root / contact_sheet_dir
        cs_dir.mkdir(parents=True, exist_ok=True)
    else:
        cs_dir = None

    for take in takes:
        if take.annotation_path is None:
            continue
        intervals = load_intervals(take, id_to_name)
        frames = decode_gray(take.video_path, size)
        src_t = len(frames)
        frames, src_idx, feat_fps = _resample_frames_to_fps(frames, take.fps, 20.0)
        T = len(frames)
        labels = frame_labels(intervals, T, feat_fps)

        npz_file = roi_path / f"{take.take_id}.npz"
        boxes = np.load(npz_file, allow_pickle=True)["boxes"] if npz_file.exists() else None
        if boxes is not None:
            boxes = np.asarray(boxes, dtype=object)
            if len(boxes) == src_t:
                boxes = boxes[src_idx]

        kin, impact = extract_kinematics_and_energy(frames, boxes, fps=feat_fps)
        contacts = detect_contact_impulses(impact, intervals, fps=feat_fps, min_gap_s=0.28)
        y_imp_full = make_impulse_heatmap(T, contacts, sigma=1.25)
        vis_full = _two_channel_vis(frames)

        if cs_dir is not None and contacts:
            times_s = [round(c / feat_fps, 2) for c in contacts[:30]]
            contact_sheet(frames, times_s, feat_fps, cs_dir / f"{take.take_id}_contacts.png")

        x_vis_list, x_kin_list, y_imp_list, y_cnt_list, centers_s = [], [], [], [], []
        for s in range(0, max(1, T - window + 1), stride):
            e = s + window
            if e > T:
                break
            win_lbls = labels[s:e]
            # 跳过大部分帧未标注的空白过渡窗口
            valid_ratio = float(np.mean(win_lbls != IGNORE))
            if valid_ratio < 0.55:
                continue
            cnt_cls = classify_window_count(contacts, s, e, fps=feat_fps)
            # 若窗口处于负类区间（如 hand_raise/reach/other/background），强制为 non_knock(0)
            if np.sum(win_lbls == KNOCK_IDX) == 0:
                cnt_cls = 0

            x_vis_list.append(np.rint(vis_full[s:e] * 255.0).astype(np.uint8))
            x_kin_list.append(kin[s:e])
            y_imp_list.append(y_imp_full[s:e])
            y_cnt_list.append(cnt_cls)
            centers_s.append(round((s + 0.5 * window) / feat_fps, 3))

        # 提取本条 take 内的单次敲击短片段 (12 帧 = 0.6s) 与纯负样本窗口，供训练折内部组合增强
        stroke_half = 6
        impulses = []
        for c in contacts:
            if c - stroke_half >= 0 and c + stroke_half <= T:
                impulses.append({
                    "vis": vis_full[c - stroke_half : c + stroke_half].copy(),
                    "kin": kin[c - stroke_half : c + stroke_half].copy(),
                })
        negatives = []
        for i, c_cls in enumerate(y_cnt_list):
            if c_cls == 0:
                negatives.append({
                    "vis": x_vis_list[i].copy(),
                    "kin": x_kin_list[i].copy(),
                })

        out[take.take_id] = {
            "fps": feat_fps,
            "fps_src": float(take.fps),
            "duration_s": float(take.duration_s),
            "frames": frames,
            "vis_full": vis_full.astype(np.float16),
            "kin_full": kin,
            "impact_full": impact,
            "labels_full": labels,
            "y_imp_full": y_imp_full,
            "contacts": contacts,
            "intervals": intervals,
            "x_vis": np.stack(x_vis_list).astype(np.uint8) if x_vis_list else np.zeros((0, window, 2, size, size), np.uint8),
            "x_kin": np.stack(x_kin_list).astype(np.float32) if x_kin_list else np.zeros((0, window, 8), np.float32),
            "y_imp": np.stack(y_imp_list).astype(np.float32) if y_imp_list else np.zeros((0, window), np.float32),
            "y_cnt": np.asarray(y_cnt_list, dtype=np.int64),
            "centers_s": np.asarray(centers_s, dtype=np.float32),
            "impulses": impulses,
            "negatives": negatives,
        }
    return out


def synthesize_training_windows(
    train_data: dict[str, dict],
    forbidden_val_take: str | None = None,
    n_per_class: int = 80,
    window: int = 48,
    fps: float = 20.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """仅从训练折提取的真实单次脉冲与真实负样本背景中，合成 1/2/3 下敲击窗口。"""
    if forbidden_val_take is not None and forbidden_val_take in train_data:
        raise ValueError(f"禁止将留出测试集 {forbidden_val_take} 传入训练合成增强器！")

    all_impulses = []
    all_negs = []
    for tid, d in train_data.items():
        all_impulses.extend(d.get("impulses", []))
        all_negs.extend(d.get("negatives", []))

    if not all_impulses or not all_negs:
        return {
            "x_vis": np.zeros((0, window, 2, 96, 96), dtype=np.float32),
            "x_kin": np.zeros((0, window, 8), dtype=np.float32),
            "y_imp": np.zeros((0, window), dtype=np.float32),
            "y_cnt": np.zeros((0,), dtype=np.int64),
        }

    rng = np.random.default_rng(seed)
    stroke_len = all_impulses[0]["vis"].shape[0]
    half = stroke_len // 2
    # 构造余弦软过渡权重，消除拼接边界伪影
    blend = np.hanning(stroke_len + 2)[1:-1].astype(np.float32)
    blend_vis = blend[:, None, None, None]
    blend_kin = blend[:, None]

    x_vis_out, x_kin_out, y_imp_out, y_cnt_out = [], [], [], []

    for target_cls in (1, 2, 3):
        for _ in range(n_per_class):
            neg = all_negs[int(rng.integers(0, len(all_negs)))]
            vis = neg["vis"].copy()
            kin = neg["kin"].copy()

            if target_cls == 1:
                c1 = int(rng.integers(half + 4, window - half - 4))
                centers = [c1]
            elif target_cls == 2:
                # 双敲：两下间隔 6~17 帧 (0.30s ~ 0.85s @ 20fps)
                gap = int(rng.integers(6, min(18, window - 2 * stroke_len - 4)))
                c1 = int(rng.integers(half + 3, window - half - gap - 3))
                centers = [c1, c1 + gap]
            else:
                # 连敲 3 下
                gap1 = int(rng.integers(5, 9))
                gap2 = int(rng.integers(5, 9))
                c1 = int(rng.integers(half + 2, max(half + 3, window - half - gap1 - gap2 - 2)))
                centers = [c1, c1 + gap1, min(window - half - 2, c1 + gap1 + gap2)]

            for c in centers:
                imp = all_impulses[int(rng.integers(0, len(all_impulses)))]
                s, e = c - half, c - half + stroke_len
                if s < 0 or e > window:
                    continue
                vis[s:e] = (1.0 - blend_vis) * vis[s:e] + blend_vis * imp["vis"]
                kin[s:e] = (1.0 - blend_kin) * kin[s:e] + blend_kin * imp["kin"]

            y_imp = make_impulse_heatmap(window, centers, sigma=1.25)
            x_vis_out.append(vis)
            x_kin_out.append(kin)
            y_imp_out.append(y_imp)
            y_cnt_out.append(target_cls)

    return {
        "x_vis": np.stack(x_vis_out).astype(np.float32),
        "x_kin": np.stack(x_kin_out).astype(np.float32),
        "y_imp": np.stack(y_imp_out).astype(np.float32),
        "y_cnt": np.asarray(y_cnt_out, dtype=np.int64),
    }


def _decode_video_bytes_to_gray(video_path: Path, size: int = 96, max_frames: int = 180) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        frames.append(cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, size, size), dtype=np.uint8)


def load_external_stress_clips(max_clips_per_source: int = 10, size: int = 96) -> list[dict]:
    """加载多源外部验证集：
    1) D:/robot_work/evidence/sessions: 真机 0.5m 录制（1 条双敲正样本 + 9 条手势/空镜硬负样本）
    2) G:/stream_memory_data/videos/charades/Charades_v1_480.zip: 真实室内桌面交互负样本
    3) G:/stream_memory_data/videos/ucf101/UCF-101.zip: 真实人类动作/节律干扰负样本
    """
    repo_root = Path(__file__).resolve().parents[2]
    clips: list[dict] = []

    # 1. 本地真机 evidence/sessions
    sess_dir = repo_root / "evidence" / "sessions"
    manifest_path = sess_dir / "manifest.jsonl"
    if manifest_path.exists():
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        for line in lines[: max_clips_per_source + 2]:
            if not line.strip():
                continue
            meta = json.loads(line)
            fdir = sess_dir / meta["dir"] / "frames"
            if sealed.is_sealed_path(meta["dir"]):
                sealed.require_final_eval([fdir], purpose="double-knock external stress final evaluation")
            if not fdir.is_dir():
                continue
            jpgs = sorted(fdir.glob("*.jpg"))
            imgs = []
            for p in jpgs:
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    imgs.append(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
            if len(imgs) >= 48:
                is_knock = meta.get("task") == "knock"
                kps_p = sess_dir / meta["dir"] / "frames.jsonl"
                clips.append({
                    "id": meta["dir"],
                    "source": "evidence_sessions",
                    "task": meta.get("task", "unknown"),
                    "is_positive_double_knock": bool(is_knock),
                    "fps": round(len(imgs) / max(float(meta.get("seconds_actual", 20.0)), 1.0), 2),
                    "frames": np.stack(imgs).astype(np.uint8),
                    "kps_path": kps_p if kps_p.exists() else None,
                })

    # 2. Charades 室内桌面动作 (c009, c011, c012, c014, c109)
    char_zip = Path("G:/stream_memory_data/videos/charades/Charades_v1_480.zip")
    char_csv = Path("G:/stream_memory_data/videos/charades/ann/Charades/Charades_v1_train.csv")
    if char_zip.exists() and char_csv.exists():
        table_classes = {"c009", "c011", "c012", "c013", "c014", "c109", "c116"}
        selected_ids = []
        with char_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                acts = row.get("actions", "")
                if any(tc in acts for tc in table_classes):
                    selected_ids.append(row["id"])
                if len(selected_ids) >= max_clips_per_source * 2:
                    break
        with zipfile.ZipFile(char_zip, "r") as zf:
            names = set(zf.namelist())
            count = 0
            for vid in selected_ids:
                cand = f"Charades_v1_480/{vid}.mp4"
                if cand not in names:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    tf.write(zf.read(cand))
                    tmp_path = Path(tf.name)
                try:
                    arr = _decode_video_bytes_to_gray(tmp_path, size=size, max_frames=160)
                    if len(arr) >= 48:
                        clips.append({
                            "id": f"charades_{vid}",
                            "source": "charades_table",
                            "task": "table_interaction_negative",
                            "is_positive_double_knock": False,
                            "fps": 20.0,
                            "frames": arr,
                        })
                        count += 1
                finally:
                    tmp_path.unlink(missing_ok=True)
                if count >= max_clips_per_source:
                    break

    # 3. UCF-101 节律与桌面/手部干扰动作
    ucf_zip = Path("G:/stream_memory_data/videos/ucf101/UCF-101.zip")
    if ucf_zip.exists():
        target_ucf = ("Typing", "WritingOnBoard", "CuttingInKitchen", "TableTennisShot", "Drumming", "Hammering")
        with zipfile.ZipFile(ucf_zip, "r") as zf:
            avi_list = [
                n for n in zf.namelist()
                if n.endswith(".avi") and any(f"/{cls}/" in n for cls in target_ucf)
            ]
            # 均匀抽样不同类别
            step = max(1, len(avi_list) // max(1, max_clips_per_source))
            chosen = avi_list[::step][:max_clips_per_source]
            for name in chosen:
                with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tf:
                    tf.write(zf.read(name))
                    tmp_path = Path(tf.name)
                try:
                    arr = _decode_video_bytes_to_gray(tmp_path, size=size, max_frames=160)
                    if len(arr) >= 48:
                        cls_name = name.split("/")[1] if "/" in name else "ucf"
                        clips.append({
                            "id": Path(name).stem,
                            "source": "ucf101_distractor",
                            "task": cls_name,
                            "is_positive_double_knock": False,
                            "fps": 20.0,
                            "frames": arr,
                        })
                finally:
                    tmp_path.unlink(missing_ok=True)

    return clips


if __name__ == "__main__":
    data = build_window_dataset(contact_sheet_dir="runs_knock/double_knock_20261115/contact_sheets")
    for tid, d in data.items():
        vals, cnts = np.unique(d["y_cnt"], return_counts=True)
        dist = {COUNT_CLASSES[int(k)]: int(v) for k, v in zip(vals, cnts)}
        print(f"{tid}: contacts={len(d['contacts'])}, windows={len(d['y_cnt'])}, class_dist={dist}")
    ext = load_external_stress_clips(max_clips_per_source=6)
    print(f"External stress clips loaded: {len(ext)}")
