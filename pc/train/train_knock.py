"""区间标注 -> 敲击检测（逐帧二分类 + 事件级指标）。

为什么不先做六分类：4 条 take 的标签是"敲击活动段"，段很长（4–10 s），
六分类在帧级上既难定义也难以解释。这里先做机器人真正关心的任务：
当前帧是否处于敲击活动段，以及能不能把敲击段从连续视频里切出来。

- 输入：dataset/takes/*/(take.mp4, annotation.json, take.json)
- 正类：label 解析后属于 knock 的区间；负类：其它已标注区间
- 未标注帧 ignore（不是负样本）
- 模型：frame = 逐帧 CNN；gru = CNN + 因果 GRU
- 评估：按 take 留一；帧级 F1/AUROC + 事件级（IoU≥0.3 的段匹配）P/R/F1

用法
    python pc/train/train_knock.py --dataset dataset --epochs 25 --out runs_knock/run1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 缺失时只影响训练增强
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_data import (  # noqa: E402
    CLASS_TO_IDX,
    IGNORE,
    build_windows,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
    window_sample,
)

from train_interval import FrameEncoder  # noqa: E402


class KnockFrameCNN(nn.Module):
    def __init__(self, feat_dim: int = 192, in_ch: int = 2):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim, in_ch)
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


class KnockCnnGRU(nn.Module):
    def __init__(self, feat_dim: int = 192, hidden: int = 128, in_ch: int = 2):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim, in_ch)
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        out, _ = self.gru(feat)
        return self.head(out).squeeze(-1)


class KnockCnnTCN(nn.Module):
    """因果空洞卷积（TCN）时间头：只看过去与当前帧，部署时可流式运行。"""

    def __init__(self, feat_dim: int = 192, hidden: int = 128, in_ch: int = 2):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim, in_ch)
        self.dilations = (1, 2, 4, 8)
        convs = []
        for i, d in enumerate(self.dilations):
            cin = feat_dim if i == 0 else hidden
            convs.append(nn.Conv1d(cin, hidden, kernel_size=3, dilation=d))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in self.dilations])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).transpose(1, 2)  # B,F,T
        t = feat.size(-1)
        for conv, norm, d in zip(self.convs, self.norms, self.dilations):
            y = torch.nn.functional.pad(feat, (2 * d, 0))
            feat = self.drop(torch.relu(norm(conv(y)[..., -t:])))
        return self.head(feat.transpose(1, 2)).squeeze(-1)


class KnockCnnBiGRU(nn.Module):
    """双向 GRU：仅作离线诊断，用来看“未来信息”能把上限抬多高。"""

    def __init__(self, feat_dim: int = 192, hidden: int = 128, in_ch: int = 2):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim, in_ch)
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        out, _ = self.gru(feat)
        return self.head(out).squeeze(-1)


def build_model(name: str, in_ch: int = 2) -> nn.Module:
    if name == "frame":
        return KnockFrameCNN(in_ch=in_ch)
    if name == "gru":
        return KnockCnnGRU(in_ch=in_ch)
    if name == "tcn":
        return KnockCnnTCN(in_ch=in_ch)
    if name == "bigru":
        return KnockCnnBiGRU(in_ch=in_ch)
    raise ValueError(name)


class KnockWindows(torch.utils.data.Dataset):
    def __init__(self, takes, frames, rings, window, size, augment: bool = False, roi_zoom: float = 0.0,
                 hard_mask: dict | None = None, hard_weight: float = 0.0):
        self.items = []
        self.frames = frames
        self.rings = rings
        self.window = window
        self.size = size
        self.augment = augment
        self.roi_zoom = float(roi_zoom)
        self.weights = []
        for take in takes:
            take_id = take.take_id
            for start, _ in build_windows(int(frames[take_id].shape[0]), window):
                self.items.append((take_id, start))
                w = 1.0
                if hard_mask is not None and hard_weight > 0:
                    stop = min(start + window, int(frames[take_id].shape[0]))
                    if stop > start:
                        w = 1.0 + hard_weight * float(np.mean(hard_mask[take_id][start:stop]))
                self.weights.append(w)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        take_id, start = self.items[idx]
        dummy = np.zeros(int(self.frames[take_id].shape[0]), dtype=np.int64)
        x, _ = window_sample(self.frames[take_id], dummy, start, self.window, self.size)
        if self.augment:
            x = self._augment(x)
        y = self.rings[take_id]
        n = int(self.frames[take_id].shape[0])
        stop = min(start + self.window, n)
        target = y[start:stop].astype(np.float32)
        pad = self.window - (stop - start)
        if pad > 0:
            target = np.concatenate([target, np.full(pad, np.nan, dtype=np.float32)])
        return torch.from_numpy(x), torch.from_numpy(target)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """视频/时序增强：光照、对比度、gamma、噪声、缩放平移、翻转、时间抖动、时间遮挡。

        输入 x 形状是 (T, 2, H, W)（通道在后）；变换在同一窗口内时空一致，
        不会每帧独立抖动，否则时序模型会学到假的差分模式。
        """
        gray = x[:, 0].copy() if x.ndim == 4 else x[0].copy()
        diff = x[:, 1].copy() if x.ndim == 4 else x[1].copy()
        gamma = 0.8 + 0.4 * np.random.rand()
        gain = 0.75 + 0.5 * np.random.rand()
        bias = (np.random.rand() - 0.5) * 0.2
        gray = np.clip((gray ** gamma) * gain + bias, 0.0, 1.0)
        if np.random.rand() < 0.3:
            gray = np.clip(gray + np.random.randn(*gray.shape).astype(np.float32) * 0.02, 0.0, 1.0)
        diff = np.clip(diff * (0.5 + 1.0 * np.random.rand()) + (np.random.rand() - 0.5) * 0.1, -1.0, 1.0)
        if cv2 is not None and np.random.rand() < 0.7:
            scale = 0.90 + 0.20 * np.random.rand()
            tx = (np.random.rand() - 0.5) * 8.0
            ty = (np.random.rand() - 0.5) * 8.0
            mat = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
            h, w = gray.shape[1:]
            gray = np.stack([
                cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                for frame in gray
            ])
            diff = np.stack([
                cv2.warpAffine(frame, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                for frame in diff
            ])
        if cv2 is not None and self.roi_zoom > 0 and np.random.rand() < self.roi_zoom:
            # 运动能量中心裁剪（"敲击特写"增强）：按窗口内平均帧差找运动质心，
            # 裁一个约 0.5–0.75 边长的方块再放大回原尺寸，让手在输入里更大。
            energy = np.abs(diff).mean(axis=0)
            total = float(energy.sum())
            h, w = gray.shape[1:]
            if total > 1e-4:
                ys, xs = np.mgrid[0:h, 0:w]
                cy = float((energy * ys).sum() / total)
                cx = float((energy * xs).sum() / total)
            else:
                cy, cx = h / 2.0, w / 2.0
            side = int(round(min(h, w) * (0.50 + 0.25 * np.random.rand())))
            side = max(24, min(side, min(h, w)))
            x0 = int(round(cx - side / 2 + (np.random.rand() - 0.5) * 0.15 * side))
            y0 = int(round(cy - side / 2 + (np.random.rand() - 0.5) * 0.15 * side))
            x0 = max(0, min(x0, w - side))
            y0 = max(0, min(y0, h - side))
            gray = np.stack([
                cv2.resize(frame[y0:y0 + side, x0:x0 + side], (w, h), interpolation=cv2.INTER_LINEAR)
                for frame in gray
            ])
            diff = np.stack([
                cv2.resize(frame[y0:y0 + side, x0:x0 + side], (w, h), interpolation=cv2.INTER_LINEAR)
                for frame in diff
            ])
        if np.random.rand() < 0.5:
            gray = gray[:, :, ::-1]
            diff = diff[:, :, ::-1]
        if np.random.rand() < 0.5:
            t = gray.shape[0]
            if np.random.rand() < 0.5:
                idx = np.sort(np.random.choice(t, t, replace=True))
            else:
                idx = np.clip(np.arange(t) + np.random.randint(-3, 4), 0, t - 1)
            gray = gray[idx]
            diff = diff[idx]
        if np.random.rand() < 0.2:
            t = gray.shape[0]
            length = np.random.randint(3, 9)
            start = np.random.randint(0, max(1, t - length))
            gray[start:start + length] = 0.0
            diff[start:start + length] = 0.0
        if x.ndim == 4:
            return np.ascontiguousarray(np.stack([gray, diff], axis=1))
        return np.ascontiguousarray(np.stack([gray, diff], axis=0))


def masked_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    mask = ~torch.isnan(target)
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], target[mask], pos_weight=pos_weight)


def masked_focal(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
                 alpha: float = 0.25, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Focal loss（Lin et al. 2017, https://arxiv.org/abs/1708.02002）。

    敲击正样本只占标注帧的一部分，难例主要是"手在动但没敲"的近似动作；
    focal 项让训练集中在这些难例上，而不是被大量容易的负帧主导。
    """
    mask = ~torch.isnan(target)
    logits = logits[mask]
    target = target[mask]
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    return loss.mean()


def train_epoch(model, loader, device, optimizer, pos_weight=None, mixup_p: float = 0.0,
                loss_name: str = "bce", focal_gamma: float = 2.0) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_p > 0 and x.size(0) > 1 and random.random() < mixup_p:
            lam = float(np.random.beta(0.4, 0.4))
            perm = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1.0 - lam) * x[perm]
        logits = model(x)
        if loss_name == "focal":
            loss = masked_focal(logits, y, gamma=focal_gamma, pos_weight=pos_weight)
        else:
            loss = masked_bce(logits, y, pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        n = int((~torch.isnan(y)).sum().item())
        total += loss.item() * n
        count += n
    return total / max(count, 1)


@torch.no_grad()
def predict_frames(model, loader, n_frames: int, window: int, device) -> np.ndarray:
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    prob = np.concatenate(probs)[:n_frames] if probs else np.zeros(n_frames, dtype=np.float32)
    return prob


def runs_above(prob: np.ndarray, threshold: float, min_len: int, max_gap: int = 0) -> list[tuple[int, int]]:
    """返回 [start, stop) 帧区间；max_gap 帧内的小间断合并，min_len 过滤抖动。"""
    mask = prob >= threshold
    raw: list[list[int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            raw.append([start, i])
            start = None
    if start is not None:
        raw.append([start, len(mask)])
    merged: list[list[int]] = []
    for seg in raw:
        if merged and seg[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    return [(a, b) for a, b in merged if b - a >= min_len]


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def event_metrics(pred: list[tuple[float, float]], gt: list[tuple[float, float]], iou_thr: float = 0.3) -> dict:
    used = set()
    tp = 0
    for p in pred:
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gt):
            if i in used:
                continue
            iou = interval_overlap(p, g)
            if iou > best_iou:
                best_i, best_iou = i, iou
        if best_iou >= iou_thr:
            used.add(best_i)
            tp += 1
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_pred": len(pred), "n_gt": len(gt), "tp": tp, "precision": precision, "recall": recall, "f1": f1}


def frame_metrics(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    mask = ~np.isnan(target)
    p = prob[mask]
    t = (target[mask] >= 0.5).astype(np.int64)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (t == 1)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = float((pred == t).mean()) if len(t) else 0.0
    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(t, p)) if len(set(t.tolist())) > 1 else 0.0
    except Exception:
        auroc = 0.0
    return {"frames": int(mask.sum()), "positive": int(t.sum()), "precision": precision,
            "recall": recall, "f1": f1, "accuracy": acc, "auroc": auroc}


def smooth_prob(prob: np.ndarray, smooth_frames: int) -> np.ndarray:
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames, dtype=np.float32) / smooth_frames
        return np.convolve(prob, kernel, mode="same")
    return prob


def normalize_probs(probs: dict[str, np.ndarray], mode: str = "none") -> dict[str, np.ndarray]:
    """按 take 做分数标准化，抵消跨 take 的阈值漂移。

    这不是"用验证集调阈值"：推理时在整段录像上算均值和方差是允许的
    （没有标签），阈值仍在训练 take 上选，只是从概率域换到 z-score 域。
    """
    if mode == "none":
        return probs
    out: dict[str, np.ndarray] = {}
    for key, prob in probs.items():
        mu = float(np.mean(prob))
        sd = float(np.std(prob)) or 1e-6
        out[key] = ((prob - mu) / sd).astype(np.float32)
    return out


def peak_indices(prob: np.ndarray, threshold: float, min_gap: int = 8) -> list[int]:
    """阈值以上的局部极大值；min_gap 帧内的峰只保留最高的那个。"""
    peaks: list[int] = []
    for i in range(1, len(prob) - 1):
        if prob[i] < threshold:
            continue
        if prob[i] < prob[i - 1] or prob[i] < prob[i + 1]:
            continue
        if peaks and i - peaks[-1] < min_gap:
            if prob[i] > prob[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def hit_metrics(pred: list[tuple[float, float]], gt: list[tuple[float, float]], duration_s: float) -> dict:
    """区间命中率 + 每分钟误报数，比事件 F1 更能说明"能不能用"。"""
    hit = sum(1 for a, b in gt if any(interval_overlap((x, y), (a, b)) > 0 for x, y in pred))
    false_alarms = sum(1 for x, y in pred if not any(interval_overlap((x, y), (a, b)) > 0 for a, b in gt))
    minutes = max(duration_s, 1e-6) / 60.0
    return {
        "hit_rate": hit / len(gt) if gt else 0.0,
        "false_alarms": false_alarms,
        "fa_per_min": false_alarms / minutes,
    }


def pair_metrics(prob: np.ndarray, fps: float, threshold: float, gt_events: list[tuple[float, float]],
                 duration_s: float, min_gap_s: float = 0.25, max_gap_s: float = 1.5,
                 margin_s: float = 0.5, peak_min_gap: int = 8, isolation_s: float = 2.0) -> dict:
    """两敲触发指标：机器人真正需要的不是"检测到敲击活动"，而是"两下敲完"。

    阈值以上取峰；相邻两峰间隔在 [min_gap_s, max_gap_s] 内算一次两敲触发；
    第二次敲击落在某个敲击区间（含 margin）内算命中，否则算误报。
    isolation_s 要求这次两敲前后安静——真实"敲两下"是孤立事件，连续乱动时峰是连片的。
    """
    peaks = peak_indices(prob, threshold, peak_min_gap)
    times = [i / fps for i in peaks]
    triggers = []
    for k, (a, b) in enumerate(zip(times, times[1:])):
        if not (min_gap_s <= b - a <= max_gap_s):
            continue
        others = times[:k] + times[k + 2:]
        if any(a - isolation_s <= t <= b + isolation_s for t in others):
            continue
        triggers.append(b)

    def in_gt(t: float) -> bool:
        return any(a - margin_s <= t <= b + margin_s for a, b in gt_events)

    tp = sum(1 for t in triggers if in_gt(t))
    fp = len(triggers) - tp
    hit = sum(1 for a, b in gt_events if any(a - margin_s <= t <= b + margin_s for t in triggers))
    minutes = max(duration_s, 1e-6) / 60.0
    precision = tp / len(triggers) if triggers else 0.0
    recall = hit / len(gt_events) if gt_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"triggers": len(triggers), "tp": tp, "fp": fp, "hit_rate": recall,
            "precision": precision, "recall": recall, "f1": f1, "fa_per_min": fp / minutes}


def micro_pair_f1(probs: dict[str, np.ndarray], gt_events: dict[str, list], fps_map: dict[str, float],
                  duration_map: dict[str, float], threshold: float, min_gap_s: float,
                  max_gap_s: float, margin_s: float, peak_min_gap: int,
                  isolation_s: float = 2.0) -> float:
    tp = fp = hit = n_gt = 0
    for take_id, prob in probs.items():
        m = pair_metrics(prob, fps_map[take_id], threshold, gt_events.get(take_id, []),
                         duration_map[take_id], min_gap_s, max_gap_s, margin_s, peak_min_gap, isolation_s)
        tp += m["tp"]
        fp += m["fp"]
        hit += int(round(m["hit_rate"] * len(gt_events.get(take_id, []))))
        n_gt += len(gt_events.get(take_id, []))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = hit / n_gt if n_gt else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def predict_take_probs(model, take_list, frames, rings, window, size, device, smooth_frames) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for take in take_list:
        ds = KnockWindows([take], frames, rings, window, size)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        prob = predict_frames(model, loader, len(rings[take.take_id]), window, device)
        out[take.take_id] = smooth_prob(prob, smooth_frames)
    return out


def aggregate_frame_f1(probs: dict[str, np.ndarray], rings: dict[str, np.ndarray], threshold: float) -> float:
    ps, ts = [], []
    for take_id, prob in probs.items():
        target = rings[take_id].astype(np.float32)
        mask = ~np.isnan(target)
        ps.append(prob[mask])
        ts.append(target[mask])
    if not ps:
        return 0.0
    p = np.concatenate(ps)
    t = np.concatenate(ts)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (t == 1)).sum())
    fp = int(((pred == 1) & (t == 0)).sum())
    fn = int(((pred == 0) & (t == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def micro_event_f1(probs: dict[str, np.ndarray], gt_events: dict[str, list], fps_map: dict[str, float],
                   threshold: float, min_len: int, max_gap: int) -> float:
    tp = fp = fn = 0
    for take_id, prob in probs.items():
        fps = fps_map[take_id]
        pred = [(a / fps, b / fps) for a, b in runs_above(prob, threshold, min_len, max_gap)]
        gt = gt_events.get(take_id, [])
        used = set()
        local_tp = 0
        for p in pred:
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt):
                if i in used:
                    continue
                iou = interval_overlap(p, g)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_iou >= 0.3:
                used.add(best_i)
                local_tp += 1
        tp += local_tp
        fp += len(pred) - local_tp
        fn += len(gt) - local_tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def pick_threshold(model, train_takes, frames, rings, gt_events, window, size, device, smooth_frames,
                   min_len: int, max_gap: int, metric: str = "event", normalize: str = "none",
                   pair_min_gap_s: float = 0.25, pair_max_gap_s: float = 1.5,
                   pair_margin_s: float = 0.5, peak_min_gap: int = 8,
                   pair_isolation_s: float = 2.0) -> tuple[float, float]:
    probs = predict_take_probs(model, train_takes, frames, rings, window, size, device, smooth_frames)
    probs = normalize_probs(probs, normalize)
    fps_map = {t.take_id: t.fps for t in train_takes}
    duration_map = {t.take_id: t.duration_s for t in train_takes}
    if normalize == "none":
        grid = np.arange(0.10, 0.901, 0.02)
    else:
        grid = np.arange(-0.5, 3.001, 0.1)
    best_thr, best_score = float(grid[0]), -1.0
    for thr in grid:
        if metric == "event":
            score = micro_event_f1(probs, gt_events, fps_map, float(thr), min_len, max_gap)
        elif metric == "pair":
            score = micro_pair_f1(probs, gt_events, fps_map, duration_map, float(thr),
                                  pair_min_gap_s, pair_max_gap_s, pair_margin_s, peak_min_gap,
                                  pair_isolation_s)
        else:
            score = aggregate_frame_f1(probs, rings, float(thr))
        if score > best_score:
            best_thr, best_score = float(thr), score
    return best_thr, best_score


def pick_pair_threshold(model, train_takes, frames, rings, gt_events, window, size, device,
                        smooth_frames, normalize, pair_min_gap_s=0.25, pair_max_gap_s=1.5,
                        pair_margin_s=0.5, peak_min_gap=8, pair_isolation_s=2.0) -> tuple[float, float]:
    """单独为“两敲触发”选峰值阈值。

    事件阈值优化的是帧级/段级敲击活动，和“两下敲完”不是同一个决策边界；
    用同一阈值会要么太松（一堆假触发）要么太紧（漏掉真触发）。
    """
    probs = predict_take_probs(model, train_takes, frames, rings, window, size, device, smooth_frames)
    probs = normalize_probs(probs, normalize)
    fps_map = {t.take_id: t.fps for t in train_takes}
    duration_map = {t.take_id: t.duration_s for t in train_takes}
    grid = np.arange(-0.5, 3.501, 0.1) if normalize != "none" else np.arange(0.30, 0.991, 0.02)
    best_thr, best_score = float(grid[0]), -1.0
    for thr in grid:
        score = micro_pair_f1(probs, gt_events, fps_map, duration_map, float(thr),
                              pair_min_gap_s, pair_max_gap_s, pair_margin_s, peak_min_gap,
                              pair_isolation_s)
        if score > best_score:
            best_thr, best_score = float(thr), score
    return best_thr, best_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="runs_knock/run")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--models", default="frame,gru")
    ap.add_argument("--takes", default="")
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=-1.0, help="固定阈值；<=0 表示在训练 take 上自动选")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="训练时做光照/翻转增强")
    ap.add_argument("--roi-zoom", type=float, default=0.5,
                    help="按运动质心做“敲击特写”裁剪增强的概率（0 关闭）")
    ap.add_argument("--loss", choices=["bce", "focal"], default="focal", help="正负样本不平衡时 focal 通常更稳")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--cosine", action=argparse.BooleanOptionalAction, default=True, help="余弦退火学习率")
    ap.add_argument("--hard-negative", type=float, default=1.0,
                    help="抬手/伸手/其它动作等易混负样本窗口的额外采样权重（0 关闭）")
    ap.add_argument("--smooth-frames", type=int, default=20)
    ap.add_argument("--min-event-frames", type=int, default=8)
    ap.add_argument("--max-gap-frames", type=int, default=10)
    ap.add_argument("--threshold-metric", choices=["event", "frame", "pair"], default="frame",
                    help="事件阈值指标；pair 表示也用两敲指标（不推荐，通常用 frame）")
    ap.add_argument("--normalize", choices=["none", "zscore"], default="zscore",
                    help="按 take 标准化分数；zscore 可缓解跨 take 阈值漂移")
    ap.add_argument("--pair-min-gap-s", type=float, default=0.25, help="两敲之间的最短间隔（秒）")
    ap.add_argument("--pair-max-gap-s", type=float, default=1.5, help="两敲之间的最长间隔（秒）")
    ap.add_argument("--pair-margin-s", type=float, default=0.5, help="命中判定时区间向外放宽的秒数")
    ap.add_argument("--pair-isolation-s", type=float, default=2.0,
                    help="两敲前后必须安静的秒数，压掉连续乱动产生的假触发")
    ap.add_argument("--pair-threshold", type=float, default=-999.0,
                    help="两敲峰值阈值；<-100 表示在训练 take 上按两敲 F1 自动选")
    ap.add_argument("--pair-threshold-mode", choices=["event", "auto"], default="event",
                    help="两敲触发用事件阈值（更稳定）还是单独在训练集上选")
    ap.add_argument("--save-folds", action="store_true", help="保存每个留一折的模型，便于离线复查")
    ap.add_argument("--mixup", type=float, default=0.2, help="训练窗口 MixUp 概率（0 关闭）")
    ap.add_argument("--peak-min-gap", type=int, default=8, help="敲击峰之间的最小间隔帧数")
    ap.add_argument("--final-model", action=argparse.BooleanOptionalAction, default=False,
                    help="额外用全部 take 训练部署模型并保存 model_<name>.pt（无留出集，指标偏乐观）")
    ap.add_argument("--skip-folds", action="store_true", help="只训练部署模型，不跑留一（配合 --final-model）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    id_to_name = load_labels(dataset_dir)
    takes = [t for t in load_takes(dataset_dir) if t.annotation_path is not None]
    if args.takes:
        wanted = {s.strip() for s in args.takes.split(",") if s.strip()}
        takes = [t for t in takes if t.take_id in wanted]
    if not takes:
        raise SystemExit("no annotated takes")

    print(f"device={args.device} takes={[t.take_id for t in takes]}")
    frames: dict[str, np.ndarray] = {}
    knock_rings: dict[str, np.ndarray] = {}
    hard_masks: dict[str, np.ndarray] = {}
    gt_events: dict[str, list[float]] = {}
    for take in takes:
        intervals = load_intervals(take, id_to_name)
        frames[take.take_id] = decode_gray(take.video_path, args.size)
        labels = frame_labels(intervals, frames[take.take_id].shape[0], take.fps)
        ring = np.where(labels == IGNORE, np.nan, (labels == 0).astype(np.float32))
        knock_rings[take.take_id] = ring
        hard = np.isin(labels, [CLASS_TO_IDX["hand_raise"], CLASS_TO_IDX["reach"], CLASS_TO_IDX["other"]])
        hard_masks[take.take_id] = hard.astype(np.float32)
        gt_events[take.take_id] = [(iv.start_s, iv.end_s) for iv in intervals if iv.label == "knock"]
        pos = int(np.nansum(ring))
        print(f"  {take.take_id}: frames={len(ring)} labeled={int((~np.isnan(ring)).sum())} knock_frames={pos} knock_events={len(gt_events[take.take_id])}")

    device = torch.device(args.device)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    fold_takes = [] if args.skip_folds else (takes[: args.folds] if args.folds else takes)
    results: dict[str, dict] = {name: {"folds": []} for name in model_names}

    for val_take in fold_takes:
        train_takes = [t for t in takes if t.take_id != val_take.take_id]
        if not train_takes:
            continue
        train_ds = KnockWindows(train_takes, frames, knock_rings, args.window, args.size,
                                augment=args.augment, roi_zoom=args.roi_zoom,
                                hard_mask=hard_masks, hard_weight=args.hard_negative)
        val_ds = KnockWindows([val_take], frames, knock_rings, args.window, args.size)
        if args.hard_negative > 0:
            sampler = torch.utils.data.WeightedRandomSampler(
                train_ds.weights, num_samples=len(train_ds.weights), replacement=True)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, sampler=sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch, shuffle=False)
        n_val = len(knock_rings[val_take.take_id])
        print(f"\nfold val={val_take.take_id}: train_windows={len(train_ds)} val_windows={len(val_ds)}")
        # 只在训练 take 上算正负比，用来平衡 BCE
        pos = sum(float(np.nansum(knock_rings[t.take_id])) for t in train_takes)
        valid = sum(float(np.sum(~np.isnan(knock_rings[t.take_id]))) for t in train_takes)
        neg = max(valid - pos, 0.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device) if pos > 0 else None
        print(f"  train positives={int(pos)} negatives={int(neg)} pos_weight={None if pos_weight is None else round(float(pos_weight.item()), 3)}")
        for name in model_names:
            torch.manual_seed(args.seed)
            model = build_model(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1)) if args.cosine else None
            t0 = time.time()
            for epoch in range(1, args.epochs + 1):
                loss = train_epoch(model, train_loader, device, optimizer, pos_weight, mixup_p=args.mixup,
                                   loss_name=args.loss, focal_gamma=args.focal_gamma)
                if scheduler is not None:
                    scheduler.step()
                if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                    print(f"  [{name}] epoch {epoch:3d} loss={loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if args.threshold > 0:
                threshold = args.threshold
            else:
                threshold, train_score = pick_threshold(
                    model, train_takes, frames, knock_rings, gt_events,
                    args.window, args.size, device, args.smooth_frames,
                    args.min_event_frames, args.max_gap_frames, args.threshold_metric,
                    normalize=args.normalize,
                    pair_min_gap_s=args.pair_min_gap_s, pair_max_gap_s=args.pair_max_gap_s,
                    pair_margin_s=args.pair_margin_s, peak_min_gap=args.peak_min_gap,
                    pair_isolation_s=args.pair_isolation_s,
                )
                print(f"  [{name}] threshold={threshold:.2f} (train {args.threshold_metric} score={train_score:.4f})")
            if args.pair_threshold_mode == "auto":
                if args.pair_threshold > -100:
                    pair_threshold = args.pair_threshold
                else:
                    pair_threshold, pair_train_score = pick_pair_threshold(
                        model, train_takes, frames, knock_rings, gt_events, args.window, args.size,
                        device, args.smooth_frames, args.normalize,
                        args.pair_min_gap_s, args.pair_max_gap_s, args.pair_margin_s,
                        args.peak_min_gap, args.pair_isolation_s)
                    print(f"  [{name}] pair_threshold={pair_threshold:.2f} (train pairF1={pair_train_score:.3f})")
            else:
                pair_threshold = threshold
            prob = predict_frames(model, val_loader, n_val, args.window, device)
            prob = smooth_prob(prob, args.smooth_frames)
            if args.normalize != "none":
                prob = normalize_probs({val_take.take_id: prob}, args.normalize)[val_take.take_id]
            target = knock_rings[val_take.take_id].astype(np.float32)
            fm = frame_metrics(prob, target, threshold)
            pred_events = [
                (a / val_take.fps, b / val_take.fps)
                for a, b in runs_above(prob, threshold, args.min_event_frames, args.max_gap_frames)
            ]
            em = event_metrics(pred_events, gt_events[val_take.take_id])
            hm = hit_metrics(pred_events, gt_events[val_take.take_id], val_take.duration_s)
            peaks = peak_indices(prob, threshold, args.peak_min_gap)
            pm_evt = pair_metrics(prob, val_take.fps, threshold, gt_events[val_take.take_id],
                                 val_take.duration_s, args.pair_min_gap_s, args.pair_max_gap_s,
                                 args.pair_margin_s, args.peak_min_gap, args.pair_isolation_s)
            pm_auto = pair_metrics(prob, val_take.fps, pair_threshold, gt_events[val_take.take_id],
                                   val_take.duration_s, args.pair_min_gap_s, args.pair_max_gap_s,
                                   args.pair_margin_s, args.peak_min_gap, args.pair_isolation_s)
            row = {"fold": val_take.take_id, "model": name, "threshold": threshold,
                   "pair_threshold": pair_threshold,
                   "train_s": round(time.time() - t0, 1), "frame": fm, "event": em, "hit": hm,
                   "pair": pm_evt, "pair_auto": pm_auto, "peaks": len(peaks)}
            results[name]["folds"].append(row)
            if args.save_folds:
                folds_dir = out_dir / "folds"
                folds_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": name,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "window": args.window, "size": args.size, "normalize": args.normalize,
                    "threshold": float(threshold), "pair_threshold": float(pair_threshold),
                    "pair_min_gap_s": args.pair_min_gap_s, "pair_max_gap_s": args.pair_max_gap_s,
                    "pair_margin_s": args.pair_margin_s, "pair_isolation_s": args.pair_isolation_s,
                    "smooth_frames": args.smooth_frames, "min_event_frames": args.min_event_frames,
                    "max_gap_frames": args.max_gap_frames, "peak_min_gap": args.peak_min_gap,
                    "trained_on": [t.take_id for t in train_takes],
                    "val_take": val_take.take_id,
                }, folds_dir / f"{val_take.take_id}_{name}.pt")
            print(f"  [{name}] frameF1={fm['f1']:.4f} P={fm['precision']:.4f} R={fm['recall']:.4f} AUC={fm['auroc']:.4f} | "
                  f"eventF1={em['f1']:.3f} ({em['tp']}/{em['n_pred']} pred, {em['n_gt']} gt) | "
                  f"pairF1(evt)={pm_evt['f1']:.3f} pair_hit={pm_evt['hit_rate']:.2f} pairFA/min={pm_evt['fa_per_min']:.1f} | "
                  f"pairF1(auto)={pm_auto['f1']:.3f} | hit={hm['hit_rate']:.2f} FA/min={hm['fa_per_min']:.1f} peaks={len(peaks)} ({row['train_s']}s)")

    summary = {}
    for name, data in results.items():
        folds = data["folds"]
        if not folds:
            continue
        agg = {}
        for key, getter in [
            ("frame_f1", lambda f: f["frame"]["f1"]),
            ("frame_auroc", lambda f: f["frame"]["auroc"]),
            ("event_f1", lambda f: f["event"]["f1"]),
            ("event_precision", lambda f: f["event"]["precision"]),
            ("event_recall", lambda f: f["event"]["recall"]),
            ("hit_rate", lambda f: f["hit"]["hit_rate"]),
            ("fa_per_min", lambda f: f["hit"]["fa_per_min"]),
            ("pair_f1", lambda f: f["pair"]["f1"]),
            ("pair_hit_rate", lambda f: f["pair"]["hit_rate"]),
            ("pair_fa_per_min", lambda f: f["pair"]["fa_per_min"]),
            ("pair_auto_f1", lambda f: f["pair_auto"]["f1"]),
            ("peaks", lambda f: f["peaks"]),
        ]:
            vals = np.array([getter(f) for f in folds], dtype=np.float64)
            agg[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "values": vals.tolist()}
        summary[name] = agg

    payload = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset": str(dataset_dir),
               "config": vars(args), "takes": [t.take_id for t in takes],
               "results": results, "summary": summary}
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 敲击检测（留一 take）", "",
             f"- 数据：{', '.join(t.take_id for t in takes)}",
             f"- 窗口={args.window} 帧，输入={args.size}px，epochs={args.epochs}，阈值={args.threshold}，"
             f"normalize={args.normalize}，mixup={args.mixup}，loss={args.loss}，roi_zoom={args.roi_zoom}",
             "", "| 模型 | 帧级 F1 | AUROC | 事件级 F1 | 命中率 | 误报/分钟 | 两敲 F1(事件阈值) | 两敲命中率 | 两敲误报/分钟 | 两敲 F1(自动阈值) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, agg in summary.items():
        lines.append(f"| {name} | {agg['frame_f1']['mean']:.4f} ± {agg['frame_f1']['std']:.4f} "
                     f"| {agg['frame_auroc']['mean']:.4f} ± {agg['frame_auroc']['std']:.4f} "
                     f"| {agg['event_f1']['mean']:.4f} ± {agg['event_f1']['std']:.4f} "
                     f"| {agg['hit_rate']['mean']:.3f} | {agg['fa_per_min']['mean']:.1f} "
                     f"| {agg['pair_f1']['mean']:.4f} ± {agg['pair_f1']['std']:.4f} "
                     f"| {agg['pair_hit_rate']['mean']:.3f} | {agg['pair_fa_per_min']['mean']:.1f} "
                     f"| {agg['pair_auto_f1']['mean']:.4f} |")
    lines += ["", "逐折："]
    for name, data in results.items():
        for f in data["folds"]:
            lines.append(f"- {name} / val={f['fold']}: frameF1={f['frame']['f1']:.4f} AUC={f['frame']['auroc']:.4f} "
                         f"eventF1={f['event']['f1']:.3f} hit={f['hit']['hit_rate']:.2f} "
                         f"FA/min={f['hit']['fa_per_min']:.1f} | pairF1={f['pair']['f1']:.3f} "
                         f"pair_detected={f['pair']['triggers']} pair_hit={f['pair']['hit_rate']:.2f} "
                         f"pair_FA/min={f['pair']['fa_per_min']:.1f} | peaks={f['peaks']} "
                         f"thr={f['threshold']:.2f} pair_thr={f['pair_threshold']:.2f}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.final_model:
        print("\n== final model on all takes ==")
        all_ds = KnockWindows(takes, frames, knock_rings, args.window, args.size,
                              augment=args.augment, roi_zoom=args.roi_zoom,
                              hard_mask=hard_masks, hard_weight=args.hard_negative)
        if args.hard_negative > 0:
            all_sampler = torch.utils.data.WeightedRandomSampler(
                all_ds.weights, num_samples=len(all_ds.weights), replacement=True)
            all_loader = torch.utils.data.DataLoader(all_ds, batch_size=args.batch, sampler=all_sampler)
        else:
            all_loader = torch.utils.data.DataLoader(all_ds, batch_size=args.batch, shuffle=True)
        pos = sum(float(np.nansum(knock_rings[t.take_id])) for t in takes)
        valid = sum(float(np.sum(~np.isnan(knock_rings[t.take_id]))) for t in takes)
        neg = max(valid - pos, 0.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device) if pos > 0 else None
        for name in model_names:
            torch.manual_seed(args.seed)
            model = build_model(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1)) if args.cosine else None
            for epoch in range(1, args.epochs + 1):
                loss = train_epoch(model, all_loader, device, optimizer, pos_weight, mixup_p=args.mixup,
                                   loss_name=args.loss, focal_gamma=args.focal_gamma)
                if scheduler is not None:
                    scheduler.step()
                if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                    print(f"  [final {name}] epoch {epoch:3d} loss={loss:.4f}")
            if args.threshold > 0:
                threshold, score = args.threshold, float("nan")
            else:
                threshold, score = pick_threshold(
                    model, takes, frames, knock_rings, gt_events,
                    args.window, args.size, device, args.smooth_frames,
                    args.min_event_frames, args.max_gap_frames, args.threshold_metric,
                    normalize=args.normalize,
                    pair_min_gap_s=args.pair_min_gap_s, pair_max_gap_s=args.pair_max_gap_s,
                    pair_margin_s=args.pair_margin_s, peak_min_gap=args.peak_min_gap,
                    pair_isolation_s=args.pair_isolation_s,
                )
            if args.pair_threshold_mode == "auto":
                if args.pair_threshold > -100:
                    pair_threshold, pair_score = args.pair_threshold, float("nan")
                else:
                    pair_threshold, pair_score = pick_pair_threshold(
                        model, takes, frames, knock_rings, gt_events, args.window, args.size,
                        device, args.smooth_frames, args.normalize,
                        args.pair_min_gap_s, args.pair_max_gap_s, args.pair_margin_s,
                        args.peak_min_gap, args.pair_isolation_s)
                    print(f"  [final {name}] pair_threshold={pair_threshold:.2f} (train pairF1={pair_score:.3f})")
            else:
                pair_threshold = threshold
            ckpt = {
                "model": name,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "window": args.window,
                "size": args.size,
                "normalize": args.normalize,
                "threshold": float(threshold),
                "pair_threshold": float(pair_threshold),
                "pair_min_gap_s": args.pair_min_gap_s,
                "pair_max_gap_s": args.pair_max_gap_s,
                "pair_margin_s": args.pair_margin_s,
                "pair_isolation_s": args.pair_isolation_s,
                "smooth_frames": args.smooth_frames,
                "min_event_frames": args.min_event_frames,
                "max_gap_frames": args.max_gap_frames,
                "peak_min_gap": args.peak_min_gap,
                "takes": [t.take_id for t in takes],
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            path = out_dir / f"model_{name}.pt"
            torch.save(ckpt, path)
            print(f"  saved {path} (threshold={threshold:.3f}, train {args.threshold_metric} score={score:.3f})")

    print(f"\nmetrics -> {out_dir / 'metrics.json'}")
    print(f"report  -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())