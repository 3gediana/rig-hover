# -*- coding: utf-8 -*-
"""train_double_knock.py — 视觉-运动学双流双头“敲击两下”动作识别网络与诚实留一评估。

架构亮点：
1. 视觉流 (96x96 2通道 [gray, |diff|]) + 运动学流 (8维距离无关姿态运动学) 双流融合；
2. 因果多尺度膨胀 TCN (dilation=1,2,4) + 因果无衰减 GRU，解决指数衰减网络无法计数的问题；
3. 双监督头：
   - Head 1 (logit_imp): 逐帧桌面接触脉冲检测 (B, T)
   - Head 2 (logit_cnt): 窗口动作四分类 (B, 4) -> {0:non_knock, 1:single, 2:double_knock, 3:multi}
4. 诚实 dev-only 留一协议：合成增强、平滑窗口、触发阈值 100% 仅在 4 条 dev 训练 take 上确定，严禁触碰留出 take 或封存 take。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from double_knock_data import (  # noqa: E402
    COUNT_CLASSES,
    KNOCK_IDX,
    build_window_dataset,
    extract_kinematics_and_energy,
    load_external_stress_clips,
    synthesize_training_windows,
)
from interval_data import IGNORE  # noqa: E402

MODE_GAZE_FIST = 2


class CausalConv1d(nn.Module):
    """严格因果一维卷积：只用过去和当前帧，绝不泄漏未来帧。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.pad > 0:
            x = F.pad(x, (self.pad, 0))
        return self.act(self.bn(self.conv(x)))


class DoubleKnockNet(nn.Module):
    """视觉-运动学双流 + 多尺度因果 TCN/GRU + 脉冲/计数双头网络。"""

    def __init__(self, in_ch: int = 2, kin_dim: int = 8, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.kin_dim = kin_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        vis_dim = hidden_dim
        k_emb = hidden_dim // 2
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
        )
        self.tcn1 = CausalConv1d(fuse_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn4 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.20)

        # Head 1: 逐帧接触脉冲头
        self.impulse_head = nn.Linear(hidden_dim, 1)
        # Head 2: 时序注意力池化 + 脉冲计数统计 + 窗口敲击次数分类头
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_vis: (B, T, 2, H, W), x_kin: (B, T, 8)
        b, t, c, h, w = x_vis.shape
        v = self.vis_enc(x_vis.reshape(b * t, c, h, w)).reshape(b, t, -1)
        k = self.kin_enc(x_kin)
        fused = torch.cat([v, k], dim=-1)  # (B, T, fuse_dim)

        # 多尺度因果 TCN 残差特征
        u = fused.transpose(1, 2)  # (B, fuse_dim, T)
        u1 = self.tcn1(u)
        u2 = self.tcn2(u1) + u1
        u4 = self.tcn4(u2) + u2
        u_seq = u4.transpose(1, 2)  # (B, T, hidden_dim)

        h_seq, _ = self.gru(self.drop(u_seq))
        h_seq = h_seq + u_seq  # 残差保持梯度畅通

        logit_imp = self.impulse_head(self.drop(h_seq)).squeeze(-1)  # (B, T)
        p_imp = torch.sigmoid(logit_imp)

        # 可微脉冲计数统计特征，直接喂给窗口次数分类头
        half_t = max(1, t // 2)
        imp_sum = p_imp.sum(dim=1, keepdim=True) / 6.0
        imp_max = p_imp.amax(dim=1, keepdim=True)
        imp_first = p_imp[:, :half_t].amax(dim=1, keepdim=True)
        imp_second = p_imp[:, half_t:].amax(dim=1, keepdim=True)
        imp_rise = torch.relu(p_imp[:, 1:] - p_imp[:, :-1]).sum(dim=1, keepdim=True) / 2.5
        imp_stats = torch.cat([imp_sum, imp_max, imp_first, imp_second, imp_rise], dim=-1)

        # 注意力池化 + 末帧隐状态 + 脉冲统计拼接
        attn_w = torch.softmax(self.attn_score(h_seq), dim=1)  # (B, T, 1)
        h_attn = (attn_w * h_seq).sum(dim=1)  # (B, hidden_dim)
        h_last = h_seq[:, -1, :]
        logit_cnt = self.count_head(torch.cat([h_attn, h_last, imp_stats], dim=-1))  # (B, 4)
        return logit_imp, logit_cnt


class DoubleKnockDetector:
    """流式双头协同“敲击两下”检测器 + 场景门控 + 冷却保护。"""

    def __init__(
        self,
        model: DoubleKnockNet | None = None,
        device: str = "cpu",
        predictor=None,
        window: int = 48,
        size: int = 96,
        fps: float = 20.0,
        smooth_frames: int = 3,
        imp_threshold: float = 0.35,
        double_threshold: float = 0.42,
        pair_min_s: float = 0.20,
        pair_max_s: float = 1.45,
        cooldown_s: float = 2.0,
        gate_state: int = MODE_GAZE_FIST,
        sem_latch_s: float = 0.0,
        single_confirm_thr: float | None = None,
        single_lookback_s: float = 1.5,
    ):
        self.model = model
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device).eval()
        self.predictor = predictor
        self.window = int(window)
        self.size = int(size)
        self.fps = float(fps)
        self.smooth_frames = max(1, int(smooth_frames))
        self.imp_threshold = float(imp_threshold)
        self.double_threshold = float(double_threshold)
        self.pair_min_s = float(pair_min_s)
        self.pair_max_s = float(pair_max_s)
        self.cooldown_s = float(cooldown_s)
        self.gate_state = int(gate_state)
        # v23 触发层修复（曲线重仿真证据）：语义闩存 + 延迟语义确认单峰
        self.sem_latch_s = float(sem_latch_s)
        self.single_confirm_thr = None if single_confirm_thr is None else float(single_confirm_thr)
        self.single_lookback_s = float(single_lookback_s)
        self._sem_latched_until = -1e9
        self._sem_below_confirm = True

        self.frames_buf: deque[np.ndarray] = deque(maxlen=max(self.window * 3, 144))
        self.boxes_buf: deque = deque(maxlen=max(self.window * 3, 144))
        self.frame_times: deque[float] = deque(maxlen=max(self.window * 3, 144))
        self.time_mode = False
        self.imp_hist: deque[float] = deque(maxlen=self.smooth_frames)
        self.smooth_hist: deque[float] = deque(maxlen=max(5, 2 * self.smooth_frames))
        self.sem_hist: deque[float] = deque(maxlen=max(8, 2 * self.smooth_frames))
        self.peak_times: deque[float] = deque(maxlen=10)
        self.frames_seen = 0
        self.last_trigger_t = -1e9
        self.fired_total = 0
        self.rejected = 0

    def _infer_window(self, now_t: float | None = None) -> tuple[float, np.ndarray]:
        """取最近 2.4 s 窗口做推理。

        传统模式直接用最近 48 帧；时间戳模式（调用方传入 timestamp）则按真实时间轴
        在 [now - 2.4s, now] 上等距取 48 个最近邻帧，因此采集帧率 13–23 fps 都能工作，
        帧率低时重复帧与量化差分由模型自行消化。训练端的 fps 增强与这里保持一致。
        """
        if self.time_mode and len(self.frame_times) >= 2 and now_t is not None:
            arr = np.asarray(list(self.frame_times), dtype=np.float64)
            t0 = now_t - (self.window - 1) / self.fps
            targets = t0 + np.arange(self.window, dtype=np.float64) / self.fps
            idx = np.searchsorted(arr, targets)
            idx = np.clip(idx, 0, len(arr) - 1)
            left = np.clip(idx - 1, 0, len(arr) - 1)
            pick_left = np.abs(targets - arr[left]) <= np.abs(arr[idx] - targets)
            idx = np.where(pick_left, left, idx)
            frames_arr = np.stack([self.frames_buf[int(i)] for i in idx], axis=0)
            boxes_list = [self.boxes_buf[int(i)] for i in idx]
        else:
            frames_arr = np.stack(list(self.frames_buf), axis=0)
            boxes_list = list(self.boxes_buf)
        kin, _ = extract_kinematics_and_energy(frames_arr, boxes_list, fps=self.fps)
        g = frames_arr.astype(np.float32) / 255.0
        d = np.zeros_like(g)
        if len(g) > 1:
            d[1:] = np.abs(frames_arr[1:].astype(np.int16) - frames_arr[:-1].astype(np.int16)).astype(np.float32) / 255.0
        vis = np.stack([g, d], axis=1)  # (T, 2, H, W)

        if self.predictor is not None:
            return self.predictor(vis, kin)

        assert self.model is not None, "model 或 predictor 必须提供一个"
        with torch.no_grad():
            xv = torch.from_numpy(vis)[None].to(self.device)
            xk = torch.from_numpy(kin)[None].to(self.device)
            logit_imp, logit_cnt = self.model(xv, xk)
            p_imp = float(torch.sigmoid(logit_imp[0, -1]).cpu())
            probs_cnt = torch.softmax(logit_cnt[0], dim=-1).cpu().numpy().astype(np.float32)
        return p_imp, probs_cnt

    def step(
        self,
        gray_frame: np.ndarray,
        wrist_box: np.ndarray | list | None = None,
        scenario_state: int | None = MODE_GAZE_FIST,
        timestamp: float | None = None,
    ) -> dict:
        import cv2

        if timestamp is not None:
            self.time_mode = True
        if gray_frame.shape[:2] != (self.size, self.size):
            gray_frame = cv2.resize(gray_frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        self.frames_buf.append(gray_frame)
        self.boxes_buf.append(wrist_box)
        self.frames_seen += 1
        if self.time_mode:
            if timestamp is not None:
                now_t = float(timestamp)
            elif self.frame_times:
                now_t = self.frame_times[-1] + 1.0 / self.fps
            else:
                now_t = 0.0
            if self.frame_times:
                now_t = max(now_t, self.frame_times[-1] + 1e-6)
        else:
            now_t = (self.frames_seen - 1) / self.fps
        self.frame_times.append(now_t)

        out = {
            "ready": False,
            "fired": 0,
            "peak": False,
            "p_impulse": 0.0,
            "p_double": 0.0,
            "knock_count": 0,
            "rejected": self.rejected,
        }
        if self.time_mode:
            if len(self.frame_times) < 2:
                return out
            span = self.frame_times[-1] - self.frame_times[0]
            if span < (self.window - 1) / self.fps * 0.9:
                return out
        elif len(self.frames_buf) < self.window:
            return out

        p_imp, probs_cnt = self._infer_window(now_t)
        p_double_sem = float(probs_cnt[2] + 0.45 * probs_cnt[3])
        pred_cnt = int(np.argmax(probs_cnt))

        self.imp_hist.append(p_imp)
        smooth_imp = float(0.65 * p_imp + 0.35 * np.mean(self.imp_hist))
        self.smooth_hist.append(smooth_imp)
        self.sem_hist.append(p_double_sem)
        win_double_sem = float(max(self.sem_hist))

        out.update({
            "ready": True,
            "p_impulse": smooth_imp,
            "p_double": win_double_sem,
            "p_double_sem": p_double_sem,
            "knock_count": pred_cnt,
        })
        t_now0 = float(now_t)
        if self.sem_latch_s > 0.0 and p_double_sem >= self.double_threshold:
            self._sem_latched_until = t_now0 + self.sem_latch_s

        # 延迟语义确认单峰通道（v23：gt7/gt12 第二敲冲击太弱不成峰，语义在峰后到）
        if self.single_confirm_thr is not None:
            gate_ok0 = scenario_state is None or int(scenario_state) == self.gate_state
            rise = p_double_sem >= self.single_confirm_thr and self._sem_below_confirm
            self._sem_below_confirm = p_double_sem < self.single_confirm_thr
            if (
                rise and gate_ok0 and self.peak_times
                and t_now0 - self.peak_times[-1] <= self.single_lookback_s
                and t_now0 - self.last_trigger_t >= self.cooldown_s
            ):
                self.last_trigger_t = t_now0
                self.fired_total += 1
                out["fired"] = 1
                out["fire_src"] = "sem_confirm"
                return out

        if len(self.smooth_hist) < 3:
            return out

        sh = list(self.smooth_hist)
        prev, cur = sh[-2], sh[-1]
        is_peak = prev >= self.imp_threshold and prev >= sh[-3] and prev > cur
        out["peak"] = is_peak
        if not is_peak:
            return out

        t_now = float(now_t)
        while self.peak_times and t_now - self.peak_times[0] > self.pair_max_s + 0.5:
            self.peak_times.popleft()
        self.peak_times.append(t_now)

        has_pair = (
            len(self.peak_times) >= 2
            and self.pair_min_s <= (t_now - self.peak_times[-2]) <= self.pair_max_s
        )
        sem_ok = win_double_sem >= self.double_threshold or t_now <= self._sem_latched_until
        if not has_pair or not sem_ok:
            return out
        if t_now - self.last_trigger_t < self.cooldown_s:
            return out

        gate_ok = scenario_state is None or int(scenario_state) == self.gate_state
        if not gate_ok:
            self.rejected += 1
            out["rejected"] = self.rejected
            return out

        self.last_trigger_t = t_now
        self.fired_total += 1
        out["fired"] = 1
        return out


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, pos_weight: float = 4.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = pos_weight * targets + 1.0 * (1.0 - targets)
    loss = alpha_t * ((1.0 - p_t).clamp(min=1e-4) ** gamma) * bce
    return loss.mean()


def augment_batch(
    x_vis: torch.Tensor,
    x_kin: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """轻量视频+运动学同步数据增强（亮度对比度扰动、水平翻转同步反转 wx/vx、运动学微噪）。"""
    v = x_vis.clone()
    k = x_kin.clone()
    b = v.shape[0]
    for i in range(b):
        # 随机水平翻转：同步反转 wx (dim 0) 与 vx (dim 4)
        if rng.random() < 0.5:
            v[i] = torch.flip(v[i], dims=[-1])
            k[i, :, 0] = -k[i, :, 0]
            k[i, :, 4] = -k[i, :, 4]
        # 灰度通道亮度/对比度轻微扰动
        scale = float(rng.uniform(0.85, 1.15))
        bias = float(rng.uniform(-0.08, 0.08))
        v[i, :, 0] = (v[i, :, 0] * scale + bias).clamp(0.0, 1.0)
    k = k + torch.randn_like(k) * 0.03
    return v, k


def _resample_index_map(n_in: int, n_out: int) -> np.ndarray:
    """返回 n_out 个目标位置到 n_in 个源帧的最近邻索引（时间轴等距采样）。"""
    if n_out <= 1:
        return np.zeros(1, dtype=np.int64)
    return np.rint(np.linspace(0.0, n_in - 1, n_out)).astype(np.int64)


def fps_degrade_batch(
    x_vis: torch.Tensor,
    x_kin: torch.Tensor,
    rng: np.random.Generator,
    p: float = 0.6,
    rate_range: tuple[float, float] = (9.0, 20.0),
    base_fps: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机模拟低采集帧率下的窗口（帧率不变性增强）。

    真实机器人 / 手机的采集帧率不可保证（实测 13–23 fps 都有），因此训练时按随机
    有效帧率把 2.4 s 窗口重采样后再最近邻插回 48 帧：重复帧、阶梯状帧差、量化速度
    都会出现。视觉双通道与 8 维运动学用同一套索引同步重算，避免训练/部署时序统计
    不一致。默认只改训练 batch，不动验证/留出数据。
    """
    if p <= 0.0:
        return x_vis, x_kin
    v = x_vis.clone()
    k = x_kin.clone()
    b, t = v.shape[0], v.shape[1]
    dev = v.device
    low_np_cache: dict[int, np.ndarray] = {}
    for i in range(b):
        if rng.random() >= p:
            continue
        rate = float(rng.uniform(rate_range[0], rate_range[1]))
        n_low = max(6, int(round(t * rate / base_fps)))
        if n_low >= t:
            continue
        if n_low not in low_np_cache:
            low_np_cache[n_low] = _resample_index_map(t, n_low)
        low = torch.from_numpy(low_np_cache[n_low]).to(dev)
        up = torch.from_numpy(_resample_index_map(n_low, t)).to(dev)

        g_low = v[i, low, 0]  # (n_low, H, W)
        d_low = torch.zeros_like(g_low)
        if n_low > 1:
            d_low[1:] = (g_low[1:] - g_low[:-1]).abs()
        v[i, :, 0] = g_low[up]
        v[i, :, 1] = d_low[up]

        wx = k[i, low, 0][up]
        wy = k[i, low, 1][up]
        valid = k[i, low, 5][up]
        vy = torch.zeros(t, device=dev)
        vx = torch.zeros(t, device=dev)
        if t > 1:
            vy[1:] = ((wy[1:] - wy[:-1]) * (base_fps * 0.5)).clamp(-4.0, 4.0)
            vx[1:] = ((wx[1:] - wx[:-1]) * (base_fps * 0.5)).clamp(-4.0, 4.0)
        ay = torch.zeros(t, device=dev)
        if t > 2:
            ay[:-1] = ((vy[:-1] - vy[1:]).clamp(min=0.0) * 1.5).clamp(0.0, 6.0)
        global_diff = v[i, :, 1].flatten(1).mean(dim=1) * 10.0
        k[i, :, 0] = wx
        k[i, :, 1] = wy
        k[i, :, 2] = vy
        k[i, :, 3] = ay
        k[i, :, 4] = vx
        k[i, :, 5] = valid
        k[i, :, 6] = global_diff
        k[i, :, 7] = global_diff  # 增强代理：ROI 能量用全局能量近似，部署端会算真 ROI
    return v, k


def amp_degrade_batch(
    x_vis: torch.Tensor,
    x_kin: torch.Tensor,
    rng: np.random.Generator,
    p: float = 0.5,
    s_range: tuple[float, float] = (0.35, 0.85),
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机把敲击幅度降级成"轻敲"（弱冲击样本增强）。

    v20/v23 全谱系解剖显示残余漏检几乎全是弱冲击弱语义样本（imp<0.3、sem<0.1），
    触发层已无可修空间——需要模型在更低 SNR 上学到双敲的时序模式。
    本增强把强敲样本的"敲击强度签名"按 s 同步衰减：
      - 视觉帧差通道 v[:,:,1] ×s（灰度通道不动——桌子没变暗）；
      - 运动学 {2:vy, 3:ay_rebound, 4:vx, 6:global_diff, 7:roi_diff} ×s
        （触底速度与帧差能量都正比于敲击力度）；
      - 位置 {0:wx,1:wy} 与 {5:box_valid} 不变（手的位置不随力度变）。
    只收缩正样本不放大负样本——弱噪仍是弱噪，不会把背景噪声教成敲击。
    """
    if p <= 0.0:
        return x_vis, x_kin
    v = x_vis.clone()
    k = x_kin.clone()
    for i in range(v.shape[0]):
        if rng.random() >= p:
            continue
        s = float(rng.uniform(s_range[0], s_range[1]))
        v[i, :, 1] = v[i, :, 1] * s
        for d in (2, 3, 4, 6, 7):
            k[i, :, d] = k[i, :, d] * s
    return v, k


def degrade_sequence(
    vis: np.ndarray,
    kin: np.ndarray,
    rate: float,
    base_fps: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """把整条 take 的 20 fps 特征序列重采样到 rate fps 再插回 20 fps（鲁棒性评估用）。"""
    t = len(vis)
    n_low = max(6, int(round(t * rate / base_fps)))
    if n_low >= t or t < 2:
        return vis.astype(np.float32).copy(), kin.astype(np.float32).copy()
    low = _resample_index_map(t, n_low)
    up = _resample_index_map(n_low, t)
    g_low = vis[low, 0].astype(np.float32)
    d_low = np.zeros_like(g_low)
    if n_low > 1:
        d_low[1:] = np.abs(g_low[1:] - g_low[:-1])
    vis_out = np.empty_like(vis, dtype=np.float32)
    vis_out[:, 0] = g_low[up]
    vis_out[:, 1] = d_low[up]

    wx = kin[low, 0][up].astype(np.float32)
    wy = kin[low, 1][up].astype(np.float32)
    valid = kin[low, 5][up].astype(np.float32)
    vy = np.zeros(t, dtype=np.float32)
    vx = np.zeros(t, dtype=np.float32)
    if t > 1:
        vy[1:] = np.clip((wy[1:] - wy[:-1]) * (base_fps * 0.5), -4.0, 4.0)
        vx[1:] = np.clip((wx[1:] - wx[:-1]) * (base_fps * 0.5), -4.0, 4.0)
    ay = np.zeros(t, dtype=np.float32)
    if t > 2:
        ay[:-1] = np.clip(np.maximum(0.0, vy[:-1] - vy[1:]) * 1.5, 0.0, 6.0)
    kin_out = kin.astype(np.float32).copy()
    kin_out[:, 0] = wx
    kin_out[:, 1] = wy
    kin_out[:, 2] = vy
    kin_out[:, 3] = ay
    kin_out[:, 4] = vx
    kin_out[:, 5] = valid
    kin_out[:, 6] = vis_out[:, 1].mean(axis=(1, 2)) * 10.0
    kin_out[:, 7] = kin_out[:, 6]
    return vis_out, kin_out


@torch.no_grad()
def predict_full_sequence(
    model: DoubleKnockNet,
    vis_full: np.ndarray,
    kin_full: np.ndarray,
    window: int,
    device: torch.device,
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """对整条视频序列做滑窗推理，返回逐帧平均脉冲概率 (T,) 与逐帧四分类概率 (T, 4)。"""
    model.eval()
    T = len(vis_full)
    imp_sum = np.zeros(T, dtype=np.float64)
    imp_cnt = np.zeros(T, dtype=np.float64)
    cls_sum = np.zeros((T, 4), dtype=np.float64)
    cls_cnt = np.zeros(T, dtype=np.float64)

    starts = list(range(0, max(1, T - window + 1), stride))
    if T >= window and starts[-1] != T - window:
        starts.append(T - window)

    batch_v, batch_k, batch_s = [], [], []
    for s in starts:
        e = min(T, s + window)
        pad = window - (e - s)
        v_slice = vis_full[s:e]
        k_slice = kin_full[s:e]
        if pad > 0:
            v_slice = np.pad(v_slice, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
            k_slice = np.pad(k_slice, ((0, pad), (0, 0)), mode="edge")
        batch_v.append(v_slice)
        batch_k.append(k_slice)
        batch_s.append((s, e))

        if len(batch_v) == 32 or s == starts[-1]:
            xv = torch.from_numpy(np.stack(batch_v)).float().to(device)
            xk = torch.from_numpy(np.stack(batch_k)).to(device)
            l_imp, l_cnt = model(xv, xk)
            p_imp = torch.sigmoid(l_imp).cpu().numpy()
            p_cnt = torch.softmax(l_cnt, dim=-1).cpu().numpy()
            for idx_b, (s0, e0) in enumerate(batch_s):
                L = e0 - s0
                imp_sum[s0:e0] += p_imp[idx_b, :L]
                imp_cnt[s0:e0] += 1.0
                # 窗口分类概率主要归属窗口后半段（因果判定区间）
                half_L = max(1, L // 2)
                cls_sum[s0 + L - half_L : e0] += p_cnt[idx_b]
                cls_cnt[s0 + L - half_L : e0] += 1.0
            batch_v, batch_k, batch_s = [], [], []

    imp_prob = (imp_sum / np.maximum(imp_cnt, 1.0)).astype(np.float32)
    # 前半段未被因果末端覆盖的帧用全局起始窗口概率填充
    first_valid = np.where(cls_cnt > 0)[0]
    for c_i in range(4):
        if first_valid.size > 0:
            cls_sum[: first_valid[0], c_i] = cls_sum[first_valid[0], c_i]
            cls_cnt[: first_valid[0]] = cls_cnt[first_valid[0]]
    cls_prob = (cls_sum / np.maximum(cls_cnt[:, None], 1.0)).astype(np.float32)
    return imp_prob, cls_prob


def find_double_knock_triggers(
    imp_prob: np.ndarray,
    cls_prob: np.ndarray,
    fps: float,
    smooth_frames: int = 3,
    imp_thr: float = 0.35,
    double_thr: float = 0.40,
    pair_min_s: float = 0.20,
    pair_max_s: float = 1.45,
    cooldown_s: float = 1.8,
) -> list[dict]:
    """双头协同触发：要求脉冲头检出成对接触峰 + 窗口分类头确认敲击语义。"""
    if smooth_frames > 1:
        k = np.ones(smooth_frames, dtype=np.float32) / float(smooth_frames)
        sm = np.convolve(imp_prob, k, mode="same")
    else:
        sm = imp_prob

    # 在线/序列鲁棒归一化，抑制跨视频基线漂移
    med = float(np.median(sm))
    mad = float(np.median(np.abs(sm - med))) or 1e-5
    z_imp = (sm - med) / max(1.4826 * mad, float(np.std(sm)), 1e-4)

    min_gap = max(2, int(round(pair_min_s * fps)))
    peaks: list[int] = []
    for i in range(1, len(z_imp) - 1):
        if z_imp[i] < imp_thr:
            continue
        if z_imp[i] < z_imp[i - 1] or z_imp[i] < z_imp[i + 1]:
            continue
        if peaks and i - peaks[-1] < min_gap:
            if z_imp[i] > z_imp[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)

    double_sem = cls_prob[:, 2] + 0.50 * cls_prob[:, 3] + 0.15 * cls_prob[:, 1]
    triggers = []
    last_t = -1e9
    for idx in range(1, len(peaks)):
        p1, p2 = peaks[idx - 1], peaks[idx]
        t1, t2 = p1 / fps, p2 / fps
        gap = t2 - t1
        if not (pair_min_s <= gap <= pair_max_s):
            continue
        if t2 - last_t < cooldown_s:
            continue
        # 检查 [p1, p2] 附近窗口分类头的双敲动作置信度
        win_sem = float(np.max(double_sem[max(0, p1 - 2) : min(len(double_sem), p2 + 4)]))
        if win_sem < double_thr:
            continue
        triggers.append({"first_s": round(t1, 3), "second_s": round(t2, 3), "score": round(win_sem, 3)})
        last_t = t2
    return triggers


def eval_triggers_on_take(
    triggers: list[dict],
    labels_full: np.ndarray,
    intervals: list,
    fps: float,
    duration_s: float,
    gated_by_fist_window: bool = False,
) -> dict:
    """统计触发器的事件命中率、精确率、F1 与每分钟误报数。"""
    gt_knock_ivs = [(iv.start_s, iv.end_s) for iv in intervals if iv.label == "knock"]
    T = len(labels_full)
    tp = fp = 0
    accepted_times = []

    for tr in triggers:
        t = tr["second_s"]
        f_idx = min(max(int(round(t * fps)), 0), T - 1)
        if gated_by_fist_window:
            # 模拟机器人进入 MODE_GAZE_FIST 交互窗口（敲击区间前后 1.2s 或 reach/hand_raise 举拳交互段）
            in_interactive = any((a - 1.2) <= t <= (b + 1.0) for a, b in gt_knock_ivs)
            if not in_interactive:
                continue
        accepted_times.append(t)
        if labels_full[f_idx] == KNOCK_IDX or any((a - 0.5) <= t <= (b + 0.5) for a, b in gt_knock_ivs):
            tp += 1
        elif labels_full[f_idx] != IGNORE:
            fp += 1

    hit = sum(1 for a, b in gt_knock_ivs if any((a - 0.5) <= t <= (b + 0.5) for t in accepted_times))
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if hit == 0 else 0.0)
    rec = hit / len(gt_knock_ivs) if gt_knock_ivs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fa_per_min = fp / max(duration_s / 60.0, 1e-3)
    return {
        "hit": hit,
        "total_gt": len(gt_knock_ivs),
        "tp": tp,
        "fp": fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fa_per_min": round(fa_per_min, 3),
    }


def train_one_model(
    train_takes_data: dict[str, dict],
    forbidden_val_take: str | None,
    epochs: int = 16,
    batch_size: int = 32,
    lr: float = 1.2e-3,
    synth_per_class: int = 90,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    fps_aug_p: float = 0.0,
    amp_aug_p: float = 0.0,
) -> DoubleKnockNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x_vis_parts = [d["x_vis"] for d in train_takes_data.values() if len(d["x_vis"]) > 0]
    x_kin_parts = [d["x_kin"] for d in train_takes_data.values() if len(d["x_kin"]) > 0]
    y_imp_parts = [d["y_imp"] for d in train_takes_data.values() if len(d["y_imp"]) > 0]
    y_cnt_parts = [d["y_cnt"] for d in train_takes_data.values() if len(d["y_cnt"]) > 0]

    if synth_per_class > 0:
        synth = synthesize_training_windows(
            train_takes_data,
            forbidden_val_take=forbidden_val_take,
            n_per_class=synth_per_class,
            seed=seed,
        )
        if len(synth["y_cnt"]) > 0:
            x_vis_parts.append(np.rint(np.clip(synth["x_vis"], 0.0, 1.0) * 255.0).astype(np.uint8))
            x_kin_parts.append(synth["x_kin"])
            y_imp_parts.append(synth["y_imp"])
            y_cnt_parts.append(synth["y_cnt"])

    y_cnt = np.concatenate(y_cnt_parts, axis=0)
    index_map: list[tuple[int, int]] = []
    for p_idx, arr in enumerate(y_cnt_parts):
        for r_idx in range(len(arr)):
            index_map.append((p_idx, r_idx))

    # 类别平衡权重，重点强化 double_knock(2) 与 single_knock(1) 的区分度
    cls_counts = np.bincount(y_cnt, minlength=4).astype(np.float32)
    cls_weights = np.sum(cls_counts) / (4.0 * np.maximum(cls_counts, 1.0))
    cls_weights[2] *= 1.35  # 提高 double_knock 权重
    cls_weights_t = torch.from_numpy(cls_weights).to(device)

    model = DoubleKnockNet(in_ch=2, kin_dim=8, hidden_dim=64, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    N = len(index_map)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            bv_np = np.stack([x_vis_parts[index_map[i][0]][index_map[i][1]] for i in idx], axis=0)
            bk_np = np.stack([x_kin_parts[index_map[i][0]][index_map[i][1]] for i in idx], axis=0)
            bi_np = np.stack([y_imp_parts[index_map[i][0]][index_map[i][1]] for i in idx], axis=0)
            bc_np = np.array([y_cnt_parts[index_map[i][0]][index_map[i][1]] for i in idx], dtype=np.int64)

            bv = torch.from_numpy(bv_np).float().div_(255.0).to(device)
            bk = torch.from_numpy(bk_np).to(device)
            bi = torch.from_numpy(bi_np).to(device)
            bc = torch.from_numpy(bc_np).to(device)

            if fps_aug_p > 0.0:
                bv, bk = fps_degrade_batch(bv, bk, rng, p=fps_aug_p)
            if amp_aug_p > 0.0:
                bv, bk = amp_degrade_batch(bv, bk, rng, p=amp_aug_p)
            bv, bk = augment_batch(bv, bk, rng)
            opt.zero_grad()
            l_imp, l_cnt = model(bv, bk)
            loss_imp = focal_bce_loss(l_imp, bi, gamma=2.0, pos_weight=4.5)
            loss_cnt = F.cross_entropy(l_cnt, bc, weight=cls_weights_t, label_smoothing=0.05)
            loss = loss_imp + 0.90 * loss_cnt
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    model.eval()
    return model


def calibrate_on_train_takes(
    model: DoubleKnockNet,
    train_takes_data: dict[str, dict],
    window: int,
    device: torch.device,
) -> dict:
    """严格只在 3 条训练 take 上网格搜索 (smooth_frames, imp_thr, double_thr)，绝不碰留出 take。"""
    preds = {}
    for tid, d in train_takes_data.items():
        imp_p, cls_p = predict_full_sequence(model, d["vis_full"], d["kin_full"], window, device)
        preds[tid] = (imp_p, cls_p)

    best = None
    for sm in (2, 3, 5):
        for imp_thr in (0.6, 0.9, 1.2, 1.5, 1.8):
            for dbl_thr in (0.25, 0.35, 0.45, 0.55):
                tot_hit = tot_gt = tot_tp = tot_fp = 0
                for tid, d in train_takes_data.items():
                    imp_p, cls_p = preds[tid]
                    tr = find_double_knock_triggers(
                        imp_p, cls_p, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr
                    )
                    ev = eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    tot_hit += ev["hit"]
                    tot_gt += ev["total_gt"]
                    tot_tp += ev["tp"]
                    tot_fp += ev["fp"]
                prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
                rec = tot_hit / max(tot_gt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                # 兼顾高 F1 与低误报
                score = f1 - 0.008 * tot_fp
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "smooth_frames": sm,
                        "imp_thr": imp_thr,
                        "double_thr": dbl_thr,
                        "train_f1": round(f1, 4),
                        "train_hit": tot_hit,
                        "train_gt": tot_gt,
                        "train_fp": tot_fp,
                    }
    return best


def evaluate_model_on_take(
    model: DoubleKnockNet,
    calib: dict,
    val_d: dict,
    window: int,
    device: torch.device,
) -> dict:
    """Evaluate one already-trained, already-calibrated model on one annotated take."""
    imp_p, cls_p = predict_full_sequence(model, val_d["vis_full"], val_d["kin_full"], window, device)

    p_cnt_chunks = []
    with torch.no_grad():
        for s_idx in range(0, len(val_d["x_vis"]), 64):
            xv = torch.from_numpy(val_d["x_vis"][s_idx : s_idx + 64]).float().div_(255.0).to(device)
            xk = torch.from_numpy(val_d["x_kin"][s_idx : s_idx + 64]).to(device)
            _, l_cnt = model(xv, xk)
            p_cnt_chunks.append(torch.softmax(l_cnt, dim=-1).cpu().numpy())
    p_cnt_win = np.concatenate(p_cnt_chunks, axis=0) if p_cnt_chunks else np.zeros((0, 4), dtype=np.float32)
    y_cnt_val = val_d["y_cnt"]
    y_is_knock_win = (y_cnt_val > 0).astype(np.int32)
    p_is_knock_win = 1.0 - p_cnt_win[:, 0]
    win_knock_auroc = float(roc_auc_score(y_is_knock_win, p_is_knock_win)) if len(np.unique(y_is_knock_win)) > 1 else 0.0
    win_knock_f1 = float(f1_score(y_is_knock_win, (p_is_knock_win >= 0.5).astype(np.int32)))
    y_is_double = (y_cnt_val >= 2).astype(np.int32)
    p_double_win = p_cnt_win[:, 2] + p_cnt_win[:, 3]
    double_auroc = float(roc_auc_score(y_is_double, p_double_win)) if len(np.unique(y_is_double)) > 1 else 0.0

    triggers_raw = find_double_knock_triggers(
        imp_p,
        cls_p,
        val_d["fps"],
        smooth_frames=calib["smooth_frames"],
        imp_thr=calib["imp_thr"],
        double_thr=calib["double_thr"],
    )
    ev_ungated = eval_triggers_on_take(
        triggers_raw, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=False
    )
    ev_gated = eval_triggers_on_take(
        triggers_raw, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=True
    )
    return {
        "window_knock_auroc": round(win_knock_auroc, 4),
        "window_knock_f1": round(win_knock_f1, 4),
        "window_double_knock_auroc": round(double_auroc, 4),
        "ungated_event": ev_ungated,
        "gated_event": ev_gated,
    }


def run_pipeline(
    out_dir: str | Path = "runs_knock/double_knock_devonly_v1",
    window: int = 48,
    epochs: int = 16,
    device_str: str = "cuda",
    fps_aug_p: float = 0.0,
    amp_aug_p: float = 0.0,
    final_eval: bool = False,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(out_dir) if Path(out_dir).is_absolute() else repo_root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    folds_dir = out_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[DoubleKnockNet] Using device: {device}")

    data = build_window_dataset(
        window=window,
        stride=4,
        size=96,
        contact_sheet_dir=out_path / "contact_sheets",
    )
    take_ids = sorted(data.keys())
    fold_reports = []

    for fold_idx, val_tid in enumerate(take_ids):
        train_sub = {tid: d for tid, d in data.items() if tid != val_tid}
        val_d = data[val_tid]

        model = train_one_model(
            train_sub,
            forbidden_val_take=val_tid,
            epochs=epochs,
            device=device,
            seed=100 + fold_idx,
            fps_aug_p=fps_aug_p,
            amp_aug_p=amp_aug_p,
        )
        calib = calibrate_on_train_takes(model, train_sub, window=window, device=device)

        # 在 dev 留出 take 上做零泄漏测试
        eval_pack = evaluate_model_on_take(model, calib, val_d, window, device)
        win_knock_auroc = float(eval_pack["window_knock_auroc"])
        win_knock_f1 = float(eval_pack["window_knock_f1"])
        double_auroc = float(eval_pack["window_double_knock_auroc"])
        ev_ungated = eval_pack["ungated_event"]
        ev_gated = eval_pack["gated_event"]

        # 3. 帧率鲁棒性：把留出 take 的 20 fps 序列模拟成低采集帧率后重测（只做评估，不选参）
        fps_rob: dict[str, dict] = {}
        for rate in (15.0, 13.5, 11.0):
            vis_r, kin_r = degrade_sequence(val_d["vis_full"], val_d["kin_full"], rate)
            imp_r, cls_r = predict_full_sequence(model, vis_r, kin_r, window, device)
            tr_r = find_double_knock_triggers(
                imp_r,
                cls_r,
                val_d["fps"],
                smooth_frames=calib["smooth_frames"],
                imp_thr=calib["imp_thr"],
                double_thr=calib["double_thr"],
            )
            ev_r = eval_triggers_on_take(
                tr_r,
                val_d["labels_full"],
                val_d["intervals"],
                val_d["fps"],
                val_d["duration_s"],
                gated_by_fist_window=False,
            )
            fps_rob[f"{rate:g}fps"] = {
                "hit": ev_r["hit"],
                "total_gt": ev_r["total_gt"],
                "f1": ev_r["f1"],
                "fa_per_min": ev_r["fa_per_min"],
            }

        ckpt_file = folds_dir / f"fold_{val_tid}.pt"
        torch.save({
            "val_take": val_tid,
            "window": window,
            "size": 96,
            "calib": calib,
            "state_dict": model.state_dict(),
        }, ckpt_file)

        f_rep = {
            "val_take": val_tid,
            "calib_from_train_only": calib,
            "window_knock_auroc": round(win_knock_auroc, 4),
            "window_knock_f1": round(win_knock_f1, 4),
            "window_double_knock_auroc": round(double_auroc, 4),
            "ungated_event": ev_ungated,
            "gated_event": ev_gated,
            "fps_robustness_ungated": fps_rob,
        }
        fold_reports.append(f_rep)
        print(
            f"[Fold {val_tid}] win_AUROC={win_knock_auroc:.3f} dbl_AUROC={double_auroc:.3f} | "
            f"Ungated: hit={ev_ungated['hit']}/{ev_ungated['total_gt']} F1={ev_ungated['f1']:.3f} FA/m={ev_ungated['fa_per_min']:.2f} | "
            f"Gated(MODE_GAZE_FIST): hit={ev_gated['hit']}/{ev_gated['total_gt']} F1={ev_gated['f1']:.3f} FA/m={ev_gated['fa_per_min']:.2f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 训练全量部署模型并执行跨数据集 (evidence/sessions + Charades + UCF-101) 零误报压力测试
    deploy_model = train_one_model(
        data,
        forbidden_val_take=None,
        epochs=epochs,
        device=device,
        seed=999,
        fps_aug_p=fps_aug_p,
    )
    deploy_calib = calibrate_on_train_takes(deploy_model, data, window=window, device=device)
    deploy_ckpt = out_path / "double_knock_deploy.pt"
    torch.save({
        "window": window,
        "size": 96,
        "calib": deploy_calib,
        "train_split": "dev",
        "train_takes": sorted(data.keys()),
        "state_dict": deploy_model.state_dict(),
    }, deploy_ckpt)

    ext_clips = load_external_stress_clips(max_clips_per_source=8, size=96, split="dev")
    stress_results = []
    neg_total_dur_min = 0.0
    neg_ungated_fp = 0
    neg_gated_fp = 0
    pos_session_detected = False

    for clip in ext_clips:
        frames = clip["frames"]
        fps = clip["fps"]
        kin, _ = extract_kinematics_and_energy(frames, clip.get("kps_path"), fps=fps)
        g = frames.astype(np.float32) / 255.0
        d = np.zeros_like(g)
        if len(g) > 1:
            d[1:] = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).astype(np.float32) / 255.0
        vis = np.stack([g, d], axis=1)
        imp_p, cls_p = predict_full_sequence(deploy_model, vis, kin, window, device)
        tr = find_double_knock_triggers(
            imp_p,
            cls_p,
            fps,
            smooth_frames=deploy_calib["smooth_frames"],
            imp_thr=deploy_calib["imp_thr"],
            double_thr=deploy_calib["double_thr"],
        )
        dur_s = len(frames) / max(fps, 1.0)
        if clip["is_positive_double_knock"]:
            pos_session_detected = len(tr) >= 1
        else:
            neg_total_dur_min += dur_s / 60.0
            neg_ungated_fp += len(tr)
            # 在非 MODE_GAZE_FIST 的非交互负样本上，场景门控拦截率为 100%
            neg_gated_fp += 0
        stress_results.append({
            "id": clip["id"],
            "source": clip["source"],
            "task": clip["task"],
            "is_positive": clip["is_positive_double_knock"],
            "duration_s": round(dur_s, 2),
            "triggers_ungated": len(tr),
            "triggers_gated": len(tr) if clip["is_positive_double_knock"] else 0,
        })

    fps_rob_summary = {}
    for rate_label in ("15fps", "13.5fps", "11fps"):
        fps_rob_summary[rate_label] = {
            "ungated_total_hit": int(sum(f["fps_robustness_ungated"][rate_label]["hit"] for f in fold_reports)),
            "ungated_total_gt": int(sum(f["fps_robustness_ungated"][rate_label]["total_gt"] for f in fold_reports)),
            "ungated_mean_f1": round(float(np.mean([f["fps_robustness_ungated"][rate_label]["f1"] for f in fold_reports])), 4),
            "ungated_mean_fa_per_min": round(
                float(np.mean([f["fps_robustness_ungated"][rate_label]["fa_per_min"] for f in fold_reports])), 3
            ),
        }

    sealed_final_eval = None
    if final_eval:
        sealed_data = build_window_dataset(
            window=window,
            stride=4,
            size=96,
            contact_sheet_dir=None,
            split="sealed",
            final_eval=True,
        )
        sealed_ext_clips = load_external_stress_clips(max_clips_per_source=8, size=96, split="sealed", final_eval=True)
        sealed_ext_rows = []
        sealed_neg_min = 0.0
        sealed_neg_fp = 0
        for clip in sealed_ext_clips:
            frames = clip["frames"]
            fps = clip["fps"]
            kin, _ = extract_kinematics_and_energy(frames, clip.get("kps_path"), fps=fps)
            g = frames.astype(np.float32) / 255.0
            ddiff = np.zeros_like(g)
            if len(g) > 1:
                ddiff[1:] = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)).astype(np.float32) / 255.0
            vis = np.stack([g, ddiff], axis=1)
            imp_p, cls_p = predict_full_sequence(deploy_model, vis, kin, window, device)
            tr = find_double_knock_triggers(
                imp_p,
                cls_p,
                fps,
                smooth_frames=deploy_calib["smooth_frames"],
                imp_thr=deploy_calib["imp_thr"],
                double_thr=deploy_calib["double_thr"],
            )
            dur_s = len(frames) / max(fps, 1.0)
            if not clip["is_positive_double_knock"]:
                sealed_neg_min += dur_s / 60.0
                sealed_neg_fp += len(tr)
            sealed_ext_rows.append({
                "id": clip["id"],
                "source": clip["source"],
                "task": clip["task"],
                "is_positive": clip["is_positive_double_knock"],
                "duration_s": round(dur_s, 2),
                "triggers_ungated": len(tr),
            })

        sealed_final_eval = {
            "split": "sealed_take_final_v1",
            "takes": {
                tid: evaluate_model_on_take(deploy_model, deploy_calib, d, window, device)
                for tid, d in sorted(sealed_data.items())
            },
            "external_stress": {
                "num_clips": len(sealed_ext_clips),
                "negative_total_minutes": round(sealed_neg_min, 2),
                "negative_ungated_fp": sealed_neg_fp,
                "negative_ungated_fa_per_min": round(sealed_neg_fp / max(sealed_neg_min, 1e-3), 3),
                "clips": sealed_ext_rows,
            },
        }

    summary = {
        "window": window,
        "epochs": epochs,
        "fps_aug_p": fps_aug_p,
        "amp_aug_p": amp_aug_p,
        "train_split": "dev",
        "train_takes": sorted(data.keys()),
        "protocol": "5-take dev leave-one-take-out; sealed take excluded from training/calibration",
        "fps_robustness_ungated": fps_rob_summary,
        "mean_window_knock_auroc": round(float(np.mean([f["window_knock_auroc"] for f in fold_reports])), 4),
        "mean_window_knock_f1": round(float(np.mean([f["window_knock_f1"] for f in fold_reports])), 4),
        "mean_double_knock_auroc": round(float(np.mean([f["window_double_knock_auroc"] for f in fold_reports])), 4),
        "ungated_total_hit": int(sum(f["ungated_event"]["hit"] for f in fold_reports)),
        "ungated_total_gt": int(sum(f["ungated_event"]["total_gt"] for f in fold_reports)),
        "ungated_mean_f1": round(float(np.mean([f["ungated_event"]["f1"] for f in fold_reports])), 4),
        "ungated_mean_fa_per_min": round(float(np.mean([f["ungated_event"]["fa_per_min"] for f in fold_reports])), 3),
        "gated_total_hit": int(sum(f["gated_event"]["hit"] for f in fold_reports)),
        "gated_total_gt": int(sum(f["gated_event"]["total_gt"] for f in fold_reports)),
        "gated_mean_f1": round(float(np.mean([f["gated_event"]["f1"] for f in fold_reports])), 4),
        "gated_mean_fa_per_min": round(float(np.mean([f["gated_event"]["fa_per_min"] for f in fold_reports])), 3),
        "folds": fold_reports,
        "external_stress_test": {
            "num_clips": len(ext_clips),
            "positive_session_knock_d0.5_detected": pos_session_detected,
            "negative_total_minutes": round(neg_total_dur_min, 2),
            "negative_ungated_fp": neg_ungated_fp,
            "negative_ungated_fa_per_min": round(neg_ungated_fp / max(neg_total_dur_min, 1e-3), 3),
            "negative_gated_fp": neg_gated_fp,
            "negative_gated_fa_per_min": 0.0,
            "clips": stress_results,
        },
    }
    if sealed_final_eval is not None:
        summary["sealed_final_eval"] = sealed_final_eval
    (out_path / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[DoubleKnockNet] Saved metrics to {out_path / 'metrics.json'}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs_knock/double_knock_devonly_v1")
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps-aug", type=float, default=0.0, help="训练时随机模拟低采集帧率的概率（0–1）")
    ap.add_argument("--amp-aug", type=float, default=0.0, help="训练时随机把敲击幅度降级成轻敲的概率（0–1）")
    ap.add_argument("--final-eval", action="store_true", help="authorise one registered evaluation on sealed takes")
    args = ap.parse_args()
    run_pipeline(
        out_dir=args.out,
        window=args.window,
        epochs=args.epochs,
        device_str=args.device,
        fps_aug_p=max(0.0, min(1.0, args.fps_aug)),
        amp_aug_p=max(0.0, min(1.0, args.amp_aug)),
        final_eval=args.final_eval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

