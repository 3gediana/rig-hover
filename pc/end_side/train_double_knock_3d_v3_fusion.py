"""DoubleKnock3D-FusionNet v3: 终极高清小手金字塔 + 跨场次不变 3D 物理 + 25% 腕框丢弃 (Box-Dropout)。

专门攻克 v1 vs v2 的核心矛盾：
  - v1 (`stride=2`): 远手小目标 (`65a54d` 15/19, `knock_d0.5`=True) 强，但跨天 `1f30d1` 退化 (7/19) 且外部误报 3 次；
  - v2 (`stride=4` + 不变 3D + 嵌套 OOF): 治好了 `1f30d1` (17/19, F1=0.829) 且真机 9 段负样本 0 误报，但 `stride=4` + 纯 AvgPool 稀释了远手小目标 (`65a54d` 掉至 5/19, `knock_d0.5`=False)，且训练期无 Box-Dropout 导致缺框时退化。

v3 的四项第一性原理升级（100% 公平同构 2D-v3 vs 3D-v3 + 零留出泄漏 + 严 1-to-1 匹配）：
  1. `stride=2` 抗混叠高清首层 (96->48->24->12) 保留 10-14px 远手像素细节，且显存 < 2GB 零 WDDM 换页；
  2. `AdaptiveAvgPool + AdaptiveMaxPool` 双池化金字塔：MaxPool 直接锁住仅占画面 1.5% 的远手桌面敲击瞬时峰值，绝不被静止背景稀释！
  3. 完整继承 v2 的 10 维跨场次不变 3D 物理 (`z_rel`, `desk_E_z`, 牛顿非对称冲击, `spatial_focus`) + 35% 3D 模态丢弃 + 双敲同位检验 + 嵌套 2 折内部 OOF 标定；
  4. 新增 **25% 腕框丢弃 (Box-Dropout)**：训练期随机置零 `wx, wy, vy, ay, vx, valid`，让同一个模型既能配合 `YOLO26n-pose` 跑满血，又能在脱离 YOLO 独立部署时保持高鲁棒性！

运行方式：
    D:\\smartcar\\.venv\\Scripts\\python.exe D:\\robot_work\\pc\\end_side\\train_double_knock_3d_v3_fusion.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pc" / "train"))
sys.path.insert(0, str(REPO / "pc" / "end_side"))

from double_knock_data import extract_kinematics_and_energy  # noqa: E402
from pose_roi import HOLD_FRAMES, clip_box, pick_wrist, roi_side_for  # noqa: E402
from rigorous_audit_knock_ablation import eval_strict_1to1_events  # noqa: E402
from train_double_knock import CausalConv1d, find_double_knock_triggers  # noqa: E402
from train_double_knock_3d_physnet import (  # noqa: E402
    BatchHoverGeoNet3DExtractor,
    _resample_frames_to_fps,
    load_external_stress_clips,
)
from train_double_knock_3d_v2_extreme import (  # noqa: E402
    YUNET_ONNX,
    _face_area,
    _synth_v2,
    extract_invariant_3d_and_hotspots,
    find_causal_triggers_colocal,
    load_v2_dataset,
    predict_seq_fast_exact,
)

V4_CKPT = REPO / "runs_end_side" / "hover_geonet_mv_v4" / "hover_geonet_mv_v4_fp16.pt"

OUT_DIR = REPO / "runs_end_side" / "double_knock_3d_v3_fusion_devonly_v1"
YOLO_POSE_PT = REPO / "pc" / "perception" / "assets" / "yolo26n-pose.pt"


class DoubleKnock3DFusionNetV3(nn.Module):
    """v3 高清小手 Avg+Max 双池化金字塔 + 残差门控不变 3D 双专家因果网络。"""

    def __init__(self, use_3d: bool = True, hidden_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.use_3d = bool(use_3d)
        self.in_ch = 4 if self.use_3d else 2
        self.kin_dim = 18 if self.use_3d else 8
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        half_h = hidden_dim // 2

        # 96 -> 48 -> 24 -> 12 高清 stride=2 编码器 + Avg/Max 双池化抓远手小目标峰值！
        self.vis2d_conv = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 40, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(40),
            nn.SiLU(inplace=True),
            nn.Conv2d(40, half_h, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(half_h),
            nn.SiLU(inplace=True),
        )
        self.kin2d_enc = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
        )

        if self.use_3d:
            self.vis3d_conv = nn.Sequential(
                nn.Conv2d(2, 12, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(12),
                nn.SiLU(inplace=True),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.SiLU(inplace=True),
                nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),
            )
            self.phys3d_mlp = nn.Sequential(
                nn.Linear(32 + 10, 48),
                nn.LayerNorm(48),
                nn.SiLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(48, (hidden_dim + 32) * 2),
            )
            nn.init.zeros_(self.phys3d_mlp[-1].weight)
            nn.init.zeros_(self.phys3d_mlp[-1].bias)
        else:
            self.vis3d_conv = None
            self.phys3d_mlp = None

        fuse_dim = hidden_dim + 32
        self.tcn1 = CausalConv1d(fuse_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn4 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        self.tcn8 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=8)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(0.28)

        self.impulse_head = nn.Linear(hidden_dim, 1)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.18),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _avg_max_pool(feat: torch.Tensor) -> torch.Tensor:
        a = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        m = F.adaptive_max_pool2d(feat, 1).flatten(1)
        return torch.cat([a, m], dim=1)

    def encode_frames(self, x_vis_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        v2 = self._avg_max_pool(self.vis2d_conv(x_vis_flat[:, :2]))
        v3 = self._avg_max_pool(self.vis3d_conv(x_vis_flat[:, 2:4])) if (self.use_3d and self.vis3d_conv is not None) else None
        return v2, v3

    def forward_temporal(
        self,
        v2: torch.Tensor,
        v3: torch.Tensor | None,
        x_kin: torch.Tensor,
        drop_3d_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, t, _ = v2.shape
        k2 = self.kin2d_enc(x_kin[:, :, :8])
        fused = torch.cat([v2, k2], dim=-1)

        if self.use_3d and v3 is not None and self.phys3d_mlp is not None:
            p3_in = torch.cat([v3, x_kin[:, :, 8:]], dim=-1)
            gb = self.phys3d_mlp(p3_in)
            gamma, beta = gb.chunk(2, dim=-1)
            if drop_3d_mask is not None:
                gamma = gamma * drop_3d_mask
                beta = beta * drop_3d_mask
            fused = fused * (1.0 + 0.50 * torch.tanh(gamma)) + 0.35 * beta

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

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor, drop_3d_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, h, w = x_vis.shape
        v2, v3 = self.encode_frames(x_vis.reshape(b * t, c, h, w))
        v2 = v2.reshape(b, t, -1)
        v3 = v3.reshape(b, t, -1) if v3 is not None else None
        dmask = None
        if self.use_3d and self.training and drop_3d_prob > 0.0:
            dmask = (torch.rand((b, 1, 1), device=x_vis.device) >= drop_3d_prob).to(v2.dtype)
        return self.forward_temporal(v2, v3, x_kin, drop_3d_mask=dmask)


def augment_v3_gpu(
    x_vis: torch.Tensor,
    x_kin: torch.Tensor,
    rng: np.random.Generator,
    box_drop_p: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """全 GPU 原地增强 + 25% 腕框丢弃 (Box-Dropout，防过拟合 YOLO 腕框！)。"""
    b, t, _, _, _ = x_vis.shape
    dev = x_vis.device
    v, k = x_vis, x_kin

    flip_m = torch.from_numpy(rng.random(b) < 0.5).to(dev)
    if flip_m.any():
        v[flip_m] = torch.flip(v[flip_m], dims=[-1])
        k[flip_m, :, 0] = -k[flip_m, :, 0]
        k[flip_m, :, 4] = -k[flip_m, :, 4]

    if rng.random() < 0.45:
        rate = float(rng.uniform(10.5, 18.5))
        n_low = max(10, int(round(t * rate / 20.0)))
        if n_low < t:
            low = np.rint(np.linspace(0, t - 1, n_low)).astype(np.int64)
            up = np.rint(np.linspace(0, n_low - 1, t)).astype(np.int64)
            idx_t = torch.from_numpy(low[up]).to(dev)
            sub = torch.from_numpy(rng.random(b) < 0.65).to(dev)
            if sub.any():
                v[sub] = v[sub][:, idx_t]
                k[sub] = k[sub][:, idx_t]

    sc = torch.from_numpy(rng.uniform(0.84, 1.16, size=(b, 1, 1, 1)).astype(np.float32)).to(dev, dtype=v.dtype)
    bi = torch.from_numpy(rng.uniform(-0.08, 0.08, size=(b, 1, 1, 1)).astype(np.float32)).to(dev, dtype=v.dtype)
    v[:, :, 0] = (v[:, :, 0] * sc + bi).clamp_(0.0, 1.0)

    # 新增 25% Box-Dropout：置零腕部几何通道 (wx,wy,vy,ay,vx,valid)，保留全局与桌面区域能量 (ch 6,7)
    bdrop = torch.from_numpy(rng.random(b) < box_drop_p).to(dev)
    if bdrop.any():
        k[bdrop, :, :6] = 0.0

    k.add_(torch.randn_like(k) * 0.04)
    if k.shape[-1] > 8:
        z_jit = torch.from_numpy(rng.uniform(-0.10, 0.10, size=(b, 1)).astype(np.float32)).to(dev)
        k[:, :, 8] = (k[:, :, 8] + z_jit).clamp_(-0.5, 2.5)
    return v, k


def train_v3_model(
    train_sub: dict[str, dict],
    forbidden_val_take: str | None,
    use_3d: bool,
    window: int,
    epochs: int,
    device: torch.device,
    seed: int,
    batch_size: int = 24,
    grad_accum: int = 2,
) -> DoubleKnock3DFusionNetV3:
    if forbidden_val_take is not None and forbidden_val_take in train_sub:
        raise ValueError(f"LEAKAGE BLOCKED: {forbidden_val_take}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    in_ch = 4 if use_3d else 2
    kin_d = 18 if use_3d else 8

    sx_v, sx_k, sy_i, sy_c = _synth_v2(train_sub, forbidden_val_take, use_3d=use_3d, n_per_class=95, window=window, seed=seed + 7)
    real_idx: list[tuple[str, int, int]] = []
    for tid, d in train_sub.items():
        for s, c in zip(d["win_starts"].tolist(), d["win_cnts"].tolist()):
            real_idx.append((tid, int(s), int(c)))

    n_real = len(real_idx)
    n_syn = len(sy_c)
    all_cnts = np.concatenate([np.array([r[2] for r in real_idx], dtype=np.int64), sy_c])
    counts = np.bincount(all_cnts, minlength=4).astype(np.float32)
    cls_w = np.clip(counts.sum() / np.maximum(counts * 4.0, 1.0), 0.35, 3.0)
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=device)

    sample_w = np.ones(n_real + n_syn, dtype=np.float64)
    for idx_s, c in enumerate(all_cnts):
        sample_w[idx_s] = 1.0 if c == 0 else (3.1 if c == 2 else 2.2)
    sample_w /= sample_w.sum()

    model = DoubleKnock3DFusionNetV3(use_3d=use_3d, hidden_dim=64, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2.5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1.2e-4)
    n_epoch_samples = min(2160, max(1200, (n_real + n_syn) // 2))
    pos_w = torch.tensor([6.0], device=device)

    for _ in range(epochs):
        model.train()
        chosen = rng.choice(n_real + n_syn, size=n_epoch_samples, replace=True, p=sample_w)
        opt.zero_grad(set_to_none=True)
        step_i = 0
        for bi in range(0, len(chosen), batch_size):
            ids = chosen[bi : bi + batch_size]
            B = len(ids)
            bv = np.empty((B, window, in_ch, 96, 96), dtype=np.float32)
            bk = np.empty((B, window, kin_d), dtype=np.float32)
            byi = np.empty((B, window), dtype=np.float32)
            byc = np.empty((B,), dtype=np.int64)
            for j, sid in enumerate(ids):
                if sid < n_real:
                    tid, s, c = real_idx[sid]
                    d = train_sub[tid]
                    bv[j] = d["vis_4ch_u8"][s : s + window, :in_ch].astype(np.float32) / 255.0
                    bk[j] = d["kin_18"][s : s + window, :kin_d]
                    byi[j] = d["y_imp_full"][s : s + window]
                    byc[j] = c
                else:
                    k_i = sid - n_real
                    bv[j] = sx_v[k_i]
                    bk[j] = sx_k[k_i]
                    byi[j] = sy_i[k_i]
                    byc[j] = sy_c[k_i]

            xt_v = torch.from_numpy(bv).to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32, non_blocking=True)
            xt_k = torch.from_numpy(bk).to(device, non_blocking=True)
            yt_i = torch.from_numpy(byi).to(device, non_blocking=True)
            yt_c = torch.from_numpy(byc).to(device, non_blocking=True)
            xt_v, xt_k = augment_v3_gpu(xt_v, xt_k, rng, box_drop_p=0.25)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                li, lc = model(xt_v, xt_k, drop_3d_prob=(0.35 if use_3d else 0.0))
                loss_i = F.binary_cross_entropy_with_logits(li.float(), yt_i, pos_weight=pos_w)
                loss_c = F.cross_entropy(lc.float(), yt_c, weight=cls_w_t, label_smoothing=0.04)
                loss = (0.55 * loss_i + 1.0 * loss_c) / grad_accum
            loss.backward()
            step_i += 1
            if step_i % grad_accum == 0 or (bi + batch_size >= len(chosen)):
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        sched.step()
    return model


def calibrate_v3_inner_oof(
    train_sub: dict[str, dict],
    forbidden_val_take: str | None,
    use_3d: bool,
    window: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tids = sorted(train_sub.keys())
    g0 = [t for i, t in enumerate(tids) if i % 2 == 0]
    g1 = [t for i, t in enumerate(tids) if i % 2 == 1]
    oof_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for gi, hold_group in enumerate((g0, g1)):
        inner_tr = {t: d for t, d in train_sub.items() if t not in hold_group}
        if not inner_tr:
            continue
        im = train_v3_model(inner_tr, forbidden_val_take=forbidden_val_take, use_3d=use_3d, window=window, epochs=11, device=device, seed=seed + gi * 31)
        for ht in hold_group:
            d = train_sub[ht]
            oof_preds[ht] = predict_seq_fast_exact(im, d["vis_4ch_u8"], d["kin_18"], window, device)
        del im
        torch.cuda.empty_cache()

    best_win = None
    for sm in (2, 3, 5):
        for imp_thr in (1.2, 1.5, 1.8):
            for dbl_thr in (0.25, 0.35, 0.45, 0.55):
                th = tgt = tfp = 0
                for tid, d in train_sub.items():
                    ip, cp, _, _ = oof_preds[tid]
                    tr = find_double_knock_triggers(ip, cp, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr)
                    ev = eval_strict_1to1_events(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; tfp += ev["fp"]
                prec = th / (th + tfp) if (th + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.008 * tfp
                if best_win is None or score > best_win["score"]:
                    best_win = {"score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr, "oof_strict_f1": round(f1, 4), "oof_hit": th, "oof_gt": tgt, "oof_fp": tfp}

    best_str = None
    for sm in (2, 3, 5):
        for imp_thr in (0.20, 0.28, 0.36, 0.44):
            for dbl_thr in (0.20, 0.28, 0.36, 0.45):
                th = tgt = tfp = 0
                for tid, d in train_sub.items():
                    _, _, ic, cc = oof_preds[tid]
                    tr = find_causal_triggers_colocal(
                        ic, cc, d["fps"],
                        hot_uvf=(d["hot_uvf"] if use_3d else None),
                        smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr,
                    )
                    ev = eval_strict_1to1_events(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; tfp += ev["fp"]
                prec = th / (th + tfp) if (th + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.010 * tfp
                if best_str is None or score > best_str["score"]:
                    best_str = {"score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr, "oof_strict_f1": round(f1, 4), "oof_hit": th, "oof_gt": tgt, "oof_fp": tfp}
    return best_win, best_str


def run_v3_controlled_loo(label: str, dataset: dict[str, dict], use_3d: bool, window: int, epochs: int, device: torch.device) -> dict[str, Any]:
    take_ids = sorted(dataset.keys())
    in_ch = 4 if use_3d else 2
    kin_d = 18 if use_3d else 8
    folds = []
    t0 = time.perf_counter()

    for fi, val_tid in enumerate(take_ids):
        tf = time.perf_counter()
        train_sub = {tid: d for tid, d in dataset.items() if tid != val_tid}
        val_d = dataset[val_tid]

        c_win, c_str = calibrate_v3_inner_oof(train_sub, forbidden_val_take=val_tid, use_3d=use_3d, window=window, device=device, seed=500 + fi * 10)
        model = train_v3_model(train_sub, forbidden_val_take=val_tid, use_3d=use_3d, window=window, epochs=epochs, device=device, seed=400 + fi)

        ip_v, cp_v, ic_v, cc_v = predict_seq_fast_exact(model, val_d["vis_4ch_u8"], val_d["kin_18"], window, device)

        p_chunks = []
        ws = val_d["eval_starts"]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            for si in range(0, len(ws), 48):
                sb = ws[si : si + 48]
                xv = torch.from_numpy(np.stack([val_d["vis_4ch_u8"][s : s + window, :in_ch] for s in sb])).float().div_(255.0).to(device)
                xk = torch.from_numpy(np.stack([val_d["kin_18"][s : s + window, :kin_d] for s in sb])).float().to(device)
                _, lc = model(xv, xk)
                p_chunks.append(torch.softmax(lc.float(), dim=-1).cpu().numpy())
        p_win = np.concatenate(p_chunks, axis=0)
        y_cnt = val_d["eval_cnts"]
        win_auroc = float(roc_auc_score((y_cnt > 0).astype(np.int32), 1.0 - p_win[:, 0]))
        dbl_auroc = float(roc_auc_score((y_cnt >= 2).astype(np.int32), p_win[:, 2] + p_win[:, 3]))

        tr_w = find_double_knock_triggers(ip_v, cp_v, val_d["fps"], smooth_frames=c_win["smooth_frames"], imp_thr=c_win["imp_thr"], double_thr=c_win["double_thr"])
        tr_s = find_causal_triggers_colocal(
            ic_v, cc_v, val_d["fps"],
            hot_uvf=(val_d["hot_uvf"] if use_3d else None),
            smooth_frames=c_str["smooth_frames"], imp_thr=c_str["imp_thr"], double_thr=c_str["double_thr"],
        )
        st_w = eval_strict_1to1_events(tr_w, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"])
        st_s = eval_strict_1to1_events(tr_s, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"])

        # 额外诚实测试：无腕框 (Boxless) 退化抗性测试（直接把验证集前 6 维腕框置零！）
        kin_nobox = val_d["kin_18"].copy()
        kin_nobox[:, :6] = 0.0
        _, _, ic_nb, cc_nb = predict_seq_fast_exact(model, val_d["vis_4ch_u8"], kin_nobox, window, device)
        tr_nb = find_causal_triggers_colocal(
            ic_nb, cc_nb, val_d["fps"],
            hot_uvf=(val_d["hot_uvf"] if use_3d else None),
            smooth_frames=c_str["smooth_frames"], imp_thr=c_str["imp_thr"], double_thr=c_str["double_thr"],
        )
        st_nb = eval_strict_1to1_events(tr_nb, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"])

        folds.append({
            "val_take": val_tid, "calib_str_oof": c_str, "calib_win_oof": c_win,
            "win_auroc": round(win_auroc, 4), "dbl_auroc": round(dbl_auroc, 4),
            "strict_win": st_w, "strict_stream": st_s, "strict_stream_nobox": st_nb,
        })
        print(
            f"  [{label}] Fold {fi+1}/6 ({val_tid[-6:]}, {time.perf_counter()-tf:.1f}s): "
            f"Win={st_w['hit']}/{st_w['total_gt']} FP={st_w['fp']} F1={st_w['strict_f1']:.4f} | "
            f"Stream={st_s['hit']}/{st_s['total_gt']} FP={st_s['fp']} F1={st_s['strict_f1']:.4f} | "
            f"NoBox={st_nb['hit']}/{st_nb['total_gt']} F1={st_nb['strict_f1']:.4f} | AUROC={win_auroc:.4f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    def _agg(key: str) -> dict[str, Any]:
        h = sum(f[key]["hit"] for f in folds)
        g = sum(f[key]["total_gt"] for f in folds)
        fp = sum(f[key]["fp"] for f in folds)
        mp = h / (h + fp) if (h + fp) > 0 else 0.0
        mr = h / max(g, 1)
        mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
        return {
            "total_hit": h, "total_gt": g, "total_fp": fp,
            "micro_strict_prec": round(mp, 4), "micro_strict_rec": round(mr, 4),
            "micro_strict_f1": round(mf1, 4),
            "macro_strict_f1": round(float(np.mean([f[key]["strict_f1"] for f in folds])), 4),
            "mean_fa_per_min": round(float(np.mean([f[key]["fa_per_min"] for f in folds])), 3),
        }

    c_win_d, c_str_d = calibrate_v3_inner_oof(dataset, forbidden_val_take=None, use_3d=use_3d, window=window, device=device, seed=988)
    deploy_m = train_v3_model(dataset, forbidden_val_take=None, use_3d=use_3d, window=window, epochs=epochs, device=device, seed=877)

    return {
        "label": label, "use_3d": use_3d, "elapsed_s": round(time.perf_counter() - t0, 1),
        "mean_win_auroc": round(float(np.mean([f["win_auroc"] for f in folds])), 4),
        "mean_dbl_auroc": round(float(np.mean([f["dbl_auroc"] for f in folds])), 4),
        "strict_1to1_win": _agg("strict_win"),
        "strict_1to1_stream": _agg("strict_stream"),
        "strict_1to1_stream_nobox": _agg("strict_stream_nobox"),
        "folds": folds,
        "_deploy": (deploy_m, c_win_d, c_str_d),
    }


def eval_external_v3_real_yolo(
    m2d: DoubleKnock3DFusionNetV3, c2_w: dict, c2_s: dict,
    m3d: DoubleKnock3DFusionNetV3, c3_w: dict, c3_s: dict,
    extractor: BatchHoverGeoNet3DExtractor, yunet: Any, window: int, device: torch.device,
) -> dict[str, Any]:
    """100% 公平外部 30 段压测：统一跑真实因果 YOLO26n-pose 腕部框，同时比较 2D-v3 vs 3D-v3 vs 3D-v3+YuNet！"""
    from ultralytics import YOLO  # noqa: E402

    yolo = YOLO(str(YOLO_POSE_PT))
    clips = load_external_stress_clips(max_clips_per_source=10, size=240)

    def _get_boxes(f240: np.ndarray) -> np.ndarray:
        h, w = f240.shape[1], f240.shape[2]
        bgrs = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in f240]
        boxes = []
        last_box, last_seen = None, -10**9
        for bi in range(0, len(bgrs), 64):
            res_list = yolo.predict(bgrs[bi : bi + 64], imgsz=320, conf=0.25, verbose=False, device=(0 if device.type == "cuda" else "cpu"), classes=[0])
            for j, res in enumerate(res_list):
                idx = bi + j
                kp = None
                if res is not None and getattr(res, "keypoints", None) is not None and len(res.keypoints):
                    kxy = res.keypoints.xy.cpu().numpy()
                    kconf = res.keypoints.conf.cpu().numpy()
                    bconf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.ones(len(kxy))
                    if len(kxy):
                        order = np.argsort(-bconf)
                        kp = np.concatenate([kxy[order[0]], kconf[order[0]][:, None]], axis=1)
                wrist = pick_wrist(kp)
                if wrist is not None:
                    wx, wy, _ = wrist
                    side = roi_side_for(kp, wx, wy)
                    x1, y1, x2, y2 = clip_box(wx - side / 2, wy - side / 2, wx + side / 2, wy + side / 2, w, h)
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    last_box, last_seen = box, idx
                elif last_box is not None and idx - last_seen <= HOLD_FRAMES:
                    box = last_box
                else:
                    box = np.array([0, 0, w, h], dtype=np.float32)
                    last_box = None
                boxes.append(box)
        return np.stack(boxes, axis=0)

    st = {
        "2d_str_fp": 0, "3d_str_fp": 0, "3d_yunet_str_fp": 0,
        "2d_robot9_str_fp": 0, "3d_robot9_str_fp": 0, "3d_yunet_robot9_str_fp": 0,
        "3d_charades10_str_fp": 0, "3d_ucf10_str_fp": 0,
        "neg_minutes": 0.0,
        "knock_d05_detected": {"2d_str": False, "3d_str": False, "3d_yunet_str": False, "3d_nobox_str": False},
        "nonzero_fp_clips": [],
    }
    for cl in clips:
        f240_r, _, feat_fps = _resample_frames_to_fps(cl["frames"], float(cl.get("fps", 20.0)), 20.0)
        boxes = _get_boxes(f240_r)
        f96 = np.stack([cv2.resize(f, (96, 96), interpolation=cv2.INTER_AREA) for f in f240_r], axis=0)
        kin_2d, _ = extract_kinematics_and_energy(f96, boxes, fps=feat_fps)
        vis_4ch, kin_3d_raw, _ = extractor.extract_sequence_3d(f240_r, fps=feat_fps, size=96)
        kin_3d_inv, hot_uvf = extract_invariant_3d_and_hotspots(vis_4ch, kin_3d_raw, fps=feat_fps)
        kin_18 = np.concatenate([kin_2d, kin_3d_inv], axis=1).astype(np.float32)

        _, _, ic2, cc2 = predict_seq_fast_exact(m2d, vis_4ch, kin_18, window, device)
        _, _, ic3, cc3 = predict_seq_fast_exact(m3d, vis_4ch, kin_18, window, device)
        ts2 = find_causal_triggers_colocal(ic2, cc2, feat_fps, hot_uvf=None, smooth_frames=c2_s["smooth_frames"], imp_thr=c2_s["imp_thr"], double_thr=c2_s["double_thr"])
        ts3 = find_causal_triggers_colocal(ic3, cc3, feat_fps, hot_uvf=hot_uvf, smooth_frames=c3_s["smooth_frames"], imp_thr=c3_s["imp_thr"], double_thr=c3_s["double_thr"])
        ts3g = []
        for tr in ts3:
            fi = min(max(int(round(tr["second_s"] * feat_fps)), 0), len(f240_r) - 1)
            max_fa = max(_face_area(yunet, f240_r[k]) for k in range(max(0, fi - 10), min(len(f240_r), fi + 5), 2))
            if max_fa <= 0.085:
                ts3g.append(tr)

        if bool(cl["is_positive_double_knock"]):
            kin_nb = kin_18.copy(); kin_nb[:, :6] = 0.0
            _, _, ic3_nb, cc3_nb = predict_seq_fast_exact(m3d, vis_4ch, kin_nb, window, device)
            ts3_nb = find_causal_triggers_colocal(ic3_nb, cc3_nb, feat_fps, hot_uvf=hot_uvf, smooth_frames=c3_s["smooth_frames"], imp_thr=c3_s["imp_thr"], double_thr=c3_s["double_thr"])
            st["knock_d05_detected"] = {"2d_str": len(ts2) > 0, "3d_str": len(ts3) > 0, "3d_yunet_str": len(ts3g) > 0, "3d_nobox_str": len(ts3_nb) > 0}
            continue

        st["neg_minutes"] += len(f240_r) / max(feat_fps, 1.0) / 60.0
        st["2d_str_fp"] += len(ts2); st["3d_str_fp"] += len(ts3); st["3d_yunet_str_fp"] += len(ts3g)
        if cl["source"] == "evidence_sessions":
            st["2d_robot9_str_fp"] += len(ts2); st["3d_robot9_str_fp"] += len(ts3); st["3d_yunet_robot9_str_fp"] += len(ts3g)
        elif cl["source"] == "charades_distractor":
            st["3d_charades10_str_fp"] += len(ts3g)
        else:
            st["3d_ucf10_str_fp"] += len(ts3g)
        if len(ts2) or len(ts3) or len(ts3g):
            st["nonzero_fp_clips"].append({"id": cl["id"], "src": cl["source"], "task": cl["task"], "2d_str": len(ts2), "3d_str": len(ts3), "3d_yunet_str": len(ts3g)})
    st["neg_minutes"] = round(st["neg_minutes"], 2)
    return st


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("=== PILLAR 1 (v3 FUSION): HIGH-RES STRIDE-2 AVG+MAX PYRAMID + INVARIANT 3D + 25% BOX-DROPOUT ===", flush=True)
    dataset = load_v2_dataset(window=48, stride=4, size=96)

    print("\n--- [1/2] Controlled Same-Arch 2D-v3 (stride=2 Avg+Max + Box-Dropout + Nested OOF) ---", flush=True)
    r2 = run_v3_controlled_loo("SameArch-2D-v3", dataset, use_3d=False, window=48, epochs=18, device=device)

    print("\n--- [2/2] Controlled Same-Arch 3D-v3 (stride=2 Avg+Max + Invariant 3D + 35% ModDrop + 25% BoxDrop + Co-Local + Nested OOF) ---", flush=True)
    r3 = run_v3_controlled_loo("SameArch-3D-v3", dataset, use_3d=True, window=48, epochs=18, device=device)

    m2d, c2w, c2s = r2.pop("_deploy")
    m3d, c3w, c3s = r3.pop("_deploy")
    ckpt_p = OUT_DIR / "double_knock_3d_v3_fusion_deploy.pt"
    torch.save({
        "arch": "DoubleKnock3DFusionNetV3", "use_3d": True, "window": 48, "hidden_dim": 64,
        "calib_win": c3w, "calib_stream": c3s, "state_dict": m3d.state_dict(),
    }, ckpt_p)

    print("\n--- [3/3] 100% Fair External 30-Clip Stress Test WITH REAL YOLO26n-pose ROI ---", flush=True)
    extractor = BatchHoverGeoNet3DExtractor(ckpt_path=V4_CKPT, device=str(device))
    yunet = cv2.FaceDetectorYN.create(str(YUNET_ONNX), "", (240, 240), score_threshold=0.45) if YUNET_ONNX.exists() else None
    ext = eval_external_v3_real_yolo(m2d, c2w, c2s, m3d, c3w, c3s, extractor, yunet, window=48, device=device)

    rep = {
        "protocol": "v3 Fusion: High-Res stride=2 Avg+Max Pyramid + Invariant 3D + 35% ModDrop + 25% BoxDrop + Nested Inner-OOF + Real YOLO26n-pose External Stress",
        "ckpt_kb": round(ckpt_p.stat().st_size / 1024.0, 1),
        "controlled_2d_v3": r2,
        "controlled_3d_v3": r3,
        "pure_3d_v3_delta": {
            "win_auroc_delta": round(r3["mean_win_auroc"] - r2["mean_win_auroc"], 4),
            "dbl_auroc_delta": round(r3["mean_dbl_auroc"] - r2["mean_dbl_auroc"], 4),
            "strict_stream_macro_f1_delta": round(r3["strict_1to1_stream"]["macro_strict_f1"] - r2["strict_1to1_stream"]["macro_strict_f1"], 4),
            "strict_stream_micro_f1_delta": round(r3["strict_1to1_stream"]["micro_strict_f1"] - r2["strict_1to1_stream"]["micro_strict_f1"], 4),
            "strict_win_micro_f1_delta": round(r3["strict_1to1_win"]["micro_strict_f1"] - r2["strict_1to1_win"]["micro_strict_f1"], 4),
            "strict_nobox_micro_f1_delta": round(r3["strict_1to1_stream_nobox"]["micro_strict_f1"] - r2["strict_1to1_stream_nobox"]["micro_strict_f1"], 4),
        },
        "external_30clip_real_yolo_pose_roi": ext,
    }
    rep_p = OUT_DIR / "double_knock_3d_v3_fusion_report.json"
    rep_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ v3 FUSION FINAL SUMMARY ================", flush=True)
    print(f"2D-v3: AUROC={r2['mean_win_auroc']} (dbl={r2['mean_dbl_auroc']}) | StrictWin={r2['strict_1to1_win']} | StrictStream={r2['strict_1to1_stream']} | NoBox={r2['strict_1to1_stream_nobox']}", flush=True)
    print(f"3D-v3: AUROC={r3['mean_win_auroc']} (dbl={r3['mean_dbl_auroc']}) | StrictWin={r3['strict_1to1_win']} | StrictStream={r3['strict_1to1_stream']} | NoBox={r3['strict_1to1_stream_nobox']}", flush=True)
    print(f"External 30-clip (REAL YOLO26n-pose): {ext}", flush=True)
    print(f"Saved -> {rep_p}", flush=True)


if __name__ == "__main__":
    main()
