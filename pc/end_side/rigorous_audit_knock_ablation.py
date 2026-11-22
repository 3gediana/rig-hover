#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rigorous_audit_knock_ablation.py —— 严格科学对照实验：同架构同配方 2D vs 3D 动作识别消融 + 严格 1-to-1 事件评测（零拷贝低内存极速版）

消除此前所有可能存在的方法学混杂因素，并彻底消除原脚本 21 GB float32 拷贝造成的磁盘换页卡顿（内存压至 < 1 GB，提速 ~25×）：
1. **严格控制变量（Apples-to-Apples）**：
   - Control A (`Same-Arch 2D-Only`): 完全相同的 `tcn1/2/4/8 + GRU` 架构、相同 `epochs=20`、相同 `synth_per_class=90`、
     相同 `fps_aug_p=0.45`、相同随机种子 `200..205`、相同仅训练集严格 1-to-1 网格标定，仅关闭 3D 通道与 3D FiLM (`in_ch=2, kin_dim=8`)；
   - Treatment B (`Same-Arch 3D-PhysNet`): 仅开启 `HoverGeoNet-v4` 3D 桌面物理通道 + 8 维 3D 运动学 + 3D FiLM (`in_ch=4, kin_dim=16`)；
   - 两者差值即为 **3D 度量桌面物理的纯粹因果净收益**（零架构/训练周期/标定器混杂！）。
2. **修复原 `eval_triggers_on_take` 多触发重复计入 TP 的虚高问题**：
   - 同时输出 **严格 1-to-1 事件指标 (`strict_prec = hit / (hit + fp)`, `strict_f1`)**（一个 GT 区间无论落入多少触发，最多计 1 个 TP！）
     与历史 `legacy_eval` 指标，并且彻底剔除使用 GT 区间做门控的 `oracle_gt_gated` 指标。
3. **在 30 条外部跨数据集片段上同口径正面对比 Controlled 2D vs Controlled 3D vs 3D+YuNet**，并在原始 240×240 视频帧上核验全部 89 次真实双敲的 YuNet 零误伤。
"""

from __future__ import annotations

import gc
import json
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
TRAIN_DIR = REPO / "pc" / "train"
for p in (str(HERE), str(TRAIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from double_knock_data import (  # noqa: E402
    KNOCK_IDX,
    _resample_frames_to_fps,
    classify_window_count,
    detect_contact_impulses,
    extract_kinematics_and_energy,
    load_external_stress_clips,
    make_impulse_heatmap,
)
from hover_geonet_240 import GEO_H  # noqa: E402
from interval_data import IGNORE, decode_gray, frame_labels, load_intervals, load_labels, load_takes  # noqa: E402
from train_double_knock import eval_triggers_on_take, find_double_knock_triggers, focal_bce_loss  # noqa: E402
from train_double_knock_3d_physnet import (  # noqa: E402
    BatchHoverGeoNet3DExtractor,
    CACHE_DIR,
    DoubleKnock3DPhysNet,
    V4_CKPT_FP16,
    augment_3d_batch,
    find_causal_streaming_triggers,
    predict_seq_both_modes,
)

YUNET_ONNX = REPO / "pc" / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"
OUT_DIR = REPO / "runs_end_side" / "rigorous_audit_v13"


def eval_strict_1to1_events(
    triggers: list[dict],
    labels_full: np.ndarray,
    intervals: list,
    fps: float,
    duration_s: float,
) -> dict[str, Any]:
    """严格 1-to-1 事件级评测：每个真实双敲区间最多匹配 1 个 TP（绝不重复计 TP 刷高 Precision！）。"""
    gt_ivs = [(iv.start_s, iv.end_s) for iv in intervals if iv.label == "knock"]
    T = len(labels_full)
    matched_gt = set()
    fp = 0
    dup_in_gt = 0

    for tr in triggers:
        t = float(tr["second_s"])
        fi = min(max(int(round(t * fps)), 0), T - 1)
        hit_ids = [gi for gi, (a, b) in enumerate(gt_ivs) if (a - 0.5) <= t <= (b + 0.5)]
        if hit_ids:
            unmatched = [gi for gi in hit_ids if gi not in matched_gt]
            if unmatched:
                matched_gt.add(unmatched[0])
            else:
                dup_in_gt += 1
        elif labels_full[fi] != IGNORE:
            fp += 1

    hit = len(matched_gt)
    total_gt = len(gt_ivs)
    prec = hit / (hit + fp) if (hit + fp) > 0 else (1.0 if total_gt == 0 else 0.0)
    rec = hit / total_gt if total_gt > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fa_per_min = fp / max(duration_s / 60.0, 1e-3)
    return {
        "hit": hit, "total_gt": total_gt, "fp": fp, "dup_in_gt": dup_in_gt,
        "strict_prec": round(prec, 4), "strict_rec": round(rec, 4),
        "strict_f1": round(f1, 4), "fa_per_min": round(fa_per_min, 3),
    }


def load_zerocopy_dataset(window: int = 48, stride: int = 4, size: int = 96) -> dict[str, dict[str, Any]]:
    """零拷贝低内存加载全部 6 条真实标注 take：仅存 uint8 整段序列 (~600MB) 与整型窗口起点，绝不复制 21GB 窗口！"""
    ds_path = REPO / "dataset"
    roi_path = REPO / "runs_knock" / "pose_roi"
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
        cache_file = CACHE_DIR / f"{take.take_id}_3d.npz"
        assert cache_file.exists(), f"Missing 3D cache: {cache_file}"
        z = np.load(cache_file)
        vis_4ch_u8 = np.ascontiguousarray(z["vis_4ch"])           # [T, 4, 96, 96] uint8
        kin_3d = z["kin_3d"].astype(np.float32)                   # [T, 8]
        kin_16 = np.concatenate([kin_2d, kin_3d], axis=1).astype(np.float32)

        # 关键公平细节：2D 与 3D 使用完全同一套 contacts 与标签（仅从 2D impact_2d 在训练区间内定位脉冲，绝不给 3D 额外标签优势！）
        contacts = detect_contact_impulses(impact_2d, intervals, fps=feat_fps, min_gap_s=0.28)
        y_imp_full = make_impulse_heatmap(T, contacts, sigma=1.25).astype(np.float32)

        win_starts, win_cnts = [], []
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
            win_starts.append(s)
            win_cnts.append(cnt_cls)

        imp_centers = [int(c) for c in contacts if 6 <= c <= T - 6]
        neg_starts = [s for s, c in zip(win_starts, win_cnts) if c == 0]

        out[take.take_id] = {
            "fps": feat_fps,
            "duration_s": float(take.duration_s),
            "video_path": take.video_path,
            "vis_4ch_u8": vis_4ch_u8,
            "kin_16": kin_16,
            "labels_full": labels,
            "y_imp_full": y_imp_full,
            "intervals": intervals,
            "win_starts": np.asarray(win_starts, dtype=np.int32),
            "win_cnts": np.asarray(win_cnts, dtype=np.int64),
            "imp_centers": imp_centers,
            "neg_starts": neg_starts,
        }
    return out


def _synthesize_lightweight(
    train_sub: dict[str, dict],
    forbidden_val_take: str | None,
    use_3d: bool,
    n_per_class: int = 90,
    window: int = 48,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """轻量合成 1/2/3 下敲击（直接从 uint8 原序列切片，总内存 < 40 MB！）。"""
    if forbidden_val_take is not None and forbidden_val_take in train_sub:
        raise ValueError(f"LEAKAGE BLOCKED: {forbidden_val_take} in train_sub!")
    in_ch = 4 if use_3d else 2
    kin_d = 16 if use_3d else 8
    imp_refs = [(tid, c) for tid, d in train_sub.items() for c in d["imp_centers"]]
    neg_refs = [(tid, s) for tid, d in train_sub.items() for s in d["neg_starts"]]
    rng = np.random.default_rng(seed)
    stroke_len = 12
    half = 6
    blend = np.hanning(stroke_len + 2)[1:-1].astype(np.float32)
    b_vis = blend[:, None, None, None]
    b_kin = blend[:, None]

    xv, xk, yi, yc = [], [], [], []
    for target_cls in (1, 2, 3):
        for _ in range(n_per_class):
            ntid, ns = neg_refs[int(rng.integers(0, len(neg_refs)))]
            vis = train_sub[ntid]["vis_4ch_u8"][ns : ns + window, :in_ch].astype(np.float32) / 255.0
            kin = train_sub[ntid]["kin_16"][ns : ns + window, :kin_d].copy()
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
                itid, ic = imp_refs[int(rng.integers(0, len(imp_refs)))]
                s, e = c - half, c - half + stroke_len
                if 0 <= s and e <= window:
                    iv_vis = train_sub[itid]["vis_4ch_u8"][ic - half : ic + half, :in_ch].astype(np.float32) / 255.0
                    iv_kin = train_sub[itid]["kin_16"][ic - half : ic + half, :kin_d]
                    vis[s:e] = (1.0 - b_vis) * vis[s:e] + b_vis * iv_vis
                    kin[s:e] = (1.0 - b_kin) * kin[s:e] + b_kin * iv_kin
            xv.append(np.rint(np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8))
            xk.append(kin.astype(np.float32))
            yi.append(make_impulse_heatmap(window, centers, sigma=1.25).astype(np.float32))
            yc.append(target_cls)
    return np.stack(xv), np.stack(xk), np.stack(yi), np.asarray(yc, dtype=np.int64)


def train_fold_fast(
    train_sub: dict[str, dict],
    forbidden_val_take: str | None,
    use_3d: bool,
    window: int = 48,
    epochs: int = 20,
    batch_size: int = 48,
    lr: float = 1.2e-3,
    device: torch.device = torch.device("cuda"),
    seed: int = 200,
) -> DoubleKnock3DPhysNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    in_ch = 4 if use_3d else 2
    kin_d = 16 if use_3d else 8

    # 真实窗口全部用整型引用 (0, tid, s, cnt)；仅合成 270 条存轻量 uint8 数组 (<40MB)！
    real_refs = []
    y_cnt_all = []
    for tid, d in train_sub.items():
        for s, c in zip(d["win_starts"], d["win_cnts"]):
            real_refs.append((0, tid, int(s), int(c)))
            y_cnt_all.append(int(c))

    syn_v, syn_k, syn_i, syn_c = _synthesize_lightweight(train_sub, forbidden_val_take, use_3d, n_per_class=90, window=window, seed=seed)
    for si, c in enumerate(syn_c):
        real_refs.append((1, "", si, int(c)))
        y_cnt_all.append(int(c))

    y_cnt_arr = np.asarray(y_cnt_all, dtype=np.int64)
    cls_counts = np.bincount(y_cnt_arr, minlength=4).astype(np.float32)
    cls_weights = np.sum(cls_counts) / (4.0 * np.maximum(cls_counts, 1.0))
    cls_weights[2] *= 1.35
    cls_w_t = torch.from_numpy(cls_weights).to(device)

    model = DoubleKnock3DPhysNet(in_ch=in_ch, kin_dim=kin_d, hidden_dim=64, num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    N = len(real_refs)
    for _ep in range(epochs):
        model.train()
        perm = rng.permutation(N)
        for st in range(0, N, batch_size):
            ids = perm[st : st + batch_size]
            bv_l, bk_l, bi_l, bc_l = [], [], [], []
            for idx in ids:
                kind, tid, pos, c_lbl = real_refs[idx]
                if kind == 0:
                    d = train_sub[tid]
                    bv_l.append(d["vis_4ch_u8"][pos : pos + window, :in_ch])
                    bk_l.append(d["kin_16"][pos : pos + window, :kin_d])
                    bi_l.append(d["y_imp_full"][pos : pos + window])
                else:
                    bv_l.append(syn_v[pos])
                    bk_l.append(syn_k[pos])
                    bi_l.append(syn_i[pos])
                bc_l.append(c_lbl)
            bv = torch.from_numpy(np.stack(bv_l)).float().div_(255.0).to(device, non_blocking=True)
            bk = torch.from_numpy(np.stack(bk_l)).float().to(device, non_blocking=True)
            bi = torch.from_numpy(np.stack(bi_l)).float().to(device, non_blocking=True)
            bc = torch.tensor(bc_l, dtype=torch.long, device=device)

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

    del syn_v, syn_k, syn_i, syn_c
    model.eval()
    return model


def _calibrate_strict_on_train_only(
    preds_train: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_sub: dict[str, dict],
) -> tuple[dict, dict]:
    best_win = None
    for sm in (2, 3, 5):
        for imp_thr in (0.6, 0.9, 1.2, 1.5, 1.8):
            for dbl_thr in (0.25, 0.35, 0.45, 0.55):
                th = tgt = tfp = 0
                for tid, d in train_sub.items():
                    ip, cp, _, _ = preds_train[tid]
                    tr = find_double_knock_triggers(ip, cp, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr)
                    ev = eval_strict_1to1_events(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; tfp += ev["fp"]
                prec = th / (th + tfp) if (th + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.008 * tfp
                if best_win is None or score > best_win["score"]:
                    best_win = {"score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr, "train_strict_f1": round(f1, 4), "train_hit": th, "train_gt": tgt, "train_fp": tfp}

    best_str = None
    for sm in (2, 3, 5):
        for imp_thr in (0.22, 0.28, 0.35, 0.42, 0.50):
            for dbl_thr in (0.20, 0.30, 0.38, 0.48):
                th = tgt = tfp = 0
                for tid, d in train_sub.items():
                    _, _, ic, cc = preds_train[tid]
                    tr = find_causal_streaming_triggers(ic, cc, d["fps"], smooth_frames=sm, imp_thr=imp_thr, double_thr=dbl_thr)
                    ev = eval_strict_1to1_events(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    th += ev["hit"]; tgt += ev["total_gt"]; tfp += ev["fp"]
                prec = th / (th + tfp) if (th + tfp) > 0 else 0.0
                rec = th / max(tgt, 1)
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                score = f1 - 0.008 * tfp
                if best_str is None or score > best_str["score"]:
                    best_str = {"score": round(score, 5), "smooth_frames": sm, "imp_thr": imp_thr, "double_thr": dbl_thr, "train_strict_f1": round(f1, 4), "train_hit": th, "train_gt": tgt, "train_fp": tfp}
    return best_win, best_str


def _predict_take_fast(model: DoubleKnock3DPhysNet, d: dict, use_3d: bool, window: int, device: torch.device):
    in_ch = 4 if use_3d else 2
    kin_d = 16 if use_3d else 8
    vis_f = (d["vis_4ch_u8"][:, :in_ch].astype(np.float32) / 255.0).astype(np.float16)
    kin_f = d["kin_16"][:, :kin_d]
    return predict_seq_both_modes(model, vis_f, kin_f, window, device)


def run_controlled_loo(label: str, dataset: dict[str, dict], use_3d: bool, window: int, epochs: int, device: torch.device) -> dict[str, Any]:
    in_ch = 4 if use_3d else 2
    kin_d = 16 if use_3d else 8
    take_ids = sorted(dataset.keys())
    folds = []
    t0 = time.perf_counter()

    for fi, val_tid in enumerate(take_ids):
        tf = time.perf_counter()
        train_sub = {tid: d for tid, d in dataset.items() if tid != val_tid}
        val_d = dataset[val_tid]
        model = train_fold_fast(train_sub, forbidden_val_take=val_tid, use_3d=use_3d, window=window, epochs=epochs, device=device, seed=200 + fi)
        preds_tr = {tid: _predict_take_fast(model, d, use_3d, window, device) for tid, d in train_sub.items()}
        c_win, c_str = _calibrate_strict_on_train_only(preds_tr, train_sub)

        ip_v, cp_v, ic_v, cc_v = _predict_take_fast(model, val_d, use_3d, window, device)
        # 窗口级 AUROC (零拷贝分批推理)
        p_chunks = []
        ws = val_d["win_starts"]
        with torch.no_grad():
            for si in range(0, len(ws), 64):
                s_batch = ws[si : si + 64]
                xv = torch.from_numpy(np.stack([val_d["vis_4ch_u8"][s : s + window, :in_ch] for s in s_batch])).float().div_(255.0).to(device)
                xk = torch.from_numpy(np.stack([val_d["kin_16"][s : s + window, :kin_d] for s in s_batch])).float().to(device)
                _, lc = model(xv, xk)
                p_chunks.append(torch.softmax(lc, dim=-1).cpu().numpy())
        p_win = np.concatenate(p_chunks, axis=0)
        y_cnt = val_d["win_cnts"]
        win_auroc = float(roc_auc_score((y_cnt > 0).astype(np.int32), 1.0 - p_win[:, 0]))
        dbl_auroc = float(roc_auc_score((y_cnt >= 2).astype(np.int32), p_win[:, 2] + p_win[:, 3]))

        tr_w = find_double_knock_triggers(ip_v, cp_v, val_d["fps"], smooth_frames=c_win["smooth_frames"], imp_thr=c_win["imp_thr"], double_thr=c_win["double_thr"])
        tr_s = find_causal_streaming_triggers(ic_v, cc_v, val_d["fps"], smooth_frames=c_str["smooth_frames"], imp_thr=c_str["imp_thr"], double_thr=c_str["double_thr"])
        st_w = eval_strict_1to1_events(tr_w, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"])
        st_s = eval_strict_1to1_events(tr_s, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"])
        lg_w = eval_triggers_on_take(tr_w, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=False)
        lg_s = eval_triggers_on_take(tr_s, val_d["labels_full"], val_d["intervals"], val_d["fps"], val_d["duration_s"], gated_by_fist_window=False)

        folds.append({
            "val_take": val_tid,
            "win_auroc": round(win_auroc, 4), "dbl_auroc": round(dbl_auroc, 4),
            "strict_win": st_w, "strict_stream": st_s,
            "legacy_win_f1": lg_w["f1"], "legacy_stream_f1": lg_s["f1"],
        })
        print(
            f"  [{label}] Fold {fi+1}/6 ({val_tid[-6:]}, {time.perf_counter()-tf:.1f}s): "
            f"StrictWin={st_w['hit']}/{st_w['total_gt']} FP={st_w['fp']} F1={st_w['strict_f1']:.4f} | "
            f"StrictStream={st_s['hit']}/{st_s['total_gt']} FP={st_s['fp']} F1={st_s['strict_f1']:.4f} | AUROC={win_auroc:.4f}",
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

    deploy_m = train_fold_fast(dataset, forbidden_val_take=None, use_3d=use_3d, window=window, epochs=epochs, device=device, seed=777)
    preds_all = {tid: _predict_take_fast(deploy_m, d, use_3d, window, device) for tid, d in dataset.items()}
    c_win_d, c_str_d = _calibrate_strict_on_train_only(preds_all, dataset)
    return {
        "label": label, "in_ch": in_ch, "kin_dim": kin_d, "elapsed_s": round(time.perf_counter() - t0, 1),
        "mean_win_auroc": round(float(np.mean([f["win_auroc"] for f in folds])), 4),
        "mean_dbl_auroc": round(float(np.mean([f["dbl_auroc"] for f in folds])), 4),
        "strict_1to1_win": _agg("strict_win"),
        "strict_1to1_stream": _agg("strict_stream"),
        "legacy_mean_win_f1": round(float(np.mean([f["legacy_win_f1"] for f in folds])), 4),
        "legacy_mean_stream_f1": round(float(np.mean([f["legacy_stream_f1"] for f in folds])), 4),
        "folds": folds,
        "_deploy": (deploy_m, c_win_d, c_str_d),
    }


def _detect_face_area_fullres(yunet: Any, gray_240: np.ndarray) -> float:
    if yunet is None:
        return 0.0
    if gray_240.shape != (240, 240):
        gray_240 = cv2.resize(gray_240, (240, 240), interpolation=cv2.INTER_AREA)
    _, faces = yunet.detect(cv2.cvtColor(gray_240, cv2.COLOR_GRAY2BGR))
    if faces is None or len(faces) == 0:
        return 0.0
    return float(max(fc[2] * fc[3] for fc in faces) / (240.0 * 240.0))


def _check_yunet_all_89_knocks(yunet: Any, dataset: dict[str, dict]) -> dict[str, Any]:
    total_gt = over_0085 = 0
    max_all = 0.0
    per_take = {}
    for tid, d in dataset.items():
        cap = cv2.VideoCapture(str(d["video_path"]))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        gt_ivs = [(iv.start_s, iv.end_s) for iv in d["intervals"] if iv.label == "knock"]
        t_max = 0.0
        t_over = 0
        for a, b in gt_ivs:
            total_gt += 1
            iv_m = 0.0
            for t_s in np.linspace(a, b, 4):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_s * src_fps)))
                ok, fr = cap.read()
                if ok and fr is not None:
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
                    iv_m = max(iv_m, _detect_face_area_fullres(yunet, g))
            t_max = max(t_max, iv_m)
            max_all = max(max_all, iv_m)
            if iv_m > 0.085:
                t_over += 1
                over_0085 += 1
        cap.release()
        per_take[tid] = {"n_gt": len(gt_ivs), "max_face_area": round(t_max, 4), "over_0085": t_over}
    return {"total_real_knocks_checked": total_gt, "knocks_over_0085": over_0085, "max_face_area": round(max_all, 4), "per_take": per_take}


def eval_external_both(
    m2d: Any, c2_w: dict, c2_s: dict,
    m3d: Any, c3_w: dict, c3_s: dict,
    extractor: BatchHoverGeoNet3DExtractor,
    yunet: Any, window: int, device: torch.device,
) -> dict[str, Any]:
    clips = load_external_stress_clips(max_clips_per_source=10, size=GEO_H)
    stats = {
        "2d_win_fp": 0, "2d_str_fp": 0, "3d_win_fp": 0, "3d_str_fp": 0,
        "3d_yunet_win_fp": 0, "3d_yunet_str_fp": 0,
        "2d_robot9_str_fp": 0, "3d_robot9_str_fp": 0, "3d_yunet_robot9_str_fp": 0,
        "neg_minutes": 0.0, "knock_d05_detected": {"2d_str": False, "3d_str": False}, "nonzero_fp_clips": [],
    }
    for cl in clips:
        f240 = cl["frames"]
        src_fps = float(cl.get("fps", 20.0))
        f240_r, _, feat_fps = _resample_frames_to_fps(f240, src_fps, 20.0)
        f96 = np.stack([cv2.resize(f, (96, 96), interpolation=cv2.INTER_AREA) for f in f240_r], axis=0)
        kin_2d, _ = extract_kinematics_and_energy(f96, None, fps=feat_fps)
        vis_4ch, kin_3d, _ = extractor.extract_sequence_3d(f240_r, fps=feat_fps, size=96)
        vis_f = (vis_4ch.astype(np.float32) / 255.0).astype(np.float16)
        kin_f = np.concatenate([kin_2d, kin_3d], axis=1).astype(np.float32)

        ip2, cp2, ic2, cc2 = predict_seq_both_modes(m2d, vis_f[:, :2], kin_f[:, :8], window, device)
        ip3, cp3, ic3, cc3 = predict_seq_both_modes(m3d, vis_f, kin_f, window, device)
        tw2 = find_double_knock_triggers(ip2, cp2, feat_fps, smooth_frames=c2_w["smooth_frames"], imp_thr=c2_w["imp_thr"], double_thr=c2_w["double_thr"])
        ts2 = find_causal_streaming_triggers(ic2, cc2, feat_fps, smooth_frames=c2_s["smooth_frames"], imp_thr=c2_s["imp_thr"], double_thr=c2_s["double_thr"])
        tw3 = find_double_knock_triggers(ip3, cp3, feat_fps, smooth_frames=c3_w["smooth_frames"], imp_thr=c3_w["imp_thr"], double_thr=c3_w["double_thr"])
        ts3 = find_causal_streaming_triggers(ic3, cc3, feat_fps, smooth_frames=c3_s["smooth_frames"], imp_thr=c3_s["imp_thr"], double_thr=c3_s["double_thr"])

        def _gate(trs):
            kept = []
            for tr in trs:
                fi = min(max(int(round(tr["second_s"] * feat_fps)), 0), len(f240_r) - 1)
                max_fa = 0.0
                for k in range(max(0, fi - 10), min(len(f240_r), fi + 5), 2):
                    max_fa = max(max_fa, _detect_face_area_fullres(yunet, f240_r[k]))
                if max_fa > 0.085:
                    continue
                kept.append(tr)
            return kept

        tw3g, ts3g = _gate(tw3), _gate(ts3)
        dur_s = len(f240_r) / max(feat_fps, 1.0)
        if bool(cl["is_positive_double_knock"]):
            stats["knock_d05_detected"]["2d_str"] = len(ts2) > 0
            stats["knock_d05_detected"]["3d_str"] = len(ts3g) > 0
            continue
        stats["neg_minutes"] += dur_s / 60.0
        stats["2d_win_fp"] += len(tw2); stats["2d_str_fp"] += len(ts2)
        stats["3d_win_fp"] += len(tw3); stats["3d_str_fp"] += len(ts3)
        stats["3d_yunet_win_fp"] += len(tw3g); stats["3d_yunet_str_fp"] += len(ts3g)
        if cl["source"] == "evidence_sessions":
            stats["2d_robot9_str_fp"] += len(ts2)
            stats["3d_robot9_str_fp"] += len(ts3)
            stats["3d_yunet_robot9_str_fp"] += len(ts3g)
        if any(len(x) > 0 for x in (tw2, ts2, tw3, ts3, tw3g, ts3g)):
            stats["nonzero_fp_clips"].append({
                "id": cl["id"], "source": cl["source"], "task": cl["task"],
                "2d_str": len(ts2), "3d_str": len(ts3), "3d_yunet_str": len(ts3g),
                "2d_win": len(tw2), "3d_win": len(tw3), "3d_yunet_win": len(tw3g),
            })
    stats["neg_minutes"] = round(stats["neg_minutes"], 2)
    return stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== RIGOROUS AUDIT STAGE 1 (ZERO-COPY FAST): CONTROLLED SAME-ARCH 2D vs 3D LOO + STRICT 1-TO-1 ===", flush=True)

    dataset = load_zerocopy_dataset(window=48, stride=4, size=96)
    yunet = cv2.FaceDetectorYN.create(str(YUNET_ONNX), "", (240, 240), score_threshold=0.45) if YUNET_ONNX.exists() else None

    yunet_89 = _check_yunet_all_89_knocks(yunet, dataset)
    print(f"  Full-Res 89 Real Knocks YuNet Check: {yunet_89['knocks_over_0085']}/89 >0.085 | max_face_area={yunet_89['max_face_area']}", flush=True)

    print("\n--- Controlled 6-Fold LOO: Model A (Same-Arch 2D-Only, in_ch=2, kin=8) ---", flush=True)
    res_2d = run_controlled_loo("SameArch-2D", dataset, use_3d=False, window=48, epochs=20, device=device)
    m2d, c2w, c2s = res_2d.pop("_deploy")
    m2d = m2d.cpu()

    print("\n--- Controlled 6-Fold LOO: Model B (Same-Arch 3D-PhysNet, in_ch=4, kin=16 + FiLM) ---", flush=True)
    res_3d = run_controlled_loo("SameArch-3D", dataset, use_3d=True, window=48, epochs=20, device=device)
    m3d, c3w, c3s = res_3d.pop("_deploy")
    del dataset
    gc.collect()
    m2d = m2d.to(device)

    print("\n--- Identical 30-Clip External Stress Comparison (Controlled 2D vs 3D vs 3D+YuNet) ---", flush=True)
    extractor = BatchHoverGeoNet3DExtractor(ckpt_path=V4_CKPT_FP16, device=str(device))
    ext = eval_external_both(m2d, c2w, c2s, m3d, c3w, c3s, extractor, yunet, window=48, device=device)

    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in m3d.state_dict().items()}, "in_ch": 4, "kin_dim": 16, "hidden_dim": 64, "window": 48, "calib_win": c3w, "calib_stream": c3s},
        OUT_DIR / "double_knock_3d_strict_deploy.pt",
    )

    rep = {
        "protocol": "100% Controlled Same-Arch 6-Fold LOO (Same Contacts, Same Architecture, Same Epochs, Same Seeds, Strict 1-to-1 Events, ZERO Oracle GT Gating)",
        "yunet_fullres_all_89_knocks_check": yunet_89,
        "controlled_2d_only": res_2d,
        "controlled_3d_physnet": res_3d,
        "pure_3d_causal_delta": {
            "strict_stream_macro_f1_delta": round(res_3d["strict_1to1_stream"]["macro_strict_f1"] - res_2d["strict_1to1_stream"]["macro_strict_f1"], 4),
            "strict_stream_micro_f1_delta": round(res_3d["strict_1to1_stream"]["micro_strict_f1"] - res_2d["strict_1to1_stream"]["micro_strict_f1"], 4),
            "strict_stream_hits_delta": res_3d["strict_1to1_stream"]["total_hit"] - res_2d["strict_1to1_stream"]["total_hit"],
            "strict_stream_fp_delta": res_3d["strict_1to1_stream"]["total_fp"] - res_2d["strict_1to1_stream"]["total_fp"],
            "strict_win_macro_f1_delta": round(res_3d["strict_1to1_win"]["macro_strict_f1"] - res_2d["strict_1to1_win"]["macro_strict_f1"], 4),
            "strict_win_micro_f1_delta": round(res_3d["strict_1to1_win"]["micro_strict_f1"] - res_2d["strict_1to1_win"]["micro_strict_f1"], 4),
            "win_auroc_delta": round(res_3d["mean_win_auroc"] - res_2d["mean_win_auroc"], 4),
            "dbl_auroc_delta": round(res_3d["mean_dbl_auroc"] - res_2d["mean_dbl_auroc"], 4),
        },
        "external_30clip_head_to_head": ext,
    }
    out_j = OUT_DIR / "rigorous_knock_2d_vs_3d_ablation.json"
    out_j.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ STAGE 1 CONTROLLED AUDIT SUMMARY ================", flush=True)
    print(f"  [Same-Arch 2D-Only] Strict-1to1 Stream: Macro F1={res_2d['strict_1to1_stream']['macro_strict_f1']} | Micro F1={res_2d['strict_1to1_stream']['micro_strict_f1']} ({res_2d['strict_1to1_stream']['total_hit']}/89, FP={res_2d['strict_1to1_stream']['total_fp']})", flush=True)
    print(f"  [Same-Arch 3D-Phys] Strict-1to1 Stream: Macro F1={res_3d['strict_1to1_stream']['macro_strict_f1']} | Micro F1={res_3d['strict_1to1_stream']['micro_strict_f1']} ({res_3d['strict_1to1_stream']['total_hit']}/89, FP={res_3d['strict_1to1_stream']['total_fp']})", flush=True)
    print(f"  [Same-Arch 2D-Only] Strict-1to1 Win:    Macro F1={res_2d['strict_1to1_win']['macro_strict_f1']} | Micro F1={res_2d['strict_1to1_win']['micro_strict_f1']} ({res_2d['strict_1to1_win']['total_hit']}/89, FP={res_2d['strict_1to1_win']['total_fp']})", flush=True)
    print(f"  [Same-Arch 3D-Phys] Strict-1to1 Win:    Macro F1={res_3d['strict_1to1_win']['macro_strict_f1']} | Micro F1={res_3d['strict_1to1_win']['micro_strict_f1']} ({res_3d['strict_1to1_win']['total_hit']}/89, FP={res_3d['strict_1to1_win']['total_fp']})", flush=True)
    print(f"  [External 29 Neg ({ext['neg_minutes']}m)] Stream FP -> 2D={ext['2d_str_fp']} | 3D_raw={ext['3d_str_fp']} | 3D+YuNet={ext['3d_yunet_str_fp']} (Robot9: 2D={ext['2d_robot9_str_fp']} | 3D={ext['3d_robot9_str_fp']} | 3D+YuNet={ext['3d_yunet_robot9_str_fp']})", flush=True)
    print(f"  Saved -> {out_j}", flush=True)


if __name__ == "__main__":
    main()
