"""手腕 ROI 敲击检测：CNN 编码器 + GRU 时序头，按 take 留一验证。

输入是 YOLO pose 腕点裁出的 96x96 手部 ROI（见 `pose_roi.py`），不再是整帧：
手在画面里占比更大，背景干扰更少。帧差在预处理时一次算好，训练时按起点批量
gather，避免 Python 逐样本开销（第一版就是这么慢到跑不动的）。

用法（在 D:\\robot_work 下）：
    python pc/train/train_knock_roi.py --dataset dataset --roi runs_knock/pose_roi \
        --out runs_knock/roi_20261109 --models gru --epochs 20 --device cuda:0

评测纪律：阈值/平滑只在训练 take 上选，留出 take 只评估；误报只统计已标注负样本
帧（未标注区单独记为 unknown，不算负样本）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_data import (CLASS_TO_IDX, IGNORE, frame_labels, load_intervals,
                           load_labels, load_takes)
from train_knock import KnockCnnGRU, KnockCnnTCN, peak_indices, smooth_prob

ROI_SIZE = 96
GRID = np.round(np.arange(-0.5, 3.01, 0.1), 2)

_LABELS: dict[str, str] = {}


def load_take_arrays(take, roi_dir: Path, context: bool = False):
    """返回 gray/diff/labels/intervals，以及可选的整帧下采样上下文。"""
    npz = np.load(roi_dir / f"{take.take_id}.npz")
    gray = np.ascontiguousarray(npz["frames"])
    diff = np.zeros_like(gray)
    diff[1:] = np.abs(np.diff(gray.astype(np.int16), axis=0)).astype(np.uint8)
    intervals = load_intervals(take, _LABELS)
    labels = frame_labels(intervals, gray.shape[0], take.fps)
    g_gray = g_diff = None
    if context:
        from interval_data import decode_gray

        g_gray = decode_gray(take.video_path, ROI_SIZE)
        g_diff = np.zeros_like(g_gray)
        g_diff[1:] = np.abs(np.diff(g_gray.astype(np.int16), axis=0)).astype(np.uint8)
    return gray, diff, labels, intervals, g_gray, g_diff


def gather_batch(gray: np.ndarray, diff: np.ndarray, starts: np.ndarray, window: int,
                 g_gray: np.ndarray | None = None, g_diff: np.ndarray | None = None):
    """把若干窗口起点 gather 成 (B, window, C, 96, 96) uint8；context 时 C=4。"""
    n = gray.shape[0]
    idx = starts[:, None] + np.arange(window)[None, :]
    np.clip(idx, 0, n - 1, out=idx)
    channels = 4 if g_gray is not None else 2
    x = np.empty((len(starts), window, channels, gray.shape[1], gray.shape[2]), dtype=np.uint8)
    x[:, :, 0] = gray[idx]
    x[:, :, 1] = diff[idx]
    if g_gray is not None:
        x[:, :, 2] = g_gray[idx]
        x[:, :, 3] = g_diff[idx]
    return x


def gather_labels(labels: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    n = labels.shape[0]
    idx = starts[:, None] + np.arange(window)[None, :]
    np.clip(idx, 0, n - 1, out=idx)
    y = labels[idx]
    # 窗口超出末尾的补齐帧不参与 loss
    valid = starts[:, None] + np.arange(window)[None, :] < n
    y = np.where(valid, y, IGNORE)
    return y


def predict_take(model, gray, diff, window: int, stride: int, device, batch: int = 32,
                 g_gray=None, g_diff=None) -> np.ndarray:
    n = gray.shape[0]
    starts = np.arange(0, max(1, n - 1), stride)
    acc = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            s = starts[i:i + batch]
            xb = gather_batch(gray, diff, s, window, g_gray, g_diff)
            x = torch.from_numpy(xb).to(device).float().div_(255.0)
            prob = torch.sigmoid(model(x)).cpu().numpy()
            for k, st in enumerate(s):
                stop = min(st + window, n)
                acc[st:stop] += prob[k, : stop - st]
                cnt[st:stop] += 1.0
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def zscore(p: np.ndarray) -> np.ndarray:
    return ((p - float(np.mean(p))) / (float(np.std(p)) or 1e-6)).astype(np.float32)


def frame_scores(prob: np.ndarray, labels: np.ndarray, smooth: int, threshold: float):
    p = zscore(smooth_prob(prob, smooth))
    mask = labels != IGNORE
    target = (labels == CLASS_TO_IDX["knock"]) & mask
    pred = (p >= threshold) & mask
    tp = int((pred & target).sum())
    fp = int((pred & ~target & mask).sum())
    fn = int((~pred & target).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec


def auroc(prob: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != IGNORE
    p = prob[mask]
    y = (labels == CLASS_TO_IDX["knock"])[mask]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pair_metrics_labeled(prob, fps, threshold, gt_events, labels, duration_s,
                         min_gap_s=0.25, max_gap_s=1.5, margin_s=0.5, peak_min_gap=8):
    """两敲触发；误报只计“落在已标注负样本帧”的触发，未标注区单独统计。"""
    peaks = peak_indices(prob, threshold, peak_min_gap)
    times = [i / fps for i in peaks]
    triggers = [b for a, b in zip(times, times[1:]) if min_gap_s <= b - a <= max_gap_s]
    tp = fp = unknown = 0
    for t in triggers:
        if any(a - margin_s <= t <= b + margin_s for a, b in gt_events):
            tp += 1
            continue
        idx = min(int(round(t * fps)), len(labels) - 1)
        if 0 <= idx < len(labels) and labels[idx] == IGNORE:
            unknown += 1
        else:
            fp += 1
    hit = sum(1 for a, b in gt_events if any(a - margin_s <= t <= b + margin_s for t in triggers))
    prec = tp / len(triggers) if triggers else 0.0
    rec = hit / len(gt_events) if gt_events else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    labeled_neg_frames = int(((labels != IGNORE) & (labels != CLASS_TO_IDX["knock"])).sum())
    neg_minutes = max(labeled_neg_frames / fps / 60.0, 1e-6)
    return {"triggers": len(triggers), "tp": tp, "fp": fp, "unknown": unknown, "hit": hit,
            "gt": len(gt_events), "precision": prec, "recall": rec, "f1": f1,
            "fa_per_min_labeled": fp / neg_minutes}


def choose_smooth_threshold(probs, labels):
    best = (-1.0, 3, 0.0)
    for smooth in (3, 5, 10, 20):
        for thr in GRID:
            f1s = [frame_scores(v, labels[k], smooth, float(thr))[0] for k, v in probs.items()]
            f1 = float(np.mean(f1s))
            if f1 > best[0]:
                best = (f1, smooth, float(thr))
    return {"train_f1": best[0], "smooth": best[1], "threshold": best[2]}


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """GPU 上做轻量增强：镜像、亮度/对比度抖动、短时遮挡。"""
    b, t, c, h, w = x.shape
    flip = torch.rand(b, device=x.device) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[4])
    scale = 1.0 + (torch.rand(b, 1, 1, 1, 1, device=x.device) - 0.5) * 0.24
    bias = (torch.rand(b, 1, 1, 1, 1, device=x.device) - 0.5) * 0.12
    x = (x * scale + bias).clamp_(0.0, 1.0)
    if torch.rand(1).item() < 0.1:
        m = int(torch.randint(2, 6, (1,)).item())
        s = int(torch.randint(0, max(1, t - m), (1,)).item())
        x[:, s:s + m] = x[:, s:s + m] * 0.2
    return x


def train_fold(args, train_takes, val_take, data, device):
    t0 = time.time()
    window = args.window
    in_ch = 4 if args.context else 2
    model = (KnockCnnGRU if args.models == "gru" else KnockCnnTCN)(in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pos = neg = 0
    for t in train_takes:
        lab = data[t.take_id]["labels"]
        pos += int((lab == CLASS_TO_IDX["knock"]).sum())
        neg += int(((lab != IGNORE) & (lab != CLASS_TO_IDX["knock"])).sum())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    plan = []
    for t in train_takes:
        n = data[t.take_id]["gray"].shape[0]
        plan += [(t.take_id, s) for s in range(0, max(1, n - 1), args.stride)]
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(plan))
        total = 0.0
        steps = 0
        for i in range(0, len(order), args.batch):
            chunk = [plan[j] for j in order[i:i + args.batch]]
            xs, ys = [], []
            for take_id, s in chunk:
                d = data[take_id]
                xs.append(gather_batch(d["gray"], d["diff"], np.array([s]), window,
                                       d.get("global_gray"), d.get("global_diff"))[0])
                ys.append(gather_labels(d["labels"], np.array([s]), window)[0])
            x = torch.from_numpy(np.stack(xs)).to(device).float().div_(255.0)
            y = torch.from_numpy(np.stack(ys)).to(device)
            x = augment_batch(x)
            opt.zero_grad()
            logit = model(x)
            mask = y != IGNORE
            target = (y == CLASS_TO_IDX["knock"]).float()  # 二分类：knock=1，其余已标注=0
            loss = (loss_fn(logit, target) * mask).sum() / mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        sched.step()
        print(f"  epoch {epoch + 1}/{args.epochs} loss {total / max(steps, 1):.4f}", flush=True)

    train_probs = {t.take_id: predict_take(model, data[t.take_id]["gray"], data[t.take_id]["diff"],
                                           window, args.pred_stride, device,
                                           g_gray=data[t.take_id].get("global_gray"),
                                           g_diff=data[t.take_id].get("global_diff"))
                   for t in train_takes}
    sel = choose_smooth_threshold(train_probs, {t.take_id: data[t.take_id]["labels"] for t in train_takes})
    val_prob = predict_take(model, data[val_take.take_id]["gray"], data[val_take.take_id]["diff"],
                            window, args.pred_stride, device,
                            g_gray=data[val_take.take_id].get("global_gray"),
                            g_diff=data[val_take.take_id].get("global_diff"))
    vp = zscore(smooth_prob(val_prob, sel["smooth"]))
    labels = data[val_take.take_id]["labels"]
    f1, prec, rec = frame_scores(val_prob, labels, sel["smooth"], sel["threshold"])
    auc = auroc(vp, labels)
    gt = [(iv.start_s, iv.end_s) for iv in data[val_take.take_id]["intervals"] if iv.label == "knock"]
    pm = pair_metrics_labeled(vp, val_take.fps, sel["threshold"], gt, labels, val_take.duration_s)
    return {"val": {"frame_f1": f1, "precision": prec, "recall": rec, "auroc": auc, **pm},
            "selection": sel, "seconds": round(time.time() - t0, 1), "model": model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--roi", default="runs_knock/pose_roi")
    ap.add_argument("--out", default="runs_knock/roi_20261109")
    ap.add_argument("--models", choices=["gru", "tcn"], default="gru")
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--pred-stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--context", action="store_true", help="额外输入整帧下采样上下文（4 通道）")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    global _LABELS
    dataset = Path(args.dataset)
    _LABELS = load_labels(dataset)
    takes = [t for t in load_takes(dataset) if (Path(args.roi) / f"{t.take_id}.npz").exists()]
    if len(takes) < 2:
        raise SystemExit("至少需要 2 条带 ROI 的 take")
    data = {}
    for t in takes:
        gray, diff, labels, intervals, g_gray, g_diff = load_take_arrays(t, Path(args.roi), args.context)
        data[t.take_id] = {"gray": gray, "diff": diff, "labels": labels, "intervals": intervals,
                           "global_gray": g_gray, "global_diff": g_diff}
        print(f"{t.take_id}: frames={gray.shape[0]} fps={t.fps} "
              f"knock_frames={int((labels == CLASS_TO_IDX['knock']).sum())} "
              f"labeled={int((labels != IGNORE).sum())}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.set_num_threads(6)
    results = {}
    for val_take in takes:
        if args.only and val_take.take_id != args.only:
            continue
        train_takes = [t for t in takes if t.take_id != val_take.take_id]
        print(f"\n=== fold: held out {val_take.take_id} ===", flush=True)
        res = train_fold(args, train_takes, val_take, data, device)
        results[val_take.take_id] = {k: v for k, v in res.items() if k != "model"}
        torch.save({"state_dict": res["model"].state_dict(), "window": args.window,
                    "selection": res["selection"], "models": args.models,
                    "in_ch": 4 if args.context else 2},
                   out / f"fold_{val_take.take_id}.pt")
        v = res["val"]
        print(f"frameF1={v['frame_f1']:.4f} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"AUC={v['auroc']:.4f} | pair hit={v['hit']}/{v['gt']} tp={v['tp']} fp={v['fp']} "
              f"unknown={v['unknown']} F1={v['f1']:.3f} FA/min={v['fa_per_min_labeled']:.2f} | "
              f"sel smooth={res['selection']['smooth']} thr={res['selection']['threshold']} "
              f"({res['seconds']}s)", flush=True)

    metrics = {
        "config": vars(args),
        "takes": [t.take_id for t in takes],
        "folds": results,
        "mean": {
            "frame_f1": float(np.mean([r["val"]["frame_f1"] for r in results.values()])),
            "auroc": float(np.mean([r["val"]["auroc"] for r in results.values()])),
            "pair_f1": float(np.mean([r["val"]["f1"] for r in results.values()])),
            "hit": int(sum(r["val"]["hit"] for r in results.values())),
            "gt": int(sum(r["val"]["gt"] for r in results.values())),
            "fp": int(sum(r["val"]["fp"] for r in results.values())),
            "unknown": int(sum(r["val"]["unknown"] for r in results.values())),
        } if results else {},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 手腕 ROI 敲击检测（按 take 留一验证）", ""]
    for take_id, r in results.items():
        v = r["val"]
        lines += [f"## {take_id}",
                  f"- 帧级 F1 {v['frame_f1']:.4f}（P {v['precision']:.3f} / R {v['recall']:.3f}），AUROC {v['auroc']:.4f}",
                  f"- 两敲触发：命中 {v['hit']}/{v['gt']}，TP {v['tp']}，已标注负样本误报 {v['fp']}，未标注区 {v['unknown']}，F1 {v['f1']:.3f}，FA/min {v['fa_per_min_labeled']:.2f}",
                  f"- 训练 take 上选的平滑 {r['selection']['smooth']} / z 阈值 {r['selection']['threshold']}（训练 F1 {r['selection']['train_f1']:.3f}）",
                  ""]
    if results:
        m = metrics["mean"]
        lines.append(f"**平均**：帧级 F1 {m['frame_f1']:.4f}，AUROC {m['auroc']:.4f}，"
                     f"两敲 {m['hit']}/{m['gt']}，误报 {m['fp']}，未标注 {m['unknown']}，pair F1 {m['pair_f1']:.3f}")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(json.dumps(metrics["mean"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()