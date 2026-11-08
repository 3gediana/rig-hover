"""诚实留一评估：阈值只来自各折 checkpoint（在训练 take 上选），不看留出 take。

- gru 折来自 loo8，tcn/bigru 折来自 loo9；每个 checkpoint 都只训练过另外 3 条 take。
- 集成概率 = 三个模型 z 分数的平均；阈值 = 三个 checkpoint 阈值的平均。
- 帧级 F1 / AUROC；事件级；两敲触发（已标注负样本口径）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect_knocks as dk  # noqa: E402
from interval_data import (  # noqa: E402
    CLASS_TO_IDX,
    IGNORE,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
)
from train_knock import (  # noqa: E402
    build_model,
    event_metrics,
    normalize_probs,
    runs_above,
    peak_indices,
    smooth_prob,
)

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
FOLDS = {
    "gru": ROOT / "runs_knock" / "loo8_20261108" / "folds",
    "tcn": ROOT / "runs_knock" / "loo9_20261108" / "folds",
    "bigru": ROOT / "runs_knock" / "loo9_20261108" / "folds",
}
KNOCK = CLASS_TO_IDX["knock"]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    labels, gt = {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        labels[tid] = frame_labels(iv, take.video_frames, take.fps)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]

    ckpts: dict[str, dict] = {}
    for name, folder in FOLDS.items():
        for path in sorted(folder.glob(f"*_{name}.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpts.setdefault(ckpt["val_take"], {})[name] = ckpt

    print("| 留出 take | 帧F1 | AUROC | 事件F1 | 两敲 P | 两敲 R | 两敲 F1 | 误报/分钟 |")
    print("|---|---|---|---|---|---|---|---|")
    totals = []
    total_minutes = 0.0
    total_fp = 0
    total_hit = 0
    for val, take in takes.items():
        parts = []
        thresholds, pair_thresholds = [], []
        for name in FOLDS:
            ckpt = ckpts[val][name]
            model = build_model(ckpt["model"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            frames = decode_gray(take.video_path, int(ckpt["size"]))
            p = dk.predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]))
            p = smooth_prob(p, int(ckpt["smooth_frames"]))
            p = normalize_probs({"v": p}, ckpt["normalize"])["v"]
            parts.append(p.astype(np.float32))
            thresholds.append(float(ckpt["threshold"]))
            pair_thresholds.append(float(ckpt.get("pair_threshold", ckpt["threshold"])))
        prob = np.mean(parts, axis=0)
        thr = float(np.mean(thresholds))
        pair_thr = float(np.mean(pair_thresholds))
        fps = float(take.fps)
        target = (labels[val] == KNOCK).astype(np.float32)
        mask = labels[val] != IGNORE
        pred = (prob >= thr) & mask
        tp = float(np.sum(pred & (target == 1)))
        fp_frame = float(np.sum(pred & (target == 0)))
        fn = float(np.sum((~pred) & (target == 1)))
        prec = tp / (tp + fp_frame) if tp + fp_frame else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        # AUROC（只用已标注帧）
        order = np.argsort(prob[mask])
        y = target[mask][order]
        ranks = np.arange(1, len(y) + 1)
        n_pos = float(np.sum(y == 1))
        n_neg = float(len(y) - n_pos)
        auroc = (np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if n_pos and n_neg else float("nan")
        runs = runs_above(prob, thr, int(ckpt["min_event_frames"]), int(ckpt["max_gap_frames"]))
        pred_events = [(a / fps, b / fps) for a, b in runs]
        em = event_metrics(pred_events, gt[val])

        peak_gap = int(ckpt.get("peak_min_gap", 8))
        iso = float(ckpts[val]["gru"].get("pair_isolation_s", 0.0))
        trig = [t["second_s"] for t in
                dk.find_triggers(prob, fps, pair_thr, peak_gap, 0.25, 1.5, iso)]
        tp_pair = fp_pair = 0
        for t in trig:
            i = min(max(int(round(t * fps)), 0), len(labels[val]) - 1)
            if labels[val][i] == KNOCK:
                tp_pair += 1
            elif labels[val][i] != IGNORE:
                fp_pair += 1
        hit = sum(1 for a, b in gt[val] if any(a - 0.5 <= t <= b + 0.5 for t in trig))
        prec_pair = tp_pair / (tp_pair + fp_pair) if tp_pair + fp_pair else 0.0
        rec_pair = hit / len(gt[val]) if gt[val] else 0.0
        fa = fp_pair / (take.duration_s / 60.0)
        print(f"| {val} | {f1(prec, rec):.3f} | {auroc:.3f} | {em['f1']:.3f} | "
              f"{prec_pair:.3f} | {rec_pair:.3f} | {f1(prec_pair, rec_pair):.3f} | {fa:.2f} |")
        totals.append((f1(prec, rec), auroc, em["f1"], f1(prec_pair, rec_pair)))
        total_minutes += take.duration_s / 60.0
        total_fp += fp_pair
        total_hit += hit

    mean = np.nanmean(np.array(totals, dtype=np.float64), axis=0)
    print(f"\n均值：帧F1 {mean[0]:.3f}  AUROC {mean[1]:.3f}  事件F1 {mean[2]:.3f}  两敲F1 {mean[3]:.3f}")
    print(f"两敲总体：命中 {total_hit}/46，误报 {total_fp} 次 / {total_minutes:.1f} 分钟 = {total_fp / total_minutes:.2f} 次/分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())