"""诚实标定：平滑窗口和阈值只用训练 take 选（在训练集上），再原样套到留出 take。

集成三个模型（gru/tcn/bigru，各自留一折），输出留出 take 上的两敲触发指标。
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
from train_knock import build_model, normalize_probs, smooth_prob  # noqa: E402

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
FOLDS = {
    "gru": ROOT / "runs_knock" / "loo8_20261108" / "folds",
    "tcn": ROOT / "runs_knock" / "loo9_20261108" / "folds",
    "bigru": ROOT / "runs_knock" / "loo9_20261108" / "folds",
}
KNOCK = CLASS_TO_IDX["knock"]
SMOOTHS = (3, 5, 10, 20)
THRS = np.arange(-0.5, 1.21, 0.1)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def triggers(prob: np.ndarray, fps: float, thr: float, peak_gap: int = 8) -> list[float]:
    return [t["second_s"] for t in dk.find_triggers(prob, fps, thr, peak_gap, 0.25, 1.5, 0.0)]


def pair_counts(trig, labels: np.ndarray, frames: int, fps: float, gt) -> tuple[int, int, int]:
    tp = fp = 0
    for t in trig:
        i = min(max(int(round(t * fps)), 0), frames - 1)
        if labels[i] == KNOCK:
            tp += 1
        elif labels[i] != IGNORE:
            fp += 1
    hit = sum(1 for a, b in gt if any(a - 0.5 <= t <= b + 0.5 for t in trig))
    return tp, fp, hit


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    labels, gt, fps = {}, {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        labels[tid] = frame_labels(iv, take.video_frames, take.fps)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]
        fps[tid] = float(take.fps)

    ckpts: dict[str, dict] = {}
    for name, folder in FOLDS.items():
        for path in sorted(folder.glob(f"*_{name}.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpts.setdefault(ckpt["val_take"], {})[name] = ckpt

    print("| 留出 take | 选中的平滑/阈值 | 命中 | P | R | F1 | 误报/分钟 |")
    print("|---|---|---|---|---|---|---|")
    result = []
    for val, take in takes.items():
        models = {}
        for name in FOLDS:
            ckpt = ckpts[val][name]
            model = build_model(ckpt["model"])
            model.load_state_dict(ckpt["state_dict"])
            model.to(device).eval()
            models[name] = (ckpt, model)
        train_ids = [t for t in takes if t != val]

        # 训练 take 上的集成概率（不同平滑），用于选平滑+阈值
        train_probs: dict[int, dict[str, np.ndarray]] = {s: {} for s in SMOOTHS}
        for tid in train_ids:
            frames = decode_gray(takes[tid].video_path, int(models["gru"][0]["size"]))
            parts = {s: [] for s in SMOOTHS}
            for name, (ckpt, model) in models.items():
                raw = dk.predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]), device)
                for s in SMOOTHS:
                    p = smooth_prob(raw, s)
                    parts[s].append(normalize_probs({"v": p}, ckpt["normalize"])["v"])
            for s in SMOOTHS:
                train_probs[s][tid] = np.mean(parts[s], axis=0)

        best = None
        for s in SMOOTHS:
            for thr in THRS:
                tp = fp = hit = 0
                for tid in train_ids:
                    t = triggers(train_probs[s][tid], fps[tid], float(thr))
                    a, b, c = pair_counts(t, labels[tid], len(labels[tid]), fps[tid], gt[tid])
                    tp += a
                    fp += b
                    hit += c
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = hit / 46
                score = f1(prec, rec)
                if best is None or score > best[0]:
                    best = (score, s, float(thr), prec, rec, hit, fp)

        _, s, thr, _, _, _, _ = best
        frames = decode_gray(take.video_path, int(models["gru"][0]["size"]))
        parts = []
        for name, (ckpt, model) in models.items():
            raw = dk.predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]), device)
            p = smooth_prob(raw, s)
            parts.append(normalize_probs({"v": p}, ckpt["normalize"])["v"])
        prob = np.mean(parts, axis=0)
        tr = triggers(prob, fps[val], thr)
        tp, fp, hit = pair_counts(tr, labels[val], len(labels[val]), fps[val], gt[val])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = hit / len(gt[val]) if gt[val] else 0.0
        fa = fp / (takes[val].duration_s / 60.0)
        print(f"| {val} | smooth={s}, thr={thr:+.1f} | {hit}/{len(gt[val])} | {prec:.3f} | "
              f"{rec:.3f} | {f1(prec, rec):.3f} | {fa:.2f} |")
        result.append((f1(prec, rec), hit, len(gt[val])))
    print(f"\n均值两敲 F1：{np.mean([r[0] for r in result]):.3f}；"
          f"总命中 {sum(r[1] for r in result)}/46")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())