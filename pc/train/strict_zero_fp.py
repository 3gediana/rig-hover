"""严格工作点：在训练 take 上要求“零误报”，看留出 take 还能命中多少。

协议：
1. 每个折的集成概率（gru/tcn/bigru，均未见过留出 take）。
2. 平滑候选 {3,5,10,20}，阈值候选 z ∈ [0.2, 3.0]。
3. 在训练 take 上选：已标注负样本上 **0 个误触发** 的前提下，命中敲击段最多的工作点。
4. 原样套到留出 take，报告命中/误报；同时统计落在“未标注帧”上的触发数（未知区域，单独列）。
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
THRS = np.arange(0.2, 3.01, 0.1)


def triggers(prob: np.ndarray, fps: float, thr: float) -> list[float]:
    return [t["second_s"] for t in dk.find_triggers(prob, fps, thr, 8, 0.25, 1.5, 0.0)]


def stats(trig, labels: np.ndarray, fps: float, gt) -> tuple[int, int, int, int]:
    """返回 (命中区间, 已标注负样本误报, 未标注区域触发, 敲击帧上的触发)。"""
    hit = sum(1 for a, b in gt if any(a - 0.5 <= t <= b + 0.5 for t in trig))
    fp_lab = unknown = 0
    for t in trig:
        i = min(max(int(round(t * fps)), 0), len(labels) - 1)
        if labels[i] == IGNORE:
            unknown += 1
        elif labels[i] != KNOCK:
            fp_lab += 1
    return hit, fp_lab, unknown, len(trig)


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

    print("| 留出 take | 训练集选中的参数 | 训练集零误报? | 留出命中 | 留出误报(已标注) | 未标注区域触发 |")
    print("|---|---|---|---|---|---|")
    total_hit = total_gt = total_fp = 0
    for val in takes:
        models = {}
        for name in FOLDS:
            ckpt = ckpts[val][name]
            m = build_model(ckpt["model"])
            m.load_state_dict(ckpt["state_dict"])
            models[name] = (ckpt, m.to(device).eval())
        train_ids = [t for t in takes if t != val]

        train_probs: dict[int, dict[str, np.ndarray]] = {s: {} for s in SMOOTHS}
        for tid in train_ids:
            frames = decode_gray(takes[tid].video_path, int(models["gru"][0]["size"]))
            parts = {s: [] for s in SMOOTHS}
            for name, (ckpt, m) in models.items():
                raw = dk.predict_video(m, frames, int(ckpt["window"]), int(ckpt["size"]), device)
                for s in SMOOTHS:
                    p = normalize_probs({"v": smooth_prob(raw, s)}, ckpt["normalize"])["v"]
                    parts[s].append(p)
            for s in SMOOTHS:
                train_probs[s][tid] = np.mean(parts[s], axis=0)

        chosen = None
        for s in SMOOTHS:
            for thr in THRS:
                fp = 0
                hit = 0
                for tid in train_ids:
                    tr = triggers(train_probs[s][tid], fps[tid], float(thr))
                    h, f, _, _ = stats(tr, labels[tid], fps[tid], gt[tid])
                    fp += f
                    hit += h
                if fp == 0 and (chosen is None or hit > chosen[0] or (hit == chosen[0] and thr > chosen[2])):
                    chosen = (hit, s, float(thr), fp)
        if chosen is None:
            print(f"| {val} | 无零误报工作点 | - | - | - | - |")
            continue
        train_hit, s, thr, train_fp = chosen
        frames = decode_gray(takes[val].video_path, int(models["gru"][0]["size"]))
        parts = []
        for name, (ckpt, m) in models.items():
            raw = dk.predict_video(m, frames, int(ckpt["window"]), int(ckpt["size"]), device)
            parts.append(normalize_probs({"v": smooth_prob(raw, s)}, ckpt["normalize"])["v"])
        prob = np.mean(parts, axis=0)
        tr = triggers(prob, fps[val], thr)
        hit, fp_lab, unknown, _ = stats(tr, labels[val], fps[val], gt[val])
        print(f"| {val} | smooth={s}, thr={thr:+.1f} | 是（训练命中 {train_hit}） | "
              f"{hit}/{len(gt[val])} | {fp_lab} | {unknown} |")
        total_hit += hit
        total_gt += len(gt[val])
        total_fp += fp_lab
    print(f"\n合计：命中 {total_hit}/{total_gt}，留出集已标注负样本误报 {total_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())