"""按已标注负样本口径，扫描“峰间回落”过滤对两敲触发的影响。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_knocks import predict_video, video_fps  # noqa: E402
from interval_data import (  # noqa: E402
    CLASS_TO_IDX,
    IGNORE,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
)
from train_knock import build_model, normalize_probs, peak_indices, smooth_prob  # noqa: E402

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
RUNS = {
    "gru": ROOT / "runs_knock" / "loo8_20261108" / "folds",
    "tcn": ROOT / "runs_knock" / "loo9_20261108" / "folds",
    "bigru": ROOT / "runs_knock" / "loo9_20261108" / "folds",
}
KNOCK = CLASS_TO_IDX["knock"]


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    gt, labels = {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]
        labels[tid] = frame_labels(iv, take.video_frames, take.fps)

    cache = {}
    for name, folder in RUNS.items():
        for path in sorted(folder.glob(f"*_{name}.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            val = ckpt["val_take"]
            take = takes[val]
            frames = decode_gray(take.video_path, int(ckpt["size"]))
            model = build_model(ckpt["model"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            p = predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]))
            p = smooth_prob(p, int(ckpt["smooth_frames"]))
            p = normalize_probs({"v": p}, ckpt["normalize"])["v"]
            cache[(name, val)] = p

    total_minutes = sum(t.duration_s for t in takes.values()) / 60.0
    print("\n| thr | dip | P | R | F1 | FA/min | hit | tp | fp |")
    print("|---|---|---|---|---|---|---|---|---|")
    for thr in (-0.2, 0.0, 0.2, 0.4):
        for dip in (0.0, 0.5, 1.0, 1.5):
            tp = fp = hit = 0
            for val in takes:
                ens = np.mean([cache[(n, val)] for n in RUNS if (n, val) in cache], axis=0)
                fps = float(video_fps(takes[val].video_path))
                times = [i / fps for i in peak_indices(ens, thr, 8)]
                trig = []
                for a, b in zip(times, times[1:]):
                    if not (0.25 <= b - a <= 1.5):
                        continue
                    ia, ib = int(a * fps), int(b * fps)
                    if ens[ia:ib].min() > thr - dip:
                        continue
                    trig.append(b)
                for t in trig:
                    i = min(max(int(round(t * fps)), 0), len(labels[val]) - 1)
                    if labels[val][i] == KNOCK:
                        tp += 1
                    elif labels[val][i] != IGNORE:
                        fp += 1
                hit += sum(1 for a, b in gt[val] if any(a - 0.5 <= t <= b + 0.5 for t in trig))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = hit / 46
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            print(f"| {thr} | {dip} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {fp / total_minutes:.2f} | {hit}/46 | {tp} | {fp} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())