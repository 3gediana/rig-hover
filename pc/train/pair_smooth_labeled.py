"""按标注负样本口径，比较不同平滑宽度下的两敲触发工作点。"""
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
    gt, labels, fps = {}, {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]
        labels[tid] = frame_labels(iv, take.video_frames, take.fps)
        fps[tid] = float(video_fps(take.video_path))

    raw = {}
    for name, folder in RUNS.items():
        for path in sorted(folder.glob(f"*_{name}.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            val = ckpt["val_take"]
            frames = decode_gray(takes[val].video_path, int(ckpt["size"]))
            model = build_model(ckpt["model"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            p = predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]))
            p = normalize_probs({"v": p}, ckpt["normalize"])["v"]
            raw.setdefault(val, []).append(p.astype(np.float32))

    total_minutes = sum(t.duration_s for t in takes.values()) / 60.0
    gt_total = sum(len(v) for v in gt.values())
    rows = []
    for smooth in (3, 5, 10, 20):
        ens = {val: np.mean(parts, axis=0) for val, parts in raw.items()}
        ens = {val: smooth_prob(p, smooth) for val, p in ens.items()}
        for thr in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            tp = fp = hit = 0
            for val, prob in ens.items():
                f = fps[val]
                times = [i / f for i in peak_indices(prob, thr, 8)]
                trig = [b for a, b in zip(times, times[1:]) if 0.25 <= b - a <= 1.5]
                for t in trig:
                    i = min(max(int(round(t * f)), 0), len(labels[val]) - 1)
                    if labels[val][i] == KNOCK:
                        tp += 1
                    elif labels[val][i] != IGNORE:
                        fp += 1
                hit += sum(1 for a, b in gt[val] if any(a - 0.5 <= t <= b + 0.5 for t in trig))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = hit / gt_total
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append((smooth, thr, prec, rec, f1, fp / total_minutes, hit, tp, fp))
            print(f"smooth={smooth:2d} thr={thr:+.1f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} "
                  f"FA/min={fp / total_minutes:.2f} hit={hit}/{gt_total} tp={tp} fp={fp}")
    best = max(rows, key=lambda r: r[4])
    print(f"\nbest F1: smooth={best[0]} thr={best[1]} P={best[2]:.3f} R={best[3]:.3f} "
          f"F1={best[4]:.3f} FA/min={best[5]:.2f}")
    (ROOT / "runs_knock" / "pair_smooth_labeled.json").write_text(
        json.dumps([dict(zip(("smooth", "thr", "precision", "recall", "f1", "fa_per_min",
                              "hit", "tp", "fp"), r)) for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())