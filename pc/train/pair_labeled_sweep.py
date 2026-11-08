"""按“已标注负样本”口径重算两敲触发指标：未标注帧不算误报。"""
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


def labeled_metrics(triggers, prob, fps, labels, gt, duration_s):
    n = len(labels)
    tp = fp = 0
    for t in triggers:
        i = min(max(int(round(t * fps)), 0), n - 1)
        if labels[i] == KNOCK:
            tp += 1
        elif labels[i] != IGNORE:
            fp += 1  # 明确标了“非敲击”的帧才算误报
    hit = sum(1 for a, b in gt if any(a - 0.5 <= t <= b + 0.5 for t in triggers))
    minutes = max(duration_s, 1e-6) / 60.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = hit / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "hit": hit, "n_gt": len(gt), "P": prec, "R": rec, "F1": f1,
            "FA/min": fp / minutes}


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    gt, labels, targets = {}, {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]
        labels[tid] = frame_labels(iv, take.video_frames, take.fps)
        targets[tid] = np.where(labels[tid] == IGNORE, np.nan, (labels[tid] != KNOCK).astype(np.float32))

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
            print("loaded", name, val)

    print("\n| thr | isolation | P | R | F1 | FA/min | tp | fp | hit |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for thr in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        for iso in (0.0, 0.5):
            agg = {"tp": 0, "fp": 0, "hit": 0, "n_gt": 0}
            for val in takes:
                ps = [cache[(n, val)] for n in RUNS if (n, val) in cache]
                if not ps:
                    continue
                ens = np.mean(ps, axis=0)
                fps = float(video_fps(takes[val].video_path))
                peaks = peak_indices(ens, thr, 8)
                times = [i / fps for i in peaks]
                trig = []
                for k, (a, b) in enumerate(zip(times, times[1:])):
                    if not (0.25 <= b - a <= 1.5):
                        continue
                    if iso > 0:
                        others = times[:k] + times[k + 2:]
                        if any(a - iso <= t <= b + iso for t in others):
                            continue
                    trig.append(b)
                m = labeled_metrics(trig, ens, fps, labels[val], gt[val], takes[val].duration_s)
                agg["tp"] += m["tp"]
                agg["fp"] += m["fp"]
                agg["hit"] += m["hit"]
                agg["n_gt"] += m["n_gt"]
            prec = agg["tp"] / (agg["tp"] + agg["fp"]) if agg["tp"] + agg["fp"] else 0.0
            rec = agg["hit"] / agg["n_gt"] if agg["n_gt"] else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            fa = agg["fp"] / (sum(t.duration_s for t in takes.values()) / 60.0)
            rows.append({"thr": thr, "iso": iso, "P": round(prec, 3), "R": round(rec, 3),
                         "F1": round(f1, 3), "FA/min": round(fa, 2), **agg})
            print(f"| {thr} | {iso} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {fa:.2f} | {agg['tp']} | {agg['fp']} | {agg['hit']}/46 |")
    (ROOT / "runs_knock" / "pair_labeled_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())