"""把不同留一折模型（gru/tcn/bigru）在同一 val take 上做概率平均，看集成能否提升。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_knocks import predict_video, video_fps  # noqa: E402
from interval_data import (  # noqa: E402
    IGNORE,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
)
from train_knock import build_model, normalize_probs, pair_metrics, smooth_prob  # noqa: E402

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
RUNS = {
    "gru": ROOT / "runs_knock" / "loo8_20261108" / "folds",
    "tcn": ROOT / "runs_knock" / "loo9_20261108" / "folds",
    "bigru": ROOT / "runs_knock" / "loo9_20261108" / "folds",
}


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    gt, targets = {}, {}
    for tid, take in takes.items():
        iv = load_intervals(take, id_to_name)
        gt[tid] = [(i.start_s, i.end_s) for i in iv if i.label == "knock"]
        labels = frame_labels(iv, take.video_frames, take.fps)
        targets[tid] = np.where(labels == IGNORE, np.nan, (labels == 0).astype(np.float32))

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
            cache[(name, val)] = (p, float(ckpt["threshold"]), take.duration_s)
            print("loaded", name, val)

    print("\n| val take | gru AUC | tcn AUC | bigru AUC | ensemble AUC |")
    print("|---|---|---|---|---|")
    for val in takes:
        tgt = targets[val]
        mask = ~np.isnan(tgt)
        aucs = {}
        ps = []
        for name in RUNS:
            if (name, val) not in cache:
                continue
            p = cache[(name, val)][0]
            aucs[name] = roc_auc_score((tgt[mask] >= 0.5).astype(int), p[mask]) if len(set(tgt[mask])) > 1 else 0.0
            ps.append(p)
        ens = np.mean(ps, axis=0)
        f1s = []
        for thr in np.arange(-1.0, 2.01, 0.1):
            pred = ens >= thr
            t = (tgt[mask] >= 0.5).astype(int)
            pr = pred[mask]
            tp = int(((pr == 1) & (t == 1)).sum())
            fp = int(((pr == 1) & (t == 0)).sum())
            fn = int(((pr == 0) & (t == 1)).sum())
            f1s.append((2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0, thr))
        f1, thr = max(f1s)
        ea = roc_auc_score((tgt[mask] >= 0.5).astype(int), ens[mask])
        print(f"| {val[-6:]} | {aucs.get('gru', 0):.3f} | {aucs.get('tcn', 0):.3f} | "
              f"{aucs.get('bigru', 0):.3f} | {ea:.3f} (F1 {f1:.3f} @thr {thr:.2f}) |")

    print("\n两敲触发扫参（ensemble, isolation=0）：")
    rows = []
    for thr in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        tp = fp = hit = n_gt = 0
        for val in takes:
            ps = [cache[(n, val)][0] for n in RUNS if (n, val) in cache]
            if not ps:
                continue
            ens = np.mean(ps, axis=0)
            m = pair_metrics(ens, float(takes[val].fps), thr, gt[val], takes[val].duration_s,
                             min_gap_s=0.25, max_gap_s=1.5, margin_s=0.5, peak_min_gap=8, isolation_s=0.0)
            tp += m["tp"]
            fp += m["fp"]
            hit += int(round(m["hit_rate"] * len(gt[val])))
            n_gt += len(gt[val])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = hit / n_gt if n_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fa = fp / (sum(t.duration_s for t in takes.values()) / 60.0)
        rows.append({"thr": thr, "P": round(prec, 3), "R": round(rec, 3), "F1": round(f1, 3), "FA/min": round(fa, 1)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())