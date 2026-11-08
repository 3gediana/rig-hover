"""离线扫参：用留一折模型看“两敲触发”的阈值/隔离时间在整个留出集上的表现。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_knocks import predict_video, video_fps  # noqa: E402
from interval_data import decode_gray, load_intervals, load_labels, load_takes  # noqa: E402
from train_knock import normalize_probs, pair_metrics, peak_indices, smooth_prob  # noqa: E402

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
FOLDS = ROOT / "runs_knock" / "loo8_20261108" / "folds"


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    gt = {}
    for tid, take in takes.items():
        gt[tid] = [(iv.start_s, iv.end_s) for iv in load_intervals(take, id_to_name) if iv.label == "knock"]

    models = sorted(FOLDS.glob("*_gru.pt"))
    probs = {}
    for path in models:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val = ckpt["val_take"]
        take = takes[val]
        frames = decode_gray(take.video_path, int(ckpt["size"]))
        from train_knock import build_model

        model = build_model(ckpt["model"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        p = predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]))
        p = smooth_prob(p, int(ckpt["smooth_frames"]))
        if ckpt.get("normalize", "none") != "none":
            p = normalize_probs({"v": p}, ckpt["normalize"])["v"]
        probs[val] = (p, float(ckpt["threshold"]), float(video_fps(take.video_path)), take.duration_s)
        print(f"loaded {path.name} val={val} thr={ckpt['threshold']:.2f}")

    rows = []
    for rel in (0.0, 0.2, 0.4, 0.6):
        for iso in (0.0, 0.5, 1.0, 1.5):
            tp = fp = hit = n_gt = 0
            for val, (p, thr, fps, dur) in probs.items():
                m = pair_metrics(p, fps, thr + rel, gt.get(val, []), dur,
                                 min_gap_s=0.25, max_gap_s=1.5, margin_s=0.5,
                                 peak_min_gap=8, isolation_s=iso)
                tp += m["tp"]
                fp += m["fp"]
                hit += int(round(m["hit_rate"] * len(gt.get(val, []))))
                n_gt += len(gt.get(val, []))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = hit / n_gt if n_gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"threshold_rel": rel, "isolation_s": iso, "tp": tp, "fp": fp,
                         "gt_intervals_hit": hit, "n_gt": n_gt,
                         "precision": round(precision, 3), "recall": round(recall, 3),
                         "pair_f1": round(f1, 3)})
    rows.sort(key=lambda r: -r["pair_f1"])
    print("\n| thr_rel | isolation_s | tp | fp | hit | n_gt | P | R | pairF1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| +{r['threshold_rel']:.1f} | {r['isolation_s']:.1f} | {r['tp']} | {r['fp']} | "
              f"{r['gt_intervals_hit']} | {r['n_gt']} | {r['precision']} | {r['recall']} | {r['pair_f1']} |")
    Path(FOLDS.parent / "pair_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())