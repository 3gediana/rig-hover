"""离线扫描两敲触发后处理：平滑窗口、阈值、峰间低点要求。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_knocks import predict_video, video_fps  # noqa: E402
from interval_data import decode_gray, load_intervals, load_labels, load_takes  # noqa: E402
from train_knock import build_model, normalize_probs, peak_indices, smooth_prob  # noqa: E402

ROOT = Path(r"D:\robot_work")
DATASET = ROOT / "dataset"
FOLDS = ROOT / "runs_knock" / "loo8_20261108" / "folds"


def micro(triggers_by, gt, durations, n_gt):
    tp = fp = hit = 0
    for val, trig in triggers_by.items():
        g = gt.get(val, [])
        in_gt = [any(a - 0.5 <= t <= b + 0.5 for a, b in g) for t in trig]
        tp += int(np.sum(in_gt))
        fp += int(len(trig) - np.sum(in_gt))
        hit += sum(1 for a, b in g if any(a - 0.5 <= t <= b + 0.5 for t in trig))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = hit / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fa_min = fp / (sum(durations.values()) / 60.0)
    return precision, recall, f1, fa_min, tp, fp, hit


def main() -> int:
    id_to_name = load_labels(DATASET)
    takes = {t.take_id: t for t in load_takes(DATASET)}
    gt = {tid: [(iv.start_s, iv.end_s) for iv in load_intervals(t, id_to_name) if iv.label == "knock"]
          for tid, t in takes.items()}
    n_gt = sum(len(v) for v in gt.values())

    raw = {}
    for path in sorted(FOLDS.glob("*_gru.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val = ckpt["val_take"]
        take = takes[val]
        frames = decode_gray(take.video_path, int(ckpt["size"]))
        model = build_model(ckpt["model"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        p = predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]))
        raw[val] = (p, float(video_fps(take.video_path)), take.duration_s)
        print("loaded", val)

    rows = []
    for smooth in (1, 3, 5, 10, 20):
        probs = {}
        for val, (p, fps, dur) in raw.items():
            q = smooth_prob(p, smooth) if smooth > 1 else p
            mu, sd = float(q.mean()), float(q.std()) or 1e-6
            probs[val] = ((q - mu) / sd, fps, dur)
        for thr in (-0.2, 0.0, 0.3, 0.6, 0.9, 1.2):
            for gap_drop in (None, 1.0, 1.5):
                trig_by = {}
                fps_map = {}
                dur_map = {}
                for val, (q, fps, dur) in probs.items():
                    peaks = peak_indices(q, thr, 8)
                    times = [i / fps for i in peaks]
                    trig = []
                    for k, (a, b) in enumerate(zip(times, times[1:])):
                        if not (0.25 <= b - a <= 1.5):
                            continue
                        if gap_drop is not None:
                            ia, ib = int(a * fps), int(b * fps)
                            if q[ia:ib].min() > thr - gap_drop:
                                continue
                        trig.append(b)
                    trig_by[val] = trig
                    fps_map[val] = fps
                p_, r_, f1, fa_min, tp, fp, hit = micro(
                    trig_by, gt, {k: v[2] for k, v in probs.items()}, n_gt)
                rows.append({"smooth": smooth, "thr": thr, "gap_drop": gap_drop,
                             "P": round(p_, 3), "R": round(r_, 3), "F1": round(f1, 3),
                             "FA/min": round(fa_min, 1), "tp": tp, "fp": fp, "hit": hit})
    rows.sort(key=lambda r: -r["F1"])
    print("\n| smooth | thr | gap_drop | P | R | F1 | FA/min |")
    print("|---|---|---|---|---|---|---|")
    for r in rows[:25]:
        print(f"| {r['smooth']} | {r['thr']} | {r['gap_drop']} | {r['P']} | {r['R']} | {r['F1']} | {r['FA/min']} |")
    (FOLDS.parent / "pair_sweep2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())