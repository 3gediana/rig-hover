"""在单条视频上跑敲击检测：敲击活动段、每次接触的峰、以及"两次敲击"触发次数。

用法
    python pc/train/detect_knocks.py --model runs_knock/loo4_20261108/model_gru.pt --video take.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_data import decode_gray, window_sample  # noqa: E402
from train_knock import (  # noqa: E402
    build_model,
    normalize_probs,
    peak_indices,
    runs_above,
    smooth_prob,
)


def video_fps(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.release()
    return float(fps)


@torch.no_grad()
def predict_video(model, frames: np.ndarray, window: int, size: int, device="cpu") -> np.ndarray:
    n = int(frames.shape[0])
    prob = np.zeros(n, dtype=np.float32)
    dummy = np.zeros(n, dtype=np.int64)
    for start in range(0, n, window):
        x, _ = window_sample(frames, dummy, start, window, size)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
        p = torch.sigmoid(logits).reshape(-1).cpu().numpy()
        take = min(window, n - start)
        prob[start:start + take] = p[:take]
    return prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    ap.add_argument("--min-gap-s", type=float, default=0.25, help="两次敲击峰的最小间隔（秒）")
    ap.add_argument("--pair-window-s", type=float, default=2.0, help="两次敲击算一组的最长时间（秒）")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    video = Path(args.video)
    frames = decode_gray(video, int(ckpt["size"]))
    fps = video_fps(video)
    prob = predict_video(model, frames, int(ckpt["window"]), int(ckpt["size"]), device)
    prob = smooth_prob(prob, int(ckpt["smooth_frames"]))
    if ckpt.get("normalize", "none") != "none":
        prob = normalize_probs({"v": prob}, ckpt["normalize"])["v"]
    threshold = float(ckpt["threshold"])
    runs = runs_above(prob, threshold, int(ckpt["min_event_frames"]), int(ckpt["max_gap_frames"]))
    min_gap = max(1, int(round(args.min_gap_s * fps)))
    peaks = peak_indices(prob, threshold, min_gap=max(int(ckpt.get("peak_min_gap", 8)), min_gap))
    pair_win = args.pair_window_s * fps
    triggers = sum(1 for a, b in zip(peaks, peaks[1:]) if min_gap <= b - a <= pair_win)

    out = {
        "video": str(video),
        "fps": round(fps, 3),
        "frames": int(len(prob)),
        "threshold": threshold,
        "events": [{"start_s": round(a / fps, 2), "end_s": round(b / fps, 2)} for a, b in runs],
        "peaks_s": [round(i / fps, 2) for i in peaks],
        "two_knock_triggers": triggers,
        "mean_prob": round(float(prob.mean()), 4),
        "max_prob": round(float(prob.max()), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())