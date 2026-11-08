"""在单条视频上跑敲击检测：敲击活动段、接触峰、以及"两下敲完"触发。

支持单模型或集成（多个 checkpoint 概率平均）。

用法
    python pc/train/detect_knocks.py --model runs_knock/deploy2_20261108/model_gru.pt --video take.mp4
    python pc/train/detect_knocks.py --model a.pt,b.pt,c.pt --video take.mp4 --json out.json
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


def find_triggers(prob: np.ndarray, fps: float, threshold: float, peak_min_gap: int,
                  min_gap_s: float, max_gap_s: float, isolation_s: float,
                  cooldown_s: float = 0.0) -> list[dict]:
    peaks = peak_indices(prob, threshold, peak_min_gap)
    times = [i / fps for i in peaks]
    triggers: list[dict] = []
    last_trigger = -1e9
    for k, (a, b) in enumerate(zip(times, times[1:])):
        if not (min_gap_s <= b - a <= max_gap_s):
            continue
        if b - last_trigger < cooldown_s:
            continue
        if isolation_s > 0:
            others = times[:k] + times[k + 2:]
            if any(a - isolation_s <= t <= b + isolation_s for t in others):
                continue
        triggers.append({"first_s": round(a, 2), "second_s": round(b, 2)})
        last_trigger = b
    return triggers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="一个或多个 checkpoint，逗号分隔（集成）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--json", default="", help="把结果写成 JSON")
    ap.add_argument("--pair-threshold", type=float, default=None,
                    help="覆盖两敲峰值阈值（z 空间）")
    ap.add_argument("--min-gap-s", type=float, default=None, help="两敲最小间隔（默认读 checkpoint）")
    ap.add_argument("--pair-window-s", type=float, default=None, help="两敲最大间隔（默认读 checkpoint）")
    ap.add_argument("--isolation-s", type=float, default=None, help="两敲前后安静时间；默认读 checkpoint")
    ap.add_argument("--cooldown-s", type=float, default=0.0,
                    help="触发后冷却秒数：同一段连续敲击只输出第一个触发")
    ap.add_argument("--smooth-frames", type=int, default=None,
                    help="概率平滑帧数；两敲检测建议 3（默认读 checkpoint，通常 20 会把两下糊成一下）")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in args.model.split(",") if p.strip()]
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    first = ckpts[0]
    size, window = int(first["size"]), int(first["window"])
    device = torch.device("cpu")

    video = Path(args.video)
    frames = decode_gray(video, size)
    fps = video_fps(video)

    probs = []
    smooth_frames = int(args.smooth_frames if args.smooth_frames is not None
                        else ckpts[0]["smooth_frames"])
    for ckpt in ckpts:
        if int(ckpt["size"]) != size or int(ckpt["window"]) != window:
            raise SystemExit("checkpoint 的 size/window 不一致，不能集成")
        model = build_model(ckpt["model"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        p = predict_video(model, frames, window, size, device)
        p = smooth_prob(p, smooth_frames)
        p = normalize_probs({"v": p}, ckpt.get("normalize", "none"))["v"]
        probs.append(p.astype(np.float32))
    prob = np.mean(probs, axis=0) if len(probs) > 1 else probs[0]

    threshold = float(np.mean([float(c["threshold"]) for c in ckpts]))
    pair_threshold = float(np.mean([float(c.get("pair_threshold", c["threshold"])) for c in ckpts]))
    if args.pair_threshold is not None:
        pair_threshold = args.pair_threshold
    min_gap_s = float(args.min_gap_s if args.min_gap_s is not None else ckpts[0].get("pair_min_gap_s", 0.25))
    max_gap_s = float(args.pair_window_s if args.pair_window_s is not None else ckpts[0].get("pair_max_gap_s", 1.5))
    isolation_s = float(args.isolation_s if args.isolation_s is not None else ckpts[0].get("pair_isolation_s", 0.0))
    peak_gap = int(ckpts[0].get("peak_min_gap", 8))

    runs = runs_above(prob, threshold, int(first["min_event_frames"]), int(first["max_gap_frames"]))
    triggers = find_triggers(prob, fps, pair_threshold, peak_gap, min_gap_s, max_gap_s, isolation_s, args.cooldown_s)
    peaks = peak_indices(prob, pair_threshold, peak_gap)

    out = {
        "video": str(video),
        "fps": round(fps, 3),
        "frames": int(len(prob)),
        "models": [str(p) for p in paths],
        "threshold": threshold,
        "pair_threshold": pair_threshold,
        "events": [{"start_s": round(a / fps, 2), "end_s": round(b / fps, 2)} for a, b in runs],
        "peaks_s": [round(i / fps, 2) for i in peaks],
        "triggers": triggers,
        "two_knock_triggers": len(triggers),
        "mean_prob": round(float(prob.mean()), 4),
        "max_prob": round(float(prob.max()), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())