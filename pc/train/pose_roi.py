"""离线提取手腕 ROI：用 YOLO26n-pose 找腕点，裁手部窗口存成训练用灰度帧。

用法（在 D:\\robot_work 下）：
    python pc/train/pose_roi.py --take dataset/takes/<take_id>/take.mp4 \
        --out runs_knock/pose_roi/<take_id>.npz --device cuda:0

输出 npz 字段：
    frames   (N, 96, 96) uint8   手腕 ROI 灰度图（训练输入）
    boxes    (N, 4) float32      ROI 在原始 240x240 帧里的像素框
    source   (N,) uint8          0=整帧兜底 1=姿态检测 2=沿用上一帧框
    wrist_conf (N,) float32      腕点置信度
    fps / duration_s / model / imgsz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WRIST_L, WRIST_R = 9, 10
ELBOW_L, ELBOW_R = 7, 8
SHOULDER_L, SHOULDER_R = 5, 6
ROI_SIZE = 96
HOLD_FRAMES = 24  # 检测不到手腕时最多沿用上一帧框多久（约 1.2 s @20fps）


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int):
    x1 = min(max(x1, 0.0), w - 2.0)
    y1 = min(max(y1, 0.0), h - 2.0)
    x2 = min(max(x2, x1 + 2.0), float(w))
    y2 = min(max(y2, y1 + 2.0), float(h))
    return x1, y1, x2, y2


def pick_wrist(kp: np.ndarray, min_conf: float = 0.25):
    """返回 (x, y, conf) —— 取置信度够、位置更低的那只手腕（敲桌子通常是较低的手）。"""
    if kp is None or kp.shape[0] < 11:
        return None
    cands = []
    for idx in (WRIST_L, WRIST_R):
        x, y, c = float(kp[idx, 0]), float(kp[idx, 1]), float(kp[idx, 2])
        if c >= min_conf and x > 0 and y > 0:
            cands.append((y, x, c))
    if not cands:
        return None
    y, x, c = max(cands, key=lambda t: t[0] + 0.001 * t[2])
    return x, y, c


def roi_side_for(kp: np.ndarray, wrist_x: float, wrist_y: float) -> float:
    """由手腕-肘距离估手部窗口大小；退化时用固定 72 px。"""
    best = None
    for wi, ei in ((WRIST_L, ELBOW_L), (WRIST_R, ELBOW_R)):
        if kp is None or kp.shape[0] <= ei:
            continue
        if kp[wi, 2] < 0.25 or kp[ei, 2] < 0.25:
            continue
        d = math.hypot(float(kp[wi, 0] - kp[ei, 0]), float(kp[wi, 1] - kp[ei, 1]))
        if d > 2:
            best = d if best is None else min(best, d)
    side = 72.0 if best is None else 1.25 * best
    return float(min(max(side, 56.0), 112.0))


def extract(video: Path, model_path: Path, out: Path, imgsz: int = 320,
            device: str = "cuda:0", conf: float = 0.25, sheet: bool = True) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames, boxes, sources, confs = [], [], [], []
    last_box = None
    last_seen = -10**9
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        h, w = gray.shape[:2]

        res = model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=imgsz, conf=conf,
                            verbose=False, device=device, classes=[0])
        kp = None
        if res and getattr(res[0], "keypoints", None) is not None and len(res[0].keypoints):
            kxy = res[0].keypoints.xy.cpu().numpy()
            kconf = res[0].keypoints.conf.cpu().numpy()
            bconf = res[0].boxes.conf.cpu().numpy() if res[0].boxes is not None else np.ones(len(kxy))
            if len(kxy):
                order = np.argsort(-bconf)
                kp = np.concatenate([kxy[order[0]], kconf[order[0]][:, None]], axis=1)

        wrist = pick_wrist(kp)
        if wrist is not None:
            wx, wy, wc = wrist
            side = roi_side_for(kp, wx, wy)
            x1, y1, x2, y2 = clip_box(wx - side / 2, wy - side / 2, wx + side / 2, wy + side / 2, w, h)
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            source = 1
            last_box, last_seen = box, idx
        elif last_box is not None and idx - last_seen <= HOLD_FRAMES:
            box, source, wc = last_box, 2, 0.0
        else:
            box, source, wc = np.array([0, 0, w, h], dtype=np.float32), 0, 0.0
            last_box = None

        x1, y1, x2, y2 = box.astype(int)
        crop = gray[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]
        if crop.size == 0:
            crop = gray
        crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA if crop.shape[0] > ROI_SIZE else cv2.INTER_LINEAR)
        frames.append(crop)
        boxes.append(box)
        sources.append(source)
        confs.append(wc)
        idx += 1

    cap.release()
    if not frames:
        raise SystemExit("没有读出任何帧")

    arr = np.stack(frames).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frames=arr,
        boxes=np.stack(boxes),
        source=np.asarray(sources, dtype=np.uint8),
        wrist_conf=np.asarray(confs, dtype=np.float32),
        fps=float(fps if fps > 1 else 20.0),
        duration_s=float(idx / (fps if fps > 1 else 20.0)),
        model=str(model_path),
        imgsz=int(imgsz),
    )
    stats = {
        "video": video.name,
        "frames": int(len(frames)),
        "fps": float(fps),
        "pose_frames": int(sum(1 for s in sources if s == 1)),
        "held_frames": int(sum(1 for s in sources if s == 2)),
        "fallback_frames": int(sum(1 for s in sources if s == 0)),
        "mean_wrist_conf": float(np.mean([c for c in confs if c > 0]) if any(c > 0 for c in confs) else 0.0),
        "out": str(out),
    }
    (out.with_suffix(".json")).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if sheet:
        make_sheet(arr, boxes, sources, out.with_name(out.stem + "_sheet.png"))
    return stats


def make_sheet(frames: np.ndarray, boxes: np.ndarray, sources: np.ndarray, dst: Path, n: int = 16) -> None:
    """随机抽帧拼一张 96px 网格图，确认 ROI 是否真的框住手。"""
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = []
    for i in idxs:
        tile = frames[i]
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        tag = {0: "full", 1: "pose", 2: "hold"}[int(sources[i])]
        cv2.putText(tile, f"{i}:{tag}", (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)]
    cv2.imwrite(str(dst), np.vstack(rows))
    print(f"sheet -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO pose 腕部 ROI 提取")
    ap.add_argument("--take", required=True, help="take.mp4 路径")
    ap.add_argument("--out", required=True, help="输出 npz")
    ap.add_argument("--model", default="pc/perception/assets/yolo26n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    stats = extract(Path(args.take), Path(args.model), Path(args.out),
                    imgsz=args.imgsz, device=args.device, conf=args.conf)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()