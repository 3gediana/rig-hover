# -*- coding: utf-8 -*-
"""bench_latency.py — 单帧耗时：关键点提取 与 分类 分开量

机器人主循环里两件事是分开发生的，**必须分开报**，否则调优时不知道该砍哪一段：

* 关键点提取 = YOLO26n-pose 前向（``imgsz=320``）+ 后处理
* 分类 = 骨架特征（纯 numpy）+ 手部裁剪（cv2）+ 12 帧滑窗小网络的单帧前向

用法::

    python src/bench_latency.py --n-pose 120 --n-cls 400
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from gesture_common import (COCO17, build_frame_features,  # noqa: E402
                            hand_crop)

REPORTS = os.path.join(ROOT, "reports")
IMGSZ = 320


def _stats(name, xs):
    xs = np.asarray(xs, dtype=np.float64) * 1e3
    return {"name": name, "n": int(len(xs)),
            "mean_ms": round(float(xs.mean()), 3),
            "median_ms": round(float(np.median(xs)), 3),
            "p90_ms": round(float(np.percentile(xs, 90)), 3),
            "min_ms": round(float(xs.min()), 3),
            "max_ms": round(float(xs.max()), 3)}


def bench_pose(n=120, device="cuda:0"):
    """YOLO26n-pose 单帧/多帧耗时（含 CUDA 同步，不含读盘）。"""
    import cv2
    import torch
    from ultralytics import YOLO

    manifest = os.path.join(ROOT, "data", "manifest.csv")
    paths = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i and i % 97 == 0:
                paths.append(line.split(",")[0])
            if len(paths) >= n:
                break
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    print("[bench] 关键点提取：%d 张图" % len(frames), flush=True)

    model = YOLO("yolo26n-pose.pt")
    out = {"imgsz": IMGSZ, "device": device, "n_images": len(frames)}

    for bs in (1, 8):
        for _ in range(3):                       # 预热（cuDNN autotune）
            model.predict(frames[:bs], imgsz=IMGSZ, conf=0.25, verbose=False,
                          device=device)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        lat = []
        for s in range(0, len(frames), bs):
            chunk = frames[s:s + bs]
            if len(chunk) < bs:
                break
            t0 = time.perf_counter()
            model.predict(chunk, imgsz=IMGSZ, conf=0.25, verbose=False,
                          device=device)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) / bs)
        key = "pose_bs%d_%s" % (bs, device.split(":")[0])
        out[key] = _stats(key, lat)
        print("  %-18s 单帧 %.2f ms (p90 %.2f)"
              % (key, out[key]["mean_ms"], out[key]["p90_ms"]), flush=True)
    return out


def bench_classifier(n=400, device="cuda:0"):
    """分类器单帧耗时：骨架特征 / 手部裁剪 / 网络前向，分别计时。"""
    import cv2
    import torch

    from gesture_classifier import MODEL_PATH, GestureClassifier

    clf = GestureClassifier.load(MODEL_PATH, device=device)
    z = np.load(os.path.join(ROOT, "data", "synth", "test.npz"))
    feat = z["feat"].astype(np.float32)
    crop = z["crop"]
    n = min(n, len(feat))

    # 用真实 HaGRID 图做手部裁剪的输入（像素关键点由归一化骨架反推）
    frame = cv2.imread(_first_image())
    h, w = frame.shape[:2]
    rng = np.random.default_rng(0)

    t_feat, t_crop, t_net, t_all = [], [], [], []
    kp_cache = []
    for i in range(n):
        kp_rel = (feat[i, -1, :34].reshape(17, 2) * 3.0).astype(np.float32)
        kp_px = kp_rel * 60.0 + np.array([w / 2.0, h / 2.0], np.float32)
        conf = np.clip(feat[i, -1, 34:51], 0, 1)
        kp_cache.append((kp_px, conf))

    # 1) 骨架特征（纯 numpy）
    for kp_px, conf in kp_cache:
        t0 = time.perf_counter()
        build_frame_features(kp_px, conf, img_wh=(w, h))
        t_feat.append(time.perf_counter() - t0)

    # 2) 手部裁剪（cv2 读图后裁剪 + 缩放）
    for kp_px, conf in kp_cache[:max(50, n // 4)]:
        t0 = time.perf_counter()
        hand_crop(frame, kp_px, conf)
        t_crop.append(time.perf_counter() - t0)

    # 3) 网络前向（逐帧，滑窗已填充；GPU 需要同步）
    clf.reset()
    clf.push_feat(feat[0, -1], crop[0, -1])
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    for i in range(n):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        clf.push_feat(feat[i, -1], crop[i, -1])
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_net.append(time.perf_counter() - t0)

    # 4) 端到端单帧：像素关键点 → 特征 → 裁剪 → 前向
    for i in range(n):
        kp_px, conf = kp_cache[i]
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        clf.push_kp(kp_px, conf, frame_bgr=frame)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t_all.append(time.perf_counter() - t0)

    raw = {"feat": t_feat, "crop": t_crop, "net": t_net, "end_to_end": t_all}
    out = {}
    for k, v in raw.items():
        out[k] = _stats("cls_" + k + "_" + device.split(":")[0], v)
        print("  %-26s %.3f ms (p90 %.3f)"
              % (out[k]["name"], out[k]["mean_ms"], out[k]["p90_ms"]),
              flush=True)
    out["n_params"] = int(sum(p.numel() for p in clf.model.parameters()))
    out["device"] = device
    return out


def _first_image():
    manifest = os.path.join(ROOT, "data", "manifest.csv")
    with open(manifest, "r", encoding="utf-8") as fh:
        fh.readline()
        return fh.readline().split(",")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pose", type=int, default=120)
    ap.add_argument("--n-cls", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-pose", action="store_true")
    args = ap.parse_args()

    out = {"imgsz": IMGSZ, "note": "单帧耗时，含 CUDA 同步；不含读盘/解码"}
    if not args.skip_pose:
        out["pose"] = bench_pose(args.n_pose, args.device)
    out["classifier_gpu"] = bench_classifier(args.n_cls, args.device)
    out["classifier_cpu"] = bench_classifier(max(60, args.n_cls // 4), "cpu")

    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "latency.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    g = out["classifier_gpu"]
    c = out["classifier_cpu"]
    print("\n[bench] 关键点提取 %.2f ms + 分类 %.2f ms（GPU）= 单帧 %.2f ms"
          % (out["pose"]["pose_bs1_cuda"]["mean_ms"],
             g["end_to_end"]["mean_ms"],
             out["pose"]["pose_bs1_cuda"]["mean_ms"] + g["end_to_end"]["mean_ms"]),
          flush=True)
    print("[bench] 纯 CPU 分类 %.2f ms" % c["end_to_end"]["mean_ms"],
          flush=True)
    print("[bench] → reports/latency.json", flush=True)


if __name__ == "__main__":
    main()
