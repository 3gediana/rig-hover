# -*- coding: utf-8 -*-
"""_latency_scene.py -- 场景 v2 的感知延迟拆解，以及"砍掉特征抽取"能省多少。

用户要求"比较实时地跟随人脸或拳头"。这条链的延迟在 `RESULTS.md` §9 里拆过一次
（端到端 121.5 ms，其中 capture 51.6 / YOLO 12.3 / feat 56.3）。现在的情形变了：
**策略已经不用那 64 维激活特征了**，如果特征抽取只是为了填 `feat_64`，那 56.3 ms
就是纯浪费。

本脚本把每一段单独测出来：
  * 纯检测前向（ultralytics predict，不挂 hook）
  * 挂 hook 的 predict_and_features（当前路径）
  * FeatureAdapter 的 548→64
  * 人脸/拳头/敲击信号层（纯 numpy）
并给出"不用特征"时的单帧预算。

Run:  python _latency_scene.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = os.path.abspath(os.path.join(HERE, "..", "perception"))
sys.path.insert(0, PERCEPTION)
sys.path.insert(0, HERE)

FRAME = np.full((240, 240, 3), 110, np.uint8)
REPEAT = 40


def load_real_frame():
    """用一张真实的人像照片，避免空白帧让检测后处理变便宜而低估延迟。"""
    try:
        import csv
        import random
        manifest = os.path.join(os.path.dirname(PERCEPTION), "gesture", "data",
                                "manifest.csv")
        if not os.path.isfile(manifest):
            return None, None
        rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
        random.Random(0).shuffle(rows)
        for r in rows[:50]:
            img = cv2.imread(r["path"])
            if img is None:
                continue
            h, w = img.shape[:2]
            s = 240.0 / max(h, w)
            img = cv2.resize(img, (max(int(w * s), 8), max(int(h * s), 8)),
                             interpolation=cv2.INTER_AREA)
            canvas = np.full((240, 240, 3), 110, np.uint8)
            y0, x0 = (240 - img.shape[0]) // 2, (240 - img.shape[1]) // 2
            canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
            return canvas, r["path"]
    except Exception:  # noqa: BLE001 - 基准脚本，拿不到真实图就退回灰帧
        return None, None
    return None, None


def bench(fn, warmup=10, n=REPEAT):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts), max(ts)


def main() -> int:
    import torch
    torch.cuda.synchronize()

    from run_perception import PerceptionPipeline  # noqa: E402
    import scene_signals as ss  # noqa: E402

    pipe = PerceptionPipeline(verbose=False)
    ex = pipe.extractor
    dev = pipe.device

    global FRAME
    real, src = load_real_frame()
    if real is not None:
        FRAME = real
        print(f"real frame from {os.path.basename(src)}")
    else:
        print("WARNING: 没有真实图，用空白灰帧（会低估检测后处理开销）")
    print(f"device={dev}  frame={FRAME.shape}")
    print(f"repeat={REPEAT}, 报中位/最小/最大 ms\n")

    def sync():
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()

    # 1) 纯检测前向：不挂特征 hook
    def pure_predict():
        with torch.inference_mode():
            ex.model.predict(source=FRAME, imgsz=320, conf=0.25,
                             device=dev, verbose=False)
        sync()

    # 2) 当前路径：同一次前向 + hook 抓 548 维特征
    def with_feats():
        ex.predict_and_features(FRAME, conf=0.25)
        sync()

    # 3) 适配器 548 -> 64
    x = torch.zeros(548, device=dev)
    def adapt():
        with torch.no_grad():
            pipe.adapter(x)
        sync()

    # 4) 信号层（纯 numpy，合成关键点）
    kxy, kconf = np.zeros((17, 2), np.float32), np.full(17, 0.9, np.float32)
    kxy[0] = (120, 70); kxy[1] = (112, 68); kxy[2] = (128, 68)
    def signals():
        ss.face_signals(kxy, kconf, 240, 240)
        ss.hand_signals(kxy, kconf, 240, 240)

    rows = []
    for name, fn in (("纯检测前向（无 hook）", pure_predict),
                     ("检测+特征 hook（当前）", with_feats),
                     ("FeatureAdapter 548→64", adapt),
                     ("信号层（纯 numpy）", signals)):
        med, lo, hi = bench(fn)
        rows.append((name, med, lo, hi))
        print(f"  {name:<26}{med:8.2f} ms   (min {lo:.2f}  max {hi:.2f})")

    pure = next(r for r in rows if r[0].startswith("纯检测"))[1]
    full = next(r for r in rows if r[0].startswith("检测+特征"))[1]
    save = full - pure
    print(f"\n  砍掉特征抽取可省：{save:.2f} ms/帧"
          f"（{full:.2f} -> {pure:.2f}，{full / max(pure, 1e-6):.2f}×）")

    print("\n单帧感知预算（不含 HTTP 取流）")
    for label, base in (("当前（要 64 维特征）", full),
                        ("不需要特征（场景 v2）", pure)):
        print(f"  {label:<24}{base:7.2f} ms  ≈ {1000/max(base,1e-6):5.1f} fps")

    print("\n端到端（叠加 RESULTS.md §9 实测的 HTTP 取流 51.6 ms）")
    for label, base in (("当前（要 64 维特征）", full),
                        ("不需要特征（场景 v2）", pure)):
        tot = base + 51.6
        print(f"  {label:<24}{tot:7.2f} ms  ≈ {1000/max(tot,1e-6):5.1f} fps")
    print("\n注：HTTP 取流那一块还能用持久 MJPEG 再省约 40 ms（见 RESULTS.md §9.2），")
    print("    换掉之后场景 v2 的感知到指令延迟可以进到 30 ms 量级。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
