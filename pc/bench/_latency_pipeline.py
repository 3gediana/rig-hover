# -*- coding: utf-8 -*-
"""_latency_pipeline.py -- 整条 PerceptionPipeline 每帧到底花多久，钱花在哪一段。

`_latency_scene.py` 只测了"检测前向"这一段（7 ms），但部署走的是完整管线：
检测 → 特征适配 → 手势基元 → 手势分类器（12 帧 CNN + 手部外观）→ 事件机。
手势分类器也是一次 GPU 前向，很可能才是大头——所以不能拿前向的数字当全链路。

管线自己已经埋了 t_forward_ms / t_adapter_ms / t_gesture_ms / t_total_ms，
这里跑真实照片统计它们的分布。

Run:  python _latency_pipeline.py
"""
from __future__ import annotations

import os
import random
import statistics
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = os.path.abspath(os.path.join(HERE, "..", "perception"))
sys.path.insert(0, PERCEPTION)
sys.path.insert(0, HERE)

import real_image_gap as RIG  # noqa: E402


def main() -> int:
    rows = RIG.load_rows()
    random.Random(0).shuffle(rows)
    imgs = []
    for r in rows[:30]:
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
        imgs.append(canvas)
    print(f"{len(imgs)} 张 240x240 真实照片帧")

    from run_perception import PerceptionPipeline  # noqa: E402
    pipe = PerceptionPipeline(verbose=False)

    # 预热（第一次有 cuDNN autotune）
    for img in imgs[:12]:
        pipe(img)

    keys = ("t_forward_ms", "t_adapter_ms", "t_gesture_ms", "t_total_ms")
    vals = {k: [] for k in keys}
    for img in imgs:
        out = pipe(img)
        for k in keys:
            vals[k].append(float(out.debug[k]))

    print(f"\n{'环节':<22}{'中位':>9}{'均值':>9}{'p90':>9}{'最大':>9}   占比")
    print("-" * 74)
    tot = statistics.median(vals["t_total_ms"])
    for k, name in (("t_forward_ms", "YOLO 检测前向"),
                    ("t_adapter_ms", "特征适配 548→64"),
                    ("t_gesture_ms", "手势分类器+事件机"),
                    ("t_total_ms", "合计（含信号层）")):
        v = sorted(vals[k])
        med = statistics.median(v)
        p90 = v[int(0.9 * (len(v) - 1))]
        share = f"{100*med/tot:5.1f}%" if k != "t_total_ms" else "  100%"
        print(f"{name:<22}{med:>8.2f}{statistics.mean(v):>9.2f}"
              f"{p90:>9.2f}{v[-1]:>9.2f}   {share}")

    other = tot - (statistics.median(vals["t_forward_ms"])
                   + statistics.median(vals["t_adapter_ms"])
                   + statistics.median(vals["t_gesture_ms"]))
    print(f"{'(其余：选人/信号层)':<22}{other:>8.2f}")
    print("-" * 74)
    print(f"单帧感知：{tot:.2f} ms  ≈ {1000/max(tot,1e-6):.1f} fps")
    print(f"\n叠加 RESULTS.md §9 实测的网络取流 51.6 ms 后：")
    print(f"  端到端 {tot + 51.6:.2f} ms  ≈ {1000/max(tot+51.6,1e-6):.1f} fps")
    print(f"其中网络占 {100*51.6/(tot+51.6):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
