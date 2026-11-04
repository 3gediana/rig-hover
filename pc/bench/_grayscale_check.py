# -*- coding: utf-8 -*-
"""_grayscale_check.py -- 把输入改成灰度，精度掉多少？

用户提议：这个任务可能对颜色不敏感，改成灰度能更快。这条很有依据：

* 机器人上是 **GC0308 + 软件 JPEG 编码**，RGB565 是 2 字节/像素，
  灰度是 1 字节/像素——**字节交换循环直接消失**，JPEG 也从 3 个色彩平面降到 1 个。
* 手势分类器的**外观分支本来就是单通道**（`crop (B,T,1,32,32)`，见
  `pc/gesture/src/model.py`），它本来就没用颜色。
* 只有 YOLO26n-pose 吃 RGB。

所以真正的问题是：**把灰度复制成 3 通道喂给 YOLO，检测和关键点会掉多少？**
这个可以完全离线测，不需要机器人。

测法：同一批真实照片，跑两遍完整部署链（原图 / 灰度转三通道），
逐帧比 seen、person_h、theta_img、腕部关键点、手势分类结果。

注意：这里量的是**精度**。「灰度能让编码快多少」在 PC 上量不准
（机器人用的是 esp_new_jpeg，不是 OpenCV），那部分只能上真机量。

Run:  python _grayscale_check.py --n 60
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = os.path.abspath(os.path.join(HERE, "..", "perception"))
sys.path.insert(0, HERE)
sys.path.insert(0, PERCEPTION)

import real_image_gap as RIG  # noqa: E402


def to_robot_frame(img, size=240):
    """把 HaGRID 照片放成机器人相机的 240x240 视野。"""
    h, w = img.shape[:2]
    s = float(size) / max(h, w)
    small = cv2.resize(img, (max(int(w * s), 8), max(int(h * s), 8)),
                       interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 110, np.uint8)
    y0, x0 = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tile", type=int, default=12)
    a = ap.parse_args()

    rows = RIG.load_rows()
    random.Random(0).shuffle(rows)
    rows = rows[:a.n]
    print(f"{len(rows)} 张真实照片")

    from run_perception import PerceptionPipeline  # noqa: E402
    pipe_bgr = PerceptionPipeline(verbose=False)
    pipe_gray = PerceptionPipeline(verbose=False)   # 独立实例，避免事件机状态串味

    stats = []
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        frame = to_robot_frame(img)
        gray3 = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
        out_b = out_g = None
        for _ in range(a.tile):
            out_b = pipe_bgr(frame)
            out_g = pipe_gray(gray3)
        stats.append({
            "cls": r["hagrid_class"],
            "want": RIG.HAGRID_TO_ID.get(r["hagrid_class"], -1),
            "seen_b": out_b.seen, "seen_g": out_g.seen,
            "h_b": out_b.person_h, "h_g": out_g.person_h,
            "theta_b": out_b.theta_img, "theta_g": out_g.theta_img,
            "face_ok_b": out_b.face_valid, "face_ok_g": out_g.face_valid,
            "wrist_b": int(out_b.hand_side != ""), "wrist_g": int(out_g.hand_side != ""),
            "g_raw_b": out_b.gesture_raw_id, "g_raw_g": out_g.gesture_raw_id,
            "g_conf_b": out_b.gesture_raw_conf, "g_conf_g": out_g.gesture_raw_conf,
            "knock_b": out_b.knock_fired, "knock_g": out_g.knock_fired,
        })

    n = len(stats)
    if n == 0:
        print("没有可用样本")
        return 2

    def rate(key):
        return sum(s[key] for s in stats) / n

    print("\n" + "=" * 74)
    print(f"{'指标':<30}{'BGR':>12}{'灰度':>12}{'差异':>12}")
    print("-" * 74)
    for label, kb, kg, pct in (
            ("检出人的帧比例", "seen_b", "seen_g", True),
            ("人脸关键点可用率", "face_ok_b", "face_ok_g", True),
            ("腕部关键点可用率", "wrist_b", "wrist_g", True)):
        b, g = rate(kb), rate(kg)
        print(f"{label:<30}{100*b:>11.1f}%{100*g:>11.1f}%{100*(g-b):>+11.1f}pp")

    hb = np.mean([s["h_b"] for s in stats])
    hg = np.mean([s["h_g"] for s in stats])
    print(f"{'person_h 均值':<30}{hb:>12.3f}{hg:>12.3f}{hg-hb:>+12.3f}")

    th_b = np.array([s["theta_b"] for s in stats])
    th_g = np.array([s["theta_g"] for s in stats])
    mae = float(np.abs(th_b - th_g).mean())
    print(f"{'theta_img 平均绝对差(度)':<30}{'':>12}{mae:>12.2f}")

    agree = sum(s["g_raw_b"] == s["g_raw_g"] for s in stats) / n
    print(f"{'手势逐帧分类一致率':<30}{'':>12}{100*agree:>11.1f}%")

    # 关键：两种输入互相"不一致"不等于灰度更差。要比的是**对真值的准确率**。
    acc_b = sum(s["g_raw_b"] == s["want"] for s in stats) / n
    acc_g = sum(s["g_raw_g"] == s["want"] for s in stats) / n
    print(f"{'手势对真值准确率':<30}{100*acc_b:>11.1f}%{100*acc_g:>11.1f}%"
          f"{100*(acc_g-acc_b):>+11.1f}pp")

    conf_b = np.mean([s["g_conf_b"] for s in stats])
    conf_g = np.mean([s["g_conf_g"] for s in stats])
    print(f"{'手势置信度均值':<30}{conf_b:>12.3f}{conf_g:>12.3f}{conf_g-conf_b:>+12.3f}")
    print("=" * 74)

    print("\n按类别拆开（对真值准确率）：")
    print(f"  {'类':<12}{'n':>5}{'BGR':>9}{'灰度':>9}{'差异':>9}")
    by_cls: dict[str, list] = {}
    for s in stats:
        by_cls.setdefault(s["cls"], []).append(s)
    for cls, ss in sorted(by_cls.items()):
        m = len(ss)
        ab = sum(x["g_raw_b"] == x["want"] for x in ss) / m
        ag = sum(x["g_raw_g"] == x["want"] for x in ss) / m
        print(f"  {cls:<12}{m:>5}{100*ab:>8.1f}%{100*ag:>8.1f}%"
              f"{100*(ag-ab):>+8.1f}pp")

    ok = (rate("seen_g") >= rate("seen_b") - 0.03
          and rate("wrist_g") >= rate("wrist_b") - 0.05
          and acc_g >= acc_b - 0.03)
    print("\n判定：", "灰度可用，精度损失在可接受范围" if ok
          else "灰度有明显代价，需要逐项权衡")

    # 编码侧的参考量（PC 上的 OpenCV，只作结构参考，不代表 esp_new_jpeg）
    jr = []
    for r in rows[:20]:
        im = cv2.imread(r["path"])
        if im is not None:
            jr.append(to_robot_frame(im))
    if jr:
        import time

        def enc(arr):
            t0 = time.perf_counter()
            for _ in range(50):
                cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return (time.perf_counter() - t0) / 50 * 1e3

        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in jr[:8]]
        t_bgr = float(np.mean([enc(f) for f in jr[:8]]))
        t_gray = float(np.mean([enc(g) for g in gray]))
        sz_gray = float(np.mean([len(cv2.imencode(
            ".jpg", g, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]) for g in gray]))
        sz_bgr = float(np.mean([len(cv2.imencode(
            ".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1]) for f in jr[:8]]))
        print("\n编码结构参考（PC OpenCV，**不是** esp_new_jpeg，只看比例）：")
        print(f"  灰度编码 {t_gray:.2f} ms vs 彩色 {t_bgr:.2f} ms "
              f"-> {t_bgr/max(t_gray,1e-6):.2f}× ")
        print(f"  灰度 JPEG {sz_gray:.0f} B vs 彩色 {sz_bgr:.0f} B "
              f"-> 体积 {100*sz_gray/max(sz_bgr,1e-6):.0f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
