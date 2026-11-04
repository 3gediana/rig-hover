# -*- coding: utf-8 -*-
"""_face_size_probe.py -- 人脸小到多少像素时，专用人脸检测器还找得到？

用户明确：自动标注主要要标 **人脸** 和 **拳头**，以后可能加别的**手势**。
这一份只量人脸，因为人脸有现成的小模型可以对比：

* **YuNet**（OpenCV `FaceDetectorYN`，227 KB，输出框 + 5 个关键点）
* **Haar 级联**（OpenCV 自带 `haarcascade_frontalface_default.xml`，908 KB）

两台都不用下载大模型，YuNet 只有 227 KB。

方法
====

HaGRID 里没有人脸标注，所以用"原图上的检测结果"当参考真值：先在全分辨率
（512×384）上跑 YuNet，留下高置信度的框；再把整幅图缩小到"人脸只剩 S 像素"，
重新检测，与按比例缩小的参考框算 IoU。这是标准的"尺寸退化曲线"测法，
参考真值本身在 S 较大时与人工标注一致（肉眼扫过）。

关心的尺寸
==========

机器人上人脸宽约 15 cm。按现有 FOV：

| 距离 | 240×240 下 | 640×480 下 |
|---|---|---|
| 1.0 m | 34 px | 90 px |
| 1.5 m | 23 px | 60 px |
| 2.0 m | 17 px | 45 px |

所以真正要回答的是：**17–34 像素的脸，YuNet 认不认？**

Run:  python _face_size_probe.py --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "perception", "assets")
YUNET = os.path.join(ASSETS, "face_detection_yunet_2023mar.onnx")
HAAR = os.path.join(cv2.data.haarcascades,
                    "haarcascade_frontalface_default.xml")
HAGRID = (r"D:\robot_work\pc\gesture\data\raw\hagrid"
          r"\hagrid-sample-30k-384p")

SCALES = (1.00, 0.60, 0.40, 0.28, 0.20, 0.14, 0.10)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / max(ua, 1e-12)


class Yunet:
    name = "YuNet(227KB)"

    def __init__(self):
        if not os.path.exists(YUNET):
            raise SystemExit("缺 %s" % YUNET)
        self.det = cv2.FaceDetectorYN_create(YUNET, "", (320, 320),
                                             score_threshold=0.6, nms_threshold=0.3,
                                             top_k=50)

    def __call__(self, img):
        """返回若干 xyxy 像素框（**统一成 xyxy**，因为下游要算 IoU）。"""
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _rc, faces = self.det.detect(img)
        if faces is None:
            return []
        return [[float(b[0]), float(b[1]), float(b[0] + b[2]),
                 float(b[1] + b[3])] for b in faces]


class Haar:
    name = "Haar(908KB)"

    def __init__(self):
        self.det = cv2.CascadeClassifier(HAAR)
        if self.det.empty():
            raise SystemExit("缺 %s" % HAAR)

    def __call__(self, img):
        """返回若干 xyxy 像素框（detectMultiScale 给的是 xywh）。"""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r = self.det.detectMultiScale(g, scaleFactor=1.1, minNeighbors=4,
                                      minSize=(10, 10))
        return [[float(b[0]), float(b[1]), float(b[0] + b[2]),
                 float(b[1] + b[3])] for b in r] if len(r) else []


def load_images(n, seed=0):
    rng = random.Random(seed)
    out = []
    for cls in ("fist", "palm", "like", "stop"):
        d = os.path.join(HAGRID, "hagrid_30k", "train_val_%s" % cls)
        if not os.path.isdir(d):
            continue
        files = sorted(os.listdir(d))
        rng.shuffle(files)
        for f in files[: max(1, n // 4)]:
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                out.append(img)
    return out[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    imgs = load_images(a.n)
    print("样本 %d 张（HaGRID 384×512）" % len(imgs))
    if not imgs:
        return 2

    detectors = [Yunet(), Haar()]

    # --- 1) 全分辨率：建参考真值，顺便看两个检测器谁强 ---
    refs, keep = [], []
    for img in imgs:
        y = detectors[0](img)
        if not y:
            continue
        # 取面积最大的那张脸（HaGRID 是单人近景）
        best = max(y, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        refs.append(best)
        keep.append(img)
    print("全分辨率上 YuNet 找到脸：%d/%d" % (len(keep), len(imgs)))
    if not keep:
        return 2

    # --- 2) 尺寸退化曲线 ---
    print()
    print("=" * 74)
    print("把人脸缩到 S 像素后，还能不能检出（IoU>=0.4 记为命中）")
    print("=" * 74)
    hdr = "%-14s%10s" % ("检测器", "参考脸宽")
    for s in SCALES:
        hdr += "%9s" % ("x%.2f" % s)
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for det in detectors:
        row = "%-14s%8.0fpx" % (det.name,
                                np.mean([r[2] - r[0] for r in refs]))
        rec = {}
        for s in SCALES:
            hit = fp = tot = 0
            for img, ref in zip(keep, refs):
                h, w = img.shape[:2]
                nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
                small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                boxes = det(small)
                tot += 1
                gt = [ref[0] * s, ref[1] * s, ref[2] * s, ref[3] * s]
                best = max((iou(b, gt) for b in boxes), default=0.0)
                hit += int(best >= 0.4)
                fp += max(0, len(boxes) - 1)
            acc = 100.0 * hit / max(tot, 1)
            rec[s] = {"face_px": float(np.mean([r[2] - r[0] for r in refs]) * s),
                      "recall": acc, "n": tot, "extra_boxes": fp}
            row += "%8.0f%%" % acc
        print(row)
        summary[det.name] = rec
    print("-" * len(hdr))

    # --- 3) 换算到机器人的距离 ---
    print()
    print("=" * 74)
    print("换算：15 cm 宽的人脸，在两种采集分辨率下各距离是多少像素")
    print("=" * 74)
    print("%-10s%14s%14s" % ("距离", "240×240", "640×480"))
    for d in (1.0, 1.5, 2.0, 2.5):
        ang = 2.0 * np.arctan(0.075 / d) * 180.0 / np.pi
        print("%-10s%11.0f px%11.0f px"
              % ("%.1f m" % d, ang / 60.0 * 240, ang / 60.0 * 640))
    print()
    print("第一个 240×240 的可用档位就是工作距离的下限——"
          "YuNet 能在 17 px 上保持检出的距离，就是场景能用的最远距离。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
