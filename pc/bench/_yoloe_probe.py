# -*- coding: utf-8 -*-
"""_yoloe_probe.py -- 小模型能不能替我们标框？用 YOLOE 文本提示直接找 "hand"。

背景
====

本项目标框的真实需求只有两个：

1. **人框** —— yolo26n-pose 已经免费给了（COCO 预训练）。
2. **手框** —— 现在是从 yolo26n-pose 的**腕部关键点**推出来的。腕点在小目标上
   抖动大，而手图是手势分类器的输入，抖一点分类就崩。如果有一个开放词表检测器
   能直接出 "hand" 的框，就比推关键点稳。

YOLOE 是 ultralytics 自带的开放词表模型，**YOLOE-26n-seg 只有 4.8 M 参数**
（约十几 MB），比任何 VLM 小两个数量级，而且 `ultralytics` 8.4.21 已经装好。
本脚本量三件事：

* 文本提示 ``["person", "hand"]`` 在真实照片上出不出框
* 出的框跟 HaGRID 的人工标注手框差多少（IoU）
* 换成更小的输入尺寸还能不能出框（部署时要快）

Run:  python _yoloe_probe.py --n 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "perception", "assets")
HAGRID = (r"D:\robot_work\pc\gesture\data\raw\hagrid"
          r"\hagrid-sample-30k-384p")

N_PASS = 0
N_FAIL = 0


def check(name, cond, detail=""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print("  PASS  %s" % name)
    else:
        N_FAIL += 1
        print("  FAIL  %s   %s" % (name, detail))


def iou(a, b):
    """两个归一化 xyxy 的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / max(ua, 1e-9)


def load_samples(classes, n, seed=0):
    import random
    rng = random.Random(seed)
    out = []
    for cls in classes:
        ap = os.path.join(HAGRID, "ann_train_val", "%s.json" % cls)
        if not os.path.exists(ap):
            continue
        ann = json.load(open(ap, encoding="utf-8"))
        d = os.path.join(HAGRID, "hagrid_30k", "train_val_%s" % cls)
        stems = sorted(ann.keys())
        rng.shuffle(stems)
        for s in stems:
            p = os.path.join(d, s + ".jpg")
            if not os.path.exists(p) or not ann[s].get("bboxes"):
                continue
            out.append((p, cls, [float(v) for v in ann[s]["bboxes"][0]]))
            if len([o for o in out if o[1] == cls]) >= n:
                break
    return out


def pick_model(verbose=True):
    """挑一个能用的 YOLOE 权重；返回 (YOLOE 实例, 权重名)。"""
    from ultralytics import YOLOE
    names = ["yoloe-26n-seg.pt", "yoloe-11n-seg.pt", "yoloe-v8s-seg.pt"]
    last = ""
    for name in names:
        try:
            m = YOLOE(os.path.join(ASSETS, name))
            if verbose:
                print("[yoloe] 用 %s" % name)
            return m, name
        except Exception as exc:  # noqa: BLE001 - 换下一个候选
            last = "%s: %s" % (name, str(exc)[:120])
            if verbose:
                print("[yoloe] %s 不可用，换下一个" % name)
    raise SystemExit("没有可用的 YOLOE 权重。最后一次错误：%s" % last)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", default="person,hand")
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args(argv)

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    samples = load_samples(["fist", "palm", "like"], a.n)
    if not samples:
        print("没有 HaGRID 样本")
        return 2
    print("样本 %d 张，提示词 %s，输入 %d" % (len(samples), classes, a.imgsz))

    model, wname = pick_model()
    model.set_classes(classes)

    hits = {"person": 0, "hand": 0}
    ious = []
    for path, cls, bbox in samples:
        r = model.predict(path, imgsz=a.imgsz, verbose=False, conf=0.15)[0]
        names = r.names if hasattr(r, "names") else {}
        hw, hh = r.orig_shape[1], r.orig_shape[0]
        found = {}
        if r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                lb = str(names.get(int(r.boxes.cls[i]), "")).lower()
                xy = r.boxes.xyxy[i].tolist()
                norm = [xy[0] / hw, xy[1] / hh, xy[2] / hw, xy[3] / hh]
                conf = float(r.boxes.conf[i])
                if lb not in found or conf > found[lb][0]:
                    found[lb] = (conf, norm)
        for k in hits:
            hits[k] += int(k in found)
        # HaGRID 的手框是归一化 xywh -> xyxy
        gx, gy, gw, gh = bbox
        gt = [gx, gy, gx + gw, gy + gh]
        if "hand" in found:
            ious.append(iou(found["hand"][1], gt))

    n = len(samples)
    print()
    print("=" * 66)
    print("YOLOE 文本提示检测（%s, imgsz=%d, conf=0.15）" % (wname, a.imgsz))
    print("=" * 66)
    print("  person 检出 %d/%d (%.0f%%)" % (hits["person"], n,
                                            100.0 * hits["person"] / n))
    print("  hand   检出 %d/%d (%.0f%%)" % (hits["hand"], n,
                                            100.0 * hits["hand"] / n))
    if ious:
        arr = np.asarray(ious)
        print("  hand 框与 HaGRID 人工标注的 IoU：中位 %.2f  均值 %.2f  "
              "IoU>=0.5 占 %.0f%%"
              % (float(np.median(arr)), float(arr.mean()),
                 100.0 * float((arr >= 0.5).mean())))
    print()
    check("YOLOE 能加载", True)
    check("能检出人", hits["person"] > 0, str(hits))
    check("能检出 >=1 只手", hits["hand"] > 0, str(hits))
    print("\n" + "=" * 60)
    print("%d PASS / %d FAIL" % (N_PASS, N_FAIL))
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
