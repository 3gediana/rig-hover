# -*- coding: utf-8 -*-
"""_vlm_vs_pixels.py -- 手在画面里只剩 N 个像素时，VLM 还认不认得出握拳？

为什么量这个
============

场景 v2 有三个触发条件：脸够大、握拳、敲两下。后两个都依赖手势分类器，
而它吃的是 YOLO 腕部关键点周围的手图。问题在于**手在图里到底有多大**：

    60° 水平视场 / 240 px = 0.25°/px；9 cm 的手在 1.5 m 处张角 3.4°
    → 14 px

而 Qwen3-VL 的视觉编码器把 32×32 px 压成一个 token。14 px 的手**装不满
一个 token**。这跟模型选型无关，是信息量的问题。

所以本脚本量三条曲线（同一批 HaGRID 图，真值就是 HaGRID 的类别）：

* ``frame@h``：整幅画面缩放到"手只有 h 像素"，贴到 240×240 画布上
  —— 等价于机器人现在 240×240 采图、看整个人。
* ``crop@h``：同样把图缩到"手 h 像素"，然后以手为中心裁 6h 见方、放大到 224
  —— 只增加了 token 预算，没有增加信息量。
* ``cropNative``：从**原图**裁（手 ≈150 px）再放大到 224
  —— 等价于改成 640×480 采集后在腕部裁 64 px。

三条一比就能把"分辨率不够"和"token 不够"分开：如果 ``crop@h`` 明显好于
``frame@h``，说明是 token 预算问题，裁一刀就救回来了；如果两条一起塌，
说明像素本身没了，只能提高采集分辨率或缩短工作距离。

跑法::

    # 不加载模型，只验证渲染与统计（秒级）
    python _vlm_vs_pixels.py --backend mock

    # 本地 4bit Qwen3-VL-4B
    python _vlm_vs_pixels.py --backend local --model G:\\ms\\Qwen3-VL-4B-Instruct

    # 智谱免费档 glm-4.6v-flash（需要 ZHIPUAI_API_KEY）
    python _vlm_vs_pixels.py --backend glm
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
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

HAGRID = (r"D:\robot_work\pc\gesture\data\raw\hagrid"
          r"\hagrid-sample-30k-384p")

#: COCO 画面边长（机器人现在的采集尺寸）。改采集分辨率时只改这里。
CANVAS = 240
#: 手图送进网络前的边长。与场景 v2 的手势分类器输入一致。
CROP_OUT = 224
#: 裁窗相对手宽的倍数。**要跟着部署时的真实裁法选**：现在手势分类器吃的是
#: 32×32 的腕部手图，而 1.2 m 处手宽约 21 px，所以 k≈1.5。这里取 2.0
#: （手腕再带一点前臂），既不影响结论方向，又比 1.5 宽容一点。
CROP_K = 2.0


def load_hagrid(classes, per_class, seed):
    """抽样本：返回 [(图片路径, 真值类别, 手框(归一化 xywh)), ...]。"""
    rng = random.Random(seed)
    out = []
    for cls in classes:
        ann_path = os.path.join(HAGRID, "ann_train_val", "%s.json" % cls)
        if not os.path.exists(ann_path):
            raise SystemExit("找不到标注 %s" % ann_path)
        with open(ann_path, encoding="utf-8") as fh:
            ann = json.load(fh)
        img_dir = os.path.join(HAGRID, "hagrid_30k", "train_val_%s" % cls)
        stems = sorted(ann.keys())
        rng.shuffle(stems)
        n = 0
        for stem in stems:
            p = os.path.join(img_dir, stem + ".jpg")
            boxes = ann[stem].get("bboxes") or []
            if not os.path.exists(p) or not boxes:
                continue
            bx = max(boxes, key=lambda b: b[2] * b[3])   # 取最大的那只手
            out.append((p, cls, [float(v) for v in bx]))
            n += 1
            if n >= per_class:
                break
        if n < per_class:
            print("警告：%s 只取到 %d 张" % (cls, n), file=sys.stderr)
    rng.shuffle(out)
    return out


def _scale_to_hand(img, bbox, hand_px):
    """把整幅图缩放到"手宽恰好 hand_px"，返回缩放后的图和手中心。"""
    h, w = img.shape[:2]
    native = max(1e-6, bbox[2] * w)
    s = float(hand_px) / native
    small = cv2.resize(img, (max(8, int(round(w * s))),
                             max(8, int(round(h * s)))),
                       interpolation=cv2.INTER_AREA)
    cx = (bbox[0] + bbox[2] * 0.5) * small.shape[1]
    cy = (bbox[1] + bbox[3] * 0.5) * small.shape[0]
    return small, cx, cy


def render_frame(img, bbox, hand_px):
    """整幅缩到目标手宽，居中贴到 CANVAS 方画布上（超出部分裁掉）。

    注意：HaGRID 是近景网图，手占人的比例远大于真机画面。所以按"手宽 = h"
    缩放后，人看起来比真机里小。这让测试对 VLM **更有利**（干扰像素更少），
    因此如果在 h 较小时仍然失败，结论只会更强，不会被这个偏差掩盖。
    """
    small, _cx, _cy = _scale_to_hand(img, bbox, hand_px)
    sh, sw = small.shape[:2]
    canvas = np.full((CANVAS, CANVAS, 3), 110, np.uint8)   # 中灰 = 背景
    y0, x0 = (CANVAS - sh) // 2, (CANVAS - sw) // 2
    src_y, src_x = max(0, -y0), max(0, -x0)
    dst_y, dst_x = max(0, y0), max(0, x0)
    hh = min(sh - src_y, CANVAS - dst_y)
    ww = min(sw - src_x, CANVAS - dst_x)
    if hh > 0 and ww > 0:
        canvas[dst_y:dst_y + hh, dst_x:dst_x + ww] = \
            small[src_y:src_y + hh, src_x:src_x + ww]
    return canvas


def render_crop(img, bbox, hand_px, k=CROP_K, out=CROP_OUT):
    """缩到目标手宽后，以手为中心裁 ``k`` 倍手宽见方，放大到 ``out``。"""
    small, cx, cy = _scale_to_hand(img, bbox, hand_px)
    side = max(8, int(round(hand_px * k)))
    half = side // 2
    pad = side
    padded = cv2.copyMakeBorder(small, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(110, 110, 110))
    x0 = int(round(cx)) + pad - half
    y0 = int(round(cy)) + pad - half
    crop = padded[max(0, y0):y0 + side, max(0, x0):x0 + side]
    if crop.size == 0:
        crop = small
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC)


def render_native_crop(img, bbox, out=CROP_OUT, k=CROP_K):
    """从**原图**裁手（不缩放），这是"640×480 采集 + 腕部裁图"的样子。"""
    h, w = img.shape[:2]
    native = max(1.0, bbox[2] * w)
    side = max(8, int(round(native * k)))
    cx = (bbox[0] + bbox[2] * 0.5) * w
    cy = (bbox[1] + bbox[3] * 0.5) * h
    half, pad = side // 2, side
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=(110, 110, 110))
    x0 = int(round(cx)) + pad - half
    y0 = int(round(cy)) + pad - half
    crop = padded[max(0, y0):y0 + side, max(0, x0):x0 + side]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC), native


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="VLM 在多少像素的手上还能分出握拳 / 张开 / 点赞")
    ap.add_argument("--backend", default="mock",
                    choices=("local", "qwen", "glm", "sf", "mock"))
    ap.add_argument("--model", default="G:\\ms\\Qwen3-VL-4B-Instruct")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--classes", default="fist,palm,like")
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--sizes", default="10,14,21,34,55",
                    help="手在画面里的像素宽度，逗号分隔")
    ap.add_argument("--mode", default="frame,crop,native",
                    help="frame / crop / native 的子集")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-crops", default="",
                    help="把这些渲染结果存下来看（强烈建议开一次）")
    ap.add_argument("--crop-k", type=float, default=CROP_K,
                    help="裁窗 = k × 手宽。部署时是 32px 窗口 / 21px 手 ≈ 1.5")
    a = ap.parse_args(argv)

    from PIL import Image
    import vlm_label

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    modes = [m.strip() for m in a.mode.split(",") if m.strip()]

    samples = load_hagrid(classes, a.per_class, a.seed)
    print("样本 %d 张，类别 %s" % (len(samples), classes))

    if a.backend == "local":
        be = vlm_label.make_backend("local", model=a.model)
    else:
        be = vlm_label.make_backend(a.backend, api_key=a.api_key or None)

    conds = []
    for m in modes:
        if m == "native":
            conds.append("cropNative")
        else:
            conds += ["%s@%d" % (m, s) for s in sizes]

    # stats[cond] = {cls: [n, correct, unsure]}
    stats = {c: {k: [0, 0, 0] for k in classes} for c in conds}
    native_px = []

    if a.save_crops:
        os.makedirs(a.save_crops, exist_ok=True)

    done = 0
    total = len(samples) * len(conds)
    for path, cls, bbox in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        for cond in conds:
            if cond == "cropNative":
                crop, nat = render_native_crop(img, bbox, k=a.crop_k)
                native_px.append(nat)
            elif cond.startswith("frame@"):
                crop = render_frame(img, bbox, int(cond.split("@")[1]))
            else:
                crop = render_crop(img, bbox, int(cond.split("@")[1]), k=a.crop_k)
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if a.backend == "mock":
                be.MOCK_TRUTH = cls      # mock 回显真值，只为验证链路
            if a.save_crops:
                pil.save(os.path.join(a.save_crops, "%s_%s_%02d.png"
                                      % (cond.replace("@", "-"), cls, done % 99)))
            pred = be.classify(pil, classes)
            st = stats[cond][cls]
            st[0] += 1
            st[1] += int(pred == cls)
            st[2] += int(pred == "unsure")
            done += 1
        if done % 20 < len(conds):
            print("  %d/%d" % (done, total), file=sys.stderr)

    # ---------------- 报表 ----------------
    print()
    print("=" * 78)
    print("VLM 手势识别：手在图里只有 N 像素时还认不认得（真值 = HaGRID 标注）")
    if native_px:
        print("cropNative 的手在图里约 %.0f px（原图裁切，相当于 640×480 采集）"
              % float(np.mean(native_px)))
    print("=" * 78)
    hdr = "%-14s" % "条件"
    hdr += "".join("%10s" % c for c in classes) + "%10s" % "均值"
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for cond in conds:
        row = "%-14s" % cond
        num = den = 0
        for c in classes:
            n, ok, uns = stats[cond][c]
            acc = 100.0 * ok / max(n, 1)
            row += "%9.0f%%" % acc
            num += ok
            den += n
        row += "%9.1f%%" % (100.0 * num / max(den, 1))
        print(row)
        summary[cond] = {"acc": 100.0 * num / max(den, 1),
                         "n": den,
                         "per_class": {c: [stats[cond][c][0],
                                           stats[cond][c][1],
                                           stats[cond][c][2]] for c in classes}}
    print("-" * len(hdr))
    print("随机猜 = %.1f%%" % (100.0 / len(classes)))
    print()
    print("读法：frame@h 塌而 crop@h 不塌 = token 预算不够，裁图能救；")
    print("      两条一起塌 = 像素没了，只能提高采集分辨率或让机器人靠近。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"classes": classes, "sizes": sizes, "modes": modes,
                       "per_class_n": a.per_class, "cond": summary},
                      fh, ensure_ascii=False, indent=1)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
