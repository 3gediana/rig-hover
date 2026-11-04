# -*- coding: utf-8 -*-
"""extract_skeletons.py — 用 YOLO26n-pose 把 HaGRID 图片压成骨架 + 手部裁剪

只跑一次（结果缓存到 ``data/skeletons/``），之后所有训练/评测都读缓存。
**输入尺寸 320×320**，与 ``pc/perception/run_perception.py`` 的部署配置一致，
这样离线拿到的关键点分布和机器人上看到的是同一个分布。

产出（``data/skeletons/arrays.npz`` + ``index.csv``）：

    kp      (N,17,2) float32   原图像素坐标
    conf    (N,17)   float16   关键点置信度
    crop    (N,32,32) uint8    举得更高的那只手的手部灰度裁剪
    valid   (N,)     uint8     肩宽够不够（0 = 本帧算不出来）
    label   (N,)     int8      0..5，见 gesture_common.CLASSES

用法::

    python src/extract_skeletons.py --limit 200          # 先试 200 张
    python src/extract_skeletons.py                      # 全量
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gesture_common import (CROP_PX, HAGRID_TO_CLASS, CLASS_TO_ID,  # noqa: E402
                            build_frame_features, hand_crop)

OUT_DIR = os.path.join(ROOT, "data", "skeletons")
PERSON_MIN_CONF = 0.25


def _pick_person(res):
    """选面积最大的人（= 最近），与 run_perception._pick_person 同一规则。"""
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().float().cpu().numpy()
    conf = boxes.conf.detach().float().cpu().numpy().reshape(-1)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.lexsort((-conf, -areas))[0])
    kxy = np.zeros((17, 2), dtype=np.float32)
    kconf = np.zeros(17, dtype=np.float32)
    kp = getattr(res, "keypoints", None)
    if kp is not None and kp.xy is not None and len(kp.xy) > i:
        kxy = kp.xy[i].detach().float().cpu().numpy().astype(np.float32)
        if getattr(kp, "conf", None) is not None:
            kconf = (kp.conf[i].detach().float().cpu().numpy()
                     .reshape(-1).astype(np.float32))
    return {"n_det": int(len(boxes)), "kxy": kxy, "kconf": kconf,
            "conf": float(conf[i])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    rows = []
    with open(args.manifest, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print("[extract] %d images to process" % len(rows), flush=True)

    model = YOLO(args.model)
    os.makedirs(args.out, exist_ok=True)

    n = len(rows)
    kp_arr = np.zeros((n, 17, 2), dtype=np.float32)
    conf_arr = np.zeros((n, 17), dtype=np.float16)
    crop_arr = np.zeros((n, CROP_PX, CROP_PX), dtype=np.uint8)
    valid = np.zeros(n, dtype=np.uint8)
    label = np.full(n, -1, dtype=np.int8)

    t_pose = 0.0
    t_feat = 0.0
    t0 = time.perf_counter()
    for s in range(0, n, args.batch):
        chunk = rows[s:s + args.batch]
        frames, keep = [], []
        for j, r in enumerate(chunk):
            img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(img)
            keep.append(s + j)
        if not frames:
            continue
        tp = time.perf_counter()
        results = model.predict(frames, imgsz=args.imgsz, conf=PERSON_MIN_CONF,
                                verbose=False, device=args.device)
        t_pose += time.perf_counter() - tp
        tf = time.perf_counter()
        for k, (idx, res) in enumerate(zip(keep, results)):
            r = rows[idx]
            person = _pick_person(res)
            cls = HAGRID_TO_CLASS.get(r["hagrid_class"])
            # 未映射的 12 类标成 -2（"other"）：不进训练，只用来测开放集误报
            label[idx] = -2 if cls is None else CLASS_TO_ID[cls]
            if person is None:
                continue
            frame = frames[k]
            h, w = frame.shape[:2]
            kp_arr[idx] = person["kxy"]
            conf_arr[idx] = person["kconf"].astype(np.float16)
            crop_arr[idx] = hand_crop(frame, person["kxy"], person["kconf"])
            feat = build_frame_features(person["kxy"], person["kconf"],
                                        img_wh=(w, h))
            valid[idx] = int(feat[-1] > 0.5)
        t_feat += time.perf_counter() - tf
        if (s // args.batch) % 10 == 0:
            el = time.perf_counter() - t0
            done = min(n, s + args.batch)
            print("[extract] %d/%d  %.1fs  (%.1f img/s) pose %.2fs feat %.2fs"
                  % (done, n, el, done / max(el, 1e-6), t_pose, t_feat),
                  flush=True)

    el = time.perf_counter() - t0
    np.savez_compressed(os.path.join(args.out, "arrays.npz"),
                        kp=kp_arr, conf=conf_arr, crop=crop_arr,
                        valid=valid, label=label)
    idx_path = os.path.join(args.out, "index.csv")
    with open(idx_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row", "path", "hagrid_class", "user_id", "label_class",
                     "valid", "split"])
        for i, r in enumerate(rows):
            cls = HAGRID_TO_CLASS.get(r["hagrid_class"], "skip")
            wr.writerow([i, r["path"], r["hagrid_class"], r.get("user_id", ""),
                         cls if cls else "skip", int(valid[i]),
                         r.get("split", "")])

    stats = {
        "n": int(n),
        "n_valid": int(valid.sum()),
        "n_person": int((conf_arr.astype(np.float32).sum(axis=1) > 0).sum()),
        "total_s": round(el, 2),
        "pose_s": round(t_pose, 2),
        "feat_s": round(t_feat, 2),
        "imgsz": args.imgsz,
        "per_img_pose_ms": round(1000.0 * t_pose / max(n, 1), 3),
        "per_img_total_ms": round(1000.0 * el / max(n, 1), 3),
        "images_per_s": round(n / max(el, 1e-6), 1),
    }
    with open(os.path.join(args.out, "extract_stats.json"), "w",
              encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
