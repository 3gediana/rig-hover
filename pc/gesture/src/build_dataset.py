# -*- coding: utf-8 -*-
"""build_dataset.py — 按 **user 划分** train/val/test，并生成训练窗口

四件事：

1. 把骨架缓存变成 :class:`synth.PoseBank`（按 user 索引的真实基帧）。
2. **按人划分**（不是按图片）：同一个人只会出现在 train / val / test 之一。
   HaGRID 的 ``user_id`` 是官方给的受试者哈希，所以这个划分是可验证的
   （``build_meta.json`` 里存了三份 user 列表，测试脚本会断言互不相交）。
3. 生成定长窗口 (T=12 帧 @30fps ≈ 400 ms)。窗口标签 = **最后一帧**的状态，
   40% 的窗口跨动作边界（idle→举手、挥手→放下），逼模型尽早反应。
4. 把「不关心但会出现」的 12 类手势做成分类器内部的第 7 类 ``other``：

   * ``OTHER_TRAIN`` 的 6 类（call/dislike/four/mute/ok/peace）参与训练，
     输出时概率并回 idle —— 真机上遇到这些手势不再乱触发；
   * 剩下的 7 类（rock/three/two_up/…）**完全不进训练**，留作严格开放集
     误检测试，这样「没见过的手势会不会误报」才有可验证的数字。

类别先验按真实场景给（大部分时间没有手势）：idle 47%，其余 6 类分剩下的。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gesture_common import (CLASSES, COCO17, FEAT_DIM, OTHER_ID)  # noqa: E402
from synth import PoseBank, concat_blend, make_segment  # noqa: E402

SKEL_DIR = os.path.join(ROOT, "data", "skeletons")
OUT_DIR = os.path.join(ROOT, "data", "synth")

#: 训练/评测窗口长度（帧）
WINDOW = 12
FPS = 30.0

#: 训练类序：6 个公开类 + 内部 other 类
TRAIN_CLASSES = CLASSES + ("other",)

#: 类别混合先验（近似真实场景：大部分时间没有手势）
MIX = np.array([0.47, 0.11, 0.11, 0.08, 0.08, 0.07, 0.08], dtype=np.float64)
MIX = MIX / MIX.sum()

#: 跨动作边界窗口的比例（标签取最后一帧的状态）
P_TRANSITION = 0.40

#: 参与 ``other`` 训练的 HaGRID 类（剩下 7 类严格留出，用于开放集误检测试）
OTHER_TRAIN = {"call", "dislike", "four", "mute", "ok", "peace"}


def load_bank():
    """读骨架缓存 → (PoseBank, 行列表, 全部 user 名, 统计)。

    PoseBank 的 label 用**训练类序**（0..5 + 6=other）；严格留出的 7 类手势
    单独返回，不混进来。
    """
    arr = np.load(os.path.join(SKEL_DIR, "arrays.npz"))
    kp, conf = arr["kp"], arr["conf"].astype(np.float32)
    crop, valid, label = arr["crop"], arr["valid"], arr["label"]

    rows = []
    with open(os.path.join(SKEL_DIR, "index.csv"), "r", encoding="utf-8",
              newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)

    lsh, rsh = COCO17["left_shoulder"], COCO17["right_shoulder"]
    sw = np.abs(kp[:, lsh, 0] - kp[:, rsh, 0])
    mid = 0.5 * (kp[:, lsh] + kp[:, rsh])
    with np.errstate(divide="ignore", invalid="ignore"):
        kp_rel = (kp - mid[:, None, :]) / sw[:, None, None]
    kp_rel = np.clip(np.nan_to_num(kp_rel, nan=0.0, posinf=0.0, neginf=0.0),
                     -3.0, 3.0).astype(np.float32)

    users = sorted({r["user_id"] for r in rows})
    u2i = {u: i for i, u in enumerate(users)}
    user_idx = np.array([u2i[r["user_id"]] for r in rows], dtype=np.int64)

    is_other = label == -2
    is_unseen = np.array([r["hagrid_class"] not in OTHER_TRAIN for r in rows])
    train_lab = np.where(is_other, OTHER_ID, label)
    keep = (valid == 1) & (((label >= 0) & (label <= 5)) |
                           (is_other & ~is_unseen))
    hold = (valid == 1) & is_other & is_unseen

    bank = PoseBank(kp_rel[keep], conf[keep], crop[keep], train_lab[keep],
                    user_idx[keep])
    unseen = (kp_rel[hold], conf[hold], crop[hold], user_idx[hold])
    stats = {"n_rows": len(rows), "n_usable": int(keep.sum()),
             "n_other_train": int((is_other & ~is_unseen & (valid == 1)).sum()),
             "n_unseen": int(hold.sum())}
    return bank, rows, users, stats, unseen


def split_users(bank, seed=0):
    """按人随机划分 70/15/15（同一人只落在一个集合）。"""
    users = np.array(sorted(np.unique(bank.user).tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    tr = set(users[:n_tr].tolist())
    va = set(users[n_tr:n_tr + n_va].tolist())
    te = set(users[n_tr + n_va:].tolist())
    assert not (tr & va) and not (tr & te) and not (va & te)
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}


def _window(bank, rng, user, T, p_transition):
    """生成一个 (feat, crop, y) 窗口。标签 = 最后一帧的状态。"""
    y = int(rng.choice(len(TRAIN_CLASSES), p=MIX))
    name = TRAIN_CLASSES[y]
    trans = rng.random() < p_transition

    if name == "idle":
        if trans:
            other = TRAIN_CLASSES[int(rng.integers(1, len(TRAIN_CLASSES)))]
            na = int(rng.integers(max(2, T // 3), T - 2))
            fa, ca, _ = make_segment(bank, rng, user, "hold", na,
                                     cls_name=other, fps=FPS,
                                     swap_hand_p=0.5 if other != "other" else 0.0)
            fb, cb, _ = make_segment(bank, rng, user, "idle", T - na, fps=FPS)
            f, c = concat_blend(fa, ca, fb, cb)
        else:
            f, c, _ = make_segment(bank, rng, user, "idle", T, fps=FPS)
    elif name == "wave":
        if trans:
            na = int(rng.integers(3, max(4, T - 4)))
            fa, ca, _ = make_segment(bank, rng, user, "hold", na,
                                     cls_name="open_palm", fps=FPS)
            fb, cb, _ = make_segment(bank, rng, user, "wave", T - na, fps=FPS)
            # 4 帧过渡：挥手的合成中位被抬到肩线上方，基帧可能是低位拍的，
            # 过渡太短会出现「手瞬移」
            f, c = concat_blend(fa, ca, fb, cb, blend=4)
        else:
            f, c, _ = make_segment(bank, rng, user, "wave", T, fps=FPS,
                                   swap_hand_p=0.5)
    elif name == "other":
        # 未知手势：静态举着（真机上它们通常也是静态比划）
        f, c, _ = make_segment(bank, rng, user, "hold", T, fps=FPS,
                               cls_name="open_palm",
                               rows=bank.idx(OTHER_ID, user))
        y = OTHER_ID
    else:
        if trans:
            na = int(rng.integers(3, max(4, T - 4)))
            fa, ca, _ = make_segment(bank, rng, user, "raise", na,
                                     cls_name=name, fps=FPS, swap_hand_p=0.5)
            fb, cb, _ = make_segment(bank, rng, user, "hold", T - na,
                                     cls_name=name, fps=FPS, swap_hand_p=0.5)
            f, c = concat_blend(fa, ca, fb, cb)
        else:
            f, c, _ = make_segment(bank, rng, user, "hold", T,
                                   cls_name=name, fps=FPS, swap_hand_p=0.5)
    return (f.astype(np.float32), c.astype(np.uint8), y)


def gen_split(bank, users, n, seed, T=WINDOW, p_transition=P_TRANSITION):
    rng = np.random.default_rng(seed)
    users = np.asarray(users, dtype=np.int64)
    feats = np.empty((n, T, FEAT_DIM), dtype=np.float32)
    crops = np.empty((n, T, 32, 32), dtype=np.uint8)
    ys = np.empty(n, dtype=np.int64)
    for i in range(n):
        u = int(users[rng.integers(len(users))])
        f, c, y = _window(bank, rng, u, T, p_transition)
        feats[i], crops[i], ys[i] = f, c, y
    return feats, crops, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    bank, rows, users, stats, unseen = load_bank()
    print("[build] %d 张图，%d 帧可用（含 other %d），%d 个 user，严格留出 %d"
          % (stats["n_rows"], stats["n_usable"], stats["n_other_train"],
             len(users), stats["n_unseen"]), flush=True)

    cnt = Counter(bank.label.tolist())
    print("[build] 每类基帧数: %s"
          % {TRAIN_CLASSES[k]: v for k, v in sorted(cnt.items())}, flush=True)

    sp = split_users(bank, seed=args.seed)
    print("[build] user 划分 train/val/test = %d/%d/%d"
          % (len(sp["train"]), len(sp["val"]), len(sp["test"])), flush=True)

    for name, n in (("train", args.train), ("val", args.val),
                    ("test", args.test)):
        us = sp[name]
        sub = bank.subset(np.isin(bank.user, us))
        f, c, y = gen_split(sub, us, n,
                            seed=args.seed + {"train": 1, "val": 2,
                                              "test": 3}[name])
        np.savez(os.path.join(OUT_DIR, "%s.npz" % name), feat=f, crop=c, y=y)
        np.savez(os.path.join(OUT_DIR, "bank_%s.npz" % name),
                 kp_rel=sub.kp, conf=sub.conf, crop=sub.crop,
                 label=sub.label, user=sub.user)
        print("[build] %s: %d 窗口，%d 人，类别分布 %s"
              % (name, n, len(us),
                 {TRAIN_CLASSES[k]: int(v) for k, v in
                  sorted(Counter(y.tolist()).items())}), flush=True)

    # 严格留出的 7 类手势（完全不参与训练）：只留 test 用户的
    kp_rel, conf, crop, user_idx = unseen
    te = set(sp["test"])
    m = np.isin(user_idx, list(te))
    if m.any():
        np.savez(os.path.join(OUT_DIR, "unseen_gestures.npz"),
                 kp_rel=kp_rel[m], conf=conf[m], crop=crop[m],
                 user=user_idx[m])
        print("[build] 严格留出未知手势（test 用户）：%d 帧" % int(m.sum()),
              flush=True)

    meta = {
        "window": WINDOW, "fps": FPS, "feat_dim": FEAT_DIM,
        "classes": list(CLASSES), "model_classes": list(TRAIN_CLASSES),
        "mix": MIX.tolist(), "p_transition": P_TRANSITION,
        "n_images": stats["n_rows"], "n_usable": stats["n_usable"],
        "n_other_train": stats["n_other_train"], "n_unseen": stats["n_unseen"],
        "n_users": len(users),
        "users": {k: [users[i] for i in v] for k, v in sp.items()},
        "n_windows": {"train": args.train, "val": args.val,
                      "test": args.test},
        "class_counts_frames": {TRAIN_CLASSES[k]: int(v)
                                for k, v in cnt.items()},
        "other_train_hagrid_classes": sorted(OTHER_TRAIN),
    }
    with open(os.path.join(OUT_DIR, "build_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("[build] 写出 %s" % OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
