# -*- coding: utf-8 -*-
"""eval_classifier.py — 测试集每类 P/R/F1 + 混淆矩阵 + 严格开放集误报

测试集是**按人留出**的（test 用户从不出现在 train/val），指标直接来自
``data/synth/test.npz``：静态类样本是留出用户的**真实**骨架帧，
``wave`` 样本是这些帧 + 2 连杆 IK 合成的摆动（README §局限 有说明）。

两类「不该触发」的测试：

* ``other`` 类（call/dislike/four/mute/ok/peace）**训练时见过**，输出时概率并进
  idle；测试集里就有它们，混淆矩阵会体现。
* ``unseen_gestures.npz`` 里的 7 类（rock/three/two_up/…）**完全没进过训练**，
  是真正的开放集样本。机器人上遇到这些手势时如果硬说是 wave，就是误报。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from build_dataset import TRAIN_CLASSES  # noqa: E402
from gesture_classifier import MODEL_PATH, fold_probs  # noqa: E402
from gesture_common import (CLASSES, CROP_PX, FEAT_DIM,  # noqa: E402
                            OTHER_ID, build_features_from_rel)
from metrics import confusion, per_class_report  # noqa: E402
from model import build_model  # noqa: E402

DATA = os.path.join(ROOT, "data", "synth")
REPORTS = os.path.join(ROOT, "reports")


def load_windows(path, use_appearance=True):
    z = np.load(path)
    feat = torch.from_numpy(z["feat"].astype(np.float32))
    y = z["y"].astype(np.int64)
    crop = None
    if use_appearance:
        crop = torch.from_numpy(
            z["crop"].astype(np.float32).reshape(
                len(y), -1, 1, CROP_PX, CROP_PX) / 127.5 - 1.0)
    return feat, crop, y


@torch.no_grad()
def run_model(model, feat, crop, batch=512, device="cuda:0"):
    model.eval()
    out = []
    for s in range(0, len(feat), batch):
        fb = feat[s:s + batch].to(device)
        cb = crop[s:s + batch].to(device) if crop is not None else None
        out.append(torch.softmax(model(fb, cb), dim=-1).cpu().numpy())
    n_cls = model.head[-1].out_features
    return np.concatenate(out, axis=0) if out else np.zeros((0, n_cls))


def unseen_windows(path, T=12, seed=0):
    """严格留出的未知手势 → 静态窗口（和部署时「举着不动」一样的输入）。"""
    z = np.load(path)
    kp, conf, crop = z["kp_rel"], z["conf"], z["crop"]
    rng = np.random.default_rng(seed)
    n = len(kp)
    feats = np.empty((n, T, FEAT_DIM), dtype=np.float32)
    crops = np.empty((n, T, CROP_PX, CROP_PX), dtype=np.uint8)
    for i in range(n):
        k = kp[i][None].repeat(T, axis=0)
        c = conf[i][None].repeat(T, axis=0)
        k = k + rng.normal(0, 0.012, k.shape).astype(np.float32)
        c = np.clip(c * rng.uniform(0.85, 1.0, c.shape), 0, 1)
        feats[i] = build_features_from_rel(k, c, 0.25)
        crops[i] = crop[i][None].repeat(T, axis=0)
    return (torch.from_numpy(feats),
            torch.from_numpy(crops.astype(np.float32) / 127.5 - 1.0
                             ).unsqueeze(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MODEL_PATH)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="fused")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    use_app = bool(ckpt["model_cfg"].get("use_appearance", True))
    n_par = sum(p.numel() for p in model.parameters())
    print("[eval] %s  appearance=%s  params=%d"
          % (os.path.basename(args.ckpt), use_app, n_par), flush=True)

    # ---- 1) 测试集：折成对外 6 类 ----
    feat, crop, y = load_windows(os.path.join(DATA, "test.npz"), use_app)
    p_model = run_model(model, feat, crop, device=device)
    p6 = fold_probs(p_model)
    pred = p6.argmax(axis=1)
    # 对外只有 6 类：other 的样本折成 idle 之后再算指标
    y6 = np.where(y == OTHER_ID, 0, y)
    rep = per_class_report(y6, pred, CLASSES)
    cm = confusion(y6, pred, len(CLASSES))
    macro = float(np.mean([r["f1"] for r in rep]))
    acc = float((pred == y6).mean())
    print("[eval] 测试集 macro-F1 %.4f  acc %.4f" % (macro, acc), flush=True)
    for r in rep:
        print("   %-10s P %.3f  R %.3f  F1 %.3f  n=%d"
              % (r["class"], r["precision"], r["recall"], r["f1"],
                 r["support"]), flush=True)

    # ---- 2) 7 类（含 other）的辨识率 ----
    pred7 = p_model.argmax(axis=1)
    rep7 = per_class_report(y, pred7, TRAIN_CLASSES)
    other_row = [r for r in rep7 if r["class"] == "other"][0]
    print("[eval] other(训练时见过的非目标手势) recall %.3f  P %.3f"
          % (other_row["recall"], other_row["precision"]), flush=True)

    # ---- 3) 严格开放集：没进过训练的 7 类 ----
    unseen = {}
    up = os.path.join(DATA, "unseen_gestures.npz")
    if os.path.exists(up):
        fu, cu = unseen_windows(up)
        pu = fold_probs(run_model(model, fu, cu if use_app else None,
                                  device=device))
        arg = pu.argmax(axis=1)
        top = pu.max(axis=1)
        unseen = {
            "n": int(len(pu)),
            "frac_claimed_nonidle": float((arg != 0).mean()),
            "frac_would_trigger": float(((arg != 0) & (top >= 0.55)).mean()),
            "pred_hist": {CLASSES[k]: int((arg == k).sum())
                          for k in range(len(CLASSES))},
        }
        print("[eval] 严格开放集 %d 条：被认成非 idle %.1f%%（>阈值 %.1f%%）"
              % (unseen["n"], 100 * unseen["frac_claimed_nonidle"],
                 100 * unseen["frac_would_trigger"]), flush=True)

    os.makedirs(REPORTS, exist_ok=True)
    out = {"ckpt": os.path.basename(args.ckpt), "appearance": use_app,
           "n_params": n_par, "macro_f1": macro, "accuracy": acc,
           "per_class": rep, "confusion": cm.tolist(),
           "classes": list(CLASSES), "per_class_7": rep7,
           "unseen_open_set": unseen}
    with open(os.path.join(REPORTS, "eval_%s.json" % args.tag), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[eval] → reports/eval_%s.json" % args.tag, flush=True)


if __name__ == "__main__":
    main()
