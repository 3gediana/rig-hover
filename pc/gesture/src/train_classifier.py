# -*- coding: utf-8 -*-
"""train_classifier.py — 训练事件级手势分类器（骨架窗口 + 可选手部外观）

用法::

    python src/train_classifier.py                     # 融合模型（骨架+外观）
    python src/train_classifier.py --skeleton-only     # 消融：只用骨架

两个模型都存到 ``models/``，消融结果用来回答「COCO-17 骨架到底能不能分开
fist/open_palm/pointing/thumbs_up」这个问题（答案是不能，见 reports/classifier.md）。

按人划分在 ``build_dataset.py`` 里完成，这里只训练。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from build_dataset import FPS, TRAIN_CLASSES, WINDOW  # noqa: E402
from gesture_common import CLASSES, CROP_PX, FEAT_DIM  # noqa: E402
from metrics import macro_f1, per_class_report  # noqa: E402
from model import build_model  # noqa: E402
from gesture_classifier import save_checkpoint  # noqa: E402

DATA = os.path.join(ROOT, "data", "synth")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")


def load_split(name, use_appearance=True):
    z = np.load(os.path.join(DATA, "%s.npz" % name))
    feat = torch.from_numpy(z["feat"].astype(np.float32))
    y = torch.from_numpy(z["y"].astype(np.int64))
    crop = None
    if use_appearance:
        c = z["crop"].astype(np.float32) / 127.5 - 1.0
        crop = torch.from_numpy(c).unsqueeze(2)          # (N,T,1,H,W)
    return feat, crop, y


def iterate(n, batch, shuffle, device, feat, crop, y, seed=0):
    idx = torch.arange(n)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = idx[torch.randperm(n, generator=g)]
    for s in range(0, n, batch):
        b = idx[s:s + batch]
        fb = feat[b].to(device)
        cb = crop[b].to(device) if crop is not None else None
        yb = y[b].to(device)
        yield fb, cb, yb


@torch.no_grad()
def evaluate(model, feat, crop, y, batch=512, device="cuda:0"):
    model.eval()
    preds = []
    n = len(y)
    for fb, cb, _ in iterate(n, batch, False, device, feat, crop, y):
        logits = model(fb, cb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds) if preds else np.zeros(0, np.int64)
    rep = per_class_report(y.numpy(), pred, TRAIN_CLASSES)
    return rep, float(np.mean([r["f1"] for r in rep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_app = not args.skeleton_only

    feat_tr, crop_tr, y_tr = load_split("train", use_app)
    feat_va, crop_va, y_va = load_split("val", use_app)
    if args.limit:
        feat_tr, crop_tr, y_tr = (feat_tr[:args.limit],
                                  crop_tr[:args.limit] if crop_tr is not None
                                  else None, y_tr[:args.limit])
        feat_va, crop_va, y_va = (feat_va[:args.limit // 4],
                                  crop_va[:args.limit // 4]
                                  if crop_va is not None else None,
                                  y_va[:args.limit // 4])
    n_tr = len(y_tr)
    print("[train] train %d / val %d  appearance=%s  device=%s"
          % (n_tr, len(y_va), use_app, device), flush=True)

    model_cfg = {"feat_dim": FEAT_DIM, "n_classes": len(TRAIN_CLASSES),
                 "width": args.width, "use_appearance": use_app,
                 "crop_px": CROP_PX, "pool_last": 4,
                 "dropout": args.dropout}
    model = build_model(**model_cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print("[train] 参数量 %d" % n_par, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = -1.0
    best_name = ("gesture_classifier_skeleton.pt" if args.skeleton_only
                 else "gesture_classifier.pt")
    hist = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n_seen = 0.0, 0
        for fb, cb, yb in iterate(n_tr, args.batch, True, device,
                                  feat_tr, crop_tr, y_tr, seed=args.seed + ep):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(fb, cb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(yb)
            n_seen += len(yb)
        sched.step()
        rep, mf1 = evaluate(model, feat_va, crop_va, y_va, device=device)
        acc = float(np.sum([r["recall"] * r["support"] for r in rep])
                    / max(sum(r["support"] for r in rep), 1))
        hist.append({"epoch": ep + 1, "loss": tot / max(n_seen, 1),
                     "val_macro_f1": mf1, "val_acc": acc,
                     "sec": round(time.time() - t0, 2)})
        print("[train] ep %2d  loss %.4f  val_macro_f1 %.4f  acc %.4f  %.1fs"
              % (ep + 1, tot / max(n_seen, 1), mf1, acc, time.time() - t0),
              flush=True)
        if mf1 > best:
            best = mf1
            meta = {"classes": list(CLASSES),
                    "model_classes": list(TRAIN_CLASSES),
                    "window": WINDOW, "fps": FPS,
                    "crop_px": CROP_PX, "feat_dim": FEAT_DIM,
                    "use_appearance": use_app, "val_macro_f1": mf1,
                    "n_params": n_par, "epochs": args.epochs,
                    "train_windows": int(n_tr),
                    "note": "按 user 划分；wave 由真实 palm 姿势 + 2 连杆 IK 合成"}
            name = ("gesture_classifier_skeleton.pt" if args.skeleton_only
                    else "gesture_classifier.pt")
            save_checkpoint(os.path.join(MODELS, name), model, model_cfg, meta)
            best_name = name

    os.makedirs(REPORTS, exist_ok=True)
    out = {"model": best_name, "n_params": n_par, "best_val_macro_f1": best,
           "history": hist, "skeleton_only": args.skeleton_only}
    path = os.path.join(REPORTS, "train_%s.json"
                        % ("skeleton" if args.skeleton_only else "fused"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("[train] best val macro-F1 %.4f → models/%s" % (best, best_name),
          flush=True)


if __name__ == "__main__":
    main()
