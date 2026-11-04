# -*- coding: utf-8 -*-
"""scan_false_alarms.py — 误报率扫描（次/分钟）

`calibrate_events.py` 里的 idle 序列只有 20 s，估不准。这里单独扫几分钟：

1. **纯静止**：连续 idle 段（真实 no_gesture 基帧），应 0 次。
2. **没见过的手势**：7 类完全没进训练的 HaGRID 手势，举着不动 10 s，
   理想是 0 次（模型应该说「没有我要的手势」）。
3. **举着不动但不是我们要的手势**（训练里见过的 6 类非目标手势里的 peace 等）
   已经被折成 idle，这里用 unseen 那批代表「未知」。

用法::

    python src/scan_false_alarms.py --minutes 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from gesture_classifier import MODEL_PATH, GestureClassifier  # noqa: E402
from gesture_common import CROP_PX, FEAT_DIM, build_features_from_rel  # noqa: E402
from gesture_events import GestureEventMachine  # noqa: E402
from synth import PoseBank, make_segment  # noqa: E402

DT = 1.0 / 30.0
FPS = 30.0


def _hold_windows(kp_rel, conf, crop, n_frames, rng):
    """把单帧姿势铺成 n_frames（带抖动），模拟「举着不动」。"""
    k = kp_rel[None].repeat(n_frames, axis=0)
    c = conf[None].repeat(n_frames, axis=0)
    k = k + rng.normal(0, 0.012, k.shape).astype(np.float32)
    c = np.clip(c * rng.uniform(0.85, 1.0, c.shape), 0, 1)
    feat = build_features_from_rel(k, c, 0.25)
    crops = crop[None].repeat(n_frames, axis=0)
    return feat, crops


def scan(clf, cfg, episodes, label):
    m = GestureEventMachine(cfg)
    n_on = 0
    total = 0.0
    for feat, crop in episodes:
        clf.reset()
        m.reset()
        for i in range(len(feat)):
            p = clf.push_feat(feat[i], crop[i])
            o = m.update(p, DT)
            n_on += o.gesture_onset
        total += len(feat) * DT
    rate = n_on / max(total / 60.0, 1e-9)
    print("  %-28s %4.1f 分钟  %3d 次 onset  → %.2f 次/分钟"
          % (label, total / 60.0, n_on, rate), flush=True)
    return {"label": label, "minutes": total / 60.0, "onsets": int(n_on),
            "per_min": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    clf = GestureClassifier.load(MODEL_PATH, device=args.device)
    with open(os.path.join(ROOT, "configs", "gesture_events.yaml"), "r",
              encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    z = np.load(os.path.join(ROOT, "data", "synth", "bank_test.npz"))
    bank = PoseBank(z["kp_rel"], z["conf"], z["crop"], z["label"], z["user"])
    users = bank.users
    n_ep = max(4, int(round(args.minutes * 60 / 10.0)))

    # 1) 纯静止 10 s × n
    idle_eps = []
    for i in range(n_ep):
        u = int(users[i % len(users)])
        f, c, _ = make_segment(bank, rng, u, "idle", int(10 * FPS))
        idle_eps.append((f, c))
    res = [scan(clf, cfg, idle_eps, "纯静止（no_gesture）")]

    # 2) 没进过训练的 7 类手势，各举 10 s
    up = os.path.join(ROOT, "data", "synth", "unseen_gestures.npz")
    if os.path.exists(up):
        zu = np.load(up)
        n = min(len(zu["kp_rel"]), n_ep)
        eps = []
        for i in range(n):
            eps.append(_hold_windows(zu["kp_rel"][i], zu["conf"][i],
                                     zu["crop"][i], int(10 * FPS), rng))
        res.append(scan(clf, cfg, eps, "7 类未见手势（举 10 s）"))

    # 3) 训练时见过的非目标手势（other 类），举 10 s
    other = np.flatnonzero(bank.label == 6)
    if len(other):
        n = min(len(other), max(4, n_ep // 2))
        pick = rng.choice(other, size=n, replace=False)
        eps = []
        for i in pick:
            eps.append(_hold_windows(bank.kp[i], bank.conf[i], bank.crop[i],
                                     int(10 * FPS), rng))
        res.append(scan(clf, cfg, eps, "other 类手势（举 10 s）"))

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "false_alarms.json"), "w",
              encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("[scan] → reports/false_alarms.json", flush=True)


if __name__ == "__main__":
    main()
