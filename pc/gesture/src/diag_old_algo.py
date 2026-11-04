# -*- coding: utf-8 -*-
"""诊断：老算法（wave_score>4 + 2s 衰减）在 1 Hz 慢挥上的内部状态

只回答一个问题：它为什么迟迟不触发——是「手没被判成举起」「活动手来回切换」
还是「阈值本身太高」。不改任何东西，只把内部量打出来。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, r"D:\robot_work\pc\perception")

from episodes import make_episode  # noqa: E402
from gesture import GestureEstimator  # noqa: E402
from gesture_common import skeleton_to_pixels  # noqa: E402
from calibrate_events import load_bank  # noqa: E402

DT = 1.0 / 30.0

bank = load_bank("val")
rng = np.random.default_rng(11)
for freq in (1.0, 2.0, 2.5):
    plan = [("idle", 1.2), ("raise", 0.5, "open_palm"), ("wave", 4.0),
            ("idle", 1.0)]
    ep = make_episode(bank, rng, int(bank.users[0]), plan, name="d",
                      freq_hz=freq, amp=0.55)
    est = GestureEstimator()
    ws, hu, hands = [], [], []
    for i in range(len(ep.feat)):
        kp_px = skeleton_to_pixels(ep.kp_rel[i], 60.0, (120.0, 120.0))
        g = est.update(kp_px, ep.conf[i], dt=DT)
        ws.append(g.wave_score)
        hu.append(g.hand_up_any)
        hands.append(g.debug["active_hand"])
    ws = np.array(ws)
    start = ep.segments[2]["start"]
    flips = sum(1 for i in range(1, len(hands)) if hands[i] != hands[i - 1])
    print("freq %.1f Hz: 手举起帧占比 %.2f  active_hand 切换 %d 次  "
          "wave_score 峰值 %.2f  挥手段内均值 %.2f  首次>4 的帧 %s"
          % (freq, np.mean(hu), flips, ws.max(), ws[start:].mean(),
             int(np.argmax(ws > 4)) if (ws > 4).any() else None))
