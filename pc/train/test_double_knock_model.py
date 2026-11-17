# -*- coding: utf-8 -*-
"""test_double_knock_model.py — 双流双头“敲击两下”模型与流式触发器单测。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_double_knock import (  # noqa: E402
    MODE_GAZE_FIST,
    DoubleKnockDetector,
    DoubleKnockNet,
)


class TestDoubleKnockModel(unittest.TestCase):
    def test_double_knock_net_forward_shapes(self):
        """DoubleKnockNet 前向应同时输出逐帧接触脉冲 logit (B, T) 与四分类窗口计数 logit (B, 4)。"""
        model = DoubleKnockNet(kin_dim=8, hidden_dim=64, num_classes=4)
        model.eval()
        x_vis = torch.randn(2, 48, 2, 96, 96)
        x_kin = torch.randn(2, 48, 8)
        with torch.no_grad():
            logit_imp, logit_cnt = model(x_vis, x_kin)
        self.assertEqual(tuple(logit_imp.shape), (2, 48))
        self.assertEqual(tuple(logit_cnt.shape), (2, 4))
        self.assertTrue(torch.isfinite(logit_imp).all())
        self.assertTrue(torch.isfinite(logit_cnt).all())

    def test_double_knock_detector_gating_and_cooldown(self):
        """流式检测器在非 MODE_GAZE_FIST 状态必须拦截触发，在 MODE_GAZE_FIST 下检出双敲并执行冷却。"""
        # 用注入式 fake_predictor 模拟在 t=10 和 t=18 (相隔 0.4s @ 20fps) 出现两次脉冲峰且窗口判为 double_knock
        call_idx = {"t": 0}

        def fake_predict(vis_win: np.ndarray, kin_win: np.ndarray):
            t = call_idx["t"]
            call_idx["t"] += 1
            p_imp = 0.92 if t in (10, 18, 40, 48) else 0.05
            # 概率分布: [non_knock, single, double, multi]
            probs_cnt = np.array([0.05, 0.05, 0.85, 0.05], dtype=np.float32) if p_imp > 0.5 else np.array([0.85, 0.05, 0.05, 0.05], dtype=np.float32)
            return p_imp, probs_cnt

        det = DoubleKnockDetector(
            predictor=fake_predict,
            window=16,
            fps=20.0,
            imp_threshold=0.5,
            double_threshold=0.5,
            pair_min_s=0.20,
            pair_max_s=1.40,
            cooldown_s=2.0,
        )
        dummy_gray = np.zeros((96, 96), dtype=np.uint8)

        # 第一轮 (36 帧：前 15 帧填满 window=16，后 21 帧对应 t=0..20，覆盖 t=10, 18)：处于 MODE_GAZE_FACE(1)，必须被门控拒绝
        fired_face = 0
        for _ in range(36):
            out = det.step(dummy_gray, wrist_box=None, scenario_state=1)
            fired_face += out["fired"]
        self.assertEqual(fired_face, 0)
        self.assertGreaterEqual(det.rejected, 1)

        # 第二轮 (35 帧：对应 t=21..55，覆盖 t=40, 48)：切入 MODE_GAZE_FIST(2)，应触发且仅触发 1 次
        fired_fist = 0
        for _ in range(35):
            out = det.step(dummy_gray, wrist_box=None, scenario_state=MODE_GAZE_FIST)
            fired_fist += out["fired"]
        self.assertEqual(fired_fist, 1)

    def test_detector_accepts_variable_frame_rate_timestamps(self):
        """时间戳模式下，13 fps 输入也应该在约 2.4 s 后进入 ready，并按 20 fps 时间基重建窗口。"""
        windows = []

        def fake_predict(vis_win: np.ndarray, kin_win: np.ndarray):
            windows.append((vis_win.shape, kin_win.shape))
            return 0.05, np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)

        det = DoubleKnockDetector(
            predictor=fake_predict,
            window=48,
            fps=20.0,
            imp_threshold=0.5,
            double_threshold=0.5,
        )
        dummy_gray = np.zeros((96, 96), dtype=np.uint8)
        ready = 0
        for i in range(40):  # 13 fps 下约 3.0 s
            out = det.step(dummy_gray, wrist_box=None, scenario_state=MODE_GAZE_FIST, timestamp=i / 13.0)
            ready += int(out["ready"])
        self.assertGreaterEqual(ready, 5)
        self.assertTrue(windows)
        for vis_shape, kin_shape in windows:
            self.assertEqual(tuple(vis_shape), (48, 2, 96, 96))
            self.assertEqual(tuple(kin_shape), (48, 8))


if __name__ == "__main__":
    unittest.main()
