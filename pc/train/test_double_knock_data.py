# -*- coding: utf-8 -*-
"""test_double_knock_data.py — 多源接触脉冲与“敲击两下”窗口数据集单测。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from double_knock_data import (  # noqa: E402
    COUNT_CLASSES,
    classify_window_count,
    detect_contact_impulses,
    extract_kinematics_and_energy,
    load_external_stress_clips,
    synthesize_training_windows,
)
from interval_data import Interval  # noqa: E402


class TestDoubleKnockData(unittest.TestCase):
    def test_missing_pose_boxes_produce_finite_kinematics(self):
        """当部分或全部帧腕点缺失（None/NaN）时，运动学特征不得出现 NaN/Inf。"""
        rng = np.random.default_rng(0)
        frames = rng.integers(20, 200, size=(32, 96, 96), dtype=np.uint8)
        boxes = np.empty(32, dtype=object)
        for i in range(32):
            if i % 3 == 0:
                boxes[i] = None
            elif i == 5:
                boxes[i] = np.array([np.nan, 10.0, 50.0, 60.0], dtype=np.float32)
            else:
                boxes[i] = np.array([80.0, 100.0 + (i % 5) * 4.0, 140.0, 160.0], dtype=np.float32)
        kin, impact = extract_kinematics_and_energy(frames, boxes, fps=20.0)
        self.assertEqual(kin.shape, (32, 8))
        self.assertEqual(impact.shape, (32,))
        self.assertTrue(np.isfinite(kin).all())
        self.assertTrue(np.isfinite(impact).all())

    def test_detect_contact_impulses_and_window_count(self):
        """在敲击区间内构造两次明显下探-回弹与局部冲击，应准确检出 2 次接触并判为 double_knock(2)。"""
        T = 60
        fps = 20.0
        frames = np.full((T, 96, 96), 100, dtype=np.uint8)
        boxes = np.empty(T, dtype=object)
        for t in range(T):
            y = 90.0
            if t in (15, 23):  # 相隔 8 帧 = 0.40s 的两次敲击接触
                y = 145.0
                frames[t, 45:65, 35:60] = 230
            elif t in (14, 16, 22, 24):
                y = 115.0
            boxes[t] = np.array([72.0, y - 30.0, 132.0, y + 30.0], dtype=np.float32)

        kin, impact = extract_kinematics_and_energy(frames, boxes, fps=fps)
        ivs = [Interval(start_s=0.4, end_s=2.0, label="knock")]
        peaks = detect_contact_impulses(impact, ivs, fps=fps, min_gap_s=0.20)
        self.assertEqual(len(peaks), 2)
        self.assertTrue(any(abs(p - 15) <= 1 for p in peaks))
        self.assertTrue(any(abs(p - 23) <= 1 for p in peaks))

        cls_double = classify_window_count(peaks, start_frame=0, end_frame=48, fps=fps)
        self.assertEqual(COUNT_CLASSES[cls_double], "double_knock")
        cls_single = classify_window_count([15], start_frame=0, end_frame=48, fps=fps)
        self.assertEqual(COUNT_CLASSES[cls_single], "single_knock")
        cls_multi = classify_window_count([10, 18, 26], start_frame=0, end_frame=48, fps=fps)
        self.assertEqual(COUNT_CLASSES[cls_multi], "multi_knock")
        cls_zero = classify_window_count([], start_frame=0, end_frame=48, fps=fps)
        self.assertEqual(COUNT_CLASSES[cls_zero], "non_knock")

    def test_synthesize_training_windows_forbids_val_take_leakage(self):
        """合成增强若包含留出测试 take 必须抛 ValueError，防止数据泄漏。"""
        dummy_take = {
            "x_vis": np.zeros((4, 48, 2, 96, 96), dtype=np.float32),
            "x_kin": np.zeros((4, 48, 8), dtype=np.float32),
            "y_imp": np.zeros((4, 48), dtype=np.float32),
            "y_cnt": np.array([0, 1, 2, 3], dtype=np.int64),
            "impulses": [{"vis": np.zeros((12, 2, 96, 96), dtype=np.float32),
                          "kin": np.zeros((12, 8), dtype=np.float32)}],
            "negatives": [{"vis": np.zeros((48, 2, 96, 96), dtype=np.float32),
                           "kin": np.zeros((48, 8), dtype=np.float32)}],
        }
        with self.assertRaises(ValueError):
            synthesize_training_windows(
                {"take_train": dummy_take, "take_val": dummy_take},
                forbidden_val_take="take_val",
                n_per_class=6,
            )
        synth = synthesize_training_windows(
            {"take_train": dummy_take},
            forbidden_val_take="take_val",
            n_per_class=6,
            window=48,
        )
        self.assertEqual(synth["x_vis"].shape, (18, 48, 2, 96, 96))
        self.assertEqual(synth["x_kin"].shape, (18, 48, 8))
        self.assertEqual(set(synth["y_cnt"].tolist()), {1, 2, 3})

    def test_load_external_stress_clips(self):
        """应能加载本地真机 sessions 与公开数据集（Charades/UCF101）片段用于压力测试。"""
        clips = load_external_stress_clips(max_clips_per_source=3, size=96)
        self.assertGreaterEqual(len(clips), 4)
        sources = {c["source"] for c in clips}
        self.assertIn("evidence_sessions", sources)
        for c in clips:
            self.assertEqual(c["frames"].ndim, 3)
            self.assertEqual(c["frames"].shape[1:], (96, 96))


if __name__ == "__main__":
    unittest.main()
