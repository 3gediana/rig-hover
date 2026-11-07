"""interval_data 的单元测试：标签解析、区间->帧映射、窗口构造。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import interval_data as d


class TestLabels(unittest.TestCase):
    def test_resolve_default_ids(self):
        id_to_name = {}
        self.assertEqual(d.resolve_label("knock", id_to_name), "knock")
        self.assertEqual(d.resolve_label("fist_hold", id_to_name), "fist_hold")

    def test_resolve_custom_names(self):
        id_to_name = {"label_1": "连续敲击", "label_2": "抬手但未敲", "label_3": "晃拳", "label_4": "别的"}
        self.assertEqual(d.resolve_label("label_1", id_to_name), "knock")
        self.assertEqual(d.resolve_label("label_2", id_to_name), "hand_raise")
        self.assertEqual(d.resolve_label("label_3", id_to_name), "other")
        self.assertIsNone(d.resolve_label("label_4", id_to_name))

    def test_frame_labels_uses_centers(self):
        intervals = [d.Interval(1.0, 2.0, "knock"), d.Interval(2.0, 2.5, "hand_raise")]
        y = d.frame_labels(intervals, n_frames=30, fps=10.0)
        # 帧 i 的中心 = (i+0.5)/10；[1.0,2.0) => i=10..19
        self.assertTrue((y[:10] == d.IGNORE).all())
        self.assertTrue((y[10:20] == d.CLASS_TO_IDX["knock"]).all())
        self.assertTrue((y[20:25] == d.CLASS_TO_IDX["hand_raise"]).all())
        self.assertTrue((y[25:] == d.IGNORE).all())

    def test_frame_labels_clips_out_of_range(self):
        intervals = [d.Interval(-5.0, 50.0, "background")]
        y = d.frame_labels(intervals, n_frames=10, fps=10.0)
        self.assertTrue((y == d.CLASS_TO_IDX["background"]).all())


class TestWindows(unittest.TestCase):
    def test_build_windows(self):
        self.assertEqual(d.build_windows(10, 4), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(d.build_windows(8, 4), [(0, 4), (4, 8)])
        self.assertEqual(d.build_windows(0, 4), [])

    def test_window_sample_diff_and_padding(self):
        frames = np.arange(10 * 4 * 4, dtype=np.uint8).reshape(10, 4, 4)
        labels = np.full(10, d.IGNORE, dtype=np.int64)
        labels[2:4] = 1
        x, y = d.window_sample(frames, labels, start=0, window=6, size=4)
        self.assertEqual(x.shape, (6, 2, 4, 4))
        self.assertEqual(y.shape, (6,))
        self.assertTrue((x[0, 1] == 0).all())  # 第一帧差分置 0
        self.assertTrue(np.allclose(x[1, 1], (frames[1].astype(np.int16) - frames[0]) / 255.0))
        self.assertEqual(int(y[2]), 1)

        x2, y2 = d.window_sample(frames, labels, start=8, window=6, size=4)
        self.assertEqual(x2.shape, (6, 2, 4, 4))
        self.assertEqual(int((y2 == d.IGNORE).sum()), 6)  # 2 真帧 + 4 补齐帧都没有标注
        self.assertTrue(np.allclose(x2[5, 0], frames[9] / 255.0))  # 补齐帧重复最后一帧
        self.assertTrue((x2[5, 1] == 0).all())  # 补齐帧差分置 0


if __name__ == "__main__":
    unittest.main()