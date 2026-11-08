"""ROI 敲击检测/门控规则的单元测试（不需要 torch 模型）。"""

import unittest
from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import knock_roi as kr

FRAME = np.zeros((240, 240), dtype=np.uint8)
WRIST = (120.0, 120.0)


class ScriptedScore:
    """按调用序号返回预设概率，模拟模型输出。"""

    def __init__(self, pattern):
        self.pattern = pattern
        self.i = 0

    def __call__(self, crops):
        v = self.pattern[self.i] if self.i < len(self.pattern) else 0.1
        self.i += 1
        return v


def make_signal(pattern, **kwargs):
    kw = dict(window=8, smooth_frames=1, threshold_z=0.0, fps=20.0,
              pair_min_s=0.25, pair_max_s=1.5, cooldown_s=0.5,
              gate_state=kr.MODE_GAZE_FIST, score_fn=ScriptedScore(pattern))
    kw.update(kwargs)
    return kr.KnockRoiSignal(**kw)


class RoiTrackerTest(unittest.TestCase):
    def test_tracks_and_holds_last_box(self):
        tracker = kr.RoiBoxTracker(roi_size=96, hold_frames=3)
        box = tracker.update((120, 130), 240, 240)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(float(box[2] - box[0]), 96.0)
        for _ in range(3):
            same = tracker.update(None, 240, 240)
            np.testing.assert_allclose(same, box)
        self.assertIsNone(tracker.update(None, 240, 240))

    def test_box_clipped_to_frame(self):
        tracker = kr.RoiBoxTracker(roi_size=96)
        box = tracker.update((5, 5), 240, 240)
        self.assertGreaterEqual(box[0], 0.0)
        self.assertGreaterEqual(box[1], 0.0)
        self.assertLessEqual(box[2], 240.0)

    def test_crop_fallback_uses_full_frame(self):
        crop = kr.crop_roi(FRAME, None)
        self.assertEqual(crop.shape, (96, 96))


class GateTest(unittest.TestCase):
    def _run(self, signal, n=80, state=kr.MODE_GAZE_FIST):
        fired = 0
        trigger = 0
        for _ in range(n):
            out = signal.update(FRAME, WRIST, scenario_state=state)
            fired += out["fired"]
            trigger += int(out["trigger"])
        return fired, trigger, signal.rejected

    def _pair_pattern(self, gap_frames=10):
        # 每次得分在第 22、23 帧高，第二下隔 gap_frames 帧
        p = [0.1] * 80
        for i in (22, 23):
            p[i] = 0.95
        for i in (22 + gap_frames, 23 + gap_frames):
            p[i] = 0.95
        return p

    def test_pair_trigger_fires_when_gate_open(self):
        sig = make_signal(self._pair_pattern(10))
        fired, trig, rejected = self._run(sig)
        self.assertGreaterEqual(fired, 1)
        self.assertEqual(rejected, 0)

    def test_gate_closed_rejects(self):
        sig = make_signal(self._pair_pattern(10))
        fired, trig, rejected = self._run(sig, state=1)  # 1 = MODE_GAZE_FACE，非等待敲击
        self.assertEqual(fired, 0)
        self.assertGreaterEqual(rejected, 1)

    def test_short_gap_does_not_trigger(self):
        sig = make_signal(self._pair_pattern(1))
        fired, trig, rejected = self._run(sig)
        self.assertEqual(fired, 0)

    def test_cooldown_suppresses_second_pair(self):
        p = [0.1] * 200
        for base in (22, 60):
            for i in (base, base + 1, base + 10, base + 11):
                p[i] = 0.95
        sig = make_signal(p, cooldown_s=10.0)
        fired, trig, rejected = self._run(sig, n=200)
        self.assertEqual(fired, 1)


if __name__ == "__main__":
    unittest.main()