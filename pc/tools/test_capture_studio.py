import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import capture_studio as cs


class CaptureStudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.studio = cs.Studio("127.0.0.1", 10)
        self.manifest = patch.object(cs, "OUT_ROOT", self.out / "sessions")
        self.manifest.start()
        cs.OUT_ROOT.mkdir(parents=True)

    def tearDown(self):
        self.manifest.stop()
        self.temp.cleanup()

    def test_custom_take_records_frames_events_and_metadata(self):
        started = self.studio.start_recording(
            "custom", 5, 1.2, "front", "day", "test",
            "Desk knock", "Raise hand, tap twice, then withdraw.")
        self.assertTrue(started["ok"])
        take_dir = cs.OUT_ROOT / started["dir"]
        self.studio.latest = b"jpeg-frame"
        self.studio._advance_locked(self.studio.take["t_start"] + cs.COUNTDOWN_S + 0.01)
        self.assertEqual(self.studio.take["phase"], "recording")
        t0 = self.studio.take["t0"]
        self.studio._advance_locked(t0 + 0.01)
        marked = self.studio.mark_event("first tap")
        self.assertTrue(marked["ok"])
        self.studio._advance_locked(t0 + 0.02)
        stopped = self.studio.stop_recording()
        self.assertTrue(stopped["ok"])

        meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["behavior_name"], "Desk knock")
        self.assertEqual(meta["behavior_description"], "Raise hand, tap twice, then withdraw.")
        self.assertEqual(meta["events"][0]["label"], "first tap")
        self.assertEqual(meta["frames"], 2)
        self.assertEqual((take_dir / "frames" / "00000.jpg").read_bytes(), b"jpeg-frame")
        indexes = [json.loads(line) for line in (take_dir / "frames.jsonl").read_text().splitlines()]
        self.assertEqual([row["frame_index"] for row in indexes], [0, 1])
        self.assertLess(indexes[0]["t_s"], indexes[1]["t_s"])
        events = [json.loads(line) for line in (take_dir / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[0]["frame_index"], 0)
        snapshot = self.studio.snapshot()["take"]
        self.assertEqual(snapshot["label"], "Desk knock")
        self.assertEqual(len(snapshot["events"]), 1)

    def test_custom_task_requires_a_name_and_limits_duration(self):
        self.assertFalse(self.studio.start_recording("custom", 5, None, "", "", "")["ok"])
        self.assertFalse(self.studio.start_recording("custom", 301, None, "", "", "", "Test")["ok"])
        self.assertFalse(self.studio.start_recording("unknown", 10, None, "", "", "")["ok"])

    def test_cancel_during_countdown_writes_valid_empty_session(self):
        started = self.studio.start_recording(
            "custom", 5, None, "front", "day", "", "Cancelled")
        take_dir = cs.OUT_ROOT / started["dir"]
        self.assertTrue(self.studio.stop_recording()["ok"])
        meta = json.loads((take_dir / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["frames"], 0)
        self.assertEqual(meta["seconds_actual"], 0.0)
        self.assertEqual((take_dir / "frames.jsonl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
