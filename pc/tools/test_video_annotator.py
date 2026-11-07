import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import video_annotator as va


class VideoAnnotatorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = va.Store(Path(self.tmp.name))
        (self.store.videos / "take_one.mp4").write_bytes(b"not a real video")

    def tearDown(self):
        self.tmp.cleanup()

    def test_custom_label_and_interval_round_trip(self):
        labels = self.store.add_label({"id": "tap", "name": "桌面接触", "color": "#aabbcc"})
        self.assertTrue(any(x["id"] == "tap" for x in labels))
        saved = self.store.save_annotation("take_one.mp4", 12.5, [
            {"id": "s1", "label": "tap", "start_s": 2.25, "end_s": 3.75, "note": "第一次"},
            {"id": "bad", "label": "tap", "start_s": 8, "end_s": 13},
            {"id": "unknown", "label": "does_not_exist", "start_s": 1, "end_s": 2},
        ])
        self.assertEqual(len(saved["annotations"]), 1)
        self.assertEqual(saved["annotations"][0]["start_s"], 2.25)
        loaded = self.store.annotation("take_one.mp4")
        self.assertEqual(loaded["video"], "take_one.mp4")
        self.assertEqual(loaded["annotations"][0]["note"], "第一次")
        exported = self.store.export()
        self.assertEqual(exported["videos"][0]["video"], "take_one.mp4")

    def test_invalid_color_and_duration_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_label({"name": "bad", "color": "red"})
        with self.assertRaises(ValueError):
            self.store.save_annotation("take_one.mp4", 0, [])
        with self.assertRaises(ValueError):
            self.store.save_annotation("missing.mp4", 2, [])

    def test_safe_name_blocks_path_traversal_and_preserves_extension(self):
        self.assertEqual(va.safe_name("../clip.MP4"), "clip.mp4")
        self.assertEqual(va.safe_name("bad name!.mov"), "bad_name.mov")

    def _mock_jpeg(self):
        """Encode a deterministic JPEG without a camera; returns (bytes, cleanup)."""
        import numpy as np
        path = Path(self.tmp.name) / "mock.jpg"
        self.assertTrue(va.cv2.imwrite(str(path), np.full((32, 48, 3), 127, dtype=np.uint8)))
        data = path.read_bytes()
        path.unlink()
        return data

    @staticmethod
    def _fake_connection(jpg):
        class FakeResponse:
            status = 200
            def read(self):
                return jpg

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                pass
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                # 模拟机器人 /capture 的真实响应时间，避免自由采集循环把假连接
                # 抽成无限帧（真实链路 p50 约 42–47 ms）。
                time.sleep(.04)
                return FakeResponse()
            def close(self):
                pass

        return FakeConnection

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.02)
        return predicate()

    def test_preview_reuses_one_robot_connection(self):
        jpg = b"\xff\xd8fake-jpeg\xff\xd9"
        base = self._fake_connection(jpg)
        created = []

        class CountingConnection(base):
            def __init__(self, *args, **kwargs):
                created.append(time.monotonic())
                super().__init__(*args, **kwargs)

        recorder = va.RobotRecorder(self.store, "192.168.137.50", 5)
        recorder.connection_factory = CountingConnection
        self.assertEqual(recorder.preview(), jpg)
        self.assertEqual(recorder.preview(), jpg)
        self.assertEqual(len(created), 1)
        recorder.stop()

    @unittest.skipIf(va.cv2 is None, "OpenCV required for recording test")
    def test_preview_during_recording_reuses_last_captured_frame(self):
        jpg = self._mock_jpeg()
        base = self._fake_connection(jpg)
        created = []

        class CountingConnection(base):
            def __init__(self, *args, **kwargs):
                created.append(time.monotonic())
                super().__init__(*args, **kwargs)

        recorder = va.RobotRecorder(self.store, "192.168.137.50", 5)
        recorder.connection_factory = CountingConnection
        recorder.ffmpeg = None
        started = recorder.start(fps=5)
        self.assertTrue(started["ok"])
        self._wait_for(lambda: recorder.status()["frames"] >= 2)
        before = len(created)
        self.assertEqual(recorder.preview(), jpg)  # 录制中直接给最近采集帧
        self.assertEqual(len(created), before)     # 不新增 /capture 请求，不抢相机
        stopped = recorder.stop()
        self.assertEqual(stopped["phase"], "done")

    @unittest.skipIf(va.cv2 is None, "OpenCV required for recording test")
    def test_robot_recorder_records_mock_frames_and_finalizes(self):
        jpg = self._mock_jpeg()
        recorder = va.RobotRecorder(self.store, "192.168.137.50", 5)
        recorder.connection_factory = self._fake_connection(jpg)
        recorder.ffmpeg = None  # exercise the dependency-free WebM path
        started = recorder.start()
        self.assertTrue(started["ok"])
        self._wait_for(lambda: recorder.status()["frames"] >= 12)
        stopped = recorder.stop()
        self.assertEqual(stopped["phase"], "done")
        self.assertTrue(stopped["video"].endswith(".webm"))
        self.assertGreaterEqual(stopped["frames"], 12)
        self.assertGreaterEqual(stopped["video_frames"], 2)
        self.assertGreaterEqual(stopped["frames"], stopped["video_frames"])
        video = self.store.videos / stopped["video"]
        self.assertTrue(video.exists())
        source = self.store.videos / (started["id"] + "_source")
        self.assertFalse((source / "recording.mp4").exists())
        frames = sorted((source / "frames").glob("*.jpg"))
        self.assertEqual(len(frames), stopped["frames"])
        entries = [json.loads(x) for x in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(entries), stopped["frames"])
        self.assertTrue((source / "meta.json").exists())
        metadata = self.store.metadata(video)
        self.assertGreater(metadata["duration_s"], 0)
        self.assertEqual(metadata["name"], stopped["video"])

    @unittest.skipIf(va.cv2 is None, "OpenCV required for recording test")
    def test_robot_recorder_pause_freezes_timeline(self):
        jpg = self._mock_jpeg()
        recorder = va.RobotRecorder(self.store, "192.168.137.50", 8)
        recorder.connection_factory = self._fake_connection(jpg)
        recorder.ffmpeg = None
        rejected = recorder.start(fps=25)
        self.assertFalse(rejected["ok"])
        self.assertIn("1–20", rejected["error"])
        started_at = time.monotonic()
        started = recorder.start(fps=8)
        self.assertTrue(started["ok"])
        self._wait_for(lambda: recorder.status()["frames"] >= 6)
        paused = recorder.pause()
        self.assertTrue(paused["ok"])
        self.assertEqual(recorder.status()["phase"], "paused")
        time.sleep(0.15)
        frozen_frames = recorder.status()["frames"]
        frozen_elapsed = recorder.status()["elapsed_s"]
        time.sleep(0.8)
        still = recorder.status()
        self.assertEqual(still["frames"], frozen_frames)
        self.assertGreaterEqual(still["paused_s"], 0.7)
        self.assertLessEqual(abs(still["elapsed_s"] - frozen_elapsed), 0.2)
        self.assertTrue(recorder.resume()["ok"])
        self.assertEqual(recorder.status()["phase"], "recording")
        self._wait_for(lambda: recorder.status()["frames"] > frozen_frames + 1)
        stopped = recorder.stop()
        wall = time.monotonic() - started_at
        self.assertEqual(stopped["phase"], "done")
        self.assertGreater(stopped["duration_s"], 0.2)
        self.assertLess(stopped["duration_s"], wall - 0.3)
        source = self.store.videos / (started["id"] + "_source")
        entries = [json.loads(x) for x in (source / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(entries), stopped["frames"])
        self.assertLessEqual(max(e["t_s"] for e in entries), stopped["duration_s"] + 1.0 / 8 + 0.05)

    @unittest.skipUnless(va.cv2 is not None and va.find_ffmpeg(), "OpenCV and ffmpeg required for H.264 test")
    def test_robot_recorder_transcodes_h264_mp4(self):
        jpg = self._mock_jpeg()
        recorder = va.RobotRecorder(self.store, "192.168.137.50", 5)
        recorder.connection_factory = self._fake_connection(jpg)
        self.assertTrue(recorder.ffmpeg)
        started = recorder.start()
        self.assertTrue(started["ok"])
        self._wait_for(lambda: recorder.status()["frames"] >= 8)
        stopped = recorder.stop()
        self.assertEqual(stopped["phase"], "done")
        self.assertTrue(stopped["video"].endswith(".mp4"))
        video = self.store.videos / stopped["video"]
        self.assertTrue(video.exists())
        source = self.store.videos / (started["id"] + "_source")
        self.assertFalse((source / "recording_raw.mp4").exists())
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["codec"], "h264")
        self.assertGreater(self.store.metadata(video)["duration_s"], 0)

    @unittest.skipIf(va.cv2 is None, "OpenCV required for recording test")
    def test_http_recording_endpoints_publish_video(self):
        jpg = self._mock_jpeg()
        recorder = va.RobotRecorder(self.store, "192.168.137.50", 5)
        recorder.connection_factory = self._fake_connection(jpg)
        recorder.ffmpeg = None
        handler = type("TestHandler", (va.Handler,), {"store": self.store, "recorder": recorder})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def get_json(path, data=None):
            request = urllib.request.Request(
                base + path, data=json.dumps(data).encode() if data is not None else None,
                headers={"Content-Type": "application/json"} if data is not None else {})
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode())

        def get_raw(path):
            try:
                with urllib.request.urlopen(base + path, timeout=10) as response:
                    return response.status, response.headers.get("Content-Type"), response.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.headers.get("Content-Type"), exc.read()

        try:
            status, content_type, body = get_raw("/api/preview")
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "image/jpeg")
            self.assertEqual(body, jpg)
            self.assertEqual(get_json("/api/recording")[1]["phase"], "idle")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                get_json("/api/record/start", {"robot": "example.com"})
            self.assertEqual(caught.exception.code, 409)
            status, started = get_json("/api/record/start", {"robot": "192.168.137.50", "fps": 5})
            self.assertEqual(status, 200)
            self.assertTrue(started["ok"])
            self._wait_for(lambda: get_json("/api/recording")[1]["frames"] >= 8)
            in_recording = get_raw("/api/preview")
            self.assertEqual(in_recording[0], 200)  # 录制中返回最近采集帧，不额外占用相机
            self.assertEqual(in_recording[2], jpg)
            status, paused = get_json("/api/record/pause", {})
            self.assertEqual(status, 200)
            self.assertTrue(paused["ok"])
            self.assertEqual(get_json("/api/recording")[1]["phase"], "paused")
            paused_preview = get_raw("/api/preview")
            self.assertEqual(paused_preview[0], 200)
            self.assertEqual(paused_preview[2], jpg)
            status, resumed = get_json("/api/record/resume", {})
            self.assertEqual(status, 200)
            self.assertTrue(resumed["ok"])
            self.assertEqual(get_json("/api/recording")[1]["phase"], "recording")
            resumed_preview = get_raw("/api/preview")
            self.assertEqual(resumed_preview[0], 200)
            self.assertEqual(resumed_preview[2], jpg)
            status, stopped = get_json("/api/record/stop", {})
            self.assertEqual(status, 200)
            self.assertEqual(stopped["phase"], "done")
            _, videos = get_json("/api/videos")
            names = [v["name"] for v in videos["videos"]]
            self.assertIn(stopped["video"], names)
            published = next(v for v in videos["videos"] if v["name"] == stopped["video"])
            self.assertGreater(published["duration_s"], 0)
            self.assertEqual(published["annotation_count"], 0)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()