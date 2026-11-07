#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local temporal video annotation studio for RIG-Hover.

Run:
    G:\\ai\\python11\\python.exe pc\\tools\\video_annotator.py
    # then open http://127.0.0.1:8780

The service stores original videos and interval annotations under the selected
root. It intentionally annotates temporal segments rather than image boxes.
"""
from __future__ import annotations

import argparse
import cgi
import hashlib
import http.client
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    import cv2
except Exception:  # pragma: no cover - metadata fallback when OpenCV is absent
    cv2 = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "video_annotation_data"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}
DEFAULT_LABELS = [
    {"id": "knock", "name": "敲击", "color": "#f97316"},
    {"id": "fist_hold", "name": "静止握拳", "color": "#ef4444"},
    {"id": "hand_raise", "name": "抬手/举拳", "color": "#eab308"},
    {"id": "reach", "name": "伸手但未敲", "color": "#22c55e"},
    {"id": "other", "name": "其它动作", "color": "#38bdf8"},
    {"id": "background", "name": "背景/无动作", "color": "#64748b"},
]
ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


def find_ffmpeg():
    """Locate an ffmpeg binary that can encode H.264 for in-browser playback."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


class RobotRecorder:
    def __init__(self, store, robot="192.168.137.50", fps=8.0):
        self.store = store
        self.robot = robot
        self.fps = max(1.0, float(fps))
        self.lock = threading.RLock()
        self.thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.preview_lock = threading.Lock()
        self.preview_conn = None
        self.connection_factory = http.client.HTTPConnection
        self.ffmpeg = find_ffmpeg()
        self.last_jpg = None
        self.last_frame_mono = 0.0
        self.state = {"phase": "idle", "frames": 0, "error": ""}

    def _paused_seconds_locked(self):
        total = float(self.state.get("pause_total") or 0.0)
        if self.state.get("phase") == "paused" and self.state.get("pause_started_mono"):
            total += time.monotonic() - self.state["pause_started_mono"]
        return total

    def status(self):
        with self.lock:
            out = dict(self.state)
            if out.get("phase") in ("recording", "paused"):
                paused_s = self._paused_seconds_locked()
                out["elapsed_s"] = round(time.monotonic() - out["started_mono"] - paused_s, 1)
                out["paused_s"] = round(paused_s, 1)
                out["fps"] = self.fps
                if out["elapsed_s"] > 0:
                    out["capture_fps"] = round(int(out.get("frames") or 0) / out["elapsed_s"], 1)
            for key in ("started_mono", "pause_started_mono", "pause_total"):
                out.pop(key, None)
            return out

    def start(self, robot=None, fps=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return {"ok": False, "error": "录制任务仍在结束"}
            if robot:
                try:
                    address = ipaddress.ip_address(str(robot).strip())
                    if address.version != 4 or not address.is_private:
                        raise ValueError
                    self.robot = str(address)
                except ValueError:
                    return {"ok": False, "error": "机器人地址必须是有效的内网 IPv4 地址"}
            if fps is not None:
                try:
                    chosen_fps = float(fps)
                except (TypeError, ValueError):
                    return {"ok": False, "error": "帧率无效"}
                if not 1 <= chosen_fps <= 20:
                    return {"ok": False, "error": "帧率范围为 1–20 fps"}
                self.fps = chosen_fps
            ident = datetime.now().strftime("robot_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
            suffix = ".mp4" if self.ffmpeg else ".webm"
            self.state = {"phase": "recording", "id": ident, "name": ident + suffix,
                          "frames": 0, "error": "", "started_mono": time.monotonic(),
                          "started_at": now_iso(), "pause_total": 0.0,
                          "pause_started_mono": None}
            self.stop_event.clear()
            self.pause_event.clear()
            self.last_jpg = None
            self.last_frame_mono = 0.0
            with self.preview_lock:
                self._close_preview_conn()
            self.thread = threading.Thread(target=self._record, args=(ident,), daemon=True)
            self.thread.start()
            return {"ok": True, "id": ident, "fps": self.fps}

    def stop(self):
        self.stop_event.set()
        thread = self.thread
        if thread and thread.is_alive():
            # 停止后要做重采样 + H.264 转码，长录像也需要时间，别过早返回。
            thread.join(timeout=30)
        return self.status()

    def pause(self):
        with self.lock:
            if self.state.get("phase") != "recording":
                return {"ok": False, "error": "当前没有在录制"}
            self.state["phase"] = "paused"
            self.state["pause_started_mono"] = time.monotonic()
        self.pause_event.set()
        return {"ok": True}

    def resume(self):
        with self.lock:
            if self.state.get("phase") != "paused":
                return {"ok": False, "error": "当前不是暂停状态"}
            now = time.monotonic()
            self.state["pause_total"] = float(self.state.get("pause_total") or 0.0) + (now - self.state["pause_started_mono"])
            self.state["pause_started_mono"] = None
            self.state["phase"] = "recording"
        self.pause_event.clear()
        # 暂停期间预览会重新占用相机；继续录制前先让出，避免两路抢帧。
        with self.preview_lock:
            self._close_preview_conn()
        return {"ok": True}

    def preview(self, timeout=3.0):
        """Fetch one camera frame without recording it (read-only GET /capture).

        Keeps one keep-alive connection to the robot so preview can run at ~10 fps;
        the connection is closed when recording starts. A stale connection (robot
        rebooted / server closed it) is retried once with a fresh one.
        """
        with self.lock:
            if self.state.get("phase") == "recording":
                cached = self.last_jpg
                if cached:
                    return cached
                raise RuntimeError("录制刚开始，还没有画面")
            robot = self.robot
        with self.preview_lock:
            last_error = None
            for attempt in (0, 1):
                conn = self.preview_conn
                try:
                    if conn is None:
                        conn = self.connection_factory(robot, 80, timeout=timeout)
                        self.preview_conn = conn
                    conn.request("GET", "/capture")
                    resp = conn.getresponse()
                    frame = resp.read()
                    if resp.status != 200 or not frame:
                        raise RuntimeError(f"机器人 /capture 返回 HTTP {resp.status}")
                    return frame
                except Exception as exc:
                    last_error = exc
                    self._close_preview_conn()
                    if attempt == 0:
                        continue
                    raise
            raise last_error

    def _close_preview_conn(self):
        conn = self.preview_conn
        self.preview_conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _transcode_h264(self, src, dst):
        """Convert the intermediate recording to browser-playable H.264 MP4."""
        cmd = [self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-vf", "scale=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(dst)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"ffmpeg 无法运行: {exc}")
        if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            detail = (result.stderr or "").strip()[-300:]
            try: dst.unlink()
            except OSError: pass
            raise RuntimeError(f"H.264 转码失败: {detail or 'ffmpeg returned no output'}")

    def _record(self, ident):
        folder = self.store.videos / (ident + "_source")
        frame_dir = folder / "frames"
        use_h264 = bool(self.ffmpeg)
        video_path = self.store.videos / (ident + (".mp4" if use_h264 else ".webm"))
        temp_video_path = folder / ("recording_raw.mp4" if use_h264 else "recording_final.webm")
        publish_src = folder / ("recording_final.mp4" if use_h264 else "recording_final.webm")
        if not use_h264 and temp_video_path != publish_src:
            raise RuntimeError("internal: webm publish path mismatch")
        index_path = folder / "frames.jsonl"
        writer = None
        conn = None
        start = None
        start_pause = 0.0
        count = 0
        records = []
        try:
            folder.mkdir(parents=True, exist_ok=False)
            frame_dir.mkdir()
            conn = self.connection_factory(self.robot, 80, timeout=6)
            # 自由采集：机器人 /capture 的响应时间在 ~31–141 ms 之间抖动（量化到
            # 15.625 ms），按固定槽位等节奏会错过槽位、只能补重复帧，看起来就是
            # “帧率不稳、画面一顿一顿”。这里只按相机能给的速率收帧并记录真实
            # 时间戳，停止后再按目标帧率重采样（见下面的装配段）。
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    if self.stop_event.wait(0.05):
                        break
                    continue
                try:
                    conn.request("GET", "/capture")
                    resp = conn.getresponse()
                    jpg = resp.read()
                    if resp.status != 200 or not jpg:
                        raise RuntimeError(f"robot /capture returned HTTP {resp.status}")
                except Exception:
                    try: conn.close()
                    except Exception: pass
                    conn = self.connection_factory(self.robot, 80, timeout=6)
                    if self.stop_event.wait(.2): break
                    continue
                captured_mono = time.monotonic()
                if self.pause_event.is_set():
                    continue
                with self.lock:
                    paused_s = self._paused_seconds_locked()
                    if start is None:
                        start = captured_mono
                        start_pause = paused_s
                active = max(0.0, captured_mono - start - (paused_s - start_pause))
                frame_index = count
                frame_name = f"{frame_index:06d}.jpg"
                (frame_dir / frame_name).write_bytes(jpg)
                records.append({"frame_index": frame_index, "t_s": active,
                                "captured_at": now_iso(), "file": "frames/" + frame_name})
                count += 1
                with self.lock:
                    self.state["frames"] = count
                    self.last_jpg = jpg
                    self.last_frame_mono = captured_mono
            try: conn.close()
            except Exception: pass
            conn = None
            if count == 0:
                raise RuntimeError("no frames captured")
            if cv2 is None:
                raise RuntimeError("OpenCV (cv2) is required to record an MP4")
            import numpy as np
            # 重采样到恒定帧率：每个输出槽位取时间上最近的原始帧（两指针），
            # 因此采集抖动不会变成重复帧；原始帧数可以多于视频帧数。
            duration = max(0.0, records[-1]["t_s"])
            video_frames = max(1, int(duration * self.fps + 1e-9) + 1)
            mapping = []
            cursor = 0
            for slot in range(video_frames):
                target = slot / self.fps
                while cursor + 1 < len(records) and abs(records[cursor + 1]["t_s"] - target) <= abs(records[cursor]["t_s"] - target):
                    cursor += 1
                mapping.append(cursor)
            spans = [None] * len(records)
            for slot, index in enumerate(mapping):
                if spans[index] is None:
                    spans[index] = [slot, slot + 1]
                else:
                    spans[index][1] = slot + 1
            codec = "mp4v" if use_h264 else "VP80"
            last_index = None
            last_frame = None
            for index in mapping:
                if index != last_index:
                    raw = (frame_dir / f"{records[index]['frame_index']:06d}.jpg").read_bytes()
                    last_frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if last_frame is None:
                        raise RuntimeError("could not decode camera JPEG")
                    last_index = index
                if writer is None:
                    height, width = last_frame.shape[:2]
                    writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*codec), self.fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"could not initialize {codec} video writer")
                writer.write(last_frame)
            writer.release(); writer = None
            with index_path.open("w", encoding="utf-8") as index_file:
                for i, rec in enumerate(records):
                    span = spans[i] or [0, 0]
                    index_file.write(json.dumps({"frame_index": rec["frame_index"], "t_s": round(rec["t_s"], 4),
                                                 "captured_at": rec["captured_at"], "file": rec["file"],
                                                 "video_frame_start": span[0],
                                                 "video_frame_end": span[1]}, ensure_ascii=False) + "\n")
            if use_h264:
                self._transcode_h264(temp_video_path, publish_src)
                try: temp_video_path.unlink()
                except OSError: pass
            check = cv2.VideoCapture(str(publish_src))
            try:
                if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT) or 0) < 1:
                    raise RuntimeError("recorded video failed verification")
            finally:
                check.release()
            publish_src.replace(video_path)
            capture_fps = round(count / duration, 1) if duration > 0 else 0.0
            rel_index = str(Path(ident + "_source") / "frames.jsonl")
            meta = {"source": "robot_capture", "robot": self.robot, "fps": self.fps,
                    "frames": count, "video_frames": video_frames,
                    "duration_s": round(video_frames / self.fps, 4), "capture_fps": capture_fps,
                    "codec": "h264" if use_h264 else "vp8", "container": video_path.suffix.lstrip("."),
                    "frame_index_file": rel_index, "raw_frame_dir": str(Path(ident + "_source") / "frames"),
                    "started_at": self.state["started_at"], "finished_at": now_iso()}
            (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.lock:
                self.state.update({"phase": "done", "frames": count, "video_frames": video_frames,
                                   "duration_s": meta["duration_s"], "capture_fps": capture_fps,
                                   "finished_at": meta["finished_at"], "video": video_path.name})
        except Exception as exc:
            if writer is not None:
                try: writer.release()
                except Exception: pass
            with self.lock:
                self.state.update({"phase": "error", "error": str(exc), "frames": count})


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#11151c">
<title>RIG-Hover 视频时间标注台</title>
<style>
:root{color-scheme:dark;--bg:#0b0f14;--panel:#141b24;--panel2:#0e141c;--line:#2a3645;--line2:#202a36;--text:#edf3fa;--muted:#8f9daf;--blue:#4f8cff;--blue2:#88b3ff;--red:#e25561;--green:#33c481}
*{box-sizing:border-box}html,body{height:100%;overflow:hidden}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 "Segoe UI","Microsoft YaHei",sans-serif}
button,input,select,textarea{font:inherit}#files{display:none}button{border:1px solid var(--line);background:#1b2532;color:var(--text);border-radius:6px;padding:8px 11px;cursor:pointer;min-height:36px;transition:background .15s,border-color .15s,transform .15s}button:hover{border-color:#647a98;background:#26364a}button:active{transform:translateY(1px)}button:disabled{opacity:.45;cursor:not-allowed;transform:none}.primary{background:var(--blue);border-color:var(--blue);color:#fff}.primary:hover{background:#6b9fff;border-color:#6b9fff}.danger{background:#47202a;border-color:#713542}.iconbtn{width:38px;height:38px;padding:0;font-size:17px}
.app{height:100vh;width:100%;padding:12px 16px;display:flex;flex-direction:column}.top{height:48px;flex:0 0 48px;display:flex;align-items:center;gap:10px;border-bottom:1px solid var(--line);margin-bottom:12px}.top h1{font-size:17px;margin:0;font-weight:700;letter-spacing:.01em}.top .sub{color:var(--muted);font-size:12px;margin-right:auto}.grid{display:grid;grid-template-columns:240px minmax(0,1fr) 360px;gap:12px;flex:1;min-height:0}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0;min-height:0;box-shadow:0 8px 24px #0003}.panel h2{font-size:13px;margin:0 0 10px;font-weight:650}.preview{position:relative;background:#05070a;border:1px solid var(--line);border-radius:7px;overflow:hidden;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;margin-bottom:8px}.preview img{width:100%;height:100%;object-fit:contain;display:none}.preview img.on{display:block}.preview-hint{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px;line-height:1.5;text-align:center;padding:10px;pointer-events:none}.library{display:flex;flex-direction:column;overflow:hidden}.filepick{display:flex;gap:7px;margin-bottom:10px;align-items:center}.filepick input{display:none}.filepick label{display:inline-flex;align-items:center;justify-content:center;white-space:nowrap;min-height:36px;padding:8px 12px;border-radius:6px;cursor:pointer}.video-list{overflow:auto;min-height:0;flex:1;padding-right:2px}.video-item{display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--line2);padding:10px 6px;cursor:pointer;min-width:0}.video-item:hover{background:#1a2532}.video-item.active{background:#203553;border-left:3px solid var(--blue);padding-left:3px}.video-item .name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}.video-item small{display:block;color:var(--muted);font-variant-numeric:tabular-nums;margin-top:2px}.empty{color:var(--muted);padding:16px 4px;font-size:12px}.viewer{display:flex;flex-direction:column;gap:10px;overflow:hidden}.video-wrap{background:#05070a;border:1px solid var(--line);border-radius:7px;display:flex;align-items:center;justify-content:center;min-height:250px;flex:1 1 auto;overflow:hidden}.video-wrap video{display:none;width:100%;height:100%;object-fit:contain}.video-wrap video.ready{display:block}.no-video{color:var(--muted);text-align:center;padding:20px}.transport{display:flex;align-items:center;gap:7px;flex:0 0 auto;min-width:0}.transport .time{font-variant-numeric:tabular-nums;color:var(--blue2);min-width:94px}.transport .sub{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.timeline{position:relative;height:104px;background:#080c11;border:1px solid var(--line);border-radius:7px;cursor:crosshair;user-select:none;touch-action:none;overflow:hidden}.timeline .ticks{position:absolute;inset:0;pointer-events:none}.tick{position:absolute;top:0;height:100%;border-left:1px solid #26313f;color:#718096;font-size:10px;padding-left:4px}.track{position:absolute;left:0;right:0;top:39px;height:34px}.segment{position:absolute;top:0;height:34px;border-radius:4px;border:1px solid #fff8;min-width:3px;color:#071018;font-size:11px;font-weight:700;display:flex;align-items:center;padding:0 6px;overflow:hidden;white-space:nowrap}.segment.selected{outline:2px solid #fff;z-index:3}.playhead{position:absolute;top:0;bottom:0;width:2px;background:#fff;pointer-events:none;z-index:4}.selection{position:absolute;top:27px;height:58px;border:1px dashed #dce9ff;background:#4f8cff26;pointer-events:none;z-index:2;display:none}.timeline-help{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:11px;margin-top:5px}.timeline-help span:last-child{text-align:right}.right{display:flex;flex-direction:column;gap:12px;overflow:auto;padding-right:2px}.right>.panel{flex:0 0 auto}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.field{display:flex;flex-direction:column;gap:4px;min-width:0}.field.full{grid-column:1/-1}.field label{color:var(--muted);font-size:12px}.field input,.field select,.field textarea{width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:5px;padding:8px;min-width:0;min-height:38px}.field select{background-color:var(--panel2);color:var(--text)}.field textarea{min-height:70px;resize:vertical}.actions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-top:10px}.actions button{min-width:0}.label-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 42px;gap:7px;min-width:0}.label-row input{min-width:0}.swatch{width:42px!important;height:38px;padding:3px!important}.segments{max-height:260px;overflow:auto}.seg-row{display:grid;grid-template-columns:8px minmax(0,1fr) auto;gap:8px;align-items:center;padding:9px 2px;border-bottom:1px solid var(--line2);cursor:pointer}.seg-row:hover{background:#1a2532}.seg-row.active{background:#203553}.seg-dot{width:8px;height:8px;border-radius:50%}.seg-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.seg-time{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.seg-delete{font-size:12px;padding:4px 7px}.status{color:var(--muted);font-size:12px;min-height:18px}.status.ok{color:var(--green)}.status.err{color:#ff8b8b}.kbd{color:#dce6f4;background:#232c3a;border:1px solid #39485d;border-radius:4px;padding:1px 4px;font-size:11px}
.check{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px;margin-top:8px;cursor:pointer}.check input{width:14px;height:14px;margin:0;accent-color:var(--blue)}
@media(max-width:1080px){html,body{overflow:auto}.app{height:auto;min-height:100vh}.grid{grid-template-columns:220px minmax(0,1fr);min-height:760px}.right{grid-column:1/-1;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);overflow:visible}.right>.panel:last-child{grid-column:1/-1}}@media(max-width:720px){.app{padding:10px}.top{height:auto;min-height:48px;flex-wrap:wrap;padding-bottom:10px}.top .sub{flex-basis:100%;order:3}.grid{display:flex;flex-direction:column;min-height:0}.library{height:220px}.viewer{height:560px}.video-wrap{min-height:220px}.right{display:flex}.timeline-help{flex-direction:column;gap:3px}.timeline-help span:last-child{text-align:left}.label-row{grid-template-columns:1fr 42px}.label-row select{grid-column:1/-1}}
:focus-visible{outline:2px solid var(--blue2);outline-offset:2px}
</style></head><body><main class="app">
<header class="top"><h1>RIG-Hover 视频时间标注台</h1><span class="sub">按时间段标注动作，不标单帧框</span><button id="refresh" title="刷新视频列表">刷新</button><button id="export" title="导出所有标注">导出 JSON</button></header>
<div class="grid">
<section class="panel library"><h2>采集与视频</h2><div class="preview"><div class="preview-hint" id="previewHint">预览关闭</div><img id="previewImg" alt="机器人摄像头画面"></div><div class="actions" style="margin-bottom:9px"><button id="previewToggle" style="flex:1">打开预览</button></div><div class="field"><label for="robotHost">机器人地址</label><input id="robotHost" value="192.168.137.50" inputmode="decimal"></div><div class="formgrid" style="margin-top:8px"><div class="field"><label for="recordFps">采集帧率</label><select id="recordFps" title="桌面敲击建议 15 fps，10 fps 可能漏掉接触瞬间"><option value="5">5 fps</option><option value="8">8 fps</option><option value="10">10 fps</option><option value="15" selected>15 fps</option><option value="20">20 fps</option></select></div><div class="field"><label>&nbsp;</label><button class="primary" id="recordStart">开始录制</button></div></div><div class="actions" style="margin-top:7px"><button id="recordPause" disabled title="暂停时间不计入视频；继续后时间轴接着走，适合调机位">暂停</button><button class="danger" id="recordStop" disabled title="停止并打开标注">停止并标注</button><label class="primary" for="files" style="grid-column:1/-1;display:inline-flex;align-items:center;justify-content:center;min-height:36px;padding:8px 10px;border-radius:6px;cursor:pointer">导入视频</label><input id="files" type="file" accept="video/*,.mkv,.avi,.mov" multiple></div><div class="status" id="recordStatus">录制尚未开始</div><div class="status" id="uploadStatus"></div><h2 style="margin-top:14px">视频列表</h2><div class="video-list" id="videoList"><div class="empty">尚无视频。可从机器人录制或导入文件。</div></div></section>
<section class="panel viewer"><div class="video-wrap"><video id="video" controls preload="metadata"></video><div class="no-video" id="noVideo">从左侧选择一条视频</div></div><div class="transport"><button class="iconbtn" id="play" aria-label="播放或暂停">▶</button><button id="back">−0.5s</button><button id="forward">+0.5s</button><span class="time" id="clock">0.00 / 0.00 s</span><span class="sub">拖动时间轴框选区间，或用起止时间输入框精调</span></div><div><div class="timeline" id="timeline"><div class="ticks" id="ticks"></div><div class="selection" id="selection"></div><div class="track" id="track"></div><div class="playhead" id="playhead"></div></div><div class="timeline-help"><span>拖动：框选并跟随播放 · 松手自动播放选区 · 双击：定位播放头</span><span><span class="kbd">Space</span> 播放/暂停 · <span class="kbd">I</span>/<span class="kbd">O</span> 设置起止</span></div></div><div class="status" id="status"></div></section>
<section class="right"><div class="panel"><h2>当前时间段</h2><div class="formgrid"><div class="field"><label for="start">开始（秒）</label><input id="start" type="number" min="0" step="0.01" value="0"></div><div class="field"><label for="end">结束（秒）</label><input id="end" type="number" min="0" step="0.01" value="0"></div><div class="field full"><label for="label">动作标签</label><div class="label-row"><select id="label"></select><input id="newLabel" placeholder="或输入新标签"><input id="color" class="swatch" type="color" value="#4f8cff" title="新标签颜色"></div></div><div class="field full"><label for="note">备注（可选）</label><textarea id="note" placeholder="例如：右手；第二次敲击；接触不确定"></textarea></div></div><div class="actions"><button class="primary" id="add">添加区间</button><button id="update" disabled>更新选中</button><button class="danger" id="remove" disabled>删除选中</button><button id="clear">清空表单</button></div><label class="check"><input type="checkbox" id="loopSel"> 框选后循环播放这一段</label><div class="status" id="formStatus"></div></div><div class="panel"><h2>当前视频标注 <span id="count" class="sub">0 段</span></h2><div class="segments" id="segments"><div class="empty">选择视频后，在时间轴拖出动作区间。</div></div><div class="actions"><button class="primary" id="save">保存当前视频</button></div></div><div class="panel"><h2>标注约定</h2><div class="status">一段视频可以有多个动作区间，未标区域保持未标注；不要把整段视频自动当成负样本。先标“敲击”“静止握拳”“抬手但未敲”等容易混淆的片段，后续程序会按这些区间批量切训练窗口。</div></div></section>
</div></main>
<script>
const $=id=>document.getElementById(id); let videos=[], labels=[], current=null, selected=null, drag=null, previewStop=null, loopSel=false, selectSeq=0;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>(Number(x)||0).toFixed(2); const duration=()=>Number($('video').duration)||current?.duration_s||0;
function setStatus(id,text,kind=''){const e=$(id);e.textContent=text;e.className='status '+kind;}
function labelObj(id){return labels.find(x=>x.id===id)||{id,name:id,color:'#4f8cff'};}
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json().catch(()=>({}));if(!r.ok)throw Error(j.error||`HTTP ${r.status}`);return j;}
async function loadLabels(){const j=await api('/api/labels');labels=j.labels;renderLabels();}
function renderLabels(){const old=$('label').value;$('label').innerHTML=labels.map(x=>`<option value="${esc(x.id)}">${esc(x.name)}</option>`).join('');if(old)$('label').value=old;}
async function loadVideos(){const j=await api('/api/videos');videos=j.videos;renderVideos();}
function renderVideos(){$('videoList').innerHTML=videos.length?videos.map(v=>`<div class="video-item ${current?.name===v.name?'active':''}" data-name="${esc(v.name)}"><div class="name">${esc(v.name)}<small>${fmt(v.duration_s)} s · ${v.frames||'?'} 帧 · ${v.annotation_count} 段</small></div></div>`).join(''):'<div class="empty">还没有视频。选择文件后导入。</div>';document.querySelectorAll('.video-item').forEach(e=>e.onclick=()=>selectVideo(videos.find(v=>v.name===e.dataset.name)));}
async function selectVideo(v){if(!v)return;const seq=++selectSeq;current=v;selected=null;stopPreviewSelection();$('video').classList.remove('ready');$('video').src='/media/'+encodeURIComponent(v.name);$('video').load();$('noVideo').style.display='none';renderVideos();try{const j=await api('/api/annotations?video='+encodeURIComponent(v.name));if(seq!==selectSeq||current!==v)return;v.annotations=j.annotations;v.duration_s=j.duration_s||v.duration_s||0;clearForm();renderSegments();setStatus('status',`已载入 ${v.name}，${v.annotations.length} 个区间`,'ok');}catch(e){if(seq===selectSeq&&current===v)setStatus('status',e.message,'err')}}
function renderSegments(){const d=duration()||current?.duration_s||1;$('track').innerHTML=(current?.annotations||[]).map(a=>{const l=labelObj(a.label);const left=100*a.start_s/d,w=100*(a.end_s-a.start_s)/d;return `<div class="segment ${selected?.id===a.id?'selected':''}" data-id="${a.id}" style="left:${left}%;width:${Math.max(w,.3)}%;background:${l.color}" title="${esc(l.name)} ${fmt(a.start_s)}–${fmt(a.end_s)}s">${esc(l.name)}</div>`}).join('');$('count').textContent=`${current?.annotations?.length||0} 段`;document.querySelectorAll('.segment').forEach(e=>{e.onpointerdown=ev=>ev.stopPropagation();e.onclick=ev=>{ev.stopPropagation();selected=current.annotations.find(a=>a.id===e.dataset.id);loadSelected();previewSelection(selected.start_s,selected.end_s);renderSegments();}});renderTicks(d);updatePlayhead();}
function renderTicks(d){let n=Math.min(10,Math.max(2,Math.ceil(d/5)));$('ticks').innerHTML=Array.from({length:n+1},(_,i)=>`<span class="tick" style="left:${100*i/n}%">${fmt(d*i/n)}s</span>`).join('');}
function updatePlayhead(){const d=duration()||1;$('playhead').style.left=(100*($('video').currentTime||0)/d)+'%';$('clock').textContent=`${fmt($('video').currentTime)} / ${fmt(d)} s`;}
function loadSelected(){if(!selected)return;$('start').value=fmt(selected.start_s);$('end').value=fmt(selected.end_s);$('label').value=selected.label;$('note').value=selected.note||'';$('update').disabled=false;$('remove').disabled=false;showSelection(Number(selected.start_s),Number(selected.end_s));}
function clearForm(){selected=null;$('start').value='0';$('end').value='0';$('note').value='';$('update').disabled=true;$('remove').disabled=true;hideSelection();}
function showSelection(a,b){const d=duration()||1;$('selection').style.display='block';$('selection').style.left=(100*a/d)+'%';$('selection').style.width=(100*Math.max(0,b-a)/d)+'%';}
function hideSelection(){$('selection').style.display='none';}
function stopPreviewSelection(){const v=$('video');if(previewStop){v.removeEventListener('timeupdate',previewStop);previewStop=null;}}
function scrubTo(t){const v=$('video');const d=duration();if(!d||v.readyState<1)return;stopPreviewSelection();if(!v.paused)v.pause();const x=Math.max(0,Math.min(d,Number(t)||0));if(Math.abs(v.currentTime-x)>0.012){try{v.currentTime=x}catch(e){}}updatePlayhead();}
function previewSelection(a,b){const v=$('video');const d=duration();if(!d||v.readyState<1)return;const start=Math.max(0,Math.min(d,Number(a)||0)),end=Math.max(start,Math.min(d,Number(b)||0));if(end-start<0.1)return;stopPreviewSelection();try{v.pause();v.currentTime=start}catch(e){return}const stop=()=>{if(v.currentTime>=end-0.01){if(loopSel){try{v.currentTime=start}catch(e){};v.play().catch(()=>{})}else{v.pause();v.removeEventListener('timeupdate',stop);if(previewStop===stop)previewStop=null}}};previewStop=stop;v.addEventListener('timeupdate',stop);v.play().catch(()=>{});}
function currentRange(){let a=Number($('start').value),b=Number($('end').value);const d=duration();if(!Number.isFinite(a)||!Number.isFinite(b)||a<0||b<=a||b>d+0.02)throw Error('请填写有效的开始/结束时间');return [Math.max(0,a),Math.min(d,b)];}
function newId(){return 'seg_'+Date.now().toString(36)+'_'+Math.random().toString(36).slice(2,7);}
async function addLabelIfNeeded(){let name=$('newLabel').value.trim();if(!name)return $('label').value;let id=name.toLowerCase().replace(/[^a-z0-9_-]+/g,'_').replace(/^_+|_+$/g,'')||'label_'+Date.now();if(labels.some(x=>x.id===id)){ $('label').value=id;return id;}const j=await api('/api/labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,name,color:$('color').value})});labels=j.labels;renderLabels();$('label').value=id;$('newLabel').value='';return id;}
async function mutate(mode){if(!current)return setStatus('formStatus','先选择视频','err');try{const [a,b]=currentRange();const lab=await addLabelIfNeeded();if(mode==='add')current.annotations.push({id:newId(),label:lab,start_s:a,end_s:b,note:$('note').value.trim()});else if(mode==='update'&&selected)Object.assign(selected,{label:lab,start_s:a,end_s:b,note:$('note').value.trim()});else if(mode==='remove'&&selected)current.annotations=current.annotations.filter(x=>x.id!==selected.id);selected=null;clearForm();renderSegments();setStatus('formStatus',mode==='remove'?'已删除区间':'区间已更新，点击“保存当前视频”写入磁盘','ok');}catch(e){setStatus('formStatus',e.message,'err');}}
async function save(){if(!current)return;try{await api('/api/annotations',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({video:current.name,duration_s:duration(),annotations:current.annotations})});await loadVideos();setStatus('status','已保存当前视频标注','ok');}catch(e){setStatus('status',e.message,'err');}}
$('files').onchange=async e=>{if(!e.target.files.length)return;setStatus('uploadStatus','正在导入…');const fd=new FormData();for(const f of e.target.files)fd.append('files',f);try{const j=await api('/api/upload',{method:'POST',body:fd});setStatus('uploadStatus',`已导入 ${j.uploaded.length} 条视频`,'ok');await loadVideos();}catch(err){setStatus('uploadStatus',err.message,'err')}e.target.value='';};
let previewOn=false, previewTimer=null, previewBusy=false;
function stopPreview(){previewOn=false;previewBusy=false;if(previewTimer){clearTimeout(previewTimer);previewTimer=null}$('previewImg').classList.remove('on');$('previewToggle').textContent='打开预览';$('previewHint').textContent='预览关闭';}
function schedulePreview(ms){if(previewTimer)clearTimeout(previewTimer);previewTimer=setTimeout(previewTick,ms);}
function previewTick(){if(!previewOn||previewBusy)return;const img=$('previewImg');previewBusy=true;const t0=performance.now();img.onload=()=>{previewBusy=false;if(!previewOn)return;img.classList.add('on');$('previewHint').textContent='';const cost=performance.now()-t0;schedulePreview(Math.max(10,70-cost));};img.onerror=()=>{previewBusy=false;if(!previewOn)return;img.classList.remove('on');$('previewHint').textContent='取不到画面，检查机器人地址';schedulePreview(400);};img.src='/api/preview?t='+Date.now();}
$('previewToggle').onclick=()=>{if(previewOn){stopPreview();return;}previewOn=true;$('previewToggle').textContent='关闭预览';$('previewHint').textContent='正在取图…';previewTick();};
$('refresh').onclick=loadVideos;$('export').onclick=()=>location.href='/api/export';$('play').onclick=()=>{if($('video').paused)$('video').play();else $('video').pause()};$('video').onplay=()=>$('play').textContent='❚❚';$('video').onpause=()=>$('play').textContent='▶';$('video').ontimeupdate=updatePlayhead;$('video').onerror=()=>{const code=$('video').error?$('video').error.code:0;setStatus('status',code===4?'浏览器不支持这个视频的编码，请先转成 H.264 MP4 或 WebM':'视频加载失败（错误码 '+code+'），检查文件是否完整','err');$('noVideo').style.display='block';$('noVideo').textContent='视频无法播放，请转成 H.264 MP4 或 WebM';};$('video').onloadedmetadata=()=>{$('video').classList.add('ready');if(current){current.duration_s=duration();renderSegments();}updatePlayhead()};$('back').onclick=()=>{$('video').currentTime=Math.max(0,$('video').currentTime-.5)};$('forward').onclick=()=>{$('video').currentTime=Math.min(duration(),$('video').currentTime+.5)};
let recordPhase='idle';
async function pollRecording(){try{const s=await api('/api/recording');recordPhase=s.phase;const active=s.phase==='recording'||s.phase==='paused';const paused=s.phase==='paused';$('recordStart').disabled=active;$('recordStop').disabled=!active;$('recordPause').disabled=!active;$('recordPause').textContent=paused?'继续录制':'暂停';if(s.phase==='recording'){const got=s.frames||0;setStatus('recordStatus',got?`录制中 · ${s.elapsed_s||0}s · ${got} 帧 · 输出 ${s.fps} fps${s.capture_fps?' · 采集 '+s.capture_fps+' fps':''}`:'已开始，等待机器人画面…（检查热点和电源）')}else if(paused){setStatus('recordStatus',`已暂停 · ${s.paused_s||0}s 不计时 · 已录 ${s.frames} 帧；此时可以用预览调机位`,'ok')}else if(s.phase==='done')setStatus('recordStatus',`完成 · ${s.duration_s}s · ${s.frames} 帧 · ${s.video}`,'ok');else if(s.phase==='error')setStatus('recordStatus','录制失败：'+s.error,'err');else setStatus('recordStatus','录制尚未开始');if(s.phase==='done'&&s.video&&window.lastAutoVideo!==s.video){window.lastAutoVideo=s.video;await loadVideos();const v=videos.find(x=>x.name===s.video);if(v)selectVideo(v)} }catch(e){setStatus('recordStatus',e.message,'err')}}
$('recordStart').onclick=async()=>{try{const s=await api('/api/record/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({robot:$('robotHost').value.trim(),fps:Number($('recordFps').value)})});if(s.ok){window.lastAutoVideo='';if(!previewOn){previewOn=true;$('previewToggle').textContent='关闭预览';$('previewHint').textContent='正在取图…';previewTick();}pollRecording()}else setStatus('recordStatus',s.error||'无法开始录制','err')}catch(e){setStatus('recordStatus',e.message,'err')}};
$('recordPause').onclick=async()=>{try{if(recordPhase==='paused'){await api('/api/record/resume',{method:'POST'})}else{await api('/api/record/pause',{method:'POST'})}await pollRecording()}catch(e){setStatus('recordStatus',e.message,'err')}};
$('recordStop').onclick=async()=>{setStatus('recordStatus','正在停止并封装视频…');$('recordStop').disabled=true;try{await api('/api/record/stop',{method:'POST'});await pollRecording()}catch(e){setStatus('recordStatus',e.message,'err')}};setInterval(pollRecording,1000);
$('add').onclick=()=>mutate('add');$('update').onclick=()=>mutate('update');$('remove').onclick=()=>mutate('remove');$('clear').onclick=clearForm;$('save').onclick=save;
$('start').oninput=()=>showSelection(Number($('start').value),Number($('end').value));$('end').oninput=()=>showSelection(Number($('start').value),Number($('end').value));
$('timeline').onpointerdown=e=>{if(!current)return;const r=$('timeline').getBoundingClientRect();const t=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*duration();drag={start:t,moved:false};try{$('timeline').setPointerCapture(e.pointerId)}catch(_){}$('start').value=fmt(t);$('end').value=fmt(t);showSelection(t,t);scrubTo(t);};$('timeline').onpointermove=e=>{if(!drag)return;const r=$('timeline').getBoundingClientRect();const t=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*duration();const a=Math.min(drag.start,t),b=Math.max(drag.start,t);if(b-a>0.02)drag.moved=true;$('start').value=fmt(a);$('end').value=fmt(b);showSelection(a,b);scrubTo(t);};const endTimelineDrag=()=>{if(!drag)return;const a=Number($('start').value),b=Number($('end').value),moved=drag.moved;drag=null;if(moved&&b-a>=0.12)previewSelection(a,b)};$('timeline').onpointerup=endTimelineDrag;$('timeline').onpointercancel=endTimelineDrag;$('timeline').ondblclick=e=>{const r=$('timeline').getBoundingClientRect();scrubTo(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*duration())};$('loopSel').onchange=()=>{loopSel=$('loopSel').checked};
document.onkeydown=e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click()}if(e.key.toLowerCase()==='i'){$('start').value=fmt($('video').currentTime);showSelection(Number($('start').value),Number($('end').value))}if(e.key.toLowerCase()==='o'){$('end').value=fmt($('video').currentTime);showSelection(Number($('start').value),Number($('end').value))}};
(async()=>{try{await loadLabels();await loadVideos()}catch(e){setStatus('status',e.message,'err')}})();
</script></body></html>'''


def safe_name(name: str) -> str:
    name = Path(unquote(name or "")).name
    stem, ext = os.path.splitext(name)
    stem = ID_RE.sub("_", stem).strip("._")[:120] or "video"
    ext = ext.lower()
    return stem + ext


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.videos = root / "videos"
        self.annotations = root / "annotations"
        self.config = root / "labels.json"
        self.lock = threading.RLock()
        self.videos.mkdir(parents=True, exist_ok=True)
        self.annotations.mkdir(parents=True, exist_ok=True)
        if not self.config.exists():
            self.config.write_text(json.dumps(DEFAULT_LABELS, ensure_ascii=False, indent=2), encoding="utf-8")

    def labels(self) -> list[dict]:
        with self.lock:
            try:
                data = json.loads(self.config.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else list(DEFAULT_LABELS)
            except (OSError, ValueError):
                return list(DEFAULT_LABELS)

    def add_label(self, data: dict) -> list[dict]:
        name = str(data.get("name") or "").strip()[:80]
        if not name:
            raise ValueError("标签名称不能为空")
        ident = ID_RE.sub("_", str(data.get("id") or name.lower())).strip("_")[:60] or f"label_{int(time.time())}"
        color = str(data.get("color") or "#4f8cff")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("颜色必须是 #RRGGBB 格式")
        with self.lock:
            labels = self.labels()
            if not any(x.get("id") == ident for x in labels):
                labels.append({"id": ident, "name": name, "color": color})
                self.config.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            return labels

    def metadata(self, path: Path) -> dict:
        out = {"name": path.name, "size_bytes": path.stat().st_size, "modified_at": path.stat().st_mtime, "duration_s": 0.0, "fps": 0.0, "frames": 0}
        if cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                out["fps"] = fps
                out["frames"] = frames
                out["duration_s"] = (frames / fps) if fps > 0 and frames > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0) / 1000
            finally:
                cap.release()
        ann = self.annotation(path.name)
        out["annotation_count"] = len(ann.get("annotations", [])) if ann else 0
        return out

    def list_videos(self) -> list[dict]:
        return [self.metadata(p) for p in sorted(self.videos.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

    def annotation_path(self, name: str) -> Path:
        token = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
        return self.annotations / f"{token}.json"

    def annotation(self, name: str) -> dict | None:
        p = self.annotation_path(name)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def save_annotation(self, name: str, duration_s: float, annotations: list) -> dict:
        video = safe_name(name)
        if video != name or not (self.videos / video).exists():
            raise ValueError("视频不存在")
        duration_s = float(duration_s)
        if not (duration_s > 0 and duration_s < 24 * 60 * 60):
            raise ValueError("视频时长无效")
        label_ids = {str(x.get("id")) for x in self.labels()}
        clean = []
        for a in annotations:
            try:
                start, end = float(a.get("start_s", 0)), float(a.get("end_s", 0))
            except (TypeError, ValueError):
                continue
            label = str(a.get("label") or "other")[:80]
            if not (0 <= start < end <= duration_s) or label not in label_ids:
                continue
            clean.append({"id": str(a.get("id") or f"seg_{len(clean)}")[:100], "label": label, "start_s": round(start, 4), "end_s": round(end, 4), "note": str(a.get("note") or "")[:500]})
        out = {"schema_version": 1, "video": video, "duration_s": duration_s, "annotations": sorted(clean, key=lambda x: x["start_s"]), "updated_at": now_iso()}
        self.annotation_path(video).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def export(self) -> dict:
        rows = []
        for p in sorted(self.annotations.glob("*.json")):
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return {"schema_version": 1, "exported_at": now_iso(), "labels": self.labels(), "videos": rows}


class Handler(BaseHTTPRequestHandler):
    store: Store
    recorder: RobotRecorder

    def log_message(self, fmt, *args):
        pass

    def json(self, obj, code=200, headers=None):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        try:
            if path == "/api/health": return self.json({"ok": True})
            if path == "/api/recording": return self.json(self.recorder.status())
            if path == "/api/preview":
                try:
                    frame = self.recorder.preview()
                except Exception as exc:
                    message = str(exc)
                    if any(word in message.lower() for word in ("timed out", "refused", "unreachable", "no route")):
                        message = "连不上机器人，检查移动热点和机器人电源"
                    return self.json({"error": message}, 502)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers(); self.wfile.write(frame); return
            if path == "/api/labels": return self.json({"labels": self.store.labels()})
            if path == "/api/videos": return self.json({"videos": self.store.list_videos()})
            if path == "/api/export":
                body = json.dumps(self.store.export(), ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Disposition", "attachment; filename=video_annotations.json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path == "/api/annotations":
                name = parse_qs(urlparse(self.path).query).get("video", [""])[0]
                data = self.store.annotation(safe_name(name)) or {"video": safe_name(name), "duration_s": 0, "annotations": []}
                return self.json(data)
            if path.startswith("/media/"):
                return self.media(safe_name(path[len("/media/"):]))
            return self.json({"error": "not found"}, 404)
        except Exception as exc:
            return self.json({"error": str(exc)}, 400)

    def media(self, name):
        p = self.store.videos / name
        if not p.exists() or p.suffix.lower() not in VIDEO_EXTS: return self.json({"error": "video not found"}, 404)
        total = p.stat().st_size; start, end = 0, total - 1
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header[6:].split(",", 1)[0]
            a, _, b = spec.partition("-")
            if a: start = int(a)
            if b: end = min(int(b), total - 1)
            else: end = total - 1
            if start >= total or start > end: self.send_response(416); self.end_headers(); return
            code = 206
        else: code = 200
        length = end - start + 1
        self.send_response(code); self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream"); self.send_header("Accept-Ranges", "bytes"); self.send_header("Content-Length", str(length))
        if code == 206: self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        with p.open("rb") as f:
            f.seek(start); remaining = length
            while remaining:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk: break
                self.wfile.write(chunk); remaining -= len(chunk)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/labels": return self.json({"labels": self.store.add_label(self.body_json())})
            if path == "/api/record/start":
                data = self.body_json()
                result = self.recorder.start(data.get("robot"), data.get("fps"))
                return self.json(result, 200 if result.get("ok") else 409)
            if path == "/api/record/stop":
                return self.json(self.recorder.stop())
            if path == "/api/record/pause":
                result = self.recorder.pause()
                return self.json(result, 200 if result.get("ok") else 409)
            if path == "/api/record/resume":
                result = self.recorder.resume()
                return self.json(result, 200 if result.get("ok") else 409)
            if path == "/api/upload":
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", ""), "CONTENT_LENGTH": self.headers.get("Content-Length", "")})
                uploaded=[]
                fields = form["files"] if "files" in form else []
                if not isinstance(fields, list): fields=[fields]
                for item in fields:
                    name=safe_name(item.filename or "")
                    if not name or Path(name).suffix.lower() not in VIDEO_EXTS: continue
                    dest=self.store.videos/name
                    if dest.exists(): dest=self.store.videos/(Path(name).stem+"_"+datetime.now().strftime("%H%M%S")+Path(name).suffix)
                    with dest.open("wb") as out:
                        while True:
                            chunk=item.file.read(1024*1024)
                            if not chunk: break
                            out.write(chunk)
                    uploaded.append(dest.name)
                return self.json({"uploaded": uploaded})
            return self.json({"error": "not found"}, 404)
        except Exception as exc: return self.json({"error": str(exc)}, 400)

    def do_PUT(self):
        if urlparse(self.path).path != "/api/annotations": return self.json({"error": "not found"}, 404)
        try:
            data=self.body_json(); out=self.store.save_annotation(str(data.get("video") or ""), float(data.get("duration_s") or 0), data.get("annotations") or []); return self.json(out)
        except Exception as exc: return self.json({"error": str(exc)}, 400)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--robot", default="192.168.137.50", help="default robot IPv4 for camera capture")
    ap.add_argument("--fps", type=float, default=15.0, help="default recording frame rate (1..20)")
    args = ap.parse_args()
    if not 1 <= args.fps <= 20:
        ap.error("--fps must be between 1 and 20")
    try:
        robot_ip = ipaddress.ip_address(args.robot)
        if robot_ip.version != 4 or not robot_ip.is_private:
            raise ValueError
    except ValueError:
        ap.error("--robot must be a private IPv4 address")
    store = Store(Path(args.root))
    Handler.store = store
    Handler.recorder = RobotRecorder(store, str(robot_ip), args.fps)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"video annotator on http://127.0.0.1:{args.port}", flush=True)
    print(f"data root: {store.root}", flush=True)
    print(f"robot capture default: {robot_ip} at {args.fps:g} fps", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0

if __name__ == "__main__": raise SystemExit(main())

