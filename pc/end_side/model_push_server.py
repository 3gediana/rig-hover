# -*- coding: utf-8 -*-
"""model_push_server.py — PC 局域网模型包、技能动物园与遥测黑匣子双向数据飞轮服务 (Path C)。

默认监听 0.0.0.0:8790，在手机连接同一热点 (qasd) 后可一键拉取模型与回传黑匣子日志：
  - GET  /models.json            列出可用模型与技能包 (.zip) 及 SHA-256、task、大小、下载路径
  - GET  /models/<filename>.zip  直接下载规范化模型包 (model.tflite + model.json)
  - GET  /strategies.json        返回与 PC 完全同步的策略预设列表 (供 App 一键同步)
  - POST /telemetry/upload       上传端侧黑匣子 JSONL 日志并自动执行 Huber/IRLS 系统辨识校准
  - GET  /telemetry/list         列出已上传保存的遥测黑匣子日志列表
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import zipfile
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategy_engine import builtin_presets  # noqa: E402
from telemetry_flywheel import auto_calibrate_system_id, parse_blackbox_jsonl  # noqa: E402


DEFAULT_SEARCH_DIRS = [
    Path("D:/robot_work/runs_end_side/neural_3dgs"),
    Path("D:/robot_work/runs_end_side/hover_rl_policy"),
    Path("D:/robot_work/runs_knock/double_knock_20261115/bundle"),
    Path("D:/robot_work/runs_end_side/mono_depth"),
    Path("D:/robot_work/runs_end_side/skill_zoo"),
]

DEFAULT_UPLOAD_DIR = Path("D:/robot_work/runs_end_side/uploaded_logs")


def discover_model_bundles(search_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """递归扫描目录中合法的 *-bundle.zip 模型包，读取其中的 model.json 与真实 SHA-256。"""
    dirs = search_dirs if search_dirs is not None else DEFAULT_SEARCH_DIRS
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in dirs:
        if not root.exists():
            continue
        for zip_path in sorted(root.rglob("*.zip")):
            if zip_path.name in seen:
                continue
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if "model.json" not in zf.namelist() or "model.tflite" not in zf.namelist():
                        continue
                    meta = json.loads(zf.read("model.json").decode("utf-8"))
                    tflite_bytes = zf.read("model.tflite")
                seen.add(zip_path.name)
                catalog.append({
                    "filename": zip_path.name,
                    "path": str(zip_path),
                    "downloadPath": f"/models/{zip_path.name}",
                    "sizeBytes": int(zip_path.stat().st_size),
                    "tfliteBytes": len(tflite_bytes),
                    "name": meta.get("name", zip_path.stem),
                    "version": meta.get("version", "1.0.0"),
                    "task": meta.get("task", "detect"),
                    "sha256": meta.get("sha256") or hashlib.sha256(tflite_bytes).hexdigest(),
                    "notes": meta.get("notes", ""),
                })
            except Exception:
                continue
    return catalog


def list_uploaded_telemetry_logs(upload_dir: Path | None = None) -> list[dict[str, Any]]:
    """列出上传目录中的全部黑匣子 `.jsonl` 日志摘要。"""
    target_dir = upload_dir if upload_dir is not None else DEFAULT_UPLOAD_DIR
    if not target_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(target_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            text = p.read_text(encoding="utf-8")
            n_lines = sum(1 for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#"))
            st = p.stat()
            items.append({
                "filename": p.name,
                "savedPath": str(p),
                "sizeBytes": int(st.st_size),
                "frameCount": int(n_lines),
                "modifiedTime": round(float(st.st_mtime), 3),
            })
        except Exception:
            continue
    return items


class _PushRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, search_dirs: list[Path], upload_dir: Path, **kwargs):
        self.search_dirs = search_dirs
        self.upload_dir = upload_dir
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        return  # 安静输出

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", "/health"):
            self._send_json({"ok": True, "service": "rig-hover-model-push", "port": self.server.server_port})
            return

        if path == "/models.json":
            items = [{k: v for k, v in item.items() if k != "path"} for item in discover_model_bundles(self.search_dirs)]
            self._send_json({"schemaVersion": 1, "models": items})
            return

        if path == "/strategies.json":
            presets = [s.to_dict() for s in builtin_presets()]
            self._send_json({"schemaVersion": 1, "strategies": presets})
            return

        if path == "/telemetry/list":
            logs = list_uploaded_telemetry_logs(self.upload_dir)
            self._send_json({"ok": True, "uploadDir": str(self.upload_dir), "logs": logs})
            return

        if path.startswith("/models/"):
            fname = Path(path[len("/models/"):]).name
            for item in discover_model_bundles(self.search_dirs):
                if item["filename"] == fname:
                    raw = Path(item["path"]).read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(raw)
                    return
            self._send_json({"ok": False, "error": f"model not found: {fname}"}, HTTPStatus.NOT_FOUND)
            return

        self._send_json({"ok": False, "error": f"not found: {path}"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(self.path.split("?", 1)[0])
        if path == "/telemetry/upload":
            try:
                content_len = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                content_len = 0
            if content_len <= 0:
                self._send_json({"ok": False, "error": "empty telemetry body"}, HTTPStatus.BAD_REQUEST)
                return

            raw_bytes = self.rfile.read(content_len)
            try:
                text = raw_bytes.decode("utf-8")
                records = parse_blackbox_jsonl(text)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"invalid JSONL payload: {exc}"}, HTTPStatus.BAD_REQUEST)
                return

            self.upload_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_file = self.upload_dir / f"log_{ts}.jsonl"
            # 防止极快并发请求重名
            suffix = 1
            while out_file.exists():
                out_file = self.upload_dir / f"log_{ts}_{suffix}.jsonl"
                suffix += 1
            out_file.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

            calibration = auto_calibrate_system_id(records)
            self._send_json(
                {
                    "ok": True,
                    "filename": out_file.name,
                    "savedPath": str(out_file),
                    "frameCount": len(records),
                    "calibration": calibration,
                },
                HTTPStatus.OK,
            )
            return

        self._send_json({"ok": False, "error": f"not found: {path}"}, HTTPStatus.NOT_FOUND)


def start_model_push_server(
    host: str = "0.0.0.0",
    port: int = 8790,
    search_dirs: list[Path] | None = None,
    upload_dir: Path | None = None,
) -> ThreadingHTTPServer:
    dirs = search_dirs if search_dirs is not None else DEFAULT_SEARCH_DIRS
    up_dir = upload_dir if upload_dir is not None else DEFAULT_UPLOAD_DIR
    handler_cls = partial(_PushRequestHandler, search_dirs=dirs, upload_dir=up_dir)
    return ThreadingHTTPServer((host, port), handler_cls)


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG-Hover 局域网模型下发与遥测数据飞轮服务 (Path C)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    srv = start_model_push_server(host=args.host, port=args.port)
    print(f"RIG-Hover Model & Telemetry Flywheel Server running at http://{args.host}:{args.port}/models.json")
    for m in discover_model_bundles():
        print(f"  - [{m['task']}] {m['name']} ({m['filename']}, {m['sizeBytes']} B)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
