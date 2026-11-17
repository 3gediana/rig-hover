# -*- coding: utf-8 -*-
"""model_push_server.py — PC 局域网模型与策略一键下发服务 (Android 端路径 C 拉取对接)。

默认监听 0.0.0.0:8790，在手机连接同一热点 (qasd) 后可一键从电脑拉取最新训练好的模型 .zip 与策略 JSON：
  - GET /models.json            列出当前可用模型包 (.zip) 与 SHA-256、task、大小、下载路径
  - GET /models/<filename>.zip  直接下载规范化模型包 (model.tflite + model.json)
  - GET /strategies.json        返回与 PC 完全同步的策略预设列表 (供 App 一键同步)
"""
from __future__ import annotations

import argparse
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


DEFAULT_SEARCH_DIRS = [
    Path("D:/robot_work/runs_end_side/hover_rl_policy"),
    Path("D:/robot_work/runs_knock/double_knock_20261115/bundle"),
]


def discover_model_bundles(search_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    """扫描目录中合法的 *-bundle.zip 模型包，读取其中的 model.json 与真实 SHA-256。"""
    dirs = search_dirs if search_dirs is not None else DEFAULT_SEARCH_DIRS
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in dirs:
        if not root.exists():
            continue
        for zip_path in sorted(root.glob("*.zip")):
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


class _PushRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, search_dirs: list[Path], **kwargs):
        self.search_dirs = search_dirs
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        return  # 安静输出

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

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

        if path.startswith("/models/"):
            fname = Path(path[len("/models/"):]).name
            for item in discover_model_bundles(self.search_dirs):
                if item["filename"] == fname:
                    raw = Path(item["path"]).read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                    self.end_headers()
                    self.wfile.write(raw)
                    return
            self._send_json({"ok": False, "error": f"model not found: {fname}"}, HTTPStatus.NOT_FOUND)
            return

        self._send_json({"ok": False, "error": f"not found: {path}"}, HTTPStatus.NOT_FOUND)


def start_model_push_server(
    host: str = "0.0.0.0",
    port: int = 8790,
    search_dirs: list[Path] | None = None,
) -> ThreadingHTTPServer:
    dirs = search_dirs if search_dirs is not None else DEFAULT_SEARCH_DIRS
    handler_cls = partial(_PushRequestHandler, search_dirs=dirs)
    return ThreadingHTTPServer((host, port), handler_cls)


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG-Hover 局域网模型与策略推送服务 (Path C)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    srv = start_model_push_server(host=args.host, port=args.port)
    print(f"RIG-Hover Model & Strategy Push Server running at http://{args.host}:{args.port}/models.json")
    for m in discover_model_bundles():
        print(f"  - [{m['task']}] {m['name']} ({m['filename']}, {m['sizeBytes']} B)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
