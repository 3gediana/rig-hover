# -*- coding: utf-8 -*-
"""mock_mjpeg_server.py -- 本地 MJPEG / 单帧取流服务端，用来在没有机器人的情况下
验证 :mod:`frame_source` 的两个客户端。

它不是一个测速工具。localhost 上既没有 WiFi，也不复现机器人 httpd 的单 worker
串行，所以**不能**用它给出"换成 MJPEG 能快多少"。它能验证的是：

  * multipart 帧边界解析对不对（有/无 Content-Length 两条路）
  * 长连接会不会漏帧、错帧
  * 上层感知管线能不能直接吃这个源

用法::

    python mock_mjpeg_server.py --port 8123 --fps 15
    # 然后另开一个进程：
    python frame_source_smoke.py --host 127.0.0.1 --port 8123
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import socket
import sys
import threading
import time

import cv2
import numpy as np

PERCEPTION = r"D:\robot_work\pc\perception"
MANIFEST = r"D:\robot_work\pc\gesture\data\manifest.csv"


def load_frames(n: int = 60, size: int = 240):
    """从 HaGRID 里取一批真实照片，缩放到机器人相机的 240x240 视野。"""
    if not os.path.isfile(MANIFEST):
        print(f"manifest 不存在：{MANIFEST}；改用合成色块")
        return [np.full((size, size, 3), 60 + i % 150, np.uint8) for i in range(n)]
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    random.Random(0).shuffle(rows)
    frames = []
    for r in rows:
        if len(frames) >= n:
            break
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        s = float(size) / max(h, w)
        img = cv2.resize(img, (max(int(w * s), 8), max(int(h * s), 8)),
                         interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size, 3), 110, np.uint8)
        y0, x0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        frames.append(canvas)
    return frames


def encode(img, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class MockServer:
    def __init__(self, port: int, fps: float, quality: int, size: int):
        self.frames = [encode(f, quality) for f in load_frames(size=size)]
        self.port = port
        self.period = 1.0 / max(fps, 0.1)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self._sock.listen(4)
        self._stop = threading.Event()
        self.n_capture = 0
        self.n_stream = 0

    def serve_forever(self):
        print(f"mock server on 127.0.0.1:{self.port}  "
              f"({len(self.frames)} frames, {1/self.period:.0f} fps)")
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def stop(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass

    def _handle(self, conn: socket.socket):
        try:
            req = conn.recv(1024).decode("latin-1")
            path = req.split(" ")[1] if " " in req else "/"
            if path.startswith("/stream"):
                self._stream(conn)
            else:
                self._capture(conn)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _capture(self, conn: socket.socket):
        body = self.frames[self.n_capture % len(self.frames)]
        self.n_capture += 1
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: image/jpeg\r\n"
                     b"Cache-Control: no-store\r\n"
                     b"Connection: close\r\n"
                     + f"Content-Length: {len(body)}\r\n\r\n".encode()
                     + body)

    def _stream(self, conn: socket.socket, max_frames: int = 0):
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                     b"Cache-Control: no-store\r\n"
                     b"Connection: close\r\n\r\n")
        i = 0
        t_next = time.perf_counter()
        while not self._stop.is_set():
            body = self.frames[i % len(self.frames)]
            i += 1
            self.n_stream += 1
            try:
                conn.sendall(
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body + b"\r\n")
            except OSError:
                return
            if max_frames and i >= max_frames:
                return
            t_next += self.period
            d = t_next - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                t_next = time.perf_counter()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--size", type=int, default=240)
    a = ap.parse_args()
    srv = MockServer(a.port, a.fps, a.quality, a.size)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
        print(f"\nserved capture={srv.n_capture} stream={srv.n_stream}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
