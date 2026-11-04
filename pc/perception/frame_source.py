# -*- coding: utf-8 -*-
"""frame_source.py -- 取流的两种结构，可切换，便于对比和迁移。

现在部署用的是"每帧发一次 HTTP 请求"（:class:`HttpPollSource`）：
PC 发 ``GET /capture``，机器人拍一帧、在板上 JPEG 编码、回 9.3 KB。实测这一来一回
**51.6 ms**，占端到端 60.8 ms 的 85%；而 PC 端整条感知（检测+适配+手势）只要
9.2 ms。也就是说延迟的大头不在算力，在这个请求/响应结构。

:class:`MjpegSource` 是另一条结构：**一次 TCP 连接，服务端连续推帧**
（``multipart/x-mixed-replace``）。它一次性消掉每帧的请求往返和排队。

两个源都实现同一个 :meth:`FrameSource.latest` 接口，上层（录制、实时跟随）
换源不需要改逻辑。

协议细节：MJPEG 流的每个 part 形如::

    --<boundary>\r\n
    Content-Type: image/jpeg\r\n
    Content-Length: <n>\r\n
    \r\n
    <n bytes of JPEG>\r\n

``Content-Length`` 不是每个摄像头都发，所以解析器两条路都支持：有长度就按长度读，
没有就按下一个 boundary 找。

⚠️ **延迟收益只能在真机上量。** 本地 mock 跑在 localhost，不经过 WiFi，
也不复现机器人 httpd 的单 worker 串行，所以只能用来看"结构对不对、帧有没有丢"，
不能用它给出加速比。
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np

__all__ = ["FrameSource", "HttpPollSource", "MjpegSource", "Stats"]


class Stats:
    """取流的简单统计（线程安全由调用方保证：只在单线程读）。"""

    def __init__(self):
        self.frames = 0
        self.bytes = 0
        self.errors = 0
        self.t0 = time.perf_counter()

    def fps(self) -> float:
        dt = max(1e-9, time.perf_counter() - self.t0)
        return self.frames / dt

    def __repr__(self) -> str:
        return (f"frames={self.frames} bytes={self.bytes} "
                f"errors={self.errors} fps={self.fps():.1f}")


class FrameSource:
    """取流源的共同接口。"""

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def latest(self):
        """返回 ``(frame_bgr, t_capture)``；还没有帧时返回 ``(None, -1.0)``。

        ``t_capture`` 是 :func:`time.perf_counter` 时间戳，用于统计帧年龄。
        """
        raise NotImplementedError

    @property
    def stats(self) -> Stats:
        raise NotImplementedError


class HttpPollSource(FrameSource):
    """当前部署的取流方式：每帧一次 ``GET <url>``。

    Args:
        url: 形如 ``http://192.168.137.50/capture``
        timeout: 单次请求超时（秒）
    """

    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout
        self._st = Stats()
        self._closed = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._closed = True

    @property
    def stats(self) -> Stats:
        return self._st

    def fetch(self):
        """主动取一帧（这个源是拉模式，没有后台线程）。"""
        import urllib.request
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as r:
                buf = r.read()
        except OSError:
            self._st.errors += 1
            return None
        if not buf:
            self._st.errors += 1
            return None
        img = _decode_jpeg(buf)
        if img is None:
            self._st.errors += 1
            return None
        self._st.frames += 1
        self._st.bytes += len(buf)
        return img

    def latest(self):
        img = self.fetch()
        if img is None:
            return None, -1.0
        return img, time.perf_counter()


def _decode_jpeg(buf: bytes):
    import cv2
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class MjpegSource(FrameSource):
    """持久连接的 MJPEG 取流：后台线程持续收帧，主线程永远拿到**最新**的。

    Args:
        host / port: 服务端
        path: 请求路径，默认 ``/stream``
        boundary: multipart 分隔符，默认 ``frame``
        reconnect_s: 断线后重连间隔（秒）
    """

    def __init__(self, host: str, port: int = 81, path: str = "/stream",
                 boundary: str = "frame", timeout: float = 5.0,
                 reconnect_s: float = 0.5):
        self.host = host
        self.port = int(port)
        self.path = path
        self.boundary = boundary.encode()
        self.timeout = timeout
        self.reconnect_s = reconnect_s
        self._st = Stats()
        self._lock = threading.Lock()
        self._frame = None
        self._t = -1.0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._th is not None:
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, name="mjpeg", daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
            self._th = None

    @property
    def stats(self) -> Stats:
        return self._st

    def latest(self):
        with self._lock:
            f, t = self._frame, self._t
        return f, t

    def wait_fresh(self, after_t: float, timeout: float = 2.0):
        """等一帧比 ``after_t`` 更新的帧（给"按帧驱动"的循环用）。"""
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            with self._lock:
                f, t = self._frame, self._t
            if f is not None and t > after_t:
                return f, t
            time.sleep(0.001)
        return None, -1.0

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._session()
            except OSError:
                self._st.errors += 1
            if not self._stop.is_set():
                time.sleep(self.reconnect_s)

    def _session(self) -> None:
        with socket.create_connection((self.host, self.port),
                                      timeout=self.timeout) as s:
            s.sendall(
                (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}\r\n"
                 "Connection: close\r\n\r\n").encode())
            s.settimeout(self.timeout)
            f = s.makefile("rb")
            # 吃掉响应头
            while True:
                line = f.readline()
                if not line:
                    return
                if line in (b"\r\n", b"\n"):
                    break
            # multipart 循环
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    return
                if not line.startswith(b"--"):
                    continue
                length = 0
                while True:
                    h = f.readline()
                    if not h or h in (b"\r\n", b"\n"):
                        break
                    if h.lower().startswith(b"content-length:"):
                        length = int(h.split(b":", 1)[1].strip())
                if length > 0:
                    data = f.read(length)
                else:
                    data = self._read_until_boundary(f)
                if not data:
                    continue
                img = _decode_jpeg(data)
                if img is None:
                    self._st.errors += 1
                    continue
                with self._lock:
                    self._frame = img
                    self._t = time.perf_counter()
                self._st.frames += 1
                self._st.bytes += len(data)

    def _read_until_boundary(self, f) -> bytes:
        """没有 Content-Length 时，按下一个 boundary 找帧尾。"""
        buf = bytearray()
        marker = b"--" + self.boundary
        while not self._stop.is_set():
            chunk = f.read(4096)
            if not chunk:
                return bytes(buf)
            buf.extend(chunk)
            i = buf.find(marker)
            if i >= 0:
                return bytes(buf[:i]).rstrip(b"\r\n")
        return bytes(buf)
