# -*- coding: utf-8 -*-
"""test_frame_source.py -- 两种取流结构的自检（用本地 mock 服务端）。

验证的是**正确性**：帧边界解析、长连接不漏帧、上层能直接吃这个源。
**不验证延迟收益**——localhost 上没有 WiFi，也不复现机器人 httpd 的单 worker 串行，
加速比只能在真机上量。这条写在文档里，也写在这个脚本的输出里。

Run:  python test_frame_source.py
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
sys.path.insert(0, TOOLS)

import frame_source as fs  # noqa: E402

N_PASS = 0
N_FAIL = 0
PORT = 18123


def check(name: str, cond: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {name}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def main() -> int:
    print("== 启动本地 mock 服务端 ==")
    try:
        from mock_mjpeg_server import MockServer
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  无法导入 mock 服务端：{exc}")
        return 0
    srv = MockServer(PORT, fps=20.0, quality=80, size=240)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.4)
    check("mock 服务端已监听", th.is_alive())

    print("\n== HttpPollSource：单帧请求（当前部署方式）==")
    poll = fs.HttpPollSource(f"http://127.0.0.1:{PORT}/capture")
    img = poll.fetch()
    check("能取到并解码一帧", img is not None and img.shape[0] > 0,
          str(None if img is None else img.shape))
    check("帧尺寸是 240x240",
          img is not None and img.shape[:2] == (240, 240),
          str(None if img is None else img.shape))
    check("统计计数增加", poll.stats.frames == 1, repr(poll.stats))

    bad = fs.HttpPollSource("http://127.0.0.1:1/capture", timeout=0.4)
    check("连不上时返回 None 并计错误",
          bad.fetch() is None and bad.stats.errors >= 1, repr(bad.stats))

    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.0:
        if poll.fetch() is not None:
            n += 1
    poll_fps = n / (time.perf_counter() - t0)
    print(f"  HTTP 拉取：{n} 帧 / 1.0 s  ≈ {poll_fps:.1f} fps（localhost，非真机）")

    print("\n== MjpegSource：持久连接推流 ==")
    mj = fs.MjpegSource("127.0.0.1", PORT, "/stream", timeout=3.0)
    mj.start()
    time.sleep(1.0)
    img2, t2 = mj.latest()
    check("能收到帧", img2 is not None, repr(mj.stats))
    check("帧尺寸正确", img2 is not None and img2.shape[:2] == (240, 240),
          str(None if img2 is None else img2.shape))
    check("帧是新鲜的（<0.5s）",
          t2 > 0 and (time.perf_counter() - t2) < 0.5,
          f"age={time.perf_counter() - t2:.3f}s")

    before = mj.stats.frames
    time.sleep(1.0)
    got = mj.stats.frames - before
    print(f"  MJPEG 接收：1.0 s 内 {got} 帧 ≈ {got:.1f} fps"
          f"（mock 设为 20 fps）")
    check("接收速率接近服务端 20 fps（≥12）", got >= 12, f"got={got}")

    # 帧必须是"最新"的，不是队列里积压的旧帧
    ages = []
    for _ in range(10):
        f, t = mj.latest()
        if f is None:
            continue
        ages.append(time.perf_counter() - t)
        time.sleep(0.03)
    check("latest() 不给积压的旧帧（age < 0.3s）",
          ages and max(ages) < 0.3, f"max_age={max(ages) if ages else None}")

    # wait_fresh 阻塞到新帧
    _, t_prev = mj.latest()
    f3, t3 = mj.wait_fresh(t_prev, timeout=1.0)
    check("wait_fresh 能等到更新的一帧", f3 is not None and t3 > t_prev,
          f"{t_prev} -> {t3}")

    mj.stop()
    check("stop() 后线程已退出", mj._th is None or not mj._th.is_alive())

    print("\n== 没有 Content-Length 时的回退解析 ==")
    class _FakeFile:
        def __init__(self, data: bytes):
            self._b = io.BytesIO(data)

        def read(self, n: int = -1) -> bytes:
            return self._b.read(n)

        def readline(self) -> bytes:
            return self._b.readline()

    payload = b"\xff\xd8JPEGDATA\xff\xd9"
    stream = payload + b"\r\n--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    mj2 = fs.MjpegSource("127.0.0.1", PORT, boundary="frame")
    got_bytes = mj2._read_until_boundary(_FakeFile(stream))
    check("按 boundary 找到帧尾", got_bytes == payload, repr(got_bytes[:40]))

    print("\n== 上层感知链能不能直接吃这个源 ==")
    mj3 = fs.MjpegSource("127.0.0.1", PORT, "/stream", timeout=3.0)
    mj3.start()
    time.sleep(0.6)
    try:
        from run_perception import PerceptionPipeline
        pipe = PerceptionPipeline(verbose=False)
        f, _t = mj3.latest()
        out = pipe(f) if f is not None else None
        check("PerceptionPipeline 能处理流里的帧", out is not None,
              "no frame" if f is None else "")
        if out is not None:
            check("输出字段齐全（含场景 v2 新字段）",
                  hasattr(out, "theta_face") and hasattr(out, "knock_fired"))
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP  感知管线不可用：{exc}")
    finally:
        mj3.stop()

    srv.stop()

    print("\n" + "=" * 66)
    print(f"{N_PASS} PASS / {N_FAIL} FAIL")
    print("\n⚠️  本测试只证明两种取流结构都**正确**。")
    print("    它**不能**证明 MJPEG 更快：localhost 没有 WiFi，")
    print("    也不复现机器人 httpd 的单 worker 串行。加速比必须上真机量。")
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
