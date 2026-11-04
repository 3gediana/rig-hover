# -*- coding: utf-8 -*-
"""http_floor_probe.py -- HTTP 每请求 50 ms 到底卡在哪一段？

为什么必须问这一句
==================

`RESULTS.md` §9 的实测里有一处**自相矛盾**：

| 端点 | 响应体 | ttfb 中位 |
|---|---|---|
| `/`（静态 HTML） | **5,232 B** | **8.9 ms** |
| `/api/data` | 236 B | 56.8 ms |
| `/api/set` | **11 B** | 52.2 ms |
| `/capture` | 9,277 B JPEG | 51.6 ms |

同一台 httpd、同一条连接、同一个客户端。**负载大 20 倍的 `/` 反而快 6 倍**，
返回 11 字节的 `/api/set` 却要 52 ms。

而且固件里 `CONFIG_HTTPD_ERR_RESP_NO_DELAY=y`（只有**错误响应**开了 TCP_NODELAY），
`sdkconfig` 里 **`CONFIG_LWIP_TCP_NODELAY` 根本没设** → 普通响应走 Nagle。

Nagle 只放行满 MSS（≈1460 B）的整段，不满的段会攒着等对端确认。代进去：

* `/` 5,232 B → 有满段 → 立刻走 → **快**
* `/api/set` 11 B、`/feat` ~300 B、`/status` 780 B → 全是不满段 → **攒着** → 慢

**本脚本就是去证实/证伪这一条。** 它用裸 socket，把一次请求拆成四段时间：

```
t_connect   TCP 建连
t_send      写出请求
t_first     收到第一个字节     ← Nagle 卡响应体时，这一段会「快」
t_full      响应读完整         ← 而这一段「慢」= Nagle 特征
```

判读
====

| 现象 | 结论 | 改动 |
|---|---|---|
| `t_first` 快、`t_full` 慢 ≈ +40 ms | **Nagle**（服务端） | `CONFIG_LWIP_TCP_NODELAY=y` 或 setsockopt |
| `t_first` 本身就慢 | 请求处理路径 / WiFi | 见下面的分支 |
| 新建连接 ≫ 复用连接 | 连接开销 | 客户端显式 keep-alive |
| 只有 `/` 快 | 段大小相关 → 仍是 Nagle 家族 | 同上 |

Run:  python http_floor_probe.py --host 192.168.137.50
"""
from __future__ import annotations

import argparse
import socket
import statistics
import sys
import time


def raw_roundtrip(host, port, path, keep_sock=None, timeout=8.0):
    """裸 socket 发一次 GET，返回 (建连, 请求写, 首字节, 读完) 毫秒。

    ``keep_sock`` 传一个已有 socket 就复用它（测 keep-alive）。
    返回 (sock, timings, head) —— sock 供下一次复用，head 是响应头前几行。
    """
    t0 = time.perf_counter()
    sock = keep_sock
    if sock is None:
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    t1 = time.perf_counter()
    req = ("GET %s HTTP/1.1\r\nHost: %s\r\n%s\r\n"
           % (path, host, "" if keep_sock else "Connection: keep-alive\r\n"))
    sock.sendall(req.encode())
    t2 = time.perf_counter()

    chunks = []
    first = None
    sock.settimeout(timeout)
    while True:
        try:
            blk = sock.recv(4096)
        except socket.timeout:
            break
        if not blk:
            break
        if first is None:
            first = time.perf_counter()
        chunks.append(blk)
        # 头部 + Content-Length 齐了就收工，否则靠服务端关连接
        if b"\r\n\r\n" in blk:
            head_txt = b"".join(chunks)
            hdr, _, rest = head_txt.partition(b"\r\n\r\n")
            cl = None
            for line in hdr.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    cl = int(line.split(b":")[1].strip())
            if cl is not None and len(rest) >= cl:
                break
    t3 = time.perf_counter()

    timings = {
        "connect": (t1 - t0) * 1e3,
        "send": (t2 - t1) * 1e3,
        "first": (first - t2) * 1e3 if first else float("nan"),
        "full": (t3 - t2) * 1e3,
        "bytes": sum(len(c) for c in chunks),
    }
    head = b"".join(chunks).partition(b"\r\n\r\n")[0].decode("latin-1")
    keep = False
    for line in head.split("\r\n"):
        if line.lower().startswith("connection:") and "keep-alive" in line.lower():
            keep = True
    return (sock if keep else None), timings, head


def summarize(name, rows):
    if not rows:
        print("  %-16s （无数据）" % name)
        return
    def med(k):
        v = [r[k] for r in rows if r.get(k) == r.get(k)]   # 滤掉 nan
        return statistics.median(v) if v else float("nan")
    print("  %-16s 建连 %6.1f | 写 %5.1f | **首字节 %6.1f** | 读完 %6.1f | %5.0f B"
          "   n=%d"
          % (name, med("connect"), med("send"), med("first"), med("full"),
             med("bytes"), len(rows)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.137.50")
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--paths", default="/,/api/set,/status,/feat,/capture")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args(argv)

    paths = [p.strip() for p in a.paths.split(",") if p.strip()]
    print("=" * 92)
    print("HTTP 分段延迟探针   target=%s:%d   n=%d" % (a.host, a.port, a.n))
    print("=" * 92)
    print("  Nagle 特征 = 「首字节快、读完慢」，两者差 ≈ 40 ms")
    print()

    for path in paths:
        print("--- %s ---" % path)
        # 1) 每条连接只发一次
        rows_new = []
        for _ in range(a.n):
            s = None
            try:
                s, t, head = raw_roundtrip(a.host, a.port, path)
                rows_new.append(t)
            except Exception as exc:  # noqa: BLE001
                print("    失败：%s" % str(exc)[:90])
                break
            finally:
                if s is not None:
                    s.close()
        summarize("新连接各次", rows_new)

        # 2) 同一条连接连发（keep-alive）
        rows_keep = []
        sock = None
        try:
            sock, t, head = raw_roundtrip(a.host, a.port, path)
            rows_keep.append(t)
            for _ in range(a.n - 1):
                if sock is None:
                    break
                sock, t, _h = raw_roundtrip(a.host, a.port, path, keep_sock=sock)
                rows_keep.append(t)
        except Exception as exc:  # noqa: BLE001
            print("    keep-alive 失败：%s" % str(exc)[:90])
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        summarize("同连接连发", rows_keep[1:] if len(rows_keep) > 1 else rows_keep)

        # 3) 打印一次响应头，看服务端到底怎么说的
        try:
            s, _t, head = raw_roundtrip(a.host, a.port, path)
            if s:
                s.close()
            keep = [ln for ln in head.split("\r\n")
                    if ln.lower().startswith(("connection:", "content-length:",
                                              "transfer-encoding:"))]
            print("    响应头：%s" % ("  |  ".join(keep) if keep else "（无相关头）"))
        except Exception:  # noqa: BLE001
            pass
        print()

    print("=" * 92)
    print("怎么读")
    print("=" * 92)
    print("  · 「首字节」明显小于「读完」：Nagle 在卡响应体 → 服务端关 Nagle")
    print("      sdkconfig: CONFIG_LWIP_TCP_NODELAY=y      （当前未设）")
    print("      或 accept 后 setsockopt(IPPROTO_TCP, TCP_NODELAY)")
    print("  · 「首字节」本身就慢：请求处理或 WiFi 层 → 接着看下面两条")
    print("  · 新连接 ≫ 同连接：连接没被复用 → 客户端显式 keep-alive")
    print("  · 只有 / 快：与段大小相关，仍是 Nagle 家族")
    print()
    print("  另外两个 sdkconfig 项也偏小，顺手改掉：")
    print("      CONFIG_LWIP_TCP_SND_BUF_DEFAULT=5760  <  单帧 9277 B  → 提到 16384+")
    print("      CONFIG_LWIP_TCP_RECVMBOX_SIZE=6                        → 提到 32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
