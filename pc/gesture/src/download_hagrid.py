# -*- coding: utf-8 -*-
"""download_hagrid.py — 并行分块下载 HaGRID 30k 采样集（1.0 GB）

为什么不是 ``curl -O``：

1. Windows 的 curl 走 schannel，对 HF 的 xet CDN 握手直接失败
   （``schannel: failed to receive handshake``），Python 的 TLS 栈正常。
2. 单连接会被限速且速度随时间衰减（实测 4 MB/s → 0.03 MB/s），
   而新开的连接又是满速。所以按 16 MB 分块、多线程各下各的，块内速度掉下去
   也无所谓——下一块是新连接。

块文件 ``<dest>.part/<index>.bin``，可断点续传；全部完成后拼接。
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "data", "raw", "hagrid-sample-30k-384p.zip")

MIRRORS = [
    "https://huggingface.co",
    "https://hf-mirror.com",
]
PATH = "/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip"
CHUNK = 16 << 20
WORKERS = 8


def probe_size(timeout=40):
    for host in MIRRORS:
        try:
            req = urllib.request.Request(
                host + PATH, headers={"Range": "bytes=0-0",
                                      "User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                cr = r.headers.get("Content-Range") or ""
                if "/" in cr:
                    return int(cr.split("/")[-1])
        except Exception as exc:
            print("probe %s failed: %s" % (host, exc), flush=True)
    raise RuntimeError("cannot probe size")


def fetch_chunk(args):
    idx, start, end, size = args
    part_dir = DEST + ".part"
    fp = os.path.join(part_dir, "%04d.bin" % idx)
    want = end - start + 1
    if os.path.exists(fp) and os.path.getsize(fp) == want:
        return idx, want, True
    host = MIRRORS[idx % len(MIRRORS)]
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                host + PATH, headers={"Range": "bytes=%d-%d" % (start, end),
                                      "User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != want:
                raise IOError("short read %d != %d" % (len(data), want))
            with open(fp, "wb") as fh:
                fh.write(data)
            return idx, want, False
        except Exception as exc:
            if attempt == 3:
                print("chunk %d failed: %s" % (idx, exc), flush=True)
                return idx, 0, False
            host = MIRRORS[(idx + attempt + 1) % len(MIRRORS)]
            time.sleep(1.0 + attempt)
    return idx, 0, False


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    part_dir = DEST + ".part"
    os.makedirs(part_dir, exist_ok=True)
    size = probe_size()
    print("size = %d bytes (%.1f MB)" % (size, size / 1e6), flush=True)

    jobs = []
    idx = 0
    for start in range(0, size, CHUNK):
        end = min(size - 1, start + CHUNK - 1)
        jobs.append((idx, start, end, size))
        idx += 1

    t0 = time.time()
    done_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(fetch_chunk, j) for j in jobs]
        for k, fut in enumerate(cf.as_completed(futs)):
            _, got, cached = fut.result()
            done_bytes += got
            if (k + 1) % 5 == 0 or k + 1 == len(futs):
                el = max(time.time() - t0, 1e-6)
                print("  %3d/%3d chunks  %7.1f/%.1f MB  %.2f MB/s%s"
                      % (k + 1, len(futs), done_bytes / 1e6, size / 1e6,
                         done_bytes / 1e6 / el, "  (cached)" if cached else ""),
                      flush=True)

    if os.path.exists(DEST):
        os.remove(DEST)
    with open(DEST, "wb") as out:
        for j in jobs:
            fp = os.path.join(part_dir, "%04d.bin" % j[0])
            with open(fp, "rb") as fh:
                shutil.copyfileobj(fh, out)
    got = os.path.getsize(DEST)
    print("assembled %s  %d bytes (%.1f MB)" % (DEST, got, got / 1e6),
          flush=True)
    if got != size:
        print("SIZE MISMATCH", flush=True)
        return 1
    shutil.rmtree(part_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
