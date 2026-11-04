# -*- coding: utf-8 -*-
"""Sequential resumable downloader with short timeouts.

Simpler and more predictable than the threaded one: one connection, 32 MB
ranges, 60 s timeout, retry with backoff, resume from whatever is already on
disk. Prints progress so a stall is visible instead of silent.
"""
from __future__ import annotations

import os
import sys
import time

import requests

UA = {"User-Agent": "curl/8.5.0"}
CHUNK = 32 * 1024 * 1024


def _total_size(url, timeout=30):
    """总字节数。HEAD 在 ModelScope 的 302 之后不带 Content-Length，所以用一个
    `Range: bytes=0-0` 的 GET 从 Content-Range 里读。"""
    r = requests.get(url, headers=dict(UA, **{"Range": "bytes=0-0"}),
                     stream=True, timeout=timeout)
    r.raise_for_status()
    cr = r.headers.get("Content-Range") or ""
    cl = r.headers.get("Content-Length") or "0"
    r.close()
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    return int(cl)


def grab(url, dest, chunk=CHUNK, timeout=60, retries=40):
    if os.path.exists(dest):
        done = os.path.getsize(dest)
    else:
        done = 0
    total = _total_size(url)
    print("target %d B, have %d B (%.1f%%)"
          % (total, done, 100.0 * done / max(total, 1)), flush=True)
    if total <= 0:
        raise SystemExit("拿不到文件大小")
    t0 = time.time()
    last = t0
    stall = 0
    while total and done < total:
        end = min(done + chunk, total) - 1
        headers = dict(UA)
        headers["Range"] = "bytes=%d-%d" % (done, end)
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=timeout)
            r.raise_for_status()
            if r.status_code == 206:
                with open(dest, "ab") as fh:
                    for blk in r.iter_content(262144):
                        if blk:
                            fh.write(blk)
                            done += len(blk)
            else:
                # server ignored Range: restart from scratch once
                print("  server ignored Range (%d), restarting" % r.status_code,
                      flush=True)
                with open(dest, "wb") as fh:
                    for blk in r.iter_content(262144):
                        if blk:
                            fh.write(blk)
                            done = fh.tell()
            r.close()
            stall = 0
        except Exception as exc:  # noqa: BLE001 - network flake
            stall += 1
            print("  retry %d: %s" % (stall, str(exc)[:100]), flush=True)
            time.sleep(min(2.0 * stall, 15.0))
            if stall > retries:
                raise SystemExit("too many retries, %d B done" % done)
        now = time.time()
        if now - last > 15:
            rate = done / max(now - t0, 1e-6) / 1e6
            print("  %6.2f / %6.2f GB  %5.2f MB/s"
                  % (done / 1e9, total / 1e9, rate), flush=True)
            last = now
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    print("DONE %s  %d B  %.1f s" % (dest, size, time.time() - t0), flush=True)
    return size


def main() -> int:
    base = ("https://www.modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct"
            "/resolve/master/")
    dest_dir = r"G:\ms\Qwen3-VL-2B-Instruct"
    os.makedirs(dest_dir, exist_ok=True)
    files = ["model.safetensors", "config.json", "preprocessor_config.json",
             "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "chat_template.json",
             "video_preprocessor_config.json"]
    for name in files:
        dest = os.path.join(dest_dir, name)
        try:
            grab(base + name, dest)
        except SystemExit as exc:
            print("FAILED %s: %s" % (name, exc))
            return 2
        except Exception as exc:  # noqa: BLE001
            print("FAILED %s: %s" % (name, str(exc)[:160]))
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
