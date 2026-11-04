# -*- coding: utf-8 -*-
"""_fetch_weights.py -- 多线程断点续传的模型权重下载器。

为什么自己写：`huggingface_hub` 在这台机器上反复 SSL EOF，ModelScope SDK 单线程
只有 1.5 MB/s，但两家对 **Range 请求**都给得很好（实测 ModelScope 8.2 MB/s、
HF 4.4 MB/s）。所以直接按区间分片、多连接并发、失败重试、中断可续。

    python _fetch_weights.py --repo Qwen/Qwen3-VL-4B-Instruct \
        --dest G:\\ms\\Qwen3-VL-4B-Instruct --hub modelscope --workers 8

只依赖标准库 + requests。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import requests

CHUNK = 4 * 1024 * 1024
UA = {"User-Agent": "curl/8.5.0"}

HUBS = {
    "modelscope": "https://www.modelscope.cn/models/{repo}/resolve/master/{f}",
    "hf": "https://huggingface.co/{repo}/resolve/main/{f}",
}


def list_files(repo: str, hub: str) -> list:
    """返回 [(filename, size), ...]，只取根目录下的文件。"""
    if hub == "modelscope":
        url = ("https://www.modelscope.cn/api/v1/models/%s/repo/files"
               "?Revision=master&Root=" % repo)
        r = requests.get(url, timeout=30, headers=UA)
        r.raise_for_status()
        data = r.json()["Data"]["Files"]
    else:
        url = "https://huggingface.co/api/models/%s" % repo
        r = requests.get(url, timeout=30, headers=UA)
        r.raise_for_status()
        data = r.json()["siblings"]
    out = []
    for f in data:
        if f.get("Type") == "tree":          # ModelScope 目录项
            continue
        name = f.get("Name") or f.get("rfilename")
        size = f.get("Size") or f.get("size") or 0
        if not name or "/" in name:
            continue
        out.append((name, int(size)))
    return out


def _download_chunk(url, start, end, path, part, retries=6):
    """下 [start, end] 到 part 文件的对应偏移。已存在则跳过。"""
    want = end - start + 1
    for attempt in range(retries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if have >= want:
                return True
            headers = dict(UA)
            headers["Range"] = "bytes=%d-%d" % (start + have, end)
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part, "ab") as fh:
                    for blk in r.iter_content(262144):
                        if blk:
                            fh.write(blk)
            return os.path.getsize(part) >= want
        except Exception as exc:  # noqa: BLE001 - 网络抖动要重试
            if attempt == retries - 1:
                print("    chunk %d-%d FAILED: %s" % (start, end, str(exc)[:90]),
                      file=sys.stderr)
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def fetch(url, dest, workers=8):
    """并发分片下载单个文件，支持断点续传。"""
    tmpdir = dest + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    head = requests.head(url, headers=UA, timeout=30, allow_redirects=True)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length") or 0)
    accept = head.headers.get("Accept-Ranges", "")
    if total <= 0 or "bytes" not in accept:
        # 服务端不支持分片：退回单流
        with requests.get(url, headers=UA, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for blk in r.iter_content(262144):
                    fh.write(blk)
        return os.path.getsize(dest)

    spans = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
    lock = threading.Semaphore(workers)
    threads = []
    done = {"ok": 0, "n": len(spans)}

    def work(s, e, idx):
        with lock:
            part = os.path.join(tmpdir, "%06d" % idx)
            if _download_chunk(url, s, e, dest, part):
                done["ok"] += 1

    t0 = time.time()
    for i, (s, e) in enumerate(spans):
        th = threading.Thread(target=work, args=(s, e, i), daemon=True)
        th.start()
        threads.append(th)
        if len([t for t in threads if t.is_alive()]) >= workers * 2:
            time.sleep(0.01)
    for th in threads:
        th.join()
    if done["ok"] != done["n"]:
        raise RuntimeError("%s: %d/%d 分片失败"
                           % (os.path.basename(dest), done["n"] - done["ok"],
                              done["n"]))
    with open(dest, "wb") as out:
        for i in range(len(spans)):
            with open(os.path.join(tmpdir, "%06d" % i), "rb") as fh:
                while True:
                    blk = fh.read(1 << 20)
                    if not blk:
                        break
                    out.write(blk)
    for i in range(len(spans)):
        os.remove(os.path.join(tmpdir, "%06d" % i))
    os.rmdir(tmpdir)
    dt = time.time() - t0
    print("  %-40s %7.2f MB  %5.1f s  %5.2f MB/s"
          % (os.path.basename(dest), total / 1e6, dt, total / 1e6 / max(dt, 1e-6)))
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--hub", default="modelscope", choices=sorted(HUBS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-larger-than", type=int, default=0,
                    help="只下小于该字节数的文件（GB 单位由 --skip-gb 给）")
    ap.add_argument("--only", default="", help="只下匹配这个后缀的文件，如 .safetensors")
    a = ap.parse_args(argv)

    os.makedirs(a.dest, exist_ok=True)
    files = list_files(a.repo, a.hub)
    print("repo %s @ %s: %d files, %.2f GB"
          % (a.repo, a.hub, len(files), sum(s for _, s in files) / 1e9))
    t0 = time.time()
    for name, size in files:
        if a.only and not name.endswith(a.only):
            continue
        if a.skip_larger_than and size > a.skip_larger_than:
            continue
        dest = os.path.join(a.dest, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size and size > 0:
            print("  skip %s (already %d B)" % (name, size))
            continue
        url = HUBS[a.hub].format(repo=a.repo, f=name)
        try:
            fetch(url, dest, a.workers)
        except Exception as exc:  # noqa: BLE001
            print("  %s FAILED: %s" % (name, str(exc)[:160]), file=sys.stderr)
            return 2
    print("done in %.1f s -> %s" % (time.time() - t0, a.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
