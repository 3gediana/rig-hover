"""Resumable multi-connection HTTP range downloader + small HTTP helpers.

Why not ``requests``/``hf_hub_download`` only: we need (a) arbitrary hosts
(HF mirror, motchallenge, ETH DAVIS, ai2 S3) behind one interface, (b) *many*
parallel TCP connections because single-stream throughput from these hosts is
0.5-3 MB/s, and (c) chunk-level resumability that survives a hard kill, with a
sidecar JSON so a re-run skips finished chunks instead of restarting a 7 GB file.

Public API
----------
server_info(url)                     -> dict(ok, size, ranges, final_url, reason)
download_resumable(url, dst, ...)    -> Path (raises DownloadError on failure)
probe_speed(url, seconds=20)         -> float MB/s
hf_url(repo, path, repo_type)        -> str
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

MB = 1 << 20
GB = 1 << 30

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# The host machine has a system proxy configured that breaks some direct hosts
# (it 502s localhost and is flaky for motchallenge).  These hosts are reachable
# directly; bypass the proxy for them, use it for huggingface (which is often
# only reachable *through* the mirror / proxy).
_DIRECT_HOSTS = ("motchallenge.net", "data.vision.ee.ethz.ch", "ai2-public-datasets",
                 "s3.amazonaws.com", "huggingface.co", "hf-mirror.com", "cdn-lfs")


class DownloadError(RuntimeError):
    pass


def _opener_for(url: str) -> urllib.request.OpenerDirector:
    host = urllib.parse.urlparse(url).netloc.lower()
    if any(h in host for h in ("huggingface.co", "hf-mirror.com", "cdn-lfs", "hf.co")):
        # HF is reached through the normal path (env proxy / HF_ENDPOINT rewrite)
        return urllib.request.build_opener()
    handlers: List[urllib.request.BaseHandler] = [urllib.request.ProxyHandler({})]
    return urllib.request.build_opener(*handlers)


def _request(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
             timeout: int = 60) -> urllib.request.Request:
    h = {
        "User-Agent": _UA,
        "Accept": "*/*",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    if headers:
        h.update(headers)
    # urllib.urlopen leaks the connection object unless we close; use timeout on
    # the socket level via the opener's default.  We set it per call below.
    return urllib.request.Request(url, headers=h, method=method)


_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE  # some mirrors have broken chains


def _open(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
          timeout: int = 60):
    op = _opener_for(url)
    req = _request(url, method, headers)
    return op.open(req, timeout=timeout)


# --------------------------------------------------------------------------- #
# server capability probe
# --------------------------------------------------------------------------- #
def server_info(url: str, timeout: int = 45) -> Dict[str, object]:
    """HEAD (fallback: 1-byte ranged GET) to learn size + range support."""
    info: Dict[str, object] = {"ok": False, "size": -1, "ranges": False,
                               "final_url": url, "reason": ""}
    try:
        try:
            r = _open(url, "HEAD", timeout=timeout)
            code = r.status
            hdrs = r.headers
            r.close()
        except urllib.error.HTTPError as e:
            code = e.code
            hdrs = e.headers or {}
        if code >= 400 or "content-length" not in {k.lower() for k in hdrs.keys()}:
            # retry with a ranged GET (some servers refuse HEAD)
            r = _open(url, "GET", {"Range": "bytes=0-0"}, timeout=timeout)
            code = r.status
            hdrs = r.headers
            r.close()
        size = int(hdrs.get("Content-Length") or -1)
        if code == 206:
            m = re.search(r"/(\d+)$", hdrs.get("Content-Range", ""))
            if m:
                size = int(m.group(1))
        info.update(ok=(200 <= code < 300), size=size,
                    ranges=(code == 206 or "bytes" in (hdrs.get("Accept-Ranges") or "").lower()),
                    final_url=getattr(r, "url", url) if r else url)
        if not info["ok"]:
            info["reason"] = f"HTTP {code}"
        return info
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"{type(e).__name__}: {e}"
        return info


# --------------------------------------------------------------------------- #
# speed probe
# --------------------------------------------------------------------------- #
def probe_speed(url: str, seconds: float = 20.0, chunk: int = 8 * MB) -> float:
    """Stream a range of the file for ``seconds`` and return measured MB/s."""
    info = server_info(url)
    hdrs = {"Range": f"bytes=0-{chunk - 1}"} if info["ranges"] else None
    t0 = time.time()
    got = 0
    try:
        r = _open(url, "GET", hdrs, timeout=30)
        while time.time() - t0 < seconds:
            b = r.read(256 * 1024)
            if not b:
                break
            got += len(b)
            if got >= chunk and not info["ranges"]:
                break
        r.close()
    except Exception as e:  # noqa: BLE001
        print(f"  probe error: {type(e).__name__}: {e}")
    dt = max(1e-6, time.time() - t0)
    return got / MB / dt


# --------------------------------------------------------------------------- #
# resumable multi-connection download
# --------------------------------------------------------------------------- #
class _ChunkState:
    """Sidecar JSON mapping chunk-index -> done flag, crash safe."""

    def __init__(self, path: Path, n_chunks: int, chunk_size: int, size: int,
                 url: str):
        self.path = path
        self.done = [False] * n_chunks
        self.n_chunks = n_chunks
        self.meta = {"url": url, "size": size, "chunk_size": chunk_size,
                     "n_chunks": n_chunks}
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if (d.get("size") == size and d.get("chunk_size") == chunk_size
                        and d.get("n_chunks") == n_chunks):
                    self.done = [bool(x) for x in d["done"]]
                else:
                    print("  [dl] sidecar mismatch -> restarting chunk map")
            except Exception:  # noqa: BLE001
                pass
        self._lock = threading.Lock()

    def mark(self, i: int) -> None:
        with self._lock:
            self.done[i] = True

    def flush(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**self.meta, "done": self.done}),
                           encoding="utf-8")
            os.replace(tmp, self.path)

    @property
    def n_done(self) -> int:
        return sum(self.done)


def _looks_like_archive(p: Path, size: int) -> bool:
    """Cheap format sniff: header magic (+ zip end-of-central-directory tail).

    Used to decide whether a file of the expected length is a real complete
    archive (e.g. one we obtained by hand) or a zero-filled preallocation left
    behind by a killed run.
    """
    suffix = p.suffix.lower()
    try:
        with open(p, "rb") as f:
            head = f.read(4)
            if suffix == ".zip":
                if head[:2] != b"PK":
                    return False
                f.seek(max(0, size - 65557))
                tail = f.read()
                return b"PK\x05\x06" in tail
            if suffix in (".gz", ".tgz") or p.name.endswith(".tar.gz"):
                return head[:2] == b"\x1f\x8b"
            if suffix == ".rar":
                return head[:4] == b"Rar!"
            if suffix == ".7z":
                return head[:2] == b"7z"
            return False
    except OSError:
        return False


def download_resumable(url: str, dst, n_conn: int = 8, chunk_size: int = 32 * MB,
                       retries: int = 5, quiet: bool = False,
                       info: Optional[Dict[str, object]] = None) -> Path:
    """Download ``url`` to ``dst`` with N parallel range connections.

    * Chunk-level resume via ``<dst>.chunks.json``.
    * A completed file is detected by size and skipped immediately.
    * Falls back to a single stream when the server ignores Range.
    * Pass ``info`` (from :func:`server_info`) to avoid a second capability
      probe: some hosts (motchallenge.net) throttle repeated HEAD requests.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sidecar = dst.with_suffix(dst.suffix + ".chunks.json")
    marker = dst.with_suffix(dst.suffix + ".complete")

    if info is None:
        info = server_info(url)
    if not info["ok"]:
        raise DownloadError(f"cannot reach {url}: {info['reason']}")
    size = int(info["size"])
    if size <= 0:
        raise DownloadError(f"unknown size for {url}")

    # A file at the right byte length is NOT proof of completeness: chunks are
    # written at arbitrary offsets, so the last chunk can land first and make
    # the file reach full length while the middle is still a hole.  Trust the
    # ``.complete`` sentinel; accept an externally provided file only when its
    # format magic checks out *and* we have no resume state for it.
    if dst.exists() and dst.stat().st_size == size:
        if marker.exists():
            if not quiet:
                print(f"  [dl] already complete: {dst.name} ({size / GB:.2f} GB)")
            return dst
        if not sidecar.exists() and _looks_like_archive(dst, size):
            marker.write_text(f"ok\n{url}\n{size}\n", encoding="utf-8")
            if not quiet:
                print(f"  [dl] existing file validated: {dst.name} "
                      f"({size / GB:.2f} GB)")
            return dst
        if not sidecar.exists():
            print(f"  [dl] {dst.name} has the right length but is not a valid "
                  f"archive -> re-downloading")
            dst.unlink()

    if not info["ranges"] or n_conn <= 1:
        return _download_single(url, dst, size, quiet=quiet)

    n_chunks = (size + chunk_size - 1) // chunk_size
    st = _ChunkState(sidecar, n_chunks, chunk_size, size, url)

    # Create the file but do NOT pre-allocate it.  ``f.truncate(5.9GB)`` on this
    # machine's G: volume can take minutes (measured: >120 s for a 5.9 GB file),
    # whereas seeking past EOF and writing lets NTFS extend the file lazily and
    # costs nothing.  Each worker opens the file in "r+b" mode and seeks to its
    # chunk offset, so a hole in the middle is simply read back as zeros until
    # that chunk arrives.
    if not dst.exists():
        dst.touch()

    todo = [i for i in range(n_chunks) if not st.done[i]]
    if not quiet:
        print(f"  [dl] {dst.name}: {size / GB:.2f} GB, {n_chunks} chunks, "
              f"{len(todo)} to fetch, {n_conn} conns")
    t0 = time.time()
    got = [0]
    stop = threading.Event()
    lock = threading.Lock()

    def worker(wid: int) -> None:
        sess_fail = 0
        while not stop.is_set():
            with lock:
                if not todo:
                    return
                ci = todo.pop(0)
            start = ci * chunk_size
            end = min(size, start + chunk_size) - 1
            hdrs = {"Range": f"bytes={start}-{end}"}
            ok = False
            for attempt in range(retries):
                try:
                    r = _open(url, "GET", hdrs, timeout=120)
                    try:
                        data = r.read()
                    finally:
                        r.close()
                    if len(data) != end - start + 1:
                        raise DownloadError(
                            f"short read {len(data)} != {end - start + 1}")
                    with open(dst, "r+b") as f:
                        f.seek(start)
                        f.write(data)
                    st.mark(ci)
                    with lock:
                        got[0] += len(data)
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    sess_fail += 1
                    if attempt == retries - 1:
                        print(f"  [dl] chunk {ci} failed: {type(e).__name__}: {e}")
                    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
            if not ok:
                with lock:
                    todo.append(ci)   # requeue so another worker may retry
                if sess_fail > retries * 3:
                    return
            with lock:
                if (st.n_done % 8) == 0:
                    st.flush()

    n_workers = max(1, min(n_conn, len(todo)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(worker, i) for i in range(n_workers)]
        try:
            while any(not f.done() for f in futs):
                time.sleep(2.0)
                st.flush()          # survive a hard kill of this process
                if not quiet:
                    el = max(1e-6, time.time() - t0)
                    pct = 100.0 * st.n_done / n_chunks
                    print(f"  [dl] {dst.name} {pct:5.1f}%  "
                          f"{got[0] / GB:6.2f}/{size / GB:.2f} GB  "
                          f"{got[0] / MB / el:6.1f} MB/s  "
                          f"eta {(size - got[0]) / max(1e-6, got[0] / el) / 60:5.1f} min",
                          flush=True)
                if not any(not f.done() for f in futs):
                    break
        except KeyboardInterrupt:
            stop.set()
            st.flush()
            raise
    st.flush()
    missing = [i for i in range(n_chunks) if not st.done[i]]
    if missing:
        raise DownloadError(f"{dst.name}: {len(missing)} chunks missing, rerun to resume")
    if dst.stat().st_size != size:
        raise DownloadError(f"{dst.name}: size {dst.stat().st_size} != {size}")
    if sidecar.exists():
        sidecar.unlink()
    marker.write_text(f"ok\n{url}\n{size}\n", encoding="utf-8")
    if not quiet:
        print(f"  [dl] done {dst.name} in {(time.time() - t0) / 60:.1f} min")
    return dst


def _download_single(url: str, dst: Path, size: int, quiet: bool = False) -> Path:
    """Plain streamed download (servers without Range support)."""
    part = dst.with_suffix(dst.suffix + ".part")
    got = 0
    if part.exists():
        got = part.stat().st_size
    hdrs = {"Range": f"bytes={got}-"} if got else None
    mode = "ab" if got else "wb"
    t0 = time.time()
    for attempt in range(4):
        try:
            r = _open(url, "GET", hdrs, timeout=120)
            if got and r.status != 206:
                got, mode = 0, "wb"
            with open(part, mode) as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
                    if not quiet and got % (200 * MB) < (1 << 20):
                        el = max(1e-6, time.time() - t0)
                        print(f"  [dl] {dst.name} {got / GB:.2f}/{size / GB:.2f} GB "
                              f"{got / MB / el:.1f} MB/s", flush=True)
            r.close()
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [dl] retry {attempt + 1}: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
    if not part.exists() or (size and part.stat().st_size != size):
        raise DownloadError(f"{dst.name}: incomplete ({part.stat().st_size if part.exists() else 0}/{size})")
    os.replace(part, dst)
    dst.with_suffix(dst.suffix + ".complete").write_text(
        f"ok\n{url}\n{size}\n", encoding="utf-8")
    return dst


# --------------------------------------------------------------------------- #
# HF helpers
# --------------------------------------------------------------------------- #
def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def hf_url(repo: str, path: str, repo_type: str = "dataset") -> str:
    """Resolve URL for a file in a HF repo, honouring HF_ENDPOINT / hf-mirror."""
    base = hf_endpoint()
    root = "" if repo_type == "model" else f"/{repo_type}s"
    return f"{base}{root}/{repo}/resolve/main/{urllib.parse.quote(path)}"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
