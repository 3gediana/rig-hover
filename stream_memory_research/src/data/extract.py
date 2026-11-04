"""Generic resumable feature-extraction driver + progress bookkeeping.

One :class:`ExtractionRunner` drives every dataset.  The dataset-specific part
is a single ``decode_worker(job) -> item`` function which runs inside a
``DataLoader`` worker process (parallel PyAV decoding, num_workers=8) and
returns preprocessed uint8 frames plus everything else the npz needs.  The
runner then does the GPU work in the main process (one big batched forward per
sequence), writes one npz per sequence immediately and records progress so a
re-run after a crash resumes instead of restarting.

Item contract returned by ``decode_worker`` (a plain dict):
    seq_id        str                     required
    frames        uint8 [T,224,224,3]     required (already CLIP-geometry)
    t             float32 [T]             required
    roi_crops     uint8 [T,N,224,224,3]   optional
    roi_meta      float32 [T,N,4]         optional  normalised cx,cy,w,h
    roi_valid     uint8 [T,N]             optional
    lab_*         ndarray                 optional  any per-frame labels
    meta          dict                    optional  small json-able extras
    skip_reason   str                     optional  -> recorded as skipped
"""
from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .clip_encoder import FEAT_DIM, N_TOK, TOK_DIM, CLIPEncoder


# --------------------------------------------------------------------------- #
# progress.json
# --------------------------------------------------------------------------- #
class ProgressTracker:
    """Crash-safe ``progress.json``: which seq_ids already have an npz."""

    def __init__(self, path, meta: Optional[Dict[str, Any]] = None,
                 flush_every: int = 10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = int(flush_every)
        self.done: Dict[str, Dict[str, Any]] = {}
        self.skipped: Dict[str, str] = {}
        self.meta: Dict[str, Any] = dict(meta or {})
        self._dirty = 0
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.done = d.get("done", {})
                self.skipped = d.get("skipped", {})
                self.meta.update(d.get("meta", {}))
            except Exception:  # noqa: BLE001
                print(f"[progress] corrupt {self.path}, starting fresh")

    def is_done(self, seq_id: str, npz_path=None) -> bool:
        if seq_id not in self.done:
            return False
        if npz_path is not None and not Path(npz_path).exists():
            return False
        return True

    def mark(self, seq_id: str, info: Dict[str, Any]) -> None:
        self.done[seq_id] = info
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def mark_skipped(self, seq_id: str, reason: str) -> None:
        self.skipped[seq_id] = reason[:400]
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "done": self.done,
                                   "skipped": self.skipped},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = 0


# --------------------------------------------------------------------------- #
# npz writing
# --------------------------------------------------------------------------- #
def write_seq_npz(path, arrays: Dict[str, np.ndarray]) -> Path:
    """Atomic per-sequence npz write (tmp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
class ExtractionRunner:
    """Drives decode(worker) -> encode(GPU) -> npz(write) for a list of jobs."""

    def __init__(self, dataset: str, out_dir, progress_path, encoder: CLIPEncoder,
                 num_workers: int = 8, log_every: int = 20,
                 progress_meta: Optional[Dict[str, Any]] = None):
        self.dataset = dataset
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.num_workers = int(num_workers)
        self.log_every = int(log_every)
        self.progress = ProgressTracker(progress_path, meta=progress_meta)
        self.n_ok = 0
        self.n_skip = 0
        self.t_start = time.time()

    # -- item -> npz --------------------------------------------------------
    def encode_and_store(self, item: Dict[str, Any]) -> Optional[Path]:
        seq_id = str(item["seq_id"])
        frames = item["frames"]
        T = int(frames.shape[0])
        feat, tok = self.encoder.encode(frames)
        item["frames"] = None      # let the 14.5 MB uint8 block go immediately

        arrays: Dict[str, np.ndarray] = {
            "feat": feat.astype(np.float16),
            "tok": tok.astype(np.float16),
            "t": np.asarray(item["t"], dtype=np.float32),
        }
        roi = item.get("roi_crops")
        if roi is not None and len(roi):
            Tn, N = int(np.asarray(item["roi_valid"]).shape[0]), \
                int(np.asarray(item["roi_valid"]).shape[1])
            rv = np.asarray(item["roi_valid"], bool).reshape(-1)
            flat = np.asarray(roi).reshape(-1, *np.asarray(roi).shape[-3:])
            if flat.shape[0] == Tn * N:
                # dense layout [T,N,224,224,3]
                crops = flat[rv] if rv.any() else flat[:0]
                idxs = np.flatnonzero(rv)
            else:
                # compact layout: only the valid slots were kept by the decoder
                crops = flat
                idxs = np.flatnonzero(rv)
                if len(crops) != len(idxs):
                    raise ValueError(
                        f"{seq_id}: compact roi has {len(crops)} crops but "
                        f"{len(idxs)} valid slots")
            out = np.zeros((Tn * N, FEAT_DIM), np.float16)
            if len(crops):
                rf, _ = self.encoder.encode(crops)
                out[idxs] = rf.astype(np.float16)
            arrays["roi"] = out.reshape(Tn, N, FEAT_DIM)
            arrays["roi_valid"] = np.asarray(item["roi_valid"], np.uint8)
            arrays["roi_meta"] = np.asarray(item["roi_meta"], np.float32)
        for k, v in item.items():
            if k.startswith("lab_"):
                arrays[k] = np.asarray(v)
        path = self.out_dir / f"{seq_id}.npz"
        write_seq_npz(path, arrays)
        self.progress.mark(seq_id, {"T": T, "path": str(path),
                                    "bytes": path.stat().st_size})
        return path

    # -- main loop ----------------------------------------------------------
    def run(self, jobs: Iterable[Dict[str, Any]],
            decode_worker: Callable[[Dict[str, Any]], Dict[str, Any]],
            stop_after: Optional[int] = None) -> Dict[str, Any]:
        """Run extraction.  ``jobs`` is a list of json-able job dicts.

        The decode pool is a **thread** pool, not a torch DataLoader.  Spawning
        8-10 interpreter children on this box dies with
        ``OSError [WinError 1455]: the paging file is too small`` because each
        child re-imports torch's CUDA DLLs and the venv lives on the USB G:
        volume.  PyAV decoding and cv2.resize release the GIL, so threads give
        the same parallelism from one interpreter with a warm page cache; a
        bounded prefetch (2 x workers in flight) keeps the GPU fed.
        """
        jobs = list(jobs)
        todo = [j for j in jobs
                if not self.progress.is_done(str(j["seq_id"]),
                                             self.out_dir / f"{j['seq_id']}.npz")]
        n_total = len(jobs)
        print(f"[{self.dataset}] {n_total} jobs, {n_total - len(todo)} already done, "
              f"{len(todo)} to do", flush=True)
        if stop_after:
            todo = todo[:stop_after]
        if not todo:
            self.progress.flush()
            return self._summary()

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        nw = max(1, min(self.num_workers, len(todo)))
        # one item in flight per worker: each holds a [T,224,224,3] uint8 block
        # (14.5 MB), and this box only has ~5 GB of free RAM shared with the
        # training teammates, so do not prefetch 2x
        depth = max(2, nw)
        it = iter(todo)
        n_seen = 0
        job_by_id = {str(j["seq_id"]): j for j in todo}
        retries: Dict[str, int] = {}
        max_retries = 2
        with ThreadPoolExecutor(max_workers=nw, thread_name_prefix="dec") as ex:
            pending = deque()

            def _task(job: Dict[str, Any]) -> Dict[str, Any]:
                """Run the decoder, converting any exception into a reportable item."""
                try:
                    return decode_worker(job)
                except BaseException as e:  # noqa: BLE001
                    return {"seq_id": str(job.get("seq_id")),
                            "skip_reason": f"{type(e).__name__}: {e}"}

            def _submit(job: Dict[str, Any]) -> None:
                pending.append((job, ex.submit(_task, job)))

            def _fill() -> None:
                for _ in range(depth - len(pending)):
                    nxt = next(it, None)
                    if nxt is None:
                        return
                    _submit(nxt)

            _fill()
            while pending:
                job, fut = pending.popleft()
                item = None
                try:
                    item = fut.result()
                except Exception as e:  # noqa: BLE001
                    item = {"seq_id": str(job.get("seq_id")),
                            "skip_reason": f"{type(e).__name__}: {e}"}
                n_seen += 1
                _fill()
                sid = str(item.get("seq_id"))
                if item.get("skip_reason") or item.get("error"):
                    msg = str(item.get("skip_reason") or item.get("error"))
                    # transient allocation failures get another chance
                    if ("MemoryError" in msg or "Insufficient memory" in msg) \
                            and retries.get(sid, 0) < max_retries:
                        retries[sid] = retries.get(sid, 0) + 1
                        gc.collect()
                        print(f"[{self.dataset}] RETRY {sid} "
                              f"(attempt {retries[sid]}): {msg[:100]}", flush=True)
                        _submit(job_by_id.get(sid, {"seq_id": sid}))
                        continue
                    self.n_skip += 1
                    self.progress.mark_skipped(sid, msg)
                    print(f"[{self.dataset}] SKIP {sid}: {msg[:140]}", flush=True)
                    continue
                try:
                    self.encode_and_store(item)
                    self.n_ok += 1
                except Exception as e:  # noqa: BLE001
                    msg = f"{type(e).__name__}: {e}"
                    if ("MemoryError" in msg or "Insufficient memory" in msg) \
                            and retries.get(sid, 0) < max_retries:
                        retries[sid] = retries.get(sid, 0) + 1
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"[{self.dataset}] RETRY-ENCODE {sid} "
                              f"(attempt {retries[sid]}): {msg[:100]}", flush=True)
                        _submit(job_by_id.get(sid, {"seq_id": sid}))
                        continue
                    self.n_skip += 1
                    self.progress.mark_skipped(sid, msg)
                    print(f"[{self.dataset}] ENCODE-FAIL {sid}: {msg}", flush=True)
                    if "OutOfMemory" in msg:
                        torch.cuda.empty_cache()
                if self.n_ok % self.log_every == 0:
                    el = time.time() - self.t_start
                    rate = n_seen / max(1e-6, el)
                    left = len(todo) - n_seen
                    print(f"[{self.dataset}] {self.n_ok} ok / {n_seen} seen "
                          f"({rate:.2f}/s, eta {left / max(1e-6, rate) / 60:.1f} min) "
                          f"batch={self.encoder.batch_size} "
                          f"gpu_free={_gpu_free():.1f}GB", flush=True)
        self.progress.flush()
        return self._summary()

    def _summary(self) -> Dict[str, Any]:
        el = time.time() - self.t_start
        return {"dataset": self.dataset, "n_ok": self.n_ok, "n_skip": self.n_skip,
                "seconds": round(el, 1), "done_total": len(self.progress.done),
                "skipped_total": len(self.progress.skipped)}


class _JobDataset(torch.utils.data.Dataset):
    """Unused process-based wrapper kept for reference; extraction uses threads.

    Spawning interpreter children fails on this machine (WinError 1455 from
    torch's CUDA DLLs), see :meth:`ExtractionRunner.run`.
    """

    def __init__(self, jobs, fn):
        self.jobs = jobs
        self.fn = fn

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        job = self.jobs[i]
        try:
            return self.fn(job)
        except Exception as e:  # noqa: BLE001
            return {"seq_id": str(job.get("seq_id", i)),
                    "error": f"{type(e).__name__}: {e}\n"
                             f"{traceback.format_exc()[-800:]}"}


def _identity(x):
    return x[0] if isinstance(x, list) and len(x) == 1 else x


def _gpu_free() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


# --------------------------------------------------------------------------- #
# sanity checks shared by all builders
# --------------------------------------------------------------------------- #
def check_arrays(arrays: Dict[str, np.ndarray], task: str,
                 n_feat_ok: int = FEAT_DIM) -> List[str]:
    """Return a list of problems (empty == ok) for one sequence's arrays."""
    problems: List[str] = []
    feat = arrays.get("feat")
    t = arrays.get("t")
    if feat is None or t is None:
        return ["missing feat/t"]
    T = feat.shape[0]
    if feat.shape != (T, n_feat_ok):
        problems.append(f"feat shape {feat.shape}")
    if t.shape != (T,):
        problems.append(f"t shape {t.shape}")
    elif not np.all(np.diff(t.astype(np.float64)) > 0):
        problems.append("t not strictly increasing")
    if not np.all(np.isfinite(feat.astype(np.float32))):
        problems.append("feat has NaN/Inf")
    tok = arrays.get("tok")
    if tok is not None:
        if tok.shape != (T, N_TOK, TOK_DIM):
            problems.append(f"tok shape {tok.shape}")
        elif not np.all(np.isfinite(tok.astype(np.float32))):
            problems.append("tok has NaN/Inf")
    return problems
