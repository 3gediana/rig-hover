"""Stage 1 feature extraction: UCF101 -> CLIP feature cache.

Pipeline
--------
``--extract`` (default) reads each selected clip straight out of ``UCF-101.zip``
into a per-thread scratch file on the local SSD, decodes 96 uniformly spaced
frames (PyAV, 10 decode threads), resizes to CLIP geometry, runs one batched
fp16 GPU forward per clip and writes one npz per clip immediately while
recording ``progress.json``.
``--unzip`` optionally materialises the selected avi files on disk (not needed
and not recommended: G: is a USB volume).
Then ``scripts/build_manifests_ucf101.py --finalize`` builds the manifest.

Resumable: ``progress.json`` holds {seq_id: {T, path, fps, n_frames_total}}; a
rerun skips anything already written.

Usage
-----
python scripts/extract_ucf101.py                    # extract
python scripts/extract_ucf101.py --limit 3          # smoke test
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402
from src.data import ucf101  # noqa: E402
from src.data.clip_encoder import CLIPEncoder, resize_shortest_center_crop  # noqa: E402
from src.data.extract import ExtractionRunner  # noqa: E402
from src.data.util import attach_log, dir_size_gb  # noqa: E402
from src.data.video import (VideoError, decode_selected, probe_video,  # noqa: E402
                           sample_indices_uniform, timestamp_from_index)

ZIP = r"G:\stream_memory_data\videos\ucf101\UCF-101.zip"
VIDEO_ROOT = Path(r"G:\stream_memory_data\videos\ucf101")
FEAT_ROOT = Path(r"G:\stream_memory_data\features\ucf101_action")
PLAN_PATH = contract.MANIFESTS / "ucf101_action_plan.json"
N_FRAMES = 96

# module-level config shared by the decode threads
_CFG = {"video_root": str(VIDEO_ROOT), "n_frames": N_FRAMES, "zip": ZIP}
_ZF = None       # shared ZipFile, guarded by _ZIP_LOCK
_ZIP_LOCK = threading.Lock()


def _zip() -> zipfile.ZipFile:
    global _ZF
    if _ZF is None:
        _ZF = zipfile.ZipFile(_CFG["zip"])
    return _ZF


def read_member_to(path: Path, relpath: str) -> Path:
    """Copy one clip out of UCF-101.zip into a per-worker scratch file.

    Why not decode straight from a BytesIO: ffmpeg's AVI demuxer fails
    (``InvalidDataError``) on a fair share of UCF101 clips when the input is a
    non-file seekable stream.  Why not extract to G:: G: is a *USB* drive and
    thousands of small writes there thrash it (measured 0.25 videos/s vs
    3.45 videos/s when the source is local); so each worker spills one clip at a
    time to the local-SSD temp dir, decodes it, and reuses the same path.  Adding
    the zip read + SSD write costs ~5 ms per clip.
    """
    member = "UCF-101/" + relpath
    with _ZIP_LOCK:
        data = _zip().read(member)
    if not data:
        raise VideoError(f"empty member {member}")
    with open(path, "wb") as f:
        f.write(data)
    return path


_TMP: Path | None = None


def _scratch() -> Path:
    """Per-process scratch file on the local SSD (not on the USB G: volume)."""
    global _TMP
    if _TMP is None:
        d = Path(tempfile.mkdtemp(prefix="smr_ucf101_"))
        _TMP = d / f"clip_{os.getpid()}.avi"
    return _TMP


def decode_worker(job: dict) -> dict:
    """Runs in a DataLoader worker: decode -> CLIP-geometry uint8 frames."""
    try:
        path = read_member_to(_scratch(), job["relpath"])
    except Exception as e:  # noqa: BLE001
        return {"seq_id": job["seq_id"],
                "skip_reason": f"zip read failed {type(e).__name__}: {e}"}
    try:
        info = probe_video(path)
        idx = sample_indices_uniform(info.n_frames, _CFG["n_frames"])
        frames = decode_selected(path, idx, info)
    except VideoError as e:
        return {"seq_id": job["seq_id"], "skip_reason": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"seq_id": job["seq_id"], "skip_reason": f"{type(e).__name__}: {e}"}
    if len(frames) < 2:
        return {"seq_id": job["seq_id"], "skip_reason": f"only {len(frames)} frames"}
    arr = np.stack([resize_shortest_center_crop(f) for f in frames])
    t = timestamp_from_index(idx[:len(frames)], info.fps)
    lab = np.full(len(frames), int(job["class_index"]), dtype=np.int64)
    return {
        "seq_id": job["seq_id"],
        "frames": arr,
        "t": t,
        "lab_action": lab,
        "meta": {"fps": info.fps, "n_frames_total": info.n_frames,
                 "n_frames_kept": int(len(frames)), "estimated_n": info.estimated_n,
                 "width": info.width, "height": info.height},
    }


class UCFRunner(ExtractionRunner):
    """ExtractionRunner that also stores per-seq fps/frame counts in progress."""

    def encode_and_store(self, item):
        p = super().encode_and_store(item)
        # enrich the progress entry with decode metadata used by the manifest
        m = item.get("meta", {})
        self.progress.done[str(item["seq_id"])].update(
            {"fps": m.get("fps"), "n_frames_total": m.get("n_frames_total")})
        return p


def do_unzip(args) -> None:
    plan = contract.read_json(PLAN_PATH)
    entries = [r["relpath"] for r in plan["records"]]
    t0 = time.time()
    print(f"[unzip] extracting {len(entries)} selected clips from {ZIP} ...", flush=True)
    n = ucf101.extract_entries(ZIP, entries, VIDEO_ROOT)
    print(f"[unzip] {n} new files in {time.time() - t0:.1f}s -> {VIDEO_ROOT}", flush=True)


def do_extract(args) -> None:
    plan = contract.read_json(PLAN_PATH)
    jobs = plan["records"]
    enc = CLIPEncoder(batch_size=args.batch, min_batch=16)
    enc.warmup(4)
    runner = UCFRunner(
        dataset="ucf101_action", out_dir=FEAT_ROOT,
        progress_path=FEAT_ROOT / "progress.json", encoder=enc,
        num_workers=args.workers, log_every=25,
        progress_meta={"encoder_tag": "clip-vit-base-p16-224-f16-2x2tok-v1",
                       "n_frames": N_FRAMES, "dataset": "ucf101_action"})
    t0 = time.time()
    summary = runner.run(jobs, decode_worker, stop_after=args.limit)
    print(f"[extract] {summary}", flush=True)
    print(f"[extract] wall {(time.time() - t0) / 60:.1f} min, "
          f"feat dir {dir_size_gb(FEAT_ROOT):.2f} GB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unzip", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--log", default=r"G:\stream_memory_data\logs\extract_ucf101.log")
    args = ap.parse_args()
    if not args.unzip and not args.extract:
        args.extract = True          # decode streams straight from the zip
    attach_log(args.log)
    if args.unzip:
        do_unzip(args)                # optional: materialise the avi files
    if args.extract:
        do_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
