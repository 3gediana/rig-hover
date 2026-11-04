"""Stage 3 feature extraction: Charades -> CLIP feature cache (online event detection).

  python scripts/extract_charades.py --download    # 15.2 GB, resumable
  python scripts/extract_charades.py --extract     # decode + encode -> npz
  python scripts/build_manifests_charades.py       # write the manifest

Each of the 1400 selected clips is streamed out of ``Charades_v1_480.zip`` into a
per-thread scratch file on the local SSD, decoded to 128 uniformly spaced frames
and immediately encoded; the mp4 demuxer needs a real seekable file (a BytesIO
works for AVI/UCF101 but not reliably for all these mp4s, and G: is a USB volume
so the scratch must not live there).
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

from src.data import charades  # noqa: E402
from src.data.clip_encoder import CLIPEncoder, resize_shortest_center_crop  # noqa: E402
from src.data.extract import ExtractionRunner  # noqa: E402
from src.data.util import attach_log, dir_size_gb  # noqa: E402
from src.data.video import (VideoError, decode_selected, probe_video,  # noqa: E402
                           sample_indices_uniform, timestamp_from_index)

ZIP = r"G:\stream_memory_data\videos\charades\Charades_v1_480.zip"
ANN_DIR = r"G:\stream_memory_data\videos\charades\ann\Charades"
FEAT_ROOT = Path(r"G:\stream_memory_data\features\charades_event")
DATASET = "charades_event"

_CFG = {"zip": ZIP, "ann": ANN_DIR}
_ZF = None
_ZIP_LOCK = threading.Lock()
_PLAN = None
_PLAN_LOCK = threading.Lock()


def _zip() -> zipfile.ZipFile:
    global _ZF
    if _ZF is None:
        _ZF = zipfile.ZipFile(_CFG["zip"])
    return _ZF


def _plan() -> dict:
    """Build (and cache) the deterministic selection plan."""
    global _PLAN
    with _PLAN_LOCK:
        if _PLAN is None:
            rows = charades.load_rows(_CFG["ann"])
            _PLAN = charades.build_plan(rows)
            _PLAN["member"] = {}
    return _PLAN


_TMPDIR: Path | None = None


def _scratch() -> Path:
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = Path(tempfile.mkdtemp(prefix="smr_charades_"))
    return _TMPDIR / f"clip_{os.getpid()}_{threading.get_ident()}.mp4"


def decode_worker(job: dict) -> dict:
    vid = job["seq_id"]
    plan = _plan()
    row = plan["rows"][vid]
    try:
        with _ZIP_LOCK:
            member = plan["member"].get(vid)
            if member is None:
                member = charades.find_video_member(_zip(), vid)
                if member is None:
                    return {"seq_id": vid, "skip_reason": "not in archive"}
                plan["member"][vid] = member
            data = _zip().read(member)
    except Exception as e:  # noqa: BLE001
        return {"seq_id": vid, "skip_reason": f"zip read {type(e).__name__}: {e}"}
    if not data:
        return {"seq_id": vid, "skip_reason": "empty member"}
    path = _scratch()
    try:
        with open(path, "wb") as f:
            f.write(data)
        info = probe_video(path)
        n = info.n_frames
        idx = sample_indices_uniform(n, charades.N_FRAMES)
        frames = decode_selected(path, idx, info)
    except VideoError as e:
        return {"seq_id": vid, "skip_reason": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"seq_id": vid, "skip_reason": f"{type(e).__name__}: {e}"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if len(frames) < 4:
        return {"seq_id": vid, "skip_reason": f"only {len(frames)} frames"}
    t = timestamp_from_index(idx[:len(frames)], info.fps)
    ev, seg = charades.frame_labels(t, row["_segs"], plan["classes"])
    arr = np.stack([resize_shortest_center_crop(f) for f in frames])
    return {
        "seq_id": vid, "frames": arr, "t": t,
        "lab_event": ev.astype(np.float32),
        "lab_seg": seg.astype(np.int64),
        "meta": {"fps": info.fps, "n_frames_total": info.n_frames,
                 "n_frames_kept": int(len(frames)),
                 "csv_length": row["_length"],
                 "scene": row.get("scene", ""), "subject": row.get("subject", ""),
                 "n_segments": len(row["_segs"]),
                 "n_actions_total": len({c for c, _s, _e in row["_segs"]})},
    }


def plan_jobs() -> list:
    plan = _plan()
    jobs = []
    for v in plan["ids"]:
        jobs.append({"seq_id": v, "split": plan["split"][v]})
    # interleave splits so a partial run covers all three
    order = {"train": 0, "val": 1, "test": 2}
    jobs.sort(key=lambda r: (r["seq_id"],))
    return jobs


class CharadesRunner(ExtractionRunner):
    """Stores decode metadata (fps / frame count / segments) in progress.json."""

    def encode_and_store(self, item):
        p = super().encode_and_store(item)
        m = item.get("meta", {}) or {}
        self.progress.done[str(item["seq_id"])].update(
            {"fps": m.get("fps"), "n_frames_total": m.get("n_frames_total"),
             "csv_length": m.get("csv_length"), "scene": m.get("scene"),
             "subject": m.get("subject"), "n_segments": m.get("n_segments")})
        return p


def do_download(_args) -> None:
    from src.data.download import download_resumable, server_info
    url = "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip"
    info = server_info(url)
    print(f"[charades] s3 ok={info['ok']} size={int(info['size']) / 1e9:.2f} GB "
          f"ranges={info['ranges']}", flush=True)
    p = download_resumable(url, ZIP, n_conn=16, chunk_size=16 * (1 << 20),
                           info=info)
    print(f"[charades] archive {p} {p.stat().st_size / 1e9:.2f} GB", flush=True)


def do_extract(args) -> None:
    plan = _plan()
    jobs = plan_jobs()
    print(f"[charades] {len(jobs)} clips, {len(plan['classes'])} event classes",
          flush=True)
    enc = CLIPEncoder(batch_size=args.batch, min_batch=8)
    enc.warmup(4)
    runner = CharadesRunner(
        dataset=DATASET, out_dir=FEAT_ROOT,
        progress_path=FEAT_ROOT / "progress.json", encoder=enc,
        num_workers=args.workers, log_every=25,
        progress_meta={"dataset": DATASET, "n_frames": charades.N_FRAMES,
                       "n_classes": len(plan["classes"]),
                       "encoder_tag": "clip-vit-base-p16-224-f16-2x2tok-v1"})
    t0 = time.time()
    summary = runner.run(jobs, decode_worker, stop_after=args.limit)
    print(f"[extract] {summary}", flush=True)
    print(f"[extract] wall {(time.time() - t0) / 60:.1f} min, "
          f"feat dir {dir_size_gb(FEAT_ROOT):.2f} GB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--log", default=r"G:\stream_memory_data\logs\extract_charades.log")
    args = ap.parse_args()
    if not args.download and not args.extract:
        args.extract = True
    attach_log(args.log)
    if args.download:
        do_download(args)
    if args.extract:
        do_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
