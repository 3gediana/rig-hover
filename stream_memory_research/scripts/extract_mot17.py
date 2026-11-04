"""Stage 2 feature extraction: MOT17 -> CLIP feature cache (tracking + occlusion).

  python scripts/extract_mot17.py --unzip      # extract 6 sequences from MOT17.zip
  python scripts/extract_mot17.py --extract    # decode + CLIP encode -> npz

Per sequence, at 5 fps:
  feat/tok    from the full frame
  roi         [T,8,768] CLIP features of the 8 anchor boxes (crop + 10% margin)
  roi_valid   [T,8]     0 when the anchor is occluded / absent / inside an ignore region
  roi_meta    [T,8,4]   current normalised box (cx,cy,w,h)
  lab_track   [T,8,4]   the anchor's NEXT OBSERVED normalised box
  lab_id      [T,8]     contiguous anchor id (slot index), -1 where invalid
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402
from src.data import mot17  # noqa: E402
from src.data.clip_encoder import (CLIPEncoder, FEAT_DIM, N_TOK, TOK_DIM,  # noqa: E402
                                   crop_box_with_margin, resize_shortest_center_crop)
from src.data.extract import ExtractionRunner  # noqa: E402
from src.data.util import attach_log, dir_size_gb  # noqa: E402
from src.data.video import VideoError  # noqa: E402

ZIP = r"G:\stream_memory_data\videos\mot17\MOT17.zip"
SEQ_ROOT = Path(r"G:\stream_memory_data\videos\mot17\sequences")
FEAT_ROOT = Path(r"G:\stream_memory_data\features\mot17_track")
DATASET = "mot17_track"
VIS_STRIDE = 5.0

_CFG = {"seq_root": str(SEQ_ROOT)}


def _imread(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_worker(job: dict) -> dict:
    """Decode one MOT17 sequence at 5 fps and build every per-frame array.

    Frames are streamed: only the 224x224 CLIP-geometry copy is kept, never the
    full 1920x1080 frame (retaining those cost 1.1 GB for MOT17-04 and made
    ``cv2.imread`` fail with its own OutOfMemoryError).  ROI crops are returned
    **compactly** -- only the valid slots -- because the dense [T,8,224,224,3]
    layout is up to 240 MB per sequence while often <60 % of slots are valid.
    """
    seq = job["seq_id"]
    seq_dir = Path(_CFG["seq_root"]) / seq
    try:
        data = mot17.assemble_sequence(seq_dir)
    except Exception as e:  # noqa: BLE001
        return {"seq_id": seq, "skip_reason": f"{type(e).__name__}: {e}"}
    frames_idx = data["frames"]
    fps = float(data["fps"])
    T = len(frames_idx)
    N = data["slot_box"].shape[1]

    full_frames: list = []
    crops: list = []                     # compact: one crop per VALID slot
    missing = 0
    prev = None
    for i, fr in enumerate(frames_idx):
        p = seq_dir / "img1" / f"{int(fr):06d}.jpg"
        img = _imread(p)
        if img is None:
            missing += 1
            img = prev if prev is not None else np.zeros(
                (int(data["height"]), int(data["width"]), 3), np.uint8)
        prev = img
        full_frames.append(resize_shortest_center_crop(img))
        H, W = img.shape[:2]
        for j in range(N):
            if not data["slot_valid"][i, j]:
                continue
            cx, cy, w, h = data["slot_box"][i, j]
            box = ((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H)
            crops.append(crop_box_with_margin(img, box))
        del img
    if missing > 0.2 * T:
        return {"seq_id": seq, "skip_reason": f"{missing}/{T} frames unreadable"}

    # t uses the ORIGINAL 1-based frame index so dt is real physical time
    t = ((frames_idx - 1).astype(np.float64) / fps).astype(np.float32)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = np.float32(t[i - 1] + 1e-4)

    lab_track = data["slot_box_next"].astype(np.float32)
    lab_id = np.where(data["slot_valid"] > 0,
                      np.arange(N, dtype=np.int64)[None, :].repeat(T, 0),
                      -1).astype(np.int64)
    return {
        "seq_id": seq,
        "frames": np.stack(full_frames),
        "t": t,
        "roi_crops": (np.stack(crops) if crops
                      else np.zeros((0, 224, 224, 3), np.uint8)),
        "roi_valid": data["slot_valid"].astype(np.uint8),
        "roi_meta": data["slot_box"].astype(np.float32),
        "lab_track": lab_track,
        "lab_id": lab_id,
        "meta": {"fps": fps, "n_frames_total": int(data["n_frames_total"]),
                 "n_frames_kept": int(T), "missing_frames": missing,
                 "anchor_ids": data["anchors"],
                 "n_ignore_flags": int(data["slot_ignore"].sum())},
    }


def do_unzip(args) -> None:
    seqs = list(mot17.SEQ_SPLIT)
    t0 = time.time()
    print(f"[mot17] extracting {len(seqs)} sequences from {ZIP} ...", flush=True)
    mot17.extract_sequences(ZIP, seqs, SEQ_ROOT)
    print(f"[mot17] done in {(time.time() - t0) / 60:.1f} min -> {SEQ_ROOT}", flush=True)


def do_extract(args) -> None:
    jobs = [{"seq_id": s} for s in mot17.SEQ_SPLIT]
    enc = CLIPEncoder(batch_size=args.batch, min_batch=8)
    enc.warmup(4)
    runner = ExtractionRunner(
        dataset=DATASET, out_dir=FEAT_ROOT,
        progress_path=FEAT_ROOT / "progress.json", encoder=enc,
        num_workers=args.workers, log_every=1,
        progress_meta={"dataset": DATASET, "fps": VIS_STRIDE, "n_roi": mot17.N_ROI,
                       "encoder_tag": "clip-vit-base-p16-224-f16-2x2tok-v1"})
    t0 = time.time()
    summary = runner.run(jobs, decode_worker)
    print(f"[extract] {summary}", flush=True)
    print(f"[extract] wall {(time.time() - t0) / 60:.2f} min, "
          f"feat dir {dir_size_gb(FEAT_ROOT):.2f} GB", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unzip", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--log", default=r"G:\stream_memory_data\logs\extract_mot17.log")
    args = ap.parse_args()
    if not args.unzip and not args.extract:
        args.unzip = args.extract = True
    attach_log(args.log)
    if args.unzip:
        do_unzip(args)
    if args.extract:
        do_extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
