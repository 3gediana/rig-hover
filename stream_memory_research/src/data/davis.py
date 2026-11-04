"""DAVIS 2017 single-object occlusion-recovery task construction.

Source: HF mirror of the official archive
    ``hf://datasets/yinloonga/DAVIS_2017/DAVIS-2017-trainval-480p.zip`` (0.78 GB)
which contains **both** ``JPEGImages/480p`` and ``Annotations/480p`` for the 90
trainval sequences (60 train / 30 val from ``DAVIS/ImageSets/2017/*.txt``).
``data.vision.ee.ethz.ch`` was measured at 0.04 MB/s from this machine (unusable),
the HF mirror delivered the same 832,766,765-byte file in 5.6 min.

Important honesty note about "occlusion"
----------------------------------------
DAVIS is a *video object segmentation* benchmark: the annotated object is
present in essentially every frame, so the literal rule "roi_valid=0 when the
mask is empty" almost never fires.  A mask-bbox task with no invalid frames
cannot test occlusion memory at all.  Therefore:

  * ``roi_valid`` is 0 when the mask is empty (the literal rule), **and**
  * we additionally *simulate* occlusion by invalidating the ROI on real,
    documented intervals (1-3 gaps per sequence, 0.4-1.5 s each, at most 40% of
    frames, never frame 0, gaps separated by >= 1 s of valid data).

The pixels, the CLIP features and the boxes are all real; only the "is the ROI
observable" flag is injected.  ``lab_present`` stays 1 on those frames, which is
exactly the signal a memory module must reconstruct from the history.  This is
recorded in the manifest under ``extra.occlusion_simulation`` so nobody mistakes
it for a native occlusion benchmark.

Frame rate: the DAVIS 2017 release is delivered as 24 fps image sequences (the
2017 edition resampled all clips to 24 fps); there is no per-sequence fps field
in the archive, so ``fps = 24`` is used for ``t = frame_index / 24``.
Sampling: 10 fps target -> frame indices ``round(k * 24/10)`` (nearest real
frame, so dt alternates 2/3 native frames), capped at 120 frames.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

DATASET = "davis_vos"
NATIVE_FPS = 24.0
TARGET_FPS = 10.0
MAX_FRAMES = 120
N_ROI = 1
GAP_MAX_TOTAL_FRAC = 0.40      # at most 40% of frames invalid
GAP_SEC_RANGE = (0.4, 1.5)
MIN_VALID_RUN_SEC = 1.0

JPEG_ROOT = "DAVIS/JPEGImages/480p/"
ANN_ROOT = "DAVIS/Annotations/480p/"
SETS_ROOT = "DAVIS/ImageSets/2017/"


# --------------------------------------------------------------------------- #
# zip listing
# --------------------------------------------------------------------------- #
def list_sequences(zip_path) -> Tuple[List[str], Dict[str, str]]:
    """Return (trainval sequences sorted, {seq: split}) from ImageSets/2017."""
    split: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for name, key in (("train.txt", "train"), ("val.txt", "val")):
            member = SETS_ROOT + name
            if member in names:
                for s in z.read(member).decode("utf-8").split():
                    split[s.strip()] = key
    return sorted(split), split


def list_frames(zip_path, seq: str) -> List[str]:
    """Sorted JPEG member names for one sequence."""
    pre = f"{JPEG_ROOT}{seq}/"
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n for n in z.namelist()
                      if n.startswith(pre) and n.lower().endswith(".jpg"))


def sample_indices(n: int, native_fps: float = NATIVE_FPS,
                   target_fps: float = TARGET_FPS,
                   cap: int = MAX_FRAMES) -> np.ndarray:
    """Nearest real frames to a uniform target-fps time grid (strictly increasing)."""
    step = native_fps / target_fps
    idx = np.unique(np.round(np.arange(0, n, step)).astype(np.int64))
    idx = idx[(idx >= 0) & (idx < n)]
    if len(idx) > cap:
        keep = np.linspace(0, len(idx) - 1, cap).astype(np.int64)
        idx = idx[np.unique(keep)]
    return idx


# --------------------------------------------------------------------------- #
# masks & boxes
# --------------------------------------------------------------------------- #
def read_mask_bgr(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("undecodable mask")
    return img


def mask_to_box(mask: np.ndarray, height: int, width: int) -> Tuple[np.ndarray, bool]:
    """Binary union mask -> normalised (cx, cy, w, h) box in [0,1]; visible flag."""
    if mask is None:
        return np.zeros(4, np.float32), False
    if mask.ndim == 3:
        # DAVIS instance masks are paletted PNGs; any nonzero label = object
        mask = (mask[..., :3].astype(np.uint16).sum(-1) > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return np.zeros(4, np.float32), False
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    w = x1 - x0 + 1.0
    h = y1 - y0 + 1.0
    box = np.array([(x0 + x1) / 2.0 / width, (y0 + y1) / 2.0 / height,
                    w / width, h / height], np.float32)
    return np.clip(box, 0.0, 1.0), True


# --------------------------------------------------------------------------- #
# simulated occlusion gaps
# --------------------------------------------------------------------------- #
def gap_plan(seq: str, frames: np.ndarray, t: np.ndarray,
             seed: int = 0) -> np.ndarray:
    """Deterministic boolean invalid-mask for simulated occlusion intervals."""
    T = len(frames)
    invalid = np.zeros(T, bool)
    if T < 8:
        return invalid
    rng = np.random.default_rng(_stable_seed(seed, seq))
    dur_sec = float(t[-1] - t[0]) + float(np.median(np.diff(t))) if T > 1 else 0.0
    if dur_sec <= 0:
        return invalid
    n_gaps = int(rng.integers(1, 4))
    budget = int(GAP_MAX_TOTAL_FRAC * T)
    occupied = np.zeros(T, bool)
    for _ in range(n_gaps * 3):          # try a few times to place a gap
        if invalid.sum() >= budget:
            break
        gap_sec = float(rng.uniform(*GAP_SEC_RANGE))
        gap_sec = min(gap_sec, 0.5 * dur_sec)
        start_sec = float(rng.uniform(t[0] + MIN_VALID_RUN_SEC,
                                      max(t[0] + MIN_VALID_RUN_SEC + 1e-3,
                                          t[-1] - gap_sec - MIN_VALID_RUN_SEC)))
        i0 = int(np.searchsorted(t, start_sec))
        i1 = int(np.searchsorted(t, start_sec + gap_sec))
        i0 = max(2, i0)                  # never invalidate the first frames
        i1 = min(T - 2, max(i0 + 2, i1))
        if i1 <= i0 or occupied[max(0, i0 - 3):i1 + 3].any():
            continue
        keep = budget - invalid.sum()
        i1 = min(i1, i0 + keep)
        if i1 <= i0:
            break
        invalid[i0:i1] = True
        occupied[i0:i1] = True
    return invalid


def _stable_seed(seed: int, seq: str) -> int:
    """Deterministic 63-bit seed from (seed, seq) with no Python hash salting."""
    h = 1469598103934665603
    for ch in f"{seed}:{seq}".encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) % (2 ** 63)
    return h


# --------------------------------------------------------------------------- #
# per-sequence assembly (worker)
# --------------------------------------------------------------------------- #
def decode_sequence(zip_path, seq: str, jpegs: Sequence[str],
                    native_fps: float = NATIVE_FPS) -> dict:
    """Decode + masks for one sequence.  Returns the arrays the runner wants."""
    from src.data.clip_encoder import crop_box_with_margin, resize_shortest_center_crop

    n = len(jpegs)
    idx = sample_indices(n, native_fps)
    T = len(idx)
    frames_u8, roi_crops = [], np.zeros((T, N_ROI, 224, 224, 3), np.uint8)
    boxes = np.zeros((T, N_ROI, 4), np.float32)
    mask_ok = np.zeros(T, bool)
    ann_names = [n_.replace(JPEG_ROOT, ANN_ROOT).rsplit(".", 1)[0] + ".png"
                 for n_ in jpegs]
    with zipfile.ZipFile(zip_path) as z:
        for i, fi in enumerate(idx):
            jb = z.read(jpegs[int(fi)])
            img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{seq}: undecodable jpeg {jpegs[int(fi)]}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            frames_u8.append(resize_shortest_center_crop(img))
            try:
                mb = z.read(ann_names[int(fi)])
                mask = read_mask_bgr(mb)
            except Exception:  # noqa: BLE001
                mask = None
            box, vis = mask_to_box(mask, H, W)
            boxes[i, 0] = box
            mask_ok[i] = vis
            if vis:
                px = ((box[0] - box[2] / 2) * W, (box[1] - box[3] / 2) * H,
                      box[2] * W, box[3] * H)
                roi_crops[i, 0] = crop_box_with_margin(img, px)
    t = (idx.astype(np.float64) / native_fps).astype(np.float32)
    for i in range(1, T):
        if t[i] <= t[i - 1]:
            t[i] = np.float32(t[i - 1] + 1e-4)
    invalid = gap_plan(seq, idx, t)
    roi_valid = (mask_ok & ~invalid).astype(np.uint8)[:, None]
    return {
        "frames": np.stack(frames_u8),
        "t": t,
        "roi_crops": roi_crops,
        "roi_meta": boxes,
        "roi_valid": roi_valid,
        "lab_track": boxes,                                   # [T,1,4]
        "lab_present": mask_ok.astype(np.float32)[:, None],   # [T,1]
        "meta": {"native_fps": native_fps, "n_frames_total": n,
                 "n_frames_kept": int(T), "frame_indices": idx.tolist(),
                 "n_mask_empty": int((~mask_ok).sum()),
                 "n_simulated_gap": int(invalid.sum()),
                 "gap_intervals": _intervals(invalid)},
    }


def _intervals(mask: np.ndarray) -> List[List[int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i])
            start = None
    if start is not None:
        out.append([start, len(mask)])
    return out
