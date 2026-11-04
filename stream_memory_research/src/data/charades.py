"""Charades online event-detection task construction.

Data
----
* Annotations: ``Charades.zip`` (3.5 MB) from ai2 S3 -> ``Charades_v1_train.csv``
  (7985 videos; ``actions`` column = ``"c092 11.90 21.20;c147 0.00 12.60"``,
  i.e. class-id, onset, offset in seconds) + ``Charades_v1_classes.txt`` (157).
* Videos: ``Charades_v1_480.zip`` (16,339,546,533 B) from ai2 S3.
  Measured: 0.60 MB/s single connection, **5.73 MB/s with 16 range connections**
  -> ~44 min for 15.2 GB.  The HF repo ``jinyoungkim/Charades`` has the same
  9848 clips as individual files but at **1920x1080** (~32 MB each -> 45 GB for
  1400 clips), which is 3x more download for no benefit because CLIP sees 224px;
  so the 480p archive is the right source.  ``linxxx3/Charades_v1_480`` carries
  the same 16.3 GB zip if S3 ever fails.

Task
----
1400 videos chosen deterministically from the 7985 official-train ids with an
even stride; split 70/15/15 by contiguous blocks of the sorted selection
(Charades ids are random 5-char strings, so lexicographic order is effectively
random -> contiguous blocks are a fair random split; the split unit is the video
id, never a frame).  128 frames per video sampled uniformly over the clip; the
class label at a kept frame is the interval membership of the real annotated
segment boundaries, so the event onsets are at genuine video times rather than
at frame-grid artefacts.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

N_VIDEOS = 1400
N_FRAMES = 128
N_CLASSES = 50                 # most frequent classes kept for lab_event
MIN_SEGMENTS_PER_CLASS = 60    # a class must cover at least this many segments
SPLIT_FRACS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")
CSV_NAME = "Charades_v1_train.csv"
CLASSES_NAME = "Charades_v1_classes.txt"


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def load_classes(ann_dir) -> List[str]:
    p = Path(ann_dir) / CLASSES_NAME
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_rows(ann_dir) -> List[dict]:
    """Official-train CSV rows that carry at least one action segment."""
    p = Path(ann_dir) / CSV_NAME
    out: List[dict] = []
    with open(p, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            acts = (r.get("actions") or "").strip()
            if not acts:
                continue
            segs = []
            for part in acts.split(";"):
                bits = part.split()
                if len(bits) != 3:
                    continue
                try:
                    segs.append((bits[0], float(bits[1]), float(bits[2])))
                except ValueError:
                    continue
            if not segs:
                continue
            r["_segs"] = segs
            try:
                r["_length"] = float(r.get("length") or 0.0)
            except ValueError:
                r["_length"] = 0.0
            out.append(r)
    return out


def stride_pick(pool: Sequence, n: int) -> List:
    pool = list(pool)
    N = len(pool)
    if N == 0 or n <= 0:
        return []
    if n >= N:
        return pool
    return [pool[int(i * N / n)] for i in range(n)]


def choose_classes(rows: Sequence[dict], n_classes: int = N_CLASSES,
                   min_segments: int = MIN_SEGMENTS_PER_CLASS) -> List[str]:
    """Most frequent action classes among the *selected* videos."""
    freq: Dict[str, int] = {}
    for r in rows:
        for cid, _s, _e in r["_segs"]:
            freq[cid] = freq.get(cid, 0) + 1
    ranked = [c for c, k in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
              if k >= min_segments]
    return sorted(ranked[:n_classes])


def build_plan(rows: Sequence[dict], n_videos: int = N_VIDEOS,
               n_classes: int = N_CLASSES) -> dict:
    """Deterministic selection + split + class list.  Returns a plan dict."""
    ids = sorted(r["id"] for r in rows)
    picked = stride_pick(ids, n_videos)
    picked = sorted(picked)
    n = len(picked)
    n_train = int(round(SPLIT_FRACS[0] * n))
    n_val = int(round(SPLIT_FRACS[1] * n))
    split = {}
    for i, vid in enumerate(picked):
        if i < n_train:
            split[vid] = "train"
        elif i < n_train + n_val:
            split[vid] = "val"
        else:
            split[vid] = "test"
    by_id = {r["id"]: r for r in rows}
    sel_rows = [by_id[v] for v in picked]
    classes = choose_classes(sel_rows, n_classes)
    return {"ids": picked, "split": split, "classes": classes,
            "rows": {v: by_id[v] for v in picked}}


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def frame_labels(t: np.ndarray, segs: Sequence[Tuple[str, float, float]],
                 class_order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Interval membership at each kept frame -> lab_event [T,C], lab_seg [T]."""
    T = len(t)
    C = len(class_order)
    cidx = {c: i for i, c in enumerate(class_order)}
    ev = np.zeros((T, C), np.float32)
    onsets = np.zeros(T, np.int64)
    starts = np.array(sorted(s for _c, s, _e in segs), np.float64) if segs else \
        np.zeros(0, np.float64)
    if T > 1:
        prev_t = np.concatenate([[t[0] - (t[1] - t[0])], t[:-1]])
    else:
        prev_t = t - 1e-3
    for cid, s, e in segs:
        j = cidx.get(cid)
        if j is None or e <= s:
            continue
        m = (t >= s) & (t < e)
        ev[m, j] = 1.0
    if len(starts):
        # frame t is an onset frame if a segment starts in (prev_t, t]
        idx = np.searchsorted(starts, prev_t, side="right")
        idx2 = np.searchsorted(starts, t, side="right")
        onsets = (idx2 > idx).astype(np.int64)
    return ev, onsets


def event_stats(rows: Sequence[dict], classes: Sequence[str],
                n_frames: int = N_FRAMES) -> dict:
    """Rough positive-rate estimate to sanity-check class selection."""
    keep = set(classes)
    tot_seg = sum(1 for r in rows for c, _s, _e in r["_segs"] if c in keep)
    tot_pos = 0
    tot_frames = 0
    for r in rows:
        dur = max(1e-3, r["_length"])
        t = np.linspace(0, max(1e-3, dur - 1e-3), n_frames)
        ev, _ = frame_labels(t, r["_segs"], classes)
        tot_pos += int(ev.sum())
        tot_frames += n_frames
    return {"n_segments_kept": tot_seg,
            "positive_fraction": tot_pos / max(1, tot_frames),
            "n_classes": len(classes)}


# --------------------------------------------------------------------------- #
# zip access
# --------------------------------------------------------------------------- #
def find_video_member(zf: zipfile.ZipFile, vid: str) -> str | None:
    """Locate ``<vid>.mp4`` anywhere in the Charades archive."""
    cands = (f"Charades_v1_480/{vid}.mp4", f"{vid}.mp4",
             f"Charades/{vid}.mp4", f"Charades_v1_480/Charades_v1_480/{vid}.mp4")
    names = set(zf.namelist())
    for c in cands:
        if c in names:
            return c
    tail = f"/{vid}.mp4"
    for n in names:
        if n.endswith(tail):
            return n
    return None
