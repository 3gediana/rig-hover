"""Build the Charades **pixel** cache for the pixel-to-event network.

Why this exists
---------------
The pixel-to-event network (``src/models/pixel_event.py`` /
``DESIGN-PIXEL-EVENT-NETWORK.md``) consumes raw greyscale frames, not cached CLIP
features.  This script materialises a small fixed-size frame cache so training
never has to touch the 16.3 GB archive: ``frames uint8 [96,160,160]`` per clip in
``G:\\stream_memory_data\\pixel_cache\\charades_v1\\<vid>.npz`` plus the event /
onset labels derived from the official Charades annotations.

Pipeline (single process, CPU only, no torch)
---------------------------------------------
1. ``src.data.charades.load_rows`` / ``build_plan(rows, n_videos=700)`` gives the
   deterministic selection (even stride over the sorted official-train ids) and
   the 70/15/15 contiguous-block split on video ids.
2. ``charades.choose_classes(sel_rows, n_classes=25)`` gives the event classes
   (see ``MIN_SEGMENTS`` note below for why the threshold is passed explicitly).
3. The mp4 is streamed out of ``Charades_v1_480.zip`` into memory and decoded with
   PyAV from a ``BytesIO`` (no disk extraction; falls back to a temp file for the
   rare file whose demuxer demands a real path).
4. Every decoded frame is timestamped from its own PTS; each of the 96 target
   times ``t = np.linspace(0, length - 1e-3, 96)`` is mapped to the frame whose
   presentation time is nearest (single pass, streaming lookahead -- no frame is
   ever held in RAM twice).  That is the "decode everything, take the frame at
   time t" rule from the task.
5. Selected frames are converted with ``cv2.cvtColor(BGR2GRAY)`` and
   ``cv2.resize(..., (160,160), INTER_AREA)``; labels come from
   ``charades.frame_labels(t, segs, classes)``.
6. The ``.npz`` (frames / t / event / onset) is written atomically, the index
   JSON is produced in the second stage.

Resumable: existing valid ``.npz`` files are skipped, failed clips are recorded
and abandoned instead of aborting the run.

Usage
-----
    python scripts/build_charades_pixel_cache.py probe      # 5 clips + PNGs
    python scripts/build_charades_pixel_cache.py full       # all 700 clips
    python scripts/build_charades_pixel_cache.py manifest   # index + per-class counts JSON (no decode)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import av  # noqa: E402
import cv2  # noqa: E402

from src.data import charades  # noqa: E402

# libav prints thousands of "marker does not match" lines on damaged macroblocks;
# PyAV's QUIET level is PANIC.  Real failures still raise.
try:
    av.logging.set_libav_level(av.logging.PANIC)
    av.logging.set_level(av.logging.PANIC)
except Exception:  # pragma: no cover
    pass

ZIP_PATH = Path(r"G:\stream_memory_data\videos\charades\Charades_v1_480.zip")
ANN_DIR = Path(r"G:\stream_memory_data\videos\charades\ann\Charades")
CACHE_ROOT = Path(r"G:\stream_memory_data\pixel_cache\charades_v1")
EVID_DIR = ROOT / "evidence" / "pixel_cache_probe"
INDEX_PATH = ROOT / "data_manifests" / "charades_pixel_index.json"
COUNTS_PATH = ROOT / "data_manifests" / "charades_pixel_index_counts.json"
LOG_PATH = CACHE_ROOT / "_build_log.txt"
SUMMARY_PATH = CACHE_ROOT / "_build_summary.json"

N_FRAMES = 96          # T
RES = 160              # grey frame side
N_VIDEOS = 700
N_CLASSES = 25
#: With the library default ``min_segments=60`` the 700-video subset has only 19
#: qualifying classes.  ``pixel_event.py`` defaults to ``n_event_classes=25``, so
#: the threshold is lowered until exactly N_CLASSES classes qualify (still the
#: library ``choose_classes`` ranking, nothing re-implemented).  The value used is
#: recorded in the manifest.
MIN_SEGMENTS_CANDIDATES = (60, 50, 40, 30, 25, 20, 15, 10, 5, 1)
PROBE_FRAMES = (0, 16, 32, 64, 95)
FIELDS = ("frames", "t", "event", "onset")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
_LOG_FH = None


def log(msg: str) -> None:
    global _LOG_FH
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        if _LOG_FH is None:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            _LOG_FH = open(LOG_PATH, "a", encoding="utf-8")
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()
    except Exception:  # noqa: BLE001 - logging must never kill the build
        pass


class VideoSkip(RuntimeError):
    """A clip that cannot be cached; recorded and skipped, never fatal."""


# --------------------------------------------------------------------------- #
# plan (selection / split / classes)
# --------------------------------------------------------------------------- #
def load_plan(verbose: bool = False) -> dict:
    rows = charades.load_rows(ANN_DIR)
    plan = charades.build_plan(rows, n_videos=N_VIDEOS)
    if len(plan["ids"]) != N_VIDEOS:  # pragma: no cover - guard, should not happen
        raise SystemExit(f"build_plan returned {len(plan['ids'])} ids, wanted {N_VIDEOS}")
    sel_rows = [plan["rows"][v] for v in plan["ids"]]
    classes: List[str] = []
    used_min_seg = None
    for ms in MIN_SEGMENTS_CANDIDATES:
        cand = charades.choose_classes(sel_rows, n_classes=N_CLASSES, min_segments=ms)
        if verbose:
            log(f"[plan] choose_classes(n_classes={N_CLASSES}, min_segments={ms}) -> {len(cand)}")
        if len(cand) == N_CLASSES:
            classes, used_min_seg = cand, ms
            break
        if len(cand) > len(classes):
            classes, used_min_seg = cand, ms
    if len(classes) != N_CLASSES:
        log(f"[plan] WARNING: only {len(classes)} classes qualify (wanted {N_CLASSES})")
    plan["classes"] = classes
    plan["min_segments"] = used_min_seg
    counts: Dict[str, int] = {}
    for v in plan["ids"]:
        counts[plan["split"][v]] = counts.get(plan["split"][v], 0) + 1
    plan["counts"] = counts
    return plan


# --------------------------------------------------------------------------- #
# decoding: uniform-in-time frame selection
# --------------------------------------------------------------------------- #
def _gray_resized(bgr: np.ndarray, res: int = RES) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if g.shape[0] != res or g.shape[1] != res:
        g = cv2.resize(g, (res, res), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(g)


@contextlib.contextmanager
def open_clip(data: bytes):
    """Open a clip from memory; temp file only if the demuxer refuses BytesIO."""
    try:
        container = av.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        tmpdir = tempfile.mkdtemp(prefix="smr_pix_")
        tmp = Path(tmpdir) / "clip.mp4"
        tmp.write_bytes(data)
        try:
            container = av.open(str(tmp))
        except Exception:  # noqa: BLE001
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        try:
            yield container
        finally:
            with contextlib.suppress(Exception):
                container.close()
            shutil.rmtree(tmpdir, ignore_errors=True)
        return
    try:
        yield container
    finally:
        with contextlib.suppress(Exception):
            container.close()


def decode_uniform(data: bytes, t_target: np.ndarray, res: int = RES) -> Tuple[np.ndarray, dict]:
    """Decode every frame once and return the frame nearest each target time.

    ``t_target`` is strictly increasing in seconds.  Returns
    ``(frames uint8 [T,res,res], meta)``.  A target past the last decoded frame is
    clamped to the last frame (counted in ``meta['n_clamped']``).
    """
    T = len(t_target)
    out: List[np.ndarray] = []
    meta: dict = {"fps": 0.0, "n_decoded": 0, "duration": 0.0,
                  "last_frame_t": 0.0, "n_clamped": 0, "decode_error": None,
                  "width": 0, "height": 0}
    with open_clip(data) as container:
        if not container.streams.video:
            raise VideoSkip("no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if fps <= 0:
            fps = float(stream.guessed_rate) if getattr(stream, "guessed_rate", None) else 25.0
        meta["fps"] = fps
        meta["width"] = int(stream.codec_context.width)
        meta["height"] = int(stream.codec_context.height)
        dur = float(container.duration) / 1e6 if container.duration else 0.0
        prev_bgr: Optional[np.ndarray] = None
        prev_t: Optional[float] = None
        k = 0
        n_dec = 0
        last_t = 0.0
        try:
            for frame in container.decode(stream):
                ft: Optional[float] = None
                try:
                    ft = float(frame.time) if frame.time is not None else None
                except Exception:  # noqa: BLE001
                    ft = None
                if ft is None:
                    ft = n_dec / fps
                bgr = frame.to_ndarray(format="bgr24")
                n_dec += 1
                last_t = ft
                if prev_bgr is not None:
                    mid = 0.5 * (prev_t + ft)
                    while k < T and t_target[k] <= mid:
                        out.append(_gray_resized(prev_bgr, res))
                        k += 1
                prev_bgr, prev_t = bgr, ft
        except Exception as exc:  # noqa: BLE001 - keep whatever decoded
            meta["decode_error"] = f"{type(exc).__name__}: {exc}"
        if prev_bgr is not None:
            while k < T:
                out.append(_gray_resized(prev_bgr, res))
                k += 1
        meta["n_decoded"] = n_dec
        meta["last_frame_t"] = float(last_t)
        meta["n_clamped"] = int(np.sum(t_target > last_t)) if n_dec else T
        meta["duration"] = dur if dur > 0 else (n_dec / fps if fps > 0 else 0.0)
    if len(out) != T:  # pragma: no cover - drain guarantees this
        raise VideoSkip(f"decoded only {len(out)}/{T} target frames")
    frames = np.stack(out).astype(np.uint8)
    return frames, meta


# --------------------------------------------------------------------------- #
# one clip -> cache payload
# --------------------------------------------------------------------------- #
def build_one(zf: zipfile.ZipFile, vid: str, row: dict, classes: Sequence[str]) -> dict:
    member = charades.find_video_member(zf, vid)
    if member is None:
        raise VideoSkip("not found in Charades_v1_480.zip")
    try:
        data = zf.read(member)
    except Exception as exc:  # noqa: BLE001
        raise VideoSkip(f"zip read failed: {type(exc).__name__}: {exc}") from exc
    if not data:
        raise VideoSkip("empty zip member")
    length = float(row["_length"])
    if not np.isfinite(length) or length <= 0:
        raise VideoSkip(f"bad csv length {length!r}")
    t = np.linspace(0.0, length - 1e-3, N_FRAMES).astype(np.float32)
    if len(t) > 1 and not np.all(np.diff(t) > 0):  # pragma: no cover
        raise VideoSkip("non-monotonic target times")
    frames, meta = decode_uniform(data, t.astype(np.float64), RES)
    if meta["n_decoded"] < 24:
        raise VideoSkip(f"only {meta['n_decoded']} frames decoded")
    if meta["decode_error"]:
        raise VideoSkip(f"decode error: {meta['decode_error']}")
    ev, onset = charades.frame_labels(t, row["_segs"], classes)
    event = ev.astype(np.uint8)
    onset = onset.astype(np.uint8)
    return {"frames": frames, "t": t, "event": event, "onset": onset,
            "meta": meta, "member": member, "length": length}


def npz_path(vid: str) -> Path:
    return CACHE_ROOT / f"{vid}.npz"


def is_cached(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as z:
            return (set(FIELDS).issubset(z.files)
                    and z["frames"].shape == (N_FRAMES, RES, RES)
                    and z["frames"].dtype == np.uint8
                    and z["t"].shape == (N_FRAMES,)
                    and z["event"].dtype == np.uint8
                    and z["onset"].dtype == np.uint8)
    except Exception:  # noqa: BLE001
        return False


def save_payload(payload: dict, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, frames=payload["frames"], t=payload["t"],
             event=payload["event"], onset=payload["onset"])
    os.replace(tmp, path)
    return path.stat().st_size


# --------------------------------------------------------------------------- #
# probe mode: 5 clips, PNG evidence, distribution + speed check
# --------------------------------------------------------------------------- #
def _png_dir() -> Path:
    EVID_DIR.mkdir(parents=True, exist_ok=True)
    return EVID_DIR


def run_probe(zf: zipfile.ZipFile, plan: dict, n: int = 5) -> dict:
    ids = plan["ids"]
    # spread over the three splits so the probe exercises every split's files
    picks = []
    for frac in (0.02, 0.15, 0.30, 0.72, 0.90):
        picks.append(ids[int(frac * len(ids))])
    picks = list(dict.fromkeys(picks))[:n]
    while len(picks) < n:  # pragma: no cover - only if n > len(ids)
        picks.append(ids[len(picks)])
    png = _png_dir()
    rows = {v: plan["rows"][v] for v in ids}
    classes = plan["classes"]

    log(f"[probe] {len(picks)} clips {picks}")
    log(f"[probe] classes({len(classes)}) = {classes}")
    timings, total_bytes = [], 0
    all_ok = True
    for vid in picks:
        t0 = time.perf_counter()
        payload = build_one(zf, vid, rows[vid], classes)
        size = save_payload(payload, npz_path(vid))
        dt = time.perf_counter() - t0
        timings.append(dt)
        total_bytes += size
        meta = payload["meta"]
        ev = payload["event"].astype(np.float32)
        log(f"[probe] {vid}: {dt:.3f}s {size/1e6:.2f} MB  fps={meta['fps']:.3f} "
            f"decoded={meta['n_decoded']} dur={meta['duration']:.3f}s "
            f"csv_length={payload['length']:.3f}s last_t={meta['last_frame_t']:.3f}s "
            f"clamped={meta['n_clamped']}/{N_FRAMES}  "
            f"event_pos={int(ev.sum())} onset={int(payload['onset'].sum())}")
        # per-frame distributions + PNGs
        fr = payload["frames"]
        for fi in PROBE_FRAMES:
            f = fr[fi]
            f32 = f.astype(np.float32)
            log(f"[probe]   {vid} frame {fi:02d}: shape={f.shape} dtype={f.dtype} "
                f"min={int(f.min())} max={int(f.max())} "
                f"mean={f32.mean():.2f} std={f32.std():.2f} "
                f"n_unique={int(np.unique(f).size)}")
            cv2.imwrite(str(png / f"{vid}_f{fi:03d}.png"), f)
        sheet = np.concatenate([fr[i] for i in PROBE_FRAMES], axis=1)
        cv2.imwrite(str(png / f"{vid}_sheet.png"), sheet)
        # readback check
        with np.load(npz_path(vid)) as z:
            rb = z["frames"]
            rt = z["t"]
            ok = True
            msgs = []
            if rb.shape != (N_FRAMES, RES, RES) or rb.dtype != np.uint8:
                ok = False; msgs.append(f"shape/dtype {rb.shape} {rb.dtype}")
            if not np.all(np.diff(rt) > 0):
                ok = False; msgs.append("t not strictly increasing")
            if not np.all(np.isfinite(rt)):
                ok = False; msgs.append("t not finite")
            if rt[0] < 0:
                ok = False; msgs.append("t[0] < 0")
            f32 = rb[32].astype(np.float32)
            if f32.std() < 1.0:
                ok = False; msgs.append(f"frame 32 constant (std={f32.std():.2f})")
            if f32.mean() < 2 or f32.mean() > 253:
                ok = False; msgs.append(f"frame 32 near black/white (mean={f32.mean():.2f})")
            if int(np.unique(rb[32]).size) < 8:
                ok = False; msgs.append(f"frame 32 only {np.unique(rb[32]).size} values")
            # frames should not all be identical (video actually sampled over time)
            d = np.abs(rb[95].astype(np.int32) - rb[0].astype(np.int32)).mean()
            if d < 0.5:
                ok = False; msgs.append(f"frame 0 and 95 nearly identical (mad={d:.2f})")
            all_ok = all_ok and ok
            log(f"[probe]   {vid} readback: {'OK' if ok else 'FAIL ' + '; '.join(msgs)} "
                f"t[0]={float(rt[0]):.4f} t[-1]={float(rt[-1]):.4f} "
                f"mean|f95-f0|={d:.2f}")
    speed = len(timings) / sum(timings) * 60.0
    log(f"[probe] speed {speed:.1f} videos/min "
        f"(mean {np.mean(timings):.3f}s, max {np.max(timings):.3f}s) "
        f"-> ETA for {N_VIDEOS} videos ~ {N_VIDEOS / speed:.1f} min")
    log(f"[probe] {len(picks)} npz, {total_bytes/1e6:.3f} MB total, "
        f"{total_bytes/len(picks)/1e6:.3f} MB/video -> "
        f"{N_VIDEOS*total_bytes/len(picks)/1e9:.2f} GB projected")
    log(f"[probe] all readback checks {'PASSED' if all_ok else 'FAILED'}")
    return {"ids": picks, "timings": timings, "speed_videos_per_min": speed,
            "bytes": total_bytes, "all_ok": bool(all_ok)}


# --------------------------------------------------------------------------- #
# full mode
# --------------------------------------------------------------------------- #
def run_full(zf: zipfile.ZipFile, plan: dict, resume: bool = True) -> dict:
    ids = plan["ids"]
    rows = plan["rows"]
    classes = plan["classes"]
    n_total = len(ids)
    t_start = time.perf_counter()
    done = skipped = interrupted = 0
    bytes_total = 0
    skip_reasons: Dict[str, str] = {}
    speeds: List[float] = []
    for i, vid in enumerate(ids):
        path = npz_path(vid)
        if resume and is_cached(path):
            done += 1
            bytes_total += path.stat().st_size
            continue
        t0 = time.perf_counter()
        try:
            payload = build_one(zf, vid, rows[vid], classes)
            size = save_payload(payload, path)
            bytes_total += size
            done += 1
        except KeyboardInterrupt:  # pragma: no cover
            interrupted += 1
            log(f"[full] interrupted at {i}/{n_total}")
            break
        except VideoSkip as exc:
            skipped += 1
            skip_reasons[vid] = str(exc)
            log(f"[full] SKIP {vid}: {exc}")
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            skip_reasons[vid] = f"{type(exc).__name__}: {exc}"
            log(f"[full] SKIP {vid}: {skip_reasons[vid]}")
            log(traceback.format_exc())
        dt = time.perf_counter() - t0
        if dt > 0:
            speeds.append(dt)
        if (i + 1) % 25 == 0 or i + 1 == n_total:
            el = time.perf_counter() - t_start
            rate = (i + 1) / el
            eta = (n_total - i - 1) / rate if rate > 0 else float("nan")
            log(f"[full] {i+1}/{n_total} cached={done} skipped={skipped} "
                f"{rate:.2f} vids/s eta={eta/60:.1f} min "
                f"bytes={bytes_total/1e9:.2f} GB")
    elapsed = time.perf_counter() - t_start
    summary = {
        "n_planned": n_total, "n_cached": done, "n_skipped": skipped,
        "n_interrupted": interrupted, "bytes_total": bytes_total,
        "elapsed_s": elapsed,
        "videos_per_min": (n_total / elapsed * 60.0) if elapsed > 0 else 0.0,
        "mean_decode_s": float(np.mean(speeds)) if speeds else None,
        "skip_reasons": skip_reasons,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    log(f"[full] done: {summary}")
    return summary


# --------------------------------------------------------------------------- #
# manifest mode
# --------------------------------------------------------------------------- #
def build_manifest(plan: dict) -> Tuple[dict, List[dict]]:
    """Validate every cached npz, write the index JSON, return (payload, per_class)."""
    rows = plan["rows"]
    classes = plan["classes"]
    class_set = set(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    split_names = ("train", "val", "test")
    split_stats: Dict[str, dict] = {
        s: {"n_videos": 0, "n_frames": 0, "n_positive_cells": 0,
            "n_frames_with_any_event": 0, "n_onsets": 0, "n_segments": 0}
        for s in split_names}
    seqs: List[dict] = []
    failed: Dict[str, str] = {}
    skipped_existing = 0
    pos_sum = 0
    pos_frames = 0
    onset_sum = 0
    frame_total = 0
    bad: Dict[str, str] = {}
    pos_cells_by_class = np.zeros(C, np.int64)   # cached-grid positive frames
    videos_with_event = np.zeros(C, np.int64)    # videos with >=1 positive frame
    for vid in plan["ids"]:
        path = npz_path(vid)
        if not is_cached(path):
            if path.exists():
                bad[vid] = "npz exists but failed validation"
            else:
                skipped_existing += 1
            continue
        try:
            with np.load(path) as z:
                frames = z["frames"]; t = z["t"]
                ev = z["event"]; onset = z["onset"]
            if frames.shape != (N_FRAMES, RES, RES):
                bad[vid] = f"frames shape {frames.shape}"; continue
            if ev.shape != (N_FRAMES, len(classes)):
                bad[vid] = f"event shape {ev.shape} != {(N_FRAMES, len(classes))}"; continue
            if t.shape != (N_FRAMES,) or not np.all(np.diff(t) > 0):
                bad[vid] = "t not strictly increasing"; continue
            if not np.all(np.isfinite(t)):
                bad[vid] = "t not finite"; continue
            pos_sum += int(ev.sum())
            pos_frames += int((ev.sum(axis=1) > 0).sum())
            onset_sum += int(onset.sum())
            frame_total += N_FRAMES
            pos_cells_by_class += ev.sum(axis=0).astype(np.int64)
            videos_with_event += (ev.sum(axis=0) > 0).astype(np.int64)
            st = split_stats[plan["split"][vid]]
            st["n_videos"] += 1
            st["n_frames"] += N_FRAMES
            st["n_positive_cells"] += int(ev.sum())
            st["n_frames_with_any_event"] += int((ev.sum(axis=1) > 0).sum())
            st["n_onsets"] += int(onset.sum())
        except Exception as exc:  # noqa: BLE001
            bad[vid] = f"load {type(exc).__name__}: {exc}"
            continue
        row = rows[vid]
        segs = [[cid, float(s), float(e)] for cid, s, e in row["_segs"] if cid in class_set]
        seqs.append({
            "seq_id": vid, "source_id": vid, "split": plan["split"][vid],
            "path": str(path).replace("\\", "/"), "T": N_FRAMES,
            "length": float(row["_length"]), "segs": segs,
        })
    counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for s in seqs:
        counts[s["split"]] = counts.get(s["split"], 0) + 1
    # ---- per-class segment statistics from the CSV annotations (no decode) ----
    n_segments = np.zeros(C, np.int64)          # raw csv segments
    n_segments_eff = np.zeros(C, np.int64)      # segments with offset > onset (labelable)
    n_videos_with_seg = np.zeros(C, np.int64)
    n_videos_with_2plus = np.zeros(C, np.int64)
    for s in seqs:
        vid = s["seq_id"]
        cnt_raw = np.zeros(C, np.int64)
        cnt_eff = np.zeros(C, np.int64)
        for cid, _s, _e in rows[vid]["_segs"]:
            j = cidx.get(cid)
            if j is None:
                continue
            cnt_raw[j] += 1
            if _e > _s:
                cnt_eff[j] += 1
        n_segments += cnt_raw
        n_segments_eff += cnt_eff
        n_videos_with_seg += (cnt_eff > 0).astype(np.int64)
        n_videos_with_2plus += (cnt_eff >= 2).astype(np.int64)
        split_stats[s["split"]]["n_segments"] += int(cnt_eff.sum())
    per_class: List[dict] = []
    for i, c in enumerate(classes):
        per_class.append({
            "class_id": c, "class_index": i,
            "n_positive_frames": int(pos_cells_by_class[i]),
            "n_videos_with_event": int(videos_with_event[i]),
            "n_segments": int(n_segments[i]),
            "n_segments_labeled_effective": int(n_segments_eff[i]),
            "n_videos_with_segment": int(n_videos_with_seg[i]),
            "n_videos_with_2plus_segments": int(n_videos_with_2plus[i]),
        })
    for name in split_names:
        st = split_stats[name]
        nf = max(1, st["n_frames"])
        st["positive_fraction"] = st["n_positive_cells"] / nf
        st["positive_rate_per_cell"] = st["n_positive_cells"] / max(1, nf * C)
        st["frame_event_rate"] = st["n_frames_with_any_event"] / nf
        st["onset_rate"] = st["n_onsets"] / nf
    payload = {
        "dataset": "charades_pixel",
        "resolution": RES,
        "n_frames": N_FRAMES,
        "classes": list(classes),
        "seqs": seqs,
        # ---- provenance / stats (extra keys; harmless for json.load) ----
        "task": "event",
        "counts": counts,
        "split_stats": split_stats,
        "n_cached": len(seqs),
        "n_planned": len(plan["ids"]),
        "failed": {**failed, **bad},
        "n_missing": skipped_existing,
        "cache_root": str(CACHE_ROOT).replace("\\", "/"),
        "source_zip": str(ZIP_PATH).replace("\\", "/"),
        "ann_dir": str(ANN_DIR).replace("\\", "/"),
        "class_rule": (
            f"charades.choose_classes(selected rows, n_classes={N_CLASSES}, "
            f"min_segments={plan['min_segments']}); class ids are the literal "
            f"'cXYZ' strings from Charades_v1_train.csv actions column. "
            f"The library default min_segments={charades.MIN_SEGMENTS_PER_CLASS} "
            f"leaves only 19 classes on this 700-video subset, so the threshold "
            f"was lowered until {N_CLASSES} classes qualified"),
        "selection_rule": (
            f"from the {len(plan['ids'])} charades.load_rows() clips: "
            f"charades.build_plan(rows, n_videos={N_VIDEOS}) = even stride over the "
            f"lexicographically sorted official-train ids (idx_i=floor(i*N/n)), "
            f"split 70/15/15 by contiguous blocks of that sorted selection; the "
            f"split unit is the video id, never a frame"),
        "sampling_rule": (
            f"decode every frame once; for t_k = linspace(0, length-1e-3, {N_FRAMES}) "
            f"take the decoded frame whose PTS is nearest t_k (streaming nearest-time "
            f"mapping; targets past the last frame clamp to it)"),
        "timestamp_rule": (
            "t = np.linspace(0, csv_length - 1e-3, 96) float32 seconds, strictly "
            "increasing; csv_length from Charades_v1_train.csv"),
        "preprocess": (
            f"cv2.cvtColor(frame.to_ndarray(format='bgr24'), COLOR_BGR2GRAY) then "
            f"cv2.resize(..., ({RES},{RES}), INTER_AREA), uint8"),
        "label_rule": (
            "charades.frame_labels(t, row['_segs'], classes): event[T,C] uint8 = "
            "1 iff t in [onset, offset) of that class; onset[T] uint8 = 1 iff any "
            "segment starts in (t_{k-1}, t_k]"),
        "stats": {
            "n_seqs": len(seqs),
            "n_frames_total": frame_total,
            "n_positive_cells": pos_sum,
            "positive_fraction": pos_sum / max(1, frame_total),          # mean active classes per frame (project convention)
            "positive_rate_per_cell": pos_sum / max(1, frame_total * len(classes)),
            "n_frames_with_any_event": pos_frames,
            "frame_event_rate": pos_frames / max(1, frame_total),
            "n_onsets": onset_sum,
            "onset_rate": onset_sum / max(1, frame_total),
        },
        "build": {
            "script": "scripts/build_charades_pixel_cache.py",
            "npz_compression": "none (np.savez; fast random-access loads)",
            "device": "cpu", "torch_used": False, "n_processes": 1,
        },
    }
    if "build_summary" in plan:
        payload["build_finished"] = plan["build_summary"]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_name(INDEX_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, INDEX_PATH)
    log(f"[manifest] {INDEX_PATH} n_seqs={len(seqs)} counts={counts} "
        f"stats={payload['stats']}")
    log(f"[manifest] failed={len(payload['failed'])} missing={skipped_existing} "
        f"bad={len(bad)}")
    return payload, per_class


def build_counts(plan: dict, per_class: List[dict]) -> dict:
    """Sibling stats file: per-class event / segment counts (Lead request)."""
    fields = ("n_positive_frames", "n_videos_with_event", "n_segments",
              "n_segments_labeled_effective", "n_videos_with_segment",
              "n_videos_with_2plus_segments")
    payload = {
        "dataset": "charades_pixel",
        "n_seqs": len(plan["ids"]),
        "classes": list(plan["classes"]),
        "description": (
            "per-class supervision statistics on the cached frame grid. "
            "n_positive_frames = number of cached frames (T=96, video id as split "
            "unit) whose event label is 1 for this class (from the npz event "
            "array). n_videos_with_event = videos with >=1 such frame. "
            "n_segments = raw annotated segments of this class in the 700 cached "
            "videos (Charades_v1_train.csv actions column). "
            "n_segments_labeled_effective = segments with offset > onset, i.e. the "
            "ones frame_labels can actually mark. n_videos_with_segment / "
            "n_videos_with_2plus_segments = videos with >=1 / >=2 such segments "
            "(count-task supervision availability)."),
        "class_rule": (
            f"charades.choose_classes(selected rows, n_classes={N_CLASSES}, "
            f"min_segments={plan['min_segments']}); class ids are the literal "
            f"'cXYZ' strings from Charades_v1_train.csv"),
        "per_class": per_class,
        "totals": {k: int(sum(r[k] for r in per_class)) for k in fields},
    }
    COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = COUNTS_PATH.with_name(COUNTS_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, COUNTS_PATH)
    log(f"[counts] {COUNTS_PATH} n_classes={len(per_class)} totals={payload['totals']}")
    return payload


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="probe",
                    choices=("probe", "full", "manifest"))
    ap.add_argument("--n-probe", type=int, default=5)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"[main] mode={args.mode} zip={ZIP_PATH} cache={CACHE_ROOT}")
    if not ZIP_PATH.exists():
        raise SystemExit(f"missing zip {ZIP_PATH}")
    if not ANN_DIR.exists():
        raise SystemExit(f"missing annotations {ANN_DIR}")

    plan = load_plan(verbose=True)
    log(f"[plan] {len(plan['ids'])} ids, classes={len(plan['classes'])} "
        f"(min_segments={plan['min_segments']}), counts={plan['counts']}")

    zf = zipfile.ZipFile(ZIP_PATH)
    try:
        if args.mode == "probe":
            run_probe(zf, plan, n=args.n_probe)
        else:
            if args.mode == "full":
                run_full(zf, plan, resume=not args.no_resume)
            if SUMMARY_PATH.exists():
                plan["build_summary"] = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            _, per_class = build_manifest(plan)
            build_counts(plan, per_class)
    finally:
        zf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
