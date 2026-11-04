"""PyAV decoding helpers: frame counting, uniform index sampling, sparse decode.

Design notes
------------
* ``decord`` is not installed and OpenCV leaks/fails on some of these files, so
  everything goes through PyAV (libav) which handles avi/mp4/mkv/mov.
* The datasets here span very different frame counts per sequence
  (UCF101 ~175 frames, MOT17 600-1050, Charades ~800, DAVIS 50-120), and the
  number of frames we actually want is often far smaller than the total.  So
  there are two decode strategies and :func:`decode_selected` picks one:
    - *sequential*: decode everything once, keep wanted indices.  Best when
      most frames are wanted (dense sampling).
    - *seek*: for each target index, seek to the preceding keyframe and decode
      forward a few frames.  Best when sampling sparsely (Charades 128/900).
* Timestamps are always derived from the container's own average frame rate:
  ``t = frame_index / fps``.  We never assume uniform when writing (the fps is
  stored), but the sampling itself is defined on frame indices, so t is exact
  for CFR video and approximately right for VFR (recorded in the manifest).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import av
    # UCF101 contains many MPEG-4 files with slightly damaged macroblocks; ffmpeg
    # recovers but prints thousands of "marker does not match f_code" lines from
    # libav at ERROR level.  PyAV has no QUIET level -- PANIC is the silencing
    # one -- and the libav callback is separate from the PyAV callback, so both
    # must be set.  Real failures still surface as Python exceptions.
    try:
        av.logging.set_libav_level(av.logging.PANIC)
        av.logging.set_level(av.logging.PANIC)
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover
    av = None


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    duration: float
    width: int
    height: int
    codec: str
    estimated_n: bool = False      # True when n_frames came from duration*fps


def _open(path):
    """Open a video by path **or by any seekable file-like object** (BytesIO).

    PyAV accepts a file object directly, which lets callers decode straight out
    of an archive in memory -- important here because G: is a slow USB drive and
    materialising thousands of small avi files there was the extraction
    bottleneck (measured 0.25 videos/s with disk extraction vs 3.45 videos/s
    in-memory).
    """
    if av is None:
        raise VideoError("PyAV is not installed")
    try:
        return av.open(path if hasattr(path, "read") else str(path))
    except Exception as e:  # noqa: BLE001
        raise VideoError(f"cannot open {getattr(path, 'name', path)}: "
                         f"{type(e).__name__}: {e}") from e


def probe_video(path, count_frames: bool = False) -> VideoInfo:
    """Read fps / frame count / size without decoding (unless ``count_frames``)."""
    with _open(path) as c:
        if not c.streams.video:
            raise VideoError(f"{path}: no video stream")
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else 0.0
        if fps <= 0:
            fps = float(s.guessed_rate) if getattr(s, "guessed_rate", None) else 25.0
        n = int(s.frames) if s.frames else 0
        dur = float(c.duration) / 1e6 if c.duration else 0.0
        if dur <= 0 and n > 0 and fps > 0:
            dur = n / fps
        estimated = False
        if n <= 0:
            if dur > 0:
                n = int(round(dur * fps))
                estimated = True
            elif count_frames:
                n = 0
                for _f in c.decode(s):
                    n += 1
            else:
                raise VideoError(f"{path}: cannot determine frame count")
        return VideoInfo(fps=fps, n_frames=max(1, n), duration=dur,
                         width=int(s.codec_context.width),
                         height=int(s.codec_context.height),
                         codec=s.codec_context.name, estimated_n=estimated)


def sample_indices_uniform(n_total: int, T: int) -> np.ndarray:
    """Uniform *inclusive* sampling of T frame indices from ``n_total`` frames.

    Endpoints are included and indices are strictly increasing (no repeats),
    which is required because ``t`` must be strictly increasing in the npz.
    For n_total < T every frame is used once (T is clamped to n_total).
    """
    T = int(min(T, n_total))
    if T <= 1:
        return np.array([0], dtype=np.int64)
    idx = np.floor(np.arange(T) * (n_total - 1) / (T - 1) + 0.5).astype(np.int64)
    idx = np.clip(idx, 0, n_total - 1)
    # guarantee strict increase (rounding can collapse neighbours when T~n_total)
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    if idx[-1] >= n_total:
        idx = np.arange(n_total, dtype=np.int64) if T >= n_total else \
            np.linspace(0, n_total - 1, T).astype(np.int64)
        idx = np.unique(idx)
    return idx


def sample_indices_time(n_total: int, fps: float, T: int) -> np.ndarray:
    """Uniform *in time* sampling (same as uniform over indices for CFR)."""
    return sample_indices_uniform(n_total, T)


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def _frame_to_rgb(frame) -> np.ndarray:
    arr = frame.to_ndarray(format="rgb24")
    return np.ascontiguousarray(arr)


def _decode_sequential(path, wanted: np.ndarray) -> Dict[int, np.ndarray]:
    """One pass over the stream, keep only ``wanted`` indices."""
    wanted_set = {int(i): k for k, i in enumerate(wanted)}
    last = int(wanted[-1])
    out: Dict[int, np.ndarray] = {}
    with _open(path) as c:
        s = c.streams.video[0]
        i = 0
        for frame in c.decode(s):
            if i in wanted_set:
                out[i] = _frame_to_rgb(frame)
            i += 1
            if i > last:
                break
    return out


def _decode_seek(path, wanted: np.ndarray, fps: float, info: VideoInfo,
                 max_span: int = 40) -> Dict[int, np.ndarray]:
    """Seek per target index: jump to the keyframe before it, decode forward."""
    out: Dict[int, np.ndarray] = {}
    with _open(path) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        if fps <= 0:
            fps = info.fps
        for target in wanted:
            target = int(target)
            t_sec = target / fps
            try:
                c.seek(int(max(0.0, t_sec - 1e-6) / tb), stream=s,
                       backward=True, any_frame=False)
            except Exception:  # noqa: BLE001
                c.seek(0, stream=s)
            got = None
            n_read = 0
            try:
                for frame in c.decode(s):
                    n_read += 1
                    ft = float(frame.pts * s.time_base) if frame.pts is not None \
                        else (n_read - 1) / fps
                    if ft + 0.5 / fps >= t_sec:
                        got = frame
                        break
                    if n_read > max_span + int(2 * fps):
                        break
            except Exception:  # noqa: BLE001
                pass
            if got is not None:
                out[target] = _frame_to_rgb(got)
    return out


def decode_selected(path, indices: Sequence[int], info: Optional[VideoInfo] = None,
                    prefer_seek: Optional[bool] = None) -> List[np.ndarray]:
    """Decode exactly the requested frame indices, in order.

    Missing frames (decoder faults or truncated files) are replaced by the
    nearest successfully decoded frame so the returned list always has
    ``len(indices)`` entries with strictly increasing timestamps.  Raises
    :class:`VideoError` if nothing at all could be decoded.
    """
    idx = np.asarray(sorted({int(i) for i in indices}), dtype=np.int64)
    if len(idx) == 0:
        raise VideoError("no indices requested")
    info = info or probe_video(path)
    if len(idx) > info.n_frames:
        idx = sample_indices_uniform(info.n_frames, len(idx))

    if prefer_seek is None:
        prefer_seek = len(idx) < 0.45 * info.n_frames
    got: Dict[int, np.ndarray] = {}
    if prefer_seek:
        # seek mode with a few retries (some avi files mis-seek badly)
        for batch in np.array_split(idx, max(1, int(math.ceil(len(idx) / 24)))):
            if len(batch) == 0:
                continue
            try:
                got.update(_decode_seek(path, batch, info.fps, info))
            except Exception:  # noqa: BLE001
                pass
        if len(got) < 0.8 * len(idx):
            got = _decode_sequential(path, idx)
    else:
        got = _decode_sequential(path, idx)

    if len(got) == 0:
        raise VideoError(f"{path}: decoded 0/{len(idx)} frames")
    keys = sorted(got.keys())
    frames: List[np.ndarray] = []
    for i in idx:
        if i in got:
            frames.append(got[i])
        else:
            lo = max([k for k in keys if k <= i], default=keys[0])
            frames.append(got[lo])
    return frames


def decode_all(path, cap: Optional[int] = None) -> List[np.ndarray]:
    """Decode every frame (optionally capped at ``cap``)."""
    frames = []
    with _open(path) as c:
        s = c.streams.video[0]
        for frame in c.decode(s):
            frames.append(_frame_to_rgb(frame))
            if cap is not None and len(frames) >= cap:
                break
    if not frames:
        raise VideoError(f"{path}: no frames")
    return frames


def timestamp_from_index(idx: np.ndarray, fps: float) -> np.ndarray:
    """t = frame_index / fps, float32 seconds, guaranteed strictly increasing."""
    fps = float(fps) if fps and fps > 0 else 25.0
    t = (np.asarray(idx, dtype=np.float64) / fps).astype(np.float32)
    # guard against float32 collapse for very large fps*index
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = np.float32(t[i - 1] + 1e-4)
    return t.astype(np.float32)
