"""Pixel-level datasets for the pixel-to-event network (spec §6, §7).

Two sources, both **raw pixels** (no frozen encoder anywhere):

* ``charades_pixel`` -- real video from the Charades pixel cache produced by
  ``scripts/build_charades_pixel_cache.py``.  Per-frame interval membership and
  segment onsets/offsets come from the official CSV annotations, so the phase /
  boundary / count targets are human labels, not derived from a model.
* ``robot_sessions`` -- the only robot-domain footage that exists: ten 20 s
  recordings of the RIG-Hover camera (240x240 grey, ~10 fps) captured
  2026-11-03, one scripted behaviour per session.  Tiny; used as an honest
  small-scale robot-domain check, never as a headline result.

Augmentations implement spec §6's list (variable frame rate, irregular gaps,
dropped bursts, blur, occlusion, background/camera motion, repeated frames).
Every transformed variant is generated online inside a window, so it inherits the
original video's split automatically -- a transformed sample can never migrate
across train/test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(r"D:\robot_work\stream_memory_research")
CACHE = Path(r"G:\stream_memory_data\pixel_cache")
SESS = Path(r"D:\robot_work\evidence\sessions")

#: Charades classes kept for the gesture head are irrelevant here; the event head
#: uses whichever classes the index was built with.
DEFAULT_RES = 160


# --------------------------------------------------------------------------- #
# index loading
# --------------------------------------------------------------------------- #
def load_index(name: str) -> dict:
    p = ROOT / "data_manifests" / f"{name}_index.json"
    return json.loads(p.read_text(encoding="utf-8"))


class CharadesPixel(Dataset):
    """Windows of cached grey frames + phase/boundary/count targets.

    ``count_upto[t, c]`` = number of class-``c`` segments that have *completed*
    by frame ``t``.  The causal model can only know this from the number of
    completed transitions it has observed, which is exactly the counting task.
    """

    def __init__(self, index: dict, split: str, window: int = 32, stride: int = 16,
                 train: bool = False, seed: int = 0, event_biased: float = 0.75,
                 aug: bool = True, preload: bool = True):
        self.split = split
        self.window = int(window)
        self.train = bool(train)
        self.aug = bool(aug)
        self.seed = int(seed)
        self.event_biased = float(event_biased)
        self.classes: List[str] = list(index["classes"])
        self.C = len(self.classes)
        self.res = int(index.get("resolution", DEFAULT_RES))
        rows = [s for s in index["seqs"] if s["split"] == split]
        # The whole split is 1.7 GB of uint8 for Charades; keeping it in RAM removes
        # a per-item np.load from the hot path (measured: 125 s -> ~50 s per epoch).
        self._cache: Dict[str, dict] = {}
        if preload:
            for s in rows:
                self._cache[s["path"]] = self._load(s)
        self.preload = bool(preload)
        self.items: List[dict] = []
        for s in rows:
            T = int(s["T"])
            starts = list(range(0, max(1, T - self.window + 1), stride))
            if not starts:
                starts = [0]
            for a in starts:
                self.items.append({"seq": s, "start": a})
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    # -- helpers ---------------------------------------------------------- #
    def _load(self, s: dict):
        z = np.load(s["path"])
        return (np.asarray(z["frames"]), np.asarray(z["t"], np.float32),
                np.asarray(z["event"], np.float32), np.asarray(z["onset"], np.float32))

    def _counts(self, t: np.ndarray, segs: Sequence) -> np.ndarray:
        cidx = {c: i for i, c in enumerate(self.classes)}
        out = np.zeros((len(t), self.C), np.float32)
        for seg in segs:
            cid, s0, e0 = seg[0], float(seg[1]), float(seg[2])
            j = cidx.get(cid)
            if j is None:
                continue
            out[:, j] += (t >= e0).astype(np.float32)
        return out

    def _mid_idx(self, s: dict, T: int) -> np.ndarray:
        """Frame indices of annotated segment midpoints (event-biased sampling)."""
        segs = s.get("segs") or []
        if not segs:
            return np.zeros(0, np.int64)
        L = max(1e-3, float(s.get("length") or 1.0))
        out = []
        for seg in segs:
            mid_t = 0.5 * (float(seg[1]) + float(seg[2]))
            out.append(int(round(mid_t / L * (T - 1))))
        return np.clip(np.asarray(out, np.int64), 0, T - 1)

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    # -- augmentation (spec §6) ------------------------------------------- #
    def _augment(self, frames: np.ndarray, t: np.ndarray, rng: np.random.Generator):
        """Return (frames, t) after a random stream course.  Labels are re-derived."""
        T = len(t)
        # 1. dropped bursts / variable frame rate: keep an irregular subset
        mode = rng.random()
        keep = np.ones(T, bool)
        if mode < 0.25:                       # burst drop
            L = rng.integers(2, max(3, T // 3))
            s = rng.integers(0, max(1, T - L))
            keep[s:s + L] = False
        elif mode < 0.5:                      # random frame drops (irregular dt)
            keep = rng.random(T) > 0.25
        elif mode < 0.6:                      # repeated frame (stalled stream)
            i = int(rng.integers(1, max(2, T - 1)))
            frames[i] = frames[i - 1]
        keep[0] = True
        if keep.sum() >= 4:
            frames, t = frames[keep], t[keep]
        # 2. photometric / geometric nuisance
        f = frames.astype(np.float32)
        if rng.random() < 0.5:                # global gain/offset (light change)
            f = f * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
        if rng.random() < 0.4:                # camera/head motion: translate
            dy, dx = rng.integers(-10, 11, 2)
            f = np.roll(np.roll(f, int(dy), 1), int(dx), 2)
        if rng.random() < 0.3:                # partial occlusion patch
            H, W = f.shape[1], f.shape[2]
            h, w = int(H * rng.uniform(0.15, 0.4)), int(W * rng.uniform(0.15, 0.4))
            y, x = rng.integers(0, H - h), rng.integers(0, W - w)
            f[:, y:y + h, x:x + w] = f.mean()
        if rng.random() < 0.2:                # motion blur over time
            f[1:] = 0.5 * f[1:] + 0.5 * f[:-1]
        return np.clip(f, 0, 255).astype(np.uint8), t

    # -- item ------------------------------------------------------------- #
    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        s = it["seq"]
        frames, t, event, onset = self._cache.get(s["path"]) or self._load(s)
        a = it["start"]
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            a = int(rng.integers(0, max(1, len(t) - self.window + 1)))
            # Event-biased sampling.  Only 4.9% of (frame, class) cells are positive,
            # so a uniform random window wastes most of the budget on background.
            # With probability `event_biased` put a random annotated segment's
            # midpoint inside the window instead.  This is a *sampling* change on
            # the training split only -- the val/test windows stay untouched, so no
            # label information reaches evaluation.
            mid = self._mid_idx(s, len(t))
            if mid.size and rng.random() < self.event_biased:
                m = int(mid[rng.integers(0, len(mid))])
                lo = max(0, m - self.window + 1)
                hi = min(m, len(t) - self.window) if len(t) > self.window else 0
                a = int(rng.integers(lo, hi + 1)) if hi >= lo else max(0, min(lo, len(t) - 1))
        b = min(len(t), a + self.window)
        frames = frames[a:b]
        t = t[a:b]
        if self.train and self.aug:
            rng = np.random.default_rng((self.seed, self.epoch, i, 7))
            frames, t = self._augment(frames, t, rng)
        # labels are recomputed on the (possibly reduced) timestamp grid
        ev = np.zeros((len(t), self.C), np.float32)
        on = np.zeros((len(t), 1), np.float32)
        cnt = self._counts(t, s.get("segs") or [])
        cidx = {c: i for i, c in enumerate(self.classes)}
        for seg in (s.get("segs") or []):
            j = cidx.get(seg[0])
            if j is None:
                continue
            s0, e0 = float(seg[1]), float(seg[2])
            ev[:, j] = ((t >= s0) & (t < e0)).astype(np.float32)
            if len(t) > 1:
                prev = np.concatenate([[t[0] - (t[1] - t[0])], t[:-1]])
                on[:, 0] = np.maximum(on[:, 0], ((prev < s0) & (t >= s0)).astype(np.float32))
        dt = np.zeros(len(t), np.float32)
        dt[1:] = np.clip(np.diff(t), 0.0, 5.0)
        img = torch.from_numpy(frames.astype(np.float32) / 255.0).unsqueeze(1)
        return {"img": img, "dt": torch.from_numpy(dt),
                "mask": torch.ones(len(t), dtype=torch.bool),
                "lab_event": torch.from_numpy(ev),
                "lab_onset": torch.from_numpy(on),
                "lab_count": torch.from_numpy(cnt),
                "lab_presence": torch.full((len(t),), -1, dtype=torch.long),
                "lab_gesture": torch.full((len(t),), -1, dtype=torch.long),
                "seq_id": s["seq_id"]}


# --------------------------------------------------------------------------- #
# robot-domain sessions
# --------------------------------------------------------------------------- #
ROBOT_TASKS = ("none", "fist", "open_palm", "thumbs_up", "wave", "pointing",
               "knock", "empty", "face_close", "lean_out")


def robot_manifest() -> List[dict]:
    p = SESS / "manifest.jsonl"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def load_robot_frames(rec: dict, res: int = 160) -> np.ndarray:
    """Decode one 240x240 session to ``uint8 [T,res,res]`` (cached on disk)."""
    import cv2
    d = SESS / rec["dir"] / "frames"
    cache = CACHE / "robot" / f"{rec['dir']}_{res}.npy"
    if cache.exists():
        return np.load(cache)
    files = sorted(d.glob("*.jpg"))
    arr = np.zeros((len(files), res, res), np.uint8)
    for i, f in enumerate(files):
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        arr[i] = cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


class RobotSessions(Dataset):
    """Causal gesture/presence windows from the ten robot recordings (spec §7).

    Labels available: the scripted behaviour of the session (10 classes) and
    person presence (``empty`` -> absent, everything else -> present).  There are
    **no per-event boundary or count labels** in this footage, so the event and
    count losses are disabled here -- pretending otherwise would be fabrication.
    """

    def __init__(self, held_out: Sequence[str], window: int = 32, stride: int = 8,
                 train: bool = False, seed: int = 0, res: int = 160):
        self.window = int(window)
        self.train = bool(train)
        self.seed = int(seed)
        self.res = int(res)
        recs = robot_manifest()
        self.records = [r for r in recs if r["dir"] not in set(held_out)]
        self.items = []
        for r in self.records:
            n = int(r["frames"])
            for a in range(0, max(1, n - self.window + 1), stride):
                self.items.append((r, a))
        self.fps = 10.0        # measured: ~201 frames / 20 s
        self.epoch = 0

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        r, a = self.items[i]
        frames = load_robot_frames(r, self.res)
        n = len(frames)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            a = int(rng.integers(0, max(1, n - self.window + 1)))
        b = min(n, a + self.window)
        fr = frames[a:b].astype(np.float32)
        T = len(fr)
        dt = np.full(T, 1.0 / self.fps, np.float32)
        dt[0] = 0.0
        gi = ROBOT_TASKS.index(r["task"]) if r["task"] in ROBOT_TASKS else 0
        present = 0 if r["task"] == "empty" else 2
        img = torch.from_numpy(fr / 255.0).unsqueeze(1)
        return {"img": img, "dt": torch.from_numpy(dt),
                "mask": torch.ones(T, dtype=torch.bool),
                "lab_event": torch.zeros(T, 0),
                "lab_onset": torch.zeros(T, 1),
                "lab_count": torch.zeros(T, 0),
                "lab_presence": torch.full((T,), present, dtype=torch.long),
                "lab_gesture": torch.full((T,), gi, dtype=torch.long),
                "seq_id": f"{r['dir']}_a{a}"}


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #
def collate(batch: List[dict], n_event_classes: int = 25) -> Dict[str, torch.Tensor]:
    T = max(b["img"].shape[0] for b in batch)
    B = len(batch)
    res = batch[0]["img"].shape[-1]
    img = torch.zeros(B, T, 1, res, res)
    dt = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    ev = torch.zeros(B, T, n_event_classes)
    on = torch.zeros(B, T, 1)
    cnt = torch.zeros(B, T, n_event_classes)
    pres = torch.full((B, T), -1, dtype=torch.long)
    gest = torch.full((B, T), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["img"].shape[0]
        img[i, :t] = b["img"]
        dt[i, :t] = b["dt"]
        mask[i, :t] = b["mask"]
        if b["lab_event"].shape[1] == n_event_classes:
            ev[i, :t] = b["lab_event"]
            cnt[i, :t] = b["lab_count"]
        on[i, :t] = b["lab_onset"]
        pres[i, :t] = b["lab_presence"]
        gest[i, :t] = b["lab_gesture"]
    return {"img": img, "dt": dt, "mask": mask, "lab_event": ev, "lab_onset": on,
            "lab_count": cnt, "lab_presence": pres, "lab_gesture": gest}


def make_loaders(index_name: str = "charades_pixel", window: int = 32,
                 batch: int = 8, num_workers: int = 0, seed: int = 0,
                 event_biased: float = 0.0, aug: bool = True):
    # `event_biased` defaults to 0 (plain uniform windows).  Measured on this
    # cache: biasing 75% of windows onto an annotated segment only lifts the
    # positive-cell rate from 5.39% to 6.48% (77% of uniform windows already
    # contain an event, since a 16-frame window spans ~5 s of a 30 s clip), so the
    # knob is not worth the extra variance and stays off for the reported runs.
    idx = load_index(index_name)
    n_event_classes = len(idx["classes"])
    tr = CharadesPixel(idx, "train", window, stride=window, train=True,
                       seed=seed, event_biased=event_biased, aug=aug)
    # Validation and test must be evaluated the same way the reported metrics are:
    # one full-length causal pass per video.  With a shorter window the event state
    # integrates over far fewer steps, the output distribution shifts, and a
    # threshold calibrated on windows does not transfer (measured: val threshold
    # 0.26 at 16 frames vs 0.10 at 96 frames, test macro-F1 0.297 vs 0.085).
    full = int(idx.get("n_frames", window))
    va = CharadesPixel(idx, "val", full, stride=max(1, full), train=False)
    te = CharadesPixel(idx, "test", full, stride=max(1, full), train=False)
    mk = lambda d, sh: torch.utils.data.DataLoader(
        d, batch_size=batch, shuffle=sh, num_workers=num_workers,
        collate_fn=lambda b: collate(b, n_event_classes), drop_last=False)
    return idx, mk(tr, True), mk(va, False), mk(te, False)
