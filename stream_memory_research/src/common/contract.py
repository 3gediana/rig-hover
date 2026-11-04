"""Shared contract for the stream-memory research project.

Everything here is owned by the Lead.  Teammates import it, never edit it.
See DESIGN.md for the written spec.

Public API
----------
paths                -- canonical directory constants
set_seed(seed)       -- deterministic seeding for torch/numpy/random
read_json/write_json -- utf-8 json io
SeqRecord, Manifest  -- data description objects
StreamDataset        -- torch Dataset over cached feature npz files
collate_stream       -- pad variable-length sequences into a batch
TemporalCore         -- abstract temporal core (all baselines + ours)
MultiTaskHeads       -- identical output heads shared by every core
apply_mask           -- helper: state update that respects the pad mask
masked_seq_mean      -- helper: masked mean over time
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
ROOT = Path(r"D:\robot_work\stream_memory_research")
GDATA = Path(r"G:\stream_memory_data")

SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"
CKPTS = ROOT / "checkpoints"
MANIFESTS = ROOT / "data_manifests"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
FAILURES = ROOT / "failure_cases"
LOGS = ROOT / "logs"
BEST = ROOT / "best_model"

FEATURES = GDATA / "features"
VIDEOS = GDATA / "videos"
SYNTH = GDATA / "synth"

for _p in (CONFIGS, RUNS, CKPTS, MANIFESTS, REPORTS, FIGS, FAILURES, LOGS, BEST,
           FEATURES, VIDEOS, SYNTH):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# misc helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    return str(o)


def sha1_of_file(path, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_hash() -> str:
    """Cheap fingerprint of the whole src tree (order-stable, mtime-free)."""
    h = hashlib.sha1()
    for p in sorted((SRC).rglob("*.py")):
        h.update(p.relative_to(SRC).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------- #
# data description
# --------------------------------------------------------------------------- #
LABEL_SPECS: Dict[str, Dict[str, Any]] = {
    # task -> per-step label spec used by collate + heads
    "action": {"lab_action": {"kind": "ce", "ignore": -1}},
    "event": {"lab_event": {"kind": "bce"}},
    "state": {"lab_state": {"kind": "mse"}},
    "track": {"lab_track": {"kind": "mse"}, "lab_id": {"kind": "ce", "ignore": -1}},
    "synth": {},  # synth tasks declare their own labels in the manifest
}


@dataclass
class SeqRecord:
    seq_id: str
    source_id: str
    split: str
    T: int
    path: str
    labels: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Manifest:
    dataset: str
    task: str
    feature_dim: int
    seqs: List[SeqRecord]
    tok_dim: int = 0
    roi_dim: int = 0
    n_roi: int = 0
    encoder: Dict[str, Any] = field(default_factory=dict)
    split_key: str = "source_id"
    label_specs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- io ----
    def save(self, path=None):
        path = Path(path) if path else MANIFESTS / f"{self.dataset}.json"
        d = {
            "dataset": self.dataset, "task": self.task, "feature_dim": self.feature_dim,
            "tok_dim": self.tok_dim, "roi_dim": self.roi_dim, "n_roi": self.n_roi,
            "encoder": self.encoder, "split_key": self.split_key,
            "label_specs": self.label_specs, "extra": self.extra,
            "seqs": [s.to_dict() for s in self.seqs],
        }
        write_json(path, d)
        return path

    @classmethod
    def load(cls, path) -> "Manifest":
        d = read_json(path)
        seqs = [SeqRecord(**s) for s in d.pop("seqs")]
        return cls(seqs=seqs, **d)

    # ---- views ----
    def split(self, name: str) -> List[SeqRecord]:
        return [s for s in self.seqs if s.split == name]

    def by_split(self) -> Dict[str, List[SeqRecord]]:
        out: Dict[str, List[SeqRecord]] = {}
        for s in self.seqs:
            out.setdefault(s.split, []).append(s)
        return out

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in sorted(self.by_split().items())}

    def check_split_disjoint(self) -> None:
        """Assert that no source_id leaks across splits."""
        owner: Dict[str, str] = {}
        bad = []
        for s in self.seqs:
            if s.source_id in owner and owner[s.source_id] != s.split:
                bad.append((s.source_id, owner[s.source_id], s.split))
            owner[s.source_id] = s.split
        if bad:
            raise ValueError(f"{len(bad)} source_ids leak across splits, e.g. {bad[:5]}")


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
_ARRAY_KEYS = ("feat", "t", "tok", "roi", "roi_valid", "roi_meta")


class StreamDataset(torch.utils.data.Dataset):
    """Reads per-sequence npz and returns a dict of numpy arrays.

    Parameters
    ----------
    records : list[SeqRecord]
    augment : callable(seq_dict, rng) -> seq_dict   applied on the raw arrays
    load_tok/load_roi : bool  -- skip loading big optional arrays when unused
    preload : bool            -- cache all arrays in RAM (small datasets)
    """

    def __init__(self, records: Sequence[SeqRecord], augment: Optional[Callable] = None,
                 load_tok: bool = False, load_roi: bool = False,
                 seed: int = 0, preload: bool = False):
        self.records = list(records)
        self.augment = augment
        self.load_tok = load_tok
        self.load_roi = load_roi
        self.seed = seed
        self.epoch = 0
        self._cache: Dict[int, dict] = {}
        if preload:
            for i in range(len(self.records)):
                self._cache[i] = self._load(i)

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def __len__(self):
        return len(self.records)

    def _load(self, i: int) -> dict:
        rec = self.records[i]
        with np.load(rec.path, allow_pickle=False) as z:
            seq = {"feat": z["feat"], "t": z["t"]}
            for k in _ARRAY_KEYS:
                if k in ("feat", "t"):
                    continue
                if k == "tok" and not self.load_tok:
                    continue
                if k in ("roi", "roi_valid", "roi_meta") and not self.load_roi:
                    continue
                if k in z:
                    seq[k] = z[k]
            for k in z.files:
                if k.startswith("lab_"):
                    seq[k] = z[k]
        return seq

    def raw(self, i: int) -> dict:
        seq = self._cache.get(i)
        if seq is None:
            seq = self._load(i)
        # copy the mutable pieces so augmentation never corrupts the cache
        base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in seq.items()}
        if self.augment is not None:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self.epoch * 7919 + i) % (2 ** 63))
            base = self.augment(base, rng)
        return base

    def __getitem__(self, i: int) -> dict:
        seq = self.raw(i)
        rec = self.records[i]
        out = {
            "feat": torch.from_numpy(np.ascontiguousarray(seq["feat"], dtype=np.float32)),
            "t": torch.from_numpy(np.ascontiguousarray(seq["t"], dtype=np.float32)),
            "seq_id": rec.seq_id,
            "source_id": rec.source_id,
            "split": rec.split,
            "labels": {},
        }
        for k, v in seq.items():
            if k.startswith("lab_"):
                name = k[4:]
                v = np.asarray(v)
                if np.issubdtype(v.dtype, np.integer):
                    out["labels"][name] = torch.from_numpy(v.astype(np.int64))
                else:
                    out["labels"][name] = torch.from_numpy(v.astype(np.float32))
            elif k in ("tok", "roi", "roi_valid", "roi_meta"):
                arr = np.asarray(v)
                out[k] = (torch.from_numpy(arr.astype(np.bool_)) if k == "roi_valid"
                          else torch.from_numpy(arr.astype(np.float32)))
        return out


def collate_stream(batch: List[dict]) -> dict:
    """Pad a list of per-sequence dicts into a batch (see DESIGN §1.4)."""
    B = len(batch)
    T = max(int(b["feat"].shape[0]) for b in batch)
    Dg = int(batch[0]["feat"].shape[1])

    def _pad(shape_tail, dtype, fill=0.0):
        out = torch.full((B, T, *shape_tail), fill, dtype=dtype)
        return out

    x = _pad((Dg,), torch.float32)
    t = _pad((), torch.float32)
    mask = torch.zeros(B, T, dtype=torch.bool)

    tok = tok_valid = roi = roi_valid = roi_meta = None
    if "tok" in batch[0]:
        K, Dt = batch[0]["tok"].shape[1:]
        tok = _pad((K, Dt), torch.float32)
    if "roi" in batch[0]:
        N, Dr = batch[0]["roi"].shape[1:]
        roi = _pad((N, Dr), torch.float32)
        roi_valid = _pad((N,), torch.bool)
        roi_meta = _pad((N, 4), torch.float32)
    if "roi_meta" in batch[0] and roi_meta is None:
        N = batch[0]["roi_meta"].shape[1]
        roi_meta = _pad((N, 4), torch.float32)

    label_keys = set()
    for b in batch:
        label_keys |= set(b["labels"].keys())
    labels: Dict[str, torch.Tensor] = {}
    for k in sorted(label_keys):
        sample = next(b["labels"][k] for b in batch if k in b["labels"])
        tail = tuple(sample.shape[1:])
        is_int = not sample.is_floating_point()
        # default ignore value for classification, 0 for regression
        fill = -1 if is_int else 0.0
        if k in ("event", "state", "track"):
            fill = 0.0
        labels[k] = _pad(tail, torch.int64 if is_int else torch.float32, fill)
        if is_int:
            labels[k] = torch.full((B, T, *tail), -1, dtype=torch.int64)

    for i, b in enumerate(batch):
        n = int(b["feat"].shape[0])
        x[i, :n] = b["feat"]
        t[i, :n] = b["t"]
        mask[i, :n] = True
        if tok is not None and "tok" in b:
            tok[i, :n] = b["tok"]
        if roi is not None and "roi" in b:
            roi[i, :n] = b["roi"]
            roi_valid[i, :n] = b["roi_valid"]
            roi_meta[i, :n] = b["roi_meta"]
        for k in label_keys:
            if k in b["labels"]:
                v = b["labels"][k]
                labels[k][i, :v.shape[0]] = v

    dt = torch.zeros(B, T, dtype=torch.float32)
    dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
    dt = torch.where(mask, dt, torch.zeros_like(dt))

    return {
        "x": x, "t": t, "dt": dt, "mask": mask,
        "tok": tok, "roi": roi, "roi_valid": roi_valid, "roi_meta": roi_meta,
        "labels": labels,
        "meta": [{"seq_id": b["seq_id"], "source_id": b["source_id"], "split": b["split"]}
                 for b in batch],
    }


def move_batch(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# model contract
# --------------------------------------------------------------------------- #
def apply_mask(h_new: torch.Tensor, h_prev: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """mask is bool [B]; padded steps keep the previous state."""
    m = mask_t
    while m.dim() < h_new.dim():
        m = m.unsqueeze(-1)
    return torch.where(m, h_new, h_prev)


def masked_seq_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,T,D], mask [B,T] -> [B,D] mean over valid steps."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)


class TemporalCore(nn.Module):
    """Abstract temporal core.  Subclasses must be strictly causal."""

    name: str = "core"
    out_dim: int = 0

    def forward(self, x, dt, mask, tok=None, tok_valid=None,
                roi=None, roi_valid=None, roi_meta=None):
        """x [B,T,Dg] float, dt [B,T] seconds, mask [B,T] bool.
        Returns (out [B,T,H], aux dict)."""
        raise NotImplementedError

    def init_state(self, B: int, device) -> dict:
        return {}

    @property
    def n_params(self) -> int:
        return count_params(self)


class MultiTaskHeads(nn.Module):
    """Identical output heads for every core (so only the core differs)."""

    def __init__(self, in_dim: int, output_spec: Dict[str, Dict[str, Any]],
                 hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.spec = output_spec
        self.norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.heads = nn.ModuleDict()
        for name, sp in output_spec.items():
            dim = int(sp["dim"])
            n_out = int(sp.get("n_classes", 1)) if sp["kind"] in ("ce", "bce") else dim
            self.heads[name] = nn.Linear(hidden, n_out)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(self.norm(h))
        return {k: self.heads[k](z) for k in self.spec}

    @property
    def n_params(self) -> int:
        return count_params(self)


def parse_hidden(spec: str, in_dim: int) -> int:
    """'256' -> 256 ; '0.5x' -> round(0.5*in_dim)"""
    spec = str(spec)
    m = re.fullmatch(r"([\d.]+)x", spec)
    if m:
        return max(4, int(round(float(m.group(1)) * in_dim)))
    return int(spec)


# --------------------------------------------------------------------------- #
# run bookkeeping
# --------------------------------------------------------------------------- #
def make_run_id(dataset: str, model: str, variant: str = "base", seed: int = 0) -> str:
    v = f"__{variant}" if variant and variant != "base" else ""
    return f"{dataset}__{model}{v}__s{seed}"


def run_dir(run_id: str, create: bool = True) -> Path:
    d = RUNS / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


class Tee:
    """Duplicate stdout/stderr into a log file."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.f.write(s)
        self.stdout.write(s)

    def flush(self):
        self.f.flush()
        self.stdout.flush()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t0
        return False

    @property
    def elapsed(self):
        return time.time() - self.t0


def human_time(sec: float) -> str:
    sec = float(sec)
    if sec < 90:
        return f"{sec:.1f}s"
    if sec < 5400:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.2f}h"


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3
