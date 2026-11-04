"""Dataset / DataLoader construction for training and evaluation.

Responsibilities
----------------
* manifest -> ``contract.StreamDataset`` (preloaded by default, num_workers=0,
  see DESIGN §4.2) with training augmentation (``stream_ops.make_course_fn``).
* ``train.T_max`` chunking: contiguous RANDOM-START windows of exactly T_max
  during training; the recurrent state is carried through the whole window and
  reset ONLY at the window boundary (one forward per window, exactly one state
  reset).  Evaluation always uses the full sequence with no resets.
* corrupt-sample containment: a broken npz never kills a run; it is replaced by
  a 2-step all-ignore sample and logged to ``corrupt_samples.log``.
* inference-time robustness views: the SAME checkpoint is evaluated under
  stream_ops perturbations with a seed that depends only on (seq_id, level) so
  repeated evaluation is bit-identical.
* ``downsample_batch`` used by the time-consistency auxiliary objective.

All functions are pure w.r.t. global state; randomness is derived from
(seed, epoch, index) or (seq_id, level) so runs are reproducible.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import stream_ops
from src.common.contract import (ROOT, Manifest, SeqRecord, StreamDataset,
                                 collate_stream, sha1_of_file)

SPLITS: Tuple[str, ...] = ("train", "val", "test")
DENSE_LIKE = ("dense", "none", "", None)

#: robustness suite levels (inference-time only; see src/eval/evaluator.py).
#: "shuffled" is the temporal-order control: observation rows are permuted and
#: the timestamps are kept, so a model that only uses a blurred previous frame
#: cannot pass.  Labels stay in temporal order (they are the target, not input).
ROBUST_LEVELS: Tuple[str, ...] = ("dense", "stride2", "stride4", "drop30", "drop50",
                                  "drop70", "burst", "repeat", "jitter", "mixed",
                                  "shuffled")
_OBS_KEYS = ("feat", "tok", "roi", "roi_valid", "roi_meta")


def shuffle_observations(seq: dict, rng: np.random.Generator) -> dict:
    """Permute observation rows, keep `t` and all labels in temporal order."""
    T = int(seq["t"].shape[0])
    perm = rng.permutation(T)
    out = dict(seq)
    for k in _OBS_KEYS:
        v = seq.get(k)
        if isinstance(v, np.ndarray) and v.shape[0] == T:
            out[k] = v[perm]
    return out


def make_level_fn(level: str, seed: int = 0):
    """Inference-time perturbation for a robustness level.

    Identical to ``stream_ops.make_course_fn`` for every level it defines, plus
    the extra "shuffled" temporal-order control implemented here.
    """
    if level in ("shuffled",):
        def _fn(seq, rng):
            return shuffle_observations(seq, rng)
        return _fn
    return stream_ops.make_course_fn(level, seed)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def stable_seed(*parts) -> int:
    """Deterministic 63-bit seed from any stringifiable parts (stable across runs)."""
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 63)


def resolve_manifest_path(cfg: dict) -> Path:
    p = Path(str(cfg["manifest"]))
    return p if p.is_absolute() else (ROOT / p)


def load_manifest(cfg: dict) -> Manifest:
    path = resolve_manifest_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    man = Manifest.load(path)
    if not man.seqs:
        raise ValueError(f"manifest {path} has zero sequences")
    return man


def manifest_sha(path) -> str:
    return sha1_of_file(path)


def _append_line(path: Optional[Path], line: str) -> None:
    if path is None:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
class TrainAugment:
    """Random T_max window + training-time stream augmentation.

    Order: crop to a random contiguous window of exactly T_max first (so the
    window is a real slice of the physical timeline), then apply the configured
    ``train.aug`` course.  Augmentation may shorten the window (drops); the
    batch collate pads.  No state reset happens anywhere inside a window.
    """

    def __init__(self, level: str = "medium", T_max: int = 0):
        self.level = str(level or "none")
        self.T_max = int(T_max or 0)
        self._course = (None if self.level in ("none", "")
                        else stream_ops.make_course_fn(self.level, 0))

    def __call__(self, seq: dict, rng: np.random.Generator) -> dict:
        T = int(seq["t"].shape[0])
        if self.T_max > 0 and T > self.T_max:
            start = int(rng.integers(0, T - self.T_max + 1))
            seq = stream_ops.subseq(seq, start, self.T_max)
        if self._course is not None:
            seq = self._course(seq, rng)
        return seq

    def __repr__(self):
        return f"TrainAugment(level={self.level!r}, T_max={self.T_max})"


class CourseAugment:
    """Fixed inference-time perturbation (robustness suite); level + seed only."""

    def __init__(self, level: str, seed: int = 0):
        self.level = str(level)
        self.seed = int(seed)
        self._course = make_level_fn(self.level, self.seed)

    def __call__(self, seq: dict, rng: np.random.Generator) -> dict:
        return self._course(seq, rng)


# --------------------------------------------------------------------------- #
# dataset with corrupt-sample containment
# --------------------------------------------------------------------------- #
class SafeStreamDataset(StreamDataset):
    """``StreamDataset`` that never raises on a corrupt/short sample.

    A bad sample becomes a 2-step all-ignore sample (labels absent -> collate
    fills the ignore values), is counted and appended to ``corrupt_log``.
    """

    def __init__(self, records: Sequence[SeqRecord], augment=None, load_tok: bool = False,
                 load_roi: bool = False, seed: int = 0, preload: bool = False,
                 feature_dim: int = 1, corrupt_log=None):
        self.feature_dim = int(feature_dim)
        self.corrupt_log = Path(corrupt_log) if corrupt_log else None
        self.n_corrupt = 0
        self._bad: set = set()
        super().__init__(records, augment=augment, load_tok=load_tok, load_roi=load_roi,
                         seed=seed, preload=False)
        if preload:
            for i in range(len(self.records)):
                try:
                    self._cache[i] = self._load(i)
                except Exception as exc:            # pragma: no cover - defensive
                    self._mark_bad(i, exc)

    # -- helpers --------------------------------------------------------- #
    def _mark_bad(self, i: int, exc: BaseException) -> None:
        self.n_corrupt += 1
        self._bad.add(int(i))
        rec = self.records[i]
        _append_line(self.corrupt_log,
                     f"{rec.seq_id}\t{rec.split}\t{type(exc).__name__}: {exc}")
        try:
            import warnings
            warnings.warn(f"[data] corrupt sequence {rec.seq_id!r} skipped: {exc}")
        except Exception:
            pass

    def _fallback(self, i: int) -> dict:
        rec = self.records[i]
        d = max(1, self.feature_dim)
        out = {
            "feat": np.zeros((2, d), dtype=np.float32),
            "t": np.asarray([0.0, 0.1], dtype=np.float32),
            "seq_id": rec.seq_id, "source_id": rec.source_id, "split": rec.split,
            "labels": {},
        }
        return out

    # -- overrides ------------------------------------------------------- #
    def __getitem__(self, i: int) -> dict:
        try:
            return super().__getitem__(i)
        except Exception as exc:                    # pragma: no cover - defensive
            self._mark_bad(int(i), exc)
            return self._fallback(int(i))


class PerturbDataset(SafeStreamDataset):
    """Inference-time perturbed view.

    The RNG is seeded from ``(seq_id, level, base_seed)`` only, so evaluating the
    same checkpoint twice -- with any batching or worker count -- produces
    bit-identical perturbed sequences.
    """

    def __init__(self, records, level: str, base_seed: int = 0, **kw):
        self.level = str(level)
        self.base_seed = int(base_seed)
        aug = CourseAugment(self.level, stable_seed("robust", self.level, base_seed))
        super().__init__(records, augment=aug, seed=stable_seed("perturb", self.level,
                                                              base_seed), **kw)

    def raw(self, i: int) -> dict:
        seq = self._cache.get(i)
        if seq is None:
            try:
                seq = self._load(i)
            except Exception as exc:                # pragma: no cover - defensive
                self._mark_bad(int(i), exc)
                return {"feat": np.zeros((2, max(1, self.feature_dim)), dtype=np.float32),
                        "t": np.asarray([0.0, 0.1], dtype=np.float32)}
        base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in seq.items()}
        rng = np.random.default_rng(
            stable_seed("perturb", self.level, self.base_seed, self.records[i].seq_id))
        return self.augment(base, rng)


# --------------------------------------------------------------------------- #
# records selection
# --------------------------------------------------------------------------- #
def subsample(records: Sequence[SeqRecord], n: int) -> List[SeqRecord]:
    """Deterministic, order-stable subsample of n records (spread over the split)."""
    recs = list(records)
    n = max(1, min(int(n), len(recs)))
    if n >= len(recs):
        return recs
    idx = np.unique(np.round(np.linspace(0, len(recs) - 1, n)).astype(int))
    return [recs[i] for i in idx]


def split_records(man: Manifest, split: str) -> List[SeqRecord]:
    recs = man.split(split)
    return recs


def assert_disjoint(man: Manifest) -> None:
    """No source_id leaks across splits, and no seq_id is duplicated."""
    man.check_split_disjoint()
    seen: Dict[str, str] = {}
    for s in man.seqs:
        if s.seq_id in seen and seen[s.seq_id] != s.split:
            raise ValueError(f"seq_id {s.seq_id!r} appears in two splits")
        seen[s.seq_id] = s.split
    if not man.split("train"):
        raise ValueError("manifest has no train split")


def assert_no_train(records: Sequence[SeqRecord], where: str = "evaluation") -> None:
    bad = [r.seq_id for r in records if r.split == "train"]
    if bad:
        raise ValueError(f"{where} would see {len(bad)} TRAIN sequences "
                         f"(e.g. {bad[:3]}); data leakage is not allowed")


# --------------------------------------------------------------------------- #
# dataset / loader construction
# --------------------------------------------------------------------------- #
def _corrupt_log(cfg: dict) -> Optional[Path]:
    rd = cfg.get("_run_dir")
    if not rd:
        return None
    return Path(rd) / "corrupt_samples.log"


def estimate_preload_gb(man: Manifest, records: Dict[str, Sequence[SeqRecord]]) -> float:
    """Rough RAM needed to cache raw arrays for the given records (float16/32)."""
    bytes_per_elem = 2.0
    total = 0.0
    for split, recs in records.items():
        for r in recs:
            T = float(r.T or 0)
            total += T * float(man.feature_dim) * bytes_per_elem
            if man.tok_dim:
                total += T * 4 * float(man.tok_dim) * bytes_per_elem
            if man.roi_dim:
                total += T * float(man.n_roi) * float(man.roi_dim) * bytes_per_elem
    return total / (1024.0 ** 3)


def build_datasets(cfg: dict, smoke: bool = False) -> dict:
    """Manifest -> datasets for train/val/test (no DataLoaders yet)."""
    man = load_manifest(cfg)
    assert_disjoint(man)
    mpath = resolve_manifest_path(cfg)
    tcfg = dict(cfg.get("train") or {})
    ecfg = dict(cfg.get("eval") or {})
    seed = int(cfg.get("seed", 0))
    level = tcfg.get("aug", "medium")
    T_max = int(tcfg.get("T_max", 0) or 0)
    preload = bool(tcfg.get("preload", True))
    clog = _corrupt_log(cfg)

    recs: Dict[str, List[SeqRecord]] = {s: split_records(man, s) for s in SPLITS}

    # --- smoke: "smallest useful", not "smallest possible" -----------------
    # The Lead pinned min(16, n) sequences per split (2% of a 30-sequence
    # synthetic set was 1 sequence, which makes a smoke run meaningless).
    smoke_info = None
    if smoke:
        n_smoke = int(tcfg.get("smoke_seqs", 16) or 16)
        n_smoke = max(1, n_smoke)
        before = {k: len(v) for k, v in recs.items()}
        for s in SPLITS:
            recs[s] = subsample(recs[s], min(n_smoke, len(recs[s])))
        smoke_info = {"n_seqs": n_smoke, "before": before,
                      "after": {k: len(v) for k, v in recs.items()},
                      "epochs": 2}

    kw = dict(load_tok=man.tok_dim > 0, load_roi=man.roi_dim > 0,
              preload=preload, feature_dim=man.feature_dim, corrupt_log=clog)
    preload_note = None
    max_gb = float(tcfg.get("preload_max_gb", 0) or 0)
    est_gb = estimate_preload_gb(man, recs)
    if preload and max_gb > 0 and est_gb > max_gb:
        preload = False
        kw["preload"] = False
        preload_note = (f"preload disabled: estimated {est_gb:.2f} GB > "
                        f"preload_max_gb={max_gb:.2f} GB")
    ds: Dict[str, StreamDataset] = {
        "train": SafeStreamDataset(recs["train"],
                                   augment=TrainAugment(level, T_max),
                                   seed=stable_seed(seed, "train"), **kw),
        "val": SafeStreamDataset(recs["val"], augment=None,
                                 seed=stable_seed(seed, "val"), **kw),
        "test": SafeStreamDataset(recs["test"], augment=None,
                                  seed=stable_seed(seed, "test"), **kw),
    }
    counts = {s: len(v) for s, v in recs.items()}
    meta = {
        "dataset": man.dataset, "task": man.task or cfg.get("task"),
        "feature_dim": int(man.feature_dim), "tok_dim": int(man.tok_dim),
        "roi_dim": int(man.roi_dim), "n_roi": int(man.n_roi),
        "manifest": str(mpath), "manifest_sha": manifest_sha(mpath),
        "split_key": man.split_key, "counts": counts,
        "label_specs": man.label_specs or {},
        "smoke": smoke_info, "T_max": T_max, "aug": str(level),
        "preload": preload, "preload_est_gb": round(est_gb, 3),
        "preload_note": preload_note,
        "train_windows": ("random-start contiguous window of T_max; recurrent state "
                          "carried across the window, reset only at window boundary"
                          if T_max > 0 else "full sequence; no state reset"),
    }
    return {"manifest": man, "records": recs, "datasets": ds, "meta": meta}


def make_loader(ds: StreamDataset, batch: int, shuffle: bool = False,
                seed: int = 0, epoch: int = 0, num_workers: int = 0) -> DataLoader:
    gen = None
    if shuffle:
        gen = torch.Generator()
        gen.manual_seed(stable_seed(seed, "order", epoch) % (2 ** 63))
    return DataLoader(
        ds, batch_size=max(1, int(batch)), shuffle=bool(shuffle), generator=gen,
        num_workers=max(0, int(num_workers)), collate_fn=collate_stream,
        pin_memory=bool(torch.cuda.is_available()), drop_last=False,
        persistent_workers=False,
    )


def build_loaders(cfg: dict, data: dict, splits: Sequence[str] = SPLITS,
                  epoch: int = 0) -> Dict[str, DataLoader]:
    tcfg = dict(cfg.get("train") or {})
    batch = int(tcfg.get("batch", 64))
    nw = int(tcfg.get("num_workers", 0))
    seed = int(cfg.get("seed", 0))
    out: Dict[str, DataLoader] = {}
    for s in splits:
        out[s] = make_loader(data["datasets"][s], batch, shuffle=(s == "train"),
                             seed=seed, epoch=epoch, num_workers=nw)
    return out


def make_eval_loader(cfg: dict, records: Sequence[SeqRecord], level: str = "dense",
                     batch: Optional[int] = None, feature_dim: Optional[int] = None,
                     load_tok: Optional[bool] = None, load_roi: Optional[bool] = None,
                     corrupt_log=None) -> DataLoader:
    """Loader for one evaluation view (dense or a robustness level)."""
    assert_no_train(records, where=f"eval view {level!r}")
    ecfg = dict(cfg.get("eval") or {})
    dim = int(feature_dim if feature_dim is not None else
              (cfg.get("_feature_dim") or 1))
    bs = int(batch or ecfg.get("batch", 32) or 32)
    kw = dict(load_tok=bool(load_tok), load_roi=bool(load_roi),
              feature_dim=dim, corrupt_log=corrupt_log)
    if level in DENSE_LIKE:
        ds = SafeStreamDataset(records, augment=None, seed=stable_seed("eval", level),
                               preload=bool(ecfg.get("preload", True)), **kw)
    else:
        ds = PerturbDataset(records, level,
                            base_seed=int(cfg.get("seed", 0)),
                            preload=bool(ecfg.get("preload", True)), **kw)
    return make_loader(ds, bs, shuffle=False, seed=int(cfg.get("seed", 0)),
                       num_workers=0)


# --------------------------------------------------------------------------- #
# batch utilities
# --------------------------------------------------------------------------- #
def downsample_batch(batch: dict, factor: int = 2) -> dict:
    """Every `factor`-th step; physical timestamps are preserved (not renumbered).

    Used by the time-consistency objective: dense step ``f*k`` and sparse step
    ``k`` represent the SAME physical time, which is what makes the comparison
    meaningful.
    """
    factor = max(1, int(factor))
    idx = torch.arange(0, batch["x"].shape[1], factor, device=batch["x"].device)
    out = dict(batch)
    for k in ("x", "t", "mask", "tok", "roi", "roi_valid", "roi_meta"):
        if batch.get(k) is not None:
            out[k] = batch[k][:, idx]
    out["_src_idx"] = idx
    t = out["t"]
    dt = torch.zeros_like(t)
    if t.shape[1] > 1:
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
    out["dt"] = torch.where(out["mask"], dt, torch.zeros_like(dt))
    return out


def unpack_meta(batch: dict) -> List[dict]:
    return list(batch.get("meta") or [])


def make_dummy_batch(B: int = 1, T: int = 128, feature_dim: int = 1,
                     tok_dim: int = 0, n_tok: int = 0, roi_dim: int = 0,
                     n_roi: int = 0, period: float = 1.0 / 30.0,
                     device="cpu", dtype=torch.float32) -> dict:
    """A synthetic contract-§1.4 batch (for FLOPs / latency measurement)."""
    t = torch.arange(T, dtype=dtype, device=device).unsqueeze(0) * float(period)
    t = t.expand(B, T)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    dt = torch.zeros(B, T, dtype=dtype, device=device)
    if T > 1:
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0.0)
    batch = {
        "x": torch.zeros(B, T, int(feature_dim), dtype=dtype, device=device),
        "t": t, "dt": dt, "mask": mask,
        "tok": None, "roi": None, "roi_valid": None, "roi_meta": None,
        "labels": {},
        "meta": [{"seq_id": f"dummy{i}", "source_id": f"dummy{i}", "split": "test"}
                 for i in range(B)],
    }
    if tok_dim and n_tok:
        batch["tok"] = torch.zeros(B, T, int(n_tok), int(tok_dim), dtype=dtype,
                                   device=device)
    if roi_dim and n_roi:
        batch["roi"] = torch.zeros(B, T, int(n_roi), int(roi_dim), dtype=dtype,
                                   device=device)
        batch["roi_valid"] = torch.ones(B, T, int(n_roi), dtype=torch.bool,
                                        device=device)
        batch["roi_meta"] = torch.zeros(B, T, int(n_roi), 4, dtype=dtype, device=device)
    return batch
