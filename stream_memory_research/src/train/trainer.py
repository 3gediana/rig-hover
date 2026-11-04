"""Training loop shared by every model family (DESIGN §3, §4.1).

Features
--------
* fp16 autocast + GradScaler, grad clipping, AdamW, linear warmup + cosine
  (or one-cycle / linear) schedule stepped per optimizer update.
* early stopping / best-checkpoint selection on the val main metric with the
  direction taken from the metric name (acc/f1/r2 higher, mae/mse lower);
  ``assert_best_consistency`` re-checks the selection against the whole curve.
* checkpoints with FULL state (model/aux probes/optimizer/scheduler/scaler/epoch/
  RNG) -> exact resume.
* CUDA OOM never kills the run: catch, empty_cache, halve the batch, then
  shorten T_max, then disable AMP; every backoff is recorded.
* fp16 NaN/Inf guards: a non-finite loss or grad norm is skipped (never applied)
  and repeated failures disable AMP.
* per-epoch metrics, curves.json, train.log (wired by run_one), metrics.json
  exactly in the schema of DESIGN §4.1 (both `backoff` and `backoffs` keys).
"""
from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.common import contract
from src.common.contract import (FAILURES, code_hash, gpu_mem_gb, move_batch,
                                 now_str, write_json)
from src.eval.evaluator import (evaluate, higher_is_better, per_seq_main_values,
                                resolve_robustness)
from src.eval import failure_cases as fc
from src.train import data as tdata
from src.train.factory import build_model, head_specs, tok_fusion_mode
from src.train.objectives import AuxObjectives, aux_get, task_losses

DEFAULT_AUX_WEIGHTS: Dict[str, float] = {
    "retrieval": 0.1, "crosstalk": 0.05, "time_consistency": 0.05,
    "write_budget": 0.01, "long_horizon": 0.1,
}
DEFAULT_AUX_CFG: Dict[str, dict] = {
    "long_horizon": {"horizon": 5, "tau": 1.0, "write_thr": 0.1},
    "time_consistency": {"factor": 2, "kind": "cos"},
    "crosstalk": {"w_shared": 1.0, "w_balance": 0.1},
    "write_budget": {},
    "retrieval": {},
}
ROBUST_ALL = tdata.ROBUST_LEVELS


def json_safe(obj):
    """Recursively replace non-finite floats with None (strict-JSON friendly)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


# --------------------------------------------------------------------------- #
def merge_loss_cfg(cfg: dict) -> dict:
    """Config loss block -> AuxObjectives loss config (defaults + overrides)."""
    loss = dict(cfg.get("loss") or {})
    aux = dict(DEFAULT_AUX_WEIGHTS)
    aux.update({k: float(v) for k, v in (loss.get("aux") or {}).items()})
    aux_cfg = {k: dict(v) for k, v in DEFAULT_AUX_CFG.items()}
    for k, v in (loss.get("aux_cfg") or {}).items():
        aux_cfg.setdefault(k, {}).update(v or {})
    return {"aux": aux, "aux_cfg": aux_cfg, "task": dict(loss.get("task") or {}),
            "aux_num_scale": float(loss.get("aux_num_scale", 1.0) or 1.0)}


def parse_run_id(run_id: str) -> Tuple[str, str, str, int]:
    parts = str(run_id).split("__")
    seed = 0
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
    dataset = parts[0] if parts else "dataset"
    model = parts[1] if len(parts) > 1 else "model"
    variant = "base"
    if len(parts) >= 4:
        variant = parts[2]
    return dataset, model, variant, seed


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg or "outofmemory" in msg
            or "cuda error: out of memory" in msg or "cublas" in msg and "alloc" in msg)


class LRSchedule:
    """Linear warmup -> cosine / linear decay, one call per optimizer step."""

    def __init__(self, base_lr: float, total_steps: int = 1, warmup_steps: int = 0,
                 kind: str = "cosine", min_ratio: float = 0.01):
        self.base_lr = float(base_lr)
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.kind = str(kind or "cosine")
        self.min_ratio = float(min_ratio)
        self.step_count = 0

    def factor(self, step: int) -> float:
        step = int(step)
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return max(1e-8, step / float(self.warmup_steps))
        p = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        p = min(max(p, 0.0), 1.0)
        if self.kind in ("linear", "one_cycle"):
            return self.min_ratio + (1.0 - self.min_ratio) * (1.0 - p)
        return self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    def next_lr(self) -> float:
        return self.base_lr * self.factor(self.step_count + 1)

    def step(self) -> float:
        self.step_count += 1
        return self.next_lr()

    def state_dict(self) -> dict:
        return {"base_lr": self.base_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps, "kind": self.kind,
                "min_ratio": self.min_ratio, "step_count": self.step_count}

    def load_state_dict(self, d: dict) -> None:
        for k, v in (d or {}).items():
            if hasattr(self, k):
                setattr(self, k, v)


# --------------------------------------------------------------------------- #
class Trainer:
    """One run: setup -> epochs -> best-checkpoint full evaluation."""

    def __init__(self, cfg: dict, run_dir, smoke: bool = False,
                 effective_run_id: Optional[str] = None):
        self.cfg = dict(cfg or {})
        self.smoke = bool(smoke)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.effective_run_id = str(effective_run_id or self.cfg.get("run_id"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tcfg = dict(self.cfg.get("train") or {})
        self.ecfg = dict(self.cfg.get("eval") or {})
        self.seed = int(self.cfg.get("seed", 0))
        self.loss_cfg = merge_loss_cfg(self.cfg)
        self.task_weights = self.loss_cfg["task"]

        self.batch = int(self.tcfg.get("batch", 64))
        self.num_workers = int(self.tcfg.get("num_workers", 0))
        self.epochs = 2 if self.smoke else int(self.tcfg.get("epochs", 30))
        self.variant = str(self.cfg.get("variant") or
                           parse_run_id(str(self.cfg.get("run_id", "")))[2])
        self.main_metric = str(self.cfg.get("main_metric") or "")
        self.higher_better = higher_is_better(self.main_metric or "acc")

        self.backoffs: List[dict] = []
        self.nan_skips = 0
        self.max_nan_skips = int(self.tcfg.get("max_nan_skips", 5))
        self.amp_disabled = False
        self.history: Dict[str, Any] = {
            "epoch": [], "train_loss": [], "task_loss": [], "aux_loss": [],
            "lr": [], "grad_norm": [], "val": [], "val_main": [], "t_sec": [],
            "aux": {}, "n_skips": [], "n_steps": [],
        }
        self.best: Dict[str, Any] = {"epoch": 0, "val_main": None, "val": None}
        self.start_epoch = 0
        self.wall0 = time.time()
        self.samples_seen = 0
        self._val_loader_batch = int(self.ecfg.get("batch", 32) or 32)
        self.amp_enabled = False
        self.ms_per_step_eval: Optional[float] = None
        self.forward_ms_per_step: Optional[float] = None
        self.eval_wall_sec: Optional[float] = None

    # -- logging ---------------------------------------------------------- #
    def log(self, msg: str) -> None:
        print(f"[{now_str()}] {msg}", flush=True)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        contract.set_seed(self.seed)
        dev_req = str(self.tcfg.get("device") or "auto").lower()
        if dev_req == "cpu":
            self.device = torch.device("cpu")
        elif dev_req in ("cuda", "gpu"):
            self.device = torch.device("cuda")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.cfg["_run_dir"] = str(self.run_dir)
        self.data = tdata.build_datasets(self.cfg, smoke=self.smoke)
        self.meta = dict(self.data["meta"])
        self.feature_dim = int(self.meta["feature_dim"])
        self.cfg["_feature_dim"] = self.feature_dim
        self.cfg["_tok_dim"] = int(self.meta.get("tok_dim", 0) or 0)
        self.cfg["_roi_dim"] = int(self.meta.get("roi_dim", 0) or 0)
        self.cfg["_n_roi"] = int(self.meta.get("n_roi", 0) or 0)
        self.cfg["_n_tok"] = 0
        self.dsets = self.data["datasets"]
        self.train_ds = self.dsets["train"]

        # train loader (batch may shrink on OOM -> rebuilt per epoch anyway)
        self.loaders = tdata.build_loaders(self.cfg, self.data, splits=("train",),
                                           epoch=0)
        self.eval_loaders = {}
        for s in ("val", "test"):
            if len(self.dsets[s]) == 0:
                self.eval_loaders[s] = None
                continue
            self.eval_loaders[s] = tdata.make_loader(
                self.dsets[s], self._val_loader_batch, shuffle=False, seed=self.seed)

        # model + shared heads (identical construction for every core)
        # Token fusion: resolve the TRUE input width from the actual token tensor
        # (K, Dt) so every core gets the same information and params/FLOPs are
        # measured on the real input width.
        self.tok_fusion = tok_fusion_mode(self.cfg)
        tok_shape = None
        if self.tok_fusion != "none":
            try:
                probe_batch = next(iter(self.loaders["train"]))
            except StopIteration:                          # pragma: no cover
                probe_batch = None
            tok = (probe_batch or {}).get("tok")
            if tok is None:
                raise RuntimeError(
                    f"train.tok_fusion={self.tok_fusion!r} but the first train batch "
                    f"has no 'tok' tensor (manifest tok_dim="
                    f"{self.meta.get('tok_dim')}, load_tok={self.train_ds.load_tok}). "
                    f"Provide a tok-bearing manifest or set tok_fusion='none'.")
            tok_shape = (int(tok.shape[-2]), int(tok.shape[-1])) if tok.dim() == 4 else \
                        (1, int(tok.shape[-1]))
            del probe_batch
            self.cfg["_tok_shape"] = tok_shape
        elif int(self.meta.get("tok_dim", 0) or 0) > 0:
            self.log(f"[warn] manifest provides tok (tok_dim={self.meta.get('tok_dim')}) "
                     f"but train.tok_fusion='none' -> models will NOT see per-object "
                     f"token values (Lead pinned: synth must use 'flatten', real video "
                     f"'mean')")
        self.model = build_model(self.cfg, self.feature_dim, tok_shape).to(self.device)
        self.in_dim = int(self.model.in_dim)
        self.cfg["_in_dim"] = self.in_dim
        self.meta["in_dim"] = self.in_dim
        self.meta["tok_fusion"] = self.tok_fusion
        self.meta["tok_shape"] = list(tok_shape) if tok_shape else None
        if self.tok_fusion != "none":
            self.cfg["_n_tok"] = int(tok_shape[0])
            self.cfg["_tok_dim"] = int(tok_shape[1])
            # eval/latency dummy batches must carry the same token layout
            self.cfg["_tok_shape"] = tok_shape
        self.heads_cfg = head_specs(self.cfg)
        if not self.main_metric:
            kind = str(self.heads_cfg[next(iter(self.heads_cfg))]["kind"])
            self.main_metric = {"ce": "acc", "bce": "f1", "mse": "r2"}[kind]
        self.higher_better = higher_is_better(self.main_metric)

        aux_shapes = self._inspect_aux()
        self.auxobj = AuxObjectives(self.loss_cfg, self.heads_cfg, aux_shapes).to(self.device)
        self.params = [p for p in (list(self.model.parameters())
                                   + list(self.auxobj.parameters())) if p.requires_grad]
        lr = float(self.tcfg.get("lr", 2e-3))
        wd = float(self.tcfg.get("wd", 1e-4))
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=wd)
        self.base_lr = lr
        steps = max(1, len(self.loaders["train"]))
        warm = int(float(self.tcfg.get("warmup_epochs", 2)) * steps)
        self.sched = LRSchedule(lr, total_steps=steps * self.epochs, warmup_steps=warm,
                                kind=str(self.tcfg.get("sched", "cosine")),
                                min_ratio=float(self.tcfg.get("min_lr_ratio", 0.01)))
        self.use_amp = bool(self.tcfg.get("amp", True)) and self.device.type == "cuda"
        self.amp_enabled = self.use_amp
        self.scaler = (torch.amp.GradScaler("cuda", enabled=True)
                       if self.use_amp else None)
        self.log(f"run={self.effective_run_id} model={self.cfg['model']['name']} "
                 f"device={self.device} params={self.model.n_params} "
                 f"aux_params={sum(p.numel() for p in self.auxobj.parameters())} "
                 f"train/val/test={self.meta['counts']} amp={self.use_amp} "
                 f"T_max={self.meta.get('T_max')} aug={self.meta.get('aug')}")
        self.log(f"heads={self.heads_cfg} main_metric={self.main_metric} "
                 f"higher_is_better={self.higher_better}")
        self.log(f"aux_weights={ {k: v for k, v in self.auxobj.effective_weights().items()} }")

    def _inspect_aux(self) -> dict:
        """Discover the core's readout width (for the retrieval probe)."""
        shapes: Dict[str, Any] = {"read_dim": 0, "aux_keys": []}
        try:
            batch = next(iter(self.loaders["train"]))
        except StopIteration:
            return shapes
        batch = move_batch(batch, self.device)
        self.model.eval()
        try:
            with torch.no_grad():
                _logits, aux, _out = self.model(batch)
            shapes["aux_keys"] = sorted([k for k, v in (aux or {}).items()
                                         if isinstance(v, torch.Tensor)])
            read = aux_get(aux, "read")
            if read is None:
                read = aux_get(aux, "hidden")
            if read is not None:
                shapes["read_dim"] = int(read.shape[-1])
        except Exception as exc:                       # pragma: no cover - defensive
            self.log(f"[warn] aux inspection failed: {exc}")
        finally:
            self.model.train()
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return shapes

    # ------------------------------------------------------------------ #
    # resume / checkpoints
    # ------------------------------------------------------------------ #
    def _ckpt_path(self, kind: str) -> Path:
        return self.run_dir / "checkpoints" / f"{kind}.pt"

    def _save_ckpt(self, kind: str, epoch: int) -> None:
        path = self._ckpt_path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "auxobj": self.auxobj.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": (self.scaler.state_dict() if self.scaler is not None else None),
            "rng": {"torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate()},
            "history": self.history,
            "best": self.best,
            "batch": self.batch,
            "T_max": int(getattr(getattr(self.train_ds, "augment", None), "T_max", 0) or 0),
            "backoffs": self.backoffs,
            "nan_skips": self.nan_skips,
            "use_amp": self.use_amp,
            "amp_disabled": self.amp_disabled,
            "main_metric": self.main_metric,
            "higher_better": self.higher_better,
            "run_id": self.effective_run_id,
            "smoke": self.smoke,
            "config": self.cfg,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def try_resume(self) -> bool:
        path = self._ckpt_path("last")
        if not path.exists():
            self.log(f"[resume] no checkpoint at {path}; starting fresh")
            return False
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:                              # older torch
            ck = torch.load(path, map_location="cpu")
        except Exception as exc:
            self.log(f"[resume] cannot load {path}: {exc}; starting fresh")
            return False
        try:
            self.model.load_state_dict(ck["model"])
            if ck.get("auxobj"):
                self.auxobj.load_state_dict(ck["auxobj"])
            self.opt.load_state_dict(ck["opt"])
            self.sched.load_state_dict(ck.get("sched") or {})
            if self.scaler is not None and ck.get("scaler"):
                self.scaler.load_state_dict(ck["scaler"])
            rng = ck.get("rng") or {}
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            self.history = ck.get("history") or self.history
            self.best = ck.get("best") or self.best
            self.batch = int(ck.get("batch", self.batch))
            self.backoffs = list(ck.get("backoffs") or [])
            self.nan_skips = int(ck.get("nan_skips", 0))
            self.amp_disabled = bool(ck.get("amp_disabled", False))
            if self.amp_disabled and self.scaler is not None:
                self.use_amp = False
            aug = getattr(self.train_ds, "augment", None)
            if aug is not None and ck.get("T_max") is not None:
                aug.T_max = int(ck["T_max"])
            self.start_epoch = int(ck.get("epoch", 0)) + 1
            if self.best.get("val_main") is not None:
                self.best["epoch"] = int(self.best.get("epoch", 0))
            done = self.start_epoch
            self.log(f"[resume] resumed after {done} completed epoch(s); continuing at "
                     f"epoch {self.start_epoch + 1}/{self.epochs} "
                     f"(batch={self.batch}, best_epoch={self.best.get('epoch')}, "
                     f"best={self.best.get('val_main')})")
            return True
        except Exception as exc:
            self.log(f"[resume] FAILED to restore state ({exc}); starting fresh")
            return False

    # ------------------------------------------------------------------ #
    # amp / oom
    # ------------------------------------------------------------------ #
    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _disable_amp(self, why: str = "") -> None:
        if self.use_amp:
            self.use_amp = False
            self.amp_disabled = True
            self.backoffs.append({"epoch": self.start_epoch, "action": "disable_amp",
                                  "reason": why or "non-finite loss/grad"})
            self.log(f"[amp] disabled ({why or 'non-finite loss/grad'}); continuing in fp32")

    def _oom_backoff(self, epoch: int) -> bool:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        min_batch = max(1, int(self.tcfg.get("min_batch", 4)))
        entry: Dict[str, Any] = {"epoch": epoch + 1, "batch_from": self.batch}
        if self.batch > min_batch:
            self.batch = max(min_batch, self.batch // 2)
            entry.update({"batch_to": self.batch, "action": "halve_batch"})
            self.backoffs.append(entry)
            self.log(f"[oom] epoch {epoch + 1}: batch -> {self.batch}")
            return True
        aug = getattr(self.train_ds, "augment", None)
        cur_T = int(getattr(aug, "T_max", 0) or 0)
        min_T = max(2, int(self.tcfg.get("min_T", 8)))
        new_T = int(max(min_T, math.ceil(cur_T * 0.75))) if cur_T else int(
            self.tcfg.get("T_max_fallback", 48))
        if aug is not None and (cur_T == 0 or new_T < cur_T):
            if aug is not None:
                aug.T_max = new_T
            entry.update({"batch_to": self.batch, "T_max_from": cur_T,
                          "T_max_to": new_T, "action": "shorten_T"})
            self.backoffs.append(entry)
            self.log(f"[oom] epoch {epoch + 1}: T_max {cur_T} -> {new_T}")
            return True
        if self.use_amp:
            self._disable_amp("oom")
            return True
        return False

    # ------------------------------------------------------------------ #
    # train
    # ------------------------------------------------------------------ #
    def _forward_loss(self, batch: dict):
        logits, aux, out = self.model(batch)
        task_total, task_parts = task_losses(logits, batch["labels"], self.heads_cfg,
                                            batch["mask"], self.task_weights)
        aux_parts = self.auxobj.compute(self.model, batch, logits, aux, out)
        aux_total = self.auxobj.weighted(aux_parts)
        return task_total + aux_total, task_total, aux_total, task_parts, aux_parts

    def _make_train_loader(self, epoch: int):
        return tdata.make_loader(self.train_ds, self.batch, shuffle=True,
                                 seed=self.seed, epoch=epoch,
                                 num_workers=self.num_workers)

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        attempts = 0
        while True:
            loader = self._make_train_loader(epoch)
            self.sched.total_steps = (self.sched.step_count
                                      + max(1, len(loader)) * max(1, self.epochs - epoch))
            try:
                return self._run_epoch(epoch, loader)
            except Exception as exc:
                if _is_oom(exc):
                    self.log(f"[oom] epoch {epoch + 1}: {type(exc).__name__}: {exc}")
                    if not self._oom_backoff(epoch):
                        raise RuntimeError("CUDA OOM exhausted every backoff") from exc
                    attempts += 1
                    if attempts > 12:
                        raise
                    continue
                raise

    def _run_epoch(self, epoch: int, loader) -> Dict[str, Any]:
        self.model.train()
        self.auxobj.train()
        sums = {"loss": 0.0, "task": 0.0, "aux": 0.0, "gn": 0.0}
        aux_sums: Dict[str, float] = {}
        aux_stats_sum: Dict[str, float] = {}
        n_total = 0
        n_steps = 0
        clip = float(self.tcfg.get("grad_clip", 1.0))
        log_every = int(self.tcfg.get("log_every", 50))
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = move_batch(batch, self.device)
            B = int(batch["mask"].shape[0])
            self.opt.zero_grad(set_to_none=True)
            lr = self.sched.next_lr()
            for g in self.opt.param_groups:
                g["lr"] = lr
            with self._autocast():
                total, task_total, aux_total, task_parts, aux_parts = self._forward_loss(batch)
                total_s = total if torch.is_tensor(total) else torch.tensor(float(total))
            if (not torch.isfinite(total_s)) or (not total_s.requires_grad):
                self.nan_skips += 1
                if self.nan_skips >= self.max_nan_skips:
                    self._disable_amp("non-finite loss")
                continue
            if self.use_amp:
                self.scaler.scale(total_s).backward()
                self.scaler.unscale_(self.opt)
            else:
                total_s.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.params, clip if clip > 0 else None)
            if not torch.isfinite(gn):
                self.nan_skips += 1
                self.opt.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.update()
                if self.nan_skips >= self.max_nan_skips:
                    self._disable_amp("non-finite grad norm")
                continue
            if self.use_amp:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.sched.step()
            self.samples_seen += B
            n_total += B
            n_steps += 1
            w = float(B)
            sums["loss"] += float(total_s.detach()) * w
            sums["task"] += float(task_total.detach()) * w
            sums["aux"] += float(aux_total.detach()) * w
            sums["gn"] += float(gn.detach()) * w
            for name, val in aux_parts.items():
                try:
                    aux_sums[name] = aux_sums.get(name, 0.0) + float(val.detach()) * w
                except Exception:
                    pass
            for name, st in getattr(self.auxobj, "last_stats", {}).items():
                for k, v in st.items():
                    aux_stats_sum[f"{name}_{k}"] = aux_stats_sum.get(f"{name}_{k}", 0.0) + float(v) * w
            if log_every and (it + 1) % log_every == 0:
                self.log(f"  epoch {epoch + 1} step {it + 1}/{len(loader)} "
                         f"loss={float(total_s):.4f} task={float(task_total):.4f} "
                         f"aux={float(aux_total):.4f} lr={lr:.2e} gn={float(gn):.3f}")
        # last batch may be an OOM-skip; keep the model/opt in a valid state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n = max(1, n_total)
        out = {
            "loss": sums["loss"] / n, "task": sums["task"] / n,
            "aux_total": sums["aux"] / n,
            "gn": sums["gn"] / n, "lr": float(self.sched.next_lr()),
            "n_samples": n_total, "n_steps": n_steps, "t_sec": time.time() - t0,
            "aux": {k: v / n for k, v in aux_sums.items()},
            "aux_stats": {k: v / n for k, v in aux_stats_sum.items()},
            "skips": self.nan_skips,
        }
        return out

    # ------------------------------------------------------------------ #
    # validation / early stopping
    # ------------------------------------------------------------------ #
    def validate(self, epoch: int) -> Optional[dict]:
        val_every = max(1, int(self.tcfg.get("val_every", 1)))
        if (epoch + 1) % val_every != 0:
            return None
        loader = self.eval_loaders.get("val")
        if loader is None:
            return None
        res = evaluate(self.model, {"val": loader}, self.cfg.get("task"), self.cfg,
                       splits=("val",), light=True)
        v = res.get("metrics", {}).get(self.main_metric)
        return {"val_main": (float(v) if v is not None and np.isfinite(v) else None),
                "metrics": res.get("metrics") or {}, "unified": res}

    def _is_better(self, cur: Optional[float], best: Optional[float]) -> bool:
        if cur is None or not np.isfinite(cur):
            return False
        if best is None or not np.isfinite(best):
            return True
        return bool(cur > best + 1e-9) if self.higher_better else bool(cur < best - 1e-9)

    def _record(self, epoch: int, tr: dict, val: Optional[dict]) -> None:
        self.history["epoch"].append(epoch + 1)
        self.history["train_loss"].append(float(tr["loss"]))
        self.history["task_loss"].append(float(tr["task"]))
        self.history["aux_loss"].append(float(tr["aux_total"]))
        self.history["lr"].append(float(tr["lr"]))
        self.history["grad_norm"].append(float(tr["gn"]))
        self.history["t_sec"].append(float(tr["t_sec"]))
        self.history["n_steps"].append(int(tr.get("n_steps", 0)))
        self.history["n_skips"].append(int(tr["skips"]))
        self.history["val"].append(None if val is None else val["metrics"])
        self.history["val_main"].append(None if val is None else val["val_main"])
        for k, v in tr.get("aux", {}).items():
            self.history["aux"].setdefault(k, []).append(float(v))

    def _should_stop(self, epoch: int) -> Optional[str]:
        patience = int(self.tcfg.get("patience", 6))
        best_epoch = int(self.best.get("epoch") or 0)
        if best_epoch > 0 and (epoch + 1) - best_epoch >= patience:
            return "early_stop"
        return None

    # ------------------------------------------------------------------ #
    # artifacts
    # ------------------------------------------------------------------ #
    def curves(self) -> dict:
        return {
            "epoch": list(self.history["epoch"]),
            "train_loss": list(self.history["train_loss"]),
            "task_loss": list(self.history["task_loss"]),
            "aux_loss": list(self.history["aux_loss"]),
            "lr": list(self.history["lr"]),
            "grad_norm": list(self.history["grad_norm"]),
            "val_main": list(self.history["val_main"]),
            "val": list(self.history["val"]),
            "t_sec": list(self.history["t_sec"]),
            "n_skips": list(self.history["n_skips"]),
            "aux": {k: list(v) for k, v in self.history["aux"].items()},
            "main_metric": self.main_metric,
            "higher_is_better": self.higher_better,
        }

    def write_curves(self) -> None:
        write_json(self.run_dir / "curves.json", json_safe(self.curves()))

    def _null_best(self) -> dict:
        robust = {lv: None for lv in ROBUST_ALL}
        return {
            "epoch": None, "val": None, "test": None, "prefix": None,
            "per_seq_test": None, "per_seq_val": None, "per_gap": None,
            "per_dt": None, "rollout": None, "crosstalk_matrix": None,
            "robust": robust, "robust_drop": None, "aux_stats": None,
            "latency_ms_per_step": None, "flops_per_step": None, "params": None,
            "failure_cases": None, "event": None, "consistency": None,
        }

    def _notes(self) -> dict:
        steps = int(sum(self.history.get("n_steps", []) or []))
        train_sec = float(sum(self.history.get("t_sec", []) or []))
        return {
            "train_windows": (self.meta.get("train_windows")
                              if hasattr(self, "meta") else None),
            "tok_fusion": getattr(self, "tok_fusion", None),
            "feature_dim": getattr(self, "feature_dim", None),
            "in_dim": getattr(self, "in_dim", None),
            "tok_shape": (self.meta.get("tok_shape") if hasattr(self, "meta") else None),
            # fwd+bwd+aux per TRAIN step (includes the dataloader gap: honest)
            "ms_per_step_train": round(1000.0 * train_sec / steps, 4) if steps else None,
            "ms_per_step_eval": (round(self.ms_per_step_eval, 4)
                                 if self.ms_per_step_eval is not None else None),
            "eval_wall_sec": (round(self.eval_wall_sec, 2)
                              if self.eval_wall_sec is not None else None),
            # pure forward (fp32, T=128, batch 1) from the perf measurement
            "forward_ms_per_step": (round(self.forward_ms_per_step, 4)
                                    if self.forward_ms_per_step is not None else None),
            "n_train_steps": steps,
        }

    def metrics_block(self, status: str, best_block: Optional[dict] = None) -> dict:
        wall = time.time() - self.wall0
        notes = self._notes()
        return json_safe({
            "run_id": self.effective_run_id,
            "status": status,
            "dataset": self.cfg.get("dataset"),
            "task": self.cfg.get("task"),
            "model": (self.cfg.get("model") or {}).get("name"),
            "variant": self.variant,
            "seed": self.seed,
            "params": int(self.model.n_params) if hasattr(self, "model") else None,
            "code_hash": code_hash(),
            "manifest_sha": self.meta.get("manifest_sha") if hasattr(self, "meta") else None,
            "epochs_run": len(self.history["epoch"]),
            "wall_sec": round(wall, 2),
            "peak_vram_gb": round(float(gpu_mem_gb()), 3),
            "samples_per_sec": (round(self.samples_seen / wall, 2) if wall > 0 else None),
            "backoff": list(self.backoffs),
            "backoffs": list(self.backoffs),
            "main_metric": self.main_metric,
            "main_metric_higher_is_better": bool(self.higher_better),
            "tok_fusion": getattr(self, "tok_fusion", None),
            "in_dim": getattr(self, "in_dim", None),
            "feature_dim": getattr(self, "feature_dim", None),
            "ms_per_step_train": notes.get("ms_per_step_train"),
            "ms_per_step_eval": notes.get("ms_per_step_eval"),
            "notes": notes,
            "curves": self.curves(),
            "best": best_block or self._null_best(),
            "aux_weights": self.auxobj.effective_weights() if hasattr(self, "auxobj") else None,
            "aux_cfg": (self.loss_cfg.get("aux_cfg") if hasattr(self, "loss_cfg") else None),
            "core_cfg_effective": getattr(getattr(self, "model", None), "core", None)
            and getattr(self.model.core, "cfg_effective", None),
            "core_cfg_unused": getattr(getattr(self, "model", None), "core", None)
            and getattr(self.model.core, "cfg_unused", None),
            "param_breakdown": (self.model.param_breakdown()
                                if hasattr(self, "model") else None),
            "counts": self.meta.get("counts") if hasattr(self, "meta") else None,
            "T_max": self.meta.get("T_max") if hasattr(self, "meta") else None,
            "train_windows": self.meta.get("train_windows") if hasattr(self, "meta") else None,
            "smoke": self.smoke,
            "amp": bool(self.use_amp),
            "config": self.cfg,
            "finished_at": now_str(),
        })

    def write_metrics(self, status: str, best_block: Optional[dict] = None) -> dict:
        m = self.metrics_block(status, best_block)
        write_json(self.run_dir / "metrics.json", m)
        return m

    # ------------------------------------------------------------------ #
    # final evaluation
    # ------------------------------------------------------------------ #
    def finalize(self, stop_reason: str = "max_epochs") -> dict:
        best_ckpt = self._ckpt_path("best")
        if best_ckpt.exists():
            ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"])
            if ck.get("auxobj"):
                self.auxobj.load_state_dict(ck["auxobj"])
        elif self._ckpt_path("last").exists():
            ck = torch.load(self._ckpt_path("last"), map_location="cpu", weights_only=False)
            self.model.load_state_dict(ck["model"])
            if ck.get("auxobj"):
                self.auxobj.load_state_dict(ck["auxobj"])
        self.model.eval()
        self.auxobj.eval()
        task = self.cfg.get("task")
        levels = resolve_robustness(self.cfg)

        val_res, val_seq = (None, {})
        t_eval0 = time.time()
        if self.eval_loaders.get("val") is not None:
            val_res, val_seq = evaluate(self.model, {"val": self.eval_loaders["val"]},
                                        task, self.cfg, splits=("val",), light=True,
                                        return_seq=True)
        test_res, test_seq = (None, {})
        if self.eval_loaders.get("test") is not None:
            test_res, test_seq = evaluate(self.model, {"test": self.eval_loaders["test"]},
                                          task, self.cfg, splits=("test",),
                                          robustness=levels, light=False, return_seq=True)
        self.eval_wall_sec = time.time() - t_eval0
        n_eval_steps = int((val_res or {}).get("n_steps") or 0) + int(
            (test_res or {}).get("n_steps") or 0)
        if n_eval_steps > 0:
            self.ms_per_step_eval = 1000.0 * self.eval_wall_sec / n_eval_steps
        if test_res is not None and test_res.get("latency_ms_per_step") is not None:
            self.forward_ms_per_step = float(test_res["latency_ms_per_step"])

        val_metrics = (val_res or {}).get("metrics") or {}
        test_metrics = (test_res or {}).get("metrics") or {}
        main_m = self.main_metric
        best_epoch = int(self.best.get("epoch") or 0)

        # per-sequence main metric (required by the Lead's summarizer)
        per_seq_test = per_seq_val = None
        if test_res is not None and test_seq.get("test"):
            per_seq_test = {main_m: per_seq_main_values(test_seq["test"],
                                                       test_res.get("main_head"), main_m)}
        if val_res is not None and val_seq.get("val"):
            per_seq_val = {main_m: per_seq_main_values(val_seq["val"],
                                                      val_res.get("main_head"), main_m)}

        robust_block: Dict[str, Optional[dict]] = {lv: None for lv in ROBUST_ALL}
        robust_drop = None
        if test_res is not None:
            rob = test_res.get("robustness") or {}
            for lv, mm in rob.items():
                robust_block[lv] = mm
            if rob:
                robust_drop = test_res.get("robust_drop")
                if robust_block.get("dense") is None:
                    robust_block["dense"] = {k: v for k, v in test_metrics.items()
                                             if isinstance(v, (int, float))}

        # consistency assertion pinned by the Lead: best.val[main_metric] must be
        # the best value seen across the whole curve (1e-6 slack).
        consistency = self.assert_best_consistency(val_metrics.get(main_m))

        # failure cases (worst-N sequences with a one-line diagnosis)
        fc_summary = None
        try:
            fc_summary = fc.dump_failure_cases(
                run_id=self.effective_run_id, per_seq=test_seq.get("test") or [],
                metrics=test_metrics, main_metric=main_m,
                higher_better=self.higher_better, k=int(self.ecfg.get("failure_topk", 10)),
                run_dir=self.run_dir, extra={"split": "test", "head": test_res.get("main_head")
                                             if test_res else None})
        except Exception as exc:                        # pragma: no cover - defensive
            self.log(f"[warn] failure_cases failed: {exc}")

        best_block = self._null_best()
        best_block.update({
            "epoch": best_epoch or None,
            "val": val_metrics or None,
            "test": test_metrics or None,
            "prefix": (test_res or {}).get("per_prefix"),
            "per_seq_test": per_seq_test,
            "per_seq_val": per_seq_val,
            "per_gap": (test_res or {}).get("per_gap"),
            "per_dt": (test_res or {}).get("per_dt"),
            "rollout": (test_res or {}).get("rollout"),
            "crosstalk_matrix": (test_res or {}).get("crosstalk_matrix"),
            "robust": robust_block,
            "robust_drop": robust_drop,
            "aux_stats": (test_res or {}).get("aux_stats"),
            "latency_ms_per_step": (test_res or {}).get("latency_ms_per_step"),
            "flops_per_step": (test_res or {}).get("flops_per_step"),
            "params": int(self.model.n_params),
            "failure_cases": (f"failure_cases/{self.effective_run_id}.json"
                              if fc_summary else None),
            "event": (test_res or {}).get("event"),
            "consistency": consistency,
        })
        status = "complete" if (test_metrics or val_metrics) else "failed"
        if status == "complete" and not test_metrics:
            # a single-split dataset without test still counts as complete on val
            status = "complete"
        m = self.write_metrics(status, best_block)
        self.write_curves()
        write_json(self.run_dir / "per_source.json", {
            "main_metric": main_m,
            "higher_is_better": bool(self.higher_better),
            "val": (val_res or {}).get("per_source") or {},
            "test": (test_res or {}).get("per_source") or {},
        })
        self.log(f"[done] status={status} best_epoch={best_epoch} "
                 f"val_{main_m}={val_metrics.get(main_m)} test_{main_m}={test_metrics.get(main_m)} "
                 f"stop={stop_reason} wall={m['wall_sec']}s")
        if robust_drop:
            self.log("[robust] drop vs dense: "
                     + ", ".join(f"{k}={v:+.4f}" for k, v in robust_drop.items()
                                 if v is not None and k != "dense"))
        return m

    def assert_best_consistency(self, final_val: Optional[float]) -> dict:
        vals = [v for v in self.history.get("val_main", []) if v is not None
                and np.isfinite(v)]
        info = {"checked": False, "best_in_curve": None, "final_val": final_val,
                "delta": None, "ok": True, "n_val_points": len(vals)}
        if not vals or final_val is None or not np.isfinite(final_val):
            return info
        best_curve = max(vals) if self.higher_better else min(vals)
        delta = abs(float(final_val) - float(best_curve))
        info.update({"checked": True, "best_in_curve": float(best_curve),
                     "delta": float(delta), "ok": bool(delta <= 1e-6)})
        if delta > 1e-6:
            self.backoffs.append({"action": "best_consistency_warn", "delta": delta})
            self.log(f"[warn] best.val[{self.main_metric}]={final_val} vs curve best "
                     f"{best_curve} (delta={delta:.2e})")
        if delta > 1e-3:
            raise AssertionError(
                f"best.val[{self.main_metric}]={final_val} is not the best value seen "
                f"in the curve ({best_curve}, delta={delta:.3e}); checkpoint selection "
                f"or the metric direction is inconsistent")
        return info

    # ------------------------------------------------------------------ #
    def run(self, resume: bool = False) -> dict:
        self.setup()
        resumed = self.try_resume() if resume else False
        self.write_metrics("running")
        epoch = self.start_epoch
        stop_reason = "max_epochs"
        while epoch < self.epochs:
            tr = self.train_epoch(epoch)
            val = self.validate(epoch)
            if val is not None and self._is_better(val["val_main"],
                                                   self.best.get("val_main")):
                self.best = {"epoch": epoch + 1, "val_main": val["val_main"],
                             "val": val["metrics"]}
                self._save_ckpt("best", epoch)
                improved = True
            else:
                improved = False
            self._record(epoch, tr, val)
            self._save_ckpt("last", epoch)
            self.write_curves()
            vm = None if val is None else val["val_main"]
            self.log(f"epoch {epoch + 1}/{self.epochs} loss={tr['loss']:.4f} "
                     f"task={tr['task']:.4f} aux={tr['aux_total']:.4f} "
                     f"val_{self.main_metric}={vm} "
                     f"best_epoch={self.best.get('epoch')} lr={tr['lr']:.2e} "
                     f"{'*' if improved else ''} ({tr['t_sec']:.1f}s)")
            self.write_metrics("running")
            reason = self._should_stop(epoch)
            if reason:
                stop_reason = reason
                self.log(f"early stopping at epoch {epoch + 1} "
                         f"(best epoch {self.best.get('epoch')}, patience="
                         f"{self.tcfg.get('patience')})")
                break
            epoch += 1
        if not self.history["epoch"]:
            stop_reason = "no_epochs"
        return self.finalize(stop_reason=stop_reason)
