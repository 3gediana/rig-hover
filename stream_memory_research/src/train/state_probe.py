"""Reusable probes for the 10 scientific questions (DESIGN §2.1/§2.3).

All probes take a built model + DataLoader(s) and return JSON-able dicts.  They
never train anything long and never write files (the Lead / run_suite decides
what to persist).

Implemented
-----------
(a) ``probe_dt_importance``      zero / shuffle the dt channel at eval -> metric drop
(b) ``collect_mem_curve``        per-step memory-read norm over normalized time
(c) ``probe_write_gate_hist``    write-gate histogram
(d) ``probe_state_reset``        reset the recurrent state every k steps -> drop
(e) ``probe_linear_readout``     ridge probe of the memory readout for the
                                occluded/hidden target, visible vs hidden steps
(f) ``probe_crosstalk``          N x N slot co-activation (+ K x K key overlap)

Each probe is a no-op-with-explanation when the core does not expose the needed
aux tensors (``{"available": False, "reason": ...}``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.common.contract import move_batch
from src.eval.evaluator import (batch_to_per_seq, evaluate, head_metrics,
                                higher_is_better, pick_main_head, slot_crosstalk)
from src.train import data as tdata
from src.train.factory import head_specs
from src.train.objectives import aux_get


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class TransformLoader:
    """Wrap a DataLoader and apply a transform to every batch (keeps .dataset)."""

    def __init__(self, loader, fn):
        self.loader = loader
        self.fn = fn

    def __iter__(self):
        for batch in self.loader:
            yield self.fn(batch)

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def _main_metric(cfg: dict, heads_cfg: Dict[str, Dict[str, Any]]) -> str:
    m = str((cfg or {}).get("main_metric") or "")
    if not m:
        kind = str(heads_cfg[next(iter(heads_cfg))].get("kind", "ce"))
        m = {"ce": "acc", "bce": "f1", "mse": "r2"}[kind]
    return m


def _eval_loader_metric(model, loader, cfg: dict, split: Optional[str] = None) -> Dict[str, Any]:
    split = split or "val"
    res = evaluate(model, {split: loader}, (cfg or {}).get("task"), cfg,
                   splits=(split,), light=True)
    return {"metrics": res.get("metrics") or {}, "main_value":
            res.get("metrics", {}).get(res.get("main_metric"))}


def _iter_batches(model, loader, device):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            yield move_batch(batch, device)


# --------------------------------------------------------------------------- #
# (a) dt importance
# --------------------------------------------------------------------------- #
def _transform_dt(batch: dict, mode: str, seed: int = 0) -> dict:
    out = dict(batch)
    if mode == "normal":
        return out
    if mode == "zero":
        out["dt"] = torch.zeros_like(batch["dt"])
        return out
    dt = batch["dt"].detach().cpu().numpy().copy()
    mask = batch["mask"].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    for i in range(dt.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size > 1:
            vals = dt[i, idx].copy()
            rng.shuffle(vals)
            dt[i, idx] = vals
            dt[i, idx[0]] = 0.0
    out["dt"] = torch.from_numpy(dt).to(batch["dt"].device)
    return out


def probe_dt_importance(model, loader, cfg: dict, split: Optional[str] = None,
                        seed: int = 0) -> Dict[str, Any]:
    """(a) How much does the model use physical dt?  zero/shuffle -> drop."""
    split = split or _split_name(loader)
    base = _eval_loader_metric(model, TransformLoader(loader, lambda b: _transform_dt(b, "normal")),
                               cfg, split)
    zero = _eval_loader_metric(model, TransformLoader(loader, lambda b: _transform_dt(b, "zero")),
                               cfg, split)
    shuf = _eval_loader_metric(model, TransformLoader(
        loader, lambda b: _transform_dt(b, "shuffle", seed)), cfg, split)
    m = _main_metric(cfg, head_specs(cfg))
    main = base["main_value"]
    hb = higher_is_better(m)
    def drop(v):
        if v is None or main is None or not np.isfinite(v) or not np.isfinite(main):
            return None
        return float(main - v) if hb else float(v - main)
    return {"available": True, "metric": m, "normal": main,
            "dt_zero": zero["main_value"], "dt_shuffled": shuf["main_value"],
            "drop_zero": drop(zero["main_value"]),
            "drop_shuffled": drop(shuf["main_value"])}


def _split_name(loader) -> str:
    ds = getattr(loader, "dataset", None)
    recs = getattr(ds, "records", None) or []
    return recs[0].split if recs else "val"


# --------------------------------------------------------------------------- #
# (b) memory-read norm over time  +  (c) write gate histogram
# --------------------------------------------------------------------------- #
def collect_aux_curves(model, loader, cfg: Optional[dict] = None,
                       n_bins: int = 16, gate_bins: int = 20) -> Dict[str, Any]:
    """(b)+(c) Per-normalized-time memory-read norm and write-gate histogram."""
    device = next(model.parameters()).device
    mem_sum = np.zeros(n_bins)
    mem_cnt = np.zeros(n_bins)
    gate_vals: List[np.ndarray] = []
    for batch in _iter_batches(model, loader, device):
        logits, aux, _out = model(batch)
        read = aux_get(aux, "read")
        gate = aux_get(aux, "write_gate")
        mask = batch["mask"].detach().cpu().numpy()
        if read is not None:
            norm = read.detach().float().norm(dim=-1).cpu().numpy()     # [B,T]
            for i in range(norm.shape[0]):
                n = int(mask[i].sum())
                if n <= 0:
                    continue
                idx = np.floor(np.arange(n) / max(n, 1) * n_bins).astype(int)
                idx = np.clip(idx, 0, n_bins - 1)
                np.add.at(mem_sum, idx, norm[i, :n])
                np.add.at(mem_cnt, idx, 1.0)
        if gate is not None:
            g = gate.detach().float()
            if g.dim() == 3:
                g = g.mean(-1)
            g_np = g.cpu().numpy()
            gate_vals.append(g_np[mask])
    out: Dict[str, Any] = {"available": bool(mem_cnt.sum() or gate_vals),
                           "n_bins": n_bins}
    if mem_cnt.sum() > 0:
        out["mem_norm_curve"] = (mem_sum / np.maximum(mem_cnt, 1.0)).tolist()
        out["mem_norm_last"] = float(out["mem_norm_curve"][-1])
        out["mem_norm_first"] = float(out["mem_norm_curve"][0])
        out["mem_norm_ratio"] = float(out["mem_norm_curve"][-1] /
                                      max(out["mem_norm_curve"][0], 1e-9))
    if gate_vals:
        g = np.concatenate(gate_vals)
        g = g[np.isfinite(g)]
        hist, edges = np.histogram(g, bins=gate_bins, range=(0.0, 1.0))
        out["write_gate_hist"] = {"counts": hist.astype(int).tolist(),
                                  "edges": edges.tolist(),
                                  "mean": float(g.mean()) if g.size else None,
                                  "frac_open": float((g > 0.5).mean()) if g.size else None,
                                  "n": int(g.size)}
    return out


def probe_mem_norm(model, loader, cfg=None, n_bins: int = 16) -> Dict[str, Any]:
    """(b) per-step memory-read norm over normalized time."""
    d = collect_aux_curves(model, loader, cfg, n_bins=n_bins)
    if "mem_norm_curve" not in d:
        return {"available": False, "reason": "core does not expose aux['read']"}
    return {k: d[k] for k in ("available", "mem_norm_curve", "mem_norm_first",
                              "mem_norm_last", "mem_norm_ratio")}


def probe_write_gate_hist(model, loader, cfg=None, bins: int = 20) -> Dict[str, Any]:
    """(c) write-gate histogram."""
    d = collect_aux_curves(model, loader, cfg, gate_bins=bins)
    if "write_gate_hist" not in d:
        return {"available": False, "reason": "core does not expose aux['write_gate']"}
    return {"available": True, **d["write_gate_hist"]}


# --------------------------------------------------------------------------- #
# (d) state-reset ablation
# --------------------------------------------------------------------------- #
def _slice_batch(batch: dict, s: int, e: int) -> dict:
    out = dict(batch)
    for k in ("x", "t", "dt", "mask", "tok", "roi", "roi_valid", "roi_meta"):
        v = batch.get(k)
        if v is not None:
            out[k] = v[:, s:e]
    dt = out["dt"].clone()
    if dt.shape[1] > 0:
        dt[:, 0] = 0.0          # a fresh state at a chunk boundary: dt is irrelevant
    out["dt"] = dt
    return out


def _logits_with_reset(model, batch: dict, chunk: int) -> Dict[str, torch.Tensor]:
    T = int(batch["x"].shape[1])
    if not chunk or chunk <= 0 or chunk >= T:
        with torch.no_grad():
            logits, _aux, _out = model(batch)
        return logits
    parts: List[Dict[str, torch.Tensor]] = []
    s = 0
    while s < T:
        e = min(T, s + chunk)
        with torch.no_grad():
            lg, _aux, _out = model(_slice_batch(batch, s, e))
        parts.append(lg)
        s = e
    return {k: torch.cat([p[k] for p in parts], dim=1) for k in parts[0]}


def probe_state_reset(model, loader, cfg: dict,
                      chunk_lens: Sequence[int] = (8, 16, 32, 64)) -> Dict[str, Any]:
    """(d) Reset the recurrent state every k steps and measure the drop.

    Implemented by re-running the (stateless, causal) forward on consecutive
    chunks: the first step of every chunk starts from the initial state, which
    IS a state reset.  No model internals are touched.
    """
    cfg = cfg or {}
    heads_cfg = head_specs(cfg)
    main_metric = _main_metric(cfg, heads_cfg)
    main_head = pick_main_head(heads_cfg, main_metric)
    main_kind = str(heads_cfg[main_head].get("kind", "ce"))
    device = next(model.parameters()).device
    model.eval()
    out: Dict[str, Any] = {"available": True, "metric": main_metric, "chunks": {}}
    for chunk in [0] + list(chunk_lens):
        per_seq: List[dict] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                logits = _logits_with_reset(model, batch, int(chunk))
                per_seq.extend(batch_to_per_seq(batch, logits, heads_cfg))
        hm = head_metrics(per_seq, heads_cfg, main_head, main_metric)
        out["chunks"][("none" if not chunk else f"k{int(chunk)}")] = hm["metrics"]
    base = out["chunks"].get("none", {}).get(main_metric)
    hb = higher_is_better(main_metric)
    drops = {}
    for k, mm in out["chunks"].items():
        v = mm.get(main_metric)
        if k == "none" or v is None or base is None:
            continue
        drops[k] = float(base - v) if hb else float(v - base)
    out["drops"] = drops
    return out


# --------------------------------------------------------------------------- #
# (e) linear probe of the memory readout for the hidden target
# --------------------------------------------------------------------------- #
def _hidden_flags(batch: dict, head: str) -> Optional[np.ndarray]:
    """Which steps have the target hidden (occluded).

    Priority: (1) roi_valid.all(False) -> occluded; (2) lab_mask != 0 marks the
    entries the dataset itself says are the hidden/recall phase; (3) all-zero
    target entries.
    """
    rv = batch.get("roi_valid")
    labels = batch.get("labels") or {}
    mask_np = batch["mask"].detach().cpu().numpy()
    if rv is not None:
        occ = ~rv.any(-1).detach().cpu().numpy()
        return occ & mask_np
    lm = labels.get("mask")
    if lm is not None:
        lm_np = lm.detach().cpu().numpy()
        return (lm_np != 0) & mask_np
    tgt = labels.get(head)
    if tgt is None:
        return None
    tgt_np = tgt.detach().cpu().numpy()
    zero = np.all(np.abs(tgt_np) < 1e-12, axis=-1) if tgt_np.ndim > 2 else (
        np.abs(tgt_np) < 1e-12)
    return zero & mask_np


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[1] > X.shape[0]:
        A = X @ X.T
        W = X.T @ np.linalg.solve(A + lam * np.eye(A.shape[0]), Y)
    else:
        A = X.T @ X
        W = np.linalg.solve(A + lam * np.eye(A.shape[0]), X.T @ Y)
    return W


def _collect_probe_data(model, loader, cfg: dict, feature: str = "hidden",
                        device=None) -> Dict[str, Any]:
    heads_cfg = head_specs(cfg)
    main_metric = _main_metric(cfg, heads_cfg)
    main_head = pick_main_head(heads_cfg, main_metric)
    kind = str(heads_cfg[main_head].get("kind", "ce"))
    device = device or next(model.parameters()).device
    Xs, Ys, Hs = [], [], []
    for batch in _iter_batches(model, loader, device):
        logits, aux, out = model(batch)
        f = aux_get(aux, feature)
        if f is None and feature in ("hidden", "read"):
            f = out
        if f is None:
            continue
        tgt = (batch.get("labels") or {}).get(main_head)
        if tgt is None:
            continue
        X = f.detach().float().cpu().numpy()
        Y = tgt.detach().cpu().numpy()
        mask = batch["mask"].detach().cpu().numpy()
        hid = _hidden_flags(batch, main_head)
        B, T, D = X.shape
        Xf = X.reshape(B * T, D)
        if Y.ndim == 2:
            Yf = Y.reshape(B * T, 1)
        else:
            Yf = Y.reshape(B * T, -1)
        mf = mask.reshape(-1)
        hf = hid.reshape(-1) if hid is not None else np.zeros(B * T, dtype=bool)
        if kind == "ce":
            keep = mf & (Yf[:, 0] != -1)
        elif kind == "bce":
            keep = mf & np.isfinite(Yf).all(1)
        else:
            keep = mf & (np.abs(Yf) > 0).any(1)
            if hid is None:
                hf = ~keep
        Xs.append(Xf[keep])
        Ys.append(Yf[keep])
        Hs.append(hf[keep])
    if not Xs:
        return {"available": False, "reason": "no features/labels collected"}
    return {"X": np.concatenate(Xs), "Y": np.concatenate(Ys),
            "hidden": np.concatenate(Hs), "head": main_head, "kind": kind,
            "metric": main_metric}


def _probe_scores(kind: str, Y: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    if kind == "ce":
        return {"acc": float((P.argmax(1) == Y[:, 0].astype(int)).mean())}
    if kind == "bce":
        return {"f1": float(((P >= 0.5) == (Y >= 0.5)).mean())}
    err = P - Y
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    r2 = 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float("nan")
    yv, pv = Y.reshape(-1), P.reshape(-1)
    cos = (float(np.dot(yv, pv) / (np.linalg.norm(yv) * np.linalg.norm(pv) + 1e-12))
           if np.linalg.norm(yv) > 1e-12 else float("nan"))
    return {"r2": r2, "mae": float(np.abs(err).mean()), "cos": cos}


def probe_linear_readout(model, train_loader, eval_loader, cfg: dict,
                         feature: str = "hidden", lam: float = 1e-2) -> Dict[str, Any]:
    """(e) Is the occluded target linearly decodable from the memory readout?"""
    cfg = cfg or {}
    model.eval()
    tr = _collect_probe_data(model, train_loader, cfg, feature)
    if not tr.get("available", True) or "X" not in tr:
        return {"available": False, "reason": tr.get("reason", "no data")}
    ev = _collect_probe_data(model, eval_loader, cfg, feature)
    if "X" not in ev:
        return {"available": False, "reason": ev.get("reason", "no eval data")}
    X, Y = tr["X"], tr["Y"]
    if tr["kind"] in ("ce",):
        n_cls = int(max(Y.max(), 0)) + 1
        Yoh = np.eye(n_cls)[Y[:, 0].astype(int)]
    else:
        Yoh = Y
    W = _ridge_fit(X, Yoh, lam)
    P = ev["X"] @ W
    out: Dict[str, Any] = {"available": True, "feature": feature,
                           "head": ev["head"], "kind": ev["kind"],
                           "n_train": int(X.shape[0]), "n_eval": int(ev["X"].shape[0]),
                           "dim": int(X.shape[1])}
    vis = ~ev["hidden"]
    out["visible"] = _probe_scores(ev["kind"], ev["Y"][vis], P[vis]) if vis.any() else {}
    out["hidden"] = _probe_scores(ev["kind"], ev["Y"][ev["hidden"]], P[ev["hidden"]]) \
        if ev["hidden"].any() else {}
    out["all"] = _probe_scores(ev["kind"], ev["Y"], P)
    return out


# --------------------------------------------------------------------------- #
# (f) crosstalk matrix
# --------------------------------------------------------------------------- #
def probe_crosstalk(model, loader, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """(f) Slot co-activation matrix (N x N) and, when K>1, key overlap (K x K)."""
    device = next(model.parameters()).device
    slot_sum = None
    k_sum = None
    k_cnt = 0.0
    n_seen = 0
    for batch in _iter_batches(model, loader, device):
        _logits, aux, _out = model(batch)
        route = aux_get(aux, "route")
        if route is None:
            continue
        m = slot_crosstalk(route, batch["mask"])
        if m is not None:
            slot_sum = m if slot_sum is None else slot_sum + m
            n_seen += 1
        R = route.detach().float()
        R = R / R.sum(-1, keepdim=True).clamp_min(1e-6)
        K = int(R.shape[-2])
        if K > 1:
            mask = batch["mask"].detach().float()
            C = torch.einsum("btkn,btln->btkl", R, R)               # [B,T,K,K]
            w = mask.sum()
            Csum = (C * mask[..., None, None]).sum((0, 1)).cpu().numpy()
            k_sum = Csum if k_sum is None else k_sum + Csum
            k_cnt += float(w)
    out: Dict[str, Any] = {"available": slot_sum is not None or k_sum is not None}
    if slot_sum is not None:
        out["slot_matrix"] = (slot_sum / max(n_seen, 1)).tolist()
        Mx = np.asarray(out["slot_matrix"])
        n = Mx.shape[0]
        if n > 1:
            off = (Mx.sum() - np.trace(Mx)) / (n * (n - 1))
            out["slot_offdiag_mean"] = float(off)
            out["slot_diag_mean"] = float(np.trace(Mx) / n)
    if k_sum is not None and k_cnt > 0:
        out["key_matrix"] = (k_sum / k_cnt).tolist()
    return out


# --------------------------------------------------------------------------- #
def run_all_probes(model, val_loader, cfg: dict, test_loader=None,
                   train_loader=None, chunk_lens: Sequence[int] = (8, 16, 32),
                   seed: int = 0) -> Dict[str, Any]:
    """Run every probe that the model/data supports; never raises."""
    cfg = cfg or {}
    out: Dict[str, Any] = {}
    ev = test_loader if test_loader is not None else val_loader
    for name, fn in (
        ("dt_importance", lambda: probe_dt_importance(model, ev, cfg, seed=seed)),
        ("mem_norm", lambda: probe_mem_norm(model, ev, cfg)),
        ("write_gate_hist", lambda: probe_write_gate_hist(model, ev, cfg)),
        ("state_reset", lambda: probe_state_reset(model, ev, cfg, chunk_lens)),
        ("crosstalk", lambda: probe_crosstalk(model, ev, cfg)),
    ):
        try:
            out[name] = fn()
        except Exception as exc:                        # pragma: no cover - defensive
            out[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if train_loader is not None:
        try:
            out["linear_readout"] = probe_linear_readout(model, train_loader, ev, cfg)
        except Exception as exc:                        # pragma: no cover
            out["linear_readout"] = {"available": False,
                                     "error": f"{type(exc).__name__}: {exc}"}
    return out
