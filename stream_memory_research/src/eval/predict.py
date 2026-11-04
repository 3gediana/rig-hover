"""Streaming (single-frame) inference path -- the deployed deliverable shape.

Two modes, chosen automatically:

* **native** -- the core implements ``core.step(x_t, dt_t, mask_t, state)``
  (all model-zoo cores except ``mlp`` do).  O(1) per frame.
* **prefix replay** -- no native step; the model is re-run on the whole prefix
  every frame.  Causally correct but O(T^2): used only as a fallback so the
  pipeline always has a streaming path.

    pred = StreamPredictor(model, cfg)
    pred.reset()
    for t, frame in enumerate(frames):
        out = pred.step(frame, t_s=ts[t])
        cls = out["pred"]["action"]

``predict_stream(model, seq, cfg)`` runs a whole sequence through the streaming
path and returns per-step predictions; ``measure_stream_latency`` measures the
per-frame cost of the native path (what the paper should report).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.train import data as tdata
from src.train.factory import head_specs


def _to_batch1(x_t: torch.Tensor, dt_t: torch.Tensor, mask_t: torch.Tensor,
               tok_t=None, roi_t=None, roi_valid_t=None, roi_meta_t=None) -> dict:
    """Single-frame batch in contract §1.4 layout ([B,1,...])."""
    B = int(x_t.shape[0])
    if dt_t.dim() == 1:
        dt_t = dt_t.unsqueeze(-1)
    if mask_t.dim() == 1:
        mask_t = mask_t.unsqueeze(-1)
    batch = {"x": x_t.unsqueeze(1) if x_t.dim() == 2 else x_t,
             "dt": dt_t, "mask": mask_t,
             "t": torch.zeros(B, 1, device=x_t.device, dtype=x_t.dtype),
             "labels": {}, "tok": None, "roi": None, "roi_valid": None, "roi_meta": None,
             "meta": [{"seq_id": f"stream{i}", "source_id": f"stream{i}",
                       "split": "test"} for i in range(B)]}
    if tok_t is not None:
        batch["tok"] = tok_t[:, None] if tok_t.dim() == 3 else tok_t
    if roi_t is not None:
        batch["roi"] = roi_t[:, None] if roi_t.dim() == 3 else roi_t
    if roi_valid_t is not None:
        batch["roi_valid"] = roi_valid_t[:, None] if roi_valid_t.dim() == 2 else roi_valid_t
    if roi_meta_t is not None:
        batch["roi_meta"] = roi_meta_t[:, None] if roi_meta_t.dim() == 3 else roi_meta_t
    return batch


class StreamPredictor:
    """Single-frame causal inference over a built model (any core)."""

    def __init__(self, model, cfg: Optional[dict] = None, device=None):
        self.model = model
        self.model.eval()
        self.cfg = dict(cfg or {})
        self.device = device or next(model.parameters()).device
        self.heads_cfg = head_specs(self.cfg) if self.cfg.get("heads") else {}
        self.native = hasattr(self.model, "step")
        self.reset()

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, B: int = 1) -> None:
        self.B = int(B)
        self.state = self.model.init_state(self.B, self.device) if self.native else {}
        self._x: List[torch.Tensor] = []
        self._t: List[float] = []
        self._mask: List[bool] = []
        self._last_t: Optional[float] = None

    # -- one frame -------------------------------------------------------- #
    def _head_preds(self, logits: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        out = {}
        for name, spec in self.heads_cfg.items():
            lg = logits.get(name) if isinstance(logits, dict) else None
            if lg is None:
                continue
            kind = str(spec.get("kind", "ce"))
            a = lg.detach().float()
            if kind == "ce":
                out[name] = a.argmax(-1).cpu().numpy()
            elif kind == "bce":
                out[name] = torch.sigmoid(a).cpu().numpy()
            else:
                out[name] = a.cpu().numpy()
        return out

    def step(self, feat, t_s: Optional[float] = None, mask: bool = True,
             tok=None, roi=None, roi_valid=None, roi_meta=None) -> Dict[str, Any]:
        """Feed ONE frame; returns {"pred": {head: [B,...]}, "aux": {...}, "t": t}."""
        x = torch.as_tensor(feat, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] != self.B:
            self.reset(B=int(x.shape[0]))
        t = (float(t_s) if t_s is not None else
             (0.0 if self._last_t is None else self._last_t + 1.0 / 30.0))
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        dt_t = torch.full((self.B,), float(dt), dtype=torch.float32, device=self.device)
        mask_t = torch.full((self.B,), bool(mask), dtype=torch.bool, device=self.device)
        self._last_t = t
        if self.native:
            batch = _to_batch1(x, dt_t, mask_t, tok, roi, roi_valid, roi_meta)
            logits, aux, self.state = self.model.step(batch, self.state)
        else:
            self._x.append(x.detach())
            self._t.append(t)
            self._mask.append(bool(mask))
            X = torch.stack(self._x, dim=1)                     # [B,T,D]
            T = X.shape[1]
            tt = torch.tensor(self._t, dtype=torch.float32, device=self.device)
            tt = tt.unsqueeze(0).expand(self.B, T)
            dtt = torch.zeros(self.B, T, device=self.device)
            if T > 1:
                dtt[:, 1:] = (tt[:, 1:] - tt[:, :-1]).clamp_min(0.0)
            mm = torch.tensor(self._mask, dtype=torch.bool,
                              device=self.device).unsqueeze(0).expand(self.B, T)
            batch = {"x": X, "t": tt, "dt": dtt, "mask": mm, "labels": {},
                     "tok": None, "roi": None, "roi_valid": None, "roi_meta": None,
                     "meta": [{"seq_id": f"stream{i}", "source_id": f"stream{i}",
                               "split": "test"} for i in range(self.B)]}
            with torch.no_grad():
                logits, aux, _out = self.model(batch)
            logits = {k: v[:, -1:] for k, v in logits.items()}
        return {"pred": self._head_preds(logits), "aux": aux, "t": t}

    # -- whole sequence --------------------------------------------------- #
    def run_sequence(self, seq: dict) -> Dict[str, Any]:
        """Stream one raw sequence dict (feat [T,Dg], t [T], optional tok/roi)."""
        feat = np.asarray(seq["feat"], dtype=np.float32)
        t = np.asarray(seq["t"], dtype=np.float64)
        T = feat.shape[0]
        self.reset(B=1)
        preds: Dict[str, List[np.ndarray]] = {}
        aux_last = None
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(T):
                tok_t = None
                if seq.get("tok") is not None:
                    tok_t = torch.as_tensor(seq["tok"][i], dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                roi_t = roi_v = roi_m = None
                if seq.get("roi") is not None:
                    roi_t = torch.as_tensor(seq["roi"][i], dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                    if seq.get("roi_valid") is not None:
                        roi_v = torch.as_tensor(seq["roi_valid"][i], dtype=torch.bool,
                                                device=self.device).unsqueeze(0)
                    if seq.get("roi_meta") is not None:
                        roi_m = torch.as_tensor(seq["roi_meta"][i], dtype=torch.float32,
                                                device=self.device).unsqueeze(0)
                out = self.step(feat[i], t_s=float(t[i]), tok=tok_t, roi=roi_t,
                                roi_valid=roi_v, roi_meta=roi_m)
                for k, v in out["pred"].items():
                    preds.setdefault(k, []).append(np.asarray(v).reshape(-1))
                aux_last = out["aux"]
        dt = time.perf_counter() - t0
        return {
            "seq_id": seq.get("seq_id", "stream"),
            "pred": {k: np.stack(v, 0) for k, v in preds.items()},
            "native_step": bool(self.native),
            "n_steps": int(T),
            "ms_per_step": 1000.0 * dt / max(1, T),
            "t": t,
        }


def predict_stream(model, seq: dict, cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Convenience: stream one sequence through a fresh predictor."""
    return StreamPredictor(model, cfg).run_sequence(seq)


def measure_stream_latency(model, cfg: Optional[dict], n_steps: int = 128,
                           warmup: int = 8) -> Dict[str, Any]:
    """Per-frame latency of the streaming path on the model's device.

    Native ``step()`` is measured when available (that is the deployable number);
    otherwise we report the amortized full-sequence forward with
    ``native_step=False`` so the report never confuses the two.
    """
    device = next(model.parameters()).device
    base_dim = int((cfg or {}).get("_feature_dim") or 0)
    tok_dim = int((cfg or {}).get("_tok_dim") or 0)
    n_tok = int((cfg or {}).get("_n_tok") or 0)
    fusion = str(getattr(model, "tok_fusion", "none") or "none")
    core_in = getattr(getattr(model, "core", None), "in_dim", 0) or 0
    if not base_dim:
        # recover the raw width from the resolved core width and the fusion rule
        extra = (n_tok * tok_dim if fusion == "flatten"
                 else (tok_dim if fusion == "mean" else 0))
        base_dim = int(max(1, core_in - extra)) if core_in else 1
    roi_dim = int((cfg or {}).get("_roi_dim") or 0)
    n_roi = int((cfg or {}).get("_n_roi") or 0)
    cuda = device.type == "cuda"
    model.eval()
    pred = StreamPredictor(model, cfg, device=device)
    if pred.native:
        x = torch.zeros(1, base_dim, device=device)
        tok = (torch.zeros(1, n_tok, tok_dim, device=device)
               if (fusion != "none" and n_tok and tok_dim) else None)
        with torch.no_grad():
            for i in range(max(1, warmup)):
                pred.step(x, t_s=float(i) / 30.0, tok=tok)
            pred.reset()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(int(n_steps)):
                pred.step(x, t_s=float(i) / 30.0, tok=tok)
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        return {"native_step": True, "latency_ms_per_step": 1000.0 * dt / max(1, n_steps),
                "n_steps": int(n_steps), "in_dim": int(core_in or 0)}
    # fallback: amortized full-sequence forward (NOT a per-step number)
    batch = tdata.make_dummy_batch(1, 128, base_dim, tok_dim=tok_dim, n_tok=n_tok,
                                   roi_dim=roi_dim, n_roi=n_roi, device=device)
    with torch.no_grad():
        for _ in range(2):
            model(batch)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {"native_step": False, "latency_ms_per_step": 1000.0 * dt / 128.0,
            "n_steps": 128, "note": "no native core.step(); amortized full-sequence"}
