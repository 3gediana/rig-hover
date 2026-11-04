"""Model construction: temporal core through ``src.models.build_core`` + shared heads.

Every core is built exactly one way so that cores, heads and the optimizer are
identical across model families.  ``src.models.build_core`` is resolved lazily at
call time (model-zoo owns that module and may land it after this file was
written); if it is missing, ONLY the model name ``"dummy"`` falls back to the
internal smoke-test core -- a real model name never silently falls back.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.common.contract import MultiTaskHeads, count_params

VALID_KINDS = ("ce", "bce", "mse")
HEAD_KIND_DEFAULT = "ce"

#: How optional per-object tokens reach the core.  Pinned by the Lead: this must
#: be identical for every architecture (the information the model gets must not
#: vary by model), so it happens in the shared input path, never inside a core.
#:   "none"    x = feat                                   (tok absent)
#:   "flatten" x = cat([feat, tok.reshape(B,T,-1)])       (synth: K object slots)
#:   "mean"    x = cat([feat, tok.mean(-2)])              (real video: spatial tokens)
TOK_FUSION_MODES = ("none", "flatten", "mean")


def tok_fusion_mode(cfg: Optional[dict] = None) -> str:
    cfg = cfg or {}
    tcfg = dict(cfg.get("train") or {})
    mcfg = dict(cfg.get("model") or {})
    mode = str(tcfg.get("tok_fusion") or mcfg.get("tok_fusion")
               or cfg.get("tok_fusion") or "none").lower()
    if mode not in TOK_FUSION_MODES:
        raise ValueError(f"train.tok_fusion={mode!r} not in {TOK_FUSION_MODES}")
    return mode


def resolve_in_dim(cfg: Optional[dict], feature_dim: int,
                   tok_shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """(in_dim, extra_dims) for the configured token fusion.

    ``tok_shape`` = (K, Dt) of the token tensor.  flatten -> K*Dt extra dims,
    mean -> Dt extra dims.  Raises if the mode needs tokens and the shape is
    unknown (so a config can never silently train a blind model).
    """
    mode = tok_fusion_mode(cfg)
    feature_dim = int(feature_dim)
    if mode == "none":
        return feature_dim, 0
    if tok_shape is None:
        raise ValueError(
            f"train.tok_fusion={mode!r} needs token tensors but no token shape is "
            f"known (manifest has tok_dim=0 and no tok tensor was seen). "
            f"Provide a tok-bearing manifest or set train.tok_fusion='none'.")
    K, Dt = int(tok_shape[0]), int(tok_shape[1])
    extra = K * Dt if mode == "flatten" else Dt
    return feature_dim + extra, extra


def fuse_tokens(batch: dict, mode: str) -> dict:
    """Return a batch whose ``x`` includes the token information.

    One place, all models: this is the only token-fusion implementation in the
    training/eval path.  Padded steps keep x=0 for the token part (mask handles).
    """
    mode = str(mode or "none").lower()
    if mode == "none":
        return batch
    tok = batch.get("tok")
    if tok is None:
        raise ValueError(
            f"tok_fusion={mode!r} but this batch has no 'tok' tensor. Either the "
            f"manifest has no tokens, load_tok is off, or the eval batch was built "
            f"without them.")
    x = batch["x"]
    B, T = int(x.shape[0]), int(x.shape[1])
    if tok.dim() == 4:                      # [B,T,K,Dt]
        t = tok.reshape(B, T, -1) if mode == "flatten" else tok.mean(-2)
    elif tok.dim() == 3:                    # [B,T,Dt] single token per step
        t = tok
    else:
        raise ValueError(f"unsupported token shape {tuple(tok.shape)}")
    out = dict(batch)
    out["x"] = torch.cat([x, t.to(dtype=x.dtype, device=x.device)], dim=-1)
    return out


# --------------------------------------------------------------------------- #
# build_core resolution
# --------------------------------------------------------------------------- #
def resolve_build_core():
    """Return ``src.models.build_core`` if available, else None."""
    try:
        from src import models  # noqa: F401
    except Exception:
        return None
    return getattr(models, "build_core", None)


def build_core(name: str, in_dim: int, hidden: int, **cfg) -> nn.Module:
    """Build one temporal core by name (keyword signature preferred).

    The internal ``"dummy"`` smoke core always routes to the local fallback and
    never to ``src.models`` (which does not know it); every real name goes
    through ``src.models.build_core``.
    """
    if name == "dummy":
        from src.train._fallback_core import DummyCore
        return DummyCore(int(in_dim), int(hidden), **cfg)

    fn = resolve_build_core()
    if fn is not None:
        attempts = (
            lambda: fn(name, in_dim=int(in_dim), hidden=int(hidden), **cfg),
            lambda: fn(name, int(in_dim), int(hidden), **cfg),
            lambda: fn(name, in_dim=int(in_dim), hidden=int(hidden)),
        )
        last: Optional[BaseException] = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:      # signature mismatch -> next form
                last = exc
        raise RuntimeError(
            f"src.models.build_core failed for name={name!r} in_dim={in_dim} "
            f"hidden={hidden} cfg={cfg}: {last}") from last

    raise RuntimeError(
        f"model name {name!r} requested but src.models.build_core does not exist "
        f"yet (only the internal 'dummy' smoke core is available). "
        f"Asked model-zoo for build_core(name, in_dim, hidden, **cfg).")


def core_available(name: str) -> bool:
    if resolve_build_core() is not None:
        return True
    return name == "dummy"


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
def build_heads(heads_cfg: Dict[str, Dict[str, Any]], in_dim: int,
                cfg: Optional[dict] = None) -> MultiTaskHeads:
    """Normalise the config head spec into ``contract.MultiTaskHeads``'s spec.

    Config form:  {"action": {"kind": "ce", "dim": 101, "ignore": -1}}
    Contract form: same + ``n_classes`` for ce/bce (n_out), ``dim`` for mse.
    """
    if not heads_cfg:
        raise ValueError("cfg['heads'] is empty: at least one output head is required")
    spec: Dict[str, Dict[str, Any]] = {}
    for name, raw in heads_cfg.items():
        sp = dict(raw or {})
        kind = str(sp.get("kind", HEAD_KIND_DEFAULT)).lower()
        if kind not in VALID_KINDS:
            raise ValueError(f"head {name!r}: unsupported kind {kind!r}")
        dim = int(sp.get("dim", sp.get("n_classes", 1)))
        entry: Dict[str, Any] = {"kind": kind, "dim": dim}
        if kind in ("ce", "bce"):
            entry["n_classes"] = int(sp.get("n_classes", dim))
        if "ignore" in sp and sp["ignore"] is not None:
            entry["ignore"] = int(sp["ignore"])
        spec[name] = entry

    cfg = cfg or {}
    tcfg = cfg.get("train") or {}
    hidden = int(cfg.get("head_hidden") or tcfg.get("head_hidden") or 128)
    dropout = float(cfg.get("head_dropout") or tcfg.get("head_dropout") or 0.0)
    return MultiTaskHeads(int(in_dim), spec, hidden=hidden, dropout=dropout)


def infer_out_dim(core: nn.Module, feature_dim: int, hidden: int) -> int:
    """Cores should declare ``out_dim``; if not, infer it from a 2-step forward."""
    od = getattr(core, "out_dim", 0)
    if od:
        return int(od)
    try:
        x = torch.zeros(1, 2, int(feature_dim))
        dt = torch.zeros(1, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        out = core(x, dt, mask)
        out = out[0] if isinstance(out, tuple) else out
        return int(out.shape[-1])
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"[factory] could not infer core out_dim ({exc}); "
                      f"falling back to hidden={hidden}")
        return int(hidden)


# --------------------------------------------------------------------------- #
# model wrapper
# --------------------------------------------------------------------------- #
class StreamModel(nn.Module):
    """Core + shared heads, one forward call used by training AND evaluation.

    Token fusion happens HERE (see :func:`fuse_tokens`), so training, evaluation,
    probes, the streaming path and the time-consistency aux all see the same
    input width and every architecture gets the same information.
    """

    def __init__(self, core: nn.Module, heads: MultiTaskHeads, name: str = "model",
                 tok_fusion: str = "none", in_dim: int = 0, feature_dim: int = 0):
        super().__init__()
        self.core = core
        self.heads = heads
        self.name = str(name)
        self.tok_fusion = str(tok_fusion or "none").lower()
        self.in_dim = int(in_dim)
        self.feature_dim = int(feature_dim)
        self._core_mode: Optional[str] = None

    # -- params ---------------------------------------------------------- #
    @property
    def n_params(self) -> int:
        return count_params(self)

    def param_breakdown(self) -> Dict[str, int]:
        return {"core": count_params(self.core), "heads": count_params(self.heads),
                "total": count_params(self)}

    def prepare(self, batch: dict) -> dict:
        """Apply the shared token-fusion rule (single place in the input path)."""
        return fuse_tokens(batch, self.tok_fusion)

    # -- forward --------------------------------------------------------- #
    def _core_forward(self, x, dt, mask, batch):
        kw = dict(tok=batch.get("tok"), tok_valid=batch.get("tok_valid"),
                  roi=batch.get("roi"), roi_valid=batch.get("roi_valid"),
                  roi_meta=batch.get("roi_meta"))
        if self._core_mode == "simple":
            return self.core(x, dt, mask)
        try:
            out = self.core(x, dt, mask, **kw)
            self._core_mode = "full"
            return out
        except TypeError as exc:
            warnings.warn(f"[factory] core.forward(x,dt,mask,**kw) raised TypeError "
                          f"({exc}); retrying with positional (x,dt,mask) only")
            out = self.core(x, dt, mask)
            self._core_mode = "simple"
            return out

    def forward(self, batch: dict) -> Tuple[Dict[str, torch.Tensor], dict, torch.Tensor]:
        """batch (contract §1.4) -> (logits {head: [B,T,C]}, aux dict, out [B,T,H])"""
        batch = self.prepare(batch)
        core_out = self._core_forward(batch["x"], batch["dt"], batch["mask"], batch)
        if isinstance(core_out, tuple) and len(core_out) == 2:
            out, aux = core_out
        else:
            out, aux = core_out, {}
        if aux is None:
            aux = {}
        logits = self.heads(out)
        return logits, aux, out

    # -- streaming ------------------------------------------------------- #
    def init_state(self, B: int, device=None):
        device = device if device is not None else next(self.parameters()).device
        if hasattr(self.core, "init_state"):
            return self.core.init_state(B, device)
        return {}

    def step(self, batch: dict, state=None):
        """Single-frame streaming forward when the core supports it.

        batch must have batch dim B and time dim 1.  Returns
        (logits, aux, new_state) -- raises NotImplementedError without a native
        core.step so callers can fall back to prefix replay.
        """
        if not hasattr(self.core, "step"):
            raise NotImplementedError("core has no native step(); use prefix replay")
        batch = self.prepare(batch)
        kw = dict(tok_t=None, roi_t=batch.get("roi"), roi_valid_t=batch.get("roi_valid"),
                  roi_meta_t=batch.get("roi_meta"))
        aux_tok = batch.get("tok")
        if aux_tok is not None:
            # tokens were fused into x by prepare(); cores ignore tok_t, pass it
            # only for completeness when they accept it
            kw["tok_t"] = aux_tok[:, 0] if aux_tok.dim() == 4 else aux_tok
        out, aux, new_state = self.core.step(
            batch["x"][:, 0], batch["dt"][:, 0], batch["mask"][:, 0],
            state=state, **kw)
        logits = self.heads(out)
        return logits, aux, new_state


# --------------------------------------------------------------------------- #
def build_model(cfg: dict, feature_dim: int,
                tok_shape: Optional[Tuple[int, int]] = None) -> StreamModel:
    """Build the model described by a run config (see DESIGN §3 / config schema).

    ``feature_dim`` is the raw `feat` width; the configured ``train.tok_fusion``
    decides the true input width passed to the core (``tok_shape`` = (K, Dt)).
    """
    mcfg = dict(cfg.get("model") or {})
    name = str(mcfg.get("name") or "dummy")
    hidden = int(mcfg.get("hidden") or 192)
    core_cfg = dict(mcfg.get("cfg") or {})
    mode = tok_fusion_mode(cfg)
    shape = tok_shape or cfg.get("_tok_shape")
    in_dim, extra = resolve_in_dim(cfg, int(feature_dim), shape)
    core = build_core(name, in_dim=int(in_dim), hidden=hidden, **core_cfg)
    out_dim = infer_out_dim(core, int(in_dim), hidden)
    heads = build_heads(cfg.get("heads") or {}, out_dim, cfg)
    return StreamModel(core=core, heads=heads, name=name, tok_fusion=mode,
                       in_dim=in_dim, feature_dim=int(feature_dim))


def head_specs(cfg: dict) -> Dict[str, Dict[str, Any]]:
    """The normalised head spec actually used to build the model."""
    model = build_heads(cfg.get("heads") or {}, 1, cfg)
    return dict(model.spec)
