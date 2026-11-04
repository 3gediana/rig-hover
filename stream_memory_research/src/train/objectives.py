"""Task losses + the five auxiliary objectives (DESIGN §2.3, §3).

Design rules
------------
1. ``task_losses`` uses the head spec from ``cfg["heads"]`` (kind in ce/bce/mse,
   ignore semantics from the config) and is masked over padded steps.
2. Every auxiliary objective reads tensors from the CORE's ``aux`` dict and is a
   strict no-op (returns a zero tensor, contributes no gradient, allocates no
   parameters) when the core does not expose what it needs.  Each one is
   independently switchable to 0.0 from ``cfg["loss"]["aux"]`` for ablations.
3. Auxiliary tensors, by contract with model-zoo (aliases accepted):

       read        [B,T,Dr]      fast-memory readout      (alias mem_read)
       hidden      [B,T,H]       slow state                (alias h)
       write_gate  [B,T] or [B,T,N]   write gate beta
       route       [B,T,K,N]     key/target -> slot routing (rows sum over N)
       slot_read   [B,T,N,Ds]    per-slot readout

Objective definitions
----------------------
retrieval        linear probe from `read` to the primary task label; trains the
                 memory readout to be linearly decodable (gradient flows into the
                 core).  The probe is tiny and only exists when weight > 0.
crosstalk        isolation: route rows for different keys should not share slots
                 -> mean off-diagonal of R R^T, plus a slot-coverage balance term
                 KL(usage || uniform).  Needs `route`.
time_consistency the same physical process sampled at 1x and at 1/factor x must
                 produce close core representations at the SAME physical time.
                 Needs a second forward; loss = mean(1 - cos) over matched steps.
write_budget     mean write gate (sparsify writes).  Needs `write_gate`.
long_horizon     multi-step (>=h steps, default h=5) memory-retention loss: the
                 memory readout must stay close to itself h steps later, weighted
                 by how long it has been since the last real WRITE.  Spanning >=2
                 steps is deliberate: single-step residual losses leave the write
                 channel without gradient.  Needs `read` + `write_gate` + `t`.
                 Config: loss.aux_cfg.long_horizon = {"horizon": 5, "tau": 1.0,
                 "write_thr": 0.1}.  The horizon is recorded in metrics.json.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.train.data import downsample_batch

ZERO_STATS: Dict[str, float] = {}

AUX_ALIASES: Dict[str, Tuple[str, ...]] = {
    "read": ("read", "mem_read", "memory_read", "readout"),
    "hidden": ("hidden", "h", "state"),
    "write_gate": ("write_gate", "write_gates", "beta", "gate", "g"),
    "route": ("route", "routes", "slot_use", "routing"),
    "slot_read": ("slot_read", "slot_reads", "slots_read"),
}

AUX_NAMES: Tuple[str, ...] = ("retrieval", "crosstalk", "time_consistency",
                              "write_budget", "long_horizon")


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def aux_get(aux: Optional[dict], name: str) -> Optional[torch.Tensor]:
    """Fetch a tensor from the aux dict by canonical name or known alias."""
    if not isinstance(aux, dict):
        return None
    for key in AUX_ALIASES.get(name, (name,)):
        v = aux.get(key)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            return v
    return None


def zero_loss(ref: Optional[torch.Tensor] = None, device=None) -> torch.Tensor:
    if ref is not None:
        return torch.zeros((), dtype=ref.dtype, device=ref.device)
    return torch.zeros((), device=device)


def _expand_mask(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    m = mask
    while m.dim() < ndim:
        m = m.unsqueeze(-1)
    return m


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over valid steps (x [B,T] or [B,T,...], mask [B,T])."""
    m = _expand_mask(mask, x.dim()).to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


# --------------------------------------------------------------------------- #
# task losses
# --------------------------------------------------------------------------- #
def task_loss(logits: torch.Tensor, target: torch.Tensor, spec: Dict[str, Any],
              mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """One head's loss.  Returns (scalar loss, n_supervised_elements)."""
    kind = str(spec.get("kind", "ce"))
    spec = dict(spec or {})

    if kind == "ce":
        ignore = int(spec.get("ignore", -1))
        tgt = target.long()
        if mask is not None:
            tgt = torch.where(mask, tgt, torch.full_like(tgt, ignore))
        n = int((tgt != ignore).sum())
        if n == 0 or logits.shape[-1] < 2:
            return zero_loss(logits), 0
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1), ignore_index=ignore)
        return (loss if torch.isfinite(loss) else zero_loss(logits)), n

    if kind == "bce":
        tgt = target.float()
        if logits.shape[-1] == 1 and logits.dim() > tgt.dim():
            logits = logits.squeeze(-1)
        ignore = spec.get("ignore", None)
        weight = torch.ones_like(tgt)
        if mask is not None:
            weight = weight * _expand_mask(mask, tgt.dim()).to(tgt.dtype)
        if ignore is not None:
            weight = weight * (tgt != float(ignore)).to(tgt.dtype)
        weight = weight * torch.isfinite(tgt).to(tgt.dtype)
        n = int(weight.sum())
        if n == 0:
            return zero_loss(logits), 0
        el = F.binary_cross_entropy_with_logits(logits.float(), tgt.clamp(0.0, 1.0),
                                                reduction="none")
        loss = (el * weight).sum() / weight.sum().clamp_min(1.0)
        return (loss if torch.isfinite(loss) else zero_loss(logits)), n

    if kind == "mse":
        tgt = target.float()
        ignore = spec.get("ignore", None)
        weight = torch.ones_like(tgt)
        if mask is not None:
            weight = weight * _expand_mask(mask, tgt.dim()).to(tgt.dtype)
        if ignore is not None:
            weight = weight * (tgt != float(ignore)).to(tgt.dtype)
        weight = weight * torch.isfinite(tgt).to(tgt.dtype)
        n = int(weight.sum())
        if n == 0:
            return zero_loss(logits), 0
        el = (logits.float() - tgt) ** 2
        if el.dim() == 2:
            el = el.unsqueeze(-1)
        loss = (el * weight).sum() / weight.sum().clamp_min(1.0)
        return (loss if torch.isfinite(loss) else zero_loss(logits)), n

    raise ValueError(f"unsupported loss kind {kind!r}")


def task_losses(logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor],
                heads_cfg: Dict[str, Dict[str, Any]], mask: Optional[torch.Tensor] = None,
                weights: Optional[Dict[str, float]] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted sum over heads.  CE pads are already ignore-filled by collate."""
    parts: Dict[str, torch.Tensor] = {}
    total: Optional[torch.Tensor] = None
    for name, spec in (heads_cfg or {}).items():
        if name not in logits:
            continue
        target = (labels or {}).get(name)
        if target is None:
            continue
        w = float((weights or {}).get(name, 1.0))
        if w == 0.0:
            continue
        use_mask = None if str(spec.get("kind")) == "ce" else mask
        loss, _n = task_loss(logits[name], target, spec, use_mask)
        parts[name] = loss
        total = loss * w if total is None else total + loss * w
    if total is None:
        ref = next(iter(logits.values())) if logits else None
        total = zero_loss(ref)
    return total, parts


# --------------------------------------------------------------------------- #
# auxiliary objectives (standalone, reusable for probes)
# --------------------------------------------------------------------------- #
def crosstalk_loss(aux: Optional[dict], mask: torch.Tensor,
                   cfg: Optional[dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Isolation: different keys must not share memory slots."""
    cfg = dict(cfg or {})
    route = aux_get(aux, "route")
    if route is None or route.dim() != 4:
        return zero_loss(out if (out := None) else None, device=mask.device), {"available": 0.0}
    R = route.float()
    R = R / R.sum(-1, keepdim=True).clamp_min(1e-6)          # [B,T,K,N]
    K = int(R.shape[-2])
    N = int(R.shape[-1])
    shared = zero_loss(device=R.device)
    if K > 1:
        C = torch.einsum("btkn,btln->btkl", R, R)
        off_sum = C.sum(-1).sum(-1) - C.diagonal(dim1=-2, dim2=-1).sum(-1)
        off = off_sum / float(K * (K - 1))
        shared = masked_mean(off, mask)
    balance = zero_loss(device=R.device)
    if N > 1:
        usage = masked_mean(R.mean(-2), mask)                # [B,N]
        p = usage.mean(0)
        p = p / p.sum().clamp_min(1e-6)
        balance = (p * (p.clamp_min(1e-8).log() + math.log(N))).sum()
    w_shared = float(cfg.get("w_shared", 1.0))
    w_balance = float(cfg.get("w_balance", 0.1))
    loss = w_shared * shared + w_balance * balance
    return loss, {"available": 1.0, "shared": float(shared.detach()),
                  "balance": float(balance.detach())}


def write_budget_loss(aux: Optional[dict], mask: torch.Tensor,
                      cfg: Optional[dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Penalise the mean write gate (fewer, more selective writes)."""
    cfg = dict(cfg or {})
    gate = aux_get(aux, "write_gate")
    if gate is None:
        return zero_loss(device=mask.device), {"available": 0.0}
    g = gate.float()
    if g.dim() == 3:                                        # [B,T,N] per-slot
        g = g.mean(-1)
    if g.dim() != 2:
        return zero_loss(device=mask.device), {"available": 0.0}
    mean_gate = masked_mean(g, mask)
    target = cfg.get("target", None)
    if target is None:
        loss = mean_gate
    else:
        loss = (mean_gate - float(target)) ** 2
    return loss, {"available": 1.0, "gate_mean": float(mean_gate.detach())}


def long_horizon_loss(aux: Optional[dict], mask: torch.Tensor, t: torch.Tensor,
                      cfg: Optional[dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Multi-step memory-retention loss (default horizon h=5 steps).

    For every valid step t the readout is compared with the readout h steps
    later; the pair is weighted by how long it has been since the last real
    write (age/(age+tau)).  Gradients therefore span h >= 2 steps, which is what
    gives the write channel a non-degenerate gradient signal (single-step
    residual losses do not).
    """
    cfg = dict(cfg or {})
    read = aux_get(aux, "read")
    if read is None:
        read = aux_get(aux, "hidden")
    gate = aux_get(aux, "write_gate")
    if read is None or gate is None or t is None:
        return zero_loss(device=mask.device), {"available": 0.0}
    r = read.float()
    g = gate.float()
    if g.dim() == 3:
        g = g.mean(-1)
    if g.dim() != 2 or r.dim() != 3:
        return zero_loss(device=mask.device), {"available": 0.0}
    B, T, D = r.shape
    horizon = max(1, int(cfg.get("horizon", 5)))
    if T <= horizon:
        return zero_loss(device=r.device), {"available": 1.0, "horizon": float(horizon),
                                            "n": 0.0}
    ar = torch.arange(T, device=r.device).unsqueeze(0).expand(B, T)
    thr = float(cfg.get("write_thr", 0.1))
    idx = torch.where(g > thr, ar, torch.zeros_like(ar))
    last = torch.cummax(idx, dim=1).values                         # [B,T]
    t_last = torch.gather(t.float(), 1, last)
    age = (t.float() - t_last).clamp_min(0.0)                      # [B,T]
    tau = float(cfg.get("tau", 1.0) or 1.0)

    r0, r1 = r[:, :-horizon], r[:, horizon:]
    age1 = age[:, horizon:]
    valid = mask[:, :-horizon] & mask[:, horizon:]
    w = age1 / (age1 + tau)
    diff = 1.0 - F.cosine_similarity(r0, r1, dim=-1)
    loss = masked_mean(w * diff, valid)
    if not torch.isfinite(loss):
        loss = zero_loss(device=r.device)
    return loss, {"available": 1.0, "horizon": float(horizon),
                  "n": float(valid.sum()), "age_mean": float(masked_mean(age, mask).detach())}


def time_consistency_loss(model, batch: dict, out: torch.Tensor,
                          cfg: Optional[dict] = None
                          ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Same physical process sampled at two rates must give close representations.

    Needs a second forward pass on a (1/factor)x-downsampled view of the batch.
    Timestamps are preserved so dense step f*k and sparse step k are the same
    physical instant.  Computed on the CORE output (pre-head), with the core in
    whatever dropout mode training uses (cores normally have no dropout).
    """
    cfg = dict(cfg or {})
    factor = max(2, int(cfg.get("factor", 2)))
    kind = str(cfg.get("kind", "cos"))
    mask = batch["mask"]
    if out is None or out.dim() != 3 or out.shape[1] < 2:
        return zero_loss(out, device=mask.device), {"available": 0.0}
    bds = downsample_batch(batch, factor)
    try:
        _logits2, _aux2, out2 = model(bds)
    except Exception:                                     # pragma: no cover - defensive
        return zero_loss(out, device=mask.device), {"available": 0.0}
    src = bds["_src_idx"]
    o_dense = out[:, src]
    both = bds["mask"] & mask[:, src]
    if int(both.sum()) == 0:
        return zero_loss(out, device=mask.device), {"available": 1.0, "diff": 0.0}
    if kind == "mse":
        d = ((o_dense.float() - out2.float()) ** 2).mean(-1)
    else:
        d = 1.0 - F.cosine_similarity(o_dense.float(), out2.float(), dim=-1)
    loss = masked_mean(d, both)
    if not torch.isfinite(loss):
        loss = zero_loss(out, device=mask.device)
    return loss, {"available": 1.0, "diff": float(loss.detach())}


# --------------------------------------------------------------------------- #
# aux statistics for metrics.json / probes
# --------------------------------------------------------------------------- #
def aux_statistics(aux: Optional[dict], mask: torch.Tensor) -> Dict[str, float]:
    """Cheap summary tensors of the core's internal memory state (all optional)."""
    stats: Dict[str, float] = {}
    if not isinstance(aux, dict):
        return stats
    with torch.no_grad():
        read = aux_get(aux, "read")
        if read is not None and read.dim() >= 3:
            norm = read.float().norm(dim=-1)
            stats["mem_norm_mean"] = float(masked_mean(norm, mask))
            nv = mask.sum(1).clamp_min(1) - 1
            last = torch.gather(norm, 1, nv.unsqueeze(-1)).squeeze(-1)
            stats["mem_norm_last"] = float(last.mean())
        hidden = aux_get(aux, "hidden")
        if hidden is not None and hidden.dim() >= 3:
            stats["hidden_norm_mean"] = float(
                masked_mean(hidden.float().norm(dim=-1), mask))
        gate = aux_get(aux, "write_gate")
        if gate is not None:
            g = gate.float()
            if g.dim() == 3:
                stats["write_gate_slot_mean"] = float(masked_mean(g.mean(-1), mask))
                stats["write_gate_max"] = float(g.max())
            elif g.dim() == 2:
                stats["write_gate_mean"] = float(masked_mean(g, mask))
                stats["write_gate_frac_open"] = float(
                    masked_mean((g > 0.5).float(), mask))
        route = aux_get(aux, "route")
        if route is not None and route.dim() == 4:
            p = route.float().clamp_min(1e-8)
            ent = -(p * p.log()).sum(-1)                       # [B,T,K]
            n_slots = int(p.shape[-1])
            stats["route_entropy"] = float(masked_mean(ent.mean(-1), mask))
            stats["route_entropy_norm"] = float(
                stats["route_entropy"] / max(math.log(max(n_slots, 2)), 1e-6))
            stats["route_max_mean"] = float(masked_mean(p.max(-1).values.mean(-1), mask))
        slot_read = aux_get(aux, "slot_read")
        if slot_read is not None and slot_read.dim() == 4:
            stats["slot_norm_mean"] = float(
                masked_mean(slot_read.float().norm(dim=-1).mean(-1), mask))
    return {k: (v if math.isfinite(v) else None) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
# trainer-owned auxiliary objective module (holds the retrieval probe)
# --------------------------------------------------------------------------- #
class AuxObjectives(nn.Module):
    """Computes the enabled auxiliary objectives.

    Holds the (optional) retrieval probe, so it is part of the optimizer and of
    the checkpoint.  When ``cfg["aux"][name]`` is 0 for every probe-dependent
    loss, ``n_params`` is 0 and this module is a pure no-op.
    """

    def __init__(self, loss_cfg: Optional[dict], heads_cfg: Dict[str, Dict[str, Any]],
                 aux_shapes: Optional[Dict[str, Any]] = None):
        super().__init__()
        loss_cfg = dict(loss_cfg or {})
        self.aux_num_scale = float(loss_cfg.get("aux_num_scale", 1.0) or 1.0)
        self.aux_w: Dict[str, float] = {k: float(v) for k, v in
                                        (loss_cfg.get("aux") or {}).items()}
        self.aux_cfg: Dict[str, dict] = dict(loss_cfg.get("aux_cfg") or {})
        self.task_w: Dict[str, float] = dict(loss_cfg.get("task") or {})
        self.heads_cfg = dict(heads_cfg or {})
        shapes = dict(aux_shapes or {})
        self.read_dim = int(shapes.get("read_dim") or 0)
        self.probes = nn.ModuleDict()
        self._build_probes()

    # -- config helpers -------------------------------------------------- #
    def weight(self, name: str) -> float:
        """Effective weight = (per-name aux weight, default 0) * aux_num_scale."""
        return float(self.aux_w.get(name, 0.0)) * float(self.aux_num_scale)

    def active(self) -> Dict[str, float]:
        return {k: self.weight(k) for k in self.aux_w if self.weight(k)}

    def effective_weights(self) -> Dict[str, float]:
        return {k: self.weight(k) for k in AUX_NAMES}

    def cfg_for(self, name: str) -> dict:
        c = dict(self.aux_cfg.get(name) or {})
        return c

    # -- retrieval probe -------------------------------------------------- #
    def _primary_head(self) -> Optional[str]:
        if "primary_head" in self.aux_cfg.get("retrieval", {}):
            return str(self.aux_cfg["retrieval"]["primary_head"])
        for name in self.heads_cfg:
            return name
        return None

    def _build_probes(self) -> None:
        if self.weight("retrieval") == 0.0 or self.read_dim <= 0:
            return
        head = self._primary_head()
        if head is None or head not in self.heads_cfg:
            return
        spec = self.heads_cfg[head]
        kind = str(spec.get("kind", "ce"))
        if kind in ("ce", "bce"):
            out_dim = int(spec.get("n_classes", spec.get("dim", 1)))
            self.probes["retrieval"] = nn.Linear(self.read_dim, out_dim)
        else:
            self.probes["retrieval"] = nn.Linear(self.read_dim, int(spec.get("dim", 1)))
        self._probe_head = head

    def _retrieval(self, aux: Optional[dict], batch: dict
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if "retrieval" not in self.probes:
            ref = batch["mask"]
            return zero_loss(device=ref.device), {"available": 0.0}
        read = aux_get(aux, "read")
        head = getattr(self, "_probe_head", None)
        labels = batch.get("labels") or {}
        target = labels.get(head) if head else None
        if read is None or target is None:
            return zero_loss(device=batch["mask"].device), {"available": 0.0}
        logits = self.probes["retrieval"](read.float())
        spec = self.heads_cfg[head]
        mask = batch["mask"]
        # For an MSE head the dataset may provide a PER-SLOT lab_mask; align the
        # probe's supervision with the evaluator's validity rule (repeat each
        # slot flag over its V state dims) so the probe is not trained on the
        # un-supervised zeros.
        if str(spec.get("kind")) == "mse" and labels.get("mask") is not None:
            lm = labels["mask"]
            D, K = int(target.shape[-1]), int(lm.shape[-1])
            if K > 0 and D % K == 0 and D != K:
                lm = lm.repeat_interleave(D // K, dim=-1)
            m = mask.unsqueeze(-1) & (lm > 0)
            while m.dim() < target.dim():
                m = m.unsqueeze(-1)
            mask = m
        loss, n = task_loss(logits, target, spec, mask)
        return loss, {"available": 1.0, "n": float(n)}

    # -- main entry ------------------------------------------------------- #
    def compute(self, model, batch: dict, logits: Dict[str, torch.Tensor],
                aux: Optional[dict], out: Optional[torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """Enabled aux objectives -> {name: scalar tensor} (0 for unavailable)."""
        mask = batch["mask"]
        parts: Dict[str, torch.Tensor] = {}
        self.last_stats: Dict[str, Dict[str, float]] = {}
        if self.weight("retrieval"):
            loss, st = self._retrieval(aux, batch)
            parts["retrieval"] = loss
            self.last_stats["retrieval"] = st
        if self.weight("crosstalk"):
            loss, st = crosstalk_loss(aux, mask, self.cfg_for("crosstalk"))
            parts["crosstalk"] = loss
            self.last_stats["crosstalk"] = st
        if self.weight("write_budget"):
            loss, st = write_budget_loss(aux, mask, self.cfg_for("write_budget"))
            parts["write_budget"] = loss
            self.last_stats["write_budget"] = st
        if self.weight("long_horizon"):
            loss, st = long_horizon_loss(aux, mask, batch["t"], self.cfg_for("long_horizon"))
            parts["long_horizon"] = loss
            self.last_stats["long_horizon"] = st
        if self.weight("time_consistency"):
            loss, st = time_consistency_loss(model, batch, out,
                                             self.cfg_for("time_consistency"))
            parts["time_consistency"] = loss
            self.last_stats["time_consistency"] = st
        return parts

    def weighted(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for name, val in parts.items():
            w = self.weight(name)
            if w == 0.0 or val is None:
                continue
            total = val * w if total is None else total + val * w
        if total is None:
            ref = next(iter(parts.values())) if parts else None
            total = zero_loss(ref)
        return total
