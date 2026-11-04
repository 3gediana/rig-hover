"""Shared building blocks for the temporal-core zoo (``src/models/``).

Every core in this package obeys the same contract (DESIGN.md §2.1)::

    forward(x [B,T,Dg], dt [B,T] s, mask [B,T] bool, tok=None, ...) -> (out [B,T,H], aux dict)

The pieces here are the ones that must be **identical** across cores so that a
comparison between cores isolates the temporal operator and nothing else:

``InputNorm``    identical input path: zero the padded features, optionally append
                 ``log1p(dt)`` and the validity flag, optional LayerNorm.
``safe_dt``      physical dt: NaN/inf-free, never negative, exactly 0 at padded
                 steps (the contract says ``dt[:, 0] == 0``).
``dt_struct``    the ``use_dt_struct`` ablation: replace the dt that *drives the
                 dynamics* by a constant, leaving the feature channel untouched.
``TimeDecay``    ``exp(-dt/tau)`` with per-dimension softplus ``tau``.  The
                 exponential is what makes state decay an exact semigroup in
                 physical time: ``decay(dt1)·decay(dt2) == decay(dt1+dt2)``.
``GroupedDecay`` multi-timescale decay over the value dimension (G groups with
                 rates ~[8, 2, 0.5, 0.02] /s).  One group is nearly non-decaying
                 on purpose: a single rate cannot both suppress cross-talk and
                 survive a multi-second occlusion (LIT-BRIEF).
``LoopCore``     base class for cores whose forward is a per-step recurrence.  It
                 implements ``forward`` and ``step`` from the same ``_step`` code,
                 so single-frame inference (robot interface / latency) is
                 numerically identical to a full-sequence pass.

Mask semantics everywhere: ``mask=False`` steps must not change state.  Loops use
``state = apply_mask(state_new, state_old, mask[:, t])``; linear scans force the
per-step decay to 1 and the additive update to 0, which makes the step the
identity.  Hidden states stay fp32 even under fp16 autocast (only matmuls are
cast), which is what keeps the recurrent paths NaN-free.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.contract import TemporalCore, apply_mask, count_params

__all__ = [
    "InputNorm", "safe_dt", "dt_struct", "scale_dt", "masked_input", "inv_softplus",
    "TimeDecay", "GroupedDecay", "LoopCore", "hard_st", "hard_mask_st",
    "unpack_cfg", "cp", "DECAY_RATES", "COMMON_SPEC",
]

#: default multi-timescale decay rates in 1/s (tau = 0.125 s ... 50 s).
#: The last one is deliberately ~no-decay: it is the channel that carries
#: information across long occlusions.
DECAY_RATES = (8.0, 2.0, 0.5, 0.02)

#: cfg keys every core honours (``build_core`` passes ``**cfg`` to all of them so a
#: single ablation suite can drive the whole zoo).
COMMON_SPEC = {
    "use_dt_feature": True,    # append log1p(dt) to the input features
    "use_mask_feature": True,  # append the validity flag as an input feature
    "input_norm": True,        # LayerNorm on the input
    "use_dt_struct": True,     # dt drives the dynamics (False -> constant dt_const)
    "dt_const": 0.1,           # seconds, used when use_dt_struct=False
    # Timespan unit normalisation: dt_used = dt * dt_scale, applied identically to
    # every core right after the mask/NaN sanitising and before any use.
    # dt_scale=1.0 is bit-identical to no scaling.  Needed because CfC's gate is
    # t = sigmoid(ta.z * ts + tb.z) -- the timespan multiplies *inside* a sigmoid
    # and ncps is normally used with ts ~ O(1), so with dt ~ 0.1 s CfC's time
    # channel is ~10x too weak and degenerates to sigmoid(tb.z), a plain discrete
    # RNN.  Scaling is uniform across cores so no model gets special treatment;
    # ours is expected to prefer 1.0 (its rates are initialised in 1/s, i.e.
    # already physically scaled).
    "dt_scale": 1.0,
}


def scale_dt(dt: torch.Tensor, dt_scale: float) -> torch.Tensor:
    """Apply the ``dt_scale`` timespan-unit normalisation (no-op at 1.0)."""
    s = float(dt_scale)
    return dt if s == 1.0 else dt * s


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def inv_softplus(y):
    """x such that ``softplus(x) == y`` (works for floats and tensors)."""
    if torch.is_tensor(y):
        return torch.log(torch.expm1(y.clamp_min(1e-12)))
    y = float(y)
    if y <= 0:
        return -30.0
    if y > 20.0:                      # softplus(x) ~= x for large x
        return y
    return math.log(math.expm1(y))


def cp(x):
    """Detach to CPU -- for cheap bookkeeping / logging."""
    return x.detach().to("cpu") if torch.is_tensor(x) else x


def masked_input(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero the input at padded steps.

    A *safety* measure as much as a semantic one: whatever garbage sits in a
    padded slot of ``x`` (including NaN/inf) can then never reach a gate and
    therefore can never contaminate a real step.
    """
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))


def safe_dt(dt: torch.Tensor, mask: torch.Tensor, dt_max: float = 0.0) -> torch.Tensor:
    """Sanitised physical dt: float32, in ``[0, dt_max]``, finite, 0 where padded."""
    dt = torch.nan_to_num(dt.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if dt_max and dt_max > 0:
        dt = dt.clamp_max(float(dt_max))
    return dt * mask.to(dt.dtype)


def dt_struct(dt: torch.Tensor, mask: torch.Tensor, use_dt_struct: bool = True,
              dt_const: float = 0.1) -> torch.Tensor:
    """The dt that drives the *dynamics*.

    ``use_dt_struct=True``  -> the real physical dt.
    ``use_dt_struct=False`` -> every real step behaves as if it lasted
    ``dt_const`` seconds (the "Δt does not act structurally" ablation).  The
    feature channel ``log1p(dt)`` is unaffected -- only the operator sees this.
    """
    if use_dt_struct:
        return dt
    return torch.full_like(dt, float(dt_const)) * mask.to(dt.dtype)


def hard_st(p: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    """Straight-through binarisation of a *probability*: forward 0/1, backward identity."""
    hard = (p > thresh).to(p.dtype)
    return p + (hard - p).detach()


def hard_mask_st(x: torch.Tensor, thresh: float, temp: float = 0.1) -> torch.Tensor:
    """Straight-through binary mask of a *continuous* quantity: forward ``1[x > thresh]``,
    backward through a sigmoid surrogate ``sigmoid((x - thresh)/temp)``.

    Used for "only change what changed": ``u_j = 1[||v - val_j|| > delta]``.  The
    forward value is exactly 0/1 so an unselected slot receives an *exactly zero*
    additive update (bit-identical state, unit-tested).
    """
    soft = torch.sigmoid((x - thresh) / max(temp, 1e-6))
    hard = (x > thresh).to(x.dtype)
    return hard + (soft - soft.detach())


def unpack_cfg(cfg: dict, spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split ``cfg`` into the keys this core uses and the ones it accepts-but-ignores.

    Every core accepts the full suite config (``**cfg``) so that one ablation
    file can drive every model, but it must be *explicit* about which flags it
    actually honoured -- that is what ``cfg_effective`` records into metrics.json.
    """
    c = dict(cfg)
    vals = {k: c.pop(k, d) for k, d in spec.items()}
    return vals, c


# --------------------------------------------------------------------------- #
# input path (identical for every core)
# --------------------------------------------------------------------------- #
class InputNorm(nn.Module):
    """Common input stage.

    ``mask``  -> zero the padded features (NaN/garbage safe)
    ``use_dt_feature=True``  -> append ``log1p(dt)`` (bounded for long gaps,
        ``log1p(0)==0`` at padded steps)
    ``use_mask_feature=True`` -> append the validity flag as a feature
    ``input_norm=True``       -> LayerNorm over the **feature** dimension only.

    LayerNorm is never taken over time or batch: that would leak the future and
    would normalise dt across a batch (DESIGN §2.1 requirement 3).
    """

    def __init__(self, in_dim: int, use_dt_feature: bool = True,
                 use_mask_feature: bool = True, input_norm: bool = True):
        super().__init__()
        self.use_dt_feature = bool(use_dt_feature)
        self.use_mask_feature = bool(use_mask_feature)
        self.in_dim = int(in_dim)
        extra = (1 if self.use_dt_feature else 0) + (1 if self.use_mask_feature else 0)
        self.d = self.in_dim + extra
        self.norm = nn.LayerNorm(self.d) if input_norm else nn.Identity()

    def forward(self, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = masked_input(x, mask)
        if self.use_dt_feature or self.use_mask_feature:
            feats = []
            if self.use_dt_feature:
                feats.append(torch.log1p(dt).unsqueeze(-1))
            if self.use_mask_feature:
                feats.append(mask.to(dt.dtype).unsqueeze(-1))
            x = torch.cat([x] + feats, dim=-1)
        return self.norm(x)

    def extra_repr(self) -> str:
        return (f"in_dim={self.in_dim}, d={self.d}, dt_feature={self.use_dt_feature}, "
                f"mask_feature={self.use_mask_feature}, norm={type(self.norm).__name__}")


# --------------------------------------------------------------------------- #
# time decay
# --------------------------------------------------------------------------- #
class TimeDecay(nn.Module):
    """Per-dimension exponential decay ``exp(-dt/tau)``, ``tau = softplus(raw)+eps``.

    ``exp`` is the only functional form with the semigroup property in continuous
    time -- ``exp(-dt1/tau)·exp(-dt2/tau) = exp(-(dt1+dt2)/tau)`` -- so observing
    the same process at 10 Hz or 30 Hz yields the *same* total decay.  That is the
    D-dimension ("irregular sampling robustness") argument of the project and it
    is unit-tested to 1e-6.
    """

    def __init__(self, shape, init_tau=4.0, learnable: bool = True, tau_min: float = 1e-3):
        super().__init__()
        shape = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        if len(shape) == 0:
            shape = (1,)
        self.shape = shape
        self.tau_min = float(tau_min)
        if torch.is_tensor(init_tau):
            raw = inv_softplus(init_tau.reshape(shape).clamp_min(self.tau_min + 1e-6)
                               - self.tau_min)
        else:
            raw = torch.full(shape, inv_softplus(max(float(init_tau) - self.tau_min, 1e-3)))
        self.raw = nn.Parameter(raw, requires_grad=bool(learnable))

    def tau(self) -> torch.Tensor:
        return F.softplus(self.raw) + self.tau_min

    @property
    def is_scalar(self) -> bool:
        return len(self.shape) == 1 and self.shape[0] == 1

    def step_factor(self, dt_t: torch.Tensor) -> torch.Tensor:
        """One step: ``dt_t [B] -> factor [B, *shape]``."""
        tau = self.tau().flatten()                              # [n]
        return torch.exp(-(dt_t.float().reshape(-1, 1) / tau)).reshape(-1, *self.shape)

    def factor(self, dt: torch.Tensor) -> torch.Tensor:
        """Whole sequence: ``dt [B,T] -> factor [B,T,*shape]``."""
        return self.step_factor(dt.reshape(-1)).reshape(*dt.shape, *self.shape)

    def forward(self, y: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return y * self.factor(dt)

    def extra_repr(self) -> str:
        return f"shape={self.shape}, tau~{float(self.tau().mean()):.4f}s"


class GroupedDecay(nn.Module):
    """Multi-timescale decay: ``G`` rates, one per block of the value dimension.

    ``Lambda[channel] = softplus(raw[group(channel)])``  in **1/s**, anchored at
    physically meaningful initial rates (``rates``, default ``DECAY_RATES`` =
    [8, 2, 0.5, 0.02] /s, i.e. tau = 0.125 s ... 50 s).

    Why grouped and anchored: a *single* learned rate cannot simultaneously
    (a) forget fast enough to suppress cross-talk between objects and (b) keep a
    value alive across a multi-second occlusion.  And a free multiplicative scale
    on ``Lambda*dt`` would let the optimiser cancel any difference in ``dt``,
    which would silently destroy the cross-sampling-rate experiment, so the rate
    is parameterised in physical units only.

    One group is nearly non-decaying (tau = 50 s) -- this is the long-term
    channel; ``decay=False`` sets ``Lambda == 0`` exactly.
    """

    def __init__(self, n_groups: int = 4, out_dim: int = 64,
                 rates: Sequence[float] = DECAY_RATES, learnable: bool = True,
                 decay: bool = True):
        super().__init__()
        rates = list(rates)
        n_groups = max(1, min(int(n_groups), len(rates)))
        self.n_groups = n_groups
        self.out_dim = int(out_dim)
        self.decay = bool(decay)
        self.rates = tuple(float(r) for r in rates[:n_groups])
        raw = inv_softplus(torch.tensor(self.rates, dtype=torch.float32))
        self.raw = nn.Parameter(raw, requires_grad=bool(learnable))
        # contiguous blocks of the value dim -> groups (8/2/0.5/0.02 s^-1)
        g = (torch.arange(self.out_dim) * n_groups) // max(1, self.out_dim)
        self.register_buffer("group_of", g.clamp_max(n_groups - 1), persistent=False)

    def lam(self) -> torch.Tensor:
        """Per-group rates ``[G]`` in 1/s (exactly zero when ``decay=False``)."""
        if not self.decay:
            return torch.zeros(self.n_groups, device=self.raw.device)
        return F.softplus(self.raw)

    def lam_dim(self) -> torch.Tensor:
        """Per-channel rates ``[out_dim]`` in 1/s."""
        return self.lam()[self.group_of]

    def step_factor(self, dt_t: torch.Tensor) -> torch.Tensor:
        """One step: ``dt_t [B] -> exp(-Lambda*dt) [B, out_dim]`` (<=1)."""
        return torch.exp(-dt_t.float().reshape(-1, 1) * self.lam_dim().reshape(1, -1))

    def stats(self) -> Dict[str, float]:
        """Per-group mean rate and tau, for logging."""
        lam = self.lam().detach()
        out = {}
        for i in range(self.n_groups):
            out[f"g{i}_lam"] = float(lam[i])
        return out

    def extra_repr(self) -> str:
        return f"n_groups={self.n_groups}, out_dim={self.out_dim}, decay={self.decay}"


# --------------------------------------------------------------------------- #
# loop base class
# --------------------------------------------------------------------------- #
class LoopCore(TemporalCore):
    """Base class for cores expressed as ``h_t = f(x_t, dt_t, h_{t-1})``.

    Subclasses implement::

        init_state(B, device)                 -> dict of state tensors
        _prep(x, dt, dts, mask)               -> ctx   (all work batched over T)
        _step(t, ctx, state)                  -> (out_t [B,H], new_state dict, aux_t dict)

    where ``dt`` is the physical (sanitised) dt used for the *feature channel* and
    ``dts`` is the dt that drives the dynamics (differs only under
    ``use_dt_struct=False``).

    ``forward`` runs ``_step`` over T and ``step`` runs it once from a user state;
    both share ``_step``, so streaming inference cannot drift from the batched
    path.  ``_prep`` must hoist every projection that does not depend on the state
    out of the loop -- that is what makes the per-step cost small.
    """

    H: int = 0            # output / hidden width
    out_dim: int = 0
    use_dt_struct: bool = True
    dt_const: float = 0.1
    dt_scale: float = 1.0         # timespan-unit normalisation (dt_used = dt*scale)
    dt_clamp_max: float = 0.0     # 0 = no clamp
    state_as_out: bool = False    # out_t IS the state -> aux["hidden"] aliases out
    cfg_effective: dict
    cfg_unused: dict

    # ---- subclasses must implement ---- #
    def init_state(self, B: int, device=None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _prep(self, x, dt, dts, mask, tok=None, tok_valid=None) -> Any:
        raise NotImplementedError

    def _step(self, t: int, ctx: Any, state: Dict[str, torch.Tensor]):
        raise NotImplementedError

    # ---- helpers ---- #
    def _dts(self, dt, mask):
        d = dt.clamp_max(self.dt_clamp_max) if self.dt_clamp_max > 0 else dt
        return dt_struct(d, mask, self.use_dt_struct, self.dt_const)

    def _run(self, ctx, state, T: int):
        outs: List[torch.Tensor] = []
        aux: Dict[str, List[torch.Tensor]] = {}
        for t in range(T):
            o, state, a = self._step(t, ctx, state)
            outs.append(o)
            for k, v in a.items():
                aux.setdefault(k, []).append(v)
        return outs, state, aux

    # ---- forward ---- #
    def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None, roi_valid=None,
                roi_meta=None):
        B, T, _ = x.shape
        dt = scale_dt(safe_dt(dt, mask), self.dt_scale)
        if T == 0:
            return x.new_zeros(B, 0, self.out_dim), {}
        dts = self._dts(dt, mask)
        ctx = self._prep(x, dt, dts, mask, tok=tok, tok_valid=tok_valid)
        state = self.init_state(B, x.device)
        outs, state, aux = self._run(ctx, state, T)
        out = torch.stack(outs, dim=1)
        aux = {k: torch.stack(v, dim=1) for k, v in aux.items()}
        if self.state_as_out and "hidden" not in aux:
            aux["hidden"] = out                       # free alias (no copy)
        self._alias_aux(aux)
        aux.update(self._final_aux(state))
        return out, aux

    def _alias_aux(self, aux: Dict[str, torch.Tensor]) -> None:
        """Add the name aliases agreed with trainer-eval (same tensors, no copy)."""
        for src, *dsts in (("hidden", "h", "slow", "h_traj"), ("read", "mem_read")):
            if src in aux:
                for d in dsts:
                    aux.setdefault(d, aux[src])

    def _final_aux(self, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extra aux entries derived from the final state (default: none)."""
        return {}

    # ---- single-frame / streaming path ---- #
    def step(self, x_t, dt_t, mask_t=None, state=None, tok_t=None, tok_valid_t=None,
             roi_t=None, roi_valid_t=None, roi_meta_t=None):
        """One frame -- the streaming / robot-interface entry point.

        Signature agreed with ``trainer-eval``::

            out_t [B,H], aux_t dict, state = core.step(x_t, dt_t, mask_t, state)

        ``x_t [B,Dg]``, ``dt_t [B]`` seconds, ``mask_t [B]`` bool (``None`` -> all
        valid), ``state`` from a previous call (``None`` -> ``init_state``).

        Token shapes are accepted in **both** forms and normalised here (the
        trainer's factory passes the batched form, ``predict.py`` passes the
        single-step form): ``tok_t`` is ``[B,K,Dt]`` or ``[B,1,K,Dt]``,
        ``tok_valid_t`` is ``[B,K]`` or ``[B,1,K]``.  ``roi_*`` are accepted for
        interface compatibility with ``forward`` and ignored by every core.
        """
        B = x_t.shape[0]
        if state is None:
            state = self.init_state(B, x_t.device)
        if mask_t is None:
            mask_t = torch.ones(B, dtype=torch.bool, device=x_t.device)
        mask_t = mask_t.to(torch.bool).reshape(B)
        dt2 = scale_dt(safe_dt(dt_t.reshape(B, 1), mask_t.unsqueeze(1)), self.dt_scale)
        dts2 = self._dts(dt2, mask_t.unsqueeze(1))
        # normalise the token shapes: accept [B,K,Dt] and [B,1,K,Dt]
        tok2 = tok_t
        if tok2 is not None and tok2.dim() == 3:
            tok2 = tok2.unsqueeze(1)                       # [B,K,Dt] -> [B,1,K,Dt]
        tv2 = tok_valid_t
        if tv2 is not None and tv2.dim() == 2:
            tv2 = tv2.unsqueeze(1)                         # [B,K] -> [B,1,K]
        ctx = self._prep(x_t.unsqueeze(1), dt2, dts2, mask_t.unsqueeze(1),
                         tok=tok2, tok_valid=tv2)
        o, state, a = self._step(0, ctx, state)
        aux = {k: v.unsqueeze(1) for k, v in a.items()}
        if self.state_as_out:
            aux.setdefault("hidden", o.unsqueeze(1))
        self._alias_aux(aux)
        aux.update(self._final_aux(state))
        return o, aux, state

    def forward_stepwise(self, x, dt, mask):
        """Reference implementation: ``forward`` via repeated ``step`` calls.

        The tests use it to prove the streaming path and the batched path agree.
        """
        B, T, _ = x.shape
        dt = safe_dt(dt, mask)
        state = self.init_state(B, x.device)
        outs = []
        for t in range(T):
            o, _aux, state = self.step(x[:, t], dt[:, t], mask[:, t], state)
            outs.append(o)
        return torch.stack(outs, dim=1), state
