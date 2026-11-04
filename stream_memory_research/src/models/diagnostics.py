"""Gradient-flow and memory-horizon diagnostics for the temporal cores.

These are *measurement* tools, not models.  They answer the question the whole
project turns on: **how far back in time does the model still carry information,
and is that horizon measured in steps or in seconds?**

Three quantities, all cheap, all working for every core through the shared
contract:

``input_grad_profile``
    ``|| d out_{T-1} / d x_t ||`` for every ``t``.  Because every core is causal
    and ``out_{T-1}`` depends on past observations only through the state, this
    profile *is* the memory kernel of the model (the linearised influence of an
    old observation on today's output).  Normalised by ``profile[0]`` it is a
    decay curve.

``memory_horizon``
    The number of steps back at which that kernel falls below ``frac`` of its
    value at the last step.  Run it at two sampling rates ``dt``:

    * a model whose **state decays in physical time** (dsm / dsm_slot /
      deltanet / cfc-with-timespans) has a horizon in *steps* that scales
      ~1/dt, so the horizon in **seconds** stays roughly constant.  That is the
      falsifiable prediction of the continuous-time hypothesis;
    * a discrete RNN (gru / lstm) has the same horizon in *steps* at any dt, so
      its horizon in seconds shrinks linearly with dt -- the failure mode the
      D-dimension (irregular sampling) experiment is designed to expose;
    * ``mlp`` has a horizon of 1 step by construction.

``param_grad_groups``
    Per-submodule gradient norms for one loss on the last step, so you can see
    whether the slow state, the fast memory or the output head is the bottleneck
    (e.g. a dead write gate shows as a near-zero gradient on its projections).

Caveat: the input-gradient kernel is a local (first-order) property at
initialisation.  It is the right tool for asking "is the *architecture* capable
of a physical-time horizon", not for claiming a trained model's memory.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import build_core, list_models


# --------------------------------------------------------------------------- #
# core measurements
# --------------------------------------------------------------------------- #
def input_grad_profile(core, x: torch.Tensor, dt: torch.Tensor, mask: torch.Tensor,
                       loss_step: int = -1, normalise: bool = True) -> torch.Tensor:
    """``|| d out[loss_step] / d x_t ||`` for all ``t`` -> ``[T]`` (batch-mean).

    ``x`` is used as-is (it is cloned), so the caller can pass a real batch.
    """
    x = x.clone().detach().requires_grad_(True)
    out, _aux = core(x, dt, mask)
    step = loss_step if loss_step >= 0 else out.shape[1] + loss_step
    loss = out[:, step].pow(2).sum()
    (g,) = torch.autograd.grad(loss, x, retain_graph=False)
    prof = g.detach().norm(dim=-1).mean(0)                  # [T]
    if normalise and float(prof.max()) > 0:
        prof = prof / prof.max()
    return prof


def memory_horizon(core, Dg: int, dt_value: float, T: int = 32, frac: float = 0.1,
                   B: int = 4, seed: int = 0, mask: Optional[torch.Tensor] = None,
                   noise_scale: float = 0.02) -> Tuple[int, torch.Tensor]:
    """Steps back at which the memory kernel drops below ``frac`` of its peak.

    A *small* ``noise_scale`` keeps the measurement in the linear regime (the
    gates sit near their sigmoid centres, nothing saturates), which is what makes
    it comparable across models.  The kernel is measured from the last step
    backwards: ``rev[d] = || d out_{T-1} / d x_{T-1-d} ||``, and the horizon is the
    first ``d`` with ``rev[d] < frac`` (``mlp`` -> 1 step, by construction).

    Returns ``(horizon_steps, normalised_profile)``.
    """
    g = torch.Generator().manual_seed(seed)
    x = noise_scale * torch.randn(B, T, Dg, generator=g)
    dt = torch.full((B, T), float(dt_value))
    dt[:, 0] = 0.0
    if mask is None:
        mask = torch.ones(B, T, dtype=torch.bool)
    prof = input_grad_profile(core, x, dt, mask, normalise=True)
    rev = prof.flip(0)                       # rev[d] = influence of t = T-1-d
    below = (rev < frac).nonzero(as_tuple=False)
    h = int(below[0]) if below.numel() else T - 1
    return int(h), prof


def lag_kernel(core, Dg: int, dt_value: float, lags: Sequence[int], T: int = 32,
               B: int = 4, seed: int = 0, noise_scale: float = 0.02) -> Dict[int, float]:
    """Normalised memory kernel at a few lags (steps back from the last step).

    More informative than a single threshold when the state is *multi-timescale*
    (a slow channel makes a threshold metric saturate at T).
    """
    g = torch.Generator().manual_seed(seed)
    x = noise_scale * torch.randn(B, T, Dg, generator=g)
    dt = torch.full((B, T), float(dt_value))
    dt[:, 0] = 0.0
    mask = torch.ones(B, T, dtype=torch.bool)
    prof = input_grad_profile(core, x, dt, mask, normalise=True)
    rev = prof.flip(0)
    return {int(l): (float(rev[l]) if l < T else float("nan")) for l in lags}


def param_grad_groups(core, x: torch.Tensor, dt: torch.Tensor,
                      mask: torch.Tensor) -> Dict[str, float]:
    """L2 norm of the gradient per top-level submodule, for the last-step loss."""
    core.zero_grad(set_to_none=True)
    out, _aux = core(x, dt, mask)
    out[:, -1].pow(2).sum().backward()
    acc: Dict[str, float] = {}
    for name, p in core.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".")[0]
        acc[top] = acc.get(top, 0.0) + float(p.grad.detach().pow(2).sum())
    core.zero_grad(set_to_none=True)
    return {k: math.sqrt(v) for k, v in sorted(acc.items())}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(models: Optional[Sequence[str]] = None, Dg: int = 32, hidden: int = 32,
           T: int = 32, dts: Sequence[float] = (0.02, 0.1), frac: float = 0.1,
           seed: int = 0, lags: Sequence[int] = (1, 2, 4, 8, 16)) -> Dict[str, Dict]:
    """Print and return the horizon-vs-dt table plus the kernel shape per core.

    ``horizon_s`` = ``horizon_steps * dt``.  A time-aware core keeps
    ``horizon_s`` roughly constant as dt changes; a discrete RNN does not.  The
    lag kernel is printed too because a multi-timescale state makes a single
    threshold saturate at ``T``.
    """
    models = list(models or list_models())
    print(f"\n=== memory horizon (grad-kernel < {frac:.0%} of peak), "
          f"Dg={Dg} hidden={hidden} T={T} ===")
    head = f"{'model':10s}{'params':>9s}"
    for dt in dts:
        head += f"{'h@dt=' + str(dt) + ' steps':>22s}{'seconds':>10s}"
    print(head)
    rows: Dict[str, Dict] = {}
    for name in models:
        torch.manual_seed(seed)
        core = build_core(name, Dg, hidden).eval()
        row = {"params": core.n_params, "horizon_steps": {}, "horizon_s": {}}
        line = f"{name:10s}{core.n_params:9,d}"
        for dt in dts:
            h, _ = memory_horizon(core, Dg, dt, T=T, frac=frac, seed=seed)
            row["horizon_steps"][dt] = h
            row["horizon_s"][dt] = h * dt
            line += f"{h:22d}{h * dt:10.3f}"
        rows[name] = row
        print(line)
    print(f"\n=== memory kernel ||d out_last / d x_(last-lag)|| (normalised), "
          f"T={T} ===")
    print(f"{'model':10s}" + "".join(f"{'lag' + str(l):>9s}" for l in lags)
          + "   (per dt)")
    for name in models:
        torch.manual_seed(seed)
        core = build_core(name, Dg, hidden).eval()
        for dt in dts:
            k = lag_kernel(core, Dg, dt, lags, T=T, seed=seed)
            rows[name].setdefault("kernel", {})[dt] = k
            print(f"{name:10s}" + "".join(f"{k[l]:9.3f}" for l in lags) + f"   dt={dt}")
    # the falsifiable statement, per core
    if len(dts) >= 2:
        d0, d1 = float(dts[0]), float(dts[-1])
        print(f"\n  horizon_steps(dt={d0}) / horizon_steps(dt={d1}) "
              f"(physical-time models ~ {d1 / d0:.1f}x, discrete RNN ~ 1.0x):")
        for name in models:
            a = rows[name]["horizon_steps"][d0]
            b = rows[name]["horizon_steps"][d1]
            ratio = (a / b) if b else float("inf")
            hz0, hz1 = rows[name]["horizon_s"][d0], rows[name]["horizon_s"][d1]
            sec_ratio = (hz0 / hz1) if hz1 else float("inf")
            print(f"    {name:10s} steps-ratio {ratio:5.2f}x   "
                  f"seconds-ratio {sec_ratio:5.2f}x   "
                  f"({hz0:.3f}s vs {hz1:.3f}s)")
    return rows


def grad_flow_report(models: Optional[Sequence[str]] = None, Dg: int = 32,
                     hidden: int = 32, T: int = 16, dt_value: float = 0.05,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Per-submodule gradient norms (who actually learns?)."""
    models = list(models or list_models())
    print(f"\n=== gradient norms per submodule (last-step loss), T={T} ===")
    out: Dict[str, Dict[str, float]] = {}
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, T, Dg, generator=g)
    dt = torch.full((4, T), float(dt_value))
    dt[:, 0] = 0
    mask = torch.ones(4, T, dtype=torch.bool)
    for name in models:
        torch.manual_seed(seed)
        core = build_core(name, Dg, hidden)
        groups = param_grad_groups(core, x, dt, mask)
        out[name] = groups
        tot = math.sqrt(sum(v * v for v in groups.values())) or 1.0
        s = "  ".join(f"{k}={v / tot:.2f}" for k, v in groups.items())
        print(f"  {name:10s} |g|={tot:9.2e}   " + s)
    return out


if __name__ == "__main__":
    # T=64 and three sampling rates: with dt=0.02 / 0.1 / 0.5 a physical-time
    # memory keeps the same horizon in *seconds* while its horizon in *steps*
    # grows ~1/dt; a discrete RNN keeps the horizon in *steps* instead.
    report(T=64, dts=(0.02, 0.1, 0.5), frac=0.5)
    grad_flow_report()
