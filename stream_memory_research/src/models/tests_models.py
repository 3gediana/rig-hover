"""Test-suite for the temporal-core zoo.

Run it with the project's interpreter from the project root::

    python -m src.models.tests_models              # everything (CPU)
    python -m src.models.tests_models --quick      # skip the latency grid
    python -m src.models.tests_models --no-gpu     # skip the fp16-on-GPU check

What it proves (each item is a separate test, all must pass):

a. ``causality``      -- perturbing ``x``/``dt`` at ``t >= k`` leaves ``out[:, :k]``
                         bit-identical (all 9 cores).
b. ``mask``           -- (i) garbage/NaN in padded slots does not change any output,
                         (ii) appending pads does not change earlier outputs,
                         (iii) a masked gap is equivalent to the truncated sequence.
c. ``dt sensitivity`` -- zeroing dt changes the output (dt really enters the maths),
                         and ``use_dt_feature=False, use_dt_struct=False`` makes every
                         core *exactly* dt-invariant (the ablation flags are wired).
d. ``semigroup``      -- the decay layer (Phi) satisfies
                         ``Phi(dt1) . Phi(dt2) == Phi(dt1+dt2)`` to 1e-6, for the
                         per-dim, grouped and slow-state decay operators, printed as
                         an error-vs-dt curve.
e. ``gradients``      -- finite gradients and 10 successful Adam steps on random data.
f. ``matched``        -- ``build_core_matched`` lands within 1.25x of the target.
g. ``mamba scan``     -- the chunked parallel scan equals the naive per-step loop
                         to 1e-5 (including a multi-chunk sequence).
h. ``streaming``      -- ``forward`` equals a loop of ``step`` calls (chunked /
                         single-frame inference is not a second implementation).
i. ``slot bit-exact`` -- with hard routing, a slot that is not routed is bit-identical
                         before/after the step (``torch.equal``).
j. ``fp16``           -- no NaN/inf under CPU bf16 (and CUDA fp16 if a GPU is free).

It also prints the parameter table (hidden x d_in) and the CPU latency table that
the Lead needs to fix hidden dims for parameter matching.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
import traceback
from typing import Dict, List

import torch

from . import build_core, build_core_matched, list_models, param_count
from .base import GroupedDecay, TimeDecay

MODELS = list_models()

# ---------------------------------------------------------------- helpers --- #
TESTS: List[tuple] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def make_batch(B: int, T: int, Dg: int, seed: int = 0, period: float = 0.033,
               jitter: float = 0.35):
    """Random stream: irregular physical dt, dt[:, 0] == 0, all steps valid."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, Dg, generator=g)
    dt = torch.full((B, T), float(period))
    dt = dt * torch.exp(jitter * torch.randn(B, T, generator=g))
    dt = dt.clamp(1e-3, 2.0)
    dt[:, 0] = 0.0
    mask = torch.ones(B, T, dtype=torch.bool)
    return x, dt, mask


def stream_forward(core, x, dt, mask):
    """Run a core frame by frame through the streaming API."""
    B, T, _ = x.shape
    state = core.init_state(B, x.device)
    outs = []
    for t in range(T):
        o, _aux, state = core.step(x[:, t], dt[:, t], mask[:, t], state)
        outs.append(o)
    return torch.stack(outs, dim=1)


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        return float((a.float() - b.float()).abs().max())


# ------------------------------------------------------------------ tests --- #
@test
def test_registry_and_shapes():
    """Every registered core: forward contract, attributes, cfg bookkeeping."""
    B, T, Dg, H = 3, 9, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    mask[:, 7] = False
    for name in MODELS:
        core = build_core(name=name, in_dim=Dg, hidden=H)
        out, aux = core(x, dt, mask)
        assert tuple(out.shape) == (B, T, H), (name, out.shape)
        assert out.dtype == torch.float32, (name, out.dtype)
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        assert out.requires_grad, f"{name}: output is not differentiable"
        assert isinstance(aux, dict)
        for k, v in aux.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v.float()).all(), f"{name}: aux[{k}] not finite"
        assert core.name == name or name in getattr(core, "name", ""), name
        assert core.out_dim == H, (name, core.out_dim)
        assert core.n_params > 0
        assert isinstance(core.cfg_effective, dict) and core.cfg_effective
        assert isinstance(core.cfg_unused, list)
        # init_state / step / tok+roi arguments accepted and ignored
        st = core.init_state(B, x.device)
        o1, a1, st = core.step(x[:, 0], dt[:, 0], mask[:, 0], st)
        assert tuple(o1.shape) == (B, H), (name, o1.shape)
        core(x, dt, mask, tok=torch.randn(B, T, 3, 5), roi=torch.randn(B, T, 2, 4),
             roi_valid=torch.ones(B, T, 2, dtype=torch.bool))
        # empty sequence stays shape-safe
        out0, _ = core(x[:, :0], dt[:, :0], mask[:, :0])
        assert tuple(out0.shape) == (B, 0, H), (name, out0.shape)
    print(f"    {len(MODELS)} cores: forward/step/init_state/aux/cfg OK")


@test
def test_causality():
    """out[:, :k] must not depend on steps >= k (x AND dt perturbed)."""
    B, T, Dg, H = 3, 12, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    for name in MODELS:
        core = build_core(name, Dg, H)
        out, _ = core(x, dt, mask)
        for k in (1, T // 2, T - 1):
            g = torch.Generator().manual_seed(1000 + k)
            x2 = x.clone()
            d2 = dt.clone()
            x2[:, k:] += 3.0 * torch.randn(B, T - k, Dg, generator=g)
            d2[:, k:] = d2[:, k:] * (1.0 + torch.rand(B, T - k, generator=g))
            out2, _ = core(x2, d2, mask)
            if maxdiff(out[:, :k], out2[:, :k]) != 0.0:      # exact, not just close
                raise AssertionError(
                    f"{name}: future perturbation leaked into out[:, :{k}] "
                    f"(maxdiff {maxdiff(out[:, :k], out2[:, :k]):.3e})")
    print("    exact causality (bit-identical prefix) for all cores")


@test
def test_mask():
    """Padded steps must be inert: garbage there, or removing them, changes nothing."""
    B, T, Dg, H = 3, 12, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    mask[:, 6:] = False                      # trailing pads
    d = dt * mask
    for name in MODELS:
        core = build_core(name, Dg, H)
        out, _ = core(x, dt * mask, mask)
        # (i) garbage (including NaN) in the padded slots
        xg = x.clone()
        xg[:, 6:] = float("nan")
        outg, _ = core(xg, dt * mask, mask)
        if maxdiff(out[:, :6], outg[:, :6]) != 0.0:
            raise AssertionError(f"{name}: padded garbage changed real outputs")
        # (ii) appending pads must not change *earlier* outputs
        mask2 = torch.ones(B, T, dtype=torch.bool)
        out2, _ = core(x, d, mask2)
        if maxdiff(out[:, :6], out2[:, :6]) > 1e-6:
            raise AssertionError(f"{name}: appending pads changed earlier outputs "
                                 f"({maxdiff(out[:, :6], out2[:, :6]):.3e})")
        # (iii) a masked gap == the truncated sequence
        k = 6
        out3, _ = core(x[:, :k], d[:, :k], mask[:, :k])
        if maxdiff(out[:, :k], out3) > 1e-6:
            raise AssertionError(f"{name}: masked gap != truncated sequence "
                                 f"({maxdiff(out[:, :k], out3):.3e})")
    print("    padding inert (garbage/NaN at pads, truncation equivalence)")


@test
def test_dt_sensitivity_and_ablation():
    """dt must matter; and with both dt flags off the core must be dt-invariant."""
    B, T, Dg, H = 3, 10, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    for name in MODELS:
        core = build_core(name, Dg, H)
        out, _ = core(x, dt, mask)
        out_z, _ = core(x, torch.zeros_like(dt), mask)
        d1 = maxdiff(out, out_z)
        assert d1 > 1e-6, f"{name}: zeroing dt did not change the output ({d1:.2e})"
        # ablation: no dt feature, no dt structure -> the core cannot see dt at all
        core_ab = build_core(name, Dg, H, use_dt_feature=False, use_dt_struct=False)
        a, _ = core_ab(x, dt, mask)
        b, _ = core_ab(x, dt * 3.0, mask)
        d2 = maxdiff(a, b)
        assert d2 == 0.0, (f"{name}: with use_dt_feature=False/use_dt_struct=False "
                           f"the output still depends on dt ({d2:.3e})")
    print("    dt sensitivity > 1e-6; dt-blind ablation exactly invariant")


@test
def test_decay_semigroup():
    """Phi(dt1) . Phi(dt2) == Phi(dt1 + dt2) to 1e-6, printed as a curve."""
    torch.manual_seed(0)
    print("      dt[s]    per-dim TimeDecay    grouped GroupedDecay")
    worst = 0.0
    td = TimeDecay(32, init_tau=1.0)
    gd = GroupedDecay(4, 8)                         # 8 decay groups over d_v = 8
    y = torch.randn(4, 32)
    M = torch.randn(4, 8, 32)                       # matrix memory: rows = value dim
    f_td = lambda d: td.step_factor(torch.full((4,), float(d)))          # [4,32]
    f_gd = lambda d: gd.step_factor(torch.full((4,), float(d)))          # [4,8]
    for dt in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        dt1, dt2 = 0.5 * dt, dt - 0.5 * dt
        e1 = maxdiff((f_td(dt1) * f_td(dt2)) * y, f_td(dt) * y)
        M1 = (M * f_gd(dt1).unsqueeze(-1)) * f_gd(dt2).unsqueeze(-1)
        e2 = maxdiff(M1, M * f_gd(dt).unsqueeze(-1))
        worst = max(worst, e1, e2)
        print(f"      {dt:5.2f}    {e1:18.3e}    {e2:18.3e}")
    assert worst < 1e-6, f"semigroup violated: {worst:.3e}"
    # the slow state's decay layer, exactly as DSM uses it
    from .dsm import SlowState
    slow = SlowState(32, 0.2, 2.0)
    h = torch.randn(4, 32)
    dt = torch.full((4,), 0.4)
    h1 = h * slow.dec.step_factor(dt * 0.5)
    h2 = h1 * slow.dec.step_factor(dt * 0.5)
    h3 = h * slow.dec.step_factor(dt)
    err = maxdiff(h2, h3)
    print(f"      SlowState Phi(0.2).Phi(0.2) vs Phi(0.4): {err:.3e}")
    assert err < 1e-6, f"slow-state semigroup violated: {err:.3e}"
    print(f"    worst semigroup error {worst:.2e} (< 1e-6)")


@test
def test_zero_dt_safe():
    """dt == 0 everywhere must be finite, and repeated frames must not blow up."""
    B, T, Dg, H = 2, 10, 16, 24
    x, _, mask = make_batch(B, T, Dg)
    for name in MODELS:
        core = build_core(name, Dg, H)
        out, aux = core(x, torch.zeros(B, T), mask)
        assert torch.isfinite(out).all(), f"{name}: NaN/inf with dt == 0"
        # many identical observations at dt = 0 with a soft mask: the state is held
        m2 = torch.ones(B, T, dtype=torch.bool)
        m2[:, 5] = False
        out2, _ = core(x, torch.zeros(B, T), m2)
        assert torch.isfinite(out2).all(), f"{name}: NaN/inf with masked zero-dt"
    print("    dt == 0 (and masked) is finite everywhere")


@test
def test_gradients_and_adam():
    """Finite gradients and 10 successful Adam steps for every core."""
    torch.manual_seed(0)
    B, T, Dg, H = 4, 12, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    target = torch.randn(B, T, H)
    for name in MODELS:
        core = build_core(name, Dg, H)
        opt = torch.optim.Adam(core.parameters(), lr=3e-3)
        losses = []
        for step in range(10):
            opt.zero_grad()
            out, _ = core(x, dt, mask)
            loss = (out - target).pow(2).mean()
            loss.backward()
            grads = [p.grad for p in core.parameters() if p.grad is not None]
            assert grads, f"{name}: no gradients at all"
            assert all(torch.isfinite(g).all() for g in grads), f"{name}: non-finite grad"
            opt.step()
            losses.append(float(loss.detach()))
        assert all(torch.isfinite(p).all() for p in core.parameters()), \
            f"{name}: parameters went non-finite"
        assert losses[-1] < losses[0], f"{name}: Adam did not reduce the loss " \
                                       f"({losses[0]:.4f} -> {losses[-1]:.4f})"
        print(f"    {name:9s} loss {losses[0]:.4f} -> {losses[-1]:.4f}")


@test
def test_matched_params():
    """build_core_matched lands inside 1.25x, and never silently misses."""
    for d_in, target in ((64, 150_000), (768, 300_000)):
        for name in MODELS:
            core = build_core_matched(name, in_dim=d_in, target_params=target)
            r = core.matched_params / target
            assert 1 / 1.25 <= r <= 1.25, (name, d_in, target, core.matched_params)
            print(f"    {name:9s} d_in={d_in:4d} target={target:7d} "
                  f"hidden={core.matched_hidden:4d} params={core.matched_params:8d} "
                  f"ratio={r:.3f}")


@test
def test_dt_scale():
    """`dt_scale` is the uniform timespan-unit normalisation (dt_used = dt*scale).

    It exists so that CfC's timespan channel can be put on the scale ncps is
    normally used at: its gate is ``t = sigmoid(ta.z * ts + tb.z)``, the timespan
    multiplies *inside* a sigmoid, so with dt ~ 0.1 s the CfC time channel is
    ~10x too weak and degenerates to ``sigmoid(tb.z)`` (a plain discrete RNN).
    The flag is applied identically to every core, so it is a fair, tunable
    hyperparameter rather than a per-model hack.

    Checks: (a) ``dt_scale=1.0`` is bit-identical, (b) ``dt_scale=S`` is exactly
    the same computation as feeding ``dt*S``, (c) it changes every core's output,
    (d) it is recorded in ``cfg_effective``.
    """
    import copy
    B, T, Dg, H = 3, 10, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    for name in MODELS:
        base = build_core(name, Dg, H).eval()
        o1, _ = base(x, dt, mask)
        # (a) same weights, dt_scale = 1.0 -> bit-identical (no-op path)
        c1 = copy.deepcopy(base)
        c1.dt_scale = 1.0
        o1b, _ = c1(x, dt, mask)
        assert maxdiff(o1, o1b) == 0.0, f"{name}: dt_scale=1.0 changed the output"
        # (c) dt_scale = 10 must change the output of every core
        c10 = copy.deepcopy(base)
        c10.dt_scale = 10.0
        o10, _ = c10(x, dt, mask)
        d10 = maxdiff(o1, o10)
        assert d10 > 1e-6, f"{name}: dt_scale had no effect ({d10:.2e})"
        # (b) semantics: dt_scale=10 == feeding dt*10 with dt_scale=1
        o_alt, _ = c1(x, dt * 10.0, mask)
        d_sem = maxdiff(o10, o_alt)
        assert d_sem < 1e-5, f"{name}: dt_scale != scaling dt ({d_sem:.2e})"
        # (d) recorded
        assert build_core(name, Dg, H, dt_scale=7.0).cfg_effective["dt_scale"] == 7.0
    # the reason the flag exists: CfC's time gate must actually move
    cfc1 = build_core("cfc", Dg, H).eval()
    cfc10 = copy.deepcopy(cfc1)
    cfc10.dt_scale = 10.0
    o1, a1 = cfc1(x, dt, mask)
    o10, a10 = cfc10(x, dt, mask)
    tg = float((a1["time_gate"] - a10["time_gate"]).abs().mean())
    print(f"    all {len(MODELS)} cores: dt_scale=1.0 bit-identical, "
          f"dt_scale=10 == dt*10 (<1e-5), output changes")
    print(f"    cfc: |d out| = {maxdiff(o1, o10):.4f}, "
          f"mean |d time_gate| = {tg:.4f}  (timespan channel is live)")


@test
def test_mamba_scan():
    """The chunked scan must equal the naive per-step recurrence to 1e-5."""
    import copy
    B, T, Dg, H = 3, 100, 16, 24          # T=100 with chunk=32 -> 4 chunks
    x, dt, mask = make_batch(B, T, Dg)
    mask[:, 40:50] = False
    fast = build_core("mamba", Dg, H, chunk=32, scan="fast")
    naive = copy.deepcopy(fast)
    naive.scan = "naive"
    o1, a1 = fast(x, dt, mask)
    o2, a2 = naive(x, dt, mask)
    d = maxdiff(o1, o2)
    print(f"    fast vs naive (T={T}, chunk=32): max|diff| = {d:.3e}")
    assert d < 1e-5, f"scan mismatch {d:.3e}"
    assert maxdiff(a1["dt_ssm"], a2["dt_ssm"]) < 1e-6
    # an empty mask must leave the state exactly untouched
    m = torch.ones(B, T, dtype=torch.bool)
    m[:, 3:] = False
    o3, _ = fast(x, dt, m)
    o4, _ = fast(x[:, :3], dt[:, :3], m[:, :3])
    assert maxdiff(o3[:, :3], o4) < 1e-6


@test
def test_streaming_matches_forward():
    """frame-by-frame step() must reproduce the batched forward pass."""
    B, T, Dg, H = 2, 10, 16, 24
    x, dt, mask = make_batch(B, T, Dg)
    mask[:, 6] = False
    for name in MODELS:
        core = build_core(name, Dg, H)
        out, _ = core(x, dt, mask)
        out_s = stream_forward(core, x, dt, mask)
        d = maxdiff(out, out_s)
        assert d < 1e-5, f"{name}: streaming path differs from forward ({d:.3e})"
    print("    step()-loop == forward() for all cores (< 1e-5)")


@test
def test_slot_bit_exact():
    """Hard-routed slots that are not selected must be BIT-identical."""
    B, T, Dg, H, N = 2, 12, 16, 24, 8
    core = build_core("dsm_slot", Dg, H, n_slots=N, slot_dim=16,
                      slot_hard_route=True, hard_write=True,
                      write_threshold=0.05, route_mode="content")
    x, dt, mask = make_batch(B, T, Dg)
    state = core.init_state(B, None)
    prev = state["vals"].clone()
    n_zero = 0
    for t in range(T):
        _o, aux, state = core.step(x[:, t], dt[:, t], mask[:, t], state)
        mass = aux["write_gate"].reshape(B, N)          # [B,N]
        unsel = mass == 0
        n_zero += int(unsel.sum())
        if unsel.any():
            if not torch.equal(state["vals"][unsel], prev[unsel]):
                bad = (state["vals"][unsel] - prev[unsel]).abs().max().detach()
                raise AssertionError(
                    f"unrouted slot changed at t={t} (max|diff|={float(bad):.3e})")
        assert (mass > 0).any(), f"no slot was ever written at t={t}"
        prev = state["vals"].clone()
    print(f"    {n_zero} un-routed slot-updates, all bit-identical (torch.equal)")


@test
def test_ablation_flags():
    """Every ablation flag must build, run, and be *recorded* in cfg_effective."""
    B, T, Dg, H = 2, 8, 12, 16
    x, dt, mask = make_batch(B, T, Dg)
    variants = [
        ("dsm", {}, "write_rule", "delta"),
        ("dsm", {"write_rule": "outer"}, "write_rule", "outer"),
        ("dsm", {"read_mode": "bilinear"}, "read_mode", "bilinear"),
        ("dsm", {"read_mode": "two_hop"}, "read_mode", "two_hop"),
        ("dsm", {"decay": False}, "decay", False),
        ("dsm", {"n_decay_groups": 2}, "n_decay_groups", 2),
        ("dsm", {"beta_exp": True}, "beta_exp", True),
        ("dsm", {"hard_write": True}, "hard_write", True),
        ("dsm", {"mem_off": True}, "mem_off", True),
        ("dsm", {"slow_off": True}, "slow_off", True),
        ("dsm", {"h_by_index": True}, "h_by_index", True),
        ("dsm", {"use_dt_struct": False}, "use_dt_struct", False),
        ("dsm", {"dt_clamp_max": 5.0}, "dt_clamp_max", 5.0),
        ("dsm", {"mem_dim": 32}, "mem_dim", 32),
        ("dsm", {"input_norm": False}, "input_norm", False),
        ("dsm", {"use_dt_feature": False}, "use_dt_feature", False),
        ("deltanet", {"write_rule": "outer"}, "write_rule", "outer"),
        ("deltanet", {"read_mode": "bilinear"}, "read_mode", "bilinear"),
        ("deltanet", {"decay": False}, "decay", False),
        ("dsm_slot", {"route_mode": "positional"}, "route_mode", "positional"),
        ("dsm_slot", {"route_mode": "none"}, "route_mode", "none"),
        ("dsm_slot", {"slot_hard_route": True}, "slot_hard_route", True),
        ("dsm_slot", {"n_slots": 8, "slot_dim": 32}, "n_slots", 8),
        ("dsm_slot", {"write_rule": "outer"}, "write_rule", "outer"),
        ("window", {"window_s": 1.0}, "window_s", 1.0),
        ("mamba", {"tdt": True}, "tdt", True),
        ("mamba+time", {}, "tdt", True),
        ("mamba", {"d_conv": 1}, "d_conv", 1),
        ("cfc", {"retention_gate": True}, "retention_gate", True),
        ("cfc", {"backbone_units": 0}, "backbone_units", 0),
        ("gru", {"mem_off": True, "write_rule": "outer"}, None, None),   # ignored
    ]
    for name, cfg, key, want in variants:
        core = build_core(name, Dg, H, **cfg)
        out, aux = core(x, dt, mask)
        assert torch.isfinite(out).all(), f"{name} {cfg}: non-finite"
        if key is not None:
            got = core.cfg_effective.get(key)
            assert got == want, f"{name} {cfg}: cfg_effective[{key}]={got!r} != {want!r}"
        else:
            # irrelevant keys are accepted but reported as unused, not silently used
            assert "mem_off" in core.cfg_unused, f"{name}: {cfg} not reported as unused"
        # a flag that is supposed to change the model must really change it
        if name == "dsm" and key == "mem_dim":
            assert core.cfg_effective["mem_dim_eff"] == 32
        if name == "dsm" and (cfg.get("mem_off") or cfg.get("slow_off")):
            assert core.n_params < build_core("dsm", Dg, H).n_params, \
                f"{cfg}: the disabled component still has parameters"
        if name in ("dsm", "deltanet") and key == "decay" and want is False:
            o0, _ = build_core(name, Dg, H)(x, dt, mask)
            assert maxdiff(out, o0) > 1e-6, f"{name}: decay=False changed nothing"
        if key == "write_rule" and want == "outer" and name in ("dsm", "deltanet"):
            o0, _ = build_core(name, Dg, H)(x, dt, mask)
            assert maxdiff(out, o0) > 1e-6, f"{name}: write_rule=outer changed nothing"
    print(f"    {len(variants)} ablation variants build/run and record cfg_effective")


@test
def test_tok_addressing():
    """`use_tok`: forward + streaming step, both tok shapes, invalid => no write.

    Objects live in ``tok [B,T,K,Dt]`` (channel ``tok_vis_channel`` = visibility).
    Requirements: the batched forward and the single-frame step must agree for
    **both** token shapes (``[B,K,Dt]`` and ``[B,1,K,Dt]``, the asymmetry that
    broke predict.py once), an invalid token must write *nothing* (bit-exact),
    and ``use_tok=False`` must ignore ``tok`` entirely.
    """
    B, T, K, Dt, Dg, H = 3, 9, 4, 17, 12, 16
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T, Dg, generator=g)
    dt = torch.full((B, T), 0.05)
    dt[:, 0] = 0
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 6] = False
    tok = torch.randn(B, T, K, Dt, generator=g)
    tok[..., 16] = (torch.rand(B, T, K, generator=g) > 0.3).float()
    tok[~mask] = 0.0
    tv = tok[..., 16] > 0
    for name in ("dsm", "dsm_slot"):
        core = build_core(name, Dg, H, use_tok=True, tok_dim=Dt).eval()
        out, aux = core(x, dt, mask, tok=tok)
        assert torch.isfinite(out).all(), f"{name}: tok forward not finite"
        assert tuple(aux["slot_read"].shape) == (B, T, K, 64), \
            (name, aux["slot_read"].shape)
        if name == "dsm_slot":          # routing weights for the crosstalk probe
            route = aux["route"]
            assert tuple(route.shape) == (B, T, K, 12), (name, route.shape)
            assert torch.allclose(route.sum(-1), torch.ones_like(route.sum(-1)),
                                  atol=1e-5), f"{name}: route rows do not sum to 1"
        # streaming parity, both token shapes
        for shape in ("[B,K,Dt]", "[B,1,K,Dt]"):
            st = core.init_state(B, None)
            outs = []
            for t in range(T):
                one = shape == "[B,K,Dt]"
                o, _a, st = core.step(x[:, t], dt[:, t], mask[:, t], st,
                                      tok_t=tok[:, t] if one else tok[:, t:t + 1],
                                      tok_valid_t=tv[:, t] if one else tv[:, t:t + 1])
                outs.append(o)
            d = maxdiff(out, torch.stack(outs, 1))
            assert d < 1e-5, f"{name}: tok streaming {shape} differs by {d:.2e}"
        # an invalid token must write nothing: garbage there changes nothing
        tok2 = tok.clone()
        tok2[~tv] = float("nan")
        out2, _ = core(x, dt, mask, tok=tok2)
        assert maxdiff(out, out2) == 0.0, f"{name}: invalid token leaked into output"
        s1 = core.step(x[:, 0], dt[:, 0], mask[:, 0], core.init_state(B, None),
                       tok_t=tok[:, 0], tok_valid_t=tv[:, 0])[2]
        t3 = tok[:, 0].clone()
        t3[~tv[:, 0]] = 12345.0
        s2 = core.step(x[:, 0], dt[:, 0], mask[:, 0], core.init_state(B, None),
                       tok_t=t3, tok_valid_t=tv[:, 0])[2]
        for k in s1:
            if torch.is_tensor(s1[k]):
                assert torch.equal(s1[k], s2[k]), \
                    f"{name}: state[{k}] changed by an invalid token"
        # step() with tok absent must still work on a use_tok=True core
        o_abs, _a, _s = core.step(x[:, 0], dt[:, 0], mask[:, 0], core.init_state(B, None))
        assert tuple(o_abs.shape) == (B, H)
        core_full, _ = core(x, dt, mask)
        assert tuple(core_full.shape) == (B, T, H)
        # use_tok=False must ignore tok bit-exactly (existing suites stay valid)
        core0 = build_core(name, Dg, H).eval()
        a0, _ = core0(x, dt, mask)
        b0, _ = core0(x, dt, mask, tok=tok)
        assert maxdiff(a0, b0) == 0.0, f"{name}: use_tok=False is not tok-invariant"
    print("    use_tok: forward+step (both shapes), invalid token bit-exact, "
          "tokenless path unchanged")


@test
def test_fp16_autocast(use_gpu: bool = True):
    """No NaN/inf under CPU bf16, and under CUDA fp16 when a GPU is free."""
    B, T, Dg, H = 2, 16, 8, 16
    x, dt, mask = make_batch(B, T, Dg)
    for name in MODELS:
        core = build_core(name, Dg, H)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out, aux = core(x, dt, mask)
        assert torch.isfinite(out.float()).all(), f"{name}: bf16 autocast produced NaN"
        for k, v in aux.items():
            if torch.is_tensor(v):
                assert torch.isfinite(v.float()).all(), f"{name}: aux[{k}] NaN under bf16"
    print(f"    bf16 autocast clean for {len(MODELS)} cores")
    if not use_gpu or not torch.cuda.is_available():
        print("    (CUDA fp16 check skipped)")
        return
    try:
        free, _total = torch.cuda.mem_get_info()
        if free < 1.0 * 1024 ** 3:
            print(f"    (CUDA fp16 check skipped: only {free / 1024**3:.2f} GB free)")
            return
        dev = "cuda"
        xg, dtg, maskg = x.to(dev), dt.to(dev), mask.to(dev)
        for name in MODELS:
            core = build_core(name, Dg, H).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                out, aux = core(xg, dtg, maskg)
            assert torch.isfinite(out.float()).all(), f"{name}: fp16 autocast NaN"
        print("    CUDA fp16 autocast clean (tiny batch, <50 MB)")
    except torch.cuda.OutOfMemoryError:
        print("    (CUDA fp16 check skipped: OOM, GPU is busy with feature extraction)")
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- reports --- #
HIDDENS = (64, 128, 192, 256)
D_INS = (768, 64)


def print_param_table(models=None):
    models = models or MODELS
    print("\n=== parameter table (total params, core only) ===")
    head = f"{'model':10s}"
    for d in D_INS:
        for h in HIDDENS:
            head += f"{'d' + str(d) + '/h' + str(h):>12s}"
    print(head)
    for name in models:
        row = f"{name:10s}"
        for d in D_INS:
            for h in HIDDENS:
                row += f"{param_count(name, d, h):12,d}"
        print(row)


def print_latency_table(models=None, B: int = 32, T: int = 64, reps: int = 2):
    models = models or MODELS
    print(f"\n=== CPU latency, B={B} T={T} (no_grad, {reps} reps, median) ===")
    print(f"{'model':10s}{'d_in':>6s}{'hidden':>8s}{'params':>12s}"
          f"{'fwd ms':>10s}{'ms/step':>10s}{'1-frame ms':>12s}")
    for name in models:
        for d_in, hidden in ((768, 128), (64, 64), (64, 128), (64, 192), (64, 256)):
            x, dt, mask = make_batch(B, T, d_in)
            core = build_core(name, d_in, hidden).eval()
            with torch.no_grad():
                core(x, dt, mask)                                   # warm-up
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    core(x, dt, mask)
                    ts.append(time.perf_counter() - t0)
                med = statistics.median(ts) * 1e3
                st = core.init_state(B, x.device)
                o, _a, st = core.step(x[:, 0], dt[:, 0], mask[:, 0], st)
                t0 = time.perf_counter()
                for _ in range(5):
                    core.step(x[:, 1], dt[:, 1], mask[:, 1], st)
                one = (time.perf_counter() - t0) / 5 * 1e3
            print(f"{name:10s}{d_in:6d}{hidden:8d}{core.n_params:12,d}"
                  f"{med:10.1f}{med / T:10.2f}{one:12.2f}")


# ------------------------------------------------------------------- main --- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="temporal-core zoo test-suite")
    ap.add_argument("--quick", action="store_true", help="skip the latency grid")
    ap.add_argument("--no-gpu", action="store_true", help="skip the CUDA fp16 check")
    ap.add_argument("--no-table", action="store_true", help="skip the tables")
    ap.add_argument("--ignore", nargs="*", default=[], help="test names to skip")
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.manual_seed(0)
    print(f"cores: {', '.join(MODELS)}")
    ok, failed = 0, []
    t_start = time.time()
    for name, fn in TESTS:
        if name in args.ignore:
            print(f"[SKIP] {name}")
            continue
        print(f"[TEST] {name}")
        t0 = time.time()
        try:
            if name == "test_fp16_autocast":
                fn(use_gpu=not args.no_gpu)
            else:
                fn()
            ok += 1
            print(f"  PASS ({time.time() - t0:.1f}s)")
        except Exception:
            failed.append(name)
            print(f"  FAIL ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    if not args.no_table:
        print_param_table()
        if not args.quick:
            print_latency_table()
    print(f"\n{ok} passed, {len(failed)} failed  ({time.time() - t_start:.1f}s)")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
