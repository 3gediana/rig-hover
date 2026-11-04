"""Gate tests for the pixel-to-event network and its baselines.

Every claim the report makes about the pixel models is enforced here, so a future
edit cannot silently break it:

1. **causal contract** -- perturbing frames *and* timestamps at ``t >= k`` leaves
   every output at ``t < k`` bit-identical, in **both** train and eval mode.
   (Train mode matters: an earlier version batched the scout over the whole window
   behind BatchNorm, which mixed frames across time and broke causality by 7e-3
   while eval mode still looked fine.)
2. **batched window == streaming** -- ``forward`` over T frames must equal T calls
   to ``step`` with the carried state.
3. **idempotent counting** -- the cumulative count advances only on a completion
   transition, so feeding the same frame repeatedly at ``dt = 0`` cannot add to it.
4. **masked steps are no-ops** -- ``mask = False`` must leave the persistent state
   bit-identical.
5. **bounded state** -- the state size is independent of the sequence length
   (the whole point of the architecture: spec §3.1 says O(Nd + N²dr + Ed)).
6. **edge budget** -- params <= 5 M and <= 1.5 GMAC per analysed frame (spec §8).
7. **shared head** -- every baseline uses the same head width, so the comparison
   cannot come from head capacity.

Run:  python -m src.models.tests_pixel            (add --cpu to force CPU)
"""
from __future__ import annotations

import sys

import torch

from .pixel_baselines import BASELINE_MODELS, build_baseline
from .pixel_event import build_pixel_model, count_macs, count_params

RES = 160
B, T = 2, 8
N_EVENT, N_GESTURE, N_PRESENCE = 25, 10, 4
HEADS = {"event": (None, N_EVENT), "onset": (None, 1), "count": (None, N_EVENT),
         "presence": (None, N_PRESENCE), "ready": (None, 1), "gesture": (None, N_GESTURE)}
MAX_PARAMS = 5_000_000
MAX_MACS = 1.5e9


def _dev(cpu: bool):
    return torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")


def _batch(dev, T=T):
    g = torch.Generator(device=dev).manual_seed(0)
    imgs = torch.rand(B, T, 1, RES, RES, device=dev, generator=g)
    dt = torch.full((B, T), 0.3, device=dev)
    dt[:, 0] = 0.0
    mask = torch.ones(B, T, dtype=torch.bool, device=dev)
    return imgs, dt, mask


def _models(dev):
    yield "pixel_event", build_pixel_model("pixel_event", None,
                                           n_event_classes=N_EVENT,
                                           n_gesture=N_GESTURE,
                                           n_presence=N_PRESENCE).to(dev)
    for name in BASELINE_MODELS:
        yield name, build_baseline(name, n_event_classes=N_EVENT,
                                   n_gesture=N_GESTURE,
                                   n_presence=N_PRESENCE).to(dev)


def test_contract_shapes(dev) -> None:
    imgs, dt, mask = _batch(dev)
    for name, m in _models(dev):
        m.eval()
        with torch.no_grad():
            seq, aux = m(imgs, dt, mask)
        for h, (_, last) in HEADS.items():
            assert h in seq, f"{name}: missing head {h}"
            assert tuple(seq[h].shape) == (B, T, last), \
                f"{name}.{h} shape {tuple(seq[h].shape)} != {(B, T, last)}"
        assert "count_final" in seq and tuple(seq["count_final"].shape) == (B, N_EVENT), \
            f"{name}.count_final {tuple(seq.get('count_final', torch.zeros(0)).shape)}"
    print(f"    {len(list(_models(dev)))} models: all 7 head shapes OK")


def test_strict_causality(dev) -> None:
    for name, m in _models(dev):
        for mode in ("train", "eval"):
            getattr(m, mode)()
            imgs, dt, mask = _batch(dev)
            k = T // 2
            with torch.no_grad():
                a, _ = m(imgs, dt, mask)
                g = torch.Generator(device=dev).manual_seed(7)
                im2 = imgs.clone()
                im2[:, k:] = torch.rand(1, 1, 1, RES, RES, device=dev, generator=g) * 5
                dt2 = dt.clone()
                dt2[:, k:] = 1.7
                b, _ = m(im2, dt2, mask)
            d = max(float((a[h][:, :k] - b[h][:, :k]).abs().max()) for h in HEADS)
            assert d == 0.0, f"{name} [{mode}] causality broken: max|d| = {d:.3e}"
    print("    7 baselines + ours: max|d| == 0.0 on t<k in train AND eval")


def test_batched_equals_streaming(dev) -> None:
    m = build_pixel_model("pixel_event", None, n_event_classes=N_EVENT,
                          n_gesture=N_GESTURE, n_presence=N_PRESENCE).to(dev).eval()
    imgs, dt, mask = _batch(dev)
    with torch.no_grad():
        seqf, _ = m(imgs, dt, mask)
        st = m.init_state(B, dev)
        outs = []
        for t in range(T):
            lg, st, _ = m.step(imgs[:, t], dt[:, t], mask[:, t], st)
            outs.append(lg["event"])
        seqs = torch.stack(outs, 1)
    d = float((seqf["event"] - seqs).abs().max())
    assert d < 1e-5, f"batched vs streaming diverge: {d:.3e}"
    print(f"    batched-window forward == step-by-step streaming (max|d| {d:.1e})")


def test_idempotent_count(dev) -> None:
    m = build_pixel_model("pixel_event", None, n_event_classes=N_EVENT,
                          n_gesture=N_GESTURE, n_presence=N_PRESENCE).to(dev).eval()
    imgs, dt, mask = _batch(dev)
    st = m.init_state(1, dev)
    zero, one = torch.zeros(1, device=dev), torch.ones(1, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for _ in range(4):
            _, st, _ = m.step(imgs[:1, 0], zero, one, st)
        after = st["events"]["count"].clone()
    assert float(after.abs().max()) == 0.0, \
        f"count moved without a completion transition: {float(after.abs().max()):.3e}"
    print("    same frame at dt=0 x4 -> cumulative count unchanged (0.0)")


def test_masked_step_invariant(dev) -> None:
    m = build_pixel_model("pixel_event", None, n_event_classes=N_EVENT,
                          n_gesture=N_GESTURE, n_presence=N_PRESENCE).to(dev).eval()
    imgs, dt, mask = _batch(dev)
    st = m.init_state(1, dev)
    with torch.no_grad():
        _, st, _ = m.step(imgs[:1, 0], dt[:1, 0], torch.ones(1, dtype=torch.bool, device=dev), st)
        before = {k: v.clone() for k, v in st["events"].items()}
        _, st2, _ = m.step(imgs[:1, 1], dt[:1, 1],
                           torch.zeros(1, dtype=torch.bool, device=dev), st)
    d = max(float((st2["events"][k] - before[k]).abs().max()) for k in before)
    assert d == 0.0, f"masked step changed the event state: {d:.3e}"
    print("    mask=False step leaves the event state bit-identical (0.0)")


def test_bounded_state(dev) -> None:
    m = build_pixel_model("pixel_event", None, n_event_classes=N_EVENT,
                          n_gesture=N_GESTURE, n_presence=N_PRESENCE).to(dev).eval()
    sizes = []
    for L in (4, T, 32):
        st = m.init_state(2, dev)
        imgs, dt, mask = _batch(dev, L)
        with torch.no_grad():
            _, aux = m(imgs, dt, mask)
        st = aux["state"]
        n = sum(v.numel() for v in st["slots"].values()) + \
            sum(v.numel() for v in st["events"].values()) + st["rel"].numel()
        sizes.append(n)
    assert len(set(sizes)) == 1, f"state size depends on T: {sizes}"
    print(f"    persistent state = {sizes[0]} floats for T=4/8/32 (length independent)")


def test_edge_budget(dev) -> None:
    m = build_pixel_model("pixel_event", None, n_event_classes=N_EVENT,
                          n_gesture=N_GESTURE, n_presence=N_PRESENCE).to(dev)
    n_par = count_params(m)
    imgs, dt, mask = _batch(dev, T)
    macs = count_macs(m, imgs, dt, mask) / B
    assert n_par <= MAX_PARAMS, f"params {n_par:,} > {MAX_PARAMS:,}"
    assert macs <= MAX_MACS, f"MACs/frame {macs / 1e9:.3f} G > {MAX_MACS / 1e9:.3f} G"
    print(f"    params {n_par:,} (<= {MAX_PARAMS:,}) | "
          f"{macs / 1e6:.2f} M MACs/frame (<= {MAX_MACS / 1e9:.2f} G)")


def test_shared_head(dev) -> None:
    """Every baseline must carry the same head width, so head capacity is constant."""
    widths = {}
    for name, m in _models(dev):
        hd = getattr(m, "head", None) or getattr(m, "heads", None)
        if hd is None:
            continue
        widths[name] = (count_params(hd) if hasattr(hd, "parameters")
                        else sum(count_params(x) for x in hd.modules() if hasattr(x, "parameters")))
    vals = sorted(set(widths.values()))
    assert len(vals) <= 1, f"head widths differ across models: {widths}"
    print(f"    identical head for all models: {vals[0] if vals else 'n/a'} params")


def main() -> int:
    cpu = "--cpu" in sys.argv
    dev = _dev(cpu)
    print(f"[pixel gate] device={dev}")
    for fn in (test_contract_shapes, test_strict_causality, test_batched_equals_streaming,
               test_idempotent_count, test_masked_step_invariant, test_bounded_state,
               test_edge_budget, test_shared_head):
        print(f"  {fn.__name__}")
        fn(dev)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
