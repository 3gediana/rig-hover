"""Unit tests for the controlled synthetic memory benchmark.

Run with either

    python -m src.synth.tests_synth          # self-contained runner
    pytest src/synth/tests_synth.py -q

Covers (in the order the Lead cares about)

1. tensor contract: shapes / dtypes / ranges / strictly increasing t
2. the honesty invariant: `lab_mask` marks only steps where the value is
   recoverable from the observation history but absent from the current input;
   `lab_age` counts physical seconds since the last information-bearing
   observation; `lab_gap_s` carries the episode's designated gap
3. contract interop: `StreamDataset` + `collate_stream` round trip
4. manifests: `check_split_disjoint`, stable layout signature everywhere
5. determinism: same config + episode seed -> byte-identical arrays
6. oracle sanity (the headline assertions):
     perfect memory  -> main metric 1.0 on every task
     memoryless      -> ~chance (0 cosine, 1/N_CODES identity accuracy)
     random          -> ~chance
   plus crosstalk / rollout / capacity / update-latency / event probes
7. irregular sampling + temporal-shuffle control behaviour
8. gap-sweep identical-latent property (paired comparison across gaps)
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..common.contract import Manifest, StreamDataset, collate_stream
from . import layout as L
from . import probes as P
from .base import (load_seq, perturb_raw, raw_to_seq, temporal_shuffle,
                   validate_seq)
from .config import build_plan, load_config
from .phases import MAIN_METRIC
from .tasks import TASKS, build_task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT_ROOT / "configs" / "synth_v1.json"
N_SEQS = 16          # per task in the oracle tests
SEED = 20261103


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg():
    return load_config(str(CFG_PATH) if CFG_PATH.exists() else None)


def _make_truths(task: str, n: int = N_SEQS, split: str = "test",
                 cfg=None) -> List[Dict[str, np.ndarray]]:
    cfg = cfg or _cfg()
    dt0 = float(cfg["nominal_dt"])
    lo, _hi = cfg["split_ranges"][split]
    params = cfg["resolved_task_params"][task]
    truths = []
    for i in range(n):
        seed = [SEED, hash_task_seed(task), lo + i]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        raw = build_task(task, params).make(rng)
        truths.append(raw_to_seq(raw, dt0=dt0))
    return truths


def hash_task_seed(task: str) -> int:
    from .base import stable_hash_int
    return stable_hash_int("synth_" + task)


def _seq_scalar(seq, key: str) -> float:
    return float(np.asarray(seq[key]).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# 1) layout / codec
# --------------------------------------------------------------------------- #
def test_layout_codec():
    codes = np.arange(L.N_CODES)
    vec = L.code_to_vec(codes)
    assert vec.shape == (L.N_CODES, L.V)
    back = L.decode_code(vec)
    assert np.array_equal(back, codes), (back, codes)
    # analogue dims do not disturb decoding
    rng = np.random.default_rng(0)
    noisy = vec + np.concatenate([np.zeros((L.N_CODES, L.VC)),
                                  rng.normal(0, 2.0, (L.N_CODES, L.VA))], 1)
    assert np.array_equal(L.decode_code(noisy), codes)
    # an all-zero (unknown) vector decodes deterministically to code 0
    assert int(L.decode_code(np.zeros(L.V))) == 0
    # flatten / unflatten
    st = rng.normal(size=(5, L.KMAX, L.V)).astype(np.float32)
    flat = L.flatten_state(st)
    assert flat.shape == (5, L.KMAX * L.V)
    assert np.allclose(L.unflatten_state(flat), st)
    # stable layout signature
    sig1, sig2 = L.layout_signature(), L.layout_signature()
    assert sig1 == sig2 and len(sig1) == 16
    specs = L.label_specs()
    assert specs["state"]["dim"] == L.KMAX * L.V
    assert specs["class"]["n_classes"] == L.N_CODES
    assert specs["event"]["kind"] == "bce"


# --------------------------------------------------------------------------- #
# 2) every task: shapes, dtypes, honesty invariant, lab_age
# --------------------------------------------------------------------------- #
def test_task_shapes_and_invariants():
    cfg = _cfg()
    for task in TASKS:
        truths = _make_truths(task, n=3)
        for seq in truths:
            validate_seq(seq)
            T = int(seq["t"].shape[0])
            # feat channel semantics
            assert seq["feat"].shape == (T, L.DG)
            assert np.all(seq["feat"][:, L.CH_BIAS] == 1.0)
            assert np.all(seq["feat"][:, L.FEAT_SUMMARY_DIM:] == 0.0)
            tok = seq["tok"].astype(np.float32)
            vis = tok[..., L.V] > 0.5
            assert np.allclose(seq["feat"][:, L.CH_VIS_FRAC], vis.sum(1) / L.KMAX,
                               atol=1e-3)
            # token present <=> vis channel 1 ; token absent <=> all zeros
            assert np.all((tok[..., L.V] == 0) | (tok[..., L.V] == 1))
            assert np.all(tok[~vis] == 0), "hidden token must be all zeros"
            # lab_mask is exactly "knowable but not currently fully observed";
            # the only exception is a detected-but-not-identifiable re-entry
            # (token present, identity channels zeroed)
            mask = seq["lab_mask"] > 0.5
            id_part = np.abs(tok[..., :L.VC]).sum(-1)
            idmask = vis & (id_part == 0)
            assert not np.any(mask & vis & ~idmask), \
                "masked steps must not carry a fully observed token"
            assert np.all(idmask <= (mask | ~mask)), "sanity"
            st = L.unflatten_state(seq["lab_state"])
            assert np.all(np.any(st[mask] != 0, axis=-1))
            # lab_age: 0 on observed steps, = t - t[last observation] afterwards
            age = seq["lab_age"]
            assert np.all(age[vis] == 0), "age must be 0 on observed steps"
            t = seq["t"].astype(np.float64)
            for k in range(L.KMAX):
                last = -1
                for i in range(T):
                    if vis[i, k]:
                        last = i
                        assert age[i, k] == 0
                    elif last >= 0:
                        expected = t[i] - t[last]
                        assert math.isclose(float(age[i, k]), float(expected),
                                            rel_tol=1e-4, abs_tol=1e-4), \
                            (task, i, k, age[i, k], expected)
            # lab_gap_s constant per episode
            gap = _seq_scalar(seq, "lab_gap_s")
            assert np.all(seq["lab_gap_s"] == gap)
            assert gap == -1.0 or gap >= 0.0
            # class labels in range or -1
            assert np.all((seq["lab_class"] >= -1) & (seq["lab_class"] < L.N_CODES))
            # event label is binary
            assert np.all((seq["lab_event"] == 0) | (seq["lab_event"] == 1))
            # crosstalk spec is padded with -1
            cross = seq["meta_cross"]
            assert cross.ndim == 2 and cross.shape[1] == 5
            assert np.all(cross[0] == -1.0) or np.all(cross[:, 0] >= 0)
        # a task must actually *use* the memory mask
        assert sum(int((s["lab_mask"] > 0.5).sum()) for s in truths) > 0, task
    print(f"  [ok] shapes + honesty invariant for {len(TASKS)} tasks")


def test_gap_s_sweep_parameter():
    cfg = _cfg()
    params = dict(cfg["resolved_task_params"]["recall_single"])
    dt0 = float(cfg["nominal_dt"])
    for gap_s in (0.1, 0.4, 1.6, 3.2):
        params["gap_s"] = gap_s
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, 1000]))
        raw = build_task("recall_single", params).make(rng)
        seq = raw_to_seq(raw, dt0=dt0)
        steps = max(1, int(round(gap_s / dt0)))
        assert raw.gap_s == steps * dt0, (raw.gap_s, gap_s)
        assert np.all(seq["lab_gap_s"] == steps * dt0)
        # the number of hidden-but-masked steps in the gap equals `steps`
        assert int((seq["lab_mask"] > 0.5).sum()) >= steps - 1
        # lab_age on the last gap step ~ gap
        age = seq["lab_age"]
        assert age.max() >= steps * dt0 - 1e-3
    print("  [ok] lab_gap_s / lab_age track the fixed gap sweep")


# --------------------------------------------------------------------------- #
# 3) contract interop
# --------------------------------------------------------------------------- #
def test_collate_roundtrip():
    import torch
    from ..common.contract import SeqRecord
    tmp = Path(tempfile.mkdtemp(prefix="synth_collate_"))
    try:
        truths = _make_truths("recall_multi", n=4)
        recs = []
        for i, seq in enumerate(truths):
            from .base import save_seq
            p = tmp / f"s{i}.npz"
            save_seq(seq, p)
            recs.append(SeqRecord(seq_id=f"s{i}", source_id=f"ep{i}", split="train",
                                  T=int(seq["t"].shape[0]), path=str(p)))
        ds = StreamDataset(recs, load_tok=True)
        batch = collate_stream([ds[i] for i in range(4)])
        assert batch["x"].shape[1] == max(r.T for r in recs)
        assert batch["x"].shape[2] == L.DG
        assert batch["tok"].shape[2:] == (L.KMAX, L.TOK_DIM)
        assert batch["mask"].dtype == torch.bool
        assert batch["dt"][:, 0].sum() == 0
        for k, v in batch["labels"].items():
            assert v.shape[0] == 4, k
        assert batch["labels"]["state"].shape[-1] == L.KMAX * L.V
        assert batch["labels"]["age"].shape[-1] == L.KMAX
        assert batch["labels"]["gap_s"].ndim == 2
        assert batch["labels"]["class"].dtype.is_floating_point is False
        # padding: padded steps are masked out
        for b, r in enumerate(recs):
            assert int(batch["mask"][b].sum()) == r.T
        print("  [ok] StreamDataset + collate_stream round trip")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 4) generated suite: manifests, split disjointness, identical layout
# --------------------------------------------------------------------------- #
def test_generated_suite_manifests():
    from .generate import generate_suite
    tmp = Path(tempfile.mkdtemp(prefix="synth_suite_"))
    try:
        cfg = _cfg()
        summary = generate_suite(tmp, cfg, tiny=True, references=True, quiet=True,
                                 staging_dir=None)
        assert summary["n_datasets"] > 20
        sig = L.layout_signature()
        refs = {}
        for ds in summary["datasets"]:
            man = Manifest.load(ds["manifest"])
            man.check_split_disjoint()
            assert man.feature_dim == L.DG
            assert man.tok_dim == L.TOK_DIM
            assert man.extra["layout_signature"] == sig
            assert man.extra["main_metric"] in ("cos", "acc", "f1")
            assert man.label_specs == L.label_specs()
            # same tensor layout for sweeps / shuffled / irregular
            for s in man.seqs:
                arr = load_seq(s.path)
                assert arr["feat"].shape[1] == L.DG
                assert arr["tok"].shape[1:] == (L.KMAX, L.TOK_DIM)
                assert arr["lab_state"].shape[1] == L.KMAX * L.V
            assert "reference" in ds
            refs[ds["dataset"]] = ds["reference"]
        # the required eval-only companions exist and are linked from the base
        base = Manifest.load(summary["datasets"][0]["manifest"])
        names = {d["dataset"] for d in summary["datasets"]}
        for t in ("recall_single", "recall_multi", "event_seq"):
            man = Manifest.load(tmp / "tiny" / "manifests" / f"synth_{t}.json")
            assert man.extra["eval_manifests"], t
            for n in man.extra["eval_manifests"]:
                assert n in names, (t, n)
        # gap sweep datasets carry the exact gap
        for gap in (0.1, 0.4, 1.6, 3.2):
            man = Manifest.load(tmp / "tiny" / "manifests"
                                / f"synth_recall_single_gap{gap:g}s.json")
            assert abs(float(man.extra["gap_s"]) - gap) < 1e-9
            assert man.extra["eval_only"] is True
            assert man.counts().get("test", 0) > 0
        # capacity sweeps carry K
        for k in (1, 2, 4, 8, 12, 16):
            man = Manifest.load(tmp / "tiny" / "manifests"
                                / f"synth_capacity_K{k}.json")
            assert int(man.extra["capacity_k"]) == k
            assert int(man.extra["task_params"]["k"]) == k
        print("  [ok] tiny suite: split disjointness + identical layout + "
              "eval-only links + gap/K metadata")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 5) determinism
# --------------------------------------------------------------------------- #
def test_determinism():
    cfg = _cfg()
    dt0 = float(cfg["nominal_dt"])
    params = cfg["resolved_task_params"]["recall_multi"]
    truths = []
    for rep in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, 1000]))
        raw = build_task("recall_multi", params).make(rng)
        truths.append(raw_to_seq(raw, dt0=dt0))
    for k in truths[0]:
        a, b = np.asarray(truths[0][k]), np.asarray(truths[1][k])
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b), k
    print("  [ok] episode generation is bit-deterministic")


# --------------------------------------------------------------------------- #
# 6) oracle tests (the headline assertions)
# --------------------------------------------------------------------------- #
def _oracle(task: str, truths, name: str):
    return P.make_oracle(name, task).predict(truths)


def test_oracles_main_metric():
    for task, metric in MAIN_METRIC.items():
        if task.startswith("pattern_"):
            continue          # covered by test_pattern_oracles_and_honesty
        truths = _make_truths(task, n=N_SEQS)
        perfect = P.main_scores(task, P.make_probe_seqs(
            truths, _oracle(task, truths, "perfect")))
        memless = P.main_scores(task, P.make_probe_seqs(
            truths, _oracle(task, truths, "memoryless")))
        chance = P.main_scores(task, P.make_probe_seqs(
            truths, _oracle(task, truths, "chance")))
        assert np.isfinite(perfect[metric]), (task, perfect)
        assert np.isclose(perfect[metric], 1.0, atol=1e-3), (task, perfect)
        if metric == "cos":
            assert memless["cos"] <= 0.05, (task, memless)
            assert abs(chance["cos"]) <= 0.10, (task, chance)
        elif metric == "acc":
            assert memless["acc"] <= 3.0 / L.N_CODES, (task, memless)
            assert abs(chance["acc"] - 1.0 / L.N_CODES) <= 0.10, (task, chance)
        elif metric == "f1":
            # event detection is visible -> both oracles get the events right;
            # the *memory* part of event_seq is the post-event state.
            assert np.isclose(memless["f1"], 1.0, atol=1e-2), (task, memless)
            post_p = perfect["post_event_cos"]
            post_m = memless["post_event_cos"]
            assert np.isclose(post_p["1x"], 1.0, atol=1e-3), (task, post_p)
            assert np.isclose(post_p["8x"], 1.0, atol=1e-3), (task, post_p)
            assert abs(post_m["1x"]) <= 0.05, (task, post_m)
        print(f"  [ok] {task:16s} {metric}: perfect={perfect[metric]:.4f} "
              f"memoryless={memless[metric]:.4f} chance={chance[metric]:.4f}")


def test_oracle_crosstalk():
    truths = _make_truths("recall_multi", n=N_SEQS)
    rep = {name: P.crosstalk_matrix(
        P.make_probe_seqs(truths, _oracle("recall_multi", truths, name)),
        "recall_multi") for name in ("perfect", "memoryless")}
    p, m = rep["perfect"], rep["memoryless"]
    assert p["n_cross_events"] > 0
    assert p["crosstalk_offdiag_mean"] <= 0.02, p
    assert p["crosstalk_diag_mean"] >= 0.5, p          # updates really happen
    # the memoryless model never updates on hidden steps -> zero diagonal
    assert m["crosstalk_diag_mean"] <= 0.05, m
    print(f"  [ok] crosstalk: perfect diag={p['crosstalk_diag_mean']:.3f} "
          f"offdiag={p['crosstalk_offdiag_mean']:.4f} | "
          f"memoryless diag={m['crosstalk_diag_mean']:.3f} (never updates)")


def test_oracle_rollout():
    for task in ("recall_single", "recall_multi", "capacity_scan",
                 "change_detect"):
        truths = _make_truths(task, n=N_SEQS)
        p = P.rollout(P.make_probe_seqs(truths, _oracle(task, truths, "perfect")), task)
        m = P.rollout(P.make_probe_seqs(truths, _oracle(task, truths, "memoryless")), task)
        for d in ("1", "3", "5", "10"):
            assert np.isclose(p["rollout"][d], 1.0, atol=1e-3), (task, d, p)
            assert abs(m["rollout"][d]) <= 0.05, (task, d, m)
        print(f"  [ok] rollout {task:16s} perfect 1/3/5/10 = "
              f"{p['rollout']['1']:.2f}/{p['rollout']['3']:.2f}/"
              f"{p['rollout']['5']:.2f}/{p['rollout']['10']:.2f}  "
              f"memoryless ~0")


def test_oracle_capacity():
    cfg = _cfg()
    params = dict(cfg["resolved_task_params"]["capacity_scan"])
    truths = []
    for k in (1, 2, 4, 8, 12, 16):
        params["k"] = k
        for i in range(6):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, 1000 + i, k]))
            truths.append(raw_to_seq(build_task("capacity_scan", params).make(rng),
                                     dt0=float(cfg["nominal_dt"])))
    p = P.capacity_curve(P.make_probe_seqs(truths, _oracle("capacity_scan", truths,
                                                           "perfect")),
                         "capacity_scan")
    m = P.capacity_curve(P.make_probe_seqs(truths, _oracle("capacity_scan", truths,
                                                           "memoryless")),
                         "capacity_scan")
    assert p["K_star"] == 16, p
    assert all(np.isclose(v, 1.0, atol=1e-3) for v in p["capacity_curve"].values())
    assert all(abs(v) <= 0.05 for v in m["capacity_curve"].values())
    print(f"  [ok] capacity: perfect K*={p['K_star']} curve="
          f"{ {k: round(v, 3) for k, v in p['capacity_curve'].items()} }")


def test_oracle_update_latency_false_updates():
    truths = _make_truths("change_detect", n=N_SEQS)
    p_upd = P.update_latency(P.make_probe_seqs(truths, _oracle(
        "change_detect", truths, "perfect")), "change_detect")
    m_upd = P.update_latency(P.make_probe_seqs(truths, _oracle(
        "change_detect", truths, "memoryless")), "change_detect")
    assert p_upd["n_change_events"] > 0
    assert p_upd["update_success"] == 1.0, p_upd
    assert p_upd["update_latency_samples"] <= 2.0, p_upd
    assert m_upd["update_success"] == 0.0, m_upd      # never updates while hidden
    p_fu = P.false_update_rate(P.make_probe_seqs(truths, _oracle(
        "change_detect", truths, "perfect")), "change_detect")
    assert p_fu["false_update_rate"] <= 1e-3, p_fu
    assert np.isclose(p_fu["hold_cos"], 1.0, atol=1e-3), p_fu
    print(f"  [ok] change_detect: perfect latency="
          f"{p_upd['update_latency_samples']:.2f} samples success="
          f"{p_upd['update_success']:.2f}; memoryless success="
          f"{m_upd['update_success']:.2f}; false_update_rate="
          f"{p_fu['false_update_rate']:.2e}")


def test_oracle_identity():
    truths = _make_truths("vanish_reappear", n=N_SEQS)
    p = P.identity_accuracy(P.make_probe_seqs(truths, _oracle(
        "vanish_reappear", truths, "perfect")), "vanish_reappear")
    m = P.identity_accuracy(P.make_probe_seqs(truths, _oracle(
        "vanish_reappear", truths, "memoryless")), "vanish_reappear")
    c = P.identity_accuracy(P.make_probe_seqs(truths, _oracle(
        "vanish_reappear", truths, "chance")), "vanish_reappear")
    assert np.isclose(p["acc"], 1.0, atol=1e-3), p
    assert np.isclose(p["acc_reentry_1step"], 1.0, atol=1e-3), p
    assert np.isclose(p["acc_reentry_5step"], 1.0, atol=1e-3), p
    assert m["acc"] <= 3.0 / L.N_CODES, m
    assert abs(m["acc"] - 1.0 / L.N_CODES) <= 0.12, m
    assert abs(c["acc"] - 1.0 / L.N_CODES) <= 0.12, c
    print(f"  [ok] identity: perfect={p['acc']:.3f} (reentry 1/3/5="
          f"{p['acc_reentry_1step']:.2f}/{p['acc_reentry_3step']:.2f}/"
          f"{p['acc_reentry_5step']:.2f}) memoryless={m['acc']:.3f} "
          f"chance={c['acc']:.3f}")


def test_oracle_event():
    truths = _make_truths("event_seq", n=N_SEQS)
    p = P.event_report(P.make_probe_seqs(truths, _oracle("event_seq", truths,
                                                         "perfect")), "event_seq")
    m = P.event_report(P.make_probe_seqs(truths, _oracle("event_seq", truths,
                                                         "memoryless")), "event_seq")
    assert np.isclose(p["f1"], 1.0, atol=1e-2), p
    assert np.isclose(p["event_dup_rate"], 0.0, atol=1e-6), p
    assert np.isclose(p["post_event_cos"]["1x"], 1.0, atol=1e-3), p
    assert np.isclose(p["post_event_cos"]["8x"], 1.0, atol=1e-3), p
    assert np.isclose(p["post_distractor_cos"]["4x"], 1.0, atol=1e-3), p
    assert abs(m["post_event_cos"]["4x"]) <= 0.05, m
    print(f"  [ok] event_seq: perfect f1={p['f1']:.3f} delay="
          f"{p['event_mean_delay_s']:.3f}s dup={p['event_dup_rate']:.2f} "
          f"post1x={p['post_event_cos']['1x']:.2f} post8x="
          f"{p['post_event_cos']['8x']:.2f} | memoryless post4x="
          f"{m['post_event_cos']['4x']:.2f}")


# --------------------------------------------------------------------------- #
# 7) irregular sampling + shuffled control
# --------------------------------------------------------------------------- #
def test_irregular_variants():
    cfg = _cfg()
    dt0 = float(cfg["nominal_dt"])
    params = cfg["resolved_task_params"]["recall_multi"]
    for variant in ("logdt", "drop30", "drop50", "drop70", "repeat", "burst",
                    "stretch2", "stretch4", "stretch8", "mixed"):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, 1000]))
        raw = build_task("recall_multi", params).make(rng)
        rng_p = np.random.default_rng(np.random.SeedSequence([SEED, 6, 1000]))
        pr = perturb_raw(raw, variant, rng_p, nominal_dt=dt0)
        seq = raw_to_seq(pr, dt0=dt0)
        validate_seq(seq)
        assert np.all(np.diff(seq["t"]) > 0)
        assert seq["t"].shape[0] >= 6, variant
        # perfect memory must still score 1.0: the observability mask removes
        # the frames whose write was dropped by the perturbation
        p = P.main_scores("recall_multi", P.make_probe_seqs(
            [seq], _oracle("recall_multi", [seq], "perfect")))
        assert np.isclose(p["cos"], 1.0, atol=1e-3), (variant, p)
        m = P.main_scores("recall_multi", P.make_probe_seqs(
            [seq], _oracle("recall_multi", [seq], "memoryless")))
        assert abs(m["cos"]) <= 0.05, (variant, m)
        print(f"  [ok] irregular {variant:9s} T={seq['t'].shape[0]:3d} "
              f"perfect_cos={p['cos']:.3f}")
    # time stretch must not change content, only the physical clock
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, 1000]))
    raw = build_task("recall_multi", params).make(rng)
    for factor in (2, 4, 8):
        pr = perturb_raw(raw, f"stretch{factor}",
                         np.random.default_rng(0), nominal_dt=dt0)
        assert np.allclose(pr.t, raw.t * factor, rtol=1e-5)
        assert np.array_equal(pr.vis, raw.vis)
        assert np.allclose(pr.lat, raw.lat)
    print("  [ok] 2x/4x/8x time stretch keeps content, scales only the clock")


def test_temporal_shuffle_control():
    truths = _make_truths("recall_multi", n=8)
    rng = np.random.default_rng(1234)
    shuffled = [temporal_shuffle(t, rng) for t in truths]
    for a, b in zip(truths, shuffled):
        # timestamps unchanged, rows preserved as a permutation
        assert np.array_equal(a["t"], b["t"])
        for key in ("feat", "lab_state", "lab_mask"):
            sa = sorted(map(lambda r: r.tobytes(), a[key]))
            sb = sorted(map(lambda r: r.tobytes(), b[key]))
            assert sa == sb, key
    p0 = P.main_scores("recall_multi", P.make_probe_seqs(
        truths, _oracle("recall_multi", truths, "perfect")))
    p1 = P.main_scores("recall_multi", P.make_probe_seqs(
        shuffled, _oracle("recall_multi", shuffled, "perfect")))
    assert np.isclose(p0["cos"], 1.0, atol=1e-3)
    assert p1["cos"] < 0.8, (p0, p1)       # the control really destroys memory
    print(f"  [ok] temporal shuffle: perfect cos {p0['cos']:.3f} -> "
          f"{p1['cos']:.3f} (control destroys temporal structure)")


# --------------------------------------------------------------------------- #
# 8) gap-sweep paired design
# --------------------------------------------------------------------------- #
def test_gap_sweep_paired_latent():
    cfg = _cfg()
    dt0 = float(cfg["nominal_dt"])
    params = dict(cfg["resolved_task_params"]["recall_single"])
    seeds = np.random.SeedSequence([SEED, 123, 1000])
    vals = []
    for gap in (0.1, 0.8, 3.2):
        pp = dict(params, gap_s=gap)
        rng = np.random.default_rng(seeds)
        seq = raw_to_seq(build_task("recall_single", pp).make(rng), dt0=dt0)
        # the written latent value is the same across gap datasets (paired)
        vis = seq["tok"][..., L.V] > 0.5
        first = seq["tok"][:, 0, :L.V][np.flatnonzero(vis[:, 0])[0]]
        vals.append(np.asarray(first, dtype=np.float32))
    assert np.allclose(vals[0], vals[1]) and np.allclose(vals[0], vals[2])
    print("  [ok] gap sweep datasets share the same latent value per episode "
          "(paired comparison across gaps)")


def test_recall_by_gap_and_capacity_buckets():
    cfg = _cfg()
    params = dict(cfg["resolved_task_params"]["recall_single"])
    truths = []
    for gap in (0.1, 0.8, 3.2):
        for i in range(4):
            pp = dict(params, gap_s=gap)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 77, 1000 + i]))
            truths.append(raw_to_seq(build_task("recall_single", pp).make(rng),
                                     dt0=float(cfg["nominal_dt"])))
    p = P.recall_by_gap(P.make_probe_seqs(
        truths, _oracle("recall_single", truths, "perfect")), "recall_single")
    m = P.recall_by_gap(P.make_probe_seqs(
        truths, _oracle("recall_single", truths, "memoryless")), "recall_single")
    # the small-gap episodes must land in small age buckets
    for key in ("1x", "2x", "4x"):
        assert key in p["by_gap"], (key, p["by_gap"])
    assert "64x" in p["by_gap"] or "32x" in p["by_gap"], p["by_gap"]
    for k, v in p["by_gap"].items():
        assert np.isclose(v, 1.0, atol=1e-3), (k, v)
        assert m["by_gap"][k] <= 0.05, (k, m["by_gap"])
    # single-K capacity datasets report a sensible K*
    cap_params = dict(cfg["resolved_task_params"]["capacity_scan"], k=16)
    seqs = []
    for i in range(6):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 88, 1000 + i]))
        seqs.append(raw_to_seq(build_task("capacity_scan", cap_params).make(rng),
                               dt0=float(cfg["nominal_dt"])))
    cc = P.capacity_curve(P.make_probe_seqs(
        seqs, _oracle("capacity_scan", seqs, "perfect")), "capacity_scan")
    assert cc["K_star"] == 16, cc
    assert cc["capacity_base_k"] == 16, cc
    print(f"  [ok] recall_by_gap buckets {sorted(p['by_gap'])} (perfect 1.0); "
          f"single-K capacity K*={cc['K_star']}")


# --------------------------------------------------------------------------- #
# 9) temporal-pattern family (core task candidate)
# --------------------------------------------------------------------------- #
PATTERN_CASES = [
    ("pattern_count", {}, 3),
    ("pattern_order", {}, 2),
    ("pattern_rate", {}, 2),
    ("pattern_count_rate_mixed", {}, 4),
    ("pattern_count_irregular", {"irregular": "mixed"}, 3),
    ("pattern_count", {"stretch": 2.0}, 3),
    ("pattern_count", {"stretch": 4.0}, 3),
    ("pattern_rate", {"stretch": 2.0}, 2),
    ("pattern_rate", {"stretch": 4.0}, 2),
]


def _pattern_seqs(task: str, extra: Dict, n_per_class: int = 8):
    from .pattern import build_pattern_seq, pattern_defaults
    params = pattern_defaults(task)
    params.update(extra)
    nc = int(params["n_classes"])
    truths = []
    for c in range(nc):
        for i in range(n_per_class):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, c, 5000 + i]))
            truths.append(build_pattern_seq(rng, params, c))
    return truths, nc, params


def test_pattern_oracles_and_honesty():
    for task, extra, nc in PATTERN_CASES:
        truths, nc2, params = _pattern_seqs(task, extra)
        for s in truths:
            validate_seq(s)
            # labels only on scored steps, class inside range, mask on slot 0 only
            m = s["lab_mask"] > 0.5
            assert np.all(m[:, 1:] == 0)
            scored = m[:, 0]
            assert np.all(s["lab_class"][~scored] == -1)
            assert np.all((s["lab_class"][scored] >= 0)
                          & (s["lab_class"][scored] < nc2))
            # pulse channel agrees with lab_event
            assert np.allclose(s["feat"][:, L.CH_EVENT].astype(np.float32),
                               s["lab_event"], atol=1e-3)
        refs = P.reference_scores(truths, task)
        assert np.isclose(refs["perfect"]["acc"], 1.0, atol=2e-3), (task, extra, refs)
        assert abs(refs["memoryless"]["acc"] - 1.0 / nc) <= 0.08, (task, refs)
        assert abs(refs["chance"]["acc"] - 1.0 / nc) <= 0.12, (task, refs)
        # no single frame may classify: scored-frame class means must coincide
        arr = {}
        for c in range(nc):
            sel = [s for s in truths
                   if int(s["lab_class"][s["lab_class"] >= 0][0]) == c]
            arr[c] = np.concatenate(
                [s["feat"][s["lab_mask"][:, 0] > 0.5].astype(np.float32)
                 for s in sel], 0)
        for ch in (L.CH_EVENT, L.CH_EVENT_TYPE):
            assert all(float(np.abs(arr[c][:, ch]).mean()) < 1e-6 for c in range(nc))
        gap = max(float(max(arr[c][:, 10].mean() for c in range(nc))
                        - min(arr[c][:, 10].mean() for c in range(nc))), 0.0)
        assert gap < 0.06, (task, "amplitude leaks class at scored frames", gap)
        print(f"  [ok] {task:26s} stretch={params.get('stretch', 1.0):g} "
              f"nc={nc} T~{np.mean([s['t'].shape[0] for s in truths]):.0f} "
              f"perfect={refs['perfect']['acc']:.3f} "
              f"memless={refs['memoryless']['acc']:.3f} "
              f"(chance {1.0 / nc:.3f}) amp_gap={gap:.3f}")


def test_pattern_paired_stretch():
    from .pattern import build_pattern_seq, pattern_defaults
    base = pattern_defaults("pattern_count")
    for factor in (2.0, 4.0):
        rng1 = np.random.default_rng(np.random.SeedSequence([SEED, 7, 1000]))
        s1 = build_pattern_seq(rng1, base, 2)
        p2 = dict(base)
        p2["stretch"] = factor
        rng2 = np.random.default_rng(np.random.SeedSequence([SEED, 7, 1000]))
        s2 = build_pattern_seq(rng2, p2, 2)
        # identical content, only the physical clock scales
        assert np.allclose(np.asarray(s2["t"]) / factor,
                           np.asarray(s1["t"]), rtol=1e-5), factor
        assert int((s1["feat"][:, L.CH_EVENT] > 0.5).sum()) == \
               int((s2["feat"][:, L.CH_EVENT] > 0.5).sum())
        assert int(s1["lab_class"][s1["lab_class"] >= 0][0]) == \
               int(s2["lab_class"][s2["lab_class"] >= 0][0])
        # a perfect counter still answers (= physical-time invariance of "knocks")
        refs = P.reference_scores([s2], "pattern_count")
        assert np.isclose(refs["perfect"]["acc"], 1.0, atol=1e-6)
    print("  [ok] 2x/4x stretch keeps the pulse pattern and the class; perfect "
          "memory still 1.0")


def test_pattern_shuffled_control():
    for task, thr in (("pattern_count", 0.7), ("pattern_order", 0.75),
                      ("pattern_rate", 0.75)):
        truths, nc, _ = _pattern_seqs(task, {}, n_per_class=10)
        rng = np.random.default_rng(2024)
        shuffled = [temporal_shuffle(t, rng) for t in truths]
        p0 = P.main_scores(task, P.make_probe_seqs(
            truths, _oracle(task, truths, "perfect")))
        p1 = P.main_scores(task, P.make_probe_seqs(
            shuffled, _oracle(task, shuffled, "perfect")))
        assert np.isclose(p0["acc"], 1.0, atol=2e-3), (task, p0)
        assert p1["acc"] < thr, (task, p0, p1)
        print(f"  [ok] shuffle control {task:16s} perfect acc "
              f"{p0['acc']:.3f} -> {p1['acc']:.3f} (chance {1.0 / nc:.2f})")


def test_pattern_plan():
    cfg = _cfg()
    specs = build_plan(cfg, tiny=True)
    names = {s.name for s in specs}
    pat = [s for s in specs if s.role.startswith("pattern")]
    base = [s for s in pat if s.role == "pattern_base"]
    assert len(base) == 5, [s.name for s in base]
    assert len(pat) >= 14, len(pat)
    for s in base:
        nc = int(s.params["n_classes"])
        for split in ("train", "val", "test"):
            assert s.counts[split] % nc == 0, (s.name, split, s.counts)
        assert s.eval_manifests, s.name
        for n in s.eval_manifests:
            assert n in names, (s.name, n)
    assert any(s.role == "pattern_stretch" for s in pat)
    assert any(s.role == "pattern_shuffled" for s in pat)
    # all pattern datasets keep the shared tensor/head layout
    for s in pat:
        assert s.task in ("pattern_count", "pattern_order", "pattern_rate",
                          "pattern_count_rate_mixed")
    print(f"  [ok] pattern plan: {len(base)} trainable + {len(pat) - len(base)} "
          f"eval-only datasets, balanced classes, eval manifests linked")


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _all_tests():
    return [
        test_layout_codec,
        test_task_shapes_and_invariants,
        test_gap_s_sweep_parameter,
        test_collate_roundtrip,
        test_generated_suite_manifests,
        test_determinism,
        test_oracles_main_metric,
        test_oracle_crosstalk,
        test_oracle_rollout,
        test_oracle_capacity,
        test_oracle_update_latency_false_updates,
        test_oracle_identity,
        test_oracle_event,
        test_recall_by_gap_and_capacity_buckets,
        test_irregular_variants,
        test_temporal_shuffle_control,
        test_gap_sweep_paired_latent,
        test_pattern_oracles_and_honesty,
        test_pattern_paired_stretch,
        test_pattern_shuffled_control,
        test_pattern_plan,
    ]


def main() -> int:
    import time
    fails = 0
    t0 = time.time()
    for fn in _all_tests():
        t = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t:.1f}s)", flush=True)
        except Exception as exc:  # pragma: no cover
            fails += 1
            import traceback
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
            traceback.print_exc()
    print(f"\n{'=' * 70}\n{len(_all_tests()) - fails}/{len(_all_tests())} "
          f"passed in {time.time() - t0:.1f}s\n{'=' * 70}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
