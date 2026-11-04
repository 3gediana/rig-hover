"""Evaluator / stats / probes self-tests.  ``python -m src.eval.tests_eval``

Covers the acceptance items that live in src/eval/:
  * evaluating the SAME checkpoint twice is bit-identical (metrics, curves,
    crosstalk matrix, aux stats; timing fields excluded by construction)
  * the robustness suite is exactly reproducible for the same seed
  * the evaluator refuses to see TRAIN split sequences (no leakage)
  * r2 / rollout / per-gap math matches a hand-computed oracle
  * failure_cases writes both copies; stats pairing has the right sign
  * streaming predict.py matches the batched forward (causality sanity)
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from src.common.contract import ROOT, RUNS, Manifest, read_json, write_json
from src.train import data as tdata
from src.train.make_fixture import make_fixture

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def _fixture():
    out = ROOT / "runs" / "_fixture_eval"
    if out.exists():
        shutil.rmtree(out)
    return make_fixture(out, n=12, seed=7, feature_dim=16, n_classes=4)


def _cfg(mpath) -> dict:
    return {
        "run_id": "evaltest__dummy__base__s0", "dataset": "fixture_action",
        "manifest": str(mpath), "task": "action",
        "model": {"name": "dummy", "hidden": 16, "cfg": {"mem_dim": 8, "n_slots": 4}},
        "heads": {"action": {"kind": "ce", "dim": 4, "ignore": -1}},
        "loss": {"task": {"action": 1.0}},
        "train": {"epochs": 1, "batch": 4, "T_max": 24, "aug": "none",
                  "num_workers": 0, "preload": True, "device": "cpu"},
        "eval": {"batch": 4, "latency": False, "flops": False,
                 "robustness": ["dense", "stride2", "drop50", "shuffled"]},
        "main_metric": "acc", "seed": 0,
    }


def _build(cfg):
    from src.train.factory import build_model
    torch.manual_seed(0)
    data = tdata.build_datasets(cfg, smoke=False)
    model = build_model(cfg, data["meta"]["feature_dim"])
    model.eval()
    loaders = {s: tdata.make_loader(data["datasets"][s], 4, shuffle=False, seed=0)
               for s in ("val", "test")}
    return data, model, loaders


def test_determinism() -> None:
    from src.eval.evaluator import deterministic_view, evaluate
    cfg = _cfg(_fixture())
    _data, model, loaders = _build(cfg)
    r1, seq1 = evaluate(model, loaders, "action", cfg, splits=("val", "test"),
                        return_seq=True)
    r2, seq2 = evaluate(model, loaders, "action", cfg, splits=("val", "test"),
                        return_seq=True)
    d1 = json.dumps(deterministic_view(r1), sort_keys=True, default=str)
    d2 = json.dumps(deterministic_view(r2), sort_keys=True, default=str)
    check("eval.bit_identical", d1 == d2,
          f"len={len(d1)}")
    # robustness lives on the last requested split (test when available)
    t1 = r1["test"]["robustness"]
    t2 = r2["test"]["robustness"]
    check("eval.robust_identical", json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True))
    check("eval.robust_has_shuffled", "shuffled" in t1 and t1["shuffled"].get("acc") is not None)
    check("eval.deterministic_view_small", "test" in deterministic_view(r1))


def test_no_leakage() -> None:
    from src.eval.evaluator import evaluate
    cfg = _cfg(_fixture())
    data, model, _loaders = _build(cfg)
    tr = tdata.make_loader(data["datasets"]["train"], 4, shuffle=False)
    raised = False
    try:
        evaluate(model, tr, "action", cfg, splits=("train",))
    except Exception as exc:
        raised = True
        check("eval.refuses_train_loader", "train" in str(exc).lower()
              or "leak" in str(exc).lower(), str(exc)[:120])
    check("eval.refuses_train", raised)


def test_metric_oracle() -> None:
    """r2 / rollout / gap from the evaluator vs a hand-written oracle."""
    from src.eval.evaluator import valid_entries, _seq_metric, rollout_curve
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(10, 3))
    tgt = rng.normal(size=(10, 3))
    tgt[rng.random((10, 3)) < 0.3] = 0.0
    # oracle r2 (non-zero targets are the selected entries)
    v = tgt != 0
    p, t = pred[v], tgt[v]
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2_oracle = 1.0 - ss_res / ss_tot
    per_seq = [{"seq_id": "s", "source_id": "src", "t": np.arange(10.0), "dt": np.ones(10) / 30,
                "heads": {"state": {"kind": "mse", "pred": pred, "target": tgt,
                                    "valid": valid_entries("mse", tgt)}}}]
    r2 = _seq_metric("mse", "r2", pred, tgt, valid_entries("mse", tgt))
    check("metric.r2_matches_oracle", abs(r2 - r2_oracle) < 1e-9,
          f"{r2:.6f} vs {r2_oracle:.6f}")
    roll = rollout_curve(per_seq, "state", "mse", "mae", horizons=(1, 3))
    h1 = float(np.abs(pred[:-1] - tgt[1:])[tgt[1:] != 0].mean())
    check("metric.rollout_h1_matches", abs(roll["1"]["mae"] - h1) < 1e-9,
          f"{roll['1']['mae']:.6f} vs {h1:.6f}")
    check("metric.rollout_has_1_3", set(roll) == {"1", "3"})


def test_failure_cases() -> None:
    from src.eval import failure_cases as fc
    run_id = "unittest_fc__dummy__base__s0"
    per_seq = []
    for i in range(6):
        n = 8
        pred = np.zeros(n, dtype=np.int64)
        tgt = np.zeros(n, dtype=np.int64)
        tgt[:] = 1 if i < 2 else 0
        per_seq.append({"seq_id": f"q{i}", "source_id": f"src{i}", "split": "test",
                        "T": n, "t": np.arange(n, dtype=float),
                        "dt": np.ones(n) * 0.033, "heads": {"action": {
                            "kind": "ce", "pred": pred, "target": tgt,
                            "valid": np.ones(n, dtype=bool)}}, "gap_arrays": {}})
    payload = fc.dump_failure_cases(run_id, per_seq, {"acc": 0.5}, "acc", True, k=3,
                                    run_dir=ROOT / "runs" / run_id)
    from src.common.contract import FAILURES
    p1 = FAILURES / f"{run_id}.json"
    p2 = ROOT / "runs" / run_id / "failure_cases.json"
    check("fc.both_files_written", p1.exists() and p2.exists())
    check("fc.worst_sorted", len(payload["worst"]) == 3
          and payload["worst"][0]["value"] is not None
          and payload["worst"][0]["value"] <= payload["worst"][-1]["value"])
    check("fc.has_diagnosis", all(w["diagnosis"] for w in payload["worst"]))
    shutil.rmtree(ROOT / "runs" / run_id, ignore_errors=True)
    p1.unlink(missing_ok=True)


def test_stats() -> None:
    from src.eval import stats
    root = ROOT / "runs" / "_stats_unit"
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(0)
    sources = [f"vid{i:03d}" for i in range(30)]
    base = rng.normal(0.5, 0.1, size=len(sources))
    for model, delta in (("cfc", 0.0), ("dsm", 0.08)):
        for seed in (0, 1, 2):
            d = root / f"synth_x__{model}__s{seed}"
            d.mkdir(parents=True)
            vals = base + delta + rng.normal(0, 0.01, size=len(sources))
            metrics = {"run_id": d.name, "status": "complete", "main_metric": "r2",
                       "best": {"epoch": 1, "val": {"r2": float(vals.mean())},
                                "test": {"r2": float(vals.mean())}},
                       "epochs_run": 1, "params": 1000}
            write_json(d / "metrics.json", metrics)
            write_json(d / "per_source.json", {
                "main_metric": "r2", "higher_is_better": True,
                "val": {"r2": dict(zip(sources, vals.tolist()))},
                "test": {"r2": dict(zip(sources, vals.tolist()))}})
    runs = stats.scan_runs(root)
    check("stats.scanned", len(runs) == 6, f"n={len(runs)}")
    rows = stats.comparison_table(runs, metric="r2", ref="cfc")
    dsm = [r for r in rows if r["model"] == "dsm"]
    check("stats.paired_diff_sign", bool(dsm) and dsm[0]["diff"] > 0.05,
          f"diff={dsm[0]['diff'] if dsm else None}")
    check("stats.paired_ci_excludes_0", bool(dsm) and dsm[0]["significant"] is True,
          f"ci=[{dsm[0]['diff_lo'] if dsm else None},{dsm[0]['diff_hi'] if dsm else None}]")
    check("stats.n_sources", bool(dsm) and dsm[0]["n_sources"] == 90,
          f"n_src={dsm[0]['n_sources'] if dsm else None}")
    md = stats.format_markdown(rows)
    check("stats.markdown_rows", md.count("\n") == len(rows) + 1, f"rows={len(rows)}")
    shutil.rmtree(root)


def test_predict() -> None:
    from src.eval.predict import StreamPredictor, measure_stream_latency
    cfg = _cfg(_fixture())
    data, model, _loaders = _build(cfg)
    ds = data["datasets"]["test"]
    seq = ds.raw(0)
    seq = {"feat": np.asarray(seq["feat"]), "t": np.asarray(seq["t"])}
    pred = StreamPredictor(model, cfg).run_sequence(seq)
    # batched forward reference
    rec = ds.records[0]
    batch = tdata.collate_stream([ds[0]])
    with torch.no_grad():
        logits, _aux, _out = model(batch)
    ref = logits["action"].argmax(-1)[0, :pred["n_steps"]].numpy()
    got = pred["pred"]["action"]
    agree = float((got == ref).mean())
    check("predict.stream_matches_batched", agree > 0.99,
          f"agree={agree:.3f} native={pred['native_step']} ms={pred['ms_per_step']:.3f}")
    lat = measure_stream_latency(model, cfg, n_steps=16)
    check("predict.latency_reported", lat["latency_ms_per_step"] is not None
          and lat["latency_ms_per_step"] > 0, str(lat))


def test_probes() -> None:
    from src.train.state_probe import (probe_crosstalk, probe_dt_importance,
                                       probe_mem_norm, probe_state_reset,
                                       probe_write_gate_hist)
    cfg = _cfg(_fixture())
    data, model, loaders = _build(cfg)
    nd = probe_mem_norm(model, loaders["test"], cfg, n_bins=8)
    check("probe.mem_norm_curve", nd.get("available") and len(nd["mem_norm_curve"]) == 8)
    wg = probe_write_gate_hist(model, loaders["test"], cfg)
    check("probe.write_gate_hist", wg.get("available") and "counts" in wg)
    ck = probe_crosstalk(model, loaders["test"], cfg)
    check("probe.crosstalk", ck.get("available") and len(ck["slot_matrix"]) == 4)
    dt = probe_dt_importance(model, loaders["test"], cfg, seed=0)
    check("probe.dt_importance", dt.get("available") and dt.get("normal") is not None, str(dt)[:120])
    sr = probe_state_reset(model, loaders["test"], cfg, chunk_lens=(4, 8))
    check("probe.state_reset", sr.get("available") and "none" in sr["chunks"]
          and "k4" in sr["chunks"], str(list(sr.get("chunks", {})))[:120])


def test_per_slot_mask() -> None:
    """Lead-pinned: lab_mask is a PER-SLOT mask for the `state` head only."""
    from src.eval.evaluator import valid_entries
    rng = np.random.default_rng(0)
    T, K, V = 6, 3, 4
    tgt = rng.normal(size=(T, K * V))
    lm = np.zeros((T, K))
    lm[0, 0] = 1
    lm[1, 1] = 1
    lm[2, :] = 1
    v = valid_entries("mse", tgt, lm)
    check("mask.per_slot_expand", v.shape == (T, K * V)
          and int(v.sum()) == int(lm.sum()) * V, f"{v.shape} n={int(v.sum())} "
          f"expected={int(lm.sum()) * V}")
    # ce must IGNORE lab_mask entirely (validity from -1 only)
    tgt_ce = np.array([0, 1, -1, 2, 0, 3])
    lm_ce = np.ones((6, 8))
    vc = valid_entries("ce", tgt_ce, lm_ce)
    check("mask.ce_ignores_labmask", np.array_equal(vc, tgt_ce != -1),
          f"{vc.tolist()}")
    # bce ignores it too
    vbc = valid_entries("bce", np.array([1.0, 0.0, -3.0]), np.ones((3, 4)))
    check("mask.bce_ignores_labmask", np.array_equal(vbc, np.array([True, True, True])))
    # non-dividing per-slot mask -> clear error
    bad = np.ones((T, 5))
    raised = False
    try:
        valid_entries("mse", tgt, bad)
    except ValueError as exc:
        raised = "divide" in str(exc)
    check("mask.bad_divisor_raises", raised)


def test_r2_and_gap() -> None:
    """rollout / per_gap must report r2 (per seq then macro) and an n per bucket."""
    from src.eval.evaluator import (gap_curve, rollout_curve, valid_entries,
                                    _seq_metric)
    rng = np.random.default_rng(3)
    per_seq = []
    for i in range(4):
        n = 12
        pred = rng.normal(size=(n, 4))
        tgt = rng.normal(size=(n, 4))
        age = np.linspace(0.0, 5.0, n)
        per_seq.append({
            "seq_id": f"s{i}", "source_id": f"src{i}", "T": n,
            "t": np.arange(n, dtype=float) / 10.0,
            "dt": np.ones(n) / 10.0,
            "gap_arrays": {"age": age},
            "heads": {"state": {"kind": "mse", "pred": pred, "target": tgt,
                                "valid": valid_entries("mse", tgt)}},
        })
    roll = rollout_curve(per_seq, "state", "mse", "r2", horizons=(1, 5))
    r2s = []
    for rec in per_seq:
        h = rec["heads"]["state"]
        pred, tgt, val = h["pred"][:-5], h["target"][5:], h["valid"][5:]
        r2s.append(_seq_metric("mse", "r2", pred, tgt, val))
    macro = float(np.nanmean(r2s))
    check("rollout.has_r2", roll["5"].get("r2") is not None)
    check("rollout.r2_macro_correct", abs(roll["5"]["r2"] - macro) < 1e-6,
          f"{roll['5']['r2']:.6f} vs {macro:.6f}")
    check("rollout.has_n", all("n" in v for v in roll.values()))
    gap = gap_curve(per_seq, "state", "mse", "r2")
    keys = list(gap)
    check("gap.uses_lab_age", gap[keys[0]].get("n", 0) > 0 and gap["0-0.1"].get("r2") is not None,
          f"keys={keys[:3]} first={gap[keys[0]]}")
    check("gap.bucket_has_n", all("n" in v for v in gap.values()))


def test_predict_real_cores() -> None:
    """Streaming (core.step) vs batched forward for every real core family."""
    from src.eval.predict import StreamPredictor
    from src.train.factory import build_model
    from src.train import data as tdata
    mpath = _fixture()
    man = Manifest.load(mpath)
    rec = man.split("test")[0]
    ds = tdata.SafeStreamDataset([rec], feature_dim=man.feature_dim, preload=True)
    batch = tdata.collate_stream([ds[0]])
    models = (("dsm", {"mem_dim": 8}), ("dsm_slot", {"n_slots": 4, "slot_dim": 4}),
              ("gru", {}), ("mamba", {}), ("mlp", {}))
    for name, mcfg in models:
        rcfg = {"model": {"name": name, "hidden": 16, "cfg": mcfg},
                "heads": {"action": {"kind": "ce", "dim": 4, "ignore": -1}},
                "train": {"head_hidden": 8}, "seed": 0}
        try:
            m = build_model(rcfg, man.feature_dim)
            m.eval()
            with torch.no_grad():
                lg, _aux, _out = m(batch)
            pred = StreamPredictor(m, rcfg).run_sequence(
                {"feat": np.asarray(ds.raw(0)["feat"]), "t": np.asarray(ds.raw(0)["t"])})
            ref = lg["action"][0, :pred["n_steps"]].argmax(-1).numpy()
            got = pred["pred"]["action"][:, 0]
            agree = float((ref == got).mean())
            check(f"predict.parity:{name}", agree >= 0.99,
                  f"agree={agree:.3f} native={pred['native_step']}")
        except Exception as exc:                     # pragma: no cover - defensive
            check(f"predict.parity:{name}", False,
                  f"{type(exc).__name__}: {str(exc)[:120]}")


def test_run_all_probes() -> None:
    """run_all_probes must return every probe and never raise on a normal model."""
    from src.train.state_probe import run_all_probes
    cfg = _cfg(_fixture())
    data, model, loaders = _build(cfg)
    train_loader = tdata.make_loader(data["datasets"]["train"], 4, shuffle=False)
    out = run_all_probes(model, loaders["val"], cfg, test_loader=loaders["test"],
                         train_loader=train_loader, chunk_lens=(4,))
    keys = set(out)
    check("probes.run_all_keys", {"dt_importance", "mem_norm", "write_gate_hist",
                                  "state_reset", "crosstalk", "linear_readout"} <= keys,
          str(sorted(keys)))
    check("probes.run_all_no_error",
          all(not (isinstance(v, dict) and v.get("error")) for v in out.values()),
          str({k: v.get("error") for k, v in out.items()
               if isinstance(v, dict) and v.get("error")}))


def main() -> int:
    print("== eval self-tests ==", flush=True)
    test_determinism()
    test_no_leakage()
    test_metric_oracle()
    test_per_slot_mask()
    test_r2_and_gap()
    test_failure_cases()
    test_stats()
    test_predict()
    test_predict_real_cores()
    test_probes()
    test_run_all_probes()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
