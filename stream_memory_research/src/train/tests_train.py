"""Trainer self-tests (CPU, tiny fixture).  ``python -m src.train.tests_train``

Covers the acceptance items that live in src/train/:
  * T_max chunking: random start, length <= T_max, deterministic per (seed, epoch)
  * corrupt npz containment: a run never dies, the bad seq is logged and skipped
  * aux objectives are strict no-ops when the core exposes no aux tensors
  * LR schedule: warmup then cosine decay, monotone after warmup
  * checkpoint round-trip + resume continues at the next epoch
  * metrics.json schema (DESIGN §4.1) and status transitions
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.common import contract
from src.common.contract import ROOT, RUNS, Manifest, SeqRecord, read_json, write_json
from src.train import data as tdata
from src.train.make_fixture import make_fixture

PASS = []
FAIL = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def _tmp_fixture(n=12) -> Path:
    out = ROOT / "runs" / "_fixture_unit"
    if out.exists():
        shutil.rmtree(out)
    return make_fixture(out, n=n, seed=1, feature_dim=16, n_classes=4)


def base_cfg(manifest: Path, run_id: str, **over) -> dict:
    cfg = {
        "run_id": run_id, "dataset": "fixture_action",
        "manifest": str(manifest), "task": "action",
        "model": {"name": "dummy", "hidden": 16, "cfg": {"mem_dim": 8, "n_slots": 4}},
        "heads": {"action": {"kind": "ce", "dim": 4, "ignore": -1}},
        "loss": {"task": {"action": 1.0}, "aux_num_scale": 0.1},
        "train": {"epochs": 2, "batch": 4, "lr": 0.002, "wd": 0.0, "amp": False,
                  "grad_clip": 1.0, "patience": 3, "aug": "mild", "T_max": 24,
                  "val_every": 1, "warmup_epochs": 1, "num_workers": 0,
                  "preload": True, "device": "cpu"},
        "eval": {"batch": 4, "robustness": ["dense", "stride2", "shuffled"],
                 "latency": False, "flops": True},
        "main_metric": "acc", "seed": 0,
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


# --------------------------------------------------------------------------- #
def test_chunking() -> None:
    mpath = _tmp_fixture()
    man = Manifest.load(mpath)
    recs = man.split("train")
    aug = tdata.TrainAugment("mild", T_max=12)
    ds = tdata.SafeStreamDataset(recs, augment=aug, feature_dim=man.feature_dim,
                                 seed=0, preload=True)
    lens, starts = [], []
    for i in range(len(recs)):
        out = ds.raw(i)
        lens.append(int(out["t"].shape[0]))
        starts.append(float(out["t"][0]))
    check("chunk.len<=T_max", max(lens) <= 12, f"lens={sorted(set(lens))}")
    ds.set_epoch(3)
    a = ds.raw(0)
    ds.set_epoch(3)
    b = ds.raw(0)
    same = (np.array_equal(np.asarray(a["feat"]), np.asarray(b["feat"]))
            and np.array_equal(np.asarray(a["t"]), np.asarray(b["t"])))
    check("chunk.deterministic_per_epoch", same)
    # different epochs should usually pick a different window: the first frame's
    # feature vector is the observable that reveals the random crop start
    # (t is re-normalised to 0 by subseq, so it cannot be used here).
    first_rows = set()
    for e in range(16):
        ds.set_epoch(e)
        row = np.asarray(ds.raw(0)["feat"])[0]
        first_rows.add(tuple(np.round(row.astype(float), 4)))
    check("chunk.random_start_varies", len(first_rows) > 1,
          f"{len(first_rows)} distinct first frames")


def test_corrupt() -> None:
    out = ROOT / "runs" / "_fixture_corrupt"
    if out.exists():
        shutil.rmtree(out)
    mpath = make_fixture(out, n=8, seed=2, feature_dim=8, n_classes=3, corrupt=2)
    man = Manifest.load(mpath)
    bad = [r for r in man.seqs if r.meta.get("corrupt")]
    clog = out / "corrupt.log"
    ds = tdata.SafeStreamDataset(man.seqs, feature_dim=man.feature_dim,
                                 corrupt_log=clog, preload=True)
    item = ds[bad[0].seq_id and 0]
    ok = ("feat" in item and int(item["feat"].shape[0]) == 2
          and len(item["labels"]) == 0)
    check("corrupt.sample_is_contained", ok)
    check("corrupt.log_written", clog.exists() and clog.stat().st_size > 0)


def test_aux_noop() -> None:
    from src.models import build_core
    from src.train.factory import StreamModel, build_heads
    from src.train.objectives import AuxObjectives, task_losses
    cfg = {"heads": {"action": {"kind": "ce", "dim": 4, "ignore": -1}},
           "train": {"head_hidden": 8}}
    # use_dt_feature=False: a per-frame MLP is then genuinely sampling-invariant,
    # so all five aux objectives must vanish on it (strict no-op rule).
    core = build_core("mlp", in_dim=8, hidden=16, use_dt_feature=False)
    heads = build_heads(cfg["heads"], core.out_dim or 16, cfg)
    model = StreamModel(core, heads, name="mlp")
    loss_cfg = {"aux": {"retrieval": 0.1, "crosstalk": 0.05, "time_consistency": 0.05,
                        "write_budget": 0.01, "long_horizon": 0.1},
                "aux_cfg": {}, "task": {"action": 1.0}, "aux_num_scale": 1.0}
    auxobj = AuxObjectives(loss_cfg, heads.spec, {"read_dim": 0})
    batch = tdata.make_dummy_batch(2, 6, 8)
    batch["labels"] = {"action": torch.zeros(2, 6, dtype=torch.int64)}
    logits, aux, out = model(batch)
    parts = auxobj.compute(model, batch, logits, aux, out)
    # The four aux terms that need core aux tensors must be EXACT no-ops.
    tensor_dependent = ("retrieval", "crosstalk", "write_budget", "long_horizon")
    all_zero = all(float(parts[k]) == 0.0 for k in tensor_dependent)
    # time_consistency is computed from the core output (always available); for a
    # per-frame MLP without dt features it is numerically ~0 (fp noise ~1e-9).
    tc = abs(float(parts.get("time_consistency", 0.0)))
    check("aux.noop_when_core_has_no_aux", all_zero and len(parts) == 5 and tc < 1e-6,
          f"parts={ {k: float(v) for k, v in parts.items()} }")
    total = auxobj.weighted(parts)
    check("aux.weighted_zero_value", abs(float(total)) < 1e-6, f"total={float(total)}")
    tl, tp = task_losses(logits, batch["labels"], heads.spec, batch["mask"],
                         {"action": 1.0})
    check("task.loss_is_finite", bool(torch.isfinite(tl)))


def test_schedule() -> None:
    from src.train.trainer import LRSchedule
    s = LRSchedule(1.0, total_steps=100, warmup_steps=10, kind="cosine", min_ratio=0.01)
    factors = [s.factor(i) for i in range(1, 101)]
    check("sched.warmup_rises", factors[0] < factors[9] and abs(factors[9] - 1.0) < 1e-9)
    tail = factors[10:]
    mono = all(tail[i] >= tail[i + 1] - 1e-12 for i in range(len(tail) - 1))
    check("sched.cosine_decays", mono and tail[-1] < 0.05, f"end={tail[-1]:.4f}")


def test_run_and_resume() -> None:
    from src.train.run_one import run_one
    mpath = _tmp_fixture()
    run_id = "unittest__dummy__base__s0"
    d = RUNS / run_id
    if d.exists():
        shutil.rmtree(d)
    cfg = base_cfg(mpath, run_id)
    m1 = run_one(cfg, smoke=False)
    check("run.status_complete", m1.get("status") == "complete", str(m1.get("error", ""))[:200])
    for f in ("config.json", "metrics.json", "curves.json", "train.log",
              "failure_cases.json", "per_source.json",
              "checkpoints/best.pt", "checkpoints/last.pt"):
        check(f"run.artifact:{f}", (d / f).exists())
    req = ["run_id", "status", "dataset", "task", "model", "variant", "seed", "params",
           "code_hash", "manifest_sha", "epochs_run", "wall_sec", "peak_vram_gb",
           "backoff", "backoffs", "main_metric", "main_metric_higher_is_better",
           "curves", "best"]
    missing = [k for k in req if k not in m1]
    check("run.schema_top_level", not missing, f"missing={missing}")
    best_req = ["epoch", "val", "test", "prefix", "per_seq_test", "per_seq_val",
                "per_gap", "crosstalk_matrix", "robust", "aux_stats",
                "latency_ms_per_step", "flops_per_step", "failure_cases", "rollout"]
    missing_b = [k for k in best_req if k not in (m1.get("best") or {})]
    check("run.schema_best", not missing_b, f"missing={missing_b}")
    rob = (m1.get("best") or {}).get("robust") or {}
    check("run.robust_has_dense_and_shuffled", "dense" in rob and "shuffled" in rob)
    check("run.epochs_run_2", int(m1.get("epochs_run", 0)) == 2)
    # ---- resume to 4 epochs --------------------------------------------------
    cfg2 = base_cfg(mpath, run_id, train={"epochs": 4})
    m2 = run_one(cfg2, resume=True)
    log = (d / "train.log").read_text(encoding="utf-8", errors="replace")
    check("resume.detected", "resumed after 2 completed epoch" in log, log[-400:])
    check("resume.epochs_run_4", int(m2.get("epochs_run", 0)) == 4,
          f"epochs_run={m2.get('epochs_run')}")
    check("resume.curves_continuous",
          m2.get("curves", {}).get("epoch") == [1, 2, 3, 4])
    cons = (m2.get("best") or {}).get("consistency") or {}
    check("resume.best_consistency_ok", bool(cons.get("ok", False)), str(cons))
    shutil.rmtree(d, ignore_errors=True)


def test_oom_backoff_and_json() -> None:
    """OOM ladder: halve batch -> shorten T_max -> disable AMP; JSON is NaN-free."""
    from src.train.trainer import Trainer, json_safe
    mpath = _tmp_fixture()
    run_id = "unittest_oom__dummy__base__s0"
    d = RUNS / run_id
    if d.exists():
        shutil.rmtree(d)
    cfg = base_cfg(mpath, run_id, train={"batch": 16, "T_max": 24, "min_batch": 4,
                                         "min_T": 8, "device": "cpu"})
    cfg["eval"] = {"batch": 4, "robustness": False, "latency": False, "flops": False}
    tr = Trainer(cfg, d, smoke=False, effective_run_id=run_id)
    tr.setup()
    b0 = tr.batch
    a1 = int(getattr(tr.train_ds.augment, "T_max", 0))
    ok1 = tr._oom_backoff(0)          # batch 16 -> 8
    ok2 = tr._oom_backoff(0)          # batch 8 -> 4 (min_batch)
    ok3 = tr._oom_backoff(0)          # batch at min -> shorten T
    a3 = int(getattr(tr.train_ds.augment, "T_max", 0))
    check("oom.halve_batch", ok1 and ok2 and tr.batch == 4, f"batch {b0}->{tr.batch}")
    check("oom.shorten_T", ok3 and a3 < a1, f"T_max {a1}->{a3}")
    check("oom.actions_recorded", len(tr.backoffs) >= 3
          and {b.get("action") for b in tr.backoffs}
          >= {"halve_batch", "shorten_T"}, str([b.get("action") for b in tr.backoffs]))
    tr._disable_amp("test")           # amp already off on CPU -> recorded no-op
    payload = json_safe({"a": float("nan"), "b": [1.0, float("inf"), None],
                         "c": {"d": float("-inf")}, "e": np.float32("nan")})
    check("json.no_nan", payload["a"] is None and payload["c"]["d"] is None
          and payload["e"] is None and payload["b"][1] is None)
    shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    print("== train self-tests ==", flush=True)
    test_chunking()
    test_corrupt()
    test_aux_noop()
    test_schedule()
    test_run_and_resume()
    test_oom_backoff_and_json()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
