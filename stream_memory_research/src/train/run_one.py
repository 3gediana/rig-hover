"""Run one experiment.  THIS is the library entry point the Lead's suite calls.

    from src.train.run_one import run_one
    metrics = run_one(cfg, resume=False, smoke=False)   # never raises

CLI (matches scripts/run_suite.py):

    python -m src.train.run_one --config runs/<run_id>/config.json [--smoke] [--resume]

Guarantees
----------
* never raises: OOM, a corrupt sample, a NaN loss or a crash leaves a
  ``metrics.json`` with ``status="failed"`` (and the partial curve), and the
  process exits with code 1.
* every artifact lives under ``runs/<run_id>/``:
  config.json, metrics.json, curves.json, train.log, failure_cases.json,
  per_source.json, checkpoints/{best,last}.pt, corrupt_samples.log.
* ``status="complete"`` is written ONLY after ``best.test`` is filled.
* ``--smoke`` writes to ``runs/<run_id>__smoke`` so a smoke run can never be
  mistaken for a finished real run by the suite's skip logic.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from src.common.contract import (RUNS, Tee, now_str, read_json, write_json)


def _failure_blob(cfg: dict, effective: str, exc: BaseException, tb: str,
                  prev: Optional[dict], wall: float) -> dict:
    model = (cfg.get("model") or {}).get("name")
    # `error` carries the message AND the traceback tail so a failure can be
    # triaged from the sweep log alone; `traceback` keeps the full text.
    tb_tail = "\n".join(tb.strip().splitlines()[-12:])
    blob = {
        "run_id": effective, "status": "failed",
        "error": f"{type(exc).__name__}: {exc}\n{tb_tail}",
        "error_type": type(exc).__name__,
        "traceback": tb,
        "dataset": cfg.get("dataset"), "task": cfg.get("task"),
        "model": model, "variant": cfg.get("variant", "base"),
        "seed": cfg.get("seed", 0),
        "params": None, "code_hash": None, "manifest_sha": None,
        "epochs_run": 0, "wall_sec": round(wall, 2), "peak_vram_gb": None,
        "samples_per_sec": None, "backoff": [], "backoffs": [],
        "main_metric": cfg.get("main_metric") or "acc",
        "main_metric_higher_is_better": True,
        "curves": {}, "best": None, "config": cfg, "finished_at": now_str(),
    }
    if prev:
        blob.update({k: v for k, v in prev.items() if k not in
                     ("status", "error", "error_type", "traceback")})
        blob.update({"status": "failed",
                     "error": f"{type(exc).__name__}: {exc}\n{tb_tail}",
                     "error_type": type(exc).__name__, "traceback": tb,
                     "finished_at": now_str()})
    return blob


def run_one(cfg: dict, resume: bool = False, smoke: bool = False) -> dict:
    """Train + evaluate one configuration.  Returns the metrics dict (never raises)."""
    from src.train.trainer import Trainer  # local import: keeps CLI startup cheap

    cfg = dict(cfg or {})
    run_id = str(cfg.get("run_id") or
                 f"{cfg.get('dataset', 'unknown')}__"
                 f"{(cfg.get('model') or {}).get('name', 'model')}__s{cfg.get('seed', 0)}")
    effective = run_id + ("__smoke" if smoke else "")
    run_dir = RUNS / effective
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**cfg, "_effective_run_id": effective})
    t0 = time.time()
    old_stdout = sys.stdout
    tee = None
    try:
        tee = Tee(run_dir / "train.log")
        sys.stdout = tee
        trainer = Trainer(cfg, run_dir, smoke=smoke, effective_run_id=effective)
        metrics = trainer.run(resume=bool(resume))
        return metrics
    except BaseException as exc:                      # noqa: BLE001 - never raise
        tb = traceback.format_exc()
        try:
            print(tb, flush=True)
        except Exception:
            pass
        prev = None
        mp = run_dir / "metrics.json"
        if mp.exists():
            try:
                prev = read_json(mp)
            except Exception:
                prev = None
        blob = _failure_blob(cfg, effective, exc, tb, prev, time.time() - t0)
        try:
            write_json(mp, blob)
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return blob
    finally:
        sys.stdout = old_stdout
        if tee is not None:
            try:
                tee.f.close()
            except Exception:
                pass


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Run one stream-memory experiment")
    ap.add_argument("--config", required=True, help="path to a run config json")
    ap.add_argument("--resume", action="store_true", help="resume from last.pt")
    ap.add_argument("--smoke", action="store_true",
                    help="2%% of train / 2 epochs; writes runs/<id>__smoke")
    args = ap.parse_args(argv)
    cfg = read_json(args.config)
    metrics = run_one(cfg, resume=bool(args.resume), smoke=bool(args.smoke))
    status = (metrics or {}).get("status")
    print(f"[run_one] status={status} run_id={(metrics or {}).get('run_id')}", flush=True)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
