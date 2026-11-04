"""Suite driver: expand a suite definition into runs, execute them, log everything.

This is the Lead's own tool.  It never contains model code — it only orchestrates
`src.train.run_one` in isolated subprocesses so that an OOM / NaN / crash in one
run cannot kill the suite.

Usage
-----
  python scripts/run_suite.py --suite main_v1                 # run everything not yet done
  python scripts/run_suite.py --suite main_v1 --dry-run       # list the plan
  python scripts/run_suite.py --suite main_v1 --only dsm      # substring filter on run_id
  python scripts/run_suite.py --suite main_v1 --workers 2 --max-runs 6
  python scripts/run_suite.py --suite main_v1 --seeds 0 --smoke

Suite file: configs/suites/<name>.json  (see configs/suites/main_v1.json)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402

PY = sys.executable
SUITES = ROOT / "configs" / "suites"
LOG_CSV = ROOT / "experiment_log.csv"
LOG_CSV_OVERRIDE = None


# --------------------------------------------------------------------------- #
def load_suite(name: str) -> dict:
    path = SUITES / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"suite file not found: {path}")
    return contract.read_json(path)


def expand(suite: dict, seeds: Optional[List[int]] = None,
           only: Optional[str] = None) -> List[dict]:
    """Expand suite -> list of run configs (dicts)."""
    seeds = seeds if seeds else list(suite.get("seeds", [0]))
    defaults = suite.get("defaults", {})
    runs: List[dict] = []
    for ds_name, ds in suite["datasets"].items():
        for model_entry in suite["models"]:
            if isinstance(model_entry, str):
                model_entry = {"name": model_entry}
            mname = model_entry["name"]
            variant = model_entry.get("variant", "base")
            for seed in seeds:
                run_id = contract.make_run_id(ds_name, mname, variant, seed)
                if only and only not in run_id:
                    continue
                cfg = {
                    "run_id": run_id,
                    "dataset": ds_name,
                    "manifest": ds.get("manifest", f"data_manifests/{ds_name}.json"),
                    "task": ds["task"],
                    "model": {
                        "name": mname,
                        "hidden": model_entry.get("hidden", 192),
                        "cfg": model_entry.get("cfg", {}),
                    },
                    "heads": ds["heads"],
                    "loss": {**defaults.get("loss", {}), **ds.get("loss", {}),
                             **model_entry.get("loss", {})},
                    "train": {**defaults.get("train", {}), **ds.get("train", {}),
                              **model_entry.get("train", {})},
                    "eval": {**defaults.get("eval", {}), **ds.get("eval", {})},
                    "seed": seed,
                    "main_metric": ds.get("main_metric", "acc"),
                    "suite": suite.get("name", "suite"),
                    "round": suite.get("round", "R1"),
                    "notes": model_entry.get("notes", ""),
                }
                runs.append(cfg)
    return runs


def _comparable(cfg: dict) -> dict:
    """The part of a run config that must match for a completed run to count.

    Without this, a run produced by an OLD suite version (e.g. different patience /
    epochs / batch) is silently counted as part of the new sweep -- which happened
    once and corrupted a comparison.  Now a mismatch means "not done".
    """
    keys = ("dataset", "manifest", "task", "model", "heads", "loss", "train", "seed")
    return {k: cfg.get(k) for k in keys}


def config_matches(cfg: dict) -> bool:
    p = ROOT / "runs" / cfg["run_id"] / "config.json"
    if not p.exists():
        return False
    try:
        old = contract.read_json(p)
    except Exception:
        return False
    a, b = _comparable(old), _comparable(cfg)
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def is_done(cfg: dict) -> bool:
    d = ROOT / "runs" / cfg["run_id"]
    p = d / "metrics.json"
    if not p.exists():
        return False
    try:
        m = contract.read_json(p)
    except Exception:
        return False
    if m.get("status") != "complete":
        return False
    if not config_matches(cfg):
        print(f"  [stale] {cfg['run_id']}: config differs from suite -> will re-run")
        return False
    return True


def effective_run_id(cfg: dict, smoke: bool) -> str:
    """run_one appends __smoke to the run dir so smoke results never count as real."""
    return cfg["run_id"] + "__smoke" if smoke else cfg["run_id"]


def write_config(cfg: dict) -> Path:
    d = ROOT / "runs" / cfg["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "config.json"
    contract.write_json(p, cfg)
    return p


def run_one(cfg: dict, smoke: bool, timeout_s: int = 6 * 3600) -> dict:
    """Execute one run in a subprocess.  Returns a result dict (never raises)."""
    cfg_path = write_config(cfg)
    eff_id = effective_run_id(cfg, smoke)
    cmd = [PY, "-m", "src.train.run_one", "--config", str(cfg_path)]
    if smoke:
        cmd.append("--smoke")
    log = ROOT / "runs" / cfg["run_id"] / "stdout.log"
    t0 = time.time()
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"\n=== {contract.now_str()} :: {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    wall = time.time() - t0
    out = {"run_id": eff_id, "rc": rc, "wall_s": wall}
    mp = ROOT / "runs" / eff_id / "metrics.json"
    if mp.exists():
        try:
            out["metrics"] = contract.read_json(mp)
        except Exception:
            out["metrics"] = {}
    return out


# --------------------------------------------------------------------------- #
def append_csv(cfg: dict, res: dict) -> None:
    """Append one row per run to experiment_log.csv (best-effort)."""
    import csv
    m = res.get("metrics") or {}
    best = m.get("best", {}) or {}
    val = best.get("val", {}) or {}
    test = best.get("test", {}) or {}
    mm = cfg.get("main_metric", "acc")
    row = {
        "timestamp": contract.now_str(),
        "round": cfg.get("round", ""),
        "suite": cfg.get("suite", ""),
        "dataset": cfg["dataset"],
        "task": cfg["task"],
        "model": cfg["model"]["name"],
        "variant": cfg["run_id"].split("__")[1] if "__" in cfg["run_id"] else "",
        "seed": cfg["seed"],
        "hidden": cfg["model"]["hidden"],
        "params": m.get("params", ""),
        "epochs": m.get("epochs_run", ""),
        "best_epoch": best.get("epoch", ""),
        "val_metric": val.get(mm, ""),
        "test_metric": test.get(mm, ""),
        "metric_name": mm,
        "wall_sec": round(res.get("wall_s", 0), 1),
        "peak_vram_gb": round(float(m.get("peak_vram_gb", 0) or 0), 3),
        "code_hash": m.get("code_hash", ""),
        "manifest": cfg.get("manifest", ""),
        "status": m.get("status", f"rc{res.get('rc')}"),
        "notes": cfg.get("notes", ""),
    }
    csv_path = Path(LOG_CSV_OVERRIDE) if LOG_CSV_OVERRIDE else LOG_CSV
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    cols = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--seeds", default=None, help="comma list, overrides suite seeds")
    ap.add_argument("--only", default=None, help="substring filter on run_id")
    ap.add_argument("--exclude", default=None, help="substring exclusion on run_id")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--redo", action="store_true", help="re-run even if complete")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--log-csv", default=None, help="override experiment_log.csv (for concurrent suites)")
    args = ap.parse_args()
    global LOG_CSV_OVERRIDE
    LOG_CSV_OVERRIDE = args.log_csv

    suite = load_suite(args.suite)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    runs = expand(suite, seeds=seeds, only=args.only)
    if args.exclude:
        runs = [r for r in runs if args.exclude not in r["run_id"]]
    if not args.redo:
        pending = [r for r in runs if not is_done(r)]
        skipped = len(runs) - len(pending)
        runs = pending
    else:
        skipped = 0

    # check manifests exist before starting
    missing = sorted({r["manifest"] for r in runs if not (ROOT / r["manifest"]).exists()})
    if missing:
        print("[warn] missing manifests (those datasets will fail):")
        for m in missing:
            print("   ", m)

    if args.max_runs:
        runs = runs[:args.max_runs]

    print(f"suite={args.suite} pending={len(runs)} skipped_done={skipped} workers={args.workers}")
    if args.dry_run:
        for r in runs:
            print(f"  {r['run_id']:60s} task={r['task']:8s} hidden={r['model']['hidden']} "
                  f"epochs={r['train'].get('epochs','?')}")
        return
    if not runs:
        print("nothing to do")
        return

    t_start = time.time()
    ok = fail = 0
    results = []
    if args.workers <= 1:
        for i, cfg in enumerate(runs):
            print(f"\n[{i + 1}/{len(runs)}] {cfg['run_id']}  {contract.now_str()}", flush=True)
            res = run_one(cfg, args.smoke)
            append_csv(cfg, res)
            results.append(res)
            m = res.get("metrics") or {}
            mm = cfg.get("main_metric", "acc")
            v = (m.get("best", {}).get("val", {}) or {}).get(mm)
            print(f"    rc={res['rc']} wall={contract.human_time(res['wall_s'])} "
                  f"val_{mm}={v} status={m.get('status')}", flush=True)
            if res["rc"] == 0:
                ok += 1
            else:
                fail += 1
                if args.stop_on_fail:
                    print("stop-on-fail")
                    break
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, c, args.smoke): c for c in runs}
            for k, fut in enumerate(as_completed(futs), 1):
                cfg = futs[fut]
                try:
                    res = fut.result()
                except Exception:
                    res = {"run_id": cfg["run_id"], "rc": -1, "wall_s": 0.0,
                           "traceback": traceback.format_exc()}
                append_csv(cfg, res)
                results.append(res)
                mm = cfg.get("main_metric", "acc")
                v = ((res.get("metrics") or {}).get("best", {}).get("val", {}) or {}).get(mm)
                print(f"[{k}/{len(runs)}] {cfg['run_id']:55s} rc={res['rc']} "
                      f"val_{mm}={v}", flush=True)
                ok += int(res["rc"] == 0)
                fail += int(res["rc"] != 0)

    dt = time.time() - t_start
    summary = {
        "suite": args.suite, "started": contract.now_str(), "wall_s": dt,
        "n_ok": ok, "n_fail": fail, "results": results,
    }
    outp = ROOT / "reports" / f"suite_{args.suite}_last.json"
    contract.write_json(outp, summary)
    print(f"\ndone: ok={ok} fail={fail} wall={contract.human_time(dt)} -> {outp}")


if __name__ == "__main__":
    main()
