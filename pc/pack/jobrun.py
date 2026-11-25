# -*- coding: utf-8 -*-
"""pc/pack/jobrun.py - cloud-job package format + local rehearsal harness (O-013/F2).

A job package is a directory under PACK_ROOT/jobs/<name>/ with a job.json:
  {
    "name": "a3_depth_v1",
    "version": 1,
    "command": ["<py>", "pc/end_side/train_mono_depth.py", "--out", "{WORK}/out"],
    "inputs":  {"canon": "relative/or/absolute paths the job is allowed to read"},
    "expected_outputs": ["out/train_eval_report.json"],
    "budget": {"gpu_hours_est": 0.1, "wall_min_est": 15},
    "sealed_policy": "forbidden" | "final_eval_only",
    "env": {"PYTHONDONTWRITEBYTECODE": "1"}
  }
`<py>` in the command is replaced by the current interpreter; `{WORK}` by the
rehearsal workdir. Rehearsal = run the command in an isolated workdir with
BelowNormal priority via benign.py semantics, capture stdout/stderr, verify
expected outputs exist, and write a receipt JSON to findings/raw/.

Usage:
  python pc/pack/jobrun.py --job a3_depth_smoke          # rehearse
  python pc/pack/jobrun.py --job a3_depth_smoke --check  # validate package only
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402


def load_job(name: str) -> tuple[dict, Path]:
    jdir = paths.PACK_ROOT / "jobs" / name
    jf = jdir / "job.json"
    if not jf.exists():
        raise SystemExit(f"job package not found: {jf}")
    return json.loads(jf.read_text(encoding="utf-8")), jdir


def validate(job: dict) -> list[str]:
    errs = []
    for k in ("name", "command", "expected_outputs"):
        if k not in job:
            errs.append(f"missing key: {k}")
    if job.get("sealed_policy", "forbidden") not in ("forbidden", "final_eval_only"):
        errs.append("sealed_policy must be forbidden|final_eval_only")
    return errs


def rehearse(name: str) -> int:
    job, jdir = load_job(name)
    errs = validate(job)
    if errs:
        print("job.json invalid:", errs)
        return 2
    stamp = time.strftime("%Y%m%d_%H%M%S")
    work = jdir / "rehearsals" / stamp
    work.mkdir(parents=True, exist_ok=True)
    cmd = [c.replace("<py>", sys.executable).replace("{WORK}", str(work)) for c in job["command"]]
    env = dict(os.environ)
    env.update(job.get("env", {}))
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.perf_counter()
    print(f"[jobrun] rehearsing {name} in {work}", flush=True)
    print(f"[jobrun] cmd: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(paths.REPO), env=env,
                          stdout=open(work / "stdout.log", "w", encoding="utf-8"),
                          stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0
    missing = [o for o in job["expected_outputs"] if not (work / o.replace("{WORK}/", "")).exists()
               and not Path(o.replace("{WORK}", str(work))).exists()]
    receipt = {
        "job": name,
        "workdir": str(work),
        "wall_s": round(wall, 1),
        "returncode": proc.returncode,
        "expected_outputs": job["expected_outputs"],
        "missing_outputs": missing,
        "verdict": "PASS" if proc.returncode == 0 and not missing else "FAIL",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (work / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = paths.RAW / f"jobrun_{name}_{stamp}.json"
    raw.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[jobrun] verdict={receipt['verdict']} rc={proc.returncode} missing={missing} wall={wall:.0f}s", flush=True)
    print(f"[jobrun] receipt -> {raw}", flush=True)
    return 0 if receipt["verdict"] == "PASS" else 1


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        job, jdir = load_job(a.job)
        errs = validate(job)
        print(f"{a.job}: {'ok' if not errs else errs}")
        return 0 if not errs else 2
    return rehearse(a.job)


if __name__ == "__main__":
    raise SystemExit(main())
