"""Serial driver for the pixel-to-event experiment matrix.

One GPU process at a time: on this machine concurrent torch processes exhaust the
commit limit and produce fake crashes (``WinError 1455`` / ``bad allocation``),
so the whole matrix runs in a single serial stream.  Runs that already have a
``metrics.json`` are skipped, so the driver is safe to kill and restart.

    python scripts/run_pixel_suite.py --models pixel_event gru --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\robot_work\stream_memory_research")
PY = r"D:\smartcar\.venv\Scripts\python.exe"
OUT = ROOT / "runs_pixel"
LOG = ROOT / "logs" / "pixel_suite.csv"


def run_id(data: str, model: str, window: int, seed: int, tag: str = "") -> str:
    return f"{data}__{model}__w{window}__s{seed}" + (f"__{tag}" if tag else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pixel_event"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--data", default="charades_pixel")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eval", action="store_true", default=True)
    ap.add_argument("--no-eval", dest="eval", action="store_false")
    ap.add_argument("--event-head", dest="event_head", default="direct",
                    choices=["direct", "hazard"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--scout-pretrained", dest="scout_pretrained", default=True,
                    action="store_true")
    ap.add_argument("--no-scout-pretrained", dest="scout_pretrained",
                    action="store_false")
    a = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "logs" / "pixel_suite.lock"
    if lock.exists():
        try:
            old = int(lock.read_text().strip())
        except Exception:
            old = -1
        alive = old > 0 and any(
            int(p) == old for p in __import__("subprocess").run(
                ["tasklist", "/FI", f"PID eq {old}", "/NH"],
                capture_output=True, text=True).stdout.split() if p.isdigit())
        if alive:
            print(f"[abort] another suite is running (pid {old}); "
                  f"delete {lock} if that is stale", flush=True)
            return 2
    lock.write_text(str(__import__("os").getpid()), encoding="utf-8")
    if not LOG.exists():
        LOG.write_text("ts,run_id,stage,rc,wall_s,note\n", encoding="utf-8")

    for model in a.models:
        for seed in a.seeds:
            rid = run_id(a.data, model, a.window, seed, a.tag)
            d = OUT / rid
            d.mkdir(parents=True, exist_ok=True)
            mj = d / "metrics.json"
            trained = False
            if mj.exists():
                try:
                    trained = bool(json.loads(mj.read_text(encoding="utf-8")).get("test"))
                except Exception:
                    trained = False
            if trained and (not a.eval or (d / "eval.json").exists()):
                print(f"[skip] {rid} already complete", flush=True)
                continue
            if trained:
                # trained but never evaluated: only re-run the evaluation step
                print(f"[eval-only] {rid}", flush=True)
                with open(d / "eval.log", "w", encoding="utf-8") as f:
                    rc2 = subprocess.call([PY, str(ROOT / "scripts" / "eval_pixel_event.py"),
                                          "--run", rid, "--data", a.data],
                                         stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
                print(f"[eval ] {rid} rc={rc2}", flush=True)
                continue
            t0 = time.time()
            cmd = [PY, str(ROOT / "scripts" / "train_pixel_event.py"),
                   "--model", model, "--data", a.data, "--window", str(a.window),
                   "--batch", str(a.batch), "--epochs", str(a.epochs),
                   "--lr", str(a.lr), "--patience", str(a.patience),
                   "--seed", str(seed), "--event-head", a.event_head]
            cmd.append("--scout-pretrained" if a.scout_pretrained
                       else "--no-scout-pretrained")
            print(f"[train] {rid}  ({time.time() - t0:.0f}s)", flush=True)
            with open(d / "train.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
            wall = time.time() - t0
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{rid},train,{rc},"
                        f"{wall:.0f},\n")
            if rc != 0:
                print(f"[fail] {rid} rc={rc} (see train.log)", flush=True)
                continue
            if a.eval:
                with open(d / "eval.log", "w", encoding="utf-8") as f:
                    rc2 = subprocess.call([PY, str(ROOT / "scripts" / "eval_pixel_event.py"),
                                          "--run", rid, "--data", a.data],
                                         stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{rid},eval,{rc2},"
                            f"{time.time() - t0 - wall:.0f},\n")
                print(f"[eval ] {rid} rc={rc2}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
