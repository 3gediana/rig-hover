"""Robot-domain experiment: leave-one-session-out CV on the ten RIG-Hover recordings.

This is the only robot-domain pixel data that exists: ten 20 s recordings of the
on-board camera (240x240 grey, ~10 fps), one scripted behaviour each
(`evidence/sessions/`).  There are **no per-event boundary or count labels** in
this footage, so the honest question that can be asked here is:

    can a causal pixel model recover, per frame, which scripted behaviour is
    happening and whether a person is present -- on an unseen session?

Ten sessions means the only defensible protocol is leave-one-session-out: train on
nine, evaluate on the tenth, repeat.  A single train/test split of ten clips would
be noise.

    python scripts/run_robot_experiment.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:\robot_work\stream_memory_research")
PY = r"D:\smartcar\.venv\Scripts\python.exe"
OUT = ROOT / "runs_pixel"
REP = ROOT / "reports"
MODELS = ["pixel_event", "frame_cnn"]
EPOCHS = 6
WINDOW = 32
BATCH = 8


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.data.pixel_data import robot_manifest
    dirs = [r["dir"] for r in robot_manifest()]
    res = {"n_sessions": len(dirs), "models": MODELS, "window": WINDOW,
           "epochs": EPOCHS, "folds": {}}

    for model in MODELS:
        rows = []
        for k, held in enumerate(dirs):
            rid = f"robot__{model}__w{WINDOW}__s0__f{k}"
            d = OUT / rid
            d.mkdir(parents=True, exist_ok=True)
            mj = d / "metrics.json"
            if not mj.exists():
                cmd = [PY, str(ROOT / "scripts" / "train_pixel_event.py"),
                       "--model", model, "--data", "robot", "--fold", str(k),
                       "--window", str(WINDOW), "--batch", str(BATCH),
                       "--epochs", str(EPOCHS), "--patience", str(EPOCHS),
                       "--seed", "0", "--scout-pretrained"]
                print(f"[train] {rid}", flush=True)
                with open(d / "train.log", "w", encoding="utf-8") as f:
                    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                         cwd=str(ROOT))
                if rc != 0:
                    print(f"[fail ] {rid} rc={rc}", flush=True)
                    rows.append({"held_out": held, "error": f"train rc={rc}"})
                    continue
            cmd = [PY, str(ROOT / "scripts" / "eval_pixel_event.py"),
                   "--run", rid, "--data", "robot", "--held", held]
            with open(d / "eval.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                     cwd=str(ROOT))
            if rc != 0:
                rows.append({"held_out": held, "error": f"eval rc={rc}"})
                print(f"[efail] {rid}", flush=True)
                continue
            e = json.loads((d / "eval.json").read_text(encoding="utf-8"))
            fold = e["folds"][0] if e.get("folds") else {}
            rows.append({"held_out": held, "gesture_acc": fold.get("gesture_acc"),
                         "presence_f1": fold.get("presence_f1"),
                         "params": e.get("params")})
            print(f"[ok   ] {rid} gesture_acc={fold.get('gesture_acc')} "
                  f"presence_f1={fold.get('presence_f1')}", flush=True)
        ok = [r for r in rows if r.get("gesture_acc") is not None]
        res["folds"][model] = {
            "rows": rows,
            "gesture_acc_mean": sum(r["gesture_acc"] for r in ok) / max(1, len(ok)),
            "gesture_acc_std": (sum((r["gesture_acc"] - sum(x["gesture_acc"] for x in ok)
                                     / max(1, len(ok))) ** 2 for r in ok)
                                / max(1, len(ok))) ** 0.5 if ok else None,
            "n_ok": len(ok),
        }
    # chance levels
    chance = 1.0 / len(dirs)
    res["chance_gesture_acc"] = chance
    res["chance_presence_f1"] = (2 * 0.9) / (1 + 0.9)
    (REP / "robot_results.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n=== robot leave-one-session-out ===")
    print(f"chance gesture acc = {chance:.3f}")
    for m in MODELS:
        f = res["folds"].get(m, {})
        if f.get("n_ok"):
            print(f"  {m:12s} gesture_acc={f['gesture_acc_mean']:.3f} "
                  f"+- {f['gesture_acc_std']:.3f} (n={f['n_ok']})")
    print(f"wrote {REP / 'robot_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
