# -*- coding: utf-8 -*-
"""pc/pack/a3_learning_curve.py - O-012 first tile: synth-scale learning curve for A3.

Runs train_mono_depth_pack at 3 synth scales (dev-only, fixed epochs) and records
synth heldout abs_rel + real physical metrics per scale. Output: findings/raw/
a3_learning_curve_<date>.json - the empirical basis for the cloud-scale decision.
"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

SCALES = [(400, 20), (1200, 30), (2400, 40)]

def main():
    t0 = time.perf_counter()
    points = []
    for n, ep in SCALES:
        out = Path(f"G:/robot_work_offload/cloud_train/jobs/a3_learning_curve/n{n}_e{ep}")
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "pc/end_side/train_mono_depth_pack.py", "--out", str(out), "--n-synth", str(n), "--epochs", str(ep)]
        print(f"[curve] scale n={n} epochs={ep} ...", flush=True)
        t1 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(paths.REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall = time.perf_counter() - t1
        rep = out / "train_eval_report.json"
        if proc.returncode != 0 or not rep.exists():
            print(f"[curve] n={n} FAILED rc={proc.returncode}", flush=True)
            print((proc.stdout or "")[-500:], flush=True)
            points.append({"n_synth": n, "epochs": ep, "failed": True})
            continue
        j = json.loads(rep.read_text(encoding="utf-8"))
        pt = {
            "n_synth": n, "epochs": ep,
            "synth_abs_rel": j["eval"]["synth_heldout_abs_rel"],
            "real_mad_mm": j["eval"]["real_physical"]["desk_depth_mad_mm"],
            "real_dark_ratio": j["eval"]["real_physical"]["dark_collapse_ratio"],
            "wall_s": round(wall, 1),
        }
        points.append(pt)
        print(f"[curve] n={n}: abs_rel={pt['synth_abs_rel']} mad={pt['real_mad_mm']} dark={pt['real_dark_ratio']} ({wall:.0f}s)", flush=True)
    ok = [p for p in points if not p.get("failed")]
    analysis = {}
    if len(ok) >= 2:
        rels = [p["synth_abs_rel"] for p in ok]
        mads = [p["real_mad_mm"] for p in ok]
        analysis = {
            "abs_rel_by_scale": rels,
            "mad_by_scale": mads,
            "rel_improvement_last_vs_first": round(rels[0] / max(rels[-1], 1e-4), 2),
            "saturating": bool(len(rels) >= 2 and rels[-1] > 0.85 * rels[-2]),
        }
    res = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scales": points,
        "analysis": analysis,
        "note": "real training frames fixed at 160 (dev-only); only synth scale varies. epochs scale mildly with data. abs_rel is synth-heldout exact-GT; mad/dark are real-frame physical metrics.",
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    out = paths.RAW / f"a3_learning_curve_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")

    print(f"[curve] wrote {out} in {res['runtime_s']}s", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
