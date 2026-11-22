"""DoubleKnock v5 ensemble: score-level fusion of all deployed 3D checkpoints.

Question: does fusing the causal streams of the five deployed checkpoints
(PhysNet v11, PhysNet strict-audit, PhysNetV2-extreme, FusionNetV3,
Sim2RealNetV4) beat the best single model under an identical strict 1-to-1
streaming protocol?

Honest boundary
---------------
The deploy checkpoints were trained on ALL 6 takes for export, so per-take
scores are IN-SAMPLE upper bounds -- they will NOT reproduce the LOO numbers
in the individual reports. The signal is the fusion-vs-single DELTA under an
identical protocol, plus whether fusion rescues the documented fold-level
calibration collapse (214950_1f30d1).

Two evaluation modes:
  A) deploy-realistic: each model uses its own stored calib_stream thresholds;
     fused scores use per-take LOO threshold calibration (grid search on the
     other 5 takes only -- no retraining, threshold-level honesty).
  B) fusion variants: mean(all5), mean(v2+v3+v4), max5, and 2-of-5 vote.

Outputs:
    runs_end_side/double_knock_3d_v5_ensemble/v5_ensemble_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

END_SIDE = pathlib.Path(__file__).resolve().parent
REPO = END_SIDE.parents[1]
sys.path.insert(0, str(END_SIDE))
sys.path.insert(0, str(REPO / "pc" / "train"))

from rigorous_audit_knock_ablation import eval_strict_1to1_events  # noqa: E402
from train_double_knock_3d_physnet import (  # noqa: E402
    DoubleKnock3DPhysNet,
    predict_seq_both_modes,
)
from train_double_knock_3d_v2_extreme import (  # noqa: E402
    DoubleKnock3DPhysNetV2,
    find_causal_triggers_colocal,
    load_v2_dataset,
    predict_seq_fast_exact,
)
from train_double_knock_3d_v3_fusion import DoubleKnock3DFusionNetV3  # noqa: E402
from train_double_knock_3d_v4_sim2real import DoubleKnock3DSim2RealNetV4  # noqa: E402

OUT_DIR = REPO / "runs_end_side" / "double_knock_3d_v5_ensemble"
WINDOW = 48

CKPTS = {
    "physnet_v11": ("physnet", REPO / "runs_end_side" / "double_knock_3d_v11" / "double_knock_3d_deploy.pt"),
    "physnet_strict": ("physnet", REPO / "runs_end_side" / "rigorous_audit_v13" / "double_knock_3d_strict_deploy.pt"),
    "v2_extreme": ("v2", REPO / "runs_end_side" / "double_knock_3d_v2_extreme" / "double_knock_3d_v2_extreme_deploy.pt"),
    "v3_fusion": ("v3", REPO / "runs_end_side" / "double_knock_3d_v3_fusion" / "double_knock_3d_v3_fusion_deploy.pt"),
    "v4_sim2real": ("v4", REPO / "runs_end_side" / "double_knock_3d_v4_sim2real" / "double_knock_3d_v4_sim2real_deploy.pt"),
}

# threshold grid identical to _calibrate_strict_on_train_only in rigorous_audit
SMOOTH_GRID = (3, 5)
IMP_GRID = (0.22, 0.28, 0.35, 0.42, 0.50)
DBL_GRID = (0.20, 0.30, 0.38, 0.48)


def load_model(kind: str, ckpt_path: pathlib.Path, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    if kind == "physnet":
        m = DoubleKnock3DPhysNet(in_ch=ck.get("in_ch", 4), kin_dim=ck.get("kin_dim", 16), hidden_dim=ck.get("hidden_dim", 64))
    elif kind == "v2":
        m = DoubleKnock3DPhysNetV2(use_3d=True, hidden_dim=ck.get("hidden_dim", 64))
    elif kind == "v3":
        m = DoubleKnock3DFusionNetV3(use_3d=True, hidden_dim=ck.get("hidden_dim", 64))
    else:
        m = DoubleKnock3DSim2RealNetV4(use_3d=True, hidden_dim=ck.get("hidden_dim", 64))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def causal_scores(kind: str, model: torch.nn.Module, d: dict, device: torch.device):
    if kind == "physnet":
        # predict_seq_both_modes does NOT normalize: PhysNet was trained on
        # float frames in [0,1] -- uint8 input saturates the encoder dead.
        vis = d["vis_4ch_u8"].astype(np.float32) / 255.0
        kin16 = np.concatenate([d["kin_18"][:, :8], d["kin_3d_raw"]], axis=1).astype(np.float32)
        _, _, ic, cc = predict_seq_both_modes(model, vis, kin16, WINDOW, device)
    else:
        _, _, ic, cc = predict_seq_fast_exact(model, d["vis_4ch_u8"], d["kin_18"], WINDOW, device)
    return ic, cc


def strict_eval(triggers, d) -> dict:
    return eval_strict_1to1_events(triggers, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])


def eval_scores(ic, cc, d, thr) -> dict:
    tr = find_causal_triggers_colocal(
        ic, cc, d["fps"], hot_uvf=None,
        smooth_frames=thr["smooth_frames"], imp_thr=thr["imp_thr"], double_thr=thr["double_thr"],
    )
    return strict_eval(tr, d)


def calibrate_on_takes(score_map: dict[str, tuple], train_ids: list[str], dataset: dict) -> dict:
    """Grid-search thresholds on train takes only (threshold-level LOO)."""
    best = None
    for sm in SMOOTH_GRID:
        for it in IMP_GRID:
            for dt in DBL_GRID:
                thr = {"smooth_frames": sm, "imp_thr": it, "double_thr": dt}
                hit = fp = gt = 0
                f1s = []
                for tid in train_ids:
                    ic, cc = score_map[tid]
                    st = eval_scores(ic, cc, dataset[tid], thr)
                    hit += st["hit"]; fp += st["fp"]; gt += st["total_gt"]
                    f1s.append(st["strict_f1"])
                prec = hit / (hit + fp) if hit + fp else 0.0
                rec = hit / max(gt, 1)
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                score = f1 - 0.02 * fp  # same penalty style as audit scripts
                if best is None or score > best["score"]:
                    best = {"score": round(score, 5), "train_micro_f1": round(f1, 4),
                            "train_hit": hit, "train_gt": gt, "train_fp": fp, **thr}
    return best


def aggregate(per_take: list[dict]) -> dict:
    h = sum(x["hit"] for x in per_take)
    g = sum(x["total_gt"] for x in per_take)
    fp = sum(x["fp"] for x in per_take)
    p = h / (h + fp) if h + fp else 0.0
    r = h / max(g, 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fa_min = sum(x["fp"] for x in per_take)  # filled below with fa/min mean
    return {"total_hit": h, "total_gt": g, "total_fp": fp,
            "micro_strict_prec": round(p, 4), "micro_strict_rec": round(r, 4),
            "micro_strict_f1": round(f1, 4),
            "macro_strict_f1": round(float(np.mean([x["strict_f1"] for x in per_take])), 4),
            "mean_fa_per_min": round(float(np.mean([x["fa_per_min"] for x in per_take])), 3)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v5-ens] device={device}", flush=True)

    t0 = time.perf_counter()
    dataset = load_v2_dataset(window=WINDOW)
    take_ids = sorted(dataset.keys())
    print(f"[v5-ens] dataset: {len(take_ids)} takes in {time.perf_counter()-t0:.1f}s", flush=True)

    # ---- per-model causal scores on every take ------------------------------
    scores: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for name, (kind, path) in CKPTS.items():
        if not path.exists():
            print(f"[v5-ens] SKIP {name}: {path} missing", flush=True)
            continue
        model = load_model(kind, path, device)
        scores[name] = {}
        tf = time.perf_counter()
        for tid in take_ids:
            ic, cc = causal_scores(kind, model, dataset[tid], device)
            scores[name][tid] = (ic, cc)
        print(f"[v5-ens] {name}: {len(take_ids)} takes scored in {time.perf_counter()-tf:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    names = list(scores.keys())
    report: dict[str, Any] = {
        "protocol": "in-sample streaming on all-6-takes deploy ckpts; thresholds "
                    "either stored calib_stream (per-model) or LOO grid on other 5 takes (fusion)",
        "honest_boundary": "deploy ckpts saw all takes in training -> absolute "
                           "numbers are UPPER BOUNDS; fusion-vs-single delta is the signal",
        "models": names,
    }

    # ---- A) per-model, own stored deploy thresholds --------------------------
    per_model = {}
    for name in names:
        ck = torch.load(CKPTS[name][1], map_location="cpu")
        thr = dict(ck.get("calib_stream") or {"smooth_frames": 3, "imp_thr": 0.35, "double_thr": 0.35})
        rows = [eval_scores(*scores[name][tid], dataset[tid], thr) for tid in take_ids]
        per_model[name] = {"thr_used": thr, "aggregate": aggregate(rows),
                           "per_take": dict(zip(take_ids, rows))}
        a = per_model[name]["aggregate"]
        print(f"[v5-ens] single {name}: hit {a['total_hit']}/{a['total_gt']} fp={a['total_fp']} "
              f"F1={a['micro_strict_f1']} FA/min={a['mean_fa_per_min']}", flush=True)
    report["per_model_deploy_thr"] = per_model

    # ---- B) fusion variants --------------------------------------------------
    def fuse_mean(ids_scores):  # mean over given model names
        return {tid: (np.mean([scores[n][tid][0] for n in ids_scores], axis=0),
                      np.mean([scores[n][tid][1] for n in ids_scores], axis=0))
                for tid in take_ids}

    def fuse_max(ids_scores):
        return {tid: (np.max([scores[n][tid][0] for n in ids_scores], axis=0),
                      np.max([scores[n][tid][1] for n in ids_scores], axis=0))
                for tid in take_ids}

    v234 = [n for n in names if n in ("v2_extreme", "v3_fusion", "v4_sim2real")]
    variants = {
        "mean_all5": fuse_mean(names),
        "mean_v234": fuse_mean(v234) if len(v234) >= 2 else None,
        "max_all5": fuse_max(names),
    }

    fusion_report = {}
    for vname, fmap in variants.items():
        if fmap is None:
            continue
        rows = []
        thrs = {}
        for tid in take_ids:
            train_ids = [x for x in take_ids if x != tid]
            thr = calibrate_on_takes(fmap, train_ids, dataset)
            thrs[tid] = thr
            rows.append(eval_scores(*fmap[tid], dataset[tid], thr))
        agg = aggregate(rows)
        fusion_report[vname] = {"aggregate": agg, "per_take": dict(zip(take_ids, rows)),
                                "loo_thresholds": thrs}
        print(f"[v5-ens] fusion {vname}: hit {agg['total_hit']}/{agg['total_gt']} fp={agg['total_fp']} "
              f"F1={agg['micro_strict_f1']} FA/min={agg['mean_fa_per_min']}", flush=True)
    report["fusion_variants"] = fusion_report

    # ---- C) 2-of-5 vote: union triggers kept only if >=2 models fired -------
    vote_rows = []
    for tid in take_ids:
        all_tr = []
        for n in names:
            ck_thr = dict(torch.load(CKPTS[n][1], map_location="cpu").get("calib_stream")
                          or {"smooth_frames": 3, "imp_thr": 0.35, "double_thr": 0.35})
            all_tr.append(find_causal_triggers_colocal(
                *scores[n][tid], dataset[tid]["fps"], hot_uvf=None,
                smooth_frames=ck_thr["smooth_frames"], imp_thr=ck_thr["imp_thr"],
                double_thr=ck_thr["double_thr"]))
        kept = []
        for i, trs in enumerate(all_tr):
            for tr in trs:
                agree = sum(
                    any(abs(o["second_s"] - tr["second_s"]) <= 0.6 for o in otrs)
                    for j, otrs in enumerate(all_tr) if j != i
                )
                if agree >= 1 and all(abs(k["second_s"] - tr["second_s"]) > 0.6 for k in kept):
                    kept.append(tr)
        kept.sort(key=lambda t: t["second_s"])
        vote_rows.append(strict_eval(kept, dataset[tid]))
    va = aggregate(vote_rows)
    report["vote_2of5"] = {"aggregate": va, "per_take": dict(zip(take_ids, vote_rows))}
    print(f"[v5-ens] vote 2/5: hit {va['total_hit']}/{va['total_gt']} fp={va['total_fp']} "
          f"F1={va['micro_strict_f1']} FA/min={va['mean_fa_per_min']}", flush=True)

    # ---- rescue check on the documented collapsed fold -----------------------
    for fold_focus in ("robot_20261107_214950_1f30d1",):
        if fold_focus in take_ids:
            report["collapsed_fold_focus"] = {
                "take": fold_focus,
                "per_model": {n: per_model[n]["per_take"].get(fold_focus) for n in names},
                "fusion": {v: fusion_report[v]["per_take"].get(fold_focus) for v in fusion_report},
            }

    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (OUT_DIR / "v5_ensemble_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v5-ens] report -> {OUT_DIR / 'v5_ensemble_report.json'}", flush=True)


if __name__ == "__main__":
    main()
