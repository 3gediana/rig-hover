# -*- coding: utf-8 -*-
"""edge_survey_nav_v19.py — RIG-Hover "边缘勘测 + 任意点导航"场景（v19）。

v15 的悬崖本能是"看到了就拒跳"的一次性反射；v19 反过来把每一次悬崖观测
**累积成边界样本**（`edge_survey_samples` 钩子），蛇形/接近路径上持续估计
桌沿线 `y_cliff`，再拿估计出的边界做"任意点导航"的越界判定：

  - 安全点（距边 ≥14cm）：正常导航停靠；
  - 越界点（悬崖之外）：视觉拒跳，摇头抗议，永不坠落。

零真值：边界估计只用 `detect_visual_cliff_zero_oracle`（GeoNet 点云几何），
对照组 `enable_cliff_instinct=False` 直接坠落。

场景：
  S1 survey    : 驶向近边界目标，途中逐步累积边界样本（中位数 vs 真值 -0.36m）
  S2 nav_safe  : 任意点导航——安全点正常停靠
  S3 nav_void  : 任意点导航——目标在悬崖外 → 拒跳（对照组坠落）
  S4 survey_z  : 中途切换目标（两段折线勘测，覆盖率更高）

产物：runs_end_side/edge_survey_v19/{edge_survey_v19_report.json, demo_edge_v19.mp4}
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (HERE, REPO / "pc" / "train", REPO / "pc" / "perception"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from hd_moge_contour_engine import GEONET_CKPT_FOV_H_DEG, REAL_H_CAM_M  # noqa: E402
from hd_dynamic_contour_slam import HoverGeoNet240Frontend  # noqa: E402
from hover_fine_motion_policy_v3 import load_fine_motion_policy_v3  # noqa: E402
from living_desktop_brain_v15 import (  # noqa: E402
    CliffDesktopArena3D,
    V4_CKPT,
    build_initial_map_and_sdf,
    render_v15_cinema_video,
    run_living_brain_episode,
)

OUT_DIR = REPO / "runs_end_side" / "edge_survey_v19"
TRUE_CLIFF_Y = -0.36


def _strip(d: dict) -> dict:
    out = {k: v for k, v in d.items() if k != "vis_records"}
    if d.get("vis_records"):
        out["replay_frames"] = [
            {
                "step": r["step"], "tx": r["tx"], "ty": r["ty"], "tyaw": r["tyaw"],
                "head": r["head_deg"], "mood": r["mood"], "text": r["mood_text"],
                "tgt": r["target_xy"], "cyls": r["cyls"], "tube": r["kf_tube"],
                "clear_cm": r["clear_cm"], "cliff_y_est": r.get("cliff_y_est"),
            }
            for r in d["vis_records"]
        ]
    return out


def _survey_stats(samples: list[dict]) -> dict:
    if not samples:
        return {"n": 0, "median_err_cm": None, "p90_abs_err_cm": None}
    errs = [abs(s["y_est"] - TRUE_CLIFF_Y) for s in samples]
    return {
        "n": len(samples),
        "median_err_cm": round(float(np.median(errs)) * 100.0, 2),
        "p90_abs_err_cm": round(float(np.quantile(errs, 0.9)) * 100.0, 2),
        "median_y_est": round(float(np.median([s["y_est"] for s in samples])), 4),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    cli = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261123)
    np.random.seed(20261123)
    print("=== v19 EDGE SURVEY: map the rim, then navigate anywhere ===", flush=True)

    geo_fe = HoverGeoNet240Frontend(V4_CKPT, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG)
    policy_v3 = load_fine_motion_policy_v3("cpu")
    arena_std = CliffDesktopArena3D(h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG, cliff_y_min=None)
    arena_cliff = CliffDesktopArena3D(h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG, cliff_y_min=TRUE_CLIFF_Y)
    map_pack = build_initial_map_and_sdf(arena_std, geo_fe, seed=20, build_sdf=True)

    results: dict[str, dict] = {}
    clips: list[tuple[str, list[dict], float | None]] = []
    rep_p = OUT_DIR / "edge_survey_v19_report.json"
    if cli.only and rep_p.exists():
        results.update(json.loads(rep_p.read_text(encoding="utf-8")).get("scenarios", {}))

    def run(tag: str, label: str, survey: bool = False, arena=None, **kw) -> dict | None:
        if cli.only and cli.only not in tag:
            return None
        print(f"\n--- {tag}: {label} ---", flush=True)
        samples: list[dict] | None = [] if survey else None
        r = run_living_brain_episode(
            arena if arena is not None else arena_cliff, map_pack, geo_fe, policy_v3,
            kw.pop("init_target"), edge_survey_samples=samples, record_frames=True, **kw,
        )
        st = _survey_stats(samples) if survey else {}
        r["survey"] = st
        r["survey_samples"] = samples[:400] if survey else []
        print(
            f"  steps={r['steps']} | final_err={r['final_err_cm']}cm | "
            f"cliff_det={r['cliff_detected']}@s{r['cliff_refusal_step']} | margin={r['min_cliff_margin_cm']}cm | "
            f"fell={r['fell_off_cliff']} | survey={st}",
            flush=True,
        )
        results[tag] = _strip(r)
        clips.append((label, r["vis_records"], r.get("cliff_y_est_m")))
        return r

    # S1: 驶向近边界点的过程中累积边界样本（拒跳闩存依旧生效——安全兜底）
    run(
        "S1_survey_to_rim", "S1: Survey Leg -> accumulate rim samples en-route",
        init_target=(0.44, -0.27), survey=True,
        max_steps=90, seed=19101,
    )

    # S2: 任意点导航——安全点（距边 16cm）应正常停靠
    run(
        "S2_nav_anywhere_safe", "S2: Navigate Anywhere -> safe point 16cm off rim",
        init_target=(0.50, -0.20),
        max_steps=90, seed=19201,
    )

    # S3: 任意点导航——目标在悬崖外：v19 拒跳 vs 对照组坠落
    run(
        "S3a_nav_void_instinct", "S3a: Navigate to VOID target -> visual refusal (v19)",
        init_target=(0.46, -0.44),
        max_steps=90, seed=19301,
    )
    run(
        "S3b_nav_void_baseline", "S3b: VOID target, instinct OFF -> baseline falls",
        init_target=(0.46, -0.44), enable_cliff_instinct=False,
        max_steps=90, seed=19301,
    )

    # S4: 两段折线勘测（switch_target 提高沿边覆盖率）
    run(
        "S4_survey_zigzag", "S4: Zigzag survey (2-leg rim coverage)",
        init_target=(0.30, -0.26), survey=True,
        switch_target_event={"step": 40, "xy": (0.56, -0.24)},
        max_steps=110, seed=19401,
    )

    mp4_p = render_v15_cinema_video(clips, OUT_DIR / "demo_edge_v19.mp4")
    rep = {"scenarios": results, "true_cliff_y": TRUE_CLIFF_Y, "demo_mp4": str(mp4_p)}
    rep_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved v19 report -> {rep_p}\nSaved v19 cinema MP4 -> {mp4_p}", flush=True)


if __name__ == "__main__":
    main()
