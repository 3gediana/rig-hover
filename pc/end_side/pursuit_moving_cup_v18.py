# -*- coding: utf-8 -*-
"""pursuit_moving_cup_v18.py — RIG-Hover "追捕逃逸水杯"场景（v18）。

v15 里移动水杯是需要绕开的动态障碍；v18 反过来：被拖动的杯子本身就是目标。
机器人在零真值视觉闭环下用 `RobustDynamicObstacleTracker`（卡尔曼速度估计）
实时追踪杯子，每帧把目标更新为 **预测拦截点 - 接近方向 x 保持距**，追进
`rad+2.5cm` 触碰带即视为"捕获"，随后进入护送带 (0~24cm) 伴跑直到杯子停下。

对照组：`pursuit_lead_s=0.0` —— 无速度前瞻的"追尾"策略，永远落后一拍。

场景：
  S1 linear   : 杯子直线滑过桌面 (vx=-0.05, vy=+0.10)
  S2 zigzag   : 杯子走 L 折线路径（中途变向，考验速度重估计）
  S3 baseline : 同 S1 但 lead=0（无前瞻，预期追不上或很晚才碰到）
  S4 chaos    : 追捕中叠加突降静止障碍 + +70° 劫持

产物：runs_end_side/pursuit_v18/{pursuit_v18_report.json, demo_pursuit_v18.mp4}
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

OUT_DIR = REPO / "runs_end_side" / "pursuit_v18"


def _strip(d: dict) -> dict:
    out = {k: v for k, v in d.items() if k != "vis_records"}
    if d.get("vis_records"):
        out["replay_frames"] = [
            {
                "step": r["step"], "tx": r["tx"], "ty": r["ty"], "tyaw": r["tyaw"],
                "head": r["head_deg"], "mood": r["mood"], "text": r["mood_text"],
                "tgt": r["target_xy"], "cyls": r["cyls"], "tube": r["kf_tube"],
                "clear_cm": r["clear_cm"],
            }
            for r in d["vis_records"]
        ]
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    cli = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261123)
    np.random.seed(20261123)
    print("=== v18 PURSUIT: chase & escort the runaway cup ===", flush=True)

    geo_fe = HoverGeoNet240Frontend(V4_CKPT, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG)
    policy_v3 = load_fine_motion_policy_v3("cpu")
    arena_std = CliffDesktopArena3D(h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG, cliff_y_min=None)
    map_pack = build_initial_map_and_sdf(arena_std, geo_fe, seed=20, build_sdf=True)

    results: dict[str, dict] = {}
    clips: list[tuple[str, list[dict], float | None]] = []
    rep_p = OUT_DIR / "pursuit_v18_report.json"
    if cli.only and rep_p.exists():
        # --only 时把单场景结果合并回既有报告，避免覆盖掉其他场景
        results.update(json.loads(rep_p.read_text(encoding="utf-8")).get("scenarios", {}))

    def run(tag: str, label: str, **kw) -> dict | None:
        if cli.only and cli.only not in tag:
            return None
        print(f"\n--- {tag}: {label} ---", flush=True)
        r = run_living_brain_episode(
            arena_std, map_pack, geo_fe, policy_v3, kw.pop("init_target"),
            enable_cliff_instinct=False, pursuit_mode=True, record_frames=True, **kw,
        )
        print(
            f"  captured={r['pursuit_captured']}@s{r['capture_step']} | "
            f"escort_band={r['escort_band_ratio']} | min_edge={r['min_chase_edge_cm']}cm | "
            f"min_static_clear={r['min_static_clear_cm']}cm | vel_err={r['mean_moving_vel_err_mps']}m/s",
            flush=True,
        )
        results[tag] = _strip(r)
        clips.append((label, r["vis_records"], None))
        return r

    # S1: 直线逃逸
    run(
        "S1_linear_chase", "S1: Linear Escape Cup -> Kalman-Lead Pursuit + Escort",
        init_target=(0.50, -0.30),
        moving_obs_cfg={"start_step": 4, "x0": 0.52, "y0": -0.36, "vx": -0.055, "vy": 0.105, "rad": 0.036, "height": 0.120},
        max_steps=110, seed=18101,
    )

    # S2: L 折线变向逃逸
    run(
        "S2_zigzag_chase", "S2: Zigzag L-Turn Escape -> Velocity Re-Estimation",
        init_target=(0.50, -0.30),
        moving_obs_cfg={
            "start_step": 4, "rad": 0.036, "height": 0.120, "speed": 0.105,
            "waypoints": [(0.52, -0.36), (0.28, -0.14), (0.62, 0.10)],
        },
        max_steps=130, seed=18201,
    )

    # S3: 对照组——无速度前瞻
    run(
        "S3_baseline_no_lead", "S3: BASELINE No-Velocity-Lead (chases raw position)",
        init_target=(0.50, -0.30),
        moving_obs_cfg={"start_step": 4, "x0": 0.52, "y0": -0.36, "vx": -0.055, "vy": 0.105, "rad": 0.036, "height": 0.120},
        pursuit_lead_s=0.0,
        max_steps=110, seed=18101,
    )

    # S5: 只有突降障碍（隔离变量——验证"检测→目标层规避"链路，无 kidnap 干扰）
    run(
        "S5_drop_only", "S5: Pursuit + Sudden Drop Obstacle (no kidnap)",
        init_target=(0.50, -0.30),
        moving_obs_cfg={"start_step": 4, "x0": 0.52, "y0": -0.36, "vx": -0.055, "vy": 0.105, "rad": 0.036, "height": 0.120},
        static_drop_obs={"step": 6, "cx": 0.34, "cy": -0.15, "rad": 0.036, "height": 0.115},
        max_steps=140, seed=18401,
    )

    # S4: 追捕 + 突降障碍 + 劫持（地狱版；kidnap 提前到捕获前 s12——追击中段被抱走再复位）
    run(
        "S4_chaos_pursuit", "S4: Chaos Pursuit (drop obstacle + +70deg kidnap mid-chase)",
        init_target=(0.50, -0.30),
        moving_obs_cfg={"start_step": 4, "x0": 0.52, "y0": -0.36, "vx": -0.055, "vy": 0.105, "rad": 0.036, "height": 0.120},
        static_drop_obs={"step": 6, "cx": 0.34, "cy": -0.15, "rad": 0.036, "height": 0.115},
        kidnap_event={"step": 12, "dx": -0.06, "dy": 0.07, "dyaw": 70.0},
        max_steps=140, seed=18401,
    )

    mp4_p = render_v15_cinema_video(clips, OUT_DIR / "demo_pursuit_v18.mp4")
    rep = {"scenarios": results, "demo_mp4": str(mp4_p)}
    rep_p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved v18 report -> {rep_p}\nSaved v18 cinema MP4 -> {mp4_p}", flush=True)


if __name__ == "__main__":
    main()
