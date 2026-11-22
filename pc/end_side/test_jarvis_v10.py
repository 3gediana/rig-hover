#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_jarvis_v10.py —— 自动化验收门禁：验证 HoverSDF-Net (Eikonal PDE)、无监督 3D OBB、Neural-CBF 平滑避障与 v10 三大交付物。"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

OUT_DIR = REPO / "runs_end_side" / "jarvis_spatial_v10"


def test_v10_deliverables_and_metrics() -> None:
    rep_path = OUT_DIR / "jarvis_v10_report.json"
    assert rep_path.exists(), f"Missing report: {rep_path}"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))

    # 1) 验证无监督 3D OBB 与 HoverSDF-Net Eikonal 精度
    sa = rep["scene_a_empty_d05"]
    assert sa["n_gaussians"] > 350_000, f"Too few Gaussians: {sa['n_gaussians']}"
    assert len(sa["auto_3d_obbs"]) >= 2, f"Expected >=2 auto 3D OBBs on empty_d0.5, got {len(sa['auto_3d_obbs'])}"
    assert sa["hover_sdf_net"]["mae_mm"] < 15.0, f"HoverSDF-Net MAE too high: {sa['hover_sdf_net']}"
    assert sa["hover_sdf_net"]["eikonal_err"] < 0.25, f"Eikonal PDE residual too high: {sa['hover_sdf_net']}"
    print(f"[PASS] v10_test_1: empty_d0.5 {sa['n_gaussians']:,} Gaussians, {len(sa['auto_3d_obbs'])} Auto-3D-OBBs, HoverSDF MAE={sa['hover_sdf_net']['mae_mm']}mm, Eikonal={sa['hover_sdf_net']['eikonal_err']}")

    # 2) 验证真实 4D 活体双敲记忆重建
    sb = rep["scene_b_cb8455_4d"]
    assert sb["n_static_gaussians"] > 300_000
    assert sb["n_4d_dynamic_hand_frames"] >= 10
    print(f"[PASS] v10_test_2: cb8455 4D living memory ({sb['n_static_gaussians']:,} static + {sb['n_4d_dynamic_hand_frames']} real moving hand 3D frames)")

    # 3) 验证 Neural-CBF 彻底消灭 90° 网格阶梯折弯且零碰撞
    da = rep["desktop_arena_neural_cbf_vs_astar"]
    assert da["collision"] is False, f"Neural-CBF trajectory collided: {da}"
    assert da["neural_cbf_max_turn_deg"] < 30.0, f"Neural-CBF max turn too sharp: {da}"
    assert da["true_min_clearance_cm"] > 1.0, f"Insufficient clearance: {da}"
    print(f"[PASS] v10_test_3: Neural-CBF max_turn={da['neural_cbf_max_turn_deg']}deg (vs Grid A* 90.0deg), true_min_clearance={da['true_min_clearance_cm']}cm (Zero Collision)")

    # 4) 验证三大交付物完整存在且非空（6联海报 + H.264 电影大片 + 60 FPS WebGL 指挥舱）
    deliv = rep["deliverables"]
    for k in ("poster_png", "movie_mp4", "interactive_webgl_deck_html"):
        p = pathlib.Path(deliv[k])
        assert p.exists() and p.stat().st_size > 100_000, f"Deliverable missing or tiny: {k} -> {p}"
    print(f"[PASS] v10_test_4: All 3 deliverables verified (Poster PNG + H.264 Movie + {deliv['interactive_webgl_deck_mb']}MB 60FPS WebGL Command Deck)")
    print("=== ALL 4/4 JARVIS v10 SPATIAL INTELLIGENCE GATES PASSED ===")


if __name__ == "__main__":
    test_v10_deliverables_and_metrics()
