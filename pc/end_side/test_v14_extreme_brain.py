"""v14 极限具身大脑与跨场景 3D 动作识别回归门禁（零开挂 / 严口径自动验收）。"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P1_REPORT = REPO / "runs_end_side" / "double_knock_3d_v2_extreme" / "double_knock_3d_v2_extreme_report.json"
P1_CKPT = REPO / "runs_end_side" / "double_knock_3d_v2_extreme" / "double_knock_3d_v2_extreme_deploy.pt"
V14_DIR = REPO / "runs_end_side" / "embodied_brain_v14"
V14_REPORT = V14_DIR / "embodied_brain_v14_report.json"
V14_PNG = V14_DIR / "showcase_v14_extreme_robot_brain.png"
V14_MP4 = V14_DIR / "demo_embodied_brain_v14.mp4"


def main() -> None:
    assert P1_REPORT.exists(), f"Missing Pillar 1 report: {P1_REPORT}"
    assert P1_CKPT.exists() and P1_CKPT.stat().st_size > 100_000, f"Missing Pillar 1 ckpt: {P1_CKPT}"
    assert V14_REPORT.exists(), f"Missing v14 report: {V14_REPORT}"
    assert V14_PNG.exists() and V14_PNG.stat().st_size > 80_000, f"Missing v14 poster: {V14_PNG}"
    assert V14_MP4.exists() and V14_MP4.stat().st_size > 100_000, f"Missing v14 mp4: {V14_MP4}"

    p1 = json.loads(P1_REPORT.read_text(encoding="utf-8"))
    c2 = p1["controlled_2d_v2"]
    c3 = p1["controlled_3d_v2"]

    # Pillar 1: 严格同构 2D-v2 vs 3D-v2 + 嵌套 OOF + 严 1-to-1 门禁
    assert c3["mean_win_auroc"] >= 0.89, f"3D-v2 AUROC too low: {c3['mean_win_auroc']}"
    assert c3["mean_win_auroc"] > c2["mean_win_auroc"], "3D-v2 must beat 2D-v2 on Win AUROC"
    assert c3["mean_dbl_auroc"] > c2["mean_dbl_auroc"], "3D-v2 must beat 2D-v2 on Dbl AUROC"
    assert c3["strict_1to1_stream"]["macro_strict_f1"] > c2["strict_1to1_stream"]["macro_strict_f1"], "3D-v2 must beat 2D-v2 on Strict Stream Macro F1"
    f_1f30d1 = next(f for f in c3["folds"] if "1f30d1" in f["val_take"])
    assert f_1f30d1["strict_stream"]["strict_f1"] >= 0.80, f"1f30d1 regression not fixed: {f_1f30d1['strict_stream']}"
    # 外部真实 YOLO26n-pose 抗干扰：9 段真机负样本 + 10 段 Charades 均为 0 误报
    ext_pose = p1["external_30clip_real_yolo_pose_roi_3d_v2"]
    assert ext_pose["3d_robot9_str_fp"] == 0, f"Real robot 9 neg FP must be 0, got {ext_pose['3d_robot9_str_fp']}"
    assert ext_pose["3d_charades10_str_fp"] == 0, f"Charades 10 neg FP must be 0, got {ext_pose['3d_charades10_str_fp']}"

    # Pillar 2: 5 项零开挂极限具身行为门禁
    rep = json.loads(V14_REPORT.read_text(encoding="utf-8"))

    e1 = rep["extreme1_real_video_to_behavior"]
    assert e1["fp_during_40s_real_distractors"] == 0, f"E1 distractor FP: {e1}"
    assert e1["cb8455_triggers_count"] >= 1, f"E1 no real trigger: {e1}"
    assert (not e1["collision"]) and e1["final_err_cm"] <= 5.0 and e1["lookback_face_err_deg"] <= 3.0, f"E1 nav fail: {e1}"

    e2 = rep["extreme2_kidnapping_recovery"]
    assert e2["baseline_no_recovery"]["final_err_cm"] >= 50.0, f"E2 kidnap not brutal enough: {e2['baseline_no_recovery']}"
    assert e2["v14_global_recovery"]["kidnap_recovered"] is True, f"E2 did not recover: {e2['v14_global_recovery']}"
    assert (not e2["v14_global_recovery"]["collision"]) and e2["v14_global_recovery"]["final_err_cm"] <= 8.0, f"E2 recovery err: {e2['v14_global_recovery']}"
    assert e2["v14_global_recovery"]["final_err_cm"] * 10.0 < e2["baseline_no_recovery"]["final_err_cm"], "E2 v14 must beat baseline by >10x"

    e3 = rep["extreme3_predictive_moving_obstacle"]
    e3_r = e3["baseline_reactive_snapshot"]
    e3_p = e3["v14_predictive_kalman_vo"]
    assert e3_p["first_visual_detect_step"] == 4, f"E3 detection latency > 0: {e3_p}"
    assert (not e3_p["collision"]) and e3_p["min_clearance_cm"] > e3_r["min_clearance_cm"], f"E3 predictive must clear wider than reactive: {e3_p} vs {e3_r}"
    assert e3_p["final_err_cm"] < e3_r["final_err_cm"] * 0.5, f"E3 predictive must arrive >2x closer than reactive: {e3_p} vs {e3_r}"
    assert 0.0 < e3_p["mean_obs_vel_err_mps"] < 0.08, f"E3 Kalman velocity err out of bounds: {e3_p}"

    e4 = rep["extreme4_multiframe_sudden_obstacle"]
    assert e4["first_visual_detect_step"] == 9, f"E4 detection latency > 0: {e4}"
    assert (not e4["collision"]) and e4["min_clearance_cm"] >= 2.5, f"E4 clearance not improved: {e4}"

    e5 = rep["extreme5_30ep_brutal_stress"]["summary"]
    assert e5["n_episodes"] == 30, f"E5 episode count != 30: {e5}"
    assert e5["median_arrival_err_cm"] <= 4.5, f"E5 median arrival err too high: {e5}"
    assert e5["mean_cam_pose_err_cm"] <= 4.0, f"E5 cam pose err too high: {e5}"
    assert e5["collision_count"] <= 1, f"E5 collisions too high: {e5}"

    print(
        f"[PASS] v14 Extreme Brain Gate PASSED! "
        f"P1(AUROC={c3['mean_win_auroc']}, 1f30d1={f_1f30d1['strict_stream']['strict_f1']}, Robot9FP=0) | "
        f"E1(arr={e1['final_err_cm']}cm, lb={e1['lookback_face_err_deg']}deg) | "
        f"E2(Base={e2['baseline_no_recovery']['final_err_cm']}cm -> v14={e2['v14_global_recovery']['final_err_cm']}cm in {e2['v14_global_recovery']['kidnap_recov_steps']} steps) | "
        f"E3(Pred={e3_p['final_err_cm']}cm/+{e3_p['min_clearance_cm']}cm vs React={e3_r['final_err_cm']}cm/+{e3_r['min_clearance_cm']}cm) | "
        f"E4(clear=+{e4['min_clearance_cm']}cm) | "
        f"E5(med={e5['median_arrival_err_cm']}cm, coll={e5['collision_count']}/30)"
    )


if __name__ == "__main__":
    main()
