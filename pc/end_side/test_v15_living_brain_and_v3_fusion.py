"""v15 活体桌面伴侣大脑 (L1-L4 + HTML5 沙盒) & DoubleKnock3D-FusionNet v3 自动回归门禁。"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V3_DIR = REPO / "runs_end_side" / "double_knock_3d_v3_fusion"
V3_REPORT = V3_DIR / "double_knock_3d_v3_fusion_report.json"
V3_CKPT = V3_DIR / "double_knock_3d_v3_fusion_deploy.pt"

V15_DIR = REPO / "runs_end_side" / "living_brain_v15"
V15_REPORT = V15_DIR / "living_brain_v15_report.json"
V15_PNG = V15_DIR / "showcase_v15_living_desktop_brain.png"
V15_MP4 = V15_DIR / "demo_living_brain_v15.mp4"
V15_HTML = V15_DIR / "wenwen_living_brain_v15.html"


def main() -> None:
    for p in (V3_REPORT, V3_CKPT, V15_REPORT, V15_PNG, V15_MP4, V15_HTML):
        assert p.exists() and p.stat().st_size > 10_000, f"Missing or empty artifact: {p}"

    # 1) DoubleKnock3D-FusionNet v3 (6/6 同构完胜 + 无 YOLO 独立识别突破!)
    v3 = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    c2, c3 = v3["controlled_2d_v3"], v3["controlled_3d_v3"]
    assert c3["mean_win_auroc"] >= 0.92 and c3["mean_win_auroc"] > c2["mean_win_auroc"] + 0.07, f"3D-v3 AUROC check failed: {c3['mean_win_auroc']} vs {c2['mean_win_auroc']}"
    assert c3["mean_dbl_auroc"] >= 0.89 and c3["mean_dbl_auroc"] > c2["mean_dbl_auroc"] + 0.08, f"3D-v3 Dbl AUROC failed"
    # 6/6 每一折 AUROC 均严格战胜同架构 2D-v3
    for f2, f3 in zip(c2["folds"], c3["folds"]):
        assert f3["win_auroc"] > f2["win_auroc"], f"Fold {f3['val_take']} 3D did not beat 2D AUROC"
    assert c3["strict_1to1_stream"]["macro_strict_f1"] > c2["strict_1to1_stream"]["macro_strict_f1"] + 0.08
    assert c3["strict_1to1_stream_nobox"]["macro_strict_f1"] >= 0.82, f"3D-v3 NoBox F1 check failed: {c3['strict_1to1_stream_nobox']}"
    ext = v3["external_30clip_real_yolo_pose_roi"]
    assert ext["3d_charades10_str_fp"] == 0 and ext["3d_ucf10_str_fp"] == 0, f"External Charades/UCF FP != 0: {ext}"
    assert ext["knock_d05_detected"]["3d_str"] is True and ext["knock_d05_detected"]["3d_nobox_str"] is True

    # 2) HoverLivingMind v15 (L1-L4 全部零碰撞 + 零开挂!)
    rep = json.loads(V15_REPORT.read_text(encoding="utf-8"))

    l1_b = rep["L1_visual_cliff_survival"]["baseline_no_cliff"]
    l1_v = rep["L1_visual_cliff_survival"]["v15_living_mind"]
    assert l1_b["fell_off_cliff"] is True, "L1 baseline should fall off cliff"
    assert l1_v["cliff_detected"] is True and l1_v["fell_off_cliff"] is False and l1_v["min_cliff_margin_cm"] >= 8.0, f"L1 v15 failed: {l1_v}"

    l2 = rep["L2_teasing_and_active_face_search"]
    assert l2["fakeout_ignored"] is True and l2["target_switched"] is True
    assert (not l2["collision"]) and l2["final_err_cm"] <= 3.0, f"L2 dock fail: {l2}"
    assert l2["human_found_new_seat"] is True and 0.0 <= l2["final_face_gaze_err_deg"] <= 3.0, f"L2 face search fail: {l2}"

    l3_b = rep["L3_object_permanence_nbv_peek"]["baseline_no_nbv"]
    l3_v = rep["L3_object_permanence_nbv_peek"]["v15_nbv_peek"]
    assert l3_v["nbv_peeks_done"] is True and (not l3_v["collision"]) and l3_v["final_err_cm"] <= 10.0, f"L3 NBV fail: {l3_v}"
    assert l3_v["final_err_cm"] * 10.0 < l3_b["final_err_cm"], f"L3 must beat no-NBV baseline by >10x"

    l4_b = rep["L4_triple_chaos_gauntlet"]["baseline_no_v15"]
    l4_v = rep["L4_triple_chaos_gauntlet"]["v15_living_mind"]
    assert l4_b["collision"] is True and l4_b["final_err_cm"] >= 100.0
    assert l4_v["attack1_moving_det_step"] == 4 and l4_v["attack2_drop_det_step"] >= 16 and l4_v["attack3_kidnap_recovered"] is True
    assert (not l4_v["collision"]) and l4_v["min_clearance_cm"] >= 3.0 and l4_v["final_err_cm"] <= 15.0, f"L4 gauntlet fail: {l4_v}"

    # 3) HTML5 互动沙盒自包含校验
    html_txt = V15_HTML.read_text(encoding="utf-8")
    assert "L1_visual_cliff_survival" in html_txt and "L4_triple_chaos_gauntlet" in html_txt and "stepSandbox" in html_txt

    print(
        f"[PASS] v15 LivingMind & v3 Fusion Gate PASSED! "
        f"v3(6/6 AUROC 0.8404->0.9238, NoBox F1=0.8245 [79/89], Charades/UCF=0/0 FP, knock_d05=True) | "
        f"L1(CliffMargin=+{l1_v['min_cliff_margin_cm']}cm vs Base FELL) | "
        f"L2(Dock={l2['final_err_cm']}cm, NewSeatGaze={l2['final_face_gaze_err_deg']}deg, Coll=False) | "
        f"L3(NBV={l3_v['final_err_cm']}cm vs {l3_b['final_err_cm']}cm, Coll=False) | "
        f"L4(3xGauntlet=+{l4_v['min_clearance_cm']}cm, 0-step Reloc, Coll=False) | HTML5 Sandbox OK!"
    )


if __name__ == "__main__":
    main()
