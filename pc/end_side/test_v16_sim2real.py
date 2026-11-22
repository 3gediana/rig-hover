"""test_v16_sim2real.py — Automated Regression Gate for v16 Sim2Real Recognition (v4) & Behavior Hardening (S1-S5)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
V4_REP = REPO / "runs_end_side" / "double_knock_3d_v4_sim2real" / "double_knock_3d_v4_sim2real_report.json"
V16_REP = REPO / "runs_end_side" / "sim2real_v16" / "sim2real_v16_report.json"
V16_SCRIPT = REPO / "pc" / "end_side" / "sim2real_torture_and_bridge_v16.py"


def main() -> None:
    print("=== v16 Sim2Real Regression Gate (Recognition v4 + Behavior S1-S5 + Live CLI) ===", flush=True)

    # 1. DoubleKnock3D-Sim2RealNet v4 Recognition Gate
    assert V4_REP.exists(), f"Missing v4 report: {V4_REP}"
    r4 = json.loads(V4_REP.read_text(encoding="utf-8"))
    c2 = r4["controlled_2d_v4"]
    c3 = r4["controlled_3d_v4"]
    ext = r4["external_30clip_sim2real"]

    assert c3["mean_win_auroc"] >= 0.925, f"v4 3D AUROC too low: {c3['mean_win_auroc']}"
    assert c3["mean_win_auroc"] - c2["mean_win_auroc"] >= 0.08, "v4 3D must beat 2D AUROC by >= +8%"
    # 6/6 clean AUROC sweep
    for f2, f3 in zip(c2["folds"], c3["folds"]):
        assert f3["win_auroc"] > f2["win_auroc"], f"AUROC sweep failed on {f3['val_take']}"
    assert c3["strict_1to1_stream_nobox"]["macro_strict_f1"] >= 0.78, "v4 NoBox Macro F1 regression"
    assert c3["strict_1to1_stream_nobox"]["total_fp"] <= 18, "v4 NoBox FP regression"
    assert ext["3d_robot9_night_fp"] <= 3, f"v4 Night Robot 9-clip FP must be <=3 (was 27 in v3!), got {ext['3d_robot9_night_fp']}"
    assert ext["3d_charades10_fp"] == 0, f"Charades FP must be 0, got {ext['3d_charades10_fp']}"
    assert ext["3d_ucf10_fp"] == 0, f"UCF-101 FP must be 0, got {ext['3d_ucf10_fp']}"
    print(
        f"  [PASS] Pillar 1 (DoubleKnock3D-Sim2RealNet v4): AUROC={c3['mean_win_auroc']:.4f} (6/6 sweep, +{(c3['mean_win_auroc']-c2['mean_win_auroc'])*100:.2f}%), "
        f"NoBox Macro F1={c3['strict_1to1_stream_nobox']['macro_strict_f1']:.4f} (FP={c3['strict_1to1_stream_nobox']['total_fp']}), "
        f"Night9 FP={ext['3d_robot9_night_fp']} (crushed from 27!), Charades=0, UCF=0",
        flush=True,
    )

    # 2. v16 Sim2Real Behavior Hardening Gate (S1-S5)
    assert V16_REP.exists(), f"Missing v16 report: {V16_REP}"
    r16 = json.loads(V16_REP.read_text(encoding="utf-8"))

    # S1: Black Mousepad Decoy vs Real Cliff
    s1n = r16["S1_black_mat_decoy_vs_real_cliff"]["naive_brightness_cliff"]
    s1v = r16["S1_black_mat_decoy_vs_real_cliff"]["v16_parallax_cliff"]
    assert s1n["cliff_false_alarm_on_mat"] is True and s1n["mat_safely_crossed"] is False, "S1 Naive should false-alarm on black mat"
    assert s1v["mat_safely_crossed"] is True and s1v["cliff_false_alarm_on_mat"] is False, "S1 v16 must cross black mat safely"
    assert s1v["cliff_detected"] is True and s1v["fell_off_cliff"] is False and s1v["min_cliff_margin_cm"] >= 8.0, "S1 v16 must stop safely before real cliff"
    assert s1v["mean_mat_parallax"] > s1v["mean_cliff_parallax"] * 1.8, "S1 parallax separation must be > 1.8x"

    # S2: 160ms WiFi Lag + Drop + Deadzone + Wheel Pull + JPEG Q=32 Blur + Night Glare
    s2n = r16["S2_wifi_lag_deadzone_blur"]["naive_controller"]
    s2v = r16["S2_wifi_lag_deadzone_blur"]["v16_sim2real_brain"]
    assert s2v["collision"] is False, "S2 v16 must have zero collision"
    assert s2v["mean_pose_err_cm"] < 5.0 and s2v["final_err_cm"] < 8.5 and s2v["lookback_err_deg"] < 5.0
    assert s2v["final_err_cm"] < s2n["final_err_cm"] * 0.5, "S2 v16 must dramatically beat Naive"

    # S3: Uncalibrated FOV (56 deg vs 65 deg nominal) + +2.4 deg Pitch Mount Bias + +/-10 deg Backlash
    s3n = r16["S3_uncalibrated_fov_pitch_servo"]["naive_controller"]
    s3v = r16["S3_uncalibrated_fov_pitch_servo"]["v16_sim2real_brain"]
    assert s3v["collision"] is False, "S3 v16 must have zero collision"
    assert abs(s3v["autocalib_fov_deg"] - 56.0) <= 1.0, f"S3 AutoCalib FOV inaccurate: {s3v['autocalib_fov_deg']}"
    assert abs(s3v["est_pitch_bias_deg"] - 2.4) <= 0.8, f"S3 AutoCalib pitch bias inaccurate: {s3v['est_pitch_bias_deg']}"
    assert s3v["mean_pose_err_cm"] < 6.0 and s3v["lookback_err_deg"] < 4.0 and s3v["final_err_cm"] < s3n["final_err_cm"] * 0.5

    # S4: 25-Episode Full-Reality Monte-Carlo
    s4n = r16["S4_25ep_full_reality_montecarlo"]["naive_summary"]
    s4v = r16["S4_25ep_full_reality_montecarlo"]["v16_summary"]
    assert s4v["n_episodes"] == 25
    assert s4v["median_err_cm"] < 6.0 and s4v["mean_err_cm"] < 8.0 and s4v["mean_pose_cm"] < 5.0
    assert s4v["mean_lookback_deg"] < 5.0 and s4v["collision_rate_pct"] <= 8.0
    assert s4v["mean_err_cm"] < s4n["mean_err_cm"] * 0.25, "S4 v16 must crush Naive on 25-ep Monte-Carlo"
    assert s4v["collisions"] < s4n["collisions"], "S4 v16 must reduce collisions vs Naive"

    # S5: ESP32-S3 HTTP HIL Protocol Verification
    s5 = r16["S5_esp32_http_hil"]
    assert s5["ok_closed_loop_cycles"] >= 15
    assert s5["mean_hil_pose_err_cm"] < 1.0
    assert s5["blackout_injected_and_handled"] is True
    assert s5["esp32_hw_watchdog_saved_on_blackout"] >= 1

    # Assets exist
    assert pathlib.Path(r16["poster_png"]).exists(), "Missing poster PNG"
    assert pathlib.Path(r16["demo_mp4"]).exists(), "Missing demo MP4"

    # CLI --help smoke test for --live bridge
    res = subprocess.run([sys.executable, str(V16_SCRIPT), "--help"], capture_output=True, text=True, check=True)
    assert "--live" in res.stdout and "--arm" in res.stdout, "Missing --live/--arm CLI options"

    print(
        f"  [PASS] Pillar 2 (v16 Sim2Real S1-S5): S1 Mat={s1v['mean_mat_parallax']}/Cliff={s1v['mean_cliff_parallax']} (+{s1v['min_cliff_margin_cm']}cm), "
        f"S2 Arr={s2v['final_err_cm']}cm/Pose={s2v['mean_pose_err_cm']}cm, "
        f"S3 FOV={s3v['autocalib_fov_deg']}deg/PB={s3v['est_pitch_bias_deg']}deg/Gaze={s3v['lookback_err_deg']}deg, "
        f"S4 25-Ep Med={s4v['median_err_cm']}cm (Naive {s4n['median_err_cm']}cm) Coll={s4v['collisions']}/25 (Naive {s4n['collisions']}/25), "
        f"S5 HIL={s5['median_e2e_loop_ms']}ms/Pose={s5['mean_hil_pose_err_cm']}cm",
        flush=True,
    )
    print("ALL v16 SIM2REAL REGRESSION GATES PASSED!", flush=True)


if __name__ == "__main__":
    main()
