"""Scenario capability matrix for RIG-Hover (Wenwen).

Scans every runs_end_side/*/ report and emits a single capability matrix:
which scenarios are demonstrated, at which validation tier, with what
headline metric, and which artifacts back the claim.

Validation tiers (honest ordering, weakest -> strongest):
    T0 synthetic-sim      zero-oracle DesktopArena3D / synthetic GT
    T1 real-video-replay  real take frames driving perception/decision
    T2 external-stress    clips outside the training distribution
    T3 HIL                local ESP32-compatible HTTP loop (no robot)
    T4 physical           on the real robot (nothing here yet)

Outputs:
    runs_end_side/scenario_matrix/scenario_matrix.json
    runs_end_side/scenario_matrix/scenario_matrix.md
    runs_end_side/scenario_matrix/scenario_matrix.html

Usage:
    python scenario_matrix.py
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "runs_end_side"
OUT = RUNS / "scenario_matrix"

TIERS = {
    "T0": "synthetic-sim (zero-oracle)",
    "T1": "real-video replay",
    "T2": "external-distribution stress",
    "T3": "HIL (ESP32-compatible local loop)",
    "T4": "physical robot",
}


def _load(rel: str) -> dict:
    return json.loads((RUNS / rel).read_text(encoding="utf-8"))


def _art(*rels: str) -> dict:
    """Existing artifact files, repo-relative."""
    out = {}
    for r in rels:
        p = RUNS / r
        if p.exists():
            out[p.suffix.lstrip(".")] = str(p.relative_to(REPO)).replace("\\", "/")
    return out


def entries() -> list[dict]:
    e = []

    # ------------------------------------------------------------------
    # Perception: double-knock recognition (the gate for every scenario)
    # ------------------------------------------------------------------
    d = _load("double_knock_3d_v2_extreme/double_knock_3d_v2_extreme_report.json")
    s3 = d["controlled_3d_v2"]["strict_1to1_stream"]
    e.append({
        "id": "knock-3d-v2", "name": "DoubleKnock3D-PhysNet v2 (strict stream)",
        "category": "perception", "tier": "T1",
        "source": "runs_end_side/double_knock_3d_v2_extreme/double_knock_3d_v2_extreme_report.json",
        "metrics": {
            "gt_events": s3.get("total_gt", 89), "hits": s3.get("total_hit"),
            "fp": s3.get("total_fp"),
            "f1": s3.get("micro_strict_f1"), "fa_per_min": s3.get("mean_fa_per_min"),
        },
        "artifacts": {},
        "notes": "6-fold LOO, nested Inner-OOF calibration, strict 1-to-1 matching.",
    })
    e.append({
        "id": "knock-3d-v2-ext", "name": "DoubleKnock3D-PhysNet v2 on 30 external clips",
        "category": "perception", "tier": "T2",
        "source": "runs_end_side/double_knock_3d_v2_extreme/double_knock_3d_v2_extreme_report.json",
        "metrics": {
            "neg_minutes": d["external_30clip_head_to_head"]["neg_minutes"],
            "3d_str_fp": d["external_30clip_head_to_head"]["3d_str_fp"],
            "knock_d05_detected": d["external_30clip_head_to_head"]["knock_d05_detected"],
        },
        "artifacts": {},
        "notes": "0 streaming false alarms on 30 out-of-distribution clips; missed the d0.5 boundary positive.",
    })

    d = _load("double_knock_3d_v4_sim2real/double_knock_3d_v4_sim2real_report.json")
    s4 = d["controlled_3d_v4"]["strict_1to1_stream"]
    e.append({
        "id": "knock-3d-v4", "name": "DoubleKnock3D v4 Sim2Real (GeM + rocking/ISO/JPEG aug)",
        "category": "perception", "tier": "T1",
        "source": "runs_end_side/double_knock_3d_v4_sim2real/double_knock_3d_v4_sim2real_report.json",
        "metrics": {
            "hits": s4.get("total_hit"), "fp": s4.get("total_fp"),
            "f1": s4.get("micro_strict_f1"), "fa_per_min": s4.get("mean_fa_per_min"),
        },
        "artifacts": {},
        "notes": "Robustness-augmented variant; recall-lower but cleanest FA/min.",
    })

    d = _load("rigorous_audit_v13/rigorous_knock_2d_vs_3d_ablation.json")
    e.append({
        "id": "knock-ablation", "name": "2D-vs-3D same-architecture ablation (v13 audit)",
        "category": "perception", "tier": "T1",
        "source": "runs_end_side/rigorous_audit_v13/rigorous_knock_2d_vs_3d_ablation.json",
        "metrics": {
            "samearch_2d_f1": d["controlled_2d_only"]["strict_1to1_stream"]["micro_strict_f1"],
            "samearch_3d_f1": d["controlled_3d_physnet"]["strict_1to1_stream"]["micro_strict_f1"],
            "pure_3d_delta": d["pure_3d_causal_delta"].get("strict_stream_micro_f1_delta"),
        },
        "artifacts": {},
        "notes": "3D adds only +0.0217 F1 over identical 2D arch; fold-level calibration collapse documented.",
    })

    # ------------------------------------------------------------------
    # Geometry / mapping
    # ------------------------------------------------------------------
    d = _load("rigorous_audit_v13/rigorous_geo_exact_gt_benchmark.json")
    v3 = d["HoverGeoNet-v3 (27MB Distilled)"]
    moge = d["MoGe-2-ViT-B (419MB Teacher)"]
    e.append({
        "id": "geo-exact-gt", "name": "HoverGeoNet vs MoGe-2 on exact ray-traced GT",
        "category": "geometry", "tier": "T0",
        "source": "runs_end_side/rigorous_audit_v13/rigorous_geo_exact_gt_benchmark.json",
        "metrics": {
            "geonet_v3_near_absrel": v3.get("near_tabletop_abs_rel"),
            "geonet_v3_near_rmse_mm": v3.get("near_tabletop_rmse_mm"),
            "cup_125mm_err_mm": v3.get("cup_125mm_top_err_mm"),
            "moge2_near_rmse_mm": moge.get("near_tabletop_rmse_mm"),
        },
        "artifacts": _art("geonet240_vs_moge2_v7/showcase_geonet240_vs_moge2_v7.png"),
        "notes": "27MB distilled student beats 419MB teacher near tabletop after calibration.",
    })

    d = _load("planar_se2_subcm_v11/planar_se2_subcm_v11_report.json")
    c = d["exp_c_20ep_randomized_stress"]["summary"]
    e.append({
        "id": "planar-se2-v11", "name": "Odometry-free SE(2) planar visual relocalization + Neural-SDF CBF",
        "category": "geometry", "tier": "T0",
        "source": "runs_end_side/planar_se2_subcm_v11/planar_se2_subcm_v11_report.json",
        "metrics": {
            "episodes": c.get("n_episodes", 20),
            "mean_arrival_err_cm": c.get("mean_arrival_err_cm"),
            "median_arrival_err_cm": c.get("median_arrival_err_cm"),
            "collisions": c.get("collision_count", 0),
        },
        "artifacts": _art("planar_se2_subcm_v11/showcase_v11_three_breakthroughs.png",
                          "showcase_v11_three_breakthroughs.png"),
        "notes": "Zero wheel odometry, random wheel slip 0.69-0.90, gyro drift; sub-cm-scale arrival.",
    })

    d = _load("room360_self_correct_v12/room360_self_correct_v12_report.json")
    e.append({
        "id": "room360-v12", "name": "Room-360 self-correcting relocalization (real panoramas)",
        "category": "geometry", "tier": "T1",
        "source": "runs_end_side/room360_self_correct_v12/room360_self_correct_v12_report.json",
        "metrics": {
            "heldout_rooms": d.get("unseen_heldout_rooms"),
            "zeroshot_pass2_psnr_db": (d["benchmark_4a_natural_zeroshot"][0]["pass2"].get("psnr_db")
                                      if isinstance(d.get("benchmark_4a_natural_zeroshot"), list)
                                      and d["benchmark_4a_natural_zeroshot"] else None),
            "sabotage_rooms": list(d.get("benchmark_4b_sabotage_recovery", {}).keys()),
        },
        "artifacts": _art("room360_self_correct_v12/showcase_v12_room360_self_correction.png"),
        "notes": "Zero-shot self-corrector on unseen held-out rooms; sabotage-recovery benchmark.",
    })

    # ------------------------------------------------------------------
    # Behavior / missions
    # ------------------------------------------------------------------
    d = _load("embodied_brain_v14/embodied_brain_v14_report.json")
    e.append({
        "id": "v14-realvideo-e2e", "name": "v14 real-video knock -> 3D lock -> visual dock",
        "category": "behavior", "tier": "T1",
        "source": "runs_end_side/embodied_brain_v14/embodied_brain_v14_report.json",
        "metrics": {
            "distractor_fp_40s": d["extreme1_real_video_to_behavior"]["fp_during_40s_real_distractors"],
            "final_err_cm": d["extreme1_real_video_to_behavior"]["final_err_cm"],
            "lookback_face_err_deg": d["extreme1_real_video_to_behavior"]["lookback_face_err_deg"],
        },
        "artifacts": _art("embodied_brain_v14/demo_embodied_brain_v14.mp4",
                          "embodied_brain_v14/showcase_v14_extreme_robot_brain.png"),
        "notes": "40s real distractor segment with zero false launches, then real knock triggers docking.",
    })
    e.append({
        "id": "v14-kidnap", "name": "v14 kidnap self-recovery (+95deg blind rotate + 12cm shove)",
        "category": "behavior", "tier": "T0",
        "source": "runs_end_side/embodied_brain_v14/embodied_brain_v14_report.json",
        "metrics": {
            "final_err_cm": d["extreme2_kidnapping_recovery"]["v14_global_recovery"]["final_err_cm"],
            "baseline_final_err_cm": d["extreme2_kidnapping_recovery"]["baseline_no_recovery"]["final_err_cm"],
        },
        "artifacts": {},
        "notes": "Global SIFT+PnP relocalization within 2 steps of the shove.",
    })
    e.append({
        "id": "v14-stress30", "name": "v14 30-episode brutal stress (slip + kidnap + obstacles)",
        "category": "behavior", "tier": "T0",
        "source": "runs_end_side/embodied_brain_v14/embodied_brain_v14_report.json",
        "metrics": {
            "episodes": 30,
            "mean_err_cm": d["extreme5_30ep_brutal_stress"]["summary"]["mean_arrival_err_cm"],
            "median_err_cm": d["extreme5_30ep_brutal_stress"]["summary"]["median_arrival_err_cm"],
            "collision_rate_pct": d["extreme5_30ep_brutal_stress"]["summary"]["collision_rate_pct"],
        },
        "artifacts": {},
        "notes": "1/30 collision under worst-case disturbances.",
    })

    d = _load("living_brain_v15/living_brain_v15_report.json")
    e.append({
        "id": "v15-cliff", "name": "v15 visual-cliff refusal (pure vision, no prior map)",
        "category": "behavior", "tier": "T0",
        "source": "runs_end_side/living_brain_v15/living_brain_v15_report.json",
        "metrics": {
            "cliff_y_true_m": d["L1_visual_cliff_survival"]["cliff_y_true_m"],
            "v15_min_margin_cm": d["L1_visual_cliff_survival"]["v15_living_mind"].get("min_cliff_margin_cm"),
            "baseline_fell": d["L1_visual_cliff_survival"]["baseline_no_cliff"].get("fell_off_cliff"),
        },
        "artifacts": _art("living_brain_v15/demo_living_brain_v15.mp4",
                          "living_brain_v15/wenwen_living_brain_v15.html"),
        "notes": "Detected cliff at step 17, est -0.394m drop, refused to cross.",
    })
    e.append({
        "id": "v15-chaos", "name": "v15 triple-chaos gauntlet (moving + sudden obstacle + kidnap, one mission)",
        "category": "behavior", "tier": "T0",
        "source": "runs_end_side/living_brain_v15/living_brain_v15_report.json",
        "metrics": {
            "final_err_cm": d["L4_triple_chaos_gauntlet"]["v15_living_mind"].get("final_err_cm"),
            "min_clearance_cm": d["L4_triple_chaos_gauntlet"]["v15_living_mind"].get("min_clearance_cm"),
        },
        "artifacts": {},
        "notes": "All three attacks detected at first visual contact; mission still completed.",
    })

    # ------------------------------------------------------------------
    # Sim2Real / deploy
    # ------------------------------------------------------------------
    d = _load("sim2real_v16/sim2real_v16_report.json")
    e.append({
        "id": "v16-montecarlo", "name": "v16 full-reality-gap Monte Carlo (25 eps)",
        "category": "deploy", "tier": "T0",
        "source": "runs_end_side/sim2real_v16/sim2real_v16_report.json",
        "metrics": {
            "median_err_cm": d["S4_25ep_full_reality_montecarlo"]["v16_summary"].get("median_err_cm"),
            "p90_err_cm": d["S4_25ep_full_reality_montecarlo"]["v16_summary"].get("p90_err_cm"),
            "collision_rate_pct": d["S4_25ep_full_reality_montecarlo"]["v16_summary"].get("collision_rate_pct"),
            "naive_collision_rate_pct": d["S4_25ep_full_reality_montecarlo"]["naive_summary"].get("collision_rate_pct"),
        },
        "artifacts": _art("sim2real_v16/demo_sim2real_reality_v16.mp4",
                          "sim2real_v16/showcase_v16_sim2real_gap_crushed.png"),
        "notes": "Latency/dropout/dead-zone/yaw-pull all injected; naive baseline collides 28%.",
    })
    e.append({
        "id": "v16-hil", "name": "v16 ESP32 HTTP hardware-in-the-loop (/status+/capture+/cmd+watchdog)",
        "category": "deploy", "tier": "T3",
        "source": "runs_end_side/sim2real_v16/sim2real_v16_report.json",
        "metrics": {
            "closed_loop_cycles": d["S5_esp32_http_hil"]["ok_closed_loop_cycles"],
            "median_e2e_ms": d["S5_esp32_http_hil"]["median_e2e_loop_ms"],
            "blackout_handled": d["S5_esp32_http_hil"]["blackout_injected_and_handled"],
            "watchdog_saved": d["S5_esp32_http_hil"]["esp32_hw_watchdog_saved_on_blackout"],
        },
        "artifacts": {},
        "notes": "Same endpoints the real firmware exposes; --live --host flag ready.",
    })
    e.append({
        "id": "v16-blackmat", "name": "v16 black-mat vs real-cliff parallax discrimination",
        "category": "deploy", "tier": "T0",
        "source": "runs_end_side/sim2real_v16/sim2real_v16_report.json",
        "metrics": {
            "mat_parallax": d["S1_black_mat_decoy_vs_real_cliff"]["v16_parallax_cliff"].get("mean_mat_parallax"),
            "cliff_parallax": d["S1_black_mat_decoy_vs_real_cliff"]["v16_parallax_cliff"].get("mean_cliff_parallax"),
            "mat_safely_crossed": d["S1_black_mat_decoy_vs_real_cliff"]["v16_parallax_cliff"].get("mat_safely_crossed"),
            "naive_false_alarm": d["S1_black_mat_decoy_vs_real_cliff"]["naive_brightness_cliff"].get("cliff_false_alarm_on_mat"),
        },
        "artifacts": {},
        "notes": "Naive brightness cliff detector false-alarms on the mat; flow-parallax does not.",
    })

    # ------------------------------------------------------------------
    # Skill zoo / policies
    # ------------------------------------------------------------------
    d = _load("skill_zoo/skill_zoo_metrics.json")
    e.append({
        "id": "skill-zoo", "name": "RL skill zoo (orbit / social-proxemics / figure8 / ping-point-nav)",
        "category": "policy", "tier": "T0",
        "source": "runs_end_side/skill_zoo/skill_zoo_metrics.json",
        "metrics": {"skills": [s.get("name") if isinstance(s, dict) else s
                              for s in d.get("skills", [])]},
        "artifacts": {},
        "notes": "ping_point_nav_v1 71/72 spec-compliant in 8-scenario sandbox.",
    })

    # ------------------------------------------------------------------
    # Gaps / not-yet-run candidates
    # ------------------------------------------------------------------
    gaps = [
        {"gap": "physical-robot acceptance of ANY v14-v16 scenario", "blocking": "robot offline; needs --live run with watchdog supervision"},
        {"gap": "camera intrinsics calibration (FOV_H=60deg vs dist_k=0.84 contradiction)", "blocking": "needs printed checkerboard + 5min capture on the real robot"},
        {"gap": "double-knock recall still ~65-77% under strict streaming", "blocking": "ensemble/rhythm-filter experiments below; more low-fps takes"},
        {"gap": "full E2E real-video scenario (face->fist->knock->dock->lookback) on recorded take", "blocking": "needs pc_unified_brain_app run on a full take, not just segments"},
        {"gap": "pursuit/follow-a-cup behavior", "blocking": "none - implementable in DesktopArena3D tonight"},
        {"gap": "edge-survey cliff-map then point-to-point anywhere", "blocking": "none - implementable in DesktopArena3D tonight"},
        {"gap": "Android TFLite on-device run of the GeoNet+knock bundle", "blocking": "needs G:\\robot_app build + phone"},
    ]

    return {"tiers": TIERS, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "entries": e, "gaps": gaps}


def write_md(m: dict) -> str:
    lines = [
        "# RIG-Hover Scenario Capability Matrix",
        "",
        f"Generated: {m['generated_at']} (auto-generated by `pc/end_side/scenario_matrix.py`)",
        "",
        "Validation tiers: " + " · ".join(f"**{k}** {v}" for k, v in m["tiers"].items()),
        "",
        "| Scenario | Cat | Tier | Headline metrics | Evidence |",
        "|---|---|---|---|---|",
    ]
    for en in m["entries"]:
        mets = "; ".join(f"{k}={v}" for k, v in en["metrics"].items() if v is not None)[:150]
        arts = " ".join(f"[{k}]({v})" for k, v in en["artifacts"].items()) or "report"
        lines.append(f"| {en['name']} | {en['category']} | {en['tier']} | {mets} | {arts} |")
    lines += ["", "## Gaps / next candidates", ""]
    for g in m["gaps"]:
        lines.append(f"- **{g['gap']}** — {g['blocking']}")
    lines += [
        "",
        "> Honest boundary: T0/T1/T2 are simulation or replay; T3 is a local ESP32-compatible",
        "> loop. Nothing in this matrix is a physical-robot acceptance result yet.",
    ]
    return "\n".join(lines) + "\n"


def write_html(m: dict) -> str:
    rows = []
    tcolor = {"T0": "#8b5cf6", "T1": "#0ea5e9", "T2": "#f59e0b", "T3": "#10b981", "T4": "#ef4444"}
    for en in m["entries"]:
        mets = "".join(f"<span class=metric><b>{k}</b>={v}</span>"
                      for k, v in en["metrics"].items() if v is not None)
        arts = " ".join(f"<a href='../{v}' target=_blank>{k}</a>" for k, v in en["artifacts"].items())
        rows.append(
            f"<div class=card><div class=head><span class=tier style='background:{tcolor[en['tier']]}'>"
            f"{en['tier']}</span><h3>{en['name']}</h3></div>"
            f"<div class=cat>{en['category']} · {TIERS[en['tier']]}</div>"
            f"<div class=metrics>{mets}</div>"
            f"<div class=notes>{en['notes']}</div>"
            f"<div class=arts>{arts}<span class=src>{en['source']}</span></div></div>")
    gapli = "".join(f"<li><b>{g['gap']}</b> — {g['blocking']}</li>" for g in m["gaps"])
    return f"""<!doctype html><meta charset=utf-8><title>RIG-Hover Scenario Matrix</title>
<style>
body{{font:14px/1.5 system-ui;background:#0b1020;color:#e5e7eb;margin:0;padding:32px}}
h1{{font-size:22px}} h3{{margin:0;display:inline}}
.legend span{{padding:2px 8px;border-radius:4px;color:#fff;margin-right:8px;font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:18px}}
.card{{background:#131a2e;border:1px solid #243052;border-radius:10px;padding:14px}}
.tier{{color:#fff;padding:2px 8px;border-radius:5px;font-size:11px;font-weight:700;margin-right:8px}}
.cat{{color:#94a3b8;font-size:12px;margin:6px 0}}
.metric{{background:#0f172a;border:1px solid #1e293b;border-radius:4px;padding:1px 6px;margin:2px 4px 2px 0;display:inline-block;font-size:12px}}
.notes{{color:#a5b4cf;font-size:12px;margin-top:8px}}
.arts{{margin-top:8px;font-size:12px}} .arts a{{color:#38bdf8;margin-right:10px}}
.src{{color:#475569;font-size:10px;display:block;margin-top:4px}}
h2{{margin-top:28px}} li{{margin:6px 0;color:#cbd5e1}}
.boundary{{background:#1a1030;border:1px solid #7c3aed;padding:10px 14px;border-radius:8px;margin-top:20px;color:#d8b4fe}}
</style>
<h1>RIG-Hover （文文） Scenario Capability Matrix</h1>
<div class=legend>{''.join(f"<span style='background:{tcolor[k]}'>{k} {v}</span>" for k, v in m['tiers'].items())}</div>
<div class=grid>{''.join(rows)}</div>
<h2>Gaps / next candidates</h2><ul>{gapli}</ul>
<div class=boundary><b>Honest boundary:</b> T0-T2 results are simulation or replay; T3 is a local ESP32-compatible loop.
No entry here is a physical-robot acceptance run yet. Generated {m['generated_at']}.</div>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = entries()
    (OUT / "scenario_matrix.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "scenario_matrix.md").write_text(write_md(m), encoding="utf-8")
    (OUT / "scenario_matrix.html").write_text(write_html(m), encoding="utf-8")
    print(f"[scenario_matrix] {len(m['entries'])} entries, {len(m['gaps'])} gaps -> {OUT}")


if __name__ == "__main__":
    main()
