#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_v12_room360_self_correct.py —— 突破 #4（360° 全房间建模 + 闭环自纠错）自动化回归门禁"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT_DIR = REPO / "runs_end_side" / "room360_self_correct_v12"
REP_JSON = OUT_DIR / "room360_self_correct_v12_report.json"


def main() -> int:
    assert REP_JSON.exists(), f"Missing v12 report: {REP_JSON}"
    rep = json.loads(REP_JSON.read_text(encoding="utf-8"))
    assert rep["models"]["runtime_moge2"] is False, "Runtime MoGe-2 must be False!"
    assert rep["models"]["geo_360_mb"] < 32.0, "HoverRoomGeoNet-360 too large!"
    assert rep["models"]["self_corrector_mb"] < 10.0, "RoomSelfCorrectorNet too large!"

    # 1) 验证 3 个严格未知 360° 房间的自然零样本 3-Pass 闭环提升
    nat = rep["benchmark_4a_natural_zeroshot"]
    assert len(nat) == 3, f"Expected 3 held-out rooms, got {len(nat)}"
    for r in nat:
        p1, p2, p3 = r["pass1"], r["pass2"], r["pass3"]
        assert p3["psnr_db"] >= p2["psnr_db"] > p1["psnr_db"], f"[{r['room']}] Natural PSNR not monotonically improving!"
        assert p3["seam_err_cm"] < p1["seam_err_cm"], f"[{r['room']}] Natural Pass 3 seam did not shrink!"
        assert p3["coverage_pct"] >= 97.0, f"[{r['room']}] Low 360 coverage: {p3['coverage_pct']}%"
        print(
            f"[PASS] 4A Natural Unseen '{r['room']}': PSNR {p1['psnr_db']} -> {p2['psnr_db']} -> {p3['psnr_db']} dB | "
            f"Seam {p1['seam_err_cm']} -> {p3['seam_err_cm']} cm | Cov {p3['coverage_pct']}% ({p3['n_pts']:,} pts)"
        )

    # 2) 验证 4 重极端破坏 (零偷看鼓包 + 主动补洞 45° 空洞 + 6000 漂浮幽灵 + 10° 扭曲位姿) 自愈恢复能力
    sab = rep["benchmark_4b_sabotage_recovery"]
    for rn, r in sab.items():
        p1, p3, rec = r["pass1"], r["pass3"], r["self_correction_recovery"]
        assert p3["psnr_db"] > p1["psnr_db"] + 3.0, f"[{rn}] Insufficient sabotage PSNR recovery!"
        assert rec["ghost_eviction_pct"] >= 85.0, f"[{rn}] Low ghost eviction: {rec['ghost_eviction_pct']}%"
        assert rec["hole_45deg_after_pct"] >= 96.0, f"[{rn}] 45° hole not patched: {rec['hole_45deg_after_pct']}%"
        assert rec["bulge_error_reduction_pct"] >= 50.0, f"[{rn}] Zero-peek wall bulge not corrected: {rec['bulge_error_reduction_pct']}%"
        assert rec["pose_twist_residual_deg"] < 2.0, f"[{rn}] Twisted pose not untwisted: {rec['pose_twist_residual_deg']} deg"
        print(
            f"[PASS] 4B Sabotage Self-Healing '{rn}': PSNR {p1['psnr_db']} -> {p3['psnr_db']} dB (+{rec['psnr_gain_db']} dB) | "
            f"Ghosts Evicted={rec['ghost_eviction_pct']}% | 45° Hole {rec['hole_45deg_before_pct']}% -> {rec['hole_45deg_after_pct']}% | "
            f"Bulge Cut={rec['bulge_error_reduction_pct']}% | Twist {rec['pose_twist_injected_deg']}° -> {rec['pose_twist_residual_deg']}°!"
        )

    # 3) 验证交互式 WebGL 与海报文件均存在
    for k, p in rep["artifacts"].items():
        fp = pathlib.Path(p)
        assert fp.exists() and fp.stat().st_size > 10000, f"Missing artifact {k}: {fp}"
    print("[PASS] All v12 360° Whole-Room Self-Correction Gates & WebGL/Poster Artifacts PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
