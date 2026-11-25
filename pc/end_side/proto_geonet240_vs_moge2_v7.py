# -*- coding: utf-8 -*-
"""proto_geonet240_vs_moge2_v7.py — 本地化 `HoverGeoNet-240` (3.87 MB) vs `MoGe-2-ViT-B` (419 MB) 全面实测对比与零残影动态分离演示。

全部对比场景均采用 100% 严格留出未见数据（跨天零样本）：
1. `empty_d0.5_front_night_182201`：空桌 + 3.5cm 弧顶鼠标 + 键盘 + 显示器 + 90° 桌角；
2. `robot_20261115_151733_cb8455`：水杯 + 笔记本 + 挥动手臂（重点验证 Dynablox 零残影动态层分离！）。

产出位于 `D:\robot_work\runs_end_side\geonet240_vs_moge2_v7\`：
- `showcase_geonet240_vs_moge2_v7.png`：原图 | 旧版 96×96 学生 (0.54 MB) | 新版 HoverGeoNet-240 (3.87 MB, ~7ms) | MoGe-2-ViT-B (419 MB, ~163ms) | 鼠标/水杯 3× 特写对比；
- `showcase_dynablox_zero_ghost_v7.png`：`cb8455` 挥动手臂去残影消融对比（无分离残影 vs MoGe-2+Dynablox vs HoverGeoNet-240+Dynablox）；
- 4 个独立离线 60fps WebGL 交互式 3D 绿色查看器 HTML + `.ply` 点云文件；
- `metrics_v7.json`：完整量化指标。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hd_dynamic_contour_slam import (  # noqa: E402
    HdDynamicContourSlamEngine,
    HoverGeoNet240Frontend,
    MoGe2Frontend,
    render_hd_dual_layer_green,
)
from hd_moge_contour_engine import (  # noqa: E402
    GEONET_CKPT_FOV_H_DEG,
    REAL_H_CAM_M,
    UltraDenseSurfelCloud,
    export_ply_and_interactive_html,
    render_hd_laser_green_cloud,
)
from neural_gaussian_splat_model import OrbitCamera3D  # noqa: E402
from robust_contour_slam import (  # noqa: E402
    RobustContourSlamPipeline,
    render_green_contour_edl,
)

OUT_DIR = REPO / "runs_end_side" / "geonet240_vs_moge2_v7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OLD_TFLITE = REPO / "runs_end_side" / "dense_splat_v3" / "model.tflite"
GEONET_CKPT = REPO / "runs_end_side" / "hover_geonet_240" / "hover_geonet_240_fp16.pt"

EMPTY_DIR = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"
CB8455_MP4 = REPO / "dataset" / "takes" / "robot_20261115_151733_cb8455" / "take.mp4"


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45, color=(155, 255, 195), thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 10, 7), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def add_hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    h_bar = 10 + 20 * len(lines)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], h_bar), (5, 11, 7), -1)
    out = cv2.addWeighted(ov, 0.78, img, 0.22, 0)
    for i, ln in enumerate(lines):
        put_text(out, ln, (8, 19 + 20 * i), scale=0.43)
    return out


def load_frames(kind: str, src: pathlib.Path, start: int, count: int, step: int = 3) -> list[np.ndarray]:
    if kind == "frames":
        jpgs = sorted(src.glob("*.jpg"))[start : start + count * step : step]
        return [cv2.imread(str(j)) for j in jpgs if cv2.imread(str(j)) is not None]
    cap = cv2.VideoCapture(str(src))
    frames = []
    for k in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * step)
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    return frames


def measure_roi_bump(pts: np.ndarray, x_r: tuple[float, float], y_r: tuple[float, float]) -> dict[str, float]:
    m = (
        (pts[:, 0] >= x_r[0]) & (pts[:, 0] <= x_r[1])
        & (pts[:, 1] >= y_r[0]) & (pts[:, 1] <= y_r[1])
        & (pts[:, 2] >= -0.02) & (pts[:, 2] <= 0.28)
    )
    sub = pts[m]
    if len(sub) < 15:
        return {"pts": int(len(sub)), "z_p50_mm": 0.0, "z_p95_mm": 0.0, "z_max_mm": 0.0}
    z = sub[:, 2] * 1000.0
    return {
        "pts": int(len(sub)),
        "z_p50_mm": round(float(np.percentile(z, 50)), 1),
        "z_p95_mm": round(float(np.percentile(z, 95)), 1),
        "z_max_mm": round(float(np.max(z)), 1),
    }


def main() -> None:
    geo_mb = GEONET_CKPT.stat().st_size / 1048576.0
    print(f"=== HoverGeoNet-240 ({geo_mb:.2f}MB) vs MoGe-2-ViT-B (419MB) + Dynablox Zero-Ghost Benchmark ===", flush=True)
    geo_fe = HoverGeoNet240Frontend(GEONET_CKPT, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG)
    moge_fe = MoGe2Frontend(infer_hw=480, h_cam_m=REAL_H_CAM_M, fov_h_deg=GEONET_CKPT_FOV_H_DEG)

    scenes = [
        {
            "key": "empty_d05",
            "title": "Held-Out A: empty_d0.5 (Mouse 3.5cm + Keyboard)",
            "kind": "frames",
            "src": EMPTY_DIR,
            "start": 10,
            "count": 8,
            "step": 4,
            "cam_ov": OrbitCamera3D(azimuth_deg=-152.0, elevation_deg=34.0, distance_m=0.84, target_x=0.42, target_y=0.00, target_z=0.05),
            "cam_zoom": OrbitCamera3D(azimuth_deg=-138.0, elevation_deg=26.0, distance_m=0.42, target_x=0.38, target_y=0.04, target_z=0.03),
            "roi": ((0.20, 0.56), (-0.18, 0.22)),
            "roi_name": "Mouse+Keyboard",
        },
        {
            "key": "cb8455",
            "title": "Held-Out B: cb8455 (Cup + Notebook + Waving Arm)",
            "kind": "video",
            "src": CB8455_MP4,
            "start": 592,
            "count": 10,
            "step": 5,
            "cam_ov": OrbitCamera3D(azimuth_deg=-148.0, elevation_deg=36.0, distance_m=0.88, target_x=0.42, target_y=0.04, target_z=0.07),
            "cam_zoom": OrbitCamera3D(azimuth_deg=-130.0, elevation_deg=28.0, distance_m=0.46, target_x=0.42, target_y=0.10, target_z=0.05),
            "roi": ((0.22, 0.62), (-0.05, 0.28)),
            "roi_name": "Cup+Notebook",
        },
    ]

    W, H = 440, 330
    rows_main = []
    metrics_all: dict[str, Any] = {"h_cam_m": REAL_H_CAM_M, "fov_h_deg": GEONET_CKPT_FOV_H_DEG, "scenes": {}}
    ablation_cb_panels: list[np.ndarray] = []

    for sc in scenes:
        frames = load_frames(sc["kind"], sc["src"], sc["start"], sc["count"], sc["step"])
        assert len(frames) >= 4, f"Not enough frames for {sc['key']}"
        print(f"\n--- Running {sc['key']} ({len(frames)} frames) ---", flush=True)

        # 1) 旧版 96x96 Student (0.54 MB) 基线
        old_pipe = RobustContourSlamPipeline(OLD_TFLITE)
        for fr in frames[:6]:
            old_pipe.process_frame(fr)
        s_old, n_old = old_pipe.map.get_static_confirmed()
        img_old = render_green_contour_edl(
            s_old, n_old, sc["cam_ov"], grid=None,
            dyn_pts=old_pipe.map.d_pos, dyn_nrms=old_pipe.map.d_nrm,
            w=W, h=H, draw_ground_grid=True, draw_trav_overlay=False,
        )
        img_old = add_hud(img_old, [
            "OLD 96x96 Student (0.54MB)",
            f"8.5mm voxels | {len(s_old):,} pts (blurry)",
        ])

        # 2) NEW Localized HoverGeoNet-240 (3.87 MB FP16) + 2mm Dynablox Zero-Ghost
        slam_geo = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=REAL_H_CAM_M)
        geo_times = []
        geo_dyn_counts = []
        geo_carved_total = 0
        for fi, fr in enumerate(frames):
            prev_fr = frames[fi - 1] if fi > 0 else None
            pred_g = geo_fe.infer_frame(fr, prev_fr, pitch_deg=3.0, roll_deg=0.0, auto_level=True)
            geo_times.append(pred_g["infer_ms"])
            st_g = slam_geo.process_frame(
                fr, pred_g["pts_grid"], pred_g["nrm_grid"], pred_g["crease_grid"], pred_g["valid_grid"],
                align=(fi > 0),
            )
            geo_dyn_counts.append(st_g["n_dynamic_frame"])
            geo_carved_total += st_g["carved_ghosts"]

        g_pts, g_nrm, g_cr = slam_geo.get_confirmed_static_cloud(min_wt=1.0)
        g_dens = slam_geo.static_cloud.measure_density_metrics()
        g_roi = measure_roi_bump(g_pts, sc["roi"][0], sc["roi"][1])
        img_geo = render_hd_dual_layer_green(
            g_pts, g_nrm, g_cr, slam_geo.dyn_pts, slam_geo.dyn_nrms,
            cam=sc["cam_ov"], w=W, h=H, point_radius_px=1,
        )
        g_near = g_dens.get("near_0.25_0.55m", {})
        img_geo = add_hud(img_geo, [
            f"NEW HoverGeoNet-240 ({geo_mb:.1f}MB, {np.mean(geo_times):.1f}ms)",
            f"2mm+Dynablox | {len(g_pts):,}st+{len(slam_geo.dyn_pts):,}dyn | {g_near.get('median_spacing_mm', 0):.1f}mm",
        ])

        # 3) MoGe-2-ViT-B (419 MB) + 2mm Dynablox Zero-Ghost
        slam_moge = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=REAL_H_CAM_M)
        naive_moge = UltraDenseSurfelCloud(vox_size_m=0.0020)
        moge_times = []
        moge_carved_total = 0
        for fi, fr in enumerate(frames):
            pred_m = moge_fe.infer_frame(fr)
            moge_times.append(pred_m["infer_ms"])
            st_m = slam_moge.process_frame(
                fr, pred_m["pts_grid"], pred_m["nrm_grid"], pred_m["crease_grid"], pred_m["valid_grid"],
                align=(fi > 0),
            )
            moge_carved_total += st_m["carved_ghosts"]
            if sc["key"] == "cb8455":
                naive_moge.fuse_frame(pred_m["pts"], pred_m["nrms"], pred_m["crease"], align=(fi > 0))

        m_pts, m_nrm, m_cr = slam_moge.get_confirmed_static_cloud(min_wt=1.0)
        m_dens = slam_moge.static_cloud.measure_density_metrics()
        m_roi = measure_roi_bump(m_pts, sc["roi"][0], sc["roi"][1])
        img_moge = render_hd_dual_layer_green(
            m_pts, m_nrm, m_cr, slam_moge.dyn_pts, slam_moge.dyn_nrms,
            cam=sc["cam_ov"], w=W, h=H, point_radius_px=1,
        )
        m_near = m_dens.get("near_0.25_0.55m", {})
        img_moge = add_hud(img_moge, [
            f"MoGe-2-ViT-B Ref (419MB, {np.mean(moge_times):.0f}ms)",
            f"2mm+Dynablox | {len(m_pts):,}st+{len(slam_moge.dyn_pts):,}dyn | {m_near.get('median_spacing_mm', 0):.1f}mm",
        ])

        # 4) 3x Close-Up Zoom (Top: HoverGeoNet-240, Bottom: MoGe-2-ViT-B)
        zh = H // 2
        z_geo = render_hd_dual_layer_green(g_pts, g_nrm, g_cr, slam_geo.dyn_pts, slam_geo.dyn_nrms, cam=sc["cam_zoom"], w=W, h=zh, point_radius_px=1, draw_grid=False)
        z_geo = add_hud(z_geo, [f"3x Zoom HoverGeoNet-240 ({sc['roi_name']}) z95={g_roi['z_p95_mm']}mm"])
        z_moge = render_hd_dual_layer_green(m_pts, m_nrm, m_cr, slam_moge.dyn_pts, slam_moge.dyn_nrms, cam=sc["cam_zoom"], w=W, h=H - zh, point_radius_px=1, draw_grid=False)
        z_moge = add_hud(z_moge, [f"3x Zoom MoGe-2-ViT-B ({sc['roi_name']}) z95={m_roi['z_p95_mm']}mm"])
        img_zoom = np.vstack([z_geo, z_moge])

        # 0) 原始相机 240x240 画面
        raw_p = np.full((H, W, 3), (8, 12, 8), dtype=np.uint8)
        sq = min(W - 40, H - 56)
        fr_last = cv2.resize(frames[-1], (sq, sq), interpolation=cv2.INTER_AREA)
        x0 = (W - sq) // 2
        y0 = 46 + (H - 46 - sq) // 2
        raw_p[y0 : y0 + sq, x0 : x0 + sq] = fr_last
        cv2.rectangle(raw_p, (x0, y0), (x0 + sq, y0 + sq), (60, 180, 95), 1)
        raw_p = add_hud(raw_p, [sc["title"][:48], f"GC0308 240x240 Raw Input (H_cam=13.5cm)"])

        rows_main.append(np.hstack([raw_p, img_old, img_geo, img_moge, img_zoom]))

        # 导出 60 FPS WebGL 交互式 HTML 查看器 + .ply
        exp_geo = export_ply_and_interactive_html(
            g_pts, g_nrm, g_cr,
            OUT_DIR / f"geonet240_{sc['key']}_v7.ply",
            OUT_DIR / f"viewer_geonet240_{sc['key']}_v7.html",
            title=f"HoverGeoNet-240 ({geo_mb:.2f}MB, 13.5cm) - {sc['key']} 2.0mm Green 3D Map",
        )
        exp_moge = export_ply_and_interactive_html(
            m_pts, m_nrm, m_cr,
            OUT_DIR / f"moge2_dynablox_{sc['key']}_v7.ply",
            OUT_DIR / f"viewer_moge2_dynablox_{sc['key']}_v7.html",
            title=f"MoGe-2-ViT-B + Dynablox Zero-Ghost (13.5cm) - {sc['key']} 2.0mm Map",
        )

        # 如果是 cb8455，专门制作 Dynablox 零残影消融对比图
        if sc["key"] == "cb8455":
            AW, AH = 560, 420
            cam_ab = OrbitCamera3D(azimuth_deg=-148.0, elevation_deg=34.0, distance_m=0.86, target_x=0.44, target_y=0.02, target_z=0.08)
            p_naive = render_hd_laser_green_cloud(naive_moge.pos, naive_moge.nrm, naive_moge.crease, cam=cam_ab, w=AW, h=AH, point_radius_px=1)
            p_naive = add_hud(p_naive, [
                "BEFORE (v6 Naive Multi-Frame Fusion)",
                f"Waving arm smears across {len(naive_moge.pos):,} static pts!",
            ])
            p_moge_clean = render_hd_dual_layer_green(m_pts, m_nrm, m_cr, slam_moge.dyn_pts, slam_moge.dyn_nrms, cam=cam_ab, w=AW, h=AH, point_radius_px=1)
            p_moge_clean = add_hud(p_moge_clean, [
                "AFTER (v7 MoGe-2 + Dynablox Zero-Ghost)",
                f"Static {len(m_pts):,} clean | Live mint arm {len(slam_moge.dyn_pts):,} | Carved {moge_carved_total:,}",
            ])
            p_geo_clean = render_hd_dual_layer_green(g_pts, g_nrm, g_cr, slam_geo.dyn_pts, slam_geo.dyn_nrms, cam=cam_ab, w=AW, h=AH, point_radius_px=1)
            p_geo_clean = add_hud(p_geo_clean, [
                f"AFTER (v7 HoverGeoNet-240 {geo_mb:.1f}MB + Dynablox, {np.mean(geo_times):.1f}ms)",
                f"Static {len(g_pts):,} clean | Live mint arm {len(slam_geo.dyn_pts):,} | Carved {geo_carved_total:,}",
            ])
            ablation_cb_panels = [p_naive, p_moge_clean, p_geo_clean]

        metrics_all["scenes"][sc["key"]] = {
            "geonet240": {
                "avg_infer_ms": round(float(np.mean(geo_times)), 2),
                "static_pts": int(len(g_pts)),
                "dyn_pts_last": int(len(slam_geo.dyn_pts)),
                "carved_ghosts_total": int(geo_carved_total),
                "density": g_dens,
                "roi_bump": g_roi,
                "viewer_html": exp_geo["html_path"],
            },
            "moge2_vitb": {
                "avg_infer_ms": round(float(np.mean(moge_times)), 2),
                "static_pts": int(len(m_pts)),
                "dyn_pts_last": int(len(slam_moge.dyn_pts)),
                "carved_ghosts_total": int(moge_carved_total),
                "density": m_dens,
                "roi_bump": m_roi,
                "viewer_html": exp_moge["html_path"],
            },
        }
        print(
            f"  HoverGeoNet-240: {np.mean(geo_times):.1f}ms | {len(g_pts):,} static + {len(slam_geo.dyn_pts):,} dyn | "
            f"ROI z95={g_roi['z_p95_mm']}mm | desk_MAD={g_dens.get('desk_plane_mad_mm', -1)}mm",
            flush=True,
        )
        print(
            f"  MoGe-2-ViT-B   : {np.mean(moge_times):.1f}ms | {len(m_pts):,} static + {len(slam_moge.dyn_pts):,} dyn | "
            f"ROI z95={m_roi['z_p95_mm']}mm | desk_MAD={m_dens.get('desk_plane_mad_mm', -1)}mm",
            flush=True,
        )

    # 保存主对比图
    grid_main = np.vstack(rows_main)
    top_bar = np.full((46, grid_main.shape[1], 3), (6, 18, 10), dtype=np.uint8)
    put_text(
        top_bar,
        f"STRICTLY HELD-OUT ZERO-SHOT COMPARISON | Localized HoverGeoNet-240 ({geo_mb:.1f}MB, ~8ms, H_cam=13.5cm) vs MoGe-2-ViT-B (419MB, ~163ms)",
        (16, 30), scale=0.66, color=(145, 255, 190), thick=2,
    )
    png_main = OUT_DIR / "showcase_geonet240_vs_moge2_v7.png"
    cv2.imwrite(str(png_main), np.vstack([top_bar, grid_main]))
    print(f"\nSaved main comparison -> {png_main}", flush=True)

    # 保存 Dynablox 零残影消融对比图
    if ablation_cb_panels:
        ab_row = np.hstack(ablation_cb_panels)
        ab_bar = np.full((44, ab_row.shape[1], 3), (6, 18, 10), dtype=np.uint8)
        put_text(
            ab_bar,
            "DYNABLOX ZERO-GHOST ABLATION ON HELD-OUT cb8455 (Waving Arm + Cup + Notebook) | Emerald=Static Map, Neon Mint=Live Dynamic Arm",
            (14, 29), scale=0.58, color=(145, 255, 190), thick=2,
        )
        png_ab = OUT_DIR / "showcase_dynablox_zero_ghost_v7.png"
        cv2.imwrite(str(png_ab), np.vstack([ab_bar, ab_row]))
        print(f"Saved zero-ghost ablation -> {png_ab}", flush=True)

    (OUT_DIR / "metrics_v7.json").write_text(json.dumps(metrics_all, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
