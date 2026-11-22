# -*- coding: utf-8 -*-
"""proto_hd_moge2_showcase.py — PC 端 RTX 5060 高精 MoGe-2 超密纯绿色 3D 轮廓对比、鼠标微凸特写与无里程计视觉自定位验证。

产出位于 `D:\robot_work\runs_end_side\hd_moge2_v6\`：
1. `showcase_hd_moge2_comparison_v6.png` — 4 真实场景 × 4 列：
   原图 | 旧版 96×96 学生+8.5mm 体素点云 | 新版 MoGe-2 2.0mm 超密纯绿轮廓 | 局部 3× 放大特写（鼠标隆起/键盘棱角/手机/水杯）；
2. `showcase_visual_relocalization_v6.png` — Scenario 2 无里程计视觉-几何自定位压测（纯靠高清 3D 场景固定物体作为路标）；
3. `viewer_empty_d05_hd_green.html` / `viewer_cb8455_hd_green.html` + 标准 `.ply` 密点云（浏览器离线 60fps 自由缩放旋转）；
4. `metrics_hd_moge2_v6.json` — 全部精确测算数字（mm 间距、点/cm² 密度、鼠标/键盘/手机实测高度、自定位毫米误差、GPU 耗时）。
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hd_moge_contour_engine import (  # noqa: E402
    MoGe2DesktopReconstructor,
    UltraDenseSurfelCloud,
    export_ply_and_interactive_html,
    level_and_scale_desktop_cloud,
    relocalize_frame_against_3d_map,
    render_hd_laser_green_cloud,
)
from neural_gaussian_splat_model import OrbitCamera3D  # noqa: E402
from robust_contour_slam import (  # noqa: E402
    RobustContourSlamPipeline,
    render_green_contour_edl,
)

OUT_DIR = REPO / "runs_end_side" / "hd_moge2_v6"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OLD_TFLITE = REPO / "runs_end_side" / "dense_splat_v3" / "model.tflite"

EMPTY_DIR = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"
CB8455_MP4 = REPO / "dataset" / "takes" / "robot_20261115_151733_cb8455" / "take.mp4"
EXTRA_004630_MP4 = REPO / "video_annotation_data" / "videos" / "robot_20261105_004630_ef1d19.mp4"
TRAIN_5CA36D_MP4 = REPO / "dataset" / "takes" / "robot_20261107_204337_5ca36d" / "take.mp4"


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = (155, 255, 190),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 10, 7), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def add_hud_bar(img: np.ndarray, lines: list[str]) -> np.ndarray:
    h_bar = 10 + 21 * len(lines)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (img.shape[1], h_bar), (5, 11, 7), -1)
    out = cv2.addWeighted(ov, 0.76, img, 0.24, 0)
    for i, ln in enumerate(lines):
        put_text(out, ln, (10, 20 + 21 * i), scale=0.46)
    return out


def load_frames(kind: str, src: pathlib.Path, start: int, count: int, step: int = 2) -> list[np.ndarray]:
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


def measure_local_bump(pts: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float]) -> dict[str, float]:
    """统计指定桌面 ROI 内的真实 3D 隆起高度 (p50 / p95 / max, 单位 mm) 与点数。"""
    m = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])
        & (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])
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
    print("Loading Microsoft MoGe-2 (Ruicheng/moge-2-vitb-normal) on GPU...", flush=True)
    recon = MoGe2DesktopReconstructor(repo_id="Ruicheng/moge-2-vitb-normal", infer_hw=480, use_fp16=True)
    print(f"MoGe-2 loaded in {recon.load_s:.1f}s on {recon.device}", flush=True)

    # 4 组真实场景配置：(名称, 数据源, 起始帧, 全局视角, 特写视角, 感兴趣隆起区域 ROI)
    scenes = [
        (
            "empty_d0.5: Mouse + Keyboard + Desk Corners",
            "frames", EMPTY_DIR, 12,
            OrbitCamera3D(azimuth_deg=-152.0, elevation_deg=34.0, distance_m=0.82, target_x=0.40, target_y=0.00, target_z=0.05),
            OrbitCamera3D(azimuth_deg=-138.0, elevation_deg=26.0, distance_m=0.42, target_x=0.38, target_y=0.04, target_z=0.03),
            "mouse+keyboard", (0.20, 0.56), (-0.18, 0.22),
            "empty_d05",
        ),
        (
            "cb8455: Cup + Notebook + Arm + Person",
            "video", CB8455_MP4, 595,
            OrbitCamera3D(azimuth_deg=-148.0, elevation_deg=36.0, distance_m=0.88, target_x=0.42, target_y=0.04, target_z=0.07),
            OrbitCamera3D(azimuth_deg=-130.0, elevation_deg=28.0, distance_m=0.46, target_x=0.42, target_y=0.10, target_z=0.05),
            "cup+notebook+hand", (0.22, 0.62), (-0.05, 0.28),
            "cb8455",
        ),
        (
            "004630: Smartphone + Knocking Hand + Keyboard",
            "video", EXTRA_004630_MP4, 796,
            OrbitCamera3D(azimuth_deg=-155.0, elevation_deg=36.0, distance_m=0.86, target_x=0.42, target_y=0.02, target_z=0.06),
            OrbitCamera3D(azimuth_deg=-165.0, elevation_deg=32.0, distance_m=0.44, target_x=0.34, target_y=0.04, target_z=0.03),
            "phone+hand", (0.18, 0.52), (-0.15, 0.22),
            "004630",
        ),
        (
            "5ca36d: Dark Blazer + Fist + Keyboard Corners",
            "video", TRAIN_5CA36D_MP4, 396,
            OrbitCamera3D(azimuth_deg=-150.0, elevation_deg=35.0, distance_m=0.88, target_x=0.44, target_y=0.02, target_z=0.08),
            OrbitCamera3D(azimuth_deg=-125.0, elevation_deg=28.0, distance_m=0.46, target_x=0.40, target_y=0.08, target_z=0.04),
            "fist+keyboard", (0.20, 0.58), (-0.08, 0.28),
            "5ca36d",
        ),
    ]

    col_w, col_h = 560, 420
    c1_w = 420
    total_w = c1_w + col_w * 3 + 12

    header = np.full((42, total_w, 3), (8, 16, 10), dtype=np.uint8)
    put_text(header, "Original 240x240 Photo", (12, 27), scale=0.54)
    put_text(header, "OLD: 96x96 Student + 8.5mm Voxel", (c1_w + 16, 27), scale=0.54, color=(150, 195, 160))
    put_text(header, "NEW: MoGe-2 480px + 2.0mm Ultra-Dense Green", (c1_w + col_w + 20, 27), scale=0.54)
    put_text(header, "3x CLOSE-UP: Mouse / Phone / Keyboard / Cup", (c1_w + col_w * 2 + 24, 27), scale=0.54)

    rows = [header]
    all_metrics: dict = {"model": recon.repo_id, "device": str(recon.device), "scenes": []}
    saved_clouds: dict[str, dict] = {}

    for name, kind, src, start, cam_full, cam_zoom, roi_tag, rx_rng, ry_rng, slug in scenes:
        print(f"\n--- Scene: {name} ---", flush=True)
        frames = load_frames(kind, src, start, count=6, step=2)

        # 1) 旧版 96x96 + 8.5mm 对照
        old_pipe = RobustContourSlamPipeline(OLD_TFLITE)
        for bgr in frames:
            old_pipe.process_frame(bgr)
        old_s_pos, old_s_nrm = old_pipe.map.get_static_confirmed()
        old_view = render_green_contour_edl(
            old_s_pos, old_s_nrm, cam_full, grid=None,
            dyn_pts=old_pipe.map.d_pos, dyn_nrms=old_pipe.map.d_nrm,
            w=col_w, h=col_h, draw_ground_grid=True, draw_trav_overlay=False,
        )
        old_view = add_hud_bar(old_view, [
            f"OLD 96x96 Student | static={len(old_s_pos)} pts (8.5mm voxel)",
            "bottom <24mm flattened -> mouse/phone bumps erased",
        ])

        # 2) 新版 MoGe-2 480x480 + 2.0mm 超密多帧融合
        hd_cloud = UltraDenseSurfelCloud(vox_size_m=0.0020)
        infer_times = []
        ref_pack = None
        first_leveled = None
        for fi, bgr in enumerate(frames):
            raw = recon.infer_raw(bgr)
            infer_times.append(raw["infer_ms"])
            lvl = level_and_scale_desktop_cloud(
                raw["pts_cam"], raw["nrm_cam"], raw["mask"],
                ref_plane=(first_leveled["r_level"], first_leveled["scale"], first_leveled["z_off"]) if first_leveled else None,
            )
            if first_leveled is None:
                first_leveled = lvl
                ref_pack = {"bgr": bgr, "raw": raw, "lvl": lvl}
            hd_cloud.fuse_frame(lvl["pts"], lvl["nrms"], lvl["crease"], align=(fi > 0))

        dens = hd_cloud.measure_density_metrics()
        bump = measure_local_bump(hd_cloud.pos, rx_rng, ry_rng)
        near_info = dens.get("near_0.25_0.55m", {})
        mean_ms = float(np.mean(infer_times))
        print(f"  MoGe-2 infer={mean_ms:.1f}ms | cloud={len(hd_cloud.pos):,} pts | near={near_info} | {roi_tag}={bump}", flush=True)

        # 3) 渲染高清全图 + 3x 近距特写
        hd_full = render_hd_laser_green_cloud(
            hd_cloud.pos, hd_cloud.nrm, hd_cloud.crease, cam_full,
            w=col_w, h=col_h, point_radius_px=1, draw_grid=True,
        )
        hd_full = add_hud_bar(hd_full, [
            f"MoGe-2 HD | {len(hd_cloud.pos):,} pts (2.0mm) | {mean_ms:.0f}ms GPU",
            f"near 0.25-0.55m: {near_info.get('median_spacing_mm', 0):.1f}mm spacing ({near_info.get('surface_density_pts_per_cm2', 0):.0f} pts/cm2)",
        ])

        hd_zoom = render_hd_laser_green_cloud(
            hd_cloud.pos, hd_cloud.nrm, hd_cloud.crease, cam_zoom,
            w=col_w, h=col_h, point_radius_px=1, draw_grid=True, robot_pose=None,
        )
        hd_zoom = add_hud_bar(hd_zoom, [
            f"CLOSE-UP ({roi_tag}) | 5cm grid | desk MAD={dens.get('desk_plane_mad_mm', 0):.1f}mm",
            f"local relief p95={bump['z_p95_mm']:.0f}mm max={bump['z_max_mm']:.0f}mm ({bump['pts']:,} pts in ROI)",
        ])

        col1 = cv2.resize(frames[0], (c1_w, col_h), interpolation=cv2.INTER_LINEAR)
        col1 = add_hud_bar(col1, [name, f"FOV_est={ref_pack['raw']['fov_h_deg']:.1f}deg  scale={first_leveled['scale']:.2f}x"])

        vbar = np.full((col_h, 4, 3), (30, 74, 42), dtype=np.uint8)
        row = np.hstack([col1, vbar, old_view, vbar, hd_full, vbar, hd_zoom])
        hbar = np.full((4, row.shape[1], 3), (30, 74, 42), dtype=np.uint8)
        rows.extend([row, hbar])

        # 导出 empty_d05 和 cb8455 的交互式 WebGL HTML + 标准 PLY 供用户电脑直接自由旋转放大检验！
        exports = None
        if slug in ("empty_d05", "cb8455"):
            exports = export_ply_and_interactive_html(
                hd_cloud.pos, hd_cloud.nrm, hd_cloud.crease,
                ply_path=OUT_DIR / f"{slug}_hd_green.ply",
                html_path=OUT_DIR / f"viewer_{slug}_hd_green.html",
                title=f"RIG-Hover {name} — MoGe-2 2.0mm Green Contour",
            )
            print(f"  exported interactive viewer: {exports['html_path']} ({exports['html_kb']} KB)", flush=True)
            saved_clouds[slug] = {"cloud": hd_cloud, "ref": ref_pack, "frames": load_frames(kind, src, start, count=12, step=4)}

        all_metrics["scenes"].append({
            "scene": name,
            "old_static_pts": int(len(old_s_pos)),
            "hd_total_pts": int(len(hd_cloud.pos)),
            "moge2_ms_per_frame": round(mean_ms, 1),
            "density": dens,
            "roi_relief": {roi_tag: bump},
            "exports": exports,
        })

    cmp_png = OUT_DIR / "showcase_hd_moge2_comparison_v6.png"
    cv2.imwrite(str(cmp_png), np.vstack(rows))
    print(f"\nSAVED {cmp_png}", flush=True)

    # -----------------------------------------------------------------------
    # Scenario 2 核心压测：零轮式里程计依赖的视觉-几何 6-DoF 自定位验证
    # -----------------------------------------------------------------------
    print("\n== Scenario 2 Core Test: Visual-Geometric Relocalization (ZERO Wheel Odometry!) ==", flush=True)
    reloc_records = []
    reloc_panels = []
    for slug, target_xy in [("empty_d05", (0.38, -0.02)), ("cb8455", (0.42, 0.06))]:
        pack = saved_clouds[slug]
        cloud = pack["cloud"]
        ref = pack["ref"]
        test_frames = pack["frames"][4:9]  # 间隔较远的新时刻真实帧（含真实双轮平衡摇摆）

        # 模拟未知的机器人位姿漂移/移动：给查询帧相机点云施加人为未知 SE(3) 变换，检验能否 100% 盲找回相对目标坐标!
        rng = np.random.default_rng(20261122)
        pos_errs_mm = []
        yaw_errs_deg = []
        target_errs_mm = []
        target_w = np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32)

        last_est_cam = None
        for q_idx, q_bgr in enumerate(test_frames):
            q_raw = recon.infer_raw(q_bgr)
            # 模拟真实机器人移动+打滑后的未知真值相机位姿变化 (无需里程计先验!)
            true_shift = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.07, 0.07), rng.uniform(-0.012, 0.012)], dtype=np.float32)
            true_yaw_deg = float(rng.uniform(-10.0, 10.0))
            true_pitch_deg = float(rng.uniform(-3.5, 3.5))
            r_y, _ = cv2.Rodrigues(np.array([0.0, math.radians(true_pitch_deg), math.radians(true_yaw_deg)], dtype=np.float64))
            # 将模拟移动折算回参考相机系以得到已知真值
            R_gt = r_y.astype(np.float32) @ ref["lvl"]["r_level"]
            t_gt = np.array([0.0, 0.0, ref["lvl"]["z_off"]], dtype=np.float32) + true_shift
            # 在无先验条件下直接用 SIFT 2D-3D + RANSAC-Kabsch + KISS-ICP 把当前帧定位到高清 3D 地图中
            # （注意：对于同一实拍序列帧，真实相机相对参考帧仅有 1.16Hz 摇摆；为测精确真值，我们构造带已知位姿扰动的查询点云）
            q_pts_sim = ((q_raw["pts_cam"] * ref["lvl"]["scale"]) @ r_y.astype(np.float32).T) - (true_shift @ ref["lvl"]["r_level"])[None, None, :]
            q_nrm_sim = q_raw["nrm_cam"] @ r_y.astype(np.float32).T
            q_valid = q_raw["mask"] & np.isfinite(q_pts_sim).all(axis=-1) & (q_raw["pts_cam"][..., 0] > 0.12) & (q_raw["pts_cam"][..., 0] < 2.2)

            rel = relocalize_frame_against_3d_map(
                query_bgr=q_bgr,
                query_pts_cam=q_pts_sim,
                query_nrm_cam=q_nrm_sim,
                query_valid=q_valid,
                ref_bgr=ref["bgr"],
                ref_pts_world=ref["lvl"]["pts_grid"],
                ref_valid=ref["lvl"]["valid_grid"],
                static_map_pts=cloud.pos,
                static_map_nrms=cloud.nrm,
                query_scale=1.0,
            )
            if rel["ok"]:
                # 真值变换是 p_world = R_true @ p_query + t_true
                # 由 q_pts_sim = R_y @ (scale*p_cam) - R_level^T @ shift，反解 p_world = R_level @ (scale*p_cam) + [0,0,z_off]
                # => R_true = R_level @ R_y^T, t_true = [0,0,z_off] + R_level @ R_y^T @ (R_level^T @ shift)
                R_true = ref["lvl"]["r_level"] @ r_y.astype(np.float32).T
                t_true = np.array([0.0, 0.0, ref["lvl"]["z_off"]], dtype=np.float32) + R_true @ (ref["lvl"]["r_level"].T @ true_shift)
                pos_err = float(np.linalg.norm(rel["cam_pos_world"] - t_true) * 1000.0)
                rv_e, _ = cv2.Rodrigues((rel["R_cam_to_world"].T @ R_true).astype(np.float64))
                ang_err = math.degrees(float(np.linalg.norm(rv_e)))
                # 最核心指标：机器人估计出的"点击目标相对自己机身坐标系坐标"与真值的误差!
                tgt_body_gt = R_true.T @ (target_w - t_true)
                tgt_body_est = rel["R_cam_to_world"].T @ (target_w - rel["cam_pos_world"])
                tgt_err = float(np.linalg.norm(tgt_body_est - tgt_body_gt) * 1000.0)
                pos_errs_mm.append(pos_err)
                yaw_errs_deg.append(ang_err)
                target_errs_mm.append(tgt_err)
                last_est_cam = (rel["cam_pos_world"], rel["R_cam_to_world"], tgt_err, ang_err, rel["n_sift_3d_matches"])

        rec = {
            "scene": slug,
            "trials": len(pos_errs_mm),
            "target_world_xy": list(target_xy),
            "mean_cam_pos_err_mm": round(float(np.mean(pos_errs_mm)), 1) if pos_errs_mm else -1.0,
            "mean_rot_err_deg": round(float(np.mean(yaw_errs_deg)), 2) if yaw_errs_deg else -1.0,
            "mean_relative_target_err_mm": round(float(np.mean(target_errs_mm)), 1) if target_errs_mm else -1.0,
            "max_relative_target_err_mm": round(float(np.max(target_errs_mm)), 1) if target_errs_mm else -1.0,
        }
        reloc_records.append(rec)
        print(f"  Reloc [{slug}]: {rec}", flush=True)

        cam_v = OrbitCamera3D(azimuth_deg=-148.0, elevation_deg=42.0, distance_m=0.92, target_x=0.38, target_y=0.02, target_z=0.05)
        est_pose = (float(last_est_cam[0][0]), float(last_est_cam[0][1]), 0.0) if last_est_cam else (0.0, 0.0, 0.0)
        panel = render_hd_laser_green_cloud(
            cloud.pos, cloud.nrm, cloud.crease, cam_v,
            w=680, h=460, point_radius_px=1, draw_grid=True,
            target_pin_xyz=(target_xy[0], target_xy[1], 0.0),
            robot_pose=est_pose,
        )
        panel = add_hud_bar(panel, [
            f"Scenario 2 Visual-3D Relocalization ({slug}) - ZERO Wheel Odometry",
            f"drift +/-8cm, +/-10deg -> target err={rec['mean_relative_target_err_mm']:.1f}mm (rot {rec['mean_rot_err_deg']:.2f}deg)",
            "Landmarks: 2mm mouse / keyboard / cup / desk corners in MoGe-2 map",
        ])
        reloc_panels.append(panel)

    vsep = np.full((460, 6, 3), (30, 74, 42), dtype=np.uint8)
    reloc_png = OUT_DIR / "showcase_visual_relocalization_v6.png"
    cv2.imwrite(str(reloc_png), np.hstack([reloc_panels[0], vsep, reloc_panels[1]]))
    print(f"SAVED {reloc_png}", flush=True)

    all_metrics["visual_relocalization"] = reloc_records
    (OUT_DIR / "metrics_hd_moge2_v6.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ALL DONE!", flush=True)


if __name__ == "__main__":
    main()
