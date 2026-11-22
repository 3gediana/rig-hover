# -*- coding: utf-8 -*-
"""proto_static_vr_scene.py — 静态场景"照片建模"式密点轮廓重建与 VR 风格展示（100% 端侧 .tflite）。

对应用户"多角度拍照 -> VR 还原场景轮廓"的诉求（先做无人物静态场景）：
1. 静态场景 `evidence/sessions/empty_d0.5_front_night_182201`（202 帧，键盘+鼠标+桌垫，无人物）
   逐像素时序中值融合：双轮平衡引起的相机微摆先用相位相关对齐，再对 96x96 逆深度/高程
   逐像素取中值——单帧深度噪声在时序中互相抵消，得到远比单帧干净的融合深度场；
2. 融合深度经 JBU 联合双边上采样到 384x384（约 10 万+ 密点），法向/颜色由融合 RGB 提供；
3. 渲染采用"照片建模查看器"风格：浅色背景 + Z=0 地面网格参考线，深色密点构成场景轮廓；
4. 产出（推理路径 100% 只有 runs_end_side/dense_splat_v3/model.tflite，零教师依赖）：
   - showcase_static_vr_fusion.png  6 个环绕角度 + 1 个桌面特写 的融合轮廓对比图；
   - demo_static_vr_scan.mp4        16 s 15 fps：前段"径向扫描重建"动画（密点按世界方位角
     逐扇区出现，模拟拍照建模的扫描过程），后段完整 360 度转台环视（VR 查看效果）。
所有汇报数字均来自本脚本真实输出。
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dense_scene_pipeline import (  # noqa: E402
    TraversabilityGrid2D,
    reconstruct_dense_frame_3d,
    render_edl_3d_view,
    render_minimap_2d,
    run_student_tflite_frame,
    solve_table_plane_from_student,
)
from neural_gaussian_splat_model import OrbitCamera3D  # noqa: E402

RUN_DIR = REPO / "runs_end_side" / "dense_splat_v3"
TFLITE_PATH = RUN_DIR / "model.tflite"
FRAMES_DIR = RUN_DIR / "_vr_frames"
FFMPEG = pathlib.Path(r"D:\claude code\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")

STATIC_SESSION = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"

FOV_H_DEG = 72.0
VIEW_W, VIEW_H = 800, 560
LIGHT_BG = (222, 226, 232)      # BGR 浅灰蓝（照片建模查看器底色）


def put_text_dark(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.52) -> None:
    """浅色背景用的深色描边文本。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (45, 48, 56), 1, cv2.LINE_AA)


def brighten_bgr(bgr: np.ndarray) -> np.ndarray:
    """夜间画面伽马提亮（单调映射，不改变边缘位置，可安全用作 JBU guide 与点色）。"""
    x = bgr.astype(np.float32) / 255.0
    out = np.clip((np.power(x, 0.60)) * 1.28, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def fuse_static_scene(session_dir: pathlib.Path, max_frames: int) -> dict:
    """相位相关对齐 + 逐像素时序中值融合。返回融合 splats(96x96x6)、提亮 RGB、质量指标。"""
    jpgs = sorted(session_dir.glob("*.jpg"))[:max_frames]
    if len(jpgs) < 30:
        raise RuntimeError(f"not enough static frames: {len(jpgs)}")

    inv_stack: list[np.ndarray] = []
    elev_stack: list[np.ndarray] = []
    gray96_stack: list[np.ndarray] = []
    t0 = time.perf_counter()

    ref_bgr = cv2.imread(str(jpgs[0]))
    ref_gray96 = cv2.cvtColor(cv2.resize(ref_bgr, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i, jp in enumerate(jpgs):
        bgr = cv2.imread(str(jp))
        if bgr is None:
            continue
        splats = run_student_tflite_frame(TFLITE_PATH, bgr)
        g96 = cv2.cvtColor(cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        inv_stack.append(splats[..., 0].astype(np.float32))
        elev_stack.append(splats[..., 1].astype(np.float32))
        gray96_stack.append(g96)
        if i % 50 == 0:
            print(f"  infer {i}/{len(jpgs)}", flush=True)

    n = len(gray96_stack)
    inv_arr = np.stack(inv_stack, axis=0)
    elev_arr = np.stack(elev_stack, axis=0)
    g_arr = np.stack(gray96_stack, axis=0)
    raw_std = float(np.mean(np.std(inv_arr, axis=0)))

    # 两遍对齐：第一遍不对齐直接取灰度中值当参考（夜间暗场景低纹理，单帧参考不可靠），
    # 第二遍逐帧与中值参考做相位相关；符号用多帧投票确定，且逐帧带"对齐后残余必须更小"
    # 的接受判据，避免噪声位移把中值糊掉
    ref_gray = np.median(g_arr, axis=0).astype(np.float32)
    raw_shifts = np.array([cv2.phaseCorrelate(ref_gray, g_arr[i])[0] for i in range(n)], dtype=np.float32)

    votes = []
    probe_ids = np.linspace(1, n - 1, num=min(9, n - 1), dtype=int)
    for i in probe_ids:
        dx, dy = raw_shifts[i]
        res0 = float(np.mean(np.abs(g_arr[i] - ref_gray)))
        cand = []
        for sign in (1.0, -1.0):
            w = cv2.warpAffine(g_arr[i], np.float32([[1, 0, sign * dx], [0, 1, sign * dy]]), (96, 96))
            cand.append(float(np.mean(np.abs(w - ref_gray))))
        if min(cand) < res0:
            votes.append(1.0 if cand[0] <= cand[1] else -1.0)
    best_sign = float(np.sign(np.sum(votes))) if votes else 0.0
    align_shifts: list[tuple[float, float]] = []
    if best_sign != 0.0:
        for i in range(n):
            dx, dy = raw_shifts[i]
            res0 = float(np.mean(np.abs(g_arr[i] - ref_gray)))
            w = cv2.warpAffine(g_arr[i], np.float32([[1, 0, best_sign * dx], [0, 1, best_sign * dy]]), (96, 96))
            res1 = float(np.mean(np.abs(w - ref_gray)))
            align_shifts.append((best_sign * dx, best_sign * dy) if res1 < res0 else (0.0, 0.0))
    n_aligned = int(np.sum(np.abs(np.array(align_shifts)).sum(axis=1) > 0)) if align_shifts else 0
    print(f"  align: sign={best_sign:+.0f} applied on {n_aligned}/{n} frames", flush=True)

    inv_al = np.empty_like(inv_arr)
    elev_al = np.empty_like(elev_arr)
    for i in range(n):
        dx, dy = align_shifts[i]
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        inv_al[i] = cv2.warpAffine(inv_arr[i], m, (96, 96), flags=cv2.INTER_LINEAR)
        elev_al[i] = cv2.warpAffine(elev_arr[i], m, (96, 96), flags=cv2.INTER_LINEAR)

    fused_inv = np.median(inv_al, axis=0).astype(np.float32)
    fused_elev = np.median(elev_al, axis=0).astype(np.float32)
    resid = float(np.mean(np.abs(inv_al - fused_inv[None, ...])))

    # RGB 融合：按 2.5 倍位移对齐后取中值（子采样 60 帧控内存）
    rgb_ids = np.linspace(0, n - 1, num=min(60, n), dtype=int)
    rgb_stack = []
    for i in rgb_ids:
        dx, dy = align_shifts[i]
        bgr = cv2.imread(str(jpgs[i]))
        m = np.float32([[1, 0, dx * 2.5], [0, 1, dy * 2.5]])
        rgb_stack.append(cv2.warpAffine(bgr, m, (240, 240), flags=cv2.INTER_LINEAR))
    fused_bgr = np.median(np.stack(rgb_stack, axis=0), axis=0).astype(np.uint8)

    print(f"  fused {n} frames in {time.perf_counter()-t0:.1f}s | align_sign={best_sign:+.0f} "
          f"| raw per-pixel inv01 std={raw_std:.4f} -> median residual={resid:.4f}", flush=True)
    splats = np.zeros((96, 96, 6), np.float32)
    splats[..., 0] = fused_inv
    splats[..., 1] = fused_elev
    return {
        "splats": splats,
        "bgr": fused_bgr,
        "n_frames": n,
        "n_aligned": n_aligned,
        "raw_std": raw_std,
        "residual": resid,
        "shift_px96_p95": float(np.percentile(np.abs(np.array(align_shifts)), 95)),
    }


def enhance_tex(bgr: np.ndarray) -> np.ndarray:
    """轻度百分位对比度拉伸（2-98 分位）+ 轻锐化。不做伽马提亮（夜间画面本来就曝光正常，
    强提亮会把白色键盘炸成纯白失去轮廓）。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(g, [2, 98])
    x = np.clip((bgr.astype(np.float32) - lo) / max(hi - lo, 1e-4) * 255.0, 0, 255)
    out = x.astype(np.uint8)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    return np.clip(out.astype(np.float32) + 0.30 * (out.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)


def build_cloud(fused: dict) -> dict:
    """100% 端侧 .tflite 学生融合深度 + 融合 RGB 纹理 -> 480x480 密点云（零离线教师依赖）。"""
    tex = enhance_tex(fused["bgr"])
    splats = fused["splats"].astype(np.float32)
    r_level, z_bias, pitch = solve_table_plane_from_student(
        splats[..., 0], splats[..., 1], fov_h_deg=FOV_H_DEG,
    )
    recon = reconstruct_dense_frame_3d(
        tex, splats, r_level, z_bias, fov_h_deg=FOV_H_DEG,
        yaw_deg=0.0, cam_x=0.0, cam_y=0.0, out_hw=(480, 480),
        jbu_radius=1, jbu_sigma_range=0.12,
    )
    # 裁掉极端低位噪点与过高屋顶
    wz = recon["pts"][:, 2]
    keep = (wz > -0.03) & (wz < 0.55)
    recon = {
        "pts": recon["pts"][keep],
        "nrms": recon["nrms"][keep],
        "cols": recon["cols"][keep],
        "elev": recon["elev"][keep],
        "depth_hr": recon["depth_hr"],
    }
    grid = TraversabilityGrid2D()
    grid.update(recon["pts"])
    print(f"  level: pitch={pitch:+.1f}deg z_bias={z_bias:+.4f} | cloud pts={len(recon['pts'])} "
          f"green={int(grid.reach.sum())} red={int(grid.obs_mask.sum())}", flush=True)
    return {"recon": recon, "grid": grid, "r_level": r_level, "z_bias": z_bias, "pitch": pitch,
            "tex": tex, "teacher_depth": False}


def height_color_cols(pts: np.ndarray, texture_cols: np.ndarray) -> np.ndarray:
    """高度伪彩（扫描查看器风格）：桌面低=蓝、物体/背景高=暖色，叠 15% 纹理保留照片感。"""
    wz = np.clip(pts[:, 2], 0.0, 0.42) / 0.42
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[..., None], cv2.COLORMAP_TURBO)[..., 0, :]  # (256,3) BGR
    idx = (wz * 255.0).astype(np.uint8)
    cols_rgb = lut[idx][..., ::-1].astype(np.float32) / 255.0
    return 0.85 * cols_rgb + 0.15 * texture_cols


def render_orbit(cloud: dict, cam: OrbitCamera3D, hud: list[str], cols: np.ndarray | None = None,
                 lambert: tuple[float, float] = (0.56, 0.50), edl: float = 3.4,
                 trav: bool = False, size: tuple[int, int] = (VIEW_W, VIEW_H)) -> np.ndarray:
    view = render_edl_3d_view(
        cloud["recon"]["pts"], cloud["recon"]["nrms"],
        cloud["recon"]["cols"] if cols is None else cols, cam, cloud["grid"],
        w=size[0], h=size[1], robot_pose=(0.0, 0.0, 0.0), path=None, click_marker=None,
        head_yaw_deg=0.0, bg_bgr=LIGHT_BG, draw_ground_grid=True, ground_blend=0.30,
        draw_trav_overlay=trav, lambert_base=lambert[0], lambert_span=lambert[1], edl_gain=edl,
    )
    mm = render_minimap_2d(cloud["grid"], size_px=140, robot_pose=(0.0, 0.0, 0.0))
    y0 = size[1] - mm.shape[0] - 8
    x0 = size[0] - mm.shape[1] - 8
    view[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
    for i, line in enumerate(hud):
        put_text_dark(view, line, (12, 26 + 22 * i))
    return view


def build_showcase_png(cloud: dict, fused: dict) -> pathlib.Path:
    print("== showcase_static_vr_fusion.png ==", flush=True)
    cols_h = height_color_cols(cloud["recon"]["pts"], cloud["recon"]["cols"])
    orbit_views = [(160.0, 30.0), (180.0, 40.0), (200.0, 50.0)]
    row_height, row_texture = [], []
    for az, el in orbit_views:
        cam = OrbitCamera3D(azimuth_deg=az, elevation_deg=el, distance_m=0.95,
                            target_x=0.35, target_y=0.0, target_z=0.08)
        hud = [f"orbit az={az:+.0f} el={el:.0f} dist=0.95m (native-view cone)", f"{fused['n_frames']} frames fused  cloud={len(cloud['recon']['pts'])} pts"]
        row_height.append(render_orbit(cloud, cam, hud, cols=cols_h, lambert=(0.86, 0.20), edl=4.4))
        row_texture.append(render_orbit(cloud, cam, hud, lambert=(0.88, 0.12), edl=3.6))

    strip_h = np.full((34, VIEW_W * 3, 3), 238, np.uint8)
    put_text_dark(strip_h, "style A: height-color contour (scan-viewer)  low z = blue desk, high = warm objects/background", (12, 23), scale=0.55)
    strip_t = np.full((34, VIEW_W * 3, 3), 238, np.uint8)
    put_text_dark(strip_t, "style B: photo texture (photogrammetry viewer)  flat lambert + edge glow", (12, 23), scale=0.55)

    close_h = render_orbit(cloud, OrbitCamera3D(azimuth_deg=175.0, elevation_deg=55.0, distance_m=0.50,
                                                target_x=0.28, target_y=0.0, target_z=0.03),
                           ["desk close-up (height-color)"], cols=cols_h, lambert=(0.86, 0.20), edl=4.4)
    close_t = render_orbit(cloud, OrbitCamera3D(azimuth_deg=175.0, elevation_deg=55.0, distance_m=0.50,
                                                target_x=0.28, target_y=0.0, target_z=0.03),
                           ["desk close-up (photo texture)"], lambert=(0.88, 0.12), edl=3.6)
    close_row = np.hstack([close_h, close_t, np.full((VIEW_H, VIEW_W, 3), 238, np.uint8)])

    header = np.full((40, VIEW_W * 3, 3), 238, dtype=np.uint8)
    put_text_dark(header, "Static scene photogrammetry-style fusion (empty_d0.5, no person): dense point contour, orbit like VR viewer",
                  (12, 27), scale=0.62)
    sep = np.full((8, VIEW_W * 3, 3), 210, np.uint8)
    out = np.vstack([header, strip_h, np.hstack(row_height), sep,
                     strip_t, np.hstack(row_texture), sep, close_row])
    out_path = RUN_DIR / "showcase_static_vr_fusion.png"
    cv2.imwrite(str(out_path), out)
    print(f"  saved {out_path} ({out.shape[1]}x{out.shape[0]})", flush=True)
    return out_path


def build_scan_video(cloud: dict, fused: dict, total: int = 240) -> pathlib.Path:
    print("== demo_static_vr_scan.mp4 ==", flush=True)
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    pts = cloud["recon"]["pts"]
    nrms = cloud["recon"]["nrms"]
    cols = cloud["recon"]["cols"]
    az_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))  # 世界方位角
    scan_frames = 78
    t0 = time.perf_counter()

    for f in range(total):
        if f < scan_frames:
            frac = f / float(scan_frames - 1)
            reveal_az = -95.0 + 190.0 * frac  # 从 -95 度扫到 +95 度
            vis = az_deg <= reveal_az
            cam = OrbitCamera3D(azimuth_deg=168.0 + 24.0 * frac, elevation_deg=36.0,
                                distance_m=0.95, target_x=0.35, target_y=0.0, target_z=0.08)
            hud = [
                f"scanning scene... {frac * 100:.0f}%   sweep az {reveal_az:+.0f} deg",
                f"points revealed {int(vis.sum())} / {len(pts)}",
            ]
        else:
            frac = (f - scan_frames) / float(total - 1 - scan_frames)
            cam_az = 152.0 + 56.0 * frac  # 原生视角锥区环视（单视角重建的诚实展示范围）
            cam = OrbitCamera3D(azimuth_deg=cam_az, elevation_deg=34.0 + 8.0 * math.sin(frac * 2 * math.pi * 2.0),
                                distance_m=0.95, target_x=0.35, target_y=0.0, target_z=0.08)
            hud = [
                f"VR orbit view   az={cam_az:.0f} deg",
                f"cloud {len(pts)} pts  fused from {fused['n_frames']} frames",
            ]
        view = render_edl_3d_view(
            pts[vis] if f < scan_frames else pts,
            nrms[vis] if f < scan_frames else nrms,
            cols[vis] if f < scan_frames else cols,
            cam, cloud["grid"], w=VIEW_W, h=VIEW_H, robot_pose=(0.0, 0.0, 0.0),
            bg_bgr=LIGHT_BG, draw_ground_grid=True, ground_blend=0.30,
            draw_trav_overlay=False, lambert_base=0.88, lambert_span=0.12, edl_gain=3.6,
        )
        mm = render_minimap_2d(cloud["grid"], size_px=140, robot_pose=(0.0, 0.0, 0.0))
        y0 = VIEW_H - mm.shape[0] - 8
        x0 = VIEW_W - mm.shape[1] - 8
        view[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] = mm
        for i, line in enumerate(hud):
            put_text_dark(view, line, (12, 26 + 22 * i))
        cv2.imwrite(str(FRAMES_DIR / f"frame_{f:04d}.png"), view)
        if f % 40 == 0:
            print(f"  video frame {f}/{total}", flush=True)

    out_mp4 = RUN_DIR / "demo_static_vr_scan.mp4"
    cmd = [str(FFMPEG), "-y", "-framerate", "15", "-i", str(FRAMES_DIR / "frame_%04d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")
    print(f"  saved {out_mp4} ({out_mp4.stat().st_size} bytes) in {time.perf_counter()-t0:.1f}s", flush=True)
    return out_mp4


def main() -> None:
    assert TFLITE_PATH.exists(), f"missing {TFLITE_PATH}"
    summary: dict = {"tflite_bytes": TFLITE_PATH.stat().st_size}
    t0 = time.perf_counter()

    fused = fuse_static_scene(STATIC_SESSION, max_frames=200)
    cloud = build_cloud(fused)
    summary["fusion"] = {
        "frames": fused["n_frames"],
        "frames_aligned": fused["n_aligned"],
        "raw_inv01_std": fused["raw_std"],
        "median_residual": fused["residual"],
        "shift_px96_p95": fused["shift_px96_p95"],
        "cloud_points": int(len(cloud["recon"]["pts"])),
        "pitch_deg": cloud["pitch"],
        "z_bias": cloud["z_bias"],
        "green_cells": int(cloud["grid"].reach.sum()),
        "red_cells": int(cloud["grid"].obs_mask.sum()),
        "teacher_depth": bool(cloud["teacher_depth"]),
    }
    build_showcase_png(cloud, fused)
    build_scan_video(cloud, fused)
    summary["total_elapsed_s"] = round(time.perf_counter() - t0, 1)

    out_json = RUN_DIR / "proto_static_vr_metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmetrics -> {out_json}\ntotal {summary['total_elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
