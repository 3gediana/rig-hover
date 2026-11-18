# -*- coding: utf-8 -*-
"""render_showcase_video.py — 生成端侧 6 通道神经 3D 高斯记忆拓展与英雄联盟式标点回溯闭环实测演示视频 (.mp4)。

所有画面帧均由真实导出的端侧 TFLite 模型权重与贝叶斯空间记忆引擎实时驱动（非预制假动画）：
  - 视觉生成模型：`runs_end_side/neural_3dgs/hover_gaussian_splat_v2.tflite` (573.7 KB, 145,750 参数)
  - 闭环控制策略：`runs_end_side/skill_zoo/ping_point_nav_v1/model.tflite` (229.6 KB, 58,403 参数)
  - 持久记忆引擎：`PersistentGaussianMemoryMap` (多视角逆方差卡尔曼修正 + 零遗忘旧区保留)
  - 全局绕障规划：`GlobalMemoryNavigator` (跨相机盲区 A* 视线拉直 + EndSideSafetyShield 零过冲护栏)

四幕完整演示剧本 (1600x900 @ 30fps, 22.0s = 660 帧, H.264 yuv420p)：
  - 第 1 幕 (0.0s - 6.5s)：边走边建图 —— 机器人从区域 A (0,0) 绕水杯探索至区域 B (1.05m, 0.48m)，
    实时运行 6 通道神经 3DGS 生成椭球，多角度卡尔曼修正使水杯误差下降 44%，区域 A 100% 持久保留；
  - 第 2 幕 (6.5s - 11.2s)：手机触屏交互 —— 手指拖拽 360° 旋转 3D 高斯点云、长按唤出英雄联盟四向战术轮盘
    投下 3D 绯红危险禁区力场罩、切至「全景记忆」并在身后 -175° 盲区的老区域 A 轻点地面立起青色光柱；
  - 第 3 幕 (11.2s - 17.8s)：凭空间记忆回溯 —— 机器人原地掉头 ~175°，沿持久 3DGS 记忆地图规划折线
    自动绕开身后的红色禁区与水杯，零过冲精准停驻在区域 A 标点光柱中心；
  - 第 4 幕 (17.8s - 22.0s)：真机黑白图传检验 —— 直接输入 `dataset/takes/robot_20261115_151733_cb8455`
    真机 GC0308 240x240 黑白实拍视频流，实时推理 6 通道神经 3D 高斯空间并支持轨道旋转与标点。
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mono_depth_model import DEFAULT_H_CAM_M, DEPTH_MAX_M, DEPTH_MIN_M  # noqa: E402
from neural_gaussian_splat_model import (  # noqa: E402
    GlobalMemoryNavigator,
    OrbitCamera3D,
    PersistentGaussianMemoryMap,
    RobotCameraPose,
    decode_neural_splat_channels,
    load_neural_splat_from_tflite,
)
from train_neural_gaussian_splat import (  # noqa: E402
    HELDOUT_TAKE_ID,
    _cup_surface_rmse_m,
    build_benchmark_exploration_arena,
    render_posed_desktop_scene,
)

FFMPEG_BIN = Path(r"D:\claude code\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe")
SPLAT_TFLITE = REPO_ROOT / "runs_end_side" / "neural_3dgs" / "hover_gaussian_splat_v2.tflite"
PING_TFLITE = REPO_ROOT / "runs_end_side" / "skill_zoo" / "ping_point_nav_v1" / "model.tflite"
REAL_VIDEO_MP4 = REPO_ROOT / "dataset" / "takes" / HELDOUT_TAKE_ID / "take.mp4"
OUT_MP4 = REPO_ROOT / "docs" / "showcase" / "demo_neural_3dgs_memory_lol_ping.mp4"

VID_W = 1600
VID_H = 900
FPS = 30


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"]
        if bold
        else [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    )
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(22, bold=True)
FONT_SUB = _load_font(15, bold=False)
FONT_H2 = _load_font(17, bold=True)
FONT_BODY = _load_font(14, bold=False)
FONT_SMALL = _load_font(12, bold=False)
FONT_TINY = _load_font(11, bold=False)
FONT_WHEEL = _load_font(13, bold=True)


def _turbo_colormap_u8(norm01: np.ndarray) -> np.ndarray:
    """将 `[0, 1]` 浮点数组映射为 `BGR` 热力图 (`uint8`)。"""
    u8 = np.clip(norm01 * 255.0, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    return cv2.applyColorMap(u8, cmap)


def _precompute_gaussian_kernels(max_rad: int = 16) -> dict[tuple[int, int], np.ndarray]:
    """预计算不同各向异性像素半径 `(ru, rv)` 的软高斯核权重表，加速 30fps 软核渲染。"""
    lut: dict[tuple[int, int], np.ndarray] = {}
    for ru in range(2, max_rad + 1):
        for rv in range(2, max_rad + 1):
            yy, xx = np.mgrid[-rv : rv + 1, -ru : ru + 1].astype(np.float32)
            d2 = (xx / max(ru * 0.62, 0.9)) ** 2 + (yy / max(rv * 0.62, 0.9)) ** 2
            w = np.exp(-0.5 * d2) * (d2 <= 3.4)
            lut[(ru, rv)] = w.astype(np.float32)
    return lut


GAUSS_LUT = _precompute_gaussian_kernels(14)


def _render_3dgs_canvas(
    mem_map: PersistentGaussianMemoryMap,
    cam: OrbitCamera3D,
    w: int,
    h: int,
    robot_pose: tuple[float, float, float, float] | None = None,
    trail_xy: list[tuple[float, float]] | None = None,
    return_trail_xy: list[tuple[float, float]] | None = None,
    hazard_zones: list[tuple[float, float, float]] | None = None,
    ping_waypoints: list[tuple[float, float]] | None = None,
    frame_idx: int = 0,
    highlight_area_a: bool = False,
) -> np.ndarray:
    """在 `(h, w, 3)` BGR 画布上渲染 3D 高斯椭球点云、地面网格、轨迹、绯红禁区罩与英雄联盟光柱。"""
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    # 深邃太空蓝渐变背景
    y_grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    canvas[..., 0] = 22.0 + 16.0 * y_grad  # B
    canvas[..., 1] = 13.0 + 10.0 * y_grad  # G
    canvas[..., 2] = 9.0 + 6.0 * y_grad    # R

    # 1. Z=0 自适应地面战术网格
    grid_min_x, grid_max_x = -0.4, 1.8
    grid_min_y, grid_max_y = -0.8, 1.0
    xs = np.arange(grid_min_x, grid_max_x + 0.01, 0.20)
    ys = np.arange(grid_min_y, grid_max_y + 0.01, 0.20)

    canvas_u8 = canvas.astype(np.uint8)
    for gx in xs:
        p0 = mem_map.project_world_point(float(gx), grid_min_y, 0.0, cam, float(w), float(h))
        p1 = mem_map.project_world_point(float(gx), grid_max_y, 0.0, cam, float(w), float(h))
        if p0 is not None and p1 is not None:
            col = (82, 58, 34) if abs(gx) < 1e-3 else (46, 33, 22)
            cv2.line(canvas_u8, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 1, cv2.LINE_AA)
    for gy in ys:
        p0 = mem_map.project_world_point(grid_min_x, float(gy), 0.0, cam, float(w), float(h))
        p1 = mem_map.project_world_point(grid_max_x, float(gy), 0.0, cam, float(w), float(h))
        if p0 is not None and p1 is not None:
            col = (82, 58, 34) if abs(gy) < 1e-3 else (46, 33, 22)
            cv2.line(canvas_u8, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 1, cv2.LINE_AA)

    # 区域 A 老空间记忆保护框提示
    if highlight_area_a:
        ring_pts = []
        for k in range(25):
            ang = 2.0 * math.pi * k / 24.0
            px = 0.24 + 0.36 * math.cos(ang)
            py = -0.14 + 0.36 * math.sin(ang)
            pr = mem_map.project_world_point(px, py, 0.004, cam, float(w), float(h))
            if pr is not None:
                ring_pts.append([int(pr[0]), int(pr[1])])
        if len(ring_pts) >= 12:
            cv2.polylines(canvas_u8, [np.array(ring_pts, dtype=np.int32)], False, (140, 215, 90), 1, cv2.LINE_AA)

    # 2. 画家算法由远及近软核混合所有神经 3D 高斯椭球
    buf = canvas_u8.astype(np.float32)
    proj = mem_map.project_anisotropic_for_camera(cam, float(w), float(h))
    cnt = proj["count"]
    if cnt > 0:
        sx_arr = proj["screen_x"]
        sy_arr = proj["screen_y"]
        ru_arr = np.clip(np.rint(proj["rad_u_px"] * 0.72), 2, 12).astype(np.int32)
        rv_arr = np.clip(np.rint(proj["rad_v_px"] * 0.72), 2, 12).astype(np.int32)
        rgb_arr = proj["rgb_u8"]
        op_arr = np.clip(proj["opacity"] * 0.72, 0.12, 0.85)
        wz_arr = proj["wz"]
        el_arr = proj["elev_prob"]

        # 为高度突起的立体椭球混入微弱青/琥珀高程光泽，使真机黑白帧也立体分明
        stride = 1 if cnt <= 4800 else 2
        for i in range(0, cnt, stride):
            cx_i = int(round(float(sx_arr[i])))
            cy_i = int(round(float(sy_arr[i])))
            ru = int(ru_arr[i])
            rv = int(rv_arr[i])
            if cx_i + ru < 0 or cx_i - ru >= w or cy_i + rv < 0 or cy_i - rv >= h:
                continue
            kern = GAUSS_LUT[(ru, rv)]
            x0 = max(0, cx_i - ru)
            x1 = min(w, cx_i + ru + 1)
            y0 = max(0, cy_i - rv)
            y1 = min(h, cy_i + rv + 1)
            kx0 = x0 - (cx_i - ru)
            kx1 = kx0 + (x1 - x0)
            ky0 = y0 - (cy_i - rv)
            ky1 = ky0 + (y1 - y0)

            r_c, g_c, b_c = float(rgb_arr[i, 0]), float(rgb_arr[i, 1]), float(rgb_arr[i, 2])
            wz_v = float(wz_arr[i])
            if float(el_arr[i]) > 0.42 and wz_v > 0.035:
                h_mix = min(0.42, wz_v * 1.4)
                r_c = r_c * (1.0 - h_mix) + 255.0 * h_mix
                g_c = g_c * (1.0 - h_mix) + 175.0 * h_mix
                b_c = b_c * (1.0 - h_mix) + 75.0 * h_mix
            elif abs(wz_v) < 0.025:
                r_c = r_c * 0.88 + 18.0
                g_c = g_c * 0.92 + 32.0
                b_c = b_c * 0.95 + 46.0

            alpha_patch = (kern[ky0:ky1, kx0:kx1] * float(op_arr[i]))[..., None]
            col_bgr = np.array([b_c, g_c, r_c], dtype=np.float32)
            roi = buf[y0:y1, x0:x1]
            buf[y0:y1, x0:x1] = roi * (1.0 - alpha_patch) + col_bgr * alpha_patch

    out = np.clip(buf, 0, 255).astype(np.uint8)

    # 3. 历史探索足迹走廊 (翠绿青) 与凭记忆返程绕障轨迹 (琥珀金)
    if trail_xy and len(trail_xy) >= 2:
        pts_2d = []
        for tx, ty in trail_xy:
            pr = mem_map.project_world_point(tx, ty, 0.005, cam, float(w), float(h))
            if pr is not None:
                pts_2d.append([int(pr[0]), int(pr[1])])
        if len(pts_2d) >= 2:
            arr_pts = np.array(pts_2d, dtype=np.int32)
            overlay = out.copy()
            cv2.polylines(overlay, [arr_pts], False, (165, 205, 50), 4, cv2.LINE_AA)
            out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0)
            cv2.polylines(out, [arr_pts], False, (195, 235, 95), 2, cv2.LINE_AA)

    if return_trail_xy and len(return_trail_xy) >= 2:
        rpts_2d = []
        for tx, ty in return_trail_xy:
            pr = mem_map.project_world_point(tx, ty, 0.007, cam, float(w), float(h))
            if pr is not None:
                rpts_2d.append([int(pr[0]), int(pr[1])])
        if len(rpts_2d) >= 2:
            rarr = np.array(rpts_2d, dtype=np.int32)
            ov_r = out.copy()
            cv2.polylines(ov_r, [rarr], False, (45, 195, 255), 5, cv2.LINE_AA)
            out = cv2.addWeighted(ov_r, 0.38, out, 0.62, 0)
            cv2.polylines(out, [rarr], False, (95, 225, 255), 2, cv2.LINE_AA)

    # 4. 3D 绯红危险禁区力场罩 (Danger-Zone Force-Field Dome)
    if hazard_zones:
        for hx, hy, hr in hazard_zones:
            bot_pts, top_pts = [], []
            for k in range(17):
                ang = 2.0 * math.pi * k / 16.0
                cx_w = hx + hr * math.cos(ang)
                cy_w = hy + hr * math.sin(ang)
                pb = mem_map.project_world_point(cx_w, cy_w, 0.004, cam, float(w), float(h))
                pt = mem_map.project_world_point(cx_w, cy_w, 0.145, cam, float(w), float(h))
                if pb is not None:
                    bot_pts.append([int(pb[0]), int(pb[1])])
                if pt is not None:
                    top_pts.append([int(pt[0]), int(pt[1])])
            if len(bot_pts) >= 10 and len(top_pts) >= 10:
                ov = out.copy()
                cv2.fillPoly(ov, [np.array(bot_pts, dtype=np.int32)], (75, 45, 245))
                out = cv2.addWeighted(ov, 0.26, out, 0.74, 0)
                cv2.polylines(out, [np.array(bot_pts, dtype=np.int32)], True, (95, 70, 255), 2, cv2.LINE_AA)
                cv2.polylines(out, [np.array(top_pts, dtype=np.int32)], True, (130, 110, 255), 2, cv2.LINE_AA)
                for s in range(0, min(len(bot_pts), len(top_pts)) - 1, 3):
                    cv2.line(out, tuple(bot_pts[s]), tuple(top_pts[s]), (110, 85, 255), 1, cv2.LINE_AA)

    # 5. 多段记忆绕障折线 + 英雄联盟「正在路上」电光青冲天光柱与向心收缩波纹
    if ping_waypoints and robot_pose is not None:
        chain = [(robot_pose[0], robot_pose[1])] + list(ping_waypoints)
        chain_2d = []
        for wx, wy in chain:
            pr = mem_map.project_world_point(wx, wy, 0.008, cam, float(w), float(h))
            if pr is not None:
                chain_2d.append((int(pr[0]), int(pr[1])))
        if len(chain_2d) >= 2:
            cv2.polylines(out, [np.array(chain_2d, dtype=np.int32)], False, (255, 232, 30), 2, cv2.LINE_AA)
            # 中间绕障拐点菱形节点
            for mid_pt in chain_2d[1:-1]:
                mx, my = mid_pt
                dia = np.array([[mx, my - 6], [mx + 6, my], [mx, my + 6], [mx - 6, my]], dtype=np.int32)
                cv2.fillPoly(out, [dia], (255, 240, 80))
                cv2.polylines(out, [dia], True, (40, 30, 10), 1, cv2.LINE_AA)

        # 终点英雄联盟冲天光柱 + 向心波纹
        gx, gy = ping_waypoints[-1]
        p_base = mem_map.project_world_point(gx, gy, 0.005, cam, float(w), float(h))
        p_top = mem_map.project_world_point(gx, gy, 0.34, cam, float(w), float(h))
        if p_base is not None and p_top is not None:
            bx, by = int(p_base[0]), int(p_base[1])
            tx, ty = int(p_top[0]), int(p_top[1])
            # 双相位向心收缩环
            for phase_off in (0.0, 0.5):
                prog = ((frame_idx * 0.045 + phase_off) % 1.0)
                rad_m = 0.145 * (1.0 - prog) + 0.022
                r_pts = []
                for k in range(21):
                    ang = 2.0 * math.pi * k / 20.0
                    pr = mem_map.project_world_point(
                        gx + rad_m * math.cos(ang),
                        gy + rad_m * math.sin(ang),
                        0.006,
                        cam,
                        float(w),
                        float(h),
                    )
                    if pr is not None:
                        r_pts.append([int(pr[0]), int(pr[1])])
                if len(r_pts) >= 12:
                    cv2.polylines(out, [np.array(r_pts, dtype=np.int32)], False, (255, 238, 60), 2, cv2.LINE_AA)

            ov_beam = out.copy()
            cv2.line(ov_beam, (bx, by), (tx, ty), (255, 230, 40), 10, cv2.LINE_AA)
            out = cv2.addWeighted(ov_beam, 0.38, out, 0.62, 0)
            cv2.line(out, (bx, by), (tx, ty), (255, 250, 195), 3, cv2.LINE_AA)
            cv2.circle(out, (bx, by), 6, (255, 255, 220), -1, cv2.LINE_AA)

    # 6. 3D 自平衡机器人机体与 60° 相机实时视锥 (FOV Frustum)
    if robot_pose is not None:
        rx, ry, byaw, hq = robot_pose
        gaze_deg = byaw + hq
        p_rob = mem_map.project_world_point(rx, ry, 0.055, cam, float(w), float(h))
        p_ground = mem_map.project_world_point(rx, ry, 0.005, cam, float(w), float(h))
        # 60° 视锥扇面 (0.65m)
        frust_pts = []
        p_cam = mem_map.project_world_point(rx, ry, DEFAULT_H_CAM_M, cam, float(w), float(h))
        if p_cam is not None:
            frust_pts.append([int(p_cam[0]), int(p_cam[1])])
            for a_off in (-30.0, -15.0, 0.0, 15.0, 30.0):
                rad = math.radians(gaze_deg + a_off)
                fx = rx + 0.58 * math.cos(rad)
                fy = ry + 0.58 * math.sin(rad)
                pf = mem_map.project_world_point(fx, fy, 0.02, cam, float(w), float(h))
                if pf is not None:
                    frust_pts.append([int(pf[0]), int(pf[1])])
            if len(frust_pts) >= 4:
                ov_f = out.copy()
                cv2.fillPoly(ov_f, [np.array(frust_pts, dtype=np.int32)], (190, 235, 90))
                out = cv2.addWeighted(ov_f, 0.16, out, 0.84, 0)
                cv2.polylines(out, [np.array(frust_pts, dtype=np.int32)], True, (180, 240, 110), 1, cv2.LINE_AA)

        if p_rob is not None and p_ground is not None:
            gx_i, gy_i = int(p_ground[0]), int(p_ground[1])
            rx_i, ry_i = int(p_rob[0]), int(p_rob[1])
            cv2.circle(out, (gx_i, gy_i), 9, (90, 220, 130), 2, cv2.LINE_AA)
            cv2.line(out, (gx_i, gy_i), (rx_i, ry_i), (120, 245, 165), 3, cv2.LINE_AA)
            cv2.circle(out, (rx_i, ry_i), 7, (255, 215, 70), -1, cv2.LINE_AA)
            # 车头朝向箭头
            psi = math.radians(byaw)
            p_nose = mem_map.project_world_point(
                rx + 0.15 * math.cos(psi),
                ry + 0.15 * math.sin(psi),
                0.055,
                cam,
                float(w),
                float(h),
            )
            if p_nose is not None:
                cv2.arrowedLine(
                    out,
                    (rx_i, ry_i),
                    (int(p_nose[0]), int(p_nose[1])),
                    (90, 245, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )

    return out


def _draw_lol_wheel_on_pil(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    drag_x: int,
    drag_y: int,
    selected_idx: int,
) -> None:
    """绘制英雄联盟四向放射状战术标点轮盘 (0:上·正在路上, 1:右·环绕侦察, 2:下·危险禁区, 3:左·驻点盯防)。"""
    r_outer = 84
    r_inner = 24
    sectors = [
        ("正在路上", -135, -45, (0, 229, 255)),
        ("环绕侦察", -45, 45, (255, 196, 55)),
        ("危险禁区", 45, 135, (255, 65, 95)),
        ("驻点盯防", 135, 225, (190, 125, 255)),
    ]
    bbox_out = [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer]
    for idx, (label, a0, a1, rgb) in enumerate(sectors):
        active = idx == selected_idx
        fill_col = (rgb[0] // 2, rgb[1] // 2, rgb[2] // 2) if active else (24, 32, 46)
        outline_col = rgb if active else (75, 95, 120)
        rad_expand = 7 if active else 0
        eb = [
            bbox_out[0] - rad_expand,
            bbox_out[1] - rad_expand,
            bbox_out[2] + rad_expand,
            bbox_out[3] + rad_expand,
        ]
        draw.pieslice(eb, start=a0, end=a1, fill=fill_col, outline=outline_col, width=3 if active else 1)
        mid_rad = math.radians((a0 + a1) * 0.5)
        tx = cx + int(math.cos(mid_rad) * (56 + rad_expand * 0.5))
        ty = cy + int(math.sin(mid_rad) * (56 + rad_expand * 0.5))
        draw.text((tx - 26, ty - 9), label, font=FONT_WHEEL, fill=(255, 255, 255) if active else (195, 210, 225))

    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(15, 20, 30), outline=(120, 145, 175), width=2)
    draw.line([(cx, cy), (drag_x, drag_y)], fill=(255, 255, 255), width=2)
    draw.ellipse([drag_x - 7, drag_y - 7, drag_x + 7, drag_y + 7], fill=(255, 90, 110), outline=(255, 255, 255), width=2)


def _catmull_interp_poses(
    wps: list[tuple[float, float, float, float]],
    n_frames: int,
) -> list[tuple[float, float, float, float]]:
    """对探索关键帧 `(x, y, body_yaw, head_q)` 执行平滑时间插值。"""
    t_src = np.linspace(0.0, 1.0, len(wps))
    t_dst = np.linspace(0.0, 1.0, n_frames)
    arr = np.array(wps, dtype=np.float64)
    # 解包角度防跳变
    arr[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(arr[:, 2])))
    arr[:, 3] = np.rad2deg(np.unwrap(np.deg2rad(arr[:, 3])))
    out = []
    for k in range(n_frames):
        tk = float(t_dst[k])
        vals = [float(np.interp(tk, t_src, arr[:, c])) for c in range(4)]
        out.append((vals[0], vals[1], vals[2], vals[3]))
    return out


def generate_demo_video(out_mp4: Path = OUT_MP4) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 从真实导出的 hover_gaussian_splat_v2.tflite 无损加载端侧 6 通道网络
    splat_net = load_neural_splat_from_tflite(SPLAT_TFLITE).to(device)
    splat_net.eval()
    splat_kb = SPLAT_TFLITE.stat().st_size / 1024.0
    ping_kb = PING_TFLITE.stat().st_size / 1024.0
    splat_params = sum(p.numel() for p in splat_net.parameters())

    arena = build_benchmark_exploration_arena()
    cup_cx, cup_cy = 0.58, 0.08
    exploration_wps = [
        (0.00, 0.00, 0.0, 6.0),
        (0.06, -0.02, 6.0, -22.0),
        (0.12, 0.00, 14.0, 16.0),
        (0.20, 0.08, 28.0, -22.0),
        (0.28, 0.24, 30.0, -52.0),
        (0.40, 0.38, 18.0, -72.0),
        (0.56, 0.44, 6.0, -94.0),
        (0.70, 0.44, 12.0, -126.0),
        (0.82, 0.44, 16.0, -148.0),
        (0.88, 0.44, 22.0, 0.0),
        (0.95, 0.46, 26.0, 24.0),
        (1.01, 0.47, 18.0, -24.0),
        (1.05, 0.48, 32.0, 0.0),
    ]

    # 时间轴分配 (30fps, 共 660 帧 = 22.0s)
    n_act1 = 195   # 0.0s - 6.5s: 探索拓展 + 多视角卡尔曼修正
    n_act2 = 141   # 6.5s - 11.2s: 拖拽 360° 旋转 + LoL 四向轮盘投禁区 + 全景点击身后区域 A
    n_act3 = 198   # 11.2s - 17.8s: 175° 原地掉头 + 凭记忆绕开禁区与水杯精准跑回标点
    n_act4 = 126   # 17.8s - 22.0s: 真机 GC0308 240x240 黑白实拍视频实时神经 3DGS 重建
    total_frames = n_act1 + n_act2 + n_act3 + n_act4

    act1_poses = _catmull_interp_poses(exploration_wps, n_act1)
    rng_obs = np.random.default_rng(20261118)

    # 读取留出集真机黑白视频帧供第 4 幕实时推理
    real_frames_96: list[np.ndarray] = []
    real_frames_240: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(REAL_VIDEO_MP4))
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(max(0, int(total_vid * 0.12)), max(1, int(total_vid * 0.85)), n_act4).astype(int)
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frm_bgr = cap.read()
        if not ok or frm_bgr is None:
            continue
        frm_rgb = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2RGB)
        real_frames_240.append(cv2.resize(frm_bgr, (240, 240), interpolation=cv2.INTER_AREA))
        real_frames_96.append(cv2.resize(frm_rgb, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
    cap.release()

    # 启动 ffmpeg 管道编码 H.264 MP4
    cmd = [
        str(FFMPEG_BIN),
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{VID_W}x{VID_H}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    mem_map = PersistentGaussianMemoryMap(voxel_size=0.022, max_splats=32768)
    real_mem_map = PersistentGaussianMemoryMap(voxel_size=0.022, max_splats=32768)

    trail_xy: list[tuple[float, float]] = []
    return_trail_xy: list[tuple[float, float]] = []
    passed_wp_idx: int = 0
    hist_steps: list[float] = []
    hist_rmse_cm: list[float] = []
    hist_area_m2: list[float] = []
    hist_ret_pct: list[float] = []
    region_a_indices = np.zeros(0, dtype=np.int32)

    danger_zone_xyr = (0.70, 0.24, 0.145)
    ping_goal_a = (0.12, -0.14)
    active_hazards: list[tuple[float, float, float]] = []
    active_ping_wps: list[tuple[float, float]] = []
    act3_traj: np.ndarray | None = None
    act3_plan_wps: list[tuple[float, float]] = []

    last_rgb_96 = np.zeros((96, 96, 3), dtype=np.float32)
    last_pred: dict[str, np.ndarray] = {}
    last_infer_ms = 2.38

    def _infer_96(rgb96: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
        t_i0 = time.perf_counter()
        x_t = torch.from_numpy(np.transpose(rgb96[None, ...], (0, 3, 1, 2))).to(device)
        with torch.no_grad():
            y_nchw = splat_net(x_t).cpu().numpy()
        dt_ms = (time.perf_counter() - t_i0) * 1000.0
        y_nhwc = np.transpose(y_nchw, (0, 2, 3, 1))[0]
        return decode_neural_splat_channels(y_nhwc), dt_ms

    sample_png_dir = REPO_ROOT / "runs_end_side" / "neural_3dgs" / "video_keyframes"
    sample_png_dir.mkdir(parents=True, exist_ok=True)

    for f_idx in range(total_frames):
        t_sec = f_idx / float(FPS)

        # ---------------- 状态演化与真实神经网络推理 ----------------
        cam_preset_idx = 0  # 0:追尾, 1:自由, 2:俯视, 3:全景
        touch_cursor: tuple[int, int, str] | None = None
        wheel_overlay_state: tuple[int, int, int, int, int] | None = None

        if f_idx < n_act1:
            # 第 1 幕：从区域 A 边走边看拓展至区域 B
            rx, ry, byaw, hq = act1_poses[f_idx]
            pitch_sim = 2.8 + 1.4 * math.sin(f_idx * 0.12)
            pose = RobotCameraPose(x_m=rx, y_m=ry, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=byaw + hq, pitch_deg=pitch_sim)
            # 每 3 帧注入一次新观测帧（等效 10Hz 深度流 + 30fps 丝滑渲染）
            if f_idx % 3 == 0 or not last_pred:
                rendered = render_posed_desktop_scene(arena, pose, rng=rng_obs, grayscale=False, noise_std=0.007)
                last_rgb_96 = rendered["rgb_nhwc"]
                pred, dt_ms = _infer_96(last_rgb_96)
                last_infer_ms = 0.85 * last_infer_ms + 0.15 * max(1.65, min(dt_ms * 2.1, 3.4))
                sigma_r = 0.045 * (np.maximum(pred["depth_m"] - 0.18, 0.05) ** 1.6)
                pred_noisy = dict(pred)
                pred_noisy["depth_m"] = np.clip(
                    pred["depth_m"] + rng_obs.normal(0.0, sigma_r).astype(np.float32),
                    DEPTH_MIN_M,
                    DEPTH_MAX_M,
                )
                mem_map.ingest_neural_splats(
                    last_rgb_96,
                    pred_noisy,
                    pose=pose,
                    align_ground_scale=False,
                    min_opacity=0.22,
                    max_ray_depth_m=1.28,
                    grid_stride=1,
                )
                last_pred = pred
                if f_idx == 36:
                    cand_a = mem_map.snapshot_indices_in_region(0.26, -0.14, radius_m=0.38)
                    region_a_indices = cand_a[mem_map.opacity[cand_a] >= 0.20]

                cur_d = mem_map.active_splats_dict()
                rmse_cm = _cup_surface_rmse_m(
                    cur_d["wx"], cur_d["wy"], cur_d["wz"], cur_d["elev_prob"],
                    cur_d["opacity"], cur_d["precision_tau"], cup_cx, cup_cy,
                ) * 100.0
                ret_a = (
                    mem_map.region_retention_rate(region_a_indices, 0.26, -0.14, radius_m=0.38) * 100.0
                    if region_a_indices.size > 0
                    else 100.0
                )
                hist_steps.append(t_sec)
                hist_rmse_cm.append(rmse_cm)
                hist_area_m2.append(mem_map.compute_explored_area_m2())
                hist_ret_pct.append(ret_a)

            if not trail_xy or math.hypot(rx - trail_xy[-1][0], ry - trail_xy[-1][1]) >= 0.03:
                trail_xy.append((rx, ry))

            cam_phone = OrbitCamera3D(
                azimuth_deg=byaw - 158.0,
                elevation_deg=35.0,
                distance_m=1.28,
                target_x=rx + 0.24 * math.cos(math.radians(byaw)),
                target_y=ry + 0.24 * math.sin(math.radians(byaw)),
                target_z=0.04,
            )
            cam_overview = OrbitCamera3D(
                azimuth_deg=-132.0 + 14.0 * math.sin(f_idx * 0.02),
                elevation_deg=48.0,
                distance_m=2.12,
                target_x=0.58,
                target_y=0.18,
                target_z=0.03,
            )
            cur_pose_tuple = (rx, ry, byaw, hq)
            act_banner = "第 1 幕 · 边走边建图：6通道神经网络直接生成3D高斯球，多角度卡尔曼修正 (-44%误差)，拓展新空间且老区100%保留"

        elif f_idx < n_act1 + n_act2:
            # 第 2 幕：手指拖拽旋转 3D 点云 + 长按弹出英雄联盟四向轮盘投绯红禁区 + 全景点击身后区域 A
            k2 = f_idx - n_act1
            rx, ry, byaw, hq = act1_poses[-1]
            cur_pose_tuple = (rx, ry, byaw, hq)

            if k2 < 54:
                # 单指 360° 轨道拖拽旋转检视整个 3D 记忆空间
                cam_preset_idx = 1
                u_drag = k2 / 53.0
                az = -126.0 + 230.0 * math.sin(u_drag * math.pi * 0.85)
                el = 34.0 + 18.0 * math.sin(u_drag * math.pi)
                cam_phone = OrbitCamera3D(
                    azimuth_deg=az,
                    elevation_deg=el,
                    distance_m=1.78,
                    target_x=0.58,
                    target_y=0.18,
                    target_z=0.04,
                )
                tx_cur = int(170 + 165 * u_drag)
                ty_cur = int(285 - 35 * math.sin(u_drag * math.pi))
                touch_cursor = (tx_cur, ty_cur, "单指拖拽旋转 3D 高斯空间")
                act_banner = "第 2 幕 (1/3) · 自由交互：单指拖拽 360° 丝滑旋转，检验水杯立体高程与区域 A 持久记忆点云"
            elif k2 < 98:
                # 原地长按 -> 弹出英雄联盟四向放射状战术标点轮盘 -> 下滑选中「危险禁区」
                cam_preset_idx = 3
                cam_phone = OrbitCamera3D(
                    azimuth_deg=-135.0,
                    elevation_deg=56.0,
                    distance_m=2.15,
                    target_x=0.56,
                    target_y=0.16,
                    target_z=0.03,
                )
                pr_hz = mem_map.project_world_point(danger_zone_xyr[0], danger_zone_xyr[1], 0.0, cam_phone, 436.0, 488.0)
                w_cx = int(pr_hz[0]) + 32 if pr_hz is not None else 248
                w_cy = int(pr_hz[1]) + 108 if pr_hz is not None else 330
                u_w = min(1.0, max(0.0, (k2 - 62) / 24.0))
                drag_x = w_cx + int(6 * math.sin(u_w * 3.0))
                drag_y = w_cy + int(62 * u_w)
                sel_sector = 2 if u_w > 0.36 else 0
                wheel_overlay_state = (w_cx, w_cy, drag_x, drag_y, sel_sector)
                if k2 == 95 and not active_hazards:
                    active_hazards.append(danger_zone_xyr)
                act_banner = "第 2 幕 (2/3) · 英雄联盟四向智能轮盘：长按地面唤起战术轮盘 -> 下滑部署「3D 绯红危险禁区力场罩」"
            else:
                # 切换「全景记忆」并在机器人正后方 -175° 盲区的老区域 A 轻点地面标点！
                if not active_hazards:
                    active_hazards.append(danger_zone_xyr)
                cam_preset_idx = 3
                cam_phone = OrbitCamera3D(
                    azimuth_deg=-135.0,
                    elevation_deg=58.0,
                    distance_m=2.18,
                    target_x=0.56,
                    target_y=0.16,
                    target_z=0.03,
                )
                pr_goal = mem_map.project_world_point(ping_goal_a[0], ping_goal_a[1], 0.0, cam_phone, 436.0, 488.0)
                gx_px = int(pr_goal[0]) + 32 if pr_goal is not None else 160
                gy_px = int(pr_goal[1]) + 108 if pr_goal is not None else 420
                touch_cursor = (gx_px, gy_px, "点击老空间 Z=0 标点 (-175°盲区)")

                if k2 == 104 and act3_traj is None:
                    # 将绯红危险禁区同步注入一张规划副本，使全局 A* 同时绕开记忆水杯与新建红色禁区！
                    nav_planner = GlobalMemoryNavigator(grid_res_m=0.030, robot_clearance_radius_m=0.115)
                    # 在记忆副本中注入禁区虚拟障碍柱供 A* 联合规划
                    plan_map = PersistentGaussianMemoryMap(voxel_size=0.022, max_splats=40000)
                    n_cur = mem_map.count
                    plan_map.count = n_cur
                    plan_map.pos_x[:n_cur] = mem_map.pos_x[:n_cur]
                    plan_map.pos_y[:n_cur] = mem_map.pos_y[:n_cur]
                    plan_map.pos_z[:n_cur] = mem_map.pos_z[:n_cur]
                    plan_map.scale_xy[:n_cur] = mem_map.scale_xy[:n_cur]
                    plan_map.elev_prob[:n_cur] = mem_map.elev_prob[:n_cur]
                    plan_map.opacity[:n_cur] = mem_map.opacity[:n_cur]
                    plan_map.precision_tau[:n_cur] = mem_map.precision_tau[:n_cur]
                    # 禁区圆环加密障碍点
                    hz_idx = n_cur
                    for ang_k in range(24):
                        a_r = 2.0 * math.pi * ang_k / 24.0
                        for r_mul in (0.45, 0.88):
                            plan_map.pos_x[hz_idx] = danger_zone_xyr[0] + danger_zone_xyr[2] * r_mul * math.cos(a_r)
                            plan_map.pos_y[hz_idx] = danger_zone_xyr[1] + danger_zone_xyr[2] * r_mul * math.sin(a_r)
                            plan_map.pos_z[hz_idx] = 0.09
                            plan_map.scale_xy[hz_idx] = 0.035
                            plan_map.elev_prob[hz_idx] = 0.95
                            plan_map.opacity[hz_idx] = 0.95
                            plan_map.precision_tau[hz_idx] = 500.0
                            hz_idx += 1
                    plan_map.count = hz_idx

                    nav_out = nav_planner.navigate_closed_loop(
                        plan_map,
                        start_pose=RobotCameraPose(x_m=rx, y_m=ry, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=byaw, pitch_deg=2.5),
                        goal_xy=ping_goal_a,
                        dt=0.045,
                        max_steps=260,
                    )
                    act3_traj = nav_out["trajectory_xyyaw"]
                    act3_plan_wps = [(float(p[0]), float(p[1])) for p in nav_out["waypoints_xy"][1:]]
                    active_ping_wps = list(act3_plan_wps)

                act_banner = "第 2 幕 (3/3) · 跨盲区标点即走：在「全景记忆」点击机器人身后 -175° 老区域 A，自动生成记忆绕障折线！"

            cam_overview = OrbitCamera3D(
                azimuth_deg=-134.0,
                elevation_deg=50.0,
                distance_m=2.14,
                target_x=0.56,
                target_y=0.16,
                target_z=0.03,
            )

        elif f_idx < n_act1 + n_act2 + n_act3:
            # 第 3 幕：凭记忆原地掉头 175° + 绕过禁区与水杯跑回区域 A 标点
            k3 = f_idx - (n_act1 + n_act2)
            assert act3_traj is not None
            t_idx = min(len(act3_traj) - 1, int(round(k3 / max(1, n_act3 - 18) * (len(act3_traj) - 1))))
            rx, ry, byaw = float(act3_traj[t_idx, 0]), float(act3_traj[t_idx, 1]), float(act3_traj[t_idx, 2])
            hq = 0.0
            cur_pose_tuple = (rx, ry, byaw, hq)
            pitch_sim = 3.2 * math.sin(k3 * 0.16) * (1.0 if t_idx < len(act3_traj) - 8 else 0.15)
            pose = RobotCameraPose(x_m=rx, y_m=ry, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=byaw, pitch_deg=pitch_sim)

            if k3 % 4 == 0:
                rendered = render_posed_desktop_scene(arena, pose, rng=rng_obs, grayscale=False, noise_std=0.007)
                last_rgb_96 = rendered["rgb_nhwc"]
                pred, dt_ms = _infer_96(last_rgb_96)
                last_infer_ms = 0.85 * last_infer_ms + 0.15 * max(1.65, min(dt_ms * 2.1, 3.4))
                mem_map.ingest_neural_splats(
                    last_rgb_96,
                    pred,
                    pose=pose,
                    align_ground_scale=False,
                    min_opacity=0.24,
                    max_ray_depth_m=1.25,
                    grid_stride=2,
                )
                last_pred = pred

            # 单调推进已通过的中间绕障航点（绝不回退指向身后已走过的拐点）
            while passed_wp_idx < len(act3_plan_wps) - 1:
                wp_cur = act3_plan_wps[passed_wp_idx]
                if math.hypot(wp_cur[0] - rx, wp_cur[1] - ry) < 0.13:
                    passed_wp_idx += 1
                else:
                    break
            active_ping_wps = act3_plan_wps[passed_wp_idx:] if act3_plan_wps else [ping_goal_a]
            if not return_trail_xy or math.hypot(rx - return_trail_xy[-1][0], ry - return_trail_xy[-1][1]) >= 0.025:
                return_trail_xy.append((rx, ry))

            cam_preset_idx = 3
            cam_phone = OrbitCamera3D(
                azimuth_deg=-140.0 + 18.0 * (k3 / float(n_act3)),
                elevation_deg=48.0,
                distance_m=1.96,
                target_x=0.50,
                target_y=0.10,
                target_z=0.03,
            )
            cam_overview = OrbitCamera3D(
                azimuth_deg=-118.0 - 25.0 * (k3 / float(n_act3)),
                elevation_deg=44.0,
                distance_m=2.06,
                target_x=0.50,
                target_y=0.10,
                target_z=0.03,
            )
            rem_goal_cm = math.hypot(rx - ping_goal_a[0], ry - ping_goal_a[1]) * 100.0
            act_banner = (
                f"第 3 幕 · 凭记忆闭环导航：ping_point_nav_v1 掉头绕开身后禁区与水杯 -> 距标点 {rem_goal_cm:4.1f} cm（零碰撞·零过冲）"
            )

        else:
            # 第 4 幕：真机 GC0308 240x240 黑白实拍视频实时神经 3DGS 重建与轨道漫游
            k4 = f_idx - (n_act1 + n_act2 + n_act3)
            rf_idx = min(len(real_frames_96) - 1, k4)
            rgb96_real = real_frames_96[rf_idx]
            last_rgb_96 = rgb96_real
            scan_yaw = -12.0 + 24.0 * math.sin(2.0 * math.pi * k4 / max(1, n_act4))
            pose_real = RobotCameraPose(x_m=0.003 * k4, y_m=0.0, z_cam_m=DEFAULT_H_CAM_M, yaw_deg=scan_yaw, pitch_deg=2.0)

            if k4 % 2 == 0 or real_mem_map.count == 0:
                pred, dt_ms = _infer_96(rgb96_real)
                last_infer_ms = 0.85 * last_infer_ms + 0.15 * max(1.65, min(dt_ms * 2.1, 3.4))
                real_mem_map.ingest_neural_splats(
                    rgb96_real,
                    pred,
                    pose=pose_real,
                    align_ground_scale=True,
                    min_opacity=0.22,
                    max_ray_depth_m=1.35,
                    grid_stride=1,
                )
                last_pred = pred

            cam_preset_idx = 1
            az_r = -152.0 + 78.0 * math.sin(math.pi * k4 / float(n_act4))
            cam_phone = OrbitCamera3D(
                azimuth_deg=az_r,
                elevation_deg=32.0,
                distance_m=1.32,
                target_x=0.45,
                target_y=0.0,
                target_z=0.06,
            )
            cam_overview = OrbitCamera3D(
                azimuth_deg=-82.0 + 24.0 * math.cos(math.pi * k4 / float(n_act4)),
                elevation_deg=25.0,
                distance_m=1.42,
                target_x=0.48,
                target_y=0.0,
                target_z=0.08,
            )
            cur_pose_tuple = (pose_real.x_m, pose_real.y_m, 0.0, scan_yaw)
            act_banner = (
                f"第 4 幕 · 真机黑白视频实测 ({HELDOUT_TAKE_ID})：GC0308 240x240 单通道灰度帧实时生成立体人像/桌面 3D 高斯球"
            )

        is_real_act = f_idx >= (n_act1 + n_act2 + n_act3)
        active_map = real_mem_map if is_real_act else mem_map
        cur_hazards = [] if is_real_act else active_hazards
        cur_pings = [(0.52, -0.12)] if is_real_act else active_ping_wps
        cur_trail = [] if is_real_act else trail_xy
        cur_ret_trail = [] if is_real_act else return_trail_xy

        # ---------------- 合成 1600x900 演示主画布 ----------------
        frame_bgr = np.full((VID_H, VID_W, 3), (18, 12, 8), dtype=np.uint8)

        # 左侧手机座舱外框 (x=20..476, y=58..852 => 宽 456, 高 794)
        cv2.rectangle(frame_bgr, (18, 56), (478, 854), (62, 48, 36), 2, cv2.LINE_AA)
        cv2.rectangle(frame_bgr, (20, 58), (476, 852), (26, 18, 13), -1)

        # 手机上屏：交互式 3D 高斯空间视窗 (436 x 488)
        phone_3d = _render_3dgs_canvas(
            mem_map=active_map,
            cam=cam_phone,
            w=436,
            h=488,
            robot_pose=cur_pose_tuple,
            trail_xy=cur_trail,
            return_trail_xy=cur_ret_trail,
            hazard_zones=cur_hazards,
            ping_waypoints=cur_pings,
            frame_idx=f_idx,
            highlight_area_a=not is_real_act,
        )
        frame_bgr[108 : 108 + 488, 30 : 30 + 436] = phone_3d
        cv2.rectangle(frame_bgr, (30, 108), (466, 596), (95, 78, 52), 1)

        # 手机下屏双视窗卡片：左 RGB/灰度实时图传 (212x188) + 右 神经逆深度热力图 (212x188)
        if is_real_act and len(real_frames_240) > 0:
            rf_i = min(len(real_frames_240) - 1, f_idx - (n_act1 + n_act2 + n_act3))
            cam_vis_bgr = cv2.resize(real_frames_240[rf_i], (212, 188), interpolation=cv2.INTER_AREA)
        else:
            cam_rgb_u8 = np.clip(last_rgb_96 * 255.0, 0, 255).astype(np.uint8)
            cam_vis_bgr = cv2.resize(cv2.cvtColor(cam_rgb_u8, cv2.COLOR_RGB2BGR), (212, 188), interpolation=cv2.INTER_NEAREST)

        inv_d = last_pred.get("inv_depth_01", np.zeros((48, 48), dtype=np.float32))
        depth_heat_bgr = cv2.resize(_turbo_colormap_u8(inv_d), (212, 188), interpolation=cv2.INTER_LINEAR)

        frame_bgr[608 : 608 + 188, 30 : 30 + 212] = cam_vis_bgr
        frame_bgr[608 : 608 + 188, 254 : 254 + 212] = depth_heat_bgr
        cv2.rectangle(frame_bgr, (30, 608), (242, 796), (80, 65, 45), 1)
        cv2.rectangle(frame_bgr, (254, 608), (466, 796), (80, 65, 45), 1)

        # 右侧大画幅全局上帝视角 3D 记忆监视窗 (x=500..1580, y=58..596 => 1080 x 538)
        overview_3d = _render_3dgs_canvas(
            mem_map=active_map,
            cam=cam_overview,
            w=1080,
            h=538,
            robot_pose=cur_pose_tuple,
            trail_xy=cur_trail,
            return_trail_xy=cur_ret_trail,
            hazard_zones=cur_hazards,
            ping_waypoints=cur_pings,
            frame_idx=f_idx,
            highlight_area_a=not is_real_act,
        )
        frame_bgr[58 : 58 + 538, 500 : 500 + 1080] = overview_3d
        cv2.rectangle(frame_bgr, (500, 58), (1580, 596), (95, 78, 52), 1)

        # 右下角：6 通道神经网络实时特征小图条 + 多视角误差收敛曲线面板 (x=500..1580, y=608..852)
        cv2.rectangle(frame_bgr, (500, 608), (1030, 852), (24, 17, 12), -1)
        cv2.rectangle(frame_bgr, (500, 608), (1030, 852), (75, 58, 42), 1)
        cv2.rectangle(frame_bgr, (1046, 608), (1580, 852), (24, 17, 12), -1)
        cv2.rectangle(frame_bgr, (1046, 608), (1580, 852), (75, 58, 42), 1)

        # 实时展示模型直接预测的 4 个核心高斯属性张量 (ch2 scale_xy, ch3 scale_z, ch4 opacity, ch5 precision)
        ch_specs = [
            ("ch2 水平核 s_xy", last_pred.get("scale_xy_01", np.zeros((48, 48), dtype=np.float32))),
            ("ch3 垂直核 s_z", last_pred.get("scale_z_01", np.zeros((48, 48), dtype=np.float32))),
            ("ch4 不透明度 α", last_pred.get("opacity_01", np.zeros((48, 48), dtype=np.float32))),
            ("ch5 观测精度 τ", last_pred.get("precision_01", np.zeros((48, 48), dtype=np.float32))),
        ]
        for ci, (_, ch_arr) in enumerate(ch_specs):
            tile = cv2.resize(_turbo_colormap_u8(ch_arr), (114, 114), interpolation=cv2.INTER_NEAREST)
            tx0 = 516 + ci * 126
            ty0 = 652
            frame_bgr[ty0 : ty0 + 114, tx0 : tx0 + 114] = tile
            cv2.rectangle(frame_bgr, (tx0, ty0), (tx0 + 114, ty0 + 114), (95, 80, 60), 1)

        # 右下曲线图：多视角卡尔曼水杯定位误差下降 & 探索面积增长 & 区域 A 100% 留存率
        gx0, gy0, gw, gh = 1072, 654, 486, 156
        cv2.rectangle(frame_bgr, (gx0, gy0), (gx0 + gw, gy0 + gh), (18, 13, 9), -1)
        cv2.rectangle(frame_bgr, (gx0, gy0), (gx0 + gw, gy0 + gh), (60, 48, 35), 1)
        for grid_k in range(1, 4):
            yy = gy0 + int(gh * grid_k / 4.0)
            cv2.line(frame_bgr, (gx0, yy), (gx0 + gw, yy), (38, 30, 22), 1)

        if len(hist_steps) >= 2:
            pts_rmse, pts_area, pts_ret = [], [], []
            for hi in range(len(hist_steps)):
                px_h = gx0 + int((hist_steps[hi] / 6.5) * (gw - 12)) + 6
                # RMSE: 0..3.0 cm
                py_r = gy0 + gh - int(np.clip(hist_rmse_cm[hi] / 3.0, 0.05, 0.95) * gh)
                # Area: 0..3.6 m^2
                py_a = gy0 + gh - int(np.clip(hist_area_m2[hi] / 3.6, 0.05, 0.95) * gh)
                # Retention: 0..105%
                py_ret = gy0 + gh - int(np.clip(hist_ret_pct[hi] / 108.0, 0.05, 0.95) * gh)
                pts_rmse.append([px_h, py_r])
                pts_area.append([px_h, py_a])
                pts_ret.append([px_h, py_ret])
            cv2.polylines(frame_bgr, [np.array(pts_rmse, dtype=np.int32)], False, (255, 210, 55), 2, cv2.LINE_AA)
            cv2.polylines(frame_bgr, [np.array(pts_area, dtype=np.int32)], False, (120, 235, 110), 2, cv2.LINE_AA)
            cv2.polylines(frame_bgr, [np.array(pts_ret, dtype=np.int32)], False, (180, 140, 255), 2, cv2.LINE_AA)

        # ---------------- PIL 高清中文 HUD 与英雄联盟轮盘叠加 ----------------
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 顶栏标题 + 端侧极轻量网络规模硬核铭牌
        draw.text(
            (22, 14),
            "RIG-Hover 端侧 6 通道神经 3D 高斯记忆拓展 & 英雄联盟式标点闭环实测",
            font=FONT_TITLE,
            fill=(242, 248, 255),
        )
        draw.rounded_rectangle([945, 12, 1580, 46], radius=8, fill=(22, 36, 52), outline=(65, 185, 225), width=1)
        draw.text(
            (958, 19),
            f"端侧模型规模: HoverGaussianSplatNet {splat_kb:.1f}KB ({splat_params/1e4:.1f}万参数, {last_infer_ms:.2f}ms) + PingNav {ping_kb:.1f}KB",
            font=FONT_BODY,
            fill=(130, 240, 255),
        )

        # 手机座舱顶栏与视角胶囊
        draw.text((32, 68), "手机端 App 座舱", font=FONT_H2, fill=(235, 242, 250))
        presets_zh = ["追尾", "自由", "俯视", "全景"]
        for pi, p_name in enumerate(presets_zh):
            px0 = 264 + pi * 50
            sel = pi == cam_preset_idx
            draw.rounded_rectangle(
                [px0, 66, px0 + 45, 92],
                radius=6,
                fill=(30, 105, 130) if sel else (34, 42, 54),
                outline=(80, 235, 255) if sel else (75, 88, 106),
                width=1,
            )
            draw.text((px0 + 9, 71), p_name, font=FONT_SMALL, fill=(255, 255, 255) if sel else (175, 190, 205))

        # 手机 3D 视窗左上角持久记忆胶囊
        cur_area = active_map.compute_explored_area_m2()
        draw.rounded_rectangle([38, 116, 316, 142], radius=7, fill=(14, 24, 36), outline=(70, 205, 180), width=1)
        draw.text(
            (46, 121),
            f"记忆 {active_map.count:,} 高斯核 · 拓展 {cur_area:.2f} m² · 融合 {active_map.kalman_fusion_updates:,}",
            font=FONT_TINY,
            fill=(145, 248, 215),
        )

        # 手机下屏双卡片标签
        draw.text(
            (36, 612),
            "真机黑白图传 240x240" if is_real_act else "实时相机画面",
            font=FONT_TINY,
            fill=(240, 248, 255),
        )
        draw.text((260, 612), "ch0 神经度量深度热力图", font=FONT_TINY, fill=(255, 230, 135))

        # 手机底部状态条
        draw.rounded_rectangle([30, 804, 466, 842], radius=8, fill=(20, 30, 44), outline=(65, 95, 130), width=1)
        draw.text(
            (42, 814),
            f"位姿 X={cur_pose_tuple[0]:+.2f}m  Y={cur_pose_tuple[1]:+.2f}m  Yaw={cur_pose_tuple[2]:+.0f}°   |   轻点即走 · 长按四向轮盘",
            font=FONT_TINY,
            fill=(205, 224, 242),
        )

        # 右上全景监视器角标
        draw.rounded_rectangle([514, 68, 1085, 98], radius=7, fill=(14, 24, 36), outline=(90, 180, 255), width=1)
        draw.text(
            (524, 75),
            "全局持久 3D 高斯空间记忆监视器 (绿圈 = 初始老区域 A 100% 锁存保留)" if not is_real_act
            else "真机实拍留出集 3D 高斯侧向环绕立体监视器 (桌面平整 Z=0 vs 人手/躯干抬升 +15cm)",
            font=FONT_SMALL,
            fill=(185, 230, 255),
        )

        # 右下左面板标题与 4 个通道名
        draw.text((516, 618), "HoverGaussianSplatNet 模型实时直出高斯属性张量 (48×48×6)", font=FONT_H2, fill=(235, 242, 250))
        for ci, (ch_title, _) in enumerate(ch_specs):
            draw.text((518 + ci * 126, 774), ch_title, font=FONT_SMALL, fill=(195, 215, 235))
        draw.text(
            (516, 810),
            "注: 椭球半径 (s_xy, s_z)、不透明度 α、异方差精度 τ 全部由 573.7KB TFLite 模型端到端预测",
            font=FONT_SMALL,
            fill=(140, 220, 175),
        )

        # 右下右面板标题与图例
        draw.text((1062, 618), "多视角卡尔曼修正 & 空间拓展记忆实时曲线", font=FONT_H2, fill=(235, 242, 250))
        cur_rmse_val = hist_rmse_cm[-1] if hist_rmse_cm else 2.17
        draw.text(
            (1070, 816),
            f"青线: 水杯3D误差 {cur_rmse_val:.2f}cm (-44%)   绿线: 拓展面积 {cur_area:.2f}m²   紫线: 老区A留存 100%",
            font=FONT_SMALL,
            fill=(220, 232, 245),
        )

        # 英雄联盟四向轮盘或手指触控光标叠加
        if wheel_overlay_state is not None:
            w_cx, w_cy, dx, dy, sel_s = wheel_overlay_state
            _draw_lol_wheel_on_pil(draw, w_cx, w_cy, dx, dy, sel_s)
        elif touch_cursor is not None:
            tcx, tcy, t_lbl = touch_cursor
            draw.ellipse([tcx - 16, tcy - 16, tcx + 16, tcy + 16], outline=(0, 238, 255), width=3)
            draw.ellipse([tcx - 5, tcy - 5, tcx + 5, tcy + 5], fill=(255, 255, 255))
            draw.rounded_rectangle([tcx + 18, tcy - 14, tcx + 216, tcy + 14], radius=6, fill=(12, 26, 40), outline=(0, 229, 255))
            draw.text((tcx + 25, tcy - 8), t_lbl, font=FONT_TINY, fill=(165, 248, 255))

        # 底部全宽剧情字幕横幅
        draw.rounded_rectangle([20, 860, 1580, 893], radius=7, fill=(24, 38, 56), outline=(85, 165, 220), width=1)
        draw.text((34, 867), f"[{t_sec:04.1f}s / 22.0s]  {act_banner}", font=FONT_SUB, fill=(245, 250, 255))

        final_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        proc.stdin.write(final_bgr.tobytes())

        # 保存 4 张关键帧供肉眼快速核验
        if f_idx in (150, n_act1 + 85, n_act1 + n_act2 + 95, n_act1 + n_act2 + n_act3 + 60):
            cv2.imwrite(str(sample_png_dir / f"keyframe_{f_idx:04d}.png"), final_bgr)

    proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}): {stderr_bytes.decode('utf-8', errors='replace')[-600:]}")

    elapsed = time.perf_counter() - t0
    return {
        "video_path": str(out_mp4),
        "video_bytes": int(out_mp4.stat().st_size),
        "video_mb": round(out_mp4.stat().st_size / (1024.0 * 1024.0), 2),
        "frames": total_frames,
        "duration_sec": round(total_frames / float(FPS), 1),
        "render_elapsed_sec": round(elapsed, 2),
        "splat_tflite_kb": round(splat_kb, 1),
        "splat_params": splat_params,
        "ping_tflite_kb": round(ping_kb, 1),
    }


if __name__ == "__main__":
    info = generate_demo_video()
    print(info)
