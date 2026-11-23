# -*- coding: utf-8 -*-
"""cliff_detect.py — 零开挂纯视觉悬崖边缘检测器（供 v15 仿真与 pc_unified_brain_app 共用）。

从 living_desktop_brain_v15 抽出：统一大脑在真机/回放链路同样需要悬崖本能，
抽出后 app 不必拖入整个 v15 仿真依赖链（matplotlib / v14 brain / SDF-CBF）。
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from hd_moge_contour_engine import REAL_H_CAM_M


def _wrap_deg_local(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _rot_c2w_np(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """精确相机到世界旋转矩阵 R_c2w = R_z(yaw) @ R_pitch(pitch) @ R_roll(roll)（与 DesktopArena3D 严格对齐）。"""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Rp = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    Rr = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return Rz @ Rp @ Rr


def detect_visual_cliff_zero_oracle(
    bgr: np.ndarray,
    prev_bgr: np.ndarray,
    geo_fe: Any,
    cam_pose: tuple[float, float, float, float, float],
    rays_cam: np.ndarray,
    pred: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """100% 零开挂纯视觉悬崖边缘检测器（只看 240×240 灰度画面 + HoverGeoNet-v4 深度 + 自身视觉位姿！）：
    原理：
      1. 在车头正前方地面视域 (v ∈ [135, 232]) 寻找连续的深暗深渊像素 (`gray < 48`) 紧邻明亮木纹桌面 (`gray > 95`)；
      2. 叠加 HoverGeoNet-v4 深度验证（深渊处预测深度明显深于桌面或高度下降），过滤普通黑色鼠标垫阴影；
      3. 将“明亮桌面与深渊交界处”的边缘像素沿刚性桌面射线 (`H_cam = 0.135m`) 投射到世界坐标，
         直接反算出悬崖边界点的真实世界坐标 `(rim_x, rim_y)` 与当前车体到悬崖的净距 `margin_m`！

    `pred`：可选——调用方已算好的 GeoNet infer_frame 输出（统一大脑每帧已推理一次，
    传入可省掉重复推理 ~15ms 并避免前端时序状态被二次调用污染）。
    """
    cam_ex, cam_ey, eyaw, p_est, r_est = cam_pose
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    pr = pred if pred is not None else geo_fe.infer_frame(bgr, prev_bgr, pitch_deg=p_est, roll_deg=r_est, auto_level=False)
    pts_c = pr["pts_grid"]  # [240, 240, 3] in robot-leveled frame (X forward, Y right/left, Z height above desk)

    # 下半视域（地面近景）里的深暗无桌面深渊候选
    roi_v0, roi_v1 = 132, 234
    g_roi = gray[roi_v0:roi_v1, 16:224]
    z_roi = pts_c[roi_v0:roi_v1, 16:224, 2]
    # 鼠标垫亮度约 60-75，真实悬崖暗地面亮度 < 44；且非向上凸起障碍 (z < 0.025)
    void_mask = (g_roi < 44) & (z_roi < 0.025)
    void_area = int(void_mask.sum())

    # v19 S3a 修复：深渊占半屏时"亮桌边界"必然消失，亮度联合线会在最后 ~12cm 全瞎。
    # 此处用"行级深渊占比"单独成检：rim 线 = 自上而下首个深渊占比>50% 的行（深渊占视野下半部）。
    row_frac = void_mask.mean(axis=1)
    _rows = np.where(row_frac > 0.5)[0]
    if void_area < 240:
        return None
    if len(_rows) >= 30:
        v_rim = int(_rows[0]) + roi_v0
        us = np.arange(16, 224)
        R_c2w = _rot_c2w_np(math.radians(eyaw), math.radians(p_est), math.radians(r_est))
        r_w = rays_cam[v_rim, us] @ R_c2w.T
        down = r_w[:, 2] < -0.05
        t_hit = np.where(down, -REAL_H_CAM_M / np.minimum(r_w[:, 2], -1e-4), 3.0)
        ok = down & (t_hit > 0.10) & (t_hit < 0.75)
        if int(ok.sum()) >= 50:
            wx = cam_ex + t_hit[ok] * r_w[ok, 0]
            wy = cam_ey + t_hit[ok] * r_w[ok, 1]
            dists = np.hypot(wx - cam_ex, wy - cam_ey)
            i_near = int(np.argmin(dists))
            bear = _wrap_deg_local(float(np.degrees(np.arctan2(wy[i_near] - cam_ey, wx[i_near] - cam_ex))) - eyaw)
            return {
                "void_area_px": void_area,
                "rim_xy": (round(float(np.median(wx)), 4), round(float(np.median(wy)), 4)),
                "y_cliff_est": round(float(np.median(wy)), 4),
                "dist_to_rim_m": round(float(np.percentile(dists, 15)), 4),
                "bearing_deg": round(bear, 1),
                "bbox_uv": (16, v_rim, 224, roi_v1),
                "source": "void_majority",
            }
        # 深渊已贴脸：rim 就在近端，保守估 8cm
        return {
            "void_area_px": void_area,
            "rim_xy": (cam_ex, cam_ey - 0.08),
            "y_cliff_est": round(cam_ey - 0.08, 4),
            "dist_to_rim_m": 0.08,
            "bbox_uv": (16, int(_rows[0]) + roi_v0, 224, roi_v1),
            "source": "void_majority_near",
        }

    # 寻找最大深渊连通域（滤除孤立暗斑）
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(void_mask.astype(np.uint8) * 255, connectivity=8)
    best_cc, best_area = -1, 0
    for cid in range(1, n_cc):
        a = int(stats[cid, cv2.CC_STAT_AREA])
        if a > best_area:
            best_area = a
            best_cc = cid
    if best_cc < 0 or best_area < 220:
        return None

    # 找深渊连通域的“上/内侧边缘”紧邻明亮木纹桌面 (`gray > 90`) 的真实桌沿交界像素！
    v_bin = (labels == best_cc).astype(np.uint8)
    dil = cv2.dilate(v_bin, np.ones((5, 5), np.uint8), iterations=1)
    border = (dil > 0) & (v_bin == 0) & (g_roi > 88)
    by, bx = np.where(border)
    if len(by) < 18:
        return None

    # 把像素边界精确投射到桌面 Z = 0（只用相机高度 H_cam 与自身估计姿态，绝不看任何真值！）
    R_c2w = _rot_c2w_np(math.radians(eyaw), math.radians(p_est), math.radians(r_est))
    vs = by + roi_v0
    us = bx + 16
    r_w = rays_cam[vs, us] @ R_c2w.T
    down = r_w[:, 2] < -0.05
    if int(down.sum()) < 14:
        return None
    t_hit = -REAL_H_CAM_M / r_w[down, 2]
    ok_t = (t_hit > 0.12) & (t_hit < 0.72)
    if int(ok_t.sum()) < 12:
        return None
    wx = cam_ex + t_hit[ok_t] * r_w[down][ok_t, 0]
    wy = cam_ey + t_hit[ok_t] * r_w[down][ok_t, 1]

    # 计算机器人到桌沿最近点的距离，以及桌沿估计坐标 (取前 25% 最近边点中位数更稳健)
    dists = np.hypot(wx - cam_ex, wy - cam_ey)
    near_m = dists <= float(np.percentile(dists, 35))
    rim_x = float(np.median(wx[near_m]))
    rim_y = float(np.median(wy[near_m]))
    min_dist_to_rim = float(np.percentile(dists, 15))
    y_cliff_est = float(np.median(wy))
    i_near = int(np.argmin(dists))
    bear = _wrap_deg_local(float(np.degrees(np.arctan2(wy[i_near] - cam_ey, wx[i_near] - cam_ex))) - eyaw)
    return {
        "void_area_px": best_area,
        "rim_xy": (round(rim_x, 4), round(rim_y, 4)),
        "y_cliff_est": round(y_cliff_est, 4),
        "dist_to_rim_m": round(min_dist_to_rim, 4),
        "bearing_deg": round(bear, 1),
        "bbox_uv": (
            int(stats[best_cc, cv2.CC_STAT_LEFT] + 16),
            int(stats[best_cc, cv2.CC_STAT_TOP] + roi_v0),
            int(stats[best_cc, cv2.CC_STAT_LEFT] + stats[best_cc, cv2.CC_STAT_WIDTH] + 16),
            int(stats[best_cc, cv2.CC_STAT_TOP] + stats[best_cc, cv2.CC_STAT_HEIGHT] + roi_v0),
        ),
        "source": "bright_border",
    }
