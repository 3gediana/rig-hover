# -*- coding: utf-8 -*-
"""v18-v19 修复的回归测试：tracker 失锁硬重置 / last_move_step 累计位移 /
悬崖 void-majority 盲视修复。全部 CPU 级，GeoNet 用桩。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from living_desktop_brain_v15 import (  # noqa: E402
    RobustDynamicObstacleTracker,
    _tracker_reinit,
    detect_visual_cliff_zero_oracle,
)


# ---------------- tracker 回归 ----------------

def test_14cm_gate_rejects_then_reinit_accepts() -> None:
    """移出 >14cm 的杯子检测被关联门拒绝；_tracker_reinit 硬重置绕过。"""
    tr = RobustDynamicObstacleTracker(dt=0.1)
    assert tr.update(0.40, -0.30, 0.04, 0) is True
    # 静止假设下 25cm 跳变 -> 关联门拒绝
    assert tr.update(0.65, -0.30, 0.04, 1) is False
    _tracker_reinit(tr, 0.65, -0.30, 0.04, 3)
    assert abs(tr.x[0] - 0.65) < 1e-9 and abs(tr.x[1] + 0.30) < 1e-9
    assert tr.is_moving is False and tr.n_obs == 1
    assert tr.last_obs_step == 3 and tr.last_move_step == 3
    assert tr._move_ref == (0.65, -0.30)


def test_last_move_step_uses_cumulative_displacement() -> None:
    """慢速目标每步 <3cm 但累计 >3cm 时 last_move_step 才刷新（幻影检测刷不动）。"""
    tr = RobustDynamicObstacleTracker(dt=0.1)
    tr.update(0.40, -0.30, 0.04, 0)
    # k=1 建立移动参考点（_move_ref）；k=2/3 累计 1.5/3.0cm 不刷；k=4 累计 4.5cm >3cm 刷新
    for k in range(1, 5):
        tr.update(0.40 + 0.015 * k, -0.30, 0.04, k)
    assert tr.last_move_step == 4
    tr.update(0.40 + 0.015 * 5, -0.30, 0.04, 5)
    assert tr.last_move_step == 4  # 距新参考点仅 1.5cm -> 不刷
    tr.update(0.40 + 0.015 * 7, -0.30, 0.04, 7)
    assert tr.last_move_step == 7  # 距参考点累计 4.5cm >3cm -> 再次刷新


def test_phantom_obs_does_not_count_as_movement() -> None:
    """落在冻结位置的幻影检测只刷新 last_obs_step，不刷新 last_move_step。"""
    tr = RobustDynamicObstacleTracker(dt=0.1)
    tr.update(0.40, -0.30, 0.04, 0)
    assert tr.last_move_step == 0 or tr.last_move_step <= 1
    for k in range(1, 6):
        tr.update(0.405, -0.298, 0.04, k)  # 3mm 抖动
    assert tr.last_obs_step == 5
    assert tr.last_move_step < 5  # 从未累计出 3cm 位移


def test_moving_stale_autoresets_on_jump() -> None:
    """is_moving + 失联 >=2 步时远距检测自动开新轨（无需外部 reinit）。"""
    tr = RobustDynamicObstacleTracker(dt=0.16)
    # 先让 tracker 判出移动：7 帧线性位移 >7.5cm
    for k in range(7):
        assert tr.update(0.40 + 0.015 * k, -0.30, 0.04, k) is True
    assert tr.is_moving is True
    # 失联 3 步，再在 20cm+ 外出现 -> 自动重置而非拒绝
    tr.predict(8)
    tr.predict(9)
    tr.predict(10)
    assert tr.update(0.75, -0.30, 0.04, 10) is True
    assert abs(tr.x[0] - 0.75) < 1e-9


# ---------------- 悬崖 void-majority 回归 ----------------

class _StubGeo:
    """GeoNet 对合成暗像素预测 z~0（已记录的盲区）；桩返回全零高度图。"""

    def infer_frame(self, bgr, prev_bgr, pitch_deg=0.0, roll_deg=0.0, auto_level=False):
        return {"pts_grid": np.zeros((240, 240, 3), dtype=np.float64)}


def _rays_cam() -> np.ndarray:
    """60° FOV pinhole 射线：x 前 / y 右 / z 上。v>120 的行 rz<0（朝下）。"""
    u, v = np.meshgrid(np.arange(240), np.arange(240))
    t30 = math.tan(math.radians(30.0))
    rays = np.stack(
        [np.ones_like(u, dtype=np.float64),
         (u - 119.5) / 119.5 * t30,
         -(v - 119.5) / 119.5 * t30],
        axis=-1,
    )
    return rays


def _img_dark_below_row(row: int) -> np.ndarray:
    img = np.full((240, 240, 3), 140, dtype=np.uint8)  # 亮木纹桌
    img[row:, :, :] = 20  # 深暗深渊
    return img


def test_void_majority_mid_range() -> None:
    """S3a 坠落修复：深渊占 ROI >50% 行 >=30 时单独成检（无需亮桌边界在 ROI 内）。"""
    bgr = _img_dark_below_row(175)
    det = detect_visual_cliff_zero_oracle(
        bgr, bgr, _StubGeo(), (0.0, 0.0, 0.0, 0.0, 0.0), _rays_cam())
    assert det is not None
    assert det["source"] == "void_majority"
    assert 0.35 < det["dist_to_rim_m"] < 0.72
    assert det["rim_xy"][0] > 0.30  # rim 在正前方


def test_void_majority_near_blind_case() -> None:
    """全 ROI 深渊（rim 已在近端/射线够不到）-> 保守 8cm 估距而非漏检。"""
    bgr = _img_dark_below_row(132)
    det = detect_visual_cliff_zero_oracle(
        bgr, bgr, _StubGeo(), (0.0, 0.0, 0.0, 0.0, 0.0), _rays_cam())
    assert det is not None
    assert det["source"] == "void_majority_near"
    assert abs(det["dist_to_rim_m"] - 0.08) < 1e-6


def test_no_void_returns_none() -> None:
    """亮桌面 + 小黑斑（<240px）不成检。"""
    bgr = np.full((240, 240, 3), 140, dtype=np.uint8)
    bgr[180:190, 100:110, :] = 25  # 10x10=100px 黑斑 <240 阈值
    det = detect_visual_cliff_zero_oracle(
        bgr, bgr, _StubGeo(), (0.0, 0.0, 0.0, 0.0, 0.0), _rays_cam())
    assert det is None


def test_pred_shortcut_skips_infer_frame() -> None:
    """统一大脑集成契约：传 pred 时绝不二次调用 geo_fe.infer_frame（防时序状态污染+省 15ms）。"""

    class _ExplodingGeo:
        def infer_frame(self, *a, **k):  # noqa: ANN002, ANN003
            raise AssertionError("infer_frame must not be called when pred is passed")

    bgr = _img_dark_below_row(175)
    pred = {"pts_grid": np.zeros((240, 240, 3), dtype=np.float64)}
    det = detect_visual_cliff_zero_oracle(
        bgr, bgr, _ExplodingGeo(), (0.0, 0.0, 0.0, 0.0, 0.0), _rays_cam(), pred=pred)
    assert det is not None
    assert det["source"] == "void_majority"


# ---------------- 方向感知悬崖净空门（v24 加固） ----------------

def test_cliff_guard_side_rim_does_not_freeze() -> None:
    """rim 在车体侧向 90°、距离 0.28m（<hold_m）：不接合，机器人仍可前进。"""
    from pc_unified_brain_app import _cliff_guard
    det = {"dist_to_rim_m": 0.28, "rim_xy": (0.0, -0.28)}  # 正左/右侧
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 0.0))
    assert engaged is False and clear == 0.85


def test_cliff_guard_front_rim_graded() -> None:
    """rim 正前方 0.40m（锥内）：接合且 clearance=0.40-0.30=0.10，渐刹不抱死。"""
    from pc_unified_brain_app import _cliff_guard
    det = {"dist_to_rim_m": 0.40, "rim_xy": (0.40, 0.0)}  # 车头正前
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 0.0))
    assert engaged is True and abs(clear - 0.10) < 1e-9


def test_cliff_guard_front_rim_hard_stop() -> None:
    """rim 正前方 ≤hold_m：clearance=0 硬停。"""
    from pc_unified_brain_app import _cliff_guard
    det = {"dist_to_rim_m": 0.25, "rim_xy": (0.25, 0.0)}
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 0.0))
    assert engaged is True and clear == 0.0


def test_cliff_guard_near_rim_any_bearing() -> None:
    """贴脸 rim（<0.15m）方位不可信，任何朝向都接合。"""
    from pc_unified_brain_app import _cliff_guard
    det = {"dist_to_rim_m": 0.08, "rim_xy": (0.0, 0.08)}  # 正后方也刹
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 0.0))
    assert engaged is True and clear == 0.0


def test_cliff_guard_yaw_rotated() -> None:
    """车体 yaw=90° 时 rim 在世界 +x 但相对车体在侧向 → 不接合。"""
    from pc_unified_brain_app import _cliff_guard
    det = {"dist_to_rim_m": 0.28, "rim_xy": (0.28, 0.0)}
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 90.0))
    assert engaged is False


def test_cliff_guard_none() -> None:
    from pc_unified_brain_app import _cliff_guard
    engaged, clear = _cliff_guard(None, (0.0, 0.0, 0.0))
    assert engaged is False and clear == 0.85


def test_cliff_guard_cone_boundary() -> None:
    """45° 锥边界：44° 接合、46° 放行。"""
    from pc_unified_brain_app import _cliff_guard
    d = 0.40
    for deg, want in ((44.0, True), (46.0, False)):
        r = math.radians(deg)
        det = {"dist_to_rim_m": d, "rim_xy": (d * math.cos(r), d * math.sin(r))}
        engaged, _ = _cliff_guard(det, (0.0, 0.0, 0.0))
        assert engaged is want


def test_cliff_guard_prefers_detector_bearing() -> None:
    """det 带 bearing_deg（最近点方位）时优先于 rim_xy 中位方位——L 形桌角防漏刹。"""
    from pc_unified_brain_app import _cliff_guard
    # rim 中位在 90°（锥外），但最近点在 10°（锥内）→ 应接合
    det = {"dist_to_rim_m": 0.28, "rim_xy": (0.0, -0.28), "bearing_deg": 10.0}
    engaged, clear = _cliff_guard(det, (0.0, 0.0, 0.0))
    assert engaged is True and clear == 0.0
