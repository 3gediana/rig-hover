# -*- coding: utf-8 -*-
"""pc_unified_brain_app.py — RIG-Hover PC 端统一具身大脑（场景一 + 场景二，100% 自研训练模型，零 MoGe-2 运行时依赖）。

一键切换或同时支撑两大核心场景，同时兼容离线录像回放与真机 HTTP (`http://192.168.137.50`) 闭环：
- **场景一 (`--scenario 1`)**：`YuNet` 人脸追踪 → `YOLO26n-pose` 手腕与拳头追踪 → 自研 `DoubleKnockNet` 双敲触发
  → 自研 `HoverGeoNet-ViT-240` 把敲击点直接升维锁进 `2.0 mm` 3D 桌面地图 → 视觉-3D 重定位 + 自研 `EndSidePolicyNet`
  驱车前往记忆敲击点 → 驻留后自动回看人脸；
- **场景二 (`--scenario 2`)**：自研 `HoverGeoNet-ViT-240` 构建 `2.0 mm` 纯绿色超密点云 + Dynablox 零残影动态手分离
  → 在 3D 画面上直接 `Shift+左键` 点击桌面目标 → `5 cm` A* 绕障规划 → 逐帧 2D-3D PnP + KISS-ICP 无里程计视觉重定位
  → 自研 PPO 点导航策略 `ping_point_nav_v1` + `EndSideSafetyShield` 闭环抵达！

示例命令：
  # 1) 离线回放场景二（带交互式 3D 窗口，左键旋转、滚轮缩放、Shift+左键点击桌面目标）
  python pc_unified_brain_app.py --scenario 2 --source frames --input D:\\robot_work\\evidence\\sessions\\none_d0.5_front_night_181841\\frames

  # 2) 离线回放场景一（开发集双敲录像 65a54d（封存集须显式 --final-eval），headless 输出状态与 3D 敲击落针日志）
  python pc_unified_brain_app.py --scenario 1 --source video --input D:\\robot_work\\dataset\\takes\\robot_20261115_150045_65a54d\\take.mp4 --headless --max-frames 120

  # 3) 真机连网只读预览（绝不下发控制，确认 3D 点云与重定位正常后再加 --send-cmd）
  python pc_unified_brain_app.py --scenario 2 --source live --base http://192.168.137.50
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (HERE, REPO / "pc" / "train", REPO / "pc" / "perception", REPO / "pc" / "pack"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from end_side_controller import (  # noqa: E402
    EndSideFeatureBuilder,
    EndSidePolicyNet,
    EndSideSafetyShield,
)
from end_side_protocol import DriveCommand, RobotTelemetry, UdpTelemetryReceiver  # noqa: E402
import sealed  # noqa: E402
from hd_dynamic_contour_slam import (  # noqa: E402
    HdDynamicContourSlamEngine,
    HoverGeoNet240Frontend,
    render_hd_dual_layer_green,
)
from cliff_detect import detect_visual_cliff_zero_oracle  # noqa: E402
from hover_geonet_240 import GEO_H, GEO_W  # noqa: E402
from mono_depth_model import make_camera_ray_grid  # noqa: E402
from neural_gaussian_splat_model import OrbitCamera3D, PersistentGaussianMemoryMap  # noqa: E402
from real_robot_bridge import (  # noqa: E402
    CALIB_FOV_H_DEG,
    CalibratedGazeServo,
    CameraRateKnockStream,
    FirmwareActuatorClient,
    FusedHandFacePerception,
    OnlineYawGyroBiasEstimator,
    project_knock_pixel_to_world_xy,
)
from scenario1_own_models_e2e import (  # noqa: E402
    POLICY_NAV_CKPT,
    POLICY_S1_CKPT,
    YOLO_POSE_PT,
    YUNET_ONNX,
    load_rl_policy,
)
from scenario2_visual_nav_closed_loop import Visual3DLandmarkRelocalizer, _wrap_deg  # noqa: E402
from scenario_v2 import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    MODE_NAMES,
    TargetLock,
)
from train_double_knock import DoubleKnockDetector, DoubleKnockNet  # noqa: E402

VIT_CKPT = REPO / "runs_end_side" / "hover_geonet_vit_240" / "hover_geonet_vit_b1_fp16.pt"
MBV3_CKPT = REPO / "runs_end_side" / "hover_geonet_240" / "hover_geonet_240_fp16.pt"
KNOCK_DEPLOY_CKPT = REPO / "runs_knock" / "double_knock_fpsaug2_20261115" / "double_knock_deploy.pt"
YOLOE_SEG_PT = REPO / "pc" / "perception" / "assets" / "yoloe-26n-seg.pt"

_PROJ = PersistentGaussianMemoryMap(max_splats=8)


def _cliff_guard(
    cliff_det: dict[str, Any] | None,
    est_pose: tuple[float, float, float],
    hold_m: float = 0.30,
    cone_deg: float = 45.0,
) -> tuple[bool, float]:
    """方向感知的悬崖净空门。

    真实负样本动机：rim 出现在车身侧向时二元冻结会让机器人永远开不到
    合法目标。本函数只在两种情形接合净空：
      1. rim 位于前向锥 `|bearing| <= cone_deg` 内（行进方向上真有悬崖）；
      2. `dist_to_rim_m < 0.15` 贴脸兜底（rim 几乎在轮下，方位不可信时仍刹）。

    返回 `(engaged, clearance_m)`。接合时 `clearance_m = max(0, dist - hold_m)`，
    与 Shield 的 sqrt 刹车包络天然衔接：hold+0.25m 起渐刹、hold_m 处硬停，
    取代原先"30cm 处突然抱死"的 0/1 切换。
    """
    if cliff_det is None:
        return False, 0.85
    d = float(cliff_det.get("dist_to_rim_m") or 9.9)
    bear_v = cliff_det.get("bearing_deg")
    if bear_v is not None:
        bear = abs(float(bear_v))
    else:
        rx, ry = cliff_det.get("rim_xy") or (None, None)
        if rx is None:
            engaged = d < hold_m
            return engaged, (0.0 if engaged else 0.85)
        ex, ey, eyaw = float(est_pose[0]), float(est_pose[1]), float(est_pose[2])
        dxw, dyw = float(rx) - ex, float(ry) - ey
        cy, sy = math.cos(math.radians(eyaw)), math.sin(math.radians(eyaw))
        bx = cy * dxw + sy * dyw
        by = -sy * dxw + cy * dyw
        bear = abs(math.degrees(math.atan2(by, bx)))
    if d < 0.15 or bear <= cone_deg:
        return True, max(0.0, d - hold_m)
    return False, 0.85


class UnifiedRobotBrain:
    """100% 自研模型驱动的 RIG-Hover PC 端统一具身大脑（Scenario 1 + Scenario 2）。"""

    def __init__(self, use_lite_geonet: bool = False, enable_knock: bool = True, static_decim: int = 3,
                 reloc_decim: int = 2) -> None:
        ckpt = MBV3_CKPT if (use_lite_geonet or not VIT_CKPT.exists()) else VIT_CKPT
        self.geonet_ckpt = ckpt
        # 几何前端的相机以权重为准（checkpoint 里记录了训练相机）。射线栅格和桌面先验是这个模型的输入，
        # 不能拿实测相机去喂按别的相机训练的权重；下方 SLAM 与悬崖射线都用前端自己解析出的相机。
        self.geo_fe = HoverGeoNet240Frontend(ckpt, use_fp16=True)
        self.slam = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=self.geo_fe.h_cam_m, obs_radius_m=0.095,
                                               static_decim=static_decim, async_static=True)
        self.policy_s1: EndSidePolicyNet = load_rl_policy(POLICY_S1_CKPT)
        self.policy_nav: EndSidePolicyNet = load_rl_policy(POLICY_NAV_CKPT)
        self.shield = EndSideSafetyShield()
        self.reloc_decim = max(1, int(reloc_decim))
        # 感知线程池：人脸/姿态/握拳检测与主线程 CPU SLAM 并行重叠
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="percep")

        self.fused_percep: FusedHandFacePerception | None = None
        if enable_knock:
            self.fused_percep = FusedHandFacePerception(
                yunet_onnx=YUNET_ONNX,
                yolo_pose_pt=YOLO_POSE_PT,
                yoloe_pt=YOLOE_SEG_PT,
                enable_yoloe_closeup=True,
                yoloe_stride=2,
            )
        self.gaze_servo = CalibratedGazeServo(fov_h_deg=CALIB_FOV_H_DEG)
        self.yaw_bias_est = OnlineYawGyroBiasEstimator()

        self.knock_det: DoubleKnockDetector | None = None
        if enable_knock and KNOCK_DEPLOY_CKPT.exists():
            # 双敲网络极小 (hidden=64,size=96)：固定 CPU，把 GPU 完全让给 GeoNet，
            # 且注入真实 YOLO 手腕运动学框后恢复 42% 校准门限（召回 ~90%+ 且压低误报）。
            dev = "cpu"
            kb = torch.load(KNOCK_DEPLOY_CKPT, map_location=dev, weights_only=False)
            if kb.get("train_split") != "dev":
                print("[Knock][WARN] legacy checkpoint predates dev-only training; treat metrics as contaminated until retrained", flush=True)
            km = DoubleKnockNet(in_ch=2, kin_dim=8, hidden_dim=64, num_classes=4).to(dev).eval()
            km.load_state_dict(kb["state_dict"])
            self.knock_det = DoubleKnockDetector(
                model=km,
                device=dev,
                window=int(kb.get("window", 48)),
                size=int(kb.get("size", 96)),
                imp_threshold=0.38,
                double_threshold=0.45,
                gate_state=MODE_GAZE_FIST,
                sem_latch_s=2.0,
                single_confirm_thr=0.72,
                single_lookback_s=1.5,
            )

        self.relocalizer: Visual3DLandmarkRelocalizer | None = None
        self.prev_bgr: np.ndarray | None = None
        self.mode: int = MODE_IDLE
        self.est_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # 纯机身位姿 (body_x, body_y, body_yaw_deg)
        self.target_xy: tuple[float, float] | None = None
        self.path_xy: list[tuple[float, float]] = []
        self.frames_seen: int = 0
        self.dwell_count: int = 0
        self.last_pred: dict[str, Any] | None = None
        self._last_step_t: float | None = None
        self._last_body_yaw_tel: float | None = None
        self._last_wheel_x_m: float | None = None
        self._last_wrist_box: list[float] | None = None
        self._last_peak_meta: tuple[float, list[float] | None, RobotTelemetry] | None = None
        self._cliff_standoff_cnt: int = 0
        # 视觉悬崖本能（与 living_desktop_brain_v15 共用的零开挂检测器）：
        # APPROACH 巡航中 dist_to_rim_m < hold_m 时通过 Shield clearance=0 硬刹停 vx。
        self.rays_cam = make_camera_ray_grid(GEO_H, GEO_W, fov_h_deg=self.geo_fe.fov_h_deg, pitch_deg=0.0, roll_deg=0.0)
        self.cliff_hold_m: float = 0.25
        self.cliff_cone_deg: float = 45.0
        self._cliff_active: bool = False

    def _ensure_relocalizer(self, bgr: np.ndarray, pred: dict[str, Any], cam_yaw_deg: float | None = None) -> None:
        st_pts, st_nrms, _ = self.slam.get_confirmed_static_cloud(min_wt=1.0)
        if len(st_pts) < 1000:
            return
        kf_pose = (self.est_pose[0], self.est_pose[1], float(cam_yaw_deg if cam_yaw_deg is not None else self.est_pose[2]))
        if self.relocalizer is None:
            self.relocalizer = Visual3DLandmarkRelocalizer(st_pts, st_nrms)
            self.relocalizer.add_keyframe(bgr, pred["pts_grid"], pred["valid_grid"], kf_pose)
        else:
            self.relocalizer.static_pts = st_pts
            self.relocalizer.static_nrms = st_nrms
            if len(self.relocalizer.keyframes) < 5 and self.frames_seen % 6 == 0:
                self.relocalizer.add_keyframe(bgr, pred["pts_grid"], pred["valid_grid"], kf_pose)

    def set_3d_target(self, wx: float, wy: float) -> dict[str, Any]:
        """锁定 3D 桌面目标（来自场景二用户点击或场景一双敲 3D 升维），实时触发 A* 绕障规划。"""
        self.target_xy = (float(wx), float(wy))
        trav = self.slam.rebuild_traversability_grid(self.est_pose[0], self.est_pose[1], self.est_pose[2])
        plan = trav.click_and_plan(wx, wy, robot_x=self.est_pose[0], robot_y=self.est_pose[1], robot_yaw_deg=self.est_pose[2], max_snap_m=0.12)
        if plan["accepted"] and len(plan["path"]) >= 2:
            self.path_xy = [(float(p[0]), float(p[1])) for p in plan["path"]]
        else:
            self.path_xy = [(self.est_pose[0], self.est_pose[1]), (float(wx), float(wy))]
        self.mode = MODE_APPROACH
        self.dwell_count = 0
        self._best_rem = 1e9
        self._approach_stall = 0
        self._cliff_standoff_cnt = 0
        return {"target_xy": self.target_xy, "astar_accepted": bool(plan["accepted"]), "path_pts": len(self.path_xy)}

    def _knock_step_full(self, bgr240: np.ndarray, wrist_box: list[float] | None, timestamp_s: float | None) -> dict[str, Any]:
        g240 = cv2.cvtColor(bgr240, cv2.COLOR_BGR2GRAY)
        return self.knock_det.step(g240, wrist_box=wrist_box, scenario_state=None, timestamp=timestamp_s)

    def step(
        self,
        bgr: np.ndarray,
        telemetry: RobotTelemetry | None = None,
        scenario: int = 2,
        timestamp_s: float | None = None,
        auto_level: bool = True,
        tel_now: RobotTelemetry | None = None,
        ext_knock: tuple[dict[str, Any], tuple[float, list[float] | None, RobotTelemetry] | None] | None = None,
    ) -> dict[str, Any]:
        """单帧统一推理：自研 HoverGeoNet-ViT-240 + Dynablox + 视觉-3D 重定位 + 场景一/二 RL 决策。"""
        self.frames_seen += 1
        _t = time.perf_counter
        now_s = float(timestamp_s) if timestamp_s is not None else _t()
        dt_s = 0.08 if self._last_step_t is None else float(np.clip(now_s - self._last_step_t, 0.015, 0.25))
        self._last_step_t = now_s

        tel = telemetry or RobotTelemetry(pitch=3.0, roll=0.0, yaw=self.est_pose[2], robot_state=1)
        tel_cur = tel_now or tel

        # 1) 纯机身 SE(2) 里程计传播 + QMI8658C 在线陀螺零漂抵消：
        #    严格区分头部视线航向 (tel.yaw) 与底盘航向 (body_yaw = tel.yaw - tel.head_q)，
        #    并实时减去静止期辨识出的陀螺零漂率（真机实测 -5.58°/s，恰超 imu.cc 5°/s 阈值被置零！）
        bgr240 = cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA) if bgr.shape[:2] != (240, 240) else bgr
        img_static = False
        if self.prev_bgr is not None:
            g_c = cv2.cvtColor(bgr240, cv2.COLOR_BGR2GRAY)
            g_p = cv2.cvtColor(self.prev_bgr, cv2.COLOR_BGR2GRAY)
            img_static = float(np.mean(cv2.absdiff(g_c, g_p))) < 4.2

        yaw_bias_dps = 0.0
        if telemetry is not None:
            prev_vy = getattr(self, "_cmd_f", (0.0, 0.0, 0.0))[1]
            yaw_bias_dps = self.yaw_bias_est.update(
                tel, now_s=now_s, cmd_vyaw_norm=prev_vy, dq_head_dps=self.gaze_servo.dq_filt, image_static=img_static
            )
            body_yaw_tel = _wrap_deg(float(tel.yaw) - float(tel.head_q))
            wheel_m = float(tel.wheel_x_m)
            if self._last_body_yaw_tel is not None and self._last_wheel_x_m is not None:
                raw_dyaw = _wrap_deg(body_yaw_tel - self._last_body_yaw_tel)
                dyaw_b = float(np.clip(raw_dyaw - yaw_bias_dps * dt_s, -35.0, 35.0))
                if abs(dyaw_b) < 0.18 and img_static:
                    dyaw_b = 0.0
                ds = float(np.clip(wheel_m - self._last_wheel_x_m, -0.12, 0.12)) if tel.robot_state == 1 else 0.0
                mid_rad = math.radians(self.est_pose[2] + 0.5 * dyaw_b)
                self.est_pose = (
                    self.est_pose[0] + ds * math.cos(mid_rad),
                    self.est_pose[1] + ds * math.sin(mid_rad),
                    _wrap_deg(self.est_pose[2] + dyaw_b),
                )
            self._last_body_yaw_tel = body_yaw_tel
            self._last_wheel_x_m = wheel_m

        cam_yaw_deg = _wrap_deg(self.est_pose[2] + float(tel.head_q))
        _st: dict[str, float] = {}

        # 若动态切到场景一，按需懒加载融合感知器
        if scenario == 1 and self.fused_percep is None:
            self.fused_percep = FusedHandFacePerception(
                yunet_onnx=YUNET_ONNX, yolo_pose_pt=YOLO_POSE_PT, yoloe_pt=YOLOE_SEG_PT, enable_yoloe_closeup=True, yoloe_stride=2
            )

        # 2) 主线程先跑 GeoNet GPU，随后在主线程跑 CPU SLAM 的同时把 YOLO/YuNet 丢进工人线程重叠执行
        _t0 = _t()
        reloc_due = (
            self.mode == MODE_APPROACH
            and self.relocalizer is not None
            and self.frames_seen >= 3
            and self.frames_seen % self.reloc_decim == 0
        )
        # 横躺/手持失稳姿态 (|pitch|>25 或 |roll|>25) 时跳过静态建图融合，保护桌面 3D 地图不被侧翻/手持画面污染
        tilted_off_desk = (telemetry is not None) and (abs(tel.pitch) > 25.0 or abs(tel.roll) > 25.0)
        auto_lvl_use = bool(auto_level and not tilted_off_desk)
        pitch_in = float(np.clip(tel.pitch, -25.0, 25.0)) if not tilted_off_desk else 3.0
        roll_in = float(np.clip(tel.roll, -25.0, 25.0)) if not tilted_off_desk else 0.0
        pred = self.geo_fe.infer_frame(
            bgr240, self.prev_bgr, pitch_deg=pitch_in, roll_deg=roll_in, auto_level=auto_lvl_use, cam_cloud=reloc_due
        )
        _st["geonet"] = (_t() - _t0) * 1e3
        self.last_pred = pred

        fut_percep = None
        if scenario == 1 and self.fused_percep is not None:
            need_cu = self.mode in (MODE_IDLE, MODE_GAZE_FACE, MODE_GAZE_FIST)
            fut_percep = self._pool.submit(self.fused_percep.detect, bgr240, now_s, need_cu)

        _t0 = _t()
        if tilted_off_desk:
            slam_rec = {
                "disturbed": True,
                "reason": "tilted_off_desk",
                "n_static_frame": 0,
                "n_dynamic_frame": 0,
                "n_static_map": int(len(self.slam.static_cloud.pos)),
                "carved_ghosts": 0,
                "rot_corr_deg": 0.0,
                "trans_corr_mm": 0.0,
                "slam_stage_ms": {"gate": 0.0, "icp": 0.0, "dynmask": 0.0},
            }
        else:
            slam_rec = self.slam.process_frame(
                bgr240, pred["pts_grid"], pred["nrm_grid"], pred["crease_grid"], pred["valid_grid"],
                robot_x=self.est_pose[0], robot_y=self.est_pose[1], robot_yaw_deg=cam_yaw_deg,
                pitch_deg=pitch_in, roll_deg=roll_in, align=(self.frames_seen > 1),
            )
        _st["slam"] = (_t() - _t0) * 1e3
        if reloc_due or self.frames_seen % 6 == 0:
            self._ensure_relocalizer(bgr240, pred, cam_yaw_deg=cam_yaw_deg)
        prev_bgr_for_cliff = self.prev_bgr if self.prev_bgr is not None else bgr240
        self.prev_bgr = bgr240.copy()

        # 2.5) 视觉悬崖本能（相机指向系转回机身航向门控）
        _t0 = _t()
        cliff_det = detect_visual_cliff_zero_oracle(
            bgr240, prev_bgr_for_cliff, self.geo_fe,
            (self.est_pose[0], self.est_pose[1], cam_yaw_deg, pitch_in, roll_in),
            self.rays_cam, pred=pred,
        )
        if cliff_det is not None and cliff_det.get("bearing_deg") is not None and abs(tel.head_q) > 0.5:
            # 将头部相机相对方位换算为底盘前向相对方位，防止歪头看侧面桌沿时误锁死正前行驶
            cliff_det = dict(cliff_det)
            cliff_det["bearing_deg"] = round(_wrap_deg(float(cliff_det["bearing_deg"]) + float(tel.head_q)), 2)
        _st["cliff"] = (_t() - _t0) * 1e3
        cliff_engaged, cliff_clear = _cliff_guard(cliff_det, self.est_pose, self.cliff_hold_m, self.cliff_cone_deg)
        if cliff_det is None:
            cliff_near = False
        elif self._cliff_active:
            cliff_near = cliff_det["dist_to_rim_m"] < self.cliff_hold_m + 0.03
        else:
            cliff_near = cliff_det["dist_to_rim_m"] < self.cliff_hold_m
        cliff_fired = cliff_near and not self._cliff_active
        self._cliff_active = cliff_near

        # 3) APPROACH 巡航中运行 2D-3D PnP + KISS-ICP 视觉重定位（仅高内点且低残差时融合修正机身位姿）
        reloc_info: dict[str, Any] = {"source": "map_build", "ransac_inliers": 0}
        _t0 = _t()
        if reloc_due:
            self.relocalizer.static_pts, self.relocalizer.static_nrms, _ = self.slam.get_confirmed_static_cloud(min_wt=1.0)
            prev_cam_pose = (self.est_pose[0], self.est_pose[1], cam_yaw_deg)
            ro = self.relocalizer.relocalize(
                bgr240, pred["pts_cam"], pred["nrm_cam"], pred["valid_grid"],
                imu_pitch_deg=pitch_in, imu_roll_deg=roll_in, prev_est_pose=prev_cam_pose,
            )
            inl = int(ro.get("ransac_inliers", 0))
            icp_mm = float(ro.get("icp_median_res_mm", 99.0))
            if inl >= 16 and icp_mm < 24.0:
                al = 0.55 if telemetry is not None else 0.75
                ro_body_yaw = _wrap_deg(float(ro["yaw_deg"]) - float(tel.head_q))
                self.est_pose = (
                    (1.0 - al) * self.est_pose[0] + al * float(ro["x"]),
                    (1.0 - al) * self.est_pose[1] + al * float(ro["y"]),
                    _wrap_deg(self.est_pose[2] + al * _wrap_deg(ro_body_yaw - self.est_pose[2])),
                )
            reloc_info = {"source": ro["source"], "ransac_inliers": inl, "icp_res_mm": icp_mm}
        _st["reloc"] = (_t() - _t0) * 1e3

        # 4) 收集融合人脸/握拳结果 + 24Hz 双敲流（或离线带 wrist_box 双敲）
        face_dets: list[dict[str, float]] = []
        fist_det: dict[str, Any] | None = None
        knock_out: dict[str, Any] = {"fired": 0, "p_double": 0.0}
        if fut_percep is not None:
            _t0 = _t()
            perc = fut_percep.result()
            face_dets = perc["faces"]
            fist_det = perc["fist"]
            self._last_wrist_box = perc["wrist_box"]
            _st["percep"] = (_t() - _t0) * 1e3

        if scenario == 1:
            _t0 = _t()
            peak_meta = None
            if ext_knock is not None:
                knock_out, peak_meta = ext_knock
            elif self.knock_det is not None:
                raw_ko = self._knock_step_full(bgr240, self._last_wrist_box, now_s)
                if raw_ko.get("peak"):
                    self._last_peak_meta = (now_s, self._last_wrist_box, tel)
                peak_meta = self._last_peak_meta
                if raw_ko.get("fired", 0):
                    p_sem = float(raw_ko.get("p_double", 0.0))
                    if self.mode == MODE_GAZE_FIST or (self._last_wrist_box is not None) or (p_sem >= 0.44):
                        knock_out = raw_ko
                    else:
                        knock_out = dict(raw_ko, fired=0)
                else:
                    knock_out = raw_ko
            _st["knock"] = (_t() - _t0) * 1e3

            # 高层交互状态机：IDLE <-> GAZE_FACE <-> GAZE_FIST -> (双敲) -> APPROACH -> DWELL -> GAZE_FACE
            if knock_out.get("fired", 0):
                wbox_k = (peak_meta[1] if peak_meta is not None else None) or self._last_wrist_box
                tel_k = (peak_meta[2] if peak_meta is not None else None) or tel
                if wbox_k is not None:
                    u_k = 0.5 * (float(wbox_k[0]) + float(wbox_k[2]))
                    v_k = min(236.0, float(wbox_k[3]))
                else:
                    u_k, v_k = 120.0, 176.0
                wx, wy, k_proj = project_knock_pixel_to_world_xy(
                    u_k, v_k, tel_k, self.est_pose,
                    dyn_pts_world=self.slam.dyn_pts, cliff_det=cliff_det, cliff_hold_m=self.cliff_hold_m,
                )
                knock_out["proj"] = k_proj
                self.set_3d_target(wx, wy)
            elif self.mode not in (MODE_APPROACH, MODE_DWELL):
                if fist_det is not None:
                    self.mode = MODE_GAZE_FIST
                elif face_dets:
                    self.mode = MODE_GAZE_FACE
                else:
                    self.mode = MODE_IDLE

        # EndSideFeatureBuilder 要求的无偏归一化检测槽（train_rl_policy 折叠矩阵要求 slot0.score == 1.0 以消零点偏置）
        feat_dets = [{**d, "score": 1.0} for d in face_dets] if face_dets else []

        # 5) 控制指令生成：APPROACH 视觉导航 / DWELL 驻留回看 / GAZE 静摩擦补偿丝滑伺服
        _t0 = _t()
        rem_m = 0.0
        gaze_dbg: dict[str, Any] = {}
        allow_body = (tel_cur.robot_state == 1) or (telemetry is None)

        if self.mode == MODE_APPROACH and self.target_xy is not None:
            ex, ey, eyaw = self.est_pose
            rem_m = math.hypot(self.target_xy[0] - ex, self.target_xy[1] - ey)
            bear = _wrap_deg(math.degrees(math.atan2(self.target_xy[1] - ey, self.target_xy[0] - ex)) - eyaw)
            if rem_m < getattr(self, "_best_rem", 1e9) - 0.005:
                self._best_rem = rem_m
                self._approach_stall = 0
            else:
                self._approach_stall = getattr(self, "_approach_stall", 0) + 1

            # 若已紧贴桌沿安全线且车头已对准敲击点，优雅完成停靠（不再顶着悬崖刹车空扭 15s）
            if cliff_engaged and cliff_clear <= 0.02 and abs(bear) < 18.0 and rem_m < 0.24:
                self._cliff_standoff_cnt += 1
            else:
                self._cliff_standoff_cnt = 0

            abort_unreachable = self._approach_stall > 95 and rem_m > getattr(self, "_best_rem", 0.0) * 0.96
            arrived = (rem_m <= 0.042) or (self._cliff_standoff_cnt >= 8)
            if arrived or abort_unreachable:
                if abort_unreachable:
                    self.target_xy = None
                self._best_rem = 1e9
                self._approach_stall = 0
                self._cliff_standoff_cnt = 0
                self.mode = MODE_DWELL if scenario == 1 else MODE_IDLE
                self.dwell_count = 0
                safe_cmd = self.shield.apply(DriveCommand(0.0, 0.0, tel_cur.head_q), mode=self.mode, remaining_m=0.0, pitch_deg=pitch_in)
            else:
                wp = self.target_xy
                if len(self.path_xy) >= 2:
                    di = int(np.argmin([math.hypot(px - ex, py - ey) for px, py in self.path_xy]))
                    for j in range(di, len(self.path_xy)):
                        if math.hypot(self.path_xy[j][0] - ex, self.path_xy[j][1] - ey) >= 0.13:
                            wp = self.path_xy[j]
                            break
                wp_bear = _wrap_deg(math.degrees(math.atan2(wp[1] - ey, wp[0] - ex)) - eyaw)
                tlock = TargetLock(valid=1, bearing_body=wp_bear, remaining_m=rem_m)
                f39 = EndSideFeatureBuilder.build(tel_cur, detections=feat_dets, target_lock=tlock, clearance_m=cliff_clear)
                rl_cmd = self.policy_nav.infer_numpy(f39)
                # 先对准后直行：偏角 > 22° 时优先原地平滑转向，防止窄桌面上大半径画弧扫出桌沿
                turn_gate = max(0.0, math.cos(math.radians(min(88.0, abs(wp_bear) * 1.6))))
                nom_vx = min(72.0, 88.0 * math.sqrt(max(0.0, rem_m / 0.35))) * turn_gate
                nom_vyaw = float(np.clip(wp_bear * 1.65, -58.0, 58.0))
                # 行进途中头部盯着桌面目标方向（"盯着那个位置走过去"）
                cx_tgt_equiv = float(np.clip(0.5 + (wp_bear - float(tel.head_q)) / CALIB_FOV_H_DEG, 0.05, 0.95))
                head_nav, _, _ = self.gaze_servo.step_visual_target(
                    cx_tgt_equiv, tel, tel_cur, now_s, dt_s, allow_body_turn=False
                )
                blend_cmd = DriveCommand(
                    vx=0.45 * rl_cmd.vx + 0.55 * nom_vx,
                    vyaw=0.40 * rl_cmd.vyaw + 0.60 * nom_vyaw,
                    head=head_nav,
                )
                safe_cmd = self.shield.apply(blend_cmd, mode=MODE_APPROACH, remaining_m=rem_m, pitch_deg=pitch_in, clearance_m=cliff_clear)
        elif self.mode == MODE_DWELL:
            self.dwell_count += 1
            # 到达后停轮抬头看人脸
            if face_dets:
                h_dw, _, gaze_dbg = self.gaze_servo.step_visual_target(
                    face_dets[0]["cx"], tel, tel_cur, now_s, dt_s, allow_body_turn=False
                )
            else:
                h_dw = self.gaze_servo.cmd_head_hold
            safe_cmd = DriveCommand(0.0, 0.0, h_dw)
            if self.dwell_count >= 20:
                self.mode = MODE_GAZE_FACE if face_dets else MODE_IDLE
        else:
            if self.mode == MODE_GAZE_FIST and fist_det is not None:
                h_cmd, vy_cmd, gaze_dbg = self.gaze_servo.step_visual_target(
                    fist_det["cx"], tel, tel_cur, now_s, dt_s, allow_body_turn=allow_body
                )
                safe_cmd = DriveCommand(0.0, vy_cmd, h_cmd)
            elif self.mode == MODE_GAZE_FACE and face_dets:
                h_cmd, vy_cmd, gaze_dbg = self.gaze_servo.step_visual_target(
                    face_dets[0]["cx"], tel, tel_cur, now_s, dt_s, allow_body_turn=allow_body
                )
                safe_cmd = DriveCommand(0.0, vy_cmd, h_cmd)
            elif scenario == 1:
                h_cmd, vy_cmd = self.gaze_servo.step_blind(tel_cur, now_s, dt_s, allow_body_scan=allow_body)
                safe_cmd = DriveCommand(0.0, vy_cmd, h_cmd)
            else:
                f39 = EndSideFeatureBuilder.build(tel_cur, detections=feat_dets, target_lock=None, clearance_m=1.0)
                raw_cmd = self.policy_s1.infer_numpy(f39)
                safe_cmd = self.shield.apply(raw_cmd, mode=self.mode, remaining_m=0.0, pitch_deg=pitch_in)
        _st["policy"] = (_t() - _t0) * 1e3

        # 仅对底盘 vx/vyaw 做轻度防抖平滑（head 已由静摩擦伺服+固件 167Hz LPF 精确阻尼，严禁再叠滞后！）
        if cliff_clear is not None and cliff_clear <= 0.0:
            self._cmd_f = (0.0, safe_cmd.vyaw, safe_cmd.head)
            safe_cmd = DriveCommand(0.0, safe_cmd.vyaw, safe_cmd.head)
        else:
            _p = getattr(self, "_cmd_f", None) or (safe_cmd.vx, safe_cmd.vyaw, safe_cmd.head)
            vx_sm = 0.40 * _p[0] + 0.60 * safe_cmd.vx
            vy_sm = 0.35 * _p[1] + 0.65 * safe_cmd.vyaw
            if abs(vy_sm) < 2.5:
                vy_sm = 0.0
            if abs(vx_sm) < 1.5:
                vx_sm = 0.0
            safe_cmd = DriveCommand(vx_sm, vy_sm, safe_cmd.head)
            self._cmd_f = (safe_cmd.vx, safe_cmd.vyaw, safe_cmd.head)

        return {
            "frame": self.frames_seen,
            "mode": MODE_NAMES.get(self.mode, str(self.mode)),
            "geonet_ms": round(float(pred["infer_ms"]), 2),
            "stage_ms": {k: round(v, 2) for k, v in _st.items()},
            "slam_stages": slam_rec.get("slam_stage_ms"),
            "static_pts": slam_rec["n_static_map"],
            "dyn_pts": slam_rec["n_dynamic_frame"],
            "est_pose": [round(v, 3) for v in self.est_pose],
            "yaw_bias_dps": round(yaw_bias_dps, 3),
            "tel": {"pitch": round(tel_cur.pitch, 2), "roll": round(tel_cur.roll, 2), "yaw": round(tel_cur.yaw, 2), "head_q": round(tel_cur.head_q, 2), "robot_state": int(tel_cur.robot_state)},
            "gaze": gaze_dbg,
            "target_xy": list(self.target_xy) if self.target_xy else None,
            "rem_m": round(rem_m, 3),
            "reloc": reloc_info,
            "knock": {"fired": int(knock_out.get("fired", 0)), "p_double": round(float(knock_out.get("p_double", 0.0)), 3), "src": knock_out.get("fire_src", "pair"), "imp": round(float(knock_out.get("p_impulse", 0.0)), 3), "sem": round(float(knock_out.get("p_double_sem", knock_out.get("p_double", 0.0))), 3), "pk": int(bool(knock_out.get("peak")))},
            "faces": face_dets,
            "fist": fist_det,
            "cliff": cliff_det,
            "cliff_fired": int(cliff_fired),
            "cliff_engaged": int(cliff_engaged),
            "cmd": {"vx": round(safe_cmd.vx, 2), "vyaw": round(safe_cmd.vyaw, 2), "head": round(safe_cmd.head, 2)},
        }

    def render_dashboard(self, bgr: np.ndarray, step_info: dict[str, Any], cam: OrbitCamera3D, w: int = 760, h: int = 520) -> np.ndarray:
        st_pts, st_nrm, st_cr = self.slam.get_confirmed_static_cloud(min_wt=1.0)
        pin = (self.target_xy[0], self.target_xy[1], 0.0) if self.target_xy is not None else None
        p3d = render_hd_dual_layer_green(
            st_pts, st_nrm, st_cr, self.slam.dyn_pts, self.slam.dyn_nrms,
            cam=cam, w=w, h=h, point_radius_px=1, target_pin_xyz=pin, robot_pose=self.est_pose, path_xy=self.path_xy,
        )
        side = np.full((h, 300, 3), (10, 16, 12), dtype=np.uint8)
        thumb = cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA).copy()
        for fd in step_info.get("faces") or []:
            fcx, fcy, fw, fh = fd["cx"] * 240.0, fd["cy"] * 240.0, fd["w"] * 240.0, fd["h"] * 240.0
            cv2.rectangle(thumb, (int(fcx - 0.5 * fw), int(fcy - 0.5 * fh)), (int(fcx + 0.5 * fw), int(fcy + 0.5 * fh)), (255, 220, 60), 1)
            cv2.circle(thumb, (int(fcx), int(fcy)), 3, (255, 240, 100), -1)
        fist_d = step_info.get("fist")
        if fist_d and fist_d.get("box"):
            bx1, by1, bx2, by2 = [int(v) for v in fist_d["box"]]
            cv2.rectangle(thumb, (bx1, by1), (bx2, by2), (40, 210, 255), 2)
        cv2.line(thumb, (120, 0), (120, 240), (45, 95, 60), 1)
        side[20:260, 30:270] = thumb
        cv2.rectangle(side, (30, 20), (270, 260), (70, 210, 120), 1)
        hud_lines = [
            f"Model: {self.geonet_ckpt.stem}",
            f"Frame: {step_info['frame']} | {step_info['geonet_ms']:.1f} ms",
            f"State: {step_info['mode']}",
            f"Map  : {step_info['static_pts']:,}st + {step_info['dyn_pts']:,}dyn",
            f"Pose : x={step_info['est_pose'][0]:.2f} y={step_info['est_pose'][1]:.2f}",
            f"       yaw={step_info['est_pose'][2]:.1f} deg",
            f"Knock: p={step_info['knock']['p_double']:.2f} fired={step_info['knock']['fired']}",
            f"Cliff: {step_info['cliff']['dist_to_rim_m']:.2f}m {step_info['cliff']['source']}" if step_info.get("cliff") else "Cliff: clear",
            f"Cmd  : vx={step_info['cmd']['vx']:.1f}",
            f"       vyaw={step_info['cmd']['vyaw']:.1f} h={step_info['cmd']['head']:.1f}",
            "Keys: Shift+Click=Target | 1/2=Scen",
            "      Drag=Orbit | q=Quit",
        ]
        for i, ln in enumerate(hud_lines):
            cv2.putText(side, ln, (14, 286 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 250, 190), 1, cv2.LINE_AA)
        return np.hstack([p3d, side])


def _fetch_live_robot(base_url: str) -> tuple[np.ndarray | None, RobotTelemetry]:
    import requests
    sess = requests.Session()
    sess.trust_env = False
    tel = RobotTelemetry()
    try:
        r = sess.get(f"{base_url}/status", timeout=1.5)
        tel = RobotTelemetry.from_status_json(r.json())
    except Exception:
        pass
    try:
        r = sess.get(f"{base_url}/capture", timeout=2.0)
        bgr = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        return bgr, tel
    except Exception:
        return None, tel
    finally:
        sess.close()


class _LiveFrameFetcher:
    """后台取帧线程 + ~490Hz UDP 遥测曝光对齐 + 24Hz 原速双敲推流。

    1. 订阅机器人 :9001 端口的 TLM1 二进制遥测流（零 httpd 负载），将每帧图像与曝光中点时刻
       (t_capture) 的精确 head_q / pitch / roll / yaw 对齐；若 UDP 暂未就绪则自动退回 /status。
    2. 若挂载 knock_stream，抓取到的全部 ~24fps 帧在解码当下立即送入双敲检测流，主控环就算
       跳帧取最新画面也绝不会丢掉 50ms 触桌脉冲。
    """

    def __init__(self, base_url: str, status_every: int = 5, jpg_q: int = 60, t0_ref: float | None = None) -> None:
        import threading
        import urllib.parse
        import requests
        self._sess = requests.Session()
        self._sess.trust_env = False
        self.base = base_url.rstrip("/")
        host = urllib.parse.urlparse(self.base).hostname or "192.168.137.50"
        self.udp = UdpTelemetryReceiver(host=host, port=9001)
        self.knock_stream: CameraRateKnockStream | None = None
        self.t0_ref = time.perf_counter() if t0_ref is None else float(t0_ref)
        self.status_every = max(1, int(status_every))
        self.jpg_q = int(jpg_q)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, RobotTelemetry, float, int] | None = None
        self._http_tel = RobotTelemetry()
        self._consumed = -1
        self._stop = False
        self._th = threading.Thread(target=self._loop, name="live-fetch", daemon=True)
        self._th.start()

    def _loop(self) -> None:
        i = 0
        cap_url = f"{self.base}/capture?q={self.jpg_q}"
        while not self._stop:
            try:
                # 仅在 UDP 流尚未建立时才降频轮询 /status，释放 ESP32 单线程 httpd 给相机抓取
                if self.udp.latest(max_age_s=0.35) is None and (i % self.status_every == 0):
                    try:
                        rs = self._sess.get(f"{self.base}/status", timeout=1.2)
                        self._http_tel = RobotTelemetry.from_status_json(rs.json())
                    except Exception:
                        pass
                t_req = time.perf_counter()
                r = self._sess.get(cap_url, timeout=2.0)
                t_done = time.perf_counter()
                # GC0308 读出+编码约占往返前段：取请求发出后 8ms 作为传感器曝光近似时刻
                t_exp = min(t_done, t_req + 0.008)
                tel_cap = self.udp.at_time(t_exp, max_age_s=0.5) or self._http_tel
                bgr = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    ts_rel = t_done - self.t0_ref
                    ks = self.knock_stream
                    if ks is not None:
                        ks.push_frame(bgr, ts_rel, tel_cap)
                    with self._lock:
                        self._latest = (bgr, tel_cap, ts_rel, i)
                i += 1
            except Exception:
                time.sleep(0.05)

    def get(self, wait_s: float = 0.2) -> tuple[np.ndarray | None, RobotTelemetry | None, RobotTelemetry | None, float | None]:
        """取最新一帧 (bgr, tel_at_capture, tel_latest_now, ts_s)。"""
        deadline = time.perf_counter() + wait_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest is not None and self._latest[3] != self._consumed:
                    self._consumed = self._latest[3]
                    bgr, tel_cap, ts_rel, _ = self._latest
                    tel_now = self.udp.latest(max_age_s=0.35) or tel_cap
                    return bgr, tel_cap, tel_now, ts_rel
            time.sleep(0.002)
        return None, None, None, None

    def close(self) -> None:
        self._stop = True
        self.udp.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG-Hover PC Unified Embodied Brain (100% Own Models)")
    ap.add_argument("--scenario", type=int, default=2, choices=[1, 2])
    ap.add_argument("--source", type=str, default="frames", choices=["frames", "video", "live"])
    ap.add_argument("--input", type=str, default=str(REPO / "evidence" / "sessions" / "none_d0.5_front_night_181841" / "frames"))
    ap.add_argument("--base", type=str, default="http://192.168.137.50")
    ap.add_argument("--send-cmd", action="store_true", help="Live mode only: actually send commands to robot")
    ap.add_argument("--lite", action="store_true", help="Use 6.5MB MBV3 model instead of ViT-240")
    ap.add_argument("--headless", action="store_true", help="No GUI window (batch/test mode)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="默认跑完全部帧（video/live 到 EOF、frames 到目录尾）；仅显式传值时截断。")
    ap.add_argument("--click-xy", type=str, default="", help="Optional auto-click target 'x,y' in meters for headless Scenario 2")
    ap.add_argument("--emote", action="store_true", help="Live mode only: mirror brain state onto the robot LCD via POST /emote")
    ap.add_argument("--static-decim", type=int, default=3,
                    help="静态建图降频：每 N 帧才做 2mm 融合/鬼影雕刻；1=不降频保最高图质（离线跑图用）")
    ap.add_argument("--reloc-decim", type=int, default=2,
                    help="视觉重定位降频：APPROACH 时每 N 帧跑一次")
    ap.add_argument("--jpg-q", type=int, default=60,
                    help="Live 模式 /capture?q= 的 JPEG 质量")
    ap.add_argument("--dump-dir", type=str, default="",
                    help="逐帧落盘：bgr240 原图 + info 的 jsonl")
    ap.add_argument("--dump-every", type=int, default=3, help="每 N 帧存一张图")
    ap.add_argument("--final-eval", action="store_true", help="authorise one registered replay on sealed input")
    args = ap.parse_args()

    if args.source in ("frames", "video"):
        sealed.require_final_eval([args.input], purpose="pc_unified_brain_app replay input")

    brain = UnifiedRobotBrain(use_lite_geonet=args.lite, enable_knock=(args.scenario == 1),
                              static_decim=args.static_decim, reloc_decim=args.reloc_decim)
    print(f"UnifiedRobotBrain READY | geonet={brain.geonet_ckpt.name} | scenario={args.scenario} | source={args.source}", flush=True)

    frames_iter: list[np.ndarray] = []
    cap = None
    video_fps = 15.0
    if args.source == "frames":
        p = pathlib.Path(args.input)
        jpgs = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        lim = args.max_frames if args.max_frames is not None else len(jpgs)
        frames_iter = [cv2.imread(str(j)) for j in jpgs[:lim] if cv2.imread(str(j)) is not None]
    elif args.source == "video":
        cap = cv2.VideoCapture(args.input)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0) or 15.0

    cam = OrbitCamera3D(azimuth_deg=-150.0, elevation_deg=34.0, distance_m=0.88, target_x=0.44, target_y=0.0, target_z=0.06)
    scenario = args.scenario
    drag = {"on": False, "lx": 0, "ly": 0}
    win_name = "RIG-Hover PC Unified Brain (100% Own Models)"
    W3D, H3D = 760, 520

    if not args.headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        def on_mouse(ev, x, y, flags, _param):
            if ev == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY) and x < W3D:
                basis = _PROJ.compute_camera_basis(cam)
                focal = _PROJ.compute_focal_px(cam, float(H3D))
                nx = (x - W3D * 0.5) / focal
                ny = (H3D * 0.5 - y) / focal
                dx = basis[3] + nx * basis[6] + ny * basis[9]
                dy = basis[4] + nx * basis[7] + ny * basis[10]
                dz = basis[5] + nx * basis[8] + ny * basis[11]
                if dz < -1e-4:
                    t = -basis[2] / dz
                    if t > 0:
                        wx, wy = float(basis[0] + t * dx), float(basis[1] + t * dy)
                        res = brain.set_3d_target(wx, wy)
                        print(f"[CLICK TARGET] {res}", flush=True)
            elif ev == cv2.EVENT_LBUTTONDOWN:
                drag.update(on=True, lx=x, ly=y)
            elif ev == cv2.EVENT_LBUTTONUP:
                drag["on"] = False
            elif ev == cv2.EVENT_MOUSEMOVE and drag["on"]:
                cam.azimuth_deg -= (x - drag["lx"]) * 0.4
                cam.elevation_deg = float(np.clip(cam.elevation_deg + (y - drag["ly"]) * 0.3, 6.0, 86.0))
                drag.update(lx=x, ly=y)
            elif ev == cv2.EVENT_MOUSEWHEEL:
                cam.distance_m = float(np.clip(cam.distance_m * (0.90 if flags > 0 else 1.11), 0.22, 3.0))
        cv2.setMouseCallback(win_name, on_mouse)

    t0 = time.perf_counter()
    auto_clicked = False
    bridge = None
    if (args.emote or args.send_cmd) and args.source == "live":
        from emote_bridge import EmoteBridge
        bridge = EmoteBridge(args.base, verbose=args.emote)
    prev_mode = ""
    fetcher = _LiveFrameFetcher(args.base, jpg_q=args.jpg_q, t0_ref=t0) if args.source == "live" else None
    knock_stream = None
    if fetcher is not None and brain.knock_det is not None:
        knock_stream = CameraRateKnockStream(brain.knock_det)
        fetcher.knock_stream = knock_stream
    actuator = FirmwareActuatorClient(args.base) if (args.source == "live" and args.send_cmd) else None

    dump_dir = None
    dump_jsonl = None
    if args.dump_dir:
        dump_dir = pathlib.Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl = open(dump_dir / "info.jsonl", "w", encoding="utf-8")

    idx = 0
    wait_logged_t = 0.0
    try:
        while args.max_frames is None or idx < args.max_frames:
            tel_now = None
            ext_knock = None
            if args.source == "frames":
                if idx >= len(frames_iter):
                    break
                bgr, tel = frames_iter[idx], None
                ts = idx / 15.0
            elif args.source == "video":
                ok, bgr = cap.read()
                if not ok:
                    for _r in range(5):
                        time.sleep(0.05)
                        ok, bgr = cap.read()
                        if ok:
                            break
                    if not ok:
                        print(f"[video] frame {idx} read failed after retries -> treating as EOF", flush=True)
                        break
                tel = None
                ts = idx / video_fps
            else:
                bgr, tel, tel_now, ts_cap = fetcher.get()
                if bgr is None:
                    now_w = time.perf_counter()
                    if now_w - wait_logged_t >= 5.0:
                        print(f"[live] waiting for robot stream at {args.base} ...", flush=True)
                        wait_logged_t = now_w
                    continue
                ts = float(ts_cap) if ts_cap is not None else (time.perf_counter() - t0)
                if knock_stream is not None:
                    ext_knock = knock_stream.consume_latest()

            if args.click_xy and (not auto_clicked) and idx >= 4:
                cx_s, cy_s = [float(v) for v in args.click_xy.split(",")]
                print("[AUTO-CLICK]", brain.set_3d_target(cx_s, cy_s), flush=True)
                auto_clicked = True

            info = brain.step(
                bgr, telemetry=tel, scenario=scenario, timestamp_s=ts, auto_level=(tel is None),
                tel_now=tel_now, ext_knock=ext_knock,
            )
            if knock_stream is not None:
                knock_stream.set_context(brain._last_wrist_box, brain.mode)

            if dump_dir is not None:
                if idx % args.dump_every == 0:
                    cv2.imwrite(str(dump_dir / f"f{idx:05d}.jpg"), bgr)
                dump_jsonl.write(json.dumps(info, ensure_ascii=False) + "\n")
                if idx % 10 == 0:
                    dump_jsonl.flush()

            if bridge is not None:
                if info["knock"]["fired"]:
                    bridge.event("knock")
                elif info.get("cliff_fired"):
                    bridge.event("cliff")
                elif info["mode"] == "APPROACH" and prev_mode != "APPROACH":
                    bridge.event("lock")
                else:
                    bridge.update_mode(info["mode"])
                prev_mode = info["mode"]

            if actuator is not None:
                c = info["cmd"]
                rstate = int((tel_now or tel).robot_state) if (tel_now or tel) is not None else 1
                cliff_stop = bool(info.get("cliff_engaged") and info["mode"] == "APPROACH" and c["vx"] <= 0.01)
                actuator.send_command(
                    DriveCommand(vx=c["vx"], vyaw=c["vyaw"], head=c["head"]),
                    robot_state=rstate, mode_id=int(brain.mode), cliff_danger=cliff_stop,
                    yaw_gyro_bias_dps=float(info.get("yaw_bias_dps", 0.0)),
                )

            if idx % 10 == 0 or info["knock"]["fired"] or (args.max_frames is not None and idx == args.max_frames - 1):
                print(json.dumps(info, ensure_ascii=False), flush=True)

            if not args.headless:
                dash = brain.render_dashboard(bgr, info, cam, w=W3D, h=H3D)
                cv2.imshow(win_name, dash)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("1"):
                    scenario = 1
                elif key == ord("2"):
                    scenario = 2
            idx += 1
    finally:
        if actuator is not None:
            actuator.close()
        if knock_stream is not None:
            knock_stream.close()
        if fetcher is not None:
            fetcher.close()
        if dump_jsonl is not None:
            dump_jsonl.close()
        if cap is not None:
            cap.release()
        if bridge is not None:
            print(f"[emote] sent={bridge.sent} failed={bridge.failed}", flush=True)
            bridge.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
