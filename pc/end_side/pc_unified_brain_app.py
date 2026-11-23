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
  python pc_unified_brain_app.py --scenario 2 --source frames --input D:\\robot_work\\evidence\\sessions\\empty_d0.5_front_night_182201\\frames

  # 2) 离线回放场景一（真实双敲录像 cb8455，headless 输出状态与 3D 敲击落针日志）
  python pc_unified_brain_app.py --scenario 1 --source video --input D:\\robot_work\\dataset\\takes\\robot_20261115_151733_cb8455\\take.mp4 --headless --max-frames 120

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
for p in (HERE, REPO / "pc" / "train", REPO / "pc" / "perception"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from end_side_controller import (  # noqa: E402
    EndSideFeatureBuilder,
    EndSidePolicyNet,
    EndSideSafetyShield,
)
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from hd_dynamic_contour_slam import (  # noqa: E402
    HdDynamicContourSlamEngine,
    HoverGeoNet240Frontend,
    render_hd_dual_layer_green,
)
from cliff_detect import detect_visual_cliff_zero_oracle  # noqa: E402
from hd_moge_contour_engine import REAL_FOV_H_DEG, REAL_H_CAM_M  # noqa: E402
from hover_geonet_240 import GEO_H, GEO_W  # noqa: E402
from mono_depth_model import make_camera_ray_grid  # noqa: E402
from neural_gaussian_splat_model import OrbitCamera3D, PersistentGaussianMemoryMap  # noqa: E402
from scenario1_own_models_e2e import (  # noqa: E402
    POLICY_NAV_CKPT,
    POLICY_S1_CKPT,
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
        self.geo_fe = HoverGeoNet240Frontend(ckpt, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
        self.slam = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=REAL_H_CAM_M, obs_radius_m=0.095,
                                               static_decim=static_decim, async_static=True)
        self.policy_s1: EndSidePolicyNet = load_rl_policy(POLICY_S1_CKPT)
        self.policy_nav: EndSidePolicyNet = load_rl_policy(POLICY_NAV_CKPT)
        self.shield = EndSideSafetyShield()
        self.reloc_decim = max(1, int(reloc_decim))
        # 感知线程池：yunet/双敲只依赖原始帧，全程挂在 GeoNet+SLAM 底下并行
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="percep")

        self.yunet = (
            cv2.FaceDetectorYN.create(str(YUNET_ONNX), "", (240, 240), score_threshold=0.45)
            if YUNET_ONNX.exists()
            else None
        )

        self.knock_det: DoubleKnockDetector | None = None
        if enable_knock and KNOCK_DEPLOY_CKPT.exists():
            # 双敲网络极小 (hidden=64,size=96)：固定 CPU，把 GPU 完全让给 GeoNet，
            # 且能真正与 GeoNet+SLAM 线程级并行（GPU 版只会排队等内核）。
            dev = "cpu"
            kb = torch.load(KNOCK_DEPLOY_CKPT, map_location=dev, weights_only=False)
            km = DoubleKnockNet(in_ch=2, kin_dim=8, hidden_dim=64, num_classes=4).to(dev).eval()
            km.load_state_dict(kb["state_dict"])
            self.knock_det = DoubleKnockDetector(
                model=km,
                device=dev,
                window=int(kb.get("window", 48)),
                size=int(kb.get("size", 96)),
                imp_threshold=0.35,
                double_threshold=0.32,
                gate_state=MODE_GAZE_FIST,
                sem_latch_s=2.0,
                single_confirm_thr=0.60,
                single_lookback_s=1.5,
            )

        self.relocalizer: Visual3DLandmarkRelocalizer | None = None
        self.prev_bgr: np.ndarray | None = None
        self.mode: int = MODE_IDLE
        self.est_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.target_xy: tuple[float, float] | None = None
        self.path_xy: list[tuple[float, float]] = []
        self.frames_seen: int = 0
        self.dwell_count: int = 0
        self.last_pred: dict[str, Any] | None = None
        # 视觉悬崖本能（与 living_desktop_brain_v15 共用的零开挂检测器）：
        # APPROACH 巡航中 dist_to_rim_m < 0.30 时通过 Shield clearance=0 硬刹停 vx。
        self.rays_cam = make_camera_ray_grid(GEO_H, GEO_W, fov_h_deg=REAL_FOV_H_DEG, pitch_deg=0.0, roll_deg=0.0)
        self.cliff_hold_m: float = 0.30
        self.cliff_cone_deg: float = 45.0
        self._cliff_active: bool = False

    def _ensure_relocalizer(self, bgr: np.ndarray, pred: dict[str, Any]) -> None:
        st_pts, st_nrms, _ = self.slam.get_confirmed_static_cloud(min_wt=1.0)
        if len(st_pts) < 1000:
            return
        if self.relocalizer is None:
            self.relocalizer = Visual3DLandmarkRelocalizer(st_pts, st_nrms)
            self.relocalizer.add_keyframe(bgr, pred["pts_grid"], pred["valid_grid"], self.est_pose)
        else:
            self.relocalizer.static_pts = st_pts
            self.relocalizer.static_nrms = st_nrms
            if len(self.relocalizer.keyframes) < 5 and self.frames_seen % 6 == 0:
                self.relocalizer.add_keyframe(bgr, pred["pts_grid"], pred["valid_grid"], self.est_pose)

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
        return {"target_xy": self.target_xy, "astar_accepted": bool(plan["accepted"]), "path_pts": len(self.path_xy)}

    def _knock_step_full(self, bgr240: np.ndarray, timestamp_s: float | None) -> dict[str, Any]:
        g240 = cv2.cvtColor(bgr240, cv2.COLOR_BGR2GRAY)
        return self.knock_det.step(g240, wrist_box=None, scenario_state=None, timestamp=timestamp_s)

    def step(
        self,
        bgr: np.ndarray,
        telemetry: RobotTelemetry | None = None,
        scenario: int = 2,
        timestamp_s: float | None = None,
        auto_level: bool = True,
    ) -> dict[str, Any]:
        """单帧统一推理：自研 HoverGeoNet-ViT-240 + Dynablox + 视觉-3D 重定位 + 场景一/二 RL 决策。"""
        self.frames_seen += 1
        tel = telemetry or RobotTelemetry(pitch=3.0, roll=0.0, yaw=self.est_pose[2])
        # IMU 逐帧传播 est_pose.yaw：重定位每 2 帧才纠一次，冻结期间机器人真在转，
        # 方位角来回跳曾致 vyaw 甩鞭原地打转（真机 f171-300 实证）
        if getattr(self, "_last_tel_yaw", None) is not None:
            self.est_pose = (
                self.est_pose[0], self.est_pose[1],
                _wrap_deg(self.est_pose[2] + _wrap_deg(tel.yaw - self._last_tel_yaw)),
            )
        self._last_tel_yaw = tel.yaw
        bgr240 = cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA) if bgr.shape[:2] != (240, 240) else bgr
        _t = time.perf_counter
        _st: dict[str, float] = {}
        _t0 = _t()

        # 0) 并行感知：yunet/双敲只依赖原始帧——主链跑 GeoNet+SLAM 期间它们在 CPU 线程上完成
        fut_face = None
        fut_knock = None
        if scenario == 1:
            if self.yunet is not None:
                fut_face = self._pool.submit(self.yunet.detect, bgr240)
            if self.knock_det is not None:
                fut_knock = self._pool.submit(self._knock_step_full, bgr240, timestamp_s)

        # 1) 自研 HoverGeoNet-ViT-240 推理 + 2mm Dynablox 双层建图
        # cam_cloud 只被重定位消费：跳帧时连相机系点云都不解码（省 ~5-8ms）
        reloc_due = (self.mode == MODE_APPROACH and self.relocalizer is not None
                     and self.frames_seen >= 3 and self.frames_seen % self.reloc_decim == 0)
        pred = self.geo_fe.infer_frame(bgr240, self.prev_bgr, pitch_deg=tel.pitch, roll_deg=tel.roll,
                                       auto_level=auto_level, cam_cloud=reloc_due)
        _st["geonet"] = (_t() - _t0) * 1e3
        self.last_pred = pred
        _t0 = _t()
        slam_rec = self.slam.process_frame(
            bgr240, pred["pts_grid"], pred["nrm_grid"], pred["crease_grid"], pred["valid_grid"],
            robot_x=self.est_pose[0], robot_y=self.est_pose[1], robot_yaw_deg=self.est_pose[2],
            pitch_deg=tel.pitch, roll_deg=tel.roll, align=(self.frames_seen > 1),
        )
        _st["slam"] = (_t() - _t0) * 1e3
        if reloc_due or self.frames_seen % 6 == 0:
            self._ensure_relocalizer(bgr240, pred)
        self.prev_bgr = bgr240.copy()

        # 1.5) 视觉悬崖本能：纯视觉深渊检测（与 v15 仿真同一检测器），近崖 APPROACH 经 Shield clearance=0 硬刹停
        _t0 = _t()
        cliff_det = detect_visual_cliff_zero_oracle(
            bgr240, self.prev_bgr, self.geo_fe,
            (self.est_pose[0], self.est_pose[1], self.est_pose[2], tel.pitch, tel.roll),
            self.rays_cam, pred=pred,
        )
        _st["cliff"] = (_t() - _t0) * 1e3
        cliff_engaged, cliff_clear = _cliff_guard(cliff_det, self.est_pose, self.cliff_hold_m, self.cliff_cone_deg)
        # 迟滞边沿：rim 在 hold 线附近抖动时只报一次警（进 <hold_m、出 >hold_m+3cm）
        if cliff_det is None:
            cliff_near = False
        elif self._cliff_active:
            cliff_near = cliff_det["dist_to_rim_m"] < self.cliff_hold_m + 0.03
        else:
            cliff_near = cliff_det["dist_to_rim_m"] < self.cliff_hold_m
        cliff_fired = cliff_near and not self._cliff_active
        self._cliff_active = cliff_near

        # 2) 若处于导航巡航中 (MODE_APPROACH)，运行 2D-3D PnP + KISS-ICP 无里程计视觉重定位
        reloc_info: dict[str, Any] = {"source": "map_build", "ransac_inliers": 0}
        _t0 = _t()
        if reloc_due:
            # 就在重定位前刷最新静态图（原先每帧 O(全图) 扫描是隐藏的关键路径大头）
            self.relocalizer.static_pts, self.relocalizer.static_nrms, _ = self.slam.get_confirmed_static_cloud(min_wt=1.0)
            ro = self.relocalizer.relocalize(
                bgr240, pred["pts_cam"], pred["nrm_cam"], pred["valid_grid"],
                imu_pitch_deg=tel.pitch, imu_roll_deg=tel.roll, prev_est_pose=self.est_pose,
            )
            al = 0.75
            self.est_pose = (
                (1.0 - al) * self.est_pose[0] + al * ro["x"],
                (1.0 - al) * self.est_pose[1] + al * ro["y"],
                _wrap_deg(self.est_pose[2] + al * _wrap_deg(ro["yaw_deg"] - self.est_pose[2])),
            )
            reloc_info = {"source": ro["source"], "ransac_inliers": ro["ransac_inliers"], "icp_res_mm": ro["icp_median_res_mm"]}
        _st["reloc"] = (_t() - _t0) * 1e3

        # 3) 收集并行感知结果（此时它们早已在 GeoNet+SLAM 底下跑完，等待≈0）
        face_dets: list[dict[str, float]] = []
        knock_out: dict[str, Any] = {"fired": 0, "p_double": 0.0}
        if fut_face is not None or fut_knock is not None:
            _t0 = _t()
            if fut_face is not None:
                _, faces = fut_face.result()
                if faces is not None and len(faces) > 0:
                    fx, fy, fw, fh = faces[0][:4]
                    face_dets.append({"cx": float((fx + 0.5 * fw) / 240.0), "cy": float((fy + 0.5 * fh) / 240.0), "w": float(fw / 240.0), "h": float(fh / 240.0), "score": float(faces[0][-1])})
                    if self.mode == MODE_IDLE:
                        self.mode = MODE_GAZE_FACE
            _st["yunet"] = (_t() - _t0) * 1e3
            _t0 = _t()
            if fut_knock is not None:
                knock_out = fut_knock.result()
                if knock_out.get("fired", 0):
                    # 敲击点=贴桌面的动态簇质心：dyn_pts 混着手+身体+脸，中位数会把点拉到桌外
                    # （真机实证：敲在桌前 0.25m 被算成 0.69m 桌外）。取最近 35% + |z|<0.06m 的点。
                    dp = self.slam.dyn_pts
                    if len(dp) >= 20:
                        near = dp[dp[:, 0] <= np.quantile(dp[:, 0], 0.35)]
                        near = near[np.abs(near[:, 2]) < 0.06] if len(near) else dp
                        k_med = np.median(near if len(near) >= 8 else dp, axis=0)
                        self.set_3d_target(float(np.clip(k_med[0], 0.15, 0.60)), float(np.clip(k_med[1], -0.35, 0.35)))
                    else:
                        self.set_3d_target(0.35, 0.0)
            _st["knock"] = (_t() - _t0) * 1e3

        # 3.5) 拳头注视：抬离桌面的紧凑动态簇 = 悬空拳头（统一大脑此前没有拳头检测器，
        # GAZE_FIST 是进不去的死代码——真机"不认拳"的直接原因；SLAM 动态分离现成）
        fist_det = None
        dp = self.slam.dyn_pts
        if len(dp) >= 40:
            elev = dp[(dp[:, 2] > 0.06) & (dp[:, 0] < 0.95)]
            if len(elev) >= 25:
                _c = np.median(elev, axis=0)
                spread = float(np.median(np.linalg.norm(elev[:, :2] - _c[:2], axis=1)))
                if spread < 0.25:
                    fist_det = {"bearing_deg": math.degrees(math.atan2(-_c[1], _c[0])), "n": int(len(elev))}
        if fist_det is not None and self.mode in (MODE_IDLE, MODE_GAZE_FACE):
            self.mode = MODE_GAZE_FIST
            self._fist_lost = 0
        elif self.mode == MODE_GAZE_FIST:
            if fist_det is None:
                self._fist_lost = getattr(self, "_fist_lost", 0) + 1
                if self._fist_lost > 30:
                    self.mode = MODE_GAZE_FACE if face_dets else MODE_IDLE
            else:
                self._fist_lost = 0

        # 4) 自研 EndSidePolicyNet + EndSideSafetyShield 输出最终控制指令
        _t0 = _t()
        rem_m = 0.0
        if self.mode == MODE_APPROACH and self.target_xy is not None:
            ex, ey, eyaw = self.est_pose
            rem_m = math.hypot(self.target_xy[0] - ex, self.target_xy[1] - ey)
            # 目标不可达放弃机制：悬崖挡道/距离不降时，原地扭摆毫无意义。
            # 150 帧 (~15s) 内 rem 没改善 20% -> 放弃，回盯人脸（"那儿去不了"的可读表达）
            if rem_m < getattr(self, "_best_rem", 1e9):
                self._best_rem = rem_m
                self._approach_stall = 0
            else:
                self._approach_stall = getattr(self, "_approach_stall", 0) + 1
            if self._approach_stall > 150 and rem_m > getattr(self, "_best_rem", 0) * 0.99:
                self.target_xy = None
                self._best_rem = 1e9
                self._approach_stall = 0
                self.mode = MODE_DWELL if scenario == 1 else MODE_IDLE
                self.dwell_count = 0
            if rem_m < 0.032:
                self.mode = MODE_DWELL if scenario == 1 else MODE_IDLE
                self.dwell_count = 0
                safe_cmd = self.shield.apply(DriveCommand(0.0, 0.0, 0.0), mode=self.mode, remaining_m=0.0, pitch_deg=tel.pitch)
            else:
                wp = self.target_xy
                if len(self.path_xy) >= 2:
                    di = int(np.argmin([math.hypot(px - ex, py - ey) for px, py in self.path_xy]))
                    for j in range(di, len(self.path_xy)):
                        if math.hypot(self.path_xy[j][0] - ex, self.path_xy[j][1] - ey) >= 0.13:
                            wp = self.path_xy[j]
                            break
                bear = _wrap_deg(math.degrees(math.atan2(wp[1] - ey, wp[0] - ex)) - eyaw)
                tlock = TargetLock(valid=1, bearing_body=bear, remaining_m=rem_m)
                f39 = EndSideFeatureBuilder.build(tel, detections=face_dets, target_lock=tlock, clearance_m=cliff_clear)
                raw_cmd = self.policy_nav.infer_numpy(f39)
                safe_cmd = self.shield.apply(raw_cmd, mode=MODE_APPROACH, remaining_m=rem_m, pitch_deg=tel.pitch, clearance_m=cliff_clear)
        elif self.mode == MODE_DWELL:
            self.dwell_count += 1
            safe_cmd = self.shield.apply(DriveCommand(0.0, 0.0, 0.0), mode=MODE_DWELL, remaining_m=0.0, pitch_deg=tel.pitch)
            if self.dwell_count >= 12:
                self.mode = MODE_GAZE_FACE
        else:
            f39 = EndSideFeatureBuilder.build(tel, detections=face_dets, target_lock=None, clearance_m=1.0)
            raw_cmd = self.policy_s1.infer_numpy(f39)
            safe_cmd = self.shield.apply(raw_cmd, mode=self.mode, remaining_m=0.0, pitch_deg=tel.pitch)
            # 盯脸可读性：policy 输出 ±1-5° 人耳听不到舵机声。检测到脸就直接按
            # 方位角控舵机（60°FOV，正=左转），转幅一眼能看见
            # 手持态：满天 void + 全画面动态 = 被人拿在手里，轮子空转无意义 → 锁轮只做表情
            held = (len(dp) > 25000) and (cliff_det is not None and cliff_det.get("void_area_px", 0) > 6000)
            if held:
                safe_cmd = DriveCommand(0.0, 0.0, safe_cmd.head)

            if self.mode == MODE_GAZE_FIST and fist_det:
                # 盯拳：身体转向拳方位（头是表情件；vyaw>0=左）。增量小步，防甩头
                vyaw_g = float(np.clip(fist_det["bearing_deg"] * 0.6, -25.0, 25.0))
                head_g = float(np.clip(fist_det["bearing_deg"] * 2.0, -45.0, 45.0))
                safe_cmd = DriveCommand(0.0, vyaw_g, head_g)
            elif self.mode == MODE_GAZE_FACE and face_dets:
                # 舵机云台不带相机——居中脸的职责在身体 vyaw。P 环 140 增益在 10Hz+
                # 轮上惯量下欠阻尼来回甩（真机实证）→ PD 环：P 收敛居中 + D 阻尼，
                # cx 三帧中值消框抖，死区放宽到 0.06。
                self._cx_hist = (getattr(self, "_cx_hist", []) + [face_dets[0]["cx"]])[-3:]
                cx_f = float(np.median(self._cx_hist))
                e = cx_f - 0.5
                de = e - getattr(self, "_cx_prev", e)
                self._cx_prev = e
                vyaw_g = float(np.clip(70.0 * e + 200.0 * de, -25.0, 25.0))
                if abs(e) < 0.06 and abs(de) < 0.02:
                    vyaw_g = 0.0
                head_g = float(np.clip(-e * 160.0, -45.0, 45.0))  # 连杆比吃掉大半摆角，打满±45才看得见
                safe_cmd = DriveCommand(0.0 if held else safe_cmd.vx, vyaw_g if not held else 0.0, head_g)
        _st["policy"] = (_t() - _t0) * 1e3

        # 找人本能：无脸 >6s 就缓速旋转扫描（真机曾转到面壁全程 IDLE——它不是在不理你，
        # 是根本不在看你）。舵机扫 ±55°，身体 16°/s 慢转，见脸立即停。
        self._no_face = 0 if face_dets else getattr(self, "_no_face", 0) + 1
        if self.mode in (MODE_IDLE, MODE_GAZE_FACE) and self._no_face > 60:
            sweep = 55.0 * math.sin(self._no_face * 0.04)
            safe_cmd = DriveCommand(0.0, 16.0, float(sweep))

        # 出口低通：10Hz 更新率下符号拍动就是肉眼可见的前后猛抖（真机实证）。
        # 悬崖硬停（clearance<=0）旁路不平滑——刹停必须是即时的。
        if cliff_clear is not None and cliff_clear <= 0.0:
            self._cmd_f = (safe_cmd.vx, safe_cmd.vyaw, safe_cmd.head)
        else:
            _p = getattr(self, "_cmd_f", None) or (safe_cmd.vx, safe_cmd.vyaw, safe_cmd.head)
            safe_cmd = DriveCommand(
                0.55 * _p[0] + 0.45 * safe_cmd.vx,
                0.55 * _p[1] + 0.45 * safe_cmd.vyaw,
                0.15 * _p[2] + 0.85 * safe_cmd.head,  # head 通道轻平滑要响应快
            )
            self._cmd_f = (safe_cmd.vx, safe_cmd.vyaw, safe_cmd.head)
        # vyaw 死区：±3 以下一律 0——往复摆动若有非零均值会积成净转身（真机实证"前后摆变侧对"）
        if abs(safe_cmd.vyaw) < 3.0:
            safe_cmd = DriveCommand(safe_cmd.vx, 0.0, safe_cmd.head)

        return {
            "frame": self.frames_seen,
            "mode": MODE_NAMES.get(self.mode, str(self.mode)),
            "geonet_ms": round(float(pred["infer_ms"]), 2),
            "stage_ms": {k: round(v, 2) for k, v in _st.items()},
            "slam_stages": slam_rec.get("slam_stage_ms"),
            "static_pts": slam_rec["n_static_map"],
            "dyn_pts": slam_rec["n_dynamic_frame"],
            "est_pose": [round(v, 3) for v in self.est_pose],
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
        side[20:260, 30:270] = cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA)
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
            f"RL Cmd: vx={step_info['cmd']['vx']:.1f}",
            f"        vyaw={step_info['cmd']['vyaw']:.1f} h={step_info['cmd']['head']:.1f}",
            "Keys: Shift+Click=Target | 1/2=Scen",
            "      Drag=Orbit | q=Quit",
        ]
        for i, ln in enumerate(hud_lines):
            cv2.putText(side, ln, (14, 286 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 250, 190), 1, cv2.LINE_AA)
        return np.hstack([p3d, side])


def _fetch_live_robot(base_url: str) -> tuple[np.ndarray | None, RobotTelemetry]:
    import urllib.request
    tel = RobotTelemetry()
    try:
        with urllib.request.urlopen(f"{base_url}/status", timeout=1.5) as r:
            d = json.loads(r.read().decode("utf-8", errors="ignore"))
            tel = RobotTelemetry.from_status_json(d)
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"{base_url}/capture", timeout=2.0) as r:
            arr = np.frombuffer(r.read(), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return bgr, tel
    except Exception:
        return None, tel


class _LiveFrameFetcher:
    """后台取帧线程：/capture + /status 与主计算完全并行（传输链从关键路径摘除）。

    第一性原理：相机帧周期 ~32ms 是地板，串行结构 "fetch(45-70ms) -> compute(~110ms)
    -> cmd(~10ms)" 把抓取延迟全数加进每帧；改成抓取线程持续拉流、主循环取
    **最新**帧（丢弃中间旧帧而非排队）—— 控制用的是最新鲜的一帧而不是下一帧。
    /status 在同线程每 K 帧捎带一次（遥测慢变，无需每帧往返）。
    """

    def __init__(self, base_url: str, status_every: int = 5, jpg_q: int = 60) -> None:
        import threading
        import requests
        self._sess = requests.Session()  # keep-alive：urllib 每次新建 TCP=握手+固件 socket 磨损
        self._sess.trust_env = False  # 机器人是局域网直连：系统 HTTP 代理会把每帧请求绕出去（实测拖慢到 ~6Hz）
        self.base = base_url.rstrip("/")
        self.status_every = max(1, int(status_every))
        self.jpg_q = int(jpg_q)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, RobotTelemetry, int] | None = None
        self._consumed = -1
        self._stop = False
        self._th = threading.Thread(target=self._loop, name="live-fetch", daemon=True)
        self._th.start()

    def _loop(self) -> None:
        tel = RobotTelemetry()
        i = 0
        cap_url = f"{self.base}/capture?q={self.jpg_q}"
        while not self._stop:
            try:
                if i % self.status_every == 0:
                    try:
                        r = self._sess.get(f"{self.base}/status", timeout=1.5)
                        tel = RobotTelemetry.from_status_json(json.loads(r.content.decode("utf-8", errors="ignore")))
                    except Exception:
                        pass
                r = self._sess.get(cap_url, timeout=2.0)
                bgr = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    with self._lock:
                        self._latest = (bgr, tel, i)
                i += 1
            except Exception:
                time.sleep(0.05)

    def get(self, wait_s: float = 0.2) -> tuple[np.ndarray | None, RobotTelemetry | None]:
        """取最新一帧；无新帧时短暂等待后返回 (None, None)。"""
        deadline = time.perf_counter() + wait_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest is not None and self._latest[2] != self._consumed:
                    self._consumed = self._latest[2]
                    return self._latest[0], self._latest[1]
            time.sleep(0.003)
        return None, None

    def close(self) -> None:
        self._stop = True


class _AsyncCmdSender:
    """异步指令发送：/cmd GET 不占主循环；只保最新（旧指令直接丢弃）。"""

    def __init__(self) -> None:
        import threading
        import requests
        self._sess = requests.Session()
        self._sess.trust_env = False  # 与 fetcher 同一原因：系统代理会吞掉发给机器人的指令
        self._cond = threading.Condition()
        self._pending: str | None = None
        self._stop = False
        self._th = threading.Thread(target=self._loop, name="cmd-send", daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.5)
                if self._stop:
                    return
                url, self._pending = self._pending, None
            try:
                self._sess.get(url, timeout=0.5)
            except Exception:
                pass

    def send(self, url: str) -> None:
        with self._cond:
            self._pending = url  # 覆盖式：只发最新指令
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG-Hover PC Unified Embodied Brain (100% Own Models)")
    ap.add_argument("--scenario", type=int, default=2, choices=[1, 2])
    ap.add_argument("--source", type=str, default="frames", choices=["frames", "video", "live"])
    ap.add_argument("--input", type=str, default=str(REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"))
    ap.add_argument("--base", type=str, default="http://192.168.137.50")
    ap.add_argument("--send-cmd", action="store_true", help="Live mode only: actually send /cmd to robot")
    ap.add_argument("--lite", action="store_true", help="Use 6.5MB MBV3 model instead of ViT-240")
    ap.add_argument("--headless", action="store_true", help="No GUI window (batch/test mode)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="默认跑完全部帧（video/live 到 EOF、frames 到目录尾）；仅显式传值时截断。原默认 60 曾致多次全量回放被静默截断。")
    ap.add_argument("--click-xy", type=str, default="", help="Optional auto-click target 'x,y' in meters for headless Scenario 2")
    ap.add_argument("--emote", action="store_true", help="Live mode only: mirror brain state onto the robot LCD via POST /emote")
    ap.add_argument("--static-decim", type=int, default=3,
                    help="静态建图降频：每 N 帧才做 2mm 融合/鬼影雕刻（实测该块占帧耗 ~85%）；1=不降频保最高图质（离线跑图用）")
    ap.add_argument("--reloc-decim", type=int, default=2,
                    help="视觉重定位降频：APPROACH 时每 N 帧跑一次（~50ms/次）；est_pose 0.75 平滑兜底间隔帧")
    ap.add_argument("--jpg-q", type=int, default=60,
                    help="Live 模式 /capture?q= 的 JPEG 质量：固件实测编码耗时随质量缩放（80→60 省 ~5ms 编码 + ~3ms 传输）")
    ap.add_argument("--dump-dir", type=str, default="",
                    help="逐帧落盘：bgr240 原图 + info 的 jsonl（失败现场可回放，如'为什么没识别到'）")
    ap.add_argument("--dump-every", type=int, default=3, help="每 N 帧存一张图")
    args = ap.parse_args()

    brain = UnifiedRobotBrain(use_lite_geonet=args.lite, enable_knock=(args.scenario == 1),
                              static_decim=args.static_decim, reloc_decim=args.reloc_decim)
    print(f"UnifiedRobotBrain READY | geonet={brain.geonet_ckpt.name} | scenario={args.scenario} | source={args.source}", flush=True)

    # 数据源迭代器
    frames_iter: list[np.ndarray] = []
    cap = None
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
                # 射线与桌面 Z=0 求交，点击 3D 点云锁定目标！
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
    if args.emote and args.source == "live":
        from emote_bridge import EmoteBridge
        bridge = EmoteBridge(args.base, verbose=True)
    prev_mode = ""
    fetcher = _LiveFrameFetcher(args.base, jpg_q=args.jpg_q) if args.source == "live" else None
    cmd_sender = _AsyncCmdSender() if (args.source == "live" and args.send_cmd) else None
    dump_dir = None
    dump_jsonl = None
    if args.dump_dir:
        dump_dir = pathlib.Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl = open(dump_dir / "info.jsonl", "w", encoding="utf-8")
    frame_ids = range(args.max_frames) if args.max_frames is not None else itertools.count()
    for idx in frame_ids:
        if args.source == "frames":
            if idx >= len(frames_iter):
                break
            bgr, tel = frames_iter[idx], None
        elif args.source == "video":
            ok, bgr = cap.read()
            if not ok:
                # 解码失败≠视频结束：内存压力下 ffmpeg 缓冲分配会偶发失败
                # （曾致三次回放在 ~f60 被静默截断）。重试 5 次仍失败才按 EOF 处理。
                for _r in range(5):
                    time.sleep(0.05)
                    ok, bgr = cap.read()
                    if ok:
                        break
                if not ok:
                    print(f"[video] frame {idx} read failed after retries -> treating as EOF", flush=True)
                    break
            tel = None
        else:
            bgr, tel = fetcher.get()
            if bgr is None:
                continue

        if args.click_xy and (not auto_clicked) and idx >= 4:
            cx_s, cy_s = [float(v) for v in args.click_xy.split(",")]
            print("[AUTO-CLICK]", brain.set_3d_target(cx_s, cy_s), flush=True)
            auto_clicked = True

        if args.source == "video":
            ts = idx / video_fps  # 内容时间轴：与 15fps 采集时间一致，而非解码墙钟
        else:
            ts = time.perf_counter() - t0
        info = brain.step(bgr, telemetry=tel, scenario=scenario, timestamp_s=ts, auto_level=(tel is None))

        if dump_dir is not None:
            if idx % args.dump_every == 0:
                cv2.imwrite(str(dump_dir / f"f{idx:05d}.jpg"), bgr)
            dump_jsonl.write(json.dumps(info, ensure_ascii=False) + "\n")

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

        if cmd_sender is not None:
            c = info["cmd"]
            cmd_sender.send(f"{args.base}/cmd?vx={c['vx']:.1f}&vyaw={c['vyaw']:.1f}&head={c['head']:.1f}")

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

    if cap is not None:
        cap.release()
    if bridge is not None:
        print(f"[emote] sent={bridge.sent} failed={bridge.failed}", flush=True)
        bridge.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
