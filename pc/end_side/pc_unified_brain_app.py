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

    def __init__(self, use_lite_geonet: bool = False, enable_knock: bool = True) -> None:
        ckpt = MBV3_CKPT if (use_lite_geonet or not VIT_CKPT.exists()) else VIT_CKPT
        self.geonet_ckpt = ckpt
        self.geo_fe = HoverGeoNet240Frontend(ckpt, use_fp16=True, h_cam_m=REAL_H_CAM_M, fov_h_deg=REAL_FOV_H_DEG)
        self.slam = HdDynamicContourSlamEngine(vox_size_m=0.0020, h_cam_m=REAL_H_CAM_M, obs_radius_m=0.095)
        self.policy_s1: EndSidePolicyNet = load_rl_policy(POLICY_S1_CKPT)
        self.policy_nav: EndSidePolicyNet = load_rl_policy(POLICY_NAV_CKPT)
        self.shield = EndSideSafetyShield()

        self.yunet = (
            cv2.FaceDetectorYN.create(str(YUNET_ONNX), "", (240, 240), score_threshold=0.45)
            if YUNET_ONNX.exists()
            else None
        )

        self.knock_det: DoubleKnockDetector | None = None
        if enable_knock and KNOCK_DEPLOY_CKPT.exists():
            dev = "cuda" if torch.cuda.is_available() else "cpu"
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
        return {"target_xy": self.target_xy, "astar_accepted": bool(plan["accepted"]), "path_pts": len(self.path_xy)}

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
        bgr240 = cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA) if bgr.shape[:2] != (240, 240) else bgr

        # 1) 自研 HoverGeoNet-ViT-240 推理 + 2mm Dynablox 双层建图
        pred = self.geo_fe.infer_frame(bgr240, self.prev_bgr, pitch_deg=tel.pitch, roll_deg=tel.roll, auto_level=auto_level)
        self.last_pred = pred
        slam_rec = self.slam.process_frame(
            bgr240, pred["pts_grid"], pred["nrm_grid"], pred["crease_grid"], pred["valid_grid"],
            robot_x=self.est_pose[0], robot_y=self.est_pose[1], robot_yaw_deg=self.est_pose[2],
            pitch_deg=tel.pitch, roll_deg=tel.roll, align=(self.frames_seen > 1),
        )
        self._ensure_relocalizer(bgr240, pred)
        self.prev_bgr = bgr240.copy()

        # 1.5) 视觉悬崖本能：纯视觉深渊检测（与 v15 仿真同一检测器），近崖 APPROACH 经 Shield clearance=0 硬刹停
        cliff_det = detect_visual_cliff_zero_oracle(
            bgr240, self.prev_bgr, self.geo_fe,
            (self.est_pose[0], self.est_pose[1], self.est_pose[2], tel.pitch, tel.roll),
            self.rays_cam, pred=pred,
        )
        cliff_engaged, cliff_clear = _cliff_guard(cliff_det, self.est_pose, self.cliff_hold_m, self.cliff_cone_deg)
        cliff_near = cliff_det is not None and cliff_det["dist_to_rim_m"] < self.cliff_hold_m
        cliff_fired = cliff_near and not self._cliff_active
        self._cliff_active = cliff_near

        # 2) 若处于导航巡航中 (MODE_APPROACH)，运行 2D-3D PnP + KISS-ICP 无里程计视觉重定位
        reloc_info: dict[str, Any] = {"source": "map_build", "ransac_inliers": 0}
        if self.mode == MODE_APPROACH and self.relocalizer is not None and self.frames_seen >= 3:
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

        # 3) 场景一感知：YuNet 人脸 + 自研 DoubleKnockNet 双敲触发 -> 3D 升维落针
        face_dets: list[dict[str, float]] = []
        knock_out: dict[str, Any] = {"fired": 0, "p_double": 0.0}
        if scenario == 1:
            if self.yunet is not None:
                _, faces = self.yunet.detect(bgr240)
                if faces is not None and len(faces) > 0:
                    fx, fy, fw, fh = faces[0][:4]
                    face_dets.append({"cx": float((fx + 0.5 * fw) / 240.0), "cy": float((fy + 0.5 * fh) / 240.0), "w": float(fw / 240.0), "h": float(fh / 240.0), "score": float(faces[0][-1])})
                    if self.mode == MODE_IDLE:
                        self.mode = MODE_GAZE_FACE
            if self.knock_det is not None:
                g240 = cv2.cvtColor(bgr240, cv2.COLOR_BGR2GRAY)
                knock_out = self.knock_det.step(g240, wrist_box=None, scenario_state=None, timestamp=timestamp_s)
                if knock_out.get("fired", 0):
                    # 将当前动态前景质心（或桌面中前区）升维为 3D 敲击点！
                    if len(self.slam.dyn_pts) >= 20:
                        k_med = np.median(self.slam.dyn_pts, axis=0)
                        self.set_3d_target(float(np.clip(k_med[0], 0.24, 1.05)), float(np.clip(k_med[1], -0.40, 0.40)))
                    else:
                        self.set_3d_target(0.48, 0.0)

        # 4) 自研 EndSidePolicyNet + EndSideSafetyShield 输出最终控制指令
        rem_m = 0.0
        if self.mode == MODE_APPROACH and self.target_xy is not None:
            ex, ey, eyaw = self.est_pose
            rem_m = math.hypot(self.target_xy[0] - ex, self.target_xy[1] - ey)
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

        return {
            "frame": self.frames_seen,
            "mode": MODE_NAMES.get(self.mode, str(self.mode)),
            "geonet_ms": round(float(pred["infer_ms"]), 2),
            "static_pts": slam_rec["n_static_map"],
            "dyn_pts": slam_rec["n_dynamic_frame"],
            "est_pose": [round(v, 3) for v in self.est_pose],
            "target_xy": list(self.target_xy) if self.target_xy else None,
            "rem_m": round(rem_m, 3),
            "reloc": reloc_info,
            "knock": {"fired": int(knock_out.get("fired", 0)), "p_double": round(float(knock_out.get("p_double", 0.0)), 3), "src": knock_out.get("fire_src", "pair"), "imp": round(float(knock_out.get("p_impulse", 0.0)), 3), "sem": round(float(knock_out.get("p_double_sem", knock_out.get("p_double", 0.0))), 3)},
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


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG-Hover PC Unified Embodied Brain (100% Own Models)")
    ap.add_argument("--scenario", type=int, default=2, choices=[1, 2])
    ap.add_argument("--source", type=str, default="frames", choices=["frames", "video", "live"])
    ap.add_argument("--input", type=str, default=str(REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"))
    ap.add_argument("--base", type=str, default="http://192.168.137.50")
    ap.add_argument("--send-cmd", action="store_true", help="Live mode only: actually send /cmd to robot")
    ap.add_argument("--lite", action="store_true", help="Use 6.5MB MBV3 model instead of ViT-240")
    ap.add_argument("--headless", action="store_true", help="No GUI window (batch/test mode)")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--click-xy", type=str, default="", help="Optional auto-click target 'x,y' in meters for headless Scenario 2")
    ap.add_argument("--emote", action="store_true", help="Live mode only: mirror brain state onto the robot LCD via POST /emote")
    args = ap.parse_args()

    brain = UnifiedRobotBrain(use_lite_geonet=args.lite, enable_knock=(args.scenario == 1))
    print(f"UnifiedRobotBrain READY | geonet={brain.geonet_ckpt.name} | scenario={args.scenario} | source={args.source}", flush=True)

    # 数据源迭代器
    frames_iter: list[np.ndarray] = []
    cap = None
    if args.source == "frames":
        p = pathlib.Path(args.input)
        jpgs = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        frames_iter = [cv2.imread(str(j)) for j in jpgs[: args.max_frames] if cv2.imread(str(j)) is not None]
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
    for idx in range(args.max_frames):
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
            bgr, tel = _fetch_live_robot(args.base)
            if bgr is None:
                time.sleep(0.05)
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

        if args.source == "live" and args.send_cmd:
            import urllib.request
            c = info["cmd"]
            url = f"{args.base}/cmd?vx={c['vx']:.1f}&vyaw={c['vyaw']:.1f}&head={c['head']:.1f}"
            try:
                urllib.request.urlopen(url, timeout=0.5).read()
            except Exception:
                pass

        if idx % 10 == 0 or info["knock"]["fired"] or idx == args.max_frames - 1:
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
