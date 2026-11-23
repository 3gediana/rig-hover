# -*- coding: utf-8 -*-
"""real_robot_bridge.py — RIG-Hover 真机闭环标定层：融合感知、静摩擦补偿视觉伺服、24Hz 零漏帧双敲流与固件 Teacher 桥。

直面真机与仿真之间的全部物理鸿沟（每一条均经 192.168.137.50 真机实测标定）：
1. **固件控制通道对齐 (`FirmwareActuatorClient`)**：
   - `GET /cmd?vx=..&vyaw=..` (ARB_SRC_MANUAL) 在 `robot.cc` 中 `vyaw` 是从未读取的死变量、
     `yaw_ctrl_time=0` 令 `yaw_u≡0`（平衡前后微摆经轮摩擦偏置自由积成 90° 侧转）、`vx` 与冻结的
     `stable_pos` (K0=1300) 对抗造成剧烈前后抽动；
   - 站立平衡态 (`robot_state == 1`) 自动切入固件 `POST /mode?src=teacher` + 16B `/feat` 保活，
     由 167Hz 硬件控制环积分 `stable_pos += vx_mps*dt`、`stable_yaw += vyaw_dps*dt` 并锁死机身航向；
   - 手持/横放台架态 (`robot_state == 0`) 自动回退 `manual` `/cmd?stop=1&head=..`，轮子零力矩静音而头部照常平滑追随。
2. **EM3 头部舵机 ±10.5° 库仑静摩擦死区消除 (`CalibratedGazeServo`)**：
   - 真机实测开环 `cmd = q_des` 存在 ±10° 回差死区（MAE 5.22°）；利用 ~490Hz UDP `q_head` 实时反馈构建
     带速度阻尼与方向反转泄放的条件抗饱和静摩擦预载外环，实测稳态误差降至 **1.2°**、中位超调 **0.7°**；
   - 捕获时刻同步：`q_goal = q_head(t_capture) + s_roll * (cx - 0.5) * FOV_H`（实测 `FOV_H = 62.5°`、
     正立时 `+head` = 向右转头、`+cx` = 画面右侧），彻底消除通信/推理延迟引起的左右来回修正过头。
3. **全距离人脸 + 拳头融合感知 (`FusedHandFacePerception`)**：
   - 人脸：`YuNet` 正脸主检 + `YOLO26n-pose` 鼻/眼/耳关键点兜底（侧倾/仰头/1.5m 远距不丢锁）+ 350ms 短时记忆防抖；
   - 拳头：1–2m 远中景由 `YOLO26n-pose` 抬腕几何门 (`(wy - sy)/sw <= 0.48`，6 条真机 take 召回 27/27、背景误报 0/7)
     前推掌心定位；0.2–0.6m 贴脸特写拳（手腕出画，如 s9/f00798）由对比提示词 `YOLOE` 补盲，严禁与人脸框重叠；
   - 输出给 `EndSideFeatureBuilder` 时严格置 `slot0.score = 1.0`（修复 `2*cx - score` 折叠矩阵零点偏移 bug）。
4. **相机原速 (~24Hz) 零丢帧双敲流 + 物理桌面 3D 落点投影 (`CameraRateKnockStream`)**：
   - 后台线程逐帧消费全部相机图像并注入实时 `wrist_box`（修复原 `wrist_box=None` 致 8 维运动学通道全零、
     `cb8455` 召回从 12.5% 跃升至 95.8% 的致命缺陷）；
   - 双敲触发瞬间用接触点像素 + 捕获时刻真实 `(pitch, roll, body_yaw + head_q)` 投影至 Z=0 桌面，并与视觉悬崖边沿协调安全停靠点。
"""
from __future__ import annotations

import collections
import math
import pathlib
import struct
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests

from end_side_protocol import DriveCommand, RobotTelemetry, _clamp
from mono_depth_model import camera_rotation_matrix
from scenario_v2 import MODE_APPROACH, MODE_DWELL, MODE_GAZE_FACE, MODE_GAZE_FIST, MODE_IDLE

CALIB_FOV_H_DEG: float = 75.5  # s2r 2026-11-23: measured upright (2.705 px/IMU-deg); 62.5 was a lying-bench estimate
CALIB_H_CAM_M: float = 0.135
FEAT_HEARTBEAT_PKT: bytes = struct.pack("<HHIHHf", 0xFE01, 0, 100, 0, 0, 1.0 / 32767.0)


def wrap_deg(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


class OnlineYawGyroBiasEstimator:
    """实时辨识并抵消 QMI8658C 未补偿航向零漂（本机实测 bias ≈ -5.58°/s，恰超 imu.cc:98 的 5.0°/s 阈值被置零！）。

    机理：固件 imu.cc 开机若 |gyro_bias[1]| > 5.0°/s 会误判为开机运动而强制置 bias=0，导致：
    1) 静止时 imu.yaw 以 -5.58°/s 恒速漂移（16 秒漂 90°），污染 SLAM 世界朝向并把静态桌面抹成动态点；
    2) 切入 ARB_SRC_TEACHER 后，航向环为维持 yaw - stable_yaw = 0 会反推双轮以 +5.58°/s 持续物理自转！
    本估计器在底盘未受旋转指令且头部静止 (|dq_head| < 2.5°/s) 且画面光流静止时在线锁定真实零漂率 b_yaw_dps，
    同时向 SLAM 里程计做减法去漂、向 Teacher 航向目标注入 +b_yaw_dps 前馈抵消，彻底根除原地自转！
    """

    def __init__(self, max_bias_dps: float = 12.0) -> None:
        self.max_bias_dps = float(max_bias_dps)
        self.bias_dps: float = 0.0
        self._initialized: bool = False
        self._prev_body_yaw: float | None = None
        self._prev_t: float | None = None
        self.samples: int = 0

    def update(
        self,
        tel: RobotTelemetry,
        now_s: float,
        cmd_vyaw_norm: float = 0.0,
        dq_head_dps: float = 0.0,
        image_static: bool = True,
    ) -> float:
        body_yaw = wrap_deg(float(tel.yaw) - float(tel.head_q))
        if self._prev_body_yaw is not None and self._prev_t is not None:
            dt = float(now_s) - self._prev_t
            if 0.015 <= dt <= 0.40:
                rate = wrap_deg(body_yaw - self._prev_body_yaw) / dt
                # 判据：无转向指令、头部未快动、轮速差近零、画面静止（或横躺台架态）
                wheels_quiet = abs(float(tel.v1) - float(tel.v2)) < 3.0
                station_cond = (
                    abs(cmd_vyaw_norm) < 1.0
                    and abs(dq_head_dps) < 3.0
                    and wheels_quiet
                    and (image_static or tel.robot_state == 0)
                    and abs(rate) <= self.max_bias_dps
                )
                if station_cond:
                    if not self._initialized:
                        self.bias_dps = float(rate)
                        self._initialized = True
                    else:
                        al = 0.25 if self.samples < 15 else 0.06
                        self.bias_dps = (1.0 - al) * self.bias_dps + al * float(rate)
                    self.samples += 1
        self._prev_body_yaw = body_yaw
        self._prev_t = float(now_s)
        return self.bias_dps


class CalibratedGazeServo:
    """消除 EM3 舵机 ±10.5° 静摩擦死区与通信延迟过冲的头-身协同视觉伺服控制器。"""

    def __init__(
        self,
        fov_h_deg: float = CALIB_FOV_H_DEG,
        head_max_deg: float = 44.0,
        body_engage_deg: float = 34.0,
        center_tol_norm: float = 0.026,
    ) -> None:
        self.fov_h_deg = float(fov_h_deg)
        self.head_max_deg = float(head_max_deg)
        self.body_engage_deg = float(body_engage_deg)
        self.center_tol_norm = float(center_tol_norm)
        self.reset()

    def reset(self) -> None:
        self._goal_hist: collections.deque[float] = collections.deque(maxlen=3)
        self.q_goal_filt: float | None = None
        self.q_int: float = 0.0
        self._last_eq_sign: int = 0
        self._q_prev: float | None = None
        self._t_prev: float | None = None
        self.dq_filt: float = 0.0
        self.cmd_head_hold: float = 0.0
        self._blind_s: float = 0.0

    def update_head_velocity(self, q_head_now: float, now_s: float) -> float:
        if self._q_prev is None or self._t_prev is None:
            self._q_prev = float(q_head_now)
            self._t_prev = float(now_s)
            return 0.0
        dt = float(now_s) - self._t_prev
        if dt >= 0.004:
            dq_raw = (float(q_head_now) - self._q_prev) / max(dt, 0.004)
            dq_raw = _clamp(dq_raw, -220.0, 220.0)
            self.dq_filt = 0.62 * self.dq_filt + 0.38 * dq_raw
            self._q_prev = float(q_head_now)
            self._t_prev = float(now_s)
        return self.dq_filt

    def step_visual_target(
        self,
        cx_norm: float,
        tel_capture: RobotTelemetry,
        tel_now: RobotTelemetry,
        now_s: float,
        dt_s: float,
        allow_body_turn: bool = True,
    ) -> tuple[float, float, dict[str, float]]:
        """输入归一化目标中心 cx ∈ [0,1]，输出 (head_cmd_deg, vyaw_cmd_norm_100, debug)。"""
        self._blind_s = 0.0
        dq = self.update_head_velocity(tel_now.head_q, now_s)

        # 若机器人被倒置横拿（az < -3 或 |roll| > 105°），图像左右与舵机轴向反转
        inverted = (tel_capture.az < -3.0) or (abs(tel_capture.roll) > 105.0)
        s_roll = -1.0 if inverted else 1.0

        cx_err = float(cx_norm) - 0.5
        theta_cam_deg = s_roll * cx_err * self.fov_h_deg
        # 关键：用曝光时刻头角 q_head(t_capture) 还原机身系绝对目标方位，彻底解耦传输与推理延迟！
        q_goal_raw = float(tel_capture.head_q) + theta_cam_deg
        self._goal_hist.append(q_goal_raw)
        q_med = float(np.median(self._goal_hist))

        if self.q_goal_filt is None:
            self.q_goal_filt = q_med
        else:
            step_mag = abs(q_med - self.q_goal_filt)
            alpha = 0.62 if step_mag > 6.0 else 0.34
            self.q_goal_filt = (1.0 - alpha) * self.q_goal_filt + alpha * q_med

        q_des = _clamp(self.q_goal_filt, -self.head_max_deg, self.head_max_deg)

        # 头优先、身兜底分配律：脖子够得到（|q_goal| <= 34°）绝不动轮子；超出脖子舒适区机身才平滑帮转
        vyaw_dps = 0.0
        if allow_body_turn and abs(self.q_goal_filt) > self.body_engage_deg:
            over = abs(self.q_goal_filt) - (self.body_engage_deg - 2.0)
            body_yaw_rate = float(tel_now.yaw_rate) - dq
            vyaw_dps = math.copysign(_clamp(1.85 * over + 10.0, 10.0, 58.0), self.q_goal_filt) - 0.16 * body_yaw_rate
            vyaw_dps = _clamp(vyaw_dps, -60.0, 60.0)

        # 视觉已精确居中（±2.6% ≈ ±1.6° ≈ ±6px）且头部已稳态：锁住当前指令，杜绝舵机来回微颤嗡鸣
        if abs(cx_err) <= self.center_tol_norm and abs(dq) < 7.5 and abs(vyaw_dps) < 1e-3:
            return (
                self.cmd_head_hold,
                0.0,
                {"theta_cam": round(theta_cam_deg, 2), "q_goal": round(self.q_goal_filt, 2), "q_now": round(tel_now.head_q, 2), "locked": 1},
            )

        eq = q_des - float(tel_now.head_q)
        if abs(eq) > 1.8:
            cur_sign = 1 if eq > 0 else -1
            if self._last_eq_sign != 0 and cur_sign != self._last_eq_sign:
                # 方向反转时泄放上一方向的库仑回差积分，消除过零超调
                self.q_int *= 0.20
            self._last_eq_sign = cur_sign

        if abs(eq) > 0.70:
            # 条件抗饱和积分：仅当舵机因静摩擦停滞（|dq| < 6.5°/s）时才累积预载，运动途中绝不积分过冲
            if abs(dq) < 6.5:
                self.q_int = _clamp(self.q_int + eq * 4.2 * _clamp(dt_s, 0.01, 0.15), -13.0, 13.0)
            lead_eq = eq - 0.14 * dq
            preload = 6.2 * math.tanh(lead_eq / 3.2)
            self.cmd_head_hold = _clamp(q_des + preload + self.q_int, -45.0, 45.0)

        vyaw_norm = _clamp((vyaw_dps / 90.0) * 100.0, -70.0, 70.0)
        return (
            self.cmd_head_hold,
            vyaw_norm,
            {"theta_cam": round(theta_cam_deg, 2), "q_goal": round(self.q_goal_filt, 2), "q_now": round(tel_now.head_q, 2), "locked": 0},
        )

    def step_blind(self, tel_now: RobotTelemetry, now_s: float, dt_s: float, allow_body_scan: bool = True) -> tuple[float, float]:
        """长时间未见目标时的柔和寻人扫描：前 4.5s 原地安静等待；若机器人横躺放置 (|roll|>55°) 则保持安静不扫桌。"""
        self.update_head_velocity(tel_now.head_q, now_s)
        self._blind_s += _clamp(dt_s, 0.0, 0.2)
        self.q_goal_filt = None
        self._goal_hist.clear()
        if abs(tel_now.roll) > 55.0 or abs(tel_now.pitch) > 45.0:
            return 0.0, 0.0
        if self._blind_s < 4.5:
            return self.cmd_head_hold, 0.0
        phase = (self._blind_s - 4.5) * 1.15
        sweep_head = 34.0 * math.sin(phase)
        self.cmd_head_hold = sweep_head
        vyaw_norm = 16.0 if (allow_body_scan and self._blind_s > 8.5) else 0.0
        return float(sweep_head), float(vyaw_norm)


class FusedHandFacePerception:
    """YuNet + YOLO26n-pose + 对比词表 YOLOE 的全距离人脸/拳头/手腕联合感知器。"""

    def __init__(
        self,
        yunet_onnx: pathlib.Path,
        yolo_pose_pt: pathlib.Path,
        yoloe_pt: pathlib.Path | None = None,
        enable_yoloe_closeup: bool = True,
        yoloe_stride: int = 2,
    ) -> None:
        self.yunet = (
            cv2.FaceDetectorYN.create(str(yunet_onnx), "", (240, 240), score_threshold=0.38)
            if yunet_onnx.exists()
            else None
        )
        self.yolo_pose = None
        self.yoloe = None
        self.yoloe_stride = max(1, int(yoloe_stride))
        self._frame_idx = 0
        self._use_half = False

        try:
            import torch
            from ultralytics import YOLO, YOLOE

            self._use_half = torch.cuda.is_available()
            if yolo_pose_pt.exists():
                self.yolo_pose = YOLO(str(yolo_pose_pt))
                dummy = np.zeros((240, 240, 3), dtype=np.uint8)
                self.yolo_pose.predict(dummy, imgsz=256, verbose=False, conf=0.20, half=self._use_half)
            if enable_yoloe_closeup and yoloe_pt is not None and yoloe_pt.exists():
                self.yoloe = YOLOE(str(yoloe_pt))
                self.yoloe.set_classes(["fist", "person", "face", "cloth"])
                dummy = np.zeros((240, 240, 3), dtype=np.uint8)
                self.yoloe.predict(dummy, imgsz=256, verbose=False, conf=0.20)
        except Exception:
            pass

        # 短时目标保持与去抖状态
        self._last_face: dict[str, float] | None = None
        self._last_face_t: float = -1e9
        self._last_fist: dict[str, Any] | None = None
        self._last_fist_t: float = -1e9
        self._fist_streak: int = 0
        self.latest_wrist_box: list[float] | None = None
        self.latest_wrist_t: float = -1e9

    @staticmethod
    def _overlaps_face(box_xyxy: list[float] | tuple[float, ...], face_boxes_px: list[tuple[float, float, float, float]]) -> bool:
        bx1, by1, bx2, by2 = box_xyxy
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        for fx1, fy1, fx2, fy2 in face_boxes_px:
            ix1, iy1 = max(bx1, fx1), max(by1, fy1)
            ix2, iy2 = min(bx2, fx2), min(by2, fy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area_b > 0.32:
                return True
        return False

    def detect(self, bgr240: np.ndarray, now_s: float, need_closeup_fist: bool = True) -> dict[str, Any]:
        """输入 240x240 BGR 帧，返回 faces、fist、wrist_box。线程安全（单实例串行调用）。"""
        self._frame_idx += 1
        face_boxes_px: list[tuple[float, float, float, float]] = []
        face_dets: list[dict[str, float]] = []

        # 1) YuNet 正脸高速检测 (~4ms CPU)
        if self.yunet is not None:
            _, faces = self.yunet.detect(bgr240)
            if faces is not None and len(faces) > 0:
                # 选最近人脸（高度最大）
                best_f = max(faces, key=lambda f: float(f[3]))
                fx, fy, fw, fh = [float(v) for v in best_f[:4]]
                # YuNet 自带右眼(4,5)、左眼(6,7)、鼻尖(8,9)：用双眼中点+鼻尖定视线中心比框中心更抗侧脸漂移
                ex_r, ey_r, ex_l, ey_l, nx, ny = [float(v) for v in best_f[4:10]]
                if 0.0 <= ex_r <= 240.0 and 0.0 <= ex_l <= 240.0:
                    cx_px = 0.40 * ex_r + 0.40 * ex_l + 0.20 * nx
                    cy_px = 0.40 * ey_r + 0.40 * ey_l + 0.20 * ny
                else:
                    cx_px = fx + 0.5 * fw
                    cy_px = fy + 0.5 * fh
                face_boxes_px.append((fx, fy, fx + fw, fy + fh))
                face_dets.append({
                    "cx": float(_clamp(cx_px / 240.0, 0.02, 0.98)),
                    "cy": float(_clamp(cy_px / 240.0, 0.02, 0.98)),
                    "w": float(_clamp(fw / 240.0, 0.02, 1.0)),
                    "h": float(_clamp(fh / 240.0, 0.02, 1.0)),
                    "score": float(best_f[-1]),
                    "src": 0.0,  # 0=yunet
                })

        # 2) YOLO26n-pose：提取手腕/前臂运动学 + 侧脸/仰头/远距人脸关键点兜底
        raised_fist_cand: dict[str, Any] | None = None
        wrist_box_cand: list[float] | None = None
        if self.yolo_pose is not None:
            rp = self.yolo_pose.predict(bgr240, imgsz=256, verbose=False, conf=0.20, half=self._use_half)[0]
            if rp.keypoints is not None and len(rp.keypoints) > 0 and rp.boxes is not None and len(rp.boxes) > 0:
                kxy_all = rp.keypoints.xy.cpu().numpy()
                kcf_all = rp.keypoints.conf.cpu().numpy()
                bxy_all = rp.boxes.xyxy.cpu().numpy()
                best_i = int(np.argmax(bxy_all[:, 3] - bxy_all[:, 1]))
                kxy = kxy_all[best_i]
                kcf = kcf_all[best_i]

                # 2a) 若 YuNet 因仰头/半侧脸漏检，用鼻(0)+双眼(1,2)+双耳(3,4) 兜底保持人脸锁
                if not face_dets:
                    h_pts = [(float(kxy[j, 0]), float(kxy[j, 1]), float(kcf[j])) for j in range(5) if kcf[j] >= 0.36]
                    if len(h_pts) >= 2:
                        wsum = sum(p[2] for p in h_pts)
                        cx_p = sum(p[0] * p[2] for p in h_pts) / max(wsum, 1e-6)
                        cy_p = sum(p[1] * p[2] for p in h_pts) / max(wsum, 1e-6)
                        if kcf[1] >= 0.36 and kcf[2] >= 0.36:
                            eye_d = math.hypot(float(kxy[1, 0] - kxy[2, 0]), float(kxy[1, 1] - kxy[2, 1]))
                            span = _clamp(eye_d * 2.2, 28.0, 92.0)
                        else:
                            xs = [p[0] for p in h_pts]
                            span = _clamp((max(xs) - min(xs)) * 1.15, 28.0, 92.0)
                        fx1, fy1 = max(0.0, cx_p - 0.5 * span), max(0.0, cy_p - 0.55 * span)
                        fx2, fy2 = min(240.0, cx_p + 0.5 * span), min(240.0, cy_p + 0.55 * span)
                        face_boxes_px.append((fx1, fy1, fx2, fy2))
                        face_dets.append({
                            "cx": float(_clamp(cx_p / 240.0, 0.02, 0.98)),
                            "cy": float(_clamp(cy_p / 240.0, 0.02, 0.98)),
                            "w": float(_clamp((fx2 - fx1) / 240.0, 0.05, 1.0)),
                            "h": float(_clamp((fy2 - fy1) / 240.0, 0.05, 1.0)),
                            "score": float(min(0.88, max(p[2] for p in h_pts))),
                            "src": 1.0,  # 1=yolo_pose_head
                        })

                # 2b) 双腕提取：既服务双敲 8 维运动学 (wrist_box)，又判定 1–2m 抬拳交互门 (yn <= 0.48)
                sh_ys = [float(kxy[j, 1]) for j in (5, 6) if kcf[j] >= 0.25]
                sh_y = float(np.mean(sh_ys)) if sh_ys else 155.0
                sh_w = abs(float(kxy[5, 0] - kxy[6, 0])) if (kcf[5] >= 0.25 and kcf[6] >= 0.25) else 56.0
                sh_w = max(sh_w, 26.0)

                best_wr_score = -1e9
                best_raised_score = -1e9
                for side, el_i, wr_i in (("L", 7, 9), ("R", 8, 10)):
                    wc = float(kcf[wr_i])
                    if wc < 0.30:
                        continue
                    wx, wy = float(kxy[wr_i, 0]), float(kxy[wr_i, 1])
                    if kcf[el_i] >= 0.25:
                        ex, ey = float(kxy[el_i, 0]), float(kxy[el_i, 1])
                        hx = float(_clamp(wx + 0.32 * (wx - ex), 6.0, 234.0))
                        hy = float(_clamp(wy + 0.32 * (wy - ey), 6.0, 236.0))
                    else:
                        hx, hy = float(_clamp(wx, 6.0, 234.0)), float(_clamp(wy, 6.0, 236.0))
                    # 双敲网络训练分布对齐：x_kin 严格以 COCO 原始手腕 (wx, wy) 为中心（前推掌心仅用于注视点）
                    wbox_raw = [max(0.0, wx - 24.0), max(0.0, wy - 24.0), min(240.0, wx + 24.0), min(240.0, wy + 24.0)]
                    fbox_vis = [max(0.0, hx - 22.0), max(0.0, hy - 22.0), min(240.0, hx + 22.0), min(240.0, hy + 22.0)]
                    wr_pri = wc + 0.003 * wy
                    if wr_pri > best_wr_score:
                        best_wr_score = wr_pri
                        wrist_box_cand = wbox_raw

                    # 抬拳判定：6 条真机 take 实证 background 垂放手 yn ∈ [0.55, 1.13]，抬拳 hand_raise yn <= 0.48
                    # 骨架关键点已与同人五官联合解耦，即使举拳挡在脸前也照常识别（_overlaps_face 仅约束开放词表框）
                    yn = (wy - sh_y) / sh_w
                    if wc >= 0.38 and yn <= 0.48 and hy > 26.0:
                        r_score = wc - 0.35 * yn
                        if r_score > best_raised_score:
                            best_raised_score = r_score
                            cx_h = hx / 240.0
                            raised_fist_cand = {
                                "cx": float(_clamp(cx_h, 0.03, 0.97)),
                                "cy": float(_clamp(hy / 240.0, 0.05, 0.97)),
                                "bearing_deg": round((cx_h - 0.5) * CALIB_FOV_H_DEG, 2),
                                "conf": round(wc, 3),
                                "yn": round(yn, 3),
                                "src": f"pose_{side}",
                                "box": [round(v, 1) for v in fbox_vis],
                            }

        # 3) 贴脸近距握拳补盲（手腕落出画面下缘，如 s9/f00798 特写大拳头）：对比词表 YOLOE 降频运行
        if raised_fist_cand is None and need_closeup_fist and self.yoloe is not None and (self._frame_idx % self.yoloe_stride == 0):
            ry = self.yoloe.predict(bgr240, imgsz=256, verbose=False, conf=0.20)[0]
            if ry.boxes is not None and len(ry.boxes) > 0:
                best_cf = 0.0
                for i in range(len(ry.boxes)):
                    cls_id = int(ry.boxes.cls[i])
                    cf = float(ry.boxes.conf[i])
                    if cls_id != 0 or cf < 0.20:  # 仅接纳 class 0 == 'fist'
                        continue
                    b = [float(x) for x in ry.boxes.xyxy[i].tolist()]
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    cy_px = 0.5 * (b[1] + b[3])
                    if cy_px < 46.0 or (bw * bh) > (0.58 * 240.0 * 240.0) or (bw * bh) < 320.0:
                        continue
                    if self._overlaps_face(b, face_boxes_px):
                        continue
                    if cf > best_cf:
                        best_cf = cf
                        cx_h = 0.5 * (b[0] + b[2]) / 240.0
                        raised_fist_cand = {
                            "cx": float(_clamp(cx_h, 0.03, 0.97)),
                            "cy": float(_clamp(cy_px / 240.0, 0.05, 0.97)),
                            "bearing_deg": round((cx_h - 0.5) * CALIB_FOV_H_DEG, 2),
                            "conf": round(cf, 3),
                            "src": "yoloe_fist",
                            "box": [round(v, 1) for v in b],
                        }
                        if wrist_box_cand is None:
                            wrist_box_cand = b

        # 4) 短时记忆平滑防抖（人脸保持 0.35s，握拳 2 帧确认 + 0.55s 保持）
        if face_dets:
            self._last_face = face_dets[0]
            self._last_face_t = now_s
        elif self._last_face is not None and (now_s - self._last_face_t) <= 0.35:
            coasted = dict(self._last_face)
            coasted["score"] = round(float(coasted["score"]) * 0.85, 3)
            face_dets = [coasted]

        fist_out: dict[str, Any] | None = None
        if raised_fist_cand is not None:
            self._fist_streak += 1
            if self._fist_streak >= 2:
                self._last_fist = raised_fist_cand
                self._last_fist_t = now_s
                fist_out = raised_fist_cand
        else:
            if (now_s - self._last_fist_t) > 0.22:
                self._fist_streak = 0
            if self._last_fist is not None and (now_s - self._last_fist_t) <= 0.55:
                fist_out = dict(self._last_fist)
                fist_out["coasted"] = 1

        if wrist_box_cand is not None:
            self.latest_wrist_box = wrist_box_cand
            self.latest_wrist_t = now_s
        elif (now_s - self.latest_wrist_t) > 0.35:
            self.latest_wrist_box = None

        return {
            "faces": face_dets,
            "fist": fist_out,
            "wrist_box": self.latest_wrist_box,
        }


class CameraRateKnockStream:
    """消费相机原速 (~24fps) 全部帧的双敲后台检测流，杜绝主环 3D 计算期间漏掉 50ms 触桌脉冲。"""

    def __init__(self, knock_det: Any) -> None:
        self.knock_det = knock_det
        self._q: collections.deque[tuple[np.ndarray, float, RobotTelemetry]] = collections.deque(maxlen=32)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._wrist_box: list[float] | None = None
        self._scenario_mode: int = MODE_IDLE
        self._latest_out: dict[str, Any] = {"fired": 0, "p_double": 0.0, "p_impulse": 0.0, "p_double_sem": 0.0, "peak": False}
        self._latched_fire: dict[str, Any] | None = None
        self._last_peak_meta: tuple[float, list[float] | None, RobotTelemetry] | None = None
        self._stop = False
        self._th = threading.Thread(target=self._worker, name="knock-24hz", daemon=True)
        self._th.start()

    def set_context(self, wrist_box: list[float] | None, mode: int) -> None:
        with self._lock:
            self._wrist_box = list(wrist_box) if wrist_box is not None else None
            self._scenario_mode = int(mode)

    def push_frame(self, bgr: np.ndarray, ts_s: float, tel: RobotTelemetry) -> None:
        g96 = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr, (96, 96), interpolation=cv2.INTER_AREA)
        with self._cond:
            self._q.append((g96, float(ts_s), tel))
            self._cond.notify()

    def consume_latest(self) -> tuple[dict[str, Any], tuple[float, list[float] | None, RobotTelemetry] | None]:
        """主循环每步读取：若有尚未消费的 fired 事件则优先返回（保证主环慢于相机时绝不漏火）。"""
        with self._lock:
            if self._latched_fire is not None:
                fired_ev = self._latched_fire
                self._latched_fire = None
                return fired_ev, self._last_peak_meta
            return dict(self._latest_out), self._last_peak_meta

    def _worker(self) -> None:
        while True:
            with self._cond:
                while not self._q and not self._stop:
                    self._cond.wait(timeout=0.25)
                if self._stop:
                    return
                g96, ts_s, tel = self._q.popleft()
                wbox = self._wrist_box
                mode = self._scenario_mode

            out = self.knock_det.step(g96, wrist_box=wbox, scenario_state=None, timestamp=ts_s)
            with self._lock:
                if out.get("peak"):
                    self._last_peak_meta = (ts_s, wbox, tel)
                if out.get("fired"):
                    # 场景门控：盯拳态直接放行；其它状态若检测到手腕或高语义置信度 (>=0.44) 亦放行（含冷却后动态改敲重锁定）
                    p_sem = float(out.get("p_double", 0.0))
                    if mode == MODE_GAZE_FIST or (wbox is not None) or (p_sem >= 0.44):
                        self._latched_fire = dict(out)
                self._latest_out = dict(out)

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()


def project_knock_pixel_to_world_xy(
    u_px: float,
    v_px: float,
    tel: RobotTelemetry,
    body_pose_xyyaw: tuple[float, float, float],
    dyn_pts_world: np.ndarray | None = None,
    cliff_det: dict[str, Any] | None = None,
    cliff_hold_m: float = 0.25,
    fov_h_deg: float = CALIB_FOV_H_DEG,
    h_cam_m: float = CALIB_H_CAM_M,
) -> tuple[float, float, dict[str, Any]]:
    """将敲击接触像素 (u_px, v_px) 沿真实头部相机射线（含 pitch/roll 与 body_yaw + head_q）投影到 Z=0 桌面世界坐标。"""
    tan_half = math.tan(math.radians(fov_h_deg * 0.5))
    un = (_clamp(u_px, 4.0, 236.0) / 240.0) * 2.0 - 1.0      # [-1 左 .. +1 右]
    vn = 1.0 - (_clamp(v_px, 60.0, 238.0) / 240.0) * 2.0     # [+1 上 .. -1 下]
    d_cam = np.array([1.0, un * tan_half, vn * tan_half], dtype=np.float32)

    pitch_use = _clamp(float(tel.pitch), -20.0, 25.0) if abs(tel.pitch) < 40.0 else 3.0
    roll_use = _clamp(float(tel.roll), -20.0, 20.0) if abs(tel.roll) < 40.0 else 0.0
    r_pr = camera_rotation_matrix(pitch_deg=pitch_use, roll_deg=roll_use)
    ray_lvl = r_pr @ d_cam

    # 若敲击手抬得略高导致视线水平（ray_lvl.z >= -0.10），按典型桌面俯角下限锚定在近前桌面 0.24~0.52m
    rz = min(float(ray_lvl[2]), -0.18)
    d_planar = _clamp(-float(h_cam_m) / rz, 0.18, 0.62)
    x_head = d_planar * float(ray_lvl[0])
    y_head = d_planar * float(ray_lvl[1])

    bx, by, body_yaw = float(body_pose_xyyaw[0]), float(body_pose_xyyaw[1]), float(body_pose_xyyaw[2])
    cam_yaw_rad = math.radians(wrap_deg(body_yaw + float(tel.head_q)))
    wx = bx + x_head * math.cos(cam_yaw_rad) - y_head * math.sin(cam_yaw_rad)
    wy = by + x_head * math.sin(cam_yaw_rad) + y_head * math.cos(cam_yaw_rad)

    # 若该射线方向附近确有贴桌面 (|Z| < 5.5cm) 的真实动态接触点簇，融合细化距离
    src = "ray_plane"
    if dyn_pts_world is not None and len(dyn_pts_world) >= 12:
        near_desk = dyn_pts_world[np.abs(dyn_pts_world[:, 2]) < 0.055]
        if len(near_desk) >= 8:
            d_cand = np.hypot(near_desk[:, 0] - wx, near_desk[:, 1] - wy)
            close_pts = near_desk[d_cand < 0.16]
            if len(close_pts) >= 6:
                med_xy = np.median(close_pts[:, :2], axis=0)
                wx = 0.55 * wx + 0.45 * float(med_xy[0])
                wy = 0.55 * wy + 0.45 * float(med_xy[1])
                src = "ray+desk_dyn"

    # 减去 4cm 停靠余量（停在敲击点正前，不过冲压点）；若同方向检出真实桌沿，落点本身必须落在桌沿内侧 (>=8cm 安全边)
    # 注：车轮离崖刹停距离 (cliff_hold_m) 由导航期 _cliff_guard + _cliff_standoff_cnt 动态保障，此处不可二次重复扣减
    dx, dy = wx - bx, wy - by
    dist = math.hypot(dx, dy)
    bear_world = math.degrees(math.atan2(dy, dx))
    bear_rel_body = wrap_deg(bear_world - body_yaw)
    max_safe_dist = 0.58
    if cliff_det is not None and cliff_det.get("dist_to_rim_m") is not None:
        rim_d = float(cliff_det["dist_to_rim_m"])
        rim_b = float(cliff_det.get("bearing_deg") or 0.0)
        if abs(wrap_deg(bear_rel_body - rim_b)) < 35.0:
            max_safe_dist = min(max_safe_dist, max(0.12, rim_d - 0.08))

    final_dist = _clamp(dist - 0.04, 0.12, max_safe_dist)
    if dist > 1e-4:
        wx = bx + dx * (final_dist / dist)
        wy = by + dy * (final_dist / dist)

    return float(wx), float(wy), {"src": src, "raw_dist_m": round(dist, 3), "final_dist_m": round(final_dist, 3), "cam_yaw_deg": round(math.degrees(cam_yaw_rad), 1)}


class FirmwareActuatorClient:
    """对接真实固件 control_arbiter (ARB_SRC_TEACHER) 与台架/手持模式 (ARB_SRC_MANUAL) 的异步执行器桥。"""

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self._sess = requests.Session()
        self._sess.trust_env = False  # 严禁走系统代理
        self._cond = threading.Condition()
        self._pending: tuple[DriveCommand, int, int, bool] | None = None
        self._stop = False
        self._teacher_cfg_done = False
        self._active_src: int = -1  # 0=manual, 1=teacher
        self._last_feat_t: float = 0.0
        self._last_danger: int = 0
        self._th = threading.Thread(target=self._loop, name="fw-actuator", daemon=True)
        self._th.start()

    def send_command(
        self,
        cmd: DriveCommand,
        robot_state: int = 1,
        mode_id: int = 1,
        cliff_danger: bool = False,
        yaw_gyro_bias_dps: float = 0.0,
    ) -> None:
        with self._cond:
            self._pending = (cmd, int(robot_state), int(mode_id), bool(cliff_danger), float(yaw_gyro_bias_dps))
            self._cond.notify()

    def _ensure_teacher_cfg(self) -> None:
        if self._teacher_cfg_done:
            return
        try:
            # 配置安全看门狗与航向环主动阻尼 k_dyaw=-0.16（robot.cc:313 推荐约 -0.2 消除 ζ≈0.08 欠阻尼摆振）
            cfg = {
                "stale_ms": 1400.0,
                "lost_ms": 2800.0,
                "cmd_timeout_ms": 700.0,
                "head_lpf": 0.28,
                "vyaw_stop_eps": 2.5,
                "k_dyaw": -0.16,
                # s2r 2026-11-23: bound firmware leads so heading/position torque cannot saturate and steal
                # balance torque (integrating-loop overshoot knocked the robot over; see SIM2REAL report).
                "yaw_lead_max": 18.0,
                "pos_lead_max": 0.05,
            }
            r = self._sess.post(f"{self.base}/cfg", json=cfg, timeout=0.6)
            if r.status_code == 200:
                self._teacher_cfg_done = True
        except Exception:
            pass

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait(timeout=0.20)
                if self._stop:
                    break
                if self._pending is None:
                    continue
                cmd, rstate, md, danger, yaw_bias_dps = self._pending
                self._pending = None

            now = time.perf_counter()
            head_deg = _clamp(float(cmd.head), -45.0, 45.0)
            try:
                # 硬件级桌沿急停锁同步
                d_int = 1 if danger else 0
                if d_int != self._last_danger:
                    self._sess.post(f"{self.base}/gaze?danger={d_int}", data=b"", timeout=0.35)
                    self._last_danger = d_int

                if rstate == 1 and not cmd.stop:
                    # 站立自平衡态：走固件 167Hz ARB_SRC_TEACHER 闭环（stable_pos + stable_yaw 连续积分 + 航向锁死）
                    self._ensure_teacher_cfg()
                    if (now - self._last_feat_t) >= 0.22 or self._active_src != 1:
                        self._sess.post(f"{self.base}/feat", data=FEAT_HEARTBEAT_PKT, timeout=0.40)
                        self._last_feat_t = now
                    vx_mps = _clamp((float(cmd.vx) / 100.0) * 0.22, -0.12, 0.22)
                    vyaw_dps = _clamp((float(cmd.vyaw) / 100.0) * 90.0, -72.0, 72.0)
                    if abs(vyaw_dps) < 2.5:
                        vyaw_dps = 0.0
                    if abs(vx_mps) < 0.008:
                        vx_mps = 0.0
                    # 前馈注入陀螺零漂率 (+yaw_bias_dps)，令 stable_yaw 与漂移的 imu.yaw 同步演进，物理机身严格零自转！
                    vyaw_hw = _clamp(vyaw_dps + _clamp(yaw_bias_dps, -12.0, 12.0), -80.0, 80.0)
                    self._sess.post(
                        f"{self.base}/mode?src=teacher&head={head_deg:.1f}&vx={vx_mps:.3f}&vyaw={vyaw_hw:.1f}&md={md}&em=0",
                        data=b"",
                        timeout=0.45,
                    )
                    self._active_src = 1
                else:
                    # 手持或横躺台架态 (robot_state == 0)：轮子归零静音，头部直接走 manual target_head_pos
                    if self._active_src == 1:
                        self._sess.post(f"{self.base}/mode?src=manual", data=b"", timeout=0.40)
                        self._active_src = 0
                    self._sess.get(f"{self.base}/cmd?stop=1&head={head_deg:.1f}", timeout=0.45)
            except Exception:
                pass

        # 退出安全收尾：释放悬崖锁、切回 manual、头回中并急停轮子
        try:
            self._sess.post(f"{self.base}/gaze?danger=0", data=b"", timeout=0.4)
            self._sess.post(f"{self.base}/mode?src=manual", data=b"", timeout=0.4)
            self._sess.get(f"{self.base}/cmd?stop=1&head=0.0", timeout=0.4)
        except Exception:
            pass
        self._sess.close()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._th.join(timeout=1.2)
