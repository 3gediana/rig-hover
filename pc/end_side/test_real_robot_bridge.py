# -*- coding: utf-8 -*-
"""test_real_robot_bridge.py — 真机 Sim2Real 全链路修复门禁测试套件。

覆盖 9 项真机实测暴露的核心缺陷：
1. RobotTelemetry.from_status_json 解析真实 /status（含 1024-CPR 编码器脉冲转米、cfc.block 站立态判定）；
2. 500Hz TLM1 UDP 二进制遥测解包与相机曝光时刻对齐查询；
3. CalibratedGazeServo 在 ±10.5° 库仑静摩擦舵机 + 100ms 视觉延迟下的零超调精确居中（小步 ±6° 与大步 ±28°）；
4. 头优先/身兜底分配律（|q_goal| <= 34° 时 vyaw === 0，绝不诱发底盘漂移）；
5. 敲击 3D 桌面投影包含 head_q 视线偏角（歪头 28° 敲击落点绝不退化到正前方）与桌沿安全距离钳位；
6. FirmwareActuatorClient 站立态 (ARB_SRC_TEACHER + 16B /feat + /cfg k_dyaw) 与手持态 (ARB_SRC_MANUAL) 自动切换与安全退出；
7. 真机实录帧回归（s9 特写大拳头、s10 仰头侧脸+远距举拳、s9/s5 纯人脸零误报）。
"""
from __future__ import annotations

import collections
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import pathlib
import struct
import sys
import threading
import time
import unittest

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from end_side_protocol import (  # noqa: E402
    DriveCommand,
    RobotTelemetry,
    TLM1_MAGIC,
    parse_tlm1_packet,
)
from real_robot_bridge import (  # noqa: E402
    CALIB_FOV_H_DEG,
    CalibratedGazeServo,
    FirmwareActuatorClient,
    FusedHandFacePerception,
    OnlineYawGyroBiasEstimator,
    project_knock_pixel_to_world_xy,
)
from scenario1_own_models_e2e import YOLO_POSE_PT, YUNET_ONNX  # noqa: E402

YOLOE_SEG_PT = REPO / "pc" / "perception" / "assets" / "yoloe-26n-seg.pt"


class _StictionServoPlant:
    """复刻 192.168.137.50 真机实测的 EM3 头部舵机：±10.2° 库仑静摩擦死区 + 一阶伺服速度爬坡。"""

    def __init__(self, stiction_deg: float = 10.2, tau_s: float = 0.08) -> None:
        self.stiction = float(stiction_deg)
        self.tau = float(tau_s)
        self.q = 0.0

    def step(self, target_cmd: float, dt: float) -> float:
        target_cmd = max(-45.0, min(45.0, float(target_cmd)))
        diff = target_cmd - self.q
        if abs(diff) <= self.stiction:
            eq_target = self.q
        else:
            eq_target = target_cmd - math.copysign(self.stiction, diff)
        alpha = dt / (self.tau + dt)
        self.q += alpha * (eq_target - self.q)
        return self.q


class TestRealRobotBridge(unittest.TestCase):
    def test_status_json_and_wheel_ticks_conversion(self) -> None:
        raw_live = {
            "t": 3023870,
            "imu": {"roll": -2.1, "pitch": 4.2, "yaw": -145.0, "ax": -9.6, "ay": 0.1, "az": 0.8, "pit_gyro": 0.5, "dq": 0.4, "yaw_rate": -3.2},
            "wheel": {"v1": 0.0, "v2": 0.0, "x1": 1024, "x2": 1024},
            "head": {"q": 12.5, "target": 20.0, "gimbal_target": 0.0},
            "cmd": {"vx": 0.0, "vyaw": 0.0, "mode": 0, "pos": 0.1885, "yaw_target": -157.5},
            "torque": {"t1": 18.0, "t2": -18.0},
            "pid": {},
            "cfc": {"src": 1, "block": 0},
            "heap": 8153672,
        }
        tel = RobotTelemetry.from_status_json(raw_live)
        self.assertAlmostEqual(tel.pitch, 4.2, places=3)
        self.assertAlmostEqual(tel.head_q, 12.5, places=3)
        self.assertAlmostEqual(tel.yaw_rate, -3.2, places=3)
        self.assertEqual(tel.robot_state, 1)
        # 1024 ticks == 1 wheel rev == pi * 0.06m ≈ 0.1885m
        self.assertAlmostEqual(tel.wheel_x_m, math.pi * 0.06, places=3)

        # 横躺态 block=2 (ARB_BLK_ROBOT_OFF) -> robot_state == 0
        raw_fallen = dict(raw_live, cfc={"src": 0, "block": 2}, torque={"t1": 0.0, "t2": 0.0})
        self.assertEqual(RobotTelemetry.from_status_json(raw_fallen).robot_state, 0)

    def test_tlm1_udp_packet_decode(self) -> None:
        floats = [0.0] * 34
        floats[0] = 1.5     # roll
        floats[1] = 3.8     # pitch
        floats[2] = 42.0    # yaw
        floats[7] = 9.7     # az
        floats[12] = 0.345  # wheel_x (meters)
        floats[14] = -16.4  # q_head
        floats[15] = -24.0  # target_head_pos
        floats[22] = 1.0    # robot_state
        pkt = struct.pack("<IHHII34f", TLM1_MAGIC, 1, 152, 99, 12345, *floats)
        tel = parse_tlm1_packet(pkt)
        self.assertIsNotNone(tel)
        assert tel is not None
        self.assertEqual(tel.t_ms, 12345)
        self.assertAlmostEqual(tel.head_q, -16.4, places=2)
        self.assertAlmostEqual(tel.wheel_x_m, 0.345, places=4)
        self.assertEqual(tel.robot_state, 1)

    def test_stiction_gaze_servo_eliminates_deadband_without_oscillation(self) -> None:
        """在 ±10.2° 静摩擦死区 + 2 帧视觉延迟下，小角度 (+6°) 与大角度 (-24°) 目标均精确居中且无震荡。"""
        plant = _StictionServoPlant(stiction_deg=10.2, tau_s=0.07)
        servo = CalibratedGazeServo(fov_h_deg=CALIB_FOV_H_DEG)
        dt = 0.05  # 20Hz
        delay_q: collections.deque[float] = collections.deque([0.0, 0.0], maxlen=2)

        for true_face_yaw in (6.5, -24.0, 0.0, 40.0):
            q_start = plant.q
            step_dir = 1.0 if (min(44.0, max(-44.0, true_face_yaw)) - q_start) >= 0 else -1.0
            max_overshoot = 0.0
            vyaw_out = 0.0
            for step_i in range(45):
                now_s = step_i * dt
                q_cap = delay_q[0]
                cx = 0.5 + (true_face_yaw - q_cap) / CALIB_FOV_H_DEG
                cx = max(0.02, min(0.98, cx))
                tel_cap = RobotTelemetry(head_q=q_cap, az=9.8, robot_state=1)
                tel_now = RobotTelemetry(head_q=plant.q, az=9.8, robot_state=1)
                h_cmd, vyaw_out, _ = servo.step_visual_target(cx, tel_cap, tel_now, now_s=now_s, dt_s=dt, allow_body_turn=True)
                delay_q.append(plant.q)
                plant.step(h_cmd, dt)
                if abs(true_face_yaw) <= 34.0:
                    self.assertEqual(vyaw_out, 0.0, "脖子够得到时绝不允许驱动底盘旋转")
                    ov = (plant.q - true_face_yaw) * step_dir
                    if ov > max_overshoot:
                        max_overshoot = ov

            if abs(true_face_yaw) <= 34.0:
                self.assertLess(abs(plant.q - true_face_yaw), 2.0, f"稳态未居中: q={plant.q:.2f} vs {true_face_yaw}")
                self.assertLess(max_overshoot, 3.2, f"修正过头震荡: overshoot={max_overshoot:.2f}")
            else:
                # 40° 超出 34° 舒适门限，机身必须同向辅助转向 (vyaw > 0)
                self.assertGreater(vyaw_out, 5.0)

    def test_knock_3d_projection_includes_head_yaw_and_cliff_clamp(self) -> None:
        # 机器人机身朝正前 (0°)，但歪头向右 +28° 盯着右侧敲击手（手在相机正中 u=120, v=195）
        tel = RobotTelemetry(pitch=5.0, roll=0.0, yaw=28.0, head_q=28.0, robot_state=1)
        wx, wy, info = project_knock_pixel_to_world_xy(120.0, 195.0, tel, body_pose_xyyaw=(0.0, 0.0, 0.0))
        self.assertGreater(wx, 0.12)
        self.assertGreater(wy, 0.07, f"歪头 +28° 敲击点必须在机身右侧 (+Y>0)，实得 wy={wy:.3f}")
        bear = math.degrees(math.atan2(wy, wx))
        self.assertAlmostEqual(bear, 28.0, delta=3.0)

        # 若正前 0.36m 处有真实桌沿，即使像素靠近水平线也必须钳在桌沿内侧 (<= 0.36 - 0.08 = 0.28m)
        cliff = {"dist_to_rim_m": 0.36, "bearing_deg": 28.0}
        wx_c, wy_c, _ = project_knock_pixel_to_world_xy(
            120.0, 140.0, tel, body_pose_xyyaw=(0.0, 0.0, 0.0), cliff_det=cliff, cliff_hold_m=0.25
        )
        self.assertLessEqual(math.hypot(wx_c, wy_c), 0.285)
        self.assertGreaterEqual(math.hypot(wx_c, wy_c), 0.12)

    def test_firmware_actuator_teacher_vs_manual_switching(self) -> None:
        calls: list[tuple[str, str, int]] = []

        class MockRobotHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                _ = self.rfile.read(n) if n > 0 else b""
                calls.append(("POST", self.path, n))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":1}')

            def do_GET(self):
                calls.append(("GET", self.path, 0))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *args):
                pass

        srv = HTTPServer(("127.0.0.1", 0), MockRobotHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        try:
            cli = FirmwareActuatorClient(base)
            # 1) 站立平衡态（含 -5.5°/s 陀螺零漂前馈抵消：vyaw=18.0 + (-5.5) = 12.5°/s）-> 走 /cfg + 16B /feat + /mode?src=teacher
            cli.send_command(
                DriveCommand(vx=50.0, vyaw=20.0, head=15.0),
                robot_state=1, mode_id=3, cliff_danger=False, yaw_gyro_bias_dps=-5.5,
            )
            time.sleep(0.15)
            # 2) 被人拿起手持态 (robot_state=0) -> 自动退回 manual 并仅发 /cmd?stop=1&head=..
            cli.send_command(DriveCommand(vx=50.0, vyaw=20.0, head=-12.0), robot_state=0, mode_id=1, cliff_danger=True)
            time.sleep(0.15)
            cli.close()
        finally:
            srv.shutdown()
            srv.server_close()

        paths = [p for _, p, _ in calls]
        self.assertTrue(any(p == "/cfg" for p in paths), f"未配置 /cfg: {paths}")
        self.assertTrue(any(p == "/feat" and n == 16 for _, p, n in calls), f"未发 16B /feat 保活: {calls}")
        self.assertTrue(any(p.startswith("/mode?src=teacher&head=15.0&vx=0.110&vyaw=12.5") for p in paths), f"Teacher 陀螺前馈转换错误: {paths}")
        self.assertTrue(any(p == "/gaze?danger=1" for p in paths))
        self.assertTrue(any(p == "/cmd?stop=1&head=-12.0" for p in paths), f"手持态未锁轮控头: {paths}")
        self.assertEqual(paths[-1], "/cmd?stop=1&head=0.0", f"退出未执行安全停车回中: {paths}")

    def test_online_yaw_gyro_bias_estimator_cancels_qmi8658c_drift(self) -> None:
        est = OnlineYawGyroBiasEstimator()
        b = 0.0
        for i in range(25):
            t = i * 0.08
            tel = RobotTelemetry(yaw=-5.58 * t, head_q=4.0, v1=0.0, v2=0.0, robot_state=1)
            b = est.update(tel, now_s=t, cmd_vyaw_norm=0.0, dq_head_dps=0.0, image_static=True)
        self.assertAlmostEqual(b, -5.58, delta=0.15)

    def test_fused_perception_on_recorded_live_dumps(self) -> None:
        s9_fist = REPO / "runs_end_side" / "live_dump_s9" / "f00798.jpg"
        s10_fist = REPO / "runs_end_side" / "live_dump_s10" / "f00918.jpg"
        s9_face = REPO / "runs_end_side" / "live_dump_s9" / "f00300.jpg"
        s5_face = REPO / "runs_end_side" / "live_dump_s5" / "f00642.jpg"
        if not all(p.exists() for p in (s9_fist, s10_fist, s9_face, s5_face)):
            self.skipTest("live_dump images not present on this machine")

        perc = FusedHandFacePerception(YUNET_ONNX, YOLO_POSE_PT, YOLOE_SEG_PT, enable_yoloe_closeup=True, yoloe_stride=1)

        # 1) 纯人脸帧绝不可误触发 GAZE_FIST（修复 s10 中 83% 帧卡死 GAZE_FIST 的恶性假阳性）
        for fp_img in (s9_face, s5_face):
            im = cv2.imread(str(fp_img))
            out = perc.detect(im, now_s=1.0, need_closeup_fist=True)
            self.assertEqual(len(out["faces"]), 1, f"{fp_img.name} 漏检正脸")
            self.assertIsNone(out["fist"], f"{fp_img.name} 纯人脸误判为拳头: {out['fist']}")

        # 2) s9/f00798 贴脸特写大拳头（手腕出画）连续两帧确认检出
        im_cu = cv2.imread(str(s9_fist))
        _ = perc.detect(im_cu, now_s=2.0, need_closeup_fist=True)
        out_cu = perc.detect(im_cu, now_s=2.06, need_closeup_fist=True)
        self.assertIsNotNone(out_cu["fist"], "漏检贴脸特写大拳头 s9/f00798")

        # 3) s10/f00918 仰头侧脸（YuNet 漏检）+ 远距抬拳：人脸兜底与远距抬拳同时命中
        perc2 = FusedHandFacePerception(YUNET_ONNX, YOLO_POSE_PT, YOLOE_SEG_PT, enable_yoloe_closeup=True, yoloe_stride=1)
        im_far = cv2.imread(str(s10_fist))
        _ = perc2.detect(im_far, now_s=3.0, need_closeup_fist=True)
        out_far = perc2.detect(im_far, now_s=3.06, need_closeup_fist=True)
        self.assertEqual(len(out_far["faces"]), 1, "仰头人脸兜底失败 s10/f00918")
        self.assertIsNotNone(out_far["fist"], "远距举拳漏检 s10/f00918")


if __name__ == "__main__":
    unittest.main()
