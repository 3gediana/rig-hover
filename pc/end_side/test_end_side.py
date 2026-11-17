# -*- coding: utf-8 -*-
"""test_end_side.py — 端侧闭环控制架构、39维特征契约、安全屏障与 TFLite 导出单测。"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    EndSideController,
    EndSideFeatureBuilder,
    EndSidePolicyNet,
    EndSideSafetyShield,
)
from end_side_protocol import DriveCommand, EndSideWatchdog, parse_status_json  # noqa: E402
from export_tflite_bundle import export_policy_bundle, verify_tflite_numerical  # noqa: E402


class TestEndSideArchitecture(unittest.TestCase):
    def test_feature_builder_39dim_contract_alignment(self):
        """39 维特征构建器必须与 Android 端 PolicyFeatureBuilder.kt 逐维对齐且无 NaN。"""
        raw_status = {
            "t": 1000,
            "imu": {"roll": -18.0, "pitch": 9.0, "yaw": 36.0, "ax": -2.0, "ay": -9.5, "az": 4.0, "pit_gyro": 1.0, "dq": -2.0},
            "wheel": {"v1": 20.0, "v2": -20.0, "x1": 0.5, "x2": 0.5},
            "head": {"q": 15.0, "target": 30.0, "gimbal_target": 0.0},
            "cmd": {"vx": 40.0, "vyaw": -10.0, "mode": 0, "pos": 0.4, "yaw_target": 18.0},
            "torque": {"t1": 50.0, "t2": -50.0},
            "pid": {"pos_e": 10.0, "pos_u": 20.0, "vel_e": -5.0, "vel_u": -10.0, "pit_e": 2.0, "pit_u": 4.0},
            "heap": 82000,
        }
        telemetry = parse_status_json(raw_status)
        detections = [
            {"cx": 0.55, "cy": 0.40, "w": 0.30, "h": 0.50, "score": 0.90, "cls": 0},
            {"cx": 0.20, "cy": 0.70, "w": 0.10, "h": 0.10, "score": 0.80, "cls": 1},
        ]
        feat = EndSideFeatureBuilder.build(telemetry, detections, target_lock=None)
        self.assertEqual(feat.shape, (39,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertTrue(np.isfinite(feat).all())
        self.assertAlmostEqual(float(feat[0]), -18.0 / 180.0, places=5)
        self.assertAlmostEqual(float(feat[1]), 9.0 / 180.0, places=5)
        self.assertAlmostEqual(float(feat[12]), 15.0 / 75.0, places=5)
        # 面积较大的框排在槽 0 (下标 28..32)
        self.assertAlmostEqual(float(feat[28]), 0.55, places=5)
        self.assertAlmostEqual(float(feat[38]), 1.0, places=5)

    def test_safety_shield_table_edge_zero_overshoot_and_pitch_protection(self):
        """安全屏障在剩余距离 <= 0 时必须硬钳位 vx <= 0（零过冲），在俯仰过大时必须降速。"""
        shield = EndSideSafetyShield(max_delta_vx=25.0, max_delta_vyaw=35.0, max_delta_head=15.0)
        # 1) 已经到达或越过敲击目标点 (remaining_m = -0.01)，策略仍试图往前冲 (vx=60)
        cmd_over = shield.apply(
            DriveCommand(vx=60.0, vyaw=10.0, head=12.0),
            mode=MODE_APPROACH,
            remaining_m=-0.01,
            pitch_deg=2.0,
        )
        self.assertLessEqual(cmd_over.vx, 0.0)

        # 2) 机器人俯仰角过大 (pitch_deg = 16.0 > 12.0)，前向速度必须被大幅抑制
        shield.reset()
        cmd_pitch = shield.apply(
            DriveCommand(vx=80.0, vyaw=60.0, head=5.0),
            mode=MODE_APPROACH,
            remaining_m=0.50,
            pitch_deg=16.0,
        )
        self.assertLessEqual(abs(cmd_pitch.vx), 15.0)

    def test_watchdog_keepalive_and_timeout_neutral(self):
        """800ms 看门狗守护器在超时或急停时必须立刻回退零速。"""
        wd = EndSideWatchdog(timeout_s=0.40)
        cmd = DriveCommand(vx=35.0, vyaw=-15.0, head=20.0)
        wd.update_command(cmd, now_s=1.00)
        out_ok = wd.get_safe_command(now_s=1.20)
        self.assertAlmostEqual(out_ok.vx, 35.0, places=4)

        # 超过 0.40s 未刷新指令 -> vx, vyaw 必须归零，head 保持
        out_stale = wd.get_safe_command(now_s=1.55)
        self.assertEqual(out_stale.vx, 0.0)
        self.assertEqual(out_stale.vyaw, 0.0)
        self.assertAlmostEqual(out_stale.head, 20.0, places=4)

    def test_end_side_controller_full_scenario_v2_transition(self):
        """端侧控制器应完整走通 IDLE -> GAZE_FACE -> GAZE_FIST -> 双敲锁定 APPROACH（手离开后靠里程计走完）-> DWELL -> GAZE_FACE。"""
        policy = EndSidePolicyNet()
        ctrl = EndSideController(policy_model=policy)

        # 1. 看到人脸 -> 进入 GAZE_FACE
        act = ctrl.step(
            dt=0.05,
            telemetry=parse_status_json({"head": {"q": 0.0}, "wheel": {"x1": 0.0, "x2": 0.0}}),
            scene_obs={"face_valid": 1, "theta_face": 12.0, "face_scale": 0.10, "person_h": 0.70,
                       "hand_valid": 0, "fist_prob": 0.0, "theta_fist": 0.0, "knock_fired": 0},
        )
        self.assertEqual(ctrl.mode, MODE_GAZE_FACE)

        # 2. 握拳 -> 进入 GAZE_FIST
        act = ctrl.step(
            dt=0.05,
            telemetry=parse_status_json({"head": {"q": 10.0}, "wheel": {"x1": 0.0, "x2": 0.0}}),
            scene_obs={"face_valid": 1, "theta_face": 2.0, "face_scale": 0.10, "person_h": 0.70,
                       "hand_valid": 1, "fist_prob": 0.85, "theta_fist": -5.0, "knock_fired": 0},
        )
        self.assertEqual(ctrl.mode, MODE_GAZE_FIST)

        # 3. 敲两下触发 -> 进入 MODE_APPROACH 并锁定目标
        act = ctrl.step(
            dt=0.05,
            telemetry=parse_status_json({"head": {"q": 5.0}, "wheel": {"x1": 0.0, "x2": 0.0}}),
            scene_obs={"face_valid": 1, "theta_face": 0.0, "face_scale": 0.10, "person_h": 0.70,
                       "hand_valid": 1, "fist_prob": 0.85, "theta_fist": 0.0, "knock_fired": 1},
        )
        self.assertEqual(ctrl.mode, MODE_APPROACH)
        self.assertEqual(ctrl.target_lock.valid, 1)
        target_dist = ctrl.target_lock.dist_m
        self.assertGreater(target_dist, 0.3)

        # 4. 手已经离开画面 (hand_valid=0)，机器人依靠轮式里程计走到目标距离，进入 MODE_DWELL 且绝不过冲
        act = ctrl.step(
            dt=0.05,
            telemetry=parse_status_json({"head": {"q": 0.0}, "wheel": {"x1": target_dist, "x2": target_dist}}),
            scene_obs={"face_valid": 1, "theta_face": 0.0, "face_scale": 0.14, "person_h": 0.80,
                       "hand_valid": 0, "fist_prob": 0.0, "theta_fist": 0.0, "knock_fired": 0},
        )
        self.assertEqual(ctrl.mode, MODE_DWELL)
        self.assertEqual(act.vx, 0.0)

    def test_tflite_export_and_sha256_bundle(self):
        """导出的 policy.tflite + model.json + zip 包必须通过 SHA-256 校验且与 PyTorch 数值对齐 (<1e-5)。"""
        policy = EndSidePolicyNet()
        policy.eval()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bundle = export_policy_bundle(
                policy,
                out_dir=tmp_dir,
                model_name="hover-rl-policy-test",
                version="2026.11.15",
            )
            tflite_path = Path(bundle["tflite_path"])
            json_path = Path(bundle["json_path"])
            zip_path = Path(bundle["zip_path"])

            self.assertTrue(tflite_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(zip_path.exists())

            meta = json.loads(json_path.read_text(encoding="utf-8"))
            real_sha = hashlib.sha256(tflite_path.read_bytes()).hexdigest()
            self.assertEqual(meta["sha256"], real_sha)
            self.assertEqual(meta["task"], "policy")
            self.assertEqual(meta["policy"]["featureDim"], 39)
            self.assertEqual(meta["policy"]["actionDim"], 3)

            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIn("model.tflite", zf.namelist())
                self.assertIn("model.json", zf.namelist())

            max_err = verify_tflite_numerical(policy, tflite_path)
            self.assertLess(max_err, 1e-5)


if __name__ == "__main__":
    unittest.main()
