# -*- coding: utf-8 -*-
"""test_spatial_sandbox.py — SE(2) 自车空间记忆追踪器、39维广角/净空编码、残差策略 TFLite 导出与 8 场景复杂物理沙箱单测。"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    EndSideController,
    EndSideFeatureBuilder,
    EndSidePolicyNet,
)
from end_side_protocol import DriveCommand, parse_status_json  # noqa: E402
from export_tflite_bundle import export_policy_bundle, verify_tflite_numerical  # noqa: E402
from sandbox_arena import SCENARIO_NAMES, HoverSandboxEnv  # noqa: E402
from spatial_memory import SpatialMemoryTracker  # noqa: E402


class TestSpatialMemoryAndSandbox(unittest.TestCase):
    def test_spatial_memory_pure_body_rotation_when_blind(self):
        """双敲锁定偏轴 +40° 目标后手立即消失，机身原地右转 +40° 后目标机身方位角必须归零且距离守恒。"""
        tracker = SpatialMemoryTracker(reknock_refractory_s=0.65)
        t0 = parse_status_json({"imu": {"yaw": 10.0}, "head": {"q": 10.0}, "wheel": {"x1": 0.0, "x2": 0.0}})
        tracker.step_odometry(0.05, t0)
        locked = tracker.lock_polar(now_s=0.05, bearing_body_deg=40.0, dist_m=0.60, x_px=0.8, y_px=0.7)
        self.assertEqual(locked.valid, 1)
        self.assertAlmostEqual(locked.bearing_body_deg, 40.0, places=3)
        self.assertAlmostEqual(locked.remaining_m, 0.60, places=3)

        # 机身右转 +40°，头部回到 0°（头部 IMU yaw = yaw_body + q_head = 40.0 + 0.0 = 40.0）
        for step_i in range(1, 9):
            yaw_body = 40.0 * (step_i / 8.0)
            telem = parse_status_json({
                "imu": {"yaw": yaw_body},
                "head": {"q": 0.0},
                "wheel": {"x1": 0.0, "x2": 0.0, "v1": 12.0, "v2": -12.0},
                "cmd": {"vyaw": 35.0},
            })
            st = tracker.step_odometry(0.05, telem)

        self.assertAlmostEqual(st.bearing_body_deg, 0.0, delta=0.25)
        self.assertAlmostEqual(st.remaining_m, 0.60, delta=1e-3)

    def test_spatial_memory_se2_curved_arc_accuracy(self):
        """沿二维圆弧盲走 (vx=0.20 m/s, vyaw=25 deg/s) 1.2s 后，自车系坐标应与世界真值刚体逆变换严格一致。"""
        tracker = SpatialMemoryTracker()
        # 初始机器人在世界原点 (0, 0)，朝向 yaw_body = 0 (+X 前，+Y 右)
        p_world_x = 0.75 * math.cos(math.radians(32.0))
        p_world_y = 0.75 * math.sin(math.radians(32.0))

        t_init = parse_status_json({"imu": {"yaw": 0.0}, "head": {"q": 0.0}, "wheel": {"x1": 0.0, "x2": 0.0}})
        tracker.step_odometry(0.02, t_init)
        tracker.lock_polar(now_s=0.02, bearing_body_deg=32.0, dist_m=0.75)

        rx, ry, ryaw_deg = 0.0, 0.0, 0.0
        wheel_s = 0.0
        dt = 0.02
        vx_mps = 0.20
        vyaw_dps = 25.0
        for _ in range(60):
            ds = vx_mps * dt
            dyaw_deg = vyaw_dps * dt
            mid_rad = math.radians(ryaw_deg + 0.5 * dyaw_deg)
            rx += ds * math.cos(mid_rad)
            ry += ds * math.sin(mid_rad)
            ryaw_deg += dyaw_deg
            wheel_s += ds
            q_head = 8.0 * math.sin(wheel_s * 5.0)
            telem = parse_status_json({
                "imu": {"yaw": ryaw_deg + q_head},
                "head": {"q": q_head},
                "wheel": {"x1": wheel_s, "x2": wheel_s, "v1": 25.0, "v2": 15.0},
                "cmd": {"vx": 60.0, "vyaw": 28.0},
            })
            st = tracker.step_odometry(dt, telem)

        # 世界真值转换到当前自车系
        dx_w = p_world_x - rx
        dy_w = p_world_y - ry
        psi = math.radians(ryaw_deg)
        true_xb = math.cos(psi) * dx_w + math.sin(psi) * dy_w
        true_yb = -math.sin(psi) * dx_w + math.cos(psi) * dy_w
        true_rem = math.hypot(true_xb, true_yb)
        true_bearing = math.degrees(math.atan2(true_yb, true_xb))

        self.assertAlmostEqual(st.x_body_m, true_xb, delta=2e-3)
        self.assertAlmostEqual(st.y_body_m, true_yb, delta=2e-3)
        self.assertAlmostEqual(st.remaining_m, true_rem, delta=2e-3)
        self.assertAlmostEqual(st.bearing_body_deg, true_bearing, delta=0.25)

    def test_controller_ignores_waving_hand_and_overrides_on_reknock(self):
        """APPROACH 期间抬手挥动不得篡改已锁定的敲击方位；但冷却期后新的双敲事件必须成功重锁定新位置。"""
        ctrl = EndSideController(policy_model=EndSidePolicyNet())
        t0 = parse_status_json({"imu": {"yaw": 0.0}, "head": {"q": 0.0}, "wheel": {"x1": 0.0, "x2": 0.0}})
        # 1) 在 +28° 处双敲锁定
        ctrl.step(
            dt=0.05,
            telemetry=t0,
            scene_obs={
                "face_valid": 1, "theta_face": 0.0, "face_scale": 0.11, "person_h": 0.75,
                "hand_valid": 1, "fist_prob": 0.88, "theta_fist": 28.0, "knock_fired": 1,
            },
        )
        self.assertEqual(ctrl.mode, MODE_APPROACH)
        self.assertAlmostEqual(ctrl.target_lock.bearing_body, 28.0, delta=0.5)

        # 2) 手抬起挥舞到 -42°（非双敲 knock_fired=0），锁定方位绝不能被拉到 -42°！
        for _ in range(15):
            ctrl.step(
                dt=0.05,
                telemetry=t0,
                scene_obs={
                    "face_valid": 1, "theta_face": -35.0, "face_scale": 0.11, "person_h": 0.75,
                    "hand_valid": 1, "fist_prob": 0.90, "theta_fist": -42.0, "knock_fired": 0,
                },
            )
        self.assertEqual(ctrl.mode, MODE_APPROACH)
        self.assertAlmostEqual(ctrl.target_lock.bearing_body, 28.0, delta=0.5)

        # 3) 超过 0.65s 冷却期后，在 -36° 触发新的双敲 (knock_fired=1) -> 必须立即重锁定到 -36°！
        ctrl.step(
            dt=0.05,
            telemetry=t0,
            scene_obs={
                "face_valid": 1, "theta_face": -20.0, "face_scale": 0.11, "person_h": 0.75,
                "hand_valid": 1, "fist_prob": 0.92, "theta_fist": -36.0, "knock_fired": 1,
            },
        )
        self.assertEqual(ctrl.mode, MODE_APPROACH)
        self.assertAlmostEqual(ctrl.target_lock.bearing_body, -36.0, delta=0.5)

    def test_longitudinal_pass_clamps_effective_remaining_to_zero(self):
        """当机器人纵向越过目标垂面 (x_body <= 0, |y_body| < 0.15m) 时，有效剩余距离必须立即归零以防追尾打转。"""
        tracker = SpatialMemoryTracker()
        t0 = parse_status_json({"imu": {"yaw": 0.0}, "head": {"q": 0.0}, "wheel": {"x1": 0.0, "x2": 0.0}})
        tracker.step_odometry(0.05, t0)
        tracker.lock_xy(now_s=0.05, x_body_m=0.20, y_body_m=0.04)

        t1 = parse_status_json({"imu": {"yaw": 0.0}, "head": {"q": 0.0}, "wheel": {"x1": 0.23, "x2": 0.23}})
        st = tracker.step_odometry(0.05, t1)
        self.assertLessEqual(st.x_body_m, 0.0)
        self.assertEqual(st.remaining_m, 0.0)

    def test_feature_builder_wide_angle_and_clearance_encoding(self):
        """39 维特征槽 1 必须同时编码精细视线角 (33)、广角机身角 (34，±75° 不饱和) 与前向净空 (36)。"""
        telem = parse_status_json({"head": {"q": 15.0}})
        tracker = SpatialMemoryTracker()
        tracker.step_odometry(0.05, telem)
        lock_st = tracker.lock_polar(now_s=0.05, bearing_body_deg=60.0, dist_m=0.90)
        t_lock = lock_st.to_target_lock()

        feat = EndSideFeatureBuilder.build(telem, detections=[], target_lock=t_lock, clearance_m=0.45)
        # 33: clamp(0.5 + (60 - 15)/60, 0, 1) = 1.0
        self.assertAlmostEqual(float(feat[33]), 1.0, places=5)
        # 34: clamp(0.5 + 60 / 180, 0, 1) = 0.833333 (不饱和！)
        self.assertAlmostEqual(float(feat[34]), 0.5 + 60.0 / 180.0, places=5)
        # 35: 0.90 / 1.5 = 0.60
        self.assertAlmostEqual(float(feat[35]), 0.60, places=5)
        # 36: clearance 0.45 / 1.0 = 0.45
        self.assertAlmostEqual(float(feat[36]), 0.45, places=5)
        # 37 & 38: 1.0
        self.assertAlmostEqual(float(feat[37]), 1.0, places=5)
        self.assertAlmostEqual(float(feat[38]), 1.0, places=5)

    def test_residual_policy_tflite_export_numerical_parity(self):
        """残差增强版 EndSidePolicyNet(hidden_dim=160, use_residual=True) 导出 TFLite v3 误差必须 < 1e-4。"""
        policy = EndSidePolicyNet(feature_dim=39, hidden_dim=160, action_dim=3, use_residual=True)
        policy.eval()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = export_policy_bundle(
                policy,
                out_dir=Path(tmp),
                model_name="hover-spatial-rl-v2-test",
                version="2026.11.18",
            )
            max_diff = verify_tflite_numerical(policy, Path(bundle["tflite_path"]), n_samples=24)
            self.assertLess(max_diff, 1e-4)

    def test_sandbox_arena_all_8_complex_scenarios(self):
        """8 大复杂物理沙箱场景必须全部可重置、可闭环推进且产出合法遥测、净空与事件流。"""
        self.assertEqual(len(SCENARIO_NAMES), 8)
        for idx, sc_name in enumerate(SCENARIO_NAMES):
            env = HoverSandboxEnv(scenario=sc_name, seed=100 + idx)
            obs = env.reset()
            self.assertIn("telemetry", obs)
            self.assertIn("scene_obs", obs)
            self.assertIn("clearance_m", obs)
            fired_count = 0
            for _ in range(80):
                obs, info = env.step(DriveCommand(vx=20.0, vyaw=8.0, head=5.0))
                if obs["scene_obs"].get("knock_fired", 0):
                    fired_count += 1
                self.assertTrue(math.isfinite(obs["telemetry"].pitch))
                self.assertGreaterEqual(obs["clearance_m"], 0.0)
            self.assertGreaterEqual(fired_count, 1, msg=f"Scenario {sc_name} never fired knock event")


if __name__ == "__main__":
    unittest.main()
