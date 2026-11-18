# -*- coding: utf-8 -*-
"""test_telemetry_flywheel.py — 遥测黑匣子回放、反事实评估、Huber/IRLS 系统辨识、PLY 导出与上传接口单测。"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    MODE_GAZE_FACE,
    EndSideFeatureBuilder,
    EndSidePolicyNet,
    TargetLock,
)
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from export_tflite_bundle import export_policy_bundle  # noqa: E402
from model_push_server import (  # noqa: E402
    DEFAULT_SEARCH_DIRS,
    discover_model_bundles,
    start_model_push_server,
)
from telemetry_flywheel import (  # noqa: E402
    BlackboxRecord,
    auto_calibrate_system_id,
    dump_blackbox_jsonl,
    evaluate_counterfactual_tflite,
    export_gaussian_ply,
    parse_blackbox_jsonl,
    parse_blackbox_record,
    serialize_blackbox_jsonl,
)


def _make_calibration_trajectory(
    true_dist_k: float = 0.235,
    true_lag_m: float = 0.044,
    true_deadband_deg: float = 3.2,
    true_yaw_gain: float = 0.92,
    n_frames: int = 80,
    seed: int = 20261118,
) -> list[BlackboxRecord]:
    """构造含高斯噪声与 12% 恶劣离群野值的合成接近 + 偏航扫描黑匣子日志。"""
    rng = np.random.default_rng(seed)
    records: list[BlackboxRecord] = []
    dt_s = 0.05  # 20Hz
    d0_m = 1.05
    x_w = 0.0
    psi_body = 0.0
    head_q = 4.0

    for k in range(n_frames):
        t_ms = k * 50
        # 前进速度 0.12 m/s，使累计里程约 0.48m
        v_mps = 0.12 if k < 65 else 0.0
        if k > 0:
            x_w += v_mps * dt_s

        # 扫描不同幅值的偏航指令 (覆盖死区内 0..3.2°/s 与死区外 4..36°/s)
        u_dps = float((k % 20) * 1.8)
        sgn_u = 1.0 if (k // 20) % 2 == 0 else -1.0
        u_pct = (sgn_u * u_dps) / 0.90

        # 上一帧偏航指令驱动本帧角度增量
        if k > 0:
            prev_u_dps = abs(records[-1].cmd.vyaw) * 0.90
            prev_sgn = 1.0 if records[-1].cmd.vyaw >= 0 else -1.0
            eff_rate = prev_sgn * true_yaw_gain * max(0.0, prev_u_dps - true_deadband_deg)
            noise_w = float(rng.normal(0.0, 0.08))
            if k in (17, 43, 61):  # 陀螺毛刺离群点
                noise_w += 28.0
            psi_body += (eff_rate + noise_w) * dt_s

        rem_true = max(0.22, d0_m - x_w)
        h_clean = true_dist_k / rem_true
        h_meas = float(np.clip(h_clean + rng.normal(0.0, 0.002), 0.05, 0.95))
        if k in (9, 25, 38, 52):  # 视觉框遮挡/路人跳变野值
            h_meas = 0.88 if k % 2 == 0 else 0.07

        lag_meas = true_lag_m + float(rng.normal(0.0, 0.0015))
        if k in (14, 33, 49):  # 车轮瞬时打滑野值
            lag_meas = 0.29

        cmd_vx_pct = (v_mps / 0.30) * 100.0
        cmd_pos = x_w + (lag_meas if cmd_vx_pct > 5.0 else 0.0)
        pitch_deg = -2.4 + 0.6 * math.sin(0.2 * k)

        st = RobotTelemetry(
            t_ms=t_ms,
            pitch=pitch_deg,
            yaw=psi_body + head_q,
            v1=v_mps * 100.0,
            v2=v_mps * 100.0,
            x1=x_w,
            x2=x_w,
            head_q=head_q,
            head_target=head_q,
            cmd_vx=cmd_vx_pct,
            cmd_vyaw=u_pct,
            cmd_mode=1,
            cmd_pos=cmd_pos,
            pos_e=(x_w - cmd_pos) * 100.0,
        )
        det = {"cx": 0.50, "cy": 0.42, "w": h_meas * 0.75, "h": h_meas, "score": 0.91, "cls": 0}
        t_lock = TargetLock(valid=1, bearing_body=0.0, dist_m=d0_m, remaining_m=max(0.0, rem_true - 0.25))
        cmd = DriveCommand(vx=cmd_vx_pct, vyaw=u_pct, head=head_q)
        records.append(
            BlackboxRecord(
                t_ms=t_ms,
                status=st,
                detections=[det],
                target_lock=t_lock,
                cmd=cmd,
                mode=MODE_APPROACH,
                clearance_m=1.0,
            )
        )
    return records


class TestTelemetryFlywheel(unittest.TestCase):
    def test_blackbox_jsonl_roundtrip_and_flexible_camel_case(self):
        """黑匣子解析器应支持标准嵌套 JSONL 往返及 Android 驼峰字段格式。"""
        recs = _make_calibration_trajectory(n_frames=8)
        jsonl_str = serialize_blackbox_jsonl(recs)
        parsed = parse_blackbox_jsonl(jsonl_str)
        self.assertEqual(len(parsed), 8)
        self.assertEqual(parsed[3].t_ms, recs[3].t_ms)
        self.assertAlmostEqual(parsed[3].status.x1, recs[3].status.x1, places=4)
        self.assertEqual(parsed[3].mode, MODE_APPROACH)
        self.assertIsNotNone(parsed[3].target_lock)

        # 验证驼峰扁平记录兼容性
        camel_line = json.dumps(
            {
                "t": 1250,
                "mode": "GAZE_FACE",
                "clearanceM": 0.65,
                "status": {
                    "pitch": -1.8,
                    "yaw": 22.0,
                    "headQ": 7.5,
                    "headTarget": 8.0,
                    "cmdVx": 0.0,
                    "cmdVyaw": 12.0,
                    "x1": 0.32,
                    "x2": 0.32,
                },
                "detections": [{"cx": 0.62, "cy": 0.36, "w": 0.20, "h": 0.28, "conf": 0.89, "classId": 0}],
                "targetLock": {"valid": 1, "bearingBodyDeg": 14.0, "distM": 0.60, "remainingM": 0.42},
                "cmd": {"vx": 0.0, "vyaw": 12.0, "head": 8.0},
            }
        )
        r_cam = parse_blackbox_record(camel_line)
        self.assertEqual(r_cam.t_ms, 1250)
        self.assertEqual(r_cam.mode, MODE_GAZE_FACE)
        self.assertAlmostEqual(r_cam.status.head_q, 7.5, places=4)
        self.assertAlmostEqual(r_cam.clearance_m, 0.65, places=4)
        self.assertIsNotNone(r_cam.target_lock)
        self.assertAlmostEqual(r_cam.target_lock.remaining_m, 0.42, places=4)

    def test_counterfactual_tflite_replay_self_consistency_and_shield(self):
        """反事实回放同一策略应实现 < 1e-3 L1 零偏差，回放激进策略应准确触发安全屏障统计。"""
        torch.manual_seed(7)
        policy_a = EndSidePolicyNet(feature_dim=39, hidden_dim=128, action_dim=3, use_residual=True)
        policy_a.eval()

        policy_aggressive = EndSidePolicyNet(feature_dim=39, hidden_dim=128, action_dim=3, use_residual=True)
        with torch.no_grad():
            policy_aggressive.fc3.bias.copy_(torch.tensor([2.5, 1.8, -1.2]))
        policy_aggressive.eval()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exp_a = export_policy_bundle(policy_a, tmp_path / "pol_a", model_name="test-pol-a")
            exp_agg = export_policy_bundle(policy_aggressive, tmp_path / "pol_agg", model_name="test-pol-agg")

            # 使用 policy_a 生成带真实推理动作的 GAZE_FACE 轨迹
            recs: list[BlackboxRecord] = []
            for k in range(16):
                st = RobotTelemetry(t_ms=k * 50, pitch=3.0 if k % 2 == 0 else -4.0, head_q=float(k - 8))
                dets = [{"cx": 0.45 + 0.01 * k, "cy": 0.40, "w": 0.22, "h": 0.28, "score": 0.90, "cls": 0}]
                feat = EndSideFeatureBuilder.build(st, dets, target_lock=None, clearance_m=0.8)
                cmd_a = policy_a.infer_numpy(feat)
                recs.append(BlackboxRecord(t_ms=k * 50, status=st, detections=dets, cmd=cmd_a, mode=MODE_GAZE_FACE))

            log_file = dump_blackbox_jsonl(recs, tmp_path / "trace.jsonl")

            res_self = evaluate_counterfactual_tflite(log_file, exp_a["tflite_path"])
            self.assertEqual(res_self["frame_count"], 16)
            self.assertLess(res_self["mean_action_l1_diff"], 1e-3)
            self.assertAlmostEqual(res_self["pitch_rms_deg"], math.sqrt(0.5 * (9.0 + 16.0)), places=3)

            # 激进策略在 MODE_GAZE_FACE 下输出大前向 vx，必然被 EndSideSafetyShield 拦截清零
            res_agg = evaluate_counterfactual_tflite(recs, exp_agg["tflite_path"])
            self.assertGreater(res_agg["mean_action_l1_diff"], 20.0)
            self.assertGreater(res_agg["shield_intervention_rate"], 0.85)
            self.assertGreaterEqual(res_agg["jerk_index"], 0.0)

    def test_auto_calibrate_system_id_huber_irls_with_outliers(self):
        """Huber/IRLS 系统辨识应在 12% 离群野值干扰下精确恢复 dist_k、lqr_pos_lag_m 与 yaw_deadband_deg。"""
        recs = _make_calibration_trajectory(
            true_dist_k=0.235,
            true_lag_m=0.044,
            true_deadband_deg=3.2,
            n_frames=80,
        )
        cal = auto_calibrate_system_id(recs)
        self.assertTrue(cal["converged"])
        self.assertAlmostEqual(cal["dist_k"], 0.235, delta=0.010)
        self.assertAlmostEqual(cal["lqr_pos_lag_m"], 0.044, delta=0.004)
        self.assertAlmostEqual(cal["yaw_deadband_deg"], 3.2, delta=0.35)

    def test_export_gaussian_ply_formats(self):
        """3D 高斯点云导出应输出合法 ASCII PLY 头并自动过滤非有限坐标与归一化颜色。"""
        splats_np = np.array(
            [
                [0.25, -0.10, 0.02, 0.95, 0.40, 0.15, 0.018, 0.92],
                [0.48, 0.12, 0.05, 0.20, 0.80, 0.65, 0.014, 0.88],
                [float("nan"), 0.0, 0.0, 1.0, 1.0, 1.0, 0.01, 0.5],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "cloud" / "scene_splats.ply"
            meta = export_gaussian_ply(splats_np, ply_path)
            self.assertTrue(ply_path.is_file())
            self.assertEqual(meta["vertex_count"], 2)
            lines = ply_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "ply")
            self.assertIn("element vertex 2", lines)
            self.assertIn("end_header", lines)
            hdr_end = lines.index("end_header")
            self.assertEqual(len(lines) - hdr_end - 1, 2)

            # 字典列表格式校验
            ply_dict_path = Path(tmp) / "dict_splats.ply"
            meta2 = export_gaussian_ply(
                [{"x": 0.1, "y": 0.2, "z": 0.0, "r": 255, "g": 128, "b": 64, "radius": 0.02}],
                ply_dict_path,
            )
            self.assertEqual(meta2["vertex_count"], 1)

    def test_model_push_server_recursive_zoo_and_telemetry_endpoints(self):
        """推送服务应递归发现子目录模型包，并支持 POST /telemetry/upload 与 GET /telemetry/list。"""
        dir_names = {p.name for p in DEFAULT_SEARCH_DIRS}
        self.assertIn("mono_depth", dir_names)
        self.assertIn("skill_zoo", dir_names)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested_skill_dir = root / "skill_zoo" / "orbit_sub"
            up_dir = root / "uploaded_logs"

            pol = EndSidePolicyNet(feature_dim=39, hidden_dim=128, action_dim=3, use_residual=True)
            pol.eval()
            exp = export_policy_bundle(pol, nested_skill_dir, model_name="hover-skill-test-v1")

            found = discover_model_bundles([root / "skill_zoo"])
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["name"], "hover-skill-test-v1")

            srv = start_model_push_server(
                host="127.0.0.1",
                port=0,
                search_dirs=[root / "skill_zoo"],
                upload_dir=up_dir,
            )
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                base = f"http://127.0.0.1:{srv.server_port}"
                # 1. GET /models.json
                cat = json.loads(urllib.request.urlopen(f"{base}/models.json", timeout=3).read().decode("utf-8"))
                self.assertEqual(len(cat["models"]), 1)
                self.assertEqual(cat["models"][0]["filename"], Path(exp["zip_path"]).name)

                # 2. POST /telemetry/upload
                recs = _make_calibration_trajectory(true_dist_k=0.220, true_lag_m=0.038, true_deadband_deg=2.6, n_frames=50)
                body_bytes = serialize_blackbox_jsonl(recs).encode("utf-8")
                req = urllib.request.Request(
                    f"{base}/telemetry/upload",
                    data=body_bytes,
                    headers={"Content-Type": "application/x-ndjson"},
                    method="POST",
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
                self.assertTrue(resp["ok"])
                self.assertEqual(resp["frameCount"], 50)
                self.assertTrue(Path(resp["savedPath"]).is_file())
                self.assertAlmostEqual(resp["calibration"]["dist_k"], 0.220, delta=0.012)

                # 3. GET /telemetry/list
                lst = json.loads(urllib.request.urlopen(f"{base}/telemetry/list", timeout=3).read().decode("utf-8"))
                self.assertTrue(lst["ok"])
                self.assertEqual(len(lst["logs"]), 1)
                self.assertEqual(lst["logs"][0]["frameCount"], 50)
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
