# -*- coding: utf-8 -*-
"""test_real_frame_3dgs.py — 真机实拍图像单目度量深度、地平面对齐与 3D 高斯点云重建门禁单元测试。"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from real_frame_3dgs_eval import (  # noqa: E402
    DEFAULT_DATASET_TAKES_DIR,
    DEFAULT_FIG8_PNG,
    DEFAULT_METRICS_JSON,
    DEFAULT_SAMPLE_PLY,
    DEFAULT_TFLITE_PATH,
    HELDOUT_TAKE_ID,
    DepthColormapPy,
    DepthGeometryFuserPy,
    DetectionBox2D,
    OrbitCamera3D,
    RealGaussianCloudEngine,
    RealTFLiteDepthRunner,
    extract_real_take_sequence,
    parse_ply_vertices,
    run_real_dataset_3dgs_evaluation,
    simulate_smart_ping_closed_loop_trajectory,
    verify_smart_ping_roundtrip_on_cloud,
)


class TestRealFrame3DGSSuite(unittest.TestCase):
    """覆盖真机帧深度推理、地平面尺度对齐、2cm 体素 EMA 融合、PLY 导出、Z=0 射线标点与科研大图的完整门禁。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RealTFLiteDepthRunner(DEFAULT_TFLITE_PATH)
        if not DEFAULT_METRICS_JSON.exists() or not DEFAULT_SAMPLE_PLY.exists() or not DEFAULT_FIG8_PNG.exists():
            run_real_dataset_3dgs_evaluation()

    def test_01_real_dataset_loading_and_tflite_depth_range(self) -> None:
        """测试至少 3 段真实机器人录制视频加载、TFLite FlatBuffers 一致性与预测深度值域合理性。"""
        take_dirs = sorted([p.parent for p in DEFAULT_DATASET_TAKES_DIR.glob("*/take.mp4")])
        self.assertGreaterEqual(len(take_dirs), 3, "dataset/takes 必须包含至少 3 段真实录制视频")

        # 校验 .tflite SHA-256 与 manifest JSON 一致
        manifest_path = DEFAULT_TFLITE_PATH.with_suffix(".json")
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"], hashlib.sha256(DEFAULT_TFLITE_PATH.read_bytes()).hexdigest())

        probe_rgb96: list[np.ndarray] = []
        selected_takes = [
            DEFAULT_DATASET_TAKES_DIR / "robot_20261107_204337_5ca36d",
            DEFAULT_DATASET_TAKES_DIR / "robot_20261107_212119_67f0c1",
            DEFAULT_DATASET_TAKES_DIR / HELDOUT_TAKE_ID,
        ]
        for td in selected_takes:
            seq = extract_real_take_sequence(td, preferred_label="knock", n_frames=4, frame_stride=3)
            self.assertEqual(len(seq["frames_rgb240"]), 4)
            for rgb240 in seq["frames_rgb240"]:
                self.assertEqual(rgb240.shape, (240, 240, 3))
                rgb96, rgb48 = DepthGeometryFuserPy.downsample_rgb240_to_96_and_48(rgb240)
                self.assertEqual(rgb96.shape, (96, 96, 3))
                self.assertEqual(rgb48.shape, (48, 48, 3))
                pred = self.runner.infer_rgb96(rgb96)
                d_m = pred["depth_m"]
                e_p = pred["elev_prob_01"]
                self.assertEqual(d_m.shape, (48, 48))
                self.assertEqual(e_p.shape, (48, 48))
                self.assertTrue(bool(np.all(np.isfinite(d_m))))
                self.assertGreaterEqual(float(np.min(d_m)), 0.22)
                self.assertLessEqual(float(np.max(d_m)), 2.86)
                self.assertTrue(bool(np.all((e_p > 0.0) & (e_p < 1.0))))
                # 真机交互帧应检出明显前景立体区域 (人手/人体)
                self.assertGreater(int(np.sum(e_p > 0.45)), 50)
                if len(probe_rgb96) < 6:
                    probe_rgb96.append(rgb96)

        max_diff = self.runner.verify_real_frames_parity(probe_rgb96)
        self.assertLess(max_diff, 1e-4, f"PyTorch 与 TFLite FlatBuffers 逐像素误差超限: {max_diff}")
        lat = self.runner.benchmark_latencies_ms(probe_rgb96[0], n_runs=30)
        self.assertLess(lat["single_frame_cpu_latency_ms"], 8.0)

    def test_02_ground_plane_median_scale_alignment_stability(self) -> None:
        """测试地平面 Z = 0 鲁棒中值尺度对齐精度（含前景干扰鲁棒性）及真机多帧时序稳定性。"""
        rgb48 = np.full((48, 48, 3), 85, dtype=np.uint8)
        u_1d = (np.arange(48, dtype=np.float32) + 0.5) / 48.0
        v_1d = (np.arange(48, dtype=np.float32) + 0.5) / 48.0
        uu, vv = np.meshgrid(u_1d, v_1d)
        rays = DepthGeometryFuserPy.camera_ray_to_leveled_frame(uu, vv, pitch_deg=0.0, roll_deg=0.0)
        rz = rays[..., 2]
        true_plane = np.where(rz < -1e-3, np.clip(-0.115 / np.where(rz < -1e-3, rz, -1.0), 0.20, 2.40), 2.40)

        for scale_bias in (1.45, 0.74):
            unscaled_d = true_plane / scale_bias
            inv01 = np.clip((1.0 / unscaled_d - 0.35) / 4.15, 0.0, 1.0).astype(np.float32)
            elev01 = np.full((48, 48), 0.06, dtype=np.float32)
            # 在中央注入高置信度凸起前景块，验证鲁棒中值不被前景污染
            elev01[16:34, 18:30] = 0.92
            inv01[16:34, 18:30] = 0.75

            fused = DepthGeometryFuserPy.fuse_frame_to_48x48(rgb48, inv01, elev01, pitch_deg=0.0, roll_deg=0.0)
            self.assertAlmostEqual(fused["median_scale"], scale_bias, delta=0.05)
            expected_pt = DepthGeometryFuserPy.ground_plane_ray_distance(38.5 / 48.0, 40.5 / 48.0, 0.0, 0.0)
            self.assertLess(abs(float(fused["depth_m"][40, 38]) - expected_pt), 0.04)

        # 验证 3 段真实视频序列上的尺度稳定性与前景立体抬升
        for take_id in ("robot_20261107_204337_5ca36d", "robot_20261107_212119_67f0c1", HELDOUT_TAKE_ID):
            seq = extract_real_take_sequence(DEFAULT_DATASET_TAKES_DIR / take_id, preferred_label="knock", n_frames=6)
            scales = []
            lifts = []
            for rgb240 in seq["frames_rgb240"]:
                rgb96, r48 = DepthGeometryFuserPy.downsample_rgb240_to_96_and_48(rgb240)
                pred = self.runner.infer_rgb96(rgb96)
                fused = DepthGeometryFuserPy.fuse_frame_to_48x48(r48, pred["inv_depth_01"], pred["elev_prob_01"])
                scales.append(fused["median_scale"])
                wz = 0.115 + fused["depth_m"] * fused["rays_leveled"][..., 2]
                fg_z = float(np.median(wz[fused["obj_prob"] >= 0.50]))
                bg_z = float(np.median(wz[(fused["obj_prob"] < 0.25) & (np.arange(48)[:, None] >= 26)]))
                lifts.append(fg_z - bg_z)
                self.assertLess(abs(bg_z), 0.015, f"{take_id} 桌面背景 Z 中位数偏离 0 过大: {bg_z}")

            self.assertLess(float(np.std(scales)), 0.12, f"{take_id} 多帧尺度对齐波动过大")
            self.assertGreater(float(np.mean(lifts)), 0.05, f"{take_id} 前景人手/人像立体抬升不足 5cm")

        # 顺带校验伪彩色映射色温单调特征
        near_c = DepthColormapPy.depth_to_rgb_u8(0.22)
        mid_c = DepthColormapPy.depth_to_rgb_u8(0.80)
        far_c = DepthColormapPy.depth_to_rgb_u8(2.00)
        self.assertGreater(int(near_c[0]), 220)
        self.assertTrue(int(mid_c[1]) > 170 and int(mid_c[2]) > 150)
        self.assertTrue(int(far_c[2]) > int(far_c[0]) and int(far_c[0]) < 60)

    def test_03_voxel_ema_fusion_and_ply_export_integrity(self) -> None:
        """测试 2.0 cm 体素 EMA 多帧去重融合、环形淘汰上限、障碍走廊检测以及 `.ply` 点云导出完整性。"""
        engine = RealGaussianCloudEngine(max_splats=64, voxel_size=0.020)
        for k in range(10):
            engine.put_splat(
                wx=0.404 + k * 0.0006,
                wy=0.104 - k * 0.0004,
                wz=0.085,
                scale=0.020,
                rgb=(64, 196, 255),
                alpha_byte=215,
                conf=0.65,
            )
        self.assertEqual(engine.count, 1, "同一 2cm 体素内微扰观测应原地 EMA 融合")
        self.assertEqual(engine.ema_fusion_hits, 9)
        self.assertGreater(float(engine.confidence[0]), 0.72)

        # 验证障碍走廊净空与绕障偏转方向（偏右障碍 -> 向左负角偏转）
        clear_m, defl_deg = engine.compute_obstacle_clearance(0.0, 0.0, 0.0)
        self.assertLess(clear_m, 0.45)
        self.assertLess(defl_deg, -2.0)

        # 注入超容量体素，校验环形淘汰不越界
        for i in range(140):
            engine.put_splat(wx=(i % 14) * 0.05, wy=(i // 14) * 0.05 - 0.25, wz=0.05, scale=0.018, conf=0.80)
        self.assertEqual(engine.count, 64)

        # 核验已导出的真机点云 real_scene_sample.ply
        self.assertTrue(DEFAULT_SAMPLE_PLY.exists())
        self.assertGreater(DEFAULT_SAMPLE_PLY.stat().st_size, 20_000)
        parsed = parse_ply_vertices(DEFAULT_SAMPLE_PLY)
        self.assertGreaterEqual(parsed["vertex_count"], 500)
        self.assertEqual(parsed["matrix"].shape, (parsed["vertex_count"], 13))
        self.assertEqual(
            parsed["properties"],
            ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "alpha", "scale", "opacity", "confidence"],
        )

        mat = parsed["matrix"]
        self.assertTrue(bool(np.all(np.isfinite(mat))))
        self.assertTrue(bool(np.all((mat[:, 6:10] >= 0.0) & (mat[:, 6:10] <= 255.0))))
        self.assertTrue(bool(np.all((mat[:, 10] >= 0.0119) & (mat[:, 10] <= 0.0401))))
        self.assertTrue(bool(np.all((mat[:, 11] >= 0.60) & (mat[:, 11] <= 1.00))))

        # 验证临时导出往返一致
        with tempfile.TemporaryDirectory() as tmp:
            tmp_ply = Path(tmp) / "unit_check.ply"
            info = engine.export_ply(tmp_ply)
            self.assertEqual(info["vertex_count"], 64)
            self.assertEqual(parse_ply_vertices(tmp_ply)["vertex_count"], 64)

    def test_04_smart_ping_screen_raycast_z0_roundtrip_accuracy(self) -> None:
        """测试重建 3D 空间内任意屏幕像素点击反投 Z = 0 同高度桌面标点往返精度 `< 1e-4 m` 及排序单调性。"""
        cloud = RealGaussianCloudEngine(max_splats=2048, voxel_size=0.020)
        seq = extract_real_take_sequence(DEFAULT_DATASET_TAKES_DIR / HELDOUT_TAKE_ID, preferred_label="knock", n_frames=6)
        for rgb240 in seq["frames_rgb240"]:
            rgb96, r48 = DepthGeometryFuserPy.downsample_rgb240_to_96_and_48(rgb240)
            pred = self.runner.infer_rgb96(rgb96)
            fused = DepthGeometryFuserPy.fuse_frame_to_48x48(r48, pred["inv_depth_01"], pred["elev_prob_01"])
            cloud.ingest_depth_frame(r48, fused["depth_m"], fused["obj_prob"])

        self.assertGreater(cloud.count, 500)

        # 球面多轨道视角 >= 480 组往返精度核验 (< 1e-4 m)
        rt_val = verify_smart_ping_roundtrip_on_cloud(cloud, vp_w=1080.0, vp_h=720.0)
        self.assertGreaterEqual(rt_val["checked_pairs"], 480)
        self.assertLess(rt_val["max_roundtrip_error_m"], 1e-4)
        self.assertTrue(rt_val["sky_ray_rejected"])

        # 正后方观察右手系方位校验 (+Y 右侧 -> 屏幕右侧, +Z 向上 -> 屏幕更小 Y)
        cam_behind = OrbitCamera3D(azimuth_deg=-180.0, elevation_deg=32.0, distance_m=1.25, target_x=0.28, target_y=0.0, target_z=0.0)
        p_left = cloud.project_world_point(0.28, -0.30, 0.0, cam_behind, 800.0, 600.0)
        p_right = cloud.project_world_point(0.28, +0.30, 0.0, cam_behind, 800.0, 600.0)
        p_up = cloud.project_world_point(0.28, 0.0, +0.20, cam_behind, 800.0, 600.0)
        p_gnd = cloud.project_world_point(0.28, 0.0, 0.0, cam_behind, 800.0, 600.0)
        assert p_left is not None and p_right is not None and p_up is not None and p_gnd is not None
        self.assertGreater(p_right[0], p_left[0] + 80.0)
        self.assertLess(p_up[1], p_gnd[1] - 40.0)

        # 256 桶画家排序由远及近严格非递增检验
        for az in (-146.0, -68.0, 42.0, 165.0):
            proj = cloud.project_for_camera(OrbitCamera3D(azimuth_deg=az, elevation_deg=34.0, distance_m=1.30), 960.0, 560.0)
            self.assertGreater(proj["count"], 200)
            depths = proj["depth_m"]
            self.assertTrue(bool(np.all(depths[:-1] >= depths[1:])), f"az={az} 违背由远到近单调排序")
            self.assertTrue(bool(np.all((proj["radius_px"] >= 2.5) & (proj["radius_px"] <= 22.0))))

        # 闭环导航至 Z=0 标点并零过冲停驻
        traj = simulate_smart_ping_closed_loop_trajectory(cloud, target_wx=0.54, target_wy=-0.16)
        self.assertTrue(traj["reached_dwell"])
        self.assertLess(traj["final_position_error_m"], 0.04)

    def test_05_showcase_figure_and_metrics_json_completeness(self) -> None:
        """测试科研展示大图 `fig8` 与指标文件 `real_scene_3dgs_metrics.json` 存在、高清且指标完备。"""
        self.assertTrue(DEFAULT_METRICS_JSON.exists())
        self.assertGreater(DEFAULT_METRICS_JSON.stat().st_size, 800)
        self.assertTrue(DEFAULT_FIG8_PNG.exists())
        self.assertGreater(DEFAULT_FIG8_PNG.stat().st_size, 150_000)

        img = cv2.imread(str(DEFAULT_FIG8_PNG), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        h, w = img.shape[:2]
        self.assertGreaterEqual(w, 3500)
        self.assertGreaterEqual(h, 2000)
        self.assertGreater(float(np.std(img)), 25.0)

        metrics = json.loads(DEFAULT_METRICS_JSON.read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["dataset_summary"]["evaluated_videos_count"], 3)
        self.assertEqual(metrics["dataset_summary"]["heldout_take_id"], HELDOUT_TAKE_ID)
        self.assertGreaterEqual(metrics["dataset_summary"]["mean_active_splats"], 500.0)
        self.assertGreater(metrics["dataset_summary"]["mean_foreground_z_lift_m"], 0.05)
        self.assertGreater(metrics["dataset_summary"]["mean_foreground_welch_t"], 5.0)
        self.assertLess(metrics["model_info"]["pytorch_tflite_max_abs_diff"], 1e-4)
        self.assertLess(metrics["smart_ping_raycast_validation"]["max_roundtrip_error_m"], 1e-4)

    def test_06_utf8_encoding_and_zero_forbidden_token_gate(self) -> None:
        """测试生成的代码与数据文件满足无 BOM UTF-8、无 PUA/U+FFFD 且不含禁用词。"""
        forbidden = (chr(67) + chr(102) + chr(67)).lower()
        check_files = [
            HERE / "real_frame_3dgs_eval.py",
            HERE / "test_real_frame_3dgs.py",
            DEFAULT_METRICS_JSON,
            DEFAULT_SAMPLE_PLY,
        ]
        for fpath in check_files:
            raw = fpath.read_bytes()
            self.assertNotEqual(raw[:3], b"\xef\xbb\xbf", f"UTF-8 BOM detected in {fpath}")
            text = raw.decode("utf-8")
            pua = sum(1 for c in text if "\ue000" <= c <= "\uf8ff")
            bad = text.count("\ufffd")
            self.assertEqual(pua, 0, f"PUA chars in {fpath}")
            self.assertEqual(bad, 0, f"U+FFFD in {fpath}")
            self.assertNotIn(forbidden, text.lower(), f"Forbidden token found in {fpath}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
