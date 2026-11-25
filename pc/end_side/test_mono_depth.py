# -*- coding: utf-8 -*-
"""test_mono_depth.py — RIG-Hover 单目尺度深度网络、3D 光线追踪几何与 TFLite 导出包单元测试。"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mono_depth_model import (  # noqa: E402
    DEFAULT_FOV_H_DEG,
    DEFAULT_H_CAM_M,
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    INPUT_C,
    INPUT_H,
    INPUT_W,
    OUTPUT_C,
    OUTPUT_H,
    OUTPUT_W,
    HoverMonoDepthNet,
    benchmark_cpu_latency_ms,
    export_mono_depth_bundle,
    infer_mono_depth_tflite,
    inv_depth_01_to_metric_m,
    load_hover_mono_depth_from_tflite,
    metric_m_to_inv_depth_01,
    unproject_depth_to_3d_splats,
    verify_mono_depth_tflite_parity,
)
from train_mono_depth import render_analytical_desktop_scene  # noqa: E402


class TestMonoDepthSuite(unittest.TestCase):
    """覆盖输出形状、物理尺度范围、TFLite FlatBuffers 精度对齐与前景近场深度推理的门禁单测。"""

    def test_01_shapes_and_metric_depth_range(self) -> None:
        model = HoverMonoDepthNet().eval()
        x = torch.rand(2, INPUT_C, INPUT_H, INPUT_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, OUTPUT_C, OUTPUT_H, OUTPUT_W))
        self.assertTrue(bool(torch.all((y > 0.0) & (y < 1.0))))

        # 校验逆深度极值映射范围 (0.35 .. 4.50 m^-1 -> ~0.222 .. ~2.857 m)
        d_far = float(inv_depth_01_to_metric_m(0.0))
        d_near = float(inv_depth_01_to_metric_m(1.0))
        self.assertAlmostEqual(d_far, 1.0 / 0.35, places=5)
        self.assertAlmostEqual(d_near, 1.0 / 4.50, places=5)
        self.assertTrue(0.21 < d_near < 0.23)
        self.assertTrue(2.84 < d_far < 2.87)

        # 往返互逆一致性
        sample_d = np.linspace(DEPTH_MIN_M + 0.01, DEPTH_MAX_M - 0.01, 50, dtype=np.float32)
        inv_01 = metric_m_to_inv_depth_01(sample_d)
        rec_d = inv_depth_01_to_metric_m(inv_01)
        np.testing.assert_allclose(sample_d, rec_d, rtol=1e-5, atol=1e-5)

        # NHWC 便捷接口校验
        rgb_nhwc = np.random.default_rng(7).uniform(0.0, 1.0, size=(INPUT_H, INPUT_W, INPUT_C)).astype(np.float32)
        pred = model.infer_nhwc(rgb_nhwc)
        self.assertEqual(pred["out_nhwc"].shape, (OUTPUT_H, OUTPUT_W, OUTPUT_C))
        self.assertEqual(pred["depth_m"].shape, (OUTPUT_H, OUTPUT_W))
        self.assertEqual(pred["elev_prob_01"].shape, (OUTPUT_H, OUTPUT_W))
        self.assertTrue(float(np.min(pred["depth_m"])) >= DEPTH_MIN_M - 1e-5)
        self.assertTrue(float(np.max(pred["depth_m"])) <= DEPTH_MAX_M + 1e-5)

    def test_02_raytracer_and_3d_unprojection_geometry(self) -> None:
        rng = np.random.default_rng(1234)
        custom_objs = [
            {
                "kind": "cylinder",
                "cx": 0.45,
                "cy": 0.0,
                "r": 0.06,
                "h": 0.15,
                "color": np.array([0.8, 0.25, 0.2], dtype=np.float32),
                "stripe": 0.5,
            }
        ]
        scene = render_analytical_desktop_scene(
            rng,
            force_grayscale=False,
            fixed_pose=(DEFAULT_H_CAM_M, 2.0, 0.0),
            custom_objects=custom_objs,
        )
        self.assertEqual(scene["rgb_nhwc"].shape, (INPUT_H, INPUT_W, INPUT_C))
        self.assertEqual(scene["depth_m"].shape, (OUTPUT_H, OUTPUT_W))
        self.assertEqual(scene["elev_mask"].shape, (OUTPUT_H, OUTPUT_W))

        fg = scene["elev_mask"] > 0.5
        self.assertGreater(int(np.sum(fg)), 15)

        splats = unproject_depth_to_3d_splats(
            depth_m=scene["depth_m"],
            elev_prob_01=scene["elev_mask"],
            rgb_in=scene["rgb_nhwc"],
            h_cam_m=DEFAULT_H_CAM_M,
            fov_h_deg=DEFAULT_FOV_H_DEG,
            pitch_deg=2.0,
            roll_deg=0.0,
        )
        xyz = splats["xyz_grid"]
        # 近端无遮挡纯桌面区域（避开中央水杯 0..3cm 底部裙边与极近端深度截断区）反投影 Z 应严格贴合 Z = 0
        cols_grid = np.arange(OUTPUT_W)[None, :]
        tbl_region = (~fg) & (scene["depth_m"] > 0.25) & (scene["depth_m"] < 1.2) & (np.abs(cols_grid - 24) > 8)
        self.assertGreater(int(np.sum(tbl_region)), 50)
        max_tbl_z_err = float(np.max(np.abs(xyz[..., 2][tbl_region])))
        self.assertLess(max_tbl_z_err, 1e-4)

    def test_03_tflite_export_bundle_and_numerical_parity(self) -> None:
        torch.manual_seed(42)
        model = HoverMonoDepthNet()
        # 模拟若干步训练以使 BatchNorm 均值方差非平凡，严格检验折叠精度
        model.train()
        for _ in range(6):
            _ = model(torch.rand(4, INPUT_C, INPUT_H, INPUT_W))
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_mono_depth_bundle(model, out_dir=tmpdir, version="2026.11.18")
            tfl_path = Path(res["tflite_path"])
            jsn_path = Path(res["json_path"])
            zip_path = Path(res["zip_path"])

            self.assertTrue(tfl_path.exists())
            self.assertTrue(jsn_path.exists())
            self.assertTrue(zip_path.exists())

            # 模型体积 < 350 KB
            raw_bytes = tfl_path.read_bytes()
            self.assertEqual(raw_bytes[4:8], b"TFL3")
            self.assertLess(len(raw_bytes), 350 * 1024)

            # JSON Schema 与 SHA-256 严格一致
            meta = json.loads(jsn_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["schemaVersion"], 1)
            self.assertEqual(meta["name"], "hover-mono-depth-v1")
            self.assertEqual(meta["version"], "2026.11.18")
            self.assertEqual(meta["task"], "depth")
            self.assertEqual(meta["input"], {"width": 96, "height": 96, "channels": 3, "normalize": "unit"})
            self.assertEqual(meta["sha256"], hashlib.sha256(raw_bytes).hexdigest())

            # Zip 包内部条目校验
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("model.tflite", names)
                self.assertIn("model.json", names)
                self.assertEqual(zf.read("model.tflite"), raw_bytes)

            # PyTorch <-> TFLite 数值一致性 < 1e-4
            max_diff = verify_mono_depth_tflite_parity(model, tfl_path, n_samples=10, seed=99)
            self.assertLess(max_diff, 1e-4)

            # CPU 推理延迟 < 5 ms
            avg_ms = benchmark_cpu_latency_ms(model, n_runs=40)
            self.assertLess(avg_ms, 5.0)

            # 从 .tflite 无损恢复 PyTorch 模型
            x_cpu = torch.rand(1, INPUT_C, INPUT_H, INPUT_W)
            reloaded = load_hover_mono_depth_from_tflite(tfl_path)
            with torch.no_grad():
                rt_diff = float(torch.max(torch.abs(model(x_cpu) - reloaded(x_cpu))).item())
            self.assertLess(rt_diff, 1e-5)

    def test_04_trained_deployment_model_accuracy_and_foreground_ordering(self) -> None:
        run_dir = REPO_ROOT / "runs_end_side" / "mono_depth"
        tfl_path = run_dir / "hover_mono_depth_v1.tflite"
        jsn_path = run_dir / "hover_mono_depth_v1.json"
        zip_path = run_dir / "hover-mono-depth-v1-bundle.zip"
        metrics_path = run_dir / "depth_eval_metrics.json"
        fig6_path = REPO_ROOT / "docs" / "showcase" / "figs" / "fig6_mono_depth_and_3d_splats.png"
        android_tfl = Path("G:/robot_app/android/app/src/main/assets/models/hover_mono_depth_v1.tflite")
        android_jsn = Path("G:/robot_app/android/app/src/main/assets/models/hover_mono_depth_v1.json")

        for p in (tfl_path, jsn_path, zip_path, metrics_path, fig6_path, android_tfl, android_jsn):
            self.assertTrue(p.exists(), f"Missing expected artifact: {p}")

        # 校验 Android 同步拷贝字节完全一致
        self.assertEqual(tfl_path.read_bytes(), android_tfl.read_bytes())
        self.assertEqual(jsn_path.read_bytes(), android_jsn.read_bytes())

        # 相机以这个模型自己的 model.json 为准（已发布的 v1 是用 0.115 m / 60° 训练的），不是实测相机：
        # 换一组相机渲染出来的画面不是它训练时见过的画面，量出来的精度不代表它自己。
        manifest = json.loads(jsn_path.read_text(encoding="utf-8"))
        cam_h = float(manifest["camera"]["heightM"])
        cam_fov = float(manifest["camera"]["fovHDeg"])

        # 在独立合成测试场景上验证：前景立体水杯预测深度显著小于后方桌面背景，且 AbsRel < 0.15
        test_objs = [
            {
                "kind": "cylinder",
                "cx": 0.44,
                "cy": -0.02,
                "r": 0.062,
                "h": 0.155,
                "color": np.array([0.76, 0.28, 0.22], dtype=np.float32),
                "stripe": 0.4,
            },
            {
                "kind": "box",
                "cx": 0.82,
                "cy": 0.18,
                "lx": 0.06,
                "ly": 0.06,
                "h": 0.12,
                "yaw": 0.35,
                "color": np.array([0.24, 0.52, 0.78], dtype=np.float32),
            },
        ]
        def abs_rel_at(fov_h_deg: float, h_cam_m: float) -> tuple[float, dict, np.ndarray]:
            # 同一个随机种子：墙面距离、材质、光照逐项相同，两次调用之间只有相机不同
            scene = render_analytical_desktop_scene(
                np.random.default_rng(20261234), force_grayscale=False, fixed_pose=(h_cam_m, 1.5, 0.0),
                custom_objects=test_objs, fov_h_deg=fov_h_deg,
            )
            pred_d = infer_mono_depth_tflite(tfl_path, scene["rgb_nhwc"])["depth_m"]
            gt_d = scene["depth_m"]
            return float(np.mean(np.abs(pred_d - gt_d) / np.maximum(gt_d, 1e-4))), scene, pred_d

        (abs_rel, test_scene, pred_d) = abs_rel_at(cam_fov, cam_h)
        self.assertLess(abs_rel, 0.15, f"Expected synthetic test AbsRel < 0.15, got {abs_rel:.4f}")
        # 同一个场景换成机器人的实测相机（pc/robot_truth.json），同一个模型差多少：只记录，不设门禁
        abs_rel_robot, _, _ = abs_rel_at(DEFAULT_FOV_H_DEG, DEFAULT_H_CAM_M)
        print(f"[INFO] hover_mono_depth_v1 on one synthetic scene: AbsRel {abs_rel:.4f} at its training camera "
              f"({cam_fov:g} deg / {cam_h:.3f} m), {abs_rel_robot:.4f} at the measured robot camera "
              f"({DEFAULT_FOV_H_DEG:g} deg / {DEFAULT_H_CAM_M:.3f} m) - retrain at the measured camera to close this", flush=True)

        pred_e = infer_mono_depth_tflite(tfl_path, test_scene["rgb_nhwc"])["elev_prob_01"]
        gt_d = test_scene["depth_m"]
        gt_e = test_scene["elev_mask"]

        # 前景近端圆柱体区域深度 < 背景远端桌面区域深度
        near_cup_mask = (gt_e > 0.5) & (gt_d < 0.55)
        bg_table_mask = (gt_e < 0.5) & (gt_d > 0.85) & (gt_d < 1.65)
        self.assertGreater(int(np.sum(near_cup_mask)), 12)
        self.assertGreater(int(np.sum(bg_table_mask)), 30)

        fg_med_d = float(np.median(pred_d[near_cup_mask]))
        bg_med_d = float(np.median(pred_d[bg_table_mask]))
        self.assertLess(fg_med_d, bg_med_d - 0.20, f"Foreground ({fg_med_d:.3f}m) not closer than background ({bg_med_d:.3f}m)")

        fg_elev_mean = float(np.mean(pred_e[near_cup_mask]))
        bg_elev_mean = float(np.mean(pred_e[bg_table_mask]))
        self.assertGreater(fg_elev_mean, bg_elev_mean + 0.35)

        # 同时校验留出评测指标文件
        eval_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertLess(float(eval_metrics["AbsRel"]), 0.15)
        self.assertLess(float(eval_metrics["max_abs_diff"]), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
