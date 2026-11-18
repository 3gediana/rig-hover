# -*- coding: utf-8 -*-
"""test_neural_gaussian_splat.py — 6 通道前馈神经 3D 高斯网络、多视角贝叶斯修正与持久记忆导航门禁单测。"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mono_depth_model import INPUT_C, INPUT_H, INPUT_W, OUTPUT_H, OUTPUT_W  # noqa: E402
from neural_gaussian_splat_model import (  # noqa: E402
    CHANNEL_NAMES,
    OPACITY_MAX,
    OPACITY_MIN,
    SCALE_XY_MAX_M,
    SCALE_XY_MIN_M,
    SCALE_Z_MAX_M,
    SCALE_Z_MIN_M,
    SPLAT_OUT_C,
    TAU_MAX,
    TAU_MIN,
    GlobalMemoryNavigator,
    HoverGaussianSplatNet,
    OrbitCamera3D,
    PersistentGaussianMemoryMap,
    RobotCameraPose,
    differentiable_novel_view_render_24x24,
    export_neural_splat_bundle,
    infer_neural_splat_tflite,
    load_neural_splat_from_tflite,
    make_world_ray_grid,
    opacity_01_to_alpha,
    precision_01_to_tau,
    scale_xy_01_to_m,
    scale_xy_m_to_01,
    scale_z_01_to_m,
    scale_z_m_to_01,
    verify_neural_splat_tflite_parity,
)
from train_neural_gaussian_splat import (  # noqa: E402
    DEFAULT_ANDROID_ASSETS,
    DEFAULT_FIG10_PATH,
    DEFAULT_OUT_DIR,
    build_benchmark_exploration_arena,
    render_posed_desktop_scene,
)


class TestNeuralGaussianSplatSuite(unittest.TestCase):
    """覆盖 6 通道神经输出、TFLite 精度、可微重投影梯度、各向异性/精度学习、多视角记忆拓展与盲区导航的门禁。"""

    def test_01_six_channel_neural_splat_shapes_and_physical_ranges(self) -> None:
        model = HoverGaussianSplatNet().eval()
        x = torch.rand(2, INPUT_C, INPUT_H, INPUT_W)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, SPLAT_OUT_C, OUTPUT_H, OUTPUT_W))
        self.assertTrue(bool(torch.all((y > 0.0) & (y < 1.0))))

        # 校验水平/垂直椭球半径互逆一致性与物理边界
        sxy_grid = np.linspace(SCALE_XY_MIN_M, SCALE_XY_MAX_M, 40, dtype=np.float32)
        np.testing.assert_allclose(sxy_grid, scale_xy_01_to_m(scale_xy_m_to_01(sxy_grid)), atol=1e-6)

        sz_grid = np.linspace(SCALE_Z_MIN_M, SCALE_Z_MAX_M, 40, dtype=np.float32)
        np.testing.assert_allclose(sz_grid, scale_z_01_to_m(scale_z_m_to_01(sz_grid)), atol=1e-6)

        self.assertAlmostEqual(float(opacity_01_to_alpha(0.0)), OPACITY_MIN, places=5)
        self.assertAlmostEqual(float(opacity_01_to_alpha(1.0)), OPACITY_MAX, places=5)
        self.assertAlmostEqual(float(precision_01_to_tau(0.0)), TAU_MIN, places=4)
        self.assertAlmostEqual(float(precision_01_to_tau(1.0)), TAU_MAX, places=2)

        rgb_nhwc = np.random.default_rng(11).uniform(0.0, 1.0, size=(INPUT_H, INPUT_W, INPUT_C)).astype(np.float32)
        dec = model.infer_nhwc(rgb_nhwc)
        self.assertEqual(dec["raw_nhwc"].shape, (OUTPUT_H, OUTPUT_W, SPLAT_OUT_C))
        for key in ("depth_m", "elev_01", "scale_xy_m", "scale_z_m", "opacity", "precision_tau", "sigma_m"):
            self.assertEqual(dec[key].shape, (OUTPUT_H, OUTPUT_W))
            self.assertTrue(bool(np.all(np.isfinite(dec[key]))))

    def test_02_tflite_v3_export_bundle_and_flatbuffer_parity(self) -> None:
        torch.manual_seed(2026)
        model = HoverGaussianSplatNet()
        model.train()
        for _ in range(5):
            _ = model(torch.rand(4, INPUT_C, INPUT_H, INPUT_W))
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            res = export_neural_splat_bundle(model, out_dir=tmpdir, version="2026.11.18")
            tfl_p = Path(res["tflite_path"])
            jsn_p = Path(res["json_path"])
            zip_p = Path(res["zip_path"])

            raw_bytes = tfl_p.read_bytes()
            self.assertEqual(raw_bytes[4:8], b"TFL3")
            self.assertLess(len(raw_bytes), 750 * 1024)

            meta = json.loads(jsn_p.read_text(encoding="utf-8"))
            self.assertEqual(meta["name"], "hover-gaussian-splat-v2")
            self.assertEqual(meta["task"], "depth")
            self.assertEqual(meta["output"]["channels"], 6)
            self.assertEqual(tuple(meta["output"]["channelNames"]), CHANNEL_NAMES)
            self.assertEqual(meta["sha256"], hashlib.sha256(raw_bytes).hexdigest())

            with zipfile.ZipFile(zip_p, "r") as zf:
                self.assertEqual(set(zf.namelist()), {"model.tflite", "model.json"})
                self.assertEqual(zf.read("model.tflite"), raw_bytes)

            max_diff = verify_neural_splat_tflite_parity(model, tfl_p, n_samples=8, seed=77)
            self.assertLess(max_diff, 1e-4)

            reloaded = load_neural_splat_from_tflite(tfl_p)
            x_t = torch.rand(1, INPUT_C, INPUT_H, INPUT_W)
            with torch.no_grad():
                rt_diff = float(torch.max(torch.abs(model(x_t) - reloaded(x_t))).item())
            self.assertLess(rt_diff, 1e-5)

    def test_03_differentiable_novel_view_3dgs_gradient_flow(self) -> None:
        model = HoverGaussianSplatNet().train()
        x = torch.rand(2, INPUT_C, INPUT_H, INPUT_W)
        rgb48 = F.avg_pool2d(x, kernel_size=2, stride=2)
        target24 = torch.rand(2, 3, 24, 24)

        p1 = RobotCameraPose(x_m=0.0, y_m=0.0, yaw_deg=0.0, pitch_deg=2.0)
        p2 = RobotCameraPose(x_m=0.14, y_m=-0.08, yaw_deg=12.0, pitch_deg=2.5)
        o1, r1 = make_world_ray_grid(p1)
        o2, _ = make_world_ray_grid(p2)
        rw2c2 = p2.rotation_cam_to_world().T

        o1_t = torch.from_numpy(np.stack([o1, o1]))
        r1_t = torch.from_numpy(np.stack([r1, r1]))
        o2_t = torch.from_numpy(np.stack([o2, o2]))
        rw2c2_t = torch.from_numpy(np.stack([rw2c2, rw2c2]))

        pred = model(x)
        nv = differentiable_novel_view_render_24x24(pred, rgb48, o1_t, r1_t, o2_t, rw2c2_t)
        loss = F.l1_loss(nv["rendered_rgb_24"], target24)
        loss.backward()

        self.assertIsNotNone(model.head_out.weight.grad)
        assert model.head_out.weight.grad is not None
        grad_per_ch = torch.norm(model.head_out.weight.grad.view(6, -1), dim=1)
        # 参与新视角重投影的通道 (ch0 深度, ch2 s_xy, ch3 s_z, ch4 opacity, ch5 precision) 均必须获得非零有限梯度
        for ch_idx in (0, 2, 3, 4, 5):
            g_val = float(grad_per_ch[ch_idx].item())
            self.assertTrue(math.isfinite(g_val) and g_val > 1e-7, f"Channel {ch_idx} missing photo grad: {g_val}")

    def test_04_trained_model_anisotropy_and_heteroscedastic_precision(self) -> None:
        tfl_path = DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.tflite"
        jsn_path = DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.json"
        and_tfl = DEFAULT_ANDROID_ASSETS / "hover_gaussian_splat_v2.tflite"
        and_jsn = DEFAULT_ANDROID_ASSETS / "hover_gaussian_splat_v2.json"

        for p in (tfl_path, jsn_path, and_tfl, and_jsn):
            self.assertTrue(p.exists(), f"Missing exported artifact: {p}")
        self.assertEqual(tfl_path.read_bytes(), and_tfl.read_bytes())
        self.assertEqual(jsn_path.read_bytes(), and_jsn.read_bytes())

        arena = build_benchmark_exploration_arena()
        pose = RobotCameraPose(x_m=0.0, y_m=0.0, yaw_deg=5.0, pitch_deg=3.0)
        scene = render_posed_desktop_scene(arena, pose, rng=np.random.default_rng(99), grayscale=False)
        pred = infer_neural_splat_tflite(tfl_path, scene["rgb_nhwc"])

        gt_d = scene["depth_m"]
        gt_e = scene["elev_01"]
        abs_rel = float(np.mean(np.abs(pred["depth_m"] - gt_d) / np.maximum(gt_d, 1e-4)))
        self.assertLess(abs_rel, 0.16, f"Expected AbsRel < 0.16, got {abs_rel:.4f}")

        # 验证各向异性学习：平坦桌面预测为水平宽、垂直薄的扁圆盘；立体水杯预测为高垂直半径竖椭球
        tbl_mask = (gt_e < 0.5) & (gt_d < 1.25)
        obj_mask = gt_e > 0.5
        self.assertGreater(int(np.sum(tbl_mask)), 80)
        self.assertGreater(int(np.sum(obj_mask)), 12)

        tbl_sxy = float(np.mean(pred["scale_xy_m"][tbl_mask]))
        tbl_sz = float(np.mean(pred["scale_z_m"][tbl_mask]))
        obj_sxy = float(np.mean(pred["scale_xy_m"][obj_mask]))
        obj_sz = float(np.mean(pred["scale_z_m"][obj_mask]))

        self.assertGreater(tbl_sxy, 0.014)
        self.assertGreater(tbl_sxy, obj_sxy, "平坦桌面水平高斯半径应大于精细立体物体核")
        self.assertLess(tbl_sz, 0.012)
        self.assertLess(tbl_sz / tbl_sxy, 0.45, "桌面高斯点应呈扁平水平圆盘 (s_z / s_xy < 0.45)")
        self.assertGreater(obj_sz, tbl_sz + 0.008, "立体物体垂直椭球半径应显著大于平坦桌面")
        self.assertGreater(obj_sz / max(obj_sxy, 1e-5), 0.85)

        # 验证异方差精度学习：近距离清晰区域观测精度显著高于远景墙面
        near_tau = float(np.mean(pred["precision_tau"][gt_d < 0.70]))
        far_tau = float(np.mean(pred["precision_tau"][gt_d > 1.70]))
        self.assertGreater(near_tau, far_tau * 1.8)

    def test_05_multiview_kalman_refinement_and_persistent_memory_expansion(self) -> None:
        # 1) 单元测试多视角贝叶斯精度加权融合收敛性
        mem = PersistentGaussianMemoryMap(voxel_size=0.040, max_splats=4096)
        true_pt = np.array([0.50, 0.10, 0.08], dtype=np.float64)
        rng = np.random.default_rng(12345)

        errs = []
        sigmas = []
        for view_k in range(6):
            ang = math.radians(view_k * 52.0 - 90.0)
            cam_x = float(true_pt[0] - 0.45 * math.cos(ang))
            cam_y = float(true_pt[1] - 0.45 * math.sin(ang))
            # 早期远视角噪声大且精度低，后期近侧视角噪声小且精度高
            tau_k = 120.0 if view_k < 2 else 1800.0
            std_k = 0.014 if view_k < 2 else 0.004
            obs_pt = true_pt + rng.normal(0.0, std_k, size=3)

            key = mem._pack_voxel_key(float(true_pt[0]), float(true_pt[1]), float(true_pt[2]))
            sec_bit = mem._sector_bit(cam_x, cam_y, float(obs_pt[0]), float(obs_pt[1]))
            if view_k == 0:
                mem.count = 1
                mem.pos_x[0], mem.pos_y[0], mem.pos_z[0] = obs_pt
                mem.scale_xy[0], mem.scale_z[0], mem.opacity[0] = 0.018, 0.024, 0.90
                mem.precision_tau[0] = tau_k
                mem.obs_count[0] = 1
                mem.view_sector_mask[0] = sec_bit
                mem._voxel_to_idx[key] = 0
            else:
                k_gain = (tau_k * 1.35) / (float(mem.precision_tau[0]) + tau_k * 1.35)
                mem.pos_x[0] = (1.0 - k_gain) * mem.pos_x[0] + k_gain * float(obs_pt[0])
                mem.pos_y[0] = (1.0 - k_gain) * mem.pos_y[0] + k_gain * float(obs_pt[1])
                mem.pos_z[0] = (1.0 - k_gain) * mem.pos_z[0] + k_gain * float(obs_pt[2])
                mem.precision_tau[0] += tau_k * 1.35
                mem.obs_count[0] += 1
                mem.view_sector_mask[0] |= sec_bit

            cur_xyz = np.array([mem.pos_x[0], mem.pos_y[0], mem.pos_z[0]], dtype=np.float64)
            errs.append(float(np.linalg.norm(cur_xyz - true_pt)))
            sigmas.append(1.0 / math.sqrt(float(mem.precision_tau[0])))

        self.assertLess(errs[-1], errs[0] * 0.55)
        self.assertLess(sigmas[-1], sigmas[0] * 0.35)

        # 2) 核验完整基准探索实验指标与 expanded_memory_map.ply
        metrics_file = DEFAULT_OUT_DIR / "neural_3dgs_metrics.json"
        ply_file = DEFAULT_OUT_DIR / "expanded_memory_map.ply"
        self.assertTrue(metrics_file.exists())
        self.assertTrue(ply_file.exists())

        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        bench = metrics["spatial_memory_expansion_benchmark"]

        self.assertGreater(
            bench["multiview_position_error_reduction_pct"],
            35.0,
            f"多视角卡尔曼位置误差下降需 > 35%, 实测={bench['multiview_position_error_reduction_pct']}%",
        )
        self.assertAlmostEqual(bench["region_a_splat_retention_rate"], 1.0, places=3)
        self.assertLess(bench["initial_area_m2"], 0.85)
        self.assertGreater(bench["expanded_area_m2"], 1.80, f"拓展面积需 > 1.8 m², 实测={bench['expanded_area_m2']}")

        ply_lines = ply_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(ply_lines[0].strip(), "ply")
        hdr_end = ply_lines.index("end_header")
        v_cnt = int([ln.split()[-1] for ln in ply_lines[:hdr_end] if ln.startswith("element vertex ")][0])
        self.assertGreaterEqual(v_cnt, 1500)
        self.assertEqual(len(ply_lines) - hdr_end - 1, v_cnt)

    def test_06_global_out_of_fov_click_and_closed_loop_return_navigation(self) -> None:
        mem = PersistentGaussianMemoryMap(voxel_size=0.025, max_splats=8192)
        # 在走廊中央 (0.54, 0.10) 注入持久记忆立体水杯障碍簇
        for dx in (-0.03, 0.0, 0.03):
            for dy in (-0.03, 0.0, 0.03):
                idx = mem.count
                mem.count += 1
                mem.pos_x[idx] = 0.54 + dx
                mem.pos_y[idx] = 0.10 + dy
                mem.pos_z[idx] = 0.10
                mem.scale_xy[idx] = 0.025
                mem.scale_z[idx] = 0.030
                mem.opacity[idx] = 0.92
                mem.elev_prob[idx] = 0.95
                mem.precision_tau[idx] = 1200.0

        cam = OrbitCamera3D(azimuth_deg=-135.0, elevation_deg=45.0, distance_m=2.0, target_x=0.60, target_y=0.15)
        goal_a = (0.12, -0.14)
        scr = mem.project_world_point(goal_a[0], goal_a[1], 0.0, cam, 1080.0, 720.0)
        self.assertIsNotNone(scr)
        assert scr is not None
        hit = mem.raycast_screen_to_ground_ping(scr[0], scr[1], cam, 1080.0, 720.0, 0.0)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertLess(math.hypot(goal_a[0] - hit[0], goal_a[1] - hit[1]), 1e-4)

        nav = GlobalMemoryNavigator(grid_res_m=0.030, robot_clearance_radius_m=0.115)
        pose_b = RobotCameraPose(x_m=1.05, y_m=0.48, yaw_deg=32.0)
        res = nav.navigate_closed_loop(mem, start_pose=pose_b, goal_xy=hit, dt=0.05, max_steps=260)

        self.assertTrue(res["target_initially_out_of_fov"], "回点区域 A 目标必须在当前相机 60° 视野盲区外")
        self.assertTrue(res["reached"])
        self.assertLess(res["final_error_m"], 0.04, f"返回区域 A 最终误差需 < 4cm, 实测={res['final_error_m']*100:.2f}cm")
        self.assertGreater(res["min_obstacle_surface_clearance_m"], 0.06, "绕障全程需保持安全净空")

    def test_07_showcase_fig10_and_utf8_zero_forbidden_token_gate(self) -> None:
        self.assertTrue(DEFAULT_FIG10_PATH.exists())
        self.assertGreater(DEFAULT_FIG10_PATH.stat().st_size, 150_000)
        img = cv2.imread(str(DEFAULT_FIG10_PATH), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        assert img is not None
        h, w = img.shape[:2]
        self.assertGreaterEqual(w, 3500)
        self.assertGreaterEqual(h, 2000)

        forbidden = (chr(67) + chr(102) + chr(67)).lower()
        check_files = [
            HERE / "neural_gaussian_splat_model.py",
            HERE / "train_neural_gaussian_splat.py",
            HERE / "test_neural_gaussian_splat.py",
            DEFAULT_OUT_DIR / "hover_gaussian_splat_v2.json",
            DEFAULT_OUT_DIR / "neural_3dgs_metrics.json",
            DEFAULT_OUT_DIR / "expanded_memory_map.ply",
        ]
        for fpath in check_files:
            raw = fpath.read_bytes()
            self.assertNotEqual(raw[:3], b"\xef\xbb\xbf", f"BOM in {fpath}")
            text = raw.decode("utf-8")
            pua = sum(1 for c in text if "\ue000" <= c <= "\uf8ff")
            bad = text.count("\ufffd")
            self.assertEqual(pua, 0, f"PUA in {fpath}")
            self.assertEqual(bad, 0, f"U+FFFD in {fpath}")
            self.assertNotIn(forbidden, text.lower(), f"Forbidden token in {fpath}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
