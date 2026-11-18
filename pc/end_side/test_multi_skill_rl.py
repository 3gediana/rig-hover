# -*- coding: utf-8 -*-
"""test_multi_skill_rl.py — 多技能具身强化学习动物园契约、TFLite 精度一致性与物理闭环行为单测。"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import EndSideFeatureBuilder, EndSidePolicyNet  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from export_tflite_bundle import run_tflite_flatbuffer_cpu, verify_tflite_numerical  # noqa: E402
from multi_skill_rl import (  # noqa: E402
    SKILL_SPECS,
    MultiSkillArenaEnv,
    SkillActorCritic,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOO_DIR = REPO_ROOT / "runs_end_side" / "skill_zoo"
ANDROID_MODELS_DIR = Path("G:/robot_app/android/app/src/main/assets/models")
FIG7_PATH = REPO_ROOT / "docs" / "showcase" / "figs" / "fig7_multi_skill_zoo.png"


def _infer_tflite_cmd(tflite_path: Path, feat_39: np.ndarray) -> DriveCommand:
    out = run_tflite_flatbuffer_cpu(tflite_path, [feat_39.reshape(1, 39)])[0][0]
    return DriveCommand(vx=float(out[0]), vyaw=float(out[1]), head=float(out[2])).sanitized()


class TestMultiSkillRLZoo(unittest.TestCase):
    def test_folded_policy_contract_and_world_invariance(self):
        """折叠导出后的网络必须严格同构于 EndSidePolicyNet(39,128,3,use_residual=True) 且对绝对世界坐标完全不变。"""
        torch.manual_seed(42)
        ac = SkillActorCritic(hidden_dim=128)
        ac.eval()
        folded = ac.export_folded_policy()

        self.assertIsInstance(folded, EndSidePolicyNet)
        self.assertEqual(folded.feature_dim, 39)
        self.assertEqual(folded.hidden_dim, 128)
        self.assertEqual(folded.action_dim, 3)
        self.assertTrue(folded.use_residual)

        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, size=(24, 39)).astype(np.float32)
        xt = torch.from_numpy(x)
        with torch.no_grad():
            y_ac = ac.forward_actor_norm(xt) * folded.action_scale
            y_fold = folded(xt)
        self.assertLess(float(torch.max(torch.abs(y_ac - y_fold)).item()), 1e-5)

        # 篡改绝对航向与绝对里程坐标 (idx 2, 10, 11, 18, 19)，输出必须零变化
        x_pert = x.copy()
        x_pert[:, [2, 10, 11, 18, 19]] = rng.uniform(-1.0, 1.0, size=(24, 5)).astype(np.float32)
        with torch.no_grad():
            y_pert = folded(torch.from_numpy(x_pert))
        self.assertLess(float(torch.max(torch.abs(y_fold - y_pert)).item()), 1e-6)

    def test_deployed_tflite_and_android_assets_parity(self):
        """三大技能包、Android 内置资产、SHA-256 校验和与 PyTorch<->TFLite 数值误差 (<1e-4) 必须全部就绪。"""
        self.assertTrue((ZOO_DIR / "skill_zoo_metrics.json").is_file())
        self.assertTrue(FIG7_PATH.is_file())
        self.assertGreater(FIG7_PATH.stat().st_size, 40_000)

        for skill_key, spec in SKILL_SPECS.items():
            stem = spec["asset_stem"]
            sub_dir = ZOO_DIR / stem
            pt_file = sub_dir / "policy_weights.pt"
            zip_root = ZOO_DIR / f"{spec['model_name']}-bundle.zip"
            and_tfl = ANDROID_MODELS_DIR / f"{stem}.tflite"
            and_jsn = ANDROID_MODELS_DIR / f"{stem}.json"

            self.assertTrue(pt_file.is_file(), f"Missing {pt_file}")
            self.assertTrue(zip_root.is_file(), f"Missing {zip_root}")
            self.assertTrue(and_tfl.is_file(), f"Missing Android asset {and_tfl}")
            self.assertTrue(and_jsn.is_file(), f"Missing Android manifest {and_jsn}")

            manifest = json.loads(and_jsn.read_text(encoding="utf-8"))
            self.assertEqual(manifest["task"], "policy")
            self.assertEqual(manifest["name"], spec["model_name"])
            self.assertEqual(manifest["policy"]["featureDim"], 39)
            self.assertEqual(manifest["policy"]["actionDim"], 3)

            real_sha = hashlib.sha256(and_tfl.read_bytes()).hexdigest()
            self.assertEqual(manifest["sha256"], real_sha)

            ck = torch.load(pt_file, map_location="cpu", weights_only=False)
            net = EndSidePolicyNet(
                feature_dim=int(ck["feature_dim"]),
                hidden_dim=int(ck["hidden_dim"]),
                action_dim=int(ck["action_dim"]),
                use_residual=True,
            )
            net.load_state_dict(ck["state_dict"])
            net.eval()
            err = verify_tflite_numerical(net, and_tfl, n_samples=48)
            self.assertLess(err, 1e-4, f"{skill_key} PyTorch vs TFLite parity {err} >= 1e-4")

    def test_orbit_inspector_tflite_closed_loop_physics(self):
        """Skill 1 (.tflite 真图推理): 闭环围绕目标作 r*=0.48m 360° 平滑环绕，头向内凝视 ±38°，|pitch|<8°。"""
        tfl_path = ANDROID_MODELS_DIR / "orbit_inspector_v1.tflite"
        env = MultiSkillArenaEnv(skill="orbit_inspector", seed=900, domain_rand=False, max_steps=540, lock_mode=True)
        obs = env.reset()

        radii, vxs, heads, pitches = [], [], [], []
        for _ in range(env.max_steps):
            cmd = _infer_tflite_cmd(tfl_path, obs["feat_39"])
            obs, info = env.step(cmd)
            radii.append(info["true_dist_m"])
            vxs.append(cmd.vx)
            heads.append(abs(info["head_q_deg"]))
            pitches.append(abs(info["pitch_deg"]))

        r_steady = np.asarray(radii[180:], dtype=np.float64)
        radial_rmse_m = float(math.sqrt(np.mean((r_steady - 0.48) ** 2)))
        self.assertLess(radial_rmse_m, 0.045)
        self.assertGreater(info["cum_orbit_deg"], 310.0)
        self.assertTrue(26.0 <= float(np.mean(vxs[180:])) <= 46.0)
        self.assertGreater(float(np.mean(heads[180:])), 32.0)
        self.assertLess(float(np.max(pitches)), 8.0)

    def test_social_proxemics_tflite_come_closer_and_backoff(self):
        """Skill 2 (.tflite 真图推理): h>0.34 平滑倒车退让，0.10<h<0.22 温和前进接近，稳态停在 d*≈0.75m。"""
        tfl_path = ANDROID_MODELS_DIR / "social_proxemics_v1.tflite"
        st_zero = RobotTelemetry(pitch=-2.0)

        # 1. 靠太近 (h = 0.46 > 0.34) -> 必须倒车退让 vx < -15%
        f_close = EndSideFeatureBuilder.build(
            st_zero,
            [{"cx": 0.50, "cy": 0.36, "w": 0.36, "h": 0.46, "score": 0.92, "cls": 0}],
        )
        c_close = _infer_tflite_cmd(tfl_path, f_close)
        self.assertLess(c_close.vx, -15.0)
        self.assertGreaterEqual(c_close.vx, -46.0)

        # 2. 退太远 (h = 0.15 ∈ (0.10, 0.22)) -> 必须前进接近 vx > +12%
        f_far = EndSideFeatureBuilder.build(
            st_zero,
            [{"cx": 0.50, "cy": 0.36, "w": 0.12, "h": 0.15, "score": 0.92, "cls": 0}],
        )
        c_far = _infer_tflite_cmd(tfl_path, f_far)
        self.assertGreater(c_far.vx, 12.0)

        # 3. 舒适社交平衡点 (h* = 0.28, cx = 0.50) -> 零速静稳居中
        f_eq = EndSideFeatureBuilder.build(
            st_zero,
            [{"cx": 0.50, "cy": 0.36, "w": 0.22, "h": 0.28, "score": 0.92, "cls": 0}],
        )
        c_eq = _infer_tflite_cmd(tfl_path, f_eq)
        self.assertLess(abs(c_eq.vx), 3.0)
        self.assertLess(abs(c_eq.vyaw), 2.5)
        self.assertLess(abs(c_eq.head), 2.0)

        # 4. 闭环物理轨迹验证 (远立跟进 + 贴近倒车 + 横向视线跟随)
        env = MultiSkillArenaEnv(skill="social_proxemics", seed=950, domain_rand=False, max_steps=360)
        obs = env.reset()
        dists, vxs, pitches = [], [], []
        for _ in range(env.max_steps):
            cmd = _infer_tflite_cmd(tfl_path, obs["feat_39"])
            obs, info = env.step(cmd)
            dists.append(info["true_dist_m"])
            vxs.append(cmd.vx)
            pitches.append(abs(info["pitch_deg"]))

        self.assertLess(min(vxs), -10.0)
        self.assertGreater(max(vxs), 10.0)
        self.assertAlmostEqual(float(np.mean(dists[-35:])), 0.75, delta=0.065)
        self.assertLess(max(pitches), 8.0)

    def test_figure8_patrol_tflite_active_sweep_and_instant_capture(self):
        """Skill 3 (.tflite 真图推理): 无目标时内部相位自激有界 8 字扫视；见目标瞬间 vx->0 并锁定视线。"""
        tfl_path = ANDROID_MODELS_DIR / "figure8_patrol_v1.tflite"
        env = MultiSkillArenaEnv(skill="figure8_patrol", seed=980, domain_rand=False, max_steps=420)
        obs = env.reset()
        env.appear_step = 300

        rxs, rys, heads, vxs, gaze_errs, pitches = [], [], [], [], [], []
        for _ in range(env.max_steps):
            cmd = _infer_tflite_cmd(tfl_path, obs["feat_39"])
            obs, info = env.step(cmd)
            rxs.append(info["robot_xy"][0])
            rys.append(info["robot_xy"][1])
            heads.append(info["head_q_deg"])
            vxs.append(cmd.vx)
            gaze_errs.append(abs(info["theta_img_deg"]))
            pitches.append(abs(info["pitch_deg"]))

        # 无目标段 (0..300): 自激产生显著头部扫视且有界不发散 (直线巡航 14.4s 将远超 1.1m，8 字巡检控制在 <0.58m 内)
        head_p2p = max(heads[:300]) - min(heads[:300])
        self.assertGreater(head_p2p, 42.0)
        self.assertGreater(float(np.mean(vxs[20:290])), 18.0)
        self.assertLess(max(rxs[:300]) - min(rxs[:300]), 0.58)
        self.assertLess(max(rys[:300]) - min(rys[:300]), 0.50)

        # 目标出现后 (310..420): 立即刹停前向速度并将目标捕获在视野中央
        self.assertLess(float(np.mean(np.abs(vxs[312:]))), 2.5)
        self.assertLess(float(np.mean(gaze_errs[-25:])), 3.5)
        self.assertLess(max(pitches), 8.0)


if __name__ == "__main__":
    unittest.main()
