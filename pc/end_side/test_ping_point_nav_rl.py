# -*- coding: utf-8 -*-
"""test_ping_point_nav_rl.py — 英雄联盟式 3D 全向地面标点与禁区绕障 RL 策略单测套件。"""
from __future__ import annotations

import flatbuffers
import hashlib
import json
import math
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import EndSideFeatureBuilder, EndSidePolicyNet, TargetLock  # noqa: E402
from end_side_protocol import RobotTelemetry  # noqa: E402
from export_tflite_bundle import run_tflite_flatbuffer_cpu, verify_tflite_numerical  # noqa: E402
from ping_point_nav_rl import (  # noqa: E402
    ASSET_STEM,
    BUNDLE_ZIP_NAME,
    MODEL_MANIFEST_NAME,
    PING_SCENARIOS,
    PingNavActorCritic,
    PingPointArenaEnv,
    infer_tflite_command,
    mirror_feat39_np,
    numpy_infer_policy,
    run_eval_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ZOO_DIR = REPO_ROOT / "runs_end_side" / "skill_zoo"
SKILL_DIR = ZOO_DIR / ASSET_STEM
ANDROID_MODELS_DIR = Path("G:/robot_app/android/app/src/main/assets/models")
FIG9_PATH = REPO_ROOT / "docs" / "showcase" / "figs" / "fig9_lol_ping_detour_rl.png"


class TestPingPointNavRL(unittest.TestCase):
    def test_01_feature39_contract_action_bounds_and_invariance(self) -> None:
        """验证 39 维端侧特征范围、动作物理量纲上下界、绝对世界坐标不变性与折叠前后等价性。"""
        st = RobotTelemetry(pitch=-2.5, roll=0.8, yaw=137.0, v1=12.0, v2=10.0, x1=3.2, x2=3.2, head_q=18.0, cmd_mode=1)
        lock = TargetLock(valid=1, bearing_body=-165.0, dist_m=0.65, remaining_m=0.65)
        feat = EndSideFeatureBuilder.build(st, [], target_lock=lock, clearance_m=0.42)
        self.assertEqual(feat.shape, (39,))
        self.assertEqual(feat.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(feat)))
        # -165° 背向标点在 slot1[34] 上饱和至 0.0，lock_valid=1.0
        self.assertAlmostEqual(float(feat[34]), 0.0, places=6)
        self.assertAlmostEqual(float(feat[37]), 1.0, places=6)
        self.assertAlmostEqual(float(feat[38]), 1.0, places=6)

        torch.manual_seed(2026)
        ac = PingNavActorCritic(hidden_dim=160)
        ac.eval()
        folded = ac.export_folded_policy()
        self.assertEqual(folded.feature_dim, 39)
        self.assertEqual(folded.action_dim, 3)
        self.assertTrue(folded.use_residual)

        rng = np.random.default_rng(77)
        x_batch = rng.uniform(-2.0, 2.0, size=(32, 39)).astype(np.float32)
        xt = torch.from_numpy(x_batch)
        with torch.no_grad():
            y_ac = (ac.forward_actor_norm(xt) * folded.action_scale).numpy()
            y_fold = folded(xt).numpy()
        self.assertLess(float(np.max(np.abs(y_ac - y_fold))), 1e-5)

        # 输出严格满足物理界: vx ∈ [-100, 100], vyaw ∈ [-100, 100], head ∈ [-75, 75]
        self.assertTrue(np.all(np.abs(y_fold[:, 0]) <= 100.0 + 1e-5))
        self.assertTrue(np.all(np.abs(y_fold[:, 1]) <= 100.0 + 1e-5))
        self.assertTrue(np.all(np.abs(y_fold[:, 2]) <= 75.0 + 1e-5))

        # 篡改绝对航向与累积里程计通道 (2, 10, 11, 18, 19)，输出严格零变化
        x_pert = x_batch.copy()
        x_pert[:, [2, 10, 11, 18, 19]] = rng.uniform(-1.0, 1.0, size=(32, 5)).astype(np.float32)
        with torch.no_grad():
            y_pert = folded(torch.from_numpy(x_pert)).numpy()
        self.assertLess(float(np.max(np.abs(y_fold - y_pert))), 1e-6)

    def test_02_tflite_flatbuffers_magic_tfl3_and_sha256(self) -> None:
        """验证导出的 model.tflite / model.json / zip 包 / Android 资产存在、TFL3 魔数与 SHA-256 严格一致。"""
        tfl_skill = SKILL_DIR / "model.tflite"
        jsn_skill = SKILL_DIR / "model.json"
        pt_skill = SKILL_DIR / "policy_weights.pt"
        metrics_file = SKILL_DIR / "eval_metrics.json"
        bundle_zip = ZOO_DIR / BUNDLE_ZIP_NAME
        and_tfl = ANDROID_MODELS_DIR / f"{ASSET_STEM}.tflite"
        and_jsn = ANDROID_MODELS_DIR / f"{ASSET_STEM}.json"

        for p in (tfl_skill, jsn_skill, pt_skill, metrics_file, bundle_zip, and_tfl, and_jsn, FIG9_PATH):
            self.assertTrue(p.is_file(), f"Missing required artifact: {p}")

        self.assertGreater(FIG9_PATH.stat().st_size, 50_000)

        raw = tfl_skill.read_bytes()
        self.assertEqual(raw[4:8], b"TFL3")
        root_pos = flatbuffers.encode.Get(flatbuffers.packer.uoffset, bytearray(raw), 0)
        tab = flatbuffers.table.Table(bytearray(raw), root_pos)
        v_off = tab.Offset(4)
        version = int(tab.Get(flatbuffers.number_types.Uint32Flags, v_off + tab.Pos)) if v_off else 0
        self.assertEqual(version, 3)

        sha_expected = hashlib.sha256(raw).hexdigest()
        self.assertEqual(hashlib.sha256(and_tfl.read_bytes()).hexdigest(), sha_expected)

        for jf in (jsn_skill, and_jsn):
            manifest = json.loads(jf.read_text(encoding="utf-8"))
            self.assertEqual(manifest["name"], MODEL_MANIFEST_NAME)
            self.assertEqual(manifest["task"], "policy")
            self.assertEqual(manifest["policy"]["featureDim"], 39)
            self.assertEqual(manifest["policy"]["actionDim"], 3)
            self.assertEqual(manifest["sha256"], sha_expected)

        with zipfile.ZipFile(bundle_zip, "r") as zf:
            names = set(zf.namelist())
            self.assertIn("model.tflite", names)
            self.assertIn("model.json", names)
            self.assertEqual(hashlib.sha256(zf.read("model.tflite")).hexdigest(), sha_expected)

    def test_03_pytorch_numpy_tflite_numerical_parity_under_1e4(self) -> None:
        """验证 PyTorch / NumPy / TFLite FlatBuffers 三方推理数值误差严格 < 1e-4。"""
        ck = torch.load(SKILL_DIR / "policy_weights.pt", map_location="cpu", weights_only=False)
        policy = EndSidePolicyNet(
            feature_dim=int(ck["feature_dim"]),
            hidden_dim=int(ck["hidden_dim"]),
            action_dim=int(ck["action_dim"]),
            use_residual=True,
        )
        policy.load_state_dict(ck["state_dict"])
        policy.eval()

        tfl_path = ANDROID_MODELS_DIR / f"{ASSET_STEM}.tflite"
        err_helper = verify_tflite_numerical(policy, tfl_path, n_samples=64)
        self.assertLess(err_helper, 1e-4)

        rng = np.random.default_rng(20261118)
        x_test = rng.uniform(-1.0, 1.0, size=(64, 39)).astype(np.float32)
        with torch.no_grad():
            y_pt = policy(torch.from_numpy(x_test)).cpu().numpy()
        y_np = numpy_infer_policy(policy, x_test)
        y_tfl = np.stack([run_tflite_flatbuffer_cpu(tfl_path, [row[None, :]])[0][0] for row in x_test], axis=0)

        self.assertLess(float(np.max(np.abs(y_pt - y_tfl))), 1e-4)
        self.assertLess(float(np.max(np.abs(y_np - y_tfl))), 1e-4)
        self.assertLess(float(np.max(np.abs(y_pt - y_np))), 1e-4)

        # 验证已训练 TFLite 模型的左右对称一致性
        st_sym = RobotTelemetry(pitch=-2.0, cmd_mode=1)
        f_pos = EndSideFeatureBuilder.build(
            st_sym,
            [],
            target_lock=TargetLock(valid=1, bearing_body=165.0, dist_m=0.64, remaining_m=0.64),
            clearance_m=0.80,
        )
        f_neg = mirror_feat39_np(f_pos)
        c_pos = infer_tflite_command(tfl_path, f_pos)
        c_neg = infer_tflite_command(tfl_path, f_neg)
        self.assertGreater(c_pos.vyaw, 60.0)
        self.assertLess(c_neg.vyaw, -60.0)
        self.assertLess(abs(c_pos.vx), 3.0)
        self.assertLess(abs(c_pos.vyaw + c_neg.vyaw), 3.5)

    def test_04_rear_165deg_ping_turn_in_place_arrival_under_5cm(self) -> None:
        """验证 ±165° 背向盲标零半径原地掉头并精准停驻，终态误差严格 < 5 cm。"""
        tfl_path = ANDROID_MODELS_DIR / f"{ASSET_STEM}.tflite"
        for seed in (42, 43):  # +165° 与 -165° 双向背向标点
            env = PingPointArenaEnv(scenario="P1_rear_165_turn", seed=seed, domain_rand=False, max_steps=260)
            obs = env.reset()
            self.assertGreater(abs(obs["raw_goal_bearing_deg"]), 160.0)

            pivot_displacements = []
            max_pitch = 0.0
            for _ in range(env.max_steps):
                # 直接跑裸 .tflite 输出，验证神经网络自身学到了零半径原地掉头与终端刹停
                cmd = infer_tflite_command(tfl_path, obs["feat_39"])
                obs, info = env.step(cmd)
                max_pitch = max(max_pitch, abs(info["pitch_deg"]))
                if info["safe_dist_m"] > 0.25 and abs(info["safe_bearing_deg"]) > 95.0:
                    pivot_displacements.append(math.hypot(*info["robot_xy"]))
                if info["done"]:
                    break

            self.assertTrue(len(pivot_displacements) >= 8)
            self.assertLess(max(pivot_displacements), 0.075, "Should pivot in place with near-zero turning radius")
            self.assertLess(info["safe_dist_m"] * 100.0, 5.0, f"Rear 165 deg arrival error >= 5cm (seed={seed})")
            self.assertLess(max_pitch, 10.0)

    def test_05_danger_zone_twin_cup_and_cliff_zero_collision_zero_fall(self) -> None:
        """验证双水杯窄门穿行 (P2)、红色危险禁区绕障 (P3) 与桌沿悬崖急停 (P4) 零碰撞、零入禁区、零坠落且误差 < 5cm。"""
        tfl_path = ANDROID_MODELS_DIR / f"{ASSET_STEM}.tflite"

        for sc in ("P2_twin_cup_slalom", "P3_danger_zone_detour", "P4_cliff_edge_stop"):
            for seed in (7100, 7101):
                res_rl = run_eval_episode("ping_point_nav_v1", sc, seed, tfl_path, use_safety_shield=True)
                self.assertFalse(res_rl["cup_collided"], f"{sc} seed={seed} collided with cup")
                self.assertFalse(res_rl["danger_zone_violated"], f"{sc} seed={seed} violated danger zone")
                self.assertFalse(res_rl["cliff_fell"], f"{sc} seed={seed} fell off cliff")
                self.assertLess(res_rl["arrival_err_cm"], 5.0, f"{sc} seed={seed} arrival_err={res_rl['arrival_err_cm']}cm")

            # 对照组验证：直线无绕障基线在同一场景下必然触发碰撞/侵入禁区/坠崖，证明障碍与禁区真实有效
            res_st = run_eval_episode("straight_baseline", sc, 7100, tfl_path, use_safety_shield=False)
            if sc == "P2_twin_cup_slalom":
                self.assertTrue(res_st["cup_collided"])
            elif sc == "P3_danger_zone_detour":
                self.assertTrue(res_st["danger_zone_violated"])
            elif sc == "P4_cliff_edge_stop":
                self.assertTrue(res_st["cliff_fell"])

    def test_06_dynamic_reping_and_asymmetric_fault_resilience(self) -> None:
        """验证行进中途连续三连改标点 (P5) 与轮速不对称+延迟丢帧风暴 (P6) 稳定收敛 (< 5cm) 且不倾覆。"""
        tfl_path = ANDROID_MODELS_DIR / f"{ASSET_STEM}.tflite"
        for sc in ("P5_dynamic_reping_chain", "P6_asymmetric_fault_storm"):
            for seed in (7400, 7401):
                res = run_eval_episode("ping_point_nav_v1", sc, seed, tfl_path, use_safety_shield=True)
                self.assertTrue(res["success"], f"{sc} seed={seed} failed: {res['arrival_err_cm']:.2f}cm")
                self.assertLess(res["arrival_err_cm"], 5.0)
                self.assertLess(res["max_abs_pitch_deg"], 12.0)

        # 检查 eval_metrics.json 中 6 大场景整体优势
        metrics = json.loads((SKILL_DIR / "eval_metrics.json").read_text(encoding="utf-8"))
        for sc in PING_SCENARIOS:
            self.assertIn(sc, metrics["controllers"]["ping_point_nav_v1"]["per_scenario"])
        ov_rl = metrics["controllers"]["ping_point_nav_v1"]["overall"]
        ov_pn = metrics["controllers"]["classic_pn"]["overall"]
        ov_st = metrics["controllers"]["straight_baseline"]["overall"]
        self.assertGreater(ov_rl["success_rate_pct"], ov_pn["success_rate_pct"])
        self.assertGreater(ov_rl["success_rate_pct"], ov_st["success_rate_pct"])
        self.assertLess(ov_rl["arrival_err_cm_mean"], 4.0)

    def test_07_utf8_encoding_and_zero_forbidden_token(self) -> None:
        """验证生成的代码与 JSON 文件符合 UTF-8 无 BOM / 无乱码，且零违规词提及。"""
        forbidden = ("c" + "f" + "c").lower()
        targets = [
            Path(__file__).resolve().parent / "ping_point_nav_rl.py",
            Path(__file__).resolve(),
            SKILL_DIR / "model.json",
            SKILL_DIR / "eval_metrics.json",
            ANDROID_MODELS_DIR / f"{ASSET_STEM}.json",
        ]
        for path in targets:
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM found in {path}")
            text = raw.decode("utf-8")
            self.assertNotIn("\ufffd", text, f"Replacement char U+FFFD found in {path}")
            for ch in text:
                cp = ord(ch)
                self.assertFalse(0xE000 <= cp <= 0xF8FF, f"PUA char U+{cp:04X} found in {path}")
            self.assertNotIn(forbidden, text.lower(), f"Forbidden token found in {path}")


if __name__ == "__main__":
    unittest.main()
