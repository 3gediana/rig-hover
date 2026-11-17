# -*- coding: utf-8 -*-
"""test_strategy_engine.py — 策略组合引擎 (PRIORITY/BLEND)、DoubleKnock TFLite 导出与局域网推送服务单测。"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

from end_side_controller import EndSidePolicyNet, TargetLock  # noqa: E402
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402
from export_tflite_bundle import export_double_knock_bundle, verify_double_knock_tflite_numerical  # noqa: E402
from model_push_server import start_model_push_server  # noqa: E402
from strategy_engine import (  # noqa: E402
    BehaviorRule,
    SignalSnapshot,
    StrategyComposer,
    StrategySpec,
    builtin_presets,
)
from train_double_knock import DoubleKnockNet  # noqa: E402


class TestStrategyEngineAndPushServer(unittest.TestCase):
    def test_double_knock_tflite_export_numerical_parity(self):
        """导出的 DoubleKnockNet TFLite (48帧双输入双输出) 必须与 PyTorch eval() 数值对齐 (< 1e-4)。"""
        ck_path = Path("D:/robot_work/runs_knock/double_knock_20261115/double_knock_deploy.pt")
        model = DoubleKnockNet()
        if ck_path.exists():
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["state_dict"])
        model.eval()

        with tempfile.TemporaryDirectory() as tmp:
            res = export_double_knock_bundle(model, tmp, window=48, image_size=96)
            tflite_path = Path(res["tflite_path"])
            json_path = Path(res["json_path"])
            zip_path = Path(res["zip_path"])

            meta = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["task"], "action")
            self.assertEqual(meta["action"]["window"], 48)
            self.assertEqual(meta["sha256"], hashlib.sha256(tflite_path.read_bytes()).hexdigest())
            self.assertTrue(zip_path.exists())

            errs = verify_double_knock_tflite_numerical(model, tflite_path, window=48, image_size=96, n_samples=2)
            self.assertLess(errs["max_imp_err"], 1e-4)
            self.assertLess(errs["max_cnt_err"], 1e-4)

    def test_priority_tree_fist_overrides_face_and_knock_overrides_fist(self):
        """PRIORITY 规则链：未见目标哨兵扫描 -> 有人脸看人脸 -> 握拳优先看拳头 -> 双敲进入 RL 接近与安全零过冲。"""
        presets = {p.id: p for p in builtin_presets()}
        spec = presets["priority_composed_social"]
        policy = EndSidePolicyNet()
        policy.eval()
        comp = StrategyComposer(spec, policy_model=policy)
        tel = RobotTelemetry(head_q=0.0, pitch=0.0, yaw=0.0)

        # 1. 无目标 -> 命中闲置扫描
        comp.step(SignalSnapshot(dt_s=0.1), tel)
        self.assertEqual(comp.last_active_rules, ["r_idle_patrol"])

        # 2. 只有人脸 -> 命中人脸跟踪
        comp.step(
            SignalSnapshot(
                dt_s=0.1,
                face_valid=True,
                theta_face_deg=15.0,
                detections=[{"cx": 0.72, "cy": 0.4, "w": 0.14, "h": 0.16, "score": 0.9, "cls": 0}],
            ),
            tel,
        )
        self.assertEqual(comp.last_active_rules, ["r_face_follow"])

        # 3. 人脸 + 握拳同时存在 -> 拳头优先于人脸
        comp.step(
            SignalSnapshot(
                dt_s=0.1,
                face_valid=True,
                theta_face_deg=15.0,
                hand_valid=True,
                fist_prob=0.88,
                theta_fist_deg=-18.0,
                detections=[
                    {"cx": 0.72, "cy": 0.4, "w": 0.14, "h": 0.16, "score": 0.9, "cls": 0},
                    {"cx": 0.20, "cy": 0.55, "w": 0.10, "h": 0.12, "score": 0.85, "cls": 1},
                ],
            ),
            tel,
        )
        self.assertEqual(comp.last_active_rules, ["r_fist_first"])

        # 4. 触发双敲并锁定目标 -> 命中 RL 闭环接近 + 安全零过冲保护
        cmd_over = comp.step(
            SignalSnapshot(
                dt_s=0.1,
                knock_double_fired=True,
                target_lock=TargetLock(valid=1, remaining_m=-0.02),
                detections=[{"cx": 0.50, "cy": 0.55, "w": 0.12, "h": 0.12, "score": 0.9, "cls": 1}],
            ),
            tel,
        )
        self.assertEqual(comp.last_active_rules, ["r_knock_approach"])
        self.assertLessEqual(cmd_over.vx, 0.0)  # remaining_m <= 0 安全屏障拦截前冲

    def test_blend_mixer_linear_weighting_and_json_roundtrip(self):
        """BLEND 加权混合器应对启用分支输出按权重归一化平均，且 JSON 往返完全保真。"""
        spec = StrategySpec(
            id="blend_test",
            name="测试加权混合",
            mixer_type="BLEND",
            rules=[
                BehaviorRule(
                    id="pd_left",
                    name="PD 左转分支",
                    behavior="pd_follow",
                    condition="any_detection",
                    enabled=True,
                    weight=0.75,
                    params={"targetFilter": "any", "kpTurn": 40.0, "kdTurn": 0.0, "kpDist": 0.0},
                ),
                BehaviorRule(
                    id="hold_zero",
                    name="零速锚定分支",
                    behavior="manual_hold",
                    condition="any_detection",
                    enabled=True,
                    weight=0.25,
                ),
            ],
            safety_shield=False,
        )
        restored = StrategySpec.from_dict(json.loads(spec.to_json()))
        self.assertEqual(restored.to_dict(), spec.to_dict())

        comp_blend = StrategyComposer(restored)
        comp_pure_pd = StrategyComposer(
            StrategySpec(
                id="pure_pd",
                name="pure",
                mixer_type="PRIORITY",
                rules=[restored.rules[0]],
                safety_shield=False,
            )
        )
        sig = SignalSnapshot(
            dt_s=0.1,
            face_valid=True,
            detections=[{"cx": 0.80, "cy": 0.50, "w": 0.20, "h": 0.20, "score": 0.9, "cls": 0}],
        )
        tel = RobotTelemetry(head_q=0.0)
        cmd_blend = comp_blend.step(sig, tel)
        cmd_pd = comp_pure_pd.step(sig, tel)

        self.assertAlmostEqual(cmd_blend.vyaw, 0.75 * cmd_pd.vyaw, places=4)
        self.assertAlmostEqual(cmd_blend.head, 0.75 * cmd_pd.head, places=4)
        self.assertEqual(comp_blend.last_active_rules, ["pd_left", "hold_zero"])

    def test_model_push_server_catalog_and_download(self):
        """局域网推送服务应正确列出 /models.json、/strategies.json 并提供 .zip 包下载。"""
        with tempfile.TemporaryDirectory() as tmp:
            policy = EndSidePolicyNet()
            policy.eval()
            res = export_double_knock_bundle(DoubleKnockNet(), tmp, window=48, image_size=96)
            zip_name = Path(res["zip_path"]).name

            srv = start_model_push_server(host="127.0.0.1", port=0, search_dirs=[Path(tmp)])
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                base = f"http://127.0.0.1:{srv.server_port}"
                cat = json.loads(urllib.request.urlopen(f"{base}/models.json", timeout=3).read().decode("utf-8"))
                self.assertEqual(len(cat["models"]), 1)
                self.assertEqual(cat["models"][0]["filename"], zip_name)
                self.assertEqual(cat["models"][0]["task"], "action")

                strats = json.loads(urllib.request.urlopen(f"{base}/strategies.json", timeout=3).read().decode("utf-8"))
                self.assertGreaterEqual(len(strats["strategies"]), 4)

                zip_bytes = urllib.request.urlopen(f"{base}/models/{zip_name}", timeout=3).read()
                with zipfile.ZipFile(Path(res["zip_path"]), "r") as zf:
                    self.assertEqual(len(zip_bytes), Path(res["zip_path"]).stat().st_size)
                    self.assertIn("model.tflite", zf.namelist())
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
