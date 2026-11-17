# -*- coding: utf-8 -*-
"""strategy_engine.py — RIG-Hover 跨平台（PC / Android）声明式策略组合引擎。

与 Android 端 `com.luwu.hovercontroller.strategy.*` 严格共享完全一致的 JSON Schema、
信号总线 (SignalBus)、优先级路由 (PRIORITY) 与加权混合器 (BLEND) 以及安全屏障 (EndSideSafetyShield)：

四大可插拔模型槽位 (Model Slots):
  - detector : 视觉目标检测模型 (人脸 / 拳头)
  - pose     : 姿态/手腕关键点模型 (可选)
  - action   : 时序动作模型 (DoubleKnockNet 双敲识别)
  - policy   : 39 维闭环强化学习策略 (EndSidePolicyNet)

内置可组合行为原语 (Behaviors):
  - scenario_v2 : 完整社交交互状态机 (看人脸 -> 优先看拳头 -> 双敲锁定 -> 里程计接近 -> 回看)
  - pd_follow   : 视觉框 PD 跟踪 (可过滤 targetFilter = any | face | fist)
  - rl_policy   : 端侧 39 维强化学习闭环输出
  - gaze_patrol : 头部正弦温和巡检扫描 (零轮速)
  - manual_hold : 原地待命保持 (零轮速)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    EndSideController,
    EndSideFeatureBuilder,
    EndSideSafetyShield,
    TargetLock,
)
from end_side_protocol import DriveCommand, RobotTelemetry  # noqa: E402


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class SignalSnapshot:
    """单周期感知 + 动作 + 里程计统一信号总线快照。"""

    now_s: float = 0.0
    dt_s: float = 0.1
    face_valid: bool = False
    theta_face_deg: float = 0.0
    face_scale: float = 0.0
    person_h: float = 0.0
    hand_valid: bool = False
    fist_prob: float = 0.0
    theta_fist_deg: float = 0.0
    p_impulse: float = 0.0
    p_double_knock: float = 0.0
    knock_double_fired: bool = False
    target_lock: TargetLock = field(default_factory=TargetLock)
    phase: int = MODE_IDLE
    detections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BehaviorRule:
    """策略拓扑中的一条执行节点/规则。"""

    id: str
    name: str
    behavior: str  # scenario_v2 | pd_follow | rl_policy | gaze_patrol | manual_hold
    condition: str = "always"  # always | face_valid | fist_valid | knock_double_fired | target_locked | target_locked_or_knock | any_detection
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BehaviorRule:
        return cls(
            id=str(d.get("id", "rule")),
            name=str(d.get("name", d.get("behavior", "rule"))),
            behavior=str(d.get("behavior", "manual_hold")),
            condition=str(d.get("condition", "always")),
            enabled=bool(d.get("enabled", True)),
            weight=float(d.get("weight", 1.0)),
            params=dict(d.get("params") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "behavior": self.behavior,
            "condition": self.condition,
            "enabled": self.enabled,
            "weight": float(self.weight),
            "params": dict(self.params),
        }


@dataclass
class StrategySpec:
    """PC 与 Android 严格对齐的声明式策略描述规格。"""

    schema_version: int = 1
    id: str = "social_knock_v2"
    name: str = "社交双敲迎宾"
    description: str = "盯人脸 -> 拳头优先 -> 双敲锁定目标 -> 走近停驻并回看"
    mixer_type: str = "PRIORITY"  # PRIORITY | BLEND
    slots: dict[str, str] = field(default_factory=dict)
    rules: list[BehaviorRule] = field(default_factory=list)
    safety_shield: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StrategySpec:
        return cls(
            schema_version=int(d.get("schemaVersion", 1)),
            id=str(d.get("id", "custom")),
            name=str(d.get("name", "自定义策略")),
            description=str(d.get("description", "")),
            mixer_type=str(d.get("mixerType", "PRIORITY")).upper(),
            slots={str(k): str(v) for k, v in (d.get("slots") or {}).items()},
            rules=[BehaviorRule.from_dict(r) for r in (d.get("rules") or [])],
            safety_shield=bool(d.get("safetyShield", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": int(self.schema_version),
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "mixerType": self.mixer_type,
            "slots": dict(self.slots),
            "rules": [r.to_dict() for r in self.rules],
            "safetyShield": bool(self.safety_shield),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def evaluate_condition(cond: str, sig: SignalSnapshot, controller: EndSideController | None = None) -> bool:
    """统一的触发条件判定，保证 PC 与 Android 规则命中语义完全一致。"""
    locked = sig.target_lock.valid == 1 or (controller is not None and controller.target_lock.valid == 1)
    if cond == "always":
        return True
    if cond == "face_valid":
        return sig.face_valid
    if cond == "fist_valid":
        return sig.hand_valid and sig.fist_prob >= 0.55
    if cond == "knock_double_fired":
        return sig.knock_double_fired
    if cond == "target_locked":
        return locked
    if cond == "target_locked_or_knock":
        return locked or sig.knock_double_fired
    if cond == "any_detection":
        return bool(sig.detections) or sig.face_valid or sig.hand_valid
    return False


class StrategyComposer:
    """可执行 PRIORITY / BLEND 策略拓扑的组合式推理引擎。"""

    def __init__(
        self,
        spec: StrategySpec,
        policy_model: Any | None = None,
    ):
        self.spec = spec
        self.scenario_ctrl = EndSideController(policy_model=policy_model)
        self.shield = EndSideSafetyShield()
        self._last_err_x: dict[str, float] = {}
        self._patrol_phase: float = 0.0
        self.last_active_rules: list[str] = []

    def set_policy_model(self, policy_model: Any | None) -> None:
        if policy_model is not None:
            self.scenario_ctrl.policy = policy_model

    def reset(self) -> None:
        self.scenario_ctrl.reset()
        self.shield.reset()
        self._last_err_x.clear()
        self._patrol_phase = 0.0
        self.last_active_rules = []

    def _run_pd_follow(self, rule: BehaviorRule, sig: SignalSnapshot, telemetry: RobotTelemetry) -> DriveCommand:
        target_filter = str(rule.params.get("targetFilter", "any")).lower()
        kp_turn = float(rule.params.get("kpTurn", 48.0))
        kd_turn = float(rule.params.get("kdTurn", 6.0))
        kp_dist = float(rule.params.get("kpDist", 80.0))
        target_size = float(rule.params.get("targetSize", 0.32))

        best = None
        best_area = -1.0
        for d in sig.detections:
            score = float(d.get("score", 0.0))
            w = float(d.get("w", 0.0))
            h = float(d.get("h", 0.0))
            if score < 0.35 or w <= 0.0 or h <= 0.0:
                continue
            cls_id = int(d.get("cls", 0))
            if target_filter == "face" and cls_id != 0:
                continue
            if target_filter == "fist" and cls_id != 1:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = d

        if best is None:
            # 退回到 SignalSnapshot 标量角偏差（例如仅由人脸/手势层提供角度时）
            if target_filter in ("fist", "any") and sig.hand_valid and sig.fist_prob >= 0.55:
                head_cmd = _clamp(telemetry.head_q + sig.theta_fist_deg * 0.45, -45.0, 45.0)
                return DriveCommand(vx=0.0, vyaw=_clamp(-sig.theta_fist_deg * 1.1, -45.0, 45.0), head=head_cmd)
            if target_filter in ("face", "any") and sig.face_valid:
                head_cmd = _clamp(telemetry.head_q + sig.theta_face_deg * 0.45, -45.0, 45.0)
                return DriveCommand(vx=0.0, vyaw=_clamp(-sig.theta_face_deg * 0.9, -40.0, 40.0), head=head_cmd)
            return DriveCommand()

        err_x = (float(best["cx"]) - 0.5) * 2.0
        dt = _clamp(sig.dt_s, 0.02, 0.6)
        prev = self._last_err_x.get(rule.id, err_x)
        deriv = (err_x - prev) / dt
        self._last_err_x[rule.id] = err_x

        vyaw = 0.0 if abs(err_x) < 0.06 else _clamp(-(kp_turn * err_x + kd_turn * deriv), -65.0, 65.0)
        size_err = target_size - float(best["h"])
        vx = 0.0 if (kp_dist <= 0.0 or abs(size_err) < 0.03) else _clamp(kp_dist * size_err, -60.0, 60.0)
        head = _clamp(telemetry.head_q + (float(best["cx"]) - 0.5) * 45.0, -45.0, 45.0)
        return DriveCommand(vx=vx, vyaw=vyaw, head=head).sanitized()

    def _run_rl_policy(self, rule: BehaviorRule, sig: SignalSnapshot, telemetry: RobotTelemetry) -> DriveCommand:
        pm = self.scenario_ctrl.policy
        if pm is None:
            return DriveCommand(vx=0.0, vyaw=0.0, head=telemetry.head_q)
        lock = sig.target_lock if sig.target_lock.valid else self.scenario_ctrl.target_lock
        feat = EndSideFeatureBuilder.build(
            status=telemetry,
            detections=sig.detections,
            target_lock=lock if lock.valid else None,
        )
        return pm.infer_numpy(feat).sanitized()

    def _run_scenario_v2(self, rule: BehaviorRule, sig: SignalSnapshot, telemetry: RobotTelemetry) -> DriveCommand:
        scene_obs = {
            "face_valid": 1 if sig.face_valid else 0,
            "theta_face": float(sig.theta_face_deg),
            "face_scale": float(sig.face_scale),
            "person_h": float(sig.person_h),
            "hand_valid": 1 if sig.hand_valid else 0,
            "fist_prob": float(sig.fist_prob),
            "theta_fist": float(sig.theta_fist_deg),
            "knock_fired": 1 if sig.knock_double_fired else 0,
            "detections": sig.detections,
        }
        return self.scenario_ctrl.step(dt=sig.dt_s, telemetry=telemetry, scene_obs=scene_obs)

    def _run_gaze_patrol(self, rule: BehaviorRule, sig: SignalSnapshot) -> DriveCommand:
        amp = float(rule.params.get("ampDeg", 25.0))
        freq_hz = float(rule.params.get("freqHz", 0.25))
        self._patrol_phase += 2.0 * math.pi * freq_hz * _clamp(sig.dt_s, 0.01, 0.5)
        head = _clamp(amp * math.sin(self._patrol_phase), -45.0, 45.0)
        return DriveCommand(vx=0.0, vyaw=0.0, head=head)

    def _eval_rule(self, rule: BehaviorRule, sig: SignalSnapshot, telemetry: RobotTelemetry) -> DriveCommand:
        b = rule.behavior.lower()
        if b == "scenario_v2":
            return self._run_scenario_v2(rule, sig, telemetry)
        if b == "pd_follow":
            return self._run_pd_follow(rule, sig, telemetry)
        if b == "rl_policy":
            return self._run_rl_policy(rule, sig, telemetry)
        if b == "gaze_patrol":
            return self._run_gaze_patrol(rule, sig)
        return DriveCommand(vx=0.0, vyaw=0.0, head=telemetry.head_q)

    def step(self, sig: SignalSnapshot, telemetry: RobotTelemetry) -> DriveCommand:
        """按 PRIORITY 优先级或 BLEND 加权模式求解最终动作并经统一安全屏障裁剪。"""
        enabled_rules = [r for r in self.spec.rules if r.enabled]
        self.last_active_rules = []
        raw_cmd = DriveCommand(vx=0.0, vyaw=0.0, head=telemetry.head_q)

        if self.spec.mixer_type == "BLEND":
            total_w = 0.0
            sum_vx, sum_vyaw, sum_head = 0.0, 0.0, 0.0
            for rule in enabled_rules:
                w = max(0.0, float(rule.weight))
                if w <= 1e-6 or not evaluate_condition(rule.condition, sig, self.scenario_ctrl):
                    continue
                cmd = self._eval_rule(rule, sig, telemetry)
                sum_vx += w * cmd.vx
                sum_vyaw += w * cmd.vyaw
                sum_head += w * cmd.head
                total_w += w
                self.last_active_rules.append(rule.id)
            if total_w > 1e-6:
                raw_cmd = DriveCommand(
                    vx=sum_vx / total_w,
                    vyaw=sum_vyaw / total_w,
                    head=sum_head / total_w,
                ).sanitized()
        else:
            # PRIORITY 模式：第一条命中的活跃规则获胜
            for rule in enabled_rules:
                if evaluate_condition(rule.condition, sig, self.scenario_ctrl):
                    raw_cmd = self._eval_rule(rule, sig, telemetry)
                    self.last_active_rules.append(rule.id)
                    break

        if not self.spec.safety_shield:
            return raw_cmd.sanitized()

        lock = self.scenario_ctrl.target_lock if self.scenario_ctrl.target_lock.valid else sig.target_lock
        mode = self.scenario_ctrl.mode if self.scenario_ctrl.mode != MODE_IDLE else sig.phase
        return self.shield.apply(
            raw_cmd,
            mode=mode,
            remaining_m=float(lock.remaining_m),
            pitch_deg=float(telemetry.pitch),
        )


def builtin_presets() -> list[StrategySpec]:
    """与 Android 端完全同步的 4 套内置可组合策略预设。"""
    return [
        StrategySpec(
            id="social_knock_v2",
            name="社交双敲迎宾 (Scenario v2)",
            description="盯人脸 -> 有拳头优先盯拳头 -> 敲两下锁定坐标绕过去并零过冲停住 -> 回看人脸",
            mixer_type="PRIORITY",
            slots={"action": "double-knock-v1", "policy": "hover-end-side-rl-ppo-v1"},
            rules=[
                BehaviorRule(
                    id="r_scenario_v2",
                    name="Scenario v2 端到端社交状态机",
                    behavior="scenario_v2",
                    condition="always",
                    enabled=True,
                    weight=1.0,
                ),
            ],
            safety_shield=True,
        ),
        StrategySpec(
            id="priority_composed_social",
            name="可编程优先级规则链 (拳头 > 人脸 > 巡检)",
            description="使用声明式优先级规则组合：双敲锁定奔赴 > 拳头凝视 > 人脸跟随 > 头部微扫巡检",
            mixer_type="PRIORITY",
            slots={"action": "double-knock-v1", "policy": "hover-end-side-rl-ppo-v1"},
            rules=[
                BehaviorRule(
                    id="r_knock_approach",
                    name="双敲锁定 -> RL 闭环接近",
                    behavior="rl_policy",
                    condition="target_locked_or_knock",
                    enabled=True,
                    weight=1.0,
                ),
                BehaviorRule(
                    id="r_fist_first",
                    name="发现拳头优先盯视",
                    behavior="pd_follow",
                    condition="fist_valid",
                    enabled=True,
                    weight=1.0,
                    params={"targetFilter": "fist", "kpTurn": 55.0, "kdTurn": 6.0, "kpDist": 0.0},
                ),
                BehaviorRule(
                    id="r_face_follow",
                    name="无人手动作时盯人脸",
                    behavior="pd_follow",
                    condition="face_valid",
                    enabled=True,
                    weight=1.0,
                    params={"targetFilter": "face", "kpTurn": 45.0, "kdTurn": 5.0, "kpDist": 0.0},
                ),
                BehaviorRule(
                    id="r_idle_patrol",
                    name="闲置微扫巡检",
                    behavior="gaze_patrol",
                    condition="always",
                    enabled=True,
                    weight=1.0,
                    params={"ampDeg": 24.0, "freqHz": 0.25},
                ),
            ],
            safety_shield=True,
        ),
        StrategySpec(
            id="rl_pd_hybrid_blend",
            name="连续混合策略 (70% RL + 30% 视觉PD)",
            description="同时运行端侧 RL 策略与视觉框 PD 跟踪并按连续权重融合，兼顾泛化与稳态阻尼",
            mixer_type="BLEND",
            slots={"policy": "hover-end-side-rl-ppo-v1"},
            rules=[
                BehaviorRule(
                    id="r_rl_branch",
                    name="端侧 RL 闭环策略分支",
                    behavior="rl_policy",
                    condition="any_detection",
                    enabled=True,
                    weight=0.70,
                ),
                BehaviorRule(
                    id="r_pd_branch",
                    name="经典视觉 PD 阻尼分支",
                    behavior="pd_follow",
                    condition="any_detection",
                    enabled=True,
                    weight=0.30,
                    params={"targetFilter": "any", "kpTurn": 46.0, "kdTurn": 7.0, "kpDist": 75.0, "targetSize": 0.34},
                ),
            ],
            safety_shield=True,
        ),
        StrategySpec(
            id="pure_rl_policy",
            name="纯净 39 维端侧 RL 策略",
            description="直接由端侧强化学习网络闭环输出 [vx, vyaw, head]（经安全屏障校验）",
            mixer_type="PRIORITY",
            slots={"policy": "hover-end-side-rl-ppo-v1"},
            rules=[
                BehaviorRule(
                    id="r_pure_rl",
                    name="端侧 RL 策略直接驾驶",
                    behavior="rl_policy",
                    condition="always",
                    enabled=True,
                    weight=1.0,
                ),
            ],
            safety_shield=True,
        ),
    ]
