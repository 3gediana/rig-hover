# -*- coding: utf-8 -*-
"""telemetry_flywheel.py — 端侧遥测黑匣子回放、TFLite 反事实策略评估、Huber/IRLS 系统辨识与 3D 高斯点云导出。

核心能力：
1. Blackbox JSONL 解析与序列化：兼容真机 `/status` 嵌套结构、扁平遥测结构以及 Android 端驼峰字段
   (`t_ms`, `status`, `detections`, `target_lock`, `cmd`, `mode`)；
2. `evaluate_counterfactual_tflite(records, tflite_path)`：基于 `run_tflite_flatbuffer_cpu`
   在离线黑匣子轨迹上逐帧回放任意 `.tflite` 策略网络，精确统计 `mean_action_l1_diff`、
   `jerk_index`、`pitch_rms_deg` 与 `shield_intervention_rate`；
3. `auto_calibrate_system_id(records)`：基于 Theil-Sen 初值 + Huber IRLS 鲁棒估计器，从接近轨迹日志中
   自动恢复相机测距常数 `dist_k`、底层 LQR 稳态跟随滞后 `lqr_pos_lag_m` 与航向死区 `yaw_deadband_deg`；
4. `export_gaussian_ply(splats_xyz_rgb, out_ply_path)`：将端侧重建的 3D 彩色高斯椭球点导出为标准
   ASCII `.ply` 格式，可直接拖入 MeshLab / CloudCompare / Blender 检视。
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PERC_DIR = HERE.parent / "perception"
for _p in (str(HERE), str(PERC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from end_side_controller import (  # noqa: E402
    MODE_APPROACH,
    MODE_DWELL,
    MODE_GAZE_FACE,
    MODE_GAZE_FIST,
    MODE_IDLE,
    MODE_NAMES,
    EndSideFeatureBuilder,
    EndSideSafetyShield,
    TargetLock,
)
from end_side_protocol import DriveCommand, RobotTelemetry, _clamp, parse_status_json  # noqa: E402
from export_tflite_bundle import run_tflite_flatbuffer_cpu  # noqa: E402
from spatial_memory import wrap180  # noqa: E402

_MODE_NAME_TO_INT: dict[str, int] = {
    "IDLE": MODE_IDLE,
    "MODE_IDLE": MODE_IDLE,
    "GAZE_FACE": MODE_GAZE_FACE,
    "MODE_GAZE_FACE": MODE_GAZE_FACE,
    "GAZE_FIST": MODE_GAZE_FIST,
    "MODE_GAZE_FIST": MODE_GAZE_FIST,
    "APPROACH": MODE_APPROACH,
    "MODE_APPROACH": MODE_APPROACH,
    "DWELL": MODE_DWELL,
    "MODE_DWELL": MODE_DWELL,
}


@dataclass
class BlackboxRecord:
    """单帧端侧 20Hz 飞行黑匣子日志记录。"""

    t_ms: int
    status: RobotTelemetry
    detections: list[dict[str, float]] = field(default_factory=list)
    target_lock: TargetLock | None = None
    cmd: DriveCommand = field(default_factory=DriveCommand)
    mode: int = MODE_IDLE
    clearance_m: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        st = self.status
        lock_dict = None
        if self.target_lock is not None and self.target_lock.valid:
            lock_dict = {
                "valid": int(self.target_lock.valid),
                "bearing_body": round(float(self.target_lock.bearing_body), 4),
                "dist_m": round(float(self.target_lock.dist_m), 4),
                "remaining_m": round(float(self.target_lock.remaining_m), 4),
                "clearance_m": round(float(self.clearance_m), 4),
            }
        return {
            "t_ms": int(self.t_ms),
            "mode": MODE_NAMES.get(int(self.mode), "IDLE"),
            "clearance_m": round(float(self.clearance_m), 4),
            "status": {
                "t": int(st.t_ms),
                "imu": {
                    "roll": round(float(st.roll), 4),
                    "pitch": round(float(st.pitch), 4),
                    "yaw": round(float(st.yaw), 4),
                    "ax": round(float(st.ax), 4),
                    "ay": round(float(st.ay), 4),
                    "az": round(float(st.az), 4),
                    "pit_gyro": round(float(st.pit_gyro), 4),
                    "dq": round(float(st.dq), 4),
                },
                "wheel": {
                    "v1": round(float(st.v1), 4),
                    "v2": round(float(st.v2), 4),
                    "x1": round(float(st.x1), 5),
                    "x2": round(float(st.x2), 5),
                },
                "head": {
                    "q": round(float(st.head_q), 4),
                    "target": round(float(st.head_target), 4),
                    "gimbal_target": round(float(st.gimbal_target), 4),
                },
                "cmd": {
                    "vx": round(float(st.cmd_vx), 4),
                    "vyaw": round(float(st.cmd_vyaw), 4),
                    "mode": int(st.cmd_mode),
                    "pos": round(float(st.cmd_pos), 5),
                    "yaw_target": round(float(st.cmd_yaw_target), 4),
                },
                "torque": {
                    "t1": round(float(st.t1), 4),
                    "t2": round(float(st.t2), 4),
                },
                "pid": {
                    "pos_e": round(float(st.pos_e), 4),
                    "pos_u": round(float(st.pos_u), 4),
                    "vel_e": round(float(st.vel_e), 4),
                    "vel_u": round(float(st.vel_u), 4),
                    "pit_e": round(float(st.pit_e), 4),
                    "pit_u": round(float(st.pit_u), 4),
                },
                "heap": int(st.heap),
            },
            "detections": [
                {
                    "cx": round(float(d.get("cx", 0.5)), 4),
                    "cy": round(float(d.get("cy", 0.5)), 4),
                    "w": round(float(d.get("w", 0.0)), 4),
                    "h": round(float(d.get("h", 0.0)), 4),
                    "score": round(float(d.get("score", d.get("conf", 0.0))), 4),
                    "cls": int(d.get("cls", d.get("classId", 0))),
                }
                for d in self.detections
            ],
            "target_lock": lock_dict,
            "cmd": {
                "vx": round(float(self.cmd.vx), 4),
                "vyaw": round(float(self.cmd.vyaw), 4),
                "head": round(float(self.cmd.head), 4),
                "stop": bool(self.cmd.stop),
            },
        }


def _parse_telemetry_flexible(raw_status: dict[str, Any], default_t_ms: int = 0) -> RobotTelemetry:
    """兼容嵌套 `/status` JSON 与扁平 snake_case / camelCase 遥测字典。"""
    if any(k in raw_status for k in ("imu", "wheel", "head", "pid")):
        tel = parse_status_json(raw_status)
        if tel.t_ms == 0 and default_t_ms != 0:
            tel.t_ms = int(default_t_ms)
        return tel

    def _f(*keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in raw_status and raw_status[k] is not None:
                return float(raw_status[k])
        return default

    t_val = int(raw_status.get("t_ms", raw_status.get("t", default_t_ms)) or 0)
    return RobotTelemetry(
        t_ms=t_val,
        roll=_f("roll"),
        pitch=_f("pitch"),
        yaw=_f("yaw"),
        ax=_f("ax"),
        ay=_f("ay"),
        az=_f("az", default=9.81),
        pit_gyro=_f("pit_gyro", "pitGyro"),
        dq=_f("dq"),
        v1=_f("v1"),
        v2=_f("v2"),
        x1=_f("x1"),
        x2=_f("x2"),
        head_q=_f("head_q", "headQ", "q_head"),
        head_target=_f("head_target", "headTarget"),
        gimbal_target=_f("gimbal_target", "gimbalTarget"),
        cmd_vx=_f("cmd_vx", "cmdVx"),
        cmd_vyaw=_f("cmd_vyaw", "cmdVyaw"),
        cmd_mode=int(raw_status.get("cmd_mode", raw_status.get("cmdMode", 0)) or 0),
        cmd_pos=_f("cmd_pos", "cmdPos"),
        cmd_yaw_target=_f("cmd_yaw_target", "cmdYawTarget"),
        t1=_f("t1"),
        t2=_f("t2"),
        pos_e=_f("pos_e", "posE"),
        pos_u=_f("pos_u", "posU"),
        vel_e=_f("vel_e", "velE"),
        vel_u=_f("vel_u", "velU"),
        pit_e=_f("pit_e", "pitE"),
        pit_u=_f("pit_u", "pitU"),
        heap=int(raw_status.get("heap", 0) or 0),
    )


def parse_blackbox_record(obj: dict[str, Any] | str) -> BlackboxRecord:
    """解析单条 JSONL 黑匣子记录（支持 JSON 字符串或已反序列化的字典）。"""
    data = json.loads(obj) if isinstance(obj, str) else dict(obj)
    raw_status = data.get("status")
    t_ms_top = int(data.get("t_ms", data.get("t", 0)) or 0)

    if isinstance(raw_status, dict):
        status = _parse_telemetry_flexible(raw_status, default_t_ms=t_ms_top)
    else:
        status = _parse_telemetry_flexible(data, default_t_ms=t_ms_top)

    t_ms = t_ms_top if t_ms_top != 0 else int(status.t_ms)

    raw_dets = data.get("detections") or []
    detections: list[dict[str, float]] = []
    for d in raw_dets:
        if not isinstance(d, dict):
            continue
        detections.append(
            {
                "cx": float(d.get("cx", 0.5)),
                "cy": float(d.get("cy", 0.5)),
                "w": float(d.get("w", 0.0)),
                "h": float(d.get("h", 0.0)),
                "score": float(d.get("score", d.get("conf", 0.0))),
                "cls": int(d.get("cls", d.get("classId", 0))),
            }
        )

    raw_lock = data.get("target_lock") if "target_lock" in data else data.get("targetLock")
    target_lock: TargetLock | None = None
    clearance_m = float(data.get("clearance_m", data.get("clearanceM", 1.0)) or 1.0)
    if isinstance(raw_lock, dict) and int(raw_lock.get("valid", 0) or 0) == 1:
        if "clearance_m" in raw_lock or "clearanceM" in raw_lock:
            clearance_m = float(raw_lock.get("clearance_m", raw_lock.get("clearanceM", clearance_m)))
        target_lock = TargetLock(
            valid=1,
            bearing_body=float(raw_lock.get("bearing_body", raw_lock.get("bearingBodyDeg", 0.0))),
            dist_m=float(raw_lock.get("dist_m", raw_lock.get("distM", 0.0))),
            remaining_m=float(raw_lock.get("remaining_m", raw_lock.get("remainingM", 0.0))),
        )

    raw_cmd = data.get("cmd")
    if isinstance(raw_cmd, dict):
        cmd = DriveCommand(
            vx=float(raw_cmd.get("vx", 0.0)),
            vyaw=float(raw_cmd.get("vyaw", 0.0)),
            head=float(raw_cmd.get("head", status.head_target)),
            stop=bool(raw_cmd.get("stop", False)),
        ).sanitized()
    else:
        cmd = DriveCommand(
            vx=float(status.cmd_vx),
            vyaw=float(status.cmd_vyaw),
            head=float(status.head_target),
            stop=False,
        ).sanitized()

    raw_mode = data.get("mode")
    if isinstance(raw_mode, int):
        mode = int(raw_mode)
    elif isinstance(raw_mode, str):
        m_up = raw_mode.strip().upper()
        if m_up.isdigit():
            mode = int(m_up)
        else:
            mode = _MODE_NAME_TO_INT.get(m_up, MODE_APPROACH if (target_lock is not None or abs(cmd.vx) > 0.5) else MODE_IDLE)
    else:
        mode = MODE_APPROACH if (target_lock is not None or abs(cmd.vx) > 0.5) else MODE_IDLE

    return BlackboxRecord(
        t_ms=t_ms,
        status=status,
        detections=detections,
        target_lock=target_lock,
        cmd=cmd,
        mode=mode,
        clearance_m=clearance_m,
    )


def parse_blackbox_jsonl(source: str | bytes | Path) -> list[BlackboxRecord]:
    """解析多行 JSONL 黑匣子日志（输入可为文件路径、UTF-8 字节串或 JSONL 文本）。"""
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, bytes):
        text = source.decode("utf-8")
    elif isinstance(source, str):
        stripped = source.strip()
        if "\n" not in stripped and not stripped.startswith("{") and Path(stripped).is_file():
            text = Path(stripped).read_text(encoding="utf-8")
        else:
            text = source
    else:
        raise TypeError(f"Unsupported blackbox source type: {type(source)!r}")

    records: list[BlackboxRecord] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        records.append(parse_blackbox_record(s))
    return records


def serialize_blackbox_jsonl(records: list[BlackboxRecord]) -> str:
    """将 BlackboxRecord 列表序列化为标准 UTF-8 JSONL 字符串。"""
    return "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in records)


def dump_blackbox_jsonl(records: list[BlackboxRecord], out_path: str | Path) -> Path:
    """将 BlackboxRecord 列表写入磁盘 `.jsonl` 文件。"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(serialize_blackbox_jsonl(records), encoding="utf-8")
    return p


def _ensure_records(records: list[BlackboxRecord] | str | bytes | Path) -> list[BlackboxRecord]:
    if isinstance(records, (str, bytes, Path)):
        return parse_blackbox_jsonl(records)
    return list(records)


def evaluate_counterfactual_tflite(
    records: list[BlackboxRecord] | str | bytes | Path,
    tflite_path: str | Path,
) -> dict[str, Any]:
    """在离线黑匣子轨迹上通过 `run_tflite_flatbuffer_cpu` 回放候选 `.tflite` 策略并量化安全性与平滑度。

    返回核心指标：
      - `mean_action_l1_diff`      : 候选策略动作与录制指令在 [vx, vyaw, head] 三通道上的平均 L1 偏差
      - `jerk_index`               : 候选策略归一化单步动作跳变率 (|Δvx|/100 + |Δvyaw|/100 + |Δhead|/75) / 3
      - `pitch_rms_deg`            : 轨迹实测俯仰角均方根 (deg)
      - `shield_intervention_rate` : 安全屏障 `EndSideSafetyShield` 对候选策略输出的介入裁剪比例 [0..1]
    """
    recs = _ensure_records(records)
    tfl_file = Path(tflite_path)
    if not tfl_file.is_file():
        raise FileNotFoundError(f"TFLite policy model not found: {tfl_file}")
    if not recs:
        return {
            "frame_count": 0,
            "mean_action_l1_diff": 0.0,
            "jerk_index": 0.0,
            "pitch_rms_deg": 0.0,
            "shield_intervention_rate": 0.0,
        }

    shield = EndSideSafetyShield()
    shield.reset()

    cf_actions: list[np.ndarray] = []
    logged_actions: list[np.ndarray] = []
    pitches: list[float] = []
    interventions = 0

    for rec in recs:
        feat_39 = EndSideFeatureBuilder.build(
            status=rec.status,
            detections=rec.detections,
            target_lock=rec.target_lock,
            clearance_m=rec.clearance_m,
        )
        tfl_out = run_tflite_flatbuffer_cpu(tfl_file, [feat_39.reshape(1, 39)])[0][0]
        raw_cmd = DriveCommand(
            vx=float(tfl_out[0]),
            vyaw=float(tfl_out[1]),
            head=float(tfl_out[2]),
        ).sanitized()

        if rec.target_lock is not None and rec.target_lock.valid:
            rem_m = float(rec.target_lock.remaining_m)
        elif rec.detections and float(rec.detections[0].get("h", 0.0)) > 0.05:
            rem_m = max(0.0, 0.24 / float(rec.detections[0]["h"]) - 0.12)
        else:
            rem_m = 1.0 if rec.mode == MODE_APPROACH else 0.0

        safe_cmd = shield.apply(
            raw_cmd,
            mode=rec.mode,
            remaining_m=rem_m,
            pitch_deg=rec.status.pitch,
            clearance_m=rec.clearance_m,
        )
        if (
            abs(safe_cmd.vx - raw_cmd.vx) > 0.5
            or abs(safe_cmd.vyaw - raw_cmd.vyaw) > 0.5
            or abs(safe_cmd.head - raw_cmd.head) > 0.5
        ):
            interventions += 1

        cf_actions.append(np.array([raw_cmd.vx, raw_cmd.vyaw, raw_cmd.head], dtype=np.float32))
        logged_actions.append(np.array([rec.cmd.vx, rec.cmd.vyaw, rec.cmd.head], dtype=np.float32))
        pitches.append(float(rec.status.pitch))

    cf_mat = np.stack(cf_actions, axis=0)
    log_mat = np.stack(logged_actions, axis=0)
    scale = np.array([100.0, 100.0, 75.0], dtype=np.float32)

    mean_l1_diff = float(np.mean(np.abs(cf_mat - log_mat)))
    mean_l1_norm = float(np.mean(np.abs(cf_mat - log_mat) / scale))

    if len(cf_mat) >= 2:
        diffs_norm = np.abs(np.diff(cf_mat, axis=0)) / scale
        jerk_index = float(np.mean(diffs_norm))
    else:
        jerk_index = 0.0

    pitch_arr = np.asarray(pitches, dtype=np.float64)
    pitch_rms_deg = float(math.sqrt(float(np.mean(pitch_arr ** 2))))
    shield_rate = float(interventions) / float(len(recs))

    return {
        "frame_count": len(recs),
        "mean_action_l1_diff": round(mean_l1_diff, 5),
        "mean_action_l1_norm": round(mean_l1_norm, 6),
        "jerk_index": round(jerk_index, 6),
        "pitch_rms_deg": round(pitch_rms_deg, 4),
        "shield_intervention_rate": round(shield_rate, 5),
    }


def _huber_weights(residuals: np.ndarray, delta_scale: float = 1.345) -> tuple[np.ndarray, float]:
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    sigma = max(1.4826 * mad, 1e-4)
    delta = max(delta_scale * sigma, 1e-4)
    abs_r = np.abs(residuals)
    w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-12))
    return w.astype(np.float64), delta


def _huber_location_1d(samples: np.ndarray, delta_scale: float = 1.345, n_iters: int = 20) -> float:
    """一维 Huber M-估计器（中位数初值 + IRLS 迭代），50% 崩溃点且高斯渐近高效。"""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return 0.0
    mu = float(np.median(x))
    for _ in range(n_iters):
        w, _ = _huber_weights(x - mu, delta_scale=delta_scale)
        denom = float(np.sum(w))
        if denom <= 1e-12:
            break
        mu_new = float(np.sum(w * x) / denom)
        if abs(mu_new - mu) < 1e-9:
            mu = mu_new
            break
        mu = mu_new
    return mu


def _theil_sen_huber_irls_1d(
    x: np.ndarray,
    y: np.ndarray,
    delta_scale: float = 1.345,
    n_iters: int = 25,
) -> tuple[float, float, float]:
    """一维仿射回归 y = a + b * x：Theil-Sen 中位斜率初值 + Huber IRLS 精化，返回 (a, b, inlier_ratio)。"""
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(xv)
    if n < 2:
        return (float(yv[0]) if n == 1 else 0.0), 0.0, 1.0

    # 1. Theil-Sen 配对中位斜率初值 (免疫高达 29%~50% 的离群野值)
    slopes: list[float] = []
    step = max(1, n // 64)
    idx_sub = list(range(0, n, step))
    for i_pos, i in enumerate(idx_sub):
        for j in idx_sub[i_pos + 1 :]:
            dx = xv[j] - xv[i]
            if abs(dx) > 1e-5:
                slopes.append((yv[j] - yv[i]) / dx)
    b = float(np.median(slopes)) if slopes else 0.0
    a = float(np.median(yv - b * xv))

    # 2. Huber IRLS 加权最小二乘迭代
    X_mat = np.column_stack([np.ones_like(xv), xv])
    inlier_ratio = 1.0
    for _ in range(n_iters):
        res = yv - (a + b * xv)
        w, delta = _huber_weights(res, delta_scale=delta_scale)
        inlier_ratio = float(np.mean(np.abs(res) <= delta * 1.5))
        W_X = X_mat * w[:, None]
        lhs = X_mat.T @ W_X + 1e-9 * np.eye(2)
        rhs = X_mat.T @ (w * yv)
        theta = np.linalg.solve(lhs, rhs)
        if abs(float(theta[0]) - a) + abs(float(theta[1]) - b) < 1e-9:
            a, b = float(theta[0]), float(theta[1])
            break
        a, b = float(theta[0]), float(theta[1])
    return a, b, inlier_ratio


def _huber_hinge_deadband(
    u_abs: np.ndarray,
    w_abs: np.ndarray,
    delta_scale: float = 1.345,
) -> tuple[float, float]:
    """基于 Huber 损失与 IRLS 的死区铰链模型 w = K * max(0, u - d) 全局辨识，返回 (deadband_d, gain_K)。"""
    u = np.asarray(u_abs, dtype=np.float64).reshape(-1)
    w = np.asarray(w_abs, dtype=np.float64).reshape(-1)
    if len(u) < 4 or float(np.max(u)) < 1.0:
        return 1.5, 1.0

    def _fit_gain_for_d(d_cand: float) -> tuple[float, float]:
        z = np.maximum(0.0, u - d_cand)
        z_norm2 = float(np.dot(z, z))
        if z_norm2 < 1e-6:
            return 1.0, float("inf")
        # Theil-Sen 过原点斜率初值
        active = z > 0.5
        if int(np.sum(active)) >= 2:
            k_est = float(np.median(w[active] / z[active]))
        else:
            k_est = float(np.dot(z, w) / z_norm2)
        k_est = float(_clamp(k_est, 0.15, 4.0))

        for _ in range(12):
            res = w - k_est * z
            wt, _ = _huber_weights(res, delta_scale=delta_scale)
            denom = float(np.sum(wt * z * z))
            if denom <= 1e-9:
                break
            k_est = float(_clamp(float(np.sum(wt * z * w)) / denom, 0.15, 4.0))

        res = w - k_est * z
        med = float(np.median(res))
        sigma = max(1.4826 * float(np.median(np.abs(res - med))), 0.15)
        delta = delta_scale * sigma
        abs_r = np.abs(res)
        # 固定标度下的标准 Huber 目标值 (采用全局鲁棒标度使不同 d 之间严格可比)
        huber_loss = np.where(abs_r <= delta, 0.5 * abs_r ** 2, delta * (abs_r - 0.5 * delta))
        return k_est, float(np.sum(huber_loss))

    # 先用活跃转向段做仿射回归获取高质量参考中心与固定标度
    active_mask = (w > 1.2) & (u > 1.5)
    if int(np.sum(active_mask)) >= 3:
        c0, m0, _ = _theil_sen_huber_irls_1d(u[active_mask], w[active_mask], delta_scale=delta_scale)
        d_init = float(_clamp(-c0 / m0, 0.0, 20.0)) if m0 > 0.10 else 1.5
        res_ref = w[active_mask] - (c0 + m0 * u[active_mask])
        fixed_sigma = max(1.4826 * float(np.median(np.abs(res_ref - np.median(res_ref)))), 0.12)
    else:
        d_init = 1.5
        fixed_sigma = 0.25

    fixed_delta = delta_scale * fixed_sigma

    def _eval_fixed_scale(d_cand: float) -> tuple[float, float]:
        z = np.maximum(0.0, u - d_cand)
        active = z > 0.4
        if int(np.sum(active)) < 2:
            return 1.0, float("inf")
        k_est = float(_clamp(float(np.median(w[active] / z[active])), 0.15, 4.0))
        for _ in range(12):
            res = w - k_est * z
            abs_r = np.abs(res)
            wt = np.where(abs_r <= fixed_delta, 1.0, fixed_delta / np.maximum(abs_r, 1e-12))
            denom = float(np.sum(wt * z * z))
            if denom <= 1e-9:
                break
            k_est = float(_clamp(float(np.sum(wt * z * w)) / denom, 0.15, 4.0))
        res = w - k_est * z
        abs_r = np.abs(res)
        loss = float(np.sum(np.where(abs_r <= fixed_delta, 0.5 * abs_r ** 2, fixed_delta * (abs_r - 0.5 * fixed_delta))))
        return k_est, loss

    # 粗网格 (0..18°, 步长 0.25°) + 细网格 (±0.6°, 步长 0.01°) 全局搜索
    u_max_search = min(18.0, float(np.percentile(u, 85)))
    coarse_grid = np.linspace(0.0, max(0.5, u_max_search), 73)
    best_d = d_init
    best_k, best_loss = _eval_fixed_scale(best_d)
    for d_c in coarse_grid:
        k_c, l_c = _eval_fixed_scale(float(d_c))
        if l_c < best_loss:
            best_d, best_k, best_loss = float(d_c), k_c, l_c

    fine_lo = max(0.0, best_d - 0.60)
    fine_hi = min(20.0, best_d + 0.60)
    for d_f in np.linspace(fine_lo, fine_hi, 121):
        k_f, l_f = _eval_fixed_scale(float(d_f))
        if l_f < best_loss:
            best_d, best_k, best_loss = float(d_f), k_f, l_f

    return best_d, best_k


def auto_calibrate_system_id(
    records: list[BlackboxRecord] | str | bytes | Path,
    huber_delta_scale: float = 1.345,
) -> dict[str, Any]:
    """从接近轨迹黑匣子日志中执行 Huber / IRLS 鲁棒系统辨识。

    自动恢复三大关键物理/感知耦合参数：
      1. `dist_k`           : 单目视觉框高度测距常数 (d_m = dist_k / box_h)
      2. `lqr_pos_lag_m`    : 167Hz LQR 平衡环在巡航接近阶段的稳态位置滞后 (m)
      3. `yaw_deadband_deg` : 机身偏航执行器/陀螺死区阈值 (deg/s 等效角量纲)
    """
    recs = _ensure_records(records)
    if len(recs) < 4:
        return {
            "converged": False,
            "sample_count": len(recs),
            "dist_k": 0.24,
            "lqr_pos_lag_m": 0.035,
            "yaw_deadband_deg": 1.5,
        }

    # --- 1. 相机测距常数 dist_k: 1 / h_k = a - (1 / dist_k) * s_k ---
    s_travel: list[float] = []
    inv_h: list[float] = []
    cum_s = 0.0
    prev_xw: float | None = None
    for rec in recs:
        xw = rec.status.wheel_x_avg
        if prev_xw is not None:
            ds = xw - prev_xw
            if rec.target_lock is not None and rec.target_lock.valid:
                ds *= math.cos(math.radians(_clamp(rec.target_lock.bearing_body, -75.0, 75.0)))
            cum_s += ds
        prev_xw = xw

        valid_boxes = [
            d for d in rec.detections
            if float(d.get("score", 0.0)) >= 0.35 and float(d.get("h", 0.0)) >= 0.04
        ]
        if valid_boxes:
            valid_boxes.sort(key=lambda d: float(d.get("w", 0.1)) * float(d["h"]), reverse=True)
            h_val = float(valid_boxes[0]["h"])
            s_travel.append(cum_s)
            inv_h.append(1.0 / h_val)

    dist_k_converged = False
    dist_k_inliers = 0.0
    if len(s_travel) >= 5 and (max(s_travel) - min(s_travel)) >= 0.04:
        _, slope_b, dist_k_inliers = _theil_sen_huber_irls_1d(
            np.asarray(s_travel),
            np.asarray(inv_h),
            delta_scale=huber_delta_scale,
        )
        if slope_b < -0.15:
            dist_k = float(_clamp(-1.0 / slope_b, 0.05, 2.50))
            dist_k_converged = True
        else:
            dist_k = 0.24
    else:
        dist_k = 0.24

    # --- 2. 底层 LQR 巡航位置滞后 lqr_pos_lag_m ---
    lag_samples: list[float] = []
    for rec in recs:
        st = rec.status
        v_avg_pct = 0.5 * (st.v1 + st.v2)
        if abs(st.cmd_vx) >= 6.0 or abs(v_avg_pct) >= 3.0:
            sgn = 1.0 if (st.cmd_vx if abs(st.cmd_vx) >= 1.0 else v_avg_pct) >= 0.0 else -1.0
            xw = st.wheel_x_avg
            if abs(st.cmd_pos - xw) > 1e-6:
                lag_m = sgn * (st.cmd_pos - xw)
            else:
                lag_m = -sgn * (st.pos_e / 100.0)
            lag_samples.append(lag_m)

    if len(lag_samples) >= 3:
        lqr_pos_lag_m = float(_clamp(_huber_location_1d(np.asarray(lag_samples), delta_scale=huber_delta_scale), 0.0, 0.35))
        lag_converged = True
    else:
        lqr_pos_lag_m = 0.035
        lag_converged = False

    # --- 3. 航向死区 yaw_deadband_deg ---
    u_yaw_list: list[float] = []
    w_yaw_list: list[float] = []
    for k in range(1, len(recs)):
        r_prev, r_curr = recs[k - 1], recs[k]
        dt_s = (r_curr.t_ms - r_prev.t_ms) * 1e-3
        if not (0.004 <= dt_s <= 0.50):
            continue
        u_pct = r_prev.cmd.vyaw if abs(r_prev.cmd.vyaw) > 1e-3 else r_prev.status.cmd_vyaw
        u_dps = abs(float(u_pct)) * 0.90  # 协议 100% <-> 90 deg/s

        psi_prev = wrap180(r_prev.status.yaw - r_prev.status.head_q)
        psi_curr = wrap180(r_curr.status.yaw - r_curr.status.head_q)
        dpsi = abs(wrap180(psi_curr - psi_prev)) / dt_s
        if dpsi < 1e-5 and abs(r_curr.status.vel_e) > 0.1:
            dpsi = abs(r_curr.status.vel_e) * 0.60

        u_yaw_list.append(u_dps)
        w_yaw_list.append(dpsi)

    if len(u_yaw_list) >= 6 and max(u_yaw_list) >= 2.0:
        yaw_deadband_deg, yaw_gain = _huber_hinge_deadband(
            np.asarray(u_yaw_list),
            np.asarray(w_yaw_list),
            delta_scale=huber_delta_scale,
        )
        yaw_converged = True
    else:
        yaw_deadband_deg, yaw_gain = 1.5, 1.0
        yaw_converged = False

    return {
        "converged": bool(dist_k_converged and lag_converged and yaw_converged),
        "sample_count": len(recs),
        "dist_k": round(float(dist_k), 5),
        "lqr_pos_lag_m": round(float(lqr_pos_lag_m), 5),
        "yaw_deadband_deg": round(float(yaw_deadband_deg), 4),
        "yaw_gain": round(float(yaw_gain), 4),
        "dist_k_inlier_ratio": round(float(dist_k_inliers), 4),
    }


def export_gaussian_ply(
    splats_xyz_rgb: np.ndarray | list[dict[str, Any]] | list[list[float]],
    out_ply_path: str | Path,
) -> dict[str, Any]:
    """将端侧重建的 3D 彩色高斯椭球点云导出为标准 ASCII `.ply` 文件 (兼容 MeshLab / CloudCompare / Blender)。

    支持输入格式：
      - `np.ndarray` 形状 `(N, 6+)`：`[x, y, z, r, g, b, (可选 scale, opacity)]`，颜色自动识别 `0..1` 或 `0..255`；
      - `list[dict]`：包含 `x, y, z` 与 `r, g, b` (或 `rgb`) 及可选 `scale` / `radius`、`opacity` / `alpha`。
    """
    rows: list[tuple[float, float, float, int, int, int, float, float]] = []

    if isinstance(splats_xyz_rgb, list) and len(splats_xyz_rgb) > 0 and isinstance(splats_xyz_rgb[0], dict):
        for item in splats_xyz_rgb:
            x = float(item.get("x", 0.0))
            y = float(item.get("y", 0.0))
            z = float(item.get("z", 0.0))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if "rgb" in item and isinstance(item["rgb"], (list, tuple)) and len(item["rgb"]) >= 3:
                rc, gc, bc = float(item["rgb"][0]), float(item["rgb"][1]), float(item["rgb"][2])
            else:
                rc = float(item.get("r", item.get("red", 200.0)))
                gc = float(item.get("g", item.get("green", 200.0)))
                bc = float(item.get("b", item.get("blue", 200.0)))
            if max(rc, gc, bc) <= 1.001 and (isinstance(item.get("r"), float) or max(rc, gc, bc) < 1.0):
                rc, gc, bc = rc * 255.0, gc * 255.0, bc * 255.0
            ri = int(round(_clamp(rc, 0.0, 255.0)))
            gi = int(round(_clamp(gc, 0.0, 255.0)))
            bi = int(round(_clamp(bc, 0.0, 255.0)))
            sc = float(item.get("scale", item.get("radius", 0.015)))
            op = float(item.get("opacity", item.get("alpha", 0.90)))
            rows.append((x, y, z, ri, gi, bi, sc, op))
    else:
        arr = np.asarray(splats_xyz_rgb, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 6)
        if arr.ndim != 2 or arr.shape[1] < 6:
            raise ValueError(f"Expected splats_xyz_rgb with shape (N, >=6), got {arr.shape}")
        finite_mask = np.all(np.isfinite(arr[:, :6]), axis=1)
        arr = arr[finite_mask]
        rgb_raw = arr[:, 3:6]
        is_unit_color = bool(len(rgb_raw) > 0 and float(np.max(rgb_raw)) <= 1.001 and float(np.min(rgb_raw)) >= 0.0 and np.any((rgb_raw > 0.0) & (rgb_raw < 1.0)))
        rgb_255 = np.clip(np.round(rgb_raw * 255.0 if is_unit_color else rgb_raw), 0.0, 255.0).astype(np.int32)
        scales = arr[:, 6] if arr.shape[1] >= 7 else np.full(len(arr), 0.015, dtype=np.float64)
        opacities = arr[:, 7] if arr.shape[1] >= 8 else np.full(len(arr), 0.90, dtype=np.float64)
        for i in range(len(arr)):
            rows.append(
                (
                    float(arr[i, 0]),
                    float(arr[i, 1]),
                    float(arr[i, 2]),
                    int(rgb_255[i, 0]),
                    int(rgb_255[i, 1]),
                    int(rgb_255[i, 2]),
                    float(scales[i]),
                    float(opacities[i]),
                )
            )

    out_path = Path(out_ply_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "ply",
        "format ascii 1.0",
        "comment RIG-Hover End-Side 3D Colored Gaussian Splats",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float scale",
        "property float opacity",
        "end_header",
    ]
    body_lines = [
        f"{x:.5f} {y:.5f} {z:.5f} {r:d} {g:d} {b:d} {sc:.5f} {op:.4f}"
        for (x, y, z, r, g, b, sc, op) in rows
    ]
    content = "\n".join(header_lines + body_lines) + "\n"
    out_path.write_text(content, encoding="utf-8")

    if rows:
        xyz = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float64)
        b_min = [round(float(v), 4) for v in np.min(xyz, axis=0)]
        b_max = [round(float(v), 4) for v in np.max(xyz, axis=0)]
    else:
        b_min, b_max = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    return {
        "ply_path": str(out_path),
        "vertex_count": len(rows),
        "size_bytes": int(out_path.stat().st_size),
        "bounds_min": b_min,
        "bounds_max": b_max,
    }
