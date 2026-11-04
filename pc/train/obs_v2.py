# -*- coding: utf-8 -*-
"""obs_v2.py -- 场景 v2 的观测组装（106 → 116 维）。

设计约束，每一条都是为了不把已经跑通的东西弄坏：

1. **前 106 维原样复用** `obs.assemble()`，只在尾部追加。这样 v1 的 blob、
   旧的归一化统计量、旧的评估脚本都不会错位。
2. **只吃原始量**。录制时存的是原始信号（`feat_64`、`prop_*`、感知标量），
   组装放在这里做——所以观测布局改了不需要重采数据。
3. **v2a / v2b 两种变体**：

   * ``v2a``：114/115 维给出**已经锁定的目标**方位与距离（由一个
     :class:`TargetLatcher` 在敲击那一帧记住并持续提供）。语义是
     "去哪由外部告诉你，你只管怎么走"。
   * ``v2b``：114/115 **恒为 0**。网络必须自己在 ``knock_fired`` 那一帧
     把 ``theta_fist`` 和 ``dist_est`` 存进隐状态，之后靠记忆走。

   两种都要训。v2b 才是"让 CfC 承担更多"，但先有 v2a 作基线才知道
   v2b 的差距是训练问题还是表达问题。见 ``CFC-V2-DESIGN.md`` §2。

Run（自检）:  python obs_v2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import obs as obs_v1  # noqa: E402

__all__ = ["OBS_V2_DIM", "V2_NAMES", "V2_SLICE", "TargetLatcher", "assemble_v2"]

OBS_V1_DIM = 106
#: 追加的 10 维
V2_NAMES = [
    "theta_face", "face_scale", "face_valid",
    "theta_fist", "wrist_y_norm", "hand_valid", "fist_prob",
    "knock_fired", "knock_bearing", "knock_dist",
]
V2_SLICE = slice(OBS_V1_DIM, OBS_V1_DIM + len(V2_NAMES))
OBS_V2_DIM = OBS_V1_DIM + len(V2_NAMES)   # 116

#: 固件 blob 头的 input_dim 上限由 256 个 float 的统计量表决定
CFC_MAX_INPUT_DIM = 256


def _f(x, default=0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


class TargetLatcher:
    """在敲击那一帧锁定目标，并在之后持续提供它。

    ``v2a`` 用它把"该去哪"喂给网络；``v2b`` 不用它（那两维恒 0），
    由网络的隐状态自己记。**这两种做法必须能分别训练、分别评估**，
    否则没法回答"CfC 到底记住了没有"。
    """

    def __init__(self, hold_s: float = 30.0):
        self.hold_s = float(hold_s)
        self.reset()

    def reset(self):
        self.t_lock = -1e9
        self.bearing = 0.0
        self.dist_m = 0.0
        self.valid = 0

    def update(self, t_s: float, knock_fired: float, theta_fist: float,
               dist_est: float) -> tuple[float, float]:
        """喂一帧，返回 (bearing, dist)；没有有效目标时返回 (0, 0)。"""
        if _f(knock_fired) >= 0.5:
            self.t_lock = float(t_s)
            self.bearing = _f(theta_fist)
            self.dist_m = _f(dist_est)
            self.valid = 1
        if self.valid and (float(t_s) - self.t_lock) > self.hold_s:
            self.valid = 0
        if not self.valid:
            return 0.0, 0.0
        return self.bearing, self.dist_m


def assemble_v2(perc, prop, prev_act, t_s, dist_est, scene=None,
                latcher: TargetLatcher | None = None,
                variant: str = "b") -> np.ndarray:
    """组装 116 维观测。

    Args:
        perc / prop / prev_act / t_s / dist_est: 与 :func:`obs.assemble` 同义，
            前 106 维直接由它产出。
        scene: 场景 v2 的原始信号字典，键见 :data:`V2_NAMES` 的前 8 个
            （``knock_bearing`` / ``knock_dist`` 由 latcher 决定，不从这里取）。
        latcher: ``variant="a"`` 时必需，用来锁存目标。
        variant: ``"a"`` 或 ``"b"``，见模块文档。

    Returns:
        (116,) float32
    """
    if variant not in ("a", "b"):
        raise ValueError(f"variant 必须是 'a' 或 'b'，收到 {variant!r}")

    base = np.asarray(obs_v1.assemble(perc, prop, prev_act, t_s, dist_est),
                      np.float32)
    if base.shape[0] != OBS_V1_DIM:
        raise ValueError(f"v1 观测维度应为 {OBS_V1_DIM}，实际 {base.shape[0]}")

    s = scene or {}
    knock_fired = 1.0 if _f(s.get("knock_fired")) >= 0.5 else 0.0

    knock_bearing = knock_dist = 0.0
    if variant == "a":
        if latcher is None:
            raise ValueError("variant='a' 需要传入 TargetLatcher")
        knock_bearing, knock_dist = latcher.update(
            t_s, knock_fired, _f(s.get("theta_fist")), _f(dist_est))
        # 敲击那一帧本身也要给出去，否则网络学不到"事件 → 目标"的关联
        if knock_fired and knock_bearing == 0.0 and knock_dist == 0.0:
            knock_bearing = _f(s.get("theta_fist"))
            knock_dist = _f(dist_est)

    extra = np.array([
        _f(s.get("theta_face")),          # 106
        _f(s.get("face_scale")),          # 107
        1.0 if _f(s.get("face_valid")) >= 0.5 else 0.0,      # 108
        _f(s.get("theta_fist")),          # 109
        _f(s.get("wrist_y_norm")),        # 110
        1.0 if _f(s.get("hand_valid")) >= 0.5 else 0.0,      # 111
        _f(s.get("fist_prob")),           # 112
        knock_fired,                      # 113
        knock_bearing,                    # 114
        knock_dist,                       # 115
    ], np.float32)

    out = np.concatenate([base, extra]).astype(np.float32)
    assert out.shape[0] == OBS_V2_DIM
    return out


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_test() -> int:
    n_pass = n_fail = 0

    def chk(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"  PASS  {name}")
        else:
            n_fail += 1
            print(f"  FAIL  {name}   {detail}")

    print("== 维度 ==")
    chk("OBS_V2_DIM == 116", OBS_V2_DIM == 116, str(OBS_V2_DIM))
    chk("不超过固件上限", OBS_V2_DIM <= CFC_MAX_INPUT_DIM)
    chk("v1 名称表长度仍是 106", len(obs_v1.OBS_NAMES) == OBS_V1_DIM)

    perc = {"feat_64": np.zeros(64, np.float32), "theta_img": 3.0,
            "person_w": 0.2, "person_h": 0.5, "person_conf": 0.9,
            "seen": 1, "hand_up_L": 0, "hand_up_R": 1,
            "wave_sin": 0.0, "wave_cos": 0.0, "wave_amp": 0.0,
            "gesture_conf": 0.0}
    prop = {k: 0.0 for k in (
        "roll", "pitch", "pit_gyro", "dq", "yaw", "q_head", "target_head_pos",
        "wheel1_vel", "wheel2_vel", "wheel_x", "stable_pos", "stable_yaw",
        "robot_state", "vx", "vyaw", "control_mode", "feature_age_ms",
        "gaze_danger")}
    prop["robot_state"] = 1

    print("\n== v2b：目标两维必须是 0 ==")
    lat = TargetLatcher()
    scene = {"theta_face": 5.0, "face_scale": 0.12, "face_valid": 1,
             "theta_fist": -10.0, "wrist_y_norm": 0.8, "hand_valid": 1,
             "fist_prob": 0.9, "knock_fired": 1}
    o = assemble_v2(perc, prop, None, 0.0, 1.5, scene, lat, variant="b")
    chk("维度 116", o.shape[0] == 116, str(o.shape))
    chk("v2b 的 knock_bearing 恒 0", o[114] == 0.0, str(o[114]))
    chk("v2b 的 knock_dist 恒 0", o[115] == 0.0, str(o[115]))
    chk("knock_fired 那一帧为 1", o[113] == 1.0, str(o[113]))
    chk("theta_face 落在 106", abs(o[106] - 5.0) < 1e-6, str(o[106]))
    chk("theta_fist 落在 109", abs(o[109] + 10.0) < 1e-6, str(o[109]))

    print("\n== v2a：目标被锁存并持续提供 ==")
    lat2 = TargetLatcher()
    o1 = assemble_v2(perc, prop, None, 0.0, 1.5, scene, lat2, variant="a")
    chk("敲击帧给出方位", abs(o1[114] + 10.0) < 1e-6, str(o1[114]))
    chk("敲击帧给出距离", abs(o1[115] - 1.5) < 1e-6, str(o1[115]))
    # 0.5 秒后拳头消失、knock_fired 归零
    scene_after = dict(scene)
    scene_after.update({"theta_fist": 0.0, "hand_valid": 0,
                        "knock_fired": 0})
    o2 = assemble_v2(perc, prop, None, 0.5, 0.0, scene_after, lat2, variant="a")
    chk("拳头消失后目标仍在（方位）", abs(o2[114] + 10.0) < 1e-6, str(o2[114]))
    chk("拳头消失后目标仍在（距离）", abs(o2[115] - 1.5) < 1e-6, str(o2[115]))
    chk("v2b 在同样情形下是 0", assemble_v2(
        perc, prop, None, 0.5, 0.0, scene_after, None, variant="b")[114] == 0.0)

    print("\n== 锁存会过期 ==")
    lat3 = TargetLatcher(hold_s=1.0)
    assemble_v2(perc, prop, None, 0.0, 1.5, scene, lat3, variant="a")
    o3 = assemble_v2(perc, prop, None, 2.0, 0.0, scene_after, lat3, variant="a")
    chk("超过 hold_s 后目标清零", o3[114] == 0.0 and o3[115] == 0.0,
        f"{o3[114]}, {o3[115]}")

    print("\n== 前 106 维与 v1 完全一致 ==")
    o_a = assemble_v2(perc, prop, None, 0.0, 1.5, scene, TargetLatcher(),
                      variant="a")
    v1 = obs_v1.assemble(perc, prop, None, 0.0, 1.5)
    chk("前 106 维逐元素相同", np.allclose(o_a[:106], v1),
        f"max diff {np.abs(o_a[:106] - v1).max()}")

    print("\n" + "=" * 56)
    print(f"{n_pass} PASS / {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dim":
        print(OBS_V2_DIM)
        sys.exit(0)
    sys.exit(_self_test())
