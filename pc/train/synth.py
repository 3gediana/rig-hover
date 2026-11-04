# -*- coding: utf-8 -*-
"""synth.py — 无机器人时的合成轨迹生成器（观测 + 教师标签）。

目的**只是把管道跑通**：造出"人绕机器人走动 / 远近变化 / 挥手 / 走出视野"
这几段可解释的轨迹，用 ``TeacherController`` 打标签，喂给 ``train_cfc.py``。
不追求拟真，追求三件事：

1. 观测走 :func:`obs.assemble`，与固件 ``cfc_task.c`` 的下标逐位一致；
2. 标签来自**真教师控制器**（``pc/teacher/teacher.py`` + ``config.yaml``），
   不是我自己另写一套规则；
3. 闭环：t 帧的观测里 ``obs[100..103]`` 是 t-1 帧动作，机器人的头/机身/轮子
   按动作积分 —— 与真机上的因果顺序相同（否则 BPTT 学到的相位是错的）。

一次会话（session）的时间线（比例可按 ``--duration`` 缩放）::

    0.00-0.16  从视野外左侧走近，bbox 由小变大
    0.16-0.38  绕机器人画弧（ε_body 扫过 ±50°），距离在 0.8~1.8 m 摆动
    0.38-0.50  站定挥手 ~2.5 s → wave_score 涨过 4 → 教师进 ACK → REFRACTORY
    0.50-0.72  继续跟随（小幅漂移）
    0.72-0.88  走出视野 3 s（seen=0）→ 教师 degrades 到 IDLE
    0.88-1.00  回到视野中央，重新 TRACK

用法::

    python synth.py --sessions 8 --out-dir data            # 8 条会话写成 npz
    python synth.py --sessions 1 --print-table            # 打一张人能看的轨迹表
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import obs as obsmod

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流没有 reconfigure
        pass

HERE = Path(__file__).resolve().parent
TEACHER_DIR = HERE.parent / "teacher"
if str(TEACHER_DIR) not in sys.path:
    sys.path.insert(0, str(TEACHER_DIR))

from teacher import (MODE_ACK, MODE_NAMES, MODE_REFRACTORY, load_config,  # noqa: E402
                     TeacherController, TeacherObs)

F32 = np.float32

#: 教师参数表（§8.3 的 YAML，不要在这里抄数字）
TEACHER_CFG = TEACHER_DIR / "config.yaml"

#: 每 session 的默认帧率与时长
FPS_DEFAULT = 50.0
DURATION_DEFAULT = 40.0

#: 人"环视"的最大半视场：超过它就看不见（固件 fov_h 默认 60°）
FOV_H = obsmod.ARB_DEFAULTS["fov_h"]

#: feat_64 的合成投影：固定种子，跨 session 一致，才有东西可学。
#: 真机上这里是 YOLO+FeatureAdapter 的输出；合成数据只要求"与场景有单调关系"。
_FEAT_PROJ = np.random.default_rng(1234).normal(0.0, 1.0, size=(6, obsmod.FEAT_DIM)).astype(F32)
_FEAT_BIAS = np.zeros(obsmod.FEAT_DIM, dtype=F32)


@dataclass
class Session:
    """一条会话（一个 npz 的全部内容）。

    Attributes:
        obs: ``(T, 106)`` 观测，已按固件下标组装。
        head_deg / vyaw_dps / vx_mps: ``(T,)`` 教师动作（工程单位）。
        mode / emote: ``(T,)`` int8 标签。
        meta: 额外曲线（theta_img / eps_body / dist_est / wave_score / person_h），
            只用于打印与调参，不参与训练。
    """

    obs: np.ndarray
    head_deg: np.ndarray
    vyaw_dps: np.ndarray
    vx_mps: np.ndarray
    mode: np.ndarray
    emote: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return int(self.obs.shape[0])

    def save(self, path):
        np.savez_compressed(
            path,
            obs=self.obs.astype(F32),
            head_deg=self.head_deg.astype(F32),
            vyaw_dps=self.vyaw_dps.astype(F32),
            vx_mps=self.vx_mps.astype(F32),
            mode=self.mode.astype(np.int8),
            emote=self.emote.astype(np.int8),
            **{k: np.asarray(v, dtype=F32) for k, v in self.meta.items()},
        )
        return Path(path)


def wrap180(deg):
    """角度折到 [-180, 180)。"""
    return (float(deg) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------- #
# 人的运动脚本
# --------------------------------------------------------------------------- #
def _window(t, lo, hi):
    a = np.asarray(t, dtype=np.float64)
    return (a >= lo) & (a < hi)


def event_times(rng, duration):
    """把一次会话的关键时间点算出来（秒）。

    挥手/离开这两段用**绝对秒数**而不是时长比例：比例会让短会话里的挥手只持续
    1 秒，wave_score 永远涨不到教师的 4.0 阈值，于是整份数据里一个 ACK 都没有。
    真机上的挥手也是固定 2~3 秒，不随录制时长变。

    Returns:
        dict，键为各段时间点（秒）。
    """
    jitter = lambda a: float(rng.uniform(-a, a))  # noqa: E731
    w1 = 0.36 * float(duration) + jitter(0.3)
    w2 = w1 + 9.5 + jitter(0.4)          # REFRACTORY 结束后再来一次
    leave = max(0.78 * float(duration), w2 + 4.5)
    return {
        "approach_end": 0.16 * float(duration),
        "arc_end": w1,
        "wave1": (w1, w1 + 2.8),
        "wave2": (w2, w2 + 2.4),
        "hand1": (w1 - 0.20, w1 + 4.30),
        "hand2": (w2 - 0.20, w2 + 3.40),
        "leave": (leave, leave + 3.5),
        "return_end": leave + 7.0,
        "total": leave + 7.0,
    }


def person_script(rng, fps, duration):
    """生成人的运动脚本：方位/距离/可见/举手/挥手。

    Args:
        rng: ``np.random.Generator``。
        fps: 帧率。
        duration: 期望时长（秒）；实际时长会被拉长到能装下完整状态机
            （走近 → 画弧 → 挥手×2 → 走出视野 → 回来）。

    Returns:
        ``(n, dict)``：帧数 + 键 ``t``/``bearing``/``dist``/``visible``/``hand_up``/``waving``。
    """
    ev = event_times(rng, duration)
    n = int(math.ceil((max(float(duration), ev["total"]) + 1.0) * fps)) + 1
    t = np.arange(n, dtype=np.float64) / fps
    jitter = lambda a: float(rng.uniform(-a, a))  # noqa: E731

    b0 = -62.0 + jitter(8.0)          # 从左前方进场
    b_mid = -6.0 + jitter(6.0)
    b_arc = 44.0 + jitter(10.0)       # 绕到右前方
    b_leave = 78.0 + jitter(10.0)     # 走出视野
    lv0, lv1 = ev["leave"]

    bearing = np.interp(t, [
        0.0, ev["approach_end"], ev["arc_end"], ev["wave2"][1],
        lv0 - 1.0, lv0, lv1, lv1 + 1.5, ev["return_end"], t[-1],
    ], [b0, b_mid, b_arc, b_arc * 0.92, b_arc * 0.55, b_leave, b_leave,
        -24.0, 0.0, 0.0])
    # 画弧时叠一点正弦，让角速度不是常数
    moving = _window(t, ev["approach_end"], ev["arc_end"]) | _window(
        t, ev["wave2"][1], lv0 - 1.0)
    bearing = bearing + np.where(moving, 6.0 * np.sin(2.0 * math.pi * 0.8 * t), 0.0)

    dist = np.interp(t, [
        0.0, ev["approach_end"], 0.27 * duration, ev["arc_end"], ev["wave2"][0],
        ev["wave2"][1], lv0, lv1, ev["return_end"], t[-1],
    ], [2.6 + jitter(0.4), 1.05, 1.75, 1.10, 1.10, 1.45, 2.5, 3.1, 1.6, 1.35])
    dist = dist + 0.06 * np.sin(2.0 * math.pi * 0.9 * t)

    visible = ~_window(t, lv0, lv1)
    hand_up = _window(t, *ev["hand1"]) | _window(t, *ev["hand2"])
    waving = _window(t, *ev["wave1"]) | _window(t, *ev["wave2"])

    return n, {
        "t": t,
        "bearing": bearing,
        "dist": np.clip(dist, 0.45, 4.0),
        "visible": visible,
        "hand_up": hand_up,
        "waving": waving,
    }


# --------------------------------------------------------------------------- #
# 合成 feat_64
# --------------------------------------------------------------------------- #
def synth_feat(theta_img, dist, seen, hand_up, waving, conf, noise=None):
    """造 64 维"视觉特征"：场景码的固定随机投影 + tanh + 少量噪声。

    真机上是 YOLO 内部特征经 FeatureAdapter 压到 64 维（CONTRACT §3.1）。
    这里只要求：有界（tanh 后 ∈[-1,1]，与契约一致）、非恒定、与场景单调相关。
    常数纬度会让导出的 ``input_std`` 为 0，固件会直接拒绝模型 —— 所以噪声是
    **必须**的，不是装饰。

    Args:
        theta_img: 度。
        dist: 米。
        seen / hand_up / waving: 0/1。
        conf: 检测置信度。
        noise: ``(6,)`` 噪声，缺省全 0。

    Returns:
        ``(64,) float32``
    """
    scene = np.array([
        math.sin(float(theta_img) / 30.0),
        math.cos(float(theta_img) / 30.0),
        float(dist) / 2.0,
        float(seen),
        float(bool(hand_up) and bool(waving)),
        float(conf),
    ], dtype=F32)
    if noise is not None:
        scene = scene + np.asarray(noise, dtype=F32)
    return np.ascontiguousarray(np.tanh(scene @ _FEAT_PROJ + _FEAT_BIAS), dtype=F32)


# --------------------------------------------------------------------------- #
# 主生成
# --------------------------------------------------------------------------- #
def generate_session(seed=0, fps=FPS_DEFAULT, duration=DURATION_DEFAULT,
                     teacher_cfg_path=TEACHER_CFG, verbose=False):
    """生成一条会话。

    Args:
        seed: 随机种子（决定人的轨迹抖动与噪声）。
        fps: 帧率，默认 50（= 固件 cfc_hz 默认值）。
        duration: 时长（秒）。
        teacher_cfg_path: 教师 ``config.yaml`` 路径。
        verbose: 打印一条轨迹表。

    Returns:
        :class:`Session`
    """
    rng = np.random.default_rng(int(seed) + 10007)
    dt = 1.0 / float(fps)
    n, sc = person_script(rng, fps, duration)
    # A person who waves at a desktop robot is standing right in front of it.
    # The teacher only acknowledges once gesture confidence clears 0.75, which
    # corresponds to roughly 1.5 m, so a synthetic waving segment placed further
    # out yields a real gesture with no ACK label -- and a class-weighted CE
    # with an empty class. Pull the in-range windows closer rather than loosening
    # the controller threshold, which would change deployed behaviour.
    for _k in ("waving", "hand_up"):
        _m = sc[_k]
        _lim = 1.20 if _k == "waving" else 1.40
        sc["dist"] = np.where(_m, np.minimum(sc["dist"], _lim), sc["dist"])
    teacher = TeacherController(load_config(teacher_cfg_path))
    teacher.reset()
    dist_est = obsmod.DistEstimator()
    hand_right = (int(seed) % 2 == 0)   # 奇偶会话换手，避免 hand_up_L 恒为 0

    noise = {
        "feat": rng.normal(0.0, 0.05, size=(n, 6)).astype(F32),
        "pitch": rng.normal(0.0, 0.12, size=n),
        "roll": rng.normal(0.0, 0.10, size=n),
        "w": rng.normal(0.0, 0.02, size=n),
    }

    # ---- 机体状态（与固件同名全局变量对齐）----
    q_head = 0.0            # 头部实际角（度）
    target_head = 0.0       # 头部伺服目标（度）
    body_yaw = 0.0          # 机体 yaw（度）
    yaw_rate = 0.0          # 度/秒（= obs[96] vyaw）
    vx = 0.0                # 米/秒（= obs[95]）
    wheel_x = 0.0
    stable_pos = 0.0
    stable_yaw = 0.0
    prev_act = None
    t_last_seen = None

    # ---- 手势状态（复刻 gesture.py 的口径：过零累加 + exp(-dt/2) 衰减）----
    phase = 0.0
    amp = 0.0
    score = 0.0
    prev_sin = 0.0
    # Gesture-event state. The teacher now reacts to a discrete onset edge rather
    # than the wave_score level, so this mirrors the perception event machine:
    # one onset per contiguous gesture, debounced by 0.25 s (same constant the
    # perception module uses). Without it the synthetic set contains zero ACK
    # frames and the class-weighted CE has nothing to weight.
    g_state = 0            # 0 = IDLE, 1 = GESTURE, 2 = RETURN
    g_last_end = -1.0e9
    g_onset = 0
    g_gap = 999.0

    obs_buf = np.zeros((n, obsmod.OBS_DIM), dtype=F32)
    head_deg = np.zeros(n, dtype=F32)
    vyaw_dps = np.zeros(n, dtype=F32)
    vx_mps = np.zeros(n, dtype=F32)
    mode = np.zeros(n, dtype=np.int8)
    emote = np.zeros(n, dtype=np.int8)
    meta = {k: np.zeros(n, dtype=F32) for k in
            ("theta_img", "eps_body", "dist_est", "wave_score", "person_h", "seen",
             "wave_sin", "q_head", "target_head_pos", "yaw", "t_s")}

    for i in range(n):
        t_s = i * dt
        bearing = float(sc["bearing"][i])
        dist_true = float(sc["dist"][i])
        hand_up = bool(sc["hand_up"][i])
        waving = bool(sc["waving"][i])

        # 1) 几何 → 感知
        theta_img = wrap180(bearing - (body_yaw + q_head))
        in_fov = abs(theta_img) <= (FOV_H * 0.5)
        seen = 1 if (bool(sc["visible"][i]) and in_fov) else 0

        person_h = min(max(0.90 / dist_true, 0.03), 1.0) if seen else 0.0
        person_w = float(min(max(0.40 * person_h * (1.0 + 0.06 * noise["w"][i]), 0.0), 1.0))
        conf = float(np.clip(0.55 + 0.40 * math.exp(-dist_true / 2.5), 0.0, 1.0)) if seen else 0.0

        # 2) 手势基元
        target_amp = 0.62 if waving else (0.25 if hand_up else 0.0)
        amp += (target_amp - amp) * min(1.0, dt / 0.08)
        if waving:
            # 2.0 Hz：教师要求"2 个来回"= 4 次过零，且在 integrate_window_s=2 s 内
            # 达到 wave_score=4 —— 1.3 Hz 时 score 上限只有 5.2、2 s 才涨到 3.5，
            # 永远触发不了 ACK（合成数据里一个 ACK 都没有，加权 CE 就是空转）。
            phase += 2.0 * math.pi * 2.00 * dt
        elif hand_up:
            phase += 2.0 * math.pi * 0.20 * dt
        wave_sin = amp * math.sin(phase)
        wave_cos = amp * math.cos(phase)
        if hand_up and amp > 0.35:
            if (prev_sin <= 0.0 < wave_sin) or (prev_sin >= 0.0 > wave_sin):
                score += 1.0
        prev_sin = wave_sin
        score *= math.exp(-dt / 2.0)
        gesture_conf = float(conf * min(1.0, score / 4.0))

        # One onset edge per contiguous gesture burst.
        g_onset = 0
        if g_state == 0 and score >= 4.0 and amp > 0.35:
            if (t_s - g_last_end) >= 0.25:
                g_onset = 1
                g_state = 1
        elif g_state == 1 and amp < 0.30:
            g_state = 2
        elif g_state == 2 and amp < 0.10:
            g_state = 0
            g_last_end = t_s
        g_gap = float(t_s - g_last_end) if g_last_end > -1.0e8 else 999.0

        # 3) 64 维特征
        feat = synth_feat(theta_img, dist_true, seen, hand_up, waving, conf,
                          noise["feat"][i])
        perc = {
            "feat_64": feat,
            "theta_img": theta_img,
            "person_w": person_w,
            "person_h": person_h,
            "person_conf": conf,
            "seen": seen,
            "hand_up_L": 0 if hand_right else int(hand_up),
            "hand_up_R": int(hand_up) if hand_right else 0,
            "wave_sin": wave_sin,
            "wave_cos": wave_cos,
            # obs[74] 在固件里是**透传** PerceptionOut.wave_amp（cfc_task.c:372），
            # 不是由 sin/cos 重算；合成侧也必须给这个字段，否则该维恒为 0。
            "wave_amp": amp,
            "gesture_conf": gesture_conf,
        }

        # 4) 本体感知（注意 yaw 是**头部 IMU** 的绝对 yaw，所以 yaw - q_head = 机体 yaw）
        d_est = dist_est.step(seen, person_h)
        dq = (q_head - float(meta["q_head"][i - 1])) / dt if i > 0 else 0.0
        prop = {
            "roll": 1.2 * math.sin(0.9 * t_s) + float(noise["roll"][i]),
            "pitch": 1.5 * math.sin(0.7 * t_s + 0.4) + float(noise["pitch"][i]),
            "pit_gyro": 1.5 * 0.7 * math.cos(0.7 * t_s + 0.4) + 0.05 * float(noise["pitch"][i]),
            "dq": dq,
            "yaw": wrap180(body_yaw + q_head),
            "q_head": q_head,
            "target_head_pos": target_head,
            "wheel1_vel": vx + 0.01 * float(noise["w"][i]),
            "wheel2_vel": vx - 0.01 * float(noise["w"][i]),
            "wheel_x": wheel_x,
            "stable_pos": stable_pos,
            "stable_yaw": stable_yaw,
            "robot_state": 1,
            "vx": vx,
            "vyaw": yaw_rate,
            "control_mode": 2,   # 部署时 arbiter src = ARB_SRC_CFC
            "feature_age_ms": 0 if seen else (0 if t_last_seen is None
                                              else (t_s - t_last_seen) * 1000.0),
            "gaze_danger": 0,
        }

        obs_buf[i] = obsmod.assemble(perc, prop, prev_act, t_s, d_est)

        # 5) 教师标签（教师只吃 CONTRACT §4.1 的 8 个字段）
        act = teacher.step(TeacherObs(
            theta_img=theta_img, q_head=q_head, seen=seen, person_h=person_h,
            hand_up_L=perc["hand_up_L"], hand_up_R=perc["hand_up_R"],
            wave_score=float(score), t_now=t_s,
            gesture_onset=int(g_onset),
            gesture_id=(1 if (hand_up and score >= 1.0) else 0),
            gesture_conf=gesture_conf,
            gesture_gap_s=g_gap))
        head_deg[i] = act.head_des
        vyaw_dps[i] = act.vyaw_des
        vx_mps[i] = act.vx_des
        mode[i] = act.mode
        emote[i] = act.emote
        prev_act = act

        meta["theta_img"][i] = theta_img
        meta["eps_body"][i] = act.debug["eps_body"]
        meta["dist_est"][i] = d_est
        meta["wave_score"][i] = score
        meta["person_h"][i] = person_h
        meta["seen"][i] = float(seen)
        meta["wave_sin"][i] = wave_sin
        meta["q_head"][i] = q_head
        meta["target_head_pos"][i] = target_head
        meta["yaw"][i] = prop["yaw"]
        meta["t_s"][i] = t_s

        if seen:
            t_last_seen = t_s

        # 6) 机体动力学（下一帧用；固件里动作 → arbiter → 状态，有一拍延迟，这里同构）
        #    target_head 与 q_head 各自带一阶滞后 + 速率限制，obs[85]（head_err）
        #    才不会恒为 0 —— 恒 0 的维在导出时必须靠 std 下限兜，少一维少一分风险。
        head_rate = 150.0                      # 舵机角速度上限（度/秒，估计值）
        target_head += min(1.0, dt / 0.08) * (act.head_des - target_head)
        target_head = float(np.clip(target_head, -45.0, 45.0))
        step = head_rate * dt
        q_head += max(-step, min(step, min(1.0, dt / 0.10) * (target_head - q_head)))
        q_head = float(np.clip(q_head, -45.0, 45.0))

        yaw_rate += min(1.0, dt / 0.15) * (act.vyaw_des - yaw_rate)
        body_yaw = wrap180(body_yaw + yaw_rate * dt)

        vx += min(1.0, dt / 0.15) * (act.vx_des - vx)
        wheel_x += vx * dt

        stable_yaw += min(1.0, dt / 0.40) * (body_yaw - stable_yaw)
        stable_yaw = float(np.clip(stable_yaw, body_yaw - 35.0, body_yaw + 35.0))
        stable_pos += min(1.0, dt / 0.30) * (wheel_x - stable_pos)
        stable_pos = float(np.clip(stable_pos, wheel_x - 0.20, wheel_x + 0.20))

    sess = Session(obs=obs_buf, head_deg=head_deg, vyaw_dps=vyaw_dps, vx_mps=vx_mps,
                   mode=mode, emote=emote, meta=meta)
    if verbose:
        print_session_table(sess, stride=max(1, n // 24))
    return sess


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def print_session_table(sess, stride=80, limit=40):
    """打一条人能看懂的轨迹表（只打关键列）。"""
    names = {MODE_ACK: "ACK", MODE_REFRACTORY: "REFR"}
    print("    t(s)  theta  eps   dist  person_h  wave   seen  mode        "
          "head   vyaw    vx")
    rows = 0
    for i in range(0, len(sess), max(1, stride)):
        m = int(sess.mode[i])
        mn = names.get(m, MODE_NAMES[m])
        print("  %6.2f %6.1f %6.1f %5.2f %8.2f %6.2f %5d  %-8s %6.1f %6.1f %6.3f"
              % (sess.meta["t_s"][i], sess.meta["theta_img"][i], sess.meta["eps_body"][i],
                 sess.meta["dist_est"][i], sess.meta["person_h"][i],
                 sess.meta["wave_score"][i], int(sess.meta["seen"][i]), mn,
                 sess.head_deg[i], sess.vyaw_dps[i], sess.vx_mps[i]))
        rows += 1
        if rows >= limit:
            break


def summarize(sessions):
    """打印 mode 分布、动作范围、常数维统计。"""
    mode = np.concatenate([s.mode for s in sessions])
    head = np.concatenate([s.head_deg for s in sessions])
    vyaw = np.concatenate([s.vyaw_dps for s in sessions])
    vx = np.concatenate([s.vx_mps for s in sessions])
    obs_all = np.concatenate([s.obs for s in sessions])
    print("[synth] %d 条会话，%d 帧，%.1f s"
          % (len(sessions), len(mode), len(mode) / FPS_DEFAULT))
    tot = max(len(mode), 1)
    for m in range(4):
        c = int(np.sum(mode == m))
        print("    mode %-10s %6.2f%%  (%d 帧)"
              % (MODE_NAMES[m], 100.0 * c / tot, c))
    print("    head  [%7.2f, %7.2f] deg   vyaw [%7.2f, %7.2f] deg/s   vx [%6.3f, %6.3f] m/s"
          % (head.min(), head.max(), vyaw.min(), vyaw.max(), vx.min(), vx.max()))
    std = obs_all.std(axis=0)
    flat = np.where(std < 1e-6)[0]
    print("    常数维（std < 1e-6）: %s（导出时靠 std 下限兜住）"
          % (list(flat) if flat.size else "无"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="生成合成轨迹（106 维观测 + 教师动作标签），无机器人也能跑通训练/导出链。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sessions", type=int, default=8, help="生成几条会话")
    ap.add_argument("--out-dir", default=str(HERE / "data"), help="npz 输出目录")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT, help="帧率")
    ap.add_argument("--duration", type=float, default=DURATION_DEFAULT, help="每条会话时长（秒）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（第 k 条用 seed+k）")
    ap.add_argument("--teacher-config", default=str(TEACHER_CFG), help="教师 config.yaml")
    ap.add_argument("--print-table", action="store_true", help="打印轨迹表")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sessions = []
    for k in range(max(1, args.sessions)):
        s = generate_session(seed=args.seed + k, fps=args.fps, duration=args.duration,
                             teacher_cfg_path=args.teacher_config,
                             verbose=args.print_table and k == 0)
        if args.print_table:
            print("  --- session %d 完成，%d 帧 ---" % (k, len(s)))
        p = s.save(out_dir / ("synth_s%03d.npz" % k))
        print("  [%02d] %-8s %5d 帧  %.2f s  -> %s"
              % (k, "done", len(s), time.perf_counter() - t0, p.name))
        sessions.append(s)
    summarize(sessions)
    print("PASS" if sessions else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
