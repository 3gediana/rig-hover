"""教师控制器验收测试（CONTRACT §4.3 的 7 条 + 若干补充 + 事件化新增）。

手势判定已换成 **事件化的 onset 边沿**：T4/T5/T7 因此改写成用
``gesture_onset`` 驱动（原来的 ``wave_score`` 累积与 ``refractory_s`` 冷却已删除），
语义与判据见 ``teacher._update_state`` 的文档。B7~B12 是事件化专门补的 6 条。

跑法::

    D:\\smartcar\\.venv\\Scripts\\python.exe test_teacher.py

输出每条 PASS/FAIL，最后打印一条 200 步的合成闭环轨迹表（人走动 → 头身分离 →
大误差转身 → 挥手一个 onset → ACK → 丢失回中 → 回来重新跟踪），供肉眼判断动作
序列合不合理。退出码 0 = 全 PASS。
"""

from __future__ import annotations

import math
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 中文 Windows 控制台默认 GBK，°/− 之类会炸；保留本机编码但改成替换而不是抛异常
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

from recorder import Recorder  # noqa: E402
from teacher import (  # noqa: E402
    EMOTE_NOD,
    MODE_ACK,
    MODE_IDLE,
    MODE_NAMES,
    MODE_REFRACTORY,
    MODE_TRACK,
    TIER_LARGE,
    TIER_MID,
    TIER_NONE,
    TIER_SMALL,
    TeacherController,
    TeacherObs,
    load_config,
)

CFG = load_config(HERE / "config.yaml")
DT = 1.0 / 30.0

_RESULTS: list[tuple[str, bool, str]] = []


def check(cond: bool, msg: str) -> None:
    """一条断言，失败抛 AssertionError。"""
    if not cond:
        raise AssertionError(msg)


def obs(theta=0.0, q=0.0, seen=1, ph=0.65, hl=0, hr=0, ws=0.0, t=0.0,
        go=0, gid=0, gc=0.0, gap=None) -> TeacherObs:
    """构造观测的简写。

    Args:
        go/gid/gc/gap: 事件化手势字段（onset / id / conf / gap_s）。
        ws: 已废弃的 wave_score，只为验证「它不再影响判定」而保留的入参。
    """
    return TeacherObs(
        theta_img=theta, q_head=q, seen=seen, person_h=ph,
        hand_up_L=hl, hand_up_R=hr, wave_score=ws, t_now=t,
        gesture_onset=go, gesture_id=gid, gesture_conf=gc, gesture_gap_s=gap,
    )


def run(name: str, fn) -> None:
    """跑一条测试并记录结果。"""
    try:
        detail = fn() or ""
        _RESULTS.append((name, True, detail))
        print(f"[PASS] {name}" + (f"  — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - 测试聚合器，任何异常都算 FAIL
        _RESULTS.append((name, False, str(exc)))
        print(f"[FAIL] {name}  — {exc}")
        if not isinstance(exc, AssertionError):
            traceback.print_exc()


# ===========================================================================
# CONTRACT §4.3 的 7 条
# ===========================================================================
def test_1_zero_input() -> str:
    """theta_img=0, q_head=0 → vyaw_des==0 且 head_des==0。"""
    c = TeacherController(CFG)
    a = c.step(obs(theta=0.0, q=0.0, t=0.0))
    check(a.vyaw_des == 0.0, f"vyaw_des 应为 0，实际 {a.vyaw_des}")
    check(a.head_des == 0.0, f"head_des 应为 0，实际 {a.head_des}")
    for i in range(1, 200):
        a = c.step(obs(theta=0.0, q=0.0, t=i * DT))
        check(a.head_des == 0.0 and a.vyaw_des == 0.0, f"第 {i} 步漂了：{a}")
        check(a.mode == MODE_TRACK, f"第 {i} 步 mode 应为 TRACK，实际 {MODE_NAMES[a.mode]}")
    return "200 步全为精确 0，mode 稳定 TRACK"


def test_2_small_error_head_only() -> str:
    """theta_img=3, q_head=0（小误差）→ vyaw_des == 0，只动头。★核心"""
    c = TeacherController(CFG)
    hs, vs = [], []
    for i in range(90):
        a = c.step(obs(theta=3.0, q=0.0, t=i * DT))
        hs.append(a.head_des)
        vs.append(a.vyaw_des)
        check(a.debug["tier"] == TIER_SMALL, f"第 {i} 步档位应为 small，实际 {a.debug['tier_name']}")
    check(max(abs(v) for v in vs) == 0.0, f"vyaw_des 必须恒为 0，实际最大 {max(abs(v) for v in vs)}")
    check(hs[-1] > 2.5, f"头部应转到 ~3°，实际 {hs[-1]}")
    check(hs[-1] <= 3.0 + 1e-9, f"头部不应超调，实际 {hs[-1]}")
    return f"90 步 vyaw 恒 0，head_des 0→{hs[-1]:.3f}°"


def test_3_large_error_body_turns() -> str:
    """theta_img=30, q_head=0（大误差）→ vyaw_des != 0。"""
    c = TeacherController(CFG)
    vs, hs = [], []
    for i in range(90):
        a = c.step(obs(theta=30.0, q=0.0, t=i * DT))
        vs.append(a.vyaw_des)
        hs.append(a.head_des)
        check(a.debug["tier"] == TIER_LARGE, f"第 {i} 步档位应为 large，实际 {a.debug['tier_name']}")
    check(vs[-1] != 0.0, "vyaw_des 不应为 0")
    check(vs[-1] > 0.0, f"人偏右应右转（vyaw>0），实际 {vs[-1]}")
    check(abs(vs[-1]) <= CFG["tracking"]["vyaw_max_dps"] + 1e-9, f"vyaw 超上限 {vs[-1]}")
    check(hs[-1] > 0.0, f"大误差下头仍要补偿指向人，实际 {hs[-1]}")
    check(abs(hs[-1]) < 30.0, f"大误差下头应留 25° 给机身，实际 {hs[-1]}")
    return f"vyaw 0→{vs[-1]:.2f}°/s（上限 {CFG['tracking']['vyaw_max_dps']}），head_des→{hs[-1]:.2f}°"


def test_4_gesture_onset_triggers_ack() -> str:
    """手势 onset 边沿：一个脉冲 → TRACK 变 ACK（CONTRACT §4.3-4 的事件化版）。"""
    c = TeacherController(CFG)
    thresh = c.ack_conf_thresh
    for i in range(10):  # 先进 TRACK
        a = c.step(obs(t=i * DT))
    check(a.mode == MODE_TRACK, f"先应进 TRACK，实际 {MODE_NAMES[a.mode]}")

    # 对照组：置信度从 0.55 一路涨到 1.0，但没有 onset 边沿 → 一次都不该响应
    for k in range(1, 11):
        gc = min(0.99, 0.5 + 0.05 * k)
        a = c.step(obs(hl=1, gid=1, gc=gc, gap=5.0, t=(10 + k) * DT))
        check(a.mode == MODE_TRACK,
              f"没有 onset 边沿时 conf={gc:.2f} 也不应触发，实际 {MODE_NAMES[a.mode]}")

    # 实验组：单帧 onset=1（其余条件与上面最后一帧相同）
    t_go = 21 * DT
    a = c.step(obs(hl=1, go=1, gid=1, gc=0.9, gap=5.0, t=t_go))
    check(a.mode == MODE_ACK, f"onset 边沿应触发 ACK，实际 {MODE_NAMES[a.mode]}")
    check(a.emote == EMOTE_NOD, f"ACK 应有表情（点头），实际 emote={a.emote}")
    check(a.debug["hit_ack_onset"] and a.debug["hit_ack_conf"], "debug 里应记录两道判据命中")
    check(a.debug["ack_count"] == 1, f"ACK 次数应为 1，实际 {a.debug['ack_count']}")
    return f"第 1 帧 onset 触发（conf=0.9 ≥ {thresh:.2f}），之前 10 帧高 conf 无 onset 全不响应"


def test_5_no_cooldown_after_ack() -> str:
    """ACK 后没有冷却期：同段动作不重入，但新的 onset 立刻可响应。"""
    c = TeacherController(CFG)
    ack_hold = CFG["gesture"]["ack_hold_s"]
    t = 0.0
    for _ in range(10):
        c.step(obs(t=t)); t += DT
    a = c.step(obs(go=1, gid=1, gc=0.9, gap=5.0, t=t)); t += DT
    check(a.mode == MODE_ACK, f"第一个 onset 应进 ACK，实际 {MODE_NAMES[a.mode]}")
    ack_at = t - DT

    # 同一段动作继续（GESTURE 相位，没有新 onset）：一次都不许重入
    modes = []
    while t - ack_at < ack_hold + 1.2:
        a = c.step(obs(gid=1, gc=0.95, gap=5.0, t=t)); t += DT
        modes.append(a.mode)

    ack_len = sum(1 for m in modes if m == MODE_ACK)
    check(ack_len > 0, "触发后应仍处于 ACK")
    check(abs(ack_len * DT - ack_hold) <= 2 * DT, f"ACK 持续 {ack_len * DT:.2f}s，应约 {ack_hold}s")
    check(modes[:ack_len] == [MODE_ACK] * ack_len, "ACK 必须是连续一段")
    check(MODE_ACK not in modes[ack_len:],
          f"同一段动作（无新 onset）不应重入，实际重入 {modes[ack_len:].count(MODE_ACK)} 帧")
    check(MODE_REFRACTORY not in modes, "REFRACTORY 已删除，不应再出现")

    # 新的 onset：旧实现在这里会被 refractory_s(3 s) 挡住，新实现必须立刻响应
    a = c.step(obs(go=1, gid=1, gc=0.9, gap=0.6, t=t)); t += DT
    check(a.mode == MODE_ACK,
          f"冷却已删除：ACK 结束后 {t - ack_at - ack_hold:.2f}s 的 onset 应立刻响应，"
          f"实际 {MODE_NAMES[a.mode]}")
    check(a.debug["ack_count"] == 2, f"ACK 次数应为 2，实际 {a.debug['ack_count']}")
    return (f"持续动作 {len(modes) * DT:.1f}s：ACK {ack_len * DT:.2f}s 后 0 次重入、"
            f"0 帧 REFRACTORY；ACK 结束即接受新 onset（旧实现要等 3 s）")


def test_6_random_1000_finite() -> str:
    """连续 1000 步随机输入 → 输出全部有限、全部在范围内。"""
    rng = np.random.default_rng(20261102)
    c = TeacherController(CFG)
    head_max = CFG["tracking"]["head_max_deg"]
    vyaw_max = CFG["tracking"]["vyaw_max_dps"]
    vx_max = min(CFG["distance"]["vx_max"], CFG["distance"]["vx_max_desktop"])
    t = 0.0
    theta = q = 0.0
    for i in range(1000):
        if i % 20 == 0:  # 分段常量，让状态机真的动起来
            seen = int(rng.random() > 0.25)
            hl = int(rng.random() > 0.5)
            hr = int(rng.random() > 0.7)
            ph = float(rng.uniform(0.0, 1.0))
            go = int(rng.random() > 0.5)
            gid = int(rng.integers(0, 6))
            gc = float(rng.random())
            gap = float(rng.uniform(0.0, 3.0))
        theta = float(np.clip(theta + rng.normal(0, 6.0), -180.0, 180.0))
        q = float(np.clip(q + rng.normal(0, 4.0), -45.0, 45.0))
        ws = float(abs(rng.normal(2.0, 1.5)))
        a = c.step(obs(theta=theta, q=q, seen=seen, ph=ph, hl=hl, hr=hr, ws=ws,
                       t=t, go=go, gid=gid, gc=gc, gap=gap))
        t += float(rng.uniform(0.01, 0.1))
        for nm in ("head_des", "vyaw_des", "vx_des"):
            v = getattr(a, nm)
            check(isinstance(v, float) and math.isfinite(v), f"第 {i} 步 {nm} 非有限：{v}")
        check(abs(a.head_des) <= head_max + 1e-9, f"第 {i} 步 head_des 越界：{a.head_des}")
        check(abs(a.vyaw_des) <= vyaw_max + 1e-9, f"第 {i} 步 vyaw_des 越界：{a.vyaw_des}")
        check(abs(a.vx_des) <= vx_max + 1e-9, f"第 {i} 步 vx_des 越界：{a.vx_des}")
        check(a.mode in (MODE_IDLE, MODE_TRACK, MODE_ACK, MODE_REFRACTORY), f"第 {i} 步 mode 非法：{a.mode}")
        check(a.mode != MODE_REFRACTORY, f"第 {i} 步不应再进 REFRACTORY")
        check(0 <= a.emote <= 7, f"第 {i} 步 emote 非法：{a.emote}")
        for k in ("eps_body", "mode", "wave_score", "dist_est", "tier",
                  "hit_small", "hit_mid", "hit_large", "hit_deadband", "hit_head_limit",
                  "gesture_onset", "gesture_id", "gesture_conf", "gesture_gap_s",
                  "ack_count", "hit_ack_onset", "hit_ack_conf", "hit_ack_id",
                  "hit_ack_gap", "hit_ack_hand"):
            check(k in a.debug, f"第 {i} 步 debug 缺字段 {k}")
        check(a.debug["ack_count"] >= 0, f"第 {i} 步 ack_count 非法：{a.debug['ack_count']}")
    return ("1000 步：有限、head≤42、vyaw≤60、vx≤0.20、mode/emote 合法（无 REFRACTORY）、"
            "debug（含 gesture_* 与 hit_ack_*）字段齐全")


def test_7_trajectory_table() -> str:
    """打印 200 步合成闭环轨迹表。"""
    return _print_trajectory_table()


# ===========================================================================
# 补充测试（超出 CONTRACT §4.3，但都是方案里写死的行为）
# ===========================================================================
def test_8_body_cam_mode() -> str:
    """img_frame=1 时 eps_body = theta_img，q_head 不参与。"""
    cfg_b = {**CFG, "img_frame": 1}
    cb = TeacherController(cfg_b)
    ch = TeacherController(CFG)
    a_b = cb.step(obs(theta=3.0, q=30.0))
    a_h = ch.step(obs(theta=3.0, q=30.0))
    check(abs(a_b.debug["eps_body"] - 3.0) < 1e-12, f"body_cam eps 应为 3，实际 {a_b.debug['eps_body']}")
    check(abs(a_h.debug["eps_body"] - 33.0) < 1e-12, f"head_cam eps 应为 33，实际 {a_h.debug['eps_body']}")
    check(a_b.debug["tier"] == TIER_SMALL and a_h.debug["tier"] == TIER_LARGE,
          "同一输入在两种模式下应命中不同档位")
    return "body_cam eps=3（small）vs head_cam eps=33（large）"


def test_9_distance_channel() -> str:
    """§7.2b：死区、对准门、后退减半、上限。"""
    k = CFG["distance"]["k_dist"]
    dead = CFG["distance"]["dist_dead"]
    target = CFG["distance"]["dist_target"]
    coef = CFG["distance"]["dist_k_coef"]
    lim = min(CFG["distance"]["vx_max"], CFG["distance"]["vx_max_desktop"])

    # (a) 人太远 → 前进
    c = TeacherController(CFG)
    for i in range(90):
        a = c.step(obs(theta=0.0, ph=coef / 2.0, t=i * DT))
    check(a.vx_des > 0.0, f"人远应前进，实际 {a.vx_des}")
    check(abs(a.vx_des - min(k * (2.0 - target), lim)) < 0.02, f"前进量不对：{a.vx_des}")

    # (b) 死区内 → 0
    c = TeacherController(CFG)
    ph_dead = coef / (target + 0.5 * dead)
    for i in range(60):
        a = c.step(obs(theta=0.0, ph=ph_dead, t=i * DT))
    check(a.vx_des == 0.0, f"死区内 vx 应为精确 0，实际 {a.vx_des}")
    check(a.debug["hit_vx_dead"], "应命中距离死区")

    # (c) 没对准 → 强制 0（安全约束 1）
    c = TeacherController(CFG)
    for i in range(90):
        a = c.step(obs(theta=20.0, ph=coef / 2.0, t=i * DT))
    check(a.vx_des == 0.0, f"|eps|>15° 时 vx 必须为 0，实际 {a.vx_des}")
    check(a.debug["hit_vx_align"], "应命中对准门")

    # (d) 人太近 → 后退且减半（安全约束 3）
    c = TeacherController(CFG)
    for i in range(90):
        a = c.step(obs(theta=0.0, ph=coef / 0.7, t=i * DT))
    raw = k * (0.7 - target)
    expect = raw * CFG["distance"]["vx_backward_scale"]
    check(a.vx_des < 0.0, f"人太近应后退，实际 {a.vx_des}")
    check(abs(a.vx_des - expect) < 0.02, f"后退应减半：期望 {expect:.4f}，实际 {a.vx_des:.4f}")

    # (e) 距离一阶低通：阶跃不会一步到位
    c = TeacherController(CFG)
    c.step(obs(theta=0.0, ph=coef / 1.0, t=0.0))
    a1 = c.step(obs(theta=0.0, ph=coef / 2.0, t=DT))
    check(abs(a1.debug["dist_est"] - 2.0) > 0.2, "距离应被低通拖住，不应一步跳到 2.0")
    return (f"远->前进 / 对准门(|eps|>15 时 vx=0) / 太近减半(vx={a.vx_des:.4f}) / 低通拖尾 均符合；上限 {lim} m/s")


def test_10_head_limit_forces_body() -> str:
    """头顶到限位必须让机身转（优先于「小误差只动头」）。"""
    cfg = {
        **CFG,
        "tracking": {**CFG["tracking"], "head_max_deg": 10.0, "k_med": 0.1},
    }
    c = TeacherController(cfg)
    a = None
    for i in range(60):
        a = c.step(obs(theta=15.0, t=i * DT))
    check(a.debug["hit_head_limit"], "应命中限位保护")
    check(abs(a.vyaw_des) >= cfg["tracking"]["vyaw_floor_dps"] - 0.05,
          f"限位后 vyaw 应 >= {cfg['tracking']['vyaw_floor_dps']}，实际 {a.vyaw_des}")
    check(abs(a.head_des) <= 10.0 + 1e-9, "head_des 仍要钳在 head_max 内")

    # 默认参数下这条保护会不会误伤「小误差只动头」？扫一遍 eps
    c2 = TeacherController(CFG)
    tr = CFG["tracking"]
    first_hit = None
    changed = 0
    for k in range(0, 361):
        eps = k * 0.5
        c2.reset()
        d = c2.step(obs(theta=eps, t=0.0)).debug
        if not d["hit_head_limit"]:
            continue
        if first_hit is None:
            first_hit = eps
        a_eps = abs(eps)
        if a_eps < tr["body_engage_deg"]:
            base = 0.0
        elif a_eps < tr["head_full_deg"]:
            base = tr["k_med"] * a_eps
        else:
            base = min(tr["k_fast"] * a_eps, tr["vyaw_max_dps"])
        if base < tr["vyaw_floor_dps"]:
            changed += 1
    check(first_hit is not None, "默认参数下限位保护应可命中")
    check(first_hit >= tr["body_engage_deg"], "默认参数下限位保护不应在小误差区触发")
    return (f"head_max=10 时 vyaw={a.vyaw_des:.1f}°/s（地板 {tr['vyaw_floor_dps']}）；默认参数下首次命中在 "
            f"|eps|={first_hit:.1f}°（=head_max-{tr['head_limit_margin_deg']:.0f}°+{tr['head_full_deg']:.0f}°），"
            f"小误差区 0 次，且只有 {changed} 个扫描点会因此改变输出")


def test_11_lost_degrade() -> str:
    """视觉丢失三档：stale 冻结 / lost 回中 / 2s 后 IDLE。"""
    c = TeacherController(CFG)
    t = 0.0
    for _ in range(30):  # 先让头转开、机身转起来
        a = c.step(obs(theta=30.0, t=t)); t += DT
    head_before = a.head_des
    vyaw_before = a.vyaw_des
    check(head_before > 3.0, "前置条件：头应已转开")
    check(abs(vyaw_before) > 1.0, "前置条件：机身应在转")

    def lost_step(until: float) -> TeacherAction:
        """一直喂 seen=0 直到绝对时间 until。"""
        nonlocal t, a
        while t < until:
            a = c.step(obs(seen=0, t=t)); t += DT
        return a

    t0 = t                                     # 丢失起点
    a = c.step(obs(seen=0, t=t)); t += DT      # 刚丢
    check(a.debug["degrade"] == "ok", "刚丢 1 帧不应降级")
    check(a.debug["eps_held"], "没看见人的帧应沿用最后一次有效 eps")

    a = lost_step(t0 + 0.4)
    check(a.debug["degrade"] == "stale", f"0.4s 应处于 stale，实际 {a.debug['degrade']}")
    check(a.vx_des == 0.0, f"stale 时前进必须为 0，实际 {a.vx_des}")
    check(abs(a.head_des - head_before) < 1.0, "stale 时头部应冻结")

    a = lost_step(t0 + 0.95)
    check(a.debug["degrade"] == "stale", "0.95s 仍应是 stale")
    check(abs(a.vyaw_des) < 1.0, f"stale 内机身角速度必须衰减到 0，实际 {a.vyaw_des}")

    a = lost_step(t0 + 1.2)
    check(a.debug["degrade"] == "lost", f"1.2s 应处于 lost，实际 {a.debug['degrade']}")
    check(abs(a.head_des) < abs(head_before), "lost 时头部应开始回中")

    a = lost_step(t0 + 2.5)
    check(a.mode == MODE_IDLE, f"2.5s 应进 IDLE，实际 {MODE_NAMES[a.mode]}")
    check(abs(a.head_des) < 0.5, f"IDLE 时头应回中，实际 {a.head_des}")
    check(a.vyaw_des == 0.0 and a.vx_des == 0.0, "IDLE 时全部归零")

    a = c.step(obs(theta=0.0, t=t))
    check(a.mode == MODE_TRACK, f"人回来了应立刻回 TRACK，实际 {MODE_NAMES[a.mode]}")
    return f"stale 冻结(head={head_before:.1f}°) → lost 回中 → 2s IDLE，人回来立刻 TRACK"


def test_12_rate_limit() -> str:
    """速率限幅真的生效（不让动作一步跳变）。"""
    c = TeacherController(CFG)
    prev = c.step(obs(theta=0.0, t=0.0))
    max_dh = CFG["smoothing"]["head_rate_limit_dps"] * DT + 1e-6
    max_dv = CFG["smoothing"]["vyaw_rate_limit_dps2"] * DT + 1e-6
    for i in range(1, 60):
        a = c.step(obs(theta=120.0, t=i * DT))
        check(abs(a.head_des - prev.head_des) <= max_dh,
              f"head 单步变化 {abs(a.head_des - prev.head_des):.3f} 超过 {max_dh:.3f}")
        check(abs(a.vyaw_des - prev.vyaw_des) <= max_dv,
              f"vyaw 单步变化 {abs(a.vyaw_des - prev.vyaw_des):.3f} 超过 {max_dv:.3f}")
        prev = a
    return f"head ≤{max_dh:.2f}°/步，vyaw ≤{max_dv:.2f}(°/s)/步"


def test_13_recorder() -> str:
    """录制器：npz + jsonl + 视频帧引用 + 按 session 划分无泄露。"""
    out = HERE / "_test_out"
    if out.exists():
        shutil.rmtree(out)
    try:
        c = TeacherController(CFG)
        npz_paths = []
        for sid in ("s01", "s02", "s03"):
            rec = Recorder(out, session_id=sid, video_every=3, meta={"fps": 30})
            t = 0.0
            for i in range(20):
                a = c.step(obs(theta=5.0 * math.sin(i * 0.3), ph=0.6, t=t))
                dbg = dict(a.debug)
                if i % 2 == 0:            # 只在一半帧出现的键，验证对齐不串行
                    dbg["only_even"] = float(i)
                vpath = rec.add(t, {"theta_img": 5.0 * math.sin(i * 0.3), "person_h": 0.6,
                                    "feat_raw": np.zeros(920, np.float32)},
                                a.as_dict(), dbg)
                if i % 3 == 0:
                    check(vpath is not None, f"第 {i} 帧应返回视频路径")
                t += DT
            info = rec.close()
            check(Path(info["npz"]).exists(), "npz 未生成")
            check(Path(info["jsonl"]).exists(), "jsonl 未生成")
            npz_paths.append(info["npz"])

        data = Recorder.load_npz(npz_paths[0])
        check(data["obs_feat_raw"].shape == (20, 920), f"特征维度错：{data['obs_feat_raw'].shape}")
        check(data["obs_feat_raw"].dtype == np.float32, "数值应为 float32")
        check(float(np.nanmax(np.abs(data["act_head_des"]))) <= 42.0, "录到的 head_des 越界")
        check(data["video_frame"][1] == "", "非视频帧应为空串")
        check(data["video_frame"][0].endswith("000000.jpg"), "视频帧路径格式不对")
        check(data["dbg_only_even"][0] == 0.0, "半程字段的第 0 帧应为 0")
        check(np.isnan(data["dbg_only_even"][1]), "半程字段缺失的帧应补 NaN，不能错位")
        check(data["dbg_only_even"][2] == 2.0, "半程字段的第 2 帧应为 2")
        check(int(data["n_frames"][0]) == 20, "n_frames 不对")
        n_lines = sum(1 for _ in Path(npz_paths[0]).with_suffix(".jsonl").open(encoding="utf-8"))
        check(n_lines == 20, f"jsonl 行数应为 20，实际 {n_lines}")

        # 正确的划分：按 session
        splits = Recorder.split_by_session({"s01": 20, "s02": 20, "s03": 20}, val_frac=0.34, test_frac=0.0, seed=0)
        check(len(splits["train"]) == 40 and len(splits["val"]) == 20, "划分大小不对")
        owner = {}
        for name in ("train", "val"):
            for i in splits[name]:
                sid = "s01" if i < 20 else ("s02" if i < 40 else "s03")
                check(owner.setdefault(sid, name) == name, f"session {sid} 跨集合泄露")

        # 反例：按帧随机划分会漏
        idx = np.arange(60)
        np.random.default_rng(0).shuffle(idx)
        train, val = set(idx[:40].tolist()), set(idx[40:].tolist())
        leak = sum(1 for j in val if (j - 1) in train or (j + 1) in train)
        check(leak > 10, f"按帧随机划分应产生大量相邻帧泄露，实际 {leak}/20")
        return f"3 session×20 帧；npz{data['obs_feat_raw'].shape} float32；按帧随机划分 {leak}/20 帧相邻泄露，按 session 0 泄露"
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ===========================================================================
# 事件化手势：onset 边沿语义的新增验收
# ===========================================================================
def _warm(c: TeacherController, n: int = 10, t0: float = 0.0) -> float:
    """先喂 n 帧「有人、无手势」，返回下一个时间戳。"""
    t = t0
    for _ in range(n):
        c.step(obs(t=t)); t += DT
    return t


def test_14_two_onsets_two_acks() -> str:
    """连续两段挥手（两个 onset）→ 两次 ACK；同窗口内的第二段被吸收。"""
    ack_hold = CFG["gesture"]["ack_hold_s"]
    c = TeacherController(CFG)
    # 两个 onset 间隔 2.0 s（> ack_hold），对应「挥一次、停下、再挥一次」
    at = (int(round(0.30 / DT)), int(round(2.30 / DT)))
    entries, prev = 0, MODE_IDLE
    last = None
    for i in range(int(6.0 / DT)):
        go = 1 if i in at else 0
        gap = 0.6 if (go and i == at[1]) else 5.0
        last = c.step(obs(go=go, gid=1, gc=0.9, gap=gap, t=i * DT))
        if last.mode == MODE_ACK and prev != MODE_ACK:
            entries += 1
        prev = last.mode
    check(entries == 2, f"两个 onset 应产生两次 ACK，实际 {entries} 次")
    check(last.debug["ack_count"] == 2, f"ack_count 应为 2，实际 {last.debug['ack_count']}")

    # 设计边界：ACK 是固定 ack_hold_s 的反馈脉冲，落在窗口内的新 onset 被吸收。
    # 要两次响应，两段动作的间隔必须 > ack_hold_s（上面那组）。
    c2 = TeacherController(CFG)
    at2 = (int(round(0.30 / DT)), int(round(0.30 / DT)) + int(round(0.5 / DT)))
    entries2, prev = 0, MODE_IDLE
    last2 = None
    for i in range(int(4.0 / DT)):
        go = 1 if i in at2 else 0
        last2 = c2.step(obs(go=go, gid=1, gc=0.9, gap=5.0, t=i * DT))
        if last2.mode == MODE_ACK and prev != MODE_ACK:
            entries2 += 1
        prev = last2.mode
    check(entries2 == 1, f"间隔 {ack_hold:.1f}s 内的第二段动作应被 ACK 吸收，实际 {entries2} 次")
    return (f"间隔 2.0 s 的两个 onset → {entries} 次 ACK；间隔 0.5 s（< ack_hold {ack_hold}s）"
            f"→ 被吸收成 {entries2} 次（设计如此，已在文档写明）")


def test_15_gap_debounce() -> str:
    """gesture_gap_s < ack_min_gap_s 时抑制响应；够了就放行（去抖，不是冷却）。"""
    c = TeacherController(CFG)
    gap_min = c.ack_min_gap_s
    t = _warm(c)

    a = c.step(obs(go=1, gid=1, gc=0.9, gap=gap_min * 0.4, t=t)); t += DT
    check(a.mode == MODE_TRACK, f"gap={gap_min * 0.4:.2f} < {gap_min} 应抑制，实际 {MODE_NAMES[a.mode]}")
    check(a.debug["hit_ack_gap"], "debug 应标记 gap 门命中")
    check(a.debug["hit_ack_onset"] and a.debug["hit_ack_conf"] and a.debug["hit_ack_id"],
          "其余判据应仍为通过，只有 gap 把这次响应挡掉")

    a = c.step(obs(go=1, gid=1, gc=0.9, gap=gap_min + 0.05, t=t)); t += DT
    check(a.mode == MODE_ACK, f"gap={gap_min + 0.05:.2f} ≥ {gap_min} 应放行，实际 {MODE_NAMES[a.mode]}")

    # gap 未知（None）不能静默失效：不抑制
    c2 = TeacherController(CFG)
    t2 = _warm(c2)
    a = c2.step(obs(go=1, gid=1, gc=0.9, gap=None, t=t2))
    check(a.mode == MODE_ACK, "gap_s=None（生产者没给）时不应抑制")
    return (f"gap {gap_min * 0.4:.2f}s 抑制、{gap_min + 0.05:.2f}s 放行、None 不抑制"
            f"（min_gap={gap_min}s 是去抖门，与旧 refractory 冷却语义不同）")


def test_16_gesture_id_gate() -> str:
    """只有 ack_gesture_ids 里的手势才响应；pointing/fist 默认不响应。"""
    c = TeacherController(CFG)
    accept = sorted(c.ack_gesture_ids)
    t = _warm(c)
    for gid in (0, 4, 5):        # none / pointing / fist
        t = _warm(c, 1, t)
        a = c.step(obs(go=1, gid=gid, gc=0.99, gap=5.0, t=t)); t += DT
        check(a.mode == MODE_TRACK, f"gesture_id={gid} 不在 {accept} 里，不应响应")
        check(a.debug["hit_ack_id"] is False, f"gesture_id={gid} 应判为不在响应集合")
    for gid in accept:
        c2 = TeacherController(CFG)
        t2 = _warm(c2)
        a = c2.step(obs(go=1, gid=gid, gc=0.99, gap=5.0, t=t2))
        check(a.mode == MODE_ACK, f"gesture_id={gid} 在响应集合里，应 ACK")
    return f"响应集合 {accept}；0/4/5 不响应，集合内每个 id 单独测试均触发"


def test_17_onset_ignored_when_not_ok() -> str:
    """视觉失效（stale/lost）时 onset 不响应——先要看见人再谈手势。"""
    c = TeacherController(CFG)
    t = _warm(c)
    for _ in range(int(0.5 / DT)):      # 人丢了 0.5 s → stale
        a = c.step(obs(seen=0, t=t)); t += DT
    check(a.debug["degrade"] == "stale", f"0.5 s 应 stale，实际 {a.debug['degrade']}")
    a = c.step(obs(seen=0, go=1, gid=1, gc=0.99, gap=5.0, t=t)); t += DT
    check(a.mode != MODE_ACK, f"stale 时 onset 不应触发，实际 {MODE_NAMES[a.mode]}")

    # 人回来 → 立刻可响应
    for _ in range(3):
        a = c.step(obs(seen=1, t=t)); t += DT
    a = c.step(obs(go=1, gid=1, gc=0.99, gap=5.0, t=t))
    check(a.mode == MODE_ACK, f"人回来后 onset 应能触发，实际 {MODE_NAMES[a.mode]}")
    return "stale 期间 onset 被忽略、人回来后立刻恢复响应"


def test_18_first_frame_onset() -> str:
    """第一帧就看见人并挥手：IDLE→TRACK 与 ACK 在同一帧完成，不丢事件。"""
    c = TeacherController(CFG)
    a = c.step(obs(go=1, gid=1, gc=0.9, gap=5.0, t=0.0))
    check(a.mode == MODE_ACK, f"首帧 onset 应直接进 ACK，实际 {MODE_NAMES[a.mode]}")
    check(a.debug["ack_count"] == 1, "首帧 ACK 应计一次")

    # 反之：第一帧没人、第二帧才有人挥手，也应响应
    c2 = TeacherController(CFG)
    c2.step(obs(seen=0, t=0.0))
    a = c2.step(obs(seen=1, go=1, gid=1, gc=0.9, gap=5.0, t=DT))
    check(a.mode == MODE_ACK, f"出帧后首帧 onset 应进 ACK，实际 {MODE_NAMES[a.mode]}")
    return "首帧 onset 不丢：IDLE→TRACK→ACK 在同一 step 内完成"


def test_19_garbage_gesture_stream() -> str:
    """1000 步对抗输入（非法 id / 越界 conf / NaN gap / 每帧 onset）不出非法输出。"""
    rng = np.random.default_rng(20261103)
    c = TeacherController(CFG)
    head_max = CFG["tracking"]["head_max_deg"]
    vyaw_max = CFG["tracking"]["vyaw_max_dps"]
    vx_max = min(CFG["distance"]["vx_max"], CFG["distance"]["vx_max_desktop"])
    gaps = (math.nan, -1.0, 0.0, 0.1, 0.25, 0.3, 1e9)
    t = 0.0
    for i in range(1000):
        a = c.step(obs(
            theta=float(rng.normal(0.0, 40.0)), q=float(rng.normal(0.0, 20.0)),
            seen=int(rng.random() > 0.2), ph=float(rng.random()),
            go=int(rng.random() > 0.05),                 # 几乎每帧都是边沿
            gid=int(rng.integers(-3, 12)),               # 含 0/6..11 非法 id
            gc=float(rng.normal(0.5, 0.8)),              # 可能越界/为负
            gap=float(rng.choice(gaps)),
            t=t))
        t += float(rng.uniform(0.001, 0.2))
        for nm in ("head_des", "vyaw_des", "vx_des"):
            v = getattr(a, nm)
            check(isinstance(v, float) and math.isfinite(v), f"第 {i} 步 {nm} 非有限：{v}")
        check(abs(a.head_des) <= head_max + 1e-9, f"第 {i} 步 head_des 越界：{a.head_des}")
        check(abs(a.vyaw_des) <= vyaw_max + 1e-9, f"第 {i} 步 vyaw_des 越界：{a.vyaw_des}")
        check(abs(a.vx_des) <= vx_max + 1e-9, f"第 {i} 步 vx_des 越界：{a.vx_des}")
        check(a.mode in (MODE_IDLE, MODE_TRACK, MODE_ACK), f"第 {i} 步 mode 非法：{a.mode}")
        check(0 <= a.emote <= 7, f"第 {i} 步 emote 非法：{a.emote}")
        check(isinstance(a.debug["ack_count"], int) and a.debug["ack_count"] >= 0,
              f"第 {i} 步 ack_count 非法：{a.debug['ack_count']}")
        check(isinstance(a.debug["hit_ack_onset"], bool), "hit_ack_* 应为 bool")
    check(c.mode in (MODE_IDLE, MODE_TRACK, MODE_ACK), "末态 mode 非法")
    return ("1000 步对抗输入（gid∈[-3,12]、conf 可越界、gap 含 NaN/负/1e9、95% 帧 onset）："
            "输出有限且在范围内，mode 只出现 0/1/2，ack_count 单调非负")


# ===========================================================================
# 200 步闭环轨迹表
# ===========================================================================
def _scenario(i: int) -> tuple[float, float, bool, int, int, float, float | None, int]:
    """第 i 步的剧本。

    Returns:
        ``(人的绝对方位, 人的径向速度 m/s, 是否可见, gesture_onset, gesture_id,
        gesture_conf, gesture_gap_s, hand_up)``。径向速度为正 = 人往远处走；
        真实距离由 vx 积分和人走动共同决定（见 _print_trajectory_table）。
        手势是**边沿**：整段动作里只有 i==110 那一帧 onset=1。
    """
    if i < 45:                                  # 1) 静立，只是有点远
        return 0.0, 0.0, True, 0, 0, 0.0, 5.0, 0
    if i < 80:                                  # 2) 慢速横移到 4.5°（小误差）
        return 4.5 * (i - 45) / 35.0, 0.0, True, 0, 0, 0.0, 5.0, 0
    if i < 110:                                 # 3) 继续横移到 35°（中→大误差）
        return 4.5 + 30.5 * (i - 80) / 30.0, 0.0, True, 0, 0, 0.0, 5.0, 0
    if i < 130:                                 # 4) 站住挥手：边沿只在 i==110
        onset = 1 if i == 110 else 0
        return 35.0, 0.0, True, onset, 1, 0.92, 5.0, 1
    if i < 190:                                 # 5) 走出视野 2.0s（边走远）
        return 35.0, 0.30, False, 0, 0, 0.0, (i - 129) * DT, 0
    return 22.0, 0.30, True, 0, 0, 0.0, 5.0, 0   # 6) 从另一侧回来（还在走远）


def _print_trajectory_table() -> str:
    """跑一条闭环仿真并打印 200 步表格。"""
    c = TeacherController(CFG)
    psi_body = 0.0     # 机身朝向（真实，度）
    q_head = 0.0       # 头部实际角（舵机跟随 head_des）
    vyaw_act = 0.0     # 机身实际角速度
    dist_true = 1.35
    coef = CFG["distance"]["dist_k_coef"]
    dt = DT
    ack_frames = 0
    ref_frames = 0
    modes: list[int] = []
    vyaws: list[float] = []
    heads: list[float] = []

    hdr = (f"{'step':>4} {'t(s)':>5} {'B_abs':>6} {'psi_b':>6} {'th_img':>6} {'q_head':>6} "
           f"{'eps_body':>8} {'hd_raw':>7} {'head_des':>8} {'vyaw':>6} {'vx':>6}  {'mode':<11} {'tier':<5} deg")
    print("\n" + "=" * len(hdr))
    print("200 步合成闭环轨迹（head_cam；B_abs=人的绝对方位，psi_b=机身朝向，"
          "eps_body=B_abs-psi_b 与 q_head 无关）")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    banners = {
        0: "① 人静立，真实距离 1.35m（>1.0）→ 只该慢慢向前",
        45: "② 人慢速横移 0->4.5°，|eps|<5° → 只动头，vyaw 必须为 0",
        80: "③ 横移到 35° → 中/大误差，机身开始转，头反向补偿",
        110: "④ 站住挥手：一次 onset 边沿 → ACK（表情=点头），没有冷却期",
        130: "⑤ 人走出视野 2s → stale(冻结) → lost(回中) → IDLE",
        190: "⑥ 从另一侧回来 → 重新 TRACK",
    }
    for i in range(200):
        if i in banners:
            print(f"{'-' * 4} {banners[i]}")
        B, radial_v, seen, go, gid, gc, gap, hand = _scenario(i)
        person_h = float(np.clip(coef / dist_true, 0.02, 1.0)) if seen else 0.0

        theta_img = B - (psi_body + q_head)
        o = obs(theta=theta_img, q=q_head, seen=int(seen), ph=person_h,
                hl=hand, t=i * dt, go=go, gid=gid, gc=gc, gap=gap)
        a = c.step(o)
        modes.append(a.mode)

        if i % 5 == 0 or i in banners or c.mode == MODE_ACK:
            print(f"{i:4d} {i * dt:5.2f} {B:6.1f} {psi_body:6.1f} {theta_img:6.1f} {q_head:6.1f} "
                  f"{a.debug['eps_body']:8.2f} {a.debug['head_cmd_raw']:7.2f} {a.head_des:8.2f} "
                  f"{a.vyaw_des:6.2f} {a.vx_des:6.3f}  {MODE_NAMES[a.mode]:<11} "
                  f"{a.debug['tier_name']:<5} {a.debug['degrade']}")

        # 执行机构（真实世界）
        q_head += float(np.clip(a.head_des - q_head, -180.0 * dt, 180.0 * dt))
        vyaw_act += 0.3 * (a.vyaw_des - vyaw_act)
        psi_body += vyaw_act * dt
        dist_true = float(np.clip(dist_true + (radial_v - a.vx_des) * dt, 0.3, 4.0))
        vyaws.append(a.vyaw_des)
        heads.append(a.head_des)
        ack_frames += int(a.mode == MODE_ACK)
        ref_frames += int(a.mode == MODE_REFRACTORY)

    print("-" * len(hdr))
    small_v = max(abs(v) for v in vyaws[45:80])
    small_h = max(abs(h) for h in heads[45:80])
    print(f"小误差段 45-79：max|vyaw_des|={small_v:.2f}°/s（必须为 0），max|head_des|={small_h:.2f}°（应 >1，证明确实只动头）")
    print(f"ACK {ack_frames} 帧({ack_frames * dt:.2f}s)；REFRACTORY {ref_frames} 帧（应为 0）；"
          f"累计 ACK {a.debug['ack_count']} 次；末态 |eps_body|={abs(B - psi_body):.2f}°，真实距离={dist_true:.2f}m")
    check(ack_frames > 0, "轨迹里应出现 ACK")
    check(ref_frames == 0, f"REFRACTORY 已删除，实际出现 {ref_frames} 帧")
    check(a.debug["ack_count"] == 1, f"一段挥手只应产生 1 次 ACK，实际 {a.debug['ack_count']}")
    check(MODE_IDLE in modes[130:190], "人走光 2s 内应出现 IDLE")
    check(small_v == 0.0, f"小误差段机身必须完全不动，实际 max|vyaw|={small_v}")
    check(small_h > 1.0, "小误差段头部必须跟随，实际几乎不动")
    return (f"小误差段 vyaw 恒 0 / head 最大 {small_h:.1f}°；ACK {ack_frames * dt:.2f}s 共 1 次；"
            f"REFRACTORY 0 帧；IDLE 出现")


# ===========================================================================
def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"config: {HERE / 'config.yaml'}")
    print(f"img_frame={CFG['img_frame']}  "
          f"head_max={CFG['tracking']['head_max_deg']}°  vyaw_max={CFG['tracking']['vyaw_max_dps']}°/s  "
          f"vx_max={min(CFG['distance']['vx_max'], CFG['distance']['vx_max_desktop'])} m/s\n")

    tests = [
        ("T1 零误差 → vyaw=0 且 head=0        (CONTRACT §4.3-1)", test_1_zero_input),
        ("T2 小误差 3° → vyaw==0 只动头 ★核心 (CONTRACT §4.3-2)", test_2_small_error_head_only),
        ("T3 大误差 30° → vyaw!=0             (CONTRACT §4.3-3)", test_3_large_error_body_turns),
        ("T4 手势 onset 脉冲 → TRACK→ACK     (CONTRACT §4.3-4)", test_4_gesture_onset_triggers_ack),
        ("T5 ACK 后无冷却、无重入            (CONTRACT §4.3-5)", test_5_no_cooldown_after_ack),
        ("T6 1000 步随机 → 有限且在范围内      (CONTRACT §4.3-6)", test_6_random_1000_finite),
        ("T7 200 步轨迹表                     (CONTRACT §4.3-7)", test_7_trajectory_table),
        ("B1 body_cam 模式 eps=theta_img     (§7.1b)", test_8_body_cam_mode),
        ("B2 距离通道四约束                  (§7.2b)", test_9_distance_channel),
        ("B3 头顶限位强制机身转              (§7.2)", test_10_head_limit_forces_body),
        ("B4 视觉丢失三档                    (§5.4/§8.3)", test_11_lost_degrade),
        ("B5 低通+速率限幅                   (§8.3)", test_12_rate_limit),
        ("B6 录制器 + 按 session 划分        (§9.3)", test_13_recorder),
        ("B7 两个 onset → 两次 ACK            (事件化)", test_14_two_onsets_two_acks),
        ("B8 gap 去抖门（不是冷却）           (事件化)", test_15_gap_debounce),
        ("B9 手势 id 响应集合                 (事件化)", test_16_gesture_id_gate),
        ("B10 视觉失效时 onset 不响应         (事件化)", test_17_onset_ignored_when_not_ok),
        ("B11 首帧 onset 不丢                 (事件化)", test_18_first_frame_onset),
        ("B12 1000 步对抗手势输入             (事件化)", test_19_garbage_gesture_stream),
    ]
    for name, fn in tests:
        run(name, fn)

    ok = sum(1 for _, p, _ in _RESULTS if p)
    bad = len(_RESULTS) - ok
    print("\n" + "=" * 78)
    print(f"结果：{ok} PASS / {bad} FAIL  （共 {len(_RESULTS)} 条）")
    if bad:
        for name, passed, detail in _RESULTS:
            if not passed:
                print(f"  FAIL  {name}  — {detail}")
    print("=" * 78)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
