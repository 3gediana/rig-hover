# -*- coding: utf-8 -*-
"""test_scenario_v2.py -- 场景 v2 决策层自检。

纯 numpy，秒级跑完，不需要 GPU / 机器人。

重点覆盖用户明确提的三个细节：
  * **工作距离 1–2 米**：太近（整屏是脸）和太远都不跟
  * **敲的位置要记住、要盯着走过去**：手离开后仍能走完，且到达后停住
  * **看人优先动脖子，脖子够不到才动轮子**：小误差时轮子必须为 0

Run:  python test_scenario_v2.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scenario_v2 as sv  # noqa: E402

N_PASS = 0
N_FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {name}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def main() -> int:
    P = sv.V2Params()

    print("== 距离门：1–2 米（太近太远都不跟）==")
    for scale, want, label in (
            (0.020, sv.MODE_IDLE, "太远"),
            (0.045, sv.MODE_GAZE_FACE, "下限内"),
            (0.120, sv.MODE_GAZE_FACE, "中间（≈1.5m）"),
            (0.220, sv.MODE_GAZE_FACE, "上限内"),
            (0.260, sv.MODE_IDLE, "太近（整屏是脸）")):
        c = sv.ScenarioV2(P)
        a = c.step(0.05, theta_face=0.0, face_scale=scale, face_valid=1)
        check(f"{label} scale={scale} -> {sv.MODE_NAMES[a.mode]}",
              a.mode == want, f"got {sv.MODE_NAMES[a.mode]}")

    print("\n== 优先动脖子；且目标必须在画面中心（不许斜眼）==")
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    a_db = c.step(0.05, theta_face=1.5, face_scale=0.12, face_valid=1)
    check("小偏角（1.5°）机身不动", a_db.vyaw_des == 0.0,
          f"vyaw={a_db.vyaw_des}")
    check("小偏角也要把头精确对准（不留永久偏角）",
          abs(a_db.head_des - 1.5) < 1e-6, f"head={a_db.head_des}")
    check("小偏角时残差仍为 0",
          abs(a_db.debug.get("off_center_deg", 999)) < 1e-6,
          f"off_center={a_db.debug.get('off_center_deg')}")

    for deg in (5.0, 12.0, 30.0, 40.0):
        c = sv.ScenarioV2(P)
        c.step(0.05, face_scale=0.12, face_valid=1)
        a = c.step(0.05, theta_face=deg, face_scale=0.12, face_valid=1)
        check(f"{deg:.0f}° 头满量程对准、机身不动",
              abs(a.head_des - deg) < 1e-6 and a.vyaw_des == 0.0,
              f"head={a.head_des} vyaw={a.vyaw_des}")
        check(f"{deg:.0f}° 注视残差为 0（目标在画面正中）",
              abs(a.debug.get("off_center_deg", 999)) < 1e-6,
              f"off_center={a.debug.get('off_center_deg')}")

    # 超过头的量程才动机身
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    a_over = c.step(0.05, theta_face=55.0, face_scale=0.12, face_valid=1)
    check("55° 超量程 -> 头限位 45°", abs(a_over.head_des - 45.0) < 1e-6,
          f"head={a_over.head_des}")
    check("55° 超量程 -> 机身必须转", a_over.vyaw_des > 0.0,
          f"vyaw={a_over.vyaw_des}")
    check("超量程时残差 = 超出量（10°）",
          abs(a_over.debug.get("off_center_deg", 0) - 10.0) < 1e-6,
          f"off_center={a_over.debug.get('off_center_deg')}")

    # 机身转过之后，残差必须回到 0
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    c.step(0.05, theta_face=55.0, face_scale=0.12, face_valid=1)
    a_after = c.step(0.05, theta_face=30.0, face_scale=0.12, face_valid=1)
    check("机身转过之后残差归零",
          abs(a_after.debug.get("off_center_deg", 999)) < 1e-6,
          f"off_center={a_after.debug.get('off_center_deg')}")
    check("此时机身又不动了", a_after.vyaw_des == 0.0,
          f"vyaw={a_after.vyaw_des}")

    print("\n== 握拳优先盯拳头 ==")
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    a = c.step(0.05, theta_face=0.0, face_scale=0.12, face_valid=1, fist_prob=0.9)
    check("fist 概率高 -> GAZE_FIST", a.mode == sv.MODE_GAZE_FIST,
          sv.MODE_NAMES[a.mode])
    a_keep = c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.5)
    check("fist 概率中等时保持", a_keep.mode == sv.MODE_GAZE_FIST,
          sv.MODE_NAMES[a_keep.mode])
    # 松手后要经过 fist_lost_s 才退回
    t = 0.0
    for _ in range(int(P.fist_lost_s / 0.05) + 4):
        t += 0.05
        a_back = c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.0)
    check("松手后回到 GAZE_FACE", a_back.mode == sv.MODE_GAZE_FACE,
          sv.MODE_NAMES[a_back.mode])

    print("\n== 敲两下 -> 锁定目标 -> 走过去 ==")
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9,
           theta_fist=0.0, hand_valid=1, person_h=0.5)
    a = c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9,
               theta_fist=0.0, hand_valid=1, person_h=0.5,
               knock_fired=1, knock_x=120.0, knock_y=180.0)
    check("敲击 -> APPROACH", a.mode == sv.MODE_APPROACH, sv.MODE_NAMES[a.mode])
    check("目标已锁定", c.target.valid == 1)
    check("目标距离由 person_h 估出（0.84/0.5=1.68m）",
          abs(c.target.dist_m - 1.68) < 1e-6, f"{c.target.dist_m}")
    check("锁定帧记录了敲击点像素坐标",
          c.target.x_px == 120.0 and c.target.y_px == 180.0)

    print("\n== 手离开后仍然走过去（哪怕拳头离开）==")
    wheel = 0.0
    modes = []
    for _ in range(400):
        a = c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.0,
                   hand_valid=0, wheel_x=wheel, person_h=0.5)
        modes.append(a.mode)
        wheel += max(0.0, a.vx_des) * 0.05
        if a.mode == sv.MODE_DWELL:
            break
    check("手全程不可见，仍然到达并停住",
          a.mode == sv.MODE_DWELL, sv.MODE_NAMES[a.mode])
    check("停住时 vx==0", a.vx_des == 0.0, f"vx={a.vx_des}")
    drive_cap = c.target.dist_m - P.overshoot_margin_m
    check(f"走了约 {drive_cap:.2f} 米（目标 1.68 - 保险 0.12）",
          1.35 < wheel <= drive_cap + 1e-6,
          f"wheel_x={wheel:.3f} cap={drive_cap:.3f}")
    check("**没有走到 1.68 m**（过冲保险生效）", wheel <= drive_cap + 1e-6,
          f"wheel_x={wheel:.3f}")
    check("目标记忆没有被清空", c.target.valid == 1)

    print("\n== 到达后锁死：不再前进 ==")
    # 到达后即使拳头重新出现、目标方位被修正，也必须保持 vx==0
    for _ in range(5):
        a_lock = c.step(0.05, face_scale=0.12, face_valid=1, hand_valid=1,
                        theta_fist=30.0, wheel_x=wheel, person_h=0.5)
        wheel += max(0.0, a_lock.vx_des) * 0.05
    check("到达后 vx 恒为 0", a_lock.vx_des == 0.0, f"vx={a_lock.vx_des}")
    check("到达后不再累积里程", abs(wheel - (drive_cap - P.arrive_tol_m)) < 0.15,
          f"wheel={wheel:.3f}")

    print("\n== 停一会儿再看人脸 ==")
    c.step(0.05, face_scale=0.12, face_valid=1, theta_face=5.0)
    c.step(0.05, face_scale=0.12, face_valid=1, theta_face=5.0)
    check("dwell 期间保持 DWELL", c.mode == sv.MODE_DWELL,
          sv.MODE_NAMES[c.mode])
    for _ in range(int(P.dwell_s / 0.05) + 4):
        a = c.step(0.05, face_scale=0.12, face_valid=1, theta_face=5.0)
    check("dwell 结束后回到 GAZE_FACE", a.mode == sv.MODE_GAZE_FACE,
          sv.MODE_NAMES[a.mode])
    check("回到看脸后头指向人脸", abs(a.head_des - 5.0) < 1e-6,
          f"head={a.head_des}")

    print("\n== 走过去途中人贴太近就停 ==")
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9,
           hand_valid=1, person_h=0.5)
    c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9, hand_valid=1,
           person_h=0.5, knock_fired=1, knock_x=120, knock_y=180)
    a = c.step(0.05, face_scale=0.30, face_valid=1, hand_valid=0, person_h=0.5)
    check("人脸尺度过大 -> 立刻停为 DWELL", a.mode == sv.MODE_DWELL,
          sv.MODE_NAMES[a.mode])
    check("停的原因是 person_too_close",
          a.debug.get("reason") == "person_too_close", str(a.debug.get("reason")))

    print("\n== 没人时什么都不做 ==")
    c = sv.ScenarioV2(P)
    a = c.step(0.05, face_valid=0)
    check("没人 -> IDLE", a.mode == sv.MODE_IDLE)
    check("没人 -> 三路指令全 0",
          a.head_des == 0.0 and a.vyaw_des == 0.0 and a.vx_des == 0.0)

    print("\n== 目标锁定的可达性：方位修正 ==")
    c = sv.ScenarioV2(P)
    c.step(0.05, face_scale=0.12, face_valid=1)
    c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9, hand_valid=1,
           person_h=0.5)
    c.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9, hand_valid=1,
           person_h=0.5, knock_fired=1, knock_x=200, knock_y=180)
    b0 = c.target.bearing_body
    # 敲完之后手还在视野里但方位变了 -> 目标方位应被观测修正
    a = c.step(0.05, face_scale=0.12, face_valid=1, hand_valid=1,
               theta_fist=-20.0, q_head=0.0, wheel_x=0.0, person_h=0.5)
    check("手可见时目标方位被修正",
          abs(c.target.bearing_body - (-20.0)) < 1e-6,
          f"{b0} -> {c.target.bearing_body}")
    # 前进要先让**机身**对准（否则直着走会跑到一边），-20° 超过 turn_tol 就原地转
    check("-20° 超过对正容差 -> 先原地转",
          a.debug.get("phase") == "turn", str(a.debug.get("phase")))

    # 机身基本对准之后再走
    a_drive = c.step(0.05, face_scale=0.12, face_valid=1, hand_valid=1,
                     theta_fist=-3.0, q_head=0.0, wheel_x=0.0, person_h=0.5)
    check("-3° 在对正容差内 -> 直接前进",
          a_drive.debug.get("phase") == "drive",
          str(a_drive.debug.get("phase")))
    check("前进时 vx>0", a_drive.vx_des > 0.0, f"vx={a_drive.vx_des}")
    check("前进时目标仍在画面中心（残差 0）",
          abs(a_drive.debug.get("off_center_deg", 999)) < 1e-6,
          f"off_center={a_drive.debug.get('off_center_deg')}")

    # 方位超出头量程时，前进前必须先原地转
    a_turn = c.step(0.05, face_scale=0.12, face_valid=1, hand_valid=1,
                    theta_fist=-60.0, q_head=0.0, wheel_x=0.0, person_h=0.5)
    check("方位 -60° 超量程 -> 先原地转",
          a_turn.debug.get("phase") == "turn", str(a_turn.debug.get("phase")))
    check("原地转的时候 vx==0", a_turn.vx_des == 0.0, f"vx={a_turn.vx_des}")

    print("\n== 距离估计误差下的过冲边界 ==")
    # 不做过沿检测之后，「不会掉下去」完全依赖「不过冲」。
    # 保险量给出的保证是：只要距离估计误差 e <= margin + arrive_tol，
    # 就绝不会走过真实目标点。
    def travel_with(est_m, margin):
        pp = sv.V2Params(overshoot_margin_m=margin)
        c4 = sv.ScenarioV2(pp)
        c4.step(0.05, face_scale=0.12, face_valid=1)
        # 必须先真的触发一次敲击，否则根本不会进 APPROACH（第一版漏了这步）
        c4.step(0.05, face_scale=0.12, face_valid=1, fist_prob=0.9,
                hand_valid=1, person_h=0.5,
                knock_fired=1, knock_x=120.0, knock_y=180.0)
        assert c4.mode == sv.MODE_APPROACH, sv.MODE_NAMES[c4.mode]
        c4.target.dist_m = float(est_m)
        c4.target.remaining_m = float(est_m)
        w = 0.0
        for _ in range(800):
            a4 = c4.step(0.05, face_scale=0.12, face_valid=1, hand_valid=0,
                         wheel_x=w, person_h=0.5)
            w += max(0.0, a4.vx_des) * 0.05
            if a4.mode == sv.MODE_DWELL:
                break
        return w

    truth = 1.68
    bound = P.overshoot_margin_m + P.arrive_tol_m
    w1 = travel_with(truth + 0.10, P.overshoot_margin_m)
    check(f"高估 0.10m（< 边界 {bound:.2f}）-> 不过冲", w1 <= truth,
          f"travel={w1:.2f} > truth={truth}")
    check("而且确实是欠调（安全方向）", w1 < truth - 0.02,
          f"travel={w1:.2f}")
    w2 = travel_with(truth + 0.30, P.overshoot_margin_m)
    check(f"高估 0.30m（> 边界 {bound:.2f}）-> 会过冲（这就是为什么边界有意义）",
          w2 > truth, f"travel={w2:.2f}")

    print("\n" + "=" * 60)
    print(f"{N_PASS} PASS / {N_FAIL} FAIL")
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
