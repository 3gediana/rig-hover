# -*- coding: utf-8 -*-
"""test_scene_signals.py -- 场景 v2 信号层自检。

只依赖 numpy，不加载 torch / ultralytics，因此可以在没有 GPU 的机器上跑。

覆盖：
  * 方位换算
  * 人脸信号：可用 / 缺关键点低置信 / 无输入
  * 腕部归一化与**距离无关**（同一个动作在 1 m 和 3 m 处 y_norm 相同）
  * 敲击检测器：两次敲击触发一次；单次不触发；间隔太远不触发；
    不应期内不重复；断流重置；手不在桌面高度不计数

Run:  python test_scene_signals.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scene_signals as ss  # noqa: E402

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


def near(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def make_person(img_h=240.0, img_w=240.0, scale=1.0, cx=0.5, cy=0.5):
    """造一个人的 17 关键点；scale 模拟人在画面里的大小（1.0 ≈ 近距离）。"""
    kxy = np.zeros((17, 2), np.float32)
    kconf = np.full(17, 0.9, np.float32)
    X = lambda fx: cx * img_w + (fx - 0.5) * img_w * scale
    Y = lambda fy: cy * img_h + (fy - 0.5) * img_h * scale
    # 头
    kxy[ss.COCO["nose"]] = (X(0.500), Y(0.300))
    kxy[ss.COCO["eye_l"]] = (X(0.480), Y(0.295))
    kxy[ss.COCO["eye_r"]] = (X(0.520), Y(0.295))
    kxy[ss.COCO["ear_l"]] = (X(0.460), Y(0.300))
    kxy[ss.COCO["ear_r"]] = (X(0.540), Y(0.300))
    # 上身
    kxy[ss.COCO["sh_l"]] = (X(0.440), Y(0.400))
    kxy[ss.COCO["sh_r"]] = (X(0.560), Y(0.400))
    kxy[ss.COCO["el_l"]] = (X(0.420), Y(0.550))
    kxy[ss.COCO["el_r"]] = (X(0.580), Y(0.550))
    kxy[ss.COCO["wr_l"]] = (X(0.415), Y(0.700))
    kxy[ss.COCO["wr_r"]] = (X(0.585), Y(0.700))
    # 下身
    kxy[ss.COCO["hip_l"]] = (X(0.450), Y(0.650))
    kxy[ss.COCO["hip_r"]] = (X(0.550), Y(0.650))
    kxy[ss.COCO["kn_l"]] = (X(0.450), Y(0.850))
    kxy[ss.COCO["kn_r"]] = (X(0.550), Y(0.850))
    kxy[ss.COCO["an_l"]] = (X(0.450), Y(1.000))
    kxy[ss.COCO["an_r"]] = (X(0.550), Y(1.000))
    return kxy, kconf


def main() -> int:
    print("== 方位换算 ==")
    check("画面正中 -> 0 度", near(ss.image_angle(0.5, 60.0), 0.0))
    check("右边 0.75 -> +15 度", near(ss.image_angle(0.75, 60.0), 15.0))
    check("左边 0.25 -> -15 度", near(ss.image_angle(0.25, 60.0), -15.0))

    print("\n== 人脸信号 ==")
    kxy, kconf = make_person(cx=0.5)
    f = ss.face_signals(kxy, kconf, 240.0, 240.0)
    check("有人脸时 valid=1", f["face_valid"] == 1, str(f))
    check("脸在正中 theta≈0", abs(f["theta_face"]) < 1e-3, str(f["theta_face"]))
    kxy2, kconf2 = make_person(cx=0.7)
    f2 = ss.face_signals(kxy2, kconf2, 240.0, 240.0)
    check("脸在右侧 theta>0", f2["theta_face"] > 8.0, str(f2["theta_face"]))

    kconf_low = kconf.copy()
    for idx in (ss.COCO["nose"], ss.COCO["eye_l"], ss.COCO["eye_r"]):
        kconf_low[idx] = 0.05
    f3 = ss.face_signals(kxy, kconf_low, 240.0, 240.0)
    check("关键点全低置信 -> valid=0", f3["face_valid"] == 0, str(f3))
    f4 = ss.face_signals(None, None, 240.0, 240.0)
    check("无输入 -> valid=0", f4["face_valid"] == 0)

    print("\n== 人脸尺度随距离变小 ==")
    fl = ss.face_signals(*make_person(scale=1.0), 240.0, 240.0)
    fs = ss.face_signals(*make_person(scale=0.35), 240.0, 240.0)
    check("近处 face_scale > 远处", fl["face_scale"] > fs["face_scale"],
          f"{fl['face_scale']:.4f} vs {fs['face_scale']:.4f}")
    check("尺度比约等于 1/0.35",
          near(fl["face_scale"] / max(fs["face_scale"], 1e-9), 1 / 0.35, tol=0.02),
          f"{fl['face_scale'] / max(fs['face_scale'], 1e-9):.3f}")

    print("\n== 腕部归一化与距离无关 ==")
    hl = ss.hand_signals(*make_person(scale=1.0), 240.0, 240.0)
    hs = ss.hand_signals(*make_person(scale=0.35), 240.0, 240.0)
    yl = hl["hands"]["L"]["wrist_y_norm"]
    ys = hs["hands"]["L"]["wrist_y_norm"]
    check("1m 与 3m 的 wrist_y_norm 相同", near(yl, ys, tol=1e-4),
          f"{yl:.4f} vs {ys:.4f}")
    check("体长来源是肩-髋", hl["body_scale_src"] == "shoulder-hip",
          hl["body_scale_src"])
    # 腕部的**角偏移**不是距离不变量：人在 3 m 处整体张角更小，
    # 同一个姿势的腕部自然更靠近画面中心。归一化高度才是那个不变量。
    check("远处的人腕部角偏移更小（透视正确）",
          abs(hs["hands"]["L"]["theta_fist"]) < abs(hl["hands"]["L"]["theta_fist"]),
          f"{hs['hands']['L']['theta_fist']:.2f} vs "
          f"{hl['hands']['L']['theta_fist']:.2f}")

    print("\n== 主导手选择 ==")
    side, h = ss.dominant_hand(hl["hands"], {"L": 0.1, "R": 0.9})
    check("fist 概率高的手被选中", side == "R", side)
    side2, _ = ss.dominant_hand({"L": {"valid": 0}, "R": {"valid": 0}}, None)
    check("两只手都不可用时返回 valid=0", side2 == "L")

    print("\n== 敲击检测器 ==")

    def run_traj(frames, params=None):
        det = ss.KnockDetector(params)
        outs = []
        for (t, y, x, py) in frames:
            outs.append(det.update(t, y, x, py, valid=1))
        return outs

    def knock_at(t0, y_base=0.30, y_down=0.90, dur=0.30, fps=30.0):
        """一次敲击：从 y_base 下探到 y_down 再回弹。"""
        frames = []
        n = int(dur * fps)
        for k in range(n):
            u = k / max(n - 1, 1)
            y = y_base + (y_down - y_base) * math.sin(math.pi * u)
            frames.append((t0 + k / fps, y, 120.0, 180.0))
        return frames

    traj = knock_at(0.4) + knock_at(0.9)
    outs = run_traj(traj)
    fired = sum(o.fired for o in outs)
    check("两次敲击 -> 触发 1 次", fired == 1, f"fired={fired}")
    check("触发帧带着敲击点坐标",
          any(o.fired and o.x_px > 0 for o in outs))

    outs1 = run_traj(knock_at(0.4))
    check("单次敲击 -> 不触发", sum(o.fired for o in outs1) == 0)

    outs_far = run_traj(knock_at(0.4) + knock_at(2.6))
    check("两次间隔超过窗口 -> 不触发", sum(o.fired for o in outs_far) == 0,
          f"窗口 {ss.KnockParams().window_s}s")

    outs_tri = run_traj(knock_at(0.4) + knock_at(0.9) + knock_at(1.4))
    check("三连敲 -> 只触发 1 次（不应期内不重复）",
          sum(o.fired for o in outs_tri) == 1,
          f"fired={sum(o.fired for o in outs_tri)}")

    # 手一直在桌面高度但没有下探-回弹，不该被算成敲击
    flat = [(0.05 * k, 0.70, 120.0, 180.0) for k in range(40)]
    outs_flat = run_traj(flat)
    check("手停在桌面高度 -> 不触发", sum(o.fired for o in outs_flat) == 0)

    # 手在胸前摆动（没到桌面高度）不该计数
    chest = [(0.05 * k, 0.20 + 0.05 * math.sin(k), 120.0, 120.0)
             for k in range(40)]
    outs_chest = run_traj(chest)
    check("手在胸前摆动 -> 不触发", sum(o.fired for o in outs_chest) == 0)

    # 断流：中间空一大段，峰谷状态要作废；断流**之后**的两次敲击仍要能判定。
    # 断流前后的各一次敲击本来就不该配对（间隔远超窗口），这是正确行为。
    gappy = (knock_at(0.4) + [(5.0, 0.90, 120.0, 180.0)]
             + knock_at(5.5) + knock_at(6.0))
    outs_gap = run_traj(gappy)
    check("断流后仍能正常判定", sum(o.fired for o in outs_gap) == 1,
          f"fired={sum(o.fired for o in outs_gap)}")
    outs_span = run_traj(knock_at(0.4) + knock_at(5.5))
    check("跨断流的两次敲击不配对", sum(o.fired for o in outs_span) == 0)

    print("\n" + "=" * 60)
    print(f"{N_PASS} PASS / {N_FAIL} FAIL")
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
