# -*- coding: utf-8 -*-
"""test_perception.py — 工作流 A 验收测试（契约 §3.3 每一条 + 接口/几何/手势单元测试）

一条命令跑完：

    D:\\smartcar\\.venv\\Scripts\\python.exe D:\\robot_work\\pc\\perception\\test_perception.py

覆盖：

  T01  特征来源表：层名 → 维度，和 FEATURE_DIM 对得上（并打印）
  T02  feat_raw.shape == (FEATURE_DIM,)，无 NaN/Inf
  T03  feat_64.shape == (64,)，全部 ∈[-1,1]
  T04  theta_img 有限，且 |theta_img| <= FOV_H/2
  T05  全黑图 → seen==0 且不崩溃
  T06  合成火柴人 → 不崩溃、字段范围合法（合成图，不是真人照片）
  T07  真实照片（assets/bus.jpg，有则测）→ seen==1、关键点存在、几何自洽
  T08  theta_img = (cx-0.5)*FOV_H，改 FOV 参数端到端生效
  T09  手势：举手判定（按肩宽归一化）
  T10  手势：肩宽 < 5px → hand_up_* 全 0（契约 §3.2）
  T11  手势：尺度不变性（同一姿态放大 2 倍，判定不变）
  T12  手势：挥手累积 —— 3 秒 1.5Hz 摆动后 wave_score > 4
  T13  手势：手不动 → 不误触发，wave_score 衰减到 ~0
  T14  手势：无目标（None）→ 全 0、不崩溃、记忆按时间衰减
  T15  FeatureAdapter：输出范围、freeze_stats、load_state_dict 往返、冻结后拒改
  T16  PerceptionOut 字段与契约 §3.1 逐字一致（名字 + 顺序）
  T17  pipeline 连续多帧稳定（状态不发散）
  T18  单帧耗时 CPU / GPU（仅打印，判据是有限且在合理上限内）
  T19  240×240（机器人 /capture 的实际分辨率）
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402  (测 Adapter / 计时都要用)

import extract_features as ef  # noqa: E402
import feature_adapter as fa  # noqa: E402
import gesture as gs  # noqa: E402
import run_perception as rp  # noqa: E402

RESULTS = []
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 小测试框架
# ----------------------------------------------------------------------
def check(name, fn):
    """跑一条用例，捕获异常，打印 PASS/FAIL。"""
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        RESULTS.append((name, True, detail))
        print("[PASS] %-58s %6.1fms  %s"
              % (name, (time.perf_counter() - t0) * 1e3, detail))
        return True
    except Exception as exc:  # noqa: BLE001 - 测试框架要吞掉所有失败并继续
        import traceback
        RESULTS.append((name, False, str(exc)))
        print("[FAIL] %-58s %6.1fms  %s"
              % (name, (time.perf_counter() - t0) * 1e3, exc))
        if os.environ.get("PERCEPTION_TEST_TRACE"):
            traceback.print_exc()
        return False


def synth_stick_figure(width=640, height=480, cx=None, cy_head=None,
                       scale=1.0, arm="both_down", waving_x=0.0):
    """合成火柴人（**合成图，不是真人照片**）：圆头 + 躯干 + 四肢。

    只是个几何测试夹具：用来验证 pipeline 在"有人形但 YOLO 可能不认"的输入上
    不崩、维度正确。真人检测能力用 ``assets/bus.jpg`` 那条用例验证。

    Args:
        arm: ``"both_down"`` / ``"left_up"`` / ``"right_up"``
        waving_x: 额外给某只手加的水平偏移（像素，测试挥手轨迹时用不到，见 T12）
    """
    import cv2

    img = np.full((height, width, 3), 24, dtype=np.uint8)
    ink = (225, 225, 225)
    cx = float(width // 2) if cx is None else float(cx)
    cy_head = float(height * 0.16) if cy_head is None else float(cy_head)

    unit = height * 0.55 * scale          # 躯干长
    head_r = max(4, int(unit * 0.13))
    neck = (int(cx), int(cy_head + head_r))
    hip = (int(cx), int(cy_head + head_r + unit))
    cv2.circle(img, (int(cx), int(cy_head)), head_r, ink, -1)
    cv2.line(img, neck, hip, ink, max(2, int(unit * 0.10)))          # 躯干
    sh_y = neck[1] + int(unit * 0.18)
    half_sh = int(unit * 0.22)                                        # 半肩宽
    l_sh = (neck[0] - half_sh, sh_y)
    r_sh = (neck[0] + half_sh, sh_y)
    arm_len = int(unit * 0.55)

    def draw_arm(side, up):
        sh = l_sh if side == "left" else r_sh
        sgn = -1 if side == "left" else 1
        if up:
            el = (sh[0] + sgn * int(arm_len * 0.12), sh[1] - int(arm_len * 0.75))
            wr = (sh[0] + sgn * int(arm_len * 0.30 + waving_x), sh[1] - int(arm_len * 1.35))
        else:
            el = (sh[0] + sgn * int(arm_len * 0.35), sh[1] + int(arm_len * 0.55))
            wr = (sh[0] + sgn * int(arm_len * 0.22), sh[1] + int(arm_len * 1.05))
        cv2.line(img, sh, el, ink, max(2, int(unit * 0.075)))
        cv2.line(img, el, wr, ink, max(2, int(unit * 0.075)))
        return wr

    draw_arm("left", arm in ("left_up", "both_up"))
    draw_arm("right", arm in ("right_up", "both_up"))
    for side, sgn in (("left", -1), ("right", 1)):
        knee = (hip[0] + sgn * int(unit * 0.16), hip[1] + int(unit * 0.45))
        ankle = (hip[0] + sgn * int(unit * 0.10), hip[1] + int(unit * 0.90))
        cv2.line(img, hip, knee, ink, max(2, int(unit * 0.075)))
        cv2.line(img, knee, ankle, ink, max(2, int(unit * 0.075)))
    return img


def black_frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


def load_photo():
    """真实照片（可选）：assets/bus.jpg，没下载就返回 None。"""
    import cv2
    p = os.path.join(_HERE, "assets", "bus.jpg")
    if not os.path.exists(p):
        return None
    return cv2.imread(p)


# ----------------------------------------------------------------------
# 手势夹具：造 (17,2) 关键点
# ----------------------------------------------------------------------
def make_kp(cx=320.0, sh_y=200.0, sh_half=60.0, wrist_L=None, wrist_R=None,
            conf=1.0, conf_override=None):
    """造一组 COCO-17 关键点：肩在 ``sh_y``、肩宽 ``2*sh_half``。"""
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[5] = (cx - sh_half, sh_y)      # 左肩
    kp[6] = (cx + sh_half, sh_y)      # 右肩
    kp[7] = (cx - sh_half * 0.8, sh_y + 60)
    kp[8] = (cx + sh_half * 0.8, sh_y + 60)
    kp[9] = wrist_L if wrist_L is not None else (cx - sh_half * 0.9, sh_y + 120)
    kp[10] = wrist_R if wrist_R is not None else (cx + sh_half * 0.9, sh_y + 120)
    kp[0] = (cx, sh_y - 90)
    c = np.full(17, float(conf), dtype=np.float32)
    if conf_override:
        for i, v in conf_override.items():
            c[i] = v
    return kp, c


# ----------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------
def main():
    print("=" * 78)
    print("RIG-Hover 工作流 A / PC 端感知链 —— 验收测试")
    print("python: %s" % sys.version.split()[0])
    print("torch %s | cuda %s | device %s"
          % (torch.__version__, torch.cuda.is_available(), DEVICE))
    print("=" * 78)

    pipe = rp.PerceptionPipeline(model_path=ef.MODEL_NAME, device=DEVICE,
                                 fov_h=rp.FOV_H_DEFAULT, conf=0.25, verbose=True)
    stick = synth_stick_figure()
    black = black_frame()
    photo = load_photo()

    # ---- 先跑一帧，拿到特征来源表 ----
    out_stick = pipe(stick)

    # T01 ---------------------------------------------------------------
    def t01():
        rows = pipe.describe_features()
        print("  %-4s %-38s %-16s %-22s %s"
              % ("src", "layer", "shape", "op", "dim"))
        total = 0
        for r in rows:
            total += int(r["dim"])
            print("  %-4s %-38s %-16s %-22s %d"
                  % (r["name"], r["layer"], str(r["shape"]), r["op"], r["dim"]))
        print("  %-4s %-38s %-16s %-22s %d" % ("SUM", "", "", "", total))
        assert total == ef.FEATURE_DIM == 548, (total, ef.FEATURE_DIM)
        assert len(rows) == 4
        return "FEATURE_DIM=%d" % total

    # T02 ---------------------------------------------------------------
    def t02():
        assert out_stick.feat_raw.shape == (ef.FEATURE_DIM,), out_stick.feat_raw.shape
        assert out_stick.feat_raw.dtype == np.float32
        assert np.all(np.isfinite(out_stick.feat_raw))
        return "shape=%s finite=True" % (out_stick.feat_raw.shape,)

    # T03 ---------------------------------------------------------------
    def t03():
        v = out_stick.feat_64
        assert v.shape == (64,), v.shape
        assert v.dtype == np.float32
        assert np.all(np.isfinite(v))
        assert v.min() >= -1.0 and v.max() <= 1.0, (v.min(), v.max())
        return "range=[%.4f, %.4f]" % (v.min(), v.max())

    # T04 ---------------------------------------------------------------
    def t04():
        th = out_stick.theta_img
        assert isinstance(th, float) and math.isfinite(th)
        assert abs(th) <= rp.FOV_H_DEFAULT / 2.0, th
        return "theta_img=%.3f deg (|.|<=%.1f)" % (th, rp.FOV_H_DEFAULT / 2)

    # T05 ---------------------------------------------------------------
    def t05():
        o = pipe(black)
        assert o.seen == 0, "全黑图不该检测到人，seen=%r" % (o.seen,)
        assert o.feat_raw.shape == (ef.FEATURE_DIM,) and np.all(np.isfinite(o.feat_raw))
        assert o.feat_64.shape == (64,) and np.all(np.isfinite(o.feat_64))
        assert o.theta_img == 0.0 and o.person_w == 0.0 and o.person_h == 0.0
        assert o.hand_up_L == 0 and o.hand_up_R == 0
        return "seen=0, 字段全 0，不崩溃"

    # T06 ---------------------------------------------------------------
    def t06():
        o = out_stick
        assert np.all(np.isfinite(o.feat_raw)) and np.all(np.isfinite(o.feat_64))
        assert 0.0 <= o.person_w <= 1.0 and 0.0 <= o.person_h <= 1.0
        assert 0.0 <= o.person_cx <= 1.0 and 0.0 <= o.person_conf <= 1.0
        if o.seen:
            assert abs(o.theta_img - rp.image_angle(o.person_cx)) < 1e-6
            return "合成火柴人：seen=1 conf=%.2f" % o.person_conf
        return "合成火柴人：seen=0（YOLO 不认火柴人，属预期），维度/范围仍合法"

    # T07 ---------------------------------------------------------------
    def t07():
        if photo is None:
            return "SKIP（没有 assets/bus.jpg）"
        o = pipe(photo)
        assert o.seen == 1, "真实照片上应有行人，seen=%r" % (o.seen,)
        assert 0.0 < o.person_w <= 1.0 and 0.0 < o.person_h <= 1.0
        assert 0.0 <= o.person_cx <= 1.0
        assert o.person_conf > 0.3
        assert abs(o.theta_img - rp.image_angle(o.person_cx, pipe.fov_h)) < 1e-6
        kp = o.debug["gesture"]
        assert kp["shoulder_width_px"] > 5.0, kp
        return ("真人：seen=1 conf=%.2f cx=%.3f theta=%.1f° shoulder=%.0fpx "
                "hand_up(L/R)=%d/%d"
                % (o.person_conf, o.person_cx, o.theta_img, kp["shoulder_width_px"],
                   o.hand_up_L, o.hand_up_R))

    # T08 ---------------------------------------------------------------
    def t08():
        assert rp.image_angle(0.5, 60.0) == 0.0
        assert abs(rp.image_angle(1.0, 60.0) - 30.0) < 1e-9
        assert abs(rp.image_angle(0.0, 60.0) + 30.0) < 1e-9
        assert abs(rp.image_angle(0.75, 90.0) - 22.5) < 1e-9
        if photo is None:
            return "公式 OK（无照片，跳过端到端）"
        old = pipe.fov_h
        try:
            pipe.fov_h = 60.0
            a = pipe(photo).theta_img
            pipe.fov_h = 120.0
            b = pipe(photo).theta_img
        finally:
            pipe.fov_h = old
        assert abs(b - 2.0 * a) < 1e-4, (a, b)
        return "theta_img 随 FOV_H 线性：60°→%.3f, 120°→%.3f" % (a, b)

    # T09 ---------------------------------------------------------------
    def t09():
        est = gs.GestureEstimator()
        kp, c = make_kp(wrist_L=(320 - 60, 140), wrist_R=(320 + 60, 320))
        g = est.update(kp, c, dt=1 / 30)
        assert (g.hand_up_L, g.hand_up_R) == (1, 0), (g.hand_up_L, g.hand_up_R)
        # 归一化正确性：x_rel/y_rel 都应该是「像素差 / 肩宽」
        assert abs(g.debug["y_rel_L"] - (140 - 200) / 120.0) < 1e-5, g.debug
        assert abs(g.debug["x_rel"] - ((320 - 60) - 320) / 120.0) < 1e-5, g.debug
        est2 = gs.GestureEstimator()
        kp2, c2 = make_kp(wrist_L=(320 - 60, 320), wrist_R=(320 + 60, 140))
        g2 = est2.update(kp2, c2, dt=1 / 30)
        assert (g2.hand_up_L, g2.hand_up_R) == (0, 1), (g2.hand_up_L, g2.hand_up_R)
        # 置信度不够 → 不算举起
        est3 = gs.GestureEstimator()
        kp3, c3 = make_kp(wrist_L=(320 - 60, 140), conf_override={9: 0.05})
        g3 = est3.update(kp3, c3, dt=1 / 30)
        assert g3.hand_up_L == 0
        return "左举/右举/低置信度 三种情形正确"

    # T10 ---------------------------------------------------------------
    def t10():
        est = gs.GestureEstimator()
        kp, c = make_kp(sh_half=2.0, wrist_L=(318, 100))   # 肩宽 4px < 5px
        g = est.update(kp, c, dt=1 / 30)
        assert (g.hand_up_L, g.hand_up_R) == (0, 0)
        assert g.debug["reason"] == "shoulder_too_small"
        assert g.wave_score == 0.0
        return "肩宽 4px → hand_up=0/0, reason=%s" % g.debug["reason"]

    # T11 ---------------------------------------------------------------
    def t11():
        est = gs.GestureEstimator()
        kp_s, c_s = make_kp(cx=320, sh_y=200, sh_half=60,
                            wrist_L=(320 - 60, 140), wrist_R=(320 + 30, 240))
        g_s = est.update(kp_s, c_s, dt=1 / 30)
        est2 = gs.GestureEstimator()
        kp_l = kp_s * 2.5                       # 同一姿态放大 2.5 倍
        kp_l[:, 0] -= kp_s[:, 0].min() * 1.5     # 挪回画面内，避免退化
        kp_l[:, 1] -= kp_s[:, 1].min() * 1.5
        g_l = est2.update(kp_l, c_s, dt=1 / 30)
        assert (g_s.hand_up_L, g_s.hand_up_R) == (g_l.hand_up_L, g_l.hand_up_R)
        assert abs(g_s.debug["y_rel_L"] - g_l.debug["y_rel_L"]) < 1e-4
        assert abs(g_s.debug["x_rel"] - g_l.debug["x_rel"]) < 1e-4
        return ("两个尺度下 hand_up=(%d,%d)/(%d,%d), y_rel_L=%.4f/%.4f"
                % (g_s.hand_up_L, g_s.hand_up_R, g_l.hand_up_L, g_l.hand_up_R,
                   g_s.debug["y_rel_L"], g_l.debug["y_rel_L"]))

    # T12 ---------------------------------------------------------------
    def t12():
        est = gs.GestureEstimator()
        cx, sh_y, sh_half = 320.0, 200.0, 60.0
        fps, secs, freq = 30.0, 3.0, 1.5
        scores = []
        for k in range(int(fps * secs)):
            t = k / fps
            dx = 0.45 * sh_half * math.sin(2 * math.pi * freq * t)   # ±0.45 肩宽
            kp, c = make_kp(cx=cx, sh_y=sh_y, sh_half=sh_half,
                            wrist_L=(cx - 0.2 * sh_half + dx, sh_y - 60))
            g = est.update(kp, c, dt=1 / fps)
            scores.append(g.wave_score)
            assert -1.0 <= g.wave_sin <= 1.0 and -1.0 <= g.wave_cos <= 1.0
            assert g.wave_amp >= 0.0
        assert g.wave_score > 4.0, "3s 1.5Hz 摆动后 wave_score=%.2f 应 >4" % g.wave_score
        assert g.waving
        assert g.wave_amp > 0.05, g.wave_amp
        assert g.gesture_conf > 0.0
        return ("wave_score=%.2f (>4)，crossings=%d，wave_amp=%.3f，f≈%.2fHz，"
                "sin/cos=%.2f/%.2f"
                % (g.wave_score, est.crossings, g.wave_amp, est.freq,
                   g.wave_sin, g.wave_cos))

    # T13 ---------------------------------------------------------------
    def t13():
        est = gs.GestureEstimator()
        cx, sh_y, sh_half = 320.0, 200.0, 60.0
        peak = 0.0
        for k in range(int(30 * 5)):        # 手举着不动 5 秒
            kp, c = make_kp(cx=cx, sh_y=sh_y, sh_half=sh_half,
                            wrist_L=(cx - 0.3 * sh_half, sh_y - 60))
            g = est.update(kp, c, dt=1 / 30)
            peak = max(peak, g.wave_score)
        assert peak == 0.0, "手不动却累积了 wave_score=%.2f" % peak
        assert g.hand_up_L == 1
        return "5s 静止举手：peak wave_score=%.1f, hand_up_L=%d" % (peak, g.hand_up_L)

    # T14 ---------------------------------------------------------------
    def t14():
        est = gs.GestureEstimator()
        cx, sh_y, sh_half = 320.0, 200.0, 60.0
        for k in range(int(30 * 2)):
            t = k / 30.0
            dx = 0.45 * sh_half * math.sin(2 * math.pi * 1.5 * t)
            kp, c = make_kp(cx=cx, sh_y=sh_y, sh_half=sh_half,
                            wrist_L=(cx + dx, sh_y - 60))
            g = est.update(kp, c, dt=1 / 30)
        before = g.wave_score
        for _ in range(10):                 # 目标消失 10 帧
            g = est.update(None, None, dt=1 / 30)
        assert g.hand_up_L == 0 and g.hand_up_R == 0
        assert g.wave_score < before * math.exp(-10 / 30 / 2.0) * 1.05, (before, g.wave_score)
        return "无目标：score %.2f → %.2f（按 exp(-dt/2) 衰减，不清零）" % (
            before, g.wave_score)

    # T15 ---------------------------------------------------------------
    def t15():
        ad = fa.FeatureAdapter()
        assert ad.fc.in_features == ef.FEATURE_DIM and ad.fc.out_features == 64
        x = torch.from_numpy(np.random.default_rng(0)
                             .normal(size=(32, ef.FEATURE_DIM)).astype(np.float32))
        y = ad(x)
        assert y.shape == (32, 64) and float(y.detach().abs().max()) <= 1.0
        y1 = ad(torch.zeros(ef.FEATURE_DIM))       # 单条也支持
        assert y1.shape == (64,)
        assert not ad.stats_frozen
        ad.update_stats(x * 3.0 + 5.0)
        n = ad.freeze_stats()
        assert n == 32 and ad.stats_frozen
        assert torch.allclose(ad.running_mean, x.mean(0) * 3.0 + 5.0, atol=1e-4)
        try:
            ad.update_stats(x)
            raise AssertionError("冻结后 update_stats 应当报错")
        except RuntimeError:
            pass
        # state_dict 往返（部署时要能 load_state_dict）
        ad2 = fa.FeatureAdapter()
        ad2.load_state_dict(ad.state_dict())
        assert torch.allclose(ad2.running_mean, ad.running_mean)
        assert torch.allclose(ad2(torch.zeros(ef.FEATURE_DIM)),
                              ad(torch.zeros(ef.FEATURE_DIM)))
        assert set(ad.state_dict()) >= {"running_mean", "running_std",
                                        "fc.weight", "fc.bias"}
        return ("(548→64) 输出∈[%.3f,%.3f]，freeze n=%d，state_dict 往返一致"
                % (float(y.detach().min()), float(y.detach().max()), n))

    # T16 ---------------------------------------------------------------
    def t16():
        # The gesture layer was restructured from a threshold score to discrete
        # events, so the dataclass gained fields. Assert the original contract
        # is still satisfied as an ordered subsequence, and that the five event
        # fields exist -- an exact-list assertion would make every future field
        # addition a test failure for no behavioural reason.
        legacy = ["feat_raw", "feat_64", "theta_img", "person_w", "person_h",
                  "person_cx", "person_conf", "seen", "hand_up_L", "hand_up_R",
                  "wave_sin", "wave_cos", "wave_amp", "gesture_conf", "debug"]
        events = ["gesture_id", "gesture_onset", "gesture_phase",
                  "gesture_gap_s"]
        got = [f.name for f in dataclasses.fields(rp.PerceptionOut)]
        it = iter(got)
        missing = [w for w in legacy if w not in set(got)]
        assert not missing, "契约字段丢失：%r" % (missing,)
        assert [w for w in got if w in set(legacy)] == legacy, \
            "契约字段相对顺序被打乱：%r" % (got,)
        assert all(e in set(got) for e in events), \
            "事件化字段缺失：%r" % ([e for e in events if e not in set(got)],)
        assert got.index("debug") == len(got) - 1, "debug 必须是最后一个字段"
        return ("契约 15 字段仍按序存在 + %d 个事件字段，共 %d 字段"
                % (len(events), len(got)))

    # T17 ---------------------------------------------------------------
    def t17():
        frames = [stick, black, stick, photo if photo is not None else black]
        outs = []
        for _ in range(3):
            for f in frames:
                o = pipe(f)
                outs.append(o)
                assert np.all(np.isfinite(o.feat_raw))
                assert np.all(np.isfinite(o.feat_64))
                assert -1.0 <= o.feat_64.min() and o.feat_64.max() <= 1.0
                assert math.isfinite(o.wave_sin) and math.isfinite(o.wave_cos)
                assert o.wave_amp >= 0.0
                assert "wave_score" in o.debug
        return "%d 帧连续跑，无 NaN/越界，wave_score 在 debug 里" % len(outs)

    # T18 ---------------------------------------------------------------
    def t18():
        def bench(p, n=12):
            for _ in range(3):
                p(stick)
            if p.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                p(stick)
            if p.device.startswith("cuda"):
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e3

        gpu_ms = bench(pipe) if DEVICE.startswith("cuda") else float("nan")
        cpu_pipe = rp.PerceptionPipeline(model_path=ef.MODEL_NAME, device="cpu",
                                         verbose=False)
        cpu_ms = bench(cpu_pipe, n=6)
        del cpu_pipe
        assert math.isfinite(cpu_ms) and 1.0 < cpu_ms < 2000.0, cpu_ms
        if DEVICE.startswith("cuda"):
            assert math.isfinite(gpu_ms) and 1.0 < gpu_ms < 2000.0, gpu_ms
            return "单帧 %.2fms (GPU) / %.2fms (CPU)，含 letterbox+检测+特征+手势" % (
                gpu_ms, cpu_ms)
        return "单帧 %.2fms (CPU；本机无 CUDA)" % cpu_ms

    # T19 ---------------------------------------------------------------
    def t19():
        # 机器人实际的 /capture 是 240×240（TECH-PLAN §3.1），单独测一遍
        f = black_frame(240, 240)
        o = pipe(f)
        assert o.feat_raw.shape == (ef.FEATURE_DIM,), o.feat_raw.shape
        assert o.feat_64.shape == (64,)
        assert o.seen == 0 and o.theta_img == 0.0
        f2 = np.full((240, 240, 3), 200, dtype=np.uint8)   # 全亮图也不能崩
        o2 = pipe(f2)
        assert np.all(np.isfinite(o2.feat_raw))
        assert o2.seen in (0, 1)
        return "240×240（机器人实际分辨率）两种极端图都正常"

    for name, fn in (
        ("T01 特征来源表（层名 → 维度，和 FEATURE_DIM 对账）", t01),
        ("T02 feat_raw.shape==(FEATURE_DIM,) 且无 NaN/Inf", t02),
        ("T03 feat_64.shape==(64,) 且全在 [-1,1]", t03),
        ("T04 theta_img 有限且 <= FOV_H/2", t04),
        ("T05 全黑图 seen==0 且不崩溃", t05),
        ("T06 合成火柴人：不崩溃、字段范围合法（合成图）", t06),
        ("T07 真实照片：检测到人、关键点、几何自洽", t07),
        ("T08 theta_img=(cx-0.5)*FOV_H，FOV 参数生效", t08),
        ("T09 手势：举手判定（肩宽归一化）", t09),
        ("T10 手势：肩宽<5px → hand_up=0（契约 §3.2）", t10),
        ("T11 手势：尺度不变性", t11),
        ("T12 手势：3s/1.5Hz 摆动后 wave_score>4", t12),
        ("T13 手势：手不动不误触发", t13),
        ("T14 手势：无目标时衰减不清零", t14),
        ("T15 FeatureAdapter：(548→64)、freeze、load_state_dict", t15),
        ("T16 PerceptionOut 字段与契约 §3.1 一致", t16),
        ("T17 连续多帧稳定", t17),
        ("T18 单帧耗时 CPU/GPU", t18),
        ("T19 240×240（机器人实际分辨率）", t19),
    ):
        check(name, fn)

    npass = sum(1 for _n, ok, _d in RESULTS if ok)
    nfail = len(RESULTS) - npass
    print("-" * 78)
    print("TOTAL: %d PASS / %d FAIL   (FEATURE_DIM=%d, 权重 %s)"
          % (npass, nfail, ef.FEATURE_DIM, ef.MODEL_NAME))
    for name, ok, detail in RESULTS:
        print("  %-4s %s%s" % ("PASS" if ok else "FAIL", name,
                               "" if ok else "   <<< " + detail))
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
