# -*- coding: utf-8 -*-
"""test_gesture.py — 手势识别事件的验收测试（一条命令跑完，打印 PASS/FAIL）

    python test_gesture.py

检查项：

1. 分类器在**按人留出**的测试集上的每类 F1（有回归下限）
2. 一段连续挥手 → onset 数量 == 1
3. 两段挥手、间隔 1 s → onset 数量 == 2
4. 持续静止 10 s → onset 数量 == 0
5. 所有输出字段有限且在定义域内
6. ``gesture_gap_s`` 单调性（只在 GESTURE 结束那一帧重置）

另外断言 train/val/test 的 user 列表两两不相交——「按人划分」这件事必须
是可验证的，不能只在文档里说说。

运行完打印 ``PASS``/``FAIL`` 汇总；有失败项时进程返回非零。
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
for p in (HERE, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from episodes import make_episode  # noqa: E402
from gesture_classifier import MODEL_PATH, GestureClassifier, fold_probs  # noqa: E402
from gesture_common import CLASSES, CROP_PX  # noqa: E402
from gesture_events import (GestureEventMachine, PHASE_GESTURE,  # noqa: E402
                            PHASE_IDLE, PHASE_PENDING, PHASE_RETURN)
from metrics import per_class_report  # noqa: E402
from synth import PoseBank  # noqa: E402

CFG = os.path.join(HERE, "configs", "gesture_events.yaml")
DATA = os.path.join(HERE, "data", "synth")
FPS = 30.0
DT = 1.0 / FPS

#: 每类 F1 的回归下限（比实测低一截，只用来抓退化，不粉饰结果）
MIN_F1 = {"idle": 0.80, "wave": 0.90, "open_palm": 0.65,
          "thumbs_up": 0.60, "pointing": 0.30, "fist": 0.80}
MIN_MACRO_F1 = 0.70

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                          ("  " + detail) if detail else ""), flush=True)
    return ok


# ----------------------------------------------------------------------
def test_split_disjoint():
    """train/val/test 的 user 必须两两不相交。"""
    import json
    with open(os.path.join(DATA, "build_meta.json"), "r",
              encoding="utf-8") as fh:
        meta = json.load(fh)
    u = {k: set(v) for k, v in meta["users"].items()}
    ok = (not (u["train"] & u["val"]) and not (u["train"] & u["test"])
          and not (u["val"] & u["test"]))
    return check("按人划分：train/val/test user 互不相交", ok,
                 "train %d / val %d / test %d 人"
                 % (len(u["train"]), len(u["val"]), len(u["test"])))


def test_classifier_f1(clf):
    """测试集每类 F1（按人留出）。

    整批窗口直接过模型（不做逐帧滑窗重放，两者等价但批处理快两个数量级）。
    """
    import torch

    z = np.load(os.path.join(DATA, "test.npz"))
    feat = torch.from_numpy(z["feat"].astype(np.float32))
    y = z["y"].astype(np.int64)
    y6 = np.where(y == len(CLASSES), 0, y)          # other → idle
    crop = None
    if clf.use_appearance:
        crop = torch.from_numpy(
            z["crop"].astype(np.float32) / 127.5 - 1.0).unsqueeze(2)

    t0 = time.time()
    probs = []
    batch = 512
    with torch.no_grad():
        for s in range(0, len(y), batch):
            fb = feat[s:s + batch].to(clf.device)
            cb = crop[s:s + batch].to(clf.device) if crop is not None else None
            logits = clf.model(fb, cb)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    p6 = fold_probs(np.concatenate(probs, axis=0), clf.model_classes)
    pred = p6.argmax(axis=1)
    rep = per_class_report(y6, pred, CLASSES)
    macro = float(np.mean([r["f1"] for r in rep]))
    ok = True
    for r in rep:
        floor = MIN_F1.get(r["class"], 0.0)
        good = r["f1"] >= floor
        ok = ok and good
        print("      %-10s P %.3f  R %.3f  F1 %.3f  (下限 %.2f, n=%d)%s"
              % (r["class"], r["precision"], r["recall"], r["f1"], floor,
                 r["support"], "" if good else "   ← 低于下限"),
              flush=True)
    ok = ok and macro >= MIN_MACRO_F1
    print("      macro-F1 %.4f (下限 %.2f)  用时 %.1fs"
          % (macro, MIN_MACRO_F1, time.time() - t0), flush=True)
    return check("分类器测试集每类 F1 达标", ok,
                 "macro-F1 %.3f, acc %.3f" % (macro, float((pred == y6).mean())))


def _bank(split="test"):
    z = np.load(os.path.join(DATA, "bank_%s.npz" % split))
    return PoseBank(z["kp_rel"], z["conf"], z["crop"], z["label"], z["user"])


def _run_episode(clf, ep, cfg):
    """把 episode 跑过 分类器 + 事件机，返回 (outputs, probs, latency_ms)。"""
    clf.reset()
    machine = GestureEventMachine(cfg)
    outs = []
    for i in range(len(ep.feat)):
        p = clf.push_feat(ep.feat[i], ep.crop[i])
        outs.append(machine.update(p, DT))
    starts = ep.starts_of("wave") + ep.starts_of("raise")
    lat = None
    if starts:
        s0 = min(starts)
        onsets = [i for i, o in enumerate(outs) if o.gesture_onset == 1]
        after = [i for i in onsets if i >= s0]
        if after:
            lat = 1000.0 * (after[0] - s0) * DT
    return outs, lat


def test_single_onset(clf, cfg):
    """连续挥手（0.5/1.0/2.0 Hz）只能出 1 个 onset。"""
    bank = _bank("test")
    ok, rows = True, []
    for freq in (0.5, 1.0, 2.0):
        rng = np.random.default_rng(100 + int(freq * 10))
        plan = [("idle", 1.2), ("raise", 0.5, "open_palm"), ("wave", 4.0),
                ("idle", 1.5)]
        ep = make_episode(bank, rng, int(bank.users[7]), plan,
                          name="cont_%.1f" % freq, freq_hz=freq, amp=0.55)
        outs, lat = _run_episode(clf, ep, cfg)
        n = sum(1 for o in outs if o.gesture_onset == 1)
        # onset 必须是脉冲：不能连续两帧是 1
        runs = _onset_runs(outs)
        good = (n == 1) and all(r == 1 for r in runs)
        ok = ok and good
        rows.append("%.1fHz→%d个(%s)" % (freq, n, "%.0fms" % lat
                                         if lat else "无"))
    return check("一段连续挥手只产生 1 个 onset", ok, " / ".join(rows))


def test_two_onsets(clf, cfg):
    """两次挥手、间隔 1 s → 必须出 2 个 onset。"""
    bank = _bank("test")
    ok, rows = True, []
    for gap in (1.0, 1.5):
        rng = np.random.default_rng(200 + int(gap * 10))
        plan = [("idle", 1.0), ("raise", 0.4, "open_palm"), ("wave", 1.6),
                ("idle", gap), ("raise", 0.4, "open_palm"), ("wave", 1.6),
                ("idle", 1.2)]
        ep = make_episode(bank, rng, int(bank.users[11]), plan,
                          name="two_%.1f" % gap, freq_hz=1.5, amp=0.55)
        outs, lat = _run_episode(clf, ep, cfg)
        n = sum(1 for o in outs if o.gesture_onset == 1)
        good = n == 2
        ok = ok and good
        rows.append("间隔%.1fs→%d个" % (gap, n))
    return check("间隔 1 s 的两次挥手产生 2 个 onset", ok, " / ".join(rows))


def test_no_onset_when_still(clf, cfg):
    """持续静止 10 s → 0 个 onset。"""
    bank = _bank("test")
    rng = np.random.default_rng(300)
    ep = make_episode(bank, rng, int(bank.users[3]), [("idle", 10.0)],
                      name="still")
    outs, _ = _run_episode(clf, ep, cfg)
    n = sum(1 for o in outs if o.gesture_onset == 1)
    return check("持续静止不产生 onset", n == 0, "10 s 内 %d 个" % n)


def test_fields(clf, cfg):
    """所有输出字段有限且在定义域内。"""
    bank = _bank("test")
    rng = np.random.default_rng(400)
    plan = [("idle", 1.0), ("raise", 0.5, "open_palm"), ("wave", 2.0),
            ("hold", 1.5, "fist"), ("idle", 1.0)]
    ep = make_episode(bank, rng, int(bank.users[5]), plan, freq_hz=1.5,
                      amp=0.6)
    outs, _ = _run_episode(clf, ep, cfg)
    gap_cap = float(cfg.get("gap_cap_s", 3600.0))
    bad = []
    for i, o in enumerate(outs):
        if o.gesture_id not in range(len(CLASSES)):
            bad.append((i, "gesture_id", o.gesture_id))
        if o.gesture_onset not in (0, 1):
            bad.append((i, "gesture_onset", o.gesture_onset))
        if o.gesture_phase not in (PHASE_IDLE, PHASE_PENDING, PHASE_GESTURE,
                                   PHASE_RETURN):
            bad.append((i, "gesture_phase", o.gesture_phase))
        if not (np.isfinite(o.gesture_conf) and 0.0 <= o.gesture_conf <= 1.0):
            bad.append((i, "gesture_conf", o.gesture_conf))
        if not (np.isfinite(o.gesture_gap_s) and 0.0 <= o.gesture_gap_s
                <= gap_cap):
            bad.append((i, "gesture_gap_s", o.gesture_gap_s))
    return check("输出字段有限且在范围内", not bad,
                 "%d 帧" % len(outs) if not bad else str(bad[:3]))


def test_gap_monotonic(clf, cfg):
    """gesture_gap_s 单调性：只有 GESTURE 结束（进入 RETURN）那一帧允许重置。"""
    bank = _bank("test")
    rng = np.random.default_rng(500)
    plan = [("idle", 1.0), ("raise", 0.4, "open_palm"), ("wave", 1.6),
            ("idle", 1.0), ("raise", 0.4, "open_palm"), ("wave", 1.6),
            ("idle", 1.5)]
    ep = make_episode(bank, rng, int(bank.users[13]), plan, freq_hz=1.5,
                      amp=0.55)
    outs, _ = _run_episode(clf, ep, cfg)
    bad = []
    prev = outs[0].gesture_gap_s
    for i in range(1, len(outs)):
        cur = outs[i].gesture_gap_s
        reset_ok = (outs[i].gesture_phase == PHASE_RETURN
                    and outs[i - 1].gesture_phase != PHASE_RETURN)
        if not reset_ok and cur < prev - 1e-9:
            bad.append((i, round(prev, 3), round(cur, 3)))
        prev = cur
    return check("gesture_gap_s 单调（不倒退）", not bad,
                 "%d 帧" % len(outs) if not bad else str(bad[:3]))


def _onset_runs(outs):
    """每个 onset 连续段的长度（应该恒为 1：边沿不是电平）。"""
    runs, cur = [], 0
    for o in outs:
        if o.gesture_onset == 1:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


# ----------------------------------------------------------------------
def main():
    t_all = time.time()
    print("=" * 68)
    print("手势识别事件化 — 验收测试")
    print("=" * 68, flush=True)
    if not os.path.exists(MODEL_PATH):
        print("FAIL: 找不到模型 %s（先跑 src/train_classifier.py）" % MODEL_PATH)
        return 1
    device = "cuda:0" if _cuda() else "cpu"
    clf = GestureClassifier.load(MODEL_PATH, device=device)
    import yaml
    with open(CFG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    print("分类器: %r\n事件配置: %s\n" % (clf, os.path.basename(CFG)), flush=True)

    for name, fn in (("split", lambda: test_split_disjoint()),
                     ("f1", lambda: test_classifier_f1(clf)),
                     ("one", lambda: test_single_onset(clf, cfg)),
                     ("two", lambda: test_two_onsets(clf, cfg)),
                     ("still", lambda: test_no_onset_when_still(clf, cfg)),
                     ("fields", lambda: test_fields(clf, cfg)),
                     ("gap", lambda: test_gap_monotonic(clf, cfg))):
        try:
            ok = fn()
        except Exception:
            traceback.print_exc()
            ok = check("%s 抛异常" % name, False)
        # 有失败也继续跑完，一次看全所有问题

    n = len(_RESULTS)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    print("-" * 68)
    print("%s   %d/%d 项通过   用时 %.1fs"
          % ("PASS" if passed == n else "FAIL", passed, n,
             time.time() - t_all))
    print("-" * 68)
    return 0 if passed == n else 1


def _cuda():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
