# -*- coding: utf-8 -*-
"""calibrate_events.py — 用带真值起点的长序列标定事件机参数（任务 C）

标定对象：``PENDING`` 的确认窗口（``pending_hold``）、进入/退出阈值（迟滞）、
``release``（判动作结束）。目标是**触发延迟最小**，同时约束漏检和误报。

对比的「标定前」是现在机器人上跑的那套：
``wave_score > 4`` + 每帧 ``exp(-dt/2.0)`` 衰减（``pc/perception/gesture.py``）。
同一批序列喂进去，量出它到底要多久才触发、会不会漏。

用法::

    python src/calibrate_events.py            # 搜索 + 写回 YAML + 出报告
    python src/calibrate_events.py --eval-only  # 只评测 YAML 里的现值
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from episodes import make_episode  # noqa: E402
from gesture_common import CLASSES, skeleton_to_pixels  # noqa: E402
from gesture_classifier import MODEL_PATH, GestureClassifier  # noqa: E402
from gesture_events import GestureEventMachine  # noqa: E402
from synth import PoseBank  # noqa: E402

DATA = os.path.join(ROOT, "data", "synth")
REPORTS = os.path.join(ROOT, "reports")
CFG_PATH = os.path.join(ROOT, "configs", "gesture_events.yaml")

FPS = 30.0
DT = 1.0 / FPS

#: 目标误报率上限（次/分钟）。TECH-PLAN V7 要求「≤1 次/5min」= 0.2/min。
FA_MAX_PER_MIN = 0.2


def load_bank(split="val"):
    z = np.load(os.path.join(DATA, "bank_%s.npz" % split))
    return PoseBank(z["kp_rel"], z["conf"], z["crop"], z["label"], z["user"])


def build_suite(bank, seed=7, idle_minutes=3.0, unseen_minutes=2.0):
    """标定/验收序列集：每段都有已知的动作起点与期望的 onset 数。

    **必须有足够长的 idle 与「未知手势」负样本**：只用 20 s idle 标定，
    网格搜索会挑出「延迟最低但误报 3 次/分钟」的参数——第一版就是这样翻的车。
    """
    rng = np.random.default_rng(seed)
    users = bank.users
    eps = []

    def pick(k):
        return int(users[k % len(users)])

    k = 0
    # ---- 连续挥手：不同频率/幅度（1 Hz 慢挥是现在会翻车的那个） ----
    for freq in (0.5, 1.0, 1.5, 2.0, 2.5):
        for amp in (0.35, 0.75):
            plan = [("idle", 1.2), ("raise", 0.5, "open_palm"),
                    ("wave", 4.0), ("idle", 1.0)]
            eps.append(make_episode(bank, rng, pick(k), plan,
                                    name="wave_%.1fHz_amp%.2f" % (freq, amp),
                                    freq_hz=freq, amp=amp))
            k += 1
    # ---- 两次挥手，间隔不同（1 s 必须出 2 个 onset） ----
    for gap in (0.4, 0.6, 0.8, 1.0, 1.5):
        plan = [("idle", 1.0), ("raise", 0.4, "open_palm"), ("wave", 1.6),
                ("idle", gap), ("raise", 0.4, "open_palm"), ("wave", 1.6),
                ("idle", 1.0)]
        eps.append(make_episode(bank, rng, pick(k), plan,
                                name="two_wave_gap%.1f" % gap,
                                freq_hz=1.5, amp=0.55))
        k += 1
    # ---- 静态手势（举手不动）：应各出 1 个 onset ----
    for cls in ("open_palm", "thumbs_up", "pointing", "fist"):
        plan = [("idle", 1.0), ("raise", 0.5, cls), ("hold", 3.0, cls),
                ("idle", 1.0)]
        ep = make_episode(bank, rng, pick(k), plan, name="static_%s" % cls)
        eps.append(ep)
        k += 1
    # ---- 纯静止：长 idle（误报率的分母） ----
    n_idle = max(2, int(round(idle_minutes * 60 / 20.0)))
    for j in range(n_idle):
        eps.append(make_episode(bank, rng, pick(k), [("idle", 20.0)],
                                name="idle_%02d" % j))
        k += 1
    # ---- 没见过的手势：举着不动（也是负样本） ----
    up = os.path.join(DATA, "unseen_gestures.npz")
    if unseen_minutes > 0 and os.path.exists(up):
        zu = np.load(up)
        n = max(2, int(round(unseen_minutes * 60 / 10.0)))
        rows = rng.choice(len(zu["kp_rel"]), size=min(n, len(zu["kp_rel"])),
                          replace=False)
        for j, r in enumerate(rows):
            eps.append(_hold_episode(zu["kp_rel"][r], zu["conf"][r],
                                     zu["crop"][r], 10.0,
                                     name="unseen_%02d" % j))
    return eps


def _hold_episode(kp_rel, conf, crop, seconds, name="", fps=FPS):
    """把一帧姿势铺成 seconds 长的序列（模拟「举着不动」）。"""
    from episodes import Episode
    from synth import SW_NORM_SYNTH

    n = max(1, int(round(seconds * fps)))
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    k = kp_rel[None].repeat(n, axis=0)
    c = conf[None].repeat(n, axis=0)
    k = k + rng.normal(0, 0.012, k.shape).astype(np.float32)
    c = np.clip(c * rng.uniform(0.85, 1.0, c.shape), 0, 1)
    from gesture_common import build_features_from_rel
    return Episode(feat=build_features_from_rel(k, c, SW_NORM_SYNTH),
                   crop=crop[None].repeat(n, axis=0),
                   kp_rel=k, conf=c, name=name)


# ----------------------------------------------------------------------
def episode_metrics(ep, outs):
    """从事件机输出里量：onsets、首批 onset 相对动作起点的延迟。"""
    onsets = [i for i, o in enumerate(outs) if o.gesture_onset == 1]
    m = {"name": ep.name, "n_onsets": len(onsets),
         "duration_s": len(outs) * DT, "latency_ms": None, "miss": 0}

    starts = ep.starts_of("wave") + ep.starts_of("raise")
    if starts:
        # 只统计「动作开始之后」的第一个 onset：start 之前的不算
        s0 = min(starts)
        after = [i for i in onsets if i >= s0]
        if after:
            m["latency_ms"] = 1000.0 * (after[0] - s0) * DT
            m["first_onset_after_start"] = int(after[0])
        elif ep.name.startswith(("wave_", "static_")):
            m["miss"] = 1
        m["start_frame"] = int(s0)
    return m


def summarize(metrics, kind):
    sel = [m for m in metrics if m["name"].startswith(kind)]
    if not sel:
        return {}
    lat = [m["latency_ms"] for m in sel if m["latency_ms"] is not None]
    return {
        "n_episodes": len(sel),
        "n_onsets_mean": float(np.mean([m["n_onsets"] for m in sel])),
        "n_onsets": [m["n_onsets"] for m in sel],
        "miss_rate": float(np.mean([m["miss"] for m in sel])),
        "latency_mean_ms": float(np.mean(lat)) if lat else None,
        "latency_p90_ms": float(np.percentile(lat, 90)) if lat else None,
    }


def replay(probs, cfg):
    """用给定配置把一段概率序列跑一遍事件机。"""
    m = GestureEventMachine(cfg)
    return m.run_sequence(probs, dt=DT)


def score(cfg, suite):
    """跑全部序列，返回指标 + 一个可比较的代价。

    代价的第一优先级是**可行**（漏检/误报/单事件约束都达标），可行之后才比
    触发延迟。误报率达不到要求时，延迟再低也不算好。
    """
    metrics = []
    for ep, pr in suite:
        outs = replay(pr, cfg)
        metrics.append(episode_metrics(ep, outs))
    wave = summarize(metrics, "wave_")
    two = summarize(metrics, "two_wave_")
    static = summarize(metrics, "static_")
    idle = summarize(metrics, "idle_")
    unseen = summarize(metrics, "unseen_")

    def _fa(prefix):
        sel = [m for m in metrics if m["name"].startswith(prefix)]
        mins = sum(m["duration_s"] for m in sel) / 60.0
        return sum(m["n_onsets"] for m in sel), mins

    n_idle, min_idle = _fa("idle_")
    n_uns, min_uns = _fa("unseen_")
    fa_idle = n_idle / max(min_idle, 1e-6)
    fa_unseen = n_uns / max(min_uns, 1e-6)
    fa_total = (n_idle + n_uns) / max(min_idle + min_uns, 1e-6)

    lat = wave.get("latency_mean_ms")
    lat = 1e4 if lat is None else lat
    miss = wave.get("miss_rate", 1.0)
    static_ok = float(np.mean([m["n_onsets"] == 1
                               for m in metrics
                               if m["name"].startswith("static_")]))
    split_ok = float(np.mean([m["n_onsets"] == 2
                              for m in metrics
                              if m["name"].startswith("two_wave_")]))
    cont_ok = float(np.mean([m["n_onsets"] == 1 for m in metrics
                             if m["name"].startswith("wave_")]))
    feasible = (miss <= 0.05 and cont_ok >= 0.90 and static_ok >= 0.75
                and split_ok >= 0.80 and fa_total <= FA_MAX_PER_MIN)
    cost = (lat if feasible else 1e6 + lat)
    cost += 5000.0 * miss + 1000.0 * (1 - cont_ok) + 500.0 * (1 - static_ok) \
        + 300.0 * (1 - split_ok) + 500.0 * max(0.0, fa_total
                                               - FA_MAX_PER_MIN)
    return {"wave": wave, "two_wave": two, "static": static, "idle": idle,
            "unseen": unseen, "fa_per_min": fa_total,
            "fa_idle_per_min": fa_idle, "fa_unseen_per_min": fa_unseen,
            "fa_idle_minutes": min_idle, "fa_unseen_minutes": min_uns,
            "cont_single_onset_rate": cont_ok,
            "static_one_onset_rate": static_ok,
            "two_wave_split_rate": split_ok,
            "feasible": bool(feasible),
            "cost": cost, "metrics": metrics}


def old_algorithm(episodes):
    """现在的实现（wave_score > 4 + 2 s 衰减）在同一批序列上的表现。"""
    sys.path.insert(0, r"D:\robot_work\pc\perception")
    from gesture import GestureEstimator  # 只读引用，不改

    rows = []
    for ep in episodes:
        est = GestureEstimator()
        first = None
        n_hi = 0
        for i in range(len(ep.feat)):
            kp_px = skeleton_to_pixels(ep.kp_rel[i], 60.0, (120.0, 120.0))
            g = est.update(kp_px, ep.conf[i], dt=DT)
            if g.waving:
                n_hi += 1
                if first is None:
                    first = i
        starts = ep.starts_of("wave") + ep.starts_of("raise")
        s0 = min(starts) if starts else 0
        lat = None
        if first is not None and first >= s0:
            lat = 1000.0 * (first - s0) * DT
        rows.append({"name": ep.name, "n_trigger_frames": n_hi,
                     "latency_ms": lat,
                     "miss": int(first is None and ep.name.startswith(
                         ("wave_", "static_"))),
                     "duration_s": len(ep.feat) * DT})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=MODEL_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    bank = load_bank("val")
    episodes = build_suite(bank)
    print("[cal] %d 条标定序列" % len(episodes), flush=True)

    clf = GestureClassifier.load(args.ckpt, device=args.device)
    suite = []
    for ep in episodes:
        clf.reset()
        pr = clf.run_episode(ep.feat, ep.crop, dt=DT)
        suite.append((ep, pr))
    print("[cal] 分类概率已缓存", flush=True)

    # ---- 标定前：老算法 ----
    old = old_algorithm(episodes)
    old_wave = [r for r in old if r["name"].startswith("wave_")]
    old_static = [r for r in old if r["name"].startswith("static_")]
    old_idle = [r for r in old if r["name"].startswith("idle_")]
    old_lat = [r["latency_ms"] for r in old_wave if r["latency_ms"] is not None]
    old_summary = {
        "wave_latency_mean_ms": float(np.mean(old_lat)) if old_lat else None,
        "wave_latency_by_freq": {
            r["name"]: r["latency_ms"] for r in old_wave},
        "wave_miss_rate": float(np.mean([r["miss"] for r in old_wave])),
        "static_trigger_rate": float(np.mean(
            [r["n_trigger_frames"] > 0 for r in old_static])),
        "idle_false_alarms": int(sum(r["n_trigger_frames"] > 0
                                     for r in old_idle)),
        "idle_minutes": sum(r["duration_s"] for r in old_idle) / 60.0,
    }
    print("[cal] 老算法：wave 平均延迟 %s ms，漏检 %.0f%%，idle 误报 %d"
          % (old_summary["wave_latency_mean_ms"],
             100 * old_summary["wave_miss_rate"],
             old_summary["idle_false_alarms"]), flush=True)

    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        cur = yaml.safe_load(fh)

    if args.eval_only:
        best_cfg, best = cur, score(cur, suite)
    else:
        best, best_cfg = None, None
        for enter in (0.55, 0.65, 0.75, 0.85):
            for pending in (0.10, 0.15, 0.20, 0.25, 0.30):
                for release in (0.15, 0.20, 0.25, 0.30):
                    cfg = _variant(cur, enter, pending, release)
                    sc = score(cfg, suite)
                    print("  enter %.2f pending %.2f release %.2f → "
                          "lat %6.1f ms  miss %.2f  cont %.2f  split %.2f  "
                          "静态 %.2f  fa %.2f/min%s  cost %.0f"
                          % (enter, pending, release,
                             sc["wave"].get("latency_mean_ms") or -1,
                             sc["wave"]["miss_rate"],
                             sc["cont_single_onset_rate"],
                             sc["two_wave_split_rate"],
                             sc["static_one_onset_rate"], sc["fa_per_min"],
                             "  ✓可行" if sc["feasible"] else "",
                             sc["cost"]), flush=True)
                    if best is None or sc["cost"] < best["cost"]:
                        best, best_cfg = sc, cfg
        # 写回 YAML（只改标定出来的量，其余保持人手可读）
        out = dict(cur)
        out["enter_thresh"] = float(best_cfg["enter_thresh"])
        out["exit_thresh"] = float(best_cfg["exit_thresh"])
        out["timing_s"] = dict(cur["timing_s"])
        out["timing_s"]["pending_hold"] = float(
            best_cfg["timing_s"]["pending_hold"])
        out["timing_s"]["release"] = float(best_cfg["timing_s"]["release"])
        # per_class 的按类覆盖优先级高于全局值，必须一起写，否则「标定出来的
        # 配置」和「文件里的配置」是两套数（第一版就踩了这个坑）。
        per = {}
        for k in (cur.get("per_class") or {}):
            per[k] = {"enter": float(best_cfg["enter_thresh"]),
                      "exit": float(best_cfg["exit_thresh"])}
        if per:
            out["per_class"] = per
        with open(CFG_PATH, "w", encoding="utf-8") as fh:
            yaml.safe_dump(out, fh, allow_unicode=True, sort_keys=False)
        print("[cal] 已写回 %s" % CFG_PATH, flush=True)
        # 写完再读回来评一遍：确保「报告的指标」就是「文件里的配置」的指标
        with open(CFG_PATH, "r", encoding="utf-8") as fh:
            written = yaml.safe_load(fh)
        recheck = score(written, suite)
        print("[cal] 读回校验：lat %.0f ms  cont %.2f  split %.2f  fa %.2f/min"
              % (recheck["wave"].get("latency_mean_ms") or -1,
                 recheck["cont_single_onset_rate"],
                 recheck["two_wave_split_rate"], recheck["fa_per_min"]),
              flush=True)
        best_cfg = written

    after = score(best_cfg, suite)
    detail = {r["name"]: r["latency_ms"] for r in after["metrics"]}
    old_by_name = {r["name"]: r["latency_ms"] for r in old}

    os.makedirs(REPORTS, exist_ok=True)
    out = {
        "n_episodes": len(episodes),
        "before": old_summary,
        "after": {k: v for k, v in after.items() if k != "metrics"},
        "per_episode_latency_ms_after": detail,
        "per_episode_latency_ms_before": old_by_name,
        "config_after": {k: v for k, v in best_cfg.items()
                         if k in ("enter_thresh", "exit_thresh")},
        "timing_after": best_cfg["timing_s"],
        "old_rows": old,
    }
    with open(os.path.join(REPORTS, "calibration.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("[cal] wave 延迟 老 %.0f ms → 新 %.0f ms；漏检 %.0f%% → %.0f%%；"
          "连续挥手单 onset %.0f%%；双挥手拆分 %.0f%%；"
          "误报 静止 %.2f/min、未见手势 %.2f/min（合计 %.2f/min，%s）"
          % (old_summary["wave_latency_mean_ms"] or -1,
             after["wave"].get("latency_mean_ms") or -1,
             100 * old_summary["wave_miss_rate"],
             100 * after["wave"]["miss_rate"],
             100 * after["cont_single_onset_rate"],
             100 * after["two_wave_split_rate"],
             after["fa_idle_per_min"], after["fa_unseen_per_min"],
             after["fa_per_min"],
             "达标" if after["feasible"] else "未达标"), flush=True)


def _variant(cur, enter, pending, release):
    """在现有配置上改三个标定参数，其余不动。"""
    cfg = dict(cur)
    cfg["enter_thresh"] = float(enter)
    cfg["exit_thresh"] = round(float(enter) - 0.23, 3)   # 迟滞宽度固定 0.23
    cfg["timing_s"] = dict(cur["timing_s"])
    cfg["timing_s"]["pending_hold"] = float(pending)
    cfg["timing_s"]["release"] = float(release)
    per = {}
    for k, v in (cur.get("per_class") or {}).items():
        v = dict(v or {})
        if "enter" in v:
            v["enter"] = float(enter)
        if "exit" in v:
            v["exit"] = round(float(enter) - 0.23, 3)
        per[k] = v
    cfg["per_class"] = per
    return cfg


if __name__ == "__main__":
    main()
