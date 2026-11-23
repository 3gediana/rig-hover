# -*- coding: utf-8 -*-
"""knock_trigger_resim_v23.py — 用 v23 解剖 dump 的 imp/sem 曲线离线重仿真触发层。

目的：不动模型、不重推理，只在已记录的 (p_impulse, p_double) 曲线上重放
`peak -> pair -> sem_window -> cooldown` 判定，量化"每种触发改进"在两条真实
take 上的 GT 命中 / 歧义（其它标注段内）/ 真FP 得失——证据先行，不搞盲调参。

复刻 DoubleKnockDetector.step 的判定（与部署代码逐行对齐）：
  is_peak(t) = imp[t-1] >= imp_thr and imp[t-1] >= imp[t-2] and imp[t-1] > imp[t]
  pair_ok    = 第二峰与前一峰间距 in [pair_min_s, pair_max_s]
  sem_ok     = max(sem 最近 sem_win 步) >= sem_thr      （部署 sem_win = 8 步）
  cooldown   = t - last_fire >= cooldown_s

变体：
  V0 基线（部署值 0.35/0.32/[0.20,1.45]/cd2.0/sem_win8）
  V1 sem 闩存：sem>=thr 后闩存 latch_s 秒，配对成功时闩存有效即可（修时序错位）
  V2 pair_min 放宽到 0.10（修快敲）
  V3 单峰+强语义：sem>=0.60 且窗内仅 1 个 imp 峰也可触发（修单峰漏）
  V4 imp_thr 0.30（修阈下毫厘）
  V5 组合：V1+V2+V4（闩存 + 快敲 + 低阈；不做 V3 单峰放行——风险最高）
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
TAKES = [
    ("robot_20261107_214950_1f30d1", 20.0),
    ("robot_20261115_151733_cb8455", 15.0),
    ("robot_20261107_212119_67f0c1", 20.0),
    ("robot_20261115_150045_65a54d", 15.0),
]


def load_gt(take: str) -> tuple[list, list]:
    ann = json.loads((REPO / "dataset" / "takes" / take / "annotation.json").read_text(encoding="utf-8"))
    kn = [(x["start_s"], x["end_s"]) for x in ann["annotations"] if x["label"] == "knock"]
    oth = [(x["start_s"], x["end_s"], x["label"]) for x in ann["annotations"] if x["label"] != "knock"]
    return sorted(kn), sorted(oth)


def resim(rows, *, imp_thr, sem_thr, pair_min, pair_max, cooldown, sem_win_steps,
          sem_latch_s=0.0, single_peak_sem=None, single_lookback_s=1.5,
          sem_solo_thr=None, sem_solo_hold_steps=1):
    """返回触发时刻列表（秒）。

    single_peak_sem：若非 None，加开"延迟语义确认单峰"通道——
    当 sem_raw 越过 single_peak_sem 的上升沿处回看 single_lookback_s 秒内
    存在 imp 峰（未成对）即补触发（修复"第二敲冲击太弱不成峰"漏检）。
    sem_solo_thr：若非 None，加开"纯语义高分通道"——sem_raw 上升越过
    sem_solo_thr 即触发，不要求任何 imp 峰（67f0c1 gt9 型漏检：
    sem=0.717 但整窗零 imp 峰，现有两通道全锁死）。风险：语义单点
    误报直通——阈值须明显高于一切负样本 sem 峰值。
    """
    fires, peaks, last_fire = [], [], -1e9
    latched_until = -1e9
    sem_was_high = False
    sem_was_solo = False
    solo_run = 0
    n = len(rows)
    for i in range(1, n - 1):
        t, imp, sem = rows[i]["t"], rows[i]["imp"], rows[i]["sem_raw"]
        if sem >= sem_thr:
            latched_until = t + sem_latch_s
        # --- 延迟语义确认单峰通道（先看，再算 imp 峰） ---
        if single_peak_sem is not None:
            if sem >= single_peak_sem and not sem_was_high:
                recent_pk = [p for p in peaks if t - p <= single_lookback_s]
                if recent_pk and t - last_fire >= cooldown:
                    last_fire = t
                    fires.append(t)
            sem_was_high = sem >= single_peak_sem
        # --- 纯语义高分通道（零 imp 峰样本的兜底；hold_steps>1 要求持续高分防尖刺） ---
        if sem_solo_thr is not None:
            if sem >= sem_solo_thr:
                solo_run = solo_run + 1
            else:
                solo_run = 0
            if solo_run >= sem_solo_hold_steps and not sem_was_solo and t - last_fire >= cooldown:
                last_fire = t
                fires.append(t)
            sem_was_solo = solo_run >= sem_solo_hold_steps
        prev, nxt = rows[i - 1], rows[i + 1]
        if not (prev["imp"] >= imp_thr and prev["imp"] >= rows[i - 2]["imp"] and prev["imp"] > imp):
            continue
        tpk = t  # 部署将峰记在当前检测步（now_t），非峰帧本身
        while peaks and tpk - peaks[0] > pair_max + 0.5:
            peaks.pop(0)
        peaks.append(tpk)
        pair_ok = len(peaks) >= 2 and pair_min <= (peaks[-1] - peaks[-2]) <= pair_max
        if not pair_ok:
            continue
        # sem 判定：部署=最近 sem_win 步原始 sem 的 max；闩存=当前在闩存期即可
        sem_win = max(r["sem_raw"] for r in rows[max(0, i - sem_win_steps + 1):i + 1])
        sem_ok = (sem_win >= sem_thr) or (sem_latch_s > 0 and tpk <= latched_until)
        if not sem_ok or tpk - last_fire < cooldown:
            continue
        last_fire = tpk
        fires.append(tpk)
    return fires


def score(fires, gt_kn, gt_oth):
    hits = amb = fp = 0
    matched = set()
    for t in fires:
        gi = next((j for j, (a, b) in enumerate(gt_kn) if a - 0.6 <= t <= b + 0.6), None)
        if gi is not None:
            hits += 1
            matched.add(gi)
        elif any(a <= t <= b for a, b, _ in gt_oth):
            amb += 1
        else:
            fp += 1
    return hits, amb, fp, len(gt_kn) - len(matched)


VARIANTS = {
    "V0-deploy": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8),
    "V1-semLatch1.5": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=1.5),
    "V2-pairMin0.10": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.10, pair_max=1.45, cooldown=2.0, sem_win_steps=8),
    "V7-semConfirm+single": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, single_peak_sem=0.60, single_lookback_s=1.5),
    "V8-latch2+V7": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5),
    "V9-V8+imp.30+fast": dict(imp_thr=0.30, sem_thr=0.32, pair_min=0.10, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5),
    "V4-imp0.30": dict(imp_thr=0.30, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8),
    "V5-latch+fast+imp.30": dict(imp_thr=0.30, sem_thr=0.32, pair_min=0.10, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=1.5),
    "V6-semLatch2.0": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0),
    "V10a-V8+solo.70": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5, sem_solo_thr=0.70),
    "V10b-V8+solo.80": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5, sem_solo_thr=0.80),
    "V10c-solo.70h2": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5, sem_solo_thr=0.70, sem_solo_hold_steps=2),
    "V10d-solo.70h3": dict(imp_thr=0.35, sem_thr=0.32, pair_min=0.20, pair_max=1.45, cooldown=2.0, sem_win_steps=8, sem_latch_s=2.0, single_peak_sem=0.60, single_lookback_s=1.5, sem_solo_thr=0.70, sem_solo_hold_steps=3),
}


def main() -> None:
    curve_dir = REPO / "runs_end_side" / "knock_autopsy"
    print(f"{'variant':28s} | {'take':16s} | hits amb trueFP miss | fires")
    curve_cache: dict[str, list] = {}
    for take, _fps in TAKES:
        fp_ = curve_dir / f"{take}_trigger_curves.json"
        if not fp_.exists():
            fp_ = curve_dir / f"{take[-6:]}_v8_curves.json"
        if fp_.exists():
            curve_cache[take] = json.loads(fp_.read_text(encoding="utf-8"))["rows"]
    for name, kw in VARIANTS.items():
        for take, _fps in TAKES:
            if take not in curve_cache:
                continue
            rows = curve_cache[take]
            gt_kn, gt_oth = load_gt(take)
            fires = resim(rows, **kw)
            h, a, fp, miss = score(fires, gt_kn, gt_oth)
            print(f"{name:28s} | {take[:16]:16s} | {h:4d} {a:4d} {fp:5d} {miss:4d} | {[round(t,1) for t in fires]}")


if __name__ == "__main__":
    main()
