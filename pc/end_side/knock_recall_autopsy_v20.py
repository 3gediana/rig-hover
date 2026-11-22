# -*- coding: utf-8 -*-
"""knock_recall_autopsy_v20.py — 双敲漏检解剖台（v20）。

背景：ensemble_v5 评测显示最强单模型 physnet_v11（部署阈值 in-sample 81/89、1 FP），
8 个漏检全部集中在 2026-11-15 的两个 take（65a54d 6miss、cb8455 2miss）。
本脚本不重训，只做**分数层解剖**：

  1) 对 6 个 take 跑 physnet_v11 因果滑窗，dump 逐窗 (impulse, class) 分数曲线；
  2) 对每个漏检 GT 区间解剖死因：
       A. 区间内连 2 个 imp 峰都凑不齐（峰数不足）
       B. 峰够了但 semantic < double_thr（语义否决）
       C. 峰间距越界（<0.20s 或 >1.45s）
       D. 被 cooldown 吞掉
  3) 测"迟滞触发器"（第一峰 thr、第二峰仅需 0.6*thr 且语义达标）能否召回近阈 miss 且不增 FP；
  4) 输出 per-take 阈值-召回-FP 曲线，给出是否有"免费召回空间"的诚实结论。

产物：runs_end_side/knock_autopsy_v20/autopsy_report.json + 分数曲线 npz。
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

END_SIDE = pathlib.Path(__file__).resolve().parent
REPO = END_SIDE.parents[1]
sys.path.insert(0, str(END_SIDE))
sys.path.insert(0, str(REPO / "pc" / "train"))

from ensemble_knock_v5 import CKPTS, causal_scores, eval_scores, load_model  # noqa: E402
from rigorous_audit_knock_ablation import eval_strict_1to1_events  # noqa: E402
from train_double_knock_3d_v2_extreme import load_v2_dataset  # noqa: E402

OUT_DIR = REPO / "runs_end_side" / "knock_autopsy_v20"
WINDOW = 48


def find_peaks(ic: np.ndarray, fps: float, imp_thr: float, smooth_frames: int = 3) -> list[tuple[float, int, float]]:
    """复刻触发器里的冲击峰提取（单峰判定，不做配对/语义/冷却）。"""
    from collections import deque
    imp_hist: deque[float] = deque(maxlen=max(1, smooth_frames))
    sm_hist: deque[float] = deque(maxlen=max(5, 2 * smooth_frames))
    peaks: list[tuple[float, int, float]] = []
    for i in range(len(ic)):
        p = float(ic[i])
        imp_hist.append(p)
        sm = 0.65 * p + 0.35 * float(np.mean(imp_hist))
        sm_hist.append(sm)
        if len(sm_hist) < 3:
            continue
        sh = list(sm_hist)
        prev, cur = sh[-2], sh[-1]
        if (prev >= imp_thr) and (prev >= sh[-3]) and (prev > cur):
            peaks.append(((i - 1) / fps, i - 1, prev))
    return peaks


def hysteresis_triggers(ic, cc, fps, imp_thr, hys_ratio, double_thr, smooth_frames=3,
                        pair_min_s=0.20, pair_max_s=1.45, cooldown_s=1.8):
    """迟滞触发：第二峰只需 hys_ratio*imp_thr 且窗内语义峰值达标。"""
    hi = find_peaks(ic, fps, imp_thr, smooth_frames)
    lo = find_peaks(ic, fps, imp_thr * hys_ratio, smooth_frames)
    lo_set = {t[1] for t in lo}
    hi_set = {t[1] for t in hi}
    cand = sorted(set(t for t in hi) | set(t for t in lo))
    from collections import deque
    sem_hist: deque[float] = deque(maxlen=8)
    last_t = -1e9
    out = []
    # 语义按帧推进需要逐帧 p_dbl —— 直接在峰附近取局部窗最大
    def sem_at(idx):
        w = cc[max(0, idx - 2): idx + 3]
        return float(np.max(w[:, 2] + 0.45 * w[:, 3])) if len(w) else 0.0
    for i, (t_now, idx, amp) in enumerate(cand):
        if i == 0:
            continue
        t_prev, idx_prev, amp_prev = cand[i - 1]
        if not (pair_min_s <= t_now - t_prev <= pair_max_s):
            continue
        # 迟滞规则：首峰须过 imp_thr；次峰可低至 hys_ratio*imp_thr
        if idx_prev not in hi_set and amp_prev < imp_thr:
            continue
        if idx not in hi_set and amp < imp_thr * hys_ratio:
            continue
        sem = max(sem_at(idx), sem_at(idx_prev))
        if sem >= double_thr and (t_now - last_t >= cooldown_s):
            out.append({"first_s": round(t_prev, 3), "second_s": round(t_now, 3),
                        "first_idx": idx_prev, "second_idx": idx, "score": round(sem, 3)})
            last_t = t_now
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v20] device={device}", flush=True)

    dataset = load_v2_dataset(window=WINDOW)
    take_ids = sorted(dataset.keys())
    name, (kind, path) = "physnet_v11", CKPTS["physnet_v11"]
    ck = torch.load(path, map_location="cpu")
    thr = dict(ck.get("calib_stream") or {"smooth_frames": 3, "imp_thr": 0.35, "double_thr": 0.35})
    model = load_model(kind, path, device)
    print(f"[v20] {name} thr={thr}", flush=True)

    report: dict[str, Any] = {"model": name, "thr_used": thr, "takes": {}}
    base_rows, hys_rows = [], []
    t0 = time.perf_counter()
    for tid in take_ids:
        d = dataset[tid]
        ic, cc = causal_scores(kind, model, d, device)
        np.savez_compressed(OUT_DIR / f"{tid}_scores.npz", ic=ic, cc=cc)

        # 基线：部署阈值严格评测
        base = eval_scores(ic, cc, d, thr)
        base_rows.append(base)

        # 解剖：每个 GT 区间内有多少 imp 峰、语义峰值多少
        fps = d["fps"]
        imp_thr, dbl_thr = float(thr["imp_thr"]), float(thr["double_thr"])
        pk_full = find_peaks(ic, fps, imp_thr, int(thr["smooth_frames"]))
        pk_soft = find_peaks(ic, fps, imp_thr * 0.6, int(thr["smooth_frames"]))
        intervals = d["intervals"]
        autopsy = []
        missed = 0
        for it in intervals:
            if it.label != "knock":
                continue
            s0, s1 = float(it.start_s), float(it.end_s)
            pad = 0.6
            in_iv = [p for p in pk_full if s0 - pad <= p[0] <= s1 + pad]
            in_iv_soft = [p for p in pk_soft if s0 - pad <= p[0] <= s1 + pad]
            i0f, i1f = int((s0 - pad) * fps), int((s1 + pad) * fps)
            sem_max = float(np.max(cc[max(0, i0f):i1f][:, 2] + 0.45 * cc[max(0, i0f):i1f][:, 3])) if i1f > i0f else 0.0
            status = "hit"
            # 简单复判：若区间内没有合法配对+语义 → 漏检候选
            if len(in_iv) < 2:
                status, missed = ("miss_no_pair", missed + 1) if len(in_iv_soft) >= 2 else ("miss_no_peaks", missed + 1)
            elif sem_max < dbl_thr:
                status, missed = "miss_sem_veto", missed + 1
            autopsy.append({
                "gt": [s0, s1], "status": status,
                "n_peaks_full": len(in_iv), "n_peaks_soft": len(in_iv_soft),
                "sem_max": round(sem_max, 3), "double_thr": dbl_thr,
            })
        # 迟滞触发器评测
        hys_tr = hysteresis_triggers(ic, cc, fps, imp_thr, 0.6, dbl_thr, int(thr["smooth_frames"]))
        hys = eval_strict_1to1_events(hys_tr, d["labels_full"], intervals, fps, d["duration_s"])
        hys_rows.append(hys)

        report["takes"][tid] = {"base": base, "hysteresis_0.6": hys, "autopsy": autopsy}
        print(f"[v20] {tid}: base hit={base['hit']}/{base['total_gt']} fp={base['fp']} | "
              f"hys hit={hys['hit']}/{hys['total_gt']} fp={hys['fp']}", flush=True)

    def agg(rows):
        hit = sum(r["hit"] for r in rows)
        gt = sum(r["total_gt"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fas = sum(r["fa_per_min"] for r in rows) / max(len(rows), 1)
        prec = hit / max(hit + fp, 1)
        f1 = 2 * prec * (hit / max(gt, 1)) / max(prec + hit / max(gt, 1), 1e-9)
        return {"hit": hit, "gt": gt, "fp": fp, "f1": round(f1, 4), "fa_min": round(fas, 3)}

    report["aggregate"] = {"base": agg(base_rows), "hysteresis_0.6": agg(hys_rows),
                           "elapsed_s": round(time.perf_counter() - t0, 1)}
    (OUT_DIR / "autopsy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v20] aggregate: {json.dumps(report['aggregate'], ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
