"""validate_knock.py — 用手上真机录的 knock 段验证规则检测器，而不是靠猜。

要回答的问题只有一个：**腕部高度曲线上，"敲一下"这个事件能不能用纯规则稳定
地提出来？** 如果能，"两次"就是个计数器；如果连"一次"都提不准，那才轮到
去讨论用网络学——而且要先知道规则错在哪。

输入是 capture_studio 录的会话（evidence/sessions/*/frames），每段都有
meta.json 记录操作者被要求做什么。

输出：
* 每段的 wrist_y_norm 轨迹（文本 + 一张曲线图）
* 现有 KnockParams 下 detector 报了几次命中、几次 fired
* 手动峰检测（scipy find_peaks）作为对照，看规则漏在哪
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PC = Path(r"D:\robot_work\pc")
sys.path.insert(0, str(PC / "perception"))
SESS = Path(r"D:\robot_work\evidence\sessions")
OUT = Path(r"D:\robot_work\evidence\knock_validation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="knock_d0.5_front_night_182129")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = SESS / a.session
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    frames = sorted((d / "frames").glob("*.jpg"))[:: a.every]
    fps = meta["frames"] / max(meta["seconds_actual"], 1e-6)
    print(f"session {a.session}  task={meta['task']}  "
          f"{len(frames)} frames @ {fps:.1f} fps")
    print(f"note: {meta.get('note')!r}")

    from run_perception import PerceptionPipeline  # noqa: PLC0415
    from scene_signals import KnockDetector, KnockParams, hand_signals  # noqa: PLC0415

    pipe = PerceptionPipeline(device="cuda:0")
    ys, confs, xs = [], [], []
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        if img is None:
            ys.append(np.nan)
            confs.append(0.0)
            xs.append(np.nan)
            continue
        out = pipe(img)
        kp = np.asarray(out.kp_xy)
        kc = np.asarray(out.kp_conf)
        # 取置信度更高的那只手腕
        best, bc = None, 0.0
        for ki in (9, 10):
            if ki < len(kc) and float(kc[ki]) > bc:
                best, bc = kp[ki], float(kc[ki])
        if best is None or bc < 0.2:
            ys.append(np.nan)
            confs.append(bc)
            xs.append(np.nan)
        else:
            # 直接用生产代码算好的 wrist_y_norm，避免这里的归一化定义和
            # 部署时漂移（那会让"规则在回放里能用、上机不能用"）。
            ys.append(float(out.wrist_y_norm))
            confs.append(bc)
            xs.append(float(best[0]))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(frames)}", flush=True)

    y = np.array(ys, float)
    valid = np.isfinite(y)
    print(f"\n腕部可用帧: {valid.sum()}/{len(y)} = {100*valid.mean():.1f}%")
    if valid.sum() < 10:
        print("[FAIL] 可用帧太少，这个会话没法验证")
        return 2
    print(f"wrist_y_norm  min={np.nanmin(y):.3f} max={np.nanmax(y):.3f} "
          f"mean={np.nanmean(y):.3f} std={np.nanstd(y):.3f}")

    # ---- 现有规则检测器 ----
    p = KnockParams()
    print(f"\nKnockParams: y_norm_min={p.y_norm_min} depth={p.depth} "
          f"rebound={p.rebound} window_s={p.window_s} min_gap_s={p.min_gap_s} "
          f"refractory_s={p.refractory_s}")
    det = KnockDetector(p)
    hits = fires = 0
    phases = []
    for i in range(len(y)):
        t = i / fps
        o = det.update(t, float(y[i]) if valid[i] else 0.0,
                       float(xs[i]) if valid[i] else 0.0,
                       float(xs[i]) if valid[i] else 0.0,
                       valid=int(valid[i]))
        phases.append(o.phase)
        if o.hits > hits:
            hits = o.hits
        if o.fired:
            fires += 1
    print(f"detector: hits={hits}  fired={fires}")

    # ---- 对照：直接对曲线做峰检测 ----
    try:
        from scipy.signal import find_peaks  # noqa: PLC0415
        yy = np.where(valid, y, np.nanmedian(y))
        # 下探 = y_norm 的局部极大（手放到桌面高度时归一化值最大）
        peaks, props = find_peaks(yy, prominence=0.05, distance=max(2, int(fps * 0.15)))
        print(f"find_peaks(prominence=0.05): {len(peaks)} 个峰 "
              f"@ t={[round(pk / fps, 2) for pk in peaks]}")
        if len(peaks):
            prom = props["prominences"]
            print(f"  prominence: min={prom.min():.3f} median={np.median(prom):.3f} "
                  f"max={prom.max():.3f}")
            if len(peaks) > 1:
                gaps = np.diff(peaks) / fps
                print(f"  峰间距(s): {[round(g, 2) for g in gaps]}")
    except ImportError:
        print("(no scipy, skip find_peaks)")

    # ---- 存轨迹 + 画图 ----
    np.savetxt(OUT / f"{a.session}_wrist.csv",
               np.c_[np.arange(len(y)) / fps, y, confs, valid.astype(int)],
               delimiter=",", header="t_s,wrist_y_norm,wrist_conf,valid",
               comments="", fmt="%.4f")

    H, W = 900, 1400
    canvas = np.full((H, W, 3), 18, np.uint8)
    y0, y1 = 60, H - 80
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = max(0.02, (hi - lo) * 0.1)
    lo, hi = lo - pad, hi + pad
    cv2.rectangle(canvas, (70, y0), (W - 40, y1), (60, 60, 60), 1)

    def px(i, v):
        xx = 70 + int((W - 110) * i / max(len(y) - 1, 1))
        yy = y1 - int((y1 - y0) * (v - lo) / max(hi - lo, 1e-9))
        return xx, yy

    # y_norm_min 参考线
    if lo < p.y_norm_min < hi:
        cv2.line(canvas, (70, px(0, p.y_norm_min)[1]),
                 (W - 40, px(0, p.y_norm_min)[1]), (80, 160, 255), 1)
        cv2.putText(canvas, f"y_norm_min={p.y_norm_min}", (76, px(0, p.y_norm_min)[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 160, 255), 1)
    prev = None
    for i, v in enumerate(y):
        if not valid[i]:
            prev = None
            continue
        cur = px(i, v)
        if prev is not None:
            cv2.line(canvas, prev, cur, (90, 220, 120), 2)
        prev = cur
    cv2.putText(canvas, f"{a.session}  task={meta['task']}  wrist_y_norm vs time "
                        f"({len(y)} frames @ {fps:.1f} fps)",
                (76, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
    cv2.putText(canvas, f"{valid.sum()}/{len(y)} frames with a visible wrist",
                (76, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
    cv2.imwrite(str(OUT / f"{a.session}_wrist.png"), canvas)
    print(f"\nwrote {OUT / (a.session + '_wrist.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
