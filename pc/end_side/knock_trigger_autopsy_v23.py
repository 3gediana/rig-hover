# -*- coding: utf-8 -*-
"""knock_trigger_autopsy_v23.py — 双敲触发链离线解剖：定位高语义漏检死在哪一级。

起因：E2E 回放 take 1f30d1 的 7 个 GT miss 里 5 个 `p_double` 高达 0.49-0.84
（远超阈值 0.32）却没触发——不是"信号弱"，是冲击峰配对/时序错位/冷却某级吃掉。
本脚本复刻 app 内的流式调用（相同 ckpt / 相同内容时间戳），逐帧 dump
`p_impulse`(smoothed)、`peak`、`p_double`(sem win-max)、`fired`，并给出逐 GT 死因表。

用法：
    python pc/end_side/knock_trigger_autopsy_v23.py --take dataset/takes/robot_20261107_214950_1f30d1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(HERE), str(REPO / "pc" / "perception"), str(REPO / "pc" / "train")):
    if p not in sys.path:
        sys.path.insert(0, p)

from scenario_v2 import MODE_GAZE_FIST  # noqa: E402
from train_double_knock import DoubleKnockDetector, DoubleKnockNet  # noqa: E402

KNOCK_DEPLOY_CKPT = REPO / "runs_knock" / "double_knock_fpsaug2_20261115" / "double_knock_deploy.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    take = pathlib.Path(args.take)
    video = take / "take.mp4"
    ann = json.loads((take / "annotation.json").read_text(encoding="utf-8"))
    kn = sorted([x for x in ann["annotations"] if x["label"] == "knock"], key=lambda x: x["start_s"])

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0) or 20.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kb = torch.load(KNOCK_DEPLOY_CKPT, map_location=dev, weights_only=False)
    km = DoubleKnockNet(in_ch=2, kin_dim=8, hidden_dim=64, num_classes=4).to(dev).eval()
    km.load_state_dict(kb["state_dict"])
    det = DoubleKnockDetector(
        model=km, device=dev,
        window=int(kb.get("window", 48)), size=int(kb.get("size", 96)),
        imp_threshold=0.35, double_threshold=0.32, gate_state=MODE_GAZE_FIST,
        sem_latch_s=2.0, single_confirm_thr=0.60, single_lookback_s=1.5,
    )

    rows: list[dict] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        r = det.step(g, wrist_box=None, scenario_state=None, timestamp=idx / fps)
        peaks = list(det.peak_times)
        rows.append({
            "f": idx, "t": round(idx / fps, 3),
            "imp": round(r["p_impulse"], 3), "sem": round(r["p_double"], 3),
            "sem_raw": round(float(det.sem_hist[-1]), 3) if det.sem_hist else 0.0,
            "peak": int(r["peak"]), "fired": int(r["fired"]),
            "n_peaks": len(peaks),
            "last_gap": round(peaks[-1] - peaks[-2], 2) if len(peaks) >= 2 else None,
        })
        idx += 1
    cap.release()

    # ---- 逐 GT 解剖：区间内 imp 峰数 / sem 峰值 / 最近一次可配对间距 / 冷却阴影 ----
    print(f"\n=== take {take.name} | fps={fps:.1f} | frames={idx} | GTs={len(kn)} ===")
    fires = [r["t"] for r in rows if r["fired"]]
    print("fires:", [round(t, 2) for t in fires])
    for i, x in enumerate(kn):
        s0, e0 = x["start_s"], x["end_s"]
        seg = [r for r in rows if s0 - 0.5 <= r["t"] <= e0 + 1.0]
        max_sem = max(r["sem"] for r in seg)
        imp_peaks = [r["t"] for r in seg if r["peak"]]
        max_imp = max(r["imp"] for r in seg)
        # 与区间内每个 imp 峰配对的前一峰间距分布
        gaps = [r["last_gap"] for r in seg if r["peak"] and r["last_gap"] is not None]
        # sem 峰值时刻 vs 最后 imp 峰时刻 的错位
        t_sem = seg[int(np.argmax([r["sem"] for r in seg]))]["t"] if seg else 0
        t_lastpk = imp_peaks[-1] if imp_peaks else None
        fired_in = any(r["fired"] for r in seg)
        last_fire_before = max([t for t in fires if t < s0], default=-99)
        cd_shadow = s0 - last_fire_before < 2.0
        tag = "hit" if fired_in else "MISS"
        why = ""
        if not fired_in:
            if cd_shadow:
                why = "cooldown<2s"
            elif len(imp_peaks) < 2:
                why = f"imp峰不足({len(imp_peaks)})"
            elif not any(0.2 <= g <= 1.45 for g in gaps):
                why = f"峰间距非法{gaps}"
            elif max_sem >= 0.32:
                why = f"配对存在但评估帧sem窗未达阈? (max_sem={max_sem:.2f})"
            else:
                why = "sem弱"
        print(f"gt{i:2d} {tag} {s0:6.1f}-{e0:6.1f} | max_sem={max_sem:.3f} max_imp={max_imp:.3f} "
              f"peaks={[round(t,2) for t in imp_peaks]} gaps={gaps} sem_peak@{t_sem:.1f} "
              f"cd_shadow={cd_shadow} {why}")

    out = args.out or str(REPO / "runs_end_side" / "knock_autopsy" / f"{take.name}_trigger_curves.json")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps({"take": take.name, "fps": fps, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print("\ncurves ->", out)


if __name__ == "__main__":
    main()
