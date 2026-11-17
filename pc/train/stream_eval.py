"""端侧双敲模型（DoubleKnockNet）的流式模式标定与诚实评估。

背景（2026-11-16）：训练侧离线评估用 **z-score 空间** 的脉冲阈值
（`find_double_knock_triggers` 先把平滑概率标准化再比较）；而 Python
`DoubleKnockDetector` 和 Android `TfliteDoubleKnockModel` 在流式推理时比较的是
**概率空间**。两类阈值不能混用，否则导出的 `model.json` 里 `impThreshold`
（z 值，可能 > 1）在 App 侧永远不可能达到——两种实现都会静默不触发。

本脚本因此做两件事：
1. `--deploy-ckpt`：用全量训练 take 在流式（概率）语义下网格搜索阈值，输出
   可直接写进 model.json 的 `impThreshold/doubleThreshold`；
2. `--folds-dir`：按 take 留一，在训练 take 上标定流式阈值，再原样应用到留出
   take，报告流式命中/误报（这才是 App 实际能复现的口径）。

用法：
    cd D:\\robot_work\\pc\\train
    & G:\\ai\\python11\\python.exe stream_eval.py --deploy-ckpt ..\\..\\runs_knock\\...\\double_knock_deploy.pt --out-json ...
    & G:\\ai\\python11\\python.exe stream_eval.py --folds-dir ..\\..\\runs_knock\\... --out-json ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_double_knock as tdk
from double_knock_data import build_window_dataset


PAIR_MIN_S = 0.20
PAIR_MAX_S = 1.45
COOLDOWN_S = 1.8


def _running_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """detector 的 imp_hist（长度不超过 smooth_frames）均值。"""
    csum = np.cumsum(arr, dtype=np.float64)
    idx = np.arange(arr.size)
    start = np.maximum(0, idx - window + 1)
    total = csum - np.where(start > 0, csum[np.maximum(start - 1, 0)], 0.0)
    count = (idx - start + 1).astype(np.float64)
    return (total / count).astype(np.float32)


def _sliding_max(arr: np.ndarray, window: int) -> np.ndarray:
    """detector 的 sem_hist（maxlen）窗口最大值（含当前帧，长度不足时只看已有元素）。"""
    out = np.empty(arr.size, dtype=np.float32)
    from collections import deque

    dq: deque[int] = deque()
    for i, v in enumerate(arr):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] <= v:
            dq.pop()
        dq.append(i)
        out[i] = float(arr[dq[0]]) if dq else 0.0
    return out


def simulate_triggers(
    imp_prob: np.ndarray,
    cls_prob: np.ndarray,
    fps: float = 20.0,
    smooth_frames: int = 3,
    imp_thr: float = 0.5,
    double_thr: float = 0.4,
    pair_min_s: float = PAIR_MIN_S,
    pair_max_s: float = PAIR_MAX_S,
    cooldown_s: float = COOLDOWN_S,
) -> list[dict]:
    """严格复刻 DoubleKnockDetector.step 的流式触发逻辑（概率空间）。

    返回每次触发：first_s / second_s / score。
    """
    imp_prob = np.asarray(imp_prob, dtype=np.float32)
    cls_prob = np.asarray(cls_prob, dtype=np.float32)
    if imp_prob.size == 0:
        return []

    sem = cls_prob[:, 2] + 0.45 * cls_prob[:, 3]
    smooth = 0.65 * imp_prob + 0.35 * _running_mean(imp_prob, smooth_frames)
    smooth_hist_len = max(5, 2 * smooth_frames)
    sem_hist_len = max(8, 2 * smooth_frames)
    win_sem = _sliding_max(sem, sem_hist_len)

    triggers: list[dict] = []
    peak_times: list[float] = []
    last_trigger = -1e9
    # 第 i 帧判断的是 smooth[i-1]（prev），需要 smooth 历史至少 3 帧 ⇒ i >= 2
    for i in range(2, smooth.size):
        prev = smooth[i - 1]
        is_peak = prev >= imp_thr and prev >= smooth[i - 2] and prev > smooth[i]
        if not is_peak:
            continue
        t_now = i / fps
        while peak_times and t_now - peak_times[0] > pair_max_s + 0.5:
            peak_times.pop(0)
        peak_times.append(t_now)
        if len(peak_times) < 2:
            continue
        gap = t_now - peak_times[-2]
        if not (pair_min_s <= gap <= pair_max_s):
            continue
        if win_sem[i] < double_thr:
            continue
        if t_now - last_trigger < cooldown_s:
            continue
        last_trigger = t_now
        triggers.append(
            {
                "first_s": round(t_now - gap, 3),
                "second_s": round(t_now, 3),
                "score": round(float(win_sem[i]), 3),
            }
        )
    return triggers


def calibrate_streaming(
    preds: dict[str, tuple[np.ndarray, np.ndarray]],
    takes: dict[str, dict],
    grid_smooth=(3,),
    grid_imp=tuple(np.round(np.arange(0.20, 0.96, 0.05), 2)),
    grid_double=tuple(np.round(np.arange(0.15, 0.81, 0.05), 2)),
) -> dict:
    """只在训练 take 上搜索概率空间阈值，返回最优标定。"""
    best = None
    for sm in grid_smooth:
        for imp_thr in grid_imp:
            for dbl_thr in grid_double:
                hit = gt = tp = fp = 0
                for tid, (imp_p, cls_p) in preds.items():
                    d = takes[tid]
                    tr = simulate_triggers(
                        imp_p,
                        cls_p,
                        fps=float(d["fps"]),
                        smooth_frames=sm,
                        imp_thr=float(imp_thr),
                        double_thr=float(dbl_thr),
                        pair_min_s=PAIR_MIN_S,
                        pair_max_s=PAIR_MAX_S,
                    )
                    ev = tdk.eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
                    hit += ev["hit"]
                    gt += ev["total_gt"]
                    tp += ev["tp"]
                    fp += ev["fp"]
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec = hit / max(gt, 1)
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                score = f1 - 0.008 * fp
                if best is None or score > best["score"]:
                    best = {
                        "score": round(float(score), 5),
                        "smooth_frames": int(sm),
                        "imp_thr": float(imp_thr),
                        "double_thr": float(dbl_thr),
                        "train_hit": int(hit),
                        "train_gt": int(gt),
                        "train_tp": int(tp),
                        "train_fp": int(fp),
                        "train_f1": round(float(f1), 4),
                    }
    return best


def _predict(data: dict[str, dict], model, device, window: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    preds = {}
    for tid, d in data.items():
        imp_p, cls_p = tdk.predict_full_sequence(model, d["vis_full"], d["kin_full"], window, device)
        preds[tid] = (imp_p, cls_p)
    return preds


def run_deploy(ckpt_path: Path, data: dict[str, dict], window: int, device) -> dict:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = tdk.DoubleKnockNet()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    preds = _predict(data, model, device, window)
    best = calibrate_streaming(preds, data)
    hit = gt = fp = 0
    per_take = {}
    for tid, (imp_p, cls_p) in preds.items():
        d = data[tid]
        tr = simulate_triggers(
            imp_p,
            cls_p,
            fps=float(d["fps"]),
            smooth_frames=best["smooth_frames"],
            imp_thr=best["imp_thr"],
            double_thr=best["double_thr"],
            pair_min_s=PAIR_MIN_S,
                        pair_max_s=PAIR_MAX_S,
        )
        ev = tdk.eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
        hit += ev["hit"]
        gt += ev["total_gt"]
        fp += ev["fp"]
        per_take[tid] = {"hit": ev["hit"], "gt": ev["total_gt"], "fp": ev["fp"], "f1": ev["f1"]}
    return {"calibration": best, "self_test": {"hit": hit, "gt": gt, "fp": fp}, "per_take": per_take}


def run_folds(folds_dir: Path, data: dict[str, dict], window: int, device) -> dict:
    import torch

    take_ids = sorted(data.keys())
    fold_reports = []
    total_hit = total_gt = total_fp = 0
    for val_tid in take_ids:
        ckpt_path = folds_dir / f"fold_{val_tid}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = tdk.DoubleKnockNet()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        train_ids = [t for t in take_ids if t != val_tid]
        train_data = {t: data[t] for t in train_ids}
        val_data = {val_tid: data[val_tid]}
        train_preds = _predict(train_data, model, device, window)
        calib = calibrate_streaming(train_preds, train_data)
        val_preds = _predict(val_data, model, device, window)
        d = data[val_tid]
        imp_p, cls_p = val_preds[val_tid]
        tr = simulate_triggers(
            imp_p,
            cls_p,
            fps=float(d["fps"]),
            smooth_frames=calib["smooth_frames"],
            imp_thr=calib["imp_thr"],
            double_thr=calib["double_thr"],
            pair_min_s=PAIR_MIN_S,
                        pair_max_s=PAIR_MAX_S,
        )
        ev = tdk.eval_triggers_on_take(tr, d["labels_full"], d["intervals"], d["fps"], d["duration_s"])
        total_hit += ev["hit"]
        total_gt += ev["total_gt"]
        total_fp += ev["fp"]
        fold_reports.append(
            {
                "take": val_tid,
                **{k: ev[k] for k in ("hit", "total_gt", "f1", "tp", "fp", "fa_per_min")},
                "calib": calib,
            }
        )
    minutes = sum(float(d["duration_s"]) for d in data.values()) / 60.0
    mean_f1 = float(np.mean([f["f1"] for f in fold_reports])) if fold_reports else 0.0
    return {
        "folds": fold_reports,
        "summary": {
            "hit": total_hit,
            "gt": total_gt,
            "fp": total_fp,
            "fa_per_min": round(total_fp / minutes, 3) if minutes > 0 else 0.0,
            "mean_f1": round(mean_f1, 4),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-ckpt", type=str, default="")
    ap.add_argument("--folds-dir", type=str, default="")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--roi", type=str, default=None)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset) if args.dataset else repo_root / "dataset"
    roi_dir = Path(args.roi) if args.roi else repo_root / "runs_knock" / "pose_roi"
    device = tdk.torch.device(args.device if args.device.startswith("cuda") and tdk.torch.cuda.is_available() else "cpu")
    print(f"[stream_eval] device={device}, window={args.window}, stride={args.stride}")

    data = build_window_dataset(
        dataset_dir=str(dataset_dir),
        pose_roi_dir=str(roi_dir),
        window=args.window,
        stride=args.stride,
        size=args.size,
    )
    result: dict = {"device": str(device), "window": args.window, "takes": sorted(data.keys())}
    if args.deploy_ckpt:
        result["deploy"] = run_deploy(Path(args.deploy_ckpt), data, args.window, device)
        print("deploy:", json.dumps(result["deploy"], ensure_ascii=False, indent=1)[:1500])
    if args.folds_dir:
        result["folds"] = run_folds(Path(args.folds_dir), data, args.window, device)
        print("folds:", json.dumps(result["folds"]["summary"], ensure_ascii=False))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved:", args.out_json)


if __name__ == "__main__":
    main()


