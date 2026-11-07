"""把"敲击/连续敲击"区间自动切成单次敲击候选，并生成检查图。

为什么要这一步：现在的标注是 1–10 s 的活动段，模型只能学"像不像在敲"，
学不会"敲了几下"。要从现有 take 里造出"单次敲击"和"两次敲击"样本，
必须先知道每次接触发生在哪一帧。这里用帧差能量的局部峰做候选，
输出检查图让人工快速确认/剔除，再用于训练"一下 vs 两下"。

用法
    python pc/train/cut_knock_candidates.py --take robot_20261107_213314_5c0158 \
        --dataset dataset --out runs_knock/candidates
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_data import decode_gray, load_intervals, load_labels, load_takes  # noqa: E402

KNOCK_LABELS = {"knock", "连续敲击", "敲击"}


def frame_energy(frames: np.ndarray) -> np.ndarray:
    diff = np.abs(np.diff(frames.astype(np.int16), axis=0)).astype(np.float32)
    return diff.mean(axis=(1, 2))


def segment_peaks(energy: np.ndarray, fps: float, min_gap_s: float = 0.30, k: float = 2.5) -> list[int]:
    if len(energy) < 3:
        return []
    med = float(np.median(energy))
    mad = float(np.median(np.abs(energy - med))) or 1e-6
    thr = max(med + k * 1.4826 * mad, 1.3 * med)
    min_gap = max(1, int(round(min_gap_s * fps)))
    peaks: list[int] = []
    for i in range(1, len(energy) - 1):
        if energy[i] < thr:
            continue
        if energy[i] < energy[i - 1] or energy[i] < energy[i + 1]:
            continue
        if peaks and i - peaks[-1] < min_gap:
            if energy[i] > energy[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def contact_sheet(frames: np.ndarray, times: list[float], fps: float, path: Path, offsets=(-0.4, -0.2, 0.0, 0.2, 0.4)):
    if not times:
        return
    cols = len(offsets)
    tile = 96
    rows = len(times)
    canvas = np.full((rows * tile, cols * tile, 3), 255, dtype=np.uint8)
    for r, t in enumerate(times):
        for c, off in enumerate(offsets):
            idx = int(round((t + off) * fps))
            idx = max(0, min(len(frames) - 1, idx))
            img = frames[idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            img = cv2.resize(img, (tile, tile))
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = img
        cv2.putText(canvas, f"#{r + 1} {t:.2f}s", (2, r * tile + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--take", default="", help="只处理一条 take；空表示全部")
    ap.add_argument("--out", default="runs_knock/candidates")
    ap.add_argument("--min-gap-s", type=float, default=0.30)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = load_labels(dataset_dir)
    takes = load_takes(dataset_dir)
    if args.take:
        takes = [t for t in takes if t.take_id == args.take]
    report = {"takes": []}
    for take in takes:
        if take.annotation_path is None:
            continue
        intervals = load_intervals(take, id_to_name)
        frames = decode_gray(take.video_path, 96)
        energy = frame_energy(frames)
        rows = []
        for idx, iv in enumerate(intervals):
            label = id_to_name.get(iv.label, iv.label)
            if label not in KNOCK_LABELS:
                continue
            a = max(0, int(iv.start_s * take.fps))
            b = min(len(energy), int(iv.end_s * take.fps))
            local = segment_peaks(energy[a:b], take.fps, args.min_gap_s)
            times = [round((a + p) / take.fps, 2) for p in local]
            sheet = out_dir / f"{take.take_id}_{idx:02d}.png"
            contact_sheet(frames, times, take.fps, sheet)
            rows.append({"interval_index": idx, "start_s": iv.start_s, "end_s": iv.end_s,
                         "label": label, "n_candidates": len(times), "candidates_s": times,
                         "sheet": str(sheet)})
        report["takes"].append({"take_id": take.take_id, "rows": rows})
        print(f"{take.take_id}: {len(rows)} knock intervals, "
              f"{sum(r['n_candidates'] for r in rows)} candidates")
    (out_dir / "candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out_dir / 'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())