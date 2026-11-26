# -*- coding: utf-8 -*-
"""导出 dev-only 接触级双敲候选复核包。

用途：弱敲/接触级标注不靠幅度合成臆造，而是把训练划分内自动检测到的接触时刻
导出成 sheet + JSONL，供用户逐条确认、补标 strong/weak。封存 take 永远不进入此包。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cut_knock_candidates import contact_sheet  # noqa: E402
from double_knock_data import build_window_dataset  # noqa: E402


def _interval_index_for_time(intervals, t_s: float) -> int:
    for i, iv in enumerate(intervals):
        if float(iv.start_s) <= t_s <= float(iv.end_s):
            return i
    return -1


def export_contact_review_pack(
    out_dir: str | Path = "runs_knock/contact_review_impact_dev_v1",
    dataset_dir: str | Path = "dataset",
    pose_roi_dir: str | Path = "runs_knock/pose_roi",
    rows_per_sheet: int = 30,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(out_dir) if Path(out_dir).is_absolute() else repo_root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    data = build_window_dataset(
        dataset_dir=dataset_dir,
        pose_roi_dir=pose_roi_dir,
        window=48,
        stride=4,
        size=96,
        contact_sheet_dir=None,
        split="dev",
    )

    review_rows: list[dict] = []
    take_rows: list[dict] = []
    for take_id, d in sorted(data.items()):
        fps = float(d["fps"])
        contacts = sorted(int(c) for c in np.asarray(d["contacts"], dtype=np.int64).tolist())
        impact = np.asarray(d["impact_full"], dtype=np.float32)
        times = [c / fps for c in contacts]

        sheet_for_row: dict[int, str] = {}
        for chunk_idx in range(0, len(times), rows_per_sheet):
            chunk = times[chunk_idx : chunk_idx + rows_per_sheet]
            sheet = out_path / f"{take_id}_contacts_{chunk_idx // rows_per_sheet:02d}.png"
            contact_sheet(d["frames"], chunk, fps, sheet)
            for local_i in range(len(chunk)):
                sheet_for_row[chunk_idx + local_i] = str(sheet)

        rows = []
        for row_i, c in enumerate(contacts):
            t_s = round(c / fps, 3)
            lo = max(0, c - 2)
            hi = min(len(impact), c + 3)
            peak_impact = float(np.max(impact[lo:hi])) if hi > lo else 0.0
            interval_index = _interval_index_for_time(d["intervals"], t_s)
            row = {
                "review_status": "pending",
                "candidate_source": "impact_detector_v1",
                "take_id": take_id,
                "interval_index": interval_index,
                "contact_frame": c,
                "candidate_time_s": t_s,
                "impact_score": round(float(impact[c]), 4),
                "peak_impact_pm2f": round(peak_impact, 4),
                "sheet": sheet_for_row.get(row_i, ""),
                "contact_accepted": None,
                "contact_strength": None,
                "reviewer": "",
                "notes": "",
            }
            rows.append(row)
            review_rows.append(row)

        take_rows.append({
            "take_id": take_id,
            "n_contacts": len(contacts),
            "n_intervals": len(d["intervals"]),
            "feature_fps": fps,
            "rows": rows,
        })
        print(f"{take_id}: {len(contacts)} contact candidates across {len(d['intervals'])} intervals", flush=True)

    report = {
        "split": "dev",
        "candidate_source": "impact_detector_v1",
        "review_schema": {
            "contact_accepted": "true/false；该峰是否真是手指接触桌面",
            "contact_strength": "strong/weak/uncertain；weak 用于补弱敲样本",
            "reviewer": "复核人",
            "notes": "自由备注",
        },
        "n_takes": len(take_rows),
        "n_pending_candidates": len(review_rows),
        "takes": take_rows,
    }
    (out_path / "candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_path / "review_queue.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"-> {out_path / 'candidates.json'}", flush=True)
    print(f"-> {out_path / 'review_queue.jsonl'} ({len(review_rows)} pending contact candidates)", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs_knock/contact_review_impact_dev_v1")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--roi", default="runs_knock/pose_roi")
    ap.add_argument("--rows-per-sheet", type=int, default=30)
    args = ap.parse_args()
    export_contact_review_pack(args.out, args.dataset, args.roi, args.rows_per_sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
