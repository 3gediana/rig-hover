"""把标注服务的批量导出 JSON 拆回 per-video 标注文件。

用法:
    python pc/tools/import_annotations.py --export video_annotations.json \
        --data-root D:\\robot_work\\video_annotation_data --backup

输入是 /api/export 产生的格式：
    {"schema_version": 1, "exported_at": ..., "labels": [...], "videos": [
        {"schema_version": 1, "video": "xxx.mp4", "duration_s": 98.8,
         "annotations": [{"id": ..., "label": ..., "start_s": ..., "end_s": ..., "note": ...}]}
    ]}

写出:
    <data-root>/labels.json
    <data-root>/annotations/<sha256(video_name)[:20]>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def annotation_name(video_name: str) -> str:
    return hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:20] + ".json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--data-root", default=r"D:\robot_work\video_annotation_data")
    ap.add_argument("--backup", action="store_true", help="覆盖前把 annotations/ 和 labels.json 备份")
    args = ap.parse_args()

    root = Path(args.data_root)
    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    if export.get("schema_version") != 1:
        raise SystemExit(f"unsupported export schema_version={export.get('schema_version')}")

    if args.backup:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = root / f"_backup_annotation_import_{stamp}"
        backup.mkdir(parents=True, exist_ok=True)
        if ann_dir.exists():
            shutil.copytree(ann_dir, backup / "annotations", dirs_exist_ok=True)
        labels_path = root / "labels.json"
        if labels_path.exists():
            shutil.copy2(labels_path, backup / "labels.json")
        print(f"backup -> {backup}")

    # labels.json：按 id 合并，保留原顺序；新 id 追加
    labels_path = root / "labels.json"
    if labels_path.exists():
        current = json.loads(labels_path.read_text(encoding="utf-8"))
        existing = current if isinstance(current, list) else current.get("labels", [])
    else:
        existing = []
    known = {item["id"] for item in existing}
    for item in export.get("labels", []):
        if item["id"] not in known:
            existing.append({"id": item["id"], "name": item["name"], "color": item.get("color", "#4f8cff")})
            known.add(item["id"])
    labels_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"labels -> {labels_path} ({len(existing)} labels)")

    total_intervals = 0
    for video in export.get("videos", []):
        name = video["video"]
        # 只保留工具认识的字段，按时间排序
        annotations = sorted(video.get("annotations", []), key=lambda a: (a["start_s"], a["end_s"]))
        payload = {
            "schema_version": 1,
            "video": name,
            "duration_s": video.get("duration_s"),
            "annotations": [
                {
                    "id": a.get("id", f"seg_{i}"),
                    "label": a["label"],
                    "start_s": float(a["start_s"]),
                    "end_s": float(a["end_s"]),
                    "note": a.get("note", ""),
                }
                for i, a in enumerate(annotations)
            ],
        }
        out = ann_dir / annotation_name(name)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        total_intervals += len(annotations)
        print(f"{name}: {len(annotations)} intervals -> {out.name}")
    print(f"total: {len(export.get('videos', []))} videos, {total_intervals} intervals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())