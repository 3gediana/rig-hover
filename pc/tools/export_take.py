"""Export finished robot takes from the annotation service data root into the
git-tracked ``dataset/`` folder of this repository.

What is copied per take (small, git-friendly):

    dataset/takes/<take_id>/take.mp4        H.264 MP4 published by the service
    dataset/takes/<take_id>/annotation.json  interval labels (if any)
    dataset/takes/<take_id>/meta.json        recorder metadata (fps, capture_fps, ...)
    dataset/takes/<take_id>/take.json        compact summary used by dataset/INDEX.md

The raw JPEG frames and ``frames.jsonl`` under ``<stem>_source/`` are NOT copied:
they are several tens of MB per take, contain personal imagery, and are already
kept locally for training.  Reproduce the export with:

    python pc/tools/export_take.py --video robot_20261107_204337_5ca36d

    # every take whose .mp4 is newer than a date, minus explicit exclusions
    python pc/tools/export_take.py --since 2026-11-07 \
        --exclude robot_20261105_004630_ef1d19 \
        --exclude robot_20261105_190704_3e80b8

    python pc/tools/export_take.py --all --index
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "video_annotation_data"
DEFAULT_DATASET = REPO_ROOT / "dataset"
VIDEO_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def annotation_name(video_name: str) -> str:
    """Same scheme as Store.annotation_path(): sha256(name)[:20].json."""
    return hashlib.sha256(video_name.encode("utf-8")).hexdigest()[:20] + ".json"


def find_meta(data_root: Path, video: Path) -> Path | None:
    meta = data_root / "videos" / f"{video.stem}_source" / "meta.json"
    return meta if meta.exists() else None


def export_one(data_root: Path, dataset: Path, video: Path) -> dict:
    take_id = video.stem
    out = dataset / "takes" / take_id
    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(video, out / "take.mp4")
    meta_src = find_meta(data_root, video)
    meta: dict = {}
    if meta_src is not None:
        shutil.copy2(meta_src, out / "meta.json")
        meta = json.loads(meta_src.read_text(encoding="utf-8"))

    ann_src = data_root / "annotations" / annotation_name(video.name)
    ann: dict | None = None
    ann_out = out / "annotation.json"
    if ann_src.exists():
        shutil.copy2(ann_src, ann_out)
        ann = json.loads(ann_src.read_text(encoding="utf-8"))
    elif ann_out.exists():
        ann_out.unlink()

    summary = {
        "take_id": take_id,
        "video": "take.mp4",
        "source_video": video.name,
        "duration_s": meta.get("duration_s"),
        "fps": meta.get("fps"),
        "capture_fps": meta.get("capture_fps"),
        "raw_frames": meta.get("frames"),
        "video_frames": meta.get("video_frames"),
        "codec": meta.get("codec"),
        "annotated": bool(ann and ann.get("annotations")),
        "annotation_count": len((ann or {}).get("annotations") or []),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "take.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_index(dataset: Path) -> None:
    rows = []
    for take_json in sorted((dataset / "takes").glob("*/take.json")):
        rows.append(json.loads(take_json.read_text(encoding="utf-8")))
    lines = [
        "# 数据集索引",
        "",
        "由 `pc/tools/export_take.py --index` 生成，不要手改。",
        "",
        "| take | 时长 s | 输出 fps | 采集 fps | 原始帧 | 视频帧 | 标注 | 区间数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {id} | {d} | {fps} | {cap} | {raw} | {vf} | {a} | {n} |".format(
                id=r["take_id"],
                d=r.get("duration_s") or "-",
                fps=r.get("fps") or "-",
                cap=r.get("capture_fps") or "-",
                raw=r.get("raw_frames") or "-",
                vf=r.get("video_frames") or "-",
                a="是" if r.get("annotated") else "否",
                n=r.get("annotation_count") or 0,
            )
        )
    lines += [
        "",
        "标记为未标注的 take 只放了视频与 meta，用来继续标注或当负样本素材；"
        "已经标注的 take 里 `annotation.json` 是区间标签。",
        "",
        "原始 JPEG 帧与 `frames.jsonl` 不在 git 里（单条几十 MB，且含个人影像），"
        "在采集电脑的 `video_annotation_data/videos/<take>_source/` 下。",
        "",
    ]
    (dataset / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="copy finished takes into dataset/")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--video", action="append", default=[], help="take id or file name; repeatable")
    ap.add_argument("--since", help="include videos whose mp4 mtime >= YYYY-MM-DD")
    ap.add_argument("--exclude", action="append", default=[], help="take id to skip; repeatable")
    ap.add_argument("--all", action="store_true", help="include every video under data_root/videos")
    ap.add_argument("--index", action="store_true", help="regenerate dataset/INDEX.md after export")
    args = ap.parse_args(argv)

    if not args.video and not args.since and not args.all:
        ap.error("give --video, --since or --all")
    data_root: Path = args.data_root.resolve()
    dataset: Path = args.dataset.resolve()
    videos_dir = data_root / "videos"
    if not videos_dir.is_dir():
        ap.error(f"videos dir not found: {videos_dir}")

    excluded = {x.strip() for x in args.exclude if x.strip()}
    since = datetime.strptime(args.since, "%Y-%m-%d").timestamp() if args.since else None

    wanted: list[Path] = []
    if args.video:
        for item in args.video:
            p = videos_dir / item
            if p.suffix.lower() not in VIDEO_EXTS:
                p = videos_dir / f"{item}.mp4"
            if not p.exists():
                found = list(videos_dir.glob(f"{item}*"))
                found = [x for x in found if x.suffix.lower() in VIDEO_EXTS]
                if len(found) != 1:
                    print(f"! cannot resolve --video {item!r}", file=sys.stderr)
                    continue
                p = found[0]
            wanted.append(p)
    else:
        for p in sorted(videos_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            if p.stem in excluded:
                continue
            if since is not None and p.stat().st_mtime < since:
                continue
            wanted.append(p)

    exported = []
    for p in wanted:
        summary = export_one(data_root, dataset, p)
        exported.append(summary)
        print(f"exported {summary['take_id']}: {summary['duration_s']}s, "
              f"{summary['annotation_count']} intervals")

    labels_src = data_root / "labels.json"
    if labels_src.exists():
        shutil.copy2(labels_src, dataset / "labels.json")
        print(f"labels -> {dataset / 'labels.json'}")

    if args.index:
        write_index(dataset)
        print(f"index -> {dataset / 'INDEX.md'}")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())