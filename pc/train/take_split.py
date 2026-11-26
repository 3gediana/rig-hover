# -*- coding: utf-8 -*-
"""Split-aware take metadata loading for dev-only training and registered final evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PACK_DIR = Path(__file__).resolve().parents[1] / "pack"
if str(_PACK_DIR) not in sys.path:
    sys.path.insert(0, str(_PACK_DIR))

import sealed  # noqa: E402
from interval_data import Take  # noqa: E402


def load_takes_for_split(
    dataset_dir: str | Path,
    split: str = "dev",
    final_eval: bool = False,
) -> list[Take]:
    """Load take metadata for exactly one governance split without opening sealed files by default.

    `dev` skips sealed directories before even reading their take.json. `sealed`
    requires explicit final-eval authorisation before the first sealed metadata file is opened.
    """
    if split not in {"dev", "sealed"}:
        raise ValueError(f"unknown take split: {split}")
    ds_path = Path(dataset_dir)
    rows: list[tuple[Path, bool]] = []
    auth_paths: list[Path] = []
    for take_dir in sorted((ds_path / "takes").glob("*/")):
        is_sealed = sealed.is_sealed_path(take_dir)
        if is_sealed and split == "dev":
            continue
        if (not is_sealed) and split == "sealed":
            continue
        if is_sealed:
            auth_paths.extend([take_dir / "take.json", take_dir / "annotation.json", take_dir / "take.mp4"])
        rows.append((take_dir, is_sealed))

    if auth_paths:
        sealed.require_final_eval(
            auth_paths,
            purpose="double-knock dataset final held-out evaluation",
            argv=["--final-eval"] if final_eval else [],
        )

    takes: list[Take] = []
    for take_dir, _is_sealed in rows:
        info_path = take_dir / "take.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        take = Take(
            take_id=info.get("take_id", take_dir.name),
            video_path=take_dir / "take.mp4",
            fps=float(info["fps"]),
            duration_s=float(info["duration_s"]),
            video_frames=int(info["video_frames"]),
        )
        ann_path = take_dir / "annotation.json"
        if ann_path.exists():
            take.annotation_path = ann_path
        takes.append(take)
    return takes
