"""桌面动作区间标注 -> 逐帧分类数据集。

数据来源：`dataset/takes/<take_id>/`（take.mp4 + take.json + annotation.json），
标注是时间区间（start_s, end_s），这里把区间映射成每个视频帧的类别；没有标注的
帧记为 ignore（-100），不当负样本。

非默认标签按名称合并：
    连续敲击   -> knock
    抬手但未敲 -> hand_raise
    晃拳       -> other
    检测摆头？ -> other

用法（被 train_interval.py 调用，也可单独检查）：
    python pc/train/interval_data.py --dataset dataset
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ["knock", "fist_hold", "hand_raise", "reach", "other", "background"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IGNORE = -100

# 标签名（中文）-> 统一类别；服务默认标签的 id 本身就是类别名
NAME_TO_CLASS = {
    "敲击": "knock",
    "静止握拳": "fist_hold",
    "抬手/举拳": "hand_raise",
    "伸手但未敲": "reach",
    "其它动作": "other",
    "背景/无动作": "background",
    "连续敲击": "knock",
    "抬手但未敲": "hand_raise",
    "晃拳": "other",
    "检测摆头？": "other",
}


def resolve_label(label_id: str, id_to_name: dict[str, str]) -> str | None:
    """把标注文件里的 label 字段解析成 6 类之一；无法解析返回 None。"""
    if label_id in CLASS_TO_IDX:
        return label_id
    name = id_to_name.get(label_id)
    if name is None:
        return None
    if name in CLASS_TO_IDX:
        return name
    return NAME_TO_CLASS.get(name)


@dataclass
class Interval:
    start_s: float
    end_s: float
    label: str


@dataclass
class Take:
    take_id: str
    video_path: Path
    fps: float
    duration_s: float
    video_frames: int
    annotation_path: Path | None = None
    intervals: list[Interval] = field(default_factory=list)

    @property
    def annotated(self) -> bool:
        return self.annotation_path is not None

    @property
    def n_frames(self) -> int:
        return self.video_frames


def load_takes(dataset_dir: Path) -> list[Take]:
    """扫描 dataset/takes/*/take.json，返回按 take_id 排序的 Take 列表。"""
    takes: list[Take] = []
    for take_dir in sorted((dataset_dir / "takes").glob("*/")):
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


def load_labels(dataset_dir: Path) -> dict[str, str]:
    path = dataset_dir / "labels.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(labels, dict):
        labels = labels.get("labels", [])
    return {item["id"]: item["name"] for item in labels}


def load_intervals(take: Take, id_to_name: dict[str, str]) -> list[Interval]:
    if not take.annotation_path:
        return []
    payload = json.loads(take.annotation_path.read_text(encoding="utf-8"))
    intervals: list[Interval] = []
    for item in payload.get("annotations", []):
        label = resolve_label(item["label"], id_to_name)
        if label is None:
            raise ValueError(f"{take.take_id}: unknown label {item['label']!r}")
        start = float(item["start_s"])
        end = float(item["end_s"])
        if end <= start:
            continue
        intervals.append(Interval(start, end, label))
    intervals.sort(key=lambda x: (x.start_s, x.end_s))
    return intervals


def frame_labels(intervals: list[Interval], n_frames: int, fps: float) -> np.ndarray:
    """帧中心落在 [start_s, end_s) 内的帧取该区间标签；其余 -100。"""
    labels = np.full(n_frames, IGNORE, dtype=np.int64)
    if n_frames <= 0:
        return labels
    for iv in intervals:
        first = int(np.ceil(iv.start_s * fps - 0.5))
        last = int(np.ceil(iv.end_s * fps - 0.5))  # exclusive
        first = max(first, 0)
        last = min(last, n_frames)
        if last > first:
            labels[first:last] = CLASS_TO_IDX[iv.label]
    return labels


def decode_gray(video_path: Path, size: int) -> np.ndarray:
    """把视频解码成 (T, size, size) uint8 灰度帧。"""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.shape[0] != size or gray.shape[1] != size:
            gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {video_path}")
    return np.stack(frames, axis=0)


def build_windows(n_frames: int, window: int) -> list[tuple[int, int]]:
    """把一段视频切成不重叠窗口，返回 (start, stop) 半开区间。"""
    return [(start, min(start + window, n_frames)) for start in range(0, n_frames, window)]


def window_sample(
    take_frames: np.ndarray,
    take_labels: np.ndarray,
    start: int,
    window: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造一个窗口的输入 (T, 2, size, size) 和标签 (T,)。

    通道 0 = 当前帧灰度，通道 1 = 与上一帧的差分（第一帧差分置 0）。
    窗口尾部用最后一帧补齐，补齐帧标签为 ignore。
    """
    stop = min(start + window, take_frames.shape[0])
    idx = np.arange(start, stop)
    frames = take_frames[idx]
    labels = take_labels[idx].astype(np.int64)
    prev_idx = np.maximum(idx - 1, 0)
    diff = take_frames[idx].astype(np.int16) - take_frames[prev_idx].astype(np.int16)
    pad = window - (stop - start)
    if pad > 0:
        last = frames[-1]
        frames = np.concatenate([frames, np.repeat(last[None], pad, axis=0)], axis=0)
        diff = np.concatenate([diff, np.zeros((pad, size, size), dtype=np.int16)], axis=0)
        labels = np.concatenate([labels, np.full(pad, IGNORE, dtype=np.int64)])
    x = np.stack([frames.astype(np.float32) / 255.0, diff.astype(np.float32) / 255.0], axis=1)
    return x, labels


def describe(takes: list[Take], id_to_name: dict[str, str]) -> str:
    lines = [f"{'take':34s} {'dur':>8s} {'frames':>7s} {'intervals':>9s} {'labeled':>9s}"]
    for take in takes:
        intervals = load_intervals(take, id_to_name)
        labels = frame_labels(intervals, take.n_frames, take.fps)
        annotated = int((labels != IGNORE).sum())
        lines.append(
            f"{take.take_id:34s} {take.duration_s:8.2f} {take.n_frames:7d} "
            f"{len(intervals):9d} {annotated:9d}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(__file__).resolve().parents[2] / dataset_dir
    takes = load_takes(dataset_dir)
    labels = load_labels(dataset_dir)
    print(describe(takes, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())