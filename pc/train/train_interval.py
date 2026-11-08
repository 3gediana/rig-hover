"""区间标注 -> 逐帧动作分类：最小可跑通训练/评估流水线。

特点
- 输入：dataset/takes/<take>/(take.mp4, annotation.json, take.json)
- 标签：时间区间映射到帧；未标注帧 = ignore（不当负样本）
- 模型：`frame`（逐帧 CNN 基线）/ `gru`（CNN + 因果 GRU）
- 评估：按 take 留一（leave-one-take-out），报告逐帧 macro-F1 / knock F1 / 混淆矩阵
- 只用于验证链路与数据格式；4 条 take 的绝对指标不能当泛化结论

用法
    python pc/train/train_interval.py --dataset dataset --epochs 25 --out runs_interval/run1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interval_data import (  # noqa: E402
    CLASSES,
    IGNORE,
    build_windows,
    decode_gray,
    frame_labels,
    load_intervals,
    load_labels,
    load_takes,
    window_sample,
)

NUM_CLASSES = len(CLASSES)


class FrameEncoder(nn.Module):
    def __init__(self, feat_dim: int = 192, in_ch: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, feat_dim), nn.ReLU(inplace=True), nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2, H, W) -> (B, T, feat)
        b, t = x.shape[:2]
        return self.net(x.reshape(b * t, *x.shape[2:])).reshape(b, t, -1)


class FrameCNN(nn.Module):
    def __init__(self, feat_dim: int = 192):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim)
        self.head = nn.Linear(feat_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class CnnGRU(nn.Module):
    def __init__(self, feat_dim: int = 192, hidden: int = 128):
        super().__init__()
        self.encoder = FrameEncoder(feat_dim)
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        out, _ = self.gru(feat)
        return self.head(out)


def build_model(name: str) -> nn.Module:
    if name == "frame":
        return FrameCNN()
    if name == "gru":
        return CnnGRU()
    raise ValueError(f"unknown model {name}")


class WindowDataset(Dataset):
    def __init__(self, takes: list, frames: dict, labels: dict, window: int, size: int):
        self.items: list[tuple[str, int]] = []
        self.takes = takes
        self.frames = frames
        self.labels = labels
        self.window = window
        self.size = size
        for take in takes:
            n = int(self.frames[take.take_id].shape[0])
            for start, _ in build_windows(n, window):
                self.items.append((take.take_id, start))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        take_id, start = self.items[idx]
        x, y = window_sample(self.frames[take_id], self.labels[take_id], start, self.window, self.size)
        return torch.from_numpy(x), torch.from_numpy(y)


def compute_class_weights(labels: dict, takes: list) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for take in takes:
        y = labels[take.take_id]
        valid = y[y != IGNORE]
        counts += np.bincount(valid, minlength=NUM_CLASSES)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def confusion_matrix(pred: np.ndarray, target: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for p, t in zip(pred, target):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-9), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-9), 0.0)
    support = cm.sum(axis=1)
    present = support > 0
    result = {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "knock_f1": float(f1[0]),
        "confusion": cm.tolist(),
    }
    return result


def run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        valid = int((y != IGNORE).sum().item())
        if valid:
            total += loss.item() * valid
            count += valid
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
        tgt = y.numpy().reshape(-1)
        mask = tgt != IGNORE
        preds.append(pred[mask])
        targets.append(tgt[mask])
    pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    tgt = np.concatenate(targets) if targets else np.array([], dtype=np.int64)
    if len(tgt) == 0:
        return {"frames": 0, "macro_f1": 0.0, "knock_f1": 0.0, "accuracy": 0.0,
                "per_class": {}, "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64).tolist()}
    cm = confusion_matrix(pred, tgt)
    return {"frames": int(len(tgt)), **metrics_from_cm(cm)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="runs_interval/run")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--models", default="frame,gru")
    ap.add_argument("--takes", default="", help="逗号分隔的 take_id，默认全部")
    ap.add_argument("--folds", type=int, default=0, help="只跑前 N 折；0 = 全部留一折")
    ap.add_argument("--class-weight", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    id_to_name = load_labels(dataset_dir)
    takes = load_takes(dataset_dir)
    takes = [t for t in takes if t.annotation_path is not None]
    if args.takes:
        wanted = {s.strip() for s in args.takes.split(",") if s.strip()}
        takes = [t for t in takes if t.take_id in wanted]
    if not takes:
        raise SystemExit("no annotated takes found")

    print(f"device={args.device} takes={[t.take_id for t in takes]}")
    frames: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    intervals: dict[str, list] = {}
    for take in takes:
        intervals[take.take_id] = load_intervals(take, id_to_name)
        frames[take.take_id] = decode_gray(take.video_path, args.size)
        y = frame_labels(intervals[take.take_id], frames[take.take_id].shape[0], take.fps)
        labels[take.take_id] = y
        valid = int((y != IGNORE).sum())
        print(f"  {take.take_id}: {frames[take.take_id].shape[0]} frames, {len(intervals[take.take_id])} intervals, {valid} labeled")

    device = torch.device(args.device)
    class_weight = compute_class_weights(labels, takes).to(device) if args.class_weight else None
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE, weight=class_weight)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results: dict[str, dict] = {name: {"folds": []} for name in model_names}
    fold_takes = takes[: args.folds] if args.folds else takes

    for val_take in fold_takes:
        train_takes = [t for t in takes if t.take_id != val_take.take_id]
        if not train_takes:
            continue
        train_ds = WindowDataset(train_takes, frames, labels, args.window, args.size)
        val_ds = WindowDataset([val_take], frames, labels, args.window, args.size)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
        print(f"\nfold val={val_take.take_id}: train_windows={len(train_ds)} val_windows={len(val_ds)}")
        for name in model_names:
            torch.manual_seed(args.seed)
            model = build_model(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            t0 = time.time()
            for epoch in range(1, args.epochs + 1):
                loss = run_epoch(model, train_loader, criterion, device, optimizer)
                if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                    print(f"  [{name}] epoch {epoch:3d} loss={loss:.4f}")
            metrics = evaluate(model, val_loader, device)
            metrics.update({"fold": val_take.take_id, "model": name, "train_s": round(time.time() - t0, 1)})
            results[name]["folds"].append(metrics)
            print(f"  [{name}] macroF1={metrics['macro_f1']:.4f} knockF1={metrics['knock_f1']:.4f} acc={metrics['accuracy']:.4f} ({metrics['train_s']}s)")

    summary: dict[str, dict] = {}
    for name, data in results.items():
        folds = data["folds"]
        if not folds:
            continue
        keys = ["macro_f1", "knock_f1", "accuracy"]
        agg = {}
        for key in keys:
            vals = np.array([f[key] for f in folds], dtype=np.float64)
            agg[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "values": vals.tolist()}
        summary[name] = agg

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": str(dataset_dir),
        "config": vars(args),
        "takes": [t.take_id for t in takes],
        "classes": CLASSES,
        "results": results,
        "summary": summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 区间标注逐帧分类（留一 take）", "", f"- 数据：{', '.join(t.take_id for t in takes)}", f"- 窗口={args.window} 帧，输入={args.size}px，epochs={args.epochs}", ""]
    lines.append("| 模型 | macro-F1 | knock F1 | accuracy |")
    lines.append("|---|---|---|---|")
    for name, agg in summary.items():
        lines.append(
            f"| {name} | {agg['macro_f1']['mean']:.4f} ± {agg['macro_f1']['std']:.4f} "
            f"| {agg['knock_f1']['mean']:.4f} ± {agg['knock_f1']['std']:.4f} "
            f"| {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f} |"
        )
    lines += ["", "逐折："]
    for name, data in results.items():
        for f in data["folds"]:
            lines.append(f"- {name} / val={f['fold']}: frames={f['frames']} macroF1={f['macro_f1']:.4f} knockF1={f['knock_f1']:.4f}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nmetrics -> {out_dir / 'metrics.json'}")
    print(f"report  -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())