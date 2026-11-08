"""ROI 敲击检测的推理与门控（离线回放 / 在线重放共用）。

链路:
    YOLO pose 腕点 -> 96x96 手部 ROI -> CNN 编码器 + GRU 时序头 -> 逐帧敲击概率
    -> 平滑 + 在线 z-score -> 双峰“敲两下”触发 -> 场景状态门控 -> fired 位

设计约束（对应零误报要求）:
* 模型只产生候选触发，不直接控制轮子；
* 只有场景状态是 MODE_GAZE_FIST（正在盯拳、等待敲击）时才把触发交出去；
* 两次触发之间强制冷却，避免连续敲击刷出多个事件；
* z-score 用在线均值/方差校准，不需要标签，也不需要整段录像先验。

用法（离线回放一条 take）:
    python pc/perception/knock_roi.py --model runs_knock/roi_20261109/fold_xxx.pt \
        --take dataset/takes/<take_id>/take.mp4 --roi runs_knock/pose_roi/<take_id>.npz

注意：本模块目前只做“检测 + 门控”演示，还没有替换 `scene_signals.KnockDetector`；
接入点是 `KnockRoiSignal.update(..., scenario_state=...)` 返回的 `fired`。
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np

ROI_SIZE = 96
HOLD_FRAMES = 24
MODE_GAZE_FIST = 2  # 必须与 pc/perception/scenario_v2.py 的 MODE_GAZE_FIST 一致


class RoiBoxTracker:
    """把腕点变成 ROI 框；丢检测时沿用上一帧框，超时返回 None。"""

    def __init__(self, roi_size: int = ROI_SIZE, hold_frames: int = HOLD_FRAMES):
        self.roi_size = roi_size
        self.hold_frames = hold_frames
        self.last_box: np.ndarray | None = None
        self.last_seen = -10**9
        self.frame_idx = 0

    def update(self, wrist_xy: tuple[float, float] | None, frame_w: int, frame_h: int) -> np.ndarray | None:
        self.frame_idx += 1
        if wrist_xy is not None:
            wx, wy = float(wrist_xy[0]), float(wrist_xy[1])
            side = self.roi_size
            x1, y1 = wx - side / 2, wy - side / 2
            x2, y2 = wx + side / 2, wy + side / 2
            x1 = min(max(x1, 0.0), frame_w - 2.0)
            y1 = min(max(y1, 0.0), frame_h - 2.0)
            x2 = min(max(x2, x1 + 2.0), float(frame_w))
            y2 = min(max(y2, y1 + 2.0), float(frame_h))
            self.last_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            self.last_seen = self.frame_idx
            return self.last_box
        if self.last_box is not None and self.frame_idx - self.last_seen <= self.hold_frames:
            return self.last_box
        return None


def crop_roi(gray: np.ndarray, box: np.ndarray | None, size: int = ROI_SIZE) -> np.ndarray:
    h, w = gray.shape[:2]
    if box is None:
        crop = gray
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 2, x2)), min(h, max(y1 + 2, y2))
        crop = gray[y1:y2, x1:x2]
    import cv2

    if crop.size == 0:
        crop = gray
    interp = cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(crop, (size, size), interpolation=interp)


def load_checkpoint(path: str | Path, device: str = "cuda:0"):
    """读取训练脚本保存的 ckpt，返回 (model, window, smooth, threshold)。"""
    import sys

    import torch

    train_dir = Path(__file__).resolve().parents[1] / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    from train_knock import KnockCnnGRU, KnockCnnTCN

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    name = ckpt.get("models", "gru")
    window = int(ckpt.get("window", ROI_SIZE))
    sel = ckpt.get("selection", {}) or {}
    in_ch = int(ckpt.get("in_ch", 2))
    model = KnockCnnGRU(in_ch=in_ch) if name == "gru" else KnockCnnTCN(in_ch=in_ch)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, window, int(sel.get("smooth", 10)), float(sel.get("threshold", 0.0)), in_ch


class KnockRoiSignal:
    """逐帧输入 -> 敲击分数 -> 双峰触发 -> 场景门控。"""

    def __init__(self, model_path: str | Path | None = None, device: str = "cuda:0",
                 window: int = ROI_SIZE, smooth_frames: int | None = None, threshold_z: float = 0.0,
                 pair_min_s: float = 0.25, pair_max_s: float = 1.5, cooldown_s: float = 2.0,
                 ema_decay: float = 0.99, gate_state: int = MODE_GAZE_FIST, fps: float = 20.0,
                 score_fn=None):
        self.score_fn = score_fn
        self.fps = float(fps)
        if score_fn is not None:
            self.model = None
            self.in_ch = 2
            self.window = window
            self.smooth_frames = smooth_frames or 1
            self.threshold_z = threshold_z
        else:
            if model_path is None:
                raise ValueError("model_path 或 score_fn 至少给一个")
            (self.model, ckpt_window, ckpt_smooth,
             ckpt_threshold, self.in_ch) = load_checkpoint(model_path, device)
            self.window = max(window, ckpt_window)
            self.smooth_frames = smooth_frames or ckpt_smooth
            self.threshold_z = ckpt_threshold if threshold_z == 0.0 else threshold_z
        self.pair_min_s = pair_min_s
        self.pair_max_s = pair_max_s
        self.cooldown_s = cooldown_s
        self.ema_decay = ema_decay
        self.gate_state = gate_state
        self.tracker = RoiBoxTracker()
        self.crops: deque[np.ndarray] = deque(maxlen=self.window)
        self.global_crops: deque[np.ndarray] = deque(maxlen=self.window) if self.in_ch == 4 else None
        self.prob_hist: deque[float] = deque(maxlen=self.smooth_frames)
        self.z_hist: deque[float] = deque(maxlen=max(5, 2 * self.smooth_frames))
        self.mu = 0.5
        self.var = 0.25
        self.frames_seen = 0
        self.peak_times: deque[float] = deque(maxlen=8)
        self.last_trigger_t = -1e9
        self.triggers = 0
        self.rejected = 0
        self.fired = 0
        self.last_prob = 0.0
        self.last_z = 0.0
        self.device = device

    def _score(self, crops: list[np.ndarray]) -> float:
        """对 window 帧 ROI 打分，返回最后一帧成为“敲击”的概率。"""
        if self.score_fn is not None:
            return float(self.score_fn(list(crops)))
        import cv2
        import torch

        g = np.stack(crops).astype(np.float32) / 255.0
        d = np.zeros_like(g)
        if len(g) > 1:
            d[1:] = np.abs(np.diff(np.stack(crops).astype(np.int16), axis=0)).astype(np.float32) / 255.0
        chans = [g, d]
        if self.in_ch == 4 and self.global_crops is not None:
            gg = np.stack(self.global_crops).astype(np.float32) / 255.0
            gd = np.zeros_like(gg)
            if len(gg) > 1:
                gd[1:] = np.abs(np.diff(np.stack(self.global_crops).astype(np.int16), axis=0)).astype(np.float32) / 255.0
            chans += [gg, gd]
        x = np.stack(chans, axis=1)  # (T,C,96,96)
        with torch.no_grad():
            logit = self.model(torch.from_numpy(x)[None].to(self.device))
            return float(torch.sigmoid(logit[0, -1]).cpu())

    def update(self, gray: np.ndarray, wrist_xy: tuple[float, float] | None,
               scenario_state: int | None = None) -> dict:
        h, w = gray.shape[:2]
        box = self.tracker.update(wrist_xy, w, h)
        crop = crop_roi(gray, box)
        self.crops.append(crop)
        if self.global_crops is not None:
            import cv2

            self.global_crops.append(cv2.resize(gray, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA))
        self.frames_seen += 1
        out = {"ready": False, "score": 0.0, "z": 0.0, "peak": False,
               "trigger": False, "fired": 0, "rejected": 0, "roi": None}
        if len(self.crops) < self.window:
            return out
        prob = self._score(list(self.crops))
        self.last_prob = prob
        self.prob_hist.append(prob)
        smooth = float(np.mean(self.prob_hist))
        # 在线 z-score：均值/方差只由过去帧更新，模拟部署时无标签校准
        self.mu = self.ema_decay * self.mu + (1 - self.ema_decay) * smooth
        diff = smooth - self.mu
        self.var = self.ema_decay * self.var + (1 - self.ema_decay) * diff * diff
        z = diff / math.sqrt(max(self.var, 1e-6))
        self.last_z = z
        self.z_hist.append(z)
        out.update({"ready": True, "score": smooth, "z": z, "roi": box})
        if len(self.z_hist) < 3:
            return out
        # 上一帧是否是局部峰（当前帧低于它，且它高于前一帧）
        zs = list(self.z_hist)
        prev, cur = zs[-2], zs[-1]
        peak = prev >= self.threshold_z and prev >= zs[-3] and prev > cur
        out["peak"] = peak
        if not peak:
            return out
        t = self.frames_seen / self.fps
        while self.peak_times and t - self.peak_times[0] > self.pair_max_s + 1.0:
            self.peak_times.popleft()
        self.peak_times.append(t)
        gap_ok = len(self.peak_times) >= 2 and self.pair_min_s <= t - self.peak_times[-2] <= self.pair_max_s
        if not gap_ok:
            return out
        if t - self.last_trigger_t < self.cooldown_s:
            return out
        gate_open = scenario_state is None or scenario_state == self.gate_state
        if not gate_open:
            self.rejected += 1
            out["rejected"] = self.rejected
            return out
        self.last_trigger_t = t
        self.triggers += 1
        self.fired += 1
        out.update({"trigger": True, "fired": 1})
        return out


def replay(take_mp4: str | Path, roi_npz: str | Path, model_path: str | Path,
           device: str = "cuda:0", gate_always_open: bool = False) -> dict:
    """离线回放：用已提取的 ROI 框（避免重跑 pose）逐帧喂给检测器。"""
    import cv2

    npz = np.load(roi_npz)
    boxes = npz["boxes"]
    fps = float(npz["fps"])
    cap = cv2.VideoCapture(str(take_mp4))
    signal = KnockRoiSignal(model_path, device=device)
    triggers = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        box = boxes[i] if i < len(boxes) else None
        wrist = None if box is None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        out = signal.update(gray, wrist, scenario_state=None if gate_always_open else MODE_GAZE_FIST)
        if out["trigger"]:
            triggers.append(round(i / fps, 2))
        i += 1
    cap.release()
    return {"frames": i, "triggers_s": triggers, "fired": signal.fired,
            "rejected_by_gate": signal.rejected}


def main() -> None:
    ap = argparse.ArgumentParser(description="ROI 敲击检测离线回放")
    ap.add_argument("--model", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-open", action="store_true", help="不做场景门控（对照）")
    args = ap.parse_args()
    import json

    print(json.dumps(replay(args.take, args.roi, args.model, args.device, args.gate_open),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()