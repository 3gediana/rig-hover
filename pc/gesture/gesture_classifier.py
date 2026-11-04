# -*- coding: utf-8 -*-
"""gesture_classifier.py — 部署用的增量手势分类器（骨架窗口 + 手部裁剪）

职责只有一件事：**维护最近 T 帧的滑动窗口并输出 (6,) 概率**。
事件边界、迟滞、去抖全部在 :mod:`gesture_events` 里，这里不掺和。

用法::

    >>> clf = GestureClassifier.load()          # models/gesture_classifier.pt
    >>> clf.reset()
    >>> probs = clf.push_kp(kp_xy, kp_conf, frame_bgr)   # 每帧一次
    >>> out = events.update(probs, dt)                    # 交给事件机

关键约定：

* **窗口是因果的**：第 t 帧的输出只用到 t-T+1..t 帧，所以 onset 的延迟里
  天然含 T/2 左右的窗口填充时间，这部分在标定指标里已经算进去了。
* **人不在 / 肩宽太小**时用 :meth:`push_invalid`：返回 one-hot idle 并**冻结**
  窗口（重复上一帧），不要往窗口里塞零向量——那会让模型看到一段「不存在的
  姿势」，容易在下一帧吐出假手势。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gesture_common import (CLASSES, CROP_PX, FEAT_DIM,  # noqa: E402
                            MODEL_CLASSES, build_frame_features, hand_crop)
from model import build_model  # noqa: E402

__all__ = ["GestureClassifier", "MODEL_PATH", "DEFAULT_WINDOW", "fold_probs"]

MODEL_PATH = os.path.join(HERE, "models", "gesture_classifier.pt")

#: 与训练窗口一致：12 帧 @30fps ≈ 400 ms 的因果窗口
DEFAULT_WINDOW = 12


def fold_probs(p_model, model_classes=None):
    """把分类器内部概率折成对外 6 类：``other`` 的概率并进 ``idle``。

    ``other`` 是内部类（HaGRID 里不关心但真实存在的 12 类手势）。对外契约
    只有 6 个 gesture_id，未知手势应当表现为「没有手势」，而不是随便认一个。

    Args:
        p_model: (N,) 或 (B,N) 内部概率，类别序为 ``MODEL_CLASSES``。
        model_classes: 内部类名；默认 ``MODEL_CLASSES``。

    Returns:
        同形状前维、最后一维为 6 的概率。
    """
    import numpy as _np

    p = _np.asarray(p_model, dtype=_np.float64)
    names = tuple(model_classes or MODEL_CLASSES)
    if p.shape[-1] == len(CLASSES):
        return p
    if p.shape[-1] != len(names):
        raise ValueError("概率维度 %d 与类别数 %d 不符"
                         % (p.shape[-1], len(names)))
    out = _np.empty(p.shape[:-1] + (len(CLASSES),), dtype=_np.float64)
    out[..., 0] = p[..., names.index("idle")] + p[..., names.index("other")]
    for k, name in enumerate(CLASSES[1:], start=1):
        out[..., k] = p[..., names.index(name)]
    return out


class GestureClassifier:
    """有状态的滑窗分类器。

    Args:
        model_path: 权重路径（``torch.save`` 的 dict，见 :func:`save_checkpoint`）。
        device: ``"cpu"`` / ``"cuda:0"``。
        classes: 对外类别名（默认 6 类，必须与 ``CLASSES`` 一致）。
    """

    def __init__(self, model_path=MODEL_PATH, device="cpu", classes=None):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model_cfg = dict(ckpt.get("model_cfg") or {})
        self.meta = dict(ckpt.get("meta") or {})
        self.classes = tuple(classes or self.meta.get("classes") or CLASSES)
        self.model_classes = tuple(self.meta.get("model_classes")
                                   or MODEL_CLASSES)
        self.window = int(self.meta.get("window", DEFAULT_WINDOW))
        self.crop_px = int(self.meta.get("crop_px", CROP_PX))
        self.feat_dim = int(self.meta.get("feat_dim", FEAT_DIM))
        self.fps = float(self.meta.get("fps", 30.0))
        self.device = torch.device(device if (
            device != "cuda:0" or torch.cuda.is_available()) else "cpu")
        self.model = build_model(**self.model_cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.use_appearance = bool(self.model.use_appearance)
        self.n_infer = 0
        self.reset()

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_path=MODEL_PATH, device="cpu", classes=None):
        """从权重文件构造。"""
        return cls(model_path=model_path, device=device, classes=classes)

    def reset(self):
        """清空滑窗（换人、断流重连后调用）。"""
        self._feat = np.zeros((self.window, self.feat_dim), dtype=np.float32)
        self._crop = np.zeros((self.window, self.crop_px, self.crop_px),
                              dtype=np.uint8)
        self.n_frames = 0
        self.n_infer = 0
        self._last_valid = False

    # ------------------------------------------------------------------
    def _push_buffers(self, feat, crop):
        self._feat = np.roll(self._feat, -1, axis=0)
        self._crop = np.roll(self._crop, -1, axis=0)
        self._feat[-1] = feat
        self._crop[-1] = crop
        self.n_frames += 1
        if self.n_frames < self.window:
            # 窗口未填满时用第一帧铺满（等价于「这人一直保持这个姿势」），
            # 比用零向量好：零向量在训练分布里不存在。
            k = self.window - self.n_frames
            self._feat[:k] = feat
            self._crop[:k] = crop

    def _infer(self):
        x = torch.from_numpy(self._feat[None]).to(self.device)      # (1,T,F)
        if self.use_appearance:
            c = (torch.from_numpy(self._crop[None].astype(np.float32))
                 .to(self.device) / 127.5 - 1.0)                    # (1,T,32,32)
            c = c.unsqueeze(2)                                      # (1,T,1,32,32)
        else:
            c = None
        with torch.no_grad():
            logits = self.model(x, c)
            probs = torch.softmax(logits, dim=-1)[0]
        self.n_infer += 1
        p = probs.detach().float().cpu().numpy().astype(np.float64)
        return fold_probs(p, self.model_classes)

    # ------------------------------------------------------------------
    def push_kp(self, kp_xy, kp_conf=None, frame_bgr=None, img_wh=None,
                shoulder_min_px=5.0):
        """喂一帧关键点（像素坐标），返回 (6,) 概率。

        Args:
            kp_xy: (17,2) 关键点像素坐标。
            kp_conf: (17,) 置信度，None 视为全 1。
            frame_bgr: HxWx3 uint8；给了才裁手部图（有外观分支时必须给）。
            img_wh: (w,h)，用于「肩宽/图宽」特征；None 时用图本身推断。
            shoulder_min_px: 肩宽下限，低于此值走 :meth:`push_invalid`。

        Returns:
            (6,) float64 概率（和为 1）。
        """
        if frame_bgr is not None and img_wh is None:
            img_wh = (frame_bgr.shape[1], frame_bgr.shape[0])
        feat = build_frame_features(kp_xy, kp_conf,
                                    shoulder_min_px=shoulder_min_px,
                                    img_wh=img_wh)
        if feat[-1] < 0.5:
            return self.push_invalid()
        if frame_bgr is not None:
            crop = hand_crop(frame_bgr, kp_xy, kp_conf, crop_px=self.crop_px,
                             shoulder_min_px=shoulder_min_px)
        elif self.use_appearance:
            raise ValueError("外观分支需要 frame_bgr 才能裁手部图")
        else:
            crop = np.zeros((self.crop_px, self.crop_px), dtype=np.uint8)
        self._push_buffers(feat, crop)
        self._last_valid = True
        return self._infer()

    def push_feat(self, feat, crop=None):
        """直接喂事先算好的特征（离线评测/单测用，跳过像素→特征那一步）。"""
        feat = np.asarray(feat, dtype=np.float32).reshape(-1)
        if feat.shape[0] != self.feat_dim:
            raise ValueError("feat 维度应为 %d，收到 %d"
                             % (self.feat_dim, feat.shape[0]))
        if crop is None:
            crop = np.zeros((self.crop_px, self.crop_px), dtype=np.uint8)
        self._push_buffers(feat, np.asarray(crop, dtype=np.uint8))
        self._last_valid = True
        return self._infer()

    def push_invalid(self):
        """无人 / 骨架不可信：冻结窗口，输出 one-hot idle。"""
        if self._last_valid and self.n_frames > 0:
            self._push_buffers(self._feat[-1].copy(), self._crop[-1].copy())
        else:
            self._push_buffers(np.zeros(self.feat_dim, np.float32),
                               np.zeros((self.crop_px, self.crop_px), np.uint8))
        self._last_valid = False
        p = np.zeros(len(self.classes), dtype=np.float64)
        p[0] = 1.0
        return p

    # ------------------------------------------------------------------
    def run_episode(self, feat, crop, dt=1.0 / 30.0):
        """把一整段特征跑成概率矩阵 (N, n_classes)（离线用）。"""
        out = np.empty((len(feat), len(self.classes)), dtype=np.float64)
        for i in range(len(feat)):
            out[i] = self.push_feat(feat[i], crop[i])
        return out

    def __repr__(self):
        return ("GestureClassifier(window=%d, feat_dim=%d, appearance=%s, "
                "device=%s)" % (self.window, self.feat_dim,
                                self.use_appearance, self.device))


def save_checkpoint(path, model, model_cfg, meta):
    """统一存档格式：``state_dict`` + 建模配置 + 元信息。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg,
                "meta": meta}, path)


if __name__ == "__main__":                       # 冒烟：能加载、能推理
    import numpy as np

    clf = GestureClassifier.load(device="cpu")
    print(clf)
    z = os.path.join(HERE, "data", "synth", "test.npz")
    if os.path.exists(z):
        d = np.load(z)
        clf.reset()
        for i in range(12):                      # 填满一整个窗口
            p = clf.push_feat(d["feat"][0, i], d["crop"][0, i])
        print("probs:", np.round(p, 4), "sum: %.4f" % float(p.sum()))
        print("（上面是测试集第一个窗口的 12 帧，最后一个概率即当前输出）")
    else:
        print("（没有 data/synth/test.npz，跳过推理冒烟）")
