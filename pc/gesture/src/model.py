# -*- coding: utf-8 -*-
"""model.py — 事件级手势分类器（骨架 1D-CNN + 可选手部外观分支）

结构很小（约 18 万参数，CPU 单帧 <1 ms），因为它只负责「这一小段动作是什么」，
时序确认、迟滞、事件边沿全部交给 ``gesture_events.py``。

输入两路：

* ``feat``  (B,T,FEAT_DIM)：归一化骨架关键点 + 置信度，1D 空洞卷积（膨胀 1/2/4）
  看 T 帧的运动。膨胀卷积的感受野覆盖整个窗口，用来区分「挥手（来回摆）」
  和「举着不动（姿势一样但不动）」。**这是主分支。**
* ``crop``  (B,T,1,32,32)：手部灰度小图，一个两层小 CNN 抽手形。

**为什么还要外观分支**：COCO-17 没有手指关键点，``fist``/``open_palm``/
``pointing``/``thumbs_up`` 四类在骨架上是**同一个姿势**（手举起来、腕在肩上方）。
纯骨架分支对这四个类的上界就是「分不开」，这一点用 ``--skeleton-only`` 的消融
数字证明（见 reports/classifier.md）。外观分支只有 ~5 万参数、24×24 量级的输入，
在 240×240 上仍然可靠，因为裁剪框是由肩宽归一化定位的，与分辨率/距离无关。
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GestureNet", "build_model", "MODEL_DEFAULTS"]


MODEL_DEFAULTS = {
    "feat_dim": 53,
    "n_classes": 6,
    "width": 96,
    "dilations": [1, 2, 4],
    "app_dim": 64,
    "crop_px": 32,
    "pool_last": 4,
    "dropout": 0.10,
    "use_appearance": True,
}


def _gn(ch, groups=8):
    return nn.GroupNorm(min(groups, ch), ch)


class SkeletonBranch(nn.Module):
    """逐帧线性 + 空洞因果卷积堆叠，输出每帧的宽度 width 的隐状态。"""

    def __init__(self, feat_dim, width, dilations):
        super().__init__()
        layers = [nn.Conv1d(feat_dim, width, 3, padding=1), _gn(width),
                  nn.SiLU()]
        for d in dilations:
            layers += [nn.Conv1d(width, width, 3, padding=d, dilation=d),
                       _gn(width), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.width = width

    def forward(self, feat):
        """feat: (B,T,F) → (B,width,T)"""
        return self.net(feat.transpose(1, 2))


class AppearanceBranch(nn.Module):
    """单帧手部裁剪 → app_dim 维。输入 32×32 灰度（归一化到 [-1,1]）。"""

    def __init__(self, app_dim, crop_px):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), _gn(16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), _gn(32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), _gn(48), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(48 * (crop_px // 8) * (crop_px // 8), app_dim),
            nn.SiLU(),
        )

    def forward(self, crop):
        """crop: (B*T,1,32,32) → (B*T,app_dim)"""
        return self.net(crop)


class GestureNet(nn.Module):
    """(骨架窗口, 手部裁剪窗口) → 6 类 logits。

    Args:
        cfg: 覆盖 :data:`MODEL_DEFAULTS` 的键值。
    """

    def __init__(self, **cfg):
        super().__init__()
        c = dict(MODEL_DEFAULTS)
        c.update(cfg)
        self.cfg = c
        self.use_appearance = bool(c["use_appearance"])
        self.skel = SkeletonBranch(c["feat_dim"], c["width"], c["dilations"])
        self.app = (AppearanceBranch(c["app_dim"], c["crop_px"])
                    if self.use_appearance else None)
        head_in = c["width"] * 2 + (c["app_dim"] if self.use_appearance else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 96), nn.SiLU(), nn.Dropout(float(c["dropout"])),
            nn.Linear(96, c["n_classes"]),
        )
        self.pool_last = int(c["pool_last"])

    def forward(self, feat, crop=None):
        """Args:
            feat: (B,T,53) float32。
            crop: (B,T,1,32,32) float32 ∈[-1,1]；``use_appearance`` 为 False 时可省。

        Returns:
            (B,n_classes) logits。
        """
        h = self.skel(feat)                       # (B,W,T)
        T = h.shape[-1]
        k = min(self.pool_last, T)
        hs = torch.cat([h[..., -k:].mean(dim=-1), h.max(dim=-1).values], dim=1)
        if not self.use_appearance:
            return self.head(hs)
        if crop is None:
            raise ValueError("use_appearance=True 时必须给 crop")
        B, Tw = crop.shape[0], crop.shape[1]
        a = self.app(crop.reshape(B * Tw, 1, *crop.shape[-2:]))
        a = a.reshape(B, Tw, -1)[:, -k:, :].mean(dim=1)      # 时间上做平均去噪
        return self.head(torch.cat([hs, a], dim=1))


def build_model(**cfg):
    """按配置建模型（其余键透传，未知键会被忽略——配置里带有窗口长度等元信息）。"""
    keep = {k: v for k, v in cfg.items() if k in MODEL_DEFAULTS}
    return GestureNet(**keep)
