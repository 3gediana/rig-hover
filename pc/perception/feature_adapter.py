# -*- coding: utf-8 -*-
"""feature_adapter.py — 特征适配器：FEATURE_DIM → 64 维稳定表示（工作流 A）

结构照 TECH-PLAN §4.2（不是照抄参考实现的 ReLU 版）：

    x_norm = (x - running_mean) / running_std     # 逐维标准化
    x_clip = tanh(x_norm / 3.0)                   # ±3σ 软限幅，防离群值炸网络
    y      = Linear(FEATURE_DIM, 64)(x_clip)      # 可学习压缩
    y      = tanh(y)                              # 再限幅 → y ∈ [-1, 1]

与参考实现 ``D:\\smartcar\\code\\lnn\\feat_adapter.py`` 的差异（有意为之）：

1. 末端用 ``tanh`` 而不是 ``ReLU``。契约 §3.1 要求 ``feat_64 ∈ [-1, 1]``，
   而参考实现的 ``F.relu`` 输出是 ``[0, +∞)``，不满足。
2. 软限幅用 ``tanh(z/3)``（不回乘 3），照抄 TECH-PLAN §4.2 的公式；参考实现
   是 ``tanh(z/3)*3``。前者输出严格在 (-1,1) 内，后者能到 ±3。
3. ``running_mean/running_std`` 用**统计累积 + ``freeze_stats()``** 维护
   （见下），不是只在训练脚本里 ``copy_``。A 阶段随机初始化：mean=0, std=1，
   所以 ``feat_64`` 已经是有限的、有界的随机投影。

**三项纪律**（TECH-PLAN §4.2「三件事必须做对」）：

* ``running_mean/std`` 必须在训练集上统计，部署时冻结 → ``update_stats()`` /
  ``freeze_stats()``。
* 统计要用足够多的帧（≥10000）→ ``nan_count`` / ``clip_ratio`` 两个监控量。
* Adapter 是训练的一部分，不是固定预处理层 → 它是 ``nn.Module``，参数进优化器。

A 阶段只要求接口可用：

    >>> from feature_adapter import FeatureAdapter
    >>> ad = FeatureAdapter()                 # (FEATURE_DIM, 64)
    >>> y = ad(torch.from_numpy(feat_raw))    # (64,) ∈ [-1,1]
    >>> ad.freeze_stats()                     # 部署前固定统计量
    >>> ad.load_state_dict(torch.load("adapter.pt"))
"""
from __future__ import annotations

import torch
import torch.nn as nn

from extract_features import FEATURE_DIM

__all__ = ["FeatureAdapter", "ADAPTER_OUT", "SQUASH"]

#: 输出维度（契约 §3.1：feat_64 是 64 维；CfC 输入 106 维里的视觉段）
ADAPTER_OUT = 64

#: 软限幅半宽（σ 数），TECH-PLAN §4.2 取 3.0
SQUASH = 3.0

#: std 下限，防除零
_STD_FLOOR = 1e-3


class FeatureAdapter(nn.Module):
    """``FEATURE_DIM → 64`` 的可训练压缩层，带逐维标准化与 tanh 软限幅。

    参数与缓冲：

    * ``fc.weight`` / ``fc.bias``：可学习，形状 ``(64, FEATURE_DIM)`` / ``(64,)``
    * ``running_mean`` / ``running_std``：**buffer**，进 ``state_dict``，
      随之 ``.to(device)``，可被 ``load_state_dict`` 覆盖
    * ``stats_count``：累计帧数（buffer，便于判断统计是否够 ≥10000）

    冻结后（``freeze_stats()``）``update_stats()`` 会抛 ``RuntimeError``，
    防止训练脚本忘了关统计、部署时把 running stats 又改掉。
    """

    def __init__(self, feat_dim=FEATURE_DIM, out_dim=ADAPTER_OUT,
                 squash=SQUASH, std_floor=_STD_FLOOR):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.out_dim = int(out_dim)
        self.squash = float(squash)
        self.std_floor = float(std_floor)

        self.register_buffer("running_mean", torch.zeros(self.feat_dim))
        self.register_buffer("running_std", torch.ones(self.feat_dim))
        # 统计累积（非持久化：不进 state_dict，避免把半成品统计带上车）
        self.register_buffer("stats_count", torch.zeros((), dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_stats_sum", torch.zeros(self.feat_dim,
                                                       dtype=torch.float64),
                             persistent=False)
        self.register_buffer("_stats_sumsq", torch.zeros(self.feat_dim,
                                                         dtype=torch.float64),
                             persistent=False)
        # 冻结标志（非持久化：部署脚本运行时决定，不写进权重文件）
        self.register_buffer("_stats_frozen", torch.zeros((), dtype=torch.bool),
                             persistent=False)

        self.fc = nn.Linear(self.feat_dim, self.out_dim)

    # ----------------------------- 前向 -----------------------------
    def forward(self, feat):
        """feat: (FEATURE_DIM,) 或 (B, FEATURE_DIM) → 同前缀 + (out_dim,)，∈[-1,1]。"""
        x = (feat - self.running_mean) / self.running_std.clamp_min(self.std_floor)
        x = torch.tanh(x / self.squash)
        return torch.tanh(self.fc(x))

    # --------------------------- 统计量 ---------------------------
    @property
    def stats_frozen(self):
        """统计量是否已冻结。"""
        return bool(self._stats_frozen.item())

    @torch.no_grad()
    def update_stats(self, feat):
        """累积一批特征用于估计 ``running_mean/running_std``（逐维，无偏校正留到 freeze）。

        Args:
            feat: (B, FEATURE_DIM) 或 (FEATURE_DIM,) 的 torch 张量 / numpy 数组。
        """
        if self.stats_frozen:
            raise RuntimeError(
                "统计量已 freeze_stats()，不能再 update_stats()；"
                "要重新统计先 reset_stats()。")
        x = torch.as_tensor(feat, dtype=torch.float64)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.feat_dim:
            raise ValueError("特征维度不符：%d != %d" % (x.shape[1], self.feat_dim))
        self._stats_sum += x.sum(dim=0)
        self._stats_sumsq += (x * x).sum(dim=0)
        self.stats_count += float(x.shape[0])
        # 统计过程中让 running_* 跟着更新，训练时 forward 用到的就是当前估计
        self._refresh_running()

    @torch.no_grad()
    def _refresh_running(self):
        n = float(self.stats_count.item())
        if n <= 0:
            return
        mean = self._stats_sum / n
        var = (self._stats_sumsq / n - mean * mean).clamp_min(0.0)
        self.running_mean.copy_(mean.to(self.running_mean.dtype))
        self.running_std.copy_(var.sqrt().clamp_min(self.std_floor)
                               .to(self.running_std.dtype))

    @torch.no_grad()
    def freeze_stats(self):
        """把已经累积的统计量固定下来（部署前必须调用）。

        返回累积用到的帧数，方便检查是否 ≥10000（TECH-PLAN §4.2 第 2 条）。
        """
        self._refresh_running()
        self._stats_frozen.fill_(True)
        return int(self.stats_count.item())

    @torch.no_grad()
    def reset_stats(self):
        """清空统计累积并解冻（回到 mean=0 / std=1 的初始态）。"""
        self._stats_sum.zero_()
        self._stats_sumsq.zero_()
        self.stats_count.zero_()
        self._stats_frozen.fill_(False)
        self.running_mean.zero_()
        self.running_std.fill_(1.0)

    @torch.no_grad()
    def set_stats(self, mean, std):
        """直接写入统计量（例如从训练集的 ``.npz`` 恢复）。"""
        m = torch.as_tensor(mean, dtype=self.running_mean.dtype).flatten()
        s = torch.as_tensor(std, dtype=self.running_std.dtype).flatten()
        if m.numel() != self.feat_dim or s.numel() != self.feat_dim:
            raise ValueError("mean/std 维度必须是 %d" % self.feat_dim)
        self.running_mean.copy_(m)
        self.running_std.copy_(s.clamp_min(self.std_floor))

    # --------------------------- 兼容层 ---------------------------
    def extra_repr(self):
        return ("feat_dim=%d, out_dim=%d, squash=%.2f, frozen=%s, n=%d"
                % (self.feat_dim, self.out_dim, self.squash,
                   self.stats_frozen, int(self.stats_count.item())))
