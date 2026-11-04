# -*- coding: utf-8 -*-
"""episodes.py — 生成带**已知动作起点**的长序列（标定/验收用）

训练窗口是定长 12 帧；标定触发延迟需要「动作什么时候开始」这个真值，所以
这里拼长序列（几秒到十几秒），并记住每个动作段的起始帧号。

序列的成分与训练完全共用 :func:`synth.make_segment`，只有时长和拼接方式不同，
所以标定出来的阈值不会「只在标定集上好看」。

用法::

    eps = make_episode(bank, rng, user, [("idle", 1.5), ("raise", 0.5, "open_palm"),
                                         ("wave", 3.0), ("idle", 2.0)])
    eps.feat.shape        # (N, FEAT_DIM)
    eps.segments          # [{"kind","start","stop","cls"}]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synth import concat_blend, make_segment

__all__ = ["Episode", "make_episode", "two_wave_plan", "continuous_wave_plan"]

FPS = 30.0


@dataclass
class Episode:
    """一段带真值的长序列。

    Attributes:
        feat: (N, FEAT_DIM) float32。
        crop: (N, 32, 32) uint8。
        kp_rel: (N,17,2) float32 归一化骨架（喂老算法做对比用）。
        conf: (N,17) float32。
        segments: 每段 ``{"kind","cls","start","stop"}``，``start`` 含、``stop`` 不含。
        name: 便于报告里引用。
    """

    feat: np.ndarray
    crop: np.ndarray
    kp_rel: np.ndarray = None
    conf: np.ndarray = None
    segments: list = field(default_factory=list)
    name: str = ""

    def __len__(self):
        return len(self.feat)

    def starts_of(self, kind):
        """所有某类动作段的起始帧号（触发延迟的真值）。"""
        return [s["start"] for s in self.segments if s["kind"] == kind]


def make_episode(bank, rng, user, plan, name="", fps=FPS, blend=3,
                 freq_hz=None, amp=None):
    """按 plan 拼一段长序列。

    Args:
        bank: :class:`synth.PoseBank`。
        rng: ``np.random.Generator``。
        user: 用户编号。
        plan: ``[(kind, seconds, cls_name?)...]``；kind ∈ idle/hold/wave/raise。
        name: 序列名（写报告用）。
        fps: 帧率。
        blend: 段间过渡帧数。
        freq_hz: 挥手频率（None 则每段随机抽）。
        amp: 挥手幅度（None 则每段随机抽）。

    Returns:
        :class:`Episode`
    """
    feats, crops, kps, cfs, segs = [], [], [], [], []
    t_wave_off = 0.0
    n_done = 0
    for item in plan:
        kind = item[0]
        seconds = float(item[1])
        cls_name = item[2] if len(item) > 2 else None
        n = max(1, int(round(seconds * fps)))
        kw = {"fps": fps}
        if kind == "wave":
            kw["t_offset"] = t_wave_off
            t_wave_off += seconds
            if freq_hz is not None:
                kw["freq_hz"] = float(freq_hz)
            if amp is not None:
                kw["amp"] = float(amp)
        if cls_name is not None:
            kw["cls_name"] = cls_name
        f, c, info = make_segment(bank, rng, user, kind, n, **kw)
        feats.append(f)
        crops.append(c)
        kps.append(info["kp_rel"])
        cfs.append(info["conf"])
        segs.append({"kind": kind, "cls": cls_name, "start": n_done,
                     "stop": n_done + n})
        n_done += n

    feat = np.concatenate(feats, axis=0)
    crop = np.concatenate(crops, axis=0)
    kp_rel = np.concatenate(kps, axis=0)
    conf = np.concatenate(cfs, axis=0)
    _blend_joins(feat, crop, [s["start"] for s in segs], blend)
    return Episode(feat=feat, crop=crop, kp_rel=kp_rel, conf=conf,
                   segments=segs, name=name)


def _blend_joins(feat, crop, starts, blend):
    """段接缝处把新段的头几帧向上一段的最后一帧拉过去（原地改）。

    不做这一步，``idle → raise`` 处会出现「手瞬间出现在头顶」的瞬移，
    那是有史以来最容易让时序模型学到假特征的东西。
    """
    if blend <= 0:
        return
    n = len(feat)
    for s in starts:
        if s <= 0 or s >= n:
            continue
        prev = feat[s - 1].copy()
        prev_c = crop[s - 1].copy()
        k = min(blend, n - s)
        for i in range(k):
            w = (i + 1.0) / (k + 1.0)
            feat[s + i] = (1.0 - w) * prev + w * feat[s + i]
            crop[s + i] = ((1.0 - w) * prev_c.astype(np.float32)
                           + w * crop[s + i].astype(np.float32)).astype(np.uint8)


def continuous_wave_plan(freq_hz=1.0, wave_s=4.0):
    """一次连续挥手：抬手 → 挥手 4 s。"""
    return [("idle", 1.2), ("raise", 0.5, "open_palm"), ("wave", wave_s),
            ("idle", 1.5)]


def two_wave_plan(gap_s=1.0, wave_s=1.5, freq_hz=1.5):
    """两次挥手，中间隔 ``gap_s`` 秒（手放下的 idle 段）。"""
    return [("idle", 1.0), ("raise", 0.4, "open_palm"), ("wave", wave_s),
            ("idle", gap_s), ("raise", 0.4, "open_palm"), ("wave", wave_s),
            ("idle", 1.2)]
