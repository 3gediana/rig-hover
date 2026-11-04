# -*- coding: utf-8 -*-
"""esc_baselines.py — 三条"不用 CfC 隐状态"的逐帧/手工时序基线。

验收里"CfC 头 vs logistic"必须打得公平，所以基线不是一个软柿子：

``B0  logistic_frame``
    ``pc/decision_hub`` 的可观测升级判据本身：``σ(b + Σ w_k φ_k(x_t))``。
    只用**当前帧**的势函数，这正是要被替换的那套逻辑。

``B1  logistic_ema``
    人工给每个势函数加一阶平滑（``τ`` 取验证集上最好的一档）。这代表
    "不改架构、只给判据补一个泄漏积分器"的常见做法。

``B2  linear_temporal``
    在 ``[φ_t, EMA_5(φ), EMA_20(φ), Δ8(φ)]``（48 维）上**逻辑回归**，用与 CfC
    完全相同的训练标签拟合。这是"把时序特征手工摊开喂给线性模型"的强基线：
    如果 CfC 打不过它，那 CfC 的价值就只剩"不用手写特征"这一条。

三者都不看隐状态，也都拿不到 106 维观测 —— 它们只看 12 个势函数的历史，
所以是"独立 logistic + 手写时序"的最强形态。
"""

from __future__ import annotations

import numpy as np

import esc_core as C

F32 = np.float32


# --------------------------------------------------------------------------- #
# 逐帧 / 平滑
# --------------------------------------------------------------------------- #
def ema_along_time(x, tau_frames):
    """沿时间轴的一阶指数平滑（``(N,T,...)``）。``tau_frames<=0`` 时原样返回。

    Args:
        x: ``(N,T)`` 或 ``(N,T,K)``。
        tau_frames: 时间常数（帧）。
    """
    a = 1.0 if tau_frames <= 0 else 1.0 / (1.0 + float(tau_frames))
    y = np.empty_like(x, dtype=np.float64)
    xd = np.asarray(x, dtype=np.float64)
    prev = np.zeros(xd.shape[:1] + xd.shape[2:], dtype=np.float64)
    for t in range(xd.shape[1]):
        prev = prev + a * (xd[:, t] - prev)
        y[:, t] = prev
    return y


def delta_along_time(x, k=8):
    """``Δk``：``x[t] - x[t-k]``（前 k 帧补 0）。"""
    d = np.zeros_like(x, dtype=np.float64)
    if x.shape[1] > k:
        d[:, k:] = x[:, k:] - x[:, :-k]
    return d


def logistic_frame(phi, weights=C.GATE_WEIGHTS, bias=C.GATE_BIAS):
    """B0：逐帧判据。``phi`` 为 ``(...,12)``。"""
    return C.p_esc_from_phi(phi, weights, bias)


def logistic_ema(phi, tau_frames, weights=C.GATE_WEIGHTS, bias=C.GATE_BIAS):
    """B1：把势函数先做 ``tau_frames`` 的一阶平滑再打分。"""
    return logistic_frame(ema_along_time(phi, tau_frames), weights, bias)


# --------------------------------------------------------------------------- #
# B2：手工时序特征 + 逻辑回归（在训练集上拟合）
# --------------------------------------------------------------------------- #
def temporal_features(phi):
    """``(N,T,12)`` -> ``(N,T,48)``：``[φ, EMA5(φ), EMA20(φ), Δ8(φ)]``。"""
    phi = np.asarray(phi, dtype=np.float64)
    feats = np.concatenate([phi,
                            ema_along_time(phi, 5.0),
                            ema_along_time(phi, 20.0),
                            delta_along_time(phi, 8)], axis=-1)
    return feats


class LinearTemporal:
    """在 :func:`temporal_features` 上拟合的二分类逻辑回归（numpy 实现）。

    Args:
        l2: L2 正则（作用在标准化后的权重上）。
        iters: 全批量梯度下降轮数。
        lr: 学习率。
    """

    def __init__(self, l2=1e-3, iters=600, lr=0.5, seed=0):
        self.l2 = float(l2)
        self.iters = int(iters)
        self.lr = float(lr)
        self.seed = int(seed)
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, feats, y, verbose=False):
        """``feats (N,T,K)``、``y (N,T)`` -> self（内部展平）。"""
        f = np.asarray(feats, dtype=np.float64).reshape(-1, feats.shape[-1])
        t = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mu = f.mean(axis=0)
        self.sd = np.maximum(f.std(axis=0), 1e-6)
        z = (f - self.mu) / self.sd
        n, k = z.shape
        w = np.zeros(k)
        b = float(np.log(max(t.mean(), 1e-3) / max(1.0 - t.mean(), 1e-3)))
        for it in range(self.iters):
            logit = np.clip(z @ w + b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-logit))
            g = p - t
            gw = z.T @ g / n + self.l2 * w
            gb = float(g.mean())
            w -= self.lr * gw
            b -= self.lr * gb
            if verbose and it % 100 == 0:
                nll = -np.mean(t * np.log(np.clip(p, 1e-9, 1)) +
                               (1 - t) * np.log(np.clip(1 - p, 1e-9, 1)))
                print("    [lin] it=%d nll=%.4f" % (it, nll))
        self.w, self.b = w, b
        return self

    def predict(self, feats):
        f = np.asarray(feats, dtype=np.float64).reshape(-1, feats.shape[-1])
        z = (f - self.mu) / self.sd
        logit = np.clip(z @ self.w + self.b, -30, 30)
        return (1.0 / (1.0 + np.exp(-logit))).reshape(feats.shape[:2])

    def logit(self, feats):
        f = np.asarray(feats, dtype=np.float64).reshape(-1, feats.shape[-1])
        z = (f - self.mu) / self.sd
        return (z @ self.w + self.b).reshape(feats.shape[:2])


def best_ema_tau(phi, y, taus=(1.0, 2.0, 5.0, 10.0, 20.0, 40.0)):
    """在验证集上挑一个最好的 ``tau``（B1 的超参必须在 val 上定）。"""
    best, best_auc = None, -1.0
    for tau in taus:
        p = logistic_ema(phi, tau)
        auc = C.roc_auc(np.asarray(y).reshape(-1) > 0.5, np.asarray(p).reshape(-1))
        if auc > best_auc:
            best, best_auc = float(tau), auc
    return best, best_auc
