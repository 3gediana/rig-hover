# -*- coding: utf-8 -*-
"""metrics.py — 混淆矩阵 / 每类 P·R·F1（不依赖 sklearn，便于逐字复现）"""
from __future__ import annotations

import numpy as np

__all__ = ["confusion", "per_class_report", "macro_f1"]


def confusion(y_true, y_pred, n_classes):
    """(n_classes, n_classes) 混淆矩阵，行=真值，列=预测。"""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_report(y_true, y_pred, classes):
    """每类 precision/recall/F1/support。

    Returns:
        list[dict]，顺序同 ``classes``。
    """
    n = len(classes)
    cm = confusion(y_true, y_pred, n)
    out = []
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if prec + rec else 0.0
        out.append({"class": name, "precision": prec, "recall": rec,
                    "f1": f1, "support": int(cm[i, :].sum())})
    return out


def macro_f1(y_true, y_pred, classes):
    """宏平均 F1（每类等权，不被 idle 的大样本量淹没）。"""
    rep = per_class_report(y_true, y_pred, classes)
    return float(np.mean([r["f1"] for r in rep])) if rep else 0.0
