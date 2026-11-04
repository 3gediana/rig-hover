# -*- coding: utf-8 -*-
"""extract_features.py — YOLO26 内部特征抽取（工作流 A / PC 端感知链）

权重：``yolo26n-pose.pt``。
    这是 ultralytics 8.4.21 的 ``GITHUB_ASSETS_NAMES`` 里实际存在的名字（已核对），
    联网时由 ``YOLO()`` 自动下载。本目录 ``assets/`` 已经放了一份（7.5MB），
    ``resolve_model_path()`` 会优先命中它，所以从任何工作目录起都不会重复下载。
    没有用 ``yolo26n.pt``：本链路需要 COCO-17 关键点算手势，检测权重拿不到关键点，
    多跑一次模型更贵。

输入：**320×320**（整除 32），letterbox 由 ultralytics 的 predict 或本文件的
``letterbox_bgr`` 完成。特征图因此是 40/20/10，而不是 640 输入下的 80/40/20。

层索引**不写死**：从网络末层（Pose26）的 ``f`` 属性读出 P3/P4/P5 的层号
（实测 ``[16, 19, 22]``），避免换权重大小后静默取错层。

特征来源与维度（实测，见 ``FEATURE_SOURCES``）：

| 键  | 层                                          | 形状(1,C,H,W) | 池化方式           | 维度 |
|-----|---------------------------------------------|---------------|--------------------|------|
| P3  | layers[16] C3k2（P3 neck）                   | (1,64,40,40)  | GAP                | 64   |
| P4  | layers[19] C3k2（P4 neck）                   | (1,128,20,20) | GAP                | 128  |
| P5  | layers[22] C3k2（P5 neck）                   | (1,256,10,10) | GAP                | 256  |
| CLS | layers[23] Pose26.one2one_cv3[1]（P4 分类）  | (1,1,20,20)   | sigmoid + 2×2 池化  | 100  |

⚠️ **CLS 抓的是 ``one2one_cv3[1]``，不是参考实现里的 ``cv3[1]``。** 原因：ultralytics 的
``AutoBackend`` 在第一次 ``predict`` 时会调用 ``Pose26.fuse()``，而
``Detect.fuse()`` 直接 ``self.cv2 = self.cv3 = None``（删掉 one2many 头做推理优化）。
挂 ``cv3[1]`` 的 hook 在 predict 路径上**永远不会触发**（实测 RuntimeError）。YOLO26
是 end2end 结构，推理真正用的是 ``one2one`` 分支，所以抓它才是"推理时真实参与决策的
分类热图"。没有 ``end2end`` 的旧结构（如 yolo11-pose）自动退回 ``cv3[1]``。

合计 **548** 维。和契约/技术方案里写的 920 不一样，原因有两个，都不是实现问题：

1. 参考实现算的是 **6 类检测头** 的热图（6×20×20 → 池化 → 6×10×10 = 600 维）。
   我们用的是 **pose 权重，nc=1**，``cv3[1]`` 只有 1 个通道，池化后是 1×10×10=100 维。
2. 参考实现钩了 P4 但**没有拼进特征**（只用了 cv3[1] 的热图 + P3 + P5）。
   契约 §3.2 明写「抓 P3/P4/P5/Detect.cv3[1]」，所以这里的 P4 GAP 是**真的拼进去**的。

**没有为了凑 920 做任何 padding**，FEATURE_DIM 就是实测值，``_fuse()`` 每次都会
校验并报错，不允许静默变形。

本模块不 import torch / ultralytics 到模块级（只在函数和类内部 import），这样
``from extract_features import FEATURE_DIM`` 是瞬时完成的，feature_adapter 可以直接用。
"""
from __future__ import annotations

import os

import numpy as np

__all__ = [
    "MODEL_NAME", "IMGSZ", "FEATURE_DIM", "FEATURE_SOURCES",
    "YoloFeatureExtractor", "letterbox_bgr", "letterbox_params",
    "describe_sources", "resolve_model_path",
]

_HERE = os.path.dirname(os.path.abspath(__file__))

#: 实际使用的权重名（ultralytics 8.4.21 资产名，与契约 §3.1 默认值一致）
MODEL_NAME = "yolo26n-pose.pt"

#: 网络输入边长（整除 32）
IMGSZ = 320

#: 特征维度（实测值，不是预设值）。yolo26n-pose.pt + 320×320 + 上面四个来源。
FEATURE_DIM = 548

#: 每个来源的说明：(键, 层描述, 单张图的形状, 维度, 池化操作)
#: 这里是 yolo26n-pose.pt 的实测值；换 s/m/l/x 时 P3/P4/P5 通道数会变，
#: 那时 FEATURE_DIM 会不匹配并在 __init__ 里直接报错（宁可炸，不要静默变形）。
#: 层描述里的索引在 __init__ 里按实际网络重写（head.f），这里只是默认文案。
FEATURE_SOURCES = (
    ("P3", "layers[16] C3k2 (P3 neck)", (1, 64, 40, 40), 64, "GAP"),
    ("P4", "layers[19] C3k2 (P4 neck)", (1, 128, 20, 20), 128, "GAP"),
    ("P5", "layers[22] C3k2 (P5 neck)", (1, 256, 10, 10), 256, "GAP"),
    ("CLS", "layers[23] Pose26.one2one_cv3[1]", (1, 1, 20, 20), 100,
     "sigmoid + 2x2 avgpool"),
)

#: 拼接顺序（必须与 _fuse() 一致）
_SOURCE_ORDER = ("P3", "P4", "P5", "CLS")


def describe_sources():
    """返回 [(键, 层描述, 形状, 维度, 池化方式), ...]，供测试与日志打印。"""
    return list(FEATURE_SOURCES)


def resolve_model_path(model_path=MODEL_NAME):
    """定位权重文件：先按给定路径找，再找本目录 ``assets/``，最后交给 ultralytics 下载。

    ultralytics 把权重下到**当前工作目录**，从哪个目录起脚本就会在哪留一份 7.5MB
    的文件。本仓库预先放了一份在 ``assets/yolo26n-pose.pt``，这样从任何工作目录
    起都能命中，不会重复下载、也不会往别人的目录里丢文件。
    """
    p = str(model_path)
    if os.path.isabs(p) or os.path.exists(p):
        return p
    cand = os.path.join(_HERE, "assets", os.path.basename(p))
    return cand if os.path.exists(cand) else p


def letterbox_params(h, w, size=IMGSZ):
    """给出 (缩放比 r, 左 pad dw, 上 pad dh, 缩放后宽 nw, 高 nh)。

    ``letterbox_bgr`` 与实际几何计算共用这一处，避免两条路径算歪。
    """
    if h <= 0 or w <= 0:
        raise ValueError("空图：h=%r w=%r" % (h, w))
    r = min(size / float(h), size / float(w))
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    dw, dh = (size - nw) // 2, (size - nh) // 2
    return r, dw, dh, nw, nh


def letterbox_bgr(bgr, size=IMGSZ, pad_value=114):
    """把任意尺寸 BGR 图等比缩放并居中 pad 到 ``size×size``（uint8）。

    为什么必须自己 letterbox：ultralytics 的 predict 做的是**保持长宽比**的
    rect letterbox，4:3 的帧在 ``imgsz=320`` 下会变成 320×256（不是 320×320），
    特征图变成 32×40 / 16×20 / 8×10，CLS 热图池化后是 80 维而不是 100 维，
    FEATURE_DIM 会在两条路径之间飘（实测 528 vs 548）。契约要求 320×320，
    所以统一在这里做成正方形，再把预测结果用返回的 ``meta`` 映射回原图。
    """
    import cv2

    if bgr is None or getattr(bgr, "ndim", None) != 3 or bgr.shape[2] != 3:
        raise ValueError("letterbox_bgr 需要 HxWx3 uint8，收到 %r"
                         % (None if bgr is None else getattr(bgr, "shape", type(bgr))))
    h, w = bgr.shape[:2]
    r, dw, dh, nw, nh = letterbox_params(h, w, size)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas


class YoloFeatureExtractor:
    """YOLO26 特征抽取器（forward hook，单次前向同时出检测结果和内部特征）。

    用法一（pipeline 用，单次前向）：

        >>> ex = YoloFeatureExtractor("yolo26n-pose.pt", device="cuda:0")
        >>> res, feat_raw, sources, meta = ex.predict_and_features(bgr_frame, conf=0.25)
        >>> feat_raw.shape
        (548,)

    用法二（离线批量抽特征 / 测试用，自己 letterbox）：

        >>> ex.extract(bgr_frame).shape
        (548,)

    ``sources`` 是 ``[{"name","layer","shape","dim","op"}, ...]``，用来打印
    「层名 → 维度」；维度之和必然等于 ``FEATURE_DIM``（不符会抛 RuntimeError）。
    """

    def __init__(self, model_path=MODEL_NAME, device=None, verbose=True):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            print("[extract_features] CUDA 不可用，退回 CPU")
            device = "cpu"
        self.device = str(device)
        self.model_path = resolve_model_path(model_path)
        if self.device == "cpu":
            # 与参考实现一致：CPU 下固定线程数，耗时读数才有可比性
            torch.set_num_threads(4)

        self.model = YOLO(self.model_path)
        self.net = self.model.model.float().eval().to(self.device)
        self.layers = self.net.model

        head = self.layers[-1]
        f = getattr(head, "f", None)
        if f is None or len(list(f)) != 3:
            raise RuntimeError(
                "末层 %s 没有 3 个输入索引（f=%r），无法定位 P3/P4/P5；"
                "这个权重不是 YOLO26 结构？" % (type(head).__name__, f))
        self.idx_p3, self.idx_p4, self.idx_p5 = (int(v) for v in f)
        self.idx_head = len(self.layers) - 1
        self.head = head

        # YOLO26（end2end）推理走 one2one 头；predict 第一次调用会把 cv3 置 None。
        if getattr(head, "end2end", False) and hasattr(head, "one2one_cv3"):
            self.cls_branch = head.one2one_cv3[1]
            self.cls_branch_path = "%s.one2one_cv3[1]" % type(head).__name__
        else:
            self.cls_branch = head.cv3[1]
            self.cls_branch_path = "%s.cv3[1]" % type(head).__name__

        self._desc = {
            "P3": "layers[%d] %s (P3 neck)"
                  % (self.idx_p3, type(self.layers[self.idx_p3]).__name__),
            "P4": "layers[%d] %s (P4 neck)"
                  % (self.idx_p4, type(self.layers[self.idx_p4]).__name__),
            "P5": "layers[%d] %s (P5 neck)"
                  % (self.idx_p5, type(self.layers[self.idx_p5]).__name__),
            "CLS": "layers[%d] %s" % (self.idx_head, self.cls_branch_path),
        }

        self.imgsz = IMGSZ
        self._feats = {}
        self._handles = []
        self._register_hooks()
        self.last_sources = []
        if verbose:
            print("[extract_features] %s on %s, input %dx%d"
                  % (self.model_path, self.device, IMGSZ, IMGSZ))
            self.print_sources()

    # ------------------------- hooks -------------------------
    def _hook(self, key):
        def _cb(_module, _inp, out):
            self._feats[key] = out
        return _cb

    def _register_hooks(self):
        mods = {
            "P3": self.layers[self.idx_p3],
            "P4": self.layers[self.idx_p4],
            "P5": self.layers[self.idx_p5],
            "CLS": self.cls_branch,
        }
        for key, mod in mods.items():
            self._handles.append(mod.register_forward_hook(self._hook(key)))

    def close(self):
        """摘掉 hook（同一个 YOLO 对象还要拿去干别的时用）。"""
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001 - 重复 close 无害
                pass
        self._handles = []

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - 解释器退出阶段可能已拆模块
            pass

    # ------------------------- 特征拼接 -------------------------
    def _fuse(self):
        """把 hook 抓到的张量拼成 (FEATURE_DIM,) float32，并给出每个来源的维度。"""
        import torch
        import torch.nn.functional as F

        missing = [k for k in _SOURCE_ORDER if k not in self._feats]
        if missing:
            raise RuntimeError(
                "前向没有触发这些 hook：%r（模型被换掉或 hook 被 fuse 掉）" % (missing,))

        feats = self._feats
        cls_hm = torch.sigmoid(feats["CLS"].detach().float())
        hm = F.avg_pool2d(cls_hm, 2).flatten(1)          # (1, nc*10*10)
        p3 = feats["P3"].detach().float().mean(dim=(2, 3))   # (1, 64)
        p4 = feats["P4"].detach().float().mean(dim=(2, 3))   # (1, 128)
        p5 = feats["P5"].detach().float().mean(dim=(2, 3))   # (1, 256)
        vec = torch.cat([p3, p4, p5, hm], dim=1)[0]
        arr = vec.cpu().numpy().astype(np.float32, copy=False)

        shapes = {
            "P3": tuple(feats["P3"].shape), "P4": tuple(feats["P4"].shape),
            "P5": tuple(feats["P5"].shape), "CLS": tuple(feats["CLS"].shape),
        }
        dims = {"P3": p3.shape[1], "P4": p4.shape[1], "P5": p5.shape[1],
                "CLS": hm.shape[1]}
        self.last_sources = [
            {"name": name, "layer": self._desc[name], "shape": shapes[name],
             "dim": int(dims[name]), "op": op}
            for name, _layer, _shape, _dim, op in FEATURE_SOURCES
        ]
        total = int(arr.shape[0])
        if total != FEATURE_DIM:
            raise RuntimeError(
                "特征维度不符：实测 %d，常量 FEATURE_DIM=%d。"
                "来源维度 %r。换权重（n/s/m/l/x）后通道数会变，"
                "要么改回 yolo26n，要么同步更新 FEATURE_DIM 与 FeatureAdapter。"
                % (total, FEATURE_DIM, dims))
        return arr, self.last_sources

    def print_sources(self, sources=None):
        """打印「层名 → 维度」表（验收测试要的那一条）。"""
        rows = sources if sources is not None else self.last_sources
        if not rows:
            desc = getattr(self, "_desc", {})
            rows = [{"name": n, "layer": desc.get(n, lay), "shape": sh,
                     "dim": d, "op": op}
                    for n, lay, sh, d, op in FEATURE_SOURCES]
        print("  %-4s %-32s %-16s %-22s %s" % ("src", "layer", "shape", "op", "dim"))
        total = 0
        for r in rows:
            total += int(r["dim"])
            print("  %-4s %-32s %-16s %-22s %d"
                  % (r["name"], r["layer"], str(r["shape"]), r["op"], r["dim"]))
        print("  %-4s %-32s %-16s %-22s %d  (FEATURE_DIM=%s)"
              % ("SUM", "", "", "", total, "match" if total == FEATURE_DIM
                 else "MISMATCH %d" % FEATURE_DIM))

    # ------------------------- 两条前向路径 -------------------------
    def predict_and_features(self, bgr_frame, conf=0.25, **predict_kw):
        """**单次前向**：同时返回 ultralytics 检测结果和内部特征。

        hook 挂在同一个 nn.Module 上，所以 ``YOLO.predict`` 内部那次前向就把
        特征一起抓到了——不需要为特征再跑一遍网络（省一半时间和显存带宽）。

        ⚠️ **Boxes / keypoints 是 320×320 letterbox 画布坐标系**，调用方必须用
        返回的 ``meta`` 映射回原图（``run_perception`` 负责这件事）。

        Args:
            bgr_frame: HxWx3 uint8 BGR，任意尺寸。
            conf: 置信度阈值，透传给 predict。
            **predict_kw: 其他 predict 参数（verbose 默认 False）。

        Returns:
            (Results, feat_raw np.ndarray(FEATURE_DIM,), sources list, meta dict)
            meta = {"r","dw","dh","nw","nh","h","w","size"}
        """
        predict_kw.setdefault("verbose", False)
        canvas = letterbox_bgr(bgr_frame, IMGSZ)
        h, w = bgr_frame.shape[:2]
        r, dw, dh, nw, nh = letterbox_params(h, w, IMGSZ)
        self._feats.clear()
        res = self.model.predict(
            source=canvas, imgsz=IMGSZ, conf=float(conf),
            device=self.device, **predict_kw)[0]
        feat_raw, sources = self._fuse()
        meta = {"r": r, "dw": dw, "dh": dh, "nw": nw, "nh": nh,
                "h": h, "w": w, "size": IMGSZ}
        return res, feat_raw, sources, meta

    def extract(self, bgr_frame):
        """独立抽取路径：自己 letterbox 到 320×320，直接前向网络。

        不跑 NMS / 后处理，离线抽特征缓存时更省。返回 (FEATURE_DIM,) float32。
        与 ``predict_and_features`` 对同一帧给出**同维度**的特征（都是 548）。
        """
        import torch

        img = letterbox_bgr(bgr_frame, IMGSZ)[:, :, ::-1]   # BGR -> RGB
        x = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        self._feats.clear()
        with torch.inference_mode():
            self.net(x)
        feat_raw, _ = self._fuse()
        return feat_raw

    def extract(self, bgr_frame):
        """独立抽取路径：自己 letterbox 到 320×320，直接前向网络。

        不跑 NMS / 后处理，离线抽特征缓存时更省。返回 (FEATURE_DIM,) float32。
        """
        import torch

        img = letterbox_bgr(bgr_frame, IMGSZ)
        img = img[:, :, ::-1]                      # BGR -> RGB
        x = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        self._feats.clear()
        with torch.inference_mode():
            self.net(x)
        feat_raw, _ = self._fuse()
        return feat_raw

    def extract_many(self, frames):
        """逐帧抽特征（离线缓存用；不做批处理，先求对不求快）。"""
        return np.stack([self.extract(f) for f in frames]).astype(np.float32)
