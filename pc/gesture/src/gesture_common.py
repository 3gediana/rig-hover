# -*- coding: utf-8 -*-
"""gesture_common.py — 训练与部署共用的骨架特征定义（单一真源）

**为什么单独一个模块**：训练时算的特征和机器人上算的特征必须逐位一致，否则
「离线 F1 0.9」在机器上会掉到不可用。这里只放纯函数、常量，不放 I/O。

坐标约定与 ``pc/perception/gesture.py`` 一致（契约 §3.2）：

    sw   = |kp[left_shoulder].x - kp[right_shoulder].x|     # 像素
    mid  = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2
    rel  = (kp - mid) / sw                                  # 量纲消除

**为什么不用绝对像素**：机器人摄像头 240×240，人 0.5m 和 2m 时肩宽差 4 倍，
任何像素阈值都不成立。除以肩宽后同一姿势在任意距离给出同一个数。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "COCO17", "KEYPOINT_ORDER", "CLASSES", "CLASS_TO_ID", "ID_TO_CLASS",
    "MODEL_CLASSES", "OTHER_ID", "N_CLASSES", "N_MODEL_CLASSES",
    "HAGRID_TO_CLASS", "FEAT_DIM", "CROP_PX", "build_frame_features",
    "build_features_from_rel", "hand_crop", "skeleton_to_pixels", "REL_CLIP",
]

#: COCO-17 关键点索引（YOLO26n-pose 的输出顺序）
COCO17 = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

KEYPOINT_ORDER = tuple(COCO17)

#: 判定的手势类别。**顺序即 gesture_id**，不要动（契约里的 id 编号依赖它）。
CLASSES = ("idle", "wave", "open_palm", "thumbs_up", "pointing", "fist")
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

#: 分类器**内部**多一个 ``other`` 类：训练时把 HaGRID 里没映射的 12 类手势
#: （peace/rock/ok/…）当作「未知手势」学进去，输出时把它的概率并回 idle。
#: 对外接口仍然是 6 类，gesture_id 仍是 0..5。
#: 没有这一类的实测后果：未知手势 79% 会超过触发阈值 → 真机上到处误报。
MODEL_CLASSES = CLASSES + ("other",)
OTHER_ID = len(CLASSES)
N_CLASSES = len(CLASSES)
N_MODEL_CLASSES = len(MODEL_CLASSES)

#: HaGRID 类别 → 本项目的 6 类（None = 不进训练集，留作「未知手势」误报测试）。
#: 说明见 README §数据：``stop_inverted`` 是手背朝外的「停止」，语义与 open_palm
#: 相同；``like`` 才是拇指向上，``dislike``（拇指向下）不映射到 thumbs_up。
HAGRID_TO_CLASS = {
    "no_gesture": "idle",
    "palm": "open_palm",
    "stop": "open_palm",
    "stop_inverted": "open_palm",
    "like": "thumbs_up",
    "one": "pointing",
    "fist": "fist",
    # ---- 以下 12 类不参与训练（见 README「开放集风险」）----
    "call": None, "dislike": None, "four": None, "mute": None, "ok": None,
    "peace": None, "peace_inverted": None, "rock": None, "rock_inverted": None,
    "three": None, "three2": None, "two_up": None, "two_up_inverted": None,
}

#: 每帧特征维度：17 关键点 × (x_rel, y_rel) + 17 置信度 + 2 尺度量
FEAT_DIM = 17 * 2 + 17 + 2

#: 手部裁剪边长（灰度方形）。32×32 在 240×240 的机器人视野里手部约 30~60 px，
#: 缩到 32 后仍保留指缝/手掌轮廓的大致形状，足够区分「握拳/张开/食指」。
CROP_PX = 32

#: 相对坐标的裁剪范围（±3 肩宽）。超出的关键点基本是误检，钳住防止梯度爆炸。
_REL_CLIP = 3.0

_L_SH = COCO17["left_shoulder"]
_R_SH = COCO17["right_shoulder"]
_L_EL = COCO17["left_elbow"]
_R_EL = COCO17["right_elbow"]
_L_WR = COCO17["left_wrist"]
_R_WR = COCO17["right_wrist"]


def build_features_from_rel(kp_rel, conf=None, sw_norm=0.25):
    """归一化骨架（肩宽=1，原点=肩中点）→ (FEAT_DIM,) 特征向量。

    训练时的合成序列和部署时的真实帧最终都走这里，保证两者特征逐位一致。

    Args:
        kp_rel: (17,2) 或 (T,17,2)，肩宽归一化坐标。
        conf: 与首维同长的置信度，None 视为全 1。
        sw_norm: 肩宽 / 图宽，填进第 51 维（尺度线索）。

    Returns:
        ``(FEAT_DIM,)`` 或 ``(T,FEAT_DIM)`` float32。
    """
    rel = np.asarray(kp_rel, dtype=np.float32)
    single = rel.ndim == 2
    if single:
        rel = rel[None]
    n = rel.shape[0]
    c = (np.ones((n, 17), dtype=np.float32) if conf is None
         else np.asarray(conf, dtype=np.float32).reshape(n, 17))
    feat = np.zeros((n, FEAT_DIM), dtype=np.float32)
    feat[:, :34] = (np.clip(rel, -_REL_CLIP, _REL_CLIP) / _REL_CLIP).reshape(n, -1)
    feat[:, 34:51] = np.clip(np.nan_to_num(c), 0.0, 1.0)
    feat[:, 51] = float(np.clip(sw_norm, 0.0, 4.0))
    feat[:, 52] = 1.0
    return feat[0] if single else feat


def build_frame_features(kp_xy, kp_conf=None, shoulder_min_px=5.0,
                         img_wh=None):
    """单帧关键点（像素）→ (FEAT_DIM,) float32 特征向量。

    Args:
        kp_xy: (17,2) 像素坐标。
        kp_conf: (17,) 置信度；None 视为全 1。
        shoulder_min_px: 肩宽低于此值视为「算不出来」，返回全零 + valid=0。
        img_wh: (w,h) 原图尺寸；给出时最后两维填「肩宽/图宽」和 valid 标志位。

    Returns:
        (FEAT_DIM,) float32。最后一个特征是 valid 标志（0/1），倒数第二个是
        肩宽归一化（肩宽 / 图宽，无图像尺寸时用 120px 当参考宽度）。
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    kp = np.asarray(kp_xy, dtype=np.float32).reshape(-1, 2)
    if kp.shape[0] != 17:
        return feat
    conf = (np.ones(17, dtype=np.float32) if kp_conf is None
            else np.asarray(kp_conf, dtype=np.float32).reshape(-1))
    if conf.shape[0] != 17:
        return feat

    sw = float(abs(kp[_L_SH, 0] - kp[_R_SH, 0]))
    if not np.isfinite(sw) or sw < shoulder_min_px:
        return feat

    mid = 0.5 * (kp[_L_SH] + kp[_R_SH])
    w = float(img_wh[0]) if img_wh is not None else 120.0
    return build_features_from_rel((kp - mid) / sw, conf,
                                   sw / max(w, 1e-6))


def hand_crop(frame_bgr, kp_xy, kp_conf=None, crop_px=CROP_PX,
              shoulder_min_px=5.0, conf_min=0.25):
    """按「举得更高的那只手腕」裁一块手部灰度图。

    **为什么用「更高的手」而不是标注里的 leading_hand**：机器人上拿不到标注，
    训练和部署必须用同一个可计算的规则。举起的手腕 y 更小（图像 y 向下）。

    Args:
        frame_bgr: HxWx3 uint8。
        kp_xy: (17,2) 像素坐标（原图坐标系）。
        kp_conf: (17,) 置信度；None 视为全 1。
        crop_px: 输出边长。
        shoulder_min_px: 肩宽下限。
        conf_min: 手腕置信度低于此值返回全零裁剪。

    Returns:
        (crop_px, crop_px) uint8 灰度图；无人/不可信时返回全零。
    """
    import cv2

    out = np.zeros((crop_px, crop_px), dtype=np.uint8)
    kp = np.asarray(kp_xy, dtype=np.float32).reshape(-1, 2)
    if kp.shape[0] != 17 or frame_bgr is None:
        return out
    conf = (np.ones(17, dtype=np.float32) if kp_conf is None
            else np.asarray(kp_conf, dtype=np.float32).reshape(-1))
    sw = float(abs(kp[_L_SH, 0] - kp[_R_SH, 0]))
    if not np.isfinite(sw) or sw < shoulder_min_px:
        return out

    # 选手腕：置信度够的前提下取 y 更小的（举得更高）
    cands = []
    for wr, el in ((_L_WR, _L_EL), (_R_WR, _R_EL)):
        if conf[wr] >= conf_min and np.all(np.isfinite(kp[wr])):
            cands.append((float(kp[wr, 1]), wr, el))
    if not cands:
        return out
    cands.sort()
    _, wr, el = cands[0]

    # 裁剪中心：从手腕再往「手的方向」偏移一点（手在腕关节之外，由肘→腕方向
    # 外推 0.22 肩宽）；边长 0.40 肩宽 × 2 让**手本身**占满裁剪框。
    # 0.55 肩宽的老取值会让手只占框内 ~1/3，32×32 下手指几乎消失，
    # pointing 和 open_palm 就分不开（实测 pointing recall 掉到 0.19）。
    d = kp[wr] - kp[el]
    n = float(np.linalg.norm(d))
    if n > 1e-3:
        center = kp[wr] + (d / n) * 0.22 * sw
    else:
        center = kp[wr]
    half = 0.40 * sw
    h, w = frame_bgr.shape[:2]
    x0 = int(round(center[0] - half))
    y0 = int(round(center[1] - half))
    x1 = int(round(center[0] + half))
    y1 = int(round(center[1] + half))
    # 越界用边缘复制（cv2.copyMakeBorder），避免把黑边当成手的一部分
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    patch = frame_bgr[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if patch.size == 0:
        return out
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    out = cv2.resize(gray, (crop_px, crop_px), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out, dtype=np.uint8)


def skeleton_to_pixels(kp_rel, shoulder_px=60.0, mid_xy=(120.0, 120.0)):
    """归一化骨架（肩宽=1，原点=肩中点）→ 像素坐标。合成序列喂给老算法时用。"""
    kp = np.asarray(kp_rel, dtype=np.float32).reshape(17, 2)
    return kp * float(shoulder_px) + np.asarray(mid_xy, dtype=np.float32)
