# -*- coding: utf-8 -*-
"""synth.py — 从**真实**骨架合成动态手势序列

训练/标定都需要「有时间结构」的样本，但 HaGRID 是静态图集（每张图一个姿势）。
这里的做法是把真实图片里 YOLO 估出来的姿势当作**基帧**，只对**运动**做合成：

* 静态类（idle / open_palm / thumbs_up / pointing / fist）：把同一个人同一类的
  若干真实基帧按帧序拼起来 + 加抖动。**姿势来自真人**，只有帧间抖动是合成的。
* 挥手：手已经举起（基帧是 palm/stop），让手腕沿横向正弦轨迹摆动，肘关节用
  **2 连杆 IK** 解出来（保留真实的大臂/小臂长度），再叠真实感抖动。
  频率 0.5~2.5 Hz、幅度 0.35~0.80 肩宽，来自人手挥动的常见范围。

**必须说清楚的局限**：挥手类的训练/测试样本全是这套合成运动。所以「wave F1」
证明的是「模型能从这个运动分布里认出挥手」，**不等于**真人挥手已经验证过——
真机验证要靠 TECH-PLAN §9.1 场景 7 的实录（README §局限 写了这条）。

抖动幅度按 240×240 机器人视野里的实测关键点噪声量级取（σ≈0.012 肩宽），
并带 8% 的关键点随机丢置信度，逼模型别依赖「永远干净的关键点」。
"""
from __future__ import annotations

import numpy as np

from gesture_common import (CLASS_TO_ID, COCO17, FEAT_DIM, OTHER_ID,
                            build_features_from_rel)

__all__ = ["PoseBank", "make_segment", "concat_blend", "SEGMENT_KINDS"]

L_SH, R_SH = COCO17["left_shoulder"], COCO17["right_shoulder"]
L_EL, R_EL = COCO17["left_elbow"], COCO17["right_elbow"]
L_WR, R_WR = COCO17["left_wrist"], COCO17["right_wrist"]

_ARM = {"L": (L_SH, L_EL, L_WR), "R": (R_SH, R_EL, R_WR)}

SEGMENT_KINDS = ("idle", "hold", "wave", "raise")

#: 关键点抖动的默认强度（肩宽为单位）
JITTER_SIGMA = 0.012


class PoseBank:
    """按 user 索引的真实基帧（归一化骨架 + 手部裁剪）。

    Args:
        kp_rel: (N,17,2) float32，肩宽归一化、原点在肩中点的坐标。
        conf: (N,17) float32 关键点置信度。
        crop: (N,32,32) uint8 手部裁剪。
        label: (N,) int，0..5。
        user: (N,) int，用户编号。
    """

    def __init__(self, kp_rel, conf, crop, label, user):
        self.kp = np.asarray(kp_rel, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.crop = np.asarray(crop, dtype=np.uint8)
        self.label = np.asarray(label, dtype=np.int64)
        self.user = np.asarray(user, dtype=np.int64)
        self._by = {}
        for c in np.unique(self.label):
            for u in np.unique(self.user[self.label == c]):
                self._by[(int(c), int(u))] = np.flatnonzero(
                    (self.label == c) & (self.user == u))

    @property
    def users(self):
        return np.unique(self.user)

    def idx(self, cls_id, user):
        """取「某个人的某一类」的基帧下标；该人没有这类则退回同类其他人。"""
        hit = self._by.get((int(cls_id), int(user)))
        if hit is not None and len(hit):
            return hit
        for (c, u), rows in self._by.items():
            if c == int(cls_id):
                return rows
        return np.zeros(0, dtype=np.int64)

    def subset(self, mask):
        """按行掩码切一份（用于按 user 划分 train/val/test）。"""
        return PoseBank(self.kp[mask], self.conf[mask], self.crop[mask],
                        self.label[mask], self.user[mask])


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def ik_elbow(shoulder, wrist, l1, l2, elbow_ref):
    """2 连杆 IK：给定肩、腕，解肘关节位置。

    两个解（肘向上/向下）都合法，取离**观测到的肘**更近的那支，保证不出现
    「肘反折」这种真人做不出来的姿势。
    """
    v = np.asarray(wrist, dtype=np.float32) - np.asarray(shoulder, np.float32)
    d = float(np.linalg.norm(v))
    d = min(max(d, abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    u = _unit(v)
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)
    a = np.arccos(float(np.clip(cos_a, -1.0, 1.0)))
    best, best_err = None, 1e9
    for sgn in (+1.0, -1.0):
        ca, sa = np.cos(sgn * a), np.sin(sgn * a)
        rot = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        e = np.asarray(shoulder, np.float32) + l1 * (rot @ u)
        err = float(np.linalg.norm(e - np.asarray(elbow_ref, np.float32)))
        if err < best_err:
            best, best_err = e, err
    return best.astype(np.float32)


def _active_hand(kp, conf):
    """举起的手（腕 y 更小）；两手都低时取置信度更高的。"""
    l_up = kp[L_WR, 1] < kp[L_SH, 1]
    r_up = kp[R_WR, 1] < kp[R_SH, 1]
    if l_up != r_up:
        return "L" if l_up else "R"
    if conf is not None and conf[L_WR] != conf[R_WR]:
        return "L" if conf[L_WR] > conf[R_WR] else "R"
    return "L" if kp[L_WR, 1] <= kp[R_WR, 1] else "R"


def _limb_lengths(kp, hand):
    sh, el, wr = _ARM[hand]
    l1 = float(np.linalg.norm(kp[el] - kp[sh]))
    l2 = float(np.linalg.norm(kp[wr] - kp[el]))
    # 真被遮挡时 IK 会退化，钳到合理的解剖范围（0.18~0.45 肩宽）
    l1 = float(np.clip(l1, 0.18, 0.48))
    l2 = float(np.clip(l2, 0.18, 0.45))
    return l1, l2


def _jitter(kp, conf, rng, sigma=JITTER_SIGMA, drop_p=0.08):
    """逐帧加关键点噪声 + 全身小幅度晃动 + 置信度扰动（模拟 240×240 的实测抖动）。"""
    n = kp.shape[0]
    out = kp + rng.normal(0.0, sigma, (n, 17, 2)).astype(np.float32)
    # 全身平移 + 尺度微变（摄像头抖动 / 人轻微移动）
    out += rng.normal(0.0, 0.010, (n, 1, 2)).astype(np.float32)
    s = 1.0 + rng.normal(0.0, 0.010, (n, 1, 1)).astype(np.float32)
    out = out * s
    c = conf * rng.uniform(0.85, 1.0, (n, 17)).astype(np.float32)
    drop = rng.random((n, 17)) < drop_p
    c = np.where(drop, c * 0.25, c)
    return out.astype(np.float32), np.clip(c, 0.0, 1.0).astype(np.float32)


def _crop_seq(bank, rows, n, rng):
    """按基帧循环取手部裁剪，加旋转/缩放/平移/亮度扰动。

    目的只有一个：**逼外观分支只看手形**，不要记住「这个人的手长什么样」。
    30k 采样集里只有 ~5 千张不同的人的裁剪，不加重扰动，外观分支一个 epoch
    就把人记熟了，换个真人立刻失效。
    """
    import cv2

    idx = rows[np.arange(n) % len(rows)]
    crops = bank.crop[idx].copy()
    px = crops.shape[-1]
    center = ((px - 1) / 2.0, (px - 1) / 2.0)
    for i in range(n):
        c = crops[i]
        ang = float(rng.uniform(-12.0, 12.0))
        sc = float(rng.uniform(0.85, 1.15))
        m = cv2.getRotationMatrix2D(center, ang, sc)
        m[0, 2] += float(rng.uniform(-2.0, 2.0))
        m[1, 2] += float(rng.uniform(-2.0, 2.0))
        c = cv2.warpAffine(c, m, (px, px), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        g = float(rng.uniform(0.80, 1.20))
        b = float(rng.uniform(-15.0, 15.0))
        crops[i] = np.clip(c.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    return crops


def make_segment(bank, rng, user, kind, n_frames, cls_name=None,
                 freq_hz=None, amp=None, fps=30.0, t_offset=0.0,
                 swap_hand_p=0.0, rows=None):
    """生成一段连续帧（一个动作阶段）。

    Args:
        bank: :class:`PoseBank`。
        rng: ``np.random.Generator``。
        user: 用户编号。
        kind: ``"idle"`` / ``"hold"`` / ``"wave"`` / ``"raise"``。
        n_frames: 帧数。
        cls_name: ``hold``/``raise`` 的目标类名（``open_palm`` 等）。
        freq_hz: 挥手频率（None 时在 0.5~2.5 Hz 里抽）。
        amp: 挥手幅度（肩宽单位；None 时在 0.35~0.80 里抽）。
        fps: 帧率，只用于把频率换算成相位。
        t_offset: 相位偏移（秒），把两段挥手接起来时保证相位连续。
        swap_hand_p: 以该概率镜像左右（左右撇子覆盖面）。
        rows: 直接指定基帧下标（给 ``other`` 未知手势用，跳过按类别查表）。

    Returns:
        ``(feat (n,FEAT_DIM) float32, crop (n,32,32) uint8, info dict)``
    """
    cls_name = cls_name or "open_palm"
    if cls_name == "wave":
        # 挥手的手形就是「张开手掌」，HaGRID 里没有 wave 类，用 palm 当基帧
        cls_name = "open_palm"
    if rows is None:
        if cls_name == "other":
            rows = bank.idx(OTHER_ID, user)
            cls_id = OTHER_ID
        else:
            cls_id = (CLASS_TO_ID["idle"] if kind == "idle"
                      else CLASS_TO_ID[cls_name])
            rows = bank.idx(cls_id, user)
    elif cls_name == "other":
        cls_id = OTHER_ID
    else:
        cls_id = -3                                  # other（未知手势）
    if len(rows) == 0:
        raise ValueError("empty bank for class %r" % cls_name)
    base_i = int(rows[rng.integers(len(rows))])
    kp0 = bank.kp[base_i].copy()
    conf0 = bank.conf[base_i].copy()

    if swap_hand_p > 0.0 and rng.random() < swap_hand_p:
        kp0 = _mirror(kp0)
        conf0 = _swap_lr_conf(conf0)

    hand = _active_hand(kp0, conf0)
    kp = np.repeat(kp0[None], n_frames, axis=0)
    conf = np.repeat(conf0[None], n_frames, axis=0)

    info = {"kind": kind, "cls_id": cls_id, "hand": hand}
    if kind == "wave":
        freq = float(freq_hz if freq_hz is not None
                     else rng.uniform(0.5, 2.5))
        amp_v = float(amp if amp is not None else rng.uniform(0.35, 0.80))
        sh, el, wr = _ARM[hand]
        l1, l2 = _limb_lengths(kp0, hand)
        sh0 = kp0[sh].copy()
        # 手腕的「中位」：手举着的位置；再以它为轴左右摆。
        # 基帧的手可能在腰际（HaGRID 里不少 palm 是低位拍的），那样合成出来的
        # 「挥手」其实是腰边摆手，模型判成 idle 是对的——所以这里把中位抬到
        # 肩线以上（真人挥手本来也在这个高度），横向位置沿用基帧。
        center = kp0[wr].copy()
        if center[1] > sh0[1] - 0.05:
            side = 1.0 if center[0] >= sh0[0] else -1.0
            center = np.array([sh0[0] + side * max(abs(center[0] - sh0[0]),
                                                   0.45),
                               sh0[1] - 0.40], dtype=np.float32)
        phase0 = float(rng.uniform(0.0, 2 * np.pi))
        t = (np.arange(n_frames, dtype=np.float32) / float(fps)) + t_offset
        ph = 2 * np.pi * freq * t + phase0
        # 基频 + 一点二次谐波（真人挥手不是纯正弦）
        osc = amp_v * (np.sin(ph) + 0.12 * np.sin(2 * ph + 0.7))
        bob = -0.18 * amp_v * np.abs(np.sin(ph))       # 手上下轻微起伏
        kp[:, wr, 0] = center[0] + osc
        kp[:, wr, 1] = center[1] + bob
        for f in range(n_frames):
            kp[f, el] = ik_elbow(sh0, kp[f, wr], l1, l2, kp0[el])
        info.update({"freq_hz": freq, "amp": amp_v})
    elif kind == "raise":
        # 手从体侧抬到目标姿势，再保持：0.35~0.55 s 抬起
        sh, el, wr = _ARM[hand]
        l1, l2 = _limb_lengths(kp0, hand)
        sh0 = kp0[sh].copy()
        target = kp0[wr].copy()
        hang = np.array([kp0[sh, 0] + 0.10, kp0[sh, 1] + 0.85],
                        dtype=np.float32)          # 手下垂的大致位置
        n_rise = max(3, int(round(0.45 * fps)))
        rise = np.clip(np.arange(n_frames, dtype=np.float32) / n_rise, 0, 1)
        ease = rise * rise * (3 - 2 * rise)         # smoothstep
        for f in range(n_frames):
            kp[f, wr] = hang + ease[f] * (target - hang)
            kp[f, el] = ik_elbow(sh0, kp[f, wr], l1, l2, kp0[el])
        info["rise_frames"] = n_rise
    elif kind not in ("idle", "hold"):
        raise ValueError("unknown kind %r" % kind)

    kp = kp.astype(np.float32)
    kp, conf = _jitter(kp, conf, rng)
    crops = _crop_seq(bank, rows, n_frames, rng)
    feat = build_features_from_rel(kp, conf, SW_NORM_SYNTH)
    info.update({"kp_rel": kp, "conf": conf})
    return feat, crops, info


#: 合成帧没有图像尺寸，第 51 维（肩宽/图宽）固定成一个真实分布内的值，
#: 避免这一维在训练里退化成「合成标记」。真实数据里它在 0.18~0.32。
SW_NORM_SYNTH = 0.25


def concat_blend(fa, ca, fb, cb, blend=3):
    """把 A、B 两段帧接起来，在边界用 ``blend`` 帧线性过渡。

    直接拼接会出现「上一帧手在体侧、下一帧手在头顶」的瞬移，这是合成数据才
    有的假特征；线性过渡让它看起来像人抬手/放手的真实过程。
    """
    if blend <= 0 or len(fb) == 0 or len(fa) == 0:
        return np.concatenate([fa, fb]), np.concatenate([ca, cb])
    blend = int(min(blend, len(fb)))
    w = np.linspace(0.0, 1.0, blend + 2)[1:-1].astype(np.float32)[:, None]
    f_mid = (1 - w) * fa[-1][None] + w * fb[:blend]
    w2 = w[:, :, None]                       # (blend,1,1) 适配 (N,32,32)
    c_mid = ((1 - w2) * ca[-1][None].astype(np.float32)
             + w2 * cb[:blend].astype(np.float32))
    return (np.concatenate([fa, f_mid.astype(np.float32), fb[blend:]]),
            np.concatenate([ca, c_mid.astype(np.uint8), cb[blend:]]))


def _mirror(kp):
    """左右镜像：交换 L/R 关键点索引并把 x 取反。"""
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16)]
    out = kp.copy()
    out[:, 0] = -out[:, 0]
    for a, b in pairs:
        out[a], out[b] = out[b].copy(), out[a].copy()
    return out


def _swap_lr_conf(conf):
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16)]
    out = conf.copy()
    for a, b in pairs:
        out[a], out[b] = out[b], out[a]
    return out
