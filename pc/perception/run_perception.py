# -*- coding: utf-8 -*-
"""run_perception.py — 工作流 A 主入口：一帧 BGR → PerceptionOut（契约 §3.1）

    >>> from run_perception import PerceptionPipeline
    >>> pipe = PerceptionPipeline(device="cuda:0")
    >>> out = pipe(bgr_frame)                 # HxWx3 uint8，任意尺寸
    >>> out.feat_raw.shape, out.feat_64.shape
    ((548,), (64,))
    >>> out.theta_img, out.seen
    (3.7, 1)

一帧只跑**一次**网络：``YOLO.predict`` 的前向里用 forward hook 同时把 P3/P4/P5
和分类热图抓出来（见 ``extract_features.py``），检测结果和 548 维特征来自同一次
计算，不存在"检测用一次前向、特征再跑一次"的双倍开销。

坐标约定（容易搞错，写在这里）：

* 网络输入固定 320×320（自己做的正方形 letterbox），所以 ``res`` 里的框和关键点
  是 **320×320 画布坐标**；
* ``PerceptionOut`` 里的 ``person_*`` 是**原图归一化量**（÷ 原图宽高），
  ``theta_img`` 由归一化 cx 得到；
* 喂给手势模块的关键点是**原图像素坐标**（除以缩放比、减去 pad）。

``theta_img = (person_cx - 0.5) * FOV_H``（契约 §3.2），``FOV_H`` 是构造参数。

手势：分类器 + 事件机（本轮接入，替代 ``wave_score > 4`` 阈值累积）
------------------------------------------------------------------

``GestureEstimator``（``gesture.py``，已标 deprecated）只保留 ``hand_up_L/R``
和 3 个 ``wave_*`` 兼容字段；手势本身改由 ``pc/gesture`` 的两级流水线判定：

    YOLO 关键点 + 原图 ─▶ GestureClassifier（12 帧因果滑窗 → 6 类概率）
                           └▶ GestureEventMachine（确定性状态机）
                                └▶ gesture_id / onset / phase / conf / gap_s

``gesture_onset`` 是**边沿**：一段连续动作只在进入 GESTURE 那一帧输出 1。
教师控制器用它替代 ``wave_score > 4`` 的阈值累积，因此**不需要冷却期**——
「同一段动作只产生一个 onset」是事件机的结构保证，不是靠时间窗口凑出来的。

事件机参数外置在 ``pc/gesture/configs/gesture_events.yaml``（标定后写回的
enter/exit/pending_hold/release/min_gap）。语义与槽位对照见
``GESTURE_INTEGRATION.md``。
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from extract_features import (FEATURE_DIM, FEATURE_SOURCES, MODEL_NAME,
                              YoloFeatureExtractor)
from feature_adapter import ADAPTER_OUT, FeatureAdapter
from gesture import GestureEstimator
import scene_signals as scene_sig

__all__ = ["PerceptionOut", "PerceptionPipeline", "FOV_H_DEFAULT", "image_angle",
           "GESTURE_DIR_DEFAULT"]

#: 水平视场角默认值（度）。待标定，TECH-PLAN §15-V2。
FOV_H_DEFAULT = 60.0

#: ``pc/gesture`` 的位置（本文件在 ``pc/perception/`` 下）。可用环境变量
#: ``RIG_GESTURE_DIR`` 覆盖，或在构造器里显式传 ``gesture_dir``。
GESTURE_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "gesture"


def _import_gesture_runtime(gesture_dir=None):
    """按需导入 ``pc/gesture`` 里的分类器与事件机。

    ``pc/gesture`` 不是包（没有 ``__init__.py``），只能把目录塞进 ``sys.path``。
    用 append 而不是 insert(0)：``pc/gesture/src`` 里也有 ``model.py`` 这类
    通用名字，抢到系统模块前面容易和别的工具撞名。

    Args:
        gesture_dir: 手势模块目录；``None`` 时取 ``RIG_GESTURE_DIR`` 或默认路径。

    Returns:
        ``(gesture_classifier, gesture_events)`` 两个模块。

    Raises:
        RuntimeError: 目录不存在（交给调用方决定是关掉手势还是修环境）。
    """
    d = Path(gesture_dir or os.environ.get("RIG_GESTURE_DIR") or GESTURE_DIR_DEFAULT)
    if not d.is_dir():
        raise RuntimeError(
            "找不到手势模块目录 %s；用 gesture_dir= 或环境变量 RIG_GESTURE_DIR 指定，"
            "或传 use_gesture=False 关掉手势。" % d)
    s = str(d)
    if s not in sys.path:
        sys.path.append(s)
    import gesture_classifier                      # noqa: PLC0415 - 延迟导入
    import gesture_events                          # noqa: PLC0415 - 延迟导入
    return gesture_classifier, gesture_events


def image_angle(person_cx, fov_h=FOV_H_DEFAULT):
    """归一化水平位置 → 相对视线中心的水平角（度，正 = 右）。"""
    return (float(person_cx) - 0.5) * float(fov_h)


@dataclass
class PerceptionOut:
    """单帧感知输出（字段与契约 §3.1 **逐字对齐**，不要改名字/顺序）。

    Attributes:
        feat_raw: (FEATURE_DIM,) float32，YOLO 内部特征拼接（未压缩）。
        feat_64: (64,) float32，FeatureAdapter 压缩后，tanh 后 ∈[-1,1]。
        theta_img: 人相对图像中心的水平角（度，正 = 右）。
        person_w: bbox 宽 / 图宽 ∈[0,1]。
        person_h: bbox 高 / 图高 ∈[0,1]。
        person_cx: bbox 中心 x / 图宽 ∈[0,1]。
        person_conf: 检测置信度。
        seen: 0/1，本帧是否检测到人。
        hand_up_L: 0/1，左手（COCO 语义，人自己的左手）是否举起。
        hand_up_R: 0/1。
        wave_sin: ∈[-1,1]。**已废弃**（旧相位，事件化后不用）。
        wave_cos: ∈[-1,1]。**已废弃**。
        wave_amp: 相对肩宽的摆幅，≥0。**已废弃**。
        gesture_conf: ∈[0,1]，**新**——事件机给出的当前手势置信度
            （GESTURE 相位是已确认类的平滑概率，其它相位 0）。
        gesture_id: **新**，0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist。
        gesture_onset: **新**，0/1 边沿脉冲：每次动作只有进入 GESTURE 那一帧是 1。
        gesture_phase: **新**，0=IDLE 1=PENDING 2=GESTURE 3=RETURN。
        gesture_gap_s: **新**，距上次 GESTURE 结束的秒数（连续量，上限 gap_cap_s）。
        wave_gesture_conf: **已废弃**——旧 ``gesture_conf`` 的定义
            （wave_score 饱和 × 手腕关键点置信度）。事件化后判定不读它，
            只为旧字段全保留而留着，下一轮可以整段删。
        theta_face: **新**，脸相对视线中心的水平角（度，正 = 右）。用鼻+双眼的
            置信度加权中心，不是检测框中心——框中心会随姿态漂移。
        face_scale: **新**，瞳距 / 图高 ∈[0,1]。人脸大小的线性代理量，也就
            直接是距离的代理量，用于"人脸在一定大小内才盯"的门限。
        face_valid: **新**，0/1。
        theta_fist: **新**，当前主导手（拳头）相对视线中心的水平角（度）。
        wrist_y_norm: **新**，(腕 y − 肩中点 y) / 肩髋距。**与距离无关**，
            所以同一个敲击动作在 1 m 和 3 m 处得到同一个数值。
        hand_side: **新**，主导手 "L"/"R"/""。
        knock_fired: **新**，0/1，本帧触发"敲两下"。
        knock_x / knock_y: **新**，触发时敲击点的图像像素坐标，供目标锁定。
        knock_count: **新**，触发时窗口内累计的敲击次数。
        kp_xy: (17,2) float32，所选那个人的 COCO-17 关键点，**原图像素坐标**。
            录制时必须带上它：只有腕部坐标能裁出 64px 手图，事后才能重训手势
            分类器；只存 ``theta_fist`` 这类派生标量的话，手图就永远拿不回来了。
        kp_conf: (17,) float32，逐点置信度。
        debug: 任意调试信息（含 ``wave_score`` 与事件机内部量）。
    """

    feat_raw: np.ndarray
    feat_64: np.ndarray
    theta_img: float
    person_w: float
    person_h: float
    person_cx: float
    person_conf: float
    seen: int
    hand_up_L: int
    hand_up_R: int
    wave_sin: float
    wave_cos: float
    wave_amp: float
    gesture_conf: float
    gesture_id: int = 0
    gesture_onset: int = 0
    gesture_phase: int = 0
    gesture_gap_s: float = 0.0
    wave_gesture_conf: float = 0.0
    #: 分类器**逐帧**的 argmax 类别与其概率。与 ``gesture_id`` 的区别是：
    #: 后者是事件机**确认后**的类别，需要连续多帧一致才会给出。把这两个分开
    #: 暴露，才能把端到端误差拆成"分类器错"和"事件机漏确认"两部分。
    gesture_raw_id: int = 0
    gesture_raw_conf: float = 0.0
    # ---- 场景 v2 新增（SCENARIO-v2.md 阶段 1）----
    theta_face: float = 0.0
    face_scale: float = 0.0
    face_valid: int = 0
    theta_fist: float = 0.0
    wrist_y_norm: float = 0.0
    hand_side: str = ""
    knock_fired: int = 0
    knock_x: float = 0.0
    knock_y: float = 0.0
    knock_count: int = 0
    # ---- 录制用原始量（不参与观测组装，只落盘）----
    kp_xy: np.ndarray = field(
        default_factory=lambda: np.zeros((17, 2), np.float32))
    kp_conf: np.ndarray = field(
        default_factory=lambda: np.zeros(17, np.float32))
    bbox_px: tuple = (0.0, 0.0, 0.0, 0.0)
    debug: dict = field(default_factory=dict)


class PerceptionPipeline:
    """YOLO26-pose + 特征钩子 + FeatureAdapter + 手势基元，一帧一次调用。

    Args:
        model_path: 权重路径/名字，默认 ``yolo26n-pose.pt``（联网自动下载）。
        device: ``"cuda:0"`` 或 ``"cpu"``；CUDA 不可用时自动退回 CPU。
        fov_h: 水平视场角（度），``theta_img`` 用。
        conf: 人体检测置信度阈值。
        adapter: 传入现成的 ``FeatureAdapter`` 就用它（共享训练时这么用）。
        adapter_state: 权重路径或 ``state_dict``，加载进 Adapter。
        hand_up_conf: 手势模块的关键点置信度阈值。
        use_gesture: 是否加载手势分类器 + 事件机。关掉后 5 个新字段恒为 0
            （只为跑纯视觉基准时省掉一次前向）。
        gesture_dir: ``pc/gesture`` 目录，默认按本文件位置推出。
        gesture_model: 分类器权重，默认 ``<gesture_dir>/models/gesture_classifier.pt``。
        gesture_device: 分类器设备，默认跟随 ``device``。
        gesture_config: 事件机参数（dict 或 YAML 路径），
            默认 ``<gesture_dir>/configs/gesture_events.yaml``。
        verbose: 打印模型信息与特征来源表。
    """

    def __init__(self, model_path=MODEL_NAME, device="cuda:0",
                 fov_h=FOV_H_DEFAULT, conf=0.25, adapter=None,
                 adapter_state=None, hand_up_conf=0.3, verbose=True,
                 use_gesture=True, gesture_dir=None, gesture_model=None,
                 gesture_device=None, gesture_config=None):
        self.fov_h = float(fov_h)
        self.conf = float(conf)
        self.extractor = YoloFeatureExtractor(model_path=model_path,
                                              device=device, verbose=verbose)
        self.device = self.extractor.device
        self.feature_dim = FEATURE_DIM
        self.adapter_out = ADAPTER_OUT

        if adapter is None:
            adapter = FeatureAdapter(FEATURE_DIM, ADAPTER_OUT)
        if adapter.feat_dim != FEATURE_DIM:
            raise ValueError("Adapter 输入维度 %d != FEATURE_DIM %d"
                             % (adapter.feat_dim, FEATURE_DIM))
        if adapter.out_dim != ADAPTER_OUT:
            raise ValueError("Adapter 输出维度 %d != %d"
                             % (adapter.out_dim, ADAPTER_OUT))
        if adapter_state is not None:
            if isinstance(adapter_state, str):
                state = torch.load(adapter_state, map_location="cpu")
            else:
                state = adapter_state
            adapter.load_state_dict(state)
        self.adapter = adapter.to(self.device).eval()
        for p in self.adapter.parameters():
            p.requires_grad_(False)      # A 阶段只推理；训练时由训练脚本放开

        self.gestures = GestureEstimator(hand_up_conf=hand_up_conf)
        # 场景 v2：敲击检测器。它只吃几何量（腕部归一化高度），
        # "是不是拳头"由手势分类器给，两者在下游的决策层组合，
        # 这样检测器本身可以离线单测。
        self.knock = scene_sig.KnockDetector()
        self._t_scene = 0.0
        self.gesture_clf = None
        self.event_machine = None
        if use_gesture:
            self._setup_gesture(gesture_dir, gesture_model, gesture_device,
                                gesture_config)
        self._t_prev = None
        self.n_frames = 0
        if verbose:
            print("[perception] FEATURE_DIM=%d -> feat_64=%d, FOV_H=%.1f deg, "
                  "adapter stats frozen=%s, gesture=%s"
                  % (FEATURE_DIM, ADAPTER_OUT, self.fov_h,
                     self.adapter.stats_frozen,
                     "off" if self.gesture_clf is None
                     else "classifier+events(%s)" % self.gesture_clf.device))

    # ------------------------------------------------------------------
    def _setup_gesture(self, gesture_dir, model_path, device, config):
        """加载 ``pc/gesture`` 的分类器与事件机（构造期一次）。

        Args:
            gesture_dir: 手势模块目录，None 走默认/env。
            model_path: 分类器权重路径，None 走 ``<dir>/models``。
            device: 分类器设备，None 跟随 pipeline 设备。
            config: 事件机参数（dict / YAML 路径 / None）。
        """
        gc, ge = _import_gesture_runtime(gesture_dir)
        base = Path(gesture_dir or os.environ.get("RIG_GESTURE_DIR")
                    or GESTURE_DIR_DEFAULT)
        mp = Path(model_path) if model_path else base / "models" / "gesture_classifier.pt"
        if not mp.is_file():
            raise FileNotFoundError("手势分类器权重不存在：%s" % mp)
        self.gesture_clf = gc.GestureClassifier.load(model_path=str(mp),
                                                     device=str(device or self.device))
        if isinstance(config, dict):
            self.event_machine = ge.GestureEventMachine(config)
        else:
            cp = Path(config) if config else base / "configs" / "gesture_events.yaml"
            if not cp.is_file():
                # 参数是标定写回的，不能悄悄用代码里的默认值顶替
                raise FileNotFoundError("事件机参数不存在：%s" % cp)
            self.event_machine = ge.GestureEventMachine(str(cp))

    # ------------------------------------------------------------------
    def reset(self):
        """清空时序状态（换场景 / 断流重连后调用）。"""
        self.gestures.reset()
        if self.gesture_clf is not None:
            self.gesture_clf.reset()
        if self.event_machine is not None:
            self.event_machine.reset()
        self.knock.reset()
        self._t_scene = 0.0
        self._t_prev = None
        self.n_frames = 0

    def describe_features(self):
        """返回 ``[{"name","layer","shape","dim","op"}, ...]``（层名 → 维度）。"""
        return self.extractor.last_sources or [
            {"name": n, "layer": lay, "shape": sh, "dim": d, "op": op}
            for n, lay, sh, d, op in FEATURE_SOURCES
        ]

    # ------------------------------------------------------------------
    def __call__(self, bgr_frame):
        """处理一帧。

        Args:
            bgr_frame: HxWx3 uint8 BGR 图（cv2 的默认格式），任意尺寸。

        Returns:
            PerceptionOut
        """
        t0 = time.perf_counter()
        frame = self._check_frame(bgr_frame)
        h, w = frame.shape[:2]

        res, feat_raw, sources, meta = self.extractor.predict_and_features(
            frame, conf=self.conf)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        person = self._pick_person(res, meta, w, h)

        feat_64 = self._adapt(feat_raw)
        t2 = time.perf_counter()

        dt = 0.0 if self._t_prev is None else max(0.0, t0 - self._t_prev)
        self._t_prev = t0
        if person is None:
            g = self.gestures.update(None, None, dt)
        else:
            g = self.gestures.update(person["kxy"], person["kconf"], dt)

        # 手势事件：分类器（逐帧概率）→ 事件机（边沿）
        gprobs = None
        ev = None
        if self.gesture_clf is not None:
            if person is None:
                gprobs = self.gesture_clf.push_invalid()
            else:
                gprobs = self.gesture_clf.push_kp(person["kxy"], person["kconf"],
                                                  frame)
            ev = self.event_machine.update(gprobs, dt)
        t3 = time.perf_counter()

        if person is None:
            theta = 0.0
            person_w = person_h = person_cx = person_conf = 0.0
            seen = 0
        else:
            theta = image_angle(person["cx_norm"], self.fov_h)
            person_w = person["w_norm"]
            person_h = person["h_norm"]
            person_cx = person["cx_norm"]
            person_conf = person["conf"]
            seen = 1

        # ---- 场景 v2 信号（全部来自 COCO-17 关键点，不新增模型）----
        self._t_scene += dt
        if person is None:
            face = scene_sig.face_signals(None, None, w, h, self.fov_h)
            hands = scene_sig.hand_signals(None, None, w, h, self.fov_h)
        else:
            face = scene_sig.face_signals(person["kxy"], person["kconf"],
                                          w, h, self.fov_h)
            hands = scene_sig.hand_signals(person["kxy"], person["kconf"],
                                           w, h, self.fov_h)
        # 类别 5 = fist（见 gesture 契约的类别顺序）；分类器给不出就按 0 处理，
        # 检测器本身不依赖它——"是不是拳头"在下游决策层与 knock_fired 相与。
        p_fist = 0.0
        if gprobs is not None:
            _gp = np.asarray(gprobs).reshape(-1)
            if _gp.size > 5:
                p_fist = float(_gp[5])
        side, hand = scene_sig.dominant_hand(
            hands["hands"], {"L": p_fist, "R": p_fist})
        knock = self.knock.update(self._t_scene, hand["wrist_y_norm"],
                                  hand["wrist_x"], hand["wrist_y"],
                                  valid=int(hand["valid"]))

        self.n_frames += 1
        debug = {
            "device": self.device,
            "fov_h": self.fov_h,
            "img_hw": (int(h), int(w)),
            "n_frames": self.n_frames,
            "n_det": 0 if person is None else person["n_det"],
            "feat_dim": int(feat_raw.shape[0]),
            "feat_sources": sources,
            "letterbox": {k: meta[k] for k in ("r", "dw", "dh", "nw", "nh")},
            "bbox_px": None if person is None else person["bbox_px"],
            "kp_conf_min": None if person is None else person["kp_conf_min"],
            "wave_score": float(g.wave_score),
            "waving": bool(g.waving),
            "hand_up_any": int(g.hand_up_any),
            "gesture": g.debug,
            "gesture_probs": None if gprobs is None else [float(v) for v in gprobs],
            "gesture_event": None if ev is None else ev.debug,
            "adapter_stats_frozen": bool(self.adapter.stats_frozen),
            "t_forward_ms": (t1 - t0) * 1e3,
            "t_adapter_ms": (t2 - t1) * 1e3,
            "t_gesture_ms": (t3 - t2) * 1e3,
            "t_total_ms": (t3 - t0) * 1e3,
            "dt_s": dt,
            "scene": {
                "face": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in face.items()},
                "hand_side": side,
                "theta_fist": round(float(hand["theta_fist"]), 4),
                "wrist_y_norm": round(float(hand["wrist_y_norm"]), 4),
                "body_scale_px": round(float(hands["body_scale_px"]), 2),
                "body_scale_src": hands["body_scale_src"],
                "fist_prob": round(float(p_fist), 4),
                "knock": knock.as_dict(),
            },
        }

        if gprobs is None:
            raw_id, raw_conf = 0, 0.0
        else:
            _gp = np.asarray(gprobs).reshape(-1)
            raw_id = int(_gp.argmax())
            raw_conf = float(_gp.max())

        return PerceptionOut(
            feat_raw=np.ascontiguousarray(feat_raw, dtype=np.float32),
            feat_64=np.ascontiguousarray(feat_64, dtype=np.float32),
            theta_img=float(theta),
            person_w=float(person_w),
            person_h=float(person_h),
            person_cx=float(person_cx),
            person_conf=float(person_conf),
            seen=int(seen),
            hand_up_L=int(g.hand_up_L),
            hand_up_R=int(g.hand_up_R),
            wave_sin=float(g.wave_sin),
            wave_cos=float(g.wave_cos),
            wave_amp=float(g.wave_amp),
            gesture_conf=float(ev.gesture_conf) if ev is not None else 0.0,
            gesture_id=int(ev.gesture_id) if ev is not None else 0,
            gesture_onset=int(ev.gesture_onset) if ev is not None else 0,
            gesture_phase=int(ev.gesture_phase) if ev is not None else 0,
            gesture_gap_s=float(ev.gesture_gap_s) if ev is not None else 0.0,
            wave_gesture_conf=float(g.gesture_conf),
            gesture_raw_id=raw_id,
            gesture_raw_conf=raw_conf,
            theta_face=float(face["theta_face"]),
            face_scale=float(face["face_scale"]),
            face_valid=int(face["face_valid"]),
            theta_fist=float(hand["theta_fist"]),
            wrist_y_norm=float(hand["wrist_y_norm"]),
            hand_side=str(side) if hand["valid"] else "",
            knock_fired=int(knock.fired),
            knock_x=float(knock.x_px),
            knock_y=float(knock.y_px),
            knock_count=int(knock.count),
            kp_xy=(np.zeros((17, 2), np.float32) if person is None else
                   np.asarray(person["kxy"], np.float32).reshape(17, 2).copy()),
            kp_conf=(np.zeros(17, np.float32) if person is None else
                     np.asarray(person["kconf"], np.float32).reshape(17).copy()),
            bbox_px=(tuple(float(v) for v in person["bbox_px"])
                     if person is not None else (0.0, 0.0, 0.0, 0.0)),
            debug=debug,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_frame(frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("bgr_frame 必须是 np.ndarray，收到 %r" % type(frame))
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("bgr_frame 必须是 HxWx3，收到 %r" % (frame.shape,))
        if frame.shape[0] < 8 or frame.shape[1] < 8:
            raise ValueError("图太小：%r" % (frame.shape,))
        if frame.dtype != np.uint8:
            raise ValueError("bgr_frame 必须是 uint8，收到 %r" % frame.dtype)
        return frame

    def _pick_person(self, res, meta, w, h):
        """选目标：**离得最近的那个人**（用户 2026-11-03 明确："谁靠着近就看谁"）。

        排名主键是**检测框高度**，不是面积。面积会对"被画面边缘裁掉一部分"的人
        低估——最近的人往往正好贴着画面边走，面积偏小，会被错判成更远的人。
        框高受水平裁剪影响小得多，而且它本身就是本模块一直在用的距离代理量
        （``person_h``）。面积和置信度只作平手时的次序。

        返回 None 表示本帧没人。
        """
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.detach().float().cpu().numpy()
        conf = boxes.conf.detach().float().cpu().numpy().reshape(-1)
        heights = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
        widths = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
        areas = widths * heights
        # np.lexsort 以**最后一个键为主键**：先按框高降序，再面积降序，再置信度降序
        order = np.lexsort((-conf, -areas, -heights))
        i = int(order[0])

        r, dw, dh = float(meta["r"]), float(meta["dw"]), float(meta["dh"])
        box = (xyxy[i] - np.array([dw, dh, dw, dh], dtype=np.float32)) / r
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(w))
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(h))
        bw = max(0.0, float(box[2] - box[0]))
        bh = max(0.0, float(box[3] - box[1]))

        kxy = np.zeros((17, 2), dtype=np.float32)
        kconf = np.zeros(17, dtype=np.float32)
        kp = getattr(res, "keypoints", None)
        if kp is not None and kp.xy is not None and len(kp.xy) > i:
            kxy = (kp.xy[i].detach().float().cpu().numpy()
                   - np.array([dw, dh], dtype=np.float32)) / r
            if getattr(kp, "conf", None) is not None:
                kconf = (kp.conf[i].detach().float().cpu().numpy()
                         .reshape(-1).astype(np.float32))

        return {
            "n_det": int(len(boxes)),
            "bbox_px": [float(v) for v in box],
            "conf": float(conf[i]),
            "cx_norm": float(np.clip(0.5 * (box[0] + box[2]) / w, 0.0, 1.0)),
            "w_norm": float(np.clip(bw / w, 0.0, 1.0)),
            "h_norm": float(np.clip(bh / h, 0.0, 1.0)),
            "kxy": kxy,
            "kconf": kconf,
            "kp_conf_min": float(kconf.min()) if kconf.size else 0.0,
        }

    def _adapt(self, feat_raw):
        """(FEATURE_DIM,) → (64,)，特征只做一次前向，注意别把梯度带出来。"""
        with torch.no_grad():
            x = torch.from_numpy(
                np.ascontiguousarray(feat_raw, dtype=np.float32)).to(self.device)
            y = self.adapter(x)
        return y.detach().float().cpu().numpy().reshape(-1)

    def __repr__(self):
        return ("PerceptionPipeline(model=%r, device=%r, fov_h=%.1f, "
                "FEATURE_DIM=%d, feat_64=%d)"
                % (self.extractor.model_path, self.device, self.fov_h,
                   FEATURE_DIM, ADAPTER_OUT))
