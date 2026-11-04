# -*- coding: utf-8 -*-
"""controllers.py — 四种控制器，统一接口。

====== ====================================================================
名字   说明
====== ====================================================================
Teacher 规则教师（``pc/teacher/teacher.py`` + ``config.yaml``）。上界参考。
CfC     训练好的连续时间网络（``pc/train/policy.npz``），单 cell 64 单元 +
        保留门 + 5 头。用 numpy 复刻前向，逐位对齐 torch（``--verify`` 自检）。
MLP     无记忆基线：同样的 106 维观测、同样的 5 个头，但没有隐状态。
PD      最朴素：角度 → 头/机身按固定比例 P 控制，无死区、无限幅整形、无丢帧处理。
====== ====================================================================

统一接口::

    p = make(name)
    p.reset()
    out = p.step(frame, prev_act, sv)     # -> sim.CtrlOut

四种控制器**看到同一帧观测**、**同样的控制周期**，只有算法不同 —— 否则对比不成立。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from sim import CtrlOut, Frame, SensorView

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "train"
TEACHER_DIR = HERE.parent / "teacher"
for _d in (str(TRAIN_DIR), str(TEACHER_DIR)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

F32 = np.float32


# =========================================================================== #
# 基类                                                                        #
# =========================================================================== #
class Policy:
    """控制器基类：``reset()`` + ``step(frame, prev_act, sv) -> CtrlOut``。"""

    name = "base"
    #: 该控制器自己声明能用多大范围（教师是 42/60，CfC 是 45/90 的 tanh 满量程）
    head_max = 45.0
    vyaw_max = 90.0
    vx_max = 0.30

    def reset(self):
        """每条轨迹开始前必须清状态（隐状态、滤波器、状态机）。"""

    def step(self, frame: Frame, prev_act, sv: SensorView) -> CtrlOut:
        """推进一步。

        Args:
            frame: 本帧感知（视觉 + 手势 + 距离）。
            prev_act: 上一帧动作 ``(head, vyaw, vx, mode)``，固件用它填 obs[100..103]。
            sv: 本体感知视图（只有 PD 之类不用观测向量的控制器才需要）。

        Returns:
            :class:`sim.CtrlOut`
        """
        raise NotImplementedError


# =========================================================================== #
# 1) 教师（规则）                                                             #
# =========================================================================== #
class TeacherPolicy(Policy):
    """包一层 ``pc/teacher/teacher.py``。

    教师的参数全部来自 ``pc/teacher/config.yaml``（§8.3「代码里不留数字」），
    所以这里**没有**任何阈值 —— 只做 ``Frame`` → ``TeacherObs`` 的字段搬运。
    """

    name = "teacher"
    head_max = 42.0        # config.yaml tracking.head_max_deg
    vyaw_max = 60.0        # config.yaml tracking.vyaw_max_dps
    vx_max = 0.20          # config.yaml distance.vx_max_desktop（desktop_mode=true）

    def __init__(self):
        from teacher import TeacherController, load_config
        self._mod = sys.modules["teacher"]
        self.ctrl = TeacherController(load_config(TEACHER_DIR / "config.yaml"))
        self.head_max = float(self.ctrl.head_max_deg)
        self.vyaw_max = float(self.ctrl.vyaw_max_dps)
        self.vx_max = float(self.ctrl.vx_limit)

    def reset(self):
        self.ctrl.reset()

    def step(self, frame: Frame, prev_act, sv: SensorView) -> CtrlOut:
        TeacherObs = self._mod.TeacherObs
        obs = TeacherObs(
            theta_img=frame.theta_img, q_head=frame.q_head, seen=frame.seen,
            person_h=frame.person_h, hand_up_L=frame.hand_up_L,
            hand_up_R=frame.hand_up_R, t_now=frame.t_s,
            gesture_onset=frame.gesture_onset, gesture_id=frame.gesture_id,
            gesture_conf=frame.gesture_conf, gesture_gap_s=None)
        act = self.ctrl.step(obs)
        return CtrlOut(head_des=act.head_des, vx_des=act.vx_des,
                       vyaw_des=act.vyaw_des, mode=act.mode, emote=act.emote)


# =========================================================================== #
# 2) CfC（训练好的）                                                          #
# =========================================================================== #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _lecun(x):
    """ncps 的 ``LeCun`` 激活：``1.7159 * tanh(0.666 * x)``（ncp/torch/cfc_cell.py:33）。"""
    return 1.7159 * np.tanh(0.666 * x)


class CfCPolicy(Policy):
    """固件同一份权重的 numpy 前向（``pc/train/policy.npz``）。

    逐位复刻 ``train_cfc.Policy.forward_seq`` 的单帧：

        标准化(clip 4σ) → proj+SiLU → CfCCell(bb/ff1/ff2/ta/tb)
        → 保留门 h' = u + λ(h−u) → 5 头（tanh / tanh / softmax4 / softmax8 / tanh）

    输出按 ``cfc_task.c:457-472`` 的 publish() 换算成工程单位：
    ``head=out[0]*45``、``vyaw=out[1]*90``、``vx=out[14]*0.30``。

    Args:
        npz: ``policy.npz`` 路径。
    """

    name = "cfc"
    head_max = 45.0
    vyaw_max = 90.0
    vx_max = 0.30

    def __init__(self, npz=None):
        p = Path(npz) if npz else (TRAIN_DIR / "policy.npz")
        z = np.load(p)
        self.p = {k: np.asarray(z[k], dtype=F32) for k in z.files
                  if k not in ("sparsity_mask",)}
        self.input_dim = int(self.p["input_dim"])
        self.hidden = int(self.p["hidden_dim"])
        self.ts = float(self.p["ts"]) if "ts" in self.p else 1.0
        self.clip_sigma = float(self.p["clip_sigma"]) if "clip_sigma" in self.p else 4.0
        self.mean = self.p["input_mean"].reshape(-1)[:self.input_dim]
        self.std = np.maximum(self.p["input_std"].reshape(-1)[:self.input_dim], 1e-6)
        self.lam = _sigmoid(self.p["ret"].reshape(-1))
        self.head_dim = np.asarray(self.p["head_dim"]).reshape(-1)
        self.head_act = np.asarray(self.p["head_act"]).reshape(-1)
        self.n_heads = int(self.p["n_heads"])
        # 列位置与 train_cfc.Policy 的 col_* 一致
        self.col_head, self.col_vyaw = 0, 1
        self.col_mode, self.col_emote, self.col_vx = 2, 6, 14
        self.h = np.zeros(self.hidden, dtype=F32)
        self._heads = self._split_heads()

    # ------------------------------------------------------------------ #
    def _split_heads(self):
        """把打平的 ``head_w``/``head_b`` 切成 5 份。

        固件 ``blob_parse()`` 的切法：先按顺序切全部 weight，再切全部 bias。
        """
        hs = int(self.p["n_heads"])
        dims = [int(self.head_dim[i]) for i in range(hs)]
        w = self.p["head_w"].reshape(-1)
        b = self.p["head_b"].reshape(-1)
        out, off = [], 0
        for d in dims:
            out.append((w[off:off + d * self.hidden].reshape(d, self.hidden), None))
            off += d * self.hidden
        off = 0
        res = []
        for (w_, _), d in zip(out, dims):
            res.append((w_, b[off:off + d].copy()))
            off += d
        return res

    def reset(self):
        self.h = np.zeros(self.hidden, dtype=F32)

    def normalize(self, obs):
        """``cfc_model.c`` 的域外守护：非有限→均值、clip 4σ、再减均值除标准差。"""
        v = np.asarray(obs, dtype=F32).reshape(-1)[:self.input_dim]
        v = np.where(np.isfinite(v), v, self.mean)
        lo = self.mean - self.clip_sigma * self.std
        hi = self.mean + self.clip_sigma * self.std
        v = np.clip(v, lo, hi)
        return ((v - self.mean) / self.std).astype(F32)

    def forward(self, obs):
        """单步前向，返回 15 维原始输出（已过各头激活）。

        Args:
            obs: ``(106,)`` 原始观测（未归一化）。

        Returns:
            ``(15,) float32``
        """
        return self.forward_norm(self.normalize(obs))

    def forward_norm(self, x):
        """已经归一化过的输入的单步前向（自检时与 torch 对齐用）。

        Args:
            x: ``(input_dim,)`` 已标准化的观测。

        Returns:
            ``(15,) float32``
        """
        p = self.p
        z = p["proj_w"] @ x + p["proj_b"]
        z = z * (1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))   # SiLU = z*sigmoid(z)
        cat = np.concatenate([z, self.h]).astype(F32)
        bb = _lecun(p["bb_w"] @ cat + p["bb_b"])
        ff1 = np.tanh(p["ff1_w"] @ bb + p["ff1_b"])
        ff2 = np.tanh(p["ff2_w"] @ bb + p["ff2_b"])
        # ncps 的 time_a / time_b 是**原始线性输出**（cfc_cell.py forward 里没有再套
        # 激活），sigmoid 只作用在 t_a*ts + t_b 上。套错一个就整条隐状态全错。
        t_a = p["ta_w"] @ bb + p["ta_b"]
        t_b = p["tb_w"] @ bb + p["tb_b"]
        t_interp = _sigmoid(t_a * self.ts + t_b)
        u = ff1 * (1.0 - t_interp) + t_interp * ff2
        self.h = (u + self.lam * (self.h - u)).astype(F32)
        outs = []
        for i, (w, b) in enumerate(self._heads):
            o = w @ self.h + b
            act = int(self.head_act[i])
            if act == 1:
                o = np.tanh(o)
            # act==2 是 softmax 头，但固件**只存 logits**（cfc_model.c:1154-1156
            # "softmax is applied by the consumer"），publish() 走 argmax。
            # 这里照做，不要在这里 softmax —— 否则与固件逐位对不上。
            outs.append(o.astype(F32))
        return np.concatenate(outs)

    def step(self, frame: Frame, prev_act, sv: SensorView) -> CtrlOut:
        o = self.forward(frame.obs)
        head = float(np.clip(o[self.col_head], -1.0, 1.0)) * self.head_max
        vyaw = float(np.clip(o[self.col_vyaw], -1.0, 1.0)) * self.vyaw_max
        vx = float(np.clip(o[self.col_vx], -1.0, 1.0)) * self.vx_max
        mode = int(np.argmax(o[self.col_mode:self.col_mode + 4]))
        emote = int(np.argmax(o[self.col_emote:self.col_emote + 8]))
        return CtrlOut(head_des=head, vyaw_des=vyaw, vx_des=vx, mode=mode, emote=emote)


# =========================================================================== #
# 2b) CfC + 航向阻尼                                                          #
# =========================================================================== #
class CfCDamped(CfCPolicy):
    """CfC 策略 + 航向环 D 项（固件 ``robot.cc`` 的 ``k_dyaw``）。

    策略输出与 :class:`CfCPolicy` **逐位相同** —— D 项属于固件控制律
    （``yaw_u = k_yaw*(yaw-q_head-stable_yaw) - k_dyaw*yaw_rate``），不属于
    策略，所以这里只携带增益，由 :class:`sim.ClosedLoop` 注入
    ``FirmwareControl.k_dyaw``。这样"加阻尼"这件事只有一个实现位置。

    Args:
        npz: ``policy.npz`` 路径。
        k_dyaw: 航向 D 增益 (torque-count / (deg/s))。
    """

    name = "cfc_damp"

    def __init__(self, npz=None, k_dyaw=-0.5):
        super().__init__(npz=npz)
        self.k_dyaw = float(k_dyaw)


# =========================================================================== #
# 3) MLP（无记忆）                                                            #
# =========================================================================== #
class MLPPolicy(Policy):
    """无记忆基线：``obs(106) → 标准化 → 128 → 128 → 5 头``，无隐状态。

    与 CfC **同数据、同损失、同输出头**，唯一差别是把循环单元换成前馈 ——
    所以两者的差就是"时序记忆"的贡献。

    Args:
        npz: ``mlp_policy.npz`` 路径（由 ``train_mlp.py`` 产出）。
    """

    name = "mlp"
    head_max = 45.0
    vyaw_max = 90.0
    vx_max = 0.30

    def __init__(self, npz=None):
        p = Path(npz) if npz else (HERE / "mlp_policy.npz")
        if not p.exists():
            raise FileNotFoundError(
                "%s 不存在；先跑 train_mlp.py（或直接跑 eval.py，它会自动训练）" % p)
        z = np.load(p)
        self.w = [np.asarray(z["w%d" % i], dtype=F32) for i in range(len(
            [k for k in z.files if k.startswith("w")]))]
        self.b = [np.asarray(z["b%d" % i], dtype=F32) for i in range(len(self.w))]
        self.mean = np.asarray(z["input_mean"], dtype=F32).reshape(-1)
        self.std = np.maximum(np.asarray(z["input_std"], dtype=F32).reshape(-1), 1e-6)
        self.clip_sigma = float(z["clip_sigma"])
        self.head_w = np.asarray(z["head_w"], dtype=F32).reshape(-1)
        self.head_b = np.asarray(z["head_b"], dtype=F32).reshape(-1)
        self.n_heads = int(z["n_heads"])
        self.head_dim = np.asarray(z["head_dim"]).reshape(-1)
        self.head_act = np.asarray(z["head_act"]).reshape(-1)
        self.col_head, self.col_vyaw = 0, 1
        self.col_mode, self.col_emote, self.col_vx = 2, 6, 14

    def reset(self):
        pass

    def forward(self, obs):
        v = np.asarray(obs, dtype=F32).reshape(-1)
        v = np.where(np.isfinite(v), v, self.mean)
        v = np.clip(v, self.mean - self.clip_sigma * self.std,
                    self.mean + self.clip_sigma * self.std)
        x = ((v - self.mean) / self.std).astype(F32)
        for w, b in zip(self.w, self.b):
            # 每个 FC 之后都有 ReLU（train_mlp.MLP.forward 就是这么写的）
            x = np.maximum(w @ x + b, 0.0).astype(F32)
        outs, off = [], 0
        for i, d in enumerate(self.head_dim):
            d = int(d)
            w = self.head_w[off:off + d * x.shape[0]].reshape(d, x.shape[0])
            off += d * x.shape[0]
            outs.append(w @ x)
        res, off = [], 0
        for i, (o, d) in enumerate(zip(outs, self.head_dim)):
            d = int(d)
            o = o + self.head_b[off:off + d]
            off += d
            act = int(self.head_act[i])
            if act == 1:
                o = np.tanh(o)
            elif act == 2:
                e = np.exp(o - np.max(o))
                o = e / np.sum(e)
            res.append(o)
        return np.concatenate(res)

    def step(self, frame: Frame, prev_act, sv: SensorView) -> CtrlOut:
        o = self.forward(frame.obs)
        return CtrlOut(
            head_des=float(np.clip(o[self.col_head], -1, 1)) * self.head_max,
            vyaw_des=float(np.clip(o[self.col_vyaw], -1, 1)) * self.vyaw_max,
            vx_des=float(np.clip(o[self.col_vx], -1, 1)) * self.vx_max,
            mode=int(np.argmax(o[self.col_mode:self.col_mode + 4])),
            emote=int(np.argmax(o[self.col_emote:self.col_emote + 8])))


# =========================================================================== #
# 4) PD（最朴素）                                                             #
# =========================================================================== #
class PDPolicy(Policy):
    """角度误差 → 头 + 机身同时按固定比例 P 控制，别的一概没有。

    没有死区、没有三档分配、没有丢帧处理、没有速率限幅、没有头限位保护。
    它就是"不懂这个任务的人第一次会写出来的东西"，作为下界。

    Args:
        kp_head: 头通道 P 增益（deg/deg）。
        kp_yaw: 机身通道 P 增益（(deg/s)/deg）。
        kp_vx: 距离通道 P 增益（留 0：本基线不处理前后）。
    """

    name = "pd"
    head_max = 45.0
    vyaw_max = 60.0
    vx_max = 0.0

    def __init__(self, kp_head=1.0, kp_yaw=1.5, kp_vx=0.0):
        self.kp_head, self.kp_yaw, self.kp_vx = kp_head, kp_yaw, kp_vx
        self._eps_last = 0.0

    def reset(self):
        self._eps_last = 0.0

    def step(self, frame: Frame, prev_act, sv: SensorView) -> CtrlOut:
        if frame.seen:
            self._eps_last = frame.theta_img + frame.q_head
        eps = self._eps_last
        head = max(-self.head_max, min(self.head_max, self.kp_head * eps))
        vyaw = max(-self.vyaw_max, min(self.vyaw_max, self.kp_yaw * eps))
        vx = 0.0
        if self.kp_vx:
            d_err = (frame.dist_est - 1.0) if frame.seen else 0.0
            vx = max(-0.20, min(0.20, self.kp_vx * d_err))
        return CtrlOut(head_des=head, vyaw_des=vyaw, vx_des=vx, mode=1)


# =========================================================================== #
# 工厂                                                                        #
# =========================================================================== #
NAMES = ("teacher", "cfc", "mlp", "pd", "cfc_damp")


def make(name: str, **kw) -> Policy:
    """按名字造控制器。

    Args:
        name: ``teacher`` / ``cfc`` / ``mlp`` / ``pd``。
        **kw: 透传给具体类的构造参数。

    Returns:
        :class:`Policy`
    """
    if name == "teacher":
        return TeacherPolicy()
    if name == "cfc":
        return CfCPolicy(**kw)
    if name == "cfc_damp":
        return CfCDamped(**kw)
    if name == "mlp":
        return MLPPolicy(**kw)
    if name == "pd":
        return PDPolicy(**kw)
    raise ValueError("未知控制器 %r，可选 %s" % (name, NAMES))


# =========================================================================== #
# 自检：numpy CfC 是否与 torch 逐位一致                                       #
# =========================================================================== #
def verify_cfc(n_frames=64, tol=1e-5):
    """把 numpy 前向与 ``checkpoint.pt`` 的 torch 前向逐帧对比。

    **口径说明**：torch 的 ``train_cfc.Policy.forward_seq`` 输出的是**未激活**
    的线性头；固件 ``cfc_model.c:1149-1157`` 对 ``head_act==1`` 的头再套一个
    ``tanhf``。这里是"部署口径"对比，所以参考侧补上 tanh（softmax 头按固件
    保留 logits）。这个差异见 REPORT §局限。

    Args:
        n_frames: 随机帧数。
        tol: 判据（与 ``test_e2e.py`` 的 1e-5 同口径）。

    Returns:
        ``max_abs_err``
    """
    import torch
    import train_cfc

    pol = CfCPolicy()
    ck = torch.load(TRAIN_DIR / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    model = train_cfc.Policy(**{k: ck["config"][k] for k in
                                ("input_dim", "cell_in", "hidden", "backbone_units")})
    model.load_state_dict(ck["model"])
    model.eval()
    act = [int(a) for a in pol.head_act[:pol.n_heads]]
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 1.0, size=(n_frames, pol.input_dim)).astype(F32)
    xn = np.stack([pol.normalize(o) for o in obs])
    ref = []
    with torch.no_grad():
        z = torch.nn.functional.silu(model.proj(torch.from_numpy(xn)))
        hh = torch.zeros(1, model.hidden)
        lam = torch.sigmoid(model.ret)
        for k in range(n_frames):
            u, _ = model.cell(z[k:k + 1], hh, 1.0)
            hh = u + lam * (hh - u)
            outs = []
            for i, head in enumerate(model.heads):
                o = head(hh)
                if act[i] == 1:
                    o = torch.tanh(o)
                outs.append(o)
            ref.append(torch.cat(outs, -1)[0].numpy())
    err = 0.0
    for k in range(n_frames):
        o = pol.forward_norm(xn[k])
        err = max(err, float(np.max(np.abs(o - ref[k]))))
    return err
