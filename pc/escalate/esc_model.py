# -*- coding: utf-8 -*-
"""esc_model.py — CfC 决策网络（5 头）+ 升级头（第 6 头）。

结构与 ``pc/train/train_cfc.py::Policy`` **逐层相同**（同一份 blob 布局）：

    obs(106) -> 归一化 -> Linear(106->64) -> SiLU
             -> CfCCell(cell_in=64, hidden=64, backbone=64)   # 单 cell
             -> 保留门 h' = u + lam*(h-u), lam = sigmoid(ret)
             -> 6 个头: head_yaw(tanh) vyaw(tanh) mode(softmax4) emote(softmax8) vx(tanh)
                        escalate(sigmoid)

:class:`Policy` 用 ``strict=True`` 加载 ``base/base_checkpoint.pt`` —— 能加载成功
这件事本身就是"本文件与 train_cfc.Policy 的层名/形状完全一致"的证明；
:class:`EscalatePolicy` 在它上面多一个 ``heads[5]``，加载时缺的键必须**恰好**
是 ``heads.5.*``（:func:`load_base_into` 会断言这一点）。

升级头读的是**同一个** ``h``：保留门之后、其余 5 个头用的那一份（不是另一路
分支、不是池化后的摘要）。这是本任务的核心约束，也是"固件只跑一次 cfc_step"
这个前提能成立的原因。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

F32 = np.float32

HERE = Path(__file__).resolve().parent
BASE_CKPT = HERE / "base" / "base_checkpoint.pt"

#: 6 个头的定义（顺序 = blob 里 head_dim/head_act 的下标）
#: 前 5 个与 ``pc/train/export_blob.py::HEADS`` 完全一致，第 6 个是新增的。
HEADS6 = (
    ("head_yaw", 1, "tanh"),
    ("vyaw", 1, "tanh"),
    ("mode", 4, "softmax"),
    ("emote", 8, "softmax"),
    ("vx", 1, "tanh"),
    ("escalate", 1, "sigmoid"),
)
HEAD_DIMS6 = tuple(d for _n, d, _a in HEADS6)
N_COLS6 = sum(HEAD_DIMS6)          # 16
#: 每个头在输出列里的起始下标
COL = {}
_acc = 0
for _n, _d, _a in HEADS6:
    COL[_n] = _acc
    _acc += _d
COL_ESCALATE = COL["escalate"]     # 15

#: 动作满量程（固件 arb_cfg_t 默认值）
HEAD_MAX = 45.0
VYAW_MAX = 90.0
VX_MAX = 0.30


class Policy(nn.Module):
    """与 ``train_cfc.Policy`` 逐层相同的 5 头网络（只用来加载已冻结的 base）。"""

    def __init__(self, input_dim=106, cell_in=64, hidden=64, backbone_units=64,
                 ret_init=-3.0, head_dims=(1, 1, 4, 8, 1)):
        super().__init__()
        from ncps.torch import CfCCell

        self.input_dim = int(input_dim)
        self.cell_in = int(cell_in)
        self.hidden = int(hidden)
        self.head_dims = tuple(head_dims)
        self.proj = nn.Linear(self.input_dim, self.cell_in)
        self.cell = CfCCell(self.cell_in, self.hidden, mode="default",
                            backbone_activation="lecun_tanh",
                            backbone_units=int(backbone_units),
                            backbone_layers=1, backbone_dropout=0.0)
        self.ret = nn.Parameter(torch.full((self.hidden,), float(ret_init)))
        self.heads = nn.ModuleList([nn.Linear(self.hidden, d) for d in self.head_dims])

    def forward_seq(self, x, h=None):
        """``(B,T,106)`` -> ``(B,T,15)`` 原始输出（softmax 头保留 logits）。"""
        z = F.silu(self.proj(x))
        b, t, _ = z.shape
        if h is None:
            h = torch.zeros(b, self.hidden, device=z.device, dtype=z.dtype)
        lam = torch.sigmoid(self.ret)
        outs = []
        for k in range(t):
            u, _ = self.cell(z[:, k], h, 1.0)
            h = u + lam * (h - u)
            outs.append(torch.cat([head(h) for head in self.heads], dim=-1))
        return torch.stack(outs, dim=1)


class EscalatePolicy(Policy):
    """5 头基础网络 + 第 6 个 ``escalate`` 头（共享同一个隐状态）。

    Args:
        input_dim / cell_in / hidden / backbone_units / ret_init: 同 :class:`Policy`。
        esc_init_std: 升级头权重的初始化标准差（小随机，别用 0 —— 全 0 时主干
            在第一步拿不到梯度，白白浪费 warmup）。
        esc_init_bias: 升级头偏置初值。默认 ``logit`` 一个先验基率，让训练从
            "校准过的小概率"出发而不是从 0.5 出发。
    """

    def __init__(self, input_dim=106, cell_in=64, hidden=64, backbone_units=64,
                 ret_init=-3.0, esc_init_std=0.01, esc_init_bias=None):
        super().__init__(input_dim=input_dim, cell_in=cell_in, hidden=hidden,
                         backbone_units=backbone_units, ret_init=ret_init,
                         head_dims=HEAD_DIMS6[:5])
        self.heads.append(nn.Linear(self.hidden, 1))
        self.esc_init_std = float(esc_init_std)
        b = -4.0 if esc_init_bias is None else float(esc_init_bias)
        with torch.no_grad():
            self.heads[-1].weight.normal_(0.0, self.esc_init_std)
            self.heads[-1].bias.fill_(b)
        self.head_dims = HEAD_DIMS6

    # ------------------------------------------------------------------
    def forward_seq(self, x, h=None, return_hidden=False):
        """``(B,T,106)`` -> ``(B,T,16)``（第 15 列是 escalate 的 logit）。

        Args:
            x: 已归一化的观测。
            h: 初值；``None`` 表示清零（每条序列必须清零）。
            return_hidden: 为 True 时额外返回 ``(B,T,hidden)`` 的隐状态轨迹。

        Returns:
            ``out`` 或 ``(out, hs)``。softmax/tanh 头都是**原始输出**，
            激活由消费方（固件 / numpy 参考）套 —— 与 ``cfc_model_forward`` 的
            口径一致：tanh 由模型套，softmax 留 logits。
        """
        z = F.silu(self.proj(x))
        b, t, _ = z.shape
        if h is None:
            h = torch.zeros(b, self.hidden, device=z.device, dtype=z.dtype)
        lam = torch.sigmoid(self.ret)
        outs = []
        hs = []
        for k in range(t):
            u, _ = self.cell(z[:, k], h, 1.0)
            h = u + lam * (h - u)
            outs.append(torch.cat([head(h) for head in self.heads], dim=-1))
            if return_hidden:
                hs.append(h)
        out = torch.stack(outs, dim=1)
        if return_hidden:
            return out, torch.stack(hs, dim=1)
        return out

    def escalate_logit(self, out):
        """从 ``forward_seq`` 的输出里取 escalate 头的 logit（``(B,T)``）。"""
        return out[..., COL_ESCALATE]

    def lambda_mean(self):
        with torch.no_grad():
            return float(torch.sigmoid(self.ret).mean().item())


def load_base_policy(path=BASE_CKPT, device="cpu"):
    """加载冻结的 5 头 checkpoint（``strict=True``，证明结构逐层一致）。

    Returns:
        ``(model, mean, std, ckpt)``
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = Policy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                   hidden=cfg["hidden"], backbone_units=cfg["backbone_units"],
                   ret_init=cfg.get("ret_init", -3.0),
                   head_dims=tuple(cfg["head_dims"]))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    model.to(device)
    mean = np.asarray(ck["input_mean"], dtype=F32)
    std = np.asarray(ck["input_std"], dtype=F32)
    return model, mean, std, ck


def load_base_into(esc_policy, path=BASE_CKPT):
    """把 5 头 base 的权重灌进 :class:`EscalatePolicy`（第 6 头保持初始化值）。

    Returns:
        dict：``{"loaded": [...], "missing": [...]}``。``missing`` 必须恰好是
        ``heads.5.weight/bias``，否则说明结构对不上，直接抛错。
    """
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = dict(ck["model"])
    sd["heads.5.weight"] = esc_policy.heads[5].weight.detach().clone()
    sd["heads.5.bias"] = esc_policy.heads[5].bias.detach().clone()
    missing, unexpected = esc_policy.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.startswith("heads.5.")]
    if bad_missing or unexpected:
        raise RuntimeError("base 权重与 EscalatePolicy 不匹配：missing=%s unexpected=%s"
                           % (bad_missing, list(unexpected)))
    return {"missing": list(missing), "config": ck["config"],
            "input_mean": np.asarray(ck["input_mean"], dtype=F32),
            "input_std": np.asarray(ck["input_std"], dtype=F32)}


# --------------------------------------------------------------------------- #
# 反向：把网络的输出列拆成工程单位动作（与固件 publish() 同口径）
# --------------------------------------------------------------------------- #
def decode_actions(out, with_escalate=False):
    """``(B,T,16)`` 原始输出 -> 工程单位动作 + 动作置信度 + escalate 概率。

    Args:
        out: :meth:`EscalatePolicy.forward_seq` 的输出。
        with_escalate: 为 True 时额外给出 ``p_esc``（sigmoid）。

    Returns:
        dict：``head/vyaw/vx``（工程单位）、``mode``（int argmax）、
        ``emote``（int argmax）、``mode_conf``（top1−top2 的 softmax 概率差）、
        ``p_esc``（可选的 ``(B,T)``）。
    """
    head = torch.tanh(out[..., COL["head_yaw"]]) * HEAD_MAX
    vyaw = torch.tanh(out[..., COL["vyaw"]]) * VYAW_MAX
    vx = torch.tanh(out[..., COL["vx"]]) * VX_MAX
    mode_logits = out[..., COL["mode"]:COL["mode"] + 4]
    emote_logits = out[..., COL["emote"]:COL["emote"] + 8]
    pm = torch.softmax(mode_logits, dim=-1)
    top2 = torch.topk(pm, 2, dim=-1).values
    res = {
        "head": head, "vyaw": vyaw, "vx": vx,
        "mode": mode_logits.argmax(-1),
        "emote": emote_logits.argmax(-1),
        "mode_conf": top2[..., 0] - top2[..., 1],
    }
    if with_escalate:
        res["p_esc"] = torch.sigmoid(out[..., COL_ESCALATE])
    return res


def prior_logit(rate, eps=1e-6):
    """基率 -> logit（升级头偏置的初值）。"""
    r = min(max(float(rate), eps), 1.0 - eps)
    return math.log(r / (1.0 - r))
