# -*- coding: utf-8 -*-
"""train_mlp.py — 训练无记忆基线 MLP（与 CfC 同数据、同损失、同输出头）。

唯一差别：**没有隐状态**。所以 "CfC − MLP" 就是时序记忆在本任务上的净贡献。

数据来自 ``pc/train/data/*.npz``（``synth.py`` 用真教师打的标签）。这里只读，
不写回 ``pc/train/``。归一化统计量**直接沿用 CfC 的** ``input_mean/std``，
否则两个模型的观测口径不同，对比不成立。

用法::

    python train_mlp.py                 # 训练并写 mlp_policy.npz
    python train_mlp.py --epochs 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent / "train"
OUT = HERE / "mlp_policy.npz"

HEAD_MAX, VYAW_MAX, VX_MAX = 45.0, 90.0, 0.30
MODE_WEIGHT = (1.0, 1.0, 20.0, 15.0)
HEAD_DIMS = (1, 1, 4, 8, 1)
HEAD_ACT = (1, 1, 2, 2, 1)          # 1=tanh 2=softmax(存 logits) 与 CfC blob 一致


class MLP(nn.Module):
    """``obs(106) → 128 → 128 → 5 头``。

    Args:
        input_dim: 观测维度。
        hidden: 两个隐层宽度。
    """

    def __init__(self, input_dim=106, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in HEAD_DIMS])

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        outs = []
        for i, head in enumerate(self.heads):
            o = head(h)
            if HEAD_ACT[i] == 1:
                o = torch.tanh(o)
            outs.append(o)
        return torch.cat(outs, -1)


def load_data(data_dir, mean, std, clip_sigma=4.0):
    """读全部会话，返回 ``(obs_norm, head, vyaw, vx, mode, emote)``。"""
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError("%s 下没有 npz" % data_dir)
    xs, hs, ys, vs, ms, es = [], [], [], [], [], []
    for p in files:
        z = np.load(p)
        xs.append(np.asarray(z["obs"], dtype=np.float32))
        hs.append(np.asarray(z["head_deg"], dtype=np.float32))
        ys.append(np.asarray(z["vyaw_dps"], dtype=np.float32))
        vs.append(np.asarray(z["vx_mps"], dtype=np.float32))
        ms.append(np.asarray(z["mode"], dtype=np.int64))
        es.append(np.asarray(z["emote"], dtype=np.int64))
    x = np.concatenate(xs)
    x = np.clip(x, mean - clip_sigma * std, mean + clip_sigma * std)
    x = ((x - mean) / std).astype(np.float32)
    return (x, np.concatenate(hs), np.concatenate(ys), np.concatenate(vs),
            np.concatenate(ms), np.concatenate(es))


def train(args):
    from controllers import CfCPolicy
    ref = CfCPolicy()
    mean, std = ref.mean, ref.std
    x, head, vyaw, vx, mode, emote = load_data(args.data_dir, mean, std,
                                               ref.clip_sigma)
    print("[data] %d 帧，mode 分布 %s" % (len(x),
          {int(k): int(v) for k, v in zip(*np.unique(mode, return_counts=True))}))
    dev = torch.device("cuda:0" if (args.device == "auto" and
                                    torch.cuda.is_available()) else "cpu")
    model = MLP(x.shape[1], args.hidden).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print("[model] MLP %d 参数（CfC 是 32783 —— 参数更多，所以 CfC 的优势不能归给算力）"
          % n_par)
    X = torch.from_numpy(x)
    Yh = torch.from_numpy(head / HEAD_MAX)
    Yy = torch.from_numpy(vyaw / VYAW_MAX)
    Yv = torch.from_numpy(vx / VX_MAX)
    M = torch.from_numpy(mode)
    E = torch.from_numpy(emote)
    w = torch.tensor(MODE_WEIGHT)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(X)
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            out = model(X[idx].to(dev))
            mse = (F.mse_loss(out[:, 0], Yh[idx].to(dev))
                   + F.mse_loss(out[:, 1], Yy[idx].to(dev))
                   + F.mse_loss(out[:, 14], Yv[idx].to(dev)))
            ce_m = F.cross_entropy(out[:, 2:6], M[idx].to(dev), weight=w.to(dev))
            ce_e = F.cross_entropy(out[:, 6:14], E[idx].to(dev))
            loss = mse + ce_m + ce_e
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep == 1 or ep % max(1, args.epochs // 6) == 0:
            print("[epoch %2d] loss=%.4f  (%.1fs)" % (ep, tot / nb,
                                                      time.perf_counter() - t0))
    model.eval()
    with torch.no_grad():
        pred = model(X.to(dev)).cpu().numpy()
    acc = float((pred[:, 2:6].argmax(1) == mode).mean())
    print("[eval] 全量 mode 准确率 %.3f" % acc)

    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    head_w = np.concatenate([sd["heads.%d.weight" % i] for i in range(len(HEAD_DIMS))],
                            axis=0).astype(np.float32)
    head_b = np.concatenate([sd["heads.%d.bias" % i] for i in range(len(HEAD_DIMS))],
                            axis=0).astype(np.float32)
    np.savez(
        OUT, w0=sd["fc1.weight"], b0=sd["fc1.bias"],
        w1=sd["fc2.weight"], b1=sd["fc2.bias"],
        head_w=head_w, head_b=head_b,
        n_heads=np.int32(len(HEAD_DIMS)),
        head_dim=np.asarray(HEAD_DIMS, dtype=np.int32),
        head_act=np.asarray(HEAD_ACT, dtype=np.int32),
        input_mean=np.asarray(mean, dtype=np.float32),
        input_std=np.asarray(std, dtype=np.float32),
        clip_sigma=np.float32(ref.clip_sigma),
        params=np.int32(n_par), mode_acc=np.float32(acc),
    )
    print("[save] %s（%d 参数，%.1f KB）" % (OUT, n_par, OUT.stat().st_size / 1024))
    print("PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="训练无记忆 MLP 基线（同数据/同头/同归一化，唯一差别是没有隐状态）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default=str(TRAIN_DIR / "data"))
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
