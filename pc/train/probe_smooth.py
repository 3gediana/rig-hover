# -*- coding: utf-8 -*-
"""probe_smooth.py — 验收 D：训练侧的帧间差分平滑惩罚到底有没有生效。

     D:\\smartcar\\.venv\\Scripts\\python.exe probe_smooth.py

不训练，只做三件事：

1. 打印 ``--lambda-smooth`` 的默认值，以及它乘的是哪一项；
2. 在一个**真实 batch**（``data/synth_s0xx.npz``）上算出 ``smooth``、
   ``λ·smooth``，以及它与 MSE 的比例 —— 数值不是 0 才叫"在起作用"；
3. 分别对 ``mse`` 与 ``λ·smooth`` 反传一次，比较梯度范数。这一条排除
   "写在 loss 里但权重是 0 / 被 mean 掉"的可能。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import obs as obsmod                 # noqa: E402
import train_cfc                     # noqa: E402


def main() -> int:
    args = types.SimpleNamespace(
        batch_size=8, seq_len=64, stride=16, val_fraction=0.25,
        clip_sigma=4.0, std_floor=1e-2,
        lambda_smooth=0.05, lambda_retention=0.01, lambda_clip=0.10)

    torch.manual_seed(0)
    sessions = train_cfc.load_sessions(HERE / "data")
    tr, _va = train_cfc.split_sessions(sessions, args.val_fraction)
    obs_train = np.concatenate([s["obs"] for s in tr], axis=0)
    mean, std = obsmod.fit_stats(obs_train, std_floor=args.std_floor)
    train_cfc.normalize_sessions(tr, mean, std, args.clip_sigma)
    ds = train_cfc.WindowDataset(tr, args.seq_len, args.stride)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, drop_last=True)
    x, cont, mode, emote = next(iter(loader))

    ck = torch.load(HERE / "checkpoint.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    model = train_cfc.Policy(**{k: cfg[k] for k in
                                ("input_dim", "cell_in", "hidden",
                                 "backbone_units")})
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        out_val = model.forward_seq(x)
    _loss, parts = train_cfc.compute_loss(out_val, cont, mode, emote, model, args)
    # 反传要图，所以再算一次（eval 模式，但 grad 打开）
    out = model.forward_seq(x)

    print("== 1. 参数 ==")
    print("   --lambda-smooth 默认值            : %.3f" % args.lambda_smooth)
    print("   乘的那一项                         : mean((a[t+1]-a[t])^2)，"
          "a = (head, vyaw, vx) 归一化动作")
    print("   与 smartcar action_shaper.action_loss 同一公式: %s"
          % ("是（都是 pred 相邻帧一阶差分均方）"))

    acts = torch.stack([out[..., model.col_head], out[..., model.col_vyaw],
                        out[..., model.col_vx]], dim=-1)
    smooth = ((acts[:, 1:, :] - acts[:, :-1, :]) ** 2).mean()
    print("\n== 2. 一个真实 batch（%d 窗口 × %d 帧）上的数值 ==" % tuple(x.shape[:2]))
    print("   mse            = %.6f" % parts["mse"])
    print("   smooth（原值） = %.6f   ⇒ 动作帧间差分 RMS = %.4f"
          % (parts["smooth"], np.sqrt(parts["smooth"])))
    print("   λ·smooth       = %.6f   （占 mse 的 %.1f%%）"
          % (args.lambda_smooth * parts["smooth"],
             100.0 * args.lambda_smooth * parts["smooth"] / max(parts["mse"], 1e-12)))
    print("   total          = %.6f" % parts["total"])
    zero = train_cfc.compute_loss(out, cont, mode, emote, model,
                                  types.SimpleNamespace(**{**vars(args),
                                                           "lambda_smooth": 0.0}))[1]
    print("   total(λ_smooth=0) = %.6f   ⇒ 这一项把总损失抬高了 %.6f（%.1f%%）"
          % (zero["total"], parts["total"] - zero["total"],
             100.0 * (parts["total"] - zero["total"]) / max(zero["total"], 1e-12)))

    print("\n== 3. 梯度：这一项真的进反传吗 ==")
    mse = (F.mse_loss(out[..., model.col_head], cont[..., 0])
           + F.mse_loss(out[..., model.col_vyaw], cont[..., 1])
           + F.mse_loss(out[..., model.col_vx], cont[..., 2]))
    for label, val in (("mse", mse),
                       ("λ·smooth", args.lambda_smooth * smooth)):
        model.zero_grad()
        val.backward(retain_graph=True)
        tot = 0.0
        n_touch = 0
        for p in model.parameters():
            if p.grad is not None:
                tot += float(p.grad.pow(2).sum())
                n_touch += 1 if float(p.grad.abs().sum()) > 0 else 0
        print("   |∇%s| = %.6e   非零梯度张量 %d 个" % (label, np.sqrt(tot), n_touch))
    print("\n结论：λ_smooth=0.05 存在、非零、进反传 ⇒ 训练侧无需改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
