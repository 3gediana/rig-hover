# -*- coding: utf-8 -*-
"""train_cfc.py — 训练 CfC 策略（ncps CfCCell + 保留门），并落 ``checkpoint.pt`` / ``policy.npz``。

网络结构（刻意与固件 blob 一一对应，别加层）：

    obs(106) -> 域外守护+标准化 -> Linear(106->64) -> SiLU
             -> CfCCell(cell_in=64, hidden=64, backbone_units=64)   # 单 cell
             -> 保留门 h' = u + lam*(h-u),  lam = sigmoid(ret), ret 初值 -3.0
             -> 5 个头: head_yaw(tanh) vyaw(tanh) mode(softmax4) emote(softmax8) vx(tanh)

为什么是**单个** cell 而不是 ``AutoNCP(64,2)``：`cfc_model.h` 的 blob 里只有
一份 ``cfc_cell_t`` 的权重槽（bb/ff1/ff2/ta/tb/ret 各一份，layer_off 的 14 个
层 id 全部属于同一个 cell），``cfc_model_forward()`` 也只调用一次 ``cfc_step()``。
AutoNCP(64,2) 实际是 3 个串联 cell（见 ``cfc_host/NOTES.md`` 第 5 条），塞不进
这个格式 —— 固件是权威，所以模型侧收敛到"单 cell 64 单元"，与 TECH-PLAN §6.1
（RetentionCfCCell × 1，hidden 64）一致。

损失（TECH-PLAN §10.1 + 本次任务要求）：

    L = MSE(head/45) + MSE(vyaw/90) + MSE(vx/0.30)
      + CE(mode, weight=[1,1,20,15]) + CE(emote)
      + 0.05 * 动作一阶差分平滑
      + 0.01 * mean(sigmoid(ret))                 # 轻推保留门，别学成恒等
      + 0.10 * mean(relu(|a|-1))                  # tanh 头下恒为 0，留着做保险

用法::

    python train_cfc.py --data-dir data --epochs 6
    python train_cfc.py --quick                    # 验收用的最小训练
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import export_blob
import obs as obsmod

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流没有 reconfigure
        pass

HERE = Path(__file__).resolve().parent
F32 = np.float32

#: 连续动作的满量程（= 固件 arb_cfg_t 的 head_max / vyaw_max / vx_max 默认值）
HEAD_MAX = 45.0
VYAW_MAX = 90.0
VX_MAX = 0.30

#: mode 类权重：ACK×20 / REFRACTORY×15（TECH-PLAN §10.1）
MODE_WEIGHT = (1.0, 1.0, 20.0, 15.0)

#: 5 个头的定义，顺序与固件 publish() 一致（export_blob.HEADS 是唯一来源）
HEAD_DIMS = tuple(d for _n, d, _a in export_blob.HEADS)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class Policy(nn.Module):
    """与固件 blob 一一对应的策略网络。

    Args:
        input_dim: 观测维度（默认 106）。
        cell_in: 投影层输出 = CfC cell 的输入维度（默认 64）。
        hidden: CfC 隐藏单元数（默认 64）。
        backbone_units: ncps backbone 宽度；**必须显式传 64**，ncps 默认 128。
        ret_init: 保留门 ``ret`` 初值，**-3.0**（-6 落在 sigmoid 死区）。
    """

    def __init__(self, input_dim=obsmod.OBS_DIM, cell_in=64, hidden=64,
                 backbone_units=64, ret_init=-3.0):
        super().__init__()
        from ncps.torch import CfCCell

        self.input_dim = int(input_dim)
        self.cell_in = int(cell_in)
        self.hidden = int(hidden)
        self.backbone_units = int(backbone_units)
        self.head_dims = HEAD_DIMS

        self.proj = nn.Linear(self.input_dim, self.cell_in)
        self.cell = CfCCell(self.cell_in, self.hidden, mode="default",
                            backbone_activation="lecun_tanh",
                            backbone_units=self.backbone_units,
                            backbone_layers=1, backbone_dropout=0.0)
        if len(self.cell.backbone) == 0:
            raise RuntimeError("CfCCell 没有生成 backbone（backbone_layers=1 时应当有）")
        self.ret = nn.Parameter(torch.full((self.hidden,), float(ret_init)))
        self.heads = nn.ModuleList([nn.Linear(self.hidden, d) for d in self.head_dims])
        # 输出列在 (B,T,15) 里的位置
        self.col_head = 0
        self.col_vyaw = 1
        self.col_mode = 2
        self.col_emote = 6
        self.col_vx = 14

    # ------------------------------------------------------------------
    def forward_seq(self, x, h=None):
        """整段前向（逐帧递推）。

        Args:
            x: ``(B, T, input_dim)`` 已归一化的观测。
            h: ``(B, hidden)`` 初值；``None`` 表示**清零**（每条序列必须清零）。

        Returns:
            ``(B, T, sum(head_dims))``。
        """
        from torch import sigmoid  # 局部 import 让文件头保持轻

        z = F.silu(self.proj(x))
        b, t, _ = z.shape
        if h is None:
            h = torch.zeros(b, self.hidden, device=z.device, dtype=z.dtype)
        lam = sigmoid(self.ret)
        outs = []
        for k in range(t):
            u, _ = self.cell(z[:, k], h, 1.0)
            h = u + lam * (h - u)          # 保留门，与 cfc_engine.c 同序
            outs.append(torch.cat([head(h) for head in self.heads], dim=-1))
        return torch.stack(outs, dim=1)

    # ------------------------------------------------------------------
    def lambda_mean(self):
        """当前保留门均值（每 epoch 必须打印，用来判断门有没有在动）。"""
        with torch.no_grad():
            return float(torch.sigmoid(self.ret).mean().item())


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_sessions(data_dir):
    """读 ``data_dir`` 下所有 ``*.npz``，按文件名排序返回 list[dict]。"""
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError("%s 下没有 npz，先跑 synth.py" % data_dir)
    sessions = []
    for p in files:
        z = np.load(p)
        sessions.append({
            "file": p.name,
            "obs": np.ascontiguousarray(z["obs"], dtype=F32),
            "head": np.ascontiguousarray(z["head_deg"], dtype=F32),
            "vyaw": np.ascontiguousarray(z["vyaw_dps"], dtype=F32),
            "vx": np.ascontiguousarray(z["vx_mps"], dtype=F32),
            "mode": np.ascontiguousarray(z["mode"], dtype=np.int64),
            "emote": np.ascontiguousarray(z["emote"], dtype=np.int64),
        })
    return sessions


def split_sessions(sessions, val_fraction=0.25):
    """**按会话**划分训练/验证（绝不按帧随机划分，TECH-PLAN §9.3）。

    Returns:
        ``(train, val)``
    """
    n = len(sessions)
    n_val = max(1, int(round(n * float(val_fraction)))) if n > 1 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    ordered = list(sessions)
    return ordered[:n - n_val], ordered[n - n_val:] if n_val else []


def normalize_sessions(sessions, mean, std, clip_sigma=4.0):
    """把每条会话的 obs 归一化（原地换掉 ``obs_n`` 字段），并统计 clip 比例。"""
    ratio = 0.0
    total = 0
    for s in sessions:
        x, r = obsmod.normalize_obs(s["obs"], mean, std, clip_sigma)
        s["obs_n"] = np.ascontiguousarray(x, dtype=F32)
        ratio += r * s["obs"].size
        total += s["obs"].size
    return ratio / max(total, 1)


class WindowDataset(torch.utils.data.Dataset):
    """按 (会话, 起点) 切定长窗口；窗口内标签与观测同帧。

    Args:
        sessions: 已归一化的会话列表（需要有 ``obs_n``）。
        seq_len: 窗口长度（帧）。
        stride: 窗口起点步长（帧）。
    """

    def __init__(self, sessions, seq_len=64, stride=16):
        self.items = []
        self.sessions = sessions
        for si, s in enumerate(sessions):
            t = len(s["mode"])
            if t < seq_len:
                continue
            starts = range(0, t - seq_len + 1, max(1, int(stride)))
            self.items.extend((si, st) for st in starts)
        if not self.items:
            raise ValueError("会话太短，切不出 seq_len=%d 的窗口" % seq_len)
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, st = self.items[i]
        s = self.sessions[si]
        sl = slice(st, st + self.seq_len)
        x = torch.from_numpy(s["obs_n"][sl])
        cont = np.stack([
            s["head"][sl] / HEAD_MAX,
            s["vyaw"][sl] / VYAW_MAX,
            s["vx"][sl] / VX_MAX,
        ], axis=-1)
        return (x, torch.from_numpy(np.ascontiguousarray(cont, dtype=F32)),
                torch.from_numpy(s["mode"][sl]), torch.from_numpy(s["emote"][sl]))


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def compute_loss(out, cont, mode, emote, model, args):
    """返回 ``(loss, parts)``；``parts`` 是各分项（浮点，便于打印）。"""
    pred_head = out[..., model.col_head]
    pred_vyaw = out[..., model.col_vyaw]
    pred_vx = out[..., model.col_vx]
    mse = (F.mse_loss(pred_head, cont[..., 0])
           + F.mse_loss(pred_vyaw, cont[..., 1])
           + F.mse_loss(pred_vx, cont[..., 2]))

    mode_logits = out[..., model.col_mode:model.col_mode + 4]
    emote_logits = out[..., model.col_emote:model.col_emote + 8]
    weight = torch.tensor(MODE_WEIGHT, device=out.device, dtype=out.dtype)
    ce_mode = F.cross_entropy(mode_logits.reshape(-1, 4), mode.reshape(-1), weight=weight)
    ce_emote = F.cross_entropy(emote_logits.reshape(-1, 8), emote.reshape(-1))

    acts = torch.stack([pred_head, pred_vyaw, pred_vx], dim=-1)   # (B,T,3)
    smooth = ((acts[:, 1:, :] - acts[:, :-1, :]) ** 2).mean() if acts.shape[1] > 1 \
        else torch.zeros((), device=out.device)
    lam = torch.sigmoid(model.ret).mean()
    clip = torch.relu(acts.abs() - 1.0).mean()

    loss = (mse + ce_mode + ce_emote
            + args.lambda_smooth * smooth
            + args.lambda_retention * lam
            + args.lambda_clip * clip)
    parts = {
        "mse": float(mse.item()),
        "ce_mode": float(ce_mode.item()),
        "ce_emote": float(ce_emote.item()),
        "smooth": float(smooth.item()),
        "lam": float(lam.item()),
        "clip": float(clip.item()),
        "total": float(loss.item()),
    }
    return loss, parts


def evaluate(model, loader, args, device):
    """验证集：``model.eval() + no_grad``，且**每条序列隐状态清零**。"""
    model.eval()
    tot = {}
    n = 0
    correct = 0
    frames = 0
    with torch.no_grad():
        for x, cont, mode, emote in loader:
            x = x.to(device=device, dtype=torch.float32)
            cont = cont.to(device)
            mode = mode.to(device)
            emote = emote.to(device)
            out = model.forward_seq(x)          # h=None -> 每条序列清零
            _loss, parts = compute_loss(out, cont, mode, emote, model, args)
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
            logits = out[..., model.col_mode:model.col_mode + 4]
            correct += int((logits.argmax(-1) == mode).sum().item())
            frames += int(mode.numel())
    model.train()
    avg = {k: v / max(n, 1) for k, v in tot.items()}
    avg["mode_acc"] = correct / max(frames, 1)
    return avg


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    t_start = time.perf_counter()

    sessions = load_sessions(args.data_dir)
    train_sess, val_sess = split_sessions(sessions, args.val_fraction)
    print("[data] %d 条会话: 训练 %d / 验证 %d（按会话划分，不按帧）"
          % (len(sessions), len(train_sess), len(val_sess)))
    print("        训练集: %s" % [s["file"] for s in train_sess])
    print("        验证集: %s" % [s["file"] for s in val_sess])

    obs_train = np.concatenate([s["obs"] for s in train_sess], axis=0)
    mean, std = obsmod.fit_stats(obs_train, std_floor=args.std_floor)
    print("[data] input_mean/std 只在训练集上统计（%d 帧）；std 下限 %.0e（%d 维被抬起）"
          % (obs_train.shape[0], args.std_floor, int(np.sum(std <= args.std_floor))))
    ratio_tr = normalize_sessions(train_sess, mean, std, args.clip_sigma)
    ratio_va = normalize_sessions(val_sess, mean, std, args.clip_sigma) if val_sess else 0.0
    print("[data] clip 比例: 训练 %.4f%% / 验证 %.4f%%（>5%% 说明喂了训练分布外的数据）"
          % (100.0 * ratio_tr, 100.0 * ratio_va))

    ds_tr = WindowDataset(train_sess, args.seq_len, args.stride)
    loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size,
                                            shuffle=True, drop_last=len(ds_tr) > args.batch_size)
    if val_sess:
        ds_va = WindowDataset(val_sess, args.seq_len, args.stride)
        loader_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch_size,
                                                shuffle=False)
    else:
        loader_va = None
    print("[data] 训练窗口 %d 个（seq_len=%d stride=%d batch=%d），验证窗口 %d 个"
          % (len(ds_tr), args.seq_len, args.stride, args.batch_size,
             len(ds_va) if val_sess else 0))

    model = Policy(input_dim=obs_train.shape[1], cell_in=args.cell_in,
                   hidden=args.hidden, backbone_units=args.backbone_units,
                   ret_init=args.ret_init).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print("[model] cell_in=%d hidden=%d backbone_units=%d heads=%s 参数 %d 个（%.1f KB fp32）"
          % (args.cell_in, args.hidden, args.backbone_units, HEAD_DIMS, n_par,
             n_par * 4 / 1024.0))
    print("[model] 保留门 ret 初值 %.2f -> 均值 lambda=%.4f"
          % (args.ret_init, model.lambda_mean()))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    history = []
    lam0 = model.lambda_mean()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot = {}
        n = 0
        for x, cont, mode, emote in loader_tr:
            x = x.to(device=device, dtype=torch.float32)
            cont = cont.to(device)
            mode = mode.to(device)
            emote = emote.to(device)
            out = model.forward_seq(x)
            loss, parts = compute_loss(out, cont, mode, emote, model, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v
            n += 1
        sched.step()
        tr = {k: v / max(n, 1) for k, v in tot.items()}
        tr["mode_acc"] = float("nan")
        va = evaluate(model, loader_va, args, device) if loader_va else {}
        lam = model.lambda_mean()
        history.append({"epoch": epoch, "train": tr, "val": va, "lam": lam,
                        "sec": time.perf_counter() - t0})
        print("[epoch %2d] train total=%.4f (mse=%.4f ce_mode=%.4f ce_emote=%.4f "
              "smooth=%.5f clip=%.5f) | val total=%s mode_acc=%s | lambda=%.4f | %.1fs"
              % (epoch, tr["total"], tr["mse"], tr["ce_mode"], tr["ce_emote"],
                 tr["smooth"], tr["clip"],
                 ("%.4f" % va["total"]) if va else "-",
                 ("%.3f" % va["mode_acc"]) if va else "-",
                 lam, time.perf_counter() - t0))

    elapsed = time.perf_counter() - t_start
    lam_end = model.lambda_mean()
    print("[train] 总耗时 %.1f s，lambda %.4f -> %.4f（%s）"
          % (elapsed, lam0, lam_end,
             "门在动" if abs(lam_end - lam0) > 1e-3 else "门基本没动（正常：任务不需要长记忆）"))

    # ---- 落盘 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {"input_dim": int(obs_train.shape[1]), "cell_in": args.cell_in,
                   "hidden": args.hidden, "backbone_units": args.backbone_units,
                   "ret_init": args.ret_init, "head_dims": list(HEAD_DIMS),
                   "clip_sigma": args.clip_sigma, "std_floor": args.std_floor},
        "input_mean": mean, "input_std": std,
        "history": history, "epochs": args.epochs, "elapsed_s": elapsed,
    }, ckpt_path)
    print("[save] %s" % ckpt_path)

    model.eval()
    policy = export_blob.collect_from_torch(model)
    npz_path = out_dir / "policy.npz"
    export_blob.save_policy_npz(npz_path, policy, mean, std)
    print("[save] %s（input_dim=%d cell_in=%d hidden=%d backbone=%d n_heads=%d）"
          % (npz_path, policy["input_dim"], policy["cell_in_dim"],
             policy["hidden_dim"], policy["backbone_units"], policy["n_heads"]))

    summary = {"epochs": args.epochs, "elapsed_s": elapsed, "lambda_start": lam0,
               "lambda_end": lam_end, "final_train": history[-1]["train"],
               "final_val": history[-1]["val"], "params": n_par,
               "clip_ratio_train": ratio_tr, "clip_ratio_val": ratio_va,
               "device": str(device), "train_sessions": [s["file"] for s in train_sess],
               "val_sessions": [s["file"] for s in val_sess]}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print("PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="训练 CfC 策略（单 cell 64 单元 + 保留门 + 5 个头），输出 checkpoint.pt / policy.npz。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE / "data"), help="synth.py 产出的 npz 目录")
    ap.add_argument("--out-dir", default=str(HERE), help="checkpoint.pt / policy.npz 输出目录")
    ap.add_argument("--epochs", type=int, default=6, help="训练轮数")
    ap.add_argument("--batch-size", type=int, default=8, help="批大小（窗口数）")
    ap.add_argument("--seq-len", type=int, default=64, help="每个窗口的帧数")
    ap.add_argument("--stride", type=int, default=16, help="窗口起点步长")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW 学习率")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪（0=关）")
    ap.add_argument("--val-fraction", type=float, default=0.25, help="按会话切出的验证比例")
    ap.add_argument("--cell-in", type=int, default=64, help="投影层输出维度")
    ap.add_argument("--hidden", type=int, default=64, help="CfC 隐藏单元数")
    ap.add_argument("--backbone-units", type=int, default=64,
                    help="ncps backbone 宽度（必须显式给，默认 128 会翻倍算力）")
    ap.add_argument("--ret-init", type=float, default=-3.0,
                    help="保留门 ret 初值（-6 落在 sigmoid 死区）")
    ap.add_argument("--clip-sigma", type=float, default=4.0, help="域外守护倍数")
    ap.add_argument("--std-floor", type=float, default=1e-2, help="input_std 下限")
    ap.add_argument("--lambda-smooth", type=float, default=0.05, help="动作一阶差分平滑权重")
    ap.add_argument("--lambda-retention", type=float, default=0.01, help="保留门正则权重")
    ap.add_argument("--lambda-clip", type=float, default=0.10,
                    help="防饱和权重（tanh 头下恒为 0，留作保险）")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--quick", action="store_true",
                    help="验收用最小训练（epochs=2, seq_len=32, batch=8）")
    args_raw = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(args_raw)

    if args.quick:
        if not any(a == "--epochs" or a.startswith("--epochs=") for a in args_raw):
            args.epochs = 2
        args.seq_len = min(args.seq_len, 32)
        args.batch_size = min(args.batch_size, 8)
    return train(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
