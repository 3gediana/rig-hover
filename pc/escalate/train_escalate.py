# -*- coding: utf-8 -*-
"""train_escalate.py — 给 CfC 加第 6 个头 ``escalate``（多任务联合训练）。

结构（与 ``pc/train/train_cfc.py`` 同一份 blob 布局，只多一个头）::

    obs(106) -> 归一化 -> Linear(106->64) -> SiLU
             -> CfCCell(64/64/64) -> 保留门
             -> 6 头: head_yaw/vyaw/mode/emote/vx **+ escalate(sigmoid)**

损失::

    L = w_esc · BCE(escalate_logit, y)                       # 新头，标签见 esc_scenes
      + w_ctrl · [ MSE(3 个连续头) + CE(mode) + CE(emote) ]   # 只在 arm A 帧上算
      + w_ret · mean(sigmoid(ret))                            # 保留门正则（沿用原配方）

``y`` 是混合标签：arm A 帧用蒸馏软标签（可观测判据的 ``p_esc``），arm B 帧用
时序谓词给的硬标签。掩码 ``ctrl_ok`` 保证改过观测的时序帧不参与控制头损失。

关键超参（都是"为什么这样"的产物，不是调参调出来的）：

* ``--lr-trunk 5e-5``：主干只做微调。它已经在教师数据上收敛到 mode 99%，这里
  要的是**让主干把时序证据留在 h 里**，不是重学控制；
* ``--lr-head 3e-3``：新头从随机初始化开始，慢 60 倍没有意义；
* ``--pos-weight``：硬标签里阳性 25%，不加权时 BCE 会把概率压在基率附近，
  召回率一塌糊涂。上限 4（再高会明显破坏校准，见 test 的 ECE）；
* ``--calibrate temp``：默认在**验证集**上拟合一维温度 ``T``，然后把 ``T``
  吸收进升级头权重（``W/T``、``b/T`` 是同一条线性层，固件零改动）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import esc_core as C
import esc_model as M
import esc_scenes as S

F32 = np.float32
HERE = Path(__file__).resolve().parent

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: 与 ``train_cfc.py`` 相同的 mode 类权重（ACK×20 / REFRACTORY×15）
MODE_WEIGHT = (1.0, 1.0, 20.0, 15.0)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
class WindowDS(torch.utils.data.Dataset):
    """按 (会话, 起点) 切窗口；窗口内同时给出 escalate 标签与控制标签。"""

    def __init__(self, d, idx_sessions, seq_len=96, stride=48):
        self.d = d
        self.items = []
        t = d["obs"].shape[1]
        for si in idx_sessions:
            for st in range(0, t - seq_len + 1, max(1, stride)):
                self.items.append((int(si), st))
        if not self.items:
            raise ValueError("切不出窗口：seq_len=%d > 会话长度 %d" % (seq_len, t))
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, st = self.items[i]
        sl = slice(st, st + self.seq_len)
        d = self.d
        return {
            "obs_n": torch.from_numpy(d["obs_n"][si][sl]),
            "y": torch.from_numpy(d["y"][si][sl]),
            "teacher": torch.from_numpy(d["teacher"][si][sl]),
            "hard": torch.from_numpy(d["hard"][si][sl]),
            "w": torch.from_numpy(d["weight"][si][sl]),
            "ctrl": torch.from_numpy(d["ctrl"][si][sl]),
            "mode": torch.from_numpy(d["mode"][si][sl]),
            "emote": torch.from_numpy(d["emote"][si][sl]),
            "ok": torch.from_numpy(d["ctrl_ok"][si][sl]),
            "si": si, "st": st,
        }


def normalize(d, mean, std):
    """把 obs 归一化（与固件域外守护同口径），原地写进 ``obs_n``。"""
    o = d["obs"].astype(np.float64)
    mu = mean.astype(np.float64)
    sd = std.astype(np.float64)
    bad = ~np.isfinite(o)
    v = np.where(bad, mu, o)
    v = np.clip(v, mu - 4.0 * sd, mu + 4.0 * sd)
    d["obs_n"] = ((v - mu) / sd).astype(F32)
    return d


# --------------------------------------------------------------------------- #
# 模型侧一步
# --------------------------------------------------------------------------- #
def forward_window(model, obs, device):
    x = obs.to(device=device, dtype=torch.float32)
    return model.forward_seq(x)


def compute_loss(model, batch, out, args):
    y = batch["y"].to(out.device)
    logit = out[..., M.COL_ESCALATE]
    pw = torch.tensor(float(args.pos_weight), device=out.device)
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)

    # 蒸馏锚（只在 arm A 帧上）：把升级头的 logit 拉向逐帧判据的 logit。
    # 没有它，时序硬标签（0/1）会把低概率区整体抬起来 —— 排序仍然对，
    # 但 p 的**数值**不再可信（ECE 0.32、Spearman 0.36 就是这么来的）。
    kd_m = (batch["hard"].to(out.device) < 0.0).to(out.dtype)
    t_p = torch.clamp(batch["teacher"].to(out.device), 1e-6, 1.0 - 1e-6)
    logit_t = torch.log(t_p) - torch.log1p(-t_p)
    kd = (((logit - logit_t) ** 2) * kd_m).sum() / kd_m.sum().clamp(min=1.0)

    ok = batch["ok"].to(out.device)
    ctrl = batch["ctrl"].to(out.device)
    mode = batch["mode"].to(out.device)
    emote = batch["emote"].to(out.device)
    pred = torch.stack([torch.tanh(out[..., M.COL["head_yaw"]]),
                        torch.tanh(out[..., M.COL["vyaw"]]),
                        torch.tanh(out[..., M.COL["vx"]])], dim=-1)
    mse = (((pred - ctrl) ** 2) * ok[..., None]).sum() / ok.sum().clamp(min=1.0) / 3.0
    w = torch.tensor(MODE_WEIGHT, device=out.device, dtype=out.dtype)
    ce_mode = F.cross_entropy(out[..., M.COL["mode"]:M.COL["mode"] + 4]
                              .reshape(-1, 4), mode.reshape(-1), weight=w,
                              reduction="none").reshape(mode.shape)
    ce_mode = (ce_mode * ok).sum() / ok.sum().clamp(min=1.0)
    ce_emote = F.cross_entropy(out[..., M.COL["emote"]:M.COL["emote"] + 8]
                               .reshape(-1, 8), emote.reshape(-1), reduction="none")
    ce_emote = (ce_emote.reshape(emote.shape) * ok).sum() / ok.sum().clamp(min=1.0)

    lam = torch.sigmoid(model.ret).mean()
    loss = (args.w_esc * bce
            + args.kd * kd
            + args.w_ctrl * (mse + ce_mode + ce_emote)
            + args.w_ret * lam)
    parts = {"bce": float(bce.item()), "kd": float(kd.item()),
             "mse": float(mse.item()),
             "ce_mode": float(ce_mode.item()), "ce_emote": float(ce_emote.item()),
             "lam": float(lam.item()), "total": float(loss.item())}
    return loss, parts


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def predict(model, d, sessions, seq_len, batch_size, device, stride=48):
    """在给定会话上滑窗预测，把重叠窗口的预测取平均（会话级完整覆盖）。

    Returns:
        ``pred (N,T)``、``logit (N,T)``（都是 float32）
    """
    model.eval()
    t_total = d["obs"].shape[1]
    acc = np.zeros((len(sessions), t_total), dtype=np.float64)
    cnt = np.zeros((len(sessions), t_total), dtype=np.float64)
    lacc = np.zeros((len(sessions), t_total), dtype=np.float64)
    pos = {int(s): i for i, s in enumerate(sessions)}
    with torch.no_grad():
        for st in range(0, t_total - seq_len + 1, stride):
            xs, sis = [], []
            for s in sessions:
                xs.append(d["obs_n"][s][st:st + seq_len])
                sis.append(int(s))
            x = torch.from_numpy(np.stack(xs)).to(device=device, dtype=torch.float32)
            out = model.forward_seq(x).cpu().numpy()
            for k, s in enumerate(sis):
                acc[pos[s], st:st + seq_len] += 1.0 / (1.0 + np.exp(-out[k, :, M.COL_ESCALATE]))
                lacc[pos[s], st:st + seq_len] += out[k, :, M.COL_ESCALATE]
                cnt[pos[s], st:st + seq_len] += 1.0
    cnt[cnt == 0] = 1.0
    model.train()
    return (acc / cnt).astype(F32), (lacc / cnt).astype(F32)


def evaluate(model, d, split, seq_len, batch_size, device):
    """在 ``split`` 上算三组指标：蒸馏保真、时序可分、控制头有没有退化。"""
    sessions = np.where(d["split"] == split)[0]
    pred, logit = predict(model, d, sessions, seq_len, batch_size, device)
    y = d["y"][sessions]
    teacher = d["teacher"][sessions]
    hard = d["hard"][sessions]
    arm = d["arm"][sessions]
    fam = d["fam"][sessions]
    ordered = d["ordered"][sessions]

    base_m = arm == "base"
    res = {"n_sessions": int(len(sessions))}
    if base_m.any():
        yb = teacher[base_m].reshape(-1)
        pb = pred[base_m].reshape(-1)
        _r, ece, _m = C.reliability(yb, pb)
        res["base"] = {"n": int(yb.size), "auc_vs_teacher": C.roc_auc(yb > 0.5, pb),
                       "ece": ece, "brier": C.brier(yb, pb),
                       "spearman_vs_teacher": C.spearman(pb, yb)}
        # 控制头退化检查：mode 头在 base 帧上的命中率（用教师标签）
        res["base"]["mode_acc"] = _mode_acc(model, d, sessions[base_m], seq_len, device)
    tmp_m = arm == "temporal"
    if tmp_m.any():
        gt = hard[tmp_m].reshape(-1)
        pt = pred[tmp_m].reshape(-1)
        res["temporal"] = {
            "n": int(gt.size), "gt_rate": float(gt.mean()),
            "auc_pooled": C.roc_auc(gt, pt),
            "auc_teacher_pooled": C.roc_auc(gt, teacher[tmp_m].reshape(-1)),
            "per_family": {},
        }
        # 匹配对照（各族分开算）：阳性 = ordered 的 GT=1；阴性 = control 的全部帧。
        # 关键：control 与 ordered 的逐帧字段多集相同，所以逐帧模型在这里恒 0.5。
        allp, alln = [], []
        for f in sorted(set(fam[tmp_m].tolist())):
            mm = tmp_m & (fam == f)
            ctl = mm & (~ordered)
            ordm = mm & ordered
            if not (ctl.any() and ordm.any()):
                continue
            gto = hard[ordm].reshape(-1)
            po = pred[ordm].reshape(-1)
            pos = po[gto > 0.5]
            neg = pred[ctl].reshape(-1)
            if not (pos.size and neg.size):
                continue
            yc = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            auc = C.roc_auc(yc, np.concatenate([pos, neg]))
            tpos = teacher[ordm].reshape(-1)[gto > 0.5]
            tneg = teacher[ctl].reshape(-1)
            auc_t = C.roc_auc(np.concatenate([np.ones(tpos.size), np.zeros(tneg.size)]),
                              np.concatenate([tpos, tneg]))
            res["temporal"]["per_family"][f] = {
                "auc_matched": auc, "auc_matched_teacher": auc_t,
                "n_pos": int(pos.size), "n_neg": int(neg.size)}
            allp.append(pos)
            alln.append(neg)
        if allp:
            pos = np.concatenate(allp)
            neg = np.concatenate(alln)
            res["temporal"]["auc_matched"] = C.roc_auc(
                np.concatenate([np.ones(pos.size), np.zeros(neg.size)]),
                np.concatenate([pos, neg]))
            res["temporal"]["auc_matched_macro"] = float(np.mean(
                [v["auc_matched"] for v in res["temporal"]["per_family"].values()]))
    return res, pred


def _mode_acc(model, d, sessions, seq_len, device):
    """mode 头在给定会话上的命中率（教师标签）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for s in sessions:
            s = int(s)
            for st in range(0, d["obs_n"].shape[1] - seq_len + 1, seq_len):
                x = torch.from_numpy(d["obs_n"][s][st:st + seq_len][None]).to(
                    device=device, dtype=torch.float32)
                out = model.forward_seq(x).cpu().numpy()[0]
                pred = out[:, M.COL["mode"]:M.COL["mode"] + 4].argmax(-1)
                correct += int((pred == d["mode"][s][st:st + seq_len]).sum())
                total += seq_len
    model.train()
    return correct / max(total, 1)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    t0 = time.perf_counter()

    if not S.DATASET.exists():
        print("[data] 数据集不存在，先跑 esc_scenes.py")
        S.build(device=str(device), seed=0)
    d = S.load()
    _, mean, std, _ = M.load_base_policy()
    normalize(d, mean, std)
    train_s = np.where(d["split"] == "train")[0]
    val_s = np.where(d["split"] == "val")[0]
    print("[data] 会话 %s；帧 %d；标签均值 %.4f（train）/ %.4f（val）"
          % ({s: int((d["split"] == s).sum()) for s in ("train", "val", "test")},
             d["obs"].shape[0] * d["obs"].shape[1],
             float(d["y"][train_s].mean()), float(d["y"][val_s].mean())))

    ds = WindowDS(d, train_s, args.seq_len, args.stride)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         drop_last=len(ds) > args.batch_size)
    print("[data] 训练窗口 %d 个（seq_len=%d stride=%d batch=%d）"
          % (len(ds), args.seq_len, args.stride, args.batch_size))

    model = M.EscalatePolicy(input_dim=d["obs"].shape[2], cell_in=args.cell_in,
                             hidden=args.hidden, backbone_units=args.backbone_units)
    info = M.load_base_into(model)
    if args.ret_split > 0.0:
        # 双峰 λ 初值：一半单元保持 base 的快（控制用），一半先给成慢单元
        # （升级头需要的"历史"）。ret 是 64 个 float，blob 里本来就有这一层。
        k = int(round(args.ret_split * model.hidden))
        with torch.no_grad():
            model.ret[model.hidden - k:] = float(args.ret_slow_value)
        print("[model] 保留门双峰初值：%d 个慢单元 ret=%.2f（λ=%.3f），其余保持 %.4f"
              % (k, args.ret_slow_value, 1.0 / (1.0 + np.exp(-args.ret_slow_value)),
                 float(torch.sigmoid(model.ret[:model.hidden - k]).mean())))
    model.to(device)
    print("[model] 6 头 %s；参数量 %d（base 加载成功，缺的键 = %s）"
          % (M.HEAD_DIMS6, sum(p.numel() for p in model.parameters()), info["missing"]))
    print("[model] 保留门 lambda = %.4f" % model.lambda_mean())

    head_params = list(model.heads[5].parameters())
    head_ids = {id(p) for p in head_params}
    ret_ids = {id(model.ret)}
    trunk = [p for p in model.parameters() if id(p) not in head_ids and id(p) not in ret_ids]
    opt = torch.optim.AdamW([
        {"params": trunk, "lr": args.lr_trunk},
        {"params": head_params, "lr": args.lr_head},
        # 保留门 λ=sigmoid(ret) 是"时序记忆"的唯一闸门：初值 -3.0（λ=0.047）意味着
        # 隐状态几乎不含历史（见 README §5 的诊断）。它必须能被升级损失推动，
        # 所以单独给一个较大的学习率 —— 这不改 blob，ret 一直是那 64 个 float。
        {"params": [model.ret], "lr": args.lr_ret},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    # 记录"训练前"的控制头水平，用来量化联合训练有没有把控制学坏
    base_val = evaluate(model, d, "val", args.seq_len, args.batch_size, device)[0]
    print("[before] val base mode_acc=%s / temporal auc=%s"
          % (round(base_val.get("base", {}).get("mode_acc", float("nan")), 4),
             round(base_val.get("temporal", {}).get("auc_pooled", float("nan")), 4)))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = {}, 0
        te = time.perf_counter()
        for batch in loader:
            out = forward_window(model, batch["obs_n"], device)
            loss, parts = compute_loss(model, batch, out, args)
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
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ev, _ = evaluate(model, d, "val", args.seq_len, args.batch_size, device)
        else:
            ev = {}
        history.append({"epoch": epoch, "train": tr, "val": ev,
                        "sec": time.perf_counter() - te})
        bt = ev.get("base", {})
        tp = ev.get("temporal", {})
        lam = torch.sigmoid(model.ret)
        fam_str = " ".join("%s=%.3f" % (k[:5], v["auc_matched"])
                           for k, v in tp.get("per_family", {}).items())
        print("[epoch %2d] bce=%.4f ctrl=%.4f | val auc_base=%.4f auc_tmp=%.4f "
              "match=%.4f mode_acc=%.4f [%s] | lam %.3f/max %.3f | %.1fs"
              % (epoch, tr["bce"], tr["mse"] + tr["ce_mode"] + tr["ce_emote"],
                 bt.get("auc_vs_teacher", float("nan")),
                 tp.get("auc_pooled", float("nan")),
                 tp.get("auc_matched", float("nan")), bt.get("mode_acc", float("nan")), fam_str,
                 float(lam.mean().item()), float(lam.max().item()), history[-1]["sec"]))

    elapsed = time.perf_counter() - t0
    final_val, _pred = evaluate(model, d, "val", args.seq_len, args.batch_size, device)

    # ---- 温度标定（只动升级头，且可吸收进权重）----
    calib = {"T": 1.0, "bias": 0.0}
    sessions = np.where(d["split"] == "val")[0]
    _pred_all, logit = predict(model, d, sessions, args.seq_len,
                               args.batch_size, device)
    base_m = d["arm"][sessions] == "base"
    if args.calibrate in ("temp", "temp+bias"):
        T, nll = C.fit_temperature(d["y"][sessions].reshape(-1), logit.reshape(-1))
        with torch.no_grad():
            model.heads[5].weight.div_(T)
            model.heads[5].bias.div_(T)
        logit = logit / T
        calib["T"] = float(T)
        calib["val_nll_after_T"] = float(nll)
        print("[calib] 温度 T=%.3f（W/T、b/T 吸收进升级头）" % T)
    if args.calibrate in ("bias", "temp+bias"):
        # 让 val 的 base 帧上"平均预测"等于"平均逐帧判据"。这是一个标量，
        # 代表**混合比例带来的基率偏移**：部署时时序事件的实际发生率一变，
        # 只要重拟这一个 float（写进 blob 的 head_b[5]），不用重训。
        target = float(d["teacher"][sessions[base_m]].mean())
        lo, hi = -12.0, 12.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            p = 1.0 / (1.0 + np.exp(-np.clip(logit[base_m].reshape(-1) + mid, -30, 30)))
            if float(p.mean()) < target:
                lo = mid
            else:
                hi = mid
        b_shift = 0.5 * (lo + hi)
        with torch.no_grad():
            model.heads[5].bias.add_(float(b_shift))
        calib["bias"] = float(b_shift)
        calib["target_mean_base"] = target
        print("[calib] 基率偏置 %+0.3f（吸收进 head_b[5]），目标均值 %.4f"
              % (b_shift, target))
    if args.calibrate != "none":
        after, _ = evaluate(model, d, "val", args.seq_len, args.batch_size, device)
        calib["val_ece_before"] = float(final_val.get("base", {}).get("ece", float("nan")))
        calib["val_ece_after"] = float(after.get("base", {}).get("ece", float("nan")))
        calib["val_auc_after"] = float(after.get("base", {}).get("auc_vs_teacher", float("nan")))
        final_val = after
        print("[calib] val ECE %.4f -> %.4f（AUC 保持 %.4f）"
              % (calib["val_ece_before"], calib["val_ece_after"],
                 calib.get("val_auc_after", float("nan"))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint_esc.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {"input_dim": int(d["obs"].shape[2]), "cell_in": args.cell_in,
                   "hidden": args.hidden, "backbone_units": args.backbone_units,
                   "ret_init": -3.0, "head_dims": list(M.HEAD_DIMS6),
                   "clip_sigma": 4.0, "std_floor": 1e-2},
        "input_mean": mean, "input_std": std,
        "history": history, "epochs": args.epochs, "elapsed_s": elapsed,
        "calibration": calib, "val": final_val,
        "args": {k: v for k, v in vars(args).items()},
    }, ckpt)
    print("[save] %s（%.1f s）" % (ckpt, elapsed))

    summary = {"epochs": args.epochs, "elapsed_s": elapsed, "params": int(
        sum(p.numel() for p in model.parameters())), "calibration": calib,
        "val": final_val, "lambda_end": model.lambda_mean(),
        "train_sessions": int(len(train_s)), "val_sessions": int(len(val_s)),
        "device": str(device)}
    (out_dir / "train_summary_esc.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PASS")
    return 0


def recalibrate(ckpt_path, data_path=None, device="cpu", verbose=True):
    """对已有 checkpoint 做事后标定（温度 + 基率偏置），就地写回。

    为什么要单独这一步：训练时把时序硬标签混进来会改变头的基率
    （同一个"看起来良性"的帧，在训练集里有一半概率属于某个时序升级片段），
    于是概率整体偏高。**形状（AUC/排序）不受影响，改的只是尺度与位置**，
    而这两者都能精确写进升级头本身：``W' = W/T``、``b' = b/T + shift``。
    固件侧零改动。

    标定在**验证集**上拟合（对着训练标签 ``y``），测试集不参与。这也顺便给出
    "基率随部署分布漂移时怎么办"的答案：重拟这一个 float，不用重训。

    Returns:
        dict：``{"T", "bias", "before", "after"}``
    """
    import torch
    d = S.load() if data_path is None else S.load(path=data_path)
    _, mean, std, _ = M.load_base_policy()
    normalize(d, mean, std)
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = M.EscalatePolicy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                             hidden=cfg["hidden"], backbone_units=cfg["backbone_units"])
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    vs = np.where(d["split"] == "val")[0]
    ps, logit = predict(model, d, vs, 96, 32, device)
    y = d["y"][vs].reshape(-1)
    base = d["arm"][vs] == "base"
    tea = d["teacher"][vs][base].reshape(-1)
    bf = np.repeat(base, ps.shape[1])

    def _m(p):
        return {"ece_y": C.reliability(y, p)[1],
                "ece_teacher_base": C.reliability(tea, p[bf])[1],
                "auc_y": C.roc_auc(y > 0.5, p),
                "auc_teacher_base": C.roc_auc((tea > 0.5).astype(float), p[bf])}

    before = _m(ps.reshape(-1))
    T, _nll = C.fit_temperature(y, logit.reshape(-1))
    z = logit.reshape(-1) / T
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = 1.0 / (1.0 + np.exp(-np.clip(z + mid, -30, 30)))
        if float(q.mean()) < float(y.mean()):
            lo = mid
        else:
            hi = mid
    shift = 0.5 * (lo + hi)
    with torch.no_grad():
        model.heads[5].weight.div_(T)
        model.heads[5].bias.div_(T)
        model.heads[5].bias.add_(float(shift))

    ps2, _ = predict(model, d, vs, 96, 32, device)
    after = _m(ps2.reshape(-1))
    ck["model"] = model.state_dict()
    ck["calibration"] = {"T": float(T), "bias": float(shift), "fit_on": "val",
                         "target": "y", "before": before, "after": after}
    torch.save(ck, str(ckpt_path))
    if verbose:
        print("[recalib] T=%.3f  bias=%+.3f（拟合在 val，对着训练标签 y）" % (T, shift))
        print("    ECE(y)       %.4f -> %.4f" % (before["ece_y"], after["ece_y"]))
        print("    ECE(判据|A)  %.4f -> %.4f" % (before["ece_teacher_base"],
                                                 after["ece_teacher_base"]))
        print("    AUC(y)       %.4f -> %.4f（形状不变）"
              % (before["auc_y"], after["auc_y"]))
    return {"T": float(T), "bias": float(shift), "before": before, "after": after}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="训练 CfC 的第 6 个头 escalate（多任务联合优化）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default=str(S.DATASET))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=96, help="窗口帧数（含 2 个 episode）")
    ap.add_argument("--stride", type=int, default=48, help="窗口步长（= 1 个 episode）")
    ap.add_argument("--cell-in", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--backbone-units", type=int, default=64)
    ap.add_argument("--lr-trunk", type=float, default=3e-4, help="主干微调学习率")
    ap.add_argument("--lr-head", type=float, default=3e-3, help="升级头学习率")
    ap.add_argument("--lr-ret", type=float, default=2e-2,
                    help="保留门 ret 的学习率（λ 是隐状态记忆的唯一闸门，必须能动）")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--kd", type=float, default=0.3,
                    help="蒸馏锚权重：在 arm A 帧上把 logit 拉向逐帧判据的 logit")
    ap.add_argument("--ret-split", type=float, default=0.5,
                    help="保留门双峰初值的比例（0=关）")
    ap.add_argument("--ret-slow-value", type=float, default=1.5,
                    help="慢单元的 ret 初值（1.5 -> λ=0.82）")
    ap.add_argument("--w-esc", type=float, default=1.0)
    ap.add_argument("--w-ctrl", type=float, default=3.0, help="控制头损失权重（防遗忘）")
    ap.add_argument("--w-ret", type=float, default=0.01)
    ap.add_argument("--calibrate", choices=("none", "temp", "bias", "temp+bias"), default="temp+bias")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recalibrate", default="",
                    help="只做事后标定：给一个已有 checkpoint 路径（就地在 val 上拟合 T/bias）")
    args = ap.parse_args(argv)
    if args.recalibrate:
        recalibrate(args.recalibrate, data_path=args.data,
                    device=("cpu" if args.device == "auto" else args.device))
        return 0
    if args.data != str(S.DATASET):
        S.DATASET = Path(args.data)
    return train(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
