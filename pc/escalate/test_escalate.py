# -*- coding: utf-8 -*-
"""test_escalate.py — 一条命令验收"第 6 个头 escalate"。

::

    python test_escalate.py

覆盖 7 节：

1. **判据来源对拍**：``esc_core`` 复刻的 12 个势函数 vs ``decision_hub/hub.py``
   真实现，判据 max|Δ| ≤ 1e-9（不可用时 SKIP，不算失败）；
2. **测试集 AUC / 校准 / 一致率**：蒸馏保真度（对可观测判据的 ``p_esc``）、
   可靠性曲线（ECE/MCE/Brier）、按概率分桶的一致率、按划分（train/val/test）
   与按场景族的一致性；
3. **时序优势**：4 个场景族的 ``ordered`` / ``control`` 孪生对照（逐帧字段
   多重集相同），比较 CfC 与三条基线（B0 逐帧判据 / B1 判据+EMA /
   B2 手工时序特征+逻辑回归）的匹配对照 AUC；
4. **固定误报率下的召回**：阈值在 val 上定，报 test 上的 FPR/召回；
5. **导出对拍**：``checkpoint → blob → 纯 numpy 前向`` 与 torch 逐帧比较，
   判据 max_abs_err < 1e-5；并核对 CRC / 头字段 / 与 ``train/export_blob.py``
   逐字节一致；
6. **控制头没被学坏**：mode 头在 base 帧上的准确率（基线 0.9909）；
7. **隐状态记忆诊断**：λ 分布 + "同一帧、不同历史"下的 h 与 p_esc 差异
   —— 这一节负责把"最脆弱的一点"量化出来。

每条判据都有阈值，最后打印 PASS/FAIL 汇总；退出码 0 表示全过。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import esc_baselines as B
import esc_core as C
import esc_model as M
import esc_scenes as S
import export_escalate as X

F32 = np.float32
HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoint_esc.pt"
NPZ = HERE / "policy_esc.npz"
BLOB = HERE / "model_slot0_esc.blob"
METRICS = HERE / "artifacts" / "metrics_test.json"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# --------------------------------------------------------------------------- #
# 阈值（全部写死在这里，方便一眼看出"什么算过"）
# --------------------------------------------------------------------------- #
TH = {
    "phi_parity": 1e-9,          # 与 decision_hub/hub.py 的对拍误差
    "label_auc": 0.78,           # 混合标签（蒸馏软标签 + 时序硬标签）上的 AUC
    "fidelity_auc": 0.95,        # 对逐帧判据的排序保真（arm A 帧）
    "base_ece": 0.06,            # 校准误差（对训练标签，10 桶）
    "agree_label": 0.80,         # 与训练标签在 0.5 阈值上的判决一致率
    "temporal_matched_auc": 0.70,  # CfC 在孪生对照上的匹配 AUC（族宏平均）
    "temporal_margin": 0.10,     # CfC 相对最强基线（B0/B1/B2）的 AUC 优势
    "recall_at_fpr1": 0.30,      # 1% 误报率（分母=良性帧）下的时序召回
    "export_err": 1e-5,          # torch <-> numpy(blob) 的最大绝对误差
    "mode_acc": 0.95,            # 控制头没被学坏
}


class Checker:
    """收集 PASS/FAIL/SKIP/WARN，最后汇总。

    ``soft=True`` 的判据是**质量目标**（"应该做到多少"），没达到记 ``WARN``，
    不影响退出码；硬判据（对拍、导出、校准、控制头退化）一律 ``FAIL``。
    这样"一条命令跑通"的含义是：**正确性全过，质量目标如实打印**。
    """

    def __init__(self):
        self.items = []

    def ck(self, name, cond, detail="", skip=False, soft=False):
        status = "SKIP" if skip else ("PASS" if cond else ("WARN" if soft else "FAIL"))
        self.items.append({"name": name, "status": status, "detail": str(detail)})
        print("  [%s] %s%s" % (status, name, ("  " + str(detail)) if detail else ""))
        return status != "FAIL"

    def section(self, title):
        print("\n== %s ==" % title)

    @property
    def ok(self):
        return all(i["status"] != "FAIL" for i in self.items)

    def summary(self):
        n = {"PASS": 0, "FAIL": 0, "SKIP": 0, "WARN": 0}
        for i in self.items:
            n[i["status"]] += 1
        return n


def ensure_artifacts(ck):
    """构件不全就现场生成（保证"一条命令"从零也能跑）。"""
    ck.section("0. 构件")
    if not S.DATASET.exists():
        print("[build] 数据集缺失，现场生成（约 10 s）…")
        S.build(device="cpu")
    ck.ck("数据集存在", S.DATASET.exists(), S.DATASET.name)
    if not CKPT.exists():
        print("[build] checkpoint 缺失，现场训练 60 epoch…")
        import train_escalate as T
        T.main(["--epochs", "60", "--eval-every", "30", "--calibrate", "none"])
        T.recalibrate(CKPT)
    ck.ck("checkpoint 存在", CKPT.exists(), CKPT.name)


# --------------------------------------------------------------------------- #
# 载入
# --------------------------------------------------------------------------- #
def load_all():
    import torch

    import train_escalate as T

    d = S.load()
    _, mean, std, _ = M.load_base_policy()
    T.normalize(d, mean, std)
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = M.EscalatePolicy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                             hidden=cfg["hidden"], backbone_units=cfg["backbone_units"],
                             ret_init=cfg.get("ret_init", -3.0))
    model.load_state_dict(ck["model"])
    model.eval()
    return d, model, ck, T


def pred_split(model, d, T, split, seq_len=96, stride=48, device="cpu"):
    sessions = np.where(d["split"] == split)[0]
    pred, logit = T.predict(model, d, sessions, seq_len, 32, device)
    return sessions, pred, logit


# --------------------------------------------------------------------------- #
# 1. 判据来源对拍
# --------------------------------------------------------------------------- #
def sec1_phi(ck):
    ck.section("1. 判据来源对拍（esc_core vs decision_hub/hub.py）")
    ok, err, detail = C.phi_parity_check(400)
    if ok is None:
        return ck.ck("12 个势函数与 hub.py 一致", True, detail, skip=True)
    return ck.ck("12 个势函数与 hub.py 一致", err <= TH["phi_parity"], detail)


# --------------------------------------------------------------------------- #
# 2. 基率分布：AUC / 校准 / 一致率
# --------------------------------------------------------------------------- #
def sec2_base(ck, d, model, T, metrics):
    ck.section("2. 基率分布：AUC / 校准 / 一致率（测试集）")
    sessions, pred, logit = pred_split(model, d, T, "test")
    arm = d["arm"][sessions]
    base_m = arm == "base"
    sessions_b = sessions[base_m]

    p_all = pred.reshape(-1)
    y_all = d["y"][sessions].reshape(-1)
    p_cfc = pred[base_m].reshape(-1)
    teacher = d["teacher"][sessions_b].reshape(-1)
    gt_bin = (teacher > 0.5).astype(np.float64)

    # ---- A. 校准（对着"训练标签"，也就是这个头真正要预测的东西）----
    print("  A. 校准（GT = 训练标签 y：arm A 是蒸馏软标签，arm B 是时序硬标签）")
    rows, ece, mce = C.reliability(y_all, p_all)
    res = C.summarize_scores("cfc_head", y_all, p_all)
    print("      桶  概率区间         n     平均预测   实际频率   |gap|")
    gapmax = max(max(r["gap"] for r in rows), 1e-9)
    for r in rows:
        if r["n"] == 0:
            continue
        bar = "#" * int(round(28 * r["gap"] / gapmax))
        print("      %2d  [%.2f,%.2f)  %6d    %7.4f   %7.4f   %6.4f %s"
              % (r["bin"], r["lo"], r["hi"], r["n"], r["mean_p"], r["freq"],
                 r["gap"], bar))
    ck.ck("混合标签 AUC ≥ %.2f" % TH["label_auc"], res["auc"] >= TH["label_auc"],
          "AUC=%.4f（n=%d，阳性 %d）" % (res["auc"], res["n"], res["pos"]))
    ck.ck("校准误差 ECE ≤ %.2f" % TH["base_ece"], ece <= TH["base_ece"],
          "ECE=%.4f MCE=%.4f Brier=%.4f" % (ece, mce, res["brier"]))
    agree_y, _ = C.bucket_agreement(p_all, y_all, thr=0.5, bins=10)
    ck.ck("与训练标签的判决一致率 ≥ %.2f" % TH["agree_label"],
          agree_y >= TH["agree_label"], "一致率=%.4f（阈值 0.5，n=%d）"
          % (agree_y, p_all.size))

    # ---- B. 蒸馏保真度（对着逐帧判据，只看 arm A 帧）----
    print("  B. 蒸馏保真度（arm A 帧，GT = 逐帧判据 p_esc > 0.5）")
    auc_tea = C.roc_auc(gt_bin, p_cfc)
    rho = C.spearman(p_cfc, teacher)
    mae = float(np.mean(np.abs(p_cfc - teacher)))
    agree_t, bucket_rows = C.bucket_agreement(p_cfc, teacher, thr=0.5, bins=10)
    print("      AUC=%.4f  Spearman=%.4f  MAE=%.4f  判决一致率=%.4f"
          % (auc_tea, rho, mae, agree_t))
    print("      分桶一致率（按 CfC 概率分桶）：")
    for r in bucket_rows:
        if r["n"] == 0:
            continue
        print("        %2d  [%.2f,%.2f)  n=%6d  agree=%.4f  mean_cfc=%.4f "
              "mean_logit=%.4f"
              % (r["bin"], r["lo"], r["hi"], r["n"], r["agree"],
                 r["mean_a"], r["mean_b"]))
    ck.ck("对逐帧判据的排序保真 AUC ≥ %.3f" % TH["fidelity_auc"],
          auc_tea >= TH["fidelity_auc"], "AUC=%.4f（n=%d）" % (auc_tea, p_cfc.size))

    # ---- C. 按划分一致性 ----
    val_s = np.where(d["split"] == "val")[0]
    train_s = np.where(d["split"] == "train")[0]
    tau, _ = B.best_ema_tau(d["phi"][val_s], d["y"][val_s])
    p_ema = B.logistic_ema(d["phi"][sessions_b], tau).reshape(-1)
    lin = B.LinearTemporal().fit(B.temporal_features(d["phi"][train_s]),
                                 d["y"][train_s])
    p_lin = lin.predict(B.temporal_features(d["phi"][sessions_b])).reshape(-1)
    rows_cmp = [
        {"name": "B0 logistic_frame（= 被替换的那套）",
         "auc": C.roc_auc(gt_bin, teacher), "ece": C.reliability(gt_bin, teacher)[1]},
        {"name": "B1 logistic_ema(tau=%.0f 帧)" % tau,
         "auc": C.roc_auc(gt_bin, p_ema), "ece": C.reliability(gt_bin, p_ema)[1]},
        {"name": "B2 linear_temporal（48 维手写时序特征）",
         "auc": C.roc_auc(gt_bin, p_lin), "ece": C.reliability(gt_bin, p_lin)[1]},
        {"name": "CfC escalate 头", "auc": auc_tea, "ece": ece},
    ]
    print("  C. 同一批 arm A 测试帧上的横向对比（AUC 都对着逐帧判据算）：")
    for r in rows_cmp:
        print("      %-38s AUC=%.4f  ECE=%.4f" % (r["name"], r["auc"], r["ece"]))

    split_rows = {}
    for split in ("val", "test"):
        ss, pp, _ = pred_split(model, d, T, split)
        am = d["arm"][ss] == "base"
        ppv = pp[am].reshape(-1)
        tv = d["teacher"][ss[am]].reshape(-1)
        split_rows[split] = {"auc": C.roc_auc(tv > 0.5, ppv),
                             "ece": C.reliability((tv > 0.5).astype(float), ppv)[1],
                             "n": int(tv.size)}
    dv = abs(split_rows["val"]["auc"] - split_rows["test"]["auc"])
    print("  D. 按划分一致性：val AUC=%.4f（n=%d）/ test AUC=%.4f（n=%d），差 %.4f"
          % (split_rows["val"]["auc"], split_rows["val"]["n"],
             split_rows["test"]["auc"], split_rows["test"]["n"], dv))
    ck.ck("按划分一致（val/test AUC 差 ≤ 0.05）", dv <= 0.05,
          "val=%.4f test=%.4f diff=%.4f"
          % (split_rows["val"]["auc"], split_rows["test"]["auc"], dv))

    metrics["base"] = {"calibration_vs_label": res, "ece": ece, "mce": mce,
                       "agree_with_label": agree_y,
                       "teacher_fidelity": {"auc": auc_tea, "spearman": rho,
                                            "mae": mae, "agreement": agree_t},
                       "ema_tau": tau, "comparison": rows_cmp,
                       "splits": split_rows, "reliability": rows,
                       "bucket_agreement": bucket_rows}
    return sessions, pred, base_m


# --------------------------------------------------------------------------- #
# 3. 时序优势
# --------------------------------------------------------------------------- #
def sec3_temporal(ck, d, model, T, sessions, pred, metrics):
    ck.section("3. 时序优势：ordered / control 孪生对照（逐帧字段多重集相同）")
    arm = d["arm"][sessions]
    fam = d["fam"][sessions]
    ordered = d["ordered"][sessions]
    hard = d["hard"][sessions]
    teacher = d["teacher"][sessions]

    val_s = np.where(d["split"] == "val")[0]
    train_s = np.where(d["split"] == "train")[0]
    tau, _ = B.best_ema_tau(d["phi"][val_s], d["y"][val_s])
    lin = B.LinearTemporal().fit(B.temporal_features(d["phi"][train_s]),
                                 d["y"][train_s])
    p_ema = B.logistic_ema(d["phi"][sessions], tau)
    p_lin = lin.predict(B.temporal_features(d["phi"][sessions]))

    print("    族（4 族 × ordered/control 孪生，阳性 = ordered 的 GT=1，"
          "阴性 = control 全部帧）：")
    print("      %-15s %6s %6s | %8s %8s %8s %8s" %
          ("族", "#pos", "#neg", "CfC", "B0判据", "B1+EMA", "B2线性时序"))
    per = {}
    for f in sorted(set(fam[arm == "temporal"].tolist())):
        mm = (arm == "temporal") & (fam == f)
        o_m = mm & ordered
        c_m = mm & (~ordered)
        if not (o_m.any() and c_m.any()):
            continue
        gto = hard[o_m].reshape(-1)
        pos_mask = gto > 0.5
        if not pos_mask.any():
            continue

        def matched(p):
            pos = p[o_m].reshape(-1)[pos_mask]
            neg = p[c_m].reshape(-1)
            y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
            return C.roc_auc(y, np.concatenate([pos, neg])), pos.size, neg.size

        a_cfc, n_pos, n_neg = matched(pred)
        a_tea, _, _ = matched(teacher)
        a_ema, _, _ = matched(p_ema)
        a_lin, _, _ = matched(p_lin)
        per[f] = {"auc_cfc": a_cfc, "auc_b0": a_tea, "auc_b1_ema": a_ema,
                  "auc_b2_linear": a_lin, "n_pos": int(n_pos), "n_neg": int(n_neg),
                  "gt_rate": float(gto.mean()),
                  "cfc_mean_ordered": float(pred[o_m].mean()),
                  "cfc_mean_control": float(pred[c_m].mean())}
        print("      %-15s %6d %6d | %8.4f %8.4f %8.4f %8.4f"
              % (f, n_pos, n_neg, a_cfc, a_tea, a_ema, a_lin))

    # 混合（全部族拼起来）匹配 AUC
    tmp = arm == "temporal"
    o_m = tmp & ordered
    c_m = tmp & (~ordered)
    gto = hard[o_m].reshape(-1)
    pos_mask = gto > 0.5

    def matched_all(p):
        pos = p[o_m].reshape(-1)[pos_mask]
        neg = p[c_m].reshape(-1)
        y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
        return C.roc_auc(y, np.concatenate([pos, neg]))

    pooled = {"cfc": matched_all(pred), "b0": matched_all(teacher),
              "b1": matched_all(p_ema), "b2": matched_all(p_lin)}
    # 混合池化 AUC（不做匹配对照，含 posed 与 control 的合并分布）
    gt_t = hard[tmp].reshape(-1)
    pooled_raw = {"cfc": C.roc_auc(gt_t, pred[tmp].reshape(-1)),
                  "b0": C.roc_auc(gt_t, teacher[tmp].reshape(-1)),
                  "b1": C.roc_auc(gt_t, p_ema[tmp].reshape(-1)),
                  "b2": C.roc_auc(gt_t, p_lin[tmp].reshape(-1))}
    macro = float(np.mean([v["auc_cfc"] for v in per.values()])) if per else 0.0
    best_base = max(pooled["b0"], pooled["b1"], pooled["b2"]) if per else 0.0
    print("    混合：CfC=%.4f  B0=%.4f  B1(+EMA)=%.4f  B2(线性时序)=%.4f"
          % (pooled["cfc"], pooled["b0"], pooled["b1"], pooled["b2"]))
    print("    族宏平均：CfC=%.4f；池化(未匹配)：CfC=%.4f B0=%.4f B1=%.4f B2=%.4f"
          % (macro, pooled_raw["cfc"], pooled_raw["b0"], pooled_raw["b1"],
             pooled_raw["b2"]))
    ck.ck("时序匹配对照 AUC（族宏平均）≥ %.2f" % TH["temporal_matched_auc"],
          macro >= TH["temporal_matched_auc"], "macro AUC=%.4f" % macro)
    ck.ck("相对最强基线（B0/B1/B2）的优势 ≥ %.2f" % TH["temporal_margin"],
          (macro - best_base) >= TH["temporal_margin"],
          "macro=%.4f，最强基线=%.4f，差=%+.4f"
          % (macro, best_base, macro - best_base), soft=True)
    metrics["temporal"] = {"per_family": per, "pooled_matched": pooled,
                           "pooled_raw": pooled_raw, "macro_auc_cfc": macro,
                           "ema_tau": tau,
                           "linear_temporal_weights_absmax":
                               float(np.max(np.abs(lin.w)))}
    return per, pooled


# --------------------------------------------------------------------------- #
# 4. 固定误报率下的召回
# --------------------------------------------------------------------------- #
def sec4_fpr(ck, d, model, T, metrics):
    ck.section("4. 固定误报率下的召回（阈值在 val 上定，报 test）")
    # 误报率的**分母只能是"确实良性"的帧**：判据自己 p<0.1 的 base 帧。
    # 把判据说"该升级"的帧算成误报是错的——那不是假警报，是真警报。
    BENIGN = 0.10
    val_s, val_p, _ = pred_split(model, d, T, "val")
    val_base = (d["arm"][val_s] == "base")
    val_teacher = d["teacher"][val_s]
    benign_val = np.repeat(val_base, val_p.shape[1]) & \
        (val_teacher.reshape(-1) < BENIGN)
    val_neg = val_p.reshape(-1)[benign_val]

    te_s, te_p, _ = pred_split(model, d, T, "test")
    te_arm = d["arm"][te_s]
    te_hard = d["hard"][te_s]
    te_base = te_arm == "base"
    te_pf = te_p.reshape(-1)
    te_benign = np.repeat(te_base, te_p.shape[1]) & \
        (d["teacher"][te_s].reshape(-1) < BENIGN)

    out = {"n_benign_val": int(val_neg.size), "n_benign_test": int(te_benign.sum())}
    for fpr in (0.01, 0.05):
        thr = float(np.nextafter(np.quantile(val_neg, 1.0 - fpr), np.inf))
        val_fpr = float((val_neg > thr).mean())
        te_fpr = float((te_pf[te_benign] > thr).mean())
        tmp = te_arm == "temporal"
        gt = te_hard[tmp].reshape(-1)
        p = te_pf[np.repeat(tmp, te_p.shape[1])]
        pos = gt > 0.5
        rec = float((p[pos] > thr).mean()) if pos.any() else 0.0
        out["fpr%g" % fpr] = {"thr": thr, "val_fpr": val_fpr, "test_fpr": te_fpr,
                              "recall_temporal": rec, "n_pos": int(pos.sum())}
        print("    FPR≤%4.1f%%（分母 = %d 个良性 base 帧）：thr=%.4f"
              "（val 实测 %.4f%%，test 实测 %.4f%%）-> 时序帧召回 %.3f（阳性 %d）"
              % (100 * fpr, val_neg.size, thr, 100 * val_fpr, 100 * te_fpr,
                 rec, int(pos.sum())))

    # 对照：逐帧判据（B0）在自己的 1% 工作点上的召回
    val_pf = val_p.reshape(-1)
    thr_b0 = float(np.nextafter(np.quantile(val_pf[benign_val], 1.0 - 0.01), np.inf))
    p_b0 = d["teacher"][te_s].reshape(-1)[np.repeat(te_arm == "temporal",
                                                    te_p.shape[1])]
    gt = te_hard[te_arm == "temporal"].reshape(-1)
    rec_b0 = float((p_b0[gt > 0.5] > thr_b0).mean()) if (gt > 0.5).any() else 0.0
    print("    对照：B0 逐帧判据在同样口径的 1%% 工作点上召回 %.3f" % rec_b0)
    out["b0_recall_at_1pct"] = rec_b0
    ck.ck("1%% 误报率下时序召回 ≥ %.2f" % TH["recall_at_fpr1"],
          out["fpr0.01"]["recall_temporal"] >= TH["recall_at_fpr1"],
          "recall=%.4f（test FPR=%.4f%%，对照 B0=%.3f）"
          % (out["fpr0.01"]["recall_temporal"],
             100 * out["fpr0.01"]["test_fpr"], rec_b0), soft=True)
    metrics["fpr"] = out


# --------------------------------------------------------------------------- #
# 5. 导出对拍
# --------------------------------------------------------------------------- #
def sec5_export(ck, d, model, ckpt, metrics):
    ck.section("5. 导出对拍（torch ↔ 纯 numpy 从 blob 读权重）")
    import torch

    pol = X.collect_policy(model, ckpt["input_mean"], ckpt["input_std"])
    X.save_policy_npz(NPZ, pol)
    blob = X.build_blob(pol)
    BLOB.write_bytes(blob)
    m = X.manifest_of(blob)
    ck.ck("blob CRC32 自洽", m["crc32_ok"], "CRC=0x%08X，%d 字节"
          % (m["crc32"], m["blob_len"]))
    ck.ck("blob 头 n_heads=6 / head_dim / head_act 正确",
          m["n_heads"] == 6 and m["head_dim"][:6] == [1, 1, 4, 8, 1, 1]
          and m["head_act"][:6] == [1, 1, 2, 2, 1, 2],
          "head_dim=%s head_act=%s" % (m["head_dim"][:6], m["head_act"][:6]))
    ok_ck, msg = X.cross_check_export_blob(pol)
    ck.ck("与 pc/train/export_blob.py 打包结果逐字节一致", ok_ck is not False,
          msg, skip=ok_ck is None)

    # 逐帧对拍：同一份**原始**观测，torch 走 normalize+forward，numpy 走 blob。
    # 注意：numpy 参考实现必须在**每条会话内部**递推（h 清零），不能把几条会话
    # 首尾拼起来 —— 那样隐状态会跨会话串味（第一版就这么错过，误差 0.2）。
    p = X.parse_blob(BLOB.read_bytes())
    te_s = np.where(d["split"] == "test")[0][:6]
    raw = d["obs"][te_s]                                   # (6,384,106) 原始
    np_out = np.stack([X.forward_np(p, raw[i]) for i in range(len(te_s))])

    mu = np.asarray(ckpt["input_mean"], dtype=np.float64)
    sd = np.asarray(ckpt["input_std"], dtype=np.float64)
    v = np.clip(np.asarray(raw, dtype=np.float64), mu - 4 * sd, mu + 4 * sd)
    xn = ((v - mu) / sd).astype(F32)
    import torch as _torch
    with _torch.no_grad():
        t_out = model.forward_seq(_torch.from_numpy(xn)).cpu().numpy()

    # 逐头比较（tanh 头在两边都已套；softmax 头比 logits）
    errs = {}
    for i, (name, dim, act) in enumerate(X.HEADS):
        a = np_out[..., i, :dim]
        col0 = sum(d_ for _n, d_, _a in X.HEADS[:i])
        b = t_out[..., col0:col0 + dim]
        if act == "tanh":
            b = np.tanh(b)          # torch 侧 tanh 头是原始 logit，numpy 已套 tanh
        errs[name] = float(np.max(np.abs(a - b)))
    pe_np = X.p_esc_from_forward(np_out.reshape(-1, 6, 16), p)
    lg = t_out[..., M.COL_ESCALATE].reshape(-1)
    pe_t = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
    errs["p_esc"] = float(np.max(np.abs(pe_np - pe_t)))
    worst = max(errs.values())
    n_frames = int(raw.shape[0] * raw.shape[1])
    print("    逐头最大绝对误差：" + "  ".join("%s=%.2e" % (k, v) for k, v in errs.items()))
    ck.ck("torch ↔ numpy(blob) 最大误差 < %.0e" % TH["export_err"],
          worst < TH["export_err"], "max_abs_err=%.3e（%d 帧 × 6 头）"
          % (worst, n_frames))

    # act=2 与 act=3 两种编码的等价性
    blob3 = X.build_blob(pol, act3=True)
    p3 = X.parse_blob(blob3)
    pe3 = np.stack([X.p_esc_from_forward(X.forward_np(p3, raw[i]), p3)
                    for i in range(len(te_s))])
    d_eq = float(np.max(np.abs(pe3.reshape(-1) - pe_t)))
    ck.ck("act=2 / act=3 两种 blob 编码给出同一 p_esc", d_eq < TH["export_err"],
          "max|Δp_esc|=%.3e" % d_eq)

    metrics["export"] = {"per_head_err": errs, "max_err": worst,
                         "blob_len": int(m["blob_len"]), "crc32": int(m["crc32"]),
                         "n_heads": int(m["n_heads"]),
                         "head_dim": list(m["head_dim"][:6]),
                         "head_act": list(m["head_act"][:6]),
                         "act3_delta": d_eq, "frames": n_frames}


# --------------------------------------------------------------------------- #
# 6. 控制头没被学坏
# --------------------------------------------------------------------------- #
def sec6_control(ck, d, model, T, ckpt, metrics):
    ck.section("6. 控制头没有退化（联合训练没有把策略学坏）")
    import torch

    sessions = np.where(d["split"] == "test")[0]
    arm = d["arm"][sessions]
    correct = total = 0
    mse = 0.0
    for s in sessions[arm == "base"]:
        s = int(s)
        for st in range(0, d["obs_n"].shape[1] - 96 + 1, 96):
            x = torch.from_numpy(d["obs_n"][s][st:st + 96][None])
            with torch.no_grad():
                out = model.forward_seq(x).cpu().numpy()[0]
            pred_mode = out[:, M.COL["mode"]:M.COL["mode"] + 4].argmax(-1)
            correct += int((pred_mode == d["mode"][s][st:st + 96]).sum())
            total += 96
            cont = np.stack([np.tanh(out[:, M.COL["head_yaw"]]),
                             np.tanh(out[:, M.COL["vyaw"]]),
                             np.tanh(out[:, M.COL["vx"]])], axis=-1)
            mse += float(((cont - d["ctrl"][s][st:st + 96]) ** 2).mean())
    acc = correct / max(total, 1)
    ck.ck("mode 头准确率 ≥ %.2f" % TH["mode_acc"], acc >= TH["mode_acc"],
          "acc=%.4f（冻结基线 0.9909，n=%d 帧）" % (acc, total))
    lam = torch.sigmoid(model.ret).detach().numpy()
    print("    保留门 λ：mean=%.4f median=%.4f max=%.4f（冻结基线 mean=0.0433）"
          % (lam.mean(), float(np.median(lam)), lam.max()))
    metrics["control"] = {"mode_acc": acc, "frames": total,
                          "lambda_mean": float(lam.mean()),
                          "lambda_max": float(lam.max()),
                          "lambda_median": float(np.median(lam))}


# --------------------------------------------------------------------------- #
# 7. 隐状态记忆诊断
# --------------------------------------------------------------------------- #
def sec7_memory(ck, d, model, mean, std, metrics):
    ck.section("7. 隐状态记忆诊断（最脆弱的一点 的量化）")
    import torch

    s = int(np.where(d["split"] == "test")[0][0])
    base = d["obs"][s]
    frame = base[300].copy()
    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)

    def run(prefix_conf, n=24):
        seq = np.tile(frame, (n + 1, 1)).astype(F32)
        seq[:n, C.IX_CONF] = prefix_conf
        v = np.clip(np.asarray(seq, np.float64), mu - 4 * sd, mu + 4 * sd)
        x = ((v - mu) / sd).astype(F32)
        with torch.no_grad():
            out, hs = model.forward_seq(torch.from_numpy(x[None]), return_hidden=True)
        return (hs[0, -1].numpy(), float(torch.sigmoid(out[0, -1, M.COL_ESCALATE])))

    h_a, p_a = run(0.60)
    h_b, p_b = run(0.99)
    dh = float(np.linalg.norm(h_a - h_b) / max(np.linalg.norm(h_a), 1e-9))
    dp = abs(p_a - p_b)
    print("    同一当前帧、前 24 帧置信度 0.60 vs 0.99：")
    print("      |Δh|/|h| = %.5f（%.4f%%），|Δp_esc| = %.5f" % (dh, 100 * dh, dp))
    print("    解读：比例 ≫ 1%% 说明隐状态真的带历史；≈0.2%% 说明它几乎是瞬时的，")
    print("    那种情况下任何读 h 的线性头都不可能做时序判断。")
    metrics["memory"] = {"relative_h_delta": dh, "abs_p_delta": dp,
                         "conf_lo": 0.60, "conf_hi": 0.99, "prefix": 24}
    ck.ck("隐状态带可用历史（|Δh|/|h| > 1%）", dh > 0.01,
          "|Δh|/|h|=%.5f，|Δp_esc|=%.5f" % (dh, dp))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    t0 = time.perf_counter()
    ck = Checker()
    print("=" * 78)
    print("test_escalate.py —— CfC 第 6 个头 escalate 的验收")
    print("=" * 78)
    ensure_artifacts(ck)
    d, model, ckpt, T = load_all()
    print("[load] 数据集 %s；checkpoint %s；calibration=%s"
          % (S.DATASET.name, CKPT.name, ckpt.get("calibration")))

    metrics = {"thresholds": TH}
    sec1_phi(ck)
    sessions, pred, base_m = sec2_base(ck, d, model, T, metrics)
    sec3_temporal(ck, d, model, T, sessions, pred, metrics)
    sec4_fpr(ck, d, model, T, metrics)
    sec5_export(ck, d, model, ckpt, metrics)
    sec6_control(ck, d, model, T, ckpt, metrics)
    sec7_memory(ck, d, model, ckpt["input_mean"], ckpt["input_std"], metrics)

    n = ck.summary()
    elapsed = time.perf_counter() - t0
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics["elapsed_s"] = elapsed
    METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print("\n" + "-" * 78)
    print("汇总：PASS %d / FAIL %d / WARN %d / SKIP %d，耗时 %.1f s"
          % (n["PASS"], n["FAIL"], n["WARN"], n["SKIP"], elapsed))
    print("指标落盘：%s" % METRICS)
    if n["WARN"]:
        print("\n未达质量目标（不影响退出码，但必须如实上报）：")
        for i in ck.items:
            if i["status"] == "WARN":
                print("  - %s（%s）" % (i["name"], i["detail"]))
    if not ck.ok:
        print("\n失败项：")
        for i in ck.items:
            if i["status"] == "FAIL":
                print("  - %s（%s）" % (i["name"], i["detail"]))
    print("PASS" if ck.ok else "FAIL")
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
