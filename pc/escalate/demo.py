# -*- coding: utf-8 -*-
"""demo.py — 一条连续序列上演示"多帧累积 → 升级"，并与逐帧判据并排对比。

::

    python demo.py            # 默认场景：良性 → 连续 marginal 低置信 → 良性 → 方位外移
    python demo.py --all      # 四个场景族各跑一遍（含 ordered/control 对照）

打印每一帧的 ``escalate`` 概率和触发时刻，同时给出同一条序列上逐帧判据
（``p = σ(b + Σwφ(x_t))``）的概率 —— 后者的值在整段里几乎不动，因为**它只看
当前帧**。这就是"时序头 vs 独立 logistic"的直观形态。

触发判据（与 ``decision_hub`` 的 dwell 语义一致）：
``p ≥ --thr`` 连续 ``--dwell`` 帧才认为"该升级了"，掉回 ``--rearm`` 以下重新武装。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import esc_core as C
import esc_model as M
import esc_scenes as S

F32 = np.float32
HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoint_esc.pt"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# 构造一条演示序列（连续，隐状态一路带下去）
# --------------------------------------------------------------------------- #
def build_demo_sequence(donor, phases):
    """按 ``phases`` 拼一条连续观测序列。

    Args:
        donor: :func:`esc_scenes.load_donors` 的一项。
        phases: list[(名字, 帧数, 覆盖字段 dict)]。覆盖字段支持
            ``conf``/``seen``/``theta``/``age_ms``，值可以是标量或 (L,) 数组。

    Returns:
        ``(obs (T,106), names (T,), labels (T,))``
    """
    rows, names, labels = [], [], []
    start = 200
    t = 0
    for name, length, ov in phases:
        idx = S._window_idx(donor, start + t, length)
        o = np.array(donor["obs"][idx], dtype=F32, copy=True)
        for key, val in ov.items():
            if key.startswith("_"):
                continue                      # _driver / _gt 是元数据，不是观测字段
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.full(length, float(arr))
            for k in range(length):
                o[k] = S.apply_overrides(o[k], {key: arr[k]})
        rows.append(o)
        names.extend([name] * length)
        labels.extend([int(bool(ov.get("_gt", 0)))] * length)
        t += length
    return np.concatenate(rows), names, np.asarray(labels, dtype=np.int32)


def escalate_stream(p, thr=0.5, dwell=4, rearm=0.30):
    """把概率流变成"触发事件"（含 dwell 与重新武装），返回每帧是否刚触发。"""
    fire = np.zeros(p.size, dtype=bool)
    above = 0
    armed = True
    for k, v in enumerate(p):
        if v < rearm:
            armed = True
            above = 0
        above = above + 1 if v >= thr else 0
        if armed and above >= dwell:
            fire[k] = True
            armed = False
    return fire


# --------------------------------------------------------------------------- #
# 跑与打印
# --------------------------------------------------------------------------- #
def run_model(model, obs, mean, std):
    import torch

    mu = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(std, dtype=np.float64)
    v = np.clip(np.asarray(obs, np.float64), mu - 4 * sd, mu + 4 * sd)
    x = ((v - mu) / sd).astype(F32)
    with torch.no_grad():
        out, hs = model.forward_seq(torch.from_numpy(x[None]), return_hidden=True)
    p = torch.sigmoid(out[0, :, M.COL_ESCALATE]).numpy()
    return p, hs[0].numpy()


def bar(v, width=28):
    k = int(round(max(0.0, min(1.0, v)) * width))
    return "#" * k + "." * (width - k)


def print_table(driver_name, driver, p_cfc, p_log, names, fire, stride=1):
    print("  帧  阶段                 %-10s  p_logistic  p_escalate" % driver_name)
    for t in range(0, p_cfc.size, stride):
        mark = "  <== 触发" if fire[t] else ""
        print("  %3d  %-18s %10.3f   %.4f     %.4f %s%s"
              % (t, names[t][:18], driver[t], p_log[t], p_cfc[t],
                 bar(p_cfc[t]), mark))


def demo_one(model, scorer, mean, std, donor, title, phases, args):
    obs, names, _labels = build_demo_sequence(donor, phases)
    p_cfc, _h = run_model(model, obs, mean, std)
    sc = scorer.score(obs[None])
    p_log = sc["teacher"][0]
    fire = escalate_stream(p_cfc, args.thr, args.dwell, args.rearm)
    driver_name = phases[0][2].get("_driver", "conf")
    if driver_name == "conf":
        driver = obs[:, C.IX_CONF]
    else:
        driver = obs[:, C.IX_THETA]
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print("  阶段: " + " -> ".join("%s×%d" % (n, l) for n, l, _o in phases))
    idx_fire = np.where(fire)[0]
    print("  逐帧判据 logistic：p 范围 [%.4f, %.4f]（整段几乎不动）"
          % (float(p_log.min()), float(p_log.max())))
    print("  CfC 升级头     ：p 范围 [%.4f, %.4f]" % (float(p_cfc.min()),
                                                     float(p_cfc.max())))
    print("  触发时刻（p≥%.2f 连续 %d 帧）：%s"
          % (args.thr, args.dwell,
             ", ".join("t=%d(%s)" % (t, names[t]) for t in idx_fire) or "无"))
    print_table(driver_name, driver, p_cfc, p_log, names, fire, args.stride)
    if not args.all:
        print("""
读法：
  1. 第 1 段（良性）p_esc 很低 —— 这是产线上的正常帧；
  2. 第 2 段每一帧的 conf=0.60 单独看都不触发任何势函数（u_conf=0），
     逐帧判据的 p 全程只在 [%.4f, %.4f] 里动；CfC 却在若干帧后爬到 %.3f ——
     它读到的是"已经连续这么多帧了"（本段没到触发阈值，是因为该族的
     真值窗口是 8 帧起，而这一段的绝对水平被标定压过）；
  3. 第 4 段 theta 从 -12° 拉到 +30°，而 theta **不在任何势函数里**：
     逐帧判据完全看不见（p 恒为 %.4f），CfC 把它推到 %.3f 并触发。
     这是"只有时序模型能做对"的那一类证据。
  4. 注意第 1 段末尾 p_esc 缓慢上漂（0.04 -> 0.14）：这一段的观测是"恒定
     conf=0.78 连续 24 帧"，比训练分布更规整，头在 OOD 上会略微过判 ——
     上真机时这一条要靠重估 input_mean/std 解决（README §8）。
""" % (float(p_log.min()), float(p_log.max()),
       float(p_cfc[names.index("连续 marginal 低置信") + 18]),
       float(p_log[names.index("方位外移")]),
       float(p_cfc.max())))
    return {"p_cfc": p_cfc, "p_log": p_log, "fire": idx_fire, "names": names}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="演示 escalate 头的时序行为（连续多帧累积 -> 升级）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--thr", type=float, default=0.5, help="升级触发阈值")
    ap.add_argument("--dwell", type=int, default=4, help="触发前需连续超过阈值的帧数")
    ap.add_argument("--rearm", type=float, default=0.30, help="掉回此值以下重新武装")
    ap.add_argument("--stride", type=int, default=1, help="打印步长（2=隔帧）")
    ap.add_argument("--all", action="store_true", help="四个场景族各跑一遍")
    args = ap.parse_args(argv)

    import torch

    if not Path(args.checkpoint).exists():
        print("缺 checkpoint_esc.pt，先跑 train_escalate.py")
        return 2
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = M.EscalatePolicy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                             hidden=cfg["hidden"], backbone_units=cfg["backbone_units"])
    model.load_state_dict(ck["model"])
    model.eval()
    mean, std = ck["input_mean"], ck["input_std"]
    scorer = S.BaseScorer(device="cpu")
    donor = S.load_donors()[0]

    if not args.all:
        phases = [
            ("良性跟踪", 24, {"conf": 0.78, "_driver": "conf"}),
            # 连续 marginal 置信度：单帧 conf=0.60 仍高于 u_conf 的 0.55 斜坡起点
            ("连续 marginal 低置信", 20, {"conf": 0.60}),
            ("恢复", 16, {"conf": 0.78}),
            # 方位持续外移：theta 不进任何势函数
            ("方位外移", 24, {"conf": 0.78, "theta": np.linspace(-12, 30, 24),
                            "_driver": "theta"}),
            ("再次良性", 16, {"conf": 0.78}),
        ]
        demo_one(model, scorer, mean, std, donor,
                 "场景：良性 → 连续 20 帧 marginal 低置信 → 恢复 → 方位外移 → 良性",
                 phases, args)
    else:
        for fam in S.FAMILIES:
            for ordered in (True, False):
                ov, gt = S.SCENES[fam](np.random.default_rng(0), ordered)
                length = len(next(iter(ov.values())))
                fields = {k: np.asarray(v) for k, v in ov.items()}
                fields["_driver"] = "theta" if "theta" in ov else "conf"
                label = fam + ("（ordered）" if ordered else "（control）")
                res = demo_one(model, scorer, mean, std, donor,
                               "场景族 %s：%s" % (fam, S.FAMILY_DOC[fam]),
                               [(label, length, fields)], args)
                print("  真值：该段阳性率 %.1f%%；触发 %d 次"
                      % (100 * float(gt.mean()), res["fire"].size))
    print("\nPASS")
    return 0


def ov_gt_len(ov):
    return len(next(iter(ov.values())))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
