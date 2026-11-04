# -*- coding: utf-8 -*-
"""diag_sep.py — 头身分离的两个补充实验（``scan_dyaw.py`` 的追问）。

    python diag_sep.py --exp fine        # 细扫 + 逐种子配对统计
    python diag_sep.py --exp headonly    # 「不许机身动」消融

**为什么还要跑这两个**：``scan_dyaw.py`` 的大表显示，D 项把教师/PD 的跟踪误差
和 jerk 都按数量级改善了（ζ 0.08 → 0.26~0.53），但 CfC 的头身分离比只是在
0.7~0.9 之间晃，并没有变成「只动头」。于是要分清机身那些角度行程里：

* 有多少是**控制器主动下的 vyaw 指令**（``head_share_cmd``）
* 有多少是**指令之外**的（欠阻尼振荡 + 跟随误差）

``--exp headonly`` 直接把策略的 vyaw 指令按比例压掉，看分离比天花板在哪：
如果压掉 vyaw 之后分离比能到 3~5，说明「只动头」缺的是**策略侧的取舍**，
不是固件阻尼；如果压掉之后分离比还是 ~1，那说明瓶颈在别处（头部行程不够 / FOV）。

产出：``diag_sep.md`` / ``diag_sep.json``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controllers as C          # noqa: E402
import eval as E                 # noqa: E402
import params as P               # noqa: E402
import sim                       # noqa: E402


class VyawScaled(C.Policy):
    """把一个策略的 ``vyaw_des`` 乘上 ``gain``（其它通道原样）。

    只用于消融："如果策略不在小误差区推机身，头身分离能到多少"。
    """

    def __init__(self, base, gain=1.0):
        self.base = base
        self.gain = float(gain)
        self.name = base.name
        self.head_max = base.head_max
        self.vyaw_max = base.vyaw_max
        self.vx_max = base.vx_max
        # 透传固件侧的注入点，否则包装一层就把 D 通道丢了
        if getattr(base, "k_dyaw", None) is not None:
            self.k_dyaw = base.k_dyaw
        if getattr(base, "head_rate_comp", None) is not None:
            self.head_rate_comp = base.head_rate_comp

    def reset(self):
        self.base.reset()

    def step(self, frame, prev_act, sv):
        o = self.base.step(frame, prev_act, sv)
        return sim.CtrlOut(head_des=o.head_des, vx_des=o.vx_des,
                           vyaw_des=o.vyaw_des * self.gain,
                           mode=o.mode, emote=o.emote)


KEYS = ("sep_ratio", "head_share", "eps_rms", "eps_rms_small", "jerk1",
        "body_rate_small", "head_rate_small", "success", "in_fov", "sat_head")


#: 固件 `control_arbiter.c` 的默认值（仿真 params 默认关限幅，见 params.py）
FW_DEFAULTS = {"vx_rate_max": 1.0, "vyaw_rate_max": 400.0,
               "vx_stop_eps": 0.005, "vyaw_stop_eps": 0.5}


def calibrated_params(verbose=True):
    base = replace(P.SimParams(),
                   fw=replace(P.FirmwareParams(), **FW_DEFAULTS))
    kt, ky, _ = sim.calibrate(base, verbose=verbose)
    return replace(base, plant=replace(base.plant, k_torque=kt, k_yaw_accel=ky))


def one(sp, kd, pol, scn, sd):
    sp2 = replace(sp, fw=replace(sp.fw, k_dyaw=float(kd)))
    dt = sp2.fw.ctrl_dt_ms * 1e-3
    scene = sim.make_scene(scn, sd, E.DURATION[scn], dt)
    tr = sim.ClosedLoop(sp2, pol, scene, seed=sd).run()
    m = E.run_metrics(tr)
    w = tr.warm > 0
    small = w & (np.abs(tr.eps_true) < E.BAND_DEG)
    if np.count_nonzero(small) > 4:
        m["eps_rms_small"] = float(np.sqrt(np.mean(tr.eps_true[small] ** 2)))
        m["body_rate_small"] = float(np.mean(np.abs(tr.body_rate[small])))
        m["head_rate_small"] = float(np.mean(np.abs(
            np.gradient(tr.q_head[small], tr.t[small]))))
    else:
        m["eps_rms_small"] = m["body_rate_small"] = m["head_rate_small"] = \
            float("nan")
    m.update(scenario=scn, seed=int(sd), k_dyaw=float(kd))
    return m


def mean_std(rows, k):
    v = np.array([r[k] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.mean(v)) if v.size else float("nan"),
            float(np.std(v)) if v.size else float("nan"))


def paired(rows, key, baseline_kd):
    """同一 (场景, 种子) 上 kd vs baseline 的配对差 → 均值 ± 标准误 + 改善种子占比。"""
    by = {(r["scenario"], r["seed"], r["k_dyaw"]): r[key] for r in rows}
    acc = {}
    for (scn, sd, kd), v in by.items():
        if kd == baseline_kd:
            continue
        b = by.get((scn, sd, baseline_kd))
        if b is None or not np.isfinite(v) or not np.isfinite(b):
            continue
        acc.setdefault(kd, []).append(v - b)
    out = {}
    for kd, vv in acc.items():
        v = np.array(vv, dtype=float)
        se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
        out[kd] = (float(np.mean(v)), se, float(np.mean(v > 0)), len(v))
    return out


def exp_fine(sp, args):
    names = ["cfc", "teacher", "pd"]
    kds = [0.0, -0.1, -0.2, -0.3, -0.5, -0.8]
    seeds = list(range(args.seeds))
    rows = []
    t0 = time.perf_counter()
    for kd in kds:
        for name in names:
            for scn in sim.SCENARIOS:
                for sd in seeds:
                    r = one(sp, kd, C.make(name), scn, sd)
                    r["controller"] = name
                    rows.append(r)
        print("  [kd=%+.2f] done (%.0fs)" % (kd, time.perf_counter() - t0))

    lines = ["# 补充实验 1：细扫 k_dyaw（%d 种子，逐种子配对）" % len(seeds), "",
             "配对差 = 同一 (场景, 种子) 上该 k_dyaw 减 k_dyaw=0；括号里是标准误。", ""]
    summary = {}
    for name in names:
        for scn in sim.SCENARIOS:
            sub = [r for r in rows if r["controller"] == name and r["scenario"] == scn]
            p = paired(sub, "sep_ratio", 0.0)
            pe = paired(sub, "eps_rms", 0.0)
            lines += ["## `%s` / `%s`" % (name, scn), "",
                      "| k_dyaw | 分离比 | Δ分离比(配对) | ε RMS (°) | Δε RMS(配对) | "
                      "成功率 | 机身ω | 头部ω |",
                      "|---|---|---|---|---|---|---|---|"]
            for kd in kds:
                s = [r for r in sub if r["k_dyaw"] == kd]
                sep = mean_std(s, "sep_ratio")
                eps = mean_std(s, "eps_rms")
                br = mean_std(s, "body_rate_small")
                hr = mean_std(s, "head_rate_small")
                succ = mean_std(s, "success")
                d1 = p.get(kd)
                d2 = pe.get(kd)
                cell1 = "—" if d1 is None else "%+.3f (%.3f) 改善 %.0f%%" % (
                    d1[0], d1[1], 100 * d1[2])
                cell2 = "—" if d2 is None else "%+.2f (%.2f)" % (d2[0], d2[1])
                lines.append("| %+.2f | %.2f ± %.2f | %s | %.2f | %s | %.1f%% | "
                             "%.1f | %.1f |"
                             % (kd, sep[0], sep[1], cell1, eps[0], cell2,
                                100 * succ[0], br[0], hr[0]))
            lines.append("")
            summary[(name, scn)] = {
                str(kd): {"sep": mean_std([r for r in sub if r["k_dyaw"] == kd],
                                          "sep_ratio"),
                          "eps_rms": mean_std([r for r in sub if r["k_dyaw"] == kd],
                                              "eps_rms")}
                for kd in kds}
    return rows, summary, lines


def exp_headonly(sp, args):
    seeds = list(range(args.seeds))
    gains = [1.0, 0.5, 0.2, 0.0]
    kds = [0.0, -0.3]
    rows = []
    t0 = time.perf_counter()
    for kd in kds:
        for g in gains:
            for scn in sim.SCENARIOS:
                for sd in seeds:
                    pol = VyawScaled(C.make("cfc"), gain=g)
                    r = one(sp, kd, pol, scn, sd)
                    r.update(vyaw_gain=g, controller="cfc")
                    rows.append(r)
        print("  [kd=%+.2f] done (%.0fs)" % (kd, time.perf_counter() - t0))

    lines = ["", "# 补充实验 2：「不许机身动」消融（CfC，vyaw_des × gain）", "",
             "`gain=0` 时固件拿到的 vyaw 指令恒为 0，机身只剩残余振荡与参考积分。", "",
             "| k_dyaw | vyaw gain | 分离比 | 头部占比(指令) | ε RMS (°) | 成功率 | "
             "视野内 | 头部饱和 | 机身ω |", "|---|---|---|---|---|---|---|---|---|"]
    for kd in kds:
        for g in gains:
            s = [r for r in rows if r["k_dyaw"] == kd and r["vyaw_gain"] == g]
            lines.append("| %+.2f | %.1f | %.2f ± %.2f | %.3f | %.2f | %.1f%% | "
                         "%.1f%% | %.1f%% | %.2f |"
                         % (kd, g, mean_std(s, "sep_ratio")[0],
                            mean_std(s, "sep_ratio")[1],
                            mean_std(s, "head_share_cmd")[0],
                            mean_std(s, "eps_rms")[0],
                            100 * mean_std(s, "success")[0],
                            100 * mean_std(s, "in_fov")[0],
                            100 * mean_std(s, "sat_head")[0],
                            mean_std(s, "body_rate_small")[0]))
    lines.append("")
    return rows, lines


def exp_bodyrate(sp, args):
    """D 项用"视线角速度"还是"机身角速度"。

    固件的 ``yaw_rate = -gyro_yaw_comp`` 是**视线**角速度（IMU 装在头上），
    等于机身角速度加头部角速度。头部跟踪时它可达几十 deg/s，全部灌进航向
    力矩。``comp=1`` 表示先减掉头部角速度（这里用真值 ``qd_head``，是固件
    从 ``q_head`` 回读差分能拿到的东西的**上界**）。
    """
    seeds = list(range(args.seeds))
    cases = [(0.0, 0.0, 1.0), (-0.2, 0.0, 1.0), (-0.3, 0.0, 1.0),
             (-0.2, 1.0, 1.0), (-0.3, 1.0, 1.0), (-0.5, 1.0, 1.0),
             (-0.8, 1.0, 1.0),
             (0.0, 0.0, 0.0), (-0.3, 0.0, 0.0), (-0.3, 1.0, 0.0)]
    rows = []
    t0 = time.perf_counter()
    for kd, comp, gain in cases:
        for scn in sim.SCENARIOS:
            for sd in seeds:
                pol = C.make("cfc")
                pol.k_dyaw = kd
                pol.head_rate_comp = comp
                if gain != 1.0:
                    pol = VyawScaled(pol, gain=gain)
                r = one(sp, kd, pol, scn, sd)
                r.update(vyaw_gain=gain, head_rate_comp=comp, controller="cfc")
                rows.append(r)
        print("  [kd=%+.2f comp=%d gain=%.1f] done (%.0fs)"
              % (kd, comp, gain, time.perf_counter() - t0))

    lines = ["", "# 补充实验 3：D 项用「视线角速度」还是「机身角速度」", "",
             "`comp=0` = 固件现状（`yaw_rate` = 机身 + 头部，IMU 在头上）；"
             "`comp=1` = 先减掉头部角速度（用真值 `qd_head`，是固件从 `q_head`"
             "回读差分能达到的上界）。`gain=0` 那三行是「只动头」工况，用来暴露"
             "头部角速度污染。", "",
             "| k_dyaw | comp | vyaw gain | 分离比 | ε RMS (°) | 成功率 | "
             "机身ω | 头部ω |", "|---|---|---|---|---|---|---|---|"]
    for kd, comp, gain in cases:
        s = [r for r in rows if r["k_dyaw"] == kd and r["head_rate_comp"] == comp
             and r["vyaw_gain"] == gain]
        lines.append("| %+.2f | %d | %.1f | %.2f ± %.2f | %.2f | %.1f%% | %.2f | %.2f |"
                     % (kd, comp, gain, mean_std(s, "sep_ratio")[0],
                        mean_std(s, "sep_ratio")[1], mean_std(s, "eps_rms")[0],
                        100 * mean_std(s, "success")[0],
                        mean_std(s, "body_rate_small")[0],
                        mean_std(s, "head_rate_small")[0]))
    lines.append("")
    return rows, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="fine",
                    choices=("fine", "headonly", "bodyrate", "both", "all"))
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    sp = calibrated_params()

    payload, lines = {}, []
    if args.exp in ("fine", "both", "all"):
        rows, summ, ln = exp_fine(sp, args)
        payload["fine"] = {"rows": rows, "summary": {
            "%s|%s" % k: v for k, v in summ.items()}}
        lines += ln
    if args.exp in ("headonly", "both", "all"):
        rows2, ln = exp_headonly(sp, args)
        payload["headonly"] = {"rows": rows2}
        lines += ln
    if args.exp in ("bodyrate", "all"):
        rows3, ln = exp_bodyrate(sp, args)
        payload["bodyrate"] = {"rows": rows3}
        lines += ln

    payload["meta"] = {"seeds": args.seeds, "elapsed_s": time.perf_counter() - t0}
    (HERE / "diag_sep.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    (HERE / "diag_sep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[out] diag_sep.md / diag_sep.json  [done] %.0f s"
          % (time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
