# -*- coding: utf-8 -*-
"""scan_dyaw.py — 航向环 D 增益 ``k_dyaw`` 的扫描。

    D:\\smartcar\\.venv\\Scripts\\python.exe scan_dyaw.py --seeds 4

背景：``REPORT.md`` §0.3 测出小误差区头身分离比全部 <1.4（CfC 只有 0.71），
根因是航向环"纯 P 控双积分"，闭环极点 ``-0.6 ± 7.13j`` ⇒ ζ≈0.08。本脚本扫
固件新增的 D 增益 :attr:`params.FirmwareParams.k_dyaw`，量四件事：

* ``sep_ratio``   小误差区 头部角行程 / 机身角行程（核心指标，越小越好）
* ``zeta``        航向环线性化阻尼比（``sim.closed_loop_eigs``，解析值）
* ``jerk1``       归一化动作一阶差分（平滑度）
* ``eps_rms_small`` 小误差区的 ε RMS（跟踪误差，别把"不动了"当成功）

**符号**：固件那一行是 ``yaw_u = k_yaw*(...) - k_dyaw*yaw_rate``，而
``yaw_rate = d(yaw)/dt``；``ψ̈ = -k_yaw_accel·yaw_u``（P 项必须是负反馈才可能
稳定，见 REPORT §3 的同一套推导），所以 ζ 随 ``k_dyaw`` **减小**而增大 ——
起阻尼作用的是**负的** k_dyaw。脚本把两半都扫出来，让数字说话。

产出：``dyaw_scan.md`` / ``dyaw_scan.json`` / ``figs/fig7_dyaw.png``。
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
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controllers as C          # noqa: E402
import eval as E                 # noqa: E402
import params as P               # noqa: E402
import sim                       # noqa: E402

OUT_MD = HERE / "dyaw_scan.md"
OUT_JSON = HERE / "dyaw_scan.json"
FIG = HERE / "figs" / "fig7_dyaw.png"

#: 任务要求的扫描点
KDS_REQ = (0.0, 0.2, 0.5, 1.0, 2.0)
#: 镜像（阻尼方向），用来证明符号而不是断言符号
KDS_MIRROR = (-0.2, -0.5, -1.0, -2.0)


#: 固件 `control_arbiter.c` 的默认值。仿真 `FirmwareParams` 默认把速率限幅关掉
#: （为了 eval.py 能复现 REPORT 基线），要跑"固件口径"就在这里显式打开。
FW_DEFAULTS = {"vx_rate_max": 1.0, "vyaw_rate_max": 400.0,
               "vx_stop_eps": 0.005, "vyaw_stop_eps": 0.5}


def calibrated_params(verbose=True):
    """与 ``eval.main`` 同一个标定口径（k_torque / k_yaw_accel）+ 固件默认限幅。"""
    base = replace(P.SimParams(),
                   fw=replace(P.FirmwareParams(), **FW_DEFAULTS))
    kt, ky, _ = sim.calibrate(base, verbose=verbose)
    return replace(base, plant=replace(base.plant, k_torque=kt, k_yaw_accel=ky))


def yaw_zeta(sp, kd):
    """航向环闭环阻尼比（线性化解析值，含 D 项）。"""
    _, e = sim.closed_loop_eigs(sp, k_dyaw=kd)
    lam = complex(e[np.argmax(np.imag(np.abs(e) * 0 + e))])
    # 取虚部绝对值最大的那个极点
    lam = max(e, key=lambda z: abs(complex(z).imag))
    return float(sim.damping_ratio(lam)), complex(lam)


def run_point(sp, kd, names, seeds, scenarios, verbose=True):
    """在给定的 k_dyaw 上跑 场景 × 控制器 × 种子。"""
    sp2 = replace(sp, fw=replace(sp.fw, k_dyaw=float(kd)))
    rows = []
    for scn in scenarios:
        dur = E.DURATION[scn]
        dt = sp2.fw.ctrl_dt_ms * 1e-3
        for name in names:
            for sd in seeds:
                pol = C.make(name)
                scene = sim.make_scene(scn, sd, dur, dt)
                tr = sim.ClosedLoop(sp2, pol, scene, seed=sd).run()
                m = E.run_metrics(tr)
                w = tr.warm > 0
                small = w & (np.abs(tr.eps_true) < E.BAND_DEG)
                if np.count_nonzero(small) > 4:
                    m["eps_rms_small"] = float(
                        np.sqrt(np.mean(tr.eps_true[small] ** 2)))
                    m["body_rate_small"] = float(
                        np.mean(np.abs(tr.body_rate[small])))
                else:
                    m["eps_rms_small"] = float("nan")
                    m["body_rate_small"] = float("nan")
                m.update(scenario=scn, controller=name, seed=int(sd), k_dyaw=float(kd))
                rows.append(m)
        if verbose:
            sub = [r for r in rows if r["scenario"] == scn]
            print("    [%-8s] sep=%.2f  eps=%.2f  (%ds)"
                  % (scn, np.nanmean([r["sep_ratio"] for r in sub]),
                     np.nanmean([r["eps_rms"] for r in sub]), len(sub)))
    return rows


def agg(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        out[k] = (float(np.mean(v)) if v.size else float("nan"),
                  float(np.std(v)) if v.size else float("nan"))
    return out


KEYS = ("sep_ratio", "eps_rms", "eps_rms_small", "jerk1", "jerk2",
        "body_rate_small", "success", "head_share_cmd")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="扫航向环 D 增益 k_dyaw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--controllers", default="teacher,cfc,mlp,pd")
    ap.add_argument("--scenarios", default=",".join(sim.SCENARIOS))
    ap.add_argument("--kds", default=",".join(str(k) for k in
                                               KDS_REQ + KDS_MIRROR))
    ap.add_argument("--replot", action="store_true",
                    help="只从 dyaw_scan.json 重画 fig7，不重跑仿真")
    args = ap.parse_args(argv)
    if args.replot:
        return replot() or 0

    names = [s.strip() for s in args.controllers.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    kds = [float(s) for s in args.kds.split(",") if s.strip()]
    seeds = list(range(args.seeds))

    t0 = time.perf_counter()
    print("[calib] k_torque / k_yaw_accel …")
    sp = calibrated_params(verbose=True)
    print("        k_torque=%.4e  k_yaw_accel=%.4f"
          % (sp.plant.k_torque, sp.plant.k_yaw_accel))

    rows = []
    for kd in kds:
        z, lam = yaw_zeta(sp, kd)
        print("\n[k_dyaw=%+.2f]  zeta=%.3f  poles=%s"
              % (kd, z, np.array2string(np.array([lam]), precision=3)))
        sub = run_point(sp, kd, names, seeds, scenarios)
        rows.extend(sub)

    # ---- 汇总 ----
    summary = {}
    for kd in kds:
        summary[kd] = {}
        for name in names:
            sel = [r for r in rows
                   if r["k_dyaw"] == kd and r["controller"] == name]
            summary[kd][name] = agg(sel, KEYS)
    zeta = {kd: yaw_zeta(sp, kd)[0] for kd in kds}

    OUT_JSON.write_text(json.dumps(
        {"meta": {"seeds": args.seeds, "controllers": names,
                  "scenarios": scenarios, "k_dyaw": kds,
                  "k_torque": sp.plant.k_torque,
                  "k_yaw_accel": sp.plant.k_yaw_accel,
                  "yaw_damping": sp.plant.yaw_damping,
                  "zeta": {("%.3f" % k): v for k, v in zeta.items()},
                  "elapsed_s": time.perf_counter() - t0},
         "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")

    lines = ["# 航向环 D 增益 `k_dyaw` 扫描", "",
             "仿真：`scan_dyaw.py`，%d 场景 × %d 控制器 × %d 种子 = %d 次/点。"
             " 标定：`k_torque=%.4e`、`k_yaw_accel=%.4f`、`yaw_damping=%.2f`。"
             % (len(scenarios), len(names), len(seeds),
                len(scenarios) * len(names) * len(seeds),
                sp.plant.k_torque, sp.plant.k_yaw_accel, sp.plant.yaw_damping),
             "",
             "固件控制律：`yaw_u = k_yaw*(yaw-q_head-stable_yaw) - k_dyaw*yaw_rate`，"
             "`yaw_rate = d(yaw)/dt`。ζ 是线性化解析值（`sim.closed_loop_eigs`）。", ""]
    for name in names:
        lines += ["## 控制器 `%s`" % name, "",
                  "| k_dyaw | ζ(航向) | 头身分离比 | 小误差 ε RMS (°) | ε RMS (°) | "
                  "jerk1 | jerk2 | 机身平均|ω| (°/s) | 成功率 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for kd in kds:
            g = summary[kd][name]
            lines.append(
                "| %+.2f | %.3f | %.2f ± %.2f | %.2f | %.2f | %.3f | %.3f | %.2f | %.1f%% |"
                % (kd, zeta[kd], g["sep_ratio"][0], g["sep_ratio"][1],
                   g["eps_rms_small"][0], g["eps_rms"][0], g["jerk1"][0],
                   g["jerk2"][0], g["body_rate_small"][0],
                   100.0 * g["success"][0]))
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    make_figure(summary, zeta, kds, names)
    print("\n[out] %s\n[out] %s\n[out] %s\n[done] %.0f s"
          % (OUT_MD, OUT_JSON, FIG, time.perf_counter() - t0))
    return 0


def make_figure(summary, zeta, kds, names):
    """fig7：左＝头身分离比 vs k_dyaw，右＝解析 ζ vs k_dyaw，共用 x 轴。

    两个符号域画在**同一条 x 轴**上并给 ``k_dyaw>0`` 打底灰：那半边的航向环
    ζ<0、发散，人直接出了视野，分离比塌到 0 是"人跟丢了"的假象，不是"头和
    身都不动了"。这条注释必须在图上，否则图会被读成"加大增益就能只动头"。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 与 make_report.py 同一套中文字体设置
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3),
                                   gridspec_kw={"width_ratios": [1.45, 1.0]})
    colors = {"teacher": "#888888", "cfc": "#1f77b4", "mlp": "#d62728",
              "pd": "#2ca02c"}
    xs = np.array(sorted(kds), dtype=float)
    xlim = (xs.min() - 0.15, xs.max() + 0.15)
    kd_unstable = min([k for k in xs if zeta[k] > 0.0] or [0.0])

    for ax in (ax1, ax2):
        ax.axvspan(kd_unstable, xlim[1], color="#d62728", alpha=0.10, lw=0)
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_xlabel("k_dyaw  (torque-count / (deg/s))")

    for name in names:
        y = np.array([summary[kd][name]["sep_ratio"][0] for kd in xs])
        e = np.array([summary[kd][name]["sep_ratio"][1] for kd in xs])
        ax1.errorbar(xs, y, yerr=e, marker="o", ms=4, lw=1.6, capsize=2,
                     color=colors.get(name, None), label=name)
    ax1.axhline(1.0, color="k", ls="--", lw=1.0)
    ax1.set_ylabel("头身分离比（小误差区 |ε|<15°，头行程 / 身行程）")
    ax1.set_title("头身分离比 vs k_dyaw", fontsize=10)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.text(xlim[0] + 0.05, 1.04, "=1（头和身走一样多）", fontsize=8)
    ax1.set_ylim(-0.05, 1.62)
    ax1.annotate("ζ<0：航向环发散，人出视野\n分离比塌到 0 是假象",
                 xy=(1.02, 0.05), xytext=(0.50, 1.22), fontsize=7.5,
                 color="#8c1d18",
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#8c1d18"))

    ax2.plot(xs, [zeta[kd] for kd in xs], color="#9467bd", ls="-", lw=1.6,
             marker="s", ms=4)
    ax2.axhline(0.0, color="k", lw=0.8, ls=":")
    ax2.axhline(0.7, color="#9467bd", lw=0.8, ls="--")
    ax2.text(xlim[0] + 0.05, 0.73, "ζ=0.7（常见整定目标）", fontsize=8,
             color="#9467bd")
    ax2.set_ylabel("航向环阻尼比 ζ（线性化解析值）")
    ax2.set_title("阻尼比 vs k_dyaw", fontsize=10)
    ax2.set_ylim(-1.15, 1.15)

    fig.suptitle("RIG-Hover：航向 D 项对头身分离比的影响"
                 "（5 场景 × 4 控制器 × 6 种子）", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    plt.close(fig)


def replot():
    """从 ``dyaw_scan.json`` 重画 fig7（不重跑仿真）。"""
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    rows = data["rows"]
    kds = sorted({r["k_dyaw"] for r in rows})
    names = data["meta"]["controllers"]
    summary = {kd: {n: agg([r for r in rows
                            if r["k_dyaw"] == kd and r["controller"] == n], KEYS)
                    for n in names} for kd in kds}
    zeta = {float(k): float(v) for k, v in data["meta"]["zeta"].items()}
    make_figure(summary, zeta, kds, names)
    print("[out] %s（重画，未重跑）" % FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
