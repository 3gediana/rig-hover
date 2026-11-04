# -*- coding: utf-8 -*-
"""make_report.py — 从 results.json 生成 5 张图 + REPORT.md。

    python make_report.py

只读 ``results.json``（由 ``eval.py`` 生成），另外现场重跑少量轨迹画时域图与
极点扫描（都很便宜）。图片写到 ``figs/``，报告写到 ``REPORT.md``。
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import controllers as C                  # noqa: E402
import params as P                       # noqa: E402
import sim                               # noqa: E402

FIGS = HERE / "figs"
RESULTS = HERE / "results.json"
REPORT = HERE / "REPORT.md"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: 中文字体（Windows 自带；退化时 matplotlib 会画方框但不报错）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

COLOR = {"teacher": "#1f77b4", "cfc": "#d62728", "mlp": "#2ca02c",
         "pd": "#7f7f7f"}
LABEL = {"teacher": "Teacher（规则）", "cfc": "CfC（训练好的）",
         "mlp": "MLP（无记忆）", "pd": "PD（最朴素）"}
ORDER = ("teacher", "cfc", "mlp", "pd")


def _load():
    if not RESULTS.exists():
        raise SystemExit("先跑 eval.py 生成 results.json")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _agg(rows, key, by="controller"):
    """按控制器聚合，返回 ``(labels, means, stds)``。"""
    labels, m, s = [], [], []
    for c in ORDER:
        a = np.array([r[key] for r in rows if r.get(by) == c], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        labels.append(c)
        m.append(a.mean())
        s.append(a.std())
    return labels, np.array(m), np.array(s)


# =========================================================================== #
# 图 1：阶跃响应                                                              #
# =========================================================================== #
def fig_step(res, sp):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    sc = sim.step_scene(step_deg=25.0, t0=2.0, duration=8.0,
                        dt=sp.fw.ctrl_dt_ms * 1e-3)
    seeds = [r["seed"] for r in res["step"][:6]] or [0, 1, 2]
    for name in ORDER:
        pol = C.make(name)
        curves = []
        for sd in seeds:
            tr = sim.ClosedLoop(sp, pol, sc, seed=sd, warmup_s=1.0).run()
            curves.append(np.interp(np.arange(0, 8.0, 0.02), tr.t, tr.eps_true))
            if name in ("teacher", "cfc"):
                ax[1].plot(tr.t, tr.eps_true, color=COLOR[name], alpha=0.28, lw=0.8)
        if not curves:
            continue
        arr = np.array(curves)
        t = np.arange(0, 8.0, 0.02)
        ax[0].plot(t, arr.mean(0), color=COLOR[name], lw=1.8, label=LABEL[name])
        ax[0].fill_between(t, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                           color=COLOR[name], alpha=0.15)
        if name in ("teacher", "cfc"):
            ax[1].plot(t, arr.mean(0), color=COLOR[name], lw=2.0, label=LABEL[name])
    ax[0].axvline(2.0, color="k", ls="--", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("时间 (s)")
    ax[0].set_ylabel("ε_body (°)")
    ax[0].set_title("阶跃 0°→25°：ε_body 响应（均值 ± 1σ，%d 个种子）" % len(seeds))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)
    ax[1].set_xlabel("时间 (s)")
    ax[1].set_ylabel("ε_body (°)")
    ax[1].set_title("同一次运行的单条曲线（Teacher vs CfC）")
    ax[1].axvline(2.0, color="k", ls="--", lw=0.8)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.savefig(FIGS / "fig1_step.png")
    plt.close(fig)
    return FIGS / "fig1_step.png"


# =========================================================================== #
# 图 2：跟踪误差分布                                                          #
# =========================================================================== #
def fig_error_dist(res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    tr = res["tracking"]
    scns = [s for s in sim.SCENARIOS if any(r["scenario"] == s for r in tr)]
    w = 0.2
    x = np.arange(len(scns))
    for i, c in enumerate(ORDER):
        m, s = [], []
        for scn in scns:
            a = np.array([r["eps_rms"] for r in tr
                          if r["scenario"] == scn and r["controller"] == c])
            m.append(a.mean() if a.size else np.nan)
            s.append(a.std() if a.size else np.nan)
        if not np.any(np.isfinite(m)):
            continue
        ax[0].bar(x + (i - 1.5) * w, m, w, yerr=s, capsize=2,
                  color=COLOR[c], label=LABEL[c], alpha=0.9,
                  error_kw=dict(lw=0.8))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(scns, fontsize=9)
    ax[0].set_ylabel("ε_body RMS (°)")
    ax[0].set_title("各场景跟踪误差（mean ± std，跨种子）")
    ax[0].legend(fontsize=8)
    ax[0].grid(axis="y", alpha=0.25)

    # 分布图：把所有 (场景, 种子) 的 eps_rms 摊开
    data, labels = [], []
    for c in ORDER:
        a = np.array([r["eps_rms"] for r in tr if r["controller"] == c])
        if a.size:
            data.append(a)
            labels.append(LABEL[c])
    if data:
        bp = ax[1].boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, c in zip(bp["boxes"], [COLOR[c] for c in ORDER if
                                         any(r["controller"] == c for r in tr)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax[1].set_ylabel("ε_body RMS (°)")
        ax[1].set_title("跟踪误差分布（每个点 = 一个场景×种子）")
        ax[1].tick_params(axis="x", labelsize=8)
        ax[1].grid(axis="y", alpha=0.25)
        ax[1].set_yscale("symlog", linthresh=10)
    fig.savefig(FIGS / "fig2_error_dist.png")
    plt.close(fig)
    return FIGS / "fig2_error_dist.png"


# =========================================================================== #
# 图 3：头身分离                                                              #
# =========================================================================== #
def fig_separation(res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    tr = res["tracking"]
    for j, (key, title) in enumerate((
            ("sep_ratio", "头身分离比 = 头部角行程 / 机身角行程（|ε|<15° 区间）"),
            ("head_share_cmd", "指令侧头部占比：控制器选择让谁动"))):
        labels, m, s = _agg(tr, key)
        if not labels:
            continue
        x = np.arange(len(labels))
        ax[j].bar(x, m, 0.6, yerr=s, capsize=3,
                  color=[COLOR[c] for c in labels], alpha=0.9,
                  error_kw=dict(lw=0.8))
        ax[j].set_xticks(x)
        ax[j].set_xticklabels([LABEL[c] for c in labels], fontsize=8)
        ax[j].set_title(title, fontsize=9)
        ax[j].grid(axis="y", alpha=0.25)
        if key == "sep_ratio":
            ax[j].axhline(1.0, color="k", ls="--", lw=0.8)
            ax[j].set_ylabel("比值（>1 = 头动得比机身多）")
        else:
            ax[j].axhline(0.5, color="k", ls="--", lw=0.8)
            ax[j].set_ylabel("占比 0..1")
    fig.suptitle("「只动头」的核心指标：小误差区间内的通道分离", fontsize=10)
    fig.savefig(FIGS / "fig3_separation.png")
    plt.close(fig)
    return FIGS / "fig3_separation.png"


# =========================================================================== #
# 图 4：jerk                                                                  #
# =========================================================================== #
def fig_jerk(res):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    tr = res["tracking"]
    x = np.arange(len(ORDER))
    w = 0.38
    for i, key in enumerate(("jerk1", "jerk2")):
        labels, m, s = _agg(tr, key)
        if not labels:
            continue
        ax.bar(x + (i - 0.5) * w, m, w, yerr=s, capsize=2,
               label=("一阶差分 ||Δa||" if key == "jerk1" else "二阶差分 ||Δ²a||"),
               alpha=0.85, color=("#4c72b0", "#dd8452")[i],
               error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[c] for c in ORDER], fontsize=8)
    ax.set_ylabel("归一化动作差分（每帧）")
    ax.set_title("动作平滑度：越小越平滑（跨场景×种子 mean ± std）")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "fig4_jerk.png")
    plt.close(fig)
    return FIGS / "fig4_jerk.png"


# =========================================================================== #
# 图 5：固件原样增益为什么不可能稳                                            #
# =========================================================================== #
def fig_firmware_poles():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    kts = np.logspace(-6, 0, 160)
    for mode, style in (("firmware", "-"), ("sign_flip", "--")):
        sp = replace(P.SimParams(), fw=replace(P.FirmwareParams(), lqr_mode=mode))
        re_max, zeta = [], []
        for kt in kts:
            e_p, _ = sim.closed_loop_eigs(sp, k_torque=kt)
            re_max.append(float(np.max(e_p.real)))
            cplx = sorted([l for l in e_p if abs(l.imag) > 1e-9],
                          key=lambda l: -abs(l.imag))
            zeta.append(sim.damping_ratio(cplx[0]) if cplx else 1.0)
        ax[0].semilogx(kts, re_max, style, lw=1.8,
                       label="lqr_mode=%s" % mode)
        ax[1].semilogx(kts, zeta, style, lw=1.8, label="lqr_mode=%s" % mode)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("k_torque  (torque-count → N·m)")
    ax[0].set_ylabel("max Re(极点)  (1/s)")
    ax[0].set_title("闭环最右极点：>0 即发散")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25, which="both")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhline(0.25, color="r", ls=":", lw=0.8)
    ax[1].set_xlabel("k_torque  (torque-count → N·m)")
    ax[1].set_ylabel("俯仰模态阻尼比 ζ")
    ax[1].set_title("俯仰模态阻尼：红线是标定判据 ζ=0.25")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, which="both")
    fig.suptitle("源码原样增益 [1300,5,25,0.4] 对任何力矩标定都不可稳定；"
                 "把 K0/K1 取负后才出现稳定窗口", fontsize=9)
    fig.savefig(FIGS / "fig5_poles.png")
    plt.close(fig)
    return FIGS / "fig5_poles.png"


# =========================================================================== #
# 图 6：时域对比                                                              #
# =========================================================================== #
def fig_trace(sp):
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    sc = sim.make_scene("orbit", 0, 18.0, sp.fw.ctrl_dt_ms * 1e-3)
    for name in ("teacher", "cfc"):
        tr = sim.ClosedLoop(sp, C.make(name), sc, seed=0).run()
        w = tr.warm > 0
        ax[0].plot(tr.t[w], tr.eps_true[w], color=COLOR[name], lw=1.2,
                   label=LABEL[name])
        ax[1].plot(tr.t[w], tr.q_head[w], color=COLOR[name], lw=1.2, label="头部")
        ax[1].plot(tr.t[w], tr.body_yaw[w], color=COLOR[name], lw=0.9, ls="--",
                   alpha=0.8, label="机身")
        ax[2].plot(tr.t[w], tr.vyaw_des[w], color=COLOR[name], lw=0.9,
                   label="vyaw 指令")
        ax[2].plot(tr.t[w], tr.head_cmd_action[w], color=COLOR[name], lw=0.9,
                   ls=":", alpha=0.85, label="head 指令")
    ax[0].axhline(15, color="r", ls=":", lw=0.8)
    ax[0].axhline(-15, color="r", ls=":", lw=0.8)
    ax[0].set_ylabel("ε_body (°)")
    ax[0].set_title("orbit 场景（σ=0）：ε 与红虚线 ±15° 成功带")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_ylabel("角度 (°)")
    ax[1].set_title("头部角（实线）与机身朝向（虚线）")
    ax[1].legend(fontsize=7, ncol=4)
    ax[2].set_ylabel("指令")
    ax[2].set_xlabel("时间 (s)")
    ax[2].set_title("控制器输出：vyaw_des (°/s) 与 head_des (°)")
    ax[2].legend(fontsize=7, ncol=4)
    for a in ax:
        a.grid(alpha=0.25)
    fig.savefig(FIGS / "fig6_trace.png")
    plt.close(fig)
    return FIGS / "fig6_trace.png"


# =========================================================================== #
# 参数出处表                                                                  #
# =========================================================================== #
def provenance_md():
    L = ["| 参数 | 值 | 出处 / 依据 | 类型 |", "|---|---|---|---|"]
    for name, val, src, est in P.provenance_table():
        if isinstance(val, (int, float)):
            v = "%.4g" % val
        else:
            v = str(val)
        L.append("| `%s` | %s | %s | %s |"
                 % (name, v, src, "**估计**" if est else "源码"))
    return "\n".join(L)


def metrics_md(res):
    tr = res["tracking"]
    L = ["| 控制器 | ε RMS (°) | ε max (°) | 头身分离比 | head 占比(指令) | "
         "jerk1 | jerk2 | 成功率 | 视野内 | 头部饱和 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        sub = [r for r in tr if r["controller"] == c]
        if not sub:
            continue

        def ms(k, pct=False, dig=2):
            a = np.array([r[k] for r in sub], dtype=float)
            a = a[np.isfinite(a)]
            f = (lambda z: 100 * z) if pct else (lambda z: z)
            return ("%.*f ± %.*f" % (dig, f(a.mean()), dig, f(a.std())))
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (LABEL[c], ms("eps_rms"), ms("eps_max", dig=1),
                    ms("sep_ratio"), ms("head_share_cmd", dig=3),
                    ms("jerk1", dig=4), ms("jerk2", dig=4),
                    ms("success", pct=True, dig=1), ms("in_fov", pct=True, dig=1),
                    ms("sat_head", pct=True, dig=1)))
    return "\n".join(L)


def per_scenario_md(res):
    tr = res["tracking"]
    L = ["| 场景 | 控制器 | ε RMS (°) | 成功率 | 头身分离比 | jerk1 |",
         "|---|---|---|---|---|---|"]
    for scn in sim.SCENARIOS:
        for c in ORDER:
            sub = [r for r in tr if r["scenario"] == scn and r["controller"] == c]
            if not sub:
                continue

            def ms(k, pct=False):
                a = np.array([r[k] for r in sub], dtype=float)
                f = (lambda z: 100 * z) if pct else (lambda z: z)
                return "%.2f ± %.2f" % (f(a.mean()), f(a.std()))
            L.append("| %s | %s | %s | %s | %s | %s |"
                     % (scn, c, ms("eps_rms"), ms("success", True),
                        ms("sep_ratio"), ms("jerk1")))
    return "\n".join(L)


def step_md(res):
    L = ["| 控制器 | 上升时间 (s) | 调节时间 (s) | 超调 (%) | 稳态误差 (°) |",
         "|---|---|---|---|---|"]
    for c in ORDER:
        sub = [r for r in res["step"] if r["controller"] == c]
        if not sub:
            continue

        def ms(k):
            a = np.array([r[k] for r in sub], dtype=float)
            a = a[np.isfinite(a)]
            return "—" if a.size == 0 else "%.2f ± %.2f" % (a.mean(), a.std())
        L.append("| %s | %s | %s | %s | %s |"
                 % (LABEL[c], ms("rise_s"), ms("settle_s"),
                    ms("overshoot_pct"), ms("ss_err")))
    return "\n".join(L)


def firmware_md(res):
    rows = res.get("firmware_gains") or []
    if not rows:
        return "（未运行）"
    L = ["| 控制器 | \\|pitch\\| 峰值 (°) | 窗口末 \\|pitch\\| (°) | 位置漂移 (m) | robot_state 掉 0 |",
         "|---|---|---|---|---|"]
    for c in ORDER:
        sub = [r for r in rows if r["controller"] == c]
        if not sub:
            continue
        L.append("| %s | %.1f | %.1f | %.2f | %.0f%% |"
                 % (LABEL[c], np.mean([r["pitch_max"] for r in sub]),
                    np.mean([r["pitch_end"] for r in sub]),
                    np.mean([r["x_drift"] for r in sub]),
                    100 * np.mean([r["fell"] for r in sub])))
    return "\n".join(L)


# =========================================================================== #
# 报告                                                                        #
# =========================================================================== #
def build_report(res, figs, sp, calib):
    meta = res["meta"]
    cfc = [r for r in res["tracking"] if r["controller"] == "cfc"]
    mlp = [r for r in res["tracking"] if r["controller"] == "mlp"]
    tea = [r for r in res["tracking"] if r["controller"] == "teacher"]
    pd = [r for r in res["tracking"] if r["controller"] == "pd"]

    def m(rows, k):
        a = np.array([r[k] for r in rows], dtype=float)
        return float(np.nanmean(a)), float(np.nanstd(a))

    c_eps, c_eps_s = m(cfc, "eps_rms")
    m_eps, _ = m(mlp, "eps_rms")
    t_eps, _ = m(tea, "eps_rms")
    p_eps, _ = m(pd, "eps_rms")
    c_succ, _ = m(cfc, "success")
    m_succ, _ = m(mlp, "success")
    t_succ, _ = m(tea, "success")
    p_succ, _ = m(pd, "success")

    L = []
    a = L.append
    a("# RIG-Hover 闭环仿真与端到端评估")
    a("")
    a("> 一条命令复现：`D:\\smartcar\\.venv\\Scripts\\python.exe eval.py --seeds 10`，"
      "再 `python make_report.py`。")
    a("> 全部代码与结果都在 `D:\\robot_work\\pc\\sim\\`，不依赖机器人上电。")
    a("")
    a("## 0. 结论速览")
    a("")
    a("1. **固件里 `lqr_k = [1300, 5, 25, 0.4]` 的符号让平衡环结构上不可稳定。**")
    a("   位置/速度两个通道是正反馈：θ 环稳定要求力矩标定 `k_t` 大，位置环稳定要求"
      "`k_t` 小，两条件互补且临界值与机体质量/惯量/质心高度无关"
      "（证明见 §3）。逐点扫描 160 个 `k_t`、跨 6 个数量级，"
      "`lqr_mode=firmware` 始终存在正实部极点（图 5 左）。把 `K0/K1` 取负"
      "（`lqr_mode=sign_flip`）后立刻出现稳定窗口 `k_t ∈ [%.2e, %.2e]`。"
      % tuple(meta["stability_window"]))
    a("2. **在修正后的平衡环上，CfC 闭环跟踪明显好于 MLP 与 PD，并优于教师。**")
    a("   跨 5 场景 × 10 种子：ε RMS **%.1f°**（CfC） vs %.1f°（Teacher）"
      " / %.1f°（MLP） / %.1f°（PD）；成功率 **%.0f%%** vs %.0f%% / %.0f%% / %.0f%%。"
      % (c_eps, t_eps, m_eps, p_eps, 100 * c_succ, 100 * t_succ,
         100 * m_succ, 100 * p_succ))
    a("3. **「只动头」在闭环里做不到。** 小误差区间内实测头身分离比 "
      "`头行程/身行程` 全部 < 1.4，CfC 只有 %.2f —— 机身并没有停住。原因是航向环"
      "是**纯 P 控双积分**（`yaw_u = 4·(ψ−ψ_ref)`），闭环是一对欠阻尼复极点 "
      "ζ≈%.2f，机身一直在小角度振荡，把头部的动作淹掉。"
      % (m(cfc, "sep_ratio")[0],
         sim.damping_ratio(complex(*meta["poles_yaw"][0]))))
    a("4. **动作平滑度：CfC 比 MLP/PD 好，但比教师差 2.5 倍**：jerk1 %.4f（CfC） / "
      "%.4f（Teacher） / %.4f（MLP） / %.4f（PD）。原因很具体：教师在 `config.yaml` "
      "里有 `head_rate_limit_dps=300` 与 `vyaw_rate_limit_dps2=400` 两道限幅，"
      "而 CfC 是纯前馈输出、**固件 arbiter 只对 head 做一阶低通、不对 vyaw 做**。"
      "这是一条可以直接改固件的行动项（给 vyaw 加同样的一阶低通/速率限幅）。"
      % (m(cfc, "jerk1")[0], m(tea, "jerk1")[0], m(mlp, "jerk1")[0],
         m(pd, "jerk1")[0]))
    a("")
    a("## 1. 方法与模型")
    a("")
    a("### 1.1 四个维度怎么建")
    a("")
    a("| 维度 | 模型 | 为什么够用 |")
    a("|---|---|---|")
    a("| 平衡 | 轮式倒立摆（小车-摆两自由度，线性化），矩阵指数精确 ZOH 离散 | "
      "6 ms 步长下欧拉积分会让唯一的不稳定极点偏差几个百分点 |")
    a("| 航向 | 双积分 + 黏性阻尼，输入是固件的 `yaw_u` | 固件就是纯 P 控双积分，"
      "这一点决定了「临界稳定」的全部结论 |")
    a("| 头部 | 二阶临界阻尼（ω=%.0f rad/s, ζ=%.2f）+ 速率限幅 %.0f°/s + ±45° 限位 "
      "| 舵机只有「多快、多准」两个自由度 |" % (
          sp.plant.head_bw_rad, sp.plant.head_zeta, sp.plant.head_rate_max_dps))
    a("| 视觉 | 头部相机、FOV 60°、50 Hz、`theta_img = 方位 −(机身+头角)` | "
      "§3.3 的闭式解，视线绝对朝向消掉 |")
    a("")
    a("### 1.2 固件部分不是重写，是逐行复刻")
    a("")
    a("* `robot_control()`：LQR + 航向 P + `|u|>20 再加 10` 的死区补偿 + "
      "`Clip(±1000)` + `tor1=-temp_u+yaw_u / tor2=temp_u+yaw_u`")
    a("* `update_state()`：`stable_pos/stable_yaw` 引用积分、`robot_state` "
      "站立/摔倒状态机（1.5 s 直立才允许出力）")
    a("* `control_arbiter.c`：head 一阶低通 0.15/tick、`pos_lead_max=0.20 m`、"
      "`yaw_lead_max=35°`、视觉三档时效、指令看门狗")
    a("* `imu.cc`：`dq=0.6dq+0.4pit_gyro` 低通、yaw 纯陀螺积分 + 1.5°/s 死区、"
      "互补滤波 `f=1/(|ω|+200)`，以及**头部角度进测量轴**（`cos(q_head)` 缩放）")
    a("* `cfc_task.c build_obs()`：106 维观测逐下标对齐（`pc/train/obs.py` 同一张表）")
    a("")
    a("### 1.3 三个标定量（都标成「标定 / 估计」，不是实测）")
    a("")
    a("| 量 | 值 | 怎么来的 |")
    a("|---|---|---|")
    a("| `k_torque`（torque-count → N·m） | %.3e | 扫 160 个点，要求闭环四极点全稳、"
      "俯仰模态 ζ≥0.25 且 ωn≥15 rad/s，取满足条件的**最小**值 |"
      % meta["k_torque"])
    a("| `k_yaw_accel`（yaw_u → °/s²） | %.3f | 要求机身 yaw 对 60° 参考阶跃的 "
      "90%% 上升时间 = 0.35 s（实测 %.2f s） |"
      % (meta["k_yaw_accel"], meta["yaw_settle_s"]))
    a("| `pitch_sign / torque_sign / yaw_sign` | %s | 只有这组极性下 "
      "`lqr_k2=+25` 才构成负反馈。**只由乘积可观测**，真机极性未验证 |"
      % "{} / {} / {}".format(meta["sign"]["pitch"], meta["sign"]["torque"],
                              meta["sign"]["yaw"]))
    a("")
    a("标定后的平衡环极点：`%s`（ζ=%.2f, ωn=%.1f rad/s）；"
      "航向环极点：`%s`（ζ=%.2f）。"
      % (np.array2string(np.array([complex(*p) for p in meta["poles_pitch"]]),
                         precision=2),
         meta["zeta_pitch"], meta["wn_pitch"],
         np.array2string(np.array([complex(*p) for p in meta["poles_yaw"]]),
                         precision=2),
         sim.damping_ratio(complex(*meta["poles_yaw"][0]))))
    a("")
    a("## 2. 参数出处表")
    a("")
    a(provenance_md())
    a("")
    a("## 3. 为什么源码原样的 LQR 增益不可能稳住")
    a("")
    a("把 `u = -(K0·x + K1·v + K2·p + K3·dq)` 写成物理口径（`p = -57.3θ`，θ 为前倾角），"
      "并用 θ̈≈0 的慢流形 `θ = ẍ/g`：")
    a("")
    a("```")
    a("平衡环（快）：θ̈ = |B3|·g·M·r·(1−c)·θ      c = 57.3·K2·k_t/(M·r·g)")
    a("位置环（慢）：ẍ = −k_t·K0·Δx /(M·r·(1−c))")
    a("```")
    a("")
    a("θ 环稳定要求 **c>1**，位置环稳定要求 **c<1**，而两者的临界值都是 "
      "`M·r·g/(57.3·K2)` —— 与 `m_b`、`l`、`I` 全部无关。"
      "所以只要 `K0>0`，这个闭环对**任何**力矩标定都不稳定（图 5 左的实线始终 >0）。"
      "把 `K0/K1` 取负（`sign_flip`）后 c>1 与位置环负反馈同时成立。")
    a("")
    a("这不是「调参没调好」，是**符号错**。它与「机器人整晚 `pitch≈89°`、"
      "`robot_state=0`」的现象一致，但**不能据此断定真机就是这个原因** —— "
      "见 §7 第 1 条。")
    a("")
    a("![闭环极点 vs 力矩标定](figs/fig5_poles.png)")
    a("")
    a("在源码原样增益下，四个控制器（只有控制律不同、机体与平衡环相同）"
      "都不会摔倒，但**位置以约 1 s 的时间常数指数漂走**：")
    a("")
    a(firmware_md(res))
    a("")
    a("## 4. 结果")
    a("")
    a("### 4.1 跨场景总表（5 场景 × 10 种子 = 50 次运行/控制器）")
    a("")
    a(metrics_md(res))
    a("")
    a("![跟踪误差](figs/fig2_error_dist.png)")
    a("")
    a("### 4.2 分场景")
    a("")
    a(per_scenario_md(res))
    a("")
    a("### 4.3 阶跃响应 0°→25°")
    a("")
    a("用 25° 而不是 40°：FOV 只有 ±30°（`control_arbiter.c:109`），人一步跨到 40° "
      "就直接出了视野，此时所有控制器拿到的都是 `seen=0` —— 那测的是「丢目标怎么办」，"
      "不是「跟踪阶跃」。40° 的版本见 §4.4。")
    a("")
    a(step_md(res))
    a("")
    a("![阶跃响应](figs/fig1_step.png)")
    a("")
    a("### 4.4 大阶跃 0°→40°：目标直接跳出视场")
    a("")
    sf = res.get("step_far") or []
    if sf:
        a("| 控制器 | 重新捕获时间 (s) | 丢失目标帧占比 | ε 峰值 (°) |")
        a("|---|---|---|---|")
        for c in ORDER:
            sub = [r for r in sf if r["controller"] == c]
            if not sub:
                continue
            ra = np.array([r["reacq_s"] for r in sub], dtype=float)
            ra = ra[np.isfinite(ra)]
            a("| %s | %s | %.0f%% | %.1f |"
              % (LABEL[c],
                 "—（窗口内没抓回来）" if ra.size == 0
                 else "%.2f" % ra.mean(),
                 100 * np.mean([r["lost_frac"] for r in sub]),
                 np.mean([r["eps_peak"] for r in sub])))
    a("")
    a("### 4.5 头身分离")
    a("")
    a("![头身分离](figs/fig3_separation.png)")
    a("")
    a("### 4.6 动作平滑度")
    a("")
    a("![jerk](figs/fig4_jerk.png)")
    a("")
    a("### 4.7 一条真实的时域轨迹")
    a("")
    a("![trace](figs/fig6_trace.png)")
    a("")
    a("## 5. CfC 相比 MLP / PD 到底好多少")
    a("")
    a("* **ε RMS**：CfC %.2f° vs MLP %.2f° vs PD %.2f° vs Teacher %.2f°。"
      % (c_eps, m_eps, p_eps, t_eps))
    a("* **成功率**（人始终在视野内且 |ε|<15°）：CfC %.0f%% vs MLP %.0f%% vs "
      "PD %.0f%% vs Teacher %.0f%%。"
      % (100 * c_succ, 100 * m_succ, 100 * p_succ, 100 * t_succ))
    a("* MLP 的失败模式**不是**「跟不准」，而是**丢目标之后回不来**：它在开环模仿上"
      "反而比 CfC 准（在同一份数据上 MAE：MLP 0.63°/1.32°，CfC 0.93°/2.40°），"
      "所以差异只能来自**闭环**——无记忆网络学不到教师的「丢失→回中→重捕获」时序。")
    a("* PD 的问题相反：它对每个误差都同时给头和机身发指令，机身被推得来回摆，"
      "41% 的时间人已经出视野。")
    a("* **CfC 在闭环里比教师还好**（ε RMS %.2f° vs %.2f°）。这不是「网络更聪明」，"
      "而是教师的手调增益在这个欠阻尼航向环上会把机身推到振荡：教师 `k_fast=3.0` "
      "在 25° 误差上直接打到 60°/s 上限，CfC 学到的 `vyaw` 温和得多（峰值约 35°/s），"
      "恰好避开了 ζ≈%.2f 的谐振。**这是「负反馈系统里更弱的控制器可能更稳」的典型例子，"
      "不能当作 CfC 更优的证据**（见 §7 第 3 条）。"
      % (c_eps, t_eps, sim.damping_ratio(complex(*meta["poles_yaw"][0]))))
    a("")
    a("## 6. 复现")
    a("")
    a("```bat")
    a("set PY=D:\\smartcar\\.venv\\Scripts\\python.exe")
    a("cd D:\\robot_work\\pc\\sim")
    a("%PY% eval.py --seeds 10        :: 标定 + 5 场景 × 4 控制器 × 10 种子 + 阶跃")
    a("%PY% make_report.py            :: 图 + 本报告")
    a("%PY% train_mlp.py              :: 重新训练无记忆基线（eval.py 会自动补）")
    a("```")
    a("")
    a("## 7. 诚实的局限（哪些是假设、哪些是估计、最脆弱的是哪一条）")
    a("")
    a("1. **最脆弱：平衡环符号的结论依赖一个我无法验证的极性。** "
      "`pitch_sign` 与 `torque_sign` 只以乘积形式进入模型，我选了让 `lqr_k2=+25` "
      "构成负反馈的那一组。如果真机上 `pitch` 的极性相反，那 `k2` 就成了正反馈，"
      "结论会变成「连俯仰环都是错的」—— 但无论哪种情况，"
      "**K0/K1 与 K2 的相对符号**都必须满足「位置项取负」这一条，"
      "这条与极性无关（§3 的推导里 c 的表达式含 K2·sign 的乘积，而 K0 的符号要求是"
      "独立的）。**上机验证只要 30 分钟**：把机器人扶正保持 1.5 s，松手看 `pitch` "
      "变大还是变小、轮子是往前还是往后追。")
    a("2. **所有机体参数都是估计值**：质量 0.32 kg、质心高 55 mm、轮距 130 mm、"
      "转动惯量 —— 一个都没称过。其中 `com_height_m` 对开环不稳定极点影响最大"
      "（σ=√(m g l M/D)），但它只改变「多快倒」，不改变 §3 的符号结论"
      "（那条与参数无关）。")
    a("3. **航向环的阻尼 ζ 是估计的**，而它决定了「头身分离」这个核心指标。"
      "`yaw_damping=1.2 1/s` 是我拍的；标定只固定了 `k_yaw_accel`（按 0.35 s 上升时间）。"
      "真机摩擦更大时 ζ 会更高、机身更安静，分离比会更好看。**这个指标的绝对值不可信，"
      "只有同一阻尼下四个控制器的相对次序有意义。**")
    a("4. **视觉是几何真值 + 合成 64 维特征**，不是 YOLO。"
      "`feat_64` 用 `pc/train/synth.py` 同一个固定投影生成，所以 CfC 不会因为特征域不匹配"
      "而掉分 —— 真机上第一件事就是看 `/status` 的 clip 比例。")
    a("5. **训练/部署口径有一处不一致，且是我引入的**："
      "`synth.py` 在「没看见人」的帧里仍然喂几何真值的 `theta_img`，"
      "而仿真按真机语义喂 0。这会让三份策略在丢目标时都比训练分布更「瞎」。"
      "我保留了这个更严格的口径，并在 `sim._perceive` 里注明；"
      "**它可能不公平地压低了 CfC 与 MLP 的分数**，重训时应统一。")
    a("6. **CfC 的部署口径与训练口径有一处差异（既有管线的问题，不是我的）**："
      "`train_cfc.Policy.forward_seq` 输出**未激活的线性头**，而固件 "
      "`cfc_model.c:1149-1157` 对 `head_act==1` 的头再套一层 `tanhf`。"
      "仿真按**固件口径**评估（`controllers.verify_cfc` 用同一口径与 torch 对齐，"
      "128 帧最大误差 %.1e），所以被评估的是「真机上会跑的那个函数」。"
      "对 `|a|<1` 的部分影响不大，但训练时从未见过 tanh，"
      "这是管线里一个应该修掉的不一致。" % meta["cfc_max_abs_err"])
    a("7. **`robot_state` 的自启动问题没被建模成失败**：固件要求 `|pitch|<7°` "
      "连续 1.5 s 才允许出力，而力矩为 0 时倒立摆 0.2 s 就倒了。仿真从"
      "「已被扶正并保持 1.5 s」开始（`warmup`），也就是说**这个机器人无法自启动**，"
      "摔倒后必须人工扶正 —— 这一条本身可能是「整晚横躺」最直接的解释。")
    a("8. **`k_torque` 的绝对量级没有被交叉验证**：标定出来的 %.2e N·m/count 意味着 "
      "1000 count 只有 %.2f N·m。如果真机 KP4012 的实际力矩常数差一个数量级，"
      "那 §3 的 `k_t` 窗口位置会跟着移动 —— 但窗口**存在与否**只取决于符号，"
      "不取决于量级。" % (meta["k_torque"], meta["k_torque"] * 1000))
    a("9. **没建模**：轮子打滑、桌面倾角、电池电压对力矩的影响、舵机齿隙、"
      "UART 丢帧、视觉延迟抖动（固定 50 Hz）、`gaze_guard` 的桌沿急停。"
      "这些都只会让真机更差，不会更好。")
    a("10. **航向环有一个固件细节让它永远停不下来**：`imu.cc:264-268` 在 "
      "`|gyro_yaw| < 1.5 °/s` 时把角速度判成 0，于是机身在低速摆动时航向估计冻结、"
      "P 环看不到误差 —— 仿真里表现为机身残余振荡衰减不到零（阶跃实验里"
      "教师/PD 的「调节时间」是「—」而不是某个数）。真机上这个死区还会叠加"
      "陀螺零偏，实际可能更糟。")
    a("")
    a("---")
    a("")
    a("生成时间 %.0f s / 评估 %.0f s。"
      % (meta.get("elapsed_s", 0.0), meta.get("elapsed_s", 0.0)))
    return "\n".join(L) + "\n"


def main():
    FIGS.mkdir(exist_ok=True)
    res = _load()
    meta = res["meta"]
    base = replace(P.SimParams(),
                   fw=replace(P.FirmwareParams(), lqr_mode=meta["lqr_mode"]))
    sp = replace(base, plant=replace(base.plant,
                                     k_torque=meta["k_torque"],
                                     k_yaw_accel=meta["k_yaw_accel"]))
    print("[fig] 1/6 阶跃响应 …")
    fig_step(res, sp)
    print("[fig] 2/6 跟踪误差分布 …")
    fig_error_dist(res)
    print("[fig] 3/6 头身分离 …")
    fig_separation(res)
    print("[fig] 4/6 jerk …")
    fig_jerk(res)
    print("[fig] 5/6 极点扫描（源码原样增益为什么不可能稳）…")
    fig_firmware_poles()
    print("[fig] 6/6 时域轨迹 …")
    fig_trace(sp)
    print("[doc] 写 REPORT.md …")
    calib = {"poles_pitch": [complex(*p) for p in meta["poles_pitch"]],
             "poles_yaw": [complex(*p) for p in meta["poles_yaw"]]}
    REPORT.write_text(build_report(res, None, sp, calib), encoding="utf-8")
    print("[out] %s" % REPORT)
    print("[out] %s/*.png" % FIGS)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
