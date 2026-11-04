# -*- coding: utf-8 -*-
"""eval.py — 一条命令跑完全部场景 × 全部控制器 × 全部随机种子。

    python eval.py                 # 5 场景 × 4 控制器 × 10 seed + 阶跃实验
    python eval.py --quick         # 3 seed（冒烟用）
    python eval.py --seeds 20

产出：

* ``results.json``       逐次运行的原始指标（后面 make_report.py 只读它）
* ``results_table.md``   报告用的表格（mean ± std）
* 终端打印三张表：跟踪 / 阶跃 / 固件原样增益下的失败

指标定义（全部在 ``--warmup`` 之后统计，单位见括号）：

===================== ==================================================
``eps_rms``           ε_body = 人绝对方位 − 机身朝向 的均方根（度）
``eps_max``           ε_body 最大绝对值（度）
``sep_ratio``         小误差区（|ε|<15°）内 头部角行程 / 机身角行程
``head_share``        同上，头部占 (头+身) 的比例 0..1
``jerk1``             归一化动作的一阶差分均值 ||Δa||
``jerk2``             归一化动作的二阶差分均值 ||Δ²a||
``sat_head``          |head_des| ≥ 45° 的帧比例（固件 arbiter 硬限位）
``sat_vyaw``          |vyaw_des| ≥ 90°/s 的帧比例
``success``           人始终在视野内 且 |ε|<15° 的帧比例
``in_fov``            人在视野内的帧比例
``fell``              窗口内 robot_state 是否掉到 0
===================== ==================================================

所有跨种子结果都报 **mean ± std**，不报单点值。
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
import params as P               # noqa: E402
import sim                       # noqa: E402

RESULTS = HERE / "results.json"
TABLE = HERE / "results_table.md"

#: 每个场景的时长（秒）。mixed 是串联场景，所以要长。
DURATION = {"orbit": 18.0, "approach": 16.0, "fov_exit": 16.0,
            "wave": 12.0, "mixed": 34.0}

#: 小误差区（度）：只有在这个区间里谈"头身分离"才有意义
BAND_DEG = 15.0
#: 成功判据（度）
SUCCESS_DEG = 15.0
#: 动作归一化尺度（固件满量程）
A_SCALE = np.array([45.0, 90.0, 0.30])


# =========================================================================== #
# 单次运行的指标                                                              #
# =========================================================================== #
def run_metrics(tr: sim.Trace, band=BAND_DEG):
    """从一条轨迹里算出全部指标。

    Args:
        tr: :class:`sim.Trace`。
        band: 小误差区的界（度）。

    Returns:
        dict（键见模块文档）。
    """
    w = tr.warm > 0
    if not np.any(w):
        raise ValueError("warmup 覆盖了整条轨迹，检查 duration/warmup 设置")
    eps = tr.eps_true[w]
    abs_eps = np.abs(eps)

    # ---- 头身分离：只在小误差区统计 ----
    small = abs_eps < band
    if np.count_nonzero(small) > 4:
        qh = tr.q_head[w][small]
        by = tr.body_yaw[w][small]
        head_work = float(np.sum(np.abs(np.diff(qh))))
        body_work = float(np.sum(np.abs(np.diff(np.unwrap(np.radians(by)))))) * 180 / np.pi
        # 指令侧的分离：控制器"选择"让谁动（不受机身振荡污染）
        cmd_body = float(np.sum(np.abs(tr.vyaw_des[w][small]))) * 0.006
        cmd_head = float(np.sum(np.abs(tr.head_cmd_action[w][small]))) * 0.006
    else:
        head_work = body_work = 0.0
        cmd_body = cmd_head = 0.0
    sep_ratio = head_work / (body_work + 0.5)
    head_share = head_work / (head_work + body_work + 1e-9)
    # 指令侧的"头部占比"：控制器选择让谁动。用占比而不是比值，避免除零爆炸。
    head_share_cmd = cmd_head / (cmd_head + cmd_body + 1e-9)

    # ---- 动作平滑度：归一化动作的一阶/二阶差分 ----
    act = np.stack([tr.head_cmd_action[w], tr.vyaw_cmd_action[w],
                    tr.vx_cmd_action[w]], axis=1) / A_SCALE
    d1 = np.linalg.norm(np.diff(act, axis=0), axis=1)
    d2 = np.linalg.norm(np.diff(act, n=2, axis=0), axis=1)

    in_fov = tr.in_fov[w] > 0.5
    ok = in_fov & (abs_eps < SUCCESS_DEG)
    return {
        "eps_rms": float(np.sqrt(np.mean(eps ** 2))),
        "eps_max": float(np.max(abs_eps)),
        "eps_med": float(np.median(abs_eps)),
        "sep_ratio": float(sep_ratio),
        "head_share": float(head_share),
        "head_share_cmd": float(head_share_cmd),
        "head_work_deg": head_work,
        "body_work_deg": body_work,
        "jerk1": float(np.mean(d1)) if d1.size else 0.0,
        "jerk2": float(np.mean(d2)) if d2.size else 0.0,
        "sat_head": float(np.mean(np.abs(tr.head_cmd_action[w]) >= 44.9)),
        "sat_vyaw": float(np.mean(np.abs(tr.vyaw_cmd_action[w]) >= 89.9)),
        "success": float(np.mean(ok)),
        "in_fov": float(np.mean(in_fov)),
        "seen": float(np.mean(tr.seen[w] > 0.5)),
        "fell": bool(tr.fell),
        "pitch_max": float(np.max(np.abs(tr.pitch[w]))),
        "x_drift": float(abs(np.mean(tr.x_w[w][-200:]) - np.mean(tr.x_w[w][:200]))),
        "robot_off_frac": float(np.mean(tr.robot_state[w] < 0.5)),
        "body_rate_abs": float(np.mean(np.abs(tr.body_rate[w]))),
        "head_rate_abs": float(np.mean(np.abs(
            np.gradient(tr.q_head[w], tr.t[w])))),
        "band_frac": float(np.mean(small)),
    }


# =========================================================================== #
# 批量                                                                        #
# =========================================================================== #
def eval_tracking(sp, seeds, controllers, verbose=True):
    """跑 5 个场景 × 控制器 × 种子。

    Args:
        sp: 已标定的 :class:`sim.SimParams`。
        seeds: 种子列表。
        controllers: 控制器名列表。
        verbose: 打印进度。

    Returns:
        ``list[dict]``，每项一个 (scenario, controller, seed)。
    """
    rows = []
    t0 = time.perf_counter()
    dt = sp.fw.ctrl_dt_ms * 1e-3
    for scn in sim.SCENARIOS:
        dur = DURATION[scn]
        for name in controllers:
            pol = C.make(name)
            for sd in seeds:
                scene = sim.make_scene(scn, sd, dur, dt)
                tr = sim.ClosedLoop(sp, pol, scene, seed=sd).run()
                m = run_metrics(tr)
                m.update(scenario=scn, controller=name, seed=int(sd),
                         duration=dur)
                rows.append(m)
            if verbose:
                sub = [r for r in rows if r["scenario"] == scn
                       and r["controller"] == name]
                print("  [%-8s %-8s] eps_rms=%5.2f ± %.2f  success=%.2f  "
                      "sep=%.2f  (%.0fs)"
                      % (scn, name, np.mean([r["eps_rms"] for r in sub]),
                         np.std([r["eps_rms"] for r in sub]),
                         np.mean([r["success"] for r in sub]),
                         np.mean([r["sep_ratio"] for r in sub]),
                         time.perf_counter() - t0))
    return rows


def eval_step(sp, seeds, controllers, verbose=True):
    """0° → 40° 阶跃响应。

    Args:
        sp: 已标定的仿真参数。
        seeds: 种子列表。
        controllers: 控制器名列表。
        verbose: 打印进度。

    Returns:
        ``list[dict]``。
    """
    rows = []
    for name in controllers:
        pol = C.make(name)
        for sd in seeds:
            sc = sim.step_scene(step_deg=25.0, t0=2.0, duration=9.0,
                                dt=sp.fw.ctrl_dt_ms * 1e-3)
            tr = sim.ClosedLoop(sp, pol, sc, seed=sd, warmup_s=1.0).run()
            m = sim.step_metrics(tr, step_deg=25.0, t0=2.0)
            m.update(controller=name, seed=int(sd))
            rows.append(m)
        if verbose:
            sub = [r for r in rows if r["controller"] == name]
            print("  [step25 %-8s] rise=%s settle=%s overshoot=%s"
                  % (name,
                     _fmt_missing(np.nanmean([r["rise_s"] for r in sub])),
                     _fmt_missing(np.nanmean([r["settle_s"] for r in sub])),
                     "%.1f%%" % np.nanmean([r["overshoot_pct"] for r in sub])))
    return rows


def eval_step_far(sp, seeds, controllers, step_deg=40.0, verbose=True):
    """>FOV 半角的大阶跃：人一步跨到视野外，量"多久重新抓住"。

    Returns:
        ``list[dict]``
    """
    rows = []
    for name in controllers:
        pol = C.make(name)
        for sd in seeds:
            sc = sim.step_scene(step_deg=step_deg, t0=2.0, duration=9.0,
                                dt=sp.fw.ctrl_dt_ms * 1e-3)
            tr = sim.ClosedLoop(sp, pol, sc, seed=sd, warmup_s=1.0).run()
            m = sim.reacquire_metrics(tr, t0=2.0)
            m.update(controller=name, seed=int(sd), step_deg=step_deg)
            rows.append(m)
        if verbose:
            sub = [r for r in rows if r["controller"] == name]
            print("  [step%.0f %-8s] 重新捕获=%s s  丢失占比=%.0f%%"
                  % (step_deg, name,
                     _fmt_missing(np.nanmean([r["reacq_s"] for r in sub])),
                     100 * np.nanmean([r["lost_frac"] for r in sub])))
    return rows


def eval_firmware_gains(seeds, controllers, verbose=True):
    """**源码原样**增益下的闭环：证明发散与控制器无关。

    对每个控制器跑 orbit 场景，记录 |pitch| 峰值、robot_state 掉 0 的时间。

    Returns:
        ``list[dict]``
    """
    rows = []
    base = P.SimParams()
    sp = replace(base, fw=replace(base.fw, lqr_mode="firmware"))
    # 力矩标定沿用 sign_flip 的标定值（同一台机器同一个力矩常数）
    kt, ky, _ = sim.calibrate(P.SimParams(), verbose=False)
    sp = replace(sp, plant=replace(sp.plant, k_torque=kt, k_yaw_accel=ky))
    dt = sp.fw.ctrl_dt_ms * 1e-3
    for name in controllers:
        pol = C.make(name)
        for sd in seeds:
            sc = sim.make_scene("orbit", sd, 20.0, dt)
            tr = sim.ClosedLoop(sp, pol, sc, seed=sd).run()
            w = tr.warm > 0
            off = tr.robot_state[w] < 0.5
            t_fall = float(tr.t[w][np.argmax(off)]) if np.any(off) else float("nan")
            x = tr.x_w[w]
            rows.append({
                "controller": name, "seed": int(sd),
                "pitch_max": float(np.max(np.abs(tr.pitch[w]))),
                "pitch_end": float(np.mean(np.abs(tr.pitch[w][-200:]))),
                "fell": bool(np.any(off)),
                "t_fall": t_fall,
                "x_drift": float(abs(np.mean(x[-200:]) - np.mean(x[:200]))),
                "eps_rms": float(np.sqrt(np.mean(tr.eps_true[w] ** 2))),
            })
        if verbose:
            sub = [r for r in rows if r["controller"] == name]
            print("  [firmware-gains %-8s] |pitch| peak %5.1f°  末端 |pitch| %5.1f°  "
                  "位置漂移 %5.2f m  robot_state=0: %.0f%%"
                  % (name, np.mean([r["pitch_max"] for r in sub]),
                     np.mean([r["pitch_end"] for r in sub]),
                     np.mean([r["x_drift"] for r in sub]),
                     100.0 * np.mean([r["fell"] for r in sub])))
    return rows


# =========================================================================== #
# 汇总                                                                        #
# =========================================================================== #
def _fmt_missing(v):
    return "—" if v is None or not np.isfinite(v) else "%.2f" % v


def agg(rows, key):
    """按控制器聚合一个指标，返回 ``(mean, std)``（忽略 NaN）。"""
    a = np.array([r[key] for r in rows], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(a)), float(np.std(a))


def aggregate(rows, keys, by=("scenario", "controller")):
    """把逐次结果聚合成 ``{key: {指标: (mean, std)}}``。"""
    out = {}
    groups = sorted({tuple(r[k] for k in by) for r in rows})
    for g in groups:
        sub = [r for r in rows if tuple(r[k] for k in by) == g]
        out[g] = {kk: agg(sub, kk) for kk in keys}
        out[g]["n"] = len(sub)
    return out


TABLE_KEYS = ("eps_rms", "eps_max", "sep_ratio", "head_share_cmd", "head_share",
              "jerk1", "jerk2", "sat_head", "sat_vyaw", "success", "in_fov",
              "pitch_max", "body_rate_abs", "head_rate_abs")

TABLE_HEAD = {
    "eps_rms": "ε RMS (°)", "eps_max": "ε max (°)",
    "sep_ratio": "头身分离比（实测）", "head_share_cmd": "头部占比（指令）",
    "head_share": "头部占比（实测）", "jerk1": "jerk1", "jerk2": "jerk2",
    "sat_head": "头部饱和", "sat_vyaw": "vyaw 饱和", "success": "成功率",
    "in_fov": "视野内占比", "pitch_max": "|pitch| max (°)",
    "body_rate_abs": "机身平均|ω| (°/s)", "head_rate_abs": "头部平均|ω| (°/s)",
}


def write_table_md(track_rows, step_rows, step_far_rows, fw_rows, path):
    """把四组结果写成 markdown（REPORT 直接引）。

    Args:
        track_rows: ``eval_tracking`` 的输出。
        step_rows: ``eval_step`` 的输出。
        step_far_rows: ``eval_step_far`` 的输出。
        fw_rows: ``eval_firmware_gains`` 的输出。
        path: 输出路径。
    """
    L = []
    L.append("# 闭环评估结果（mean ± std，跨随机种子）\n")
    agg_tr = aggregate(track_rows, TABLE_KEYS)
    ctrls = sorted({r["controller"] for r in track_rows})
    for scn in sim.SCENARIOS:
        L.append("## 场景 `%s` —— %s\n" % (scn, sim.SCENARIO_ZH[scn]))
        L.append("| 控制器 | " + " | ".join(TABLE_HEAD[k] for k in TABLE_KEYS) + " |")
        L.append("|" + "---|" * (len(TABLE_KEYS) + 1))
        for c in ctrls:
            g = agg_tr.get((scn, c))
            if not g:
                continue
            cells = []
            for k in TABLE_KEYS:
                m, s = g[k]
                cells.append(_cell(k, m, s))
            L.append("| %s | %s |" % (c, " | ".join(cells)))
        L.append("")
    L.append("## 阶跃响应 0° → 25°（视野内，±30° FOV）\n")
    L.append("| 控制器 | 上升时间 (s) | 调节时间 (s) | 超调 (%) | 稳态误差 (°) | 峰值误差 (°) |")
    L.append("|---|---|---|---|---|---|")
    for c in ctrls:
        sub = [r for r in step_rows if r["controller"] == c]
        if not sub:
            continue
        cells = []
        for k in ("rise_s", "settle_s", "overshoot_pct", "ss_err", "peak_err"):
            m, s = agg(sub, k)
            cells.append("—" if not np.isfinite(m) else
                         ("%.2f ± %.2f" % (m, s) if k == "overshoot_pct"
                          else "%.3f ± %.3f" % (m, s)))
        L.append("| %s | %s |" % (c, " | ".join(cells)))
    L.append("")
    if step_far_rows:
        L.append("## 大阶跃 0° → 40°（**目标直接跳出 60° 视场**）\n")
        L.append("| 控制器 | 重新捕获时间 (s) | 丢失目标帧占比 | ε 峰值 (°) |")
        L.append("|---|---|---|---|")
        for c in ctrls:
            sub = [r for r in step_far_rows if r["controller"] == c]
            if not sub:
                continue
            m, s = agg(sub, "reacq_s")
            L.append("| %s | %s | %.0f%% | %.1f |"
                     % (c, "—" if not np.isfinite(m) else "%.2f ± %.2f" % (m, s),
                        100 * np.mean([r["lost_frac"] for r in sub]),
                        np.mean([r["eps_peak"] for r in sub])))
        L.append("")
    L.append("## 源码原样 LQR 增益（`lqr_mode=firmware`）下的闭环\n")
    L.append("| 控制器 | \\|pitch\\| 峰值 (°) | 窗口末 \\|pitch\\| (°) | 位置漂移 (m) | robot_state 掉 0 |")
    L.append("|---|---|---|---|---|")
    for c in ctrls:
        sub = [r for r in fw_rows if r["controller"] == c]
        if not sub:
            continue
        L.append("| %s | %.1f | %.1f | %.2f | %.0f%% |"
                 % (c, np.mean([r["pitch_max"] for r in sub]),
                    np.mean([r["pitch_end"] for r in sub]),
                    np.mean([r["x_drift"] for r in sub]),
                    100.0 * np.mean([r["fell"] for r in sub])))
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")
    return path


def _cell(key, m, s):
    if not np.isfinite(m):
        return "—"
    if key in ("success", "in_fov", "sat_head", "sat_vyaw"):
        return "%.1f%% ± %.1f" % (100 * m, 100 * s)
    if key in ("sep_ratio",):
        return "%.2f ± %.2f" % (m, s)
    if key in ("head_share",):
        return "%.3f ± %.3f" % (m, s)
    if key in ("jerk1", "jerk2"):
        return "%.4f ± %.4f" % (m, s)
    return "%.2f ± %.2f" % (m, s)


# =========================================================================== #
# main                                                                        #
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="闭环评估：5 场景 × 4 控制器 × N 种子 + 阶跃 + 固件原样增益对照",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="每个组合的随机种子数")
    ap.add_argument("--controllers", default="teacher,cfc,mlp,pd")
    ap.add_argument("--lqr-mode", default="sign_flip",
                    choices=("sign_flip", "firmware", "care"),
                    help="平衡环 LQR 增益口径（见 params.FirmwareParams.gains）")
    ap.add_argument("--quick", action="store_true", help="3 个种子的快速冒烟")
    ap.add_argument("--no-step", action="store_true", help="跳过阶跃实验")
    ap.add_argument("--no-firmware-demo", action="store_true",
                    help="跳过源码原样增益的失败复现")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds = min(args.seeds, 3)
    names = [s.strip() for s in args.controllers.split(",") if s.strip()]

    if not (HERE / "mlp_policy.npz").exists():
        print("[setup] mlp_policy.npz 不存在，先训练无记忆基线 …")
        import train_mlp
        train_mlp.main([])

    t_start = time.perf_counter()
    base = replace(P.SimParams(),
                   fw=replace(P.FirmwareParams(), lqr_mode=args.lqr_mode))
    kt, ky, calib = sim.calibrate(base, verbose=True)
    sp = replace(base, plant=replace(base.plant, k_torque=kt, k_yaw_accel=ky))

    print("\n[verify] numpy CfC vs torch（部署口径，判据 1e-5）")
    err = C.verify_cfc()
    print("         max_abs_err = %.3e  %s" % (err, "PASS" if err < 1e-5 else "FAIL"))
    if err >= 1e-5:
        raise SystemExit("CfC 前向与 torch 不一致，停止评估")

    seeds = list(range(args.seeds))
    print("\n[trace] 跟踪场景：%d 场景 × %d 控制器 × %d seed"
          % (len(sim.SCENARIOS), len(names), len(seeds)))
    track = eval_tracking(sp, seeds, names)

    step = [] if args.no_step else eval_step(sp, seeds, names)
    step_far = [] if args.no_step else eval_step_far(sp, seeds[:max(3, args.seeds // 3)],
                                                     names)
    fw = ([] if args.no_firmware_demo
          else eval_firmware_gains(seeds[:max(3, args.seeds // 3)], names))

    payload = {
        "meta": {
            "lqr_mode": args.lqr_mode, "seeds": args.seeds,
            "controllers": names, "band_deg": BAND_DEG,
            "success_deg": SUCCESS_DEG, "duration": DURATION,
            "cfc_max_abs_err": err,
            "k_torque": kt, "k_yaw_accel": ky,
            "poles_pitch": [[p.real, p.imag] for p in calib["poles_pitch"]],
            "poles_yaw": [[p.real, p.imag] for p in calib["poles_yaw"]],
            "zeta_pitch": calib["zeta_pitch"], "wn_pitch": calib["wn_pitch"],
            "yaw_settle_s": calib["yaw_settle_s"],
            "stability_window": list(calib["stability_window"]),
            "elapsed_s": time.perf_counter() - t_start,
            "plant": {k: getattr(sp.plant, k) for k in
                      ("wheel_radius_m", "wheel_track_m", "mass_body_kg",
                       "mass_wheel_kg", "com_height_m", "k_torque",
                       "k_yaw_accel", "head_bw_rad")},
            "sign": {"pitch": sp.sign.pitch_sign, "torque": sp.sign.torque_sign,
                     "yaw": sp.sign.yaw_sign},
            "gains": list(sp.fw.gains()),
        },
        "tracking": track, "step": step, "step_far": step_far,
        "firmware_gains": fw,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    write_table_md(track, step, step_far, fw, TABLE)

    # ---- 终端汇总 ----
    print("\n" + "=" * 96)
    print("总表（跨 %d seed，mean ± std）  lqr_mode=%s" % (len(seeds), args.lqr_mode))
    print("=" * 96)
    agg_tr = aggregate(track, ("eps_rms", "eps_max", "sep_ratio", "head_share_cmd",
                               "jerk1", "success", "sat_head", "sat_vyaw"))
    print("%-9s %-8s %-16s %-16s %-16s %-16s %-9s %-9s"
          % ("场景", "控制器", "ε RMS(°)", "ε max(°)", "头身分离(实测)",
             "头部占比(指令)", "成功率", "头饱和"))
    for scn in sim.SCENARIOS:
        for c in names:
            g = agg_tr.get((scn, c))
            if not g:
                continue
            print("%-9s %-8s %-16s %-16s %-16s %-16s %-9s %-9s"
                  % (scn, c,
                     "%.2f ± %.2f" % g["eps_rms"],
                     "%.1f ± %.1f" % g["eps_max"],
                     "%.2f ± %.2f" % g["sep_ratio"],
                     "%.3f ± %.3f" % g["head_share_cmd"],
                     "%.1f%%" % (100 * g["success"][0]),
                     "%.1f%%" % (100 * g["sat_head"][0])))
    print("\n[out] %s\n[out] %s\n[done] %.0f s"
          % (args.out, TABLE, time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
