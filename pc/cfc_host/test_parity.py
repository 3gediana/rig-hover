#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_parity.py -- CfC 三方对拍（ncps 真值 <-> numpy <-> C），一条命令跑完。

    python D:\\robot_work\\pc\\cfc_host\\test_parity.py

流程:
    生成权重/输入 -> ncps 前向 -> numpy 前向 -> 写 testdata.bin
    -> cl.exe 编 test_engine.c + cfc_engine.c -> 跑 -> 读 out_c.bin -> 三方比对
    -> 打印每项最大误差 + PASS/FAIL

覆盖:
    [0] 编译                cl.exe 绝对路径，/O2 /fp:precise /W3，/W3 零告警
    [1] ncps <-> numpy      4 组维度 + ts=0.25/3.0，连续 200 步，判据 < 1e-5
    [2] numpy <-> C         同上（顺带 ncps <-> C 直比）
    [2b] AutoNCP 形态       无 backbone + 稀疏掩码折叠（W_export = W*|mask^T|）
    [2c] AutoNCP 多层        wiring 拆成 3 层 -> C 侧链 3 个 cfc_cell_t
    [3] 保留门              ret=-3（契约初值）/ 随机 / +8，三方对拍
    [4] lambda=0 逐位一致    启用保留门但 ret=-1e4（sigmoid 精确为 0）与不启用
                           的 out_c.bin **逐字节**相同
    [5] 边界输入            全 0 / 全 1 / ±10 饱和，各 200 步
    [6] 误差累积曲线        第 1/10/50/100/200 步的误差 + 趋势
    [7] C 单步耗时          100000 步 x 5 取最优，折算 ESP32-S3
    [8] 引擎自检            test_engine.exe --selftest
"""

import hashlib
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import cfc_ref
import test_parity_ncps as TP

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")  # Windows 控制台 GBK：别让罕见符号打断验收
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
TOL = 1e-5
N_STEPS = 200
BENCH_STEPS = 100000
ESP32_SLOWDOWN = 4.0  # Lead 口径：ESP32-S3 约 1/4 单核性能

CL_DIR = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
             r"\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")
CL_EXE = CL_DIR / "cl.exe"
VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
              r"\VC\Auxiliary\Build\vcvars64.bat")
EXE = HERE / "test_engine.exe"

MAGIC_IN = b"CFCBIN01"
MAGIC_OUT = b"CFCOUT01"

RESULTS = []  # (name, ok, detail)
_ENV_CACHE = {}


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# MSVC 环境（cl.exe 需要 INCLUDE/LIB，默认 shell 里为空）
# --------------------------------------------------------------------------- #
def msvc_env():
    if "env" in _ENV_CACHE:
        return _ENV_CACHE["env"]
    env = dict(os.environ)
    if not CL_EXE.exists():
        raise RuntimeError(f"找不到 cl.exe: {CL_EXE}")
    if not (env.get("INCLUDE") and env.get("LIB")):
        if not VCVARS.exists():
            raise RuntimeError(f"找不到 vcvars64.bat: {VCVARS}")
        # 用批处理文件而不是 cmd /c 一串带引号的命令：避免 cmd 的引号剥离规则把
        # 带空格的 vcvars 路径搞坏（subprocess 会再包一层引号）。
        bat = HERE / "_vcvars_env.bat"
        bat.write_text(f'@echo off\r\ncall "{VCVARS}" >nul 2>&1\r\nset\r\n', encoding="ascii")
        try:
            out = subprocess.run(["cmd.exe", "/c", bat.name], capture_output=True, text=True,
                                 encoding="mbcs", errors="replace", cwd=str(HERE))
        finally:
            try:
                bat.unlink()
            except OSError:
                pass
        if out.returncode != 0 or "INCLUDE=" not in out.stdout:
            raise RuntimeError(f"vcvars64.bat 失败 rc={out.returncode}:\n{out.stdout}\n{out.stderr}")
        parsed = {}
        for line in out.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                k, v = line.split("=", 1)
                parsed[k.upper()] = v
        env.update(parsed)
        env["PATH"] = parsed.get("PATH", env.get("PATH", "")) + ";" + str(CL_DIR)
    _ENV_CACHE["env"] = env
    return env


def run_cmd(args, timeout=600):
    """跑一条命令（list 形式，不用 shell），返回 stdout+stderr；非零退出抛异常。"""
    p = subprocess.run([str(a) for a in args], cwd=str(HERE), env=msvc_env(),
                       capture_output=True, text=True,
                       encoding="mbcs" if os.name == "nt" else "utf-8",
                       errors="replace", timeout=timeout)
    text = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"命令失败 rc={p.returncode}: {args}\n{text}")
    return text


def build():
    print("[0] 编译（cl.exe 绝对路径；/O2 /fp:precise /W3，绝不用 /fp:fast）")
    print(f"    cl = {CL_EXE}")
    t0 = time.perf_counter()
    warnings, objs = [], []
    for src in ("cfc_engine.c", "test_engine.c"):
        out = run_cmd([CL_EXE, "/nologo", "/c", "/O2", "/fp:precise", "/W3", src])
        warnings += [ln for ln in out.splitlines() if "warning" in ln.lower()]
        objs.append(Path(src).with_suffix(".obj").name)
        print(f"    {src:<14} -> {objs[-1]}   (编译器输出: {out.strip() or '<无>'})")
    out = run_cmd([CL_EXE, "/nologo", "/O2", "/fp:precise", "/W3", *objs,
                   f"/Fe:{EXE.name}"])
    warnings += [ln for ln in out.splitlines() if "warning" in ln.lower()]
    dt = time.perf_counter() - t0
    print(f"    链接 test_engine.exe  ({dt:.2f}s)")
    for w in warnings:
        print(f"    ! {w}")
    check("C 编译 /W3 零告警", not warnings, f"{len(warnings)} 条")
    if not EXE.exists():
        raise RuntimeError(f"没有生成 {EXE}")


# --------------------------------------------------------------------------- #
# testdata.bin / out_c.bin
# --------------------------------------------------------------------------- #
def _f32(a, shape):
    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.size != int(np.prod(shape)):
        raise RuntimeError(f"期望形状 {shape}，实际 {a.shape}")
    return a.reshape(shape)


def write_testdata(path, params, inputs, h0, use_retention, ret=None):
    """按 test_engine.c 文档的布局落盘（小端，float32 连续）。"""
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    n_steps, in_dim = inputs.shape
    hid = params["ff1_b"].shape[0]
    bu = 0 if params.get("bb_w") is None else params["bb_w"].shape[0]
    zu = bu if bu > 0 else in_dim + hid

    blobs = []
    if bu > 0:
        blobs.append(_f32(params["bb_w"], (bu, in_dim + hid)))
        blobs.append(_f32(params["bb_b"], (bu,)))
    for name in ("ff1", "ff2", "ta", "tb"):
        blobs.append(_f32(params[name + "_w"], (hid, zu)))
        blobs.append(_f32(params[name + "_b"], (hid,)))
    blobs.append(np.zeros(hid, np.float32) if ret is None else _f32(ret, (hid,)))
    blobs.append(_f32(h0, (hid,)))
    blobs.append(inputs)

    header = MAGIC_IN + struct.pack("<5i", n_steps, in_dim, hid, bu, int(bool(use_retention)))
    header += struct.pack("<f", np.float32(params.get("ts", 1.0)))
    Path(path).write_bytes(header + b"".join(b.tobytes() for b in blobs))


def read_out(path):
    raw = Path(path).read_bytes()
    if len(raw) < 16 or raw[:8] != MAGIC_OUT:
        raise RuntimeError(f"{path} 魔数不对")
    n_steps, hid = struct.unpack("<2i", raw[8:16])
    arr = np.frombuffer(raw[16:], dtype="<f4")
    if arr.size != n_steps * hid:
        raise RuntimeError(f"{path} 长度不符: {arr.size} != {n_steps * hid}")
    return arr.reshape(n_steps, hid).astype(np.float32), raw


def run_engine(data_path, out_path, bench=0):
    cmd = [EXE, data_path, out_path]
    if bench:
        cmd += ["--bench", str(bench)]
    text = run_cmd(cmd)
    info = dict(ln.split("=", 1) for ln in text.splitlines() if "=" in ln)
    info = {k.strip(): v.strip() for k, v in info.items()}
    if info.get("VALIDATE") != "0":
        raise RuntimeError(f"cfc_validate 返回 {info.get('VALIDATE')}\n{text}")
    hs, raw = read_out(out_path)
    return hs, raw, info


def c_run(tag, params, inputs, h0, use_retention, ret=None, bench=0):
    """跑一次 C 引擎，返回 (hs, raw_bytes, info)。tag='main' 时写 testdata.bin/out_c.bin。"""
    if tag == "main":
        data, out = HERE / "testdata.bin", HERE / "out_c.bin"
    else:
        data, out = HERE / f"_data_{tag}.bin", HERE / f"_out_{tag}.bin"
    write_testdata(data, params, inputs, h0, use_retention, ret)
    return run_engine(data, out, bench=bench)


def maxerr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def per_step_err(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ok_row = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    e = np.max(np.abs(a - b), axis=1)
    e[~ok_row] = np.inf
    return e


def fingerprint(a):
    return hashlib.sha1(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()[:8]


def np_vs_c(name, params, inputs, h0, use_retention, ret=None, torch_ref=None, tag=None):
    """numpy <-> C（给了 torch_ref 时顺带 ncps<->C）。返回 (err, hs_np, hs_c, curve, info)。"""
    hs_np, _ = cfc_ref.cfc_forward_seq(params, inputs, h0=h0)
    tag = tag or re.sub(r"[^0-9a-zA-Z]+", "_", name)
    hs_c, _, info = c_run(tag, params, inputs, h0, use_retention, ret=ret)
    if hs_np.shape != hs_c.shape:
        raise RuntimeError(f"{name}: numpy {hs_np.shape} 与 C {hs_c.shape} 形状不符")
    e = maxerr(hs_np, hs_c)
    detail = f"np<->C {e:.3e}"
    if torch_ref is not None:
        detail = f"ncps<->C {maxerr(torch_ref, hs_c):.3e}  " + detail
    check(name, e < TOL, detail)
    return e, hs_np, hs_c, per_step_err(hs_np, hs_c), info


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("CfC 三方对拍：ncps(真值) <-> numpy <-> C   (判据 max_abs_err < %.0e)" % TOL)
    print("=" * 78)

    for stale in list(HERE.glob("_data_*.bin")) + list(HERE.glob("_out_*.bin")):
        try:
            stale.unlink()
        except OSError:
            pass

    build()

    # ---------------- [1] ncps <-> numpy ----------------
    print(f"\n[1] ncps <-> numpy：4 组维度 + ts，连续 {N_STEPS} 步")
    for nm, ok, err, _ in TP.run_parity(verbose=True):
        check(nm, ok, f"{err:.3e}")

    # main 配置（contract-target 106/64/64）的结果后面反复用
    _, hs_ref_main, curve_np_main, params_main, cell_main, inputs_main = TP.run_case(
        "contract-target 106/64/64", 106, 64, 64, 1, steps=N_STEPS, verbose=False)
    h0_main = np.zeros((1, 64), dtype=np.float32)
    print(f"    main 配置指纹 sha1={fingerprint(hs_ref_main)}  "
          f"h[-1][:3]={np.round(hs_ref_main[-1][:3], 6)}")

    # ---------------- [2] numpy <-> C ----------------
    print("\n[2] numpy <-> C（权重由 ncps cell 导出，同一份权重三方共用）")
    curve_c_main, info_main = None, None
    for nm, i_d, h_d, b_u, b_l in TP.CONFIGS:
        _, hs_ref, _, params, _, inputs = TP.run_case(nm, i_d, h_d, b_u, b_l,
                                                     steps=N_STEPS, verbose=False)
        h0 = np.zeros((1, h_d), dtype=np.float32)
        is_main = nm.startswith("contract-target")
        _, _, _, curve, info = np_vs_c(f"C vs numpy {nm}", params, inputs, h0,
                                       use_retention=False, torch_ref=hs_ref,
                                       tag="main" if is_main else None)
        if is_main:
            curve_c_main, info_main = curve, info

    for ts in (0.25, 3.0):
        _, hs_ref, _, params, _, inputs = TP.run_case(
            f"contract-target ts={ts}", 106, 64, 64, 1, steps=N_STEPS, ts=ts, verbose=False)
        np_vs_c(f"C vs numpy target ts={ts}", params, inputs,
                np.zeros((1, 64), dtype=np.float32), use_retention=False, torch_ref=hs_ref)

    print("\n[2b] AutoNCP 形态：无 backbone + 稀疏掩码折叠导出（W_export = W*|mask^T|）")
    err_m, hs_ref_m, params_m, inputs_m, h0_m = TP.run_masked_case(verbose=True)
    check("ncps(带 mask) <-> numpy(掩码已折进权重)", err_m < TOL, f"{err_m:.3e}")
    np_vs_c("C vs numpy autoncp-mask", params_m, inputs_m, h0_m,
            use_retention=False, torch_ref=hs_ref_m)

    print("\n[2c] AutoNCP wiring 多层串联（units=64/out=2 -> 3 层）：C 侧链 3 个 cfc_cell_t")
    wcell, _ = TP.build_wired(input_dim=12, units=64, out=2, seed=11)
    layers = TP.wired_layers(wcell)
    layer_sizes = [n for _, n in layers]
    inputs_w = TP.make_inputs(50, 12, 555)
    hs_torch_w = TP.torch_wired_seq(wcell, inputs_w, np.zeros((1, sum(layer_sizes)), np.float32))
    hs_np_w = TP.np_wired_chain(layers, inputs_w)
    print(f"    AutoNCP(64,2) 的实际层结构 {layer_sizes}（h = 3 段拼接，共 {sum(layer_sizes)} 单元）")
    check("wired(3层) ncps<->numpy", maxerr(hs_torch_w, hs_np_w) < TOL,
          f"{maxerr(hs_torch_w, hs_np_w):.3e}")
    x, hs_c_parts = inputs_w, []
    for i, (p, n) in enumerate(layers):
        hs_i, _, _ = c_run(f"wired_L{i}", p, x, np.zeros((1, n), np.float32), False)
        hs_c_parts.append(hs_i)
        x = hs_i  # 链式：第 i 层的 C 输出直接喂第 i+1 层，误差会逐层累积
    check("wired(3层) numpy<->C（C 自链）", maxerr(hs_np_w, np.concatenate(hs_c_parts, 1)) < TOL,
          f"{maxerr(hs_np_w, np.concatenate(hs_c_parts, 1)):.3e}")

    # ---------------- [3] 保留门 ----------------
    print("\n[3] 保留门（ncps 输出 + torch 端施加保留门 <-> numpy <-> C）")
    hidden = params_main["ff1_b"].shape[0]
    ret_rng = np.random.default_rng(7)
    for tag, ret in (
        ("ret=-3 契约初值", cfc_ref._f32(np.full(hidden, -3.0))),
        ("ret=U(-2,4)", cfc_ref._f32(ret_rng.uniform(-2.0, 4.0, hidden))),
        ("ret=+8 lam~1", cfc_ref._f32(np.full(hidden, 8.0))),
    ):
        p = dict(params_main)
        p["ret"] = ret
        hs_torch = TP.torch_seq(cell_main, inputs_main, h0_main, ts=1.0, ret=ret)
        np_vs_c(f"C vs numpy 保留门 {tag}", p, inputs_main, h0_main, use_retention=True,
                ret=ret, torch_ref=hs_torch)
        hs_np, _ = cfc_ref.cfc_forward_seq(p, inputs_main, h0=h0_main)
        check(f"ncps+torch 保留门 {tag} <-> numpy", maxerr(hs_torch, hs_np) < TOL,
              f"lam_mean={cfc_ref.sigmoid(ret).mean():.4f}")

    # ---------------- [4] lambda=0 逐位一致 ----------------
    print("\n[4] 保留门 lambda=0（ret=-1e4，sigmoid 精确下溢到 0）与不启用逐位一致")
    ret_zero = cfc_ref._f32(np.full(hidden, -1e4))
    check("sigmoid(-1e4) 精确为 0", float(cfc_ref.sigmoid(ret_zero).max()) == 0.0,
          f"max lam={float(cfc_ref.sigmoid(ret_zero).max()):.1e}")

    _, raw_off, _ = c_run("lam0_off", params_main, inputs_main, h0_main, False)
    hs_c_on, raw_on, _ = c_run("lam0_on", params_main, inputs_main, h0_main, True, ret=ret_zero)
    check("C: 启用(lam=0) 与 不启用 out_c.bin 逐字节相同", raw_off == raw_on,
          f"{len(raw_off)} bytes")
    if raw_off != raw_on:
        d = np.frombuffer(raw_off[16:], "<f4") != np.frombuffer(raw_on[16:], "<f4")
        print(f"       首个不同下标 {np.flatnonzero(d)[:5]}")

    p_off, p_on = dict(params_main), dict(params_main)
    p_off["ret"], p_on["ret"] = None, ret_zero
    hs_np_off, _ = cfc_ref.cfc_forward_seq(p_off, inputs_main, h0=h0_main)
    hs_np_on, _ = cfc_ref.cfc_forward_seq(p_on, inputs_main, h0=h0_main)
    check("numpy: 启用(lam=0) 与 不启用逐字节相同",
          hs_np_off.tobytes() == hs_np_on.tobytes())
    check("C(lam=0) 与 numpy(不启用) 最大误差 < 1e-5",
          maxerr(hs_c_on, hs_np_off) < TOL, f"{maxerr(hs_c_on, hs_np_off):.3e}")

    # ---------------- [5] 边界输入 ----------------
    print(f"\n[5] 边界输入（全 0 / 全 1 / ±10 饱和），各 {N_STEPS} 步")
    for kind, label in (("zeros", "全 0"), ("ones", "全 1"), ("sat", "±10 饱和")):
        inputs = TP.make_inputs(N_STEPS, 106, 1234, kind=kind)
        hs_ref = TP.torch_seq(cell_main, inputs, h0_main, ts=1.0)
        hs_np, _ = cfc_ref.cfc_forward_seq(params_main, inputs, h0=h0_main)
        hs_c, _, _ = c_run(f"bound_{kind}", params_main, inputs, h0_main, False)
        finite = all(np.isfinite(x).all() for x in (hs_ref, hs_np, hs_c))
        check(f"边界 {label}: 三方全部有限", finite, f"|h|max={np.abs(hs_c).max():.3f}")
        check(f"边界 {label}: ncps<->numpy", maxerr(hs_ref, hs_np) < TOL,
              f"{maxerr(hs_ref, hs_np):.3e}")
        check(f"边界 {label}: numpy<->C", maxerr(hs_np, hs_c) < TOL,
              f"{maxerr(hs_np, hs_c):.3e}")

    # ---------------- [6] 误差累积曲线 ----------------
    print(f"\n[6] {N_STEPS} 步递推的误差累积曲线（contract-target 106/64/64，随机输入）")
    print(f"    {'step':>5}  {'ncps<->numpy':>13}  {'numpy<->C':>13}")
    for k in (1, 10, 50, 100, 200):
        print(f"    {k:>5}  {curve_np_main[k-1]:>13.3e}  {curve_c_main[k-1]:>13.3e}")
    for label, curve in (("ncps<->numpy", curve_np_main), ("numpy<->C", curve_c_main)):
        head, tail = float(curve[:10].max()), float(curve[-10:].max())
        ratio = tail / head if head > 0 else float("nan")
        trend = ("无累积（后10步/前10步=%.2fx）" % ratio if ratio < 10
                 else "有累积趋势（%.1fx，涨到 %.2e）" % (ratio, tail))
        print(f"    {label:<12} 前10步max={head:.2e}  后10步max={tail:.2e}  {trend}")
        check(f"{label} {N_STEPS} 步全程 < {TOL:.0e}", float(curve.max()) < TOL,
              f"max={curve.max():.3e}")

    # ---------------- [7] C 单步耗时 ----------------
    macs = int(info_main["MACS_PER_STEP"])
    bu = int(info_main["BACKBONE_UNITS"])
    print(f"\n[7] C 单步耗时（{BENCH_STEPS} 步 x 5 次取最优，PC 单核）")
    print(f"    MAC/步 = {macs}；超越函数/步 = {bu} lecun-tanh + {2*hidden} tanh + {hidden} exp"
          f"（保留门再 +{hidden} exp）")
    ret_m3 = cfc_ref._f32(np.full(hidden, -3.0))
    p_bench = dict(params_main)
    p_bench["ret"] = ret_m3
    us = {}
    for label, use_ret, r in (("无保留门", False, None), ("保留门 ret=-3（部署形态）", True, ret_m3)):
        _, _, info = c_run("main", p_bench if use_ret else params_main, inputs_main,
                           h0_main, use_ret, ret=r, bench=BENCH_STEPS)
        us[label] = float(info["BENCH_US_PER_STEP"])
        print(f"    {label:<22} PC {us[label]:8.3f} us/步  "
              f"ESP32-S3(x{ESP32_SLOWDOWN:.0f}) {us[label]*ESP32_SLOWDOWN:7.1f} us/步")
    deploy = us["保留门 ret=-3（部署形态）"]
    flops = 2.0 * macs
    for label, mflops in (("20 MFLOPS", 20e6), ("40 MFLOPS", 40e6)):
        print(f"    交叉检查：TECH-PLAN §6.4 的 {label:<9} -> {flops / mflops * 1e6:6.1f} us/步")
    check("C 单步耗时测量有效", deploy > 0.0,
          f"PC {deploy:.3f} us/步，ESP32 折算 {deploy*ESP32_SLOWDOWN:.1f} us/步")

    # ---------------- [8] 引擎自检 ----------------
    print("\n[8] 引擎自检（test_engine.exe --selftest：清零 / NULL 入参 / 超限 / lam=0 位级）")
    text = run_cmd([EXE, "--selftest"])
    selftest_ok = "SELFTEST=0" in text
    check("test_engine.exe --selftest", selftest_ok,
          " ".join(ln.strip() for ln in text.splitlines()))

    # ---------------- 汇总 ----------------
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    print("-" * 78)
    print(f"共 {len(RESULTS)} 项：{len(RESULTS) - n_fail} PASS / {n_fail} FAIL")
    print("PASS" if n_fail == 0 else "FAIL")

    if n_fail == 0:  # 全过就清掉中间文件，只留 testdata.bin / out_c.bin
        for stale in list(HERE.glob("_data_*.bin")) + list(HERE.glob("_out_*.bin")):
            try:
                stale.unlink()
            except OSError:
                pass
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
