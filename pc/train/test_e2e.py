# -*- coding: utf-8 -*-
"""test_e2e.py — 端到端闭环验收：合成数据 → 训练 → 导出 blob → **固件原版** cfc_model.c
加载 → 逐帧 torch vs C 对比 → 断言 < 1e-5。

一条命令跑完：

    D:\\smartcar\\.venv\\Scripts\\python.exe D:\\robot_work\\pc\\train\\test_e2e.py

它回答的问题只有一个：**"明早传上机器人的 blob 会不会算错"**。

流程（每步都打印 PASS/FAIL）：

    [0] 环境        cl.exe / vcvars64 / torch / GPU
    [1] 头部偏移    编译 dump_offsets.c，用 offsetof 打印真实偏移，
                    与固件 cfc_model.c 的 #define H_* 和 export_blob.py 三方比对
    [2] 合成数据    synth.py 生成 N 条会话（npz）
    [3] 训练        train_cfc.py 略跑几个 epoch（GPU），出 checkpoint.pt/policy.npz
    [4] 导出        policy.npz -> model_slot0.blob（CRC/层表/头字段逐字节）
    [5] 固件加载器  编译 cfc_model.c + cfc_engine.c（用 hoststub 顶掉 ESP-IDF），
                    upload -> select，把解析出来的字段与层指针偏移打回 JSON
    [6] 头字段核对  上面那份 JSON vs 导出侧 manifest，逐字段
    [7] 数值对拍    同一批观测逐帧跑：torch vs C，判据 max_abs_err < 1e-5
    [8] 负例        坏 CRC / 坏 magic / 零 std / 超长 input_dim / 截断
                    必须被固件加载器拒绝（能拒绝才说明校验是真的在跑）

判据：全部通过打印 PASS，退出码 0。
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ctools                                          # noqa: E402
import export_blob                                     # noqa: E402
import obs as obsmod                                   # noqa: E402

BUILD = HERE / "build" / "e2e"
TOL = 1e-5
OUT_STRIDE = 16 * 8        # CFC_MAX_HEADS * CFC_MAX_HEAD_DIM

#: int8 路径判据。逐层算子相对误差来自 cfc_selftest()（固件自己也跑一遍），
#: 全链路绝对误差只在 tanh 头上成立：softmax 头的原始 logit 无界（本模型到
#: ±7），8 bit 权重的 1% 相对误差搬到 logit 上就是 7e-2，物理上到不了 5e-3。
INT8_REL_TOL = 1e-2        # 逐层相对误差 < 1%
#: 头级别相对误差上限，**只做退化守卫**：分母是该头在测试集上的输出峰值，
#: 峰值为小量的头会被这个比值放大（本轮 1669 帧里 head_yaw 峰值约 0.36，
#: 绝对误差 7.3e-3 就是 2.0%；W8A16 时同一分母下是 1.0e-2 量级）。真正有物理
#: 含义的约束是下面的绝对误差判据，这里只挡"整头输出趋近于 0"这种退化。
INT8_HEAD_REL_TOL = 3e-2
#: tanh 头（直接驱动 yaw/vyaw/vx）的绝对误差上限。
#:
#: 口径变化说明：15 bit 激活（hi/lo session）的实测地板是 3.4e-3，当时判据是
#: 5e-3；改成单 int8 激活（W8A8，每行一次累加器会话）后地板抬到 ~6.5e-3，
#: 5e-3 不可能通过。1.5e-2 是"地板的 2 倍"而不是"看着顺眼"，换算成物理量：
#: 1.5e-2 × 45° = 0.68°、× 90°/s = 1.35°/s、× 0.30 m/s = 4.5 mm/s，仍小于
#: 2° 控制死区与传感器噪声。真要回到 5e-3，只能把会话数从 384 加回 768。
INT8_ABS_TOL = 1.5e-2
INT8_PROB_TOL = 5e-2       # softmax 概率误差（只做报告与宽松门限）
INT8_ARGMAX_TOL = 0.999    # softmax 头 argmax 一致率

#: tanh 头的物理量换算（cfc_task.c publish() 用的系数），只用于报告。
HEAD_SCALE = {"head_yaw": 45.0, "vyaw": 90.0, "vx": 0.30}
HEAD_UNIT = {"head_yaw": "°", "vyaw": "°/s", "vx": "m/s"}

RESULTS = []

# Windows 控制台/重定向默认 GBK：编码不了的字符退化成 '?'，别让 UnicodeEncodeError 打断验收
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流没有 reconfigure
        pass


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  " + detail) if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# [0] 环境
# --------------------------------------------------------------------------- #
def step_env():
    print("[0] 环境")
    ok = check("cl.exe 存在", ctools.CL_EXE.exists(), str(ctools.CL_EXE))
    env = ctools.msvc_env()
    ok &= check("MSVC INCLUDE/LIB 就绪", bool(env.get("INCLUDE")) and bool(env.get("LIB")))
    try:
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        ok &= check("torch 可用", True, "torch %s, device %s" % (torch.__version__, dev))
    except Exception as exc:                                    # noqa: BLE001
        ok &= check("torch 可用", False, repr(exc))
    return ok


# --------------------------------------------------------------------------- #
# [1] 头部偏移三方核对
# --------------------------------------------------------------------------- #
def step_offsets():
    print("[1] 头部偏移：固件 #define H_* / 编译器 offsetof / export_blob 三方比对")
    exe, warns = ctools.compile_exe(["dump_offsets.c"], "dump_offsets.exe", out_dir=BUILD)
    own = [w for w in warns if w.startswith("dump_offsets.c")]
    ok = check("dump_offsets.c 编译零告警", not own, "; ".join(own))
    out, rc = ctools.run_exe(exe, check=False)
    ok &= check("dump_offsets 自检退出码 0", rc == 0, "rc=%d" % rc)
    ok &= check("dump_offsets 自检 PASS", "PASS" in out and "FAIL" not in out)
    ok2, problems = export_blob.check_offsets(exe, verbose=False)
    ok &= check("三方偏移一致（%d 个字段，头 %d 字节）"
                % (len(export_blob.HDR_FIELDS), export_blob.CFC_HDR_BYTES),
                ok2, "; ".join(problems[:3]))
    return ok, exe


# --------------------------------------------------------------------------- #
# [2] 合成数据
# --------------------------------------------------------------------------- #
def step_synth(n_sessions=4, duration=32.0):
    print("[2] 合成数据（%d 条会话 × %.0f s）" % (n_sessions, duration))
    import synth
    data_dir = BUILD / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.glob("*.npz"):
        old.unlink()
    t0 = time.perf_counter()
    sessions = []
    for k in range(n_sessions):
        s = synth.generate_session(seed=1000 + k, fps=synth.FPS_DEFAULT,
                                   duration=duration)
        s.save(data_dir / ("synth_s%03d.npz" % k))
        sessions.append(s)
    dt = time.perf_counter() - t0
    ok = check("%d 条会话生成" % len(sessions), all(len(s) > 500 for s in sessions),
               "共 %d 帧，%.2f s" % (sum(len(s) for s in sessions), dt))
    mode = np.concatenate([s.mode for s in sessions])
    frac = {m: float(np.mean(mode == m)) for m in range(4)}
    # REFRACTORY (mode 3) is retired by design. The gesture layer now emits a
    # discrete onset edge and the ACK response is a fixed 1.5 s pulse, so the
    # controller never enters a separate cooling state; asserting four modes
    # would pin the test to a behaviour that was deliberately removed.
    # What still has to hold is that IDLE, TRACK and ACK all appear, otherwise
    # the class-weighted CE has an empty class to weight.
    ok &= check("mode 含 IDLE/TRACK/ACK（REFRACTORY 已按事件化设计退役）",
                frac[0] > 0.02 and frac[1] > 0.20 and frac[2] > 0.0,
                "IDLE %.1f%% TRACK %.1f%% ACK %.1f%%  (mode3 retired: %.1f%%)"
                % (100 * frac[0], 100 * frac[1], 100 * frac[2],
                   100 * frac[3]))
    obs_all = np.concatenate([s.obs for s in sessions])
    const = np.where(obs_all.std(axis=0) < 1e-6)[0]
    ok &= check("观测无 NaN/Inf", np.isfinite(obs_all).all(),
                "常数维 %s（固件写死 0 的 76/77 等）" % list(const))
    return ok, data_dir, sessions


# --------------------------------------------------------------------------- #
# [3] 训练
# --------------------------------------------------------------------------- #
def step_train(data_dir, epochs=2):
    print("[3] 训练（epochs=%d，控制时间）" % epochs)
    import train_cfc
    t0 = time.perf_counter()
    argv = ["--data-dir", str(data_dir), "--out-dir", str(BUILD / "model"),
            "--epochs", str(epochs), "--seq-len", "32", "--batch-size", "8"]
    rc = train_cfc.main(argv)
    dt = time.perf_counter() - t0
    npz = BUILD / "model" / "policy.npz"
    ckpt = BUILD / "model" / "checkpoint.pt"
    ok = check("训练完成且退出码 0", rc == 0, "%.1f s" % dt)
    ok &= check("checkpoint.pt / policy.npz 存在", npz.exists() and ckpt.exists())
    summary = json.loads((BUILD / "model" / "train_summary.json").read_text("utf-8"))
    lam = summary["lambda_end"]
    ok &= check("保留门 lambda 在合理区间", 0.0 < lam < 1.0,
                "lambda %.4f -> %.4f" % (summary["lambda_start"], lam))
    return ok, npz, ckpt, summary


# --------------------------------------------------------------------------- #
# [4] 导出 blob
# --------------------------------------------------------------------------- #
def step_export(npz_path, dump_exe):
    print("[4] 导出 blob")
    policy = export_blob.load_policy_npz(npz_path)
    blob = export_blob.build_blob(policy)
    blob_path = BUILD / "model_slot0.blob"
    blob_path.write_bytes(blob)
    man = export_blob.manifest_of(blob)
    man["source_npz"] = str(npz_path)
    (BUILD / "blob_manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")

    ok = check("blob 生成", len(blob) > export_blob.CFC_HDR_BYTES,
               "%d 字节（头 %d + payload %d）"
               % (len(blob), export_blob.CFC_HDR_BYTES,
                  len(blob) - export_blob.CFC_HDR_BYTES))
    ok &= check("blob 不超过槽上限", len(blob) <= export_blob.CFC_SLOT_MAX_BYTES,
                "上限 %d" % export_blob.CFC_SLOT_MAX_BYTES)
    ok &= check("CRC32 自洽（zlib 口径，crc 字段当 0）", man["crc32_ok"],
                "0x%08X" % man["crc32"])
    ok &= check("magic/ver 正确",
                man["magic"] == export_blob.CFC_BLOB_MAGIC
                and man["ver"] == export_blob.CFC_BLOB_VER,
                "'CFCM'=0x%08X ver=%d" % (man["magic"], man["ver"]))
    ok &= check("flags 同时带投影层与保留门",
                (man["flags"] & 3) == 3, "flags=0x%04X" % man["flags"])
    ok &= check("input_std 全部 >= 1e-6（固件会拒绝 0）",
                min(man["input_std"][:man["input_dim"]]) >= 1e-6,
                "最小值 %.3g" % min(man["input_std"][:man["input_dim"]]))
    ok &= check("所有 layer_off 都是 4 的倍数且未越界",
                all(o % 4 == 0 and o + n <= len(blob)
                    for o, n in zip(man["layer_off"], man["layer_len"]) if n),
                "payload 起于 %d" % export_blob.CFC_HDR_BYTES)
    ok &= check("导出侧偏移仍与固件一致",
                export_blob.check_offsets(dump_exe, verbose=False)[0])
    export_blob.describe(blob)
    return ok, blob_path, blob, man


# --------------------------------------------------------------------------- #
# [5][6] 固件加载器 + 头字段核对
# --------------------------------------------------------------------------- #
def build_host_harness():
    print("[5] 编译固件原版加载器（cfc_model.c + cfc_engine.c + hoststub）")
    srcs = [export_blob.FIRMWARE_DIR / "cfc_engine.c",
            export_blob.FIRMWARE_DIR / "cfc_model.c",
            HERE / "host_shim.c", HERE / "host_blob_main.c"]
    exe, warns = ctools.compile_exe(
        srcs, "host_blob.exe",
        include_dirs=[HERE / "hoststub", export_blob.FIRMWARE_DIR],
        out_dir=BUILD)
    own = [w for w in warns if w.startswith(("host_shim.c", "host_blob_main.c"))]
    fw = [w for w in warns if w.startswith(("cfc_model.c", "cfc_engine.c"))]
    ok = check("固件 cfc_model.c / cfc_engine.c 在 PC 上编过", exe.exists(),
               "%s（固件侧 %d 条告警，非本次代码，仅记录）" % (exe.name, len(fw)))
    for w in fw[:5]:
        print("        ! " + w)
    ok &= check("本次新增的 C 文件零告警", not own, "; ".join(own[:3]))
    return ok, exe


def step_parse_check(exe, blob_path, man):
    print("[6] 固件加载器解析 blob（upload -> select），逐字段核对")
    out, rc = ctools.run_exe(exe, ["info", blob_path], check=False)
    ok = check("upload+select 成功（CRC/维度/层长全部通过固件校验）", rc == 0,
               "rc=%d" % rc)
    if rc != 0:
        print(out)
        return False, None
    info = json.loads(out[out.index("{"):out.rindex("}") + 1])

    def same(field, got, want):
        return check("头字段 %s" % field, got == want, "固件读到 %r，导出写 %r"
                     % (got, want))

    ok &= same("input_dim", info["input_dim"], man["input_dim"])
    ok &= same("cell_in_dim", info["cell_in_dim"], man["cell_in_dim"])
    ok &= same("hidden_dim", info["hidden_dim"], man["hidden_dim"])
    ok &= same("backbone_units", info["backbone_units"], man["backbone_units"])
    ok &= same("n_heads", info["n_heads"], man["n_heads"])
    ok &= same("head_dim", info["head_dim"], man["head_dim"][:man["n_heads"]])
    ok &= same("head_act", info["head_act"], man["head_act"][:man["n_heads"]])
    ok &= same("has_proj", info["has_proj"], (man["flags"] & export_blob.FLAG_PROJ) != 0)
    ok &= same("has_ret", info["has_ret"], (man["flags"] & export_blob.FLAG_RET) != 0)
    ok &= check("clip_sigma", abs(info["clip_sigma"] - man["clip_sigma"]) < 1e-6,
                "%r vs %r" % (info["clip_sigma"], man["clip_sigma"]))
    ok &= check("ts", abs(info["ts"] - man["ts"]) < 1e-6,
                "%r vs %r" % (info["ts"], man["ts"]))
    ok &= same("blob_len", info["blob_len"], man["blob_len"])
    ok &= same("crc32", info["crc"], man["crc32"])

    # 层指针偏移：固件解析后指针相对 blob 的偏移，必须等于导出时写的 layer_off
    expected_off = {name: man["layer_off"][i] if man["layer_len"][i] else -1
                    for i, name in enumerate(export_blob.LAYER_IDS)}
    layer_ok = True
    for name in ("bb_w", "bb_b", "ff1_w", "ff1_b", "ff2_w", "ff2_b",
                 "ta_w", "ta_b", "tb_w", "tb_b", "ret", "proj_w", "proj_b"):
        got = info["off"][name]
        want = expected_off[name]
        if got != want:
            layer_ok = False
            print("        layer %s: 固件指针偏移 %r vs 导出 %r" % (name, got, want))
    ok &= check("13 个层指针偏移与写出的 layer_off 完全一致", layer_ok)
    ok &= same("input_mean 偏移", info["off"]["input_mean"], 180)
    ok &= same("input_std 偏移", info["off"]["input_std"], 1204)

    # heads 层内部切分：head_w[i] / head_b[i] 的偏移必须等于 heads 层起点 + 前缀和
    # head_w 以 float 为单位偏移，换算成字节要 ×4（固件是 float* 指针算术）
    heads_off = expected_off["heads"]
    heads_len = man["layer_len"][export_blob.LAYER_IDS.index("heads")]
    cursor = heads_off
    head_ok = True
    for i, d in enumerate(man["head_dim"][:man["n_heads"]]):
        if info["off"]["head_w_%d" % i] != cursor:
            head_ok = False
            print("        head_w_%d: %r != %r" % (i, info["off"]["head_w_%d" % i], cursor))
        cursor += d * man["hidden_dim"] * 4
    for i, d in enumerate(man["head_dim"][:man["n_heads"]]):
        if info["off"]["head_b_%d" % i] != cursor:
            head_ok = False
            print("        head_b_%d: %r != %r" % (i, info["off"]["head_b_%d" % i], cursor))
        cursor += d * 4
    ok &= check("heads 层内 10 个指针切分正确（%d 字节）" % heads_len,
                head_ok and cursor == heads_off + heads_len)
    # param_count 按固件公式复算（注意它**不含** ret，这是固件的口径）
    zu = man["backbone_units"] if man["backbone_units"] > 0 else \
        man["cell_in_dim"] + man["hidden_dim"]
    head_floats = sum(d * man["hidden_dim"] for d in man["head_dim"][:man["n_heads"]])
    head_bias = sum(man["head_dim"][:man["n_heads"]])
    expect_pc = head_floats + head_bias + man["hidden_dim"] * zu * 4 + man["hidden_dim"] * 4
    if man["backbone_units"] > 0:
        expect_pc += man["backbone_units"] * (man["cell_in_dim"] + man["hidden_dim"]) \
            + man["backbone_units"]
    expect_pc += man["cell_in_dim"] * man["input_dim"] + man["cell_in_dim"]
    ok &= same("param_count（固件公式复算）", info["param_count"], expect_pc)
    print("        固件解析: in=%d cell_in=%d hid=%d bu=%d heads=%d slot=%d param_count=%d"
          % (info["input_dim"], info["cell_in_dim"], info["hidden_dim"],
             info["backbone_units"], info["n_heads"], info["slot"], info["param_count"]))
    return ok, info


# --------------------------------------------------------------------------- #
# [7] 数值对拍 torch vs C
# --------------------------------------------------------------------------- #
def _torch_reference(ckpt_path, raw):
    """按固件口径复算一遍 torch 输出（含 tanh、softmax 头保留 logits、槽位补 0）。"""
    import torch
    from train_cfc import Policy

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = Policy(input_dim=cfg["input_dim"], cell_in=cfg["cell_in"],
                   hidden=cfg["hidden"], backbone_units=cfg["backbone_units"],
                   ret_init=cfg["ret_init"])
    model.load_state_dict(ck["model"])
    model.eval()
    mean = np.asarray(ck["input_mean"], dtype=np.float32)
    std = np.asarray(ck["input_std"], dtype=np.float32)
    # 固件每帧输出 n_heads * CFC_MAX_HEAD_DIM 个 float（head 的未用槽位补 0）
    out_cols = len(export_blob.HEADS) * export_blob.CFC_MAX_HEAD_DIM

    x, ratio = obsmod.normalize_obs(raw, mean, std, float(cfg["clip_sigma"]))
    with torch.no_grad():
        t_out = model.forward_seq(torch.from_numpy(x)[None, ...])[0].cpu().numpy()
    expected = np.zeros((raw.shape[0], out_cols), dtype=np.float32)
    cur = 0
    for i, (_n, d, act) in enumerate(export_blob.HEADS):
        block = t_out[:, cur:cur + d]
        cur += d
        if act == "tanh":                 # 固件在 C 侧已经做了 tanh；softmax 头保留 logits
            block = np.tanh(block)
        expected[:, i * export_blob.CFC_MAX_HEAD_DIM:
                 i * export_blob.CFC_MAX_HEAD_DIM + d] = block
    return expected, out_cols, ratio


def _compare(exe, blob_path, ckpt_path, raw, tag):
    """跑一遍 固件C引擎 vs torch，返回比较数据（不做断言）。"""
    expected, out_cols, ratio = _torch_reference(ckpt_path, raw)
    frames_path = BUILD / ("frames_%s.bin" % tag)
    out_path = BUILD / ("out_c_%s.bin" % tag)
    np.ascontiguousarray(raw, dtype=np.float32).tofile(frames_path)
    t0 = time.perf_counter()
    out, rc = ctools.run_exe(exe, ["run", blob_path, frames_path, out_path], check=False)
    c_dt = time.perf_counter() - t0
    if rc != 0:
        return {"ok": False, "rc": rc, "stdout": out}
    c_out = np.fromfile(out_path, dtype=np.float32).reshape(raw.shape[0], out_cols)
    mask = np.zeros(out_cols, dtype=bool)
    for i, (_n, d, _a) in enumerate(export_blob.HEADS):
        mask[i * export_blob.CFC_MAX_HEAD_DIM:
             i * export_blob.CFC_MAX_HEAD_DIM + d] = True
    err = np.abs(c_out[:, mask] - expected[:, mask])
    step_err = err.reshape(raw.shape[0], -1).max(axis=1)
    return {"ok": True, "rc": 0, "max_err": float(err.max()),
            "head10": float(step_err[:10].max()), "tail10": float(step_err[-10:].max()),
            "n_frames": int(raw.shape[0]), "n_valid": int(mask.sum()),
            "pad_nonzero": int(np.count_nonzero(c_out[:, ~mask])),
            "ratio": ratio, "sec": c_dt, "expected0": expected[0, :5],
            "c0": c_out[0, :5]}


def _report_compare(tag, data):
    """把一次比较结果变成若干条 PASS/FAIL。"""
    if not data["ok"]:
        check("%s 固件 C 引擎逐帧推理" % tag, False, "rc=%d" % data["rc"])
        print(data.get("stdout", ""))
        return False
    ok = check("%s 固件 C 引擎逐帧推理成功" % tag, True,
               "%d 帧，%.3f s（%.3f ms/步，含读盘）"
               % (data["n_frames"], data["sec"],
                  1000.0 * data["sec"] / max(data["n_frames"], 1)))
    ok &= check("%s torch vs C 最大绝对误差 < %.0e" % (tag, TOL),
                data["max_err"] < TOL,
                "max=%.3e（%d 帧 × %d 个有效输出，clip 比例 %.4f%%）"
                % (data["max_err"], data["n_frames"], data["n_valid"],
                   100 * data["ratio"]))
    ok &= check("%s 误差不随帧数累积（后 10 帧 / 前 10 帧 < 3x）" % tag,
                data["head10"] <= 0 or data["tail10"] / max(data["head10"], 1e-12) < 3.0,
                "前 10 帧 %.2e / 后 10 帧 %.2e" % (data["head10"], data["tail10"]))
    ok &= check("%s 输出填充位（head_dim 之外）全为 0" % tag, data["pad_nonzero"] == 0,
                "非零 %d 个" % data["pad_nonzero"])
    print("        首帧 torch[0..4]=%s / C[0..4]=%s"
          % (np.array2string(data["expected0"], precision=6),
             np.array2string(data["c0"], precision=6)))
    return ok


def step_numeric(exe, blob_path, ckpt_path, sessions):
    print("[7] 数值对拍：同一批观测逐帧 torch vs 固件 C 引擎（判据 < %.0e）" % TOL)
    raw = np.ascontiguousarray(sessions[-1].obs, dtype=np.float32)
    print("        用最后一条会话（验证集，%d 帧）——不拿训练集自欺欺人" % raw.shape[0])
    return _report_compare("e2e", _compare(exe, blob_path, ckpt_path, raw, "e2e"))


# --------------------------------------------------------------------------- #
# [7b] 正式产出（明早要上传的那个 blob）也走一遍同样的加载 + 对拍
# --------------------------------------------------------------------------- #
def step_deliverables(exe, sessions):
    print("[7b] 正式产出核对：%s 下已有的 policy.npz / model_slot0.blob" % HERE.name)
    blob_path = HERE / "model_slot0.blob"
    npz_path = HERE / "policy.npz"
    ckpt_path = HERE / "checkpoint.pt"
    missing = [p.name for p in (blob_path, npz_path, ckpt_path) if not p.exists()]
    if missing:
        print("        跳过（缺 %s；先跑 synth.py -> train_cfc.py -> export_blob.py）"
              % missing)
        return True
    policy = export_blob.load_policy_npz(npz_path)
    blob = blob_path.read_bytes()
    man = export_blob.manifest_of(blob)
    ok = check("正式 blob 与 policy.npz 的维度一致",
               (man["input_dim"] == int(policy["input_dim"])
                and man["hidden_dim"] == int(policy["hidden_dim"])
                and man["n_heads"] == int(policy["n_heads"])),
               "in=%d hid=%d heads=%d，%d 字节"
               % (man["input_dim"], man["hidden_dim"], man["n_heads"], len(blob)))
    out, rc = ctools.run_exe(exe, ["info", blob_path], check=False)
    ok &= check("正式 blob 通过固件加载器 upload+select", rc == 0, "rc=%d" % rc)
    if rc == 0:
        info = json.loads(out[out.index("{"):out.rindex("}") + 1])
        ok &= check("正式 blob 头字段回读一致",
                    info["input_dim"] == man["input_dim"]
                    and info["hidden_dim"] == man["hidden_dim"]
                    and info["n_heads"] == man["n_heads"]
                    and info["crc"] == man["crc32"],
                    "crc=0x%08X param_count=%d" % (man["crc32"], info["param_count"]))
    raw = np.ascontiguousarray(sessions[-1].obs, dtype=np.float32)
    ok &= _report_compare("正式产出", _compare(exe, blob_path, ckpt_path, raw, "deliver"))
    return ok


# --------------------------------------------------------------------------- #
# [8] 负例：校验器必须真的拒绝坏 blob
# --------------------------------------------------------------------------- #
def step_negative(exe, blob, man):
    print("[8] 负例：坏 blob 必须被固件加载器拒绝")
    ok = True
    crc_off = export_blob.HDR_FIELD_OFFSETS["crc32"]

    def with_crc(b):
        b = bytearray(b)
        b[crc_off:crc_off + 4] = b"\x00\x00\x00\x00"
        struct.pack_into("<I", b, crc_off, export_blob.blob_crc32(bytes(b)))
        return bytes(b)

    cases = []
    bad = bytearray(blob)
    bad[-1] ^= 0xFF                                   # payload 改一个字节，CRC 不修
    cases.append(("payload 被改坏（CRC 不修）", bytes(bad), "crc32"))
    bad = bytearray(with_crc(blob))
    struct.pack_into("<I", bad, 0, 0xDEADBEEF)        # magic 错，CRC 修好
    cases.append(("magic 错（CRC 修好）", with_crc(bytes(bad)), "magic"))
    bad = bytearray(with_crc(blob))
    struct.pack_into("<H", bad, 4, 2)                 # ver 错
    cases.append(("ver=2 不支持", with_crc(bytes(bad)), "version"))
    bad = bytearray(with_crc(blob))
    struct.pack_into("<H", bad, export_blob.HDR_FIELD_OFFSETS["input_dim"], 300)
    cases.append(("input_dim=300 越界", with_crc(bytes(bad)), "input_dim"))
    bad = bytearray(with_crc(blob))
    struct.pack_into("<f", bad, export_blob.HDR_FIELD_OFFSETS["input_std"], 0.0)
    cases.append(("input_std 有一维为 0", with_crc(bytes(bad)), "input statistics"))
    bad = bytearray(with_crc(blob))
    struct.pack_into("<f", bad, export_blob.HDR_FIELD_OFFSETS["clip_sigma"], 0.0)
    # clip_sigma=0 不是错误（固件回落 4.0），这条用来确认"不是什么都拒绝"
    cases.append(("clip_sigma=0（应接受，回落 4.0）", with_crc(bytes(bad)), None))
    cases.append(("截断到 1000 字节", bytes(blob[:1000]), "size"))

    tmp = Path(tempfile.mkdtemp(dir=str(BUILD), prefix="neg_"))
    for i, (name, data, want_err) in enumerate(cases):
        p = tmp / ("case_%02d.blob" % i)
        p.write_bytes(data)
        out, rc = ctools.run_exe(exe, ["reject", p], check=False)
        if want_err is None:
            # 期望被接受：用 info 验证
            out2, rc2 = ctools.run_exe(exe, ["info", p], check=False)
            ok &= check("负例[%d] %s" % (i, name), rc2 == 0, "rc=%d" % rc2)
        else:
            detail = ""
            if rc == 0 and "{" in out:
                detail = json.loads(out[out.index("{"):out.rindex("}") + 1])["err"]
            good = (rc == 0) and (want_err.lower() in detail.lower())
            ok &= check("负例[%d] %s 被拒绝" % (i, name), good,
                        "err=%r rc=%d" % (detail, rc))
    return ok


# --------------------------------------------------------------------------- #
# [9] 引擎自检：LUT tanh / sigmoid 与 int8 逐层量化误差
# --------------------------------------------------------------------------- #
def step_engine_selftest(host_exe):
    print("[9] 引擎自检：LUT tanh/sigmoid 对比 libm，int8 逐层量化误差")
    out, rc = ctools.run_exe(host_exe, ["selftest"], check=False)
    ok = check("selftest 命令退出码 0", rc == 0, "rc=%d" % rc)
    if rc != 0:
        print(out)
        return False
    st = json.loads(out[out.index("{"):out.rindex("}") + 1])
    ok &= check("LUT tanh 对 libm 最大误差 < 1e-3", st["tanh_max_err"] < 1e-3,
                "max=%.3e（定义域 ±16 密集扫描）" % st["tanh_max_err"])
    ok &= check("LUT tanh 在饱和点 ±8 也对得上", st["tanh_sat_err"] < 1e-6,
                "%.3e" % st["tanh_sat_err"])
    ok &= check("LUT sigmoid 对 libm 最大误差 < 1e-3", st["sigmoid_max_err"] < 1e-3,
                "max=%.3e（±20 扫描）" % st["sigmoid_max_err"])
    ok &= check("int8 逐层相对误差 < %.0f%%" % (100 * INT8_REL_TOL),
                st["q8_dot_max_rel"] < INT8_REL_TOL,
                "max=%.3e（16 行 × 64 抽头，相对该层最大输出）" % st["q8_dot_max_rel"])
    return ok


# --------------------------------------------------------------------------- #
# [10] int8 路径：导出 -> 固件加载 -> 与 float32 对拍
# --------------------------------------------------------------------------- #
def _run_c(exe, blob_path, raw, tag):
    """用固件 C 引擎跑一遍，返回 (T, 80) 的输出矩阵。"""
    frames_path = BUILD / ("frames_%s.bin" % tag)
    out_path = BUILD / ("out_c_%s.bin" % tag)
    np.ascontiguousarray(raw, dtype=np.float32).tofile(frames_path)
    out, rc = ctools.run_exe(exe, ["run", blob_path, frames_path, out_path],
                             check=False)
    if rc != 0:
        return None, out
    out_cols = len(export_blob.HEADS) * export_blob.CFC_MAX_HEAD_DIM
    return np.fromfile(out_path, dtype=np.float32).reshape(
        raw.shape[0], out_cols), out


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def step_int8(host_exe, npz_path, ckpt_path, sessions, blob32_len):
    print("[10] int8 路径：导出 int8 blob -> 固件加载 -> 与 float32 参考对拍")
    policy = export_blob.load_policy_npz(npz_path)
    blob8 = export_blob.build_blob(policy, int8=True)
    p8 = BUILD / "model_slot0_int8.blob"
    p8.write_bytes(blob8)
    man8 = export_blob.manifest_of(blob8)

    ok = check("int8 blob flags 带 bit2/bit3（proj/ret 不变）",
               man8["flags"] == (export_blob.FLAG_PROJ | export_blob.FLAG_RET
                                 | export_blob.FLAG_INT8
                                 | export_blob.FLAG_LAYERPREC),
               "flags=0x%04X" % man8["flags"])
    ok &= check("int8 blob CRC 自洽", man8["crc32_ok"], "0x%08X" % man8["crc32"])
    ok &= check("int8 blob 体积降到 float32 的 1/2 以下",
                len(blob8) < blob32_len * 0.5,
                "%d 字节 vs %d 字节（%.1f%%）"
                % (len(blob8), blob32_len, 100.0 * len(blob8) / blob32_len))

    out, rc = ctools.run_exe(host_exe, ["info", p8], check=False)
    ok &= check("int8 blob 通过固件 upload+select（层长/CRC/维度全过校验）",
                rc == 0, "rc=%d" % rc)
    if rc != 0:
        print(out)
        return False
    info = json.loads(out[out.index("{"):out.rindex("}") + 1])
    ok &= check("固件解析出 use_int8=1", info["use_int8"] == 1)
    # 固件自己解析出来的逐层精度必须与导出侧 manifest 一致：这是"逐层可选"在
    # 加载器一侧的唯一证据（只比 use_int8 的话，全有全无也能通过）。
    ok &= check("固件解析出的 layer_prec 与导出侧逐层一致",
                info["layer_prec"] == man8["layer_prec"],
                "固件 %s vs 导出 %s"
                % ([i for i, v in enumerate(info["layer_prec"]) if v],
                   [i for i, v in enumerate(man8["layer_prec"]) if v]))
    ok &= check("heads / 偏置 / ret 仍是 float32（逐层表里为 0）",
                all(man8["layer_prec"][export_blob.LAYER_IDS.index(n)] == 0
                    for n in ("heads", "ret", "bb_b", "ff1_b", "proj_b")),
                "heads=%d" % man8["layer_prec"][export_blob.LAYER_IDS.index("heads")])
    ok &= check("int8 blob 的头字段与 float32 版一致",
                (info["input_dim"], info["hidden_dim"], info["n_heads"],
                 info["param_count"]) == (man8["input_dim"], man8["hidden_dim"],
                                          man8["n_heads"], info["param_count"]))

    # 每个 int8 层：固件算出的权重指针 == layer_off，scale 指针 == layer_off+layer_len
    bad = []
    for name in export_blob.INT8_CAPABLE:
        i = export_blob.LAYER_IDS.index(name)
        off = man8["layer_off"][i]
        if info["off8"][name] != off:
            bad.append("%s w=%d!=%d" % (name, info["off8"][name], off))
        if info["off8"][name + "_s"] != off + man8["layer_len"][i]:
            bad.append("%s scale=%d!=%d" % (name, info["off8"][name + "_s"],
                                            off + man8["layer_len"][i]))
    ok &= check("6 个 int8 层的权重指针与 scale 指针偏移全部对上", not bad,
                "; ".join(bad[:4]))

    # 数值对拍：同一批观测，float32 blob vs int8 blob，两边都是固件 C 引擎
    raw = np.ascontiguousarray(sessions[-1].obs, dtype=np.float32)
    p32 = BUILD / "model_slot0.blob"
    c32, _ = _run_c(host_exe, p32, raw, "i8ref")
    c8, log8 = _run_c(host_exe, p8, raw, "i8")
    if c32 is None or c8 is None:
        check("int8 逐帧推理", False, str(log8))
        return False
    check("int8 逐帧推理成功", True, "%d 帧" % raw.shape[0])

    err = np.abs(c8 - c32)
    per_head = []
    for i, (name, d, act) in enumerate(export_blob.HEADS):
        blk = err[:, i * 16:i * 16 + d]
        ref = np.abs(c32[:, i * 16:i * 16 + d]).max()
        per_head.append((name, d, act, float(blk.max()), float(ref)))
    tanh_err = max(e for _n, _d, a, e, _r in per_head if a == "tanh")
    ok &= check("int8 vs float32：tanh 头（yaw/vyaw/vx）最大绝对误差 < %.0e"
                % INT8_ABS_TOL, tanh_err < INT8_ABS_TOL,
                "max=%.3e（%s）"
                % (tanh_err, " / ".join("%s %.2e" % (n, e)
                                        for n, _d, a, e, _r in per_head
                                        if a == "tanh")))
    # 换算到控制器真正消费的物理量（publish() 的 head_max/vyaw_max/vx_max）
    for name, _d, act, e, _r in per_head:
        if act == "tanh":
            print("        %-9s 最大误差 %.2e -> %.3f %s"
                  % (name, e, e * HEAD_SCALE[name], HEAD_UNIT[name]))
    rels = [(n, e / max(r, 1e-9)) for n, _d, _a, e, r in per_head]
    ok &= check("int8 vs float32：5 个头的相对误差都 < %.0f%%"
                % (100 * INT8_HEAD_REL_TOL),
                max(r for _n, r in rels) < INT8_HEAD_REL_TOL,
                " / ".join("%s %.2e" % (n, r) for n, r in rels))

    prob_err = 0.0
    am_ok = 1.0
    for i, (name, d, act) in enumerate(export_blob.HEADS):
        if act != "softmax":
            continue
        a32 = c32[:, i * 16:i * 16 + d]
        a8 = c8[:, i * 16:i * 16 + d]
        p32s = _softmax(a32)
        p8s = _softmax(a8)
        prob_err = max(prob_err, float(np.abs(p32s - p8s).max()))
        am_ok = min(am_ok, float((p32s.argmax(1) == p8s.argmax(1)).mean()))
    ok &= check("int8 vs float32：softmax 头 argmax 一致率 >= %.1f%%"
                % (100 * INT8_ARGMAX_TOL), am_ok >= INT8_ARGMAX_TOL,
                "%.2f%%（概率最大偏差 %.2e）" % (100 * am_ok, prob_err))
    ok &= check("int8 vs float32：softmax 输出概率偏差 < %.0e" % INT8_PROB_TOL,
                prob_err < INT8_PROB_TOL, "max=%.3e" % prob_err)

    # 负例：把某个 int8 层的长度改成 float32 口径，固件必须拒绝
    crc_off = export_blob.HDR_FIELD_OFFSETS["crc32"]
    bad_blob = bytearray(blob8)
    li = export_blob.LAYER_IDS.index("ff1_w")
    len_off = export_blob.HDR_FIELD_OFFSETS["layer_len"] + 4 * li
    struct.pack_into("<I", bad_blob, len_off,
                     man8["layer_len"][li] * 4)
    bad_blob[crc_off:crc_off + 4] = b"\x00\x00\x00\x00"
    struct.pack_into("<I", bad_blob, crc_off, export_blob.blob_crc32(bytes(bad_blob)))
    pb = BUILD / "model_int8_badlen.blob"
    pb.write_bytes(bytes(bad_blob))
    out, rc = ctools.run_exe(host_exe, ["reject", pb], check=False)
    detail = ""
    if rc == 0 and "{" in out:
        detail = json.loads(out[out.index("{"):out.rindex("}") + 1])["err"]
    ok &= check("负例：int8 层长度按 float32 写会被拒绝",
                rc == 0 and "layer size mismatch" in detail,
                "err=%r rc=%d" % (detail, rc))

    # 负例：把 heads 标成 int8（格式不允许）——必须被拒绝，而不是当成 float32 跑
    bad = bytearray(blob8)
    prec_off = export_blob.HDR_FIELD_OFFSETS["layer_prec"]
    bad[prec_off + export_blob.LAYER_IDS.index("heads")] = export_blob.PREC_INT8
    bad[crc_off:crc_off + 4] = b"\x00\x00\x00\x00"
    struct.pack_into("<I", bad, crc_off, export_blob.blob_crc32(bytes(bad)))
    pb = BUILD / "model_int8_badprec.blob"
    pb.write_bytes(bytes(bad))
    out, rc = ctools.run_exe(host_exe, ["reject", pb], check=False)
    detail = ""
    if rc == 0 and "{" in out:
        detail = json.loads(out[out.index("{"):out.rindex("}") + 1])["err"]
    ok &= check("负例：layer_prec 把 heads 标成 int8 会被拒绝",
                rc == 0 and "layer_prec" in detail,
                "err=%r rc=%d" % (detail, rc))

    print("        int8 %.1f%% 体积，最大误差 tanh 头 %.3e / softmax 概率 %.3e"
          % (100.0 * len(blob8) / blob32_len, tanh_err, prob_err))
    return ok


# --------------------------------------------------------------------------- #
# [11] 向后兼容：老导出器写的 blob（头 2240，没有 layer_prec 表）必须照旧加载
# --------------------------------------------------------------------------- #
def step_legacy(host_exe):
    print("[11] 向后兼容：老 blob（无 layer_prec 表）仍被加载器接受")
    legacy_dir = HERE / "build" / "old_fw"
    f32_blob = legacy_dir / "model_slot0_legacy_f32.blob"
    i8_blob = legacy_dir / "model_slot0_legacy_int8.blob"
    if not f32_blob.exists() or not i8_blob.exists():
        print("        跳过（缺 build/old_fw/model_slot0_legacy_*.blob）")
        return True
    ok = True
    for path, want_int8 in ((f32_blob, 0), (i8_blob, 1)):
        raw = path.read_bytes()
        flags = struct.unpack_from("<H", raw, export_blob.HDR_FIELD_OFFSETS["flags"])[0]
        out, rc = ctools.run_exe(host_exe, ["info", path], check=False)
        if rc != 0:
            ok &= check("老 blob %s 通过固件 upload+select" % path.name, False,
                        "rc=%d %s" % (rc, out[-200:]))
            continue
        info = json.loads(out[out.index("{"):out.rindex("}") + 1])
        ok &= check("老 blob %s 是 bit3=0 的旧格式（头 2240）" % path.name,
                    (flags & export_blob.FLAG_LAYERPREC) == 0
                    and info["blob_len"] == len(raw),
                    "flags=0x%04X len=%d" % (flags, info["blob_len"]))
        ok &= check("老 blob %s use_int8=%d（按老规矩解析）"
                    % (path.name, want_int8), info["use_int8"] == want_int8,
                    "use_int8=%d layer_prec=%s" % (info["use_int8"],
                                                   info["layer_prec"][:14]))
        ok &= check("老 blob %s heads 仍 float32" % path.name,
                    info["layer_prec"][export_blob.LAYER_IDS.index("heads")] == 0)
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(sessions=4, duration=32.0, epochs=2, skip_deliverables=False):
    t_start = time.perf_counter()
    BUILD.mkdir(parents=True, exist_ok=True)
    for f in BUILD.glob("*.blob"):
        f.unlink()

    ok = step_env()
    ok &= step_offsets()[0]
    data_ok, data_dir, sessions = step_synth(sessions, duration)
    ok &= data_ok
    train_ok, npz_path, ckpt_path, summary = step_train(data_dir, epochs)
    ok &= train_ok
    exp_ok, blob_path, blob, man = step_export(npz_path, BUILD / "dump_offsets.exe")
    ok &= exp_ok
    host_ok, host_exe = build_host_harness()
    ok &= host_ok
    parse_ok, info = step_parse_check(host_exe, blob_path, man)
    ok &= parse_ok
    ok &= step_numeric(host_exe, blob_path, ckpt_path, sessions)
    ok &= step_negative(host_exe, blob, man)
    ok &= step_engine_selftest(host_exe)
    ok &= step_int8(host_exe, npz_path, ckpt_path, sessions, len(blob))
    ok &= step_legacy(host_exe)
    if not skip_deliverables:
        ok &= step_deliverables(host_exe, sessions)

    dt = time.perf_counter() - t_start
    n_fail = sum(1 for _n, good, _d in RESULTS if not good)
    print("-" * 78)
    print("共 %d 项，%d PASS / %d FAIL，总耗时 %.1f s"
          % (len(RESULTS), len(RESULTS) - n_fail, n_fail, dt))
    print("e2e 本轮: blob %d 字节（头 %d），CRC 0x%08X，训练 %.1f s，lambda %.4f -> %.4f，"
          "最终 train loss %.4f"
          % (len(blob), export_blob.CFC_HDR_BYTES, man["crc32"],
             summary["elapsed_s"], summary["lambda_start"], summary["lambda_end"],
             summary["final_train"]["total"]))
    deliv = HERE / "model_slot0.blob"
    if deliv.exists():
        dman = export_blob.manifest_of(deliv.read_bytes())
        print("正式产出: %s  %d 字节（头 %d），CRC 0x%08X，in=%d hid=%d heads=%d"
              % (deliv.name, dman["blob_len"], export_blob.CFC_HDR_BYTES, dman["crc32"],
                 dman["input_dim"], dman["hidden_dim"], dman["n_heads"]))
    if n_fail:
        for name, good, detail in RESULTS:
            if not good:
                print("  FAIL: %s  %s" % (name, detail))
        print("FAIL")
        return 1
    print("PASS")
    return 0


def cli(argv=None):
    """命令行入口：一条命令跑完全部验收。"""
    import argparse

    ap = argparse.ArgumentParser(
        description="端到端闭环验收：合成数据 -> 训练 -> 导出 blob -> 固件原版加载器加载 "
                    "-> torch vs C 逐帧对拍（< 1e-5）-> 坏 blob 必须被拒。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--sessions", type=int, default=4, help="e2e 内部用的会话数")
    ap.add_argument("--duration", type=float, default=32.0, help="每条会话时长（秒）")
    ap.add_argument("--epochs", type=int, default=2, help="e2e 内部训练的轮数（别调大）")
    ap.add_argument("--skip-deliverables", action="store_true",
                    help="跳过对 train/ 下已有 policy.npz+model_slot0.blob 的核对")
    args = ap.parse_args(argv)
    return main(sessions=args.sessions, duration=args.duration,
                epochs=args.epochs, skip_deliverables=args.skip_deliverables)


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
