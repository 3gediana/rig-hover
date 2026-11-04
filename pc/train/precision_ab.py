# -*- coding: utf-8 -*-
"""precision_ab.py -- 三个方案的头部误差 A/B：float32 / 旧 15-bit int8 / 新 W8A8。

三个方案都是**固件原版代码**在 PC 上跑出来的，不是 numpy 近似：

  f32      : 当前 cfc_engine.c + cfc_model.c，float32 blob
  W8A16    : 改造前那一份 cfc_engine.c（hi/lo 双 int8 激活），旧 int8 blob
  W8A8     : 当前 cfc_engine.c + cfc_model.c，bulk int8 + heads fp32 blob

旧源码从哪个目录来由 ``--old-fw`` 指定（默认 ``build/old_fw``，里面有
cfc_engine.c / cfc_engine.h / cfc_model.c / cfc_model.h 的快照 + run_frames.c）。
没有快照时只跑 f32 vs W8A8 两种。

判据口径与 test_e2e.py 一致：绝对误差是"归一化输出"（tanh 头），乘上
``head_max`` / ``vyaw_max`` / ``vx_max`` 就是控制器真正看到的物理量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ctools          # noqa: E402
import export_blob     # noqa: E402

BUILD = HERE / "build" / "ab"
#: tanh 头的物理量换算（= 固件 publish() 用的 head_max/vyaw_max/vx_max）
SCALE = {"head_yaw": 45.0, "vyaw": 90.0, "vx": 0.30}
UNIT = {"head_yaw": "deg", "vyaw": "deg/s", "vx": "m/s"}


def build_engines(host_src, old_fw):
    """返回 {方案名: 可执行文件}；没有旧快照时只返回当前的。"""
    BUILD.mkdir(parents=True, exist_ok=True)
    fw = export_blob.FIRMWARE_DIR
    engines = {}
    exe, warns = ctools.compile_exe(
        [fw / "cfc_engine.c", fw / "cfc_model.c", HERE / "host_shim.c",
         HERE / "host_blob_main.c"],
        "host_now.exe", include_dirs=[HERE / "hoststub", fw], out_dir=BUILD)
    if not exe.exists():
        raise RuntimeError("当前固件源码编译失败: %s" % "; ".join(warns[:5]))
    engines["now"] = exe
    if old_fw is not None and (old_fw / "cfc_engine.c").exists() and \
            (old_fw / "run_frames.c").exists():
        exe_old, warns_old = ctools.compile_exe(
            [old_fw / "cfc_engine.c", old_fw / "cfc_model.c",
             HERE / "host_shim.c", old_fw / "run_frames.c"],
            "host_old.exe", include_dirs=[HERE / "hoststub", old_fw],
            out_dir=BUILD)
        if exe_old.exists():
            engines["old"] = exe_old
        else:
            print("旧快照编译失败: %s" % "; ".join(warns_old[:5]))
    return engines


def run_blob(exe, blob_path, raw, tag, old_style=False):
    """逐帧跑一个 blob，返回 (T, 80) 输出矩阵。"""
    frames = BUILD / ("frames_%s.bin" % tag)
    out = BUILD / ("out_%s.bin" % tag)
    np.ascontiguousarray(raw, dtype=np.float32).tofile(frames)
    argv = ([str(blob_path), str(frames), str(out)] if old_style
            else ["run", str(blob_path), str(frames), str(out)])
    log, rc = ctools.run_exe(exe, argv, check=False)
    if rc != 0:
        raise RuntimeError("%s 失败 rc=%d: %s" % (tag, rc, log[-400:]))
    cols = len(export_blob.HEADS) * export_blob.CFC_MAX_HEAD_DIM
    return np.fromfile(out, dtype=np.float32).reshape(raw.shape[0], cols)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def per_head(ref, got):
    """每个头的最大绝对误差 + 相对（分母=参考峰值的绝对值）。"""
    rows = []
    for i, (name, d, act) in enumerate(export_blob.HEADS):
        a = ref[:, i * 16:i * 16 + d]
        b = got[:, i * 16:i * 16 + d]
        e = float(np.abs(a - b).max())
        rows.append((name, d, act, e, float(np.abs(a).max())))
    return rows


def argmax_agree(ref, got):
    worst = 1.0
    for i, (_n, d, act) in enumerate(export_blob.HEADS):
        if act != "softmax":
            continue
        a = ref[:, i * 16:i * 16 + d]
        b = got[:, i * 16:i * 16 + d]
        worst = min(worst, float((a.argmax(1) == b.argmax(1)).mean()))
    return worst


def prob_err(ref, got):
    worst = 0.0
    for i, (_n, d, act) in enumerate(export_blob.HEADS):
        if act != "softmax":
            continue
        a = ref[:, i * 16:i * 16 + d]
        b = got[:, i * 16:i * 16 + d]
        worst = max(worst, float(np.abs(softmax(a) - softmax(b)).max()))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description="float32 / 旧 15-bit int8 / 新 W8A8 的头部误差 A/B")
    ap.add_argument("--npz", default=str(HERE / "policy.npz"))
    ap.add_argument("--obs", default=str(HERE / "data" / "synth_s007.npz"),
                    help="观测序列 npz（键 obs，形状 (T, input_dim)），跨帧隐状态连续")
    ap.add_argument("--old-fw", default=str(HERE / "build" / "old_fw"),
                    help="改造前固件源码快照目录（含 run_frames.c）")
    ap.add_argument("--json-out", default=str(HERE / "build" / "precision_ab.json"))
    args = ap.parse_args(argv)

    policy = export_blob.load_policy_npz(args.npz)
    raw = np.ascontiguousarray(np.load(args.obs)["obs"], dtype=np.float32)
    BUILD.mkdir(parents=True, exist_ok=True)

    blobs = {}
    blobs["f32"] = BUILD / "ab_f32.blob"
    blobs["f32"].write_bytes(export_blob.build_blob(policy))
    blobs["w8a8"] = BUILD / "ab_w8a8.blob"
    blobs["w8a8"].write_bytes(export_blob.build_blob(policy, int8=True))
    old_fw = Path(args.old_fw)
    legacy = old_fw / "model_slot0_legacy_int8.blob"
    if legacy.exists():
        blobs["w8a16"] = legacy

    engines = build_engines(HERE / "host_blob_main.c", old_fw)
    print("引擎: %s；观测 %d 帧（%s）"
          % (", ".join(engines), raw.shape[0], Path(args.obs).name))

    ref = run_blob(engines["now"], blobs["f32"], raw, "f32")
    results = {}
    for tag, exe_key, old_style in (("w8a8", "now", False),
                                    ("w8a16", "old", True)):
        if tag not in blobs or exe_key not in engines:
            continue
        got = run_blob(engines[exe_key], blobs[tag], raw, tag, old_style=old_style)
        results[tag] = {
            "bytes": blobs[tag].stat().st_size,
            "per_head": per_head(ref, got),
            "softmax_prob": prob_err(ref, got),
            "argmax": argmax_agree(ref, got),
        }

    names = [n for n, _d, _a in export_blob.HEADS]
    print("\n%s" % ("=" * 74))
    print("方案               体积       " + "  ".join("%-12s" % n for n in names[:5]))
    print("-" * 74)
    for tag in ("w8a16", "w8a8"):
        if tag not in results:
            continue
        r = results[tag]
        cells = "  ".join("%-12.3e" % e for _n, _d, _a, e, _r in r["per_head"][:5])
        print("%-16s %7d B   %s" % (tag, r["bytes"], cells))
    print("\n物理量（最大绝对误差）")
    for tag in ("w8a16", "w8a8"):
        if tag not in results:
            continue
        parts = []
        for n, _d, act, e, _r in results[tag]["per_head"]:
            if act == "tanh":
                parts.append("%s %.3f %s" % (n, e * SCALE[n], UNIT[n]))
        print("  %-8s %s" % (tag, " / ".join(parts)))
        print("           softmax 概率最大偏差 %.3e，argmax 一致率 %.4f"
              % (results[tag]["softmax_prob"], results[tag]["argmax"]))
    Path(args.json_out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    print("\n写出 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
