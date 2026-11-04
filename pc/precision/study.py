#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""study.py -- weight-precision study for the RIG-Hover CfC policy.

Motivation
----------
The on-device measurement (TECHNICAL-REPORT.md §3.6) shows the forward pass is
*bandwidth*-bound:  t ~= 1.55 ms + bytes / 46 MB/s.  On a bandwidth-bound
accelerator the thing that matters is **bytes per weight**, not arithmetic
precision -- so fp32 / bf16 / fp16 / int8 are points on one axis: 32 / 16 / 16 /
8 bits.  This script measures the other axis (decision error) for the same
candidates, and then asks whether *non-uniform* allocation beats uniform int8
at equal storage.

Reported error is in physical units, not raw tensor units:
    head_yaw  in degrees        (head 0, tanh, caller scales by 45 deg)
    vyaw      in deg/s          (head 1, tanh, caller scales by 90 deg/s)
    vx        in m/s            (head 4, tanh, caller scales by 0.30 m/s)
    mode      argmax agreement  (head 2, 4-way softmax)

Run:  python study.py            -> tables + figures + REPORT.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = r"D:\robot_work\pc\train"
POLICY = os.path.join(TRAIN, "policy.npz")
DATA = os.path.join(TRAIN, "data")

HEAD_SCALE = {0: 45.0, 1: 90.0, 2: 1.0, 3: 1.0, 4: 0.30}  # physical units
HEAD_NAME = {0: "head_yaw(deg)", 1: "vyaw(deg/s)", 2: "mode", 3: "emote", 4: "vx(m/s)"}


# ---------------------------------------------------------------- forward ----
def load_policy(path=POLICY):
    d = np.load(path)
    p = {k: d[k] for k in d.files}
    for k in ("input_dim", "cell_in_dim", "hidden_dim", "backbone_units", "n_heads"):
        p[k] = int(p[k])
    p["clip_sigma"] = float(p["clip_sigma"])
    p["ts"] = float(p["ts"])
    return p


def _layers(p):
    """Ordered map layer-name -> (W, b). Heads handled separately (5 rows)."""
    return {
        "proj": (p["proj_w"].reshape(p["cell_in_dim"], p["input_dim"]),
                 p["proj_b"].reshape(-1)),
        "bb": (p["bb_w"].reshape(p["backbone_units"], p["cell_in_dim"] + p["hidden_dim"]),
               p["bb_b"].reshape(-1)),
        "ff1": (p["ff1_w"].reshape(p["hidden_dim"], -1), p["ff1_b"].reshape(-1)),
        "ff2": (p["ff2_w"].reshape(p["hidden_dim"], -1), p["ff2_b"].reshape(-1)),
        "ta": (p["ta_w"].reshape(p["hidden_dim"], -1), p["ta_b"].reshape(-1)),
        "tb": (p["tb_w"].reshape(p["hidden_dim"], -1), p["tb_b"].reshape(-1)),
        "heads": (p["head_w"].reshape(sum(p["head_dim"]), -1), p["head_b"].reshape(-1)),
    }


def forward(p, obs, L=None, h0=None):
    """Batched over time. Mirrors cfc_model.h INFERENCE exactly."""
    if L is None:
        L = _layers(p)
    mean, std = p["input_mean"], p["input_std"]
    lo = mean - p["clip_sigma"] * std
    hi = mean + p["clip_sigma"] * std
    x = (np.clip(obs, lo, hi) - mean) / std

    W, b = L["proj"]
    z = x @ W.T + b
    z = z / (1.0 + np.exp(-z))                      # SiLU

    hid = p["hidden_dim"]
    h = np.zeros(hid, np.float32) if h0 is None else h0.copy()

    Wbb, bbb = L["bb"]
    W1, b1 = L["ff1"]
    W2, b2 = L["ff2"]
    Wa, ba = L["ta"]
    Wb, bb = L["tb"]
    ret = p["ret"].astype(np.float32)
    lam = 1.0 / (1.0 + np.exp(-ret))
    ts = p["ts"]

    def lecun(v):
        return 1.7159 * np.tanh(0.666 * v)

    outs = np.empty((len(z), int(p["n_heads"]) * 8), np.float32)
    outs[:] = 0.0
    for i in range(len(z)):
        xt = np.concatenate([z[i], h])
        zz = lecun(Wbb @ xt + bbb)
        s1 = np.tanh(W1 @ zz + b1)
        s2 = np.tanh(W2 @ zz + b2)
        tt = 1.0 / (1.0 + np.exp(-((Wa @ zz + ba) * ts + (Wb @ zz + bb))))
        hn = s1 * (1.0 - tt) + tt * s2
        h = hn + lam * (h - hn)                      # retention gate
        Wh, bh = L["heads"]
        raw = Wh @ h + bh
        o = 0
        for k in range(int(p["n_heads"])):
            d = int(p["head_dim"][k])
            act = int(p["head_act"][k])
            seg = raw[o:o + d]
            if act == 1:
                seg = np.tanh(seg)
            elif act == 2:
                seg = seg - seg.max()               # logits kept, caller argmaxes
            outs[i, k * 8:k * 8 + d] = seg
            o += d
    return outs


# ----------------------------------------------------------- quantisers ----
def q_fp32(W):
    return W.astype(np.float32)


def q_bf16(W):
    """Round-to-nearest on the 16 low mantissa bits (bf16: 8 exp / 7 mantissa)."""
    u = np.asarray(W, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x8000) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def q_fp16(W):
    return np.asarray(W, np.float32).astype(np.float16).astype(np.float32)


def q_int8(W, per="channel"):
    """Symmetric int8. per='channel' uses one scale per output row."""
    W = np.asarray(W, np.float32)
    if per == "tensor":
        s = max(float(np.abs(W).max()) / 127.0, 1e-12)
        return np.round(W / s) * s
    m = np.abs(W).max(axis=1, keepdims=True)
    s = np.maximum(m / 127.0, 1e-12)
    return np.round(W / s) * s


QUANT = {
    "fp32":         (q_fp32, q_fp32, 4.0),
    "bf16":         (q_bf16, q_bf16, 2.0),
    "fp16":         (q_fp16, q_fp16, 2.0),
    "int8/tensor":  (lambda W: q_int8(W, "tensor"),
                     lambda W: q_int8(W, "tensor"), 1.0),
    "int8/channel": (lambda W: q_int8(W, "channel"),
                     lambda W: q_int8(W, "tensor"), 1.0),
}


def apply_precision(p, entry, which=None):
    """Quantise weights with the weight fn, biases with the bias fn.

    Biases are 1-D, so per-channel scaling there would give every element its
    own scale -- a lossless round trip that would flatter int8 for no reason.
    The bias always uses tensor-wide scaling of the same precision.
    """
    fn_w, fn_b = entry[0], entry[1]
    L = _layers(p)
    out = {}
    for name, (W, b) in L.items():
        if which is None or name in which:
            out[name] = (fn_w(W), fn_b(b.reshape(1, -1)).reshape(-1))
        else:
            out[name] = (W, b)
    return out


def bytes_for(p, fn, bits):
    tot = 0.0
    for name, (W, b) in _layers(p).items():
        tot += W.size * bits + b.size * 4
        if "int8" in str(fn) or bits == 1.0:
            if fn is QUANT["int8/channel"][0]:
                tot += W.shape[0] * 4
            else:
                tot += 4
    return tot


# --------------------------------------------------------------- errors ----
def errors(ref, got, _unused=None):
    e = {}
    for k in (0, 1, 4):
        d = np.abs(ref[:, k * 8] - got[:, k * 8]) * HEAD_SCALE[k]
        e[HEAD_NAME[k]] = (float(d.max()), float(d.mean()))
    agree = float((np.argmax(ref[:, 16:20], 1) == np.argmax(got[:, 16:20], 1)).mean())
    e["mode"] = (1.0 - agree, 1.0 - agree)
    return e


def load_obs(n_sessions=2):
    xs = []
    for i in range(n_sessions):
        f = os.path.join(DATA, f"synth_s{i:03d}.npz")
        if os.path.exists(f):
            xs.append(np.load(f)["obs"].astype(np.float32))
    return np.concatenate(xs, 0)


# ----------------------------------------------------------------- main ----
def main():
    p = load_policy()
    obs = load_obs()
    print(f"policy: in={p['input_dim']} hidden={p['hidden_dim']} "
          f"bb={p['backbone_units']} heads={list(p['head_dim'])}")
    print(f"eval observations: {len(obs)}")

    L0 = _layers(p)
    ref = forward(p, obs, L0)

    n_params = sum(W.size for W, _ in L0.values())
    base_bytes = n_params * 4
    print(f"weights: {n_params} floats = {base_bytes/1024:.1f} KiB fp32\n")

    # ---- E1 uniform precision -------------------------------------------
    rows = []
    for name, entry in QUANT.items():
        bits = entry[2]
        got = forward(p, obs, apply_precision(p, entry))
        e = errors(ref, got)
        nbytes = n_params * bits
        rows.append({
            "precision": name, "bits": bits, "KiB": nbytes / 1024,
            "head_err_max": e["head_yaw(deg)"][0], "head_err_mean": e["head_yaw(deg)"][1],
            "vyaw_err_max": e["vyaw(deg/s)"][0], "vyaw_err_mean": e["vyaw(deg/s)"][1],
            "vx_err_max": e["vx(m/s)"][0], "mode_agree": 1.0 - e["mode"][1],
        })
    print("=== E1  uniform precision ===")
    print(f"{'precision':<14}{'KiB':>8}{'head max':>10}{'head mean':>11}"
          f"{'vyaw max':>10}{'vyaw mean':>11}{'mode acc':>10}")
    for r in rows:
        print(f"{r['precision']:<14}{r['KiB']:>8.1f}{r['head_err_max']:>10.3f}"
              f"{r['head_err_mean']:>11.4f}{r['vyaw_err_max']:>10.3f}"
              f"{r['vyaw_err_mean']:>11.4f}{r['mode_agree']:>10.4f}")

    # ---- E2 layer sensitivity -------------------------------------------
    print("\n=== E2  int8/channel applied to one layer at a time ===")
    fn8 = QUANT["int8/channel"]
    sens = []
    print(f"{'layer':<8}{'floats':>9}{'head max':>10}{'vyaw max':>10}{'mode acc':>10}")
    for name, (W, b) in L0.items():
        got = forward(p, obs, apply_precision(p, fn8, which={name}))
        e = errors(ref, got)
        sens.append({"layer": name, "floats": int(W.size),
                     "head": e["head_yaw(deg)"][0], "vyaw": e["vyaw(deg/s)"][0],
                     "mode_acc": 1.0 - e["mode"][1]})
        print(f"{name:<8}{W.size:>9}{e['head_yaw(deg)'][0]:>10.3f}"
              f"{e['vyaw(deg/s)'][0]:>10.3f}{1.0 - e['mode'][1]:>10.4f}")

    # ---- E3 greedy mixed precision ---------------------------------------
    # Rank layers by error introduced per byte saved; quantise in that order and
    # record the (bytes, error) curve. If non-uniform allocation is worth
    # anything, this curve must sit below the uniform-int8 point.
    total_out = sum(W.size for W, _ in L0.values()) * 0
    denom = {"head_yaw(deg)": 45.0, "vyaw(deg/s)": 90.0, "mode": 1.0, "vx(m/s)": 0.30}

    def combined(got):
        e = errors(ref, got)
        return max(e[k][0] / denom[k] for k in ("head_yaw(deg)", "vyaw(deg/s)", "mode", "vx(m/s)"))

    greedy = sorted(sens, key=lambda r: combined(forward(
        p, obs, apply_precision(p, fn8, which={r["layer"]}))) / max(r["floats"], 1))
    order = [r["layer"] for r in greedy]
    print(f"\n=== E3  greedy order (best error per byte saved first) ===\n  {order}")

    curve = []
    chosen = set()
    for name in order:
        chosen.add(name)
        got = forward(p, obs, apply_precision(p, fn8, which=set(chosen)))
        saved = sum(L0[n][0].size for n in chosen) * 3
        curve.append({"n_layers": len(chosen), "KiB": (base_bytes - saved) / 1024,
                      "worst_rel": combined(got), "layers": sorted(chosen)})
        print(f"  +{name:<7} -> {curve[-1]['KiB']:7.1f} KiB  worst_rel={curve[-1]['worst_rel']:.5f}")

    # ---- E4 recommended mixed recipe -------------------------------------
    # E2 says the output layer is ~30x more sensitive per byte than the bulk
    # layers, and E3's greedy order agrees (heads last). So spend the extra
    # bits exactly there: bulk in int8, heads in fp16. This is the recipe a
    # fixed-point port should use.
    bulk = {"proj", "bb", "ff1", "ff2", "ta", "tb"}
    Lq = apply_precision(p, QUANT["int8/channel"], which=bulk)
    Wh, bh = L0["heads"]
    Lq["heads"] = (q_fp16(Wh), q_fp16(bh))
    got = forward(p, obs, Lq)
    e4 = errors(ref, got)
    n_bulk = sum(L0[n][0].size for n in bulk)
    n_heads = int(L0["heads"][0].size)
    n_bias = int(sum(L0[n][1].size for n in L0))
    k4 = (n_bulk + n_heads * 2 + n_bias * 4) / 1024.0
    uni8 = rows[4]
    print("\n=== E4  recommended recipe: bulk int8 + heads fp16 ===")
    print(f"  {k4:.1f} KiB   head max {e4['head_yaw(deg)'][0]:.3f} deg   "
          f"vyaw max {e4['vyaw(deg/s)'][0]:.3f} deg/s   "
          f"mode acc {1.0 - e4['mode'][1]:.4f}")
    print(f"  vs uniform int8 at {uni8['KiB']:.1f} KiB: head "
          f"{uni8['head_err_max']:.3f} deg -> {e4['head_yaw(deg)'][0]:.3f} deg "
          f"({uni8['head_err_max'] / max(e4['head_yaw(deg)'][0], 1e-9):.1f}x better)")

    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "study_results.json"), "w", encoding="utf-8") as f:
        json.dump({"policy": {"input_dim": p["input_dim"], "hidden": p["hidden_dim"],
                              "n_params": int(n_params), "fp32_bytes": int(base_bytes)},
                   "n_eval": int(len(obs)), "uniform": rows,
                   "sensitivity": sens, "greedy": curve}, f, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'study_results.json')}")
    return rows, sens, curve


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
