# -*- coding: utf-8 -*-
"""quant_explore.py -- 在真实会话上比较几种 int8 量化变体的端到端误差。"""
import sys

sys.path.insert(0, r"D:\robot_work\pc\train")
import numpy as np
import export_blob

pol = export_blob.load_policy_npz(r"D:\robot_work\pc\train\policy.npz")


def g(k):
    return np.asarray(pol[k], dtype=np.float32)


in_dim = int(pol["input_dim"])
cell_in = int(pol["cell_in_dim"])
hid = int(pol["hidden_dim"])
bu = int(pol["backbone_units"])
W = {
    "bb": g("bb_w").reshape(bu, cell_in + hid),
    "ff1": g("ff1_w").reshape(hid, bu),
    "ff2": g("ff2_w").reshape(hid, bu),
    "ta": g("ta_w").reshape(hid, bu),
    "tb": g("tb_w").reshape(hid, bu),
    "proj": g("proj_w").reshape(cell_in, in_dim),
}
B = {k: g(k + "_b") for k in W}
ret = g("ret")
ts = float(pol["ts"])
mean = g("input_mean")
std = g("input_std")
hw = g("head_w").reshape(-1)
hb = g("head_b").reshape(-1)
HD = [1, 1, 4, 8, 1]
HA = [1, 1, 2, 2, 1]
z = np.load(r"D:\robot_work\pc\train\data\synth_s007.npz")["obs"].astype(np.float32)


def qrow(Wm):
    am = np.abs(Wm).max(axis=1)
    s = np.where(am > 0, am / 127.0, 1.0).astype(np.float32)
    r = Wm / s[:, None]
    q = np.clip(np.trunc(r + np.copysign(np.float32(0.5), r)), -127, 127).astype(np.int8)
    return q, s


WQ = {k: qrow(v) for k, v in W.items()}


def q8(v):
    am = np.abs(v).max()
    if am <= 0:
        return np.zeros(v.shape, dtype=np.int8), 0.0
    s = am / 127.0
    r = v / s
    q = np.clip(np.trunc(r + np.copysign(np.float32(0.5), r)), -127, 127).astype(np.int8)
    return q, np.float32(s)


def q16(v):
    """int8 高位 + int8 低位：等效 ~15 bit 激活。"""
    am = np.abs(v).max()
    if am <= 0:
        return (np.zeros(v.shape, np.int8), np.zeros(v.shape, np.int8), 0.0)
    s = am / 127.0
    r = v / s
    hi = np.trunc(r)
    lo = np.trunc((r - hi) * 128.0)
    hi = np.clip(hi, -127, 127).astype(np.int8)
    lo = np.clip(lo, -127, 127).astype(np.int8)
    return hi, lo, np.float32(s)


CASES = ("f32", "W8", "A8", "W8A8", "W8A16")
NFR = z.shape[0]


def run(mode):
    h = np.zeros(hid, dtype=np.float32)
    out = []
    for t in range(NFR):
        x = np.clip((z[t] - mean) / std, -4, 4)

        def dot(name, vec):
            Wm = W[name]
            q, s = WQ[name]
            if mode == "f32":
                return Wm @ vec + B[name]
            if mode == "W8":
                return (q.astype(np.float32) @ vec) * s + B[name]
            if mode == "A8":
                qx, sx = q8(vec)
                return Wm @ (qx.astype(np.float32) * sx) + B[name]
            if mode == "W8A8":
                qx, sx = q8(vec)
                acc = (q.astype(np.int32) @ qx.astype(np.int32)).astype(np.float32)
                return acc * s * sx + B[name]
            if mode == "W8A16":
                hi, lo, sx = q16(vec)
                qi = q.astype(np.int32)
                acc = (qi @ hi.astype(np.int32)).astype(np.float32) + \
                      (qi @ lo.astype(np.int32)).astype(np.float32) / 128.0
                return acc * s * sx + B[name]
            raise ValueError(mode)

        zp = dot("proj", x)
        zp = zp / (1.0 + np.exp(-zp))
        xc = np.concatenate([zp, h])
        zz = 1.7159 * np.tanh(0.666 * dot("bb", xc))
        a1 = dot("ff1", zz)
        a2 = dot("ff2", zz)
        ta = dot("ta", zz)
        tb = dot("tb", zz)
        tt = 1.0 / (1.0 + np.exp(-(ta * ts + tb)))
        hn = np.tanh(a1) * (1 - tt) + tt * np.tanh(a2)
        lam = 1.0 / (1.0 + np.exp(-ret))
        h = (hn + lam * (h - hn)).astype(np.float32)
        o = []
        cw = 0
        cb = 0
        for i, d in enumerate(HD):
            w = hw[cw:cw + d * hid].reshape(d, hid)
            cw += d * hid
            b = hb[cb:cb + d]
            cb += d
            lg = w @ h + b
            o.append(np.tanh(lg) if HA[i] == 1 else lg)
        out.append(np.concatenate(o))
    return np.array(out)


def softmax(a):
    e = np.exp(a - a.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    ref = run("f32")
    print("%-8s %10s %10s %10s %10s" % ("mode", "yaw", "vyaw", "vx", "softmax_p"))
    for mode in CASES:
        if mode == "f32":
            continue
        v = run(mode)
        e = np.abs(v - ref)
        sp = 0.0
        for (o, d) in ((2, 4), (3, 8)):
            sp = max(sp, float(np.abs(softmax(ref[:, o:o + d]) - softmax(v[:, o:o + d])).max()))
        print("%-8s %10.3e %10.3e %10.3e %10.3e"
              % (mode, e[:, 0].max(), e[:, 1].max(), e[:, 4].max(), sp))
