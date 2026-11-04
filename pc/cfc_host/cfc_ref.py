# -*- coding: utf-8 -*-
"""cfc_ref.py — 通用 CfC 单步前向的纯 numpy 参考实现（零框架依赖，只 import numpy）。

数学口径与 ncps.torch.CfCCell(mode="default", backbone_activation="lecun_tanh")
逐元素一致（CONTRACT §5.2）::

    x     = concat(inp, h)                              # (input_dim + hidden_dim)
    z     = lecun(bb_w @ x + bb_b)                      # (backbone_units)
    ff1   = tanh(ff1_w @ z + ff1_b)                     # (hidden_dim)
    ff2   = tanh(ff2_w @ z + ff2_b)                     # (hidden_dim)
    t     = sigmoid((ta_w @ z + ta_b) * ts + (tb_w @ z + tb_b))
    h_new = ff1 * (1 - t) + t * ff2                     # 没有 h 自环项

保留门（CONTRACT §5.2 的本地扩展，不是 ncps 的一部分）::

    lam   = sigmoid(ret)                                # ret: (hidden_dim,)，初值 -3.0
    h_out = h_new + lam * (h - h_new)

`params["ret"]` 为 None / 缺省 → 不启用保留门，此时 `h_out ≡ h_new`。

约定
----
* 所有数组 float32、行主序 `(out_features, in_features)`，与 `cfc_engine.c` 的
  `const float *` 内存布局一一对应（导出的权重可以直接 memcpy 进固件）。
* `bb_w` 为 None 表示**没有 backbone**（等价 ncps 的 backbone_layers=0，AutoNCP
  的实际形态）：此时 `z = concat(inp, h)`，ff1/ff2/ta/tb 的权重是
  `(hidden_dim, input_dim + hidden_dim)`。
* `inp` 可以是 `(input_dim,)` 或 `(batch, input_dim)`；`h` 同理，两者 rank 必须一致。
* `ts` 优先取函数参数，其次 `params["ts"]`，缺省 1.0。
* 不做任何 clip / 归一化：入参归一化属于网络外层（TECH-PLAN §6.1 的 LayerNorm、
  投影层、head）——那是 cfc_numpy.py 的车赛专用形态，不是通用形态。
"""

import numpy as np

F32 = np.float32

# 参数键名（与 cfc_engine.h / test_engine.c 的读取顺序一致）
WEIGHT_KEYS = (
    "bb_w", "bb_b",
    "ff1_w", "ff1_b",
    "ff2_w", "ff2_b",
    "ta_w", "ta_b",
    "tb_w", "tb_b",
)


# --------------------------------------------------------------------------- #
# 基础算子（与 C 端逐算子对应）
# --------------------------------------------------------------------------- #
def lecun(v):
    """LeCun tanh：1.7159 * tanh(0.666 * v)。

    注意 0.666 **不是** 2/3 —— ncps 源码里就是截断过的 0.666，照抄，不要"修正"。
    """
    return F32(1.7159) * np.tanh(F32(0.666) * v)


def sigmoid(v):
    """1 / (1 + exp(-v))，float32 口径；exp 溢出为 inf 时自然得到 0 / 1。"""
    with np.errstate(over="ignore"):
        return F32(1.0) / (F32(1.0) + np.exp(-v))


def _f32(a):
    return np.ascontiguousarray(a, dtype=F32)


# --------------------------------------------------------------------------- #
# 参数构造
# --------------------------------------------------------------------------- #
def xavier_uniform(rng, shape):
    """等价 torch.nn.init.xavier_uniform_：U(-a, a)，a = sqrt(6/(fan_in+fan_out))。"""
    fan_out, fan_in = shape[0], shape[1]
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return _f32(rng.uniform(-a, a, size=shape))


def linear_bias(rng, shape):
    """等价 nn.Linear 的 bias 初始化：U(-1/sqrt(fan_in), 1/sqrt(fan_in))。"""
    bound = 1.0 / np.sqrt(shape[-1])
    return _f32(rng.uniform(-bound, bound, size=shape))


def make_params(input_dim, hidden_dim, backbone_units, seed=0, ret_init=None, ts=1.0):
    """按 ncps 的初始化分布造一组参数（xavier 权重 + nn.Linear 偏置）。

    backbone_units == 0 时退化成 ncps `backbone_layers=0` 的形态（无 backbone），
    此时 `bb_w`/`bb_b` 为 None，z 直接等于 concat(inp, h)。
    """
    rng = np.random.default_rng(seed)
    z_units = backbone_units if backbone_units > 0 else input_dim + hidden_dim
    p = {}
    if backbone_units > 0:
        p["bb_w"] = xavier_uniform(rng, (backbone_units, input_dim + hidden_dim))
        p["bb_b"] = linear_bias(rng, (backbone_units,))
    else:
        p["bb_w"] = None
        p["bb_b"] = None
    for name in ("ff1", "ff2", "ta", "tb"):
        p[name + "_w"] = xavier_uniform(rng, (hidden_dim, z_units))
        p[name + "_b"] = linear_bias(rng, (hidden_dim,))
    p["ret"] = None if ret_init is None else _f32(np.full(hidden_dim, ret_init))
    p["ts"] = float(ts)
    p["input_dim"] = int(input_dim)
    p["hidden_dim"] = int(hidden_dim)
    p["backbone_units"] = int(backbone_units)
    return p


def init_state(params, batch=1):
    """隐状态清零：h0 = 0。batch=None 返回 1-D。"""
    hidden = params["ff1_b"].shape[0]
    if batch is None:
        return np.zeros(hidden, dtype=F32)
    return np.zeros((batch, hidden), dtype=F32)


def has_retention(params):
    return params.get("ret") is not None


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #
def cfc_step_np(params, inp, h, ts=None):
    """单步前向，返回新隐状态（float32，与 `inp`/`h` 同 batch 形状）。

    参数
    ----
    params : dict
        至少含 ff1/ff2/ta/tb 的 `_w`/`_b`；`bb_w` 为 None 时无 backbone；
        `ret` 非 None 时启用保留门；`ts` 缺省 1.0。
    inp : (input_dim,) 或 (batch, input_dim)
    h   : (hidden_dim,) 或 (batch, hidden_dim)，**不会被修改**
    ts  : 标量，覆盖 `params["ts"]`
    """
    inp = _f32(inp)
    h = _f32(h)
    if inp.ndim != h.ndim:
        raise ValueError(
            f"inp.ndim={inp.ndim} 与 h.ndim={h.ndim} 必须一致（(dim,) 或 (batch, dim)）"
        )
    ts = F32(params.get("ts", 1.0) if ts is None else ts)

    hidden_dim = params["ff1_b"].shape[0]
    in_dim = inp.shape[-1]
    if h.shape[-1] != hidden_dim or h.shape[0] != inp.shape[0]:
        raise ValueError(f"h 形状 {h.shape} 与 (batch={inp.shape[0]}, hidden={hidden_dim}) 不符")

    x = np.concatenate([inp, h], axis=-1)  # (batch, in+hidden)

    bb_w = params.get("bb_w")
    if bb_w is not None:
        z = lecun(x @ _f32(bb_w).T + _f32(params["bb_b"]))
    else:
        z = x  # ncps backbone_layers=0：concat 直接喂给 ff1/ff2/time_a/time_b

    ff1 = np.tanh(z @ _f32(params["ff1_w"]).T + _f32(params["ff1_b"]))
    ff2 = np.tanh(z @ _f32(params["ff2_w"]).T + _f32(params["ff2_b"]))
    ta = z @ _f32(params["ta_w"]).T + _f32(params["ta_b"])
    tb = z @ _f32(params["tb_w"]).T + _f32(params["tb_b"])
    t = sigmoid(ta * ts + tb)

    h_new = ff1 * (F32(1.0) - t) + t * ff2

    ret = params.get("ret")
    if ret is not None:
        lam = sigmoid(_f32(ret))  # (hidden_dim,)
        # lam==0 的单元直接取 h_new：与 cfc_engine.c 一样保证"启用但 λ=0"与
        # "不启用"逐位一致（避免 h_new 为 -0.0 时被 +0.0 翻转符号位）。
        upd = h_new + lam * (h - h_new)
        h_new = np.where(lam != F32(0.0), upd, h_new).astype(F32, copy=False)
    return _f32(h_new)


def cfc_forward_seq(params, inputs, h0=None, ts=None):
    """连续递推：返回 (hs, h_last)。

    inputs 为 `(T, input_dim)` 或 `(T, batch, input_dim)`；`h0` 可为 `(hidden,)` 或
    `(batch, hidden)`。返回的 hs 形状与 inputs 的 rank 对应（T, hidden）或
    （T, batch, hidden），h_last 同理。
    """
    inputs = _f32(inputs)
    if inputs.ndim not in (2, 3):
        raise ValueError(f"inputs 必须是 (T, input_dim) 或 (T, batch, input_dim)，收到 {inputs.shape}")
    x3 = inputs if inputs.ndim == 3 else inputs[:, None, :]
    hidden = params["ff1_b"].shape[0]
    if h0 is None:
        h = np.zeros((x3.shape[1], hidden), dtype=F32)
    else:
        h = _f32(h0)
        if h.ndim == 1:
            h = h[None, :]
        if h.shape != (x3.shape[1], hidden):
            raise ValueError(f"h0 形状 {h.shape} 与 (batch={x3.shape[1]}, hidden={hidden}) 不符")
    T = x3.shape[0]
    hs = np.empty((T, x3.shape[1], hidden), dtype=F32)
    for k in range(T):
        h = cfc_step_np(params, x3[k], h, ts=ts)
        hs[k] = h
    if inputs.ndim == 2:
        return hs[:, 0, :], h[0]
    return hs, h
