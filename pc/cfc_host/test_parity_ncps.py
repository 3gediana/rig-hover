# -*- coding: utf-8 -*-
"""test_parity_ncps.py — numpy ↔ ncps 对拍。

真值唯一来源：`ncps.torch.CfCCell(mode="default", backbone_activation="lecun_tanh")`
（`D:\\smartcar\\.venv\\Lib\\site-packages\\ncps\\torch\\cfc_cell.py`）。

判据（CONTRACT §5.4）：连续递推 200 步后，**最大绝对误差 < 1e-5**。

既可单独跑（`python test_parity_ncps.py`），也被 `test_parity.py` import 当作第一阶段。
"""

import sys

import numpy as np
import torch
from ncps.torch import CfCCell

import cfc_ref

# Windows 控制台默认 GBK：编码不了的符号退化成 '?'，不要让 UnicodeEncodeError 打断验收
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 — 老 Python / 被重定向的流没有 reconfigure
        pass

TOL = 1e-5
N_STEPS = 200

# (名字, input_dim, hidden_dim, backbone_units, backbone_layers)
CONFIGS = [
    ("contract-target 106/64/64", 106, 64, 64, 1),
    ("tiny 3/5/7", 3, 5, 7, 1),
    ("autoncp-no-backbone 8/6/0", 8, 6, 0, 0),
    ("odd-dims 5/33/17", 5, 33, 17, 1),
]


# --------------------------------------------------------------------------- #
# torch 侧：真值
# --------------------------------------------------------------------------- #
def build_cell(input_dim, hidden_dim, backbone_units, backbone_layers, seed=0):
    """造一个 ncps CfCCell，参数分布 = ncps 自己的 init_weights（xavier_uniform 权重）。"""
    torch.manual_seed(seed)
    cell = CfCCell(
        input_dim,
        hidden_dim,
        mode="default",
        backbone_activation="lecun_tanh",
        backbone_units=backbone_units,
        backbone_layers=backbone_layers,
        backbone_dropout=0.0,
    )
    cell.eval()
    return cell


def _np(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)


def params_from_cell(cell):
    """把 ncps cell 的权重导成 cfc_ref 的参数 dict（float32，行主序 (out, in)）。"""
    p = {}
    bb = getattr(cell, "backbone", None)
    if bb is not None and len(bb) > 0:
        lin = bb[0]
        p["bb_w"] = _np(lin.weight)
        p["bb_b"] = _np(lin.bias)
    else:
        p["bb_w"] = None
        p["bb_b"] = None
    for name, mod in (("ff1", cell.ff1), ("ff2", cell.ff2),
                      ("ta", cell.time_a), ("tb", cell.time_b)):
        p[name + "_w"] = _np(mod.weight)
        p[name + "_b"] = _np(mod.bias)
    p["ret"] = None
    p["ts"] = 1.0
    p["input_dim"] = int(cell.input_size)
    p["hidden_dim"] = int(cell.hidden_size)
    p["backbone_units"] = int(p["ff1_w"].shape[1]) if p["bb_w"] is None else int(p["bb_w"].shape[0])
    return p


def params_from_masked_cell(cell):
    """掩码折叠导出：只对 ff1/ff2 折叠，**time_a / time_b 不折**。

    ncps 的 `sparsity_mask` 入参形状是 (in, out)，构造时 `.T` 再取绝对值，
    所以 `cell.sparsity_mask` 形状是 (out, in)、与 weight 同形。

    但 `CfCCell.forward` 里只有 ff1/ff2 走 `F.linear(x, W*mask, b)`，
    time_a / time_b 直接 `self.time_a(x)`、**完全不看 mask**（AutoNCP 的
    时间常数投影因此仍是稠密的）。把 mask 也乘到 ta/tb 上会让 h 直接错
    ~0.6 —— 这一条是被 test_parity.py 的 [2b] 抓出来的，别"修"回去。
    """
    p = params_from_cell(cell)
    if getattr(cell, "sparsity_mask", None) is None:
        return p
    m = _np(cell.sparsity_mask)
    for name in ("ff1", "ff2"):
        w = np.asarray(p[name + "_w"], np.float32) * m
        p[name + "_w"] = np.ascontiguousarray(w, dtype=np.float32)
    return p


def torch_seq(cell, inputs, h0, ts=1.0, ret=None):
    """逐帧调用 ncps cell，返回每步之后的隐状态 (T, hidden) float32。

    `ret` 非 None 时复用同一 cell 的输出再施加保留门（CONTRACT §5.2 的扩展）：
        h_out = h_new + sigmoid(ret) * (hx - h_new)
    这样保留门的公式本身也能进 torch ↔ numpy ↔ C 三方对拍。
    """
    hs = []
    lam = None if ret is None else torch.sigmoid(torch.from_numpy(np.asarray(ret, np.float32)))
    with torch.no_grad():
        h = torch.from_numpy(np.ascontiguousarray(h0, dtype=np.float32))
        for k in range(inputs.shape[0]):
            x = torch.from_numpy(np.ascontiguousarray(inputs[k:k + 1], dtype=np.float32))
            hx = h
            h, _ = cell(x, hx, ts)
            if lam is not None:
                h = h + lam * (hx - h)
            hs.append(h.numpy()[0].copy())
    return np.stack(hs)


# --------------------------------------------------------------------------- #
# 对拍
# --------------------------------------------------------------------------- #
def make_inputs(T, input_dim, seed, kind="normal"):
    rng = np.random.default_rng(seed)
    if kind == "normal":
        return np.asarray(rng.standard_normal((T, input_dim)), dtype=np.float32)
    if kind == "zeros":
        return np.zeros((T, input_dim), dtype=np.float32)
    if kind == "ones":
        return np.ones((T, input_dim), dtype=np.float32)
    if kind == "sat":  # tanh 饱和：正负交替 ±10
        return np.where((np.arange(T * input_dim) % 2).reshape(T, input_dim),
                        np.float32(10.0), np.float32(-10.0)).astype(np.float32)
    raise ValueError(kind)


def run_case(name, input_dim, hidden_dim, backbone_units, backbone_layers,
             steps=N_STEPS, seed=0, ts=1.0, ret=None, input_kind="normal", verbose=True):
    """跑一个配置的 ncps ↔ numpy 对拍，返回 (max_err, hs_np, params, cell)。"""
    cell = build_cell(input_dim, hidden_dim, backbone_units, backbone_layers, seed=seed)
    params = params_from_cell(cell)
    if ret is None:
        params["ret"] = None
    else:
        params["ret"] = cfc_ref._f32(np.asarray(ret, dtype=np.float32))
    params["ts"] = float(ts)

    inputs = make_inputs(steps, input_dim, seed + 1000, kind=input_kind)
    h0 = np.zeros((1, hidden_dim), dtype=np.float32)

    hs_ref = torch_seq(cell, inputs, h0, ts=ts, ret=params["ret"])
    hs_np, _ = cfc_ref.cfc_forward_seq(params, inputs, h0=h0, ts=ts)

    per_step = np.abs(hs_ref - hs_np).reshape(steps, -1).max(axis=1)
    max_err = float(per_step.max())
    if verbose:
        print(f"  {name:<28} steps={steps:<4} ts={ts:<5g} "
              f"max_abs_err={max_err:.3e}  {'PASS' if max_err < TOL else 'FAIL'}")
    return max_err, hs_np, per_step, params, cell, inputs


def run_masked_case(seed=5, steps=N_STEPS, keep=0.22, verbose=True):
    """AutoNCP 形态：无 backbone + 稀疏掩码，验证"掩码折进权重"的导出配方。

    返回 (max_err, hs_ref, params, inputs, h0)。ncps 侧是真值，numpy 侧用
    折叠后的稠密权重 —— 两者一致就说明固件不需要任何 mask 字段。
    """
    input_dim, hidden_dim = 12, 9
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cat = input_dim + hidden_dim
    # 入参形状 (in, out)；内部会 .T 成 (out, in) 与 weight 相乘
    m_in_out = (rng.random((cat, hidden_dim)) < keep).astype(np.float32)
    for j in range(hidden_dim):  # 保证每个输出单元至少一条输入连接，避免退化
        m_in_out[int(rng.integers(0, cat)), j] = 1.0

    cell = CfCCell(input_dim, hidden_dim, mode="default",
                   backbone_activation="lecun_tanh", backbone_units=0,
                   backbone_layers=0, sparsity_mask=m_in_out)
    cell.eval()

    params = params_from_masked_cell(cell)
    zero_frac = float(np.mean(params["ff1_w"] == 0.0))
    inputs = make_inputs(steps, input_dim, seed + 77)
    h0 = np.zeros((1, hidden_dim), dtype=np.float32)
    hs_ref = torch_seq(cell, inputs, h0)
    hs_np, _ = cfc_ref.cfc_forward_seq(params, inputs, h0=h0)
    err = float(np.max(np.abs(hs_ref - hs_np)))
    if verbose:
        print(f"  {'autoncp-mask 12/9/0':<28} steps={steps:<4} ts=1     "
              f"max_abs_err={err:.3e}  {'PASS' if err < TOL else 'FAIL'}  "
              f"(权重零占比 {zero_frac:.0%})")
    return err, hs_ref, params, inputs, h0


# --------------------------------------------------------------------------- #
# AutoNCP wiring（多层串联）—— 部署形态，见 test_parity.py [2c]
# --------------------------------------------------------------------------- #
def build_wired(input_dim=12, units=64, out=2, seed=11):
    """AutoNCP wiring + WiredCfCCell。

    注意：AutoNCP(units, out) 的 num_layers 不是 1 —— 例如 units=64/out=2 会分成
    [38, 24, 2] 三层，即**三个串联的 CfCCell**。固件必须链 3 个 cfc_cell_t。
    """
    from ncps.torch import WiredCfCCell
    from ncps.wirings import AutoNCP

    torch.manual_seed(seed)
    wiring = AutoNCP(units, out)
    wiring.build(input_dim)
    cell = WiredCfCCell(input_dim, wiring, mode="default")
    cell.eval()
    return cell, wiring


def wired_layers(cell):
    """返回 [(params, n_units)]：每层一个 CfCCell 的参数（掩码已折进 ff1/ff2）。"""
    return [(params_from_masked_cell(cell._layers[i]), int(cell._layers[i].hidden_size))
            for i in range(cell.num_layers)]


def torch_wired_seq(cell, inputs, h0):
    """WiredCfCCell 逐帧前向，返回每步之后的**全层拼接**状态 (T, units)。"""
    hs = []
    with torch.no_grad():
        h = torch.from_numpy(np.ascontiguousarray(h0, dtype=np.float32))
        for k in range(inputs.shape[0]):
            x = torch.from_numpy(np.ascontiguousarray(inputs[k:k + 1], dtype=np.float32))
            _, h = cell(x, h, 1.0)
            hs.append(h.numpy()[0].copy())
    return np.stack(hs)


def np_wired_chain(layers, inputs, ts=1.0):
    """numpy 侧串联：第 i 层的输入是第 i-1 层的 h，各层各自从 0 起递推。"""
    T = inputs.shape[0]
    states = [np.zeros((1, n), dtype=np.float32) for _, n in layers]
    outs = []
    for k in range(T):
        x = inputs[k:k + 1]
        new_states = []
        for (p, _n), s in zip(layers, states):
            s2 = cfc_ref.cfc_step_np(p, x, s, ts=ts)
            new_states.append(s2)
            x = s2
        states = new_states
        outs.append(np.concatenate(states, axis=1)[0])
    return np.stack(outs)


def run_parity(verbose=True):
    """所有 ncps ↔ numpy 用例。返回 [(name, ok, max_err, per_step)]。"""
    results = []
    if verbose:
        print("[1] numpy <-> ncps  (判据 max_abs_err < %.0e)" % TOL)
    for name, i_d, h_d, b_u, b_l in CONFIGS:
        err, _, per_step, _, _, _ = run_case(name, i_d, h_d, b_u, b_l, verbose=verbose)
        results.append((f"ncps<->numpy {name}", err < TOL, err, per_step))

    # ts 不是 1 的情形（ncps 的 ts 只乘 time_a 那一半）
    for ts in (0.25, 3.0):
        err, _, per_step, _, _, _ = run_case(f"contract-target ts={ts}", 106, 64, 64, 1,
                                             ts=ts, verbose=verbose)
        results.append((f"ncps<->numpy ts={ts}", err < TOL, err, per_step))

    # AutoNCP 形态（无 backbone + 稀疏掩码折叠导出）
    err_mask, _, _, _, _ = run_masked_case(verbose=verbose)
    results.append(("ncps<->numpy autoncp-mask", err_mask < TOL, err_mask, None))
    return results


def main():
    ok_all = True
    for name, ok, err, _ in run_parity(verbose=True):
        ok_all &= ok
        if not ok:
            print(f"  FAIL {name}: max_abs_err={err:.3e} >= {TOL:.0e}")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
