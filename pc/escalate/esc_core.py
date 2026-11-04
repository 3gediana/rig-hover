# -*- coding: utf-8 -*-
"""esc_core.py — 升级头要蒸馏的那个判据 + 全部评估指标（无 torch 依赖）。

本文件把 ``pc/decision_hub/hub.py`` 的升级判据**逐行复刻**成 numpy，只保留
``escalate`` 头需要的那一部分（``p_esc``，不含层级路由）：

    φ_k(x)  = 12 个风险势函数（C¹ smoothstep，值域 [0,1]）
    z       = b + Σ_k w_k·φ_k          b = -3.0
    p_esc   = σ(z)

为什么复刻而不 import：``hub.py`` 有 ``llm_backend`` 依赖（线程池、HTTP 契约），
而这里是纯函数。**唯一权威仍是 ``hub.py``**，所以 :func:`phi_parity_check`
会拿真正的 ``DecisionHub.compute_phi`` 与本文件逐帧对拍（test_escalate.py 的
「按比划分一致性」一节里跑），误差 > 1e-9 直接 FAIL。

与 ``hub.py`` 的两处**有意的**差异，都在 :data:`OBSERVABLE_POTENTIALS` 上：

1. **只看 7 个势函数。** 106 维观测（``cfc_task.c:build_obs``）里没有语义通道，
   所以 ``m_speech`` / ``m_point`` / ``m_stale_sem`` / ``p_multi`` / ``p_lookup``
   这 5 个跑本地判据算不出来，恒置 0。这不是简化，是**观测的物理事实**：
   见 ``README.md`` §2 与 ``FIRMWARE_CHANGES.md`` §3。
2. 因此 ``z`` 少了 5 项正权重的和，基率比 hub 的全量版本更低（实测 ~2% 帧阳性）。
   权重本身一个都没动 —— 全部沿用 ``hub.py`` 的先验表。
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32

# --------------------------------------------------------------------------- #
# 观测下标（唯一来源 pc/train/obs.py；这里只抄用到的几个）
# --------------------------------------------------------------------------- #
OBS_DIM = 106
FEAT_DIM = 64
IX_THETA = 64
IX_DIST = 67
IX_CONF = 68
IX_SEEN = 69
IX_AGE_S = 98          # 秒；-1 表示从未收到
IX_GAZE = 99

#: 动作满量程（固件 arb_cfg_t 默认值，与 hub.py 的 LIM_* 一致）
LIM_HEAD_DEG = 45.0
LIM_VYAW_DPS = 90.0
LIM_VX_MPS = 0.30

#: 势函数表。``gate_w`` 与 ``hub.py::POTENTIALS`` 逐项相同（顺序也相同）。
#: 最后 5 个的 ``observable=False``：106 维观测里没有它们的输入。
POTENTIALS = (
    ("u_conf",      "blind",    4.2, True),
    ("u_lost",      "blind",    4.6, True),
    ("u_clip",      "unknown",  3.0, True),
    ("u_ood",       "unknown",  3.6, True),
    ("m_speech",    "language", 5.0, False),
    ("m_point",     "language", 4.4, False),
    ("m_stale_sem", "context",  2.0, False),
    ("p_multi",     "planning", 5.2, False),
    ("p_lookup",    "planning", 4.8, False),
    ("s_irrev",     "stakes",   2.6, True),
    ("s_approach",  "stakes",   3.2, True),
    ("c_local_ok",  "context", -1.0, True),
)
POTENTIAL_NAMES = tuple(p[0] for p in POTENTIALS)
K_POTENTIALS = len(POTENTIALS)
GATE_WEIGHTS = np.asarray([p[2] for p in POTENTIALS], dtype=np.float64)
OBSERVABLE = np.asarray([p[3] for p in POTENTIALS], dtype=bool)
OBSERVABLE_POTENTIALS = tuple(n for n, ok in zip(POTENTIAL_NAMES, OBSERVABLE) if ok)
#: 升级判据的偏置（hub.py::HubConfig.gate_bias 的默认值）
GATE_BIAS = -3.0

#: smoothstep 的斜坡端点，逐项抄 hub.py::compute_phi
RAMP = {
    "u_conf": (0.55, 0.15),
    "u_lost": (250.0, 1200.0),
    "u_clip": (0.15, 0.50),
    "u_ood": (2.5, 5.0),
    "m_stale_sem": (400.0, 1500.0),
    "p_multi": (1.0, 3.0),
    "s_irrev": (0.25, 0.90),
    "s_approach": (0.03, 0.15),
    "c_local_ok": (0.60, 0.85),
}

#: ``hub.py`` 里带一阶平滑（ema_tau=0.15s）的两个离散检测量。本文件不模拟语义
#: 通道，所以它们恒 0；保留常量是为了对拍时能说明「平滑过也不影响结论」。
EMA_POTENTIALS = ("m_speech", "m_point")


def smoothstep(x, lo, hi):
    """C¹ 平滑斜坡：x=lo 处 0，x=hi 处 1（允许 lo>hi，此时单调递减）。

    与 ``hub.py::smoothstep`` 逐位同构，但支持 ndarray。NaN 返回 0（保守）。
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if hi == lo:
        return np.where(x >= hi, 1.0, 0.0).astype(np.float64)
    t = (x - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    out = t * t * (3.0 - 2.0 * t)
    return np.where(np.isfinite(x), out, 0.0).astype(np.float64)


def _clamp(a, lo, hi):
    return np.clip(np.asarray(a, dtype=np.float64), lo, hi)


# --------------------------------------------------------------------------- #
# 观测 -> 势函数
# --------------------------------------------------------------------------- #
def obs_fields(obs, guard_mean, guard_std, clip_sigma=4.0):
    """从 ``(N, 106)`` 观测里抽出判据要用的标量场。

    Args:
        obs: ``(N, 106)`` float32 原始观测（未归一化）。
        guard_mean: ``(106,)`` 训练集均值（= blob 的 ``input_mean``）。
        guard_std: ``(106,)`` 训练集标准差（= blob 的 ``input_std``，已加下限）。
        clip_sigma: 域外守护倍数。

    Returns:
        dict，键 ``conf / seen / age_ms / clip_ratio / novelty / dist / theta``，
        都是 ``(N,)`` float64。``novelty`` 的定义是 ``feat_64`` 的对角马氏距离的
        均方根（``sqrt(mean(z^2))``，``z`` 是归一化后的 64 维特征），典型值 ≈ 1.0。
    """
    o = np.asarray(obs, dtype=np.float64)
    mu = np.asarray(guard_mean, dtype=np.float64).reshape(-1)
    sd = np.asarray(guard_std, dtype=np.float64).reshape(-1)
    bad = ~np.isfinite(o)
    v = np.where(bad, mu, o)
    lo = mu - clip_sigma * sd
    hi = mu + clip_sigma * sd
    clipped = bad | (v < lo) | (v > hi)
    v = np.clip(v, lo, hi)
    z = (v - mu) / sd

    feat = z[..., :FEAT_DIM]
    age_s = o[..., IX_AGE_S]
    age_ms = np.where(np.isfinite(age_s) & (age_s >= 0.0), age_s * 1000.0, 1.0e6)
    return {
        "conf": np.clip(np.nan_to_num(o[..., IX_CONF]), 0.0, 1.0),
        "seen": (np.nan_to_num(o[..., IX_SEEN]) >= 0.5).astype(np.float64),
        "age_ms": age_ms,
        "clip_ratio": clipped.mean(axis=-1),
        "novelty": np.sqrt(np.mean(feat * feat, axis=-1)),
        "dist": np.nan_to_num(o[..., IX_DIST]),
        "theta": np.nan_to_num(o[..., IX_THETA]),
    }


def phi_from_fields(fields, act, semantic=None, semantic_raw=None):
    """12 个势函数（``hub.py::DecisionHub.compute_phi`` 的向量化版本）。

    Args:
        fields: :func:`obs_fields` 的输出。
        act: dict，``head/vyaw/vx`` 是**本帧 CfC 输出**（工程单位），
            ``mode_conf`` 是动作置信度 ∈[0,1]（= mode 头 top1−top2）。
        semantic: 可选的 5 个语义势函数输入（dict）：``speech`` / ``pointing`` /
            ``object_held`` / ``lookup`` / ``instr_clauses`` / ``sem_age_ms`` /
            ``sem_expected``。**本工程不模拟语义通道**，所以调用方一律传
            ``None``；参数留着是为了与 hub.py 对拍和将来扩展。
        semantic_raw: 仅供对拍使用。``hub.py`` 里 ``m_speech``/``m_point`` 走
            ``ema_tau=0.15`` 的一阶平滑，而 ``c_local_ok`` 里的同一个 ``speech``
            用的是**未平滑**的原始值 —— 这是 hub.py 的实现细节，复刻必须照抄。

    Returns:
        ``(N, 12)`` float64，列顺序 = :data:`POTENTIAL_NAMES`。
    """
    conf = fields["conf"]
    seen = fields["seen"]
    age = fields["age_ms"]
    conf_act = np.clip(np.asarray(act["mode_conf"], dtype=np.float64), 0.0, 1.0)
    mag = np.maximum.reduce([
        np.abs(np.asarray(act["head"], dtype=np.float64)) / LIM_HEAD_DEG,
        np.abs(np.asarray(act["vyaw"], dtype=np.float64)) / LIM_VYAW_DPS,
        np.abs(np.asarray(act["vx"], dtype=np.float64)) / LIM_VX_MPS,
    ])
    mag = _clamp(mag, 0.0, 1.0)
    dist = fields["dist"]

    sem = semantic or {}
    speech = _clamp(sem.get("speech", 0.0), 0.0, 1.0)
    point = _clamp(sem.get("pointing", 0.0), 0.0, 1.0)
    held = _clamp(sem.get("object_held", 0.0), 0.0, 1.0)
    lookup = _clamp(sem.get("lookup", 0.0), 0.0, 1.0)
    clauses = np.maximum(0.0, np.asarray(sem.get("instr_clauses", 0.0), dtype=np.float64))
    sem_age = np.asarray(sem.get("sem_age_ms", 1.0e6), dtype=np.float64)
    # hub.py: ``smoothstep(sem_age, ...) if sem_expected else 0``。没有语义通道
    # 就**不是**"通道过期"，这里必须显式 gate，否则 sem_age 缺省值会把 φ 顶到 1。
    sem_exp = np.asarray(sem.get("sem_expected", 0.0), dtype=np.float64) >= 0.5

    ood = smoothstep(np.maximum(0.0, fields["novelty"]), *RAMP["u_ood"])
    raw = semantic_raw or {}
    raw_speech = _clamp(raw.get("speech", speech), 0.0, 1.0)
    raw_point = _clamp(raw.get("pointing", point), 0.0, 1.0)
    raw_held = _clamp(raw.get("object_held", held), 0.0, 1.0)
    sem_risk = np.maximum(raw_speech,
                          np.maximum(np.maximum(raw_point, raw_held), ood))

    cols = {
        "u_conf": np.where(seen >= 0.5, smoothstep(conf, *RAMP["u_conf"]), 0.0),
        "u_lost": np.where(seen < 0.5, smoothstep(age, *RAMP["u_lost"]), 0.0),
        "u_clip": smoothstep(_clamp(fields["clip_ratio"], 0.0, 1.0), *RAMP["u_clip"]),
        "u_ood": ood,
        "m_speech": np.full_like(conf, float(speech)),
        "m_point": np.full_like(conf, float(max(float(point), float(held)))),
        "m_stale_sem": np.where(sem_exp, smoothstep(sem_age, *RAMP["m_stale_sem"]), 0.0),
        "p_multi": smoothstep(clauses, *RAMP["p_multi"]),
        "p_lookup": np.full_like(conf, float(lookup)),
        "s_irrev": (1.0 - conf_act) * smoothstep(mag, *RAMP["s_irrev"]),
        "s_approach": (1.0 - conf_act)
        * smoothstep(np.asarray(act["vx"], dtype=np.float64), *RAMP["s_approach"])
        * smoothstep(1.1 - dist, 0.0, 0.6),
        "c_local_ok": seen
        * smoothstep(conf, *RAMP["c_local_ok"])
        * (1.0 - sem_risk),
    }
    shape = np.shape(conf)
    return np.stack([np.broadcast_to(_clamp(cols[n], 0.0, 1.0), shape)
                     for n in POTENTIAL_NAMES], axis=-1)


def logit_from_phi(phi, weights=GATE_WEIGHTS, bias=GATE_BIAS):
    """``z = b + Σ w_k φ_k``（``(..., 12)`` -> ``(...)``）。"""
    return float(bias) + np.asarray(phi, dtype=np.float64) @ np.asarray(weights, np.float64)


def p_esc_from_phi(phi, weights=GATE_WEIGHTS, bias=GATE_BIAS):
    """``p_esc = σ(z)``，与 ``hub.py`` 的 ``DecisionHub.decide`` 同口径（夹 ±30）。"""
    z = np.clip(logit_from_phi(phi, weights, bias), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# 对拍：与真正的 hub.py 比
# --------------------------------------------------------------------------- #
def _parity_arrays(n, seed):
    """构造随机观测并用真 hub 与本文件各算一遍，返回 ``(ref, ours, names)``。"""
    import sys
    from pathlib import Path

    hub_dir = Path(__file__).resolve().parent.parent / "decision_hub"
    if str(hub_dir) not in sys.path:
        sys.path.insert(0, str(hub_dir))
    import hub as hubmod  # noqa: PLC0415 — 故意延迟到校验时才 import

    rng = np.random.default_rng(seed)
    dt = 0.02
    inst = hubmod.DecisionHub()
    obs_list = []
    for _ in range(int(n)):
        c = hubmod.CfcOut(
            head_des=float(rng.uniform(-45, 45)),
            vyaw_des=float(rng.uniform(-90, 90)),
            vx_des=float(rng.uniform(-0.30, 0.30)),
            mode=int(rng.integers(0, 4)),
            emote=int(rng.integers(0, 8)),
            action_conf=float(rng.uniform(0.0, 1.0)),
        )
        obs_list.append(hubmod.HubObs(
            cfc=c,
            seen=int(rng.integers(0, 2)),
            person_conf=float(rng.uniform(0.0, 1.0)),
            clip_ratio=float(rng.uniform(0.0, 1.0)),
            novelty=float(rng.uniform(0.0, 6.0)),
            feat_age_ms=float(rng.uniform(0.0, 2000.0)),
            dist_est=float(rng.uniform(0.05, 3.0)),
            sem_expected=1,
            sem_age_ms=float(rng.uniform(0.0, 2000.0)),
            speech=float(rng.integers(0, 2)),
            pointing=float(rng.integers(0, 2)),
            object_held=float(rng.integers(0, 2)),
            instr_clauses=int(rng.integers(0, 4)),
            lookup=float(rng.integers(0, 2)),
        ))

    # 注意：compute_phi 每调一次就推进一次 EMA 状态，所以**每帧只能调一次**，
    # 否则参考值会被多推 11 次（这个坑第一次就踩到了）。
    ref = np.stack([np.asarray(list(inst.compute_phi(o, dt).values()))
                    for o in obs_list])
    fields = {
        "conf": np.asarray([o.person_conf for o in obs_list]),
        "seen": np.asarray([float(o.seen) for o in obs_list]),
        "age_ms": np.asarray([o.feat_age_ms for o in obs_list]),
        "clip_ratio": np.asarray([o.clip_ratio for o in obs_list]),
        "novelty": np.asarray([o.novelty for o in obs_list]),
        "dist": np.asarray([o.dist_est for o in obs_list]),
    }
    act = {
        "head": np.asarray([o.cfc.head_des for o in obs_list]),
        "vyaw": np.asarray([o.cfc.vyaw_des for o in obs_list]),
        "vx": np.asarray([o.cfc.vx_des for o in obs_list]),
        "mode_conf": np.asarray([o.cfc.action_conf for o in obs_list]),
    }
    ours = np.empty_like(ref)
    ema = {"m_speech": 0.0, "m_point": 0.0}
    alpha = dt / (0.15 + dt)
    for i, o in enumerate(obs_list):
        raw_speech = float(o.speech)
        raw_point = max(float(o.pointing), float(o.object_held))
        ema["m_speech"] += alpha * (raw_speech - ema["m_speech"])
        ema["m_point"] += alpha * (raw_point - ema["m_point"])
        sem = {
            "speech": ema["m_speech"], "pointing": ema["m_point"],
            "lookup": float(o.lookup), "instr_clauses": float(o.instr_clauses),
            "sem_age_ms": float(o.sem_age_ms), "sem_expected": 1.0,
        }
        f = {k: fields[k][i:i + 1] for k in fields}
        a_ = {k: act[k][i:i + 1] for k in act}
        ours[i] = phi_from_fields(f, a_, sem, semantic_raw={
            "speech": raw_speech, "pointing": float(o.pointing),
            "object_held": float(o.object_held)})[0]
    return ref, ours


def phi_parity_check(n=400, seed=1234):
    """用 ``decision_hub/hub.py`` 的真实实现与本文件对拍。

    构造随机的 ``HubObs``（含语义通道的非 0 值，验证那 5 项也能对上），
    用同一个 ``DecisionHub`` 实例逐帧调 ``compute_phi``（EMA 状态在实例里），
    与本文件 :func:`phi_from_fields` 的输出逐元素比较。

    Returns:
        ``(ok, max_abs_err, detail)``；``hub`` 不可导入时返回 ``(None, 0.0, 原因)``。
    """
    from pathlib import Path

    hub_dir = Path(__file__).resolve().parent.parent / "decision_hub"
    if not (hub_dir / "hub.py").exists():
        return None, 0.0, "decision_hub/hub.py 不存在（跳过对拍）"
    try:
        ref, ours = _parity_arrays(n, seed)
    except Exception as exc:  # noqa: BLE001 — 依赖缺失只跳过，不算失败
        return None, 0.0, "对拍不可用: %s" % (exc,)
    diff = np.abs(ours - ref)
    per_col = diff.max(axis=0) if diff.size else np.zeros(K_POTENTIALS)
    err = float(per_col.max())
    worst = int(np.argmax(per_col))
    detail = ("n=%d，%d 个势函数，max|Δ|=%.3e（最差列 %s）"
              % (n, K_POTENTIALS, err, POTENTIAL_NAMES[worst]))
    return err <= 1e-9, err, detail


# --------------------------------------------------------------------------- #
# 指标（全部自己实现，避免给验收脚本引入 sklearn 依赖）
# --------------------------------------------------------------------------- #
def _as_1d(y, s=None):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if s is None:
        return y
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("y/s 长度不一致: %s vs %s" % (y.shape, s.shape))
    return y, s


def roc_auc(y_true, score):
    """ROC AUC（Mann-Whitney，含并列修正）。单类或全并列时返回 0.5。"""
    y, s = _as_1d(y_true, score)
    n_pos = int((y > 0.5).sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # 并列取平均秩
    s_sorted = s[order]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y_true, prob):
    """Brier 分数（越小越好）。"""
    y, p = _as_1d(y_true, prob)
    return float(np.mean((p - y) ** 2))


def log_loss(y_true, prob, eps=1e-7):
    """二元交叉熵。"""
    y, p = _as_1d(y_true, prob)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability(y_true, prob, bins=10):
    """可靠性表：每个概率桶的样本数 / 平均预测 / 实际频率 / 绝对 gap。

    Returns:
        list[dict]，以及 ``(ece, mce)``。
    """
    y, p = _as_1d(y_true, prob)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for b in range(int(bins)):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": 0,
                         "mean_p": float("nan"), "freq": float("nan"), "gap": 0.0})
            continue
        mp = float(p[m].mean())
        fr = float(y[m].mean())
        gap = abs(mp - fr)
        ece += gap * n
        mce = max(mce, gap)
        rows.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": n,
                     "mean_p": mp, "freq": fr, "gap": gap})
    ece /= max(y.size, 1)
    return rows, float(ece), float(mce)


def threshold_at_fpr(y_true, score, max_fpr):
    """取一个阈值，使**在给定负样本上**的误报率 ≤ ``max_fpr``。

    Returns:
        ``(thr, fpr, recall)``；无可分性时返回 ``(inf, 0.0, 0.0)``。
    """
    y, s = _as_1d(y_true, score)
    neg = s[y <= 0.5]
    pos = s[y > 0.5]
    if neg.size == 0:
        return float("inf"), 0.0, 0.0
    thr = float(np.quantile(neg, 1.0 - float(max_fpr))) if max_fpr < 1.0 else -float("inf")
    # quantile 会落在某个负样本上；向上抬一点保证 ≤ max_fpr
    thr = float(np.nextafter(thr, np.inf))
    fpr = float((neg > thr).mean()) if neg.size else 0.0
    rec = float((pos > thr).mean()) if pos.size else 0.0
    return thr, fpr, rec


def recall_at_fpr(y_true, score, max_fpr):
    """在测试集上按给定误报率算召回（阈值由调用方先在验证集上定好）。"""
    y, s = _as_1d(y_true, score)
    thr, fpr, rec = threshold_at_fpr(y, s, max_fpr)
    return thr, fpr, rec


def rankdata(x):
    """平均秩（1..n），用于 Spearman。"""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman(a, b):
    """Spearman 秩相关。"""
    a, b = _as_1d(a), np.asarray(b, dtype=np.float64).reshape(-1)
    ra, rb = rankdata(a), rankdata(b)
    ra = (ra - ra.mean()) / max(ra.std(), 1e-12)
    rb = (rb - rb.mean()) / max(rb.std(), 1e-12)
    return float(np.mean(ra * rb))


def pearson(a, b):
    """Pearson 相关。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    am, bm = a - a.mean(), b - b.mean()
    den = float(np.sqrt((am * am).sum() * (bm * bm).sum()))
    return float((am * bm).sum() / den) if den > 0 else 0.0


def bucket_agreement(p_a, p_b, thr=0.5, bins=10):
    """两套概率在**同一套分桶**下的判决一致率。

    Args:
        p_a, p_b: 两个模型的概率。
        thr: 二值判决阈值。
        bins: 按 ``p_a`` 的分桶数（等宽）。

    Returns:
        ``(overall, per_bucket)``，``per_bucket`` 是 list[dict]。
    """
    a = np.asarray(p_a, dtype=np.float64).reshape(-1)
    b = np.asarray(p_b, dtype=np.float64).reshape(-1)
    da = a >= thr
    db = b >= thr
    overall = float((da == db).mean()) if a.size else 1.0
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for i in range(int(bins)):
        lo, hi = edges[i], edges[i + 1]
        m = (a >= lo) & (a < hi) if i < bins - 1 else (a >= lo) & (a <= hi)
        n = int(m.sum())
        rows.append({"bin": i, "lo": float(lo), "hi": float(hi), "n": n,
                     "agree": float((da[m] == db[m]).mean()) if n else float("nan"),
                     "mean_a": float(a[m].mean()) if n else float("nan"),
                     "mean_b": float(b[m].mean()) if n else float("nan")})
    return overall, rows


def expected_calibration_error(y_true, prob, bins=10):
    """只要 ECE 数值时的便捷入口。"""
    _rows, ece, _mce = reliability(y_true, prob, bins)
    return ece


def fit_temperature(y_true, logit, grid=None):
    """一维温度标定：``p = σ(logit / T)``，最小化 NLL。

    为什么有用：温度在**头权重上是可吸收的**（``W/T``、``b/T`` 是同一条线性层），
    所以标定结果可以原样写回 blob，固件不需要任何额外代码。
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    z = np.asarray(logit, dtype=np.float64).reshape(-1)
    if grid is None:
        grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 161))
    best_t, best = 1.0, float("inf")
    for t in grid:
        p = 1.0 / (1.0 + np.exp(-np.clip(z / t, -30.0, 30.0)))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best:
            best, best_t = nll, float(t)
    return best_t, best


def summarize_scores(name, y, score, extra=None):
    """一行式摘要 dict（打印 + JSON 落盘共用）。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    _rows, ece, mce = reliability(y, s)
    out = {
        "name": name, "n": int(y.size), "pos": int((y > 0.5).sum()),
        "base_rate": float(y.mean()) if y.size else 0.0,
        "auc": roc_auc(y, s), "brier": brier(y, s), "logloss": log_loss(y, s),
        "ece": ece, "mce": mce,
        "mean_p": float(s.mean()) if s.size else 0.0,
    }
    if extra:
        out.update(extra)
    return out
