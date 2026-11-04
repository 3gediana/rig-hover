# -*- coding: utf-8 -*-
"""esc_scenes.py — 合成场景数据集：A（蒸馏）+ B（规则时序）混合标签。

数据分两半，**同一份 106 维观测、同一套 base CfC**：

A. **蒸馏帧**（``arm="base"``）：``pc/train/data`` 里 8 条教师闭环会话（HD 视频级的
   合成剧本）重放 + 轻量增广（conf ×U(.92,1.08)、theta +N(0,1.5°)）。标签 = 可观测
   升级判据在**该帧**上的 ``p_esc``（软标签）。这一半教的是「基率长什么样」。

B. **规则合成帧**（``arm="temporal"``）：4 个**只会由时序信息决定**的场景族，
   每族一对孪生会话 ``ordered`` / ``control``：

   ================  ==================================  ====================
   族                时序证据（ordered 才成立）           control 怎么构造
   ================  ==================================  ====================
   sustained_conf    连续 ≥8 帧 marginal 置信度           同 multiset，按 2 低/1 高
                     (0.60，单帧 u_conf=0)               交替 → 最长连低 2 帧
   bearing_rate      目标方位 8 帧斜率 > 5°（走出 FOV）   同一条轨迹**反向遍历**
   flicker           20 帧窗口内 ≥3 次掉帧（每次 <250ms，  同掉帧**总帧数**，
                     单帧 u_lost=0）                     改成 4 次、间隔 11 帧
   conf_drift        16 帧内置信度单调下滑 >0.03          同一条曲线**反向遍历**
                    （全程 conf>0.85，单帧 φ 完全不变）
   ================  ==================================  ====================

   这两半的设计要点：每族的 ``control`` 与 ``ordered`` **逐帧字段的多重集完全相同**
   （或几乎相同），所以任何**逐帧**模型在两边看到的是同一个分布，AUC 恒 0.5。
   标签由时序谓词给出（规则合成，不是蒸馏），因此不存在「用教师教学生再考教师」
   的循环论证。谓词实现见 :func:`_gt_sustained_conf` 等。

为什么标签不能全靠蒸馏（arm A）：蒸馏的上限就是教师本身 —— 逐帧 logistic
看不见「连续 8 帧」「8 帧斜率」「20 帧窗口内 3 次」这类量，所以它能给的标签
在这些时刻恒为 0.02~0.05。**要把时序信息教进去，标签里必须有时序信息**，
这就是 arm B 存在的唯一理由（见 README §2）。

产出 ``data/esc_dataset.npz``：``obs (N,T,106) f32``、``teacher (N,T)``（逐帧判据的
``p_esc``，永远存）、``hard (N,T)``（-1 = 无硬标签）、``phi (N,T,12)``、
``split``、``fam``、``arm``、``donor`` 等元数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import esc_core as C
import esc_model as M

F32 = np.float32
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / "base"
SESSION_DIR = BASE_DIR / "sessions"
DATA_DIR = HERE / "data"
DATASET = DATA_DIR / "esc_dataset.npz"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

#: 时序场景每段（一个 episode）的帧数
EPISODE_LEN = 48
#: 时序会话 = 几段拼接
EPISODES_PER_SESSION = 8
#: 每条会话的帧数（50 Hz → 7.68 s）；两类会话必须等长，才能拼成一个张量
SESSION_LEN = EPISODE_LEN * EPISODES_PER_SESSION

FAMILIES = ("sustained_conf", "bearing_rate", "flicker", "conf_drift")

#: 场景族的中文一句话说明（打印与 README 共用）
FAMILY_DOC = {
    "sustained_conf": "连续 ≥8 帧 marginal 置信度（单帧 φ 恒定）",
    "bearing_rate": "目标方位 8 帧斜率 >5°（theta 不进任何势函数）",
    "flicker": "20 帧窗口内 ≥3 次短掉帧（单帧 u_lost=0）",
    "conf_drift": "16 帧内置信度单调下滑 >0.03（全程 conf>0.85，φ 不变）",
}


# --------------------------------------------------------------------------- #
# 基座 CfC（冻结的 5 头网络）：只用来算「本帧动作」-> 动作置信度 -> φ
# --------------------------------------------------------------------------- #
class BaseScorer:
    """冻结 base CfC + 可观测判据，把观测序列打成 ``(φ, p_esc, 动作)``。

    Args:
        device: ``cpu`` / ``cuda:0``。
        batch: 一次前向的会话条数。
    """

    def __init__(self, device="cpu", batch=16):
        import torch  # noqa: PLC0415 — 只有这个类需要 torch

        self._torch = torch
        self.model, self.mean, self.std, self.ckpt = M.load_base_policy(device=device)
        self.device = device
        self.batch = int(batch)

    # ------------------------------------------------------------------
    def normalize(self, obs):
        """``(...,106)`` 原始观测 -> 归一化观测（与固件 ``cfc_model_forward`` 同口径）。"""
        o = np.asarray(obs, dtype=np.float64)
        mu = self.mean.astype(np.float64)
        sd = self.std.astype(np.float64)
        bad = ~np.isfinite(o)
        v = np.where(bad, mu, o)
        v = np.clip(v, mu - 4.0 * sd, mu + 4.0 * sd)
        return ((v - mu) / sd).astype(F32)

    def score(self, obs):
        """``(N,T,106)`` -> dict（``phi``/``teacher``/``head``/``vyaw``/``vx``/``mode_conf``）。"""
        torch = self._torch
        obs = np.ascontiguousarray(obs, dtype=F32)
        n, t, _ = obs.shape
        x = self.normalize(obs)
        outs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.batch):
                xb = torch.from_numpy(x[i:i + self.batch]).to(self.device)
                outs.append(self.model.forward_seq(xb).cpu().numpy())
        out = np.concatenate(outs, axis=0)                     # (N,T,15)
        head = np.tanh(out[..., M.COL["head_yaw"]]) * M.HEAD_MAX
        vyaw = np.tanh(out[..., M.COL["vyaw"]]) * M.VYAW_MAX
        vx = np.tanh(out[..., M.COL["vx"]]) * M.VX_MAX
        ml = out[..., M.COL["mode"]:M.COL["mode"] + 4]
        pm = np.exp(ml - ml.max(axis=-1, keepdims=True))
        pm /= pm.sum(axis=-1, keepdims=True)
        top2 = np.sort(pm, axis=-1)[..., -2:]
        mode_conf = top2[..., 1] - top2[..., 0]

        fields = C.obs_fields(obs.reshape(-1, obs.shape[-1]), self.mean, self.std)
        fields = {k: v.reshape(n, t) for k, v in fields.items()}
        act = {"head": head, "vyaw": vyaw, "vx": vx, "mode_conf": mode_conf}
        phi = C.phi_from_fields(
            {k: fields[k].reshape(-1) for k in fields},
            {k: act[k].reshape(-1) for k in act})
        phi = phi.reshape(n, t, C.K_POTENTIALS)
        teacher = C.p_esc_from_phi(phi)
        return {"phi": phi, "teacher": teacher, "fields": fields, **act}


# --------------------------------------------------------------------------- #
# 载入基座会话
# --------------------------------------------------------------------------- #
def load_donors():
    """读 ``base/sessions/*.npz``（``pc/train/data`` 的只读副本）里的 obs 与教师动作。

    Returns:
        list[dict]，键 ``name / obs / ctrl / mode / emote``。``ctrl`` 是
        ``(T,3)`` 的 ``[head/45, vyaw/90, vx/0.30]``，与 ``train_cfc`` 的连续头
        目标同口径。
    """
    files = sorted(SESSION_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError("%s 下没有会话 npz（应先从 pc/train/data 拷一份）"
                                % SESSION_DIR)
    out = []
    for p in files:
        z = np.load(p)
        ctrl = np.stack([
            np.asarray(z["head_deg"], dtype=F32) / M.HEAD_MAX,
            np.asarray(z["vyaw_dps"], dtype=F32) / M.VYAW_MAX,
            np.asarray(z["vx_mps"], dtype=F32) / M.VX_MAX,
        ], axis=-1)
        out.append({
            "name": p.name,
            "obs": np.ascontiguousarray(z["obs"], dtype=F32),
            "ctrl": np.ascontiguousarray(ctrl, dtype=F32),
            "mode": np.ascontiguousarray(z["mode"], dtype=np.int64),
            "emote": np.ascontiguousarray(z["emote"], dtype=np.int64),
        })
    return out


def _window_idx(donor, start, length):
    """donor 上的等长窗口下标（越界回绕）。"""
    t = donor["obs"].shape[0]
    return (np.arange(int(length)) + int(start)) % t


def _window(donor, idx):
    """按下标取一帧窗口（obs 是拷贝，其它是只读视图的拷贝）。"""
    return (np.array(donor["obs"][idx], dtype=F32, copy=True),
            np.array(donor["ctrl"][idx], dtype=F32, copy=True),
            np.array(donor["mode"][idx], dtype=np.int64, copy=True),
            np.array(donor["emote"][idx], dtype=np.int64, copy=True))


# --------------------------------------------------------------------------- #
# arm B：四个时序场景族。全部返回 (overrides, gt)
# --------------------------------------------------------------------------- #
def _finalize(ov, gt):
    return ov, np.asarray(gt, dtype=F32)


def _gt_runs(x, thr, min_run):
    """``x < thr`` 连续 ≥ ``min_run`` 帧 -> 该段内（第 min_run 帧起）为 1。"""
    n = x.size
    gt = np.zeros(n, dtype=F32)
    run = 0
    for t in range(n):
        run = run + 1 if x[t] < thr else 0
        gt[t] = 1.0 if run >= min_run else 0.0
    return gt


def _gt_window_count(events, window, need):
    """``events[t]=1`` 表示 t 处有一次事件；(t-window, t] 内 ≥need 次 -> 1。"""
    n = events.size
    gt = np.zeros(n, dtype=F32)
    for t in range(n):
        lo = max(0, t - int(window) + 1)
        gt[t] = 1.0 if int(events[lo:t + 1].sum()) >= need else 0.0
    return gt


def _gt_decline(x, window, drop):
    """``x[t] - max(x[t-window:t]) < -drop`` -> 1（需要满窗）。"""
    n = x.size
    gt = np.zeros(n, dtype=F32)
    for t in range(n):
        if t < window:
            continue
        ref = x[t - window:t].max()
        gt[t] = 1.0 if (x[t] - ref) < -float(drop) else 0.0
    return gt


def scene_sustained_conf(rng, ordered):
    """连续多帧 marginal 置信度（本任务点名的那个场景）。

    ordered: ``[0.76]×16 + [0.60]×32``；``conf=0.60`` 时单帧
    ``u_conf = smoothstep(0.60; 0.55→0.15) = 0`` —— **每一帧单独看都是良性的**。
    真值：连续 ≥8 帧 ``conf < 0.65``（视觉在退化，但没跌进"看不见"的区间）。
    control: ``[0.60,0.60,0.76]×16``，multiset 完全相同，最长连低 2 帧 → 真值恒 0。
    """
    n = EPISODE_LEN
    if ordered:
        conf = np.concatenate([np.full(16, 0.76), np.full(n - 16, 0.60)])
    else:
        # 2 低 + 1 高：(16 个高 + 32 个低) 与 ordered 完全同一个 multiset
        conf = np.tile(np.array([0.60, 0.60, 0.76]), n // 3)[:n]
    assert np.allclose(np.sort(conf), np.sort(
        np.concatenate([np.full(16, 0.76), np.full(n - 16, 0.60)]))), "multiset 必须一致"
    gt = _gt_runs(conf, 0.65, 8)
    return _finalize({"conf": conf}, gt)


def scene_bearing_rate(rng, ordered):
    """目标方位持续外移（``theta_img`` 不进任何势函数，逐帧 φ 与场景无关）。

    ordered: ``theta: -12° → +30°`` 线性（0.894°/帧 = 44.7°/s），人正走出 FOV。
    真值：8 帧斜率 > 5° **且** |theta| > 10°（约 0.2 s 后目标出 60° FOV）。
    control: 同一条轨迹**反向遍历**（人正在走进画面中心）：逐帧 theta 的多重集
    完全相同、逐帧 φ 完全相同，只有 8 帧斜率符号相反 -> 真值恒 0。
    这也是唯一一种"同 multiset、只在时间方向上不同"的严格对照。
    """
    n = EPISODE_LEN
    theta = np.linspace(-12.0, 30.0, n)
    if not ordered:
        theta = theta[::-1].copy()
    slope8 = np.zeros(n)
    for t in range(8, n):
        slope8[t] = theta[t] - theta[t - 8]
    gt = ((slope8 > 5.0) & (np.abs(theta) > 10.0)).astype(F32)
    if not ordered:
        assert gt.sum() == 0.0, "control 必须被谓词判成 0"
    return _finalize({"theta": theta}, gt)


def scene_flicker(rng, ordered):
    """短掉帧的**聚簇**（每次掉帧只有 40~80 ms，单帧 ``u_lost`` 恒为 0）。

    ordered: 掉帧起点 8/14/20/32/38/44（各 1 帧）。
    真值：(t-20, t] 内 ≥3 次掉帧 —— 「检测器不稳」这个判断本身是窗口统计量。
    control: 掉帧总帧数相同（6 帧），改成 4 次、间隔 11 帧 → 任何 20 帧窗口 ≤2 次。
    """
    n = EPISODE_LEN
    seen = np.ones(n)
    age_ms = np.zeros(n)
    if ordered:
        for t0 in (8, 14, 20, 32, 38, 44):
            seen[t0] = 0.0
            age_ms[t0] = 40.0
    else:
        for t0, ln in ((4, 2), (15, 2), (26, 1), (37, 1)):
            for k in range(ln):
                seen[t0 + k] = 0.0
                age_ms[t0 + k] = 40.0 * (k + 1)
    events = np.zeros(n)
    prev = 1.0
    for t in range(n):
        if seen[t] < 0.5 and prev >= 0.5:
            events[t] = 1.0
        prev = seen[t]
    gt = _gt_window_count(events, 20, 3)
    return _finalize({"seen": seen, "age_ms": age_ms}, gt)


def scene_conf_drift(rng, ordered):
    """置信度缓慢单调下滑（全程 ``conf > 0.85``，``c_local_ok`` 恒 1，φ 一字不变）。

    ordered: ``0.99 → 0.86`` 线性。
    真值：16 帧内下滑 > 0.03（跟踪器退化的**趋势**，不是单帧水平）。
    control: 同一条曲线**反向遍历**（置信度在回升）：逐帧 conf 的多重集完全
    相同（所以 ∃逐帧模型的输出分布也相同），只有趋势符号相反 -> 真值恒 0。
    """
    n = EPISODE_LEN
    conf = np.linspace(0.99, 0.86, n)
    if not ordered:
        conf = conf[::-1].copy()
    gt = _gt_decline(conf, 16, 0.03)
    if not ordered:
        assert gt.sum() == 0.0, "control 必须被谓词判成 0"
    return _finalize({"conf": conf}, gt)


SCENES = {
    "sustained_conf": scene_sustained_conf,
    "bearing_rate": scene_bearing_rate,
    "flicker": scene_flicker,
    "conf_drift": scene_conf_drift,
}

#: overrides 里的键 -> 观测下标
_OVR_IX = {"conf": C.IX_CONF, "seen": C.IX_SEEN, "theta": C.IX_THETA}


# --------------------------------------------------------------------------- #
# 组装会话
# --------------------------------------------------------------------------- #
def apply_overrides(row, ov):
    """把 ``ov`` 里的字段写进一帧观测（原地，``row`` 是拷贝）。"""
    if "conf" in ov:
        row[C.IX_CONF] = F32(ov["conf"])
    if "seen" in ov:
        row[C.IX_SEEN] = F32(ov["seen"])
    if "theta" in ov:
        row[C.IX_THETA] = F32(ov["theta"])
    if "age_ms" in ov:
        row[C.IX_AGE_S] = F32(ov["age_ms"] / 1000.0)
    return row


def build_base_sessions(donors, scorer, variants=6, seed=0):
    """arm A：donor 重放 + 轻量增广。

    Returns:
        ``(obs, teacher, hard, phi, meta, ctrl, mode, emote, ctrl_ok)``。
        ``ctrl_ok`` 标记"这一帧的控制标签是可信的"（时序会话的 obs 被改过，
        教师动作不再是它的标签，所以那一半的 ``ctrl_ok`` 全 0）。
    """
    rng = np.random.default_rng(seed)
    obs_list, ctrl_list, mode_list, emote_list, meta = [], [], [], [], []
    for v in range(int(variants)):
        for di, d in enumerate(donors):
            start = int(rng.integers(0, d["obs"].shape[0]))
            idx = _window_idx(d, start, SESSION_LEN)
            o, c, mo, em = _window(d, idx)
            o[:, C.IX_CONF] = np.clip(
                o[:, C.IX_CONF] * rng.uniform(0.92, 1.08), 0.0, 1.0).astype(F32)
            o[:, C.IX_THETA] = (o[:, C.IX_THETA]
                                + rng.normal(0.0, 1.5, size=SESSION_LEN)).astype(F32)
            obs_list.append(o)
            ctrl_list.append(c)
            mode_list.append(mo)
            emote_list.append(em)
            meta.append({"arm": "base", "fam": "base", "donor": d["name"],
                         "variant": v, "start": start})
    obs = np.stack(obs_list)
    sc = scorer.score(obs)
    return (obs, sc["teacher"], np.full(obs.shape[:2], -1.0, F32), sc["phi"], meta,
            np.stack(ctrl_list), np.stack(mode_list), np.stack(emote_list),
            np.ones(obs.shape[:2], F32))


def build_temporal_sessions(donors, scorer, per_cell=6, seed=1):
    """arm B：4 族 × {ordered, control} × ``per_cell`` 条会话。

    每个 episode 的底子来自一段**良性** donor 窗口（逐帧判据 ``p_esc < 0.15``），
    这样把 control 段的硬标签写成 0 才不会和"这一帧本来就危险"打架。
    """
    rng = np.random.default_rng(seed)
    # 良性窗口的候选起点：按 64 帧一块抽样算一遍逐帧判据
    benign = []
    block = 64
    for di, d in enumerate(donors):
        nblk = max(1, d["obs"].shape[0] // block)
        idx = np.arange(nblk) * block
        seg = np.stack([_window(d, _window_idx(d, int(s), block))[0] for s in idx])
        sc = scorer.score(seg)
        for k, s in enumerate(idx):
            if float(sc["teacher"][k].max()) < 0.15:
                benign.append((di, int(s), block))
    if not benign:
        benign = [(i, 0, block) for i in range(len(donors))]

    obs_list, hard_list, ctrl_list, mode_list, emote_list, meta = [], [], [], [], [], []
    for fam in FAMILIES:
        for rep in range(int(per_cell)):
            # 关键：**同一 rep 的 ordered / control 用完全相同的 donor 窗口**。
            # 否则两边除了覆盖字段以外连底图都不一样，匹配对照就失效了
            # （第一版没有这么做，结果 B2 基线在同一批数据上刷到了 0.64，
            #  那是 donor 差异泄漏，不是时序信息）。
            picks = []
            for _ep in range(EPISODES_PER_SESSION):
                di, st, blk = benign[int(rng.integers(0, len(benign)))]
                off = int(rng.integers(0, max(1, blk - EPISODE_LEN)))
                picks.append((di, _window_idx(donors[di], st + off, EPISODE_LEN)))
            for ordered in (True, False):
                frames, ctls, modes, emotes, gts = [], [], [], [], []
                for ep, (di, idx) in enumerate(picks):
                    ov, gt = SCENES[fam](rng, ordered)
                    src, c, mo, em = _window(donors[di], idx)
                    for t in range(EPISODE_LEN):
                        src[t] = apply_overrides(src[t], {k: v[t] for k, v in ov.items()})
                    frames.append(src)
                    ctls.append(c)
                    modes.append(mo)
                    emotes.append(em)
                    gts.append(gt)
                obs_list.append(np.concatenate(frames, axis=0))
                ctrl_list.append(np.concatenate(ctls, axis=0))
                mode_list.append(np.concatenate(modes))
                emote_list.append(np.concatenate(emotes))
                hard_list.append(np.concatenate(gts))
                meta.append({"arm": "temporal", "fam": fam,
                             "ordered": bool(ordered), "rep": rep})
    obs = np.stack(obs_list)
    hard = np.stack(hard_list).astype(F32)
    sc = scorer.score(obs)
    return (obs, sc["teacher"], hard, sc["phi"], meta,
            np.stack(ctrl_list), np.stack(mode_list), np.stack(emote_list),
            np.zeros(obs.shape[:2], F32))


# --------------------------------------------------------------------------- #
# 划分
# --------------------------------------------------------------------------- #
def make_split(meta, seed=7, fracs=(0.6, 0.2, 0.2)):
    """**按"单元"划分** train/val/test，单元内部绝不拆开。

    一个单元 = 必须待在同一侧的一组会话：

    * base 会话：一条会话一个单元（增广份数 ``variant`` 相同的一组算一个单元）；
    * 时序会话：**同一 rep 的 ordered/control 孪生必须同侧** —— 它们用的是同一批
      donor 窗口，一边在训练一边在测试就等于把答案漏过去了。

    每个 ``(arm, fam)`` 组内按 60/20/20 分。
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for i, m in enumerate(meta):
        if m["arm"] == "temporal":
            # 时序：同一 rep 的 ordered/control 必须同侧（它们用同一批 donor 窗口）
            unit = ("temporal", m["fam"], m.get("rep", 0))
        else:
            # base：一条会话就是一个单元。别按 variant 分组 —— 那样 test 只会拿到
            # 一个增广份数（8 条会话），实测该份数正好落在全良性的窗口上，
            # 导致"测试集里没有一帧判据为阳性"，AUC 无从计算。
            unit = ("base", i)
        group = (m["arm"], m["fam"])
        groups.setdefault(group, {}).setdefault(unit, []).append(i)

    split = np.empty(len(meta), dtype="<U5")
    for _group, units in groups.items():
        keys = list(units.keys())
        rng.shuffle(keys)
        n = len(keys)
        n_tr = max(1, int(round(fracs[0] * n)))
        n_va = max(1, int(round(fracs[1] * n))) if n >= 5 else 0
        if n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        for k, key in enumerate(keys):
            s = ("train" if k < n_tr
                 else "val" if k < n_tr + n_va else "test")
            for i in units[key]:
                split[i] = s
    return split


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build(seed=0, variants=6, per_cell=6, device="cpu", out=DATASET, verbose=True):
    """生成数据集并落盘。返回 ``(path, summary)``。"""
    donors = load_donors()
    scorer = BaseScorer(device=device)
    if verbose:
        print("[data] %d 条 donor 会话（%d 帧/条），base CfC 已冻结"
              % (len(donors), donors[0]["obs"].shape[0]))

    obs_b, teacher_b, hard_b, phi_b, meta_b, ctrl_b, mode_b, em_b, ok_b = \
        build_base_sessions(donors, scorer, variants=variants, seed=seed)
    obs_t, teacher_t, hard_t, phi_t, meta_t, ctrl_t, mode_t, em_t, ok_t = \
        build_temporal_sessions(donors, scorer, per_cell=per_cell, seed=seed + 1)

    obs = np.concatenate([obs_b, obs_t], axis=0)
    teacher = np.concatenate([teacher_b, teacher_t], axis=0)
    hard = np.concatenate([hard_b, hard_t], axis=0)
    phi = np.concatenate([phi_b, phi_t], axis=0)
    ctrl = np.concatenate([ctrl_b, ctrl_t], axis=0)
    mode = np.concatenate([mode_b, mode_t], axis=0)
    emote = np.concatenate([em_b, em_t], axis=0)
    ctrl_ok = np.concatenate([ok_b, ok_t], axis=0)
    meta = meta_b + meta_t
    split = make_split(meta)

    # 训练目标：有硬标签的帧用硬标签，其余用蒸馏软标签
    y = np.where(hard >= 0.0, hard, teacher)
    weight = np.where(hard >= 0.0, 1.0, 1.0)

    arm = np.asarray([m["arm"] for m in meta])
    fam = np.asarray([m["fam"] for m in meta])
    ordered = np.asarray([bool(m.get("ordered", False)) for m in meta])
    donor = np.asarray([m.get("donor", "") for m in meta])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, obs=obs, teacher=teacher.astype(F32), hard=hard.astype(F32),
        y=y.astype(F32), phi=phi.astype(F32), split=split, arm=arm, fam=fam,
        ordered=ordered, donor=donor, weight=weight.astype(F32),
        ctrl=ctrl.astype(F32), mode=mode.astype(np.int64),
        emote=emote.astype(np.int64), ctrl_ok=ctrl_ok.astype(F32),
        session_len=np.int32(EPISODE_LEN))

    summary = {
        "n_sessions": int(obs.shape[0]), "T": int(obs.shape[1]),
        "frames": int(obs.shape[0] * obs.shape[1]),
        "families": list(FAMILIES),
        "split_counts": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "frame_counts": {s: int((split == s).sum() * obs.shape[1])
                         for s in ("train", "val", "test")},
        "temporal_gt_rate": float(hard[hard >= 0.0].mean()) if (hard >= 0).any() else 0.0,
        "base_positive_rate": float((teacher[arm == "base"] > 0.5).mean()),
        "teacher_mean_base": float(teacher[arm == "base"].mean()),
        "teacher_mean_temporal": float(teacher[arm == "temporal"].mean()),
        "target_mean": float(y.mean()),
    }
    (DATA_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print("[data] %d 条会话 × %d 帧 = %d 帧；划分 %s"
              % (obs.shape[0], obs.shape[1], summary["frames"],
                 summary["split_counts"]))
        print("[data] 基率：base 帧判据 p>0.5 占 %.2f%%，判据均值 %.4f；"
              "时序帧硬标签阳性率 %.1f%%；训练目标均值 %.4f"
              % (100 * summary["base_positive_rate"], summary["teacher_mean_base"],
                 100 * summary["temporal_gt_rate"], summary["target_mean"]))
        print("[data] -> %s" % out)
    return out, summary


def load(path=DATASET):
    """读数据集，返回 dict（数组已拆好）。"""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["session_len"] = int(d["session_len"])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="生成升级头数据集（A 蒸馏 + B 时序规则）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", type=int, default=6, help="每条 donor 的增广份数")
    ap.add_argument("--per-cell", type=int, default=6,
                    help="每个 (族, ordered/control) 单元的会话条数")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(DATASET))
    args = ap.parse_args(argv)
    build(seed=args.seed, variants=args.variants, per_cell=args.per_cell,
          device=args.device, out=Path(args.out))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
