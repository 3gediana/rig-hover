#!/usr/bin/env python
"""test_hub.py -- DecisionHub 的验收测试。一条命令跑完：``python test_hub.py``。

覆盖（每一项都对应设计文档里的一个断言）：

1. 各级路由正确性（12 个场景 + 安全否决 + 不可用层级过滤）
2. 超时回退（L3 超时 -> L2）、云端不可用降级、全挂 -> L1
3. 延迟统计（各级 dispatch / skill / L0 决策耗时都真的记下来了）
4. 非法输入不崩溃（观测 fuzz、上层 JSON fuzz、迟到回复、坏后端）
5. 不变量：L0 永不阻塞；未校验的指令绝不执行；动作永远在 CfC 的物理范围内
6. 可训练性：fit_gate / fit_router 能从数据里把注入的权重学回来

测试用**真实时间**驱动（后端在线程里真阻塞），所以超时/降级不是假的；
为了让一条命令几秒内跑完，所有 mock 延迟都是 5~200 ms 量级。
"""

from __future__ import annotations

import json
import math
import random
import sys
import time

try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

import hub as H
from hub import (
    CfcOut,
    CtrlAction,
    DecisionHub,
    HubConfig,
    HubObs,
    LEVEL_AGENT,
    LEVEL_CLOUD,
    LEVEL_NAMES,
    LEVEL_REFLEX,
    LEVEL_SKILL,
    POTENTIAL_NAMES,
    SkillExecutor,
    fit_gate,
    fit_router,
    percentile,
    smoothstep,
)
from llm_backend import (
    Backend,
    BackendUnavailable,
    CloudLLMBackend,
    InvalidInstruction,
    MockAgentBackend,
    MockCloudBackend,
    build_escalation_request,
    make_instruction,
    parse_instruction,
    probe_cloud_transport,
)

BAR = "=" * 74
PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    """记一条断言。cond 为假就记 FAIL（不抛，跑完所有用例再汇总）。"""
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def expect_raises(name: str, exc, fn, *a, **k) -> None:
    """断言 fn 抛指定异常（不抛或抛错都算 FAIL）。"""
    try:
        fn(*a, **k)
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:60]}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"抛了 {type(e).__name__}，期望 {exc.__name__}")
    else:
        check(name, False, "没有抛异常")


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------


def mk_hub(*, l2=None, l3=None, cfg: HubConfig | None = None) -> DecisionHub:
    """造一个 hub。l2/l3 传 None 表示该级不可用。"""
    backends: dict = {}
    if l2 is not None:
        backends[LEVEL_AGENT] = l2
    if l3 is not None:
        backends[LEVEL_CLOUD] = l3
    return DecisionHub(cfg or fast_cfg(), backends=backends)


def fast_cfg(**kw) -> HubConfig:
    """测试用配置：关掉触发驻留和离散量平滑，让判定在当拍就确定。

    默认配置（dwell_ms=150 / ema_tau=0.15）的行为由 ``test_dwell`` 单独覆盖。
    """
    kw.setdefault("dwell_ms", 0)
    kw.setdefault("ema_tau", {n: 0.0 for n in POTENTIAL_NAMES})
    return HubConfig(**kw)


def drive(hub: DecisionHub, obs_fn, ticks: int = 40, dt: float = 0.02,
          pace: float = 0.002) -> list:
    """按真实时间驱动 hub，返回每一拍的 HubOut。"""
    t0 = time.perf_counter()
    outs = []
    for _ in range(ticks):
        t = time.perf_counter() - t0
        obs = obs_fn(t)
        obs.t_now = t
        outs.append(hub.tick(obs, dt))
        if pace > 0:
            time.sleep(pace)
    return outs


def steady_phi(hub: DecisionHub, obs: HubObs, n: int = 40) -> dict:
    """把离散检测量的平滑跑收敛，得到"稳定态"的势函数向量。"""
    hub._phi_ema = {k: 0.0 for k in hub._phi_ema}
    phi: dict = {}
    for _ in range(n):
        phi = hub.compute_phi(hub._sanitize_obs(obs), 0.02)
    return phi


def O(**kw) -> HubObs:
    """简写：默认一个"看得见、置信度高"的人。"""
    base = dict(cfc=CfcOut(head_des=3.0, vyaw_des=0.0, vx_des=0.0, mode=1, emote=0,
                           action_conf=0.94),
                person_conf=0.94, feat_age_ms=20.0)
    base.update(kw)
    return HubObs(**base)


# --------------------------------------------------------------------------
# 1. 各级路由
# --------------------------------------------------------------------------


def test_routing() -> None:
    print("\n[1] 各级路由正确性")
    hub = mk_hub(l2=MockAgentBackend(), l3=MockCloudBackend())

    cases = [
        # (名字, 观测, 期望层级, p_esc 下界, p_esc 上界)
        ("静止跟踪", O(), LEVEL_REFLEX, 0.0, 0.20),
        ("挥手(本地ACK)", O(cfc=CfcOut(3, 0, 0, 2, 1, 0.90)), LEVEL_REFLEX, 0.0, 0.20),
        ("看不清(conf .18)", O(person_conf=0.18, cfc=CfcOut(0, 0, 0, 0, 0, 0.4)),
         LEVEL_SKILL, 0.5, 1.0),
        ("目标丢 800ms", O(seen=0, person_conf=0.0, feat_age_ms=800.0,
                       cfc=CfcOut(0, 0, 0, 1, 0, 0.6)), LEVEL_SKILL, 0.35, 1.0),
        ("指着东西", O(pointing=1.0, point_dir_deg=30.0,
                    cfc=CfcOut(5, 0, 0, 1, 0, 0.9)), LEVEL_AGENT, 0.5, 1.0),
        ("说话+两动作", O(speech=1.0, instr_clauses=2,
                      cfc=CfcOut(0, 0, 0, 1, 0, 0.9)), LEVEL_CLOUD, 0.8, 1.0),
        ("需要查资料", O(speech=1.0, lookup=1.0, instr_clauses=1,
                     cfc=CfcOut(0, 0, 0, 1, 0, 0.9)), LEVEL_CLOUD, 0.8, 1.0),
        ("场景OOD", O(novelty=6.0, cfc=CfcOut(0, 0, 0, 1, 0, 0.6)),
         LEVEL_AGENT, 0.4, 1.0),
        ("特征出域", O(clip_ratio=0.8, cfc=CfcOut(0, 0, 0, 1, 0, 0.6)),
         LEVEL_AGENT, 0.2, 1.0),
        ("没把握地靠近", O(dist_est=0.55, cfc=CfcOut(0, 0, 0.24, 1, 0, 0.3)),
         LEVEL_AGENT, 0.2, 1.0),
        ("语义通道过期", O(sem_age_ms=3000.0), LEVEL_SKILL, 0.0, 1.0),
        ("满血反证", O(person_conf=0.99, cfc=CfcOut(0, 0, 0, 1, 0, 0.99)),
         LEVEL_REFLEX, 0.0, 0.05),
    ]
    for name, obs, want, lo, hi in cases:
        phi = steady_phi(hub, obs)
        d = hub.decide(phi, obs)
        ok = d.level == want and lo <= d.p_esc <= hi
        check(f"路由 {name} -> L{want}({LEVEL_NAMES[want]})", ok,
              f"实际 L{d.level} p_esc={d.p_esc:.3f} 路由={' '.join(f'L{k}:{v:.2f}' for k, v in d.pfix.items())}")

    # 安全否决：桌沿危险 / 摔倒 / 视觉彻底丢失 -> 不升级
    for tag, obs in (
        ("桌沿", O(person_conf=0.3, pointing=1.0, gaze_danger=1)),
        ("摔倒", O(pointing=1.0, robot_state=0)),
        ("视觉彻底丢失", O(seen=0, person_conf=0.0, feat_age_ms=5000.0)),
    ):
        d = hub.decide(steady_phi(hub, obs), obs)
        check(f"安全否决 {tag} -> L0 且带 veto", d.level == 0 and bool(d.veto),
              f"level={d.level} veto={d.veto}")

    # 层级不可用就不参与竞争
    hub2 = mk_hub()
    obs = O(speech=1.0, instr_clauses=3, cfc=CfcOut(0, 0, 0, 1, 0, 0.9))
    d = hub2.decide(steady_phi(hub2, obs), obs)
    check("没有 L2/L3 后端时最高只到 L1", d.level in (0, 1) and 2 not in d.allowed,
          f"level={d.level} allowed={d.allowed}")
    if d.level == LEVEL_SKILL:
        check("P_fix 在可用层级内归一化到 1", abs(d.pfix[LEVEL_SKILL] - 1.0) < 1e-6,
              f"P_fix(L1)={d.pfix[LEVEL_SKILL]:.4f}")

    # decide 是纯函数
    phi = steady_phi(hub, O(pointing=1.0))
    before = hub.stats.escalations
    a, b = hub.decide(phi, O(pointing=1.0)), hub.decide(phi, O(pointing=1.0))
    check("decide 无副作用且可重放", a.as_dict() == b.as_dict()
          and hub.stats.escalations == before)

    hub.close()
    hub2.close()


def test_potentials() -> None:
    print("\n[2] 势函数的形状（可解释的前提）")
    hub = mk_hub()
    check("smoothstep 单调且两端精确", smoothstep(-999, 0, 1) == 0.0
          and smoothstep(999, 0, 1) == 1.0 and smoothstep(0.5, 0, 1) > smoothstep(0.4, 0, 1))
    check("smoothstep 支持反向区间", smoothstep(0.1, 1.0, 0.0) > 0.5)
    check("smoothstep 对 NaN 返回 0（不把脏数据当成风险）",
          smoothstep(float("nan"), 0, 1) == 0.0)

    phi = steady_phi(hub, O())
    check("全部 12 个势函数都在 [0,1]", all(0.0 <= v <= 1.0 for v in phi.values())
          and len(phi) == len(POTENTIAL_NAMES))
    phi_bad = steady_phi(hub, O(person_conf=0.1))
    check("置信度下降 -> u_conf 上升", phi_bad["u_conf"] > phi["u_conf"],
          f"{phi['u_conf']:.2f} -> {phi_bad['u_conf']:.2f}")
    phi_near = steady_phi(hub, O(dist_est=0.5, cfc=CfcOut(0, 0, 0.25, 1, 0, 0.3)))
    check("靠近且没把握 -> s_approach 上升", phi_near["s_approach"] > 0.2,
          f"s_approach={phi_near['s_approach']:.2f}")
    # logit 精确等于各项贡献之和（这是"可解释"的硬定义）
    d = hub.decide(phi, O())
    tot = sum(c for _, c in d.contributions)
    check("logit == 偏置 + 各势函数贡献之和", abs((tot + hub.cfg.gate_bias) - d.logit) < 1e-9,
          f"sum={tot:.6f} bias={hub.cfg.gate_bias}")
    hub.close()


# --------------------------------------------------------------------------
# 3. 超时 / 降级
# --------------------------------------------------------------------------


def test_timeout_and_fallback() -> None:
    print("\n[3] 超时回退与降级")
    cfg = fast_cfg(request_ttl_ms={1: 0, 2: 120, 3: 90}, total_budget_ms=2000)
    l3 = MockCloudBackend(latency_s=0.60, fail_mode="timeout")
    l2 = MockAgentBackend(latency_s=0.01)
    hub = mk_hub(l2=l2, l3=l3, cfg=cfg)
    obs = O(speech=1.0, instr_clauses=1, cfc=CfcOut(0, 0, 0, 1, 0, 0.9))
    outs = drive(hub, lambda t: obs, ticks=90, pace=0.002)
    levels = [o.level for o in outs]
    executed = [o.level for o in outs if o.payload.get("status") == "executing"]
    attempts: list = []
    for o in outs:
        if o.payload.get("attempts"):
            attempts = o.payload["attempts"]
    chain = [a["level"] for a in attempts]
    check("L3 超时后切到 L2 并真的执行了", LEVEL_CLOUD in levels
          and LEVEL_AGENT in executed, f"触达={sorted(set(levels))} 执行={sorted(set(executed))}")
    check("尝试链记录为 [3, 2]", chain[:2] == [LEVEL_CLOUD, LEVEL_AGENT], f"chain={chain}")
    check("超时计数 +1", hub.stats.timeouts >= 1, f"timeouts={hub.stats.timeouts}")
    check("降级计数 +1", hub.stats.fallbacks >= 1, f"fallbacks={hub.stats.fallbacks}")
    check("超时也作为延迟样本被记下", len(hub.stats.dispatch_ms[LEVEL_CLOUD]) >= 1,
          f"L3 样本={len(hub.stats.dispatch_ms[LEVEL_CLOUD])}")
    hub.close()

    # 云端不可用（没 key / 服务挂了）：不派发，直接当这一级不存在
    hub = mk_hub(l2=MockAgentBackend(latency_s=0.01),
                 l3=MockCloudBackend(fail_mode="down"))
    obs = O(speech=1.0, instr_clauses=2, cfc=CfcOut(0, 0, 0, 1, 0, 0.9))
    phi = steady_phi(hub, obs)
    d = hub.decide(phi, obs)
    check("云端 down -> 不在可选层级里", LEVEL_CLOUD not in d.allowed, f"allowed={d.allowed}")
    check("云端 down -> 改为路由到 L2", d.level == LEVEL_AGENT, f"level={d.level}")
    hub.close()

    # 全挂：降到 L1 本地技能，机器人不僵住
    hub = mk_hub(l2=MockAgentBackend(fail_mode="down"),
                 l3=MockCloudBackend(fail_mode="down"))
    outs = drive(hub, lambda t: obs, ticks=40, pace=0.002)
    exec_lvls = {o.level for o in outs if o.payload.get("status") == "executing"}
    check("L2/L3 全挂 -> 执行 L1 本地技能", exec_lvls == {LEVEL_SKILL}, f"exec={exec_lvls}")
    check("L1 是本地的：没有网络等待", all(
        o.payload.get("status") != "waiting" for o in outs))
    hub.close()

    # 后端返回非法 JSON：拒绝 + 降级，不崩溃
    hub = mk_hub(l2=MockAgentBackend(latency_s=0.01, fail_mode="garbage"), l3=None)
    obs_p = O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9))
    outs = drive(hub, lambda t: obs_p, ticks=60, pace=0.002)
    check("后端给非法指令 -> 被拒并降级到 L1",
          hub.stats.invalid_instruction >= 1
          and {o.level for o in outs if o.payload.get("status") == "executing"} == {LEVEL_SKILL},
          f"invalid={hub.stats.invalid_instruction}")
    hub.close()


class _StaleBackend(Backend):
    """永远回一个 id 对不上的指令（模拟上一次升级的迟到回复）。"""

    level = LEVEL_AGENT
    name = "stale"

    def __init__(self):
        self.latency_s = 0.01

    def ask(self, request, timeout_s):
        time.sleep(self.latency_s)
        return make_instruction(level=2, intent="stale_reply",
                                prog=[{"op": "head", "deg": 30.0, "ms": 200}],
                                source="stale", esc_id=int(request["id"]) + 99)


def test_stale_reply() -> None:
    print("\n[4] 迟到回复")
    hub = mk_hub(l2=_StaleBackend(), l3=None)
    obs = O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9))
    outs = drive(hub, lambda t: obs, ticks=40, pace=0.002)
    intents = [o.payload.get("intent") for o in outs]
    check("迟到的回复不会被执行", "stale_reply" not in intents, f"intents={set(i for i in intents if i)}")
    check("迟到回复被计数", hub.stats.stale_reply >= 1, f"stale={hub.stats.stale_reply}")
    hub.close()


# --------------------------------------------------------------------------
# 5. 不变量
# --------------------------------------------------------------------------


def test_l0_never_blocks() -> None:
    print("\n[5] 不变量：L0 永不阻塞")
    cfg = fast_cfg(request_ttl_ms={1: 0, 2: 400, 3: 400})
    hub = mk_hub(l2=MockAgentBackend(latency_s=0.25), l3=None, cfg=cfg)
    obs = O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9))
    outs = drive(hub, lambda t: obs, ticks=60, pace=0.002)
    waiting = [o for o in outs if o.payload.get("status") == "waiting"]
    check("等待期间每一拍都还出指令", len(waiting) >= 5, f"waiting={len(waiting)}")
    check("等待期间动作全部合法", all(
        abs(o.action.head_des) <= 45.0 and abs(o.action.vyaw_des) <= 90.0
        and abs(o.action.vx_des) <= 0.30 for o in outs))
    check("等待期间车轮清零（不盲走）", all(o.action.vx_des == 0.0 and o.action.vyaw_des == 0.0
                                          for o in waiting))
    check("等待期间头还在跟人（不是全停）", any(abs(o.action.head_des) > 0.5 for o in waiting))
    hub.close()


def test_skill_executor() -> None:
    print("\n[6] 上层指令落到 CfC 的动作空间")
    ins = make_instruction(
        level=2, intent="test",
        prog=[{"op": "head", "deg": 44.0, "ms": 100},
              {"op": "yaw", "dps": 90.0, "ms": 100},
              {"op": "vx", "mps": 0.30, "ms": 100},
              {"op": "emote", "id": 7, "ms": 100},
              {"op": "hold", "ms": 100}],
        source="t")
    ex = SkillExecutor(ins, CtrlAction(head_des=0.0))
    acts = []
    for _ in range(40):
        a = ex.step(0.02)
        if a is None:
            break
        acts.append(a)
    check("程序按 50Hz 展开为动作流", len(acts) >= 20, f"steps={len(acts)}")
    check("所有动作都在物理范围内", all(
        abs(a.head_des) <= 45.0 and abs(a.vyaw_des) <= 90.0 and abs(a.vx_des) <= 0.30
        for a in acts))
    check("头部真的走到了目标角", any(abs(a.head_des - 44.0) < 0.5 for a in acts))
    check("程序结束后返回 None", ex.step(0.02) is None)

    ins2 = make_instruction(level=2, intent="handback",
                            prog=[{"op": "track", "ms": 0}], source="t")
    ex2 = SkillExecutor(ins2, CtrlAction())
    check("track 立即交还控制权", ex2.step(0.02) is None)


def test_instruction_validation() -> None:
    print("\n[7] 上层 JSON 校验（非法内容一律拒绝）")
    bad = [
        ("不是对象", "hello"),
        ("空列表", []),
        ("未知 op", {"prog": [{"op": "teleport"}]}),
        ("op 不是字符串", {"prog": [{"op": 3}]}),
        ("缺 deg", {"prog": [{"op": "head", "ms": 100}]}),
        ("deg 越界", {"prog": [{"op": "head", "deg": 90.0, "ms": 100}]}),
        ("ms 为负", {"prog": [{"op": "hold", "ms": -5}]}),
        ("NaN", {"prog": [{"op": "vx", "mps": float("nan"), "ms": 100}]}),
        ("level 越界", {"level": 9, "prog": []}),
        ("ttl 超大", {"ttl_ms": 99999, "prog": []}),
        ("say 类型错", {"say": 123, "prog": []}),
        ("prog 不是列表", {"prog": "head"}),
        ("步数太多", {"prog": [{"op": "hold", "ms": 1}] * 40}),
        ("args 类型错", {"args": [], "prog": []}),
    ]
    for tag, obj in bad:
        expect_raises(f"拒绝 {tag}", InvalidInstruction, parse_instruction, obj)

    good = {"id": 7, "level": 2, "intent": "look_at",
            "prog": [{"op": "head", "deg": 30, "ms": 300}], "ttl_ms": 1200,
            "say": "看着那边", "confidence": 0.8}
    ins = parse_instruction(good)
    check("合法指令被接受并规范化", ins.level == 2 and ins.intent == "look_at"
          and ins.prog[0]["deg"] == 30.0 and ins.ttl_ms == 1200)
    check("指令可以序列化回 JSON", json.dumps(ins.as_dict(), allow_nan=False) != "")


def test_invalid_obs_fuzz() -> None:
    print("\n[8] 观测 fuzz（脏数据不崩溃、动作永远合法）")
    hub = mk_hub()   # 没有上层：任何升级都只会落到 L1
    rng = random.Random(1234)
    nasty = [float("nan"), float("inf"), -float("inf"), None, "x", [], {}, 1e12, -1e12,
             True, 0.0, -3.0]
    obs0 = O()
    bad_actions = 0
    crashed = 0
    for i in range(600):
        obs = O(
            t_now=rng.choice([rng.random() * 10, 0.0, -5.0]),
            person_conf=rng.choice(nasty),
            clip_ratio=rng.choice(nasty),
            novelty=rng.choice(nasty),
            feat_age_ms=rng.choice(nasty),
            dist_est=rng.choice(nasty),
            seen=rng.choice(nasty),
            speech=rng.choice(nasty),
            pointing=rng.choice(nasty),
            instr_clauses=rng.choice(nasty),
            sem_age_ms=rng.choice(nasty),
            theta_img=rng.choice(nasty),
            q_head=rng.choice(nasty),
            cfc=CfcOut(head_des=rng.choice(nasty), vyaw_des=rng.choice(nasty),
                       vx_des=rng.choice(nasty), mode=rng.choice(nasty),
                       emote=rng.choice(nasty), action_conf=rng.choice(nasty)),
        )
        try:
            out = hub.tick(obs, rng.choice([0.02, 0.0, -1.0, 5.0, float("nan")]))
        except Exception as e:  # noqa: BLE001
            crashed += 1
            print(f"      crash: {type(e).__name__}: {e}")
            continue
        a = out.action
        if not (abs(a.head_des) <= 45.0 and abs(a.vyaw_des) <= 90.0
                and abs(a.vx_des) <= 0.30 and 0 <= a.mode <= 3 and 0 <= a.emote <= 7):
            bad_actions += 1
    check("600 组脏观测没有抛异常", crashed == 0, f"crashed={crashed}")
    check("输出动作全部在物理范围内", bad_actions == 0, f"bad={bad_actions}")
    check("脏字段被计数", hub.stats.invalid_input > 0, f"invalid_input={hub.stats.invalid_input}")

    check("tick(None) 不崩溃", hub.tick(None).action is not None)  # type: ignore[arg-type]
    check("cfc 字段类型错也不崩溃", hub.tick(HubObs(cfc="nope")).level >= 0)  # type: ignore[arg-type]
    hub.close()


class _QuickBackend(Backend):
    """立刻回一条很短的指令（40ms 程序），用来测"升级-执行-冷却-闩锁"全流程。"""

    level = LEVEL_AGENT
    name = "quick"

    def ask(self, request, timeout_s):
        time.sleep(0.002)
        return make_instruction(level=2, intent="quick_look",
                                prog=[{"op": "head", "deg": 20.0, "ms": 40}],
                                source="quick", esc_id=int(request["id"]))


def test_latched() -> None:
    print("\n[9] 事件闩锁：同一件事不反复问")
    hub = mk_hub(l2=_QuickBackend(), l3=None, cfg=fast_cfg(cooldown_ms=60))
    obs = O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9))
    outs = drive(hub, lambda t: obs, ticks=90, pace=0.002)   # ~180ms：够走完 执行+冷却
    check("持续事件只升级一次", hub.stats.escalations == 1,
          f"escalations={hub.stats.escalations}")
    latched = [o for o in outs if o.payload.get("gated_by") == "latched"]
    check("之后的拍被标为 latched", len(latched) > 0, f"latched_ticks={len(latched)}")
    # 事件消失后应该重新武装
    obs_ok = O()
    drive(hub, lambda t: obs_ok, ticks=20, pace=0.0)
    check("事件消失后重新武装", hub._armed is True)
    hub.close()


def test_dwell() -> None:
    print("\n[9b] 触发驻留")
    hub = mk_hub(l2=MockAgentBackend(latency_s=0.01), l3=None,
                 cfg=fast_cfg(dwell_ms=150))
    obs = O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9))
    o1 = hub.tick(obs, 0.02)
    check("第一拍不升级（判据还没稳定）", o1.level == 0
          and o1.payload.get("gated_by") == "dwell", f"gate={o1.payload.get('gated_by')}")
    outs = drive(hub, lambda t: obs, ticks=120, pace=0.002)
    check("驻留够了就升级", hub.stats.escalations == 1, f"escalations={hub.stats.escalations}")
    hub.close()


# --------------------------------------------------------------------------
# 10. 延迟统计
# --------------------------------------------------------------------------


def test_stats() -> None:
    print("\n[10] 延迟统计")
    cfg = fast_cfg(request_ttl_ms={1: 0, 2: 200, 3: 200})
    hub = mk_hub(l2=MockAgentBackend(latency_s=0.02), l3=MockCloudBackend(latency_s=0.02),
                 cfg=cfg)
    drive(hub, lambda t: O(), ticks=20, pace=0.0)                    # 只有 L0
    drive(hub, lambda t: O(pointing=1.0, cfc=CfcOut(5, 0, 0, 1, 0, 0.9)),
          ticks=40, pace=0.002)                                      # L2
    s = hub.stats.summary()
    check("L0 决策耗时被记录", s["local_us_p50"] > 0 and len(hub.stats.local_us) >= 50,
          f"p50={s['local_us_p50']:.1f}us n={len(hub.stats.local_us)}")
    check("L2 dispatch 延迟被记录", s["levels"]["2"]["n"] >= 1
          and s["levels"]["2"]["dispatch_p50_ms"] > 0,
          f"p50={s['levels']['2']['dispatch_p50_ms']:.1f}ms")
    check("本地技能时长被记录", len(hub.stats.skill_ms[2]) >= 1,
          f"skill_ms={hub.stats.skill_ms[2][:3]}")
    check("分位数函数正确", abs(percentile([0, 10], 0.5) - 5.0) < 1e-9
          and abs(percentile([1, 2, 3], 1.0) - 3.0) < 1e-9 and percentile([], 0.5) == 0.0)
    txt = hub.format_stats()
    check("统计能打成人读的表", "L0 decision path" in txt and "p50" in txt)
    hub.close()


# --------------------------------------------------------------------------
# 11. 协议
# --------------------------------------------------------------------------


def test_protocol() -> None:
    print("\n[11] 上下行协议")
    req = build_escalation_request(
        esc_id=3, level=2, p_esc=0.87, phi={n: 0.5 for n in POTENTIAL_NAMES},
        contributions=[("m_point", 2.1), ("u_conf", 0.4)],
        obs={"seen": 1, "person_conf": float("nan"), "speech": 0},
        ttl_ms=900,
    )
    js = json.dumps(req, allow_nan=False)   # NaN 必须已被清洗，否则这里会抛
    check("请求可严格 JSON 序列化（无 NaN）", js.startswith("{") and "NaN" not in js)
    back = json.loads(js)
    check("请求带齐 level/p_esc/why/ttl/obs",
          back["level"] == 2 and abs(back["p_esc"] - 0.87) < 1e-6
          and back["why"][0]["k"] == "m_point" and back["ttl_ms"] == 900
          and back["obs"]["person_conf"] == 0.0)
    check("schema 版本存在", back["schema"] == "esc/1" and back["v"] == 1)


def test_cloud_backend() -> None:
    print("\n[12] 真实云端后端（无 key 时的行为）")
    be = CloudLLMBackend(api_key="")
    check("没有 key 时 available() 为假", be.available() is False)
    expect_raises("没有 key 时 ask 抛 BackendUnavailable", BackendUnavailable,
                  be.ask, {"id": 1}, 0.2)
    check("探测函数不抛异常（有没有网都能跑）",
          isinstance(probe_cloud_transport(timeout_s=2.0), dict))


# --------------------------------------------------------------------------
# 13. 可训练性
# --------------------------------------------------------------------------


def test_trainable() -> None:
    print("\n[13] 可训练性：权重能从数据里学回来")
    rng = random.Random(7)
    true_w = {"m_speech": 3.0, "u_conf": 2.0, "p_multi": 1.5}
    samples = []
    for _ in range(2500):
        phi = {n: rng.random() for n in POTENTIAL_NAMES}
        z = -1.5 + sum(true_w[n] * phi[n] for n in true_w)
        y = 1 if rng.random() < 1.0 / (1.0 + math.exp(-z)) else 0
        samples.append((phi, y))

    cfg0 = HubConfig(gate_bias=0.0, gate_w={n: 0.0 for n in POTENTIAL_NAMES})
    w, b, info = fit_gate(samples, init=cfg0, epochs=500, lr=0.5)
    check("fit_gate 收敛（准确率 > 0.75）", info["accuracy"] > 0.75,
          f"acc={info['accuracy']:.3f} bce={info['bce']:.4f} pos={info['pos_rate']:.2f}")
    check("学回了注入的三个正权重", all(w[n] > 0.8 for n in true_w),
          ", ".join(f"{n}={w[n]:+.2f}" for n in true_w))
    check("无关势函数的权重被压到 0 附近",
          max(abs(w[n]) for n in POTENTIAL_NAMES if n not in true_w) < 0.6,
          f"max_irrelevant={max(abs(w[n]) for n in POTENTIAL_NAMES if n not in true_w):.2f}")
    check("学回的偏置接近真值 -1.5（负）", b < 0.0, f"b={b:+.2f}")

    # 路由头
    samples2 = []
    for _ in range(1500):
        phi = {n: 0.0 for n in POTENTIAL_NAMES}
        phi["m_speech"] = rng.random()
        phi["u_conf"] = rng.random()
        lvl = 3 if phi["m_speech"] > 0.6 else (1 if phi["u_conf"] > 0.6 else 2)
        samples2.append((phi, lvl))
    rw, rb, info2 = fit_router(samples2, init=cfg0, epochs=400, lr=0.5)
    check("fit_router 学会分层（准确率 > 0.9）", info2["accuracy"] > 0.9,
          f"acc={info2['accuracy']:.3f}")
    check("说话 -> L3 的权重为正", rw[3]["m_speech"] > rw[1]["m_speech"],
          f"L3={rw[3]['m_speech']:+.2f} L1={rw[1]['m_speech']:+.2f}")

    hub = mk_hub()
    hub.cfg.gate_w, hub.cfg.gate_bias = w, b
    hub.cfg.validate()
    check("学到的权重能直接灌回 HubConfig", True)
    hub.close()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    t_start = time.perf_counter()
    print(BAR)
    print("test_hub.py -- DecisionHub 验收测试")
    print(BAR)

    test_routing()
    test_potentials()
    test_timeout_and_fallback()
    test_stale_reply()
    test_l0_never_blocks()
    test_skill_executor()
    test_instruction_validation()
    test_invalid_obs_fuzz()
    test_latched()
    test_dwell()
    test_stats()
    test_protocol()
    test_cloud_backend()
    try:
        test_trainable()
    except Exception as e:  # noqa: BLE001
        check("可训练性测试执行", False, f"{type(e).__name__}: {e}")

    dt = time.perf_counter() - t_start
    print(f"\n{BAR}")
    total = PASS + FAIL
    print(f"结果: {PASS}/{total} PASS, {FAIL} FAIL   用时 {dt:.2f}s")
    if FAILED:
        print("失败项:")
        for n in FAILED:
            print(f"  - {n}")
    print("*** PASS ***" if FAIL == 0 else "*** FAIL ***")
    print(BAR)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
