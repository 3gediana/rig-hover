# 固件改动清单（escalation 通道）

> 本文只描述**要改哪里、改成什么样**，不改任何固件源码。
> 目标：给 `control_arbiter` 增加第 4 个控制源 `ARB_SRC_ESCALATE`，并打通
> 「机器人本地判定 → PC/云端 → 结构化动作程序 → 本地按 50Hz 播放」这条链路。
>
> 所有行号基于当前 `D:\robot_work\rig_omni` 的快照。

---

## 0. 一句话方案

**升级决策留在机器人上**（`cfc_task` 里，~150 周期），
机器人只把「我处理不了，请给我一条动作程序」这件事**暴露**成 `GET /esc`；
PC/云端**轮询**取走、算完，用 `POST /upper` 回一条**已经落到 CfC 动作空间**的原语程序；
机器人本地用 `upper_exec` 按 50Hz 播放，走和 CfC 完全相同的仲裁与安全壳。

**为什么不是机器人主动 POST 给 PC**：那需要在固件里加 HTTP 客户端、
需要处理重试/断连/多后端，而收益只是省掉轮询间隔。轮询间隔 20ms 相对 L2 的 100ms 可以忽略，
且「PC 挂了」这件事在轮询模型下自动表现为「没人取走 → 按 ttl 超时 → 落 L1」，
不需要任何额外的健康检查代码。

---

## 1. 不改什么（先划边界）

| 不动的东西 | 原因 |
|---|---|
| `robot_control()` 的 6ms 主环 | 升级通道完全不进这个环，只在里面读一个已发布的快照 |
| 原 LQR 平衡环 | 与本次改动无关 |
| `safety_shell()` 的检查顺序与语义 | escalation 是「又一个控制源」，它必须过同一道壳 |
| `cfc_model.c` 的 `CFC_BLOB_VER` | **绝对不要升版本**：`cfc_model.c:480` 是严格相等判断，升版会让已在 flash 里的模型全部失效 |
| `cfc_feat_ingest()` 的 `/feat` 二进制帧格式 | 语义标志改走独立的 `POST /sem`（见 §3.3），不动已测通的链路 |
| `arbiter_post_raw()` 的 NaN 拒绝逻辑 | 新通道复用同一套校验思路 |

---

## 2. `main/boards/hover/control_arbiter.h`

### 2.1 控制源枚举（第 30–34 行）

```c
enum {
    ARB_SRC_MANUAL   = 0,
    ARB_SRC_TEACHER  = 1,
    ARB_SRC_CFC      = 2,
    ARB_SRC_ESCALATE = 3   /* 上层（PC agent / 云端）下发的动作程序 */
};
```

### 2.2 阻塞原因位（第 137–144 行）

```c
#define ARB_BLK_ESC       (1 << 6)  /* 上层指令过期/为空 */
#define ARB_BLK_ESCWAIT   (1 << 7)  /* 正在等上层，主动清零车轮 */
```

`ARB_BLK_ESCWAIT` 是**主动**行为（不是故障）：等待期把 `vx/vyaw` 清零是设计，
但要能在 `/status` 里和「指令没了」区分开。

### 2.3 配置字段（`arb_cfg_t`，第 50–74 行）

```c
int   esc_en;           /* 1 = 允许 escalation 抢占（默认 0，先关着上线） */
float esc_p_enter;      /* 升级门限，对应 cfg.esc_p_enter，默认 0.30（先验阈值见 DESIGN §3.4） */
float esc_p_rearm;      /* 事件复位门限，默认 0.30 */
int   esc_dwell_ms;     /* 判据需在门限上停留多久，默认 150 */
int   esc_cooldown_ms;  /* 一次升级结束后多久不再升级，默认 600 */
int   upper_ttl_ms;     /* 上层指令的有效期上限，默认 2500 */
int   upper_max_level;  /* 允许的最高层，默认 3 */
```

全部要走 `control_arbiter.c` 的 `s_fields[]`（第 60 行）和 `cfg_defaults()`（第 91 行）注册，
这样 `GET/POST /cfg` 自动生效、遥测自动可见——**不要再加一套并行的参数表**。

### 2.4 生产者 / 查询接口（第 119–135 行附近）

```c
/* upper_exec 每 50Hz 调一次：把"当前该执行的程序步"贴进来。
 * 语义等价于 arbiter_post_raw()，但写的是独立的 upper 槽位。 */
void arbiter_post_upper(const ctrl_out_t *cmd);

/* 上层指令是否还有效（ttl 内且非空）。safety_shell 用它决定要不要置 ARB_BLK_ESC。 */
bool arbiter_upper_valid(void);
int  arbiter_upper_ttl_left_ms(void);   /* <0 = 没有有效指令 */

/* /status 用：最近一次升级判定的快照 */
typedef struct {
    float p_esc;        /* 0..1 */
    int   level;        /* 路由器给出的层级 0..3 */
    int   reason;       /* 主要风险族的位掩码，见 esc_hub.h */
    int64_t t_ms;       /* 判定时刻 */
    int   pending;      /* 1 = 有未被取走的升级包 */
} arb_esc_state_t;
const arb_esc_state_t *arbiter_esc_state(void);
void arbiter_esc_publish(float p_esc, int level, int reason);  /* cfc_task 调 */
```

---

## 3. `main/boards/hover/control_arbiter.c`

### 3.1 源切换的边界检查（第 245 行）

```c
if (src < ARB_SRC_MANUAL || src > ARB_SRC_ESCALATE) {   /* 原为 > ARB_SRC_CFC */
    return -1;
}
```

同时在第 249–255 行的「丢弃上一源的陈旧指令」块里，把 upper 槽位也清零
（和 `s_teacher_ms = 0` 并排加 `s_upper_ms = 0`），否则新源第一拍会继承旧的头角。

### 3.2 `safety_shell()`：按源选槽位（第 383 行）

```c
/* 原： (void)raw_read(&raw);
 * 改为按源读不同的生产者槽位： */
if (s_cfg.src == ARB_SRC_ESCALATE) {
    (void)upper_read(&raw);
} else {
    (void)raw_read(&raw);
}
```

**后面所有的检查一行都不要动**——桌沿、摔倒、NaN、视觉阶梯、限幅、`mode/emote` 合法性，
escalation 通道必须和 CfC 过完全相同的壳。这是本改动最重要的一条纪律。

### 3.3 看门狗（第 419–435 行，在 teacher / cfc 两个分支后面加第三个）

```c
} else if (s_cfg.src == ARB_SRC_ESCALATE) {
    if (!arbiter_upper_valid()) {
        o->vx_des = 0.0f;
        o->vyaw_des = 0.0f;
        block |= ARB_BLK_ESC;
    }
}
```

注意它**不**做 `o->head_des = 0.0f`：上层程序跑完时头应该停在最后的位置，
而不是「唰」地回中（回中由 `head_lpf` 慢慢来，或者由 CfC 下一拍给新目标）。

### 3.4 `arbiter_tick_2ms()`（第 470 行）

**不动。** 它读 `arbiter_get()`，而 `arbiter_get()` 拿到的已经是 `safety_shell()`
处理过的、源无关的 `ctrl_out_t`。escent 通道天然复用同一套
`stable_pos/stable_yaw` 积分、`yaw_ctrl_time` 刷新、`target_head_pos` 写入。

> 唯一要确认的是第 523 行的 `arbiter_feature_fresh()` 前置条件：
> 视觉不新鲜时积分被冻结，这对上层程序同样成立（上层指令也要靠视觉闭环）。

---

## 4. 模型 blob：**不升版本**也能装下升级头

`cfc_model.h` 的头部已经有 `reserved`（offset 18）和一个 8 通道的 head 表，够用。

| 项 | 改法 |
|---|---|
| `H_RESERVED`（`cfc_model.c:49`） | 复用为 `H_ESC_DIM`。**当前代码不校验这个字段**（已核对，`cfc_model.c:485-510` 没读它），老 blob 里它是 0，正好等于「没有升级头」，向后兼容 |
| `flags` | 新增 `FLAG_ESC 0x0004`（现有 `FLAG_PROJ 0x0001` / `FLAG_RET 0x0002`） |
| 层 id | 新增 `L_ESC = 14`（现有 0..13 已占用，`CFC_MAX_LAYERS=16` 还有余量） |
| 层内容 | `esc_w[12]` + `esc_b[1]` + `route_w[3][12]` + `route_b[3]` = **37 个 float（148 字节）** |
| `head_act` | **不动**（仍只允许 0/1/2）。可选的「有界残差」用现有 `act=1`(tanh) 的头，见 design.md §3.8 |

解析侧要加的三处（都在 `cfc_model.c` 的 validate 里）：

```c
const int esc_dim = rd_u16(blob + H_RESERVED);        /* 0 或 12 */
if (esc_dim != 0 && esc_dim != ESC_PHI_DIM) {         /* 12 */
    set_err("bad esc_dim"); return -1;
}
if ((flags & FLAG_ESC) && !layer_ok(blob, len, L_ESC,
        (uint32_t)(esc_dim + 1 + 3 * esc_dim + 3))) {
    set_err("esc layer size mismatch"); return -1;
}
```

`cfc_model_t` 加：

```c
int          esc_dim;         /* 0 = 没有升级头 */
const float *esc_w, *esc_b;   /* esc_dim, 1        */
const float *route_w, *route_b;/* 3*esc_dim, 3     */
```

打包器（PC 侧）只需要多写一层；`HEAD_ACT=3` 这种新增枚举**不要做**，
否则要改 `cfc_model.c:556` 的校验和 `cfc_model.c:987` 的 switch，收益为零。

---

## 5. `cfc_task.c`：把判定算出来并发布

### 5.1 每拍多做三件事（`cfc_task_main`，第 504 行，`cfc_model_forward()` 之后）

```c
/* 1) 12 个势函数（新文件 esc_hub.c，见 §6） */
esc_phi(&s_esc, &feat, &o, obs, age_ms, &phi);

/* 2) 门 + 路由（37 MAC + 1 sigmoid + 3 exp） */
esc_decision_t d;
esc_step(&s_esc, phi, backends_available, &d);   /* d.p_esc, d.level, d.reason */

/* 3) 发布：无论 200ms 内没人取走都照常覆盖，最新的才是对的 */
arbiter_esc_publish(d.p_esc, d.level, d.reason);
esc_publish_packet(&s_esc, &d, &feat, &o);
```

### 5.2 关键：**escalation 生效期间 CfC 要继续跑**

即使 `src == ARB_SRC_ESCALATE`，`cfc_task` 也要照常 `arbiter_post_raw()`。
两个理由：
1. CfC 的隐状态要保持热（程序播完要能立刻接手，而不是从零状态重新收敛）；
2. 上层程序一旦过期，下一拍就能无缝切回 CfC——这就是「L0 永不等待」在固件侧的形态。

### 5.3 干预回声（可选，v2 模型）

新增一维观测 `obs[106] = 1.0` 表示「最近 N 拍内被上层接管过」。
这需要 `input_dim` 从 106 变 107，属于**新模型**，老 blob 不受影响
（`build_obs()` 已经会按 `m->input_dim` 截断/补零，`cfc_task.c:414-419`）。
本阶段不做，但建议把这一维的位置先留出来（放在 106，不插在中间）。

### 5.4 遥测

`cfc_feat_reply_t` 加一个 `int esc_level;`（`cfc_task.h`），
`/feat` 的响应里带上，PC 侧就知道机器人当前在哪一级。

---

## 6. 新文件 `main/boards/hover/esc_hub.c/.h`（约 250 行）

```c
#define ESC_PHI_DIM 12

typedef enum { ESC_FAM_BLIND=0, ESC_FAM_UNKNOWN, ESC_FAM_LANGUAGE,
               ESC_FAM_PLANNING, ESC_FAM_STAKES, ESC_FAM_CONTEXT } esc_family_t;

typedef struct {
    float phi[ESC_PHI_DIM];      /* [0,1] */
    float p_esc;
    int   level;                 /* 0..3 */
    int   reason;                /* 主要风险族，供 /status */
    /* 平滑状态（离散检测量的一阶积分） */
    float ema[ESC_PHI_DIM];
    /* 事件闩锁 */
    int   armed;
    int   last_level_req;
    int64_t t_above_ms, t_last_esc_ms;
} esc_state_t;

void  esc_reset(esc_state_t *s);
void  esc_phi(esc_state_t *s, const feat_snap_t *f, const ctrl_out_t *cfc,
              const float *obs, int age_ms, float *phi);
void  esc_step(esc_state_t *s, const float *phi, int level_mask, esc_decision_t *d);
```

实现要点（和 Python 逐行对应，`hub.py::compute_phi` 是权威）：

- `smoothstep(x, lo, hi)`：`t=clamp((x-lo)/(hi-lo),0,1); return t*t*(3-2*t);`
  **无反三角、无 exp**，一段三次多项式，~6 个周期。
- 12 个势函数的常量直接照抄 `hub.py` 的表（0.55/0.15、250/1200、0.15/0.50、2.5/5.0 …），
  建议定义成 `static const float esc_ramp[][2]`，**不要在 C 里重新拍数**。
- `esc_step()`：`z = b + Σ w·φ`，`p = 1/(1+expf(-z))`；
  路由 `softmax` 可以只算相对值（减最大值再 exp），3 次 `expf` ≈ 600 周期，20ms 一次可以接受。
- 阈值/代价 `C_L` 也从 blob 里读（放进 `esc_w` 后面），方便不重烧调参。

**内存**：`esc_state_t` ≈ 12×4×2 + 杂项 ≈ 128 B，栈上分配即可。

---

## 7. `main/boards/hover/upper_exec.c/.h`（约 150 行）

```c
int  upper_load(const uint8_t *json, size_t len, char *err, size_t err_len); /* 校验并装载 */
void upper_tick_50hz(void);   /* 由 cfc_task 驱动，每 20ms */
void upper_stop(void);        /* ttl 到期 / 被 CfC 抢占 */
int  upper_ttl_left_ms(void);
```

- 播放规则和 `hub.py::SkillExecutor` 完全一致：`head` 线性插值到目标角、
  `yaw` 带头部反向补偿、`vx` 恒速、`emote` 保持、`hold/stop` 清零、`track` 立即交还。
- 每拍产出 `ctrl_out_t` → `arbiter_post_upper()`。**不要**直接写 `stable_yaw`。
- 校验就是 Python `parse_instruction()` 的 C 版：op 白名单 + 每个字段的范围检查，
  任何一项不过就整条拒绝（返回 `{"ok":0,"err":"..."}`），**绝不部分执行**。
- 程序播完或 ttl 到期 → `upper_stop()` → 下一拍 `arbiter_upper_valid()==false`
  → 安全壳置 `ARB_BLK_ESC` → 源可以自动切回 CFC（可选，见 §8.3）。

---

## 8. `main/boards/hover/hover_debug_server.cc`

### 8.1 端点（第 1043–1088 行）

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/esc` | 取出待处理的升级包；没有则 `204 No Content` |
| `POST` | `/esc?on=1\|0` | 开/关 escalation（等价于写 `cfg.esc_en`） |
| `POST` | `/esc?ack=<id>` | 上层确认已取走（机器人停止重复给出同一个包） |
| `POST` | `/upper` | **上层下发动作程序**（JSON body） |
| `GET` | `/upper` | 当前程序 + 剩余 ttl |
| `POST` | `/sem` | PC 侧语义标志（speech/pointing/…），允许迟到 |

`config.max_uri_handlers = 20`（第 1043 行）→ **24**；已注册 18 个，新增 6 个。

### 8.2 `GET /esc` 的响应体（就是 Python 侧 `build_escalation_request()` 的输出）

```json
{
  "schema": "esc/1", "v": 1, "id": 17, "level": 2, "p_esc": 0.802, "ttl_ms": 900,
  "phi": {"u_conf":0.0,"u_lost":0.0,"u_clip":0.0,"u_ood":0.0,"m_speech":0.0,
          "m_point":1.0,"m_stale_sem":0.0,"p_multi":0.0,"p_lookup":0.0,
          "s_irrev":0.0,"s_approach":0.0,"c_local_ok":0.0},
  "why": [{"k":"m_point","c":4.40},{"k":"c_local_ok","c":-1.00}],
  "obs": {"seen":1,"person_conf":0.93,"theta_img":9.0,"q_head":4.0,
          "clip_ratio":0.0,"novelty":0.0,"feat_age_ms":22.0,"dist_est":1.18,
          "speech":0,"pointing":1,"object_held":0,"point_dir_deg":32.0,
          "instr_clauses":0,"lookup":0,"mode":1,
          "action":{"head_des":9.0,"vyaw_des":0.0,"vx_des":0.0,"mode":1,"emote":0,"action_conf":0.55}},
  "fallback": [], "capture": "GET http://<robot>/capture"
}
```

**体积**：约 700 B（`phi` 和 `why` 占大头）。若嫌大，可以 `GET /esc?lite=1`
只回 `id/level/p_esc/why/obs`，`phi` 用位图近似。当前不做。

### 8.3 `POST /upper` 的请求体

```json
{"id":17,"level":2,"intent":"look_at_target",
 "prog":[{"op":"head","deg":40.0,"ms":300},{"op":"hold","ms":250}],
 "ttl_ms":1500,"say":null,"confidence":0.66,"args":{"target":"pointed_object"}}
```

响应：`{"ok":1,"id":17,"steps":2,"ttl_ms":1500}` 或 `{"ok":0,"err":"prog[0].deg out of range"}`。

**`id` 必须和最近一次 `/esc` 的 `id` 一致**，否则 400——
这是防止「上一轮的回复把这一轮带歪」的唯一手段（Python 侧对应 `stale_reply` 计数）。

### 8.4 `/status` 增补（`status_handler`，第 302 行）

```json
"esc": {"p_esc":0.80,"level":2,"reason":"language","pending":1,"ttl_left_ms":760},
"upper": {"active":1,"intent":"look_at_target","step":"1/2","ttl_left_ms":1400}
```

### 8.5 `POST /sem`（新，语义通道路由）

```json
{"t_ms":1730000000123,"speech":1,"pointing":0,"object_held":1,
 "point_dir_deg":32.0,"instr_clauses":1,"lookup":0}
```

- 独立端点而不是塞进 `/feat` 的二进制帧：**不碰已经测通的链路**，
  而且这个通道本来就是 best-effort 的（它迟到 / 缺失本身就是一个势函数 `m_stale_sem`）。
- 固件只存 `{stamp, 7 个字段}`，`esc_phi()` 用它和 `now_ms` 算 `sem_age_ms`。

---

## 9. 任务与时序（别搞错）

```
Core 0                                Core 1
├─ WiFi / HTTP                        ├─ robot_control()   ~6ms   ★优先级最高
│  ├─ GET /esc   → 读快照（不加锁）      │   读 arbiter_get()（无锁）
│  ├─ POST /upper → 校验 → upper_load  ├─ cfc_task  50Hz
│  ├─ POST /sem  → 写语义快照           │   ├─ build_obs + cfc_model_forward  ~1.5ms
│  └─ /status    → 读快照               │   ├─ esc_phi + esc_step              ~0.01ms
└─ （无新增任务）                        │   ├─ arbiter_post_raw()        ← 永远发
                                        │   └─ upper_tick_50hz()         ← 有程序时才动
                                        └─ （upper_exec 不单独开 task）
```

- **不要给 escalation 单独开 task。** 它的全部工作（~150 周期）挂在 `cfc_task` 的
  现有周期里就够，多一个 task 只是多一处优先级反转的机会。
- `upper_tick_50hz()` 放在 `cfc_task` 里而不是 `robot_control()` 里，
  因为播放动作和 CfC 推理是同一个时间尺度（20ms），而控制环是 6ms。
- 跨任务数据继续用 seqlock（`raw_write`/`raw_read` 那套），**不要引入互斥量**。

---

## 10. 烧录后怎么验证（不用编译就能全测）

```bash
# 1) 打开 escalation，确认配置项都在
curl -X POST 'http://<robot>/cfg?esc_en=1'
curl 'http://<robot>/cfg' | grep esc_

# 2) 注入一次「本地处理不了」的场景：让 PC 侧把 pointing 置 1
curl -X POST http://<robot>/sem -d '{"speech":0,"pointing":1,"point_dir_deg":30,"instr_clauses":0}'
curl 'http://<robot>/esc'                 # 应在 150ms 内出现升级包，level=2
curl 'http://<robot>/status' | grep '"esc"'

# 3) 上层回一条程序
curl -X POST 'http://<robot>/upper' -d '{"id":<id>,"level":2,"intent":"look_at",
  "prog":[{"op":"head","deg":30,"ms":300}],"ttl_ms":1500}'
curl 'http://<robot>/upper'                # 应看到 active:1 且 step 在推进
curl -X POST 'http://<robot>/mode?src=escalate'

# 4) 超时行为：不发 /upper 的 ack，等 ttl 到期
sleep 3; curl 'http://<robot>/status' | grep -E 'ARB_BLK|esc'
# 期望：block 里出现 ESC 位，vx/vyaw 归零，头停在最后位置

# 5) 安全壳没被绕过：桌沿 + escalation 同时开着
curl -X POST 'http://<robot>/gaze?danger=1'
# 期望：即使 src=escalate，vx/vyaw 仍然被立刻清零（ARB_BLK_GAZE 优先）
```

---

## 11. 风险与未做的事

1. **`/esc` 轮询会浪费带宽**：700B × 20Hz = 14KB/s。可以只在一个短窗口内轮询
   （PC 侧看到 `/status.esc.pending=1` 再来取），但那就多了一拍延迟。
   建议先按最简实现跑，实测后再优化。
2. **升级包的重复读取**：`GET /esc` 是幂等的，同一个 `id` 在 ack 之前可以被多次取走。
   上层必须用 `id` 去重（Python 侧 `Instruction.esc_id` 已经在做）。
3. **没有做干预回声**（§5.3）。上层接管结束后 CfC 会按自己的计划走，
   目前只靠 `head_lpf` 和 3s `stale_yaw` 冻结窗口缓解。这是**已知的行为级缺陷**，
   修它需要重训模型（107 维观测）。
4. **`P_fix` 的先验矩阵是手写的**（design.md §3.4）。固件侧只负责执行它，
   真正的风险在 PC 侧训练与校准。
5. **没有做「机器人主动放弃升级」**：如果 `esc_en=1` 但 PC 一直不取，
   行为等价于「每次升级都超时 → 落 L1」，会看到机器人反复做扫描技能。
   建议上线时先 `esc_en=0`，用 `/status` 观察 `p_esc` 的分布再打开。
