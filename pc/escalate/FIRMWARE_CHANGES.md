# FIRMWARE_CHANGES.md — 让固件消费第 6 个头 `escalate`

> 一句话：**blob 格式一个字节都不用改**（`n_heads: 5 → 6`，第 6 个头
> `head_dim=1 / head_act=2`），固件只改 3 处：`cfc_task.c::publish()` 多做一次
> sigmoid、`control_arbiter.{h,c}` 加一个 `p_esc` 通道、`/status` 多两个字段。
> `cfc_model.c`（解析器 + 前向）**不需要改动**。
>
> 本文只描述要改什么，**没有真的改固件源码**（`rig_omni/` 有并行任务）。

---

## 0. 结论表

| 文件 | 改动 | 行数量级 | 必做？ |
|---|---|---|---|
| `cfc_model.h` | 无（`CFC_MAX_HEADS=8` 已够） | 0 | — |
| `cfc_model.c` | 无（`n_heads`/`head_dim`/`head_act` 都是数据驱动） | 0 | — |
| `cfc_task.c` | `publish()` 里读第 6 个头 + 一次 sigmoid | ~8 行 | ✅ |
| `control_arbiter.h` | `ctrl_out_t` 加 `float p_esc;`；加 `arbiter_esc_publish()` 声明 | ~4 行 | ✅ |
| `control_arbiter.c` | 转存 `p_esc`；`/status` 输出 | ~6 行 | ✅ |
| `cfc_task.h` | `cfc_feat_reply_t` 加 `float p_esc`（遥测） | ~1 行 | 建议 |
| `esc_logic.c/.h`（新，可选） | 阈值/dwell/闩锁（策略，不是模型） | ~60 行 | 见 §5 |

---

## 1. blob：格式零变更

第 6 个头只是 `n_heads` 从 5 变 6，头表里多一项：

| 字段 | v1（5 头） | v2（6 头） |
|---|---|---|
| `n_heads`（off 16） | 5 | **6** |
| `head_dim[8]`（off 20） | `[1,1,4,8,1,0,0,0]` | **`[1,1,4,8,1,1,0,0]`** |
| `head_act[8]`（off 36） | `[1,1,2,2,1,0,0,0]` | **`[1,1,2,2,1,2,0,0]`** |
| `reserved`（off 18） | 0 | 0（**不占用** `H_RESERVED`） |
| `layer_prec[16]`（off 2240） | 全 0 | 全 0（本工程全 float32，不做量化） |
| `heads` 层长度 | 3900 B（975 float） | **4160 B（1040 float）** |
| blob 总长 | 133388 B | **133648 B**（+260 B = 65 个 float） |
| CRC32 | `0xFB5F0AE2` | 重算（`zlib` 口径，自身 4 字节当 0） |

> **头长注意**：本工程开工时头是 2240 字节；并行任务把它改成了 **2256**
> （末尾追加 `layer_prec[16]`，见 `cfc_model.h` "PER-LAYER PRECISION"）。
> 原有 18 个字段的偏移一个都没动，所以本工程只多写 16 个 0、payload 基址
> 从 2240 变 2256。若手上的固件还是 2240 那一版，把
> `export_escalate.py` 的 `CFC_HDR_BYTES` 改回 2240 并去掉 `layer_prec`
> 那一项即可，其余字节完全相同。

层 id（0..13）一个都没变，层数仍是 14 ≤ `CFC_MAX_LAYERS`。
`pc/train/export_blob.py` 的打包函数**不需要修**：它本来就是按
`policy["n_heads"]/head_dim/head_act` 走数据驱动的。本目录的
`export_escalate.py` 独立实现了同一套偏移表，并且与它**逐字节一致**
（`test_escalate.py` §5 每次跑都验证这一点，当前 133648 B / CRC `0x…`）。

---

## 2. `head_act[5]` 用 2 而不是新增 3 —— 这是本次唯一需要判断的地方

两种编码在数值上完全等价（我们两种都实现了，`p_esc` 逐位相同），差别在**部署风险**：

| | `head_act=2`（采用） | 新增 `head_act=3`（不采用） |
|---|---|---|
| `cfc_model.c` 改动 | **0 行** | 2 行（`a > 2` 改 `a > 3`；`switch` 加 `case 3`） |
| 老固件加载新 blob | **成功**，`n_heads=6` 但只读前 5 头，第 6 头被忽略 | **失败**：`blob_parse()` 直接 `set_err("bad head_act")` 拒绝整份模型 |
| 语义 | "激活交给消费方"——**softmax 头本来就是这个约定**（`cfc_model.c` 保留 logits，由 `cfc_task.c` 做 softmax/argmax） | 头自解释 |
| sigmoid 在哪算 | `cfc_task.c`，每拍 1 次 `expf` | `cfc_model.c`，每拍 1 次 `expf` |

关键细节：`head_act=2` 的**消费方归一化要按维度分派**——
`dim > 1` 是 softmax，`dim == 1` 必须是 **sigmoid**。
`softmax` 归一化到 1 类恒等于 `1.0`，没有信息；所以"用 softmax 的 1 类变体"
这句话的正确实现就是"保留 logit + 消费方 sigmoid"。这段语义已经写进
`export_escalate.py::consumer_act()` 和 `export_escalate.py` 的模块文档，
`test_escalate.py` §5 也断言 `act=2` 与 `act=3` 两种 blob 给出同一个 `p_esc`。

如果将来一定要头自解释，按 §2 右列改 2 行即可，**blob 侧只要把
`--act-3` 打开重导一次**（其余字节完全相同）。

---

## 3. `cfc_model_forward()`：不用改

第 3 步的头部循环是数据驱动的：

```c
for (i = 0; i < m->n_heads; i++) {          /* 自动覆盖第 6 个头 */
    ...
    switch (m->head_act[i]) { case 1: tanhf(...); case 2: break; default: break; }
    out[i * CFC_MAX_HEAD_DIM + k] = acc;    /* 第 6 头在 out[5*16 + 0] */
}
```

唯一值得加的是编译期断言（`CFC_MAX_HEADS >= 6`），但 `cfc_model.h` 里
`CFC_MAX_HEADS` 已经是 8。

> **`out` 的读法**：`out[5 * CFC_MAX_HEAD_DIM + 0]`，即偏移 `80`，**不是** `out[5]`。

---

## 4. `cfc_task.c::publish()`：唯一必改的消费点

现在 `publish()` 只处理到第 5 个头（`vx`，`cfc_task.c:471`）。加：

```c
/* 6) escalate: head_act[5]=2 => 模型只给 logit，sigmoid 在消费方做 */
if (m->n_heads > 5 && m->head_dim[5] == 1) {
    float z = out[5 * CFC_MAX_HEAD_DIM];
    if (!isfinite(z)) z = 0.0f;                 /* 不许把 NaN 传进策略层 */
    if (z < -30.0f) z = -30.0f; else if (z > 30.0f) z = 30.0f;
    s_p_esc = 1.0f / (1.0f + expf(-z));         /* ~60 周期，50 Hz 可忽略 */
} else {
    s_p_esc = 0.0f;                             /* 老 blob：没有升级头，永不升级 */
}
o.p_esc = s_p_esc;
```

要点：

1. **`n_heads > 5` 的判断是向后兼容的全部**：老 blob 加载进来 `n_heads=5`，
   `s_p_esc` 恒 0，行为与今天完全一致。
2. 夹 ±30 是为了 `expf` 不溢出；`blob_parse()` 里已经拒绝非有限权重，这里只是兜底。
3. 不需要 `argmax`。`head_dim=1`，只有一个值。
4. `p_esc` 是**模型输出**，不是判决：不要在这里做阈值比较（见 §5）。

---

## 5. `control_arbiter.{h,c}`：把 `p_esc` 变成一条通道

### 5.1 `ctrl_out_t` 加一个字段

```c
typedef struct {
    float head_des, vx_des, vyaw_des, p_esc;   /* p_esc: [0,1], new */
    int   mode, emote;
} ctrl_out_t;
```

`arbiter_post_raw(const ctrl_out_t *raw)` 已经是整结构传递，加字段自动带出去，
`arbiter_get()` / `/status` 也自动能看到（`control_arbiter.h:120`）。

### 5.2 生产者接口（与 `decision_hub` 的接口同名，方便两套实现互换测试）

```c
/* p_esc: 模型输出；level: 0..3（0=不升级）；reason: 主要风险族下标 */
void arbiter_esc_publish(float p_esc, int level, int reason);
int  arbiter_esc_pesc_x1000(void);     /* /status 用整数，省一个 printf("%f") */
```

### 5.3 策略不要塞进模型

`p_esc` 到"要不要升级、升到哪一级"之间还有一层策略（阈值 / dwell / 事件闩锁 /
层级路由 / 安全否决）。它**不属于**这次的头，建议仍放在
`decision_hub/FIRMWARE_CHANGES.md §6` 的 `esc_hub.c` 里，只是把
"算 12 个势函数 + 13 个乘加"那一段换成"读一次 `p_esc`"：

```c
/* 旧： esc_phi(...) + esc_step(...)  ->  d.p_esc
 * 新： d.p_esc = arbiter_esc_pesc();  esc_step_threshold(d.p_esc, &d); */

d.level = 0;
if (p_esc >= p_thr) { above_ms += dt_ms; } else { above_ms = 0; }
if (above_ms >= cfg.esc_dwell_ms && armed) { d.level = route_level(...); armed = 0; }
if (p_esc < cfg.esc_rearm) armed = 1;
```

阈值 `p_thr` 必须**按混合比例重新标定**。实测（`test_escalate.py` §4）：
在 1% 误报率下阈值 ≈ 0.5（见 README §4 的实测数字），这与 `hub.py` 里
由代价比推出来的 `τ₂=0.107` 不是一个口径——后者是"p_esc 已经是校准概率"时的
贝叶斯阈值，而这里 p_esc 的基率被训练集的混合比例影响。**改基率就重拟一个
float（`head_b[5]`），不用重训**，见 README §5。

### 5.4 `src == ARB_SRC_ESCALATE` 期间 CfC 必须继续跑

这条在 `decision_hub/FIRMWARE_CHANGES.md §5.2` 已经写过，现在更硬：
`p_esc` 本身就是 CfC 的输出，**停跑 CfC 就等于停掉升级判据**，
而且等上层超时回来时隐状态已经冷了。

---

## 6. 遥测（建议）

```c
typedef struct {
    ...
    float p_esc;      /* 新增：本拍升级概率 */
    int   esc_level;  /* 新增：当前所处层级 0..3 */
} cfc_feat_reply_t;
```

`/status` 同时建议上报 `esc_logit` 与保留门均值 `lambda_mean`
（`ret` 那 64 个 float 的 sigmoid 均值）：这两个量是"头还在不在工作"的
最便宜的健康指标——λ 掉回 0.04 附近说明隐状态又变成无记忆的了
（见 README §5 的诊断方法）。

---

## 7. 不用硬件就能做的回归（照 `pc/train/hoststub` 的路子）

1. **加载**：用现有的 hoststub 编 `cfc_model.c` + 新 blob，断言
   `n_heads==6`、`head_dim[5]==1`、`head_act[5]==2`、`crc` 通过。
2. **数值**：把 `export_escalate.py` 的 numpy 参考实现当 oracle（它已经和
   torch 对拍到 1e-6），喂同一份观测，断言 `max|Δp_esc| < 1e-5`。
3. **向后兼容**：老 bin 加载新 blob 应**成功**（`n_heads=6`，前 5 头正常）。
4. **向前兼容**：新 bin 加载老 blob 应**成功**且 `p_esc ≡ 0`。
5. **域外**：喂 `NaN`/`Inf`/超量程观测，`p_esc` 必须仍是有限值
   （`cfc_model_forward` 的域外守护 + §4 的 `isfinite` 兜底两道）。

---

## 8. 成本

| 项 | 增量 |
|---|---|
| Flash（blob） | **+260 B**（65 个 float） |
| RAM | 0（不新增状态；`ctrl_out_t` +4 B） |
| 每拍算力 | **64 MAC + 1 `expf`** ≈ 120 周期（240 MHz 下 0.5 µs，占 20 ms 周期的 0.0025%） |
| 代码 | `cfc_task.c` ~8 行，`control_arbiter.*` ~10 行 |
