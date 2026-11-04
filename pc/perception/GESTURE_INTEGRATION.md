# 手势事件化：感知 → 固件 → 训练 的对接说明

本轮把 `pc/gesture` 的「分类器 + 事件机」接进了感知链和教师控制器。
**一句话**：手势判定从「`wave_score > 4` 电平 + 冷却」改成「`gesture_onset` 边沿」，
106 维观测**维度不变**，只换掉 72~75 四个槽位的含义。

读这份文档前先读 `pc/gesture/README.md` 与 `pc/gesture/reports/events.md`（标定数据）。

---

## 1. 感知侧：`PerceptionOut`

### 1.1 新增的 5 个字段（事件机输出，逐帧）

| 字段 | 类型 | 语义 | 取值 |
|---|---|---|---|
| `gesture_id` | int | 当前手势 | 0=none 1=wave 2=open_palm 3=thumbs_up 4=pointing 5=fist |
| `gesture_onset` | int | **边沿脉冲**：进入 GESTURE 那一帧 = 1，其余 = 0 | 0/1 |
| `gesture_phase` | int | 状态机相位 | 0=IDLE 1=PENDING 2=GESTURE 3=RETURN |
| `gesture_conf` | float | 事件机给出的置信度（GESTURE 相位是已确认类的平滑概率） | 0..1 |
| `gesture_gap_s` | float | 距上次 GESTURE **结束**的秒数 | ≥0，上限 `gap_cap_s`(3600) |

三件必须搞清楚的事：

1. **`gesture_onset` 是边沿不是电平。** 一段连续挥手恰好 1 个 onset，两次挥手（间隔
   ≥0.6 s）2 个，静止 10 s 0 个——这是事件机状态机的**结构保证**，不是靠时间窗口。
   所以下游**不需要冷却期**。实测见 `pc/gesture/reports/events.md`。
2. **`gesture_conf` 改变了语义**（同名，但是新的）。旧定义是
   `min(1, wave_score/4) × 手腕关键点置信度`；新定义是事件机对**已确认类**的概率。
   旧值搬到了 `wave_gesture_conf`（deprecated）。
3. **`gesture_gap_s` 在 GESTURE 相位里不会清零**——它记的是「上一次手势**结束**」，
   手势进行中一直往上走。只有进 RETURN 才重置为 0。

### 1.2 旧字段的去留

| 字段 | 状态 | 说明 |
|---|---|---|
| `hand_up_L` / `hand_up_R` | **有效** | `GestureEstimator` 还在算，教师 `require_hand_up` 用 |
| `wave_sin` / `wave_cos` / `wave_amp` | 已废弃 | 只为兼容而留；事件化后没有下游用途 |
| `wave_gesture_conf` | 已废弃 | 旧 `gesture_conf` 的值（改名保留，避免老代码读不到） |
| `debug["wave_score"]` | 已废弃 | 仍然输出，便于对照旧日志 |
| `debug["gesture_probs"]` | 新增 | (6,) 分类器原始概率，排查误判用 |
| `debug["gesture_event"]` | 新增 | 事件机内部量（phase / p_smooth / pending_t / n_onsets …） |

`pc/perception/gesture.py` 整个模块已标 deprecated，但**行为一行没改**
（`test_perception.py` 的 T12~T14 还在测它）。

### 1.3 构造函数

```python
PerceptionPipeline(..., use_gesture=True, gesture_dir=None,
                   gesture_model=None, gesture_device=None, gesture_config=None)
```

* 默认从 `pc/gesture/models/gesture_classifier.pt` +
  `pc/gesture/configs/gesture_events.yaml` 加载；目录可用 `RIG_GESTURE_DIR` 覆盖。
* `use_gesture=False` 时 5 个新字段恒为 0（跑纯视觉基准时省一次前向）。
* 事件机参数**必须**是标定写回的 YAML；文件缺失直接抛 `FileNotFoundError`，
  不静默退回代码里的默认值（默认值未标定，误报率差一个数量级）。
* 单帧开销：分类 1.6 ms（GPU）/ 1.5 ms（CPU）+ 事件机 ~0.02 ms，见 README §5。

---

## 2. 教师控制器

`TeacherObs` 新增 `gesture_onset / gesture_id / gesture_conf / gesture_gap_s`，
`wave_score` 保留但**不参与任何判定**（只为 `pc/train/synth.py` 还能构造观测）。

ACK 的触发条件（`teacher._update_state`）：

```
gesture_onset == 1
and gesture_conf >= ack_conf_thresh          # 0.75
and gesture_id in ack_gesture_ids            # [1,2,3] = wave/open_palm/thumbs_up
and gesture_gap_s >= ack_min_gap_s           # 0.25 s
and (hand_up 或 not require_hand_up)
and degrade == ok and mode == TRACK
```

删掉的东西：`wave_cycles_needed` / `integrate_window_s` / `wave_score_thresh` /
`wave_release` / `refractory_s` 及其冷却状态迁移。`MODE_REFRACTORY` 常量**保留**
（契约 §4.1 的 mode 0..3 是线格式，固件 `obs[102] = mode/3`），但事件化后不再进入。

### `ack_min_gap_s` 与 `refractory_s` 不是一件事

| | 旧 `refractory_s` = 3.0 s | 新 `ack_min_gap_s` = 0.25 s |
|---|---|---|
| 语义 | **冷却**：ACK 后 3 s 内不响应，无论人做几次动作 | **去抖**：只在「上一个手势结束到现在 <0.25 s」时拒绝 |
| 为什么存在 | 旧判定吃 `wave_score` **电平**，挥手期间一直高于阈值，没有冷却会每帧重入 | 挡「同一次动作被切成两个边沿」 |
| 副作用 | 「挥两次手」只响应一次 | 间隔 0.5 s 的两次挥手，两次都响应 |
| 冗余度 | 必需（唯一防线） | 冗余（事件机 `min_gap=0.25 s` 已经保证单段单边沿） |

另有一条设计边界：**ACK 是固定 `ack_hold_s`(1.5 s) 的反馈脉冲**，落在 ACK 窗口内的
新 onset 被吸收（不会延长、也不会叠加第二次 ACK）。要两次响应，两段动作间隔必须
> `ack_hold_s`。已由 `test_teacher.py` 的 B7 钉住。

---

## 3. 固件侧：`cfc_task.c`

### 3.1 结论：**106 维不增加，保持 106**

`70..77` 这 8 个槽位现在是：2 个「举手位」+ 3 个废弃的 `wave_*` + 1 个
`gesture_conf`（而且固件里 `gesture_conf = person_conf`，是占位值）+ 2 个恒 0 的保留位。
事件化需要 5 个值（id / onset / phase / conf / 可选 gap），
**旧 `wave_sin/wave_cos/wave_amp` 让出的 3 个槽 + 现成的 `gesture_conf` 槽**刚好装下
id/onset/phase/conf 四个，`gesture_gap_s` 也有 76 号空位可放——
所以**无论放不放 gap，都是 106**，不需要动 `OBS_DIM`、`CFC_MAX_INPUT_DIM`、
`obs.py` 的 `OBS_DIM`、导出 blob 的 `input_dim`。

### 3.2 槽位对照表（`build_obs()`，cfc_task.c:367-375）

| 槽位 | 旧含义（本行现状） | **新含义** | 写入的归一化值 |
|---|---|---|---|
| 70 | `hand_up_l` | `hand_up_l`（不变） | 0/1 |
| 71 | `hand_up_r` | `hand_up_r`（不变） | 0/1 |
| 72 | `wave_sin` | **`gesture_id`** | `gesture_id / 5.0` → 0,0.2,0.4,0.6,0.8,1.0 |
| 73 | `wave_cos` | **`gesture_onset`** | 0/1 |
| 74 | `wave_amp`（`sqrt(sin²+cos²)` 重算） | **`gesture_phase`** | `gesture_phase / 3.0` → 0,0.33,0.67,1.0 |
| 75 | `gesture_conf`（**= person_conf 占位**） | **`gesture_conf`（事件机真值）** | 0..1 |
| 76 | 恒 0 保留 | 恒 0 保留（**可选**改放 `gesture_gap_s`，见 3.5） | 0 |
| 77 | 恒 0 保留 | 恒 0 保留 | 0 |

为什么 72/74 要除 5 和 3：这两个值域本来就是有界小整数，除以量程后落进 [0,1]，
归一化的 `mean/std` 不会因为「平台切换」而漂；同时它们不再恒 0，
**不需要**碰 `obs.py` 的 `CONSTANT_OBS_DIMS = (76, 77)`。

顺手确认这三个旧槽确实是废的（实测，不是推断）：`obs[74]` 由
`sqrt(wave_sin² + wave_cos²)` 算出，而 `phase` 只在 `freq>0` 时才推进，
所以静止时 `sin/cos = (0,1)`、**这一维恒等于 1.0**，与「挥幅」没有关系；
`obs[75]` 在固件里干脆就是 `person_conf` 的副本。换句话说旧设计里这 3 个
「运动学」槽没有携带过挥手信息，换掉不损失任何东西。

### 3.3 `/feat` 帧格式：复用 16 字节 obj，长度不变

`TECH-PLAN §5.2` 的 obj 是 16 字节（`OBJ_STRIDE_BIG`）。`wave_sin/wave_cos`
两个 i16 正好让给新字段，**不需要加长帧**——固件的
`need = 12 + 16×n_obj + 4 + 2×n_feat` 一个字都不变，`"length mismatch"` 检查不受影响：

| 偏移 | 旧 | **新** |
|---|---|---|
| +0 | `int16 cx` ×1000 | 不变 |
| +2 | `int16 cy` ×1000 | 不变（固件未用） |
| +4 | `int16 w` ×1000 | 不变 |
| +6 | `int16 h` ×1000 | 不变 |
| +8 | `uint8 conf` ×255 | 不变 |
| +9 | `uint8 cls` | 不变（未用，建议填 0） |
| +10 | `uint8 flags`：bit0=hand_up_L bit1=hand_up_R | **bit0/1 不变，新增 bit2-3=`gesture_phase`(0..3)，bit4=`gesture_onset`** |
| +11 | `uint8 _pad` | 不变 |
| +12 | `int16 wave_sin` ×1000 | **`int16 gesture_id`（0..5，不再 ×1000）** |
| +14 | `int16 wave_cos` ×1000 | **`int16 gesture_conf` ×1000** |

`OBJ_STRIDE_SMALL`(12 字节) 那种省略手势的旧形式：建议**拒收**（`reply_fail`），
或至少把 id/onset/phase/conf 全置 0 并打一次日志——否则模型会看到
「有 hand_up 但永远没有手势」的分布，是真机行为与训练分布不一致的经典来源。

### 3.4 `cfc_task.c` 逐处改动清单

1. **`feat_snap_t`（:58-77）**：`float wave_sin/wave_cos/wave_amp` 三个字段删掉，换成
   `int gesture_id; int gesture_onset; int gesture_phase;`；`gesture_conf` 保留原名。
2. **收包解析（:279-288）**：
   ```c
   snap.hand_up_l = (o[10] & 0x01u) ? 1 : 0;
   snap.hand_up_r = (o[10] & 0x02u) ? 1 : 0;
   snap.gesture_phase = (int)((o[10] >> 2) & 0x03u);
   snap.gesture_onset = (int)((o[10] >> 4) & 0x01u);
   if (stride == OBJ_STRIDE_BIG) {
       snap.gesture_id   = clampi((int)rd_i16(o + 12), 0, 5);
       snap.gesture_conf = clampf((float)rd_i16(o + 14) / 1000.0f, 0.0f, 1.0f);
   }
   ```
   同时**删掉**这两行旧补丁：`snap.wave_amp = clampf(sqrtf(...))` 与
   `snap.gesture_conf = snap.person_conf;`（后者是占位，现在有真值了）。
3. **`build_obs()`（:367-375）**：
   ```c
   obs[70] = (float)f->hand_up_l;
   obs[71] = (float)f->hand_up_r;
   obs[72] = (float)f->gesture_id / 5.0f;
   obs[73] = (float)f->gesture_onset;
   obs[74] = (float)f->gesture_phase / 3.0f;
   obs[75] = f->gesture_conf;
   obs[76] = 0.0f;
   obs[77] = 0.0f;
   ```
4. **不用改**：`OBS_DIM`(106)、`FEAT_MAX_DIM`(64)、`CFC_MAX_INPUT_DIM`、
   `m->input_dim` 的截断/补零逻辑、mode/emote 通路、`s_prev_*` 反馈槽。
5. **副作用（不是 bug，但要知道）**：`MODE_REFRACTORY` 不再出现，
   `obs[102] = mode/3` 的取值域从 {0,⅓,⅔,1} 收窄到 {0,⅓,⅔}。
   **旧 blob 必须重训重导**——obs[72..75] 的含义变了，拿旧模型读新观测等于喂噪声。

### 3.5 可选项：要不要把 `gesture_gap_s` 也放进观测？

建议**不放**。它只是教师规则的去抖门（PC 侧），CfC 策略学的是「onset → 动作」，
不需要「上次手势结束多久」。真要放，用 76 号空位：

```
obs[76] = clampf(f->gesture_gap_s, 0.0f, 2.0f);   // 秒，必须钳位
```
钳位是必须的：`gesture_gap_s` 上限是 `gap_cap_s = 3600`，不钳会让这一维
把归一化带飞。放进去之后 `obs.py` 的 `CONSTANT_OBS_DIMS` 要改成 `(77,)`。

---

## 4. 训练侧：`pc/train/obs.py` 需要怎么改（**只说明，本轮没改代码**）

### 4.1 `obs.py` 必改的 4 处

| 位置 | 现状 | 要改成 |
|---|---|---|
| `_build_names()`（:61-106） | `72 wave_sin / 73 wave_cos / 74 wave_amp` | `72 gesture_id_n / 73 gesture_onset / 74 gesture_phase_n`；75 `gesture_conf` 名字不变、语义变 |
| `assemble()`（:266-274） | 写 `hand_up_L/R`、`wave_sin/cos`、`obs[74]=sqrt(sin²+cos²)`、`obs[75]=gesture_conf` | 写 `obs[72]=gid/5`、`obs[73]=onset`、`obs[74]=phase/3`、`obs[75]=conf`，**删掉 sqrt 重算那一行** |
| 模块 docstring 第 2 条（:20-22） | 「obs[74] 是固件用 sqrt 重算的」「obs[75] 固件直接等于 person_conf」 | 两条都不再成立，改成「72/74 是 id/5 与 phase/3，75 是事件机置信度」 |
| `assemble()` 的 `perc` 参数说明（:229-232） | 列出 `wave_sin/wave_cos/gesture_conf` | 列出 `gesture_id/gesture_onset/gesture_phase/gesture_conf` |

**必须与固件逐位一致**：`/5.0`、`/3.0`、`int16/1000` 这三处只要有一边写错，
训练和部署看到的就是两个不同的观测。建议在 `obs.py` 的 CLI 自检里加一条
「72/74 ∈ {0,0.2,…,1}」的形状断言。

`OBS_DIM`(106)、`CONSTANT_OBS_DIMS`(76,77)、`FEATURE_DIM`(64)、`DomainGuard`、
`fit_stats` 都**不用改**（前提：76/77 继续写 0）。

### 4.2 连带影响（`train/` 里其他文件，同样只说明）

1. **`pc/train/synth.py:378-381` —— 这是最容易被漏掉的一条。**
   它用 `wave_score=float(score)` 驱动教师，教师现在**完全不看** `wave_score`，
   所以合成数据里**一个 ACK 都没有**（`mode` 只剩 IDLE/TRACK）。
   修法（推荐）：让 synth 自己跑一遍 `GestureEventMachine`——把 `waving/hand_up`
   映射成 6 类概率（`idle` 或对应手势类 + 少量噪声），喂 `evt.update(probs, dt)`，
   再把 `evt` 的 5 个输出塞进 `TeacherObs`；这样标签和部署走的是**同一个状态机**，
   ACK 的时间点也天然对齐。次选：在 `waving` 上升沿那一帧手动给 `onsets=1`。
2. **`pc/train/test_e2e.py:131`**：「mode 四类都出现过（ACK/REFRACTORY 不能是 0）」
   会失败——REFRACTORY 已经不存在。改成断言「三/四类中实际出现过的类别数」，
   或明确断言 REFRACTORY 帧数为 0。
3. **`pc/train/train_cfc.py:62`** 的类权重 `ACK×20 / REFRACTORY×15` 要重定：
   REFRACTORY 永不出现，权重无意义；ACK 帧占比会略升（REFRACTORY 那 14.6% 现在
   归到 TRACK/IDLE）。
4. **`pc/train/README.md:111`** 的类别分布（IDLE 18.9 / TRACK 59.2 / ACK 7.3 /
   REFRACTORY 14.6）全部作废，重训后重新统计。
5. **blob**：重训后重新导出 `blob_manifest.json` / `model_slot0*.blob`，
   并与 3.1 的固件改动**同一次**上线（旧模型 + 新观测 = 静默失效）。

---

## 5. 上线顺序（顺序错了会静默失效）

1. 感知侧：`run_perception` + `gesture_events`（本轮已完成），跑 `test_gesture.py`
   与感知验收，确认 `gesture_onset` 的计数符合 `reports/events.md`。
2. 教师侧：`config.yaml` + `teacher.py`（本轮已完成），`test_teacher.py` 19/19。
3. 训练侧：改 `obs.py`（§4.1）→ 改 `synth.py`（§4.2.1）→ 重训 → 重导 blob。
4. 固件侧：`cfc_task.c`（§3.4）+ PC 发送端（按 §3.3 的 16 字节布局）**同时**上线，
   与新 blob 一起。三者中间任何一步单独上线，都会出现「观测含义对不上」。
5. 真机：先验 V1/V6（延迟），再验「挥一次手 → 一次点头」。

## 6. 已知脆弱点

1. **`wave` 没有真人数据验证**（HaGRID 是静态图集，wave 是 IK 合成）。0.97 的 F1
   只证明「能认出这种运动分布」。真机第一次跑，先看 `debug["gesture_probs"]`
   而不是看动作。
2. **`pointing` F1 0.574**，所以默认不在 `ack_gesture_ids` 里；别随手加进集合。
3. **手部裁剪依赖「举得更高的那只手腕」**，双手都抬起时裁剪内容可能在两手之间跳。
4. **开放集误报 1.12 次/分钟**（7 类没进训练的手势会被认成有手势）；静止场景是
   0 次/8 分钟。如果现场有人常比 rock/three 之类，会误触发 ACK。
5. **`gesture_conf` 同名换义**：这是本轮最容易被老代码静默读到的地方。外部若有
   脚本按旧定义用 `gesture_conf`（`wave_score` 饱和），必须改成读 `wave_gesture_conf`。
6. **`ACK` 窗口内的第二个 onset 被吸收**（§2），不是 bug，但和「两次挥手两次点头」
   的直觉不一致；要改就得把 ACK 从「固定脉冲」改成「可延长」，会牵动训练标签。
