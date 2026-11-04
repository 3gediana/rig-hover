# DESIGN — Stream Memory Research (v1 charter)

**项目根**: `D:\robot_work\stream_memory_research\`  
**大数据**: `G:\stream_memory_data\`（视频 / 特征缓存 / 合成数据）  
**Python**: `D:\smartcar\.venv\Scripts\python.exe`（torch 2.10+cu128, sm_120, RTX 5060 8GB, 只有 ~5.5GB 可用）

本文件是所有并行工作成员的**唯一共享契约**。写代码前先读完 §1–§4。任何改动契约的
需求先发给 Lead，不要各自改接口。

---

## 0. 科学问题与判据（不可妥协）

要回答 10 个问题（见用户任务书 §4）。核心判据：

- ≥3 个真实视频任务、≥3 个随机种子；
- 在 A长期记忆 / B多目标低串扰 / C事件理解 / D不规则采样鲁棒 / E因果在线 中 ≥4 个维度显著超过单 CfC；
- 其中 ≥2 个维度还要超过 GRU/LSTM 与 Mamba/DeltaNet 类；
- 优势 95% bootstrap CI 不跨 0；参数量不能靠 10× 堆砌换来。

**实验诚实性铁律**

1. 视觉编码器冻结（阶段 1）；所有模型用**同一份缓存特征**。
2. 划分按 `source_id`（视频 / session / 人物）分组，绝不按帧随机切。
3. 同一视频的所有扰动变体必须留在同一 split（用 `source_id` 保证）。
4. 训练 / 验证 / 测试三分离；测试集最后只评估一次。
5. 因果性由 `tests` 强制：改动未来帧不影响过去输出（见 `src/common/tests_causality.py`）。
6. 每个 run 记录 seed / config / 参数量 / 数据 manifest hash / git-ish 代码 hash / 指标 JSON。

---

## 1. 数据契约

### 1.1 一个"序列"（sequence）= 一个可因果流式喂入的样本

磁盘布局（每个 sequence 一个 npz）：

```
G:\stream_memory_data\features\<dataset_name>\<seq_id>.npz
    feat      [T, Dg]  float16   全局/整帧特征（必须）
    t         [T]      float32   物理时间戳，单位秒，严格递增
    tok       [T, K, Dt] float16 可选：空间 token（K 个小格，如 4x4→16 或 2x2→4）
    roi       [T, N, Dr] float16 可选：目标/候选 ROI 特征（跟踪类任务）
    roi_valid [T, N]   uint8     可选：该 ROI 在此帧是否有效
    roi_meta  [T, N, 4] float32  可选：ROI 归一化框 (cx, cy, w, h) ∈ [0,1]
    lab_<name>[T] or [T,C]       标签（见 1.3）
```

- `T` 各序列可变（典型 32–128）。
- `t` 必须严格递增；重复时间戳不允许（重复帧用相同内容、递增 t 表示）。
- 特征由**冻结编码器**产生，`dataset_name` 里带编码器标识与版本，不同版本不得混用。

### 1.2 manifest（`data_manifests/<dataset_name>.json`）

```json
{
  "dataset": "ucf101_action",
  "task": "action",
  "feature_dim": 768,
  "tok_dim": 128,
  "encoder": {"name": "openai/clip-vit-base-patch16", "input": 224, "pool": "4x4", "proj": "pca128", "hash": "..."},
  "split_key": "source_id",
  "seqs": [
    {"seq_id": "...", "source_id": "...", "split": "train",
     "T": 96, "path": "G:/stream_memory_data/features/ucf101_action/xxx.npz",
     "labels": ["action"], "meta": {"class": "ApplyEyeMakeup", "subject": "g01"}}
  ]
}
```

`split` ∈ {train, val, test}。`source_id` 是划分单元（视频 id / 说话人 / session）。

### 1.3 任务与标签

| task | 每帧标签 | 头 | 损失 |
|---|---|---|---|
| `action` | `lab_action [T]` int64 类别，-1=ignore | 分类 | CE(ignore_index=-1) |
| `event` | `lab_event [T,C]` float 0/1 多标签 | 分类 | BCE |
| `state` | `lab_state [T,K]` float 回归目标 | 回归 | MSE |
| `track` | `lab_track [T, N, S]`（每 ROI 的目标量）+ `lab_id [T,N]` | 回归/分类 | MSE/CE |
| `synth_mem` | 见 `src/synth/` 生成器（自带标签语义） | 多种 | 见 §1.5 |

**ignore 语义**：训练时忽略，评估时不计入（不是负样本）。

### 1.4 批次张量（`collate_stream` 输出）

```python
batch = {
  "x":        FloatTensor [B,T,Dg],      # feat
  "t":        FloatTensor [B,T],         # 秒
  "dt":       FloatTensor [B,T],         # dt[:,0]=0
  "mask":     BoolTensor  [B,T],         # True=真实观测
  "tok":      FloatTensor [B,T,K,Dt] or None,
  "roi":      FloatTensor [B,T,N,Dr] or None,
  "roi_valid":BoolTensor  [B,T,N] or None,
  "roi_meta": FloatTensor [B,T,N,4] or None,
  "labels":   {name: Long/FloatTensor [B,T,...]},
  "meta":     [{"seq_id":..., "source_id":...}, ...],
}
```

填充位置：`mask=False`，`dt=0`，标签填 ignore 值。

### 1.5 合成受控任务（`src/synth/`，Lead 与 synth-lab 共同定义）

合成任务输出**特征空间序列**（不经视觉编码器），这样记忆容量实验才是干净的、可精确归因的。
每个任务给出 `feat [T,Dg]`（Dg 默认 64）、`t [T]`、`tok [T,N,Dt]`（可选的"目标槽"）、
逐帧 ground-truth，以及**自带评估函数**。

任务族（必须全部实现）：

| id | 内容 | 主要测量 |
|---|---|---|
| `recall_single` | 单目标在 t0 出现、写入一个随机值，随后消失 D 秒，再出现时要求读出原值 | 长期回忆（D 扫描到 8×间隔） |
| `recall_multi` | K 个目标各自独立变化；随机只更新一个 | 无干扰回忆 + 串扰矩阵 |
| `overwrite` | 新目标覆盖旧目标；旧目标再次出现 | 容量/覆盖 |
| `vanish_reappear` | 目标离开画面后重新出现，需保持身份 | 遮挡恢复 |
| `change_detect` | 目标**确实改变**时必须及时更新 | 更新及时性 |
| `capacity_scan` | 目标数 1→16 扫描 | 有效容量 |
| `irregular` | 上述任务 + 不规则 Δt / 30-50-70% 掉帧 / 重复帧 / burst drop | 采样鲁棒性 |
| `event_seq` | 稀疏事件（敲击、挥手）后必须保持状态 | 事件理解 + 长期任务 |

所有扰动必须由 `seed` 决定并可复现。

### 1.6 受控任务的精确定义（可证伪性来自这里，必须逐条实现）

**通用约定**
- 名义采样间隔 `Δ0 = 0.1 s`；观测通道只在目标"可见"时携带信息，隐藏时置 0 并把
  `vis` 标志置 0。标签**每一步都是真实潜状态**（隐藏段也一样）。
- 潜变量每个 episode 重新随机（`~N(0,1)`，归一化），所以不可能靠记忆数据集来作弊。
- 评估时按 source（episode 生成组）切分；测试集是**独立生成的** episode 组。

**测什么**

1. `recall_single` — 单目标长期回忆
   - 结构：`visible → hidden(G 秒) → visible`。
   - 观测 `feat` [T,64] = `[value(32), vis(1), gap_phase(1), noise(30)]`，隐藏段 value=0。
   - 标签 `lab_state` [T,32] = 当前真实 value。
   - **测试扫描**：G ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2} 秒（=1..32 步），只评估隐藏段。
   - 指标：`recall_by_gap` = 隐藏段上的 cosine 相似度 / 归一化 MSE。

2. `recall_multi` — 多目标无干扰回忆 + 串扰
   - K 个对象，每个有独立 value(16 维)；观测 = 每个对象 `[value(16), vis(1)]` 拼接。
   - 标签 = 所有 K 个对象的 value 展平 [T, 16K]。
   - **串扰探针（在预测上测，不是在内部状态上测，保证对基线公平）**：
     让对象 B 在 `[t1,t2]` 隐藏；在 `t1.5` 只更新对象 A。
     取 `Δ_B = |pred_B(t1.9) − pred_B(t1.2)|`，`Δ_A = |pred_A(t1.9) − pred_A(t1.2)|`。
     **正确行为**：`Δ_B ≈ 0` 且 `Δ_A` 大。所有 (被更新 j, 被读取 i) 组合 → 串扰矩阵。
   - 判据：非对角均值 `dsm` ≤ 0.5 × `cfc`，同时对角的 `Δ_A` 不下降（防止"什么都不更新"骗分）。

3. `overwrite` — 新目标覆盖旧目标
   - 同一条目先写 v1，再写 v2（v1 不再出现）；随后 v1 重新出现。
   - 指标：覆盖后立即读 v2 的准确率；v1 重现后是否**正确地**回到 v1。
   - 这条同时检验"该覆盖时覆盖、该恢复时恢复"，避免模型只是顽固不写。

4. `vanish_reappear` — 目标离开画面后恢复身份
   - K 个候选身份（编码在观测里）；目标离开（vis=0）G 秒后重新出现。
   - 标签 `lab_id` [T] = 身份 id；指标 `identity_accuracy` 在重现后 1/3/5 步。

5. `change_detect` — 该保持时保持、该更新时及时更新
   - 目标长时间不变（测**假更新率** = 稳定期预测方差 / 总方差），然后在随机时刻改变一次。
   - 指标：`false_update_rate`、`update_latency_s`（真实变化到预测越过阈值的物理秒数）。

6. `capacity_scan` — 有效容量
   - K ∈ {1,2,4,8,12,16}，全部隐藏 G 秒后一次性预测。
   - 指标：`capacity_curve`；**饱和点 K\* = 仍保持 ≥ 0.9×K=1 精度的最大 K**。

7. `event_seq` — 事件后长期保持（直接对应机器人 v2b 场景）
   - 稀疏事件脉冲（如 knock），事件帧携带目标位置；之后目标消失，必须继续输出目标位置。
   - 同时存在**干扰事件**（类似但不该锁定目标的脉冲）与重复触发。
   - 指标：事件 F1（容差 0.2 s，去重间隔 0.2 s）、onset 延迟、重复触发率、
     事件后 1×/2×/4×/8× 延迟处的状态准确率。

8. `irregular` — 采样语义
   - 上述任务在 (a) lognormal Δt（CV 可配）、(b) 随机掉帧 30/50/70%、
     (c) 重复帧、(d) burst 丢帧、(e) 时间拉伸 2×/4×/8× 下重新生成。
   - **关键**：掉帧后 `dt` 变大但物理时间轴不变；标签按物理时间对齐。

**必须报告的对照**：每个任务都要有"无记忆 MLP"与"单状态 CfC/GRU"的数字，
并且要能说明"这个任务是单状态容量不足导致的失败，而不是超参没调好"
（用 capacity_scan 与 K\* 证明，并给出所有基线在同一预算下的最优结果）。


---

## 2. 模型契约

### 2.1 统一时序核心（`src/models/`）

```python
class TemporalCore(nn.Module):
    name: str
    def forward(self, x, dt, mask, tok=None, tok_valid=None, roi=None, roi_valid=None, roi_meta=None):
        """返回 (out [B,T,H], aux: dict)"""
    @property
    def n_params(self) -> int
    def init_state(self, B, device) -> dict     # 便于逐步推理
```

**硬性要求**

1. **因果**：`out[:, :k]` 只依赖 `x[:, :k]`、`dt[:, :k]`、`mask[:, :k]`。禁止跨时间 `mean/std`、
   禁止 `x[:, k:].any()`、禁止双向。
2. **mask 语义**：`mask=False` 的步不得改变状态（`h_t = where(mask_t, h_new, h_prev)`）。
3. **dt 语义**：`dt[:,0]=0`；物理秒；不得对 dt 做跨 batch 归一化。
4. `aux` 可以是 `{}`；用于辅助损失与可解释性探针（如 `mem_read`, `write_gate`, `slot_usage`）。

### 2.2 基线矩阵（同一 core 接口，同一 head，同一特征）

| id | 说明 | 备注 |
|---|---|---|
| `mlp` | 逐帧 MLP，无状态 | 无记忆下界 |
| `gru` | 标准 GRU | |
| `lstm` | 标准 LSTM | |
| `cfc` | ncps 口径 CfC（`timespans` 用真实 Δt） | **主基线**，见 `pc/cfc_host/cfc_ref.py` 的精确算子 |
| `mamba` | 纯 torch 选择性 SSM（S6 简化版，因果） | 内部 Δ 由输入生成；另做 `mamba+tdt` 变体把真实 Δt 折进去 |
| `deltanet` | 单快权记忆 + delta 写（无慢状态） | 强基线 |
| `dsm` | **我们的：慢状态 + 矩阵快记忆** | |
| `dsm_slot` | **我们的：慢状态 + slot 快记忆** | |

**公平性**：所有模型 `hidden` 维度按参数量匹配（目标：与 `dsm` 相差 <1.25×，以 `n_params` 报告为准）。
head 结构完全一致。特征 & 划分 & 种子完全一致。

### 2.3 我们的模型（DSM：Dual-State Memory）— v2（已吸收 literature red-team 修正）

> 修正依据：`reports/LIT-BRIEF.md`（10 条结论 + 21 条红队意见）。与之冲突的旧版本作废。
> 三个必须遵守的外部事实：
> (a) **残差(delta)写才是容量来源**，衰减只做干扰管理（Gated DeltaNet 自述 "Decay hurts
> memory retention"：8K S-NIAH DeltaNet 98.8 / 有衰减无残差 30.4 / Gated DeltaNet 91.8）；
> (b) **单一衰减率无法同时抑制串扰和保住长期记忆**（exp(−ΛΔ) 跨数量级会下溢清零）→ 必须多时间尺度；
> (c) **h 也必须按 Δ 演化**，否则整个系统没有连续时间极限，双状态就只是输出层。

`M ∈ R^{d_v × d_k}`，`k` 单位化，`Δ = clamp(dt, 0, 10s)`：

```
g_h    = exp(-Γ_h * Δ)                                     # Γ_h = softplus，τ_h 初值 ∈ [0.2, 2] s
h_dec  = g_h * h
h_new  = h_dec + sigmoid(W_g[z,h_dec]) ⊙ (tanh(W_u[z,h_dec]) − h_dec)
h      = where(mask, h_new, h)

M      = M * exp(-Λ[grp] * Δ)                              # G=4 组多时间尺度
                                                           # 速率初值 ≈ [8, 2, 0.5, 0.02] /s
q,k,v  = W_q[z,h], unit(W_k z), W_v[z,h]
r      = read(M, q)                                        # linear: Mq  |  bilinear: U(Mv)
β      = sigmoid(w·[z,h,r,log1p Δ])                        # 写强度来自 surprise，不来自衰减
M      = M − β (M k) kᵀ + β v kᵀ                            # 精确 Householder delta 写
o      = read(M, q)
out    = LN( MLP([h, o, z, pool(tok)]) )
```

**设计约束（都有对应消融，不许默认其有效）**

| 项 | 默认 | 消融/对照 |
|---|---|---|
| 写规则 | Householder delta | `outer`（加性外积，已知更差；用于归因容量） |
| 衰减 | 多尺度 4 组，最慢一组 λ≈0.02 | `no_decay`（Λ=0）；**必须单独报告"衰减到底有没有用"** |
| h 的时间语义 | 也按 Δ 演化 | `h_by_index`（h 只按观测序号更新）→ 验证连续时间极限 |
| 读法 | `linear` (Mq)，容量 O(d_k) | `bilinear`（U(Mv)，容量 ~d²/log），报告必须写清用哪种 |
| 写入门 | 软 β | `hard`（二值掩码 + straight-through，逐位不动检查） |
| β 指数化 | 关 | `beta_exp`：只当**稳定性**开关，有已发表负结果，不得声称提精度 |

**辅助损失（5 项，各自可关，且必须跨 ≥2 步）**
1. retrieval：`k→v` 的 top-1 检索一致性；
2. multi-target isolation：读出矩阵非对角能量惩罚（用 `aux["slot_read"]` / `aux["route"]`）；
3. time-consistency：同一物理区间两种采样密度的读出应一致（`stream_ops.resample_consistency_pairs`）；
4. write-budget：写入强度/次数预算（soft 计数或 skip 门预算，防止每步全量重写）；
5. long-horizon：≥5 步 rollout 损失（单步损失会让写通道拿不到梯度）。

**数值稳定性（不满足则整轮实验作废）**：Δ clip 到 [0,10] s；监控 `‖M‖`、`ΛΔ`、`surprise`、
NaN；`B(h,0)=0`；槽位未被选中的检查必须用 `torch.equal` 断言逐位不变。

**不得主张的东西**：不主张"支持不规则 Δ"是差异点（CfC 原文也按 timestamp）；
不主张系统层半群（含写入门后不成立）；半群只在 Φ（纯时间演化）层声明，并给出
`‖Φ_{Δ1}∘Φ_{Δ2} − Φ_{Δ1+Δ2}‖` 的数值曲线。差异化必须落在
**遮挡后长期记忆 / 多对象低串扰 / 对象级硬路由**三处。


---

## 3. 训练契约（`src/train/`）

- 配置 = 一个 JSON/YAML dict，`configs/<name>.json`；`runs/<run_id>/config.json` 存副本。
- `run_id = f"{dataset}__{model}__{variant}__s{seed}"`。
- 每 run 产出：`checkpoints/{best,last}.pt`、`metrics.json`、`train.log`、
  `curves.json`、`failure_cases.json`、`preds_val.npz`（可选）。
- 支持 `--resume`；`--smoke` 用 2% 数据跑 2 epoch 验证管线。
- 自动混合精度（fp16 autocast）+ 梯度裁剪；OOM 时自动降 batch 并记录。
- 早停：按 val 主指标，patience 可配。

## 4. 评估契约（`src/eval/`）

每次评估输出统一结构：

```json
{"task": "...", "split": "test", "metrics": {"acc": .., "f1": .., "..."},
 "per_prefix": {"0.25": .., "0.5": .., ...}, "per_gap": {...}, "crosstalk_matrix": [[..]],
 "latency_ms_per_step": .., "params": .., "flops": ..}
```

统计：≥3 seeds → 报告 mean ± std，并做**配对 bootstrap**（按 source_id 重采样 1000 次）
得到 95% CI 与 vs-CfC 的差值 CI。

### 4.1 `runs/<run_id>/metrics.json` 固定 schema（契约，不可改）

```json
{
  "run_id": "...", "status": "complete|failed|running",
  "dataset": "...", "task": "...", "model": "...", "variant": "...", "seed": 0,
  "params": 123456, "code_hash": "abc123", "manifest_sha": "...",
  "epochs_run": 22, "wall_sec": 321.0, "peak_vram_gb": 1.2, "samples_per_sec": 1234.0,
  "backoffs": [{"epoch": 3, "batch_from": 64, "batch_to": 32}],
  "main_metric": "acc",
  "curves": {"epoch": [1,2,...], "train_loss": [...], "val_main": [...]},
  "best": {
    "epoch": 17,
    "val":    {"acc": 0.51, "f1_macro": 0.44},
    "test":   {"acc": 0.50, "f1_macro": 0.43},
    "prefix": {"p@0.25": 0.31, "p@0.5": 0.44, "p@1.0": 0.51},
    "robust": {"dense": {"acc": 0.50}, "stride2": {"acc": 0.49},
               "stride4": {"acc": 0.47}, "drop30": {"acc": 0.48},
               "drop50": {"acc": 0.46}, "drop70": {"acc": 0.41},
               "burst": {"acc": 0.45}, "repeat": {"acc": 0.50},
               "jitter": {"acc": 0.49}, "mixed": {"acc": 0.42}},
    "aux_stats": {"write_gate_mean": 0.3, "mem_norm_mean": 1.2},
    "latency_ms_per_step": 0.42, "flops_per_step": 123456,
    "failure_cases": "failure_cases/<run_id>.json",
    "per_seq_test": {"acc": {"<seq_id>": 0.5, "...": 0.3}}
  }
}
```

**`best.per_seq_test` 是必需项**（测试集每条序列的逐条指标）。有了它，才能做
**按序列配对的 bootstrap**（n≈1500，远比按 seed 配对（n=3）可信）。没有它，
报告里所有置信区间都只能按 seed 配对，必须显式注明这个局限。

`sweep` 只依赖 `status`、`best.val.<main_metric>`、`best.test.<main_metric>`、
`best.robust.<level>.<main_metric>`、`params`。

### 4.2 训练吞吐要求

特征已缓存，模型很小。默认 `num_workers=0` + `StreamDataset(preload=True)`，避免
Windows spawn 复制整个数据集。UCF101 单 epoch 应在 20 s 内（batch 64, T=96, hidden 192）。
如果超时，先查数据加载而不是加 GPU。

---

## 5. 目录与写权限（避免并行冲突）

| 路径 | 所有者 | 内容 |
|---|---|---|
| `src/common/` | **Lead** | 契约、IO、指标、扰动、统计、因果测试 |
| `src/models/` | model-zoo | 所有时序核心 |
| `src/data/` + `scripts/dl_*` + `data_manifests/` | data-video | 真实视频下载/解码/特征缓存/划分 |
| `src/synth/` | synth-lab | 受控合成任务 + 评估函数 |
| `src/train/` + `src/eval/` + `configs/` | trainer-eval | 训练循环、评估、消融驱动 |
| `scripts/run_suite.py`, `progress.md`, `hypotheses.md`, `experiment_log.csv`, `DESIGN.md` | **Lead** | |
| `reports/`, `best_model/` | Lead（末期）/报告成员 | |
| `runs/`, `checkpoints/`, `failure_cases/`, `logs/` | 所有（按 run_id 子目录隔离） | |

**任何人不得修改别人 scope 内的文件。** 需要改就发消息给 Lead。
