# 训练流水线：从区间标注到敲击检测

这条流水线把 8780 标注服务产出的**时间区间标注**变成逐帧监督信号，训练一个
"当前帧是否处于敲击活动段"的检测器，并按 take 做留一评估。

> 现状：4 条 take、同一个人、同一个房间。这里的目标是**把链路跑通、指标可复现**，
> 不是给出泛化结论。4 条 take 的留一结果只能当冒烟测试，见文末"局限"。

## 数据流

```
8780 标注服务
  └─ /api/export ──▶ video_annotations.json
        │  pc/tools/import_annotations.py
        ▼
video_annotation_data/annotations/<sha256(video)[:20]>.json   （每条视频一个文件）
        │  pc/tools/export_take.py
        ▼
dataset/takes/<take_id>/{take.mp4, annotation.json, meta.json, take.json}
        │  pc/train/interval_data.py     （解析标签、区间→帧、切窗口）
        ▼
pc/train/train_knock.py                （留一 take 训练 + 帧级/事件级评估）
        ▼
runs_knock/<run>/metrics.json, report.md
```

原始 JPEG 帧和 `frames.jsonl` 不进仓库，也不参与这条流水线；`take.mp4` 是
20 fps 恒定帧率封装，时间轴与标注服务一致，足够做当前阶段的训练。

## 标签合并

服务有 6 个默认类别，用户还会自建中文标签。训练前按下面的规则并到统一类别：

| 原始标签 | 合并到 |
|---|---|
| `knock` / 敲击 | knock |
| `fist_hold` / 静止握拳 | fist_hold |
| `hand_raise` / 抬手/举拳 / 抬手但未敲 | hand_raise |
| `reach` / 伸手但未敲 | reach |
| `other` / 其它动作 / 晃拳 / 检测摆头？ | other |
| `background` / 背景/无动作 | background |
| 连续敲击 | knock |

当前二分类任务只用 **knock 为正类、其它已标注类别为负类**；没有标注的帧是
`ignore`（既不进正类也不进负类），不当负样本。改映射改
`pc/train/interval_data.py` 的 `NAME_TO_CLASS`。

## 任务与评估

- 输入：窗口 128 帧（20 fps ≈ 6.4 s），两通道 —— 灰度帧 + 与上一帧的差分。
  下采样到 96×96（4 条 take 上最稳的配置；192/128 的配置在 `loo2` 里试过，
  同样的 epoch 数下欠拟合更严重）。
- 模型：
  - `frame`：逐帧 CNN 基线（无时序）
  - `gru`：CNN 特征 + 因果 GRU（推荐）
- 训练：BCE + `pos_weight`，训练时做亮度/对比度抖动和水平翻转增强；
  事件级后处理为"概率平滑 + 阈值 + 合并短间隙（`--max-gap-frames`）+
  过滤短段（`--min-event-frames`）"。默认 gap=0 / min_len=4，
  本次结果用的 gap=10 / min_len=8。
- 划分：**按 take 留一**（4 折），绝不按帧随机切。
- 指标：
  - 帧级：F1 / precision / recall / AUROC（只在有标注的帧上统计）
  - 事件级：预测的敲击段与标注段做 IoU≥0.3 的贪心匹配，算 P/R/F1
- 阈值：不用验证集调，只在**训练 take** 上选事件级 F1 最优的阈值，再原样用到
  留出的 take。

## 命令

```powershell
$py = 'G:\ai\python11\python.exe'

# 1) 把服务导出的一份 JSON 拆回 per-video 标注（会先备份）
& $py pc\tools\import_annotations.py --export video_annotations.json --backup

# 2) 导出 take（MP4 + annotation + meta）并刷新索引
& $py pc\tools\export_take.py --since 2026-11-07 --exclude robot_20261105_004630_ef1d19 `
    --exclude robot_20261105_190704_3e80b8 --index

# 3) 训练 + 留一评估（CPU 也能跑；GPU 被别的任务占满时用 --device cpu）
$env:OMP_NUM_THREADS='8'
& $py pc\train\train_knock.py --dataset dataset --out runs_knock/loo_20261108 `
    --epochs 15 --models frame,gru --window 128 --size 96 --batch 8 --device cpu `
    --max-gap-frames 10 --min-event-frames 8 --threshold-metric frame

# 只看数据统计
& $py pc\train\interval_data.py --dataset dataset

# 单测
cd pc\train; & $py -m unittest test_interval_data -v
```

`runs_knock/` 在 `.gitignore` 里，训练产物不入库；要看结果重新跑即可。

## 当前结果（2026-11-08 增强 + 分数标准化）

数据：4 条 take，77 个区间，10009 / 12791 帧有标注。

| 模型 | 帧级 F1 | AUROC | 事件级 F1 | 事件 P | 事件 R | 命中率 | 误报/分钟 |
|---|---|---|---|---|---|---|---|
| gru（主配置） | **0.7612 ± 0.0718** | 0.8162 ± 0.0837 | **0.4269 ± 0.1689** | 0.3462 | 0.6062 | **98.4%** | 2.9 |

配置：窗口 128、输入 96、20 epochs、batch 8、CPU、MixUp 0.2、
`--normalize zscore`、`--max-gap-frames 10 --min-event-frames 8
--threshold-metric frame`。逐折明细见 `runs_knock/loo5_20261108/report.md`
（本地生成，不入库）。

同一数据上的演进（用来判断改动到底有没有用）：

| run | 改了什么 | gru 帧级 F1 | gru 事件级 F1 | 命中率 |
|---|---|---|---|---|
| `loo3_20261108` | 老增强（只动第一帧）+ 15 epochs，无 MixUp | 0.6133 | 0.3157 | — |
| `loo4_20261108` | 新增强（光照/缩放平移/时间抖动/遮挡）+ MixUp 0.2 | 0.7525 | 0.3353 | 95.3% |
| `loo5_20261108` | 在 run4 基础上加 `--normalize zscore` | **0.7612** | **0.4269** | **98.4%** |

- 帧级 F1 的涨幅主要来自新增强；事件级 F1/命中率的涨幅主要来自
  **按 take 做 z-score 标准化**——阈值在训练 take 的 z 空间里选，跨 take 不那么容易塌。
- 命中率 98.4% = 几乎每个标注的敲击活动段都至少被检出一次；但**误报 2.9 次/分钟**
  仍然是硬伤（190 s 的 take 上约 9 次误报）。机器人直接拿它当触发器会频繁误动作，
  所以它现在是"可测的基线 + 可用的调试工具"，不是可上机的检测器。
- 表里的 `peaks` 是阈值以上的局部极大值数量（每条约 100+），目前只能当诊断量，
  不能当"敲了几下"。

### 一次必须记住的失败

fold `val=204337` 上 frame 模型 AUC=0.941，但按训练集选出的阈值 0.90 在留出
take 上一帧都不过线，帧级 F1 直接变成 0 —— **阈值跨 take 会漂移**。
z-score 标准化缓解了这个问题（run5 该折事件 F1=0.667），但没有根除：
fold `214950` 的 AUROC 仍然只有 0.68。

## 单条视频推理

训练完（`--final-model`）会保存 `model_gru.pt`，可以直接对一条新视频跑：

```powershell
& $py pc\train\train_knock.py --dataset dataset --out runs_knock/deploy_20261108 `
    --epochs 20 --models gru --window 128 --size 96 --device cpu `
    --max-gap-frames 10 --min-event-frames 8 --threshold-metric frame `
    --mixup 0.2 --normalize zscore --final-model --skip-folds

& $py pc\train\detect_knocks.py `
    --model runs_knock\deploy_20261108\model_gru.pt `
    --video dataset\takes\robot_20261107_214950_1f30d1\take.mp4 `
    --json runs_knock\deploy_20261108\detect_demo.json
```

输出：`events`（敲击活动段）、`peaks_s`（每次接触的候选时刻）、
`two_knock_triggers`（相隔 0.25–2 s 的成对峰数）。
注意这是**部署模型在训练 take 上的自测**，数字偏乐观；真正的泛化数字看上面的留一表。

## 单次敲击候选（为"一下 vs 两下"准备数据）

现在的标注是活动段，没有"每次接触"的时刻。`pc/train/cut_knock_candidates.py`
用帧差能量的局部峰自动切候选，并为每个敲击区间生成检查图：

```powershell
& $py pc\train\cut_knock_candidates.py --dataset dataset `
    --take robot_20261107_214950_1f30d1 --out runs_knock/candidates
```

产物：`runs_knock/candidates/<take>_<区间号>.png`（每行一个候选，5 帧 = 峰前后
±0.4 s）和 `candidates.json`。已抽查：`214950` 的 28.05–29.86 s 区间切出 2 个
相隔 0.30 s 的候选，画面确实是手落到桌面两次；35.29–37.25 s 区间只切出 1 个。
**这是启发式候选，不是标注真值**，要人工确认后才能拿来训练"单次/两次敲击"。

## 局限（必须一起读）

1. **只有 4 条 take、1 个人、1 个房间**，留一就是 3 训 1 测；表里的均值方差
   没有统计意义，只能说明链路能跑、标签能学。
2. 标注是**敲击活动段**（1–10 s），不是单次接触瞬间。所以事件级 F1 衡量的是
   "能不能把这个活动段圈出来"，不是"数清楚敲了两下"。`cut_knock_candidates.py`
   能自动切出候选接触帧和检查图，但**候选未经人工确认**，目前还没有用于训练
   "一下 vs 两下"；峰值数量也不能当敲击次数。
3. 模型看到的是整帧；手在画面里大时够用，但 1–2 m 部署距离下只有 10–21 px，
   这条流水线的分辨率假设不成立。下一步应先加手部 ROI（YOLO 或现有感知链的
   腕点）再裁图，或提高采集分辨率。
4. 训练/推理帧率不一致（采集 23 fps、视频 20 fps、部署候选 15 fps）会改变
   时序统计；上机前要用目标帧率重新采样训练数据。
5. 阈值在训练 take 上选，跨 take 会漂移（fold 204337 上 frame 阈值直接失效），
   而且没有单独的校准集；真部署要么按目标域重新校准，要么用与阈值无关的
   指标（AUROC）做上限参考。

## 下一步

- 继续采 take（目标 20–30 条起），按 `COLLECTION-AND-ANNOTATION.md` 的协议。
- 用 `cut_knock_candidates.py` 的检查图把候选确认成单次敲击标注（每行确认/剔除），
  再训练"窗口内两下 vs 一下"的分类器——这是机器人真正要判的东西。
- 输入升级：手部 ROI 裁剪 + 更高分辨率，而不是继续调网络。
- 只有数据量和标注粒度上来之后，才值得比较不同时序结构（GRU / TCN / CfC）。