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

## 当前结果（2026-11-08 诚实留一）

数据：4 条 take，77 个区间，10009 / 12791 帧有标注。**评测口径：阈值和平滑窗口
只在训练 take 上选，绝不在留出 take 上扫。** 之前一版把阈值扫在留出 take 上，
数字虚高，已作废（见“评测口径事故”）。

### 帧级 / 事件级（阈值来自训练 take）

| 模型 | 帧级 F1（各折 / 均值） | AUROC（各折 / 均值） | 事件级 F1（均值） |
|---|---|---|---|
| gru | 0.703 / 0.766 / 0.844 / 0.742（0.764） | 0.905 / 0.846 / 0.814 / 0.826 | 0.419 |
| gru + tcn + bigru 集成 | 0.782 / 0.802 / 0.847 / 0.779（**0.802**） | 0.964 / 0.916 / 0.800 / 0.886（**0.892**） | 0.354 |

集成做法的前提：三折模型训练时都**没有见过**对应留出 take（gru 来自
`loo8`，tcn/bigru 来自 `loo9`），阈值取各 checkpoint 自己在训练 take 上选出的
阈值均值。评估脚本：`pc/train/ensemble_honest.py`。

配置：窗口 96、输入 96、2 通道（灰度 + 相邻帧差）、ROI 运动质心特写增强
0.5–0.75×、focal loss（γ=2, α=0.25）、MixUp 0.2、按 take z-score、20 epochs、
`--threshold-metric frame`。

演进（同一批 4 条 take；括号内为诚实阈值口径）：

| run | 改了什么 | gru 帧级 F1 |
|---|---|---|
| `loo3` | 老增强（只动第一帧）+ 15 epochs | 0.613 |
| `loo4` | 新增强（光照/缩放平移/时间抖动/遮挡）+ MixUp | 0.752 |
| `loo5` | run4 + `--normalize zscore` | 0.761 |
| `loo7` | + ROI 特写 + focal + 窗口/输入 96 | 0.774 / 0.794 / 0.875 / 0.797 |
| `loo8`+`loo9` | gru / tcn / bigru 分头训练再集成 | 集成帧级 0.802、AUROC 0.892 |

### 两敲触发（平滑 + 阈值都只在训练 take 上选）

对每个留出 take：只在训练 take 上选 `(平滑, 阈值)`，再原样套到留出 take。
脚本：`pc/train/honest_calibration.py`。

| 留出 take | 选中参数 | 命中 | P | R | F1 | 误报/分钟 |
|---|---|---|---|---|---|---|
| 204337 | smooth=5, thr=+0.1 | 4/4 | 1.000 | 1.000 | 1.000 | 0.00 |
| 212119 | smooth=3, thr=-0.1 | 16/16 | 0.741 | 1.000 | 0.851 | 9.14 |
| 213314 | smooth=3, thr=0.0 | 7/7 | 0.934 | 1.000 | 0.966 | 1.89 |
| 214950 | smooth=5, thr=+0.2 | 14/19 | 0.806 | 0.737 | 0.770 | 4.89 |

均值 F1 0.897，总命中 41/46。关键在**轻平滑**：20 帧平滑会把相隔 0.3–0.6 s 的
两下糊成一个包，选平滑时必须把 3/5 帧也纳入候选。

但误报 0–9 次/分钟、波动很大：**不能常开触发**。机器人上必须先进入
“等敲击”的交互窗口（盯拳之后几秒）再开检测，预期每个窗口 0.1–0.8 次误报，
并且要加冷却（`--cooldown-s`）。

### 评测口径事故（必须记住）

第一版把峰阈值和 3 帧平滑直接在留出 take 上扫，得到“46/46、P 0.85”的漂亮数字，
又拿全量 4 条 take 训练的 `deploy2` 模型 demo 当效果展示——**两处都是把测试集
当验证集**。已作废：

- 留出 take 只能算最终指标，不能参与任何参数选择（阈值、平滑、模型结构、增强
  都算）。模型结构和增强本身也是看了这几折结果才定的，所以 AUROC 0.892 仍带
  轻微乐观；正式验收需要新的、没参与过任何选择的 take。
- deploy 模型在训练过的 take 上跑只能说明代码通，不能当效果数字。

### 零误报目标（2026-11-08 夜实测）

要求是“一次误报都不能有”。试了最直接的办法：在训练 take 上把阈值抬到
**已标注负样本零误触发**，再原样套到留出 take（`pc/train/strict_zero_fp.py`）：

| 留出 take | 留出命中 | 留出已标注负样本误报 | 未标注区域触发 |
|---|---|---|---|
| 204337 | 4/4 | 0 | 1 |
| 212119 | 4/16 | 3 | 1 |
| 213314 | 3/7 | 1 | 1 |
| 214950 | 0/19 | 0 | 0 |

合计只命中 11/46，仍然有 4 次误报。结论：**单靠视觉分类器给不出“零误报”的
保证**——抬高阈值先杀掉的是召回，误报并没有清零（阈值跨 take 漂移）。
零误报必须靠系统级设计：模型只产生“可疑敲击”事件，机器人只在交互窗口内
（已确认人脸 + 拳头 + 等待敲击）采纳，并且要求二次确认/冷却；动作前再做一次
状态检查。验收必须用“故意不敲、只做干扰动作”的负样本视频做压力测试
（建议 5–10 分钟/段），在它上面 0 误触发才允许上机。

## 单条视频推理

训练完（`--final-model`）会保存 `model_<结构>.pt`。部署模型 `runs_knock/deploy2_20261108/`
是在全部 4 条 take 上训练的（自测数字偏乐观，泛化看上面的留一表），一次加载三个模型做集成：

```powershell
& $py pc\train\train_knock.py --dataset dataset --out runs_knock/deploy2_20261108 `
    --window 96 --size 96 --epochs 20 --models gru,tcn,bigru --device cuda `
    --roi-zoom 0.6 --loss focal --threshold-metric frame --normalize zscore `
    --hard-negative 0 --pair-isolation-s 0 --final-model --skip-folds

& $py pc\train\detect_knocks.py `
    --model runs_knock\deploy2_20261108\model_gru.pt,runs_knock\deploy2_20261108\model_tcn.pt,runs_knock\deploy2_20261108\model_bigru.pt `
    --video dataset\takes\robot_20261107_214950_1f30d1\take.mp4 `
    --smooth-frames 3 --pair-threshold 0.4 --cooldown-s 4 `
    --json runs_knock\deploy2_20261108\detect_demo.json
```

`--smooth-frames 3` 是两敲检测的关键（checkpoint 默认 20，会把第二下糊掉）；
`--pair-threshold` 必须**在没参与训练的验证 take 上标定**，不能看留出测试 take 调；
`--cooldown-s` 让一段连续敲击只出第一个触发。输出：`events`（敲击活动段）、
`peaks_s`（接触候选时刻）、`two_knock_triggers`（成对触发数）、`triggers`（每次触发的两峰时刻）。

`deploy2` 是在全部 4 条 take 上训练的，对它跑任何一条 take 都是**训练集自测**，
只能说明代码通，不能当效果数字。要评估，必须留出没参与训练和调参的 take
（见 `pc/train/ensemble_honest.py` 和 `honest_calibration.py` 的协议）。

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
2. 标注是**敲击活动段**（1–10 s），不是单次接触瞬间。两敲触发现在能给出
   “0.25–1.5 s 内两个接触峰”的事件，但还没有“单次 vs 两次”的人工标注真值，
   无法训练/验证“数清楚一下还是两下”。`cut_knock_candidates.py`
   能自动切出候选接触帧和检查图，但**候选未经人工确认**，目前还没有用于训练
   “一下 vs 两下”。
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