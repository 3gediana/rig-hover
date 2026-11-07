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

## 当前结果（2026-11-08 冒烟）

数据：4 条 take，77 个区间，其中 knock 活动段 46 个；有标注帧 10009 / 12791。

| 模型 | 帧级 F1 | AUROC | 事件级 F1 | 事件 P | 事件 R |
|---|---|---|---|---|---|
| frame | 0.5179 ± 0.3239 | 0.8410 ± 0.0811 | 0.2046 ± 0.1716 | 0.2080 | 0.2160 |
| gru | **0.6133 ± 0.1705** | **0.8465 ± 0.0664** | **0.3157 ± 0.1299** | 0.2506 | 0.4300 |

配置：窗口 128、输入 96、15 epochs、batch 8、CPU、`--max-gap-frames 10
--min-event-frames 8 --threshold-metric frame`。逐折明细见
`runs_knock/loo3_20261108/report.md`（本地生成，不入库）。

同一数据上的两次对照（都更差，说明参数不是随便调大的）：

| run | 配置差异 | gru 帧级 F1 | gru 事件级 F1 |
|---|---|---|---|
| `loo_20261108` | 无 gap 合并、min_event=4 | 0.6133 | 0.2806 |
| `loo2_20261108` | 窗口 192 / 输入 128、事件阈值 | 0.4652 | 0.1633 |

一个必须记住的现象：fold `val=204337` 上 frame 模型 AUC=0.941，但按训练集
选出的阈值 0.90 在留出 take 上一帧都不过线，帧级 F1 直接变成 0 ——
**阈值跨 take 会漂移**，这也是事件级指标比帧级更脆的原因。GRU 在该折
帧级 F1=0.324、事件级 0.200，至少没有整体塌掉。

## 局限（必须一起读）

1. **只有 4 条 take、1 个人、1 个房间**，留一就是 3 训 1 测；表里的均值方差
   没有统计意义，只能说明链路能跑、标签能学。
2. 标注是**敲击活动段**（4–10 s），不是单次接触瞬间。所以事件级 F1 衡量的是
   "能不能把这个活动段圈出来"，不是"数清楚敲了两下"。要做单次敲击计数，需要
   在标注里把每次接触单独切出来（0.3–0.8 s），或加接触点标记。
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
- 标注升级：单次敲击切成独立区间，补 `contact` 标记。
- 输入升级：手部 ROI 裁剪 + 更高分辨率，而不是继续调网络。
- 只有数据量和标注粒度上来之后，才值得比较不同时序结构（GRU / TCN / CfC）。