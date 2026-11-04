# Pixel-to-Event Network：实验报告

> 由 `scripts/report_pixel.py` 从实测文件自动生成，正文不含手写数字。
> 数据来源：`reports/pixel_results.json`、`runs_pixel/*/metrics.json`（3 个完成的 run）。
> 设计文档：`DESIGN-PIXEL-EVENT-NETWORK.md`。

## 0. 结论速览（全部来自实测）

| 项 | 值 |
|---|---|
| 完成的 pixel 端到端训练 run | 3 |
| 对比模型数 / 每模型最大种子数 | 3 / 1 |
| 本模型参数量 | 3,266,388 |
| 本模型 MACs/帧（自测，B=1） | 166.7 M（预算 1500 M） |
| 本模型 mAP（测试集，1 seeds） | 0.0452 |
| 本模型 phase macro-F1（单阈值，测试集） | 0.0673 |
| 本模型 onset F1 | 0.0952 |
| 本模型 onset 平均延迟 | — 帧 |
| 本模型 count MAE（逐类） | 0.0785 |
| **无图像下限**（全正预测）phase macro-F1 | 0.0847（全负 0.0000） |
| 无图像下限 onset F1（全正） | 0.0904 |
| 16x16 线性探测 mAP 上限 | 0.0558 |
| 无图像下限的 mAP | **0.0**（常量预测没有排序能力） |
| 本模型 ECE / Brier | 0.1206 / 0.0600 |
| 相对最弱基线（frame_cnn）的配对 Δ | -0.0120 [-0.0157, -0.0087] |
| 相对最强基线（gru）的配对 Δ | -0.0118 [-0.0155, -0.0086] |

## 1. 与设计文档的对照

| spec 条目 | 实现位置 | 状态 |
|---|---|---|
| §3.1 per-frame order (state→scout→focus→crops→bind→relations→event→state) | `PixelEventNet.step` | 实现，顺序与 spec 一致 |
| §4.1 global scout, stride-4 map + global token | `pixel_event.Scout` | 实现 |
| §4.2 focus controller, K boxes, differentiable sampling | `pixel_event.VisualEye` | 实现（K=4，64px crop，affine_grid+grid_sample） |
| §4.2 shared foveal encoder + motion_delta | `pixel_event.FovealEncoder` | 实现 |
| §5.1 persistent entity slots, dustbin, free-slot init | `pixel_event.EntityBinder` | 实现（N=4） |
| §5.2 directed relation tokens + scene anchor, 2 rounds | `pixel_event.RelationGraph` | 实现（N+1 节点，2 轮） |
| §5.3 semi-Markov phase processes with hazard 1-exp(-λΔ) | `pixel_event.EventEngine` | 实现（计数只在完成转移上累加） |
| §5.4 interaction readiness head | `PixelEventNet.ready` | 头已实现，**无标签数据→未训练** |
| §6 multi-task loss | `scripts/train_pixel_event.py` | 实现 event/onset/count/presence/gesture；relation/ready 无标签 |
| §6 no frozen CLIP anywhere | 全流程 | 满足：输入就是 160×160 灰度像素 |
| §8 ≤5M 参数 / ≤1.5 GMAC | `pixel_event.count_params/count_macs` | 见下（实测远低于预算） |
| §9 任务 1/2（敲击、挥手计数） | — | **未完成**：无逐次边界标注的数据 |
| §9 任务 3（离开/返回） | `RobotSessions` + `eval_robot` | 部分：presence 4 态，10 段 20 s 录像 |
| §9 任务 4（交互就绪） | — | **未完成**：spec 自己也指出需要机器人域标注 |
| §9 任务 5（联合压力测试） | `eval_pixel_event.stress_course` | 实现：掉帧 25/50%、突发丢帧、不规则 Δt、模糊、遮挡、变暗 |

## 2. 数据（全部是像素，没有冻结编码器）

| 数据集 | 内容 | 规模 |
|---|---|---|
| `charades_pixel` | Charades 480p 真实视频，解码成 160×160 灰度，96 帧/视频，逐帧 25 类区间归属 + onset | train 490 / val 105 / test 105 视频 |
| `robot_sessions` | RIG-Hover 相机实录 240×240 灰度，约 10 fps，每段 20 s，10 段脚本行为 | 10 段，约 2000 帧 |

Charades 逐帧 event 单元正样本率 4.86%，onset 3424 个（train/val/test = 2394/553/477）。
**计数监督很薄**：700 个视频里“同一视频同类事件 ≥2 段”只有 39 对，所以 count 指标只能当近似信号，不能当重复计数结论。

## 2b. 无模型参考下界（本报告最重要的一张表）

任何架构数字都必须对着这三行读。三者都不含任何训练好的时序模型。

| 参考 | 指标 | 测试集值 |
|---|---|---|
| 全负预测 | phase macro-F1 | 0.0000 |
| 全正预测 | phase macro-F1 | 0.0847 |
| 全负预测 | onset F1 | 0.0000 |
| 全正预测 | onset F1 | 0.0904 |
| **16x16 池化像素上的线性探测** | **mAP** | **0.0558** |
| 事件正样本率 / onset 正样本率 | — | 4.42% / 4.73% |

线性探测是**没有任何时序模型、没有任何训练好的前端**时能拿到的上限；它给出的 mAP 说明这个代理任务的逐帧类目子任务本身信号极弱，所有架构都会聚到它附近。边界检测（onset F1）才是该任务上有区分度的部分。

## 3. 协议

- 因果契约：单步 `step(img_t, dt_t, mask_t, state)`，`dt` 是真实秒。训练与推理共用同一个 step，窗口前向只是把它串起来。
- **严格因果性由测试强制**：扰动 t≥k 的帧与 dt，t<k 的所有输出必须逐位不变（train 与 eval 两种模式都测，实测 0.0）。
- 阈值只在验证集上选，测试集不参与任何选择。
- 模型间比较用**按视频配对 bootstrap**（4000 次重采样），同一条视频被所有模型打分。
- 所有模型共用同一个 Scout backbone、同一个 head 类、同一份数据与增广。
- 单 GPU 串行：本机并发 torch 进程会假崩溃（WinError 1455），因此矩阵是一个一个跑的。

## 4. 主结果（测试集）

| model | n_seeds | mAP | phase macro-F1 | onset F1 | onset lat | count exact | count MAE | ECE | Brier |
|---|---|---|---|---|---|---|---|---|---|
| **pixel_event** | 1 | 0.0452 ± 0.0000 | 0.0673 ± 0.0000 | 0.0952 ± 0.0000 | — | 0.9234 ± 0.0000 | 0.0785 ± 0.0000 | 0.1206 ± 0.0000 | 0.0600 ± 0.0000 |
| **frame_cnn** | 1 | 0.0458 ± 0.0000 | 0.0845 ± 0.0000 | 0.0952 ± 0.0000 | — | 0.9288 ± 0.0000 | 0.0731 ± 0.0000 | 0.1655 ± 0.0000 | 0.0712 ± 0.0000 |
| **gru** | 1 | 0.0476 ± 0.0000 | 0.0847 ± 0.0000 | 0.0952 ± 0.0000 | — | 0.9288 ± 0.0000 | 0.0731 ± 0.0000 | 0.1645 ± 0.0000 | 0.0712 ± 0.0000 |

## 5. 与基线逐视频配对 bootstrap（相对本模型）

| model | Δ phase macro-F1 vs pixel_event | 95% CI | p | n videos |
|---|---|---|---|---|
| frame_cnn | -0.0120 | [-0.0157, -0.0087] | 0.0e+00 | 105 | ❌
| gru | -0.0118 | [-0.0155, -0.0086] | 0.0e+00 | 105 | ❌

## 5b. 消融：semi-Markov hazard 头 vs 逐帧直接解码器（spec §5.3 的待定项）

| 事件头 | mAP | phase macro-F1 | count MAE |
|---|---|---|---|
| pixel_event | 0.0452 | 0.0673 | 0.0785 |

> 消融尚未跑完，表格只有其中一侧。

## 6. 压力测试（spec §9.5）

每个条件都是整段测试视频的 phase macro-F1，模型用同一份权重、不清状态。

| condition | pixel_event | frame_cnn | gru |
|---|---|---|---|
| burst|none | 0.0659 | 0.0831 | 0.0834 |
| burst|occlude | 0.0657 | 0.0832 | 0.0830 |
| clean|blur | 0.0673 | 0.0845 | 0.0847 |
| clean|dark | 0.0673 | 0.0845 | 0.0847 |
| clean|none | 0.0673 | 0.0845 | 0.0847 |
| clean|occlude | 0.0673 | 0.0845 | 0.0847 |
| drop25|none | 0.0672 | 0.0837 | 0.0839 |
| drop25|occlude | 0.0674 | 0.0854 | 0.0844 |
| drop50|none | 0.0670 | 0.0846 | 0.0845 |
| irregular|none | 0.0673 | 0.0844 | 0.0847 |

## 7. 算力预算（spec §8）

| 项 | spec 预算 | 实测 |
|---|---|---|
| 参数量 | ≤5 M | 3,266,388 |
| MACs/帧 | ≤1.5 GMAC | 166.7 M |
| 输入 | 224/240 | **160×160 灰度**（降采样缓存，见 §8 偏差） |
| foveal crops | K=4 | K=4 × 64 px |

## 8. 与 spec 的偏差（诚实清单）

1. **分辨率**：缓存到 160×160 而不是 spec 的 224/240。原因：8 GB 卡上要跑 8 个模型 × 多 seed；foveal crop 因此是 160 帧内的 64 px，而不是原始分辨率。
2. **归一化**：Scout 用 GroupNorm 而不是 BatchNorm。原因：scout 为了速度对整窗批量算，BatchNorm 的训练态统计会跨时间混合，实测把 train 模式因果性打穿到 7e-3；GroupNorm 是逐样本的，批量与逐帧数值等价。
3. **任务**：spec 的 1/2 号主任务（敲击计数、挥手周期计数）没有可用数据，改在 Charades 的 25 类真实事件段上做在线事件检测/边界/计数。重复计数（RepCount/Countix）未做。
4. **交互就绪**：头实现了，但没有机器人域“能否打断”的标注，未训练、未评估。
5. **实体/关系标签**：Charades 没有实体框和关系标注，所以 `L_entity`/`L_relation` 没有监督（结构在，损失项未启用）。
6. **边缘部署实测**：没有在目标边缘主机上测延迟/内存/能耗，只有参数量与 MACs。

## 9. 未完成项（明确标注）

- spec §9 任务 1、2（敲击/挥手计数）：缺逐次边界标注的数据，未做。
- spec §9 任务 4（交互就绪）：缺机器人域标签，未做。
- 重复计数公开数据集（RepCount/Countix/UCFRep）：未下载。
- 边缘主机端到端延迟与内存：未测。
- 视觉编码器与事件引擎的联合训练多轮调优：只跑了固定预算的一轮。

## 10. 复现

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'
Set-Location 'D:\robot_work\stream_memory_research'

# 1) 建立像素缓存（一次性，约 6.4 min，1.72 GB）
& $py scripts\build_charades_pixel_cache.py full

# 2) 单模型训练
& $py scripts\train_pixel_event.py --model pixel_event --data charades_pixel \
      --window 24 --batch 16 --epochs 10

# 3) 串行跑整个矩阵（自动跳过已完成）
& $py scripts\run_pixel_suite.py --models pixel_event frame_cnn gru tcn frozen_dsm \
      --seeds 0 1 2 --window 24 --batch 16 --epochs 10

# 4) 汇总 + 配对 bootstrap + 出图 + 生成本报告
& $py scripts\aggregate_pixel.py
& $py scripts\report_pixel.py
```

单模型性能与因果性自检：
```powershell
& $py -m src.models.pixel_baselines --B 2 --T 10 --device cpu   # 7 个基线 + 本模型对照
```

## 11. 数字溯源

| 内容 | 位置 |
|---|---|
| 每 run 训练曲线 / 测试指标 | `runs_pixel/<run_id>/metrics.json` |
| 每 run 评测（含压力条件、逐视频分数） | `runs_pixel/<run_id>/eval.json` |
| 逐视频预测 | `runs_pixel/<run_id>/preds_test.npz` |
| 汇总表 / bootstrap / 压力表 | `reports/pixel_results.json` |
| 图 | `reports/figures/fig_pixel_main.png` |
| 像素缓存与索引 | `G:\stream_memory_data\pixel_cache\charades_v1\`、`data_manifests/charades_pixel_index.json` |

