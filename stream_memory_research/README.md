# Stream Memory Research — 流式视频记忆网络

> 目标：设计、训练并验证一个真正强大的因果流式视频记忆网络，并在真实视频任务上
> 显著超过单一 CfC 隐状态模型。
>
> 立项：2026-11-03 夜。目录：`D:\robot_work\stream_memory_research\`
> 大数据：`G:\stream_memory_data\`（G: 340 GB 空闲，D: 100 GB 空闲，按用户要求大文件优先 G:）
> 环境：`D:\smartcar\.venv\Scripts\python.exe`，torch 2.10.0+cu128，RTX 5060 Laptop 8 GB。

---

## 0. 一句话

把"慢状态 + 快记忆"组织成一个可训练、可评估、可归因的因果流式网络，并用
**受控合成记忆任务 + 真实视频任务 + 严格统计**回答 10 个科学问题。

## 1. 怎么读这个项目

| 文件 | 内容 |
|---|---|
| `DESIGN.md` | **唯一技术契约**：数据格式、模型接口、训练/评估协议、并行写权限 |
| `progress.md` | 每一轮实验的假设 / 配置 / 结果 / 失败原因 / 下一轮 |
| `hypotheses.md` | 科学假设清单与判定标准（H1–H14） |
| `experiment_log.csv` | 所有 run 的机器可读账本 |
| `reports/` | 最终报告、对比表、消融表、曲线、失败分析 |
| `best_model/` | 最终 checkpoint + 结构配置 + 归一化 + 推理代码联动 |
| `src/common/` | 数据契约、扰动算子、指标、统计 |
| `src/models/` | 8 个时序核心（基线 + 我们的） |
| `src/synth/` | 受控合成记忆任务与探针 |
| `src/data/` | 真实视频下载/解码/冻结 CLIP 特征缓存/划分 |
| `src/train/`, `src/eval/` | 训练与评估引擎 |
| `scripts/` | 数据下载、特征提取、sweep 驱动 |

## 2. 核心设计（不是最终答案，是要被实验检验的假设）

```
(I_i, τ_i)  ──►  冻结视觉编码器 E  ──►  z_i (全局) + tokens (空间/目标)
                                          │
                    ┌─────────────────────┴──────────────────────┐
                    ▼                                            ▼
              慢状态 h（连续时间）                        快记忆 M（矩阵 delta / slot）
     h⁻ = h + (h_prior − h)(1 − e^{−Δt/τ_h})       M⁻ = M · e^{−Δt/τ_M}
     h  = h⁻ + g ⊙ (tanh(W_u[z,h⁻]) − h⁻)         r = M⁻q ;  β = σ(w·[z,h,r,log1p Δt])
                                                    M = M⁻ + β(v−r)kᵀ/(‖k‖²+ε)
                    └─────────────────────┬──────────────────────┘
                                          ▼
                        out_i = LN(MLP([h, o=Mq, z])) ──► 任务头
```

两条关键性质是**结构性**的而不是学出来的：

1. 衰减项 `1 − e^{−Δt/τ}` 满足**半群**：先走 Δt₁ 再走 Δt₂ ≡ 一次走 Δt₁+Δt₂（精确）。
   因此"同一物理过程、不同采样率"在状态转移层面是一致的。
2. delta 写 `M ← M + β(v − r)kᵀ/(‖k‖²+ε)` 是**误差修正**写：只写"读出来的和实际的不一样"
   的那部分，因此写入量有界、不覆盖无关内容。

## 3. 三类数据任务

| 族 | 数据 | 测什么 |
|---|---|---|
| **S 受控合成** | 自生成，精确真值，原生不规则 Δt | 记忆容量、长期回忆、多目标串扰、更新及时性、Δt 是否真起作用 |
| **V 真实视频** | UCF101（在线动作）、Charades（事件检测）、MOT17（多目标身份）、DAVIS（遮挡恢复） | 因果在线前缀准确率、事件 F1/延迟、ID 保持、遮挡后恢复 |
| **R 机器人** | 真机相机帧（`evidence/sessions`）+ 已有合成 v2a/v2b | 域内离线推理接口与适配（第二阶段，非验收依据） |

## 4. 基线（同一编码器 / 同一划分 / 同一种子 / 参数匹配）

`mlp`（无记忆）· `gru` · `lstm` · `cfc`（主基线）· `mamba`（选择性 SSM）·
`deltanet`（单快权）· `dsm`（慢+矩阵快记忆）· `dsm_slot`（慢+slot 快记忆）

## 5. 判定"显著强大"的标准

- ≥3 个真实视频任务 × ≥3 seeds；
- A 长期记忆 / B 多目标低串扰 / C 事件理解 / D 不规则采样鲁棒 / E 因果在线 五维中 **≥4 维** 显著超过单 CfC；
- 其中 **≥2 维** 还超过 GRU/LSTM 与 Mamba/DeltaNet；
- 优势的 95% 配对 bootstrap CI 不跨 0；
- 参数量不靠 10× 堆砌（`n_params` 必须报告且匹配）；
- 密集视频上不明显退化；完成一次独立最终训练 + 一次性测试。

达不到就写"尚未达到强模型标准"，并继续重设计。**不包装。**

## 6. 复现

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'
cd D:\robot_work\stream_memory_research

# 1) 数据（公开视频 → 冻结 CLIP 特征缓存）
& $py scripts\dl_datasets.py --dataset ucf101
& $py scripts\extract_features.py --dataset ucf101
& $py scripts\build_manifests.py --dataset ucf101
& $py scripts\verify_dataset.py --all

# 2) 受控合成任务
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --config configs\synth_v1.json

# 3) 单次训练
& $py -m src.train.run_one --config configs\<run>.json

# 4) 整套对比（会自动写 experiment_log.csv）
& $py scripts\run_suite.py --suite main_v1 --seeds 0,1,2
```

## 7. 诚实性约束

1. 视觉编码器**冻结**，所有模型用同一份缓存特征。
2. 划分按视频 / 序列 / session，绝不按帧随机切；扰动变体继承原 split。
3. 因果性由单元测试强制（扰动未来帧不得改变过去输出）。
4. `dt` 反事实实验（置零 / 打乱 / 跨采样率）用于证明模型真的用了物理时间。
5. 测试集只在最终训练后评估一次。
