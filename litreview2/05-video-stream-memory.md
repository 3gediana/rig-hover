# 视频流 / 在线视觉中的对象与局部状态记忆 —— 与候选结构的逐条对照

被测候选结构（ASCII）：

    M_i- = exp(-Lambda * Delta_i) ⊙ M_{i-1}     # 矩阵快权重关联记忆，按真实时间间隔衰减；Lambda 逐通道对角
    q_i  = Q(z_i, h_i); k_i = K(z_i, h_i); v_i = V(z_i, h_i)
    M_i  = M_i- + alpha_i * v_i k_i^T            # 内容寻址写入
    r_i  = R(M_i, q_i)                            # 读
    h_i  = H(h_{i-1}, r_i, z_i)                   # 另一路隐状态

- 检索人：wm-latent（文献调研子代理）；检索日期 2026-11-03。只做网络检索，未训练、未改他人文件。
- 工具：tavily_search / tavily_extract 拉 PDF 或 arXiv HTML 正文；元数据用 export.arxiv.org API 与 PMLR/CVF/Nature 页面二次核对。
- 覆盖 14 条（11 条视频域 + 3 条跨专题锚点），每条都标注了公式的**提取等级**：[正文公式] / [正文结构级] / [仅摘要级]。
- 判定口径：先看"候选的某个零件是否已被逐字实现"，再看"组合是否被实现"，最后看"是否有定理"。不为了支持候选结构而挑文献。

---

## 0. 结论先行（负面优先）

**0.1 候选结构的五个零件，逐个都有逐字级先例**

| 候选零件 | 逐字先例 | 原文证据 |
|---|---|---|
| $M_i = M_{i-} + \alpha_i v_i k_i^\top$（加性外积写入 + 学习率） | **Fast Weight Programmers**，ICML 2021 | "additive outer products of self-invented activation patterns (today called keys and values)"；"The FWP also learns to compute **dynamically changing learning rates**" |
| 衰减 $\exp(-\Lambda\Delta)\odot M$ | **Mamba**（ZOH 离散化），2023 | $\bar A=\exp(\Delta A)$，$\bar B=(\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B$（式 4） |
| 双状态（矩阵记忆 + 另一路隐状态） | **SAVi**（predictor / corrector），ICLR 2022；**MeMOT**（memory buffer + DMAT），CVPR 2022 | SAVi 原文："Inspired by **predictor-corrector methods** … a prediction and a correction step" |
| 读 = 内容寻址 $R(M,q)$ | **MeMViT**（KV 记忆 attention），CVPR 2022；**StreamPETR**（object query 队列传播），ICCV 2023 | "queries attend to an extended set of keys and values, which come from both the current time and the past" |
| 写入侧的"变化才写 / 不变不重写" | **MovieChat**（相似帧合并），CVPR 2024；**MA-LMM**（MBC），CVPR 2024 | "we observe significant temporal redundancy … we propose a method to merge adjacent similar frames" |

**0.2 视频域内真正空着的一格：真实墙钟时间 $\Delta_i$**

- 我对本清单 11 篇视频论文的可提取正文做了全文扫描：**没有任何一篇**把"真实经过时间（秒）"作为记忆更新、衰减或写入强度的输入；时间只以帧号 / clip 序号 / "time stamp" 这类离散标记出现，且只用于位置编码或数据对齐。
- 这是候选结构在**视频流**语境下唯一没有被占据的轴。但"缺席"不等于"新颖"：$\Delta$ 进入状态衰减在 SSM 里是标准做法（Mamba 式 4），只是那里的 $\Delta$ 是**学出来的时间尺度**，不是墙钟。

**0.3 因此新颖性只能落在三个非常具体的地方**

1. **把墙钟 $\Delta t$ 接到快权重记忆的衰减与写入强度上**（视频记忆域内确实没有；SG-STM 的 $\Delta$ 语义与 SSM 的 $\Delta$ 语义不同）。
2. **双状态的职责划分**：$M$ 负责长时内容寻址、$h$ 负责短时不可寻址的状态——SAVi/MeMOT 有双状态，但没有"矩阵记忆 + 内容寻址写"这一对。
3. **"变化才写"的显式触发条件用 $\Delta$ 而不是 similarity**：现有实现（MovieChat/MA-LMM/Flash-VStream）用相似度或信息密度触发，没有一个用时间间隔。

**0.4 需要提前接受的三个反对事实**

- 候选的写规则和衰减项都是"已有机制的重新组合"（FWP + Mamba），审稿人会直接要引用。
- 14 篇里**没有一篇给出关于记忆的定理**（容量/稳定性/采样率不变/误差界）——这既是候选结构的机会（理论空白），也说明这个子领域不以定理为评价标准。
- 视频记忆的容量问题已经被"相似度合并"（MovieChat/MA-LMM）和"注意力分数压缩"（StreamMem）占了，候选结构若只加一个 $\Delta$ 衰减，必须解释它相对"按相似度合并"的增量收益。

---

## 1. 六问速答表（记忆存什么 / 写入触发 / 时间语义 / 容量淘汰 / 不重写机制 / 定理）

| # | 文献 | 记忆存什么 | 写入触发 | 时间语义 | 容量与淘汰 | "不变不重写"机制 | 定理 |
|---|---|---|---|---|---|---|---|
| 1 | SAVi (ICLR'22) | $K$ 个 slot 向量 $\mathbb R^{K\times D}$ | 每帧（corrector 全量更新） | 帧号 | 固定 $K$ 个槽 | 无（GRU 全量写） | 无 |
| 2 | MeMViT (CVPR'22) | 各层压缩后的 $K,V$ 张量 | 每个 clip | clip 序号 | 固定迭代窗；学习式池化压缩 | 无（只压缩，不判重） | 无 |
| 3 | MeMOT (CVPR'22) | $X\in\mathbb R^{N\times T\times d}$ + DMAT token | 每帧 | 帧号 | $N_{\max}=300/600$，$T_{\max}=24$，FIFO | 无 | 无 |
| 4 | SlotFormer (ICLR'23) | slot 序列 | 每帧（编码器）+ 自回归 | 帧号 | 固定 $K$ 槽 × 固定长度 | 无 | 无 |
| 5 | StreamPETR (ICCV'23) | object query 队列 | 每帧 push | 帧号 | 队列长度固定，FIFO | 无 | 无 |
| 6 | MovieChat (CVPR'24) | 短时密集 token + 长时稀疏帧 | 短时满 → 合并弹出 | 帧号 | $R_L$ 帧（合并后） | **相似度贪心合并**（式 3） | 无 |
| 7 | MA-LMM (CVPR'24) | memory bank（KV/特征） | 每帧 | 帧号 | 长度恒定（MBC） | **选最相似相邻帧平均** | 无 |
| 8 | Flash-VStream (ICCV'25) | 低容量 context + 高容量 augmentation 记忆 | 每帧（按信息密度） | 帧号 | 双档容量 | **信息密度分布**驱动选择 | 无 |
| 9 | SOLD (ICML'25) | object slots | 每帧 | 帧号 | 固定槽数 | 无 | 无 |
| 10 | SSM-WM (ICCV'25) | Mamba 隐状态（块式扫描） | 每帧 | 帧号（Mamba 的 $\Delta$ 为学习时间尺度） | 固定隐状态维度 | 无 | 无（引用 Mamba 的分辨率不变性） |
| 11 | StreamMem (2025) | 固定预算 KV cache | 每帧，流式压缩 | 帧号 | 固定预算，按注意力分数 | 无（按分数淘汰，非按相似） | 无 |
| 12 | FWP (ICML'21) | 快权重矩阵 $W$ | 每 token（外积） | token 序号 | 秩/容量有讨论 | 部分（delta 变体只修正映射） | 无 |
| 13 | DeltaNet (2024) | 快权重矩阵 $S$ | 每 token（delta 规则） | token 序号 | — | 是（delta 规则 = 修正而非覆盖） | 无 |
| 14 | Mamba (2023) | SSM 状态 $h$ | 每 token | token 序号（$\Delta$ 学习得到） | 固定 $N$ | 无 | 分辨率不变（引用） |

---

## 2. 逐篇核对

### 1. Conditional Object-Centric Learning from Video (SAVi)
- 作者 / 年份 / 会议或期刊 / 链接：Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, Klaus Greff / 2021（arXiv，v2 2022）/ **ICLR 2022**（arXiv 注释明写 "Published at ICLR 2022"）/ https://arxiv.org/pdf/2111.12594
- 记忆状态：$K$ 个 slot 向量 $S_t=[s^1_t,\dots,s^K_t]\in\mathbb R^{K\times D}$（向量集合，不是矩阵记忆）。[正文公式]
- 读操作精确公式：槽归一化交叉注意力（softmax 在槽维度，而非输入维度）
  $$A_t=\mathrm{softmax}_K\!\Big(\tfrac{1}{\sqrt D}k(h_t)\cdot q(S_t)^\top\Big)\in\mathbb R^{N\times K},\qquad U_t=\frac{1}{Z_t}\sum_{n=1}^{N}A_{t,n}\odot v(h_{t,n})\in\mathbb R^{K\times D}$$
- 写操作精确公式：$\hat s^k_t=\mathrm{GRU}(u^k_t,s^k_t)$，可迭代多次（共享参数）；预测器为 $\tilde S_{t+1}=\mathrm{TransformerEnc}(S_t)$，$S_{t+1}=\mathrm{LN}(\mathrm{MLP}(\tilde S_t)+\tilde S_t)$。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**部分**——有状态更新，但记忆不是矩阵、写入是 GRU 式替换而非 $v k^\top$ 累加。
- $\alpha_t$ 归类：**学出的门**（GRU 的写门）。
- 用物理时间还是观测序号：观测序号（帧下标）。无 $\Delta$。
- 若用 $\Delta$：不适用。
- 支持不规则采样/丢帧/重复帧：**不支持**（帧间等距假设，预测器不含间隔）。
- 有无定理：**无**。
- 用于：视频（无监督多物体分割/跟踪；后续被当作世界模型前端）。
- 与候选结构异同：**这是"预测 + 修正"双阶段在视频里的逐字先例**。原文："Inspired by **predictor-corrector methods** for integration of ODEs, SAVi performs two steps for each observed video frame: a prediction and a correction step." 差异： (i) 修正不是加性残差，而是 cross-attn + GRU 重推断； (ii) 没有矩阵记忆，读不构成 $q^\top M$； (iii) 没有 $\Delta$、没有衰减。
- 裁决：**部分覆盖**——覆盖"双状态 + 先验/修正"的语义框，覆盖"读=注意力"；不覆盖矩阵快权重、加性写入、$\Delta$ 衰减。

### 2. MeMViT: Memory-Augmented Multiscale Vision Transformer for Efficient Long-Term Video Recognition
- 作者 / 年份 / 会议或期刊 / 链接：Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer / 2022 / CVPR 2022 / https://arxiv.org/pdf/2201.08383
- 记忆状态：**缓存的 transformer keys/values**（逐层，张量；先存未压缩版，下次迭代再压缩）。[正文公式]
- 读操作精确公式：$\bar Q=P_Q(X)$ 等池化后投影（式 3–4），注意力
  $$Z=\mathrm{Attn}(Q,K,V)=\mathrm{Softmax}\!\big(QK^\top/\sqrt d\big)V$$
  其中 $K,V$ 由"当前 clip + 历史缓存"拼接而成。
- 写操作精确公式：每次迭代缓存当前 clip 的未压缩 $K,V$；下一次迭代经学习式 $f_K,f_V$ 池化为 $\hat K,\hat V$ 写入。[正文结构级]
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**部分**——是"追加 + 压缩"，不是"加性外积累加"。
- $\alpha_t$ 归类：**attention 权重**（读侧）；写侧是被学习的池化压缩算子。
- 用物理时间还是观测序号：观测序号（clip 序号）。无 $\Delta$。
- 若用 $\Delta$：不适用。
- 支持不规则采样/丢帧/重复帧：不支持。
- 有无定理：**无**。
- 用于：视频（AVA / EPIC-Kitchens 动作识别与预测）。
- 与候选结构异同：候选的 $r_i=R(M_i,q_i)$ 与这里的"query 对历史 KV 做内容寻址"**同构**；差异是记忆写成"追加式 KV 缓存"而非 $v k^\top$ 累加，且没有衰减、没有真实时间。它的"学习式压缩"正好回答候选结构必须回答的容量问题。
- 裁决：**部分覆盖**（读=内容寻址，容量=学习压缩）；不覆盖加性写、衰减、$\Delta$。

### 3. MeMOT: Multi-Object Tracking with Memory
- 作者 / 年份 / 会议或期刊 / 链接：Jiarui Cai, Mingze Xu, Wei Li, Yuanjun Xiong, Wei Xia, Zhuowen Tu, Stefano Soatto / 2022 / CVPR 2022 (Oral) / https://arxiv.org/pdf/2203.16761
- 记忆状态：**时空记忆缓冲** $X\in\mathbb R^{N\times T\times d}$（每个被跟踪对象的历史状态序列），外加 **DMAT**（Dynamic Memory Aggregation Token，每个对象一个，跨步迭代更新）——**真·双状态**。[正文公式]
- 读操作精确公式：短时分支 query $=X_{t-1}$（最近状态）对 $X_{t-T_s:t-1}$ 做多头 cross-attn；长时分支 query $=Q^{\text{dmat}}_{t-1}$ 对更长的 $X_{t-T_l:t-1}$ 做 cross-attn；两支输出经 fusion 得到 track embedding $Q^{\text{tck}}_t$ 与更新后的 $Q^{\text{dmat}}_t$。
- 写操作精确公式：每步把当前对象状态写入缓冲；超过 $T_{\max}$ 时"the oldest state graduates from the memory"（FIFO）。[正文结构级]
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**是**（缓冲 FIFO 写入 + DMAT 迭代）。
- $\alpha_t$ 归类：**attention 权重**（跨帧聚合权重；没有显式学习门，也没有衰减系数）。
- 用物理时间还是观测序号：**观测序号**；容量单位是帧数（$T_{\max}=24$）。
- 若用 $\Delta$：不适用（且帧率变化会让 24 帧窗口对应不同真实时长——这是隐含的采样率缺陷）。
- 支持不规则采样/丢帧/重复帧：不支持（丢帧会让窗口实际跨越更长时间而不自知）。
- 有无定理：**无**。
- 用于：视频（MOT，MOTChallenge）。
- 与候选结构异同：双状态 ✓；读=对历史做内容寻址注意力 ✓；差异：记忆是"向量序列"而非矩阵，写是 push 而非外积累加，无衰减、无 $\Delta$。它的 DMAT 与候选的 $h_i$ 角色最接近（可寻址记忆之外的第二路聚合状态）。
- 裁决：**部分覆盖**（双状态 + 内容寻址读）；不覆盖矩阵写、衰减、$\Delta$。

### 4. SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models
- 作者 / 年份 / 会议或期刊 / 链接：Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg / 2022 / ICLR 2023 / https://arxiv.org/html/2210.05861v2 （PDF 端点抓取失败，用 HTML）
- 记忆状态：对象 slot 序列（$K$ 个 slot × $D$ 维），由预训练 object-centric 模型逐帧提取。[正文结构级]
- 读操作精确公式：slot attention 编码（与 SAVi 同族）；动力学模块用 Transformer 在 slot 上做时空推理并自回归预测未来 slot。
- 写操作精确公式：每帧重新编码得到 slot；预测器以已观测 slot 为条件生成未来 slot（训练用重构损失，特征空间 + 图像空间）。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**部分**（自回归 slot 更新）。
- $\alpha_t$ 归类：**attention 权重**。
- 用物理时间还是观测序号：观测序号。
- 支持不规则采样：不支持。
- 有无定理：**无**。
- 用于：视频（视频预测 / VQA / 规划用世界模型）。
- 与候选结构异同：候选的"对象级局部状态"与 slot 同构；但 SlotFormer 的记忆是"每帧整体重算的槽"，没有"不重写"，没有矩阵记忆，没有 $\Delta$。
- 裁决：**未覆盖**（只覆盖"对象级状态序列"这一前提）。

### 5. Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (StreamPETR)
- 作者 / 年份 / 会议或期刊 / 链接：Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, Xiangyu Zhang / 2023 / ICCV 2023 / https://arxiv.org/pdf/2303.11926
- 记忆状态：**历史 object query 队列**（少量向量，不是密集特征图）。[正文结构级]
- 读操作精确公式：propagation transformer 以当前帧 object query 为 query，与队列中的历史 query 做长程时空交互（注意力）。[正文结构级]
- 写操作精确公式：更新后的 object query 生成 3D 框后 **push 进队列**；队列满则丢弃最旧（FIFO）。另有 MLN（motion-aware layer normalization）用不同 time stamp 的运动量做条件归一化。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**部分**（FIFO push）。
- $\alpha_t$ 归类：**attention 权重**。
- 用物理时间还是观测序号：观测序号（帧）；"time stamp" 只用于 MLN 的运动编码，不是 $\Delta$ 进记忆更新。
- 支持不规则采样：不支持。
- 有无定理：**无**。
- 用于：视频 / 自动驾驶在线感知（nuScenes）。
- 与候选结构异同：候选的"对象级局部状态 + 内容寻址读"在这里几乎是逐字的（object-centric query 传播）；差异：写是整帧 push，不是 $v k^\top$；无衰减；MLN 用运动而非时间间隔。
- 裁决：**部分覆盖**（读 + 局部对象化）；不覆盖写规则、衰减、$\Delta$。

### 6. MovieChat: From Dense Token to Sparse Memory for Long Video Understanding
- 作者 / 年份 / 会议或期刊 / 链接：Enxin Song, Wenhao Chai, Guanhong Wang 等（共 13 作者）/ 2023（arXiv）/ CVPR 2024 / https://arxiv.org/html/2307.16449v4
- 记忆状态：**短时记忆**（密集视觉 token，固定长度 FIFO）+ **长时记忆**（稀疏帧特征，$R_L$ 帧）。[正文公式]
- 读操作精确公式：记忆经投影层进入 LLM 上下文（全量拼接，注意力由 LLM 完成）。
- 写操作精确公式：短时记忆满 → 弹出到 consolidation 模块；合并准则为相邻帧 token 的平均余弦相似度
  $$s=\frac{1}{N}\sum_{j=1}^{N}\cos\big(x_i^{j},x_{i+1}^{j}\big)$$
  按 ToMe 方式**贪心合并最相似的相邻帧**（加权平均），迭代直到保留 $R_L$ 帧；算法无参数。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**是**（FIFO + 相似度触发合并）。
- $\alpha_t$ 归类：**attention 权重之外的第三种：相似度**——它是"要不要写"的触发器，不是学出的门。[正文公式]
- 用物理时间还是观测序号：观测序号（帧号）；文中 timestamp 只出现在数据描述里，未进入更新式。
- 支持不规则采样/丢帧/重复帧：**部分**——重复帧/静止画面会被合并（这正是它的优点），但没有时间概念，丢帧不会被感知。
- 有无定理：**无**。
- 用于：视频（>10K 帧长视频问答）。
- 与候选结构异同：**"不变内容不反复重写"的最接近实现**：相邻帧相似就合并成一条稀疏记忆。候选结构的差异是把触发条件换成 $\Delta$（时间）而不是 similarity（内容）。候选若不能证明"时间触发"优于"相似度触发"，这一条就是它的直接对手。
- 裁决：**部分覆盖**（不重写机制）；不覆盖矩阵记忆、加性写、衰减。

### 7. MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding
- 作者 / 年份 / 会议或期刊 / 链接：Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, Ser-Nam Lim / 2024 / CVPR 2024 / https://arxiv.org/pdf/2404.05726
- 记忆状态：**memory bank**（过去帧的 KV/特征，在线追加）。[正文结构级]
- 读操作精确公式：LLM 的注意力直接读 memory bank（附带"从 bank 检索相关记忆"的读出设计）。[正文结构级]
- 写操作精确公式：Memory Bank Compression（MBC）——"selecting and averaging the most similar adjacent frame features"，使记忆长度不随视频长度增长。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**是**（追加 + 相似度压缩）。
- $\alpha_t$ 归类：**相似度**（选择/平均权重）。
- 用物理时间还是观测序号：观测序号。
- 支持不规则采样：不支持（同 MovieChat）。
- 有无定理：**无**。
- 用于：视频（长视频 QA / captioning）。
- 与候选结构异同：与 MovieChat 同一条路线；**它还明确批评了 FIFO**（"features from the earliest time step are removed … results in the loss of earlier historical information"），这正是候选结构"按时间衰减"会遇到的同一个批评：衰减也是丢信息。
- 裁决：**部分覆盖**（压缩式写入）；不覆盖矩阵、加性写、$\Delta$。

### 8. Flash-VStream: Efficient Real-Time Understanding for Long Video Streams
- 作者 / 年份 / 会议或期刊 / 链接：Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Xiaojie Jin（6 人，arXiv API 核对）/ 2025 / ICCV 2025（官方代码库标注）/ https://arxiv.org/html/2506.23825v2
- 记忆状态：**双档记忆**：低容量 **context memory**（聚合长时程信息并建模"信息密度分布"）+ 高容量 **augmentation memory**（按密度分布检索细节）。[正文结构级]
- 读操作精确公式：按 context memory 给出的信息密度分布，从 augmentation memory 检索细节 token（然后用 LLM 回答）。[正文结构级，未取到逐字公式]
- 写操作精确公式：新帧特征按信息密度更新 context memory（高密度保留、低密度丢弃/压缩）。[正文结构级，未取到逐字公式]
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**是**（密度驱动的写入/淘汰）。
- $\alpha_t$ 归类：**surprise 的弱形式**——信息密度（变化量）驱动写入，但不是预测误差。
- 用物理时间还是观测序号：观测序号。
- 支持不规则采样：不支持。
- 有无定理：**无**。
- 用于：视频（流式长视频 VLM，实时响应）。
- 与候选结构异同：候选的"变化才写"在这里以"信息密度"实现，双档容量与候选的"矩阵记忆 + 隐状态"是同一类分工（一个存细节、一个存摘要），但没有任何时间量。
- 裁决：**部分覆盖**（双档记忆 + 变化驱动写入）；不覆盖矩阵写、$\Delta$ 衰减。

### 9. SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels
- 作者 / 年份 / 会议或期刊 / 链接：Malte Mosbach, Jan Niklas Ewertz, Angel Villar-Corrales, Sven Behnke / 2024（arXiv）/ ICML 2025（官方代码库标注）/ https://arxiv.org/html/2410.08822v2
- 记忆状态：object slots（向量，槽数固定）。[仅摘要级]
- 读/写公式：slot attention 编码器从像素得到槽；槽级转移模型在潜空间做一步预测；用于 model-based RL 的想象 rollout。[仅摘要级，未抄公式]
- 是否存在 $M(t{+}1)=F(\cdot)$：**是**（槽状态转移）。
- $\alpha_t$ 归类：attention 权重（slot attention 内）。
- 用物理时间还是观测序号：观测序号。
- 支持不规则采样：不支持。
- 有无定理：**无**。
- 用于：视频→控制（机器人操作的 MBRL）。
- 与候选结构异同：提供"对象级局部状态 + 潜空间动力学"的当代范式；与候选的差别同 SlotFormer。
- 裁决：**未覆盖**（只提供对象级状态的对照背景）。

### 10. Long-Context State-Space Video World Models
- 作者 / 年份 / 会议或期刊 / 链接：Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, Xun Huang / 2025 / ICCV 2025 / https://openaccess.thecvf.com/content/ICCV2025/papers/Po_Long-Context_State-Space_Video_World_Models_ICCV_2025_paper.pdf
- 记忆状态：**Mamba SSM 的递归隐状态**（块式扫描，跨帧长期记忆）。[正文结构级；PDF 文本抽取丢失符号，未取到逐字公式]
- 读/写：标准 SSM 递归 $h_t=\bar A h_{t-1}+\bar B x_t$，$\bar A=\exp(\Delta A)$；$\Delta$ 是**学习得到的、输入相关的**时间尺度（Mamba 的选择机制），**不是墙钟时间**。
- 是否存在 $M(t{+}1)=F(\cdot)$：**是**（线性递归记忆）。
- $\alpha_t$ 归类：**学出的门 / 学习时间尺度**（$\Delta$ 与门控同源）。
- 用物理时间还是观测序号：观测序号（token/帧）；$\Delta$ 非墙钟。
- 支持不规则采样：不支持（帧被当作等距 token）。
- 有无定理：**无**（作者引用 Mamba 的分辨率不变性，未自证）。
- 用于：视频（视频 world model / 长时一致性）。
- 与候选结构异同：**候选的衰减项在这里以"SSM 递归"的形态完整存在**（$\bar A=\exp(\Delta A)$ 逐字同构），差别只有 $\Delta$ 的语义：学习时间尺度 vs 墙钟秒。这是候选"衰减 + Δ"组合最接近的现代实现，只是不在显式记忆矩阵上，而在 SSM 状态上。
- 裁决：**部分覆盖**（衰减 + 学习式 $\Delta$）；不覆盖显式矩阵记忆、内容寻址写。

### 11. StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding
- 作者 / 年份 / 会议或期刊 / 链接：Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, Mengye Ren / 2025 / arXiv:2508.15717 / https://arxiv.org/html/2508.15717v1
- 记忆状态：**固定大小的 KV cache**（视觉 token 的 key/value）。[正文结构级]
- 读操作精确公式：注意力读取（LLM 侧）。[正文结构级]
- 写操作精确公式：流式编码新帧，用 **visual token 与一组 generic query token 的注意力分数**压缩 KV cache，维持固定预算（query-agnostic，不需要事先知道问题）。
- 是否存在 $M(t{+}1)=F(\cdot)$：**是**（每帧压缩 + 定长维护）。
- $\alpha_t$ 归类：**attention 权重**（分数决定保留/淘汰）。
- 用物理时间还是观测序号：观测序号。
- 支持不规则采样：不支持。
- 有无定理：**无**。
- 用于：视频（流式视频 QA）。
- 与候选结构异同：与 MeMViT 同族（KV 记忆 + 压缩），但把"容量约束下的选择"做到了极致；候选结构若把 $\alpha_i$ 当"写入强度"，这里的注意力分数就是同一位置上的另一种权重量。
- 裁决：**部分覆盖**（定长记忆 + 分数选择）；不覆盖加性写、衰减、$\Delta$。

### 12. Linear Transformers Are Secretly Fast Weight Programmers（跨专题锚点）
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021（PMLR 139）/ https://arxiv.org/pdf/2102.11174
- 记忆状态：**快权重矩阵 $W$**（外积和，秩随时间增长）。[正文/摘要]
- 写操作精确公式：把线性注意力的"加性外积"解释为快权重编程指令：$W_t=W_{t-1}+v_tk_t^\top$（原文："additive outer products of self-invented activation patterns (today called keys and values)"）；并提出 delta 规则式指令以"more easily learn to **correct the current mapping from keys to values**"。
- 是否存在 $M(t{+}1)=F(M_t,k_t,v_t,\alpha_t)$：**是**——这是候选写规则的逐字祖先。
- $\alpha_t$ 归类：**学出的门**——原文："The FWP also learns to compute **dynamically changing learning rates**."
- 用物理时间还是观测序号：token 序号。
- 有无定理：**无**（有关于记忆容量的分析性讨论）。
- 与候选结构异同：候选第 3 行 $M_i=M_{i-}+\alpha_i v_ik_i^\top$ 与 FWP 的加性程序指令**逐字同构**，$\alpha_i$ 就是它说的"动态学习率"。候选的新增只有 $\alpha$ 与 $\Delta$ 的耦合。
- 裁决：**完整覆盖写规则**（不含 Δ）。

### 13. Parallelizing Linear Transformers with the Delta Rule over Sequence Length（跨专题锚点）
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim / 2024 / arXiv:2406.06484 / https://arxiv.org/pdf/2406.06484
- 记忆状态：快权重矩阵 $S$（线性注意力状态）。[正文/摘要]
- 写操作精确公式：把线性 Transformer 的**加性**更新替换为 **delta 规则**（"replace the additive update in linear transformers with the delta rule [DeltaNet]"）：即写入前先用 $k_t$ 读出旧值、写入"误差"而非原值，等价于对已有映射做修正。本文未在本报告中逐字抄出矩阵形式（未提取该式），仅引用其定义性表述。 [部分摘要级]
- 是否存在 $M(t{+}1)=F(\dots)$：**是**。
- $\alpha_t$ 归类：**学出的门**（$\beta_t$ 学习率）。
- 用物理时间还是观测序号：token 序号。
- 有无定理：无（提供并行化算法与实验）。
- 与候选结构异同：与 FWP 同源；delta 规则正是"不变内容不反复重写"的数学化——如果 $v$ 已经能被 $k$ 正确读出，写入量趋零（$v_t-\hat v_t\approx0$）。
- 裁决：**完整覆盖**"修正式写入"语义（不含 Δ）。

### 14. Mamba: Linear-Time Sequence Modeling with Selective State Spaces（跨专题锚点）
- 作者 / 年份 / 会议或期刊 / 链接：Albert Gu, Tri Dao / 2023 / arXiv:2312.00752 / https://arxiv.org/pdf/2312.00752
- 记忆状态：SSM 隐状态 $h\in\mathbb R^{N}$（逐通道）。[正文公式]
- 写操作精确公式（ZOH 离散化，式 4）：
  $$\bar A=\exp(\Delta A),\qquad \bar B=(\Delta A)^{-1}\big(\exp(\Delta A)-I\big)\cdot\Delta B,\qquad h_t=\bar A h_{t-1}+\bar B x_t$$
- $\alpha_t$ 归类：**学出的门 / 学习时间尺度**（$\Delta$ 输入相关、由选择机制产生）。
- 用物理时间还是观测序号：**token 序号**；$\Delta$ 是模型内部的时间尺度参数，**不是真实经过时间**。原文："the pair $(f_A,f_B)$ is called a discretization rule … zero-order hold (ZOH)"。
- 有无定理：引用分辨率不变性（resolution invariance）与归一化性质，未自证。
- 与候选结构异同：候选第 1 行 $\exp(-\Lambda\Delta_i)\odot M_{i-1}$ 与 $\bar A=\exp(\Delta A)$ **逐字同构**（$A$ 即 $-\Lambda$）。唯一差别是 $\Delta$ 的语义：Mamba 学它，候选测量它。
- 裁决：**完整覆盖衰减项**（不含墙钟语义）。

---

## 3. 反对结论与反例

### 3.1 反对一：候选的"写规则"不是新结构，是 2021 年就被命名的机制

- FWP（ICML 2021）已经把"加性外积 + 学习率"写成了快权重程序指令，并明确指出它等价于线性化注意力；DeltaNet（2024）进一步把加性写换成 delta 修正。候选第 3 行 $M_i=M_{i-}+\alpha_i v_i k_i^\top$ 与它们的差别仅在于 $\alpha_i$ 多了 $\Delta$ 依赖。
- **直接后果**：候选不能主张"矩阵快权重关联记忆"或"内容寻址写入"的任何一项为新；只能主张"$\alpha$ 与 $\Delta$ 的耦合"以及"读-写-双状态的具体分工"。

### 3.2 反对二：衰减项 $\exp(-\Lambda\Delta)$ 逐字等于 SSM 的 ZOH 离散化

- Mamba 式 4：$\bar A=\exp(\Delta A)$，$h_t=\bar A h_{t-1}+\bar B x_t$。把 $A$ 取成负的对角矩阵、$\Delta$ 取成真实间隔，就是候选的第 1 行。
- **直接后果**：候选的衰减在法律上属于"连续时间 SSM 的离散化"，而不是新的记忆机制。审稿人会问：为什么不用现成的 SSM 层，而要在显式记忆矩阵上重做一遍？候选必须给出 SSM 做不到的东西（例如逐对象的键值寻址、外部可写、可解释的淘汰）。

### 3.3 反对三：视频域的"变化才写"已经被相似度/密度实现，且它们有实验支持

- MovieChat（CVPR 2024）用相邻帧平均余弦相似度触发合并，理由原文："we observe significant temporal redundancy in videos, where activities span multiple frames with minimal visual changes"——**这正是候选"不变内容不反复重写"的原话级动机**，而 2024 年已经用相似度实现了。
- MA-LMM 用"选最相似相邻帧并平均"；Flash-VStream 用"信息密度分布"。三者都**没有**用时间。
- **直接后果**：候选把触发条件从"相似度"换成"$\Delta$"，必须证明在真实掉帧/变速采样下更优；而在固定帧率视频（上述三篇的全部实验设置）里，$\Delta$ 与相似度高度相关，收益可能为零。

### 3.4 反对四：全清单零定理

- 11 篇视频论文 + 3 篇跨专题锚点里，**没有一篇**给出关于记忆的定理（容量界、稳定性、采样率不变性、误差界）。Mamba 只引用他人证明的分辨率不变性；SSM-WM 只引用 Mamba。
- **直接后果**： (i) 候选结构若想靠"$\Delta$ 带来采样率鲁棒性"取胜，必须自己给出命题与证明（例如：固定帧率下 $\exp(-\Lambda\Delta)$ 记忆的等效半衰期不随帧率变化，而按帧数衰减的记忆半衰期随帧率线性漂移），否则与同行一样只是经验说法； (ii) 这同时是候选的机会——理论空白真实存在。

### 3.5 已知失效模式（这批文献共同暴露的）

| 失效模式 | 文献证据 | 对候选结构的含义 |
|---|---|---|
| 容量-保真权衡：FIFO 丢旧信息 vs 合并丢细节 | MA-LMM 明确批评 FIFO；MovieChat 用合并换容量 | 衰减等价于"连续版 FIFO"，同样会丢掉旧内容；需要论证"该丢的正好是旧的" |
| 相似度合并的误合并 | MovieChat 贪心合并相似相邻帧（仅在相似时安全） | 候选用 $\Delta$ 触发写，但"该不该丢"仍需内容判据；$\Delta$ 不能替代相似度 |
| 时间尺度数值问题 | TAWM 论文（另见 01 世界模型报告）把 $\Delta t$ 做 $\log_{10}$ 归一化才稳定 | 候选的 $\Lambda\Delta$ 在 $\Delta$ 跨越几个数量级时会爆；需要同样的归一化或参数化 |
| 记忆塌缩为"最后写入" | FWP 论文指出纯加性外积的记忆容量受限；DeltaNet 因此改用 delta 规则 | 若 $\alpha_i$ 不受约束，$M$ 会被最近若干次写入主导，长时记忆实质失效 |
| 无采样率概念 | 全部 11 篇视频论文按帧号，帧率变化即静默失效 | 这正是候选的卖点，但也意味着没有可比基线——必须自建可变帧率实验 |
| 丢帧/重复帧无处理 | 无一篇显式处理 | 候选的 $\Delta$ 天然可处理；但这属于"补一个缺失",不是新机制 |

---

## 4. 最接近的先例排序

1. **FWP（ICML 2021）**——写规则逐字：$W_t=W_{t-1}+v_tk_t^\top$ + 学习率 $\alpha_t$。
2. **Mamba（2023）**——衰减逐字：$\bar A=\exp(\Delta A)$。
3. **SAVi（ICLR 2022）**——双状态与 predictor/corrector 的**命名与语义**逐字："Inspired by predictor-corrector methods … a prediction and a correction step"。
4. **MovieChat / MA-LMM（CVPR 2024）**——"不变内容不重写"的显式实现（相似帧合并），且动机与候选一致。
5. **MeMOT（CVPR 2022）**——双状态（历史缓冲 + DMAT）+ 内容寻址读，视频域内结构最接近。
6. **MeMViT / StreamPETR / StreamMem**——读侧内容寻址与定长记忆管理。

**有没有与候选结构逐字对应的已有工作？**
- **写规则**（$M\mathrel{+}=\alpha v k^\top$）：有，FWP 2021 / DeltaNet 2024，逐字。
- **衰减**（$\exp(-\Lambda\Delta)$）：有，Mamba 2023，逐字（但 $\Delta$ 非墙钟）。
- **双状态 + 先验/修正**：有，SAVi 2022（命名逐字）、MeMOT 2022。
- **视频域内把真实墙钟 $\Delta$ 接进记忆的衰减/写入**：**未找到**（14 篇全查，0 篇）。这是唯一空白。
- 因此：候选结构 = "FWP 写规则 + Mamba 衰减 + SAVi 双状态"的三件套重组，其中只有"$\Delta$ 是真实时间"这一参数化是新的。**新颖性强度：弱到中**，取决于能否给出"墙钟 $\Delta$ 在变速/丢帧流上带来可测收益"的证明或实验。

---

## 5. 核实与检索记录

- 元数据：全部经 export.arxiv.org API 或 PMLR / CVF / ICCV 官方页面核对；ICLR 2022（SAVi）、CVPR 2022（MeMViT、MeMOT）、ICLR 2023（SlotFormer）、ICCV 2023（StreamPETR）、CVPR 2024（MovieChat、MA-LMM）、ICCV 2025（Flash-VStream、SSM-WM）、ICML 2025（SOLD，据官方代码库标注）均以 arXiv 注释或会议页面为据。
- 公式提取等级逐条标注：[正文公式] 表示从 PDF/HTML 正文抄出的式子；[正文结构级] 表示正文给出了机制但没有可逐字抄写的公式（多为 PDF 文本抽取丢失符号）；[仅摘要级] 表示只拿到摘要。
- 未取到逐字公式的条目（已标注）：MeMViT 写侧压缩算子、MeMOT 的注意力式、SlotFormer/StreamPETR 的传播式、Flash-VStream 的更新式、SOLD 全部公式、SSM-WM 的 SSM 式、DeltaNet 的矩阵形式。
- 全文时间词扫描（用于第 0.2 节断言）：对 11 篇视频论文的可提取正文做 timestamp / time stamp / elapsed / wall-clock / Δt / delta t / fps / frame rate / seconds 计数，结果只有 fps 与 seconds 出现在数据集与速度描述里；"Δt / delta t / elapsed / wall-clock" 全部为 0。
- 检索词（tavily, advanced）：streaming video memory attention bank；persistent video memory long-term；object-centric slot memory video；online video object tracking memory bank；content-addressable video state；memory augmented video transformer；long video memory consolidation。
- 排除说明：STEVE（NeurIPS 2022）未在 arXiv 上找到可抓取的正文入口，故以 SAVi / SlotFormer / SOLD 覆盖 object-centric slot memory 方向，未强行引用。
