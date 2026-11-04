# 06 · 对抗式检索（第二轮）：矩阵快权重 + 真实 Δt

> 被检验的候选结构（用户版，ASCII）：
>
>     M_i- = exp(-Lambda * Delta_i) . M_{i-1}      # 矩阵快权重关联记忆，按真实时间间隔衰减；Lambda 逐通道对角
>     q_i = Q(z_i, h_i);  k_i = K(z_i, h_i);  v_i = V(z_i, h_i)
>     M_i  = M_i- + alpha_i * v_i k_i^T            # 内容寻址写入
>     r_i  = R(M_i, q_i)                            # 读
>     h_i  = H(h_{i-1}, r_i, z_i)                   # 另一路隐状态
>
> 任务：尽最大努力证明"这个结构已经被提出过"，并逐条检验用户的六条直觉。检索/核对日期 2026-11-03；只做网络检索与原文核对，不读也不改上一轮 litreview/ 下任何文件。
> **结论先行：衰减矩阵记忆 + 内容寻址写 + 用 q 读，这三件在 GLA (ICML 2024) 式 (3) 已逐字写出；把 α 换成 exp(-ΛΔt) 正是 ZOH 离散化，Mamba 式 (4) / Schöne 式 (7) 逐字给出；矩阵快权重 + 衰减 + 外积写在 Ba et al. (2016) 已逐字写出。候选结构 = 三篇已发表工作的并集，缺的是"同篇"。**

---

## 0. 结论先行（TL;DR）

1. **GLA 式 (3) 与候选前三行同构**：S_t = Diag(α_t) S(t−1) + k_tᵀ v_t，o_t = S_tᵀ q_t。α_t 是逐通道 sigmoid 门（正是候选 Λ 的位置），写是 rank-1 外积（正是 α v kᵀ），读是 q。**唯一差别：α_t 是学出的门，不是 exp(-ΛΔt)。**
2. **exp(-ΛΔt) 不是新想法，是零阶保持（ZOH）离散化**：Mamba 式 (4) 的 Ā = exp(ΔA)、B̄ = (ΔA)⁻¹(exp(ΔA) − I)ΔB 同时含 e^{ΛΔ} 与 (e^{ΛΔ}−1)/Λ；Schöne et al. 式 (7) 在**事件驱动**场景直接用真实 Δt 写出 Λ̄ = e^{ΛΔ}、B̄ = Λ⁻¹(Λ̄ − 1)B。**这对系数是教科书结论。**
3. **矩阵快权重 + 衰减 + 外积写的原始形态是 Ba et al. 2016**：A(t) = λ A(t−1) + η h(t) h(t)ᵀ，且 A 与 h 并存（双状态）。候选把 λ 换成 exp(-ΛΔt)、把 h 拆成 k/v。
4. **"双状态"不是空白**：Titans (2025) 的 MAC/MAG 就是"长期记忆 M + 核心序列模型"两路；Kimi Linear 是 KDA + MLA 逐层混合。
5. **最伤的一点**：候选的写是**纯 Hebbian 加性**（M = M- + α v kᵀ）。"只改变化部分"的机制恰恰**不是**它，而是 delta rule 的 (I − β k kᵀ) 前乘（DeltaNet 2021 / Gated DeltaNet 2025 / KDA 2025）。**直觉 (c) 对候选结构本身不成立。**
6. **六条直觉裁决**：(a) 有反证；(b) 部分成立（须同时满足半群 + 输入项缩放，否则只是重参数化）；(c) **错**；(d) **错/条件成立**；(e) 部分成立（容量，不是表达力）；(f) 只是直觉。

---

## 1. 检索方法与检索词

主工具 tavily_search（search_depth=advanced）+ tavily_extract 抓 arxiv /html/、官方 proceedings PDF、Nature 正文。每条引用链接均实际拉取；拿不到标 [未核实]、只有摘要标 [仅摘要级]、只有二手转述标 [结构级]。

| 方向 | 实际检索串（节选） | 命中 |
|---|---|---|
| 1 衰减矩阵记忆这一族 | gated linear attention state update S_t = Diag(alpha_t) S_t-1 + k_t v_t transpose per-channel decay GLA；continuous-time gated linear attention time decay matrix memory physical time exponential decay state；leaky integrator state update coefficient 1 - exp(-Delta) discretization exponential integrator memory write | **GLA 式 (3) 逐字**；KDA 式 (1) 逐字；Mamba 式 (4) ZOH 系数逐字；Schöne 式 (7) 真实 Δt 逐字 |
| 2 attention 从读到写 | Kimi Delta Attention KDA per-channel diagonal decay delta rule Kimi Linear 2025；Neural Turing Machines 内容寻址；另有上一轮已核的 DeltaNet/TTT/Titans | 时间线：Schmidhuber 1992 → Ba 2016 → Schlag 2021 → TTT 2024 → Titans 2025 → KDA 2025 |
| 3 真实 Δt 控制衰减 | timestamped memory update physical time decay continuous-time forgetting recurrent memory neural network；event-by-event processing neuromorphic sensory signals deep state-space models；closed-form continuous-time neural networks CfC | **Schöne 式 (7)**（事件 Δt）；Nature Comm 2025（同式）；CfC（Δt 显式闭式，状态为向量）；GRU-D；CRU |
| 4 六条直觉的证伪材料 | fast weight memory capacity limitation overcapacity crosstalk linear attention Schlag 2021；Illusion of State in State-Space Models；KDA 逐通道门动机 | 容量限制、TC⁰ 上界、稀疏更新并行性反证均取到 |
| 5 三者同篇 | 上述串交叉 + "time-aware gated delta rule"、"timestamped key-value memory attention" | **未找到同篇**（见下） |

**明确"搜过但没找到"的清单：**
- **没有**论文同时具备"矩阵关联记忆 M + 真实 Δt 控制衰减 + 内容寻址读"。最接近的是 Schöne / Nature Comm（有真实 Δt，无关联记忆，只有输入投影 B）与 KDA / GLA（有关联记忆，但 Δ 是学出的门）。
- **没有**论文把"双状态（矩阵记忆 + 向量隐状态）"作为正式结构分析；工程上到处都是（Titans MAC/MAG、KDA+MLA）。
- 方向 1 有一条**反向检索结果**：检索 "1 - exp(-Delta) coefficient" 命中一份内部研究稿，把 KDA 的 Euler 系数换成精确指数积分系数，**实验结论是否定的**（长度外推变差）。来源非同行评审，仅作弱证据（详见 §7.3）。

---

## 2. 逐篇条目

### 1. Gated Linear Attention Transformers with Hardware-Efficient Training（GLA）★ 最致命
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim（完整列表 [部分核实]，以 poster 首页与 ICML 页为准）/ 2024 / ICML 2024 / arXiv:2312.06635 / <https://arxiv.org/html/2312.06635>
- 记忆状态：**KV 矩阵** S_t ∈ R^(d_k × d_v)（原文自述 "an RNN with 2D (matrix-valued) hidden states"）
- 读操作精确公式：o_t = q_tᵀ S_t（该文式 (3) 只给递推，读式见其线性注意力小节）
- 写操作精确公式（逐字，式 (3)）：S_t = (α_tᵀ 1) ⊙ S(t−1) + k_tᵀ v_t = **Diag(α_t) S(t−1) + k_tᵀ v_t**；α_t 由低秩线性层 + sigmoid 于 x_t 得到（数据依赖、逐通道）
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)：**是**
- α_t 归类：**学出的门**（逐通道 sigmoid）
- 用物理时间还是观测序号：**观测序号**（token 步；α_t 只依赖 x_t）
- 若用 Δt，影响：**不适用**（无 Δt）
- 支持不规则采样/丢帧/重复帧吗：**不支持**——丢帧或重复帧改变衰减次数，语义随采样率漂移（正是候选想修的问题）
- 有无定理：无半群/采样率定理；有 chunkwise 并行与 I/O 复杂度分析
- 用于：仅语言建模
- 与候选结构异同：**衰减项** Diag(α_t) 与 exp(-ΛΔ) 位置完全相同（逐通道对角、左乘状态）；**写入项** k_tᵀ v_t 与 α v kᵀ 只差标量门与转置顺序；**读**相同；**双状态**无；**Δ 落点**：GLA 用学出的 α，候选换成 exp(-ΛΔt)。**候选 = GLA + 把 α 重新参数化为 exp(-ΛΔt)。**
- 裁决：**完整覆盖**（除 Δt 语义本身）——证据：式 (3) 与候选前两行逐项同构。

### 2. Kimi Linear: An Expressive, Efficient Attention Architecture（内含 KDA）★ 第二致命
- 作者 / 年份 / 会议或期刊 / 链接：Kimi Team (Moonshot AI) / 2025 / arXiv:2510.26692 / <https://arxiv.org/html/2510.26692>
- 记忆状态：**KV 矩阵** S_t ∈ R^(d_k × d_v)（逐字："S_t ∈ R^{d_k×d_v} represents the matrix-form memory state"）
- 读操作精确公式（逐字，式 (1) 后半）：o_t = S_tᵀ q_t
- 写操作精确公式（逐字，式 (1)）：**S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S(t−1) + β_t k_t v_tᵀ**
- 是否存在 M <- M + 内容寻址更新：**是**（且比候选更强：带 delta 前乘）
- α_t 归类：**学出的门**（逐通道 Diag(α_t)；原文明确对比 GDN 的 head-wise 门）
- 用物理时间还是观测序号：**观测序号**
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：无半群/采样率定理；有 DPLR 转移矩阵等价性与 chunkwise 正确性推导
- 用于：仅语言建模（另报 RL scaling）
- 与候选结构异同：**衰减项** Diag(α_t)；**写入项** β_t k vᵀ（候选少了 (I − β k kᵀ) 前乘）；**读**同；**双状态**：KDA 层内无，Kimi Linear 整体是 KDA + MLA 逐层混合（= 双状态）；**Δ 落点**：无。**候选是 KDA 的严格简化版。**
- 裁决：**完整覆盖**——证据：式 (1) 含候选三项，且写更精细。

### 3. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- 作者 / 年份 / 会议或期刊 / 链接：Albert Gu, Tri Dao / 2023 arXiv, COLM 2024 / arXiv:2312.00752 / <https://arxiv.org/html/2312.00752>（作者由官方仓库 bibtex 核对）
- 记忆状态：**对角状态向量**（SSM 状态 h_t ∈ R^N；Mamba2 才显式成矩阵）
- 读操作精确公式（逐字，式 (1)）：y(t) = C h(t)；离散后 y_t = C h_t
- 写操作精确公式（逐字，式 (4)）：**Ā = exp(ΔA)，B̄ = (ΔA)⁻¹(exp(ΔA) − I) · ΔB**；h_t = Ā h(t−1) + B̄ x_t
- 是否存在 M <- M + 内容寻址更新：**部分**（输入相加，非关联写；但 Δ 输入依赖 = 选择性）
- α_t 归类：**外部步长 Δ 经 exp 变换**（Δ 是学出的、输入依赖的步长，不是物理秒）
- 用物理时间还是观测序号：**名义步长**
- 若用 Δt，影响：**状态转移（Ā = exp(ΔA)）且写入强度（B̄ 含 (ΔA)⁻¹ 因子与 ΔB）**
- 支持不规则采样/丢帧/重复帧吗：**部分**（Δ 可逐样本变，但训练时由输入学出，不等于真实经过时间）
- 有无定理：有——ZOH 精确离散化（连续系统到离散系统的等价性，教科书）
- 用于：语言建模（后被移植到视觉/传感器）
- 与候选结构异同：**衰减项** exp(ΔA) 与 exp(-ΛΔ) **完全同形**（A 取负对角即 Λ）；**写入项** B̄ x_t，不是关联 v kᵀ；**读** C h_t，不是 q 检索；**双状态**无；**Δ 落点**：状态转移**和**写入强度。**候选的衰减系数就是 Mamba 的 Ā；候选缺的正是 Mamba 的输入项缩放。**
- 裁决：**部分覆盖**——覆盖"exp(-ΛΔ) 衰减"这一半（且证明它是 ZOH），不覆盖关联写/读。

### 4. Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models ★ 真实 Δt 直接先例
- 作者 / 年份 / 会议或期刊 / 链接：Mark Schöne 等 / 2024 arXiv (v3), ICONS 2025 / arXiv:2404.18508 / <https://arxiv.org/html/2404.18508v3>（标题、作者由 arXiv 页与官方仓库 efficiency 仓库核对）
- 记忆状态：**对角状态向量**（SSM 状态 x_k）
- 读操作精确公式（逐字，式 (7) 同段）：y_k = C x_k（C̄ = C）
- 写操作精确公式（逐字，式 (7)）：**Λ̄ = e^{ΛΔ}，B̄ = Λ⁻¹(Λ̄ − 1) B，C̄ = C，D̄ = D**，Δ 是**两事件之间的真实时间间隔**，x_k = x(t_k)、u_k = u(t_k)
- 是否存在 M(t+1)=F(M_t,...)：**是**（F 由 (Λ̄, B̄) 给出，Λ̄ 是 Δ 的函数）
- α_t 归类：**外部真实 Δt**（直接进衰减系数）
- 用物理时间还是观测序号：**真实物理时间 Δt**
- 若用 Δt，影响：**衰减（e^{ΛΔ}）与写入强度（Λ⁻¹(Λ̄−1) = (e^{ΛΔ}−1)/Λ）**
- 支持不规则采样/丢帧/重复帧吗：**支持**（目标场景即事件流）
- 有无定理：ZOH 精确离散化（教科书；ZOH 自动满足半群）
- 用于：传感器（神经形态事件相机）
- 与候选结构异同：**衰减项** exp(ΛΔ) 与 exp(-ΛΔ) 同形（Λ 取负）；**写入项** 是 B̄ u 而非关联 v kᵀ；**关键**：它给出候选声称"新"的那对系数 **(e^{ΛΔ}, (e^{ΛΔ}−1)/Λ)**；**双状态**无；**Δ 落点**：衰减**和**写入强度。
- 裁决：**部分覆盖（但覆盖的正是候选最想主张的那一半）**——证据：式 (7) 逐字给出 Δ 依赖的衰减与写入系数。

### 5. Compute-in-memory implementation of state space models for event sequence processing
- 作者 / 年份 / 会议或期刊 / 链接：X. Zhang, M. Hu, S. Lu, S. Kim, E. Y. J. Lee, Y. Liu, W. D. Lu / 2025 / Nature Communications / <https://www.nature.com/articles/s41467-025-68227-w>（作者经 Google Scholar 条目核对；卷期页码 [部分核实]）
- 记忆状态：**对角状态向量**（硬件上是 RRAM 状态节点组）
- 读操作精确公式：y(t) = C x(t)（C 为输出矩阵，逐字）
- 写操作精确公式（逐字）：对间隔 Δt 的两个事件，"the state update equation becomes [式 (2)]，where the state transition matrix Λ̄(Δt) and the input projection matrix B̄ are defined as [式 (3)]"，并明确第一项实现 exponential decay（e^{−Δt₁}, e^{−Δt₂}）
- 是否存在 M <- M + 内容寻址更新：是（状态更新，非关联记忆写）
- α_t 归类：**外部真实 Δt**
- 用物理时间还是观测序号：**真实 Δt**（事件间隔）
- 若用 Δt，影响：衰减 + 写入强度（同 Schöne）
- 支持不规则采样/丢帧/重复帧吗：**支持**（事件驱动）
- 有无定理：ZOH 精确离散化；另有"无输入时被动指数衰减"的物理论证
- 用于：传感器（事件/脉冲），硬件实现
- 与候选结构异同：同 §2.4。**额外价值**：它把"Δ 控制衰减"做进了硬件，说明这不是待研究问题，而是已工程化的常识。
- 裁决：**部分覆盖**（Δt→衰减已被硬件实现），不覆盖关联记忆。

### 6. Using Fast Weights to Attend to the Recent Past ★ 逐字先例（矩阵快权重 + 衰减 + 外积写 + 双状态）
- 作者 / 年份 / 会议或期刊 / 链接：Jimmy Ba, Geoffrey E. Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu / 2016 / NeurIPS 2016 / <https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past>；方程逐字 <https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf>
- 记忆状态：**矩阵快权重** A，与向量隐状态 h(t) 并存 → **双状态**
- 读操作精确公式：快权重与当前隐状态相乘 A(t) h(t)，再与普通 RNN 输出结合（该行未逐字取到，标 [结构级]）
- 写操作精确公式（逐字）：**A(t) = λ A(t−1) + η h(t) h(t)ᵀ**
- 是否存在 M <- M + 内容寻址更新：**是**
- α_t 归类：**固定超参衰减 λ + 学习率 η**（非门、非 Δt）
- 用物理时间还是观测序号：**观测序号**
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：无
- 用于：分类与 RL 任务
- 与候选结构异同：**衰减项** λA(t−1) 与 exp(-ΛΔ) 同位；**写入项** η h hᵀ 与 α v kᵀ 同形（把 h 拆成 v、k）；**读**：A 乘查询；**双状态**：**有**；**Δ 落点**：无。**候选 = Ba 2016 + 把 λ 换成 exp(-ΛΔt) + 把 h 拆成 k/v + 把读写成 q。**
- 裁决：**完整覆盖（除 Δt）**——证据：逐字写出"衰减矩阵 + 外积写 + 双状态"。

### 7. Linear Transformers Are Secretly Fast Weight Programmers（DeltaNet 原版）
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139 / <https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf>
- 记忆状态：**KV 矩阵** W（快权重）+ 控制器网络 → **双状态**
- 读操作精确公式（逐字，式 (20)）：v̄(i) = W(i−1) φ(k(i))
- 写操作精确公式（逐字，式 (21)(22)）：β(i) = σ(W_β x(i))；v_new(i) = β(i) v(i) + (1 − β(i)) v̄(i)；写入为外积（等价残差形式 S_t = S(t−1)(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ 在 GatedDeltaNet 文中被逐字引用）
- 是否存在 M <- M + 内容寻址更新：**是（误差修正型）**
- α_t 归类：**学出的门**（β(i)）
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：**有**——"We infer a memory capacity limitation of recent linearised softmax attention variants"，并讨论 overcapacity 区
- 用于：仅语言建模/翻译 + 合成检索
- 与候选结构异同：**衰减项**：无（纯外积累积）；**写入项**：delta rule，比候选的纯加性更强（只写变化量）；**读**同；**双状态**：有；**Δ 落点**：无。**候选的"写"是它的退化版本。**
- 裁决：**部分覆盖**（写/读/双状态覆盖，衰减与 Δt 未覆盖；且它的写更强）。

### 8. Gated Delta Networks: Improving Mamba2 with Delta Rule
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Jan Kautz, Ali Hatamizadeh / 2025 / ICLR 2025 / arXiv:2412.06464 / <https://arxiv.org/html/2412.06464v3>
- 记忆状态：**KV 矩阵** S
- 读操作精确公式：o_t = S_t q_t（Mamba2 段）
- 写操作精确公式（逐字，式 (10)）：**S_t = S(t−1)[α_t (I − β_t k_t k_tᵀ)] + β_t v_t k_tᵀ**（α_t 标量门）
- 是否存在 M <- M + 内容寻址更新：**是**
- α_t 归类：**学出的门**（标量，head-wise）
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：把更新导出为 online learning 问题**闭式解**（引 Liu et al. 2024）；无采样率/半群定理
- 用于：仅语言建模
- 与候选结构异同：衰减项 α_t 标量版（KDA 是逐通道版）；写入 = delta + 加性；读同；双状态无；Δ 无。
- 裁决：**部分覆盖**（同构但不含 Δt；且写更强）。

### 9. Learning to (Learn at Test Time): RNNs with Expressive Hidden States（TTT）
- 作者 / 年份 / 会议或期刊 / 链接：Yu Sun, Xinhao Li, Karan Dalal 等 / 2024 arXiv，ICML 2025 (PMLR 267) / arXiv:2407.04620 / <https://arxiv.org/html/2407.04620v4>
- 记忆状态：**参数或梯度**（隐状态 W 本身是线性模型/MLP 的权重）
- 读操作精确公式（逐字，式 (1)）：z_t = f(x_t; W_t)
- 写操作精确公式（逐字，式 (2)）：**W_t = W(t−1) − η ∇ℓ(W(t−1); x_t)**
- 是否存在 M <- M + 内容寻址更新：**是**（梯度即内容寻址的写方向）
- α_t 归类：**预测误差**（经 ∇ℓ；ℓ 为重构时 ∝ (D(h−) − z)）+ 学习率 η
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：无（给出表达力 vs 长上下文的经验论证）
- 用于：仅语言建模
- 与候选结构异同：**衰减项**：无；**写入项**：梯度而非外积（线性 f + MSE 时 ≈ delta rule）；**读**：f(x;W)；**双状态**：无；**Δ 落点**：无。
- 裁决：**部分覆盖**（内容寻址写矩阵记忆；衰减与 Δt 未覆盖）。

### 10. Titans: Learning to Memorize at Test Time（含 MAC/MAG 双状态）
- 作者 / 年份 / 会议或期刊 / 链接：Ali Behrouz, Peilin Zhong, Vahab Mirrokni / 2025 / NeurIPS 2025 / arXiv:2501.00663 / <https://arxiv.org/html/2501.00663v1>
- 记忆状态：**KV 矩阵** M（神经长期记忆）+ 核心序列模型 → **双状态**
- 读操作精确公式：由 M 对查询做关联检索（论文以 "a forward pass retrieves a response for a query" 表述，[结构级]）
- 写操作精确公式（逐字，式 (13)(14)(16)）：M_t = (1 − α_t) M(t−1) + S_t；S_t = η_t S(t−1) − θ_t ∇ℓ(M(t−1); x_t)；合并 **M_t = (1 − α_t) M(t−1) − θ_t ∇ℓ(M(t−1); x_t)**
- 是否存在 M <- M + 内容寻址更新：**是**
- α_t 归类：**学出的门**（衰减）+ **surprise**（梯度模长决定写入量）
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：无半群/采样率定理；有"衰减机制是现代循环模型遗忘机制的推广"分析
- 用于：仅语言建模
- 与候选结构异同：**衰减项** (1 − α_t) 与 exp(-ΛΔ) 同位；**写入项** 梯度（弱化为线性关联时 = delta rule），候选是纯外积；**读**同；**双状态**：**有**（MAC/MAG 两种融合）；**Δ 落点**：无。
- 裁决：**完整覆盖（除 Δt）**——式 (16) 同时含"衰减矩阵记忆 + 内容寻址写"，并显式给双状态融合。

### 11. Transformers are SSMs: State Space Duality（Mamba-2）
- 作者 / 年份 / 会议或期刊 / 链接：Tri Dao, Albert Gu / 2024 / ICML 2024 / arXiv:2405.21060 / <https://arxiv.org/html/2405.21060>
- 记忆状态：**矩阵**（SSD 框架下与 attention 对偶的矩阵形式）
- 读操作精确公式：y_t = S_tᵀ x_t 型（SSD 矩阵-注意力对偶形式，[结构级]）
- 写操作精确公式（逐字，经 GatedDeltaNet 论文转述）：**S_t = α_t S(t−1) + v_t k_tᵀ**（标量 α_t ∈ (0,1)）
- 是否存在 M <- M + 内容寻址更新：**是**
- α_t 归类：**学出的门**（标量，head-wise）
- 用物理时间还是观测序号：观测序号（Δ 是参数）
- 若用 Δt，影响：衰减（α = exp(-Δ·softplus(a)) 形式，见 Mamba 式 (4) 的对角情形）
- 支持不规则采样/丢帧/重复帧吗：名义上可（Δ 可变），实质不
- 有无定理：SSD 等价性定理（SSM = 半可分矩阵 = 线性注意力）
- 用于：仅语言建模
- 与候选结构异同：最接近的"矩阵记忆 + 标量衰减 + 外积写"；差别在衰减是标量（KDA 才逐通道）、写无 delta。
- 裁决：**部分覆盖**。

### 12. Retentive Network: A Successor to Transformer for Large Language Models（RetNet）
- 作者 / 年份 / 会议或期刊 / 链接：Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei / 2023 / arXiv:2307.08621 / <https://arxiv.org/pdf/2307.08621>
- 记忆状态：**KV 矩阵** S
- 读操作精确公式：o_t = S_t q_t（retention 读）
- 写操作精确公式：**S_t = γ S(t−1) + k_tᵀ v_t**（γ 为固定标量衰减，非输入依赖）
- 是否存在 M <- M + 内容寻址更新：**是**
- α_t 归类：**固定的超参衰减**（非门、非 Δt）
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：三种表示（并行/循环/chunkwise）等价
- 用于：仅语言建模
- 与候选结构异同：候选的 Λ 逐通道 vs RetNet 的 γ 标量；其余同。**历史价值：衰减矩阵记忆在 2023 年已是"标配三件套"之一。**
- 裁决：**部分覆盖**（衰减 + 外积写 + q 读，缺 Δt 与逐通道）。

### 13. Closed-form Continuous-time Neural Networks（CfC）
- 作者 / 年份 / 会议或期刊 / 链接：Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus / 2022 / Nature Machine Intelligence 4:992–1003 / <https://research-explorer.ista.ac.at/record/12147>；正文 <https://arxiv.org/html/2106.13898>
- 记忆状态：**向量**（隐状态 x(t) ∈ R^n，非矩阵）
- 读操作精确公式：由隐状态读出（[结构级]）
- 写操作精确公式：**[结构级]** 论文明确 "we replace the exponential decay term with a reversed sigmoidal nonlinearity σ(.)"，并以 (1 − σ(·)) 作门控平衡在两条极限间插值；时间以显式 Δt 进闭式解，逐字："since time appears explicitly in closed-form, the formulation relaxes the need for complex numerical solvers"
- 是否存在 M(t+1)=F(M_t,...)：是（但 M 是向量）
- α_t 归类：**外部真实 Δt**（进门控）
- 用物理时间还是观测序号：**真实 Δt**
- 若用 Δt，影响：**状态转移/门控（衰减项）**
- 支持不规则采样/丢帧/重复帧吗：**支持**（原文强调 explicit time dependency 可处理不规则采样）
- 有无定理：闭式解与 LTC ODE 语义的逼近性（"provably tight when the integral appears as the exponent of an exponential decay"）
- 用于：传感器/时间序列/控制（机器人部署文献常用）
- 与候选结构异同：**衰减项**：CfC 把 exp 换成 sigmoid 并挂在 Δt 上，候选保留 exp(-ΛΔt)；**写入项**：两条 MLP 的门控混合，无关联记忆；**双状态**：无；**Δ 落点**：门控。
- 裁决：**部分覆盖**（"Δ 控制衰减"在向量状态上早已做过；候选新意只能是"矩阵状态 + 关联写"）。

### 14. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- 作者 / 年份 / 会议或期刊 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / Scientific Reports 8:6085 / <https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216>
- 记忆状态：**向量**
- 读操作精确公式：隐状态输出（[结构级]）
- 写操作精确公式（逐字，式 (10)）：**γ_t = exp{ −max(0, W_γ δ_t + b_γ) }**，δ_t 为真实时间间隔；γ 用于衰减输入与隐状态
- 是否存在 M <- M + 内容寻址更新：否（无关联写）
- α_t 归类：**外部真实 Δt**（δ_t 进入指数衰减）
- 用物理时间还是观测序号：**真实 Δt**
- 若用 Δt，影响：**衰减（状态转移）**
- 支持不规则采样/丢帧/重复帧吗：**支持**
- 有无定理：无
- 用于：传感器/临床时间序列
- 与候选结构异同：候选的 exp(-ΛΔ) 与 γ_t = exp(−W δ + b) 同形（逐通道对角 vs 逐维）。**"Δ 控制衰减"在 2018 年已是标准操作。**
- 裁决：**部分覆盖**（衰减项 + Δt；不覆盖矩阵记忆与内容寻址）。

### 15. Modeling Irregular Time Series with Continuous Recurrent Units（CRU）
- 作者 / 年份 / 会议或期刊 / 链接：Mona Schirmer 等（Bosch Center for AI）/ ICML 2022 / PMLR 162 / <https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf>
- 记忆状态：**向量**（高斯均值 + 协方差）
- 读操作精确公式：解码器 g_φ(μ⁺, Σ⁺)
- 写操作精确公式（逐字，式 (8)(11)）：**μ⁻_t = exp(A(t − τ(t))) μ⁺_τ(t)**；μ⁺_t = μ⁻_t + K_t (y_t − H μ⁻_t)
- 是否存在 M <- M + 内容寻址更新：部分（Kalman 更新，非关联写）
- α_t 归类：**外部真实 Δt**（进 exp(AΔ)）+ 增益由协方差闭式给出
- 用物理时间还是观测序号：**真实 Δt（秒）**
- 若用 Δt，影响：**状态转移 + 增益**
- 支持不规则采样/丢帧/重复帧吗：**支持**
- 有无定理：连续-离散 Kalman 滤波闭式（教科书）
- 用于：传感器/时间序列
- 与候选结构异同：**衰减项** exp(AΔ) 与 exp(-ΛΔ) 同形（A 取负对角即 Λ）；无关联记忆写；无 q 读；无双状态。
- 裁决：**部分覆盖**（Δt→状态转移这一轴；且证明 Δ 也进增益）。

### 16. Neural Turing Machines
- 作者 / 年份 / 会议或期刊 / 链接：Alex Graves, Greg Wayne, Ivo Danihelka / 2014 / arXiv:1410.5401 / <https://arxiv.org/abs/1410.5401>
- 记忆状态：**外部 slots**（N×M 记忆矩阵 + 读/写头权重）
- 读操作精确公式：**[结构级]** 用权重 w_t 对记忆矩阵加权和读出（read/write 方程在 §3.2，本稿未逐字取到）
- 写操作精确公式：**[结构级]** 由擦除向量 e_t 与加向量 a_t 组合写；逐字原文："The key vector, k_t, and key strength, β_t, are used to perform content-based addressing of the memory matrix, M_t"
- 是否存在 M <- M + 内容寻址更新：**是**（内容寻址权重 → 读/写）
- α_t 归类：**attention 权重**（内容寻址权重 w_t + key strength β_t）
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不支持
- 有无定理：无
- 用于：算法学习/序列任务
- 与候选结构异同：**"key 找到记忆并更新"在 2014 年就是 NTM 的核心**；NTM 用显式地址权重，候选把地址隐式化（k 与记忆相乘即寻址 = 线性注意力/快权重）。**候选的"内容寻址" = NTM 的连续松弛版 + 线性注意力。**
- 裁决：**部分覆盖**（内容寻址读写；不覆盖 Δt 与对角衰减）。

### 17. The Illusion of State in State-Space Models
- 作者 / 年份 / 会议或期刊 / 链接：William Merrill, Jackson Petty, Ashish Sabharwal / 2024 / arXiv:2404.08819（ICML 2024）/ <https://arxiv.org/html/2404.08819v1>
- 记忆状态：SSM 状态（覆盖一般线性 SSM 与 S6）
- 读操作精确公式（逐字）：y_i = C_i h_i + x_i
- 写操作精确公式：线性 SSM 递推（分析对象）
- 是否存在 M <- M + 内容寻址更新：不适用
- α_t 归类：不适用
- 用物理时间还是观测序号：观测序号
- 若用 Δt，影响：不适用
- 支持不规则采样/丢帧/重复帧吗：不涉及
- 有无定理：**有，且致命**——"SSMs cannot express computation outside the complexity class TC⁰"（L-uniform TC⁰），因此**无法做置换组合等状态跟踪**；"the 'state' in an SSM is an illusion"（逐字）
- 用于：理论（表达力）
- 与候选结构异同：候选的 M 是矩阵关联记忆，仍属"含矩阵状态的线性/近线性递归"一族；**该定理直接反驳"矩阵记忆比普通隐状态更有表达力（能跟踪状态）"**。
- 裁决：**未覆盖但构成反证**（不覆盖候选结构；但否证"矩阵状态 = 更强状态跟踪"）。

---

## 3. 候选先例表（重合度 + 还差什么）

| 候选先例 | 链接 | 重合度 | 还差什么 |
|---|---|---|---|
| GLA (ICML 2024) 式 (3) | https://arxiv.org/html/2312.06635 | **高** | 只差 α_t 的来源（学出的门 vs exp(-ΛΔt)） |
| KDA / Kimi Linear (2025) 式 (1) | https://arxiv.org/html/2510.26692 | **高** | 只差 Δt 语义；候选的写比它弱（无 delta 前乘） |
| Ba et al. 2016 快权重 | https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf | **高** | 只差 Δt；候选把 h 拆成 k/v（= 线性注意力重写） |
| Titans (NeurIPS 2025) 式 (16) | https://arxiv.org/html/2501.00663v1 | **高** | 只差 Δt；写是梯度而非外积 |
| Mamba (2023) 式 (4) ZOH 系数 | https://arxiv.org/html/2312.00752 | **中** | 无关联记忆；但 (e^{ΛΔ}, (e^{ΛΔ}−1)/Λ) 逐字给出 |
| Schöne et al. (ICONS 2025) 式 (7) | https://arxiv.org/html/2404.18508v3 | **中** | 真实 Δt + 对角衰减 + 写入强度缩放全有；只差关联记忆 |
| Nature Comm 2025 事件 SSM | https://www.nature.com/articles/s41467-025-68227-w | **中** | 同上，且已硬件化 |
| Gated DeltaNet (ICLR 2025) 式 (10) | https://arxiv.org/html/2412.06464v3 | **中–高** | 无 Δt；候选写更弱 |
| TTT (ICML 2025) 式 (2) | https://arxiv.org/html/2407.04620v4 | **中** | 无衰减、无 Δt |
| Mamba-2 / SSD (ICML 2024) | https://arxiv.org/html/2405.21060 | **中** | 标量衰减；无 Δt |
| RetNet (2023) | https://arxiv.org/pdf/2307.08621 | **中** | 固定 γ；无 Δt；无逐通道 |
| CfC (NMI 2022) | https://arxiv.org/html/2106.13898 | **中** | Δt 进衰减（向量状态）；无矩阵记忆 |
| GRU-D (2018) 式 (10) | https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216 | **中** | Δ 进衰减；无矩阵记忆 |
| CRU (ICML 2022) 式 (8) | https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf | **中** | Δ 进状态转移；无关联记忆 |
| NTM (2014) | https://arxiv.org/abs/1410.5401 | **中（内容寻址）** | 无 Δt、无对角衰减 |
| Schlag 2021 | https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf | **中** | 无衰减；写是 delta |
| Illusion of State (2024) | https://arxiv.org/html/2404.08819v1 | 低（反证） | 不是覆盖，是否证（表达力） |

---

## 4. 最致命的三条先例

### 4.1 GLA 式 (3)：候选的前三行几乎逐字
S_t = Diag(α_t) S(t−1) + k_tᵀ v_t，o_t = S_tᵀ q_t。逐字对照：M_i- ← Diag(α_t)S(t−1)；M_i ← M_i- + k_tᵀ v_t；r_i ← q_tᵀ S_t。
**候选唯一新增的是 α_t := exp(-Λ Δ_i)**，这在 GLA 里是可选重参数化，不是结构创新；且 GLA 的 α_t 输入依赖，表达力**严格更强**（exp(-ΛΔ) 只依赖 Δ）。**不加约束时候选只剩"α 不依赖内容、只依赖真实时间"——一个更弱的假设。**要变成贡献，必须证明该限制带来 GLA 没有的好处（采样率不变性/鲁棒性），那是定理工作。

### 4.2 KDA 式 (1)：候选的"更新"比已发表的弱
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S(t−1) + β_t k_t v_tᵀ。它具备候选全部三项（逐通道衰减、矩阵记忆、内容寻址 rank-1 写）+ 候选没有的 delta 前乘。**候选把 (I − β k kᵀ) 去掉，等于退回 GLA 的纯加性写**；而文献演进方向恰恰是加上它（纯加性在 overcapacity 下互相干扰）。**还剩什么：只剩 Δt 语义，而那一半被 4.3 覆盖。**

### 4.3 Mamba 式 (4) / Schöne 式 (7)：那对系数是 ZOH，不是新发现
Mamba：Ā = exp(ΔA)，B̄ = (ΔA)⁻¹(exp(ΔA) − I)ΔB。Schöne：Λ̄ = e^{ΛΔ}，B̄ = Λ⁻¹(Λ̄ − 1)B，Δ 是**两事件之间的真实时间**。取 A = −Λ 即候选的 exp(-ΛΔ)，而 (e^{ΛΔ}−1)/Λ 正是"1 − e^{−ΛΔ}"除以 Λ。
**候选的衰减项与写入系数一个字没变，只是把 A 写成 Λ、把 Δ 从"步长"解释成"真实秒"。还剩什么：只剩"关联记忆（v kᵀ）与这对系数同篇出现"——即 §1 那个组合空缺。**

---

## 5. 六条直觉逐条裁决

### (a) "快权重一定记得住长期目标" —— **有反证**
反证 1：Schlag 2021 逐字 "We infer a memory capacity limitation of recent linearised softmax attention variants"，并讨论 overcapacity 区：容量随 key 正交性下降，超容量即互相串扰。<https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf>
反证 2：Ba 2016 的 λ<1 意味着指数遗忘，有效时间窗 ≈ 1/(1−λ) 步。候选把 λ 换成 exp(-ΛΔ)，Λ>0 同样遗忘。**"记得住长期目标"与"有衰减"直接矛盾**，除非 Λ 在目标相关通道被学到≈0，而这不是结构保证。
**裁决：只是直觉，被容量与衰减同时限制。**

### (b) "加 Δt 就自动有连续时间语义" —— **部分成立（有条件）**
成立部分：若 Φ_Δ 真是某个连续系统 x'=f(x) 的 flow，半群 Φ_{Δ1}∘Φ_{Δ2}=Φ_{Δ1+Δ2} 自动满足，采样率不变（ZOH 保证）。
不成立部分：**只把衰减换成 exp(-ΛΔ) 不够**。系统 M' = −ΛM + u(t) 的精确采样是 M(t+Δ) = e^{−ΛΔ}M(t) + Λ⁻¹(e^{−ΛΔ}−1)(输入项)，即**输入项也须按 (1−e^{−ΛΔ})/Λ 缩放**。候选的 α_i 是自由门、不随 Δ 变，**因此候选不是任何连续时间系统的采样**。
证据：Mamba 式 (4) 的 B̄ 带该缩放；Schöne 式 (7) 同理。
**裁决：只有同时满足半群 + 输入项缩放才有连续时间语义，否则只是把 α 换个参数化，语义是假的。**

### (c) "attention 写入天然只改变化部分" —— **错（对候选结构本身不成立）**
候选写 M ← M + α v kᵀ：**每步在整个 M 上叠加**，与内容是否变化无关。要"只改变化部分"，必须像 DeltaNet 先读后写、写**残差**：S_t = S(t−1)(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ，其中 (I − βkkᵀ) 做的就是"先减掉旧关联再写新的"。
逐字证据：Schlag 2021 式 (20)-(22)；Yang 2024 博客 "DeltaNet applies this error-correction principle"；KDA 式 (1)。
**裁决：用户把 delta rule 的性质错记到了 Hebbian 写上。这条对候选结构是反的。**

### (d) "稀疏更新天然更高效" —— **错 / 条件成立**
反证：DeltaNet (Yang et al. 2024) 的整个动机是原版**逐 token 顺序更新**不能并行、"hardware-inefficient"；解法是 chunkwise **稠密矩阵乘**（WY 表示）。KDA 同样用 chunkwise 稠密算法。
**在 GPU 上，"少更新/稀疏更新"是慢路径，"看起来浪费"的稠密批量更新才是快路径。** 稀疏只在存储/事件驱动硬件上有利（Nature Comm 那类）。
**裁决：错（训练场景）。**

### (e) "矩阵记忆比普通隐状态更有表达力" —— **部分成立（容量），不成立（表达力）**
成立：矩阵状态有 d_k×d_v 个参数，关联召回（key→value）确实需要矩阵。
不成立：**可计算的函数类不因此提升**。Illusion of State 证明线性 SSM 族（含 S4/S6 与含矩阵状态的近线性递归）"cannot express computation outside TC⁰"，无法做置换组合等状态跟踪；"the 'state' in an SSM is an illusion"。此外容量仍有上界（Schlag 2021）。
**裁决：矩阵状态买的是"关联容量"，不是"计算表达力"。**

### (f) "记忆更新越少越稳定" —— **只是直觉**
反证 1：Titans 逐字——"the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local minima), and missing information about some parts of the sequence"，因此必须加动量与衰减来救。<https://arxiv.org/html/2501.00663v1>
反证 2：GLA/Mamba2 引入门控的动机是稳定性与遗忘，不是"少更新"；RetNet 的 γ 衰减同理。
反证 3：把更新系数换成"更精确的连续时间系数"也未带来收益（§7.3 那条负实验）。
**裁决：稳定性来自衰减/收缩性，不来自更新频率。"少更新"通常等于"不学习"。**

---

## 6. 仍可能形成数学贡献的清单

1. **采样率不变性定理（最有希望，但不是结构创新）**：证明当 α_i = (1 − e^{−ΛΔ})/Λ 时 M_i 等于连续时间关联记忆的精确采样，且对任意采样细分不变。**先例**：ZOH 部分已在 Mamba 式 (4)/Schöne 式 (7) 逐字给出，但"关联记忆版本"未被单独陈述。贡献等级：小（教科书推论）。
2. **双状态的分离定理（真空白）**：构造任务族证明"矩阵记忆 M + 向量隐状态 h"严格优于任一单状态，或给参数量-误差下界。**先例**：Titans MAC/MAG、Kimi Linear 的 KDA+MLA 只是工程做法，**无人给分离定理**。可做成真贡献。
3. **Δ 不均匀采样下的容量-衰减权衡（真空白）**：给定 Δ 分布，矩阵快权重有效容量如何随 Λ 变化？给误差界。**先例**：Schlag 的容量分析只针对均匀步长，无 Δ 加权版本。
4. **"不变内容不重写"的形式化（半空白）**：定义相对 (Λ,k,v) 的"Δ-不变子空间"，证明纯 Hebbian 写会破坏它而 delta 写不会。**先例**：DeltaNet 系列事实上这么做，但作为定理没被写出；且候选现在用的是 Hebbian，方向相反。
5. **退化判据（教学性，保险）**：给出 (Λ, α, K, V, R) 使系统分别等价于 GLA / KDA / Ba-快权重 / Kalman 的充要条件。
6. **稳定性条件（工程价值高）**：给出 Λ 与 α 的 Lipschitz/收缩条件使 M 有界且读出的 r 不放大噪声。**先例**：Kalman/观测器有，学习式矩阵记忆没有。

**明确不建议当作贡献的**：exp(-ΛΔ) 本身、Diag 衰减、v kᵀ 写、q 读、双状态的存在——全部有先例。

---

## 7. 反对结论与反例（独立一节）

### 7.1 直接反对"这个结构是新的"的结论（按杀伤力）

1. **GLA 式 (3) 就是候选的前三行**，只差 α_t 的来源。[链接](https://arxiv.org/html/2312.06635)
2. **Ba et al. 2016 就是"矩阵快权重 + 衰减 + 外积写 + 双状态"**，只差真实 Δ。[链接](https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf)
3. **KDA 式 (1) 是候选的加强版**（多 delta 前乘，少 Δt）。[链接](https://arxiv.org/html/2510.26692)
4. **exp(-ΛΔ) 与 (1−e^{−ΛΔ})/Λ 是 ZOH 系数**，Mamba 式 (4)、Schöne 式 (7) 逐字给出；Nature Comm 2025 已做成硬件。[Mamba](https://arxiv.org/html/2312.00752) / [Schöne](https://arxiv.org/html/2404.18508v3) / [Nature Comm](https://www.nature.com/articles/s41467-025-68227-w)
5. **"Δ 控制衰减"在向量状态上 2018 年就有**（GRU-D 式 (10)），连续时间上 2022 年就有（CfC、CRU）。[GRU-D](https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216) / [CfC](https://arxiv.org/html/2106.13898) / [CRU](https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf)
6. **双状态不是新结构**：Titans 的 MAC/MAG、Kimi Linear 的 KDA+MLA 混合。[Titans](https://arxiv.org/html/2501.00663v1)
7. **"内容寻址读写矩阵记忆"是 2014 年 NTM 的核心**，候选只是用了线性注意力的隐式寻址版本。[NTM](https://arxiv.org/abs/1410.5401)

### 7.2 反例（把候选结构按不同退化打回已知物）

1. **Λ → 0（不衰减）**：退化为 GLA（α=1）→ 线性注意力 S＝Σ k vᵀ，容量随长度饱和、串扰（Schlag 指出的 overcapacity）。
2. **Λ → ∞（瞬间遗忘）**：M_i ≈ α_i v kᵀ，只剩当前帧 → 退化为**单帧线性变换**，无记忆。
3. **K/V 取恒等、h 不参与**：退化为**纯线性注意力**（Katharopoulos 2020 的累积形式）。
4. **去掉 h（单状态）**：退化为 GLA/KDA/Titans 之一——**h 这条路是候选唯一的"额外"结构，但它没有语义定义**（用户未说 h 负责什么）。这是候选最脆弱处。
5. **α_i 取常数 1、Λ 取常数**：退化为指数加窗 Hebbian 记忆（1960s 的 leaky integrator / 指数平滑），见 Ba 2016 的特例。
6. **把写换成 delta 写（I−βkkᵀ）**：直接变成 KDA，**并失去候选自己的"纯加性"特征**——即候选的新颖性主张与直觉 (c) 互斥。

### 7.3 这类结构的已知失效模式

- **容量/串扰**：矩阵关联记忆容量随 key 正交性下降，超容量后互相干扰（Schlag 2021）。
- **表达力上界**：线性/近线性 SSM 族 ⊆ TC⁰，不能做置换组合等状态跟踪（Illusion of State 2024）。**矩阵状态不救这一点。**
- **衰减与"长期记忆"矛盾**：衰减越强越忘得快；要记得久必须 Λ→0，那就退化成无衰减 GLA（又有串扰）。
- **Δ 的数值病态**：Δt 很小时 exp(-ΛΔ)≈I，梯度病态；Δt 很大时 M_i- ≈ 0，状态被清空（等价于重新初始化）。
- **纯加性写在相关性下双计**：h 与 r 相关时信息被重复写入、过冲；DeltaNet 系的存在就是为了修这个。
- **"精确连续时间系数"没有实测收益**：一份非同行评审内部研究稿（analemma.ai）把 KDA 的 Euler 系数换成 α̃ = −expm1(−βλ)/max(λ,ε)（正是 (1−e^{−ΛΔ})/Λ），自报 "it does not improve accuracy at length extrapolation ... underperforms ... on Palindrome (−0.81 pp) and Stack (−2.98 pp) at 4× extrapolation"，且 "neither the integrator alone nor the removal of L2 normalization provides accuracy benefits"。**[非同行评审，只作弱证据]** <https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_59af4b6d-78d3-4f27-90bf-2d9c319a0093/main.pdf>
- **双状态无稳定性分析**：M 与 h 的耦合无任何已发表稳定性条件，工程上靠层归一化与残差。

### 7.4 有没有逐字对应的工作？

**有，但是"三篇各覆盖一部分"，没有同篇全覆盖：**

| 候选的组成部分 | 逐字已有的工作 |
|---|---|
| M_i- = Λ 对角衰减 ⊙ M | GLA 式 (3) Diag(α_t)S(t−1)；Mamba2 标量版；Ba 2016 的 λA(t−1) |
| M_i = M_i- + α v kᵀ | GLA 式 (3) 的 + k_tᵀ v_t；Ba 2016 的 + η h hᵀ；KDA 的 + β k vᵀ |
| r = R(M, q) | GLA/KDA/RetNet 的 o = Sᵀ q |
| h 作为第二状态 | Ba 2016（A 与 h 并存）、Titans MAC/MAG |
| exp(-ΛΔt) 这一具体参数化 | 无同篇；但作为 ZOH 系数见 Mamba 式 (4)、Schöne 式 (7) |

**结论**：候选 = **Ba 2016 / GLA 的衰减矩阵记忆 + Mamba/Schöne 的 ZOH 时间参数化 + Titans 的双状态** 的并集。**唯一没被同篇写出的，是"用真实 Δt 生成对角衰减 + 关联记忆写 + 双状态"这个三件套**——一个组合空缺，不是概念空缺。审稿人完全可以把它读成 "GLA with time-aware decay"，而那正是 Schöne/Nature Comm 对 SSM 做过的事。

### 7.5 给 lead 的三条建议
1. **别把"衰减矩阵 + 外积写 + q 读"写成贡献**，直接引 GLA 式 (3) 与 Ba 2016，说"我们沿用"。
2. **真正能主张的只有两格**：双状态的分离定理/稳定性条件（§6 第 2、6 条）；不均匀 Δ 下的容量-误差界（§6 第 3 条）。检索范围内确实没人做。
3. **必须修掉两个内部矛盾**：(a) 写是纯加性却主张"只改变化部分"——要改成 delta 写（那就变 KDA，需额外论证）；(b) h 这路状态没有语义定义——要么去掉，要么给出它严格不可替代的定理。
