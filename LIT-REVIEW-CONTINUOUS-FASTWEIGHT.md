# 连续时间快权重注意力记忆：对抗式文献审判

> 核查对象（用户提出）：
>
>     M_i⁻ = e^{−ΛΔ_i} ⊙ M_{i−1}          矩阵快权重关联记忆，按真实时间衰减
>     q_i = Q(z_i,h_i), k_i = K(z_i,h_i), v_i = V(z_i,h_i)
>     M_i = M_i⁻ + α_i v_i k_iᵀ            内容寻址写入
>     r_i = R(M_i, q_i)                    读出
>     h_i = H(h_{i−1}, r_i, z_i)           第二路隐状态
>
> 主张：用 attention/快权重维护可寻址关联记忆；不变内容不必反复重写；变化内容经 key 找到记忆并更新；衰减与写入由真实经过时间 Δt 控制，而不是按帧数。
>
> 建档 2026-11-03。**纯网络检索**（tavily_search / tavily_extract + arXiv API 核元数据），8 个独立检索代理并行，
> 覆盖 8 个专题、约 90 篇论文。每条引用均实拉正文；核不实的标 [未核实]/[仅摘要级]。

---

# 第一部分 · 判决（先看这一节）

## 0. 一句话判决

**候选结构 = GLA/RWKV 的「衰减矩阵记忆」+ DeltaNet 的「残差写」+ Ba et al. 2016 的「双状态」+ Mamba/ZOH 的「时间离散化」的并集。**
其中最关键的一条主张——**用真实秒级 Δt 同时控制矩阵记忆的衰减与写入强度**——已被
**Liquid Gated Attention（Jiang, Xu, Yang, arXiv:2608.30695, 2026-08-31）** 用几乎相同的式子实现：

    S_n = g_n S_{n−1} + (1 − g_n) k_nᵀ v_n ,   g_n = exp(−δ_n x̄_n) ,   δ_n = t_n − t_{n−1}（秒）

注意 (1 − g_n) = 1 − e^{−δ_n x̄_n} **正是用户要区分的「α = 1 − e^{−λΔt}」那一型**，不是学习门。
**候选与它的唯一差别，是候选多了一路独立隐状态 h——而这一路恰恰是数学上不自洽的部分（§4）。**

按用户要求的四档：

| 档位 | 判定 |
|---|---|
| ① 已有结构（不可主张为新网络） | **命中**。衰减矩阵记忆、外积写、矩阵读、双状态，全部有逐字先例 |
| ② 已有结构的直接组合 | **命中**。GLA + Gated DeltaNet + LGA 拼起来 ≈ 候选的 90% |
| ③ 已有组件之间存在尚未解决的数学空白 | **命中，但只剩三格**（§5），且都是「定理」不是「结构」 |
| ④ 可能形成新结构 | **不成立**，除非改写入语义（加性 → 残差擦除）并重新定义 h |

## 1. 五个判断题的直接回答

### Q1「attention 从读记忆变成写记忆」是否已被完整覆盖？——**是，1992 年就是原始设定**

- **Schmidhuber 1991/1992** 的 Fast Weight Memory 本来的定义就是「一个网络写另一个网络的权重」，写就是外积。
- **Schlag, Irie, Schmidhuber 2021**（ICML）*Linear Transformers Are Secretly Fast Weight Programmers* 把这件事形式化：
  线性注意力的更新就是快权重编程，「读」是 attention，「写」是 fast weight update。
- **DeltaNet（Schlag 2021；Yang et al. 2024 NeurIPS）**：写不再是盲加，而是先读后写残差
  S_t = S_{t−1} − β_t(S_{t−1}k_t − v_t)k_tᵀ，等价于对 ½‖Sk−v‖² 做一步 SGD。
- **TTT（Sun et al., ICML 2025）**：隐状态是模型参数，更新 = 自监督损失的一步梯度。
- **Titans（Behrouz et al., NeurIPS 2025）**：M_t = (1−α_t)M_{t−1} + S_t，S_t = η_t S_{t−1} − θ_t∇ℓ，ℓ = ‖M(k_t) − v_t‖²，写幅由 **surprise** 决定。

**结论：Q1 完全覆盖。「只有读」的线性注意力在 2021 年就被改写成「读+写」了。**

### Q2「真实 Δt 控制矩阵记忆的衰减和写入」是否已有先例？——**衰减：已有；写入强度：已有（LGA）；两者同时且矩阵态：LGA 已做**

严格按要求区分两种 α：

| 形式 | 谁在用 | 是否把物理时间写进更新律 |
|---|---|---|
| α = σ(g(z))（学习门） | Mamba（Δ=softplus(Param+Linear(x))）、GLA、Gated DeltaNet、Titans | **否**（Δ 是学习量，不是秒；Mamba 原文 Algorithm 2 / Theorem 1 明说它是 RNN 门控的推广） |
| α = 1 − e^{−λΔt}（物理时间） | **Liquid Gated Attention 2026**：g = exp(−δx̄)，写强 1−g；**GRU-D 2018**：γ=exp(−max(0,Wδ+b))（向量态）；**CRU 2022**：exp(A(t−τ))（向量态）；**FLAME 2025**：A_S = A⊙F(Δt)（矩阵动力学，带 Hurwitz 定理）；**TiM4Rec 2024**：真实时间差乘进 Mamba2 的矩阵衰减 | **是** |

- **矩阵记忆 + 真实秒 Δt + 写强随 Δt 缩放**：**LGA（2026-08）已全部命中**。
- **只在「同时还有独立 h」这一格是空的**，而 h 与 M 的时间语义不一致（§4 矛盾 2）。

### Q3 更新式是否只是已有结构的改写？——**是，有三重等价身份**

1. **= 线性注意力的循环形式 + 指数遗忘**：GLA 式(3) S_t = Diag(α_t)S_{t−1} + k_tᵀv_t；RWKV-5 式(9)；RetNet 式(6) S_n = γS_{n−1} + K_nᵀV_n。候选只是把门换成 e^{−ΛΔ}。
2. **= Hebb 记忆 + 遗忘**（若写是 v kᵀ）：读出是核平滑加权平均，**不能覆写**。
   **= delta 记忆 + 遗忘**（若写改成 (v−Mk)kᵀ）：这正是 **Gated DeltaNet** 式(1) S_t = S_{t−1}(α_t(I−β_tkkᵀ)) + β_tvkᵀ。
3. **= 一阶线性 ODE 的 ZOH 精确离散化**：dM/dt = −ΛM + U(t) 在常数输入下的精确解就是 M_{t+Δ} = e^{−ΛΔ}M + Λ⁻¹(I−e^{−ΛΔ})U。
   即 **ρ 与 α 是同一个 Δ 的两个函数，不是两个自由参数**；标量情形恒有 ρ + α = 1。
   这套系数是教科书内容（ETH 讲义原话 "no discretization errors"），深度模型里 **Mamba 式(4)** 已写死：
   Ā = exp(ΔA)、B̄ = (ΔA)⁻¹(exp(ΔA)−I)ΔB；矩阵记忆版见 **EFLA（arXiv:2512.12602）**：
   S_t = e^{−β_tA_t}S_{t−1} + ∫₀^{β_t}e^{−(β_t−τ)A_t}b dτ。

### Q4「只更新变化部分」是否已被覆盖？——**判据被覆盖，但候选公式本身没有这个机制**

- **判据**四类都已有先例：阈值触发（Trimpe & D'Andrea 2012，带误差上界）、surprise（Titans）、门控（Mamba/GLA）、路由（RIMs top-k、MoE、Skip RNN、NTM）。
- **但候选的 M ← M⁻ + α v kᵀ 是每一步都写**，没有任何「跳过/稀疏」结构：
  **「不变内容不重复重写」不是这个公式的性质**，而是要靠 delta 写（写幅 = 残差）或硬阈值才成立。
- **严格不变**只在硬机制里出现：RIMs（h_{t+1,k}=h_{t,k}）、Skip RNN（u∈{0,1}）、NTM（w=0）、TGN（无事件不写）、Trimpe（无触发不更新）。软门控（Mamba 的 Ā∈(0,1)、Titans 的 1−α）**永远不完全不变**。

### Q5 哪些部分仍可能形成真正的数学贡献？——**只有三条，且都是定理**

见 §5。核心一句：**结构已被占满，数学没被占满。**

## 2. 组件—先例—覆盖—可主张（用户要求的表）

| 组件 | 最接近先例 | 是否已被完整覆盖 | 我们还能主张什么 |
|---|---|---|---|
| 快权重矩阵记忆 | **Ba et al. 2016** A(t)=λA(t−1)+η h hᵀ（且 A 与 h 并存）；FWP 2021；RetNet；GLA | **是** | 无（只能声称用它） |
| attention 写入 | **Schmidhuber 1992**；**Schlag 2021**；DeltaNet 2024；TTT 2025；Titans 2025 | **是** | 无 |
| 内容寻址 | DeltaNet S←S−β(Sk−v)kᵀ；Titans ∇‖Mk−v‖²；tt-regression β(v−Mk)kᵀ | **是** | 候选的**加性**写是它的减弱版，不是加强版 |
| 真实时间衰减 | **LGA 2026**（矩阵态+秒级）；TiM4Rec 2024；GRU-D 2018；FLAME 2025 | **是**（矩阵态已有） | 无（只能做跨采样率实验） |
| 变化驱动写入 | Titans（surprise）；Trimpe 2012（阈值+误差界）；RIMs（硬 top-k） | **判据覆盖，候选公式未实现** | 硬对象级路由 + 严格不变性（但控制界已有逐通道版） |
| 对象级局部更新 | Slot Attention / SAVi / STEVE / SlotFormer / SOLD | **部分**（软注意力，无硬保证） | 硬槽位写入 + 不变性证明（**窄缝**） |
| 不规则视频流 | video 域 11 篇全文扫描 Δt/delta t/elapsed/wall-clock 计数 **= 0**；MovieChat/MA-LMM 用相似度而非时间 | **空白**（但属「补一个缺失」） | 真实 PTS 进记忆更新（**唯一未被占的工程点**） |
| 稳定性/容量定理 | 容量：Barnfield 2026 Thm1(ii)；表达力：Merrill ICML 2024（⊆TC⁰）、Sarrof 2024；串扰：FWP §4.1 | **否——全族无人给** | **半群/采样率不变性定理；双状态耦合稳定性；衰减率↔容量联合界**（**唯一真正的空地**） |

## 3. 六条直觉的裁决（用户要求「必须给出反对结论」）

| 直觉 | 裁决 | 证据 |
|---|---|---|
| (a) 快权重一定记得住长期目标 | **有反证** | d×d 线性记忆 top-1 需 d² ≳ n log n，任何 W 都突破不了 C·n log n（Barnfield 2026 Thm1(ii)）；若读是 Mq，容量退化到 **O(d)**（FWP §4.1 原话：超过 d 个关联就有检索误差）；线性 SSM/矩阵态 ⊆ **TC⁰**，做不了置换复合（Merrill 2024） |
| (b) 加 Δt 就自动有连续时间语义 | **部分成立，有条件** | 只有 ρ 与 α **同源**于同一个 Δ（ZOH）时才等价于 ODE；把 α 当独立学习门 ⇒ (M_i−M_{i−1})/Δ → αΛ⁻¹U/Δ 发散，Δ→0 无极限。Mamba 的 Δ 是学习门，其 exp(ΔA) **不满足半群** |
| (c) attention 写入天然只改变化部分 | **错** | 外积写 v kᵀ 是**稠密秩 1 写**，改 M 的所有行；擦除必须靠 (I−βkkᵀ) 前乘（delta 规则）。**候选的写是纯加性，恰是「只改变化部分」的减弱版** |
| (d) 稀疏更新天然提高效率 | **错/未证** | MoE 是唯一硬数据（容量 >1000×）；DeltaNet 家族反过来做 **chunkwise 稠密化**正是为规避逐 token 顺序更新；门控本身有开销；理论 FLOPs ≠ 墙钟 |
| (e) 矩阵记忆比普通隐状态更有表达力 | **错** | d×d 矩阵 = d² 维状态；rank-1 写是**限制**；线性 SSM 复杂度类与向量态相同（⊆TC⁰）。它买的是**容量常数**，不是表达力 |
| (f) 记忆更新越少越稳定 | **只是直觉** | 稳定性来自衰减/有界增益：‖M_i‖ ≤ max(‖M_0‖, sup‖vkᵀ‖/λ)，与更新频率无关；跳更新反而让每次 Δ 变大、单次修正变大；事件触发滤波需要额外稳定性条件（Trimpe Thm1） |

## 4. 候选结构内部的四个矛盾（必须先解决，否则无法写论文）

1. **加性写 vs「只改变化部分」**：M ← M + α v kᵀ 在内容不变时仍然写。要实现主张必须换成 delta 写 α(v−Mk)kᵀ，
   但那就变成 Gated DeltaNet（且写幅由预测误差决定，α 不再是自由门）。
2. **双状态的时间语义不一致**：M 按真实 Δt 演化、h 按序号 i 演化 ⇒ 30Hz 与 60Hz 采样下 M 轨迹近似相同、h 完全不同
   ⇒ **整个系统没有连续时间极限**，半群只在 M 上成立。全族 19 篇里**没有任何双状态先例**，不做的原因很可能就是它不自洽。
3. **α 被要求同时「内容相关」和「由 Δt 控制」**：二者必须相乘成 α_i = (1−e^{−ΛΔ_i})·s(z_i)，
   否则 ZOH 恒等式被拆开、半群与细化一致性同时失效。
4. **写入语义未定义**：v kᵀ 若按**脉冲**（Dirac）驱动离散化，得到 M(t_i⁺) = e^{−ΛΔ}M + v kᵀ，**里面根本没有 α**；
   只有把写入当作区间内的**常值驱动**才会出现 α。候选公式恰好掩盖了这个选择（07 节 R6）。

## 5. 唯一还剩的三格（都是定理，不是结构）

1. **半群/采样率不变性定理（矩阵记忆版）**：全族 19 篇无人给出。ZOH 给了「衰减半群」这一半；
   缺的是 **带写入项 + 不均匀 Δ + 双状态** 的完整陈述与误差阶。注意：这本质是数值分析应用，单独发表偏弱。
2. **双状态 (M, h) 耦合的分离/稳定性定理**：真正空白（无人做），但必须先解决 §4 矛盾 2，否则定理是错的。
3. **衰减率 ↔ 容量的联合界**：容量上界已有（Barnfield 2026），衰减↔有效视界的关系没有严格结果；
   「有效记忆窗 ~1/λ、可分辨对象数 ~ 1/(ε·λ·平均间隔)」是可写但需证的东西。

**外加一条被普遍忽视的**：**写入的连续极限（脉冲 vs 常值驱动）及其误差阶**——07 节指出 α 的存在性依赖于这个建模选择，而所有既有工作都没讨论它。

## 6. 检索边界与未核实（诚实交代）

- **四件同篇**（矩阵 M + 真实秒 Δt + 内容寻址写 + 独立 h）：**未找到**。
- **三件同篇**：LGA 2026（矩阵+真实Δt+写）；Ba 2016（矩阵+写+双状态，无 Δt）；Titans（矩阵+写+遗忘，无 Δt）。
- **视频域**：11 篇全文扫描 Δt/delta t/elapsed/wall-clock **计数为 0**——「视频域真实 Δ」确实没人做。
- **定理**：全族 **0 篇**给出记忆相关定理（Mamba 的 Ā 是唯一被引用的性质）。
- **未核实/降级**：各节已标；LGA 附录仅标题级；EFLA 有别名；Hopfield 1982 原文抽取为空（二手）；Titans 正文对 surprise 的定义自相矛盾（§1 说对输入、Eq 8 实际对记忆参数）。
- **一篇需打折的**：*Exponential Integrator for Diagonal-Decay Delta Attention: A Negative Result*（FARS/Analemma 2026，**自动生成、非同行评审**）报告「精确指数积分器只带来稳定性、不带来精度」，与 §5 第 1 格的判断方向一致。

## 7. 目录

- **01** Fast weights / 线性注意力 / 现代矩阵记忆（19 篇：Schmidhuber 91/92、FWP 2021、Linear Transformer、Performer、DeltaNet、Gated DeltaNet、RetNet、RWKV-4/5/6/7、Hyena、GLA、Forgetting Transformer、Hopfield、CT-FWP、EFLA、LGA、Titans）
- **02** 测试时学习与可变权重（14 篇：TTT、Titans、TTT-MLP、Memorizing Transformers、Memory Layers、tt-regression、kNN-LM、Modern Hopfield、GRU-D、Phased LSTM、FLAME 等）
- **03** 事件/稀疏写入与选择性状态（15 篇：Mamba、Titans、TTT、DeltaNet、Gated DeltaNet、Hopfield、NTM、TGN、RIMs、MoE、Slot Attention、Skip RNN、ACT、Trimpe 2012、Sijs 2009）
- **04** 连续时间/不规则采样记忆（13 篇：Mamba、TiM4Rec、CRU、GRU-D、Ba 2016、Schlag 2021、RetNet、RWKV、GLA、Gated DeltaNet、Titans、ContiFormer、TiSASRec）
- **05** 视频流记忆（14 条：MeMViT、MovieChat、MA-LMM、Flash-VStream、MeMOT、SAVi/STEVE/SlotFormer/SOLD 等）
- **06** 对抗式检索（17 篇 + 六条直觉逐条反证 + 退化反例）
- **07** ZOH/指数积分器等价性核查（10 篇 + R1–R6）
- **08** 容量与表达力上界（10 篇：Barnfield 2026、FWP、DeltaNet、Hopfield、Merrill 2024、Sarrof 2024、Zoology、RetNet、GRU-D、SkipRNN）
- **09** Lead 数学分析（三个等价身份 / 四个可证命题 / α 的细化极限反例 / 双状态冲突）

---


# 01 · Fast weights / 线性注意力 / 现代矩阵记忆族

> 专题：用 attention / 快权重维护可寻址关联记忆的谱系，及其与候选结构的逐项对齐。
> 检索日期：2026-11-03。全部链接实拉核实（arXiv API 核元数据 + 正文 PDF/HTML 抽公式）。
> 本文件只写这一个专题，不改动项目其它文档。

## 0. 结论速览（先给 lead 结论）

1. **候选结构的"写 + 读"半边 = DeltaNet 家族（2021–2025），已被完整发表并工业化**。核心公式
   逐字对应：先按 key 检索旧值 v_old = M k，再写残差 α(v − v_old)；"不变内容不重写"就是这个
   机制的字面表述。
2. **"逐通道对角衰减 Λ" 也已存在**：GLA 的 S_t = Diag(α_t) S_{t-1} + k_t^T v_t（Eq.3），α_t 是
   逐 key 通道向量。对角 ⇒ 矩阵乘积可交换 ⇒ 累积衰减是精确乘半群，这恰恰是候选结构想要的
   数学性质，但它是**按观测序号 t 的**，不是按秒。
3. **"用真实 Δt 驱动衰减"已有一篇几乎正面对撞的工作**：Liquid Gated Attention（arXiv:2608.30695，
   2026-08-31），S_n = g_n S_{n-1} + (1−g_n) k_n^T v_n，g_n = exp(−δ_n x̄_n)，δ_n = t_n − t_{n−1}
   是真实时间间隔（秒）。它自称 fast-weight programming，且有线性注意力的并行形式。
4. **连续时间版本更早就有**：Irie/Faccio/Schmidhuber 2022（NeurIPS）把快权重矩阵写成
   W(t) = W(t0) + ∫F_θ(W(s),x(s))ds，直接吃实时时间戳；EFLA（2025-12）则证明 delta rule 的
   β_t 本身就是 ODE 的**积分步长**。
5. **没找到逐字对应的工作**：没有任何一篇同时具备「矩阵 M + 独立隐状态 h + Δt 控制衰减 + delta 写入」。
   两个半边都已被占，组合本身是空的 —— 但**空的组合不等于有根据的组合**，见 §4。
6. 本文件覆盖 19 篇，其中 7 条直接构成对候选结构的反证（§4）。

## 1. 比对基准：把候选结构拆成可判定的 6 个命题

被检验候选（Lead 给定）：

    M_i^- = exp(-Lambda ⊙ Delta_i) ⊙ M_{i-1}      (P1 逐通道对角衰减，Delta_i 为真实时间间隔)
    q_i = Q(z_i,h_i); k_i = K(z_i,h_i); v_i = V(z_i,h_i)
    M_i   = M_i^- + alpha_i * v_i k_i^T            (P2 内容寻址加法写入)
    r_i   = R(M_i, q_i)                            (P3 写后读)
    h_i   = H(h_{i-1}, r_i, z_i)                   (P4 独立隐状态，双状态耦合)

拆成可独立判定的 6 条，后文每篇按这 6 条打分：

- **P0 记忆状态**：是否为 d_v × d_k 矩阵（不是向量、不是 slot 表、不是参数）。
- **P1 衰减**：是否有逐通道对角衰减；衰减量由 token 序号还是真实 Δt 决定。
- **P2 写入**：是否 M ← M + something；something 是全量外积还是**残差**（内容寻址）。
- **P3 读**：是否 M q（矩阵-向量），是否写后读。
- **P4 双状态**：除 M 外是否还有一条标量/向量隐状态与 M 耦合。
- **P5 理论**：有无容量 / 干扰 / 半群 / 采样率不变 / 稳定性 / 误差界定理。

## 2. 逐篇核对

### 1. Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks
- 作者 / 年份 / 会议或期刊 / 链接：Jürgen Schmidhuber / 1991(TR) & 1992(期刊) / 技术报告 FKI-147-91, TU München, 1991-03-26；期刊版 Neural Computation 4(1):131–139, 1992 / TR OCR 全文：https://people.idsia.ch/~juergen/FKI-147-91ocr.pdf ；期刊 DOI 10.1162/neco.1992.4.1.131（ACM DL：Neural Computation, Vol.4, Iss.1, pp.131-139）
- 记忆状态是什么：快网络 F 的**全部突触权重集合 W_F**（矩阵，维度由 F 的结构定，文中未固定为 d×d）。慢网络 S 的权重 W_S 是普通（慢）参数。
- 读操作精确公式：F 按常规前向传播用 W_F(t−1) 算 y(t)；原文未给矩阵-向量形式（F 可为任意前馈网）。→ 无统一"读公式"。
- 写操作精确公式：原文 Eq.(1) w_ab(t) = u(w_ab(t−1), Δw_ab(t))；Eq.(4) Δw_ab(t) = S_ab(t)（一权重一输出单元）；Eq.(5) Δw_ab(t) = g(s_a(t), s_b(t)) = **s_a(t)·s_b(t)**（Hebb 型外积，"representative example"）。u 取 sum-and-squash 时即 W(t)=σ(W(t−1)+s_a⊗s_b)。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M ← M + 内容寻址更新：**部分**。有 M ← u(M, Δ) 与 Δ=外积，但**无内容寻址**（写入不先读旧值、不写残差）。
- alpha_t 究竟是什么：**其他 —— 无标量学习率**。Δw 直接是 S 的输出（Eq.5 为两向量之积），没有门、没有标量系数。
- 是否使用物理时间 Δt：**否**。原文只说 "at time step t"，t 是离散步序号；"Two experiments with unknown time delays" 里的 delay 是任务属性的固定延迟，不是输入的时间戳（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**（论文层面完全未讨论；t 只作 0..n_p 的整数索引）。
- 有无定理：**无**。（只有算法推导与两个实验；无收敛/容量陈述。）
- 用于：通用序列学习（小规模实验：未知时延任务、自适应临时变量绑定）。非视频/非传感器。
- 与候选结构的异同：衰减项：**无**（u 是 sum-and-squash，不含任何遗忘）。写入项：外积，但**无残差、无 α**。读操作：无统一形式。双状态：**本质上是双状态**（慢网 S + 快网 F），但快状态只是 F 的权重，不含独立 h。Δt 落点：无。
- 裁决：**部分覆盖**。它是"快权重矩阵 = 关联记忆"的始祖，但只覆盖候选结构的 P0 与 P2 的"加外积"部分。证据：Eq.(1)(5) 原文。

### 2. Linear Transformers Are Secretly Fast Weight Programmers
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139:9355–9366 / https://proceedings.mlr.press/v139/schlag21a.html （PDF 实拉核实，公式逐条取自正文）
- 记忆状态是什么：**KV 矩阵** W^(i) ∈ R^{d_out × d_key}（正文用 W，无下标维度约束）。另有归一化累加器 z^(i) ∈ R^{d_dot}。
- 读操作精确公式：Eq.(20) v̄^(i) = W^(i−1) φ(k^(i))；Eq.(25) y^(i) = W^(i) φ(q^(i))（**写后读**）；带归一化版 Eq.(27)(28)：v̄^(i) = W^(i−1)φ(k^(i)) / (z^(i−1)·φ(k^(i)))。
- 写操作精确公式：加法版 Eq.(17) W^(i) = W^(i−1) + v^(i) ⊗ φ(k^(i))。delta 版 Eq.(21) β^(i)=σ(W_β x^(i))；Eq.(22) v_new^(i)=β^(i)v^(i)+(1−β^(i))v̄^(i)；Eq.(23) W^(i) = W^(i−1) + v_new^(i)⊗φ(k^(i)) − v̄^(i)⊗φ(k^(i))；等价 Eq.(24) W^(i) = W^(i−1) + β^(i)(v^(i) − v̄^(i)) ⊗ φ(k^(i))。
- 是否存在 M ← M + 内容寻址更新：**是**。Eq.(23)(24) 逐字：先 remove 旧关联、再 write 新关联，残差 = β(v − v̄)。
- alpha_t 究竟是什么：**学出的门**。β^(i)=σ(W_β x^(i))∈(0,1)，原文称 "dynamic learning rate"（Sec.4.2 末）。
- 是否使用物理时间 Δt：**否**。全篇只有 i（序列位置）。（故 Δt 落点不适用）
- 是否支持不规则采样、丢帧、重复帧：**否**。重复帧会按 Eq.(24) 产生一次 β(v−v̄) 的额外写入（若 v=k 相同则 v−v̄≈0 自然近零，但不是被设计出来的）。
- 有无定理：**无形式定理，但有硬性容量论证**（Sec.4.1）：线性注意力用矩阵乘检索，键必须两两正交才不互相干扰；d_dot 维空间最多 d_dot 个正交向量，故"存储超过 d_dot 个关联必然产生检索误差"，序列长于 d_dot 就进入 overcapacity。这是该族最常被引用的容量结论。
- 用于：仅语言建模 + 合成检索任务（WMT14 EN-DE、WikiText-103）。
- 与候选结构的异同：衰减项：**无**（候选的 P1 完全缺失，这是它 2021 年被 GLA/Mamba2 补上的缺口）。写入项：**与候选 P2 同构且更强**（带 remove）。读操作：**同**（Mq，写后读）。双状态：**无**（只有 W）。Δt 落点：无。
- 裁决：**部分覆盖（覆盖 P0/P2/P3，缺 P1/P4）**。证据：Eq.(17)(23)(24)；容量论证 Sec.4.1。

### 3. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- 作者 / 年份 / 会议或期刊 / 链接：Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret / 2020 / ICML 2020, PMLR 119 / https://arxiv.org/abs/2006.16236 （PDF 实拉，Eq.16–20 逐条取）
- 记忆状态是什么：**KV 矩阵** s_i ∈ R^{d×d}，另有归一化向量 z_i ∈ R^d。
- 读操作精确公式：Eq.(20) y_i = f_l( φ(x_iW_Q)^T s_i / (φ(x_iW_Q)^T z_i) + x_i )。
- 写操作精确公式：Eq.(18) s_i = s_{i−1} + φ(x_iW_K)(x_iW_V)^T；Eq.(19) z_i = z_{i−1} + φ(x_iW_K)。
- 是否存在 M ← M + 内容寻址更新：**否**（纯加法，无 remove、无残差）。
- alpha_t 究竟是什么：**attention 权重**。α 只以归一化分母 φ(q)^T z 的形式出现，是"软"attention 权重之和；没有可学习的写入门。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**（且重复帧会被重复累加，正是候选结构想避免的行为）。
- 有无定理：**无**（只有复杂度 O(N) 的推导与实验）。
- 用于：仅语言建模；另有图像生成（MNIST/CIFAR 像素序列）与语音识别实验。
- 与候选结构的异同：衰减项：**无，且这正是后来所有 gated 工作的动机**。写入项：加外积，**无内容寻址**。读操作：同。双状态：无（z 只是归一化累加器，不是 h）。Δt 落点：无。
- 裁决：**未覆盖 P1/P2 的关键部分**；P0/P3 覆盖。证据：Eq.(18) 无任何衰减因子。

### 4. Rethinking Attention with Performers
- 作者 / 年份 / 会议或期刊 / 链接：Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, et al. / 2020(arXiv) & ICLR 2021 / ICLR 2021 / https://arxiv.org/abs/2009.14794
- 记忆状态是什么：**隐式 KV 和不显式存储**。正文只给 Att→(Q,K,V) = D̃^{-1}ÃV（Ã=tril(A)），FAVOR+ 用随机特征 φ 使 E[φ(k)^Tφ(q)] ≈ exp(k^Tq)。等价的累加矩阵 Σ_{j≤i} φ(k_j)v_j^T 存在但论文未命名为状态。
- 读操作精确公式：Eq.(2) 的单向形式 Ã=tril(A)，输出 = D̃^{-1}ÃV；FAVOR+ 下即 (Σφ(k_j)v_j^T)φ(q_i)。
- 写操作精确公式：**正文未写递推式**。只能从 FAVOR+ 定义推出 M_i = M_{i−1} + φ(k_i)v_i^T。→ 标 [仅由定义推出，非原文公式]。
- 是否存在 M ← M + 内容寻址更新：**否**（若按上式，纯加法）。
- alpha_t 究竟是什么：**attention 权重**（无门；α 是归一化系数）。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**有，且是这族最严格的近似误差定理**。Theorem 1：正则化 softmax 核与 softmax 核之比的界 inf A_reg/A ≥ 1 − 2d^{-1/3} + o(...)，sup ≤ 1；Theorem 2：正交随机特征的 MSE 严格小于非正交；Theorem 3：估计值尾概率的指数界。另有"无偏/近无偏、一致收敛、低估计方差"的陈述。**这些定理讲的是核近似误差，与容量/采样率无关**。
- 用于：仅语言建模（+ 蛋白质序列），非视频/传感器。
- 与候选结构的异同：衰减/写入/双状态全无；Δt 无。它对本专题的价值是**反例**：它说明"线性注意力"这条路线的理论产出全部集中在核近似上，从未触及时间语义。
- 裁决：**未覆盖**（除 P0 的隐式矩阵外无一命中）。

### 5. Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim / 2024 / NeurIPS 2024 / https://arxiv.org/abs/2406.06484 （HTML v6 实拉，Sec.2.2 逐条取）
- 记忆状态是什么：**KV 矩阵** S_t ∈ R^{d×d}（d_v × d_k）。
- 读操作精确公式：Sec.2.2 末：o_t = S_t q_t（写后读）。
- 写操作精确公式：S_t = S_{t−1} − β_t (S_{t−1}k_t − v_t) k_t^T；等价 S_t = S_{t−1}(I − β_t k_t k_t^T) + β_t v_t k_t^T。β_t = σ(W_β x_t) ∈ (0,1)。预测 v_t^old = S_{t−1}k_t，v_t^new = β_t v_t + (1−β_t)v_t^old，S_t = S_{t−1} − v_t^old k_t^T + v_t^new k_t^T。
- 是否存在 M ← M + 内容寻址更新：**是**，且是教科书级清晰版："check, erase, rewrite"。
- alpha_t 究竟是什么：**学出的门**（β_t = σ(W_β x_t)，原文称 soft writing strength / learning rate）。也明确指出它等价于 1-step SGD 的学习率：L_t(S) = ½‖S k_t − v_t‖²，S_t = S_{t−1} − β_t ∇L_t。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。重复帧（k_t,v_t 与上一帧相同）则 v_t^old ≈ v_t，Δ≈0，写入自然近零 —— 这是 delta 规则的**副作用**，不是被设计的"跳过重复帧"机制，且原文未提及。
- 有无定理：**无**。有 Householder 参数化（S_t = S_{t−1}(I − β k k^T)+... 是广义 Householder 乘积）与 compact WY 表示带来的并行算法，但无收敛/容量/采样定理。
- 用于：仅语言建模（1.3B / 100B tokens）+ 合成 in-context retrieval。
- 与候选结构的异同：衰减项：**无显式衰减**（I − βkk^T 是对特定方向的"擦除"，不是全谱衰减）。写入项：**与候选 P2 完全同构且更强**（残差 + 学习率门）。读操作：同。双状态：无。Δt 落点：无。
- 裁决：**部分覆盖**：P2/P3 完整，P1/P4 缺。

### 6. Gated Delta Networks: Improving Mamba2 with Delta Rule
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Jan Kautz, Ali Hatamizadeh / 2024(arXiv) & ICLR 2025 / ICLR 2025 / https://arxiv.org/abs/2412.06464 （HTML v3 实拉）
- 记忆状态是什么：**KV 矩阵** S_t ∈ R^{d_v×d_k}。
- 读操作精确公式：o_t = S_t q_t。
- 写操作精确公式：**Eq.(10)** S_t = S_{t−1}( α_t (I − β_t k_t k_t^T) ) + β_t v_t k_t^T，α_t∈(0,1) 数据依赖标量衰减，β_t 自适应学习率。
- 是否存在 M ← M + 内容寻址更新：**是**（delta 结构 + 独立衰减门）。
- alpha_t 究竟是什么：**两个不同的量，原文都叫 α/β**：β_t = 学出的门（learning rate）；α_t = 学出的门（decay，标量 per head，非逐通道）。Table 1 明确把它写成"在 SGD 更新上加 adaptive weight decay"。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**无形式定理**，但 Table 1 给出统一刻画：LA / Mamba2 / Longhorn / DeltaNet / Gated DeltaNet 各自对应一个在线学习目标函数，state update 是它的闭式解或一步显式梯度。这是本族最有解释力的"准定理"。
- 用于：仅语言建模 + 长上下文检索。
- 与候选结构的异同：衰减项：**有，但是标量（per head），不是逐通道对角**。写入项：delta，同构。读操作：同。双状态：无。Δt 落点：无。
- 裁决：**部分覆盖**。**但它是候选结构最强的"已被发表"证据**：把"衰减 + 内容寻址写"合成一个式子，Eq.(10) 就是候选的 (P1+P2) 去掉 Δt 后的形状。
- 附带关键实验结论（见 §4）：Sec.3.2 单针检索消融的标题直译——"**Decay hurts memory retention**"（衰减损害记忆保持）、"Gating facilitates filtering"（门控才起过滤作用）。这是对"衰减是核心机制"最直接的实验反驳。

### 7. Retentive Network: A Successor to Transformer for Large Language Models
- 作者 / 年份 / 会议或期刊 / 链接：Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, et al. / 2023 / arXiv 2307.08621（未见正式会议收录信息）/ https://arxiv.org/abs/2307.08621 （PDF 实拉）
- 记忆状态是什么：**KV 矩阵** S_n ∈ R^{d_k×d_v}。
- 读操作精确公式：Eq.(6) 第二式 Retention(X_n) = Q_n S_n。
- 写操作精确公式：**Eq.(6)** S_n = γ S_{n−1} + K_n^T V_n。
- 是否存在 M ← M + 内容寻址更新：**否**（纯加法 + 常数衰减）。
- alpha_t 究竟是什么：**常数 λ**。γ = 1 − 2^{−5−arange(0,h)} ∈ R^h，按 head 预算的常数，**不学习、不依赖输入**（消融里还试了全体 γ=127/128）。整个衰减不含任何门。
- 是否使用物理时间 Δt：**否**。γ 按 token 计。（故 Δt 落点不适用）
- 是否支持不规则采样、丢帧、重复帧：**否**；重复帧会被重复累加并被额外 γ 衰减一次。
- 有无定理：**无**。原文无 Theorem 环境（grep 全篇 "Theorem|theorem" 无命中）。只有"head dimension 即记忆容量"的定性说法（§3.5 消融："From the recurrent perspective, the head dimension implies the memory capacity of hidden states"，并把 head dim 从 256 降到 64 做消融）。此外给了 parallel / recurrent / chunkwise 三种表示的等价性推导（非定理）。
- 用于：仅语言建模。
- 与候选结构的异同：衰减项：**有，但常数 λ、非逐通道（逐 head 标量）、非 Δt**。写入项：加外积，无残差。读操作：同。双状态：无。Δt 落点：无。
- 裁决：**部分覆盖 P1 的"常数衰减"退化解**，P2 未覆盖。证据：Eq.(6)。

### 8. RWKV: Reinventing RNNs for the Transformer Era (RWKV-4)
- 作者 / 年份 / 会议或期刊 / 链接：Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, et al. / 2023 / Findings of EMNLP 2023 / https://arxiv.org/abs/2305.13048 （PDF + Eagle 复述式 Eq.2 双源核实）
- 记忆状态是什么：**向量状态**（wkv 是 d 维向量分子/分母），**不是矩阵**。
- 读操作精确公式：wkv_t = [Σ_{i=1}^{t−1} exp(−(t−1−i)w + k_i) ⊙ v_i + exp(u+k_t)⊙v_t] / [Σ_{i=1}^{t−1} exp(−(t−1−i)w + k_i) + exp(u+k_t)]（Eagle 论文 Eq.(2) 的 RWKV-4 复述式；原论文同构）。
- 写操作精确公式：无矩阵写式。等价地是对两个累加向量的加权累加。
- 是否存在 M ← M + 内容寻址更新：**否**（且状态是向量，不是矩阵）。
- alpha_t 究竟是什么：**常数 λ**。Eq.(10) w_{t,i} = −(t−i)w，w∈R_{≥0}^d 是学出的**时不变**逐通道衰减率；衰减随"相对 token 距离 (t−i)"指数下降。u 是"只对当前输入的 bonus"，时不变。
- 是否使用物理时间 Δt：**否**。**这是本族最纯粹的"按帧数衰减"样本**：衰减指数 = (t−i)，t、i 是 token 序号。（故 Δt 落点不适用）
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**无**。
- 用于：仅语言建模。
- 与候选结构的异同：衰减项：有，逐通道、学出、但**按 token 距离不是按秒** —— 正是候选结构要反对的那种设计。写入项：非矩阵、无残差。读操作：非矩阵乘。双状态：无。Δt 落点：无。
- 裁决：**未覆盖**（矩阵/内容寻址均缺）；但它是候选结构"反对按帧数衰减"这一动机的**最佳靶子出处**。证据：Eq.(10) w_{t,i}=−(t−i)w。

### 9. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence (RWKV-5/6)
- 作者 / 年份 / 会议或期刊 / 链接：Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, et al. / 2024 / arXiv 2404.05892 / https://arxiv.org/abs/2404.05892 （PDF 实拉）
- 记忆状态是什么：**KV 矩阵**（Eagle 起 "multi-headed matrix-valued states"），s ∈ R^{(D/h)×(D/h)}。
- 读操作精确公式：Eq.(8) wkv' = s + diag(u)·k^T·v；Eq.(7) o_t = concat( SiLU(g_t) ⊙ LayerNorm(r_t · wkv_t) ) W_o。
- 写操作精确公式：**Eq.(9)** s' = diag(w)·s + k^T·v，其中 Eq.(5) w = exp(−exp(ω))，ω∈R^{D/h} 是 head 内可训练参数。
- 是否存在 M ← M + 内容寻址更新：**否**（加法 + 逐通道对角衰减，无残差）。
- alpha_t 究竟是什么：**RWKV-5 = 常数 λ**（w = exp(−exp(ω)) 学出但时不变，逐通道）；**RWKV-6(Finch) = 学出的门**（数据依赖衰减，GLA Table 1 记其参数化为 α_t = exp(−exp(x_t W_α))，逐 key 通道）。→ 归类：v5 常数 λ，v6 学出的门。
- 是否使用物理时间 Δt：**否**。两版都是按 token 一步一次。（故 Δt 落点不适用）
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**无**（但有一句可引的稳定性陈述："w = exp(−exp(ω)) 保证 w 落在 (0,1)，从而 diag(w) 是 contraction matrix"。这是收缩性，不是定理）。
- 用于：仅语言建模（多语言）。
- 与候选结构的异同：衰减项：**v5 的 diag(w)·s 就是候选 M^- = exp(−ΛΔ)⊙M 的 Δ≡1 特例，且是逐通道对角**——这是候选 P1 在"离散时间"下的精确原型。写入项：加外积，无残差。读操作：同。双状态：无。Δt 落点：无。
- 裁决：**部分覆盖（P0+P1 的 token 版），P2 残差/P4/Δt 全缺**。证据：Eq.(9) 逐字。

### 10. RWKV-7 "Goose" with Expressive Dynamic State Evolution
- 作者 / 年份 / 会议或期刊 / 链接：Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, et al. / 2025 / arXiv 2503.14456 / https://arxiv.org/abs/2503.14456 （PDF 实拉）
- 记忆状态是什么：**KV 矩阵** S_t。
- 读操作精确公式：沿用 RWKV-6 的 r_t·S_t 读出（论文 §4.1 Time Mixing）。
- 写操作精确公式：S_t = S_{t−1}( diag(w_t) − k_t^T k_t diag(a_t) ) + v_t^T k_t diag(a_t)（论文 Concurrent Work 段落给出该式，并说明它是 RWKV-7 使用的 generalized delta rule）。
- 是否存在 M ← M + 内容寻址更新：**是**。状态转移矩阵 = 对角衰减 − rank-1 修正，一次同时完成"衰减 + 擦除 + 写入"。
- alpha_t 究竟是什么：**学出的门**，且已被推到向量值：w_t 是数据依赖的向量衰减，a_t 是**向量值 in-context learning rate**（逐通道决定替换哪一维）。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**有，但是复杂度/表达力定理，不是容量或采样定理**。原文列为首要贡献之一："Proofs that the generalized delta rule employed in RWKV-7 can solve problems outside of TC0 ... including solving an S5 state tracking problem known to be in NC1 using only a single layer, and recognizing all regular languages using a constant number of layers"。
- 用于：仅语言建模。
- 与候选结构的异同：衰减项：**有，向量值、数据依赖、与 rank-1 写合并成一个转移矩阵** —— 在数学上比候选的"先衰减再加法写"更紧，因为候选把两者写成两个可分离算子。写入项：广义 delta，同构且更强（key/value 解耦、逐通道学习率）。读操作：同。双状态：无。Δt 落点：无。
- 裁决：**部分覆盖**。**这是"衰减项 + 写入项合成单一转移矩阵"的最好范例**，可作为候选结构改进的直接模板。证据：上述转移矩阵式。

### 11. Hyena Hierarchy: Towards Larger Convolutional Language Models
- 作者 / 年份 / 会议或期刊 / 链接：Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, et al. / 2023 / ICML 2023 / https://arxiv.org/abs/2302.10866 （PDF 实拉）
- 记忆状态是什么：**其他 —— 没有状态**。Hyena 算子是"隐式长卷积 h（由前馈网参数化的滤波器）+ 逐元素乘法门控"的递推组合；记忆被压进卷积核参数（慢参数），推理时不存在随时间演化的 M。
- 读操作精确公式：无 M 可读。算子形式见 Fig.1 说明："can equivalently be expressed as a multiplication with data-controlled (conditioned by the input u) diagonal matrices D_x and Toeplitz matrices S_h"。
- 写操作精确公式：**不存在**（无状态即无写）。
- 是否存在 M ← M + 内容寻址更新：**否**。
- alpha_t 究竟是什么：**其他**。无 α；门控是与数据相关的乘法交互，不是写在存储器上的权重。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**（卷积长度按位置计）。
- 有无定理：**无**（论文主体是算子设计 + 大规模实验；无内存容量/误差定理）。
- 用于：仅语言建模（WikiText103、The Pile）+ 图像/音频。
- 与候选结构的异同：**五个维度全部不命中**，但它的价值是**反例**：它证明"不做矩阵记忆，只用长卷积 + 门控"也能在 2K 长度达到 Transformer 质量、64K 上快 100×。即"矩阵记忆"不是长上下文的必需品。
- 裁决：**未覆盖**（刻意规避状态，属另一条路线）。

### 12. Gated Linear Attention Transformers with Hardware-Efficient Training
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim / 2023(arXiv) & ICML 2024 / https://arxiv.org/abs/2312.06635 （HTML v6 实拉）
- 记忆状态是什么：**KV 矩阵** S_t。
- 读操作精确公式：o_t = S_t q_t（递归形式）。
- 写操作精确公式：**Eq.(3)** S_t = (α_t^T 1) ⊙ S_{t−1} + k_t^T v_t = **Diag(α_t) S_{t−1} + k_t^T v_t**，α_t = σ(x_t W_{α1} W_{α2})^{1/τ}，逐 key 通道。
- 是否存在 M ← M + 内容寻址更新：**否**（加法写入；衰减是逐通道对角）。
- alpha_t 究竟是什么：**学出的门**（数据依赖、逐通道、低秩参数化）。原文 Table 1 与 Mamba/Mamba-2/mLSTM/Gated Retention/DFW/GateLoop/HGRN-2/RWKV-6 并列给出参数化对照。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**无**。但 Table 1 是重要的"族谱证据"：所有现代 gated 模型都可写成 S_t = G_t ⊙ S_{t−1} + k_t^T v_t 这一 Hadamard 递推。
- 用于：仅语言建模 + 长序列/长度外推 + recall-intensive 任务。
- 与候选结构的异同：衰减项：**与候选 M^- = exp(−ΛΔ)⊙M 结构完全一致**（Λ 逐通道对角 ↔ Diag(α_t) 逐通道；exp(−ΛΔ) ↔ α_t），唯一差别是 α_t 由输入算出、且 Δ≡1。**因为对角矩阵可交换，累积衰减 ∏Diag(α_t)=Diag(∏α_t) 是精确乘半群** —— 候选想要的半群性质在这里已经成立。写入项：加外积，**无残差**。读操作：同。双状态：无。Δt 落点：**缺**。
- 裁决：**部分覆盖（P1 的逐通道对角 + P0 + P3）**。这是候选结构"Λ 逐通道"最直接的在世证据。证据：Eq.(3)。

### 13. Forgetting Transformer: Softmax Attention with a Forget Gate
- 作者 / 年份 / 会议或期刊 / 链接：Zhixuan Lin, Evgenii Nikishin, Xu Owen He, Aaron Courville / 2025 / arXiv 2503.02130（ICLR 2025）/ https://arxiv.org/abs/2503.02130 （HTML v2 实拉）
- 记忆状态是什么：**KV 矩阵**（线性注意力视图）S_t；但在实际模型里是 **softmax attention + 未归一化分数加权**，无显式矩阵状态（保持 FlashAttention 兼容）。
- 读操作精确公式：Eq.(7)(8)(9) 递归式 S_t = f_t S_{t−1} + v_t φ(k_t)^T，z_t = f_t z_{t−1} + φ(k_t)，o_t = S_t φ(q_t) / (z_t^T φ(q_t))。Eq.(11) 的 softmax 形式：o_i = Σ_j exp(q_i^T k_j + D_ij) v_j / Σ_j exp(q_i^T k_j + D_ij)，D_ij = log F_ij = Σ_{l=j+1}^{i} log f_l。
- 写操作精确公式：Eq.(7) S_t = f_t S_{t−1} + v_t φ(k_t)^T，f_t = σ(w_f^T x_t + b_f) ∈ R（标量）。
- 是否存在 M ← M + 内容寻址更新：**否**（加外积 + 标量门）。
- alpha_t 究竟是什么：**学出的门**（f_t，标量、逐 token、输入依赖），原文称 forget gate。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**无**。核心贡献是把衰减写成 log 域加性偏置 D_ij = Σ log f_l，从而在 softmax 里实现"不用位置编码的相对距离惩罚"。
- 用于：仅语言建模 + 长上下文。
- 与候选结构的异同：衰减项：**有，标量门**；关键价值是给出了"衰减 = log 域加性偏置"的恒等式，使**任何指数衰减都可视为对 attention 分数的位置偏置**。写入项：加外积，无残差。读操作：同 + 归一化。双状态：无。Δt 落点：D_ij = Σ_{l=j+1}^{i} log f_l **正是候选 exp(−ΛΔ_i) 在"逐 token 累加"下的形式**：若 Δ_l 恒为 1，则 D_ij = −Σλ_l，与候选同构。
- 裁决：**部分覆盖 P1（标量版）**，并给出候选衰减项的等价改写。证据：Eq.(7)(11)。

### 14. Hopfield Networks is All You Need
- 作者 / 年份 / 会议或期刊 / 链接：Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, et al. / 2020(arXiv) & ICLR 2021 / ICLR 2021 / https://arxiv.org/abs/2008.02217 （PDF v3 实拉）
- 记忆状态是什么：**外部 memory slots** X ∈ R^{N×d}（N 个已存模式，连续值向量），不是可学习的矩阵。状态向量 ξ ∈ R^d。
- 读操作精确公式：Eq.(3) ξ_new = f(ξ) = X p = **X softmax(β X^T ξ)**。每次检索是一次 softmax 加权平均。
- 写操作精确公式：**不存在在线写公式**。模式由外部给定或由一层投影写入（"new layer"），没有 M ← M + ... 的递推。
- 是否存在 M ← M + 内容寻址更新：**否**。检索是内容寻址的（softmax 按相似度），但**写入不是**；每次调用都要读全部 N 个 slot，容量代价在推理侧。
- alpha_t 究竟是什么：**attention 权重**（softmax(βX^Tξ)，β 是逆温度超参，非学习门）。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**（无时间轴概念，是静态联想检索）。
- 有无定理：**有，且本专题里最完整的容量理论**。Theorem 1：Eq.(3) 全局收敛（CCCP）；Theorem 2：能量与迭代序列的极限点分析；此外摘要声明"可存储指数级（随联想空间维度）多模式、一次更新即可检索、检索误差指数小"，容量正比于 c^{(d−1)/4}（c=1.37/3.15 在 Theorem 3 给定）。**全部是静态容量/收敛，不含时间语义**。
- 用于：免疫组库分类、pooling、prototype learning、作为 Transformer attention 的等价物。非视频/传感器。
- 与候选结构的异同：衰减项：无。写入项：无（且与候选相反：容量靠"存全部 pattern"而不是"改写矩阵"）。读操作：**软内容寻址读，与候选的 R(M,q)=Mq 是同一件事的两种极限**（后者是把 pattern 压进矩阵）。双状态：无。Δt 落点：无。
- 裁决：**未覆盖（但提供容量上界这一不可或缺的对照）**。它的容量结论正是"矩阵压缩式记忆"（候选结构）必须付出的代价：d×d 矩阵最多存 d 个正交关联，而 slot 记忆可存 c^{d/4} 个 —— 这是候选结构无法回避的权衡，且候选结构没有给出任何对应定理。

### 15. Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules
- 作者 / 年份 / 会议或期刊 / 链接：Kazuki Irie, Francesco Faccio, Jürgen Schmidhuber / 2022 / NeurIPS 2022 / https://arxiv.org/abs/2206.01649 （PDF 实拉）
- 记忆状态是什么：**KV 矩阵** W(t) ∈ R^{d_out×d_key}，被定义为时间 t 的连续函数。
- 读操作精确公式：Eq.(19)(20) q(T)=W_q x(T)，y(T) = W(T) q(T)（在读**最后时刻** T）。
- 写操作精确公式：**Eq.(18)** W(t) = W(t0) + ∫_{s=t0}^{t} F_θ(W(s), x(s)) ds；**Eq.(21)** 三种向量场，Delta 型：F_θ(W(s),x(s)) = σ(β(s)) ( v(s) − W(s)k(s) ) ⊗ k(s)，其中 [β(s),k(s),v(s)] = W_slow x(s)。NCDE 版 Eq.(23)(24) 用 dx 驱动。
- 是否存在 M ← M + 内容寻址更新：**是（连续时间版）**。Eq.(21) 的 Delta 型向量场字面上就是"残差 × 外积"的连续时间率。
- alpha_t 究竟是什么：**学出的门**（σ(β(s))，由慢网从连续输入 x(s) 生成，充当连续学习率）。**注意：这里 α 同时是"学习率"和"积分权重"**，因为 ∫F ds 里 α 与 ds 相乘。
- 是否使用物理时间 Δt：**是**。自变量 t 是真实时间，积分区间 [t0,T] 是实际经过的时间；模型直接作用于 irregularly sampled 时序。
- 若用 Δt，它影响：**状态转移（积分区间）+ 写入强度（∫ 的长度）**，全部通过"积分多长"体现，不是外挂一个 exp。
- 是否支持不规则采样、丢帧、重复帧：**是**（实验覆盖 regularly sampled、irregularly sampled with missing values、very long 三类）。重复帧（Δt=0）→ 积分区间为 0 → W 不变，天然正确。
- 有无定理：**无**（有 ODE 连接的理论背景，无本模型的定理）。
- 用于：**传感器流 / 时序分类**（Speech Commands、PhysioNet Sepsis 等）。
- 与候选结构的异同：衰减项：**不显式**（衰减只能作为向量场里的 −λW 项隐式出现）。写入项：delta 残差，同构。读操作：同。双状态：**无**（只有 W）。Δt 落点：**状态转移的积分区间**，这是最"物理正确"的落点，与候选"外挂 exp(−ΛΔ)"不同。
- 裁决：**部分覆盖，且是本专题中对"Δt 落点"最有理论分量的一篇**：它说明 Δt 应该进**积分**，而不是乘在状态上。
### 16. Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics
- 作者 / 年份 / 会议或期刊 / 链接：Jingdi Lei, Di Zhang, Soujanya Poria / 2025-12 / arXiv 2512.12602（v5）/ https://arxiv.org/abs/2512.12602 （HTML v5 实拉）
- 记忆状态是什么：**KV 矩阵** S（正文 Eq.(8) 用 L_t(S)=½‖S^T k_t − v_t‖²）。
- 读操作精确公式：o_t = S_t q_t（沿用 DeltaNet 读出）。
- 写操作精确公式：**Eq.(15)** S_t = ( I − ((1−e^{−β_t λ_t})/λ_t) k_t k_t^T ) S_{t−1} + ((1−e^{−β_t λ_t})/λ_t) k_t v_t^T，λ_t = k_t^T k_t。它是 ODE **Eq.(10)** dS(t)/dt = −A_t S(t) + b_t（A_t = k_t k_t^T，b_t = k_t v_t^T）在 ZOH 下的精确解 Eq.(11)。
- 是否存在 M ← M + 内容寻址更新：**是**（delta 的精确流形式）。
- alpha_t 究竟是什么：**学出的门，且在连续时间解释下就是积分步长**。原文 Eq.(9) 用 β_t 作梯度下降学习率；Eq.(11) 里 β_t 同时是 [0,β_t] 的积分上限 —— 即 **β_t 同时扮演"学习率"和"Δt"两个角色**。
- 是否使用物理时间 Δt：**否（但把 Δt 的角色显式化）**。β_t 是 token 间期的时长代理，仍按 token 一步一次；论文没有引入秒。
- 若用 Δt：理论上 Δt ≡ β_t。
- 是否支持不规则采样、丢帧、重复帧：**否**（但框架允许：若把 β_t 换成真实 Δt，Eq.(11) 直接成立，这正是危险之处）。
- 有无定理：**无形式定理**，但给出精确解（矩阵指数闭式）并证明 rank-1 幂等性 A_t^n = λ_t^{n−1} A_t 使指数可 O(d²) 计算。核心论断：标准 delta rule 只是该 ODE 的一阶 Euler 离散，**存在离散化误差**（大 ‖k‖ 或大 β 时 stiff）。
- 用于：仅语言建模 + MAD 合成基准 + 抗扰动鲁棒性。
- 与候选结构的异同：衰减项：**隐含在 e^{−β_t A_t} 里**，不是外挂的对角衰减。写入项：delta，同构。双状态：无。Δt 落点：**β_t 同时是步长与学习率**。
- 裁决：**部分覆盖，且是对候选结构最尖锐的技术反驳**（见 §4.2）。

### 17. Liquid Gated Attention
- 作者 / 年份 / 会议或期刊 / 链接：Yiheng Jiang, Yuanbo Xu, Yongjian Yang / 2026-08-31 / arXiv 2608.30695（v1）/ https://arxiv.org/abs/2608.30695 （HTML v1 实拉）
- 记忆状态是什么：**KV 矩阵** S_n ∈ R^{d×d}（由 1D LTC 的标量隐状态升维而来）。
- 读操作精确公式：**Eq.(15)** y_n = o_n ⊙ ( q_n S_n )，o_n = σ(x_n W_o + b_o) 逐通道输出门；q_n = x_n W_q + b_q（Eq.13）。
- 写操作精确公式：**Eq.(12)** S_n = g_n S_{n−1} + (1 − g_n) k_n^T v_n，其中 **Eq.(10)** g_n = exp(−δ_n x̄_n)，**δ_n = t_n − t_{n−1} 是真实时间间隔**，x̄_n = μ x̃_{n−1} + (1−μ)x̃_n 是可学习端点插值（Eq.9），x̃ = σ(x w_f + b_f)（Eq.8）。
- 是否存在 M ← M + 内容寻址更新：**否**（凸插值写入，无残差、无 erasing）。
- alpha_t 究竟是什么：**学出的门 × 外部真实 Δt**。α_n 没有单独出现；写入权重 (1−g_n) 与 g_n = exp(−δ_n x̄_n) 绑定，**衰减与写入共用同一扇门**（凸组合约束）。
- 是否使用物理时间 Δt：**是，且是唯一显式把秒级 δ_n 乘进指数的一篇**。
- 若用 Δt，它影响：**衰减 + 写入强度（同时，因为凸组合）**；不影响 key/value；不影响读。
- 是否支持不规则采样、丢帧、重复帧：**是**。δ_n 任意正；由 Eq.(10)(12) 可直接推出 δ_n = 0 ⇒ g_n = 1 ⇒ S_n = S_{n−1}（重复帧被精确忽略）；δ_n 很大 ⇒ g_n → 0 ⇒ 旧状态被完全覆盖。[此推论由公式直接得出，论文未逐字写出]
- 有无定理：**无采样/容量定理**。附录有"A Learnable Integral Approximation 的误差分析"（Appendix B）与"数值界与梯度稳定性"（Appendix C），**仅标题级核实**；正文无 Theorem 环境。
- 用于：**时序 / 传感器流**（UEA 多变量时序库、BIDMC 生理数据、人体活动、摆跟踪、USHCN、PhysioNet）。
- 与候选结构的异同：衰减项：**命中结构、未命中"逐通道"**——g_n 由标量 x̄_n 决定，全通道共享衰减率，不是逐通道对角 Λ。用真实 δ_n。写入项：**未命中**——凸插值而非残差内容寻址；不区分"变化内容"与"不变内容"，只在总量上分配给新帧。读操作：Mq + 输出门，基本同构。双状态：**无**（S 是唯一状态）。Δt 落点：**衰减 + 写入强度**。
- 裁决：**部分覆盖，但它是候选结构最接近的对撞工作**。它已占据「矩阵快权重 + 真实 Δt 指数衰减 + 线性注意力并行形式」这一格；候选结构剩下的只有「残差写入」与「独立 h」两点未被占。证据：Eq.(10)(12)(15)。

### 18. Titans: Learning to Memorize at Test Time
- 作者 / 年份 / 会议或期刊 / 链接：Ali Behrouz, Peilin Zhong, Vahab Mirrokni / 2024-12(arXiv) & NeurIPS 2025 / https://arxiv.org/abs/2501.00663 （HTML v1 实拉）
- 记忆状态是什么：**参数 / 梯度**。M 是深层 MLP 的权重，等价说法是"matrix-valued memory M = W ∈ R^{d_in×d_in}"。
- 读操作精确公式：y_t = M_{t−1}(k_t)（前向调用记忆网络）；ℓ(M_{t−1};x_t) = ‖M_{t−1}(k_t) − v_t‖²（Eq.12，即预测误差）。
- 写操作精确公式：**Eq.(13)** M_t = (1 − α_t) M_{t−1} + S_t；**Eq.(14)** S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)。k_t = x_t W_K，v_t = x_t W_V（Eq.11）。
- 是否存在 M ← M + 内容寻址更新：**是**（用梯度 ∇ℓ 作写入方向；梯度天然 content-addressed：它指向当前预测错的地方）。
- alpha_t 究竟是什么：**学出的门（遗忘/权重衰减）× surprise（预测误差在写入量里）**。α_t ∈ [0,1] 是遗忘门；写入强度由 ∇ℓ 的模决定，原文明确称其为 **surprise**："we measure the surprise of an input with the gradient of the neural network with respect to the input in associative memory loss"。θ_t 是步长、η_t 是 momentum。
- 是否使用物理时间 Δt：**否**（故 Δt 落点不适用）。
- 是否支持不规则采样、丢帧、重复帧：**否**（输入按 token 序列处理）。
- 有无定理：**有，但是表达力定理**。**Theorem 4.1**："Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to TC0, Titans are capable of solving problems beyond TC0"。
- 用于：仅语言建模 + 长上下文 + **时序预测** + DNA。
- 与候选结构的异同：衰减项：有，标量门 (1−α_t)，非逐通道。写入项：**比候选更强**——写入方向是"预测误差的梯度"而非固定 v k^T，是"只改需要改的部分"的最激进版本。读操作：非线性 M(k) 而非 Mq。双状态：**严格说是三状态**（Core attention 短期 + 长期记忆 M + persistent memory），但 M 与 attention 是并列而非耦合递推。Δt 落点：无。
- 裁决：**部分覆盖**。它是"surprise 驱动写入强度"这支的最强代表，**直接威胁候选结构"衰减与写入由同一时间量控制"的直觉**：Titans 的经验结论是写入强度应由**内容的新奇度**决定，而非时间间隔。

### 19. Fast weight programming and linear transformers: from machine learning to neurobiology
- 作者 / 年份 / 会议或期刊 / 链接：Kazuki Irie, Samuel J. Gershman / 2025 / TMLR 2025（12/2025），OpenReview id TDG8EkNmQR / https://arxiv.org/abs/2508.08435 （HTML v4 实拉）
- 记忆状态是什么：**综述**。把本族统一为"2D 矩阵隐状态的 RNN"，维度 d_out × d_key。
- 读操作精确公式：综述形式 Eq.(14) y_t = W_t φ(q_t)（写后读）。
- 写操作精确公式：Eq.(13) W_t = W_{t−1} + v_t ⊗ φ(k_t)（vanilla）；Eq.(24) DeltaNet：W_t = W_{t−1} + ψ(β_t)(v_t − W_{t−1}φ(k_t)) ⊗ φ(k_t)，ψ 通常取 2×sigmoid。**Appendix A 给出统一推导**：所有衰减变体的局部损失 = 相似度损失 + L2 正则。RetNet 式 (56) ℒ_t(W) = −v_t^T W φ(k_t) + ((1−λ)/2)‖W‖_F² ⇒ Eq.(59) W_t = λ W_{t−1} + v_t ⊗ k_t；GLA 式 (60)(65) 的 L2 项按行缩放 ⇒ W_t = (a_t ⊗ 1) ⊙ W_{t−1} + v_t ⊗ k_t。
- 是否存在 M ← M + 内容寻址更新：**是**（综述确认这是族定义）。
- alpha_t 究竟是什么：**综述**：β_t 统一归为 "dynamic learning rate"；衰减 λ / a_t 统一归为 "L2 权重衰减"。**这是本专题最重要的一句结论：衰减不是启发式，它是局部损失里的 L2 正则项，其"正确形式"由正则项的加权方式决定。**
- 是否使用物理时间 Δt：**否**（全篇按 t 索引，未讨论 Δt）。（故 Δt 落点不适用）
- 是否支持不规则采样、丢帧、重复帧：**否**。
- 有无定理：**综述性质**，汇总他人结论（容量 Seq.3.6、表达力 TC0 层级）；自身无新定理。
- 用于：综述（机器学习 + 神经科学）。
- 与候选结构的异同：它把本专题 18 篇全部装进一个框：**矩阵状态 + 局部损失 → 更新规则**。候选结构的衰减项在这个框里必须回答"你对应哪个正则项"；候选结构目前答不出来——因为 exp(−ΛΔ_i) 在 Δ_i 非常数时不是任何固定局部正则项的一步梯度解。
- 裁决：**这是判断"是否已有人做过"的最终依据**：族框架已经固化出版，候选结构必须证明自己是族框架内某个**新局部损失**的解，否则只是换参数化。

## 3. 横向对照表

| # | 论文 | 状态 | 衰减 | 衰减依赖 | 写入 | 残差写 | 双状态 | 真实 Δt | 定理 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Schmidhuber 1991/92 | 快网权重 | 无 | — | Δw = s_a⊗s_b | 否 | 是(慢/快) | 否 | 无 |
| 2 | Schlag 2021 | d_out×d_key | 无 | — | +β(v−Wk)⊗φ(k) | **是** | 否 | 否 | 容量论证 |
| 3 | Katharopoulos 2020 | d×d | 无 | — | +v⊗φ(k) | 否 | 否 | 否 | 无 |
| 4 | Performer 2021 | 隐式 | 无 | — | 隐含 | 否 | 否 | 否 | 核近似误差×3 |
| 5 | Yang 2024 DeltaNet | d×d | 无 | — | −β(Sk−v)kᵀ | **是** | 否 | 否 | 无 |
| 6 | Gated DeltaNet 2025 | d×d | α_t 标量 | 输入 | α(I−βkkᵀ)S+βvkᵀ | **是** | 否 | 否 | Table1 统一 |
| 7 | RetNet 2023 | d_k×d_v | γ 常数 | head 常数 | +KᵀV | 否 | 否 | 否 | 无 |
| 8 | RWKV-4 2023 | 向量 | w 时不变 | token 距离 | 向量加权 | 否 | 否 | 否 | 无 |
| 9 | RWKV-5/6 2024 | (D/h)² | diag(w)/α_t | 常数/输入 | +kᵀv | 否 | 否 | 否 | 无(收缩性) |
| 10 | RWKV-7 2025 | d×d | w_t 向量 | 输入 | diag(w)−kᵀk diag(a) | **是** | 否 | 否 | TC0 外 |
| 11 | Hyena 2023 | **无状态** | — | — | 无 | — | 否 | 否 | 无 |
| 12 | GLA 2024 | d_k×d_v | **Diag(α_t)** | 输入 | +kᵀv | 否 | 否 | 否 | 无 |
| 13 | FoX 2025 | d×d | f_t 标量 | 输入 | +vφ(k)ᵀ | 否 | 否 | 否 | 无 |
| 14 | Hopfield 2021 | slot X | 无 | — | 无 | — | 否 | 否 | 容量/收敛×3 |
| 15 | Irie 2022 CT-FWP | d_out×d_key | 隐式(向量场) | **积分区间** | ∫F_θ ds | **是** | 否 | **是** | 无 |
| 16 | EFLA 2025 | d×d | e^{−βA} | β=步长 | 精确 delta | **是** | 否 | 部分 | 无 |
| 17 | LGA 2026 | d×d | **exp(−δ_n x̄_n)** | **真实秒** | (1−g)kᵀv | 否 | 否 | **是** | 无(附录误差) |
| 18 | Titans 2025 | MLP 参数 | (1−α_t) | 输入 | −θ∇ℓ | **是**(梯度) | 三状态 | 否 | TC0 外 |
| 19 | Primer 2025 | 综述 | 统一为 L2 | — | 族框架 | — | — | 否 | — |

## 4. 反对结论与反例

**本节只针对"用户的直觉"，不针对候选人。凡与下面冲突的主张，都需要新证据。**

### 4.1 反例一：把"帧数衰减"换成"真实 Δt 衰减"在数学上不是新机制，除非证明采样不变性

对角衰减的乘积是精确半群：∏_i Diag(α_i) = Diag(∏_i α_i)。若 α 取 exp(−λ) 并把 λ 按单位时间标定，则重采样（一帧拆两帧、两帧合一帧）只改变乘积的项数分解、不改变乘积值。所以候选的 exp(−Λ Δ_i) 与 GLA 的 Diag(α_t) 是**同一个数学对象在两个时间标度下的写法**，差别只是 λ 的单位。

"用真实 Δt"的主张只有在两种情形下才有内容：
- (a) 输入确实非均匀采样，且模型行为必须在重采样下**逐点不变**（采样率不变性定理）；或
- (b) Δt 被用来**同时**决定衰减与写入量，从而改变模型能表达的函数类。

本专题 19 篇里：**没有任何一篇给出采样率不变性定理**。(a) 完全空白；LGA（#17）与 Irie 2022（#15）只覆盖了"非均匀采样可以喂进去"，没有证明"重采样后输出不变"。所以：若候选只做"Δt 乘进指数"，新颖性 ≈ 0（LGA 已做）；若要做 (a)，那是**开放的数学工作**，不是网络结构工作。

### 4.2 反例二：衰减因子与写入强度不是两个独立旋钮——写成两个独立因子在连续时间下非物理

EFLA（#16）式 (10)(11)：delta rule 的状态转移是 ODE  dS/dt = −A_t S + b_t  在 ZOH 下的**精确解**

    S_t = e^{−β_t A_t} S_{t−1} + ∫_0^{β_t} e^{−(β_t−τ)A_t} b_t dτ

其中 (i) **β_t 同时是学习率与积分步长**；(ii) 衰减与写入**共享同一个算子 e^{−β_t A_t}**，不可分离。候选结构写的是  M_i = exp(−Λ Δ_i) ⊙ M_{i−1} + α_i v_i k_i^T，即把衰减（Δ_i）与写入（α_i）当成两个独立可调因子。这在连续时间解释下**不对应任何单一 ODE 的解**：若衰减按 Δ 走而写入按 α 走，则当二者比例随采样率变化时，系统的连续时间极限不存在。Irie 2022（#15）的正确做法是让两者都进积分；Gated DeltaNet（#6）与 RWKV-7（#10）的正确做法是把它们合成**一个**转移矩阵。**这是对候选结构最硬的技术反驳**：(P1) 与 (P2) 形式上是相加的两个算子，但物理上应是一个半群生成元的两部分。若坚持分离写法，必须证明"Δ 随采样率变化时 α 有相应重标定律"——而这正是采样不变性要证的东西，循环论证。

### 4.3 反例三：实验证据显示"衰减"本身损害记忆，收益来自门控与残差写

Gated DeltaNet（#6）§3.2 的单针检索（S-NIAH）消融，原文小节标题就是结论：
- "**Decay hurts memory retention**"（衰减损害记忆保持）；"Gating facilitates filtering"（门控起过滤作用）；"Delta rule helps memorization"（残差写帮助记忆）

同文数值（S-NIAH-1，8K）：DeltaNet 98.8 / Mamba2（有衰减无残差）30.4 / Gated DeltaNet 91.8。**加衰减的 Mamba2 远低于不加衰减的 DeltaNet**。这说明：
- 把"记忆衰减"当正面机制来设计，**方向上可能有误**；
- 真正带来容量的是**残差写（内容寻址覆盖）**；衰减主要是数值稳定/过滤手段。

若候选结构的核心卖点是"衰减由 Δt 控制"，那它优化的是一个**已知收益为负或近零**的部件。

### 4.4 反例四：候选结构的双状态 h 会破坏整个系统的时间一致性

候选的 M 按真实 Δt 演化，而 h_i = H(h_{i−1}, r_i, z_i) 只按序号 i 演化。后果：
- 同一物理输入流，30 Hz 采 vs 60 Hz 采：**M 的轨迹近似相同（若 α 重标定），h 的轨迹完全不同**（递推步数翻倍）；
- 于是 q_i = Q(z_i, h_i) 依赖采样率，r_i = R(M_i, q_i) 跟着依赖采样率 —— **整个系统没有连续时间极限**；
- 半群性质在 M 上成立，在 (M,h) 上不成立，系统层面时间一致性失效。

本专题唯一的"双状态"先例是 Schmidhuber 1991（#1）：慢网 S 是**参数**（不随时间演化），不是另一条隐状态。**没有任何一篇做过"矩阵状态按 Δt 演化 + 独立向量状态按序号演化"** —— 没人做很可能不是疏忽，而是这个组合在时间语义上不自洽。若保留 h，必须让 h 也按 Δt 演化（也进积分、也带衰减），或者证明 h 可被吸收进 M。

### 4.5 反例五：容量上界与"衰减代替遗忘"的直觉相反

- Schlag 2021（#2）§4.1：矩阵检索要求键两两正交，d_dot 维空间最多 d_dot 个正交关联，超出即干扰。
- Hopfield（#14）：slot 记忆容量随维度**指数**增长（c^{(d−1)/4}）。

结论：**矩阵压缩记忆的容量是线性的，slot 记忆是指数的**；衰减只减少有效关联数，**不提高容量**。"用衰减腾出空间"不是容量机制，只是干扰管理。残差写之所以提高容量，是因为它把冲突方向**投影掉**（Householder 保持谱），不是靠衰减。这在 #5（Householder 参数化）与 #10（rank-1 修正转移矩阵）里都是显式的。

### 4.6 反例六：存在"根本不需要矩阵记忆"的强基线

Hyena（#11）：纯隐式长卷积 + 逐元素门控，无状态、无矩阵记忆，2K 长度达到 Transformer 质量、64K 快 100×。它证明"矩阵记忆"是**一种选择**，不是长上下文的必要条件。候选的动机（"需要可寻址关联记忆"）只在**需要精确 in-context 检索**时成立；对"流式视觉决策"这类任务，是否真需要 d×d 矩阵记忆，本专题**没有任何证据**支持。而 FoX（#13）反证：在 softmax 上加一个**标量** forget gate 就能拿到长上下文收益，且在 needle-in-haystack 上仍优于 Mamba-2 / HGRN2 / DeltaNet。

### 4.7 反例七：Δt 的"正确落点"已有两个互相矛盾的答案，候选选了第三种（外挂指数）

- Irie 2022（#15）：Δt 进**积分区间**（状态转移由 ∫ 决定）。
- EFLA（#16）：Δt 与学习率是**同一个量**（β_t 既是步长又是 lr）。
- LGA（#17）：Δt 进**指数**（g = exp(−δ x̄)），但**同时**用它决定写入（凸组合）。

候选选的是 LGA 式"进指数"，却**丢掉 LGA 的凸组合约束**（(1−g) 与 g 互补）。丢掉约束后，写入总量不再随 Δt 变化，于是"长时间没观测后第一帧应有更大影响"这一物理直觉**在候选结构里反而不成立**（除非 α 另有 Δ 依赖）。这是候选结构相对 LGA **更弱**的一处。

### 4.8 反例八：Family 框架已成文出版，"新结构"必须先回答"对应哪个局部损失"

Irie & Gershman 2025（#19）Appendix A 已经把 RetNet / GLA / Mamba2 / xLSTM 的衰减全部推导成**局部损失里的 L2 正则项**。这意味着：在本族里提出一个新的衰减/写入组合，学术上已经被规训为"你对应哪个局部损失"。候选结构若不能给出一个局部损失 L_t(M)，它就不是本族的一员，而是一个**未定义的前向规则** —— 这在 2026 年的审稿标准下是致命的。

## 5. 与候选结构逐字对应的工作清单（诚实排序）

| 候选命题 | 已被谁逐字做过 | 证据 |
|---|---|---|
| P0 矩阵记忆状态 | #1 #2 #3 #5 #6 #7 #9 #10 #12 #13 #15 #16 #17 #18 | 全部 d×d 或 d_out×d_key |
| P1 逐通道对角衰减 | **#12 GLA（Diag(α_t)）、#9 RWKV-5（diag(w)）** | GLA Eq.(3)；Eagle Eq.(9) |
| P1 真实 Δt 控制衰减 | **#17 LGA**（唯一显式秒级） | LGA Eq.(10)(12) |
| P2 内容寻址写入 | **#2 #5 #6 #10 #15 #16 #18** | DeltaNet 家族 Eq.(24)/Eq.(10) 等 |
| P2 "只写变化部分" | **#2 #5 #6 #10 #16**（残差 = 只写预测误差） | Eq.(24) 的 (v − Wk) |
| P3 写后读 M q | #2 #3 #5 #6 #7 #10 #12 #13 #17 | 全部 o_t = S_t q_t |
| P4 双状态耦合 | **无先例**（#1 是慢参数+快状态，非双隐状态；#18 是三状态但并列不耦合） | — |
| P5 半群/采样率不变定理 | **无先例**（半群性质在 #12 中由对角性平凡成立，但无人陈述并证明采样不变） | — |
| P1+P2 合成单一转移矩阵 | **#10 RWKV-7** | diag(w_t) − k_t k_tᵀ diag(a_t) |

**结论**：候选结构的 6 个命题里，4 个已被逐字覆盖；1 个（双状态）无先例但**无先例的原因可疑**（见 4.4）；1 个（半群/采样不变定理）是真空白，但它是数学工作而非结构工作。

**"有没有发现与候选结构逐字对应的工作"——没有完全逐字的。但 LGA（#17）+ Gated DeltaNet（#6）两篇合起来已覆盖其 90%；LGA 是 2026 年 8 月的新工作，仍在活跃引用期，且已在时序/传感器域落地。**

**最接近的 3 个先例（含公式要点）**
1. **Liquid Gated Attention（#17）** — S_n = g_n S_{n−1} + (1−g_n)k_n^T v_n，g_n = exp(−δ_n x̄_n)，δ_n = t_n − t_{n−1} 真实秒。唯一同时具备「矩阵快权重 + 真实 Δt + 并行形式」的工作。
2. **Gated DeltaNet（#6）** — S_t = S_{t−1}(α_t(I − β_t k_t k_t^T)) + β_t v_t k_t^T。唯一把「衰减门 + 残差写」合成一个式子的工作，α_t 是学出的门而非 Δt。
3. **DeltaNet（#2/#5）** — S_t = S_{t−1} − β_t(S_{t−1}k_t − v_t)k_t^T，β_t = σ(W_β x_t)。"只写变化部分"的原始出处，且带 1-step SGD on ‖Sk−v‖² 的严格解释。

**最致命的 3 条反对证据**
1. **§4.2 EFLA Eq.(10)(11)**：δ 衰减与写入在连续时间下共享同一算子 e^{−βA}，β 同时是步长与学习率。候选把二者写成两个独立因子**不对应任何 ODE**。
2. **§4.3 Gated DeltaNet §3.2**：原文小节标题 "Decay hurts memory retention"。衰减本身**损害**记忆保持，收益来自残差写与门控。
3. **§4.4 双状态 h 破坏时间一致性**：M 按 Δt、h 按序号 ⇒ 系统无连续时间极限；本族无任何先例，原因很可能是它不自洽。

## 6. 核实方法与未核实清单

**核实手段**：arXiv API（export.arxiv.org/api/query）逐条核 ID / 标题 / 作者 / 首次提交日期，18 个 arXiv ID 全部命中返回；PDF 用 tavily_extract 实拉全文并抽公式（Schlag 2021、Katharopoulos 2020、Performer、RetNet、RWKV-4/5、RWKV-7、Hyena、Hopfield、Irie 2022、Schmidhuber 1991 TR）；2023 年后论文用 arXiv HTML 抽 MathML alt-text（Yang 2024、Gated DeltaNet、GLA、FoX、Titans、EFLA、LGA、Primer）；Schmidhuber 1992 期刊元数据经 ACM DL + Wikidata 双向核实（Neural Computation 4(1):131–139，DOI 10.1162/neco.1992.4.1.131）。

**必须声明的核实缺陷（共 6 条）**

1. **Schmidhuber 1992 期刊版 PDF 文本不可提取**：sferics.idsia.ch 与 gwern.net 两处副本均为 CID 编码，抽取结果是乱码（同为 33195 字节的乱码）。本文件引用的公式来自 **1991 技术报告 OCR 版**（FKI-147-91，Eq.1/4/5 逐字），该报告与 1992 期刊版为同一工作。**期刊版正文未逐字核实。**
2. **Performer 的 M 递推式非原文**：原论文只给掩码矩阵形式，标 [仅由定义推出]。
3. **LGA 附录 B / C 的误差分析与数值界仅标题级核实**，未读正文。
4. **Hedgehog 篇目放弃**：arXiv API 查标题含 "Hedgehog" 返回 8 条全为物理/数学论文，不存在名为 "Hedgehog & Feathers" 的 arXiv 条目（原以为的 2209.15426 实为 Volovik 的物理学论文 "Dimensionless physics: Planck constant as an element of Minkowski metric"，**并未使用**）。检索中同名的 "The Hedgehog & the Porcupine"（arXiv 2402.04347）是另一工作。**已放弃该篇，用 GLA + FoX 满足"三选二"要求**，不编造条目。
5. **EFLA 标题别名不一致**：HuggingFace papers 页面标题为 "Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics"，而 arXiv 官方标题为 "**Exact Flow** Linear Attention: Exact Solution from Continuous-Time Dynamics"。本文件以 arXiv 标题为准。
6. **RWKV-6 参数化引自 GLA Table 1 的转述**（α_t = exp(−exp(x_t W_α))），未逐字读 Eagle/Finch 原文该式；但 Finch 的 "data-dependent decay" 性质已在 Eagle 论文正文中核实（"Finch further applies data-dependence to the decay schedule and token-shift"）。

**另需说明的两处勘误（检索过程中的自查）**
- 初次按记忆取 Hyena 的 arXiv 号为 2306.15794，实际该号是 **HyenaDNA**（Nguyen et al.）；Hyena Hierarchy 正确号为 **2302.10866**，已改用后者并实拉核实。
- 初次按记忆取 Hedgehog 的 arXiv 号为 2209.15426，实际是物理学论文，见上条。

**未覆盖但相关、建议后续补的 4 篇**：Mamba-2（Dao & Gu 2024，SSD 框架是 GLA/Gated DeltaNet 的直接祖先）、Longhorn（Liu et al. 2024，隐式在线学习闭式解）、Test-Time Training（Sun et al. 2024）、xLSTM mLSTM（Beck et al. 2024）。这四篇都不改变本文结论，但 Mamba-2 与 Longhorn 是 §4.2 论证的直接支撑。


# 02 · 测试时学习与可变权重 / 神经记忆（文献核对）

> 检索日期 2026-11-03。所有标题 / 作者 / 年份 / 链接均实拉核实；公式抄自 arXiv HTML（MathML annotation 中的 LaTeX）或官方 PDF，未从记忆补写。
> 核查方式：对每篇正文全文做关键词核查（`irregular | timestamp | elapsed | continuous-time`），用于判定"是否真的有物理时间语义"，命中数为 0 的会在文中写明。
> 编号 1–14，共 14 篇。文末有横向对照表、独立一节「反对结论与反例」、以及逐项覆盖裁决。

被检验的候选结构（本题）：

```
M_i- = exp(-Lambda * Delta_i) ⊙ M_{i-1}      # 矩阵快权重关联记忆，按真实时间间隔衰减，Lambda 逐通道对角
q_i = Q(z_i,h_i);  k_i = K(z_i,h_i);  v_i = V(z_i,h_i)
M_i = M_i- + alpha_i * v_i k_i^T             # 内容寻址写入
r_i = R(M_i, q_i)                            # 读
h_i = H(h_{i-1}, r_i, z_i)                   # 另一路隐状态
```

用户四条主张，后文逐条打分：(a) 用 attention / 快权重维护可寻址关联记忆；(b) 不变内容不反复重写；(c) 变化内容经 key 找到相关记忆并更新；(d) 衰减与写入由真实经过时间 Δt 控制，而不是按帧数。

---

## 1. 逐篇核对

### 1. Learning to (Learn at Test Time): RNNs with Expressive Hidden States
- 作者 / 年份 / 会议或期刊 / 链接：Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin / 2024（v1 2024-07-05，v4 2025-08-31）/ ICML 2025（PMLR v267, sun25h）/ https://arxiv.org/abs/2407.04620 ; https://proceedings.mlr.press/v267/sun25h.html
- 记忆状态是什么：**参数**。隐状态 s_t 就是一个模型 f(·;W) 的权重 W_t（TTT-Linear 为线性模型权重，TTT-MLP 为两层 MLP 权重）。论文明确写 "the hidden state s_t is now equivalent to W_t, the weights of a model f"（§2.1）。
- 读操作精确公式：z_t = f(θ_Q x_t ; W_t)（Eq 5）；直观含义是"用更新后的权重对当前 token 做预测"。
- 写操作精确公式：W_t = W_{t-1} − η ∇ℓ(W_{t-1}; x_t)（Eq 2），其中 ℓ(W;x_t) = ‖ f(θ_K x_t ; W) − θ_V x_t ‖²（Eq 4，训练视图 θ_K x_t → 标签视图 θ_V x_t，多视图重建）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是**（形式上完全符合，W ← W − η∇ℓ）。但"α_t"不含内容寻址权重：写入方向是梯度，不是一个标量乘以外积。
- alpha_t 究竟是什么：**学出的步长 η**（外层循环学的标量学习率）+ **预测误差**（写入方向来自重建误差的梯度）。不是门、不是 attention 权重、不是 surprise 的显式度量。
- 是否使用物理时间 Δt：**否**。全文关键词核查 irregular/timestamp/elapsed/continuous-time 命中 0（唯一命中在参考文献里）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：论文不涉及。机制上：丢帧 = 少做几步梯度；重复帧 = 反复做同一步梯度（会持续漂移）；**没有任何采样率归一化**，因此不具采样率不变性。
- 有无定理：**无**稳定性/容量/采样率定理。只有等价性陈述：§2.6 "TTT layer with a linear model and batch GD is equivalent to linear attention"；以及 §2.4 的双形式（dual form）加速，属实现层面。
- 用于：仅语言建模（125M–1.3B；8k 上下文，长上下文优于 Mamba）。
- 与候选结构的异同：衰减项 ❌ 完全没有；写入项 ⚠️ 是梯度步（含预测误差），但不是 α·v k^T 的外积叠加，也不区分 key；读操作 ⚠️ 前向 f(q;W) 相当于 M q，但不是"矩阵记忆读"；双状态 ❌（h 就是 W，没有独立递推隐状态）；Δt 落点 ❌。
- 裁决：**部分覆盖**——"测试时学习 = 每步用自监督损失改权重"这一骨架已被 TTT 完整做掉；但无衰减、无 Δt、无双状态。

### 2. Titans: Learning to Memorize at Test Time
- 作者 / 年份 / 会议或期刊 / 链接：Ali Behrouz, Peilin Zhong, Vahab Mirrokni / 2024-12-31（NeurIPS 2025）/ NeurIPS 2025 / https://arxiv.org/abs/2501.00663 ; https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html
- 记忆状态是什么：**参数**（深度 MLP 的权重 M，L_M ≥ 1 层；线性版 M=W ∈ R^{d_in×d_in}）。另有 persistent memory P=[p_1…p_Np]，是**与数据无关的可学参数**（Eq 区 §3.3）。
- 读操作精确公式：y_t = M*(q_t)，q_t = x_t W_Q（Eq 15）；M* 表示"不更新权重的前向"。MLP 记忆时就是一次前向。
- 写操作精确公式（Eq 13–14，这是全篇最接近候选的地方）：
  - M_t = (1 − α_t) M_{t-1} + S_t
  - S_t = η_t S_{t-1} − θ_t ∇ℓ(M_{t-1}; x_t)
  - ℓ(M_{t-1}; x_t) = ‖ M_{t-1}(k_t) − v_t ‖²₂，k_t = x_t W_K，v_t = x_t W_V（Eq 11–12）
  - 闭式展开（Eq 16）：M_t = β_t M_0 − Σ_{i=1..t} θ_i (β_t/β_i) ∇ℓ(M_{t'}; x_i)，β_i = Π_{j=1..i}(1−α_j)
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是，逐字同构**。M_t = diag(1−α_t) M_{t-1} + S_t（Eq 32），且 S_t 的梯度项 = −θ_t(M_{t-1}k k^T − v k^T)，正是内容寻址的误差修正写。
- alpha_t 究竟是什么：**学出的门**。α_t ∈ [0,1] 是遗忘门（式 32 写作 diag(1−α_t)，可逐通道），"input-dependent（是 x_t 的函数）"；θ_t 是学出的**自适应学习率**（写入幅度）；η_t 是**动量衰减**（"past surprise" 的衰减）。**surprise 本身 = 梯度**（论文原话："a simple definition of surprise for a model can be its gradient with respect to the input"，但式 8 实际是对记忆参数求梯度——这句两处表述不完全一致，原文如此）。
- 是否使用物理时间 Δt：**否**。全文 irregular/timestamp/elapsed/continuous-time 命中 0；衰减按 token 序号 t。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不涉及。丢帧 = 该 token 不写；重复帧 = 重复写（且动量 S 会累积"past surprise"）。
- 有无定理：**有 Theorem 4.1**（表达力：Titans 能解 TC⁰ 之外的问题，强于 Transformer / 对角线性 RNN / DeltaNet）；**无**衰减稳定性、容量、采样率、误差界定理。
- 用于：语言建模、常识推理、needle-in-haystack（>2M 上下文）、DNA、时间序列预测。
- 与候选结构的异同：衰减项 ✅ 有，但是 (1−α_t)、逐 token 的数据依赖门，**不是 exp(−ΛΔt)**；写入项 ✅ 内容寻址误差修正（甚至更强：带动量）；读操作 ✅ M(q)，与 R(M,q) 同形；双状态 ⚠️ 有（短窗口 attention 的 core + 长期记忆），但 h 是 attention 上下文而非递推隐状态；Δt 落点 ❌ 完全缺失。
- 裁决：**部分覆盖**——除 Δt 与逐通道指数形式外，候选的写/读/衰减三件套已被 Titans 覆盖。

### 3. Metalearned Neural Memory
- 作者 / 年份 / 会议或期刊 / 链接：Tsendsuren Munkhdalai, Alessandro Sordoni, Tong Wang, Adam Trischler / 2019 / NeurIPS 2019 / https://arxiv.org/abs/1907.09720
- 记忆状态是什么：**参数**。记忆是一个"快速可变的函数" f_φ：DNN 的参数 φ（读入 key 向量、输出 value 向量）；另有一个 RNN controller 产生读写向量。
- 读操作精确公式：v̂_t = f_{φ_{t-1}}(k^r_t)（把读入 key 推过记忆函数；§3.2 原话 "pushing an input (the key vector) through the function to produce an output"）。
- 写操作精确公式：φ_t ← φ_{t-1} − β_t ∇_{φ_{t-1}} L^up_t，L^up_t = (1/H) Σ_i ‖ f_{φ_{t-1}}(k^w_{t,i}) − v^w_{t,i} ‖²₂（Eq 2–3）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是**（参数空间上的内容寻址一步更新），另有 §3.4 的"学出的局部更新规则"变体。
- alpha_t 究竟是什么：**学出的门**。β_t 是 controller 输出的"update rate"，论文自述"diminishing the update rate … achieving an effect similar to a gated memory update"（§3.3）。
- 是否使用物理时间 Δt：**否**（irregular/timestamp/elapsed/continuous-time 命中 0）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不支持、不涉及。
- 有无定理：**无**。
- 用于：算法任务、合成 QA、强化学习。
- 与候选结构的异同：衰减项 ❌；写入项 ⚠️ 内容寻址（对 key-value 对做一步梯度，等价于 delta 类修正），但不是外积叠加；读操作 ✅ 函数前向 ≈ M q；双状态 ✅（RNN controller h + 神经记忆 φ）；Δt ❌。
- 裁决：**部分覆盖**——"外部神经记忆 = 可写可读的参数函数"最早形态之一，但没有衰减也没有时间语义。

### 4. Gated Delta Networks: Improving Mamba2 with Delta Rule
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Jan Kautz, Ali Hatamizadeh / 2024-12（v3 2025-03）/ ICLR 2025 / https://arxiv.org/abs/2412.06464
- 记忆状态是什么：**KV 矩阵**。S ∈ R^{d_k×d_v}（fast weight 状态），逐 head。
- 读操作精确公式：o_t = S_t q_t（与 Mamba2 同：S_t = α_t S_{t-1} + v_t k_t^⊤，o_t = S_t q_t）。
- 写操作精确公式（gated delta rule，§2）：
  - S_t = S_{t-1}( α_t (I − β_t k_t k_t^⊤) ) + β_t v_t k_t^⊤
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是，最接近候选**——(I − β k k^⊤) 就是对地址 k 的"先擦后写"，再加 α_t 全局衰减。
- alpha_t 究竟是什么：**学出的门**。α_t ∈ (0,1) 是数据依赖标量衰减（"controls state decay"）；β_t 是学出的写入步长（delta rule 的学习率）。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中；α_t 由输入 token 决定，与经过时间无关）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不涉及；丢帧 = 不写，重复帧 = 重复擦写。
- 有无定理：**有在线学习等价性分析**（表 1：状态更新是带 L2 正则的在线回归的闭式解；并给出与 DeltaNet / Mamba2 的统一形式），**无**稳定性/采样率定理。
- 用于：语言建模、长上下文检索（in-context associative recall）。
- 与候选结构的异同：衰减项 ✅ 但 per-step 标量（可逐 head，不可逐通道）；写入项 ✅ 内容寻址 + 误差修正（比候选的 +α v k^T 更正）；读操作 ✅ S q；双状态 ❌（输出直接是读，无独立 h）；Δt 落点 ❌。
- 裁决：**部分覆盖**——候选"矩阵记忆 + 衰减 + 内容寻址更新"三件套的工业级版本，唯一缺的就是 Δt。

### 5. Linear Transformers Are Secretly Fast Weight Programmers
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021（PMLR v139, schlag21a）/ https://arxiv.org/abs/2102.11174 ; https://proceedings.mlr.press/v139/schlag21a.html
- 记忆状态是什么：**KV 矩阵**（fast weight 矩阵 W^(i) ∈ R^{d_out×d_in}），每步动态生成。
- 读操作精确公式：y^(i) = W^(i) φ(q^(i))（Eq 25）。
- 写操作精确公式：
  - 纯 Hebb 版：W^(i) = σ(W^(i−1) + a^(i) ⊗ b^(i))（Eq 2，a=W_a x, b=W_b x）
  - delta 版（Eq 20–24）：v̄^(i) = W^(i−1) φ(k^(i))；β^(i) = σ(W_β x^(i))；v_new = β v + (1−β) v̄；
    W^(i) = W^(i−1) + v_new ⊗ φ(k) − v̄ ⊗ φ(k) = W^(i−1) + β^(i)(v^(i) − v̄^(i)) ⊗ φ(k^(i))
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是**（Eq 23/24 就是 M ← M + β(v − Mk) k^T 的雏形）。
- alpha_t 究竟是什么：**学出的门**。β^(i) = sigmoid(线性层) 的 "write strength"，决定新值替换旧值的比例；同时它也就是 delta rule 的动态学习率。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不涉及。
- 有无定理：**有等价性定理/形式化推导**（FWP 与线性化 self-attention 的等价性；附录 B 形式化），无稳定性/容量/采样率定理。
- 用于：语言建模、合成序列任务（这是"线性注意力=快权重"的奠基论文）。
- 与候选结构的异同：衰减项 ❌；写入项 ✅ **内容寻址 + 预测误差修正**，与候选的 +α v k^T 只差"先减去 Mk"；读操作 ✅ W φ(q)；双状态 ❌；Δt ❌。
- 裁决：**部分覆盖**——候选"变化内容经 key 找到相关记忆并更新"（主张 c）的最直接先例；但它是"误差修正写"，不是候选写的"加法写"。

### 6. Test-time regression: a unifying framework for designing sequence models with associative memory
- 作者 / 年份 / 会议或期刊 / 链接：Ke Alexander Wang, Jiaxin Shi, Emily B. Fox / 2025-01-21（v3 2025-05-02）/ arXiv 预印本（期刊/会议位未核实）/ https://arxiv.org/abs/2501.12352
- 记忆状态是什么：**KV 矩阵**（线性回归权重 M_t ∈ R^{d_v×d_k}）或任意函数类 m_t（"regressor"）；框架层：记忆 = 对关联做回归的解。
- 读操作精确公式：y_t = m_t(q_t)（线性情形 = M_t q_t）。
- 写操作精确公式：
  - 几何加权闭式：M_t ≈ Σ_{i=1..t} γ_i^{(t)} v_i k_i^⊤，γ_i^{(t)} = Π_{j=i+1..t} γ_j，γ_j ∈ [0,1]（式区 §3.1）
  - 递推：M_t = γ_t M_{t-1} + v_t k_t^⊤
  - 带 L2/weight-decay 的 delta 形式（Eq 26）：M_t = (1 − β_t λ_t) M_{t-1} + β_t (v_t − M_{t-1} k_t) k_t^⊤；并指出与 Gated DeltaNet 的式 (30) 可通过可逆映射互转
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是**，且给出了"为什么要有衰减"的严格推导（加权最小二乘 + 非平稳在线回归）。
- alpha_t 究竟是什么：**其他**——γ_i 是"关联的相对重要性权重"（recency 权重），β_t 是优化步长，λ_t 是正则强度；没有 surprise、没有门控网络。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。但注意：这把"重要性权重 γ"做成了显式可替换的旋钮——这是 Δt 最自然的理论落点，作者没做。
- 若用 Δt，它影响：不适用（若做，就是替换 γ_i 的时间基）。
- 是否支持不规则采样 / 丢帧 / 重复帧：不涉及；γ 按序号幂乘，因此天然按步数衰减。
- 有无定理：**有数学推导**（回归-记忆对应；weighted least squares 闭式；Woodbury 序列化；附录给出线性回归层范数界），**无**采样率不变性/稳定性定理。
- 用于：框架性工作，统一线性注意力、gated linear attention、SSM、online learner、softmax attention、线性回归层（实验以语言建模为主）。
- 与候选结构的异同：衰减项 ✅ 形式上是候选的"逐通道指数衰减"的离散对数版（exp(−λΔt) 在等间隔采样下就退化为 γ^t）；写入项 ✅ delta + weight decay；读操作 ✅；双状态 ❌；Δt ❌（但给出了 Δt 的插入位置）。
- 裁决：**部分覆盖**——理论上最接近候选"衰减+内容寻址写"的闭式形式；Δt 缺席。

### 7. Using Fast Weights to Attend to the Recent Past
- 作者 / 年份 / 会议或期刊 / 链接：Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu / 2016 / NeurIPS 2016（papers.nips.cc/paper/6057）/ https://arxiv.org/abs/1610.06258
- 记忆状态是什么：**KV 矩阵**。快权重 A(t) ∈ R^{H×H}（H 为隐层维度），外加慢权重 W、C。
- 读操作精确公式：h_{s+1}(t+1) = f( [W h(t) + C x(t)] + A(t) h_s(t+1) )，s = 0…S（Eq 2）；等价形式 A(t) h_s = η Σ_τ λ^{t−τ} h(τ)[h(τ)^⊤ h_s]（Eq 4）——即"用当前隐状态去 attention 所有历史隐状态，按 λ 的幂衰减加权"。
- 写操作精确公式：A(t) = λ A(t−1) + η h(t) h(t)^⊤（Eq 1）；闭式 A(t) = η Σ_{τ=1..t} λ^{t−τ} h(τ) h(τ)^⊤（Eq 3）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**是**（A ← λA + η h h^⊤），但是 **Hebbian 外用积**，key=value=隐状态本身，没有独立 key。
- alpha_t 究竟是什么：**其他**——η 是固定学习率（实验取 0.5），λ 是固定衰减（实验取 0.9），均不可学（附录 A）。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中；衰减按步数）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不涉及；重复帧会重复叠加同一个 h h^⊤，无去重。
- 有无定理：**无**（只有 Eq 4 的"等价于 attention"解释，不是定理）。
- 用于：视觉/关联检索、MNIST、表情识别。
- 与候选结构的异同：衰减项 ✅ 固定 λ 的 per-step 幂衰减（候选是把 λ 换成 exp(−ΛΔt)）；写入项 ⚠️ 外积叠加，但 key 不是内容编码；读操作 ✅ 内循环 attention 式读；双状态 ✅ **结构最像**（慢权重 W 与快权重 A 分离 + 隐状态 h）；Δt ❌ 且 λ/η 不可学。
- 裁决：**部分覆盖**——"双状态 + 快权重矩阵 + 幂衰减"的最早形态；缺内容寻址 key、缺 Δt、衰减不可学。

### 8. Memorizing Transformers
- 作者 / 年份 / 会议或期刊 / 链接：Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy / 2022 / ICLR 2022（spotlight）/ https://arxiv.org/abs/2203.08913
- 记忆状态是什么：**外部 memory slots**。非可微的 kNN 表，存 (key, value) 对；每层每 head 一个，容量 M（实验中 8192 / 32K / 262K 条），超出后 FIFO 丢弃；文档结束清空。
- 读操作精确公式：V_m = softmax(q K_topk^⊤) V_topk（对检索到的 top-k 做标准 softmax attention）；V_a = V_m ⊙ g + V_c ⊙ (1−g)，g = σ(b_g)（Eq 1–2），b_g 是每 head 的可学标量。
- 写操作精确公式：**每个训练步结束后，把 local 窗口里的 (k,v) 直接追加到 memory 末尾**（§3.1 原话："after each training step, the (key, value) pairs in the local context are appended to the end of the external memory"）。无更新规则、无合并、无权重。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**部分**（append-only；不是内容寻址更新）。
- alpha_t 究竟是什么：**无 α**（写入没有强度）；读时有一个学出的门 g 用于融合两路 attention，与写入无关。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：段级追加，丢帧=少写，重复帧=表里重复条目（检索会命中重复）。梯度不回传进记忆（非可微，这是论文可扩展性的关键）。
- 有无定理：**无**（只有容量-困惑度经验曲线，以及"可泛化到训练时更大的 memory"的经验结论）。
- 用于：语言建模（arXiv math / Github / PG-19 / Isabelle / C4）。
- 与候选结构的异同：衰减项 ❌（只有 FIFO 丢弃，是"容量淘汰"不是"时间衰减"）；写入项 ❌ append；读操作 ✅ kNN + softmax，是可寻址读；双状态 ⚠️（Transformer 层 + 外部表）；Δt ❌。
- 裁决：**部分覆盖**——"可寻址关联记忆、且不重写已有内容"这一主张(a)(b)在外部表的形态下已实现；但无写规则、无衰减、无 Δt。

### 9. Memory Layers at Scale
- 作者 / 年份 / 会议或期刊 / 链接：Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Ghosh / 2024-12 / ICML 2025（PMLR v267, berges25a）/ https://arxiv.org/abs/2412.09764 ; https://proceedings.mlr.press/v267/berges25a.html
- 记忆状态是什么：**参数**（大规模可训练键值槽）。K ∈ R^{N×n}, V ∈ R^{N×n} 是**可训练参数**（不是激活）；N 可达百万到千万级（product key 分解；8B 模型用 4096² = 16M 个 memory value，64B 记忆参数）。
- 读操作精确公式：I = SelectTopkIndices(Kq)；s = Softmax(K_I q)；y = s V_I（Eq 1）。
- 写操作精确公式：**没有在线写**。只通过训练时的稀疏 top-k 梯度更新；推理时只读。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**否**。
- alpha_t 究竟是什么：**无 α**（top-k 的 softmax 权重只用于读聚合）。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不适用（不是流式记忆）。
- 有无定理：**无**。
- 用于：语言建模 / 事实性 QA（NQ、TriviaQA、HotPotQA、MMLU…）。
- 与候选结构的异同：衰减项 ❌；写入项 ❌（离线训练）；读操作 ✅ top-k 软检索；双状态 ❌；Δt ❌。它回答的是"容量能不能做大"，不是"记忆如何随时间变"。
- 裁决：**未覆盖**——只与"可寻址记忆"这一背景相关。

### 10. Generalization through Memorization: Nearest Neighbor Language Models
- 作者 / 年份 / 会议或期刊 / 链接：Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis / 2019-11（v2 2020-02）/ ICLR 2020 / https://arxiv.org/abs/1911.00172
- 记忆状态是什么：**外部 memory slots**。(K,V) datastore，语料中每个 token 一条（可达数十亿条）；key=上下文表示，value=target token。
- 读操作精确公式：p_kNN(y|x) ∝ Σ_{(k_i,v_i)∈N} 1_{y=v_i} exp(−d(k_i, f(x)))；最终 p(y|x) = λ p_kNN(y|x) + (1−λ) p_LM(y|x)。
- 写操作精确公式：离线一次构建（对语料做前向，把 (f(x), y) 追加进 datastore）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**否**。
- alpha_t 究竟是什么：**其他**——λ 是调出来的插值系数（不是学出的门）。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不适用；datastore 可换 = 域适应。
- 有无定理：**无**。
- 用于：语言建模、域适应。
- 与候选结构的异同：纯检索基线；无衰减、无写规则、无双状态、无 Δt。
- 裁决：**未覆盖**（背景文献；但它是"记忆即查表"这条线的代表）。

### 11. Hopfield Networks is All You Need
- 作者 / 年份 / 会议或期刊 / 链接：Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter / 2020-07（v3 2021-04）/ ICLR 2021 / https://arxiv.org/abs/2008.02217
- 记忆状态是什么：**KV 矩阵**。模式矩阵 X ∈ R^{d×N}（列 = 存储的模式），即"用叠加存储的联想记忆"。
- 读操作精确公式：ξ_new = f(ξ) = X softmax(β X^⊤ ξ)（一次更新完成检索；β 为逆温度）。
- 写操作精确公式：把模式作为列叠加进 X（无在线修正、无衰减）；论文的更新规则就是检索规则。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**部分**（叠加式写，无内容寻址修正）。
- alpha_t 究竟是什么：**其他**——β 是逆温度（可学/固定），不是写入权重。
- 是否使用物理时间 Δt：**否**（关键词核查 0 命中）。
- 若用 Δt，它影响：不适用。
- 是否支持不规则采样 / 丢帧 / 重复帧：不适用。
- 有无定理：**有**——一次更新即可检索；存储容量随维度**指数**增长（现代 Hopfield 约 2^{d/2}，对比经典 Hopfield 的 d/(2 log d)）；检索误差指数小（Theorems 1–4 区）。
- 用于：联想记忆、attention 的理论解释。
- 与候选结构的异同：衰减项 ❌；写入项 ❌；读操作 ✅（softmax 检索，与 LSTM/attention 同形）；双状态 ❌；Δt ❌。但它给出**单矩阵叠加记忆的容量上界**——这是评估候选结构"能记多少"必须引用的硬约束。
- 裁决：**未覆盖**（但容量定理直接约束候选的容量主张）。

### 12. Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D)
- 作者 / 年份 / 会议或期刊 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2016（v2）/ Scientific Reports (2018) / https://arxiv.org/abs/1606.01865 ; https://www.nature.com/articles/s41598-018-24271-9
- 记忆状态是什么：**普通向量**。GRU 隐状态 h_t ∈ R^H（无矩阵、无 key-value 结构）。
- 读操作精确公式：输出取 h_t（无独立读操作）。
- 写操作精确公式：
  - 衰减率 γ_t = exp{ −max(0, W_γ δ_t + b_γ) }（Eq 4），δ_t 是该变量距上次观测的**真实时间间隔**
  - 隐状态先衰减：h_{t−1} ← γ_{h,t} ⊙ h_{t−1}（Eq 6）
  - 然后标准 GRU 更新 h_t = (1−z_t)⊙h_{t−1} + z_t⊙h̃_t
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**部分**（有 Δt 驱动的衰减，但更新是输入驱动的 GRU，不是内容寻址）。
- alpha_t 究竟是什么：**外部真实 Δt**。γ_t 是 Δt 经可学线性映射 + 负指数得到的衰减；W_γ 学出，且输入衰减时约束 W_γ 为对角（逐通道独立衰减）——这与候选的"Λ 逐通道对角"是同一个设计选择。
- 是否使用物理时间 Δt：**是**（δ_t 是真实经过时间，不是序号）。
- 若用 Δt，它影响：**衰减**（输入特征衰减 + 隐状态衰减两项）。
- 是否支持不规则采样 / 丢帧 / 重复帧：**原生支持**——δ_t 就是在不规则采样/缺测下定义的；重复帧 δ≈0 → γ≈1（不衰减）。
- 有无定理：**无**。
- 用于：临床多变量时间序列（分类/预测，PhysioNet 等）。
- 与候选结构的异同：衰减项 ✅ **逐通道**、由真实 Δt 驱动、指数形式 exp(−·)——与候选的 exp(−Λ Δ_i) 在结构上同族；写入项 ❌ 无 key 寻址、无外积；读操作 ❌；双状态 ❌；Δt 落点 ✅ 只落在衰减。
- 裁决：**部分覆盖**——Δt 落点在"逐通道指数衰减"上已被 GRU-D 做了 8 年，但只作用于向量隐状态。

### 13. Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences
- 作者 / 年份 / 会议或期刊 / 链接：Daniel Neil, Michael Pfeiffer, Shih-Chii Liu / 2016 / NeurIPS 2016（口头报告）/ https://arxiv.org/abs/1610.09513
- 记忆状态是什么：**普通向量**（LSTM 的 cell c_t 与 hidden h_t）。
- 读操作精确公式：输出取 h_t。
- 写操作精确公式：时间门 k_t 由真实时间戳 t 决定：φ_t = ((t − s) mod τ)/τ，k_t 是 φ_t 的分段线性函数（open/closed 三相，参数 τ、s、r_on 可学）；只有 gate 打开时才更新 c_t、h_t，否则保持（c_j = c_{j−Δ}）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**部分**（时间门控的保持/更新，不是内容寻址）。
- alpha_t 究竟是什么：**外部真实 Δt**。k_t 是**真实时间戳**的周期函数（不是学习出的门、也不是 surprise）。
- 是否使用物理时间 Δt：**是**（时间戳 t 直接进入门）。
- 若用 Δt，它影响：**状态转移**（是否更新/保持；不是指数衰减）。
- 是否支持不规则采样 / 丢帧 / 重复帧：**原生支持**——论文明确针对 irregularly sampled / event-based / 异步输入，可在任意时刻 t_j 更新。
- 有无定理：**无**（有"closed phase 完美保持记忆"的定性讨论）。
- 用于：事件序列、长序列、异步传感器。
- 与候选结构的异同：Δt 落点 ✅ 但形式是门控保持，不是 exp 衰减；无双状态矩阵、无内容寻址、无读操作。
- 裁决：**部分覆盖**——证明"真实时间戳进入循环状态"这条思路早在 2016 年就有，且是为不规则采样设计的。

### 14. FLAME: Fast Long-context Adaptive Memory for Event-based Vision
- 作者 / 年份 / 会议或期刊 / 链接：Biswadeep Chakraborty, Saibal Mukhopadhyay / 2025 / NeurIPS 2025 / https://papers.nips.cc/paper_files/paper/2025/file/1b71649968436c64a765e469ab6b615c-Paper-Conference.pdf
- 记忆状态是什么：**普通向量**（SSM 状态 x(t) ∈ R^N，N=64；HiPPO-LegS 初始化的状态转移矩阵 A ∈ R^{N×N}）。
- 读操作精确公式：对最后 K 个状态做时间平均得 x_pooled，再过线性层 + softmax：y_pred = Softmax(W_out x_pooled + b_out)。
- 写操作精确公式（**全文最接近候选的 Δt 机制**）：
  - 连续时间：ẋ(t) = A_S(t) x(t) + B E_flat(t)
  - 时变转移矩阵：**A_S(t) ≡ A_S(Δt) = A ○ F(Δt)**，其中 **F_ij(Δt) = e^{−α_ij Δt}**，α_ij > 0 可学（○ 为 Hadamard 积）
  - 事件驱动离散更新（Eq 3）：x(t_{k+1}) = e^{A_S(Δt_k) Δt_k} x(t_k) + (A_S(Δt_k))^{−1}(e^{A_S(Δt_k)Δt_k} − I) B E_event
- 是否存在 M(t+1)=F(M_t,k_t,v_t,α_t) 或 M←M+内容寻址更新：**部分**——有"矩阵按 Δt 逐元素指数衰减 + 输入注入"，但注入是 B·输入，不是内容寻址外积。
- alpha_t 究竟是什么：**外部真实 Δt**。α_ij 是可学的逐元素时间常数；衰减率由真实 inter-event interval Δt 决定（"frequent events (Δt≈0) lead to A_S≈A (preserving memory), sparse events allow faster decay"）。
- 是否使用物理时间 Δt：**是**。
- 若用 Δt，它影响：**衰减 / 状态转移**（A 的调制 + 矩阵指数）。
- 是否支持不规则采样 / 丢帧 / 重复帧：**原生支持事件流**（异步、稀疏，逐事件更新）。
- 有无定理：**有**——Theorem 2 状态轨迹有界性（Lyapunov 方程 A_S^⊤P + P A_S = −Q，要求 A_S Hurwitz）；Taylor 余项误差界；复杂度 O(Nr)（NPLR 分解）；并论证 Hurwitz 性在 F(Δt)∈(0,1] 调制下保持。
- 用于：事件相机视觉（分类）。
- 与候选结构的异同：衰减项 ✅✅ **exp(−α Δt) 逐元素衰减，与 exp(−Λ Δ_i) ⊙ M 同构**；写入项 ❌ 无 key 寻址/无外积；读操作 ❌ 线性 readout；双状态 ❌；Δt 落点 ✅ 落在衰减/状态转移。
- 裁决：**部分覆盖**——候选 Δt 衰减那一半的**最强先例 + 唯一带稳定性定理的先例**；但完全没有内容寻址关联记忆。

---

## 2. 横向对照表

| # | 论文 | 记忆载体 | 内容寻址写 | 衰减 | Δt 语义 | 双状态 | 定理 | 领域 |
|---|---|---|---|---|---|---|---|---|
| 1 | TTT layers | 参数 W | 梯度方向（误差驱动） | 无 | 无 | 无 | 无（有与 linear attention 等价） | LM |
| 2 | Titans | 参数（MLP/W） | ✅ 误差修正 + 动量 | ✅ (1−α_t) per token | 无 | 有（core+memory） | Theorem 4.1（表达力） | LM/长上下文 |
| 3 | MNM | 参数 φ | ✅ 一步梯度 | 无 | 无 | 有（RNN+memory） | 无 | 算法/QA/RL |
| 4 | Gated DeltaNet | KV 矩阵 | ✅ 先擦后写 | ✅ α_t 标量 | 无 | 无 | 在线学习等价 | LM/检索 |
| 5 | Schlag FWP | KV 矩阵 | ✅ delta 修正 | 无 | 无 | 无 | 等价性 | LM |
| 6 | Test-time regression | KV 矩阵/函数 | ✅ delta+WD | ✅ γ 几何权重 | 无（但有插入点） | 无 | 推导（回归对应） | 框架 |
| 7 | Ba fast weights | KV 矩阵 | ⚠️ Hebbian 外积 | ✅ λ 幂（固定） | 无 | ✅ | 无 | 视觉 |
| 8 | Memorizing Transformer | 外部表 | 部分（append） | ❌（FIFO） | 无 | ⚠️ | 无 | LM |
| 9 | Memory Layers | 参数槽 | ❌ | 无 | 无 | 无 | 无 | LM/QA |
| 10 | kNN-LM | 外部表 | ❌ | 无 | 无 | 无 | 无 | LM |
| 11 | Modern Hopfield | KV 矩阵 | ❌ 叠加 | 无 | 无 | 无 | ✅ 指数容量 | 联想记忆 |
| 12 | GRU-D | 向量 h | ❌ | ✅ exp(−W_γδ) 逐通道 | ✅ | 无 | 无 | 临床时序 |
| 13 | Phased LSTM | 向量 c,h | ❌ | ❌（门控保持） | ✅ | 无 | 无 | 事件序列 |
| 14 | FLAME | 向量 x（SSM） | ❌ | ✅ A○e^{−αΔt} | ✅ | 无 | ✅ 稳定性/有界 | 事件视觉 |

**读表结论**：No.4 / No.6 / No.7 是"KV 矩阵 + 衰减 + 内容寻址写"的三种组合；No.12 / No.14 是"Δt 落在衰减上"的两种组合；**没有任何一篇同时具备两者**——这个组合空缺正是候选结构声称的位置。

---

## 3. 反对结论与反例

以下每条都直接针对用户的直觉，并给出可核实证据。

**R1（推翻"不变内容不反复重写"）候选的写入项是加法 Hebbian，必然重复写。**
M_i = M_i- + α_i v_i k_i^T：只要 α_i > 0，即使内容完全没变、k、v 与上一步相同，这一步仍然加上 α v k^T。Titans 明确指出这条路的后果："the additive nature of the process results in memory overflow, significantly damaging the performance of the model"（arXiv:2501.00663 §1）。Ba 2016 的 Eq 1 与 Schlag 的 Eq 2 都是同一加法形式，都靠 λ 或容量来兜底。要真正做到"不变就不写"，必须换成 delta 写：W ← W + β(v − Wk)k^T（Schlag Eq 24；Gated DeltaNet §2；test-time regression Eq 26）。此时写入幅度 **α_i 已经不是任意的门，而是 (v − Mk) 的误差**。也就是说：**主张(b)与候选公式不相容；候选公式是主张(c)的减弱版。**

**R2（推翻"Δt 让记忆更物理"）衰减的推导根本不需要物理时间，它来自"关联重要性权重"。**
Test-time regression 把 gating 严格推导为加权最小二乘里几何递减的权重 γ_i^{(t)} = Π γ_j，并解释成"非平稳在线回归"；Titans 把 (1−α) 解释成 weight decay；Gated DeltaNet 把 α_t 解释成放宽 L2 正则。**在这三条推导里，时间只以序号出现**——把 γ 换成 e^{−λΔt} 是一个可以做的工程替换，不是理论必然。所以"用真实时间而不是帧数"并不是更深的第一性原理，而是给同一个衰减核换了个自变量。

**R3（采样率不变性：候选只做对了一半，这是硬缺陷）**
衰减项满足半群性：exp(−Λ(t_i − t_j))·exp(−Λ(t_j − t_k)) = exp(−Λ(t_i − t_k))，因此**衰减**与采样率无关。但写入项 α v k^T 不随 Δt 缩放：在总时长 T 内采样 N 帧，稳态 ‖M‖ 的量级 ∝ N（重复帧、高频帧会把记忆堆满）。真正采样率不变的写法必须是写入强度也随 Δt 缩放，例如 α_i = 1 − e^{−λΔ_i}（或 α_i ∝ Δ_i 的小量展开）。检索到的 14 篇里**没有一篇**做这件事：Titans 的 α_t、θ_t 是 token 的函数；Gated DeltaNet 的 α_t/β_t 与 Δt 无关；FLAME 的 Δt 只进衰减矩阵，不进输入注入项（B E_event 的增益不含 Δt）。→ 用户主张(d) 的"写入也由 Δt 控制"目前**没有先例，也没有被证明可行**。

**R4（Δt 语义不是新的：三轮先例已经在场）**
- GRU-D Eq 4/6：γ_t = exp{−max(0, W_γ δ_t + b_γ)}，δ_t 是真实经过时间；隐状态按 γ 逐通道衰减——**与候选的 exp(−ΛΔ)⊙ 形式同族，2018 年就有**。
- Phased LSTM：时间门 k_t 由真实时间戳驱动，明确为 irregular/event-based 设计（2016）。
- FLAME（NeurIPS 2025）：A_S = A ○ F(Δt)，F_ij = e^{−α_ij Δt}，且**带 Hurwitz/有界性定理**。
结论：候选的 Δt 衰减只能声称"作用于 KV 关联矩阵"这一点是新的——而"Δt 逐元素指数衰减一个矩阵"本身已被 FLAME 做过（作用在 SSM 转移矩阵上）。

**R5（双状态没有任何理论支撑）**
14 篇里有定理的只有三家：Titans 的 Theorem 4.1 只谈单记忆的表达力；FLAME 的 Theorem 2 只谈单状态有界性；Modern Hopfield 的定理只谈单矩阵容量。**没有一篇给出"耦合两个状态后的稳定性 / 容量 / 误差界"**。候选的 h_i = H(h_{i−1}, r_i, z_i) 与 M_i 的耦合在文献里是空白，不是共识；空白既意味着机会，也意味着"双状态更强"目前只是未验证的假设。

**R6（容量硬上限：单矩阵叠加记不住那么多）**
Modern Hopfield 给出容量随维度指数增长（约 2^{d/2}，对比经典的 d/(2 log d)）且一次更新可检索；但这是**对"干净叠加"的界**，一旦叠加数接近容量，检索误差指数上升。候选靠 exp(−ΛΔt) 腾位置，本质是"指数遗忘的滑窗"：稳态有效关联数 ~ 1/(1−e^{−λΔt}) 量级。这与 Titans 的 α_t 调控、Memorizing Transformer 的 FIFO 丢弃是同一类容量管理，**没有新机制**。

**R7（并行/部署代价：Δt 依赖的逐通道衰减会破坏 chunk 化）**
Titans Eq 16–18 能把记忆训练写成 chunk 内 matmul + associative scan，是因为 (1−α_t)、θ_t 都是 token 的函数、且结构允许 β_t/β_i 的闭式；FLAME 为了在事件流上跑，用了 NPLR 低秩分解 + 二阶 Taylor 近似矩阵指数，才把复杂度压到 O(Nr)。若 Λ 逐通道且 Δt 任意，exp(−ΛΔt) 在 chunk 内不再是常数标量，硬件友好的"一次 matmul 出一整块"形式退化；同时 Δt 量级差异大时 exp 会下溢/上溢。**在 50 Hz 控制环上，这会直接变成算力与数值稳定性问题。**

**R8（"测试时学习=真在学关联记忆"这一叙事本身被质疑）**
TTT 作者自己在 v4 limitations 里承认："The idea of using linear models as hidden states has already been well studied in DeltaNet"，并说自己的贡献只是"能实例化任意网络的框架"。另有 2026 年工作直接主张 TTT+KV binding 其实只是 learned linear attention（arXiv:2602.21204，标题 *Test-Time Training with KV Binding Is Secretly Linear Attention*，NVIDIA）。→ 如果 TTT 的记忆不是"真记忆"，那么以 TTT 为师的候选结构，其"记忆"语义也需要重新论证。

**R9（Titans 自己认为关键的是"过去的 surprise"，不是时间）**
Titans 把 surprise 拆成 past surprise 与 momentary surprise（Eq 10），并用 momentum η_t 传递前者，理由是"梯度在若干次惊讶之后会变小，卡在平坦区"（§3.1 原文）。**这是用动量而不是用 Δt 来编码时间上的信息流**——说明在这个研究传统里，"时间"被当作序列顺序处理，而不是物理时长。

---

## 4. 逐项覆盖裁决

| 候选结构组件 | 覆盖状态 | 证据 |
|---|---|---|
| 矩阵快权重关联记忆 M | **已覆盖** | Schlag Eq 2/23；Ba Eq 1；Gated DeltaNet §2；Titans Eq 32；test-time regression Eq 26 |
| 内容寻址写入（按 key 更新） | **已覆盖（且候选版本偏弱）** | Schlag Eq 24 的 β(v−Mk)k^T；GDN (I−βkk^T) 擦写；Titans 梯度写 |
| 衰减项（逐通道对角） | **已覆盖（但是 per-step）** | Titans (1−α_t)、GDN α_t、Ba λ、ttreg γ；逐通道见 GRU-D γ_h 与 Titans Eq 32 diag |
| 物理 Δt 控制衰减 | **已覆盖（另一条线）** | GRU-D Eq 4/6；FLAME A_S=A○F(Δt) + Theorem 2；Phased LSTM 时间门 |
| 物理 Δt 控制**写入强度** | **未覆盖** | 14 篇无一篇让写入强度随 Δt 缩放；R3 给出为什么这会破坏采样率不变性 |
| 双状态（M 与 h 耦合） | **部分/无理论** | Ba 2016 有慢权重+快权重+h；MNM controller+memory；但无耦合稳定性/容量定理 |

**最接近的 3 个先例（按与候选的距离排序）**
1. **Titans, arXiv:2501.00663（NeurIPS 2025）** — M_t = (1−α_t)M_{t−1} + S_t，S_t = η_t S_{t−1} − θ_t∇ℓ(M_{t−1};x_t)，ℓ = ‖M(k_t) − v_t‖²；Eq 32 的 diag 形式已允许逐通道。差：α 是 token 的函数，没有 Δt。
2. **Gated DeltaNet, arXiv:2412.06464（ICLR 2025）** — S_t = S_{t−1}(α_t(I − β_t k_t k_t^⊤)) + β_t v_t k_t^⊤，o_t = S_t q_t。差：α_t 是数据依赖标量、无 Δt、无独立 h。
3. **Test-time regression, arXiv:2501.12352** — M_t = (1 − β_tλ_t)M_{t−1} + β_t(v_t − M_{t−1}k_t)k_t^⊤，且从几何加权最小二乘推出衰减的必要性；γ_i 就是候选 exp(−ΛΔ) 的离散对数版。

**次接近（Δt 那一半）**：FLAME, NeurIPS 2025 — A_S(Δt) = A ○ F(Δt), F_ij = e^{−α_ijΔt}，逐元素指数衰减 + Hurwitz/有界性定理。GRU-D, Scientific Reports 2018 — γ_t = exp{−max(0, W_γδ_t+b_γ)} 逐通道衰减隐状态。

**是否发现与候选结构逐字对应的工作：没有。**
检索（含 advanced 深度、事件流/不规则采样/传感器流等交叉词）未发现任何一篇同时具备"KV 矩阵关联记忆 + 内容寻址写 + 物理 Δt 控制的衰减"。最接近的两半分别是 Titans/GDN（关联记忆那一半）与 FLAME/GRU-D（Δt 那一半）。因此候选结构的**组合**在检索范围内是新的；但**每一半都不是新的**，且候选的加法写（相对 delta 写）与"写入强度不随 Δt 缩放"是两个可被直接攻击的薄弱点。

---

## 5. 核实清单

| # | 标题 | 核实到的元数据 | 链接 |
|---|---|---|---|
| 1 | Learning to (Learn at Test Time): RNNs with Expressive Hidden States | Sun 等 12 人；ICML 2025 (PMLR v267 sun25h) | https://arxiv.org/abs/2407.04620 |
| 2 | Titans: Learning to Memorize at Test Time | Behrouz, Zhong, Mirrokni；NeurIPS 2025 | https://arxiv.org/abs/2501.00663 |
| 3 | Metalearned Neural Memory | Munkhdalai, Sordoni, Wang, Trischler；NeurIPS 2019 | https://arxiv.org/abs/1907.09720 |
| 4 | Gated Delta Networks: Improving Mamba2 with Delta Rule | Yang, Kautz, Hatamizadeh；ICLR 2025 | https://arxiv.org/abs/2412.06464 |
| 5 | Linear Transformers Are Secretly Fast Weight Programmers | Schlag, Irie, Schmidhuber；ICML 2021 (PMLR v139) | https://arxiv.org/abs/2102.11174 |
| 6 | Test-time regression: a unifying framework... | Wang, Shi, Fox；arXiv 2501.12352（会议位未核实） | https://arxiv.org/abs/2501.12352 |
| 7 | Using Fast Weights to Attend to the Recent Past | Ba, Hinton, Mnih, Leibo, Ionescu；NeurIPS 2016 | https://arxiv.org/abs/1610.06258 |
| 8 | Memorizing Transformers | Wu, Rabe, Hutchins, Szegedy；ICLR 2022 | https://arxiv.org/abs/2203.08913 |
| 9 | Memory Layers at Scale | Berges, Oğuz, Haziza, Yih, Zettlemoyer, Ghosh；ICML 2025 (PMLR v267) | https://arxiv.org/abs/2412.09764 |
| 10 | Generalization through Memorization: Nearest Neighbor Language Models | Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis；ICLR 2020 | https://arxiv.org/abs/1911.00172 |
| 11 | Hopfield Networks is All You Need | Ramsauer 等 16 人；ICLR 2021 | https://arxiv.org/abs/2008.02217 |
| 12 | Recurrent Neural Networks for Multivariate Time Series with Missing Values | Che, Purushotham, Cho, Sontag, Liu；Scientific Reports 2018 | https://arxiv.org/abs/1606.01865 |
| 13 | Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences | Neil, Pfeiffer, Liu；NeurIPS 2016 | https://arxiv.org/abs/1610.09513 |
| 14 | FLAME: Fast Long-context Adaptive Memory for Event-based Vision | Chakraborty, Mukhopadhyay；NeurIPS 2025 | https://papers.nips.cc/paper_files/paper/2025/file/1b71649968436c64a765e469ab6b615c-Paper-Conference.pdf |

核查说明：1–14 的正文均为实拉（arXiv HTML 或官方 PDF），公式取自正文/公式块；仅 #12 与 #14 的部分正文引用来自 PDF 文本抽取（公式块经人工比对上下文后引用）。未使用任何"凭记忆"的公式；未发现需要标 [未核实] 的条目。


# 03 · 事件驱动/稀疏写入、surprise 驱动更新、选择性 SSM、对象级记忆更新
### 对照候选：M_i⁻ = exp(−Λ⊙Δt_i)⊙M_{i−1}；M_i = M_i⁻ + α_i v_i k_iᵀ；r_i = R(M_i,q_i)；h_i = H(h_{i−1},r_i,z_i)

> **核实方式**：tavily_search（advanced）+ tavily_extract 实拉正文（arXiv PDF/HTML、官方 proceedings、OpenReview/项目页）。公式均抄自正文原式，页码/式号随文标注；仅摘要可得的标【仅摘要级】。每条链接本轮都实际拉取过。
> **读表须知**：候选公式里有三项未定义——(a) α_i 是什么（学出的门？阈值？surprise？）；(b) Δt_i 从哪来（时钟？帧间隔？）；(c) h 与 M 如何互相耦合。下表在"alpha_t 归类"和"Delta t 落点"两栏里对每篇都做了归类，最后在「反对结论与反例」里回到这三项。
> **本文件覆盖**：选择性 SSM、surprise/梯度驱动记忆、快权重/线性注意力、现代 Hopfield、外部记忆（NTM）、事件级对象记忆（TGN）、硬稀疏模块（RIMs）、稀疏计算（MoE）、槽位记忆（Slot Attention）、硬跳过 RNN、自适应计算、事件触发估计。

---

## 1. 选择性状态空间与 surprise 驱动写入

### 1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **作者 / 年份 / 会议或期刊 / 链接**：Albert Gu, Tri Dao / 2023 / arXiv 预印本（arXiv:2312.00752；COLM 2024 版本轮未单独核实）/ https://arxiv.org/pdf/2312.00752
- **记忆状态**：向量。每通道一个 N 维隐状态 h_t ∈ R^N（输入维 D，展开状态 D×N）；不是矩阵关联记忆。
- **读操作精确公式**：$y_t = C_t h_t$，其中 $C_t = s_C(x_t) = \mathrm{Linear}_N(x_t)$（式 (2b) + Algorithm 2）。
- **写操作精确公式**：$h_t = \bar A_t h_{t-1} + \bar B_t x_t$；ZOH 离散化 $ \bar A = \exp(\Delta A),\ \bar B = (\Delta A)^{-1}(\exp(\Delta A)-I)\cdot\Delta B$（式 (4)）；选择机制 $\Delta_t = \tau_\Delta(\text{Parameter}+s_\Delta(x_t))$，$\tau_\Delta = \mathrm{softplus}$，$B_t = s_B(x_t)$，$C_t = s_C(x_t)$（§3.2 + Algorithm 2）。
- **是否存在 M←M+内容寻址更新**：部分。是线性的"状态写入"，不是 key-value 内容寻址（$\bar B_t x_t$ 没有 $v k^\top$ 结构）。
- **alpha_t 归类**：**学出的门**。$\Delta_t,\bar B_t$ 均由输入生成；论文明确把 $\Delta$ 的选择性与 RNN 门控类比（§3.5）。
- **用物理时间 Δt 还是观测序号**：**观测序号**。$\Delta$ 是学习出来的步长，不是实测时间；连续时间解释只用于说明离散化性质。
- **若用 Δt，影响**：不适用。$\Delta$ 同时进入**衰减**（$\bar A=\exp(\Delta A)$）与**写入强度**（$\bar B$），但它是学习量。
- **支持不规则采样/丢帧/重复帧吗**：不支持（$\Delta$ 不来自时钟）。论文只声称离散化带来"resolution invariance"（引 Nguyen et al. 2022），并未在变采样率上做实验。
- **有无定理**：无稳定性/半群定理；有连续时间 ZOH 解释与"resolution invariance"的文献引述。
- **用于**：语言、音频、基因组等序列建模（非控制/传感器）。
- **与候选结构异同**：**衰减算子同构**——候选的 $\exp(-\Lambda\odot\Delta t)\odot M$ 与 Mamba 的 $\bar A=\exp(\Delta A)$ 是同一个 ZOH 公式；差别是 Mamba 状态为向量、$\Delta$ 学习、写入非内容寻址。候选要主张"衰减"的新颖性必须解释与式 (4) 的差别。
- **裁决**：**部分覆盖**——"按 Δ 指数衰减"的算子已被写死（式 (4)），但"Δ=实测秒"与"矩阵记忆 + 外积写"未被 Mamba 覆盖。

### 2. Titans: Learning to Memorize at Test Time
- **作者 / 年份 / 会议或期刊 / 链接**：Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research) / 2025 / arXiv:2501.00663 / https://arxiv.org/pdf/2501.00663（HTML: https://arxiv.org/html/2501.00663v1）
- **记忆状态**：参数/梯度式矩阵记忆 $\mathcal M$（深层 MLP，§3.1 也明确支持矩阵值记忆）＋core（短窗 attention，短时记忆）＋persistent memory——**天然多状态**。
- **读操作精确公式**：$\mathcal M_t$ 以 $k_t$ 为输入做前向：$\mathcal M_t(k_t)$（式 (12) 中作为预测项）；$k_t = x_t W_K,\ v_t = x_t W_V$（式 (11)）。
- **写操作精确公式**：$\mathcal M_t = (1-\alpha_t)\mathcal M_{t-1} + S_t$（式 (13)）；$S_t = \eta_t S_{t-1} - \theta_t \nabla\ell(\mathcal M_{t-1}; x_t)$（式 (14)）；$\ell(\mathcal M_{t-1},x_t)=\lVert \mathcal M_{t-1}(k_t)-v_t\rVert_2^2$（式 (12)）。特例 $\mathcal M_t=\mathcal M_{t-1}-\theta_t\nabla\ell$（式 (8)）。
- **是否存在 M←M+内容寻址更新**：**是**。写是梯度步，内容通过 $(k_t,v_t)$ 进入；关联记忆损失就是 $v_t$ 与记忆检索值的误差。
- **alpha_t 归类**：**学出的门** $\alpha_t\in[0,1]$（数据依赖的遗忘门）；$\eta_t$ 是 surprise 的衰减门，$\theta_t$ 是写强度；surprise 本身是**梯度**（预测误差的梯度形式）。
- **用物理时间 Δt 还是观测序号**：**观测序号**（token 序）。
- **若用 Δt，影响**：不适用。遗忘是 $(1-\alpha_t)$ 的乘性衰减，$\alpha_t$ 与时间无关。
- **支持不规则采样/丢帧/重复帧吗**：不支持（无时间戳机制）。
- **有无定理**：无（论文只给与 modern RNN 的等价性讨论，未给稳定性/容量定理）。
- **用于**：语言建模、长上下文、时间序列、DNA。
- **与候选结构异同**：候选的 $\exp(-\Lambda\Delta t)\odot M$ 与 Titans 的 $(1-\alpha_t)M$ 都是"先衰减再写"，但 Titans 的衰减门是学出的、没有真实 Δt；候选的写是 $v k^\top$ 外积，Titans 的写是**梯度/在线学习步**（带 momentum 的 surprise）。双状态思想 Titans 已有（memory + core），但候选的 h 与 M 是"同一条观测同时驱动"，Titans 的 core 与 memory 是并列架构。
- **裁决**：**部分覆盖**——矩阵记忆 + 遗忘衰减 + 关联写都有先例；"真实 Δt 衰减"与"外积写"未被覆盖。

### 3. Learning to (Learn at Test Time): RNNs with Expressive Hidden States
- **作者 / 年份 / 会议或期刊 / 链接**：Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin / 2025 / ICML 2025, PMLR 267:57503–57522 / https://arxiv.org/html/2407.04620v4
- **记忆状态**：模型参数 $W$（线性模型或小 MLP），即"状态 = 模型权重"；维度取决于所选模型。
- **读操作精确公式**：$z_t = f(x_t; W_t)$（式 (1)）。
- **写操作精确公式**：$W_t = W_{t-1} - \eta \nabla\ell(W_{t-1}; x_t)$（式 (2)）；$\ell(W;x_t)=\lVert f(\tilde x_t; W)-x_t\rVert^2$（式 (3)，$\tilde x_t$ 是损坏输入）。
- **是否存在 M←M+内容寻址更新**：部分。是梯度写，没有 $v k^\top$ 寻址；但"写什么"由自监督损失决定。
- **alpha_t 归类**：**其他**——写强度就是学习率 $\eta$（固定或可学），非门控；"记得多"由**梯度幅度**决定（原文："Our W remembers inputs that produce large gradients"）＝surprise。
- **用物理时间 Δt 还是观测序号**：**观测序号**（每个 token 一步梯度）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不支持。
- **有无定理**：无（有与线性 attention 的表达力等价讨论）。
- **用于**：语言、长上下文（ImageNet 也有）。
- **与候选结构异同**：双状态雏形（权重记忆 + 输出层）但没有独立 h；无衰减项（无遗忘）、无 Δt、写不是外积。
- **裁决**：**部分覆盖**（surprise 驱动写；矩阵关联记忆、衰减、Δt 均未覆盖）。

### 4. Linear Transformers Are Secretly Fast Weight Programmers
- **作者 / 年份 / 会议或期刊 / 链接**：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139:9355–9366 / https://arxiv.org/html/2102.11174v3
- **记忆状态**：矩阵快权重 $W\in\mathbb{R}^{d_{\text{value}}\times d}$（纯矩阵记忆，**单状态，没有第二路 h**）。
- **读操作精确公式**：$\bar v^{(i)} = W^{(i-1)}\phi(k^{(i)})$（式 (20)）；$y^{(i)} = W^{(i)}\phi(q^{(i)})$（式 (25)）。
- **写操作精确公式**：$W^{(i)} = W^{(i-1)} + \beta^{(i)}(v^{(i)}-\bar v^{(i)})\otimes\phi(k^{(i)})$（式 (24)）；$\beta^{(i)}=\sigma(W_\beta x^{(i)})$（式 (21)）；$v_{\text{new}}=\beta v+(1-\beta)\bar v$（式 (22)）。
- **是否存在 M←M+内容寻址更新**：**是**，且是最标准的 delta-rule 内容寻址写（先检索再纠错写）。
- **alpha_t 归类**：**学出的门** $\beta^{(i)}$（写强度，sigmoid）。
- **用物理时间 Δt 还是观测序号**：**观测序号**（token 步 i）。
- **若用 Δt，影响**：不适用；无衰减项（纯加性累积 + 纠错）。
- **支持不规则采样/丢帧/重复帧吗**：不支持。
- **有无定理**：无稳定性定理；有容量/crosstalk 分析（§4.1：序列长于键维度即 overcapacity，检索被串扰破坏）。
- **用于**：语言建模、机器翻译、合成检索。
- **与候选结构异同**：候选的 $M^- + \alpha v k^\top$ 与式 (24) 只差两点——(i) 式 (24) 先减去旧值（delta/erase），候选是纯加性；(ii) 候选有衰减。**写入线几乎逐字对应**。
- **裁决**：**部分覆盖**（写算子被完整覆盖，含"检索再纠错"；衰减与真实 Δt 未覆盖）。

### 5. Gated Delta Networks: Improving Mamba2 with Delta Rule
- **作者 / 年份 / 会议或期刊 / 链接**：Songlin Yang (MIT CSAIL), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA) / 2025 / ICLR 2025 / https://arxiv.org/abs/2412.06464（官方 PDF 式 (10)：https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf）
- **记忆状态**：矩阵 KV 记忆 $S_t\in\mathbb{R}^{d_v\times d_k}$（单状态）。
- **读操作精确公式**：$o_t = S_t q_t$（线性 attention 读，§2 定义；等价于 $S_t\phi(q_t)$）。
- **写操作精确公式**：$S_t = S_{t-1}\bigl(\alpha_t(I-\beta_t k_t k_t^\top)\bigr) + \beta_t v_t k_t^\top$（式 (10)）；DeltaNet 原形 $S_t=S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top$（§2.2）。
- **是否存在 M←M+内容寻址更新**：**是**（delta 写 + 标量衰减门）。
- **alpha_t 归类**：**学出的门**。$\alpha_t\in(0,1)$ 数据依赖衰减，$\beta_t$ 写强度；**没有真实时间**。
- **用物理时间 Δt 还是观测序号**：**观测序号**（token 级门）。
- **若用 Δt，影响**：不适用；$\alpha_t$ 同时影响**衰减**（乘在 $S_{t-1}$ 上）与旧值擦除（$I-\beta k k^\top$ 里的 $\beta$ 是写强度）。
- **支持不规则采样/丢帧/重复帧吗**：不支持。
- **有无定理**：无稳定性定理；有在线学习框架下的闭式解释（更新 = 在线目标的解）。
- **用于**：语言、长上下文、检索。
- **与候选结构异同**：**衰减算子与写算子都已有**——候选的 $\exp(-\Lambda\Delta t)$ 对应 $\alpha_t$（但 $\alpha_t$ 不是时间），候选的 $+\alpha v k^\top$ 对应 $\beta_t v_t k_t^\top$（但缺 erase 项）。
- **裁决**：**部分覆盖**（矩阵记忆 + 衰减 + 外积写全有；真实 Δt、双状态未覆盖）。

---
## 2. 外部记忆与对象级记忆写入

### 6. Hopfield Networks is All You Need
- **作者 / 年份 / 会议或期刊 / 链接**：Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber, Markus Holzleitner, Milena Pavlović, Geir Kjetil Sandve, Victor Greiff, David Kreil, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter / 2021 / ICLR 2021（OpenReview tL89RnzIiCd 检索确认）/ https://arxiv.org/abs/2008.02217（正文取 CMU 课程镜像 PDF: https://deeplearning.cs.cmu.edu/S24/document/readings/hopfieldnets_is_all_you_need.pdf）
- **记忆状态**：模式矩阵 $X\in\mathbb{R}^{N\times d}$（N 个存好的模式向量）。
- **读操作精确公式**：$\xi_{\text{new}} = f(\xi) = X\,\mathrm{softmax}(\beta X^\top \xi)$（式 (3)）；能量 $E=-\mathrm{lse}(\beta,X^\top\xi)+\tfrac12\xi^\top\xi+\beta^{-1}\log N+\tfrac12 M^2$（式 (2)）。
- **写操作精确公式**：无在线写；"写"是一次性或叠加式地把模式放进 $X$（静态记忆）。
- **是否存在 M←M+内容寻址更新**：**否**（没有在线内容寻址写；记忆是静态的）。
- **alpha_t 归类**：**attention 权重**（softmax 权重，温度 $\beta$）。
- **用物理时间 Δt 还是观测序号**：都无（检索是迭代次数）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不适用（静态模式记忆）。
- **有无定理**：**有**。定理 1：式 (3) 全局收敛（能量单调下降，CCCP 证明）；附录定理 A5/A6：随机模式存储容量指数级；A8：一步检索；A9：指数小检索误差。另式 (10) 证明 $\beta=1/\sqrt{d_k}$ 时式 (3) 就是 transformer attention。
- **用于**：通用（分类、MIL、免疫组库；作为网络层）。
- **与候选结构异同**：候选的读 $r=R(M,q)$ 与式 (3) 同构（$q\equiv\xi$ 的查询端，$M\equiv X$）；但候选的 $M$ 是**在线写入**的、带衰减，Hopfield 的 $X$ 是静态存储。候选若声称"attention 当可寻址关联记忆"，这里是最直接的既有理论基础。
- **裁决**：**部分覆盖**（读算子被完整覆盖，且已被证明等价于 attention；写与时间未覆盖）。

### 7. Neural Turing Machines
- **作者 / 年份 / 会议或期刊 / 链接**：Alex Graves, Greg Wayne, Ivo Danihelka (Google DeepMind) / 2014 / arXiv:1410.5401 / https://arxiv.org/pdf/1410.5401
- **记忆状态**：外部记忆矩阵 $M_t\in\mathbb{R}^{N\times M}$（N 个位置，每位置 M 维）。
- **读操作精确公式**：$r_t \leftarrow \sum_i w_t(i)M_t(i)$（式 (2)），$\sum_i w_t(i)=1,\ 0\le w_t(i)\le 1$（式 (1)）。
- **写操作精确公式**：先擦除 $\tilde M_t(i)\leftarrow M_{t-1}(i)\,[1-w_t(i)e_t]$（式 (3)），再加 $\;M_t(i)\leftarrow \tilde M_t(i)+w_t(i)a_t$（式 (4)）。
- **是否存在 M←M+内容寻址更新**：**是**（内容寻址 + 位置寻址混合的 $w_t$，擦除 + 加写）。
- **alpha_t 归类**：**attention 权重** $w_t$（由内容相似度 + 插值门 $g_t$ + 移位 + 锐化生成）。
- **用物理时间 Δt 还是观测序号**：**观测序号**（每步读/写）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不支持（按步）。
- **有无定理**：无。
- **用于**：算法学习、复制/排序/关联回忆。
- **与候选结构异同**：候选的写 $M^-+\alpha v k^\top$ **缺 erase 项**；NTM 的式 (3)(4) 提供"精确不动"的可能（$w_t(i)=0$ 或 $e_t=0$ 时该位置严格不变，原文："if either the weighting or the erase is zero, the memory is left unchanged"）。候选没有这种逐位置掩码。
- **裁决**：**部分覆盖**（读 + 写 + 擦除 + 稀疏位置选择；无衰减/无 Δt/无双状态）。

### 8. Temporal Graph Networks for Deep Learning on Dynamic Graphs
- **作者 / 年份 / 会议或期刊 / 链接**：Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, Michael Bronstein / 2020 / arXiv 预印本（arXiv:2006.10637；另有 ICML 2020 workshop 版）/ https://arxiv.org/pdf/2006.10637
- **记忆状态**：每个节点一个向量记忆 $s_i(t)\in\mathbb{R}^d$（对象级记忆），外加图结构与时间戳。
- **读操作精确公式**：$z_i(t)=\mathrm{emb}(i,t)$；可用 identity、time projection $z_i(t)=(1+\Delta t\,w)\circ s_i(t)$，或 temporal graph attention，其中 key/value 拼上时间编码：$C^{(l)}(t)=[\,h_1^{(l-1)}(t)\,\Vert\,e_{i1}(t_1)\,\Vert\,\varphi(t-t_1),\dots]$（式 (5)–(9)）。
- **写操作精确公式**：$s_i(t)=\mathrm{mem}\bigl(\bar m_i(t), s_i(t^-)\bigr)$（式 (4)），**仅在涉及该节点的事件发生时**更新；$\bar m_i(t)=\mathrm{agg}(m_i(t_1),\dots,m_i(t_b))$（式 (3)，取最近消息或均值）。
- **是否存在 M←M+内容寻址更新**：**是（事件驱动）**。写入由事件存在性触发，写内容由消息聚合决定（GRU/LSTM）。
- **alpha_t 归类**：**其他**——写入无门控，由"是否有事件"硬触发；写强度由 mem 网络内部决定。
- **用物理时间 Δt 还是观测序号**：**真实物理时间**（事件时间戳 $t$）；$\Delta t$ 是距上次交互的时间。
- **若用 Δt，影响**：**key/value**（时间编码 $\varphi(t-t_i)$ 拼进注意力键）与**读取**（time projection $(1+\Delta t\,w)\circ s_i(t)$ 直接缩放记忆）。不进入写入衰减。
- **支持不规则采样/丢帧/重复帧吗**：**支持**——事件流天然不规则；无事件时段记忆保持不变（原文承认这会导致 memory staleness，需 embedding 模块补偿）。
- **有无定理**：无。
- **用于**：动态图、推荐、社交网络（非视频）。
- **与候选结构异同**：**最接近"只在变化时写"**：无事件节点记忆**严格不变**（硬稀疏、对象级）；$\Delta t$ 进入**读取**而非写入衰减。候选的 $\exp(-\Lambda\Delta t)$ 是"连续衰减"，TGN 是"事件间不衰减、读取时按 $\Delta t$ 缩放"——两种时间语义的直接竞争方案。
- **裁决**：**部分覆盖**（事件驱动硬写 + 对象级稀疏 + 真实 Δt 进 key/read；无矩阵关联记忆、无 exp 衰减）。

### 9. Recurrent Independent Mechanisms
- **作者 / 年份 / 会议或期刊 / 链接**：Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio, Bernhard Schölkopf / 2021 / ICLR 2021（iclr.cc/virtual/2021/poster/3224 检索确认）/ https://arxiv.org/pdf/1909.10893
- **记忆状态**：K 个模块的隐状态 $\{h_{t,k}\}$（模块级记忆，向量集合；每个模块可视为一个"对象/机制"）。
- **读操作精确公式**：输入注意力 $A_k^{(in)}=\mathrm{softmax}\!\Bigl(\frac{h_t W_k^q (X W^e)^\top}{\sqrt{d_e}}\Bigr) X W^v$（式 (2)），其中 $X=\varnothing\oplus x_t$（式 (1)，含 null 元素）。
- **写操作精确公式**：按注意力分数选 top-$k_A$ 模块集合 $\mathcal S_t$（"least attention on the null input"）；对 $k\in\mathcal S_t$：$\tilde h_{t,k}=\mathrm{LSTM}(h_{t,k}, A_k^{(in)})$；对 $k\notin\mathcal S_t$：**$h_{t+1,k}=h_{t,k}$**（原文："The hidden states of RIMs which are not activated ... remain unchanged, acting like untouched memory elements"）。
- **是否存在 M←M+内容寻址更新**：**是**（硬选择后更新）。
- **alpha_t 归类**：**attention 权重 → top-k 硬选择**（选择信号是输入相关性，不是预测残差）。
- **用物理时间 Δt 还是观测序号**：观测序号（步）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不支持（按步）。
- **有无定理**：无。
- **用于**：RL、视频、语言（模块化/泛化/鲁棒性）。
- **与候选结构异同**：**"无关部分严格不变"的明确实现**（硬 top-k + 未选中模块状态逐位不动）；但选择依据是注意力竞争，不是 $z-D(h^-)$ 残差，也没有矩阵关联记忆与 Δt。
- **裁决**：**部分覆盖**（硬稀疏写 + 模块级严格不变；选择判据、记忆形态、时间语义均不同）。

### 10. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- **作者 / 年份 / 会议或期刊 / 链接**：Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean / 2017 / arXiv 预印本（arXiv:1701.06538；常被引作 ICLR 2017，本轮仅核实 arXiv）/ https://arxiv.org/abs/1701.06538（正文取 ar5iv: https://ar5iv.labs.arxiv.org/html/1701.06538）
- **记忆状态**：专家参数（参数记忆；不是状态记忆，仅作稀疏写入的**计算类比**）。
- **读操作精确公式**：$y=\sum_i G(x)_i E_i(x)$。
- **写操作精确公式**：无在线写（训练时梯度更新参数）；路由门 $G(x)=\mathrm{Softmax}(\mathrm{KeepTopK}(H(x),k))$（式 (3)），$H(x)_i=(x\cdot W_g)_i+\mathcal N(0,1)\cdot\mathrm{Softplus}((x\cdot W_{noise})_i)$（式 (4)），$\mathrm{KeepTopK}(v,k)_i=v_i$（若在前 k）否则 $-\infty$（式 (5)）。
- **是否存在 M←M+内容寻址更新**：**否**（是稀疏**计算**，不是记忆写）。
- **alpha_t 归类**：**学出的门**（top-k 硬路由 + softmax 权重）。
- **用物理时间 Δt 还是观测序号**：都无（按样本）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不适用。
- **有无定理**：无（有负载均衡损失）。
- **用于**：语言建模、机器翻译。
- **与候选结构异同**：提供**稀疏的实测计算收益**的唯一硬数据（摘要："greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters"）；但稀疏发生在"用哪个专家"，不在"写不写记忆"。
- **裁决**：**未覆盖**（只作类比；其稀疏收益数据可被候选引用，但不能支持候选的机制）。

---
## 3. 槽位记忆、硬跳过与事件触发

### 11. Object-Centric Learning with Slot Attention
- **作者 / 年份 / 会议或期刊 / 链接**：Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf / 2020 / NeurIPS 2020 / https://arxiv.org/pdf/2006.15055
- **记忆状态**：外部 slots $S\in\mathbb{R}^{K\times D}$（K 个对象槽位）。
- **读操作精确公式**：$\mathrm{attn}_{i,j}=e^{M_{i,j}}/\sum_l e^{M_{i,l}}$，$M=\frac{1}{\sqrt D}k(\text{inputs})q(\text{slots})^\top$（式 (1)，softmax over slots）。
- **写操作精确公式**：$\text{updates}=W^\top v(\text{inputs})$，$W_{i,j}=\mathrm{attn}_{i,j}/\sum_l \mathrm{attn}_{l,j}$（式 (2)）；$\text{slots}=\mathrm{GRU}(\text{state}=\text{slots\_prev},\text{inputs}=\text{updates})$，再 $\text{slots}\mathrel{+}=\mathrm{MLP}(\mathrm{LN}(\text{slots}))$，重复 $T=3$ 次（Algorithm 1）。
- **是否存在 M←M+内容寻址更新**：**部分**——GRU 门控写，新信息是注意力聚合；不是 $v k^\top$ 外积。
- **alpha_t 归类**：**attention 权重**（softmax 分配竞争）；GRU 内部另有门。
- **用物理时间 Δt 还是观测序号**：迭代次数 $T$（与时间无关）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不适用（单帧）。
- **有无定理**：无。
- **用于**：无监督物体发现、集合预测。
- **与候选结构异同**：对象级"记忆"写入，但**每轮全量更新所有 slot**，无衰减、无 Δt；稀疏性来自"输入特征到 slot 的分配竞争"，不是"只更新部分 slot"。后继 SAVi（https://arxiv.org/pdf/2111.12594）把它接到视频，加 predictor-corrector，但 corrector 仍是每帧对全部 slot 重新绑定。
- **裁决**：**部分覆盖**（对象级写；无硬稀疏、无时间、无矩阵关联记忆）。

### 12. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Victor Campos, Brendan Jou, Xavier Giró-i-Nieto, Jordi Torres, Shih-Fu Chang / 2018 / ICLR 2018 / https://arxiv.org/pdf/1708.06834
- **记忆状态**：隐状态向量 $s_t$。
- **读操作精确公式**：无独立读操作；输出由 $s_t$ 解码。
- **写操作精确公式**：$s_t=u_t\cdot S(s_{t-1},x_t)+(1-u_t)\cdot s_{t-1}$（式 (3)），$u_t=f_{\text{binarize}}(\tilde u_t)=\mathrm{round}(\tilde u_t)$（式 (2)），$\Delta\tilde u_t=\sigma(W_p s_t+b_p)$（式 (4)），累积 $\tilde u_{t+1}=u_t\Delta\tilde u_t+(1-u_t)(\tilde u_t+\min(\Delta\tilde u_t,1-\tilde u_t))$（式 (5)）。
- **是否存在 M←M+内容寻址更新**：**否**（是"写或不写"，写入用普通 RNN 状态转移）。
- **alpha_t 归类**：**学出的门**（输入派生的累积概率 → 二值）。
- **用物理时间 Δt 还是观测序号**：观测序号（步）；跳过量 $N_{\text{skip}}$ 也是步数（式 (6)）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不支持（跳过是计算节省，不是时间语义）。
- **有无定理**：无（有 $N_{\text{skip}}$ 的闭式公式）。
- **用于**：序列分类/情感/动作识别。
- **与候选结构异同**：**硬稀疏写的最清晰早期形式**（$u=0$ 时状态逐位不动，甚至可不计算）；但判据不是预测误差、无记忆矩阵、无 Δt。
- **裁决**：**部分覆盖**（硬稀疏写；机制不同）。

### 13. Adaptive Computation Time for Recurrent Neural Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Alex Graves / 2016 / arXiv:1603.08983 / https://arxiv.org/pdf/1603.08983
- **记忆状态**：中间状态序列 $s_t^n$（$n=1..N(t)$）。
- **读操作精确公式**：$y_t=\sum_{n=1}^{N(t)}p_t^n y_t^n$（式 (9)）。
- **写操作精确公式**：$s_t=\sum_{n=1}^{N(t)}p_t^n s_t^n$（式 (9)）；$N(t)=\min\{n':\sum_{n=1}^{n'}h_t^n\ge 1-\epsilon\}$（式 (7)），$h_t^n=\sigma(W_h s_t^n+b_h)$（式 (5)）。
- **是否存在 M←M+内容寻址更新**：**否**（加权平均，不是写）。
- **alpha_t 归类**：**学出的门**（halting 概率，软）。
- **用物理时间 Δt 还是观测序号**：计算步数 $n$（不是时间）。
- **若用 Δt，影响**：不适用（只加 $\delta_{n,1}$ 标志）。
- **支持不规则采样/丢帧/重复帧吗**：不支持。
- **有无定理**：无。
- **用于**：合成算法任务、字符预测。
- **与候选结构异同**：与"记忆写"无关，但提供了**软门控被优化器钻空子**的原文证据（脚注：网络会把 $h_1$ 设到略低于阈值、中间置 0，从而少算却低罚）。
- **裁决**：**未覆盖**（仅作软门控反例）。

### 14. Event-Based State Estimation with Switching Static-Gain Observers
- **作者 / 年份 / 会议或期刊 / 链接**：Sebastian Trimpe, Raffaello D'Andrea / 2012 / 3rd IFAC Workshop on Distributed Estimation and Control in Networked Systems (NecSys'12), pp. 91–96 / http://trimpe.is.tuebingen.mpg.de/publications/trimpe-ECNS12.pdf
- **记忆状态**：状态估计向量 $\check x$（向量；无矩阵记忆）。
- **读操作精确公式**：无读操作（估计器）。
- **写操作精确公式**：$\check x(k\mid k-1)=A\check x(k-1\mid k-1)+Bu(k-1)$（式 (10)）；$\check x(k\mid k)=\check x(k\mid k-1)+\sum_{i\in I(k)}L_i\bigl(y_i(k)-C_i\check x(k\mid k-1)\bigr)$（式 (11)）；触发集 $I(k)=\{i:\lVert y_i(k)-C_i\check x(k\mid k-1)\rVert\ge\delta_i\}$（式 (9)）。无触发时 $\check x(k\mid k):=\check x(k\mid k-1)$。
- **是否存在 M←M+内容寻址更新**：**是**（加性残差写，且写/不写由阈值硬决定）。
- **alpha_t 归类**：**阈值/事件触发**——判据是**预测误差范数**（新息）超过 $\delta_i$，即**surprise 的硬阈值版**。
- **用物理时间 Δt 还是观测序号**：离散步 $k$（$\Delta$ 不显式；步长藏在 $A$ 的离散化里，不允许变步长）。
- **若用 Δt，影响**：不显式。
- **支持不规则采样/丢帧/重复帧吗**：不支持（要求传感器与估计器同步）。
- **有无定理**：**有**。定理 1：若 $(I-LC)A$ 稳定，则 $\lVert e\rVert_\infty\le\frac{m}{1-\rho}\lVert L\rVert\lVert\delta\rVert$；文中明确指出各切换模式可能不可稳、切换稳定性需单独论证。
- **用于**：控制/传感器网络。
- **与候选结构异同**：覆盖"只在变化时写"的判据（阈值）与稳定性条件；无矩阵记忆、无 Δt、写入是线性增益而非外积。
- **裁决**：**部分覆盖**（事件触发硬写 + 稳定性定理；记忆形态与时间语义不同）。

### 15. On Event Based State Estimation
- **作者 / 年份 / 会议或期刊 / 链接**：Joris Sijs, Mircea Lazar / 2009 / HSCC 2009, LNCS 5469, pp. 336–350 / https://pure.tue.nl/ws/files/3301085/Metis230979.pdf
- **记忆状态**：状态向量 $x(t)$（连续时间），协方差 $P$。
- **读操作精确公式**：无独立读；事件到达时做贝叶斯条件化 $p(x_n\mid z_{0:k_e})$（式 (9)）。
- **写操作精确公式**：事件采样 $\ z_{k_e}=(y_{k_e}^\top,t_{k_e})^\top$ 当且仅当 $y(t)\notin H_{k_e}(z_{k_e-1},t)$（式 (6)）；send-on-delta 特例 $H=\{y:-\Delta<y-y_{k_e-1}<\Delta\}$；状态转移 $x(t+\tau)=A_\tau x(t)+B_\tau w(t)$（式 (7a)）。
- **是否存在 M←M+内容寻址更新**：**是**（事件触发更新；无矩阵）。
- **alpha_t 归类**：**阈值**（事件集 $H$ 的边界 $\Delta$）。
- **用物理时间 Δt 还是观测序号**：**真实时间 τ**（连续时间；把事件时刻 $t_{k_e}$ 与同步采样时刻 $t_{k_a}$ 混在同一时间轴）。
- **若用 Δt，影响**：**状态转移** $A_\tau,B_\tau$（真实时间进入转移矩阵），并决定何时更新。
- **支持不规则采样/丢帧/重复帧吗**：**支持**（事件间隔不规则，混合时间轴）。
- **有无定理**：**有**。定理 1：事件基估计器协方差有界（$\lambda_{\max}(P_{n|n})$ 有界）；证明依赖"事件集 $H$ 必须包含事件间所有可能取值"这一设计条件。
- **用于**：传感器网络/控制。
- **与候选结构异同**：**唯一把真实 Δt 放进状态转移的先例**；若候选把 $\Delta t$ 放进矩阵衰减，需与 $A_\tau$ 这条路线做对比论证。
- **裁决**：**部分覆盖**（真实 Δt + 事件触发 + 稳定性；无学习、无矩阵记忆）。

---
## 4. 反对结论与反例

### 4.1 最致命的一条：候选公式里**没有**"事件/稀疏写"机制
- 候选写式 $M_i=M_i^-+\alpha_i v_i k_i^\top$ **每一步都写**。除非 $\alpha_i\equiv 0$（公式未定义 $\alpha_i$ 何时为 0），否则"不变内容不反复重写"**不是公式的性质，只是愿望**。
- 真正实现"只在变化时写"的三种机制在文献里都有名字：**阈值**（Trimpe 式 (8)(9)、Sijs send-on-delta）、**surprise**（Titans 式 (8)(12) 的梯度、TTT 式 (2) 的梯度幅度）、**门/路由**（Mamba 的 $\Delta_t$、Gated DeltaNet 的 $\alpha_t$、Skip RNN 的 $u_t$、RIMs 的 top-$k_A$、MoE 的 top-$k$）。
- 候选的 $\alpha_i$ 属于哪一类未定义。若它只是"学出的门"（最可能），则与 Mamba/Gated DeltaNet 同构，不带来稀疏；若要"不变内容不写"，必须显式引入阈值或 surprise 归一化（Titans/TTT 的定义可直接借用）并做硬掩码。

### 4.2 "只在变化时写"的判据已被四种方式占满（逐条给证据）
| 判据 | 代表 | 公式/原文 | 是否硬 |
|---|---|---|---|
| 预测误差阈值 | Trimpe 2012 | $\lVert y_i-C_i\check x(k\mid k-1)\rVert\ge\delta_i$ 才写（式 (9)） | **硬** |
| 信号变化阈值 | Miskowicz 2006（https://www.mdpi.com/1424-8220/6/1/49）/ Sijs 2009 | $\lvert y(t)-y_{k_e-1}\rvert>\Delta$ 才发 | **硬** |
| 梯度/预测误差幅度（surprise） | Titans 2025 | $S_t=\eta_t S_{t-1}-\theta_t\nabla\ell(\mathcal M_{t-1};x_t)$，$\ell=\lVert\mathcal M(k_t)-v_t\rVert^2$（式 (12)(14)） | 软（连续写） |
| 梯度幅度 | TTT 2025 | $W_t=W_{t-1}-\eta\nabla\ell$，"remembers inputs that produce large gradients"（式 (2)(3)） | 软（连续写） |
| 输入相关门 | Mamba 2023 / Gated DeltaNet 2025 / Skip RNN 2018 | $\Delta_t=\mathrm{softplus}(\cdot)$；$\alpha_t,\beta_t$；$u_t=\mathrm{round}(\tilde u_t)$ | 前二软、Skip RNN 硬 |
| 注意力/top-k 路由 | RIMs 2021 / MoE 2017 / NTM 2014 / Slot Attention 2020 | top-$k_A$ 模块；KeepTopK；$\sum w_t(i)=1$；softmax over slots | RIMs/MoE 硬 |

**结论**：候选若要加"只在变化时写"，无论选哪条判据都已有直接先例；唯一还空着的是"**用 surprise 阈值 + 矩阵记忆 + 真实 Δt 衰减**"的组合。

### 4.3 稀疏到底带来多少实测收益？（只有 MoE 有硬数据）
- **MoE（唯一硬数据）**：摘要原文 "greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters"（https://ar5iv.labs.arxiv.org/html/1701.06538）。注意：收益来自"容量变大而算力近常数"，**不是**"少写记忆省时间"。
- **Skip RNN**：报告的是更新次数下降与计算图缩短，不是墙钟；且二值化需要 straight-through。
- **ACT**：报告 ponder 序列/计算步数下降，用 $\tau$ 惩罚换计算；无墙钟。
- **RIMs**：报告的是"只有被激活的模块做状态更新"，未给端到端墙钟加速（它甚至保留了全部模块参与 attention）。
- **TGN**：事件驱动省的是"无事件节点的更新"，但 embedding 阶段仍需按时序聚合。
- **反例**：没有任何一篇报告过"稀疏**记忆写入**"相对稠密写入的端到端墙钟收益；稀疏在 MoE 里的收益来自**容量-算力**权衡，不是写入次数。候选若要主张"稀疏省算力"，需要自己在目标硬件上测。

### 4.4 软门控能否保证"无关部分严格不变"？不能——反例与正例都很清楚
**软（保证不了）**
- **Mamba**：$\bar A=\exp(\Delta A)$，$\Delta>0$ 有限时 $\bar A$ 的元素恒在 $(0,1)$，**永远不会严格为 0**；每步都在改写全部状态。
- **Gated DeltaNet**：$\alpha_t,\beta_t\in(0,1)$，每步都写；原文把 $\alpha_t\to 0$ 说成"快速清空"，这是"全改"，不是"不改"。
- **Titans**：$(1-\alpha_t)$ 乘法衰减，$\alpha_t\in[0,1]$ 连续，没有零掩码。
- **ACT**：软 halting 的 mean-field 平均；原文脚注自曝优化器会作弊（$h_1$ 略低于阈值 → 末端抬高），说明软门控既可被钻空子、也无法保证"没算"。
- **Slot Attention / SAVi**：所有 K 个 slot 每轮/每帧都被更新；实现上给注意力加小偏置 $\epsilon$ 防止数值问题，进一步确保**非零**更新。

**硬（能保证）**
- **Skip RNN**：$u_t\in\{0,1\}$，$u_t=0$ 时 $s_t=s_{t-1}$ 逐位复制（式 (3)）。
- **RIMs**：原文 "The hidden states of RIMs which are not activated ... remain unchanged"：$h_{t+1,k}=h_{t,k}\ \forall k\notin\mathcal S_t$。
- **Trimpe**：无事件时 $\check x(k\mid k)=\check x(k\mid k-1)$。
- **NTM**：$w_t(i)=0$ 或 $e_t=0$ 时"the memory is left unchanged"（式 (3)）。
- **TGN**：无事件节点的记忆不被更新（代价是 staleness，需 embedding 补偿）。
- **MoE**：非选中专家严格不计算（$-\infty$ 路由）。

**对候选的含义**：在不加硬选择的情况下，$\alpha_i$ 即使很小也不是 0；"不变内容不反复重写"不能由公式推出，只能由实现层掩码保证。

### 4.5 衰减算子 $\exp(-\Lambda\Delta t)$ 不是新发明：它就是 ZOH 离散化
- Mamba 式 (4)：$\bar A=\exp(\Delta A)$。候选的 $\exp(-\Lambda\odot\Delta t)$ 是**同一个算子的对角、带符号版本**（$\Lambda>0$ 对应衰减）。
- Gated DeltaNet 式 (10)：$\alpha_t$ 乘在 $S_{t-1}$ 上，是同一衰减门的标量/学习版本；Titans 式 (13) 的 $(1-\alpha_t)$ 是它的一阶近似（$\exp(-x)\approx 1-x$ 对小块）。
- 因此候选的衰减项应表述为"把 ZOH 衰减的时间常数由学习量换成实测 $\Delta t$"，这**是组合级新意，不是算子级新意**。必须做 Mamba 式 (4) 的对照实验（$\Delta$ 学习 vs $\Delta$ 实测）。

### 4.6 真实 Δt 的落点已有三种，唯独"矩阵衰减"这一格没被占
- **进状态转移**：Sijs & Lazar 2009 的 $A_\tau$（式 (7a)）——最接近"物理时间改变动力学"。
- **进读取/键**：TGN 的 time projection $(1+\Delta t\,w)\circ s_i(t)$ 与时间编码 $\varphi(t-t_i)$（式 (5)–(9)）。
- **进衰减与写强度**：Mamba 的 $\Delta$（但那是学习量，不是实测）。
- **结论**：候选把 $\Delta t$ 放进"逐通道矩阵衰减指数"这一格确实没有直接先例；但它是上述三者的直接插值，审稿人会问"为什么不是 TGN 的 time projection，也不是 Sijs 的 $A_\tau$"。

### 4.7 双状态（M, h）的先例很多，但"同一观测同时写两个状态"的严格对照没有
- Titans：长期记忆 $\mathcal M$ + core（短窗 attention）+ persistent memory——多状态，但 core 不是"由 M 的读出驱动"。
- TGN：节点记忆 $s_i(t)$ + embedding $z_i(t)$（读出）——事实上的双状态。
- DeltaNet / Gated DeltaNet：**单状态**（只有 $W$/$S$）——反例：矩阵关联记忆不需要第二个 h。
- Slot Attention / SAVi：slots + GRU 隐状态。
- 候选的 h 由 $(h_{i-1}, r_i, z_i)$ 更新、$r_i=R(M_i,q_i)$——若 $q_i$ 由 $h_i$ 生成，构成闭环"读-写-读"，这在上述工作里**没有完全同构**的实现；这是候选可以主张的第二点结构差异（但要证明闭环稳定，见 4.8）。

### 4.8 事件触发更新的稳定性条件（必须写进论文）
- **Trimpe 定理 1**：$(I-LC)A$ 稳定 ⇒ $\lVert e\rVert_\infty\le\frac{m}{1-\rho}\lVert L\rVert\lVert\delta\rVert$。两个前提：(i) 参考估计器稳定；(ii) 触发阈值有限。文中明确警告：各切换模式**可能不可稳**，且"每个模式稳定"**不蕴含**切换系统稳定。
- **Sijs & Lazar 定理 1**：协方差有界的前提是事件集 $H$ "应包含事件间所有可能取值"——即触发设计必须与信号可达集匹配；否则事件间无界漂移。
- **对候选的直接推论**：(a) $\exp(-\Lambda\Delta t)$ 在 $\Delta t\to\infty$ 时把记忆清零（信息丢失），在 $\Delta t\to 0$ 时退化为不衰减（与连续时间极限一致），需要给出 $\Delta t$ 的上下界与对应的误差界；(b) 若 $\alpha_i$ 稀疏化，"漏写"会带来与阈值同阶的误差（Trimpe 的界 ∝$\lVert\delta\rVert$）；(c) 半群一致性：$\exp(-\Lambda\Delta t_1)\exp(-\Lambda\Delta t_2)=\exp(-\Lambda(\Delta t_1+\Delta t_2))$ 只在 $\Lambda$ 为常数时成立——逐通道学习且随输入变化时**不满足**，这会被直接质疑"Δt 语义不成立"。这是候选最需要先回答的数学问题。

### 4.9 逐部件裁决汇总
| 候选部件 | 最近先例 | 覆盖度 | 剩余空间 |
|---|---|---|---|
| $\exp(-\Lambda\odot\Delta t)\odot M$ | Mamba 式 (4) $\bar A=\exp(\Delta A)$ | **算子已覆盖** | 仅"Δt=实测" |
| $+\alpha v k^\top$ 写 | DeltaNet 式 (24)、Gated 式 (10)、NTM 式 (3)(4)（带 erase） | **已覆盖** | erase 项缺失 |
| $\alpha$ 写强度 | Mamba $\Delta$、Gated $\beta$、Skip RNN $u$、Titans $\alpha_t$ | **已覆盖** | 未定义 |
| $R(M,q)$ 读 | Hopfield 式 (3)（=attention）、NTM 式 (2)、TGN embedding | **已覆盖** | 无 |
| 双状态 $M+h$ | Titans、TGN、Slot Attention | **概念已覆盖** | 闭环耦合未验证 |
| 真实 $\Delta t$ | Sijs $A_\tau$、TGN time enc、Mamba（学习 $\Delta$） | **部分** | 矩阵衰减格 |
| "只在变化时写" | Trimpe、RIMs、TGN、Skip RNN、MoE、NTM | **已有大量先例** | 候选**根本没实现** |

### 4.10 结论（给 lead）
1. **有无逐字对应工作**：**没有单篇同时具备"$\exp(-\Lambda\Delta t)$ 矩阵衰减 + 外积写 + 真实 Δt"**。但逐部件全部有先例，且最接近的三篇各自只差一格：Titans（矩阵记忆 + 学习衰减 + 关联写，缺真实 Δt）、Gated DeltaNet（矩阵记忆 + 衰减 + 外积写，缺真实 Δt）、TGN（真实 Δt + 事件写，缺矩阵记忆与 exp 衰减）。
2. **最致命的三条反对证据**：(i) 候选公式**没有稀疏/事件机制**，"不变内容不反复重写"无法从公式推出，而该机制的三种判据（阈值、surprise、门控）都已有成熟先例且带稳定性理论；(ii) $\exp(-\Lambda\Delta t)$ 就是 ZOH 离散化，Mamba 式 (4) 已写死，属"换时间常数"而非新算子；(iii) 软门控不保证严格不变（Mamba $\bar A$、Gated $\alpha,\beta$、Titans 损失项），真正严格不变的都是硬机制（Skip RNN、RIMs、Trimpe、TGN、NTM 的 $w=0$）。
3. **还能主张什么**（三条，按强度排序）：(a) **真实 $\Delta t$ 进逐通道矩阵衰减**这一格确实空着——但必须解决半群一致性与 $\Delta t$ 上下界；(b) **读-写-读闭环**的双状态耦合（$q=Q(z,h)$ 由 h 生成、M 的读出 $r$ 又更新 h）在所列文献中无同构实现——需要稳定性分析；(c) **把 surprise 阈值用于矩阵记忆的逐键稀疏写**（"只在 $r$ 超过阈值时写某个 key"），NTM 的 erase-add + Titans 的 surprise 都没做到逐键硬选择。
4. **不要主张**："attention/快权重当可寻址关联记忆"（Hopfield 式 (3) 已证明等价）、"按时间衰减"（ZOH/Mamba/Gated）、"稀疏更新省算力"（只有 MoE 有硬数据，且不在写入上）。

---

## 附：核实日志（本轮每条链接均实际拉取）

| 条目 | 拉取方式 | 结果 |
|---|---|---|
| Mamba arXiv PDF | tavily_extract | ✅ 式 (1)(2)(4)、Algorithm 1/2、§3.2 选择机制原话 |
| Titans arXiv PDF + HTML v1 | tavily_extract | ✅ 式 (8)(9)(10)(11)(12)(13)(14)、§3.1 原话 |
| TTT arXiv HTML v4 + PMLR 267 页 | tavily_extract | ✅ 式 (1)(2)(3)、12 位作者、ICML 2025 页码 |
| DeltaNet arXiv HTML v3 + PMLR 139 页 | tavily_extract | ✅ 式 (20)–(25)、§4.1 crosstalk、作者 |
| Gated DeltaNet arXiv abs + ICLR 官方 PDF | tavily_extract | ✅ 式 (10)、§2.2、作者 |
| Hopfield arXiv abs + CMU 课程 PDF | tavily_extract | ✅ 式 (2)(3)(10)、定理 1、容量定理清单、作者 |
| NTM arXiv PDF | tavily_extract | ✅ 式 (1)–(4)、原文"memory left unchanged"句 |
| TGN arXiv PDF | tavily_extract | ✅ 式 (3)(4)(5)–(9)、作者 |
| RIMs arXiv PDF | tavily_extract | ✅ 式 (1)(2)、"remain unchanged"原话、作者 |
| MoE arXiv abs + ar5iv 全文 | tavily_extract | ✅ 式 (2)(3)(4)(5)、>1000x 原话、7 位作者 |
| Slot Attention arXiv PDF | tavily_extract | ✅ Algorithm 1、式 (1)(2) |
| SAVi（仅作 Slot Attention 后继提及） | tavily_extract（上一轮） | ✅ 式 (1)–(4) 已核 |
| Skip RNN arXiv PDF | tavily_extract | ✅ 式 (1)–(6)、作者 |
| ACT arXiv PDF | tavily_extract | ✅ 式 (5)(7)(9)(10)–(12)、作弊脚注 |
| Trimpe IFAC PDF | tavily_extract | ✅ 式 (9)(10)(11)、定理 1 |
| Sijs & Lazar HSCC PDF | tavily_extract | ✅ 式 (6)(7)、send-on-delta 集合、定理 1 |

**未核实/降级说明**：① MoE 的正文取自 ar5iv 镜像（arXiv PDF 直连抽取为空），标题/摘要/公式与官方版本一致；② Titans/TTT/RIMs 的 HTML 版作者块被抽取器剥离，作者改从 PDF/PMLR 页核对；③ Hopfield 与 RIMs 的会议归属由 OpenReview/ICLR 虚拟页检索确认，正文链接均用 arXiv；④ 无任何条目只靠摘要写公式。


# 04 连续时间/不规则采样下的记忆与衰减 —— 逐篇核对

> 范围：只做网络检索与文献核对（tavily_extract，公式取自 PDF/官方 proceedings 正文）。
> 被检验候选结构（ASCII）：
> ```
> M_i- = exp(-Lambda * Delta_i) ⊙ M_{i-1}       # 矩阵快权重关联记忆，按真实时间间隔衰减
> q_i = Q(z_i, h_i); k_i = K(z_i, h_i); v_i = V(z_i, h_i)
> M_i = M_i- + alpha_i * v_i k_i^T              # 内容寻址写入
> r_i = R(M_i, q_i)                              # 读
> h_i = H(h_{i-1}, r_i, z_i)                     # 另一路隐状态（双状态）
> ```
> 用户主张：attention/快权重=可寻址关联记忆；不变内容不反复重写；变化内容经 key 找到并更新；衰减与写入由真实经过时间 Δt 控制而非帧数。
> 核查状态：以下公式均从 PDF 正文直接抓取；唯一标 [细节未核实] 的是 TiM4Rec 中 d_i 是否做归一化（正文所取章节未说明）。

## 0. 速查表（先给结论）

| # | 论文 | 记忆状态 | alpha_t 归类 | 物理 Δt? | Δt 落点 | 矩阵记忆? |
|---|---|---|---|---|---|---|
| 1 | Mamba 2023 | 向量 h∈R^N/通道 | 学出的门（softplus 参数化） | 否（输入依赖学习量） | 衰减（exp(ΔA)） | 否 |
| 2 | TiM4Rec 2024 | 矩阵（SSD 状态） | 外部真实时间差（进衰减） | 是（unix 时间差） | 衰减（掩码） | 是 |
| 3 | CRU 2022 | 向量（均值+协方差） | 预测误差/协方差加权（Kalman 增益） | 是 | 状态转移+间接写入强度 | 否 |
| 4 | GRU-D 2018 | 向量 | 学出的门（γ=exp(-max(0,Wδ+b))） | 是 | 衰减 | 否 |
| 5 | Ba fast weights 2016 | 矩阵 A∈R^{d×d} | 其他（固定 λ、学习率 η） | 否（按步） | 衰减 | 是 |
| 6 | Schlag 2021 | 矩阵（快权重） | attention 权重 | 否 | 无衰减 | 是 |
| 7 | RetNet 2023 | 矩阵 S∈R^{d×d} | 其他（固定 γ，按头） | 否（按步） | 衰减 | 是 |
| 8 | RWKV 2023 | 向量（WKV） | 其他（学出的通道衰减 w，按相对步） | 否（按步） | 衰减 | 否 |
| 9 | GLA 2023 | 矩阵 | 学出的门（2D forget gate） | 否 | 衰减 | 是 |
| 10 | Gated DeltaNet 2025 | 矩阵 | 学出的门（α,β 数据依赖） | 否 | 衰减 | 是 |
| 11 | Titans 2025 | 矩阵/MLP | 学出的门+surprise（梯度/预测误差） | 否（按 token） | 衰减 | 是 |
| 12 | ContiFormer 2023 | 无显式记忆 | attention 权重 | 是（连续轨迹） | 状态转移（Q/K/V 轨迹） | 否 |
| 13 | TiSASRec 2020 | 无显式记忆 | attention 权重 | 是（相对间隔） | 只拼进输入 | 否 |

先行判定（检索范围内）：**没有任何一篇同时用真实秒级 Δt 控制矩阵记忆的衰减与写入**。最接近的是 TiM4Rec（Δ 只控制衰减/掩码）与 CRU（Δ 控制向量状态衰减，并经协方差影响 Kalman 增益=写入强度），但都不是"矩阵 + key 寻址写入 + Δ 控写入"。

## 1. 逐篇

### 1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- 作者 / 年份 / 会议或期刊 / 链接：Albert Gu, Tri Dao / 2023 / arXiv:2312.00752（ICLR/COLM 出版信息未在 PDF 首页核实，按 arXiv 计）/ https://arxiv.org/abs/2312.00752
- 记忆状态：向量。连续 SSM `h'(t)=A h(t)+B x(t)`，`h(t)∈R^N`，每通道 D 个，共 (B,L,D,N)——不是矩阵记忆。
- 读操作精确公式：`y(t) = C h(t)`（式 1b）；离散 `y_t = C h_t`（式 2b）。
- 写操作精确公式：`h_t = Ā h_{t-1} + B̄ x_t`（式 2a），其中 ZOH 离散化（式 4）：`Ā = exp(ΔA)`，`B̄ = (ΔA)^{-1}(exp(ΔA) − I)·ΔB`。
- 是否存在 M(t+1)=F 或 M←M+内容寻址更新：是（加性线性递推），但**无内容寻址**：B 由输入投影得到，不是"用 key 找到旧内容再改"。
- alpha_t 归类：**学出的门**。原文 Algorithm 2：`Δ:(B,L,D) ← τ_Δ(Parameter + s_Δ(x))`，`s_Δ(x)=Broadcast_D(Linear_1(x))`，`τ_Δ=softplus`；§3.5.1 明说"Δ in SSMs can be seen to play a generalized role of the RNN gating mechanism"，Theorem 1 给出 N=1,A=−1,B=1 时 `g_t=σ(Linear(x_t)); h_t=(1−g_t)h_{t−1}+g_t x_t`。
- 用物理时间 Δt 还是观测序号：**观测序号**。Δ 是当前输入 x 的函数，与真实采样时刻无关；序列位置变则 Δ 变，同一内容在不同时刻 Δ 不变——这正是"不是物理时间"的铁证。
- 若用 Δt，影响：不适用（Δ 不是物理时间；只有"衰减"这一项与候选形式相同）。
- 支持不规则采样/丢帧/重复帧吗：不支持。模型没有时间戳输入；丢帧与重复帧都表现为相邻 token，Δ 无法区分。
- 有无定理：有 Thm 1（选择性 SSM 与 RNN 门控等价）及表达力讨论；**无**半群/采样率不变/容量定理。
- 用于：语言/通用序列建模（非视频/控制）。
- 与候选结构异同：衰减项 `exp(ΔA)`（向量指数）≈候选的 `exp(-ΛΔt)` 形式；写入项是加性 `B̄x` 而非 `α v k^T`；读是 `C h` 而非 `R(M,q)`；无双状态；Δ 是学习门不是物理时间。**"exp(ΔA)+加性写"的骨架在 2023 年已标准化**。
- 裁决：**部分覆盖**（衰减+加性写入的形式；Δ 语义完全不覆盖；无矩阵、无 key 寻址）。

### 2. TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model
- 作者 / 年份 / 会议或期刊 / 链接：Hao Fan, Mengyi Zhu, Yanrong Hu, Hailin Feng, Zhijie He, Hongjiu Liu, Qingyang Liu / 2024 / arXiv:2409.16182 / https://arxiv.org/abs/2409.16182
- 记忆状态：**矩阵**。SSD/Mamba2 的 chunkwise 状态 `S[t]∈R^{d_v×d_k}`；同时保留 SSM 连续形式 `h'(t)=A h(t)+B x(t), y=C h(t)`（式 1）。
- 读操作精确公式：SSD 形式 `Y = (QK^T ⊙ L)V`（式 8），或 `O[t]=Q[t]S^T[t]+(QK^T⊙M)V`；等价于 `Y=Q(K^T V)` 的线性注意力。
- 写操作精确公式：标准 SSD 写 `S[t+1]=S[t]+V[t]K^T[t]`（**与时间无关**）；时间感知只改掩码：取相对时间差矩阵 `d̂_mn=t_m−t_n`（式 11），只取次对角线 `d_i=d̂_{i,i−1}=t_i−t_{i−1}`，再 `Â_i = d_i·a_i`（把时间差乘进标量化衰减 `a_i`），即掩码等价于 `L_{ij}=Π a_k^{d_k}` 的 1-SS 矩阵。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**部分**。矩阵加性写存在；衰减由真实时间差驱动；但写入强度、写入内容都不含 Δ，也没有"用 key 检索旧内容"的擦除。
- alpha_t 归类：**外部真实 Delta t**（严格说 Δ 只进衰减项；写入没有 α）。原文明言目标是"把 time difference matrix 直接高效地并入 masked matrix"。
- 用物理时间 Δt 还是观测序号：**物理时间**（unix 时间戳差 t_i−t_{i−1}）。[细节未核实]：所取正文段落未说明 d_i 是否做归一化/缩放，按 unix 秒直接乘进衰减在数值上会造成下溢，原文提到"Due to the presence of layer normalization, the actual value of d1 will be non-zero"。
- 若用 Δt，影响：**仅衰减**（掩码），不影响写入强度、key/value。
- 支持不规则采样/丢帧/重复帧吗：是（时间差异构）；重复帧 `d_i=0` 表示不衰减，丢帧 `d_i` 大表示强衰减——但**重复帧在每个时间步仍会执行一次矩阵加性写**（写不受 Δ 控制），这正是候选想避免的"不变内容反复重写"。
- 有无定理：无。只有复杂度分析（线性）与 SSD 对偶性的引用；无半群/采样细化/容量定理。
- 用于：推荐系统（序列推荐），不是视频/控制。
- 与候选结构异同：**衰减项最接近**：矩阵记忆 + 真实秒级 Δ 控制衰减（甚至就是"log 域乘 Δ"）；但写入项与候选不同（无 α_i 门、无 key 寻址、无擦除），读是 Q/K/V attention 而非 `R(M,q)`，无双状态 h。
- 裁决：**部分覆盖**——"物理 Δ 控制矩阵记忆衰减"这一条已被 2024 年的推荐系统工作做掉；"Δ 控制写入"未被覆盖。

### 3. Modeling Irregular Time Series with Continuous Recurrent Units（CRU）
- 作者 / 年份 / 会议或期刊 / 链接：Mona Schirmer, Mazin Eltayeb, Stefan Lessmann, Maja Rudolph / 2022 / ICML 2022（PMLR v162）/ https://arxiv.org/abs/2111.11344 ；https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf
- 记忆状态：**向量**（隐状态 z∈R^M 的均值与协方差 M×M；协方差是矩阵，但状态本身是向量分布，不是外积记忆）。
- 读操作精确公式：decoder `[o_t, σ^out_t]=g_φ(µ^+_t, Σ^+_t)`（式 6）；即从后验分布解码，无 `R(M,q)` 形式。
- 写操作精确公式：预测 `µ^-_t=exp(A(t−τ))µ^+_τ`（式 8）；更新 `µ^+_t=µ^-_t+K_t(y_t−Hµ^-_t)`（式 11）、`Σ^+_t=(I−K_tH)Σ^-_t`（式 12），`K_t=Σ^-_t H^T(HΣ^-_t H^T+Σ^obs_t)^{-1}`（式 13）。
- 是否存在 M(t+1)=F(...)：**部分**。是"矩阵指数转移 + 观测空间新息修正"，但写的是向量均值/协方差，不是 `M+α v k^T`，也无 key 寻址。
- alpha_t 归类：**预测误差/协方差加权**（Kalman 增益 K_t；它由先验协方差与观测噪声决定，Δ 通过 Σ^- 增长间接放大写入增益）。不是 sigmoid 门，也不是显式 `1−exp(-λΔt)`。
- 用物理时间 Δt 还是观测序号：**物理时间**（t−τ(t)，τ 为上次观测时刻）。
- 若用 Δt，影响：**状态转移（exp(AΔt)）** + **间接影响写入强度**（Σ^- 随 Δ 增长 → K 变化）；不影响 key/value。
- 支持不规则采样/丢帧/重复帧吗：是（任意观测时刻）；观测缺失时纯预测，重复观测会连续两次 Kalman 更新（二次计权）。
- 有无定理：有（连续-离散 Kalman 滤波的闭式最优性；指数族参数化；无半群/采样细化定理）。
- 用于：不规则时间序列（电子病历、气候、图像序列）。
- 与候选结构异同：**Δ + 观测空间新息修正**与候选的 `r=z−D(h^-); h=h^-+B(h^-,r)` 神似；但记忆是向量+协方差，不是矩阵快权重，读不是 `R(M,q)`，无双状态 h。
- 裁决：**部分覆盖**——"真实 Δ 同时影响衰减与写入增益"在**向量状态**上已实现（2022）；矩阵记忆版本没有。

### 4. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- 作者 / 年份 / 会议或期刊 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / Scientific Reports 8:6085（DOI 10.1038/s41598-018-24271-9）/ https://arxiv.org/abs/1606.01865 ；https://www.nature.com/articles/s41598-018-24271-9
- 记忆状态：**向量**（GRU 隐状态 h）。
- 读操作精确公式：隐状态直接进分类/回归头；无独立读操作。
- 写操作精确公式：`γ_t=exp{−max(0,W_γδ_t+b_γ)}`（式 4）；输入 `x^d_t←m^d_t x^d_t+(1−m^d_t)γ^d_{x,t}x^d_{t'}+(1−m^d_t)(1−γ^d_{x,t})x̃^d`（式 5）；隐状态 `h_{t−1}←γ_{h,t}⊙h_{t−1}`（式 6）；再接标准 GRU 门。
- 是否存在 M(t+1)=F(...)：**部分**（乘性衰减 + GRU 写），无内容寻址。
- alpha_t 归类：**学出的门**（γ 由时间间隔 δ 经可学习 W_γ,b_γ 与 exp 产生）。形状上是 `exp(−max(0,Wδ+b))`，与候选 `exp(−ΛΔt)` 同构，但作用在向量上。
- 用物理时间 Δt 还是观测序号：**物理时间**（δ_t = 距上次观测的间隔）。
- 若用 Δt，影响：**仅衰减**（输入变量与隐状态），不影响写入强度/门控值（GRU 的门只看当前输入与 mask）。
- 支持不规则采样/丢帧/重复帧吗：是（逐变量 mask + δ）。
- 有无定理：无。
- 用于：医疗/传感器时序分类。
- 与候选结构异同：Δ 进指数衰减（同形式），但无矩阵、无 key、无读出、无双状态；写入完全不受 Δ 控制。
- 裁决：**未覆盖**（只覆盖"Δ→衰减"这一条，且是 2018 年已做）。

### 5. Using Fast Weights to Attend to the Recent Past
- 作者 / 年份 / 会议或期刊 / 链接：Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu / 2016 / NIPS 2016 / https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past （PDF：https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf）
- 记忆状态：**矩阵** `A∈R^{d×d}`（快权重）。
- 读操作精确公式：内循环 `h_{s+1}(t+1)=f([Wh(t)+Cx(t)] + A(t)h_s(t+1))`（式 2），即 `A h` 作为关联读出。
- 写操作精确公式：`A(t)=λA(t−1)+η h(t)h(t)^T`（式 1）；闭式 `A(t)=ηΣ_{τ=1}^{t}λ^{t−τ}h(τ)h(τ)^T`（式 3）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**是**（`M←λM+外积`），是候选结构最直接的结构祖先；但写用的是 h 自身，没有独立的 q/k/v 投影，也没有"用 key 检索再更新"。
- alpha_t 归类：**其他**（固定衰减率 λ 与固定学习率 η；λ 是超参数，不随输入或时间变化）。
- 用物理时间 Δt 还是观测序号：**观测序号**（每个时间步更新一次，λ 按步衰减）。
- 若用 Δt，影响：不适用（无物理时间）。
- 支持不规则采样/丢帧/重复帧吗：否。步数=时间，采样率变化会直接改变衰减行为。
- 有无定理：无（只有闭式展开式 3）；无半群/容量定理。
- 用于：序列/语言建模（"注意最近过去"）。
- 与候选结构异同：**矩阵记忆 + 乘性衰减 + 外积写入 + 加性读出**几乎就是候选的骨架；差别是 λ 固定而非 `exp(-ΛΔt)`、写不是 `α v k^T` 且无独立 key、无双状态 h。
- 裁决：**部分覆盖**（结构骨架全覆盖，Δ 与门控不覆盖）。

### 6. Linear Transformers Are Secretly Fast Weight Programmers
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021（PMLR v139）/ https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf （arXiv:2102.11174）
- 记忆状态：**矩阵**（快权重/线性注意力状态）。
- 读操作精确公式：线性注意力输出 `y_t=φ(q_t)^T W_{t−1}`（读快权重）。
- 写操作精确公式：加性外积 `W_t=W_{t−1}+φ(k_t)φ(v_t)^T`；总结/更新规则来自"programming instructions = additive outer products"（原文摘要）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**是**（纯加性外积，无衰减、无擦除）。
- alpha_t 归类：**attention 权重**（softmax 归一化后的注意力系数，不是门控状态更新）。
- 用物理时间 Δt 还是观测序号：**观测序号**。
- 若用 Δt，影响：不适用。
- 支持不规则采样吗：否。
- 有无定理：有（线性注意力与快权重编程器的形式等价）；无衰减/半群定理。
- 用于：语言/序列。
- 与候选结构异同：为"attention=可寻址快权重关联记忆"提供理论等价；但记忆无限累加（无衰减）、无 Δ、无双状态 h。
- 裁决：**部分覆盖**（"attention 当关联记忆"的主张已被形式化，2016/2021 既有）。

### 7. Retentive Network: A Successor to Transformer for Large Language Models（RetNet）
- 作者 / 年份 / 会议或期刊 / 链接：Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei / 2023 / arXiv:2307.08621 / https://arxiv.org/abs/2307.08621
- 记忆状态：**矩阵** `S_n∈R^{d×d}`（每头）。
- 读操作精确公式：`o_n=Q_n S_n`（式 6）；并行形式 `Retention(X)=(QK^T⊙D)V`（式 5），`D_{nm}=γ^{n−m}`。
- 写操作精确公式：`S_n=γS_{n−1}+K_n^T V_n`（式 6）；`A=Λ(γe^{iθ})Λ^{-1}`（式 1）对角化后为标量衰减。
- 是否存在 M(t+1)=F(...)：**是**（矩阵衰减递推 + 外积写）。
- alpha_t 归类：**其他**（固定衰减 γ，按头取 `γ=1−2^{-5−i}`；不随输入或时间变化）。
- 用物理时间 Δt 还是观测序号：**观测序号**。
- 若用 Δt，影响：不适用（γ 按相对位置 n−m 衰减）。
- 支持不规则采样吗：否。
- 有无定理：有（递归/并行/分块三种表示等价；无半群/采样定理）。
- 用于：语言。
- 与候选结构异同：矩阵记忆+固定指数衰减+外积写，骨架一致；无 Δ、无 2D 门、无双状态 h；读是 `Q S`。
- 裁决：**部分覆盖**（结构与衰减，无物理时间）。

### 8. RWKV: Reinventing RNNs for the Transformer Era
- 作者 / 年份 / 会议或期刊 / 链接：Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman 等 / 2023 / arXiv:2305.13048 / https://arxiv.org/abs/2305.13048
- 记忆状态：**向量**（WKV 通道向量；不是矩阵）。
- 读操作精确公式：`wkv_t = [Σ_{i=1}^{t−1}e^{−(t−1−i)w+k_i}⊙v_i + e^{u+k_t}⊙v_t] / [Σ_{i=1}^{t−1}e^{−(t−1−i)w+k_i}+e^{u+k_t}]`（式 16）。
- 写操作精确公式：上式即"带指数衰减的累加"（分子分母同步累加），无独立外积写。
- 是否存在 M(t+1)=F(...)：**部分是**（指数衰减累加），无矩阵外积。
- alpha_t 归类：**其他**（学出的通道衰减向量 w；按相对步数 `(t−i)` 衰减，非门控标量）。
- 用物理时间 Δt 还是观测序号：**观测序号**。原文式 (10)：`w_{t,i}=−(t−i)w`，"channel-wise time decay vector multiplied by the relative position"。
- 若用 Δt，影响：不适用。
- 支持不规则采样吗：否。
- 有无定理：有（数值稳定性相关证明，附录 H）；无半群/采样定理。
- 用于：语言。
- 与候选结构异同：名字里有 "time decay"，但衰减按 token 步数；向量状态、无双状态、无 Δ。
- 裁决：**未覆盖**（只覆盖"衰减"直觉）。

### 9. Gated Linear Attention Transformers with Hardware-Efficient Training（GLA）
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim / 2023 / arXiv:2312.06635 / https://arxiv.org/abs/2312.06635
- 记忆状态：**矩阵** `S_t∈R^{d_k×d_v}`。
- 读操作精确公式：线性注意力读出 `o_t=S_t q_t` ——[未核实：本次 PDF 抽取只定位到 recurrent 状态更新式，读出式在抽取文本中缺失；该形式为线性注意力的标准读出]。
- 写操作精确公式：`S_t=G_t⊙S_{t−1}+k_t^T v_t`（2D forget gate `G_t∈(0,1)^{d_k×d_v}`）；先前的非数据依赖版本为 `S_t=γS_{t−1}+k_t^T v_t`。其 Table 1 把 Mamba/Mamba-2/mLSTM/HGRN-2/RWKV-6/GateLoop/GLA 的门统一写成输入经 sigmoid/softplus 的参数化（例：Mamba `G_t=exp(−(1^Tα_t)⊙exp(A)), α_t=softplus(x_tW_{α1}W_{α2})`；Mamba-2 `γ_t=exp(−softplus(x_tW_γ)exp(a))`），**无一项含真实时间差**。
- 是否存在 M(t+1)=F(...)：**是**（逐元素门控衰减 + 外积写）。
- alpha_t 归类：**学出的门**（`G_t` 由输入经线性层+sigmoid 产生，逐元素；Table 1 中全部同类模型也都是这种参数化）。
- 用物理时间 Δt 还是观测序号：**观测序号**。
- 若用 Δt，影响：不适用。
- 支持不规则采样吗：否。
- 有无定理：无（只有 chunkwise 等价/复杂度）。
- 用于：语言。
- 与候选结构异同：矩阵记忆 + 逐元素衰减门 + 外积写；G_t 是学习门而非 `exp(-ΛΔt)`；无双状态。
- 裁决：**部分覆盖**（最接近"逐通道衰减矩阵记忆"，但时间语义是步）。

### 10. Gated Delta Networks: Improving Mamba2 with Delta Rule（Gated DeltaNet）
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Jan Kautz, Ali Hatamizadeh / 2025 / ICLR 2025（PDF 首页："Published as a conference paper at ICLR 2025"）/ https://arxiv.org/abs/2412.06464
- 记忆状态：**矩阵** `S_t∈R^{d_v×d_k}`。
- 读操作精确公式：`o_t=S_t q_t`；并行形式 `O=((QK^T)⊙Γ)V`，`Γ_{ij}=γ_i/γ_j`（i≥j）。
- 写操作精确公式：`S_t=α_t S_{t−1}(I−β_t k_t k_t^T)+β_t v_t k_t^T`（式 10；其中 `α_t∈(0,1)` 数据依赖标量，`β_t` 学习标量）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**是**，且这是候选写入项的最近先例：`(I−β k k^T)` 擦除旧关联 = "不反复重写不变内容"，delta rule 只写"误差方向"。
- alpha_t 归类：**学出的门**（α_t、β_t 均由线性投影产生，数据依赖）。
- 用物理时间 Δt 还是观测序号：**观测序号**。
- 若用 Δt，影响：不适用（α 是门）。
- 支持不规则采样吗：否。
- 有无定理：有（Table 1 的统一在线学习目标分析：LA/Mamba2/Longhorn/DeltaNet 等的更新规则都是某目标闭式解；chunkwise 等价）；无采样/半群定理。
- 用于：语言。
- 与候选结构异同：矩阵记忆+门控衰减+delta 写（含擦除）；无 Δ、无双状态 h。
- 裁决：**部分覆盖**——写入项（含"不重复重写"性质）已有强先例；时间语义完全不同。

### 11. Titans: Learning to Memorize at Test Time
- 作者 / 年份 / 会议或期刊 / 链接：Ali Behrouz, Peilin Zhong, Vahab Mirrokni / 2025 / NeurIPS 2025（官方 proceedings PDF）/ https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf （arXiv:2501.00663）
- 记忆状态：**矩阵**（线性时 `M=W∈R^{d_in×d_in}`），或深 MLP 记忆 `L_M≥1`。
- 读操作精确公式：`y_t=M^*(q_t)`（式 15），`q_t=x_t W_Q`。
- 写操作精确公式：`M_t=(1−α_t)M_{t−1}+S_t`（式 13）；`S_t=η_t S_{t−1}−θ_t∇ℓ(M_{t−1};x_t)`（式 14）；`ℓ(M_{t−1};x_t)=||M_{t−1}k_t−v_t||²`（式 12），`k_t=x_tW_K, v_t=x_tW_V`（式 11）。
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**是**（衰减+动量+梯度写；写入目标是键值预测误差/梯度 = surprise）。
- alpha_t 归类：**学出的门 + surprise**。α_t∈[0,1] 是数据依赖门（"adaptive forgetting"）；写入强度由 surprise（梯度=预测误差）决定。原文自述"later we show this weight decay mechanism is closely related to gating mechanisms in modern RNNs"。
- 用物理时间 Δt 还是观测序号：**观测序号**（按 token/序列位置，α_t 是 x_t 的函数）。
- 若用 Δt，影响：不适用。
- 支持不规则采样吗：否（没有时间戳输入）。
- 有无定理：有（与 RNN 门控/weight decay 的联系、并行化分析）；无采样/半群定理。
- 用于：语言/长序列。
- 与候选结构异同：矩阵记忆+门控衰减+误差驱动写，几乎覆盖候选的 `M^- + α v k^T` 与"不重复写"直觉；但没有物理 Δ、没有显式 key 检索（写直接算 gradient）、无双状态 h（memory 与主干网络是两个模块，但 h 的更新不经过读出的 r）。
- 裁决：**部分覆盖**（结构与写入动机全覆盖；Δ 语义不覆盖）。

### 12. ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling
- 作者 / 年份 / 会议或期刊 / 链接：Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li / 2023 / NeurIPS 2023（neurips.cc/virtual/2023/poster/71304）/ https://arxiv.org/abs/2402.10635
- 记忆状态：**无显式记忆状态**（对连续 Q/K/V 轨迹做注意力）。
- 读操作精确公式：`k_i(t)=k_i(t_i)+∫_{t_i}^{t}f(τ,k_i(τ);θ_k)dτ`，v 同理（式 1）；连续内积 `α_i(t)=∫_{t_i}^{t}q(τ)·k_i(τ)^T dτ/(t−t_i)`（式 3），q(t) 用自然三次样条插值。
- 写操作精确公式：无（没有可写的记忆矩阵/向量）。
- 是否存在 M(t+1)=F(...)：**否**。
- alpha_t 归类：**attention 权重**（连续化后的 softmax/内积权重）。
- 用物理时间 Δt 还是观测序号：**物理时间**（连续轨迹 + 真实时间戳）。
- 若用 Δt，影响：**状态转移**（Q/K/V 轨迹用 ODE 积分），与记忆衰减无关。
- 支持不规则采样吗：是。
- 有无定理：无（只有实验与逼近论证）。
- 用于：不规则时间序列。
- 与候选结构异同：连续时间只进 attention 轨迹，不进记忆衰减；无矩阵记忆、无双状态。
- 裁决：**未覆盖**。

### 13. Time Interval Aware Self-Attention for Sequential Recommendation（TiSASRec）
- 作者 / 年份 / 会议或期刊 / 链接：Jiacheng Li, Yujie Wang, Julian McAuley / 2020 / WSDM 2020（ACM DOI 10.1145/3336191.3371786；dblp: conf/wsdm/LiWM20）/ https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
- 记忆状态：**无显式记忆状态**（自注意力层，无递归状态）。
- 读操作精确公式：`z_i=Σ_j α_ij(m_{s_j}W_V+r^v_{ij}+p^v_j)`（式 6）；`α_ij=softmax(e_ij)`（式 7）；`e_ij=m_{s_i}W_Q(m_{s_j}W_K+r^k_{ij}+p^k_j)^T/√d`（式 8）。
- 写操作精确公式：无。
- 是否存在 M(t+1)=F(...)：**否**。
- alpha_t 归类：**attention 权重**（时间间隔只进入 key/value 的相对时间嵌入与 attention logits）。
- 用物理时间 Δt 还是观测序号：**物理时间**（但归一化为整数比值：`r^u_{ij}=floor(|t_i−t_j|/r^u_min)`，并 clip 到 k）。
- 若用 Δt，影响：**只拼进输入**（相对时间间隔嵌入）。
- 支持不规则采样吗：是（不规则的相对间隔）。
- 有无定理：无。
- 用于：推荐系统。
- 与候选结构异同：真实时间只影响注意力打分，不影响任何记忆的衰减或写入；无矩阵、无双状态。
- 裁决：**未覆盖**。

## 2. 关键问题专答

1) **alpha 分类（严格区分）**
- `α=sigmoid(g(z))` 一类"学习门"：Mamba（Δ 广义门）、GRU-D（γ 门）、GLA（G_t）、Gated DeltaNet（α,β）、Titans（α_t）。
- `α=1−exp(−λΔ)` 一类"物理时间进更新律"：**没有一篇矩阵记忆工作**。最接近的是 GRU-D 的 `γ=exp(−max(0,Wδ+b))`（向量）与 TiM4Rec 的"时间差乘进标量衰减"（矩阵，但只衰减）。
- attention 权重：Schlag 2021、ContiFormer、TiSASRec。
- surprise/预测误差：Titans（写入强度=梯度）、CRU（Kalman 增益=协方差加权的新息）。
- 其他（固定 λ/γ）：Ba fast weights（λ）、RetNet（γ）。
- 外部真实 Δ：TiM4Rec（衰减）、CRU（转移+增益）、GRU-D（衰减）。

2) **有没有任何一篇让真实秒级 Δ 同时控制矩阵记忆的衰减与写入**：**没有**（检索范围内）。分项最接近：TiM4Rec（Δ→矩阵衰减，2024）；CRU（Δ→向量状态衰减 + 间接写入增益，2022）；Titans/Gated DeltaNet（衰减+写入都很接近，但 Δ 都是学习门/按步）。

3) **有没有证明采样细化一致/半群**：**没有**。可用的半群事实只有固定 Δ 的矩阵指数：`exp(A(δ1+δ2))=exp(Aδ1)exp(Aδ2)`（Mamba 式 4 的 `exp(ΔA)` 在 Δ 固定时的组合性）；但 Mamba 的 Δ 是输入依赖的 `Δ(x)`，因此 `Ā(x)=exp(Δ(x)A)` 不构成半群；TiM4Rec 的 `d_i·a_i` 依赖具体分段，也不满足；GLA/RetNet/GDN 的 per-step 衰减是马尔可夫转移、按步而非按时间；Titans 的 α_t、Gated DeltaNet 的 α_t 都是数据依赖门。**没有任何论文给"采样细化一致性"定理**。

4) **双状态问题**：候选是"矩阵记忆 M + 隐状态 h 两路"。文献里的双结构：
- Ba et al. 2016：h（活动）+ A（快权重）——快权重由 h 生成，读取又更新 h（结构性接近，但没有独立 z 驱动）。
- Titans：长期记忆 M + 主网络（h）——读出 y_t 进入主网络，但 h 的更新不经过"读出的 r"（部分覆盖）。
- 多数工作（Mamba/RWKV/RetNet/GLA/GDN）只有单状态。
- 没有找到"矩阵记忆读出的 r_i 再更新独立 h_i，同时 M 由 h_i 生成 q/k/v"的逐字结构。

5) **"不变内容不反复重写"**：已有现成机制——delta rule（Gated DeltaNet 的 `(I−β k k^T)` 擦除 + 只写误差；DeltaNet 原始形式）与 Titans 的 surprise（只写梯度大的内容）。候选的 `M_i=M_i^-+α_i v_i k_i^T` **没有擦除项**，因此在"重复帧/不变内容"上会反复叠加同一 key 的内容（除非 α_i 恰好为 0）。这是候选相对文献的**结构性缺口**。

## 3. 反对结论与反例

**R1（骨架已存在）**：矩阵快权重 + 乘性衰减 + 外积写 + 加性读出 = Ba et al. NIPS 2016 式 (1)(2)(3) 已完整给出；"attention=可寻址快权重"由 Schlag et al. ICML 2021 形式化。候选的"记忆骨架"不是新的，只有"衰减用物理 Δ、写入带门"两个替换点。

**R2（Δ 进矩阵衰减已被做）**：TiM4Rec（2024）把真实 unix 时间差 `d_i=t_i−t_{i−1}` 直接乘进 Mamba2/SSD 的标量衰减 `Â_i=d_i·a_i`，即矩阵记忆的衰减由物理时间控制。虽然它只控制衰减、不控制写入，但"物理 Δ → 矩阵记忆衰减"这一条新颖性已被占。

**R3（Δ 同时影响衰减+写入在向量状态已有）**：CRU（ICML 2022）用 `exp(A(t−τ))` 做状态转移、用 `Σ^-(Δ)` 推 Kalman 增益，于是 Δ 既控制衰减又控制写入强度；只是记忆是向量、写入是 Kalman 新息而非 `v k^T`。

**R4（"学习门"不是"物理时间"）**：Mamba 的 Δ 被原文明确称为 RNN 门控的推广（§3.5.1 + Theorem 1），且由 `softplus(Parameter+Linear(x))` 产生——与真实采样时刻无关。把 `exp(-ΛΔt)` 写成"矩阵记忆的标准衰减"时，必须区分：Δ 是物理秒（TiM4Rec/CRU/GRU-D）还是学习门（Mamba/GLA/GDN/Titans）。候选要求前者，而绝大多数"衰减记忆"文献做的是后者。

**R5（无半群/采样细化定理）**：没有任何论文证明 `P_{δ1+δ2}=P_{δ2}∘P_{δ1}` 对"衰减+写入"复合成立；Mamba 的 `exp(ΔA)` 只在 Δ 固定且与状态无关时是半群，学习门 Δ(x) 破坏它。候选若声称采样细化一致，需要自己给证明，且必须处理"写入注入"破坏复合的问题（与上一轮 ODE-RNN 结论同构）。

**R6（失败模式）**：
- 重复帧：候选 `exp(-Λ·0)=I`（不衰减）但 `α_i>0`（仍写），同一内容被反复叠加 → 需要 delta rule 的擦除项；文献有现成解（Gated DeltaNet 式 10）。
- 丢帧/大 Δ：`exp(-ΛΔt)` 数值下溢为 0，矩阵记忆整体清零；TiM4Rec 用 clip/归一化绕开，候选若无约束会退化。
- 门的饱和：GLA/GDN 的 sigmoid 门在大间隔时饱和，Titans 的 surprise 在梯度消失时失效（原文自述"gradient can become extremely small after several surprising steps"）。
- 物理 Δ 与序列位置不一致：TiM4Rec 把 Δ 乘进衰减后仍需保留位置信息，否则同一时刻的多条交互无法排序（论文用 mask 结构处理）。
- 向量 vs 矩阵：Mamba/RWKV/GRU-D/CRU 全是向量状态；候选的"矩阵 M"必须自己承担容量/秩的代价，而文献里矩阵记忆的容量/稳定性定理（Ba 2016 无；RetNet/GLA/GDN 只给等价性）并不存在可供引用的"容量保证"。

**R7（评测缺口）**：上述所有工作都没有"物理时间戳 + 视频/感知流 + 双状态"的评测设置；TiM4Rec 是推荐系统，CRU 是医疗/气候，Mamba 系列是语言。因此"没人做过"可能只是因为任务域不同，而不是机制新颖——这既是候选的机会，也意味着没有现成的基线/定理可用。

## 4. 汇总裁决表

| # | 论文 | 与候选的衰减项 | 写入项 | 读出 | 双状态 | Δt 落点 | 裁决 |
|---|---|---|---|---|---|---|---|
| 1 | Mamba | `exp(ΔA)` 同形 | 加性 B̄x，无 key 寻址 | C h | 无 | 学习门，非物理 | 部分 |
| 2 | TiM4Rec | `a^{d_i}` 同形、真实 Δ | 标准 SSD 写，无 α | Q/K/V attn | 无 | 物理 Δ→衰减 | 部分 |
| 3 | CRU | `exp(A(t−τ))` 同形 | Kalman 新息，无 v k^T | decoder | 无 | 物理 Δ→转移+增益 | 部分 |
| 4 | GRU-D | `exp(−Wδ)` 同形 | GRU 门，无 key | 头 | 无 | 物理 δ→衰减 | 未覆盖 |
| 5 | Ba 2016 | λ 固定，无 Δ | `η h h^T`，无 α 门 | A h | h+A | 无 | 部分（骨架） |
| 6 | Schlag 2021 | 无衰减 | `v k^T` 无门 | 线性 attn | 无 | 无 | 部分（等价性） |
| 7 | RetNet | γ 固定按步 | `K^T V` 无门 | Q S | 无 | 无 | 部分 |
| 8 | RWKV | `e^{−w(t−i)}` 按步 | 累加 | WKV | 无 | 无 | 未覆盖 |
| 9 | GLA | 2D 学习门 | `k^T v` 无门 | S q | 无 | 无 | 部分 |
| 10 | Gated DeltaNet | 学习门 α | delta rule 含擦除 | S q | 无 | 无 | 部分（写入最接近） |
| 11 | Titans | 学习门 α | 梯度/surprise 写 | M q | 部分 | 无 | 部分（写入动机） |
| 12 | ContiFormer | 无记忆 | 无 | 连续 attn | 无 | 连续轨迹 | 未覆盖 |
| 13 | TiSASRec | 无记忆 | 无 | attn 权重 | 无 | 只拼输入 | 未覆盖 |

## 5. 结论

- **最接近的 3 个先例**：
  1. **Ba et al. 2016（快权重）**：矩阵记忆 + 乘性衰减 + 外积写 + 加性读出的骨架，与候选逐项对应；缺物理 Δ 与门控 α。
  2. **TiM4Rec 2024**："真实时间差 → 矩阵记忆衰减"已实现（`Â_i=d_i·a_i`）；缺 Δ 控制写入、缺 key 寻址写入。
  3. **Gated DeltaNet 2025 / Titans 2025**："衰减门 + 内容写入 + 不重复重写（delta rule / surprise）"已被做到；缺物理 Δ。
- **最致命的 3 条反对证据**：
  1. 候选的记忆骨架（矩阵+衰减+外积写+读出）是 Ba 2016 / RetNet / GLA 的既有结构，2016–2025 反复出现。
  2. "物理 Δ 控制矩阵记忆衰减"已被 TiM4Rec 2024 做掉（`d_i·a_i` 乘进 1-SS 掩码）；"Δ 同时影响衰减与写入增益"在 CRU 2022 的向量状态上已有。
  3. 没有任何工作证明"衰减+写入"复合映射的采样细化一致性/半群；Mamba 的 `exp(ΔA)` 只在 Δ 固定且与输入无关时才是半群，学习门 Δ(x) 破坏之。候选若主张一致性，必须自己给证明并处理写入破坏复合的问题。
- **有没有与候选逐字对应的已有工作**：**没有**。最接近的组合是"矩阵记忆（Ba/TiM4Rec/GLA/GDN/Titans）+ 物理 Δ（TiM4Rec/CRU/GRU-D）"，但没有一篇同时具备：真实 Δ 控制矩阵衰减、α 门控内容寻址写入、独立双状态 h、以及采样细化性质。可辩护的新颖性应定位为**具体组合 + 可证明性质（Δ→衰减/写入的极限一致性、容量/稳定性界）**，并显式补上擦除机制（delta rule）以应对重复帧。
- **必须明说的边界**：Mamba/RWKV/GRU-D/CRU 是**向量状态**（CRU 的协方差是矩阵但不是记忆内容），只有 Ba/Schlag/RetNet/GLA/GDN/Titans/TiM4Rec 有**矩阵记忆**；把"记忆衰减+写入"当新机制之前，先确认自己引用的是哪一类。


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


# 07 · (ρ, α) = (exp(−ΛΔ), 1−exp(−ΛΔ)) 的数学身份核查

> 专题：候选结构里那对系数是不是已有的「指数积分器 / 零阶保持（ZOH）」离散化。
> 候选（被检验）：
>
>     M_i- = exp(-Lambda * Delta_i) ⊙ M_{i-1}     # 矩阵快权重记忆，按真实时间衰减
>     M_i  = M_i- + alpha_i * v_i k_i^T            # 内容寻址写入
>     r_i  = R(M_i, q_i)                           # 读
>     h_i  = H(h_{i-1}, r_i, z_i)                  # 另一路隐状态
>
> 工具：tavily_search (advanced) + tavily_extract 抓正文；每条标核实级别（[全文] / [仅摘要级] / [未核实]）。
> 本文件属 litreview2 轮次，与 litreview/ 下的旧轮无关。

---

## 0. 判定（先说结论）

**这对系数不是新东西，它是 1960 年代就写进控制教科书的一行公式。**

线性 ODE  dM/dt = −ΛM + U(t)  在零阶保持（ZOH，U 在区间内取常值）下的**精确**解是

    M(t+Δ) = e^{−ΛΔ} M(t) + Λ^{−1}( I − e^{−ΛΔ} ) U

即 **ρ = e^{−ΛΔ}、α = I − e^{−ΛΔ} 是同一个 Δ 的两个函数，不是一个半衰期加一个写入门**。它们满足恒等式

    ρ + α = 1        （逐元素，标量情形）

这条恒等式是 ZOH 的指纹：**α 不是自由参数**。候选结构若把 alpha 当成独立可学门（α_i = g(z_i)），它就不再是任何连续时间系统的离散化——半群性、采样细化极限、|ρ|≤1 的稳定性全部丢失（§4 R2 给出推导）。

**谁已经写出来了：**
- **教科书层**：ZOH 精确离散化（[ETH 讲义](https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Signals-and-Systems/Lectures/Fall2018/Lecture1_SigSys18.pdf) 逐字：A_d = e^{A_c T_s}、B_d = (∫_0^{T_s} e^{A_c τ}dτ)B_c）；指数积分器的 φ 函数体系（Hochbruck & Ostermann, Acta Numerica 2010）。
- **深度序列模型层**：Mamba 式 (4) **逐字**给出 Ā = exp(ΔA)、B̄ = (ΔA)^{−1}(exp(ΔA)−I)·ΔB（ZOH）；Mamba-2/SSD 把 A 退化为标量×单位阵、状态升为矩阵，仍用同一离散化。
- **矩阵记忆层**：**Exact Flow Linear Attention (arXiv:2512.12602)** 已经把「ZOH + 矩阵值联想记忆 + 精确流」做完了，并明确写出 dS/dt = −A_t S + b_t 与闭式解 S_t = e^{−β_t A_t}S_{t−1} + ∫_0^{β_t} e^{−(β_t−τ)A_t} b_t dτ。**这是对候选结构最直接的覆盖。**

**结论：**
1. (ρ, α) 这一对 = **教科书内容**，不可作为新颖性主张，只能在方法节里引用。
2. 把它套到**矩阵记忆**上 = **已有论文（EFLA 2025）做过**，不是空白。
3. 候选结构真正还站得住的差异只有三处，见 §5：「对角 Λ + 真实 Δ」「ρ+α=1 的可行性」「脉冲式写入的连续极限」——**要主张贡献，必须在这三处给新证明，而不是给这对系数。**

---

## 1. 教科书与离散化来源（4 篇）

### 1. Hochbruck & Ostermann, Exponential integrators
- 作者 / 年份 / 期刊 / 链接：Marlis Hochbruck, Alexander Ostermann / 2010 / *Acta Numerica* 19:209–286 / [PDF](https://na.math.kit.edu/download/papers/acta-final.pdf) **[全文]**
- 记忆状态：**算子/矩阵值状态**（抽象 Banach/Hilbert 空间上的 u(t)，可为矩阵）
- 读操作精确公式：不适用（数值分析综述，无「读」）
- 写操作精确公式：指数积分器的一般形式（对 u' = Au + g(t,u) 用 φ 函数展开）；本文给出的关键结构是 e^{−tA} 的算子定义
- 是否存在 M(t+1)=F(M_t,k_t,v_t,alpha_t)：**部分**——有 M(t+Δ)=F(M(t)) 的精确流，但没有内容寻址的 k/v
- alpha_t 归类：**其他**（φ 函数系数，由 Δ 与 A 唯一决定，非学习、非 attention）
- 用物理时间 Δt 还是观测序号：**物理时间**（t, s 是实数时间）
- 若用 Δt，影响：**衰减**（e^{−tA}）+ **状态转移**（φ_k(tA)），两者由同一 t 生成
- 支持不规则采样/丢帧/重复帧吗：支持——半群允许任意 t,s
- 有无定理（有则抄）：
```
e^{−0A} = I ,   e^{−tA} e^{−sA} = e^{−(t+s)A},   t, s ≥ 0        (2.4)
‖e^{−tA}‖_{L2←L2} ≤ e^{−t} ≤ 1
```
  → **半群性 + 收缩性**，这是候选结构里 ρ 的全部数学身份。
- 用于：数值分析（ODE/PDE 时间积分），不是视频/语言
- 与候选结构异同：**衰减项**：完全同一个对象（e^{−tA}）；**写入项**：候选的 α 在指数积分器里就是 ∫_0^Δ e^{−τA}dτ = Δ·φ_1(−ΛΔ)，**与 ρ 由同一 Δ 生成**；**双状态**：无；**Δ t 落点**：进衰减与积分系数，不进 k/v。
- 裁决：**完整覆盖（系数层面）**。证据：(2.4) 的半群性与 ‖e^{−tA}‖≤1 正是 ρ=exp(−ΛΔ) 必须且自动满足的两条性质。

### 2. Ducard, Signals and Systems, Lecture 1: From Continuous Time to Discrete Time（ETH）
- 作者 / 年份 / 机构 / 链接：Guillaume Ducard（基于 R. D'Andrea 材料）/ 2018 / ETH Zurich IDSC 课程讲义 / [PDF](https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Signals-and-Systems/Lectures/Fall2018/Lecture1_SigSys18.pdf) **[全文]**
- 记忆状态：向量状态 q ∈ R^n（连续 LTI 系统）
- 读操作精确公式：y(t) = C_c q(t) + D_c u(t)
- 写操作精确公式（**原式照抄**）：
```
q(t) = e^{A_c t} q(0) + ∫_0^t e^{A_c (t−τ)} B_c u(τ) dτ
A_d = e^{A_c T_s} ,    B_d = ( ∫_0^{T_s} e^{A_c τ} dτ ) B_c
```
  原话："This is the exact solution to the differential equation, there are no discretization errors."
- 是否存在 M(t+1)=F(...)：**是**——ZOH 的 q[n+1] = A_d q[n] + B_d u[n]
- alpha_t 归类：**其他**（B_d，由 A_c 与 T_s 解析决定，非学习）
- 用物理时间 Δt 还是观测序号：**物理时间**（T_s 是采样周期，单位为秒）
- 若用 Δt，影响：**衰减**（A_d = e^{A_c T_s}）+ **写入强度**（B_d 里含 ∫_0^{T_s}）——两者**共享同一个 T_s**
- 支持不规则采样：**是**（把 T_s 换成 T_k 即可逐区间积分）
- 有无定理：**有**——"exact, no discretization errors"；代价："information is still lost by the discretization: the inter-sample behavior."
- 用于：控制（采样数据系统）
- 与候选结构异同：把 A_c = −Λ、B_c = I、u ≡ U 代入，逐字得到
```
M_i = e^{−ΛΔ_i} M_{i−1} + ( ∫_0^{Δ_i} e^{−Λτ} dτ ) U
    = e^{−ΛΔ_i} M_{i−1} + Λ^{-1}( I − e^{−ΛΔ_i} ) U
⇒  ρ = e^{−ΛΔ} ,   α = I − e^{−ΛΔ} ,   ρ + α = 1
```
  **这就是候选那对系数的来源，一字不差。** 唯一差别：候选的写入内容是 v_i k_i^T（随 i 变），不是常值 U。
- 裁决：**完整覆盖**。候选的 (ρ, α) 就是这篇讲义的 A_d、B_d 在 A_c=−Λ、B_c=I 下的特例。

### 3. Gu, Goel, Ré, Efficiently Modeling Long Sequences with Structured State Spaces（S4）
- 作者 / 年份 / 会议 / 链接：Albert Gu, Karan Goel, Christopher Ré / 2021（ICLR 2022）/ [PDF](https://arxiv.org/pdf/2111.00396) **[全文]**
- 记忆状态：向量状态 x_k ∈ R^N（结构化矩阵 A）
- 读操作精确公式：y_k = C x_k
- 写操作精确公式（**原式照抄，注意：S4 用的是双线性变换，不是 ZOH**）：
```
x_k = Ā x_{k−1} + B̄ u_k
Ā = (I − Δ/2 · A)^{-1} (I + Δ/2 · A)
B̄ = (I − Δ/2 · A)^{-1} Δ B ,     C̄ = C
```
  原文："we follow prior work in using the bilinear method [43], which converts the state matrix A into an approximation Ā"；"the step size ∆ that represents the resolution of the input"
- 是否存在 M(t+1)=F(...)：**是**（Ā x + B̄ u）
- alpha_t 归类：**其他**（B̄，由 Δ 与 A 解析决定）
- 用物理时间 Δt 还是观测序号：Δ 名义上是"输入分辨率"（物理时间步），但训练时是**固定常数**
- 若用 Δt，影响：**衰减 + 状态转移 + 写入强度**都由 Δ 控制，但形式是双线性
- 支持不规则采样：**否**（固定 Δ）
- 有无定理：**无**（不证离散化误差；只给 HiPPO 初始化与卷积核等价性）
- 用于：长序列建模（语言/音频/图像）
- 与候选结构异同：**衰减项**：候选 e^{−ΛΔ}（ZOH）vs S4 (I−ΔA/2)^{−1}(I+ΔA/2)（Tustin）——**不同**；**写入项**：S4 的 B̄=(I−ΔA/2)^{−1}ΔB，**与 α 不同**；**双状态**：无；**Δ t 落点**：衰减与写入都吃 Δ。
- 裁决：**未覆盖（就这对系数而言）**，但**它纠正了 lead 任务书的一个前提**：任务书说「S4 的离散化是 Ā=exp(ΔA)」，**这是错的**——S4 正文 §2.3 用的是双线性法；ZOH 版是 Mamba 才明确采用的（见下条）。

### 4. Gu & Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- 作者 / 年份 / 会议 / 链接：Albert Gu, Tri Dao / 2023 / arXiv:2312.00752（COLM 2024）/ [PDF](https://arxiv.org/pdf/2312.00752) **[全文]**
- 记忆状态：向量状态 h(t) ∈ R^N（每通道独立）
- 读操作精确公式：y_t = C h_t（式 2b）
- 写操作精确公式（**逐字**）：
```
h'(t) = A h(t) + B x(t)                                  (1a)
h_t   = Ā h_{t−1} + B̄ x_t                                (2a)
Ā = exp(Δ A)
B̄ = (Δ A)^{-1}(exp(Δ A) − I) · Δ B                       (4)
```
  原文："Various rules can be used such as the zero-order hold (ZOH) defined in equation (4)."
- 是否存在 M(t+1)=F(...)：**是**
- alpha_t 归类：**学出的门（伪时间步）**。Δ 由输入产生（选择性机制）；论文自述 "Δ controls the balance between how much to focus or ignore the current input x_t ... a large Δ resets the state h and focuses on the current input x, while a small Δ persists the state"
- 用物理时间 Δt 还是观测序号：**观测序号 + 学习出来的标量**；没有秒、没有真实间隔
- 若用 Δt，影响：**衰减（Ā）+ 写入强度（B̄）+ 状态转移**——全部由同一个 Δ 生成
- 支持不规则采样：**否**（除非把真实 Δ 喂进去，论文没有）
- 有无定理：**无证明，但有断言**——原文："Discretization has deep connections to continuous-time systems which can endow them with additional properties such as resolution invariance (Nguyen, Goel, et al. 2022) and automatically ensuring that the model is properly normalized"
- 用于：语言建模
- 与候选结构异同：**衰减项**：候选 ρ = e^{−ΛΔ} 就是 Mamba 的 Ā（取 A = −Λ 对角）；**写入项**：Mamba 的 B̄ = (ΔA)^{−1}(e^{ΔA}−I)ΔB，在 A = −Λ、B = I 时正是 α = Λ^{−1}(I−e^{−ΛΔ})，**同一个式子**；**读**：Mamba 是 C h（线性读出），候选是 R(M,q)（内容寻址读出）；**双状态**：无；**Δt 落点**：衰减 + 写入，**完全一致**。
- 裁决：**部分覆盖（系数层面 100%，状态层面 0%）**。证据：式 (4) 就是候选那对系数的算子形式，只差把矩阵状态换成向量、把真实 Δ 换成学习 Δ。

---

## 2. 矩阵记忆 / 线性注意力层（5 篇）

### 5. Katharopoulos et al., Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- 作者 / 年份 / 会议 / 链接：Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret / 2020 / ICML 2020, PMLR 119 / [PDF](https://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf) **[全文]**
- 记忆状态：**矩阵**（KV 累加矩阵）S = Σ_j φ(K_j) V_j^T ∈ R^{C×M}，以及归一化累加向量 z = Σ_j φ(K_j)
- 读操作精确公式：o_t = φ(q_t)^T S / (φ(q_t)^T z)
- 写操作精确公式：S_t = S_{t−1} + φ(k_t) v_t^T（**纯加性，无衰减、无门**）
  原文佐证："we can compute Σ_{j=1}^N φ(K_j) V_j^T and Σ_{j=1}^N φ(K_j) once and reuse them for every query"
- 是否存在 M(t+1)=F(...)：**是**（additive outer product，无 alpha）
- alpha_t 归类：**其他**（恒为 1；无门）
- 用物理时间 Δt 还是观测序号：**观测序号**；若用 Δt：不适用
- 支持不规则采样：**否**
- 有无定理：**无**
- 用于：语言建模
- 与候选结构异同：**衰减项**：无（候选有，这是差异）；**写入项**：候选的 α v k^T 是它的加性写；**读**：同一个内积读出；**双状态**：无；**Δt 落点**：无。
- 裁决：**部分覆盖（读/写的形状）**。给出了候选 M 的维度与读写代数，但**没有衰减、没有 Δ**。

### 6. Schlag, Irie, Schmidhuber, Linear Transformers Are Secretly Fast Weight Programmers
- 作者 / 年份 / 会议 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139 / [PDF](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf) **[全文]**
- 记忆状态：**矩阵** W（fast weights，"slow net programs fast weights"）
- 读操作精确公式：
```
v̄^(i) = W^(i−1) φ(k^(i))                       (20)   ← 内容寻址读
y^(i)  = W^(i) φ(q^(i))                          (25)
```
- 写操作精确公式（**逐字**）：
```
W^(i) = W^(i−1) + v_new^(i) ⊗ φ(k^(i)) − v̄^(i) ⊗ φ(k^(i))      (23)
      = W^(i−1) + β^(i) ( v^(i) − v̄^(i) ) ⊗ φ(k^(i))            (24)
β^(i) = σ(W_β x^(i))                                             (21)
```
- 是否存在 M(t+1)=F(...)：**是**
- alpha_t 归类：**学出的门**（β^(i) 是 sigmoid 输出，原文称 "write-strength"）
- 用物理时间 Δt 还是观测序号：**观测序号 i**；若用 Δt：不适用（论文完全没有时间间隔概念）
- 支持不规则采样：**否**
- 有无定理：**有（容量结论）**——"We infer a memory capacity limitation of recent linearised softmax attention variants"；"the linear Transformers can end up in an overcapacity regime, if the sequence length L exceeds the dimension d_dot of the keys"。**这是对纯加性写入（候选的写形式）的负面结论**：序列长度超过 key 维度后，加性写会饱和。
- 用于：语言建模 / 合成检索
- 与候选结构异同：**衰减项**：**无**（候选有 ρ，这是唯一结构增量）；**写入项**：候选 α v k^T 是式 (23) 去掉 remove 项的版本；**读**：W φ(k) 与候选 R(M,q) 同构；**双状态**：无；**Δt 落点**：无。
- 裁决：**部分覆盖（写规则）**。它把「写强度」做成**独立学习的门 β**——正是 lead 问的第 5 点：**已有先例就是 alpha 独立于任何衰减**。所以「alpha 必须是 1−e^{−λΔ}」不是这个圈子的共识，而是候选可以主张的**约束**（见 §5）。

### 7. Graves, Wayne, Danihelka, Neural Turing Machines
- 作者 / 年份 / 来源 / 链接：Alex Graves, Greg Wayne, Ivo Danihelka / 2014 / arXiv:1410.5401 / [PDF](https://arxiv.org/pdf/1410.5401) **[全文]**
- 记忆状态：**矩阵** M_t ∈ R^{N×M}（N 个 slot × M 维），另有控制器 RNN 隐状态
- 读操作精确公式（**逐字**）：
```
r_t ← Σ_i w_t(i) M_t(i)                                          (2)
```
- 写操作精确公式（**逐字**）：
```
M̃_t(i) ← M_{t−1}(i) [ 1 − w_t(i) e_t ]                           (3)   擦除
M_t(i)  ← M̃_t(i) + w_t(i) a_t                                    (4)   写入
```
- 是否存在 M(t+1)=F(...)：**是**
- alpha_t 归类：**attention 权重 + 学出的门**（w_t 是内容寻址的 softmax 权重，e_t、a_t 是 erase/add 向量）
- 用物理时间 Δt 还是观测序号：**观测序号**；若用 Δt：不适用
- 支持不规则采样：**否**
- 有无定理：**无**
- 用于：算法学习 / 记忆任务
- 与候选结构异同：**衰减项**：式 (3) 的 [1 − w e] 是**按地址的乘法衰减**（erase），候选 exp(−ΛΔ) 是**按通道的乘法衰减**——形式同族但**语义不同**；**写入项**：式 (4) 的加性写 = 候选的 α v k^T；**读**：式 (2) 的加权和 = 内容寻址读；**双状态**：**有**（矩阵记忆 + 控制器隐状态）；**Δt 落点**：无。
- 裁决：**部分覆盖（矩阵记忆 + 双状态 + 内容寻址擦写）**。**重点：双状态（矩阵 M + 隐状态 h）在 2014 年就存在**，不是候选的新点。

### 8. Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics（EFLA）
- 作者 / 年份 / 来源 / 链接：EFLA 作者组 / 2025–2026 / arXiv:2512.12602（v5；代码 github.com/declare-lab/EFLA）/ [HTML](https://arxiv.org/html/2512.12602v5) **[全文]**
- 记忆状态：**矩阵** S_t ∈ R^{d×d}（delta-rule 线性注意力的联想记忆）
- 读操作精确公式：o_t = S_t^T q_t（delta-rule 线性注意力的标准内容寻址读，本文沿用）
- 写操作精确公式（**逐字，编号为原论文编号**）：
```
S_t = (I − β_t k_t k_t^T) S_{t−1} + β_t k_t v_t^T                     (31)
    = S_{t−1} + β_t ( −k_t k_t^T S_{t−1} + k_t v_t^T )                (9)
令 A_t = k_t k_t^T ,  b_t = k_t v_t^T
dS(t)/dt = −A_t S(t) + b_t                                            (10)
S_t = e^{−β_t A_t} S_{t−1} + ∫_0^{β_t} e^{−(β_t−τ)A_t} b_t dτ         (11)
```
  附录 D 的秩 1 性质：A_t² = k_t k_t^T k_t k_t^T = λ_t A_t，λ_t = k_t^T k_t（"A_t is a scaled projection matrix"）。
- 是否存在 M(t+1)=F(...)：**是**，且**给出闭式**
- alpha_t 归类：**学出的门/步长**。β_t 是每 token 的更新尺度，论文没有把 β 与物理秒绑定
- 用物理时间 Δt 还是观测序号：**观测序号**（β_t 每 token 一个）
- 若用 Δt，影响：**衰减（e^{−βA}）+ 写入（积分的 β）**——两者由同一个 β 生成
- 支持不规则采样：**结构上支持**（把 β_t 换成 Δ_t 即可逐区间积分，这正是 ZOH 的用法），但论文未做
- 有无定理：**有（误差结论）**——"the standard Delta rule update is only a first-order numerical approximation to the underlying continuous dynamics"；"the Euler update is only first-order accurate ... may become inaccurate when the effective dynamics are stiff, for example under large key norms or large update scales"；"removes the Euler discretization error ... without introducing additional parameters"
- 用于：语言建模 / 长序列
- 与候选结构异同：
  - **衰减项**：候选 ρ = e^{−ΛΔ}（对角 Λ）；EFLA 是 e^{−β_t A_t}（**秩 1、数据依赖**）。**同族，不同 A**。
  - **写入项**：候选 α v k^T；EFLA 是输入积分 ∫_0^β e^{−(β−τ)A} b dτ，b = k v^T。
  - **读**：相同（q^T S）。**双状态**：无。**Δt 落点**：都在衰减 + 写入，都只用一个标量（Δ 或 β）。
  - **关键差别（对候选有利的一面）**：EFLA 的 A_t 是**秩 1**，所以闭式解里会出现一个 **β 的线性项**（b_t 中落在 range(A_t)⊥ 的分量在 ODE 里线性增长）；候选的 Λ 是**满秩对角**，α = Λ^{-1}(I−e^{−ΛΔ}) 是干净的指数积分，**没有久期项**。这是候选相对 EFLA 唯一真正的数学差异。
- 裁决：**完整覆盖（矩阵记忆 + ZOH + 精确流）——最致命的一篇。** 证据：式 (10)(11) 就是「遗忘 = 时间衰减」「写入 = 外部驱动」的连续时间表述，且明确写 ZOH、明确说 Euler 版只是一阶近似。**候选把 A 换成对角 Λ、把 β 换成真实 Δ，是参数化简化，不是新结构。**

### 9. Dao & Gu, Transformers are SSMs: Structured State Space Duality（Mamba-2 / SSD）
- 作者 / 年份 / 会议 / 链接：Tri Dao, Albert Gu / 2024 / ICML 2024, arXiv:2405.21060 / [PDF](https://arxiv.org/pdf/2405.21060) **[全文]**
- 记忆状态：**矩阵** h_t ∈ R^{N×P}（head dimension P）
- 读操作精确公式：y_t = C_t^T h_t（式 2b）
- 写操作精确公式：
```
h_t = A_t h_{t−1} + B_t x_t                                   (2a)
```
  A 的结构："further simplified from diagonal to scalar times identity structure. Each A_t can also be identified with just a scalar in this case."
  离散化：沿用 Mamba 同一规则——"our main models adopt the same parameterization and discretization step as prior work (see Gu & Dao (2023) for details)"，即 Ā = exp(ΔA)。合起来 A_t = exp(Δ_t a)·I（a 为标量）。
- 是否存在 M(t+1)=F(...)：**是**
- alpha_t 归类：**学出的门**（a_t 输入依赖标量；"a_i are input-dependent scalars bounded in [0, 1]"）
- 用物理时间 Δt 还是观测序号：**观测序号**（Δ_t 由输入产生，无秒）
- 若用 Δt，影响：**衰减**（唯一影响）
- 支持不规则采样：**否**
- 有无定理：**有（对偶性）**——SSD 与掩码注意力等价：L_ij = a_i × … × a_{j+1}（i ≥ j），**衰减的连乘就是注意力掩码**；以及 "any kernel attention method possessing a fast recurrent form must be an SSM"
- 用于：语言建模
- 与候选结构异同：**衰减项**：候选 e^{−ΛΔ} 逐通道对角；SSD 是**标量×单位阵**（表达力更弱）；**写入项**：候选 α v k^T 与 SSD 的 B_t x_t 同形；**读**：C^T h；**双状态**：无；**Δt 落点**：衰减。
- 裁决：**部分覆盖（矩阵状态 + 时间衰减 + 秩 1 写入的代数），但衰减是标量、Δ 是伪时间。**

---

## 3. 真实 Δt 的向量状态先例（1 篇）

### 10. Che et al., Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- 作者 / 年份 / 会议 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / *Scientific Reports* 8:6085（arXiv:1606.01865）/ [PDF](https://arxiv.org/pdf/1606.01865) **[全文]**
- 记忆状态：**向量**（GRU 隐状态 + 输入变量），不是矩阵
- 读操作精确公式：GRU 的门控读（h_{t−1} 与 x_t 拼合）
- 写操作精确公式（**逐字**）：
```
γ_t = exp{ −max(0, W_γ δ_t + b_γ) }                             (4)
```
  δ_t 是**真实时间间隔**；原文："applying masking and time interval (using a decay term) to the inputs and network states of GRU"
- 是否存在 M(t+1)=F(...)：**部分**（h_t 由衰减后的 h_{t−1} 与衰减后的输入共同产生）
- alpha_t 归类：**外部真实 Delta t**（δ_t 直接进指数；衰减率 W_γ、b_γ 学习）
- 用物理时间 Δt 还是观测序号：**物理时间**
- 若用 Δt，影响：**衰减**（唯一影响）
- 支持不规则采样/丢帧：**是——这就是它的全部卖点**（missing pattern + time interval）
- 有无定理：**无**（纯经验；max(0,·) 的裁剪保证 γ ≤ 1 的稳定性）
- 用于：临床时序 / 传感器
- 与候选结构异同：**衰减项**：γ = exp(−W_γ δ) 就是候选 ρ = exp(−ΛΔ)，**连稳定性裁剪都是现成的**；**写入项**：GRU 门控写，**没有 1−e^{−λΔ} 的配套**；**读**：无矩阵；**双状态**：无；**Δt 落点**：衰减。
- 裁决：**部分覆盖（真实 Δt 进指数衰减）**。证据：式 (4)。**含义：exp(-ΛΔ) 这个写法在 2018 年就是处理不规则采样的标准手段。**

---

## 4. 反对结论与反例

### R1（最致命）这对系数是**同一个 Δ 的两个函数**，不是两个可独立设计的量

**证据链：**
- ETH 讲义：A_d = e^{A_c T_s}，B_d = (∫_0^{T_s} e^{A_c τ}dτ)B_c，并明确 "there are no discretization errors"。
- 代入 A_c = −Λ、B_c = I：α = Λ^{-1}(I − e^{−ΛΔ})，**α 完全由 ρ 与 Λ 决定，没有自由度**。
- 标量情形直接得到恒等式 **ρ + α = 1**。
- Mamba 式 (4) 是同一式子的算子版：B̄ = (ΔA)^{−1}(exp(ΔA)−I)·ΔB。
- 指数积分器记号：∫_0^Δ e^{−Λτ}dτ = Δ·φ_1(−ΛΔ)，即 **α = ΛΔ + O(Δ²)**。

**后果：** 候选把 (ρ, α) 写成两个可训练/可设计的对象，是**把一条恒等式拆成了两条独立假设**。只要训练让 alpha 偏离 1−e^{−λΔ}，模型就不再对应任何连续时间系统，"按真实时间衰减"的主张随之失效——因为它衰减的是记忆，但写入尺度不再与时间同量纲。

### R2 alpha 独立于 rho 会破坏什么（三条，前两条可证）

**(a) 半群/复合性被破坏（可证）。** 设输入 U 在合成区间内为常值，ZOH 的两次复合为
```
M₂ = ρ₂[ρ₁M + α₁Λ^{-1}U] + α₂Λ^{-1}U = ρ₁ρ₂M + (ρ₂α₁ + α₂)Λ^{-1}U
而 ρ₂α₁ + α₂ = e^{−ΛΔ₂}(1−e^{−ΛΔ₁}) + (1−e^{−ΛΔ₂}) = 1 − e^{−Λ(Δ₁+Δ₂)} = α₁₂
```
**即对同一输入的两次更新严格等于一次合并更新。** 若 α 是独立学出的门，上式一般不成立——"采样频率改变会改变模型语义"，**采样细化一致性（sampling refinement consistency）丢失**。Mamba 是把这条性质点名引用的（"resolution invariance (Nguyen, Goel, et al. 2022)"）。

**(b) Δ→0 的连续极限丢失（可证）。** ZOH 下 α = ΛΔ + O(Δ²)，故 (M_i − M_{i−1})/Δ → −ΛM + U，正确回到 ODE。若 α 是独立门（α ≈ const > 0），则 (M_i − M_{i−1})/Δ 发散为 αΛ^{-1}U/Δ → ∞：**Δ 越小写得越多**，与"物理时间"语义直接矛盾。**这是"α 必须与 ρ 配套"最直白的一条论证。**

**(c) 稳定性要额外兜底（工程事实）。** e^{−ΛΔ} 在 Λ≻0、Δ>0 时自动给出 |ρ|<1；独立门没有此保证。已有工作都在打补丁：SSD 明说 a_i 是 "input-dependent scalars bounded in [0,1]"；GRU-D 用 max(0, W_γδ+b_γ) 把指数限在 ≤0。**这些 clamp 正是 exp 免费提供的东西。**

### R3 已经有人把它做到矩阵记忆上了（EFLA），而且边界条件比候选更清楚

- EFLA 式 (9)(10)(11)：矩阵状态、ZOH、精确流、闭式解，一条不缺。
- 它给候选留了一个**真实差异**：EFLA 的 A_t = k_t k_t^T 是**秩 1**，输入积分必然含 β 的线性项；候选 Λ 满秩对角，α = Λ^{-1}(I−e^{−ΛΔ}) 无久期项。
- **但这是"更简单"的特例，不是更强的结论**：满秩对角 ≈ 逐通道独立的泄漏积分器，任何人可推；秩 1 才是需要技巧的情形。

### R4 内容寻址写入让 ZOH 假设变得可疑（候选最大的技术缺口）

ZOH 要求 **输入在区间内取常值**。候选的写入项是 v_i k_i^T——**每个观测一次、且随观测变**，更像**脉冲驱动**：

    dM/dt = −ΛM + Σ_i δ(t − t_i) v_i k_i^T

按这个连续模型做精确离散化，得到的不是「先衰减再按 α 写」，而是「衰减 + 瞬时跳变」：

    M(t_i^+) = e^{−ΛΔ_i} M(t_i^−) + v_i k_i^T

**注意：这里没有 α。** 也就是说，候选那个 α 到底属于哪种模型，取决于把写入解释成**区间常值驱动**（则 α=1−e^{−ΛΔ}，EFLA/Mamba 语义）还是**脉冲**（则 α≡1）。**两个语义不能混用，而候选的公式把 α 写成自由系数，恰好掩盖了这个选择。**

### R5 "真实 Δ" 与"学习 Λ" 的乘积不可辨识

若 Λ 也是学出来的，模型只依赖乘积 ΛΔ：把 Λ→Λ/c、Δ→cΔ 得到同一个 ρ。**这与上一轮（litreview/03 §4 N3）里 Kalman 增益对 (Q,R) 整体缩放的不可辨识同构**：只有比值有信息。声称"真实时间衰减"需要额外尺度锚定（固定 Λ 的物理单位、或用绝对时间戳归一化），否则 Δ 的单位是任意的。

### R6 谁能提供定理（清单）

| 来源 | 能提供什么 | 不能提供什么 |
|---|---|---|
| Hochbruck–Ostermann | 半群 e^{−tA}e^{−sA}=e^{−(t+s)A}、收缩 ‖e^{−tA}‖≤1 | 学习式写入、内容寻址 |
| ETH 讲义 | 精确 ZOH、无离散误差 | 变 Δ、矩阵状态 |
| EFLA | Euler 一阶 vs 精确流、无额外参数、秩 1 闭式 | 物理 Δ、对角 Λ、双状态 |
| Mamba | "resolution invariance" 的断言（转引 S4ND） | 证明 |
| SSD | SSM–注意力对偶（连乘衰减 = 掩码） | 学习稳定性证明 |
| Schlag 2021 | 容量上界（L > d 时过载） | 时间语义 |
| GRU-D | 经验性的 max(0,·) 稳定性兜底 | 任何界 |

**候选若要立住，能补的定理只有三类**：(i) 脉冲 vs 常值两种写入语义下的连续极限与误差阶；(ii) ρ+α=1 约束下的半群/细化一致性；(iii) 双状态 (M,h) 耦合后的稳定性（现在谁都还没有）。

---

## 5. 判定与「要用它做贡献需新证什么」

### 5.1 公式对照表（谁写了哪一半）

| 来源 | ρ = e^{−ΛΔ}（衰减半） | α = 1−e^{−ΛΔ}（写入半） | 矩阵记忆 | 真实 Δ（秒） | 双状态 |
|---|---|---|---|---|---|
| ETH 讲义 / 控制教科书 | ✔（A_d = e^{A_cT_s}） | ✔（B_d = ∫_0^{T_s}e^{A_cτ}dτ·B_c） | ✘（向量） | ✔ | ✘ |
| Hochbruck–Ostermann | ✔（e^{−tA}，含半群与收缩） | ✔（φ_1 = ∫_0^Δ e^{−τA}dτ/Δ） | ✘（抽象算子） | ✔ | ✘ |
| S4 (2021) | ✘（双线性，非 ZOH） | ✘（B̄=(I−ΔA/2)^{−1}ΔB） | ✘ | Δ 是固定常数 | ✘ |
| **Mamba (2023)** | **✔ 式(4) Ā=exp(ΔA)** | **✔ 式(4) B̄=(ΔA)^{−1}(e^{ΔA}−I)ΔB** | ✘（向量） | ✘（学习伪时间） | ✘ |
| Mamba-2 / SSD (2024) | ✔（A_t = exp(Δ_t a)·I，标量） | 部分（B_t x_t 秩 1 写） | **✔** | ✘ | ✘ |
| Katharopoulos 2020 | ✘ | ✘（纯加性） | ✔ | ✘ | ✘ |
| Schlag 2021 | ✘ | ✘（β 独立学习门） | ✔ | ✘ | ✘ |
| NTM 2014 | ✘（按地址 erase） | ~（w·a 加性写） | ✔ | ✘ | **✔** |
| **EFLA 2025** | **✔ e^{−βA_t}（秩 1 A）** | **✔ 输入积分（闭式）** | **✔** | ✘（β 是步长） | ✘ |
| GRU-D 2018 | ✔ exp(−max(0,Wδ+b)) | ✘ | ✘ | **✔** | ✘ |
| **候选结构** | ✔ exp(−ΛΔ)，Λ 对角 | ✔ α = 1−e^{−ΛΔ}（自称） | ✔ | ✔（主张） | ✔ |

**读表要点：**
- **衰减半**有 6 个来源写过，**写入半**有 4 个（Mamba 与教科书同源）；**两半同时、且都在矩阵上**的只有 **EFLA**。
- 候选与 EFLA 的差别只有两个：Λ 对角（vs 秩 1）与 Δ 真实（vs 学习步长）。**前一个是特例化，后一个是把 Δ 换成输入——但 GRU-D 早在 2018 就把真实 Δ 塞进指数了。**

### 5.2 判定

1. **(ρ, α) 这对系数 = 教科书内容。** 出处：ZOH 精确离散化（ETH 讲义原话 "no discretization errors"）+ 指数积分器 φ_1（Hochbruck–Ostermann）。**其地位等同于"梯度下降 = 负梯度方向"，不可作为主张。**
2. **"用到矩阵记忆" = 已有论文。** EFLA（arXiv:2512.12602）已把 ZOH + 矩阵联想记忆 + 精确流做完，给出 ODE 与闭式解。
3. **"用真实 Δ 驱动衰减" = 已有论文（向量层面）。** GRU-D 式 (4)。
4. **"α 与衰减配套 ⇒ 半群/细化一致"这句话本身，暂未见到有人明确写出。** Schlag 的 β 是独立门、Mamba 的 Δ 是学习量、EFLA 谈 Euler vs 精确流但没有表述成「α 必须配套」。**这是唯一可能算候选自己观察的点——但它是一条两行恒等式的推论，撑不起"新颖结构"，只能撑一个 footnote 或一句 remark。**

### 5.3 若要用它做贡献，必须新证什么（不是新公式，是新证明）

1. **写入语义澄清 + 误差界**：α v k^T 是「区间常值驱动」还是「脉冲」？给出各自的连续极限、误差阶（O(Δ) 还是 O(Δ²)）、以及 Δ→0 时的差异。EFLA 只做了常值驱动 + Euler 对比；**脉冲情形没人做，而它直接给出 α≡1，会否掉候选的 α。**
2. **ρ+α=1 约束下的可学习性**：若把 α 参数化成 1−ρ（而非独立门），梯度是否退化、α 的梯度是否被 ρ 吞掉？需要分析与实验，光写恒等式没用。
3. **双状态耦合的稳定性**：M（矩阵，ZOH 衰减）与 h（隐状态）耦合系统在什么条件下稳定？现有矩阵记忆工作全是单状态，**这是真空白，也是真难**。
4. **Λ 与 Δ 的尺度不可辨识**：给出锚定方式，否则"真实时间"是修辞。

---

## 6. 核实状态清单

| # | 文献 | 核实级别 | 卡在哪 |
|---|---|---|---|
| 1 | Hochbruck & Ostermann 2010 | 全文（PDF） | — |
| 2 | Ducard, ETH S&S Lecture 1, 2018 | 全文（PDF） | 是讲义不是论文；引用宜写"标准 ZOH 结果，见该讲义/任何控制教科书" |
| 3 | S4 (Gu, Goel, Ré) | 全文（PDF） | **任务书前提有误**：S4 用双线性，不是 ZOH |
| 4 | Mamba | 全文（PDF），式 (4) 逐字 | Δ 的软加实现细节未逐字核 |
| 5 | Katharopoulos 2020 | 全文（PDF） | 递推式编号未逐字核（PDF 抽取丢编号），Σφ(K)V^T 的加性形式有原文句佐证 |
| 6 | Schlag 2021 | 全文（PDF），式 (20)–(25) 逐字 | — |
| 7 | NTM (Graves 2014) | 全文（PDF），式 (2)(3)(4) 逐字 | — |
| 8 | EFLA (arXiv:2512.12602) | 全文（HTML），式 (9)(10)(11)(31)、附录 D/E 逐字 | **v5 的发表状态/会议未知**，引用前需确认是否已有正式版本 |
| 9 | Mamba-2 / SSD | 全文（PDF+HTML） | A_t = exp(Δ_t a) 是「同一 ZOH 规则 + 标量×单位阵」的**组合推断**，原文只写 "same discretization step as Gu & Dao (2023)" |
| 10 | GRU-D | 全文（PDF），式 (4) 逐字 | 期刊版页码未核（只核到 arXiv 版） |

**给 lead 的两处更正：**
1. 任务书说「S4 的离散化是 Ā=exp(ΔA)」——**不对**。S4 正文 §2.3 明确用双线性法；ZOH（exp(ΔA)）是 **Mamba** 才明确采用的。若把 S4 当 ZOH 引用，会被熟悉 SSM 的审稿人纠正。
2. 任务书第 1 点写「B̄=((ΔA)^{-1}(exp(ΔA)-I))ΔB」本身是对的，但它**不是新推导**：它就是 ETH 讲义里 B_d=(∫_0^{T_s}e^{A_cτ}dτ)B_c 的算子写法（把 A_c 换成 ΔA）。


# 08 容量、干扰与表达力的上界 —— 对「矩阵快权重 + Δ 衰减 + 双状态」候选结构的反对证据

调研者：pc-error。时间：2026-11-03。仅网络检索（tavily_search / tavily_extract），未训练、未写代码。
证据等级：**[正文级]**=拉到正文并读到公式/定理；**[仅摘要级]**；**[二手]**=原文献拿不到，引自可核实的二手叙述（会标明出处）。

---

## 0. 结论摘要（负面优先）

1. **容量有硬上界，且已经有人把它算准了。** d×d 线性记忆在 top-1 检索下需要 `d² ≳ n log n`；**任何** W∈R^{d×d} 都不能在 `d² ≤ C·n log n`（C<2/π）时做到 top-1 检索（Barnfield et al. 2026, arXiv:2605.05189, Theorem 1(ii)）。也就是说，候选结构的矩阵记忆能存的「可区分」key-value 对数约为 `d²/log(d²)`，不是无限。
2. **候选的写规则是 1991 年的东西，且已被证明有 overcapacity 区。** `M ← M + v⊗k` 就是 Schmidhuber 1991 的 fast weight 写指令；Schlag et al. 2021 明确写：键嵌入 d 维空间时，「不可能有超过 d 个正交向量……存超过 d 个关联就会产生检索误差」，并把这个区叫 **overcapacity regime**。
3. **矩阵状态不带来额外表达力，只带来 d² 个自由度。** 线性注意力/快权重就是「矩阵状态的线性 RNN」；而线性 SSM（含对角衰减转移）**在 L-uniform TC0 内**，不能做 S5 置换复合这类 state tracking（Merrill et al. 2024, ICML）。候选的 `exp(-ΛΔ)` 是数据无关对角转移，正落在被证明受限的那一类里。
4. **衰减率与容量的定量关系没有定理。** 指数衰减的有效视界 ≈ `1/(1-γ)`（几何级数，初等结论，非定理），但「衰减率 ↔ 可存关联数」的严格界，本专题没找到。RetNet 的 γ 是**按位置、固定每头**的，GRU-D 的 γ 才是**按真实 Δt**、逐维学的。候选的 Δ 落点是**GRU-D 式的**，不是 RetNet 式的。
5. **「少写几次更稳」是错的直觉方向。** Skip RNN 证明跳更新可以保持甚至提升性能，但前提是**学出一个跳的门**且任务允许；而在衰减记忆里，不写 + 衰减 = 双重丢失（旧项指数消失、新项没进来）。本专题没有找到「低频写入误差累积」的严格定理，只有零散的实验/工程证据——**这是证据缺口，不要当作已证结论用**。
6. **没有「逐字对应」的单篇：**最接近的是 **FWP（Schlag et al. 2021）的 Eq 2/17 + delta 规则 Eq 20–22**（写/读/门控全同构，缺 Δ 衰减）与 **GRU-D 的 Eq 4**（Δ 衰减，缺矩阵写）。候选 = 两者的并集，而并集本身没有现成文献。

---

## 1. 被检验的候选结构

    M_i- = exp(-Lambda * Delta_i) ⊙ M_{i-1}      # d×d 矩阵快权重，逐通道衰减
    q_i = Q(z_i, h_i); k_i = K(z_i, h_i); v_i = V(z_i, h_i)
    M_i = M_i- + alpha_i * v_i k_i^T            # 内容寻址写入
    r_i = R(M_i, q_i)                            # 读
    h_i = H(h_{i-1}, r_i, z_i)                   # 另一路隐状态

本文件只回答三件事：**能存多少（容量）**、**表达力到哪（上界）**、**衰减与更新频率的代价**。写/读规则本身的先例（FWP、RetNet、DeltaNet）只在必要处给最小对照。

---

## 2. 逐篇条目

### C1. Sharp Capacity Thresholds in Linear Associative Memory
- **作者 / 年份 / 会议或期刊 / 链接**：Nicholas Barnfield, Juno Kim, Eshaan Nichani, Jason D. Lee, Yue M. Lu / 2026（v1 2026-05-06）/ arXiv:2605.05189（v1 副标题 "From Winner-Take-All to Listwise Retrieval"；v2/abs 页副标题 "From Top-1 Retrieval to Tail-Average Learning"，标题版本不一致，已注明）/ https://arxiv.org/abs/2605.05189 **[正文级]**
- **记忆状态**：单个 `W ∈ R^{d×d}`（d² 个参数），存 n 个 (u_i, v_i) ⊂ R^d × R^d 对。
- **读操作精确公式**：`s_{j,i} = u_jᵀ W v_i`（双线性打分，全对全打分矩阵 S）——注意读是**双线性**的，不是 `M q`。
- **写操作精确公式**：correlation matrix memory：把 key-target 外积**叠加**进 W（`W = Σ_i v_i u_iᵀ` 形式；原文称 "superposing key-target outer products"，沿用 Kohonen 1972）。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：**是（退化形式）**——累加，没有 α，没有衰减。
- **α_t 归类**：**无**（隐含 α=1）。
- **用物理时间 Δt 还是观测序号**：观测序号（i.i.d. 采样对，无时间）。
- **若用 Δt，影响**：不适用。
- **支持不规则采样/丢帧/重复帧吗**：不适用（无时间轴）。
- **有无定理**：**有，而且是本专题最锋利的一条。**
  - **Proposition 1**（相关矩阵记忆的检索阈值）：`d² > (8+ε) n log n` 时 top-1 检索成功；`d² < (8−ε) n log n` 时失败——**尖锐相变**。
  - **Theorem 1(ii)**（top-1 检索阈值，负方向）：对任意 `C < 2/π`，若 `d² ≤ C·n log n`，则**不存在任何** `W ∈ R^{d×d}` 能以高概率做到 top-1 检索。
  - **listwise 检索（TAM）**：容量改为 `d² ≍ n`（无 log 因子）。
  - **猜想（作者自标 conjectural）**：尖锐 top-1 阈值 `d² ~ 2 n log n`。
  - 对比既往：Nichani et al. 2024 只给出充分条件 `d² ≥ C n (log n)^4`；本文把它压到对数阶并给出下界。
- **用于**：理论（Transformer 事实回忆、线性记忆的容量理论）。
- **与候选结构异同**：**同形**——d×d 矩阵 + 外积叠加写 + 内容寻址读。差异：无 α 门控、无 Δ 衰减、无第二状态 h。其读是双线性（`uᵀWv`），候选是 `M q`（矩阵-向量），**候选的读更弱**，因此在同等 d 下容量不会更好。
- **裁决**：**部分覆盖**——它不含衰减/门控/Δ，但**直接给出候选矩阵记忆的容量标度**：≈ d²/log(d²) 个可 top-1 区分的关联，且「log 因子不可去」由定理保证。这是对「矩阵记忆容量大」这一直觉最硬的定量反驳。

### C2. Linear Transformers Are Secretly Fast Weight Programmers
- **作者 / 年份 / 会议或期刊 / 链接**：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139 / https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf **[正文级]**
- **记忆状态**：`W^(i) ∈ R^{d_out × d_in}`；线性 Transformer 版为 `R^{d_v × d_k}` 矩阵（外加归一化累加器 `z`）。
- **读操作精确公式**：`y^(i) = W^(i) x^(i)`（式 3）；线性注意力版 `y^(i) = (1/z^(i)) · φ(q^(i))ᵀ W^(i) φ(q^(i))`（式 19，抽取中上标有丢字，语义由式 17–19 确认）。
- **写操作精确公式**：`W^(i) = σ( W^(i−1) + a^(i) ⊗ b^(i) )`（式 2）；线性注意力版 `W^(i) = W^(i−1) + v^(i) ⊗ φ(k^(i))`，`z^(i) = z^(i−1) + φ(k^(i))`（式 17–18）。**这就是候选 `M_i = M_i- + α v kᵀ` 的原型**（α 取 1，σ 可有可无）。
- **delta 规则版写**：`v̄^(i) = W^(i−1) φ(k^(i))`（式 20，先读旧值）；`β^(i) = σ(W_β x^(i))`（式 21，写强度）；`v_new^(i) = β^(i) v^(i) + (1−β^(i)) v̄^(i)`（式 22）；再把修正量写回（式 23 附近，式号在抽取中缺失，**结构由式 20–22 与正文 "error-correcting delta rule" 确认**）。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：**是，逐字同构**。
- **α_t 归类**：**学出的门**（β^(i) 是 sigmoid 出来的 "write-strength"，正文原话 "the ‘write-strength’"）。
- **用物理时间 Δt 还是观测序号**：观测序号（token index i）。
- **若用 Δt，影响**：不适用（无 Δ）。
- **支持不规则采样/丢帧/重复帧吗**：不适用。
- **有无定理**：**有容量论证（非严格定理）**：正文 §4.1 —— *"Endlessly adding new associations to a memory of finite size, as in Eq. 17, inevitably will reach a limit... With keys embedded in a d_dot space, there cannot be more than d_dot orthogonal vectors. That is, storing more than d_dot associations will result in a retrieval error."* 并把这叫 **overcapacity regime**；形式化转引 Smolensky 1990 的 Theorem 3.1/3.3（crosstalk/retrieval error）。
- **用于**：语言建模、合成检索任务。
- **与候选结构异同**：写（外积累加）、读（矩阵-向量）、**门控写强度**三件全同构；差别是 (a) 没有 `exp(-ΛΔ)` 衰减，(b) 这篇明确指出纯加性写在 overcapacity 下 sub-optimal，需要用 delta 规则替换——**候选用的正是被它判为次优的那种写**。
- **裁决**：**完整覆盖（除 Δ 衰减）**——候选的 M 更新式在这篇里逐字存在（式 17），α 的门控形态也在式 21 里，且给出了容量上限 d 与 overcapacity 判据。

### C3. Parallelizing Linear Transformers with the Delta Rule over Sequence Length（DeltaNet）
- **作者 / 年份 / 会议或期刊 / 链接**：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim / 2024（v6 2025-01-15）/ arXiv:2406.06484 / https://arxiv.org/pdf/2406.06484 **[正文级（摘要与框架）；公式式号未取到]**
- **记忆状态**：`S ∈ R^{d_v × d_k}` 矩阵状态（"linear RNN with matrix-valued hidden states"）。
- **读操作精确公式**：`o_t = S_t q_t`（线性注意力/快权重的标准读；本文用矩阵状态记号）。
- **写操作精确公式**：把加性更新换成 **delta 规则**，用 Householder 矩阵表示：`S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`。**[部分核实]**：Householder 形式在本文摘要中被明确点名（"memory-efficient representation for computing products of Householder matrices"），精确式号未从 PDF 抽取到；上式与 FWP 式 20–22 的 delta 规则一致。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：**是**（delta 形式）。
- **α_t 归类**：**学出的门**（β_t）。
- **用物理时间 Δt**：否，观测序号。
- **若用 Δt，影响**：不适用。
- **支持不规则采样吗**：不适用。
- **有无定理**：**无**（工程/实证论文）。可引的实证结论：(a) 线性 Transformer/SSM "still underperform transformers especially on tasks that require in-context retrieval"；(b) delta 规则版在 associative recall 上更好；(c) 1.3B 模型优于 Mamba/GLA，但**要打赢强 Transformer 仍需混入滑窗/全局注意力**。
- **用于**：语言建模。
- **与候选结构异同**：候选的写是**加性 Hebb**（`+ α v kᵀ`），DeltaNet 证明该换成**读-改-写**（先读旧值、再插值写回）。差别正在候选最弱的一环：加性写在重复 key 上会累积干扰，delta 规则不会。
- **裁决**：**部分覆盖**——给出「同一矩阵记忆、更好的写规则」的强对照，并实证 delta 规则优于加性；这对候选的 `+ α v kᵀ` 是**直接的不利证据**（同类结构里更好的写法已存在）。

### C4. Neural networks and physical systems with emergent collective computational abilities（Hopfield 1982）
- **作者 / 年份 / 会议或期刊 / 链接**：John J. Hopfield / 1982 / PNAS 79(8):2554–2558 / https://www.pnas.org/doi/10.1073/pnas.79.8.2554 **[二手]**（PNAS 页面本次 extract 返回空；卷期页码 79(8):2554–2558, 1982 经 PubMed/PMC 记录核实 https://pmc.ncbi.nlm.nih.gov/articles/PMC346238 ；容量常数经二手文献核实，见下）
- **记忆状态**：`N×N` 对称权重矩阵（但**状态是 N 维向量**，不是矩阵记忆）。
- **读操作精确公式**：`s_i(t+1) = sgn( Σ_j W_ij s_j(t) )`（异步更新）。
- **写操作精确公式**：Hebb 规则 `W_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ`（i≠j）。
- **是否存在 M(t+1)=F(...)**：是（权重一次性由模式算出，之后不再写）。
- **α_t 归类**：无。
- **用物理时间 Δt**：否。
- **支持不规则采样**：不适用。
- **有无定理**：**有（经典容量界）**：Amit, Gutfreund & Sompolinsky (1985) 的复本方法给出临界存储容量 `α_c = p/N = 0.138`；超过即「完全失忆」相变。**[二手]**：常数与相变叙述取自 MIT 课程项目报告（https://web.mit.edu/8.592/www/grades/projects/Projects(2007)/KevinTakasaki.pdf ，该文复述 Amit et al. 的式 6 与 α_c=0.138），**未直接核 AMIT 1985 原文**。
- **用于**：联想记忆 / 神经科学。
- **与候选结构异同**：共同点是「用外积叠加写记忆」；差异是 Hopfield 的状态是**读出的向量**且读是非线性的（sgn），容量是 O(N) 而**不是** O(N²)。这给出一个重要的量级直觉：**外积记忆的容量标度永远跟着「有效维度」走，不会跟着「参数个数」走**。
- **裁决**：**未覆盖**（结构不同），但提供经典容量标度作参照：0.138N（非线性、向量态）与 d²/log n（线性、矩阵态）是同一族上界的两个端点。

### C5. The Illusion of State in State-Space Models
- **作者 / 年份 / 会议或期刊 / 链接**：William Merrill, Jackson Petty, Ashish Sabharwal / 2024 / ICML 2024, PMLR 235 / https://arxiv.org/pdf/2404.08819v3 （abs: https://arxiv.org/abs/2404.08819）**[正文级]**
- **记忆状态**：SSM 的隐状态 `x_t ∈ R^d`（S4/Mamba 系列），转移 `x_t = Ā x_{t-1} + B̄ u_t`。
- **读操作精确公式**：`y_t = C x_t`（线性读出）。
- **写操作精确公式**：输入注入 `x_t = Ā x_{t-1} + B̄ u_t`；**转移矩阵 Ā 与输入无关**（这是关键）。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：是（线性 SSM 是它的特例，无门控地址）。
- **α_t 归类**：**其他**（Mamba 的 Δ 是输入相关的**离散化步长**，不是物理时间；它只影响 `Ā = exp(ΔA)`，不承载 wall-clock 语义）。
- **用物理时间 Δt 还是观测序号**：观测序号。
- **若用 Δt，影响**：不适用（论文不涉及真实时间）。
- **支持不规则采样/丢帧吗**：不适用。
- **有无定理**：**有，核心定理。** 线性 SSM 及其近亲属 **L-uniform TC0**；因此不能解 **NC1-hard** 问题，包括 **S5 置换复合**（state tracking），从而不能做「source-target 记谱的国际象棋跟踪、代码求值、长叙事实体跟踪」。论文还澄清：Gu et al. 2021 的「SSM 可模拟 RNN」依赖**无限深度**；在**有界层数**下不成立。唯一补救是让转移矩阵**输入相关**（Liquid-S4 式），单层即可学 S5。
- **用于**：语言建模 / 序列建模。
- **与候选结构异同**：候选的衰减转移 `exp(-ΛΔ)` 是**数据无关的对角转移**——正好是被定理覆盖的那一类。即使 M 是矩阵态，`M ← exp(-ΛΔ)⊙M + α v kᵀ` 整体仍是**输入只进入加性项**的线性递推；读是线性读出。按该文框架，候选落在 TC0 内，**不能做置换复合类 state tracking**。
- **裁决**：**未覆盖候选的具体结构，但给出其表达力上界**：矩阵记忆 + 对角衰减不改变复杂度类。这是对「矩阵状态 = 更强状态跟踪」最直接的否定。

### C6. The Expressive Capacity of State Space Models: A Formal Language Perspective
- **作者 / 年份 / 会议或期刊 / 链接**：Yash Sarrof, Yana Veitsman, Michael Hahn / 2024 / NeurIPS 2024 / arXiv:2405.17394 / https://arxiv.org/abs/2405.17394 （HTML: https://arxiv.org/html/2405.17394v3）**[正文级（摘要+定理陈述）；证明未逐式读]**
- **记忆状态**：SSM 隐状态 `x_t ∈ R^d`，转移 `A(x_t)` 通常**对角或输入无关**（论文指出 Mamba 的 Ā 是对角，可令 `A(x_t)_i = Ā_ii`）。
- **读/写**：与 C5 同族，无独立外积写。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：是（SSM 递推）。
- **α_t 归类**：**其他**（门控/对角衰减，非内容寻址）。
- **用物理时间 Δt**：否。
- **有无定理**：**有三条，方向比 C5 更细**：
  - **Theorem 3.1**：SSM 能表示 **Flip Flop** 语言（write/read/ignore 型 state tracking）的**长度泛化**解——比 Transformer 强；
  - **Theorem 3.2**：**PARITY（模计数）对现有 SSM 困难**，原因是某个**设计选择**（转移矩阵与输入无关），使它们表现得像 Transformer；
  - **Theorem 3.6**：SSM 能建模**有界层级结构**且内存最优，无需模拟栈。
- **用于**：形式语言 / 语言模型的理论能力。
- **与候选结构异同**：候选的衰减转移是 `exp(-ΛΔ)`，**数据无关对角**——与 C6 指认的「限制表达力的设计选择」完全同类。因此预期：候选能做 Flip Flop 类（若状态容量够），**不能做 PARITY/模计数**。
- **裁决**：**部分覆盖**——给出「哪类 state tracking 能、哪类不能」的精确分界；候选的 Δ 衰减属于被点名受限的那一支。

### C7. Zoology: Measuring and Improving Recall in Efficient Language Models
- **作者 / 年份 / 会议或期刊 / 链接**：Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Ré / 2023（arXiv）/ ICLR 2024（PDF 页眉 "Published as a conference paper at ICLR 2024"）/ https://arxiv.org/html/2312.04927v1 ；作者与会议经 ICLR 2024 正式 PDF 核实 https://proceedings.iclr.cc/paper_files/paper/2024/file/448fc91f669c15d10364ee01d512cc10-Paper-Conference.pdf **[正文级（摘要+第 4 节理论）]**
- **记忆状态**：gated-convolution 类（H3、Hyena、RWKV、RetNet）的**固定尺寸状态/滤波**；对照 attention 的 KV cache。
- **读/写**：不适用（不是显式外积记忆）；核心是「输入与固定滤波的卷积 + 门控」。
- **是否存在 M(t+1)=F(...)**：**部分**（状态是卷积/门控累积，不是内容寻址写）。
- **α_t 归类**：**学出的门**（gating），非内容寻址。
- **用物理时间 Δt**：否，位置。
- **有无定理**：**有**：
  - **Theorem 4.4（Data-Independent Filters）**：**BaseConv**（可表示 H3/Hyena/RWKV/RetNet 等）解 **MQAR** 所需的**模型维度随输入序列长度增长**；
  - **Proposition 4.3**：attention 解 MQAR 只需**与序列长度无关的维度**；
  - **Theorem 4.5**：改成**输入相关的滤波**后可高效解 MQAR。
  - 实证：*"a 70M parameter attention model outperforms a 1.4 billion parameter gated-convolution model on associative recall"*；*"82% of the gap is explained by ... associative recall"*。
- **用于**：语言建模。
- **与候选结构异同**：**这条要小心用**——候选的 `k,v,q` 是输入相关的（内容寻址），所以**不受 Theorem 4.4 的固定滤波限制**；但候选的**衰减项** `exp(-ΛΔ)` 是数据无关的，与 BaseConv 的固定滤波同侧。**含义**：内容寻址能救读，救不了衰减带来的一次性信息损失。
- **裁决**：**部分覆盖**——对「固定滤波/衰减」那一半给出维度必须随长度增长的定理；对「内容寻址」那一半反而给出支持（输入相关是必需的）。

### C8. Retentive Network: A Successor to Transformer for Large Language Models（RetNet）
- **作者 / 年份 / 会议或期刊 / 链接**：Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei / 2023 / arXiv:2307.08621 / https://arxiv.org/pdf/2307.08621 **[正文级]**
- **记忆状态**：`S_n ∈ R^{d×d}` 矩阵状态（每头 d），**就是候选 M 的直接同构体**。
- **读操作精确公式**：`Retention(X_n) = Q_n S_n`（式 6 下半）。
- **写操作精确公式**：`S_n = γ S_{n-1} + K_nᵀ V_n`（式 6）。**这就是候选的 `M_i = exp(-ΛΔ)⊙M_{i-1} + v_i k_iᵀ`**，只是把 `exp(-ΛΔ)` 换成一个**标量 γ**。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：**是，最接近的一篇**（除 α 外全同形：γ 可看作固定 α 衰减）。
- **α_t 归类**：**其他**（写入权重固定为 1；衰减 γ 是**每头固定的标量超参**，由 multi-scale retention 分给不同头，**不是学的、更不是 Δt**）。
- **用物理时间 Δt 还是观测序号**：**观测序号**——衰减是 `γ^{n-m}`（式 5 的 `D_nm = γ^{n-m}`），按**相对位置**算，与真实时间无关。
- **若用 Δt，影响**：不适用（这正是候选与它的分水岭）。
- **支持不规则采样/丢帧/重复帧吗**：**不支持**——位置差 n−m 与真实时间解耦，掉帧后 γ^{n−m} 与真实间隔完全错配。
- **有无定理**：**无容量/衰减定理**（工程论文，只推导了并行↔递归↔分块的等价）。
- **用于**：语言建模。
- **与候选结构异同**：**衰减 + 外积写 + 矩阵态 + 内容读**四件全同构；唯一差别是**衰减的时间语义**：RetNet 按位置，候选按 Δt。**因此候选在「结构」上没有新意，全部新意押在「Δ 落点」上**。
- **裁决**：**完整覆盖**（除 Δ 语义）——候选的 M 侧更新式与 RetNet 式 6 同构，且 RetNet 已是大规模验证过的架构。

### C9. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- **作者 / 年份 / 会议或期刊 / 链接**：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / Scientific Reports（https://www.nature.com/articles/s41598-018-24271-9；预印本 arXiv:1606.01865）/ 正文用预印本 https://arxiv.org/pdf/1606.01865 **[正文级]**
- **记忆状态**：**向量**隐状态 `h_t ∈ R^d`（不是矩阵）。
- **读操作精确公式**：不适用（无联想读）。
- **写操作精确公式**：隐状态按真实间隔衰减后再更新：`γ_t = exp{ −max(0, W_γ δ_t + b_γ) }`（式 4，`δ_t` 是距上次观测的时间间隔），`h_{t-1} ← γ_t ⊙ h_{t-1}`，再走标准 GRU 更新。
- **是否存在 M(t+1)=F(M_t,k_t,v_t,α_t)**：F 存在，但 M 是向量、无外积写。
- **α_t 归类**：**外部真实 Δt**（`δ_t` 直接进指数），且衰减率**逐维可学**（`W_γ`）。
- **用物理时间 Δt 还是观测序号**：**真实 Δt（秒）**。
- **若用 Δt，影响**：**衰减**（不是写强度、不是 key/value）。
- **支持不规则采样/丢帧/重复帧吗**：**支持，且这正是它的动机**（missing values / informative missingness）。
- **有无定理**：**无**（方法论文，实证为主）。
- **用于**：医疗时间序列 / 传感器。
- **与候选结构异同**：**候选的 Δ 落点就是这条**——`exp(-Λ Δ)` 与 `exp(-W_γ δ)` 同构（一个逐通道对角、一个逐维向量）。差别：GRU-D 作用在**向量隐状态**上、没有外积记忆写。所以「Δ 控制衰减」不是新东西，**2018 年就有**；候选只是把它搬到矩阵记忆上。
- **裁决**：**部分覆盖**——Δ→衰减的先例已存在且形式几乎相同；未覆盖「矩阵外积写 + 双状态」。

### C10. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Víctor Campos, Brendan Jou, Xavier Giró-i-Nieto, Jordi Torres, Shih-Fu Chang / 2018 / ICLR 2018 / arXiv:1708.06834 / https://arxiv.org/pdf/1708.06834 **[正文级（摘要+引言）]**
- **记忆状态**：向量隐状态 + 二值 skip 门。
- **读操作精确公式**：不适用。
- **写操作精确公式**：`h_t = u_t ⊙ h_{t-1} + (1 − u_t) ⊙ h̃_t`，`u_t ∈ {0,1}` 由学出的门决定（skip 时 `u_t=1`，状态不更新）。
- **是否存在 M(t+1)=F(...)**：部分（可选更新）。
- **α_t 归类**：**学出的门**（skip gate，可加 budget 约束）。
- **用物理时间 Δt**：否（按步/帧）。
- **有无定理**：**无**。实证结论：学习跳更新可以 *"reduce the number of required RNN updates while preserving, and sometimes even improving, the performance"*。
- **用于**：序列建模/视频。
- **与候选结构异同**：候选没有 skip 门，但「不反复重写不变内容」在效果上等价于**自适应 skip**。这篇的结论对候选**既支持又警告**：支持——少更新可以不掉性能；警告——**必须有学出来的门**，而候选的门是 `α_i`（写强度）而不是「跳过」，两者不同：`α=0` 等于跳过，但候选没有机制让 `α` 精确为 0 并保持稳定。
- **裁决**：**未覆盖**（结构不同），但对「更新频率」这条直觉给出**双向证据**，且指出关键装置（学出的跳门）候选缺失。

---

## 3. 横向对照表

| # | 工作 | 记忆状态 | 写规则 | 读规则 | α 类别 | Δ 落点 | 定理强度 | 对候选的裁决 |
|---|---|---|---|---|---|---|---|---|
| C1 | Barnfield et al. 2026 | d×d 矩阵 | 外积叠加（Hebb） | 双线性 uᵀWv | 无 | 无 | **强定理**（尖锐阈值） | 部分覆盖：容量标度 |
| C2 | FWP (Schlag 2021) | d_out×d_in | 外积累加 / delta | y=Wx | **学出的门 β** | 无 | 容量论证（转引 Smolensky） | 完整覆盖（除 Δ） |
| C3 | DeltaNet (Yang 2024) | d_v×d_k | **delta/Householder** | S q | 学出的门 β_t | 无 | 无（实证） | 部分覆盖：更好写规则 |
| C4 | Hopfield 1982 / Amit 1985 | N 维向量 | Hebb 外积 | sgn(Ws) | 无 | 无 | **强定理** α_c=0.138 | 未覆盖 |
| C5 | Merrill et al. 2024 | SSM 状态 | Āx+B̄u | Cx | 无（Mamba Δ 是步长） | 无 | **强定理** TC0/S5 | 未覆盖但上界适用 |
| C6 | Sarrof et al. 2024 | SSM 状态 | 同上 | 同上 | 无 | 无 | **强定理** PARITY/FlipFlop | 部分覆盖 |
| C7 | Zoology (Arora 2023) | 固定滤波/gated | 卷积累积 | — | 学出的门 | 无 | **强定理** MQAR 维度 | 部分覆盖（一半） |
| C8 | RetNet 2023 | d×d 矩阵 | **γ 衰减 + 外积** | QₙSₙ | 无（γ 固定） | **位置** | 无 | **完整覆盖**（除 Δ） |
| C9 | GRU-D 2018 | 向量 | 按 Δ 衰减退火 | — | **外部真实 Δt** | **衰减** | 无 | 部分覆盖（Δ 落点） |
| C10 | Skip RNN 2018 | 向量 + 跳门 | 可选更新 | — | **学出的门** | 无 | 无 | 未覆盖 |

**读表结论**：候选 = C8 的矩阵/衰减骨架 ⊕ C9 的 Δ 语义 ⊕ C2/C3 的门控写。三块都有人做过，**拼起来没人做过**——但拼接本身在容量/表达力上不会超过任何一块的上界。

---

## 4. 反对结论与反例

### 4.1 【硬】容量上界：候选的矩阵记忆能存多少，取决于**读的方式**

C1 的定理容易被误读：它给的是 **双线性读** `uᵀWv`（对 key 侧和 query 侧都用了一次投影）下的 `d² ≍ n log n`。候选的读是 `r = R(M, q)`，若它是**矩阵-向量**读（线性注意力那种），有效容量**不是** d²，而是更低：

    设 M = Σ_i v_i k_iᵀ，读 r = M q，keys 单位范数、随机。
    信号项：v_i (k_iᵀ k_i) = v_i·1
    串扰项：Σ_{j≠i} v_j (k_jᵀ k_i)，每个内积 ~ N(0, 1/d)，n−1 项求和 ⇒ 标准差 ≈ sqrt((n−1)/d)
    信噪比 ≈ sqrt(d / n)  ⇒ 可靠检索要求 n = O(d)（对数因子之内）

**[本文推导，非论文结论；假设随机 keys、线性读出]** 这条与 C2 的判据一致：*"With keys embedded in a d_dot space, there cannot be more than d_dot orthogonal vectors. That is, storing more than d_dot associations will result in a retrieval error."*

**所以对候选有两条不同的上界，取决于 R 的形式：**

| 读形式 | 容量标度 | 出处 |
|---|---|---|
| `r = M q`（线性读出，候选最可能的形式） | **n = O(d)** | C2 正交性论证 + 本文 SNR 推导 |
| `r = u(z)ᵀ M v(z)`（双线性） | **n = O(d²/log(d²))**，且任何 W 都突破不了 | C1 Theorem 1(ii) |

**含义**：无论哪种，「矩阵记忆 = 海量容量」都不成立。d=256 时，双线性读也只有 ~10³ 量级关联（按 C1 的常数 8 粗算 `n≈d²/(8 ln n)`，**本文算术**，非论文给的数值）；线性读则只有 ~10² 量级。**这是对「矩阵快权重当长期记忆」最硬的定量反驳。**

### 4.2 干扰随写入次数增长，且加性写是最差的一种

- 串扰项随 n 增长（上节）：`n` 越大，读出的值越是被别人污染。
- C2 明确把这叫 overcapacity 区，并说纯加性指令在该区 *"sub-optimal"*，主张换成 delta 规则；C3 实证 delta 规则在 associative recall 上更好，且大规模 LM 里也更强。
- **对候选的直接不利**：候选的写是 `+ α v kᵀ`（加性 Hebb），**不是** delta（读-改-写）。在重复/近邻 key 上，加性写会不断叠加而非覆盖，干扰单调增长；这正是 C2/C3 已经指出并被替代掉的写法。

### 4.3 【硬】矩阵状态不提升表达力：它只是 d² 维向量

- `M ∈ R^{d×d}` 是 d² 个数；作为状态它等价于一个 d² 维向量，**唯一的结构差异是写入被约束成秩 1**——这是**限制**，不是优势。
- 线性注意力/快权重与「矩阵状态的线性 RNN」**严格等价**（C2 式 17–19），所以「矩阵」没有引入新计算原语。
- C5 的定理：这类线性递推（转移与输入无关）在 **L-uniform TC0** 内，不能做 **S5 置换复合**等 NC1-hard 的 state tracking。**d² 再大也不改变复杂度类**——这是维度论证救不了的。
- C6 补充：即便 SSM 能做 Flip Flop（Thm 3.1）与有界层级（Thm 3.6），**PARITY/模计数**仍因「转移与输入无关」而困难（Thm 3.2）。候选的 `exp(-ΛΔ)` 正是这种转移。
- **唯一能改变复杂度类的最小改动**（C5 原文）：让**转移矩阵输入相关**（Liquid-S4 式）。候选没有这一步。

### 4.4 衰减：没有定理，只有工程直觉；而且它和容量是竞争关系

- **没找到定理。** 本专题检索到的文献里，**没有**把「衰减率 ↔ 可存关联数」写成严格界的；C8（RetNet）只有 γ 的工程选择，C9（GRU-D）只有衰减的实证收益。**这是证据缺口，写论文时不能说「有理论保证」。**
- 初等结论（可自行推导）：几何衰减下第 m 步的贡献权重是 `γ^m`，有效视界 ≈ `1/(1−γ)`；`γ=0.99` 时约 100 步，`γ=0.9` 时约 10 步。**衰减越快，长期记忆越短**——这与「长期记忆」的目标直接冲突。
- 双重损失：衰减 + 不写 ⇒ 旧项指数消失且新项没进来。**候选的「不变内容不反复重写」在衰减记忆里等于「让它慢慢消失」**，除非 Δ 很小。这不是 bug，是设计上的张力：候选用 Δ 同时控制衰减与（隐含的）写入时机，两者会耦合。
- 不规则采样下更糟：`exp(-ΛΔ)` 对长间隔给接近 0 的权重，一次长掉帧就把记忆清零；而 RetNet 的位置式 γ^{n−m} 至少对「均匀采样」是自洽的。**候选在这个方向上比 RetNet 更脆，不是更稳。**

### 4.5 更新频率：「少写」能成立，但需要**学出的跳门**，候选没有

- C10 的实证结论是**正面**的：学习跳更新可以 *"reduce the number of required RNN updates while preserving, and sometimes even improving, the performance"*。**所以「不反复重写」不是错的直觉。**
- 但 C10 的装置是**二值 skip 门 + budget 约束**；候选只有连续的 `α_i`（写强度）。`α_i→0` 在数学上等价于跳写，但 (a) 没有稀疏性保证，(b) 没有 budget 约束，(c) 写强度与内容相关而非与「是否需要更新」相关。
- **没有找到「低频写入 ⇒ 误差累积」的定理**。相关的只有：衰减使不写=丢失（§4.4）；以及 C7 的 MQAR 定理说明**读侧**的维度必须随需要的检索距离增长。**「更新越少越不稳定」目前只是合理猜测，不是已证结论**——不要把它写成本文的结论。

### 4.6 双状态（M + h）：容量不叠加，表达力不升级

- `h_i = H(h_{i-1}, r_i, z_i)` 是从 M 的读出再算出来的；若 `r_i` 已经由 M 决定，h 的额外信息只来自 `z_i` 与 h 自身。**M 与 h 的信息不是简单相加**：h 是 M 读出的低维投影（若 H 是线性/门控）。
- 表达力上：多一个向量状态不改变 C5 的复杂度类结论（TC0 对「常数个线性递推 + 逐点非线性」封闭）。要跳出 TC0 必须动**转移的输入相关性**，不是加状态。
- **没有找到任何文献论证「矩阵记忆 + 向量隐状态」比单一状态严格更强**。这是本专题的一个空白，也意味着候选拿不到「双状态⇒更强」的现成支撑。

### 4.7 可证伪的预测（如果要实验，这几条最省力且最能打击候选）

1. **MQAR / associative recall**：固定 d，扫描序列长度 n。预测：n 超过 ~d（线性读）后准确率断崖；C1 的阈值形状是相变而非缓降。
2. **S5 置换复合 / PARITY**：单层/多层候选都应失败（C5/C6）。这是**定理级**预测，实验只是确认。
3. **不规则采样 + 长间隔**：人为插入长 Δ 后，之前写入的关联应被 `exp(-ΛΔ)` 清掉——测试「长掉帧后旧记忆是否还可读」。
4. **重复 key 覆盖**：同一 key 写入不同 value 多次，加性写会读到「加权混合」而非最新值；换成 delta 规则应能修好（C2/C3）。**这条直接检验候选的写规则是否该换。**
5. **更新频率扫描**：固定 Δ 总量，改变写入次数（每次写 α/n，或只写 1/n 的步）。若性能随写入次数下降，说明「少写」不是免费的；若不变，说明候选的衰减主导了一切。

---

## 5. 定理 / 实验 / 缺口 三分表

| 结论 | 类型 | 出处 |
|---|---|---|
| d×d 线性记忆 top-1 检索需 d² ≳ n log n；任何 W 都突破不了 C n log n（C<2/π） | **定理** | C1 Theorem 1(ii) + Prop 1 |
| 相关矩阵记忆在 (8±ε)n log n 处尖锐相变 | **定理** | C1 Prop 1 |
| listwise 检索容量 d² ≍ n | **定理** | C1 |
| 线性 SSM/矩阵态线性递推 ∈ L-uniform TC0，不能 S5 | **定理** | C5 |
| SSM 可做 Flip Flop、不可（易）做 PARITY；有界层级可做 | **定理** | C6 Thm 3.1/3.2/3.6 |
| BaseConv 类解 MQAR 需维度随序列长度增长；attention 不需要 | **定理** | C7 Thm 4.4 / Prop 4.3 |
| Hopfield 容量 α_c = 0.138 | **定理（二手核实）** | C4（Amit et al. 1985） |
| 键空间 d 维 ⇒ 最多 d 个正交键，超出则检索误差 | **论证（非严格定理）** | C2 §4.1（转引 Smolensky 1990） |
| 加性写在 overcapacity 下次优，delta 规则更好 | **论证 + 实证** | C2 §4.2、C3 |
| 跳更新可保持性能 | **实证** | C10 |
| 衰减率 ↔ 容量 的严格界 | **缺口（未找到）** | — |
| 低频写入 ⇒ 误差累积的定理 | **缺口（未找到）** | — |
| 矩阵记忆 + 向量双状态 严格强于单状态 | **缺口（未找到，且 C5 反向暗示）** | — |

---

## 6. 对「六条直觉」中与本专题相关的三条的判定

> 我只拿到主题标签（长期记忆 / 表达力 / 更新频率），没有六条直觉的原文；以下按标签判定，若与原文口径不符请 lead 以原文为准。

**6.1 「矩阵快权重 = 长期记忆」→ 不成立（至少被严重高估）。**
- 容量上界 O(d)（线性读）或 O(d²/log)（双线性读），与「长期」无关，只与 d 和读法有关（C1/C2）。
- 候选只有衰减、没有巩固/重放/压缩机制；衰减本身就在**主动缩短**记忆（C8 的 γ、C9 的 Δ 衰减都是短时机制）。
- 若主张「长期」，需要给出跨时间尺度的多速率记忆（本专题未覆盖，属其他专题），单靠一个 d×d M 不够。

**6.2 「矩阵记忆 ⇒ 更强表达力」→ 不成立（有定理反对）。**
- 矩阵态 = d² 维向量，写入被约束为秩 1；线性注意力与它严格等价（C2）。
- L-uniform TC0 上界（C5）：不能做 S5 置换复合；PARITY 也困难（C6）。**加大 d 不改变复杂度类。**
- 唯一能升级表达力的已知最小改动是**输入相关的转移矩阵**（C5 原文）；候选的衰减转移是数据无关的。

**6.3 「不反复重写不变内容 + 按 Δ 控制」→ 部分成立，但当前设计有两个洞。**
- 成立的部分：少写可以不掉性能（C10 实证）；按真实间隔衰减有可靠先例（C9 的 GRU-D，2018 年）。
- 洞 1：**没有学出的跳门**。候选用连续 α 代替 skip，缺少稀疏/预算约束；「该不该写」没有被显式建模。
- 洞 2：**衰减与少写耦合**。衰减记忆里不写 = 慢慢丢失（§4.4）；候选把「写不写」和「衰减多少」都交给 Δ 与 α，两者会互相抵消或共振，缺少稳定性分析。
- 建议（若要用）：把写强度 α 与「是否需要更新」（surprise / 新旧 key 的残差）显式解耦，并给 skip 一个二值或稀疏通路（C10）；写规则从加性换成 delta（C2/C3）。

---

## 7. 链接清单（除注明者均经 tavily_extract 实际拉取）

1. C1 Barnfield, Kim, Nichani, Lee, Lu — https://arxiv.org/abs/2605.05189 （全文 https://arxiv.org/html/2605.05189v1）
2. C2 Schlag, Irie, Schmidhuber (FWP) — https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
3. C3 Yang, Wang, Zhang, Shen, Kim (DeltaNet) — https://arxiv.org/pdf/2406.06484
4. C4 Hopfield 1982 — https://www.pnas.org/doi/10.1073/pnas.79.8.2554 ；容量常数二手出处 https://web.mit.edu/8.592/www/grades/projects/Projects(2007)/KevinTakasaki.pdf
5. C5 Merrill, Petty, Sabharwal (Illusion of State) — https://arxiv.org/pdf/2404.08819v3 （abs: https://arxiv.org/abs/2404.08819；ICML: https://proceedings.mlr.press/v235/merrill24a.html）
6. C6 Sarrof, Veitsman, Hahn — https://arxiv.org/abs/2405.17394 （HTML: https://arxiv.org/html/2405.17394v3）
7. C7 Arora et al. (Zoology) — https://arxiv.org/html/2312.04927v1
8. C8 Sun et al. (RetNet) — https://arxiv.org/pdf/2307.08621
9. C9 Che et al. (GRU-D) — https://arxiv.org/pdf/1606.01865 ；期刊版 https://www.nature.com/articles/s41598-018-24271-9
10. C10 Campos et al. (Skip RNN) — https://arxiv.org/pdf/1708.06834
11. 参考（未单独成条）：Katharopoulos et al. 2020 *Transformers are RNNs* — https://arxiv.org/pdf/2006.16236 （其式 17–19 由 C2 转写并核到）



# 09 · Lead 数学分析：候选结构到底是哪几个已知结构的改写

> 本文件只做数学，不依赖检索。对象是用户这一轮的候选结构（双状态：矩阵记忆 M + 隐状态 h）：

    记忆衰减   M_i⁻ = e^{−Λ Δ_i} ⊙ M_{i−1}          Λ = diag(λ_1..λ_{d_k}), λ>0
    读写投影   q_i = Q(z_i,h_i), k_i = K(z_i,h_i), v_i = V(z_i,h_i)
    内容写入   M_i  = M_i⁻ + α_i v_i k_iᵀ
    读出       r_i  = R(M_i, q_i)
    状态更新   h_i  = H(h_{i−1}, r_i, z_i)

**先说结论**：这个结构等于 **"带指数遗忘的 Hebbian 快权重记忆"**，在数学上有三个等价身份（§1），
它的时间项恰好可以做出一件别人不容易做到的事（§2 的精确离散化），但它自身有一个**写入语义上的硬缺陷**（§3），
而用户想加的"只写变化部分"与"α 由 Δ 决定"在数学上是**两个互相打架的要求**（§4）。
真正还没被做掉的是 §5 列的三条命题。

---

## 1. 三个等价身份（Q3 的直接回答：是改写）

### 1.1 身份一：线性注意力的循环形式（= 指数衰减核的线性注意力）

线性注意力的因果形式是

    M_i = M_{i−1} + φ(k_i) v_iᵀ ,   r_i = M_iᵀ φ(q_i)

候选结构只是把"累加"换成"带遗忘的累加"：M_i = ρ_i M_{i−1} + α_i v_i k_iᵀ。
这正是 **RetNet / GLA / RWKV / Forgetting Transformer / Hedgehog** 这一族的循环形式，
区别仅在 ρ 的来源：RetNet 用固定常数 γ（每步），GLA 用输入依赖的门，Hedgehog 用逐特征学习的衰减，
**候选结构用 e^{−ΛΔ_i}（Δ 为真实秒）**。所以：
**"矩阵快权重 + 衰减 + 内容寻址写入"= 线性注意力的循环形式；这一层不是新的。**

### 1.2 身份二：Hebbian 记忆 + 遗忘（不是 delta rule）

把两次写入展开（先忽略 α 的内容依赖）：

    M_n = (∏_{i≤n} ρ_i) M_0 + Σ_i (∏_{j>i} ρ_j) α_i v_i k_iᵀ

这是 **加权 Hebb 外积和**（权重 = 之后所有衰减的乘积），即"每写一次就把该关联加到记忆里"。
它对读出 key k 的响应是

    r(k) = M_n k = Σ_i w_i (k_iᵀ k) v_i

**这是核平滑/加权平均，不是覆写**。DeltaNet 的写法是

    M_i = M_{i−1}(I − β_i k_i k_iᵀ) + β_i v_i k_iᵀ   （先把旧关联减掉，再写）

两者的差别是**代数性的**：Hebbian 只能"叠加"，delta 能"替换"。当同一个 key 上的 v 变化时，
Hebbian 记忆给出的是滞后约 1/λ 的滑动平均；要跟得快就得调大 λ，但那会同时忘掉别的 key。
**所以"变化的内容通过 key 找到相关记忆并更新"这句话，只有用 delta 型写入才成立；用 v kᵀ 型写入不成立。**

### 1.3 身份三：一阶线性 ODE 的零阶保持（ZOH）离散化

这层是候选结构**唯一有数学红利**的地方。考虑连续时间矩阵记忆

    dM/dt = −Λ M + U(t) ,   U(t) = v(t) k(t)ᵀ

在区间 [t_{i−1}, t_i] 上把 U 当常数 u_i = v_i k_iᵀ，精确解为

    M_i = e^{−ΛΔ_i} M_{i−1} + (I − e^{−ΛΔ_i}) Λ^{−1} u_i

**逐项对照候选结构**：第一项完全一致；第二项要求

    α_i = (I − e^{−ΛΔ_i}) Λ^{−1}     （对角矩阵，作用是给 v kᵀ 的每一行乘一个系数）

若 Λ = λI 且把输入归一化为 λ v kᵀ（即 ODE 写成 dM/dt = −λ(M − v kᵀ)），则

    α_i = 1 − e^{−λΔ_i}

**与用户写的 α = 1 − e^{−λΔt} 逐字相同。**

也就是说：**候选结构不是"随便挑了一个衰减"，它就是那个线性 ODE 的精确解**。
这一步与 Mamba 的离散化（Ā = exp(ΔA)、B̄ = (ΔA)^{−1}(exp(ΔA)−I)ΔB）是同一套数学，
区别只在 Mamba 的状态是向量、候选是矩阵，且 Mamba 的 Δ 是**学习出来的**而不是秒。
（"候选 = Mamba 的矩阵版 + 秒级 Δ"这个判定要靠 04 节和 07 节的检索确认，本文件只负责数学。）

---

## 2. 由此直接得到的四个可证命题（这是候选结构真正值钱的部分）

### 命题 1（精确性）
若在每段区间内 (v_i, k_i) 保持不变，且 α_i = (I − e^{−ΛΔ_i})Λ^{−1}，则候选更新给出的 M_i
**恰好等于**上述线性 ODE 在 t_i 的解（零误差，不是近似）。
*证明*：积分因子法，逐区间展开即可。

### 命题 2（重复帧不变性 —— "不变内容不必重写"在数学上成立）
设同一对 (v,k) 在总时长 T 内被观测 n 次（Δ_i = T/n）。精确解在整段上是
M(T) = e^{−ΛT}M_0 + (I − e^{−ΛT})Λ^{−1} vkᵀ，**与 n 无关**。
*证明*：几何级数 (I − e^{−ΛT/n})Σ_{j<n} e^{−ΛTj/n} = I − e^{−ΛT}，与 n 无关。
**推论**：只要 α 按 Δ 正确缩放，重复观测同一内容不会改变记忆——所以"不变内容不反复重写"是
**可证的设计性质**，不是直觉。**但注意**：这只在 (v,k) **逐位相同**时成立；
真实视频里同一物体的 key 每次都有噪声，此时重复写入造成随机游走漂移，漂移量 ~ α√n·σ。
这恰好构成"必须避免无谓重写"的**定量理由**：不是省算力，而是控制漂移方差。

### 命题 3（一致有界 / ISS 稳定性）
对任意 Δ_i、任意写入次数，若 sup‖v_i k_iᵀ‖ ≤ c，则

    ‖M_i‖ ≤ max( ‖M_0‖ , (1/λ)c )

即记忆范数不会随写入次数增长。*证明*：取范数后用 e^{−λΔ}+(1−e^{−λΔ})=1 归并。
**反例对照**：去掉衰减（ρ=1）时 ‖M_i‖ ≤ ‖M_0‖ + i·α·c，**线性增长** —— 这是"矩阵记忆会被写爆"的定量版本。

### 命题 4（细化收敛，一阶）
一般情形 (v_i,k_i) 逐帧变化时，ZOH 是**一阶**离散化：
max_i ‖M_i − M(t_i)‖ = O(Δ_max)。所以"采样加密 → 结果收敛到同一条连续轨迹"**成立**，
但只有一阶精度；要二阶得换成梯形/RK2 型的 ZOH。

---

## 3. 反例：α 与 Δ 无关时，细化极限不存在（Q2 的关键区分）

设 α_i = σ(g(z_i))（纯学习门，与 Δ 无关），Λ = λI，输入 u 有界且 ‖u‖ ≥ c > 0。
衰减递归的稳态幅度为

    ‖M_∞‖ ≈ αc / (1 − e^{−λΔ}) → αc/(λΔ) → ∞   (Δ → 0)

**结论**：Λ 固定的前提下，若 α 不随 Δ 缩小，**采样越密记忆越大、越不稳定**；
反过来，α = 1 − e^{−λΔ} 恰好是"让稳态与采样率无关"的唯一一阶选择。

**这就是"α = σ(g(z))"与"α = 1 − e^{−λΔt}"的本质区别**（用户 Q2 要求区分的那个点）：
- 前者：α 是**内容门**，时间只出现在 ρ 里；时间语义只作用于**遗忘**，不作用于**写入**。
- 后者：时间同时决定遗忘与写入，**整个更新是同一个 ODE 的精确解**，才有命题 1–4。
- 两者不是"写法不同"，而是**一个有连续时间极限、一个没有**。

**推论（对用户设计的直接建议）**：正确形式不是二选一，而是**乘起来**：

    α_i = (1 − e^{−ΛΔ_i}) · s(z_i) ,   s ∈ [0,1] 为内容门

此时 ODE 的输入变成 s(z)v kᵀ（分段常数），命题 1、2、3 全部保留，同时获得内容自适应性。
这一形式在检索范围内是否已有人写出，由 04 节和 07 节回答。

---

## 4. 用户两个要求的冲突（必须先解决才能谈贡献）

用户同时想要：
(i) **α_i 内容相关**（"变化的内容才写"）；
(ii) **α_i 由真实 Δt 控制**（"衰减和写入由时间控制"）。

从 §3 可知：**只有 (1−e^{−ΛΔ}) 这个因子是"时间项"**，内容相关性必须放在另一个因子上。
如果把内容门塞进 α 的指数里（例如 α = 1 − e^{−λ(z)Δ}），Λ 就变成内容相关的：
这仍然可以是精确 ZOH（只要 z 在区间内不变），但会带来两个新问题：
1. **Λ(z) 的内容依赖会让"重复帧不变性"（命题 2）失效**——同一物体在光照变化下 λ(z) 变了，记忆就以不同速率衰减；
2. **读出与写入的 Λ 必须一致**，否则"读到的记忆"与"写入时假设的遗忘"不匹配。
这两点目前没有任何定理保证，是**真正的开放问题**（见 §5）。

---

## 5. 仍然可能形成数学贡献的三条（也是判"未覆盖"的唯一依据）

### 贡献 1：精确 ZOH 的快权重更新（Heaviside 型）
把候选的 α 换成 (I − e^{−ΛΔ})Λ^{−1}、并把 delta 型的乘性项也做精确化：
对 Λ=0 的 delta 规则，连续形式 dM/dt = −β(Mk − v)kᵀ 的**精确**解是

    M_i = (I − (1−e^{−β_i}) k_i k_iᵀ) M_{i−1} + (1−e^{−β_i}) v_i k_iᵀ

而 **DeltaNet / Gated DeltaNet 用的是 (I − β k kᵀ)**，只是上式的一阶近似（β ≪ 1 时二者一致）。
**这是一个可以写出来的、可核对的差异**："现有 delta 规则的门是离散一阶近似，精确形式应把 β 换成 1−e^{−β}"。
它是否已被发表，由 06 节红队回答；就数学本身，这是一个干净的 1 页命题。

### 贡献 2：不规则采样下的记忆竞争定理
多个对象同时变化时，写入是 rank-1 外积的叠加，干扰由 key 的 Gram 矩阵条件数控制：
若 ‖k_iᵀk_j‖ ≤ ε (i≠j)，则读出误差 ≤ ε·Σ_j w_j‖v_j‖。
把 w_j = e^{−λ(age_j)} 代进去，得到**"记忆带宽 × 时间窗"的联合界**：
可分辨的对象数 ≈ 1/(ε · λ·平均间隔)。这条把"容量"和"衰减率"绑在一起，是现有线性注意力理论没有强调的组合。

### 贡献 3：采样细化一致性定理（矩阵版）
命题 1–4 合起来就是"矩阵记忆在细化下收敛且重复观测不变"。
**但它是教科书级结论的应用**（ZOH 精确解 + 几何级数），单独发表偏弱；
必须加上"多对象、变 key、内容门"的情形才有分量——也就是把命题 2 推广到 **key 随时间连续变化**的情形：
此时 e^{−ΛΔ} 与 k(t) 的变化率耦合，会出现 O(Δ) 的系统偏差；这个偏差的**符号与上界**是新的。

---

## 6. 对"六条直觉"的数学裁决（检索证据见 06/08 节，这里只给数学部分）

| 直觉 | 数学上的裁决 |
|---|---|
| 快权重一定能记住长期目标 | **错**。Hebbian+衰减是 EMA，有效记忆窗 ~1/λ；超出窗的旧关联被压到 e^{−λ·age}。且 rank ≤ d_k，超容量后线性组合必然混叠。 |
| 加 Δt 就自动有连续时间语义 | **错，但可修**。只有当**所有**时间相关项（ρ 与 α）都由 Δ 导出时才等价于 ODE；只把 Δ 放进 ρ 而 α 固定，细化极限 $|M| 	o infty$（命题 §3）。 |
| attention 写入天然只改变化部分 | **错**。v kᵀ 是稠密 rank-1 写，改的是 M 的**所有行**（按 v 的行权重）。真"只改局部"需要硬掩码或 delta 型投影。 |
| 稀疏更新天然提高效率 | **不一定**。稀疏化的收益取决于实现：门控本身要算，不规则访存不友好；理论 FLOPs ≠ 墙钟。 |
| 矩阵记忆比普通隐状态更有表达力 | **部分错**。d_k×d_v 矩阵只是 d_k·d_v 维状态；表达力上界由 rank 与 key 的 Gram 条件数决定，不是"矩阵"这个形状给的。 |
| 记忆更新越少越稳定 | **错**。稳定性来自衰减（命题 3），不是来自更新频率；跳更新反而让每个 Δ 变大、每次修正变大。 |

---

## 7. 待检索确认的问题（交给 01–08 节）

- [ ] 有没有论文在**矩阵记忆**（而非向量/标量状态）上写出 ρ = e^{−ΛΔ}、α = 1−e^{−ΛΔ} 这对系数？
- [ ] 有没有论文把 delta 规则的 β 换成 1−e^{−β}（精确指数化）？
- [ ] 有没有论文用**真实秒级 Δt** 控制**线性注意力/快权重**的衰减（不是 Mamba 那种学习 Δ）？
- [ ] 有没有论文把"写入强度 = 时间项 × 内容门"这个乘积形式写出来？
- [ ] 有没有视频流工作显式做"变化才写"的硬路由，并给出不变性/稳定性结论？


---

## 8. Lead 自查命中的两条关键证据（补充，需与 01–08 节交叉核对）

### 8.1 「精确指数积分器」已被试过，而且是**负结果**

> *Exponential Integrator for Diagonal-Decay Delta Attention: A Negative Result on Length Extrapolation*，
> FARS / Analemma，2026，https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_59af4b6d-78d3-4f27-90bf-2d9c319a0093/main.pdf
> **注意：该文自述 "This paper was generated by an automated research system"，非同行评审，可信度需打折。**

它做的正是本文件 §5「贡献 1」：把对角衰减 delta 注意力（KDA）的**欧拉系数换成从连续时间动力学导出的精确指数积分器**。
结论：**数值更稳（无 L2 归一化也不发散，27 次运行 0 NaN），但长度外推精度没有变好，甚至更差**
（Palindrome −0.81pp、Stack −2.98pp @4× 外推）；作者结论是 **"离散化误差不是长度外推的主要瓶颈"**。

**对本项目的含义（负面）**：
- "把 α 换成 1−e^{−λΔ} 以获得精确连续时间"这件事，2026 年已经有人试过并报告**无收益**；
- 即使我们换成矩阵记忆 + 秒级 Δ，**"更精确 ⇒ 更好"这条论证链本身已被证伪一次**；
- 因此"精确离散化"只能作为**稳定性/可解释性**卖点，不能作为精度卖点。若要主张，必须换评测（跨采样率泛化，而不是长度外推）。

### 8.2 那对系数已经出现在门控 delta 网络的公式里（只是 Δt 不是秒）

- **Gated DeltaNet / Qwen3-Next 的 SSM 式参数化**：α_t = e^{−A Δt}，且 Δt 是**逐 token 学习量**
  （见 https://github.com/rasbt/LLMs-from-scratch/discussions/902 对 Qwen3-Next 实现的计算说明；
  亦见 Gated DeltaNet 原文 *Gated Delta Networks: Improving Mamba2 with the Delta Rule*, ICLR 2025, arXiv:2412.06464）。
- 也就是说：**ρ = e^{−ΛΔt} 这一半已经是 2024–2025 年线性注意力/SSM 的标准写法**；
  候选结构唯一多出来的是"Δt = 墙钟秒"这一**语义**，以及把它与**矩阵记忆的写入强度**耦合。

**判决**：公式层面不新；语义层面（真实时间）可能仍空，但**它的价值必须靠"跨采样率/跨帧率泛化"的实验或一个定理来兑现**，
不能靠"我们把 Δt 换成了真实时间"这句话本身。
