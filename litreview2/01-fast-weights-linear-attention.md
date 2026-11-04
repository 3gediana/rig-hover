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
