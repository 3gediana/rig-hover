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

