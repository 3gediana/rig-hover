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
