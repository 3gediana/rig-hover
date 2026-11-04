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
