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
