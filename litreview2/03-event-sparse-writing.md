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
