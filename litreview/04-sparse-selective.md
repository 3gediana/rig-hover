# 04 · 稀疏状态改写 / 选择性更新 / 事件驱动 RNN / 对象中心槽位视频模型
### —— 与 h⁻ = Φ_Δ(h)、r = z − D(h⁻)、h = h⁻ + B(h⁻, r) 的逐条对照

> **检索与核实方式**：主用 tavily_search（advanced）+ tavily_extract 抓 arXiv abs/HTML/PDF 与会议/期刊官方 PDF；每篇链接都实际拉取过正文。公式优先取正文原式，只有摘要可用的条目明确标 【仅摘要级】。个别显示公式在 PDF→文本抽取中被丢弃时，用正文文字陈述复述并在条目里注明。
> **本文件只覆盖**：稀疏/选择性状态改写、事件触发估计、事件驱动 RNN、对象中心/槽位视频模型、"delta networks"。
> **七轴**：A 显式先验 h⁻；B 残差在观测空间 r = z − D(h⁻)；C 加性修正 h = h⁻ + B；D B 同时依赖 (h⁻, r)；E 真实时间间隔 Δ（秒）；F 稀疏/选择性/局部化；G 训练目标。
> **Q1/Q2/Q3**：Q1 每次观测是否用预测误差修正隐状态；Q2 是否按真实时间 Δ 演化、Δ 从哪进入；Q3 用途是给规划/生成未来（model-based RL）还是通用在线状态估计。

---

## 1. 控制文献：事件触发估计 —— 本专题最致命的一类先例

### 1. Event-Based State Estimation with Switching Static-Gain Observers
- **作者 / 年份 / 会议或期刊 / 链接**：Sebastian Trimpe, Raffaello D'Andrea / 2012 / 3rd IFAC Workshop on Distributed Estimation and Control in Networked Systems (NecSys'12), Santa Barbara, pp. 91–96 / http://trimpe.is.tuebingen.mpg.de/publications/trimpe-ECNS12.pdf
- **核心更新方程（原文式 (8)–(11)，公式已从 PDF 正文核对）**：
  $$
  \check{x}(k\mid k-1)=A\,\check{x}(k-1\mid k-1)+B\,u(k-1)\tag{10}
  $$
  $$
  \text{send } y_i(k)\iff \lVert y_i(k)-C_i\check{x}(k\mid k-1)\rVert\ge \delta_i\tag{8}
  $$
  $$
  \check{x}(k\mid k)=\check{x}(k\mid k-1)+\sum_{i\in I(k)}L_i\bigl(y_i(k)-C_i\check{x}(k\mid k-1)\bigr)\tag{11}
  $$
  $$
  I(k)=\{\,i\mid \lVert y_i(k)-C_i\check{x}(k\mid k-1)\rVert\ge\delta_i\,\}\tag{9}
  $$
  若无传感器触发（I(k) 为空集），则 $\check{x}(k\mid k):=\check{x}(k\mid k-1)$（**状态被原样拷贝，硬不更新**）。
  稳定性：定理 1 给出误差上界 $\lVert e\rVert_\infty\le \frac{m}{1-\rho}\lVert L\rVert\lVert\delta\rVert$，其中 e 是"全通信估计器 − 事件触发估计器"之差。
- **与目标公式异同（A–G）**：
  - **A 有**：式 (10) 就是 $\Phi_\Delta(h)$ 的线性形式（先验/预测）。
  - **B 有，且逐字对应**：$\Delta_i(k)=y_i(k)-C_i\check{x}(k\mid k-1)$ 就是 $r=z-D(h^-)$，D 是测量模型 $C_i$。
  - **C 有**：式 (11) 就是 $h=h^-+B(h^-,r)$，B 是静态增益块 $L_i$（本文刻意不用时变卡尔曼增益）。
  - **D 有**：残差本身依赖 $h^-$，增益 $L_i$ 按传感器分块。
  - **E 无**：k 是离散步；时间只通过固定的 A 隐含（A 里可塞采样周期，但公式没有显式 Δ，也不允许变步长）。
  - **F 有（硬）**：式 (8) 是**按预测误差范数决定"改不改"**的硬触发；不触发时状态完全不动，这是硬稀疏而非软门控。
  - **G 无训练**：控制/估计目标，增益离线设计；没有任何学习目标。
- **任务与训练目标**：用少量测量达到接近全通信估计器的性能，并给出误差上界；无训练。
- **最接近的先例（该文自引/同期）**：Trimpe & D'Andrea 2011（卡尔曼滤波版测量触发）；方差触发（variance-based triggering，Trimpe & D'Andrea 2011b，本文明确对比）；Miskowicz 的 send-on-delta；Lunze & Lehmann 的事件触发控制。
- **Q1/Q2/Q3**：Q1 是——每次事件都用新息修正；Q2 否——按步索引，Δ 不出现；Q3 通用在线状态估计。
- **新颖性判断**：**直接削弱**。目标公式的 A/B/C/D + "用预测误差大小决定改不改"的硬稀疏更新，在这里 2012 年就已经完整出现，而且带稳定性上界。目标公式相对它多出来的只有"Δ（秒）显式进入"和"B 可学习"。

### 2. On Event Based State Estimation
- **作者 / 年份 / 会议或期刊 / 链接**：Joris Sijs, Mircea Lazar / 2009 / HSCC 2009, LNCS 5469, pp. 336–350 / https://pure.tue.nl/ws/files/3301085/Metis230979.pdf
- **核心更新方程（原式 (6)(7a)(7b)，已从 PDF 正文核对）**：
  $$
  z_{k_e}=(y_{k_e}^\top,t_{k_e})^\top \ \text{ sampled iff }\ y(t)\notin H_{k_e}(z_{k_e-1},t)\tag{6}
  $$
  $$
  x(t+\tau)=A_\tau x(t)+B_\tau w(t),\qquad y(t)=Cx(t)+v(t)\tag{7a,7b}
  $$
  Send-on-delta 特例（正文明确给出）：事件时刻 $k_e$ 在 $|y(t)-y_{k_e-1}|$ 超过阈值 Δ 时发生，对应
  $H_{k_e}=\{y\in\mathbb{R}\mid -\Delta<y-y_{k_e-1}<\Delta\}$。
  估计器是混合体：既在事件时刻 $t_{k_e}$、也在同步采样时刻 $t_{k_a}$ 更新，并对所有 n 保证 $\lambda_{\max}(P_{n|n})$ 有界（定理 1，证明中给出式 (31) 的协方差递推）。
- **与目标公式异同（A–G）**：
  - **A 有（隐含）**：事件之间做预测传播（$A_\tau$），事件时刻做贝叶斯更新；预测-更新两段结构清晰。
  - **B 部分**：更新用收到的新观测做条件化；等价卡尔曼形式下就是观测空间残差，但**该文正文未把残差写成显式一项**（抽取文本中残差符号未出现，故 B 判为"等价成立、非显式"）。
  - **C 部分**：贝叶斯条件化在卡尔曼情形等价于加性修正；同样非显式。
  - **D 部分**：同 B/C。
  - **E 有，而且比目标公式更彻底**：**真实时间 τ 直接进状态转移** $A_\tau,B_\tau$，并且把"事件时刻"与"同步采样时刻"混在一个时间轴 $\{t_n\}=\{t_{k_a}\}\cup\{t_{k_e}\}$ 上。这正是"$\Phi_\Delta$ + 事件触发"的组合。
  - **F 有（硬）**：只有 $y(t)\notin H$ 才产生事件；阈值 Δ 越大越稀疏。
  - **G 无训练**：滤波器设计，无学习目标。
- **任务与训练目标**：构造适用于任意事件采样律的事件基状态估计器，并证明协方差有界。无训练。
- **最接近的先例（该文自引/同期）**：Miskowicz send-on-delta；Åström & Bernhardsson 的 Lebesgue 采样；Tabuada / Wang & Lemmon 事件触发控制。
- **Q1/Q2/Q3**：Q1 是（事件发生才修正）；Q2 是，Δ 是"事件间隔"的物理时间，且 τ 进入 $A_\tau$；Q3 通用在线估计。
- **新颖性判断**：**直接削弱**。"真实时间间隔进入状态转移"+"用观测是否越界决定改不改"在 2009 年已有，且带协方差有界性定理。

### 3. Send-On-Delta Concept: An Event-Based Data Reporting Strategy
- **作者 / 年份 / 会议或期刊 / 链接**：Marek Miskowicz / 2006 / Sensors 6(1):49–63, doi:10.3390/s6010049 / https://www.mdpi.com/1424-8220/6/1/49
- **核心更新方程（原文第 3 节，文字定义已核对）**：当被测量**偏离最近一次上报值**达到置信区间 ±Δ 时上报；原文式 (1) 的显示公式在 PDF 抽取中被丢弃，但正文明确其为"相邻两次上报的信号值之差恒等于 Δ"，即
  $|x(t_i)-x(t_{i-1})|=\Delta$（触发时取阈值），Δ>0。
  注：比较对象是"上一次上报值"，**不是模型预测**。
- **与目标公式异同（A–G）**：
  - **A 无**：没有预测器，因此没有 $h^-$。
  - **B 否**：残差定义在"当前测量 − 上次上报"，不是 $z-D(h^-)$。
  - **C/D 无**：没有修正函数，只有"发/不发"。
  - **E 部分**：定义是连续时间的，但公式里只有幅值阈值 Δ，没有时间量。
  - **F 有（硬）**：经典电平交叉 / Lebesgue 采样，硬阈值。
  - **G 无**。
- **任务与训练目标**：在给定分辨率下降低上报率、给出平均上报率的界与"有效性"下界；无训练。
- **最接近的先例（该文自引）**：1950 年代末 Ellis 的"只传显著数据"；Åström & Bernhardsson 的 Lebesgue 采样（Riemann vs Lebesgue 采样对照）。
- **Q1/Q2/Q3**：Q1 否（无预测）；Q2 否；Q3 数据上报策略，不是状态估计。
- **新颖性判断**：削弱"按变化大小决定是否更新"这一**思想**的原创性，但单看它比目标公式弱（无 $h^-$、无 D、无 B）。它的价值在于说明该思想属于控制/传感领域 60 年的常识。

---
## 2. 选择性更新的 RNN

### 4. Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Victor Campos, Brendan Jou, Xavier Giró-i-Nieto, Jordi Torres, Shih-Fu Chang / 2018 / ICLR 2018 / https://arxiv.org/pdf/1708.06834（公式取自 PDF 正文式 (1)–(6)）
- **核心更新方程（原文式 (1)–(6)）**：
  $$
  s_t=S(s_{t-1},x_t)\tag{1};\qquad u_t=f_{\text{binarize}}(\tilde u_t)\tag{2}
  $$
  $$
  s_t=u_t\cdot S(s_{t-1},x_t)+(1-u_t)\cdot s_{t-1}\tag{3}
  $$
  $$
  \Delta\tilde u_t=\sigma(W_p s_t+b_p)\tag{4};\qquad
  \tilde u_{t+1}=u_t\cdot\Delta\tilde u_t+(1-u_t)\cdot\bigl(\tilde u_t+\min(\Delta\tilde u_t,1-\tilde u_t)\bigr)\tag{5}
  $$
  $$
  N_{\text{skip}}(t)=\min\{n:n\cdot\Delta\tilde u_t\ge 0.5\}-1\tag{6}
  $$
  其中 $u_t=\mathrm{round}(\tilde u_t)$（确定性阶跃，或 Bernoulli 采样）。
- **与目标公式异同（A–G）**：
  - **A 无**：没有"预测"与"观测"的区分，$s_t$ 是 RNN 状态本身，不存在 $h^-$。
  - **B 否**：没有观测空间、没有解码映射 D、没有残差。
  - **C 部分是"替换"不是"加性残差"**：$s_t=u_t S+(1-u_t)s_{t-1}$ 是二值门控选择，不是 $h^-+B$。
  - **D 否**：门控 $\Delta\tilde u_t$ 只依赖 $s_t$（即输入派生的隐状态），不看任何"误差"。
  - **E 否**：按时间步索引；论文的核心卖点是"跳过若干个 step"，不涉及秒。
  - **F 有（最硬的那种）**：$u_t\in\{0,1\}$，$u_t=0$ 时状态被逐位复制、**保证完全不动**，甚至可以不计算（式 (6) 允许提前算出跳过步数）。这正是"无关部分完全不动"的既有实现。
  - **G**：任务损失 + 更新次数预算惩罚（代码里为 COST_PER_SAMPLE；论文正文为 budget constraint），训练用 straight-through。
- **任务与训练目标**：减少 RNN 顺序操作数（加速/省内存/缩短计算图），在加和、频率判别、MNIST、情感分析、动作识别上验证；**不是**状态估计。
- **最接近的先例（该文自引）**：LSTM-Jump（REINFORCE，需人工奖励）、Phased LSTM、conditional computation（Bengio 2013）。
- **Q1/Q2/Q3**：Q1 **否——跳过与否由输入驱动的累积概率决定，不是由预测误差决定**；Q2 否；Q3 都不是（是计算效率）。
- **新颖性判断**：**削弱"稀疏/选择性状态更新"这个说法的新颖性**（硬门控 + 完全不动 + 预算约束都已做过），但**不削弱残差式**——它的选择信号与预测误差无关，这一点必须写清楚，否则容易被误当成先例。

### 5. Adaptive Computation Time for Recurrent Neural Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Alex Graves / 2016 / arXiv:1603.08983 (Google DeepMind) / https://arxiv.org/pdf/1603.08983（公式取自 PDF 正文式 (3)–(13)）
- **核心更新方程（原式 (5)–(13)）**：
  $$
  h_t^n=\sigma(W_h s_t^n+b_h)\tag{5};\qquad
  N(t)=\min\Bigl\{n':\sum_{n=1}^{n'}h_t^n\ge 1-\epsilon\Bigr\}\tag{7}
  $$
  $$
  s_t=\sum_{n=1}^{N(t)}p_t^n s_t^n,\qquad y_t=\sum_{n=1}^{N(t)}p_t^n y_t^n\tag{9}
  $$
  $$
  \rho_t=N(t)+R(t)\tag{10};\qquad P(x)=\sum_{t=1}^T\rho_t\tag{11};\qquad
  \hat{\mathcal L}(x,y)=\mathcal L(x,y)+\tau P(x)\tag{12}
  $$
- **与目标公式异同（A–G）**：
  - **A 无**：没有观测模型，也没有"先验状态"与"后验状态"的区分；重复的是同一个状态转移 S。
  - **B 否**：没有任何 $z-D(h)$。
  - **C 否**：输出是对中间状态的**加权平均**（mean-field），不是加性残差修正。
  - **D 否**。
  - **E 否**："多算几步"的 n 是计算步数，输入里只加一个 $\delta_{n,1}$ 标志，与真实时间无关。
  - **F 有（软选择性）**：自适应计算深度；但门控是连续的 $h_t^n\in(0,1)$，靠 $\sum h\ge 1-\epsilon$ 停止。
  - **G**：任务损失 + ponder cost（对计算量加罚）。
- **任务与训练目标**：让网络自己决定每个输入"想多久"；四个合成任务 + 字符预测。
- **最接近的先例（该文自引）**：Self-Delimiting Neural Networks（阈值 halting）、conditional computation、神经栈/队列的可微 push-pop。
- **Q1/Q2/Q3**：Q1 否；Q2 否；Q3 都不是（计算分配）。
- **新颖性判断**：**削弱"选择性更新"的宽泛表述**，但与残差式无关。它的价值是提供"软门控会被优化器钻空子"的直接证据（见 §6.2）。

---

## 3. 对象中心 / 槽位视频模型

### 6. Object-Centric Learning with Slot Attention
- **作者 / 年份 / 会议或期刊 / 链接**：Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf / 2020 / NeurIPS 2020 / https://arxiv.org/pdf/2006.15055（Algorithm 1 + 式 (1)(2) 已核对）
- **核心更新方程（Algorithm 1 与式 (1)(2)）**：
  $$
  M=\frac{1}{\sqrt D}k(\text{inputs})\cdot q(\text{slots})^\top\in\mathbb{R}^{N\times K},\qquad
  \mathrm{attn}_{i,j}=\frac{e^{M_{i,j}}}{\sum_{l}e^{M_{i,l}}}\ \ (\text{softmax over slots})\tag{1}
  $$
  $$
  \text{updates}=W^\top v(\text{inputs})\in\mathbb{R}^{K\times D},\quad W_{i,j}=\frac{\mathrm{attn}_{i,j}}{\sum_{l=1}^{N}\mathrm{attn}_{l,j}}\tag{2}
  $$
  伪代码：slots = GRU(state=slots_prev, inputs=updates)，再 slots += MLP(LayerNorm(slots))，重复 T=3 次。
- **与目标公式异同（A–G）**：
  - **A 无（静态图像）**：slots 从正态分布 $\mathcal N(\mu,\mathrm{diag}\,\sigma)$ 随机初始化，没有"预测先验"。
  - **B 否**：没有任何"观测预测残差"；注意力权重是 slot 与输入特征的相似度，不是 $z-D(h^-)$。
  - **C 部分**：GRU(slots_prev, updates) 在形式上确实是"旧状态 + 新信息"的加性/门控组合，但新信息不是残差。
  - **D 部分**：GRU 同时吃 slots_prev 与 updates，结构上像 $B(h^-,r)$，但 r 不是误差。
  - **E 无**：T 是迭代次数，与时间无关。
  - **F 软**：稀疏性来自 **softmax 在 slot 维归一化造成的竞争**（每个输入特征被"分配给"某个 slot），不是"只更新部分 slot"。所有 K 个 slot 每轮都被 GRU 更新；且实现上加小偏置 ε 防止数值问题，意味着**没有严格零更新**。
  - **G**：重构（+ 有监督属性预测）。
- **任务与训练目标**：无监督物体发现 / 有监督集合预测；重构损失。
- **最接近的先例（该文自引）**：胶囊网络（Sabour 2017）、object files（Kahneman/Treisman 概念）、SPACE/SCAN/AIR。
- **Q1/Q2/Q3**：Q1 否（训练时用重构误差，推理时没有逐观测的预测误差）；Q2 否；Q3 通用物体中心表示。
- **新颖性判断**：不直接冲击残差式；但**削弱"对象级状态 + 稀疏更新"的新颖性**：这里的"稀疏"是输入-槽位分配竞争，不是观测残差驱动的选择。

### 7. Conditional Object-Centric Learning from Video (SAVi)
- **作者 / 年份 / 会议或期刊 / 链接**：Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold 等 / 2022 / ICLR 2022 / https://arxiv.org/pdf/2111.12594（式 (1)–(4) 已核对）
- **核心更新方程（原式 (1)–(4)）**：
  $$
  U_t=\frac{1}{Z_t}\sum_{n=1}^{N}A_{t,n}\odot v(h_{t,n})\in\mathbb{R}^{K\times D},\qquad
  A_t=\mathrm{softmax}_K\Bigl(\frac{1}{\sqrt D}k(h_t)\cdot q(\mathcal S_t)^\top\Bigr)\tag{1}
  $$
  $$
  \hat s_t^k=\mathrm{GRU}(u_t^k,s_t^k);\qquad
  \mathcal S_{t+1}=\mathrm{LN}\bigl(\mathrm{MLP}(\tilde{\mathcal S}_t)+\tilde{\mathcal S}_t\bigr),\ \ \tilde{\mathcal S}_t=\mathrm{LN}\bigl(\mathrm{MHSA}(\hat{\mathcal S}_t)+\hat{\mathcal S}_t\bigr)\tag{2}
  $$
  训练损失：光流/重构的像素平方误差 $\mathcal L_{\text{rec}}=\sum_t\lVert y_t-y_t^{\text{true}}\rVert^2$（式 (4)）。
- **与目标公式异同（A–G）**：
  - **A 有，且作者自己就这么定义**：原文："Inspired by predictor-corrector methods for integration of ODEs, SAVi performs two steps for each observed video frame: a prediction and a correction step." 式 (2) 的 predictor 就是 $\Phi_\Delta$ 的槽位版本（Δ = 一帧）。
  - **B 否，这是关键区别**：corrector **不是**在预测残差上工作，而是拿当前帧特征 $h_t$ 从头做一次 slot attention 重绑定；预测 $\mathcal S_t$ 只作为 query/初始化。即使画面与预测完全一致，corrector 依然会写一遍。
  - **C 部分**：$\hat s=\mathrm{GRU}(u,s)$ 是门控加性，但增量 u 不是 $z-D(h^-)$。
  - **D 部分**：GRU 同时吃 (上一状态, 注意力聚合)，但第二项不是残差。
  - **E 否**：t 是帧索引；帧率变化不在模型里。
  - **F 软**：每帧更新全部 K 个 slot，没有"只改一个对象"的机制。
  - **G**：一步预测（光流）或重构，平方误差，自监督。
- **任务与训练目标**：视频中多物体分割与跟踪；自监督（光流）训练。
- **最接近的先例（该文自引）**：Slot Attention（Locatello 2020）、ODE predictor-corrector（数值积分）、SCAN/SPACE。
- **Q1/Q2/Q3**：Q1 **是"每观测都修正"，但不是"用预测误差修正"**；Q2 否；Q3 视频表示/跟踪，不做规划。
- **新颖性判断**：**削弱 A 轴**：predictor-corrector 的显式两段式在槽位模型里 2022 年已有先例且命名明确。B/C/D/E 仍与目标公式不同。

### 8. Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos (STEVE)
- **作者 / 年份 / 会议或期刊 / 链接**：Gautam Singh, Yi-Fu Wu, Sungjin Ahn / 2022 / NeurIPS 2022 / https://arxiv.org/pdf/2205.14065（递推式与损失已核对）
- **核心更新方程（正文 Section 3）**：
  $$
  A_t=\mathrm{softmax}_N\Bigl(\frac{q(s_{t-1})\cdot k(e_t)^\top}{\sqrt D}\Bigr),\qquad
  r_{t,n}=\frac{\sum_{i=1}^{HW}A_{t,n,i}\,v(e_{t,i})}{\sum_{j=1}^{HW}A_{t,n,j}}
  $$
  $$
  \tilde s_{t,n}=f^{\mathrm{RNN}}_\phi(r_{t,n},s_{t-1,n}),\qquad s_t=f^{\mathrm{interact}}_\phi(\tilde s_t)
  $$
  $$
  \mathcal L_{\text{STEVE}}=\mathcal L_{\text{CE}}+\mathcal L_{\text{dVAE}},\quad
  z_{t,1:L}=f^{\text{dVAE}}(x_t),\ o_{t,l}=g^{\text{slot-transformer}}(s_t;z_{t,1:l-1})
  $$
- **与目标公式异同（A–G）**：
  - **A 部分**：$s_{t-1}$ 充当先验，但没有显式写成"预测的观测"。
  - **B 否**：$r_{t,n}$ 是注意力加权特征，不是 $z-D(h^-)$。
  - **C 部分**：RNN 门控加性。
  - **D 部分**：$f^{\mathrm{RNN}}(r,s_{t-1})$ 同时吃二者。
  - **E 否**：帧索引。
  - **F 软**：全部 slot 每帧更新；注意力权重归一化（分母 $\sum_j A$）保证不会全零。
  - **G**：dVAE + 自回归 transformer 解码的 token 交叉熵（等价于重构/自监督），另有 dVAE 的图像重构损失。
- **任务与训练目标**：复杂/自然视频的无监督物体中心表示（MOVi 等）；纯重构目标。
- **最接近的先例（该文自引）**：SLATE（Singh 2022）、SAVi（Kipf 2022）、Slot Attention。
- **Q1/Q2/Q3**：Q1 是"每帧修正"但非误差驱动；Q2 否；Q3 表示学习，不做规划。
- **新颖性判断**：与目标公式不同的关键点是"修正量不是残差"；同时它进一步证明槽位模型的"稀疏"只是注意力竞争。

---
### 9. SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models
- **作者 / 年份 / 会议或期刊 / 链接**：Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg / 2023 / ICLR 2023 / https://arxiv.org/html/2210.05861v2（HTML 正文，公式已核对）
- **核心更新方程（Section 3.1–3.2）**：
  $$
  \mathcal S_t=f_{SA}(\tilde{\mathcal S}_t,h_t),\qquad \tilde{\mathcal S}_t=f_{trans}(\mathcal S_{t-1})\ (t\ge 2)
  $$
  $$
  G_t=\mathrm{Linear}(\mathcal S_t)+\mathrm{P.E.},\qquad \hat{\mathcal S}_{t+1}=\mathcal T(G_{\le t})\ \text{（自回归 rollout）}
  （注：predictor 与 rollout 的紧凑记法 T(G≤t) 为本文概括；原文以文字描述「Transformer predicts slots at the next timestep」。）
  $$
  训练：特征空间与图像空间的重构损失。
- **与目标公式异同（A–G）**：
  - **A 有**：$f_{trans}$ 是显式先验（想象下一帧槽位），并被用作 slot attention 的初始化。
  - **B 否**：观测修正仍是 slot attention 重绑定，不是残差。
  - **C 否**：没有 $h^-+B$ 的加性残差形式。
  - **D 部分**：$f_{SA}$ 吃先验槽位 $\tilde{\mathcal S}_t$ 与观测特征 $h_t$。
  - **E 否**：帧/步索引，rollout 的 k 是 horizon，不是秒。
  - **F 否**：全部 slot 一起更新；"对象"是表示单元，不是更新单元。
  - **G**：重构 + 一步预测（latent rollout），自监督。
- **任务与训练目标**：从视频预测未来对象状态（rollout），用于视频预测、VQA、goal-conditioned planning。
- **最接近的先例（该文自引）**：OCVT（Wu 2021）、SAVi/STEVE、Dreamer 系列（世界模型）。
- **Q1/Q2/Q3**：Q1 是"先验→观测修正"两段，但修正非残差驱动；Q2 否；**Q3 是——世界模型 + 规划**（摘要明确："goal-conditioned planning"）。
- **新颖性判断**：**削弱 A 轴 + 明确"世界模型"用途已被占**。目标公式若定位在"给规划服务"，SlotFormer 已经是槽位世界模型的规划先例。

### 10. SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos
- **作者 / 年份 / 会议或期刊 / 链接**：Gamaleldin F. Elsayed, Aravindh Mahendran, Sjoerd van Steenkiste, Klaus Greff, Michael C. Mozer, Thomas Kipf / 2022 / NeurIPS 2022 / https://arxiv.org/html/2206.07764v2
- **核心更新方程**：更新律与 SAVi 相同（Slot Attention corrector + Transformer predictor）。该文改变的是**监督信号**：以**深度预测**替代光流，从而去掉"物体必须独立运动"的假设；并去掉首帧条件输入、使用均匀先验。【公式与 SAVi 同式，本文只核对架构段落，未逐式重抄】
- **与目标公式异同（A–G）**：
  - **A 有（继承 SAVi）**；**B 否**（同 SAVi，corrector 是重绑定）；**C/D 部分**；**E 否**；**F 软**；**G** 深度回归（自监督、逐像素），而非重构/ELBO。
- **任务与训练目标**：真实世界视频（Waymo）的物体中心表示；以深度为训练信号，验证"监督信号可替换"。
- **最接近的先例（该文自引）**：SAVi、Slot Attention、自监督深度/光流。
- **Q1/Q2/Q3**：Q1 是"每帧修正"非误差驱动；Q2 否；Q3 表示学习。
- **新颖性判断**：不直接冲击残差式。它的价值在于说明**槽位模型的学习信号是"换一个重建目标"，而不是"由观测残差决定更新"**。

### 11. SOLD: Slot Object-Centric Latent Dynamics Models for Relational Manipulation Learning from Pixels
- **作者 / 年份 / 会议或期刊 / 链接**：Malte Mosbach, Jan Ewertz, Angel Villar-Corrales, Sven Behnke / 2025 / ICML 2025 / https://arxiv.org/abs/2410.08822（项目页 https://slot-latent-dynamics.github.io/；SAVi 部分式 (1)(2) 已核对）
- **核心更新方程（原式 (1)(2)）**：
  $$
  A=\mathrm{softmax}_N\Bigl(\frac{q(Z_{t-1})\cdot k(F_t)^\top}{\sqrt D}\Bigr)\in\mathbb{R}^{N\times L},\qquad
  Z_t=\mathrm{MLP}\bigl(\mathrm{GRU}(A\cdot v(F_t),\,Z_{t-1})\bigr),\quad A_{n,l}\leftarrow\frac{A_{n,l}}{\sum_i A_{n,i}}\tag{1,2}
  $$
  动力学：基于 OCVP 的 transformer（时间注意力 + 关系注意力）自回归预测 $Z_{t+1}$，条件于历史槽位与动作；世界模型 + imagined rollout 训练 actor/critic。【动力学具体公式未逐式抽取，标为结构性描述】
- **与目标公式异同（A–G）**：
  - **A 有**：动力学模型输出下一时刻先验槽位。
  - **B 否**（继承 SAVi/OCVP，修正非残差）。
  - **C 部分**；**D 部分**；**E 否**（动作步）；**F 否**；**G**：RL 回报（imagined rollout）+ 重构/一致性。
- **任务与训练目标**：像素输入的 model-based RL（操作任务），object-centric latent dynamics + 想象 rollout。
- **最接近的先例（该文自引）**：SAVi、OCVP（Villar-Corrales 2023）、DreamerV3/TD-MPC2（对比基线）。
- **Q1/Q2/Q3**：Q1 是"每步后验修正"非残差；Q2 否；**Q3 是——给规划/控制服务**。
- **新颖性判断**：不冲击残差式；但补上"槽位世界模型已用于 MBRL"这一条，进一步压缩"槽位状态 + 未来生成"的原创空间。

---

## 4. "delta networks" 到底指什么

检索（query: "delta networks" neural network delta rule state update meaning）得到的不是某个叫 "Delta Network" 的单一架构，而是三条线，其中最贴近的是第三条：

1. **delta rule（Widrow–Hoff, 1960）**：$w\leftarrow w+\eta\,(t-y)\,x$，感知机/自适应滤波的误差修正学习律。这就是"残差修正"思想的原始出处。
2. **DeltaNet（Schlag, Irie, Schmidhuber, ICML 2021）**：把线性注意力的 fast-weight 纯加性写入换成 delta rule，见 §4.12。
3. **Gated DeltaNet（Yang, Kautz, Hatamizadeh, ICLR 2025）**，论文标题字面就是 *Gated Delta **Networks***，第 2.2 节标题为 "DELTA NETWORKS: LINEAR ATTENTION WITH DELTA RULE"。**结论：'delta networks' 指 delta-rule 系列线性注意力/快权重状态更新网络**（arXiv:2412.06464 原文措辞已核对），**不是**某种"用 Δ 表示时间间隔"的网络。

### 12. Linear Transformers Are Secretly Fast Weight Programmers
- **作者 / 年份 / 会议或期刊 / 链接**：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139:9355–9366 / https://arxiv.org/html/2102.11174v3（PMLR 题录页已核对）
- **核心更新方程（原式 (20)–(25)，HTML 正文已核对）**：
  $$
  \bar v^{(i)}=W^{(i-1)}\phi(k^{(i)})\tag{20},\qquad
  \beta^{(i)}=\sigma(W_\beta x^{(i)})\tag{21}
  $$
  $$
  v_{\text{new}}^{(i)}=\beta^{(i)}v^{(i)}+(1-\beta^{(i)})\bar v^{(i)}\tag{22}
  $$
  $$
  W^{(i)}=W^{(i-1)}+\underbrace{v^{(i)}_{\text{new}}\otimes\phi(k^{(i)})}_{\text{write}}-\underbrace{\bar v^{(i)}\otimes\phi(k^{(i)})}_{\text{remove}}
  =W^{(i-1)}+\beta^{(i)}\bigl(v^{(i)}-\bar v^{(i)}\bigr)\otimes\phi(k^{(i)})\tag{23,24}
  $$
  $$
  y^{(i)}=W^{(i)}\phi(q^{(i)})\tag{25}
  $$
- **与目标公式异同（A–G）**：
  - **A 有**：$W^{(i-1)}$ 就是先验状态；但**转移是恒等映射**，没有 $\Phi_\Delta$ 的动力学。
  - **B 有，形式几乎逐字对应**：$\bar v^{(i)}=W^{(i-1)}\phi(k^{(i)})$ 就是 $D(h^-)$（"用当前键去记忆里检索预测值"），$v^{(i)}-\bar v^{(i)}$ 就是 $z-D(h^-)$。
  - **C 有**：$W^{(i)}=W^{(i-1)}+\beta\,r\otimes\phi(k)$ 就是加性修正。
  - **D 有**：r 依赖 $W^{(i-1)}$，写强度 $\beta^{(i)}=\sigma(W_\beta x^{(i)})$ 依赖输入；组合起来 B 依赖 $(h^-,r)$。
  - **E 无**：i 是 token 索引，没有任何时间间隔。
  - **F "局部化"但不是稀疏**：更新是沿 $\phi(k^{(i)})$ 的秩一外积（局部于被查询的键方向），但 $\beta^{(i)}\in(0,1)$ 永不为 0，**每次都会写**；论文自己也把 overcapacity 下的键间串扰（crosstalk）当成主要问题。
  - **G**：语言建模/机器翻译的端到端任务损失（无重构、无 ELBO）。
- **任务与训练目标**：序列记忆与线性注意力；把记忆更新从"纯加性外积"改成"误差修正 delta rule"，并学习动态写强度。
- **最接近的先例（该文自引）**：Widrow & Hoff 1960 delta rule；Schmidhuber 1991 的 fast weight programmers；Schlag & Schmidhuber 2018。
- **Q1/Q2/Q3**：Q1 **是，且是最纯粹的"用预测误差修正状态"之一**；Q2 否；Q3 通用序列建模。
- **新颖性判断**：**严重削弱**。把 $h^-$、$r=z-D(h^-)$、$h=h^-+B(h^-,r)$ 三条同时拿掉 Δ 之后，就是式 (24)。区别只在：目标公式的状态是"世界隐状态"而非"键值记忆"，且需要 $\Phi_\Delta$ 与时间语义。

### 13. Gated Delta Networks: Improving Mamba2 with Delta Rule
- **作者 / 年份 / 会议或期刊 / 链接**：Songlin Yang (MIT CSAIL), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA) / 2025 / ICLR 2025 / https://arxiv.org/abs/2412.06464（官方 PDF: https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf，式 (10) 已核对）
- **核心更新方程（原式 (10)）**：
  $$
  S_t=S_{t-1}\bigl(\alpha_t(I-\beta_t k_t k_t^\top)\bigr)+\beta_t v_t k_t^\top\tag{10}
  $$
  其中 $\alpha_t\in(0,1)$ 是数据依赖的衰减门（快速清空记忆），$\beta_t$ 是写强度（纯 delta 规则当 $\alpha_t\to1$）。DeltaNet 原形见该文第 2.2 节：
  $$
  S_t=S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_tk_t^\top
  $$
- **与目标公式异同（A–G）**：
  - **A 有**：$S_{t-1}$ 先验；转移是"衰减 + 单位阵擦除"的线性映射（$\alpha_t(I-\beta kk^\top)$）——**形式上是一个真正的 $\Phi$，但 Δ 不是时间，是 token 级学习门**。
  - **B 有**：残差仍是 $v_t-S_{t-1}k_t$（体现在 $(I-\beta kk^\top)$ 里）。
  - **C 有**：加性写入 $\beta_t v_tk_t^\top$。
  - **D 有**。
  - **E 无**：$\alpha_t$ 虽被称为 decay，但按 token 索引，不携带秒；论文自己把它解释为"记忆管理"而非时间采样。
  - **F 软选择性**：作者原话是"selectively updates specific content without affecting other information"——这是**软**的 targeted update；不是 L0/掩码/事件触发，且 $\beta_t>0$ 意味着永远在写。
  - **G**：语言建模等任务损失。
- **任务与训练目标**：长上下文序列建模、检索、长度外推；并行训练算法（chunkwise/WY 表示）。
- **最接近的先例（该文自引）**：DeltaNet（Schlag 2021；Yang 2024 并行化）、Mamba2（GLA 衰减）、Widrow–Hoff。
- **Q1/Q2/Q3**：Q1 是（delta 版）；Q2 否；Q3 通用序列建模。
- **新颖性判断**：**削弱**。它甚至把"选择性改写"与"遗忘衰减"统一成一个 Φ 矩阵，这正是"$\Phi_\Delta$ + 残差修正"的线性代数版本，只是时间不是秒。

---
## 5. 七轴汇总表

| # | 工作 | A 显式先验 | B 观测残差 | C 加性修正 | D B(h⁻,r) | E 真实 Δ | F 稀疏/局部化 | G 训练目标 |
|---|---|---|---|---|---|---|---|---|
| 1 | Trimpe & D'Andrea 2012 | ✅ 式(10) | ✅ 式(8) | ✅ 式(11) | ✅ | ❌ | ✅ 硬事件 | 无（控制设计） |
| 2 | Sijs & Lazar 2009 | ✅ 隐含 | ~ 等价 | ~ 等价 | ~ | ✅ $A_\tau$ | ✅ 硬事件 | 无 |
| 3 | Miskowicz 2006 | ❌ | ❌（对上次上报） | ❌ | ❌ | ~ | ✅ 硬阈值 | 无 |
| 4 | Skip RNN 2018 | ❌ | ❌ | ~（替换） | ❌ | ❌ | ✅ 硬门控 | 任务+预算 |
| 5 | ACT 2016 | ❌ | ❌ | ❌（平均） | ❌ | ❌ | ~ 软深度 | 任务+ponder |
| 6 | Slot Attention 2020 | ❌ | ❌ | ~ GRU | ~ | ❌ | ~ 分配竞争 | 重构 |
| 7 | SAVi 2022 | ✅ predictor | ❌（重绑定） | ~ GRU | ~ | ❌ | ❌ | 光流/重构 |
| 8 | STEVE 2022 | ~ | ❌ | ~ RNN | ~ | ❌ | ❌ | 重构 CE |
| 9 | SlotFormer 2023 | ✅ | ❌ | ❌ | ~ | ❌ | ❌ | 重构+预测 |
| 10 | SAVi++ 2022 | ✅ | ❌ | ~ | ~ | ❌ | ❌ | 深度回归 |
| 11 | SOLD 2025 | ✅ | ❌ | ~ | ~ | ❌ | ❌ | RL 回报+重构 |
| 12 | DeltaNet 2021 | ✅ 恒等转移 | ✅ 核心 | ✅ | ✅ | ❌ | ~ 秩一、永不全零 | 任务损失 |
| 13 | Gated DeltaNet 2025 | ✅ 衰减转移 | ✅ | ✅ | ✅ | ❌ | ~ 软 targeted | 任务损失 |

**目标公式命中的组合**：A+B+C+D 同时成立的是 #1（Trimpe）、#12（DeltaNet）、#13（Gated DeltaNet）；其中 #1 还额外命中 F（硬触发）且带稳定性定理。E（真实秒级 Δ）只在 #2 中以 $A_\tau$ 形式出现，但那篇没有学习式 B、也没有"部分状态不动"的稀疏性。

---

## 6. 负面结果与反例

### 6.1 有没有与目标公式逐字对应的已有工作？
**分成两半，各自都有近乎逐字的先例。**
- **前半句 $h^-=\Phi_\Delta(h)$ + 后半句 $r=z-D(h^-)$, $h=h^-+B$**：Trimpe & D'Andrea 2012 式 (10)(11) 是完整命中（线性情形），且式 (8) 是"|r| 超阈值才修正"的硬稀疏版本，附带 $\lVert e\rVert_\infty\le\frac{m}{1-\rho}\lVert L\rVert\lVert\delta\rVert$ 的稳定性上界。
- **$r=z-D(h^-)$、$h=h^-+B(h^-,r)$、B 可学习**：DeltaNet 式 (24) 完全命中（$D(h^-)=W\phi(k)$，$B=\beta r\phi(k)^\top$），Gated 版式 (10) 再把 Φ 写成衰减矩阵。
- **Φ 带真实时间 Δ**：Sijs & Lazar 2009 的 $A_\tau,B_\tau$ 命中；但它配的是事件触发贝叶斯滤波，不是学习式 B。
- **结论**：**没有任何单篇把"秒级 $\Phi_\Delta$ + 学习式残差 B + 硬稀疏选择"三件套同时写下**；但三件套的每一件、以及其中两两组合，都已被发表过（2012 / 2009 / 2021）。**结构层面不具备原创性；最多是组合 + 时间语义的工程化。** 这一点必须如实报告。

### 6.2 软稀疏不保证"不动"
- **Slot Attention**：所有 K 个 slot 每轮都被 GRU 更新；实现上"we further add a small offset ε to the attention coefficients to avoid numerical instability"（原文），因此没有严格零权重；"稀疏"只是 softmax 在 slot 维的**分配竞争**。
- **SAVi / STEVE / SlotFormer / SAVi++ / SOLD**：每帧都对全部 slot 做一次 slot attention 重绑定。**即使画面完全符合预测，corrector 仍然会写一遍**（SAVi 式 (1) 的输入是当前帧特征 $h_t$，不是 $h_t-\hat h_t$）。
- **STEVE**：注意力是加权平均，分母 $\sum_j A_{t,n,j}$ 归一化，聚合值恒非零。
- **ACT**：mean-field 加权平均，所有中间状态都参与输出。
- **软门控会被优化器利用**：ACT 论文脚注亲口承认，若按"期望 ponder 时间"训练，网络会**作弊**——把 $h_1$ 设到略低于阈值、中间置 0、最后一步抬高，从而以极低惩罚实现"其实没少算"。这是"软稀疏可被钻空子"的直接文献证据。
- **真正保证"完全不动"的只有硬机制**：Skip RNN 的 $u_t\in\{0,1\}$（$u_t=0$ 时 $s_t=s_{t-1}$ 逐位复制）、Trimpe 的事件触发（无事件时 $\check x(k\mid k)=\check x(k\mid k-1)$）、send-on-delta 的阈值。**这三者的选择依据分别是：输入派生的累积概率（Skip RNN）、预测误差范数（Trimpe）、信号变化幅值（SoD）**——只有第二个是预测误差驱动。

### 6.3 事件触发滤波有几十年历史，且有稳定性理论
- Miskowicz 2006 明确把电平交叉采样追溯到 **1950 年代末 Ellis**："it is not necessary to sample periodically, but only when quantized data change from one possible value to the next"；并列出 event-based sampling / level-crossing / magnitude-driven / Lebesgue sampling / deadband 等一长串同义词。
- Sijs & Lazar 2009 给出事件采样的一般集合形式 $y(t)\notin H(z_{k_e-1},t)$ 与 send-on-delta 特例 $|y(t)-y_{k_e-1}|>\Delta$，并证明事件基估计器的协方差有界（定理 1）。
- Trimpe & D'Andrea 2012 给出**以预测误差为触发条件**的完整闭环：触发律 (8)、切换观测器 (11)、误差上界 (17)。**"$\lVert r\rVert>\delta$ 才更新"不是新想法，是 2012 年就有稳定性证明的标准做法。**
- 由此推出：目标公式中"用预测误差大小决定改不改"的部分，若要主张新颖性，必须证明学习式 B 与硬触发**联合**带来了控制理论没有的东西（例如可微训练、对象级选择、非线性观测模型）——仅"结构"不够。

### 6.4 Skip RNN 的选择依据不是预测误差
- 式 (4)(5)：$\Delta\tilde u_t=\sigma(W_p s_t+b_p)$，$\tilde u_{t+1}=u_t\Delta\tilde u_t+(1-u_t)\bigl(\tilde u_t+\min(\Delta\tilde u_t,1-\tilde u_t)\bigr)$。门控只看隐状态（输入派生），**没有任何"预测—观测"比较**。
- 论文动机是计算效率（"shortens the effective size of the computational graph"），不是状态估计；它自己也承认**累积门控不擅长一次跳过很多步**（"Skip RNN models show some difficulties skipping a large number of updates at once, probably due to the cumulative nature of $\tilde u_t$"）。
- 因此：它可以支持"硬稀疏更新已有先例"，**不能**支持"按预测误差选择更新"这一说法。把 Skip RNN 当作目标公式的先例是**误引**；反过来，用它可证明"选择性更新"本身不新。

### 6.5 槽位稀疏性不是由观测残差驱动
- Slot Attention 的"竞争"发生在 **输入特征到 slot 的分配**上（softmax over slots，式 (1)），训练信号是重构；没有任何一步把 $z-\hat z$ 反传给"该更新哪几个 slot"。
- SAVi/STEVE/SlotFormer/SOLD 的"预测器"与"修正器"是**串联**的，不是"残差 → 门控"：预测结果只用于初始化 query 或作为 rollout 起点。
- 结果是：**对象级状态 + 稀疏性**这两个卖点都有人做过，但**"用观测残差决定哪个对象要改"这个连接没有先例**——这可能是目标公式唯一还站得住的空隙（但属于"把两个已知模块接起来"的组合创新，且需要证明比全量重绑定更好）。

### 6.6 这类结构的已知失效模式（用于攻击/防守都可以）
1. **阈值/门控的迟滞与抖动**：阈值 δ 太小学 Trimpe 的极限结论（δ→0 退化为全更新，性能回到全通信）；太大则误差上界线性放大（∝‖δ‖）。实践中需要自适应/随机化阈值，目标公式未处理。
2. **切换系统本身可能不稳定**：Trimpe 明确指出每个模式可能不可稳（$(A,C_i)$ 不必可检测），**即使每个模式稳定也不保证切换稳定**。这条对"只更新部分状态/部分对象"的方案是硬约束：局部更新的稳定性需要单独证明。
3. **累积门控的一次跳多步失效**（Skip RNN 自述）与**二值化不可导**（round/STE）带来的训练噪声。
4. **软门控被优化器利用**：ACT 的"halting 作弊"（见 §6.2）。
5. **状态改写缺乏"清空"能力**：Gated DeltaNet 明确指出纯 delta 规则"only modifies a single key-value pair at a time, lacks the ability to rapidly clear outdated information"，因此需要 $\alpha_t$ 衰减门——也就是说 Φ 不能只是恒等，**必须有遗忘/衰减**，否则旧状态会污染新观测。
6. **记忆串扰/容量上限**：DeltaNet 第 4.1 节论证当序列长于键维度时进入 overcapacity，检索被 crosstalk 破坏；"局部化更新"≠"不互相干扰"。
7. **槽位绑定错误/坍缩**：对象中心模型依赖注意力竞争，物体数、遮挡、静止物体都会破坏绑定（SAVi 需光流先验、SAVi++ 需深度先验正是因为这个）；这与"只改相关对象"的前提冲突——先得知道哪个 slot 是"相关对象"。
8. **Δ 缺失导致采样率变化即语义漂移**：所有槽位/delta 模型都按帧或 token 索引，输入帧率一变，"一步"的物理含义就变了。Sijs & Lazar 用 $A_\tau$ 显式处理了这一点；目标公式若把 Δ 只当超参而不做半群/采样一致性分析，会被审稿人直接指出。

### 6.7 给 lead 的三条结论
1. **最接近的三个先例**：Trimpe & D'Andrea 2012（预测误差触发 + 加性残差修正 + 稳定性上界）、DeltaNet 2021 / Gated DeltaNet 2025（残差式状态改写 + 学习写强度，缺真实时间）、SAVi 2022（显式 predictor-corrector 的槽位版本，缺残差）。
2. **最致命的三个负面证据**：(i) 式 (8)+(10)+(11) 已把"‖z−D(h⁻)‖>δ 才做加性修正"写全并证明稳定；(ii) 式 (24) 已把"h = h⁻ + β·r⊗φ(k)"写全并端到端训练；(iii) 事件触发/电平交叉采样有 1950 年代起的谱系与稳定性理论，"按误差大小决定更新"属于领域常识。
3. **还能主张什么**：只能主张**组合与时间语义**——(a) 秒级 Δ 显式进入 $\Phi_\Delta$ 且与学习式 B 联合训练；(b) **对象/槽位级别**的硬选择（由观测残差驱动），现有槽位模型全是全量重绑定；(c) 相应的稳定性/一致性分析（把 Trimpe 的 $\lVert e\rVert_\infty$ 界、Sijs 的协方差有界性推广到学习式非线性 B 与非均匀采样）。**不能主张"结构原创"。**

---

## 附：核实日志（每条链接均实际拉取）

| 条目 | 拉取方式 | 结果 |
|---|---|---|
| Trimpe & D'Andrea 2012 IFAC PDF | tavily_extract | ✅ 式 (8)(9)(10)(11)、定理 1、页 91–96 |
| Sijs & Lazar 2009 HSCC PDF (TU/e) | tavily_extract | ✅ 式 (6)(7)、send-on-delta 集合、定理 1 |
| Miskowicz 2006 Sensors (MDPI 全文) | tavily_extract | ✅ 第 3 节定义；式 (1) 显示公式抽取丢失，按正文文字复述 |
| Skip RNN arXiv PDF | tavily_extract | ✅ 式 (1)–(6)、作者、ICLR 2018 |
| ACT arXiv PDF | tavily_extract | ✅ 式 (3)–(13)、作弊脚注 |
| Slot Attention arXiv PDF | tavily_extract | ✅ Algorithm 1、式 (1)(2) |
| SAVi arXiv PDF | tavily_extract | ✅ 式 (1)–(4)、predictor-corrector 原话 |
| STEVE arXiv PDF | tavily_extract | ✅ 递推式、损失 |
| SlotFormer arXiv HTML | tavily_extract | ✅ 式、planning/VQA 用途（摘要） |
| SAVi++ arXiv HTML | tavily_extract | ✅ 背景段；架构与 SAVi 同 |
| SOLD arXiv PDF + 项目页 | tavily_extract | ✅ 作者/ICML 2025/式 (1)(2)；动力学公式未逐式抽取 |
| DeltaNet arXiv HTML + PMLR 页 | tavily_extract | ✅ 式 (20)–(25)、作者、ICML 2021 |
| Gated DeltaNet ICLR PDF | tavily_extract | ✅ 式 (10)、第 2.2 节标题、作者 |

**未核实项（如实标注）**：① Miskowicz 式 (1) 的显示公式（PDF 文本层丢失，仅按正文复述）；② SOLD 的动力学具体方程（只取到结构描述）；③ Trimpe & D'Andrea 2014 TAC 的 variance-based triggering 仅作为该文自引提及，未单独拉取原文，故未单列条目。
