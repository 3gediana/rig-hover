# 03 · 滤波器、观测器与 innovation form —— 目标公式的既有文献对照

> 专题：Kalman 滤波 / 非线性观测器 / 学习式观测器 / neural Kalman filter / innovation form
> 工具：tavily_search (advanced) + tavily_extract 抓正文；每条标注核实级别。
>
> 被检验的目标公式（ASCII）：
>
>     h- = Phi_Delta(h)        # 带真实时间间隔 Delta 的状态转移（预测/先验）
>     r  = z - D(h-)           # 观测空间残差/新息
>     h  = h- + B(h-, r)       # 残差式修正（学习式增益）
>
> 七轴：**A** 显式先验 h⁻；**B** 观测空间残差 r=z−D(h⁻)；**C** 加性修正；**D** B 同时依赖 (h⁻,r)；**E** 真实时间间隔 Δ；**F** 稀疏/选择性/局部化；**G** 训练目标。
> 三问：**Q1** 是否用新息修正隐状态；**Q2** 时间语义是观测索引还是秒、Δ 从哪进入；**Q3** 用途是规划/生成还是通用在线状态估计。
>
> 核实级别：**[全文]** = 取到正文并核对公式；**[仅摘要级]** = 只取到摘要/检索片段；**[未核实]** = 取不到。

---
## 0. 先给结论（不客气的版本）
目标公式不是一个新结构。它逐项等于：

1. **Kalman (1960) 的 (20)–(22) 式**：新息 ỹ = y − M x*，更新 x*(t+1|t) = Φ*(t+1;t)x*(t|t−1) + Δ*(t)y(t)，Φ* = Φ − Δ*M。把 x* 记作 h⁻、M 记作 D、Δ* 记作 K，两者逐字同构。
2. **连续-离散 EKF 的标准形式**（Jørgensen 2007 的 (2a)–(6c)，即 Jazwinski 1970 / Simon 2006 那一套）：ŷ=h(t_k,x̂⁻)、e=y−ŷ、x̂⁺=x̂⁻+Ke，而 Δ=t_{k+1}−t_k 通过 dx̂/dt=f(t,x̂) 在区间上的积分进入预测。**连"Δ 只进预测、不进修正"这一点都相同。**
3. **Recurrent Kalman Network (ICML 2019) 的 (4) 式**：z⁺=z⁻+q⊙(w−Hz⁻)，q=σ⁻⊘(σ⁻+σ_obs)。这是"学习式增益 + 观测空间新息 + 加性修正"的现成神经网络实现。

唯一可能剩下的新意：(i) 增益交给网络学（KalmanNet 2022 已做，增益输入含新息差与后验状态差）；(ii) Φ 显式吃真实 Δ（连续-离散滤波 1960 年代就有，2023–2026 的 neural-ODE 观测器与 Mamba-2 CD-EKF 又把它与学习拼在一起）。

**判断：结构层面新颖性已被吃掉；可辩护的剩余空间只能落在"用什么训练信号/约束把 B 钉成有物理意义的增益"和"Δ 参与增益"这两处，而不是方程形式本身。**

---
## 1. 经典估计理论（4 篇）
### 1. A New Approach to Linear Filtering and Prediction Problems
- 作者 / 年份 / 期刊 / 链接：Rudolf E. Kalman / 1960 / *Journal of Basic Engineering* (Trans. ASME, Series D) 82(1):35–45 / [PDF](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/k/Kalman1960.pdf) **[全文]**
- 核心更新方程（原式照抄，编号为原论文编号；页脚 "Transactions of the ASME–Journal of Basic Engineering, 82 (Series D): 35-45. Copyright © 1960 by ASME" 已核）：
```
   ỹ(t|t−1) = y(t) − M(t)x*(t|t−1)                        (20)   ← 新息
   x*(t+1|t) = Φ*(t+1;t)x*(t|t−1) + Δ*(t)y(t)             (21)
   Φ*(t+1;t) = Φ(t+1;t) − Δ*(t)M(t)                       (22)
   Δ*(t) = Φ(t+1;t)P*(t)M'(t)[M(t)P*(t)M'(t)]⁻¹           (28)   ← 增益
   P*(t+1) = Φ*(t+1;t)P*(t)Φ'(t+1;t) + Q(t)               (30)   ← Riccati
   x̃(t+1|t) = Φ*(t+1;t)x̃(t|t−1) + u(t)                   (23)   ← 误差动力学
```
  (21)+(22) 等价于 x*⁺ = Φ x* + Δ*(y − M x*)。作者自述："optimal estimation is performed by a linear dynamic system of the same form as (14). The state of the estimator is the previous estimate, the input is the last measured value of the observable random variable y(t)."
- 与目标公式异同（A–G）：
  - **A 有**。x*(t|t−1) 就是 h⁻，且显式区分先验/后验（t|t−1 vs t|t）。
  - **B 有**。式 (20) 是显式 y − Mx*；(19) 说 E[x(t+1)|Z(t)] 的最后一层必须是 ỹ(t|t−1) 的线性运算。
  - **C 有**。(21)–(22) 的等价改写就是加性修正。
  - **D 无**。Δ* 依赖 (Φ,P*,M,Q)，**不依赖** (x*,ỹ)；只有乘积 Δ*·ỹ 里出现 r。这正是目标公式 B(h⁻,r) 与 Kalman 的唯一差别（见 §4 N2）。
  - **E 无（但留了门）**。采样周期固定为 1；Remarks (j)："By letting the sampling period (equal to one so far) approach zero, the method can be used to obtain the specification of a differential equation for the optimal filter." 作者本人已把连续时间极限当作 obvious extension。
  - **F 无**。全状态、全观测、稠密矩阵。
  - **G 无训练**。解析 MMSE，无数据、无损失。
- 任务与训练目标：含噪量测下的 MMSE 状态估计；正交投影推导，无学习。
- 最接近的先例：Wiener–Hopf 积分方程（文中 [1]）；作者明说式 (32) 的 Riccati 递推 "plays the role of the Wiener-Hopf equation"。
- 新颖性判断：**强烈削弱。** 目标公式三行里三行在 1960 年已成文，"新息"对应的量 ỹ(t|t−1) 有编号、有物理意义、有误差动力学 (23)。任何把"预测-新息-修正"写成贡献的表述都会被一句话打回。
### 2. Observing the State of a Linear System（Luenberger 观测器）
- 作者 / 年份 / 期刊 / 链接：David G. Luenberger / 1964 / *IEEE Transactions on Military Electronics* 8:74–80（DOI 10.1109/TME.1964.4323124，书目经 [SCIRP 引用记录](https://www.scirp.org/reference/referencespapers?referenceid=1291918) 核实）。**原文全文未取到**（IEEE 付费墙）；公式取自 [UW ME547 讲义](https://faculty.washington.edu/chx/teaching/me547/2_5_observer_and_observerStateFeedback.pdf) 复述 **[全文·讲义]**
- 核心更新方程：
```
   ẋ̂ = A x̂ + B u + L(y − ŷ) = A x̂ + B u + L(y − C x̂) = (A − LC)x̂ + L y + B u
   ŷ  = C x̂
   e  = x − x̂,   ė = A e − L C e = (A − LC) e
```
  （讲义原话："the Luenberger observer adds correction based on output differences"；若 A−LC 的全部特征值在左半平面，则误差渐近稳定。Gajic 讲义亦称 "The theory of observers started with the work of Luenberger (1964, 1966, 1971)"，见 [observers.pdf](http://eceweb1.rutgers.edu/~gajic/psfiles/observers.pdf)）
- 与目标公式异同（A–G）：
  - **A 有**。A x̂ + Bu 就是 Φ(h) 的线性特例。
  - **B 有**。y − Cx̂ 是显式输出误差。
  - **C 有**。
  - **D 无**。L 是设计常数矩阵（极点配置），与 (x̂, y−Cx̂) 无关——与"学习式 B(h⁻,r)"最大的形式差别。
  - **E 部分**。连续时间 t 出现，但 L 不随 Δ 变；采样/间隔语义完全缺失。
  - **F 有（弱的）**。reduced-order observer 只估计 n−p 维（p=观测维数），可视为"只重构可观测方向"的雏形；但这是结构化降维，不是选择性门控。
  - **G 无**。极点配置，无训练目标。
- 任务与训练目标：确定性线性系统全状态重构；目标是 A−LC 特征值可任意配置（(A,C) 可观测）。
- 最接近的先例：Luenberger 1966（降维观测器）、Luenberger 1971 "An introduction to observers"（IEEE TAC 16(6):596–602）。
- 新颖性判断：**削弱。** 目标公式 = Luenberger 观测器 + 非线性 Φ/D/B。这是控制理论里的 output injection 标准形式；把 B 换成神经网络只把常值增益换成可学映射，不改变结构类别。**注意：目标公式没写 B(h⁻,0)=0**——若 B(h⁻,0)≠0，它连观测器都不是（零新息时还在改状态）。
### 3. 连续-离散 Kalman/EKF 的标准形式（Jazwinski 1970 / Simon 2006 / Jørgensen 2007）
- 作者 / 年份 / 会议 / 链接：John Bagterp Jørgensen / 2007 / 6th ECCE (European Control Conference) / [PDF](https://skoge.folk.ntnu.no/prost/proceedings/ecce6_sep07/upload/3520.pdf) **[全文]**
  书：A. H. Jazwinski, *Stochastic Processes and Filtering Theory*, Academic Press, 1970（书未取到；"continuous-discrete 模型"归属经两处独立引用核实：[Hagen CD-UKF 讲义](https://www.fernuni-hagen.de/wirtschaftswissenschaft/emeriti/docs/singer/ukf.pdf) 写 "the recursive scheme (Jazwinski, 1970)"）；D. Simon, *Optimal State Estimation*, Wiley, 2006。
- 核心更新方程（原式照抄，编号为原论文编号）：
```
   系统:  dx(t) = f(t,x(t))dt + σ(t)dω(t),   y_k = h(t_k, x(t_k)) + v_k,  v_k ~ Niid(0,R_k)   (1a)(1b)

   观测预测:  ŷ_{k|k−1} = h(t_k, x̂_{k|k−1})                       (2a)   ← D(h⁻)
   新息:      e_k = y_k − ŷ_{k|k−1}                                (3)    ← r = z − D(h⁻)
   增益:      K_{fx,k} = P_{k|k−1} C_k' R_{k|k−1}⁻¹,  C_k=∂h/∂x   (4)    ← 依赖 P, C, R
   修正:      x̂_{k|k} = x̂_{k|k−1} + K_{fx,k} e_k                   (5a)   ← h = h⁻ + B(h⁻,r)
              P_{k|k} = P_{k|k−1} − K_{fx,k} R_{e,k} K_{fx,k}'      (5b)

   时间传播:  dx̂_k(t)/dt = f(t, x̂_k(t)),  x̂_k(t_k) = x̂_{k|k}      (6a)   ← Φ_Δ，Δ=t_{k+1}−t_k
              dP_k(t)/dt = A(t)P_k + P_k A(t)' + σσ'                (6b)
   即 x̂_{k+1|k} = x̂(t_{k+1}; t_k, x̂_{k|k})
```
  同构的连续时间写法（Simon 2006 记法，经 [Chen et al., ICML 2021 补充材料 式(59)](https://proceedings.mlr.press/v139/chen21t/chen21t-supp.pdf) 逐字核实）：K = P C' R̃⁻¹，ẋ̂ = f(x̂,u,0,t) + K[z − h(x̂,0,t)]，Ṗ⁻ = AP⁻ + P⁻A' + Q̃ − P⁻C'R̃⁻¹CP⁻。
- 与目标公式异同（A–G）：
  - **A 有**，且 (2a)(3)(5a) 三行与目标公式三行一一对应。
  - **B 有**，e_k = y_k − h(t_k, x̂_{k|k−1})。
  - **C 有**，(5a) 就是加性。
  - **D 无**。K = P C'R⁻¹ 只依赖协方差与雅可比——这是"最优"的全部来源，目标公式把这个依赖删掉了。
  - **E 有，本节最强的一条**。Δ=t_{k+1}−t_k **不是观测索引而是秒**；它通过 (6a) 的 ODE 积分（等价地通过转移矩阵 Φ(t_{k+1},t_k) 的流映射）进入预测；修正步 (3)(5a) **完全不含 Δ**。目标公式"Δ 只进 Φ_Δ、不进 B"与教科书一字不差。
  - **F 无**。
  - **G 无训练**（参数来自模型，不是数据）。
- 任务与训练目标：连续时间系统 + 离散观测的在线状态估计；无学习目标；论文本身是 critical discussion，重点在数值稳定（ESDIRK 积分 + 平方根/array 算法）。
- 最接近的先例：Jazwinski 1970；Simon 2006；(6) 的数值解法参考 ESDIRK 系列。
- 新颖性判断：**最致命的一击。** 三行逐项对应、Δ 的位置对应。任何以"带真实时间间隔的预测 + 新息修正"为主张的写法都等于重述 CD-EKF。
### 4. An innovations approach to least-squares estimation, Pt. I: Linear filtering in additive noise
- 作者 / 年份 / 期刊 / 链接：Thomas Kailath / 1968 / *IEEE Transactions on Automatic Control* AC-13(6):646–655 / [IEEE 记录](https://ieeexplore.ieee.org/abstract/document/1099025)（只抓到页眉，付费墙）；书目经 [Kailath 官方论文列表](http://web.stanford.edu/~tkailath/cgi-bin/PubsList.php) 核实（原文："An innovations approach to least-squares estimation, Pt. I: Linear filtering in additive noise. IEEE Trans. Automat. Control, AC-13(6):646--655, Dec. 1968"）**[仅摘要级]**
- 核心断言："The innovations approach to linear least-squares approximation problems is first to 'whiten' the observed data by a causal and invertible operation, ..."（ADS 摘要，经检索片段获得，未直连抓取）**[仅摘要级]**
- 与目标公式异同（A–G）：
  - **A/B/C 概念上有**：innovations 理论的全部内容就是"用 y−ŷ 这个白化过程承载全部新信息"。**具体方程未取到，不逐字引用。**
  - **D 无**。讨论最优滤波器的结构性质，不是学习式增益。
  - **E 无**。**F 无**。**G 无**（纯理论）。
- 任务与训练目标：线性最小二乘的"新息表示"（innovations representation），由此推出 KF/Riccati。
- 最接近的先例：Kailath & Frost, Pt. II（线性平滑）；Kailath 1970 综述。
- 新颖性判断：**削弱（命名层面 + 判据层面）。** 一，"innovation/新息"有 57 年历史、有专名、有综述，把它当作构成要素不构成主张。二，它给出**可检验判据**：最优滤波器的新息必须白（不相关于过去观测）；任意学出来的 B 生成不了白新息，所以目标公式的 r 形式上叫"新息"，**语义上不再是新息，只是残差 (residual)**。

---
## 2. 学习式滤波 / neural Kalman filter（7 篇）
### 5. Deep Kalman Filters
- 作者 / 年份 / 来源 / 链接：Rahul G. Krishnan, Uri Shalit, David Sontag / 2015 / arXiv:1511.05121（**无正式会议版**；后续工作为 AAAI 2017 *Structured Inference Networks for Nonlinear State Space Models*）/ [abs](https://arxiv.org/abs/1511.05121) · [HTML](https://arxiv.org/html/1511.05121) **[全文]**
- 核心更新方程（生成模型 + 结构化变分后验）：
```
   生成:  z_t ~ N(f_θ(z_{t-1}, u_{t-1}), ·),   x_t ~ p_θ(x_t | z_t, u_t)
   变分:  q_φ(z_t | z_{t-1}, x_{1:T}, u_{1:T})        （识别网络，含 RNN 记忆）
   ELBO 的 KL 项: Σ_t E[ KL( q_φ(z_t|z_{t-1},x⃗,u⃗) ‖ p_0(z_t|z_{t-1},u_{t-1}) ) ]
```
  （KL 分解式经 arXiv HTML 的 LaTeX alt 文本逐字核对）
- 与目标公式异同（A–G）：
  - **A 有（概率意义）**。先验 p_0(z_t|z_{t-1},u_{t-1}) 对应 Φ；但**不是**由 z 前向算出的确定性 h⁻，而是分布。
  - **B 无显式形式**。没有 r=z−D(h⁻) 这一行；观测被识别网络 q_φ **整体吞掉**（吃 z_{t-1} 与整段 x），不是先算残差再加。
  - **C 无显式加性**。**D 不适用**。
  - **E 无 Δ**。时间是观测索引 t，没有秒、没有间隔；变采样率需重训或插值。
  - **F 无**。
  - **G ELBO**（重构 + KL），非一步预测、非 RL。
- 任务与训练目标：高维观测的生成式时序建模 + 反事实推断（医疗 EHR、Healing-MNIST）；变分下界。
- 最接近的先例：Kalman 1960（线性特例）；EKF/UKF（Jazwinski, Wan et al.）；VAE（Kingma & Welling, Rezende et al.）；Raiko & Tornio (2009)。
- 新颖性判断：**不直接削弱结构新颖性，但削弱"学习式滤波"这个大方向**（2015 年就有）。**反向有用**：它证明"把 Kalman 做成深度生成模型"这条线上，目标公式的 B 只是把识别网络换成新息式更新——一个**更弱**的假设。要主张优势，必须证明"新息式参数化优于自由识别网络"，这是实验问题，不是公式新颖性。
### 6. Deep Variational Bayes Filters (DVBF): Unsupervised Learning of State Space Models from Raw Data
- 作者 / 年份 / 会议 / 链接：Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt / 2017 / ICLR 2017, arXiv:1605.06432 / [HTML](https://arxiv.org/html/1605.06432)（arXiv 元数据注 "Published as a conference paper at ICLR 2017"）**[全文]**
- 核心更新方程：
```
   z_{t+1} = f(z_t, u_t, β_t)                                   (8)   ← 给定 β 后确定性
   识别模型不再推断 z_t，而是推断转移参数 β_t
   （原文：the recognition model no longer infers latent states z_t, but transition parameters β_t）
```
- 与目标公式异同（A–G）：
  - **A 有（但不是新息式）**。转移是 f(z_t,u_t,β_t)，先验来自 β 的边缘化。
  - **B 无**。没有观测空间残差；观测通过解码器 p(x_t|z_t) 与 ELBO 反传，误差不经过 z−D(z)。
  - **C 无**。修正不是加性，整条轨迹由变分后验重参数化。
  - **D 不适用**。**E 无 Δ**（离散 t；论文强调 "enabling backpropagation through transitions"）。**F 无**。
  - **G ELBO**（重构 + 转移先验 KL）。
- 任务与训练目标：从原始像素无监督学习状态空间模型，服务下游控制 / model-based RL（论文原话提到 "control or model-based reinforcement learning"）。
- 最接近的先例：E2C（Watter et al. 2015）；SGVB；Kalman VAE 线。
- 新颖性判断：**削弱"学习式状态空间模型"整体新颖性**；对 B 结构**无直接对应**（它故意避免显式观测残差）。这也说明：**"不用新息"也是可行设计**，新息式不是必需结构，不能当卖点。
### 7. Kalman Variational Auto-Encoder (KVAE): A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning
- 作者 / 年份 / 会议 / 链接：Marco Fraccaro, Simon Kamronn, Ulrich Paquet, Ole Winther / 2017 / NeurIPS 2017 / [PDF](https://papers.nips.cc/paper_files/paper/2017/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf) **[全文]**
- 核心更新方程：
```
   LGSSM 骨干:  p(z_t|z_{t-1},u_t) = N(z_t; A_t z_{t-1} + B_t u_t, Q)      (1)
                 p(a_t|z_t)         = N(a_t; C_t z_t, R)                     (1)
   结构化变分:  q(a,z|x,u) = q_φ(a|x) · p_γ(z|a,u),  q_φ(a|x)=∏_t q_φ(a_t|x_t)   (5)
                其中 p_γ(z|a,u) 是**精确的 Kalman 平滑后验**
   动力学参数网络: γ_t = [A_t,B_t,C_t] = learnable(a_{0:t-1})                  (8)
```
- 与目标公式异同（A–G）：
  - **A 有，且在内部**。A_t z_{t-1} + B_t u_t；Kalman 平滑时先验/后验分离。
  - **B 有，但在伪观测空间**。Kalman 更新作用在 a_t（VAE 编码的低维流形）上，a_t = C_t z_t + noise，残差是 a_t − C_t ẑ_t。
  - **C 有**（Kalman 更新是加性），但**不在识别网络里**——q_φ(a_t|x_t) 是逐帧编码器，与前一状态无关。
  - **D 无**。增益来自 A_t,C_t,Q,R 的 Riccati 解。**E 无 Δ**。**F 无**。
  - **G ELBO**（含精确 Kalman 平滑因子；论文用它 principled 地处理缺失数据）。
- 任务与训练目标：视频/物理系统的无监督解耦表征 + 未来想象 + 缺失补全；ELBO。
- 最接近的先例：LGSSM 精确 KF/RTS 平滑；VAE；同组 SRNN（Fraccaro et al., NeurIPS 2016, [arXiv:1605.07571](https://arxiv.org/html/1605.07571v2)，其结构化后验 p_θ(z_t|z_{t-1},d_{t:T},x_{t:T}) 我已 [全文] 核实）。
- 新颖性判断：**削弱，但方式微妙。** 它把 Kalman 更新当作 ELBO 里的一个精确因子，说明"预测+新息修正"在深度生成模型里早是标准构件。**同时暴露目标公式的弱点**：KVAE 能用 Kalman 是因为骨干**线性高斯**、有闭式解；目标公式的 B(h⁻,r) 在学习式场景下没有闭式，也就没有最优性这层含义，只剩结构。
### 8. Backprop KF: Learning Discriminative Deterministic State Estimators
- 作者 / 年份 / 会议 / 链接：Tuomas Haarnoja, Anurag Ajay, Sergey Levine, Pieter Abbeel / 2016 / NeurIPS 2016 / [PDF](http://papers.neurips.cc/paper/6090-backprop-kf-learning-discriminative-deterministic-state-estimators.pdf) **[全文]**
- 核心更新方程（论文没有单一公式，而是"把滤波器当计算图"）：
```
   滤波器内部状态  s_t = (μ_{x_t}, Σ_{x_t})
   滤波器算子      s_{t+1} = κ(s_t, z_{t+1})     ← 原话：we use a standard Kalman filter update as κ
   输出算子        q(s_t) = (C_y μ, C_y Σ C_y')
   BPTT:  dL/ds_{t-1} = (dφ/ds_{t-1})(dL/dφ) + (ds_t/ds_{t-1})(dL/ds_t)      (1)
   其中 z_t = g_θ(o_t) 由卷积网络从图像产生，θ 与滤波器一起端到端训练
```
- 与目标公式异同（A–G）：
  - **A 有**（KF 更新内含先验）。**B 有**（观测编码器输出当作量测）。**C 有**。
  - **D 无**（标准 KF 增益）。**E 无 Δ**（离散步）。**F 无**。
  - **G 判别式一步预测损失**（负对数似然于标签 y_t），**不是 ELBO**——这是它最重要的贡献：KF 可用判别损失直接训。
- 任务与训练目标：视觉跟踪、KITTI 车辆位姿；标签驱动的判别训练。
- 最接近的先例：KF/EKF/UKF 生成式滤波；BPTT。
- 新颖性判断：**削弱。** 它把"Kalman 更新 = RNN cell"写成标题级主张（"reframing them as representationally equivalent deterministic models"）。目标公式若主张"隐状态由预测-修正递归更新"，这只是 RNN cell 设计，而这类 cell 已被明确命名并端到端训练过。
### 9. Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces
- 作者 / 年份 / 会议 / 链接：Philipp Becker, Harit Pandya, Gregor Gebhardt, Cheng Zhao, James Taylor, Gerhard Neumann / 2019 / ICML 2019, PMLR 97:544–552 / [PDF](http://proceedings.mlr.press/v97/becker19a/becker19a.pdf) **[全文]**
- 核心更新方程（原式照抄，编号为原论文编号）：
```
   预测:  z⁻_{t+1} = A_t z⁺_t ,      Σ⁻_{t+1} = A_t Σ⁺_t A_t' + I·σ_trans          (1)
   H = [I_m  0_{m×(n−m)}],  w_t = 编码器输出的潜在观测,  σ^obs_t 为其方差
   增益:  q^u_t = σ^{u,−}_t ⊘ (σ^{u,−}_t + σ^obs_t)                               (2)
          q^l_t = σ^{s,−}_t ⊘ (σ^{u,−}_t + σ^obs_t)                               (3)
   修正:  z⁺_t = z⁻_t + [q^u_t ; q^l_t] ⊙ [ w_t − z^{u,−}_t ;  w_t − z^{u,−}_t ]   (4)
   协方差: σ^{u,+}_t = (1_m − q^u_t) ⊙ σ^{u,−}_t, ...                              (5)-(7)
```
- 与目标公式异同（A–G）：
  - **A 有**，z⁻_{t+1}=A_t z⁺_t 就是 h⁻=Φ(h)，A_t 由网络生成（locally linear）。
  - **B 有，逐字同构**。(4) 里的 w_t − z^{u,−}_t 就是 z − D(h⁻)。
  - **C 有，逐字同构**。(4) 是纯加性。
  - **D 部分**。(4) 的 B 是 diag(q)·r，**q 只依赖方差 σ⁻ 与 σ_obs，不依赖 h⁻**；但 σ⁻ 是先验协方差的一部分，可视为 h⁻ 的扩展状态。**这是与目标公式最强的一处可比性：RKN 的 B = 对角增益 × 新息。**
  - **E 无 Δ**。A_t 由前一后验经 MLP 生成，没有时间间隔输入；离散步语义。
  - **F 有（强）**。**factorized**：协方差只保留 3m 个对角块；观测通道与记忆通道分离（p_t 存可观测部分、m_t 存速度这类只能靠时间推断的量）——"局部化/选择性"的现成实例。
  - **G 一步预测负对数似然**（高斯或伯努利），**直接优化，不用 ELBO**；摘要点名批评变分方法 "makes learning more complex and often less scalable due to approximation errors"。
- 任务与训练目标：高维观测（图像）的状态估计 + 预测 + 图像补全；直接似然训练。
- 最接近的先例：KF 1960；E2C/P2T (Watter 2015)；KVAE；DKF。论文原话："(Fraccaro et al., 2017) ... directly use the Kalman update equations in the latent state, which limits these approaches to rather low dimensional latent states"。
- 新颖性判断：**最强负面证据之一。** ICML 2019 的 (4) 式与目标公式第三行**数学上完全同构**，且"增益由不确定性决定、以新息为输入、加性修正"整套机制写清楚了。目标公式若要主张新颖性，只能靠 B 的函数形式更一般（吃 h⁻ 而不只是方差）——但这恰好是**更弱**的假设（丢掉最优性论证，见 §4 N2）。
### 10. KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics
- 作者 / 年份 / 期刊 / 链接：Guy Revach, Nir Shlezinger, Xiaoyong Ni, Adrià López Escoriza, Ruud J. G. van Sloun, Yonina C. Eldar / **2022 / IEEE Transactions on Signal Processing 70:1532–** / [期刊版 PDF](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf) · [arXiv:2107.10043](https://arxiv.org/abs/2107.10043) **[全文]**
  > **对 lead 任务书的更正**：任务书写 "Revach et al. KalmanNet (ICML 2022)"。实际是 **IEEE TSP vol. 70 (2022), pp. 1532–**（PDF 页眉逐字："1532 IEEE TRANSACTIONS ON SIGNAL PROCESSING, VOL. 70, 2022"；arXiv 注 "Accepted for publication in IEEE Transactions on Signal Processing"）。ICML 2022 论文集中没有这篇。
- 核心更新方程：
```
   EKF 流程保留:  x̂_{t|t−1} = f(x̂_{t−1|t−1}),   ŷ_{t|t−1} = h(x̂_{t|t−1})
   新息:          Δy_t = y_t − ŷ_{t|t−1}                                          (7)
   更新:          x̂_{t|t} = x̂_{t|t−1} + K_t Δy_t                          ← 与 (5a) 同形
   增益:          K_t 由 RNN 输出，**不**由 P C' R⁻¹ 计算
   RNN 输入特征:  F1 Δỹ_t = y_t − y_{t−1}             (观测差)
                  F2 Δy_t = y_t − ŷ_{t|t−1}           (**新息差**)
                  F3 Δx̃_t = x̂_{t|t} − x̂_{t−1|t−1}     (后验差)
                  F4 Δx̂_t = x̂_{t|t} − x̂_{t|t−1}       (更新量差)
```
  原文："The difference operation removes the predictable components, and thus the time series of differences is mostly affected by the noise statistics that we wish to learn."
- 与目标公式异同（A–G）：
  - **A 有**（x̂_{t|t−1}）。**B 有**（Δy_t 是显式观测空间新息）。**C 有**。
  - **D 有——最接近目标公式 B(h⁻,r) 的已发表实例**。RNN 输入同时含 F2（新息差）与 F3/F4（状态估计差），因此学到的增益**隐式依赖 (h⁻,r) 的历史**。
  - **E 无 Δ**。离散 t；**没有真实时间间隔输入**。
  - **F 有（弱）**。选哪几个特征（{F1,F2,F4} 或 {F1,F3,F4}）是选择性的，但结构本身是全状态稠密。
  - **G 监督学习（有真实状态标签）**。"The resulting system uses labeled data to learn to carry out Kalman filtering in a supervised manner."
- 任务与训练目标：部分已知动力学 + 未知噪声统计下的实时状态估计；监督、序列长度不变训练。
- 最接近的先例：EKF（被替换的基线）、RKN 2019、KVAE/DKF、作者自己的 model-based deep learning 系列。
- 新颖性判断：**削弱，且几乎逐字对应。** 它已把"gain ← 神经网络，输入含新息与状态差"做成完整方法并在 TSP 发表。与目标公式的差别只剩：(i) KalmanNet 保留 x̂⁻=f(x̂⁺) 的显式模型，目标公式连 Φ_Δ 也可学；(ii) KalmanNet 的增益网络是 RNN（有记忆），目标公式的 B **无记忆**——**这是劣势不是优势**：原文明确论证 "The recursive nature of the KG computation indicates that its learned module should involve an internal memory element"，因为增益本质上是二阶统计量的函数，二阶统计量需要记忆来跟踪。
### 11. Recursive KalmanNet: Deep Learning-Augmented Kalman Filtering for State Estimation with Consistent Uncertainty Quantification
- 作者 / 年份 / 会议 / 链接：Hassan Mortada, Cyril Falcon, Yanis Kahil, Mathéo Clavaud, Jean-Philippe Michel / 2025 / EUSIPCO 2025, arXiv:2506.11639 / [HTML](https://arxiv.org/html/2506.11639v1) **[全文]**
- 核心更新方程：
```
   K_t = P_{t|t−1} H_t' S_t⁻¹                                    (4)
   x̂_{t|t} = x̂_{t|t−1} + K_t ŷ_t                                (5a)
   P_{t|t} = (I − K_t H_t) P_{t|t−1}                             (5b)   ← 仅当 K 满足 (4) 才有效
   Joseph 形式（因为学出来的 K 不满足 (4)）：
   P_{t|t} = (I − K_t H_t) P_{t|t−1} (I − K_t H_t)' + K_t R_t K_t'      (6)
```
  原文："a learned RNN does not necessarily align with the analytical form in equation (4), potentially invalidating equation (5b) ... this inconsistency may result in a non-symmetric covariance matrix."
- 与目标公式异同（A–G）：
  - **A/B/C/D 同 KalmanNet**（就是 KalmanNet + 协方差递推）。**E 无 Δ**。**F 无**。**G 监督（状态 + 协方差）**。
- 任务与训练目标：在学增益的同时给出**一致**的不确定度量化。
- 最接近的先例：KalmanNet (2022)；Joseph (1968) 协方差更新式。
- 新颖性判断：**削弱，并附带一条对目标公式的直接打击。** 它用一句话证明：**一旦 B 是学出来的、不再等于 P C'S⁻¹，标准协方差代数就崩了**。目标公式连协方差都没有，所以连"崩"的资格都没有——它根本无法给出不确定度。若目标公式用于任何需要置信度的决策，"没有协方差"是硬缺陷而**不是简化**。
### 12. Recurrent Neural Networks and Universal Approximation of Bayesian Filters
- 作者 / 年份 / 会议 / 链接：Adrian N. Bishop, Edwin V. Bonilla / 2023 / AISTATS 2023, PMLR 206 / [PDF](https://proceedings.mlr.press/v206/bishop23a/bishop23a.pdf) **[全文]**
- 核心更新方程（原式照抄，编号为原论文编号）：
```
   线性高斯模型:  X_t = F X_{t−1} + V_t,   Y_t = H X_t + W_t                 (4.28)(4.29)
   真 KF 的 RNN 形式:
       X_t = E[X_t|Y_1..Y_t] = (F − K*_t H F) X_{t−1} + K*_t Y_t            (4.30)
       C_t = (I − K*_t H)(F C_{t−1} F' + Q)                                  (4.31)
       K*_t = (F C F' + Q)H' [R + H(F C F' + Q)H']⁻¹                         (4.32)
   泛函逼近: 存在 θ* 使 E‖ S_t − Ψ̂_{θ*}(s_{t−1}, Y_t) ‖ ≤ ε 对所有 t 成立       (4.33)
```
- 与目标公式异同（A–G）：
  - **A 有**（F X_{t−1} 项与 −K*HF X_{t−1} 项合并）。
  - **B 隐含**：展开 (4.30) 即 X_t = F X_{t−1} + K*_t(Y_t − H F X_{t−1})，新息形式。
  - **C 有**（展开后）。**D 无**（K*_t 由协方差算）。**E 无 Δ**。**F 无**。
  - **G 无训练目标（理论论文）**，但给出 RNN 的逼近误差界（含 time-uniform 界）。
- 任务与训练目标：证明"最优滤波器 = 某个 RNN"，给逼近界。
- 最接近的先例：Kalman 1960；Bain & Crisan 2008；早期神经网络滤波（Lo 1994; Parisini & Zoppoli 1994 等，论文自己列举）。
- 新颖性判断：**削弱，且是"理论上位"的一击。** 它把 **h_t = (F − K_t H F) h_{t−1} + K_t z_t**（即 (I−KH)A 与 K 合并）**逐字写成公式 (4.30)**，并证明这类结构是 RNN 的一个特例族。也就是说：**目标公式不是"把 Kalman 塞进网络"，而是"Kalman 本来就是一类 RNN"。** 所有基于"KF 与 RNN 结合"的新颖性主张都必须避开这条包含关系。

---
## 3. 学习式非线性观测器 / 神经状态估计器（3 篇）
### 13. Learning Robust State Observers using Neural ODEs
- 作者 / 年份 / 会议 / 链接：Keyan Miao, Konstantinos Gatsis / 2023 / L4DC 2023, PMLR 211:1–12 / [PDF](https://proceedings.mlr.press/v211/miao23a/miao23a.pdf) **[全文]**
- 核心更新方程（原式 (15) 照抄）：
```
   系统:        ẋ(t) = A x(t) + g(x(t)),     y(t) = C x(t)
   学习式观测器:  ẋ̂(t) = A x̂(t) + ĝ(x̂(t), θ) + G(y(t) − ŷ(t)),   ŷ(t) = C x̂(t)      (15)
                 G 取使 A − GC 为 Hurwitz；ĝ 为三层网络
   KKL 版本:     ż = D(z, θ) + F y,    x̂ 由 z 映射回
```
- 与目标公式异同（A–G）：
  - **A 有**。A x̂(t) + ĝ(x̂(t),θ) 是预测项（连续时间流）。
  - **B 有**。y − ŷ 是显式输出误差。
  - **C 有**。加性 + G(y−ŷ)。
  - **D 无**。G 是常值增益（保证收敛的 LMI 设计），不依赖 (x̂, 残差)；但 ĝ 依赖 x̂，严格说是"预测项非线性、增益线性"。
  - **E 有（连续时间）**。ẋ̂ 是时间导数、真实秒；但 G 不随 Δ 变——**这是与 CD-EKF 的关键差别**：没有"离散观测之间 ODE 传播 + 观测时刻离散修正"的两段结构，而把修正项直接塞进连续流（即 Simon 2006 连续形式）。
  - **F 无**。**G 数据驱动**（需要真实状态与输出测量）。
- 任务与训练目标：部分已知/完全未知非线性系统的 Luenberger 型与 KKL 型观测器；强调**收敛速度 vs 鲁棒性的可调权衡**。
- 最接近的先例：Luenberger 1964；KKL 观测器（Kazantzis & Kravaris 1998）；高增益观测器（Bornard & Hammouri 1991）；EKF（Reif & Unbehauen 1999）。原文："High-gain observers (HGO) ... show poor transient performance and a high sensitivity to noise, while EKF only guarantee local convergence."
- 新颖性判断：**削弱。** 这是"学习式 Luenberger 观测器"的现代版本，式 (15) 与目标公式在连续时间下同构度极高（预测项 + 输出误差 × 增益）。差别仅在于 Φ 是离散映射还是 ODE 流、以及 G 是否依赖残差。
### 14. An Adaptive Physics-Informed Neural Network Observer for state estimation in nonlinear dynamical systems
- 作者 / 年份 / 期刊 / 链接：Ayoub Farkane, Mohamed Boutayeb, Mustapha Oudani, Mounir Ghogho / 2025 / *Engineering Applications of Artificial Intelligence* / [出版页](https://www.sciencedirect.com/science/article/abs/pii/S0952197625026867) **[仅摘要级——HAL 开放版抓取返回空，公式未逐字核实]**
- 核心内容（摘要级）：提出 APINN-Obs，"we propose a novel Adaptive Physics-Informed Neural Network-based Observer (APINN-Obs) design that directly learns the optimal observer gain while ensuring convergence of the estimated ..."（摘要片段）。即：**增益由网络自适应学习，同时用收敛性约束保证误差动力学稳定**。
- 与目标公式异同（A–G，基于摘要，标注不确定）：
  - **A 有**（物理信息观测器必有模型预测项）。**B 有**（观测器必有输出误差项）。**C 有**（假定加性）。
  - **D [不确定]**：若 "learns the optimal observer gain" 指 G=G(·) 且依赖状态/残差，则成立；摘要不足以判定。
  - **E [不确定]**。**F 无**。**G 物理信息损失 + 收敛性约束**。
- 任务与训练目标：非线性系统状态估计，强调**收敛性证明**（与 KalmanNet 类方法的关键区别：要有稳定性保证）。
- 最接近的先例：Luenberger；PINN；KKL 观测器；LMI 型神经观测器（如 *LMI-based neural observer*, Int. J. Robust Nonlinear Control, 2024，[DOI](https://onlinelibrary.wiley.com/doi/10.1002/rnc.7327)）。
- 新颖性判断：**削弱。** 2025 年已有"学增益 + 保证收敛"的观测器设计。**给目标公式的教训**：观测器社区要求回答收敛性（e→0 的条件），而目标公式的 B(h⁻,r) 完全自由、连 B(h⁻,0)=0 都没写，无法证明任何收敛性。**这是最容易被审稿人抓的空洞。**
### 15. Mamba-2-Based Continuous-Discrete Extended Kalman Filter for Passive UAV Bearings-Only Tracking with Uncertain Measurement Noise
- 作者 / 年份 / 期刊 / 链接：Hao Wu, Guoxu Zeng 等 / 2026 / *Drones* 10(7):533（MDPI 开放获取）/ [HTML](https://www.mdpi.com/2504-446X/10/7/533) **[全文 HTML，但正文公式以图片渲染，符号定义未逐字核到]**
- 核心更新方程（文本级，公式为图片）：
```
   平方根 CD-EKF 时间更新: QR 分解预测状态与协方差（"the zero block"、"QR decomposition"）
   测量更新: H_k = ∂h/∂x;  K = P H' S⁻¹, S = 新息协方差;  x̂⁺ = x̂⁻ + K·ν, ν = 新息
   Mamba-2 模块: 从**新息序列**学习，直接预测**时变观测噪声方差 R**
```
  原文 Highlights："The proposed Mamba-2 block utilizes the state-space model that captures long-term dependencies in innovation sequences, enabling direct and accurate prediction of time-varying measurement noise variance."
- 与目标公式异同（A–G）：
  - **A 有**（CD-EKF 时间更新）。**B 有**（新息 ν）。**C 有**（x̂⁺=x̂⁻+Kν）。
  - **D 无**。K 由 CD-EKF 用 P,H,S 算；网络只学 R，**不改增益结构**。**注意这与目标公式正好相反**：目标公式让 B 自由而不管协方差；这篇让增益严格由协方差决定、只去学一个协方差参数 R。**后者在理论上更可辩护。**
  - **E 有（强）**。连续-离散：真实时间间隔进入 ODE 时间更新（同 §1.3）。
  - **F 有（弱）**。Mamba-2 的 S6 选择机制是输入依赖门控，算"选择性"。
  - **G 似然训练**（学 R 使滤波残差自洽）。
- 任务与训练目标：无源 UAV 纯方位跟踪，未知且时变的量测噪声。
- 最接近的先例：CD-EKF（Jazwinski 1970 / Simon 2006）；平方根滤波（array algorithm）；KalmanNet。
- 新颖性判断：**削弱"学习 + 连续-离散 + 新息"这个组合的新颖性。** 到 2026 年，这个三件套已是工程论文的标准配方，且只在"学 R"这一层动刀。**它反向证明了一个成熟做法：要学就学协方差参数，别去学增益本身**——学增益会毁掉统计一致性（见 §2 第 11 篇）。

---
## 4. 负面结果与反例
### N1（最致命）目标公式就是连续-离散 EKF 的教科书形式，连 Δ 的位置都相同
**证据链：**
- Kalman 1960 式 (20)(21)(22)：ỹ=y−Mx*、x*⁺=Φ*x*+Δ*y、Φ*=Φ−Δ*M。
- Jørgensen 2007 式 (2a)(3)(4)(5a)(6a)：ŷ=h(t_k,x̂⁻)、e=y−ŷ、K=PC'R⁻¹、x̂⁺=x̂⁻+Ke、dx̂/dt=f(t,x̂) 在 [t_k,t_{k+1}] 上积分。
- Simon 2006 记法（经 ICML 2021 补充材料逐字核实）：ẋ̂=f(x̂)+K[z−h(x̂)]。

**结论：** 目标公式三行 = CD-EKF 三行。Phi_Delta 是 ODE 从 t_k 到 t_{k+1} 的流映射，D 是 h(t_k,·) 或 C，B 是 K。**把它当贡献点等于重述 1960/1970 年代结论。** 可辩护的写法只有一种：明确承认这是 CD-EKF，然后把贡献限定在 B 的**学习方式**与 Φ_Δ 的**学习方式**。
### N2 B 不看协方差/精度，就失去全部最优性论证
**证据：**
- Kalman 1960 式 (28)：Δ*(t)=Φ P* M'(MP*M')⁻¹——**增益是误差协方差 P* 的函数，不是 (x*,ỹ) 的函数**。
- CD-EKF 式 (4)：K=P_{k|k−1}C'R⁻¹——依赖先验协方差与噪声协方差。
- RKN 式 (2)(3)：q=σ⁻⊘(σ⁻+σ_obs)——依赖先验方差与观测方差。
- KalmanNet 的立论就是"KG 封装了对噪声统计的依赖，所以用 RNN 去逼近它"，并把二阶矩隐式存在 RNN 记忆里。

**推论（本文推导，非引用）：** 若 B(h⁻,r) 是任意函数且不携带协方差（也无内部记忆估计二阶矩），则
1. 无法声称 h 是 MMSE/条件均值——最优性定理的前提（高斯 + 已知二阶矩）根本不成立；
2. 无法给出误差协方差——因此没有不确定性量化（见 N3、第 11 篇）；
3. 最坏情况下 h 只是"带反馈的确定性观测器"，属 Luenberger 类，收敛性需单独证明；而目标公式连 B(h⁻,0)=0 都没写，零新息时仍在改状态，**连观测器的基本前提都不满足**。

**可辩护的补丁只有两种**：(a) 把协方差（或其对角/低秩近似）作为 h 的一部分一起演化（RKN 的做法）；(b) 用有记忆的网络在内部估计二阶矩（KalmanNet 的做法）。目标公式两者都没有。
### N3 分解不唯一 / 增益缩放不可辨识
**(a) (Φ_Δ, B) 联合不可辨识（本文推导）。** 目标公式把状态演化写成"先预测、再用残差修"，但这个分解在函数层面不唯一：对任意可微 Ψ，令 Φ'_Δ(h)=Φ_Δ(h)+Ψ(h)、B'(h⁻,r)=B(h⁻,r)−Ψ(h⁻)，则整体更新 h⁺ 逐点完全相同。**只观测 h 的轨迹时，无法区分"预测错了"还是"增益补上了"。** 除非对 Φ_Δ 施加约束（例如它就是已知物理模型的离散化）或对 B 施加约束（例如 B(h⁻,0)=0 且 ∂B/∂r 有界/正定），否则两个函数**各自都不可辨识**。后果不是学理问题：训练时会出现大量不同分解在同一损失下等价，超参数与初始化决定落到哪个，泛化行为不可控。

**(b) 稳态增益对 (Q,R) 的整体缩放不变（本文推导）。** Kalman 稳态增益 K=P C'(CPC'+R)⁻¹ 在 (Q,R)→(αQ,αR) 下不变（DARE 在该变换下 P→αP，K 不变）。这意味着**从数据里学"增益"时噪声统计的绝对尺度不可辨识，只有比值有信息**；学出来的 B 在不同传感器/噪声水平下部署，等价类内的漂移是免费的。RKN 用"让网络输出 σ_obs、再解析算 q"的方式显式暴露了这一点：**要可辨识，就得把不确定性显式化。**

**(c) 新息白性检验（Kailath 1968）。** 最优滤波器的新息序列必须与过去观测不相关（白）。任意学出来的 B 生成的 r **不是**新息，只是残差。因此目标公式里 r 的命名会被追问：**能不能通过白性检验？不能的话，"新息"这个词就该删掉。**
### N4 不可观测子空间永不被修正
**证据：**
- 误差动力学：e_{k+1}=(A−LC)e_k（[UW ME547 讲义](https://faculty.washington.edu/chx/teaching/me547/2_5_observer_and_observerStateFeedback.pdf)）；Kalman 1960 式 (23) 给出同一形式 x̃(t+1|t)=Φ*(t+1;t)x̃(t|t−1)+u(t)，Φ*=Φ−Δ*M。
- 不可观测子空间定义：N_CA = ker O(C,A) = ker C ∩ ker CA ∩ … ∩ ker CA^{n−1}（[ASU Lecture 06](https://control.asu.edu/Classes/MAE598/598Lecture06.pdf) Def. 5）。

**推论（标准结论 + 本文推导）：** N_CA 在 A 下不变且 N_CA ⊆ ker C，因此 (A−LC)N_CA ⊆ N_CA，且 A−LC 在 N_CA 上的谱与 A 在 N_CA 上相同。**任何输出注入式修正都无法在该子空间上产生衰减**——误差初值若落在不可观测方向，永远不被修正。

**对目标公式的直接后果：** 只要 B(h⁻,r) 通过 r=z−D(h⁻) 进入（"输出注入"），**能量落在 ker ∂D/∂h 方向上的误差分量不可达**。这是结构性的，与 B 多强、训练数据多少无关。目标公式对此没有任何声明，因此不能声称它是"通用在线状态估计结构"，除非加上可观测性假设。
### N5 EKF 在强非线性下失效，目标公式继承其全部失效模式
**证据：**
- Jørgensen 2007 §III 是专门的 critical discussion：即使数值上稳健（ESDIRK + 平方根），估计质量与收敛速度仍受传感器配置根本限制（"These limitations also limits the resulting closed-loop performance achievable by any controller including a nonlinear predictive controller."）。
- KalmanNet（IEEE TSP 2022）引言："When the model is highly non-linear, the local linearity approximation may not hold, and the EKF can result in degraded performance."
- Miao & Gatsis 2023："EKF only guarantee local convergence"；HGO "show poor transient performance and a high sensitivity to noise"。

**对目标公式的后果：** 若 D 是非线性解码（图像/特征），r=z−D(h⁻) 在 h⁻ 远离真值时会给出**误导性方向**（雅可比不准确的经典失效）。目标公式把 D 当可学映射，但**没有处理"解码映射的局部线性化不可信"这一层**：r 的几何意义在非线性 D 下并不成立（残差大的方向不一定是要修的方向）。**这是所有"新息式"方法在感知任务上的共同软肋**，也是 DKF/DVBF/KVAE 那一支选择"不显式算残差、直接学后验"的根本原因。
### N6 Δ 语义已被连续-离散滤波彻底解决，"Δ 进入 Φ"不可作为贡献
**证据：** CD-EKF 式 (6a)(6b) 把 Δ=t_{k+1}−t_k 完整编码进 ODE 积分区间；高阶积分器 + 平方根算法是标准做法（Jørgensen 2007 全文）。同构的连续时间形式 ẋ̂=f(x̂)+K[z−h(x̂)]（Simon 2006）把修正写成连续流，Δ 隐含在测量到达时刻。

**可能剩下的空隙（唯一值得写的）：** 若 Δ 本身参与**增益**计算（例如 B(h⁻,r,Δ)，且 Δ→0 时 B→0 或退化为连续修正），那是**异步/平流观测**方向的东西，而不是"Δ 进入转移"。注意这条空隙与 CfC/timespan 专题重叠，必须交叉核对，不能各写各的。
### N7 学习式增益的已知失效模式清单
| 失效模式 | 来源 | 具体表现 |
|---|---|---|
| 协方差代数失效 | Recursive KalmanNet (2025) | 学出的 K 不满足 K=PC'S⁻¹，(I−KH)P⁻ 产生非对称协方差，必须退回 Joseph 形式 |
| 需要真实状态标签 | KalmanNet (2022) 原文 | "uses labeled data ... in a supervised manner"；无标签的在线感知里不可用 |
| 训练/推理分布漂移 | KalmanNet, RKN | 增益网络只见过训练噪声水平；N3(b) 的尺度不可辨识使漂移不可察觉 |
| 对角/因子化近似失真 | RKN 自己承认 | "This might be a crude approximation for many systems"（协方差只保留对角块） |
| 长序列误差累积 | Backprop KF / RKN | BPTT 训练，长程稳定性无保证（Bishop & Bonilla 2023 才给出 time-uniform 界） |
| 不可观测方向 | 观测器理论（N4） | 与学习无关的结构性下界 |

---
## 5. 三问汇总表
| # | 文献 | Q1 新息修正 | Q2 时间语义 / Δ 位置 | Q3 用途 | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Kalman 1960 | 是 | 索引（Δ=1），(j) 讨论连续极限 | 通用估计 | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | — |
| 2 | Luenberger 1964 | 是 | 连续 t，无 Δ | 通用观测器 | ✔ | ✔ | ✔ | ✘ | ~ | ~ | — |
| 3 | CD-EKF (Jazwinski/Simon/Jørgensen) | 是 | **秒；Δ 进 ODE 流映射** | 通用估计 | ✔ | ✔ | ✔ | ✘ | **✔** | ✘ | — |
| 4 | Kailath 1968 innovations | 是（概念） | 无 | 理论 | ~ | ~ | ~ | ✘ | ✘ | ✘ | — |
| 5 | DKF 2015 | 隐式（变分后验） | 索引 | 生成/反事实 | ~ | ✘ | ✘ | — | ✘ | ✘ | ELBO |
| 6 | DVBF 2017 | 否 | 索引 | 生成 + MBRL | ~ | ✘ | ✘ | — | ✘ | ✘ | ELBO |
| 7 | KVAE 2017 | 是（LGSSM 因子内、精确） | 索引 | 生成/想象/补全 | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | ELBO |
| 8 | Backprop KF 2016 | 是 | 索引 | 视觉状态估计 | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | 一步预测（判别） |
| 9 | **RKN 2019** | **是（逐字同构）** | 索引 | 估计+预测+补全 | ✔ | ✔ | ✔ | ~（q 依赖方差） | ✘ | **✔** | 一步预测 NLL |
| 10 | **KalmanNet 2022** | **是** | 索引 | 实时估计 | ✔ | ✔ | ✔ | **✔**（经 RNN 特征） | ✘ | ~ | 监督 |
| 11 | Recursive KalmanNet 2025 | 是 | 索引 | 估计 + 不确定度 | ✔ | ✔ | ✔ | ✔ | ✘ | ✘ | 监督 |
| 12 | Bishop & Bonilla 2023 | 是（展开后） | 索引 | 理论 | ✔ | ✔ | ✔ | ✘ | ✘ | ✘ | — |
| 13 | Miao & Gatsis 2023 | 是（连续形式） | **秒**（连续流），G 不随 Δ | 通用观测器 | ✔ | ✔ | ✔ | ✘ | ~ | ✘ | 数据驱动 |
| 14 | Farkane et al. 2025 | 是 | 未核实 | 通用观测器 | ✔ | ✔ | ✔ | ? | ? | ✘ | PINN + 收敛约束 |
| 15 | Wu et al. 2026 (Mamba-2 CD-EKF) | 是 | **秒**（CD-EKF） | 跟踪 | ✔ | ✔ | ✔ | ✘（网络只学 R） | **✔** | ~ | 似然 |

> 注：✔ 明确成立；~ 部分/需解释；✘ 不成立；— 不适用。
> **A/B/C 三列在 15 篇里 13 篇为 ✔，说明目标公式的前三行是整个领域的公共骨架，不具备区分度。真正的区分度只可能在 D/E/F/G 四列，而目标公式在 D 上弱于 KalmanNet（无记忆）、在 F 上完全没有、在 G 上未指定。**

---
## 6. 与目标公式逐字对应的已有工作（直接回答"有没有人做过"）
**有，至少三处逐字同构：**

1. **Kalman 1960 式 (20)–(22)** —— h = h⁻ + K(z − D h⁻)，符号换名即可。**数学上完全等价，无任何余地。**
2. **Becker et al., ICML 2019, 式 (4)** —— z⁺ = z⁻ + q⊙(w − Hz⁻)，q 由方差算、w−Hz⁻ 是观测空间新息、更新是加性。**这是"神经网络实现的目标公式"，发表在 ICML。**
3. **Miao & Gatsis, L4DC 2023, 式 (15)** —— ẋ̂ = Ax̂ + ĝ(x̂,θ) + G(y − Cx̂)。**这是"学习式目标公式"的连续时间版。**

**接近但不等价：**
- **KalmanNet 2022**：增益是神经网络，输入含新息与状态差 → 命中最难命中的一轴（B 依赖 h⁻ 与 r），但增益有 RNN 记忆且需标签。
- **RKN 2019**：结构最像，但增益是解析的（方差比），记忆在协方差里。
- **Bishop & Bonilla 2023**：证明 (F−KHF) 与 K 就是 RNN 的权重，把"KF 是 RNN"变成定理。

**没有任何一篇**把 B 当成"完全自由的 (h⁻,r) 映射、无协方差、无记忆、无稳定性约束"——**但这是缺陷清单，不是新颖性空间**：去掉这些约束后，结构退化成 1964 年的 Luenberger 观测器，且失去全部理论保证。

---
## 7. 可辩护的剩余空间（本报告判断，供 lead 决策）
1. **Δ 参与增益**：现有 CD-EKF/学习式观测器里 Δ 只进预测，**没有一篇让 B 显式吃 Δ 并在 Δ→0 时退化为连续修正**。这是唯一能看到的结构性空隙（需与 CfC/timespan 专题交叉核对，避免撞车）。
2. **B 的可辨识性约束**：把 B(h⁻,0)=0、∂B/∂r ≻ 0、以及"增益由显式不确定性变量参数化"写成硬约束，可以在理论上把目标公式从"任意 feedback"钉成"观测器"。**贡献在约束与证明，不在方程形式。**
3. **不可观测子空间的显式处理**：现有学习式滤波器无人显式建模 ker O(C,A)；若 F 轴（稀疏/选择性）能落在这里（只修正可观测子空间、其余交给 Φ 外推），那是有内容的。

**明确不可主张的：** "预测-新息-修正"三行结构、Δ 进转移、增益网络化、用残差训练——分别对应 Kalman 1960、Jazwinski 1970、KalmanNet 2022、Backprop KF 2016。

---
## 8. 核实状态清单
| # | 文献 | 核实级别 | 卡在哪 |
|---|---|---|---|
| 1 | Kalman 1960 | 全文 | — |
| 2 | Luenberger 1964 | 原文未取到；公式据 UW 讲义；书目据 SCIRP 记录 | IEEE 付费墙 |
| 3 | Jørgensen 2007 | 全文 | 两本书（Jazwinski/Simon）未取到，书目经他人引用交叉核实 |
| 4 | Kailath 1968 | 仅摘要级 | IEEE 付费墙；摘要来自 ADS 记录的检索片段 |
| 5 | DKF 2015 | 全文（HTML） | 无正式会议版（arXiv only），已在文中注明 |
| 6 | DVBF 2017 | 全文（HTML） | 初次检索的 arXiv id 1605.08104 是错的（实为 Deep Predictive Coding），已改为 1605.06432 |
| 7 | KVAE 2017 | 全文（PDF） | 注意：arXiv:1605.07571 是**前身 SRNN**，不是 KVAE；KVAE 正式版只在 NeurIPS 论文集 |
| 8 | Backprop KF 2016 | 全文（PDF） | — |
| 9 | RKN 2019 | 全文（PDF） | — |
| 10 | KalmanNet | 全文（期刊版 PDF） | **任务书 "ICML 2022" 有误**，实为 IEEE TSP 70:2022 |
| 11 | Recursive KalmanNet 2025 | 全文（HTML） | — |
| 12 | Bishop & Bonilla 2023 | 全文（PDF） | 合作者是 **Bonilla**，任务书的 "Del Moral" 未在该论文中出现 |
| 13 | Miao & Gatsis 2023 | 全文（PDF） | — |
| 14 | Farkane et al. 2025 | 仅摘要级 | HAL 开放版抓取返回空；作者四人已核（Farkane, Boutayeb, Oudani, Ghogho） |
| 15 | Wu et al. 2026 | 全文（HTML），公式为图片 | 符号定义未能逐字核实；作者只核到前两位（Hao Wu, Guoxu Zeng） |

**给 lead 的三处更正：**
1. KalmanNet 不是 ICML 2022，是 **IEEE TSP 70 (2022), 1532–**。
2. Bishop 2023 的合作者是 **Edwin V. Bonilla**，不是 Del Moral。
3. KVAE 的 arXiv 号与 SRNN 不同；若报告引 arXiv:1605.07571，引到的是 SRNN（2016），不是 KVAE（2017）。
