# 世界模型与潜状态预测（model-based RL）—— 与候选结构的逐条对照

被测公式（用户提供，ASCII）：

    h- = Phi_Delta(h)            # 带真实时间间隔 Delta 的先验/预测
    r  = z - D(h-)               # 观测空间残差/新息
    h  = h- + B(h-, r)           # 残差式修正（学习式增益）

- 检索人：wm-latent（文献调研子代理）；检索日期 2026-11-03。
- 主工具 tavily_search / tavily_extract（全部条目均拉取 PDF 或 arXiv HTML 正文核对）；元数据用 export.arxiv.org API 二次核对；web_fetch 仅作补充。
- 覆盖 12 篇，全部正文级；**无 [未核实] 条目**。唯一 [仅摘要级] 的是第 6 节引用的 2605.01694（只用于一句分类学断言）。
- 判定口径：A–G 只描述"公式层面是否出现"，不评价效果好坏；支持/削弱新颖性必须给原文证据。

---

## 0. 先给结论（负面优先）

**0.1 与目标公式逐字级对应的已有工作：有，而且是两篇分开占据的。**

1. **第 2、3 行**（$r=z-D(h^-)$ 与 $h=h^-+B(h^-,r)$）在 2019 年的 Recurrent Kalman Network 里有逐字对应：先验 $z^-_{t}$，新息 $w_t-z^{u,-}_t$，加性修正 $z^+_t=z^-_t+Q_t\odot r_t$，而且"观测缺失时把后验设成先验"都写进了正文（ICML 2019, PMLR 97）。
2. **第 1 行**（$h^-=\Phi_\Delta(h)$）在 2025 年的 Time-Aware World Model 里有逐字对应：$\hat z_{t+\Delta t}=z_t+d(z_t,a_t,\Delta t)\cdot\tau(\Delta t)$，$\Delta t$ 以**秒**为单位、按 log-uniform 采样训练，基线就是 TD-MPC2（ICML 2025, arXiv:2506.08441）。
3. 因此：**七轴里没有任何一轴是空的**。RKN 占了 A/B/C/D，TAWM 占了 E，PlaNet–Dreamer 家族占了"后验=滤波"的表述与 KL 训练机制。

**0.2 三条不能回避的事实**

- RSSM 的后验**不是**"从预测状态出发用观测残差修正"，而是"用先验确定状态 $h_t$ 与当前观测编码联合**重新推断**"；PlaNet 自己把这件事叫作 *filtering posterior*（原文："We use the filtering posterior that conditions on past observations"）。所以"用观测修正先验状态"这一语义本身是 2018–2019 的公共财产。
- 世界模型家族里确实**没有人**写 $r=z-D(h^-)$，但这**不是没想到**：解码器式世界模型把修正放在 KL/重编码路径里（$q(z_t|h_t,x_t)\parallel p(z_t|h_t)$），decoder-free 的世界模型（TD-MPC 系）根本没有 $D$，残差只能在潜空间对"下一步编码"定义。换言之，写不写成观测残差，是**表述选择**，不是机制差异。
- $\Delta$ 已经进入过 latent MBRL（TAWM 2025）和连续时间 MBRL（CT-MBRL 2021），且两篇都明确论证了"无 $\Delta$ 的采样率依赖"是缺陷。所以"把 $\Delta$ 放进状态转移"并不是新发现。

**0.3 还剩什么可以主张（给 lead 的建议）**

- 把 $\Phi_\Delta$、$r=z-D(h^-)$、$h=h^-+B(h^-,r)$ **三者放进同一个潜状态世界模型并同时用 RL 目标训练**：本次检索没找到完全同构的一篇。但 TAWM 已经把 $\Delta$ 放进 TD-MPC2，RKN 已经把残差修正放进潜状态滤波——把两者拼起来属于**直接组合**，不是新机制。
- 真正站得住的新颖性只能来自： (i) 对 $B$ 的**结构性约束**（稀疏/选择性/局部化，见轴 F 全空）； (ii) 关于新息修正如何改变**多步误差传播**的可证明命题； (iii) 在**真实不规则采样**（抖动、丢帧）下相对 RSSM 家族的定量优势。若只主张"结构新"，会被 RKN+TAWM 直接打掉。

---

## 1. 七轴速查（谁最早占了哪一格）

| 轴 | 判据 | 最早/最直接先例 | 状态 |
|---|---|---|---|
| A 显式先验 $h^-$ | 有独立的预测状态，且它先于观测产生 | PlaNet 2019（$h_t$）；RKN 2019（$z^-_t$） | 已被占据 |
| B 观测空间残差 $r=z-D(h^-)$ | 在观测（或观测编码）空间显式做差 | RKN 2019（$w_t-Hz^-_t$） | 已被占据（潜观测空间） |
| C 加性修正 $h=h^-+B$ | 后验 = 先验 + 修正项 | RKN 2019；DVBF 2017（$C_tw_t$ 加在转移上） | 已被占据 |
| D $B$ 同时依赖 $(h^-,r)$ | 增益由先验不确定度与残差共同决定 | RKN 2019（增益 = 预测方差与观测噪声之比，乘在 $r$ 上） | 已被占据 |
| E 真实 $\Delta$（秒） | 秒进入转移，训练覆盖多采样率 | CT-MBRL 2021（状态空间）；TAWM 2025（**潜空间**） | 已被占据 |
| F 稀疏/选择性/局部化 | 修正只作用于状态的一部分子集 | 无（RKN 的逐维增益算弱形式） | **空** |
| G 训练目标 | 重构 / ELBO / 一步预测 / RL 回报 / 对比 / 自监督 | 全部都有 | 不构成新颖性 |

---

## 2. 逐篇核对（按年代）

### 1. Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data
- 作者 / 年份 / 会议或期刊 / 链接：Maximilian Karl, Maximilian Soelch, Justin Bayer, Patrick van der Smagt / 2016（arXiv）/ ICLR 2017 / https://arxiv.org/pdf/1605.06432
- 核心更新方程：转移是**确定性**的 $z_{t+1}=f(z_t,u_t,\beta_t)$；$\beta_t=(w_t,v_t)$ 由识别网络给出，$w_t\sim q_\phi(w_t\mid z_t,x_{t+1},u_t)$。3.3 节的局部线性实例：
  $$(A,B,C)_t=\sum_{i=1}^{M}\alpha^{(i)}_t(A,B,C)^{(i)},\quad z_{t+1}=A_tz_t+B_tu_t+C_tw_t$$
  目标为 ELBO（式 11）：$\mathcal L=\mathbb E_{q}[\ln p_\theta(x_{1:T}\mid z_{1:T})]-\mathrm{KL}(q_\phi(\beta_{1:T}\mid x_{1:T},u_{1:T})\parallel p(\beta_{1:T}))$。
- 三问：Q1 **是**——$t+1$ 的状态被 $x_{t+1}$ 显式修正（原文："the recognition model as a **filter**"）；Q2 **否**，离散定步长，全文无 $\Delta t$/秒；Q3 **不是规划**，是通用潜变量状态估计/系统辨识（摆、弹球）。
- 与目标公式异同：
  - A **部分有**：$z_t$ 就是 $h^-$，但它和修正后的状态没有分成"先验量/后验量"两套。
  - B **无**：$w_t$ 不是 $z-D(h^-)$，是生成过程里的潜空间随机偏移；没有解码器回投。
  - C **有**：$C_tw_t$ 是加在 $A_tz_t$ 上的加性项，形态与 $h=h^-+B$ 同类。
  - D **弱**：$w_t$ 由 $(z_t,x_{t+1},u_t)$ 推断，同时吃"先验"和"下一帧观测"，但不是对残差的函数。
  - E **无**。
  - F **无**：$w_t$ 是无结构过程噪声。
  - G **ELBO**（重构 + KL）。
- 任务与训练目标：从原始像素学状态空间模型，ELBO + BPTT；附录实验为摆与弹球。
- 最接近的先例：自引 Deep Kalman Filters（Krishnan 2015）；PlaNet/Dreamer 都把 DVBF 列为潜序列模型先例。
- 新颖性判断：**削弱**。"用观测修正预测"有 2017 年的潜变量版本。更重要的是它自己写下了目标公式最危险的退化模式：原文承认"the learning algorithm could still learn the transition as $z_{t+1}=w_t$"——即修正项可以吸收一切，动力学退化成恒等映射（见 4.4）。

### 2. World Models
- 作者 / 年份 / 会议或期刊 / 链接：David Ha, Jürgen Schmidhuber / 2018 / arXiv:1803.10122（技术报告版，正文已抓取：https://arxiv.org/pdf/1803.10122 ；NeurIPS 2018 版题名为 *Recurrent World Models Facilitate Policy Evolution*，https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution ）
- 核心更新方程：三段式 $V,M,C$。
  $$z_t=V(x_t)\ (\text{VAE, }z\sim\mathcal N(\mu,\sigma)),\qquad P(z_{t+1}\mid a_t,z_t,h_t)\ \text{(MDN 混合高斯)},\qquad a_t=W_c[z_t;h_t]+b_c$$
  推理伪代码：z=vae.encode(obs); a=controller([z,h]); obs=env.step(a); h=rnn.forward([a,z,h])。
- 三问：Q1 **部分**——预测误差只以 $-\ln P(z_{t+1}\mid\cdot)$ 进入训练，推理时不存在"修正状态"的动作；Q2 **否**，固定帧率，全文只有"2100 步≈60 秒"这类任务时长；Q3 **是给规划/想象服务的**——M 被包装成 gym.Env，控制器在"梦"里训练。
- 与目标公式异同：
  - A **半有**：$h_t$ 是 RNN 隐状态，$P(z_{t+1}\mid a_t,z_t,h_t)$ 是显式一步预测；但没有"先验状态/后验状态"的二分。
  - B **无显式**：观测残差从不被计算为向量，只在似然里。
  - C **无显式加性**：观测 $z_t$ 作为 RNN 输入被混进 $h_{t+1}$，是"输入"，不是"残差"。
  - D **无**。
  - E **无**。
  - F **无**；温度参数 $\tau$ 是全局采样旋钮，不是局部化结构。
  - G **一步预测（MDN 似然）+ VAE 重构**；控制器侧是 RL 回报（CMA-ES）。
- 任务与训练目标：CarRacing / VizDoom；M 额外预测 done 标志以构成可训练环境。
- 最接近的先例：Schmidhuber 1990 年代 RNN 世界模型系列（文中自述）；MDN（Bishop 1994）。
- 新颖性判断：**削弱"世界模型=预测+想象"框架**的新颖性；但对 A–D 几乎不构成威胁，因为它连显式先验都没有。反过来它也说明：这个领域的"标准答案"从一开始就是**预测未来 + 在预测里规划**，而不是"在线状态估计"。

### 3. Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces
- 作者 / 年份 / 会议或期刊 / 链接：Philipp Becker, Harit Pandya, Gregor Gebhardt, Cheng Zhao, James Taylor, Gerhard Neumann / 2019 / ICML 2019（PMLR 97）/ https://proceedings.mlr.press/v97/becker19a/becker19a.pdf ；arXiv:1905.07357
- 核心更新方程（潜空间学习式卡尔曼滤波，$H=[I_m\ 0_{m\times(n-m)}]$）：
  $$z^-_{t+1}=A_tz^+_t,\qquad \Sigma^-_{t+1}=A_t\Sigma^+_tA_t^\top+I\sigma_{\text{trans}}$$
  $$q^u_t=\sigma^{u,-}_t\oslash(\sigma^{u,-}_t+\sigma^{\text{obs}}_t),\qquad q^l_t=\sigma^{s,-}_t\oslash(\sigma^{u,-}_t+\sigma^{\text{obs}}_t)$$
  $$z^+_t=z^-_t+\begin{bmatrix}q^u_t\\ q^l_t\end{bmatrix}\odot\begin{bmatrix}w_t-z^{u,-}_t\\ w_t-z^{u,-}_t\end{bmatrix}$$
  其中 $w_t$（及观测噪声 $\sigma^{\text{obs}}_t$）由编码器从观测给出，$A_t$ 由网络从潜状态给出。
- 三问：Q1 **是**，而且是显式新息；Q2 **否**，定步长离散递归，无 $\Delta$；Q3 **不是规划**：通用在线状态估计（图像摆的滤波、KITTI 位姿、图像插补），训练目标是观测/图像似然。
- 与目标公式异同：
  - A **有**：$z^-$ 是严格意义的先验。
  - B **有**：$w_t-z^{u,-}_t$ 就是新息；严格说是"编码后的潜观测空间"残差（$H$ 线性选维），不是像素空间。
  - C **有**：加性，形态完全一致。
  - D **有**：增益 $q$ 由先验不确定度 $\sigma^-$（$h^-$ 的函数）与学到的观测噪声共同决定，并乘在 $r$ 上。
  - E **无**。
  - F **无**（因子化协方差给出逐维增益，算局部化的弱形式）。
  - G **观测/图像重构 log-likelihood + BPTT**，非 RL。
- 任务与训练目标：高维特征的在线滤波；与 LSTM/GRU 对比；有"缺测"实验。
- 最接近的先例：经典 Kalman filter（Kalman 1960）；本文贡献是把增益与协方差做成可学、可因子化。
- 新颖性判断：**最强削弱**。原文还有一句直接对应目标公式的边界条件："the RKN provides a principled method to deal with absent inputs by just **omitting the update step and setting the posterior to the prior**"——这正是 $h=h^-$（无观测时）的规范化版本。目标公式等价于"把 RKN 的线性 $H$ 换成非线性解码器 $D$、把解析增益换成神经网络 $B$"。

### 4. Learning Latent Dynamics for Planning from Pixels (PlaNet)
- 作者 / 年份 / 会议或期刊 / 链接：Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson / 2019 / ICML 2019（PMLR 97）/ https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf ；arXiv:1811.04551
- 核心更新方程（RSSM，式 4）：
  $$h_t=f(h_{t-1},s_{t-1},a_{t-1}),\quad s_t\sim p(s_t\mid h_t),\quad o_t\sim p(o_t\mid h_t,s_t),\quad r_t\sim p(r_t\mid h_t,s_t)$$
  编码器 $q(s_{1:T}\mid o_{1:T},a_{1:T})=\prod_t q(s_t\mid h_t,o_t)$；ELBO（式 3）：$\sum_t\{\mathbb E_q[\ln p(o_t\mid s_t)]-\mathbb E\,\mathrm{KL}[q(s_t\mid o_{\le t})\parallel p(s_t\mid s_{t-1})]\}$；latent overshooting（式 5–6）训练多步先验。
- 三问：Q1 **部分**——每一步都用当前观测重新推断 $s_t$，但形式是重采样而非残差；Q2 **否**，离散下标 $t$，唯一的"时间"旋钮是 action repeat $R$；Q3 **是规划**（CEM/MPC 在潜空间滚动），观测模型只做训练信号。
- 与目标公式异同：
  - A **有**：$h_t$ 是确定性的先验路径。
  - B **无**：误差在 KL 里，不在观测空间；重构项是似然，不构成 $r$。
  - C **无**：后验是独立编码器（CNN + 接 $h_t$ 的 MLP），不是加法。
  - D **无**。
  - E **无**。
  - F **无**。
  - G **ELBO + 重构 + 奖励/终止似然 + latent overshooting**。
- 任务与训练目标：DeepMind Control Suite 从像素；MPC 规划。
- 最接近的先例：自引 Buesing 2018、Chung 2015（VRNN）、DVBF、DKF。
- 新颖性判断：**削弱**。原文明确写："We use the **filtering posterior** that conditions on past observations since we are ultimately interested in using the model for planning, but one may also use the full smoothing posterior during training."——即"后验=在线滤波"是 2019 年的官方框架语汇；"用观测修正预测"这一层意思早已被占用，只是没有写成 $h=h^-+B$。

### 5. Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)
- 作者 / 年份 / 会议或期刊 / 链接：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi / 2020 / ICLR 2020 / https://arxiv.org/pdf/1912.01603
- 核心更新方程（正文式 1，注意此版 p/q 记法相反：p 表示环境/表示模型，q 表示可想象的近似）：
  $$\text{Representation: }p(s_t\mid s_{t-1},a_{t-1},o_t),\quad \text{Transition: }q(s_t\mid s_{t-1},a_{t-1}),\quad \text{Reward: }q(r_t\mid s_t)$$
  4.2 节交代实际实现："We implement the transition model as a **recurrent state space model (RSSM)**; the representation model by combining the RSSM with a CNN"——即正文是 SSM 写法、实做是 RSSM；潜变量为 30 维对角高斯。
- 三问：Q1 **部分**（表示模型吃当前观测，属重推断）；Q2 **否**；Q3 **是规划/想象**（在潜空间用 $V_\lambda$ 做演员-评论家）。
- 与目标公式异同：
  - A **有**（实做 RSSM 的 $h_t$；纯 SSM 写法下没有显式先验路径）。
  - B **无**。
  - C **无**。
  - D **无**。
  - E **无**。
  - F **无**。
  - G **ELBO**（$\beta=1$，KL 截断在 3 free nats）+ 想象回报。
- 任务与训练目标：DMC 20 个视觉任务；想象内 actor-critic，解析梯度回传。
- 最接近的先例：PlaNet（同组）、PILCO/MPC 系、Ha & Schmidhuber。
- 新颖性判断：**削弱**。原文："The model **mimics a non-linear Kalman filter** (Kalman, 1960), latent state space model, or HMM with real-valued states."——作者自己把潜状态世界模型等同于非线性卡尔曼滤波。这句可以直接引用来判定"预测-修正"结构的先验归属。

### 6. Mastering Atari with Discrete World Models (DreamerV2)
- 作者 / 年份 / 会议或期刊 / 链接：Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, Jimmy Ba / 2021 / ICLR 2021 / https://arxiv.org/pdf/2010.02193
- 核心更新方程（式 1）：
  $$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\phi(z_t\mid h_t,x_t),\quad \hat z_t\sim p_\phi(\hat z_t\mid h_t)$$
  图像/奖励/折扣预测器吃 $(h_t,z_t)$；损失（式 2）$=\sum_t[-\ln p(x_t\mid h_t,z_t)-\ln p(r_t\mid h_t,z_t)-\ln p(\gamma_t\mid h_t,z_t)+\beta\,\mathrm{KL}(q(z_t\mid h_t,x_t)\parallel p(z_t\mid h_t))]$，$\beta=0.1$（Atari）/1.0（连续控制）；随机状态是 32 个 categorical、每类 32 类（straight-through）。
- 三问：Q1 **部分**——KL 把先验拉向后验，但状态从未被残差加性修正；Q2 **否**；Q3 **是想象训练**。
- 与目标公式异同：
  - A **有**；B **无**；C **无**；D **无**；E **无**；F **无**。
  - G **ELBO 变体**，且是本清单里第一次出现"非对称 KL 加权"（KL balancing）。
- 任务与训练目标：Atari 26 游戏；KL balancing $\alpha=0.8$ 给先验、$1-\alpha$ 给后验。
- 最接近的先例：PlaNet/DreamerV1（自引）；VRNN 系。
- 新颖性判断：**削弱**，且提供关键反证据。原文动机："learning the transition function is difficult and we want to **avoid regularizing the representations toward a poorly trained prior**"——这句话等于承认：$(q\parallel p)$ 的单向耦合在训练早期会破坏表征。若目标公式要主张"预测状态被观测修正"更优，必须先解释为什么这个 2021 年就被认定为有害的耦合在残差形式下会变好。

### 7. Continuous-Time Model-Based Reinforcement Learning
- 作者 / 年份 / 会议或期刊 / 链接：Çağatay Yıldız, Markus Heinonen, Harri Lähdesmäki / 2021 / ICML 2021（PMLR 139）/ https://arxiv.org/pdf/2102.04764
- 核心更新方程（式 4、式 7）：
  $$s_{i+1}=s_i+f(s_i,a_i,\Delta t_i),\qquad \Delta t_i=t_{i+1}-t_i$$
  $$\hat s_i\mid(s_0,a_{0:t})=s_0+\int_0^{t_i}f_\theta(\hat s_\tau,a_\tau)\,d\tau$$
  背景为贝叶斯神经 ODE，$\Delta t$ 由观测时刻直接给出，允许不规则到达。
- 三问：Q1 **否**（状态本身可观测，无需新息修正）；Q2 **是**，秒级 $\Delta t$ 显式进入转移；Q3 **是规划/想象**（在 ODE 上做 CT actor-critic）。
- 与目标公式异同：
  - A **有**（$s_i$ 即先验）；B **无**（不需解码器）；C **加性但非残差**（是 Euler 积分项，不是观测残差）；D **不适用**；E **有**；F **无**；G **轨迹 ELBO + RL**。
- 任务与训练目标：新造的 ODE-RL 套件；贝叶斯神经 ODE + 连续时间 actor-critic。
- 最接近的先例：Neural ODE（Chen 2018）；CT RL（Bradtke & Duff 1994, Doya 2000）。
- 新颖性判断：**直接削弱轴 E**。原文给出了目标公式支持者最喜欢引用的论证："If the observations arrive non-uniformly, i.e., $\Delta t_i\neq\Delta t_j$, then the next states would be different ... resulting data triplets have the same inputs $(s,a)$ and different targets, which would lead to training issues."（3.1 节）——这个论证是 2021 年公开的，不能当新观察。

### 8. Temporal Difference Learning for Model Predictive Control (TD-MPC)
- 作者 / 年份 / 会议或期刊 / 链接：Nicklas Hansen, Xiaolong Wang, Hao Su / 2022 / ICML 2022（PMLR 162）/ https://arxiv.org/pdf/2203.04955 ；arXiv:2203.04955
- 核心更新方程（式 6、式 10）：
  $$z_t=h_\theta(s_t),\quad z_{t+1}=d_\theta(z_t,a_t),\quad \hat r_t=R_\theta(z_t,a_t),\quad \hat q_t=Q_\theta(z_t,a_t),\quad \hat a_t\sim\pi_\theta(z_t)$$
  $$\mathcal L(\theta;\Gamma_i)=c_1\|R_\theta-r\|^2+c_2\|Q_\theta-(r+\gamma Q_{\theta^-})\|^2+c_3\|d_\theta(z_i,a_i)-h_{\theta^-}(s_{i+1})\|^2$$
- 三问：Q1 **否**（在线阶段不修正状态：每次规划都重新 $z=h_\theta(s_t)$ 重锚定，预测状态被丢弃）；Q2 **否**（无 $\Delta$；原文脚注只说 "environment steps / decision steps"，时间语义就是步数乘 action repeat）；Q3 **是规划**（MPPI/MPC），但推理时也靠重编码而不是新息。
- 与目标公式异同：
  - A **有**（$z_{t+1}$ 是预测先验）。
  - B **只有潜空间版本**，且**跨了一步**：$d(z_t,a_t)$ 是 $t\to t+1$ 的预测，$h(s_{t+1})$ 是下一步的编码；这是"下一状态预测误差"，不是同一时刻的观测残差 $z-D(h^-)$，且**只进 loss，不进状态**。
  - C **无**；D **无**；E **无**；F **无**（temporal weighting $\lambda$ 是损失层面的，不是状态层面的）。
  - G **潜空间一致性（joint-embedding）+ 奖励回归 + TD(Q)**。
- 任务与训练目标：DMC 连续控制；decoder-free 的隐式世界模型 + 在线规划。
- 最接近的先例：Dreamer 系（对照对象）；BYOL/SimSiam 式 joint-embedding 预测；PETS/MPPI 系规划。
- 新颖性判断：**削弱"观测残差是新息"这一表述的普适性**。decoder-free 的含义是：**$D$ 不存在**，所以 $r=z-D(h^-)$ 在第 3 篇的严格形式下无法定义；能定义的只有"对下一步编码的预测误差"。若目标公式的 $D$ 是解码器，则它无法覆盖 TD-MPC 这类主流世界模型；若 $D$ 是编码器，则 $r$ 就是 RKN 2019 已经写过的"潜观测残差"。

### 9. Mastering Diverse Domains through World Models (DreamerV3)
- 作者 / 年份 / 会议或期刊 / 链接：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap / 2023（arXiv）/ 期刊版 Nature 640:647–653 (2025)，DOI 10.1038/s41586-025-08744-2 / https://arxiv.org/pdf/2301.04104 ；https://www.nature.com/articles/s41586-025-08744-2
- 核心更新方程（式 1、式 3）：
  $$h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\phi(z_t\mid h_t,x_t),\quad \hat z_t\sim p_\phi(\hat z_t\mid h_t)$$
  $$\mathcal L_{\text{dyn}}=\max(1,\mathrm{KL}[\mathrm{sg}(q(z_t\mid h_t,x_t))\parallel p(z_t\mid h_t)]),\quad \mathcal L_{\text{rep}}=\max(1,\mathrm{KL}[q(z_t\mid h_t,x_t)\parallel \mathrm{sg}(p(z_t\mid h_t)])$$
  加上 symlog/twohot 预测损失；free bits 1 nat；categorical 参数化为 "1% uniform + 99% 网络输出" 的混合。
- 三问：Q1 **部分**（同 DreamerV2，KL 方向对称化）；Q2 **否**；Q3 **是想象训练 + 通用域**。
- 与目标公式异同：A **有**；B **无**；C **无**；D **无**；E **无**；F **无**；G **ELBO 变体（free bits + 非对称 KL）**。
- 任务与训练目标：跨域（Atari/DMC/MyDoom 等）固定超参；Nature 版强调单套超参。
- 最接近的先例：DreamerV1/V2（自引）。
- 新颖性判断：**削弱**，并给出"朴素先验-后验耦合会退化"的工程证据：需要 (i) 1 nat free bits、(ii) 1% uniform 混合、(iii) stop-grad 的非对称 KL 三件套才能稳定。若目标公式的 $B(h^-,r)$ 无同等约束，其训练稳定性不会自动优于这套 2023–2025 的方案。

### 10. TD-MPC2: Scalable, Robust World Models for Continuous Control
- 作者 / 年份 / 会议或期刊 / 链接：Nicklas Hansen, Hao Su, Xiaolong Wang / 2024 / ICLR 2024 / https://arxiv.org/pdf/2310.16828 ；arXiv:2310.16828
- 核心更新方程（式 2、式 3、式 4）：
  $$z=h(s,e),\quad z'=d(z,a,e),\quad \hat r=R(z,a,e),\quad \hat q=Q(z,a,e),\quad \hat a=p(z,e)$$
  $$\mathcal L(\theta)=\mathbb E_{\mathcal B}\Big[\sum_{t=0}^{H}\lambda^t\big(\|z'_t-\mathrm{sg}(h(s'_t))\|_2^2+\mathrm{CE}(\hat r_t,r_t)+\mathrm{CE}(\hat q_t,q_t)\big)\Big]$$
  多任务嵌入 $e$；奖励/价值用 log 变换后的离散回归。
- 三问：Q1 **否**（同 TD-MPC：重编码重锚定）；Q2 **否**；Q3 **是**（规划 + 想象策略先验）。
- 与目标公式异同：A **有**；B **只有潜空间、只进 loss**；C **无**；D **无**；E **无**；F **无**；G **joint-embedding + CE 奖励 + CE TD**。
- 任务与训练目标：多域连续控制、大规模多任务；decoder-free。
- 最接近的先例：TD-MPC（同组）、Dreamer 系、BYOL 系。
- 新颖性判断：**削弱**，同上；并且它把"潜空间残差只当训练信号"这一取舍做到了 2024 年的 SOTA，说明社区已经验证过"不把残差加回状态"也能工作——目标公式必须证明"加回去"带来增量。

### 11. Time-Aware World Model for Adaptive Prediction and Control (TAWM)
- 作者 / 年份 / 会议或期刊 / 链接：Anh N. Nhu, Sanghyun Son, Ming Lin / 2025 / ICML 2025 / https://arxiv.org/html/2506.08441v1 ；arXiv:2506.08441
- 核心更新方程（4.1.2 节，TD-MPC2 基线 + Euler 积分）：
  $$\hat z_{t+\Delta t}=z_t+d(z_t,a_t,\Delta t)\cdot\tau(\Delta t),\qquad \tau(\Delta t)=\max(0,\log_{10}\Delta t+5)$$
  强制 $\hat z_{t+\Delta t}\mid_{\Delta t=0}=z_t$；除编码器外**所有组件**条件于 $\Delta t$（$R,Q,p$ 均吃 $\Delta t$）；训练时每个 episode 从 log-uniform 采 $\Delta t$ 并设观测/控制率为 $1/\Delta t$；消融 Euler vs RK4。
- 三问：Q1 **否**（在线靠重编码重锚定，无残差修正）；Q2 **是**（秒级 $\Delta t$ 显式进入潜空间转移；实践范围约 $10^{-3}$–$5\times10^{-2}$ s，理论分析 $10^{-3}$–$1$ s）；Q3 **是规划/控制**。
- 与目标公式异同：
  - A **有**，且是"$\Delta t=0$ 时恒等"的强先验形态，与 $\Phi_\Delta$ 同类。
  - B **无**；C **无**；D **无**；E **有**（本清单里唯一"潜空间 + 秒级 $\Delta$"的组合）。
  - F **无**；G **TD-MPC2 目标**（潜一致性 + 奖励 + TD）。
- 任务与训练目标：Meta-World 机器人操作 + PDE-control 环境；跨观测率鲁棒性。
- 最接近的先例：TD-MPC2（基线）；Neural ODE 系。
- 新颖性判断：**最强削弱之一（针对第 1 行）**。原文开头即写："One notable limitation of TD-MPC2 and other state-of-the-art world models is that they **ignore the effect of the observation rate $\Delta t$ at inference time**"（4.1.2 节）。这正是目标公式第 1 行要解决的问题，且已被 2025 年的 ICML 论文用"$\Delta t$ 显式条件化 + log-uniform 采样"解决并给出跨采样率实测。

### 12. Generative Bayesian Filtering for State Estimation
- 作者 / 年份 / 会议或期刊 / 链接：Lei Cao, Sihang Feng, Jixin Yan, Tao Sun, Naichen Shi / 2026 / arXiv:2607.20521（预印本）/ https://arxiv.org/html/2607.20521v1
- 核心更新方程（式 1）：
  $$p(z_t\mid x_t,z_{t-1})=\frac{p(x_t\mid z_t)\,p(z_t\mid z_{t-1})}{p(x_t\mid z_{t-1})}$$
  先验来自转移模型，似然来自预训练 CVAE，测量更新被表述为**后验采样**（SGLD），而不是加性残差；"for online inference, GBF performs a Bayesian prediction–update recursion".
- 三问：Q1 **是**（贝叶斯意义上的用观测修正）；Q2 **否**（离散时刻；沿时间做递归但无 $\Delta$ 参数）；Q3 **纯在线状态估计**（工业过程监控、ECG 诊断），明确不是 RL。
- 与目标公式异同：A **有**；B **无显式残差**（用似然而非 $z-D$）；C **无加性**（后验采样）；D **无**；E **无**；F **无**；G **预训练生成模型 ELBO + 后验采样**。
- 任务与训练目标：传感器流的高维观测；状态估计精度与鲁棒性。
- 最接近的先例：Kalman filter、CVAE、score-based sampling。
- 新颖性判断：**削弱"通用在线滤波版本"的新颖性**。它证明到 2026 年，"转移先验 + 生成式观测模型 + 在线递推"仍是活跃且被独立发表的框架；把目标公式解释成"通用在线状态估计"时，它几乎没有空位。

---

## 3. 七轴横向总表

| # | 文献（年） | A 先验 $h^-$ | B 观测残差 $r=z-D(h^-)$ | C 加性 $h=h^-+B$ | D $B$ 依赖 $(h^-,r)$ | E 真实 $\Delta$(秒) | F 稀疏/选择/局部 | G 训练目标 |
|---|---|---|---|---|---|---|---|---|
| 1 | DVBF (2017) | 部分 | 无 | 有 | 弱 | 无 | 无 | ELBO |
| 2 | World Models (2018) | 半 | 无 | 无 | 无 | 无 | 无 | 一步预测 + 重构 + RL |
| 3 | RKN (2019) | 有 | **有** | **有** | **有** | 无 | 弱 | 重构似然 |
| 4 | PlaNet (2019) | 有 | 无 | 无 | 无 | 无 | 无 | ELBO + 重构 |
| 5 | Dreamer (2020) | 有 | 无 | 无 | 无 | 无 | 无 | ELBO + 想象 RL |
| 6 | DreamerV2 (2021) | 有 | 无 | 无 | 无 | 无 | 无 | ELBO + KL 加权 |
| 7 | CT-MBRL (2021) | 有 | 无 | 积分项 | — | **有** | 无 | ELBO + RL |
| 8 | TD-MPC (2022) | 有 | 潜空间/仅 loss | 无 | 无 | 无 | 无 | 一致性 + 奖励 + TD |
| 9 | DreamerV3 (2023/2025) | 有 | 无 | 无 | 无 | 无 | 无 | ELBO 变体 |
| 10 | TD-MPC2 (2024) | 有 | 潜空间/仅 loss | 无 | 无 | 无 | 无 | 一致性 + CE + TD |
| 11 | **TAWM (2025)** | 有 | 无 | 无 | 无 | **有** | 无 | TD-MPC2 目标 |
| 12 | GBF (2026) | 有 | 无 | 无 | 无 | 无 | 无 | 生成式 ELBO + 后验采样 |
| — | **目标公式** | 有 | 有（观测空间） | 有 | 有 | 有 | ？（空） | 未定 |

读表结论：**目标公式 = RKN 的 (A,B,C,D) + TAWM 的 (E) + 一个尚未被占据的 (F)**。这就是它能主张的全部空间。

---

## 4. 负面结果与反例

### 4.1 证据一：RSSM 的后验不是残差修正，而是重推断

- PlaNet 的编码器写成 $q(s_{1:T}\mid o_{1:T},a_{1:T})=\prod_t q(s_t\mid h_t,o_t)$，每个 $s_t$ 都是**新采样的随机变量**，不是对 $h_t$ 的偏移。若真是残差修正，$s_t$ 应当是 $h_t$ 上的加性增量；实际是重参数化采样。
- DreamerV2 明确了后验网络结构："the representation model is implemented as a CNN followed by an **MLP that receives the image embedding and the deterministic recurrent state**"——联合输入，不是加法。
- DreamerV3 用 $\mathcal L_{\text{dyn}}$ 与 $\mathcal L_{\text{rep}}$ 两条 stop-grad KL 把先验与后验**互相拉**；若是残差修正，只需要一条把修正项推向"使似然最大"的通路，不需要方向相反的双 KL。
- 结论：把这件事命名为 *filtering* 是文献共识（PlaNet 原文、DVBF 原文、Dreamer 原文的 "mimics a non-linear Kalman filter"），但**没有人把它写成 $h=h^-+B(h^-,r)$**。因此目标公式在"写不写成残差"这一层是表述创新，而不是机制创新；除非能证明残差参数化带来可测的优化优势。

### 4.2 证据二：decoder-free 世界里"新息"概念缺失，$r$ 无处定义

- TD-MPC/TD-MPC2 没有 $D$。它们的一致性项是 $\|d(z_t,a_t)-\mathrm{sg}(h(s_{t+1}))\|^2$：**跨一步**的"下一状态编码"误差，且带 stop-grad、只出现在损失里。
- 与目标公式的 $r=z-D(h^-)$ 不是同一个量：后者是"同一时刻观测 vs 由预测状态解码出的观测"，前者是"下一步编码 vs 由当前状态预测出的下一步状态"。二者维度、时间对齐、角色都不同。
- 定量后果：TD-MPC2（ICLR 2024）与 TAWM（ICML 2025）都**不做在线残差修正**却达到 SOTA。这是对"必须用新息修正状态"的直接反例——至少说明"不做也不错"。
- 反过来，如果目标公式把 $D$ 定义为编码器 $h$，那 $r$ 就是 RKN 2019 的 $w_t-Hz^-_t$，新颖性归零。

### 4.3 证据三：没有 $\Delta$ 的采样率依赖，已被两篇论文正面处理

- TAWM（ICML 2025）："TD-MPC2 and other state-of-the-art world models ... ignore the effect of the observation rate $\Delta t$ at inference time"，并用跨观测率实验展示退化；其解药是 $\Delta t$ 显式条件化 + log-uniform 采样。
- CT-MBRL（ICML 2021）：非均匀到达时"same inputs, different targets"，离散 MBRL 的训练目标自相矛盾。
- 因此：**"有 $\Delta$"是 2021/2025 已占据的格子**；反过来，"没有 $\Delta$ 的固定步长 RSSM 在变采样率下会退化"是已知缺陷，但修好它不构成新方法。

### 4.4 证据四：这类结构（预测 + 修正）的四个已知失效模式

1. **修正项吞掉动力学（退化）**。DVBF 自己的话："in theory, the learning algorithm could still learn the transition as $z_{t+1}=w_t$"——若 $B$ 无约束，$h=h^-+B(h^-,r)$ 的最优解可以是"忽略 $h^-$、直接复制观测编码"，此时世界模型退化成编码器，长程预测能力归零。**这是目标公式最致命的理论反例**，必须在方法里用结构或正则排除。
2. **先验没训好时，拉向先验有害**。DreamerV2 KL balancing 的动机原文（"avoid regularizing the representations toward a poorly trained prior"）；DreamerV3 进一步要 1 nat free bits + 1% unimix + 非对称 KL 才能稳定，并报告过 KL spike。
3. **一步最优 ≠ 多步最优**。PlaNet 的 latent overshooting 章节：有限容量下，只训一步预测的模型不是多步预测最好的模型。对目标公式的含义：$B$ 只要让一步残差变小，不等于让 $K$ 步滚动误差变小；"残差修正"本身不提供多步一致性保证。
4. **潜状态漂移 / 表征塌缩**。TD-MPC/2 必须用 EMA 目标网络 + stop-grad + 归一化（simnorm/LayerNorm）来阻止表征塌缩；DreamerV3 用 unimix 阻止 categorical 分布变确定性。目标公式若没有对应的稳定器，$D(h^-)$ 与 $z$ 的尺度会漂移，$r$ 的统计量随时间非平稳。

### 4.5 失效模式速查表

| 失效模式 | 文献证据 | 对目标公式的后果 |
|---|---|---|
| 修正项吸收动力学、$h^-\to$ 无用 | DVBF 原文承认 $z_{t+1}=w_t$ 可达 | $B$ 必须被约束（小增益、稀疏、或对 $h^-$ 的显式依赖损失） |
| 先验-后验单向耦合破坏表征 | DreamerV2 KL balancing 动机；DreamerV3 free bits+unimix | 残差通路同样需要正则；否则等同于用观测覆盖状态 |
| 一步误差小但多步发散 | PlaNet latent overshooting | 需要多步（overshooting）或 $\Delta$-条件化的滚动训练 |
| 表征塌缩 / 漂移 | TD-MPC/2 的 EMA+sg+归一化 | $D$ 与 $h$ 必须共享/对齐，否则 $r$ 不可比 |
| 变采样率下失效 | TAWM 2025；CT-MBRL 2021 | 若无 $\Delta$，跨频率迁移会掉点（目标公式此处占优，但不新） |
| 缺测/丢帧 | RKN：缺测时 posterior = prior | 目标公式天然支持（$r$ 置零即可），但这也是 RKN 已有的性质 |

---

## 5. 与目标公式最接近的先例（排序）

1. **RKN（ICML 2019）**——第 2、3 行逐字对应：先验 $z^-$、新息、解析/学习增益、加性更新、缺测退化为先验。差 $D$ 的非线性与 RL 目标。
2. **TAWM（ICML 2025）**——第 1 行逐字对应：潜空间 Euler 步 + 秒级 $\Delta t$ + 多采样率训练，基线就是 TD-MPC2。
3. **DVBF（ICLR 2017）**——加性修正项 $C_tw_t$ 由观测推断，且本人写下退化风险。
4. **PlaNet / Dreamer 家族**——"posterior = filtering" 的正式表述与 KL 训练机制；提供了"用观测修正先验"的语义先例（只是不写成残差）。
5. **CT-MBRL（ICML 2021）**——$\Delta$ 进转移 + 非均匀采样的目标冲突论证。
6. **GBF（2026）**——预测-更新递推 + 生成式观测似然的当代形态，说明"通用在线滤波"版本没有空位。

**有没有与目标公式逐字对应的已有工作？**
- 三行**各自**都有逐字对应的先例（第 2/3 行：RKN 2019；第 1 行：TAWM 2025）。
- 三行**同时**出现在同一个潜状态世界模型里、并用 RL 目标训练的：本次检索**未找到**。这是唯一的空白，但它属于"把两个已有模块接起来"，需要额外的理论或实证理由才够新。

---

## 6. 核实与检索记录

- 元数据（题名/作者/年份/会议）：全部经 export.arxiv.org API 或 PMLR/NeurIPS/Nature 官方页面核对；ICML 年份与 PMLR 卷号取自 PDF 首页页眉（PMLR 97 = ICML 2019，PMLR 139 = ICML 2021，PMLR 162 = ICML 2022，PMLR 80 = ICML 2018）。
- 方程：全部来自 PDF/HTML 正文抓取，未凭记忆。原文引用句均可在所给链接中检索到。
- [仅摘要级]：arXiv:2605.01694（Keon Woo Kim, 2026, *Latent State Design for World Models under Sufficiency Constraints*，https://arxiv.org/abs/2605.01694 ）——只抓了 abs 页，仅用于一句分类学断言（该文把 "recurrent belief state" 列为世界模型潜状态六种角色之一），**未引用其公式**。
- 检索词（tavily，advanced）：world models latent state observation prediction online filtering；RSSM posterior inference prediction error update；recurrent state space model deterministic stochastic posterior prior；model-based RL latent state estimation observation residual；latent ODE world model irregular time steps delta；time-aware world model delta t。
- 未纳入但相关的方向（供 lead 决定是否另建条目）：Latent ODEs for Irregularly-Sampled Time Series（Rubanova 2019，非 RL）；Probabilistic Recurrent State-Space Models（Doerr 2018，GP-SSM + 变分平滑，无在线残差更新）；Orthogonal JEPA（2026，潜世界模型的分解式预测状态）。
