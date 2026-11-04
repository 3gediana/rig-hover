# 05 不规则时间与连续时间状态转移 —— 逐篇核对

> 范围：只做网络检索与文献核对（tavily_extract 优先；PDF 原文取公式）。
> 被检验的目标公式：
> `h^- = Phi_Delta(h)`；`r = z - D(h^-)`；`h = h^- + B(h^-, r)`
> 符号：Phi_Delta 状态转移（真实时间间隔 Delta）、D 观测/解码映射、B 修正函数、z 观测、h 隐状态。
> 所有条目均用 arXiv PDF 或官方 proceedings PDF 拉取正文核对（不是摘要级）；公式按 PDF 文本抓取，个别上下标在 PDF 抽取中会丢层级，已用同段文字互校。

## 0. 判定口径与先给结论

七轴：A 显式预测 h^-；B 误差定义在观测空间 r=z-D(h^-)；C 修正是加性 h=h^-+B；D B 同时依赖 h^- 与 r；E 真实时间间隔 Delta；F 稀疏/选择性/局部化；G 训练目标（重构/ELBO/一步预测/RL/对比/自监督）。
三问：Q1 每次观测是否用预测误差/新息修正隐状态；Q2 是否按真实 Delta 演化、Delta 从哪进入；Q3 服务规划/生成未来，还是本身就是通用在线感知/状态估计。

先给横向结论（Delta 从哪里进公式，全部来自原文）：

| 论文 | Delta 进入位置 | 时间语义 |
|---|---|---|
| Neural ODE 2018 | ODE 求解积分区间 [t0,T]；向量场可吃 t | 真实时间；但模型本身不感知采样 |
| Latent ODE / ODE-RNN 2019 | ODESolve 积分区间 (t_{i-1},t_i)；备选把 Delta 拼进 RNNCell | 真实秒 |
| GRU-ODE-Bayes 2019 | 积分区间；观测在 t[k] 处触发跳变 | 真实秒 |
| Neural CDE 2020 | 时间作为控制路径 X 的额外通道（X∈R^{v+1}），取 dX/dt | 真实时间戳 |
| Neural Flows 2021 | flow 的自变量 t（可为 φ(t)=αt 或 Fourier 特征） | 真实时间 |
| mTAN 2021 | 时间嵌入 φ_h(t)=ωt+α / sin(ωt+α)，进 attention 权重 | 真实时间 |
| LTC 2021 | Euler 步长 Delta 线性出现在分子与分母 | 求解器步长 |
| CfC 2022 | 指数/sigmoid 的非线性自变量：σ(-f·t)，t 与 f 相乘 | 真实时间戳（原文：按 timestamp 或顺序赋值） |
| GRU-D 2018 | 衰减门 γ=exp(-max(0,Wδ+b))，δ=距上次观测的间隔 | 真实秒 |
| NJ-ODE 2021 | 作为向量场显式输入 (t - t_{i-1}) | 真实秒 |
| Vid-ODE 2021 | ODESolve 积分区间 | 真实时间戳 |
| Consistency 2023 | 扩散时间 t（非物理秒） | 生成时间 |
| Flow map 2025 | 区间 [s,t] 的长度 t-s | 生成时间 |

能力对照（谁能证半群、谁只数值近似、谁明显不满足）见 §2。

## 1. 逐篇

### 1. Neural Ordinary Differential Equations
- 作者 / 年份 / 会议或期刊 / 链接：Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud / 2018 / NeurIPS 2018（PDF 原文页脚："32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada"）/ https://arxiv.org/abs/1806.07366
- 核心更新方程（PDF 式 (1)(2)）：
  ```
  ResNet/Euler:  h_{t+1} = h_t + f(h_t, theta_t)            (1)
  Neural ODE:    dh(t)/dt = f(h(t), t, theta)               (2)
                 h(T) = ODESolve(f, h(0), (t0, T))
  ```
  反向用 adjoint（常数内存）；CNF 用瞬时变量替换公式算精确 log-likelihood。
- 与目标公式异同：
  - A 有：h(t)=Phi_{t-t0}(h0) 是先验状态；但整条轨迹由初值唯一决定，没有"每次观测后的先验"。
  - B 无：论文里没有任何 z - D(h^-) 形式的观测空间新息。
  - C 无：只有转移，没有修正项。
  - D 无：B 不存在，谈不上依赖 (h^-, r)。
  - E 有：Delta=t-t0 是积分区间；时间也作为 f 的显式输入。
  - F 无：状态连续、稠密，无选择性/局部化。
  - G 最大似然/精确 log-likelihood（CNF）；adjoint 梯度；无 RL。
- 任务与训练目标：连续深度网络、连续归一化流生成；时序只做了 toy 的 latent ODE。
- 最接近的先例（该文自引或同期）：ResNet 的 Euler 视角（式 1 即其连续极限）；后续 Latent ODE、Neural CDE 全部建在它上面。
- 三问：Q1 否；Q2 是（真实时间进积分区间与 f(t)，但采样时刻不进模型）；Q3 通用建模/生成，不是规划。
- 新颖性判断：**削弱**目标公式的 A、E 两根轴（连续转移+真实 Delta 自 2018 就是标准件）；不涉及 B/C/D，因此目标的真正卖点若在残差修正，与本文无关。

### 2. Latent ODEs for Irregularly-Sampled Time Series（含 ODE-RNN）
- 作者 / 年份 / 会议或期刊 / 链接：Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud / 2019 / NeurIPS 2019 / https://arxiv.org/abs/1907.03907 ；proceedings PDF：https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf
- 核心更新方程（PDF §3.1；并与式 (1)(2) 对照）：
  ```
  h'_i = ODESolve(f_theta, h_{i-1}, (t_{i-1}, t_i))     # 观测间：连续流 = Phi_Delta
  h_i  = RNNCell(h'_i, x_i)                             # 观测时刻：跳变 = B 的位置
  对比基线 (1): h_i = RNNCell(h_{i-1}, Delta_t, x_i)    # 把 Delta 当特征拼接
  对比基线 (2): h_i = RNNCell(h_{i-1} * exp(-tau*Delta_t), x_i)  # 指数衰减
  ```
- 与目标公式异同：
  - A 有：h'_i 就是显式先验状态（原文称 pre-update state）。
  - B 无（显式）：修正吃的是原始 x_i，不是 z-D(h^-)；观测空间误差只以 VAE 重构/ELBO 的形式间接出现。
  - C 部分是：GRU 更新 h_t = z⊙h_{t-1} + (1-z)⊙g 为凸组合，不是 h^-+B 的残差写法。
  - D 半是：RNNCell 同时见到 h'_i 与 x_i，信息上等价于见 (h^-, 新息)，但没有显式 r。
  - E 有：Delta=t_i-t_{i-1} 是 ODE 积分区间，真实秒；论文明确说"模型不需要固定采样率"。
  - F 无：不处理部分观测（那是 GRU-D 等的 mask 路线）。
  - G ODE-RNN 自回归/一步预测；Latent ODE 为 ELBO + 观测时刻的 Poisson 过程似然。
- 任务与训练目标：不规则采样序列的插值/外推/生成（PhysioNet 死亡率、人类活动、toy 周期信号）。
- 最接近的先例：Neural ODE（连续流）；同期的 GRU-ODE；文中把 Che et al. 2018 的衰减 RNN 与 Mozer et al. 2017 列为基线并指出"指数衰减没有超过标准 RNN"。
- 三问：Q1 是（每个观测都更新隐状态）；Q2 是（真实间隔进积分区间；原文："Both ODE-RNNs and Latent ODEs can naturally handle arbitrary time gaps between observations"）；Q3 通用序列建模（预测/生成），非规划。
- 新颖性判断：**最强削弱之一**。ODE-RNN 已经是"连续流 Phi_Delta + 观测跳变"的合体（2019，NeurIPS）；目标公式与它只差"修正写成观测空间残差 + 加性增益"这一层。原文"a standard RNN update h_i = RNNCell(h'_i, x_i)"就是目标 B 的位置。

### 3. GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series
- 作者 / 年份 / 会议或期刊 / 链接：Edward De Brouwer, Jaak Simm, Adam Arany, Yves Moreau / 2019 / NeurIPS 2019 / https://arxiv.org/abs/1905.12374
- 核心更新方程（PDF §2.1–2.4）：
  ```
  dh(t)/dt = (1 - z(t)) ⊙ (g(t) - h(t))                       (3)  # GRU-ODE 连续流
  h(t+) = GRU( h(t-), f_prep(y[k], m[k], h(t-)) )              (4)  # 观测时刻跳变
  L_pre  = -Σ m_j log p(y_j | theta = f_obs(h^-)_j)                  # 观测空间负对数似然
  L_post = Σ m_j D_KL( p_Bayes,j || p_post,j ),  p_Bayes ∝ p_pre·p_obs  # 模仿 Bayesian 后验
  ```
- 与目标公式异同：
  - A 有：h(t-) 是先验。
  - B 隐式有：f_obs(h^-) 给出观测分布，L_pre 等价于观测空间残差的负对数似然（高斯下即 (y-f_obs)^2）；但没有显式向量 r。
  - C 无显式加性：用 GRU 更新；但整体是"预测步 + 更新步"的 Kalman 式两段结构（原文自述）。
  - D 有：f_prep 同时吃 h^- 与 y（含 mask）。
  - E 有：观测间用 ODE 连续积分，观测时刻 t[k] 触发跳变。
  - F 有：逐维 mask；还有逐维序贯更新变体 GRU-ODE-Bayes-seq。
  - G 观测负对数似然 + 对 Bayes 后验的 KL（自监督/近似 ELBO），非 RL。
- 任务与训练目标：稀疏观测时序（ICU、气候）的滤波、插值、预测；声称能量化不确定性并"精确表示 OU 过程的 Fokker–Planck 动力学"。
- 最接近的先例：Neural ODE-VAE（Chen 2018 的 latent ODE）、Kalman/EKF（原文："Like the celebrated Kalman filter, it alternates between a prediction (GRU-ODE) and a filtering (GRU-Bayes) phase"）。
- 三问：Q1 是（明确的新息式修正）；Q2 是（真实观测时刻 + 连续积分）；Q3 通用在线滤波/状态估计。
- 新颖性判断：**强削弱**。连续时间预测 + 观测更新 + 贝叶斯式修正已被 2019 年做到；目标公式只差把修正写成"显式观测残差 + 增益网络"。且它已经指出核心动机正是"滤波"。

### 4. Neural Controlled Differential Equations for Irregular Time Series
- 作者 / 年份 / 会议或期刊 / 链接：Patrick Kidger, James Morrill, James Foster, Terry Lyons / 2020 / NeurIPS 2020（Spotlight）/ https://arxiv.org/abs/2005.08926
- 核心更新方程（PDF 式 (3)(4)(5)）：
  ```
  z_t = z_{t0} + ∫_{t0}^{t} f_theta(z_s) dX_s ,  z_{t0}=zeta(x0,t0)        (3)
  X : [t0,tn] -> R^{v+1} 为过 (t_i, (x_i, t_i)) 的自然三次样条               # 时间=额外通道
  g_{theta,X}(z,s) = f_theta(z) · dX/ds                                    (4)
  z_t = z_{t0} + ∫_{t0}^{t} g_{theta,X}(z_s, s) ds                         (5)  # 可用 ODE 求解器
  ```
- 与目标公式异同：
  - A 无：没有"先验/后验"二分，隐状态沿数据路径连续演化。
  - B 无：数据以控制路径的微分进入，不是误差。
  - C 无：不做跳变修正（这正是论文卖点：不需要在观测时刻改状态）。
  - D 无。
  - E 有：时间进 X 的额外通道，真实时间戳；但没有 Delta 乘子。
  - F 无（支持不规则、部分观测；选择性靠样条插值）。
  - G 监督训练（adjoint），分类/预测。
- 任务与训练目标：不规则/部分观测序列分类与预测（CharacterTrajectories、sepsis、Speech Commands）；证明对 stream 函数类通用逼近。
- 最接近的先例：Neural ODE（把"driven by identity"换成"driven by data"）；rough path / CDE 理论（Lyons）。
- 三问：Q1 否（观测连续进入，无跳变式新息）；Q2 是（真实时间戳进控制路径）；Q3 通用序列建模。
- 新颖性判断：**削弱"必须 B"的隐含假设**：不跳变、不做残差修正也能处理不规则采样并达 SOTA；同时它的三问显示，连续时间语义完全可以由"数据路径"承担而非由"Delta 转移"承担。

### 5. Neural Flows: Efficient Alternative to Neural ODEs
- 作者 / 年份 / 会议或期刊 / 链接：Marin Biloš, Johanna Sommer, Syama Sundar Rangapuram, Tim Januschowski, Stephan Günnemann / 2021 / NeurIPS 2021 / https://arxiv.org/abs/2110.13040
- 核心更新方程（PDF 式 (2)(5)(6)，以及式 (4) 的替换）：
  ```
  ODE-RNN/GRU-ODE:  h_bar_{t1} = ODESolve(h0, GRU-ODE, t0, t1);  h_{t1} = GRUCell(h_bar_{t1}, x_{t1})   (4)
  ResNet flow:      F(t,x) = x + φ(t) g(t,x)                                (2)
  GRU flow:         F(t,h) = h + φ(t)(1 - z(t,h)) ⊙ (c(t,h) - h)            (5)
  Coupling flow:    F(t,x)_A = x_A exp(u(t,x_B) φ_u(t)) + v(t,x_B) φ_v(t)   (6)
  用法：把 (4) 里的 ODESolve 换成 F(t1-t0, ·)（编码器仍保留观测时刻的 GRU 跳变）
  ```
- 与目标公式异同：
  - A 有：h_i = F(Δt, h_{i-1}) 后接 GRU，与 ODE-RNN 同构；F 就是"显式可算的 Phi_Delta"。
  - B 无：无观测空间残差。
  - C 无：flow 内部的 x+φg 是转移的残差，不是"修正项的加性"。
  - D 无。
  - E 有：t 作为 flow 的自变量（真实时间），可为 αt 或 Fourier 特征。
  - F 无。
  - G 变分推断/ELBO（不规则序列 latent variable model）+ 密度估计；非 RL。
- 任务与训练目标：用闭式 flow 替代 ODE 求解器，做不规则时序建模与密度估计，速度提升而保持精度。
- 最接近的先例：Neural ODE、ODE-RNN、GRU-ODE（式 (4) 直接引用 De Brouwer 等）。
- 三问：Q1 是（编码器保留观测跳变 GRU）；Q2 是（真实时间进 F 与 φ）；Q3 通用。
- 新颖性判断：**双重削弱**。(i) 它给出"Delta→闭式转移映射"的直接可学习参数化，是目标 Phi_Delta 的最近替代；(ii) 它自称 flow，但只列了两条条件——原文 i) F(0,x0)=x0，ii) F(t,·) 可逆——并声称"与 ODE flow 精确对应（原文引 [47, Thm 9.12]）"；该定理的完整条件还包含半群 X_{u,t}(X_{s,u}(x))=X_{s,t}(x)。按其自述条件，ResNet flow 一般不满足半群（见 §2 代数），这说明"叫 flow"在文献里并不等于"可组合"。

### 6. Multi-Time Attention Networks for Irregularly Sampled Time Series
- 作者 / 年份 / 会议或期刊 / 链接：Satya Narayan Shukla, Benjamin M. Marlin / 2021 / ICLR 2021 / https://arxiv.org/abs/2101.10318
- 核心更新方程（PDF 式 (1)–(4)）：
  ```
  φ_h(t)[i] = ω_h·t + α_h          (i=0)
            = sin(ω_h·t + α_h)     (0<i<d_r)                       (1)
  mTAN(t,s)[j] = Σ_h Σ_d x_hat_hd(t,s) · U_hdj                       (2)
  x_hat_hd(t,s) = Σ_i κ_h(t, t_id) x_id                             (3)
  κ_h(t, t_id) = softmax_i( φ_h(t) w v^T φ_h(t_id)^T / sqrt(d_k) )  (4)
  ```
  原文关键句："for a given difference Δ, φ_h(t+Δ) can be represented as a linear function of φ_h(t)"——时间嵌入本身是仿射可加的，但 attention 权重不是。
- 与目标公式异同：
  - A 无：没有任何状态转移/先验。
  - B 无：decoder 的重构在观测空间，但不是逐观测修正。
  - C 无。
  - D 无。
  - E 有：真实时间进时间嵌入与 attention 的相对时间；但只用于加权观测。
  - F 有：soft attention = 对观测的软选择，稀疏数据可局部化。
  - G VAE 的 ELBO（重构 + KL）+ 分类交叉熵。
- 任务与训练目标：不规则、稀疏、多元时序的插值与分类（EHR），训练显著快于 ODE 类。
- 最接近的先例：核平滑插值（Shukla & Marlin 2019）、Latent ODE。
- 三问：Q1 否（无递归状态修正）；Q2 是（时间进嵌入，但语义是"给观测加权"）；Q3 通用插值/分类。
- 新颖性判断：**削弱"必须有显式 Phi_Delta"的论据**——不做状态转移、不做误差修正，用时间嵌入 attention 就能处理不规则采样。它也是"Delta 只做特征/权重"的典型：把时间放在嵌入里不等于连续时间动力学。

### 7. Liquid Time-constant Networks
- 作者 / 年份 / 会议或期刊 / 链接：Ramin Hasani, Mathias Lechner, Alexander Amini, Daniela Rus, Radu Grosu / 2021 / AAAI 2021（PDF 版权行："Copyright © 2021, Association for the Advancement of Artificial Intelligence"）/ https://arxiv.org/abs/2006.04439
- 核心更新方程（PDF 式 (1) 与 Algorithm 1 的 fused step）：
  ```
  dx(t)/dt = -[ 1/τ + f(x(t), I(t), t, θ) ] x(t) + f(x(t), I(t), t, θ) A      (1)
  fused step: x(t+Δt) = x(t) + Δt · f(x,I,t,θ) ⊙ A / ( 1 + Δt (1/τ + f(x,I,t,θ)) )
  ```
- 与目标公式异同：
  - A 有：积分给出先验。
  - B 无：输入 I(t) 直接进向量场，没有观测残差。
  - C 无。
  - D 无。
  - E 有：Δt 是求解器步长（线性出现在分子/分母）；"liquid time-constant" τ_sys=τ/(1+τf) 随输入变化。
  - F 无。
  - G 有监督序列任务（时间序列回归/分类）+ 端到端驾驶的行为克隆（NCP）。
- 任务与训练目标：稳定、可解释的连续时间 RNN；用 fused explicit/implicit Euler 处理刚性 ODE。
- 最接近的先例：Neural ODE、CT-RNN（Funahashi & Nakamura 1993）。
- 三问：Q1 否（无观测修正）；Q2 是（连续时间 + 步长 Discrete Δ）；Q3 部分是控制策略（端到端驾驶）。
- 新颖性判断：Delta 进离散更新（线性缩放）的早期例子；Euler 离散不满足半群，且无误差修正。作为"Delta 只进步长"的反例有价值。

### 8. Closed-form Continuous-time Neural Networks
- 作者 / 年份 / 会议或期刊 / 链接：Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus / 2022 / Nature Machine Intelligence 4:992–1003（DOI 10.1038/s42256-022-00556-7；Nature 页面"Published: 15 November 2022"）/ https://arxiv.org/abs/2106.13898 与 https://www.nature.com/articles/s42256-022-00556-7
- 核心更新方程（arXiv v2 式 (9)(10)，Lemma 1）：
  ```
  x(t) = B ⊙ e^{-[w_τ + f(x,I;θ)] t} ⊙ f(-x,-I;θ) + A                                  (9)
  x(t) = σ(-f(x,I;θ_f) t) ⊙ g(x,I;θ_g) + (1 - σ(-[f(x,I;θ_f)] t)) ⊙ h(x,I;θ_h)          (10)
  Lemma 1: |x(t) - x_tilde(t)| ≤ |x(0)-A| e^{-w_τ t}（且界是紧的）
  ```
  时间语义（原文）："for each incoming sample t is set based on its time-stamp or order"——对不规则数据就是真实间隔。
- 与目标公式异同：
  - A 有：x(t) 由上一状态沿 t 演化（式 9 的 exp 项）。
  - B 无：无观测残差。
  - C 无：不是 h^-+B，而是 g 与 h 的凸插值（gate 混合）。
  - D 无。
  - E 有：Delta 进指数/sigmoid 的非线性自变量（t 与 f 相乘）；这是"Delta 进闭式转移"的最强先例。
  - F 无。
  - G 监督/行为克隆/RL 实验（bit-stream、情感、医疗、机器人运动学、驾驶）。
- 任务与训练目标：把 LTC 的积分近似成闭式，绕开数值求解器（训练/推理快 1–5 个数量级）；兼顾表达力。
- 最接近的先例：LTC（它的源）；Neural ODE。
- 三问：Q1 否；Q2 是（真实 Delta 进闭式非线性）；Q3 通用序列 + 控制。
- 新颖性判断：**对"Phi_Delta 用真实 Delta 算闭式"是正面先例（削弱）**；但它不满足半群（§2 给代数），也没有观测修正。若目标的贡献是"闭式转移 + 真实 Delta"，CfC 已占位（2022）；若贡献是"转移+修正的组合并证明采样细化一致性"，CfC 反而是反例。

### 9. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- 作者 / 年份 / 会议或期刊 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / Scientific Reports 8:6085（DOI 10.1038/s41598-018-24271-9；PubMed 29666385）/ https://arxiv.org/abs/1606.01865 与 https://www.nature.com/articles/s41598-018-24271-9
- 核心更新方程（PDF 式 (4)(5)(6)）：
  ```
  γ_t = exp{ -max(0, W_γ δ_t + b_γ) }                                   (4)
  x_t^d ← m_t^d x_t^d + (1-m_t^d) γ_{x,t}^d x_{t'}^d + (1-m_t^d)(1-γ_{x,t}^d) x_tilde^d   (5)
  h_{t-1} ← γ_{h,t} ⊙ h_{t-1}                                            (6)
  z_t = σ(W_z x_t + U_z h_{t-1} + V_z m_t + b_z),  ...（GRU 常规更新）
  ```
- 与目标公式异同：
  - A 无：没有显式先验状态。
  - B 无。
  - C 无：衰减是乘法，不是加性修正。
  - D 无。
  - E 有：δ_t = 距上次观测的时间间隔，真实秒；γ 是 δ 的指数函数。
  - F 有：逐变量 mask（选择/缺失模式）。
  - G 监督（死亡率预测、活动识别）。
- 任务与训练目标：缺失/不规则采样时序分类；把时间间隔与缺失模式编码进 GRU。
- 最接近的先例：RNN-Decay 一族及其 GRU 版本（Latent ODE 论文原话：Mozer et al. 2017 "found that empirically, exponential-decay dynamics did not improve predictive performance over standard RNN approaches"）。
- 三问：Q1 否（只做衰减，无新息修正）；Q2 是（真实 δ_t）；Q3 通用。
- 新颖性判断：**"Delta 进门控/衰减"早在 2016/2018**；且纯衰减算子是半群（e^{-τδ1}e^{-τδ2}=e^{-τ(δ1+δ2)}），一旦与观测注入复合就不再满足——正好预示目标公式的陷阱（§2）。

### 10. Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering
- 作者 / 年份 / 会议或期刊 / 链接：Calypso Herrera, Florian Krach, Josef Teichmann / 2021 / ICLR 2021 / https://arxiv.org/abs/2006.04727
- 核心更新方程（PDF 式 (6)(7)，Algorithm 1）：
  ```
  h'_i := ODESolve( f_θ, (h_{i-1}, x_{i-1}, t_{i-1}, t - t_{i-1}), (t_{i-1}, t_i) )   (6)
  h_i   := jumpNN(x_i)
  Φ_N(θ) = (1/N) Σ_j (1/n_j) Σ_i ( |x_i - y_i|  +  |y_i - y_i^-| )^2            (7)
           # 第一项=观测处误差(jump part)，第二项=跳变幅度(continuous part)
  ```
- 与目标公式异同：
  - A 有：h'_i 是先验；outputNN(h) 给出预测 y。
  - B 无显式残差：输出空间的误差只出现在 loss；jumpNN 直接吃 x_i。
  - C 无：跳变是状态替换，不是加性。
  - D 半是：jumpNN 只依赖 x_i（作者论证马尔可夫性下无需 h^-）。
  - E 有：t - t_{i-1} 作为 f_θ 的显式输入（Delta 进向量场）。
  - F 无。
  - G L2 回归到条件期望；理论证明输出 L1 收敛到 L2-最优在线预测。
- 任务与训练目标：连续时间在线预测/滤波（Black-Scholes、OU、Heston），并提出可证收敛的训练框架。
- 最接近的先例：ODE-RNN（其改进对象）、GRU-ODE-Bayes。
- 三问：Q1 是；Q2 是（Delta 显式进 f）；Q3 通用在线状态估计/滤波。
- 新颖性判断：**强削弱**：连续流 + 观测跳变 + Delta 显式进向量场 + 收敛保证，2021 已有；目标的剩余差异只有"观测空间残差 + 加性增益"这一层，以及（若主张）半群性质。

### 11. Vid-ODE: Continuous-Time Video Generation with Neural Ordinary Differential Equation
- 作者 / 年份 / 会议或期刊 / 链接：Sunghyun Park, Kangyeol Kim, Junsoo Lee, Jaegul Choo, Joonseok Lee, Sookyung Kim, Edward Choi / 2021 / AAAI 2021（arXiv abs："Accepted to AAAI 2021"）/ https://arxiv.org/abs/2010.08188
- 核心更新方程（PDF 式 (1)(2)）：
  ```
  h^-_{t_i} = ODESolve(f_θ, h_{t_{i-1}}, (t_{i-1}, t_i))          (1)
  h_{t_i}   = ConvGRUCell(h^-_{t_i}, E(X_{t_i}))
  h_{s1..sK} = ODESolve(f_φ, h_T, (s1..sK));  (F,D,M) = G(h_si, h_si-1)   (2)
  X_hat_si = M ⊙ W(F, X_hat_si-1) + (1-M) ⊙ D                    # 光流+差分图组合
  ```
- 与目标公式异同：
  - A 有：h^-_{t_i} 是先验。
  - B 无：解码在像素空间做光流/差分，但修正用的是 ConvGRU(h^-, E(X))，不是 z-D(h^-)。
  - C 无：ConvGRU 替换状态。
  - D 半是：ConvGRU 同时见 h^- 与帧嵌入。
  - E 有：真实帧时间戳，积分区间。
  - F 无。
  - G 对抗训练 + 像素重构/差分；生成式（视频预测/插帧/外推）。
- 任务与训练目标：任意时间戳的视频生成（帧率可变、缺帧）；连续时间视频生成。
- 最接近的先例：ODE-RNN（其作者明确说是 ODE-RNN 的卷积版）、Neural ODE。
- 三问：Q1 是；Q2 是；Q3 生成未来（model-based 生成），不是通用感知。
- 新颖性判断：**削弱**"视频上的连续流+观测修正"的新颖性（2021 已有）；缺的只是残差形式和半群性质。若目标的应用叙事是"连续时间视频/感知"，必须直面这篇。

### 12. Consistency Models
- 作者 / 年份 / 会议或期刊 / 链接：Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever / 2023 / ICML 2023（arXiv abs comments："ICML 2023"）/ https://arxiv.org/abs/2303.01469
- 核心更新方程（PDF §2 Definition、式 (4)(5)）：
  ```
  PF-ODE 轨迹上的 consistency function f: (x_t, t) ↦ x_ε
  self-consistency（Definition，原文非编号式）:  f(x_t, t) = f(x_{t'}, t')  对所有同一 PF-ODE 轨迹上的 (x_t,t),(x_{t'},t')
  boundary condition（式 4 的参数化）: f(x_ε, ε) = x_ε，即 fθ(x,t)=x 当 t=ε
  skip 参数化: f_θ(x,t) = c_skip(t) x + c_out(t) F_θ(x,t),  c_skip(ε)=1, c_out(ε)=0     (5)
  训练（consistency distillation）：用 ODE solver 走一步 x_hat_{t_n}=Φ_{t_{n+1}→t_n}(x_{t_{n+1}})，再令 f_θ(x_{t_{n+1}},t_{n+1}) ≈ f_θ(x_hat_{t_n}, t_n)
  ```
- 与目标公式异同：
  - A 无隐状态先验（x_t 本身就是状态）。
  - B 无：没有观测，只有采样轨迹。
  - C 无。
  - D 无。
  - E 半是：时间是连续扩散时间，不是物理秒；"任意 t,t' 一致"是它对任意 Delta 的离散/连续混合版本。
  - F 无。
  - G 去噪分数匹配/蒸馏（CD、ECT）；不是 RL。
- 任务与训练目标：一步或少步生成（图像/音频/视频），把 ODE 求解次数压到 1。
- 最接近的先例：概率流 ODE；原文明确说"a similar definition is used for neural flows (Biloš et al., 2021)"；progressive distillation。
- 三问：Q1 否；Q2 半是（连续 t，但语义是生成时间）；Q3 生成。
- 新颖性判断：**提供"把组合一致性当训练目标"的现成框架**：这正是目标公式若要给 Phi_Delta 加"采样细化一致性"损失时可借鉴的范式；但也说明这种约束需要特殊参数化（boundary condition）与蒸馏信号。它不涉及观测修正，故不构成对 r/B 的先例。

### 13. How to Build a Consistency Model: Learning Flow Maps via Self-Distillation
- 作者 / 年份 / 会议或期刊 / 链接：Nicholas M. Boffi, Michael S. Albergo, Eric Vanden-Eijnden / 2025 / NeurIPS 2025 / https://proceedings.neurips.cc/paper_files/paper/2025/file/2ff3605c9e442d55156d99fe4f5f228d-Paper-Conference.pdf
- 核心更新方程（PDF Proposition 2.2、2.3）：
  ```
  flow map X_{s,t}: R^d -> R^d,  X_{s,t}(x) 表示从时刻 s 的 x 出发到 t 的状态（X_{s,s}=id）
  (i)   Lagrangian: ∂_t X_{s,t}(x) = v_{t,t}(X_{s,t}(x))                         (8)
  (ii)  Eulerian:  ∂_s X_{s,t}(x) + ∇X_{s,t}(x) v_{s,s}(x) = 0                   (9)
  (iii) Semigroup: X_{u,t}( X_{s,u}(x) ) = X_{s,t}(x)  对所有 0≤s≤u≤t≤1 与所有 x    (10)
  Prop 2.3: X_{s,t}(x) = x + (t-s) v_{s,t}(x)，用自蒸馏损失同时罚 (i)(ii)(iii)
  ```
  原文对 (10) 的解释："The semigroup condition states that any two jumps can be replaced by a single jump."
- 与目标公式异同：
  - A 无（纯流，无观测）；B 无；C 无；D 无。
  - E 有：区间长度 t-s 是时间参数（生成时间，不是秒）。
  - F 无。
  - G 分数匹配 + 半群/拉格朗日/欧拉条件作为正则（自蒸馏）。
- 任务与训练目标：直接学习 flow map（一致性模型的一般化），把半群条件变成可训练目标。
- 最接近的先例：Consistency Models（Song 2023）、flow map matching（Boffi et al. 2024）、progressive distillation。
- 三问：Q1 否；Q2 是（区间长度进 X_{s,t}）；Q3 生成。
- 新颖性判断：这是文献里对"P_{δ1+δ2}=P_{δ2}∘P_{δ1}"最明确的陈述与训练化（式 (10)）。但它只处理无观测注入的纯流；目标公式的复合映射含 B，落在该框架之外（§2 论证），因此不能直接引用它替目标背书。

## 2. 半群/流一致性专节：谁能证明 P_{δ1+δ2}=P_{δ2}∘P_{δ1}

- **精确成立（无观测注入）**：
  1. 自治 ODE 的精确解（flow）。Boffi et al. 2025 式 (10) 给出显式条件 X_{u,t}(X_{s,u}(x))=X_{s,t}(x)；autonomous 情形 X_{s,t}=Φ_{t-s}，即 P_{δ1+δ2}=P_{δ2}∘P_{δ1}。
  2. 线性 ODE 的矩阵指数：Neural Flows 原文举线性 ODE `f(t,x)=Ax, x(0)=x0`，解 `F(t,x0)=exp(At)x0`，参数 A 乘 t——这正是 `exp(A(δ1+δ2))=exp(Aδ2)exp(Aδ1)` 的半群。注意这要求 A 与状态无关。
  3. CfC 式 (9) 在"系数冻结"（f、A、B 均为常数）时退化为仿射半群；一旦 f 依赖 x/I，它只是对每一步的冻结近似（论文自己给了 Lemma 1 的误差界 |x-x_tilde|≤|x0-A|e^{-wτ t}），不是精确半群。
- **只数值近似**：Neural ODE / ODE-RNN / GRU-ODE / Vid-ODE（ODE solver 离散）、LTC（fused Euler，原文明确 RK 不适合其刚性方程）、NJ-ODE（固定 Δt 逐步推进）。求解器误差随步长/容差变化，Neural ODE 原文只承诺"Modern ODE solvers provide guarantees about the growth of approximation error"。
- **自称 flow 但条件缺半群**：Neural Flows。原文列 i) F(0,x0)=x0，ii) F(t,·) 可逆，并称与 ODE flow 精确对应（引 Lee Thm 9.12）；但 ResNet flow `F(t,x)=x+φ(t)g(x)` 不满足 `F(δ2,F(δ1,x))=F(δ1+δ2,x)`：
  ```
  左 = x + φ1 g(x) + φ2 g(x + φ1 g(x))
  右 = x + φ(δ1+δ2) g(x)
  仅当 g 为常数/线性且 φ 可加时相等。
  ```
- **明确不满足半群**：CfC 式 (10)。取 g,h,f 为常数（合法参数化），令 σ=σ(-f·)：
  ```
  单步:   Φ_{δ}(x) = σ(fδ) g + (1-σ(fδ)) h            （σ(fδ) := σ(-fδ)）
  复合:   Φ_{δ2}(Φ_{δ1}(x)) = [σ2 + (1-σ2)σ1] g + (1-σ2)(1-σ1) h
  单步δ1+δ2: [σ(δ1+δ2)] g + (1-σ(δ1+δ2)) h
  系数不等（例 f=1, δ1=δ2=1: 复合系数=σ(-1)+...≠σ(-2)），故非半群。
  ```
- **含观测注入的复合映射一般不满足半群（论证）**：设 Φ_δ 是纯流，观测修正 J_z(h)=h+B(h, z-D(h))，复合一步 P_δ^z = J_z∘Φ_δ。于是：
  1. P 的下标不止 δ，还依赖"在哪个时刻注入了什么观测"——它不是单参数半群，无法写成 P_{δ1+δ2}；
  2. 即使固定一个观测 z 在固定时刻，连续注入两次与一次不同（重复计入同一信息）；
  3. 若两个观测 z1,z2 分别落在 t+δ1 与 t+δ1+δ2，则复合结果依赖 δ1（观测到达时刻），而"合并成一步"的 P_{δ1+δ2} 无法表达这种依赖。
  故 `P_{δ1+δ2}=P_{δ2}∘P_{δ1}` 只能在"区间内无观测（B≡0）"时成立。要谈一致性，只能退而求其次：观测点之间的流一致性、采样细化极限、或误差阶（见 §3）。
- **观测注入次序/求解器步长**：ODE-RNN 需在 t_i 精确停下注入；Latent ODE §3.4 说明 batch 内要对"所有时间点的并集"求解（不同序列观测时刻不同）；NJ-ODE 用固定小步长 Δt 循环到 t_i。三者都表明"注入时刻"是强约束：早一步/晚一步注入会带来 O(step) 的一阶误差，且会改变隐状态轨迹。

## 3. 负面结果与反例

**N1（结构性，最直接）ODE-RNN 已是"连续流 + 观测跳变修正"**：`h'_i=ODESolve(f,h_{i-1},(t_{i-1},t_i)); h_i=RNNCell(h'_i,x_i)`（Rubanova et al. 2019, NeurIPS）。目标公式的 Φ_Delta 与"观测时刻更新"两段与之逐位对应，差别只在修正是"GRU(h^-,x)"还是"h^-+B(h^-,r)"。若要主张新颖，必须主张"残差式学习增益"本身，而不是"流+修正"这个骨架。

**N2 Delta 当输入特征不等于连续时间**：ODE-RNN 的基线式 (1) `h_i=RNNCell(h_{i-1},Δt,x_i)` 就是"把 Delta 拼进特征"，论文随后指出这"raises the question of how to define the hidden state h between observations"；GRU-D 用 `γ=exp(-max(0,Wδ+b))` 把 Delta 变成衰减门；NJ-ODE 把 `t-t_{i-1}` 显式塞进向量场。这三种做法都没有流、没有半群、也无法在观测间任意时刻求值（GRU-D 与 Δ-as-feature 只能给出离散状态）。因此"Delta 进了公式"不构成连续时间性质，必须给出"任意时刻可求值 + 组合一致（或误差界）"才算。

**N3 求解器步长误差与观测注入次序**：(i) 所有 ODE 类模型（Neural ODE/ODE-RNN/GRU-ODE/Vid-ODE）的转移是数值解，误差随容差与步长变化；LTC 原文说明其方程刚性，RK 类求解器不适用，必须用 fused Euler。(ii) Latent ODE §3.4 明确指出 batch 求解要在"所有观测时刻的并集"上输出，观测时刻必须先验已知；(iii) NJ-ODE 的 Algorithm 1 用固定 Δt 循环到 t_i，注入发生在网格上。这些说明"Delta 精确进转移"在实现层面是一个近似，目标公式若声称"按真实 Delta 精确演化"需要声明求解器假设。

**N4 含观测注入的复合映射一般不满足半群**：见 §2 的论证与 CfC/Neural Flows 代数。这条直接打击"目标的 Phi_Delta 具备半群/采样细化一致性"这类主张：B≠0 时它不可能对任意 Delta 成立。可行的弱化命题只有：(a) 观测点之间的流一致；(b) B≡0 时的半群；(c) 采样细化下的一致性误差界（需要显式误差阶）。

**N5（跨专题，最致命）目标更新式 = EKF/Kalman 新息更新，且"学习增益"已有人做**：KalmanNet（Revach et al., IEEE TSP 2022, DOI 10.1109/TSP.2022.3158588, arXiv:2107.10043）原文式 (5a)(7)(8b)：
```
x_hat_{t|t} = x_hat_{t|t-1} + K_t · Δy_t        (5a)
Δy_t = y_t - y_hat_{t|t-1}                       (7)   # 新息 = z - D(h^-)
y_hat_{t|t-1} = h(x_hat_{t|t-1})                 (8b)
K_t 由 GRU 从 (Δy_t, Δx_hat_t 等特征) 输出        # 学习式增益
```
这与目标的 `r=z-D(h^-); h=h^-+B(h^-,r)` 在结构上同构（K_t=B、h^-=先验、Δy=r）。差别：KalmanNet 是离散时间索引，无真实 Delta，增益依赖 RNN 内部状态而非显式 (h^-,r)。因此"观测空间残差 + 加性学习增益"不是新思想（EKF 自 1960 年代；学习增益 2022），目标只能把新颖性押在"连续时间 Delta 转移 + 该修正的组合及其性质"上。该条属状态估计专题，建议由对应文件展开，此处仅作硬约束记录。

**N6 CfC 的 timespan 是否满足半群：不满足**。式 (10) 的 σ(-f·t) 把 Delta 与状态相关的 f 相乘后再过 sigmoid，复合两步得到 `σ2+(1-σ2)σ1`，与单步 `σ(δ1+δ2)` 不同（§2 代数）。式 (9) 的 exp 形式只有在 f、A、B 全为常数时才是精确仿射半群；对状态/输入依赖的 f，原文 Lemma 1 只给近似界。原文还说这种形式有梯度问题（"causes vanishing gradient factors"），因此把 exp 换成 sigmoid——这正是"Δ 进非线性"的已知代价。

**N7 "必须有显式转移+修正"这一前提本身不被文献支持**：Neural CDE（不跳变，数据作控制路径）与 mTAN（无状态转移，时间嵌入 attention）都在不规则采样上达到有竞争力的结果。说明目标的 B 步是设计选择而非必要结构；这降低"结构新颖性"的说服力，除非能用实验/证明说明它带来别处没有的性质。

**N8 已知失效模式清单（来自原文，不是猜测）**：
- ODE 类模型即使无观测也要连续积分，计算量不随稀疏性下降（Latent ODE §3.4 原文："as the ODE must be continuously solved even when no observations occur, the compute cost does not scale with the sparsity of the data"；ODE-RNN 比 GRU 慢约 60%）。
- 刚性方程 + RK 求解器会失败（LTC 原文：RK-based solvers "not suitable for LTCs"，需要 fused Euler）。
- 时间嵌入/衰减类方法在观测极稀疏时退化（GRU-D 的 decay 只能收敛到经验均值/零；ODE-RNN 论文报告 decay 未超过标准 RNN）。
- 插值路径会引入模型误差（Neural CDE 依赖自然三次样条；观测很稀疏时样条是强假设）。
- 注意力类方法代价 O(n²)（mTAN 对长序列）。
- 闭式近似有明确误差上界但不为零（CfC Lemma 1），且替换 exp→sigmoid 是工程妥协。
- 一步生成类（Consistency Models）需要先有（或蒸馏自）一个流模型，且训练不稳/依赖特殊参数化——若要借它的框架给目标加一致性损失，这些代价会转移过来。

## 4. A–G 汇总矩阵

| # | 论文 | A 先验 h^- | B 观测残差 | C 加性修正 | D B(h^-,r) | E 真实 Delta | F 稀疏/选择 | G 目标 |
|---|---|---|---|---|---|---|---|---|
| 1 | Neural ODE 2018 | 有 | 无 | 无 | 无 | 有（积分区间） | 无 | 似然/ELBO |
| 2 | ODE-RNN 2019 | 有 | 无（隐式重构） | 半（GRU 凸组合） | 半（h^-,x） | 有 | 无 | ELBO/一步预测 |
| 3 | GRU-ODE-Bayes 2019 | 有 | 隐式（似然） | 否（GRU） | 有（f_prep） | 有 | 有 | NegLL+KL |
| 4 | Neural CDE 2020 | 无 | 无 | 无 | 无 | 有（路径通道） | 无 | 监督/adjoint |
| 5 | Neural Flows 2021 | 有 | 无 | 无 | 无 | 有（flow t） | 无 | ELBO/密度 |
| 6 | mTAN 2021 | 无 | 无 | 无 | 无 | 有（嵌入/权重） | 有 | ELBO+CE |
| 7 | LTC 2021 | 有 | 无 | 无 | 无 | 有（步长） | 无 | 监督/模仿 |
| 8 | CfC 2022 | 有 | 无 | 无（凸插值） | 无 | 有（闭式非线性） | 无 | 监督/RL |
| 9 | GRU-D 2018 | 无 | 无 | 无（衰减） | 无 | 有（δ 门） | 有 | 监督 |
| 10 | NJ-ODE 2021 | 有 | 无（loss 中） | 无 | 半（x only） | 有（f 输入） | 无 | 回归/滤波证明 |
| 11 | Vid-ODE 2021 | 有 | 无 | 无 | 半（ConvGRU） | 有 | 无 | 生成/对抗 |
| 12 | Consistency 2023 | 无 | 无 | 无 | 无 | 半（生成时间） | 无 | 分数匹配+蒸馏 |
| 13 | Flow map 2025 | 无 | 无 | 无 | 无 | 有（t-s） | 无 | 自蒸馏+半群罚 |

## 5. 结论：对目标公式的判定

- **最接近的 3 个先例**：
  1. **ODE-RNN / Latent ODE（2019）**：连续流 + 观测跳变，位置与目标公式的 Φ 与 B 逐位对应；差在残差形式。
  2. **GRU-ODE-Bayes（2019）**：预测-更新两段式、明确自比 Kalman filter、连续时间 + 观测空间似然，最接近目标的"语义"。
  3. **NJ-ODE（2021）** 或 **CfC（2022）**：前者把 Delta 显式进向量场并给出滤波收敛保证；后者把真实 Delta 进闭式转移。二者分别占住 E 轴的两种极端实现。
- **最致命的 3 个负面证据**：
  1. KalmanNet/EKF 的更新式 `x̂_{t|t}=x̂_{t|t-1}+K_t Δy_t, Δy_t=y_t-h(x̂_{t|t-1})` 与目标的 `h=h^-+B(h^-,r)` 结构同构，且 K 已由网络学（2022，IEEE TSP）；"残差 + 加性学习增益"不是新结构。
  2. ODE-RNN（2019）已经给出"连续流 + 观测时刻更新"的完整算法；目标公式的骨架在 2019 年即存在。
  3. 含观测注入的复合映射一般不满足半群（§2 论证 + CfC 代数），Neural Flows 又只列了不含半群的条件——因此"目标 Φ_Delta 具有采样细化一致性/半群"这类主张在文献与数学上都站不住，除非把命题弱化为"区间内无观测时"或"给出误差阶/极限一致性"。
- **有没有与目标公式逐字对应的已有工作**：没有找到以完全相同符号/完全相同三项结构（真实 Delta 流 + 观测空间残差 + 学习式加性增益）写出的论文。但结构组件全部各自存在：Φ_Delta（Neural ODE/Latent ODE/Neural Flows/CfC）、r=z-D(h^-) 与加性增益（EKF/KalmanNet）、二者组合的动机（GRU-ODE-Bayes 已自称 filtering/Kalman 式）。因此合理的判定是：**组合是新的，思想不是新的**；可辩护的新颖性只能落在"具体组合 + 可证明的性质（误差界/极限一致性/观测间半群）"上，且必须正面处理 §3 的 N4。
- **唯一还有空间的轴**：F（稀疏/选择性/局部化）。但该轴在文献里通常由 mask/attention 实现（GRU-ODE-Bayes、mTAN、GRU-D），若目标只是在 B 上加稀疏性，属于组合式增量而非新机制。
- 需要**谨慎**的表述：不要把"Delta 进入公式"写成"连续时间建模"（N2）；不要把"数值积分"写成"精确半群"（N3/N4）；不要把"预测误差修正"当成新发现（N1/N5）。
