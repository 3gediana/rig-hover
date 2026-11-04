# 06 · 对抗式检索：目标公式的先例覆盖、逐字对应与退化反例

> 待检验目标公式（用户候选结构）：
>
>     h- = Phi_Delta(h)          # 带真实时间间隔 Delta 的状态转移（先验）
>     r  = z - D(h-)             # 观测空间残差 / 新息
>     h  = h- + B(h-, r)         # 残差式修正（学习式增益）
>
> 本文件的任务不是综述，是尽最大努力证明"这四行前人已经做过"，并找出它必然退化的证据。
> 结论：**做过，而且有一半是逐字做过的**。检索/核对日期 2026-11-03，仅网络检索与原文核对，未读本地任何旧文档。

---

## 0. 结论先行（TL;DR）

1. **A/B/C/D 四轴在 1960 年就定型**：Kalman (1960) 的新息更新 x̂(k|k) = x̂(k|k-1) + K_k (z_k - H_k x̂(k|k-1)) 与目标公式逐项同构；Luenberger 观测器是 B 取常增益的特例；Widrow–Hoff (1960) 的 delta rule 是 D 线性、Phi=I 的特例。**这四行不是新结构，是"带学习式增益的观测器"。**
2. **E 轴（真实时间间隔 Δ）同样被覆盖**：连续-离散 Kalman 滤波 / CRU (ICML 2022) 在同一对式子里同时给出 μ⁻_t = exp(A(t-τ(t))) μ⁺ 与 μ⁺_t = μ⁻_t + K_t(y_t - H μ⁻_t)；GRU-D (2018) 把 δ_t 塞进状态衰减；Mamba (2024) 用 Ā = exp(ΔA) 离散化；EFLA (2025) 直接把 delta-rule 解释成连续时间系统的 Euler 离散并给出精确 flow。**连"增益随 Δ 变"都被 CRU 覆盖**（K_t 依赖 Σ⁻_t，而 Σ⁻_t 依赖 Δ）。
3. **学习式 D 与学习式 B** 也被覆盖：TTT (2024/25) 令 h 本身是一个模型、更新是一步自监督梯度；Titans (2025) 把"衰减 + 梯度写入"合成一步；KalmanNet (2022) 用 RNN 从新息直接学出增益。
4. **因此：单凭这四行公式，新颖性为零。** 还能站住的只有 (i) 额外数学约束 + 定理/反例边界（§6）、(ii) F 轴（稀疏/选择性/局部化，目标公式目前完全没有）、(iii) 用途级组合。
5. 未找到"逐字完全相同 + 同一用途（机器人策略的在线状态估计 + RL 回报训练）"的工作。但那是**用途级空隙，不是公式级空隙**；审稿人一句 "this is a Kalman filter with a learned gain" 就能挡回来。

---

## 1. 检索方法与检索词

主工具 tavily_search（search_depth=advanced，5–10 条）+ tavily_extract 抓 arxiv /html/、官方 proceedings PDF、讲义 PDF，逐条核对标题/作者/年份/公式。每条引用链接均实际拉取；拉不到的标 [未核实]。web_search 仅作补充（本稿实际未需要）。

### 1.1 八个必查方向 × 实际检索串 × 命中

| 方向 | 实际检索串（节选） | 命中情况 |
|---|---|---|
| 1 delta rule / DeltaNet / 线性注意力 | "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"；DeltaNet delta rule state update v - h k error correction linear attention；Gated Delta Networks improved memory control delta rule；DeltaNet linear transformer fast weight delta rule Schlag Irie 2021 | 直接命中，**原式逐字**；再追到 EFLA (2025) 连续时间版 |
| 2 TTT / fast weights / 动态评估 | Learning to Learn at Test Time RNNs with Expressive Hidden States TTT layers Sun 2024 gradient descent hidden state；Using Fast Weights to Attend to the Recent Past Ba Hinton 2016 fast weight outer product error correction；Titans Learning to Memorize at Test Time surprise metric gradient memory decay adaptive forgetting | 全部命中，h ← h - η∇ℓ 逐字拿到 |
| 3 控制：predictor-corrector / MRAC / Luenberger | Kalman filter 1960 A New Approach to Linear Filtering and Prediction Problems predict correct update K(z - H x)；Luenberger observer state estimate correction residual error gain y - C x_hat；neural observer state correction gain learned filter generalization deep learning；Learning State Observers for Recurrent Neural Network Models | 全部命中；控制侧无人做"学习式 B + 神经 D"，但 **KalmanNet 已经做了** |
| 4 信息滤波 | information filter information matrix Y = P inverse additive update y + H transpose R inverse z | 命中，信息形式逐字 |
| 5 PSR / 预测误差唯一学习信号 | Predictive State Representations Littman Sutton Singh 2002 predictive state update；predictive state representation update predictive error Bayes filter test outcome probability | 定义级命中；逐字更新式未取到（仅定义级） |
| 6 神经科学：预测编码 / 主动推断 | predictive coding discrete time inference step prediction error update equations Rao Ballard Bogacz tutorial；Bogacz 2017 tutorial predictive coding free energy precision weighted prediction error hierarchical；active inference discrete time belief update prediction error variational filtering Friston | 命中；**"一次推理步 = 一次加性残差修正"有原文表述** |
| 7 通用措辞 | predict then correct neural hidden state observation residual update learned gain；innovation driven recurrent neural network state update learned observer gain；neural observer state correction gain learned filter generalization deep learning；time-aware gated delta rule irregular sampling timestamps recurrent state update | 通用措辞**没有**命中同构工作；命中 IRNN [未核实]、learned observers [仅摘要级] |
| 8 半群/flow 一致性 + 观测修正 | semigroup property flow map composition error continuous-discrete filtering sampling refinement；continuous-time delta rule linear attention with elapsed time delta t state transition | 半群条件在 flow-map / consistency model 里逐字命中；**但没有任何工作把它与观测修正结合**——这是真缺口 |

### 1.2 明确"搜过但没找到"的清单（负面检索结果）

- time-aware gated delta rule irregular sampling timestamps recurrent state update：只回到 Gated DeltaNet 与 mTAND（不规则采样注意力），**没有**"把 Δ 作为 delta-rule 状态转移入参"的专门工作（除 §2.13 EFLA 的连续时间版）。
- predict then correct neural hidden state observation residual update learned gain：返回的是通用 RNN 教材/HMM 页面，**无同构论文**。
- neural observer state correction gain learned filter generalization deep learning：返回 learned observer（Bonassi 等）与 Learning Robust State Observers using Neural ODEs，均为**控制用途、非 RL**，且未取到逐字方程 → [仅摘要级]。
- "IRNN: Innovation-driven Recurrent Neural Network"（ResearchGate 命中，标题与目标公式语义最近）：**ResearchGate 拒绝抓取，正文与方程均未取得 → [未核实]**，本稿不作为证据使用，但建议主代理补查（若真存在，是第四条致命先例）。
- 没有任何论文把"半群一致性约束 + 观测残差修正"合起来写。

---

## 2. 逐篇条目

### 1. A New Approach to Linear Filtering and Prediction Problems
- 作者 / 年份 / 会议或期刊 / 链接：Rudolf E. Kalman / 1960 / Journal of Basic Engineering 82(1):35–45, ASME / DOI 10.1115/1.3662552；方程逐字核对自 Wikipedia 条目 <https://en.wikipedia.org/wiki/Kalman_filter>（该页引用即上述文献）
- 核心更新方程（逐字）：
  - 先验（predict）：x̂(k|k-1) = F_k x̂(k-1|k-1)（含控制项 + B_k u_k）
  - 新息（innovation / pre-fit residual）：ỹ_k = z_k - H_k x̂(k|k-1)
  - 增益：K_k = P(k|k-1) H_kᵀ (H_k P(k|k-1) H_kᵀ + R_k)⁻¹
  - 后验（update）：x̂(k|k) = x̂(k|k-1) + K_k ỹ_k
- Q1 / Q2 / Q3：Q1 **是**（每次观测都用新息修正）；Q2 **秒**（F_k = F(Δt_k)，Δ 经状态转移矩阵进入，连续-离散形式为 F = exp(AΔt)）；Q3 通用在线状态估计，**不为规划服务**。
- A 有，x̂(k|k-1) 就是 h-。 B 有，ỹ_k = z_k - H_k h- 与 r = z - D(h-) 同构，D = H_k（线性）。 C 有，h = h- + K_k ỹ_k 是纯加性。 D 有，K_k 依赖 P(k|k-1)（即 h- 的不确定性），残差显式入式。 E 有（连续-离散形式），但 Δ 只进 F_k，不进 K_k 的显式函数式。 F 无稀疏/选择性。 G 无训练；K_k 由高斯假设解析给出（闭式最优），非学习。
- 任务与训练目标：线性高斯最优滤波；最小均方误差（MMSE），无梯度训练。
- 最接近的先例：Wiener 滤波（频域）；后续 Kailath (1968) 的 innovations approach 把"新息是唯一携带新信息的量"形式化（<http://web.stanford.edu/~tkailath/cgi-bin/PubsList.php>，条目已核对存在）。
- 新颖性判断：**严重削弱**。目标公式去掉"学习式 D/B"后就是 Kalman 的 EKF 推广；K_k 换成神经网络即 KalmanNet。这不是"相似"，是**同一结构在不同年代的两次发明**。

### 2. Observing the State of a Linear System（Luenberger 观测器）
- 作者 / 年份 / 会议或期刊 / 链接：David G. Luenberger / 1964（原始）；方程取自 A. Bemporad《Automatic Control 1》讲义 p.5，逐字核对 <https://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/06b-estimator.pdf>。原文（IEEE Trans. Military Electronics, 1964）**未逐字核实** [未核实]
- 核心更新方程（逐字）：x̂(k+1) = A x̂(k) + B u(k) + L(y(k) - C x̂(k))，L ∈ R^(n×p) 为观测器增益
- Q1 / Q2 / Q3：Q1 **是**（用输出估计误差 y - Cx̂ 反馈修正）；Q2 **观测索引**（讲义用 k，Δ 隐含在 A 里）；Q3 为状态反馈控制/NMPC 服务的**通用观测器**。
- A 有（A x̂(k)）。 B 有（y - C x̂，观测空间）。 C 有（加性 +L(...)）。 D 部分：L 是常数矩阵，只依赖 r，**不依赖 h-**——这是目标公式与它最实质的差异。 E 无显式 Δ（离散索引）。 F 无。 G 无训练，L 由极点配置/最优控制设计。
- 任务与训练目标：使估计误差 e = x̂ - x 渐近稳定（A - LC 的谱半径 < 1）；无学习。
- 最接近的先例：Kalman (1960)；Luenberger 的可观测性判据。
- 新颖性判断：**削弱**。它证明"加性残差修正"在 1960s 已是标准范式；目标公式的 B(h-, r) 相对它的全部增量只是"增益可以是 (h-, r) 的非线性函数"——而这正是 KalmanNet 与 TTT 已做的。

### 3. Information Formulation of the UDU Kalman Filter（信息滤波）
- 作者 / 年份 / 会议或期刊 / 链接：Christopher D'Souza, Renato Zanetti / 2018（UT Austin NEAR 技术报告）/ <https://sites.utexas.edu/near/wp-content/uploads/sites/6030/2018/05/UDU_Information.pdf>（作者与题名从 PDF 首页逐字核对）
- 核心更新方程（逐字，式 (9)(10)）：P_k⁻¹ = P̄_k⁻¹ + H_kᵀ R_k⁻¹ H_k；K_k = P_k H_kᵀ R_k⁻¹。等价的信息向量更新为 y_k = ȳ_k + H_kᵀ R_k⁻¹ z_k（**该式本稿未逐字取到，按标准信息滤波写，标 [结构级]**）
- Q1 / Q2 / Q3：Q1 是；Q2 秒（时间传播步同样过 Φ(Δ)）；Q3 通用估计。
- A 有。 B 隐含（信息形式把残差吸收进信息向量）。 C **有且被强调**：信息形式的更新是**纯加法**（+ HᵀR⁻¹H），正是目标公式 h = h- + B 的加法性的极端体现。 D 无（HᵀR⁻¹ 不依赖 h-）。 E 有（时间传播）。 F 无。 G 无训练。
- 任务与训练目标：数值稳定的滤波（P=∞ 时也定义良好）；MMSE。
- 最接近的先例：Kalman (1960)、Kailath (1968)。
- 新颖性判断：**削弱（对 C 轴致命）**。"修正项以加法进入"在估计理论里不是设计选择而是信息论的必然结果；目标公式把它当作结构特征，学术上不成立。

### 4. Modeling Irregular Time Series with Continuous Recurrent Units ★ 最致命
- 作者 / 年份 / 会议或期刊 / 链接：Mona Schirmer 等（Bosch Center for AI；完整作者列表 [部分核实]，PDF 首页可见 Mona Schirmer 为通讯作者）/ ICML 2022 / PMLR 162 / <https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf>
- 核心更新方程（逐字，式 (8)(11)(13)）：
  - 预测：μ⁻_t = exp(A (t - τ(t))) μ⁺_τ(t)，Δ := t - τ(t) 为**真实时间间隔**（秒），τ(t) 是上一次观测时刻
  - 新息与修正：μ⁺_t = μ⁻_t + K_t (y_t - H μ⁻_t)
  - 增益：K_t = Σ⁻_t Hᵀ (H Σ⁻_t Hᵀ + Σ_obs,t)⁻¹
  - 协方差：Σ⁻_t = exp(AΔ) Σ⁺ exp(AΔ)ᵀ + ∫ exp(A(t-s)) GQGᵀ exp(A(t-s))ᵀ ds
- Q1 / Q2 / Q3：Q1 **是**（新息显式）；Q2 **秒**（Δ 进 exp(AΔ)，**并且通过 Σ⁻_t 进增益 K_t**）；Q3 不规则时间序列的插值/外推（编码-解码框架），**不是规划**。
- A 有，μ⁻_t 字面上就是 h-。 B 有，y_t - H μ⁻_t 字面上就是 r = z - D(h-)，D = H。 C 有，字面加性。 D 有，K_t 依赖 Σ⁻_t（h- 的二阶矩），r 显式。 E **有，且这是它的核心卖点**。 F 无（全维高斯）。 G 训练目标是 ELBO/重构（编码-解码），f_θ 与 g_ϕ 端到端学。
- 任务与训练目标：不规则采样时间序列的 ELBO 训练（变分自编码 + 连续-离散卡尔曼递推）。
- 最接近的先例：Kalman–Bucy 连续-离散滤波；ODE-RNN / Latent ODE (Rubanova 2019)；其 Sec. 3.2 明确写 "a CRU is a recurrent neural architecture that uses the predict and update step of a continuous-discrete Kalman filter"。
- 新颖性判断：**致命**。目标公式的 A–E 五轴在 CRU 一篇里**全中**，且 Δ 是真实秒、且增益随 Δ 变、且 D 侧的观测映射 H/f_θ 是网络参数化。差别只剩：(i) D 是非线性神经解码器（TTT 已做）、(ii) B 是任意非线性网络而非由协方差闭式给出（KalmanNet 已做）、(iii) 用途是策略而非插值。

### 5. KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics
- 作者 / 年份 / 会议或期刊 / 链接：Guy Revach, Nir Shlezinger, Xiaoyang Ni, Adria Lopez Escoriza, Ruud J. G. van Sloun, Yonina C. Eldar / 2022 / IEEE Transactions on Signal Processing / PDF: <https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf>（作者由 Weizmann 出版页核对 <https://weizmann.elsevierpure.com/en/publications/kalmannet-neural-network-aided-kalman-filtering-for-partially-kno>）
- 核心更新方程（逐字，式 (5a)(7)）：x̂(t|t) = x̂(t|t-1) + K_t · Δy_t；Δy_t = y_t - ŷ(t|t-1)（原文："the innovation ... the only term that depends on the observed data"）。K_t 不用协方差算，而是**用一个 GRU 从 Δy_t 及特征回归出来**。
- Q1 / Q2 / Q3：Q1 **是**；Q2 离散索引（Δ 隐含在 f(·) 里）；Q3 通用在线状态估计（含非线性/部分已知动力学），**不是规划**。
- A 有。 B 有。 C 有。 D **有且正是其贡献**：K_t 由 (Δy_t, 历史) 学出，实质上就是 B(h-, r)。 E 无显式 Δ（这是它与目标公式的主要剩余差别）。 F 无。 G 端到端监督训练，损失是状态/观测的 MSE（**不是** RL）。
- 任务与训练目标：在动力学模型部分未知时用数据学增益以逼近/超过 EKF；监督学习。
- 最接近的先例：EKF；Recurrent Kalman Networks（Becker et al., ICML 2019，<https://arxiv.org/abs/1905.07357>，用 Kalman 结构做因子化变分推断，[仅摘要级]）。
- 新颖性判断：**严重削弱**。目标公式相对 KalmanNet 的唯一增量是"增益同时吃 h- 与 r 并且 Δ 是显式的"。h- 与 r 共生（r 由 h- 算出），所以"同时依赖"几乎平凡满足。**把 Kalman 增益换成神经网络，是 2022 年的论文标题。**

### 6. Latent ODEs / ODE-RNN for Irregularly-Sampled Time Series
- 作者 / 年份 / 会议或期刊 / 链接：Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud / 2019 / NeurIPS 2019 / <https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf>（作者与题名从 PDF 首页逐字核对）
- 核心更新方程：论文自述（逐字）："We generalize RNNs to have continuous-time hidden dynamics defined by ordinary differential equations (ODEs), a model we call ODE-RNNs." 递推结构为：先 h-_t = ODESolve(f_θ, h(t-1), (t(t-1), t_t))，再用观测更新 h_t = GRUCell(h-_t, x_t)。**[结构级转述：本稿未取到逐字式，PDF 文本抽取未返回公式行；Schirmer 等 ICML 2022 的 related work 亦有同义转述，已核对]**
- Q1 / Q2 / Q3：Q1 **是**（观测时刻做一次隐状态更新，含残差语义）；Q2 **秒**（ODE 在 Δt 上积分）；Q3 不规则时间序列的概率建模（可外推），**不服务规划**。
- A 有（h- = ODE 积分结果）。 B 部分/无：更新是 GRUCell，r = z - D(h-) 只在 D 为恒等且有显式重构时才可见，**没有把残差写成观测空间的显式减式**。 C 部分：GRU 是门控混合，可写成凸组合形式，不是纯加性。 D 部分（GRU 同时吃 h- 与 x_t）。 E **有**。 F 无。 G ELBO（变分自编码）。
- 任务与训练目标：不规则采样序列的重构/外推，ELBO。
- 最接近的先例：Neural ODE (Chen et al. 2018)；Kalman–Bucy 滤波。
- 新颖性判断：**削弱**。它证明"Φ_Δ（连续时间）+ 观测时刻更新"这个骨架在 2019 年已是标准做法；目标公式相对它的增量只是把 GRU 换成"显式观测残差 + 加性增益"，而后者正是 Kalman/DeltaNet。

### 7. Recurrent Neural Networks for Multivariate Time Series with Missing Values（GRU-D）
- 作者 / 年份 / 会议或期刊 / 链接：Zhengping Che, Sanjay Purushotham, Kyunghyun Cho, David Sontag, Yan Liu / 2018 / Scientific Reports 8:6085 / <https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216>（作者由 PubMed 条目与多篇引文核对）
- 核心更新方程（逐字，式 (10)）：γ_t = exp{ -max(0, W_γ δ_t + b_γ) }，δ_t 是距上次观测的**时间间隔**；γ_t 用于衰减输入与隐状态（h(t-1) 先乘 γ 再进 GRU）— 衰减的应用方式为 [结构级]
- Q1 / Q2 / Q3：Q1 部分（GRU 内部用观测，但不是"观测空间残差"）；Q2 **秒**（δ_t 显式、单调、可学习）；Q3 分类/回归的缺失值鲁棒建模，**不服务规划**。
- A 部分：γ ⊙ h(t-1) 就是 Φ_Δ(h) 的一个标量对角特例。 B 无。 C 无显式加性残差（GRU 门控）。 D 无。 E **有，且是最直接的先例**：Δ 通过 exp(-W_γ Δ + b) 进入状态转移。 F 有（逐维 decay）。 G 端到端监督（交叉熵/MSE）。
- 任务与训练目标：临床时间序列的预测，端到端监督。
- 最接近的先例：Phased LSTM、T-LSTM（时间衰减 LSTM 家族）；后续被 S4/SSM 的 Ā=exp(ΔA) 一般化。
- 新颖性判断：**削弱 E 轴**。"Φ_Δ(h) = exp(-WΔ) ⊙ h，Δ 为真实秒"在 2018 年就是标准操作。目标公式若声称"Δ 进入状态转移"是新贡献，直接被这一篇挡回。

### 8. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- 作者 / 年份 / 会议或期刊 / 链接：Albert Gu, Tri Dao / 2023 arXiv, COLM 2024 / arXiv:2312.00752 / <https://arxiv.org/html/2312.00752>（作者由官方仓库 bibtex 核对 <https://github.com/state-spaces/mamba>）
- 核心更新方程（逐字，式 (4)）：Ā = exp(ΔA)，B̄ = (ΔA)⁻¹(exp(ΔA) - I) · ΔB；递推 h_t = Ā h(t-1) + B̄ x_t
- Q1 / Q2 / Q3：Q1 **否**（没有观测空间残差，输入是 x_t 不是观测 z）；Q2 **秒（名义上）**：Δ 是步长参数，Mamba 让它**输入依赖**（selective），但**不是真实采样间隔**；Q3 序列建模，**不服务规划**。
- A 有（Ā h(t-1)）。 B 无。 C 无（+B̄x 不是残差修正）。 D 无。 E **有（形式）**：Δ 进离散化。 F **有**：Δ 随输入变 = 选择性。 G 下一 token 交叉熵（自回归）。
- 任务与训练目标：语言建模（next-token CE）。
- 最接近的先例：S4 (Gu et al. 2021) 的 (Δ,A,B,C) 离散化；GLA/RetNet 的门控线性注意力。
- 新颖性判断：**削弱 E 轴**。"Φ_Δ = exp(ΔA)、Δ 是标量时间参数"是 SSM 的教科书第一步。真正的差异在"Δ 是真实秒还是 token 步长"——而这恰好被 CRU/GRU-D 补上了。

### 9. Adaptive Switching Circuits（delta rule / LMS）
- 作者 / 年份 / 会议或期刊 / 链接：Bernard Widrow, Marcian E. Hoff / 1960 / IRE WESCON Convention Record, Part 4, 96–104 / 书目核对 <https://www.bibsonomy.org/bibtex/2ace34c5debd5abc08e714f8ff1030b3>；方程从讲义逐字摘录 <https://philadelphia.edu.jo/academics/amahasneh/uploads/7th_lecture_WIDROW_HOFF_LEARNING_ALGORITHM_s1_21_22.pdf>（E = ½(d-y)²，y = wᵀx，w = w - α∇E）。原文 PDF 未取得 [原文未核实]
- 核心更新方程：由讲义逐字行推出 w ← w + α (d - wᵀx) x（线性输出，r = d - wᵀx）
- Q1 / Q2 / Q3：Q1 **是**（每样本用预测误差更新）；Q2 **索引**（样本序，无时间）；Q3 通用在线学习/自适应滤波，**不服务规划**。
- A 有（w 即 h-）。 B 有（d - wᵀx，D(h) = hᵀx）。 C 有（加性）。 D 无（B = α r x，只依赖 r 与输入）。 E 无。 F 无。 G MSE（监督）。
- 任务与训练目标：自适应线性分类/滤波；MSE 梯度下降。
- 最接近的先例：Rosenblatt 感知机（同年代）；后被 Schlag et al. (2021) 明确称为 "the famous error-correcting delta-rule (Widrow & Hoff, 1960)"（**该句已从 Schlag PDF 逐字核对**）。
- 新颖性判断：**致命（对 B/C/D）**。取 D(h) = hᵀx、B = α r x、Φ = I，目标公式**精确退化为 delta rule**。任何"预测误差修正隐状态"的表述都绕不开 1960 年。

### 10. Linear Transformers Are Secretly Fast Weight Programmers（DeltaNet 原版）
- 作者 / 年份 / 会议或期刊 / 链接：Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber / 2021 / ICML 2021, PMLR 139 / <https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf>
- 核心更新方程（逐字，式 (20)–(22)）：
  - v̄(i) = W(i-1) φ(k(i))  ← 这就是 D(h-)
  - β(i) = σ(W_β x(i))  ← 学习式增益标量
  - v_new(i) = β(i) v(i) + (1 - β(i)) v̄(i)  ← 凸组合（残差）
  - 写入为外积（ΔW = v_new φ(k)ᵀ；该外积式本稿未逐字取到，但其等价残差形式在 §2.11/§2.12 中被逐字引用为 S_t = S(t-1)(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ）
- Q1 / Q2 / Q3：Q1 **是**（先取回旧值 v̄ 再按差写入）；Q2 **索引**（token 步，无 Δ）；Q3 序列建模（关联召回），**不服务规划**。
- A 有（v̄ = 预测）。 B 有（v(i) - v̄(i) 是观测空间残差）。 C 有（凸组合 = 加性）。 D 有（β 由 x(i) 学出，v̄ 依赖 W）。 E 无。 F 有（β 逐键选择性）。 G 监督（语言建模/翻译）。
- 任务与训练目标：合成检索任务 + WMT + 语言建模；端到端。
- 最接近的先例：Widrow–Hoff (1960)；Schmidhuber 1991/1992 的 fast weight programmer；其 2021 另一篇 Learning to Control Fast-Weight Memories。
- 新颖性判断：**致命**。这一篇是"误差修正式状态写入"在神经网络里的**直系祖先**，且它明确把 v̄ 写成"当前 key 对应的旧值"。

### 11. Parallelizing Linear Transformers with the Delta Rule over Sequence Length
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim / 2024 / NeurIPS 2024 / arXiv:2406.06484 / <https://arxiv.org/html/2406.06484v3>（作者与题名从作者本人 poster PDF 首页逐字核对）
- 核心更新方程（逐字，作者博客同式）：S_t = S(t-1) - β_t (S(t-1) k_t - v_t) k_tᵀ = S(t-1) - β_t S(t-1) k_t k_tᵀ + β_t v_t k_tᵀ，等价 S_t = S(t-1)(I - β_t k_t k_tᵀ) + β_t v_t k_tᵀ（后者逐字见 GatedDeltaNet 式 (9) 上方）。博客 <https://sustcsonglin.github.io/blog/2024/deltanet-1>
- Q1 / Q2 / Q3：Q1 **是**（原博客："minimizes MSE at each step"）；Q2 **索引**；Q3 线性注意力训练加速 / 长上下文，**不服务规划**。
- A 有（S(t-1)）。 B 有（S(t-1)k_t - v_t 就是 D(h-) - z 的负号）。 C 有。 D 有（β_t 输入依赖，残差显式）。 E 无。 F 有（β 逐 token 选择；Householder 结构）。 G 语言建模 CE（**不是**重构、不是 RL）。
- 任务与训练目标：1.3B 模型 / 100B token 语言建模；chunkwise 并行（WY 表示）。
- 最接近的先例：Schlag 2021；GLA (Yang 2024)。
- 新颖性判断：**致命**。这是目标公式在"隐状态是矩阵"情形下的**逐字实现**：r = z - D(h-) 就是 v_t - S(t-1)k_t，B = β_t r k_tᵀ。任何把目标公式写成"残差修正"的贡献声明都正面撞上这一篇。

### 12. Gated Delta Networks: Improving Mamba2 with Delta Rule ★ 第二致命
- 作者 / 年份 / 会议或期刊 / 链接：Songlin Yang, Jan Kautz, Ali Hatamizadeh / 2025 / ICLR 2025 / arXiv:2412.06464 / <https://arxiv.org/html/2412.06464v3>（作者由官方仓库 bibtex 核对 <https://github.com/NVlabs/GatedDeltaNet>）
- 核心更新方程（逐字，式 (10)）：S_t = S(t-1) [ α_t (I - β_t k_t k_tᵀ) ] + β_t v_t k_tᵀ，α_t ∈ (0,1) 为**数据依赖的状态衰减门**
- Q1 / Q2 / Q3：Q1 **是**；Q2 **索引**（α_t 是学出的门，**不是**真实 Δ）；Q3 序列建模，**不服务规划**。
- A 有，且**是可乘转移**：h- = α_t S(t-1)（更准确 α_t(I-β_t k kᵀ)S(t-1)）。 B 有（v_t - α_t S(t-1)k_t，见其 Mamba2 对比段落）。 C 有。 D 有（α_t、β_t 都输入依赖）。 E **无**（这是它与目标公式唯一的结构性差别：Δ 被替换成学出来的 α）。 F **有**（双重选择性：α 全局擦除 + β 局部精确改写）。 G 语言建模 CE。
- 任务与训练目标：语言建模 / 常识推理 / 长上下文检索；端到端 CE。
- 最接近的先例：Mamba2 的 S_t = α_t S(t-1) + v_t k_tᵀ；DeltaNet；其 §3.1 用 online learning 框架（Liu et al. 2024）导出闭式。
- 新颖性判断：**致命**。把 α_t 解释成 Φ_Δ 的一个特例，目标公式与式 (10) **完全同构**；唯一差别是 α_t 不是 exp(-λΔ)。而"让 α 由真实 Δ 生成"恰好是 GRU-D / SSM 离散化 / EFLA 都做过的事。**两条文献一拼，目标公式就被完全覆盖。**

### 13. Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics（EFLA）
- 作者 / 年份 / 会议或期刊 / 链接：J. Lei 等（首作者由 arXiv 检索摘要 "by J Lei · 2025" 核对；完整作者列表未取到，标 [部分核实]）/ 2025 / arXiv:2512.12602（v5）/ <https://arxiv.org/html/2512.12602v5>
- 核心更新方程：**[仅摘要级]** 论文自述（逐字）："we show that the delta-rule update can be interpreted as an explicit Euler discretization of an underlying continuous-time system. EFLA replaces this first-order update with the exact closed-form flow ... By exploiting the rank-1 structure of the dynamics matrix, both the matrix exponential and the input integral collapse to a simple update." 逐字闭式未取到（HTML 公式未随正文返回）→ 不臆写。
- Q1 / Q2 / Q3：Q1 **是**（继承 delta-rule）；Q2 **有 Δ 的形式**：在零阶保持（ZOH）假设下 "the current key and value are treated as fixed within each token interval"——即 Δ 是 token 间隔；Q3 序列建模，**不服务规划**。
- A/B/C/D 均继承 delta-rule（有）。 E **有形式上的 Δ**（且是连续时间的精确 flow，不只是 Euler 一步）。 F 有（rank-1 结构）。 G 语言建模 CE。
- 任务与训练目标：语言建模 + 鲁棒性测试；"removes the Euler discretization error of the delta-rule dynamics without introducing additional parameters"。
- 最接近的先例：DeltaNet 家族；S4/Mamba 的 exp(ΔA)；Neural ODE。
- 新颖性判断：**严重削弱 E 轴**。它把"delta-rule 的连续时间版本 + 真实步长 Δ"明确做出来了。目标公式若声称"Φ_Δ 带真实 Δ 且残差修正在同一式里"，EFLA 是其最近的同族工作（差别：Δ 是 token 间隔而非传感器秒）。

### 14. Learning to (Learn at Test Time): RNNs with Expressive Hidden States（TTT）
- 作者 / 年份 / 会议或期刊 / 链接：Yu Sun, Xinhao Li, Karan Dalal 等 / 2024 arXiv，ICML 2025 (PMLR 267) / arXiv:2407.04620 / <https://arxiv.org/html/2407.04620v4>；PMLR: <https://proceedings.mlr.press/v267/sun25h.html>（作者由论文 Contributions 段逐字核对）
- 核心更新方程（逐字，式 (1)(2)）：z_t = f(x_t; W_t)；W_t = W(t-1) - η ∇ℓ(W(t-1); x_t)，ℓ 为自监督损失（如重构 x_t）
- Q1 / Q2 / Q3：Q1 **是**（每步用自监督预测误差的梯度修正隐状态）；Q2 **索引**（无 Δ）；Q3 序列建模，**不服务规划**。
- A 有（W(t-1)）。 B **有且是核心**：ℓ 若取重构/线性回归，∇ℓ ∝ (D(h-) - z)，即观测空间残差。 C 有（加性梯度步）。 D 有（∇ℓ 在 W(t-1) 处求值，同时依赖 h- 与 x_t）。 E 无。 F 有（mini-batch / 大梯度优先记忆："Our W remembers inputs that produce large gradients"）。 G 自监督（重构/掩码预测），**不是 RL**。
- 任务与训练目标：长上下文语言建模；隐状态即模型，测试时训练。
- 最接近的先例：fast weight programmer（Schmidhuber 1991/1992）；DeltaNet（线性 f + MSE 时与 delta rule 同构，作者博客明确说 DeltaNet "minimizes MSE at each step"）。
- 新颖性判断：**致命**。"h 是一个可微模型、更新是一步预测误差梯度"= 目标公式把 D 与 B 都换成神经网络后的一般形式。**目标公式是 TTT 的一个特例**（B 限制为一步梯度）。

### 15. Titans: Learning to Memorize at Test Time
- 作者 / 年份 / 会议或期刊 / 链接：Ali Behrouz, Peilin Zhong, Vahab Mirrokni / 2025 / NeurIPS 2025 / arXiv:2501.00663 / <https://arxiv.org/html/2501.00663v1>（作者由多条引用核对）
- 核心更新方程（逐字，式 (13)(14)(16)）：M_t = (1 - α_t) M(t-1) + S_t；S_t = η_t S(t-1) - θ_t ∇ℓ(M(t-1); x_t)；合并为 M_t = (1 - α_t) M(t-1) - θ_t ∇ℓ(M(t-1); x_t)
- Q1 / Q2 / Q3：Q1 **是**；Q2 **索引**（α_t 是学出的门，非真实 Δ）；Q3 序列建模 + 长时记忆，**不服务规划**。
- A 有，(1-α_t)M(t-1) 就是可乘转移。 B 有（∇ℓ 对关联记忆损失取，等价于预测误差）。 C 有。 D 有。 E 无。 F 有（surprise 驱动）。 G 自监督（关联记忆损失）+ 语言建模。
- 任务与训练目标：长上下文语言建模 / 常识推理 / NIAH（BABILong）。
- 最接近的先例：TTT (Sun et al. 2024)；Mamba2 的衰减门（论文自述其衰减机制是 "the generalization of forgetting mechanism in modern recurrent models"）。
- 新颖性判断：**致命**。(1-α_t)M(t-1) - θ_t∇ℓ 与目标公式同构，且额外给了"衰减 + 动量"两种形式（目标公式连动量都没有）。Titans 还自证 α 与 ∇ℓ 是两条独立可控通道——比目标公式更细。

### 16. Using Fast Weights to Attend to the Recent Past
- 作者 / 年份 / 会议或期刊 / 链接：Jimmy Ba, Geoffrey E. Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu / 2016 / NeurIPS 2016 / <https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past>；方程逐字取自 <https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf>
- 核心更新方程（逐字）：A(t) = λ A(t-1) + η h(t) h(t)ᵀ
- Q1 / Q2 / Q3：Q1 **否**（纯 Hebbian 外积，非误差修正）；Q2 **索引**；Q3 近期记忆的注意力，**不服务规划**。
- A 有（λ A(t-1)）。 B 无。 C 有（加性）。 D 无。 E 无。 F 无。 G 监督（分类/强化任务）。
- 任务与训练目标：用快权重做"近期过去"的注意力；分类与 RL 任务。
- 最接近的先例：Schmidhuber 1991/1992 fast weight controller；后继 Schlag 2021 把它改成误差修正。
- 新颖性判断：**轻微削弱（C/E）**。它给出"Φ 是可乘衰减 + 加性写入"的最简范例；但**没有残差**，所以它是目标公式的负面对照，说明"加性"本身不稀奇。

### 17. Predictive Coding（离散时间预测误差修正）
- 作者 / 年份 / 会议或期刊 / 链接：Rao & Ballard 1999 提出（经下述综述转述）；本稿逐字来源：Predictive Coding Theories of Cortical Function / 2021 arXiv:2112.10048 / <https://arxiv.org/html/2112.10048v3>。**[综述级：Rao & Ballard 1999 原文未逐字核实]**；另有 Bogacz 2017 教程（J. Math. Psych.，<https://pubmed.ncbi.nlm.nih.gov/28298703>，[仅摘要级]）
- 核心更新方程（逐字摘录，式 (2.9) 附近）：r_t 的更新为 ... + (k_1/σ_td²)(r_td,(t-1) - r̂(t-1)) - (k_1/2) g'(r̂(t-1))；且原文照写："This prediction is then subtracted from the lower-level representation I to generate the bottom-up error (I - f(U r)), which is filtered by the feedforward weights Uᵀ ... The neural response vector r is updated based on a weighted combination of the bottom-up prediction error (I - f(Ur)) and the top-down prediction error"
- Q1 / Q2 / Q3：Q1 **是**；Q2 **索引**（离散时间步）；Q3 感知/皮层功能建模，**不服务规划**。
- A 有（生成的预测 f(Ur)）。 B **有，且是教科书表述**：I - f(Ur) 是观测空间残差。 C 有（加权组合 = 加性）。 D 有。 E 无。 F 有（层级/局部误差）。 G 自由能（变分下界）。
- 任务与训练目标：解释皮层响应；最小化预测误差/自由能。
- 最接近的先例：Rao & Ballard (1999)、Friston 主动推断、Kalman 滤波（综述本身把二者并列）。
- 新颖性判断：**严重削弱（A–D）**。"一次推理步 = 一次加性残差修正"在计算神经科学里是**标准表述**，已有 25 年。目标公式若以"预测误差驱动隐状态修正"为卖点，与神经科学的主流建模语言完全重叠。

### 18. Predictive Representations of State（PSR）
- 作者 / 年份 / 会议或期刊 / 链接：Michael L. Littman, Richard S. Sutton, Satinder Singh / 2002 / NeurIPS 2001（NIPS 14, pp.1555–1561）/ 定义逐字核对 <https://en.wikipedia.org/wiki/Predictive_state_representation>
- 核心更新方程：**[仅定义级]** "PSR captures the state of a system as a vector of predictions for future tests"；更新由对测试结果的贝叶斯更新 + 归一化给出。**逐字更新式未取得 → 不臆写。**
- Q1 / Q2 / Q3：Q1 **是**（状态本身就是预测，观测后重投影）；Q2 **索引**（离散测试序列）；Q3 为规划/控制服务的预测表示（**Q3 与目标公式最接近的一篇**）。
- A 部分（状态 = 预测，等价于先验预测向量）。 B 部分（预测误差驱动投影，但不是显式 z - D(h-)）。 C 无（更新是贝叶斯重投影/归一化，不是加性）。 D 无。 E 无。 F 有（只保留核心测试集）。 G 一步预测（自监督式的预测概率）。
- 任务与训练目标：部分可观测系统的预测表示；用于规划。
- 最接近的先例：POMDP；后续 Learning PSRs（Singh et al. 2003, <https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICML03.pdf>）。
- 新颖性判断：**削弱（Q3 轴）**。它证明"状态 = 预测 + 观测后修正"在 2002 年已是为**规划**服务的正统框架；目标公式的"预测误差作为唯一学习信号"在此已有雏形。

### 19. Dreamer / RSSM（model-based RL 侧）
- 作者 / 年份 / 会议或期刊 / 链接：Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi / 2020 / ICLR 2020 / arXiv:1912.01603（作者由 papertalk / statwiki 条目核对）；方程从重述文献逐字取 <https://arxiv.org/html/2502.20168v1> 式 (8)
- 核心更新方程（该文重述 RSSM，逐字）：Encoder: z_t ~ q_φ(·|o_t)；Sequence model: x_t, y_t = SSM(z(t-1), a(t-1), c(t-1), x(t-1))；Dynamics: ẑ_t ~ p_φ(·|y_t)；Decoder: ô_t ~ p_φ(·|y_t, z_t)
- Q1 / Q2 / Q3：Q1 **是**（先验 ẑ_t 与后验 z_t 分离，后验吃观测）；Q2 **索引**（每环境步，非秒）；Q3 **model-based RL / 想象规划**——Q3 唯一正面对上的先例。
- A 有（y_t / 先验 ẑ_t）。 B 无显式观测残差（后验是学出的分布，不是 z - D(h-)）。 C 无（后验推断，非加性）。 D 有（后验同时依赖先验与观测）。 E 无。 F 有（离散/连续混合、类别 latent）。 G **ELBO + RL 回报**（双目标）。
- 任务与训练目标：latent imagination 上的 actor-critic；世界模型 ELBO + 回报。
- 最接近的先例：PlaNet（Hafner 2019）；World Models (Ha & Schmidhuber 2018)。
- 新颖性判断：**削弱 Q3**。"先验状态 + 观测后验修正 + 服务规划"在 model-based RL 里是范式。**但** Dreamer 的修正不是残差式加性，这是目标公式唯一可能站住的差异点（也是最脆弱的一点，因为 KalmanNet/TTT 已把残差式加性做进学习系统）。

### 20. Learning Flow Maps via Self-Distillation（半群一致性）
- 作者 / 年份 / 会议或期刊 / 链接：2025 / NeurIPS 2025 / arXiv:2505.18825 / <https://arxiv.org/html/2505.18825v2>（作者列表未取到 [部分核实]）
- 核心更新方程（逐字）："The semigroup condition states that any two jumps can be replaced by a single jump." 即 Φ_Δ1 ∘ Φ_Δ2 = Φ_(Δ1+Δ2)
- Q1 / Q2 / Q3：Q1 **否**（生成模型采样，无观测）；Q2 **有 Δ**（跳步长度）；Q3 生成式采样加速，**不服务规划**。
- A 有（Φ_s(x)）。 B 无。 C 无。 D 无。 E **有**（跳步 Δ，且要求半群一致）。 F 无。 G 自蒸馏（flow map matching）。
- 任务与训练目标：少步生成；自蒸馏。
- 最接近的先例：Consistency Models (Song et al. 2023)；Shortcut models。
- 新颖性判断：**削弱 E 轴（半群部分）**："Φ_Δ 必须满足半群一致性"是 2023–2025 生成模型的核心约束，且已有成熟训练法。**但**没找到任何工作把"半群一致性"与"观测残差修正"结合——这是目标公式**唯一真空中**的组合（见 §6 第 8 行）。

---

## 3. 候选先例总表（重合度 高/中/低）

| 候选先例 | 关键检索词 | 链接 | 重合度 | 还差什么 |
|---|---|---|---|---|
| CRU (ICML 2022) | continuous recurrent units irregular Kalman | https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf | **高（A–E 全中）** | B 是协方差闭式而非任意网络；用途是插值 |
| Kalman (1960) | Kalman filter predict correct innovation | https://en.wikipedia.org/wiki/Kalman_filter | **高（A–D）** | 无学习；Δ 只进 F |
| Gated DeltaNet (ICLR 2025) | gated delta rule delta net | https://arxiv.org/html/2412.06464v3 | **高（A–D、F）** | 无真实 Δ |
| DeltaNet / Yang 2024 | delta rule parallel linear transformer | https://arxiv.org/html/2406.06484v3 | **高（A–D、F）** | 同上 |
| Schlag 2021 | linear transformers fast weight programmers delta rule | https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf | **高（A–D）** | 同上 |
| TTT (ICML 2025) | test time training layers hidden state gradient | https://arxiv.org/html/2407.04620v4 | **高（A–D）** | 无 Δ；G 是自监督 |
| Titans (NeurIPS 2025) | titans memorization surprise gradient decay | https://arxiv.org/html/2501.00663v1 | **高（A–D、F）** | 无真实 Δ |
| KalmanNet (TSP 2022) | neural network aided Kalman filter gain RNN | https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf | **高（A–D，学习式 B）** | 无显式 Δ |
| GRU-D (2018) | time gap decay hidden state delta_t | https://pmc.ncbi.nlm.nih.gov/articles/PMC5904216 | **中（A、E）** | 无残差、无加性 |
| ODE-RNN / Latent ODE (2019) | latent ODE irregularly sampled continuous discrete Kalman | https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf | **中（A、E）** | 修正不是显式观测残差 |
| EFLA (2025) | continuous-time delta rule exact flow linear attention | https://arxiv.org/html/2512.12602v5 | **中–高（A–F，含 Δ）** | Δ 是 token 间隔；无观测空间 z |
| Mamba (2024) | selective state space discretization Delta Abar exp | https://arxiv.org/html/2312.00752 | **中（A、E、F）** | 无残差、无观测 |
| Widrow–Hoff (1960) | delta rule LMS Widrow Hoff weight update | https://www.bibsonomy.org/bibtex/2ace34c5debd5abc08e714f8ff1030b3 | **中–高（A–D，特例）** | 无 Δ、无选择性 |
| Luenberger 观测器 | Luenberger observer correction gain | https://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/06b-estimator.pdf | **中（A–C）** | B 不依赖 h- |
| 信息滤波 | information filter P inverse additive update | https://sites.utexas.edu/near/wp-content/uploads/sites/6030/2018/05/UDU_Information.pdf | **低–中（C 轴）** | 无残差形式 |
| 预测编码 | predictive coding prediction error update step | https://arxiv.org/html/2112.10048v3 | **中–高（A–D）** | 无 Δ、非参数化 |
| PSR (2002) | predictive state representation | https://en.wikipedia.org/wiki/Predictive_state_representation | **低–中（Q3、A）** | 更新非加性 |
| RSSM / Dreamer (2020) | RSSM prior posterior model based RL | https://arxiv.org/html/2502.20168v1 | **中（A、D、Q3、G）** | 修正非残差式 |
| Flow-map consistency (2025) | semigroup flow map consistency | https://arxiv.org/html/2505.18825v2 | **低（E 轴半群）** | 无观测 |
| Ba 2016 fast weights | fast weights outer product decay | https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past | **低（A、C）** | 无残差 |
| IRNN innovation-driven RNN | innovation driven recurrent neural network | https://www.researchgate.net/publication/391658338_IRNN_Innovation-driven_Recurrent_Neural_Network_for_Time-Series_Data_Modeling_and_Prediction **[未核实，抓取被拒]** | 未知 | 待主代理补查 |

---

## 4. 最致命的三条先例

### 4.1 CRU（ICML 2022）：Φ_Δ 与残差修正在同一对式子里
μ⁻_t = exp(A(t-τ(t)))μ⁺，μ⁺_t = μ⁻_t + K_t(y_t - Hμ⁻_t)。**这就是目标公式换符号**：Φ_Δ = exp(AΔ)、D = H、B = K_t·。
**不加额外约束时目标公式还剩什么？** 只剩两条：(i) D 与 B 是**任意非线性神经网络**（而 CRU 是线性的 H 与协方差闭式 K_t）；(ii) 用途是策略/RL 而非插值。第一条被 TTT + KalmanNet 补掉（非线性 D 与学习式 B 都已存在），第二条**是用途不是数学**。

### 4.2 Gated DeltaNet 式 (10)：可乘转移 + 加性残差修正，逐字同构
S_t = S(t-1)[α_t(I - β_t k_t k_tᵀ)] + β_t v_t k_tᵀ。把 α_t 读成 Φ_Δ 的标量特例、把 v_t - α_t S(t-1)k_t 读成 r、把 β_t r k_tᵀ 读成 B，则**四项逐一对上**，且 F 轴（选择性）它比目标公式强。
**还剩什么？** 只有"α_t 由真实 Δ 生成"这一条。而"让门/衰减由真实 Δ 生成"是 GRU-D (exp(-WΔ))、Mamba (Ā=exp(ΔA))、CRU (exp(AΔ))、EFLA（连续时间 flow）四篇的交集。**把两条文献一拼，目标公式被完全覆盖。**

### 4.3 TTT / Titans：h 是模型，更新是一步预测误差梯度
TTT 式 (2) W_t = W(t-1) - η∇ℓ(W(t-1); x_t)；Titans 式 (16) M_t = (1-α_t)M(t-1) - θ_t∇ℓ(M(t-1); x_t)。
**还剩什么？** 目标公式相当于 TTT 的一个**受限特例**：把 B 限制为"观测空间残差经一个网络"，而不是完整梯度步。也就是说 B(h-, r) 的表达力**弱于** TTT 的 -η∇ℓ（后者在 D 非线性时会带 Jacobian，不只是 r）。
**这一条最伤的地方**：目标公式不是 TTT 的推广，而是 TTT 的**退化**。声称"学习式残差修正"是新贡献，会被问："为什么不直接用 TTT / 为什么限制成 B(h-, r)？"

---

## 5. 退化反例清单（每个都是明确构造）

1. **D ≡ 0**：则 r = z，h = h- + B(h-, z)。修正与预测无关 → **退化为"无穷增益 / dead-beat 观测器"**：一步就用观测覆盖状态，噪声不滤波（增益 = 1）。已知结构：dead-beat observer、直接状态映射。
2. **B 线性且不依赖 h-（B = K r）**：则 h = h- + K(z - D h-) = (I - KD)h- + Kz。**精确退化为 Luenberger 观测器**（K 常数）或 **Kalman 滤波**（K 由协方差算）。若 Φ = I、D = xᵀ、K = αx，**精确退化为 Widrow–Hoff delta rule**。
3. **Φ_Δ ≡ I**：则 h- = h，公式退化为**按观测索引更新的普通 RNN**；半群性质变得平凡（Φ∘Φ' = I），Δ 完全失去意义。此时"带真实时间间隔"的声称**自我否定**——这也说明：**Δ 的全部内容都压在 Φ_Δ 上，一旦 Φ 不是 Δ 的函数，公式就没有时间语义。**
4. **B 只依赖 r（B = β r，标量 β）**：则 h = h- + β(z - D(h-))。β = 1 时为 dead-beat（不滤波）；β ∈ (0,1) 时是**指数移动平均/一阶低通**；β 由 ‖r‖ 决定时为**归一化 LMS / 变步长自适应滤波**。**全部是 1960–1980 年代的已知滤波器。**
5. **D 线性、Φ_Δ 为可乘对角**：h = Φ_Δ(h) + β(v - h-k)kᵀ 且 Φ_Δ(h) = (1-α)h → **精确退化为 Titans 式 (16) / Gated DeltaNet 式 (10)**（把 β 换成 θ、α 换成 α）。
6. **r 恒为 0（完美模型）**：则 h = h-，修正通道完全失效，公式退化为**纯开环预测**（此时只有 Φ_Δ 起作用，就是一个 Neural ODE / SSM）。这说明目标公式的"感知"能力完全依赖 D 的失配——**而 D 与 Φ 是联合训练的，训练会把 r 压小**（自监督预测类模型的已知 collapse 风险）。

---

## 6. 若要保持非退化必须加的约束（每条配先例判断）

| # | 约束 | 是否已有先例 | 链接 |
|---|---|---|---|
| 1 | Φ_Δ 必须满足半群 Φ_Δ1∘Φ_Δ2=Φ_(Δ1+Δ2)，且不能是 exp(AΔ) 的已知族 | **半群约束已有大量先例**（flow map / consistency models）；exp(AΔ) 族已有 S4/Mamba/CRU | https://arxiv.org/html/2505.18825v2 |
| 2 | B 必须是 (h-, r) 的联合非线性，且不能写成 K r | **已有**：KalmanNet（RNN 学增益）、TTT（梯度步）、Titans | https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf |
| 3 | r 是唯一学习信号（无重构、无 ELBO、无回报） | **已有**：预测编码、PSR、TTT 的自监督损失 | https://arxiv.org/html/2112.10048v3 |
| 4 | Δ 同时进入 Φ 与 B（即 B(h-, r, Δ)） | **已有（间接）**：CRU 的 K_t 依赖 Σ⁻_t，而 Σ⁻_t 依赖 Δ | https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf |
| 5 | D 必须是可学习的非线性解码器（神经观测映射） | **已有**：TTT-MLP、EKF 的非线性 h(·)、KalmanNet | https://arxiv.org/html/2407.04620v4 |
| 6 | 稀疏/选择性/局部化：B 只在稀疏子空间/事件上非零 | **部分已有**：Mamba 的输入依赖 Δ、DeltaNet 的 β 选择、GLA 门；**事件触发观测器方向本稿未检索，建议主代理补** | https://arxiv.org/html/2312.00752 |
| 7 | 稳定性定理：给出 B 的 Lipschitz/收缩条件使 h 收敛到真状态（控制理论里唯一"值钱"的增量） | **已有**：Luenberger/Kalman 的收敛性；**但"学习式 B 的收敛保证"是真空**（KalmanNet 未给保证） | https://cse.lab.imtlucca.it/~bemporad/teaching/ac/pdf/06b-estimator.pdf |
| 8 | 采样细化一致性：Φ_Δ 满足"两次小步 = 一次大步"且与观测修正可交换（半群 + 观测修正的联合约束） | **未找到任何先例**（检索方向 8）→ **这是唯一可能真正空的格子** | 无 |
| 9 | 用途级：把上述结构用于连续时间策略并以 RL 回报训练 | **部分已有**：Dreamer/RSSM（ELBO+回报，但非残差式） | https://arxiv.org/html/2502.20168v1 |

---

## 7. 负面结果与反例（单独一节）

### 7.1 直接削弱目标公式新颖性的结论（按杀伤力排序）

1. **"带真实 Δ 的预测-修正" = 连续-离散卡尔曼滤波，1960s 起就是标准形**，2019–2022 已被神经网络化（ODE-RNN / CRU）。目标公式的 E 轴不是新贡献。[CRU 式 (8)(11)](https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf)
2. **"观测空间残差 + 加性增益修正" = 新息（innovation）驱动估计**，Kalman、Luenberger、KalmanNet 三代同构。A–D 不是新贡献。[KalmanNet 式 (5a)(7)](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/KalmanNet_Neural_Network_Aided_Kalman_Filtering_for_Partially_Known_Dynamics_0.pdf)
3. **"残差修正隐状态"在机器学习里已逐字出现**：DeltaNet 家族原式就是 S = S + β(v - Sk)kᵀ；Gated 版本还多了可乘转移。[GatedDeltaNet 式 (10)](https://arxiv.org/html/2412.06464v3)
4. **"学习式 B"已被做透**：TTT 的 -η∇ℓ、Titans 的 (1-α)M - θ∇ℓ、KalmanNet 的 RNN 增益。目标公式的 B(h-, r) 表达力**弱于**这些。[TTT 式 (2)](https://arxiv.org/html/2407.04620v4)
5. **"Δ 进状态转移"在深度学习里是 SSM 的第一课**：Ā = exp(ΔA)。目标公式若以此为核心，审稿人只会问 "why is this not an SSM with a measurement update?"。[Mamba 式 (4)](https://arxiv.org/html/2312.00752)
6. **"一次推理步 = 一次加性预测误差修正"在计算神经科学里是标准表述**，已有 25 年。[预测编码综述](https://arxiv.org/html/2112.10048v3)
7. **即使加上"连续时间"，delta-rule 的精确 flow 版也已发表**（EFLA 2025）："delta-rule update can be interpreted as an explicit Euler discretization of an underlying continuous-time system"。[EFLA](https://arxiv.org/html/2512.12602v5)
8. **Q3（为规划服务）也不是空白**：RSSM 就是"先验 + 观测后验修正 + 想象规划"，2020 年。[RSSM](https://arxiv.org/html/2502.20168v1)

### 7.2 这类结构的已知失效模式（对目标公式直接适用）

- **模型失配下发散**：Luenberger/Kalman 的稳定性依赖 A-LC 的谱条件；一旦 B 是学出来的、没有收缩性保证，误差可发散。**目标公式没有任何稳定性条件**（§6 第 7 行是真空）。
- **纯预测误差信号会饱和/collapse**：Titans 明确写了 "the gradient can become extremely small after several surprising steps, leading to stocking in a flat area (i.e., local minima), and missing information about some parts of the sequence"——这是对"r 是唯一学习信号"的**直接反例**。[Titans](https://arxiv.org/html/2501.00663v1)
- **纯加性修正无法表达"替换/覆盖"语义**：这正是 DeltaNet 用 (I - βkkᵀ)（Householder 收缩）而不只是 +β(v-Sk)kᵀ 的原因；纯加性 B 在 h- 与 r 相关时会**双计**（把预测里已有的信息再加一遍），导致过冲。**加性 B 是一阶近似，高增益下不稳定。** [GatedDeltaNet 关于 Householder 转移矩阵的段落](https://arxiv.org/html/2412.06464v3)
- **dead-beat 增益放大噪声**：B 只看 r 且增益≈1 时，h 直接等于观测的逆映射，噪声完全不滤波（§5 反例 1）。Kalman 的全部价值就是**限制**这一点。
- **Δ 的病态区间**：Δ → 0 时 Φ_Δ ≈ I，r 中时间信息消失，梯度病态；Δ → ∞ 时线性化失效（EKF/CRU 的已知问题）。**目标公式没有对 Δ 的定义域与数值稳定处理**。
- **误差累积（compounding error）**：h- 由 Φ_Δ 反复复合得到，误差随复合次数增长；半群一致性正是为此提出，但现有 flow-map 工作只在**无观测**的生成模型里验证过。
- **D/Φ 联合训练会把 r 压小**：自监督预测结构的已知退化（trivial solution：D∘Φ ≈ identity，r ≈ 0，修正通道空转）。**目标公式没有排除这种退化解的机制**（§5 反例 6）。

### 7.3 有没有与目标公式"逐字对应"的已有工作？

**有。三条，按逐字程度排序：**

| 目标公式 | 已有工作的逐字形式 | 差在哪里 |
|---|---|---|
| h- = Φ_Δ(h) / r = z - D(h-) / h = h- + B(h-,r) | CRU：μ⁻_t = exp(A(t-τ(t)))μ⁺_τ(t)；μ⁺_t = μ⁻_t + K_t(y_t - Hμ⁻_t) | D=H 线性；B 由协方差闭式而非网络；用途=插值 |
| 同上 | GatedDeltaNet：S_t = S(t-1)[α_t(I-β_t k_t k_tᵀ)] + β_t v_t k_tᵀ | 无真实 Δ（α_t 是学出的门）；无观测 z（v_t 是输入值） |
| 同上 | TTT：W_t = W(t-1) - η∇ℓ(W(t-1); x_t) | 无 Δ；B 是梯度步而非显式观测量残差 |

**结论**：目标公式 = **Kalman/CRU 的骨架** + **TTT/DeltaNet 的学习式 D/B**。两个部件都已有逐字先例，**组合方式**（B 显式吃观测残差、Δ 显式进 Φ、且用于策略）在检索范围内**没有找到完全相同的论文**——但这是"组合没被同一篇论文写出"，不是"组合不可能被想到"。

---

## 8. 附：七轴对照矩阵（✓=有，△=部分/间接，✗=无）

| # | 工作 | A 显式先验 | B 观测残差 | C 加性修正 | D B 依赖(h-,r) | E 真实 Δ | F 稀疏/选择 | G 训练目标 |
|---|---|---|---|---|---|---|---|---|
| 1 | Kalman 1960 | ✓ | ✓ | ✓ | ✓ | ✓(连续-离散) | ✗ | 无（闭式最优） |
| 2 | Luenberger 观测器 | ✓ | ✓ | ✓ | △（K 常数） | ✗ | ✗ | 无 |
| 3 | 信息滤波 | ✓ | △ | ✓ | ✗ | ✓ | ✗ | 无 |
| 4 | **CRU 2022** | ✓ | ✓ | ✓ | ✓（K 依赖 Σ⁻） | **✓ 秒** | ✗ | ELBO |
| 5 | KalmanNet 2022 | ✓ | ✓ | ✓ | **✓ 学习式** | ✗ | ✗ | 监督 MSE |
| 6 | ODE-RNN 2019 | ✓ | △ | △ | △ | **✓ 秒** | ✗ | ELBO |
| 7 | GRU-D 2018 | △ | ✗ | ✗ | ✗ | **✓ 秒** | ✓ | 监督 |
| 8 | Mamba 2024 | ✓ | ✗ | ✗ | ✗ | ✓（步长） | ✓ 输入依赖 | 自回归 CE |
| 9 | Widrow–Hoff 1960 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | MSE |
| 10 | Schlag 2021 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 语言建模 |
| 11 | DeltaNet 2024 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 语言建模 |
| 12 | **GatedDeltaNet 2025** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓✓ | 语言建模 |
| 13 | EFLA 2025 | ✓ | ✓ | ✓ | ✓ | ✓（token 间隔） | ✓ | 语言建模 |
| 14 | **TTT 2025** | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 自监督 |
| 15 | Titans 2025 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | 自监督 + CE |
| 16 | Ba 2016 | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | 监督/RL |
| 17 | 预测编码 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ 层级 | 自由能 |
| 18 | PSR 2002 | △ | △ | ✗ | ✗ | ✗ | ✓ | 一步预测 |
| 19 | RSSM/Dreamer 2020 | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ELBO + RL |
| 20 | Flow-map 2025 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | 自蒸馏 |
| — | **目标公式** | ✓ | ✓ | ✓ | ✓ | ✓ | **✗** | **?（未声明）** |

**读法**：目标公式在 A–E 五轴的每一列上都至少有一篇 ✓；**唯一整列为空的组合是"E 真实 Δ + B 学习式且吃残差 + 用于策略"的同篇工作**，以及 **F 轴（目标公式本身完全没有稀疏/选择性结构）**。这两点就是全部可辩护空间。

---

## 9. 给主代理的三条作战建议

1. **不要把 A–D 写成贡献**。写成"我们采用标准的预测-修正（新息）形式"，然后把贡献压到 §6 第 8 行（半群一致性 + 观测修正的联合约束）或第 7 行（学习式 B 的收敛定理）。这两格在检索范围内是空的。
2. **必须正面引用 CRU (ICML 2022) 与 GatedDeltaNet (ICLR 2025)**。不引会被审稿人当作没做文献工作；引了并说清差异（D 非线性 + B 任意网络 + 用途）才可能过。
3. **F 轴（稀疏/选择性/局部化）是现成的差异化入口**：目标公式完全没有任何选择性结构，而 DeltaNet/Mamba 的整个演进史就是"加选择性"。从这里切入比从"Δ 语义"切入安全得多——因为"Δ 语义"这条已经被 GRU-D / CRU / Mamba / EFLA 四篇封死。
