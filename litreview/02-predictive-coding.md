# 02 预测编码 / PredNet / 时间预测编码：与目标公式的逐条对照

调研者：pc-error（文献调研子代理）。时间：2026-11-03。
方法：只用网络检索（tavily_search / tavily_extract）。未训练、未写代码、未访问真机、未读本地旧文档。
证据等级：**[正文级]**=已用 tavily_extract 拉到正文并读到公式；**[仅摘要级]**=只拿到摘要；**[符号抽取丢失]**=PDF 数学符号在抽取中乱码，结构由正文文字确认。

---

## 0. 结论摘要（先给判词，后面给证据）

1. **目标公式的骨架 = 状态空间模型的「预测–新息–增益」更新。** 线性高斯下就是 Kalman 滤波的 correction step；非线性下它的 h⁻ 依赖增益就是 EKF 的雅可比增益（Rao & Ballard 1997 已在视觉皮层模型里这么写，并用了 innovations 这个词）。
2. **PC 文献有两种「用预测误差改状态」的写法：**
   - (a) **静态 PC**（Rao & Ballard 1999、Friston 2005、Whittington & Bogacz 2017）：没有来自时间的先验 h⁻，只有来自上一层的 top-down 预测；修正是梯度流 ẋ = −∂F/∂x，**迭代到稳态**才算完成推理。
   - (b) **时间 PC**（Millidge, Tang, Osanlouy, Harper & Bogacz, PLOS Comput Biol 2024，下称 tPC）：显式引入 h⁻ = A·f(x̂_{k−1}) + B·u_k，稳态解**逐项等于目标公式**：x̂_k = x̂⁻_k + K(y_k − C x̂⁻_k)，K = Σx·Cᵀ(CΣx Cᵀ + Σy)⁻¹。
3. **结构层面（A/B/C/D 四轴）几乎没有新意。** 唯一没被 PC 文献占的两点：(i) B 是**学出来的、同时依赖 (h⁻, r)** 的函数；(ii) **Δ 作为外部注入的真实时间间隔**。这两点分别属于「学习式滤波增益」（KalmanNet 一类）与「时间感知 RNN / Neural ODE」两个成熟方向，不是 PC 的领地。
4. **tPC 作者自己写死了「只做一次修正」的性质：** 单步推理等价于「每个样本只走一步梯度」，并明说 *"in many cases, such an algorithm simply would not work because a single step is nowhere near enough to approach the optimum"*；他们能用它的前提是线性凸 + 世界变化慢 + 上一步的解可热启动。
5. **Δ 在 PC 文献里从不作为时间间隔进入状态转移。** 时间要么是观测索引 k（tPC、PredNet），要么是微分算子/广义坐标（Friston 2008）。tPC 唯一出现 Δk 的地方是把 A 矩阵按 Δk 参数化（Eq 27，仿真设定），**网络本身不吃 Δ**。
6. **文献归属需更正：** 任务里的 "Pinchetti et al., Benchmarking PC Networks (ICML 2022)" 在 PMLR v162 查不到；ICML 2022 的 PC 论文是 Frieder & Lukasiewicz 的收敛性论文。Pinchetti 的 benchmark 是 arXiv:2407.01163（2024），Pinchetti 的 NeurIPS 是 2022 年的 *Predictive Coding beyond Gaussian Distributions*。详见 §5。

---

## 1. 目标公式与七轴判据

    h- = Phi_Delta(h)          # 带真实时间间隔 Delta 的状态转移（先验/预测）
    r  = z - D(h-)             # 观测空间残差/新息
    h  = h- + B(h-, r)         # 残差式修正（学习式增益）

| 轴 | 判据 | 说明 |
|---|---|---|
| A | 有无显式先验 h⁻ | **按「来自时间演化的预测态」判**；只来自上层空间的 top-down 预测记为「部分」。若按「任何 prior」判，所有 PC 都算有 A，轴就失去区分度。条目里两种判法都写。 |
| B | 误差是否定义在观测空间 r = z − D(h⁻) | 区分「观测/像素空间误差」与「层间表征空间误差」 |
| C | 修正是否加性 h = h⁻ + B | 区分加性残差修正与门控/乘性/重参数化更新 |
| D | B 是否同时依赖 h⁻ 与 r | 固定增益→否；雅可比/精度加权→依赖 h⁻；学出来的→可能两者都依赖 |
| E | 有无真实时间间隔 Δ | 观测索引 k 不算；Euler 步长 Δt 不算（那是推理内步长） |
| F | 稀疏/选择性/局部化 | 逐层、逐通道、逐像素的误差与增益（precision weighting、正负误差群体） |
| G | 训练目标 | 重构 / ELBO / 一步预测 / RL 回报 / 对比 / 自监督 |

每条另答三问：**Q1** 每次观测是否用预测误差修正隐状态；**Q2** 是否按真实 Δ 演化、时间语义是什么；**Q3** 用途是给规划/生成未来（model-based RL）还是通用在线感知/状态估计。

---

## 2. 逐篇条目

### G1. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects
- **作者 / 年份 / 会议或期刊 / 链接**：Rajesh P. N. Rao, Dana H. Ballard / 1999 / Nature Neuroscience 2(1):79–87 / https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf **[正文级]**
- **核心更新方程**（Methods，页 86–87）：

      生成：I = f(U r) + n,   f(U r) = f(Σ_j U_j r_j)
      层级：r = r_td + n_td,  r_td = f(U_h r_h)
      代价：E1 = (1/σ²)(I − f(Ur))ᵀ(I − f(Ur)) + (1/σ_td²)(r − r_td)ᵀ(r − r_td)
            E  = E1 + g(r) + h(U)
      推理：dr/dt = −(k1/2)·∂E/∂r
            = (k1/σ²)·(∂f/∂x)ᵀ·Uᵀ·(I − f(Ur)) + (k1/σ_td²)·(r_td − r) − (k1/2)·g′(r)
      线性特例（正文 Eq 7/8）：dr/dt = UᵀI + (r_td − r) − g′(r) − W r,   W = UᵀU
      学习：dU/dt = (k2/σ²)(I − f(Ur)) rᵀ − k2·λ·U
      时间扩展（Methods 明写，但论文明确说未使用）：r(t+1) = f(V r(t)) + m

- **三问**：Q1 是——每次输入用 bottom-up 残差与 top-down 残差一起改 r；Q2 否——时间只出现在推理迭代步 dr/dt，状态没有跨观测演化，**没有 Δ**；Q3 通用在线感知/状态估计（图像编码 + V1 感受野解释），不服务规划。
- **A–G**：
  - **A**：**无**（按时间先验判）。预测来自上一层 r_td = f(U_h r_h)，不是上一时刻。宽判「任何 prior」则算有，但那是空间先验。
  - **B**：**部分**。(I − f(Ur)) 在图像/观测空间，(r_td − r) 在表征空间。
  - **C**：**是**（梯度项加性），但加的是梯度不是「增益×残差」；Uᵀ(I−f(Ur)) 恰好是残差经 Uᵀ 映射。
  - **D**：**依赖两者但不是增益形式**——(∂f/∂x)ᵀ 依赖状态，残差依赖观测。
  - **E**：**无**。
  - **F**：**有**。逐层局部误差、kurtotic 先验产生**稀疏**表征、σ²/σ_td² 是逐通道固定的精度（等价固定选择性增益）。
  - **G**：无监督；最小化 MDL 代价 E（平方预测误差 + 先验罚）。
- **任务与训练目标**：自然图像块的无监督编码；解释 V1 简单细胞感受野与 end-stopping/非经典感受野。
- **最接近的先例**：作者自己 1997 年的 Kalman/EKF 视觉模型（引言称 "a model based on the principle of Kalman filtering"，Methods 给出时间递推式）。
- **新颖性判断**：**削弱 A/B/C 三轴。** 1999 年已给出「预测−残差−用残差改状态」的完整骨架，并**明确写出时间递推式** r(t+1) = f(V r(t)) + m，只是声明静态图像够用所以不用。**不削弱 D/E**：增益是固定精度×雅可比，B 不是学出来的函数，Δ 不存在。

### G2. Dynamic model of visual recognition predicts neural response properties in the visual cortex
- **作者 / 年份 / 会议或期刊 / 链接**：Rajesh P. N. Rao, Dana H. Ballard / 1997 / Neural Computation（抓到的是技术报告版，页眉写 "To appear in Neural Computation"）/ https://homes.cs.washington.edu/~rao/dynrecNC97.pdf **[正文级 / 符号抽取丢失]**
- **核心更新方程**（符号在 PDF 抽取中丢字，结构由正文英文句子逐句确认）：

      状态演化：r(t+1) = f(W · r̂(t)) + 噪声          (式 12，W 为 state transition matrix，正文称对应 prediction weights)
      代价：    (I − W r̂)ᵀ Σ⁻¹ (I − W r̂) + (r_td − r̂)ᵀ Σ_td⁻¹ (r_td − r̂) + 权重先验
      更新：    r̂ = r̂⁻ + K_bu·(I − W r̂⁻) + K_td·(r_td − r̂⁻)        (式 33 附近)
      协方差：  P(t+1) = f′·P·f′ᵀ + Σ                                  (式 36)

  原文：*"The mean r̂ is updated by filtering the current residuals or ‘innovations’ [Maybeck, 1979] (I − W r̂) and (r_td − r̂) using [...]"*；*"Equation 33 implements an efficient trade-off between information from three different sources: the system prediction, the top-down prediction r_td, and the bottom-up data I"*。
- **三问**：Q1 是——每个时刻用**新息（innovations）**修正状态；Q2 **部分**——有跨时刻转移矩阵 W 与离散 t，但**无 Δ**，W 不随间隔变化；Q3 通用在线感知/状态估计（自由观看视觉皮层），不服务规划。
- **A–G**：
  - **A**：**有**。r̂⁻ = f(W r̂(t)) 是标准 time-evolved prior，最早一批之一。
  - **B**：**是**。(I − W r̂) 是观测/重建空间新息；(r_td − r̂) 是第二个（空间层级）新息。
  - **C**：**是**。r̂ = r̂⁻ + 增益×新息，纯加性。
  - **D**：**增益依赖协方差/雅可比，不直接依赖 r**。协方差 P 在线更新，所以增益随不确定度变，但不是 (h⁻, r) 的学出来的函数。
  - **E**：**无**。
  - **F**：**有**。逐层逐单元 Σ⁻¹ 精度加权、双通路、局部 Hebbian。
  - **G**：MDL/EM（EKF 滤波 + Hebbian 权重），无监督。
- **任务与训练目标**：解释自由观看/注视下 V1–V4 的非经典感受野调制；动态识别。
- **最接近的先例**：Kalman/EKF（自述 "using a form of the extended Kalman filter as given by the MDL principle"）；同组 1999 静态简化版（G1）。
- **新颖性判断**：**最致命的先例之一。** 三步（先验→新息→增益×新息加性修正）在 1997 年已完整出现，连 innovations 都是原文用词。与目标公式唯一实质差别：**增益由协方差/雅可比给定（P 在线传播），不是 (h⁻,r) 学出来的；且无 Δ**。若新颖性主张落在「结构」上，这篇足以否掉；若落在「Δ 条件化 + 学习式增益」上，它只是强先例。

### G3. A theory of cortical responses
- **作者 / 年份 / 会议或期刊 / 链接**：Karl Friston / 2005 / Philosophical Transactions of the Royal Society B（PDF 只能确认刊名与年份；正文可见页码 816、821–823，**卷期页码未核到**）/ https://www.fil.ion.ucl.ac.uk/~karl/A%20theory%20of%20cortical%20responses.pdf **[正文级]**
- **核心更新方程**（§3.6–3.11、§4.1–4.2）：

      生成：u = g(v, θ) + ω,   v = v_p + ω_p
      误差（精度加权）：ξ_u = Σ_u^(−1/2)(u − g(v,θ)),  ξ_p = Σ_p^(−1/2)(v − v_p)
      层级：u = g₁(v₂,θ₁)+ω₁,  v₂ = g₂(v₃,θ₂)+ω₂, …
      本层误差：ξ_i = f_i − g_i(f_{i+1}, θ_i) − l_i ξ_i                              (3.10)
      状态更新：ḟ_{i+1} = ∂F/∂f_{i+1} = −(∂ξ_iᵀ/∂f_{i+1})·ξ_i − (∂ξ_{i+1}ᵀ/∂f_{i+1})·ξ_{i+1}   (3.11)
      学习：θ̇_i = ∂F/∂θ_i（Hebbian；4.2：θ̇ = (1+l_i)⁻¹⟨ξ_i f_{i+1}ᵀ⟩）

- **三问**：Q1 是——每个刺激用本层与下层误差一起改表征；Q2 否——**没有时间维度**，E 步在同一观测上梯度上升；Q3 通用感知/推断 + 学习，不服务规划。
- **A–G**：
  - **A**：**无时间先验**；「先验」是上层 g_i(f_{i+1}) 给出的空间先验 v_p。
  - **B**：**部分**。最底层 ξ_u 是观测空间，中间层是表征空间。
  - **C**：**是**（对 ḟ 加性），但加的是两项梯度。
  - **D**：**是**——∂ξ/∂f 依赖状态、ξ 依赖数据；但这是梯度不是参数化增益。
  - **E**：**无**。
  - **F**：**有且更细**——每误差单元独立精度 Σ_i；侧向 l_i 白化/去相关误差；只允许相邻层连接。
  - **G**：变分自由能 F 的 E/M 步（variational EM / empirical Bayes）。
- **任务与训练目标**：解释诱发响应、重复抑制、MMN 等；「感知 = 最小化自由能」。
- **最接近的先例**：Rao & Ballard 1999（文中直接承接）；Helmholtz 式贝叶斯推断。
- **新颖性判断**：**部分削弱。** 它把「误差=预测−输入，误差直接进状态时间导数」做得比 Rao & Ballard 更彻底，所以 C/D 的直觉 2005 年就有；但**完全没有时间轴**（无 h⁻），不削弱 A/E，也不涉及「一次修正 vs 迭代到收敛」（它明确迭代）。

### G4. Hierarchical Models in the Brain（补充：Δ 轴的反面证据）
- **作者 / 年份 / 会议或期刊 / 链接**：Karl Friston / 2008 / PLoS Computational Biology 4(11): e1000211, doi:10.1371/journal.pcbi.1000211 / https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000211 **[正文级]**
- **核心更新方程**（连续时间 + 广义坐标）：

      ẋ = f(x, v) + w,   y = g(x, v) + z            (式 1，连续时间状态空间)
      广义坐标：x̃ = [x, x′, x″, …] = [x[0], x[1], x[2], …]
      运动学约束：x′ = x[1], x[1]′ = x[2], …（把「导数一致」变成先验，而不是把 h⁻ 变成 h 的函数）
      DEM：D-step（状态）/ E-step（参数）/ M-step（超参数），全为自由能梯度上升

- **三问**：Q1 是——用广义坐标下的精度加权误差改状态；Q2 **连续时间但不是 Δ**——时间以导数进入，**没有间隔参数**；Q3 通用在线感知/去卷积。
- **A–G**：
  - **A**：**形式上最接近但没有 Δ**——先验来自广义运动连续性，等价于把 Taylor 展开当先验，步长隐含。
  - **B**：**是**，ε̃_y = ỹ − g(x̃) 在广义观测空间。
  - **C**：**是**，但仍为梯度流（ẋ 含 D 算子）。
  - **D**：**是**（雅可比依赖状态）。
  - **E**：**无 Δ，但时间语义是「秒」**（连续时间导数）。全部 PC 文献里时间语义最接近目标的一个，但它把 Δ 消掉了：预测只需知道导数，不需知道隔了多久。
  - **F**：**有**（逐层、逐阶广义坐标、逐精度）。
  - **G**：变分自由能（DEM），无监督。
- **任务与训练目标**：层级动态模型的统一反演（GLM、卷积模型、非线性时间序列皆为特例）。
- **最接近的先例**：Friston 2005（静态版）；DEM / variational filtering（自引）。
- **新颖性判断**：**反向支持目标公式的 Δ 主张。** 若有人主张「PC 早就有真实时间」，答案是不能：Friston 用微分算子替代 Δ，且 Millidge 综述指出广义坐标实践上要截断到 **4–6 阶**（见 G7）。PC 传统处理时间 = 导数 + 截断，不是间隔 Δ。目标公式若把 Δ 作为外部标量注入转移，**在 PC 内部没有对应物**；但这也意味着它在 PC 的自由能推导里没有理论根据——Δ 不是 PC 的变量。

### G5. An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity
- **作者 / 年份 / 会议或期刊 / 链接**：James C. R. Whittington, Rafal Bogacz / 2017 / Neural Computation 29:1229–1262, doi:10.1162/NECO_a_00949 / https://www.bndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf **[正文级]**
- **核心更新方程**（§2.2.1–2.2.4）：

      层间均值：μ^(l)_i = Σ_j θ^(l+1)_{i,j} · f(x^(l+1)_j)                        (2.11)
      预测误差：ε^(l)_i = (x^(l)_i − μ^(l)_i) / Σ^(l)_i                          (2.17)
      状态动力学：ẋ^(a)_b = − ε^(a)_b + Σ_i ε^(a−1)_i · θ^(a)_{i,b} · f′(x^(a)_b)    (2.18)
      学习：Δθ^(a)_{b,c} ∝ ε^(a−1)_b · f(x^(a)_c)                                (2.19)
      （权重更新发生在状态收敛之后；正文 §2.2.4："Once the network has reached its steady state, the parameters are updated"）

- **三问**：Q1 是——每次样本用本层误差与下层误差修正状态；Q2 否——**没有时间轴**，样本 i.i.d.；Q3 通用感知/监督学习结构，不服务规划。
- **A–G**：
  - **A**：**无时间先验**。μ 来自上一层，是空间层级先验。
  - **B**：**否**。误差 ε^(l) 定义在**层间表征空间** x^(l) − μ^(l)，不是观测空间 z − D(h⁻)。
  - **C**：**是**（对 ẋ 加性），但为梯度形式。
  - **D**：**是**——f′(x) 依赖状态、ε 依赖数据。
  - **E**：**无**。
  - **F**：**有**——每单元 1/Σ^(l)_i 精度；逐层局部；只用相邻层连接。
  - **G**：监督学习（最小化输出层平方误差 / 分类），局部 Hebbian 逼近 BP；MNIST 等实验。
- **任务与训练目标**：证明 PC 网络能在纯局部 Hebbian 规则下做监督学习，并在特定参数下收敛到 BP 的权重更新。
- **最接近的先例**：Rao & Ballard 1999、Friston 2005（正文自述）；Bogacz 2017 tutorial。
- **新颖性判断**：**削弱 C/D 两轴**，把「误差驱动状态修正」确立为 PC 的标准语言；**不削弱 A/E**（无时间先验、无 Δ），也**不支持**目标公式的「观测空间新息」写法（B 轴反而不同）。它同时是「PC 推理是迭代的」这一事实的权威出处。

### G6. Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning（PredNet）
- **作者 / 年份 / 会议或期刊 / 链接**：William Lotter, Gabriel Kreiman, David Cox / 2017（ICLR 2017；arXiv v5 标注 "Published as a conference paper at ICLR 2017"）/ arXiv:1605.08104 / https://arxiv.org/pdf/1605.08104 **[正文级]**
- **核心更新方程**（正文 Eq 1–5 + Algorithm 1）：

      A_t^l = x_t                              if l = 0
            = MAXPOOL(RELU(CONV(E_t^{l−1})))   if l > 0                       (1)
      Â_t^l = SATLU(RELU(CONV(R_t^0)))        if l = 0；否则 RELU(CONV(R_t^l))   (2)
      E_t^l = [ RELU(A_t^l − Â_t^l) ; RELU(Â_t^l − A_t^l) ]                    (3)
      R_t^l = CONVLSTM( E_{t−1}^l , R_{t−1}^l , UPSAMPLE(R_t^{l+1}) )          (4)
      L_train = Σ_t λ_t Σ_l (λ_l / n_l) Σ_{n_l} E_t^l                          (5)

- **关键结构判读（重要）**：PredNet 的时序是 **R_t → Â_t → E_t → R_{t+1}**，不是「先算出 h⁻ 再用 r 改 h」。误差 E_t **直接进入下一时刻的循环状态**（ConvLSTM 的输入之一），但**没有 h⁻ + B(h⁻,r) 这种同一步内的两段式**。它更像「把一次修正摊销进一个门控 RNN」。
- **是否 truncate**：正文**没有误差截断**；截断发生在训练——KITTI 上按 **10 帧序列**采样做 BPTT（"Sequences of 10 frames were sampled"），Adam，学习率 1e−3 半程降 10 倍，损失权重 λ_l：最优是**只压最底层**（λ0=1, λl>0=0）；**每层等权损失会 considerably degrade predictions**；给上三层一个低一个数量级的权重（λ0=1, λl>0=0.1）只"lead to only slightly worse predictions"。推理时把预测回灌可做多步外推，但正文自述 "the model naturally breaks down when extrapolating further into the future"，需专门微调。
- **三问**：Q1 是——每帧都用误差更新循环状态；Q2 否——时间 = **帧索引 t**，无 Δ、无时间戳、无变帧率机制；Q3 **生成未来（下一帧视频预测）**，用于无监督表征学习（可解码姿态/转向角），**不是 planning/RL**。
- **A–G**：
  - **A**：**部分**。Â_t 是「当前帧的预测」，但 h⁻ 不是从上一状态解析演化出来的；先验藏在 ConvLSTM 状态里。
  - **B**：**是**，误差在像素/特征图空间（A 与 Â 之差），最低层就是像素。
  - **C**：**否（严格判）**。E 进入的是 ConvLSTM 的门控更新，不是加性 h⁻ + B。
  - **D**：**是**——R_{t−1}（状态）、E_{t−1}（误差）、上层 R 三路同时进入。
  - **E**：**无**。
  - **F**：**有，且最彻底的一篇**。逐层、逐像素、正/负误差分成两个 ReLU 群体（对应 on-center/off-surround 与 off-center/on-surround），loss 按层加权 λ_l。
  - **G**：一步预测（L1 on error units），自监督/无监督；截断 BPTT。
- **任务与训练目标**：KITTI/CalTech 下一帧预测；表征可解码物体姿态、相机转向角。
- **最接近的先例**：Rao & Ballard 1999、Friston 2005（作者明确自引）；Chalasani & Principe 2013 的 Deep Predictive Coding Networks（稀疏编码 + 线性动力系统 + 逐层贪心）。
- **新颖性判断**：**削弱 F 轴与「误差进状态」这个直觉**，而且它给出了一个**与目标公式不同的解法**：不做迭代推理，而是把修正摊销进 RNN（一次前向）。这反过来是目标公式「一次加性修正」的强对照——PredNet 用 RNN 学 B，效果可测（MSE 3.13e−3 / SSIM 0.884，优于 CNN-LSTM Enc-Dec 的 3.67e−3 / 0.865）。**不削弱 A/E**：它连 C 轴都不满足。

### G7. Predictive Coding: a Theoretical and Experimental Review
- **作者 / 年份 / 会议或期刊 / 链接**：Beren Millidge, Anil K. Seth, Christopher L. Buckley / 2022（v4，页脚 "14th July, 2022"）/ arXiv:2107.12979 / https://arxiv.org/pdf/2107.12979 **[正文级；公式未取]**
- **说明**：该综述的公式块在 PDF 抽取中未可靠还原，**本文不引用它的具体方程**，只引用可核实的文字结论。
- **核心可引用结论**：
  - 范式分类（Figure 3）里单列 **Temporal Predictive Coding**：输入 = 当前观测，预测 = **next timestep 的观测**。
  - 推理是迭代的：*"The latent variable μs then follow the dynamics of equation (PC) to minimize prediction errors. Once the network has settled into an equilibrium, or else the dynamics have been run for some prespecified number of steps, then the parameters θ are updated"*。
  - 广义坐标（Friston 系）实践上要**截断到 4–6 阶**，只提供短时动力记忆。
  - 空位自述：*"there has not yet even been any large scale study of the performance characteristics of generalized coordinates on spatio-temporal video data"*；temporal PC 的实验远少于静态分类。
  - 对 PC 可证伪性的保留：*"it has been challenging to either experimentally confirm or refute predictive coding as a theory up to now"*。
- **三问**：Q1 综述层面是；Q2 综述层面两种都有（离散 k 与广义坐标），**没有 Δ**；Q3 覆盖感知/学习，temporal PC 明确指向「预测下一帧」。
- **A–G**：作为综述不逐条适用；关键点：它确认了 (a) PC 推理=迭代到平衡/固定步数，(b) temporal PC 的目标是**预测下一时刻观测**（而不是「用上一时刻状态做先验」），(c) 时间处理的现有手段是导数+截断。
- **最接近的先例**：Rao & Ballard 1999、Friston 系列、Whittington & Bogacz 2017（综述内串联）。
- **新颖性判断**：**对目标公式最有用的一条是「空位」证据**：temporal PC 缺大规模实证。但这是**实证空位，不是结构空位**——目标公式若主张结构新颖，这篇综述帮不上忙，反而说明「预测下一帧」是 PC 的既定目标之一。

### G8. Predictive coding networks for temporal prediction（tPC）
- **作者 / 年份 / 会议或期刊 / 链接**：Beren Millidge, Mufeng Tang, Mahyar Osanlouy, Nicol S. Harper, Rafal Bogacz / 2024 / PLOS Computational Biology 20(4): e1011183, doi:10.1371/journal.pcbi.1011183（bioRxiv 预印本 2023.05.15.540906）/ https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183 **[正文级]**
- **核心更新方程**（式 1–6、8–14、21–27；LaTeX 已按论文符号还原）：

      生成：x_k = A f(x_{k−1}) + B u_k + ω_x,   y_k = C f(x_k) + ω_y            (1)(2)
      自由能：F_k = ½ (y_k − C f(x_k))ᵀ Σ_y⁻¹ (y_k − C f(x_k))
                  + ½ (x_k − A f(x̂_{k−1}) − B u_k)ᵀ Σ_x⁻¹ (x_k − A f(x̂_{k−1}) − B u_k)   (6)
      误差：ε_y = Σ_y⁻¹ (y_k − C f(x_k))；ε_x = Σ_x⁻¹ (x_k − A f(x̂_{k−1}) − B u_k)   (8)(9)
      动力学：τ dx_k/dt = − ∂F_k/∂x_k = − ε_x + f′(x_k) Cᵀ ε_y                    (10)
             等价单步形：τ dx_k/dt = − x_k + A f(x̂_{k−1}) + B u_k + f′(x_k) Cᵀ ε_y    (12)
      Euler：x_k(t+Δt) = x_k(t) + (Δt/τ)[ −x_k + A f(x̂_{k−1}) + B u_k + f′(x_k) Cᵀ ε_y ]   (13)
      单步算法（每帧只推一次）：x̂_k = x_k + (Δt/τ)[ −x_k + A f(x̂_{k−1}) + B u_k + f′(x_k) Cᵀ ε_y ]   (14)
      线性稳态解：x̂⁻_k = A x̂_{k−1} + B u_k
                  x̂_k  = x̂⁻_k + K (y_k − C x̂⁻_k),   K = Σ_x Cᵀ [C Σ_x Cᵀ + Σ_y]⁻¹       (26)
      对照 Kalman：x̂⁻_k = A x̂_{k−1} + B u_k；x̂_k = x̂⁻_k + K(y_k − C x̂⁻_k)，
                  K = Σ⁻_k Cᵀ[C Σ⁻_k Cᵀ + Σ_y]⁻¹，Σ⁻_k = A Σ_{k−1} Aᵀ + Σ_x              (21)(22)
      学习：ΔA = η ε_x f(x̂_{k−1})ᵀ,  ΔB = η ε_x u_kᵀ,  ΔC = η ε_y f(x_k)ᵀ        (11)

- **作者对 Kalman 关系的原话**：tPC 假设上一步后验是 **Dirac 分布** x_{k−1}|y_{1:k−1} ~ δ(x_{k−1} − x̂_{k−1})（式 23），因此先验方差退化为 Σ_x（不含 A Σ Aᵀ），于是 *"predictive coding ignores the uncertainty estimated at the previous step"*、*"tPC does not propagate the uncertainty estimations"*。结论是 tPC ≈ **固定后验方差的 Kalman 变体**。
- **作者对「一次修正」的原话**：单步算法 *"is equivalent to taking only a single gradient step for each example. Clearly, in many cases, such an algorithm simply would not work because a single step is nowhere near enough to approach the optimum"*；能奏效靠两点——线性时目标凸（可用较大步长）+ 世界变化慢（上一步的解接近本步最优）。仿真结果：5 步推理与完全收敛都追平 Kalman；**单步（Fig 2C 电路）跟踪更差**，只是给出平滑轨迹。
- **三问**：Q1 **是**，而且同时用两类误差——sensory ε_y 与 temporal ε_x；Q2 **否**（就网络而言）——状态按**观测帧索引 k** 演化，推理时间 t 是内层；Δk 只在仿真里把 A 参数化为 [[1,Δk,Δk²/2],[0,1,Δk],[0,0,1]]（式 27，Δk=0.001），**网络不以 Δ 为输入**；Q3 **通用在线感知/状态估计（filtering + tracking + system identification）**，不服务规划。
- **A–G**：
  - **A**：**是**。x̂⁻_k = A f(x̂_{k−1}) + B u_k 是真正的 time-evolved prior——**这是全部 PC 文献里 A 轴最干净的一个**。
  - **B**：**是**。ε_y = y_k − C f(x_k) 就是观测空间新息（线性时 r = y − C x̂⁻）。
  - **C**：**是**（在稳态/线性下严格成立：x̂_k = x̂⁻_k + K r）。但实现上是**弛豫动力学**，不是一步赋值；只有 Eq 14 的单步版才是「一次加性修正」。
  - **D**：**是**。f′(x_k) 依赖状态，ε_y 依赖观测；但 K 本身**不依赖 r**，也不依赖 h⁻（除了线性 A 的平凡依赖）。
  - **E**：**否**（见 Q2）。这是全篇最关键的空缺：**tPC 是「等间隔离散时间」的 PC，不是「带 Δ 的 PC」**。
  - **F**：**有**。Σ_x、Σ_y 是逐维精度矩阵；实现上可以编码进学到的 A、C（正文：*"scaling by the inverse covariance matrices could be encoded in the learnt A and C matrices"*）。
  - **G**：变分自由能（式 5/6）+ 局部 Hebbian 学习 A、B、C；与 Kalman 对比 tracking MSE；自然视频上得到 Gabor 型、运动敏感的 receptive field。
- **任务与训练目标**：线性/非线性滤波与跟踪、在线系统辨识、自然动态刺激下的感受野学习。
- **最接近的先例**：Kalman 滤波（1960，被本文当作 benchmark 与推导对照）；Rao & Ballard 1997 的 EKF 视觉模型；同组 NeurIPS 2023 的 tPC 序列记忆版（Tang, Barron, Bogacz, https://proceedings.neurips.cc/paper_files/paper/2023/file/8a8b9c7f979e8819a7986b3ef825c08a-Paper-Conference.pdf ，**[仅摘要级]**）。
- **新颖性判断**：**逐字对应，最致命。** Eq 26 就是目标公式：h⁻ = A x̂ + Bu、r = y − C h⁻、h = h⁻ + Kr，K 由 Σ_x、Σ_y 给定。Eq 14 更直接：**只做一次修正的版本已被明确写出、实现成电路（Fig 2C）、并分析了何时失效**。目标公式剩给原创性的只有两点：(i) K/B 被换成**学出来的、依赖 (h⁻, r) 的函数**；(ii) Φ 被换成**吃 Δ 的函数**。这两点 tPC 都没有。

### G9. (Non-)Convergence Results for Predictive Coding Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Simon Frieder, Thomas Lukasiewicz / 2022 / ICML 2022, PMLR 162:6793–6810 / https://proceedings.mlr.press/v162/frieder22a.html **[仅摘要级]**（只拉到 PMLR 摘要页，正文 PDF 未逐式核对）
- **核心结论**：用动力系统理论形式化 PCN 的收敛性——**权重与初始化足够小时，PCN 对任意输入收敛**；**离开这个区域可以用反例发散**，具体是在一个小规模 PCN 里识别出 **Neimark–Sacker 分岔**，产生不稳定不动点与其周围的 invariant curve。原文：*"we show via a counterexample that PCNs can diverge, countering common beliefs held in the community"*。
- **三问**：Q1 是（研究对象就是 PC 推理动力学）；Q2 无关（静态样本）；Q3 理论分析，不服务规划。
- **A–G**：作为理论论文不逐条适用；与本专题相关的只有一条：**PC 的迭代修正只在受限参数域内稳定**——小权重、小初始化。
- **最接近的先例**：Whittington & Bogacz 2017、Rao & Ballard 1999 的迭代推理（被本文作为「此前只靠数值实验判断收敛」的对象）。
- **新颖性判断**：**不针对目标公式，但给出它最直接的失效风险。** 目标公式只做**一次**修正，天然绕开迭代发散；但若 B 的谱半径不受控（学出来的 B 很容易），单步修正同样会放大误差——「只走一步所以稳定」不是免费的。

### G10. Understanding and Improving Optimization in Predictive Coding Networks
- **作者 / 年份 / 会议或期刊 / 链接**：Nicholas Alonso, Jeffrey Krichmar, Emre Neftci / 2024 / AAAI 2024 / https://ojs.aaai.org/index.php/AAAI/article/view/28954/29814 **[正文级（摘要+引言）]**
- **核心结论**（摘要原文）：*"IL is computationally demanding, and without memory-intensive optimizers like Adam, IL may converge to poor local minima"*；*"although IL can reduce loss more quickly than BP, the reasons for these speedups or their robustness remains unclear"*。三项贡献：改电路实现以降算力、提出不增内存的新优化器、给出 IL 对二阶信息的敏感性条件。
- **三问**：Q1 是（改的是 PC 推理/学习）；Q2 无关；Q3 训练算法分析。
- **A–G**：不逐条适用；关键是对 **G（训练目标）** 的补充：PC 的推理学习（IL）在优化层面**依赖 Adam 这类外置优化器**，否则局部极小。
- **最接近的先例**：Whittington & Bogacz 2017、Pinchetti et al. 2024（同组引用）。
- **新颖性判断**：**削弱「PC 式修正作为通用学习结构」的可行性**，对目标公式的**推理规则**没有直接冲击——但若目标公式声称「B(h⁻,r) 可以在线学而不需要外置优化器」，这篇是反证：PC 的在线局部规则目前仍要 Adam 兜底。

### G11. Benchmarking Predictive Coding Networks – Made Simple
- **作者 / 年份 / 会议或期刊 / 链接**：Luca Pinchetti, Chang Qi, Oleh Lokshyn, Gaspard Oliviers, Cornelius Emde, Mufeng Tang, Amine M'Charrak, Simon Frieder, Bayar Menzat, Rafal Bogacz, Thomas Lukasiewicz, Tommaso Salvatori / 2024 / arXiv:2407.01163（配套库 PCX）/ https://arxiv.org/abs/2407.01163 **[正文级]**
- **注意**：任务里写的 "ICML 2022" **核不到**（PMLR v162 中无此文），见 §5。
- **核心结论（可引用原文/数据）**：
  - 规模上限：*"it is still unclear why PC is able to perform as well as classical gradient descent with backprop only up to a certain scale, which is of small convolutional models trained to classify the CIFAR10 dataset"*。
  - **能量/误差不平衡**：把误差集中到输出层（forward init）后，*"the energy in the last layer is orders of magnitude larger than the one in the input layer, even after performing several inference steps"*。
  - **状态学习率必须很小**：最好的性能一致地出现在 γ **显著小于 1.0**；作者怀疑现有训练设定偏好小 γ 因而难以加深。
  - **训练稳定性**：用 Adam 时，**隐藏层宽度会强烈影响使训练稳定的 γ 区间**；用 SGD 或 BP 都不出现这个现象（*"This behavioral difference with BP is unexpected"*）。
  - 定位：把「可扩展性」列为该领域**最主要的开放问题**。
- **三问**：Q1 是；Q2 无关；Q3 训练/基准，不服务规划。
- **A–G**：不逐条适用；对 **G** 的补充：PC 的训练目标是自由能，但**超参敏感 + 层间能量不平衡**是已报告的硬约束。
- **最接近的先例**：Whittington & Bogacz 2017、Pinchetti & Bogacz NeurIPS 2022 *Predictive Coding beyond Gaussian Distributions*（https://proceedings.neurips.cc/paper_files/paper/2022/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html）。
- **新颖性判断**：**不针对目标公式，但把「PC 式结构可扩展」这条路标成未解**。对目标公式的实际含义：**不要用「PC 的局部修正规则」给新颖性加分**——它在规模上还没证明自己。

---

## 3. 七轴对照总表

符号：✓ 满足；~ 部分/有条件；✗ 不满足；— 不适用。A 轴按「来自时间演化的先验」判。

| # | 工作 | A 时间先验 h⁻ | B 观测空间新息 | C 加性修正 | D 增益依赖(h⁻,r) | E 真实 Δ | F 稀疏/局部/选择 | G 训练目标 | 与目标公式的距离 |
|---|---|---|---|---|---|---|---|---|---|
| G1 | Rao & Ballard 1999 | ~（空间先验） | ~ | ✓（梯度） | ~ | ✗ | ✓（稀疏先验+精度） | 重构/MDL | **近**：有骨架，无时间先验、无 Δ |
| G2 | Rao & Ballard 1997 | ✓ | ✓ | ✓ | ~（协方差/雅可比） | ✗ | ✓ | MDL/EM | **最近之一**：EKF 版逐项对应 |
| G3 | Friston 2005 | ~ | ~ | ✓（梯度） | ✓ | ✗ | ✓（逐单元精度/侧向） | 变分自由能 | 中：无时间轴 |
| G4 | Friston 2008 | ~（导数先验） | ✓ | ✓（梯度流） | ✓ | ✗（连续时间，无 Δ） | ✓ | 变分自由能 | 中：时间语义最近，但 Δ 被消掉 |
| G5 | Whittington & Bogacz 2017 | ✗ | ✗（层间空间） | ✓（梯度） | ✓ | ✗ | ✓（1/Σ 精度） | 监督 | 中：确立 PC 修正语言 |
| G6 | PredNet 2017 | ~ | ✓ | ✗（门控 RNN） | ✓ | ✗ | ✓✓（逐层逐像素正负群体） | 一步预测 L1 | 中：解法**不同**（摊销而非迭代） |
| G7 | Millidge et al. 2022 综述 | — | — | — | — | — | — | — | 元证据：确认空位与范式 |
| G8 | tPC 2024 | ✓ | ✓ | ✓（稳态严格） | ✓ | ✗ | ✓（精度矩阵） | 自由能/Hebbian | **逐字对应** |
| G9 | Frieder & Lukasiewicz 2022 | — | — | — | — | — | — | — | 理论：迭代收敛的边界 |
| G10 | Alonso et al. 2024 | — | — | — | — | — | — | 优化 | 训练可行性反证 |
| G11 | Pinchetti et al. 2024 | — | — | — | — | — | — | 基准 | 可扩展性反证 |

**读表要点**：

1. **E 列全 ✗**，包括 tPC 与 Friston 2008。这不是巧合：PC 的推导从自由能出发，自由能里没有「时间间隔」这个变量；时间要么被离散成索引，要么被微分算子吃掉。
2. **A+B+C 同时 ✓ 的只有 G2（Rao & Ballard 1997）与 G8（tPC 2024）**。这两篇就是目标公式在文献中的落点，其余都是「相邻结构」。
3. **D 列的 ✓ 大多是雅可比/精度带来的**，而不是「学出来的增益函数」。tPC 的 K 不含 r；Rao & Ballard 1997 的 K 来自协方差。**没有任何一篇 PC 让 B 本身成为可训练网络并吃 (h⁻,r)**。
4. **F 列普遍 ✓**：精度加权是 PC 的标配，它天然是「逐单元选择性增益」——但增益是**标量/对角且固定**，不是状态相关的。

---

## 4. 负面结果与反例（直接削弱新颖性的部分）

### 4.1 最致命：PC 的推理是「迭代到收敛」，一次加性修正只是它的退化特例

证据链：
- 静态 PC 三篇（G1/G3/G5）全部是 `dr/dt = −∂E/∂r` 这类梯度流，**稳态条件只有一个：∂E/∂r = 0**。所谓「修正」是连续时间的弛豫过程，不是一步映射。
- tPC（G8）把这个关系**显式写成了一个等式**：单步时 *"is equivalent to taking only a single gradient step for each example"*，且 *"a single step is nowhere near enough to approach the optimum"*。他们把单步版本做成了电路（Fig 2C）并测了：**单步跟踪精度差于 5 步与完全收敛**，只给出「平滑了的轨迹」。
- 目标公式 `h = h⁻ + B(h⁻, r)` 恰好就是 tPC 的 Eq 14/Eq 26 的单步特例。

**含义（对新颖性不利）**：如果主张「一次残余修正就能替代迭代推理」，PC 文献已经给出了它的成立条件（线性凸 + 世界变化慢 + 热启动）和失效条件（非线性、快变、初值差）。**这个设计判断不是新的，是 tPC 明确讨论过的**。

### 4.2 增益在 PC 里是「给定的」，不是学出来的——但这也正好是目标公式的缝隙

- Rao & Ballard 1999：增益 = 精度 σ⁻²/σ_td⁻² 与 (∂f/∂x)ᵀUᵀ 的乘积，**精度是超参**。
- Rao & Ballard 1997：增益 = 协方差矩阵导出的 EKF 增益，**协方差在线传播但增益不是函数逼近器**。
- Friston 2005 / Whittington & Bogacz 2017：增益 = 1/Σ_i（逐单元精度），学习的是**参数 θ，不是增益**。
- tPC：K = Σ_x Cᵀ[CΣ_x Cᵀ+Σ_y]⁻¹，**Σ 可以学、可以编码进 A/C，但 K 的形式固定**。

**含义**：`B(h⁻, r)` 作为**可训练网络**确实是 PC 里没有的东西。**但这不是 PC 的新颖性来源**——在滤波文献里它就是「学习式增益」（KalmanNet, Revach et al. 2022, arXiv:2107.10043；Deep Kalman Filter 一类），在序列模型里它就是「门控 RNN 的更新单元」。所以这一点只在「PC 框架内」新，在方法层面不新。

### 4.3 tPC 与 Kalman 的确切差异，以及它的代价

- 差异只有一条：tPC 假设上一步后验是 **Dirac δ**（式 23），于是先验方差从 `AΣ_{k−1}Aᵀ + Σ_x` 退化为 `Σ_x`，**后验方差完全不传播**。作者原话：*"predictive coding ignores the uncertainty estimated at the previous step"*、*"does not propagate the uncertainty estimations"*。
- **代价（已报告）**：增益因此**不能随不确定度自适应**。Kalman 在观测噪声突变/先验变差时会自动降权，tPC 不会——它的 K 只由固定的 Σ_x、Σ_y 与 C 决定。作者的辩护是「在 benchmark 上打平」，但这是**在平稳高斯噪声设定下**的实证结论，不是一般结论。
- **对目标公式的含义**：目标公式的 B 如果只依赖 (h⁻, r)，**结构上等价于 tPC/固定增益**；若 B 能隐式携带不确定度（例如 r/h⁻ 的统计量），**才**回到 Kalman 的自适应增益路线——而那条路线已经有 EKF 与 KalmanNet。中间没有空位。

### 4.4 PC 训练慢/不稳：直接命中「用 PC 式规则做在线学习」的可行性

- **收敛性**（G9）：小权重/小初始化时才保证收敛；之外可发散（Neimark–Sacker 分岔，不稳定不动点 + invariant curve）。
- **优化成本**（G10）：*"IL is computationally demanding, and without memory-intensive optimizers like Adam, IL may converge to poor local minima"*。
- **可扩展性**（G11）：PC 只在小型卷积 + CIFAR10 规模上与 BP 可比；层间能量差几个数量级；状态学习率必须远小于 1；Adam 下隐藏宽度会决定稳定 γ 区间（BP 无此现象）。

**含义**：目标公式若把「PC 式局部修正」当作卖点，必须回答：**为什么它不需要 Adam、不需要多层迭代、不会能量失衡**？现有证据是反对的。

### 4.5 PredNet 的实测失效（对「一步修正」的另一种反证）

- 外推崩溃：*"the model naturally breaks down when extrapolating further into the future"*；要外推必须用 15 步损失重新微调（前 10 步喂真值，后 5 步喂预测）。
- 时间打乱对照：把输入时间顺序打乱后，模型只输出「模糊的上一帧」——说明它学到的预测**强依赖时序连续性**，而不是学会了动力学。
- 分层损失权重：**每层等权**的损失 *"considerably degraded predictions"*；最优模型只压最底层误差（另配低一个数量级的 λl>0=0.1 只稍差）。**这与「误差逐层局部地改写状态」的直觉相反**。
- 架构对照（附录 5.1）：逐元素相减（E = A − Â）确实有用、正负分裂有小幅增益——即**误差的表示形式**比「如何用误差改状态」更决定性能。

### 4.6 这类结构的已知失效模式清单（可直接写进 risk 分析）

1. **精度误设** → 固定增益：观测噪声被低估时，B 会放大离群新息（Kalman 的经典失效；PC 用固定 Σ 时同样）。
2. **单步 Euler 的步长条件**：tPC 的 Eq 13/14 里 Δt/τ 是步长，**Δt/τ 稍大就振荡/发散**（需要小 γ；见 G11）。
3. **误差被「解释掉」**：状态若自由可变，可以把 r 压到 0 而学不到世界模型（Millidge 综述对无约束 autoencoder 的 identity 解评论）。
4. **层级能量失衡**：误差集中在最后一层，前层得不到有效更新（G11 实测差几个数量级）。
5. **迭代不收敛 / 分岔**：小网络即可出现 Neimark–Sacker（G9）。
6. **误差通道饱和**：PredNet 用 ReLU 分裂正负误差 + 最低层 SatLU 限幅；不加限幅的线性残差在预测失败时会爆。
7. **Δ 与离散更新的错配**：真实 Δ 变化时（掉帧、变帧率），固定 A 的离散模型与「等间隔假设」不一致——**这是目标公式唯一有明显立足点的地方，但 PC 文献完全没有处理过它**。

---

## 5. 核对失败与更正记录（诚实项）

1. **"Pinchetti et al., Benchmarking PC Networks (ICML 2022)" 归属错误。** PMLR v162（ICML 2022）中检索不到该文；ICML 2022 的 PC 论文是 Frieder & Lukasiewicz 的 *(Non-)Convergence Results for Predictive Coding Networks*（PMLR 162:6793–6810）。Pinchetti 的 benchmark 实为 **arXiv:2407.01163（2024）**，作者列表见 G11；Pinchetti 2022 年的 NeurIPS 论文是 *Predictive Coding beyond Gaussian Distributions*。**结论：任务描述里的年份/会议按后者修正，不影响任何技术判断。**
2. **Rao & Ballard 1997 版本**：抓到的是标注 "To appear in Neural Computation" 的技术报告版（罗切斯特 TR），**卷期页码未核到**（PubMed 记录 9161021 存在但未逐页核对），因此正文只写年份与刊名。
3. **Friston 2005 卷期页码未核到**：PDF 页眉只能确认 Phil. Trans. R. Soc. B (2005) 与页 816/821–823，故不写 360:815–836 这类未经核实的页码。
4. **Millidge 综述（G7）的公式未取**：PDF 数学符号抽取不可靠，正文只引用可核实的英文结论，**没有转写它的方程**。
5. **Frieder & Lukasiewicz（G9）为 [仅摘要级]**：只拉了 PMLR 摘要页，未读正文 PDF 的证明。
6. **tPC 的 bioRxiv 正式发表版**：任务要求「找正式发表版」，已找到 = **PLOS Computational Biology 2024, 20(4): e1011183**（全文已核）。
7. **PredNet 的 "truncate"**：正文没有误差截断；只有 10 帧序列的截断 BPTT。若 lead 指的是「误差是否被截断/限幅」，答案是：最低层用 SatLU(p_max) 限幅，其余层无。

---

## 6. 最接近的先例排序与「还剩什么」

**按与目标公式的逐项吻合度排序：**

1. **tPC 2024（G8）Eq 26 / Eq 14** —— 逐字对应：先验、观测新息、固定增益加性修正。**唯一缺 Δ，且 B 不是学出来的函数。**
2. **Rao & Ballard 1997（G2）EKF** —— 先验、innovations、增益加性修正；缺 Δ；增益由协方差而非学习给定。
3. **Kalman 滤波本身**（经 tPC 的 Eq 21–22 核到） —— 目标公式的线性特例，1960 年。
4. **PredNet 2017（G6）** —— 误差进循环状态、逐层逐像素局部；但更新是门控 RNN 而非 h⁻+B，且无 Δ。
5. **Rao & Ballard 1999 / Friston 2005 / Whittington & Bogacz 2017** —— 确立「误差改状态」的语言，但无时间先验。

**因此，若目标是争取原创性，可辩护的只剩两点（且都很窄）：**

- **(i) B 的函数形式**：让增益成为 `B_φ(h⁻, r)`，即**状态与残差都参与的非线性增益**。文献里没有 PC 工作这么做；EKF 让增益依赖 h⁻（雅可比）但不依赖 r 的**数值**；KalmanNet 让增益由 RNN 学出但不以 (h⁻,r) 的显式形式为输入。**可辩护，但新颖性只值一个模块，不值一个框架。**
- **(ii) Δ 的显式注入**：把真实时间间隔作为 Φ 的输入 `Φ_Δ`，使同一网络在变帧率/掉帧下保持语义。**PC 文献零先例**（E 列全 ✗）；但必须承认这不是 PC 的延伸，而是时间感知序列模型（Neural ODE / ODE-RNN / 时间门控 RNN）的领域，那边大概率已有等价物（本文件不覆盖，需由其他专题确认）。

**明确不成立的新颖性主张（会被审稿人一击必杀）：**

- ✗ 「用预测误差修正隐状态」——Rao & Ballard 1999 起就是 PC 的定义。
- ✗ 「先验状态 + 观测残差 + 加性修正」的三段式——Rao & Ballard 1997 的 EKF 与 tPC 2024 的 Eq 26 完全相同。
- ✗ 「单步/摊销式修正替代迭代推理」——tPC 明确分析过（Eq 14 + Fig 2C），PredNet 是工程化先例。
- ✗ 「分层/局部/稀疏的误差通道」——PC 全系（precision weighting、正负误差群体）与 PredNet 都已做。

---

## 7. 链接清单（除标注外均经 tavily_extract 实际拉取正文）

1. Rao & Ballard 1999 — https://homes.cs.washington.edu/~rao/Rao-Ballard-NN-1999.pdf
2. Rao & Ballard 1997 — https://homes.cs.washington.edu/~rao/dynrecNC97.pdf
3. Friston 2005 — https://www.fil.ion.ucl.ac.uk/~karl/A%20theory%20of%20cortical%20responses.pdf
4. Friston 2008 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000211
5. Whittington & Bogacz 2017 — https://www.bndu.ox.ac.uk/sites/default/files/pdf_files/Whittington%20Bogacz%202017_Neural%20Comput.pdf
6. PredNet — https://arxiv.org/pdf/1605.08104 （abs: https://arxiv.org/abs/1605.08104）
7. Millidge/Seth/Buckley 2022 — https://arxiv.org/pdf/2107.12979
8. tPC 2024 (PLOS CB) — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183
9. tPC 预印本 (bioRxiv 2023) — https://www.biorxiv.org/content/10.1101/2023.05.15.540906v1.full
10. Tang, Barron, Bogacz NeurIPS 2023（tPC 序列记忆；**仅检索结果级**，未单独 extract） — https://proceedings.neurips.cc/paper_files/paper/2023/file/8a8b9c7f979e8819a7986b3ef825c08a-Paper-Conference.pdf
11. Frieder & Lukasiewicz 2022 — https://proceedings.mlr.press/v162/frieder22a.html
12. Alonso, Krichmar, Neftci 2024 — https://ojs.aaai.org/index.php/AAAI/article/view/28954/29814
13. Pinchetti et al. 2024 — https://arxiv.org/abs/2407.01163 （全文 https://arxiv.org/html/2407.01163v1）
14. Pinchetti & Bogacz NeurIPS 2022 — https://proceedings.neurips.cc/paper_files/paper/2022/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html
15. KalmanNet（学习式增益先例；**仅检索结果级**，未 extract，故正文只作方向性提及） — https://arxiv.org/abs/2107.10043

