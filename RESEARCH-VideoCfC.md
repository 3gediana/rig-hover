# 流式视觉决策网络：文献、数学基础与推导任务

> 建档日期：2026-11-03。  
> 状态：**文献收集第一轮完成；尚未开始原创数学推导、代码实现或实验。**  
> 接续规则：上下文压缩后，先按第 11 节进行数学建模和证明；在用户明确要求之前，不训练模型、不跑真机实验、不写实现代码。  
> 本文取代此前把 `CNN+CfC` 当成已定架构的研究提案。此前的性能比较和实验计划不构成当前结论。

---

## 0. 研究目标

研究一种以有时间戳的视频流为输入、逐时刻输出决策、并允许视觉编码与时序决策模块分别替换和部署的网络系统。用户特别要求先从数学上理解 CNN、RNN、LSTM、GRU、Transformer、Mamba、CfC 等结构的设计理由，再讨论是否值得构造新网络。

本轮的产物是**研究章程与文献地图**，不是新网络的完整推导。下一轮应建立统一数学问题，推导离散序列模型和连续时间模型的关系，尝试证明明确的结构性质，并写出每个结论的假设和反例边界。经验性能问题不能伪装成数学定理。

### 0.1 目标性质

- 输入是按观测时间到达的图像序列，而不是必须等到完整视频后才能处理的固定长度张量。
- 在时刻 `t_i` 的输出只能依赖当前和历史图像、时间戳及内部状态，不依赖未来帧。
- 视觉编码器与时序决策器在软件接口、状态传递、特征缓存和部署上可分开替换。
- 训练器支持片段分类、逐时刻状态或事件标签；输出任务必须和标签时间分辨率相符。
- CfC 是优先研究的候选时序核心，但它与 GRU、LSTM、因果 TCN、Transformer/SSM 等的相对优劣尚未确立。
- “像 YOLO 一样易训练”指数据格式、训练入口、断点续训、评估及结果可视化等工具体验，不要求采用目标检测的网络结构。

### 0.2 非目标

- 当前不决定最终网络一定使用 CfC，也不把 `CNN+CfC` 本身称为新架构。
- 当前不训练、不评估、不跑数据探针、不访问机器人、不修改固件。
- 当前不证明某个模型在真实视频上准确率更高、延迟更低或更抗掉帧。此类结论需要数据和实验。
- 当前不让神经网络重复学习能够由明确、可审计的确定性规则处理的逻辑。
- 当前不把在线动作识别指标当作机器人控制验收。

---

## 1. 文献事实、数学结论与经验结论

后续论证必须区分以下三类语句。

| 类别 | 可以支持什么 | 不可以支持什么 |
|---|---|---|
| 文献事实 | 某篇论文定义了某结构、报告了某数据集/设置/结果 | 不可把特定数据集结果外推为本项目必然结果 |
| 数学结论 | 在列明假设下，从定义推出不变量、等价性、误差界或不可能性 | 不可省略假设，或把局部模型性质写成任意网络的性质 |
| 经验结论 | 在指定数据、预处理、硬件、划分和随机性下测得的表现 | 不可仅凭结构直觉或数学定义断言真实任务获胜 |

“连续时间更自然”“更适合不规则视频”“参数更少”“更易部署”等词，若没有附带定义与条件，只是动机或假设，不是证明。

---

## 2. 任务的统一数学表述

令观测为

\[
\mathcal S = \{(I_i,\tau_i,q_i)\}_{i=0}^{N}, \qquad
\tau_0 < \tau_1 < \cdots < \tau_N,
\]

其中 `I_i` 是第 `i` 帧，`τ_i` 是该帧的实际呈现时间戳，`q_i` 可表示可选的图像质量或观测可信度。时间戳单位应固定为秒；采集帧率不是时间戳的替代物。

视觉编码器给出

\[
z_i = \phi_\theta(I_i),
\]

时序系统维护状态 `h_i` 并更新

\[
h_i = U_\psi(h_{i-1},z_i,\delta_i,q_i),
\qquad \delta_i = \tau_i-\tau_{i-1},
\]

逐时刻输出为

\[
\hat y_i = G_\omega(h_i,z_i).
\]

严格因果性要求 `\hat y_i` 只依赖 `I_0,...,I_i`、`τ_0,...,τ_i`、对应质量信息和过去状态。输出 `y_i` 可以是动作类别、事件分数、几何量或控制目标，但这些任务的监督目标与评价指标不同。

一个实现可有统一的状态接口：

```text
reset(session_id)
step(frame, pts_seconds, state) -> per_time_decisions, new_state
```

这只是接口候选，不是已实现功能。批量训练须与连续 `step` 推理使用相同的状态更新语义，不能在训练窗口边界静默清零而推理时持续保留状态。

### 2.1 两种时间语义不可混淆

1. **观测序号时间**：模型看到第 1、2、3 个样本。普通离散 RNN 通常按这个索引更新。
2. **物理时间**：样本之间实际经过了 `δ_i` 秒。连续时间模型或显式 time-aware 模型可将它作为状态转移的一部分。

只把 `δ_i` 拼进输入向量，不等于模型具有某种连续时间演化定律；将 `δ_i` 传给 CfC 的 `timespans` 也不意味着 CfC 自动读取 PTS，调用方必须自行计算并传入。

### 2.2 观测之间的状态演化

后续推导要先选定语义，而不能将其藏在 API 中。一个通用的混合状态模型可写为

\[
h(\tau_i^-) = P_{\delta_i}(h(\tau_{i-1})), \qquad
h(\tau_i) = V(h(\tau_i^-),z_i),
\]

其中 `P_δ` 描述无新观测期间状态如何演化，`V` 描述新图像到达时如何修正状态。

另一类更新直接写为 `h_i=U(h_{i-1},z_i,δ_i)`，将间隔影响折入观测时刻更新。二者不应先验等同。后续数学研究需说明 CfC 的 `timespans` 属于哪种语义、其与 `P_δ`/`V` 分解在何种假设下可能等价、以及何时不等价。

---

## 3. 传统网络的设计理由与数学骨架

本节先记录通用数学骨架，不声称任一结构天然最优。

### 3.1 CNN：局部连接、权值共享与平移等变性

二维卷积层可写为

\[
(\operatorname{Conv}_K x)(p) = \sum_{r\in\mathcal K} K(r)x(p-r).
\]

局部核限制每个输出位置依赖的输入邻域；相同核在不同空间位置共享参数。忽略边界填充影响时，对平移算子 `T_a`，卷积满足

\[
\operatorname{Conv}_K(T_a x)=T_a(\operatorname{Conv}_K x),
\]

即平移等变，而不是平移不变。池化或全局汇聚才可能产生一定不变性，但会损失位置信息。步幅、边界填充和非均匀采样也会破坏严格等变关系。

设计动机：图像像素存在空间邻域结构；局部连接和权值共享比全连接层更有效地表达该结构。限制：CNN 本身不定义跨帧记忆或真实时间演化。桌面画面中的手部较小，过度下采样可能抹掉局部信息；“全局池化一定合适”不是数学结论。

参考：LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998, DOI: https://doi.org/10.1109/5.726791

### 3.2 普通 RNN：有限维状态递归

标准递归形式为

\[
h_i = F_\theta(h_{i-1},x_i), \qquad y_i=G_\theta(h_i).
\]

它将任意长历史压入固定维状态。参数跨时间共享，逐步推理只需保留状态，不必保存全部历史。反向传播沿递归链展开，长序列梯度含有多次 Jacobian 乘积：

\[
\frac{\partial h_n}{\partial h_k}
= J_n J_{n-1}\cdots J_{k+1},\qquad
J_j=\frac{\partial F_\theta(h_{j-1},x_j)}{\partial h_{j-1}}.
\]

若这些乘积的范数长期远小于或大于 1，就会出现梯度消失或爆炸的条件风险。这是对梯度链的分析，不是说任一 RNN 训练必然失败。

设计动机：对序列建立状态递归和参数共享。普通离散 RNN 的默认一步对应一次观测更新，不自动知道相邻观测相隔多少秒。

参考：Elman, “Finding Structure in Time,” *Cognitive Science*, 1990, DOI: https://doi.org/10.1207/s15516709cog1402_1

### 3.3 LSTM：用门控和加性记忆路径改善长期状态传递

一种常用 LSTM 形式为

\[
\begin{aligned}
i_i &= \sigma(W_i x_i+U_i h_{i-1}+b_i),\\
f_i &= \sigma(W_f x_i+U_f h_{i-1}+b_f),\\
o_i &= \sigma(W_o x_i+U_o h_{i-1}+b_o),\\
\tilde c_i &= \tanh(W_c x_i+U_c h_{i-1}+b_c),\\
c_i &= f_i\odot c_{i-1}+i_i\odot\tilde c_i,\\
h_i &= o_i\odot\tanh(c_i).
\end{aligned}
\]

细胞状态的加性通道允许在特定门值下保留较直接的梯度路径；输入门、遗忘门和输出门学习写入、保留和暴露哪些信息。它改善某些长期依赖的优化路径，但不保证任意长期依赖都可学，也不赋予真实时间间隔语义。

参考：Hochreiter & Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 1997, DOI: https://doi.org/10.1162/neco.1997.9.8.1735

### 3.4 GRU：合并状态与门控更新

一种常见 GRU 参数化为

\[
\begin{aligned}
r_i &= \sigma(W_r x_i+U_r h_{i-1}),\\
z_i &= \sigma(W_z x_i+U_z h_{i-1}),\\
\tilde h_i &= \tanh(W_h x_i+U_h(r_i\odot h_{i-1})),\\
h_i &= (1-z_i)\odot\tilde h_i+z_i\odot h_{i-1}.
\end{aligned}
\]

重置门 `r_i` 调节候选状态使用的旧信息，更新门 `z_i` 在候选状态与旧状态之间插值。GRU 将 LSTM 的若干门和状态合并为较简洁的递归单元。不同论文/框架的 reset/update 约定和矩阵顺序可能不同，公式比较必须对齐约定。

GRU-D 针对缺失观测建模遮罩与距上次观测的时间，并用学习到的衰减项处理缺失；它是时间间隔建模的相关基线，但不等于 CfC。

参考：Cho et al., “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation,” EMNLP 2014: https://arxiv.org/abs/1406.1078 ；Che et al., “Recurrent Neural Networks for Multivariate Time Series with Missing Values,” *Scientific Reports*, 2018: https://doi.org/10.1038/s41598-018-24271-9

### 3.5 Transformer：通过内容相似度进行并行全局交互

给定查询、键和值矩阵，单头注意力写为

\[
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
\]

其中 `M` 是掩码；因果掩码令时刻 `i` 不可读取 `j>i` 的键值。多头、前馈层、残差和归一化构成常用 Transformer 块。

设计动机：任一位置可按内容直接访问其他位置，序列位置计算可高度并行。标准全注意力对长度 `N` 的注意力矩阵有二次空间/计算项；因果自回归推理可缓存历史键值，但缓存会随上下文增长。位置编码通常描述序列索引或位置，除非专门定义，不能假定它代表真实帧间秒数。

参考：Vaswani et al., “Attention Is All You Need,” NeurIPS 2017: https://arxiv.org/abs/1706.03762

### 3.6 结构化状态空间与 Mamba：线性状态动力学及输入选择

线性连续状态空间模型可写为

\[
\dot h(t)=Ah(t)+Bu(t), \qquad y(t)=Ch(t)+Du(t).
\]

对固定步长 `Δ`，可按选定离散化方法得到递归更新；例如零阶保持下

\[
\bar A_\Delta=e^{A\Delta},\qquad
\bar B_\Delta=\int_0^\Delta e^{A(\Delta-s)}B\,ds,
\qquad h_i=\bar A_\Delta h_{i-1}+\bar B_\Delta u_i.
\]

S4 等结构化状态空间模型利用特定矩阵结构使长序列训练可计算。Mamba 引入输入相关的选择参数，并用选择性扫描实现序列建模。其内部输入生成的 `Δ_i` 是模型参数化中的选择/离散尺度，不自动等于相机 PTS 差值 `τ_i-τ_{i-1}`。

因此严谨陈述是：**标准 Mamba 不保证状态转移按外部墙钟时间推进；可将时间信息显式加入输入或改变状态转移，但这种模型须单独定义和分析。**不能从这点推出“Mamba 不可能处理不规则时间”。VideoMamba 是视频 SSM 先例，但其离线/双向设置不能自动证明因果流式性能。

参考：Gu, Goel & Ré, “Efficiently Modeling Long Sequences with Structured State Spaces,” ICLR 2022: https://arxiv.org/abs/2111.00396 ；Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” COLM 2024: https://arxiv.org/abs/2312.00752 ；Li et al., “VideoMamba: State Space Model for Efficient Video Understanding,” ECCV 2024: https://arxiv.org/abs/2403.06977

### 3.7 LTC 与 CfC：连续时间状态及近似闭式更新

LTC 的状态由输入驱动的微分方程刻画。Hasani et al. 给出的一类形式为

\[
\frac{dx}{dt}= -[w_\tau+f(x,I,\theta)]\odot x + A\odot f(x,I,\theta),
\]

其中具体 `f`、连接和约束由网络实例决定。LTC 可用数值方法推进连续状态，代价与求解步数/求解器相关。

CfC 从特定 LTC 动力学出发，对积分项作近似，构造显式依赖时间间隔的闭式形式，避免每个更新都运行一般数值 ODE 求解器。准确陈述是“针对 LTC 类动力学的近似闭式更新”，不是“任意视频连续动力学的精确闭式积分”。CfC 的 `timespans` 必须由调用方提供；API 不会自行解析视频 PTS。

论文给出 CfC 在若干时间序列和视觉/决策设置的结果，也已有 CNN 感知模块与液态递归控制器的先例。因此 `CNN+CfC` 组合不是本项目足以单独主张的新颖性。

参考：Hasani et al., “Liquid Time-constant Networks,” AAAI 2021: https://arxiv.org/abs/2006.04439 ；Hasani et al., “Closed-form Continuous-time Neural Networks,” *Nature Machine Intelligence* 2022: https://doi.org/10.1038/s42256-022-00556-7 ，全文：https://arxiv.org/abs/2106.13898

### 3.8 Neural CDE：以连续控制路径驱动隐藏状态

Neural CDE 将观测插值成路径 `X(t)`，并定义

\[
dz(t)=f_\theta(z(t))\,dX(t).
\]

它直接以连续插值路径的变化驱动隐藏状态，目标之一是处理不规则采样和部分观测。时间戳、插值方案、求解方法均属于模型语义；不同插值会给出不同连续路径，不能忽略。

参考：Kidger et al., “Neural Controlled Differential Equations for Irregular Time Series,” NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html

---

## 4. 可先严格推导的统一连续时间关系

本节定义下一轮的推导入口，不预先声称 CfC 胜出。

考虑区间内输入 `u(t)=u` 为常数的线性连续状态系统：

\[
\dot h(t)=Ah(t)+Bu.
\]

对长度 `\delta\ge 0`，变参数公式给出

\[
h(t+\delta)=e^{A\delta}h(t)+\int_0^\delta e^{A(\delta-s)}B u\,ds.
\]

若将同一段恒定输入的区间拆成 `\delta_1+\delta_2` 两段，精确状态转移满足分段复合等于整段转移。这是连续自治流/线性系统的半群性质；后续需从矩阵指数与积分项完整写出证明，并逐条列出恒定输入、相同动力学和精确求解的假设。

对普通离散 RNN，若相同输入 `z` 连续更新 `n` 次，则状态为 `F_z^{\circ n}(h)`；单次更新是 `F_z(h)`。二者一般不相等，除非 `F_z` 满足额外条件，例如相应不动点/幂等条件。故离散索引更新一般对“同一物理时间内多采几帧”不具有自动不变性。

这只说明：**对特定连续动力学，显式时间转移可表达采样间隔和分段一致性；而不带真实时间语义的离散递归没有自动保证该性质。**它不证明某个连续模型更准确、CfC 满足精确半群性质、视觉决策的真实损失更低，或实际机器人一定获益。

---

## 5. 需要建立的核心命题与证明义务

下一阶段按命题而不是按网络流行度推进。每个命题都要写明假设、结论、证明、反例/失败条件。

### 命题 A：连续状态转移的分段一致性

在线性/自治 ODE 条件下证明精确流 `P_\delta` 满足

\[
P_{\delta_2}\circ P_{\delta_1}=P_{\delta_1+\delta_2}.
\]

之后说明在输入在拆分区间内变化、模型在观察时注入新观测，或更新含近似时，该等式如何变化。不可把精确 ODE 的性质直接移植到 CfC。

### 命题 B：离散 RNN 的采样次数依赖

对固定离散转移证明重复观测更新 `F_z^{\circ n}` 通常依赖观测次数。刻画何时它可能与物理间隔无关或退化成时间一致的半群；说明若训练/推理采样频率不同，状态语义为何可能改变。

### 命题 C：显式 `Δt` 输入和连续时间转移不是同一结构

分别定义：

\[
\text{feature-time model: } h_i=F(h_{i-1},z_i,\delta_i),
\]

以及

\[
\text{flow model: } h_i=P_{\delta_i}(h_{i-1})\text{ followed by }V(z_i).
\]

研究二者何时可相互表示、需要多大状态/函数类，是否有不可表示的例子。必须区分“把数字 `δ_i` 当特征”与“状态遵循具有时间转移结构的动力系统”。

### 命题 D：CfC 更新的数学保证范围

从原论文方程和指定实现版本出发，推导 CfC 中 `timespans` 进入更新的具体位置。逐项核对近似来源和假设；确定是否满足半群、单调性、稳定性或误差界，若不满足则构造简单反例。未经推导，不声称 CfC 时间重采样不变或精确积分。

同时区分 `timespans` 与额外特征 `δ_i`。二者可分别或同时提供，但存在信息重复和捷径风险；本项仅记录为后续消融设计，不在当前轮跑实验。

### 命题 E：门控、梯度路径和记忆

从 RNN、LSTM、GRU 的雅可比递推出发，分析门控和加性状态路径如何改变梯度乘积。结论要写成充分/必要条件或界，不可把“缓解梯度消失”写成“解决长期依赖”。对 CfC 的时间门控也要用同样的梯度链分析，不能从连续时间名称推断稳定性。

### 命题 F：注意力与状态压缩的计算/信息权衡

对固定上下文长度分析全注意力的矩阵成本；对因果缓存分析每步新增计算和缓存量。与固定维递归状态比较时，应区分计算复杂度、记忆复杂度、可表示函数类和实际精度。复杂度定理本身不能推出哪种模型能保留特定任务信息。

### 命题 G：模块化的定义

将视觉模块输出视为接口变量 `z_i`。证明何种条件下冻结编码器后可独立优化时序模块；说明联合优化中梯度链

\[
\frac{\partial L}{\partial\theta_{vision}}
=\sum_i\frac{\partial L}{\partial z_i}\frac{\partial z_i}{\partial\theta_{vision}}
\]

仍然耦合两模块。分别定义工程模块化（缓存、替换、部署）和优化解耦（参数梯度/目标函数分离），不混用术语。

### 命题 H：限制与不可能性

对固定有限维状态、有限精度/参数化和目标函数类，研究是否存在对任意长历史保持充分统计量的不可能性或容量下界。必须先定义目标历史函数族与误差度量；未给定这些对象前，“有限状态记忆不足”不是可证明的具体命题。

---

## 6. 研究候选架构，不是既定设计

```text
(I_i, τ_i, optional q_i)
        │
        ▼
VisionEncoder φθ: frame / ROI / multi-scale features
        │ z_i
        ▼
TemporalCore Uψ: RNN / LSTM / GRU / TCN / Transformer / SSM / CfC / CDE
        │ h_i
        ▼
Task-specific heads Gω: state / event / geometry / optional control
```

工程上的模块接口候选：

```text
VisionEncoder(frames, preprocessing) -> features, feature_metadata
TemporalCore.step(features, elapsed_seconds, state) -> state, temporal_features
TaskHead(temporal_features) -> task_outputs
```

- 视觉特征可能需要保留多尺度/ROI信息，不能先假定全局池化足以识别约 30–40 px 的桌面手部。
- CfC `timespans` 应从严格递增 PTS 显式导出。时间单位、重复/非单调 PTS、首帧、极端间隔和状态重置必须有明确定义。
- `Δt` 是否也拼入特征是可切换假设，不是默认设计。未来理论分析应分别写 `timespan-only`、`feature-Δt-only` 与两者同时提供的映射。
- 语义状态、事件、几何和控制头是不同任务。控制头需要对齐控制标签和独立的安全/执行评估。
- `VideoCfC`/`Stream-CfC` 仅作原型代号，不是原创性声明。

---

## 7. 任务与监督：输出粒度决定标签粒度

| 任务 | 输出 | 最低相符监督 | 主要注意事项 |
|---|---|---|---|
| Clip classification | 整段类别/多标签 | 片段级标签 | 不给出可靠事件起止时刻 |
| Online state/action recognition | 每个观测时刻的状态类别 | 帧/采样点标签或时间区间展开标签 | 只能使用当前和过去；按 session 留出 |
| Event detection | 事件分数及发生时间 | 稀疏事件时刻，最好有容差/区间 | 需明确 onset、延迟及重复事件合并规则 |
| Temporal localization | 事件区间 `[start,end]` | 时间边界标签 | 离线方法可能偷看未来；TAL 与在线任务分开 |
| Robot control prediction | `head_des/vx/vyaw` 或目标几何 | 时间对齐教师轨迹/状态 | 需要控制 MAE、时延、平滑与安全仲裁 |
| Weak clip supervision | 整段存在标签 | clip 标签 | MIL 定位是额外学习问题，不能默认输出精确边界 |

未标注不是负样本；数据格式需表达 `ignore` 和可选标签置信度。数据按完整 session/主体/场景划分，不能随机拆相邻帧。

在线动作检测文献关注因果逐时刻动作类别，不等价于机器人控制策略。MiniROAD 指出训练窗口状态与连续推理状态不一致会损害 RNN 训练，并研究非均匀时间步损失权重；其末步损失不是所有稀疏事件任务的通用最优解。

相关来源：

- LRCN：Donahue et al., CVPR 2015, CNN 特征与循环序列模型的成熟组合：https://arxiv.org/abs/1411.4389
- MiniROAD：An et al., ICCV 2023，在线动作检测与训练/推理状态差异：https://openaccess.thecvf.com/content/ICCV2023/papers/An_MiniROAD_Minimal_RNN_Framework_for_Online_Action_Detection_ICCV_2023_paper.pdf
- LSTR：Xu et al., NeurIPS 2021，在线长短期上下文：https://arxiv.org/abs/2107.03377
- OnlineTAS：Zhong et al., NeurIPS 2024，自适应在线上下文与分段连续性：https://proceedings.neurips.cc/paper_files/paper/2024/file/6c6c5fccf3c8661fcae219be7ca226f7-Paper-Conference.pdf
- ActionFormer：Zhang et al., ECCV 2022，anchor-free temporal localization；论文使用 focal 分类与 DIoU 边界回归，属于离线 TAL 参考，不是本原型的在线证明：https://arxiv.org/abs/2202.07925
- SCOAD：Ye et al., ACCV 2022，单帧 click 监督的在线动作检测：https://openaccess.thecvf.com/content/ACCV2022/html/Ye_SCOAD_Single-Frame_Click_Supervision_for_Online_Action_Detection_ACCV_2022_paper.html

---

## 8. 已收集的相关工作与本项目关系

| 工作 | 数学/任务贡献 | 对本研究的用处 | 不可据此推出 |
|---|---|---|---|
| LRCN (2015) | CNN 逐帧特征 + RNN 序列预测 | 证明模块化视觉-时序组合早已有之 | CNN+CfC 本身新颖 |
| TSM (ICCV 2019) | 通道沿时间移位；提供单向在线版本 | 因果视频建模、低算力先例 | CfC 必然更优 |
| Neural CDE (NeurIPS 2020) | 连续控制路径驱动态，面向不规则/部分观测序列 | 连续时间对照及路径插值形式化 | 任何插值/求解器都等价 |
| LTC (AAAI 2021) | 输入相关时间常数的连续神经动力学 | CfC 的动力学来源 | LTC 数值解必然适合实时硬件 |
| LSTR (NeurIPS 2021) | 在线因果长短期上下文记忆 | 流式任务基线 | 离线注意力可以直接用于实时 |
| CfC (NMI 2022) | LTC 特定动力学的近似闭式网络；显式时间依赖 | 研究带 elapsed time 的递归更新 | 一般 ODE 精确闭式解；本视频任务性能领先 |
| ActionFormer (ECCV 2022) | 多尺度 temporal point 分类与边界回归 | 未来离线 TAL 头的参考 | 可直接替换在线决策头 |
| MiniROAD (ICCV 2023) | 简化 RNN 在线动作检测；训练推理上下文差异分析 | 状态预热/训练协议必须控制 | 末步损失对所有任务都最佳 |
| VideoMamba (ECCV 2024) | Mamba 类视频时空建模 | 视觉/时序 SSM 相关工作 | 标准模型自动用外部 PTS，或已证明因果流式能力 |
| MODE (2026 预印本) | 低秩 ODE 与 Mamba 的不规则时间序列方案 | Mamba 时间处理是活跃研究问题 | CfC 是 MODE 的等价/修正版 |
| CRU (ICML 2022) | 连续-离散状态空间建模不规则序列 | 连续观测时间模型的其他参照 | 可直接代表一般高维视频决策 |
| Δt-Mamba3D (2025 预印本) | 纵向医学图像的显式时间间隔建模 | 时间感知 SSM 已有相关先例 | 是常规在线视频动作检测结果 |

其他条目与链接见第 12 节。

---

## 9. 数据/训练工具资料：只为之后的工具设计服务

YOLO 的易用性来自统一数据配置、`train/val/predict/export/benchmark` 流程、预训练初始化、断点续训、指标和结果可视化。Stream-CfC 若进入实现阶段，应借鉴这些用户体验，而不是复刻检测框架。

### 9.1 候选自有数据格式

每个视频记录 URI、内容哈希、PTS/帧索引、时长、采集 session、split、预处理版本、许可证和标签。标签可按任务承载 clip、区间、事件时刻、逐帧状态或控制目标；`ignore` 与 background 必须分开。特征缓存标注视觉权重和预处理版本，防止异源特征混用。

### 9.2 工具与维护风险

- TorchCodec 提供 PyTorch 视频解码与按时间取帧 API；实际 PTS/帧选取仍需落入清单/测试：https://pytorch.org/blog/torchcodec
- PyTorchVideo 有视频数据与模型组件，但其安装说明存在较旧 Python/PyTorch 建议；应验证并锁定环境，或只借用模型代码：https://github.com/facebookresearch/pytorchvideo/blob/main/INSTALL.md
- MMAction2 覆盖动作分类、检测和定位，但 MMCV/PyTorch/CUDA 扩展匹配会提高安装维护成本：https://github.com/open-mmlab/mmaction2/blob/main/docs/en/get_started/installation.md
- Lightning 可提供 checkpoint/resume、训练循环和日志，不替代任务定义：https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
- Ultralytics 的训练命令体现了用户体验参考；其许可证需按部署/分发用途审核：https://docs.ultralytics.com/modes/train 与 https://www.ultralytics.com/license
- CVAT/FiftyOne 可支持视频时间线和数据集检查；插值标签不等于人工确认的真实边界：https://www.cvat.ai/academy/track-mode 与 https://docs.voxel51.com/user_guide/using_datasets.html
- Decord/PyAV/TorchCodec 均须经同一组视频的帧数、PTS、颜色、尺寸一致性检查；Decord 的维护状态需单独核实：https://github.com/dmlc/decord ，PyAV：https://github.com/PyAV-Org/PyAV
- VideoMAE 模型须遵守各 checkpoint 的 processor 配置；上游仓库/权重许可证要分开检查：https://huggingface.co/docs/transformers/en/model_doc/videomae 与 https://github.com/MCG-NJU/VideoMAE
- Torchvision 视频 API 标为 Beta，不能成为稳定公开接口假设：https://docs.pytorch.org/vision/main/models.html
- PyAV/FFmpeg 分发涉及构建配置与 FFmpeg 许可证义务：https://www.ffmpeg.org/legal.html

若实现，封装窄的 `DecoderAdapter`、`EncoderAdapter`，不把第三方框架张量/配置直接变成项目数据契约。当前不安装依赖、不选定后端。

---

## 10. 当前项目相关的既有事实与边界

以下是已有工程记录，不是新网络的数学证据：

- 桌面摄像头链路历史测量约 `42.8–42.9 ms p50`，约 `21 Hz`；相机帧周期约 `32 ms`。请以 `LATENCY-MAP.md` 和对应测量日志为准。
- 受控桌面数据的手部约 `30–40 px`；旧 HaGRID 手势分类器在桌面域失败，不能拿占位模型表现推断新结构。
- 21 Hz 下 150 ms 事件间隔约 3 个采样点，事件形状可能欠采样、时间顺序辨别风险高；这不是数学上的不可检测定理，需条件采样研究。
- 预训练脸/姿态检测器若在部署域保持召回，可能降低框标注需求；这不免除端到端模型的事件/状态/控制监督，也不保证诊断时不需要 ROI 标签。
- 用户指出，用符号规则已能判定的确定事件不应再交给网络学习；网络应处理图像到语义等不可直接规格化部分。
- 设备、固件、Label Studio 凭据等运行态信息不在本数学研究章程中重述；查 `AGENTS.md` 和本地受控环境记录，避免把可能变化的状态写成当前事实。

---

## 11. 下一轮任务：纯数学分析，不做实验

上下文压缩后，首先读完本文，再从第一条命题开始。目标是形成可审查的数学推导，不追求用“模型名称”替代推理。

### 第一步：定义性质

先精确定义：

- 视频观测流与真实时间；
- 因果输出；
- 隐状态的语义；
- 采样加密、采样稀疏、重复观测和缺帧；
- 状态转移的时间一致性/半群性质；
- 误差度量与允许的模型函数类。

没有这些定义，不谈“不规则采样鲁棒”或“连续模型更自然”。

### 第二步：从最简单系统开始推导

从标量/矩阵线性 ODE 开始，使用变参数公式推导精确 `P_δ`，证明分段复合关系；再构造固定步离散递归作为对照。先完成可逐行核查的代数，不引用模型性能。

### 第三步：逐个映射传统网络

把 RNN/LSTM/GRU 视为离散状态更新，把 Transformer 视为有限历史内容寻址，把 SSM/Mamba 视为参数化状态递推，把 LTC/CfC 视为特定连续动力学及其近似更新。对每一个结构回答：

1. 它的状态变量是什么？
2. 时间是观测索引还是实时时间？
3. 哪些量能并行计算？
4. 状态/缓存随序列长度如何增长？
5. 哪个性质由公式保证，哪个只由训练数据希望学得？
6. 哪些等价映射存在，哪些必须给出反例或条件？

### 第四步：严格审查 CfC

回到 Hasani et al. 论文和指定实现版本，逐式核对 LTC 方程、积分近似、CfC 门控表达式、`timespans` 的传入方式。对“半群性质”“重采样一致性”“任意时间间隔”“有限状态记忆”等说法逐一证明或否定。CfC 若不满足拟证明的性质，应明确缩小命题，不用术语遮掩。

### 第五步：讨论能否提出新结构

只有完成上述定义和证明后，才尝试设计超出标准组件拼接的机制。候选方向包括连续流转移与观测更新分离、时间戳与观测置信度共同影响状态、可证明的采样细化一致性等；这些目前都只是研究问题，不是本项目已设计的新方法。

### 数学脚本规则

当前不运行脚本。若后续解析推导完成后需要脚本，只能做符号化/数值化的数学 sanity check，例如矩阵指数恒等式的随机矩阵检查或近似误差计算；脚本结果不能替代证明，也不能做训练、超参搜索、真实视频评估或硬件基准。

---

## 12. 文献索引

### 网络与连续时间模型

1. LeCun et al., CNN/局部共享基础，1998：https://doi.org/10.1109/5.726791
2. Elman, 简单递归网络，1990：https://doi.org/10.1207/s15516709cog1402_1
3. Hochreiter & Schmidhuber, LSTM，1997：https://doi.org/10.1162/neco.1997.9.8.1735
4. Cho et al., GRU 系列来源，EMNLP 2014：https://arxiv.org/abs/1406.1078
5. Vaswani et al., Transformer，NeurIPS 2017：https://arxiv.org/abs/1706.03762
6. Che et al., GRU-D，2018：https://doi.org/10.1038/s41598-018-24271-9
7. Gu, Goel & Ré, S4，ICLR 2022：https://arxiv.org/abs/2111.00396
8. Hasani et al., LTC，AAAI 2021：https://arxiv.org/abs/2006.04439
9. Hasani et al., CfC，NMI 2022：https://doi.org/10.1038/s42256-022-00556-7 ; https://arxiv.org/abs/2106.13898
10. Kidger et al., Neural CDE，NeurIPS 2020：https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html
11. Schirmer et al., Continuous Recurrent Unit，ICML 2022：https://proceedings.mlr.press/v162/schirmer22a.html
12. Gu & Dao, Mamba，COLM 2024：https://arxiv.org/abs/2312.00752
13. MODE，低秩 ODE + Mamba，2026 预印本：https://arxiv.org/abs/2601.00920
14. Zhou et al., Δt-Mamba3D，2025 预印本：https://arxiv.org/abs/2510.19003
15. Schirmer et al., “Modeling Irregular Time Series with Continuous Recurrent Units,” ICML 2022：https://arxiv.org/abs/2111.11344 ; https://proceedings.mlr.press/v162/schirmer22a.html

### 视频与因果时序任务

15. Donahue et al., LRCN，CVPR 2015：https://arxiv.org/abs/1411.4389
16. Lin et al., Temporal Shift Module，ICCV 2019：https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.pdf
17. Kondratyuk et al., MoViNets，CVPR 2021：https://arxiv.org/abs/2103.11511
18. Xu et al., LSTR，NeurIPS 2021：https://arxiv.org/abs/2107.03377
19. An et al., MiniROAD，ICCV 2023：https://openaccess.thecvf.com/content/ICCV2023/papers/An_MiniROAD_Minimal_RNN_Framework_for_Online_Action_Detection_ICCV_2023_paper.pdf
20. Li et al., VideoMamba，ECCV 2024：https://arxiv.org/abs/2403.06977
21. Zhong et al., OnlineTAS，NeurIPS 2024：https://proceedings.neurips.cc/paper_files/paper/2024/file/6c6c5fccf3c8661fcae219be7ca226f7-Paper-Conference.pdf
22. Zhang et al., ActionFormer，ECCV 2022：https://arxiv.org/abs/2202.07925
23. Ye et al., SCOAD，ACCV 2022：https://openaccess.thecvf.com/content/ACCV2022/html/Ye_SCOAD_Single-Frame_Click_Supervision_for_Online_Action_Detection_ACCV_2022_paper.html
24. Shou et al., AutoLoc，ECCV 2018：https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zheng_Shou_AutoLoc_Weakly_Supervised_Temporal_ECCV_2018_paper.pdf

### 工具链

25. TorchCodec, PyTorch video/media decoding：https://pytorch.org/blog/torchcodec
26. PyTorchVideo：https://pytorchvideo.readthedocs.io/
27. MMAction2：https://mmaction2.readthedocs.io/en/latest/get_started/overview.html
28. PyTorch Lightning checkpoint/resume：https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
29. Ultralytics train workflow：https://docs.ultralytics.com/modes/train
30. CVAT video track interpolation：https://www.cvat.ai/academy/track-mode
31. FiftyOne video/temporal labels：https://docs.voxel51.com/user_guide/using_datasets.html
32. PyAV：https://github.com/PyAV-Org/PyAV
33. VideoMAE processor/model docs：https://huggingface.co/docs/transformers/en/model_doc/videomae

---

## 13. 研究纪律

1. 优先给定义、假设、推导和反例，再谈架构命名。
2. 只在假设完整时写“证明”；经验直觉用“假设/猜想/待验证”。
3. 任何连续时间优势都要写出相对于哪个离散模型、在哪种输入保持规则、在哪种误差度量下成立。
4. CfC/Mamba/Transformer 的复杂度比较须同序列长度、状态维度、参数量、硬件实现口径；跨论文数字不能直接当公平比较。
5. “模块化”“端到端”“在线”“连续时间”“解耦”都需给出操作定义。
6. 数学推导阶段不训练、不做真机测试、不运行模型基准。用户若之后明确要求实验，再将实验设计与证明分开记录。
7. 机器人安全仲裁与控制约束不能由离线分类指标替代。
8. 既有占位模型失败只说明那个模型/数据/训练设置失败，不证明 CfC、Mamba 或新结构的优劣。

---

## 14. 当前结论

- 研究问题是**带外部时间戳的因果视觉流如何递归地产生逐时刻决策**。
- CNN+时序核心是成熟组合；工程解耦有用，但不是自动的新数学结构。
- CfC 的特殊价值候选是显式 elapsed-time 参数与无通用数值 ODE 求解器；其近似更新是否满足希望的时间一致性，须从方程推导。
- Mamba 的输入相关选择步长不是相机实时时间；这构成比较问题，不构成不可能性结论。
- 理论可证明时间转移/状态更新的结构性质；是否提高真实视频识别、检测或控制性能属于经验问题，不能纯数学推出。
- **下一步唯一任务：在不跑实验的前提下完成第 5、11 节的定义和首批推导。**
