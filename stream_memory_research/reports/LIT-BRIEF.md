# LIT-BRIEF：流式视觉记忆网络（DSM）设计简报

> **来源**：`litreview/01–07`、`litreview2/01–09`、`LIT-REVIEW-CONTINUOUS-FASTWEIGHT.md`、`LIT-REVIEW-PREDICTIVE-STATE.md`（均 2026-11-03，纯网络检索 + 原文核对）。
> **对象**：DSM = 慢连续状态 h（精确半群 Δt 衰减）+ 快矩阵记忆 M（delta 写 + Δt 衰减）+ 内容寻址槽位。
> **标注**：〔定理〕原文有严格结论；〔论证〕原文有推导但非定理；〔实证〕有实验数据；〔缺口〕文档明确写"未找到"。
> **目标**：在"遮挡后长期记忆 / 多对象低串扰 / 事件检测 / 不规则 Δ / 因果在线前缀"五项上打赢单个 CfC。

---

## 10 条设计结论

1. **写规则必须是"读-改-写"（delta），不能是加性外积。** 加性写 `M←M+αvkᵀ` 在重复/近邻 key 上单调累积干扰，而 Gated DeltaNet 的 S-NIAH 消融原文标题即 "**Decay hurts memory retention**"、数字是 DeltaNet 98.8 / Mamba2（有衰减无残差）30.4 / Gated DeltaNet 91.8（8K）。→ 主容量来自残差写，衰减只做干扰管理与数值稳定。
2. **(ρ, α) = (e^{−ΛΔ}, 1−e^{−ΛΔ}) 不是两个旋钮，是同一个 ODE 的两个函数。** Mamba 式(4)、Schöne 式(7)（事件流、真实 Δt）、EFLA 式(11)、ETH 讲义（"no discretization errors"）四处逐字给出；标量情形恒有 ρ+α=1。→ 正确形式是 `α_i=(1−e^{−ΛΔ_i})·s(z_i)`，把内容门乘在外面，而不是替换时间项。
3. **"精确指数积分器"只能卖稳定性，卖不了精度——已有负结果。** FARS/Analemma 2026 把 KDA 的 Euler 系数换成精确指数积分器后"27 次运行 0 NaN、数值更稳"，但 4× 外推 Palindrome −0.81pp、Stack −2.98pp，作者结论是"离散化误差不是长度外推的主要瓶颈"（非同行评审，需打折）。→ 该改进只进"稳定性/可解释性"栏，评测必须换成**跨采样率/掉帧**，不能讲"更精确所以更准"。
4. **双状态 h 是全族唯一没有被写过的格子，也是数学上最可疑的一格。** M 按真实 Δt 演化而 h 按观测序号 i 演化 ⇒ 30 Hz 与 60 Hz 下 M 轨迹近似相同、h 完全不同，系统层面**没有连续时间极限**（LIT-REVIEW §4.4）；08 又指出"矩阵记忆 + 向量状态"容量不叠加、表达力不升级（TC⁰）。→ h 必须也按 Δt 演化并带衰减，否则只能作为输出层，不被当作结构贡献。
5. **容量有硬上界，且取决于"怎么读"。** 双线性读 `uᵀMv`：`d² ≳ n log n`，且任何 W 都突破不了 `C·n log n`（C<2/π，Barnfield 2026 Thm 1(ii)〔定理〕）；线性读 `r=Mq` 只有 `n=O(d)`（Schlag §4.1 + 信噪比 `√(d/n)` 推导〔论证〕）。→ d 与读法要按"同时要记住几个对象"反推，并做读法 A/B；不要以为 d² 自动扩容。
6. **衰减与"长期记忆"直接冲突，长掉帧会把记忆清零。** `e^{−ΛΔ}` 对 Δ 跨越数量级会下溢（08 §4.4、06 §7.3）；TAWM 甚至要对 Δ 做 `log₁₀` 归一化才稳定。→ 必须有 Δ 的上下界/clip、Λ 分组（快/中/慢，参考 RetNet 的 `γ=1−2^{−5−i}` 多尺度），并让至少一路 λ≈0 承担遮挡后记忆。
7. **"只在变化时写"必须落成硬机制，软门控不算。** Mamba 的 `Ā∈(0,1)`、Gated DeltaNet 的 `α,β∈(0,1)`、Slot Attention 加 ε、ACT 的 halting 作弊脚注都说明软门控既不保证零、也会被优化器钻空子；真正逐位不变的是 Skip RNN `u∈{0,1}`、RIMs `h_{t+1,k}=h_{t,k}`、Trimpe 无事件时 posterior=prior、TGN 无事件不写。→ 对象级"不更新"要做成二值掩码，并配稳定性条件（Trimpe Thm 1：`‖e‖∞ ≤ m/(1−ρ)·‖L‖·‖δ‖`）。
8. **一步残差修正有明确的失效条件，且 B 很容易拿不到梯度。** tPC 作者自述"a single step is nowhere near enough"，实测单步跟踪劣于 5 步/收敛版；PredNet 外推崩溃、逐层等权损失"considerably degrade"。更要紧的是：若损失只作用在先验 `D(h⁻)` 上，B **完全没有梯度**（07 §5）。→ 损失必须跨 ≥2 步，或直接用教师状态标签监督滤波后状态。
9. **"支持不规则 Δ"不是差异化点。** CfC 原文即"t is set based on its time-stamp or order"，CRU/GRU-D/TiM4Rec/LGA 也都做了；而且 CfC 本身**不满足半群**（05 §2 代数：复合系数 `σ2+(1−σ2)σ1 ≠ σ(δ1+δ2)`）。→ 不要在"我们支持真实 Δ"上主张；半群只能在 **Φ 层**声明（复合映射 T=C∘Φ 一般不是半群），差异化必须落到"遮挡后记忆 / 多对象低串扰 / 对象级硬路由"。
10. **视频域仍是真空，但"缺席 ≠ 新颖"，而且对手已经存在。** 11 篇视频记忆论文全文扫描 `Δt / delta t / elapsed / wall-clock` 计数为 **0**（05-video §0.2），可"不变内容不重写"已由**相似度**实现（MovieChat 相邻帧余弦相似度贪心合并、MA-LMM 的 MBC），Δ 还已落在三处（Sijs 的 `A_τ`、TGN 的 time projection、Mamba 的学习 Δ）。→ 固定帧率下 Δ 与相似度高度相关，收益可能为零；必须自建变帧率/丢帧协议并与相似度触发正面对照。

---

## 已知失败模式 / 红队意见

### (a) 快权重 / 线性注意力记忆

- 「**Decay hurts memory retention**；Gating facilitates filtering；Delta rule helps memorization」——Gated DeltaNet §3.2 小节标题。〔实证〕
  → 对策：衰减只作为稳定性/干扰管理项，主容量押在 delta 写；消融里"有/无衰减"必须单列，且预期**收益接近于零**，写进报告时不要预设它有用。
- 「With keys embedded in a d_dot space, there cannot be more than d_dot orthogonal vectors. That is, storing more than d_dot associations will result in a retrieval error.」——Schlag 2021 §4.1，称 overcapacity regime。
  → 对策：把"关联数 n / 维度 d"作为一等实验变量，报相变点，不要只报均值精度（见 §3 容量协议）。
- 「cannot express computation outside TC⁰」「the 'state' in an SSM is an illusion」——Merrill 2024；Sarrof 2024 Thm 3.2 另指 PARITY 因"转移与输入无关"而困难。
  → 对策：**不主张** state tracking / 置换复合能力；若真要，唯一已知最小改动是让**转移矩阵输入相关**（Liquid-S4 式），而这会破坏我们想要的精确半群——两者不可兼得，必须在论文里挑明。
- 「纯加性 B 在 h⁻ 与 r 相关时会**双计**、高增益下过冲」；DeltaNet 用 Householder `(I−βkkᵀ)` 正是为此。
  → 对策：写规则用 `M←M(I−βkkᵀ)+βvkᵀ`（或 Gated 版 `α_t(I−βkkᵀ)`），并对 β 加上界/收缩参数化。
- 「矩阵态 = d² 维向量，rank-1 写是**限制**不是优势」；容量买的是常数，不是表达力。
  → 对策：报告"有效状态维度"与"可用关联数"，不要用参数量当卖点。
- 「稀疏更新天然更高效 —— **错**」：DeltaNet 家族反过来做 chunkwise **稠密化**正是为规避逐 token 顺序写；MoE 的 >1000× 收益是"容量/算力"而非"少写记忆"。
  → 对策：稀疏只从**算法/表示**角度论证（少写=少漂移），不做墙钟效率主张；真要主张就自己测目标硬件的墙钟。
- 「Δt 很小时 e^{−ΛΔ}≈I，梯度病态；Δt 很大时 M⁻≈0，状态被清空（等价于重新初始化）」——06 §7.3。
  → 对策：ΛΔ 设上下界（如 clamp 到 [1e−4, 5]），并监控 `‖M‖` 与 NaN；长 Δ 后单独测"旧关联是否还可读"。
- 「Family 框架已成文出版，新衰减/写组合必须先回答'对应哪个局部损失'」——Irie & Gershman 2025 App. A 把 RetNet/GLA/Mamba2/xLSTM 的衰减都推成 L2 正则项。
  → 对策：为 `M` 的每步更新写出在线目标 `L_t(M)=‖Mk_t−v_t‖² + λ(Δ_t)·‖M‖²`，并说明我们的更新是它的（近似）解；否则在 2026 年的审稿标准下就是"未定义的前向规则"。
- 「双状态无任何已发表稳定性条件，工程上靠 LayerNorm 与残差」——06 §7.3 最后一条。
  → 对策：显式给谱/范数约束（`‖M‖ ≤ max(‖M_0‖, c/λ)`，09 命题 3）与 h 的收缩条件，并做稳定性消融。

### (b) 连续时间模型（含 CfC / LTC）

- 「CfC 式(10) 的 `σ(−f·t)` 把 Δ 与状态相关的 f 相乘后再过 sigmoid，复合两步得 `σ2+(1−σ2)σ1`，与单步 `σ(δ1+δ2)` 不同 ⇒ **不满足半群**」——05 §2 代数，N6。〔论证，可自验〕
  → 对策：我们的"精确半群"只能声明在 **Φ 层**；对系统层（含写入门）只能声明 D2 细化收敛或误差阶，并给出数值检验曲线 `‖Φ_{Δ1}∘Φ_{Δ2} − Φ_{Δ1+Δ2}‖` 随 Δ 的变化。
- 「CfC 把 exp 换成 sigmoid，是因为 exp 形式 'causes vanishing gradient factors'」——05 N6 引原文。
  → 对策：若我们用 exp，必须实测梯度范数随 Δ 的分布；这是 CfC 相对我们的**已知代价互换**，要在对比里如实报告（我们换回了 exp，就继承了它的问题）。
- 「LTC 的方程刚性，RK 类求解器 'not suitable for LTCs'，必须 fused Euler」；Neural ODE 类"状态转移是数值解，误差随容差变化"——05 N3/N8。
  → 对策：若出现 ODE 版，必须声明求解器与容差，并把求解器误差算进总误差预算；能闭式就闭式。
- 「ODE 类即使无观测也要连续积分，计算量不随稀疏性下降」（Latent ODE §3.4 原文；ODE-RNN 比 GRU 慢约 60%）。
  → 对策：流式场景下不承诺"稀疏省算力"，也不要把 ODE 求解器放进 50 Hz 级的热路径。
- 「含观测注入的复合映射 T=C_z∘Φ_Δ 一般**不满足**半群」——07 §6.2（三条论证：下标依赖注入了什么观测；重复注入会重复计权；观测到达时刻改变结果）。
  → 对策：只声明三种弱化之一：D1 流一致（最易证、最无信息量）、D2 细化收敛（要误差阶）、D3 观测更新可交换（高斯下 Kalman 满足、EKF 一般不满足）。不要混着说。
- 「若 B 与 Δ 无关且 r 不随 Δ→0 消失，则单步修正 `‖B‖=O(1)`、总变化 `Σ‖B‖=N·O(1)→∞`，**细化极限不存在**」——07 §6.3 命题 5〔论证〕。
  → 对策：定理必须把 `r=O(Δ)`（D 的 Lipschitz 常数 + 流误差）或 `B∝Δ` 写进假设；同时说明噪声随 Δ 的缩放（Kalman-Bucy 的经典前提：噪声按 Δ 缩放才收敛，否则 ΣKr 是随机游走）。
- 「不可观测子空间 `ker O(C,A)` 在 A−LC 下不变 ⇒ 任何输出注入式修正都无法修正该方向」——03 N4〔控制界标准结论〕。
  → 对策：显式用教师状态标签监督状态（我们**有**标签，这正是相对 RSSM/Dreamer 纯重构的差异点），并做"可观测方向/不可观测方向"的分组误差报告。
- 「学出的增益不满足 K=PCᵀS⁻¹，协方差代数直接崩，必须退回 Joseph 形式」——Recursive KalmanNet 2025〔论证〕。
  → 对策：不主张不确定度量化；若要，显式带对角方差并全程用 Joseph 形式。
- 「单步修正 = tPC 的 Eq 14 特例，而作者自述 'a single step is nowhere near enough'」；PC 侧另有收敛域小（Frieder 2022 Neimark–Sacker 反例）、必须 Adam 兜底（Alonso 2024）、层间能量差几个数量级且只在小型 CIFAR 规模可比（Pinchetti 2024）。
  → 对策：不把"PC 式局部规则"当卖点；只用"教师状态显式监督"这一条差异化。

### (c) 容量测量

- 「矩阵记忆 = 长期记忆 → **不成立（至少被严重高估）**」：上界只与 d 和读法有关；而且候选只有衰减、**没有巩固/重放/压缩**，衰减本身就在主动缩短记忆（08 §6.1）。
  → 对策：容量与"长期"分开测：容量用 n–d 相变，长期用 age–recall 曲线；要主张"长期"就必须加多时间尺度或巩固机制。
- 「衰减率 ↔ 可存关联数的严格界：**缺口（未找到）**」「低频写入 ⇒ 误差累积的定理：**缺口（未找到）**」「矩阵记忆+向量双状态严格强于单状态：**缺口（未找到，且 TC⁰ 结论反向暗示）**」——08 §5 三分表。
  → 对策：这三条一律不写"有理论保证"；在报告里明确标为 open，最多给可证伪预测。
- 「容量标度取决于读算子」：线性读 `O(d)`（本报告推导）、双线性读 `O(d²/log d²)`（定理）、经典 Hopfield `α_c=0.138N`（二手核实）、现代 Hopfield 指数容量（`c^{(d−1)/4}`）。
  → 对策：**文档在这里给出四个不同标度，互相不可混用**；引用时必须写清是哪一种读法、哪一种记忆。我们若用 `Mq` 就按 `O(d)` 报，不要挪用 `d²/log` 的数字（这是 LIT-REVIEW §4.5 与 litreview2/08 之间最容易误引的一处）。
- 「加性写在 overcapacity 区 sub-optimal，delta 规则更好」〔论证 + 实证 C2/C3〕。
  → 对策：容量实验必须同时跑"加性 vs delta"两条写规则，否则容量数字不能归因。

### (d) 我们具体的实验设计

- **冻结/缓存特征**：「r 的几何意义在非线性 D 下并不成立（残差大的方向不一定是要修的方向）」——03 N5；这正是 DKF/DVBF/KVAE 那一支选择"不显式算残差、直接学后验"的原因。
  → 对策：残差定义在**特征空间且先归一化**（单位化 z 与 D(h)）；另跑一个"学习后验"对照组（不做残差）以证明残差式确有增益。
- **缓存特征的退化风险**：「自监督预测结构会把 r 压小（trivial solution：D∘Φ≈identity，r≈0，修正通道空转）」——06 §7.2；DVBF 原文承认 `z_{t+1}=w_t` 可达。
  → 对策：硬约束 `B(h,0)=0`，加"残差不可吞掉动力学"的正则（例如对 `‖r‖` 设下限带，或冻结 Φ 的前若干步并单独监督 `D∘Φ` 的多步预测），并把 `‖r‖` 的时间序列当监控指标上报。
- **合成任务**：「一步最优 ≠ 多步最优」（PlaNet latent overshooting）；PredNet 的**时间打乱对照**显示模型只是输出"模糊的上一帧"。
  → 对策：所有合成任务一律加 (i) temporal-shuffle 对照、(ii) 多步 rollout（≥5 步）指标、(iii) 长程外推单列；只用单步指标会被这两篇直接打回。
- **鲁棒性扰动**：「全部 11 篇视频论文按帧号，帧率变化即静默失效；无一篇显式处理丢帧/重复帧」——05-video §3.5。
  → 对策：扰动协议自建：Δ ~ log-uniform（覆盖 1/60 s–1 s）、随机丢帧率 {0,10,30,50}%、重复帧、抖动（高斯 PTS 噪声）；测试集 Δ 分布必须与训练不同（否则测的是插值）。
- **参数匹配**：CfC 的 Δ 也是真实时间戳（05 §1.8 原文），所以"我们支持不规则 Δ"不构成解释力；而"矩阵状态 d² 维"会天然多出参数。
  → 对策：对齐**参数量 + 状态维 + 训练步数 + 优化器**，并额外报"每步 FLOPs / 状态字节"；同时跑一个"把 CfC 状态维放大到与我们参数量相同"的对照，防止审稿人把增益归给容量。
- **训练信号**：「纯预测误差作唯一学习信号会饱和/卡局部极小（Titans 原文：the gradient can become extremely small after several surprising steps ... local minima）」。
  → 对策：辅助损失必须有 retrieval（k→v top-1）、multi-target isolation（off-diagonal 能量惩罚）、time-consistency（同区间两种采样密度读出应一致）、write-budget（Σα 或 skip 门预算，参考 Skip RNN 的 budget 约束）、long-horizon（多步，参考 latent overshooting）五项，且逐项消融。

---

## 容量理论要点

**要测的量**：`d`（key 维）、`d_v`（value 维）、`n`（写入关联数）、`age`（秒）、`λ`（衰减率）、`N_obj`（对象数）。

1. **检索相变曲线（第一优先）**。固定 d ∈ {32,64,128,256}，扫 n，报 top-1 retrieval accuracy vs `n`（同时给 `d²/(n ln n)` 作横轴）。报 50% 临界点 `n*` 与 `n*/d`。理论参照：双线性读 `n*≈d²/(8 ln n)`（C1 Prop 1，常数 8 为粗算）；线性读 `n*≈d`。**若实测 n\* 远大于 O(d) 且我们用 Mq 读，说明读出并非纯线性，必须解释而不是庆祝。**
2. **串扰/信噪比**。对每个 key 单独测读出，报混淆矩阵与 off-diagonal 能量比；同时报解析预测 `SNR≈√(d/n)` 与实测拟合，两者同图。
3. **覆盖 vs 叠加（写规则鉴别）**。同一 key 写不同 value 多次，测 "latest-value recall"。加性写预期读到加权混合，delta 写预期替换——这是**区分我们写规则是否真的实现**的判据。
4. **有效视界（长期记忆的正规测法）**。测 recall vs age（秒），拟合 `exp(−λ_eff·age)`，报半衰期 `t½=ln2/λ_eff`。**关键检验：把采样率改变 2×/4×，t½ 必须不变**——这是"真实 Δt 衰减"与"按帧数衰减"唯一可观测的差别，也是我们相对 GLA/RetNet 的实测卖点。
5. **长 Δ 恢复测试**。人工插入 Δ∈{0.1, 1, 10} s 的掉帧，测旧关联可读性，报"清零阈值"（即 recall 掉到随机的 Δ）。
6. **写入次数扫描**（对应"少写更稳"这条直觉）。固定总 Δ，把写入次数改为 k 次（每次 α/k）：若性能随 k 下降，说明"少写"不免费；若不变，说明衰减主导——两种结果都要报。
7. **Δ–容量联合界**（文档里唯一可写但未被证的组合）。可分辨对象数 ≈ `1/(ε·λ·平均间隔)`（09 §5 贡献 2〔论证〕），其中 ε 是 key 间内积上界。实验上扫 (λ, 平均间隔) 二维网格，画等精度线，检验该标度是否成立。
8. **定理级 falsification（作为边界声明，不作为结果）**。MQAR（Zoology Thm 4.4 对固定滤波的维度要求）、S5 置换复合（Merrill 2024）、PARITY（Sarrof Thm 3.2）：**预期失败**。预先写下"我们会失败"，比事后解释体面。

**协议纪律**：keys 单位化且随机；报 5 seed 的均值与方差；d 至少 4 个量级点；固定训练预算；容量与"长期"分开报；**任何"容量大"的结论必须同时给出读法、写规则、d 三个条件**（因为文档给出三种互不兼容的标度）。

---

## 与 CfC 对比时必须控制的变量

1. **Δ 的语义两边一致**：CfC 的 t 也按 timestamp 赋值（05 §1.8 原文）。两模型喂**同一份 PTS**；不得把"支持不规则 Δ"当我们的增益来源。
2. **半群主张的层次**：CfC 不满足半群（05 §2），我们若声称半群，只能在 Φ 上，并附数值误差曲线；否则两边都没有系统级半群，比较无意义。
3. **参数 / 状态维 / FLOPs / 状态字节**：四项都要对齐或至少同时报告；额外跑"CfC 放大到同参数量"。
4. **特征与归一化**：同一份缓存特征、同一归一化、同一 clip 长度；CfC 的输入尺度不能与我们不同。
5. **训练信号**：同样的 task loss；若我们加 5 项辅助损失，必须给 CfC 加等效正则（或报告"CfC + 同预算"两档），否则增益归因不清。
6. **Δ 分布**：训练用 log-uniform（覆盖训练分布外），测试跨采样率；只在训练 Δ 上比较等于没比。
7. **五项指标分别定义**，不许用单一综合分：
   - 遮挡后记忆：固定遮挡时长/对象数，报 recall@delay 曲线；
   - 多对象 ID：关联准确率 + **混淆矩阵**（低串扰要看 off-diagonal，不看均值）；
   - 事件检测：事件级 F1 + 检测延迟（帧级 F1 会掩盖拖尾）；
   - 不规则 Δ：性能–Δ 曲线 + 掉帧率扫描；
   - 因果在线前缀：逐前缀累计精度（prefix-accuracy），不是整段离线精度。
8. **消融必须单列四个部件**：衰减、delta 写、对象级硬路由、双状态 h。文献说收益主要来自 delta 写与门控（GDN），若我们把四项打包报增益，会被一击必杀。
9. **数据切分按序列/段，不按帧随机切**（同段相邻帧高度相关，随机切等于漏测试集）。
10. **不把离线合成集指标当验收**：必须有一档 held-out 真序列（或真机回放）作为最终判据。
11. **报告负结果**：至少报告"衰减消融"和"精确离散化"两项预期接近零收益的消融——这两项若被我们藏起来，等于把最容易被攻击的点留给审稿人。

---

## 建议的 DSM 设计修正（按预期影响排序）

1. **把 α 改成乘积形式并锚定尺度**：`α_i = (1−e^{−ΛΔ_i})·s(z_i)`，`s∈[0,1]` 为内容门；Λ 固定物理单位（或只学一个共享标量 λ 的 log 尺度，其余按组固定）。理由：ρ 与 α 必须同源于一个 Δ（结论 2），且 `ΛΔ` 的乘积不可辨识（07 R5）——不锚定的话，"跨采样率不变"的实验会被学到的 Λ 自动抵消，测不出东西。
2. **写规则换 delta + Householder，并把 β 指数化作为可切换项**：`M ← e^{−ΛΔ}⊙M`，再 `M ← M(I−β kkᵀ)+β v kᵀ`；β 由 `s(z)` 给出。同时实现 09 §5 贡献 1 的精确版 `β→1−e^{−β}`，**作为稳定性开关而非精度改进**（Analemma 负结果）。理由：加性写是已知更差的一侧（结论 1），而精确指数化已有负结果（结论 3）。
3. **双状态 h 也按 Δt 演化**：`h⁻ = e^{−ΓΔ}⊙h`，再 `h = h⁻ + H(h⁻, r, z)`，与 M 共用同一个 Δ、同一套 ZOH。然后做"30 Hz vs 60 Hz 下 h 轨迹一致性"实验。若做不到一致，就把 h 降级为输出层并不再作为结构贡献。理由：这是全族唯一未证且最可能不自洽的一格（结论 4），两份文档在此**互相矛盾**（00-INDEX 说是"真空白"，LIT-REVIEW §4.4 说"没人做很可能因为它不自洽"）——文档没有裁决，我们只能自己测。
4. **读法 A/B：`Mq` vs 双线性 `u(z)ᵀMv(z)`**。前者容量 `O(d)`、后者 `d²/log`，代价是参数与算力。理由：容量结论 5 直接取决于此；不测就不知道 116 维特征下我们到底能记几个对象。
5. **对象级硬路由 + 严格不变性**：`‖r_slot‖ > δ` 才写该 slot，用二值掩码（Straight-through），未命中 slot 做**逐位不动**检查。理由：软门控不保证零（结论 7）；且"用观测残差决定哪个对象要改"这一格在文献里确实空着（04 §6.5），是我们少数能主张的组合。附带必做：δ 扫描 + Trimpe 形式的误差界 + "切换系统可能不稳"的显式声明。
6. **Λ 多时间尺度分组**：把 Λ 分成 3–4 组，λ 跨 10×–1000×（参考 RetNet 的 `γ=1−2^{−5−i}`），最慢一组 λ→0 专供"遮挡后长期记忆"。理由：单 λ 无法同时满足"抗长掉帧"（结论 6）与"抑制串扰"；这也是 08 §6.1 指出"单靠一个 d×d M 不够做长期"的直接补丁。
7. **辅助损失五项具体化**（每项都要能单独关掉）：retrieval（`k→v` top-1）、multi-target isolation（读出矩阵的 off-diagonal 能量）、time-consistency（同物理区间两种采样密度读出一致 = D2 的可微版）、write-budget（`Σα` 的 L1 或 skip 门预算）、long-horizon（≥5 步 rollout）。理由：纯预测误差会饱和（结论 8 前半），且"少写"必须有预算约束才不是空话。
8. **给 B/写通道建立梯度通路**：所有损失至少跨 2 步；另加教师状态标签对"滤波后状态"的直接监督。理由：只监督先验时 B 拿不到梯度（结论 8 后半）；而对齐的正是我们的独有资产。
9. **稳定性工程前置**：`B(h,0)=0`、`∂B/∂r` 有界、`‖M‖ ≤ max(‖M_0‖, c/λ)` 的在线检查、ΛΔ 的 clamp、NaN 监控。理由：09 命题 3 给了范数界，但前提是衰减正确配套；这些是"不满足就整轮实验作废"的项。
10. **写/衰减解耦**：写不写由 surprise（`‖r‖` 或 `‖v−Mk‖`）决定，衰减多少由 Δ 决定，两者不共用同一个门。理由：08 §6.3 洞 2 指出耦合会让两者互相抵消或共振；Titans 也是把 `α_t`（衰减）与 `θ_t`（写强）分成两条通道的。
11. **先做"衰减到底有没有用"的消融**（把 1 与 2 的正确性固定后立刻做）。理由：如果衰减在本任务上收益为零，早死早超生——这决定我们后续是继续投理论还是改设计。

---

## 值得复现的对照方法

| 方法 | 为什么必须做对照 |
|---|---|
| **CfC**（Hasani 2022, Nature MI） | 待超越的基线；同一 PTS、同参数量。必须跑"CfC + 同参数预算"与"CfC + 等效正则"两档。 |
| **LTC**（Hasani 2021, AAAI） | CfC 的源；用来判定"闭式"本身是否贡献增益，还是 ODE 结构在起作用。 |
| **GRU-D**（Che 2018, Sci Rep） | "真实 Δ → 指数衰减"的最简对照。若我们只比 CfC 好而不比 GRU-D 好，说明增益来自 Δ 而非我们的记忆。 |
| **CRU**（Schirmer 2022, ICML） | 唯一同时有"真实秒级 Δ + 精确半群转移 + 新息加性修正"的向量态实现，是审稿人最先拿来打我们的东西。 |
| **DeltaNet / Gated DeltaNet**（Yang 2024 NeurIPS / 2025 ICLR） | 写规则的直接对照（加性 vs delta）与容量探针（S-NIAH 式）；也是"我们的写规则是不是退步"的判据。 |
| **GLA**（Yang 2024, ICML） | `Diag(α_t)` 学习门衰减矩阵记忆——"把 α 换成 exp(−ΛΔ)"到底值多少，只能靠它比。 |
| **RetNet**（Sun 2023） | 固定 γ 按位置衰减；用于分离"Δ 语义"与"位置衰减"（跨采样率实验里它应当掉点）。 |
| **Mamba / Mamba-2**（Gu & Dao 2024 / Dao & Gu 2024） | 向量态 ZOH 的廉价基线；若我们打不过它，d×d 矩阵记忆的存在理由要被质疑。 |
| **Titans / TTT**（Behrouz 2025 NeurIPS / Sun 2025 ICML） | "只写变化部分"的最强实现（surprise/梯度写）；用来检验我们的硬路由是否真比软 surprise 强。 |
| **Liquid Gated Attention**（Jiang 2026, arXiv） | 唯一逐字实现"矩阵记忆 + 真实 Δt 衰减 + 写强随 Δ 缩放"的工作，且带凸组合约束 `(1−g)`。不引、不比会被当作没做文献工作。 |
| **MovieChat / MA-LMM 的相似度合并** | "不变内容不重写"的现行最佳实现（内容触发）。我们的 Δ 触发必须与它正面对照，否则"用时间换相似度"没有说服力。 |
| **SAVi**（Kipf 2022, ICLR） | 对象级 predictor-corrector 的命名与语义先例；也是"全量重绑定 vs 残差硬路由"的对照（它每帧重写全部 slot）。 |
| **Trimpe & D'Andrea 事件触发观测器**（2012） | 硬触发 + 稳定性上界的算法级参照；我们的 δ 扫描必须复现"δ→0 退化为全更新、δ 大误差 ∝‖δ‖"的形状。 |
| **现代 Hopfield**（Ramsauer 2021, ICLR） | 读算子的理论参照（`X·softmax(βXᵀξ)` 即 attention）；用于说明我们的读出属于哪一族、容量标度取哪一支。 |

---

## 引用清单

> 前 25 条是决策直接依赖的核心；26–35 是正文点到、但只作单点证据的补充条目。

1. Hasani, Lechner, Amini, Liebenwein, Ray, Tschaikowski, Teschl, Rus. *Closed-form Continuous-time Neural Networks*. Nature Machine Intelligence 4:992–1003, 2022. https://www.nature.com/articles/s42256-022-00556-7 ｜ arXiv:2106.13898
2. Hasani, Lechner, Amini, Rus, Grosu. *Liquid Time-constant Networks*. AAAI 2021. https://arxiv.org/abs/2006.04439
3. Che, Purushotham, Cho, Sontag, Liu. *Recurrent Neural Networks for Multivariate Time Series with Missing Values (GRU-D)*. Scientific Reports 8:6085, 2018. https://www.nature.com/articles/s41598-018-24271-9
4. Schirmer, Eltayeb, Lessmann, Rudolph. *Modeling Irregular Time Series with Continuous Recurrent Units (CRU)*. ICML 2022, PMLR 162. https://proceedings.mlr.press/v162/schirmer22a/schirmer22a.pdf
5. Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers*. ICML 2021, PMLR 139. https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf
6. Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet)*. NeurIPS 2024. https://arxiv.org/abs/2406.06484
7. Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with the Delta Rule*. ICLR 2025. https://arxiv.org/abs/2412.06464
8. Yang, Wang, Shen, Panda, Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training (GLA)*. ICML 2024. https://arxiv.org/abs/2312.06635
9. Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei. *Retentive Network: A Successor to Transformer for Large Language Models*. 2023. https://arxiv.org/abs/2307.08621
10. Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024. https://arxiv.org/abs/2312.00752
11. Dao, Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024. https://arxiv.org/abs/2405.21060
12. Ba, Hinton, Mnih, Leibo, Ionescu. *Using Fast Weights to Attend to the Recent Past*. NeurIPS 2016. https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past
13. Schmidhuber. *Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks*. 1991 技术报告 FKI-147-91 ／ Neural Computation 4(1):131–139, 1992.
14. Behrouz, Zhong, Mirrokni. *Titans: Learning to Memorize at Test Time*. NeurIPS 2025. https://arxiv.org/abs/2501.00663
15. Sun, Li, Dalal, et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT)*. ICML 2025, PMLR 267. https://arxiv.org/abs/2407.04620
16. Barnfield, Kim, Nichani, Lee, Lu. *Sharp Capacity Thresholds in Linear Associative Memory*. 2026. https://arxiv.org/abs/2605.05189
17. Merrill, Petty, Sabharwal. *The Illusion of State in State-Space Models*. ICML 2024, PMLR 235. https://arxiv.org/abs/2404.08819
18. Sarrof, Veitsman, Hahn. *The Expressive Capacity of State Space Models: A Formal Language Perspective*. NeurIPS 2024. https://arxiv.org/abs/2405.17394
19. Arora, Eyuboglu, Timalsina, Johnson, Poli, Zou, Rudra, Ré. *Zoology: Measuring and Improving Recall in Efficient Language Models*. ICLR 2024. https://arxiv.org/abs/2312.04927
20. Lei, et al. *Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics (EFLA)*. arXiv:2512.12602, 2025. https://arxiv.org/html/2512.12602v5 （发表状态未核，引用前需确认）
21. Jiang, Xu, Yang. *Liquid Gated Attention*. arXiv:2608.30695, 2026. （附录仅标题级核实）
22. Schöne, et al. *Scalable Event-by-event Processing of Neuromorphic Sensory Signals With Deep State-Space Models*. ICONS 2025 ／ arXiv:2404.18508. https://arxiv.org/html/2404.18508v3
23. Fan, Zhu, Hu, Feng, He, Liu, Liu. *TiM4Rec: An Efficient Sequential Recommendation Model Based on Time-Aware Structured State Space Duality Model*. 2024. https://arxiv.org/abs/2409.16182
24. Trimpe, D'Andrea. *Event-Based State Estimation with Switching Static-Gain Observers*. IFAC NecSys 2012, pp. 91–96. http://trimpe.is.tuebingen.mpg.de/publications/trimpe-ECNS12.pdf
25. Sijs, Lazar. *On Event Based State Estimation*. HSCC 2009, LNCS 5469:336–350. https://pure.tue.nl/ws/files/3301085/Metis230979.pdf
26. Campos, Jou, Giró-i-Nieto, Torres, Chang. *Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks*. ICLR 2018. https://arxiv.org/abs/1708.06834
27. Kipf, Elsayed, Mahendran, Stone, Sabour, Heigold, Jonschkowski, Dosovitskiy, Greff. *Conditional Object-Centric Learning from Video (SAVi)*. ICLR 2022. https://arxiv.org/abs/2111.12594
28. Song, Chai, Wang, et al. *MovieChat: From Dense Token to Sparse Memory for Long Video Understanding*. CVPR 2024. https://arxiv.org/abs/2307.16449
29. He, Li, Jang, Jia, Cao, Shah, Shrivastava, Lim. *MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding*. CVPR 2024. https://arxiv.org/abs/2404.05726
30. Millidge, Tang, Osanlouy, Harper, Bogacz. *Predictive Coding Networks for Temporal Prediction (tPC)*. PLOS Computational Biology 20(4):e1011183, 2024. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183
31. Lotter, Kreiman, Cox. *Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning (PredNet)*. ICLR 2017. https://arxiv.org/abs/1605.08104
32. Becker, Pandya, Gebhardt, Zhao, Taylor, Neumann. *Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces*. ICML 2019, PMLR 97. https://arxiv.org/abs/1905.07357
33. Revach, Shlezinger, Ni, López Escoriza, van Sloun, Eldar. *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics*. IEEE Trans. Signal Processing 70:1532–, 2022. https://arxiv.org/abs/2107.10043 （注意：不是 ICML 2022）
34. Ramsauer, et al. *Hopfield Networks is All You Need*. ICLR 2021. https://arxiv.org/abs/2008.02217
35. Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein. *Temporal Graph Networks for Deep Learning on Dynamic Graphs*. 2020. https://arxiv.org/abs/2006.10637

---

*本简报只做压缩与对齐，不新增未经文档支持的结论。凡文档间冲突处（F 轴是否空白、双状态是"真空白"还是"不自洽"、容量标度取哪一支、精确离散化是否有收益）已在正文显式标出，未擅自裁决。*
