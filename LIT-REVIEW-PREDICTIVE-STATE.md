# 预测-修正型状态更新公式：文献审判与新颖性核查

> 核查对象（用户提出）：
>
>     h⁻ = Φ_Δ(h)                 先验/预测
>     r  = z − D(h⁻)              观测空间残差（新息）
>     h  = h⁻ + B(h⁻, r)          加性残差修正（学习式增益）
>
> 建档：2026-11-03。方法：**纯网络检索**（tavily_search / tavily_extract + arXiv API 核实元数据），
> 未训练、未跑实验、未读项目旧文档。6 个专题由 6 个独立检索代理并行完成，
> 每条引用的标题/作者/年份/链接均实际抓取正文核对；无法核实的条目在各节内标 [未核实]/[仅摘要级]。

---

# 第一部分 · 判决摘要（先看这一节）

## 0. 一句话判决

**这三行公式在结构上不新颖。** 它就是 **连续-离散贝叶斯滤波 / 新息（innovation）型观测器**的记号变体，
1960 年成文（Kalman 1960 式 20–22：ỹ = y − Mx*，x*⁺ = Φ*x* + Δ*y，Φ* = Φ − Δ*M），
2019–2025 已被神经网络化至少五次（RKN、CRU、ODE-RNN、NCDSSM、TAWM）。
**三行中的每一行都有逐字先例；七轴（A 先验 / B 观测残差 / C 加性修正 / D B 依赖 (h⁻,r) / E 真实 Δ / F 稀疏选择 / G 训练目标）没有一轴是空的。**

能支撑新颖性的只剩**定理形状**的三件事（见 §6），不是结构。

## 1. 三行分别被谁占了（逐字级）

| 目标公式 | 逐字先例 | 轴 |
|---|---|---|
| h⁻ = Φ_Δ(h)，Δ 为真实秒 | **CRU**（Schirmer, Eltayeb, Lessmann, Rudolph, ICML 2022）：μ⁻ = exp(A(t−τ))μ⁺ ／ **TAWM**（Nhu, Son, Lin, ICML 2025）：ẑ(t+Δt) = z + d(z,a,Δt)·τ(Δt) | E |
| r = z − D(h⁻) | **RKN**（Becker et al., ICML 2019）：w − Hz⁻ ／ **Kalman 1960** 式(20)：ỹ = y − Mx* ／ **tPC**（Millidge, Tang, Osanlouy, Harper, Bogacz, PLOS Comput Biol 2024）式26：x̂ = x̂⁻ + K(y − Cx̂⁻) | B |
| h = h⁻ + B(h⁻, r) | 同上三篇 + **Trimpe & D'Andrea**（IFAC NecSys 2012）式(11)：x̌⁺ = x̌⁻ + Σ Lᵢ(yᵢ − Cᵢx̌⁻) | C |
| 三者同时、且 Δ 进 Φ | **CRU（ICML 2022）一篇全中 A–E** | A–E |
| 三者同时 + 硬稀疏（改哪些状态） | **Trimpe & D'Andrea 2012**：‖yᵢ − Cᵢx̌‖ ≥ δᵢ 才更新，未触发通道原样拷贝，并给误差上界 ‖e‖∞ ≤ m/(1−ρ)·‖L‖·‖δ‖ | A–F |

## 2. 用户三个问题（Q1/Q2/Q3）的直接答案

| 家族 | Q1 每次观测都用预测误差修正隐状态？ | Q2 按真实时间 Δ 演化？ | Q3 为规划/生成未来，还是通用在线感知？ |
|---|---|---|---|
| RSSM / PlaNet / Dreamer v1–v3 | **部分**：用当前观测**重推断**后验（编码器联合输入），不是残差加法 | **否**（观测索引；action repeat 是唯一时间旋钮） | 规划/想象；后验只是想象的起点 |
| TD-MPC / TD-MPC2 | **否**：在线重编码重锚定，预测状态被丢弃；残差只进 loss 且跨一步 | 否 | 规划（MPPI/MPC）；decoder-free ⇒ D 不存在 |
| TAWM (2025) | 否 | **是**（秒级，log-uniform 多采样率训练） | 规划/控制 |
| Kalman / EKF / CD-EKF / Kalman-Bucy | **是**（新息 × 增益） | **是**（区间内 ODE 传播 + 观测时刻修正） | 通用在线状态估计 |
| Luenberger / 非线性观测器 | 是 | 是（连续时间） | 控制 |
| RKN (2019) | **是**（显式新息，潜空间） | 否 | 通用在线估计 |
| CRU (2022) | **是**（新息） | **是**（exp(A(t−τ))，真实秒） | 插值/估计 |
| KalmanNet (2022) | 是（RNN 学增益） | 否 | 状态估计 |
| NCDSSM (2023) | 是（贝叶斯后验更新） | 是 | 不规则时序估计 |
| 经典预测编码 (Rao–Ballard 1999 等) | 是（**迭代**推理步，收敛才算完） | 否 | 感知/神经解释 |
| tPC (2024) | **是**（单步即目标公式；Eq14 还给"每帧只修一次"版） | 否（离散步） | 时序感知 |
| PredNet (2017) | 是（误差直接进循环状态，**分层局部**） | 否 | 视频预测/表征 |
| ODE-RNN / Latent ODE (2019) | 是（观测间 ODE 传播 + 观测时刻 GRU 跳变） | **是** | 序列建模 |
| Neural CDE (2020) | 部分（插值路径驱动，无显式残差） | 是 | 序列建模 |
| LTC / CfC (2021/2022) | **否**（输入驱动，无观测预测、无解码器） | **是**（timespan 进入门控） | 序列建模/控制 |
| GRU-D (2018) | 否（衰减，非残差） | 是（γ = exp(−max(0,Wδ+b))） | 时序分类 |
| 事件触发估计 (Trimpe 2012 / Sijs 2009) | 是（阈值触发，**硬**更新） | 部分（Sijs 用 A_τ 把真实时间放进转移） | 估计/通信 |
| Skip RNN / ACT | **否**（输入驱动的门控，不是预测误差） | 否 | 计算效率 |
| Slot Attention / SAVi / STEVE | 部分（slot 分解；SAVi 自称 predictor-corrector，但 corrector 是重绑定） | 否 | 对象中心表征/视频 |
| DeltaNet / Gated DeltaNet (2021/2025) | **是**（delta rule 显式误差修正，B 可学习） | 否（α_t 是学出的门，不是 Δ） | 序列建模/记忆 |
| TTT / Titans (2025) | **是**（h 是模型，更新是预测误差的梯度步） | 否 | 序列建模/记忆 |

**读表结论**：
1. "用预测误差修正状态"在文献里**既**是规划工具**也**是通用在线感知结构——所以这个二分不构成新颖性轴。
2. 主流**世界模型**恰恰**不是**残差修正，而是"重推断"；**残差修正**是**滤波/预测编码/快权重**这一支的传统。
3. **真实 Δ 只出现在连续时间谱系**里，且 Δ 进入方式决定了数学性质（见 §4）。

## 3. 七轴总表（跨 6 个专题，取每轴最强先例）

| 轴 | 已占据的最强先例 | 结论 |
|---|---|---|
| A 显式先验 h⁻ | Kalman 1960；PlaNet 2019 | 不新 |
| B 观测空间残差 r = z − D(h⁻) | Kalman 1960 式(20)；RKN 2019；tPC 2024 式26 | 不新 |
| C 加性修正 h = h⁻ + B | 同上；Trimpe 2012 式(11) | 不新 |
| D B 依赖 (h⁻, r) | Kalman 增益（依赖 P⁻）；RKN 方差比；KalmanNet 学习增益 | 不新（"学习式且吃两者"=KalmanNet） |
| E 真实 Δ | CRU 2022（秒，exp(AΔ)）；TAWM 2025；ODE-RNN 2019；Sijs 2009 | 不新 |
| F 稀疏/选择/局部 | **Trimpe & D'Andrea 2012**（逐传感器硬触发 + 误差上界）；PredNet 分层；Slot 模型 | **不新**（见 §5 更正） |
| G 训练目标 | 重构/ELBO/一步预测/RL/自监督 全部有 | 不构成新颖性 |

## 4. 时间轴与数学性质（这一节决定"能不能证明什么"）

- **精确半群**（Φ_{Δ₁+Δ₂} = Φ_{Δ₂}∘Φ_{Δ₁}）：只有"某生成元的流"——exp(AΔ)（Kalman/CRU/Mamba/S4）、
  真 ODE 解（Neural ODE/CDE，数值近似）、显式 flow map（Neural Flows）。**1、2 节的逐式核对见 05 节。**
- **不满足半群**：把 Δ 拼成特征的 MLP、欧拉步（只 O(Δ²)）、门控/时间常数式（CfC 的 timespan 进非线性自变量）。
- **含观测注入的复合映射 T = C_z ∘ Φ_Δ 一般不满足半群**，所以"采样细化一致性"必须重新定义，
  且需要噪声/观测随 Δ 缩放的假设（05 节 N3、07 节 §6）。
- **命题（本报告 07 节 §6.3）**：若 B 与 Δ 无关且预测误差 r 不随 Δ→0 消失，
  则细化极限下发散（Σ‖B‖ = N·O(1) → ∞）。**"采样细化一致"的定理必须把"r → 0"或"B ∝ Δ"写进假设**，否则是空的。

## 5. 分歧与更正（负面结果优先，互相打脸的地方照实写）

1. **01 与 06 说"F 轴是唯一空格"——错，范围放错了。**
   在"**学习式**世界模型/序列模型"里 F 轴确实接近空；但**控制界的 F 轴 2012 年就占满了**：
   Trimpe & D'Andrea (IFAC NecSys 2012) 用 ‖rᵢ‖ ≥ δᵢ 决定哪个通道更新，未触发通道状态完全不动，
   并给出误差上界；还可上溯到 send-on-delta / 电平交叉采样（Miskowicz 2006，引 1950 年代工作）。**以 04 节为准。**
2. **06 说"半群 + 观测修正的联合约束检索未见"——需要收窄。**
   **CRU (ICML 2022) 已经有**精确半群（exp(AΔ)）+ 新息加性修正 + 真实秒级 Δ。真正剩下的只有：
   (i) **非线性**且精确满足半群的 Φ；(ii) 对**复合映射** T_i = C_{z_i}∘Φ_{Δ_i} 的细化一致性给出定理与误差界。
3. **03 说"B 显式吃 Δ 的孔隙 15 篇没有"——部分成立但价值有限。**
   CRU 的增益 K_t 通过 Σ⁻ 隐式依赖 Δ；"B 以 Δ 为显式输入"确未见到，但这更像工程细节，不是定理级差异。
4. **归属更正**（各节已核）：
   - **KalmanNet** 正式出处是 **IEEE Trans. Signal Processing 70 (2022)**，不是 ICML 2022。
   - **Pinchetti et al. "Benchmarking PC Networks"** 不在 PMLR v162（ICML 2022 的 PC 论文是 Frieder & Lukasiewicz，给出**发散反例**）；Pinchetti 的 benchmark 是 **arXiv:2407.01163 (2024)**。
   - **CRU** 的正确题名是 *Modeling Irregular Time Series with Continuous Recurrent Units*（ICML 2022, PMLR 162）。
   - **"delta networks"** 不是"Δ 时间网络"，指 **delta-rule 线性注意力/快权重族**（DeltaNet；Gated DeltaNet 论文标题字面即 *Gated Delta Networks*, ICLR 2025）。

## 6. 最后判决：新颖性分级与唯一可辩护的三条路线

| 层面 | 判决 | 依据 |
|---|---|---|
| 结构/方程形式 | **不新颖（0/10）** | §1、§3；CRU 已 A–E 全中；tPC Eq26 逐字；Trimpe 2012 已含硬稀疏 + 误差界 |
| 统一记号 | **边际价值**（把 PC / KF / DeltaNet / TTT / ODE-RNN 写在同一行） | 但"统一记法"通常不算论文贡献 |
| 定理 | **仍有空间，但只有三处**，且都必须与 CRU / Kalman-Bucy / 事件触发滤波明确划界 | 见下 |

**三条可辩护路线（都是"证明"，不是"结构"）：**

- **路线 A（时间）**：**非线性且精确半群**的 Φ_Δ + 复合映射 T = C∘Φ 的**细化一致性定理与误差界**。
  必须写清噪声随 Δ 的缩放（否则命题 5 直接发散），并与 CRU 的 exp(AΔ) 线性情形、Kalman-Bucy 极限划界。
  *为什么还有空间*：CRU 只覆盖线性 Φ；现有 flow-map 一致性工作（Consistency Models、Boffi et al. 2025）都只在**无观测**的生成模型里验证。
- **路线 B（学习式增益的稳定性）**：给出 B_φ(h⁻,r) 的 Lipschitz/收缩条件，使滤波误差 e = h − h* 有界或收敛；
  必须处理"纯预测误差梯度会饱和"（Titans 自证）、"h⁻ 与 r 相关时加性修正会双计过冲"（DeltaNet 用 Householder (I−βkkᵀ) 正是为此）。
  *为什么还有空间*：KalmanNet / RKN / TTT 全部**没有给出**学习式增益的收敛保证。
- **路线 C（对象级硬路由）**：把 Trimpe 2012 的"逐通道触发"升级到**对象/slot 级**，并证明未命中子状态**严格不变**（数值上不是"接近不变"）。
  *为什么还有空间*：控制界是逐传感器，slot 模型（SAVi/STEVE/SlotFormer）是软注意力、无硬保证。

**审稿人几乎必然给出的第一句话**：
> "这就是 CRU (ICML 2022) 的 μ⁻ = exp(A(t−τ))μ⁺、μ⁺ = μ⁻ + K(y − Hμ⁻)，把线性观测矩阵换成非线性解码器、把闭式增益换成网络。
> 请说明在什么假设下这个替换产生**可证明**的差异。"

## 7. 本次调研的负面结果清单（"别只挑支持这个想法的论文"）

1. 用**预测误差**作为唯一学习信号会饱和/卡局部极小（Titans 原文明说）。
2. 纯加性 B 在 h⁻ 与 r 相关时**双计**、高增益过冲；DeltaNet 用 Householder 收缩正是为此。
3. **单步修正不够**：tPC 作者自述 "a single step is nowhere near enough"，实测单步跟踪劣于多步/收敛版。
4. 修正项可以**吞掉动力学**：DVBF 原文承认可退化为 z_{t+1} = w_t（世界模型退化成编码器）。
5. 先验没训好时**拉向先验有害**：DreamerV2 KL balancing、DreamerV3 free bits+unimix 都是补救。
6. 一步最优 ≠ 多步最优（PlaNet latent overshooting）。
7. 软稀疏**不保证不动**：SAVi 每帧全量重绑定；Slot Attention 加 ε 无严格零；ACT 自曝 halting 作弊。
8. decoder-free 世界里 **D 不存在**，r 无法定义（TD-MPC/2）。
9. **不可观测子空间永不被修正**：e_{k+1} = (A−LC)e_k，ker O(C,A) 在 A−LC 下不变——"只改该改的"有一半是被动的。
10. 若**不写 B(h⁻,0) = 0**，零新息也会改状态，连观测器的基本前提都不满足。

## 8. 目录（下文为逐篇条目，统一格式）

- **01** 世界模型与潜状态预测（12 篇：DVBF / World Models / RKN / PlaNet / Dreamer / DreamerV2 / CT-MBRL / TD-MPC / DreamerV3 / TD-MPC2 / TAWM / GBF）
- **02** 预测编码与预测误差更新（11 篇：Rao–Ballard 1997/1999 / Friston 2005/2008 / Whittington–Bogacz 2017 / PredNet / Millidge 综述 / tPC / Frieder–Lukasiewicz / Alonso / Pinchetti）
- **03** 滤波器、观测器与 innovation form（15 篇：Kalman 1960 / Luenberger / 信息滤波 / CD-EKF / NCDSSM / DKF / KVAE / DVBF / RKN / KalmanNet / 学习式观测器 / Bishop–Bonilla 等 + N1–N7 负面结果）
- **04** 稀疏改写、选择性更新与对象状态（13 篇：Trimpe 2012 / Sijs–Lazar 2009 / Miskowicz 2006 / Skip RNN / ACT / Slot Attention / SAVi / STEVE / SlotFormer / SAVi++ / SOLD / DeltaNet / Gated DeltaNet）
- **05** 不规则时间与连续状态转移（13 篇：Neural ODE / Latent ODE·ODE-RNN / GRU-ODE-Bayes / Neural CDE / Neural Flows / mTAN / LTC / CfC / GRU-D / NJ-ODE / Vid-ODE / Consistency Models / Boffi flow-map）
- **06** 对抗式检索（20 篇候选先例 + 检索实录 + 退化反例构造 + 七轴矩阵）
- **07** 目标公式的重参数化、退化与新颖性判据（定理 1 通用性 / 定理 2 ≡ Kalman ≡ 线性 RNN / 命题 3 B 的三族 / 命题 4 训练目标 / 命题 5 细化极限 / 反例 N1–N12 / 约束 C1–C8）

---


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


# 07 · 目标公式的重参数化、退化与新颖性判据

> 本文件只做数学与判据，不依赖实验。对象是下面这个候选结构：

    先验预测   h⁻ = Φ_Δ(h)                (1)
    观测残差   r  = z − D(h⁻)              (2)
    加性修正   h  = h⁻ + B(h⁻, r)          (3)

    h ∈ ℝⁿ  隐状态；z ∈ ℝᵐ 观测或其编码；Δ = 相邻观测的真实时间间隔（秒）；
    Φ_Δ 带时间参数的转移；D 观测/解码映射；B 学习式修正函数（"增益"）。

**一句话判决**：式 (1)(2)(3) 作为三行写法，是"连续-离散贝叶斯滤波 / Luenberger 观测器 + 学习式增益"的记号级重述；它对函数类没有任何约束，因此**结构上不新颖**。能支撑新颖性的只剩三条**定理形状**的窄缝（§10）。下面把"为什么"逐条证出来。

---

## 1. 记号与隐含预设

式 (1)-(3) 实际上偷偷预设了四件事，先把它们显式化，后面每一条都会单独检验：

- P1 **时间可分离**：状态在观测之间只按 Φ_Δ 演化，观测只通过 (2)(3) 注入。
- P2 **误差在观测空间**：修正的唯一依据是 r，而不是原始 z。
- P3 **修正是加性的**：h 与 h⁻ 只差一个 B，不经过任何门控乘性覆盖。
- P4 **B 是学出来的**：没有给 B 指定解析形式（这正是它想主张的地方）。

只要 P1-P4 里任意一条被"补上具体形式"，公式就落进一个已有家族；如果不补，公式就是一个空的参数化。这是本文件的主线。

---

## 2. 定理 1（结构性通用）：写得出 ≠ 有结构

**命题 1.1** 任意因果 RNN

    h_i = F(h_{i-1}, z_i)                                  (4)

都可由 (1)-(3) 精确表示。

**构造 (a)**：取 D ≡ 0，Φ_Δ ≡ Id。则 r = z_i，h⁻ = h_{i-1}，
令 B(h, r) = F(h, r) − h，立即得到 (4)。

**构造 (b)**（更"像样"的版本）：取 D = Id（m = n），则 r = z_i − h⁻，
令 B(h, r) = F(h, h + r) − h，同样得到 (4)。

**构造 (c)**（带非平凡时间转移）：取任意可逆 Φ_Δ，令
B(h, r) = F(Φ_Δ⁻¹(h), h + r) − h，仍得到 (4)。

**推论 1.2** 式 (1)-(3) 定义的函数类 ⊇ 全部因果 RNN。因此"能写成先验+残差+修正"这件事不构成任何架构约束；新颖性不可能来自这三行，只能来自"(Φ, D, B) 被怎样限制"与"被证明了什么"。

**命题 1.3（分解不唯一，且没有免费的正则）** 只要不额外要求

    B(h, 0) = 0      （无预测误差 ⇒ 不修正）              (5)

则 (Φ, B) 的拆分完全任意：(Φ, B) 与 (Φ', B') = (Φ + B(·,0), B − B(·,0)) 给出同一条轨迹。
另一方面，即使加上 (5)，仍有**规范自由度**：对任意可逆 S，

    h' = S h,  Φ'_Δ = S Φ_Δ S⁻¹,  D' = D S⁻¹,  B'(h', r) = S B(S⁻¹h', r),  r' = r

给出同一观测序列下的等价模型。**结论**：任何"公式改写了状态的第 k 个分量"这类说法在潜空间里没有内禀意义，必须先固定规范（例如 D 的解码器尺度、h 的单位）。这在状态空间模型的可辨识性文献里是常识，但在新架构提案里经常被忽略。

**这条同时是负面结果**：如果 B(h, 0) ≠ 0（GRU/LSTM 式修正就是如此），那"先验"和"修正"的界线是人为划的，公式退化成"带输入注入的 RNN"而已。

---

## 3. 定理 2（线性情形）：它就是 Kalman 滤波，也就是线性 RNN

取线性参数化

    Φ_Δ(h) = A(Δ) h,   D(h) = C h,   B(h, r) = K(Δ) r

代入 (1)-(3)：

    h_i = (I − K C) A(Δ) h_{i-1} + K z_i                      (6)

**这是标准线性 RNN 的形式**，权重被约束成 W = (I − K C) A(Δ)、U = K。

**推论 2.1（≡ 卡尔曼）** (6) 与"稳态增益卡尔曼滤波器"逐字相同，其连续-离散形式（观测之间用 ODE 传播，观测时刻用新息修正）就是教科书里的 **CD-EKF**：

    预测：  x̂⁻(t_i) = 状态方程从 t_{i-1} 积分到 t_i
    新息：  e_i = z_i − h(x̂⁻(t_i))
    修正：  x̂(t_i) = x̂⁻(t_i) + K_i e_i ,   K_i = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹

把 x̂ 换成 h、h(·) 换成 D、K_i e_i 换成 B 即式 (1)-(3)。**所以目标公式的骨架不是"像"卡尔曼，而是"就是"卡尔曼的记号变体**。区别只有一处：卡尔曼的 K 由协方差递推给出，而 (3) 的 B 是任意函数。也就是说目标公式是**严格更宽**的类，因而必然丢掉卡尔曼的最优性定理（§10 的 C4）。

**推论 2.2（增益不是唯一确定的）** 在 (6) 里，只要 A − KCA 稳定，任何 K 都给出一个"合法"滤波器；K 的唯一性来自最小方差准则，而不是来自结构。没有协方差/噪声假设时，"学一个 B"学到的只是**某个**稳定化增益，不能声称是贝叶斯更新。

**推论 2.3（可观测子空间不会被修正 —— "只改该改的"有一半是自动的）**
若存在 v ≠ 0 使 C A(Δ) v = 0（对全部 Δ），则 K C A(Δ) v = 0，修正项永远碰不到 span{v}。这正是控制里的**不可观测子空间不变性**。两个后果：

- 好的一面："无关部分不动"在**不可观测方向**上是自动成立的，不需要任何稀疏机制。
- 坏的一面：需要修正但不可观测的方向，学也修正不了；这解释了 latent 滤波器在部分观测下的漂移（dead-reckoning drift）。

**推论 2.4（一步修正的秩上界）** 修正量 h − h⁻ = K r 落在 col(K) 里，维数 ≤ m = dim(z)。
非线性 B 同理：h − h⁻ 落在 ∂B/∂r 的像里，一般 ≤ m 维。
因此当 **观测维度远小于状态维度**（本项目就是：116 维特征 vs 可能更宽的状态），**单次观测只能把状态在一个 m 维子空间里推一把**；要动到全部方向必须靠多步累积（Φ 把修正传播开）。这条给出一个可检验的结构性预言，也说明"只改一部分状态"是**被 rank 逼出来的**，不是设计出来的。

### 3A. 最接近的已有实现（三篇，分属三行）

| 行 | 已有实现 | 差在哪 |
|---|---|---|
| **(1)(2)(3) 三行全中** | **CRU**（Schirmer, Eltayeb, Lessmann, Rudolph, ICML 2022）：μ⁻ = exp(A(t−τ))μ⁺，μ⁺ = μ⁻ + K(y − Hμ⁻)，Δ = t−τ **是真实秒**，K 随 Δ 变 | Φ 是线性矩阵指数（非线性版缺失）；K 闭式非网络；用途=插值 |
| (2)(3) 残差+加性修正 | **RKN**（Becker et al., ICML 2019）：z⁺ = z⁻ + q ⊙ (w − z⁻)，缺测时后验=先验 | 用线性 H 与解析增益；无 Δ；无 RL |
| (1) 带 Δ 的潜空间转移 | **TAWM**（Nhu, Son, Lin, ICML 2025, arXiv:2506.08441）：ẑ(t+Δt) = z + d(z,a,Δt)·τ(Δt) | 无残差修正；在线靠重编码 |
| (1)+(2) 连续-离散 + 学习式 | **NCDSSM**（Ansari et al., ICML 2023）：连续动力学 + 观测时刻贝叶斯更新 | 用 FPK/协方差；非残差 MLP |

结论不变：三行**各自**都有逐字先例；"三行同时出现在一个潜状态模型里"没找到，但那属于把已有模块接起来。

下面是 NCDSSM 的细节，因为它最像"学习式连续-离散滤波"的完整实现：

如果只允许指出一篇"把本公式用神经网络实现掉"的论文，那是：

> Ansari, Heng, Lim, Soh, *Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series*, ICML 2023 (PMLR 202)。
> https://proceedings.mlr.press/v202/ansari23a/ansari23a.pdf

它的定义与 (1)-(3) 一一对应：状态按连续时间动力学（随机微分方程）在观测**之间**演化（= Φ_Δ，Δ = t_k − t_{k−1} 是真实时间），
观测时刻按

    p(z_{t_k} | A_{t_k}) = p(a_{t_k} | z_{t_k}) p(z_{t_k} | A_{t_{k−1}}) / p(a_{t_k} | A_{t_{k−1}})

做贝叶斯更新（= 残差修正），并用神经网络参数化动力学与（线性高斯）观测模型。
作者明确说这是 Jazwinski (1970) 第 6 章的连续-离散滤波框架 + 神经参数化 + 变分推断。

**含义**：
- "连续流 + 离散观测修正 + 学习式参数化"这一组合**已经在 2023 年被发表**，且是在 irregular time series 这个正对的场景上。
- 它与目标公式的差别缩小到：NCDSSM 的修正是**贝叶斯/变分后验**（带协方差、用 FPK 方程），目标公式的修正是**无协方差的加性 MLP**。
  于是目标公式在"更弱、更启发式"的一侧——除非能证明"无协方差的 B 仍能逼近/超越贝叶斯更新"，否则它只是 NCDSSM 的简化版。
- 这也是 §10 判决里最硬的一条证据：不是"像"，而是"同一问题的已有解法"。

---

## 4. 命题 3：B 的三族取法，已经覆盖预测编码 / 卡尔曼 / 学习式观测器

B 是唯一真正自由的部件。把三种自然取法列出来：

| B 的取法 | 等价结构 | 出处 |
|---|---|---|
| B(h, r) = K r，K 由协方差递推 | 卡尔曼 / EKF | Kalman 1960；Jazwinski 1970 |
| B(h, r) = K r，K 由 LMI 或极点配置选 | Luenberger 型观测器 | Luenberger 1964/1971 |
| B(h, r) = η J_D(h)ᵀ Σ⁻¹ r | 预测编码的**一次推理步**（自由能下降一步） | Rao & Ballard 1999；Friston 2005；Bogacz 2017 |
| B(h, r) = β kᵀ (v − k h) | delta 规则 / DeltaNet 状态更新 | Widrow & Hoff 1960；Schlag et al. 2021；Yang et al. 2024 |
| B(h, r) = −η ∇_W ℓ(W; z) | test-time training 的快速权重更新 | Schmidhuber 1992；Ba et al. 2016；Sun et al. 2024 |
| B(h, r) = RNN(h, r) 学出来 | KalmanNet / RKN / 神经观测器 | Krishnan 2015；Becker 2019；Revach 2022 |

**关键点**：预测编码的推理步与卡尔曼的修正在代数上是同一个东西，
r 乘上一个（可能依赖 h 的）矩阵。tPC 的作者自己就写明：temporal PC 近似卡尔曼，
但**不跟踪后验方差**（Millidge, Tang, Osanlouy, Harper, Bogacz, PLoS Comput Biol 2024, e1011183）。
换句话说：**"用观测残差加性修正隐状态"不是本公式的贡献，它是预测编码与滤波理论四十年的公共骨架。**

更狠的一条：DeltaNet 的更新式

    S_t = S_{t-1} + β_t k_tᵀ (v_t − k_t S_{t-1})

就是 (1)-(3) 在 **D 随时间变化（D_t(h) = k_t h）、增益秩为 1、Φ ≡ Id** 时的特例，
而且 k_t、v_t 都由输入决定。它被明确地称为 "error-correcting delta rule"。
所以"先预测、再把观测空间残差加回去"在现代序列模型里是一个**被反复重新发明的标准动作**。

---

## 5. 命题 4：修正项不解释当前观测 —— 训练目标上的真差别

这是 (1)-(3) 与 RSSM/Dreamer 家族**唯一实质性的结构差别**，值得单独拎出来：

- 在 (3) 里，h_i 由 z_i 算出，但 D(h_i) 的作用是预测 **z_{i+1}**；
  当前观测 z_i 只是被"减掉"了一次，**没有被解释**。
- 在 RSSM 里，后验 s_t 吸收 z_t 之后要**重构 x_t 本身**（或至少其特征），
  并且先验 p(s_t|h_t) 被拉向后验（KL）。于是"编码"和"预测"是两个不同的路径。

后果（可判定，不需要实验）：

1. 若训练损失只作用在**先验** D(h⁻) 上，则 B **完全拿不到梯度**——(3) 退化成一段无人训练的自由参数。这不是小问题：它意味着"残差修正是核心机制"这个说法必须配一个**跨步**（≥2 步）预测或显式滤波监督目标，否则机制是死的。
2. 若损失是 Σ ‖z_{i+1} − D(h_i)‖²，B 只有**一步延迟**的梯度；此时 B 的最优解是条件期望
   E[h_i − h⁻ | h⁻]，即"滤波后验均值减先验"，这也正是卡尔曼更新在高斯下的含义——**又一次落回同一族**。
3. 因此真要主张新东西，训练目标必须写成"给定观测流，输出滤波后状态"的**显式状态监督**（有真值标签或仿真真值），而不是纯重构。这一点与本项目"教师状态机可给状态标签"的条件是吻合的——这可能是比结构更值钱的差异点。

---

## 6. 时间一致性：半群、复合、细化极限（唯一可能有真内容的部分）

### 6.1 Φ_Δ 的半群条件

要求 Φ 是**流**：

    Φ_{Δ₁+Δ₂} = Φ_{Δ₂} ∘ Φ_{Δ₁}   对所有 Δ₁, Δ₂ ≥ 0        (7)

逐类判断（结论见 05 节的逐式核对）：只有"某生成元的指数"才精确满足 (7)：
线性时不变 A 的 exp(AΔ)、Neural ODE/Neural CDE 的数值积分（近似满足，误差由求解器控制）、
以及显式构造的 flow map（Neural Flows）。而

- 欧拉式 h⁻ = h + Δ f(h)：只精确到 O(Δ²)，复合误差随段数累积；
- 把 Δ 拼成特征 h⁻ = MLP(h, Δ)：一般**不满足** (7)，且没有理由近似满足；
- 门控/时间常数式（LTC/CfC 的 timespan）：Δ 进入非线性自变量，**不保证** (7)，且 Δ→0 的极限必须在原式上逐项核对是否退化为恒等映射（这一条 05 节负责给判定，本文件不预设答案）。

**因此**：只要 Φ 不是显式 flow，式 (1) 的"连续时间"就只是"带 Δ 特征的离散递归"，属 05 节命题 C 的 feature-time model，而不是流。

### 6.2 复合映射不是半群（必须先承认这一点）

每个观测时刻的完整一步是

    T_i = C_{z_i} ∘ Φ_{Δ_i},   其中 C_{z}(h) = h + B(h, z − D(h))     (8)

T_i 一般**不满足**半群：观测是外部注入，不同观测数/不同插入顺序给出不同结果。所以"采样细化一致性"不能写成"T 满足半群"，只能重新定义。三个可操作的定义：

- **D1（流一致）**：Φ 单独满足 (7)。这是最容易证、也最没有信息量的一个。
- **D2（细化收敛）**：固定物理区间 [t₀, t₁]，把观测密度乘 2、4、8…，滤波输出 h(t₁) 是 Cauchy 列，且极限只依赖观测路径，不依赖采样密度。
- **D3（观测更新可交换）**：同一时刻有两个观测时，先修正 z 再修正 z' 与反序结果相同。高斯下的卡尔曼满足；EKF 一般**不满足**；任意学习式 B 几乎肯定不满足。

### 6.3 命题 5（细化极限的必要条件）—— 这条是负面结果

设 Φ 为精确流，D 光滑，B 连续且与 Δ 无关，观测密度 N = T/Δ 趋于无穷。则单步修正量

    ‖h_i − h⁻_i‖ = ‖B(h⁻, r_i)‖

若 r_i 不随 Δ 趋于 0（即模型**有偏**，这正是需要修正的情形），则总变化 ≈ N·‖B‖ → ∞，
序列**没有极限**，D2 失败。

反之，若数据确实由同一 Φ、D 生成，则 r_i = z_i − D(Φ_{Δ}h) = O(Δ)（D 光滑 + 流预测准确），
于是 ‖B‖ = O(Δ)，Σ‖B‖ = O(T)，D2 有希望成立。

**这就是循环性**：公式的采样细化一致性，前提是"预测误差趋于 0"，而预测误差趋于 0 恰恰是修正机制想达到的目标。所以**修正在细化极限下不起作用，只起一阶小量作用**——这条必须写进任何"采样细化一致"的定理的假设里，否则定理是空的。

补充（卡尔曼侧的对照）：经典结论是离散卡尔曼滤波在观测噪声按 Δ 缩放（等价于"连续观测流"）时收敛到 Kalman-Bucy 滤波器（Kalman & Bucy 1961）。**噪声不缩放时，加密观测根本没有良定义的极限**（新息是 i.i.d. 的，Σ K r 是随机游走，发散）。所以"采样细化一致"这句话里必须写明**噪声/观测模型如何随 Δ 缩放**，这是 05 节和第 6.3 节的共同前提。

---

## 7. B 的结构约束一览（每条配上是否已有先例）

| 约束 | 归约成什么 | 是否已有先例 |
|---|---|---|
| B 线性、K 常数 | 卡尔曼/观测器（线性 RNN） | 是（1960/1964） |
| B = η J_Dᵀ Σ⁻¹ r | 预测编码一次推理步 | 是（Rao & Ballard 1999；tPC 2024） |
| B = β kᵀ(v − kh) | delta 规则 / DeltaNet | 是（Widrow & Hoff 1960；Yang 2024） |
| B = −η∇ℓ | 快速权重 / TTT | 是（Schmidhuber 1992；Sun 2024） |
| B(h,0)=0 | 无误差不修正 | 是（新息/残差的定义本身） |
| B 只在 ‖r‖>θ 时触发 | 事件触发滤波 / send-on-delta | 是（控制文献，带稳定性证明） |
| B 只作用于部分 slot/通道 | slot attention / 硬掩码 / Skip RNN | 部分是（但触发依据不是观测残差） |
| B 依赖协方差特征 | RKN / KalmanNet | 是（2019/2022） |

**读法**：任何一种"把 B 定死"的自然方式，都对应一篇已发表的工作。剩下唯一自由的是"B 是 (h, r) 的任意网络"——而这正是 KalmanNet/RKN/学习式观测器已经在做的事。**于是公式的自由度恰恰落在已被占据的位置上。**

---

## 8. 反例与退化清单（负面结果）

| 编号 | 反例构造 | 归约成 |
|---|---|---|
| N1 | D ≡ 0 | 任意 RNN（B 吃掉全部结构） |
| N2 | D = Id | 同上，且 r 变成"输入减状态" |
| N3 | B 与 h⁻ 无关 | 输入注入式 RNN / GRU 变体 |
| N4 | B 线性 + Φ 线性 | 线性卡尔曼 ≡ 线性 RNN |
| N5 | B = −η∇_h ‖z−D(h)‖² | 预测编码推理步；固定点 = MAP/IEKF |
| N6 | Φ_Δ ≡ Id | 纯前馈残差精化（一次迭代的 deep equilibrium 式结构） |
| N7 | Φ_Δ = MLP(h, Δ) | feature-time model，不满足半群（命题 C） |
| N8 | Φ 精确 flow、B 与 Δ 无关、模型有偏 | 细化极限发散（命题 5） |
| N9 | B(h,0) ≠ 0 | 先验/修正界线人为，退化为普通门控 RNN（命题 1.3） |
| N10 | 无解码器（TD-MPC 式） | r 无定义，公式在该架构里不成立 |
| N11 | 观测维度 m ≪ 状态维度 n | 单步只能在 m 维子空间里修正（推论 2.4） |
| N12 | 存在不可观测方向 | 该方向永不被修正，"无关部分不动"是被动的（推论 2.3） |

N10 值得单独说：TD-MPC 家族是 **decoder-free** 的，没有 D，也就没有"观测空间残差"这个概念。
要在那类架构里用 (2)，必须重新定义"误差"（如 next-embedding prediction、consistency loss），
一旦这么做，它就不再是式 (2) 了。**这说明本公式强依赖于"有一个能重构观测的解码器"这一前提**，而该前提在高效 model-based RL 里是被有意放弃的。

---

## 9. 非退化所需约束清单（每题：是否已有先例）

- **C1** B(h,0)=0，保证"无误差不修正"。先例：新息定义。**不能算新**。
- **C2** Φ_Δ 精确满足半群 (7)。先例：Neural ODE / Neural CDE / Neural Flows，以及 **CRU 的 exp(AΔ)**（矩阵指数天然精确满足半群）。**不能算新**。
- **C3** 细化极限 D2 的**收敛定理 + 误差界**（含噪声随 Δ 缩放的假设）。先例：离散→Kalman-Bucy 的收敛是经典结果。
  **必须修正一个容易过度乐观的说法**："半群 Φ + 观测修正同处一篇"并不空——**CRU 已经做到了**（精确半群的 exp(AΔ) + 新息修正 + 真实秒）。
  真正剩下的只有：**(i) 非线性且精确半群的 Φ；(ii) 对复合映射 T_i = C ∘ Φ_Δ 的细化一致性给出定理与误差界**（CRU 没给，也没有任何人给学习式版本）。
  → **可主张，但这是定理，不是结构**。
- **C4** B 由不确定度决定并证明逼近最优增益。先例：卡尔曼/EKF/UKF、RKN、KalmanNet。**不能算新**。
- **C5** 误差驱动的**硬**稀疏/对象级路由，且**证明**无关子空间严格不变 + 稳定性。先例：事件触发卡尔曼有稳定性证明，但没有"槽位/对象"版本。→ **窄缝，可主张，同样是定理**。
- **C6** 多观测/异步观测下更新可交换（D3）。先例：高斯下序列更新可交换；非线性无。→ **窄缝**。
- **C7** 端到端共享 D 并训练。先例：VAE/RSSM/Dreamer。**不能算新**。
- **C8** Δ 与增益耦合（增益 ∝ Δ 或 ∝ 1/噪声）。先例：连续-离散滤波、Kalman 的 Q(Δ)。**不能算新**。

**结论**：C1、C2、C4、C7、C8 都已有先例，且很多是"教科书级"先例；
只有 C3、C5、C6 还留着。它们三个的共同点是：**都是"证明某个性质"，而不是"提出某个结构"**。

---

## 10. 新颖性判决（分级）

- **作为结构（先验+残差+加性修正）**：**不新颖**。证据：§2（通用性）、§3（≡ CD-EKF ≡ 线性 RNN）、§4（B 的每个自然取法都有主）、§7（约束表全被占）。
- **作为统一记号**：**有边际价值**。把 PC、KF、DeltaNet、TTT、ODE-RNN 写在同一行里，确实省事；但"统一记法"本身在审稿标准里几乎不算贡献，除非它导出一个新定理或一个反直觉的否定结论。
- **作为定理载体**：**仍有空间**，但只有 C3/C5/C6 三条，而且必须给出：完整假设、反例边界、与 Kalman-Bucy / 事件触发滤波的明确划界。

**最可能的审稿意见**（提前写下来，几乎必然出现）：
"这条公式和 CRU（Schirmer et al., ICML 2022）的 μ⁻ = exp(A(t−τ))μ⁺、μ⁺ = μ⁻ + K(y − Hμ⁻) 差别在哪？
对方已经有真实秒级 Δ、精确半群转移、新息加性修正三件套；本稿只是把线性 H 换成非线性 D、把闭式 K 换成网络 B。"

"这就是 continuous-discrete extended Kalman filter，把增益换成 MLP，把转移换成 Neural ODE。
这和 ODE-RNN（Rubanova et al. 2019）的区别只有'跳变输入取原始特征还是取观测残差'一行；
这和 DeltaNet（Yang et al. 2024）的区别只有'增益是否低秩、观测映射是否随时间变化'；
而这套东西 NCDSSM（Ansari et al. ICML 2023）已经用带协方差的贝叶斯更新做过了，
本稿的 B 反而是它的无协方差简化版。
作者需要说明在什么假设下这个差异产生可证明的后果。"

---

## 11. 对项目（流式视觉决策）的最小可辩护增量

结合本项目条件（有教师状态机 ⇒ **有状态标签**；有真实 PTS ⇒ **有 Δ**；部署在 MCU ⇒ 状态要小），
可以主张的东西不是"新架构"，而是下面三选一：

1. **有状态监督的连续-离散滤波**：用教师状态标签直接监督滤波后状态（§5 第 3 点），
   这在 RSSM/Dreamer（纯重构 + KL）里没有，在 ODE-RNN（纯重构）里也没有。
   可检验命题：有状态监督时，B 的可辨识性从"不可辨识"变成"可辨识到规范自由度"。
2. **Δ 与残差的耦合定理**：证明"增益必须随 Δ 缩放"才能让细化极限存在（命题 5 的加强版），
   并给出非渐近误差界。这是纯数学，且与工程动机（21 Hz 抖动、掉帧）直接对应。
3. **误差驱动的对象级硬路由 + 严格不变性**：把 slot/对象状态做成硬掩码，
   证明未命中对象的子状态在数值上严格不变（不是"接近不变"），并给出稳定性条件。

三条都不是"三行公式"能换来的；反过来说，**如果只交三行公式，它会被上面任何一条先例吃掉。**

---

## 12. 待完成的证明义务（下一轮）

- [ ] 把 §6.3 的命题 5 写成完整定理：给出 r_i = O(Δ) 的**定量的**条件（D 的 Lipschitz 常数、Φ 的流误差），以及模型有偏时的发散速率。
- [ ] 给出 B(h,0)=0 且 B 为 Lipschitz 时，滤波映射 T 的**不动点存在性**与**偏差界**（与卡尔曼的偏差比较）。
- [ ] 在 Δ 依赖的增益 K(Δ) 下，重算 (6) 的连续极限，验证是否复现 Kalman-Bucy（若复现，则 C4 彻底失守；若不复现，找出差异项）。
- [ ] 把 §2 的通用性构造 (a)(b)(c) 写成正式命题并检查 m ≠ n 的边界情形。
- [ ] 明确 CfC 的 Δ→0 极限（交给 05 节的逐式核对结果）。

---

*本文件只给判据与构造，不给"这个想法一定行/不行"的经验结论。所有"已有先例"的判断都以 01-06 节的逐篇条目为准；若某条先例在 01-06 节被核实为不成立，本文件的对应结论应同步收回。*
