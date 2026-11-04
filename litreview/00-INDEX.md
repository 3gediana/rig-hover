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
