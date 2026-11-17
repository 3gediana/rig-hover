# RIG-Hover 端侧“敲击两下”时空动作识别与端侧强化学习决策架构设计规范

**日期**：2026-11-15  
**作者**：Antigravity（应用户“自主工程决策 + 进组投名状/请柬作品”要求制定）  
**状态**：Approved（用户已明确授权由 Agent 自主做全部技术决策并直接落地验证）

---

## 1. 背景与核心意图（Intent & Why）

本项目针对陆吾智能 RIG-Hover（代号“文文”，ESP32-S3 双轮自平衡桌面机器人，单自由度水平摇头舵机，GC0308 240×240 摄像头）构建完整的**视觉交互与自主控制闭环**。

### 1.1 为什么这是一份面向机械臂/强化学习导师的“进组投名状”
用户是大二人工智能专业学生，希望进入导师课题组直接参与机械臂与强化学习（RL）核心研发，而非打杂。一位做机器人控制与强化学习的导师在审阅本科生作品时，最看重四个核心素养：
1. **问题建模的科学性（Problem Formulation）**：能分清哪里是高维感知（像素+姿态运动学 $\to$ 接触脉冲与动作语义）、哪里是时序工作记忆（Target Lock 敲击点保持）、哪里是受约束动力学控制（轮式倒立摆 + $\zeta \approx 0.08$ 欠阻尼航向环），绝不“拿神经网络硬学三行 `if` 规则”，也绝不“把测试集当验证集刷虚高指标”。
2. **小样本与跨数据集工程能力（Data-Centric Rigor）**：在没有新录真机数据的约束下，能从粗粒度区间标注中联合运动学与视觉冲击能量挖掘**单次接触脉冲（Contact Impulse）**，结合训练折内部运动学时序组合增强（Compositional Augmentation）与外部公开数据集（Charades 室内桌面动作、UCF-101 节律动作、本地 `evidence/sessions` 真机干扰集）完成**零误报压力测试**。
3. **架构演进判断力（End-Side Architecture Shift）**：基于实测数据（板上 48 KB SRAM 争抢 + 437 次实验证明指数衰减 CfC 在时序计数任务上准确率仅 0.336 ≈ 随机猜测），果断将决策模型从 ESP32-S3 迁至**端侧（PC / Android 手机端）**，机器人退化为受 800 ms 硬件看门狗保护的纯底层平衡与伺服执行机构（`GET /status`、`GET /capture`、`GET /cmd`）。
4. **强化学习与安全控制（Safe RL & Sim-to-Real）**：在依据固件源码逐行复刻并标定的精确零阶保持（ZOH）轮式倒立摆闭环仿真器中，用**行为克隆预热 + 近端策略优化（BC Warm-Start + PPO）**结合**非对称桌沿防过冲安全屏障（Asymmetric Safety Barrier）**与**动作平滑正则（Jerk Regularization）**训练端侧策略，并导出可直接装入 Android 控制端的 `float32` `policy.tflite`（39 维输入 $\to$ 3 维输出）。

---

## 2. 目标、非目标与反目标（Goals, Non-Goals, Anti-Goals）

### 2.1 目标（Goals）
1. **多源“敲击两下”数据集构建（`pc/train/double_knock_data.py`）**：
   - 从 `dataset/takes/` 的 4 条已标 take（12,791 帧，46 个粗粒度敲击区间，31 个硬负样本区间）中，融合 YOLO26n-pose 腕部竖直运动学（$y_{\text{wrist}}, v_y, a_y$）与手桌局部帧差能量，精确定位每次桌面接触脉冲时刻，将窗口划分为四类动作：`0=non_knock`、`1=single_knock`、`2=double_knock`（敲击两下）、`3=multi_knock`（3 次及以上连续敲击）。
   - 挖掘本地未并入训练集的真机数据（`evidence/sessions/` 下 `knock_d0.5_front_night_182129` 及 9 条非敲击手势片段，共 1,987 帧）与本地公开动作数据集（`G:\stream_memory_data\videos\charades\Charades_v1_480.zip` 中的桌面交互类 `c009/c011/c012/c014/c109`，以及 `UCF-101.zip` 中的节律/手部动作类 `Drumming/Hammering/Typing/TableTennisShot/WritingOnBoard/CuttingInKitchen`），构建跨数据集硬负样本与节律测试集。
2. **双流双头“敲击两下”动作识别网络（`DoubleKnockNet`）**：
   - 设计视觉流（96×96 2通道灰度+帧差 CNN）+ 运动学流（8 维距离无关姿态运动学特征）$\to$ 多尺度膨胀 TCN + 因果 GRU（无衰减记忆状态）$\to$ **接触脉冲头 $\hat{p}_{\text{imp}}(t)$ + 窗口动作计数头 $\hat{p}_{\text{cnt}}(c \in \{0,1,2,3\})$**。
   - 在严格按 take 留一（4 折 LOO，一切平滑/阈值仅在 3 条训练 take 上标定）下评估单次接触检测与“敲击两下”动作识别性能，并在跨数据集干扰集上验证状态门控（`MODE_GAZE_FIST`）+ 双头协同的零误报能力。
3. **端侧闭环决策架构与强化学习策略（`pc/end_side/`）**：
   - 实现完整端侧控制环（`end_side_protocol.py` + `end_side_controller.py`），通过 HTTP Keep-Alive 对接机器人 `GET /status`、`GET /capture`、`GET /cmd?vx=...&vyaw=...&head=...`（含 800 ms 看门狗喂狗保证）。
   - 在 ZOH 物理闭环仿真器中实现端侧强化学习训练器（`train_rl_policy.py`：BC Warm-Start + PPO + 域随机化 + 非对称过冲惩罚 + Jerk 正则），在 5 类经典注视场景 + 完整 Scenario v2（盯脸 $\to$ 盯拳 $\to$ 敲击两下触发 $\to$ 锁定目标接近（手离开后继续）$\to$ 驻留 $\to$ 回看人脸）上系统对比 `PD`、`Rule Teacher`、`On-Board CfC`、`Pure BC` 与 `End-Side RL Policy`。
   - 导出符合 [`G:\robot_app\android\model_schema.md`](file:///G:/robot_app/android/model_schema.md) 规范的 39 维输入、3 维输出 `float32` `policy.tflite` + `model.json` + `.zip` 导入包，并通过 TFLite 解释器与 SHA-256 校验。

### 2.2 非目标（Non-Goals）
- 不修改或烧录 ESP32-S3 固件（当前无真机在线，且端侧架构直接复用固件现有的 `/status`、`/capture`、`/cmd` 接口）。
- 不声称跨人泛化结论（当前真机桌面 take 来自同一操作者与房间，报告中如实披露局限性）。

### 2.3 反目标（Anti-Goals — 触犯即视为失败）
- **严禁测试集调参（Zero Test-Set Leakage）**：留出 take 绝不参与阈值扫描、平滑窗口选择或合成增强生成。
- **严禁把规则硬塞进黑盒而放弃安全边界**：桌面防跌落（零过冲）与 800 ms 看门狗必须在端侧具备可证明的安全屏障（Safety Shield）。

---

## 3. 方案对比与决策（Approaches Considered）

### 3.1 “敲击两下”动作识别方案对比

| 方案 | 做法 | 优点 | 缺点 | 决策 |
|---|---|---|---|---|
| **A. 纯区间二分类 + 规则数峰**（旧方案 `train_knock.py`） | 只训逐帧“是否在敲击区间”，后处理用固定阈值找局部峰 | 实现简单 | 粗区间内非接触帧（抬手/蓄力）全被标为 1，概率包粘连，跨折误报高达 0–9.14 次/分，无法直接判别 1 下 vs 2 下 vs 3+ 下 | ❌ 淘汰（仅作基线对照） |
| **B. 纯 3D CNN / Video Transformer 端到端四分类** | 直接把 64 帧像素喂给 3D ResNet 预测 $\{0,1,2,3\}$ | 端到端无需关键点 | 240×240 下手仅 10–21 px，4 条 take 极易过拟合背景与衣服纹理，且丢失距离无关的腕部运动学先验 | ❌ 淘汰 |
| **C. 视觉-运动学双流时空网络 + 接触脉冲/窗口计数双头（`DoubleKnockNet`）+ 训练折运动学合成增强** | 融合 96×96 局部/全局帧差视觉特征与 8 维距离无关腕部运动学（$y_w, v_y, a_y$），经多尺度 TCN + 因果 GRU 同时预测逐帧接触脉冲 $\hat{p}_{\text{imp}}(t)$ 与窗口敲击次数 $\hat{p}_{\text{cnt}}(c\in\{0,1,2,3\})$；训练折内做单击脉冲组合增强 | 既利用像素级桌沿接触冲击，又具备距离无关运动学与无衰减计数状态；双头互验大幅压低误报 | 需先从粗区间中提取并校验单次接触脉冲真值 | ✅ **选用（我方决策）** |

### 3.2 端侧决策模型方案对比

| 方案 | 做法 | 优点 | 缺点 | 决策 |
|---|---|---|---|---|
| **A. 端侧直接跑旧版 116 维 CfC** | 把原 ESP32 上的 CfC 挪到 PC 跑 | 改动最小 | `stream_memory_research` 已证实 CfC 在时序记忆/计数上接近随机，且与 Android 端 39 维 `PolicyFeatureBuilder` 契约不兼容 | ❌ 淘汰（仅作基线对照） |
| **B. 纯行为克隆（BC）模仿 `ScenarioV2` 教师** | 用教师状态机生成轨迹做 MSE 监督学习 | 训练快、稳定 | 受限于教师本身缺陷（教师未考虑倒立摆加减速俯仰耦合与 $\zeta\approx 0.08$ 航向振荡，闭环成功率仅 77.3%，且动作变化率 jerk 较大） | ❌ 不单独采用（仅作 RL 预热与消融对照） |
| **C. 端侧分层架构：工作记忆状态机 + BC 预热 PPO 强化学习策略 + 非对称安全屏障** | 端侧高层维护可解释的 `TargetLock`（敲两下锁定方位与距离，手离开后里程计闭环保持），底层控制策略用 BC Warm-Start + PPO 在 ZOH 倒立摆仿真（含域随机化、非对称防过冲奖励、Jerk 平滑正则）中强化训练，同时导出 39 维 `policy.tflite` 与端侧 PyTorch 控制器 | 兼顾可解释性、零过冲桌面安全、低抖动阻尼控制，且完美展示 RL 核心能力（MDP 建模、奖励塑形、Sim-to-Real 域随机化、TFLite 端侧部署） | 需要实现完整的 PPO 微调闭环与 TFLite 导出验证 | ✅ **选用（我方决策）** |

---

## 4. 详细技术设计（Detailed Design）

### 4.1 多源数据挖掘与接触脉冲标注提取（`pc/train/double_knock_data.py`）

1. **接触脉冲定位算法（Contact Impulse Pinpointing）**：
   - 在已标注的 `knock` 区间 $[t_a, t_b]$ 内，给定灰度帧序列 $I_t$ 与 YOLO26n-pose 腕部轨迹 $(x_w(t), y_w(t), c_w(t))$ 及人体尺度 $S_{\text{body}}(t)$：
   - 计算归一化腕部竖直速度 $v_y(t) = \frac{y_w(t) - y_w(t-1)}{S_{\text{body}}(t)}$（向下为正）与触底回弹减速度 $a_{\text{rebound}}(t) = \max(0, v_y(t) - v_y(t+1))$；
   - 计算手部 ROI（若无腕点则取全帧下半区桌面带）的局部帧差冲击能量 $E_{\text{roi}}(t)$；
   - 综合冲击得分 $S_{\text{impact}}(t) = z(E_{\text{roi}}(t)) + 0.8 \cdot z(a_{\text{rebound}}(t))$，采用自适应中位数绝对偏差（MAD）门限与最小生理敲击间隔 $\Delta t_{\min} = 0.20\text{ s}$ 提取局部极大值作为接触脉冲帧 $\{t_k^*\}$。
2. **窗口级四分类数据集构建**：
   - 滑窗长度 $T_w = 48\text{ 帧}$（20 fps 下为 $2.4\text{ s}$，足以完整容纳一次 $0.25\text{–}1.2\text{ s}$ 间隔的双敲动作及其前后上下文）：
   - 窗口内含接触脉冲数 $K = 0 \implies c = 0$（`non_knock`，细分来源：`fist_hold`、`hand_raise`、`reach`、`other`、`background`）；
   - $K = 1 \implies c = 1$（`single_knock`）；
   - $K = 2$ 且两峰间隔 $\Delta t \in [0.18, 1.40]\text{ s} \implies c = 2$（`double_knock`，“敲击两下”正类）；
   - $K \ge 3$（或间隔不在双敲范围内的高频连击）$\implies c = 3$（`multi_knock`）。
3. **训练折专属运动学时序组合增强（Training-Only Compositional Augmentation）**：
   - 仅在每个留一折的 **3 条训练 take** 内，将切出的单次敲击脉冲片段（$\pm 0.35\text{ s}$）与真实负样本背景/举拳片段按可控间隔：
     - 合成确定性的 `single_knock`（1 次脉冲）、`double_knock`（2 次脉冲，间隔在 $[0.22, 0.85]\text{ s}$ 均匀采样）和 `multi_knock`（3–4 次脉冲），平衡四类样本分布。
   - **铁律**：留出测试 take 绝不使用任何合成样本，100% 在原始真实视频流上评测。
4. **跨数据集挖掘（Local Sessions + Charades + UCF-101）**：
   - 本地 `evidence/sessions/`：`knock_d0.5_front_night_182129`（真机 0.5m 双敲测试集）+ 9 条非敲击手势录制（1,798 帧真机硬负样本）。
   - `Charades_v1_480.zip`：筛选包含桌面动作标注（`c009 Putting something on a table`, `c011 Sitting at a table`, `c012 Tidying up a table`, `c014 Working at a table`, `c109 Putting a cup/glass/bottle somewhere`）的真实室内视频片段，解码为 240×240 20fps 灰度流。
   - `UCF-101.zip`：抽取 `Typing`, `WritingOnBoard`, `CuttingInKitchen`, `TableTennisShot`, `Drumming`, `Hammering` 真实视频片段，检验模型与门控在跨场景复杂动作下的误报率。

### 4.2 `DoubleKnockNet` 网络结构与双头协同推理（`pc/train/train_double_knock.py`）

```
输入窗口 T=48 帧 (2.4s @ 20fps)
├─ 视觉分支: (B, T, 2, 96, 96) [灰度 + 相邻帧绝对差]
│    └─ 3 层深度可分离/标准 Conv2d + BN + SiLU + GlobalAvgPool ─▶ v_t ∈ R^64
├─ 运动学分支: (B, T, 8) [wx, wy_norm, vy, ay_rebound, vx, hand_conf, face_valid, roi_energy]
│    └─ Linear(8→32) + LayerNorm + SiLU ─▶ k_t ∈ R^32
└─ 融合特征: x_t = Concat(v_t, k_t) ∈ R^96
     ├─ 多尺度膨胀因果 TCN (dilation = 1, 2, 4, kernel = 3) ─▶ u_t ∈ R^96
     └─ 因果 GRU (hidden = 96, 无指数衰减自环以保持计数状态) ─▶ h_t ∈ R^96
          ├─ Head 1 (逐帧接触脉冲头): Linear(96→1) ─▶ logit_imp(t) ∈ R^T
          └─ Head 2 (窗口动作计数头): AttentionPool(h_{1:T}) + Linear(96→4) ─▶ logit_cnt ∈ R^4
```

- **损失函数**：
  $$\mathcal{L} = \mathcal{L}_{\text{Focal-BCE}}\big(\hat{p}_{\text{imp}}(1:T),\, y_{\text{imp}}(1:T)\big) + \lambda_{\text{cnt}} \mathcal{L}_{\text{CE}}\big(\hat{p}_{\text{cnt}},\, c_{\text{window}}\big)$$
- **在线双头协同触发器（`DoubleKnockTrigger`）**：
  - 条件 1（动作语义确认）：窗口分类头预测 $\hat{P}(c=\text{double\_knock}) + 0.35\,\hat{P}(c=\text{multi\_knock}) \ge \tau_{\text{cls}}$（阈值 $\tau_{\text{cls}}$ 仅在训练折标定）；
  - 条件 2（物理脉冲计数确认）：逐帧脉冲头经轻量平滑与在线 z-score 后，在最近 $[0.20, 1.45]\text{ s}$ 窗口内恰好检测到 $\ge 2$ 个显著接触脉冲峰（$\ge \tau_{\text{imp}}$）；
  - 条件 3（系统级状态机门控 + 冷却）：当接入端侧机器人状态机时，仅在 `mode == MODE_GAZE_FIST`（已确认人脸并盯住拳头等待敲击）下放行，触发后进入 `cooldown_s = 2.5 s` 不应期。

### 4.3 端侧闭环决策架构与强化学习策略（`pc/end_side/`）

1. **端侧通信协议与看门狗守护（`pc/end_side/end_side_protocol.py`）**：
   - 严格对齐 [`G:\robot_app\PROTOCOL.md`](file:///G:/robot_app/PROTOCOL.md)：
     - `GET /status` $\to$ 解析 28 维本体遥测 `RobotTelemetry`；
     - `GET /capture` $\to$ 获取 240×240 JPEG 图像帧；
     - `GET /cmd?vx=...&vyaw=...&head=...` $\to$ 下发控制指令（`vx, vyaw` $\in [-100, 100]$，`head` $\in [-75^\circ, +75^\circ]$），内置最大间隔 `< 400 ms`（满足 800 ms 硬件看门狗）的心跳保活与急停锁。
2. **39 维端侧策略特征契约对齐（`EndSideFeatureBuilder`）**：
   - 与 Android 端 [`PolicyFeatureBuilder.kt`](file:///G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ml/PolicyFeatureBuilder.kt) 逐维完全对齐（下标 `0..27` 为归一化本体遥测，`28..32` 为主目标槽 `[cx, cy, w, h, conf]`，`33..37` 为次目标槽/锁定敲击目标槽，`38` 为目标有效位）。
   - **巧妙设计（向后兼容 Android App 的 39 维接口）**：
     - 在 `GAZE_FACE` / `GAZE_FIST` 阶段，目标槽 0 填入当前注视目标（人脸或拳头），目标槽 1 全 0；
     - 在 `DoubleKnockNet` 触发“敲击两下”进入 `APPROACH` 阶段后，端侧高层工作记忆 `TargetLock` 将锁定的敲击点相对方位与剩余距离映射为虚拟目标框填入**目标槽 1**（`conf = 1.0` 表示已触发双敲锁定巡航）！这样**不需要改动 Android App 的 39 维张量定义**，同一个 39 维 `policy.tflite` 策略网络就能同时学会“原地柔顺注视（头身分离）”与“双敲锁定后平滑接近并在安全距离精准刹停”！
3. **强化学习训练（`pc/end_side/train_rl_policy.py`：BC Warm-Start + PPO）**：
   - **环境（MDP）**：基于 `pc/sim/sim.py` 的精确 ZOH 轮式倒立摆 + 欠阻尼航向环 + 二阶头部伺服，扩展支持完整 Scenario v2 阶段切换与**域随机化（Domain Randomization）**：每回合随机扰动机体质量 $\pm 15\%$、轮面阻尼 $\pm 20\%$、航向增益 $\pm 15\%$、传感器噪声与 1–2 步通信延迟。
   - **奖励函数（Reward Shaping）**：
     $$r_t = r_{\text{gaze}}(t) + r_{\text{head\_sep}}(t) + r_{\text{approach}}(t) - c_{\text{overshoot}}(t) - c_{\text{pitch}}(t) - c_{\text{jerk}}(t)$$
     - $r_{\text{gaze}} = \exp\big(-(\varepsilon_{\text{gaze}} / 10^\circ)^2\big)$：鼓励视线精确对准目标中心；
     - $r_{\text{head\_sep}} = 0.25 \cdot \mathbb{I}(|\varepsilon_{\text{body}}| \le 42^\circ) \cdot \exp\big(-(v_{\text{yaw}} / 15^\circ\text{/s})^2\big)$：脖子够得到时严禁乱动轮子，主动抑制 $\zeta \approx 0.08$ 航向振荡；
     - $r_{\text{approach}} = \mathbb{I}(\text{APPROACH}) \cdot \big(1.5 \exp(-(d_{\text{rem}}/0.25)^2) + 0.5\,\dot{d}_{\text{progress}}\big)$：驱动机器人平稳走向锁定敲击点；
     - $c_{\text{overshoot}} = 10.0 \cdot \max(0, -d_{\text{rem}})$（**非对称桌沿安全屏障**）：一旦越过安全停靠点（过冲），施加极重惩罚；
     - $c_{\text{jerk}} = 0.15 \cdot \|\mathbf{u}_t - \mathbf{u}_{t-1}\|_2^2$：惩罚指令突变，消除俯仰点头与航向抽搐。
   - **两阶段训练**：
     1. **Stage 1（BC Warm-Start）**：在阻尼增强版专家轨迹上预训练 Actor 网络（`39 → 128 → 128 → 3` Tanh MLP，完全兼容标准 TFLite 算子）；
     2. **Stage 2（PPO Fine-Tuning）**：用 Generalized Advantage Estimation ($\text{GAE}(\gamma=0.99, \lambda=0.95)$) + Clipped Surrogate Objective ($\epsilon=0.2$) 在域随机化闭环仿真中微调 Actor-Critic，联合端侧确定性安全屏障（Safety Shield）实测对比。
4. **多平台导出与校验（`pc/end_side/export_tflite_bundle.py`）**：
   - 将训练好的端侧 RL 策略导出为 IEEE-754 `float32` `policy.tflite`（输入 `[1, 39]`，输出 `[1, 3]` = `[vx, vyaw, head]`），生成包含真实 SHA-256 的 `model.json` 并打包为：
     - `G:\robot_app\android\app\src\main\assets\models\hover_rl_policy.tflite` + `hover_rl_policy.json`（内置路径 A）
     - `D:\robot_work\runs_end_side\hover_rl_policy_bundle.zip`（应用内一键导入包路径 B）
   - 使用 Python `tensorflow.lite.Interpreter`（或纯 FlatBuffers TFLite 生成+解析校验器）验证 PyTorch 与 `.tflite` 数值一致性误差 $< 10^{-5}$。
