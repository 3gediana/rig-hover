# RIG-Hover 全链路升级设计规范：SE(2) 空间记忆、2D 复杂沙箱强化学习与自适应数字孪生座舱 App

**日期**：2026-11-18  
**范围**：PC 端空间记忆与沙箱强化学习（`D:\robot_work\pc\end_side\`） + Android 端策略外壳、性能优化、内置数字孪生沙箱与 UI 座舱重构（`G:\robot_app\android\`）

---

## 1. 背景与核心痛点诊断

围绕核心目标——**“记住桌面双敲的真实空间位置，哪怕人手立刻离开、路人横穿干扰、中途改敲新位置或存在桌面障碍/延迟丢帧，机器人都能稳健、平滑、零过冲地走过去停住，并在手机 App 上提供兼具高扩展性、丝滑性能与直观可视化的操控体验”**——对现有链路审计发现两类关键瓶颈：

### 1.1 空间记忆与复杂场景决策的四个结构性盲区
1. **盲走旋转失真**：原 `EndSideController` 与 `ScenarioV2Engine` 在手离开视野（`hand_valid = 0`）后将 `bearing_body` 冻结为常数，未扣除机身偏航角变化 $\Delta\psi_{\text{body}}$；一旦敲击点偏离机身中轴线（如 $+40^\circ$）且手迅速撤离，机器人转向后仍以为目标在 $+40^\circ$，产生原地打转或偏航发散。
2. **一维标量里程误差**：原方案使用 `remaining_m = dist_m - (wheel_x - wheel_x0)`，把曲线行驶弧长当作朝目标连线的投影位移；在大偏角弧线接近或绕障时存在显著几何误差。
3. **非敲击手势与路人干扰劫持**：原逻辑在 `APPROACH` 期间只要 `hand_valid = 1` 就覆盖 `bearing_body`，用户敲完桌后抬手挥动会把桌面锁定点拖走；同时原 RL 策略训练时在 `APPROACH` 阶段仍从 `slot0`（视觉检测框）读取转向角，若有旁人经过视野（`slot0` 被路人占据）会被诱骗偏航。
4. **大偏角饱和与中途改敲缺失**：原 `slot1.cx = clamp(0.5 + (bearing_body - head_q)/60, 0, 1)` 在相对视角超过 $\pm 30^\circ$ 时饱和；且 `APPROACH`/`DWELL` 阶段屏蔽了双敲检测，无法支持“走向 A 点途中改敲 B 点即时重路由”。

### 1.2 Android App 布局、性能与无真机演示体验瓶颈
1. **竖屏紧凑布局挤压**：原 `ControlScreen` 在不可滚动 `Column` 内垂直堆叠 8 个固定高度组件（累计约 `671dp`），在常见手机高度下将核心图传画面挤压至不足 `110dp`，横屏下更无法正常排布。
2. **高频重组与内存分配抖动**：`ControlScreen` 顶层同时订阅 26 个 `StateFlow`（含 10–20Hz `status`/`stats`），每次遥测刷新触发整页重组；`PolicyFeatureBuilder` 每次推理创建 `Sequence + sorted List`；`DoubleKnockKinematicsAndGate` 每帧拷贝 `toList()` 并分配多层临时浮点数组。
3. **缺少直观的 2D 空间记忆可视化与离线数字孪生验证台**：没有真机在手时（如办公室现场向导师演示），App 停留在未连接向导页，无法直接展示端侧 TFLite 策略网络的空间记忆、抗干扰与避障闭环能力。

---

## 2. 方案选型与架构设计

### 2.1 备选方案权衡
- **方案 A：纯端到端循环网络黑盒记忆（LSTM/GRU 直接吃原始序列）**  
  缺点：轮速里程计与陀螺偏航满足精确的二维刚体运动学（SE(2)），让神经网络从零拟合三角函数刚体变换违背工程第一性原理（参见 `AGENTS.md` §5.7 #40），在未见转角与中途接管下易漂移且不可解释。
- **方案 B（采用）：SE(2) 刚体里程计空间记忆 + 特权非对称 Actor-Critic 沙箱强化学习 + 2D 安全屏障**  
  将确定性的二维刚体逆运动学交给 `SpatialMemoryTracker` 精确维护自车坐标系下的锁定点 $(x_b, y_b)$，将**多自由度非线性协同（头身分离、欠阻尼航向阻尼、大偏角弧线平滑追击、抗路人干扰门控、障碍偏转与倒立摆俯仰刹停）**交给端侧残差 Tanh 策略网络 `EndSidePolicyNet`（通过三阶段复杂沙箱 PPO 训练），外层套二维净空与桌沿零过冲 `EndSideSafetyShield`。数学边界清晰、100% 兼容标准 TFLite 内置算子、双端（PC + Android）逐行对齐。

### 2.2 SE(2) 自车空间记忆追踪器（`SpatialMemoryTracker`）
- **锁定初始化（`lock_from_polar`）**：仅当 `knock_double_fired == True`（且距上次锁定超过 `0.65s` 防抖窗口，支持 `APPROACH`/`DWELL` 中途重敲覆写！）时，利用当前机身方位角 $\beta_0 = \theta_{\text{fist}} + q_{\text{head}}$ 与去过冲安全距离 $d_0 = \max(0, \hat{d} - d_{\text{margin}})$ 初始化自车系二维坐标：
  $$x_b = d_0 \cos\beta_0, \qquad y_b = d_0 \sin\beta_0$$
  （约定：$+x_b$ 为机身正前方，$+y_b$ 为机身右侧，与 `eps_body > 0` 向右一致。）
- **里程计刚体逆变换更新（`update_odometry`）**：每步由轮速编码器均值增量 $\Delta s = \bar{x}_w(t) - \bar{x}_w(t-\Delta t)$（无编码器增量时回退 $v_x \Delta t$）与纯机身航向角增量 $\Delta\psi = \text{wrap}_{180}((\psi_{\text{imu}} - q_{\text{head}})_t - (\psi_{\text{imu}} - q_{\text{head}})_{t-\Delta t})$ 执行精确半角刚体逆变换：
  $$
  \begin{bmatrix} x_b' \\ y_b' \end{bmatrix} =
  \begin{bmatrix} \cos\Delta\psi & \sin\Delta\psi \\ -\sin\Delta\psi & \cos\Delta\psi \end{bmatrix}
  \begin{bmatrix} x_b - \Delta s \cos(\Delta\psi/2) \\ y_b - \Delta s \sin(\Delta\psi/2) \end{bmatrix}
  $$
  由此实时解出：
  - 机身方位角 $\beta_{\text{body}} = \operatorname{atan2}(y_b', x_b') \in [-180^\circ, 180^\circ]$；
  - 真实欧氏剩余距离 $r = \sqrt{x_b'^2 + y_b'^2}$（当纵向分量 $x_b' \le 0$ 且 $|y_b'| < 0.12\text{m}$ 时判定已越过目标垂面，令有效前向剩余距离 $r_{\text{eff}} = 0$，触发零过冲刹停！）；
  - **防诱骗规则**：锁定期间忽略一切未触发双敲的普通手部移动（`S3` 挥手不改写 $(x_b, y_b)$）与路人视觉框（`S2`）。

### 2.3 39 维特征向量无损升级（兼容 `[1, 39] -> [1, 3]` TFLite 契约）
保持输入 `[1, 39]`、输出 `[1, 3]` 形状不变，充分利用原先冗余常值位 `slot1[34..36]`：
- `out[33]`：`clamp(0.5 + (bearing_body - head_q) / 60.0, 0.0, 1.0)` —— 精细视线相对误差（$\pm 30^\circ$ 高分辨率，中心为 `0.5`）；
- `out[34]`：`clamp(0.5 + bearing_body / 180.0, 0.0, 1.0)` —— **广角机身相对误差**（$\pm 90^\circ$ 线性不饱和，中心为 `0.5`，解决超大偏轴盲区）；
- `out[35]`：`clamp(remaining_m / 1.5, 0.0, 1.0)` —— 欧氏剩余距离归一化；
- `out[36]`：`clamp(clearance_m / 1.0, 0.0, 1.0)` —— 前向净空/安全裕度（畅通为 `1.0`，接近障碍物/桌沿/近距人脸时平滑降至 `0.0`）；
- `out[37]`：`1.0`（双敲空间记忆激活标志；未锁定时 `33..37` 为次视觉框或全 `0`，且常规次视觉框 `score` 封顶 `0.85` 或在策略门控中通过 `target_lock` 明确区分）。
同时保持 $39\times 39$ 零中心化投影矩阵 $P$ 折叠至首层权重 $W_1^{\text{folded}} = W_1 P$，确保正前方静止平衡点严格零偏置。

### 2.4 2D 复杂桌面物理沙箱（`HoverSandboxEnv`）与 8 大压力测试集
耦合 `pc/sim/sim.py` 的 167Hz LQR 二阶自平衡倒立摆动力学与二维桌面平面运动学 $(x_w, y_w, \psi_{\text{body}})$，内置延时队列、丢帧噪声、舵机死区回差、车轮滑移、圆形障碍物排斥势与矩形桌面悬崖边界，覆盖 8 个场景：
1. `S1_blind_vanish`：双敲后人手与人脸立即全部消失，纯盲走曲线接近；
2. `S2_bystander_distractor`：双敲后手消失，另一张路人人脸横穿视野试图诱骗 `slot0`；
3. `S3_waving_decoy`：双敲后手抬起大幅挥舞（非双敲），检验空间记忆不被拖走；
4. `S4_mid_reknock`：走向敲击点 A 途中，在另一侧敲击点 B 再次双敲，检验动态重锁定与平滑变轨；
5. `S5_extreme_off_axis`：转头状态下在机身偏角 $\pm 68^\circ$ 处双敲，检验广角协同转身与弧线入位；
6. `S6_obstacle_detour`：连线侧方存在桌面水杯障碍物，利用局部势场偏转角与净空通道绕行入位；
7. `S7_cliff_guard`：靠近桌面边缘（`1.2m × 1.0m`）敲击并叠加轮速打滑，检验桌沿硬防坠与零跌落；
8. `S8_latency_dropout_storm`：`60ms` 通信延迟 + `35%` 视觉随机丢帧 + `±3.5°` 舵机死区 + 陀螺零偏漂移。

### 2.5 Android App UI、布局、性能与内置数字孪生沙箱升级
1. **内置 20Hz 数字孪生沙箱（`SandboxSimulator` + `SandboxRobotTransport`）**：实现 `RobotTransport` 接口，无需连接真实硬件即可在手机本地闭环运行真实 `hover_rl_policy.tflite` + `ScenarioV2Engine` + `SpatialMemoryTracker` + `EndSideSafetyShield`，支持在雷达桌面上**单击放置人脸 / 双击任意位置触发双敲目标 / 一键切换 4 种压力挑战**。
2. **三模主视窗与 2D 空间记忆雷达（`SpatialMemoryRadar`）**：支持 `LIVE CAM`、`2D 雷达`、`PiP 双视窗` 一键切换，实时绘制桌面网格、安全边界、机身朝向与 $60^\circ$ 头部视锥、历史轨迹尾迹、障碍物以及脉冲闪烁的敲击记忆锚点。
3. **自适应座舱布局**：
   - 顶部紧凑 `TelemetryRibbon`（含姿态横滚/俯仰微缩地平仪胶囊，点击展开完整轮速/PID/延迟详情），释放约 `95dp` 纵向空间；
   - 底部采用**双页签战术甲板**（`驾驶与策略` / `云台与表情`），彻底消除小屏竖向挤压；
   - 横屏或宽屏（`maxWidth >= 640.dp`）自动切换为左右双栏专业地面站布局。
4. **性能改造**：核心状态类添加 `@Immutable`、高频遥测下沉至叶子组件订阅、`PolicyFeatureBuilder` 改为零分配单趟 Top-2 扫描、`DoubleKnockKinematicsAndGate` 预分配环形帧池与运动学工作数组、非激活态动画自动休眠，并消除全部过期图标告警。
