# 交接文档：RIG-Hover 桌面机器人二次开发

> 生成时间：2026-11-02 23:4x（Asia/Shanghai）
> 用途：上下文压缩后恢复记忆。**先读本文，再按文末"下一步"继续。**

---

## 0. 一句话现状

机器人**仍是原厂固件**（未刷机）；我们已完成：硬件能力摸清、调试接口打通、固件补丁写好但**编译未成功**、Android App 源码已产出（未打包）。**方案已定，但用户明确要求先对齐需求再动固件。**

---

## 1. 设备事实（已实测验证，可直接引用）

| 项 | 值 |
|---|---|
| 型号 | **RIG-Hover（文文）**，陆吾智能 LuwuDynamics，桌面双轮自平衡机器人 |
| 开源固件 | https://github.com/LuwuDynamics/rig_omni （v3.8.0，ESP-IDF v5.5.3） |
| 主控 | ESP32-S3 QFN56 rev0.2，双核 240MHz，512KB SRAM，**8MB Octal PSRAM**，**16MB Flash** |
| 执行器 | 2× KP4012 轮毂电机（差速）+ 1× EM3 串行舵机（头部偏航 ±75°） |
| 传感器 | QMI8658C 六轴 IMU + GC0308 摄像头（240×240 RGB565）+ 1 麦 |
| 显示 | GC9A01 240×240 圆屏（28 种 EAF 表情）|
| MAC | `90:da:72:4f:9b:e0` |
| 原厂 UUID | `fadb62b7-ed37-41ad-8d99-f84e1a364a55` |
| 串口 | **COM3**（WCH CH343，VID_1A86/PID_55D3），115200 8N1 |
| 分区表 | nvs 0x9000/16K，otadata 0xD000/8K，phy 0xF000/4K，ota_0 0x20000/4M，ota_1 0x410000/4M，assets 0x800000/8M |

### 机器人控制变量（来自 `main/boards/hover/robot.h`）

```
vx, vyaw            速度指令（±100）
stable_pos, stable_yaw   位置/朝向目标
q_head, target_head_pos  头部实际角/目标角（±75）
gimbal_pitch_target      云台俯仰目标（±45）
control_mode             0=小车 1=云台
wheel1_vel/wheel2_vel    左右轮速
wheel1_x/wheel2_x        里程计
tor1, tor2               输出力矩
pid_pos/pid_vel/pid_pit  三个 PID 环（含 fpE 误差 / fpU 输出）
lqr_k[4]                 LQR 增益
```
IMU 侧（`imu.h`）：`roll, pitch, yaw, accel_x/y/z, pit_gyro, dq`

### 控制环频率（源码确认）
- `robot_control()` ≈ 250Hz（core 0）
- `imu_read_once()` 每 2ms ≈ 500Hz（core 1）
- `motor_rx()` 每 3ms ≈ 333Hz（core 1）

---

## 2. 已商定的方案（**重要，不要推翻**）

### 2.1 架构

```
PC/手机：只跑 YOLO26 做感知 → 下发视觉特征（10Hz）
                    ↓  POST /vision  二进制
机器人：本体感知 500Hz + 视觉特征融合 → CfC 决策 → 现有 LQR 环执行
```

### 2.2 CfC 的角色定位（用户核心观点）

> **"重点在于什么时候做出决策"**

CfC 不是低层控制器，是**决策器**：决定什么时候切换行为、什么时候确认手势、什么时候打断、什么时候放弃。

**网络结构：单 CfC 主干 + 三头多任务**

| 头 | 输出 | 损失 |
|---|---|---|
| 连续控制头 | `[head_yaw(±75°绝对角), vyaw(±100速度), head_pitch(±45°)]` | MSE |
| 行为模式头 | `[跟随/打招呼/待机/停止]` softmax | 交叉熵 |
| 手势确认头 | `[无/挥手/握拳/指向]` softmax | 交叉熵 |

- 规模：64–128 神经元（约 1–3 万参数）
- **CfC 决策频率 50–100Hz**（不是 500Hz，100Hz 下推理约 1ms 占 10% CPU）
- 权重导出为 C 数组放 `.rodata`（flash），机器人本地手写前向传播（~100 行 C）

### 2.3 场景：视觉跟随（v1 范围）

**做：**
- 人移动 → 机器人转向面对人
- **区分头转 vs 轮转**：误差小用头部舵机（快、无扰动）；误差大用轮子转向；**身体转动时头部反向补偿做"稳像"**
- 手势交互（由 CfC 学时序判断，非硬规则）
- 表情随交互状态变化

**v1 明确不做：**
- ❌ 前进跟随（vx）—— 桌面边缘有跌落风险
- ❌ 防跌落检测（用户说"肯定要做，但第一版先算了"）
- ❌ 主动行为（如"没人理 10 秒自己转圈"，建议留 v2）

### 2.4 训练路线

1. **先写教师控制器**（PC 上的 Python，带时序状态机：手势去抖、迟滞切换、冷却时间）
2. 用它跑 60–90 分钟，录 (状态序列 → 教师输出) 数据，约 10–16 万帧 @30Hz
3. 行为克隆训 CfC（PyTorch + `ncps` 库，`AutoNCP` wiring）
4. 导出权重 → C 头文件 → 机器人端实现
5. **基线对比**：CfC vs 教师，同一段真实轨迹

**关键**：教师控制器既是训练数据来源，也是评价基线。必须先把它写好。

### 2.5 YOLO26

- **很可能不用训练** —— COCO 预训练模型本来就会检测 `person`，先跑起来看效果
- 手势用 `yolo26n-pose`（COCO 17 关键点或手部 21 关键点）
- 若需微调：`model.export(format="tflite", imgsz=320)`；训练用 Ultralytics，参考 https://docs.ultralytics.com/guides/finetuning-guide
- 数据要**用机器人自己的摄像头采**（域差距）

---

## 2.1 `D:\smartcar` 调研结论（同构参考实现，**务必先读**）

> 2026-11-02 派子 agent 只读调研的结果。`D:\smartcar` 是智能车竞赛仓库（88k 文件 / 7GB），**真代码在 `code\`**（386 文件 / 约 10.2 万行）。硬件是阿克曼 T710 + GD32，与我们不同，**但"YOLO 特征 + CfC + 规则安全壳"的架构和我们完全同构**。

### 可直接搬的 6 个文件（价值极高）

| 文件 | 用途 |
|---|---|
| `D:\smartcar\code\lnn\extract_features.py` | YOLO 内部特征钩子，**精确层号 P3(16)/P4(19)/P5(22)/Detect.cv3[1]**，920 维特征构造方式，v8/v26 家族判定逻辑。改 letterbox 尺寸即可用 |
| `D:\smartcar\code\lnn\cfc_numpy.py` | **49 行纯 numpy CfC 前向，零依赖** —— 在 ESP32 上写 C 版本 CfC 的黄金参考 |
| `D:\smartcar\code\lnn\cfc_net.py` + `feat_adapter.py` | `CfcPolicy`（标准化→投影→CfC→head→clip，含 export_npz/onnx）、`FeatureAdapter`、`RetentionCfCCell` |
| `D:\smartcar\code\lnn\reports\cfc_memory_20261030.md`（485 行） | **CfC 记忆时标的完整实证**，与我们的时序决策最相关，必读 |
| `D:\smartcar\code\runtime\scheduler.py` + `degradation.py` | 多速率"最新值总线"调度 + 5 态降级机（阈值全调好了），直接照搬语义 |
| `D:\smartcar\code\lnn\action_shaper.py` | 动作 rate limiter + 训练时一阶差分惩罚，双轮平衡车同样需要 |

### ⚠️ 三条必须提前设计规避的坑（会改动我们的方案）

1. **`ncps` 的 CfC 记忆时标只有 ~2.2 秒**（实测自相关 lag=180 步归零）。**"手势确认跨越数秒"这个需求，CfC 自带记忆不够用** → 必须自己加自环门 `h' = u + λ(h−u)`（`RetentionCfCCell`，λ=sigmoid(ret) 逐单元，λ=0 时与父类逐位一致）或显式状态位。
2. **高维原始特征直塞时序网络会崩**。他们实测"YOLO→CfC 六连败"，修复靠 `FeatureAdapter`：逐维标准化 → `tanh` 软限幅(±3σ) → 学习线性压缩(**920→64**)。**我们不能把上千维特征直接喂 CfC。**
3. **控制循环必须固定节拍**。他们不限流时空转 177Hz，同一帧被处理 6 次，rate limiter 的 dt 假设失效（有效舵率放大 5.9 倍）。ESP32 上必须硬定时。

### 其他值得抄的工程纪律

- **训练/部署逐算子对拍 < 1e-6**（torch ↔ ONNX ↔ numpy 三方），`test_deploy_parity.py` 是模板
- **每条轨迹隐状态必须清零**，忘了清零实验直接作废
- **域外输入守卫**：观测超出训练包络时不报错、静默跑飞，最难查
- **别把指令当反馈**：他们 GD32 无遥测，`/status.speed` 只是下发指令回读 —— 我们的 `/status` 是真遥测，这点我们更强
- **上界类闸门只能判绝对值不能判可达性**：一次误拒后观测永久冻结，比原 bug 更坏

### smartcar 没有的东西（我们要从零做）

- **手势识别：完全空白**（全仓无 pose/keypoint/mediapipe）
- 无 LQR/MPC（我们是现成的 LQR，不用重写）
- 无行为树（只有 FSM）

---

## 3. 已完成的工作与产物路径

| 产物 | 路径 | 状态 |
|---|---|---|
| **协议规格（权威）** | `G:\robot_app\PROTOCOL.md` | ✅ 完整，写客户端必读 |
| Android App 源码 | `G:\robot_app\android\` | ✅ 76 文件，未打包 |
| └ App 文档 | `...\README.md`、`DESIGN.md`、`model_schema.md` | ✅ |
| 打补丁的固件源码 | `D:\robot_work\rig_omni\` | ⚠️ 补丁已写，**编译未成功** |
| ESP-IDF 工具链 | `G:\esp\esp-idf` + `G:\esp\.espressif` | ✅ 装好（5.6GB）|
| 原始 NVS 备份 | `C:\Users\sans\AppData\Local\Temp\robot_probe\nvs_orig.bin` | ✅ |
| 原始固件转储 | `...\robot_probe\flash_0_6m.bin`（0x0–0x600000，含完整 ota_0）| ✅ |
| 可恢复的 NVS 镜像 | `...\robot_probe\nvs_final3.bin` | ✅ |
| DHCP 辅助脚本 | `...\robot_probe\dhcpd_long.py` | 可能已停 |
| Android 子 agent 报告 | 见本对话历史 | ✅ |

### 已打的固件补丁（4 个文件）

1. `main/boards/common/wifi_board.cc` — 连上 WiFi 立即 `esp_wifi_set_ps(WIFI_PS_NONE)`（原厂只在云端激活后关省电，导致 RTT 100ms）
2. `main/boards/common/esp32_camera.{h,cc}` — 新增 `CaptureJpeg()`（抓帧 + 字节交换 + `image_to_jpeg`）
3. `main/boards/hover/hover_debug_server.cc` — 新增 `/status`（28 维遥测）、`/cmd`（含 800ms 看门狗）、`/capture`、POST `/vision` 二进制
4. `sdkconfig.defaults.esp32s3` — `BOARD_TYPE_PUPPY` → `BOARD_TYPE_HOVER`

### 机器人当前 NVS 内容（已恢复）

`board.uuid`（原厂）、`wifi`（ssid=`qasd`，测试热点）、`mqtt`（完整云配置，凭据在此略）、`websocket.url`、`power_mode=performance`

> 云端配置已恢复，但因热点无外网，云端语音仍不可用。原始凭据可从 `nvs_orig.bin` 用自写的 NVS 解析脚本恢复（格式：entry CRC = `crc32_le(init=0, [ns|type|span|chunk]+key+data) ^ 0xFFFFFFFF`；位图 2bit/slot，`10`=已写、`11`=空）。

---

## 4. 实测数据（可直接引用）

| 指标 | 值 |
|---|---|
| HTTP keep-alive 吞吐 | **553 KB/s（4.5 Mbps）**，107 请求/秒 |
| RTT（省电关闭） | **~2ms** |
| RTT（省电开启，原厂状态） | 24ms 平均 / 110ms 最大，抖动 40ms |
| **短连接** | ❌ **会超时**，必须长连接 |
| JPEG 单帧 | 10–25 KB |
| `/status` 响应 | ~400 字节 |
| 启动到联网 | 6.8 秒（其中 WiFi 扫描 4.7s） |
| 运行时内部 SRAM 空闲 | 48–63 KB（**紧张，大缓冲要放 PSRAM**）|

---

## 5. 踩过的坑（**不要再踩**）

1. **ESP32 不支持 WPA2/WPA3 混合模式**。原厂扫到 `Authmode: 7` 时认证失败重试 3 次。必须用**纯 WPA2**（Windows 热点需设 `TetheringWiFiAuthenticationKind.Wpa2`）。
2. **机器人只支持 2.4GHz**。热点默认 5GHz 时机器人扫不到。
3. **NVS 很脆弱**：固件写 NVS 时复位会损坏 → `nvs_flash_init()` 失败 → `main.cc` 无条件 `nvs_flash_erase()`，UUID 和 WiFi 配置全丢。**尽量少用 RTS 复位**。
4. **Windows ICS 的 DHCP 服务不工作**（`Get-NetUDPEndpoint -LocalPort 67` 为空）→ 自写了 DHCP 服务器才让机器人拿到 IP。注意：单 socket 绑 `0.0.0.0:67` + 定向广播 `192.168.137.255` 才走得通。
5. **ESP-IDF 安装时 SSL 证书间歇性失败**（公司/运营商 MITM）→ 最终解法：用 `curl.exe` 手动下载所有工具包到 `.espressif\dist\`，并在 `tools.json` 的 `win64` 字典里按**键名**（就是期望的 dist 文件名）命名。
6. **残留的 `sdkconfig` 会覆盖 `sdkconfig.defaults.*`** → 编译出 PUPPY 板型 + 2MB flash。**`idf.py set-target` 前先删掉 sdkconfig**。
7. **PyPI 太慢** → 用清华镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`。
8. **资源打包脚本需要 numpy + pillow**，装到 IDF 的 python env 里（`G:\esp\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe`）。

---

## 6. 编译环境使用方式

```powershell
$env:IDF_TOOLS_PATH = 'G:\esp\.espressif'
$env:PIP_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
cmd /c "call G:\esp\esp-idf\export.bat && cd /d D:\robot_work\rig_omni && idf.py build"
```

**分工**：`esp-idf` 在 G 盘（5.6GB，只读）；项目 + build 在 **D 盘 SSD**（用户要求：重的东西放 G，I/O 密集的放 D）。

**上次编译停在**：`[184/2385]`，报 `ModuleNotFoundError: No module named 'numpy'`（**已装好 numpy 2.4.6 + pillow 13.0.0，下次应该能过**）。

---

## 7. 工程决策（已由我拍板，用户授权「工程问题自己决定」）

> 全部决策 + 理由 + 完整施工图见 **`D:\robot_work\TECH-PLAN-v1.md`**（v1.0）。这里只列结论。

| # | 决策 | 结论 |
|---|---|---|
| D1 | 跨秒级记忆 | 加 `RetentionCfCCell`，但**不指望它记事件**。关键洞察：「手抬起」每帧都可观测，真正需要记忆的只有「刚响应过」→ 用显式不应期（REFRACTORY 态） |
| D2 | 教师控制器 | **只写一版保守的**，阈值全外置成 YAML |
| D3 | 主动行为 | **不做**，列为后续独立课题（见方案 §16） |
| D4 | 特征维度 | 920 → **64**（PC 端 FeatureAdapter 压缩） |
| D5 | CfC 规模 | **64 隐藏单元**（≈2.5 万 MAC/步，约 2.5ms，50Hz 下占 12.5% CPU） |
| D6 | 执行接口 | **不改 LQR**；`cfc_vyaw` 积分进 `stable_yaw`；头走 `target_head_pos` |
| D7 | 通信 | **新增 `/feat`**，`/vision` 原样保留 |
| D8 | 转向阻尼 | **yaw 环加 D 项**（现状纯 P 控双积分 = 临界稳定） |
| **D9** | **固件形态** | **做成"通用策略运行时"：权重/参数全走 WiFi 热更新，第一次烧录后永不再编译**。用户原话："你要是第一版只做一个小玩具，编译的时间都花不少了，不如一开始做完全一点" |
| **D10** | **前进通道** | **vx 纳入 v1**（vx 本就进 LQR，加它几乎零成本，但机器人从"原地转"变成"能跟着走"） |
| **D11** | **防跌落** | **纳入，与 vx 捆绑**。无新硬件，用摄像头同一帧做桌沿检测（C1 几何法优先，A6 阶段独立可砍项） |
| **D12** | **双模型槽** | 固件常驻 2 槽，运行时 `/mode` 切换，为了 CfC vs 基线 A/B |

> **v1.1 关键新增**：§7.1b **双模设计**（摄像头在头上/机身两种都实现，`/cfg` 里 `img_frame` 一个参数切换 → 消灭"测完要重烧"的风险）；§7.6 **抗积分饱和**（参考值相对实际状态钳位，不做会"松手猛冲"）。
> **里程碑**：A 阶段（不用机器人）5.5d + B 阶段（唯一一次烧录）0.8d + C 阶段（全走 WiFi 迭代）4.5d = **10.8d / 1 次烧录**（原方案 9.5d / 3 次烧录）。

### 7.1 从固件源码新挖出的关键事实（写进方案 §2 了，这里复述要点）

- **控制环 500Hz**（`vTaskDelay(pdMS_TO_TICKS(2))`）
- **头部姿态已进平衡环**：`pitch_ref = imu_zero*cos(q_head)` → 动头会扰动平衡
- **`vyaw` 是死变量**：`robot.cc` 里只赋值从不读取。**`/cmd?vyaw=` 无效**
- **唯一转向通道是 `stable_yaw`**（航向位置目标）
- **`stable_yaw` 会在 3s 无指令后自动冻结到当前朝向** → 我们下发时必须每 <3s 刷 `yaw_ctrl_time`
- **头舵机实际范围 ±45°**（`1500 - target_head_pos*10` → 1050~1950µs），**不是之前记的 ±75°**
- **IMU 在头部**（证据：`cos(q_head)` 补偿、`stable_yaw = yaw - q_head`）→ 机身朝向 = `imu.yaw - head.q`，**用户要的「头转 vs 机身转」区分固件层已实现**
- **已白送的安全机制**：摔倒检测（pitch≥60°/roll≥40° 持续 500ms）、站立检测（<7° 持续 1500ms）、抱起检测（轮速>350 连续 8 拍）、单击摇头/双击转圈

### 7.2 通道分离的闭式解（方案 §7.1，核心公式）

```
ε_body = θ_img + q_head      // 机身要转的角度；ψ_gaze 恰好消掉
```
- `|ε|<5°`：只动头，轮子不动（用户核心要求）
- `5°~25°`：头对准 ×0.7，机身慢跟（留 30% 误差作驱动源）
- `>25°`：机身快转，头反向补偿保持注视
- `|head_des|` 接近 ±42° 限位：强制机身转

### 7.3 M0 硬门槛（方案 §15，8 个必须先测的）

**最关键的是 V1：摄像头装在头上还是固定在机身上。** 测法：读 `/capture` → 让头发动 `/cmd?head=40` → 再 `/capture`，看画面是否平移。这条决定 §7.1 公式的写法。
其余：FOV 标定、头转多少度破坏平衡、有没有 gyro Z、`/capture` 耗时、端到端延迟、CfC 实测耗时、头角度标定。

---

## 8. 下一步（建议顺序）

| 步骤 | 内容 | 前置 |
|---|---|---|
| ~~A~~ | ~~调研 `D:\smartcar`~~ → **已完成，结论见 §2.1** | ✅ |
| ~~B~~ | ~~用户拍板~~ → **已完成，决策见 §7** | ✅ |
| ~~C~~ | ~~写技术方案~~ → **已完成：`D:\robot_work\TECH-PLAN-v1.md`** | ✅ |
| **D** | **M0：跑 8 个前置测试**（需先烧固件才能测 V1） | 用户批准 |
| **E** | M1→M7 按方案 §13 里程碑执行 | D |

> ⚠️ **用户明确说过："先别动固件，我们先把方案/需求聊清楚"。方案已写完，等用户点头再编译烧录。**

> ⚠️ **用户明确说过："先别动固件，我们先把方案/需求聊清楚"。在 B/C 完成前不要编译烧录。**

---

## 9. 用户偏好与沟通要点

- 是学生，老师给的课题，目标是"跑通 + 摸清能力边界 + 做出具体应用场景"
- **很在意效率**，对我反复编译失败、把 D 盘搞乱表达过不满
- 要求：大文件放 G 盘（移动硬盘），轻的放 D 盘（SSD），**及时清理无用文件**
- **很在意 App 的 UI 设计**，要求"抄现成的高级设计，不要自己发明"，动效要灵动
- 目前**不要打包 APK**
- 希望我**先对齐需求再动手**，不要自作主张推进

---

## 10. 建议下一轮加载的技能

- `mobile-android-design` / `android-material3-design-system` / `android-compose-foundations` —— 若继续做 App UI
- `handoff` —— 再次压缩上下文时

---

## 11. 安全与还原

- 原始固件完整备份在 `flash_0_6m.bin`（ota_0 完整），可用 esptool 写回 `0x20000` 还原原厂
- 原始 NVS 备份 `nvs_orig.bin`，写回 `0x9000` 还原出厂配置
- 机器人当前**未被改坏**，仍是原厂固件在跑
