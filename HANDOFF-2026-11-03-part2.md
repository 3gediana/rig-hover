# 交接文档 · 2026-11-03 第二部分（详细版）

> 这份是**增量**交接：第一份 `HANDOFF-2026-11-03.md` 仍然有效，写的是固件、崩溃修复、
> 编码事故那一晚。**本文档只写那之后发生的事情。**
>
> 本文档刻意写得很细，包含大量本可以在别处查到、但"查一次要重跑实验"的数字。
> 规格类内容仍然引用其它文件，不整段复制。

---

## 目录

0. [三条铁律](#0-三条铁律违反会出事按严重程度排)
1. [项目全景](#1-项目全景)
2. [硬件与固件事实](#2-硬件与固件事实你不需要再查一遍)
3. [当前卡点](#3-当前卡点机器人不通)
4. [本轮工作详解](#4-本轮工作详解)
5. [用户定位与方向纠正历史](#5-用户定位与方向纠正历史重要)
6. [逐个文件说明](#6-逐个文件说明)
7. [测试清单](#7-测试清单每个套件到底测什么)
8. [已验证 vs 未验证](#8-已验证-vs-未验证)
9. [下一步](#9-下一步含具体命令)
10. [踩坑记录](#10-踩坑记录详细)
11. [开放问题 / 待决策](#11-开放问题--待决策)
12. [命令速查](#12-命令速查)
13. [文件索引](#13-文件索引)

---

## 0. 三条铁律（违反会出事，按严重程度排）

### 铁律 1：含非 ASCII 的文件只能用 `write`/`edit` 工具改

**绝不能用 PowerShell 的 `Set-Content` / `Out-File` / `>` 重定向。**

原因：`Get-Content -Raw` 按系统 GBK 解码 UTF-8。当中文字符后面跟着一个 `< 0x40`
的字节（换行、引号、空格都会触发）时，该字节会被当成非法双字节的尾字节**吞掉**，
输出成 `?`。**丢字节是不可逆的。**

这个项目已经因此损坏过 4 个文件：
- `pc/train/obs.py`（7 处注释吞代码 + 5 处引号 + 3 处名字条目合并）——已修
- `pc/train/session_from_robot.py`（末尾 print 块）——已修
- `rig_omni/main/boards/hover/hover_board.cc`（14 处引号）——已修
- `RESULTS.md`（1859 处中文变乱码 + BOM）——**无法修复，已重写**

允许用 `pwsh` 跑 Python 做**读取和统计**。写中文一律走工具。

### 铁律 2：没在硬件上验证过的固件改动不要提交

这个项目已经**两次**因为"编译通过、逻辑看着对、上机就炸"而开机崩溃循环：

- `RESULTS.md` §5 的 #9：4 KB `sys_evt` 栈上跑 httpd 启动 → 栈溢出踩坏 TCB。
  **更糟的是它有一个"建任务失败就内联调用"的降级路径，而内联调用正是崩溃路径本身。**
- `RESULTS.md` §5 的 #14：`cfc_model_init()` 里一行 `cfc_q8_simd_enabled()` 跑了完整
  SIMD 数值扫描 → 撑爆 `dbg_boot` 栈。**加栈无效**，因为问题是那行调用本身。

**所以这一轮我唯一主动没做的核心工作，就是那 100 行独立 TCP 流式任务**——
机器人现在不通，栈大小和长连接行为无法验证。见 §4.7。

### 铁律 3：不要拿当前模型的性能做工程补丁

用户 2026-11-03 明确纠正过。见 §5。**当前模型是临时占位。**

---

## 1. 项目全景

**RIG-Hover（文文）**：ESP32-S3 桌面自平衡机器人。摄像头装在头部（随头转），
头部 1 个舵机，两个轮子自平衡。

五个交付项与当前状态：

| # | 交付项 | 状态 |
|---|---|---|
| 1 | 固件 CfC 运行时 + WiFi 热更新 + 仲裁器 + 安全壳 + 全部端点 + 遥测 | ✅ 编译、烧录、逐项验证过 |
| 2 | 可用的 CfC（三档 + 手势确认 + 不应期），离线对教师验证、量化指标 | ✅ 有指标，**但模型是占位**（§5） |
| 3 | 手势识别（公开数据集 + YOLO 姿态 + 真机阈值标定） | ✅ 宏 F1 0.805，`pointing` 弱（F1 0.574，结构原因） |
| 4 | 分层决策中枢（本地 → PC/手机 agent → 云端 LLM） | ✅ 90/90 + 58/58 测试，真实副作用 |
| 5 | 技术报告 + 可复现验收证据 | ✅ 23/23 门禁，`FINAL-VERIFICATION.txt` 带时间戳 |

**2026-11-03 上半场新增的第六件事**：用户提出新场景，已拆解成可实现的规格。
见 `SCENARIO-v2.md` 与 `CFC-V2-DESIGN.md`。

---

## 2. 硬件与固件事实（你不需要再查一遍）

### 2.1 硬件

| 项 | 值 | 来源 |
|---|---|---|
| 主控 | ESP32-S3 QFN56 rev0.2，双核 240 MHz | esptool |
| MAC | `90:da:72:4f:9b:e0` | 8:06 烧录日志 |
| 内存 | 512 KB SRAM + 8 MB Octal PSRAM | sdkconfig |
| Flash | 16 MB，应用分区 4032 KB | 分区表 |
| 串口 | `COM3` = CH343 (`USB\VID_1A86&PID_55D3`) | 设备管理器 |
| 摄像头 | **GC0308** 240×240，内嵌头部随头转，`FOV_H=60°` | `esp32_camera.cc:29` |
| 舵机 | **1 个**（`board_config.h:5`：`1-servo + 2-wheel hovercraft`） | 源码 |
| 控制环 | **6 ms（167 Hz）**，不是 500 Hz | 3× `vTaskDelay(2)` + `FREERTOS_HZ=1000` |

### 2.2 关键源码位置

| 事实 | 位置 |
|---|---|
| 摄像头配置 | `rig_omni/main/boards/arm_board.cc:172-180` |
| `PIXFORMAT_RGB565` / `FRAMESIZE_240X240` / `jpeg_quality=12` / `fb_count=2` / `GRAB_LATEST` | 同上 |
| `CaptureJpeg` 实现 | `rig_omni/main/boards/common/esp32_camera.cc:59` |
| RGB565 字节交换循环（115,200 B） | `esp32_camera.cc:100` |
| 软件 JPEG 编码调用 | `esp32_camera.cc:132` → `image_to_jpeg()` |
| `PIXFORMAT_GRAYSCALE` 分支**已存在** | `esp32_camera.cc:114` → `V4L2_PIX_FMT_GREY` |
| `WIFI_PS_NONE` **已设** | `wifi_board.cc:119` |
| `/capture` handler | `hover_debug_server.cc:457` |
| `max_uri_handlers = 20` | `hover_debug_server.cc:1091` |
| 固件日志级别 WARN | sdkconfig `CONFIG_LOG_DEFAULT_LEVEL=2` |

### 2.3 关键 sdkconfig（会影响性能的那些）

```
CONFIG_ESP_CONSOLE_UART_DEFAULT=y     # UART0
CONFIG_ESP_CONSOLE_UART_BAUDRATE=115200
CONFIG_LOG_DEFAULT_LEVEL=2            # WARN —— 正常运行不打印任何东西
CONFIG_BOOTLOADER_LOG_LEVEL_NONE=y    # 二级引导器也不打印

CONFIG_LWIP_TCP_SND_BUF_DEFAULT=5760  # ← 小于一帧 9,277 B，每帧多一个窗口轮次
CONFIG_LWIP_TCP_WND_DEFAULT=5760      # ← 同上
CONFIG_LWIP_TCP_MSS=1440
CONFIG_LWIP_TCP_TMR_INTERVAL=250
CONFIG_LWIP_TCP_MSL=5000              # 已从 60000 调小（修 socket 耗尽）

CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM=3
CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM=6
CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM=32

CONFIG_CAMERA_CORE0=y                 # 摄像头/编码在核 0，lwIP 无亲和 → 可能抢 CPU
CONFIG_CAMERA_TASK_STACK_SIZE=4096
```

### 2.4 blob 二进制格式（改模型参数必须知道）

头部 2256 字节，字段偏移（`pc/train/export_blob.py::HDR_FIELDS`）：

| 字段 | 偏移 | 长度 |
|---|---|---|
| `magic` | 0 | 4 |
| `ver` | 4 | 2 |
| `flags` | 6 | 2（bit0 proj / bit1 ret / bit2 int8 / bit3 layer_prec） |
| `input_dim` | 8 | 2 |
| `cell_in_dim` | 10 | 2 |
| `hidden_dim` | 12 | 2 |
| `backbone_units` | 14 | 2 |
| `n_heads` | 16 | 2 |
| `head_dim[8]` | 20 | 16 |
| `head_act[8]` | 36 | 16 |
| `layer_off[16]` | 52 | 64 |
| `layer_len[16]` | 116 | 64 |
| **`input_mean`** | **180** | 1024（256 × f32） |
| **`input_std`** | **1204** | 1024 |
| `clip_sigma` | 2228 | 4 |
| `ts` | 2232 | 4 |
| **`crc32`** | **2236** | 4（计算时把这 4 字节当 0） |
| `layer_prec[16]` | 2240 | 16 |

其它常量：
- `CFC_SLOT_MAX_BYTES = 262144`（**256 KB，这是模型容量的真正上限**，不是算力）
- `CFC_MAX_INPUT_DIM = 256`（`input_mean` 表有 256 个 float）
- `blob_crc32` = zlib/IEEE-802.3，Python `zlib.crc32` 与固件 `cfc_crc32()` 逐位一致

**改 `input_mean/std` 只需要改偏移 180/1204 的数组 + 重算偏移 2236 的 CRC，
权重一个字节都不用动。** 工具：`pc/train/calibrate_inputs.py`。

### 2.5 观测 v1 的 106 维布局（`pc/train/obs.py::OBS_NAMES`）

```
 0–63   feat_00 .. feat_63      YOLO 内部特征（FeatureAdapter 548→64）
64      theta_img               人的方位角（度）
65      person_w                bbox 宽 / 图宽
66      person_h                bbox 高 / 图高  ← 一直是距离代理量
67      dist_est                低通后的距离估计
68      person_conf
69      seen
70–71   hand_up_l / hand_up_r
72–73   wave_sin / wave_cos
74      wave_amp                ← 曾经出过大 bug，见 RESULTS.md §5 #12
75      gesture_conf
76–77   reserved（固件写死 0，std 恒为 0，导出时 std 要设下限）
78–81   roll / pitch / pit_gyro / dq
82–85   yaw / q_head / target_head_pos / head_err
86–89   wheel1_vel / wheel2_vel / wheel_dvel / wheel_x
90–93   stable_pos / pos_err / body_yaw / yaw_err
94      robot_state
95      vx
96      vyaw
97      control_mode
98      feat_age_s
99      gaze_danger
100–103 prev_head_n / prev_vyaw_n / prev_mode_n / prev_vx_n
104–105 clk_sin / clk_cos
```

---

## 3. 当前卡点：机器人不通

### 3.1 证据（2026-11-03 10:00–13:00 多次复测）

| 检查 | 结果 |
|---|---|
| `COM3` 枚举 | ✅ CH343 在，`Status=OK` |
| 打开串口 | ✅ 能打开（DTR/RTS 都设 false，避免触发复位） |
| 监听串口 | **0 字节**（6 秒 / 32 秒都试过） |
| 复位脉冲 | **四种 DTR/RTS 极性全试过，0 字节** |
| `esptool 4.12`（8:06 烧录成功那个版本）`chip_id` | ❌ `Failed to connect: No serial data received` |
| `esptool 5.6`（smartcar venv）`chip_id` | ❌ 同上 |
| `ping 192.168.137.50` | ❌ 100% loss |
| 扫 `192.168.137.0/24` 的 TCP:80 | **0 个设备** |
| ARP 表 | `192.168.137.50 → 00-00-00-00-00-00`（请求无响应） |
| 热点适配器 | ✅ `192.168.137.1` 存在 |
| `SharedAccess` / `icssvc` / `WlanSvc` | ✅ 全部 Running |

### 3.2 结论与需要用户做的事

**主控没在跑。** CH343 有电（所以 COM3 能枚举），但 ESP32 本身不响应。

物理原因只可能是：**电源开关关着，或电池放空**。这类板子的 USB 未必给主控供电
（可能只给串口桥）。

用户说"我连上 usb 后就走了，是开机了的"，但所有软件侧证据都指向主控无响应。

**需要用户确认的唯一一件事**：屏幕亮不亮、开机有没有响声。

### 3.3 ⚠️ 一个被误读过的信号

固件是 `CONFIG_LOG_DEFAULT_LEVEL=2`（WARN）。
`hover_board.cc:592` 有 `ESP_LOGW(TAG, "WiFi IP: %s", ...)`，
`hover_board.cc:606/611/614` 有几条 `ESP_LOGW(TAG, "dbg_boot: internal_free=...")`。

**这些只在开机那一刻打印。机器人正常运行期间一个字都不输出。**

所以此前文档里"串口静默 / 0 reboots"被当成异常，**是误读**。
判活只能靠 `GET /status` 返回 200。

### 3.4 后台 watcher

```powershell
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\tools\robot_watch.py `
  --ip 192.168.137.50 --minutes 240 --period 45
```

它每 45 秒探一次 TCP:80，一旦通了就打 `/status` 并打印 JSON。

---

## 4. 本轮工作详解

### 4.1 发现并量化了"模型在真机上比什么都不做还差"

**背景**：之前 16 个验收套件过了 15 个，离线留出集 head MAE 0.86°、教师 ACK 召回
75.9%。但这**所有评估会话都来自训练分布，没有一帧来自机器人自己的摄像头**。

**做了什么**：用 `data_real/`（5 段 × 120 帧）和 `data_realtest/`（115 帧）——
这些是 `session_from_robot.py` 从机器人取流时录的，**观测是真机的，标签是教师在那批
帧上打的**（`head_deg` / `vyaw_dps` / `mode` 都在 npz 里）。

**结果**：

| 条件 | head MAE | vyaw MAE | 模式一致率 | 教师 ACK 召回 |
|---|---|---|---|---|
| 出厂 `m8.blob` | **5.63°** | **10.58** | 96.5% | **0/24** |
| 什么都不做（全零） | **1.61°** | **0.80** | 0.1% | — |

**策略比全零基线差 3.5 倍，24 个教师 ACK 帧一个都没检出。**

**根因**：`input_mean/std` 在合成数据上拟合，真机特征分布偏 **2.72σ**、尺度差 **3.28×**，
归一化后整段饱和到 ±4。

**注意**：全零基线在这个数据上是**严肃的竞争者**——教师对"人基本在正前方"的场景
大多命令接近零的转头角，一个什么都不做的策略在 head MAE 上并不难看。
**这个数据上真正有分辨力的指标是 ACK 召回和模式一致率。**

完整推导见 `TECHNICAL-REPORT.md` §4.3c。**该节顶部有适用范围声明，引用前先读。**

### 4.2 特征块的性质（这里我犯过错，被用户纠正两次）

**第一次错误**：我写了"64 维激活特征是净负债、携带不了可迁移信息"。

**第二次被纠正后做的实验**（`pc/bench/_feat_info.py`）：

| 预测目标 | 用特征块（64 维） | 用其余通道（42 维） |
|---|---|---|
| `theta_img` **同分布**（训练集 → 同生成器留出集） | **R² = +0.846** | 1.000（它本身就在观测里） |
| `theta_img` **跨域到真机帧** | R² = **−83.3** | 1.000 |

分布本身逐维对比：

| | \|Δμ\|/σ 中位 | σ 比中位 |
|---|---|---|
| 训练集 vs **同生成器留出集** | **0.061** | **0.979** |
| 训练集 vs **真机帧** | **2.720** | **3.280** |

**结论**：
1. 激活特征**有信息**——同分布下线性回归就能读到 R²=0.846
2. 训练集与留出集**确实同分布**（0.061σ），这条链本身没问题
3. 故障 **100% 出在"训练取的视觉域 ≠ 部署的视觉域"**
4. **为什么它在这份模型上用不上**：教师的三个输出（`head_deg`/`vyaw_dps`/`mode`）
   全部由观测里已有的 `theta_img`、`person_h`、手势基元算出。策略要做的是**几何，不是视觉**。
   一个已经拿到方位角的网络，从特征块里再读一遍方位角没有增量。
5. **准确的表述**：特征块携带真实信息，但**当前这个标签函数让这份信息冗余**；
   而把它硬塞进部署域又变成饱和常数，于是从"冗余"变成"有害"。

**所以"删掉特征块"只对合成数据训出来的模型成立，不是普遍结论。** 见 §5。

### 4.3 场景 v2 的信号层（`pc/perception/scene_signals.py`）

用户提的场景需要三样东西，全部从**已有的 COCO-17 关键点**导出，**不新增任何模型**：

| 信号 | 做法 | 为什么这么做 |
|---|---|---|
| `theta_face` | 鼻+双眼的**置信度加权中心**的方位角 | **不用检测框中心**——框中心随人体姿态漂移 |
| `face_scale` | 瞳距（像素）/ 图高 | 人脸大小的线性代理，也就是距离代理 |
| `theta_fist` | 腕关节方位角 | |
| `wrist_y_norm` | (腕 y − 肩中点 y) / 肩髋距 | **与距离无关**，否则同一个动作在 1 m 和 3 m 处幅度差三倍，阈值没法统一。同 `wave_amp` 按肩宽归一化的思路 |
| 主导手 | fist 概率 ×2 + 关键点置信度 | |

**敲击检测器 `KnockDetector`**：腕部归一化高度的"下探—回弹"双峰判定。

参数（`KnockParams`，**全部未在真机标定**）：

```
y_norm_min  = 0.55   # 低于它说明手没放到桌面高度
depth       = 0.18   # 一次下探的有效深度
rebound     = 0.10   # 回弹幅度
window_s    = 1.50   # 两次敲击的最大间隔
min_gap_s   = 0.12   # 最小间隔，防止一次抖动算两次
refractory_s= 1.20   # 触发后不应期
max_dt_s    = 0.35   # 超过就重置峰谷（断流保护）
```

**设计上它只吃几何量**，"是不是拳头"由手势分类器给，两者在下游决策层相与。
这样检测器本身可以脱离 GPU 单测。

**24 项单测**（`test_scene_signals.py`），其中几条是刻意设计的坑：
- `wrist_y_norm` 在 1 米和 3 米处**数值相同**（距离不变性）
- 手停在桌面高度不下探 → 不触发
- 三连敲 → **只触发一次**（不应期）
- 手在胸前摆动 → 不触发
- 断流后仍能正常判定，且**跨断流的两次敲击不配对**

### 4.4 场景 v2 的决策状态机（`pc/perception/scenario_v2.py`）

**注意：它的定位是教师，不是运行时策略。** 见 `CFC-V2-DESIGN.md`。

```
IDLE --face_scale∈[min,max]--> GAZE_FACE
GAZE_FACE --fist_prob≥0.55--> GAZE_FIST
GAZE_FIST --松手 hold 0.4s--> GAZE_FACE
GAZE_FIST --knock_fired--> APPROACH（目标锁定）
APPROACH --到达/超时/人太近--> DWELL
DWELL --2.5s--> GAZE_FACE
```

**用户指定的三个细节，全部落进代码并有断言：**

**(a) 不许斜眼。** 相机装在头上，**头对准 = 目标在画面中心**。所以：

```
|eps| ≤ 43°(head_max - margin)：head = eps（精确对准），vyaw = 0
|eps| > 43°：头限位 45°，机身把超出的角度转掉
```

这与教师控制器的 `_allocate` **故意不同**：教师在中误差档给 `head = eps*0.7`、
大误差档给 `head = eps - 25°`，**那是为了让机身有"驱动源"，但结果就是目标偏离画面中心
= 斜眼看人**。用户明确不要这个。

**头没有死区**——死区意味着允许一个永久偏角。抖动交给固件 `head_lpf` 滤。

量化指标 `off_center_deg = eps - head_cmd`，测试断言它在不饱和时恒为 **0**。

**(b) 头优先，够不到才动轮子。** `|eps| ≤ 43°` 时 `vyaw == 0` 有断言。

**(c) 敲击位置要记住、要盯着走过去。**

- 敲击那一帧锁定成 `TargetLock`（方位 + 距离明细）
- 手还在视野里 → **每帧用观测修正方位**（"盯着那个位置走过去"）
- 手没了 → 保持最后的值，靠里程计走完剩余距离（"哪怕拳头离开"）

**参数（`V2Params`，全部未经真机标定）：**

```
face_scale_min   = 0.045    # 低于它太远，不盯
face_scale_max   = 0.230    # 高于它太近（整屏是脸），不盯
head_max_deg     = 45.0
head_deadband_deg= 2.0      # 只用于报告"已足够居中"，不参与指令
head_limit_margin= 2.0
k_fast           = 12.0
vyaw_max_dps     = 90.0
vyaw_floor_dps   = 12.0
fist_prob_on     = 0.55
fist_prob_off    = 0.30
fist_lost_s      = 0.40
vx_max           = 0.15
turn_tol_deg     = 6.0      # 超过它先原地转再走
k_turn           = 8.0
k_v              = 1.2
arrive_tol_m     = 0.06
overshoot_margin_m = 0.12   # ← 见 §4.9
approach_timeout_s = 12.0
stop_if_face_scale = 0.260
dwell_s          = 2.5
dist_k           = 0.84     # d ≈ dist_k / person_h，**猜的，必须标定**
dist_person_h_min= 0.06
```

**52 项单测**（`test_scenario_v2.py`）。

### 4.5 最近的人选择（`run_perception.py::_pick_person`）

用户要求："谁靠着近，你就看谁就好了。"

**原来的实现按检测框面积排名。** 有一个具体的失效场景：**最近的人常常贴着画面边缘**，
框被裁掉一截，面积反而小于稍远但完整的人 → 机器人会去看那个更远的。

**改成按检测框高度排名**（`person_h` 本身就是距离代理量，且不受水平裁剪影响），
面积和置信度只作平手时的次序。

**10 项单测**（`test_pick_person.py`），其中一条专门造出面积法会选错的布局来对照。

### 4.6 观测 v2（`pc/train/obs_v2.py`）

106 → **116 维**，只在尾部追加 10 维，**前 106 维原样复用** `obs.assemble()`。
（自检里有一条断言：`前 106 维与 v1 逐元素相同，max diff = 0`）

| 维 | 名称 | 说明 |
|---|---|---|
| 106 | `theta_face` | 度 |
| 107 | `face_scale` | 瞳距/图高 |
| 108 | `face_valid` | 0/1 |
| 109 | `theta_fist` | 度 |
| 110 | `wrist_y_norm` | 与距离无关 |
| 111 | `hand_valid` | 0/1 |
| 112 | `fist_prob` | 0–1 |
| 113 | `knock_fired` | **脉冲**，每次"敲两下"成立只有一帧为 1 |
| 114 | `knock_bearing` | **v2a 给真实值；v2b 恒 0** |
| 115 | `knock_dist` | **v2a 给真实值；v2b 恒 0** |

**v2a / v2b 是"CfC 有没有记忆"的对照开关**：

- `v2a`：由 `TargetLatcher` 在敲击帧锁存并持续提供 → "去哪"由外部给，只学怎么走
- `v2b`：114/115 恒 0 → **CfC 必须自己在敲击帧把 `theta_fist` + `dist_est` 存进隐状态**

**v2b 才是用户要的"让 CfC 承担更多"。** 先训 v2a 作基线，再训 v2b。

输出头：`mode` 从 4 类扩到 **6 类**：`0 IDLE / 1 TRACK / 2 ACK / 3 GAZE_FIST /
4 APPROACH / 5 DWELL`。其余四个头不动（`head_yaw`/`vyaw`/`vx` 是连续量，
CfC 可以直接学会"接近时减速"，不需要额外的距离输出头）。

**16 项自检**（`obs_v2.py` 直接跑就是自检）。

### 4.7 延迟：测量结果推翻了我自己的假设

`pc/bench/_latency_pipeline.py` 实测**整条**感知链（30 张真实照片，跑完整
`PerceptionPipeline.__call__`，读它自己埋的 `t_forward_ms` / `t_adapter_ms` /
`t_gesture_ms` / `t_total_ms`）：

| 环节 | 中位 | 占比 |
|---|---|---|
| PC 端 YOLO 检测前向 | **7.29 ms** | 79% |
| PC 端特征适配 548→64 | 0.54 ms | 6% |
| PC 端手势分类器 + 事件机 | 1.30 ms | 14% |
| **PC 端合计** | **9.20 ms（109 fps）** | 15% |
| **机器人取流（每帧一次 HTTP）** | **51.6 ms** | **85%** |

**我犯的错**：一度以为 `RESULTS.md` §9 里那个 `feat 56.3 ms` 是特征抽取开销，
想"反正不用特征了，砍掉能省一大截"。**实测：检测+特征 hook 7.26 ms vs 纯检测
7.04 ms，只差 0.2 ms。** 同一次前向顺手就把特征抓下来了。

**真正的瓶颈是结构，不是算力也不是带宽**：

报告里有三条实验能钉死这一点（`RESULTS.md` §9）：
1. ICMP ping **2 ms**、TCP 握手 **0.5 ms** → WiFi 链路本身很快
2. 把机器人计算从 50 Hz 压到 5 Hz，ttfb 不变（49.8 → 60.1 ms）→ 不是 CPU 抢占
3. **并发压测**：1/2/3/4 个并发客户端，吞吐全卡在 **~16 请求/秒**，而单请求 p50
   随并发**线性增长**（65 / 112 / 173 / 232 ms）→ **httpd 是单 worker 完全串行的**

再加上 `/capture` 那 51.6 ms 里**包含机器人在板上做软件 JPEG 编码**
（GC0308 只出 RGB565，没有硬件 JPEG；`CaptureJpeg` 还要先做 115,200 字节的字节交换）。

**修法**：持久 MJPEG / UDP。PC 侧已实现并测过（`pc/perception/frame_source.py`，
15/15）。**固件侧的独立 TCP 流式任务没写**——见铁律 2。

**固件侧的关键约束**：ESP-IDF `httpd` 是**单任务**处理所有请求。阻塞式推流会占住
唯一那个 worker，`/status`、`/cfg`、**`/feat` 全部无响应**——而场景 v2 恰恰要
一边推流一边 `POST /feat`。所以必须是**独立的 TCP 服务任务**，不是给 httpd 加 handler。

**一个容易被忽略的观念**：**"实时" ≠ "高帧率"。** 15 fps 但端到端 30 ms，
主观体验远好于 30 fps 但延迟 150 ms。正确设计是：固定 12–15 fps、每帧走低延迟路径、
**永远取最新帧、旧帧直接丢、任何环节都不许排队**。

### 4.8 灰度化（用户提出，实测支持）

用户："这个任务他或许对颜色不敏感，你要不整成灰白的，或许能够更快一点。"

先测了精度（`pc/bench/_grayscale_check.py`，80 张真实照片跑完整部署链）：

| 指标 | 彩色 BGR | 灰度 | 差异 |
|---|---|---|---|
| 检出人的帧比例 | 100.0% | 100.0% | ±0 |
| 人脸关键点可用率 | 98.8% | 98.8% | ±0 |
| 腕部关键点可用率 | 100.0% | 100.0% | ±0 |
| `person_h` 均值 | 0.672 | 0.672 | +0.001 |
| `theta_img` 平均绝对差 | — | — | **0.26°** |
| **手势对真值准确率** | **85.0%** | **85.0%** | **±0** |

按类别拆开，**灰度在关键类别上反而更好**：

| 类 | n | 彩色 | 灰度 | 差异 |
|---|---|---|---|---|
| **fist** | 14 | 71.4% | **78.6%** | **+7.1pp** |
| like | 12 | 83.3% | **91.7%** | **+8.3pp** |
| palm | 26 | 76.9% | 76.9% | ±0 |
| no_gesture | 28 | 100.0% | 92.9% | **−7.1pp** |

`fist` 正是这个场景的唯一触发条件，灰度让它涨了 7.1 个点。
`no_gesture` 掉了 7.1pp（28 帧里 2 帧假阳性）——靠 `fist_prob_on` 阈值 + 事件机 hold 缓解。

**结构上更快**：

| 项 | 彩色 RGB565 | 灰度 |
|---|---|---|
| 每帧数据量 | 115,200 B | **57,600 B** |
| 字节交换循环 | 115 KB CPU 循环 | **完全消失** |
| JPEG 编码平面 | 3 个 | **1 个** |
| PC 上编码耗时（结构参考） | 0.11 ms | **0.07 ms（1.48×）** |
| JPEG 体积 | 8,138 B | **7,037 B（86%）** |

**为什么灰度不掉点**：手势分类器的外观分支**本来就是单通道**（`crop (B,T,1,32,32)`，
见 `pc/gesture/src/model.py`），它压根没用颜色。只有 YOLO 吃 RGB。

**固件改动**：`arm_board.cc:175` 的 `PIXFORMAT_RGB565` → `PIXFORMAT_GRAYSCALE`，
`CaptureJpeg` 已有 `V4L2_PIX_FMT_GREY` 分支（`esp32_camera.cc:114`）。

**需在真机确认**：GC0308 能否直接输出 GRAYSCALE（esp32-camera 的支持因传感器而异）。
退路：仍取 RGB565，编码前做一次单通道转换（比字节交换便宜，收益小一些但方向不变）。

### 4.9 桌沿检测：实现了，但用户决定先不用

**起因**：场景要求机器人在桌面上前进，"掉下桌子"是唯一会造成硬件损失的事故模式。
之前一直是"前进功能在桌沿保护验证之前不应开启"的阻塞项。

**用户的关键观察**：这板子头低不下去，但它是**自平衡车**——往前加速就必须**整车前倾**，
整台车连同摄像头一起低头；而且它矮（相机离桌面约 0.2 m），只要不完全平行于桌面
就能看到桌沿。

> **机器人唯一会朝桌子前进的时刻，恰好就是它唯一能看清桌沿的时刻。**

**实现**（`pc/perception/table_edge.py`，13 项自检）：

```
光轴以下 α 角的光线：  α = atan((v - v_c) / f)
相对水平面的夹角：      θ = pitch + α          （pitch = 车体前倾角）
落到桌面的距离：        d = cam_h / tan(θ)
```

所以"桌沿在哪一行"直接等价于"桌沿在多少米外"，**不需要立体视觉，也不需要学习**。
检测：逐行纹理能量 + 区域内 90 分位当参考 + 从下往上数连续非桌面行。

**然后用户决定不做**："敲击点在桌上，精准停在那里就不可能掉下去。"

这个推理成立，但**只在不过冲时成立**。代价不对称——少走一点只是位置差，
多走一点可能是摔下去。所以改成**过冲保险**：

- `overshoot_margin_m = 0.12`：从估计距离里预先扣掉，宁可少走绝不多走
- 到达后**锁死**（`_arrived`），此后任何观测修正都不能让 vx 重新变正
  （没有这个锁，目标方位被观测修正时可能形成推拉振荡）
- **严格保证：距离估计误差 ≤ `margin + arrive_tol` = 0.18 m 时，绝不走过真实目标点**

两条边界钉在单测里：高估 0.10 m → 欠调（安全）；高估 0.30 m → 过冲。

**所以"距离估计精度优于 0.18 m"是硬前提，不是可选优化。**
现在 `dist_k = 0.84` 是猜的，**真机必须标定**。

`table_edge.py` 保留不启用。它的合成图测试还暴露：**桌面有杂物时检测误差从 3 行
退化到 10–11 行**（经典 CV 的弱点）。真要用得配 Hough 长水平线或运动视差。

### 4.10 录制脚本改存原始量

**问题**：`session_from_robot.py` 原来只存**组装好的 106 维 `obs`**。
一旦改观测布局（比如加 `theta_face`/`theta_fist`/`knock_*`），**已采的数据全废，得重采**。

**已改**：除 `obs` 外还存
- `meta["feat_64"]`：(N, 64) 每帧感知特征
- `meta["prop_*"]`：**全部**本体状态字段，每键一条 (N,)
- `meta` 里的全部感知标量（含场景 v2 新信号、手势原始概率）

**一份数据可以反复用**：改观测空间只需重写 `assemble()`，不必回机器人重采。

---

## 5. 用户定位与方向纠正历史（重要）

这一轮方向被纠正过三次，每次都改变了工作重点。**按时间顺序**：

### 5.1 "不能否认激活特征的作用"

我写了"特征是净负债"。用户："既然是数据集的问题，那就修数据集，你不能否认这个激活特征
的作用啊，你可以换一个训练输入还有测试输入数据分布相同的任务，看看这个激活特征有没有用。"

→ 做了 `_feat_info.py`，证明同分布 R²=0.846。**我用词过重，已全部改正**（见 §4.2）。

### 5.2 "现在的模型不是部署到真机上的模型"

用户："你以为你现在训练这个模型就是部署到真机上面的模型吗？怎么可能啊？
所以你不要拿现在的这个模型的性能去做补丁。以后肯定要我要自己搞数据集，让你来训练的呀。"

→ **这是最重要的一次纠正。**

| 之前的动作 | 真实定位 |
|---|---|
| `out_nofeat/m8_inert.blob` | **临时可部署占位**，不是最终设计 |
| `calibrate_inputs.py` | 给合成数据模型打的补丁，真机数据到位后不需要 |
| "把特征块删掉" | 只对这份模型成立；**有真机数据就该恢复并训练它** |

→ 产出 `DATASET-PLAN.md`（数据集规格 + 训练验收标准），
并改了录制脚本存原始量（§4.10）。

### 5.3 "让 CfC 承担更多功能"

用户："我觉得 CFc 决策更多是体现在这个敲两下拳头、他会往那个拳头坐标移动这个行为上面去。
……我希望你尽可能让这个 CFC 决策网络承担更多的功能，使得他的行为能够更加灵动点。"

→ **架构纠正**：`scenario_v2.py` 的定位从"要部署的运行时策略"降级为**教师**
（用来生成标签），CfC 才是最终策略。产出 `CFC-V2-DESIGN.md` + 观测 v2 的
v2a/v2b 对照开关（§4.6）。

### 5.4 其它零散要求

- "谁靠着近，你就看谁就好了" → §4.5
- "他头是低不下去的，但是它本身前进的时候头应该是往前倾的" → §4.9
- "先不检测桌面边缘了" → §4.9

---

## 6. 逐个文件说明

### 6.1 感知层（`pc/perception/`）

| 文件 | 作用 | 关键点 |
|---|---|---|
| `run_perception.py` | 主感知管线，一帧一次 `PerceptionOut` | `_pick_person` 按**框高**选最近的；`__call__` 里算场景信号 + 敲击 |
| `scene_signals.py` | 场景 v2 信号层 + `KnockDetector` | 只吃关键点，不加载 torch，可独立单测 |
| `scenario_v2.py` | 场景 v2 决策状态机（**教师**） | `_allocate` 与教师控制器故意不同：头 100% 对准，不许斜眼 |
| `table_edge.py` | 桌沿检测（**停用，备用**） | 地面单应：`d = cam_h / tan(pitch + atan((v-v_c)/f))` |
| `frame_source.py` | 取流抽象：`HttpPollSource` / `MjpegSource` | 两个源同一接口 `latest()` |
| `test_scene_signals.py` | 24 项单测 | |
| `test_scenario_v2.py` | 52 项单测 | |
| `test_pick_person.py` | 10 项单测 | 桩对象喂 `_pick_person`，不需 GPU |
| `test_frame_source.py` | 15 项自检 | 起本地 mock 服务端 |
| `_smoke_scene.py` | 新信号在真实照片上的冒烟 | |

**`PerceptionOut` 新增的 10 个字段**（带默认值，追加在末尾）：
`theta_face` / `face_scale` / `face_valid` / `theta_fist` / `wrist_y_norm` /
`hand_side` / `knock_fired` / `knock_x` / `knock_y` / `knock_count`。

### 6.2 训练侧（`pc/train/`）

| 文件 | 作用 |
|---|---|
| `obs.py` | v1 观测组装（106 维）+ `OBS_NAMES` |
| `obs_v2.py` | v2 观测组装（116 维）+ `TargetLatcher` + v2a/v2b。**直接跑就是 16 项自检** |
| `session_from_robot.py` | 真机录制。**已改为存原始量**（§4.10） |
| `calibrate_inputs.py` | 重写 blob 的 `input_mean/std`。`--sessions` 从录制拟合，或 `--feat-mean/--feat-std` 给常量 |
| `_make_nofeat.py` | 造特征块恒置 0 的训练集 |
| `export_blob.py` | blob 导出 + 偏移三方核对 + CRC |
| `train_cfc.py` | 训练。**目前只认 106 维，改成能吃 v2 是第一优先待办** |
| `test_e2e.py` | 数值一致性套件（torch↔numpy↔C），不是重训入口 |

### 6.3 基准与分析（`pc/bench/`）

| 文件 | 作用 |
|---|---|
| `deploy_gate.py` | **真机帧部署回归门禁**。候选按出厂评分，必须打赢合成特征基线 |
| `_compare_blobs.py` | 候选 blob 在真机帧上的横评表 |
| `_feat_info.py` | 特征块是否携带信息（R² 实验） |
| `_grayscale_check.py` | 灰度 vs 彩色精度对比 |
| `_latency_pipeline.py` | 整条感知链延迟拆解 |
| `_fist_distance.py` | 握拳在不同距离下的可分性 |
| `eval_session.py` | 用**固件那份 C 引擎**跑会话，对教师标签评分 |
| `obs_domain_check.py` | **诊断项**，设计上为红（拿错参照，见 §7.2） |
| `_real_feat_control.py` | 特征块是否承重的控制实验 |
| `_recalib_cv.py` / `_recalib_test.py` | 归一化重拟合的交叉验证 |

### 6.4 工具（`pc/tools/`）

| 文件 | 作用 |
|---|---|
| `robot_watch.py` | 轮询机器人回来 |
| `net_probe.py` | 热点/网段/机器人可达性诊断 |
| `sniff_serial.ps1` | 只读监听 COM3（**DTR/RTS 都设 false**，避免触发复位） |
| `reboot_and_capture.py` | 脉冲复位 + 抓开机日志（会重启机器人，有响声） |
| `reset_sweep.py` | 四种 DTR/RTS 极性全试 |
| `mock_mjpeg_server.py` | 本地 MJPEG/单帧服务端，给 `test_frame_source.py` 用 |

### 6.5 顶层文档

| 文件 | 内容 |
|---|---|
| `TECHNICAL-REPORT.md` | 主报告，~1055 行。**§4.3c 有适用范围声明** |
| `SCENARIO-v2.md` | 场景规格 + 阶段状态 + 采集协议 + 延迟分析 |
| `CFC-V2-DESIGN.md` | 架构：状态机是教师、CfC 是策略、v2a/v2b、三个消融实验 |
| `DATASET-PLAN.md` | **数据集规格 + 训练验收标准** |
| `RESULTS.md` | 历史实测数据（从编码事故重建，785 行） |
| `HANDOFF-2026-11-03.md` | 第一份交接 |
| `FINAL-VERIFICATION.txt` | 带时间戳的验收输出 |
| `README.md` | 入口 + 5 分钟自查 |

---

## 7. 测试清单（每个套件到底测什么）

### 7.1 门禁（23 项，全部必须绿）

```powershell
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\run_all_tests.py --build
```

| id | 测什么 |
|---|---|
| `cfc_host` | CfC 三方数值对拍（ncps ↔ numpy ↔ C） |
| `train` | 训练 + blob 导出 + 固件 C 引擎端到端 |
| `obsv2` | 观测 v2 组装（16 项，含"前 106 维与 v1 逐元素相同"） |
| `precision` | 权重精度研究（bf16/fp16/int8/混合） |
| `perception` | 感知链（T01–T19） |
| `scenesig` | 场景 v2 信号层（24 项） |
| `scenariov2` | 场景 v2 状态机（52 项） |
| `framesrc` | 取流两源（15 项） |
| `pickperson` | 最近的人选择（10 项） |
| `tableedge` | 桌沿检测（13 项，**备用**） |
| `gesture` | 手势分类 + 事件分割 |
| `teacher` | 教师控制器 |
| `hub` | 分层决策中枢（90 项） |
| `escalate` | CfC 升级头 |
| `bridge` | 本地 agent 桥（真实副作用） |
| `sim` | 闭环仿真，4 种控制器 |
| `realimg` | 真人照片端到端手势 |
| `e2ereplay` | 离线回放：固件引擎 vs 教师 |
| `evalsession` | 留出会话评估（目标 `out_nofeat/m8_inert.blob`） |
| `deploygate` | **真机帧部署回归门禁** |
| `threetier` | 三档分配行为 |
| `safety` | 固件安全壳 7 项防御 |
| `firmware` | `idf.py build`（`--build` 才跑） |

### 7.2 诊断（1 项，设计上为红，不计分）

| id | 为什么红 |
|---|---|
| `obsdomain` | 它拿 **512×512 的 HaGRID 手部特写**当参照，而机器人摄像头拍的是 **240×240 房间视野**——参照分布本身错了。信号很明确：用 `--framing robot` 模拟机器人取景后 clip 比例不降反升（40.1% → 47.7%）。真正回答这个问题的是 `deploygate`：用真机帧，并且**按行为判（教师 ACK 召回），不按统计量判** |

### 7.3 单独跑

```powershell
& $py D:\robot_work\run_all_tests.py --only deploygate
& $py D:\robot_work\run_all_tests.py --only scenariov2
& $py D:\robot_work\pc\perception\test_scenario_v2.py     # 直接跑也行
```

---

## 8. 已验证 vs 未验证

### ✅ 已验证（有数字）

- 固件编译、烧录、全部端点（`RESULTS.md` §1）
- CfC 真机推理性能：fp32 **4.037 ms**、int8 **1.4–1.6 ms**
- 三方数值一致：torch↔numpy↔C 最大差 **2.4e-6**
- 手势：宏 F1 **0.8051**；真人照片端到端 **72%**（真值骨架 85.4%）
- 决策中枢：L0 **38–64 µs**、L2 **99.9 ms**、L3 **740–860 ms**
- 端到端延迟：**121.5 ms**（HTTP 90%）
- 灰度化：精度不变、fist **+7.1pp**
- 真机帧可部署性：出厂 **5.63° / ACK 0/24** → 占位模型 **2.26° / ACK 22/24**

### ❌ 未验证 / 需要真机

| 项 | 为什么卡住 |
|---|---|
| **真机闭环行为** | 整晚横躺（`pitch≈90°`），`robot_state=0` |
| **`lqr_k=[1300,5,25,0.4]` 的符号** | 仿真认为符号使平衡环**结构上不可稳定**，从没在真机上验过。**这是头号阻塞项** |
| **`dist_k = 0.84`**（距离估计） | 猜的。决定 §4.9 那个 0.18 m 保证能不能成立 |
| **场景 v2 的一切** | 信号层/状态机全部只是单测过，没有真实数据 |
| **`KnockDetector` 阈值** | 全部是先验，未标定 |
| **灰度化在真机上** | GC0308 能否直接出 GRAYSCALE 未验证 |
| **MJPEG 流式固件** | 未实现（铁律 2） |
| **桌沿检测** | 合成图 13/13，但**抗杂物能力未验证**，且已停用 |
| 真人挥手识别 | HaGRID 是静态图，挥动手势是 IK 合成 |
| `pointing` 类 | F1 0.574，COCO-17 没有手指关键点，结构性不可救 |
| fp16/bf16 真机耗时 | 未测 |
| 云端 LLM 真调 | 环境无 API key，L2/L3 走 mock（降级路径已测） |

---

## 9. 下一步（含具体命令）

### 9.1 机器人一通，按这个顺序

**(1) 确认站立 + 验证 LQR 符号**（30 分钟，**最高优先级**）

```
机器人站立 → GET /api/set?i=10&v=-1300  （改 K0 符号）→ 观察收敛还是发散
```

**验证前不要改固件。** 这一步不过，后面全是空的。

**(2) 标定 `dist_k`**

现在 `d ≈ 0.84 / person_h`。做法：把机器人放在离人已知距离处，读 `/status` 里的
`person_h`，拟合 `person_h → 距离` 的关系。

**这决定 §4.9 那个 0.18 m 的保证能不能成立。** 如果标定后精度做不到，就得接回
`table_edge.py`。

**(3) 串口开着写固件流式任务**

关键约束（重复一遍，很容易踩）：
- **不要挂在 httpd 上**——它单 worker，推流会堵死 `/feat`
- 独立 FreeRTOS 任务 + lwIP socket + 另一个端口（如 81）
- 默认关闭（配置开关），刷进去不影响现有功能
- 边烧边看串口，重点验证**栈大小**和**长连接行为**

**(4) 采数据**

按 `DATASET-PLAN.md` §2 的脚本演。**每段必须有完整的"敲两下 → 手离开 → 走过去"**，
否则教师产生不了 APPROACH/DWELL 标签，CfC 学不到。

### 9.2 不需要机器人、现在就能做

**(5) 把 `train_cfc.py` 改成能吃 v2 观测** ← **这是我离开前最建议的下一步**

现在它只认 106 维。用户数据一采完就应该能直接跑训练，不要再等一轮开发。
改法：`--obs-version {1,2}` + `--scene-variant {a,b}`，内部调 `obs_v2.assemble_v2`。

**(6) 写两个小工具**

- `_check_dataset.py`：确认录出来的 npz 里有 `feat_64` 和 `prop_*`
- `_knock_replay.py`：敲击检测在录制数据上的触发次数 vs 实际敲了多少次

**(7) 跑三个记忆消融**（`CFC-V2-DESIGN.md` §4）

| 实验 | 做法 | 判据 |
|---|---|---|
| A. 隐状态清零 | 每帧把 h 归零（退化成前馈） | "拳头消失后仍走对"成功率显著下降 → 记忆在起作用 |
| B. 敲击脉冲延迟 | 延迟 N 帧再动作 | 平滑推迟而非乱走 |
| C. 目标方位扰动 | 敲击那帧方位偏转 ±20° | v2b 应走**偏转后**的位置 |

### 9.3 明确不要做

- ❌ 围绕 `m8_inert` 继续优化（见 §5.2）
- ❌ 没有硬件验证就改固件（铁律 2）
- ❌ 把 `evalsession` 这类离线合成集指标当验收依据（这轮已经吃过一次亏）
- ❌ 相信旧表格里的数字而不重测（见 §10）

---

## 10. 踩坑记录（详细）

| # | 坑 | 教训 |
|---|---|---|
| 1 | 认为"串口静默 = 死机" | 日志级别 WARN，正常运行本来就不打印。判活只能看 `/status` 200 |
| 2 | 认为"特征抽取 56.3 ms" | 实测 **0.2 ms**。**引用旧表格前先重测** |
| 3 | 把当前模型当最终部署目标做工程 | 用户直接纠正。见 §5.2 |
| 4 | 写测试时自己错了 4 次 | 死区取值（3° 恰好落在 3° 死区边界）、透视不变量（腕部角偏移**本来**该随距离变小）、敲击窗口（间隔 5 秒本来就不该配对）、接近相位（漏传 knock 事件导致根本没进 APPROACH）。**先怀疑测试再怀疑代码，但每次都要真的查清是哪边错** |
| 5 | 桌沿检测把"画面最下面一行"当桌面参考 | 桌沿在画面里时最下面恰恰在桌沿外，阈值被压到地板水平 → **越接近桌沿越检测不到**。这是最危险的方向 |
| 6 | 用 PowerShell 批量改含中文的文件 | 毁过 4 个文件。铁律 1 |
| 7 | 单次幸运结果当结论 | 之前报过 2.80°/ACK 25%，5 个种子复现是 8.5°±3.9%/ACK 0% |
| 8 | `RESULTS.md` 判定"编码往返不可逆" | **这个判断错了一半**：失败是因为用了 `gbk`，`gb18030` 可以回编（GBK 的 PUA 字符在 `gbk` 编解码器里不能回编）。真正不可逆的只有被 `?` 替换掉的那部分字节 |
| 9 | 用 `jieba` 词频猜回丢失的字 | 它猜出"项/值"→"须/倒"、"表"→"血"、"法"→"泥"，读着通顺全错。**技术报告里悄悄插错字比留空洞更糟**，已放弃 |

**特别提醒 2 和 8**：这两条都是"我引用了没验证的东西"。这个项目里旧文档的数字
有相当一部分是当时环境下的测量，**环境变了就不一定成立**。

---

## 11. 开放问题 / 待决策

| # | 问题 | 现状 |
|---|---|---|
| 1 | 观测 v2 里放不放人脸/拳头的**连续量** | 已按"放"实现（`theta_face`/`theta_fist`/`wrist_y_norm`）。理由：让 CfC 自己学决策，把手写规则塞进感知层就失去意义 |
| 2 | `mode` 从 4 类扩到 6 类，固件能不能吃 | 固件从 blob 头读 `head_dim`，理论上是数据变化。**必须在 `eval_session.py`（走固件 C 引擎）上跑通再烧** |
| 3 | v2b 训不训得出来 | 未知。先训 v2a 作基线，用 §9.2 的消融判断 |
| 4 | 灰度化的固件可行性 | GC0308 能否直接出 GRAYSCALE 未验证 |
| 5 | 距离估计精度能不能优于 0.18 m | 取决于 `dist_k` 标定和距离估计方法（最好改用手部尺度而不是人体尺度） |
| 6 | 板载 64 维特征块最终删不删 | **不删**。真机数据到位后恢复并训练它 |

---

## 12. 命令速查

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'

# 全量验收（含固件编译）
& $py D:\robot_work\run_all_tests.py --build

# 单个套件
& $py D:\robot_work\run_all_tests.py --only deploygate

# 场景 v2 相关
& $py D:\robot_work\pc\perception\test_scene_signals.py
& $py D:\robot_work\pc\perception\test_scenario_v2.py
& $py D:\robot_work\pc\perception\test_pick_person.py
& $py D:\robot_work\pc\perception\test_frame_source.py
& $py D:\robot_work\pc\perception\table_edge.py
& $py D:\robot_work\pc\train\obs_v2.py

# 分析
& $py D:\robot_work\pc\bench\_compare_blobs.py
& $py D:\robot_work\pc\bench\_grayscale_check.py --n 80
& $py D:\robot_work\pc\bench\_latency_pipeline.py
& $py D:\robot_work\pc\bench\_feat_info.py

# 机器人
& $py D:\robot_work\pc\tools\robot_watch.py --ip 192.168.137.50 --minutes 240
& $py D:\robot_work\pc\tools\net_probe.py

# 烧录固件
$env:IDF_TOOLS_PATH='G:\esp\.espressif'
cmd /c "call G:\esp\esp-idf\export.bat >nul 2>&1 && cd /d D:\robot_work\rig_omni && idf.py -p COM3 flash"

# 抓串口（DTR/RTS 都设 false，避免触发复位）
powershell -NoProfile -ExecutionPolicy Bypass -File D:\robot_work\pc\tools\sniff_serial.ps1 -Port COM3 -Seconds 6

# 编码检查（每次写完中文文件都跑）
& $py D:\robot_work\_check_encoding.py

# 从无标签真机帧重拟合 blob 归一化
& $py D:\robot_work\pc\train\calibrate_inputs.py --sessions D:\robot_work\pc\train\data_real `
  --blob D:\robot_work\pc\train\out_sw2_b256_80e\m8.blob `
  --out D:\robot_work\pc\train\out_sw2_b256_80e\m8_cal.blob
```

**串口参数**：COM3，115200，8N1，`DtrEnable=false`，`RtsEnable=false`。

---

## 13. 文件索引

| 文件 | 内容 |
|---|---|
| `D:\robot_work\TECHNICAL-REPORT.md` | 主报告，~1055 行，15 条 bug 记录 + §4.3c |
| `D:\robot_work\SCENARIO-v2.md` | 场景规格 + 阶段状态 + 采集协议 + 延迟分析 |
| `D:\robot_work\CFC-V2-DESIGN.md` | 架构纠正 + v2a/v2b + 三个消融实验 |
| `D:\robot_work\DATASET-PLAN.md` | **数据集规格 + 训练验收标准** |
| `D:\robot_work\RESULTS.md` | 历史实测数据（785 行，已重建） |
| `D:\robot_work\HANDOFF-2026-11-03.md` | 第一份交接（固件/崩溃/编码事故） |
| `D:\robot_work\HANDOFF-2026-11-03-part2.md` | **本文档** |
| `D:\robot_work\FINAL-VERIFICATION.txt` | 带时间戳的验收输出 |
| `D:\robot_work\README.md` | 入口 + 5 分钟自查 |
| `D:\robot_work\TECH-PLAN-v1.md` | 施工级技术方案（1412 行） |
| `D:\robot_work\CONTRACT.md` | 四工作流接口契约 |
| `D:\robot_work\_check_encoding.py` | 交付文档编码检查 |

**产物**：
- `pc/train/out_nofeat/m8_inert.blob`（116 KB，int8）—— **当前推荐的可部署占位**
- `pc/train/out_sw2_b256_80e/m8.blob`（113 KB）—— 合成特征基线，真机上不可用
- 固件：`rig_omni/build/rig-hover.bin` = **2,983,936 B**
- 原厂 app 备份：`C:\Users\sans\AppData\Local\Temp\robot_probe\factory_app.bin`（3.94 MB）
