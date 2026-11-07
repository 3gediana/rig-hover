# AGENTS.md —— RIG-Hover（文文）工作手册与文档索引

> 给**下一个接手这个项目的 agent**。目标：你读完这一份，就知道
> 东西在哪、怎么跑、别踩哪些坑、现在到哪一步了。
>
> 最后更新：2026-11-07。
> 用户偏好见 §8，**动手前先读那一节**。
> **GitHub 仓库**：https://github.com/3gediana/rig-hover （**public**；`dataset/` 里的视频会公开，本地工作副本即 `D:\robot_work`）

---

## 0. 三十秒了解

**是什么**：一台 ESP32-S3 的双轮自平衡桌面机器人（代号 文文 / RIG-Hover）。
一个舵机只能摇头（**不能低头**），GC0308 摄像头，8 MB PSRAM，16 MB flash。

**做什么**：场景 v2 —— 1–2 米内盯人脸 → 人握拳就盯拳头 → 在桌上敲两下 →
朝敲击点走过去并停住（哪怕拳头已经离开）→ 停一会儿回看人脸。

**核心架构**（这一条理解错，后面全错）：

```
PC 端（GPU）                          机器人 ESP32-S3
┌──────────────────────┐             ┌────────────────┐
│ YOLO26n-pose 感知链   │  116 维观测  │                │
│ 场景信号（脸/拳/敲）   │ ──HTTP──▶   │   CfC  策略     │
│ 教师状态机（训练时）   │  /feat      │   50 Hz         │
└──────────────────────┘             └────────────────┘
```

* **CfC 是唯一部署到机器人上的模型**。它吃 116 维数字，**不碰图像**。
* YOLO、手势分类器、教师状态机**全在 PC 上**。
* 教师状态机**是标签机**，不是运行时策略：它逐帧生成训练标签，本身不上机。

**当前阶段**：真机链路已打通并优化过一轮（端到端 75 ms / 12.9 Hz），
但**部署的模型是临时占位**，最终要用用户采的真机数据重训。

---

## 1. 五分钟上手

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'

# 机器人地址（PC 的 ICS 热点网段，热点评段 192.168.137.0/24）
# 机器人 IP 固定为 192.168.137.50（固件内有静态兜底）
curl.exe --noproxy '*' http://192.168.137.50/status

# 端到端延迟（需要机器人在线）
& $py D:\robot_work\pc\tools\e2e_latency.py --n 60 --base http://192.168.137.50 --warmup 10

# 全部测试（23 个门禁 + 1 个诊断）
& $py D:\robot_work\run_all_tests.py

# 构建固件
$env:IDF_TOOLS_PATH='G:\esp\.espressif'
. 'G:\esp\esp-idf\export.ps1'
Set-Location 'D:\robot_work\rig_omni'
idf.py build
idf.py -p COM3 -b 460800 app-flash
```

**关键路径**

| 东西 | 位置 |
|---|---|
| GitHub 仓库 | https://github.com/3gediana/rig-hover（**public**；`dataset/` 里的视频会公开） |
| PC 端代码 | `D:\robot_work\pc\` |
| 固件 | `D:\robot_work\rig_omni\` |
| 文档 | `D:\robot_work\*.md` |
| Python 虚拟环境 | `D:\smartcar\.venv\`（`site-packages` 在 `G:\ai\python11`） |
| ESP-IDF | `G:\esp\esp-idf`，tools `G:\esp\.espressif` |
| 串口 | `COM3`（CH343） |

---

## 2. 文档索引

先看这一列："什么时候读它"。

| 文档 | 行数 | 什么时候读 |
|---|---|---|
| **`AGENTS.md`（本文）** | — | **第一个读。** 索引 + 踩坑 + 纪律 |
| **`COLLAB.md`** | — | **外部协作者（马伟坚）第一个读。** 分工、数据与标注约定、红线 |
| `dataset/README.md` | — | 数据集样例目录：收录规则、导出命令、annotation 格式 |
| `dataset/docs/ROBOT-SETUP.md` | — | **新电脑/新操作者先读。** 电脑热点 `qasd`/`88888888`、重新配网、服务启动、红线 |
| `dataset/docs/COLLECTION-AND-ANNOTATION.md` | — | 采什么、六类标签定义与易混对、数量目标、按 take 划分 |
| `dataset/docs/EXAMPLE-TAKE.md` | — | 已标注范本 take 的逐区间表与自定义标签合并建议 |
| `dataset/docs/TRAIN-PIPELINE.md` | — | **区间标注 → 逐帧敲击检测**的训练/评估流水线：数据流、标签合并、留一协议、命令与当前结果 |
| `VIDEO-ANNOTATOR.md` | ~92 | 8780 采集/标注服务：启动、API、帧率与暂停语义 |
| `WIFI-AUTOCONNECT-FIX.md` | ~108 | 机器人热点自动重连的根因、烧录记录与验证方法 |
| **`RESEARCH-VideoCfC.md`** | ~580 | **当前科研课题。** 流式视觉决策的统一数学问题、CNN/RNN/LSTM/GRU/Transformer/Mamba/CfC 设计与方程、文献地图、待证明命题；当前阶段只做数学推导，不训练或实验 |
| `LATENCY-MAP.md` | ~250 | 改性能、查延迟、诊断网络瓶颈 |
| `TECHNICAL-REPORT.md` | 1097 | 要写报告 / 要引用数字 / 查 bug 记录（§5 有 15 条） |
| `RESULTS.md` | 785 | 要**实测数字**（精度、延迟、消融）——数字都可交叉验证 |
| `TECH-PLAN-v1.md` | 1412 | 要原始设计意图、契约出处 |
| `CONTRACT.md` | 220 | 改 `PerceptionOut` 字段 / 观测布局前**必读** |
| `SCENARIO-v2.md` | 284 | 改场景逻辑、状态机、数据采集协议 |
| `CFC-V2-DESIGN.md` | 179 | 改 CfC 架构 / 观测 v2 / 记忆消融实验 |
| `DATASET-AND-ANNOTATION.md` | 398 | 采数据前必读：采什么、要不要标注、人脸/拳头的小模型选型 |
| `DATASET-PLAN.md` | 174 | 前一版数据集计划。**被上一份部分修正**，冲突时以新份为准 |
| `TRANSPORT-REDESIGN.md` | ~300 | 传输层历史决策。**注意：其核心前提已被推翻，见文首状态更新** |
| `HANDOFF-2026-11-03.md` | 401 | 那一夜的完整交接（固件/崩溃/编码事故） |
| `HANDOFF-2026-11-03-part2.md` | 963 | 更详细的交接（14 节，含 blob 格式表、106 维逐维说明） |
| `HANDOFF-rig-hover.md` | 309 | 更早的交接 |
| `MORNING-CHECKLIST.md` | 454 | 用户早上要按顺序做的操作清单 |
| `FINAL-VERIFICATION.txt` | 51 | 时间戳验收日志 |

**读文档的顺序**：本文 → `LATENCY-MAP.md`（了解现状）→
`TECHNICAL-REPORT.md` §5（了解踩过什么）→ 按任务查具体文档。

---

## 3. 代码结构

### 3.1 `pc/` —— PC 端

| 目录 | 内容 |
|---|---|
| `perception/` | **感知链主干**。`run_perception.py` 是入口，`scene_signals.py` 是脸/拳/敲击信号层，`scenario_v2.py` 是**教师状态机**，`table_edge.py` 是桌沿检测（已实现但用户决定不用），`frame_source.py` 是取流 |
| `train/` | 训练与导出。`obs.py`/`obs_v2.py` 观测组装，`train_cfc.py` 训练，`export_blob.py` 导出 blob，`session_from_robot.py` **真机录制**，`synth.py` 合成数据 |
| `bench/` | 测量与门禁。`deploy_gate.py` 部署回归门禁，`e2e_replay.py` 离线回放，`eval_session.py` 留出集评估，以及各 `_*.py` 一次性实验脚本 |
| `teacher/` | 旧版教师（`teacher.py`），v2 用 `perception/scenario_v2.py` |
| `gesture/` | 手势分类器（12 帧因果滑窗）+ 事件机 + HaGRID 数据处理 |
| `tools/` | 运维工具：`e2e_latency.py`、`http_floor_probe.py`、`vlm_label.py`、`net_probe.py`、`robot_watch.py` |
| `cfc_host/` | CfC 三方数值一致性（ncps / numpy / C） |
| `decision_hub/`、`escalate/`、`sim/`、`precision/` | 并列的实验分支，主线不依赖 |

### 3.2 `rig_omni/main/` —— 固件

| 路径 | 内容 |
|---|---|
| `boards/hover/` | 主逻辑。`hover_board.cc` 板级初始化，`hover_debug_server.cc` **所有 HTTP 端点**，`cfc_task.c` CfC 推理任务，`control_arbiter.c` 仲裁器与安全壳，`cfc_engine.c`/`cfc_model.c` 推理引擎 |
| `boards/common/` | 共用板级代码。`wifi_board.cc` **WiFi 连接与静态 IP 兜底**，`esp32_camera.cc` **取图与 JPEG 编码** |
| `display/lvgl_display/jpg/` | `image_to_jpeg.cpp` 图像→JPEG 转换（支持 GREY/YUYV/RGB565） |

### 3.3 关键命令

```powershell
# 全部门禁（23 套 + 1 诊断）
& $py D:\robot_work\run_all_tests.py
& $py D:\robot_work\run_all_tests.py --build      # 额外跑固件构建

# 单独跑某套
& $py D:\robot_work\pc\perception\test_scenario_v2.py
& $py D:\robot_work\pc\train\test_recorder.py     # 录制是否真的存了图

# 检查文档编码（UTF-8 / 无 BOM / 无 PUA / 无 U+FFFD）
& $py D:\robot_work\_check_encoding.py
```

---

## 4. 硬件与环境事实

| 项 | 值 | 出处 |
|---|---|---|
| MCU | ESP32-S3 QFN56 rev0.2，双核 240 MHz，512 KB SRAM | 启动日志 |
| PSRAM / Flash | 8 MB Octal / 16 MB | 同上 |
| MAC | `90:da:72:4f:9b:e0` | `esptool read_mac` |
| 控制环 | **6 ms（167 Hz）**，不是 500 Hz（3× `vTaskDelay(2)`） | 源码 |
| CfC 推理 | 1.4–4.0 ms（int8 / f32），`cfc_hz=50` | 实测 |
| blob 上限 | `CFC_SLOT_MAX_BYTES = 262144`（256 KB），在 PSRAM 里 | 源码 |
| `input_dim` 上限 | 256（现在用 116） | 源码 |
| 相机 | GC0308，**驱动支持到 VGA 640×480**，现在配 240×240 RGB565 | `gc0308.c:182` |
| FOV | `FOV_H = 60°`（**未标定，和 `dist_k` 矛盾，见 §6.4**） | 常量 |
| 日志级别 | 2026-11-03 夜已改为 **INFO**（原来是 WARN） | `sdkconfig` |

---

## 5. 踩坑清单

### 5.1 文件与编码 —— 已经毁过 4 个文件

1. **`Get-Content -Raw` 会把 UTF-8 当 GBK 解码。** 一个中文字符后面紧跟一个
   `< 0x40` 的字节时，那个字节会被吞成 `?`。**不可逆**。
2. **`gb18030` 往返能救回大部分**（`gbk` 不行，它编不了 PUA 字符）。
   `RESULTS.md` 就是这样救回 ~85% 的，1472 字节永久丢失。
3. **永远不要用 PowerShell here-string 写代码文件**——引号会被吃掉。
   用 `write` 工具。上面第 2 条的直接后果。
4. **`Tee-Object` 写的是 UTF-16**。日志文件被它写过之后，后续工具会读成乱码。
5. 写完文档跑一次 `_check_encoding.py`，它查 UTF-8 / BOM / PUA / U+FFFD。

### 5.2 固件与烧录

6. **串口静默 ≠ 死机。** 日志级别 WARN 时健康机器人一个字都不打。
   唯一判据是 `/status` 返回 200。
7. **`CONFIG_LOG_MAXIMUM_EQUALS_DEFAULT=y`** 时，`LOG_MAXIMUM_LEVEL` 跟着
   `LOG_DEFAULT_LEVEL` 走。改日志级别要把两个都改，否则 INFO 被编译掉。
8. **崩溃循环怎么查**：抓完整 panic 的 Backtrace，然后
   ```powershell
   & 'G:\esp\.espressif\tools\xtensa-esp-elf\esp-14.2.0_20251107\xtensa-esp-elf\bin\xtensa-esp32s3-elf-addr2line.exe' `
       -pfiaC -e 'D:\robot_work\rig_omni\build\rig-hover.elf' 0xADDR1 0xADDR2 ...
   ```
   `EXCVADDR` 是最快的线索：`0x000000c0` = 192，正是 `IP2STR` 里的第一个字节被
   当成指针解引用（`%s` 配了 `IP2STR`，应该用 `IPSTR` 配 `IP2STR`）。
9. **CH343 会卡死在 Windows 错误 31**（"设备未正常运作"），设备管理器却显示
   `CM_PROB_NONE`。软件禁用/启用、`pnputil /restart-device` 都无效
   （后者还要管理员）。**只能物理重插 USB。**
10. `idf.py app-flash` 失败时先看 `build\log\idf_py_stdout_output_*.log`，
    ninja 的报错本身不含原因。
11. 用 `idf.py -p COM3 -b 460800 app-flash` 比 `-b 921600` 稳。

### 5.3 网络 —— 这次的三个坑

12. **热点"显示 On"不代表在发信标。** 判据是看热点虚拟网卡的
    **TX 字节增量**：`Get-NetAdapterStatistics -Name 'WLAN 3'`。
    0 增量 = 没在广播，机器人永远扫不到。
13. **DHCP 不回时，机器人只有 L2 没有 L3。** 串口日志里会看到
    `wifi:connected with ... aid = N` 之后**没有** `Got IP`。
14. **`esp_netif_set_ip_info()` 自己会发 `IP_EVENT_STA_GOT_IP`。**
    不要再手动调 `OnNetworkEvent(Connected)`——连接流程会跑两遍，
    调试服务器初始化两次，立刻崩溃循环。
15. `keep_alive_enable = true` 已经开了，但**没解决 50 ms 那个问题**，
    因为那是 Nagle，不是连接复用。
16. **IDF 5.5 没有 `CONFIG_LWIP_TCP_NODELAY`**（已验证）。要关 Nagle 只能
    在 handler 里对 socket 单独 `setsockopt`。

### 5.4 感知与数据

17. **录制必须存原图。** 派生量（`feat_64`、`theta_img`、教师标签）都能从原图
    重算，反过来不行。`session_from_robot.py --save-frames` **必须开**。
    没有像素就换不了感知模型、裁不出手图、也没法用 VLM 补标。
18. **要存腕部关键点。** 手势分类器吃的是腕部周围的 64 px 手图，只存
    `theta_fist` 这类派生标量的话，手图永远拿不回来。
19. **240×240 下拳头只有 10–21 像素。** 9 cm 的手在 1 m 处 21 px，2 m 处 10 px。
    这是"握拳"识别在 1–2 米只有 56–69% 的根本原因。
20. **人脸有 227 KB 的专用模型就够。** YuNet（OpenCV `FaceDetectorYN`）在
    24 px 上召回 92%，17 px 上 85%——1–2 米全程可用，240×240 都不用改。
21. **不要用大 VLM 做小目标的标注。** 通用视觉语言模型 1 个 token 约 32×32 像素，
    14 像素的手装不满一个 token。专用小模型（227 KB / 14 MB）才是对的工具。

### 5.5 评估方法

22. **离线合成集指标不能当验收依据。** 吃过一次亏：离线 0.86°、ACK 75.9% 的模型，
    上真机比"什么都不做"还差（真机 head MAE 5.63°，而全零基线 1.61°）。
23. **按段（session）切分，绝不按帧随机切。** 同段相邻帧高度相关，
    随机切等于把测试集漏进训练集。
24. **测什么就要说清测的是哪一段。** `http.client.getresponse()` 读的是
    "响应头收齐"，不是"首字节"，据此得出"不是 Nagle"的错误结论，
    方向错了一整轮。

### 5.6 判断与协作

25. **先翻自己的文档。** 用户明确批评过："你知不知道你现在的这些问题在你原先
    就已经踩过了，你就没有考虑过看看你过去写得一些日志吗"。接手先读 §2 索引。
26. **不要在拿到证据前下"不可优化"的结论。** "HTTP 每请求 50 ms 是固有开销、
    只能绕开"这个判断让整个传输层重写方案跑偏了一轮。
27. **当前模型是占位，不是最终模型。** 不要拿它的性能去设计补丁——
    用户会自己采数据集重训。
28. **估算常比实测乐观 2–3 倍。** 推理时延估 1.5 ms、实测 4.04 ms。
    先量再改。

### 5.7 延迟攻坚这一轮新踩的（2026-11-03 夜）

29. **`%lld` 在这个工程里打不出来。** `CONFIG_NEWLIB_NANO_FORMAT=y`，nano
    printf 不支持 64 位格式：`ESP_LOGI("...%lld...", (long long)x)` 会原样
    打印 `ld`，后面的参数**全部错位**（于是出现 `out=520` 这种鬼数字）。
    用 `%ld` + `(long)` 强转。看上去"日志有输出"不等于它是对的。
30. **打开 COM3 会让 ESP32 复位。** pyserial 一 `open()` 就拉 RTS→EN。
    必须先把两条线置低**再**开：
    ```python
    p = serial.Serial(); p.port = "COM3"; p.baudrate = 115200
    p.dtr = False; p.rts = False; p.open()
    ```
    **不能**写成 `with serial.Serial() as p:` —— 上下文管理器先 open 再执行
    你的赋值，照样复位。这个坑让我把"灰度编码卡死 httpd"误判过一次。
31. **httpd 和 CFC 模型抢 internal RAM，启动顺序不能反。**
    `cfc_model_init()` 把 43 KB 的 int8 blob 放进 internal SRAM（故意的，
    避开 PSRAM cache miss）。若先 init 再起 httpd，`httpd_start()` 时只剩
    4.8 KB < `CONFIG_HTTPD_STACK_SIZE`(4 KB)，直接返回
    `ESP_ERR_HTTPD_TASK` —— **机器人能 ping 通、平衡照跑、一个端点都没有**。
    正确顺序：先 `httpd_start()`（此时 41 KB 可用）→ 再 init 策略运行时
    → 最后注册 URI handler（init 那 ~100 ms 内的请求拿 404，不会撞上
    半初始化的 arbiter）。
    **同一个坑的第二层**：`cfc_task` 要 12 KB 连续栈，模型 blob 要 ~33 KB，
    两者加起来正好压在这 48 KB 的边角上，谁先谁后决定谁能活。
    实测出现过 `E CfcTask: failed to create cfc_task (inference disabled)`
    而板子其它一切正常。正确顺序是**必须的在前、可选的在后**：
    `cfc_task_start()` → `cfc_model_init()`（它 internal 分配失败会自动
    退到 PSRAM）→ `emote_follow`（装饰性，永远最后）。
32. **`idf.py build` 会把直接改的 sdkconfig 改回去。** 手改
    `CONFIG_ESP_CONSOLE_UART_BAUDRATE=921600`，下一次 build 后又是 115200
    （而 `sdkconfig.defaults` 里根本没有这一项）。要稳定的覆盖就写进
    `sdkconfig.defaults`，或者用 `idf.py menuconfig`。
    `CONFIG_LWIP_TCP_SND_BUF_DEFAULT` 这类改动倒是活下来了——
    所以**别假设"改了就在"**，改完要回读确认。
33. **`. export.ps1` 会把 `python` 换成 IDF 解释器。** 之后跑 PC 侧脚本会
    `ModuleNotFoundError: No module named 'torch'`。PC 侧一律用绝对路径
    `G:\ai\python11\python.exe`。
34. **相机帧周期就是延迟地板。** GC0308 在 240×240 下约 **32 ms 一帧**；
    `/capture` 实测 31 ms 时已经贴着上限，再优化代码也不会更快，
    只能改采集模式（分辨率 / 帧率 / 传感器窗口）。
35. **串口日志属于测量误差的一部分。** 115200 下一行 70 B ≈ **6 ms 阻塞**。
    在热路径里打日志等于给自己加 6 ms 噪声；采样 1/8，并且放在响应发出之后。
36. **测执行机构，"指令值"和"到位值"是两个量。** 我测头部舵机时只等 1.2 秒就
    读回，得出"行程只有 ±7°"的结论 —— 错的。等 8 秒再读，实际能到 −28.9°。
    正确做法：每个角度等它饱和（连续两次读回不变），并且**同时记录
    `head.target` 和 `head.q`**，两个数一起看。
37. **同一个指令重复给，位置可能不一样。** `head=0` 两次分别停在 −12.7° 和
    +9.2°。**重复性 ±10°** 的执行机构，配 `head_deadband_deg: 2.0` 的死区没有意义。
    凡是"控制器输出"对"执行机构"的映射，都要先量重复性再定死区。
38. **分类器输出在真实场景里会"高置信度地错"。** 手势分类器把清晰可见的拳头
    判成 thumbs_up，置信度 0.77。**置信度不是正确性的保证**，必须把模型
    实际看到的裁剪图（这次是 64 px 手图）dump 出来肉眼看，
    否则会一直在错误的数字上做决策。
39. **别搞混 `evidence/live/` 和 `evidence/sessions/`。** 前者是我用后台脚本
    偷录的（用户不知道何时开始、也没有动作剧本），**只能当背景图**；
    后者是用户在网页采集台上自己点开始录的受控数据。我导错过一次，
    把 400 帧废数据当标注集导进了 Label Studio。
40. **让网络去学一个确定性规则是反模式。** 我提过"用规则检出敲击冲击、
    再让 CfC 学'两次'"——用户当场指出是脱裤子放屁。凡是你已经能用几行
    if 写死的东西，就不要训模型：多一倍数据、多一层不可解释性、
    没有任何收益。**该学的是"写不出规则"的那部分（像素→语义）。**
41. **烧录后别用启动日志的 "Compile time" 判断固件版本。** ESP-IDF 增量构建不会
    重新生成 `app_desc` 里的编译时间字符串，烧了新 app 也照样打印旧日期。
    判断运行版本用功能行为，或者重新 `idf.py build` 再烧。
42. **同一个 COM3 不能同时被两个串口脚本打开**，第二个会 `PermissionError(13)`。
    要复位 + 抓启动日志，就在同一个脚本里先拉 EN 再读；空口 PASSIVE 监听前
    务必先把 DTR/RTS 置低再 `open()`（见第 30 条）。
43. **改热点 SSID / 密码 = 必须重新给机器人配网。** 凭据在机器人 NVS 里，
    改了电脑热点它只会一直扫旧名字。配网：开机单击 BOOT（状态还是 Starting
    且未连上时），或长按 BOOT >3 s 擦 NVS 后重启；机器人开 AP `RIG-Omni-9BE0`，
    电脑连它 → `http://192.168.4.1` 填新 SSID/密码。细节在
    `dataset/docs/ROBOT-SETUP.md`。
44. **仓库是 public，`dataset/` 里的视频会公开。** 导出 take 之前先确认内容
    可以公开（操作者的手、桌面、房间背景都会进仓库），别把带隐私的画面推进去。

---

## 6. 当前状态（2026-11-03 夜）

### 6.1 已完成

- 感知链 + 场景信号层 + 教师状态机（46 + 24 项单测）
- 观测 v2 组装（116 维，v2a/v2b 变体）
- 录制脚本存原图 + 关键点 + 事件标记（`--save-frames` / `--keys`）
- **真机链路优化完成**：端到端 121.5 → **42.8 ms p50 / 46.1 p90**，
  8.0 → **21.4 Hz**
  - Nagle 修复（每个 handler 里 `TCP_NODELAY`）：`POST /feat` 56.3 → **8.0 ms**
  - TCP 窗口 5760 → 32768（否则一帧 6.5 KB 撑爆窗口，`/capture` 退化到 92 ms）
  - 固件内静态 IP 兜底（不再依赖 PC 侧 DHCP 守护进程）
  - 传感器改**原生灰度**（GC0308 `OUTPUT_FMT=0xb1`）：`/capture` 58 → 42 ms
  - **`POST /step`** 把「送 116 维观测 + 取下一帧」合成一次往返：
    51.0 → **42.8 ms**
- 工具：`http_floor_probe.py`、`e2e_latency.py --step`、`quality_sweep.py`、
  `vlm_label.py`、`_face_size_probe.py`、`_yoloe_probe.py`
- **2026-11-05：WiFi 自动重连修复已烧录并验证**（详见 `WIFI-AUTOCONNECT-FIX.md`）：
  Hover 扫描间隔 5–15 s、60 s 不超时不进配网；热点晚开 18.3 s 自动上线，
  已连接时热点断开 85 s 后 7.7 s 自动重连。烧录前整片备份在
  `_fw_backup_20261105_old.bin`（4 MB，可回滚）。
- **2026-11-05：录完直接标的合并服务**在 `pc/tools/video_annotator.py`（8780），
  录制走 `GET /capture`（只读），发布 H.264 MP4（ffmpeg）/ WebM 兜底，
  原图帧 + `frames.jsonl` 旁车目录保留。见 `VIDEO-ANNOTATOR.md`。
  同日晚加**暂停/继续录制**（暂停时长不计入视频时间轴，暂停时可用预览调机位）
  和 **15 fps 默认**（实测 `/capture` p50 42.3 ms，链路天花板约 23.6 fps；
  10 fps 每帧 100 ms 可能漏掉 50–100 ms 的敲击接触瞬间）。
  接口：`GET /api/recording`、`POST /api/record/{start,stop,pause,resume}`、
  `GET /api/preview`。
- **2026-11-07：数据集样例入库**在 `dataset/`（2026-11-07 的 4 条 take，2026-11-08 全部补齐标注）：
  `robot_20261107_204337_5ca36d`（12 区间）作为**标注教材**，另外三条也已标注（26/12/27 区间）；
  2026-11-05 的两条调试片有意排除。只放 MP4 + `annotation.json` + `meta.json`，原始 JPEG 帧不入库。
  导出脚本 `pc/tools/export_take.py`（`--since` / `--exclude` / `--index`）；
  新电脑/队友上手先读 `dataset/docs/ROBOT-SETUP.md`，标注规范见
  `dataset/docs/COLLECTION-AND-ANNOTATION.md`。
- **2026-11-08：区间标注 → 逐帧敲击检测的流水线跑通**（冒烟）：
  `pc/tools/import_annotations.py` 把服务导出 JSON 拆回 per-video 标注；
  `pc/train/interval_data.py` 做标签合并 + 区间→帧 + 窗口切分；
  `pc/train/train_knock.py` 按 take 留一训练（frame 基线 / CNN+GRU），
  输出帧级 F1/AUROC 与事件级 F1。4 条 take、77 个区间，留一结果：
  GRU 帧级 F1 **0.613**、AUROC **0.847**、事件级 F1 **0.316**（frame 基线更低）。
  这只说明链路能跑、标签能学，不能当泛化结论；阈值跨 take 会漂移。
  说明与配置见 `dataset/docs/TRAIN-PIPELINE.md`。

### 6.2 这一轮的实测拆分（240×240 灰度，q=80）

| 阶段 | 耗时 | 归属 |
|---|---|---|
| 等相机下一帧（`fb_get`） | ~10 ms | **传感器帧周期 ~32 ms，地板** |
| 软件 JPEG 编码 | ~21 ms | 机器人（S3 无 JPEG 硬件编码器） |
| HTTP 传输（5.3 KB） | ~9 ms | 已打 `TCP_NODELAY` |
| **`POST /step` 合计** | **33.3 ms** | 一次往返 |
| YOLO26n-pose + 特征（PC） | 9–10 ms | GPU |
| **端到端** | **42.8 ms p50** | **21.4 Hz** |

**43 ms 里机器人占 33 ms、PC 占 10 ms；而机器人的 33 ms 中 21 ms 是 JPEG
编码、~10 ms 是等下一帧——已经不是 HTTP 的锅了。** 再往下要动的是采集模式
（分辨率 / 帧率 / 传感器窗口），不是传输层。

### 6.3 卡在物理条件

| 项 | 卡在 | 谁来做 |
|---|---|---|
| 手势分类器重训 | 需要桌面域的握拳/张掌/无手势数据（HaGRID 域差太大） | 用户采 + agent 训 |
| 桌面场景数据采集 | 需要真人按剧本做动作 | 用户 |
| FOV / `dist_k` 标定 | 需要 A4 纸 + 一个已知距离 | 用户 |
| 640×480 采集 | 固件改动 | agent |

### 6.4 一个必须校准的矛盾

`FOV_H = 60°` 和 `dist_k = 0.84` **不能同时成立**（差 1.75 倍）：

```
FOV_V=60°  ⇒ f_px = 207.8 ⇒ dist_k = 1.47
dist_k=0.84 ⇒ f_px = 118.6 ⇒ FOV_V = 90°
```

`DATASET-PLAN.md` 里"1–2 米对应 person_h 0.42–0.84"是从 0.84 反推的，
**不是实测**。如果真实 FOV 是 90°，手在 1.5 米只有 9 像素不是 14 像素。

**采完第一批数据第一件事**：放一张 A4 纸（29.7 cm）竖着，在已知距离各录 20 帧，
用框高反解 `f_px`。图现在能存了，这是纯离线计算。

---

## 7. 常用端点

| 端点 | 方法 | 用途 |
|---|---|---|
| `/status` | GET | 全量遥测（IMU / 轮 / 头 / CfC / 堆） |
| **`/step`** | **POST** | **主链路：body = 116 维观测，响应 body = 下一帧 JPEG，`X-Cfc` 头带 CfC 回执** |
| `/capture` | GET | 单帧 JPEG（240×240 灰度，约 5.3 KB；`?q=10..95` 调质量） |
| `/feat` | POST | 只上传 116 维观测（旧的两段式链路，保留做对照） |
| `/cfg` | GET/POST | 运行时参数（`cfc_hz` / `vx_max` / `stale_ms` / `head_lpf` …） |
| `/model` | POST | blob 热更新（`?slot=N`） |
| `/mode` | POST | 控制源切换（`?src=manual`） |
| `/cmd` | GET | 手动控制 |

> ⚠️ 所有 HTTP 测试都要加 **`curl.exe --noproxy '*'`**，本机有代理会返回 502。

---

## 8. 工作纪律（用户明确要求过）

1. **工程判断全部自己做，不要停下来等确认。** 原话："你不要去等我下指令，
   你自己想办法去干"、"别问我"。
2. **遇到必须物理介入的记下来留给用户**（比如机器人要立起来、USB 要重插）。
3. **不要拿当前占位模型的性能去设计补丁。** 用户会自己采数据集重训。
4. **诚实高于漂亮。** 自纠正在这个项目里是被鼓励的，掩盖不是。
5. **要具体数字、命令、文件路径，不要空话。**
6. **不要只做字面指令**，要有探索性。
7. **不要落地未验证的固件。**
8. 机器人要保持安静（`output volume forced to 1`）。

---

## 9. 下一步（按优先级）

> 2026-11-03 21:06 起，科研主线转为 `RESEARCH-VideoCfC.md` §11：**纯数学建模与推导**。
> 用户要求上下文压缩后先研究各类网络的数学设计理由，当前不得跑训练、数据探针、模型测试或真机实验。

1. **定义流式问题与性质**：输入时间戳序列、因果输出、隐状态语义、采样细化/稀疏采样、时间一致性和误差度量。见 `RESEARCH-VideoCfC.md` §2、§11。
2. **推导线性连续状态转移与半群关系**，再与按观测次数更新的普通离散 RNN 比较。明确证明所需假设和反例边界。
3. **从原论文方程分析 CfC**：核对 LTC 积分近似、CfC 门控、`timespans` 语义；不把近似闭式更新说成通用 ODE 精确解。
4. **分析门控循环、Transformer、SSM/Mamba 的数学骨架**，讨论梯度路径、并行计算、状态/缓存及外部时间戳语义。所有性能优劣保持未定。
5. 完成数学推导并与用户讨论后，再决定是否写原型和设计实验。没有后续明确授权时不训练或访问真机。

### 9.1 暂停中的工程工作

- 冻结视觉特征的 clip/online 探针、训练器、Stream-CfC 实现和核心消融。
- 高帧率相机及双机同步采集。
- 640×480 固件改动、FOV / `dist_k` 标定和采集流水线改造。
- Label Studio 的 331 帧标注、手势分类器重训、控制策略重训及部署验收。

以上工作不因本次数学分析自动启动；重启前先完成数学章程中的定义与推导，并按用户后续要求决定是否实验。

---

## Project Experience

已验证、可复用的项目经验。详细笔记在 `.agents/experience/`，**按需读，不必全读**。
每条要么是踩过坑修好的，要么是跑过实验确认的；不写猜测。

> **状态更新**：§9 的"暂不训练"已被 2026-11-04 的通宵任务取代。用户明确要求立刻
> 设计/训练/验证流式记忆网络，产物在 `stream_memory_research/`（437 个 run、1071 行报告）。
> 再碰这个课题**先读最后一条**，不要重复已经做过的实验。

| 经验 | 什么时候读 |
|---|---|
| [只给自己的模型调超参会制造假优势](.agents/experience/research-fair-hyperparameter-tuning.md) | 做任何"我们 vs 基线"的对比之前。实测因此撤回了一个 +14.3 点的假胜利 |
| [`status: complete` 不等于"这次配置的结果"](.agents/experience/run-config-fingerprint.md) | 写或改"按 run_id 跳过已完成实验"的 sweep 时。旧 pilot 被误当新结果，污染过一次排名 |
| [冻结视频特征缓存](.agents/experience/frozen-video-feature-caching.md) | 要在 8 GB 卡上比多个时序/融合结构时。这是 8 族模型 × 17 数据集 × 3 种子的前提 |
| [本机并发 torch 进程会伪造崩溃](.agents/experience/windows-concurrent-torch-limits.md) | 见到 `WinError 1455/1114`、`bad allocation`、`cudaErrorAlreadyMapped` 时。先查并发，再怀疑代码 |
| [流式记忆网络实测结论](.agents/experience/cfc-and-temporal-memory-findings.md) | 再碰"视频流决策网络"之前。含 CfC 的致命数字、已验证的时序任务构造法、未解决的方向 |
| [D: 盘大文件已搬到 G:（2026-11-04）](stream_memory_research/reports/DISK-OFFLOAD-2026-11-04.md) | 找不到 `runs/<run_id>/checkpoints/*.pt` 时。权重在 `G:\stream_memory_data\research_checkpoints`，`eval_manifests.py` 已加回退；D: 释放 6.15 GB |
