# 早间操作手册

> 机器人整晚横躺（`pitch≈89°`），所以**所有闭环行为都只在仿真里验证过**。
> 这份清单按优先级排，每步都给了确切命令和"看到什么算通过"。
> 预计总耗时 1~1.5 小时。

---

## ⚠️ 先读这一段：那个"电机卡住声"是崩溃重启，已经修好了

机器人昨晚整夜在**开机重启循环**里，每 ~7.5 秒一次。每次重启瞬间平衡环上电、电机被驱动一下，
听起来就像"电机卡住 + 电流声"。**不是电机故障，也不是蜂鸣器。**

**根因**（2026-11-03 早上用串口 + 二分定位查明）：
`cfc_model_init()` 里那行 `cfc_q8_simd_enabled()` 执行了**完整的 SIMD 数值扫描**
（7 组长度 × 3 条路径），把 `dbg_boot` 任务栈撑爆 → 栈溢出 → 重启。
讽刺的是源码注释明写"完整扫描交给 PC 端做，这里只跑几百字节的探测"——**注释描述的意图是对的，实现违背了它**。

**已修复、已烧录、已验证**：

| 项 | 状态 |
|---|---|
| 固件 | **我们那版，已修复**（不需要再刷） |
| 崩溃 | **0 重启 / 无 Guru**；连跑 20 次 `/status` 全部 200 |
| HTTP 端点 | `/status` `/cfg` `/model` `/gaze` `/emote` 全部可用 |
| CfC 模型 | 已加载（slot 0 / slot 1 均 `valid:1`），**未激活**（`active:-1`、`cfc_hz:0`） |
| 机身状态 | **躺着**（`/status` 报 `roll≈-88°`），需要你把它立起来 |

**所以第 0 步直接上电即可，固件不用再刷。**

---

## ⚠️⚠️ 更重要的一条：**策略还没训好，先别急着测跟随**

昨晚最后测出一件事，它改变了整个早上的优先级：

> `pc/train/synth.py::synth_feat` 生成的 64 维"视觉特征"是**编造的**（6 个场景量的随机投影 + tanh），
> 不是 YOLO 输出。实测在 200 张真实照片上（走完整感知链），
> **64 个特征维有 28.1% 落在训练包络之外，18/64 维裁切超过 50% 的帧。**

**所以"CfC 能在 ESP32 上跑"是真的，"策略能跟随人"还没被验证过——它从没学过真实特征。**

**好消息：修复流程我昨晚已经在真实图像上端到端跑通并验证了**，不需要机器人就能确认统计量问题可修（29.7% → 2.2%）。
早上只剩"把图像换成机器人摄像头"这一步。

正确的早上顺序：

| 优先级 | 做什么 | 为什么 |
|---|---|---|
| **1** | 充电 + 重新烧录 + 确认 `/status` 200 | 先把固件跑起来 |
| **2** | **录制真机数据**（下节给了确切命令） | **这是真正的关键路径** |
| **3** | 重训 CfC → 重导 blob → 上传 | 策略变成真的能用 |
| 4 | 再跑跟随 / 手势演示 | 这时才有意义 |

**不要跳过第 2 步去测跟随**——合成特征训出来的策略在真机上大概率是乱动的，那样测出来的东西没有信息量。

### 第 2 步：录制真机数据（附 mock 预演）

**录制前建议先花 20 秒跑一次 mock 预演**，确认整条录制链路是通的——不需要机器人：

```powershell
# 终端 A：起一个假机器人（服务 /capture 和 /status）
%PY% D:\robot_work\pc\train\mock_robot.py --port 8099

# 终端 B：对着假机器人录 12 秒
%PY% D:\robot_work\pc\train\session_from_robot.py --source robot ^
  --host 127.0.0.1:8099 --seconds 12 --out D:\robot_work\pc\train\data_mock
```

期望看到 `wrote ... (88 frames, obs (88, 106))` 和 `mode distribution: {1: ..., 2: ...}`。

**这条链路已经在真机器人上验证过**（2026-11-03，机器人躺着）：

| 指标 | 实测 |
|---|---|
| `/capture` | 200，7.6 KB JPEG，单次 **60 ms** |
| `/status` | 200，单次 **55 ms** |
| 录制吞吐 | **15 秒 115 帧**（≈7.7 fps） |
| 感知稳态 | 10 ms/帧 |

> ⚠️ **如果只录到 1~2 帧，说明预热没生效**。CUDA 首次推理要 **16.2 秒**（cuDNN 自动调优），
> 脚本里已加空帧预热，正常应看到 `perception warmed up in X s` 这行。
> 没有这行就是旧版脚本。

确认无误后，换成真机器人（去掉 `--host` 或指向 `192.168.137.50:80`）：

```powershell
%PY% D:\robot_work\pc\train\session_from_robot.py ^
  --source robot --seconds 90 --out D:\robot_work\pc\train\data_real --session-id real_100
```

> 如果真机器人这条路失败，**先跑 mock 判断是"代码问题"还是"网络/固件问题"**。
> mock 能过而真机不过 → 问题在机器人侧（IP、端点、相机）；两边都不过 → 问题在代码。

---

```powershell
set PY=D:\smartcar\.venv\Scripts\python.exe

# 2.1 录 3~5 段、每段 60~120 秒，换不同场景（人静止/横移/走近走出/挥手/空房间）
%PY% D:\robot_work\pc\train\session_from_robot.py ^
  --source robot --seconds 90 --out D:\robot_work\pc\train\data_real --session-id real_100

# 2.2 确认录到了东西（应看到 obs (T,106)、mode 分布里有 1=TRACK）
dir D:\robot_work\pc\train\data_real\*.npz

# 3.1 训练（统计量会自动变成真实的）
%PY% D:\robot_work\pc\train\train_cfc.py ^
  --data-dir D:\robot_work\pc\train\data_real --out-dir D:\robot_work\pc\train\out_real --epochs 40

# 3.2 导出 blob
%PY% D:\robot_work\pc\train\export_blob.py ^
  --npz D:\robot_work\pc\train\out_real\policy.npz ^
  --out D:\robot_work\pc\train\out_real\model_real.blob

# 3.3 自检：应该从 RISK 变成 OK
%PY% D:\robot_work\pc\bench\obs_domain_check.py ^
  --n 100 --policy D:\robot_work\pc\train\out_real\policy.npz

# 3.4 上传到机器人
curl.exe -X POST "http://192.168.137.50/model?slot=0" ^
  --data-binary "@D:\robot_work\pc\train\out_real\model_real.blob" ^
  -H "Content-Type: application/octet-stream"
curl.exe -X POST "http://192.168.137.50/mode?src=cfc&slot=0"
```

**3.3 必须显示 `OK -- envelope holds` 才继续**。如果还是 RISK，说明录的数据太少或场景太单一。

### 3.5 上传后调一个参数，把"只动头"找回来

训练出来的策略**结构是对的**（头部分担比例随误差递减：82.7% → 80.4% → 76.4%），
但**把教师的硬死区平滑掉了**——小误差时教师要求 vyaw 精确为 0，策略却输出约 2°/s。

仲裁器里已有 `vyaw_stop_eps`（命中即吸附到零，默认 0.5°/s），提到 **3°/s** 就能补回来：

```powershell
curl.exe -X POST http://192.168.137.50/cfg -H "Content-Type: application/json" -d "{\"vyaw_stop_eps\":3.0}"
```

**不用改固件、不用重编译。** 调完再让人站在正前方小幅移动，用手感受机身是否真的不动。

### 3.6 还有一个已验证但**需要改固件**的修复：mode 去抖

测出来策略的 ACK 会**抖动**：教师给 5 个 12 帧的脉冲，策略给 15 个 3 帧的短脉冲
（切换精确率只有 42.9%）。这是因为网络在模式边界附近来回越阈值。

**离线验证过修法**：对 `mode` 通道做"新模式要连续 6 帧占优才接受"的去抖 →

| 去抖 | 切换精确率 | 切换召回率 | ACK 脉冲 |
|---|---|---|---|
| 无 | 42.9% | 90% | 15（3 帧） |
| **6 帧** | **88.9%** | 80% | **4（11 帧）** ← 教师是 5（12 帧） |

落点：`control_arbiter.c` 的 `safety_shell()` 之后，**只在 TRACK/ACK 之间去抖，
IDLE（含安全壳强制归零）必须立即生效**，否则会拖慢安全响应。建议初值 6 帧（0.75 秒），默认关闭。

**我没改固件源码**——无法烧录验证的改动在凌晨留风险大于收益。这段等你醒了判断要不要加。

> 提醒：`session_from_robot.py` 里本体感知是从 `/status` 读的，所以**录制时机器人必须在线且 `/status` 正常**。
> 如果机器人还没站起来（横躺），录到的 pitch 会是 90°，训出来的策略没见过直立状态——**先立起来再录**。

### 如果一时录不到真机数据——panning 数据在**数据量够**时是可用的

我用**真实照片横向平移**造了数据（真实像素/关键点/特征/角度，只有时序是人造的）。
**关键是源图数量**：

| 源图数 | 留出集 head MAE | ACK 召回 | 判断 |
|---|---|---|---|
| 30 张 | 8.5° ± 3.9° | 0% | ❌ ≈ 基线（过拟合） |
| **160 张** | **1.11° ± 0.08°** | **70%** | ✅ **比基线（7.79°）好 7 倍** |

```powershell
%PY% D:\robot_work\pc\train\session_from_realpan.py --n-images 200 --seed 0 ^
  --out D:\robot_work\pc\train\data_pan_all
# 最后 40 段留出（同一次打乱，无重叠）
%PY% D:\robot_work\pc\train\train_cfc.py --data-dir D:\robot_work\pc\train\data_pan_all ^
  --out-dir D:\robot_work\pc\train\out_big --epochs 80 --seed 0
%PY% D:\robot_work\pc\train\export_blob.py --npz D:\robot_work\pc\train\out_big\policy.npz ^
  --out D:\robot_work\pc\train\out_big\m.blob
```

**产物已生成好**：`out_big/m.blob`（160 段训练，可用的）。直接上传就能看到效果。

> ⚠️ **但它的输入包络不合格**（40–48% 裁切）：panning 生成把人在画面里的占比和位置分布改了。
> 我试过用"机器人 240×240 取景"重测——**没有改善，反而更差（47.7%）**，所以不是方图/宽幅的问题。
> 结论：**panning 数据只解决了"有时序标签"，没解决"分布对"**。真机数据才能同时满足两者。
> 所以真机数据仍是最终必需项，panning 只是"在没有真机数据前的可用替代"。

> 教训记一笔：我第一次用 30 张图训出 2.80° 就想报成果，换种子重测发现是 **8.5° ± 3.9°**（幸运抽样）。
> **这类实验必须跑多 seed 取均值±标准差**，否则会把运气当能力。

（如果想先看看管线通不通，可以用 `/status` 确认各端点、用 `/capture` 看图像，这些跟策略无关。）

---

然后按下面顺序走：充电 → 烧回我们修好的固件 → 再做后面所有验证。

```powershell
# 烧回修好的固件（app 分区，不动 NVS/assets）
G:\esp\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe -m esptool ^
  --chip esp32s3 -p COM3 -b 460800 --before default_reset --after hard_reset ^
  write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m ^
  0xd000 D:\robot_work\rig_omni\build\ota_data_initial.bin ^
  0x20000 D:\robot_work\rig_omni\build\rig-hover.bin
```

烧完等 20 秒，`curl.exe http://192.168.137.50/status` 能返回 JSON 才算起来。

### ⚠️ 怎么判断"烧录真的生效了"

**唯一可靠的判据是 `/status` 返回 200。** 别用"能 ping 通"或"能打开网页"判断——**原厂固件也有 HTTP 服务**。

原厂的 `hover_debug_server` 只注册了 **`/`、`/api/data`、`/api/set`** 三个端点。
所以如果你看到"这三个能用、其他全 404"，**那不是新固件的 bug，是板子还在跑原厂固件**——烧录没生效。

排查顺序：
1. 确认 otadata 也写了（只写 app 不写 otadata，bootloader 可能还启动旧槽）
2. 用 `esptool.py flash_id` 确认能连上
3. 读串口看启动日志里有没有 `[RIG-CUSTOM]` 或 `Debug panel` 字样

**如果又出现"反复重启"**：读串口看 `stack overflow in task xxx` 里的任务名，把那部分逻辑挪到独立任务里。
（上一版就是 `sys_evt` 溢出：调试面板的启动代码跑在 4 KB 的 WiFi 事件任务栈上。）

---

## 0. 前置检查（2 分钟）

```powershell
# 机器人应该在 192.168.137.50（PC 的 ICS 热点网段）
ping 192.168.137.50
curl.exe http://192.168.137.50/status
```

**通过标准**：返回含 `"cfc"` 字段的 JSON。

**如果 ping 通但 HTTP 全失败** → 是 httpd socket 耗尽的老问题，等 10 秒重试，或断电重启。

**如果完全 ping 不通** → DHCP 掉了。我留了自动重启的守护进程，但如果你重启过电脑它会没了。手动拉起：
```powershell
cd $env:TEMP\robot_probe
python dhcpd_long.py
```
（这个脚本自己跑满 4 小时会退出，所以别只开它一次）

---

## 1. 把机器人立起来（最关键的一步）

⚠️ **你现在刷的固件是"官方 v3.8.0 + 我们加的端点/仲裁器"**，控制律一个字没改。
**但**：`arbiter_tick_2ms()` 已经接进 `robot_control()`，默认 `src=MANUAL` 所以不会接管——但请确认：

```powershell
curl.exe "http://192.168.137.50/cfg"
```
**必看 `"src":0`**。如果是 2，说明 CfC 在接管控制，先切回手动：
```powershell
curl.exe -X POST "http://192.168.137.50/mode?src=manual"
```

**立起来后观察 30 秒**：
- 机器人应该自己站住（`pitch` 在 ±7° 内）
- `curl.exe http://192.168.137.50/status` 里 `robot_state` 相关行为正常

**如果站不住** → 直接跳到第 2 步（那正是我们怀疑的地方）。

---

## 2. 验证 LQR 符号假设（最高优先级，30 分钟）

**背景**：仿真的结论是 `lqr_k=[1300,5,25,0.4]` 的符号可能让平衡环结构上不可稳定。
但**这是假设不是结论**——符号极性只以乘积形式可观测，仿真选了其中一组。

### 测试步骤

```powershell
# 记录基线
curl.exe http://192.168.137.50/status

# 试 K0 取负（位置项）
curl.exe "http://192.168.137.50/api/set?i=10&v=-1300"
# 观察 10 秒：是收敛还是发散？

# 恢复
curl.exe "http://192.168.137.50/api/set?i=10&v=1300"
```

### 判读

| 现象 | 结论 |
|---|---|
| K0 取负后**明显变稳** | 仿真对了 → 出厂增益确实有问题，值得报告上游 |
| K0 取负后**更差/立刻倒** | 仿真符号错 → 出厂增益是对的，仿真结论作废 |

**无论结果如何都告诉我**——这是本次开发唯一一个"能证伪整套仿真"的判据。

⚠️ **测试时手扶在机器人旁边**，随时准备接住。

---

## 3. 验证 yaw 阻尼（15 分钟）

**背景**：仿真测出航向环是纯 P 控双积分，**ζ≈0.08**，机身一直小角度振荡，
把"只动头"完全淹掉——头身分离比只有 0.71（理想应远小于 1）。

已加 `k_dyaw` 参数，**默认 0**（行为不变）。`/cfg` 里当前新增 5 个字段，全部实测存在：

| 参数 | 当前值 | 含义 |
|---|---|---|
| `k_dyaw` | **0** | 航向环 D 项增益（0 = 原行为） |
| `vx_rate_max` | 1.0 | 前进指令每秒最大变化量（m/s²） |
| `vyaw_rate_max` | 400 | 转向指令每秒最大变化量（°/s²） |
| `vx_stop_eps` | 0.005 | 前进"吸附到零"的判据（**故意独立于 clip 下界**，见下） |
| `vyaw_stop_eps` | 0.5 | 转向"吸附到零"的判据 |

> 后两个参数的存在是有原因的：smartcar 那个项目踩过坑——如果用 `target <= bounds[0] + eps`
> 判断停车，一旦将来允许负速度，倒车指令会被吸附成满速倒车。所以吸附点必须独立于 clip 下界。

```powershell
curl.exe http://192.168.137.50/cfg          # 确认 k_dyaw
curl.exe -X POST http://192.168.137.50/cfg -H "Content-Type: application/json" -d "{\"k_dyaw\":0.5}"
```

### 观察方法
让人站在机器人正前方**小幅左右移动**（±10°），看两个东西：

1. **轮子是不是还在转**（应该有明显减少）
2. 用手感受机身是不是还在"抖"

### 仿真给出的推荐值（实测过，但**符号必须先验**）

| k_dyaw | 阻尼比 ζ | 头身分离比 | 小误差跟踪误差 |
|---|---|---|---|
| 0（当前） | 0.08 | 0.74 | 6.07° |
| **−0.1** | 0.17 | **0.91** | 6.04° |
| **−0.2（推荐）** | 0.26 | **0.90** | 6.03° |
| −0.5 | 0.53 | 0.81 | 6.05° |
| +0.2 | **−0.10（发散）** | 0.23（假象） | 8.67° |

⚠️ **起阻尼作用的是负值**。`ζ=(1.2−12.814·k)/14.27`，`k>0.094` 就发散。
**填反了就是正反馈，比不加更糟。** 先按第 2 步的手段验 yaw 极性，再决定符号。

教师控制器的收益是数量级的：ε 11.7°→6.4°、成功率 77%→97%。

### 一个必须知道的事实：「只动头」做不到

- 加阻尼后分离比 0.71 → **0.90**，仍远达不到"机身基本不动"（≥5）
- **主因不是阻尼**，是**策略自己 62% 的时间在下 vyaw 指令**
- 消融把 vyaw 压到 0 → 分离比 **5.49**，但代价是 ε 25°、只有 72% 时间人在视野内
- **几何上限**：头 ±45°、FOV ±30° → |方位| ≲ 30° 之外，头单独覆盖不了，机身必须转

**所以这是一个取舍，不是 bug。** 演示时应该讲清楚："小角度只动头、大角度必须转身"。

---

## 4. 加载 int8 模型（2 分钟）

```powershell
# 上传训练好的 int8 模型（40.8 KB，比 fp32 快 2.2 倍）
curl.exe -X POST "http://192.168.137.50/model?slot=0" ^
  --data-binary "@D:\robot_work\pc\train\model_slot0_int8.blob" ^
  -H "Content-Type: application/octet-stream"

curl.exe -X POST "http://192.168.137.50/mode?src=cfc&slot=0"
curl.exe http://192.168.137.50/status
```

**通过标准**：`cfc.us` 在 **1400~1900** 之间（fp32 是 3100~3900）。

---

## 5. 第一次真机跟随测试（20 分钟）

⚠️ **先确认第 1 步通过了**（能站住）。然后：

```powershell
# 让 PC 端感知链跑起来（另开一个终端）
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\perception\run_perception.py
```

### 测试序列
1. 人从左侧进入视野 → **只应动头，轮子不动**
2. 人继续左移超过 25° → 机身转，头逐渐回正
3. 人挥手 → 机器人应低头 + LCD 表情
4. 人走出视野 → 3 秒后头回中

### 记录什么
- 哪些步骤失败了
- `curl.exe http://192.168.137.50/status` 里的 `cfc.feat_age_ms`（视觉延迟）
- 有没有 `cfc.dropped` 增长

---

## 6. 校准特征统计量（15 分钟，如果第 5 步表现差）

**症状**：跟随很迟钝、动作不连贯、或者完全不响应。

**原因**：`input_mean/std` 是在**合成观测**上统计的，真机分布不同。4σ 夹取后可能大面积饱和。

**诊断**：
```powershell
curl.exe http://192.168.137.50/status    # 看 cfc 块里有没有 clip 相关字段
```

**救法**（不用重训）：用真机数据重算统计量，重导 blob，重新上传。详见 `D:\robot_work\pc\train\README.md`。

---

## 6.5 最有价值的下一步：换传输层（不用先做，但要知道）

我把整条链接起来测了端到端，结论和预期完全不同：

| 项 | 实测 |
|---|---|
| 端到端延迟 | **121.5 ms** |
| 其中两次 HTTP 往返 | **~108 ms（90%）** |
| 其中全部计算 | **12 ms（10%）** |
| HTTP 吞吐上限 | **16 请求/s**（单 worker 串行，并发不帮忙） |
| 由此的帧率上限 | **8 fps**（每帧 2 次请求），实测 8.3 Hz |

**所以整晚的算子优化（推理 4.04ms → 1.4ms）只省了 2.6ms，占预算 2%。**
真正有效的是：特征上行改 **UDP 单向**（−55ms）、图像下行改 **MJPEG 流**（−40ms），
端到端可降到 **30~40 ms**。

完整设计见 `TRANSPORT-REDESIGN.md`。

**这条不需要今天做**，但如果你要写报告/答辩，它是很好的一个点：
> "我先做了性能剖析，发现优化方向和直觉相反，于是及时止损换方向。"

---

## 我留给你的东西

| 想看什么 | 去哪 |
|---|---|
| 一分钟看懂 | `README.md` |
| 全部实测数字 | `RESULTS.md` |
| 主报告 | `TECHNICAL-REPORT.md` |
| 一条命令跑完所有验证 | `python run_all_tests.py` |
| 真机压测 | `python pc\tools\soak_test.py --seconds 600` |

**最后一次全量验证：8/8 套件通过。**

---

## 诚实提醒

这次开发**验证了"算得对、跑得快、想得清"，但没有验证"动得好"**。
机器人整晚横躺，所有闭环结论都来自仿真，而仿真里：

- 质量、惯量、质心高、舵机带宽、航向阻尼是**估计值**
- 三个符号极性存在**未解歧义**（第 2 步就是去解决它）
- LQR 增益、6 ms 周期、±45° 限位、FOV 是**从源码抄的**

所以第 1~3 步不是"验收"，是**探路**。真实的机器人永远比模型多一点东西。
