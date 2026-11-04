# pc/train —— 训练 + 模型导出链（工作流 D）

> 一句话：`合成数据 → 训练 CfC → 导出 model_slot0.blob → **用固件原版的 cfc_model.c**
> 在 PC 上加载并逐帧对拍`，一条命令验收，判据 `max_abs_err < 1e-5`，实测 **2.4e-06**。

```bat
:: 唯一验收命令（打印 PASS，退出码 0）
D:\smartcar\.venv\Scripts\python.exe D:\robot_work\pc\train\test_e2e.py
```

## 1. 文件

| 文件 | 作用 |
|---|---|
| `obs.py` | 106 维观测组装器。**逐下标**对齐固件 `cfc_task.c:323 build_obs()`，不是照抄 TECH-PLAN 的表。带 `DistEstimator`（复刻固件的距离低通）与 `DomainGuard`（复刻 `cfc_model.c:936` 的域外守护）。`--layout` 打印下标表，`--self-test` 跑自检。 |
| `synth.py` | 合成数据生成器。人造"人走近 → 绕行 → 挥手×2 → 走出视野 → 回来"的剧本，用**真教师控制器**打标签，闭环积分机体状态（观测里 `obs[100..103]` 是上一帧动作）。输出 `data/synth_sNNN.npz`。 |
| `train_cfc.py` | 训练。ncps `CfCCell` + 保留门（`ret` 初值 −3.0）+ 5 个头。按**会话**划分训练/验证，每 epoch 打印 λ 均值。产出 `checkpoint.pt` + `policy.npz`。 |
| `export_blob.py` | **最重要**。`policy.npz → 固件 blob`，头里 18 个字段逐字节按 `cfc_model.h` / `cfc_model.c` 的 `H_*` 硬偏移打包；CRC32 = zlib 口径（自身 4 字节当 0）。`--self-test` 自检，`--check-offsets` 三方核对，**`--int8` 产出 int8 权重 blob**。 |
| `dump_offsets.c` | 用 `offsetof()` 打印头里每个字段的**真实偏移**（打包与非打包两种结构各一份），与 Python 打包代码、固件 `#define H_*` 三方比对。 |
| `hoststub/` + `host_shim.c` | 把 ESP-IDF 的 `esp_partition/esp_heap_caps/esp_timer/esp_log/FreeRTOS` 换成 PC 替身（RAM 模拟 flash：写=AND、擦=0xFF、擦除需 4 KB 对齐）。**目的是能直接编译固件原版 `cfc_model.c`**，而不是重写一份解析器。 |
| `host_blob_main.c` | PC 侧驱动：`info`（upload+select+把解析出的字段/层指针偏移打成 JSON，int8 blob 额外给 `off8`）、`run`（逐帧推理）、`reject`（期望被拒绝）、`selftest`（LUT/int8 数值自检）。 |
| `bench.py` | **真机实测**。拉 `/status` 的 `cfc.us` 取分位数 + 饱和法（`cfc_hz` 顶到 1000，固件钳到 200）；`--blob/--upload` 传模型。绕开系统代理（否则 502）。 |
| `bench_pct.py` | 真机分位数专用：60 次 `/status` 采样给 p0/p10/p50，并把分段耗时（norm/proj/cell/heads）一起打出来。**别用 `/bench` 端点**，它在 httpd 任务里跑推理会崩（见 §4.7）。 |
| `quant_explore.py` | 选型研究：在真实会话上比较 W8 / A8 / W8A8 几种量化组合的端到端误差（numpy）。 |
| `precision_ab.py` | **精度 A/B**：float32 / 改造前 15-bit int8 / 新 W8A8，三份都走固件原版 C 代码，输出每个头的最大绝对误差 + 物理量换算。旧源码快照放 `build/old_fw/`。 |
| `flash_chunked2.ps1` | CH343 串口在长写时会掉线，这个脚本按 32 KB 分片、多轮重试把 app 刷进去。 |
| `ctools.py` | MSVC 环境 + 编译封装。`/O2 /fp:precise`，绝不用 `/fp:fast`。 |

产物：`checkpoint.pt`、`policy.npz`、**`model_slot0.blob`**、`blob_manifest.json`、`train_summary.json`、`data/*.npz`。

## 2. 跑完整链路（4 条命令）

```bat
set PY=D:\smartcar\.venv\Scripts\python.exe
set T=D:\robot_work\pc\train
%PY% %T%\synth.py  --sessions 8 --duration 40 --out-dir %T%\data
%PY% %T%\train_cfc.py --data-dir %T%\data --epochs 6
%PY% %T%\export_blob.py --npz %T%\policy.npz --out %T%\model_slot0.blob
%PY% %T%\test_e2e.py            :: 顺带把 train\model_slot0.blob 也过一遍固件加载器
```

每个脚本都有 `--help` 和合理默认值。

## 3. blob 布局（实测，明早对固件用这张表）

**总 133388 字节 = 头 2256 + payload 131132，CRC32 = 0x101EF9CB。**
（头部从 2240 长到 2256，腾出 16 字节放 `layer_prec[16]`；payload 与旧格式逐字节相同。）

头字段（`cfc_model.c` 的 `H_*` 偏移，`dump_offsets.c` 用 `offsetof` 复算一致）：

| 字段 | 偏移 | 字节 | 值 |
|---|---|---|---|
| magic | 0 | 4 | `0x4D434643` ('CFCM') |
| ver | 4 | 2 | 1 |
| flags | 6 | 2 | `0x000B`（bit0 投影层 + bit1 保留门 + bit3 有 layer_prec 表） |
| input_dim | 8 | 2 | 106 |
| cell_in_dim | 10 | 2 | 64 |
| hidden_dim | 12 | 2 | 64 |
| backbone_units | 14 | 2 | 64 |
| n_heads | 16 | 2 | 5 |
| reserved | 18 | 2 | 0 |
| head_dim[8] | 20 | 16 | `[1,1,4,8,1,0,0,0]` |
| head_act[8] | 36 | 16 | `[1,1,2,2,1,0,0,0]`（0=linear 1=tanh 2=softmax） |
| layer_off[16] | 52 | 64 | 见下表 |
| layer_len[16] | 116 | 64 | 见下表 |
| input_mean[256] | 180 | 1024 | 前 106 个有效，其余 0 |
| input_std[256] | 1204 | 1024 | 前 106 个有效（最小 1e-2），其余 1 |
| clip_sigma | 2228 | 4 | 4.0 |
| ts | 2232 | 4 | 1.0 |
| crc32 | 2236 | 4 | `0x101EF9CB` |
| layer_prec[16] | 2240 | 16 | 全 0（这份是 float32 blob） |

层表（payload 从 2256 起，按层 id 顺序紧密排列，全部 4 字节对齐）：

| id | 名字 | off | len | float 数 |
|---|---|---|---|---|
| 0 | bb_w | 2256 | 32768 | 8192 = 64×(64+64) |
| 1 | bb_b | 35024 | 256 | 64 |
| 2 | ff1_w | 35280 | 16384 | 4096 = 64×64 |
| 3 | ff1_b | 51664 | 256 | 64 |
| 4 | ff2_w | 51920 | 16384 | 4096 |
| 5 | ff2_b | 68304 | 256 | 64 |
| 6 | ta_w | 68560 | 16384 | 4096 |
| 7 | ta_b | 84944 | 256 | 64 |
| 8 | tb_w | 85200 | 16384 | 4096 |
| 9 | tb_b | 101584 | 256 | 64 |
| 10 | ret | 101840 | 256 | 64 |
| 11 | proj_w | 102096 | 27136 | 6784 = 64×106 |
| 12 | proj_b | 129232 | 256 | 64 |
| 13 | heads | 129488 | 3900 | 975 = 15×64 + 15 |

`heads` 层内部：`head_w[0..4]` 依次 1×64、1×64、4×64、8×64、1×64 个 float，
然后 `head_b[0..4]` 依次 1、1、4、8、1 个（固件 `blob_parse()` 就是这么切的）。
固件侧 `param_count` 读出来 = **32719**（注意固件公式**不含 ret** 那 64 个）。

## 4. int8 权重路径（`export_blob.py --int8`）：bulk W8A8 + 输出层 fp32

固件侧 `flags` **bit3**（有 `layer_prec` 表）+ **bit2**（至少一层 int8）一起打开。
默认仍是 float32，精度逐层写在头里，选择权在导出侧。

```bat
%PY% %T%\export_blob.py --npz %T%\policy.npz --out %T%\model_slot0_int8.blob --int8
%PY% %T%\export_blob.py --npz %T%\policy.npz --out x.blob --int8-layers ta_w,tb_w
```

### 4.1 逐层精度（`layer_prec[16]`）

头部从 2240 长到 2256，末尾追加 16 个字节的表：`0=float32 1=int8`，一个层 id 一
个字节；**已有字段的偏移一个都没动**（`layer_off` 本来就是绝对偏移，所以老 blob
的 payload 起始位置 2240 仍然有效）。

| 位 | 含义 |
|---|---|
| `flags` bit2 | 至少一层 int8（老固件只看这一位） |
| `flags` bit3 | 头里有 `layer_prec` 表。**必须靠它判断**：老 blob 那 16 个字节是 `bb_w` 载荷的开头，不是全 0 的表 |
| 表里 =1 | 只允许 `bb_w` `ff1_w` `ff2_w` `ta_w` `tb_w` `proj_w`；`heads`/偏置/`ret` 标 1 会被加载器拒绝（不是当 float32 跑） |

老 blob（bit3=0）按老规矩解析：bit2=1 → 六个权重矩阵 int8、其余 float32，只需要
2240 字节头。`test_e2e.py [11]` 用改造前导出的两份真 blob 守这条。

### 4.2 量化方案

| 项 | 做法 |
|---|---|
| 权重 | **逐输出通道对称 int8**：`W[i] ≈ q[i]·scale[i]`，`scale[i]=max|W[i]|/127`，`q` 夹在 ±127（不用 −128，保证 int32 累加不溢出） |
| 量化层 | `bb_w` `ff1_w` `ff2_w` `ta_w` `tb_w` `proj_w` 六个矩阵；**`heads` 永远 float32**（占 3% 参数，却是全部敏感度来源：int8 化它让头误差涨 2.8 倍，只省 8% 体积） |
| 激活 | **单个 int8**，`scale=max|x|/127`，每步动态量化（原先是 hi/lo 双 int8 ≈15 bit） |
| 累加 | int32：`acc = Σ q[i][j]·qx[j]`，出口一次 `acc·scale_w[i]·scale_x` 回 float |
| 激活函数 | int8 路径用 **256 点 LUT + 线性插值**（±8 饱和）；float32 路径**保持 libm**，所以 1e-5 对拍判据不受影响 |

**为什么激活退回 8 bit**：`EE.VMULAS.S8.ACCX` 只有一个 40 位累加器，每个输出行都要
"清零 → 累加 → 读回"一次会话，实测 ~300 周期，而 64 抽头的 SIMD 工作只有 ~20 周期。
一步 384 行，hi/lo 就是 768 次会话（≈0.96 ms），单 int8 只要 384 次。

代价是激活分辨率。`quant_explore.py` 在真实会话上量的（2051 帧，判据是三个 tanh 头）：

| 组合 | yaw | vyaw | vx |
|---|---|---|---|
| W8（激活精确） | 3.4e-3 | 3.3e-3 | 3.4e-3 |
| A8（权重精确） | 4.6e-3 | 6.6e-3 | — |
| **W8A8（本实现）** | **5.8e-3** | **6.6e-3** | **1.1e-2** |

**所以这是一次明确的精度换速度**：tanh 头误差从 3.4e-3 抬到 5.8e-3 量级（约 1.9×），
不是"顺便更准"。逐层精度只解决*权重*该不该量化，管不到*激活*；而 `heads` 本来就
一直是 float32，所以"输出层保精度"在固件里是既有事实，不是这次的新增收益。

### 4.3 blob 布局（int8）

`layer_len` = **元素数**（4 字节对齐），`layer_off` 处是 int8 字节，紧跟
`float32 scale[out_dim]`（`layer_off`/`layer_len` 都是 4 的倍数，所以 scale 天然对齐）。
`heads`、所有偏置、`ret` 仍是 float32。实测：

| 层 | off | len | 形态 |
|---|---|---|---|
| bb_w | 2256 | 8192 | int8 + 64 个 scale |
| ff1_w / ff2_w / ta_w / tb_w | 10960 / 15568 / 20176 / 24784 | 4096 | int8 + 64 个 scale |
| proj_w | 29648 | 6784 | int8 + 64 个 scale |
| bb_b / ff1_b…tb_b / ret / proj_b | | 256 | float32 |
| heads | 36944 | 3900 | float32 |

**总 40844 字节**（float32 版 133388；多出来的 16 字节就是 `layer_prec` 表，
payload 与旧 int8 blob 逐字节相同），CRC32 = `0x67D61012`，`flags=0x000F`。

### 4.4 精度：三个方案（都是固件原版 C 代码，2051 帧真实观测）

`precision_ab.py`：float32 blob / 改造前的 15-bit int8 blob（旧源码快照重编）/
新 W8A8 blob，同一个 policy、同一批观测、隐状态跨帧连续。

| 方案 | 体积 | head_yaw | vyaw | mode logit | emote logit | vx |
|---|---|---|---|---|---|---|
| float32 | 133388 B | 0 | 0 | 0 | 0 | 0 |
| **W8A16（改造前）** | 40828 B | **3.38e-3** | **3.32e-3** | 6.37e-2 | 6.52e-2 | **3.60e-3** |
| **W8A8（现在）** | 40844 B | **5.80e-3** | **6.60e-3** | 9.82e-2 | 1.10e-1 | **1.11e-2** |

换算成控制器真正消费的物理量（`publish()` 的 `head_max`/`vyaw_max`/`vx_max`）：

| 方案 | yaw | vyaw | vx | softmax 概率偏差 | argmax 一致率 |
|---|---|---|---|---|---|
| W8A16 | 0.152° | 0.299 °/s | 1.1 mm/s | 2.8e-2 | 0.9995 |
| W8A8 | 0.261° | 0.594 °/s | 3.3 mm/s | 4.6e-2 | 0.9995 |

**结论：精度退了一步（tanh 头 ~1.9×），换会话数减半。** 都还在 2° 控制死区以下、
也小于传感器噪声，但 5e-3 的门槛过不去——`test_e2e.py` 的判据因此从 5e-3 改到
1.5e-2（= 新地板的 2 倍，物理量 0.68°/1.35°/s/4.5 mm/s），并在注释里写明这次改动。
softmax 头的 logit 无界，绝对误差按概率算（4.6e-2 < 5e-2），2051 帧里 argmax 仍只翻
1 帧。

### 4.5 真机实测

**新内核的单步耗时没测到**：板子在刷写完新固件后失联（串口无输出、HTTP 不通、
`esptool` 连 `flash_id` 都 sync 不上），需要上电/换线后再测。下面只列同一台机器、
同一个会话里已经拿到的数（`bench_pct.py`，`/status` 的 `cfc.us`，60 采样）。

| 版本 | p0 | **p10** | p50 | 备注 |
|---|---|---|---|---|
| float32（133372 B，旧固件） | 2994 | **3679** | 4513 | 本次会话实测（README 早先记的是 3109/3555/3906） |
| int8 15-bit（40828 B，旧固件） | 1356 | **1599** | 1834 | 同上 |
| **int8 W8A8（新固件）** | — | **—** | — | 待测 |

分段耗时（旧固件 int8，p0）：normalize 38 / proj 375 / cell 768 / heads 75 µs。

**恢复后怎么测**（一条命令）：
```bat
%PY% bench_pct.py --label w8a8 --blob model_slot0_int8.blob --upload --select --n 60
```
若串口刷写再次掉线，用 `flash_chunked2.ps1`（32 KB 分片 + 多轮重试，实测 92 片
5 分钟写完）。

### 4.6 会话数减半能省多少（推算，不是实测）

会话数：proj 64 行 + cell 320 行 = 384 行/步；hi/lo 时 768 次会话。
每次会话 ~300 周期 = 1.25 µs（240 MHz）→ 减半省 **~480 µs**：

| | p0 | p10 | p50 |
|---|---|---|---|
| 旧 int8 实测 | 1356 | 1599 | 1834 |
| 减 480 µs 的推算 | ~876 | ~1119 | ~1354 |

**即 p10 达标、p50 不达标**。原因是 p50 里混着系统抖动：旧方案 p50−p0 = 478 µs，
那部分是 WiFi/LVGL/摄像头抢 PSRAM 与 cache，推理本身没变。要压 p50 得先把权重
真正钉在内部 SRAM（`/status` 新加的 `cfc.mem` 就是查这个：1=SRAM 0=PSRAM）。

### 4.7 顺带修掉的两个真机 bug

1. **开机反复重启**：`hover_debug_server_start()` 在 WiFi 事件任务（`sys_evt`，栈
   4096 B）里跑，要建 3 个任务 + 起 HTTP 服务，栈溢出踩坏相邻 TCB，崩在
   `vTaskSwitchContext`。现在把它挪到自己的 8 KB 任务（`hover_board.cc`）。
   改造前的固件同样会在这里崩（`build/bootlog.txt` 里就是）。
2. **`/bench` 端点会崩板子**：它在 httpd 任务（4 KB 栈）里调 `cfc_model_forward()`
   （栈帧 ~4 KB），一发就重启。`bench.py` 早先就绕开了它，别再按回去。

### 4.8 没结论的一处

新固件启动后 HTTP 只有前 3 个 handler 生效（`/`、`/api/data`、`/api/set`，
`/status` 起全部 404）。已把每次 `httpd_register_uri_handler()` 的返回值、以及
启动时的内部堆余量记进日志（`HoverDebug: httpd start: ...`），但板子在那之前失联，
**根因未确认**。下次上电先看这两行日志再下结论。

## 5. 四个"看起来能省但其实不能省"的地方

1. **模型是单个 CfCCell，不是 `AutoNCP(64,2)` 的 3 层链。**
   `cfc_model.h` 的 blob 里只有**一份** `cfc_cell_t` 的权重槽（14 个层 id 全属于同一个
   cell），`cfc_model_forward()` 也只调一次 `cfc_step()`。`AutoNCP(64,2)` 实际是
   `[38,24,2]` 三个串联 cell（`cfc_host/NOTES.md` 第 5 条），**当前 blob 格式表达不了**。
   固件是权威 → 训练侧收敛到"单 cell、hidden 64、backbone_units 64"，也正是
   TECH-PLAN §6.1 与 CONTRACT §5.5 的算力口径。要用 3 层链必须改固件 blob 格式（不是导出能解决的）。
2. **`backbone_units` 必须显式 64。** ncps 默认 128，算力翻倍。
3. **mask 只折 `ff1`/`ff2`，`time_a`/`time_b` 永远稠密。**
   `export_blob.fold_sparsity()` 抄的是 `cfc_host/test_parity_ncps.py:params_from_masked_cell`
   的已验收配方；本次训练是稠密 wiring（mask 为空，折叠是恒等），但配方留着，
   自检里也守了"只折传入的那个权重"和"(in,out) 朝向要报错"。
4. **`input_std` 必须加下限。** 固件 `blob_parse()` 对 `sd < 1e-6` 直接
   `bad input statistics` 拒绝整个模型；而 `cfc_task.c` 把 `obs[76]`/`obs[77]` 写死为 0，
   `robot_state`/`control_mode`/`gaze_danger` 也常是常数。导出时 `--std-floor 1e-2`
   兜住这 8~9 维（部署时它们仍恒为常数，归一化后恰好 0，不影响行为）。

## 6. 实测数字（本机 RTX 5060 Laptop）

| 项 | 值 |
|---|---|
| 数据 | 8 条会话 × 2051 帧 @50 Hz = 16408 帧（IDLE 18.9% / TRACK 59.2% / ACK 7.3% / REFRACTORY 14.6%） |
| 网络 | 32783 参数（128.1 KB fp32）= 投影 106→64 + 单 cell(64/64/64) + 门 64 + 5 头 |
| 训练 | 6 epoch / **76 s**（GPU），val loss 0.452 → **0.053**，val mode 准确率 **99.1%** |
| λ（保留门） | 0.0474 → **0.0433**（基本没打开：合成任务不需要长记忆，见 §7） |
| blob | 133372 字节，头 2240，CRC 0xFB5F0AE2，槽上限 262144 |
| **torch vs 固件 C** | **max_abs_err = 2.384e-06**（1669 帧 × 15 个输出，判据 1e-5，余量 4x），不随帧数累积 |
| C 引擎速度 | PC 上 0.024 ms/步（含读盘），折算 ESP32-S3 ≈ 0.3~0.5 ms/步 |

## 7. 明早真机上最可能翻车的三处

1. **域统计量是合成的。** `input_mean/std` 只用合成观测统计，真机上 `feat_64` 是
   YOLO+FeatureAdapter 的真实输出、本体感知是真实 IMU/轮速 —— 归一化会整体错位，
   4σ 夹取后模型基本在饱和区工作。上机第一件事看 `/status` 的 clip 比例（固件
   `clip_ratio() > 0.5` 会直接丢弃该帧）。**修法**：录 10~20 分钟真机数据，重算
   `input_mean/std`（`obs.fit_stats`）后重导 blob，权重可以先不重训。
2. **`vyaw` 正方向没核实**（TECH-PLAN §15-V4 仍未验证）。符号反了机器人会往反方向转，
   而训练/对拍全绿也发现不了。上机时把 `cfg.vyaw_max` 先调到 10 度/秒试。
3. **摄像头装在头上还是机身上没核实**（§15-V1）。合成数据按 `img_frame=0`（头随动）造，
   `theta_img = 人绝对方位 −(yaw+q_head)`；若实际是机身固定，模型学到的映射整体偏一个
   `q_head`。教师 `config.yaml` 也是按 0 配的，改的话两边要一起改。

## 8. 与固件/契约对不上的地方（已确认，不是 bug）

1. `cfc_model.h` 里**没有** `struct model_hdr`：头是 `cfc_model.c` 里的一组 `#define H_*`
   硬偏移 + `rd_u32(blob+H_MAGIC)` 手读。所以本目录的 `dump_offsets.c` 造了一个
   **打包结构体**只为用 `offsetof` 复算这些偏移（并证明不打包也一样，18 个字段零填充），
   它不参与固件编译。三方（固件宏 / 编译器 offsetof / 打包代码）在 `--check-offsets` 里逐项比对。
2. `TECH-PLAN §4.3` 的观测表和固件 `build_obs()` 有细微差别（固件多写了 `dq`、`head_err`、
   `pos_err`、`body_yaw`、`yaw_err`、`control_mode`、`feat_age`，`obs[76..77]` 是保留 0）。
   **实现以固件为准**，`obs.py` 的下标表就是逐行抄的 `cfc_task.c`。
3. `TECH-PLAN §10.1` 的 `L_clip` 对 tanh 头恒等于 0（`|a|≤1` 时 `relu(|a|-1)=0`），
   本实现保留该项（`--lambda-clip 0.1`）但实测一直是 0，属预期。
4. `cfc_model.c` 的 `param_count` 公式**不含** `ret`（保留门那 64 个 float），
   验收按固件公式复算，两边都是 32719。
