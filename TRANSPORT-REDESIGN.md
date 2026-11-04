# 传输层改造设计：从 HTTP 轮询到 UDP + MJPEG

> **2026-11-03 深夜状态更新（必读）**
>
> 本文最初的前提是"HTTP 每请求 ~50 ms 是 ESP-IDF 固有开销，只能绕开"。
> **那个前提被推翻了**：真因是 **Nagle 算法**，一个 `setsockopt(TCP_NODELAY)`
> 就把 `POST /feat` 从 56.3 ms 打到 **8.0 ms**，端到端 121.5 → **75.1 ms**。
>
> 所以本文 §2（UDP 上行）和 §3（MJPEG 下行）**暂时不做**——先把 HTTP 用对，
> 收益已经有 42%。真实的新数据见 **`LATENCY-MAP.md`**。
>
> 仍然有效的部分：
> * §0 的问题定义与实测数据（改前）
> * §7.4 的执行顺序与"先量再改"的纪律
> * §3 MJPEG 的实施要点（若将来还需要，见 §3 的独立任务警告）
>
> 下面原文保留，作为**决策历史**——它记录的是一次判断失误，以及为什么
> 当时看起来那么合理（`ttfb ≈ total` 被当成了排除 Nagle 的证据）。

> 2026-11-03
> 触发原因：端到端实测（`RESULTS.md` §9）证明 HTTP 是唯一瓶颈。

---

## 0. 为什么必须改

实测数据（全部来自本机本机器人）：

| 事实 | 数值 |
|---|---|
| 端到端延迟 | **121.5 ms**（p50） |
| 其中两次 HTTP 往返 | **~108 ms（90%）** |
| 其中全部计算（YOLO + 特征 + CfC） | **12 ms（10%）** |
| HTTP 单请求开销 | **~50–65 ms**（`/api/data` 原厂 handler 同样 57 ms） |
| HTTP 吞吐上限 | **~16 请求/s**（并发 1/2/3/4 恒定，延迟线性增长 = 完全串行） |
| 由此得到的端到端帧率上限 | **8 fps**（每帧 2 次请求） |
| 实测帧率 | 8.3 Hz ← 与上限吻合 |

**结论：继续优化推理没有意义。** 推理从 4.04 ms 降到 1.4 ms 只省 2.6 ms，占预算 2%。

> **⚠️ 2026-11-03 晚追加**：用户明确"HTTP 占 100 ms 太不可接受"。
> 追查时发现实测数据里有一处**自相矛盾**，它可能改变结论 ——
> **`/`（5,232 B 静态 HTML）只要 8.9 ms，而 `/api/set`（11 B）要 52 ms。**
> 同一台 httpd、同一条连接、同一个客户端，**负载大 20 倍的反而快 6 倍**。
>
> 也就是说 **50 ms 不是 HTTP 的物理下限**。在重写传输层之前，必须先分这个叉。
> 见 **§7**。

---

## 1. 目标

| 指标 | 现状 | 目标 |
|---|---|---|
| 端到端延迟 | 121.5 ms | **< 40 ms** |
| 端到端帧率 | 8 fps | **≥ 20 fps** |
| 机器人 CPU 占用（除推理） | — | 不增加 |

---

## 2. 上行：特征走 UDP 单向

### 为什么是 UDP
- 避免请求/响应往返：省掉 ~55 ms 的整次 HTTP 事务
- 视觉特征是**流式数据**，最新的才有用，旧帧丢了无所谓 —— 天然适合 UDP
- 机器人已有 UDP 通道（`telemetry_stream.c`，端口 9001，500 Hz 推送遥测），
  说明 UDP 路径在这套 WiFi 上可用

### 帧格式
直接复用现有 `POST /feat` 的载荷布局（`TECH-PLAN-v1.md` §5.2），保持不变：

```
u16 magic = 0xFE01
u16 n_obj
u32 t_ms
u16 n_feat            (64)
u16 reserved
obj[] : { i16 cx, i16 cy, i16 w, i16 h, u8 conf, u8 cls, u8 flags, u8 pad,
          i16 wave_sin, i16 wave_cos }
f32 scale
i16 feat[n_feat]
```
单个 UDP 数据报即可容纳（约 200 B），无需分片。

### 机器人侧
- 新增一个 UDP 接收任务（或复用 `telemetry_stream` 的 socket 逻辑），绑定固定端口（如 **9002**）
- 收到即解析并写入现有的 seqlock 双缓冲（`cfc_task.c` 已有）
- **不需要改动 `cfc_task` 的消费逻辑** —— 只是换了数据来源
- 端口进 `/cfg`（`feat_udp_port`），可在不重编译的情况下切换/关闭

### PC 侧
- `socket.sendto()` 发到 `192.168.137.50:9002`
- 不需要等回应 → 单帧耗时 ≈ 0.1 ms
- **仍然保留 HTTP `/feat` 作为回退**（调试、抓包、无 UDP 时）

### 风险与对策
| 风险 | 对策 |
|---|---|
| UDP 丢包 | 已有 `stale_ms`/`lost_ms` 降级链；且视觉帧本就允许丢 |
| 乱序 | 帧内带 `t_ms`，机器人只接受更新的时间戳 |
| 网络策略阻止 UDP | `/cfg` 里可切回 HTTP 回退 |

---

## 3. 下行：图像走 MJPEG 流

### 为什么是 MJPEG 流
`GET /capture` 每次要一次完整 HTTP 事务（51 ms），但**数据本身只有 1.5–9 KB**。
改成一次连接持续推帧，每帧只剩编码 + 传输成本。

### 机器人侧
- 新增 `GET /stream`：`Content-Type: multipart/x-mixed-replace; boundary=...`
- 连接保持打开，按目标帧率（如 15 fps）推送 JPEG
- **每个连接占用一个 socket**（现有 `max_open_sockets=13` 够用）
- 帧率进 `/cfg`（`stream_fps`）
- 注意：ESP-IDF httpd 的分块发送需要避免长时间阻塞 worker —— 优先用**独立任务 + 裸 socket**，
  或者确认 httpd 的 `httpd_resp_send_chunk` 在长连接下的行为

### PC 侧
- 一次连接，持续解析 multipart 帧
- 与现有 `cv2.imdecode` 管线无缝对接
- **同时保留 `/capture`** 作为单帧抓取接口（校准、调试用）

### 预期
| 项 | 现状 | 改造后 |
|---|---|---|
| 每帧额外延迟 | 51 ms（HTTP 事务） | ~编码 10–20 ms（无事务开销） |
| 帧率瓶颈 | HTTP 16 req/s | 摄像头/编码能力 |

---

## 4. 预期的合并收益

| 段 | 现状 | 改造后 | 依据 |
|---|---|---|---|
| 图像下行 | 51.6 ms | ~15 ms（估算） | 去掉 ~50 ms 的事务开销 |
| 解码 | 0.2 ms | 0.2 ms | 不变 |
| YOLO + 特征 | 12.3 ms | 12.3 ms | 不变 |
| 特征上行 | 56.3 ms | ~1 ms | UDP 无往返 |
| **合计** | **121.5 ms** | **~30 ms** | |

帧率上限从 8 fps 提升到**受限于摄像头和编码**（预计 15–20 fps）。

> ⚠️ 这些是**估算**，必须实测。项目已经有过一次教训：估算 vs 实测差 2~3 倍
> （推理时延估 1.5 ms、实测 4.04 ms）。

---

## 5. 实施顺序（每步都可独立验证）

1. **机器人加 UDP 接收 + `/cfg {feat_udp_port}`**
   - 验证：PC 发 1000 个 UDP 包，`/status` 里 `cfc.feat_age_ms` 稳定更新、`dropped` 不涨
2. **PC 侧切 UDP 上行，保留 HTTP 回退**
   - 验证：重跑 `e2e_latency.py`，上行段应从 56 ms 降到 ~1 ms
3. **机器人加 `/stream` MJPEG**
   - 验证：`curl` 连续收帧，帧间隔稳定
4. **PC 侧切流式下行**
   - 验证：重跑 `e2e_latency.py`，总延迟应 < 40 ms
5. **重跑整套 `run_all_tests.py`**，确认没有回归

---

## 6. 不在本次范围

- 降低图像分辨率 —— 240×240 已经很小
- 压缩特征（64 维 float32 = 256 B）—— 不是瓶颈
- 图像编解码加速 —— 编码只占 ~40 ms 的一部分

> 原文里还有一条"改 HTTP 层本身 —— 收益不确定"，**现在撤回**：
> 见 §7，`/` 能在 8.9 ms 内响应，说明这张 50 ms 的牌我们还没打完。

---

## 7. 加急：HTTP 这一百毫秒到底能不能砍掉

### 7.1 先看这处矛盾（它决定后续所有工作）

| 端点 | 响应体 | ttfb 中位 | 性质 |
|---|---|---|---|
| `/` | **5,232 B** | **8.9 ms** | 静态 HTML，handler 只有两行 |
| `/api/set` | **11 B** | 52.2 ms | 原厂 handler，返回静态小串 |
| `/api/data` | 236 B | 56.8 ms | 原厂 handler |
| `/status` | 780 B | ~55 ms | cJSON 构造 |
| `/capture` | 9,277 B | 51.6 ms | 相机 + 软件 JPEG |

**同一台 httpd、同一条连接、同一个客户端。负载大 20 倍的 `/` 快 6 倍；
返回 11 字节静态串的 `/api/set` 却要 52 ms。**

已排除的（`RESULTS.md` §9.1，全部实测）：网络（ICMP 2 ms、TCP 建连 0.5 ms）、
CPU 抢占（`cfc_hz` 50→5 不影响 ttfb）、Nagle（`ttfb ≈ total`）、cJSON（11 字节串一样慢）。

**没排除的两条**：

1. **连接没被复用**。`hover_debug_server.cc:1102` 已经设了
   `config.keep_alive_enable = true`，但 `RESULTS.md` §8 记录过"高频请求后 socket 耗尽、
   TIME_WAIT 120 秒"——**TIME_WAIT 只有服务端主动关连接才会累积**。
   所以 keep-alive 很可能实际没生效（客户端没发 `Connection: keep-alive`，
   或某处分支提前关了 socket）。
2. **handler 路径的差别**。`index_handler` 只有两行：
   ```c
   httpd_resp_set_type(req, "text/html");
   httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
   ```
   而慢的那些要么建 cJSON、要么调 `httpd_resp_set_hdr`、要么取相机。
   差 40 ms 不该是这些 CPU 操作，但**值得逐行二分**。

### 7.2 首要嫌疑：Nagle 算法（一个配置项）

查 `sdkconfig` 发现：

```
CONFIG_HTTPD_ERR_RESP_NO_DELAY=y          ← 只有「错误响应」开了 TCP_NODELAY
（CONFIG_LWIP_TCP_NODELAY 这一项根本不在 sdkconfig 里 = 未启用）
```

**也就是说，普通 HTTP 响应全部走默认的 Nagle 算法。**

Nagle 的规则是：*如果还有未确认的数据在途，就只放行满 MSS 的整段，小段先攒着。*
MSS 在 IPv4/WiFi 上约 **1460 B**。代进去：

| 端点 | 响应体 | 段数构成 | Nagle 行为 | 实测 ttfb |
|---|---|---|---|---|
| `/` | 5,232 B | **3 个满段 + 残余** | 满段立刻放行 | **8.9 ms** ✅ |
| `/api/set` | **11 B** | 一个极小段 | **攒着等 ACK** | 52.2 ms |
| `/feat`（响应） | ~300 B | 一个不满的段 | **攒着等 ACK** | 56.3 ms |
| `/status` | 780 B | 一个不满的段 | **攒着等 ACK** | 55 ms |
| `/api/data` | 236 B | 一个不满的段 | **攒着等 ACK** | 56.8 ms |
| `/capture` | 9,277 B | ≈7 个段 | 满段立刻放行 | 51.6 ms（但这里面主要是**相机 + 软件 JPEG 编码**，不是 HTTP） |

**这张表和实测完全对得上**，而且解释了为什么"负载大 20 倍的 `/` 反而快 6 倍"——
`/` 是唯一一个能凑出满 MSS 段的响应。

> **反证（必须说）**：`RESULTS.md` §9.1 记的是 "`ttfb ≈ total`，所以不是 Nagle"。
> 但 Python `http.client.getresponse()` 是**读完整个响应头才返回**的，
> 它测的其实是"响应头收齐"，不是"首字节"。如果 Nagle 卡的是**响应体**，
> ttfb 快、total 慢，`ttfb ≈ total` 就不该出现——所以这条反证本身也可疑。
> **`http_floor_probe.py` 会分别量 ttfb 和完整读取，一次说清。**
>
> 另外注意：Nagle 的延迟来自**对端延迟确认**，客户端开 `TCP_NODELAY` 无效
> （那管的是客户端自己发包）。必须**服务端**关 Nagle。

#### 如果成立，改法（都是一行）

| 改哪里 | 现在 | 改成 |
|---|---|---|
| `sdkconfig` | 未设 | **`CONFIG_LWIP_TCP_NODELAY=y`** |
| 或代码 | — | accept 后对 socket `setsockopt(IPPROTO_TCP, TCP_NODELAY)` |
| `CONFIG_LWIP_TCP_SND_BUF_DEFAULT` | **5,760** < 单帧 9,277 B | **≥ 16,384**（否则每帧多一个窗口往返） |
| `CONFIG_LWIP_TCP_WND_DEFAULT` | 5,760 | ≥ 16,384 |
| `CONFIG_LWIP_TCP_RECVMBOX_SIZE` | **6** | 32（收包邮箱太小，突发时丢包重传） |

**预期**：`/feat` 56.3 → ~10 ms，`/status` 55 → ~8 ms。
端到端 119.6 → **~71 ms**，**零协议改动、零 PC 端改动**。

剩下的 51.6 ms 是相机 + 软件 JPEG 编码（无硬件编码器），
靠灰度化（实测编码快 **1.48×**、载荷 86%）再压到 ~35 ms。

### 7.3 诊断脚本：`pc/tools/http_floor_probe.py`

机器人一上线跑 2 分钟，它会分别测：

- **同连接连发 10 次**（并打印响应头里的 `Connection:` / `Transfer-Encoding:`）
- **每次新建连接** 10 次
- **裸 socket**（自己拼 HTTP 请求，绕开 `http.client`）
- 端点族对比：`/` vs `/api/set` vs `/status` vs `/capture`

判读规则写在脚本末尾：

| 现象 | 结论 | 下一步 |
|---|---|---|
| 连发 ≪ 新建连接 | 是**连接开销** | 客户端显式复用 socket 就能解决，**不用换协议** |
| 连发 ≈ 新建连接 | 是 httpd **每请求**路径 | 绕开：WS / 裸 TCP / UDP |
| 只有 `/` 快 | httpd 能到 9 ms，是我们的 handler 有问题 | 二分 `index_handler` 与 `status_handler` |
| 全部都 50+ | WiFi/AP 层 | 查省电、DTIM、AP 模式 |

### 7.3 四条路，按"改动/收益"排序

| 方案 | 每帧 HTTP 事务 | 预期单帧延迟 | 预期帧率 | 改动量 | 依据 |
|---|---|---|---|---|---|
| **现状** | 2 | 120 ms | 8 fps | — | 实测 |
| **A. 合并成 1 个请求** | 1 | **~62 ms** | **16 fps** | 固件加一个 handler + PC 客户端 | 直接砍掉一次事务 |
| **B. 客户端显式复用连接** | 2（同一连接） | 同 A 或更好 | 同 A | **只改 PC 端** | 零固件改动，先试这个 |
| **C. WebSocket 单连接双向** | 1 次握手，之后 0 | **~25–35 ms** | 20–30 fps | 固件 + 客户端 | ESP32 实测每消息 **10–14.4 ms**（[Electric UI](https://electricui.com/blog/latency-comparison)） |
| **D. 裸 TCP 自定义协议** | 0 | **~15–25 ms** | 30+ fps | 固件 + 客户端 | 同来源：裸 TCP 比 WS 再快 **~6 ms** |
| **E. UDP 上行 + MJPEG 下行** | 0 | ~20 ms | 受摄像头限制 | 固件 + 客户端 | 绕开 httpd，但两套东西都要写 |

**外部证据**：
- ESP32 上 WebSocket 每消息 **10–14.4 ms**，裸 TCP 再快约 6 ms
  （[Electric UI 的 ESP32 延迟基准](https://electricui.com/blog/latency-comparison)）
- 另有开发者报告 PC↔ESP32 往返回环"约 40–50 ms"
  （[SO](https://stackoverflow.com/questions/65676365/trying-to-reduce-tcp-socket-latency-on-a-esp32)），
  与我们观测一致——说明这不是我们独有的坑，但也说明**它确实可以更低**
- ESP-IDF 默认只对**错误响应**开 `TCP_NODELAY`
  （`CONFIG_HTTPD_ERR_RESP_NO_DELAY`），普通响应不走这条捷径
  （[Kconfig](https://docs.espressif.com/projects/esp-idf/en/v4.3-beta2/esp32s2/api-reference/kconfig.html)）

### 7.4 执行顺序（每步都能独立验证，失败就退回上一步）

1. **跑 `http_floor_probe.py`**（机器人上线后 2 分钟）→ 决定走 A/B 还是 C/D
2. **B**：PC 端显式 `Connection: keep-alive` + 复用 socket，重测
3. **A**：固件加 `POST /step`（请求体=116 维观测，响应体=下一帧 JPEG），
   PC 端一次往返完成"发特征 + 取图"，重测
4. **C/D**：若 1–3 之后仍 > 40 ms，再上 WebSocket 或裸 TCP
5. **E**：最后再做，因为它同时改两个方向、两套代码

> **不要跳过第 1 步直接上 WebSocket。** 如果 50 ms 其实是连接开销，
> 那 A/B 两步就能把端到端压到 60 ms 以下，改动量比 C 小一个数量级。
> 项目已经吃过一次"估算比实测乐观 2–3 倍"的亏（推理估 1.5 ms、实测 4.04 ms），
> 这次先量再改。

### 7.5 顺带要动的固件常数

| 项 | 现在 | 建议 | 理由 |
|---|---|---|---|
| `CONFIG_LWIP_TCP_NODELAY` | 未确认 | **开** | 普通响应不走 Nagle |
| `CONFIG_HTTPD_ERR_RESP_NO_DELAY` | 默认开 | 保持 | — |
| `TCP_MSL` | 已 60000→5000 | 保持 | 防止重新陷入 socket 耗尽 |
| `max_open_sockets` | 13 | 保持 | WS 长连接要占一个 |
| `keep_alive_enable` | true | 保持并**验证真的生效** | 见 §7.1 |
