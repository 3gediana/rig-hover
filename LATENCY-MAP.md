# 数据流通链与延迟地图（2026-11-03 夜，真机实测）

> **本文取代了同名的旧版本。** 旧版结论是"HTTP 每请求 50 ms 是 ESP-IDF
> 固有开销，只能绕开"。**这个结论是错的。** 真因是 Nagle 算法，一个
> `setsockopt` 就砍掉了每帧 ~49 ms。
>
> 数据来源分四类，逐条标注：`[实测]` `[固件常数]` `[未测]` `[估算]`。

---

## 0. 一句话

端到端 **121.5 ms → 75.1 ms**（p50），帧率 **8.0 → 12.9 Hz**，
其中 `POST /feat` 从 56.3 ms 降到 8.0 ms。**剩下的大头是相机 + 软件 JPEG
编码（58 ms），不是网络。**

---

## 1. 改前 / 改后

### 1.1 分段

| # | 环节 | 改前 p50 | 改后 p50 | 说明 |
|---|---|---|---|---|
| 1 | `GET /capture`（含相机读出 + 软件 JPEG 编码） | 51.6 ms | **58.0 ms** | 传输只占 ~5 ms，其余是编码 [实测] |
| 2 | JPEG 解码（PC） | 0.2 ms | 0.2 ms | [实测] |
| 3 | YOLO26n-pose + 548 维特征 + adapter + 手势 | 12.3 ms | **8.5 ms** | [实测] GPU 空闲时 9.0–11.5 抖动 |
| 4 | `POST /feat`（300 B 上行） | **56.3 ms** | **8.0 ms** | **7×**，Nagle 修复 [实测] |
| | **端到端 p50** | **121.5 ms** | **75.1 ms** | −38% |
| | 端到端 p95 | 130.6 ms | **79.9 ms** | |
| | 可达帧率 | 8.0 Hz | **12.9 Hz** | +61% |

### 1.2 单端点

| 端点 | 改前 | 改后 | 备注 |
|---|---|---|---|
| `/status` | 55–79 ms | **14.7 ms**（15 次中位，13.7–17.0） | 已打补丁 |
| `/feat` | 56.3 ms | **8.0 ms** | 已打补丁 |
| `/capture` | 51.6 ms | **58.0 ms** | 已打补丁，编码占大头 |
| `/cfg` | ~58 ms | **58–71 ms** | **未打补丁，作对照组** |

`/cfg` 没打补丁、还是 50+ ms，`/status` 打了补丁、掉到 15 ms——
同一固件、同一时刻，这就是 Nagle 假说的直接证据。

---

## 2. 全景图（改后）

```
┌──────────────── 机器人 ESP32-S3（240 MHz，8 MB PSRAM）────────────────┐
│                                                                       │
│  GC0308 摄像头 240×240 RGB565 → PSRAM                                  │
│        │  曝光/读出 + 软件 JPEG 编码（无硬件编码器）  ≈50 ms  [实测推断]  │
│        ▼                                                              │
└────────┼──────────────────────────────────────────────────────────────┘
         │  GET /capture  58.0 ms
         │    └ 其中传输 ~5 ms，其余是编码
         ▼
┌──────────────── PC（RTX 5060 Laptop 8 GB）─────────────────────────────┐
│  解码 0.2 ms                                                          │
│  感知链 8.5 ms  ├ YOLO26n-pose + 548 维特征   8.5 ms                  │
│                ├ FeatureAdapter 548→64        ~0.7 ms  [实测]         │
│                └ 手势分类器 + 事件机           ~1.6 ms  [实测]         │
│  组装 116 维观测 < 0.1 ms                                             │
└────────┼──────────────────────────────────────────────────────────────┘
         │  POST /feat（300 B）  8.0 ms
         ▼
┌──────────────── 机器人 ───────────────────────────────────────────────┐
│  /feat 写缓存                                                          │
│    → 等下一个 CfC 步  [固件常数] cfc_hz=50 → 20 ms 周期，均值 10 ms      │
│  CfC 推理  1.4–4.0 ms  [实测·旧]                                       │
│    → 等仲裁 tick  [固件常数] 6 ms，均值 3 ms                            │
│  仲裁器 + 安全壳  stale=300 / lost=1000 / cmd_timeout=500               │
│    → head_lpf = 0.15/tick  ⇒ τ=36.9 ms，90% 到位 85 ms [固件常数]       │
│    → 舵机 UART  [未测]                                                 │
└───────────────────────────────────────────────────────────────────────┘
```

从"拍到一帧"到"CfC 拿到这帧的观测"：**75 ms**。
到舵机指令发出再加 ~14 ms。到头部实际到位再加 ~85 ms。

---

## 3. 三个根因（按节省的毫秒排序）

### 3.1 Nagle —— 省 ~49 ms/帧，主因

**症状**：裸 socket 探针量出 **首字节 5.1 ms，body 再等 50–75 ms**。
`ttfb ≈ total` 只说明"延迟不在传输尾部"，不排除 Nagle。

**机理**：`esp_http_server` 把状态行/响应头和 body 分两次小段 `send()`。
lwIP 默认开 Nagle：不满 MSS（1440 B）的段要攒着等对端确认。
`httpd_txrx.c:490/531` 里确实调了 `TCP_NODELAY`，但**只对错误响应**
（`CONFIG_HTTPD_ERR_RESP_NO_DELAY`），正常响应走默认。

这解释了那个一直对不上的矛盾：

| 端点 | 响应体 | 段构成 | 改前实测 |
|---|---|---|---|
| `/` | 5,232 B | 有满 MSS 段 | **8.9 ms** ✅ |
| `/api/set` | 11 B | 一个极小段 | 52.2 ms |
| `/feat` 响应 | ~300 B | 不满段 | 56.3 ms |
| `/status` | 780 B | 不满段 | 55 ms |

**负载大 20 倍的 `/` 反而快 6 倍**——因为它是唯一能凑出满段的响应。

**修法**（`hover_debug_server.cc`）：

```c
static void force_nodelay(httpd_req_t *req) {
    int fd = httpd_req_to_sockfd(req);
    if (fd < 0) return;
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}
```

在 `/`、`/capture`、`/status`、`/feat` 的 handler 开头调用。

> IDF 5.5 **没有** `CONFIG_LWIP_TCP_NODELAY` 这个全局开关（已验证），
> 所以只能按连接设。**其余端点（`/cfg` `/model` `/cmd` `/emote` `/gaze`
> `/mode` `/vision`）还没打补丁**，仍然 50+ ms。

### 3.2 TCP 窗口只有 5760 B —— 一帧 6.5 KB 要多次窗口往返

`CONFIG_LWIP_TCP_SND_BUF_DEFAULT=5760`、`CONFIG_LWIP_TCP_WND_DEFAULT=5760`，
MSS 1440 → 窗口不到 4 个段。改成 **32768**，`TCP_RECVMBOX_SIZE` 6 → 32。

改前 `/capture` 一度跑到 92 ms（6.5 KB / 84 KB/s），改后回到 58 ms。
`[实测]`，但这一项和 Wi-Fi 环境相关，波动比 Nagle 那项大。

### 3.3 连不上 / 拿不到 IP —— 两个独立问题

**(a) Windows 移动热点"显示 On 但没在发信标"。**
`TetheringOperationalState = On`、`ClientCount = 0`，但 `WLAN 3` 的
**TX 10 秒 0 字节**。重新配一次热点配置后 TX 变 8930 B/10s，机器人立刻扫到并完成
WPA2 握手。**判据：看热点虚拟网卡的 TX 增量，别看 UI 状态。**

**(b) DHCP 从来不回，机器人只有 L2 没有 L3。**
串口日志能直接看到：`wifi:connected with qasd ... aid = 11` 之后**没有**
`Got IP`，也没有 `IP_EVENT_STA_GOT_IP`。

历史上这个坑是用一个跑在 `%TEMP%\robot_probe\dhcpd_long.py` 的
**自建 DHCP 守护进程**填的——它跑满 4 小时会静默退出，重启电脑就没了。
`TECHNICAL-REPORT.md` §5 第 8 条已经把它列为"最难查的一类问题：
基础设施里埋了超时"。

**现在改成固件内兜底**：关联后 12 秒仍无 IP，就
`esp_netif_dhcpc_stop()` + `esp_netif_set_ip_info(192.168.137.50/24)` +
DNS。**不再依赖任何 PC 侧进程。**

> 注意：`esp_netif_set_ip_info()` **自己会发 `IP_EVENT_STA_GOT_IP`**。
> 不需要（也**不能**）再手动调 `OnNetworkEvent(Connected)`——
> 那样整个连接流程会跑两遍，`hover_debug_server_start()` 被初始化两次，
> 立刻 `LoadProhibited` 崩溃循环。

---

## 4. 诊断方法（可复用）

`pc/tools/http_floor_probe.py`：裸 socket 发请求，把一次事务拆成四段

```
t_connect  TCP 建连
t_send     写出请求
t_first    收到第一个字节   ← Nagle 卡 body 时这里会「快」
t_full     响应读完整       ← 而这里「慢」= Nagle 特征
```

再加"同连接连发 vs 每次新建连接"的对比，和响应头里的 `Connection:` 字段。

判读表：

| 现象 | 结论 | 改什么 |
|---|---|---|
| 首字节快、读完整慢 ≈ +40 ms | **Nagle**（服务端） | `setsockopt(TCP_NODELAY)` |
| 首字节本身就慢 | 请求处理 / Wi-Fi | 查 handler、查省电 |
| 新连接 ≫ 同连接 | 连接没复用 | 客户端显式 keep-alive |
| 只有 `/` 快 | 与段大小相关 → Nagle 家族 | 同上 |

**教训**：`RESULTS.md` §9.1 当时记的是"`ttfb ≈ total`，所以不是 Nagle"，
据此把 HTTP 判成不可优化的固有开销，方向错了整整一轮。
`http.client.getresponse()` 读的是"响应头收齐"，不是"首字节"——
**测什么就要说清测的是哪一段**。

---

## 5. 还剩什么

| 项 | 现状 | 下一步 | 预期 |
|---|---|---|---|
| `/capture` 58 ms | 相机读出 + 软件 JPEG | 灰度化（实测编码快 **1.48×**、载荷 86%） | **~40 ms** |
| 每帧两次 HTTP | 2 次往返 | 合并成 `POST /step`（请求体=特征，响应体=下一帧 JPEG） | 帧率上限翻倍 |
| 其余端点 50+ ms | 未打补丁 | 同样加 `force_nodelay` | 各降到 ~10 ms |
| 算法软延迟 | 手势窗 12 帧 @8fps = **1.5 s** | 见 `SCENARIO-v2.md` | — |

**目标：45–50 ms / 20+ Hz。**

---

## 6. 未测清单（不许当实测用）

| 项 | 为什么没测 |
|---|---|
| 相机曝光/读出 vs JPEG 编码各占多少 | 混在 `/capture` 的 58 ms 里，没拆开 |
| 舵机 UART 与机械响应 | 需要机器立起来 |
| 640×480 下整条链 | 固件还没改 |
| MJPEG / UDP 的实际收益 | 真机才能测；localhost 测不出来 |
| `cfc_hz` 提到 100 能否跟上 | 板上验证 |

---

## 7. 复现命令

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'

# 1) 端到端（需要机器人在线，且注意启动要 14 秒才拿到 IP）
& $py D:\robot_work\pc\tools\e2e_latency.py --n 60 --base http://192.168.137.50 --warmup 10

# 2) HTTP 分段探针（定位 Nagle）
& $py D:\robot_work\pc\tools\http_floor_probe.py --host 192.168.137.50 --n 6

# 3) 只测感知链（不需要机器人）
& $py D:\robot_work\pc\bench\_latency_pipeline.py

# 4) 单端点计时（必须加 --noproxy，本机有代理会返回 502）
curl.exe --noproxy '*' -s -o NUL -w "%{http_code} %{time_total}`n" http://192.168.137.50/status
```

---

## 8. 一条容易看反的因果

`cfc_hz` 从 50 降到 5，`ttfb` 几乎不变（49.8 → 60.1 ms）。
说明 **CfC 推理没有和 httpd 抢 CPU**，瓶颈在 httpd 自己的响应路径——
所以优化 CfC（4.0 ms → 1.4 ms）对端到端没有意义，而给连接关 Nagle 有 49 ms。

---

## 9. 第二轮（原生灰度 + `/step`）—— §1–§8 的绝对数字已过时

2026-11-03 夜第二轮。§1–§8 测的是 75.1 ms 那一版，**定性结论仍然成立**
（Nagle 是主因、TCP 窗口必须 ≥ 一帧），但所有绝对数字作废。

### 9.1 最终数字

| | 起点 | 第一轮 | **第二轮（当前）** |
|---|---|---|---|
| 端到端 p50 | 121.5 ms | 75.1 ms | **42.8 ms** |
| p90 | — | 79.9 ms | **46.1 ms** |
| 控制率 | 8.0 Hz | 12.9 Hz | **21.4 Hz** |

```
e2e_latency.py --n 80 --step
POST /step 33.3 | decode 0.1 | YOLO+features 9.8 | TOTAL 42.8 ms | 80/80 accepted
```

### 9.2 这一轮的三个改动

1. **传感器输出原生灰度**（`hover_board.cc`: `PIXFORMAT_GRAYSCALE`）。
   GC0308 有 Y-only 输出模式（`OUTPUT_FMT=0xb1`），DMA 搬的字节数减半，
   同时省掉原来 11 ms 的软件 RGB565→Y 转换。
2. **`POST /step`**：body 是 `/feat` 那份 116 维观测，响应 body 是下一帧
   JPEG，CfC 回执放 `X-Cfc` 头。一次往返替代 `GET /capture` + `POST /feat`，
   省掉一整个 RTT（约 7 ms）。
3. 删掉 `capture_handler` 里每帧一行的 `ESP_LOGI`（115200 下约 3 ms，
   而且发生在响应之后，正好砸进 keep-alive 的下一个请求）。

### 9.3 机器人的 33 ms 花在哪（板上实测，不是估算）

固件在 `Esp32Camera::CaptureJpeg` 里分段打点（1/8 采样，避免日志本身污染）：

| 阶段 | 原生灰度 | 改之前（RGB565 + 软件转换） |
|---|---|---|
| `esp_camera_fb_get()`，等下一帧 | 10.0 ms | 0.0 ms |
| RGB565→Y 软件转换 | 0.0 ms | 11.1 ms |
| 软件 JPEG 编码 | 21.7 ms | 22.4 ms |
| 小计（不含 HTTP） | 31.6 ms | 33.5 ms |
| HTTP 传输（5.3 KB） | ~9 ms | ~9 ms |

注意第一行和第二行的**互换**：改成原生灰度后，省下的 11 ms 转换时间被
"等下一帧"吃掉了。原因很简单——**相机帧周期约 32 ms，那就是地板**。
处理得再快也还是在等下一帧，`/capture` 实测 31 ms 时已经贴着上限。

想再往下走只有两条路：
- 改采集模式（分辨率 / 帧率 / 传感器窗口），不是优化 HTTP；
- 后台自由跑「采集+编码」流水线，`/step` 只发最新 JPEG —— 但那提升的是
  **控制率**（21 → ~31 Hz），**不是延迟**（帧周期不变，只会引入帧龄）。

### 9.4 一个反直觉的对照：同一份固件的"劣化态"

| 状态 | `/capture` | `/feat` | `/status` |
|---|---|---|---|
| 刚开机 14 s | 43.8 ms | 7.8 ms | 11.0 ms |
| 开机 90 s | 43.3 ms | 7.8 ms | 10.2 ms |
| 经过 200+ 次连接后 | **68.6 ms** | **15.4 ms** | **20.0 ms** |

劣化态下 internal RAM 只剩 399 B（CFC 模型 blob 占 43 KB internal SRAM），
OTA 的 TLS 每次都失败重试。

**测延迟前先复位板子。** 这一轮我为此白查了一圈，一度以为是代码退化。
判据：先读 `/status` 的 `"t"`（uptime），> 120 s 就先复位再测。

