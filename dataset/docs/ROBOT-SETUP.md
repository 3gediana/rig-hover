# 机器人 + 热点 + 采集服务：上手指南

> 给**第一次拿到这台机器人的队友（和他的 agent）**。
> 目标：把这个仓库跑起来、让机器人自动连上你的电脑、用 8780 的网页服务录数据。
> 机器人本体、架构、踩坑的完整说明在仓库根目录 [`AGENTS.md`](../../AGENTS.md)，
> 本文只写“换一台电脑 / 换一个人操作”必须知道的部分。

---

## 1. 机器人是什么（一分钟）

ESP32-S3 双轮自平衡桌面机器人（代号 文文 / RIG-Hover）：

| 项 | 值 |
|---|---|
| MCU | ESP32-S3 QFN56 rev0.2，双核 240 MHz，512 KB SRAM |
| 内存 / Flash | 8 MB PSRAM（Octal）/ 16 MB flash |
| 摄像头 | GC0308，当前 240×240 原生灰度，约 30 ms 一帧 |
| 云台 | **只有一个舵机，只能左右摇头，不能低头** |
| 网络 | 仅 2.4 GHz WiFi |
| MAC | `90:da:72:4f:9b:e0` |
| 串口 | CH343，Windows 下是 `COM3` |

**架构（理解错就全错）**：机器人上只跑一个吃 116 维数字的 CfC 决策模型，
不碰图像；人脸/拳头/敲击的视觉感知全在 PC 端算完再把特征发给机器人。
所以采集时机器人只做一件事：**被只读地取图**（`GET /capture`）。

---

## 2. 热点：让机器人自动连上你的电脑

机器人靠**电脑的移动热点**上网（没有别的网络入口）。当前使用的配置：

| 项 | 值 |
|---|---|
| SSID | `qasd`（全小写，5 个字符） |
| 密码 | `88888888`（8 个 8） |
| 频段 | **2.4 GHz**（不能用 5 GHz） |
| 安全 | **WPA2-PSK**（不要选 WPA2/WPA3 混合，ESP32 不支持） |
| Windows 热点网卡 | `WLAN 3`，固定 `192.168.137.1/24`（ICS） |
| 机器人地址 | 固定 `192.168.137.50/24`，网关 `192.168.137.1` |

**设置步骤（Windows 11）**

1. 设置 → 网络和 Internet → **移动热点**；
2. “共享我的以下 Internet 连接”选你实际在用的网卡（例如 `SZTU-student`）；
3. “连接到”选 **Wi-Fi**；编辑属性：名称 `qasd`，密码 `88888888`，
   **网络频段 2.4 GHz**，打开“使用 WPA2”；
4. 打开移动热点开关。

> 热点的 SSID / 密码必须和机器人 NVS 里保存的一致。
> **改了热点名或密码，机器人会一直扫旧名字，必须重新配网（见 §4）。**

判据（GitHub 上踩过的坑）：Windows 设置里显示“移动热点 开”不代表它在发信标。
看热点网卡的 TX 字节是否在涨：

```powershell
Get-NetAdapterStatistics -Name 'WLAN 3'
```

也可以用 Windows 的“热点客户端数”做辅助判断（机器人连上后会 +1）。

---

## 3. 机器人联网行为（固件已修过，别再改回旧逻辑）

机器人开机后：

1. 从 NVS 读已保存的 WiFi；有则开始扫描；
2. 找不到 AP 时按 **5 s → 10 s → 15 s** 的间隔持续重扫，**永不**因为超时
   进入配网模式（旧固件 60 s 后会 `StopStation` 进配网模式，这是机器人“热点明明开着
   却永远连不上”的根本原因）；
3. 找到 AP 后连接；若 DHCP 12 s 没给地址，用静态兜底 `192.168.137.50/24`；
4. 连上后关闭 WiFi 省电（`WIFI_PS_NONE`），否则预览/录制会周期性卡顿；
5. 热点中途断掉：约 2.4 s 一次重试，AP 回来后 **~8 s 内自动重连**；
   热点晚开：机器人会一直等，AP 出现后 **~18 s 内上线**（实测，停止热点 80 s 再开）。

看 robot 是否在线：

```powershell
curl.exe --noproxy '*' http://192.168.137.50/status
curl.exe --noproxy '*' -o D:\tmp\frame.jpg http://192.168.137.50/capture
```

> 所有 HTTP 测试都要加 **`curl.exe --noproxy '*'`**，本机代理会返回 502。

---

## 4. 换热点 / 换密码之后：重新配网

WiFi 凭据存在机器人 NVS 里。换了 SSID 或密码，**不会自己更新**，要重新配网：

1. 让机器人进入配网模式（两种方法）：
   * **开机时单击 BOOT 键**：在设备状态还是 `Starting` 且尚未连上 WiFi 时单击；
   * **长按 BOOT（或触摸键）超过 3 秒**：会擦除 NVS 并重启，重启后自动进配网模式。
2. 机器人会开一个配网热点，SSID = `RIG-Omni-` + MAC 后两字节，
   这台是 **`RIG-Omni-9BE0`**。
3. 电脑连上这个热点，浏览器打开 **`http://192.168.4.1`**，按页面提示填
   目标 WiFi 的 SSID 和密码（填 `qasd` / `88888888`），保存。
4. 机器人重启并连上电脑热点；用 §3 的 `/status` 验证。

> 配网热点是机器人自己开的 AP，和电脑的 `qasd` 不是同一个；连它的时候电脑
> 会临时断网。配完记得切回电脑热点。

---

## 5. 启动采集/标注服务

服务是单文件 Python 程序，不依赖机器人固件改动，只读 `GET /capture`：

```powershell
& 'G:\ai\python11\python.exe' D:\robot_work\pc\tools\video_annotator.py `
    --port 8780 --root D:\robot_work\video_annotation_data `
    --robot 192.168.137.50 --fps 15
```

浏览器打开 **http://127.0.0.1:8780**。依赖：

* Python 3.10+，`opencv-python`（读写视频/JPEG）；
* **ffmpeg**：优先用 `imageio-ffmpeg` 自带的二进制（`pip install imageio-ffmpeg`），
  否则找系统 PATH 里的 `ffmpeg`。没有 ffmpeg 时视频会退化成 WebM VP8；
* 机器人地址必须和电脑热点同一网段（默认 `192.168.137.50`）。

页面上：**打开预览** → 摆好机位 → **开始录制**（5/8/10/15/20 fps 可选，
默认 15）→ 需要调机位时 **暂停**（暂停时间不计入视频时间轴）→
**停止并标注**。录完的视频会自动出现在左侧列表并载入播放器。

录制的输出：

```
video_annotation_data/
├── videos/<take_id>.mp4                发布用 H.264（浏览器可播）
├── videos/<take_id>_source/
│   ├── frames/000000.jpg …             原始相机帧（权威数据）
│   ├── frames.jsonl                    每帧真实采集时间 t_s 与对应视频槽位
│   └── meta.json                       帧数/输出fps/采集fps/时长/codec
└── annotations/<sha256(video_name)[:20]>.json   区间标注
```

**原图帧必须保留**：派生量都能从原图重算，反过来不行。仓库里的 `dataset/`
只放 MP4 + 标注 + meta，原图旁车留在采集电脑上（见 `dataset/README.md`）。

---

## 6. 红线（照做，不然会出事故或白干）

1. **不要未经用户授权烧录固件。** 烧前用 `esptool read_flash` 整片备份，
   `idf.py -p COM3 -b 460800 app-flash`；烧完别用启动日志的 “Compile time”
   判断版本（增量构建不刷新那行），要看行为或重新校验。
2. **不要给机器人发运动/控制命令**（`/cmd`、`/mode`、`/step` 带控制意图等）。
   它现在是躺着/桌上的状态，采集只需要只读 `/capture`。
3. 机器人保持静音（`output volume forced to 1`）。
4. 打开 COM3 前先把 DTR/RTS 置低，否则会复位机器人；
   **同一个 COM3 不能被两个脚本同时打开**（`PermissionError(13)`）。
5. 写文档/代码一律 UTF-8 无 BOM，写完跑 `_check_encoding.py`
   （这个项目因为编码事故毁过 4 个文件）。
6. 采数据前把机器人**立起来**并固定好（当前是平躺状态）。
   立起来属于物理操作，如果机器人不在你手上，先和用户确认。

---

## 7. 常见问题

| 现象 | 检查 |
|---|---|
| `/status` 连不上（curl 返回 `000`） | 电脑热点是否开；网卡 `WLAN 3` 是否有 `192.168.137.1`；热点是否 2.4 GHz WPA2；等 15 s 再试 |
| 机器人亮着但一直不连 | 热点 SSID/密码是否被改过（改了要按 §4 重新配网）；串口日志是否有 `Found AP` / `No AP found, next scan` |
| 网页预览 502 | 机器人是否在线；服务启动参数 `--robot` 是否对；`curl /capture` 是否 200 |
| 刚点开始录制时预览报“录制刚开始，还没有画面” | 正常，等第一帧（<100 ms）后预览会显示刚采到的画面 |
| 视频在浏览器里放不了 | 服务发布的是 H.264 MP4（有 ffmpeg）或 WebM VP8（无 ffmpeg）；不支持的编码请装 ffmpeg 重录 |
| 录制帧率不稳 | 采集是自由跑（约 17–18 fps），输出按所选 fps 最近邻重采样；看 `/api/recording` 的 `capture_fps` 与 `frames` |
| 端口 8780 被占用 | `Get-NetTCPConnection -LocalPort 8780 -State Listen` 查 PID；或换 `--port 8781` |
| 机器人 IP 不是 .50 | 固件静态兜底应给 `.50`；用 `arp -a` 或路由器/热点客户端列表确认 |

---

## 8. 相关文档

* 仓库总手册（架构、端点、踩坑 42+ 条）：[`AGENTS.md`](../../AGENTS.md)
* 采集与标注规范：[`COLLECTION-AND-ANNOTATION.md`](COLLECTION-AND-ANNOTATION.md)
* 标注范本：[`EXAMPLE-TAKE.md`](EXAMPLE-TAKE.md)
* 热点自动重连修复记录：[`WIFI-AUTOCONNECT-FIX.md`](../../WIFI-AUTOCONNECT-FIX.md)
* 采集/标注服务说明：[`VIDEO-ANNOTATOR.md`](../../VIDEO-ANNOTATOR.md)