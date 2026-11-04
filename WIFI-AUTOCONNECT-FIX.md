# 机器人自动连热点：根因、修复与烧录验证（2026-11-05）

> 起因：用户反馈"机器人不会自动连上 PC 热点"（m01247）。
> 结论：**是机器人固件的问题，不是 Windows 热点的问题。已烧录修复版并现场验证。**

## 1. 现象

- Windows 移动热点 `qasd`（2.4 GHz / WPA2）开启，`WLAN 3` = `192.168.137.1/24`，驱动日志
  `mtkwecx` 有 `4000 HOTSPOT GO`，但机器人始终不出现；
- PC 侧 `netsh wlan show networks` 只列出 `SZTU-student`（Win11 定位限制导致扫描列表
  本来就看不到热点 SSID，**不能**据此判断热点没广播）；
- 机器人串口只有每 10 s 的 SystemInfo（`free sram` / IMU），没有任何 WiFi 扫描日志；
- `/status` 全超时（curl 返回 000）。

## 2. 根因（两层）

1. **机器人跑的是 11-03 14:33 的旧固件。** 旧 `WifiBoard::OnWifiConnectTimeout` 在
   60 s 内没连上就 `StopStation()` 并进入配网模式，之后**再也不扫描**。所以只要开机时
   热点没就绪（或热点中途重启超过 60 s），机器人就永久离线，直到手动复位。
2. `wifi_station.cc`（`78__esp-wifi-connect`）在找不到 AP 时按 **10 s → 300 s 指数退避**
   重新扫描；即使没超时，热点晚开也会等很久才重连。

## 3. 修复（Hover 专用，`rig_omni/main/boards/common/wifi_board.cc`）

```cpp
#if CONFIG_BOARD_TYPE_HOVER
    config.station_scan_min_interval_seconds = 5;
    config.station_scan_max_interval_seconds = 15;
#endif
```

- Hover 分支不再启动 60 s `connect_timer_`，`OnWifiConnectTimeout` 只打日志
  `"WiFi still scanning for saved hotspot; retry remains active"`，**不 StopStation、不进配网**；
- 静态 IP 兜底改到 `WifiEvent::Connecting`（真的发现保存的 AP）后才开始计时，避免在热点
  不存在时提前占坑；12 s 后 `ApplyStaticIpFallback()` = `192.168.137.50/24 gw 192.168.137.1`；
- 连上后 `esp_wifi_set_ps(WIFI_PS_NONE)`，关闭省电。

非 Hover 板卡逻辑保持原样。该改动在 2026-11-04 21:39 的 `build/rig-hover.bin` 里。

## 4. 烧录记录

| 项 | 值 |
|---|---|
| 固件 | `rig_omni/build/rig-hover.bin`，3,089,376 B，mtime 2026-11-04 21:39:36 |
| 烧录前整片备份 | `D:\robot_work\_fw_backup_20261105_old.bin`（4 MB，含 bootloader / 分区表 / otadata / NVS） |
| 命令 | `idf.py -p COM3 -b 460800 app-flash`（esptool 4.12.0） |
| 结果 | 写入 3,089,376 B（压缩 1,789,509 B）到 `0x20000`，39.6 s，`Hash of data verified.` |
| 烧录后 | 硬复位；`/status` 200；`/capture` 200（约 13.3 KB JPEG） |

回滚（如需）：

```powershell
& 'G:\esp\.espressif\python_env\idf5.5_py3.11_env\Scripts\esptool.exe' `
  -p COM3 -b 460800 write_flash 0x0 'D:\robot_work\_fw_backup_20261105_old.bin'
```

## 5. 验证（决定性证据）

**场景 A：机器人开机时热点是关的（旧固件必死）**

1. 关热点 → 串口复位机器人；
2. 热点关着的 80 s 内 `/status` = 000；
3. 再开热点。

串口日志（`_serial_boot_nohotspot.txt`）：

```
I (4381)   WifiStation: No AP found, next scan in 5 seconds
I (11794)  WifiStation: No AP found, next scan in 10 seconds
I (24205)  WifiStation: No AP found, next scan in 15 seconds
I (41617)  ... 15 seconds
I (59029)  ... 15 seconds
I (76441)  ... 15 seconds
I (93853)  WifiStation: Found AP: qasd, RSSI: -36, Channel: 11
I (95251)  wifi:connected with qasd
I (105855) WifiStation: Got IP: 192.168.137.50
```

- 60 s 处**没有** "entering config mode"，扫描一直在继续；
- 热点开启后 **18.3 s** 自动上线（`ROBOT_ONLINE_AFTER_HOTSPOT_START=18.3s`）。

**场景 B：已连接状态下热点长时间断开（85 s）再开**

- 断开后重试 5 次（每次约 2.4 s），然后 `next scan in 5/10/15 seconds`；
- 热点恢复后 **7.7 s** 自动重连并拿到 `192.168.137.50`。

**烧录后稳定性**：连续 10 次 `/status` 全部 200，3 次 `/capture` 全部 200（13.3 KB 左右），
`heap≈8,153,508`，`cfc.active=0` 仍是占位。

## 6. 注意：不要用 "Compile time" 判断固件版本

烧录后启动日志仍打印：

```
I (114) app_init: Compile time: Nov  3 2026 14:33:14
```

这是 ESP-IDF 增量构建时 `app_desc` 里的编译时间字符串没重新生成导致的**陈旧字段**，
不代表烧进去的是旧 app（`app-flash` 已校验 hash，且扫描行为 5/10/15 s 只在修复版里存在）。
判断运行版本请用**功能行为**（扫描间隔、60 s 不配网），或者重新 `idf.py build` 后再烧。

## 7. 现场脚本

- `_hotspot_control.ps1`：WinRT 开关热点（`-Action stop|start|status`）；
- `_serial_reset.py`：串口拉 EN 复位机器人（COM3 必须独占）；
- `_serial_boot_watch.py`：复位 + 抓 110 s 启动日志；
- `_serial_watch_scanfix.py`：被动抓 155 s 串口日志。

**同一个 COM3 不能同时被两个脚本打开**，会 `PermissionError(13)`。