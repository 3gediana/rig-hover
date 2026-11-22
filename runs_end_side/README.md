# runs_end_side —— 端侧大脑场景产物目录

每个 `*_vNN/` 子目录对应一个场景 runner 的产物（报告 JSON + demo MP4 + 日志）。
能力总表由 `pc/end_side/scenario_matrix.py` 自动汇总到 `scenario_matrix/`。

## 验证层级（诚实边界）

- **T0** 纯仿真（`DesktopArena3D`/`CliffDesktopArena3D` 合成渲染闭环）
- **T1** 仿真+真机录像片段回放
- **T2** 完整真机录像回放（里程计不动，验证感知/决策链）
- **T3** 本地 ESP32 兼容闭环（HIL）
- **T4** 真机验收 —— **本目录目前没有任何 T4 结果**（机器人离线）

## 目录索引（重生成命令）

| 目录 | 内容 | 重生成 |
|---|---|---|
| `pursuit_moving_cup_v18/` 见 `pursuit_v18/` | 追捕移动水杯 5 场景 | `pc/end_side/pursuit_moving_cup_v18.py` |
| `edge_survey_v19/` | 边缘勘测→任意点导航+越界拒跳 | `pc/end_side/edge_survey_nav_v19.py` |
| `knock_autopsy_v20/` | 双敲漏检解剖+阈值扫描 | `pc/end_side/knock_recall_autopsy_v20.py` |
| `e2e_replay/` | 完整真机录像 E2E 回放总结 | `pc/end_side/e2e_replay_summary.py <log> <take>` |
| `ensemble_knock_v5/` | 5 模型双敲集成对比 | `pc/end_side/ensemble_knock_v5.py` |
| `scenario_matrix/` | 能力矩阵 JSON/MD/HTML | `pc/end_side/scenario_matrix.py` |

## 入库策略

入库：`*.json` 报告、demo MP4、小日志。
忽略（`.gitignore`）：`_*/` 帧缓存、`*.npz`/`*.pt` 权重、`*.ply` 点云、
一次性的 `_*` 临时文件。

**仿真纪律**：runner 一律 BelowNormal 优先级、单进程串行（GeoNet+SLAM 很重，
多开曾把机器卡死）。后台跑务必先 `mkdir -p` 日志目录再 `tee`。
