# HANDOFF · 文文离线攻坚

> 执行契约：`GOAL-offline-sprint-2026-11-24.md`。项目手册：`AGENTS.md`。
> 数据包根目录 `PACK_ROOT = G:\robot_work_offload\cloud_train\`（本文件的同步副本在 `PACK_ROOT\LEDGER.md`）。
> 每轮开工：先读本文件的状态头和轮次记录的最后一条，再读契约里对应工作点，不重复吞历史。

## 状态头（每轮必更新）

```text
整体状态: IN PROGRESS
当前前沿：F2（A3 几何赛道纵向切片——K-3 是唯一非死的证伪结果）
下一动作: 1) 建 pc/pack/sealed.py 唯一入口与训练入口 tripwire；
          2) 生成 splits\*.json（带哈希）与预注册划分规则；
          3) 回归测试改用非封存夹具或登记访问；
          4) O-003 完成后才允许按实测相机重训 GeoNet / 单目深度（见 D-009）
最后更新: 2026-11-25 15:22
机器人:   R4 没有碰。最后一次只读探测 2026-11-25 00:19：站立、手动模式、/cfg 为开机默认、
          imu_zero = −2°。现在是否仍在线未知；下次用之前先跑 pc/s2r/readonly_probe.py。
```

## 完成矩阵

状态取值：`COMPLETE / PARTIAL / MISSING / parked / N/A`。证据级别：T0–T4（机器人行为）、E2 / E4（数据与包）。

| 编号 | 用户可见结果 | 正常入口 | 失败/恢复路径 | 证据级别与链接 | 状态 | 缺口 |
|---|---|---|---|---|---|---|
| O-001 | `BASELINE.md`：资产清单 + 带命令的基线数字 + AUD 缺陷 | `pc/pack/inventory.py` → `BASELINE.md` | 只读脚本，失败即重跑 | E2；`PACK_ROOT\BASELINE.md`、`canon\assets.md`、`findings\raw\*.log` | COMPLETE | 资产清点第三次运行完成（27 组、91,558 文件、46.7 GiB、65,131 帧、802 组完全重复；EXIT_RC=0）；`BASELINE.md` 已写；3 个旧报告数字已从真实入口复现（航向仿真 p90=3.0°、click-to-go 30/30 到点、手势 macro-F1=0.8051）；全仓回归 25/25 原样记录 |
| O-002 | `robot_truth` 真值文件 + 常量处置清单 | `pc/robot_truth.json`；`python pc/pack/const_scan.py --check`；`run_all_tests.py --only truth` | 只读扫描，新冲突即退出码 1 | E2；提交 `39db690`；`findings\CONST_SCAN.md` | COMPLETE | 真值 44 条事实（各带值/方法/日期/出处/置信/层级）+ `model_artifacts`；扫描 46 处命中，43 处冲突全部处置（保留并说明 30 / 已标注 11 / 休眠 2），0 处未处置；s2r、`real_robot_bridge`、MoGe-2、稠密场景与新训练改读真值；门禁 `truth` 16 项。遗留的不是本项缺口而是 AUD-015/016：已发布权重都要按实测相机重训（D-009） |
| O-003 | 封存集隔离 + tripwire + `splits\*.json` | `pc/pack/sealed.py`、`make_splits.py`、`pc/tests/test_sealed.py` | 直接改入口参数即可回退 | E2；`sealed/manifest.json`、grep 审计 JSON、7 项 pytest | COMPLETE | mirror+sha256（250 文件 verify ok）、唯一入口+访问日志、4 个划分清单、11 个训练入口 tripwire、关闭式 grep 审计（41 文件四分处置，D-010）、回归测试改非封存夹具、unified app 默认输入改非封存；残留：派生缓存含 cb8455 特征 → F2 重构 |
| O-004 | `sensor_domain.json` + 转换函数 + 对抗 AUC | `pc/pack/sensor_measure.py`、`sensor_domain.py`、`adversarial_auc.py` | 重跑测量脚本即重建 | E2；`canon/sensor_domain.json`、`adversarial_auc_2026-11-25.json`、7 项 pytest | COMPLETE | 实测噪声/邻像素相关/暗角/色调/JPEG 块效应；转换链 7 步带随机化；对抗 AUC：fist 1.0→0.935、palm 1.0→0.984（转换后仍可分——域差主要在内容/尺度不在噪声）；金丝雀 YuNet：真机人脸召回 1.0 vs 改造公开帧 0.93 |
| O-005 | K-1 / K-2 / K-3 证伪结论 + `FRONTIER1_REPORT.md` | `k1_fist_transfer.py`、K-2 配对分析、`k3_desk_geometry.py` | 重跑脚本 | E2；`k1/k2/k3` JSON + `findings/FRONTIER1_REPORT.md` | COMPLETE | K-1=UX（公开特写拳全帧不迁移，AUC≈0.48 双臂）；K-2=U0/no-go（弱敲幅度合成剧毒 75.4%→36.9%）；K-3=U1-weak（平面角判决随训练规模翻转=主指标不可靠；平整度 MAD 与暗区比率跨规模一致利好合成；纯真机训越久几何越差=伪 GT 噪声过拟合） |
| O-006 | 真机资产规范库 + 清单 | `canon/assets.md/json` + `file_hashes.json` | 重跑 inventory.py | E2 | PARTIAL-strong | 27 组资产逐组登记路径/帧数/来源/许可/隐私/划分归属，逐文件 sha256；802 组重复已识别（Label Studio/早期手部图=字节级副本）未物理去重（防破坏引用）；缺：逐样本反查抽查审计 |
| O-007 | 公开数据许可台账 + 效用分级 | `canon/public_sources.json`（12 行机器可读）+ 核实报告 | 按 JSON 执行 | E2 | COMPLETE-ledger（未下载） | 每行 URL/许可/门槛/补哪格/效用分级齐；K-1 后全部按裁剪改造重估：P12 YCB-Video（黑物格、零门槛）与 P03 IPN Hand（A2 动作线索、零门槛）优先；P05/P09 不混合条款跳过；consent_needed 行留给用户本人点击 |
| O-008 | 合成数据生成器族 | （待建） | | E2 | MISSING | 全部 |
| O-009 | A1 拳头/手部数据集 | （待建） | | E4 | MISSING | 全部；起点缺陷见 AUD-018（idle 类 3,097/3,098 是手势图） |
| O-010 | A2 双敲数据集 | （待建） | | E4 | MISSING | 全部 |
| O-011 | A3 桌面几何数据集 | `pc/end_side/train_mono_depth_pack.py` + 数据卡 | 重跑 job | E2；job 包 `jobs/a3_depth_v1` 足量排练 PASS（synth abs_rel 0.0486、MAD 33.3mm）、`findings/F2_SLICE_A3.md` | PARTIAL（切片级闭环） | 矩阵覆盖 pitch×3、桌沿近/远、强制黑物 35%、薄负样本 25%；缺口诚实登记（深渊不可表达、光照/材质未控、真机无日光格）；评测 MAD/暗区比可信、平面角存疑、桌沿距离留 T4 |
| O-012 | 三条赛道学习曲线 | （待建） | | E4 | MISSING | 全部 |
| O-013 | 数据卡 + 建包流水线 + 采集套件 | `pc/pack/jobrun.py`、A3 数据卡范本 | | E2 | PARTIAL | job 包格式 + 本地排练 harness 落地（冒烟 PASS、receipt 自动回流 findings/raw）；采集套件未动 |
| O-014 | `CLAIMS_LEDGER.md` | 断言提取：`findings\CLAIMS_extracted.md`（子代理，只读） | | E2 | PARTIAL | 提取停在 1,266 行 / C866，末尾仍有 `<!-- CONTINUE -->`；续做子代理已被用户停掉；`CLAIMS_LEDGER.md` 未建 |
| O-015 | 数字孪生 v2 + 留出验证误差表 | （待建） | | T0–T3 | MISSING | 全部 |
| O-016 | 末次验收离线取证 | `pc/s2r/session_analyze.py <session_dir>` | | T1/T3 | PARTIAL | 证据已采：`runs_end_side/s2r_live/session_*`（rest / head / heading / pulse / ramp，500 Hz × 34 列）与 `probe_*`，结论在 AUD-002…011、017。缺：11-24 早上 −12.6°/s 自旋的根因仍未知（当时没有任何遥测，AUD-005）；取证报告未成文 |
| O-017 | 控制策略再训练云端 job | （待建） | | T0 + E4 | MISSING | 全部 |
| O-018 | Sim2Real 上机门禁清单 | （待建） | | T3 | MISSING | 全部 |
| O-019 | 边缘情况目录 v1 | （待建） | | E2 | MISSING | 全部 |
| O-020 | 可复现用例库 | （待建） | | E2 | MISSING | 全部 |
| O-021 | 修复与加固 | （待建） | | E2 | MISSING | 全部 |
| O-022 | 模糊与性质测试 | （待建） | | E2 | MISSING | 全部 |
| O-023 | 更聪明的行为 | （待建） | | T0/T1 | MISSING | 全部 |
| O-024 | 行为级延迟预算表 | （待建） | | E2 | MISSING | 全部 |
| O-025 | PC 侧链路优化前后对比 | （待建） | | E2 | MISSING | 全部 |
| O-026 | 固件与协议方案（备而不烧） | （待建） | | 编译 | MISSING | 全部 |
| O-027 | 软延迟对策 | （待建） | | T0 | MISSING | 全部 |
| O-028 | 延迟真值测量法 | （待建） | | E2 | MISSING | 全部 |
| O-029 | 云端任务包 | （待建） | | E2 | MISSING | 全部 |
| O-030 | 端到端排练（隔离 Linux） | （待建） | | E4 | MISSING | 全部 |
| O-031 | 回流落地脚本 | （待建） | | E2/E4 | MISSING | 全部 |
| O-032 | 云端作战计划 + `README_FIRST.md` | （待建） | | E2 | MISSING | 全部 |
| O-033 | T4 真机验收协议 | （待建） | | E2 | MISSING | 全部 |
| O-034 | 文档、经验笔记、独立总审查 | （待建） | | E2 | MISSING | 全部 |

## 决策记录

- **D-001（2026-11-24）原始日志不改写**：命令输出原样存 `PACK_ROOT\findings\raw\`，其中的历史标识符（套件 id、路径）保持原样，因为它们是证据而不是我写的文本；我自己写的代码/注释/JSON/文档一律不用那条废弃简称，历史文件和既有路径不改名。
- **D-002（2026-11-24 22:30）机器人边界由用户本人放宽**：契约 §9 的"零接触"是默认值。用户主动把机器人连上网并告知，因此只读访问（`GET /status`、`GET /capture`、TLM1 UDP 订阅）视为已授权并已使用。通电运动会改变物理世界且 11-23 有过摔倒，所以等用户回"跑"再执行；烧录仍按契约不在范围内；不开串口。
- **D-003（2026-11-24）数据包工具放 `pc/pack/`**，产物只进 `PACK_ROOT` 和被 `.gitignore` 排除的目录；一律经 `pc/pack/benign.py` 以 BelowNormal 启动（启动器设不上优先级就拒绝运行）。
- **D-004（2026-11-24）先做只读探测再谈运动**：新增 `pc/s2r/readonly_probe.py`（只有 GET 和 UDP 订阅，没有任何 POST），把 TLM1 的 34 个浮点全部落盘。原因：历史日志从未记录站立时的 roll 和扭矩，孪生和"末次自旋"取证都需要它们。
- **D-005（2026-11-24 23:35）通电运动的授权与做法**：用户原话"随你便，别把机器人搞掉下桌子就行"。做法：每个动作都包在 `pc/s2r/guarded_session.py` 的围栏里（500 Hz 逐包判定 + 看门狗，位移包络 / 倾斜 / 航向 / 速度 / 遥测丢失，触发即"归零→切手动→用遥测里的 `src` 核实"）；围栏先过 18 项回放测试（含真实误触发夹具）再上桌；每个阶段的脚本先在 `pc/s2r/mock_robot.py` 上端到端跑通。幅度从小到大，一次只做一个阶段。
- **D-006（2026-11-25 00:15）今晚不做大幅平移**：桌面右侧近处有大物体、前方有笔，而且 2 cm 的位置阶跃已经证明会激起前后摇摆（俯仰到 6.8°）。平移类实验留到清空桌面、有人在旁边的 T4 会话。
- **D-007（2026-11-24）测试期望的 HaGRID 路径用目录联接补上**：`pc\gesture\data\raw\hagrid\hagrid-sample-30k-384p` → `G:\stream_memory_data\gesture_raw\hagrid\hagrid-sample-30k-384p`（与既有 `synth` 联接同样做法；`pc\gesture\data\` 被 gitignore）。没有改任何测试。
- **D-008（2026-11-25）仓库里有三种相机，不许混用**：(1) **实测相机**——`pc/robot_truth.json`（75.5° / 0.135 m），给真机链路、把相机当输入的通用模型（MoGe-2）和所有新训练；(2) **已发布权重的训练相机**——记录在权重或 `model.json` 里（GeoNet 65°、单目深度 v1 与高斯溅射 v2 60°/0.115 m、稠密溅射 v3 72°/0.115 m），加载它的代码必须用它；(3) **合成或虚拟相机**——某个渲染器或某个模型的输入定义，标注清楚即可。起因：名为 `REAL_FOV_H_DEG` 的 65° 其实是 MoGe-2 从画面内容猜的。把它改成真值以后，端侧套件立刻暴露出混用——16 个 GeoNet 实验脚本会把 75.5° 塞给 65° 的权重（Dynablox 门禁里水杯点数 209→148），已发布的单目深度 v1 在实测相机下 AbsRel 从 0.046 变成 0.234。处置：这 16 个脚本改用 `GEONET_CKPT_FOV_H_DEG`（含义自述的名字），v1 模型包用 `RELEASED_V1_*`，门禁从它加载的权重/`model.json` 里读相机；测试 `test_robot_truth.py` 把这条规则钉住（已发布权重不得被喂实测相机、训练脚本不得从已发布相机的常量起步）。代价：这些已发布权重在真机上的精度不是文档里那个数（AUD-015/016）。
- **D-009（2026-11-25）按实测相机重训排在 O-003 之后**：GeoNet-240 本机重训约 40 分钟（11-22 的日志），规模上算"本机小规模"；但 `train_hover_geonet_240.py` 结尾会直接在两个封存集上评估，而封存集的访问日志和 tripwire 还没有。顺序：O-003 → 重训 → 经登记的 `--final-eval` 路径评估一次。在那之前已发布权重照旧，警告照打。
- **D-010（2026-11-25）封存访问规矩（关闭式审计后的裁定）**：全仓 grep 审计（`findings
aw\sealed_grep_audit_2026-11-25.json`，41 个文件含封存标记）分四类处置：
  (1) **训练入口 11 个**全部有 tripwire——不带 `--final-eval` 直接 `SealedAccessError` 退出，通过则写一行 `sealed_access.log`（含脚本、参数、目的、git HEAD）；
  (2) **回归测试一律改用非封存夹具**：`test_robust_contour_slam` test_5 从 cb8455 改 65a54d，`test_hover_geonet_240` 延迟冒烟从 empty_d0.5 改 none_d0.5；只读历史报告 JSON（冻结数字）不算访问封存集， grandfathered；
  (3) **活跃应用 `pc_unified_brain_app` 默认输入从封存的 empty_d0.5 改为 none_d0.5**，`--input` 指向封存路径时必须显式 `--final-eval`，文档示例同步改；
  (4) **其余约 20 个一次性 proto/showcase 脚本宣布冻结**：它们不再运行；将来任何人要跑，必须先经 `sealed.py` 登记。强制执行靠 (1)-(3) 的入口守卫 + 审计 JSON + 完成门禁的干净访问日志，而不是逐个改写冻结脚本。
  已知残留泄漏面（后续切片处理，不是借口）：`_knock3d_cache_v4` 等派生缓存含 cb8455 特征，旧 6 折 LOO 把 cb8455 当训练折——F2 重构双敲赛道时必须改成"dev 训练、sealed 只终评"。

## 已确认的审计发现

- **AUD-001（已更正）静止时的轮扭矩不是差动**：`tor1 = −temp_u + yaw_u`、`tor2 = +temp_u + yaw_u`（两电机前进符号相反）。2026-11-24 22:36 的读数 ∓12.9 是共模平衡扭矩 `temp_u = +12.9`、`yaw_u = 0`。我第一版把它读成"持续差动扭矩"，那是错的。
- **AUD-002 "4.6 cm 平衡偏差"不是常数，是固件 LQR 的稳态偏移**：`temp_u = −(1300·x_e + 5·(v−vx) + 25·(pitch − imu_zero) + 0.4·dq)`，`x_e = wheel_x − stable_pos`，`imu_zero = −2°`。静止时 `1300·x_e + 25·(pitch+2) + temp_u ≈ 0`，所以位置偏移 ≈ `(25/1300) × (机械平衡角 − imu_zero)` = **每差 1° 偏 1.92 cm**。三次独立静止读数都吻合（俯仰 0.19°→4.2 cm；1.0°→6.80 cm；2.84°→9.56 cm），回归测试 `test_firmware_lqr_identity_on_real_rest_log` 把它钉住。`link.Robot.drive_cfg()` 运行时现算所以不受影响；文档里的 4.6 cm 和孪生里的常数受影响。
- **AUD-003 站立 roll 是原始加速度计角，会有单点尖峰**：`roll = atan(...)` 逐样本直算、无滤波。头舵机起步时 2 个样本（6 ms）从 −4.9° 跳到 +6.29°（2026-11-24 23:46，围栏因此误触发一次，安全停车已核实）。围栏已加 60 ms 去抖；夹具 `pc/s2r/testdata/head_start_spike_tlm_20261124.csv`。站立 roll 本身在 −5.0° 左右（早上跟人日志的中位 +1.4°/−2.7°、范围到 −12°），不是异常。
- **AUD-004 固件运行时配置 `fov_h = 60`**（`GET /cfg`），实测 75.5°；同一份配置里 `yaw_lead_max = 35`、`pos_lead_max = 0.20`、`vx_max = 0.30`、`vyaw_max = 90`，都是 11-24 之前没有压过的默认值。`/cfg` 只在内存，重启复位。归 O-002 处置。
- **AUD-005 零漂补丁有效**：多次 8–30 s 窗口航向漂移在 ±0.02 °/s 内，遥测 447–500 Hz、0 丢包。11-24 早上的"−12.6 °/s 自旋"此刻不存在，原因仍未查到（当时没有遥测记录）。
- **AUD-006 全仓回归的真实基线是 24/24，不是 21/23**：`realimg`、`e2ereplay` 失败只是因为测试期望的目录不存在；联接后两项都通过（原样日志 `PACK_ROOT\findings\raw\run_all_tests_hagrid_link_2026-11-24.log`）。诊断项 `obsdomain` 按设计为红。
- **AUD-007 位置超前量 ≤ 2–3 cm 时，机器人不平移，而是前后摇摆并改变俯仰**（2026-11-25 00:07，T4）：开环脉冲 `vx=+0.05 m/s × 0.4 s`（位置目标前移 20.9 mm）。俯仰 0.98→6.77°（围栏在 6.70° 触发），1.12 Hz 摇摆，轮子 ±1 cm；停指令后约 45 s 才静止。净结果：轮子 **−4.4 mm（反方向）**、静止俯仰 +0.98→+2.84°、超前量 7.04→9.56 cm——三点都落在 AUD-002 的平衡线上。这就是"前进不动"和"前后抖"的真机解剖。更小的 5 mm 脉冲完全不动。证据：`runs_end_side/s2r_live/session_20261125_000722_pulse/`、`probe_20261125_000827_after_pulse/`、`probe_20261125_001013_decay_watch/`。
- **AUD-008 头舵机 20–30° 阶跃对平衡的扰动比文档大**（T4）：轮子前后晃 2–3 cm，机身航向被带偏 2.7–4.5°（小于静摩擦带，纠不回来），单轮扭矩峰值 95–118（平衡扭矩贴着 ±100 上限）。围栏记录最大位置偏移 5.25 cm。死区复测：回零指令从负侧停在 −8.2°/−6.3°，从正侧停在 +3.1°/+5.4°（带宽约 13.6°）。证据：`session_20261124_235030_head/`。
- **AUD-009 航向阶跃复测（T4）**：两轮 12 次原地转身，停止误差 0.44–4.32°，全部是"差一点没转到位"、零冲过头，峰值超前量 9.2–13.0°，俯仰 ≤ 2.0°。与早上的 0.8–3.8° 一致，但这次有 500 Hz 全量遥测。证据：`session_20261124_235702_heading/`。
- **AUD-010 视场角第三次独立复核**：IMU 在头上，遥测的 `yaw` 就是相机绝对方位（不要再叠加 `q_head`）。头舵机阶段 7 对保持帧的水平位移 ÷ `Δyaw` = 2.0–2.9 px/°（均值约 2.5），与 2.705 px/°（75.5°）一致。
- **AUD-011 固件 LQR 公式在运动数据上成立**：含 ±10 摩擦凸起与 ±100 限幅，线性区 RMS 1.24、整体 RMS 2.52（最大 17 在凸起切换处）。孪生的控制器侧可以直接用这条公式。
- **AUD-012 历史真机日志的信息缺口**：`s2r_morning/*/motion_log.json` 只有 30 Hz × 6 列（时间、目标、相对航向、超前量、vyaw、俯仰），没有 roll、扭矩、轮速、`stable_pos`；站立时的 roll 和扭矩此前从未被记录过。`guarded_session.py` 的 `tel.csv` 补上（500 Hz × 34 列）。
- **AUD-013 两条 11-15 take 的原始帧不在这台机器的约定位置**：契约 §1.5 说 6 条 take 的原始帧都在 `video_annotation_data\videos\<take>_source\`，实际那里只有 4 条 11-07 和 2 条 11-05 调试片；`65a54d`、`cb8455`（封存集）只有 `dataset\takes\` 里的 15 fps H.264 MP4。`videos\` 目录的修改时间是 2026-11-07 21:52，说明 11-15 的四条 take（含两条未标注短测试片 `ac8283`、`84575a`）**从未写进过这个目录**——当天用的是别的数据根目录（元数据里机器人地址是 `192.168.137.137`，不是平时的 `.50`）。后果：这两条 take 没有逐帧采集时间戳（`frames.jsonl`），像素是重编码有损的；封存集 `cb8455` 的"原始抓取未被改写"无法在本机证明。处置：规范库里这两条的"事实来源"记为 MP4，并列入"需要你做"。
- **AUD-014 "9 cm 的手在 1 m 处 21 px"是按旧视场角算的**：AGENTS §5.4-19 的 21 px / 10 px 对应焦距约 233 px（视场约 55°）。按实测焦距 155 px，是 **14 px（1 m）/ 7 px（2 m）**。拳头比文档说的小三分之一，A1 赛道的覆盖矩阵按 14/7 px 规划。
- **AUD-015 名为"实测"的视场角常数是猜的**：`pc/end_side/hd_moge_contour_engine.py` 定义 `REAL_FOV_H_DEG = 65.0`，被统一大脑/稠密场景链路的 24 个文件引用（含真机上跑过的 `pc_unified_brain_app.py`、`cliff_detect.py`、`real_robot_bridge` 的邻居）。`hover_geonet_240.py` 的注释写明来历是"MoGe-2 跨多场景反推均值 ~64.5°"——让模型从画面猜的，不是量的。焦距 188 px 对实测 155 px，这条 3D 链路在真机上把地面距离**系统性高估约 21%**（文档里的"rim≈0.28 m"真值约 0.23 m）。`pc/pack/const_scan.py` 的全仓扫描（修复后的最终结果）：46 处命中，3 处与真值一致，43 处冲突分布在 23 个文件、全部处置（保留并说明 30 / 已标注 11 / 休眠 2），代码里共出现过 9 种视场角 [50, 52, 56, 60, 65, 70, 72, 75.5, 120] 和 3 种相机高度 [0.115, 0.135, 0.2]。处置已完成，见 O-002 与 D-008。重训留到 O-003 之后（D-009）。
- **AUD-016 已发布的单目深度 v1 / 高斯溅射 v2 是按 0.115 m / 60° 训练的**（`model.json` 记录在案，Android 里的副本逐字节相同）：同一个合成场景、同一份权重，单目深度 v1 的 AbsRel 在它的训练相机下是 **0.0455**，在实测相机（75.5° / 0.135 m）下是 **0.2342**；高斯溅射 v2 在训练相机下 0.0166，实测相机下未测。门禁里的"AbsRel < 0.15"只对一台机器人并不拥有的相机成立。`test_mono_depth.py::test_04` 每次运行都打印这两个数。修复 = 按实测相机重训并重新导出（D-009）；Android 工程里的副本要用户本人同步。
- **AUD-017 慢速改 `imu_zero`，机身靠俯仰吸收，轮子基本不动**（T4，2026-11-25 00:16–00:19）：`imu_zero` 以 0.2 °/s 从 −2° 斜坡到 −1° 再回 −2°，围栏未触发（最大位置偏移 1.18 cm，俯仰偏离 ≤ 2.9°）。上行段：俯仰 **+0.73 °/°**、轮子 **+0.72 cm**、超前量 −0.19 cm/°；下行段：+0.95 °/°、+0.70 cm、−0.27 cm/°。按"轮子可以自由滚"预测的超前量斜率是 −1.92 cm/°，实测只有它的十分之一到七分之一。结论：AUD-002 的平衡线描述的是**静止下来之后**的位置偏移，不是机器人会沿着走的轨迹；在静摩擦带里调 `imu_zero` 改的是姿态而不是位置，消不掉静止位置偏差。证据：`runs_end_side/s2r_live/session_20261125_001651_ramp/`（`analysis.json`）。
- **AUD-018 手势数据集的 `idle` 类几乎全是手势图**：HaGRID 30k 采样集的 `labels` 列按检测框的顺序排列，并不是"手势手在前"。`pc/gesture/src/prepare_hagrid.py` 取 `labels[0]`，`gesture_common.py` 把 `no_gesture` 映射成 `idle`。被标成 `idle` 的 3,098 张图里**只有 1 张**真的只有 `no_gesture`；其余 **3,097 张都带一个真实手势框**（like 220、one 209、fist 173、palm 174、stop 165、stop_inverted 186，以及未映射的 peace/rock/ok 等）。fist 类因此少了 173 / 1,734 = **10.0%**（README 的 fist 1,561 + 173 = 1,734）。裁手用的是 YOLO 姿态里"举得更高的手腕"而不是标注框，所以一部分握拳图被当作 idle 训练。`prepare_hagrid.py` 文档串里"这 3,098 张是要求做手势但没做出来"的说法不成立。已用 `python -I` 只读独立复核（31,833 行、7,973 个 user_id）。影响大小未测：需要用修正标签（取非 `no_gesture` 的那个标签，或 `leading_hand` 对应的框）重训对比，归 A1 赛道（O-009）。发现者：许可核实子代理；本地采样集是 HaGRID **v1** 的第三方 30k 抽样（`cj-mills`，哈希与 Hugging Face 上的文件一致），许可以作者 PDF 为准。

## parked 清单

（空）

## 需要你做 / 需要你决定（只收必须用户本人的事，持续追加）

1. ~~要不要现在跑通电验收~~——已答复"随你便，别把机器人搞掉下桌子就行"（2026-11-24 23:35）。已按此执行，见 D-005。
2. **【10 秒钟，可选，能省掉 ±4° 的投影误差】相机标定**：打印 `pc/s2r/calib/` 里的 A4 棋盘格（待生成），举到机器人正前方 15–40 cm 处换几个角度停 20 秒。有了它才有焦距、畸变和相机安装俯仰的精确值。在那之前，点击落点的距离误差在 40 cm 处可到 10 cm。
3. **【下次 T4 会话前】清出桌面**：机器人四周至少 30 cm 内不放东西（今晚右侧 10–15 cm 处有显示器支架，正前方有笔和鼠标垫）。平移类验收（前进 6 cm、点击导航）需要这个条件。
4. 需要你本人同意许可的数据集（agent 不能替你点同意、替你注册）：**Something-Something v2 与 Jester**（Qualcomm 账号 + 研究用途许可）、**EgoGesture**（打印签字扫描、机构邮箱、等审批）、**H2O**（ETH 表单）、**Assembly101**（Hugging Face 登录过门）、**LeapGestRecog**（Kaggle 登录；许可只有 Kaggle 页面上的一个标签）、**HO-3D**（下载即接受"仅学术、不得修改"）、**FPHA**（Google 表单申请）。其中 Something-Something v2、Jester、EgoGesture、H2O、HO-3D 的许可**禁止与第三方数据混合或再分发**，不能进混合训练包，也不能上公开仓库——要用的话请你决定。细节在 `findings\PUBLIC_LICENSES_research.md` §2–§3、§8。
5. 到云端时登录、过验证码（魔搭开机拼图）。
6. 可选：用手机在机器人视角（离桌面约 13.5 cm 高、不俯仰）照脚本拍十分钟"桌面高度的握拳、占满画面的近距拳、由轻到重的敲击"。
7. 可选：花一刻钟复核接触级敲击候选标注（O-010 出包后）。
8. 按实测相机重训单目深度之后：`G:\robot_app\android\...\assets\models\` 里的 `hover_mono_depth_v1` / `hover_gaussian_splat_v2` 是按 60° / 0.115 m 训练的旧模型，新模型要你本人同步进去（我不碰那个目录；门禁会核对两处副本逐字节一致）。

## 轮次记录（追加）

### R1 · 2026-11-24 22:20–22:45 · 切片：接管 + 基线回归 + 机器人只读探测

- 契约和台账此前不存在，本轮创建；`PACK_ROOT` 创建。
- **全仓回归**（BelowNormal、单进程）：`GATES: 22/24 passed`，失败的是 `realimg`、`e2ereplay`（`pc\gesture\data\raw\hagrid\` 原图缺失），诊断项 `obsdomain` 同因。与文档的 21/23 一致（多出的 1 项是 11-24 新增的 s2r 套件）。原样日志：`PACK_ROOT\findings\raw\run_all_tests_baseline_2026-11-24.log`。**后被 AUD-006 修正：真实基线 24/24。**
- 测试套件不连机器人：逐个入口 grep 过 `192.168.137` / `COM3` / 串口，只有注释命中；`test_strategy_engine`、`test_telemetry_flywheel` 的 `urlopen` 打的是本机模型服务。
- 用户连上机器人后做了 12 s 只读探测，见 AUD-001…005（其中 AUD-001、AUD-003 的第一版结论已在 R2 更正）。
- 新增文件：`pc/pack/benign.py`、`pc/s2r/readonly_probe.py`、本台账。
- 目录普查：`PACK_ROOT\findings\raw\census_2026-11-24.txt`（真机数据约 0.4 GiB，公开数据 44.5 GiB）。

### R2 · 2026-11-24 22:45 – 2026-11-25 00:15 · 切片：真机窗口（用户送上门）——围栏工具链 + T4 证据

- 用户 22:30 连上机器人；我只读探测（12 s）后发现三处读数异常，**没有直接开动**，而是先问了一个是非题；用户 23:35 答复"随你便，别掉下桌子"。
- **工具链（全部先在 mock 与回放上验证再上桌）**：
  - `pc/s2r/tellog.py`：全速率 TLM1 记录（34 浮点，CSV 流式落盘）。
  - `pc/s2r/guarded_session.py`：围栏会话（rest / head / heading / ramp / drive / pulse）。围栏 500 Hz 逐包判定：俯仰偏离、roll 绝对值与偏离、位置包络、原始轮位移、航向包络、轮速、转向角速度、`robot_state`、遥测丢失、`ABORT` 文件。触发后"归零→切手动→用遥测 `src` 核实"。
  - `pc/s2r/mock_robot.py`：本机假机器人（同样的 UDP 遥测和 HTTP 端点 + 简化对象），供所有阶段脚本和以后的 T4 协议排练。
  - `pc/s2r/session_analyze.py`：从 `tel.csv` + `ctrl.jsonl` 出各阶段指标。
  - 测试：`test_guarded_session.py` 18 项（每条限制都要在回放上触发；真实站立日志和真实尖峰日志必须不触发；固件 LQR 恒等式），`test_session_on_mock.py` 6 项（每个阶段端到端，必须以手动模式 + 零指令收尾；围栏经真实通路触发后不得再有教师指令）。连同原有 `test_s2r.py` 9 项共 33 项通过。
- **真机实测**：rest 30 s；head 10 个保持段；heading 2 轮 12 步；pulse 5 mm（不动）与 2 cm（激起摇摆）。数字见 AUD-002/003/007/008/009/010/011。
- **一次围栏误触发**（AUD-003）：头舵机起步的 roll 尖峰，旧围栏 2 个样本就触发；安全停车核实无误；修正后用真实尖峰做回归夹具。
- **沿途修掉的坑**：Windows 上 UDP 套接字向已关闭端口发包后会收到 `WSAECONNRESET`（mock 的订阅线程曾因此静默退出）；`python3` 在本机是会挂起的商店占位程序（一条命令因此超时，已停掉，文件未被改动）；安全停车前先发一段零速度指令，防止被 TCP 重传延后的教师模式请求把固件切回去。
- 下一切片第一动作：`guarded_session.py --stage ramp --imu-zero-to -1.0 --ramp-rate 0.2`（1° 的小幅斜坡，验证平衡线是靠平移还是靠俯仰吸收）。

### R3 · 2026-11-25 00:16 – 14:10 · 切片：真值文件与常量对齐（O-002）+ 审计发现的独立复核

- **真机（只此一项，00:16–00:19）**：`imu_zero` 1° 斜坡，围栏未触发，结果见 AUD-017；00:19 只读探测收尾（站立、手动、`/cfg` 开机默认）。此后没有再碰机器人。
- **O-002 完成**（提交 `1b51236` 数据包骨架与资产登记、`39db690` 真值与相机对齐）：
  - `pc/robot_truth.json` + `robot_truth.py`：44 条事实，每条带值、方法、日期、出处、置信、验证层级；加载器拒绝没有出处的事实，也拒绝"没有值却声称有层级"的事实。另有 `model_artifacts`：每个已发布模型的训练相机、记录位置、与实测相机的差距。
  - `pc/pack/const_scan.py --check`：扫 `pc/` 与固件里的视场角和相机高度字面量。46 处命中，43 处冲突全部有处置理由，出现任何新冲突或过期处置即退出码 1。报告 `findings\CONST_SCAN.md`（含"已修复的 8 处"一节）。
  - 真机链路改读真值：`pc/s2r/link.py`、`controllers.py`（像素↔角度改成实测焦距的精确针孔形式：旧的线性比例把画面外推成 88.7°）、`real_robot_bridge.py`（62.5°→75.5°）、MoGe-2 提示（65°→75.5°）、通用几何默认值、五个训练脚本的起始相机。GeoNet 的教师缓存目录带上相机参数（旧目录改名为 `geonet240_h135_f65`），新权重的输出目录同样带参数，不会覆盖已发布权重。
  - D-008（三种相机）与 D-009（重训排在 O-003 之后）。AUD-015 补全、AUD-016 新增。
- **测试**：端侧 101 项通过（49 s）；s2r 33 项通过；新增门禁套件 `truth` 16 项。**全仓回归 `GATES: 25/25 passed`**（原样日志 `findings\raw\run_all_tests_2026-11-25_post_O002.log`；基线 24 + 新增 1），诊断项 `obsdomain` 按设计为红。中途两次真实的套件失败（`test_hover_geonet_240` 水杯点数 148 < 150；`test_mono_depth` AbsRel 0.234）都是"把实测相机喂给旧权重"造成的，按 D-008 修正后恢复，没有改任何阈值。
- **子代理**：两个都曾两次因 5 小时用量上限中断（重置时间 05:02、06:15）。许可核实在中断前已写完（`findings\PUBLIC_LICENSES_research.md`，10 节，12 个数据集有完整小节）；断言提取停在 1,266 行 / C866，末尾仍有 `<!-- CONTINUE -->`；14:00 起续做的子代理已被用户手动停掉，未产生可见续写。
- **独立复核**（我自己重跑，不引用子代理的数字）：HaGRID 标签顺序（AUD-018）用 `python -I` 只读读 parquet 复核一致。
- **资产清点**：第二次运行其实没有跑完——在最后一行打印处 `human(g.bytes())` 把整型属性当方法调用而崩溃（`findings\raw\inventory_run2_2026-11-25.log`），`canon\assets.*` 仍是第一次运行的产物。已修，并把哈希缓存改成每组落盘一次；第三次运行已在接手前完成（`inventory_run3_2026-11-25.log` EXIT_RC=0；27 组、91,558 文件、46.7 GiB、65,131 帧、802 组完全重复，560.2 s）。
- 下一切片第一动作：读清点结果 → 写 `PACK_ROOT\BASELINE.md`（资产清单、≥3 个旧报告数字的复现、`run_all_tests.py` 原文）→ O-003。

### R4 · 2026-11-25 15:02 · 切片：接管落盘 → O-001 BASELINE

- 15:02 接手。已读交接提示、契约全文、台账状态头与 R3、`AGENTS.md` §5/§8/§9 及相关经验笔记；未碰机器人、未开串口、未烧录。
- 现状核对：分支 `feat/embodied-skills-v3` ahead 4；除台账与 `pc/pack/inventory.py` 的未提交修改外，还有两个未跟踪文件 `HANDOFF-PROMPT-next-agent-2026-11-25.md`、`runs_end_side/e2e_replay_67f0c1_err.log`（先保留，不归入本次改动）。
- 资产清点第三次运行已完成：`inventory_run3_2026-11-25.log` EXIT_RC=0；27 组、91,558 文件、46.7 GiB、65,131 帧、802 组完全重复，耗时 560.2 s；`canon\assets.json` 已写出。
- 断言提取文件当前 1,266 行、到 C866，末尾仍是 `<!-- CONTINUE -->`；续做子代理被用户停掉后没有可见续写。
- 接手第一件事已执行：修正 R3 两条过时状态并落盘本条。随后读 `canon\assets.*`、选择 ≥3 个不碰封存集的旧报告数字复现，运行真实命令并写 `PACK_ROOT\BASELINE.md`。
- 15:20 O-001 收尾：`PACK_ROOT\BASELINE.md` 已写（217 行），含 27 组资产清单、封存资产预登记、全仓回归 25/25 原样记录、3 个旧报告数字的真实命令与输出、AUD-001…018 索引、未复现数字清单。三轮复现日志：`repro_sim_plant_2026-11-25.log`、`repro_click_to_go_2026-11-25.log`、`repro_gesture_eval_2026-11-25_utf8.log`。
- 验证：BASELINE UTF-8 检查通过；`_check_encoding.py` 输出 `all clean`；未触碰封存集、未碰机器人。
- 下一切片第一动作：实现 O-003 的 `pc/pack/sealed.py`、训练入口 tripwire 和带哈希划分。
- 15:45 O-003 第一切片落地：`pc/pack/sealed.py` 建立唯一入口；`--materialize` 已复制（不是移动）250 个封存文件到 `PACK_ROOT\sealed\mirror`，`--verify` 输出 `verify ok`；`pc/pack/make_splits.py` 已生成 `take_splits.json`（6 个 take）、`evidence_splits.json`（24 个目录）、`live_session_splits.json`（11 个会话）、`preregistered_rules.json`，逐文件 sha256。
- tripwire 已加到 9 个训练入口/数据构建入口：`train_hover_geonet_240.py`、`train_dense_splat_student.py`、`train_mono_depth.py`、`train_neural_gaussian_splat.py`、`train_hover_geonet_mv_selfsup_v4.py`、`train_double_knock.py`/`double_knock_data.py`、`train_double_knock_3d_physnet.py`、`train_double_knock_3d_v2_extreme.py`、`train_double_knock_3d_v4_sim2real.py`。无前 `--final-eval` 手动运行前 4 个均 rc=1 并抛 `SealedAccessError`，未写访问日志。
- 新增 `pc/tests/test_sealed.py` 7 项与 `run_all_tests.py` 的 `sealed` 门禁；验证：`pytest pc/tests/test_sealed.py` 7/7 PASS，`run_all_tests.py --only sealed` PASS，所有改动 `py_compile` 通过，`_check_encoding.py` all clean。
- 重要限制：O-003 仍为 PARTIAL。还没有做“全仓库所有直接打开封存路径”的关闭式审计；`eval_session.py`、`proto_3d_edl_traversable.py`、`test_robust_contour_slam.py` 等评估/原型入口未逐一改造；旧 `run_pipeline` 在显式 final-eval 后仍按历史 6 折流程处理 `cb8455`，后续必须把“训练 dev、只最终评估 sealed”重构清楚。回归测试究竟是改用非封存夹具还是登记访问，也还没写决策。下一步先补 D-010 决策与关闭式 grep 审计，再继续训练入口重构。
