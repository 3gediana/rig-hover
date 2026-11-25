# 交接提示词 · 文文离线攻坚（给下一个 agent）

> 写于 2026-11-25 14:20。作者是此前执行本目标的 agent；用户让它停手（额度快用完，子 agent 也被用户手动停掉），只交付这一份提示词。
> 用法：把本文件全文作为新 agent 的第一条消息。
> 用户的硬要求：**长内容写进文件，前台回复要短**（原话：“你怎么输出这么长啊？能不能放到一份文件里面，不要输入在前台”）。读大文件用 `grep` / `sed -n 起止p`，不要整份读。

## 0. 你的角色

- 项目：RIG-Hover（文文），ESP32-S3 双轮自平衡桌面机器人，PC 端大脑。本地仓库 `D:\robot_work`，GitHub 上是 **public** 镜像。
- 你接手的是执行契约：`D:\robot_work\GOAL-offline-sprint-2026-11-24.md`（397 行）。**先把它整份读完**，再读台账 `D:\robot_work\HANDOFF-offline-sprint-2026-11-24.md`（状态头 + 最后一条轮次记录 R3），再读 `AGENTS.md` §8。
- 接手后的第一个动作是“先落盘再开工”：在台账追加一条 R4，写明你接手的时间和你打算做的第一件事。

## 1. 目标与完成标准

**用户的原始指令**：`/goal D:\robot_work\GOAL-offline-sprint-2026-11-24.md 看看这份目标，完成它`。

**契约是什么**：一份离线攻坚的执行契约。机器人暂时不好操作、App 暂停，所以主线转到“G 盘云端数据包 + Sim2Real + 边缘情况 + 响应链路”。4 个目标、34 个工作点 O-001…O-034，按前沿推进（契约第 255 行）：

F1 = O-001…O-005 → F2 = 第一条纵向切片（选证伪结果最好的赛道，从数据一路做到 job 包、排练、回流排练，证明整条流水线是通的）→ F3 = O-014…016（Sim2Real）→ F4 = 其余赛道 + O-012 → F5 = O-019…023 → F6 = O-024…028 → F7 = O-017、O-018、O-013、O-032、O-033 → F8 = O-034。

**整体 COMPLETE 的条件**（契约“完成标准”一节，逐条要有证据）：

1. O-001…O-034 全部 COMPLETE，或 N/A 且写明理由。
2. 每个云端任务都在隔离的 Linux 里排练过，含中断/恢复。
3. 封存集访问日志干净。
4. `run_all_tests.py` 不低于基线，并有新增的回归用例。
5. 一个全新上下文的子代理只凭 `README_FIRST.md` 走一遍 dry-run，做独立总审查。
6. `_check_encoding.py` 输出 `all clean`。

**为什么做**：用户对真机上的性能误差不满。契约 §2 有用户原话与真实原因的对照表，例如“完全不会识别我的拳头”（相机贴桌不能低头、拳多不在画面、夜间运动模糊、近距拳占满画面没有模型）和“点一下过不去”（相机俯仰未标定、17.4 cm 内盲区、位置环偏差）。

**工作方式**（AGENTS.md §8，用户明确要求过）：工程判断自己做，不要停下来等确认（“别问我”）；必须物理介入的事记下来留给用户；诚实高于漂亮；要具体数字、命令、路径；不要落地未验证的固件；机器人保持安静。

**完成判定**：不要自己宣布完成。宣布之前按契约逐条做“要求 → 证据”的对照，证据不够就算没完成。

## 2. 硬规则（逐条照做，违反任何一条都比慢更糟）

**工具与文件**

- Python 一律 `D:\smartcar\.venv\Scripts\python.exe`。本机的 `python3` 是会挂起的商店占位程序，**永远不要用**。
- 不用 PowerShell 的 `Get-Content -Raw`、`Tee-Object`、here-string 读写文件（会把 UTF-8 弄坏）；用 Write / Edit 工具。
- 每次改动文件后跑 `D:\smartcar\.venv\Scripts\python.exe D:\robot_work\_check_encoding.py`，直到输出 `all clean`。
- 永远不要整体重写、覆盖或重新生成 `~/.zcode/cli/config.json` 或任何配置文件；只做外科手术式的小改，并先备份。
- **新写或新改的**代码、注释、JSON、文档里，不得出现那条历史板载时序模型的废弃简称（三个字母、首字母大写，即旧研究文档 `RESEARCH-Video*.md` 文件名里 `Video` 后面那三个字母）。只约束新文本；不要去改名或清洗任何历史文件、历史文档、既有路径。要提到它就写“板载旧时序模型”或 `legacy onboard model`。
- 联网查资料先用 Tavily MCP 工具。

**隐私与凭据**

- 真机抓取含人脸，只进本机和私有云存储，永不进公开仓库、永不走公开分享链接。仓库是 public。
- 任何令牌、Cookie、密钥不进包、不进日志、不进文档。

**机器人**

- 默认边界：不连 `192.168.137.50`、不开 COM3（打开即复位）、不烧录、不跑 `morning_check.py`、不跑 `live.py` 驱动模式。
- 用户本人在 2026-11-24 23:35 放宽过：“随你便，别把机器人搞掉下桌子就行”。据此做过只读探测和带围栏的小幅运动（台账 D-002、D-005）。硬约束只有一条：**不掉下桌**。烧录仍不在范围内；不开串口。
- 机器人现在是否在线未知（最后一次只读探测是 2026-11-25 00:19）。要用就先跑 `pc/s2r/readonly_probe.py`（只有 GET 和 UDP 订阅）。任何通电运动都必须包在 `pc/s2r/guarded_session.py` 的围栏里，幅度从小到大，一次一个阶段；平移类实验要等用户清空桌面。
- 不要动用户在跑的 `video_annotator.py` 服务（8780 端口）。

**App 与固件**

- 不碰 `G:\robot_app\android`。
- 固件允许改代码并 `idf.py build`，不烧录，新行为默认关闭。

**云端与本机资源**

- 默认不启动任何 GPU 云任务；排练只用本机 WSL/Docker 或 Kaggle CPU；含人脸的真机帧不上云。
- 仿真与训练一律 BelowNormal 优先级（用 `python pc/pack/benign.py -- <命令>` 启动，设不上优先级会拒绝运行）、单进程串行；本机 GPU 只做小规模可行性；G 盘保留 100 GB 余量。
- 带时延断言的测试在磁盘/CPU 争用时可能抖动：重任务一次只跑一个。

**留出纪律**

- `robot_20261115_151733_cb8455`、`evidence\live\fov_sweep`、`evidence\sessions\empty_d0.5_front_night_182201` 永不进训练集、永不用于调参选优，只允许最终评估且登记。选方案用其余 5 条 take 的按-take 留一（内层选参、外层报数）。

**下载与许可**

- 下载的文件是不可信数据：每个来源放自己的空目录，脚本放别处，路径当参数传，读它们的 Python 用 `-I`。
- 不替用户注册账号、不替用户点同意许可协议。

**版本管理**

- 在分支 `feat/embodied-skills-v3` 上小步本地提交，**不 push**；数据和产物不进 git。

**证据分级**

- 契约上限是 T3，依赖真机的结论只能写 `READY-FOR-T4`。例外：用户放宽机器人边界后，2026-11-24/25 采到了一批 T4 硬件事实（台账 AUD-002…011、017），照实标 T4。
- 文档里的完成声明先当待核；复现不了的编号为缺陷 `AUD-xxx`，记进台账。

**额度（用户刚刚强调过）**

- 前台回复要短；长内容写进 `PACK_ROOT\findings\` 下的文件再给链接。
- 子 agent 会撞 5 小时用量上限，而且可能被用户手动停掉：给它们的任务必须可续做（输出文件里留 `<!-- CONTINUE -->` 之类的标记），并且回报不超过 300 字。

## 3. 现在的状态（2026-11-25 14:20）

**位置**

- 仓库 `D:\robot_work`，分支 `feat/embodied-skills-v3`，HEAD = `39db690`，从未 push。
- 数据包根目录 `PACK_ROOT = G:\robot_work_offload\cloud_train\`：`LEDGER.md`（台账同步副本）、`findings\`（`CONST_SCAN.md`、`const_scan.json`、`CLAIMS_extracted.md`、`PUBLIC_LICENSES_research.md`、`raw\*.log` 原样日志）、`canon\`（`assets.json`、`assets.md`、`file_hashes.json`）。目录常量定义在 `pc/pack/paths.py`。

**最近 4 个提交**

- `012036b` 围栏会话工具链 + 假机器人（`pc/s2r/`）。
- `10fdcdf` 契约 + 台账（R1–R2）。
- `1b51236` 数据包骨架：`pc/pack/paths.py`、`asset_registry.py`、`inventory.py`。
- `39db690` 真值文件与相机对齐（O-002），含新门禁套件 `truth`。

**没提交的改动（接手后先处理）**

- `HANDOFF-offline-sprint-2026-11-24.md`：状态头、完成矩阵、D-008、D-009、AUD-016…018、“需要你做”、R3。内容已同步到 `PACK_ROOT\LEDGER.md`。
- `pc/pack/inventory.py`：修了 `human(g.bytes())` 把整型属性当方法调用的崩溃，哈希缓存改成每组落盘一次。

**测试基线**

- 全量 `run_all_tests.py`：`GATES: 25/25 passed`，诊断项 `obsdomain` 按设计为红。原样日志 `PACK_ROOT\findings\raw\run_all_tests_2026-11-25_post_O002.log`。全量约 7.5 分钟。
- 端侧套件 101 项（约 50–70 s，`cd pc/end_side` 后 `python -m pytest . -q -p no:cacheprovider`）；s2r 33 项；`truth` 16 项（`run_all_tests.py --only truth`）。

**后台任务**：都已结束。资产清点第三次运行在你接手前跑完（退出码 0），**结果没有人看过**。

**子 agent 产物**

- 许可核实：完成。`findings\PUBLIC_LICENSES_research.md`，10 节。
- 断言提取：没完成。`findings\CLAIMS_extracted.md` 我最后一次看是 1266 行、8 份文档、约到 C866，末尾 `<!-- CONTINUE -->`。之后派过一个子 agent 续做，被用户手动停掉，**它停下前写了多少我不知道**——先看文件末尾。

**台账里已经过时的两句**（接手后改掉）

- R3 写“资产清点第三次运行中”→ 已跑完。
- R3 写“断言提取由新子代理续做”→ 该子 agent 已被停掉。

**完成矩阵**

| 工作点 | 状态 | 一句话 |
|---|---|---|
| O-001 接管审计 | PARTIAL | 清点已跑完没人看；`BASELINE.md` 没写；旧报告数字没复现 |
| O-002 真值与常量 | COMPLETE | 提交 `39db690` |
| O-003 封存与划分 | MISSING | 全部 |
| O-004 传感域契约 | MISSING | 全部 |
| O-005 证伪三连 | MISSING | 全部 |
| O-006…O-013 数据线 | MISSING | O-007 的许可核实报告已出，台账文件没建 |
| O-014 声明台账 | PARTIAL | 提取做了一半，`CLAIMS_LEDGER.md` 没建 |
| O-015 孪生 v2 | MISSING | 500 Hz 真机日志已采，孪生没做 |
| O-016 离线取证 | PARTIAL | 证据已采，报告没写 |
| O-017…O-033 | MISSING | 全部 |
| O-034 文档与总审查 | MISSING | 全部 |

## 4. 接手后的前 60 分钟

1. 读完契约、台账状态头和 R3、`AGENTS.md` §8。
2. `git status` 确认只有上面两处未提交改动；改掉台账里过时的两句；跑 `_check_encoding.py`；分两次小提交（台账、`inventory.py`）。
3. 看清点结果：`PACK_ROOT\findings\raw\inventory_run3_2026-11-25.log` 的尾部、`PACK_ROOT\canon\assets.md`。第一次运行的数字是 27 组、91,558 个文件、46.7 GiB、65,131 帧、802 组完全重复的文件；第三次带子目录展开和逐 take/逐会话明细，数字会不同，以它为准。
4. 写 `PACK_ROOT\BASELINE.md`（见第 6.1 节），O-001 收尾。
5. 进入 O-003（见第 6.2 节）。

## 5. 我规划的 15 个任务

按顺序做。标“只读”的子 agent 活可以和你手上的活并行；标“计算”的必须串行。

1. **【已完成】F1 / O-002 真值与常量对齐**（`39db690`）。交付：`pc/robot_truth.json` + `pc/robot_truth.py`（44 条事实 + `model_artifacts`）、`pc/pack/const_scan.py --check`（46 处命中，43 处冲突全部处置，0 处未处置）、`pc/tests/test_robot_truth.py` 16 项。细节见第 8、9 节和台账 D-008。
2. **【进行中】台账维护**。每轮开工前和收工前都要更新：状态头、完成矩阵、决策记录、parked 清单、“需要你做”、轮次记录；然后 `cp` 到 `PACK_ROOT\LEDGER.md`。
3. **F1 / O-001 接管审计**：读清点结果 → 抽至少 3 个旧报告里的关键数字复现 → 把全仓回归原文放进去 → 写 `PACK_ROOT\BASELINE.md`。见 6.1。
4. **子 agent 的活**：断言提取续做（只读）、独立总审查（最后做）、可选的只读挖掘。完整提示词见第 7 节。
5. **F1 / O-003 封存与划分**：封存集物理隔离；tripwire（训练入口读封存集必须拒绝，`--final-eval` 才放行并写访问日志）；`splits\*.json` 一次生成带哈希；给未来新 take 写预注册划分规则。见 6.2。**做完这一项才允许重训并在封存集上评估。**
6. **F1 / O-004 传感器域契约 v1**：`sensor_domain.json` + 把任意 RGB 图/视频变成机器人画面的转换函数 + 转换函数的测试 + 对抗验证 AUC + 金丝雀指标。见 6.3。
7. **F1 / O-005 证伪三连 K-1 / K-2 / K-3**：定义在契约 §5（我没有把定义抄在这里，先读契约）；每个来源类型给 go/no-go 和分级；写 `FRONTIER1_REPORT.md`。见 6.4。
8. **真机离线分析**：相机俯仰粗估（垂直消失点）+ 标定棋盘格 + 采集脚本 + 标定工具，供用户做 10 秒标定。见 6.5。
9. **F2 第一条纵向切片**：选证伪结果最好的赛道，从数据做到 job 包 → 隔离 Linux 排练 → 回流排练。见 6.6。
10. **F3 Sim2Real**：O-014 声明台账、O-015 数字孪生 v2（用 2026-11-24/25 的 500 Hz 日志辨识，留出验证）、O-016 离线取证报告。见 6.7。
11. **F4 其余赛道数据集 + O-006…O-013 数据线**：含用实测相机重训 GeoNet / 单目深度、修手势标签（AUD-018）。见 6.8。
12. **F5 算法鲁棒性 O-019…O-023**：边缘情况目录 v1、可复现用例库、修复与加固、模糊与性质测试、更聪明的行为。见 6.9。
13. **F6 响应链路 O-024…O-028**：行为级延迟预算表、PC 侧优化前后对比、固件与协议方案（编译通过、默认关闭、不烧录）、软延迟对策、延迟真值测量法。见 6.10。
14. **F7 O-017、O-018、O-029…O-033**：控制策略再训练的云端 job、Sim2Real 上机门禁清单、云端任务包、隔离 Linux 端到端排练（含中断恢复）、回流落地脚本、云端作战计划 + `README_FIRST.md`、T4 真机验收协议。见 6.11。
15. **F8 / O-034 文档、经验笔记、独立总审查**：全新上下文的子代理只凭 `README_FIRST.md` 做 dry-run。见 6.12 和 7.3。

## 6. 各工作点的细节与我的设计想法

“我的想法”都可以改，改了在台账决策记录里写一行理由。每个工作点的验收标准以契约那张表为准，这里是我对它的理解和入口。

### 6.1 O-001 → `PACK_ROOT\BASELINE.md`

契约要求：资产清单含路径、条数、帧数、来源、许可、隐私标记、划分归属；每个基线数字有命令和输出；旧报告里用到的关键数字至少抽 3 个复现；`run_all_tests.py` 结果原样记录；旧文档里的完成声明一律先当待核，复现不了的编号为缺陷。

- 素材：`canon\assets.md` / `assets.json`（资产登记在 `pc/pack/asset_registry.py`：R01–R15 真机与研究数据、M01–M05 自训模型、U01–U03 预训练模型、P01–P04 公开数据）；`findings\raw\run_all_tests_2026-11-25_post_O002.log`；台账里的 AUD-001…018；`findings\PUBLIC_LICENSES_research.md`（许可）。
- 可以复现、又不碰封存集的候选（确切命令去 `SIM2REAL-REPORT-2026-11-23.md`、`RESULTS.md`、`dataset\docs\TRAIN-PIPELINE.md`、`SHOWCASE-REPORT-FOR-ADVISOR.md` 里找；实际跑的命令和输出尾部原样存进 `findings\raw\`）：
  1. 航向扫掠与注视仿真：直接运行 `pc/s2r/sim_plant.py`。
  2. 点击导航收敛：`pc/s2r/eval_click_to_go.py`。
  3. 手势分类器按人留出的宏平均 F1（文档值 0.8051）：`pc/gesture/src/eval_classifier.py`。
  4. 11-07 四条 take 的按-take 留一双敲结果（文档值：帧级 F1 0.802、AUROC 0.892、41/46、均值 F1 0.897）：`pc/train/ensemble_honest.py`、`honest_calibration.py`。
  5. 策略包的 SHA-256 与 TFLite 数值对齐：已经在门禁里。
- **O-003 之前不要重算**这些（它们在封存集上）：6 条 take 的留一结果 69/89、cb8455 上的双敲命中、GeoNet 的留出 absRel。先记成“已报告、未复现：封存集，待 O-003 之后走登记评估”。
- 资产登记里已知的坑：AUD-013（两条 11-15 take 的原始帧不在这台机器的约定位置，只有 MP4）；R11、R12 是 R04、R05 帧的副本；本地 HaGRID 采样集的许可以作者 PDF 为准。

### 6.2 O-003 封存与划分

我已经查到会直接读封存集的入口（请再 grep 一遍 `cb8455`、`empty_d0.5`、`fov_sweep`、`heldout`、`HELDOUT` 补全）：

- `pc/end_side/train_hover_geonet_240.py`：训练末尾直接在两个封存会话上评估（缓存 `test_heldout_80.npz` 在 `pc/bench/_teacher_cache/geonet240_h135_f65/`）。它的内部验证集用的是 `65a54d` 和 `lean_out_d0.5`，不是封存集。
- `pc/end_side/train_dense_splat_student.py`：`real_distill_heldout_v3.npz`。
- `pc/end_side/proto_3d_edl_traversable.py`：`HELDOUT_CB8455`、`EMPTY_SESSION_DIR`。
- `pc/end_side/test_robust_contour_slam.py` 的 `test_5_real_heldout_cb8455_end_to_end`：**每跑一次全仓回归就读一次封存集**。
- `pc/train/double_knock_data.py`、`pc/train/interval_data.py`、`pc/bench/eval_session.py`、`pc/end_side/train_mono_depth.py`（“5 段训练 / 1 段留出”）：留出的是哪一段需要你确认。
- `pc/end_side/test_hover_geonet_240.py` 只读已经存在的报告文件，不重算。

我的设计想法：

- 新建 `pc/pack/sealed.py` 作为唯一入口：`open_sealed(rel_path, purpose)`。只有命令行带 `--final-eval` 才放行，每次放行往 `PACK_ROOT\sealed_access.log` 追加一行（时间、脚本、路径、用途、git 提交号）；其它情况直接抛异常。
- tripwire 测试（加进 `run_all_tests.py`，基线 +1）：每个训练入口，不带 `--final-eval` 读封存集必须被拒绝，而且被拒绝时不写访问日志。
- `splits\*.json`：对现有 6 条 take 和各会话一次生成训练/验证/测试划分，文件里带每个文件的 sha256（哈希缓存在 `canon\file_hashes.json`）。预注册规则：按 take / 按人划分；封存集只进 final-eval；新 take 进来按规则自动落位，不手挑。
- “物理隔离”：把封存集复制到 `PACK_ROOT\sealed\` 并用哈希核对。**移动或删除原件是不可逆操作，先问用户**；在那之前靠 tripwire 挡住。
- 回归测试读封存集这件事要定规矩：改用非封存夹具，或者也走访问日志。定完记进决策记录。

### 6.3 O-004 传感器域契约 v1

- `sensor_domain.json`：把和画面有关的事实与退化参数写成机器可读的契约——240×240、原生 8 位灰度、f≈155 px / 75.5°、相机高 0.135 m、光轴俯仰未标定、盲区 0.174 m、软件 JPEG、采集 13–24 fps、帧周期约 32 ms、夜间长曝光运动模糊、暗角与镜头畸变未标定。
- 转换函数：任意 RGB 图/视频 → 机器人画面。要按机器人的几何来裁（9 cm 的手在 1 m 处是 14 px），不是简单缩放。输入带视场角/焦距元数据时用真实几何，没有时写明假设。
- 测试：直方图、分辨率、几何一致性（已知尺寸的物体落在正确的像素尺寸上）、确定性。
- 对抗验证 AUC：训练一个分类器区分“真机帧”和“改造后的公开帧”，50% 为不可分。真机帧只用非封存数据（11-05、11-07 的 take，`runs_end_side/live_dump_*`），按 take / 按人划分，**封存集不参与**。
- 金丝雀指标：YuNet 人脸召回在真机帧与改造帧上的对比（模型在 `pc/perception/assets/face_detection_yunet_2023mar.onnx`）。
- 现有诊断项 `obsdomain` 是反面教材：参考分布用错了（把 HaGRID 的 512×512 手部特写当成机器人的输入分布），加 `--framing robot` 之后比值反而更差。别重复。

### 6.4 O-005 证伪三连 K-1 / K-2 / K-3

- 定义在契约 §5，先读。原则（契约原话）：“在投入大量采集和生成之前，先用最便宜的实验回答‘从别处搞来的数据到底对真机有没有用’”。契约点到的方法：对抗验证 → O-004；数据卡 → O-013；学习曲线外推 → O-012。
- 已经知道的相关事实：HaGRID 域差太大（手势分类器需要桌面域数据）；真机帧上 YuNet 检出率 98–100%；手部 ROI 链路没有赢过整帧（见 AGENTS.md §6.1）。
- 产出 `PACK_ROOT\findings\FRONTIER1_REPORT.md`：每个来源类型一个 go/no-go 和一个分级，写清实验、数字、结论的置信度。

### 6.5 真机离线分析：相机标定工具

- 现状：`camera.optical_pitch_deg` 是 null（怀疑 ±4°，在 40 cm 处最多造成 10 cm 的落点误差）；垂直视场角只是“假设等于水平视场角”，没测；镜头畸变没标定；水平视场角不确定度约 ±2°。
- 已有的代码：`pc/s2r/controllers.py` 的 `pixel_to_table(u, v, pitch_cam_deg, ...)`；`pc/s2r/live.py` 的环境变量 `S2R_CAM_TILT_DEG`（nan 表示用 IMU 俯仰，未验证）。真机帧在 `runs_end_side/s2r_live/probe_*/`、`runs_end_side/live_dump_*` 和各 take 里。
- 要做：(a) 离线粗估——从真机帧里的垂直结构（门框、显示器边缘、墙角）用垂直消失点估相机俯仰，给出估计值和不确定度，写进真值文件并标清级别；(b) 生成 A4 棋盘格 PDF 到 `pc/s2r/calib/`；(c) 采集脚本，只用 `/capture`，用户举着棋盘时存帧；(d) 标定脚本：OpenCV `calibrateCamera` 得到 fx、fy、cx、cy 和畸变系数，棋盘竖直放置时反解光轴俯仰；(e) 结果写回 `pc/robot_truth.json`，带方法、日期、出处。

### 6.6 F2 第一条纵向切片

- 先确认本机有没有 WSL / Docker（我没有查过）。排练环境优先级：本机 WSL/Docker → Kaggle CPU。
- job 包用到的目录常量在 `pc/pack/paths.py`（JOBS、UPLOAD、DATASETS 等）。
- 含人脸的真机帧不上云，所以 job 包里只能放公开数据或合成数据。
- 回流排练：结果从“云端”落回本机并核对哈希。中断/恢复也要排练。

### 6.7 F3 Sim2Real（O-014、O-015、O-016）

**O-014 `CLAIMS_LEDGER.md`** = 断言提取（`findings\CLAIMS_extracted.md`）+ 每条的验证层级 + 复现状态（对得上 `BASELINE.md` 里的复现）+ 与 T4 实测冲突的标记。已知的冲突：

- “9 cm 的手在 1 m 处 21 px” → 实测焦距下是 14 px（AUD-014）。
- “静止位置偏差约 4.6 cm 是常数” → 随俯仰变，每差 1° 偏 1.92 cm（AUD-002）。
- “实测视场角 65°” → 是模型从画面猜的，实测 75.5°（AUD-015）。
- “单目深度 AbsRel < 0.15” → 只在 60° / 0.115 m 的相机下成立（AUD-016）。
- 手势 `idle` 类（AUD-018）。

**O-015 孪生 v2**。数据：`runs_end_side/s2r_live/session_*/tel.csv`（500 Hz × 34 列）和 `ctrl.jsonl`，分析工具 `pc/s2r/session_analyze.py <会话目录>`。已经确认的结构：

- 固件 LQR 公式（AUD-011，运动数据上线性区 RMS 1.24）。
- 静摩擦约 30 扭矩单位：扭矩没超过它轮子不动。
- 前后摇摆 1.12 Hz；2 cm 的位置阶跃不平移，只激起摇摆，净位移 −4.4 mm（AUD-007）。
- 航向静摩擦超前量约 14°；停止误差 0.44–4.32°。
- 慢速调 `imu_zero` 靠俯仰吸收（AUD-017）。
- roll 是未滤波的加速度计角，静止时约 −5°，舵机起步有单点尖峰（AUD-003）。

现有的航向对象模型在 `pc/s2r/sim_plant.py`（只有航向，没有前后）。做法：先写下哪些会话用来辨识、哪些留出（例如 rest / pulse / head 辨识，ramp / heading 留出），再拟合；交付一张“留出验证预测误差表”（自由响应的时间序列 RMSE、峰值、频率）。配套的“上机前必须通过的孪生门禁”和 O-018 一起做。

**O-016 取证**：写 `findings\FORENSICS_last_acceptance.md`。11-24 早上的失败（人脸偏离画面中心 11–15°、转身冲过头摔倒、−12.6°/s 自旋）当时只有 30 Hz × 6 列的 `motion_log.json`，没有 roll、扭矩、轮速、`stable_pos`，所以自旋的根因无法从数据确定（AUD-005、AUD-012）。之后的固件零漂补丁有效（多次 8–30 s 窗口漂移在 ±0.02°/s 内）。结论只写到证据允许的级别。

### 6.8 F4 其余赛道数据集 + O-006…O-013

- **O-007**：把 `PUBLIC_LICENSES_research.md` 整理成台账文件——12 个数据集加 6 个筛选过的来源，每个写许可、是否需要用户本人同意、能否进混合训练包、能否上云、效用分级。许可禁止混入训练包的有 Something-Something v2、Jester、EgoGesture、H2O、HO-3D。
- **O-009 A1 拳头/手部**：先修 AUD-018——取 `labels` 里不是 `no_gesture` 的那个标签，或 `leading_hand` 对应的框；真负类只取确实只有 `no_gesture` 的行或 HaGRID v2 的 `no_gesture` 类。然后重训对比 fist 召回。覆盖矩阵按 14 px（1 m）/ 7 px（2 m）、桌面高度、近距占满画面、夜间运动模糊来规划。
- **O-011 A3 桌面几何**：渲染器复用 `hover_geonet_240.py` 的 `render_synthetic_desktop_micro_scene`（光线追踪）。
- **按实测相机重训**（台账 D-009，排在 O-003 之后）：
  - `train_hover_geonet_240.py` 现在默认用实测相机；输出到 `runs_end_side/hover_geonet_240_h135_f75.5/`，不会覆盖已发布的 65° 权重；缓存在 `pc/bench/_teacher_cache/geonet240_h135_f75.5/`，要重新生成。
  - 11-22 那次的日志：缓存 03:18–03:47，小模型 04:00 存盘，ViT-B 到 05:52。
  - 训练末尾的封存集评估要走 `--final-eval`。`train_mono_depth.py`、`train_dense_splat_student.py` 同理。
  - 重训后把门禁指向新权重，并在台账里写清哪些旧数字作废。Android 工程里的副本由用户本人同步。
- **O-012**：三条赛道各自的数据量—指标曲线加外推。**O-013**：数据卡、建包流水线、采集套件。**O-006**：真机资产规范库。

### 6.9 F5 算法鲁棒性（O-019…O-023）

- 边缘情况的来源：`SIM2REAL-REPORT-2026-11-23.md`；`.agents/experience/end-side-pursuit-cliff-knock-findings.md`（14 cm 关联门、幻影刷新、FOV 失联自旋、GeoNet 暗像素盲区、部署阈值即最优）；`runs_end_side/scenario_matrix/scenario_matrix.md`（能力缺口清单）；本轮的 AUD-003、007、008、017、018。
- 用例库要可复现：夹具 + 固定种子。
- 模糊与性质测试的对象：围栏 `pc/s2r/guarded_session.py`（已有 18 项回放测试可扩展）、`pc/s2r/controllers.py` 的 `px_to_deg` / `deg_to_px` 互逆、`pixel_to_table` 在地平线以上返回 None。

### 6.10 F6 响应链路（O-024…O-028）

- 延迟现状：传感器端约 33 ms 是地板（帧周期约 32 ms + 软件 JPEG 21.7 ms）；端到端 42.8 ms p50 / 21.4 Hz（2026-11-03 的数）；s2r 干跑 40–56 ms。拆分在 `LATENCY-MAP.md`。
- O-026 固件与协议方案：只 `idf.py build`，默认关闭，不烧录。
- O-028：手机 240 fps 慢动作同时拍刺激和机器人，脚本逐帧定位“刺激帧”与“首次运动帧”；分析脚本先用合成视频测通，下次真机十分钟量完。

### 6.11 F7（O-017、O-018、O-029…O-033）

- O-017：控制策略再训练的云端 job。现有代码 `pc/end_side/train_rl_policy.py`（BC 预热 + PPO），把其中的对象模型换成孪生 v2。
- O-018：上机门禁清单 = 孪生门禁 + 在 `pc/s2r/mock_robot.py` 上各阶段端到端 + 围栏回放测试。
- O-029…O-031：云端任务包、隔离 Linux 端到端排练（含中断恢复）、回流落地脚本。
- O-032：`README_FIRST.md` 是给“另一台只有账号的机器上的另一个 agent”看的唯一入口。
- O-033：T4 真机验收协议，先在 `mock_robot.py` 上排练。

### 6.12 F8 / O-034

- 更新 `AGENTS.md` 里已经不对的事实：§4 硬件事实表的 FOV 60°、§5.4 第 19 条的 21 px、§6.4 的 FOV 与 `dist_k` 矛盾（实测 75.5°）。改的时候别新增那条废弃简称，已有的不动。
- 经验笔记放 `.agents/experience/`，只写验证过的。
- 独立总审查见 7.3。

## 7. 子 agent 的活

### 7.1 给所有子 agent 的共同约束（写进每份提示词）

- 第 2 节的硬规则全部适用。除此之外：不连机器人、不开串口、不碰 `G:\robot_app\android`、不 push、不烧录。
- 只能写指定的那一个输出文件（都在 `PACK_ROOT\findings\` 下），其它地方只读。
- 输出文件要能续做：每做完一部分就在末尾留 `<!-- CONTINUE -->`，全部完成换成 `<!-- END -->`。
- 最终回报不超过 300 字：做了多少、停在哪个 id、哪些没读到、输出文件路径。
- 只读文档的子 agent 可以和你的工作并行；要跑训练/评测的子 agent 必须和你手上的重任务串行。
- 你引用它们的数字之前要自己抽查复核（AUD-018 就是这样核过的）。

### 7.2 断言提取续做（只读，可并行）——可以直接发的提示词

```text
你在接手一份只读审计，上一个 agent 做到一半被停掉了。工作目录 D:\robot_work（双轮自平衡桌面机器人项目，文档多为中文）。

已有文件：G:\robot_work_offload\cloud_train\findings\CLAIMS_extracted.md —— 项目文档里的量化声明表。先看它的开头 20 行、结尾 40 行，确认现在到哪个编号、有没有上一个 agent 没写完的半截行。
- 开头的“读表说明”是规则：列的定义、重复出现的数字用后缀 b/c/d、level 列只记来源自己给出的验证层级（T0 合成仿真 / T1 真机录像回放 / T2 外部分布 / T3 硬件在环 / T4 真机；“离线”按被测数据的来源归类；来源没给就写 not stated 并把限定词放进括号）。严格照做。
- 如果结尾不是一行完整的表格行，先删掉那半截。

任务：按下面的顺序继续提取，每份文档一个新小节（小节号接着现有的往下编）：
D:\robot_work\HANDOFF-2026-11-03-part2.md
D:\robot_work\DATASET-AND-ANNOTATION.md
D:\robot_work\DATASET-PLAN.md
D:\robot_work\SCENARIO-v2.md
D:\robot_work\CONTRACT.md
D:\robot_work\dataset\docs\TRAIN-PIPELINE.md
D:\robot_work\runs_end_side\scenario_matrix\scenario_matrix.md
D:\robot_work\.agents\experience\end-side-pursuit-cliff-knock-findings.md
D:\robot_work\MORNING-CHECKLIST.md
D:\robot_work\README.md
D:\robot_work\TRANSPORT-REDESIGN.md
D:\robot_work\WIFI-AUTOCONNECT-FIX.md
D:\robot_work\HANDOFF-rig-hover.md
D:\robot_work\VIDEO-ANNOTATOR.md
D:\robot_work\COLLAB.md 和 D:\robot_work\REQUIREMENTS-FOR-TEAMMATE.md
已经有小节的文档不要重做。

提取什么：量化的性能 / 测量 / 结果声明（精度、时延、帧率、体积、通过数、距离、误差、数据集规模、硬件时序、测试通过数等）。设计目标和计划也算一行，level 写 “not stated (plan / target)”。
不提取：GOAL-offline-sprint-2026-11-24.md、HANDOFF-offline-sprint-2026-11-24.md（本轮的契约和台账）；RESEARCH-Video*.md、LIT-REVIEW-*.md，以及根目录下板载旧时序模型的 v2 设计文档（文件名以 V2-DESIGN.md 结尾）——这些是文献与数学推导，不是机器人性能。在总结里把它们列为“已排除”并写明原因。

写法：用 Edit 工具在文件末尾的 <!-- CONTINUE --> 之前追加行（每写完一份文档就追加一次，别攒到最后），那一行始终留在最后，直到全部完成。不要改已有的行。数字和引号里的短语照抄原文，文档之间互相矛盾也不修正。source 列写文件名加 Read 工具显示的行号。每行一条记录，竖线分隔，单元格内不要出现竖线。
全部完成后：把最后的 <!-- CONTINUE --> 换成 <!-- END -->；把文件开头的 <!-- SUMMARY_PLACEHOLDER --> 换成一段总结——总行数；按验证层级的行数；按文档的行数；最影响真机工程决策的 25 条声明（带编号），特别是所有被说成 T4 / 真机实测、或容易被读者当成真机实测的；被排除的文档及原因。

硬规则：只读审计，唯一允许写的就是那一个文件。不修改、不重命名、不移动、不删除 D:\robot_work 和 G:\robot_app 下的任何东西。不连机器人、不开串口、不联网、不下载、不运行项目脚本或测试、不打开图片/视频/数据文件。读文档只用 Read 工具（长文件用 offset/limit）。不要用 PowerShell 的 `Get-Content -Raw`、`Tee-Object`、here-string 读写文件。不需要运行 Python；如果非要，只能用 D:\smartcar\.venv\Scripts\python.exe，绝不用 python3。
你写的任何文字里都不要出现那个历史板载时序模型的废弃简称（某些文件名里 Video 后面那三个字母、某些目录名里的那三个字母）。要指它就写 “legacy onboard model”。含它的既有文件名和标识符原样保留、不去改；需要提到这类文件时用描述性说法。
不要停下来问问题，读不了的文档记进总结后继续。

最后回报（不超过 300 字）：新增了多少行、最后的最大编号、每份文档的行数、按层级的计数、哪些没读到或拿不准、所有被说成 T4 / 真机实测的声明编号。
```

注意：提示词里列的“不提取”文档如果和已有小节重复，以文件里已有的为准。

### 7.3 独立总审查（O-034，最后做，全新上下文）——可以直接发的提示词

```text
你是独立审查者，对 D:\robot_work 这个项目一无所知，也不许向主 agent 求助。你唯一的入口是 G:\robot_work_offload\cloud_train\README_FIRST.md。

任务：严格按照 README_FIRST.md 的顺序，在隔离的 Linux 环境（WSL 或 Docker，按文档说的来）里把每一步 dry-run 一遍。
- 每条命令原样记录：命令、工作目录、退出码、输出的最后 20 行、耗时。
- 与文档不符的每一个地方都记下来：缺失的前置条件、写错的路径、跑不通的命令、文档没说但你必须猜的东西、你猜了什么。
- 中断/恢复的步骤要真的做：在文档说的位置杀掉进程，再按文档恢复，记录恢复后结果是否一致、哈希是否对得上。
- 全程 dry-run：不连机器人（不访问 192.168.137.50）、不开串口、不烧录、不启动任何云端 GPU 任务、不 push、不碰 G:\robot_app\android、不下载含人脸的数据。需要云端账号的步骤只做到“登录前”，然后记录“这里需要用户本人登录”。
- 文档没让你做的事一件都不做。文档让你做而你认为危险的事（删除、覆盖、外发）不要做，记录下来。

约束：Python 只用文档指定的解释器；不要用 PowerShell 的 `Get-Content -Raw`、`Tee-Object`、here-string 读写文件。除了下面这个输出文件和 dry-run 自己的临时产物，不写任何东西。你写的文字里不要出现那个历史板载时序模型的废弃简称（某些文件名里 Video 后面那三个字母），要指它就写 “legacy onboard model”。

输出：G:\robot_work_offload\cloud_train\findings\REVIEW_independent_<日期>.md —— 先给结论（能不能照着这份文档从零走通），然后是逐条结果表（步骤、通过/失败/跳过、原因），再是全部偏差的清单（按严重程度排序），最后是你作为新 agent 会卡住的地方和建议改的字句。

最终回报不超过 300 字：通过多少步、失败多少步、最严重的三个问题、输出文件路径。
```

### 7.4 可选的其它子 agent 活

- **边缘情况目录的原始材料**（只读，可并行，服务 O-019）：把 6.9 节列的那些文档里所有的失败/事故/误触发/漏检逐条抽出来（原话、出处行号、当时的验证层级、后来修没修），写成 `findings\EDGE_CASES_raw.md`。做法和提示词结构照 7.2。
- **许可补充核实**（只读联网，可并行，服务 O-007）：`PUBLIC_LICENSES_research.md` §7 列了没能从一手来源核实的许可。只有当某条赛道真的要用那个数据集时才去补，不必全做。
- **不要交给子 agent 的活**：任何接触机器人、串口、固件烧录、Android 工程的事；任何重计算（训练、全量评测）与主 agent 的重任务同时跑。

## 8. 机器人真值与关键数字（都在 `pc/robot_truth.json` 里，这里只是速查）

**相机**

- GC0308，240×240，原生 8 位灰度，软件 JPEG（编码 21.7 ms），帧周期约 32 ms，采集 13–24 fps。
- 水平视场角 75.5°（2.705 px/°，f≈155 px）。T4，不确定度约 ±2°，镜头畸变没标定。垂直视场角只是假设等于水平的。
- 相机高 0.135 m（±0.01）。相机和 IMU 刚性同在头上，只能摇头，不能低头。
- 光轴俯仰没标定（怀疑 ±4°）。盲区：离车 0.174 m 以内的桌面看不见。
- 遥测里的 `yaw` 就是相机的绝对方位，不要再叠加 `q_head`。
- 9 cm 的手：1 m 处约 14 px，2 m 处约 7 px（文档里的 21 / 10 是按旧视场角算的）。

**头舵机**

- 站立时死区半宽约 7°（回零带宽约 13.6°）；命令 ±45° 实际到 +32.7° / −36.9°。
- 20–30° 的阶跃会让轮子前后晃 2–3 cm、机身航向被带偏 2.7–4.5°、单轮扭矩峰值 95–118。

**平衡（固件 LQR）**

- `temp_u = −(1300·x_e + 5·(v − vx) + 25·(pitch − imu_zero) + 0.4·dq)`，`imu_zero = −2°`；限幅：平衡扭矩 ±100、转向扭矩 ±80、|u| > 20 时加 ±10 的摩擦凸起。
- 静止位置偏移 ≈ 每差 1° 俯仰偏 1.92 cm（是稳态结果，不是机器人会沿着的轨迹）。
- 静摩擦约 30 扭矩单位。前后摇摆 1.12 Hz。2 cm 的位置阶跃不平移。
- 站立时 roll ≈ −5°，是未滤波的加速度计角。

**航向**

- manual 模式（`GET /cmd`）没有航向环；要控制机身走 teacher 模式。
- 航向环是纯 P + 大静摩擦：超前量要堆到约 14° 才起转。绝不能把误差积进 `stable_yaw`。
- 运行时用 `/cfg` 把 `yaw_lead_max` 压到 18、`pos_lead_max` 压到 0.05 左右（`link.Robot.drive_cfg()` 会按静止超前量现算）。
- 轮差 0.219°/tick；机身航向 ≈ 1190° × 轮差(m)，符号相反。
- 零漂：固件补丁后 ±0.02°/s（11-24 早上的 −12.6°/s 自旋原因未知）。

**固件与链路**

- 控制环 6 ms（167 Hz）。遥测 TLM1：UDP 9001，约 500 Hz，152 字节 34 个浮点，发给“最后一个发过来数据报的订阅者”——一个会话只能有一个套接字（用 `pc/s2r/tellog.py` 的 `RawTel`）。
- `/cfg` 只在内存，重启复位。开机默认：`fov_h 60`、`yaw_lead_max 35`、`pos_lead_max 0.20`、`vx_max 0.30`、`vyaw_max 90`、`stale_ms 300`、`cmd_timeout_ms 500`、`head_lpf 0.15`。
- 延迟：传感器端约 33 ms 是地板；端到端 42.8 ms p50 / 21.4 Hz（2026-11-03）；s2r 干跑 40–56 ms。
- 轮子直径 0.06 m，1024 tick/圈。

**HTTP 端点**：`/status`、`/capture?q=`、`/cfg`、`/mode?src=teacher|manual`、`/cmd?stop=1&head=`、`/feat`、`/api/data`、`/api/set?i=8&v=`（`imu_zero`）。本机有代理：`curl` 要加 `--noproxy '*'`，`requests` 要 `trust_env=False`。

## 9. 文件地图

**真值与常量**

- `pc/robot_truth.json` / `robot_truth.py`：事实与加载器。加载器拒绝没有出处的事实，也拒绝“没有值却声称有层级”的事实。
- `pc/pack/const_scan.py`：`--check` 扫 `pc/` 和固件里的视场角/相机高度字面量；处置表 `DISPOSITIONS` 在文件里，新冲突或过期处置都会让它退出码 1。
- 三种相机的名字（台账 D-008）：实测 = `robot_truth.*`；已发布权重的训练相机 = `GEONET_CKPT_FOV_H_DEG`（65°，在 `pc/end_side/hd_moge_contour_engine.py`）、`RELEASED_V1_H_CAM_M` / `RELEASED_V1_FOV_H_DEG`（0.115 m / 60°，在 `mono_depth_model.py`）、`STUDENT_V3_*`（0.115 m / 72°，在 `dense_splat_student.py`）。`REAL_FOV_H_DEG` / `REAL_H_CAM_M` 现在就是实测值。
- 16 个 GeoNet 实验脚本（`sim2real_torture_and_bridge_v16.py`、`jarvis_4d_holo_engine_v10.py`、`scenario1_own_models_e2e.py`、`scenario2_own_models_e2e.py` 等）用的是 `GEONET_CKPT_FOV_H_DEG`，因为它们加载的是 65° 的权重。给它们换权重时记得相机要跟着换。

**数据包工具** `pc/pack/`：`paths.py`、`asset_registry.py`、`inventory.py`、`const_scan.py`、`benign.py`。

**Sim2Real 工具** `pc/s2r/`：`link.py`（UDP 遥测、取帧、执行器）、`controllers.py`（航向伺服、注视、转向-直行、桌面投影）、`guarded_session.py`（围栏会话：rest / head / heading / ramp / drive / pulse）、`mock_robot.py`、`readonly_probe.py`、`tellog.py`、`session_analyze.py`、`sim_plant.py`、`live.py`（驱动模式，默认不跑）、`eval_click_to_go.py`。

**端侧大脑** `pc/end_side/`：统一入口 `pc_unified_brain_app.py`（`--source frames|video|live`，带 `--send-cmd` 才真发指令）、`real_robot_bridge.py`、`hd_moge_contour_engine.py`、`hd_dynamic_contour_slam.py`、`hover_geonet_240.py`、`mono_depth_model.py`、`dense_splat_student.py`、`dense_scene_pipeline.py`、各 `train_*.py`。

**证据与日志**

- `runs_end_side/s2r_live/`（gitignore）：`probe_20261124_223608_boot`、`session_20261124_234353_rest`、`session_20261124_234614_head`（围栏误触发，AUD-003）、`session_20261124_235030_head`、`session_20261124_235702_heading`、`session_20261125_000334_pulse`（5 mm，不动）、`session_20261125_000722_pulse`（2 cm，激起摇摆）、`session_20261125_001651_ramp`、`probe_20261125_001858_final`。
- `PACK_ROOT\findings\raw\`：`run_all_tests_baseline_2026-11-24.log`（22/24）、`run_all_tests_hagrid_link_2026-11-24.log`（24/24）、`run_all_tests_2026-11-25_post_O002.log`（25/25）、`inventory_run2_2026-11-25.log`（崩溃）、`inventory_run3_2026-11-25.log`、端侧套件的三次日志、`census_2026-11-24.txt`。

**杂项**

- `pc\gesture\data\raw\hagrid\hagrid-sample-30k-384p` 是指向 `G:\stream_memory_data\gesture_raw\hagrid\` 的目录联接；没有它两项门禁会失败。
- GeoNet 的旧教师缓存目录已改名为 `pc/bench/_teacher_cache/geonet240_h135_f65/`（gitignore，1.6 GB）。
- `runs_end_side/e2e_replay_67f0c1_err.log` 是 2026-11-23 留下的 0 字节文件，不是本轮产生的，不要提交也不要删。
- 跑全量测试时，第一个套件的目录下会临时出现 `_data_*.bin` / `_out_*.bin`，跑完会清掉。

## 10. 踩过的坑

- `python3` 是会挂起的商店占位程序；用绝对路径的解释器。
- Edit 工具要求先用 Read 工具读过那个文件；用 `sed -n` 打印过不算。
- Git Bash 里的 `/tmp`，Windows 的 Python 看不到；f-string 的表达式部分不能有反斜杠；终端把中文显示成乱码只是显示问题，文件是好的。
- 后台命令的工作目录不会保留到下一条命令；`cd` 只在主会话里保留。
- 仓库里 LF 和 CRLF 混用，`git add` 时的 “LF will be replaced by CRLF” 警告是正常的。
- pytest 加 `-x` 会在第一个失败处停下；想看全部失败就不要加。
- 端侧套件依赖 `runs_end_side/` 下的权重和 `G:/robot_app/android/.../assets/models/` 的副本（只读比对），换机器会缺。
- 大范围 `find /d /g` 会把磁盘拖死；用目录修改时间当证据。
- Windows 上 UDP 套接字向已关闭的端口发包会收到 `WSAECONNRESET`，线程里要当非致命错误处理。
- 并发的 torch 进程会伪装成崩溃（WinError 1455 / 1114）：先查并发。
- 子 agent 连续两次撞 5 小时用量上限；给它们的输出文件留续做标记。
- 测试运行期间同时跑重任务，带时延断言的测试可能抖。

## 11. 审计发现一览（细节在台账“已确认的审计发现”）

- AUD-001 静止时的轮扭矩是共模平衡扭矩，不是差动（第一版读错，已更正）。
- AUD-002 “4.6 cm 平衡偏差”是随俯仰变的稳态偏移：每差 1° 偏 1.92 cm。
- AUD-003 站立 roll 是原始加速度计角，舵机起步有单点尖峰；围栏已加 60 ms 去抖。
- AUD-004 固件 `/cfg` 的 `fov_h = 60`，实测 75.5°（已处置：它只喂板载旧策略的 `theta_img`）。
- AUD-005 零漂补丁有效；11-24 早上的 −12.6°/s 自旋原因仍未知。
- AUD-006 全仓回归的真实基线是 24/24（现在是 25/25）。
- AUD-007 ≤ 2–3 cm 的位置超前量不平移，只激起摇摆。
- AUD-008 头舵机 20–30° 阶跃对平衡的扰动比文档大。
- AUD-009 12 次原地转身停止误差 0.44–4.32°，零冲过头。
- AUD-010 视场角第三次复核：2.0–2.9 px/°。
- AUD-011 固件 LQR 公式在运动数据上成立。
- AUD-012 历史真机日志只有 30 Hz × 6 列。
- AUD-013 两条 11-15 take 的原始帧不在这台机器的约定位置。
- AUD-014 “9 cm 的手 21 px”是按旧视场角算的，实测焦距下是 14 px。
- AUD-015 “实测”视场角常数 65° 是猜的；那条 3D 链路在真机上的地面距离偏大约 22%。
- AUD-016 已发布的单目深度 v1 / 高斯溅射 v2 在实测相机下精度大幅下降（AbsRel 0.0455 → 0.2342）。
- AUD-017 慢速调 `imu_zero` 靠俯仰吸收，轮子不动。
- AUD-018 手势数据集的 `idle` 类 3,097 / 3,098 是带手势框的图，fist 类少了 10%。

## 12. 需要用户本人做的事（持续追加，别替用户做）

1. 相机标定（10 秒，可选）：打印 `pc/s2r/calib/` 的棋盘格（还没生成），举到机器人正前方 15–40 cm 处换几个角度停一停。
2. 下次真机会话前清出桌面：机器人四周至少 30 cm 内不放东西。
3. 许可协议：需要用户本人同意的数据集清单在台账“需要你做”第 4 条。其中 Something-Something v2、Jester、EgoGesture、H2O、HO-3D 的许可禁止混入训练包或再分发。
4. 11-15 那两条 take 的原始帧在哪个数据根目录（AUD-013）。
5. 云端登录与验证码。
6. 可选：用手机在机器人视角拍十分钟的握拳和敲击。
7. 重训后的模型要用户本人同步进 Android 工程。

（完）

---

# 2026-11-25 15:40 追加：F1 已关闭，F2 进行中

## F1 完成状态（commits 0fd8d6d → f7c7432 → 57f2772 → 27e35e4 → ea2322b）

- O-001..O-005 全部 COMPLETE，证据在 `G:\robot_work_offload\cloud_train\findings\FRONTIER1_REPORT.md`
- 关键证伪结论：
  - K-1 UX：公开特写拳/掌数据全帧不迁移到真机 10-20px 小拳头（AUC≈0.48 双臂，传感器域契约救不回来——尺度/构图差主导）
  - K-2 U0/no-go：弱敲幅度+帧率合成剧毒（dev 召回 75.4%→36.9%，最难 take 10/19→0/19）
  - K-3 U1-weak：平面角指标在高误差区退化（判决随训练规模翻转！），可信副指标 MAD/暗区比跨规模一致利好合成混合训练；纯真机训越久几何越差=伪 GT 噪声过拟合
  - 对抗 AUC：域转换后 fist 0.935/palm 0.984，离不可分 0.5 很远——域差在内容不在噪声；金丝雀 YuNet 1.0 vs 0.93 任务信息保留
- O-003 封存：250 文件镜像 verify ok、11 入口 tripwire、零访问日志、回归测试已改非封存夹具

## F2 进行中：A3 几何赛道纵向切片（选它因为 K-3 是唯一非死结果）

已落地：
- `pc/end_side/train_mono_depth_pack.py`：覆盖矩阵采样 + dev-only（显式 take 列表，无 glob，零封存接触）+ 自带物理评测 + 数据卡
- `pc/pack/jobrun.py`：云端 job 包格式 + 本地排练 harness，receipt 自动回流 findings/raw
- job 包：`G:\robot_work_offload\cloud_train\jobs\a3_depth_smoke`（PASS）和 `a3_depth_v1`（足量，正在跑 3200 步）

## 下一步（按优先级）

1. 收 a3_depth_v1 足量结果（rehearsals/20261125_153442/），对比冒烟与足量，写回台账
2. GeoNet 实测相机重训 job 包（D-009 已解禁）——注意 train_hover_geonet_240.py 的封存 tripwire，需要 dev-only 化
3. A2 双敲赛道重构为 dev-only（堵 cb8455 泄漏面，D-010）+ 接触级标注（O-010）
4. A1 拳头赛道走"手部检测/裁剪+尺度匹配"改造路线（K-1 已判死全帧迁移）
5. F3 Sim2Real（O-014..016）未动

## 环境事实

- Python：`D:\smartcar\.venv\Scripts\python.exe`
- 重训练一律 `pc/pack/benign.py --` 包一层（BelowNormal）
- 不要并发跑多个 torch 训练（WinError 1455 假崩溃教训）
- 台账：`HANDOFF-offline-sprint-2026-11-24.md`，镜像 `G:\robot_work_offload\cloud_train\LEDGER.md`，两边都要改
