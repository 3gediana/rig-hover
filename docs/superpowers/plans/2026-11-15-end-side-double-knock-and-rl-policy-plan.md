# 端侧“敲击两下”时空动作识别与端侧强化学习决策架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在无真机在线与无新增实录 take 的条件下，挖掘多源数据（4 条已标 take 的接触脉冲分解 + 本地真机干扰集 + Charades/UCF-101 跨数据集样本）训练高可靠“敲击两下”动作识别模型（`DoubleKnockNet`），并将机器人决策模型完整迁至端侧（`pc/end_side/`），通过 BC 预热 + PPO 强化学习训练端侧策略并导出可直接装入 Android App 的 `float32` `policy.tflite` 包与导师展示报告。

**Architecture:** 视觉-运动学双流时空网络（`DoubleKnockNet`：96×96 2通道 CNN + 8维距离无关姿态运动学 $\to$ 多尺度 TCN + 因果 GRU $\to$ 接触脉冲头 + 四分类窗口计数头）在端侧负责识别“敲击两下”并锁定目标点；端侧控制环（`EndSideController`）通过 HTTP Keep-Alive（`/status`、`/capture`、`/cmd`，含 800 ms 看门狗守护）驱动机器人，底层 39 维策略网络（`EndSidePolicyNet`）在精确 ZOH 轮式倒立摆仿真器中经 BC + PPO（含域随机化、非对称防过冲安全屏障与动作平滑正则）训练，同时支持 PC 端 PyTorch 运行时与 Android 端 `.tflite` 部署。

**Tech Stack:** Python 3.11 (`G:\ai\python11\python.exe`), PyTorch (CUDA), OpenCV, NumPy, FlatBuffers/TFLite, Matplotlib.

**Spec:** [`docs/superpowers/specs/2026-11-15-end-side-double-knock-and-rl-policy-design.md`](file:///d:/robot_work/docs/superpowers/specs/2026-11-15-end-side-double-knock-and-rl-policy-design.md)

## Global Constraints

- Python 解释器固定使用 `G:\ai\python11\python.exe`。
- 所有源码与 Markdown 文档必须为 UTF-8 无 BOM 编码，完成后必须通过 `python _check_encoding.py` 校验。
- 严格遵守诚实评估纪律（`AGENTS.md` §5 第 45 条）：留出 take 绝不参与阈值选择、平滑参数搜索或合成增强样本生成。
- Android 端侧策略模型输入必须为 `[1, 39]` `float32`，输出必须为 `[1, 3]` `float32`（`vx, vyaw, head`），元数据严格符合 `G:\robot_app\android\model_schema.md`。

## Review Focus

- 无姿态框帧回退：当某帧 `pose_roi` 腕点缺失（`None` 或 `NaN`）时，运动学特征必须平滑插值或回退为 0，绝不能产出 `NaN` 污染 GRU 隐状态。
- 留一折隔离：合成双敲增强函数 `synthesize_double_knock_windows` 只接收当前折的 `train_takes`，若传入留出测试 take 应直接触发断言或被单测捕获。
- 看门狗超时保护：端侧控制器在超过 400 ms 未下发指令或发生推理异常时，必须立即输出零速 `DriveCommand(0, 0, head)` 防止触发固件 800 ms 失控或异常冲出桌面。
- 桌沿零过冲约束：在 `APPROACH` 阶段，当估计剩余距离 $d_{\text{rem}} \le 0$ 时，安全屏障必须将前向速度 `vx` 硬钳位为 $0$。
- TFLite 数值与 SHA-256 一致性：导出的 `.tflite` 模型与 PyTorch 前向在随机 39 维输入上的最大绝对误差必须 $< 10^{-5}$，且 `model.json` 中的 `sha256` 与文件真实哈希完全一致。

---

### Task 1: 多源接触脉冲提取与窗口级“敲击两下”数据集构建（`pc/train/double_knock_data.py`）

**Files:**
- Create: `pc/train/double_knock_data.py`
- Test: `pc/train/test_double_knock_data.py`

**Interfaces:**
- Consumes: `pc/train/interval_data.py` (`load_takes`, `load_intervals`, `load_labels`, `decode_gray`), `runs_knock/pose_roi/<take_id>.npz`
- Produces:
  - `extract_kinematics_and_energy(frames: np.ndarray, boxes: np.ndarray | None, fps: float) -> tuple[np.ndarray, np.ndarray]` — 返回 `(kinematics_Tx8, impact_score_T)`
  - `detect_contact_impulses(impact_score: np.ndarray, intervals: list, fps: float, min_gap_s: float = 0.20) -> list[int]` — 返回在敲击区间内定位到的单次接触帧索引列表
  - `build_window_dataset(takes: list, id_to_name: dict, window: int = 48, stride: int = 6, size: int = 96) -> dict[str, dict]` — 按 `take_id` 返回 `{x_vis, x_kin, y_imp, y_cnt, centers_s}`，其中 `y_cnt` $\in \{0, 1, 2, 3\}$ 对应 `{non_knock, single_knock, double_knock, multi_knock}`
  - `synthesize_training_windows(train_data: dict[str, dict], n_per_class: int = 120, seed: int = 42) -> dict[str, np.ndarray]` — 仅从训练折真实单击脉冲与负样本合成平衡的 `1/2/3` 敲击训练窗口
  - `load_external_stress_clips(max_clips_per_source: int = 12, size: int = 96) -> list[dict]` — 加载 `evidence/sessions`（含 `knock_d0.5_front_night_182129` 与 9 类手势负样本）、`Charades` 桌面交互视频与 `UCF-101` 动作视频

- [ ] **Step 1: Write the failing test in `pc/train/test_double_knock_data.py`**
- [ ] **Step 2: Run test to verify it fails**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\train -p "test_double_knock_data.py" -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'double_knock_data'`)
- [ ] **Step 3: Implement `pc/train/double_knock_data.py` and generate contact verification sheets**
- [ ] **Step 4: Run test to verify it passes**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\train -p "test_double_knock_data.py" -v`
  Expected: PASS

---

### Task 2: 视觉-运动学双流双头“敲击两下”模型训练与留一/跨数据集零误报评测（`pc/train/train_double_knock.py`）

**Files:**
- Create: `pc/train/train_double_knock.py`
- Test: `pc/train/test_double_knock_model.py`

**Interfaces:**
- Consumes: `pc/train/double_knock_data.py`
- Produces:
  - `DoubleKnockNet(nn.Module)`: 输入 `x_vis: (B, T, 2, H, W)` 与 `x_kin: (B, T, 8)`，返回 `(logit_imp: (B, T), logit_cnt: (B, 4))`
  - `DoubleKnockDetector`: 在线/离线流式检测器，融合脉冲头与四分类窗口头 + `MODE_GAZE_FIST` 状态门控 + 冷却期，输出 `{"fired": 0|1, "p_double": float, "p_impulse": float, "knock_count": int}`
  - 产物目录 `runs_knock/double_knock_20261115/`：包含 4 折留一权重、全量部署权重 `double_knock_deploy.pt`、诚实留一评测与跨数据集零误报压力测试结果 `metrics.json` 与 `report.md`

- [ ] **Step 1: Write the failing test in `pc/train/test_double_knock_model.py`**
- [ ] **Step 2: Run test to verify it fails**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\train -p "test_double_knock_model.py" -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'train_double_knock'`)
- [ ] **Step 3: Implement `pc/train/train_double_knock.py` and run 4-fold LOO training + stress evaluation**
- [ ] **Step 4: Run test and training verification**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\train -p "test_double_knock_model.py" -v`
  Expected: PASS

---

### Task 3: 端侧闭环控制架构、强化学习策略训练（BC + PPO）与 Android TFLite 导出（`pc/end_side/`）

**Files:**
- Create: `pc/end_side/__init__.py`
- Create: `pc/end_side/end_side_protocol.py`
- Create: `pc/end_side/end_side_controller.py`
- Create: `pc/end_side/train_rl_policy.py`
- Create: `pc/end_side/export_tflite_bundle.py`
- Test: `pc/end_side/test_end_side.py`

**Interfaces:**
- Consumes: `pc/sim/sim.py`, `pc/perception/scenario_v2.py`, `pc/train/train_double_knock.py`
- Produces:
  - `EndSideFeatureBuilder.build(status_dict, detections, target_lock) -> np.ndarray`（`(39,)` `float32`，与 `PolicyFeatureBuilder.kt` 完全对齐）
  - `EndSidePolicyNet(nn.Module)`：输入 `(B, 39)`，输出 `(B, 3)` $\in [-1, 1]$，缩放为 `vx` ($\pm 100$)、`vyaw` ($\pm 100$)、`head` ($\pm 75^\circ$)
  - `EndSideSafetyShield`: 实施桌沿零过冲硬约束（$d_{\text{rem}} \le 0 \implies v_x \le 0$）、俯仰角过大减速保护（$|\theta_{\text{pitch}}| > 12^\circ$）、指令变化率平滑与 800 ms 看门狗心跳守护
  - `EndSideController`: 统一端侧闭环控制器（支持真机 HTTP `GET /status` + `GET /capture` + `GET /cmd` 与离线仿真器闭环）
  - `export_tflite_bundle`: 生成真 `float32` FlatBuffers `.tflite` 模型、`model.json`（含 SHA-256）及 `.zip` 导入包

- [ ] **Step 1: Write the failing test in `pc/end_side/test_end_side.py`**
- [ ] **Step 2: Run test to verify it fails**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\end_side -p "test_end_side.py" -v`
  Expected: FAIL
- [ ] **Step 3: Implement `pc/end_side/` modules, run BC + PPO RL training & closed-loop evaluation, and export TFLite bundle**
- [ ] **Step 4: Run test to verify it passes**
  Run: `& 'G:\ai\python11\python.exe' -m unittest discover -s D:\robot_work\pc\end_side -p "test_end_side.py" -v`
  Expected: PASS

---

### Task 4: 导师展示报告、对比图表生成与全量门禁回归（`SHOWCASE-REPORT-FOR-ADVISOR.md`）

**Files:**
- Create: `pc/end_side/make_showcase_figures.py`
- Create: `SHOWCASE-REPORT-FOR-ADVISOR.md`
- Modify: `AGENTS.md`（同步最新端侧架构与双敲/RL 进展）

- [ ] **Step 1: Run `pc/end_side/make_showcase_figures.py` to generate publication-quality figures in `docs/showcase/figs/`**
- [ ] **Step 2: Write `SHOWCASE-REPORT-FOR-ADVISOR.md` and update `AGENTS.md`**
- [ ] **Step 3: Run encoding check and full test suite**
  Run: `& 'G:\ai\python11\python.exe' D:\robot_work\_check_encoding.py` and unit tests
  Expected: Exit code 0, all tests passing
