# SE(2) Spatial Memory, 2D Complex Sandbox RL & Adaptive Cockpit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver end-to-end SE(2) ego-centric knock-position spatial memory, an 8-scenario 2D self-balancing complex sandbox simulator with Asymmetric Actor-Critic PPO training + TFLite deployment, and a high-performance adaptive Android App with built-in 20Hz Digital-Twin Sandbox & 2D Spatial Memory Radar.

**Architecture:** Deterministic SE(2) rigid-body odometry updates ego-centric knock coordinates `(x_body, y_body)` across arbitrary body rotations and curved paths, feeding an upgraded zero-centered 39-dim TFLite residual Tanh policy (`EndSidePolicyNet`) trained via BC + Privileged-Critic PPO inside `HoverSandboxEnv`, wrapped by a 2D clearance & cliff-aware `EndSideSafetyShield`. On Android, `SandboxRobotTransport` implements `RobotTransport` so the exact production `ControlLoop` + TFLite pipeline drives an interactive virtual robot when offline, rendered inside a recomposition-isolated adaptive cockpit UI.

**Tech Stack:** Python 3.11, PyTorch, NumPy, FlatBuffers TFLite v3, Matplotlib; Kotlin 1.9.24, Jetpack Compose (Material 3), TensorFlow Lite 2.16.1, JUnit 4.

**Spec:** [`docs/superpowers/specs/2026-11-18-spatial-memory-sandbox-and-ui-overhaul-design.md`](file:///d:/robot_work/docs/superpowers/specs/2026-11-18-spatial-memory-sandbox-and-ui-overhaul-design.md)

## Global Constraints

- Preserve TFLite policy input `[1, 39]` float32 and output `[1, 3]` float32 (`vx`, `vyaw`, `head`), pure builtin TFLite ops (`FULLY_CONNECTED`, `TANH`, `ADD`, `MUL`).
- Zero firmware modifications; preserve HTTP `/status`, `/capture`, `/cmd`, `/emote` wire compatibility.
- Zero `"CfC"` strings in Android sources/resources or advisor-facing report (`SHOWCASE-REPORT-FOR-ADVISOR.md`).
- No `material-icons-extended` dependency in Android; use `Icons.AutoMirrored.Filled.*` / core `Icons.Filled.*` and custom `Canvas` drawings, achieving 0 Kotlin compiler warnings.
- UTF-8 without BOM / PUA / `U+FFFD` across all workspace docs (`_check_encoding.py`).

## Review Focus

- **Off-axis blind turn (`+45°` knock then fist vanishes):** `SpatialMemoryTracker` must rotate `bearing_body` down toward `0°` as `imu.yaw - head.q` increases by `+45°`, rather than spinning forever.
- **Post-knock hand wave (`hand_valid=1, knock_fired=0`) & bystander face (`slot0` occupied):** Neither may overwrite `TargetLock` coordinates nor pull the policy away from `slot1` during `APPROACH`.
- **Mid-flight re-knock (`knock_fired=1` after `0.65s` refractory):** Must cleanly overwrite `(x_body, y_body)` in both Python and Kotlin and redirect smoothly without pitch divergence.
- **Target overshoot longitudinal sign (`x_body <= 0`):** Euclidean `hypot(x_body, y_body)` stays positive if the robot passes slightly beside/beyond the point; `effective_remaining_m` must clamp to `0.0` once `x_body <= 0.0` within arrival corridor so the robot never chases behind itself.
- **Compact portrait phone height (`~720dp`) & 20Hz telemetry updates:** Viewport must retain generous height via `TelemetryRibbon` + tabbed lower deck, and 20Hz `RobotStatus` updates must not recompose static control tabs.

---

### Task 1: Python SE(2) Spatial Memory & 8-Scenario Sandbox Simulator

**Files:**
- Create: `D:/robot_work/pc/end_side/spatial_memory.py`
- Create: `D:/robot_work/pc/end_side/sandbox_arena.py`
- Modify: `D:/robot_work/pc/end_side/end_side_controller.py`
- Modify: `D:/robot_work/pc/end_side/export_tflite_bundle.py`
- Test: `D:/robot_work/pc/end_side/test_spatial_sandbox.py`

**Interfaces:**
- Produces: `SpatialMemoryTracker` (`lock_polar`, `step_odometry`, `state -> SpatialTargetState`), `HoverSandboxEnv` (`SCENARIO_NAMES`, `reset(scenario, seed)`, `step(cmd)`), upgraded `EndSideFeatureBuilder.build(..., clearance_m=1.0)` & `EndSidePolicyNet(use_residual=True)`.

- [ ] **Step 1: Write failing tests in `pc/end_side/test_spatial_sandbox.py`** covering SE(2) rotation invariance, curved arc accuracy, decoy wave immunity, mid-flight re-knock override, longitudinal pass clamp (`x_body <= 0`), 39-dim wide-angle `slot1` encoding, residual policy TFLite export parity (`< 1e-4`), and all 8 sandbox scenarios stepping cleanly.
- [ ] **Step 2: Run unittest to verify RED failure.**
- [ ] **Step 3: Implement `spatial_memory.py`, `sandbox_arena.py`, and update `end_side_controller.py` & `export_tflite_bundle.py`.**
- [ ] **Step 4: Run `test_spatial_sandbox.py` and `test_end_side.py` to verify GREEN.**

### Task 2: Asymmetric Actor-Critic Curriculum RL Training & Benchmark

**Files:**
- Create: `D:/robot_work/pc/end_side/train_spatial_rl.py`
- Modify: `D:/robot_work/SHOWCASE-REPORT-FOR-ADVISOR.md`

**Interfaces:**
- Consumes: `HoverSandboxEnv`, `SpatialMemoryTracker`, `EndSidePolicyNet`, `export_policy_bundle`.
- Produces: Trained `runs_end_side/hover_rl_policy/model.tflite` + `hover-end-side-rl-ppo-v1-bundle.zip`, copied to `G:/robot_app/android/app/src/main/assets/models/hover_rl_policy.{tflite,json}`, benchmark metrics `runs_end_side/hover_rl_policy/spatial_sandbox_eval.json`, and `docs/showcase/figs/fig5_spatial_sandbox_trajectories.png`.

- [ ] **Step 1: Implement `PrivilegedActorCritic` (`39->160->160->96->3` Actor with $39\times 39$ zero-bias folding + `51->192->192->1` Privileged Critic), 2D SE(2) curved expert, and 3-stage BC + PPO curriculum in `train_spatial_rl.py`.**
- [ ] **Step 2: Run full training + 5-controller × 8-scenario evaluation + TFLite bundle export + figure generation.**
- [ ] **Step 3: Verify TFLite CPU max diff `< 1e-4`, zero straight-ahead bias, high scenario completion & zero cliff falls, and update `SHOWCASE-REPORT-FOR-ADVISOR.md`.**

### Task 3: Android SE(2) Spatial Memory, On-Device Sandbox Twin & Performance Engine

**Files:**
- Create: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/strategy/SpatialMemoryTracker.kt`
- Create: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/sim/SandboxSimulator.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ml/PolicyFeatureBuilder.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ml/TfliteDoubleKnockModel.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/strategy/ScenarioV2Engine.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/strategy/EndSideSafetyShield.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/core/{RobotState,RobotController}.kt`
- Test: `G:/robot_app/android/app/src/test/java/com/luwu/hovercontroller/{SpatialMemoryTrackerTest,SandboxSimulatorTest,PolicyFeatureBuilderTest}.kt`

- [ ] **Step 1: Write failing Kotlin JUnit4 tests (`SpatialMemoryTrackerTest`, `SandboxSimulatorTest`, updated `PolicyFeatureBuilderTest`).**
- [ ] **Step 2: Run `:app:testDebugUnitTest` to verify RED failure.**
- [ ] **Step 3: Implement `SpatialMemoryTracker.kt`, `SandboxSimulator.kt` (`SandboxRobotTransport`), zero-allocation `PolicyFeatureBuilder` & ring-buffered `DoubleKnockKinematicsAndGate`, `@Immutable` annotations, and `RobotController` sandbox lifecycle (`startSandbox`, `tapSandboxKnock`, `selectSandboxPreset`).**
- [ ] **Step 4: Run `:app:testDebugUnitTest` to verify GREEN.**

### Task 4: Android Adaptive Cockpit UI, 2D Spatial Radar & Zero-Warning Build

**Files:**
- Create: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ui/components/SpatialMemoryRadar.kt`
- Create: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ui/components/TelemetryRibbon.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ui/screens/{ControlScreen,ConnectPanel,StrategyScreen,ModelManagerScreen,SettingsScreen}.kt`
- Modify: `G:/robot_app/android/app/src/main/java/com/luwu/hovercontroller/ui/components/{Indicators,PreviewSurface}.kt`
- Modify: `G:/robot_app/android/app/src/main/res/values/strings.xml`

- [ ] **Step 1: Implement `TelemetryRibbon.kt` (1-line glass bar + pitch horizon micro-pill + expandable sheet) and `SpatialMemoryRadar.kt` (interactive 2D tabletop, FOV cone, trail, obstacle, pulsing knock pin, touch ripple, scenario preset chips).**
- [ ] **Step 2: Refactor `ControlScreen.kt` into recomposition-isolated leaf composables, `CAM / RADAR / PiP` viewport switcher, 2-tab lower tactical deck (`驾驶与策略` / `云台与社交`), and `BoxWithConstraints` landscape 2-column ground station; add 1-tap Sandbox Twin card to `ConnectPanel.kt`; replace deprecated icons with `Icons.AutoMirrored.Filled.*`.**
- [ ] **Step 3: Run `:app:testDebugUnitTest :app:assembleDebug --no-daemon`, all Python tests (`run_all_tests.py` + new suites), `rg -n -i "cfc"`, and `_check_encoding.py`.**
