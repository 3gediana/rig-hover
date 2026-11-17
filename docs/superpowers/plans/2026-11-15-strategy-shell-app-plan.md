# Implementation Plan: End-Side Strategy Shell & Expressive Mobile/PC App

1. **TFLite Export & Model Push Server (PC `pc/end_side/`)**
   - Extend `export_tflite_bundle.py`:
     - Keep existing `export_policy_bundle` (39-dim -> 3-dim RL policy).
     - Add `build_double_knock_tflite(model: DoubleKnockNet)` using strict core TFLite v3 ops (folded BatchNorm Conv2d + SiLU + AdaptiveAvgPool MEAN + LayerNorm + causal Conv1d + unrolled 48-step causal GRU + differential impulse stats + softmax attention pooling + count MLP), plus `verify_double_knock_tflite_numerical` & `export_double_knock_bundle`.
     - Add CLI entry point exporting both RL Policy & DoubleKnockNet `.zip` bundles into `D:\robot_work\runs_knock\bundles\` and copying `double_knock_v1.tflite` + `double_knock_v1.json` into `G:\robot_app\android\app\src\main\assets\models\`.
   - Create `pc/end_side/strategy_engine.py`:
     - Implement declarative `StrategySpec` dataclass + JSON parser matching Android 1:1 (`PRIORITY` and `BLEND` mixers, `SignalSnapshot`, `scenario_v2`, `pd_follow`, `rl_policy`, `gaze_patrol`, `manual_hold`, `EndSideSafetyShield`).
   - Create `pc/end_side/model_push_server.py`:
     - Zero-dep LAN HTTP server (`:8790`) exposing `/models.json`, `/models/<file>.zip`, `/strategies.json` so Android can pull newly trained models & strategies directly.
   - Create `pc/end_side/test_strategy_engine.py` covering TFLite DoubleKnock parity (`<1e-4`), Priority/Blend mixer math, ScenarioV2 rule preset equivalence, and ModelPushServer HTTP endpoints.

2. **Android Core Engine, Discovery & Hardware Feedback (`G:\robot_app\android`)**
   - **Networking & Feedback**:
     - `NetworkUtils.kt`: enumerate all site-local IPv4 interfaces (`ap0`, `wlan0`, `rndis0`) + active hotspot subnet detection + `/proc/net/arp` peer lookup.
     - `RobotDiscovery.kt`: scan prioritized target offsets (`.50`, `.2..20`, ARP entries) across local hotspot subnets before `/24` sweep.
     - `RobotTransport.kt` & `HttpRobotTransport.kt`: add `sendEmote(name: String): Boolean` (`POST /emote?name=...`) and `sendHead(headDeg: Float)`.
     - `RobotController.kt`: wire two-way connect feedback (`happy` emote + safe micro head-nod when idle), mode/knock/estop emote feedback, direct manual head control (`manualHeadDeg`), live manual takeover priority, `StrategyComposer` pipeline, and `EndSideSafetyShield`.
   - **ML Models & LAN Pull**:
     - `ModelManifest.kt`: add `ModelTask.ACTION` (`task = "action"`, `window`, `kinDim`, `numClasses`, dual input/output metadata) and clean `Policy` labels (remove `CfC` text).
     - `ModelInspector.kt` & `ModelImporter.kt`: inspect multi-input/multi-output tensors accurately (preserve backward compatibility for single-input models) and load bundled `double_knock_v1` asset automatically alongside `hover_rl_policy`.
     - `TfliteDoubleKnockModel.kt`: stateful 48-frame sliding buffer computing `[gray, |diff|]` (96x96) + 8-dim kinematics + exact cooldown/pair-gap trigger logic matching Python `DoubleKnockDetector`.
     - `HttpModelFetcher.kt`: Path C pull from PC `model_push_server.py` (`/models.json` + `.zip` download -> feed straight to `ModelImporter`).
   - **Unified Strategy Shell**:
     - `PolicyFeatureBuilder.kt`: support `TargetLockState` in slot 1 (`out[33..37]`) matching Python 1:1.
     - `EndSideSafetyShield.kt` & `ScenarioV2StateMachine.kt`: Kotlin 1:1 port of Python `EndSideSafetyShield` and `ScenarioV2` phases (`IDLE`, `GAZE_FACE`, `GAZE_FIST`, `APPROACH`, `DWELL`) with `TargetLock`.
     - `StrategySpec.kt`, `StrategyComposer.kt`, `StrategyPresetStore.kt`: declarative JSON strategy presets, `PRIORITY` and `BLEND` mixers, plus `BridgeServer.kt` `/strategies` endpoints.

3. **Expressive UI Motion & Screens (`ui/`)**
   - `HoverMotion.kt`: spring tokens, animated discovery radar (`DiscoveryRadar`), morphing phase pill (`PhaseMorphPill`), animated knock impulse waveform meter (`KnockImpulseBar`), animated strategy rule glow.
   - `ConnectPanel.kt`: `qasd` / `88888888` 1-tap copy & Android Hotspot settings launch + animated radar discovery.
   - `PreviewSurface.kt`, `Controls.kt`, `ControlScreen.kt`: HUD phase pill, TargetLock reticle, Double-Knock impulse bar, SafetyShield badge, quick strategy switcher, manual **Head Yaw Servo Slider (`-75°..+75°`)** + Center button, Expressive Emote bar, and `MANUAL TAKEOVER` live badge.
   - `StrategyScreen.kt` & `HoverNavHost.kt`: interactive Strategy Composer & Mixer route (`strategies`) + animated priority rule toggle & weight blend sliders + model slot selectors.
   - `ModelManagerScreen.kt`: "Pull from PC LAN" sheet + `ACTION` task badge + scrubbed strings.

4. **Verification**
   - Run all Python unit & offline benchmark tests + `_check_encoding.py`.
   - Configure `G:\robot_app\android\local.properties`, Gradle JDK 21 (`C:\Program Files\Microsoft\jdk-21.0.7.6-hotspot`), add Android JUnit4 unit tests, and run `:app:testDebugUnitTest :app:assembleDebug`.
