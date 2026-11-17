# RIG-Hover End-Side Strategy Shell & Expressive Mobile/PC App Design

> **Date**: 2026-11-15  
> **Scope**: `G:\robot_app\android` (Kotlin + Jetpack Compose) & `D:\robot_work\pc\end_side` (Python)  
> **Architecture Goal**: Unified declarative **Strategy Shell Architecture** (model slots + signal bus + priority/blend composer + safety shield) shared identically between Android and PC, paired with expressively animated UI motion, hot-swappable on-device TFLite models, two-way robot hardware feedback, and full live takeover controls.

---

## 1. System Topology & Hardware Interconnection

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                   PC Workstation (D:\robot_work)                    │
  │  • Train detectors / DoubleKnockNet / EndSidePolicyNet (PPO+BC)     │
  │  • export_tflite_bundle.py ──▶ FlatBuffers TFL3 (.tflite + .zip)    │
  │  • model_push_server.py    ──▶ LAN HTTP (/models.json, /models/*)   │
  │  • strategy_engine.py      ──▶ Identical StrategySpec Composer       │
  └──────────────────────────┬───────────────────────▲──────────────────┘
                             │ LAN Pull (Path C)     │ Optional Bridge
                             ▼                       │ (:8765 SSE/CMD)
  ┌─────────────────────────────────────────────────────────────────────┐
  │              Android App (G:\robot_app\android)                     │
  │  Hotspot Preset: SSID=qasd, Pass=88888888, WPA2-PSK, 2.4 GHz        │
  │  • SmartDiscovery: All local NIC subnets + /proc/net/arp + .50 fast │
  │  • ModelRegistry: DETECT / POSE / ACTION (DoubleKnock) / POLICY     │
  │  • SignalBus -> StrategyComposer (PRIORITY tree | BLEND mixer)      │
  │  • EndSideSafetyShield + 800ms Watchdog + Live Manual Takeover      │
  └──────────────────────────▲───────────────────────┬──────────────────┘
                             │ HTTP Keep-Alive (:80) │
                             │ GET /status, /capture │
                             │ GET /cmd?vx&vyaw&head │
                             │ POST /emote?name=...  │
                             ▼                       ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  RIG-Hover ESP32-S3 Robot (167 Hz LQR)              │
  │  • Connects automatically to Android/PC hotspot (qasd / 88888888)   │
  │  • Visual Feedback: GC9A01 Round LCD (/emote) + Head Servo Nod      │
  └─────────────────────────────────────────────────────────────────────┘
```

### 1.1 Hotspot, Multi-Subnet Discovery & Hardware Feedback
1. **Exact Hotspot Credentials**: Prominently surface `SSID: qasd`, `Pass: 88888888`, `WPA2-PSK`, `2.4 GHz`, with 1-tap copy-password & direct Android Portable Hotspot settings deep-link (`ACTION_TETHER_SETTINGS` / wireless settings fallback).
2. **Smart Discovery (`RobotDiscovery.kt`)**:
   - Enumerates all non-loopback site-local IPv4 `NetworkInterface` addresses (`ap0`, `wlan0`, `rndis0`, etc.) to identify actual active hotspot subnets (Android commonly assigns `192.168.43.x`, `192.168.137.x`, `192.168.123.x`, `172.20.10.x`).
   - Reads `/proc/net/arp` to probe connected peers immediately first.
   - Prioritizes common static/DHCP host offsets (`.50`, `.2..20`) before sweeping remaining addresses, and automatically re-syncs if active subnet changes.
3. **Two-Way Robot Hardware Feedback (Zero Firmware Changes)**:
   - On Connect: `POST /emote?name=happy` + optional subtle safe idle head greeting (`head: 0 -> +10 -> -10 -> 0`).
   - On Double-Knock Trigger: `POST /emote?name=surprised` during target lock, shifting to `confident` while approaching and `loving` upon dwell look-back.
   - On Mode / E-Stop Change: `scanning` / `remote_mode` / `sad`.

---

## 2. Unified Declarative Strategy Shell (`StrategySpec` v1)

Both PC (`pc/end_side/strategy_engine.py`) and Android (`strategy/StrategySpec.kt`, `StrategyComposer.kt`) execute the exact same JSON schema:

### 2.1 Four Pluggable Model Slots
| Slot | Task | Input | Output |
|---|---|---|---|
| `detector` | `detect` | RGB image `[1, H, W, 3]` | `[cx, cy, w, h, score, classId]` |
| `pose` | `pose` | RGB image `[1, H, W, 3]` (optional) | Wrist/face keypoints |
| `action` | `action` (`DoubleKnockNet`) | Dual tensor window: visual `[1, 48, 96, 96, 2]` (or single `[48, 96, 96, 2]`) + kinematics `[1, 48, 8]` | `imp_logits [1, 48]`, `count_logits [1, 4]` |
| `policy` | `policy` (`EndSidePolicyNet`) | 39-dim unified feature vector (`[1, 39]`) | `[vx, vyaw, head]` |

### 2.2 Signal Bus (`SignalSnapshot`)
Each control tick merges perception + action + odometry into one normalized signal bus:
- `faceValid`, `thetaFaceDeg`, `faceScale`, `personHeight`
- `handValid`, `fistProb`, `thetaFistDeg`
- `pImpulse`, `pDoubleKnock`, `knockDoubleFired`
- `targetLock` (`valid`, `bearingBodyDeg`, `distM`, `remainingM`)
- `phase`: `IDLE | GAZE_FACE | GAZE_FIST | APPROACH | DWELL`

### 2.3 Composable Behavior Nodes & Mixers
Every node produces `(vx, vyaw, head)`:
- `scenario_v2`: Full social knock interaction (Face -> prioritize Fist -> Double-Knock -> TargetLock approach using wheel odometry even after hand leaves -> Dwell & look back at human face).
- `pd_follow`: Responsive PD tracker (configurable target filter: `any | face | fist`).
- `rl_policy`: Pure 39-dim end-side RL closed-loop policy.
- `gaze_patrol`: Smooth sinusoidal head search sweep (`head = A * sin(w * t)`) while keeping wheels stationary until a target appears.
- `manual_hold`: Zero wheel velocity standby.

Two mixer topologies can be toggled and edited live:
1. **`PRIORITY` (Event-Driven Priority Tree)**: Ordered rule list `(condition, behaviorId, params)`; the first matching rule wins (with hysteresis / ScenarioV2 state preservation). Default `"社交双敲迎宾"` preset is expressed cleanly as a inspectable rule stack.
2. **`BLEND` (Continuous Weighted Mixer)**: Computes `u = Σ(w_i * u_i) / Σ(w_i)` across enabled behaviors, enabling hybrid combinations such as `70% RL Policy + 30% PD Follow`.

### 2.4 Unified End-Side Safety Shield & Direct Takeover
- **Direct Manual Takeover**: Touching the joystick or head yaw slider in **any** autonomous mode smoothly overrides output immediately (with an amber `MANUAL TAKEOVER` badge) and yields cleanly on release.
- **Safety Shield (`EndSideSafetyShield`)**: Enforces zero-overshoot (`vx <= 0` when `remaining_m <= 0` in `APPROACH`), high-pitch speed clamp (`|pitch| > 12°` caps `|vx| <= 15`), and per-step jerk slew limits (`dvx`, `dvyaw`, `dhead`).
- **800 ms Watchdog**: Stale inference (>400 ms) drops `vx/vyaw` to zero while preserving safe head pose.

---

## 3. DoubleKnockNet Pure-TFLite Export & Model Hub

- Extend `pc/end_side/export_tflite_bundle.py` with `build_double_knock_tflite(model)` using strictly core TFLite v3 built-in operators (`CONV_2D`, `PAD`, `MEAN`, `FULLY_CONNECTED`, `LOGISTIC`, `TANH`, `RELU`, `MUL`, `ADD`, `SUB`, `DIV`, `RSQRT`, `SOFTMAX`, `REDUCE_MAX`, `SUM`, `SLICE`, `CONCATENATION`, `RESHAPE`) — zero Flex delegates.
- Fold `BatchNorm2d` / `BatchNorm1d` weights into their preceding convolutions and verify numerical parity against PyTorch (`max_abs_err < 1e-4`).
- Provide `pc/end_side/model_push_server.py` (`GET /models.json`, `/models/<id>.zip`, `/strategies.json`) so Android can pull newly trained models & strategies over the local Wi-Fi hotspot in one tap.

---

## 4. Expressive UI Motion Design & Screens

1. **Motion System (`HoverMotion.kt`)**: Spring-physics tokens (`snappy`, `standard`, `bouncy`, `gentle`), staggered entry modifiers, breathing glows, and spring waveform bars.
2. **Connect Sheet**: Concentric animated `DiscoveryRadar`, instant found-blip animations, 1-click hotspot credential helper (`qasd` / `88888888`), and robot greeting trigger.
3. **Control Console (`ControlScreen`)**:
   - `PreviewSurface`: `240×240` live video + class-coded bounding boxes + TargetLock reticle + morphing `PhaseMorphPill` + real-time Double-Knock impulse meter bar + SafetyShield indicator badge.
   - Quick Strategy Preset Strip + Direct **Head Yaw Servo Control (`-75°..+75°` + center tap)** + **Spring Joystick Pad** (with live takeover indicator) + **Expressive Emote Bar**.
4. **Strategy Composer Screen (`Routes.STRATEGIES`)**:
   - Switch between saved strategy presets or create/import JSON specs.
   - Switch `PRIORITY` vs `BLEND` topology with animated weight sliders, rule priority reordering, live active-node pulsing highlight, and model-slot picker cards (`DETECT` / `ACTION` / `POLICY`).
5. **Model Manager (`ModelManagerScreen`)**:
   - Support `.zip`, raw `.tflite`, and **"Pull from PC LAN"** dialog (auto-guesses gateway/PC IP on `:8790`).
   - Display `ACTION` badge and multi-tensor input/output details cleanly.

---

## 5. Verification Plan
- **PC Python Unit Tests**: `test_double_knock_tflite_export`, `test_strategy_engine.py` (Priority + Blend + ScenarioV2 parity), `test_end_side.py`, `test_double_knock_*.py`, `_check_encoding.py`.
- **Android Unit Tests & APK Build**: JUnit4 tests for `StrategyComposer`, `PolicyFeatureBuilder` (with `TargetLock`), `SafetyShield`, `ModelManifest`, `RobotDiscovery` parsing + Gradle `:app:testDebugUnitTest :app:assembleDebug`.
