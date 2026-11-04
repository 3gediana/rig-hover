# Pixel-to-Event Relational Stream Network

> Status: architecture specification, not an implemented or validated model.
> Date: 2026-11-04.
> Purpose: replace feature-only temporal-memory research as the main design target with causal, end-to-end visual state and event understanding for RIG-Hover.

## 1. Design objective

The model must take raw video frames as its visual input and learn both where to look and how to interpret changes over time. Its main outputs are not generic memory vectors: they are persistent entities, relations between entities, action phases, completed event pulses, event counts, person presence/return state, and an interaction-readiness estimate.

The target is useful robot behavior under variable action speed, long pauses, missing frames, brief occlusion, and changing backgrounds. Periodicity is optional evidence, not an assumption. A tap is counted when its learned event process completes; a wave count is one completed left-right-left cycle. The count convention is explicit so labels and outputs agree.

This is a proposed architecture. It is not yet claimed to outperform existing video models.

## 2. Input and causal contract

At arrival time i:

\[
X_i=(I_i,\tau_i),\quad \Delta_i=\tau_i-\tau_{i-1}>0.
\]

`I_i` is the camera frame, initially 240x240 or the highest stable camera mode. `tau_i` is capture presentation time in seconds, not inference completion time. A valid implementation must not use future frames, clip-wide normalization, or any feature extracted from future frames.

At each step the model emits:

- person/entity presence, location, identity association, and visibility confidence;
- per-action phase probabilities and start/completion probabilities;
- cumulative counts for configured event classes;
- person state (present, leaving, absent, returning, returned);
- interaction-readiness probability and calibrated confidence;
- compact robot-facing action/state output.

The state passed from one call to the next is explicit and bounded in size. It is reset per recording/session and carried continuously at inference.

## 3. Architecture overview

```text
Raw frame I_i + capture timestamp tau_i
                 |
                 v
+---------------------------------------+
| A. Task-conditioned visual eye        |
|                                       |
| Global scout: coarse whole-frame view |
| Focus controller: proposes K regions  |
| Foveal encoder: revisits regions at   |
| original resolution                  |
| Motion evidence: current-vs-prior     |
| local feature change                  |
+-------------------+-------------------+
                    | entity/local tokens + global token
                    v
+---------------------------------------+
| B. Relational scene and event engine  |
|                                       |
| Persistent entity binding             |
| Entity-relation graph                 |
| Phase-aware event processes           |
| Presence/return and readiness heads   |
| Completion pulses -> cumulative count |
+-------------------+-------------------+
                    |
                    v
       event/state/count/confidence outputs
                    |
                    v
       robot safety shell and controller
```

The visual eye and relational/event engine are one trainable model. Task losses must backpropagate through the eye, focus selection, entity binding, and event engine. Cached CLIP embeddings are not accepted as the final model's required input.

### 3.1 Concrete per-frame computation

The order is fixed to avoid a circular dependency between attention and event state:

1. Propagate the previous entity/event state using elapsed time and visibility uncertainty; do not globally decay all semantic content.
2. Run the current RGB frame through the global scout.
3. Choose regions using the current scout map, propagated entities, and previous event-phase queries.
4. Encode the selected RGB crops and produce local tokens.
5. Associate local tokens to persistent entity slots; initialize slots for unmatched salient regions.
6. Build and update entity-relation tokens.
7. Update event phases from current relations plus prior event state; emit boundaries/counts/readiness.
8. Return the new bounded state for the next frame.

Initial tensor budget for a small edge-host model at 240x240:

```text
scout input                    [B, 3, 160, 160]
stride-4 scout feature map     [B, 40, 40, 64]
global scene token             [B, 128]
selected RGB crops             [B, 4, 3, 80, 80]
shared crop encoder tokens     [B, 4, 128]
entity slots                   [B, 4, 128]
relation tokens                [B, 4, 4, 64]
event states                   [B, E, 128]   # E configured event processes
```

These are starting shapes for implementation, not measured optima. The compact persistent state is O(Nd + N^2 d_r + E d), independent of video length. The event count is a scalar/counter per configured event and is not stored as an unbounded frame history.

## 4. Part A: task-conditioned visual eye

### 4.1 Global scout

A compact, trainable convolutional backbone processes the entire frame at low resolution (initial proposal: 160x160). It produces a feature pyramid that preserves a stride-4 map for small hands and contact regions, plus a global scene token. The first implementation should use a mobile depthwise-separable backbone initialized from image-pretraining weights; all stages are fine-tunable. Pretraining weights initialize parameters only; runtime input remains pixels.

The scout must preserve global coverage. Local crops alone are not allowed to hide a person entering the scene or an unexpected new event.

### 4.2 Focus controller and foveal reads

A small focus controller receives:

- the global scout token and spatial feature map;
- previous entity slots and their uncertainty/visibility;
- current event-phase queries (e.g. monitoring a hand-table interaction);
- a persistent novelty query that searches for new entities even when no existing slot requests attention.

It predicts a small set of `K` boxes and scales. Initial proposal: K=4 regions, each represented by a 64x64 or 80x80 crop sampled from the original frame. A shared-weight foveal encoder processes the crops. The controller is not a YOLO preprocessing stage: region choice is internal, learned, and receives task gradients.

Training uses soft spatial sampling or a straight-through differentiable top-K approximation so task loss can train region selection. Inference uses deterministic top-K crop selection. If the focus controller is uncertain, it can allocate a crop to the highest-novelty region rather than only following existing slots.

Each visual token contains:

`appearance, center_xy, scale, source_scale, motion_delta, visibility, quality, focus_reason`.

`motion_delta` is a learned feature difference between current and previously associated local features, optionally supplemented by a low-cost local correlation/flow estimate. Full-frame optical flow is not mandatory in the first edge profile.

### 4.3 Fine-detail constraint

Foveal processing can preserve detail present in the input pixels; it cannot reconstruct detail absent from the sensor. If a hand occupies only about 10 pixels, resizing its crop to 80 pixels does not create finger evidence. The model must output calibrated uncertainty and may abstain from a confident gesture/event decision when image evidence is insufficient.

## 5. Part B: relational scene and event engine

### 5.1 Persistent entity binding

Maintain a fixed number N of entity slots (initial proposal N=4) with bounded state:

`slot = {appearance, position, velocity, visibility, identity_confidence, age, role_distribution}`.

Current visual tokens compete to update existing slots through content-and-position association. For token `u_m` and prior slot `s_j`, form a compatibility score

\[
\ell_{mj}=q(u_m)^\top k(s_j)/\sqrt d + b_{pos}(p_m-\hat p_j)+b_{vis}(s_j).
\]

A dustbin/new-entity column lets unmatched tokens avoid corrupting an existing identity. Normalize scores across candidate slots, then aggregate token values per slot. Each slot updates with a learned gated residual:

\[
\tilde s_j=\mathrm{LN}(s_j+\mathrm{CrossAttn}(s_j,U_j)),\qquad
s_j^+=s_j+g_j\odot\mathrm{MLP}([\tilde s_j,\mathrm{context}_j])-g_j\odot s_j.
\]

The slot state contains appearance, position, velocity, visibility, identity confidence, age, and a learned role embedding (not a forced class label). Unassigned salient tokens can initialize a free slot. During brief occlusion the content/identity embedding is retained while visibility confidence falls and the predicted position uncertainty rises; missing frames do not erase object identity. Reappearance is resolved by matching new visual evidence against identity, location prediction, and scene context.

The global scene token remains separate from entity slots so camera/background changes do not overwrite object identity.

### 5.2 Relation graph

At each time step construct directed pairwise relation tokens between entities and scene anchors. Initial relation vocabulary is learned, not hard-coded to the final task:

- relative position and motion;
- approaching / separating;
- hand near or contacting a surface;
- person orientation/attention relative to the robot;
- entity entering or leaving the visible scene.

Use two lightweight rounds of sparse graph attention/message passing. The graph's purpose is to bind who did what to which object/surface. It prevents a sequence model from receiving unrelated per-frame features with no explicit entity correspondence.

### 5.3 Phase-aware event processes

For each configured event family, maintain a compact event-state token with:

`event identity, current phase distribution, phase entry time, elapsed phase duration, accumulated count, confidence`.

The event engine consumes current entity and relation tokens plus prior event states. It predicts phase transitions and boundary probabilities. Durations are variable; elapsed time enters explicitly as `Delta_i` and time-since-phase-entry. The model does not assume a fixed cycle period.

A lightweight semi-Markov transition head is the initial candidate. Let `e_i` be the fused scene/event input and `a_i` the persistent event state. First update a task-specific content state with a gated residual:

\[
\bar a_i=\mathrm{GRUCell}(e_i,a_{i-1}),
\]

then score allowed phase transitions from the previous phase distribution and `\bar a_i`. For a phase with predicted transition hazard `lambda_i >= 0`, the probability of transitioning during the observed interval is parameterized as:

\[
p_i(\text{transition})=1-\exp(-\lambda_i\Delta_i).
\]

The phase transition is constrained to the event's allowed transition graph (for example, tap-down -> contact -> rebound -> complete), with a learned background/no-transition probability. Completion is a transition probability/event pulse, not a per-frame class. The explicit phase distribution and `last_completed_event_id` make repeated frame evaluation idempotent: a count increments only when a new completion transition is committed.

The hazard is conditioned on visual, relational, and event-state evidence. This is a continuous-time parameterization of transition opportunity, not a claim that all event dynamics are exponential. Compare it against a learned discrete-time phase decoder and retain whichever validates better under irregular sampling.

Examples of phase vocabularies for supervision (not deterministic pixel rules):

- Tap: approach/downstroke -> contact -> release/rebound -> completed event.
- Wave cycle: leftward -> rightward -> leftward completion, with pauses allowed at any phase.
- Person presence: present -> departing -> absent -> re-entering -> returned.

Phase labels are training targets where available; inference remains a neural prediction. A count increments only on the learned completion transition, not on each frame with high event probability. The output includes both a completion pulse and cumulative count so repeated frames cannot silently add duplicates.

### 5.4 Interaction readiness

A dedicated head estimates `p(interaction_ready | current scene graph, recent event states)`. It may use cues such as person presence, orientation, ongoing manual activity, and an explicit gesture, but it must learn their combination from labels rather than encode an unvalidated hand-written rule. The label definition must be operational (e.g. whether a scripted participant intentionally pauses and faces the robot), and uncertainty is exposed. Public action datasets do not provide this construct; a small robot-domain labeled set is ultimately required.

## 6. Learning objectives

Train the entire model from pixel video. A multi-task objective is:

\[
L=\lambda_{phase}L_{phase}
+\lambda_{boundary}L_{boundary}
+\lambda_{count}L_{count}
+\lambda_{entity}L_{entity}
+\lambda_{relation}L_{relation}
+\lambda_{presence}L_{presence}
+\lambda_{ready}L_{ready}
+\lambda_{focus}L_{focus}.
\]

Only include terms supported by labels for a dataset. The main objectives are phase/boundary/count and the robot states of interest. Do not make generic frame prediction or memory retention the primary objective.

- `L_phase`: per-frame phase cross-entropy with ignored/ambiguous intervals masked.
- `L_boundary`: focal or class-balanced loss for onset/completion timestamps.
- `L_count`: count regression/classification plus event-pulse consistency; report exact count accuracy and MAE.
- `L_entity`: identity association and box/center supervision when annotations exist; otherwise omit rather than fabricate.
- `L_relation`: relation/interaction labels when annotated.
- `L_presence`: person present/leave/return state labels.
- `L_ready`: readiness labels on robot-domain sequences only.
- `L_focus`: optional box/crop alignment where regions are annotated, plus task loss through learned sampling. No requirement for manually annotated boxes on every dataset.

Use video-level/session-level split. Augment with variable frame rate, irregular timestamp gaps, dropped bursts, motion blur, camera movement, partial occlusion, background changes, and repeated frames. All transformed variants inherit the original split. Do not let labels or timestamps leak future event boundaries into causal inputs.

Initialize the visual backbone from a compact image-pretrained model, then fine-tune it jointly. No permanently frozen CLIP feature stream in the final architecture. A frozen encoder may be used only as a temporary baseline or initialization/teacher, never as the model's visual input contract.

## 7. Training data plan

### Public data for broad pretraining and benchmarking

- Repetition counting: RepCount, Countix, UCFRep. Use per-repetition intervals where available, not only video-level count labels.
- Temporal action/state recognition: datasets with untrimmed videos and temporal segments (e.g. Charades, FineAction, EPIC-KITCHENS where licensing/access allow).
- Person tracking and re-entry: DAVIS/YouTube-VOS/TAO or MOT-family data with sequence identity annotations.

These datasets support generic action phases, repetitions, person/object continuity, and background variation. They do not guarantee robot-domain “can I interrupt?” labels or small-hand contact labels.

### Robot-domain adaptation

The RIG-Hover task ultimately needs original camera frames for:

- person leaves and returns;
- waving with explicit cycle-count convention;
- hand/table taps with each event boundary and count;
- background activity while no person is present;
- person present but busy vs available to interact;
- distance, illumination, camera yaw, hand size, occlusion, and negative examples.

Record original frames and capture timestamps. Annotate event intervals/states rather than generating a feature vector as the training input. The user does not need to supervise architecture decisions, but high-quality robot-domain labels/data are an unavoidable dependency for claiming this exact scene works well.

## 8. Edge deployment profile

There are two deployment boundaries:

1. **Visual edge host**: runs the pixel-to-event model locally on a companion PC/edge GPU/NPU. It emits compact states, counts, readiness, confidence, and optional target bearing to the robot.
2. **RIG-Hover ESP32-S3**: runs the existing safety shell, actuator arbitration, and balance/control loops. It does not run the full pixel encoder.

The ESP32-S3 has 240 MHz CPU, limited internal SRAM, and no NPU. The requested raw-pixel visual understanding network is not a realistic all-on-MCU target at present. Do not claim otherwise. Keep the model compact and quantization-aware for a local edge host; measure the full end-to-end latency and memory before choosing a deployment claim. The robot still receives learned semantic outputs, but all motor safety constraints remain local and deterministic.

Initial edge target (a budget to validate, not a measured result): <=5M parameters, <=1.5 GMAC per analyzed frame at 224/240 input, bounded K=4 foveal crops, INT8-capable operators. A lower-rate visual analysis path may run independently from the high-rate balance loop. Report latency, peak RAM/VRAM, and dropped-frame behavior on the actual chosen host.

## 9. Evaluation that answers the real question

Primary tasks:

1. Online tap event detection and exact count under variable intervals, pauses, and missed frames.
2. Online wave-cycle detection/count under variable speed and pauses.
3. Person departure and return state recognition with identity continuity.
4. Interaction readiness classification on robot-domain clips.
5. Joint stress test combining background motion, occlusion, camera/head motion, and irregular frame delivery.

For each, report:

- per-frame phase macro-F1;
- event onset/completion precision, recall, F1, and latency;
- exact count accuracy, count MAE, over/under-count rates;
- departure/return precision, recall, delay, and identity-switch rate;
- readiness calibration (Brier score/ECE) and abstention coverage;
- performance by event speed, pause duration, hand pixel size, occlusion length, frame rate, and distance;
- parameter count, compute, latency, memory, energy if measurable.

Compare against framewise MobileNet, causal TCN, GRU/LSTM, streaming transformer/SSM, RepNet/TransRAC-style repetition baselines, and the existing CfC/DSM only where their input contracts make the comparison meaningful. All learned baselines receive the same pixel data and training split for the end-to-end comparison. Also report a separate frozen-feature temporal-core ablation to preserve continuity with the previous study.

## 10. Architecture decisions and open risks

Fixed for the first implementation:

- Pixel input is the model contract.
- Full-scene scout plus learned high-resolution foveal crops; preserve global coverage.
- Persistent entity binding and explicit relations precede event interpretation.
- Actions are represented as variable-duration phase processes; count only completed events.
- Readiness is a learned, labeled task output with confidence.
- Edge deployment means local visual edge host plus MCU safety/control, not full vision on ESP32-S3.

Open and to be settled by targeted experiments:

- K and crop size under tiny-hand conditions;
- hard top-K versus soft/straight-through selection;
- explicit relation graph versus slot self-attention;
- semi-Markov hazard head versus causal segment decoder;
- RGB versus grayscale/mixed input at the current sensor;
- which visual backbone meets edge latency while preserving fine detail;
- whether a local host can satisfy real-time requirements.

No component is retained merely because it sounds novel. Keep it only if ablations show it improves the specified robot tasks at acceptable compute cost.

## 11. Current status

- Previous 447-run study used frozen CLIP embeddings; it evaluated temporal cores and did not train the model to select visual details from pixels.
- Its DSM result on UCF101 order is a narrow temporal-order finding, not evidence of robot scene understanding.
- CfC was weak on several temporal integration tasks; it is not assumed to be the final core.
- This document specifies the new pixel-to-event architecture only. No training, download, or robot experiment has been launched as part of writing this specification.
