# Cached Robot CLIP Feature Probe Audit

## Scope

Read-only analysis of existing `robot_gesture` frozen CLIP features. No downloads, re-encoding, firmware edits, or model training. This is an identifiability audit plus temporal feature statistics, not a supervised accuracy claim.

## Data and feature dimensions

- Encoder: `openai/clip-vit-base-patch16`; input `224px`.
- Loaded `28/28` sequences, `1792` cached frames, `10` independent scripted sessions; manifest source count is `1987` frames (cache shortfall `195`).
- Per-frame global CLS dimension: `768`; patch tokens: `[4, 768]` = `3072` flattened values.
- Mean frame L2 norm: `7.156368785138641`; median per-coordinate std: `0.1136305183172226`; marginal-variance dispersion index: `624.127727999819` (not a covariance rank).
- Adjacent CLS cosine: `0.9390242216161606`; lag-4 cosine: `0.9072818616957969`; lag-1 CLS delta L2: `2.4120617783832228`.
- Adjacent flattened patch-token cosine: `0.9710109246566507`; lag-1 token delta L2: `4.760688227033776`.
- CLS cosine across lags 1/2/4/8/16: `[0.939, 0.919, 0.907, 0.903, 0.89]`; delta L2: `[2.412, 2.813, 3.014, 3.079, 3.268]`.

## Decision-task probe results

No leak-free single-frame/short-window/long-window accuracy is identifiable from the existing robot cache. There is one session per scripted behavior, and labels are constant per session. The cache split's chunks are contiguous pieces of that same recording; treating them as independent train/test examples would measure recording/background/timing memorization.

| Task | Single frame | Short window | Long window | Why no valid score |
|---|---|---|---|---|
| fist | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | One scripted session contains the entire fist class; no independent positive session or frame labels. |
| knock | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | One session labeled 'two knocks' but no per-frame contact/phase labels or per-knock boundaries/count labels; no target coordinate labels. |
| wave | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | One session labeled only 'wave'; no per-frame left/right phase or cycle boundaries/count labels, and no independent wave sessions. |
| lean_out | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | One session labeled 'leave and return'; no per-frame present/leaving/absent/returning labels. |
| target_position_change | not_estimable | not_estimable | not_estimable | No target position/box coordinates or changed-vs-stable segments are annotated in these sessions. |
| empty | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |
| face_close | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |
| none | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |
| open_palm | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |
| pointing | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |
| thumbs_up | not_estimable_leak_free | not_estimable_leak_free | not_estimable_leak_free | Only one independent scripted source/session for this label; the available chunks are from the same recording. |

## Label-free within-session feature change

These are descriptive statistics over the available cached prefix of each session, not task scores. The 0.1 s frame cadence implies lag 1 is about 0.1 s and lag 4 about 0.4 s.

| Session task | Cached frames | CLS lag-1 cosine | CLS lag-4 cosine | CLS lag-1 delta L2 | Patch-token lag-1 cosine |
|---|---:|---:|---:|---:|---:|
| fist | 192 | 0.942 | 0.911 | 2.315 | 0.975 |
| knock | 128 | 0.928 | 0.905 | 2.560 | 0.969 |
| lean_out | 192 | 0.937 | 0.899 | 2.447 | 0.968 |
| none | 192 | 0.954 | 0.927 | 2.061 | 0.981 |
| open_palm | 192 | 0.942 | 0.911 | 2.454 | 0.971 |
| wave | 192 | 0.913 | 0.852 | 2.897 | 0.963 |

## What the temporal statistics do and do not say

Adjacent/lagged cosine and delta norms quantify that embeddings change over time. They do not identify fists, event phases, person-return states, or target coordinates: without corresponding frame/interval labels, high temporal variation may be hand motion, camera/scene changes, or compression noise.

## Reusable prior evidence (different task)

The existing `ucf101_order` study is a valid generic temporal-order check: it pairs forward/reversed versions of each source clip within the same split, so single-frame features are exactly label-ambiguous; the report records framewise MLP at 0.5000 and temporal DSM at 0.9429 (GRU 0.9280). This shows that frozen CLIP sequences retain generic motion/order evidence, but it does not establish fist, knock-phase, leave-return, or target-position information for the robot scene.

## Existing pixel-run audit

Pixel robot runs are not a usable held-out reference despite their fold names. The dataset constructor excludes the supplied `held_out` set, the training script passes the same held label to both `tr` and `te` and sets validation equal to test, and the evaluator again constructs `RobotSessions([held])` (which excludes it). The observed completed robot pixel_event folds f0-f4 report gesture_acc 0.161-0.179 and presence macro-F1 0.820-0.886, but these are same-session training/validation windows; `eval.log` additionally shows `TypeError: eval_robot() ... 4 were given`. Fold f5 has a config but empty train log and no observed metrics/checkpoint. These numbers are not valid held-out performance.

## Label-free reproducible feature-change path

The script `run_cached_probe_audit.py` reads the existing NPZs, concatenates chunks only within their original session, computes CLS and flattened patch-token cosine at lags 1/2/4/8/16, frame-to-frame L2 deltas, within-session centered autocorrelation, per-dimension temporal standard deviations, frame norms, and timestamp cadence. It makes no fitted classifier and uses no task labels to claim semantics. Re-run with `D:\smartcar\.venv\Scripts\python.exe agent_feature_probe\run_cached_probe_audit.py`. This describes whether the features change over time, not what those changes mean.

## Failures / next data needed

- `fist`: record multiple independently started sessions for fist and matched non-fist hands; annotate visibility/gesture per frame. Current clip has one fist session only.
- `knock`: annotate each knock's approach/contact/rebound/onset/completion and desired table target coordinate; the current `knock` session has only a clip-level 'two knocks' label, and only 128 of its 189 source frames are in the feature cache.
- `wave`: annotate left/right phase and each completed cycle boundary/count across multiple independently scripted sessions; current clip has only a clip-level 'wave' label.
- `leave/return`: annotate present/leaving/absent/returning intervals across multiple sessions with different durations and backgrounds; current script label is not a framewise state.
- `target position change`: annotate coordinates and stable/change intervals; no position labels exist.
- Split all variants by recording/session before training; preserve a final held-out session set. A minimum probe should compare framewise linear probe, causal 0.3-0.5 s window probe, and full-prefix causal probe on identical held-out sessions, selecting regularization/window on validation sessions only.

Cache errors: `0`. Full per-sequence/session statistics are in `results.json`.
