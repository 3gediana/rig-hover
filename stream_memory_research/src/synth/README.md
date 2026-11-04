# `src/synth` — controlled synthetic memory benchmark

Owner: **synth-lab**. Contract: `DESIGN.md` §1.5 + §1.6 + the Lead's final
label/head layout. This file is the definitive spec sheet for the report.

The benchmark answers one question **falsifiably**: does a temporal core have
memory?  Every observation carries *only what is visible now*; `lab_state`
carries the **true latent state at every step**, including the steps where the
object is hidden.  Latent values are re-randomised per episode, so nothing can
be memorised.  A memoryless model therefore scores exactly 0 (cosine) / chance
(identity) on the scored steps; a perfect memory scores 1.0.  Both bounds are
computed by the shipped reference oracles and stored in `summary.json`.

---

## 1. Sequence format (one npz per sequence)

```
feat       [T, 64]      float16   coarse GLOBAL summary, see §2
tok        [T, 16, 17]  float16   per-object tokens, see §2
t          [T]          float32   physical seconds, strictly increasing, t[0]=0
lab_state  [T, 256]     float32   16 slots x 16 state dims, slot-major flatten
lab_mask   [T, 16]      float32   1.0 iff the slot is ASKED FOR at this step
lab_class  [T]          int64     -1 = ignore; identity id (vanish) / event type
lab_event  [T]          float32   0/1 event indicator
lab_age    [T, 16]      float32   seconds since that slot's last informative obs
lab_gap_s  [T]          float32   the episode's designated hidden gap (s), or -1

meta_phase [T]          int8      phase id (names in manifest.extra.phases)
meta_flags [T]          int8      bit1 idmask, bit2 query, bit4 event, bit8 cross
meta_k     [T]          int8      number of existing object slots
meta_gap_s [T]          float32   duplicate of lab_gap_s (kept for probes)
meta_cross [Nc, 5]      float32   (t_prev, t_write, t_post, writer j, reader i)
meta_mix   [1, 6]       float32   (V, VC, N_CODES, C_EVENT, KMAX, layout version)
meta_dt0   [1]          float32   nominal inter-sample interval Δ0 = 0.1 s
```

**The honesty invariant**

* `lab_mask == 1`  ⇔  the slot's true value is recoverable from the observation
  history **but is not present in the current observation** (hidden object, or an
  object detected on re-entry while it is not yet identifiable).  Every metric is
  computed on `lab_mask == 1` only.
* `lab_state` is the true latent value scaled by *knowability*: it is 0 exactly
  where the slot is unused or its value is unknowable (e.g. a write that was
  dropped by an irregular-sampling perturbation).  Because the mask is part of
  the tensor contract, the drop-frame case cannot silently become an unfair
  "guess the unknowable value" test.
* A visible object's token carries its current value; a hidden object
  contributes **zeros everywhere** (`feat` summary channels and `tok`).

## 2. Observation channels (identical for all 70 datasets)

`feat` (channels 0..15 meaningful, **16..63 are always 0.0 reserved**):

| ch | name | meaning |
|---|---|---|
|0|bias|constant 1.0|
|1|vis_frac|number of currently observed object tokens / 16|
|2|any_hidden|1.0 if ≥1 existing object is currently hidden|
|3|query|1.0 on steps flagged as a recall query|
|4|event_pulse|1.0 on an event step (knock / wave impulse)|
|5|event_type_norm|event type / (C_EVENT-1) ∈ [0,1]; 0 when no event|
|6|elapsed_s_60|elapsed physical time / 60 s|
|7|event_age_s_60|seconds since last event / 60 s|
|8|since_change_s_60|seconds since any visible token value last changed / 60 s|
|9|hidden_asked_frac|number of existing+hidden slots / 16|
|10|mean_vis_norm|mean L2 norm of currently observed token values|
|11|n_existing_frac|number of existing object slots / 16|
|12..15|reserved|0.0|

`tok` (`[T, KMAX=16, 17]`): token `j` = `[value_j(16), vis_j(1)]`.
`value_j` is object `j`'s current 16-dim state **as observed now**; when object
`j` is not observed the whole token is zeroed (so `tok[j][16]` is 1 iff the
object is visible).  On a detected-but-not-identifiable re-entry the identity
dims of the token are zeroed but the analogue dims are present.

> ⚠️ Per-object values live in `tok`, not `feat`.  A core that ignores `tok`
> sees only the global summary and cannot read any object value at all.

**Identity code** inside each 16-dim state: dims `0..7` are a *signed one-hot*
(exactly one of ±1, the rest 0) → `N_CODES = 16` identities; dims `8..15` are
`N(0,1)` analogue state.  `layout.decode_code` maps any 16-dim vector to a code
in `0..15`, so a regression-only model can be scored for identity accuracy.
State entries are all O(1) (code ±1 + N(0,1)).

## 3. Tasks (`manifest.extra.task_id`) and their main metric

| task_id | structure | main metric | memory probe |
|---|---|---|---|
| `recall_single` | visible write → hidden gap G (0.1..3.2 s random) → query → re-verify | cos | cosine on the hidden/query stretch |
| `recall_multi` | K∈{2..8} slots; blocks of *refresh all* → *hide all except a writer* (writer updated mid-window) | cos | per-slot cosine + crosstalk matrix |
| `overwrite` | same slot: v1 written → v2 overwrites → v1 returns → v1's analogue state changes | cos | cos per phase (`read_v2`, `read_v1`) |
| `vanish_reappear` | object leaves for G s, re-enters detected but **not identifiable** for 5 frames | acc (identity) | identity at re-entry steps 1/3/5 |
| `change_detect` | visible write → long stable hidden hold → ONE observed change → hidden hold | cos | hold accuracy + update latency + false-update rate |
| `capacity_scan` | K∈{1,2,4,8,12,16} presented one by one, all hidden, then ask for all | cos | capacity curve + K* (≥0.9×K=1) |
| `event_seq` | sparse knock impulses; target events carry a payload, distractor impulses carry none; bursts = duplicate triggers | f1 | post-event cosine at 1x/2x/4x/8x + distractor non-corruption |

Main metric is declared in `manifest.extra.main_metric` (`cos` / `acc` / `f1`)
and matches `src/synth/phases.py:MAIN_METRIC`.

`lab_class` is used as the identity target for `vanish_reappear` (only on
memory-only steps, so the metric measures memory, not perception) and as the
event type target for `event_seq`.

> `event_seq`'s `f1` is **perception-only**: a memoryless model also scores
> `f1 = 1.0` because the impulse is visible.  The memory claim for that task is
> `post_event_cos` at 1x..8x (perfect 1.0, memoryless 0.0), reported by the
> `event_report` probe and in the summary reference block.

## 3b. Temporal-pattern family (`pattern_*`) — core task candidate

Motivation: "交叉遮挡保持一个值" is solved structurally by an un-decayed GRU
(dsm 0.92 vs gru 0.99).  The real target is **temporal structure**: two knocks
are only distinguishable from one knock if the model integrates a window.
`pattern_*` is a causal per-frame **classification** family where **no single
frame is sufficient**:

| task_id | question | classes | main metric |
|---|---|---|---|
| `pattern_count` | how many target pulses (1/2/3)? | 3 | acc |
| `pattern_order` | A→B or B→A? | 2 | acc |
| `pattern_rate` | same two pulses, fast (0.2 s) or slow (0.6 s)? | 2 | acc |
| `pattern_count_irregular` | count task under lognormal Δt + 30/50/70 % dropout + burst drops + repeat + per-episode stretch x1/2/4 | 3 | acc |
| `pattern_count_rate_mixed` | 1 pulse / 2 fast / 2 slow / 3 pulses | 4 | acc |

**Honesty construction** (the key to the task):

1. The **last target pulse is time-aligned across classes** (`anchor_s`), so at
   the scored steps (`t >= anchor + 0.3 s`) the amplitude envelope, background,
   elapsed clock and all distractor channels have the *same distribution for
   every class*.  A single-frame logistic classifier on the scored frames scores
   chance (measured: 0.342/0.503/0.535/0.236 vs chance .333/.5/.5/.25);
   `ZeroMemoryOracle` scores exactly 1/n_classes.
2. The amplitude envelope **resets at each pulse**, so earlier pulses leave no
   residue that could leak the count/order/interval into one frame.
3. Pulses are visible only while they happen (`feat[4]`, plus type in `feat[5]`),
   so only memory can carry the history to the scored steps.
4. Distractor channels: background amplitude (`feat[9]`), noise-event pulse
   (`feat[11]`, never changes the class), noise-event age (`feat[7]`), and
   per-frame amplitude jitter.  Noise events are *not* filtered near pulses
   (filtering would make the distractor count class-dependent).

Observation layout is byte-identical to the memory tasks (`layout_signature`
unchanged), so the same heads / evaluator work.  Pattern semantics are
documented in `manifest.extra["pattern_channel_layout"]`:

```
feat[4]  target pulse (1.0 at pulse frames)
feat[5]  event type (0.0 = A, 1.0 = B) at pulse frames (pattern_order)
feat[6]  elapsed physical time / 60
feat[7]  seconds since last noise event / 60
feat[9]  background amplitude        feat[10] amplitude envelope
feat[11] noise pulse                 ch 3 = query on scored steps
tok slot 0 = [amp, bg, pulse, noise, type, 0...] / vis = 1.0 always observed
lab_mask  = slot 0, 1.0 on scored (evidence-complete) steps
lab_class = class on scored steps, -1 otherwise        lab_event = pulse
lab_state = slot 0 one-hot class + schedule summary     meta_pattern = (n_classes, kind, stretch, irregular)
```

**Reference points (full suite, test splits)**: perfect memory 1.000 on every
pattern dataset (including 2x/4x stretch); memoryless = 1/n_classes
(0.333 / 0.500 / 0.250); random chance = 1/n_classes.  The temporal-shuffle
control drops the perfect oracle to 0.51 (count), 0.50 (order), 0.59 (rate),
0.47 (mixed) — i.e. the temporal structure, not the frame content, carries the
answer.

## 4. Datasets (70 per suite: 56 memory + 14 pattern)

| group | names | splits | counts (full / tiny) |
|---|---|---|---|
| base | `synth_{recall_single, recall_multi, overwrite, vanish_reappear, change_detect, event_seq}` | train/val/test | 200/20/20 (12/4/4) |
| single gap sweep | `synth_recall_single_gap{0.1,0.2,0.4,0.8,1.6,3.2}s` | test only | 40 (3) |
| multi gap sweep | `synth_recall_multi_gap{...}s` (K=8, hide-all-G-then-ask) | test only | 40 (3) |
| event gap sweep | `synth_event_seq_gap{0.2,0.4,0.8,1.6,3.2,6.4}s` (fixed inter-event quiet gap) | test only | 40 (3) |
| capacity sweep | `synth_capacity_K{1,2,4,8,12,16}` | test only | 40 (3) |
| shuffle control | `synth_<base>_shuffled` | test only | 30 (3) |
| irregular | `synth_irregular_{recall_single,recall_multi}_{logdt,drop30,drop50,drop70,repeat,burst,stretch2,stretch4,stretch8,mixed}` | train/val/test | 60/8/8 (2/1/1) |
| pattern base | `synth_pattern_{count, order, rate, count_irregular, count_rate_mixed}` | train/val/test | 201/21/21 for 3-class, 200/20/20 for 2/4-class (tiny: balanced subset) |
| pattern stretch (eval-only) | `synth_pattern_{count,rate}_stretch{2,4}` | test only | 21 / 20 |
| pattern shuffle (eval-only) | `synth_pattern_<base>_shuffled` | test only | 20 |

* The gap sweep datasets are **eval-only** and generated with the *same RNG
  stream* as their base task, so their latent content is paired across gaps —
  a checkpoint trained on `synth_recall_single` can be pointed at the sweep by
  name (`--eval-on`).
* Each base manifest lists its companions in
  `extra.eval_manifests` (+ `extra.eval_manifest_paths`); gap datasets carry
  `extra.gap_s`, capacity datasets `extra.capacity_k`.
* The **temporal-shuffle control** permutes the observation *rows* in time
  (timestamps unchanged and sorted, every obs/label pair kept intact).  A
  frame-copying model is unaffected; a model that exploits temporal continuity
  collapses (measured: perfect-memory cosine 1.00 → 0.34 on `recall_multi`).
* **Irregular sampling** (DESIGN §1.6 a–e) is applied to the *same*
  `RawEpisode` as the base dataset before the observation tensors are built, so
  physical time is preserved: dropping a frame only makes the next `dt` larger.
  After perturbation the observability mask is recomputed, hence the perfect
  oracle still scores 1.0 and unknowable steps are masked out.
* `manifest.extra.layout_signature` is identical for all 70 datasets (tested),
  as are `feature_dim`, `tok_dim` and `label_specs`; one checkpoint head fits
  every sweep.

## 5. Splits & determinism

* Split by **episode generator seed group**, never by frame:
  `train = episodes [0,800)`, `val = [800,1000)`, `test = [1000,1200)`;
  `source_id = "ep<index>"` is unique per episode.
  The full suite generates the first 200/20/20 episodes of each group
  (config `counts` is the knob for more).
* Episode RNG = `SeedSequence([seed_base, hash64(dataset_rng_name), episode])`;
  `seed_base = 20261103`.  Both are recorded in the manifest
  (`extra.seed_base`, each `seq.meta.seed`; generator metadata in `seq.meta`).
* Bit-deterministic: same config + episode seed → byte-identical arrays
  (unit-tested).  `hash()` is never used for seeding (it is randomised) —
  `src/synth/base.stable_hash_int` is used instead.

## 6. Probes (`src/synth/probes.py`)

All probes are **model-agnostic**: they take per-sequence *read-outs* only.

```python
pred = {"state": [T,256],            # required
        "class": [T] or [T,16],      # optional (ints or logits; else decoded from state)
        "event": [T] or [T,1]}       # optional event probability / logit
```

| probe | returns |
|---|---|
| `masked_cosine` | `cos` (= main metric of the recall tasks), `cos_hidden`, `cos_by_phase` |
| `recall_by_gap` | `by_gap` = cos per hidden-age bucket, in units of Δ0 (1x, 2x, 4x ... 128x), using `lab_age` |
| `rollout` | cos at hidden steps {1,3,5,10,20,40} — the "does it hold" curve |
| `crosstalk_matrix` | `[16,16]` (updated j → read i) relative read-out movement, `crosstalk_diag_mean`, `crosstalk_offdiag_mean`, `crosstalk_isolation` (recall_multi only) |
| `capacity_curve` | cos per K, `capacity_norm`, `capacity_base_k`, `K_star` (largest K ≥ 0.9×base) |
| `update_latency` | `update_latency_s`, `update_latency_samples`, `update_success`, `update_success_by_h` |
| `false_update_rate` | prediction variance / total true variance over a stable hold, `false_update_step_rate`, `hold_cos` |
| `identity_accuracy` | `acc`, `acc_absence`, `acc_reentry`, `acc_reentry_{1,3,5}step` |
| `event_report` | `f1`, precision/recall, `event_mean_delay_s`, `event_dup_rate`, `post_event_cos` at 1x..16x, `post_distractor_cos`, `event_type_acc` |
| `pattern_report` | `acc` (= main metric of the pattern family), `balanced_acc`, `acc_per_class`, `confusion`, `acc_by_age` (0-0.5/0.5-1/1-2/>2 s since the last pulse), `pulse_f1` |
| `probe_all(task, seqs)` | union of all probes that apply to the task |

Oracle reference numbers (full suite, test splits; `reference_probes.json`):
perfect memory scores 1.000 everywhere (memory + pattern, including 2x/4x
stretch); memoryless scores 0.000 cosine, 0.079 identity acc (chance 0.062),
1/n_classes pattern accuracy, crosstalk diagonal 0 (never updates) and
post-event cosine 0.000.

§1.6 crosstalk formula (measured on predictions): for a protocol triple
`(t_prev, t_write, t_post, writer j, reader i)` with reader `i` hidden over the
window, `Delta_i = ||pred_i(t_post) − pred_i(t_prev)|| / ||true_i(t_prev)||`.
Correct: off-diagonal ≈ 0 **and** the diagonal (the writer) is large (which
rules out the degenerate "never update" solution).

For evaluator integration:

```python
from src.synth import probes
rep = probes.evaluate_manifest(manifest_path, preds_by_seq_id, split="test")
# -> {"dataset","task","split","main_metric","n_seqs","metrics":{...,"main":{...}}}
```

### Reference oracles (shipped, used for the sanity bounds in `summary.json`)

* `PerfectMemoryOracle` — streams the observation, writes on every fully
  observed token, holds otherwise → 1.0 on every memory metric.
* `PerfectPatternOracle` — integrates the pulse train for the `pattern_*`
  family (count / order / interval, threshold scaled to the actual sampling
  rate so 2x/4x stretch stays answerable) → 1.0.
* `ZeroMemoryOracle` — outputs the current observation only → cosine 0.0,
  identity accuracy 1/16, pattern accuracy 1/n_classes, crosstalk diagonal 0.
* `RandomOracle` — random read-out → the chance level.

## 7. CLI

```powershell
$py = 'D:\smartcar\.venv\Scripts\python.exe'
cd D:\robot_work\stream_memory_research
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --config configs\synth_v1.json   # full
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --tiny                          # tiny (<=30 s)
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --only synth_pattern_count --overwrite --append
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --verify                        # oracle table + reference_probes.json
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --resummary                     # rebuild summary.* only
& $py -m src.synth.make_all --out G:\stream_memory_data\synth --validate                      # structural check of every npz
& $py -m src.synth.tests_synth                                                               # 21 unit/oracle tests
```

`--only` runs merge into the existing `summary.json` / `index.json` (append
mode), so a new task family can be generated without touching older datasets.

Outputs: `<out>/<dataset>/*.npz`, `<out>/manifests/<dataset>.json`
(`contract.Manifest`), `<out>/summary.json` (per-dataset stats + oracle
reference points), `<out>/summary.md` (the same table as Markdown, ASCII so
PowerShell does not mangle it), `<out>/index.json`, `<out>/config_used.json`,
`<out>/reference_probes.json` (per-dataset perfect / memoryless / chance probe
dicts, written by `--verify`).

While `G:\stream_memory_data` is busy, generation is staged on the local temp
disk and bulk-copied with Robocopy (`--no-stage` / `--staging-dir` to change);
the summary reports `generation_seconds` and `copy_seconds` separately.

Measured (2026-11-03/04, CPU only, suite v1): full suite **5428 sequences /
70 datasets** (4040 memory-task + 1388 pattern-task sequences); pattern family
generation ≈ 8 s, Robocopy copy ≈ 34 s.  Tiny fixture: 70 datasets / ~500
sequences; unit tests 7.7 s.

## 8. Files

| file | content |
|---|---|
| `layout.py` | channel contract, identity codec, label specs, layout signature |
| `base.py` | `RawEpisode`, `Timeline`, observability replay, `raw_to_seq`, npz io, perturbations, temporal shuffle |
| `tasks.py` | the seven memory-family generators |
| `pattern.py` | temporal-pattern family generator (`pattern_*`), channel layout, class decode |
| `phases.py` | phase-id maps, main-metric table |
| `probes.py` | probes + reference oracles + evaluator helpers |
| `config.py` | config schema + dataset plan (base/sweeps/capacity/shuffled/irregular/pattern) |
| `generate.py` | dataset writer, manifests, summary, staging store |
| `make_all.py` | CLI |
| `tests_synth.py` | 21 unit/oracle tests (≈8 s) |
