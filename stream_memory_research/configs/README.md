# `configs/` — run configurations (owner: trainer-eval)

`configs/suites/` belongs to the Lead (suite definitions for `scripts/run_suite.py`).
`configs/synth_*.json` belongs to synth-lab. Everything else in this directory is
the single-run config format consumed by `src/train/run_one.py`.

```
python -m src.train.run_one --config configs/<file>.json            # real run
python -m src.train.run_one --config configs/<file>.json --smoke    # 2% data, 2 epochs, runs/<id>__smoke
python -m src.train.run_one --config configs/<file>.json --resume   # continue runs/<id>/checkpoints/last.pt
```

## Schema (frozen by the Lead; unknown keys are recorded, not rejected)

```jsonc
{
  "run_id":  "dataset__model__variant__s0",   // enforced: writes runs/<run_id>/
  "dataset": "ucf101_action",
  "manifest": "data_manifests/ucf101_action.json",  // relative to project root
  "task": "action",                            // action|event|state|track|synth
  "model": {"name": "dsm", "hidden": 192, "cfg": {"mem_dim": 64}},
  "heads": {"action": {"kind": "ce", "dim": 101, "ignore": -1}},
  "loss": {
    "task": {"action": 1.0},                   // per-head task-loss weight
    "aux_num_scale": 0.1,                      // multiplies ALL aux weights at once
    "aux": {"long_horizon": 0.0},              // per-name override (0 disables one term)
    "aux_cfg": {"long_horizon": {"horizon": 5}} // per-loss hyper-parameters
  },
  "train": {
    "epochs": 30, "batch": 64, "lr": 2e-3, "wd": 1e-4, "amp": true,
    "grad_clip": 1.0, "patience": 6, "aug": "medium", "T_max": 96,
    "val_every": 1, "warmup_epochs": 2, "num_workers": 0, "preload": true,
    "preload_max_gb": 6.0, "sched": "cosine", "min_lr_ratio": 0.01,
    "min_batch": 4, "min_T": 8, "device": "auto"
  },
  "eval": {
    "batch": 32,
    "robustness": ["dense","stride2","stride4","drop30","drop50","drop70",
                   "burst","repeat","jitter","mixed","shuffled"],
    "robust_max_seqs": 0, "latency": true, "flops": true,
    "failure_topk": 10, "perf_T": 128
  },
  "main_metric": "acc",   // acc|f1|f1_macro (higher) | r2 (higher) | mae (lower)
  "seed": 0
}
```

Default auxiliary weights (when `loss.aux` is absent) are
`retrieval 0.1, crosstalk 0.05, time_consistency 0.05, write_budget 0.01,
long_horizon 0.1`, each multiplied by `loss.aux_num_scale`. Every term is a
strict no-op when the core does not expose the tensors it needs, so one config
drives the whole model zoo.

## Files here

| file | purpose |
|---|---|
| `exp_fixture_dummy_smoke.json` | pipeline smoke config (internal `dummy` core) |
| `exp_fixture_dummy_full.json` | full-data fixture run (checks the pipeline learns) |
| `exp_fixture_dsm.json` | integration check against the real DSM core |
| `exp_ucf101_action__dsm.json` | template: action recognition, ours |
| `exp_ucf101_action__cfc.json` | template: action recognition, main baseline |
| `exp_mot17_track__dsm.json` | template: tracking / occlusion (`track` task) |
| `exp_charades_event__dsm.json` | template: online event detection (`event` task) |

The Lead's suite files override most of these; the templates exist so a single
run can be launched without hand-writing 40 lines of JSON.
