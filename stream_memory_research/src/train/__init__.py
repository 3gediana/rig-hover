"""Training engine (owner: trainer-eval).  DESIGN §3, §4.1, §4.2.

Entry points
------------
    src.train.run_one.run_one(cfg, resume=False, smoke=False) -> dict
        Train + evaluate one run, write runs/<run_id>/{config,metrics,curves,
        train,per_source}.json + checkpoints/{best,last}.pt.  Never raises.
        CLI: python -m src.train.run_one --config <cfg.json> [--smoke] [--resume]

    src.train.factory.build_model(cfg, feature_dim)
        core via src.models.build_core + the ONE shared MultiTaskHeads.
    src.train.data.build_datasets(cfg, smoke) / build_loaders / make_eval_loader
        manifest -> datasets; T_max random-start windows; robustness views.
    src.train.objectives
        task losses (ce/bce/mse) + the five aux objectives (retrieval, crosstalk,
        time_consistency, write_budget, long_horizon); every aux is a strict
        no-op when the core does not expose its tensors.
    src.train.trainer.Trainer
        AMP (fp16 + GradScaler), grad clip, warmup+cosine LR, AdamW, early
        stopping on the val main metric, OOM backoff (batch -> T_max -> AMP),
        NaN guards, full-state checkpoints, resume.
    src.train.state_probe
        reusable probes: dt importance, mem-read norm curve, write-gate
        histogram, state-reset ablation, linear readout probe, crosstalk matrix.
    src.train.make_fixture
        tiny 20-sequence fixtures under runs/_fixture* (dev only).
    src.train.tests_train
        self-tests: python -m src.train.tests_train

Import note: this package deliberately does NOT import submodules in
``__init__`` (it would create an import cycle with src.eval).
"""
