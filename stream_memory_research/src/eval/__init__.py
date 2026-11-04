"""Evaluation engine (owner: trainer-eval).  DESIGN §4, §4.1.

Entry points
------------
    src.eval.evaluator.evaluate(model, loader, task, cfg, splits=("val","test"),
                                robustness=None, light=False, return_seq=False)
        Unified metric dict: task metrics (acc/f1/f1_macro/r2/mae/cos),
        per_prefix, per_gap (age buckets, uses lab_age), per_dt, rollout
        {1,3,5,10}, event onset metrics, crosstalk matrix, aux stats, per-source
        values, params/FLOPs/latency.  Robustness suite (dense/stride2/stride4/
        drop30/drop50/drop70/burst/repeat/jitter/mixed/shuffled) is inference
        only and bit-reproducible per (seq_id, level).
    src.eval.stats
        multi-seed aggregation + paired bootstrap by source_id vs a reference
        model -> comparison rows + markdown table.
    src.eval.failure_cases
        worst-N sequences with a rule-based diagnosis -> failure_cases/<id>.json
        and runs/<id>/failure_cases.json.
    src.eval.predict
        streaming single-frame path (native core.step(), prefix replay fallback).
    src.eval.tests_eval
        self-tests: python -m src.eval.tests_eval

Import note: this package deliberately does NOT import submodules in
``__init__`` (trainer imports evaluator; keeping __init__ light avoids cycles).
"""
