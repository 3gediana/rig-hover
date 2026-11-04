"""Phase-id maps for the synthetic tasks (stored in `meta_phase`)."""
from __future__ import annotations

from typing import Dict

# phase id -> human readable name, per task
PHASES: Dict[str, Dict[int, str]] = {
    "recall_single": {0: "pre", 1: "write", 2: "gap", 3: "query", 4: "verify"},
    "recall_multi": {0: "pre", 1: "refresh", 2: "window"},
    "capacity_scan": {0: "present", 1: "retention", 2: "query"},
    "overwrite": {0: "pre", 1: "write_v1", 2: "hide_v1", 3: "write_v2",
                  4: "read_v2", 5: "v1_return", 6: "read_v1"},
    "vanish_reappear": {0: "pre", 1: "write", 2: "absence", 3: "reappear_masked",
                        4: "normal", 5: "final_hidden"},
    "change_detect": {0: "pre", 1: "write", 2: "hold", 3: "change", 4: "post_hold"},
    "event_seq": {0: "pre", 1: "hold", 2: "event"},
    # ---- temporal-pattern family -----------------------------------------
    "pattern_count": {0: "history", 1: "scored"},
    "pattern_order": {0: "history", 1: "scored"},
    "pattern_rate": {0: "history", 1: "scored"},
    "pattern_count_irregular": {0: "history", 1: "scored"},
    "pattern_count_rate_mixed": {0: "history", 1: "scored"},
    "pattern_count_stretch2": {0: "history", 1: "scored"},
    "pattern_count_stretch4": {0: "history", 1: "scored"},
    "pattern_rate_stretch2": {0: "history", 1: "scored"},
    "pattern_rate_stretch4": {0: "history", 1: "scored"},
}

# masked frames of these phases are aggregated by the task-specific probes
PROBE_PHASES: Dict[str, Dict[str, int]] = {
    "overwrite": {"read_v2": 4, "read_v1": 6},
    "vanish_reappear": {"absence": 2, "reappear_masked": 3},
    "change_detect": {"hold": 2, "post_hold": 4},
    "event_seq": {"hold": 1, "event": 2},
    "capacity_scan": {"retention": 1, "query": 2},
}

# main metric per dataset id (Lead's final uniform metric table)
MAIN_METRIC = {
    "recall_single": "cos",
    "recall_multi": "cos",
    "overwrite": "cos",
    "vanish_reappear": "acc",
    "change_detect": "cos",
    "capacity_scan": "cos",
    "event_seq": "f1",
    # temporal-pattern family: causal per-frame classification
    "pattern_count": "acc",
    "pattern_order": "acc",
    "pattern_rate": "acc",
    "pattern_count_irregular": "acc",
    "pattern_count_rate_mixed": "acc",
    "pattern_count_stretch2": "acc",
    "pattern_count_stretch4": "acc",
    "pattern_rate_stretch2": "acc",
    "pattern_rate_stretch4": "acc",
}
