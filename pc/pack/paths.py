# -*- coding: utf-8 -*-
"""pc/pack/paths.py - the only place the data-pack locations are defined.

PACK_ROOT (default G:\\robot_work_offload\\cloud_train) holds everything the
cloud training run needs plus its evidence. Override with the environment
variable PACK_ROOT for rehearsals (an isolated Linux tree, a scratch disk).
Nothing in this package writes into the repository except through explicit
arguments.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PACK_ROOT = Path(os.environ.get("PACK_ROOT", r"G:\robot_work_offload\cloud_train"))
FINDINGS = PACK_ROOT / "findings"
RAW = FINDINGS / "raw"            # verbatim command outputs; never edited
CANON = PACK_ROOT / "canon"       # canonical library manifests (sources stay where they are)
SEALED = PACK_ROOT / "sealed"     # held-out data, physically separate, see sealing.py
SPLITS = PACK_ROOT / "splits"
DATASETS = PACK_ROOT / "datasets"
JOBS = PACK_ROOT / "jobs"
UPLOAD = PACK_ROOT / "upload"

# Public data the project already has on disk (never uploaded wholesale: 44 GiB of old research bulk).
PUBLIC_ROOT = Path(r"G:\stream_memory_data")

# Held-out material: never trained on, never used to pick anything (GOAL section 9).
SEALED_TAKE_IDS = ("robot_20261115_151733_cb8455",)
SEALED_SESSION_IDS = ("empty_d0.5_front_night_182201",)
SEALED_DIRS = ("evidence/live/fov_sweep",)


def is_sealed_rel(rel: str) -> bool:
    """Does a repo-relative path belong to held-out material?"""
    r = rel.replace("\\", "/")
    if any(t in r for t in SEALED_TAKE_IDS) or any(s in r for s in SEALED_SESSION_IDS):
        return True
    return any(r.startswith(d) for d in SEALED_DIRS)
