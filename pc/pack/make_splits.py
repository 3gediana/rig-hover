# -*- coding: utf-8 -*-
"""pc/pack/make_splits.py - generate the pre-registered split manifests once.

Outputs PACK_ROOT/splits/*.json. Files are hashed so later audits can prove
which exact bytes belonged to which side of the split. Generation is copy-free:
nothing is moved or modified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paths
from sealed import sha256_file

DEV_TAKE_IDS = [
    "robot_20261107_204337_5ca36d",
    "robot_20261107_212119_67f0c1",
    "robot_20261107_213314_5c0158",
    "robot_20261107_214950_1f30d1",
    "robot_20261115_150045_65a54d",
]
SEALED_TAKE_IDS = ["robot_20261115_151733_cb8455"]
SEALED_SESSION_IDS = ["empty_d0.5_front_night_182201"]
SEALED_LIVE_DIRS = ["fov_sweep"]


def rel(p: Path) -> str:
    return p.resolve().relative_to(paths.REPO.resolve()).as_posix()


def file_entry(p: Path) -> dict:
    return {"path": rel(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def files_under(root: Path) -> list[Path]:
    return sorted((Path(dp) / f for dp, _dn, fn in __import__("os").walk(root) for f in fn), key=lambda p: p.as_posix())


def group_dir(root: Path, split: str, group_id: str, role: str) -> dict:
    files = [file_entry(p) for p in files_under(root)]
    return {
        "id": group_id,
        "root": rel(root),
        "split": split,
        "role": role,
        "files": files,
        "n_files": len(files),
        "bytes": sum(f["bytes"] for f in files),
    }


def make_take_splits() -> dict:
    rows = []
    base = paths.REPO / "dataset" / "takes"
    for tid in DEV_TAKE_IDS:
        rows.append(group_dir(base / tid, "dev", tid, "leave-one-take-out development pool"))
    for tid in SEALED_TAKE_IDS:
        rows.append(group_dir(base / tid, "sealed", tid, "final evaluation only through pc/pack/sealed.py"))
    return {
        "name": "take_splits",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "split by whole take, never by frame; new annotated takes enter dev unless pre-registered as sealed before any training use",
        "rows": rows,
    }


def make_evidence_splits() -> dict:
    rows = []
    for root in sorted((paths.REPO / "evidence" / "sessions").glob("*/")):
        split = "sealed" if root.name in SEALED_SESSION_IDS else "dev"
        rows.append(group_dir(root, split, root.name, "controlled evidence session"))
    for root in sorted((paths.REPO / "evidence" / "live").glob("*/")):
        split = "sealed" if root.name in SEALED_LIVE_DIRS else "context"
        rows.append(group_dir(root, split, root.name, "early unscripted live evidence; context only unless reviewed"))
    return {
        "name": "evidence_splits",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "split by whole session/directory, never by frame; controlled sessions may enter dev only after label review",
        "rows": rows,
    }


def make_live_splits() -> dict:
    rows = []
    base = paths.REPO / "runs_end_side"
    for root in sorted(base.glob("live_dump_s*/")) + sorted(base.glob("live_verify_v25*/")):
        rows.append(group_dir(root, "unassigned", root.name, "unified-brain live dump; assign by whole session only"))
    return {
        "name": "live_session_splits",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": "unassigned by default; before training use, assign a whole session to dev or sealed in this file and record the decision in the ledger",
        "rows": rows,
    }


def make_rules() -> dict:
    return {
        "name": "preregistered_split_rules",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rules": [
            "split unit is a whole take/session/generator seed group; frames or contiguous clips must never be assigned independently",
            "sealed material is final-evaluation-only and must be opened through pc/pack/sealed.py with --final-eval",
            "a new real-robot take defaults to dev only if it was not used for any design, threshold, model, or prompt decision; otherwise it must be sealed or quarantined",
            "synthetic variants inherit the split of their generator seed group; changing a visual seed creates a new group",
            "public human-subject data is split by subject whenever subject IDs exist",
            "teacher caches inherit the split of their source pixels and must be rebuilt when this manifest changes",
        ],
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify that the outputs exist")
    args = ap.parse_args()
    payloads = [make_take_splits(), make_evidence_splits(), make_live_splits(), make_rules()]
    names = ["take_splits.json", "evidence_splits.json", "live_session_splits.json", "preregistered_rules.json"]
    if args.check:
        missing = [n for n in names if not (paths.SPLITS / n).exists()]
        if missing:
            print("missing", missing)
            return 1
        print("splits present")
        return 0
    for name, payload in zip(names, payloads):
        write_json(paths.SPLITS / name, payload)
        print(name, len(payload.get("rows", payload.get("rules", []))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
