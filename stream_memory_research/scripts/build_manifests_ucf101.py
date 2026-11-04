"""Stage 1 manifest builder for UCF101 causal online action recognition.

Two steps, because ``T`` and the npz paths are only known after extraction:

  python scripts/build_manifests_ucf101.py --plan
      -> data_manifests/ucf101_action_plan.json   (deterministic split + quota)
  python scripts/build_manifests_ucf101.py --finalize
      -> data_manifests/ucf101_action.json        (contract.Manifest)

The plan is a plain JSON list so extraction and manifest building never
disagree about the split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402

SPLIT_ZIP = r"G:\stream_memory_data\videos\ucf101\UCF101TrainTestSplits-RecognitionTask.zip"
VIDEO_ROOT = r"G:\stream_memory_data\videos\ucf101"
FEAT_ROOT = r"G:\stream_memory_data\features\ucf101_action"
DATASET = "ucf101_action"
PLAN_PATH = contract.MANIFESTS / f"{DATASET}_plan.json"
MANIFEST_PATH = contract.MANIFESTS / f"{DATASET}.json"
SOURCE_URLS = [
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP (UCF-101.zip, 6.96 GB)",
    "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP "
    "(UCF101TrainTestSplits-RecognitionTask.zip -- official fold files)",
    "official: https://www.crcv.ucf.edu/data/UCF101/UCF101.rar (not used, cannot unrar)",
]


def do_plan(args) -> None:
    from src.data import ucf101
    owner, classes = ucf101.load_official_split(SPLIT_ZIP, fold=1)
    print(f"[plan] official fold-1: {len(owner)} clips, {len(classes)} classes")
    plan, missing = ucf101.build_plan(owner, classes)
    counts: dict = {}
    for r in plan:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"[plan] selected {len(plan)} clips: {counts}  (missing classes: {missing})")
    payload = {
        "dataset": DATASET,
        "task": "action",
        "fold": 1,
        "classes": list(classes),
        "selection_rule": (
            "official fold-1 train/test lists; per class: test=stride_pick(official "
            "test pool, 12); trainval=stride_pick(official train pool, 50); "
            "val=trainval[9::10] (5); train=trainval minus val (45).  stride_pick "
            "takes idx_i=floor(i*N/n) over lexicographically sorted clip ids."
        ),
        "quotas": {"train_per_class": ucf101.N_TRAIN_PER_CLASS,
                   "val_per_class": ucf101.N_VAL_PER_CLASS,
                   "test_per_class": ucf101.N_TEST_PER_CLASS},
        "counts": counts,
        "classes_missing": missing,
        "records": plan,
    }
    contract.write_json(PLAN_PATH, payload)
    print(f"[plan] wrote {PLAN_PATH}")


def do_finalize(args) -> None:
    from src.data import ucf101
    from src.data.clip_encoder import (ENCODER_TAG, FEAT_DIM, MODEL_NAME,
                                       N_TOK, TOK_DIM)
    plan = contract.read_json(PLAN_PATH)
    prog_path = Path(FEAT_ROOT) / "progress.json"
    prog = contract.read_json(prog_path) if prog_path.exists() else {"done": {}, "skipped": {}}
    done = prog.get("done", {})
    classes = plan["classes"]

    seqs = []
    for r in plan["records"]:
        if r["seq_id"] not in done:
            continue
        info = done[r["seq_id"]]
        seqs.append(contract.SeqRecord(
            seq_id=r["seq_id"],
            source_id=r["source_id"],
            split=r["split"],
            T=int(info["T"]),
            path=info["path"].replace("\\", "/"),
            labels=["action"],
            meta={"class": r["cls_name"], "class_index": r["class_index"],
                  "subject": r["subject"], "clip_tag": r["clip_tag"],
                  "relpath": r["relpath"],
                  "fps": info.get("fps"), "n_frames_total": info.get("n_frames_total"),
                  "n_frames_kept": int(info["T"])},
        ))
    manifest = contract.Manifest(
        dataset=DATASET, task="action", feature_dim=FEAT_DIM, seqs=seqs,
        tok_dim=TOK_DIM, roi_dim=0, n_roi=0,
        encoder={"name": MODEL_NAME, "tag": ENCODER_TAG, "input": 224,
                 "pool": "2x2", "n_tok": N_TOK, "dtype": "float16",
                 "frozen": True, "preprocess": "CLIPImageProcessor",
                 "aggregation": "CLS -> feat; patch 14x14 avgpool 2x2 -> tok"},
        split_key="source_id",
        label_specs={"lab_action": {"kind": "ce", "ignore": -1,
                                    "n_classes": len(classes), "dim": 1}},
        extra={
            "selection_rule": plan["selection_rule"],
            "quotas": plan["quotas"],
            "classes": classes,
            "class_index_map": {c: i for i, c in enumerate(classes)},
            "sampling": "96 uniformly spaced frame indices per clip, endpoints "
                        "inclusive (idx_i=round(i*(N-1)/(T-1))); if the clip has "
                        "fewer than 96 frames all frames are used",
            "timestamp_rule": "t = frame_index / container average_rate (fps per "
                              "sequence stored in meta.fps)",
            "labels": "lab_action[T] int64 = class index, constant over the clip "
                      "(single-label clip classification, causal online eval can "
                      "read the label at every prefix)",
            "source_urls": SOURCE_URLS,
            "source": "HF quchenyuan/UCF101-ZIP (identical bytes to official UCF101.rar)",
            "encoder_config": {"model": MODEL_NAME, "input_size": 224,
                               "resize": "shortest_edge=224 bicubic",
                               "crop": "center 224", "dtype": "float16",
                               "device": "cuda", "batch_size": "128-256",
                               "normalize": "CLIP mean/std"},
            "seed": 0,
            "deterministic": True,
            "rng_used": False,
            "split_unit": "video id (source_id); official fold-1 test set reserved "
                          "for test, val carved from official train",
            "subject_leakage_note": "UCF101 subjects span official train and test; "
                                    "split is by video, not subject. meta.subject "
                                    "stored for optional regrouping.",
            "n_planned": len(plan["records"]),
            "n_extracted": len(seqs),
            "skipped": prog.get("skipped", {}),
            "classes_missing": plan.get("classes_missing", []),
        },
    )
    manifest.check_split_disjoint()
    manifest.save(MANIFEST_PATH)
    print(f"[manifest] {MANIFEST_PATH}  counts={manifest.counts()}  T range="
          f"{min(s.T for s in seqs) if seqs else 0}..{max(s.T for s in seqs) if seqs else 0}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if not args.plan and not args.finalize:
        args.plan = args.finalize = True
    if args.plan:
        do_plan(args)
    if args.finalize:
        do_finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
