"""Stage 3 manifest builder: Charades online event detection.

  python scripts/build_manifests_charades.py            # plan + manifest
  python scripts/build_manifests_charades.py --plan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402
from src.data import charades  # noqa: E402

DATASET = "charades_event"
ANN_DIR = r"G:\stream_memory_data\videos\charades\ann\Charades"
FEAT_ROOT = Path(r"G:\stream_memory_data\features\charades_event")
PLAN_PATH = contract.MANIFESTS / f"{DATASET}_plan.json"
MANIFEST_PATH = contract.MANIFESTS / f"{DATASET}.json"

SOURCE_URLS = [
    "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip "
    "(16,339,546,533 B; 0.60 MB/s single conn -> 5.73 MB/s with 16 range conns)",
    "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades.zip "
    "(3.5 MB annotations: Charades_v1_train.csv, Charades_v1_classes.txt)",
    "HF linxxx3/Charades_v1_480 holds the same 16.3 GB zip (fallback)",
    "HF jinyoungkim/Charades holds the 9848 clips as 1920x1080 individual files "
    "(~32 MB each -> 45 GB for 1400 clips, not used)",
]


def do_plan(_args) -> None:
    rows = charades.load_rows(ANN_DIR)
    plan = charades.build_plan(rows)
    sel = [plan["rows"][v] for v in plan["ids"]]
    stats = charades.event_stats(sel, plan["classes"])
    counts: dict = {}
    for v in plan["ids"]:
        counts[plan["split"][v]] = counts.get(plan["split"][v], 0) + 1
    payload = {
        "dataset": DATASET, "task": "event",
        "classes": plan["classes"], "counts": counts,
        "selection_rule": (
            f"from the {len(rows)} official-train clips that carry at least one "
            f"action segment, pick {len(plan['ids'])} with an even stride over the "
            f"lexicographically sorted video ids (idx_i=floor(i*N/n)); split into "
            f"70/15/15 by contiguous blocks of that sorted selection (Charades ids "
            f"are random 5-char strings so id order is effectively random; the "
            f"split unit is the video id)"),
        "class_rule": (f"the {charades.N_CLASSES} most frequent action classes "
                       f"among the selected clips, each with >= "
                       f"{charades.MIN_SEGMENTS_PER_CLASS} annotated segments"),
        "event_stats": stats,
        "records": [{"seq_id": v, "split": plan["split"][v],
                     "n_segments": len(plan["rows"][v]["_segs"]),
                     "length": plan["rows"][v]["_length"]} for v in plan["ids"]],
    }
    contract.write_json(PLAN_PATH, payload)
    print(f"[plan] {PLAN_PATH} counts={counts} classes={len(plan['classes'])} "
          f"stats={stats}")


def do_finalize(_args) -> None:
    from src.data.clip_encoder import ENCODER_TAG, FEAT_DIM, MODEL_NAME, N_TOK, TOK_DIM
    plan = contract.read_json(PLAN_PATH)
    prog_path = FEAT_ROOT / "progress.json"
    prog = contract.read_json(prog_path) if prog_path.exists() else {"done": {}}
    done = prog.get("done", {})
    split_of = {r["seq_id"]: r["split"] for r in plan["records"]}
    classes = plan["classes"]
    seqs = []
    for vid in sorted(done):
        info = done[vid]
        seqs.append(contract.SeqRecord(
            seq_id=vid, source_id=vid, split=split_of.get(vid, "train"),
            T=int(info["T"]), path=info["path"].replace("\\", "/"),
            labels=["event", "seg"],
            meta={"scene": info.get("scene"), "subject": info.get("subject"),
                  "fps": info.get("fps"),
                  "n_frames_total": info.get("n_frames_total"),
                  "n_frames_kept": int(info["T"]),
                  "n_segments": info.get("n_segments"),
                  "csv_length": info.get("csv_length")}))
    m = contract.Manifest(
        dataset=DATASET, task="event", feature_dim=FEAT_DIM, seqs=seqs,
        tok_dim=TOK_DIM, roi_dim=0, n_roi=0,
        encoder={"name": MODEL_NAME, "tag": ENCODER_TAG, "input": 224,
                 "pool": "2x2", "n_tok": N_TOK, "dtype": "float16",
                 "frozen": True, "preprocess": "CLIPImageProcessor"},
        split_key="source_id",
        label_specs={
            "lab_event": {"kind": "bce", "n_classes": len(classes),
                          "dim": len(classes)},
            "lab_seg": {"kind": "ce", "n_classes": 2, "ignore": -1, "dim": 1},
        },
        extra={
            "selection_rule": plan["selection_rule"],
            "class_rule": plan["class_rule"],
            "classes": classes,
            "class_index_map": {c: i for i, c in enumerate(classes)},
            "class_names": _class_names(classes),
            "sampling": f"{charades.N_FRAMES} uniformly spaced frame indices per "
                        f"clip over the container frame count (endpoints "
                        f"inclusive); shorter clips keep all frames",
            "timestamp_rule": "t = frame_index / container average_rate",
            "labels": {
                "lab_event": "[T,C] float32 multi-label: class c is 1 at frame t "
                             "iff t lies in an annotated [onset, offset) segment "
                             "of class c (real temporal boundaries from the CSV)",
                "lab_seg": "[T] int64 onset indicator: 1 if any annotated segment "
                           "starts in (t_{f-1}, t_f]",
            },
            "event_stats": plan.get("event_stats"),
            "source_urls": SOURCE_URLS,
            "encoder_config": {"model": MODEL_NAME, "input_size": 224,
                               "dtype": "float16", "device": "cuda"},
            "seed": 0, "deterministic": True, "rng_used": False,
            "n_planned": len(plan["records"]),
            "n_extracted": len(seqs),
            "skipped": prog.get("skipped", {}),
        },
    )
    m.check_split_disjoint()
    m.save(MANIFEST_PATH)
    print(f"[manifest] {MANIFEST_PATH} counts={m.counts()}")


def _class_names(classes) -> dict:
    try:
        full = charades.load_classes(ANN_DIR)
        mp = {}
        for line in full:
            bits = line.split(" ", 1)
            if len(bits) == 2:
                mp[bits[0]] = bits[1]
        return {c: mp.get(c, "") for c in classes}
    except Exception:  # noqa: BLE001
        return {c: "" for c in classes}


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
