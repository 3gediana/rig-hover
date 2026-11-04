"""Stage 2 manifest builder: MOT17 tracking / occlusion-memory task.

  python scripts/build_manifests_mot17.py            # plan + final manifest
  python scripts/build_manifests_mot17.py --plan     # only the plan
  python scripts/build_manifests_mot17.py --finalize # only the manifest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402
from src.data import mot17  # noqa: E402

DATASET = "mot17_track"
FEAT_ROOT = Path(r"G:\stream_memory_data\features\mot17_track")
SEQ_ROOT = Path(r"G:\stream_memory_data\videos\mot17\sequences")
PLAN_PATH = contract.MANIFESTS / f"{DATASET}_plan.json"
MANIFEST_PATH = contract.MANIFESTS / f"{DATASET}.json"

SOURCE_URLS = ["https://motchallenge.net/data/MOT17.zip (5.86 GB, direct, no login)",
               "HF mirrors Lekim89/MOT17 and Morrison1025/MOT17 exist but ship "
               "images only (no gt.txt) -> not usable for tracking labels"]


def do_plan(_args) -> None:
    recs = []
    for seq, split in mot17.SEQ_SPLIT.items():
        sd = SEQ_ROOT / seq
        try:
            data = mot17.assemble_sequence(sd)
        except Exception as e:  # noqa: BLE001
            print(f"[plan] {seq}: not ready ({type(e).__name__}: {e})")
            continue
        recs.append({
            "seq_id": seq, "source_id": seq, "split": split,
            "T": int(len(data["frames"])), "fps": float(data["fps"]),
            "n_frames_total": int(data["n_frames_total"]),
            "anchor_ids": [int(a) for a in data["anchors"]],
            "roi_valid_frac": float(data["slot_valid"].mean()),
            "ignore_flags": int(data["slot_ignore"].sum()),
        })
    counts: dict = {}
    for r in recs:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    payload = {
        "dataset": DATASET, "task": "track",
        "detector_family": "FRCNN",
        "split_by": "sequence",
        "counts": counts,
        "records": recs,
        "selection_rule": (
            "One detector family (FRCNN) to avoid triplicating scenes; split by "
            "sequence: train={02,04,09}, val={05}, test={10,11}."),
    }
    contract.write_json(PLAN_PATH, payload)
    print(f"[plan] wrote {PLAN_PATH}  counts={counts}")
    for r in recs:
        print(f"[plan]   {r['seq_id']:<16} {r['split']:<5} T={r['T']:<4} "
              f"fps={r['fps']:.1f} anchors={r['anchor_ids']} "
              f"roi_valid={r['roi_valid_frac']:.2f} ignore={r['ignore_flags']}")


def do_finalize(_args) -> None:
    from src.data.clip_encoder import ENCODER_TAG, FEAT_DIM, MODEL_NAME, N_TOK, TOK_DIM
    plan = contract.read_json(PLAN_PATH)
    prog_path = FEAT_ROOT / "progress.json"
    prog = contract.read_json(prog_path) if prog_path.exists() else {"done": {}, "skipped": {}}
    done = prog.get("done", {})
    by_id = {r["seq_id"]: r for r in plan["records"]}

    seqs = []
    for seq_id, info in sorted(done.items()):
        rec = by_id.get(seq_id, {})
        anchors = rec.get("anchor_ids", [])
        seqs.append(contract.SeqRecord(
            seq_id=seq_id, source_id=seq_id,
            split=rec.get("split", "train"),
            T=int(info["T"]), path=info["path"].replace("\\", "/"),
            labels=["track", "id"],
            meta={
                "detector": "FRCNN",
                "fps": rec.get("fps"),
                "n_frames_total": rec.get("n_frames_total"),
                "n_frames_kept": int(info["T"]),
                "subsample_fps": mot17.SUBSAMPLE_FPS,
                "n_roi": mot17.N_ROI,
                "anchor_ids_original": anchors,
                "id_map": {str(j): int(a) for j, a in enumerate(anchors)},
                "roi_valid_frac": rec.get("roi_valid_frac"),
                "n_ignore_flagged": rec.get("ignore_flags"),
                "seq": seq_id,
            }))
    manifest = contract.Manifest(
        dataset=DATASET, task="track", feature_dim=FEAT_DIM, seqs=seqs,
        tok_dim=TOK_DIM, roi_dim=FEAT_DIM, n_roi=mot17.N_ROI,
        encoder={"name": MODEL_NAME, "tag": ENCODER_TAG, "input": 224,
                 "pool": "2x2", "n_tok": N_TOK, "dtype": "float16",
                 "frozen": True, "preprocess": "CLIPImageProcessor",
                 "roi_preprocess": "crop box + 10% margin, stretch to 224"},
        split_key="source_id",
        label_specs={"lab_track": {"kind": "mse", "dim": 4},
                     "lab_id": {"kind": "ce", "ignore": -1,
                                "dim": mot17.N_ROI}},
        extra={
            "selection_rule": plan["selection_rule"],
            "sampling": f"5 fps: original frame indices 1,1+step,... with "
                        f"step=round(fps/5) (fps=30 -> step 6); no cap",
            "timestamp_rule": "t = (original_1based_frame - 1) / seqinfo fps "
                              "(real physical time, dt=0.2 s)",
            "label_semantics": {
                "lab_track": "[T,N,4] normalised (cx,cy,w,h) of the anchor's NEXT "
                             "OBSERVED box (absolute box, not a delta). Defined "
                             "even while roi_valid=0 -> that is the occlusion "
                             "recovery target.",
                "lab_id": "[T,N] contiguous anchor id == slot index; -1 where "
                          "the anchor is invalid. Anchor slots are FIXED per "
                          "sequence so the same pedestrian always occupies the "
                          "same slot; meta.id_map maps slot -> original MOT id.",
                "roi_valid": "0 when conf!=1, class!=1, visibility<=0, box "
                             "degenerate, or the box overlaps an ignore region "
                             "(class 2/7/8/12, IoU>0.5).",
            },
            "anchor_selection": "the 8 pedestrian ids valid in the most frames "
                                "(cls==1, conf==1, visibility>0, w,h>1), ties "
                                "broken by ascending original id; final slot "
                                "order sorted by original id",
            "source_urls": SOURCE_URLS,
            "source_note": "MOT17 test split has no public GT; the 'test' here is "
                           "a held-out set of train-set sequences, split by "
                           "sequence exactly as required.",
            "encoder_config": {"model": MODEL_NAME, "input_size": 224,
                               "dtype": "float16", "device": "cuda"},
            "seed": 0, "deterministic": True,
            "n_planned": len(plan["records"]),
            "n_extracted": len(seqs),
            "skipped": prog.get("skipped", {}),
        })
    manifest.check_split_disjoint()
    manifest.save(MANIFEST_PATH)
    print(f"[manifest] {MANIFEST_PATH} counts={manifest.counts()}")


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
