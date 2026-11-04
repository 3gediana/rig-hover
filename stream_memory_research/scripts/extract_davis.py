"""Stage 4 feature extraction: DAVIS 2017 -> CLIP feature cache (occlusion recovery).

  python scripts/extract_davis.py --extract      # decode + encode -> npz
  python scripts/extract_davis.py --finalize     # build data_manifests/davis_vos.json

Everything streams out of the 0.78 GB zip (no 12,600-file disk extraction):
``src.data.davis.decode_sequence`` reads the JPEGs and annotation PNGs it needs
from the archive in a worker thread and returns CLIP-geometry frames, ROI crops
from the GT mask bbox, boxes, labels and the simulated-occlusion mask.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402
from src.data import davis  # noqa: E402
from src.data.clip_encoder import CLIPEncoder, FEAT_DIM, N_TOK, TOK_DIM  # noqa: E402
from src.data.extract import ExtractionRunner  # noqa: E402
from src.data.util import attach_log, dir_size_gb  # noqa: E402

ZIP = r"G:\stream_memory_data\videos\davis\DAVIS-2017-trainval-480p.zip"
FEAT_ROOT = Path(r"G:\stream_memory_data\features\davis_vos")
DATASET = "davis_vos"
SEED = 0

_CFG = {"zip": ZIP}
_FRAMES: dict = {}


def decode_worker(job: dict) -> dict:
    seq = job["seq_id"]
    try:
        jpegs = davis.list_frames(_CFG["zip"], seq)
        if not jpegs:
            return {"seq_id": seq, "skip_reason": "no frames in zip"}
        d = davis.decode_sequence(_CFG["zip"], seq, jpegs)
    except Exception as e:  # noqa: BLE001
        return {"seq_id": seq, "skip_reason": f"{type(e).__name__}: {e}"}
    return {"seq_id": seq, "frames": d["frames"], "t": d["t"],
            "roi_crops": d["roi_crops"], "roi_valid": d["roi_valid"],
            "roi_meta": d["roi_meta"], "lab_track": d["lab_track"],
            "lab_present": d["lab_present"], "meta": d["meta"]}


def plan_jobs() -> list:
    seqs, split = davis.list_sequences(ZIP)
    if not seqs:
        raise RuntimeError(f"{ZIP}: no sequences found")
    jobs = [{"seq_id": s, "split": split.get(s, "train")} for s in seqs]
    # deterministic: shuffle-free, but interleave train/val so the first
    # sequences extracted already cover both splits (helps a partial run)
    jobs.sort(key=lambda r: (r["split"] != "val", r["seq_id"]))
    return jobs


def do_extract(args) -> None:
    jobs = plan_jobs()
    print(f"[davis] {len(jobs)} sequences: "
          f"{sum(1 for j in jobs if j['split'] == 'train')} train / "
          f"{sum(1 for j in jobs if j['split'] == 'val')} val", flush=True)
    enc = CLIPEncoder(batch_size=args.batch, min_batch=8)
    enc.warmup(4)
    runner = ExtractionRunner(
        dataset=DATASET, out_dir=FEAT_ROOT,
        progress_path=FEAT_ROOT / "progress.json", encoder=enc,
        num_workers=args.workers, log_every=10,
        progress_meta={"dataset": DATASET, "native_fps": davis.NATIVE_FPS,
                       "target_fps": davis.TARGET_FPS, "n_roi": davis.N_ROI,
                       "encoder_tag": "clip-vit-base-p16-224-f16-2x2tok-v1"})
    t0 = time.time()
    summary = runner.run(jobs, decode_worker, stop_after=args.limit)
    print(f"[extract] {summary}", flush=True)
    print(f"[extract] wall {(time.time() - t0) / 60:.1f} min, "
          f"feat dir {dir_size_gb(FEAT_ROOT):.2f} GB", flush=True)


def do_finalize(_args) -> None:
    from src.data.clip_encoder import ENCODER_TAG, MODEL_NAME
    seqs, split = davis.list_sequences(ZIP)
    prog_path = FEAT_ROOT / "progress.json"
    prog = contract.read_json(prog_path) if prog_path.exists() else {"done": {}}
    done = prog.get("done", {})
    recs = []
    for seq in sorted(done):
        info = done[seq]
        recs.append(contract.SeqRecord(
            seq_id=seq, source_id=seq, split=split.get(seq, "train"),
            T=int(info["T"]), path=info["path"].replace("\\", "/"),
            labels=["track", "present"],
            meta={"native_fps": davis.NATIVE_FPS, "target_fps": davis.TARGET_FPS,
                  "n_frames_kept": int(info["T"]),
                  "split_official": split.get(seq, "train")}))
    m = contract.Manifest(
        dataset=DATASET, task="track", feature_dim=FEAT_DIM, seqs=recs,
        tok_dim=TOK_DIM, roi_dim=FEAT_DIM, n_roi=davis.N_ROI,
        encoder={"name": MODEL_NAME, "tag": ENCODER_TAG, "input": 224,
                 "pool": "2x2", "n_tok": N_TOK, "dtype": "float16",
                 "frozen": True, "preprocess": "CLIPImageProcessor",
                 "roi_preprocess": "crop mask-bbox + 10% margin, stretch 224"},
        split_key="source_id",
        label_specs={
            "lab_track": {"kind": "mse", "dim": 4},
            "lab_present": {"kind": "bce", "n_classes": 1, "dim": 1}},
        extra={
            "sampling": f"{davis.TARGET_FPS:.0f} fps target: frame indices "
                        f"round(k * {davis.NATIVE_FPS:.0f}/{davis.TARGET_FPS:.0f}) "
                        f"over the native image sequence, cap {davis.MAX_FRAMES}",
            "timestamp_rule": f"t = frame_index / {davis.NATIVE_FPS:.0f} "
                              f"(DAVIS 2017 release is 24 fps; no per-sequence "
                              f"fps field exists in the archive)",
            "split": "official DAVIS 2017 ImageSets/2017 train (60) / val (30), "
                     "split by sequence, never by frame",
            "roi": "mask bbox of the annotated object, normalised (cx,cy,w,h)",
            "roi_valid": "0 where the mask is empty OR inside a simulated "
                         "occlusion gap (see occlusion_simulation)",
            "labels": {"lab_track": "[T,1,4] normalised bbox of the object at "
                                    "frame t -- supervised even while "
                                    "roi_valid=0 (memory target)",
                       "lab_present": "[T,1] 1 if the object mask is non-empty "
                                      "(true visibility, not the injected flag)"},
            "occlusion_simulation": {
                "why": "DAVIS is a VOS benchmark: the annotated object is present "
                       "in essentially every frame, so the literal 'mask empty' "
                       "rule produces no invalid frames and cannot test memory.",
                "how": "per sequence, 1-3 gaps of 0.4-1.5 s (seeded by "
                       "(seed, seq), deterministic), total <= 40% of frames, "
                       "never frame 0, separated by >= 1 s of valid data",
                "what_is_real": "pixels, CLIP features, mask bboxes, lab_present",
                "what_is_injected": "only the roi_valid flag",
                "seed": SEED,
            },
            "source_urls": [
                "https://huggingface.co/datasets/yinloonga/DAVIS_2017 "
                "(DAVIS-2017-trainval-480p.zip, 832,766,765 B, contains "
                "JPEGImages/480p + Annotations/480p for all 90 sequence)",
                "official https://data.vision.ee.ethz.ch/csergi/share/davis/"
                "DAVIS-2017-trainval-480p.zip measured at 0.04 MB/s -> not used",
            ],
            "encoder_config": {"model": MODEL_NAME, "input_size": 224,
                               "dtype": "float16", "device": "cuda"},
            "seed": SEED, "deterministic": True,
            "n_extracted": len(recs),
            "skipped": prog.get("skipped", {}),
        })
    m.check_split_disjoint()
    m.save(contract.MANIFESTS / f"{DATASET}.json")
    print(f"[manifest] {contract.MANIFESTS / (DATASET + '.json')} "
          f"counts={m.counts()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--log", default=r"G:\stream_memory_data\logs\extract_davis.log")
    args = ap.parse_args()
    if not args.extract and not args.finalize:
        args.extract = True
    attach_log(args.log)
    if args.extract:
        do_extract(args)
    if args.finalize:
        do_finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
