"""Build the robot-domain (case study) task from the real robot camera sessions.

Source: `D:\\robot_work\\evidence\\sessions\\<session>\\frames\\*.jpg`
        240x240 GREY JPEG captured by the deployed GC0308 camera, ~10 fps,
        10 scripted sessions (one gesture/event class each), ~200 frames each.

Why this task exists
--------------------
It is the only real data from the actual deployment camera.  It is far too small
to be a benchmark (10 classes x ~200 frames, one session per class), so it is
reported **only as a case study / qualitative domain check** and is NEVER part of
the main scientific claim.  See DESIGN.md and the final report's limitation section.

Protocol (honest about its weakness)
------------------------------------
Each session is chopped into contiguous sequences of 64 frames (stride 64, no
overlap).  Sequence index 0/1/2 of every session goes to train/val/test, so each
split contains all 10 classes.  `source_id` is the sequence id, so the split
invariant holds -- but sequences from the same session remain temporally adjacent,
so absolute accuracy is optimistic.  All models face the identical protocol, so
relative comparison is still informative; it is not used to support any claim.

Output
------
G:\\stream_memory_data\\features\\robot_gesture\\seq_<session>_<k>.npz
    feat [T,768] f16, tok [T,4,768] f16, t [T] f32,
    lab_class [T] int64 (session class index)
data_manifests/robot_gesture.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402

SESSIONS_DIR = Path(r"D:\robot_work\evidence\sessions")
DATASET = "robot_gesture"
SEQ_LEN = 64
SEQ_STRIDE = 64
FPS = 10.0  # robot link is ~10 fps for /capture; recorded frames are 1 per capture

# stable class order (alphabetical by session task, but keep a readable list)
CLASS_ORDER = ["empty", "face_close", "fist", "knock", "lean_out",
               "none", "open_palm", "pointing", "thumbs_up", "wave"]


def read_sessions() -> List[dict]:
    out = []
    for d in sorted(SESSIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        frames = sorted((d / "frames").glob("*.jpg"))
        if len(frames) < SEQ_LEN:
            continue
        meta_p = d / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        task = meta.get("task") or d.name.split("_")[0]
        if task not in CLASS_ORDER:
            print(f"[skip] unknown task {task!r} in {d.name}")
            continue
        out.append({"dir": d, "frames": frames, "task": task,
                    "cls": CLASS_ORDER.index(task), "meta": meta})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit-sessions", type=int, default=0)
    args = ap.parse_args()

    from src.data.clip_encoder import CLIPEncoder

    sessions = read_sessions()
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
    print(f"[robot] {len(sessions)} sessions, "
          f"{sum(len(s['frames']) for s in sessions)} frames total")

    enc = CLIPEncoder(device=args.device, batch_size=args.batch)
    enc.warmup()

    out_dir = contract.FEATURES / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    seqs: List[SeqRecord] = []
    t0 = __import__("time").time()
    n_frames_done = 0
    for si, s in enumerate(sessions):
        import cv2
        imgs = []
        for f in s["frames"]:
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            imgs.append(np.repeat(g[:, :, None], 3, axis=2))  # grey -> RGB
        if len(imgs) < SEQ_LEN:
            print(f"[skip] {s['dir'].name}: only {len(imgs)} decodable frames")
            continue
        # encode in chunks to bound memory
        feats, toks = [], []
        CH = 256
        for i in range(0, len(imgs), CH):
            f_, t_ = enc.encode_frames(imgs[i:i + CH])
            feats.append(f_)
            toks.append(t_)
        feat = np.concatenate(feats, 0)
        tok = np.concatenate(toks, 0)
        n_frames_done += len(imgs)

        n_chunks = (len(imgs) - SEQ_LEN) // SEQ_STRIDE + 1
        for k in range(min(n_chunks, 3)):
            a = k * SEQ_STRIDE
            b = a + SEQ_LEN
            seq_id = f"{s['dir'].name}_k{k}"
            split = ["train", "val", "test"][k]
            npz = out_dir / f"seq_{seq_id}.npz"
            np.savez_compressed(
                npz,
                feat=feat[a:b],
                tok=tok[a:b],
                t=(np.arange(b - a, dtype=np.float32) / FPS),
                lab_class=np.full(b - a, s["cls"], dtype=np.int64),
            )
            seqs.append(SeqRecord(
                seq_id=seq_id, source_id=seq_id, split=split, T=SEQ_LEN,
                path=str(npz), labels=["class"],
                meta={"session": s["dir"].name, "task": s["task"],
                      "cls": s["cls"], "chunk": k,
                      "distance_m": s["meta"].get("distance_m"),
                      "light": s["meta"].get("light")},
            ))
        print(f"[{si + 1}/{len(sessions)}] {s['dir'].name}: {len(imgs)} frames -> {min(n_chunks, 3)} seqs")

    man = Manifest(
        dataset=DATASET, task="action", feature_dim=768, tok_dim=768,
        seqs=seqs,
        encoder={"name": "openai/clip-vit-base-patch16", "input": 224,
                 "pool": "2x2", "feats": "CLS + 4 pooled patch tokens"},
        label_specs={"class": {"kind": "ce", "ignore": -1, "dim": len(CLASS_ORDER)}},
        extra={
            "note": "case study only; one session per class; sequences from the same "
                    "session stay temporally adjacent -> absolute numbers optimistic",
            "classes": CLASS_ORDER, "fps": FPS,
            "source": str(SESSIONS_DIR), "seq_len": SEQ_LEN, "stride": SEQ_STRIDE,
            "n_frames_encoded": n_frames_done,
            "encode_sec": round(__import__("time").time() - t0, 1),
            "main_metric": "acc",
        },
    )
    p = man.save(contract.MANIFESTS / f"{DATASET}.json")
    man.check_split_disjoint()
    print(f"[robot] wrote {p}  seqs={len(seqs)}  splits={man.counts()}  "
          f"frames={n_frames_done}  {man.extra['encode_sec']}s")


if __name__ == "__main__":
    main()
