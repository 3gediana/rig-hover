"""Per-session perception dump for the studio recordings.

Runs the deployed pipeline over every frame of every studio session and writes
one row per frame, including the wrist confidence the gesture classifier
actually crops around. That last column is what separates "the model saw a hand"
from "the model hallucinated a hand out of a sleeve".

Writes:
    evidence/sessions_analysis/all_frames.csv
    evidence/sessions_analysis/<session>/detections.csv   (per-session detail)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SESS = Path(r"D:\robot_work\evidence\sessions")
OUT = Path(r"D:\robot_work\evidence\sessions_analysis")
YUNET = Path(r"D:\robot_work\pc\perception\assets/face_detection_yunet_2023mar.onnx")

FIELDS = ["session", "task", "idx", "pred", "conf", "person_h", "seen",
          "face", "theta_fist", "wrist_conf", "wrist_xy", "bbox_px"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, r"D:\robot_work\pc\perception")
    from run_perception import PerceptionPipeline  # noqa: PLC0415

    rows = [json.loads(l) for l in (SESS / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    pipe = PerceptionPipeline(device="cuda:0")
    face = cv2.FaceDetectorYN_create(str(YUNET), "", (240, 240), 0.45, 0.3, 20)

    all_rows: list[dict] = []
    for r in rows:
        sess, task = r["dir"], r["task"]
        d = SESS / sess / "frames"
        files = sorted(d.glob("*.jpg"))
        per = []
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            out = pipe(img)
            face.setInputSize((img.shape[1], img.shape[0]))
            _rc, fs = face.detect(img)
            has_face = fs is not None and len(fs) > 0

            kp = np.asarray(out.kp_xy)
            kc = np.asarray(out.kp_conf)
            wrists = []
            for ki in (9, 10):                     # COCO left/right wrist
                if ki < len(kc) and float(kc[ki]) >= 0.20:
                    wrists.append((round(float(kp[ki, 0]), 1),
                                   round(float(kp[ki, 1]), 1),
                                   round(float(kc[ki]), 3)))
            row = {
                "session": sess, "task": task, "idx": i,
                "pred": int(out.gesture_raw_id),
                "conf": round(float(out.gesture_raw_conf), 3),
                "person_h": round(float(out.person_h), 3),
                "seen": int(out.seen), "face": int(has_face),
                "theta_fist": round(float(out.theta_fist), 2),
                "wrist_conf": round(max([c for (_x, _y, c) in wrists],
                                        default=0.0), 3),
                "wrist_xy": str(wrists),
                "bbox_px": str(tuple(round(float(v), 1) for v in out.bbox_px)),
            }
            per.append(row)
            all_rows.append(row)
        sub = OUT / sess
        sub.mkdir(exist_ok=True)
        with (sub / "detections.csv").open("w", newline="",
                                          encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(per)
        print(f"{sess:<32}{task:<12}{len(per):>5} frames", flush=True)

    with (OUT / "all_frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\ntotal {len(all_rows)} frames -> {OUT/'all_frames.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
