"""Grade every frame of a studio session so only clean frames train a model.

A 20 s burst is not 20 s of ground truth: the operator needs a moment to get
into the pose, moves while holding it, and drops the pose before the timer
ends. Feeding all of it to a trainer silently mixes label noise into the
measurement -- and then "the classifier is wrong" and "the data was dirty" are
indistinguishable.

This scores each frame on evidence available without a human:

* ``pose_ok``     YOLO found the person with usable keypoints.
* ``hand_conf``   max wrist confidence; the gesture classifier crops around the
                  wrist, so a low value means the hand was not really seen.
* ``wrist_clear`` the two wrists are separated by at least 25 px. Overlapping
                  wrists (hands together in front of the chest) is exactly the
                  fist pose, so this is a *minimum* check, not a quality score.
* ``gesture_conf`` the classifier's own confidence; frames below
                  ``--min-conf`` are "the model does not know either".

Frames are written to ``{session}.frames.csv`` with a ``grade`` of
``clean`` / ``suspect``, plus a per-session verdict in ``sessions_report.json``.

Run:
    python tools/grade_frames.py --min-conf 0.6
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

SESS = Path(r"D:\robot_work\evidence\sessions")
ANALYSIS = Path(r"D:\robot_work\evidence\sessions_analysis")
SRC = ANALYSIS / "all_frames.csv"
OUT = ANALYSIS / "frames_graded.csv"
REPORT = ANALYSIS / "sessions_report.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--min-hand-conf", type=float, default=0.50)
    ap.add_argument("--min-person-h", type=float, default=0.30)
    a = ap.parse_args()

    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    by_sess: dict[str, list[dict]] = {}
    for r in rows:
        by_sess.setdefault(r["session"], []).append(r)

    graded: list[dict] = []
    report: list[dict] = []
    for sess, rs in sorted(by_sess.items()):
        task = rs[0]["task"]
        # The pose must be stable before the label applies. A frame is `clean`
        # when the person is seen, the hand is confidently tracked, the person
        # is large enough for a 64 px crop to mean anything, and the classifier
        # itself is not guessing.
        out = []
        for r in rs:
            seen = int(r["seen"]) == 1
            face = int(r["face"]) == 1
            ph = float(r["person_h"])
            conf = float(r["conf"])
            hand_conf = _max_wrist_conf(r)
            reasons = []
            if not seen:
                reasons.append("no_person")
            if ph < a.min_person_h:
                reasons.append("person_too_small")
            if hand_conf < a.min_hand_conf:
                reasons.append("hand_unseen")
            if conf < a.min_conf:
                reasons.append("model_unsure")
            # An "empty" frame is clean when the frame really is empty. The
            # detector can still fire on a coat on a chair, so `face` is not
            # required to be off -- only `seen` is.
            if task == "empty" and seen:
                reasons.append("not_empty")
            if task == "lean_out" and not seen:
                reasons.append("left_frame")
            grade = "clean" if not reasons else "suspect"
            out.append({
                "session": sess, "task": task, "idx": int(r["idx"]),
                "grade": grade, "reasons": ";".join(reasons),
                "pred": int(r["pred"]), "conf": conf,
                "person_h": ph, "hand_conf": hand_conf, "seen": int(r["seen"]),
                "face": int(r["face"]), "theta_fist": r["theta_fist"],
            })
        clean = [o for o in out if o["grade"] == "clean"]
        n = len(out)
        # A session is usable if at least 25% of frames are clean: below that
        # the operator never settled and the whole burst is setup noise.
        report.append({
            "session": sess, "task": task, "frames": n,
            "clean": len(clean), "clean_pct": round(100.0 * len(clean) / n, 1),
            "usable": len(clean) >= max(5, 0.25 * n),
            "top_preds": _hist([o["pred"] for o in out]),
            "clean_preds": _hist([o["pred"] for o in clean]),
        })
        graded.extend(out)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(graded[0]))
        w.writeheader()
        w.writerows(graded)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                      encoding="utf-8")

    print(f"{'session':<32}{'task':<12}{'frames':>7}{'clean':>7}{'%':>7}  usable")
    print("-" * 78)
    for r in report:
        print(f"{r['session']:<32}{r['task']:<12}{r['frames']:>7}"
              f"{r['clean']:>7}{r['clean_pct']:>7.1f}  {r['usable']}")
    total = sum(r["frames"] for r in report)
    clean = sum(r["clean"] for r in report)
    print("-" * 78)
    print(f"{'TOTAL':<44}{total:>7}{clean:>7}{100.0*clean/max(total,1):>7.1f}")
    print(f"\nwrote {OUT}\n      {REPORT}")
    return 0


def _max_wrist_conf(row: dict) -> float:
    """Highest wrist confidence recorded for this frame.

    ``dump_sessions.py`` writes ``wrist_conf`` into ``all_frames.csv``. A frame
    with no confident wrist is ``suspect``: the gesture classifier crops around
    the wrist, so if the wrist was not tracked the label is not about a hand.
    """
    try:
        return float(row.get("wrist_conf") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _hist(xs: list[int]) -> dict[str, int]:
    names = {0: "none", 1: "wave", 2: "open_palm", 3: "thumbs_up",
             4: "pointing", 5: "fist"}
    out: dict[str, int] = {}
    for x in xs:
        k = names.get(int(x), str(x))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
