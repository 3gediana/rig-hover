"""Confusion matrix on graded frames, plus what the clean subset proves.

Reports the deployed gesture classifier against the studio's own task labels,
separated into "all frames" and "clean frames only". The gap between the two is
the label-noise floor: accuracy that only appears after cleaning is not accuracy
the model has, it is accuracy the measurement method had.

Run:
    python tools/confusion_report.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ANALYSIS = Path(r"D:\robot_work\evidence\sessions_analysis")
FRAMES = ANALYSIS / "all_frames.csv"
GRADED = ANALYSIS / "frames_graded.csv"
NAMES = {0: "none", 1: "wave", 2: "open_palm", 3: "thumbs_up",
         4: "pointing", 5: "fist"}
GT = {"none": 0, "wave": 1, "open_palm": 2, "thumbs_up": 3, "pointing": 4,
      "fist": 5}


def main() -> int:
    rows = list(csv.DictReader(FRAMES.open(encoding="utf-8")))
    grades = {(r["session"], int(r["idx"])): r["grade"]
              for r in csv.DictReader(GRADED.open(encoding="utf-8"))}

    groups = {"all": rows,
              "clean": [r for r in rows
                        if grades.get((r["session"], int(r["idx"]))) == "clean"]}

    for label, data in groups.items():
        tasks = [t for t in GT if any(r["task"] == t for r in data)]
        print("=" * 74)
        print(f"{label} frames  (n={len(data)})   rows=ground truth, cols=predicted")
        print("=" * 74)
        hdr = f"{'truth':<11}" + "".join(f"{NAMES[i]:>11}" for i in range(6)) + f"{'recall':>9}"
        print(hdr)
        print("-" * len(hdr))
        accs = []
        for t in tasks:
            want = GT[t]
            sub = [r for r in data if r["task"] == t]
            if not sub:
                continue
            counts = defaultdict(int)
            for r in sub:
                counts[int(r["pred"])] += 1
            n = len(sub)
            row = f"{t:<11}"
            for i in range(6):
                c = counts.get(i, 0)
                row += f"{(f'{c}' if c else '.'):>11}"
            acc = 100.0 * counts.get(want, 0) / n
            accs.append(acc)
            row += f"{acc:>8.1f}%"
            print(row)
        print("-" * len(hdr))
        # The ACK decision is what the teacher acts on: any of wave/open_palm/
        # thumbs_up counts as "the operator called me".
        ack_hits = ack_tot = 0
        for t, want in GT.items():
            if want not in (1, 2, 3):
                continue
            sub = [r for r in data if r["task"] == t]
            if not sub:
                continue
            ack_tot += len(sub)
            ack_hits += sum(1 for r in sub if int(r["pred"]) in (1, 2, 3))
        false_ack = 0
        false_tot = 0
        for t in ("none", "fist"):
            sub = [r for r in data if r["task"] == t]
            if not sub:
                continue
            false_tot += len(sub)
            false_ack += sum(1 for r in sub if int(r["pred"]) in (1, 2, 3))
        print(f"gesture accuracy (exact class)  {sum(accs)/max(len(accs),1):5.1f}%")
        if ack_tot:
            print(f"ACK recall (call-gesture->any of 1/2/3)  {100.0*ack_hits/ack_tot:5.1f}%"
                  f"  ({ack_hits}/{ack_tot})")
        if false_tot:
            print(f"false ACK (none/fist -> any of 1/2/3)   "
                  f"{100.0*false_ack/false_tot:5.1f}%  ({false_ack}/{false_tot})")
        print()

    out = ANALYSIS / "confusion.json"
    payload = {}
    for label, data in groups.items():
        payload[label] = {}
        for t, want in GT.items():
            sub = [r for r in data if r["task"] == t]
            if not sub:
                continue
            counts = defaultdict(int)
            for r in sub:
                counts[int(r["pred"])] += 1
            payload[label][t] = {NAMES[k]: v for k, v in sorted(counts.items())}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
