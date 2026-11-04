import glob
import os
import re
import subprocess
import sys

PY = sys.executable
BENCH = r"D:\robot_work\pc\bench"
TRAIN = r"D:\robot_work\pc\train"
HOLD = os.path.join(TRAIN, "data_pan_holdout2")

rows = []
for d in sorted(glob.glob(os.path.join(TRAIN, "out_sw_*"))):
    blob = os.path.join(d, "m.blob")
    if not os.path.exists(blob):
        rows.append((os.path.basename(d)[7:], "no blob", "", ""))
        continue
    r = subprocess.run([PY, "eval_session.py", "--blob", blob, "--sessions", HOLD],
                       cwd=BENCH, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    mae = base = ack = "?"
    for line in (r.stdout or "").splitlines():
        if line.startswith("POOLED"):
            parts = line.split()
            # POOLED  3880  0.97  7.80  98.9%  0.3%
            try:
                mae, base = parts[2], parts[3]
            except IndexError:
                pass
        if line.startswith("ACK "):
            parts = line.split()
            try:
                ack = parts[3]
            except IndexError:
                pass
    rows.append((os.path.basename(d)[7:], mae, base, ack))

print(f"{'config':<22}{'held-out MAE':>14}{'baseline':>10}{'ACK recall':>12}")
print("-" * 58)
for name, mae, base, ack in rows:
    print(f"{name:<22}{mae:>14}{base:>10}{ack:>12}")
