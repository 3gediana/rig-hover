"""Render the ucf101_order task construction with REAL frames (no GPU needed).

Shows that the two classes contain the SAME frames in a different order, so any
model that pools frames / looks only at the current frame is at chance by
construction.  Output: reports/figures/task_ucf101_order_construction.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
VDIR = Path(r"G:\stream_memory_data\videos\ucf101")
OUT = ROOT / "reports" / "figures" / "task_ucf101_order_construction.png"

CELL = 132
NCOL = 8
PAD_L = 210          # space for the row labels
PAD_T = 126
PAD_B = 54


def pick_clips(man: dict, n: int = 2) -> list[dict]:
    """Two visually obvious clips (different classes) from the test split."""
    seen, out = set(), []
    for s in sorted(man["seqs"], key=lambda r: r["seq_id"]):
        if s["split"] != "test" or s["meta"].get("order_label") != 0:
            continue
        cls = s["meta"]["class"]
        if cls in seen:
            continue
        seen.add(cls)
        out.append(s)
        if len(out) == n:
            break
    return out


ZIP = VDIR / "UCF-101.zip"


def _materialise(rel: str) -> Path:
    """UCF101 was never unzipped -- pull one clip out of the 6.95 GB zip."""
    if (VDIR / rel).exists():
        return VDIR / rel
    tmp = ROOT / "logs" / "_ucf_frames"
    tmp.mkdir(parents=True, exist_ok=True)
    dst = tmp / Path(rel).name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    import zipfile
    with zipfile.ZipFile(ZIP) as z:
        name = next((n for n in z.namelist()
                     if n.replace("\\", "/").endswith(rel.replace("\\", "/"))), None)
        if name is None:
            raise SystemExit(f"{rel} not in {ZIP}")
        with z.open(name) as f, open(dst, "wb") as o:
            o.write(f.read())
    return dst


def read_frames(s: dict, n: int) -> tuple[list[np.ndarray], str]:
    """Decode the clip and return n evenly-spaced BGR frames + the class name."""
    rel = s["meta"]["relpath"]
    path = _materialise(rel)
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise SystemExit(f"cannot decode {path}")
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx], s["meta"]["class"]


def label(img: np.ndarray, lines: list[str], color=(0, 0, 0),
          scale: float = 0.52, y0: int = 0) -> None:
    """Multi-line row label, top-left of the cell block, offset by y0."""
    for k, text in enumerate(lines):
        y = y0 + 26 + 26 * k
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, 1, cv2.LINE_AA)


def main() -> int:
    man = json.load(open(ROOT / "data_manifests" / "ucf101_order.json", encoding="utf-8"))
    clips = pick_clips(man, 2)

    rows: list[tuple[list[str], list[np.ndarray], tuple]] = []
    for s in clips:
        fr, cls = read_frames(s, NCOL)
        rows.append(([cls, "forward", "(t ->)  label=0"], fr, (30, 110, 30)))
        rows.append(([cls, "REVERSED", "(t <-)  label=1"], list(reversed(fr)),
                     (30, 30, 170)))

    H = PAD_T + len(rows) * CELL + PAD_B
    W = PAD_L + NCOL * CELL
    canvas = np.full((H, W, 3), 255, np.uint8)

    for r, (texts, frs, color) in enumerate(rows):
        y0 = PAD_T + r * CELL
        label(canvas, texts, color, 0.46, y0)
        # the row label wraps over two lines for readability
        for c, fr in enumerate(frs):
            x0 = PAD_L + c * CELL
            cell = cv2.resize(fr, (CELL - 4, CELL - 4), interpolation=cv2.INTER_AREA)
            canvas[y0:y0 + CELL - 4, x0:x0 + CELL - 4] = cell
            cv2.putText(canvas, f"#{c}", (x0 + 4, y0 + CELL - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"#{c}", (x0 + 4, y0 + CELL - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    t1 = ("ucf101_order  |  real UCF101 video  |  SAME 48 frames, only the ORDER differs")
    t2 = (f"T=48, dt=0.08s, span=3.24s   train 4902 / val 146 / test 1212 pairs   "
          f"per-frame MLP is at 50.0% by construction")
    cv2.putText(canvas, t1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, t2, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, "bottom row of each pair = time-reversed frame order (same frames)",
                (12, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), canvas)
    print("wrote", OUT, canvas.shape)
    for s in clips:
        print("  clip:", s["meta"]["relpath"], "class:", s["meta"]["class"],
              "n_frames_total:", s["meta"]["n_frames_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
