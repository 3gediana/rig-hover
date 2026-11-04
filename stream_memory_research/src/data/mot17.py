"""MOT17 multi-object tracking / occlusion-memory task construction.

Dataset facts
-------------
* ``MOT17.zip`` (motchallenge.net, 5.86 GB) contains the 21 *train* sequences
  (7 original scenes x 3 detector families) plus test sequences.  Only the
  train split ships ``gt/gt.txt``; the test GT is withheld.  We therefore use
  train-set sequences only and treat the requested "test" sequences as a
  held-out set of *tracked* sequences -- see the split note below.
* ``gt.txt`` rows: ``frame, id, x, y, w, h, conf, class, visibility``
  (1-based frame, class 1 = pedestrian, conf 1 = valid annotation).
* One detector family only (FRCNN) to avoid the same scene appearing 3x.

Split (by sequence, never by frame -- DESIGN.md §0.2)
    train : MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-09-FRCNN
    val   : MOT17-05-FRCNN
    test  : MOT17-10-FRCNN, MOT17-11-FRCNN

Anchor-slot design
------------------
The 8 ROI slots are *fixed anchor tracks* chosen once per sequence (the tracks
valid in the most frames).  Slot ``j`` therefore always refers to the same
physical pedestrian, which is what makes ``roi_valid=0`` meaningful: when the
anchor is occluded/absent the slot goes invalid and only memory can predict
where it will reappear.  ``lab_track[t]`` is the anchor's *next observed*
normalised box (cx,cy,w,h) so the label is available even while the ROI is
invalid -- that is exactly the occlusion-recovery signal the research needs.
"""
from __future__ import annotations

import configparser
import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

N_ROI = 8
SUBSAMPLE_FPS = 5.0
MAX_FRAMES = 200                   # MOT17-05 is 14 fps -> 279 frames at 5 fps
IGNORE_CLASSES = {2, 7, 8, 12}     # person-on-vehicle / static / distractor / reflection
VIS_MIN = 0.0                      # visibility must be > this to count as valid
IOU_IGNORE = 0.5                   # anchor box overlapping an ignore region -> invalid

SEQ_SPLIT: Dict[str, str] = {
    "MOT17-02-FRCNN": "train",
    "MOT17-04-FRCNN": "train",
    "MOT17-09-FRCNN": "train",
    "MOT17-05-FRCNN": "val",
    "MOT17-10-FRCNN": "test",
    "MOT17-11-FRCNN": "test",
}


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def read_seqinfo(seq_dir: Path) -> dict:
    ini = configparser.ConfigParser()
    path = seq_dir / "seqinfo.ini"
    out = {"fps": 30.0, "seq_length": 0, "im_width": 0, "im_height": 0, "name": seq_dir.name}
    if path.exists():
        ini.read(path, encoding="utf-8")
        s = ini["Sequence"] if "Sequence" in ini else {}
        out["fps"] = float(s.get("frameRate", 30))
        out["seq_length"] = int(s.get("seqLength", 0))
        out["im_width"] = int(s.get("imWidth", 0))
        out["im_height"] = int(s.get("imHeight", 0))
        out["name"] = s.get("name", seq_dir.name)
    return out


def read_gt(seq_dir: Path) -> Dict[int, List[dict]]:
    """frame (1-based) -> list of {id, box(x,y,w,h), conf, cls, vis}."""
    gt = seq_dir / "gt" / "gt.txt"
    out: Dict[int, List[dict]] = {}
    if not gt.exists():
        return out
    with open(gt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                fr = int(float(parts[0]))
                tid = int(float(parts[1]))
                x, y, w, h = (float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]))
                conf = int(float(parts[6])) if len(parts) > 6 else 1
                cls = int(float(parts[7])) if len(parts) > 7 else 1
                vis = float(parts[8]) if len(parts) > 8 else 1.0
            except ValueError:
                continue
            out.setdefault(fr, []).append(
                {"id": tid, "box": (x, y, w, h), "conf": conf, "cls": cls, "vis": vis})
    return out


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def choose_anchors(gt: Dict[int, List[dict]], n: int = N_ROI) -> List[int]:
    """Pick ``n`` stable pedestrian ids: most frames with a valid visible box."""
    counts: Dict[int, int] = {}
    for fr, rows in gt.items():
        for r in rows:
            if r["cls"] == 1 and r["conf"] == 1 and r["vis"] > VIS_MIN \
                    and r["box"][2] > 1 and r["box"][3] > 1:
                counts[r["id"]] = counts.get(r["id"], 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return sorted(tid for tid, _ in ranked)


def frame_indices(seq_len: int, fps: float, target_fps: float = SUBSAMPLE_FPS,
                  cap: Optional[int] = None) -> np.ndarray:
    """1-based original frame indices subsampled to ~target_fps."""
    step = max(1, int(round(fps / target_fps)))
    idx = np.arange(1, seq_len + 1, step, dtype=np.int64)
    if cap is not None and len(idx) > cap:
        keep = np.linspace(0, len(idx) - 1, cap).astype(np.int64)
        idx = idx[np.unique(keep)]
    return idx


def assemble_sequence(seq_dir: Path, n_roi: int = N_ROI) -> dict:
    """Build all per-frame structure for one sequence from gt only (no pixels).

    Returns dict with ``frames``(1-based), ``fps``, ``anchor_ids``(orig ids),
    ``slot_valid`` [T,N] uint8, ``slot_box`` [T,N,4] normalised (cx,cy,w,h),
    ``slot_box_next`` [T,N,4], ``slot_ignore`` [T,N] bool, ``next_frame`` [T].
    """
    info = read_seqinfo(seq_dir)
    gt = read_gt(seq_dir)
    if not gt:
        raise FileNotFoundError(f"{seq_dir}: no gt/gt.txt")
    fps = float(info["fps"]) or 30.0
    seq_len = int(info["seq_length"]) or max(gt.keys())
    W = float(info["im_width"]) or 1920.0
    H = float(info["im_height"]) or 1080.0
    frames = frame_indices(seq_len, fps, cap=MAX_FRAMES)
    anchors = choose_anchors(gt, n_roi)
    if not anchors:
        raise ValueError(f"{seq_dir}: no valid pedestrian tracks")
    aidx = {tid: j for j, tid in enumerate(anchors)}
    T, N = len(frames), len(anchors)

    slot_box = np.zeros((T, N, 4), np.float32)      # normalised current box
    slot_valid = np.zeros((T, N), np.uint8)
    slot_ignore = np.zeros((T, N), bool)

    for i, fr in enumerate(frames):
        rows = gt.get(int(fr), [])
        ignore_boxes = [r["box"] for r in rows if r["cls"] in IGNORE_CLASSES]
        for r in rows:
            if r["cls"] != 1 or r["conf"] != 1:
                continue
            j = aidx.get(r["id"])
            if j is None:
                continue
            x, y, w, h = r["box"]
            if w <= 1 or h <= 1 or r["vis"] <= VIS_MIN:
                continue
            inside = any(iou_xywh((x, y, w, h), ib) > IOU_IGNORE
                         for ib in ignore_boxes)
            if inside:
                slot_ignore[i, j] = True
                continue
            # MOT17 boxes routinely extend past the image border.  Clip to the
            # frame so roi_meta stays in [0,1] (the contract) and so the stored
            # box is exactly what the ROI crop shows; a box entirely outside the
            # frame is degenerate -> invalid slot.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            cw, ch = x1 - x0, y1 - y0
            if cw <= 1 or ch <= 1:
                slot_ignore[i, j] = True
                continue
            cx = (x0 + cw / 2) / W
            cy = (y0 + ch / 2) / H
            slot_box[i, j] = (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                              min(cw / W, 1.0), min(ch / H, 1.0))
            slot_valid[i, j] = 1

    # Next *observed* box for each slot: the first frame strictly after i where
    # the slot is valid again; if the slot never comes back we fall back to the
    # last known box at i.  Getting this label while ``roi_valid=0`` is exactly
    # the occlusion-recovery supervision the task is built around.
    slot_box_next = np.zeros_like(slot_box)
    for j in range(N):
        valid_i = np.flatnonzero(slot_valid[:, j] > 0)
        for i in range(T):
            k = int(np.searchsorted(valid_i, i + 1))
            slot_box_next[i, j] = (slot_box[valid_i[k], j] if k < len(valid_i)
                                   else slot_box[i, j])
    return {
        "info": info, "fps": fps, "frames": frames, "anchors": anchors,
        "slot_box": slot_box, "slot_valid": slot_valid,
        "slot_ignore": slot_ignore, "slot_box_next": slot_box_next,
        "width": W, "height": H, "n_frames_total": seq_len,
    }


# --------------------------------------------------------------------------- #
# zip handling
# --------------------------------------------------------------------------- #
def find_sequence_roots(zf: zipfile.ZipFile,
                        seq_names: Sequence[str]) -> Dict[str, str]:
    """Map sequence name -> zip member prefix (folder) inside MOT17.zip."""
    names = zf.namelist()
    roots: Dict[str, str] = {}
    for seq in seq_names:
        hits = [n for n in names if f"/{seq}/" in n or n.startswith(f"{seq}/")]
        if not hits:
            continue
        # prefix up to and including the sequence folder
        first = hits[0]
        if n_has := first.startswith(f"{seq}/"):
            roots[seq] = f"{seq}/"
        else:
            prefix = first.split(f"/{seq}/")[0]
            roots[seq] = f"{prefix}/{seq}/"
    return roots


def list_sequence_files(zf: zipfile.ZipFile, roots: Dict[str, str],
                        keep_subdirs=("img1/", "gt/", "det/")) -> Dict[str, List[str]]:
    """For each sequence, the members we need (images, gt, det)."""
    names = zf.namelist()
    out: Dict[str, List[str]] = {}
    for seq, root in roots.items():
        members = []
        for n in names:
            if n.endswith("/"):          # directory entry -- not a file to write
                continue
            if not n.startswith(root):
                continue
            rest = n[len(root):]
            if rest.startswith("seqinfo.ini") or any(rest.startswith(k) for k in keep_subdirs):
                members.append(n)
        out[seq] = members
    return out


def extract_sequences(zip_path, seq_names: Sequence[str], dest_root) -> Dict[str, Path]:
    """Extract only the needed MOT17 sequences (img1/gt/det/seqinfo.ini)."""
    dest_root = Path(dest_root)
    out: Dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        roots = find_sequence_roots(zf, seq_names)
        missing = [s for s in seq_names if s not in roots]
        if missing:
            raise FileNotFoundError(
                f"{missing} not found in {zip_path}; roots found for "
                f"{list(roots)}. First 12 members: {zf.namelist()[:12]}")
        files = list_sequence_files(zf, roots)
        for seq in seq_names:
            root = roots[seq]
            tgt_dir = dest_root / seq
            n_new = 0
            for member in files[seq]:
                rel = member[len(root):]
                outp = tgt_dir / rel
                if outp.exists() and outp.stat().st_size > 0:
                    continue
                outp.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(outp, "wb") as dst:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        dst.write(b)
                n_new += 1
            out[seq] = tgt_dir
            print(f"  [mot17] {seq}: {len(files[seq])} members, {n_new} extracted")
    return out
