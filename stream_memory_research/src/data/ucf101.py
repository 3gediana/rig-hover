"""UCF101 specifics: official fold parsing + deterministic subsampling plan.

Official protocol (used here, so the numbers are comparable to the literature):
    ``ucfTrainTestlist/trainlist01.txt``  9537 clips
    ``ucfTrainTestlist/testlist01.txt``   3783 clips
    ``ucfTrainTestlist/classInd.txt``     101 classes

We need a *three*-way split (train/val/test), and the val part must not leak.
The official protocol only defines train/test, so val is carved **out of the
official train pool** by video (never by frame), leaving the official test set
untouched.  Everything is deterministic: no RNG is involved anywhere.

Selection rule (recorded verbatim in the manifest ``extra``)
-----------------------------------------------------------
1. For each class, sort the candidate clip ids lexicographically (this also
   sorts by subject ``gXX`` then clip ``cXX`` because the ids have the shape
   ``v_<Class>_g01_c01``).
2. Draw ``n`` clips from a sorted pool of ``N`` with an even stride:
   ``idx_i = floor(i * N / n)`` for ``i in 0..n-1``.  A stride spreads the
   sample over the whole id range (hence over subjects) instead of taking a
   prefix, and is trivially reproducible.
3. test  = stride_pick(official_test_pool[class], 12)
   trainval = stride_pick(official_train_pool[class], 50)
   val   = trainval[9::10]  (5 clips, spread across the id range)
   train = trainval minus val (45 clips)
=> 101 * 45 = 4545 train, 101 * 5 = 505 val, 101 * 12 = 1212 test  (6262 total)

Subject leakage note: UCF101 subjects appear in both the official train and
test lists by construction; the *video* is the split unit (``source_id``), so no
clip is ever seen in two splits.  If the Lead wants a subject-disjoint
analysis, per-sequence ``meta.subject`` is stored for regrouping.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CLASS_RE = re.compile(r"v_([A-Za-z]+)_g(\d+)_c(\d+)")
SUBJ_RE = re.compile(r"_g(\d+)_")
CLIP_RE = re.compile(r"_c(\d+)\.")

# --- subsampling quotas per class (see rule above) -------------------------
N_TRAIN_PER_CLASS = 45
N_VAL_PER_CLASS = 5
N_TEST_PER_CLASS = 12
VAL_STRIDE = 10          # val = trainval[VAL_STRIDE-1::VAL_STRIDE]
VAL_OFFSET = 9


def stride_pick(pool: Sequence[str], n: int) -> List[str]:
    """Deterministically pick ``n`` items from a sorted pool with even stride."""
    pool = list(pool)
    N = len(pool)
    if N == 0 or n <= 0:
        return []
    if n >= N:
        return pool
    idx = [int(i * N / n) for i in range(n)]
    # dedupe defensively (floor division can only repeat if n > N, excluded)
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(pool[i])
    return out


def read_txt_from_zip(zip_path, member: str) -> List[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        cand = [n for n in names if n.endswith("/" + member) or n == member]
        if not cand:
            raise FileNotFoundError(f"{member} not in {zip_path}")
        return z.read(cand[0]).decode("utf-8", "replace").splitlines()


def load_official_split(split_zip, fold: int = 1) -> Tuple[Dict[str, str], List[str]]:
    """Return ({clip_relpath: 'train'|'test'}, ordered class names)."""
    train_lines = read_txt_from_zip(split_zip, f"trainlist{fold:02d}.txt")
    test_lines = read_txt_from_zip(split_zip, f"testlist{fold:02d}.txt")
    train = [ln.split()[0].strip() for ln in train_lines if ln.strip()]
    test = [ln.strip() for ln in test_lines if ln.strip()]
    owner: Dict[str, str] = {}
    for p in train:
        owner[p] = "train"
    for p in test:
        owner[p] = "test"
    try:
        class_lines = read_txt_from_zip(split_zip, "classInd.txt")
        classes = [ln.split()[1].strip() for ln in class_lines if ln.strip()]
    except Exception:  # noqa: BLE001
        classes = sorted({p.split("/")[0] for p in owner})
    return owner, classes


def build_plan(owner: Dict[str, str], classes: Sequence[str],
               n_train: int = N_TRAIN_PER_CLASS,
               n_val: int = N_VAL_PER_CLASS,
               n_test: int = N_TEST_PER_CLASS) -> Tuple[List[dict], List[str]]:
    """Build the deterministic (clip -> split) plan.  Returns (plan, skipped)."""
    by_class: Dict[str, Dict[str, List[str]]] = {}
    for rel, split in owner.items():
        cls = rel.split("/")[0]
        clip = Path(rel).stem
        by_class.setdefault(cls, {"train": [], "test": []})[split].append(clip)
    for cls in by_class:
        for k in by_class[cls]:
            by_class[cls][k].sort()

    plan: List[dict] = []
    skipped: List[str] = []
    for ci, cls in enumerate(classes):
        d = by_class.get(cls)
        if d is None:
            skipped.append(cls)
            continue
        test_sel = stride_pick(d["test"], n_test)
        pool = stride_pick(d["train"], n_train + n_val)
        val_sel = set(pool[VAL_OFFSET::VAL_STRIDE])
        train_sel = [v for v in pool if v not in val_sel]
        for clip in train_sel:
            plan.append(_rec(clip, cls, ci, "train"))
        for clip in sorted(val_sel):
            plan.append(_rec(clip, cls, ci, "val"))
        for clip in test_sel:
            plan.append(_rec(clip, cls, ci, "test"))
    plan.sort(key=lambda r: (r["split"], r["class_index"], r["seq_id"]))
    return plan, skipped


def _rec(clip: str, cls: str, ci: int, split: str) -> dict:
    m_sub = SUBJ_RE.search(clip)
    m_clip = CLIP_RE.search(clip)
    return {
        "seq_id": f"{cls}__{clip}",
        "source_id": clip,
        "split": split,
        "cls_name": cls,
        "class_index": int(ci),
        "relpath": f"{cls}/{clip}.avi",
        "subject": f"g{m_sub.group(1)}" if m_sub else "",
        "clip_tag": f"c{m_clip.group(1)}" if m_clip else "",
    }


def extract_entries(zip_path, entries: Sequence[str], dest_root,
                    log_every: int = 500) -> int:
    """Extract only the selected zip members to ``dest_root`` (idempotent)."""
    dest_root = Path(dest_root)
    n_new = 0
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for i, rel in enumerate(entries):
            # the archive stores files under UCF-101/<Class>/<clip>.avi
            cand = [m for m in (rel, f"UCF-101/{rel}") if m in names]
            if not cand:
                continue
            member = cand[0]
            out = dest_root / member
            if out.exists() and out.stat().st_size > 0:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out, "wb") as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    dst.write(b)
            n_new += 1
            if log_every and n_new % log_every == 0:
                print(f"  [unzip] {n_new} files extracted (last {member})", flush=True)
    return n_new
