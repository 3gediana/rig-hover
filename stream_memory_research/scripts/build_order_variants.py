"""Build arrow-of-time (temporal-direction) tasks at ARBITRARY temporal extent
from already-cached features -- no GPU, no video decode.

Why this exists
---------------
`ucf101_order` (T=48, span 3.24 s, every 2nd clip) is the only dimension where
the model significantly beats every baseline.  The mechanism analysis says the
advantage comes from a *bank of decay timescales* acting as a band-pass filter
over time, so the natural amplification knob is **how much chronological span the
decision has to integrate over**.  Before spending GPU on new video we can test
that knob on data we already paid for:

  ucf101_action  cached at T=96, span 6.52 s, ALL 6260 clips   -> unused
  charades_event cached at T=128, span 30.63 s, 1400 clips     -> unused for order
  ucf101_order   cached at T=48, span 3.24 s, 3130 clips       -> already reported

So `--src ucf101_action --dst ucf101_order96` gives 12 520 sequences (2424 test
pairs, vs 1212 today) at 2x span, at zero extraction cost.

Construction is IDENTICAL to build_ucf_order_task.py (do not diverge -- the
comparison is only meaningful if the task is the same):
  label 0 -> frames in original order
  label 1 -> frames reversed in time (timestamps still increase)
`tok` reversed the same way.  Both order variants of a clip stay in the SAME
split, so this cannot leak.

Usage
-----
  python scripts/build_order_variants.py --src ucf101_action --dst ucf101_order96 --nf 96
  python scripts/build_order_variants.py --src charades_event --dst charades_order128 --nf 128
  python scripts/build_order_variants.py --src ucf101_action --dst ucf101_order24 --nf 24 --no-tok
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract  # noqa: E402
from src.common.contract import Manifest, SeqRecord  # noqa: E402


def build(src: str, dst: str, nf: int, pick: str, keep_tok: bool) -> Path:
    man = Manifest.load(contract.MANIFESTS / f"{src}.json")
    out = contract.FEATURES / dst
    out.mkdir(parents=True, exist_ok=True)

    seqs = sorted(man.seqs, key=lambda r: r.seq_id)
    if pick == "even":                      # same rule as ucf101_order: every 2nd clip
        seqs = seqs[::2]
    elif pick != "all":
        raise SystemExit(f"bad --pick {pick!r}")

    n_ok = n_short = 0
    t0 = time.time()
    recs: list[SeqRecord] = []
    for i, r in enumerate(seqs):
        with np.load(r.path, allow_pickle=False) as z:
            feat = np.asarray(z["feat"])[:nf]
            tok = (np.asarray(z["tok"])[:nf]
                   if (keep_tok and "tok" in z.files) else None)
            t = np.asarray(z["t"], dtype=np.float32)[:nf]
        if feat.shape[0] < nf:              # clip shorter than requested -> skip clip
            n_short += 1
            continue
        n_ok += 1
        for rev in (0, 1):
            tag = "rev" if rev else "fwd"
            seq_id = f"{r.seq_id}__{tag}"
            p = out / f"{seq_id}.npz"
            arrays = {
                "feat": np.ascontiguousarray(feat[::-1] if rev else feat),
                "t": t.copy(),                       # timestamps still increase
                "lab_order": np.full(t.shape[0], rev, dtype=np.int64),
            }
            if tok is not None:
                arrays["tok"] = np.ascontiguousarray(tok[::-1] if rev else tok)
            np.savez(p, **arrays)                    # uncompressed: load speed matters
            recs.append(SeqRecord(
                seq_id=seq_id, source_id=r.source_id, split=r.split,
                T=int(t.shape[0]), path=str(p), labels=["order"],
                meta={**r.meta, "order_label": rev, "origin": r.seq_id},
            ))
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(seqs)} clips  {el:.0f}s  "
                  f"({(i + 1) / max(el, 1e-6):.1f} clips/s)", flush=True)

    newman = Manifest(
        dataset=dst, task="order", feature_dim=int(man.feature_dim),
        tok_dim=int(man.tok_dim) if keep_tok else 0, seqs=recs, encoder=man.encoder,
        label_specs={"order": {"kind": "ce", "ignore": -1, "dim": 2, "n_classes": 2}},
        extra={
            "note": ("arrow-of-time task at a larger temporal extent: two sequences per "
                     "clip, original vs reversed frame order. Same frames, same split, "
                     "same source_id -> a model that ignores temporal order is at chance "
                     "by construction."),
            "classes": ["forward", "reversed"],
            "source_dataset": src,
            "construction": (f"feat{'/tok' if keep_tok else ''} reversed in time; t unchanged "
                             f"(increasing); {nf} frames; pick={pick}"),
            "temporal_span_s": round(float(np.load(recs[0].path)["t"][-1]
                                           - np.load(recs[0].path)["t"][0]), 3),
            "main_metric": "acc",
            "eval_manifests": [],
        },
    )
    p = newman.save(contract.MANIFESTS / f"{dst}.json")
    newman.check_split_disjoint()
    print(f"wrote {p}")
    print("counts:", newman.counts(), "n_seqs:", len(recs),
          f"clips_ok={n_ok} clips_short={n_short}")
    print("span_s:", newman.extra["temporal_span_s"])
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--nf", type=int, required=True)
    ap.add_argument("--pick", default="all", choices=["all", "even"])
    ap.add_argument("--no-tok", action="store_true")
    a = ap.parse_args()
    build(a.src, a.dst, a.nf, a.pick, not a.no_tok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
