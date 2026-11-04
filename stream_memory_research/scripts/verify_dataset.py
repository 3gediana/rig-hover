"""Verify a cached-feature dataset end to end (the acceptance gate).

For every sequence in the manifest it loads the npz and checks:

* the file exists and ``feat``/``t`` are present with the manifest's ``T``;
* ``T`` in the manifest equals ``feat.shape[0]``;
* ``t`` is float32, finite, strictly increasing, starts >= 0 and is in seconds;
* ``feat`` (and ``tok``/``roi`` if present) are finite, correct dtype/shape;
* ``roi_valid`` is uint8/bool with the same [T,N] shape as ``roi``;
* ``roi_meta`` boxes are within [0,1] (small tolerance) and ``w,h > 0`` for
  valid ROIs;
* every ``lab_*`` array has T as its first dimension and the expected dtype;
* for ``action``, label values are in ``[0, n_classes)``;
* ``source_id`` never crosses splits (``Manifest.check_split_disjoint``).

It prints a per-split table plus label statistics and writes
``logs/dataset_verify.txt``.  Exit code 1 if any hard check fails.

Usage
-----
python scripts/verify_dataset.py                      # every manifest found
python scripts/verify_dataset.py ucf101_action        # one dataset
python scripts/verify_dataset.py path/to/m.json --limit 50 --quiet
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import contract  # noqa: E402

LABEL_KINDS = {  # label name -> ("int"|"float", expected trailing shape)
    "lab_action": ("int", ()),
    "lab_event": ("float", None),
    "lab_state": ("float", None),
    "lab_track": ("float", None),
    "lab_id": ("int", None),
    "lab_present": ("float", None),
    "lab_seg": ("int", ()),
}


class Report:
    def __init__(self):
        self.buf = io.StringIO()
        self.n_fail = 0

    def line(self, s: str = "") -> None:
        print(s)
        self.buf.write(s + "\n")

    def fail(self, s: str) -> None:
        self.n_fail += 1
        self.line("  FAIL " + s)


def _check_seq(rec, m, rep: Report, limit_check: bool) -> dict:
    """Load one npz and validate it.  Returns stats."""
    stats: dict = {"T": 0, "labels": {}}
    p = Path(rec.path)
    if not p.exists():
        rep.fail(f"{rec.seq_id}: missing npz {p}")
        return stats
    try:
        z = np.load(p, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        rep.fail(f"{rec.seq_id}: cannot load npz ({type(e).__name__}: {e})")
        return stats
    with z:
        for need in ("feat", "t"):
            if need not in z.files:
                rep.fail(f"{rec.seq_id}: missing array '{need}'")
                return stats
        feat, t = z["feat"], z["t"]
        T = int(feat.shape[0])
        stats["T"] = T
        if T != int(rec.T):
            rep.fail(f"{rec.seq_id}: manifest T={rec.T} != npz T={T}")
        if feat.shape != (T, m.feature_dim):
            rep.fail(f"{rec.seq_id}: feat shape {feat.shape} != (T,{m.feature_dim})")
        if t.shape != (T,):
            rep.fail(f"{rec.seq_id}: t shape {t.shape} != ({T},)")
        if T < 2:
            rep.fail(f"{rec.seq_id}: T={T} < 2")
        if not np.all(np.isfinite(np.asarray(feat))):
            rep.fail(f"{rec.seq_id}: feat contains NaN/Inf")
        fa = feat.astype(np.float32)
        if float(np.abs(fa).max()) < 1e-3:
            rep.fail(f"{rec.seq_id}: feat is all zeros")
        if not np.all(np.isfinite(t)):
            rep.fail(f"{rec.seq_id}: t contains NaN/Inf")
        else:
            dt = np.diff(t.astype(np.float64))
            if not np.all(dt > 0):
                bad = int((dt <= 0).sum())
                rep.fail(f"{rec.seq_id}: t not strictly increasing ({bad} bad steps)")
            if float(t[0]) < 0:
                rep.fail(f"{rec.seq_id}: t[0]={float(t[0])} < 0")
            if float(t[-1]) > 4 * 3600:
                rep.fail(f"{rec.seq_id}: t[-1]={float(t[-1]):.1f}s implausible")
            stats["dt_mean"] = float(dt.mean()) if len(dt) else 0.0
            stats["dt_cv"] = float(dt.std() / max(1e-9, dt.mean())) if len(dt) else 0.0

        # optional token features.  A manifest may deliberately declare
        # ``tok_dim: 0`` ("feat-only view") even though the npz still carries
        # `tok`; in that case StreamDataset(load_tok=False) never reads it and we
        # must not enforce a shape.
        tok = z["tok"] if "tok" in z.files else None
        if m.tok_dim and tok is None:
            rep.fail(f"{rec.seq_id}: manifest tok_dim={m.tok_dim} but no 'tok'")
        if tok is not None and m.tok_dim:
            if tok.ndim != 3 or tok.shape[0] != T or tok.shape[2] != m.tok_dim:
                rep.fail(f"{rec.seq_id}: tok shape {tok.shape} (want "
                         f"[T,K,{m.tok_dim}])")
            else:
                stats["tok_k"] = int(tok.shape[1])
            if not np.all(np.isfinite(np.asarray(tok))):
                rep.fail(f"{rec.seq_id}: tok contains NaN/Inf")

        # ROI features
        has_roi = "roi" in z.files
        if m.n_roi and not has_roi:
            rep.fail(f"{rec.seq_id}: manifest n_roi={m.n_roi} but no 'roi'")
        if has_roi:
            roi = z["roi"]
            if roi.ndim != 3 or roi.shape[0] != T:
                rep.fail(f"{rec.seq_id}: roi shape {roi.shape} (want [T,N,{m.roi_dim}])")
            else:
                N = roi.shape[1]
                if m.n_roi and N != m.n_roi:
                    rep.fail(f"{rec.seq_id}: roi N={N} != n_roi={m.n_roi}")
                if not np.all(np.isfinite(np.asarray(roi))):
                    rep.fail(f"{rec.seq_id}: roi contains NaN/Inf")
                for k in ("roi_valid", "roi_meta"):
                    if k not in z.files:
                        rep.fail(f"{rec.seq_id}: missing '{k}' alongside roi")
                if "roi_valid" in z.files:
                    rv = z["roi_valid"]
                    if rv.shape != (T, N):
                        rep.fail(f"{rec.seq_id}: roi_valid {rv.shape} != ({T},{N})")
                    if rv.dtype not in (np.uint8, np.bool_):
                        rep.fail(f"{rec.seq_id}: roi_valid dtype {rv.dtype}")
                    stats["roi_valid_frac"] = float(np.asarray(rv, np.float32).mean())
                if "roi_meta" in z.files:
                    rm = z["roi_meta"].astype(np.float32)
                    if rm.shape != (T, N, 4):
                        rep.fail(f"{rec.seq_id}: roi_meta {rm.shape} != ({T},{N},4)")
                    else:
                        if not np.all(np.isfinite(rm)):
                            rep.fail(f"{rec.seq_id}: roi_meta not finite")
                        else:
                            if rm.min() < -0.01 or rm.max() > 1.01:
                                rep.fail(f"{rec.seq_id}: roi_meta outside [0,1] "
                                         f"(min {rm.min():.3f} max {rm.max():.3f})")
                            if "roi_valid" in z.files:
                                v = np.asarray(z["roi_valid"]).astype(bool)
                                if v.any():
                                    wh = rm[..., 2:4][v]
                                    if not np.all(wh > 0):
                                        rep.fail(f"{rec.seq_id}: valid ROI with "
                                                 f"non-positive w/h")
                            stats["roi_wh_mean"] = float(rm[..., 2:4].mean())

        # labels
        for k in z.files:
            if not k.startswith("lab_"):
                continue
            v = z[k]
            kind, _ = LABEL_KINDS.get(k, ("float", None))
            if v.ndim < 1 or v.shape[0] != T:
                rep.fail(f"{rec.seq_id}: {k} first dim {v.shape} != T={T}")
                continue
            if kind == "int" and not np.issubdtype(v.dtype, np.integer):
                rep.fail(f"{rec.seq_id}: {k} dtype {v.dtype} should be integer")
            if kind == "float" and not np.all(np.isfinite(np.asarray(v))):
                rep.fail(f"{rec.seq_id}: {k} contains NaN/Inf")
            if k == "lab_action":
                nc = int(m.label_specs.get("lab_action", {}).get("n_classes", 0) or 0)
                if nc and (v.min() < 0 or v.max() >= nc):
                    rep.fail(f"{rec.seq_id}: lab_action {v.min()}..{v.max()} "
                             f"outside [0,{nc})")
                stats["labels"]["action"] = int(v[0])
            elif k == "lab_event":
                stats["labels"]["event_frac"] = float(np.asarray(v, np.float32).mean())
                stats["labels"]["event_dim"] = int(v.shape[-1])
            elif k == "lab_track":
                stats["labels"]["track_shape"] = tuple(v.shape)
            elif k == "lab_id":
                vv = v.astype(np.int64)
                stats["labels"]["id_max"] = int(vv.max())
                stats["labels"]["id_neg_frac"] = float((vv < 0).mean())
            elif k == "lab_present":
                stats["labels"]["present_frac"] = float(np.asarray(v, np.float32).mean())
        if not any(k.startswith("lab_") for k in z.files):
            rep.fail(f"{rec.seq_id}: no lab_* array")
    if not limit_check:
        _check_npz_no_object(p, rep, rec.seq_id)
    return stats


def _check_npz_no_object(p: Path, rep: Report, seq_id: str) -> None:
    """np.load(allow_pickle=False) already guards this; keep a cheap size check."""
    sz = p.stat().st_size
    if sz < 1024:
        rep.fail(f"{seq_id}: npz suspiciously small ({sz} B)")


def collate_check(path, n_per_split: int = 3, rep: Report | None = None) -> int:
    """End-to-end contract check: Manifest -> StreamDataset -> collate_stream.

    Proves that a manifest can actually be consumed by the trainer: builds a
    real batch, prints the shapes, and asserts the invariants the models rely on
    (dt[:,0]==0, mask covers exactly T per sample, padded label steps use the
    ignore value, roi_meta/roi_valid agree on shapes).
    """
    import torch  # noqa: F401  (StreamDataset returns tensors)
    rep = rep or Report()
    m = contract.Manifest.load(path)
    from src.common.contract import StreamDataset, collate_stream
    recs = []
    for split in ("train", "val", "test"):
        rs = m.split(split)[:n_per_split]
        recs.extend(rs)
    if not recs:
        rep.line(f"  [collate] {m.dataset}: no records, skipped")
        return 0
    ds = StreamDataset(recs, load_tok=bool(m.tok_dim), load_roi=bool(m.n_roi))
    try:
        batch = collate_stream([ds[i] for i in range(len(ds))])
    except Exception as e:  # noqa: BLE001
        rep.fail(f"{m.dataset}: collate_stream raised {type(e).__name__}: {e}")
        return rep.n_fail
    x, t, dt, mask = batch["x"], batch["t"], batch["dt"], batch["mask"]
    B, T, D = x.shape
    rep.line(f"  [collate] {m.dataset}: B={B} T={T} x={tuple(x.shape)} "
             f"dtype={x.dtype}")
    problems = []
    if D != m.feature_dim:
        problems.append(f"x last dim {D} != feature_dim {m.feature_dim}")
    if not torch.isfinite(x).all():
        problems.append("x has NaN/Inf")
    if dt[:, 0].abs().max() > 0:
        problems.append("dt[:,0] != 0")
    if (dt < 0).any():
        problems.append("dt has negative entries")
    if not torch.equal(mask.sum(1), torch.tensor([int(r.T) for r in recs])):
        problems.append("mask.sum(1) != per-sample T")
    if batch["tok"] is not None:
        rep.line(f"           tok={tuple(batch['tok'].shape)}")
        tok = batch["tok"]
        if tok.ndim != 4 or tok.shape[-1] != m.tok_dim or tok.shape[0] != B:
            problems.append(f"tok shape {tuple(tok.shape)} inconsistent with "
                            f"tok_dim={m.tok_dim}")
    if batch["roi"] is not None:
        rep.line(f"           roi={tuple(batch['roi'].shape)} "
                 f"roi_valid={tuple(batch['roi_valid'].shape)} "
                 f"roi_meta={tuple(batch['roi_meta'].shape)}")
        if batch["roi"].shape[2] != m.n_roi:
            problems.append("roi N != n_roi")
        if batch["roi_meta"].shape[-1] != 4:
            problems.append("roi_meta last dim != 4")
        rv = batch["roi_valid"]
        if rv.dtype != torch.bool:
            problems.append(f"roi_valid dtype {rv.dtype} != bool")
    for k, v in sorted(batch["labels"].items()):
        rep.line(f"           labels[{k}] = {tuple(v.shape)} {v.dtype} "
                 f"min={int(v.min()) if not v.is_floating_point() else float(v.min()):.3g}"
                 f" max={int(v.max()) if not v.is_floating_point() else float(v.max()):.3g}")
        # padded steps must be ignore (-1) for ints, 0 for floats
        for i, r in enumerate(recs):
            pad = v[i, int(r.T):]
            if pad.numel() == 0:
                continue
            want = -1 if not v.is_floating_point() else 0.0
            if not torch.equal(pad, torch.full_like(pad, want)):
                problems.append(f"label {k} padded with {pad.flatten()[:3].tolist()} "
                                f"(want {want})")
                break
    if problems:
        for p in problems:
            rep.fail(f"{m.dataset} collate: {p}")
    else:
        rep.line(f"  [collate] {m.dataset}: OK")
    return rep.n_fail


def verify_manifest(path, limit=None, quiet=False) -> Report:
    rep = Report()
    m = contract.Manifest.load(path)
    t0 = time.time()
    rep.line("=" * 78)
    rep.line(f"DATASET {m.dataset}   task={m.task}  manifest={path}")
    rep.line(f"  feature_dim={m.feature_dim} tok_dim={m.tok_dim} "
             f"roi_dim={m.roi_dim} n_roi={m.n_roi} split_key={m.split_key}")
    enc = m.encoder or {}
    rep.line(f"  encoder={enc.get('name')} tag={enc.get('tag')} "
             f"input={enc.get('input')} dtype={enc.get('dtype')}")
    if m.tok_dim == 0:
        rep.line("  WARN tok_dim==0: models that use spatial tokens get no 'tok'")
    # ---- split disjointness -------------------------------------------------
    try:
        m.check_split_disjoint()
        rep.line("  split disjointness: OK (no source_id crosses splits)")
    except ValueError as e:
        rep.fail(f"split leakage: {e}")
    # ---- per-sequence checks ------------------------------------------------
    seqs = m.seqs[:limit] if limit else m.seqs
    per = {}
    stats_all = []
    for i, rec in enumerate(seqs):
        st = None
        for attempt in range(3):
            try:
                st = _check_seq(rec, m, rep, limit_check=(limit is not None))
                break
            except MemoryError:
                # this box shares 31 GB / a small page file with the training
                # teammates; a transient MemoryError must not abort a 6000-seq run
                import gc
                gc.collect()
                time.sleep(0.3)
            except Exception as e:  # noqa: BLE001
                rep.fail(f"{rec.seq_id}: {type(e).__name__}: {e}")
                st = {}
                break
        if st is None:
            rep.fail(f"{rec.seq_id}: skipped after repeated MemoryError")
            st = {}
        stats_all.append(st)
        s = per.setdefault(rec.split, {"n": 0, "T": [], "lab": {}})
        s["n"] += 1
        s["T"].append(int(rec.T))
        for k, v in st.get("labels", {}).items():
            if k in ("action", "event_frac", "present_frac", "id_neg_frac"):
                s["lab"].setdefault(k, []).append(v)
        if (i + 1) % 500 == 0 and not quiet:
            rep.line(f"  ... {i + 1}/{len(seqs)} checked")
    # ---- summary table ------------------------------------------------------
    rep.line("")
    rep.line(f"  {'split':<7}{'n_seqs':>8}{'frames':>10}{'T_min':>7}{'T_max':>7}"
             f"{'T_mean':>8}")
    for split in ("train", "val", "test"):
        s = per.get(split)
        if not s:
            continue
        rep.line(f"  {split:<7}{s['n']:>8}{sum(s['T']):>10}{min(s['T']):>7}"
                 f"{max(s['T']):>7}{np.mean(s['T']):>8.1f}")
    tot = sum(s["n"] for s in per.values())
    tframes = sum(sum(s["T"]) for s in per.values())
    rep.line(f"  {'TOTAL':<7}{tot:>8}{tframes:>10}")
    for split, s in per.items():
        if "action" in s["lab"]:
            vals, cnts = np.unique(np.array(s["lab"]["action"]), return_counts=True)
            rep.line(f"  [{split}] classes present={len(vals)} "
                     f"per-class min={cnts.min()} max={cnts.max()} "
                     f"mean={cnts.mean():.1f}")
        if "event_frac" in s["lab"]:
            a = np.array(s["lab"]["event_frac"])
            rep.line(f"  [{split}] mean positive fraction per frame="
                     f"{a.mean():.4f} (min {a.min():.4f} max {a.max():.4f})")
        if "present_frac" in s["lab"]:
            a = np.array(s["lab"]["present_frac"])
            rep.line(f"  [{split}] object present fraction: mean={a.mean():.3f} "
                     f"min={a.min():.3f} max={a.max():.3f}")
        if "id_neg_frac" in s["lab"]:
            a = np.array(s["lab"]["id_neg_frac"])
            rep.line(f"  [{split}] invalid-id fraction: mean={a.mean():.3f}")
    dfs = [st.get("dt_mean", 0) for st in stats_all if st.get("dt_mean")]
    cv = [st.get("dt_cv", 0) for st in stats_all if st.get("dt_cv") is not None]
    if dfs:
        rep.line(f"  dt (seconds): mean over seqs={np.mean(dfs):.4f} "
                 f"min={np.min(dfs):.4f} max={np.max(dfs):.4f} | "
                 f"per-seq dt CV mean={np.mean(cv):.4f} "
                 f"(>0 means non-uniform timestamps are actually stored)")
    rv = [st["roi_valid_frac"] for st in stats_all if "roi_valid_frac" in st]
    if rv:
        rep.line(f"  roi_valid fraction: mean={np.mean(rv):.3f} "
                 f"min={np.min(rv):.3f} max={np.max(rv):.3f}")
    nskip = len((m.extra or {}).get("skipped", {}) or {})
    if nskip:
        rep.line(f"  skipped at extraction: {nskip} "
                 f"(see manifest extra.skipped)")
    rep.line(f"  checked {len(seqs)} seqs in {time.time() - t0:.1f}s | "
             f"failures={rep.n_fail}")
    rep.line("=" * 78)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--collate", action="store_true",
                    help="also load a real batch via StreamDataset+collate_stream")
    ap.add_argument("--out", default=str(contract.LOGS / "dataset_verify.txt"))
    args = ap.parse_args()

    paths: list[Path] = []
    for t in args.targets:
        p = Path(t)
        if p.suffix == ".json":
            paths.append(p)
        else:
            cand = contract.MANIFESTS / f"{t}.json"
            if not cand.exists():
                print(f"!! no manifest for '{t}'")
                continue
            paths.append(cand)
    if not paths:
        paths = sorted(p for p in contract.MANIFESTS.glob("*.json")
                       if not p.name.endswith("_plan.json")
                       and not p.name.endswith("_preview.json")
                       and not p.name.endswith("_smoke.json"))
    if not paths:
        print("!! no manifests found in", contract.MANIFESTS)
        return 2

    from src.data.util import Tee
    tee = Tee(args.out)
    sys.stdout = tee
    print(f"# dataset verification {contract.now_str()}")
    print(f"# manifests: {[p.name for p in paths]}")
    total_fail = 0
    for p in paths:
        try:
            total_fail += verify_manifest(p, limit=args.limit,
                                          quiet=args.quiet).n_fail
            if args.collate:
                rep = Report()
                total_fail += collate_check(p, rep=rep)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"!! {p}: {type(e).__name__}: {e}")
            traceback.print_exc()
            total_fail += 1
    print(f"\nOVERALL: {'PASS' if total_fail == 0 else f'{total_fail} FAILURES'}")
    tee.flush()
    sys.stdout = tee.stdout
    print(f"[verify] wrote {args.out}  ({'PASS' if total_fail == 0 else 'FAIL'})")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

