# -*- coding: utf-8 -*-
"""pc/pack/inventory.py - the asset inventory: walk asset_registry.GROUPS and write down what is really there.

Output (PACK_ROOT/canon/):
  assets.json   machine-readable: per group path, files, bytes, frames, extension mix,
                license, privacy, split, plus per-take / per-session detail and exact duplicates
  assets.md     the same as tables, for BASELINE.md and the data cards

Nothing is moved or modified. Groups marked hash=True get a sha256 for every
file, which is also what finds exact duplicates across takes and sessions.

Usage:
    python pc/pack/inventory.py            # full (hashes ~1.3 GiB, a few minutes)
    python pc/pack/inventory.py --no-hash  # counts only
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import glob
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from asset_registry import GROUPS, Group  # noqa: E402
import paths  # noqa: E402

ABS = re.compile(r"^[A-Za-z]:[\\/]")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def resolve(p: str) -> Path:
    return Path(p) if ABS.match(p) else paths.REPO / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(root: Path) -> list:
    return [root] if root.is_file() else [Path(dp) / f for dp, _dn, fn in os.walk(root) for f in fn]


def cached_sha256(f: Path, size: int, hcache: dict):
    """sha256 reused while size and mtime are unchanged (the HaGRID sample alone takes 23 min to hash)."""
    try:
        mt = f.stat().st_mtime_ns
    except OSError:
        return None
    c = hcache.get(str(f))
    if c and c[0] == size and c[1] == mt:
        return c[2]
    try:
        d = sha256_file(f)
    except OSError:
        return None
    hcache[str(f)] = [size, mt, d]
    return d


def walk_group(g: Group, do_hash: bool, hashes: dict, hcache: dict) -> None:
    g.missing_paths, g.ext, g.sub = [], collections.Counter(), []
    g.exists = g.n_files = g.bytes = g.n_frames = 0
    roots = []
    for p in g.paths:
        rp = str(resolve(p))
        hits = sorted(glob.glob(rp)) if any(ch in rp for ch in "*?[") else ([rp] if os.path.exists(rp) else [])
        if not hits:
            g.missing_paths.append(p)
        roots += [Path(h) for h in hits]
    units = []                                   # (name, path, files)
    for root in roots:
        if g.subdirs and root.is_dir():
            loose = sorted(f for f in root.iterdir() if f.is_file())
            if loose:
                units.append(("_root", root, loose))
            for d in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda q: q.name):
                units.append((d.name, d, list_files(d)))
        else:
            units.append((root.name, root, list_files(root)))
    for name, root, files in units:
        sub = {"name": name, "path": str(root), "n_files": 0, "bytes": 0, "n_frames": 0, "detail": {}}
        for f in files:
            try:
                sz = f.stat().st_size
            except OSError:
                continue
            ext = f.suffix.lower() or "(none)"
            g.n_files += 1
            g.bytes += sz
            sub["n_files"] += 1
            sub["bytes"] += sz
            g.ext[ext] += 1
            is_frame = bool(g.frames_glob) and (
                fnmatch.fnmatch(f.name, os.path.basename(g.frames_glob)) or
                (g.frames_glob.endswith(".jpg") and ext == ".jpg"))
            if is_frame:
                g.n_frames += 1
                sub["n_frames"] += 1
            if do_hash and g.hash and ext not in (".pyc",):
                digest = cached_sha256(f, sz, hcache)
                if digest:
                    hashes.setdefault(digest, []).append(str(f))
        g.sub.append(sub)
    g.exists = bool(roots)
    g.ext = dict(g.ext.most_common())


# --------------------------------------------------------------------------- detail extractors
def take_detail(source_dir: Path) -> dict:
    d = {}
    meta = source_dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        d.update({k: m.get(k) for k in ("fps", "capture_fps", "frames", "video_frames", "duration_s",
                                         "started_at", "finished_at", "codec")})
    fj = source_dir / "frames.jsonl"
    if fj.exists():
        ts = []
        n = 0
        with open(fj, encoding="utf-8") as fh:
            for line in fh:
                n += 1
                try:
                    ts.append(float(json.loads(line)["t_s"]))
                except Exception:
                    pass
        d["frames_jsonl_rows"] = n
        if len(ts) > 2:
            diffs = [b - a for a, b in zip(ts, ts[1:]) if b > a]
            if diffs:
                d["capture_interval_ms"] = {"median": round(statistics.median(diffs) * 1000, 1),
                                            "p05": round(sorted(diffs)[int(0.05 * len(diffs))] * 1000, 1),
                                            "p95": round(sorted(diffs)[int(0.95 * len(diffs))] * 1000, 1)}
                d["capture_fps_median"] = round(1000.0 / (statistics.median(diffs) * 1000), 2)
    return d


def annotation_detail(take_id: str) -> dict:
    ap = paths.REPO / "dataset" / "takes" / take_id / "annotation.json"
    if not ap.exists():
        return {"annotated": False}
    a = json.loads(ap.read_text(encoding="utf-8"))
    labels = collections.Counter(x["label"] for x in a["annotations"])
    knock_s = sum(x["end_s"] - x["start_s"] for x in a["annotations"] if x["label"] == "knock")
    return {"annotated": True, "n_annotations": len(a["annotations"]), "labels": dict(labels),
            "knock_seconds": round(knock_s, 1), "duration_s": a.get("duration_s")}


def session_detail(session_dir: Path) -> dict:
    d = {}
    meta = session_dir / "meta.json"
    if meta.exists():
        d.update(json.loads(meta.read_text(encoding="utf-8")))
    ij = session_dir / "info.jsonl"
    if ij.exists():
        n = 0
        keys = None
        with open(ij, encoding="utf-8") as fh:
            for line in fh:
                n += 1
                if keys is None and line.strip():
                    try:
                        keys = list(json.loads(line).keys())
                    except Exception:
                        keys = None
        d["info_rows"] = n
        d["info_keys"] = keys
    return d


def enrich(g: Group) -> None:
    for sub in g.sub:
        name = sub["name"]
        root = Path(sub["path"])
        if name.endswith("_source"):
            take_id = name[:-len("_source")]
            sub["detail"] = {"take_id": take_id, **take_detail(root), **annotation_detail(take_id)}
            sub["detail"]["sealed"] = take_id in paths.SEALED_TAKE_IDS
        elif name.startswith("robot_2026111") and (root / "take.json").exists():
            tj = json.loads((root / "take.json").read_text(encoding="utf-8"))
            sub["detail"] = {"take_id": tj["take_id"], **{k: tj.get(k) for k in ("fps", "capture_fps", "raw_frames", "video_frames", "duration_s")},
                             **annotation_detail(tj["take_id"]), "sealed": tj["take_id"] in paths.SEALED_TAKE_IDS,
                             "raw_frames_on_disk": (paths.REPO / "video_annotation_data" / "videos" / (tj["take_id"] + "_source")).exists()}
        elif (root / "manifest.jsonl").exists() or (root / "meta.json").exists():
            sub["detail"] = session_detail(root) if (root / "meta.json").exists() else {}
            if (root / "frames").is_dir():
                sub["detail"]["n_frames"] = sub["n_frames"]
            sub["detail"]["sealed"] = name in paths.SEALED_SESSION_IDS
        elif (root / "info.jsonl").exists():
            sub["detail"] = session_detail(root)
        elif (root / "tel.csv").exists():
            rows = sum(1 for _ in open(root / "tel.csv", encoding="utf-8")) - 1
            sub["detail"] = {"telemetry_rows": rows}
            if (root / "summary.json").exists():
                try:
                    s = json.loads((root / "summary.json").read_text(encoding="utf-8"))
                    sub["detail"]["stage_or_tag"] = s.get("stage") or s.get("tag") or ""
                    sub["detail"]["abort"] = s.get("abort", "")
                except Exception:
                    pass
            if (root / "frames").is_dir():
                sub["detail"]["n_frames"] = sum(1 for _ in (root / "frames").glob("*.jpg"))


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GiB"


def render_md(inv: dict) -> str:
    L = ["# Asset inventory (generated by pc/pack/inventory.py)", "",
         f"Generated: {inv['generated']}  |  repository: `{inv['repo']}`  |  hashed: {'yes' if inv['hashed'] else 'no'}", "",
         "| id | what | kind | path | files | size | frames | source | license | privacy | split |",
         "|---|---|---|---|---:|---:|---:|---|---|---|---|"]
    for g in inv["groups"]:
        p = "; ".join(f"`{x}`" for x in g["paths"])
        if g["missing_paths"]:
            p += " **MISSING: " + "; ".join(g["missing_paths"]) + "**"
        L.append(f"| {g['id']} | {g['title']} | {g['kind']} | {p} | {g['n_files']} | {human(g['bytes'])} | "
                 f"{g['n_frames']} | {g['source']} | {g['license']} | {g['privacy']} | {g['split']} |")
    L += ["", "## Takes", "",
          "| take | raw frames on disk | jsonl rows | fps (output) | capture fps (median) | interval p05 / p95 ms | duration s | annotations | knock segments | knock s | labels | sealed |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|"]
    for g in inv["groups"]:
        for s in g["sub"]:
            d = s.get("detail") or {}
            if "take_id" not in d:
                continue
            ci = d.get("capture_interval_ms") or {}
            L.append(f"| {d['take_id']} | {'yes' if d.get('raw_frames_on_disk', True) else '**NO - MP4 only**'} | "
                     f"{d.get('frames_jsonl_rows', '-')} | {d.get('fps', '-')} | {d.get('capture_fps_median', d.get('capture_fps', '-'))} | "
                     f"{ci.get('p05', '-')} / {ci.get('p95', '-')} | {d.get('duration_s', '-')} | {d.get('n_annotations', 0)} | "
                     f"{(d.get('labels') or {}).get('knock', 0)} | {d.get('knock_seconds', 0)} | "
                     f"{sum(1 for k in (d.get('labels') or {}) if k.startswith('label_'))} custom | {'**SEALED**' if d.get('sealed') else ''} |")
    L += ["", "## Controlled sessions (evidence/sessions)", "",
          "| session | task | label | frames | distance m | light | started | sealed |", "|---|---|---|---:|---:|---|---|---|"]
    for g in inv["groups"]:
        for s in g["sub"]:
            d = s.get("detail") or {}
            if "task" in d:
                L.append(f"| {s['name']} | {d.get('task')} | {d.get('label')} | {d.get('frames', s['n_frames'])} | "
                         f"{d.get('distance_m')} | {d.get('light')} | {d.get('started_at')} | {'**SEALED**' if d.get('sealed') else ''} |")
    L += ["", "## Live sessions, probes and guarded sessions", "",
          "| directory | frames | info/telemetry rows | stage | abort | sealed |", "|---|---:|---:|---|---|---|"]
    for g in inv["groups"]:
        for s in g["sub"]:
            d = s.get("detail") or {}
            if "info_rows" in d or "telemetry_rows" in d:
                rows = d.get("info_rows", d.get("telemetry_rows"))
                L.append(f"| {s['name']} | {d.get('n_frames', s['n_frames'])} | {rows} | {d.get('stage_or_tag', '')} | "
                         f"{d.get('abort', '')} | {'**SEALED**' if d.get('sealed') else ''} |")
    dups = inv["duplicates"]
    L += ["", "## Exact duplicates (same sha256)", "",
          f"Groups of identical files: **{len(dups)}**; files involved: **{sum(len(d['files']) for d in dups)}**; "
          f"redundant bytes: {human(sum(d['bytes'] * (len(d['files']) - 1) for d in dups))}."]
    for d in dups[:40]:
        L.append(f"- `{d['sha256'][:16]}` x{len(d['files'])} ({human(d['bytes'])} each): " +
                 "; ".join(f"`{os.path.relpath(f, paths.REPO) if f.startswith(str(paths.REPO)) else f}`" for f in d["files"][:6]) +
                 (" ..." if len(d["files"]) > 6 else ""))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args()
    do_hash = not a.no_hash
    t0 = time.time()
    hashes: dict[str, list[str]] = {}
    hcache_path = paths.CANON / "file_hashes.json"
    hcache: dict = {}
    if do_hash and hcache_path.exists():
        try:
            hcache = json.loads(hcache_path.read_text(encoding="utf-8"))
        except Exception:
            hcache = {}
    n_cached = 0
    for g in GROUPS:
        t1 = time.time()
        before = len(hcache)
        walk_group(g, do_hash, hashes, hcache)
        enrich(g)
        print(f"{g.id} {g.title[:52]:<52} files={g.n_files:>6} frames={g.n_frames:>6} {human(g.bytes):>12} "
              f"{'MISSING ' + str(g.missing_paths) if g.missing_paths else ''}[{time.time() - t1:.1f}s]", flush=True)
        if do_hash:
            # saved after every group: hashing 46 GiB is the slow part and must not be repeated after an interruption
            hcache_path.parent.mkdir(parents=True, exist_ok=True)
            hcache_path.write_text(json.dumps(hcache), encoding="utf-8")
    dup_groups = []
    for digest, files in hashes.items():
        if len(files) > 1:
            try:
                sz = os.path.getsize(files[0])
            except OSError:
                sz = 0
            dup_groups.append({"sha256": digest, "files": sorted(files), "bytes": sz})
    dup_groups.sort(key=lambda d: -d["bytes"] * (len(d["files"]) - 1))
    inv = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "repo": str(paths.REPO), "hashed": do_hash,
           "elapsed_s": round(time.time() - t0, 1), "groups": [g.to_dict() for g in GROUPS], "duplicates": dup_groups,
           "totals": {"files": sum(g.n_files for g in GROUPS), "bytes": sum(g.bytes for g in GROUPS),
                      "frames": sum(g.n_frames for g in GROUPS),
                      "real_robot_frames": sum(g.n_frames for g in GROUPS if g.kind == "real_robot")}}
    paths.CANON.mkdir(parents=True, exist_ok=True)
    (paths.CANON / "assets.json").write_text(json.dumps(inv, indent=1, ensure_ascii=False), encoding="utf-8")
    (paths.CANON / "assets.md").write_text(render_md(inv), encoding="utf-8")
    print(f"\n{len(GROUPS)} groups, {inv['totals']['files']} files, {human(inv['totals']['bytes'])}, "
          f"{inv['totals']['frames']} frames; {len(dup_groups)} exact-duplicate groups; {inv['elapsed_s']} s")
    print("wrote", paths.CANON / "assets.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
