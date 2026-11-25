# -*- coding: utf-8 -*-
"""pc/pack/sealed.py - the only sanctioned doorway to held-out material.

Default policy: sealed data does not exist. A script that is about to evaluate
on it must pass --final-eval (or set ROBOT_PACK_FINAL_EVAL=1 for a controlled
test harness), state its purpose, and accept one JSON-lines entry in
PACK_ROOT/sealed_access.log. Without that explicit opt-in the guard raises
SealedAccessError before the data is opened.

The sealed originals are never moved by this module. --materialize mirrors them
into PACK_ROOT/sealed/mirror with a sha256 manifest; --verify checks that the
mirror still matches the originals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import paths

SEALED_REL_DIRS = (
    "dataset/takes/robot_20261115_151733_cb8455",
    "evidence/live/fov_sweep",
    "evidence/sessions/empty_d0.5_front_night_182201",
)
SEALED_DERIVED_MARKERS = (
    "cb8455",
    "empty_d0.5",
    "fov_sweep",
    "test_heldout",
    "real_distill_heldout",
)


class SealedAccessError(PermissionError):
    pass


def norm_rel(path: str | Path) -> str:
    s = str(path).replace("\\", "/")
    try:
        p = Path(path)
        if p.is_absolute():
            s = p.resolve().relative_to(paths.REPO.resolve()).as_posix()
    except Exception:
        pass
    return s


def is_sealed_path(path: str | Path) -> bool:
    r = norm_rel(path)
    if paths.is_sealed_rel(r):
        return True
    low = r.lower()
    return any(m in low for m in SEALED_DERIVED_MARKERS)


def final_eval_requested(argv: list[str] | None = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    return "--final-eval" in args or os.environ.get("ROBOT_PACK_FINAL_EVAL") == "1"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except Exception:
        return "unknown"


def access_log_path() -> Path:
    return paths.PACK_ROOT / "sealed_access.log"


def log_access(rel_paths: list[str], purpose: str) -> None:
    logp = access_log_path()
    logp.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": norm_rel(sys.argv[0]),
        "argv": sys.argv[1:],
        "purpose": purpose,
        "paths": rel_paths,
        "git_head": _git_head(),
    }
    with open(logp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def require_final_eval(paths_to_open: list[str | Path], purpose: str, argv: list[str] | None = None) -> list[str]:
    """Raise unless every sealed path in paths_to_open is covered by --final-eval.

    Returns the sealed repo-relative paths that were authorised. Non-sealed
    paths are ignored and never logged.
    """
    sealed = sorted({norm_rel(p) for p in paths_to_open if is_sealed_path(p)})
    if not sealed:
        return []
    if not final_eval_requested(argv):
        raise SealedAccessError(
            "sealed data access refused for " + purpose + ": " + ", ".join(sealed) +
            "; pass --final-eval and state the evaluation purpose"
        )
    log_access(sealed, purpose)
    return sealed


def open_sealed(rel_path: str | Path, purpose: str, mode: str = "rb", argv: list[str] | None = None):
    require_final_eval([rel_path], purpose, argv=argv)
    p = Path(rel_path)
    if not p.is_absolute():
        p = paths.REPO / rel_path
    return open(p, mode)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sealed_sources() -> list[Path]:
    return [paths.REPO / r for r in SEALED_REL_DIRS]


def materialize_sealed() -> dict:
    """Copy (never move) the sealed tree into PACK_ROOT/sealed/mirror + manifest."""
    root = paths.PACK_ROOT / "sealed" / "mirror"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for src in sealed_sources():
        if not src.exists():
            raise FileNotFoundError(src)
        rel = src.relative_to(paths.REPO).as_posix()
        dst_root = root / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for dp, _dn, fn in os.walk(src):
            for name in fn:
                f = Path(dp) / name
                frel = f.relative_to(paths.REPO).as_posix()
                dst = root / frel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if (not dst.exists()) or dst.stat().st_size != f.stat().st_size or dst.stat().st_mtime_ns != f.stat().st_mtime_ns:
                    shutil.copy2(f, dst)
                entries.append({
                    "path": frel,
                    "bytes": f.stat().st_size,
                    "sha256": sha256_file(f),
                })
    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": "copy-only mirror; originals stay in place; access still requires --final-eval",
        "entries": sorted(entries, key=lambda x: x["path"]),
    }
    out = paths.PACK_ROOT / "sealed" / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_sealed() -> tuple[bool, list[str]]:
    manifest_path = paths.PACK_ROOT / "sealed" / "manifest.json"
    if not manifest_path.exists():
        return False, ["missing manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for e in manifest["entries"]:
        src = paths.REPO / e["path"]
        dst = paths.PACK_ROOT / "sealed" / "mirror" / e["path"]
        if not dst.exists():
            problems.append("mirror missing " + e["path"])
        elif sha256_file(dst) != e["sha256"]:
            problems.append("mirror hash mismatch " + e["path"])
        if not src.exists():
            problems.append("original missing " + e["path"])
        elif sha256_file(src) != e["sha256"]:
            problems.append("original hash mismatch " + e["path"])
    return not problems, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--require", default=None)
    ap.add_argument("--purpose", default="manual sealed access check")
    ap.add_argument("--final-eval", action="store_true")
    args = ap.parse_args()
    if args.materialize:
        m = materialize_sealed()
        print(f"materialized {len(m['entries'])} sealed files")
    if args.verify:
        ok, problems = verify_sealed()
        print("verify", "ok" if ok else "FAIL")
        for p in problems:
            print(p)
        return 0 if ok else 1
    if args.require:
        try:
            got = require_final_eval([args.require], args.purpose)
        except SealedAccessError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print("authorised", json.dumps(got, ensure_ascii=False))
    if not (args.materialize or args.verify or args.require):
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
