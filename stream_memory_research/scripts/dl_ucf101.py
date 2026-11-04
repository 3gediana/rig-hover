"""Stage 1 downloader: UCF101 archive + official recognition-task split file.

Source order (first that works wins), see logs/data_video_status.md:
  (a) HF  quchenyuan/UCF101-ZIP   UCF-101.zip                       6.96 GB
  (b) HF  MichiganNLP/ucf-101     UCF101.zip                        6.96 GB
  (c) official crcv.ucf.edu       UCF101.rar  (cannot unrar -> last resort)

Also fetches ``UCF101TrainTestSplits-RecognitionTask.zip`` which contains the
*official* 3-fold train/test split (``*_trainlist0X.txt`` / ``*_testlist0X.txt``).
That split covers 25 videos/class, which is what we subsample from.

Usage
-----
python scripts/dl_ucf101.py [--dest G:\\stream_memory_data\\videos\\ucf101] [--conn 8]
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import (DownloadError, GB, download_resumable,  # noqa: E402
                               hf_url, server_info)
from src.data.util import attach_log  # noqa: E402

CANDIDATES = [
    ("quchenyuan/UCF101-ZIP", "UCF-101.zip"),
    ("MichiganNLP/ucf-101", "UCF101.zip"),
]
SPLIT_CANDIDATES = [
    ("quchenyuan/UCF101-ZIP", "UCF101TrainTestSplits-RecognitionTask.zip"),
    ("MichiganNLP/ucf-101", "UCF101TrainTestSplits-RecognitionTask.zip"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=r"G:\stream_memory_data\videos\ucf101")
    ap.add_argument("--conn", type=int, default=20)
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--log", default=r"G:\stream_memory_data\logs\dl_ucf101.log")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()
    if not args.probe_only:
        attach_log(args.log)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # ---- official split file (tiny, try once) --------------------------------
    for repo, name in SPLIT_CANDIDATES:
        url = hf_url(repo, name)
        try:
            p = download_resumable(url, dest / name, n_conn=2, quiet=True)
            print(f"[split] got {p.name} from {repo} ({p.stat().st_size} B)")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[split] {repo} failed: {type(e).__name__}: {e}")
    else:
        print("[split] WARNING no official split file obtained -> deterministic split")

    # ---- big archive ---------------------------------------------------------
    for repo, name in CANDIDATES:
        url = hf_url(repo, name)
        info = server_info(url)
        print(f"[probe] {repo}/{name}: ok={info['ok']} size={int(info['size']) / GB:.2f} GB "
              f"ranges={info['ranges']} {info['reason']}")
        if not info["ok"]:
            continue
        if args.probe_only:
            return 0
        t0 = time.time()
        try:
            p = download_resumable(url, dest / "UCF-101.zip",
                                   n_conn=args.conn,
                                   chunk_size=args.chunk_mb * (1 << 20))
        except DownloadError as e:
            print(f"[dl] {repo} failed: {e}")
            continue
        print(f"[dl] archive ready: {p} ({p.stat().st_size / GB:.2f} GB) "
              f"in {(time.time() - t0) / 60:.1f} min")
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
        print(f"[dl] zip has {len(names)} entries, e.g. {names[:3]}")
        return 0

    print("[dl] FAILED: no UCF101 archive source reachable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
