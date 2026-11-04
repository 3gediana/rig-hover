"""Stage 2-4 downloaders: MOT17, Charades, DAVIS 2017 (resumable, multi-conn).

python scripts/dl_datasets.py mot17
python scripts/dl_datasets.py charades [--probe]
python scripts/dl_datasets.py davis
python scripts/dl_datasets.py charades_ann
python scripts/dl_datasets.py all

Every download is chunk-resumable (see src/data/download.py); killing the
process and rerunning resumes.  Charades additionally measures real throughput
for 20 s first and reports it, because the task instructions require deciding
"download the 15 GB 480p zip" vs "fall back to a smaller HF mirror" based on
measured speed.
"""
from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download import (DownloadError, GB, download_resumable,  # noqa: E402
                               hf_url, probe_speed, server_info)
from src.data.util import attach_log  # noqa: E402

VDATA = Path(r"G:\stream_memory_data\videos")
LOGS = Path(r"G:\stream_memory_data\logs")

TARGETS = {
    # name: (url, dest, n_conn)
    "mot17": ("https://motchallenge.net/data/MOT17.zip", VDATA / "mot17" / "MOT17.zip", 12),
    "charades": ("https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip",
                 VDATA / "charades" / "Charades_v1_480.zip", 16),
    "charades_ann": ("https://ai2-public-datasets.s3.amazonaws.com/charades/Charades.zip",
                     VDATA / "charades" / "Charades.zip", 4),
    "davis_img": ("https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
                  VDATA / "davis" / "DAVIS-2017-trainval-480p.zip", 8),
    # HF mirrors, tried in order if the primary dies
    "charades_hf": (hf_url("linxxx3/Charades_v1_480", "Charades/Charades_v1_480.zip"),
                    VDATA / "charades" / "Charades_v1_480_hf.zip", 12),
    "davis_hf": (hf_url("jinyoungkim/davis", "DAVIS-2017-trainval-480p.zip"), None, 8),
}


def show_zip(path: Path, limit: int = 12) -> None:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        print(f"  [zip] {path.name}: {len(names)} entries; sample={names[:limit]}")
    except Exception as e:  # noqa: BLE001
        print(f"  [zip] cannot list {path.name}: {type(e).__name__}: {e}")


def get(name: str, n_conn: int = 0, probe_seconds: float = 20.0) -> Path:
    url, dest, default_conn = TARGETS[name]
    assert dest is not None, f"{name} has no destination configured"
    info = server_info(url)
    print(f"[{name}] ok={info['ok']} size={int(info['size']) / GB:.3f} GB "
          f"ranges={info['ranges']} {info['reason']}", flush=True)
    if not info["ok"]:
        raise DownloadError(f"{name}: unreachable {info['reason']}")
    nc = n_conn or default_conn
    if name == "charades":
        mbps = probe_speed(url, seconds=probe_seconds)
        print(f"[charades] measured 1-connection throughput: {mbps:.2f} MB/s -> "
              f"6.48 GB estimate {15.2 * 1024 / max(mbps, 1e-3) / 60:.1f} min",
              flush=True)
    t0 = time.time()
    p = download_resumable(url, dest, n_conn=nc, chunk_size=16 * (1 << 20),
                           info=info)
    print(f"[{name}] done {p} {p.stat().st_size / GB:.3f} GB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    show_zip(p)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", default=["all"])
    ap.add_argument("--conn", type=int, default=0)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--log", default=str(LOGS / "dl_datasets.log"))
    args = ap.parse_args()
    attach_log(args.log)

    names: list[str] = []
    for w in args.what:
        if w == "all":
            names += ["mot17", "charades_ann", "charades", "davis_img"]
        else:
            names.append(w)
    if args.probe_only:
        for n in names:
            url = TARGETS[n][0]
            i = server_info(url)
            sp = probe_speed(url, seconds=8.0) if i["ok"] else 0.0
            print(f"[probe] {n}: ok={i['ok']} size={int(i['size']) / GB:.3f} GB "
                  f"speed={sp:.2f} MB/s", flush=True)
        return 0
    rc = 0
    for n in names:
        try:
            get(n, args.conn)
        except Exception as e:  # noqa: BLE001
            print(f"[{n}] FAILED: {type(e).__name__}: {e}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
