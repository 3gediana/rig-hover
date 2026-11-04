"""Probe availability of public video datasets (HEAD requests + HF/ModelScope search).

Run:  python scripts/probe_data_sources.py
Writes: logs/data_source_probe.txt (and prints)
No downloads, only reachability + Content-Length.
"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import urllib.error
import urllib.request

socket.setdefaulttimeout(20)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

CANDIDATES = [
    # --- single-object video object segmentation / occlusion ---
    ("DAVIS17 trainval 480p", "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"),
    ("DAVIS17 test-dev 480p", "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip"),
    ("DAVIS16 480p", "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2016-trainval.zip"),
    # --- action recognition / event detection ---
    ("Charades_v1_480", "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip"),
    ("Charades_annotations", "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades.zip"),
    ("Charades_v1_720", "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_720.zip"),
    ("UCF101 rar", "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"),
    ("UCF101 splits zip", "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"),
    ("HMDB51 rar", "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"),
    ("ActivityNet", "http://activity-net.org/download.html"),
    ("THUMOS14", "https://www.crcv.ucf.edu/THUMOS14/THUMOS14.zip"),
    # --- multi object tracking ---
    ("MOT17 zip", "https://motchallenge.net/data/MOT17.zip"),
    ("MOT20 zip", "https://motchallenge.net/data/MOT20.zip"),
    ("KITTI tracking", "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"),
    # --- misc ---
    ("Kinetics600 mini", "https://storage.googleapis.com/deepmind-media/Datasets/kinetics600.tar.gz"),
]

HF_SEARCH = [
    "ucf101", "hmdb51", "something something", "charades", "davis", "mot17",
    "youtube vos", "epic kitchens", "speech commands video", "breakfast actions",
    "jhmdb", "kinetics", "videomae", "ssv2", "robot video", "egocentric",
]


def head(url: str):
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        r = urllib.request.urlopen(req, timeout=20, context=CTX)
        size = r.headers.get("Content-Length")
        return f"OK {r.status} size={int(size)/1e9:.2f}GB" if size else f"OK {r.status} size=?"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.reason}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL {type(e).__name__}: {str(e)[:80]}"


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    out = []
    out.append("=" * 78)
    out.append("DIRECT DOWNLOAD CANDIDATES")
    out.append("=" * 78)
    for name, url in CANDIDATES:
        line = f"{name:26s} {head(url):45s} {url}"
        print(line, flush=True)
        out.append(line)

    out.append("")
    out.append("=" * 78)
    out.append("HUGGINGFACE DATASET SEARCH")
    out.append("=" * 78)
    for q in HF_SEARCH:
        try:
            data = get_json(
                f"https://huggingface.co/api/datasets?search={urllib.parse.quote(q)}&limit=25"
                "&sort=downloads&direction=-1"
            )
            line = f"\n## {q}  ({len(data)} hits)"
            print(line, flush=True)
            out.append(line)
            for d in data[:12]:
                subs = ",".join(d.get("tags", [])[:6])
                line = f"   {d['id']:60s} dl={d.get('downloads', 0):>10} {subs}"
                print(line, flush=True)
                out.append(line)
        except Exception as e:  # noqa: BLE001
            line = f"\n## {q}  FAIL {type(e).__name__}: {str(e)[:80]}"
            print(line, flush=True)
            out.append(line)

    with open("logs/data_source_probe.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\nwrote logs/data_source_probe.txt")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    import urllib.parse  # noqa: F401
    main()
