# -*- coding: utf-8 -*-
"""calibrate_inputs.py -- refit a shipped blob's input normalisation on real data.

The blob header carries ``input_mean`` / ``input_std``, and the firmware clips
every input to ``mean +- clip_sigma*std`` before the CfC sees it. Those
statistics were fitted on synthetic training data. The 64 visual feature dims
cannot be covered by synthetic data at all -- they are YOLO internal activations
whose real population is a property of the camera, the room and the person, not
of the interface contract. Measured on recorded robot frames, the training
envelope misses them by 2.7 training-sigma at the median with a 3.3x scale
ratio, so the network runs on saturated inputs.

This script patches **only the 64 feature dims**. The remaining 42 are left as
trained on purpose: their real-data statistics can only be sampled with the
robot standing, and fitting them from an upright robot recorded while lying down
would bake in the wrong distribution. The feature dims have no such dependency
-- they depend on what the camera sees, which a lying-down robot still sees.

Supply unlabeled recordings; no teacher labels are used. Feed-forward only, so
the fit is unsupervised and the same recording can later be used to evaluate.

Run:
    python calibrate_inputs.py --sessions data_real data_realtest ^
        --blob out_sw2_b256_80e/m8.blob --out out_sw2_b256_80e/m8_cal.blob
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_blob as EB  # noqa: E402

FEAT_DIMS = 64
OBS_DIM = 106


def load_feat(patterns):
    rows = []
    used = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=True)
            if "obs" not in z.files:
                continue
            o = np.asarray(z["obs"], np.float32)
            if o.ndim == 3:
                o = o.reshape(-1, o.shape[-1])
            if o.shape[1] != OBS_DIM:
                continue
            rows.append(o[:, :FEAT_DIMS])
            used.append((os.path.basename(p), int(o.shape[0])))
    if not rows:
        raise SystemExit("no usable recordings")
    return np.concatenate(rows, axis=0), used


def read_stats(blob):
    off_m = EB.HDR_FIELD_OFFSETS["input_mean"]
    off_s = EB.HDR_FIELD_OFFSETS["input_std"]
    mean = np.frombuffer(blob[off_m:off_m + 1024], dtype="<f4", count=256)
    std = np.frombuffer(blob[off_s:off_s + 1024], dtype="<f4", count=256)
    return mean.astype(np.float32), std.astype(np.float32)


def patch_stats(blob, mean, std, ts=None):
    b = bytearray(blob)
    off_m = EB.HDR_FIELD_OFFSETS["input_mean"]
    off_s = EB.HDR_FIELD_OFFSETS["input_std"]
    b[off_m:off_m + 1024] = np.asarray(mean, "<f4").tobytes()
    b[off_s:off_s + 1024] = np.asarray(std, "<f4").tobytes()
    if ts is not None:
        off_t = EB.HDR_FIELD_OFFSETS["ts"]
        b[off_t:off_t + 4] = struct.pack("<f", float(ts))
    off_c = EB.HDR_FIELD_OFFSETS["crc32"]
    b[off_c:off_c + 4] = b"\x00\x00\x00\x00"
    crc = EB.blob_crc32(bytes(b))
    b[off_c:off_c + 4] = struct.pack("<I", crc)
    return bytes(b), crc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", default=None,
                    help="directories or glob patterns of recorded session npz")
    ap.add_argument("--blob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--std-floor", type=float, default=1e-6,
                    help="lower bound on the fitted std")
    ap.add_argument("--feat-mean", type=float, default=None,
                    help="constant mean for all 64 feature dims instead of a "
                         "fitted one; --feat-mean 0 --feat-std 1 makes the "
                         "block inert for a policy trained with it zeroed")
    ap.add_argument("--feat-std", type=float, default=None,
                    help="constant std, paired with --feat-mean")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    const = a.feat_mean is not None or a.feat_std is not None
    if const and (a.feat_mean is None or a.feat_std is None):
        raise SystemExit("--feat-mean and --feat-std must be given together")
    if not const and not a.sessions:
        raise SystemExit("give --sessions, or both --feat-mean and --feat-std")

    if const:
        mu = np.full(FEAT_DIMS, a.feat_mean, np.float32)
        sd = np.full(FEAT_DIMS, a.feat_std, np.float32)
        used = []
        print(f"fixed feature stats: mean {a.feat_mean}  std {a.feat_std}")
    else:
        pats = []
        for s in a.sessions:
            pats.append(s if any(c in s for c in "*?[")
                        else os.path.join(s, "*.npz"))
        F, used = load_feat(pats)
        print(f"calibration frames: {F.shape[0]}")
        for name, n in used:
            print(f"  {name:24s} {n:5d}")
        mu = F.mean(axis=0)
        sd = np.maximum(F.std(axis=0), a.std_floor)
        print(f"\nfitted feature stats: mean range [{mu.min():+.4f}, "
              f"{mu.max():+.4f}]  std range [{sd.min():.4f}, {sd.max():.4f}]")

    blob = open(a.blob, "rb").read()
    print(f"blob {a.blob}: {len(blob)} bytes")
    m0, s0 = read_stats(blob)
    print(f"  before: feat mean range [{m0[:64].min():+.4f}, {m0[:64].max():+.4f}]"
          f"  std range [{s0[:64].min():.4f}, {s0[:64].max():.4f}]")

    m1, s1 = m0.copy(), s0.copy()
    m1[:FEAT_DIMS] = mu
    s1[:FEAT_DIMS] = sd

    new, crc = patch_stats(blob, m1, s1, ts=time.time())
    if len(new) != len(blob):
        raise SystemExit("blob size changed; refusing to write")
    check = np.frombuffer(new[EB.HDR_FIELD_OFFSETS["input_mean"]:
                              EB.HDR_FIELD_OFFSETS["input_mean"] + 1024],
                         dtype="<f4", count=256)
    if not np.allclose(check[:FEAT_DIMS], mu, atol=1e-6):
        raise SystemExit("patch did not round-trip")

    print(f"  after : feat mean range [{m1[:64].min():+.4f}, {m1[:64].max():+.4f}]"
          f"  std range [{s1[:64].min():.4f}, {s1[:64].max():.4f}]")
    print(f"  crc32 {crc:#010x}")

    if a.dry_run:
        print("\n(dry run, nothing written)")
        return 0

    with open(a.out, "wb") as f:
        f.write(new)
    side = os.path.splitext(a.out)[0] + ".calib.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump({
            "source_blob": os.path.abspath(a.blob),
            "mode": "fixed" if const else "fitted",
            "calibration_frames": (0 if const else int(F.shape[0])),
            "sources": used,
            "patched_dims": [0, FEAT_DIMS],
            "fitted_mean": [round(float(v), 8) for v in mu],
            "fitted_std": [round(float(v), 8) for v in sd],
            "crc32": crc,
            "ts": time.time(),
        }, f, indent=1)
    print(f"\nwrote {a.out}")
    print(f"wrote {side}")
    print("next: evaluate on held-out real recordings, e.g.")
    print(f"  python ..\\bench\\eval_session.py --blob {a.out} --sessions data_real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
