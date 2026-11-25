# -*- coding: utf-8 -*-
"""pc/pack/sensor_measure.py - measure the GC0308 sensor domain from real robot frames.

Writes PACK_ROOT/canon/sensor_domain.json. Every parameter says how it was
measured (or that it is assumed) and how confident we are. Point estimates are
avoided where the honest answer is a range: the transform randomises inside it.

Usage: python pc/pack/sensor_measure.py [--n 240]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

# Non-sealed real-robot frame sources only (D-010).
SOURCES = [
    "evidence/live/calib_1/frames",
    "evidence/live/gesture_1/frames",
    "evidence/live/gesture_2/frames",
    "evidence/live/label_none/frames",
    "evidence/sessions/none_d0.5_front_night_181841/frames",
    "evidence/sessions/fist_d0.5_front_night_181907/frames",
    "evidence/sessions/face_close_d0.5_front_night_182232/frames",
]


def load_sample(n: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pool = []
    for rel in SOURCES:
        d = paths.REPO / rel
        if not d.is_dir():
            continue
        for j in sorted(d.glob("*.jpg")):
            pool.append((rel + "/" + j.name, j))
    rng.shuffle(pool)
    out = []
    for name, jp in pool[:n]:
        im = cv2.imread(str(jp), cv2.IMREAD_GRAYSCALE)
        if im is not None and im.shape == (240, 240):
            out.append((name, im))
    return out


def measure_noise(frames: list[np.ndarray]) -> dict:
    """High-frequency residual in flat patches: pixel minus 3x3 box mean, MAD-estimated sigma."""
    sigmas = []
    for im in frames:
        f = im.astype(np.float32)
        mu = cv2.boxFilter(f, -1, (3, 3), normalize=True)
        resid = f - mu
        grad = np.abs(np.diff(f, axis=1))
        flat = np.zeros_like(resid, dtype=bool)
        flat[:, :-1] = grad < 4.0
        flat[:, [0, -1]] = False
        vals = resid[flat]
        if vals.size > 2000:
            mad = np.median(np.abs(vals - np.median(vals)))
            sigmas.append(1.4826 * mad)
    arr = np.array(sigmas)
    return {
        "sigma_mean": round(float(arr.mean()), 3),
        "sigma_p10": round(float(np.percentile(arr, 10)), 3),
        "sigma_p90": round(float(np.percentile(arr, 90)), 3),
        "method": "MAD of pixel-minus-3x3-mean residual in flat patches (grad<4), per-frame then averaged",
        "n_frames": int(len(arr)),
        "confidence": "medium: conflates sensor noise with JPEG ringing; treat as effective noise",
    }


def measure_neighbor_corr(frames: list[np.ndarray]) -> dict:
    """Adjacent-pixel correlation of the flat-patch residual: 0 = no blur, higher = blur/ISP filtering."""
    corrs = []
    for im in frames[:80]:
        f = im.astype(np.float32)
        mu = cv2.boxFilter(f, -1, (3, 3), normalize=True)
        r = f - mu
        a = r[:, :-1].ravel()
        b = r[:, 1:].ravel()
        if a.std() > 1e-6:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    arr = np.array(corrs)
    return {
        "corr_mean": round(float(arr.mean()), 3),
        "corr_p10": round(float(np.percentile(arr, 10)), 3),
        "corr_p90": round(float(np.percentile(arr, 90)), 3),
        "method": "Pearson correlation of horizontally adjacent flat-patch residuals; interpretation: optical low-pass + ISP + JPEG smoothing",
        "confidence": "low-to-medium: cannot separate defocus blur from ISP filtering; used to pick blur sigma range",
    }


def measure_vignetting(frames: list[np.ndarray]) -> dict:
    """Median radial intensity falloff. Scene content averages out across enough frames."""
    h, w = 240, 240
    yy, xx = np.mgrid[0:h, 0:w]
    cx = cy = 119.5
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / 169.7  # 0 centre .. ~1 corner
    bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]
    ratios_per_frame = []
    for im in frames:
        f = im.astype(np.float32) + 1.0
        means = []
        for lo, hi in bins:
            m = (r >= lo) & (r < hi)
            means.append(float(f[m].mean()))
        centre = means[0]
        if centre > 8.0:
            ratios_per_frame.append([m / centre for m in means])
    arr = np.median(np.array(ratios_per_frame), axis=0)
    return {
        "radial_ratio_median": [round(float(x), 4) for x in arr],
        "bin_edges": [list(b) for b in bins],
        "method": "per-frame radial bin mean normalised to centre bin, median over frames",
        "n_frames": int(len(ratios_per_frame)),
        "confidence": "medium: assumes scene brightness is radially symmetric on average; night sessions bias centre",
    }


def measure_tone(frames: list[np.ndarray]) -> dict:
    ps = [1, 5, 25, 50, 75, 95, 99]
    per = np.percentile(np.concatenate([im.ravel() for im in frames[:: max(1, len(frames) // 40)]]), ps)
    means = [float(im.mean()) for im in frames]
    return {
        "percentiles": {str(p): round(float(v), 1) for p, v in zip(ps, per)},
        "mean_of_frame_means": round(float(np.mean(means)), 2),
        "std_of_frame_means": round(float(np.std(means)), 2),
        "method": "percentiles over a 1/40 subsample of all pixels",
        "confidence": "high for these sessions; exposure is auto so treat as distribution not constant",
    }


def measure_blockiness(frames: list[np.ndarray]) -> dict:
    """JPEG 8x8 block-edge energy ratio: mean abs diff across block boundaries vs inside blocks."""
    edge_e, inside_e = [], []
    for im in frames[:80]:
        f = im.astype(np.float32)
        d = np.abs(np.diff(f, axis=1))
        edge_cols = d[:, 7::8].mean()
        inside_cols = np.delete(d, np.s_[7::8], axis=1).mean()
        edge_e.append(float(edge_cols))
        inside_e.append(float(inside_cols))
    ratio = float(np.mean(edge_e) / max(np.mean(inside_e), 1e-6))
    return {
        "edge_to_inside_ratio": round(ratio, 3),
        "method": "mean |diff| at x%8==7 columns vs other columns, averaged over frames",
        "confidence": "medium: confirms JPEG generation; transform re-encodes at q=80 which reproduces this",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    args = ap.parse_args()
    t0 = time.perf_counter()
    sample = load_sample(args.n)
    frames = [im for _n, im in sample]
    print(f"loaded {len(frames)} real robot frames (non-sealed dev sources)", flush=True)
    assert len(frames) >= 100, "not enough frames"

    doc = {
        "version": 1,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": {
            "width": 240, "height": 240, "native_gray": True,
            "fov_h_deg": 75.5, "focal_px": 154.99,
            "jpeg_quality": 80,
            "jpeg_quality_source": "firmware capture default q=80 (robot_truth / LATENCY-MAP)",
            "frame_period_ms": 32.0,
        },
        "measured": {
            "noise": measure_noise(frames),
            "neighbor_corr": measure_neighbor_corr(frames),
            "vignetting": measure_vignetting(frames),
            "tone": measure_tone(frames),
            "jpeg_blockiness": measure_blockiness(frames),
        },
        "assumed": {
            "gray_conversion": {"weights": "BT.601", "confidence": "assumed: no colour reference frames exist"},
            "defocus_blur_sigma_px": [0.3, 0.9],
            "defocus_basis": "chosen so transformed sharp images reach the measured neighbour-correlation band",
            "motion_blur": {"prob": 0.35, "len_px": [2, 9], "basis": "night long exposure; frame period 32 ms and visible streaking in gesture sessions"},
            "exposure_gain_jitter": {"scale": [0.85, 1.12], "gamma": [0.9, 1.12], "basis": "auto exposure observed across sessions"},
        },
    }
    out = paths.CANON / "sensor_domain.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} in {time.perf_counter()-t0:.1f}s", flush=True)
    print(json.dumps({k: v for k, v in doc["measured"].items() if k != "tone"}, ensure_ascii=False, indent=1)[:800], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
