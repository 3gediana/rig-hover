# -*- coding: utf-8 -*-
"""pc/pack/sensor_domain.py - the sensor-domain contract: turn any RGB/gray image into a robot-camera frame.

The chain and its parameters live in PACK_ROOT/canon/sensor_domain.json (measured by sensor_measure.py).
Every randomised step draws inside the measured or assumed range; pass a seeded rng for determinism.

Chain order (order matters — noise and blur happen before JPEG, exposure before everything optical):
  crop/resize -> exposure/gamma jitter -> vignetting -> defocus blur -> motion blur -> noise -> gray -> JPEG q~80
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

_JSON = paths.CANON / "sensor_domain.json"


def load_params(path: Path | None = None) -> dict:
    p = path or _JSON
    return json.loads(p.read_text(encoding="utf-8"))


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    # BT.601 (assumed, see sensor_domain.json)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)


def _center_square(img: np.ndarray, rng: np.random.Generator, crop_jitter: float) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    jit = 1.0 + rng.uniform(-crop_jitter, crop_jitter)
    side = int(np.clip(side / max(jit, 1e-3), 32, side))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def _vignette(imgf: np.ndarray, strength: float) -> np.ndarray:
    h, w = imgf.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) / (h / np.sqrt(2))
    prof = 1.0 - strength * np.clip(r - 0.4, 0, None) ** 2
    return imgf * prof.astype(np.float32)


def _motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    c = length // 2
    rad = np.deg2rad(angle_deg)
    for i in range(length):
        x = int(round(c + (i - c) * np.cos(rad)))
        y = int(round(c + (i - c) * np.sin(rad)))
        if 0 <= y < length and 0 <= x < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k)


def robotify(
    img: np.ndarray,
    params: dict | None = None,
    rng: np.random.Generator | None = None,
    out_size: int = 240,
    jpeg_quality: int | None = None,
) -> np.ndarray:
    """Transform any RGB/gray image into the GC0308 robot sensor domain. Returns uint8 gray out_size x out_size."""
    if rng is None:
        rng = np.random.default_rng()
    if params is None:
        params = load_params()
    a = params["assumed"]
    m = params["measured"]

    g = _to_gray(img)
    g = _center_square(g, rng, crop_jitter=0.08)
    g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
    f = g.astype(np.float32) / 255.0

    # exposure / gain jitter (auto exposure)
    scale = rng.uniform(*a["exposure_gain_jitter"]["scale"])
    gamma = rng.uniform(*a["exposure_gain_jitter"]["gamma"])
    f = np.clip(f * scale, 0, 1) ** gamma

    # vignetting: measured falloff to ~0.91 at r=0.6..0.8 is weak; randomise 0..1x of that strength
    strength = rng.uniform(0.0, 0.35)
    f = _vignette(f, strength)

    f = np.clip(f * 255.0, 0, 255).astype(np.uint8)

    # defocus blur
    sigma = rng.uniform(*a["defocus_blur_sigma_px"])
    if sigma > 0.05:
        ksz = int(np.ceil(sigma * 3)) * 2 + 1
        f = cv2.GaussianBlur(f, (ksz, ksz), sigma)

    # motion blur (night long exposure)
    if rng.random() < a["motion_blur"]["prob"]:
        length = int(rng.integers(a["motion_blur"]["len_px"][0], a["motion_blur"]["len_px"][1] + 1))
        f = _motion_blur(f, length, float(rng.uniform(0, 180)))

    # sensor noise (effective, pre-JPEG)
    ns = m["noise"]
    noise_sigma = rng.uniform(ns["sigma_p10"], ns["sigma_p90"]) * rng.uniform(0.8, 1.6)
    f = np.clip(f.astype(np.float32) + rng.normal(0, noise_sigma, f.shape), 0, 255).astype(np.uint8)

    # JPEG compression round-trip
    q = jpeg_quality if jpeg_quality is not None else int(rng.integers(76, 86))
    enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
    f = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return f


def robotify_video_frame_rgb888(rgb: np.ndarray, **kw) -> np.ndarray:
    """Convenience for RGB-ordered input (e.g. PIL / numpy RGB rather than cv2 BGR)."""
    return robotify(rgb[..., ::-1], **kw)
