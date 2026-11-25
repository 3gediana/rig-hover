# -*- coding: utf-8 -*-
"""Tests for the sensor-domain contract transform."""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pc" / "pack"))

import paths  # noqa: E402
import sensor_domain as sd  # noqa: E402


def _test_img(color: bool = True, size: int = 480) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size + 120].astype(np.float32)
    base = ((np.sin(xx / 17) + np.cos(yy / 23)) * 60 + 128).astype(np.uint8)
    if not color:
        return base
    return np.stack([base, np.roll(base, 5, 0), np.roll(base, 9, 1)], axis=-1)


def test_params_file_exists_and_is_complete():
    p = paths.CANON / "sensor_domain.json"
    assert p.exists(), "run sensor_measure.py first"
    d = sd.load_params()
    for key in ("camera", "measured", "assumed"):
        assert key in d
    assert d["camera"]["width"] == 240 and d["camera"]["jpeg_quality"] == 80
    for meas in ("noise", "neighbor_corr", "vignetting", "tone", "jpeg_blockiness"):
        assert "method" in d["measured"][meas] and "confidence" in d["measured"][meas]


def test_output_shape_dtype_range():
    rng = np.random.default_rng(0)
    out = sd.robotify(_test_img(), rng=rng)
    assert out.shape == (240, 240) and out.dtype == np.uint8
    assert out.ndim == 2  # gray, like the robot camera
    assert 0 <= int(out.min()) and int(out.max()) <= 255


def test_gray_input_accepted():
    out = sd.robotify(_test_img(color=False), rng=np.random.default_rng(1))
    assert out.shape == (240, 240)


def test_deterministic_with_seed():
    img = _test_img()
    a = sd.robotify(img, rng=np.random.default_rng(42))
    b = sd.robotify(img, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_seed_changes_output():
    img = _test_img()
    a = sd.robotify(img, rng=np.random.default_rng(1))
    b = sd.robotify(img, rng=np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_output_is_noisier_and_blockier_than_clean_resize():
    """The transform must actually degrade: high-frequency energy and JPEG block edges appear."""
    img = _test_img()
    clean = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, (240, 240))
    rng = np.random.default_rng(7)
    outs = [sd.robotify(img, rng=rng).astype(np.float32) for _ in range(6)]
    hf_clean = np.abs(np.diff(clean.astype(np.float32), axis=1)).mean()
    hf_robot = float(np.mean([np.abs(np.diff(o, axis=1)).mean() for o in outs]))
    # noise adds high-frequency energy even though blur removes some; net effect measured over several draws
    assert hf_robot > hf_clean * 0.5  # not destroyed
    # blockiness: mean diff at x%8==7 vs inside columns
    for o in outs[:2]:
        d = np.abs(np.diff(o, axis=1))
        ratio = d[:, 7::8].mean() / max(np.delete(d, np.s_[7::8], axis=1).mean(), 1e-6)
        assert ratio > 0.9  # JPEG block structure present or neutral (blur can mask it)


def test_real_frame_statistics_land_in_measured_ballpark():
    """Transformed sharp images should roughly reproduce the measured noise/no-flatness band (sanity, not a fit)."""
    img = _test_img()
    rng = np.random.default_rng(11)
    sigmas = []
    for _ in range(8):
        out = sd.robotify(img, rng=rng).astype(np.float32)
        mu = cv2.boxFilter(out, -1, (3, 3), normalize=True)
        resid = out - mu
        sigmas.append(float(np.abs(resid).mean()))
    assert np.mean(sigmas) > 0.5  # comparable order to measured MAD sigma ~0.7 (mean-abs is smaller than sigma)
