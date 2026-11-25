# -*- coding: utf-8 -*-
"""pc/robot_truth.py - the one place code gets the physical robot's numbers from.

robot_truth.json next to this file holds the facts (value, how measured, when,
evidence, confidence, verification level). Scripts import the names below
instead of writing a field of view or a camera height of their own: eight
different fields of view were hard-coded in this repository before 2026-11-24,
and the measured one is the only value that belongs in code.

    from robot_truth import FOV_H_DEG, F_PX, CAM_HEIGHT_M, fact
"""
from __future__ import annotations

import json
import math
from pathlib import Path

TRUTH_PATH = Path(__file__).with_name("robot_truth.json")
REQUIRED = ("value", "method", "date", "source", "confidence", "level")
LEVELS = ("T0", "T1", "T2", "T3", "T4", "untested")


def load(path: Path = TRUTH_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    facts = data["facts"]
    bad = [k for k, v in facts.items() if any(r not in v for r in REQUIRED) or v["level"] not in LEVELS
           or (v["value"] is None and v["level"] != "untested")]
    if bad:
        raise ValueError(f"robot_truth: facts missing required fields, with an unknown level, "
                         f"or with no value but a level above untested: {bad}")
    return data


DATA = load()


def fact(name: str):
    """The value of a fact; KeyError (naming the available facts) if it does not exist."""
    try:
        return DATA["facts"][name]["value"]
    except KeyError:
        raise KeyError(f"robot_truth has no fact '{name}'; available: {', '.join(sorted(DATA['facts']))}") from None


def fact_meta(name: str) -> dict:
    return DATA["facts"][name]


# ---------------------------------------------------------------- measured / declared
IMG_W, IMG_H = fact("camera.resolution_px")
FOV_H_DEG = float(fact("camera.fov_h_deg"))
CAM_HEIGHT_M = float(fact("camera.height_m"))
HEAD_DEADBAND_DEG = float(fact("head.deadband_half_width_deg"))
YAW_STICTION_LEAD_DEG = float(fact("heading.stiction_lead_deg"))
LQR_K0, LQR_K1, LQR_K2, LQR_K3 = (float(x) for x in fact("balance.lqr_gains"))
IMU_ZERO_DEG = float(fact("balance.imu_zero_deg"))
STATIC_FRICTION_TORQUE = float(fact("balance.static_friction_torque"))
YAW_LEAD_MAX_DEG = float(fact("firmware.cfg_defaults")["yaw_lead_max"])
POS_LEAD_MAX_M = float(fact("firmware.cfg_defaults")["pos_lead_max"])
WHEEL_DIAMETER_M = float(fact("wheel.diameter_m"))
WHEEL_TICKS_PER_REV = float(fact("wheel.ticks_per_rev"))

# ---------------------------------------------------------------- derived (do not store these twice)
F_PX = IMG_W / (2.0 * math.tan(math.radians(FOV_H_DEG) / 2.0))     # focal length in pixels, image centre
PX_PER_DEG = F_PX * math.pi / 180.0                                # small-angle scale at the image centre
HALF_FOV_RAD = math.radians(FOV_H_DEG) / 2.0
# Nearest ground point a level camera can see: h / tan(half field of view). Anything closer is invisible.
BLIND_ZONE_M = CAM_HEIGHT_M / math.tan(HALF_FOV_RAD)
TICK_M = math.pi * WHEEL_DIAMETER_M / WHEEL_TICKS_PER_REV
DEG_PER_TICK = 0.2190                                              # body yaw per tick of wheel difference (measured)
DEG_PER_WDIFF_M = -DEG_PER_TICK / TICK_M                           # sign: body yaw decreases with wheel1 - wheel2
REST_OFFSET_M_PER_DEG = -LQR_K2 / LQR_K0                           # (wheel_x - stable_pos) per deg of balance-angle mismatch
FRICTION_LEAD_M = STATIC_FRICTION_TORQUE / LQR_K0                  # position lead that static friction can absorb
