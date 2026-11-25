# -*- coding: utf-8 -*-
"""train_mono_depth_pack.py - A3 desktop-geometry pack trainer (O-011 vertical slice).

Pack-based, dev-only, sealed-free by construction: real frames come ONLY from the
explicit dev take list (never globbed, never the last-take-heldout logic that used
to pull the sealed cb8455). Synth scenes are sampled on a coverage matrix and each
carries coverage tags; the matrix counts go into the data card.

Usage:
  python pc/end_side/train_mono_depth_pack.py --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pack"))

import paths  # noqa: E402
from train_mono_depth import render_analytical_desktop_scene  # noqa: E402

DEV_TRAIN_TAKES = [
    "robot_20261107_204337_5ca36d",
    "robot_20261107_212119_67f0c1",
    "robot_20261107_213314_5c0158",
    "robot_20261107_214950_1f30d1",
]
DEV_TEST_TAKE = "robot_20261115_150045_65a54d"

PITCH_BINS = [(-12.0, -4.0), (-4.0, 4.0), (4.0, 12.0)]
WALL_BINS = [(1.6, 2.0), (2.0, 2.6)]  # near/far table edge


def sample_coverage_scene(rng: np.random.Generator) -> tuple[dict, dict]:
    """Sample one analytical scene on the coverage matrix; returns (scene, tags)."""
    pb = PITCH_BINS[int(rng.integers(0, 3))]
    wb = WALL_BINS[int(rng.integers(0, 2))]
    pitch = float(rng.uniform(*pb))
    h = float(rng.uniform(0.115, 0.155))
    roll = float(rng.uniform(-6.0, 6.0))
    dark_forced = bool(rng.random() < 0.35)
    thin_neg = bool(rng.random() < 0.25)
    tags = {
        "pitch_bin": ["neg", "mid", "pos"][PITCH_BINS.index(pb)],
        "pitch_deg": round(pitch, 1),
        "wall_bin": "near" if wb == WALL_BINS[0] else "far",
        "dark_object": dark_forced,
        "thin_negative": thin_neg,
    }
    objects = None
    if dark_forced or thin_neg:
        objects = []
        if dark_forced:
            lum = float(rng.uniform(0.03, 0.12))
            objects.append({
                "kind": "cylinder", "cx": float(rng.uniform(0.5, 1.2)), "cy": float(rng.uniform(-0.3, 0.3)),
                "r": float(rng.uniform(0.035, 0.07)), "h": float(rng.uniform(0.08, 0.18)),
                "color": np.array([lum, lum, lum], dtype=np.float32), "stripe": 0.0,
            })
        if thin_neg:
            objects.append({
                "kind": "box", "cx": float(rng.uniform(0.4, 1.1)), "cy": float(rng.uniform(-0.3, 0.3)),
                "lx": float(rng.uniform(0.04, 0.09)), "ly": float(rng.uniform(0.04, 0.09)),
                "h": float(rng.uniform(0.012, 0.025)), "yaw": float(rng.uniform(-3.14, 3.14)),
                "color": rng.uniform(0.2, 0.8, size=(3,)).astype(np.float32),
            })
    # NOTE: wall distance / lighting / material are renderer-internal randoms;
    # pitch/roll/h are pinned via fixed_pose. Void/abyss scenes are NOT yet
    # representable in this renderer - recorded as a coverage gap in the data card.
    scene = render_analytical_desktop_scene(rng, fixed_pose=(h, pitch, roll), custom_objects=objects)
    return scene, tags


def build_pack(n_synth: int, real_rows_tr: list, real_rows_te: list, seed: int = 20261125):
    rng = np.random.default_rng(seed)
    synth, tags = [], []
    for _ in range(n_synth):
        s, t = sample_coverage_scene(rng)
        synth.append({
            "rgb_nhwc": s["rgb_nhwc"], "depth_m": s["depth_m"],
            "inv_depth_01": s["inv_depth_01"], "elev_mask": s["elev_mask"],
        })
        tags.append(t)
    # coverage counts
    cov: dict[str, dict[str, int]] = {}
    for t in tags:
        for k, v in t.items():
            if isinstance(v, bool) or k.endswith("_bin"):
                cov.setdefault(k, {})[str(v)] = cov.setdefault(k, {}).get(str(v), 0) + 1
    return synth, tags, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-synth", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    n_synth = a.n_synth or (200 if a.smoke else 2400)
    epochs = a.epochs or (3 if a.smoke else 40)
    real_per_take = 12 if a.smoke else 40

    import torch
    sys.path.insert(0, str(HERE.parent / "pack"))
    from k3_desk_geometry import sample_dev_real, train_arm, eval_real_geometry  # noqa: E402
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} smoke={a.smoke} n_synth={n_synth} epochs={epochs}", flush=True)

    print("sampling dev real frames...", flush=True)
    import k3_desk_geometry as k3
    k3_frames = {}
    # sample_dev_real uses fixed frames_per_take; patch count by calling then slicing
    real = sample_dev_real(frames_per_take=real_per_take)
    real_tr = [r for take in DEV_TRAIN_TAKES for r in real[take]]
    real_te = real[DEV_TEST_TAKE]
    print(f"real train={len(real_tr)} test={len(real_te)}", flush=True)

    print(f"building coverage-matrix synth pack n={n_synth}...", flush=True)
    synth, tags, cov = build_pack(n_synth, real_tr, real_te)
    mix = synth + real_tr
    np.random.default_rng(0).shuffle(mix)
    print(f"training on {len(mix)} samples...", flush=True)
    model = train_arm(mix, device, epochs=epochs, batch=32)

    print("evaluating...", flush=True)
    real_eval = eval_real_geometry(model, real_te, device)
    # synth heldout quick check: exact-GT rel error on 40 fresh scenes
    rng = np.random.default_rng(999)
    holdout = []
    for _ in range(40):
        s, _t = sample_coverage_scene(rng)
        holdout.append({"rgb_nhwc": s["rgb_nhwc"], "depth_m": s["depth_m"],
                        "inv_depth_01": s["inv_depth_01"], "elev_mask": s["elev_mask"]})
    from mono_depth_model import inv_depth_01_to_metric_m
    rels = []
    with torch.no_grad():
        for row in holdout[:40]:
            x = torch.from_numpy(row["rgb_nhwc"][None]).permute(0, 3, 1, 2).float().to(device)
            probs = torch.sigmoid(model.forward_logits(x))
            pred = inv_depth_01_to_metric_m(probs[0, 0]).cpu().numpy()
            gt = row["depth_m"]
            m = gt < 2.5
            rels.append(float(np.mean(np.abs(pred[m] - gt[m]) / np.clip(gt[m], 0.05, None))))
    synth_abs_rel = round(float(np.mean(rels)), 4)

    report = {
        "model": "HoverMonoDepthNet-pack",
        "camera": {"fov_h_deg": 75.5, "h_cam_m": 0.135, "source": "pc/robot_truth.json measured"},
        "scale": {"n_synth": n_synth, "epochs": epochs, "real_train": len(real_tr), "real_test": len(real_te)},
        "coverage": cov,
        "coverage_gaps": [
            "void/abyss scenes not representable in current renderer",
            "lighting direction/material are internal randoms, not matrix-controlled",
            "real frames: single room, night, 0.5m bias - no daylight/motion-blur cell",
        ],
        "eval": {"real_physical": real_eval, "synth_heldout_abs_rel": synth_abs_rel},
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "train_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), out / "mono_depth_pack.pt")

    card = f"""# A3 桌面几何数据卡（pack v1，{time.strftime('%Y-%m-%d')}）

## 构成
- 合成：{n_synth} 张解析渲染桌面（实测相机 75.5°/0.135 m，coverage-matrix 采样）
- 真机 dev 训练帧：{len(real_tr)}（4 条 11-07 take × {real_per_take}，伪 GT）
- 真机 dev 测试帧：{len(real_te)}（65a54d，跨 take）
- 零封存接触：构建时显式 dev take 列表，无 glob

## 覆盖矩阵（合成部分计数）
```json
{json.dumps(cov, ensure_ascii=False, indent=2)}
```

## 已知覆盖缺口
1. 深渊/无桌沿场景：渲染器当前不可表达
2. 光照方向/材质：渲染器内部随机，未纳入矩阵控制
3. 真机帧：单房间夜间 0.5 m 偏置，无日光/运动模糊格

## 评测
- 真实物理（免标注）：`train_eval_report.json: eval.real_physical`（MAD/暗区比可信；平面角指标在高误差区退化，见 K-3 分析）
- 合成留出 abs_rel：{synth_abs_rel}
"""
    (out / "DATA_CARD.md").write_text(card, encoding="utf-8")
    print(f"done in {report['train_seconds']}s -> {out}", flush=True)
    print(f"real_physical={real_eval}", flush=True)
    print(f"synth_abs_rel={synth_abs_rel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
