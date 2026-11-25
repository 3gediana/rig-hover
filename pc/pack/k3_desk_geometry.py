# -*- coding: utf-8 -*-
"""pc/pack/k3_desk_geometry.py - O-005/K-3 falsification.

Question: does adding rendered analytical desktop scenes to training improve
real-frame geometry — desk plane angle error and the dark-region depth-collapse
artifact — compared to training on real pseudo-GT only?

Design (dev-only, zero sealed contact):
  real train: 4 dev takes of 2026-11-07 (40 frames each, pseudo-GT from the
              same _build_real_pseudo_gt the shipped pipeline uses)
  real test : 65a54d dev take (cross-take)
  arm A: real only
  arm B: 600 rendered analytical scenes + the same real frames (80/20-ish mix)
  same HoverMonoDepthNet, same epochs/batch/lr/seed.

Metrics on held-out real frames (no labels needed — physical truths):
  plane_angle_err_deg: fit a plane to predicted desk-region world points
      (measured camera, Z=0 is the real desk); angle of the fitted normal vs vertical
  desk_depth_mad_mm: median absolute deviation of fitted-plane residuals — flatness
  dark_collapse_ratio: median predicted depth in the darkest 5% desk-region pixels
      vs median in bright desk pixels; 1.0 = no collapse, <<1 = black-as-cliff artifact
Output: PACK_ROOT/findings/raw/k3_desk_geometry_YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

END_SIDE = paths.REPO / "pc" / "end_side"
sys.path.insert(0, str(END_SIDE))

DEV_TRAIN_TAKES = [
    "robot_20261107_204337_5ca36d",
    "robot_20261107_212119_67f0c1",
    "robot_20261107_213314_5c0158",
    "robot_20261107_214950_1f30d1",
]
DEV_TEST_TAKE = "robot_20261115_150045_65a54d"


def sample_dev_real(frames_per_take: int = 40):
    """Dev-only equivalent of train_mono_depth.sample_real_robot_frames (no sealed contact)."""
    from train_mono_depth import _build_real_pseudo_gt
    from mono_depth_model import INPUT_H, INPUT_W, metric_m_to_inv_depth_01
    out = {}
    for take in DEV_TRAIN_TAKES + [DEV_TEST_TAKE]:
        mp4 = paths.REPO / "dataset" / "takes" / take / "take.mp4"
        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(15, max(16, total - 15), num=frames_per_take, dtype=int)
        raw = []
        for fid in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb96 = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                raw.append((int(fid), rgb96))
        cap.release()
        grays = np.stack([0.299 * r[..., 0] + 0.587 * r[..., 1] + 0.114 * r[..., 2] for _, r in raw], axis=0)
        bg = np.median(grays, axis=0).astype(np.float32)
        rows = []
        for fid, rgb96 in raw:
            d_m, e_m = _build_real_pseudo_gt(rgb96, bg)
            rows.append({"rgb_nhwc": rgb96, "depth_m": d_m, "inv_depth_01": metric_m_to_inv_depth_01(d_m),
                         "elev_mask": e_m, "take": take, "frame_idx": fid})
        out[take] = rows
    return out


def train_arm(samples, device, epochs=6, batch=32, seed=0, lr=2.6e-3):
    import torch
    from mono_depth_model import HoverMonoDepthNet, inv_depth_01_to_metric_m
    from train_mono_depth import compute_multiscale_grad_loss
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HoverMonoDepthNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.tensor([2.4], device=device)

    def batch_xy(rows):
        idx = np.random.randint(0, len(rows), size=min(batch, len(rows)))
        rgb = torch.from_numpy(np.stack([rows[i]["rgb_nhwc"] for i in idx])).permute(0, 3, 1, 2).float().to(device)
        inv = torch.from_numpy(np.stack([rows[i]["inv_depth_01"] for i in idx]))[:, None].float().to(device)
        dep = torch.from_numpy(np.stack([rows[i]["depth_m"] for i in idx]))[:, None].float().to(device)
        elv = torch.from_numpy(np.stack([rows[i]["elev_mask"] for i in idx]))[:, None].float().to(device)
        return rgb, inv, dep, elv

    steps = epochs * max(1, len(samples) // batch)
    for _st in range(steps):
        rgb, inv, dep, elv = batch_xy(samples)
        logits = model.forward_logits(rgb)
        probs = torch.sigmoid(logits)
        pred_inv = probs[:, 0:1]
        pred_elev_logit = logits[:, 1:2]
        l_inv = torch.abs(pred_inv - inv).mean()
        pred_d = inv_depth_01_to_metric_m(pred_inv.squeeze(1))
        gt_d = dep.squeeze(1)
        l_rel = (torch.abs(pred_d - gt_d) / torch.clamp(gt_d, min=0.15)).mean()
        l_grad = compute_multiscale_grad_loss(pred_inv.squeeze(1), inv.squeeze(1), torch.ones(rgb.shape[0], device=device))
        l_elev = torch.nn.functional.binary_cross_entropy_with_logits(pred_elev_logit, elv, pos_weight=pos_weight)
        loss = l_inv + 0.4 * l_rel + 0.6 * l_grad + 0.5 * l_elev
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def predict_depth(model, rgb96, device):
    import torch
    from mono_depth_model import inv_depth_01_to_metric_m
    with torch.no_grad():
        x = torch.from_numpy(rgb96[None]).permute(0, 3, 1, 2).float().to(device)
        probs = torch.sigmoid(model.forward_logits(x))
        inv = probs[0, 0].cpu().numpy()
        elev = probs[0, 1].cpu().numpy()
    return inv_depth_01_to_metric_m(inv), elev


def eval_real_geometry(model, test_rows, device) -> dict:
    """Label-free physical metrics on held-out real frames, using the measured camera."""
    import robot_truth as rt
    f_px = rt.F_PX
    h_cam = rt.CAM_HEIGHT_M
    angles, mads, dark_ratios = [], [], []
    for row in test_rows[:25]:
        d, elev = predict_depth(model, row["rgb_nhwc"], device)
        h, w = d.shape
        scale = 240.0 / h  # depth-map resolution -> 240px intrinsics
        yy, xx = np.mgrid[0:h,0:w].astype(np.float32)
        # world unproject with measured camera (resize-normalised intrinsics); camera looks ahead, desk below
        fx = f_px / scale
        cx = cy = 119.5 / scale
        # desk region: lower half of image, low elevation mask confidence
        mask = (yy > h * 0.55) & (d < 1.2) & (d > 0.05)
        if mask.sum() < 200:
            continue
        Z = h_cam  # camera height; points lie on rays: depth d along ray
        # simple pinhole: ray directions
        dx = (xx - cx) / fx
        dy = (yy - cy) / fx
        # forward-looking camera, y down. depth along forward axis; world point = (dx, dy, 1) * d_norm
        norm = np.sqrt(dx ** 2 + dy ** 2 + 1.0)
        px = dx[mask] * d[mask] / norm[mask]
        py = dy[mask] * d[mask] / norm[mask]
        pz = d[mask] / norm[mask]
        pts = np.stack([px, py, pz], axis=1)
        # fit plane pz = a*px + b*py + c — desk should be a=0,b=tan(pitch)~small; angle of normal vs vertical-forward plane
        A = np.concatenate([pts[:, :2], np.ones((len(pts), 1), np.float32)], axis=1)
        try:
            sol, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b, c = sol
        nrm = np.array([a, b, -1.0])
        nrm /= np.linalg.norm(nrm)
        vertical = np.array([0, -1.0, 0])  # desk normal in camera frame for level camera points downward
        ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(nrm, vertical)), 0, 1))))
        resid = pts[:, 2] - (A @ sol)
        mad = float(np.median(np.abs(resid - np.median(resid)))) * 1000.0
        angles.append(ang)
        mads.append(mad)
        # dark collapse: darkest 5% desk pixels vs bright desk pixels
        gray = cv2.resize(row["rgb_nhwc"].mean(axis=-1), (w, h), interpolation=cv2.INTER_AREA)
        desk = mask
        if desk.sum() > 300:
            gvals = gray[desk]
            thr = np.quantile(gvals, 0.05)
            dark = desk & (gray <= thr)
            bright = desk & (gray > np.quantile(gvals, 0.5))
            if dark.sum() > 20 and bright.sum() > 100:
                dark_ratios.append(float(np.median(d[dark]) / max(np.median(d[bright]), 1e-3)))
    return {
        "plane_angle_err_deg": round(float(np.median(angles)), 2) if angles else None,
        "desk_depth_mad_mm": round(float(np.median(mads)), 1) if mads else None,
        "dark_collapse_ratio": round(float(np.median(dark_ratios)), 3) if dark_ratios else None,
        "n_frames_evaluated": len(angles),
    }


def main() -> int:
    t0 = time.perf_counter()
    import torch
    from train_mono_depth import render_analytical_desktop_scene
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("sampling dev real frames...", flush=True)
    real = sample_dev_real(frames_per_take=40)
    real_tr = [r for take in DEV_TRAIN_TAKES for r in real[take]]
    real_te = real[DEV_TEST_TAKE]
    print(f"real train={len(real_tr)} test={len(real_te)}", flush=True)

    print("rendering analytical scenes...", flush=True)
    rng = np.random.default_rng(20261125)
    synth = [render_analytical_desktop_scene(rng) for _ in range(600)]
    # unify keys: synth rows carry rgb/inv/depth/elev under different names; adapt
    syn_rows = []
    for s in synth:
        syn_rows.append({
            "rgb_nhwc": s["rgb"] if "rgb" in s else s["rgb_nhwc"],
            "depth_m": s["depth_m"],
            "inv_depth_01": s["inv_depth_01"],
            "elev_mask": s["elev_mask"] if "elev_mask" in s else s["elev"],
        })

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}
    mix = syn_rows + real_tr
    np.random.default_rng(0).shuffle(mix)
    for name, rows in (("A_real_only", real_tr), ("B_synth_plus_real", mix)):
        print(f"training {name} on {len(rows)} samples...", flush=True)
        model = train_arm(rows, device, epochs=6, batch=32)
        ev = eval_real_geometry(model, real_te, device)
        report["arms"][name] = {"n_train": len(rows), "eval": ev}
        print(f"{name}: {ev}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a, b = report["arms"]["A_real_only"]["eval"], report["arms"]["B_synth_plus_real"]["eval"]
    da = (a["plane_angle_err_deg"] or 999) - (b["plane_angle_err_deg"] or 999)
    report["plane_angle_gain_deg"] = round(da, 2)
    if b["plane_angle_err_deg"] is not None and b["plane_angle_err_deg"] < a["plane_angle_err_deg"] * 0.8 and b["dark_collapse_ratio"] > a["dark_collapse_ratio"]:
        verdict = "U2-go: rendered desktop scenes improve both plane geometry and dark-region behaviour"
    elif b["plane_angle_err_deg"] is not None and b["plane_angle_err_deg"] < a["plane_angle_err_deg"] * 0.9:
        verdict = "U1-weak: plane geometry improves, dark-collapse unresolved"
    else:
        verdict = "U0-no-gain: rendered scenes do not improve real-frame geometry at this scale"
    report["verdict"] = verdict
    report["limitations"] = [
        "real pseudo-GT is heuristic background subtraction — noisy supervision in both arms",
        "plane-angle truth assumes the desk is horizontal and the camera near-level (measured rigidity, pitch unknown)",
        "table-edge distance error cannot be measured offline without edge GT — marked for T4",
    ]
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    out = paths.RAW / f"k3_desk_geometry_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {verdict}", flush=True)
    print(f"wrote {out} in {report['runtime_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
