"""Evaluation for the pixel-to-event network and every baseline (spec §9).

Runs the trained model causally over whole test videos, then reports the metrics
the design document asks for:

    per-frame phase macro-F1 (threshold swept on VAL only)
    event onset precision / recall / F1 / latency at +-tol frames
    exact-count accuracy, count MAE, over/under-count rates
    calibration: Brier + ECE on the event probabilities
    presence macro-F1 / gesture accuracy (robot sessions)
    the same numbers under frame drops, burst drops, irregular dt and occlusion
      (spec §9.5 "joint stress test")

Two engineering notes that matter for the numbers being affordable and honest:

* videos are **batched** (padded to the longest sequence in the batch, with the
  mask carried through) instead of one sequence per forward -- single-sequence
  forwards made evaluation ~34 min per run because the per-frame loop is
  launch-bound, and 12 runs x 20 conditions would have taken 7 h;
* the class threshold is chosen on the val split and applied unchanged to test.

    python scripts/eval_pixel_event.py --run charades_pixel__pixel_event__w16__s0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.data.pixel_data import (ROBOT_TASKS, RobotSessions,  # noqa: E402
                                 load_index, load_robot_frames, robot_manifest)
from src.eval.pixel_metrics import (abstention_coverage, boundary_prf,  # noqa: E402
                                    calibration_metrics, count_metrics,
                                    mean_average_precision, presence_metrics)

OUT = ROOT / "runs_pixel"
N_GESTURE = len(ROBOT_TASKS)
#: (stream course, photometric nuisance).  Ten conditions keep the stress sweep
#: affordable; each one is a real degradation from spec §6/§9.5.
STRESS = [("clean", "none"), ("drop25", "none"), ("drop50", "none"),
          ("burst", "none"), ("irregular", "none"), ("clean", "occlude"),
          ("drop25", "occlude"), ("clean", "blur"), ("clean", "dark"),
          ("burst", "occlude")]


def build(name: str, C: int) -> torch.nn.Module:
    if name in ("pixel_event",):
        from src.models.pixel_event import build_pixel_model
        return build_pixel_model("pixel_event", None, n_event_classes=C,
                                 n_gesture=N_GESTURE, n_presence=4)
    from src.models.pixel_baselines import build_baseline
    return build_baseline(name, n_event_classes=C, n_gesture=N_GESTURE, n_presence=4)


# --------------------------------------------------------------------------- #
# stream perturbations (spec §6 / §9.5)
# --------------------------------------------------------------------------- #
def stress_course(frames: np.ndarray, t: np.ndarray, kind: str, rng) -> tuple:
    T = len(t)
    if kind == "clean":
        return frames, t
    if kind == "irregular":
        step = max(1e-3, float(np.mean(np.diff(t))) if T > 1 else 0.1)
        t2 = t + rng.uniform(-0.12, 0.12, T) * step
        t2 = np.maximum.accumulate(t2 + 1e-4)
        return frames, (t2 - t2[0]).astype(np.float32)
    keep = np.ones(T, bool)
    if kind == "drop25":
        keep = rng.random(T) > 0.25
    elif kind == "drop50":
        keep = rng.random(T) > 0.50
    elif kind == "burst":
        L = max(3, T // 5)
        s = int(rng.integers(T // 4, max(T // 4 + 1, T - L - T // 4)))
        keep[s:s + L] = False
    keep[0] = True
    if keep.sum() < 4:
        return frames, t
    return frames[keep], t[keep]


def photometric(frames: np.ndarray, kind: str, rng) -> np.ndarray:
    if kind == "none":
        return frames
    f = frames.astype(np.float32)
    if kind == "blur":
        f[1:] = 0.5 * f[1:] + 0.5 * f[:-1]
    elif kind == "occlude":
        H, W = f.shape[1], f.shape[2]
        h, w = int(H * 0.35), int(W * 0.35)
        y, x = rng.integers(0, H - h), rng.integers(0, W - w)
        f[:, y:y + h, x:x + w] = f.mean()
    elif kind == "dark":
        f = f * 0.6
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_batch(model, seqs, device, res: int = 160) -> dict:
    """``seqs`` = list of (frames uint8 [T,H,W], t float32 [T]) -> list of outputs."""
    B = len(seqs)
    L = max(len(t) for _f, t in seqs)
    imgs = torch.zeros(B, L, 1, res, res)
    dt = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, (f, t) in enumerate(seqs):
        n = len(t)
        imgs[i, :n, 0] = torch.from_numpy(f.astype(np.float32) / 255.0)
        if n > 1:
            dt[i, 1:n] = torch.from_numpy(np.clip(np.diff(t), 0.0, 5.0))
        mask[i, :n] = True
    imgs, dt, mask = imgs.to(device), dt.to(device), mask.to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        seq, _aux = model(imgs, dt, mask)
    out = {"event": torch.sigmoid(seq["event"]).float().cpu().numpy(),
           "onset": torch.sigmoid(seq["onset"][..., 0]).float().cpu().numpy(),
           "count": seq["count_final"].float().cpu().numpy(),
           "presence": seq["presence"].argmax(-1).cpu().numpy(),
           "gesture": seq["gesture"].argmax(-1).cpu().numpy()}
    return [{k: v[i, :len(t)] for k, v in out.items()} for i, (_f, t) in enumerate(seqs)]


def gt_on_grid(t2: np.ndarray, segs, classes) -> tuple:
    C = len(classes)
    ev = np.zeros((len(t2), C), np.float32)
    on = np.zeros(len(t2), np.float32)
    if len(t2) == 0:
        return ev, on, np.zeros(C, np.float32)
    prev = np.concatenate([[t2[0] - 1e-3], t2[:-1]])
    cnt = np.zeros(C, np.float32)
    for seg in (segs or []):
        j = classes.index(seg[0]) if seg[0] in classes else -1
        if j < 0:
            continue
        s0, e0 = float(seg[1]), float(seg[2])
        ev[:, j] = np.maximum(ev[:, j], ((t2 >= s0) & (t2 < e0)).astype(np.float32))
        on = np.maximum(on, ((prev < s0) & (t2 >= s0)).astype(np.float32))
        cnt[j] += float(e0 <= t2.max())
    return ev, on, cnt


def macro_f1(probs, gts, present, thr, per_seq=False):
    tp = np.zeros(len(present)); fp = np.zeros(len(present)); fn = np.zeros(len(present))
    for p, y in zip(probs, gts):
        pr = p > thr; yy = y > 0.5
        tp += (pr & yy).sum(0); fp += (pr & ~yy).sum(0); fn += (~pr & yy).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    out = {"macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
           "precision": float((tp / (tp + fp + 1e-9))[present].mean()) if present.any() else float("nan"),
           "recall": float((tp / (tp + fn + 1e-9))[present].mean()) if present.any() else float("nan"),
           "per_class_f1": f1.tolist()}
    if per_seq:
        # per-video macro-F1 over the classes that are present in the corpus, so
        # models can be compared with a paired bootstrap on identical videos
        per = []
        for p, y in zip(probs, gts):
            pr = p > thr
            yy = y > 0.5
            tpv = (pr & yy).sum(0).astype(float)
            fpv = (pr & ~yy).sum(0).astype(float)
            fnv = (~pr & yy).sum(0).astype(float)
            f1v = 2 * tpv / (2 * tpv + fpv + fnv + 1e-9)
            per.append(float(f1v[present].mean()) if present.any() else float("nan"))
        out["per_seq_f1"] = per
    return out


def onsets_from_prob(p, thr, min_gap=2):
    idx = np.where(p > thr)[0]
    if len(idx) == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= min_gap:
            keep.append(i)
    return np.asarray(keep)


# --------------------------------------------------------------------------- #
def eval_charades(model, index: dict, device, seed: int = 0, bs: int = 8,
                  only_split: str = "all") -> dict:
    classes, C = index["classes"], len(index["classes"])
    raw = {}
    for sp in ("val", "test"):
        recs = [s for s in index["seqs"] if s["split"] == sp]
        raw[sp] = [(r, np.load(r["path"])) for r in recs]

    def run_condition(sp, course, photo):
        rng = np.random.default_rng(abs(hash((sp, course, photo, seed))) % (2 ** 31))
        prepared, gts = [], []
        for r, z in raw[sp]:
            fr = np.asarray(z["frames"]); t = np.asarray(z["t"], np.float32)
            f2, t2 = stress_course(fr, t, course, rng)
            f2 = photometric(f2, photo, rng)
            ev, on, cnt = gt_on_grid(t2, r.get("segs") or [], classes)
            if len(t2) == 0:
                continue
            prepared.append((f2, t2))
            gts.append({"ev": ev, "on": on, "cnt": cnt, "seq_id": r["seq_id"]})
        outs = []
        for i in range(0, len(prepared), bs):
            outs.extend(forward_batch(model, prepared[i:i + bs], device))
        return outs, gts

    # ---- val: pick the threshold ----
    vouts, vgts = run_condition("val", "clean", "none")
    present = np.zeros(C, bool)
    for g in vgts:
        present |= (g["ev"].sum(0) > 0)
    best = (-1.0, 0.3)
    for thr in np.arange(0.1, 0.91, 0.04):
        v = macro_f1([o["event"] for o in vouts], [g["ev"] for g in vgts], present,
                     float(thr))["macro_f1"]
        if np.isfinite(v) and v > best[0]:
            best = (v, float(thr))
    thr = best[1]

    def summarise(outs, gts, with_per_seq=False):
        ph = macro_f1([o["event"] for o in outs], [g["ev"] for g in gts], present, thr,
                      per_seq=with_per_seq)
        # mAP is the primary event-detection number: it does not depend on the
        # single operating point picked by the threshold sweep.
        ph["map"] = mean_average_precision([o["event"] for o in outs],
                                           [g["ev"] for g in gts], present)["map"]
        bnd = {"precision": [], "recall": [], "f1": [], "lat": []}
        for o, g in zip(outs, gts):
            r = boundary_prf(onsets_from_prob(o["onset"], 0.5), np.where(g["on"] > 0.5)[0],
                             tolerance_frames=2)
            bnd["precision"].append(r["precision"]); bnd["recall"].append(r["recall"])
            bnd["f1"].append(r["f1"])
            if not np.isnan(r["latency_signed_mean"]):
                bnd["lat"].append(r["latency_signed_mean"])
        # Per-(video, class) counts are the meaningful unit here: Charades has ~0-2
        # instances of a given class per video, so a total-across-classes error is
        # dominated by class count rather than by counting ability.  Both are reported.
        cp = np.stack([o["count"] for o in outs])
        cg = np.stack([g["cnt"] for g in gts])
        cm = count_metrics(np.rint(cp).astype(int).ravel(), np.rint(cg).astype(int).ravel())
        cm_total = count_metrics(np.rint(cp.sum(1)).astype(int),
                                 np.rint(cg.sum(1)).astype(int))
        P = np.concatenate([o["event"].ravel() for o in outs])
        Y = np.concatenate([g["ev"].ravel() for g in gts]).astype(int)
        cal = calibration_metrics(P, Y)
        out = {"phase": ph,
               "boundary": {k: (float(np.mean(v)) if v else float("nan"))
                            for k, v in bnd.items()},
               "count": cm, "count_total": cm_total,
               "calibration": {"brier": cal["brier"], "ece": cal["ece"], "mce": cal["mce"]}}
        if with_per_seq:
            out["per_seq"] = [{"seq_id": g["seq_id"], "f1": f}
                              for g, f in zip(gts, ph.pop("per_seq_f1", []))]
        return out

    res = {"n_classes": C, "classes_present": int(present.sum()), "thr": thr,
           "val": summarise(vouts, vgts), "stress": {}}
    for course, photo in STRESS:
        key = f"{course}|{photo}"
        if only_split != "all" and key not in ("clean|none",):
            continue
        o, g = run_condition("test", course, photo)
        res["stress"][key] = summarise(o, g, with_per_seq=(key == "clean|none"))
        if key == "clean|none":
            res["test"] = res["stress"][key]
    return res


# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_robot(model, device, seed: int = 0) -> dict:
    dirs = [r["dir"] for r in robot_manifest()]
    folds = []
    for held in dirs:
        ds = RobotSessions([held], window=32, stride=16, train=False)
        if len(ds) == 0:
            continue
        seqs, gts = [], []
        for i in range(len(ds)):
            b = ds[i]
            fr = (b["img"][:, 0].numpy() * 255).astype(np.uint8)
            t = np.arange(len(fr), dtype=np.float32) / ds.fps
            seqs.append((fr, t))
            gts.append((int(b["lab_gesture"][0]), int(b["lab_presence"][0])))
        outs = []
        for i in range(0, len(seqs), 8):
            outs.extend(forward_batch(model, seqs[i:i + 8], device))
        acc = [float((o["gesture"] == g[0]).mean()) for o, g in zip(outs, gts)]
        sp = np.concatenate([o["presence"] for o in outs])
        sg = np.concatenate([np.full(len(o["presence"]), g[1]) for o, g in zip(outs, gts)])
        conf = np.concatenate([torch.softmax(torch.from_numpy(o["gesture"]).float(),
                                            -1).max(-1).values.numpy() for o in outs])
        corr = np.concatenate([o["gesture"] == g[0] for o, g in zip(outs, gts)])
        folds.append({"held_out": held, "gesture_acc": float(np.mean(acc)),
                      "presence_f1": presence_metrics(sp, sg).get("macro_f1"),
                      "abstention": abstention_coverage(conf, corr)})
    out = {"folds": folds}
    for k in ("gesture_acc", "presence_f1"):
        v = [f[k] for f in folds if f.get(k) is not None
             and not (isinstance(f[k], float) and np.isnan(f[k]))]
        out[k + "_mean"] = float(np.mean(v)) if v else float("nan")
        out[k + "_std"] = float(np.std(v)) if v else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="charades_pixel")
    ap.add_argument("--index", default="charades_pixel")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--held", default=None,
                    help="robot: evaluate only this held-out session")
    a = ap.parse_args()
    d = OUT / a.run
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = load_index(a.index)
    C = len(index["classes"])
    model = build(cfg["model"], C).to(device)
    ck = torch.load(d / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    res = (eval_robot(model, device, a.seed, a.held) if a.data == "robot"
           else eval_charades(model, index, device, a.seed, a.bs))
    res.update({"run": a.run, "model": cfg["model"],
                "params": int(sum(p.numel() for p in model.parameters())),
                "best_epoch": int(ck.get("epoch", -1))})
    (d / "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                 encoding="utf-8")
    t = res.get("test") or {}
    print(json.dumps({"run": a.run, "thr": res.get("thr"),
                      "phase_f1": (t.get("phase") or {}).get("macro_f1"),
                      "onset_f1": (t.get("boundary") or {}).get("f1"),
                      "count_mae": (t.get("count") or {}).get("mae"),
                      "ece": (t.get("calibration") or {}).get("ece"),
                      "gesture_acc": res.get("gesture_acc_mean")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
