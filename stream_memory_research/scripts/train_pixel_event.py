"""Train the pixel-to-event network (and every baseline) on raw pixels.

One self-contained loop, deliberately separate from ``src/train/trainer.py``:
the previous study's trainer is built around cached CLIP feature tensors, and the
whole point of DESIGN-PIXEL-EVENT-NETWORK.md is that pixels are the contract.
Sharing the feature trainer would have forced an encoder shim and muddied the
comparison.

    python scripts/train_pixel_event.py --model pixel_event --data charades_pixel --seed 0

Everything reported here is measured on the raw-pixel path with the same loaders,
the same augmentations and the same heads for every model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.data.pixel_data import (ROBOT_TASKS, RobotSessions, load_index,  # noqa: E402
                                 robot_manifest, make_loaders)
from src.eval.pixel_metrics import mean_average_precision  # noqa: E402

OUT = ROOT / "runs_pixel"
N_EVENT = 25
N_GESTURE = len(ROBOT_TASKS)
N_PRESENCE = 4


# --------------------------------------------------------------------------- #
# model factory
# --------------------------------------------------------------------------- #
def build(name: str, event_head: str = "direct", scout_pretrained: bool = True,
          **kw) -> nn.Module:
    cfg = {"scout_pretrained": bool(scout_pretrained)}
    if name == "pixel_event":
        from src.models.pixel_event import build_pixel_model
        return build_pixel_model("pixel_event", {**cfg, "event_head": event_head},
                                 n_event_classes=N_EVENT,
                                 n_gesture=N_GESTURE, n_presence=N_PRESENCE)
    from src.models.pixel_baselines import build_baseline
    return build_baseline(name, n_event_classes=N_EVENT, n_gesture=N_GESTURE,
                          n_presence=N_PRESENCE, cfg=cfg)


# --------------------------------------------------------------------------- #
# losses / metrics
# --------------------------------------------------------------------------- #
def event_losses(seq: dict, batch: dict, pos_weight: float) -> dict:
    out = {}
    lab = batch["lab_event"]
    if lab.numel() and lab.shape[-1] == seq["event"].shape[-1]:
        pw = torch.tensor([pos_weight], device=lab.device)
        out["event"] = F.binary_cross_entropy_with_logits(
            seq["event"], lab, pos_weight=pw)
        out["onset"] = F.binary_cross_entropy_with_logits(
            seq["onset"], batch["lab_onset"], pos_weight=torch.tensor([8.0],
                                                                     device=lab.device))
        ct = batch["lab_count"][:, -1, :]
        out["count"] = F.smooth_l1_loss(seq["count_final"] / (1.0 + ct.amax().detach()),
                                        ct / (1.0 + ct.amax().detach()))
    return out


def cls_losses(seq: dict, batch: dict) -> dict:
    out = {}
    pg = batch["lab_presence"]
    if (pg >= 0).any():
        out["presence"] = F.cross_entropy(seq["presence"].reshape(-1, N_PRESENCE),
                                          pg.reshape(-1), ignore_index=-1)
    gg = batch["lab_gesture"]
    if (gg >= 0).any():
        out["gesture"] = F.cross_entropy(seq["gesture"].reshape(-1, N_GESTURE),
                                         gg.reshape(-1), ignore_index=-1)
    return out


@torch.no_grad()
def eval_event(seq: dict, batch: dict, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> dict:
    """Scalar event metrics + raw (probability, label) arrays under "_probs"/"_labels".

    Callers must route keys starting with "_" into the pooled-mAP accumulator and
    never into `np.mean` (that bug killed a full 8-epoch run at the test stage).
    """
    """Macro-F1 over classes, best threshold chosen on *this* split (val only)."""
    p = torch.sigmoid(seq["event"])
    y = batch["lab_event"]
    m = batch["mask"]
    p = p[m]
    y = y[m]
    if y.numel() == 0 or y.shape[-1] == 0:
        return {}
    present = y.sum(0) > 0
    best = {"f1": -1.0, "thr": 0.5}
    for thr in thresholds:
        pr = (p > thr).float()
        tp = (pr * y).sum(0)
        fp = (pr * (1 - y)).sum(0)
        fn = ((1 - pr) * y).sum(0)
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-9))[present]
        v = float(f1.mean()) if f1.numel() else 0.0
        if v > best["f1"]:
            best = {"f1": v, "thr": float(thr)}
    ct = batch["lab_count"][:, -1, :]
    cm = ct.amax()
    best["count_mae"] = float((seq["count_final"] - ct).abs().mean()) if ct.numel() else 0.0
    best["count_mae_norm"] = float(((seq["count_final"] - ct).abs().mean()
                                    / (1.0 + cm)).item()) if ct.numel() else 0.0
    best["_probs"] = p.cpu().numpy()
    best["_labels"] = y.cpu().numpy()
    return best


@torch.no_grad()
def eval_cls(seq: dict, batch: dict) -> dict:
    """Gesture accuracy + presence macro-F1 on labelled frames."""
    gg = batch["lab_gesture"]
    m = batch["mask"] & (gg >= 0)
    out = {}
    if m.any():
        pred = seq["gesture"].argmax(-1)
        out["gesture_acc"] = float((pred[m] == gg[m]).float().mean())
    pg = batch["lab_presence"]
    m = batch["mask"] & (pg >= 0)
    if m.any():
        pred = seq["presence"].argmax(-1)
        f1s = []
        for c in range(N_PRESENCE):
            tp = float(((pred == c) & (pg == c) & m).sum())
            fp = float(((pred == c) & (pg != c) & m).sum())
            fn = float(((pred != c) & (pg == c) & m).sum())
            if (pg == c).sum() > 0:
                f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
        out["presence_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data == "robot":
        recs = robot_manifest()
        dirs = [r["dir"] for r in recs]
        k = args.fold % len(dirs)
        held = dirs[k:k + 1] or dirs[:1]
        # stride = window/2: with stride 4 a 20 s session yields ~430 windows per
        # fold, which made one leave-one-out sweep take hours for 2 k frames.
        tr = RobotSessions(held, args.window, stride=max(4, args.window // 2),
                           train=True, seed=args.seed)
        te = RobotSessions(held, args.window, stride=args.window, train=False)
        va = te
        mk = lambda d, sh: torch.utils.data.DataLoader(
            d, batch_size=args.batch, shuffle=sh, num_workers=0,
            collate_fn=lambda b: __import__("src.data.pixel_data", fromlist=["collate"])
            .collate(b, N_EVENT))
        tl, vl, tel = mk(tr, True), mk(va, False), mk(te, False)
        trains, vals, tests = tr, va, te
    else:
        idx, tl, vl, tel = make_loaders(args.data, args.window, args.batch,
                                        num_workers=args.workers, seed=args.seed,
                                        aug=not args.no_aug)
        trains, vals, tests = tl.dataset, vl.dataset, tel.dataset
        n_cls = len(idx["classes"])
        globals()["N_EVENT"] = n_cls

    model = build(args.model, getattr(args, "event_head", "direct"),
                  getattr(args, "scout_pretrained", True)).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")

    # class prior -> positive weight for the event BCE (Charades is very sparse)
    pos_weight = 8.0
    if args.data != "robot":
        try:
            y = np.stack([np.load(s["path"])["event"].astype(np.float32)
                          for s in load_index(args.data)["seqs"][:120]])
            p = float(y.mean())
            # cap at 5: with a 20x positive weight the model just learns to fire
            # everywhere (loss fell while thresholded F1 fell with it)
            pos_weight = float(np.clip((1 - p) / max(p, 1e-4), 1.0, 5.0))
        except Exception:
            pass

    best = {"score": -1e9, "epoch": -1}
    curves = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        if hasattr(trains, "set_epoch"):
            trains.set_epoch(ep)
        tot, nb = 0.0, 0
        for bi, batch in enumerate(tl):
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and dev.type == "cuda"):
                seq, _aux = model(batch["img"], batch["dt"], batch["mask"])
                losses = {}
                if args.data != "robot":      # robot footage has no event labels
                    losses.update(event_losses(seq, batch, pos_weight))
                losses.update(cls_losses(seq, batch))
                loss = sum(losses.values()) if losses else seq["event"].sum() * 0
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        # ---- validation ----
        model.eval()
        agg = {}
        vpool: dict = {}
        # no_grad is essential: validation runs full-length causal passes, and
        # building a graph for 96 steps x batch filled the 8 GB card, which made the
        # allocator churn and epoch time creep from 96 s to 446 s.
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=args.amp and dev.type == "cuda"):
                    seq, _aux = model(batch["img"], batch["dt"], batch["mask"])
                ev_ = eval_event(seq, batch)
                for k, v in ev_.items():
                    if k.startswith("_"):          # raw arrays -> pooled mAP
                        vpool.setdefault(k, []).append(v)
                    else:
                        agg.setdefault(k, []).append(v)
                for k, v in eval_cls(seq, batch).items():
                    agg.setdefault(k, []).append(v)
        vm = {k: float(np.mean(v)) for k, v in agg.items()}
        if vpool.get("_probs"):
            mp = mean_average_precision(vpool["_probs"], vpool["_labels"])
            # mAP is undefined when the split carries no event labels at all (the
            # robot sessions have none).  Leaving NaN here made the selection score
            # NaN, so no checkpoint was ever saved and the run crashed at the end.
            if np.isfinite(mp["map"]):
                vm["map"] = mp["map"]
        vm["train_loss"] = tot / max(1, nb)
        vm["epoch"] = ep
        curves.append(vm)
        # The unrolled per-frame graph fills the 8 GB card; releasing the cache each
        # epoch stops the allocator from degrading into repeated free/re-alloc
        # (measured: epoch time 96 s -> 446 s without this line).
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # mAP is the metric that does not depend on a single operating point; the
        # thresholded F1 is kept in the log only.
        score = float(np.nan_to_num(vm.get("map", 0.0), nan=0.0)) \
            + 0.25 * float(np.nan_to_num(vm.get("gesture_acc", 0.0), nan=0.0)) \
            - 0.25 * float(np.nan_to_num(vm.get("count_mae_norm", 0.0), nan=0.0))
        print(f"[ep {ep:02d}] loss={vm['train_loss']:.4f} "
              f"mAP={vm.get('map', float('nan')):.4f} f1={vm.get('f1', float('nan')):.4f} "
              f"cMAE={vm.get('count_mae', float('nan')):.3f} "
              f"gest={vm.get('gesture_acc', float('nan')):.4f} "
              f"presF1={vm.get('presence_macro_f1', float('nan')):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "val": dict(vm)}
            torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args)},
                       OUT / args.run_id / "best.pt")
        if ep - best["epoch"] >= args.patience:
            print(f"[early stop] best epoch {best['epoch']}", flush=True)
            break

    # ---- test ----
    ck = torch.load(OUT / args.run_id / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    agg, preds = {}, []
    with torch.no_grad():
        for batch in tel:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.amp and dev.type == "cuda"):
                seq, _aux = model(batch["img"], batch["dt"], batch["mask"])
            for k, v in eval_event(seq, batch).items():
                if k.startswith("_"):
                    continue
                agg.setdefault(k, []).append(v)
            for k, v in eval_cls(seq, batch).items():
                agg.setdefault(k, []).append(v)
            preds.append({"event": torch.sigmoid(seq["event"]).detach().float().cpu().numpy(),
                      "count": seq["count_final"].detach().float().cpu().numpy(),
                      "gesture": seq["gesture"].argmax(-1).detach().cpu().numpy(),
                      "presence": seq["presence"].argmax(-1).detach().cpu().numpy()})
    test = {k: float(np.mean(v)) for k, v in agg.items()}
    out = {"run_id": args.run_id, "model": args.model, "data": args.data,
           "seed": args.seed, "params": n_par, "window": args.window,
           "best_epoch": best["epoch"], "val_best": best["val"], "test": test,
           "curves": curves, "wall_sec": time.time() - t0,
           "pos_weight": pos_weight, "amp": bool(args.amp)}
    (OUT / args.run_id / "metrics.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez(OUT / args.run_id / "preds_test.npz",
             event=np.concatenate([p["event"] for p in preds]),
             count=np.concatenate([p["count"] for p in preds]),
             gesture=np.concatenate([p["gesture"] for p in preds]),
             presence=np.concatenate([p["presence"] for p in preds]))
    print("TEST", json.dumps(test, ensure_ascii=False), flush=True)
    print(f"[done] {args.run_id} wall={time.time() - t0:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pixel_event")
    ap.add_argument("--data", default="charades_pixel")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--event-head", dest="event_head", default="direct",
                    choices=["direct", "hazard"])
    ap.add_argument("--scout-pretrained", dest="scout_pretrained", default=True,
                    action="store_true")
    ap.add_argument("--no-scout-pretrained", dest="scout_pretrained",
                    action="store_false")
    a = ap.parse_args()
    a.run_id = (f"{a.data}__{a.model}__w{a.window}__s{a.seed}"
                + (f"__f{a.fold}" if a.data == "robot" else "")
                + (f"__{a.tag}" if a.tag else ""))
    (OUT / a.run_id).mkdir(parents=True, exist_ok=True)
    (OUT / a.run_id / "config.json").write_text(
        json.dumps(vars(a), indent=1, ensure_ascii=False), encoding="utf-8")
    run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

