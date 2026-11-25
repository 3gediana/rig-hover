# -*- coding: utf-8 -*-
"""pc/pack/adversarial_auc.py - O-004 adversarial validation + canary.

Question: after the sensor-domain transform, can a classifier still tell
"real robot frame" from "transformed public frame"? AUC 0.5 = indistinguishable = good.

Design:
  real   = robot frames from evidence gesture sessions (dev, non-sealed), content-matched hand classes
  public = HaGRID images of the same gesture classes, split by user_id
  (a) baseline: public resized only -> expect AUC near 1.0
  (b) robotified: public through sensor_domain.robotify -> want AUC toward 0.5

Canary: YuNet face recall on real robot face_close frames vs robotified HaGRID
frames that had a detectable face before transformation.

Outputs PACK_ROOT/findings/raw/adversarial_auc_YYYY-MM-DD.json + prints the summary.
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
import sensor_domain as sd  # noqa: E402

HAGRID = Path(r"G:\stream_memory_data\gesture_raw\hagrid\hagrid-sample-30k-384p")

REAL_SESSIONS = {
    "fist": "evidence/sessions/fist_d0.5_front_night_181907/frames",
    "palm": "evidence/sessions/open_palm_d0.5_front_night_181943/frames",
}
FACE_SESSION = "evidence/sessions/face_close_d0.5_front_night_182232/frames"


def load_real(cls_sessions: dict[str, str], per_cls: int) -> dict[str, list[np.ndarray]]:
    out = {}
    for cls, rel in cls_sessions.items():
        d = paths.REPO / rel
        jpgs = sorted(d.glob("*.jpg"))
        imgs = [cv2.imread(str(j), cv2.IMREAD_GRAYSCALE) for j in jpgs]
        imgs = [im for im in imgs if im is not None]
        # temporal split halves are handled by caller; here subsample evenly
        idx = np.linspace(0, len(imgs) - 1, min(per_cls, len(imgs))).astype(int)
        out[cls] = [imgs[i] for i in idx]
    return out


def load_hagrid(classes: list[str], per_cls: int, seed: int = 0) -> tuple[dict[str, list], dict[str, list]]:
    """Returns (by_class_images, user_ids parallel)."""
    import pandas as pd
    df = pd.read_parquet(HAGRID / "annotations_df.parquet")
    rng = np.random.default_rng(seed)
    out, users = {}, {}
    for cls in classes:
        subdir = HAGRID / "hagrid_30k" / f"train_val_{cls}"
        if not subdir.is_dir():
            continue
        files = sorted(subdir.glob("*.jpg"))
        rng.shuffle(files)
        imgs, us = [], []
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            uid = str(df.loc[f.stem]["user_id"]) if f.stem in df.index else "unknown"
            imgs.append(im)
            us.append(uid)
            if len(imgs) >= per_cls:
                break
        out[cls] = imgs
        users[cls] = us
    return out, users


def prep_real(cls: str, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Real frames: temporal middle-out split — first half train, second half test (same session, disjoint windows)."""
    imgs = load_real({cls: REAL_SESSIONS[cls]}, 10_000)[cls]
    half = len(imgs) // 2
    tr, te = imgs[:half], imgs[half:]
    def pick(arr, n):
        idx = np.linspace(0, len(arr) - 1, min(n, len(arr))).astype(int)
        return [arr[i] for i in idx]
    return np.stack(pick(tr, n_train)), np.stack(pick(te, n_test))


def prep_public(imgs: list, users: list, n_train: int, n_test: int) -> tuple[list, list]:
    """Split by user_id so test subjects are unseen."""
    order = np.argsort(users)
    uniq = sorted(set(users))
    half = len(uniq) // 2
    train_users = set(uniq[:half])
    tr = [im for im, u in zip(imgs, users) if u in train_users]
    te = [im for im, u in zip(imgs, users) if u not in train_users]
    return tr[:n_train], te[:n_test]


def small_cnn():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(), nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1),
    )


def tensorize(imgs, size=96):
    import torch
    arr = np.stack([cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im, (size, size), interpolation=cv2.INTER_AREA) for im in imgs])
    return torch.from_numpy(arr[:, None].astype(np.float32) / 255.0)


def eval_auc(real_tr, real_te, pub_tr, pub_te, device, epochs=6, seed=0) -> dict:
    import torch
    torch.manual_seed(seed)
    xtr = torch.cat([tensorize(real_tr), tensorize(pub_tr)])
    ytr = torch.cat([torch.ones(len(real_tr)), torch.zeros(len(pub_tr))]).float()
    xte = torch.cat([tensorize(real_te), tensorize(pub_te)])
    yte = np.concatenate([np.ones(len(real_te)), np.zeros(len(pub_te))])
    model = small_cnn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            loss = lossf(model(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb)
    model.eval()
    with torch.no_grad():
        logits = model(xte.to(device)).squeeze(-1).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yte, logits))
    return {"auc": round(auc, 4), "n_train": int(n), "n_test": int(len(xte))}


def robotify_all(imgs, seed=0):
    rng = np.random.default_rng(seed)
    return [sd.robotify(im, rng=rng) for im in imgs]


def canary_yunet(device) -> dict:
    onnx = paths.REPO / "pc" / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"
    det = cv2.FaceDetectorYN.create(str(onnx), "", (240, 240), score_threshold=0.6)
    # real robot face recall (face_close session)
    jpgs = sorted((paths.REPO / FACE_SESSION).glob("*.jpg"))
    real_imgs = [cv2.imread(str(j)) for j in jpgs[::5]]
    real_hit = sum(1 for im in real_imgs if det.detect(im)[1] is not None and len(det.detect(im)[1]))
    real_recall = real_hit / max(len(real_imgs), 1)
    # public: hagrid images with a detectable face before transform
    pub, _users = load_hagrid(["fist"], per_cls=120)
    rng = np.random.default_rng(0)
    had_face, robo_hit = 0, 0
    for im in pub["fist"]:
        im240 = cv2.resize(im, (240, 240))
        faces = det.detect(im240)[1]
        if faces is None or len(faces) == 0:
            continue
        had_face += 1
        rf = cv2.cvtColor(sd.robotify(im, rng=rng), cv2.COLOR_GRAY2BGR)
        f2 = det.detect(rf)[1]
        if f2 is not None and len(f2):
            robo_hit += 1
    pub_recall = robo_hit / max(had_face, 1)
    return {
        "real_face_close_recall": round(real_recall, 3),
        "n_real": len(real_imgs),
        "robotified_public_face_recall": round(pub_recall, 3),
        "n_public_with_face": had_face,
        "note": "recall on robot frames vs on robotified public frames that had a detectable face pre-transform",
    }


def main() -> int:
    t0 = time.perf_counter()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device), "classes": {}}

    for cls in ("fist", "palm"):
        real_tr, real_te = prep_real(cls, 350, 150)
        pub_imgs, users = load_hagrid([cls], per_cls=600)
        pub_tr, pub_te = prep_public(pub_imgs[cls], users[cls], 350, 150)
        # (a) baseline: resized only
        base = eval_auc(real_tr, real_te, pub_tr, pub_te, device)
        # (b) robotified
        rb_tr = robotify_all(pub_tr, seed=1)
        rb_te = robotify_all(pub_te, seed=2)
        rob = eval_auc(real_tr, real_te, rb_tr, rb_te, device)
        report["classes"][cls] = {"baseline_resize_only": base, "robotified": rob}
        print(f"{cls}: baseline AUC={base['auc']}  robotified AUC={rob['auc']}", flush=True)

    print("canary yunet...", flush=True)
    report["canary_yunet"] = canary_yunet(device)
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    out = paths.RAW / f"adversarial_auc_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["canary_yunet"], ensure_ascii=False), flush=True)
    print(f"wrote {out} in {report['runtime_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
