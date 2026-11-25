# -*- coding: utf-8 -*-
"""pc/pack/k1_fist_transfer.py - O-005/K-1 falsification.

Question: does public fist/palm data help recall real-robot fists AFTER the
sensor-domain transform, compared to the same data untransformed?

Cheap, honest design:
  train ONLY on public data (HaGRID fist positive vs other classes negative)
    arm A: resized+gray only
    arm B: robotified through the sensor contract (same pixels, same labels)
  evaluate on real robot frames never seen in training:
    positive GT: evidence/sessions/fist_d0.5 (201 frames, 0.5 m night)
    negative GT: none/open_palm/wave/pointing/thumbs_up sessions
  metric: real-frame fist recall at the operating point where real negative FP rate <= 5%,
          plus recall at fixed threshold 0.5, plus AUC. Compare arm A vs arm B.

Known limitation (written into the report): HaGRID hands are close-ups, real fists
are 10-20 px at 0.5-1 m. Scale gap is part of what K-1 is meant to expose; both arms
share it, so the comparison isolates the sensor transform.

Output: PACK_ROOT/findings/raw/k1_fist_transfer_YYYY-MM-DD.json
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
from adversarial_auc import load_hagrid, small_cnn, tensorize  # noqa: E402

HAGRID_NEG_CLASSES = ["palm", "like", "ok", "stop", "mute", "peace", "call", "dislike"]
REAL_POS = "evidence/sessions/fist_d0.5_front_night_181907/frames"
REAL_NEG = [
    "evidence/sessions/none_d0.5_front_night_181841/frames",
    "evidence/sessions/open_palm_d0.5_front_night_181943/frames",
    "evidence/sessions/wave_d0.5_front_night_182035/frames",
    "evidence/sessions/pointing_d0.5_front_night_182101/frames",
    "evidence/sessions/thumbs_up_d0.5_front_night_182007/frames",
]


def load_real_frames(rel: str, every: int = 2) -> list[np.ndarray]:
    jpgs = sorted((paths.REPO / rel).glob("*.jpg"))
    return [im for im in (cv2.imread(str(j), cv2.IMREAD_GRAYSCALE) for j in jpgs[::every]) if im is not None]


def train_fist_classifier(pos_imgs, neg_imgs, device, epochs=8, seed=0):
    import torch
    torch.manual_seed(seed)
    xtr = torch.cat([tensorize(pos_imgs), tensorize(neg_imgs)])
    ytr = torch.cat([torch.ones(len(pos_imgs)), torch.zeros(len(neg_imgs))]).float()
    model = small_cnn().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(xtr)
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            xb, yb = xtr[idx].to(device), ytr[idx].to(device)
            loss = lossf(model(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def logits_on(model, imgs, device):
    import torch
    with torch.no_grad():
        return model(tensorize(imgs).to(device)).squeeze(-1).cpu().numpy()


def eval_real(model, real_pos, real_neg, device) -> dict:
    from sklearn.metrics import roc_auc_score
    lp = logits_on(model, real_pos, device)
    ln = logits_on(model, real_neg, device)
    y = np.concatenate([np.ones(len(lp)), np.zeros(len(ln))])
    s = np.concatenate([lp, ln])
    auc = float(roc_auc_score(y, s))
    p05 = np.exp(lp) / (1 + np.exp(lp))
    n05 = np.exp(ln) / (1 + np.exp(ln))
    rec05 = float((p05 > 0.5).mean())
    fp05 = float((n05 > 0.5).mean())
    # recall at threshold where FP rate on real negatives <= 5%
    thr = float(np.quantile(np.exp(ln) / (1 + np.exp(ln)), 0.95)) if len(ln) else 0.5
    rec_at_fp5 = float((p05 > thr).mean())
    return {"auc": round(auc, 4), "recall@0.5": round(rec05, 4), "fp_rate@0.5": round(fp05, 4),
            "recall@fp<=5%": round(rec_at_fp5, 4), "thr_for_fp5": round(thr, 4),
            "n_pos": len(real_pos), "n_neg": len(real_neg)}


def main() -> int:
    t0 = time.perf_counter()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    rng = np.random.default_rng(0)
    # public training data
    pos_pub, _u1 = load_hagrid(["fist"], per_cls=500)
    neg_imgs = []
    negs, _u2 = load_hagrid(HAGRID_NEG_CLASSES, per_cls=90)
    for cls, arr in negs.items():
        neg_imgs.extend(arr)
    rng.shuffle(neg_imgs)
    neg_imgs = neg_imgs[:500]
    print(f"public train: pos={len(pos_pub['fist'])} neg={len(neg_imgs)}", flush=True)

    # real GT
    real_pos = load_real_frames(REAL_POS)
    real_neg = []
    for rel in REAL_NEG:
        real_neg.extend(load_real_frames(rel, every=6))
    print(f"real GT: pos={len(real_pos)} neg={len(real_neg)}", flush=True)

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": {}}

    # arm A: resized+gray only
    gray_pos = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in pos_pub["fist"]]
    gray_neg = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in neg_imgs]
    m_a = train_fist_classifier(gray_pos, gray_neg, device)
    ra = eval_real(m_a, real_pos, real_neg, device)
    report["arms"]["A_raw_resize_only"] = ra
    print(f"arm A raw: {ra}", flush=True)
    del m_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # arm B: robotified
    rb_pos = [sd.robotify(im, rng=rng) for im in pos_pub["fist"]]
    rb_neg = [sd.robotify(im, rng=rng) for im in neg_imgs]
    m_b = train_fist_classifier(rb_pos, rb_neg, device)
    rb = eval_real(m_b, real_pos, real_neg, device)
    report["arms"]["B_robotified"] = rb
    print(f"arm B robotified: {rb}", flush=True)

    # verdict
    gain = rb["recall@fp<=5%"] - ra["recall@fp<=5%"]
    report["recall_gain_at_fp5"] = round(gain, 4)
    if gain >= 0.10 and rb["recall@fp<=5%"] >= 0.5:
        verdict = "U2-go: robotified public data transfers meaningfully to real robot fists"
    elif gain >= 0.05 or rb["auc"] > ra["auc"] + 0.05:
        verdict = "U1-weak: some transfer benefit, scale/content gap still dominant"
    elif rb["auc"] < 0.6 and ra["auc"] < 0.6:
        verdict = "UX: neither arm transfers — full-frame close-up public data does not reach 10-20px real fists regardless of sensor domain"
    else:
        verdict = "U0-no-gain: transform does not help (or hurts)"
    report["verdict"] = verdict
    report["limitations"] = [
        "HaGRID hands are close-ups; real fists are 10-20 px at 0.5-1 m — scale gap shared by both arms",
        "real positive GT is one session, one person, one distance (0.5 m night): falsification-grade only",
        "full-frame classification; the A1 track will need detection/cropping regardless",
    ]
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    out = paths.RAW / f"k1_fist_transfer_{time.strftime('%Y-%m-%d')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {verdict}", flush=True)
    print(f"wrote {out} in {report['runtime_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
