"""Exploratory within-clip feature probe on one robot leave/return recording.

Labels are coarse manual visibility states transcribed from timestamped contact sheets.
This is not a generalization benchmark: there is only one person/session/environment.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(r"D:\robot_work\stream_memory_research")
CACHE = Path(r"G:\stream_memory_data\features\robot_gesture")
OUT = ROOT / "agent_feature_probe"
SESSION = "lean_out_d0.5_front_night_182257"

# Approximate per-frame labels from the 5-fps contact sheets; ambiguous transition
# bands are excluded. Class IDs: absent=0, visible=1, transition/partial=2.
# Frame times are about 10 Hz; only first 192 cached frames are available.
N = 192
labels = np.full(N, -1, np.int64)
segments = [
    (0, 9, 1), (9, 29, 2), (29, 67, 0), (67, 80, 2), (80, 102, 1),
    (102, 111, 2), (111, 132, 0), (132, 145, 2), (145, 192, 1),
]
for a,b,c in segments: labels[a:b]=c

paths = sorted(CACHE.glob(f"seq_{SESSION}_k*.npz"))
if len(paths) != 3:
    raise SystemExit(f"Expected 3 chunks, got {len(paths)}: {paths}")
zs=[np.load(p, allow_pickle=False) for p in paths]
cls=np.concatenate([z["feat"].astype(np.float32) for z in zs])[:N]
tok=np.concatenate([z["tok"].astype(np.float32) for z in zs])[:N]
# Current CLS, pooled patch tokens, causal 5-frame mean, and causal feature delta.
patch=tok.reshape(N,-1)
mean5=np.stack([cls[max(0,i-4):i+1].mean(0) for i in range(N)])
delta=np.vstack([np.zeros((1,cls.shape[1]),np.float32),np.diff(cls,axis=0)])
# Raw low-resolution pixels are an information-preserving lower bound relative to
# the encoder family (not an equal-capacity learned model).
raw=[]
for i in range(N):
    im=cv2.imread(str(ROOT / ".." / "evidence" / "sessions" / SESSION / "frames" / f"{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
    raw.append(cv2.resize(im,(16,16),interpolation=cv2.INTER_AREA).reshape(-1))
raw=np.asarray(raw,dtype=np.float32)
features={"raw_16x16_current":raw,"cls_current":cls,"patch_current":patch,
          "cls_plus_delta":np.concatenate([cls,delta],1),
          "cls_causal_mean5":mean5,"cls_mean5_plus_delta":np.concatenate([mean5,delta],1)}
# Blocked time split: first half train; second half test. This still shares one recording,
# and is reported only as a within-clip diagnostic, not a held-session estimate.
train_idx=np.arange(0,96); test_idx=np.arange(96,N)
results={"session":SESSION,"n_frames":N,"split":"first 96 frames train; final 96 test (same recording)",
         "label_note":"Coarse visual states manually assigned from contact sheets; transition frames are class 2; not independent GT.",
         "segments":segments,"classes":{"0":"absent","1":"visible","2":"transition/partial"},"probes":{}}
for name,X in features.items():
    tr=train_idx[labels[train_idx]>=0]
    te=test_idx[labels[test_idx]>=0]
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",C=0.1))
    model.fit(X[tr],labels[tr])
    pred=model.predict(X[te])
    results["probes"][name]={"accuracy":float(accuracy_score(labels[te],pred)),
        "macro_f1":float(f1_score(labels[te],pred,average="macro",labels=[0,1,2],zero_division=0)),
        "n_train":int(len(tr)),"n_test":int(len(te)),
        "test_support":{str(k):int((labels[te]==k).sum()) for k in [0,1,2]},
        "confusion_matrix":confusion_matrix(labels[te],pred,labels=[0,1,2]).tolist(),
        "report":classification_report(labels[te],pred,labels=[0,1,2],output_dict=True,zero_division=0)}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/"leave_return_probe.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
lines=["# One-clip robot feature probe", "", "**Exploratory only:** one 0.5 m, front-facing, night session; same recording is split by time. Labels are coarse manual visibility states from contact sheets, not frame annotations collected independently.","",f"Frames: {N}; train 0-95; test 96-{N-1}. Classes: 0=absent, 1=visible, 2=transition/partial.","", "| Features | Accuracy | Macro-F1 | Test support (0/1/2) |", "|---|---:|---:|---|"]
for name,r in results["probes"].items():
    sup=r["test_support"]
    lines.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {sup['0']}/{sup['1']}/{sup['2']} |")
lines += ["", "This probe only checks whether the selected feature sequence contains a signal for coarse presence in this same clip. It cannot establish leave/return generalization, identity continuity across people/scenes, interaction readiness, fist recognition, knock phases/count, or target location. The test shares recording/background/person with training; it is not a valid deployment estimate.", "", "No knock/wave count scores are produced because the recordings lack reliable event-level boundaries/count annotations."]
(OUT/"leave_return_probe.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print((OUT/"leave_return_probe.md").read_text(encoding="utf-8"))
