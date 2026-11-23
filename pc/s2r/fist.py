# -*- coding: utf-8 -*-
"""s2r/fist.py - recall-first fist detector for the desk robot, measured against hand-labelled real frames.

Diagnosis (gt set, 2026-11-24): of the clear fist frames the afternoon stack missed, most had a wrist keypoint
at high confidence that the old (wy-shoulder_y)/shoulder_w <= 0.48 "raised" gate threw away (fist held out at
desk height, not above the shoulder), and the YOLOE close-up path had its class list in the wrong order.
This path keeps the fist whenever a confident wrist exists and the hand is in the interaction band, and uses
YOLOE('fist',...) as class 0 only as a fallback when no wrist is found (fist fills the frame).
"""
from __future__ import annotations
import math
import numpy as np

F_PX_PER_DEG = 2.705


class FistDetector:
    def __init__(self, pose_model, yoloe_model=None, wrist_conf=0.30, band=(0.30, 0.98), hold_s=0.5):
        self.pose = pose_model
        self.yoloe = yoloe_model
        if self.yoloe is not None:
            self.yoloe.set_classes(["fist", "hand", "fist of a person", "face"])
        self.wrist_conf = wrist_conf
        self.band = band
        self.hold_s = hold_s
        self.last = None
        self.last_t = -1e9
        self.streak = 0

    def _band_ok(self, cy_norm):
        return self.band[0] <= cy_norm <= self.band[1]

    def detect(self, bgr, now_s, face_box_px=None):
        H, W = bgr.shape[:2]
        cand = None
        rp = self.pose.predict(bgr, imgsz=256, verbose=False, conf=0.15, half=False)[0]
        if rp.keypoints is not None and len(rp.keypoints) and rp.boxes is not None and len(rp.boxes):
            kxy = rp.keypoints.xy.cpu().numpy(); kcf = rp.keypoints.conf.cpu().numpy()
            best = -1.0
            for b in range(len(kcf)):
                for el_i, wr_i in ((7, 9), (8, 10)):
                    wc = float(kcf[b][wr_i])
                    if wc < self.wrist_conf:
                        continue
                    wx, wy = float(kxy[b][wr_i][0]), float(kxy[b][wr_i][1])
                    # push a little past the wrist toward the hand along the forearm
                    if float(kcf[b][el_i]) >= 0.2:
                        ex, ey = float(kxy[b][el_i][0]), float(kxy[b][el_i][1])
                        hx = wx + 0.35 * (wx - ex); hy = wy + 0.35 * (wy - ey)
                    else:
                        hx, hy = wx, wy
                    cy = hy / H
                    if not self._band_ok(cy):
                        continue
                    # reject only if the hand point is INSIDE the face box (that is the head, not a fist)
                    if face_box_px is not None:
                        fx1, fy1, fx2, fy2 = face_box_px
                        if fx1 <= hx <= fx2 and fy1 <= hy <= fy2:
                            continue
                    if wc > best:
                        best = wc
                        cand = {"cx": float(np.clip(hx / W, 0.03, 0.97)), "cy": float(np.clip(cy, 0.05, 0.97)),
                                "bearing_deg": round((hx / W - 0.5) * (W / F_PX_PER_DEG), 2), "conf": round(wc, 3), "src": "wrist"}
        if cand is None and self.yoloe is not None:
            ry = self.yoloe.predict(bgr, imgsz=256, verbose=False, conf=0.15)[0]
            if ry.boxes is not None and len(ry.boxes):
                best = 0.0
                for j in range(len(ry.boxes)):
                    if int(ry.boxes.cls[j]) != 0:   # class 0 == 'fist'
                        continue
                    cf = float(ry.boxes.conf[j])
                    x1, y1, x2, y2 = [float(v) for v in ry.boxes.xyxy[j].tolist()]
                    area = (x2 - x1) * (y2 - y1); cyp = 0.5 * (y1 + y2)
                    if area < 400 or area > 0.6 * H * W or not self._band_ok(cyp / H):
                        continue
                    if face_box_px is not None:
                        fx1, fy1, fx2, fy2 = face_box_px
                        ix = max(0.0, min(x2, fx2) - max(x1, fx1)) * max(0.0, min(y2, fy2) - max(y1, fy1))
                        if ix / max(1.0, area) > 0.35:
                            continue
                    if cf > best:
                        best = cf; cxp = 0.5 * (x1 + x2)
                        cand = {"cx": float(np.clip(cxp / W, 0.03, 0.97)), "cy": float(np.clip(cyp / H, 0.05, 0.97)),
                                "bearing_deg": round((cxp / W - 0.5) * (W / F_PX_PER_DEG), 2), "conf": round(cf, 3), "src": "yoloe"}
        # 2-frame confirm + short hold to debounce
        if cand is not None:
            self.streak += 1
            self.last = cand; self.last_t = now_s
            return cand if self.streak >= 1 else None
        self.streak = 0
        if self.last is not None and now_s - self.last_t <= self.hold_s:
            return dict(self.last, coasted=1)
        return None
