"""Inspect a desk-scene burst: YuNet face pixels + deployed pose/gesture output."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np

PC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PC / "perception"))
from run_perception import PerceptionPipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("burst", type=Path)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=519)
    ap.add_argument("--out", type=Path, default=Path(r"D:\robot_work\evidence\live\desk_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    yunet_path = PC / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"
    face = cv2.FaceDetectorYN_create(str(yunet_path), "", (240,240), 0.45, 0.3, 20)
    pipe = PerceptionPipeline(device="cuda:0")
    rows=[]; tiles=[]
    folder = args.burst / "frames"
    for idx in range(args.start, args.end+1, args.step):
        p=folder/f"{idx:05d}.jpg"
        img=cv2.imread(str(p))
        if img is None: continue
        face.setInputSize((img.shape[1],img.shape[0]))
        _, faces=face.detect(img)
        # FaceDetectorYN row: x,y,w,h, five (x,y) landmarks, score at index 14.
        f=[] if faces is None else sorted(faces.tolist(),key=lambda b:b[14],reverse=True)
        out=pipe(img)
        x1,y1,x2,y2=map(float,out.bbox_px)
        kp=np.asarray(out.kp_xy); kc=np.asarray(out.kp_conf)
        wrists=[]
        for ki in (9,10):
            if ki < len(kc) and kc[ki] >= .2:
                wrists.append((round(float(kp[ki,0]),1),round(float(kp[ki,1]),1),round(float(kc[ki]),2)))
        face_px=max((float(b[3]) for b in f),default=0.0)
        row={"idx":idx,"person":int(out.seen),"person_conf":round(float(out.person_conf),3),
             "bbox_w":round(x2-x1,1),"bbox_h":round(y2-y1,1),
             "face_count":len(f),"face_w":round(max((float(b[2]) for b in f),default=0),1),
             "face_h":round(face_px,1),"face_conf":round(float(f[0][14]),3) if f else 0,
             "gesture_raw":int(out.gesture_raw_id),"gesture_conf":round(float(out.gesture_raw_conf),3),
             "gesture_event":int(out.gesture_id),"fist_angle":round(float(out.theta_fist),2),
             "wrists":str(wrists)}
        rows.append(row)
        vis=img.copy()
        if out.seen:
            cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
            for ki in (9,10):
                if ki < len(kc) and kc[ki] >= .2:
                    cv2.circle(vis,tuple(kp[ki].astype(int)),3,(0,0,255),-1)
        for b in f:
            fx,fy,fw,fh=map(int,b[:4]); cv2.rectangle(vis,(fx,fy),(fx+fw,fy+fh),(255,0,0),1)
        cv2.putText(vis,f"{idx} face={face_px:.0f}px g={row['gesture_raw']}:{row['gesture_conf']:.2f}",(3,12),cv2.FONT_HERSHEY_SIMPLEX,.29,(255,255,0),1)
        tiles.append(cv2.resize(vis,(240,240)))
        print("idx={idx:03d} person={person} pconf={person_conf:.2f} bbox={bbox_w:.0f}x{bbox_h:.0f} face={face_count} {face_w:.0f}x{face_h:.0f}@{face_conf:.2f} gest={gesture_raw}:{gesture_conf:.2f} wrists={wrists}".format(**row),flush=True)
    csvp=args.out/"detections.csv"
    with csvp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        w.writeheader(); w.writerows(rows)
    cols=6; rowsn=(len(tiles)+cols-1)//cols
    sheet=np.full((rowsn*240,cols*240,3),24,np.uint8)
    for i,t in enumerate(tiles): sheet[(i//cols)*240:(i//cols+1)*240,(i%cols)*240:(i%cols+1)*240]=t
    cv2.imwrite(str(args.out/"contact.jpg"),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
    valid=[r for r in rows if r['person']]
    detected=[r for r in rows if r['face_count']]
    print(f"\nframes={len(rows)} people={len(valid)} faces={len(detected)}; outputs={args.out}")
    if detected:
        print("face size px p10/p50/p90 =",np.percentile([r['face_h'] for r in detected],[10,50,90]).round(1).tolist())
    return 0
if __name__=="__main__": raise SystemExit(main())
