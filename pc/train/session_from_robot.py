#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""session_from_robot.py -- turn real robot frames into a training session.

Why this exists: the shipped policy was trained on *fabricated* features
(`synth.synth_feat` is a random projection of six scene variables, not YOLO
output), which is why 28% of real inputs fall outside the exported blob's
input envelope. The fix is to train on real observations -- but `train_cfc.py`
consumes `synth.Session` npz files, so real data has to be produced in exactly
that format.

This script does that one job: drive the deployed perception chain plus the
teacher controller over real frames and emit a `Session`-compatible npz. The
existing trainer then works unchanged, and because it fits `input_mean/std`
from whatever data it is given, the exported blob's statistics become real
ones automatically.

Sources:
    --source robot          pull /capture + /status from the robot
    --source dir:<path>     iterate image files (dry run / laptop rehearsal)

Run:
    python session_from_robot.py --source robot --seconds 60 --out data_real
    python session_from_robot.py --source dir:D:\\shots --limit 200 --out data_real
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PERCEPTION = r"D:\robot_work\pc\perception"
TEACHER = r"D:\robot_work\pc\teacher"
ROBOT = "http://192.168.137.50"

sys.path.insert(0, HERE)
sys.path.insert(0, PERCEPTION)
sys.path.insert(0, TEACHER)


def robot_frames(host, seconds, fps, timeout=6.0):
    """Yield (frame_bgr, proprio_dict) from the live robot."""
    import http.client
    import socket
    import cv2

    hostname, _, port_s = host.partition(":")
    port = int(port_s) if port_s else 80
    conn = http.client.HTTPConnection(hostname, port, timeout=timeout)
    conn.connect()
    try:
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    # The HTTP layer costs ~50 ms per request on this board (RESULTS 搂9), so the
    # achievable rate is ~8 Hz. Ask for whatever we can get rather than lying
    # about the fps -- a wrong dt corrupts the observation's time features.
    period = 1.0 / max(fps, 1.0)
    t_end = time.time() + seconds
    n = 0
    while time.time() < t_end:
        t0 = time.perf_counter()
        try:
            conn.request("GET", "/capture")
            r = conn.getresponse()
            jpg = r.read()
            conn.request("GET", "/status")
            r2 = conn.getresponse()
            st = json.loads(r2.read())
        except Exception:
            # reconnect once, then give up on this frame
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection(hostname, port, timeout=timeout)
            try:
                conn.connect()
            except Exception:
                time.sleep(0.2)
                continue
            continue
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield img, proprio_from_status(st), n
        n += 1
        sleep = period - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def proprio_from_status(st):
    """Map /status JSON onto the proprio dict obs.assemble expects."""
    imu = st.get("imu", {})
    wh = st.get("wheel", {})
    hd = st.get("head", {})
    cm = st.get("cmd", {})
    cf = st.get("cfc", {})
    return {
        "roll": float(imu.get("roll", 0.0)),
        "pitch": float(imu.get("pitch", 0.0)),
        "pit_gyro": float(imu.get("pit_gyro", 0.0)),
        "dq": float(imu.get("dq", 0.0)),
        "yaw": float(imu.get("yaw", 0.0)),
        "q_head": float(hd.get("q", 0.0)),
        "target_head_pos": float(hd.get("target", 0.0)),
        "wheel1_vel": float(wh.get("v1", 0.0)),
        "wheel2_vel": float(wh.get("v2", 0.0)),
        "wheel_x": float(cf.get("pos", 0.0)),
        "stable_pos": float(cm.get("pos", 0.0)),
        "stable_yaw": float(cm.get("yaw_target", 0.0)),
        "robot_state": 1,
        "vx": float(cm.get("vx", 0.0)),
        "vyaw": float(cm.get("vyaw", 0.0)),
        "control_mode": int(cm.get("mode", 0)),
        "feature_age_ms": float(cf.get("feat_age_ms", 0) or 0),
        "gaze_danger": int(cf.get("danger", 0) or 0),
    }


def fake_proprio():
    """Rehearsal propio: upright, still. Only the perception domain matters here."""
    return {
        "roll": 0.0, "pitch": -2.0, "pit_gyro": 0.0, "dq": 0.0, "yaw": 0.0,
        "q_head": 0.0, "target_head_pos": 0.0, "wheel1_vel": 0.0,
        "wheel2_vel": 0.0, "wheel_x": 0.0, "stable_pos": 0.0,
        "stable_yaw": 0.0, "robot_state": 1, "vx": 0.0, "vyaw": 0.0,
        "control_mode": 0, "feature_age_ms": 0.0, "gaze_danger": 0,
    }


def dir_frames(path, limit, fps):
    import cv2
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))
             if f.lower().endswith(exts)]
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no images under {path}")
    per = 1.0 / max(fps, 1.0)
    for i, p in enumerate(files):
        img = cv2.imread(p)
        if img is not None:
            yield img, fake_proprio(), i
            time.sleep(per * 0.01)   # keep wall clock advancing for dt features


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="robot | dir:<path>")
    ap.add_argument("--host", default="192.168.137.50:80",
                    help="robot host:port. Point this at a mock to rehearse the "
                         "recording path when the robot is unavailable.")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=8.0,
                    help="target rate; the robot's HTTP layer caps it near 8 Hz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data_real"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--session-id", default="real_000")
    ap.add_argument("--save-frames", default="",
                    help="Save every frame as an image into this directory. "
                         "Without this the recording contains no pixels, so the "
                         "perception chain can never be re-run over it and no "
                         "image label can ever be added later. Pass the same "
                         "directory for every session; files are prefixed with "
                         "--session-id.")
    ap.add_argument("--frame-every", type=int, default=1,
                    help="Save one frame out of N (1 = all).")
    ap.add_argument("--frame-ext", default="png", choices=("png", "jpg"),
                    help="png is lossless and is what hand crops must be taken from.")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--keys", action="store_true",
                    help="Space marks the current frame. Used to record ground "
                         "truth for events (e.g. 'the two knocks happened now') "
                         "without anyone labelling anything afterwards.")
    a = ap.parse_args()

    save_frames = a.save_frames.strip()
    if save_frames:
        os.makedirs(save_frames, exist_ok=True)
    key_state = {"pressed": False}
    if a.keys:
        try:
            import msvcrt
        except ImportError:
            raise SystemExit("--keys needs Windows (msvcrt)")
        key_state["msvcrt"] = msvcrt
        print("按键标记已开启：录制时按 <空格> 打一个标记（例如【敲第二下的瞬间】），"
              "标记落在 meta['mark'] 里，事后不用再看视频标注。")

    import cv2
    import obs as obsmod
    from run_perception import PerceptionPipeline
    from teacher import TeacherController, TeacherObs, load_config
    from synth import Session

    os.makedirs(a.out, exist_ok=True)
    pipe = PerceptionPipeline(device=a.device)
    # CUDA/cuDNN warm-up. The first inference costs ~16 s of kernel autotuning
    # on this machine (measured on the real robot, 2026-11-03); without it a
    # 15 s recording yields exactly one frame because warm-up eats the window.
    _t0 = time.perf_counter()
    pipe(np.zeros((240, 240, 3), np.uint8))
    print(f"perception warmed up in {time.perf_counter() - _t0:.1f} s")
    tcfg = load_config(os.path.join(TEACHER, "config.yaml"))
    teacher = TeacherController(tcfg)
    teacher.reset()
    dist_est = obsmod.DistEstimator()

    if a.source == "robot":
        it = robot_frames(a.host, a.seconds, a.fps)
    elif a.source.startswith("dir:"):
        it = dir_frames(a.source[4:], a.limit, a.fps)
    else:
        raise SystemExit("--source must be 'robot' or 'dir:<path>'")

    obs_rows, head, vyaw, vx, mode, emote = [], [], [], [], [], []
    # 原始信号：录下来的是**未组装的感知输出 + 本体状态**，不是组装好的 106 维
    # 观测。理由很实际：观测布局一定会改（比如加人脸/拳头/敲门通道），
    # 如果只存组装结果，改一次布局就得重采一次数据。存原始量则可以事后重排、
    # 甚至用新的归一化重算，一份数据反复用。
    feat_rows = []
    prop_rows: dict[str, list] = {}
    meta = {k: [] for k in ("theta_img", "eps_body", "dist_est", "wave_score",
                            "person_h", "seen", "wave_sin", "q_head",
                            "target_head_pos", "yaw", "t_s",
                            # 场景 v2 信号
                            "theta_face", "face_scale", "face_valid",
                            "theta_fist", "wrist_y_norm", "knock_fired",
                            "knock_x", "knock_y", "gesture_raw_id",
                            "gesture_raw_conf",
                            # 检测框（原图像素）与 COCO-17 关键点。有了关键点就能
                            # 事后按腕部裁手、按鼻/眼重算方位，不必回机器人重采。
                            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                            "person_conf",
                            # 手工事件标记（--keys）与帧文件对齐索引
                            "mark", "frame_idx")}
    for _i in range(17):
        for _s in ("x", "y", "c"):
            meta["kp_%s%d" % (_s, _i)] = []
    frame_files: list[str] = []
    prev_act = None
    t_prev = None
    print(f"recording from {a.source} -> {a.out}/{a.session_id}.npz")
    for img, prop, i in it:
        t_s = float(i) / max(a.fps, 1.0)
        dt = (t_s - t_prev) if t_prev is not None else 0.0
        t_prev = t_s
        # ---- 原图落盘：这是整份数据集里唯一无法事后重建的东西 ----
        if save_frames and len(obs_rows) % max(a.frame_every, 1) == 0:
            if a.frame_ext == "png":
                fname = "%s_%06d.png" % (a.session_id, len(obs_rows))
                cv2.imwrite(os.path.join(save_frames, fname), img)
            else:
                fname = "%s_%06d.jpg" % (a.session_id, len(obs_rows))
                cv2.imwrite(os.path.join(save_frames, fname), img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(a.jpg_quality)])
            frame_files.append(fname)
            meta["frame_idx"].append(float(len(obs_rows)))
        else:
            meta["frame_idx"].append(-1.0)
        # ---- 事件标记：空格按下那一帧记 1 ----
        mark = 0.0
        if a.keys:
            mc = key_state["msvcrt"]
            if mc.kbhit():
                ch = mc.getch()
                if ch in (b" ", 32):
                    mark = 1.0
                    print("  [mark] frame %d  t=%.2fs" % (len(obs_rows), t_s))
        meta["mark"].append(mark)
        p = pipe(img)
        perc = {
            "feat_64": p.feat_64, "theta_img": p.theta_img,
            "person_w": p.person_w, "person_h": p.person_h,
            "person_conf": p.person_conf, "seen": p.seen,
            "hand_up_L": p.hand_up_L, "hand_up_R": p.hand_up_R,
            "wave_sin": p.wave_sin, "wave_cos": p.wave_cos,
            "wave_amp": p.wave_amp,
            "gesture_conf": p.wave_gesture_conf,
        }
        d_est = dist_est.step(p.seen, p.person_h) if hasattr(dist_est, "step") else 0.0
        obs_rows.append(obsmod.assemble(perc, prop, prev_act, t_s, d_est))
        act = teacher.step(TeacherObs(
            theta_img=p.theta_img, q_head=prop["q_head"], seen=p.seen,
            person_h=p.person_h, hand_up_L=p.hand_up_L, hand_up_R=p.hand_up_R,
            t_now=t_s, gesture_id=p.gesture_id, gesture_onset=p.gesture_onset,
            gesture_conf=p.gesture_conf, gesture_gap_s=p.gesture_gap_s))
        head.append(act.head_des); vyaw.append(act.vyaw_des)
        vx.append(act.vx_des); mode.append(act.mode); emote.append(act.emote)
        prev_act = act
        meta["theta_img"].append(p.theta_img)
        meta["eps_body"].append(act.debug.get("eps_body", 0.0))
        meta["dist_est"].append(d_est)
        meta["wave_score"].append(getattr(p, "wave_gesture_conf", 0.0))
        meta["person_h"].append(p.person_h)
        meta["seen"].append(float(p.seen))
        meta["wave_sin"].append(p.wave_sin)
        meta["q_head"].append(prop["q_head"])
        meta["target_head_pos"].append(prop["target_head_pos"])
        meta["yaw"].append(prop["yaw"])
        meta["t_s"].append(t_s)
        meta["theta_face"].append(float(getattr(p, "theta_face", 0.0)))
        meta["face_scale"].append(float(getattr(p, "face_scale", 0.0)))
        meta["face_valid"].append(float(getattr(p, "face_valid", 0)))
        meta["theta_fist"].append(float(getattr(p, "theta_fist", 0.0)))
        meta["wrist_y_norm"].append(float(getattr(p, "wrist_y_norm", 0.0)))
        meta["knock_fired"].append(float(getattr(p, "knock_fired", 0)))
        meta["knock_x"].append(float(getattr(p, "knock_x", 0.0)))
        meta["knock_y"].append(float(getattr(p, "knock_y", 0.0)))
        meta["gesture_raw_id"].append(float(getattr(p, "gesture_raw_id", 0)))
        meta["gesture_raw_conf"].append(
            float(getattr(p, "gesture_raw_conf", 0.0)))
        # 检测框与关键点：存原图像素坐标。事后要按腕部裁 64px 手图、按鼻/眼重算
        # 方位、或者换一个更宽/更窄的 FOV 重投影，都只依赖这两组数。
        bb = getattr(p, "bbox_px", None) or (0.0, 0.0, 0.0, 0.0)
        meta["bbox_x1"].append(float(bb[0]))
        meta["bbox_y1"].append(float(bb[1]))
        meta["bbox_x2"].append(float(bb[2]))
        meta["bbox_y2"].append(float(bb[3]))
        meta["person_conf"].append(float(p.person_conf))
        kxy = getattr(p, "kp_xy", None)
        kcf = getattr(p, "kp_conf", None)
        kxy = None if kxy is None else np.asarray(kxy, np.float32).reshape(-1, 2)
        kcf = None if kcf is None else np.asarray(kcf, np.float32).reshape(-1)
        for _j in range(17):
            if kxy is not None and _j < kxy.shape[0]:
                meta["kp_x%d" % _j].append(float(kxy[_j, 0]))
                meta["kp_y%d" % _j].append(float(kxy[_j, 1]))
            else:
                meta["kp_x%d" % _j].append(0.0)
                meta["kp_y%d" % _j].append(0.0)
            meta["kp_c%d" % _j].append(
                float(kcf[_j]) if kcf is not None and _j < kcf.shape[0] else 0.0)
        # 原始量：感知特征向量 + 全部本体状态的键值
        feat_rows.append(np.asarray(p.feat_64, np.float32))
        for k, v in prop.items():
            prop_rows.setdefault(k, []).append(float(v))
        if (len(obs_rows)) % 25 == 0:
            print(f"  {len(obs_rows)} frames  seen={p.seen}  "
                  f"theta={p.theta_img:+.1f}  head={act.head_des:+.1f}")

    if not obs_rows:
        print("recorded nothing")
        return 2

    # 原始量一并写进 meta：`feat_64` 是 (N,64)，`prop_*` 是每键一条 (N,)。
    # 有了这些，改观测布局时只要重写 assemble()，不需要重新采集。
    meta["feat_64"] = np.stack(feat_rows).astype(np.float32)
    for k, v in prop_rows.items():
        meta[f"prop_{k}"] = np.asarray(v, np.float32)

    sess = Session(
        obs=np.stack(obs_rows).astype(np.float32),
        head_deg=np.asarray(head, np.float32),
        vyaw_dps=np.asarray(vyaw, np.float32),
        vx_mps=np.asarray(vx, np.float32),
        mode=np.asarray(mode, np.int8),
        emote=np.asarray(emote, np.int8),
        meta={k: np.asarray(v, np.float32) for k, v in meta.items()},
    )
    path = sess.save(os.path.join(a.out, f"{a.session_id}.npz"))
    O = sess.obs
    if save_frames:
        side = os.path.join(a.out, f"{a.session_id}.frames.json")
        with open(side, "w", encoding="utf-8") as fh:
            json.dump({
                "session_id": a.session_id,
                "frame_dir": os.path.abspath(save_frames),
                "n_frames": int(len(sess)),
                "stride": int(max(a.frame_every, 1)),
                "files": frame_files,
                "note": "files[i] 对应 npz 第 frame_idx[i] 帧；stride>1 时其余帧没图",
            }, fh, ensure_ascii=False, indent=1)
        print(f"  原图：{len(frame_files)} 张 → {save_frames}（索引 {side}）")
    marks = np.asarray(meta["mark"], np.float32)
    if marks.sum() > 0:
        idx = np.flatnonzero(marks > 0.5)
        print(f"  事件标记 {len(idx)} 个：帧 {idx.tolist()}  "
              f"时刻 {[round(float(meta['t_s'][k]), 2) for k in idx]}")
    print(f"\nwrote {path}  ({len(sess)} frames, obs {O.shape})")
    print(f"  原始量：feat_64 {meta['feat_64'].shape}，"
          f"{len(prop_rows)} 个本体字段（prop_*），{len(meta)} 个 meta 键")

    # Report the real statistics -- these are what the blob will carry.
    const = np.where(O.std(axis=0) < 1e-6)[0]
    print(f"constant dims (std<1e-6): {len(const)} {list(const)[:12]}")
    print(f"obs range: [{O.min():.3f}, {O.max():.3f}]")
    counts = {int(m): int((sess.mode == m).sum()) for m in np.unique(sess.mode)}
    print(f"mode distribution: {counts}")
    base = os.path.basename(a.out.rstrip("/\\"))
    print("\n下一步（三条命令，`train_cfc.py` 会从这批数据拟合 input_mean/std，"
          "所以导出的 blob 统计量自动变成真实的）：")
    print(f"  1) train_cfc.py --data-dir {base} --out-dir out_{base} --epochs 80")
    print(f"  2) export_blob.py --npz out_{base}/policy.npz --out out_{base}/model.blob")
    print(f"  3) python ..\\bench\\obs_domain_check.py --n 100 "
          f"--policy out_{base}/policy.npz")
    print("  第 3 步必须显示 `OK -- envelope holds`；仍是 RISK 说明录的数据")
    print("  太少或场景太单一。")
    print("\n（`test_e2e.py` 是数值一致性套件，不是重训入口，别拿它当训练命令。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

