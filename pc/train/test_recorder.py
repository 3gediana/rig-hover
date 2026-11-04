# -*- coding: utf-8 -*-
"""test_recorder.py -- 录制脚本自检：**图到底有没有落盘**。

为什么单独立一条门禁
====================

这个项目里唯一**事后补不回来**的东西是原图。派生量（`feat_64`、`theta_img`、
`person_h`、教师标签）全都能从原图重算，反过来不行。如果哪天有人重构
`session_from_robot.py` 把 `--save-frames` 弄坏了，症状是**录制一切正常、
pnz 里数字都对**，直到要重训感知或者要标手图的时候才发现手里没有一张图。
那种失败必须在 CI 里挡掉，不能靠人记得。

覆盖：
  * ``PerceptionOut`` 的 ``kp_xy`` / ``kp_conf`` / ``bbox_px`` 三个原始量存在、
    形状对、默认值是零
  * 端到端录一段，落盘的 PNG 数量 == 帧数，侧车 json 与 npz 对齐
  * ``--frame-every 2`` 只存一半，没存的帧 ``frame_idx = -1``
  * ``meta["mark"]`` 存在（``--keys`` 的真值通道）

Run:  python test_recorder.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

N_PASS = 0
N_FAIL = 0
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
HAGRID = (r"D:\robot_work\pc\gesture\data\raw\hagrid"
          r"\hagrid-sample-30k-384p")


def check(name: str, cond: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print("  PASS  %s" % name)
    else:
        N_FAIL += 1
        print("  FAIL  %s   %s" % (name, detail))


def _make_images(dirpath: str, n: int) -> None:
    """优先用 HaGRID 真图（能测到"有人"的分支），没有就造灰底矩形。"""
    import cv2
    src_dirs = []
    root = os.path.join(HAGRID, "hagrid_30k")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root))[:3]:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                src_dirs.append(p)
    got = 0
    if src_dirs:
        for d in src_dirs:
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(".jpg"):
                    continue
                img = cv2.imread(os.path.join(d, f))
                if img is None:
                    continue
                cv2.imwrite(os.path.join(dirpath, "img_%03d.jpg" % got), img)
                got += 1
                if got >= n:
                    return
    while got < n:                      # 兜底：不依赖任何数据集
        img = np.full((240, 240, 3), 110, np.uint8)
        cv2.rectangle(img, (80, 40), (160, 200), (60, 60, 60), -1)
        cv2.imwrite(os.path.join(dirpath, "img_%03d.jpg" % got), img)
        got += 1


def _run(out_dir: str, img_dir: str, session_id: str, extra) -> subprocess.CompletedProcess:
    cmd = [PY, "-u", os.path.join(HERE, "session_from_robot.py"),
           "--source", "dir:" + img_dir,
           "--limit", "6", "--out", out_dir, "--session-id", session_id,
           "--save-frames", os.path.join(out_dir, "frames"),
           "--device", "cpu"] + list(extra)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=600)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="rigrec_")
    img_dir = os.path.join(tmp, "imgs")
    os.makedirs(img_dir)
    _make_images(img_dir, 6)

    # ---- 1) PerceptionOut 的原始量字段 ----
    print("\n[1] PerceptionOut 原始量字段")
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "perception"))
    try:
        import run_perception as rp
        o = rp.PerceptionOut(
            feat_raw=np.zeros(548, np.float32), feat_64=np.zeros(64, np.float32),
            theta_img=0.0, person_w=0.0, person_h=0.0, person_cx=0.5,
            person_conf=0.0, seen=0, hand_up_L=0, hand_up_R=0,
            wave_sin=0.0, wave_cos=0.0, wave_amp=0.0, gesture_conf=0.0)
        check("kp_xy 是 (17,2)", np.asarray(o.kp_xy).shape == (17, 2),
              str(np.asarray(o.kp_xy).shape))
        check("kp_conf 是 (17,)", np.asarray(o.kp_conf).shape == (17,),
              str(np.asarray(o.kp_conf).shape))
        check("bbox_px 长度 4", len(o.bbox_px) == 4, str(o.bbox_px))
        check("默认全零", float(np.abs(np.asarray(o.kp_xy)).sum()) == 0.0)
        check("两个实例不共享数组",
              o.kp_xy is not rp.PerceptionOut(
                  feat_raw=o.feat_raw, feat_64=o.feat_64, theta_img=0.0,
                  person_w=0.0, person_h=0.0, person_cx=0.5, person_conf=0.0,
                  seen=0, hand_up_L=0, hand_up_R=0, wave_sin=0.0,
                  wave_cos=0.0, wave_amp=0.0, gesture_conf=0.0).kp_xy)
    except Exception as exc:  # noqa: BLE001 - 环境问题要报清楚，不要静默跳过
        check("导入 run_perception 并构造 PerceptionOut", False, str(exc)[:200])

    # ---- 2) 端到端录制：图真的落盘了吗 ----
    print("\n[2] 端到端录制（--frame-every 1）")
    out1 = os.path.join(tmp, "run1")
    r = _run(out1, img_dir, "rec1", [])
    check("录制进程退出码 0", r.returncode == 0,
          (r.stderr or "")[-400:] + (r.stdout or "")[-400:])
    npz = os.path.join(out1, "rec1.npz")
    frames = os.path.join(out1, "frames")
    side = os.path.join(out1, "rec1.frames.json")
    check("npz 存在", os.path.isfile(npz))
    check("侧车 json 存在", os.path.isfile(side))
    if os.path.isfile(npz):
        z = np.load(npz, allow_pickle=True)
        for k in ("frame_idx", "mark", "kp_x0", "kp_y10", "kp_c10",
                  "bbox_x1", "bbox_y2", "person_conf"):
            check("npz 里有 %s" % k, k in z.files)
        n = len(z["frame_idx"]) if "frame_idx" in z.files else -1
        check("6 帧", n == 6, str(n))
        if "frame_idx" in z.files:
            check("frame_idx 全记（stride=1）",
                  np.asarray(z["frame_idx"]).tolist() == [0, 1, 2, 3, 4, 5],
                  str(np.asarray(z["frame_idx"]).tolist()))
        pngs = [f for f in os.listdir(frames)] if os.path.isdir(frames) else []
        check("落盘 6 张图", len(pngs) == 6, str(len(pngs)))
        check("图是 PNG", all(p.endswith(".png") for p in pngs), str(pngs[:3]))
        if os.path.isfile(side):
            d = json.load(open(side, encoding="utf-8"))
            check("侧车 files 数 == 6", len(d["files"]) == 6, str(len(d["files"])))
            check("侧车 stride == 1", d["stride"] == 1, str(d["stride"]))
            check("侧车指向真目录", os.path.isdir(d["frame_dir"]), d["frame_dir"])
            check("侧车文件名与磁盘一致",
                  sorted(d["files"]) == sorted(pngs))

    # ---- 3) --frame-every 2：只存一半，其余 frame_idx = -1 ----
    print("\n[3] --frame-every 2")
    out2 = os.path.join(tmp, "run2")
    r2 = _run(out2, img_dir, "rec2", ["--frame-every", "2"])
    check("录制进程退出码 0", r2.returncode == 0, (r2.stderr or "")[-300:])
    if os.path.isfile(os.path.join(out2, "rec2.npz")):
        z2 = np.load(os.path.join(out2, "rec2.npz"), allow_pickle=True)
        fi = np.asarray(z2["frame_idx"]).tolist()
        check("frame_idx 隔一帧记一次", fi == [0, -1, 2, -1, 4, -1], str(fi))
        pngs2 = os.listdir(os.path.join(out2, "frames"))
        check("落盘 3 张图", len(pngs2) == 3, str(len(pngs2)))

    print("\n" + "=" * 60)
    print("%d PASS / %d FAIL" % (N_PASS, N_FAIL))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
