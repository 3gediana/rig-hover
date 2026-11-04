# -*- coding: utf-8 -*-
"""prepare_hagrid.py — 解压 30k 采样集并按官方标注生成 manifest.csv

HaGRID 的 30k 采样集有两份标签：

* ``hagrid_30k/train_val_<类名>/<图片id>.jpg``：目录名是**拍摄意图**的类别；
* ``annotations_df.parquet``：**官方标注**，``labels`` 列里有 ``no_gesture``。

两者不一致的 3,098 张，正是「拍摄时要求做某个手势但实际没做出来」的图，
官方把它们的真值标成 ``no_gesture``。**以 parquet 为准**——这正是我们要的
idle 负类：真人、真实光照、手没在比划。若按目录名取，这 3,098 张会被当成
手势样本喂进去，idle 就只剩 0 个。

注意：``user_id`` 是**按受试者哈希**给出的（7,973 个），所以按人划分是
可验证的，不是「按图片随机划分」。

产出：``data/manifest.csv``（path, hagrid_class, user_id, split），外加
``data/hagrid_meta.json`` 的统计。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from gesture_common import HAGRID_TO_CLASS, CLASSES  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "hagrid-sample-30k-384p.zip")
EXTRACT_DIR = os.path.join(RAW_DIR, "hagrid")
MANIFEST = os.path.join(ROOT, "data", "manifest.csv")

#: 「未映射的 12 类」抽样多少张，用来测开放集误报
N_OTHER = 2500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-other", type=int, default=N_OTHER)
    args = ap.parse_args()

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    flag = os.path.join(EXTRACT_DIR, ".extracted")
    if not os.path.exists(flag):
        print("[prepare] 解压 %s" % ZIP_PATH, flush=True)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(EXTRACT_DIR)
        with open(flag, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
    else:
        print("[prepare] 已解压，跳过", flush=True)

    base = os.path.join(EXTRACT_DIR, "hagrid-sample-30k-384p")
    with zipfile.ZipFile(ZIP_PATH) as z:
        df = pd.read_parquet(io.BytesIO(
            z.read("hagrid-sample-30k-384p/annotations_df.parquet")))
    df["label1"] = df["labels"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and len(x) else "no_gesture")
    print("[prepare] 标注 %d 行，%d 个 user" % (len(df), df["user_id"].nunique()),
          flush=True)

    img_dir = os.path.join(base, "hagrid_30k")
    rows = []
    for img_id, r in df.iterrows():
        cls = str(r["label1"])
        folder = "train_val_%s" % cls
        path = os.path.join(img_dir, folder, "%s.jpg" % img_id)
        if not os.path.exists(path):
            # parquet 的标签可能和目录名不一致（no_gesture 就是这种情况），
            # 此时按目录名前缀找真实路径
            cand = os.path.join(img_dir, "train_val_" + _guess_folder(img_dir,
                                                                      img_id),
                                "%s.jpg" % img_id)
            if os.path.exists(cand):
                path = cand
            else:
                continue
        rows.append({"path": path, "hagrid_class": cls,
                     "user_id": str(r["user_id"]), "split": "train_val"})

    mapped = [r for r in rows if HAGRID_TO_CLASS.get(r["hagrid_class"])]
    other = [r for r in rows if not HAGRID_TO_CLASS.get(r["hagrid_class"])]
    rng = np.random.default_rng(0)
    if len(other) > args.n_other:
        idx = rng.choice(len(other), size=args.n_other, replace=False)
        other = [other[i] for i in idx]

    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["path", "hagrid_class", "user_id",
                                            "split"])
        wr.writeheader()
        for r in mapped + other:
            wr.writerow(r)

    cnt = {}
    for r in mapped:
        cls = HAGRID_TO_CLASS[r["hagrid_class"]]
        cnt[cls] = cnt.get(cls, 0) + 1
    meta = {
        "n_rows": len(rows), "n_mapped": len(mapped), "n_other": len(other),
        "n_users": int(df["user_id"].nunique()),
        "class_frame_counts": cnt,
        "users_per_class": {
            c: int(len({r["user_id"] for r in mapped
                        if HAGRID_TO_CLASS[r["hagrid_class"]] == c}))
            for c in CLASSES if c != "wave"},
        "note": "wave 不在 HaGRID 中，由真实 palm/stop 姿势 + IK 合成",
    }
    with open(os.path.join(ROOT, "data", "hagrid_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def _guess_folder(img_dir, img_id):
    """parquet 标签和目录名不一致时，扫一次目录找真实所在类。"""
    for name in os.listdir(img_dir):
        if os.path.exists(os.path.join(img_dir, name, "%s.jpg" % img_id)):
            return name.replace("train_val_", "")
    return "no_gesture"


if __name__ == "__main__":
    main()
