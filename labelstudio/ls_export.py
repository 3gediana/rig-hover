"""ls_export.py — 把 Label Studio 里标好的标注导出成 YOLO 训练集。

分流逻辑的另一半：``ls_preannotate.py`` 管"哪些图需要人标"，这个管"标完怎么用"。

切分按**段**不按帧。文件名前缀就是采集时的 session id（``label_fist_00003.jpg``
→ ``label_fist``），同一段相邻帧高度相关，按帧随机切等于把验证集漏进训练集。
``--holdout`` 指定的段整段进 val，一个帧都不混。

用法：
    G:\\ai\\python11\\python.exe ls_export.py --out D:\\robot_work\\data\\dataset
    G:\\ai\\python11\\python.exe ls_export.py --holdout label_none,calib_1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LABELS = ["face", "fist", "hand"]
OUT_ROOT = ROOT / "data" / "dataset"


def get_token(cli):
    if cli:
        return cli.strip()
    if os.environ.get("LS_TOKEN"):
        return os.environ["LS_TOKEN"].strip()
    f = HERE / ".ls_token"
    if f.is_file():
        t = f.read_text(encoding="utf-8").strip()
        if t:
            return t
    print("[FAIL] 没有 API token：--token / 环境变量 LS_TOKEN / .ls_token 三选一")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--url", default=os.environ.get("LS_URL", "http://127.0.0.1:3334"))
    ap.add_argument("--token", default=None)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--holdout", default="",
                    help="整段留作验证集的 session 前缀，逗号分隔（如 label_none）")
    ap.add_argument("--only-completed", action="store_true", default=True)
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="把没标的任务也导成空标注（默认跳过）")
    a = ap.parse_args()

    import requests  # noqa: PLC0415

    S = requests.Session()
    S.trust_env = False
    S.headers["Authorization"] = f"Token {get_token(a.token)}"
    base = a.url.rstrip("/")

    # ---- 1. 拉全部任务 + 标注 ----
    tasks, page = [], 1
    while True:
        r = S.get(f"{base}/api/tasks",
                  params={"project": a.project, "fields": "all",
                          "page_size": 250, "page": page})
        if r.status_code != 200:
            print(f"[FAIL] 读任务失败 HTTP {r.status_code}: {r.text[:200]}")
            return 1
        d = r.json()
        tasks.extend(d.get("tasks", []))
        if len(d.get("tasks", [])) < 250:
            break
        page += 1
    print(f"[任务] {len(tasks)} 个")

    holdout = {h.strip() for h in a.holdout.split(",") if h.strip()}
    rows = []          # (session, image_path, [label, cx, cy, w, h] normalized)
    skipped_empty = 0
    for t in tasks:
        data = t.get("data") or {}
        img_url = str(data.get("image") or "")
        if not img_url:
            continue
        anns = t.get("annotations") or []
        if not anns:
            # 预测不算标注：模型猜的框不能当训练目标
            if not a.include_unlabeled:
                skipped_empty += 1
            continue
        result = anns[-1].get("result") or []
        boxes = []
        for r in result:
            if r.get("type") != "rectanglelabels":
                continue
            v = r.get("value") or {}
            labs = v.get("rectanglelabels") or []
            if not labs:
                continue
            lab = labs[0]
            if lab not in LABELS:
                continue
            x, y = float(v.get("x", 0)), float(v.get("y", 0))
            w, h = float(v.get("width", 0)), float(v.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            # Label Studio 的 x/y 是左上角百分比；YOLO 要中心点归一化
            boxes.append((lab, (x + w / 2) / 100.0, (y + h / 2) / 100.0,
                          w / 100.0, h / 100.0))
        if not boxes and not a.include_unlabeled:
            skipped_empty += 1
            continue
        # 本地路径：LS 把上传的文件放在 <data-dir>/media/upload/<proj>/<uuid8>-<name>
        name = os.path.basename(urlparse(img_url).path)
        rel = name.split("-", 1)[1] if "-" in name else name
        session = rel.split("_")[0] if "_" in rel else "misc"
        local = _find_media(name)
        if local is None:
            print(f"  [warn] 找不到本地文件 {name}，跳过")
            continue
        rows.append((session, local, rel, boxes))
    print(f"[标注] {len(rows)} 张有标注，跳过 {skipped_empty} 张未标注/空标注")

    if not rows:
        print("[FAIL] 没有可导出的标注，先去标几张")
        return 1

    # ---- 2. 按段切分 ----
    train = [r for r in rows if r[0] not in holdout]
    val = [r for r in rows if r[0] in holdout]
    if not val:
        # 没指定 holdout 就按段轮转：每 5 段取 1 段，仍然整段不拆
        sessions = sorted({r[0] for r in rows})
        val_sessions = set(sessions[::5])
        val = [r for r in rows if r[0] in val_sessions]
        train = [r for r in rows if r[0] not in val_sessions]
        print(f"[切分] 未指定 --holdout，按段轮转留出: {sorted(val_sessions)}")
    print(f"[切分] train {len(train)} 张（{len({r[0] for r in train})} 段）/ "
          f"val {len(val)} 张（{len({r[0] for r in val})} 段）")

    # ---- 3. 落盘 ----
    out = Path(a.out)
    for split, data in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for _sess, src, rel, boxes in data:
            stem = Path(rel).stem
            shutil.copy2(src, out / "images" / split / rel)
            lines = [f"{LABELS.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for lab, cx, cy, w, h in boxes]
            (out / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml = out / "data.yaml"
    yaml.write_text(
        "\n".join([
            "# RIG-Hover 场景 v2 —— 由 ls_export.py 生成",
            f"path: {out.as_posix()}",
            "train: images/train", "val: images/val", "",
            f"nc: {len(LABELS)}", "names:"] +
            [f"  {i}: {n}" for i, n in enumerate(LABELS)]) + "\n", encoding="utf-8")

    # ---- 4. 类别分布（一眼看出某类是不是采歪了）----
    from collections import Counter  # noqa: PLC0415
    cnt = Counter(lab for _s, _p, _r, bs in rows for lab, *_ in bs)
    print("\n[类别分布] " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    empty = sum(1 for _s, _p, _r, bs in rows if not bs)
    print(f"[空标注图] {empty} 张（全负样本，训练时有用）")
    print(f"\n[完成] 数据集 → {out}")
    print(f"       {yaml}")
    return 0


def _find_media(name: str) -> Path | None:
    """LS 上传文件的落地位置：<data-dir>/media/upload/<proj>/<uuid8>-<name>。"""
    roots = [ROOT / "labelstudio" / "data" / "media" / "upload",
             Path(os.environ.get("LABEL_STUDIO_BASE_DATA_DIR",
                                 "D:/robot_work/labelstudio/data")) / "media" / "upload"]
    for root in roots:
        if not root.is_dir():
            continue
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
