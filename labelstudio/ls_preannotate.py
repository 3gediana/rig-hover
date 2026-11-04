"""ls_preannotate.py — RIG-Hover 的模型预标注：高置信度自动入库，低置信度推给人工。

设计照搬 D:\\smartcar\\labelstudio\\preannotate.py 的分流思路，但模型换成
RIG-Hover 现有的感知链：

* **face**  ← YuNet（227 KB，桌面 40 px 实测召回 98.6%），score = 检测置信度
* **fist**  ← YOLO-pose 腕部关键点 + 手势分类器 fist 概率，框取腕部周围
* **hand**  ← 同上但分类器输出 open_palm

为什么这样拆：场景 v2 只需要 face / fist / hand 三类框（SCENARIO-v2.md §2）。
现有链里 YuNet 直接给脸框；拳头没有检测器，只有"是不是拳"的分类概率和腕部
坐标，所以框是**从腕部合成**的——位置对、大小是猜的，标注时人要改大小。

⚠️ 第一轮请加 ``--no-auto``：YuNet 的框在桌面域**没有经过人工复核**，
自动入库等于把未验证的标注喂给训练。等人工标过一批、确认框可信之后，
再去掉 --no-auto 让它自动分流。

用法（必须用带 torch 的解释器，labelstudio\\venv 没有）：
    G:\\ai\\python11\\python.exe ls_preannotate.py --dir <帧目录> [--no-auto]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PC = ROOT / "pc"

LABELS = ["face", "fist", "hand"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMPORT_CHUNK = 20
YUNET = PC / "perception" / "assets" / "face_detection_yunet_2023mar.onnx"
LABELED_INBOX = ROOT / "data" / "labeled_inbox"
DONE = ROOT / "data" / "done"
REPORT_DIR = HERE / "reports"
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"

#: Session prefix -> ground-truth class that MUST be present, or ``None`` when
#: the frame is supposed to hold nothing at all.
#:
#: This comes from what the operator was told to do, never from the model. That
#: distinction is the whole point: an ``empty`` session is safe to auto-label as
#: "no boxes" because a human confirmed the frame was empty, whereas a ``fist``
#: frame where the model found nothing is a *miss* and must go to a human.
#: Getting this backwards teaches the detector that fists do not exist.
SESSION_GT = {
    "empty": None,       # 人离开画面 —— 真值是"什么都没有"
    "none": "face",      # 双手放下，但人在画面里，有脸
    "face": "face",      # face_close：人脸特写
    "fist": "fist",
    "knock": "fist",     # 敲桌时手里是拳
    "lean": "face",      # 离开再回来：脸出现/消失
}

#: Upload order. Fists and knocks first: they are the labels the deployment
#: actually needs and the ones the model is worst at. Negatives and empty
#: frames come last so they never occupy the first screen.
SESSION_ORDER = ["fist", "knock", "face", "none", "lean", "empty"]


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


def parse_label_config(xml):
    labels = re.findall(r'<Label\s+value="([^"]+)"', xml or "")
    m = re.search(r'<RectangleLabels\s+name="([^"]+)"\s+toName="([^"]+)"', xml or "")
    if not m:
        m = re.search(r'<RectangleLabels\s+toName="([^"]+)"\s+name="([^"]+)"', xml or "")
        if m:
            m = (m.group(2), m.group(1))
    if not labels or not m:
        return None
    return labels, m.group(1), m.group(2)


def write_data_yaml(path: Path) -> None:
    """YOLO 类别表：标注和训练共用同一份，避免类别号错位。"""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ["# RIG-Hover 场景 v2 —— 由 ls_preannotate.py 自动生成，勿手改类别顺序",
            "path: .", "train: images/train", "val: images/val", "",
            "nc: %d" % len(LABELS), "names:"]
    body += [f"  {i}: {n}" for i, n in enumerate(LABELS)]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"[data.yaml] 生成 {path}（类别 {LABELS}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="模型预标注：高置信自动入库，低置信上传 Label Studio")
    ap.add_argument("--dir", required=True, help="待标注帧目录（含子目录递归扫）")
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--url", default=os.environ.get("LS_URL", "http://127.0.0.1:3334"))
    ap.add_argument("--token", default=None)
    ap.add_argument("--conf-low", type=float, default=0.35, help="低于此分的框不写")
    ap.add_argument("--auto-accept", type=float, default=0.70,
                    help="最低框分 >= 此值的图直接写 YOLO 标注，否则上传 LS")
    ap.add_argument("--no-auto", action="store_true",
                    help="全部上传 LS 人工复核（第一轮必须加，见模块 docstring）")
    ap.add_argument("--fist-box", type=int, default=56,
                    help="拳头合成框的边长（像素）。腕部定位准、大小是猜的")
    ap.add_argument("--face-conf", type=float, default=0.45, help="YuNet 阈值")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import requests  # noqa: PLC0415

    write_data_yaml(DATA_YAML)

    S = requests.Session()
    S.trust_env = False  # 本机有 HTTP_PROXY，绕开
    S.headers["Authorization"] = f"Token {get_token(a.token)}"
    base = a.url.rstrip("/")

    r = S.get(f"{base}/api/projects/{a.project}")
    if r.status_code != 200:
        print(f"[FAIL] 读项目 {a.project} 失败：HTTP {r.status_code} {r.text[:200]}")
        return 1
    project = r.json()
    cfg = parse_label_config(project.get("label_config"))
    if not cfg:
        print("[FAIL] 解析不出 RectangleLabels 标注配置")
        return 1
    proj_labels, from_name, to_name = cfg
    missing = [l for l in LABELS if l not in proj_labels]
    if missing:
        print(f"[FAIL] 项目标签 {proj_labels} 缺 {missing}，拒写（会静默丢框污染训练集）")
        return 1
    print(f"[项目] {project['id']} {project['title']}  标签: {proj_labels}")

    sys.path.insert(0, str(PC / "perception"))
    from run_perception import PerceptionPipeline  # noqa: PLC0415
    pipe = PerceptionPipeline(device="cuda:0")
    face_det = cv2.FaceDetectorYN_create(str(YUNET), "", (240, 240),
                                         a.face_conf, 0.3, 20)
    print("[模型] YuNet(face) + yolo26n-pose(wrist) + gesture classifier(fist/hand)")

    src = Path(a.dir)
    if not src.is_dir():
        print(f"[FAIL] 目录不存在: {src}")
        return 1
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)

    def session_of(p: Path) -> str:
        """帧文件名前缀就是采集时的 session id（fist_00003.jpg -> fist）。"""
        return p.name.split("_", 1)[0] if "_" in p.name else "misc"

    # 上传顺序：拳头/敲击优先，负样本和空镜垫后。
    # 否则按文件名排序时空镜排最前，打开标注页先看到 64 张没东西可标的图。
    rank = {s: i for i, s in enumerate(SESSION_ORDER)}
    imgs.sort(key=lambda p: (rank.get(session_of(p), 99), p.name))
    if a.limit:
        imgs = imgs[:a.limit]
    if not imgs:
        print(f"[OK] {src} 里没有图")
        return 0
    by_sess: dict[str, int] = {}
    for p in imgs:
        by_sess[session_of(p)] = by_sess.get(session_of(p), 0) + 1
    print(f"[推理] {len(imgs)} 张  顺序: "
          + " → ".join(f"{s}×{by_sess[s]}" for s in SESSION_ORDER if s in by_sess))

    rows = []
    t0 = time.time()
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [warn] 读不了 {p.name}，跳过")
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = []

        # --- face：YuNet 直接给框和分 ---
        face_det.setInputSize((w, h))
        _rc, faces = face_det.detect(img)
        if faces is not None:
            for f in faces:
                fx, fy, fw, fh = (float(v) for v in f[:4])
                boxes.append(("face", float(f[14]),
                              (fx / w * 100, fy / h * 100, fw / w * 100, fh / h * 100)))

        # --- fist / hand：分类器给类别和分，腕部关键点给位置 ---
        out = pipe(img)
        raw_id = int(out.gesture_raw_id)
        raw_conf = float(out.gesture_raw_conf)
        want = {5: "fist", 2: "hand"}.get(raw_id)
        if want is not None and raw_conf >= a.conf_low:
            kp = np.asarray(out.kp_xy)
            kc = np.asarray(out.kp_conf)
            best, best_c = None, 0.0
            for ki in (9, 10):
                if ki < len(kc) and float(kc[ki]) > best_c:
                    best, best_c = kp[ki], float(kc[ki])
            if best is not None:
                cx, cy = float(best[0]), float(best[1])
                s = float(a.fist_box)
                # 腕部在边缘时把框拉回图内，否则 Label Studio 里超出部分会被裁掉
                x1 = min(max(cx - s / 2, 0.0), w - s)
                y1 = min(max(cy - s / 2, 0.0), h - s)
                boxes.append((want, raw_conf,
                              (x1 / w * 100, y1 / h * 100, s / w * 100, s / h * 100)))
        min_conf = min((c for _, c, _ in boxes), default=0.0)
        rows.append((p, min_conf, boxes))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(imgs)}")
    print(f"[推理完成] {len(rows)} 张，{time.time() - t0:.1f}s")

    ts = datetime.now().strftime("%m%d_%H%M")
    # 空镜会话整段自动标成"无框"：真值来自"人当时确实离开了画面"，
    # 不是模型猜的，所以没有把漏检教成负样本的风险。
    # 反过来不成立——fist 会话里模型没检出是漏检，必须送人。
    def sess_of(p: Path) -> str:
        return p.name.split("_", 1)[0] if "_" in p.name else "misc"

    auto_empty = [r for r in rows if SESSION_GT.get(sess_of(r[0])) is None]
    auto_rows = [] if a.no_auto else [r for r in rows if r[2] and r[1] >= a.auto_accept]
    ls_rows = [r for r in rows if r not in auto_rows and r not in auto_empty]
    n_zero = sum(1 for r in ls_rows if not r[2])
    print(f"[分流] 上传 LS 人工标 {len(ls_rows)} 张（其中模型零框 {n_zero}）"
          f" / 空镜自动标注 {len(auto_empty)} 张"
          f" / 高置信自动标注 {len(auto_rows)} 张")
    if a.dry_run:
        print("[dry-run] 不写任何东西。预览：")
        for img, mc, bs in auto_rows[:8]:
            print(f"  [自动] {img.name} min={mc:.2f} 框={len(bs)}")
        for img, mc, bs in ls_rows[:8]:
            print(f"  [LS]   {img.name} min={mc:.2f} 框={len(bs)}{'（零框）' if not bs else ''}")
        return 0

    # --- 上传 LS：去重后分块导入，再写预测 ---
    new_tasks = []
    if ls_rows:
        existing, page = set(), 1
        while True:
            d = S.get(f"{base}/api/tasks",
                      params={"project": a.project, "fields": "all",
                              "page_size": 250, "page": page}).json()
            for t in d.get("tasks", []):
                existing.add(os.path.basename(str((t.get("data") or {}).get("image") or "")))
            if len(d.get("tasks", [])) < 250:
                break
            page += 1
        todo, skipped = [], []
        for img, mc, bs in ls_rows:
            if not a.force and any(b == img.name or b.endswith("-" + img.name)
                                   for b in existing):
                skipped.append(img.name)
            else:
                todo.append((img, mc, bs))
        if skipped:
            print(f"[去重] LS 已有同名图 {len(skipped)} 张，跳过（--force 重导）")
        for c0 in range(0, len(todo), IMPORT_CHUNK):
            chunk = todo[c0:c0 + IMPORT_CHUNK]
            files, handles = {}, []
            try:
                for i, (img, _, _) in enumerate(chunk):
                    fh = open(img, "rb")
                    handles.append(fh)
                    files[f"f{i}"] = (img.name, fh, "image/jpeg")
                r = S.post(f"{base}/api/projects/{a.project}/import",
                           files=files, data={"commit_to_project": "true"})
            finally:
                for hh in handles:
                    hh.close()
            if r.status_code != 201:
                print(f"[FAIL] 导入失败 HTTP {r.status_code}: {r.text[:300]}")
                return 1
            print(f"[导入] {r.json().get('task_count')} 张 → 项目 {a.project}")
            fresh, page = [], 1
            while True:
                d = S.get(f"{base}/api/tasks",
                          params={"project": a.project, "fields": "all",
                                  "page_size": 250, "page": page}).json()
                fresh.extend(d.get("tasks", []))
                if len(d.get("tasks", [])) < 250:
                    break
                page += 1
            by_name = {}
            for t in fresh:
                ref = os.path.basename(str((t.get("data") or {}).get("image") or ""))
                for img, _, _ in chunk:
                    if ref == img.name or ref.endswith("-" + img.name):
                        by_name[img.name] = t["id"]
            for img, mc, bs in chunk:
                if img.name in by_name:
                    new_tasks.append((by_name[img.name], img, mc, bs))
                else:
                    print(f"  [warn] 没认回任务 {img.name}，留下轮重试")
        ok, fail = 0, []
        for tid, img, mc, bs in new_tasks:
            result = [{
                "from_name": from_name, "to_name": to_name,
                "type": "rectanglelabels", "source": f"${to_name}",
                "score": round(c, 4),
                "value": {"x": round(x, 2), "y": round(y, 2), "width": round(ww, 2),
                          "height": round(hh, 2), "rotation": 0,
                          "rectanglelabels": [lab]},
            } for lab, c, (x, y, ww, hh) in bs]
            r = S.post(f"{base}/api/predictions",
                       json={"task": tid, "model_version": "rig-v2-pose+yunet",
                             "score": round(mc, 4), "result": result})
            if r.status_code == 201:
                ok += 1
            else:
                fail.append((img.name, r.status_code, r.text[:200]))
        print(f"[预测] 成功 {ok}，失败 {len(fail)}")
        for name, code, txt in fail:
            print(f"  [fail] {name} HTTP {code} {txt}")
        if ok and (project.get("model_version") or "") != "rig-v2-pose+yunet":
            S.patch(f"{base}/api/projects/{a.project}",
                    json={"model_version": "rig-v2-pose+yunet"})

    # --- 自动标注：高置信度直接写 YOLO txt；空镜写空 txt ---
    auto_batch = None
    if auto_rows or auto_empty:
        auto_batch = LABELED_INBOX / f"auto_{ts}"
        (auto_batch / "images").mkdir(parents=True, exist_ok=True)
        (auto_batch / "labels").mkdir(parents=True, exist_ok=True)
        for img, mc, bs in auto_rows:
            lines = [f"{LABELS.index(lab)} {x/100:.6f} {y/100:.6f} "
                     f"{ww/100:.6f} {hh/100:.6f}" for lab, _c, (x, y, ww, hh) in bs]
            shutil.copy2(img, auto_batch / "images" / img.name)
            (auto_batch / "labels" / (img.stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8")
        for img, _mc, _bs in auto_empty:
            # 空标注：文件存在但内容为空。YOLO 训练把空 txt 当纯负样本，
            # 这是检测器学会"这里什么都没有"的唯一途径。
            shutil.copy2(img, auto_batch / "images" / img.name)
            (auto_batch / "labels" / (img.stem + ".txt")).write_text("", encoding="utf-8")
        print(f"[自动标注] {len(auto_rows)} 张有框 + {len(auto_empty)} 张空镜 "
              f"→ {auto_batch}")

    # --- 归档处理过的图 ---
    moved, archive = 0, DONE / f"round_{ts}"
    done_imgs = ([img for _, img, _, _ in new_tasks]
                 + [img for img, _, _ in auto_rows]
                 + [img for img, _, _ in auto_empty])
    for img in done_imgs:
        if not img.is_file():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        dst = archive / img.name
        if dst.exists():
            dst = archive / f"{img.stem}_{moved}{img.suffix}"
        shutil.move(str(img), str(dst))
        moved += 1
    if moved:
        print(f"[归档] {moved} 张 → {archive}（源目录只剩未处理的）")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rep = REPORT_DIR / f"preannotate_{ts}.md"
    lines = [
        f"# 预标注报告 {ts}", "",
        f"- 项目: {project['title']} (id={a.project})",
        f"- 源目录: {src}", f"- 帧数: {len(rows)}",
        f"- 阈值: 写框 conf >= {a.conf_low}；最低框分 >= {a.auto_accept} 自动标注"
        f"{'（本轮 --no-auto，全部人工）' if a.no_auto else ''}",
        f"- 拳头合成框边长: {a.fist_box} px（腕部定位准，大小是猜的，标注时要改）", "",
        "## 上传 LS 待人工标注", "",
    ]
    if new_tasks:
        lines += ["| 图片 | 任务 id | 任务分数 | 框数 | 原因 |", "| --- | --- | --- | --- | --- |"]
        for tid, img, mc, bs in new_tasks:
            reason = "模型完全没检出" if not bs else "有框但置信度不足"
            lines.append(f"| {img.name} | {tid} | {mc:.2f} | {len(bs)} | {reason} |")
        lines += ["", "Data Manager → Filters → **Prediction score** → less → 填阈值，"
                      "列出的就是上表。", ""]
    else:
        lines += ["（本轮没有需要人工标注的图）", ""]
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"[报告] {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
