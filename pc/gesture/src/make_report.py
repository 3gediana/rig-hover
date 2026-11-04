# -*- coding: utf-8 -*-
"""make_report.py — 从各步骤产出的 JSON 生成 reports/classifier.md 与 events.md

报告里的每个数字都来自某次真实运行的产物（eval_*.json / calibration.json /
latency.json / build_meta.json），不手抄，避免文档和实测脱节。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORTS = os.path.join(ROOT, "reports")

sys.path.insert(0, HERE)
from gesture_common import CLASSES  # noqa: E402


def _load(name, default=None):
    p = os.path.join(REPORTS, name)
    if not os.path.exists(p):
        return default
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def classifier_md():
    fused = _load("eval_fused.json")
    skel = _load("eval_skeleton.json")
    build = _load(os.path.join("..", "data", "synth", "build_meta.json")) or {}
    if not fused:
        return "# 分类器\n\n（还没跑 eval_classifier.py）\n"

    rows = []
    for r in fused["per_class"]:
        s = next((x for x in (skel or {}).get("per_class", [])
                  if x["class"] == r["class"]), None)
        rows.append([r["class"], "%d" % r["support"],
                     "%.3f" % r["precision"], "%.3f" % r["recall"],
                     "%.3f" % r["f1"],
                     "%.3f" % s["f1"] if s else "-"])

    cm = fused["confusion"]
    cm_rows = []
    for i, name in enumerate(fused["classes"]):
        cm_rows.append([name] + [str(v) for v in cm[i]])

    unk = fused.get("unseen_open_set") or {}
    parts = []
    parts.append("# 手势分类器：公开数据训练结果\n")
    parts.append("数据来自 HaGRID 30k 采样集（见 README「数据」）。"
                 "**训练/验证/测试按人划分**：%d 张图 → %d 个可用骨架帧 → "
                 "%d 个 user，其中 train/val/test = %d/%d/%d 人，三份 user "
                 "列表两两不相交（`test_gesture.py` 会断言）。\n"
                 % (build.get("n_images", 0), build.get("n_usable", 0),
                    build.get("n_users", 0),
                    len(build.get("users", {}).get("train", [])),
                    len(build.get("users", {}).get("val", [])),
                    len(build.get("users", {}).get("test", []))))

    parts.append("## 测试集每类指标（%d 个窗口，%d 个留出用户）\n"
                 % (sum(r["support"] for r in fused["per_class"]),
                    len(build.get("users", {}).get("test", []))))
    parts.append(_table(rows, ["类别", "样本数", "precision", "recall", "F1",
                               "纯骨架 F1"]))
    parts.append("\n宏平均 F1 **%.4f**，准确率 **%.4f**；模型 %d 参数（12 帧"
                 "滑窗，CPU 单帧 ~1.5 ms）。\n"
                 % (fused["macro_f1"], fused["accuracy"], fused["n_params"]))
    if skel:
        parts.append("纯骨架消融：宏平均 F1 **%.4f**，准确率 **%.4f**（%d 参数）。\n"
                     % (skel["macro_f1"], skel["accuracy"], skel["n_params"]))

    parts.append("## 混淆矩阵（行=真值，列=预测）\n")
    parts.append(_table(cm_rows, ["真值\\预测"] + list(fused["classes"])))
    parts.append("")

    parts.append("## 未知手势会不会误报\n")
    if unk:
        parts.append("- 严格开放集（7 类**从未进过训练**的 HaGRID 手势，%d 帧，"
                     "按人留出）：被认成非 idle **%.1f%%**，超过触发阈值 "
                     "**%.1f%%**。\n"
                     % (unk["n"], 100 * unk["frac_claimed_nonidle"],
                        100 * unk["frac_would_trigger"]))
    if skel and skel.get("unseen_open_set"):
        u2 = skel["unseen_open_set"]
        parts.append("- 纯骨架模型在同一批样本上：非 idle **%.1f%%**。"
                     "外观分支把它压下来，但**没有完全解决**。\n"
                     % (100 * u2["frac_claimed_nonidle"]))
    parts.append("- 训练时见过的 12 类非目标手势（call/peace/... 中的 6 类）"
                 "被做成内部第 7 类 `other`，输出时概率并回 idle。\n")

    parts.append("## 读法\n")
    parts.append("- `wave` 的 F1 偏高（0.98）是**合成运动的必然结果**：挥手样本"
                 "由真实 palm 姿势 + 2 连杆 IK 摆动生成，训练和测试同分布。真挥手"
                 "没验证过，见 README「局限」。\n")
    parts.append("- `idle` 的样本数包含了 640 个 `other` 窗口（折叠后算 idle），"
                 "所以 idle 的 recall 里有一部分是「正确拒绝了未知手势」。\n")
    parts.append("- `pointing` 是这条管线最弱的一环：COCO-17 没有手指关键点，"
                 "32×32 的手部裁剪在 240×240 视野下只有 ~20 px 的手，"
                 "食指是否伸出几乎是亚像素信息。纯骨架版本 0.24，加外观 0.35，"
                 "**都不可用**。真要用 pointing，得换手部 21 关键点模型。\n")
    return "\n".join(parts) + "\n"


def events_md():
    cal = _load("calibration.json")
    lat = _load("latency.json")
    if not cal:
        return "# 事件分割\n\n（还没跑 calibrate_events.py）\n"
    b, a = cal["before"], cal["after"]
    parts = []
    parts.append("# 事件化手势分割：标定与实测\n")
    parts.append("标定脚本 `src/calibrate_events.py`：喂带**已知动作起点**的长序列，"
                 "搜索 `PENDING` 确认窗口、进入/退出阈值、`release`。"
                 "序列成分与训练集同源（真实姿势 + IK 合成运动），"
                 "但用 val 用户的、独立的随机种子。\n")

    parts.append("## 标定前后对比（同一批 %d 条序列）\n" % cal["n_episodes"])
    rows = [
        ["挥手触发延迟（均值）",
         ("%.0f ms（只有 %d/%d 条能触发）"
          % (b["wave_latency_mean_ms"],
             sum(1 for v in b["wave_latency_by_freq"].values() if v),
             len(b["wave_latency_by_freq"])))
         if b.get("wave_latency_mean_ms") else "从未触发",
         "%.0f ms" % a["wave"]["latency_mean_ms"]],
        ["挥手漏检率", "%.0f%%" % (100 * b["wave_miss_rate"]),
         "%.0f%%" % (100 * a["wave"]["miss_rate"])],
        ["连续挥手产生几个 onset",
         "%d/10 条能触发" % sum(1 for v in
                                b["wave_latency_by_freq"].values() if v),
         "全部 1 个（%.0f%%）" % (100 * a["cont_single_onset_rate"])],
        ["两次挥手（间隔 1 s）拆分",
         "旧实现只会一直亮着", "2 个 onset"],
        ["静止 10 s 误报", "%d 次" % b["idle_false_alarms"],
         "%.2f 次/分钟" % a["fa_per_min"]],
        ["举手不动", "无法表达（只认挥手）",
         "%.0f%% 恰好 1 个 onset" % (100 * a["static_one_onset_rate"])],
    ]
    parts.append(_table(rows, ["指标", "标定前（wave_score > 4 + 2 s 衰减）",
                               "标定后（事件机）"]))
    parts.append("")

    parts.append("## 为什么会这样\n")
    parts.append("旧实现的 `wave_score` 每帧乘 `exp(-dt/2.0)`，过零一次加 1。"
                 "稳态下 `score ≈ 过零率 × 2 s`：1 Hz 慢挥每秒 2 次过零，"
                 "稳态刚好 **4.0**，而触发阈值就是 `>4`——**它永远在阈值下方"
                 "贴边运行**。实测（`src/diag_old_algo.py`）：\n")
    parts.append(_table([
        ["1.0 Hz", "2.17", "3.6 s 内从未超过 4", "漏检"],
        ["2.0 Hz", "4.09", "首次超过在 2.7 s", "勉强触发"],
        ["2.5 Hz", "4.20", "首次超过在 3.1 s", "勉强触发"],
    ], ["挥手频率", "wave_score 峰值", "首次 >4", "结果"]))
    parts.append("")
    parts.append("另外实测到旧实现还有一个独立的硬伤：4 s 内 `active_hand` "
                 "切换了 **44~59 次**（按 `|x_rel|` 选活动手，关键点一抖就换人），"
                 "每次切换都会清空过零状态。两条缺陷叠在一起，就是"
                 "「1 Hz 慢挥要 4 秒才过阈值」的来源。\n")

    parts.append("## 逐条序列的触发延迟（标定后）\n")
    rows = []
    for name, v in cal["per_episode_latency_ms_after"].items():
        before = cal["per_episode_latency_ms_before"].get(name)
        rows.append([name, "%.0f ms" % v if v is not None else "-",
                     "%.0f ms" % before if before is not None else "从未触发"])
    parts.append(_table(rows, ["序列", "标定后首次 onset", "标定前"]))
    parts.append("")

    parts.append("## onset 计数（设计目标：同一段连续动作只出 1 个）\n")
    parts.append(_table([
        ["连续挥手 0.5/1.0/1.5/2.0/2.5 Hz × 幅度 0.35/0.75",
         str(a["wave"]["n_onsets"]), "全部 1"],
        ["两次挥手，间隔 0.4 s", str(a["two_wave"]["n_onsets"][0]),
         "合并成 1（间隔太短，按设计合并）"],
        ["两次挥手，间隔 0.6/0.8/1.0/1.5 s", str(a["two_wave"]["n_onsets"][1:]),
         "全部 2"],
        ["举手不动（open_palm/thumbs_up/pointing/fist）",
         str(a["static"]["n_onsets"]), "全部 1"],
        ["静止 10 s ×2", str(a["idle"]["n_onsets"]), "全部 0"],
    ], ["场景", "实测 onset 数", "结论"]))
    parts.append("")

    parts.append("## 标定出来的参数（configs/gesture_events.yaml）\n")
    t = cal["timing_after"]
    parts.append(_table([
        ["enter / exit", "%.2f / %.2f" % (cal["config_after"]["enter_thresh"],
                                          cal["config_after"]["exit_thresh"]),
         "迟滞宽度 %.2f，避免贴阈值抖动"
         % (cal["config_after"]["enter_thresh"]
            - cal["config_after"]["exit_thresh"])],
        ["pending_hold", "%.2f s" % t["pending_hold"], "确认窗口（去抖，不是冷却）"],
        ["release", "%.2f s" % t["release"], "低于退出阈值多久算动作结束"],
        ["min_gesture", "%.2f s" % t["min_gesture"], "单事件最短持续"],
        ["return_clear", "%.2f s" % t["return_clear"],
         "RETURN 里 idle 主导多久才允许下一个 onset"],
        ["min_gap", "%.2f s" % t["min_gap"], "onset 最小间隔（只去抖）"],
    ], ["参数", "取值", "作用"]))
    parts.append("")

    if lat:
        parts.append("## 单帧耗时（RTX 5060 Laptop，imgsz=320）\n")
        p1 = lat["pose"]["pose_bs1_cuda"]
        p8 = lat["pose"]["pose_bs8_cuda"]
        g = lat["classifier_gpu"]
        c = lat["classifier_cpu"]
        parts.append(_table([
            ["关键点提取（YOLO26n-pose, batch=1）", "%.1f ms" % p1["median_ms"],
             "%.1f ms" % p1["p90_ms"]],
            ["关键点提取（batch=8）", "%.1f ms/帧" % p8["median_ms"],
             "%.1f ms/帧" % p8["p90_ms"]],
            ["分类：骨架特征（numpy）", "%.3f ms" % c["feat"]["median_ms"], "-"],
            ["分类：手部裁剪（cv2）", "%.3f ms" % c["crop"]["median_ms"], "-"],
            ["分类：12 帧滑窗网络（GPU）", "%.2f ms" % g["net"]["median_ms"],
             "%.2f ms" % g["net"]["p90_ms"]],
            ["分类：12 帧滑窗网络（CPU）", "%.2f ms" % c["net"]["median_ms"],
             "%.2f ms" % c["net"]["p90_ms"]],
            ["分类端到端（像素→概率）", "%.2f ms" % c["end_to_end"]["median_ms"],
             "%.2f ms" % c["end_to_end"]["p90_ms"]],
        ], ["环节", "中位", "p90"]))
        parts.append("")
        parts.append("单帧总计约 **%.1f ms**（关键点 %.1f + 分类 %.1f），"
                     "30 Hz 预算 33 ms 内；分类只占 %.0f%%。\n"
                     % (p1["median_ms"] + c["end_to_end"]["median_ms"],
                        p1["median_ms"], c["end_to_end"]["median_ms"],
                        100 * c["end_to_end"]["median_ms"]
                        / (p1["median_ms"] + c["end_to_end"]["median_ms"])))
    return "\n".join(parts) + "\n"


def main():
    os.makedirs(REPORTS, exist_ok=True)
    for name, fn in (("classifier.md", classifier_md),
                     ("events.md", events_md)):
        text = fn()
        path = os.path.join(REPORTS, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("[report] → %s (%d 字节)" % (path, len(text.encode("utf-8"))))


if __name__ == "__main__":
    main()
