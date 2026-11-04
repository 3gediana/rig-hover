"""Aggregate finished runs into the comparison / robustness / ablation tables.

Reads every `runs/*/metrics.json` with status == "complete" and produces
reports/summary_main.md, summary_robust.md, summary_diff.md, summary_ablation.md
and reports/summary.json.

Paired statistics:
  * if `best.per_seq_test` exists  -> paired bootstrap over SEQUENCES (strong)
  * else                          -> paired bootstrap over SEEDS (weak, flagged)

Usage
-----
  python scripts/summarize.py --ref cfc
  python scripts/summarize.py --ref cfc --datasets ucf101_action,charades_event
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(r"D:\robot_work\stream_memory_research")
sys.path.insert(0, str(ROOT))

from src.common import contract, metrics as M  # noqa: E402

ROBUST_LEVELS = ["dense", "stride2", "stride4", "drop30", "drop50", "drop70",
                 "burst", "repeat", "jitter", "mixed"]

HIGHER_IS_BETTER = {"acc", "accuracy", "f1", "f1_macro", "precision", "recall",
                    "auroc", "auc", "id_recall", "cosine", "cos", "r2",
                    "prefix_acc", "auc_prefix"}
LOWER_IS_BETTER = {"mae", "mse", "rmse", "loss", "latency", "divergence"}


def is_higher(metric: str) -> bool:
    """Direction of a metric name.  Overridable per dataset by the run itself."""
    m = (metric or "").lower()
    if m in HIGHER_IS_BETTER:
        return True
    if m in LOWER_IS_BETTER:
        return False
    return True


# --------------------------------------------------------------------------- #
def load_runs() -> List[dict]:
    out = []
    for p in sorted((ROOT / "runs").glob("*/metrics.json")):
        try:
            d = contract.read_json(p)
        except Exception:
            continue
        if d.get("status") != "complete":
            continue
        d.setdefault("variant", "")
        out.append(d)
    return out


def key_of(d: dict) -> Tuple[str, str, str]:
    return (d.get("dataset", "?"), d.get("model", "?"), d.get("variant", "") or "base")


def label_of(k: Tuple[str, str, str]) -> str:
    ds, model, variant = k
    return model if variant in ("", "base") else f"{model}[{variant}]"


def collect(runs: List[dict], metric: str) -> Dict[str, Dict[Tuple[str, str, str], List[dict]]]:
    """dataset -> key -> list of per-seed dicts."""
    out: Dict[str, Dict[Tuple[str, str, str], List[dict]]] = defaultdict(lambda: defaultdict(list))
    for d in runs:
        b = d.get("best", {}) or {}
        out[d["dataset"]][key_of(d)].append({
            "seed": d.get("seed", 0),
            "val": (b.get("val") or {}).get(metric, float("nan")),
            "test": (b.get("test") or {}).get(metric, float("nan")),
            "params": d.get("params", 0),
            "epochs": d.get("epochs_run", 0),
            "wall": d.get("wall_sec", 0.0),
            "robust": {lv: ((b.get("robust") or {}).get(lv) or {}).get(metric, float("nan"))
                       for lv in ROBUST_LEVELS},
            "prefix": b.get("prefix") or {},
            "per_seq": (b.get("per_seq_test") or {}).get(metric),
            "latency": b.get("latency_ms_per_step"),
        })
    return out


def value_stats(entries: List[dict], field: str = "test") -> dict:
    v = np.array([e[field] for e in entries], dtype=np.float64)
    return M.bootstrap_ci(v)


def fmt(ci: dict, pct: bool = True) -> str:
    if not np.isfinite(ci.get("mean", float("nan"))):
        return "—"
    s = f"{ci['mean'] * 100:.2f} ± {ci['std'] * 100:.2f}" if pct else f"{ci['mean']:.4f}"
    return s


def better(a: float, b: float, higher_is_better: bool = True) -> str:
    if not (np.isfinite(a) and np.isfinite(b)):
        return "?"
    if abs(a - b) < 1e-9:
        return "="
    wins = (a > b) if higher_is_better else (a < b)
    return "▲" if wins else "▼"


# --------------------------------------------------------------------------- #
def table_main(data, ref_label: str, metric: str, higher_is_better=True,
               title="主对比", group_filter=None) -> str:
    lines = [f"### {title}", ""]
    for ds in sorted(data):
        lines.append(f"**{ds}**")
        lines.append("")
        lines.append("| 模型 | test " + metric + " (mean ± std) | 95% CI | val " + metric + " | params | epochs |")
        lines.append("|---|---|---|---|---|---|")
        hb = is_higher(metric)
        ordered = sorted(data[ds].items(),
                         key=lambda kv: value_stats(kv[1])["mean"], reverse=hb)
        for k, entries in ordered:
            lab = label_of(k)
            st = value_stats(entries, "test")
            sv = value_stats(entries, "val")
            p = int(np.mean([e["params"] for e in entries]))
            ep = int(np.mean([e["epochs"] for e in entries]))
            mark = " **(ref)**" if lab == ref_label else ""
            lines.append(f"| {lab}{mark} | {fmt(st)} | [{st['lo'] * 100:.2f}, {st['hi'] * 100:.2f}] "
                         f"| {fmt(sv)} | {p:,} | {ep} |")
        lines.append("")
    return "\n".join(lines)


def table_robust(data, ref_label: str, metric: str, title="鲁棒性") -> str:
    lines = [f"### {title}", "", "每格 = 相对 dense 的保持率（perturbed/dense），越高越鲁棒。", ""]
    for ds in sorted(data):
        labels = {label_of(k): v for k, v in data[ds].items()}
        if not labels:
            continue
        lines.append(f"**{ds}**")
        lines.append("")
        hdr = "| 模型 | " + " | ".join(ROBUST_LEVELS) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(ROBUST_LEVELS) + 1))
        for lab in sorted(labels):
            entries = labels[lab]
            dense = np.nanmean([e["robust"]["dense"] for e in entries])
            cells = []
            for lv in ROBUST_LEVELS:
                v = np.nanmean([e["robust"][lv] for e in entries])
                cells.append("1.000" if lv == "dense" else
                             (f"{v / dense:.3f}" if dense and np.isfinite(v) else "—"))
            mark = " **(ref)**" if lab == ref_label else ""
            lines.append(f"| {lab}{mark} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def table_diff(data, ref_label: str, metric: str, higher_is_better=True) -> str:
    """Paired difference vs reference, per dataset. Uses per-seq if available."""
    lines = ["### 相对参考模型的配对差异（正 = 更好）", "",
             "配对单位优先为**测试序列**（可信），否则为种子（n=3，必须注明局限）。", ""]
    for ds in sorted(data):
        by_lab = {label_of(k): v for k, v in data[ds].items()}
        if ref_label not in by_lab:
            continue
        ref_entries = by_lab[ref_label]
        lines.append(f"**{ds}**")
        lines.append("")
        lines.append("| 模型 | Δ vs " + ref_label + " | 95% CI | p | 效应量 d | 配对单位 | n |")
        lines.append("|---|---|---|---|---|---|---|")
        sign = 1.0 if is_higher(metric) else -1.0
        for lab in sorted(by_lab):
            if lab == ref_label:
                continue
            entries = by_lab[lab]

            # --- per-sequence pairing (preferred) ---
            ps_ref = ref_entries[0].get("per_seq")
            ps_new = entries[0].get("per_seq")
            if ps_ref and ps_new:
                common = sorted(set(ps_ref) & set(ps_new))
                if len(common) > 20:
                    a = np.array([ps_new[s] for s in common], dtype=np.float64) * sign
                    b = np.array([ps_ref[s] for s in common], dtype=np.float64) * sign
                    unit, n = "seq", len(common)
                    d = M.paired_bootstrap_diff(a, b, seed=0)
                    eff = M.cohens_d(a, b)
                    lines.append(f"| {lab} | {d['diff'] * 100:+.2f} | "
                                 f"[{d['lo'] * 100:+.2f}, {d['hi'] * 100:+.2f}] | "
                                 f"{d['p']:.4f} | {eff:+.2f} | {unit} | {n} |")
                    continue
            # --- per-seed pairing (fallback) ---
            seeds = sorted({e["seed"] for e in entries} & {e["seed"] for e in ref_entries})
            if len(seeds) >= 2:
                a = np.array([next(e["test"] for e in entries if e["seed"] == s)
                              for s in seeds]) * sign
                b = np.array([next(e["test"] for e in ref_entries if e["seed"] == s)
                              for s in seeds]) * sign
                d = M.paired_bootstrap_diff(a, b, seed=0)
                lines.append(f"| {lab} | {d['diff'] * 100:+.2f} | "
                             f"[{d['lo'] * 100:+.2f}, {d['hi'] * 100:+.2f}] | "
                             f"{d['p']:.4f} | {M.cohens_d(a, b):+.2f} | seed⚠ | {len(seeds)} |")
            else:
                lines.append(f"| {lab} | — | — | — | — | insufficient | 0 |")
        lines.append("")
    return "\n".join(lines)


def table_prefix(data, metric: str = "acc") -> str:
    lines = ["### 在线前缀准确率（因果，只用到当前为止的帧）", ""]
    for ds in sorted(data):
        lines.append(f"**{ds}**")
        lines.append("")
        lines.append("| 模型 | " + " | ".join(["p@10%", "p@25%", "p@50%", "p@75%", "p@100%"]) + " |")
        lines.append("|" + "---|" * 6)
        for k, entries in sorted(data[ds].items()):
            pref = entries[0].get("prefix") or {}
            if not pref:
                continue
            cells = []
            for key in ["p@0.1", "p@0.25", "p@0.5", "p@0.75", "p@1"]:
                v = pref.get(key, pref.get(key.replace("p@1", "p@1.0")))
                cells.append(f"{v * 100:.1f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—")
            lines.append(f"| {label_of(k)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def table_cost(data) -> str:
    lines = ["### 成本（参数量 / 推理延迟）", "",
             "| 数据集 | 模型 | params | ms/step | epochs |", "|---|---|---|---|---|"]
    for ds in sorted(data):
        for k, entries in sorted(data[ds].items()):
            p = int(np.mean([e["params"] for e in entries]))
            lat = [e["latency"] for e in entries if e.get("latency")]
            lat_s = f"{np.nanmean(lat):.3f}" if lat else "—"
            ep = float(np.mean([e["epochs"] for e in entries]))
            lines.append(f"| {ds} | {label_of(k)} | {p:,} | {lat_s} | {ep:.0f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="cfc")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--metric", default=None, help="override main metric (default: per-run main_metric)")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()

    runs = load_runs()
    if args.datasets:
        keep = set(args.datasets.split(","))
        runs = [r for r in runs if r["dataset"] in keep]
    if not runs:
        print("no completed runs found")
        return

    # group by (dataset, main metric) since metric differs per task
    by_ds: Dict[str, List[dict]] = defaultdict(list)
    for r in runs:
        by_ds[r["dataset"]].append(r)

    metric_of = {ds: rs[0].get("main_metric", "acc") for ds, rs in by_ds.items()}
    if args.metric:
        metric_of = {ds: args.metric for ds in by_ds}

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sections: List[str] = [f"# 结果汇总（自动生成 {contract.now_str()}）",
                          f"完成 run 数：{len(runs)}；数据集：{', '.join(sorted(by_ds))}", ""]
    machine: Dict[str, dict] = {}

    for ds in sorted(by_ds):
        metric = metric_of[ds]
        data = collect([r for r in runs if r["dataset"] == ds], metric)
        sections.append(table_main(data, args.ref, metric, title=f"{ds} 主对比（metric={metric}）"))
        sections.append(table_diff(data, args.ref, metric))
        sections.append(table_robust(data, args.ref, metric, title=f"{ds} 鲁棒性"))
        sections.append(table_prefix(data, metric))
        machine[ds] = {
            label_of(k): {"mean": value_stats(v)["mean"], "std": value_stats(v)["std"],
                          "n": len(v), "params": int(np.mean([e["params"] for e in v]))}
            for k, v in data[ds].items()
        }

    md = "\n".join(sections)
    (outdir / "summary_all.md").write_text(md, encoding="utf-8")
    contract.write_json(outdir / "summary.json", machine)
    print(f"wrote {outdir / 'summary_all.md'}  ({len(md)} chars) and summary.json")
    print(f"datasets: {sorted(by_ds)}")


if __name__ == "__main__":
    main()
