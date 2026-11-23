# -*- coding: utf-8 -*-
"""E2E 真机录像回放总结器：解析 pc_unified_brain_app --source video 的
JSON-lines 日志，输出敲火事件 vs GT knock 区间、模式迁移、重定位质量、
GeoNet 时延统计 -> runs_end_side/e2e_replay/<take>_summary.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _load_label_names() -> dict[str, str]:
    """dataset/labels.json 的 id->中文名映射（自定义 label_<ts> 也有实名，如'连续敲击'）。"""
    try:
        items = json.loads((REPO / "dataset" / "labels.json").read_text(encoding="utf-8"))
        return {x["id"]: x.get("name", x["id"]) for x in items}
    except Exception:
        return {}


def summarize(log_path: Path, ann_path: Path, fps: float = 15.0, pad_s: float = 0.6) -> dict:
    recs = [json.loads(l) for l in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip().startswith("{")]
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    names = _load_label_names()
    all_segs = ann.get("annotations", [])
    gt = [(float(a["start_s"]), float(a["end_s"])) for a in all_segs if a.get("label") == "knock"]
    other = [(a.get("label", "?"), float(a["start_s"]), float(a["end_s"])) for a in all_segs if a.get("label") != "knock"]

    fires = [(r["frame"], r["frame"] / fps) for r in recs if r.get("knock", {}).get("fired")]
    hits, fp, amb, matched = [], [], [], set()
    for fr, t in fires:
        hit = next((i for i, (s0, s1) in enumerate(gt) if s0 - pad_s <= t <= s1 + pad_s), None)
        if hit is not None:
            matched.add(hit)
            hits.append({"frame": fr, "t_s": round(t, 2), "gt_idx": hit})
            continue
        seg = next((s for s in other if s[1] - pad_s <= t <= s[2] + pad_s), None)
        if seg is not None:
            amb.append({"frame": fr, "t_s": round(t, 2), "inside_label": names.get(seg[0], seg[0]), "label_id": seg[0]})
        else:
            fp.append({"frame": fr, "t_s": round(t, 2)})
    misses = [{"gt_idx": i, "gt_s": list(g)} for i, g in enumerate(gt) if i not in matched]

    modes: dict[str, int] = {}
    geoms, inls, icps = [], [], []
    transitions = []
    prev = None
    for r in recs:
        m = r.get("mode", "?")
        modes[m] = modes.get(m, 0) + 1
        if m != prev:
            transitions.append({"frame": r["frame"], "mode": m})
            prev = m
        if r.get("geonet_ms"):
            geoms.append(r["geonet_ms"])
        rl = r.get("reloc") or {}
        if rl.get("ransac_inliers"):
            inls.append(rl["ransac_inliers"])
        if rl.get("icp_res_mm"):
            icps.append(rl["icp_res_mm"])

    import statistics as st
    return {
        "log": str(log_path), "n_frames_logged": len(recs), "fps_assumed": fps,
        "gt_knocks": len(gt), "fires": len(fires),
        "hits": hits, "false_positives": fp, "ambiguous_in_other_labels": amb, "misses": misses,
        "mode_frame_counts": modes, "mode_transitions": transitions,
        "geonet_ms": {"median": round(st.median(geoms), 2), "p90": round(st.quantiles(geoms, n=10)[8], 2)} if geoms else None,
        "reloc_inliers_median": round(st.median(inls), 1) if inls else None,
        "icp_res_mm_median": round(st.median(icps), 2) if icps else None,
        "tier": "T2-real-video-replay",
    }


def main() -> None:
    log = Path(sys.argv[1])
    take = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    if take is None:
        # 默认从日志文件名推 take：e2e_replay_<tid>_full.log
        tid = log.stem.replace("e2e_replay_", "").replace("_full", "")
        take = REPO / "dataset" / "takes" / tid
    out = summarize(log, take / "annotation.json", fps)
    out_dir = REPO / "runs_end_side" / "e2e_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{take.name}_summary.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_frames_logged", "gt_knocks", "fires", "mode_frame_counts")}, ensure_ascii=False))
    print(f"[e2e_summary] -> {p}")


if __name__ == "__main__":
    main()
