"""Audit cached robot CLIP features for leak-free decision probes.

This script deliberately does not train or score a classifier when each class is
represented by only one recording or the task lacks frame-level labels. It reuses
G:\\stream_memory_data\\features\\robot_gesture and writes a compact audit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_MANIFEST = Path(r"D:\robot_work\stream_memory_research\data_manifests\robot_gesture.json")
DEFAULT_OUT = Path(r"D:\robot_work\stream_memory_research\agent_feature_probe")

TASK_STATUS = {
    "fist": {
        "single_frame": "not_estimable_leak_free",
        "short_window": "not_estimable_leak_free",
        "long_window": "not_estimable_leak_free",
        "reason": "One scripted session contains the entire fist class; no independent positive session or frame labels.",
    },
    "knock": {
        "single_frame": "not_estimable_leak_free",
        "short_window": "not_estimable_leak_free",
        "long_window": "not_estimable_leak_free",
        "reason": "One session labeled 'two knocks' but no per-frame contact/phase labels or per-knock boundaries/count labels; no target coordinate labels.",
    },
    "wave": {
        "single_frame": "not_estimable_leak_free",
        "short_window": "not_estimable_leak_free",
        "long_window": "not_estimable_leak_free",
        "reason": "One session labeled only 'wave'; no per-frame left/right phase or cycle boundaries/count labels, and no independent wave sessions.",
    },
    "lean_out": {
        "single_frame": "not_estimable_leak_free",
        "short_window": "not_estimable_leak_free",
        "long_window": "not_estimable_leak_free",
        "reason": "One session labeled 'leave and return'; no per-frame present/leaving/absent/returning labels.",
    },
    "target_position_change": {
        "single_frame": "not_estimable",
        "short_window": "not_estimable",
        "long_window": "not_estimable",
        "reason": "No target position/box coordinates or changed-vs-stable segments are annotated in these sessions.",
    },
}


def summarize_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def unit_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = np.linalg.norm(a, axis=-1)
    bn = np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.maximum(an * bn, 1e-12)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(manifest: dict) -> dict:
    classes = list(manifest.get("extra", {}).get("classes", []))
    grouped = defaultdict(list)
    all_feat = []
    all_tok = []
    seq_rows = []
    errors = []

    for rec in manifest.get("seqs", []):
        path = Path(rec["path"])
        if not path.exists():
            errors.append({"seq_id": rec.get("seq_id"), "error": "missing_cache", "path": str(path)})
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                feat = np.asarray(z["feat"])
                tok = np.asarray(z["tok"]) if "tok" in z.files else None
                ts = np.asarray(z["t"], dtype=np.float64)
                labels = np.asarray(z["lab_class"]) if "lab_class" in z.files else None
        except Exception as exc:  # noqa: BLE001
            errors.append({"seq_id": rec.get("seq_id"), "error": type(exc).__name__, "detail": str(exc)[:200]})
            continue

        session = rec.get("meta", {}).get("session", rec.get("seq_id", "unknown"))
        task = rec.get("meta", {}).get("task", "unknown")
        split = rec.get("split", "unknown")
        if feat.ndim != 2 or ts.ndim != 1 or len(feat) != len(ts):
            errors.append({"seq_id": rec.get("seq_id"), "error": "shape_mismatch", "feat": list(feat.shape), "t": list(ts.shape)})
            continue
        if len(ts) > 1 and np.any(np.diff(ts) <= 0):
            errors.append({"seq_id": rec.get("seq_id"), "error": "non_increasing_timestamps"})
        if not np.isfinite(feat).all() or (tok is not None and not np.isfinite(tok).all()):
            errors.append({"seq_id": rec.get("seq_id"), "error": "non_finite_features"})

        row = {
            "seq_id": rec.get("seq_id"), "session": session, "task": task,
            "split": split, "frames": int(len(ts)), "feat_shape": list(feat.shape),
            "feat_dtype": str(feat.dtype), "tok_shape": list(tok.shape) if tok is not None else None,
            "tok_dtype": str(tok.dtype) if tok is not None else None,
            "label_values": sorted(int(v) for v in np.unique(labels)) if labels is not None else None,
            "dt_median_s": float(np.median(np.diff(ts))) if len(ts) > 1 else None,
        }
        seq_rows.append(row)
        grouped[session].append((task, split, feat.astype(np.float32), tok.astype(np.float32) if tok is not None else None, ts))
        all_feat.append(feat.astype(np.float32))
        if tok is not None:
            all_tok.append(tok.astype(np.float32))

    per_session = []
    class_sessions = defaultdict(set)
    class_frames = Counter()
    lag_cos = defaultdict(list)
    lag_delta_norm = defaultdict(list)
    lag_autocorr = defaultdict(list)
    tok_lag1_cos = []
    tok_lag1_delta = []

    for session, chunks in sorted(grouped.items()):
        chunks.sort(key=lambda x: x[0])
        task = chunks[0][0]
        feat = np.concatenate([x[2] for x in chunks], axis=0)
        tok_parts = [x[3] for x in chunks if x[3] is not None]
        tok = np.concatenate(tok_parts, axis=0) if len(tok_parts) == len(chunks) else None
        ts = np.concatenate([x[4] for x in chunks], axis=0)
        diffs = np.diff(ts)
        # Chunk-local timestamps restart at zero. Restore a continuous relative clock
        # using the measured median cadence, without changing frame order.
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.1
        t = np.arange(len(feat), dtype=np.float64) * dt
        class_sessions[task].add(session)
        class_frames[task] += int(len(feat))
        centered = feat - feat.mean(axis=0, keepdims=True)
        denom = np.sum(centered[:-1] ** 2) if len(feat) > 1 else 0.0
        local_cos = {}
        local_delta = {}
        for lag in (1, 2, 4, 8, 16):
            if len(feat) <= lag:
                continue
            c = unit_cos(feat[:-lag], feat[lag:])
            delta = feat[lag:] - feat[:-lag]
            local_cos[lag] = c
            local_delta[lag] = np.linalg.norm(delta, axis=-1)
            lag_cos[lag].extend(c.tolist())
            lag_delta_norm[lag].extend(local_delta[lag].tolist())
            if denom > 0:
                lag_autocorr[lag].append(float(np.sum(centered[:-lag] * centered[lag:]) / denom))
        if tok is not None:
            tok_flat = tok.reshape(len(tok), -1)
            tok_cos1 = unit_cos(tok_flat[:-1], tok_flat[1:]) if len(tok_flat) > 1 else np.array([])
            tok_delta1 = np.linalg.norm(tok_flat[1:] - tok_flat[:-1], axis=-1) if len(tok_flat) > 1 else np.array([])
            tok_lag1_cos.extend(tok_cos1.tolist())
            tok_lag1_delta.extend(tok_delta1.tolist())
        else:
            tok_cos1, tok_delta1 = np.array([]), np.array([])

        per_session.append({
            "session": session, "task": task, "frames_in_cache": int(len(feat)),
            "chunks": len(chunks), "manifest_split_chunks": sorted({x[1] for x in chunks}),
            "time_span_s_approx": float(max(0, len(feat) - 1) * dt),
            "dt_median_s": dt,
            "feat_frame_norm": summarize_values(np.linalg.norm(feat, axis=-1)),
            "feat_per_dim_std_mean": float(np.mean(np.std(feat, axis=0))),
            "feat_per_dim_std_median": float(np.median(np.std(feat, axis=0))),
            "feat_per_dim_std_p90": float(np.quantile(np.std(feat, axis=0), 0.90)),
            "feat_frame_mean_std": float(np.std(feat)),
            "feat_adjacent_cosine": summarize_values(local_cos.get(1, np.array([]))),
            "feat_lag4_cosine": summarize_values(local_cos.get(4, np.array([]))),
            "feat_adjacent_delta_l2": summarize_values(local_delta.get(1, np.array([]))),
            "tok_flat_dim": int(tok.shape[1] * tok.shape[2]) if tok is not None else None,
            "tok_adjacent_cosine": summarize_values(tok_cos1),
            "tok_adjacent_delta_l2": summarize_values(tok_delta1),
        })

    feat_matrix = np.concatenate(all_feat, axis=0) if all_feat else np.zeros((0, 0), np.float32)
    tok_matrix = np.concatenate([x.reshape(len(x), -1) for x in all_tok], axis=0) if all_tok else np.zeros((0, 0), np.float32)
    feat_var = np.var(feat_matrix, axis=0) if len(feat_matrix) else np.zeros(0)
    tok_var = np.var(tok_matrix, axis=0) if len(tok_matrix) else np.zeros(0)
    feat_var_sum = float(np.sum(feat_var))
    tok_var_sum = float(np.sum(tok_var))
    # Participation ratio over marginal per-coordinate variances, explicitly not
    # a covariance/effective-rank estimate.
    feat_dispersion_index = float(feat_var_sum ** 2 / max(float(np.sum(feat_var ** 2)), 1e-20)) if len(feat_var) else None
    tok_dispersion_index = float(tok_var_sum ** 2 / max(float(np.sum(tok_var ** 2)), 1e-20)) if len(tok_var) else None

    sessions_by_task = {task: sorted(values) for task, values in sorted(class_sessions.items())}
    task_status = dict(TASK_STATUS)
    for task in classes:
        if task not in task_status:
            task_status[task] = {
                "single_frame": "not_estimable_leak_free",
                "short_window": "not_estimable_leak_free",
                "long_window": "not_estimable_leak_free",
                "reason": "Only one independent scripted source/session for this label; the available chunks are from the same recording.",
            }

    return {
        "dataset": manifest.get("dataset"),
        "encoder": manifest.get("encoder"),
        "declared_feature_dim": manifest.get("feature_dim"),
        "declared_tok_dim": manifest.get("tok_dim"),
        "manifest_rows": len(manifest.get("seqs", [])),
        "source_declared_frames": manifest.get("extra", {}).get("n_frames_encoded"),
        "loaded_sequences": len(seq_rows),
        "loaded_frames": int(sum(r["frames"] for r in seq_rows)),
        "cache_frame_deficit_vs_source": (int(manifest.get("extra", {}).get("n_frames_encoded", 0)) - int(sum(r["frames"] for r in seq_rows))) if manifest.get("extra", {}).get("n_frames_encoded") is not None else None,
        "unique_sessions": len(grouped),
        "errors": errors,
        "sequence_rows": seq_rows,
        "sessions": per_session,
        "class_sessions": sessions_by_task,
        "class_session_counts": {k: len(v) for k, v in sessions_by_task.items()},
        "class_cached_frame_counts": dict(sorted(class_frames.items())),
        "feature_stats": {
            "feat_shape_per_frame": list(feat_matrix.shape[1:]) if len(feat_matrix) else None,
            "feat_dim": int(feat_matrix.shape[1]) if len(feat_matrix) else None,
            "tok_shape_per_frame": [4, int(manifest.get("tok_dim") or 0)] if len(tok_matrix) else None,
            "tok_flat_dim": int(tok_matrix.shape[1]) if len(tok_matrix) else None,
            "feat_global_frame_norm": summarize_values(np.linalg.norm(feat_matrix, axis=-1)) if len(feat_matrix) else None,
            "feat_global_per_dim_std_mean": float(np.mean(np.std(feat_matrix, axis=0))) if len(feat_matrix) else None,
            "feat_global_per_dim_std_median": float(np.median(np.std(feat_matrix, axis=0))) if len(feat_matrix) else None,
            "feat_marginal_variance_dispersion_index_not_rank": feat_dispersion_index,
            "tok_global_per_dim_std_mean": float(np.mean(np.std(tok_matrix, axis=0))) if len(tok_matrix) else None,
            "tok_marginal_variance_dispersion_index_not_rank": tok_dispersion_index,
            "adjacent_cosine_by_lag": {str(lag): summarize_values(np.asarray(vals)) for lag, vals in sorted(lag_cos.items())},
            "adjacent_delta_l2_by_lag": {str(lag): summarize_values(np.asarray(vals)) for lag, vals in sorted(lag_delta_norm.items())},
            "within_session_temporal_autocorr_by_lag": {str(lag): summarize_values(np.asarray(vals)) for lag, vals in sorted(lag_autocorr.items())},
            "tok_adjacent_cosine": summarize_values(np.asarray(tok_lag1_cos)),
            "tok_adjacent_delta_l2": summarize_values(np.asarray(tok_lag1_delta)),
        },
        "decision_task_probe_status": task_status,
        "class_split_audit": {
            "all_labels_have_multiple_independent_sessions": bool(sessions_by_task) and all(len(v) >= 2 for v in sessions_by_task.values()),
            "usable_for_supervised_leave_session_out_class_probe": False,
            "reason": "Each behavior class is a single scripted recording. The existing train/val/test chunks are adjacent portions of that same source, so using those splits for generalization would leak session identity and scene conditions.",
        },
        "existing_pixel_run_audit": {
            "scope": "Existing runs_pixel robot fold metrics, no new training.",
            "completed_folds_observed": [0, 1, 2, 3, 4],
            "fold5_state": "config exists but train.log is empty; no metrics.json or checkpoint observed",
            "protocol_finding": "RobotSessions(held_out) excludes held_out sessions. train_pixel_event.py constructs both tr and te with the same held list, and sets va=te; consequently internal reported test/validation windows are from the same nine sessions used for training, not from held_out. eval_pixel_event.eval_robot also calls RobotSessions([held]) and therefore excludes held; its CLI currently calls eval_robot with one extra positional argument and existing eval.log shows TypeError.",
            "usable_as_heldout_result": False,
            "metrics_context": "Completed fold metrics report gesture_acc 0.1607-0.1791 and presence_macro_f1 0.8199-0.8857, but they are not valid held-out estimates and presence labels are session-constant proxy labels.",
        },
    }


def write_report(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fs = result["feature_stats"]
    lines = [
        "# Cached Robot CLIP Feature Probe Audit",
        "",
        "## Scope",
        "",
        "Read-only analysis of existing `robot_gesture` frozen CLIP features. No downloads, re-encoding, firmware edits, or model training. This is an identifiability audit plus temporal feature statistics, not a supervised accuracy claim.",
        "",
        "## Data and feature dimensions",
        "",
        f"- Encoder: `{result.get('encoder', {}).get('name')}`; input `{result.get('encoder', {}).get('input')}px`.",
        f"- Loaded `{result['loaded_sequences']}/{result['manifest_rows']}` sequences, `{result['loaded_frames']}` cached frames, `{result['unique_sessions']}` independent scripted sessions; manifest source count is `{result.get('source_declared_frames')}` frames (cache shortfall `{result.get('cache_frame_deficit_vs_source')}`).",
        f"- Per-frame global CLS dimension: `{fs.get('feat_dim')}`; patch tokens: `{fs.get('tok_shape_per_frame')}` = `{fs.get('tok_flat_dim')}` flattened values.",
        f"- Mean frame L2 norm: `{fs.get('feat_global_frame_norm', {}).get('mean')}`; median per-coordinate std: `{fs.get('feat_global_per_dim_std_median')}`; marginal-variance dispersion index: `{fs.get('feat_marginal_variance_dispersion_index_not_rank')}` (not a covariance rank).",
        f"- Adjacent CLS cosine: `{fs.get('adjacent_cosine_by_lag', {}).get('1', {}).get('mean')}`; lag-4 cosine: `{fs.get('adjacent_cosine_by_lag', {}).get('4', {}).get('mean')}`; lag-1 CLS delta L2: `{fs.get('adjacent_delta_l2_by_lag', {}).get('1', {}).get('mean')}`.",
        f"- Adjacent flattened patch-token cosine: `{fs.get('tok_adjacent_cosine', {}).get('mean')}`; lag-1 token delta L2: `{fs.get('tok_adjacent_delta_l2', {}).get('mean')}`.",
        f"- CLS cosine across lags 1/2/4/8/16: `{[round(fs['adjacent_cosine_by_lag'][str(k)]['mean'], 3) for k in (1, 2, 4, 8, 16)]}`; delta L2: `{[round(fs['adjacent_delta_l2_by_lag'][str(k)]['mean'], 3) for k in (1, 2, 4, 8, 16)]}`.",
        "",
        "## Decision-task probe results",
        "",
        "No leak-free single-frame/short-window/long-window accuracy is identifiable from the existing robot cache. There is one session per scripted behavior, and labels are constant per session. The cache split's chunks are contiguous pieces of that same recording; treating them as independent train/test examples would measure recording/background/timing memorization.",
        "",
        "| Task | Single frame | Short window | Long window | Why no valid score |",
        "|---|---|---|---|---|",
    ]
    for task, status in result["decision_task_probe_status"].items():
        lines.append(f"| {task} | {status['single_frame']} | {status['short_window']} | {status['long_window']} | {status['reason']} |")
    lines.extend([
        "",
        "## Label-free within-session feature change",
        "",
        "These are descriptive statistics over the available cached prefix of each session, not task scores. The 0.1 s frame cadence implies lag 1 is about 0.1 s and lag 4 about 0.4 s.",
        "",
        "| Session task | Cached frames | CLS lag-1 cosine | CLS lag-4 cosine | CLS lag-1 delta L2 | Patch-token lag-1 cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in result["sessions"]:
        if row["task"] in {"fist", "knock", "wave", "lean_out", "none", "open_palm"}:
            lines.append(
                f"| {row['task']} | {row['frames_in_cache']} | "
                f"{row['feat_adjacent_cosine']['mean']:.3f} | "
                f"{row['feat_lag4_cosine']['mean']:.3f} | "
                f"{row['feat_adjacent_delta_l2']['mean']:.3f} | "
                f"{row['tok_adjacent_cosine']['mean']:.3f} |"
            )
    lines.extend([
        "",
        "## What the temporal statistics do and do not say",
        "",
        "Adjacent/lagged cosine and delta norms quantify that embeddings change over time. They do not identify fists, event phases, person-return states, or target coordinates: without corresponding frame/interval labels, high temporal variation may be hand motion, camera/scene changes, or compression noise.",
        "",
        "## Reusable prior evidence (different task)",
        "",
        "The existing `ucf101_order` study is a valid generic temporal-order check: it pairs forward/reversed versions of each source clip within the same split, so single-frame features are exactly label-ambiguous; the report records framewise MLP at 0.5000 and temporal DSM at 0.9429 (GRU 0.9280). This shows that frozen CLIP sequences retain generic motion/order evidence, but it does not establish fist, knock-phase, leave-return, or target-position information for the robot scene.",
        "",
        "## Existing pixel-run audit",
        "",
        "Pixel robot runs are not a usable held-out reference despite their fold names. The dataset constructor excludes the supplied `held_out` set, the training script passes the same held label to both `tr` and `te` and sets validation equal to test, and the evaluator again constructs `RobotSessions([held])` (which excludes it). The observed completed robot pixel_event folds f0-f4 report gesture_acc 0.161-0.179 and presence macro-F1 0.820-0.886, but these are same-session training/validation windows; `eval.log` additionally shows `TypeError: eval_robot() ... 4 were given`. Fold f5 has a config but empty train log and no observed metrics/checkpoint. These numbers are not valid held-out performance.",
        "",
        "## Label-free reproducible feature-change path",
        "",
        "The script `run_cached_probe_audit.py` reads the existing NPZs, concatenates chunks only within their original session, computes CLS and flattened patch-token cosine at lags 1/2/4/8/16, frame-to-frame L2 deltas, within-session centered autocorrelation, per-dimension temporal standard deviations, frame norms, and timestamp cadence. It makes no fitted classifier and uses no task labels to claim semantics. Re-run with `D:\\smartcar\\.venv\\Scripts\\python.exe agent_feature_probe\\run_cached_probe_audit.py`. This describes whether the features change over time, not what those changes mean.",
        "",
        "## Failures / next data needed",
        "",
        "- `fist`: record multiple independently started sessions for fist and matched non-fist hands; annotate visibility/gesture per frame. Current clip has one fist session only.",
        "- `knock`: annotate each knock's approach/contact/rebound/onset/completion and desired table target coordinate; the current `knock` session has only a clip-level 'two knocks' label, and only 128 of its 189 source frames are in the feature cache.",
        "- `wave`: annotate left/right phase and each completed cycle boundary/count across multiple independently scripted sessions; current clip has only a clip-level 'wave' label.",
        "- `leave/return`: annotate present/leaving/absent/returning intervals across multiple sessions with different durations and backgrounds; current script label is not a framewise state.",
        "- `target position change`: annotate coordinates and stable/change intervals; no position labels exist.",
        "- Split all variants by recording/session before training; preserve a final held-out session set. A minimum probe should compare framewise linear probe, causal 0.3-0.5 s window probe, and full-prefix causal probe on identical held-out sessions, selecting regularization/window on validation sessions only.",
        "",
        f"Cache errors: `{len(result['errors'])}`. Full per-sequence/session statistics are in `results.json`.",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    result = analyze(manifest)
    result["manifest_path"] = str(args.manifest)
    write_report(result, args.out)
    print(json.dumps({
        "manifest_rows": result["manifest_rows"],
        "loaded_sequences": result["loaded_sequences"],
        "loaded_frames": result["loaded_frames"],
        "unique_sessions": result["unique_sessions"],
        "feat_dim": result["feature_stats"]["feat_dim"],
        "tok_flat_dim": result["feature_stats"]["tok_flat_dim"],
        "adjacent_cosine_mean": result["feature_stats"]["adjacent_cosine_by_lag"].get("1", {}).get("mean"),
        "errors": len(result["errors"]),
        "out": str(args.out),
    }, indent=2))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
