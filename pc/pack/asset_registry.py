# -*- coding: utf-8 -*-
"""pc/pack/asset_registry.py - what data, labels, models and public sets exist, declared once.

Every group says where it is, where it came from, under what license, whether
it can contain people (privacy), and which split policy governs it. The
inventory script only walks these declarations; nothing is discovered by
guessing at run time, so a new asset that is not registered here is visibly
absent from the pack.

License keys
  user-own          recorded by the user from their own robot; no third-party rights
  repo-public       already published in the public git repository by the user's decision
  third-party-*     public dataset; the license itself is tracked in LICENSES.md
  upstream-model    pretrained weights redistributed by their authors under their own license
  unverified        a license statement exists only on a secondary page; must be verified
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Group:
    id: str
    title: str
    kind: str                      # real_robot | label | derived | model | public | pretrained
    paths: list[str]               # repo-relative directories or globs; absolute if a drive letter is given
    source: str
    license: str
    privacy: str                   # FACE-POSSIBLE | NONE | UNKNOWN
    split: str
    frames_glob: str = "**/*.jpg"  # what counts as a frame
    hash: bool = False             # sha256 every file (also enables exact-duplicate detection)
    subdirs: bool = False          # report each immediate sub-directory (one take / session / label set) separately
    notes: str = ""
    exists: bool = False
    n_files: int = 0
    bytes: int = 0
    ext: dict = field(default_factory=dict)
    n_frames: int = 0
    sub: list = field(default_factory=list)
    missing_paths: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (v if not isinstance(v, dict) else dict(v)) for k, v in self.__dict__.items()}


def _g(*a, **k) -> Group:
    return Group(*a, **k)


SEALED_NOTE = "cb8455 is held out: never trained on, never used to pick anything (GOAL section 9)."

GROUPS: list[Group] = [
    # ---------------------------------------------------------------- real-robot captures
    _g("R01", "4 annotated takes of 2026-11-07: raw frames + capture timestamps", "real_robot",
       ["video_annotation_data/videos/robot_20261107_*_source"],
       source="robot camera GET /capture, 240x240 native grayscale JPEG, annotator service on port 8780",
       license="user-own", privacy="FACE-POSSIBLE", split="dev pool: leave-one-take-out over 5 takes (see splits/)",
       frames_glob="frames/*.jpg", hash=True, notes="originals; never moved or rewritten by the pack"),
    _g("R02", "2 unannotated debug takes of 2026-11-05: raw frames", "real_robot",
       ["video_annotation_data/videos/robot_20261105_*_source"],
       source="robot camera, early debugging sessions", license="user-own", privacy="FACE-POSSIBLE",
       split="excluded from training (dataset/README: debug); usable as background negatives only after review",
       frames_glob="frames/*.jpg", hash=True),
    _g("R03", "published dataset: 6 takes as 15-20 fps MP4 + interval annotations", "label",
       ["dataset/takes"],
       source="derived from R01 and from the two 2026-11-15 takes; tracked in the public repository",
       license="repo-public (repository has no LICENSE file)", privacy="FACE-POSSIBLE",
       split="4 of 11-07 + 65a54d in the dev pool; " + SEALED_NOTE, frames_glob="**/*.mp4", subdirs=True,
       notes="the two 11-15 takes (65a54d, cb8455) have no raw-frame directory on this machine: the MP4 is the only copy"),
    _g("R04", "evidence/live: background captures of 2026-11-03", "real_robot", ["evidence/live"],
       source="background capture scripts, no action script (AGENTS 5.4-39: context only)",
       license="user-own", privacy="FACE-POSSIBLE",
       split="evidence/live/fov_sweep is SEALED; the rest is early unscripted footage, context only",
       frames_glob="**/*.jpg", hash=True, subdirs=True),
    _g("R05", "evidence/sessions: 10 controlled gesture sessions at 0.5 m, night", "real_robot",
       ["evidence/sessions"],
       source="controlled sessions with manifest.jsonl (task, label, distance, light, frames)",
       license="user-own", privacy="FACE-POSSIBLE",
       split="empty_d0.5_front_night_182201 is SEALED; the other 9 are external negatives / gesture positives",
       frames_glob="**/*.jpg", hash=True, subdirs=True),
    _g("R06", "live dumps of the unified brain (9 sessions + v25 verification)", "real_robot",
       ["runs_end_side/live_dump_s*", "runs_end_side/live_verify_v25*"],
       source="pc_unified_brain_app.py on the real robot; frames every 3rd capture + per-frame info.jsonl",
       license="user-own", privacy="FACE-POSSIBLE",
       split="unassigned: must be split by session (s-number), never by frame", frames_glob="**/*.jpg", hash=True),
    _g("R07", "hardware characterisation standing on the desk, 2026-11-23", "real_robot",
       ["runs_end_side/hw_char_upright"],
       source="35 head-servo steps + teacher_log.json (624 rows)", license="user-own", privacy="FACE-POSSIBLE",
       split="identification only; never a training input", frames_glob="**/*.jpg", hash=True),
    _g("R08", "s2r acceptance runs 2026-11-23/24", "real_robot", ["runs_end_side/s2r_morning"],
       source="morning_check.py and live.py --mode gaze", license="user-own", privacy="FACE-POSSIBLE",
       split="T4 evidence and identification only", frames_glob="**/*.jpg", hash=True),
    _g("R09", "s2r probes and guarded sessions 2026-11-24/25 (500 Hz telemetry)", "real_robot",
       ["runs_end_side/s2r_live"],
       source="readonly_probe.py and guarded_session.py; tel.csv holds all 34 telemetry floats",
       license="user-own", privacy="FACE-POSSIBLE (frames); telemetry itself has no personal content",
       split="identification / holdout for the digital twin (see twin_splits.json)", frames_glob="**/*.jpg",
       hash=True),
    _g("R10", "hardware calibration lying on the bench, 2026-11-23", "real_robot", ["runs_end_side/hw_calib"],
       source="side-lying servo tests (not valid for the standing robot, AGENTS 5.8-47)", license="user-own",
       privacy="FACE-POSSIBLE", split="identification only; superseded by R07", frames_glob="**/*.jpg", hash=True),
    _g("R11", "Label Studio task store and export", "label", ["labelstudio/data"],
       source="label-studio instance on port 3334", license="user-own", privacy="FACE-POSSIBLE",
       split="labels only: the images are byte-identical copies of R04/R05 frames (inventory duplicate report)",
       frames_glob="**/*.jpg", hash=True, subdirs=True,
       notes="never count these as additional images; use them for the annotations they carry"),
    _g("R12", "early hand images with YOLO-format labels", "label", ["data"],
       source="early hand/fist detection attempt (data/done, data/labeled_inbox, data/dataset/data.yaml)",
       license="user-own", privacy="UNKNOWN",
       split="labels only: the images are byte-identical copies of R04/R05 frames (inventory duplicate report)",
       frames_glob="**/*.jpg", hash=True, subdirs=True),
    _g("R13", "real 360-degree panoramas of the room", "real_robot", ["pc/bench/_room360_real_panos"],
       source="phone panoramas of the user's room", license="user-own", privacy="FACE-POSSIBLE",
       split="not for the three tracks", frames_glob="**/*.jpg", hash=True),
    _g("R14", "old pipeline recordings (npz of observations/features)", "real_robot", ["pc/train/data_*"],
       source="session_from_robot.py recordings of the first architecture; features and labels, few pixels",
       license="user-own", privacy="UNKNOWN",
       split="as named by the directories (data_pan_holdout2 is a holdout of that pipeline)", frames_glob=None,
       hash=False),
    _g("R15", "teacher caches (labels recomputed from pixels)", "derived",
       ["pc/bench/_teacher_cache", "pc/bench/_knock3d_cache_v4"],
       source="offline teacher / model outputs cached for training", license="user-own", privacy="UNKNOWN",
       split="inherits the split of its source takes; must never point at SEALED material", frames_glob=None,
       hash=False),
    # ---------------------------------------------------------------- models
    _g("M01", "double-knock detectors (2D stream, deploy checkpoints)", "model",
       ["runs_knock/double_knock_*"], source="pc/train/train_double_knock.py leave-one-take-out runs",
       license="user-own", privacy="NONE", split="models: evaluated on held-out takes only", frames_glob=None),
    _g("M02", "3D double-knock stream models (physnet family)", "model", ["runs_end_side/double_knock_3d_*"],
       source="pc/end_side/train_double_knock_3d_*.py", license="user-own", privacy="NONE",
       split="models: evaluated on held-out takes only", frames_glob=None),
    _g("M03", "HoverGeoNet depth + dense splat student + mono depth", "model",
       ["runs_end_side/hover_geonet_240", "runs_end_side/hover_geonet_mv_v4", "runs_end_side/hover_geonet_vit_240",
        "runs_end_side/dense_splat_v3", "runs_end_side/mono_depth", "runs_end_side/neural_3dgs"],
       source="pc/end_side/train_*.py", license="user-own", privacy="NONE",
       split="models; trained on rendered tables + ~1087 frames of one real desk", frames_glob=None),
    _g("M04", "end-side policy bundle (tflite) and skill zoo", "model",
       ["runs_end_side/hover_rl_policy", "runs_end_side/skill_zoo", "runs_end_side/hover_fine_policy_v3"],
       source="pc/end_side/train_rl_policy.py, multi_skill_rl.py", license="user-own", privacy="NONE",
       split="models (simulation-trained; no T4)", frames_glob=None),
    _g("M05", "gesture classifier + legacy policy blobs", "model",
       ["pc/gesture/models", "pc/train/out_*", "pc/escalate"],
       source="pc/gesture, pc/train, pc/escalate", license="user-own", privacy="NONE",
       split="models of the earlier pipelines; kept for regression gates", frames_glob=None),
    # ---------------------------------------------------------------- pretrained (upstream)
    _g("U01", "YuNet face detector (onnx)", "pretrained", ["pc/perception/assets/face_detection_yunet_2023mar.onnx"],
       source="OpenCV zoo", license="upstream-model (see LICENSES.md)", privacy="NONE", split="n/a", frames_glob=None),
    _g("U02", "YOLO26n-pose weights", "pretrained", ["pc/perception/assets/yolo26n-pose.pt"],
       source="ultralytics", license="upstream-model (AGPL-3.0 or Ultralytics enterprise; see LICENSES.md)",
       privacy="NONE", split="n/a", frames_glob=None),
    _g("U03", "YOLOE-26n-seg weights + text encoder", "pretrained",
       ["pc/perception/assets/yoloe-26n-seg.pt", "mobileclip2_b.ts"], source="ultralytics",
       license="upstream-model (AGPL-3.0 or Ultralytics enterprise; see LICENSES.md)", privacy="NONE",
       split="n/a", frames_glob=None, notes="mobileclip2_b.ts is 242 MB and gitignored; s2r/live.py chdir()s to the repo root so ultralytics finds it instead of downloading"),
    # ---------------------------------------------------------------- public data on disk
    _g("P01", "HaGRID sample 30k @384p (18 classes incl. fist/palm/stop/one/no_gesture)", "public",
       [r"G:\stream_memory_data\gesture_raw\hagrid\hagrid-sample-30k-384p"],
       source="Kaggle sample of the HaGRID dataset (hukenovs/hagrid)", license="unverified: Kaggle page says CC BY-SA 4.0; primary LICENSE text to be confirmed",
       privacy="UNKNOWN (third-party hand images)", split="by subject if the annotations allow; never mixed with real-robot folds",
       frames_glob="**/*.jpg", hash=True),
    _g("P02", "action / segmentation archives (Charades, UCF101, DAVIS, MOT17)", "public",
       [r"G:\stream_memory_data\videos"], source="their respective publishers",
       license="unverified: each archive must be checked against its own LICENSE before use",
       privacy="UNKNOWN", split="external negatives for zero-false-alarm stress tests only", frames_glob="**/*.jpg",
       hash=False, notes="28.6 GiB, mostly zips; inventoried by size only"),
    _g("P03", "synthetic gesture sequences generated by the project", "derived",
       [r"G:\stream_memory_data\synth"], source="pc/train/synth.py and gesture synthesis", license="user-own",
       privacy="NONE", split="synthetic variants must not cross folds (generator seed groups)", frames_glob=None,
       hash=False),
    _g("P04", "research bulk (features, pixel caches, research checkpoints)", "derived",
       [r"G:\stream_memory_data\features", r"G:\stream_memory_data\pixel_cache",
        r"G:\stream_memory_data\research_checkpoints", r"G:\stream_memory_data\cache", r"G:\stream_memory_data\tmp",
        r"G:\stream_memory_data\logs"],
       source="stream_memory_research runs", license="user-own", privacy="UNKNOWN",
       split="out of scope: never uploaded (GOAL 1.7 forbids moving the 45 GiB wholesale)", frames_glob=None,
       hash=False),
]
