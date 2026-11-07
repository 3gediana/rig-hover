# dataset —— 真机敲击数据集样例与标注规范

这个目录是**录完即标的合并服务**（`pc/tools/video_annotator.py`，端口 8780）
产出的 take 的 git 版存档。它解决两件事：

1. 给接下来接手机器人的队友一个**可对照的标注范本**
   （`takes/robot_20261107_204337_5ca36d/`）；
2. 给多人标注提供**统一格式和统一标签**，让不同人标出来的区间能直接合并训练。

2026-11-08 用户已把 4 条 take 全部标注（77 个区间），并跑通了
"区间标注 → 逐帧敲击检测"的训练流水线，见
[`docs/TRAIN-PIPELINE.md`](docs/TRAIN-PIPELINE.md)。

> 仓库是 **public** 的。这里的视频能看到操作者的手、桌面和一部分房间背景。
> 新增 take 之前先确认可以公开；不要往里放人脸特写、证件、屏幕内容等隐私信息。

---

## 目录结构

```
dataset/
├── README.md                     本文
├── INDEX.md                      自动生成的 take 索引（时长/帧率/标注区间数）
├── labels.json                   标签表（默认 6 类 + 用户自定义标签）
├── takes/
│   └── <take_id>/
│       ├── take.mp4              服务发布的 H.264 视频（240×240，CFR）
│       ├── annotation.json       时间区间标注（未标注的 take 没有这个文件）
│       ├── meta.json             录制元数据：frames/video_frames/duration_s/fps/capture_fps
│       └── take.json             导出摘要（给 INDEX.md 用）
└── docs/
    ├── ROBOT-SETUP.md            机器人 + 热点 + 采集服务的上手指南（队友先读这份）
    ├── COLLECTION-AND-ANNOTATION.md  采什么、怎么标、标多少
    ├── EXAMPLE-TAKE.md           已标注范本 take 的逐区间说明与合并建议
    └── TRAIN-PIPELINE.md         区间标注 → 逐帧敲击检测的训练/评估流水线
```

**没有放进 git 的东西**：原始 JPEG 帧（`<take_id>_source/frames/*.jpg`）和
`frames.jsonl` 时间戳旁车。一条 190 s 的 take 原图约 50 MB，且含个人影像，
留在录制电脑的 `video_annotation_data/videos/<take_id>_source/` 下。
需要像素级训练（人脸/拳头检测）时应在录制机器上用原图，而不是拿这里的 MP4 反推。

---

## 收录规则

| take | 是否收录 | 说明 |
|---|---|---|
| `robot_20261107_204337_5ca36d` | ✅ 已收录 | 98.85 s，**12 个区间，作为标注范本（教材）** |
| `robot_20261107_212119_67f0c1` | ✅ 已收录 | 190.45 s，26 个区间（16 个敲击段） |
| `robot_20261107_213314_5c0158` | ✅ 已收录 | 190.7 s，12 个区间 |
| `robot_20261107_214950_1f30d1` | ✅ 已收录 | 159.55 s，27 个区间 |
| `robot_20261105_004630_ef1d19` | ❌ 排除 | 2026-11-05 调试期误录的 374 s 长片，无剧本 |
| `robot_20261105_190704_3e80b8` | ❌ 排除 | 旧 pacing 管线录的 17.4 s 测试片 |

以后新录的 take 默认都该进来，但**只在用户确认内容可用之后**才导出。

---

## 怎么把新 take 放进仓库

采集服务把视频放在 `D:\robot_work\video_annotation_data\videos\`。
导出命令（在仓库根目录跑）：

```powershell
# 指定 take（文件名或 take_id 都可以，可重复 --video）
& 'G:\ai\python11\python.exe' pc\tools\export_take.py `
    --video robot_20261107_204337_5ca36d `
    --video robot_20261107_212119_67f0c1 `
    --video robot_20261107_213314_5c0158 `
    --index

# 或者按日期批量导出，再显式排除不要的
& 'G:\ai\python11\python.exe' pc\tools\export_take.py `
    --since 2026-11-07 `
    --exclude robot_20261105_004630_ef1d19 `
    --exclude robot_20261105_190704_3e80b8 `
    --index
```

脚本只复制 `take.mp4` / `annotation.json` / `meta.json` / `take.json`，
并刷新 `labels.json` 与 `INDEX.md`；原图旁车永远不会被复制。

---

## 标注文件格式

`annotation.json`（与服务的 `annotations/<sha256(video_name)[:20]>.json` 内容一致）：

```json
{
  "schema_version": 1,
  "video": "robot_20261107_204337_5ca36d.mp4",
  "duration_s": 98.85,
  "annotations": [
    {"id": "seg_mw6iyta2_ifdzg", "label": "knock", "start_s": 3.83, "end_s": 8.32, "note": ""}
  ],
  "updated_at": "2026-11-07T12:43:00+00:00"
}
```

* **时间基准 = 发布 MP4 的时间轴**（第 0 帧为 0 s，CFR）。
  视频由采集帧按最近邻重采样成 15/20 fps，时间戳与真实采集时刻的偏差
  不超过半个输出帧（15 fps 时 ≈33 ms）。
* `label` 必须是 `labels.json` 里的 `id`；`start_s < end_s`，单位秒。
* `note` 可选，用来写“这里其实是两次敲击”“手挡住桌面”等备注。

`labels.json` 是标签表：`[{"id": "...", "name": "...", "color": "#rrggbb"}]`。
默认六类：`knock` 敲击、`fist_hold` 静止握拳、`hand_raise` 抬手/举拳、
`reach` 伸手但未敲、`other` 其它动作、`background` 背景/无动作。
新标注使用这六类基础标签；经用户确认，本批数据另保留“连续敲击”标签。
含义与易混对见
[`docs/COLLECTION-AND-ANNOTATION.md`](docs/COLLECTION-AND-ANNOTATION.md)。

---

## 从标注到训练

`pc/train/train_knock.py` 直接读 `takes/*/{take.mp4,annotation.json}`：knock 区间当正类，
其它已标注区间当负类，未标注帧忽略；按 take 留一做训练/评估。标签合并规则、命令、
当前冒烟结果和局限见 [`docs/TRAIN-PIPELINE.md`](docs/TRAIN-PIPELINE.md)。

---

## 一句话原则

> 这是**逐事件**的数据，不是“一类一段”的旧数据。
> 标得紧、标得一致，比标得多重要；含义不清的地方宁可留空，不要硬凑标签。
