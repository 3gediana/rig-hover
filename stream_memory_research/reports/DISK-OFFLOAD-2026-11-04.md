# D: 盘大文件迁移记录（2026-11-04）

D: 是系统盘（可用空间紧张），G: 有 300 GB。本次把两类大文件搬到 G:，**D: 释放约 6.15 GB**。

## 搬走了什么

| 内容 | 原位置 | 新位置 | 体积 | 文件数 |
|---|---|---|---|---|
| 旧研究 447 个 run 的权重 | `D:\robot_work\stream_memory_research\runs\<run_id>\checkpoints\{best,last}.pt` | `G:\stream_memory_data\research_checkpoints\<run_id>\checkpoints\` | **4.28 GB** | 932 |
| 手势数据原始集（HaGRID 30k 384p） | `D:\robot_work\pc\gesture\data\raw\` | `G:\stream_memory_data\gesture_raw\` | **1.87 GB** | 31,854 |

`runs/` 现在只剩 0.05 GB（全是 json/log），`stream_memory_research/` 整体从 4.52 GB 降到 0.24 GB。

## 为什么安全

- **权重是可重建的**：`best_model/`（51 MB，13 个 run）本来就是报告引用的交付物副本，仍在 D: 上。
  报告和统计只读 `runs/<run_id>/metrics.json` / `curves.json` / `per_source.json`，这些**全部留在 D:**。
- **手势原始集没有任何代码引用**：`Select-String` 扫 `pc/gesture/*.py`、`configs/`、`*.md`
  对 `raw` 的引用为 0；其中的 `hagrid-sample-30k-384p.zip`(0.94 GB) 与已解压的 `hagrid/`(0.93 GB)
  本来就是重复内容。`pc/gesture` 的代码只读 `data/synth/`（仍在 D:）。

## 唯一需要改代码的地方

`scripts/eval_manifests.py` 会从 `runs/<rid>/checkpoints/best.pt` 重新加载模型做 manifest 评估。
已加回退：

```python
CKPT_MIRROR = Path(r"G:\stream_memory_data\research_checkpoints")

def best_ckpt(run_id: str) -> Path:
    """优先本地，其次 G: 镜像。"""
    local = ROOT / "runs" / run_id / "checkpoints" / "best.pt"
    return local if local.exists() else CKPT_MIRROR / run_id / "checkpoints" / "best.pt"
```

已验证：`runs/charades_event_feat__cfc__s0` 等 run 的 checkpoint 能正确解析到 G: 镜像。
**其他脚本不受影响**：`runs_pixel/*/best.pt`（当前 pixel-to-event 实验的权重，约 0.1 GB）**没有动**，
仍在 D: 上，因为它们是要反复加载做评测的活跃文件。

## 没有动的东西（以及原因）

| 目录 | 体积 | 为什么不动 |
|---|---|---|
| `D:\robot_work\rig_omni\build` | 0.36 GB | `idf.py build` 会重新生成，搬走等于强制重编 |
| `D:\robot_work\rig_omni\managed_components` | 0.69 GB | ESP-IDF 托管依赖，搬走要重新下载 |
| `D:\robot_work\pc\gesture\data\synth` | 0.79 GB | 手势训练代码按相对路径 `HERE/data/synth` 读取，搬走要改代码 |
| `D:\lnn_data` | 13.9 GB | 不属于本工程，未确认用途，不动 |
| `D:\smartcar` | 6.8 GB | Python venv（`site-packages` 已在 G:），动它会破坏运行环境 |

## 恢复办法

如需把某个 run 的权重搬回来：

```powershell
robocopy "G:\stream_memory_data\research_checkpoints\<run_id>" `
         "D:\robot_work\stream_memory_research\runs\<run_id>" /E
```

手势原始集同理（`G:\stream_memory_data\gesture_raw` → `pc\gesture\data\raw`）。

## 实测磁盘数字

| | 迁移前 | 迁移后 |
|---|---|---|
| D: 可用 | 95.0 GB | **101.1 GB** |
| G: 可用 | 301.2 GB | 294.9 GB |
