# `status: complete` 不等于"这次配置跑出来的结果"

**适用**：任何"按 run_id 跳过已完成实验"的 sweep/实验编排器。

## 现场

实验驱动脚本 `scripts/run_suite.py` 用 `is_done(run_id)` 判断是否跳过：

```python
m = read_json(runs/<run_id>/metrics.json)
return m.get("status") == "complete"
```

改过一次 suite 配置（`patience` 8→40、`epochs` 40→50、`batch` 128→64）后重跑，
**旧配置的 run 目录仍在磁盘上、`status` 仍是 `complete`**，于是被当成新配置的结果跳过。

后果：`synth_recall_single` 上 `cfc` 的三个种子实际是
**旧 pilot（9 epoch 早停，0.182）+ 两个新 run（0.483 / 0.368）**，
3-seed 均值 0.251 被写进对比表；而真实值应在 0.45 附近。这一条污染过一次排名。

## 陷阱

run_id 只标识"哪个模型/数据集/种子"，**不标识"哪一版配置"**。配置一改，
同一个 run_id 就对应两个不同的实验，但磁盘上只有一个目录。

## 验证后的做法

`is_done` 必须**比对配置指纹**：

```python
KEYS = ("dataset", "manifest", "task", "model", "heads", "loss", "train", "seed")

def config_matches(cfg):
    old = read_json(runs / cfg["run_id"] / "config.json")
    return json.dumps({k: old.get(k) for k in KEYS}, sort_keys=True, default=str) == \
           json.dumps({k: cfg.get(k) for k in KEYS}, sort_keys=True, default=str)

def is_done(cfg):
    m = read_json(runs / cfg["run_id"] / "metrics.json")
    if m.get("status") != "complete":
        return False
    if not config_matches(cfg):
        print(f"[stale] {cfg['run_id']}: config differs -> will re-run")
        return False
    return True
```

改完后立刻扫出 3 个 stale run（`synth_recall_single__{mlp__s2, gru__s0, cfc__s0}`），
删掉重跑后数字才正确。同理，**smoke 结果必须写进 `__smoke` 后缀的目录**，
否则也会被当成正式结果复用。

## 边界

- 只比对**会影响结果**的键；`notes`、时间戳、绝对路径这类噪声键不要进指纹。
- 若同一 run_id 想保留多个版本，应把版本号写进 run_id，而不是依赖配置比对。

## 出处

`D:\robot_work\stream_memory_research\scripts\run_suite.py`（`_comparable` / `config_matches` /
`is_done`），`reports/中文总结报告.md` §5.1 的排错记录。
