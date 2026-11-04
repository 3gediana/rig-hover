"""数据录制器（TECH-PLAN §8.5 / §9.3）。

逐帧收 ``(timestamp, obs_dict, action_dict, debug_dict)``，落两份：

* ``<session>.npz`` —— 数值，float32，给训练脚本直接 ``np.load``
* ``<session>.jsonl`` —— 每帧一行，人读 / 排查「为什么这一刻它这么动」

再加一个容易搞错、所以单独说明的接口：``Recorder.split_by_session()``，
**必须按 session 划分训练/验证集，不能按帧随机划分**（§9.3）。

用法::

    rec = Recorder("data/session_01")
    for ...:
        vpath = rec.add(t, obs_dict, action_dict, debug_dict)
        if vpath:            # 每 3 帧返回一次路径，调用方负责写 JPEG
            cv2.imwrite(vpath, frame)
    rec.close()

    splits = Recorder.split_by_session({"s01": 2000, "s02": 1800, "s03": 1500},
                                       val_frac=0.2, test_frac=0.2, seed=0)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_NUMERIC = (int, float, np.integer, np.floating, bool)
_ARRAY = (np.ndarray, list, tuple)


def _is_array(v: Any) -> bool:
    """判断是不是定长数值数组（特征向量那类）。"""
    if isinstance(v, np.ndarray):
        return v.ndim >= 1
    if isinstance(v, (list, tuple)):
        return len(v) > 0 and all(isinstance(x, _NUMERIC) for x in v)
    return False


def _jsonable(v: Any, max_items: int = 8) -> Any:
    """把值转成 JSON 能写的类型；长数组只留摘要，别把 jsonl 撑爆。

    920 维特征逐元素写进 jsonl 会让日志比 npz 还大（108k 帧 ≈ 900MB），
    所以超过 ``max_items`` 的数组在 jsonl 里写成 ``"<array shape=(920,)>"``，
    真值只看 npz。短数组（如 4 维 one-hot）照常展开，排查时还是要能直接看。
    """
    if isinstance(v, np.ndarray):
        if v.size > max_items:
            return f"<array shape={v.shape} dtype={v.dtype}>"
        return v.tolist()
    if isinstance(v, (list, tuple)) and len(v) > max_items:
        return f"<array shape=({len(v)},)>"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Recorder:
    """单 session 录制器。

    Args:
        out_dir: 输出目录（不存在会自动建）。
        session_id: 本段录制的 session 标识，决定文件名，也是划分数据集的最小单位。
        video_every: 每几帧留一个视频帧引用；0 表示不留。
        meta: 自由附加信息（帧率、摄像头位置、参数快照等），写进 npz 的 ``meta_json``。
        flush_every: 每几帧 flush 一次 jsonl；1 = 每帧 flush（掉电最多丢一帧）。

    数据结构上帧是 append 的，``close()`` 时才落 npz；所以内存占用是
    O(帧数 × 单帧维度)。108k 帧 × 984 浮点的量级约 430MB，v1 能接受，
    真要采满 60 分钟建议按 session 切分并改为 memmap 追加写。

    两个对齐约定：

    * 某个字段只在部分帧出现（例如只在 ACK 期间打的 debug 键）时，缺的帧在
      npz 里补 ``NaN``（字符串补空串），**不会把后面的帧错位**。
    * 超过 8 个元素的数组（920 维特征）在 jsonl 里只写 ``"<array shape=(920,)>"``，
      真值只看 npz —— 否则 jsonl 会比 npz 还大。
    """

    def __init__(
        self,
        out_dir: str | Path,
        session_id: str = "session_000",
        video_every: int = 3,
        meta: Mapping[str, Any] | None = None,
        flush_every: int = 1,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.video_every = int(video_every)
        self.meta = dict(meta or {})
        self.flush_every = max(1, int(flush_every))

        self.npz_path = self.out_dir / f"{self.session_id}.npz"
        self.jsonl_path = self.out_dir / f"{self.session_id}.jsonl"

        # 列按「帧号 → 值」存，而不是 append 列表：某个字段中途才出现
        # （比如 ACK 期间才有的 debug 键）时，缺的帧补 NaN/空串，绝不把后面的帧错位。
        self._scalars: dict[str, dict[int, float]] = {}
        self._vectors: dict[str, dict[int, np.ndarray]] = {}
        self._strings: dict[str, dict[int, str]] = {}
        self._video: list[str] = []
        self._n = 0
        self._closed = False
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8")

    # ------------------------------------------------------------------ 属性
    def __len__(self) -> int:
        return self._n

    def video_path(self, index: int) -> str | None:
        """第 index 帧如果要存 JPEG，返回相对路径；否则 None。"""
        if self.video_every <= 0 or index % self.video_every != 0:
            return None
        return f"frames/{self.session_id}/{index:06d}.jpg"

    # ------------------------------------------------------------------ 写入
    def add(
        self,
        timestamp: float,
        obs: Mapping[str, Any],
        action: Mapping[str, Any],
        debug: Mapping[str, Any] | None = None,
    ) -> str | None:
        """写一帧。

        Args:
            timestamp: 单调时间戳，秒。
            obs: 观测字典（``TeacherObs`` 的扁平版）。
            action: 动作字典（``TeacherAction`` 的扁平版）。
            debug: 教师控制器中间量（可选）。

        Returns:
            需要保存视频帧时返回 JPEG 路径，否则 None。

        Raises:
            RuntimeError: recorder 已关闭。
        """
        if self._closed:
            raise RuntimeError("Recorder 已关闭，不能再 add")

        index = self._n
        debug = dict(debug or {})
        flat: dict[str, Any] = {"t": float(timestamp)}
        for prefix, mapping in (("obs", obs), ("act", action), ("dbg", debug)):
            for key, val in mapping.items():
                flat[f"{prefix}_{key}"] = val

        vpath = self.video_path(index)
        self._video.append(vpath or "")

        for key, val in flat.items():
            if _is_array(val):
                self._vectors.setdefault(key, {})[index] = np.asarray(val, dtype=np.float32).ravel().copy()
            elif isinstance(val, _NUMERIC):
                self._scalars.setdefault(key, {})[index] = float(val)
            elif isinstance(val, str):
                self._strings.setdefault(key, {})[index] = val
            elif val is None:
                self._scalars.setdefault(key, {})[index] = math.nan
            else:  # 嵌套 dict 之类，只进 jsonl
                self._strings.setdefault(key, {})[index] = json.dumps(
                    val, ensure_ascii=False, default=_jsonable
                )

        line = {
            "i": index,
            "t": float(timestamp),
            "session": self.session_id,
            "video": vpath or "",
            "obs": {k: _jsonable(v) for k, v in obs.items()},
            "action": {k: _jsonable(v) for k, v in action.items()},
            "debug": {k: _jsonable(v) for k, v in debug.items()},
        }
        self._jsonl.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._jsonl.flush()
        return vpath

    # ------------------------------------------------------------------ 收尾
    def close(self) -> dict:
        """写完 npz 并关闭 jsonl。

        Returns:
            ``{"npz": path, "jsonl": path, "n_frames": n, "session_id": sid}``
        """
        if self._closed:
            return {
                "npz": str(self.npz_path),
                "jsonl": str(self.jsonl_path),
                "n_frames": self._n,
                "session_id": self.session_id,
            }
        self._closed = True
        self._jsonl.flush()
        self._jsonl.close()

        data: dict[str, np.ndarray] = {}
        n = self._n
        for key, col in self._scalars.items():
            arr = np.full(n, np.nan, dtype=np.float32)
            for i, v in col.items():
                arr[i] = v
            data[key] = arr
        for key, col in self._vectors.items():
            shapes = {v.shape for v in col.values()}
            if len(shapes) != 1:
                raise ValueError(f"字段 '{key}' 的数组形状不一致：{sorted(shapes)}")
            dim = next(iter(shapes))[0]
            arr = np.full((n, dim), np.nan, dtype=np.float32)
            for i, v in col.items():
                arr[i] = v
            data[key] = arr
        for key, col in self._strings.items():
            width = max((len(s) for s in col.values()), default=1)
            arr = np.full(n, "", dtype=f"<U{max(1, width)}")
            for i, v in col.items():
                arr[i] = v
            data[key] = arr
        data["video_frame"] = np.asarray(self._video, dtype="<U{0}".format(
            max(1, max((len(s) for s in self._video), default=1))))
        data["session_id"] = np.asarray([self.session_id] * self._n, dtype="<U{0}".format(
            max(1, len(self.session_id))))
        data["n_frames"] = np.asarray([self._n], dtype=np.int64)
        data["meta_json"] = np.asarray([json.dumps(self.meta, ensure_ascii=False, default=_jsonable)])
        np.savez_compressed(self.npz_path, **data)

        return {
            "npz": str(self.npz_path),
            "jsonl": str(self.jsonl_path),
            "n_frames": self._n,
            "session_id": self.session_id,
        }

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def load_npz(path: str | Path) -> dict[str, np.ndarray]:
        """读回一个 session 的 npz。"""
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    # ------------------------------------------------------- ★ 数据划分
    @staticmethod
    def split_by_session(
        session_ids: Mapping[str, int] | Sequence[str] | np.ndarray,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        seed: int = 0,
    ) -> dict:
        """**按 session 划分**训练/验证/测试索引（TECH-PLAN §9.3）。

        为什么不能按帧随机划分：同一 session 内相邻帧相隔 33ms，几乎相同。
        随机划分会让 val 的帧在 train 里都有 33ms 之前的"孪生帧"，
        验证 loss 虚低、真机一跑就崩。**划分的最小单位必须是 session。**

        本函数只做"整段 session 分给某一侧"，保证同一个 session 的帧
        不会同时出现在两个集合里（``_assert_disjoint`` 会自检）。

        Args:
            session_ids: 两种输入都行 —
                ``{"s01": 2000, "s02": 1800}``（session → 帧数，帧索引按顺序累加），
                或者逐帧的 session 标识序列（``npz["session_id"]``）。
            val_frac: 验证集参与的 session 比例。
            test_frac: 测试集参与的 session 比例。
            seed: session 排序的随机种子（同一份数据可复现）。

        Returns:
            ``{"train": int64 数组, "val": ..., "test": ...,
               "sessions": {"train": [sid...], ...}}``

        Raises:
            ValueError: session 数不够划分，或同一 session 跨集合（内部自检失败）。
        """
        per_session = Recorder._group_indices(session_ids)
        sids = list(per_session)
        n = len(sids)
        if n == 0:
            raise ValueError("没有任何 session，无法划分")

        need = (1 if val_frac > 0 else 0) + (1 if test_frac > 0 else 0)
        if n < need + 1:
            raise ValueError(
                f"只有 {n} 个 session，不够划分 train/val/test"
                f"（val_frac={val_frac}, test_frac={test_frac}）。"
                "§9.3 要求按 session 划分，请多采几段。"
            )

        order = np.arange(n)
        if seed is not None:
            np.random.default_rng(seed).shuffle(order)
        sids = [sids[i] for i in order]

        n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
        # 至少给 train 留一个 session
        while n_test + n_val >= n and (n_test + n_val) > 0:
            if n_test > 0 and (n_val == 0 or n_test >= n_val):
                n_test -= 1
            else:
                n_val -= 1

        groups = {
            "train": sids[: n - n_val - n_test],
            "val": sids[n - n_val - n_test : n - n_test],
            "test": sids[n - n_test :] if n_test else [],
        }
        splits = {
            "train": np.concatenate([per_session[s] for s in groups["train"]]) if groups["train"] else np.empty(0, np.int64),
            "val": np.concatenate([per_session[s] for s in groups["val"]]) if groups["val"] else np.empty(0, np.int64),
            "test": np.concatenate([per_session[s] for s in groups["test"]]) if groups["test"] else np.empty(0, np.int64),
            "sessions": groups,
        }
        Recorder._assert_disjoint(splits, session_ids)
        return splits

    @staticmethod
    def _group_indices(session_ids: Mapping[str, int] | Sequence[str] | np.ndarray) -> dict[str, np.ndarray]:
        """session → 该 session 全部帧的全局索引（保持首次出现顺序）。"""
        per: dict[str, list[int]] = {}
        if isinstance(session_ids, Mapping):
            nxt = 0
            for sid, count in session_ids.items():
                cnt = int(count)
                if cnt < 0:
                    raise ValueError(f"session '{sid}' 的帧数是负数：{cnt}")
                per.setdefault(str(sid), []).extend(range(nxt, nxt + cnt))
                nxt += cnt
        else:
            for i, sid in enumerate(session_ids):
                per.setdefault(str(sid), []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in per.items()}

    @staticmethod
    def _assert_disjoint(splits: Mapping[str, Any], session_ids: Any) -> None:
        """自检：同一个 session 的帧绝不能同时落在两个集合里。"""
        per = Recorder._group_indices(session_ids)
        owner: dict[str, str] = {}
        for name in ("train", "val", "test"):
            idx = set(int(i) for i in splits[name])
            for sid, frames in per.items():
                hit = len(idx.intersection(frames.tolist()))
                if hit and hit != len(frames):
                    raise ValueError(
                        f"session '{sid}' 被切开了：{hit}/{len(frames)} 帧进了 {name}。"
                        "§9.3 禁止按帧划分。"
                    )
                if hit:
                    prev = owner.get(sid)
                    if prev and prev != name:
                        raise ValueError(f"session '{sid}' 同时出现在 {prev} 和 {name}")
                    owner[sid] = name
