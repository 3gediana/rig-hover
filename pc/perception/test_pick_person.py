# -*- coding: utf-8 -*-
"""test_pick_person.py -- 多人时选谁？答案：**最近的那个**。

用户 2026-11-03 明确："谁靠着近，你就看谁就好了"。

原来的实现按**检测框面积**排名。面积有个具体的失效场景：**最近的人常常贴着
画面边缘**，框被裁掉一截，面积反而小于稍远但完整的人——于是机器人会去看
那个更远的。这里把排名主键换成框高（`person_h` 本身就是距离的代理量，
且不受水平裁剪影响），并用测试把这个行为钉住。

不需要 GPU：`_pick_person` 只吃 boxes/keypoints 两个容器，这里用桩对象喂它。

Run:  python test_pick_person.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_PASS = 0
N_FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  PASS  {name}")
    else:
        N_FAIL += 1
        print(f"  FAIL  {name}   {detail}")


class _T:
    """最小 torch 张量桩：只实现 _pick_person 用到的那几个方法。"""

    def __init__(self, a):
        self._a = np.asarray(a, np.float32)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def reshape(self, *shape):
        return _T(self._a.reshape(*shape))

    def __getitem__(self, i):
        return _T(self._a[i])

    def __len__(self):
        return len(self._a)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _T(xyxy)
        self.conf = _T(conf)

    def __len__(self):
        return len(self.conf._a)


class _KP:
    def __init__(self, xy, conf):
        self.xy = _T(xy)
        self.conf = _T(conf)

    def __len__(self):
        return len(self.xy._a)


class _Res:
    def __init__(self, xyxy, conf, kp_xy=None, kp_conf=None):
        self.boxes = _Boxes(xyxy, conf)
        if kp_xy is None:
            self.keypoints = None
        else:
            self.keypoints = _KP(kp_xy, kp_conf)


META = {"r": 1.0, "dw": 0.0, "dh": 0.0, "nw": 320.0, "nh": 320.0}


def kp_block(n: int, marker: float):
    """造 n 组 17 关键点，用 marker 区分是第几个人。"""
    xy = np.zeros((n, 17, 2), np.float32)
    xy[:, :, 0] = marker
    xy[:, :, 1] = marker
    conf = np.full((n, 17), 0.9, np.float32)
    return xy, conf


def main() -> int:
    from run_perception import PerceptionPipeline

    # 不需要真的构造 pipeline（那要加载 YOLO），直接借用未绑定的方法
    pick = PerceptionPipeline._pick_person

    class _Self:
        pass

    me = _Self()

    print("== 两人，一近一远 ==")
    # 近：框高 200；远：框高 80
    xyxy = [[20, 20, 120, 220], [150, 60, 210, 140]]
    conf = [0.80, 0.95]
    kx, kc = kp_block(2, 0.0)
    kx[0, :, 0] = 111.0   # 标记第一个人
    kx[1, :, 0] = 222.0
    r = pick(me, _Res(xyxy, conf, kx, kc), META, 240.0, 240.0)
    check("选了框高的那个（更近）",
          r is not None and abs(r["kxy"][0, 0] - 111.0) < 1e-6,
          f"kxy0={None if r is None else r['kxy'][0, 0]}")
    check("n_det 报告 2 个人", r is not None and r["n_det"] == 2)
    check("person_h 约等于 200/240",
          r is not None and abs(r["h_norm"] - 200.0 / 240.0) < 1e-3,
          f"{None if r is None else r['h_norm']}")

    print("\n== 最近的人被画面边缘裁掉（面积法的失效场景）==")
    # 人 A：贴左边被裁，宽只有 30，但高 220（很近）
    # 人 B：完整，宽 120、高 150（稍远）
    #   面积：A = 6600，B = 18000 -> 面积法会选 B（错的）
    #   高度：A = 220 > B = 150 -> 框高法选 A（对的）
    xyxy = [[0, 10, 30, 230], [90, 40, 210, 190]]
    conf = [0.95, 0.95]
    kx, kc = kp_block(2, 0.0)
    kx[0, :, 0] = 111.0
    kx[1, :, 0] = 222.0
    r = pick(me, _Res(xyxy, conf, kx, kc), META, 240.0, 240.0)
    check("边缘被裁的近人仍然被选中",
          r is not None and abs(r["kxy"][0, 0] - 111.0) < 1e-6,
          f"kxy0={None if r is None else r['kxy'][0, 0]}")
    areaA = 30 * 220
    areaB = 120 * 150
    check("（对照）按面积会选错的确实是另一人", areaA < areaB,
          f"{areaA} < {areaB}")

    print("\n== 同高时比面积 ==")
    # 两人框高都 120，A 宽 40、B 宽 80 -> B 面积大
    xyxy = [[10, 10, 50, 130], [100, 10, 180, 130]]
    conf = [0.90, 0.90]
    kx, kc = kp_block(2, 0.0)
    kx[0, :, 0] = 111.0
    kx[1, :, 0] = 222.0
    r = pick(me, _Res(xyxy, conf, kx, kc), META, 240.0, 240.0)
    check("同高选面积大的", r is not None and abs(r["kxy"][0, 0] - 222.0) < 1e-6)

    print("\n== 同高同面积时比置信度 ==")
    xyxy = [[10, 10, 90, 130], [110, 10, 190, 130]]
    conf = [0.60, 0.99]
    kx, kc = kp_block(2, 0.0)
    kx[0, :, 0] = 111.0
    kx[1, :, 0] = 222.0
    r = pick(me, _Res(xyxy, conf, kx, kc), META, 240.0, 240.0)
    check("同高同面积选置信度高的",
          r is not None and abs(r["kxy"][0, 0] - 222.0) < 1e-6)

    print("\n== 边界情况 ==")
    check("没有人 -> None", pick(me, _Res([], []), META, 240.0, 240.0) is None)
    r = pick(me, _Res([[10, 10, 90, 130]], [0.8]), META, 240.0, 240.0)
    check("一个人也能正常返回", r is not None and r["n_det"] == 1)
    check("没有关键点时不崩",
          r is not None and r["kxy"].shape == (17, 2))

    print("\n" + "=" * 56)
    print(f"{N_PASS} PASS / {N_FAIL} FAIL")
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
