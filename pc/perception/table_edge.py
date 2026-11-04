# -*- coding: utf-8 -*-
"""table_edge.py -- 桌沿检测：把画面里的桌面边界换算成"离桌沿还有多少米"。

## 为什么这件事恰好和"走向敲击点"是同一个时刻

板子是 **1-servo + 2-wheel hovercraft**（`board_config.h:5`）。单个舵机只做偏航，
**头不能低头**。但它是自平衡车 —— 要往前加速就必须**整车前倾**，整台车连同
摄像头一起低头，于是前下方的桌面进入视野。

也就是说：**机器人唯一会朝桌子前进的时刻，就是它唯一能看清桌沿的时刻。**
用户 2026-11-03 提的正是这一点（"往拳头标定点走的时候摄像机肯定是低下去看路的"）。

## 为什么不用新模型

俯仰角已经在观测里（`obs[79]`），相机高度和焦距是一次性标定的常数。
计算机视觉上这是个**地面单应**问题：知道相机朝向和高度，图像上的行号就唯一
对应桌面上的一段距离。

    光轴以下 α 角的那条光线：  α = atan((v - v_c) / f)
    这条光线相对水平面的夹角：  θ = pitch + α          （pitch 为车体前倾角）
    落到桌面的距离：            d = h / tan(θ)

所以"桌沿在哪一行"直接等价于"桌沿在多少米外"，**不需要立体视觉，也不需要学习**。

## 检测什么

桌面上、以及越过桌沿之后的区域，在灰度图里的统计特性不同（纹理能量、逐行亮度
水平、边缘密度）。逐行算一个"这还是桌面吗"的分数，从下往上找分数突降的那一行，
就是桌沿。

⚠️ **真机上必须先标定**（`cam_h_m` / `f_px` / `v_center`），
并且用真实桌面验证阈值。本模块的算法结构可单测，但参数不能照抄。

Run（自检）:  python table_edge.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["EdgeParams", "TableEdge", "TableEdgeDetector",
           "row_to_distance_m"]


def row_to_distance_m(v: float, pitch_deg: float, cam_h_m: float,
                      f_px: float, v_center: float) -> float:
    """图像行号 → 该行对应桌面点离机器人的水平距离（米）。

    Args:
        v: 图像行（0 在顶部，向下增大）
        pitch_deg: 车体前倾角（**正 = 低头看向桌面**），度
        cam_h_m: 相机光心离桌面的高度，米
        f_px: 以像素为单位的焦距
        v_center: 光心所在行（通常 = 图像高度/2）

    Returns:
        距离（米）。光线朝上或几乎平行时返回 ``inf``。
    """
    if cam_h_m <= 0 or f_px <= 0:
        return float("inf")
    alpha = math.degrees(math.atan((float(v) - float(v_center)) / float(f_px)))
    theta = float(pitch_deg) + alpha          # 相对水平面往下的夹角
    if theta <= 0.5:                           # 光线朝上，永远打不到桌面
        return float("inf")
    t = math.tan(math.radians(theta))
    if t <= 1e-6:
        return float("inf")
    return float(cam_h_m / t)


@dataclass
class EdgeParams:
    """检测与标定参数。**默认值全部是需要真机标定的先验，不能照抄。**"""
    #: 相机光心离桌面高度（米）
    cam_h_m: float = 0.20
    #: 焦距（像素）。240x240、水平 FOV 60° 时 f ≈ (240/2)/tan(30°) ≈ 208
    f_px: float = 208.0
    #: 光心所在行，0 起
    v_center: float = 120.0
    #: 只看这一行以下（太靠上的是背景，不是桌面）
    v_min: int = 110
    #: 逐行分数用的窗口半宽（行）
    streak: int = 5
    #: 分数从"桌面水平"掉到它的多少倍以下就算越过桌沿
    drop_ratio: float = 0.45
    #: 分数绝对下限，低于它一律算非桌面
    score_floor: float = 2.0
    #: 至少连续多少行被判为非桌面才认账（抗单行噪声）
    min_rows: int = 4
    #: 认为桌沿检测有效的最近/最远距离
    d_min_m: float = 0.05
    d_max_m: float = 2.50
    #: 安全检查用的停车余量（米）
    stop_margin_m: float = 0.12


@dataclass
class TableEdge:
    """一帧的桌沿检测结果。"""
    valid: int = 0
    #: 桌沿所在行；-1 表示画面内没找到
    row: int = -1
    #: 桌沿距离（米）；inf 表示不可用
    dist_m: float = float("inf")
    #: 置信度 0–1
    conf: float = 0.0
    #: 逐行分数（调试用，可不下发）
    scores: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"valid": self.valid, "row": self.row,
                "dist_m": (None if math.isinf(self.dist_m)
                           else round(self.dist_m, 3)),
                "conf": round(self.conf, 3)}


class TableEdgeDetector:
    """逐行"这还是桌面吗"打分，从下往上找突降点。"""

    def __init__(self, params: EdgeParams | None = None):
        self.p = params or EdgeParams()

    # ------------------------------------------------------------------
    def row_scores(self, gray: np.ndarray) -> np.ndarray:
        """每一行的"桌面可能性"分数。

        做法：对每一行取一条横向切片，算它的**梯度能量**（纹理丰富度）。
        桌面通常有稳定的纹理；越过桌沿之后，要么是远处的背景（高频细节不同），
        要么是下方的地面（亮度跳变）。单行噪声较大，所以用 ``streak`` 行的
        滑动窗口平滑。
        """
        g = np.asarray(gray)
        if g.ndim == 3:
            g = g.mean(axis=2)
        g = g.astype(np.float32)
        h, w = g.shape[:2]
        gy, gx = np.gradient(g)
        energy = np.sqrt(gx * gx + gy * gy).mean(axis=1) * 255.0
        if self.p.streak > 1:
            k = int(self.p.streak)
            ker = np.ones(k, np.float32) / k
            energy = np.convolve(energy, ker, mode="same")
        return energy[:h]

    def detect(self, gray: np.ndarray, pitch_deg: float) -> TableEdge:
        """检测桌沿。

        Args:
            gray: HxW 或 HxWx3 的 uint8 图
            pitch_deg: 车体前倾角（度，正 = 低头）

        Returns:
            TableEdge
        """
        p = self.p
        g = np.asarray(gray)
        h = g.shape[0]
        scores = self.row_scores(g)

        v0 = max(int(p.v_min), 1)
        if v0 >= h - 1:
            return TableEdge(scores=[float(v) for v in scores])

        # 参考水平取"这片区域里较高的那部分"，**不能取最下面一行**：
        # 桌沿出现在画面里时，最下面一行恰恰在桌沿之外，用它当参考会把
        # 阈值压到地板水平，于是永远找不到边界（这是第一版的 bug）。
        reg = scores[v0:h]
        ref = float(np.percentile(reg, 90))
        thresh = max(ref * p.drop_ratio, p.score_floor)

        out = TableEdge(scores=[round(float(v), 3) for v in scores])
        # 从下往上数"连续多少行已经不像桌面了"；数够 min_rows，边界就在
        # 最后一行"还像桌面"的下一行。
        run = 0
        edge_row = -1
        for v in range(h - 1, v0 - 1, -1):
            if scores[v] >= thresh:
                if run >= p.min_rows:
                    edge_row = v + 1
                    break
                run = 0
            else:
                run += 1
        if edge_row < 0 and run >= p.min_rows:
            # 整段都像非桌面：边界在 v0
            edge_row = v0

        if edge_row < 0:
            # 画面里没有桌沿 —— 这本身是好消息：前方一段都是桌面
            out.valid = 1
            out.row = -1
            out.dist_m = float("inf")
            out.conf = 0.5
            return out

        d = row_to_distance_m(edge_row, pitch_deg, p.cam_h_m, p.f_px,
                              p.v_center)
        if not (p.d_min_m <= d <= p.d_max_m):
            return out   # invalid：距离不合理，多半是误检

        out.valid = 1
        out.row = int(edge_row)
        out.dist_m = float(d)
        # 置信度：参考水平越明确、突降越干脆，越可信
        out.conf = float(min(1.0, max(0.0, 1.0 - thresh / max(ref, 1e-6))))
        return out

    def safe_to_advance(self, edge: TableEdge, vx_cmd: float) -> tuple[float, str]:
        """把桌沿约束作用到前进指令上。

        Returns:
            (允许的 vx, 原因)。桌沿太近时返回 0。
        """
        if edge.valid and math.isfinite(edge.dist_m):
            if edge.dist_m <= self.p.stop_margin_m:
                return 0.0, f"edge {edge.dist_m:.2f}m <= margin"
            return vx_cmd, "ok"
        # 检测无效：保守处理，但不阻断（完全阻断会让机器人一动不能动）
        return vx_cmd, "no_edge_seen"


# ---------------------------------------------------------------------------
# 自检：用合成图验证算法结构与几何换算
# ---------------------------------------------------------------------------
def _self_test() -> int:
    n_pass = n_fail = 0

    def chk(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"  PASS  {name}")
        else:
            n_fail += 1
            print(f"  FAIL  {name}   {detail}")

    print("== 几何换算 ==")
    P = EdgeParams()
    # 行号越往下（越大），看得越近
    d_top = row_to_distance_m(140, 0.0, 0.2, 208.0, 120.0)
    d_mid = row_to_distance_m(180, 0.0, 0.2, 208.0, 120.0)
    d_bot = row_to_distance_m(230, 0.0, 0.2, 208.0, 120.0)
    chk("行号越大距离越近", d_top > d_mid > d_bot,
        f"{d_top:.2f} {d_mid:.2f} {d_bot:.2f}")
    # 前倾（低头）会让同一行对应更近的距离
    d_p0 = row_to_distance_m(180, 0.0, 0.2, 208.0, 120.0)
    d_p10 = row_to_distance_m(180, 10.0, 0.2, 208.0, 120.0)
    chk("低头后同一行看得更近", d_p10 < d_p0, f"{d_p0:.2f} -> {d_p10:.2f}")
    chk("光轴以上打不到桌面",
          math.isinf(row_to_distance_m(0, 0.0, 0.2, 208.0, 120.0)))

    print("\n== 合成图：上桌面 / 下悬空 ==")

    def make_frame(edge_row, h=240, w=240, seed=0):
        rng = np.random.default_rng(seed)
        img = np.zeros((h, w), np.uint8)
        # 桌面：中等亮度 + 稳定纹理
        table = np.full((edge_row, w), 140, np.float32)
        table += rng.normal(0, 6, table.shape)
        # 桌面上的细纹理（模拟木纹/网格），保证梯度能量不低
        for y in range(0, edge_row, 7):
            table[y:y + 1, :] += 22
        img[:edge_row] = np.clip(table, 0, 255).astype(np.uint8)
        # 桌沿以下：地面，明显更暗、更平
        floor = np.full((h - edge_row, w), 55, np.float32)
        floor += rng.normal(0, 3, floor.shape)
        img[edge_row:] = np.clip(floor, 0, 255).astype(np.uint8)
        return img

    det = TableEdgeDetector(EdgeParams(cam_h_m=0.2, f_px=208.0, v_center=120.0))
    for edge_row, pitch in ((200, 14.0), (170, 10.0), (220, 20.0)):
        img = make_frame(edge_row)
        e = det.detect(img, pitch)
        ok = e.valid and abs(e.row - edge_row) <= 6
        chk(f"edge_row={edge_row} 检出 row={e.row} (容差6行)", ok,
            f"got row={e.row} valid={e.valid} d={e.dist_m}")

    print("\n== 纯桌面（画面里没有桌沿）==")
    flat = make_frame(240)
    e = det.detect(flat, 5.0)
    chk("没有桌沿时 valid=1 且 distance=inf",
          e.valid == 1 and math.isinf(e.dist_m), f"{e.as_dict()}")

    print("\n== 桌面有杂物时会不会误判（经典 CV 的真实弱点）==")

    def make_cluttered(edge_row, h=240, w=240, seed=0):
        """桌面区域里放几个高对比的物体（书/杯子），制造纹理突变。"""
        img = make_frame(edge_row, h, w, seed)
        rng = np.random.default_rng(seed + 7)
        for _ in range(5):
            y = int(rng.integers(max(120, edge_row - 90), max(121, edge_row - 8)))
            x = int(rng.integers(0, w - 40))
            bw, bh = int(rng.integers(18, 45)), int(rng.integers(8, 22))
            img[y:y + bh, x:x + bw] = 235   # 亮色的书/杯子
        return img

    for edge_row, pitch in ((200, 14.0), (220, 20.0)):
        img = make_cluttered(edge_row)
        e = det.detect(img, pitch)
        got = e.row if e.valid else -1
        ok = e.valid and e.row >= 0 and abs(e.row - edge_row) <= 12
        chk(f"杂物场景 edge_row={edge_row} -> row={got}", ok,
            f"got={got} valid={e.valid} conf={e.conf:.2f}")
        if not ok:
            print(f"        ↑ 经典 CV 在杂物下失效：纹理能量被物体边缘污染")

    print("\n== 安全约束 ==")
    near = TableEdge(valid=1, row=200, dist_m=0.10, conf=0.9)
    vx, why = det.safe_to_advance(near, 0.15)
    chk("桌沿 0.10m < 余量 -> vx 归零", vx == 0.0, why)
    far = TableEdge(valid=1, row=160, dist_m=1.20, conf=0.9)
    vx2, why2 = det.safe_to_advance(far, 0.15)
    chk("桌沿 1.20m -> 放行", vx2 == 0.15, why2)
    none = TableEdge(valid=0)
    vx3, why3 = det.safe_to_advance(none, 0.15)
    chk("检测无效时不阻断（否则动不了）", vx3 == 0.15, why3)

    print("\n== 参数合理性 ==")
    # 240x240, 60° 水平 FOV -> f = 120/tan(30°)
    f_expected = 120.0 / math.tan(math.radians(30.0))
    chk(f"f_px 默认值与 60° FOV 自洽（{f_expected:.0f}）",
          abs(EdgeParams().f_px - f_expected) < 2.0,
          f"default={EdgeParams().f_px} expect={f_expected:.1f}")

    print("\n" + "=" * 56)
    print(f"{n_pass} PASS / {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
