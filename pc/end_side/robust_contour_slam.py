# -*- coding: utf-8 -*-
"""robust_contour_slam.py — 开源 SLAM 工业范式驱动的纯绿色轮廓 3D 重建、静态场景精修与抗扰动引擎。

直接落地多套成熟开源 3D 建图/SLAM 项目的核心数学机制（零重型二进制依赖，算法 100% 可平移到 Android Kotlin）：
1. **固定场景精修与漂移修正** — 源自 **KISS-ICP** (`PRBonn/kiss-icp`) + **ElasticFusion** (`mp3guy/ElasticFusion`)
   + **DenseSurfelMapping** (`HKUST-Aerial-Robotics/DenseSurfelMapping`)：
   - 仅利用已确认静态面元 (Confirmed Static Surfels) 建立局部地图，执行自适应阈值 (Adaptive Threshold)
     + Geman-McClure 鲁棒核加权 Point-to-Plane / SE(3) 混合高斯-牛顿配准，自动修正双轮自平衡俯仰摇摆
     (`±6°`)、舵机回程差 (`±10°`) 与图-遥测时间偏移；
   - 面元稳定度生命周期 (Tentative -> Confirmed)、斜视角入射权重 `w_angle = max(0.15, |n·v|)`、
     逆方差位置/法向融合与桌面 `Z=0` 平面平滑吸附。
2. **动态物体分离与抗扰动** — 源自 **Dynablox** (`ethz-asl/dynablox`) + **NVIDIA nvblox** (`nvidia-isaac/nvblox`)
   + **UFOMap** (`UnknownFreeOccupied/ufomap`) + **Unblur-SLAM** (`SlamMate/Unblur-SLAM`)：
   - **Ever-Free 3D 体素网格 (Dynablox)**：记录曾被干净视线多次穿透的"历史空域"；人物/手一旦进入
     历史空域，即便短暂静止也不会被烤进静态地图，也不污染 ICP 配准，而是直通实时动态层；
   - **双层地图架构 (nvblox)**：`StaticMap`（永久精修静态骨架）+ `DynamicLayer`（1~2 帧快速衰减实时动态层）；
   - **扰动质量门控 (Unblur-SLAM)**：联合评估拉普拉斯清晰度比、IMU/位姿角冲击率与 ICP 内点率；
     遇到机身撞击/急停模糊帧时冻结静态地图更新（保护骨架绝不糊化或歪斜），同时保持实时动态层流畅显示。
3. **纯绿色雷达轮廓渲染 (无 RGB 彩色)** — 源自 **Potree / CloudCompare Eye-Dome Lighting (EDL)**：
   - 彻底抛弃相机暗淡灰度/RGB 反照率，完全由 3D 几何驱动纯绿色轮廓：
     高度分层暗绿桌面 + 法向朗伯明暗 + Fresnel 掠射轮廓高光 `(1 - |n·v|)^2`
     + 双半径 EDL 深度边界暗描边 + 亮绿前沿轮廓提取。
   - 深色衣服、黑色键盘、暗桌垫与亮色物体获得完全一致的清晰绿色立体轮廓。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import pathlib
import sys
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dense_scene_pipeline import (  # noqa: E402
    TraversabilityGrid2D,
    reconstruct_dense_frame_3d,
    render_minimap_2d,
    run_student_tflite_frame,
    solve_table_plane_from_student,
)
from mono_depth_model import DEFAULT_H_CAM_M  # noqa: E402
from neural_gaussian_splat_model import OrbitCamera3D, PersistentGaussianMemoryMap  # noqa: E402

_PROJ = PersistentGaussianMemoryMap(max_splats=8)


# ---------------------------------------------------------------------------
# 1. 扰动质量门控 (Unblur-SLAM / MBA-SLAM 关键帧拦截机制)
# ---------------------------------------------------------------------------


@dataclass
class DisturbanceStatus:
    """单帧扰动与清晰度诊断报告。"""

    is_disturbed: bool
    reason: str
    laplacian_var: float
    blur_ratio: float
    dpitch_deg: float
    droll_deg: float
    dyaw_deg: float
    icp_inlier_ratio: float = 1.0
    icp_rot_corr_deg: float = 0.0
    icp_trans_corr_m: float = 0.0
    icp_median_res_m: float = 0.0


class DisturbanceQualityGate:
    """评估运动模糊、机身平衡角冲击与 ICP 跟踪失稳，阻止坏帧污染静态地图。"""

    def __init__(
        self,
        blur_ratio_thresh: float = 0.42,
        min_abs_lap_var: float = 10.0,
        max_dpitch_deg: float = 7.5,
        max_droll_deg: float = 6.5,
        max_dyaw_deg: float = 12.0,
        min_icp_inlier_ratio: float = 0.34,
    ) -> None:
        self.blur_ratio_thresh = float(blur_ratio_thresh)
        self.min_abs_lap_var = float(min_abs_lap_var)
        self.max_dpitch_deg = float(max_dpitch_deg)
        self.max_droll_deg = float(max_droll_deg)
        self.max_dyaw_deg = float(max_dyaw_deg)
        self.min_icp_inlier_ratio = float(min_icp_inlier_ratio)

        self._sharp_hist: list[float] = []
        self._prev_pose: tuple[float, float, float] | None = None
        self._consec_shock = 0

    def reset(self) -> None:
        self._sharp_hist.clear()
        self._prev_pose = None
        self._consec_shock = 0

    @staticmethod
    def compute_sharpness(bgr: np.ndarray) -> float:
        """拉普拉斯方差清晰度评分（越大越清晰，运动模糊时骤降）。"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        if gray.shape[0] > 160:
            gray = cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return float(lap.var())

    def evaluate_pre_icp(
        self,
        bgr: np.ndarray,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> DisturbanceStatus:
        """帧级前置扰动检测（模糊 + IMU/舵机瞬态角跳变）。"""
        lap_var = self.compute_sharpness(bgr)
        if len(self._sharp_hist) >= 3:
            ref_sharp = float(np.median(self._sharp_hist[-24:]))
            blur_ratio = lap_var / max(ref_sharp, 1e-3)
        else:
            blur_ratio = 1.0

        if self._prev_pose is None:
            dp, dr, dy = 0.0, 0.0, 0.0
        else:
            dp = abs(float(pitch_deg) - self._prev_pose[0])
            dr = abs(float(roll_deg) - self._prev_pose[1])
            dy_raw = abs(float(yaw_deg) - self._prev_pose[2]) % 360.0
            dy = min(dy_raw, 360.0 - dy_raw)

        reasons: list[str] = []
        if lap_var < self.min_abs_lap_var or (len(self._sharp_hist) >= 3 and blur_ratio < self.blur_ratio_thresh):
            reasons.append("motion_blur")
        if dp > self.max_dpitch_deg:
            reasons.append("pitch_shock")
        if dr > self.max_droll_deg:
            reasons.append("roll_shock")
        if dy > self.max_dyaw_deg:
            reasons.append("yaw_shock")

        if not reasons:
            self._prev_pose = (float(pitch_deg), float(roll_deg), float(yaw_deg))
            self._consec_shock = 0
            self._sharp_hist.append(lap_var)
            if len(self._sharp_hist) > 64:
                self._sharp_hist.pop(0)
        else:
            # 坏帧不污染参考位姿，避免冲击结束后恢复正常的首帧被误杀；
            # 若连续 >3 帧都处于新姿态（如机器人真的完成了大角度转向），再重置参考防止死锁。
            self._consec_shock += 1
            if self._consec_shock >= 4:
                self._prev_pose = (float(pitch_deg), float(roll_deg), float(yaw_deg))
                self._consec_shock = 0

        return DisturbanceStatus(
            is_disturbed=bool(reasons),
            reason="+".join(reasons) if reasons else "clean",
            laplacian_var=lap_var,
            blur_ratio=blur_ratio,
            dpitch_deg=dp,
            droll_deg=dr,
            dyaw_deg=dy,
        )


# ---------------------------------------------------------------------------
# 2. Dynablox / UFOMap Ever-Free 3D 空域体素网格
# ---------------------------------------------------------------------------


class EverFreeVoxelGrid:
    """Dynablox (`ethz-asl/dynablox`) 风格 3D Ever-Free 自由空间记忆栅格。

    任何体素只要在干净帧中被视线多次穿破（位于实测表面前方 > margin），即标记为 `ever_free`。
    后续一旦有手/人体进入该历史空域，立刻判定为动态前景并路由至 DynamicLayer，
    绝不写入静态地图、绝不污染静态 ICP 配准。
    """

    def __init__(
        self,
        res_m: float = 0.030,
        x_range: tuple[float, float] = (-0.30, 1.80),
        y_range: tuple[float, float] = (-1.10, 1.10),
        z_range: tuple[float, float] = (-0.04, 0.72),
        confirm_hits: int = 2,
    ) -> None:
        self.res_m = float(res_m)
        self.origin = np.array([x_range[0], y_range[0], z_range[0]], dtype=np.float32)
        self.nx = int(math.ceil((x_range[1] - x_range[0]) / self.res_m))
        self.ny = int(math.ceil((y_range[1] - y_range[0]) / self.res_m))
        self.nz = int(math.ceil((z_range[1] - z_range[0]) / self.res_m))
        self.confirm_hits = int(confirm_hits)
        self.free_hits = np.zeros((self.nx, self.ny, self.nz), dtype=np.uint8)

    def reset(self) -> None:
        self.free_hits.fill(0)

    def integrate_clean_rays(
        self,
        cam_pos: np.ndarray,
        hit_pts: np.ndarray,
        safety_margin_m: float = 0.045,
        min_ray_m: float = 0.16,
        stride: int = 4,
        n_samples: int = 48,
    ) -> int:
        """沿干净帧的相机射线在 `[min_ray_m, d_hit - safety_margin_m]` 内累加自由空间计数（步长满足体素奈奎斯特采样）。"""
        if len(hit_pts) == 0:
            return 0
        sub = hit_pts[:: max(1, int(stride))]
        vec = sub - cam_pos[None, :]
        dist = np.linalg.norm(vec, axis=-1)
        valid = dist > (min_ray_m + safety_margin_m + 0.02)
        if not np.any(valid):
            return 0
        vec = vec[valid]
        dist = dist[valid]
        dirs = vec / dist[:, None]
        # 距离自适应安全余量 (Dynablox / UFOMap)：近景 4.5cm 精细隔离桌面空域，远景按 9% 距离放宽余量防止单目远墙深度噪声自雕刻
        adaptive_margin = np.maximum(float(safety_margin_m), 0.09 * dist)
        d_end = np.minimum(dist - adaptive_margin, 1.85)
        keep_r = d_end > (min_ray_m + 0.02)
        if not np.any(keep_r):
            return 0
        dirs = dirs[keep_r]
        d_end = d_end[keep_r]

        t_frac = np.linspace(0.0, 1.0, int(n_samples), dtype=np.float32)[None, :]  # (1, S)
        r_dist = min_ray_m + (d_end[:, None] - min_ray_m) * t_frac  # (R, S)
        samples = cam_pos[None, None, :] + dirs[:, None, :] * r_dist[:, :, None]  # (R, S, 3)
        flat = samples.reshape(-1, 3)

        # 不雕刻桌面 Z < 2.2cm 的极近桌面层，防止桌面表面被斜向光线误判成 ever_free
        flat = flat[flat[:, 2] >= 0.024]
        if len(flat) == 0:
            return 0

        idx = np.floor((flat - self.origin[None, :]) / self.res_m).astype(np.int32)
        inb = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.nz)
        )
        idx = idx[inb]
        if len(idx) == 0:
            return 0

        # 单帧去重：每帧每个体素最多累加 +1 次，避免单帧噪声把未知空域直接确认
        lin = (idx[:, 0] * self.ny + idx[:, 1]) * self.nz + idx[:, 2]
        uniq = np.unique(lin)

        # Dynablox §III-A Occupied-Protected Ever-Free：
        # 本帧被静态物体表面实际占据的体素绝不能被邻近背景掠射光线误标为 ever_free，
        # 彻底杜绝杯子/立柱/笔记本边缘因体素混叠被错划进 DynamicLayer！
        occ_idx = np.floor((hit_pts - self.origin[None, :]) / self.res_m).astype(np.int32)
        occ_inb = (
            (occ_idx[:, 0] >= 0) & (occ_idx[:, 0] < self.nx)
            & (occ_idx[:, 1] >= 0) & (occ_idx[:, 1] < self.ny)
            & (occ_idx[:, 2] >= 0) & (occ_idx[:, 2] < self.nz)
        )
        if np.any(occ_inb):
            oi = occ_idx[occ_inb]
            occ_lin = np.unique((oi[:, 0] * self.ny + oi[:, 1]) * self.nz + oi[:, 2])
            # 从本帧待累加自由空域中剔除所有被表面占据的体素
            uniq = np.setdiff1d(uniq, occ_lin, assume_unique=True)

        flat_hits = self.free_hits.ravel()
        if len(uniq) > 0:
            flat_hits[uniq] = np.minimum(flat_hits[uniq].astype(np.int32) + 1, 15).astype(np.uint8)
        return int(len(uniq))

    def protect_static_voxels(self, static_pts: np.ndarray) -> None:
        """将已确认静态物体占据的体素从 ever_free 清零，防止静态边缘因邻近空光线产生体素混叠闪烁。"""
        if len(static_pts) == 0:
            return
        idx = np.floor((static_pts - self.origin[None, :]) / self.res_m).astype(np.int32)
        inb = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.nz)
        )
        if np.any(inb):
            vi = idx[inb]
            self.free_hits[vi[:, 0], vi[:, 1], vi[:, 2]] = 0

    def query_ever_free(self, pts: np.ndarray, expand_clusters: bool = True) -> np.ndarray:
        """查询每个 3D 点是否落入历史已确认空域 `ever_free`（含 Dynablox §III-B 团簇邻域扩张）。

        Dynablox 关键细节：悬停在桌面上方 3~5 cm 的手/物体，主体落入 `ever_free` 种子区，
        最底端因紧贴桌面处于光线安全余量内；通过 3D 邻域团簇扩张（仅限 `Z >= 3.2cm` 抬升点），
        可将整只手完整归入动态层，既不伤及 `Z=0` 静态桌面，也杜绝手部底缘漏进静态图。
        """
        n = len(pts)
        out = np.zeros((n,), dtype=bool)
        if n == 0:
            return out
        elev = pts[:, 2] >= 0.026
        if not np.any(elev):
            return out
        idx = np.floor((pts[elev] - self.origin[None, :]) / self.res_m).astype(np.int32)
        inb = (
            (idx[:, 0] >= 0) & (idx[:, 0] < self.nx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < self.ny)
            & (idx[:, 2] >= 0) & (idx[:, 2] < self.nz)
        )
        res_e = np.zeros((int(np.sum(elev)),), dtype=bool)
        if np.any(inb):
            valid_idx = idx[inb]
            res_e[inb] = self.free_hits[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] >= self.confirm_hits
        out[elev] = res_e

        # Dynablox 团簇级动态扩张：当存在显著动态种子 (>= 15 点) 时，把 2 个体素 (6cm) 内的紧邻抬升点一并归入动态层
        if expand_clusters and int(np.sum(out)) >= 15:
            seed_pts = pts[out]
            s_idx = np.floor((seed_pts - self.origin[None, :]) / self.res_m).astype(np.int32)
            s_inb = (
                (s_idx[:, 0] >= 0) & (s_idx[:, 0] < self.nx)
                & (s_idx[:, 1] >= 0) & (s_idx[:, 1] < self.ny)
                & (s_idx[:, 2] >= 0) & (s_idx[:, 2] < self.nz)
            )
            s_idx = s_idx[s_inb]
            if len(s_idx) > 0:
                seed_vol = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
                # 围绕种子体素做快速 1 步 3x3x3 + 竖直向下 2 格扩张（覆盖贴近桌面的物体底缘）
                for dx in (-1, 0, 1):
                    x_s = np.clip(s_idx[:, 0] + dx, 0, self.nx - 1)
                    for dy in (-1, 0, 1):
                        y_s = np.clip(s_idx[:, 1] + dy, 0, self.ny - 1)
                        for dz in (-2, -1, 0, 1):
                            z_s = np.clip(s_idx[:, 2] + dz, 0, self.nz - 1)
                            seed_vol[x_s, y_s, z_s] = True
                cand = (pts[:, 2] >= 0.032) & (~out)
                if np.any(cand):
                    c_idx = np.floor((pts[cand] - self.origin[None, :]) / self.res_m).astype(np.int32)
                    c_inb = (
                        (c_idx[:, 0] >= 0) & (c_idx[:, 0] < self.nx)
                        & (c_idx[:, 1] >= 0) & (c_idx[:, 1] < self.ny)
                        & (c_idx[:, 2] >= 0) & (c_idx[:, 2] < self.nz)
                    )
                    exp_res = np.zeros(int(np.sum(cand)), dtype=bool)
                    if np.any(c_inb):
                        ci = c_idx[c_inb]
                        exp_res[c_inb] = seed_vol[ci[:, 0], ci[:, 1], ci[:, 2]]
                    out[cand] = exp_res

        return out

    @property
    def confirmed_voxel_count(self) -> int:
        return int(np.sum(self.free_hits >= self.confirm_hits))


# ---------------------------------------------------------------------------
# 3. KISS-ICP 自适应阈值 + Geman-McClure 鲁棒 Point-to-Plane 静态配准器
# ---------------------------------------------------------------------------


class KissIcpStaticAligner:
    """源自 KISS-ICP (`PRBonn/kiss-icp`) + ElasticFusion 的自适应鲁棒静态场景配准器。

    围绕相机中心求解小量 6-DoF 修正量 `xi = [dt, domega] ∈ se(3)`：
    - 仅以 `StaticMap` 中已确认静态面元为目标；
    - 自适应对应阈值 `sigma_t` 按历史修正幅度自动收敛；
    - Geman-McClure 鲁棒核 `w(r) = tau^2 / (tau^2 + r^2)^2` + 法向一致性权重 `max(0.15, |n_p·n_q|)`
      天然免疫剩余未剔除动态外点；
    - Point-to-Plane (0.85) + 弱 Point-to-Point (0.15) 混合残差 + LM 阻尼，彻底避免纯平面滑动退化。
    """

    def __init__(
        self,
        init_sigma_m: float = 0.09,
        min_sigma_m: float = 0.022,
        max_sigma_m: float = 0.16,
        max_iters: int = 7,
        max_corr_rot_deg: float = 12.0,
        max_corr_trans_m: float = 0.14,
        max_source_pts: int = 1600,
    ) -> None:
        self.init_sigma_m = float(init_sigma_m)
        self.min_sigma_m = float(min_sigma_m)
        self.max_sigma_m = float(max_sigma_m)
        self.max_iters = int(max_iters)
        self.max_corr_rot_deg = float(max_corr_rot_deg)
        self.max_corr_trans_m = float(max_corr_trans_m)
        self.max_source_pts = int(max_source_pts)

        self._sigma_m = self.init_sigma_m
        self._dev_sq_sum = 0.0
        self._dev_count = 0

    def reset(self) -> None:
        self._sigma_m = self.init_sigma_m
        self._dev_sq_sum = 0.0
        self._dev_count = 0

    @property
    def adaptive_sigma_m(self) -> float:
        return self._sigma_m

    def align_to_static(
        self,
        src_pts: np.ndarray,
        src_nrms: np.ndarray,
        static_pts: np.ndarray,
        static_nrms: np.ndarray,
        cam_pos: np.ndarray,
        allow_translation: bool = True,
    ) -> dict[str, Any]:
        """将当前帧候选静态点配准到已确认静态地图。返回 (dR, dt, 质量指标)。"""
        eye3 = np.eye(3, dtype=np.float32)
        zero3 = np.zeros(3, dtype=np.float32)
        if len(src_pts) < 120 or len(static_pts) < 150:
            return {
                "dR": eye3,
                "dt": zero3,
                "inlier_ratio": 1.0,
                "median_res_m": 0.0,
                "rot_deg": 0.0,
                "trans_m": 0.0,
                "converged": False,
            }

        # 空间均匀子采样以保证 < 4 ms CPU 实时性
        if len(src_pts) > self.max_source_pts:
            step = max(1, len(src_pts) // self.max_source_pts)
            p_cur = src_pts[::step].astype(np.float64).copy()
            n_cur = src_nrms[::step].astype(np.float64).copy()
        else:
            p_cur = src_pts.astype(np.float64).copy()
            n_cur = src_nrms.astype(np.float64).copy()

        if len(static_pts) > 8000:
            s_step = max(1, len(static_pts) // 8000)
            q_ref = static_pts[::s_step].astype(np.float64)
            nq_ref = static_nrms[::s_step].astype(np.float64)
        else:
            q_ref = static_pts.astype(np.float64)
            nq_ref = static_nrms.astype(np.float64)

        tree = cKDTree(q_ref)
        pivot = cam_pos.astype(np.float64)

        r_total = np.eye(3, dtype=np.float64)
        t_total = np.zeros(3, dtype=np.float64)

        # KISS-ICP 自适应阈值：搜索半径 = 3 * sigma，Geman-McClure 尺度 tau = 0.65 * sigma
        max_corr_dist = float(np.clip(3.0 * self._sigma_m, 0.06, 0.36))
        tau = float(np.clip(0.65 * self._sigma_m, 0.014, 0.09))
        tau_sq = tau * tau

        inlier_ratio = 0.0
        med_res = 0.0

        for _ in range(self.max_iters):
            dists, nn_idx = tree.query(p_cur, distance_upper_bound=max_corr_dist)
            valid = np.isfinite(dists) & (nn_idx < len(q_ref))
            inlier_ratio = float(np.mean(valid))
            if int(np.sum(valid)) < 60:
                break

            pv = p_cur[valid]
            nv = n_cur[valid]
            qv = q_ref[nn_idx[valid]]
            nqv = nq_ref[nn_idx[valid]]

            # 法向一致性权重 (ElasticFusion)：拒绝反面/正交错配
            n_dot = np.abs(np.sum(nv * nqv, axis=-1))
            good_n = n_dot > 0.30
            if int(np.sum(good_n)) < 50:
                break
            pv, qv, nqv, n_dot = pv[good_n], qv[good_n], nqv[good_n], n_dot[good_n]

            diff = pv - qv
            r_pln = np.sum(nqv * diff, axis=-1)
            med_res = float(np.median(np.abs(r_pln)))

            # Geman-McClure 鲁棒权重: w(r) = tau^2 / (tau^2 + r^2)^2
            # 用归一化标度使权重在 [0, 1] 区间，保证阻尼矩阵尺度稳定
            r_sq = r_pln * r_pln + 0.12 * np.sum(diff * diff, axis=-1)
            w_gm = (tau_sq / np.maximum(tau_sq + r_sq, 1e-12)) ** 2
            w_ang = np.maximum(0.15, n_dot)
            w = w_gm * w_ang

            pc = pv - pivot[None, :]
            # Point-to-Plane Jacobian: J_pln = [n^T, (pc x n)^T]
            j_pln = np.concatenate([nqv, np.cross(pc, nqv)], axis=-1)  # (M, 6)
            w_pln = w * 0.85
            h_mat = (j_pln * w_pln[:, None]).T @ j_pln
            g_vec = j_pln.T @ (w_pln * r_pln)

            # 弱 Point-to-Point 锚定 (权重 0.15)，防止光滑平面上切向漂移
            w_pt = w * 0.15
            # 构建 3 轴向量化 J_pt^T W J_pt 与 J_pt^T W diff
            # e = diff + dt - [pc]_x domega
            sum_w = float(np.sum(w_pt))
            if sum_w > 1e-8:
                h_mat[0, 0] += sum_w
                h_mat[1, 1] += sum_w
                h_mat[2, 2] += sum_w
                g_vec[:3] += np.sum(w_pt[:, None] * diff, axis=0)

                # 交叉项 -[pc]_x
                wx = w_pt * pc[:, 0]
                wy = w_pt * pc[:, 1]
                wz = w_pt * pc[:, 2]
                sx, sy, sz = float(np.sum(wx)), float(np.sum(wy)), float(np.sum(wz))
                c_block = np.array(
                    [
                        [0.0, sz, -sy],
                        [-sz, 0.0, sx],
                        [sy, -sx, 0.0],
                    ],
                    dtype=np.float64,
                )
                h_mat[:3, 3:] += c_block
                h_mat[3:, :3] += c_block.T

                # 旋转块 [pc]_x^T W [pc]_x
                xx = float(np.sum(wx * pc[:, 0]))
                yy = float(np.sum(wy * pc[:, 1]))
                zz = float(np.sum(wz * pc[:, 2]))
                xy = float(np.sum(wx * pc[:, 1]))
                xz = float(np.sum(wx * pc[:, 2]))
                yz = float(np.sum(wy * pc[:, 2]))
                h_mat[3:, 3:] += np.array(
                    [
                        [yy + zz, -xy, -xz],
                        [-xy, xx + zz, -yz],
                        [-xz, -yz, xx + yy],
                    ],
                    dtype=np.float64,
                )
                # g_rot = (- [pc]_x)^T (w_pt * diff) = pc x (w_pt * diff)
                g_vec[3:] += np.sum(np.cross(pc, w_pt[:, None] * diff), axis=0)

            # Levenberg-Marquardt 阻尼；若禁用平移则极大加重平移阻尼，仅精修姿态 (pitch/roll/yaw)
            trans_damp = 4.0 if allow_translation else 1e6
            damp = np.diag([trans_damp, trans_damp, 2.5 if allow_translation else 1e6, 0.8, 0.8, 0.8])
            try:
                xi = -np.linalg.solve(h_mat + damp, g_vec)
            except np.linalg.LinAlgError:
                break

            dt_step = xi[:3]
            dw_step = xi[3:]
            # 限制单步更新幅度以保证数值稳定
            dw_norm = float(np.linalg.norm(dw_step))
            if dw_norm > math.radians(5.0):
                dw_step = dw_step * (math.radians(5.0) / dw_norm)
            dt_norm = float(np.linalg.norm(dt_step))
            if dt_norm > 0.06:
                dt_step = dt_step * (0.06 / dt_norm)

            dr_step, _ = cv2.Rodrigues(dw_step.reshape(3, 1))
            p_cur = (p_cur - pivot[None, :]) @ dr_step.T + pivot[None, :] + dt_step[None, :]
            n_cur = n_cur @ dr_step.T

            r_total = dr_step @ r_total
            t_total = dr_step @ t_total + dt_step

            if dw_norm < 1e-4 and dt_norm < 1e-4:
                break

        # 物理限幅保护：单帧总修正超限则拒绝（防极端外点把地图拉飞）
        rvec, _ = cv2.Rodrigues(r_total)
        rot_deg = math.degrees(float(np.linalg.norm(rvec)))
        trans_m = float(np.linalg.norm(t_total))
        if rot_deg > self.max_corr_rot_deg or trans_m > self.max_corr_trans_m:
            return {
                "dR": eye3,
                "dt": zero3,
                "inlier_ratio": inlier_ratio,
                "median_res_m": med_res,
                "rot_deg": rot_deg,
                "trans_m": trans_m,
                "converged": False,
            }

        # 更新 KISS-ICP 自适应阈值：sigma = sqrt(mean(delta^2))
        # 以 0.7m 处等效位移度量姿态与平移偏差
        equiv_dev = math.sqrt(trans_m * trans_m + (0.70 * math.radians(rot_deg)) ** 2)
        if equiv_dev > 0.004:
            self._dev_sq_sum += equiv_dev * equiv_dev
            self._dev_count += 1
            est_sigma = math.sqrt(self._dev_sq_sum / self._dev_count)
            self._sigma_m = float(np.clip(est_sigma, self.min_sigma_m, self.max_sigma_m))

        return {
            "dR": r_total.astype(np.float32),
            "dt": t_total.astype(np.float32),
            "inlier_ratio": float(inlier_ratio),
            "median_res_m": float(med_res),
            "rot_deg": float(rot_deg),
            "trans_m": float(trans_m),
            "converged": True,
        }


# ---------------------------------------------------------------------------
# 4. 双层静态精修 + 动态分离 3D 轮廓建图引擎 (ElasticFusion + Dynablox + nvblox)
# ---------------------------------------------------------------------------


class RobustDualLayerContourMap:
    """结合 KISS-ICP、ElasticFusion、Dynablox、nvblox 与 Unblur-SLAM 的双层 3D 轮廓地图。

    核心架构：
    - `StaticMap`（静态骨架层）：0.85 cm 面元哈希表，包含 `tentative -> confirmed` 稳定度毕业、
      斜视角降权、逆方差位置/法向融合、桌面 `Z=0` 平面平滑吸附、干净射线穿透雕刻；
    - `EverFreeVoxelGrid`（Dynablox 历史空域）：标记已确认空域，防止进入空域的人/手烤入静态层；
    - `DynamicLayer`（nvblox 动态瞬态层）：承接历史空域入侵点、深度跳变动态前景以及扰动帧实时点，
      每帧自动衰减更新，实现零重影实时刷新；
    - `KissIcpStaticAligner`：每帧先剔除历史空域动态点，再用剩余静态候选点对齐 `Confirmed StaticMap`，
      自动修正平衡俯仰摇摆与舵机回程差。
    """

    def __init__(
        self,
        vox_size: float = 0.0085,
        fov_h_deg: float = 72.0,
        confirm_hits: int = 2,
        enable_icp: bool = True,
        enable_ever_free: bool = True,
        enable_disturbance_gate: bool = True,
    ) -> None:
        self.vox_size = float(vox_size)
        self.fov_h_deg = float(fov_h_deg)
        self.confirm_hits = int(confirm_hits)
        self.enable_icp = bool(enable_icp)
        self.enable_ever_free = bool(enable_ever_free)
        self.enable_disturbance_gate = bool(enable_disturbance_gate)

        self.gate = DisturbanceQualityGate()
        self.ever_free = EverFreeVoxelGrid(res_m=0.030, confirm_hits=2)
        self.aligner = KissIcpStaticAligner()

        # StaticMap (SoA)
        self.s_pos = np.zeros((0, 3), dtype=np.float32)
        self.s_nrm = np.zeros((0, 3), dtype=np.float32)
        self.s_wt = np.zeros((0,), dtype=np.float32)
        self.s_hits = np.zeros((0,), dtype=np.int32)
        self.s_last_step = np.zeros((0,), dtype=np.int32)

        # DynamicLayer (nvblox 实时瞬态前景层，每帧刷新)
        self.d_pos = np.zeros((0, 3), dtype=np.float32)
        self.d_nrm = np.zeros((0, 3), dtype=np.float32)

        self.step = 0
        self.static_updates = 0
        self.disturbed_frames_blocked = 0
        self.last_carved_static = 0
        self.last_dynamic_routed = 0
        self.last_status = DisturbanceStatus(False, "init", 100.0, 1.0, 0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.gate.reset()
        self.ever_free.reset()
        self.aligner.reset()
        self.s_pos = np.zeros((0, 3), dtype=np.float32)
        self.s_nrm = np.zeros((0, 3), dtype=np.float32)
        self.s_wt = np.zeros((0,), dtype=np.float32)
        self.s_hits = np.zeros((0,), dtype=np.int32)
        self.s_last_step = np.zeros((0,), dtype=np.int32)
        self.d_pos = np.zeros((0, 3), dtype=np.float32)
        self.d_nrm = np.zeros((0, 3), dtype=np.float32)
        self.step = 0
        self.static_updates = 0
        self.disturbed_frames_blocked = 0
        self.last_carved_static = 0
        self.last_dynamic_routed = 0

    def _pack_keys(self, pts: np.ndarray) -> np.ndarray:
        inv_vs = 1.0 / self.vox_size
        qi = np.floor(pts * inv_vs).astype(np.int64) + 1024
        return (qi[:, 0] << 24) | (qi[:, 1] << 12) | qi[:, 2]

    def _compute_oblique_weights(self, pts: np.ndarray, nrms: np.ndarray, cam_pos: np.ndarray) -> np.ndarray:
        """ElasticFusion 斜视角降权 + 距离反平方精度权重。"""
        v = cam_pos[None, :] - pts
        dist = np.maximum(np.linalg.norm(v, axis=-1), 0.14)
        v_dir = v / dist[:, None]
        cos_inc = np.abs(np.sum(nrms * v_dir, axis=-1))
        w_angle = np.clip(cos_inc, 0.15, 1.0)
        w_dist = 1.0 / (1.0 + 0.65 * dist * dist)
        return (w_angle * w_dist).astype(np.float32)

    def update(
        self,
        bgr: np.ndarray,
        pts: np.ndarray,
        nrms: np.ndarray,
        depth_hr: np.ndarray,
        r_level: np.ndarray,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        cam_x: float = 0.0,
        cam_y: float = 0.0,
        z_bias: float = 0.0,
    ) -> DisturbanceStatus:
        """融合一帧 3D 点云：扰动拦截 -> Ever-Free 动态分流 -> KISS-ICP 静态配准 -> 双层地图更新。"""
        self.step += 1
        cam_pos = np.array([cam_x, cam_y, DEFAULT_H_CAM_M - z_bias], dtype=np.float32)

        # 1) Unblur-SLAM 前置扰动质量门控
        status = self.gate.evaluate_pre_icp(bgr, pitch_deg=pitch_deg, roll_deg=roll_deg, yaw_deg=yaw_deg)
        if not self.enable_disturbance_gate:
            status.is_disturbed = False
            status.reason = "gate_disabled"

        if len(pts) == 0:
            self.d_pos = np.zeros((0, 3), dtype=np.float32)
            self.d_nrm = np.zeros((0, 3), dtype=np.float32)
            self.last_status = status
            return status

        # 若触发运动模糊 / 剧烈角冲击，冻结 StaticMap 保护静态骨架，实时帧仅推至 DynamicLayer 渲染
        if status.is_disturbed:
            self.disturbed_frames_blocked += 1
            self.d_pos = pts[::2].astype(np.float32)
            self.d_nrm = nrms[::2].astype(np.float32)
            self.last_dynamic_routed = len(self.d_pos)
            self.last_carved_static = 0
            self.last_status = status
            return status

        # 2) Dynablox 历史空域查询（粗分离：进入已知空域的物体立即标为动态，不参加静态 ICP）
        if self.enable_ever_free and self.ever_free.confirmed_voxel_count > 0:
            is_dyn_pre = self.ever_free.query_ever_free(pts)
        else:
            is_dyn_pre = np.zeros(len(pts), dtype=bool)

        # 3) KISS-ICP 鲁棒静态配准：仅用非历史空域点对齐已确认静态面元，消除俯仰晃动与舵机回程差
        conf_mask = self.s_hits >= self.confirm_hits
        pts_corr = pts
        nrms_corr = nrms
        dR = np.eye(3, dtype=np.float32)
        dt = np.zeros(3, dtype=np.float32)

        if self.enable_icp and int(np.sum(conf_mask)) >= 250:
            src_mask = ~is_dyn_pre
            icp_res = self.aligner.align_to_static(
                src_pts=pts[src_mask],
                src_nrms=nrms[src_mask],
                static_pts=self.s_pos[conf_mask],
                static_nrms=self.s_nrm[conf_mask],
                cam_pos=cam_pos,
                allow_translation=True,
            )
            status.icp_inlier_ratio = icp_res["inlier_ratio"]
            status.icp_rot_corr_deg = icp_res["rot_deg"]
            status.icp_trans_corr_m = icp_res["trans_m"]
            status.icp_median_res_m = icp_res["median_res_m"]

            if icp_res["converged"]:
                dR = icp_res["dR"]
                dt = icp_res["dt"]
                pts_corr = ((pts - cam_pos[None, :]) @ dR.T + cam_pos[None, :] + dt[None, :]).astype(np.float32)
                nrms_corr = (nrms @ dR.T).astype(np.float32)
                # 配准后再做一次精确历史空域查询
                if self.enable_ever_free and self.ever_free.confirmed_voxel_count > 0:
                    is_dyn_pre = self.ever_free.query_ever_free(pts_corr)
            elif self.enable_disturbance_gate and icp_res["inlier_ratio"] < self.gate.min_icp_inlier_ratio:
                # 跟踪严重失配：冻结静态融合，防止重影叠图
                status.is_disturbed = True
                status.reason = "icp_tracking_loss"
                self.disturbed_frames_blocked += 1
                self.d_pos = pts[::2].astype(np.float32)
                self.d_nrm = nrms[::2].astype(np.float32)
                self.last_dynamic_routed = len(self.d_pos)
                self.last_status = status
                return status

        # 4) 二次几何一致性检测：若已确认静态地图在当前射线更远处有清晰静态背景，
        #    则近处突然冒出的大残差块（即使首帧未建立 ever_free）也自动标为动态前景
        is_dyn = is_dyn_pre.copy()
        hr_h, hr_w = depth_hr.shape
        rad = math.radians(float(yaw_deg))
        cy, sy = math.cos(rad), math.sin(rad)
        tan_h = math.tan(math.radians(self.fov_h_deg * 0.5))
        fx = (hr_w * 0.5) / tan_h
        fy = (hr_h * 0.5) / tan_h

        carved_cnt = 0
        if len(self.s_pos) > 0:
            # 将静态点逆变换到已修正的当前相机坐标系
            # 正向：p_world = cam_pos + dR @ R_yaw @ R_level @ p_cam + dt
            # 逆向：p_cam = R_level^T @ R_yaw^T @ dR^T @ (p_world - cam_pos - dt)
            rel_w = (self.s_pos - cam_pos[None, :] - dt[None, :]) @ dR  # dR^T 右乘等价于行向量乘 dR
            lx = rel_w[:, 0] * cy + rel_w[:, 1] * sy
            ly = -rel_w[:, 0] * sy + rel_w[:, 1] * cy
            lz = rel_w[:, 2]
            cam_p = np.stack([lx, ly, lz], axis=-1) @ r_level

            cx_v = cam_p[:, 0]
            in_front = cx_v > 0.14
            u_px = np.rint((hr_w * 0.5 - 0.5) + (cam_p[:, 1] / np.maximum(cx_v, 1e-4)) * fx).astype(np.int32)
            v_px = np.rint((hr_h * 0.5 - 0.5) - (cam_p[:, 2] / np.maximum(cx_v, 1e-4)) * fy).astype(np.int32)
            in_fov = in_front & (u_px >= 2) & (u_px < hr_w - 2) & (v_px >= 2) & (v_px < hr_h - 2)

            keep = np.ones(len(self.s_pos), dtype=bool)
            if np.any(in_fov):
                idx = np.where(in_fov)[0]
                d_new_at = depth_hr[v_px[idx], u_px[idx]]
                d_old_at = cx_v[idx]
                # 自由空间光线雕刻：原静态点落在当前实测深度前方 (>4.2cm)，已被视线穿破
                carved = d_old_at < (d_new_at - 0.042)
                carved_cnt = int(np.sum(carved))
                if carved_cnt > 0:
                    self.s_wt[idx[carved]] *= 0.08
                    self.s_hits[idx[carved]] = np.maximum(0, self.s_hits[idx[carved]] - 2)
                keep[self.s_wt < 0.14] = False

            # 淘汰长时间未被二次确认的 tentative 单次噪点 (ElasticFusion 垃圾回收)
            age = self.step - self.s_last_step
            tentative_stale = (self.s_hits < self.confirm_hits) & (age > 6)
            keep &= ~tentative_stale

            if not np.all(keep):
                self.s_pos = self.s_pos[keep]
                self.s_nrm = self.s_nrm[keep]
                self.s_wt = self.s_wt[keep]
                self.s_hits = self.s_hits[keep]
                self.s_last_step = self.s_last_step[keep]

        # 5) 拆分静态候选与实时动态层
        static_in_mask = ~is_dyn
        dyn_pts = pts_corr[is_dyn]
        dyn_nrm = nrms_corr[is_dyn]
        st_pts = pts_corr[static_in_mask]
        st_nrm = nrms_corr[static_in_mask]

        # 实时动态层去重压缩，供纯绿轮廓高亮显示
        if len(dyn_pts) > 0:
            d_keys = self._pack_keys(dyn_pts)
            _, d_first = np.unique(d_keys, return_index=True)
            self.d_pos = dyn_pts[d_first].astype(np.float32)
            self.d_nrm = dyn_nrm[d_first].astype(np.float32)
        else:
            self.d_pos = np.zeros((0, 3), dtype=np.float32)
            self.d_nrm = np.zeros((0, 3), dtype=np.float32)
        self.last_dynamic_routed = len(self.d_pos)

        # 6) 融合静态候选点进 StaticMap（逆方差位置/法向融合 + 平衡折返点角速度加权 + 桌面平面平滑吸附）
        # 双轮自平衡在 ~1.16Hz 前后摇摆：过零点角速度大（略糊），上下折返点角速度≈0（最锐利），
        # 按 w_gyro = 1 / (1 + (dpitch/2.2)^2) 加权让折返清晰帧自然主导静态轮廓收敛！
        if len(st_pts) > 0:
            w_gyro = 1.0 / (1.0 + (float(status.dpitch_deg) / 2.2) ** 2 + (float(status.dyaw_deg) / 5.0) ** 2)
            st_w = self._compute_oblique_weights(st_pts, st_nrm, cam_pos + dt) * np.float32(max(0.20, w_gyro))
            new_keys = self._pack_keys(st_pts)
            uniq_new_keys, first_idx = np.unique(new_keys, return_index=True)
            n_pts = st_pts[first_idx]
            n_nrm = st_nrm[first_idx]
            n_w = st_w[first_idx]

            if len(self.s_pos) == 0:
                self.s_pos = n_pts.astype(np.float32)
                self.s_nrm = n_nrm.astype(np.float32)
                self.s_wt = n_w.astype(np.float32)
                self.s_hits = np.ones(len(n_pts), dtype=np.int32)
                self.s_last_step = np.full(len(n_pts), self.step, dtype=np.int32)
            else:
                old_keys = self._pack_keys(self.s_pos)
                # 用排序二分查找实现 O((N+M) log N) 快速静态体素原位融合
                order_old = np.argsort(old_keys)
                sorted_old = old_keys[order_old]
                pos_ins = np.searchsorted(sorted_old, uniq_new_keys)
                pos_clip = np.minimum(pos_ins, len(sorted_old) - 1)
                matched = (pos_ins < len(sorted_old)) & (sorted_old[pos_clip] == uniq_new_keys)

                if np.any(matched):
                    old_idx = order_old[pos_clip[matched]]
                    m_pts = n_pts[matched]
                    m_nrm = n_nrm[matched]
                    m_w = n_w[matched]

                    w_prev = np.minimum(self.s_wt[old_idx], 12.0)
                    w_sum = w_prev + m_w
                    alpha = (m_w / np.maximum(w_sum, 1e-6))[:, None]

                    self.s_pos[old_idx] = (1.0 - alpha) * self.s_pos[old_idx] + alpha * m_pts
                    blended_n = (1.0 - alpha) * self.s_nrm[old_idx] + alpha * m_nrm
                    blended_n /= np.maximum(np.linalg.norm(blended_n, axis=-1, keepdims=True), 1e-6)
                    self.s_nrm[old_idx] = blended_n
                    self.s_wt[old_idx] = np.minimum(w_sum, 14.0)
                    self.s_hits[old_idx] = np.minimum(self.s_hits[old_idx] + 1, 1000)
                    self.s_last_step[old_idx] = self.step

                    # 桌面平面精修吸附：已确认水平桌面点平滑投影向 Z=0，消除橘皮噪点
                    desk_snap = (
                        (self.s_hits[old_idx] >= self.confirm_hits)
                        & (np.abs(self.s_pos[old_idx, 2]) < 0.024)
                        & (self.s_nrm[old_idx, 2] > 0.84)
                    )
                    if np.any(desk_snap):
                        ds_idx = old_idx[desk_snap]
                        self.s_pos[ds_idx, 2] *= 0.78

                if np.any(~matched):
                    add_pts = n_pts[~matched]
                    add_nrm = n_nrm[~matched]
                    add_w = n_w[~matched]
                    self.s_pos = np.concatenate([self.s_pos, add_pts.astype(np.float32)], axis=0)
                    self.s_nrm = np.concatenate([self.s_nrm, add_nrm.astype(np.float32)], axis=0)
                    self.s_wt = np.concatenate([self.s_wt, add_w.astype(np.float32)], axis=0)
                    self.s_hits = np.concatenate([self.s_hits, np.ones(len(add_pts), dtype=np.int32)], axis=0)
                    self.s_last_step = np.concatenate([self.s_last_step, np.full(len(add_pts), self.step, dtype=np.int32)], axis=0)

        # 7) 用本帧干净静态点沿射线更新 Dynablox Ever-Free 空域栅格，并保护已确认静态表面体素
        if self.enable_ever_free and len(pts_corr) > 0:
            self.ever_free.integrate_clean_rays(cam_pos + dt, pts_corr, safety_margin_m=0.045, stride=4, n_samples=48)
            conf_now = self.s_hits >= self.confirm_hits
            if np.any(conf_now):
                self.ever_free.protect_static_voxels(self.s_pos[conf_now])

        self.static_updates += 1
        self.last_carved_static = carved_cnt
        self.last_status = status
        return status

    def get_static_confirmed(self) -> tuple[np.ndarray, np.ndarray]:
        """返回已稳定确认的静态骨架点与法向（若初期尚无 >= confirm_hits 点则返回全部静态点）。"""
        if len(self.s_pos) == 0:
            return self.s_pos, self.s_nrm
        conf = self.s_hits >= self.confirm_hits
        if int(np.sum(conf)) >= 200:
            return self.s_pos[conf], self.s_nrm[conf]
        return self.s_pos, self.s_nrm

    def get_combined_for_traversability(self) -> np.ndarray:
        """返回用于 5 cm 可通行域计算的点集（已确认静态障碍 + 当前实时动态前景）。"""
        s_pos, _ = self.get_static_confirmed()
        if len(self.d_pos) == 0:
            return s_pos
        if len(s_pos) == 0:
            return self.d_pos
        return np.concatenate([s_pos, self.d_pos], axis=0)

    def desk_flatness_metrics(self) -> dict[str, float]:
        """量化评估当前静态地图中桌面平面（|z|<0.05m 且 nz>0.70）的平整度（std 与 MAD，单位 mm）。"""
        s_pos, s_nrm = self.get_static_confirmed()
        if len(s_pos) == 0:
            return {"desk_pts": 0.0, "z_std_mm": 0.0, "z_mad_mm": 0.0}
        m = (np.abs(s_pos[:, 2]) < 0.055) & (s_nrm[:, 2] > 0.70)
        if int(np.sum(m)) < 30:
            return {"desk_pts": float(np.sum(m)), "z_std_mm": 0.0, "z_mad_mm": 0.0}
        zs = s_pos[m, 2]
        med = float(np.median(zs))
        return {
            "desk_pts": float(np.sum(m)),
            "z_std_mm": float(np.std(zs) * 1000.0),
            "z_mad_mm": float(np.median(np.abs(zs - med)) * 1000.0),
        }


# ---------------------------------------------------------------------------
# 5. 纯绿色雷达轮廓 EDL 渲染器 (Potree / CloudCompare 风格，完全抛弃 RGB 颜色)
# ---------------------------------------------------------------------------


def render_green_contour_edl(
    static_pts: np.ndarray,
    static_nrms: np.ndarray,
    cam: OrbitCamera3D,
    grid: TraversabilityGrid2D | None = None,
    dyn_pts: np.ndarray | None = None,
    dyn_nrms: np.ndarray | None = None,
    w: int = 640,
    h: int = 440,
    robot_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
    path: list[tuple[float, float]] | None = None,
    click_marker: tuple[float, float, tuple[int, int, int]] | None = None,
    head_yaw_deg: float = 0.0,
    draw_ground_grid: bool = True,
    draw_trav_overlay: bool = False,
    edl_gain: float = 4.6,
    contour_boost: float = 0.62,
) -> np.ndarray:
    """纯绿色立体轮廓 3D 点云渲染器（完全不依赖相机 RGB/灰度反照率）。

    视觉设计：
    - 深黑绿战术雷达底色 `#060D09` + 暗翡翠 10cm 地面透视网格；
    - 几何明暗 = 高度分层基调 + 法向朗伯光照 + Fresnel 轮廓掠射高光 `(1 - |n·v|)^2`；
    - 屏幕空间双半径 Eye-Dome Lighting (EDL)：
      - 后景深度跳变暗描边（清晰分离前后重叠物体）；
      - 前景深度边缘亮绿描边（让每个水杯、键盘、笔记本、手臂、拳头、人物轮廓刀锋般锐利）；
    - 静态骨架 (`#16E66E` 雷达翡翠绿) 与实时动态前景 (`#7BFFB8` 亮薄荷绿) 同属纯绿色系，
      一眼即可分辨"固定物体轮廓"与"正在移动的人/手轮廓"。
    """
    basis = _PROJ.compute_camera_basis(cam)
    focal = _PROJ.compute_focal_px(cam, float(h))
    cx, cy = w * 0.5, h * 0.5
    cam_pos = np.array([basis[0], basis[1], basis[2]], dtype=np.float32)

    # BGR 深黑绿雷达底色 (#060D09)
    bg_bgr = (9, 13, 6)
    img = np.full((h, w, 3), np.array(bg_bgr, dtype=np.uint8), dtype=np.uint8)

    # 10cm 暗绿透视参考地面网格
    if draw_ground_grid:
        gg = img.copy()
        step = 0.10
        x_lo, x_hi = -0.30, 1.60
        y_lo, y_hi = -1.00, 1.00
        for i in range(int(round((x_hi - x_lo) / step)) + 1):
            wx = x_lo + i * step
            p0 = _PROJ.project_world_point(float(wx), float(y_lo), 0.0, cam, float(w), float(h))
            p1 = _PROJ.project_world_point(float(wx), float(y_hi), 0.0, cam, float(w), float(h))
            if p0 is not None and p1 is not None:
                major = (i % 5 == 0)
                col_g = (28, 72, 22) if major else (16, 40, 12)
                cv2.line(gg, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col_g, 1, cv2.LINE_AA)
        for i in range(int(round((y_hi - y_lo) / step)) + 1):
            wy = y_lo + i * step
            p0 = _PROJ.project_world_point(float(x_lo), float(wy), 0.0, cam, float(w), float(h))
            p1 = _PROJ.project_world_point(float(x_hi), float(wy), 0.0, cam, float(w), float(h))
            if p0 is not None and p1 is not None:
                major = (i % 5 == 0)
                col_g = (28, 72, 22) if major else (16, 40, 12)
                cv2.line(gg, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col_g, 1, cv2.LINE_AA)
        img = cv2.addWeighted(gg, 0.72, img, 0.28, 0)

    # 可选 5cm 可达域贴片（偏克制的暗绿底纹，不抢占 3D 亮绿轮廓视觉重心）
    if draw_trav_overlay and grid is not None:
        res = grid.res
        ov = img.copy()
        for r_i, wy in enumerate(grid.ys):
            for c_i, wx in enumerate(grid.xs):
                is_g = bool(grid.reach[r_i, c_i])
                is_r = bool(grid.obs_mask[r_i, c_i]) and not is_g
                if not (is_g or is_r):
                    continue
                corners = []
                ok_c = True
                for dx, dy in ((0, 0), (res, 0), (res, res), (0, res)):
                    pr = _PROJ.project_world_point(float(wx + dx), float(wy + dy), 0.0, cam, float(w), float(h))
                    if pr is None:
                        ok_c = False
                        break
                    corners.append([int(round(pr[0])), int(round(pr[1]))])
                if not ok_c:
                    continue
                poly = np.array(corners, dtype=np.int32)
                if is_g:
                    cv2.fillPoly(ov, [poly], (24, 96, 22))
                    cv2.polylines(ov, [poly], True, (36, 145, 34), 1, cv2.LINE_AA)
                else:
                    cv2.polylines(ov, [poly], True, (38, 42, 165), 1, cv2.LINE_AA)
        img = cv2.addWeighted(ov, 0.36, img, 0.64, 0)

    c_buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    z_buf = np.full((h, w), 1e4, dtype=np.float32)
    is_dyn_buf = np.zeros((h, w), dtype=bool)
    is_elev_buf = np.zeros((h, w), dtype=bool)

    # 合并静态层与动态层统一按深度从远到近光栅化
    pts_list = []
    nrm_list = []
    dyn_flag_list = []
    if len(static_pts) > 0:
        pts_list.append(static_pts)
        nrm_list.append(static_nrms)
        dyn_flag_list.append(np.zeros(len(static_pts), dtype=bool))
    if dyn_pts is not None and len(dyn_pts) > 0:
        pts_list.append(dyn_pts)
        nrm_list.append(dyn_nrms if dyn_nrms is not None else np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(dyn_pts), 1)))
        dyn_flag_list.append(np.ones(len(dyn_pts), dtype=bool))

    if pts_list:
        all_pts = np.concatenate(pts_list, axis=0)
        all_nrm = np.concatenate(nrm_list, axis=0)
        all_dyn = np.concatenate(dyn_flag_list, axis=0)

        rx = all_pts[:, 0] - basis[0]
        ry = all_pts[:, 1] - basis[1]
        rz = all_pts[:, 2] - basis[2]
        z_cam = rx * basis[3] + ry * basis[4] + rz * basis[5]
        vis = z_cam > 0.08
        if np.any(vis):
            pts_v = all_pts[vis]
            nrms_v = all_nrm[vis]
            dyn_v = all_dyn[vis]
            z_v = z_cam[vis]

            x_cam = rx[vis] * basis[6] + ry[vis] * basis[7] + rz[vis] * basis[8]
            y_cam = rx[vis] * basis[9] + ry[vis] * basis[10] + rz[vis] * basis[11]
            sx = np.rint(cx + (x_cam / z_v) * focal).astype(np.int32)
            sy = np.rint(cy - (y_cam / z_v) * focal).astype(np.int32)
            inb = (sx >= 1) & (sx < w - 1) & (sy >= 1) & (sy < h - 1)
            sx, sy, z_v = sx[inb], sy[inb], z_v[inb]
            pts_v, nrms_v, dyn_v = pts_v[inb], nrms_v[inb], dyn_v[inb]

            # 1) 视线向量与 Fresnel 轮廓高光 (1 - |n·v|)^2
            view_vec = cam_pos[None, :] - pts_v
            view_vec /= np.maximum(np.linalg.norm(view_vec, axis=-1, keepdims=True), 1e-6)
            ndotv = np.clip(np.abs(np.sum(nrms_v * view_vec, axis=-1)), 0.0, 1.0)
            fresnel_rim = (1.0 - ndotv) ** 2.0

            # 2) 法向朗伯光照（顶斜前主灯 + 轻侧补光，立体面分隔清晰）
            ldir1 = np.array([-0.42, -0.28, 0.86], dtype=np.float32)
            ldir1 /= np.linalg.norm(ldir1)
            ldir2 = np.array([0.55, 0.55, 0.62], dtype=np.float32)
            ldir2 /= np.linalg.norm(ldir2)
            ndotl = 0.72 * np.clip(np.sum(nrms_v * ldir1[None, :], axis=-1), 0.0, 1.0) + \
                    0.28 * np.clip(np.sum(nrms_v * ldir2[None, :], axis=-1), 0.0, 1.0)

            # 3) 高度分层 + 竖直面强调：桌面保持克制深绿，立体物体与垂直立面明亮清晰
            pz = pts_v[:, 2]
            is_desk = (np.abs(pz) < 0.025) & (nrms_v[:, 2] > 0.78) & (~dyn_v)
            h_norm = np.clip((pz - 0.015) / 0.36, 0.0, 1.0)
            verticality = 1.0 - np.clip(np.abs(nrms_v[:, 2]), 0.0, 1.0)

            lum_desk = 0.22 + 0.14 * ndotl
            lum_obj = 0.36 + 0.34 * ndotl + 0.24 * h_norm + 0.22 * verticality + 0.48 * fresnel_rim
            lum_dyn = 0.52 + 0.30 * ndotl + 0.52 * fresnel_rim
            lum = np.where(dyn_v, lum_dyn, np.where(is_desk, lum_desk, lum_obj))
            lum = np.clip(lum, 0.10, 1.15)

            # 4) 100% 纯绿色战术调色板（静态=雷达翡翠绿，动态=高能薄荷绿）
            r_ch = np.where(dyn_v, 0.30 * lum, np.where(is_desk, 0.04 * lum, 0.08 * lum + 0.14 * fresnel_rim))
            g_ch = np.where(dyn_v, np.minimum(1.0, 1.08 * lum), np.minimum(1.0, 0.96 * lum))
            b_ch = np.where(dyn_v, 0.58 * lum, np.where(is_desk, 0.14 * lum, 0.28 * lum + 0.18 * fresnel_rim))
            rgb = np.clip(np.stack([r_ch, g_ch, b_ch], axis=-1), 0.0, 1.0).astype(np.float32)

            order = np.argsort(-z_v)
            sx, sy, z_v, rgb = sx[order], sy[order], z_v[order], rgb[order]
            dyn_s = dyn_v[order]
            elev_s = ~is_desk[order]

            # 紧凑 9 点钻石面片（~3.5px 直径，轮廓比大圆盘更锐利）
            for oy, ox in ((0, 0), (0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                py = np.clip(sy + oy, 0, h - 1)
                px = np.clip(sx + ox, 0, w - 1)
                z_buf[py, px] = z_v
                c_buf[py, px] = rgb
                is_dyn_buf[py, px] = dyn_s
                is_elev_buf[py, px] = elev_s

    # 屏幕空间双半径 Eye-Dome Lighting (EDL) + 亮绿前景轮廓线提取
    has_pt = z_buf < 1e3
    log_z = np.where(has_pt, np.log2(np.maximum(z_buf, 0.05)), 10.0)
    resp_dark = np.zeros((h, w), dtype=np.float32)
    resp_rim = np.zeros((h, w), dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        nbr = np.roll(np.roll(log_z, dy, axis=0), dx, axis=1)
        wt = 1.0 if abs(dx) + abs(dy) == 1 else 0.55
        # 正差 (我比邻居更远) -> 被前景遮挡的暗描边 (标准 EDL)
        resp_dark += np.maximum(0.0, log_z - nbr) * wt
        # 负差 (我比邻居更近) -> 前景物体的外部轮廓边缘，叠加亮绿霓虹轮廓描边!
        nbr_valid = np.roll(np.roll(has_pt, dy, axis=0), dx, axis=1)
        jump = np.where(nbr_valid, np.maximum(0.0, nbr - log_z), 0.14)
        resp_rim += np.minimum(jump, 0.35) * wt

    edl = np.exp(-float(edl_gain) * resp_dark)
    c_shaded = np.where(has_pt[..., None], c_buf * edl[..., None], c_buf)

    # 仅在有效物体前景轮廓边缘叠加激光亮绿高光 (#44FF92 / 动态 #9BFFCC)
    edge_strength = np.clip((resp_rim - 0.14) * 2.2, 0.0, 1.0) * float(contour_boost)
    edge_strength = np.where(has_pt & is_elev_buf, edge_strength, edge_strength * 0.20)
    edge_rgb = np.zeros_like(c_shaded)
    edge_rgb[..., 0] = np.where(is_dyn_buf, 0.45, 0.18)
    edge_rgb[..., 1] = 1.00
    edge_rgb[..., 2] = np.where(is_dyn_buf, 0.72, 0.48)
    c_final = np.clip(c_shaded + edge_strength[..., None] * edge_rgb, 0.0, 1.0)

    out_bgr = cv2.cvtColor((c_final * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # A* 导航路径（柔和亮绿白折线，契合全绿战术观感）
    if path and len(path) >= 2:
        ppx = []
        for wx, wy in path:
            pr = _PROJ.project_world_point(wx, wy, 0.015, cam, float(w), float(h))
            if pr is not None:
                ppx.append((int(round(pr[0])), int(round(pr[1]))))
        for i in range(len(ppx) - 1):
            cv2.line(out_bgr, ppx[i], ppx[i + 1], (155, 255, 195), 2, cv2.LINE_AA)
        if ppx:
            cv2.circle(out_bgr, ppx[-1], 6, (90, 255, 140), -1, cv2.LINE_AA)
            cv2.circle(out_bgr, ppx[-1], 9, (210, 255, 225), 1, cv2.LINE_AA)

    if click_marker is not None:
        pr = _PROJ.project_world_point(click_marker[0], click_marker[1], 0.02, cam, float(w), float(h))
        if pr is not None:
            cv2.drawMarker(out_bgr, (int(pr[0]), int(pr[1])), click_marker[2], cv2.MARKER_TILTED_CROSS, 16, 2, cv2.LINE_AA)

    # 3D 机器人位置与视野扇区（绿色线框图标）
    rx0, ry0, ryaw = robot_pose
    view_yaw = ryaw + head_yaw_deg
    pr0 = _PROJ.project_world_point(rx0, ry0, 0.0, cam, float(w), float(h))
    pr_top = _PROJ.project_world_point(rx0, ry0, 0.09, cam, float(w), float(h))
    pr_nose = _PROJ.project_world_point(
        rx0 + 0.16 * math.cos(math.radians(view_yaw)),
        ry0 + 0.16 * math.sin(math.radians(view_yaw)),
        0.04,
        cam,
        float(w),
        float(h),
    )
    if pr0 is not None and pr_top is not None:
        cv2.circle(out_bgr, (int(pr0[0]), int(pr0[1])), 9, (70, 220, 110), 2, cv2.LINE_AA)
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(pr_top[0]), int(pr_top[1])), (110, 255, 155), 3, cv2.LINE_AA)
        cv2.circle(out_bgr, (int(pr_top[0]), int(pr_top[1])), 5, (180, 255, 210), -1, cv2.LINE_AA)
    if pr_top is not None and pr_nose is not None:
        cv2.arrowedLine(out_bgr, (int(pr_top[0]), int(pr_top[1])), (int(pr_nose[0]), int(pr_nose[1])), (150, 255, 185), 2, cv2.LINE_AA, tipLength=0.35)
    p_fov_l = _PROJ.project_world_point(rx0 + 0.75 * math.cos(math.radians(view_yaw + 36.0)), ry0 + 0.75 * math.sin(math.radians(view_yaw + 36.0)), 0.0, cam, float(w), float(h))
    p_fov_r = _PROJ.project_world_point(rx0 + 0.75 * math.cos(math.radians(view_yaw - 36.0)), ry0 + 0.75 * math.sin(math.radians(view_yaw - 36.0)), 0.0, cam, float(w), float(h))
    if pr0 is not None and p_fov_l is not None and p_fov_r is not None:
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(p_fov_l[0]), int(p_fov_l[1])), (45, 165, 85), 1, cv2.LINE_AA)
        cv2.line(out_bgr, (int(pr0[0]), int(pr0[1])), (int(p_fov_r[0]), int(p_fov_r[1])), (45, 165, 85), 1, cv2.LINE_AA)

    return out_bgr


# ---------------------------------------------------------------------------
# 6. 端到端便捷封装：单入口接入 .tflite 学生模型 -> 鲁棒双层轮廓 SLAM
# ---------------------------------------------------------------------------


class RobustContourSlamPipeline:
    """端到端纯端侧 `.tflite` 推理 + 开源 SLAM 双层轮廓建图 + 5cm 可达栅格。"""

    def __init__(
        self,
        tflite_path: pathlib.Path | str,
        fov_h_deg: float = 72.0,
        enable_icp: bool = True,
        enable_ever_free: bool = True,
        enable_disturbance_gate: bool = True,
    ) -> None:
        self.tflite_path = pathlib.Path(tflite_path)
        self.fov_h_deg = float(fov_h_deg)
        self.map = RobustDualLayerContourMap(
            vox_size=0.0085,
            fov_h_deg=self.fov_h_deg,
            confirm_hits=2,
            enable_icp=enable_icp,
            enable_ever_free=enable_ever_free,
            enable_disturbance_gate=enable_disturbance_gate,
        )
        self.grid = TraversabilityGrid2D()
        self.r_level: np.ndarray | None = None
        self.z_bias = 0.0
        self.last_pitch_deg = 0.0

    def reset(self) -> None:
        self.map.reset()
        self.grid = TraversabilityGrid2D()
        self.r_level = None
        self.z_bias = 0.0
        self.last_pitch_deg = 0.0

    def process_frame(
        self,
        bgr: np.ndarray,
        yaw_deg: float = 0.0,
        cam_x: float = 0.0,
        cam_y: float = 0.0,
        imu_pitch_deg: float | None = None,
        imu_roll_deg: float = 0.0,
        injected_pitch_err_deg: float = 0.0,
        injected_yaw_err_deg: float = 0.0,
        out_hw: tuple[int, int] = (192, 192),
    ) -> dict[str, Any]:
        """运行单帧学生推理与双层鲁棒建图（支持可选注入姿态扰动用于消融压测）。"""
        splats = run_student_tflite_frame(self.tflite_path, bgr)
        r_lvl, z_b, est_pitch = solve_table_plane_from_student(
            splats[..., 0].astype(np.float32),
            splats[..., 1].astype(np.float32),
            fov_h_deg=self.fov_h_deg,
            prev_r_level=self.r_level,
            prev_z_bias=self.z_bias,
        )
        self.r_level = r_lvl
        self.z_bias = z_b
        self.last_pitch_deg = est_pitch

        # 若为扰动压测，向初始位姿注入平衡俯仰摆动或舵机偏航误差
        if abs(injected_pitch_err_deg) > 1e-4:
            th = math.radians(float(injected_pitch_err_deg))
            r_err = np.array(
                [
                    [math.cos(th), 0.0, math.sin(th)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(th), 0.0, math.cos(th)],
                ],
                dtype=np.float32,
            )
            r_used = (r_err @ r_lvl).astype(np.float32)
        else:
            r_used = r_lvl

        yaw_used = float(yaw_deg + injected_yaw_err_deg)
        recon = reconstruct_dense_frame_3d(
            bgr=bgr,
            splats_96=splats,
            r_level=r_used,
            z_bias=z_b,
            fov_h_deg=self.fov_h_deg,
            yaw_deg=yaw_used,
            cam_x=cam_x,
            cam_y=cam_y,
            out_hw=out_hw,
        )

        pitch_for_gate = float(imu_pitch_deg) if imu_pitch_deg is not None else float(est_pitch + injected_pitch_err_deg)
        status = self.map.update(
            bgr=bgr,
            pts=recon["pts"],
            nrms=recon["nrms"],
            depth_hr=recon["depth_hr"],
            r_level=r_used,
            yaw_deg=yaw_used,
            pitch_deg=pitch_for_gate,
            roll_deg=imu_roll_deg,
            cam_x=cam_x,
            cam_y=cam_y,
            z_bias=z_b,
        )

        comb = self.map.get_combined_for_traversability()
        self.grid.update(comb, robot_x=cam_x, robot_y=cam_y, robot_yaw_deg=yaw_deg)
        return {
            "splats": splats,
            "recon": recon,
            "status": status,
        }
