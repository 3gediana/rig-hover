# -*- coding: utf-8 -*-
"""hd_dynamic_contour_slam.py — 统一 2.0 mm 超密纯绿色轮廓建图 + Dynablox 零残影动态分离引擎。

同时支持两套可插拔 3D 几何前端（统一物理尺度 `H_cam = 0.135 m / 13.5 cm`, `FOV_H = 65.0°`）：
1. **`HoverGeoNet240Frontend`**（本地化轻量专用网络：FP16 ~3.87 MB，RTX 5060 ~7 ms/帧，原生 240×240 不缩图）；
2. **`MoGe2Frontend`**（微软开源大模型参考前端：419 MB，RTX 5060 ~163 ms/帧）。

核心抗残影与抗 1.16 Hz 自平衡摇摆机制（彻底解决 v6 中 `cb8455` 挥手多帧拖影问题）：
- **Dynablox Occupied-Protected `Ever-Free` 3D 空域网格**：进入历史空域的手/人体即使悬停静止也直通实时动态层；
- **相机摇摆补偿帧差动态门**：利用 `phaseCorrelate` 抵消 1.16 Hz 自平衡全局像素垂移后提取前景真实运动，
  第 1 帧挥手即可立刻分离，无需等待多帧预热；
- **UFOMap 暂态鬼影光线雕刻 (Tentative Free-Space Carving)**：若第 0 帧恰好被移动手遮挡而短暂写入
  低权重暂态面元 (`wt < confirm_wt`)，一旦后续干净视线穿透该处并击中后方桌面/物体，立即擦除暂态残影，
  同时绝对保护已确认静态面元（鼠标、键盘、水杯、桌角）；
- **Unblur-SLAM 冲击拦截 (`7.5°` 阈值) + 回转零速峰值陀螺加权 + KISS-ICP 点到面精配准**；
- **全绿色激光双层渲染 + 5 cm 通行栅格 A* 路径规划**（严格零 RGB 彩图）。
"""
from __future__ import annotations

import math
import pathlib
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dense_scene_pipeline import TraversabilityGrid2D  # noqa: E402
from hd_moge_contour_engine import (  # noqa: E402
    REAL_FOV_H_DEG,
    REAL_H_CAM_M,
    MoGe2DesktopReconstructor,
    UltraDenseSurfelCloud,
    level_and_scale_desktop_cloud,
    render_hd_laser_green_cloud,
)
from hover_geonet_240 import (  # noqa: E402
    GEO_H,
    GEO_IN_CH,
    GEO_W,
    HoverGeoNet240,
    build_robot_physics_input,
    create_hover_geonet,
    decode_geonet_to_world_cloud,
)
from mono_depth_model import camera_rotation_matrix, make_camera_ray_grid  # noqa: E402
from neural_gaussian_splat_model import OrbitCamera3D, PersistentGaussianMemoryMap  # noqa: E402
from robust_contour_slam import (  # noqa: E402
    DisturbanceQualityGate,
    EverFreeVoxelGrid,
    KissIcpStaticAligner,
)

_PROJ = PersistentGaussianMemoryMap(max_splats=8)


# ---------------------------------------------------------------------------
# 1. 可插拔 3D 前端：`HoverGeoNet240Frontend` 与 `MoGe2Frontend`
# ---------------------------------------------------------------------------


class HoverGeoNet240Frontend:
    """本地化 3.87 MB FP16 物理条件几何前端（支持真实 IMU 与无 IMU 自动配平双模式）。"""

    def __init__(
        self,
        ckpt_path: pathlib.Path | str,
        use_fp16: bool = True,
        h_cam_m: float = REAL_H_CAM_M,
        fov_h_deg: float = REAL_FOV_H_DEG,
    ) -> None:
        self.ckpt_path = pathlib.Path(ckpt_path)
        self.h_cam_m = float(h_cam_m)
        self.fov_h_deg = float(fov_h_deg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")

        blob = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        arch = str(blob.get("arch", "mbv3"))
        base = int(blob.get("base", 32 if arch == "vit_b1" else 24))
        in_ch = int(blob.get("in_ch", GEO_IN_CH))
        self.arch = arch
        # 相机射线栅格逐帧重算是纯浪费（FOV/分辨率固定）：缓存一次终身用
        self._rays_cam: np.ndarray | None = None
        self.model = create_hover_geonet(arch=arch, in_ch=in_ch, base=base, pretrained_backbone=False).to(self.device).eval()
        sd = blob["state_dict_fp16"] if "state_dict_fp16" in blob else blob["state_dict"]
        self.model.load_state_dict({k: v.to(self.device) for k, v in sd.items()}, strict=False)
        if self.use_fp16:
            self.model = self.model.half()

        # 预热 CUDA 内核
        x0 = torch.zeros(1, in_ch, GEO_H, GEO_W, device=self.device, dtype=(torch.float16 if self.use_fp16 else torch.float32))
        with torch.no_grad():
            _ = self.model(x0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def infer_frame(
        self,
        bgr_cur: np.ndarray,
        bgr_prev: np.ndarray | None = None,
        pitch_deg: float = 3.0,
        roll_deg: float = 0.0,
        auto_level: bool = False,
        valid_thresh: float = 0.45,
        cam_cloud: bool = True,
    ) -> dict[str, Any]:
        x7, rays_in, inv_d_prior = build_robot_physics_input(
            bgr_cur,
            bgr_prev,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            fov_h_deg=self.fov_h_deg,
            h_cam_m=self.h_cam_m,
        )
        xt = torch.from_numpy(x7).unsqueeze(0).to(self.device)
        if self.use_fp16:
            xt = xt.half()

        t0 = time.perf_counter()
        out5 = self.model(xt)[0].float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dec = decode_geonet_to_world_cloud(
            out5,
            rays_in,
            inv_d_prior,
            h_cam_m=self.h_cam_m,
            valid_thresh=valid_thresh,
            auto_level=auto_level,
        )
        # 同时提供相机系度量点云供视觉-几何自定位 (RANSAC-Kabsch + KISS-ICP)
        if cam_cloud:
            if self._rays_cam is None:
                self._rays_cam = make_camera_ray_grid(GEO_H, GEO_W, fov_h_deg=self.fov_h_deg, pitch_deg=0.0, roll_deg=0.0)
            pts_cam = (self._rays_cam * dec["depth_m"][..., None]).astype(np.float32)
            r_in = camera_rotation_matrix(pitch_deg=pitch_deg, roll_deg=roll_deg)
            nrm_cam = (dec["nrm_grid"] @ r_in).astype(np.float32)
            nrm_cam /= np.maximum(np.linalg.norm(nrm_cam, axis=-1, keepdims=True), 1e-6)
            dec["pts_cam"] = pts_cam
            dec["nrm_cam"] = nrm_cam
        else:
            dec["pts_cam"] = dec["nrm_cam"] = None
        dec["infer_ms"] = infer_ms
        dec["scale"] = 1.0
        return dec


class MoGe2Frontend:
    """微软开源 MoGe-2-ViT-B-Normal 参考前端（严格对齐 H_cam=0.135m 与 FOV=65.0°）。"""

    def __init__(self, infer_hw: int = 480, h_cam_m: float = REAL_H_CAM_M, fov_h_deg: float = REAL_FOV_H_DEG) -> None:
        self.h_cam_m = float(h_cam_m)
        self.rec = MoGe2DesktopReconstructor(infer_hw=infer_hw, fov_x_deg=fov_h_deg)

    def infer_frame(
        self,
        bgr_cur: np.ndarray,
        bgr_prev: np.ndarray | None = None,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        auto_level: bool = True,
        valid_thresh: float = 0.45,
    ) -> dict[str, Any]:
        raw = self.rec.infer_raw(bgr_cur)
        lvl = level_and_scale_desktop_cloud(raw["pts_cam"], raw["nrm_cam"], raw["mask"], h_cam_m=self.h_cam_m)
        lvl["pts_cam"] = (raw["pts_cam"] * lvl["scale"]).astype(np.float32)
        lvl["nrm_cam"] = raw["nrm_cam"]
        lvl["infer_ms"] = float(raw["infer_ms"])
        return lvl


# ---------------------------------------------------------------------------
# 2. 相机摇摆补偿动态前景检测（抵消 1.16 Hz 自平衡微摇，精准框出挥动人手）
# ---------------------------------------------------------------------------


def compute_compensated_motion_mask(
    bgr_cur: np.ndarray,
    bgr_prev: np.ndarray | None,
    target_hw: tuple[int, int],
    diff_thresh: float = 20.0,
    disoccluded_mask: np.ndarray | None = None,
) -> np.ndarray:
    """先用相位相关抵消双轮自平衡导致的全局画面小幅平移，再提取真实局部动态前景掩码。"""
    th, tw = target_hw
    if bgr_prev is None:
        return np.zeros((th, tw), dtype=bool)

    gc = cv2.cvtColor(bgr_cur, cv2.COLOR_BGR2GRAY) if bgr_cur.ndim == 3 else bgr_cur
    gp = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2GRAY) if bgr_prev.ndim == 3 else bgr_prev
    if gc.shape != (th, tw):
        gc = cv2.resize(gc, (tw, th), interpolation=cv2.INTER_AREA)
    if gp.shape != (th, tw):
        gp = cv2.resize(gp, (tw, th), interpolation=cv2.INTER_AREA)

    # 仅在画面下边缘纯桌面区域估计相机 1.16Hz 摇摆引起的微小全局平移，防止大幅挥动手臂带偏相位相关!
    h_lo = int(th * 0.70)
    strip_c = gc[h_lo:, :].astype(np.float32)
    strip_p = gp[h_lo:, :].astype(np.float32)
    win = cv2.createHanningWindow((tw, strip_c.shape[0]), cv2.CV_32F)
    shift, resp = cv2.phaseCorrelate(strip_p, strip_c, window=win)
    dx, dy = float(np.clip(shift[0], -5.0, 5.0)), float(np.clip(shift[1], -5.0, 5.0))
    if resp > 0.15 and (abs(dx) > 0.4 or abs(dy) > 0.4):
        m_aff = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        gp_warp = cv2.warpAffine(gp, m_aff, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # 仅当补偿确实降低了全局中位残差时才采用
        if float(np.median(cv2.absdiff(gc, gp_warp))) < float(np.median(cv2.absdiff(gc, gp))) * 0.92:
            gp_use = gp_warp
        else:
            gp_use = gp
    else:
        gp_use = gp

    # 轻度高斯平滑压掉 GC0308 颗粒噪点
    diff = cv2.absdiff(cv2.GaussianBlur(gc, (5, 5), 1.0), cv2.GaussianBlur(gp_use, (5, 5), 1.0))
    raw_mask = (diff.astype(np.float32) >= float(diff_thresh)).astype(np.uint8)
    # 膨胀前先剔除去遮挡（物体离开露出更远背景墙）像素，彻底杜绝离开后留下 6px 光晕假阳性!
    if disoccluded_mask is not None and disoccluded_mask.shape == raw_mask.shape:
        raw_mask[disoccluded_mask] = 0

    # 形态学开闭运算：剔除散点噪点，连通整只挥动的手臂
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    clean = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
    dilated = cv2.dilate(clean, k_dil, iterations=1)
    if disoccluded_mask is not None and disoccluded_mask.shape == dilated.shape:
        dilated[disoccluded_mask] = 0
    # 若运动面积过小（< 0.3% 画面），视为噪声忽略
    if int(np.sum(dilated > 0)) < int(th * tw * 0.003):
        return np.zeros((th, tw), dtype=bool)
    return dilated > 0


# ---------------------------------------------------------------------------
# 3. 2.0 mm 超密静态地图 + Dynablox Ever-Free + 暂态鬼影雕刻引擎
# ---------------------------------------------------------------------------


class HdDynamicContourSlamEngine:
    """结合 2.0 mm 超密面元、Dynablox Occupied-Protected Ever-Free 与暂态鬼影雕刻的双层建图引擎。"""

    def __init__(
        self,
        vox_size_m: float = 0.0020,
        ever_free_res_m: float = 0.030,
        confirm_wt: float = 2.6,
        h_cam_m: float = REAL_H_CAM_M,
        obs_radius_m: float = 0.095,
        static_decim: int = 1,
        async_static: bool = False,
    ) -> None:
        self.vox_size = float(vox_size_m)
        self.confirm_wt = float(confirm_wt)
        self.h_cam_m = float(h_cam_m)
        # 静态融合降频：fuse/ef_rays/ef_protect/carve 是 O(图尺寸) 的重活
        # （实测占单帧 ~85%），静态世界低速变化容忍每 N 帧更新一次；
        # 动态分离与 ICP 对齐仍每帧全速跑。1 = 不降频（保持原行为）。
        self.static_decim = max(1, int(static_decim))
        self._pf_idx = 0
        # 静态建图工人线程：fuse/carve 彻底离开控制关键路径（消灭每 N 帧 ~200ms 尖峰）。
        # busy 时丢帧不排队（"最新优先"语义）；点云经 _v 原子视图发表，读者无撕裂。
        # 默认关：测试/离线回放要确定性语义（异步会让 ever-free 确认不足、动态路由失真）。
        self._static_pool = None
        self._static_fut = None
        if async_static:
            from concurrent.futures import ThreadPoolExecutor
            self._static_pool = ThreadPoolExecutor(1, thread_name_prefix="staticmap")

        self.static_cloud = UltraDenseSurfelCloud(vox_size_m=self.vox_size)
        self.ever_free = EverFreeVoxelGrid(res_m=ever_free_res_m, confirm_hits=3)
        self.quality_gate = DisturbanceQualityGate(max_dpitch_deg=7.5, max_droll_deg=6.5, max_dyaw_deg=12.0)

        # 实时动态层（仅存当前帧动态手/人，绝不残影拖尾）
        self.dyn_pts = np.zeros((0, 3), dtype=np.float32)
        self.dyn_nrms = np.zeros((0, 3), dtype=np.float32)
        self.dyn_crease = np.zeros((0,), dtype=np.float32)

        self.trav_grid = TraversabilityGrid2D(res=0.05, obs_radius_m=float(obs_radius_m))
        self.prev_bgr: np.ndarray | None = None
        self.prev_depth: np.ndarray | None = None
        self.stats_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.static_cloud = UltraDenseSurfelCloud(vox_size_m=self.vox_size)
        self.ever_free.reset()
        self.quality_gate.reset()
        self.dyn_pts = np.zeros((0, 3), dtype=np.float32)
        self.dyn_nrms = np.zeros((0, 3), dtype=np.float32)
        self.dyn_crease = np.zeros((0,), dtype=np.float32)
        self.prev_bgr = None
        self.prev_depth = None
        self.stats_log.clear()

    def _carve_tentative_ghosts(self) -> int:
        """UFOMap/Dynablox 暂态鬼影雕刻：若暂态面元 (`wt < confirm_wt` 且 `Z >= 3.2cm`) 落入已确认 `ever_free` 空域，立即清除！"""
        if len(self.static_cloud.pos) == 0:
            return 0
        is_tentative = (self.static_cloud.wt < self.confirm_wt) & (self.static_cloud.pos[:, 2] >= 0.032)
        if not np.any(is_tentative):
            return 0
        tent_idx = np.flatnonzero(is_tentative)
        in_free = self.ever_free.query_ever_free(self.static_cloud.pos[tent_idx], expand_clusters=False)
        if not np.any(in_free):
            return 0
        drop = np.zeros(len(self.static_cloud.pos), dtype=bool)
        drop[tent_idx[in_free]] = True
        keep = ~drop
        p_, n_, c_, w_ = self.static_cloud.get_view()
        self.static_cloud._set_v(pos=p_[keep], nrm=n_[keep], crease=c_[keep], wt=w_[keep])
        self.static_cloud._sorted_keys = None  # 掩码重排使缓存失效
        return int(np.sum(drop))

    def process_frame(
        self,
        bgr_cur: np.ndarray,
        pts_grid: np.ndarray,
        nrm_grid: np.ndarray,
        crease_grid: np.ndarray,
        valid_grid: np.ndarray,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        align: bool = True,
    ) -> dict[str, Any]:
        """处理单帧 3D 点图：扰动门控 -> 摇摆补偿运动+Ever-Free 双判动态分离 -> 静态 2mm 精修 -> 暂态残影雕刻。"""
        _t = time.perf_counter
        _st: dict[str, float] = {}
        _t0 = _t()
        gate = self.quality_gate.evaluate_pre_icp(bgr_cur, pitch_deg=pitch_deg, roll_deg=roll_deg, yaw_deg=robot_yaw_deg)
        _st["gate"] = (_t() - _t0) * 1e3

        # 将局部机身系变换到全局世界系 (robot_x, robot_y, robot_yaw_deg)
        rad = math.radians(float(robot_yaw_deg))
        cy, sy = math.cos(rad), math.sin(rad)
        r_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        pts_loc = pts_grid[valid_grid].astype(np.float32)
        nrm_loc = nrm_grid[valid_grid].astype(np.float32)
        cr_loc = crease_grid[valid_grid].astype(np.float32)

        pts_w = (pts_loc @ r_yaw.T) + np.array([robot_x, robot_y, 0.0], dtype=np.float32)[None, :]
        nrm_w = (nrm_loc @ r_yaw.T).astype(np.float32)
        cam_pos = np.array([robot_x, robot_y, self.h_cam_m], dtype=np.float32)

        # 0) 严格遵循 SLAM 正确时序：先用 KISS-ICP (Geman-McClure 鲁棒核天然抗 25~40% 动态外点)
        #    把本帧对齐到已有静态骨架，消掉 1.16Hz 摇摆造成的 2.4°/38mm 瞬时偏移，再做动态检测与 Ever-Free 查询！
        rot_corr = 0.0
        trans_corr_mm = 0.0
        _t0 = _t()
        mp_pos, mp_nrm = self.static_cloud.get_view()[:2]
        if align and (not gate.is_disturbed) and len(mp_pos) > 5000 and len(pts_w) > 400:
            res_icp = self.static_cloud.aligner.align_to_static(
                pts_w, nrm_w, mp_pos, mp_nrm, cam_pos, pre_q_ref=self.static_cloud.get_icp_ref()
            )
            if res_icp["converged"] and res_icp["inlier_ratio"] > 0.42:
                dR, dt = res_icp["dR"], res_icp["dt"]
                pts_w = ((pts_w - cam_pos[None, :]) @ dR.T + cam_pos[None, :] + dt[None, :]).astype(np.float32)
                nrm_w = (nrm_w @ dR.T).astype(np.float32)
                rot_corr = float(res_icp["rot_deg"])
                trans_corr_mm = float(res_icp["trans_m"] * 1000.0)
        _st["icp"] = (_t() - _t0) * 1e3
        _t0 = _t()

        # 用配准后的世界 X 距离更新对齐深度网格（彻底消除 1.16Hz 摇摆引起的假深度跳变）
        cur_depth = np.zeros(valid_grid.shape, dtype=np.float32)
        cur_depth[valid_grid] = pts_w[:, 0]

        # 1) 双源动态掩码（基于已配准世界点云！）：
        #    (a) Dynablox Occupied-Protected Ever-Free（精准捕获进入历史空域的手/人体，即便悬停静止！）
        #    (b) 配准后 3D 深度跳变验证的前景运动门：仅当“亮度显著变化”且“配准后前景深度跳变 > 6.5cm”
        #        且“距离 < 0.85m”时判为动态前景！空桌面 1.16Hz 摇摆零误触发！
        disoccluded = (
            (cur_depth > (self.prev_depth + 0.045))
            if (self.prev_depth is not None and self.prev_depth.shape == cur_depth.shape)
            else None
        )
        mot_2d = compute_compensated_motion_mask(
            bgr_cur, self.prev_bgr, valid_grid.shape, diff_thresh=25.0, disoccluded_mask=disoccluded
        )
        if self.prev_depth is not None and self.prev_depth.shape == cur_depth.shape:
            both_v = (cur_depth > 0.12) & (self.prev_depth > 0.12)
            depth_moved = both_v & (np.abs(cur_depth - self.prev_depth) > 0.065) & (cur_depth < 0.85)
            k_dm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            depth_moved_dil = cv2.dilate(depth_moved.astype(np.uint8), k_dm, iterations=1) > 0
            mot_2d = mot_2d & depth_moved_dil & (cur_depth < 0.85)
        else:
            mot_2d = np.zeros_like(valid_grid, dtype=bool)
        mot_flat = mot_2d[valid_grid] & (pts_w[:, 2] >= 0.032)
        # 仅在交互桌面近景范围 (X < 0.90m) 触发 Ever-Free 动态分离
        ef_flat = self.ever_free.query_ever_free(pts_w, expand_clusters=True) & (pts_w[:, 0] < 0.90)
        is_dyn = ef_flat | mot_flat

        # 若动态点存在显著种子，做 3D 团簇邻域扩张（把整只手底缘/手指完整分离，不留碎片！）
        if int(np.sum(is_dyn)) >= 40:
            dyn_seed = pts_w[is_dyn]
            # 用 3cm 体素快速膨胀 1 步覆盖手边缘像素
            s_idx = np.floor((dyn_seed - self.ever_free.origin[None, :]) / self.ever_free.res_m).astype(np.int32)
            inb_s = (
                (s_idx[:, 0] >= 0) & (s_idx[:, 0] < self.ever_free.nx)
                & (s_idx[:, 1] >= 0) & (s_idx[:, 1] < self.ever_free.ny)
                & (s_idx[:, 2] >= 0) & (s_idx[:, 2] < self.ever_free.nz)
            )
            s_idx = s_idx[inb_s]
            if len(s_idx) > 0:
                ny_ef, nz_ef = self.ever_free.ny, self.ever_free.nz
                s_lin = np.unique((s_idx[:, 0] * ny_ef + s_idx[:, 1]) * nz_ef + s_idx[:, 2])
                ux = s_lin // (ny_ef * nz_ef)
                rem = s_lin % (ny_ef * nz_ef)
                uy = rem // nz_ef
                uz = rem % nz_ef
                vol = np.zeros((self.ever_free.nx, self.ever_free.ny, self.ever_free.nz), dtype=bool)
                for dx in (-1, 0, 1):
                    xs = np.clip(ux + dx, 0, self.ever_free.nx - 1)
                    for dy in (-1, 0, 1):
                        ys = np.clip(uy + dy, 0, self.ever_free.ny - 1)
                        for dz in (-1, 0, 1):
                            zs = np.clip(uz + dz, 0, self.ever_free.nz - 1)
                            vol[xs, ys, zs] = True
                cand = (pts_w[:, 2] >= 0.034) & (~is_dyn)
                if np.any(cand):
                    ci = np.floor((pts_w[cand] - self.ever_free.origin[None, :]) / self.ever_free.res_m).astype(np.int32)
                    inb_c = (
                        (ci[:, 0] >= 0) & (ci[:, 0] < self.ever_free.nx)
                        & (ci[:, 1] >= 0) & (ci[:, 1] < self.ever_free.ny)
                        & (ci[:, 2] >= 0) & (ci[:, 2] < self.ever_free.nz)
                    )
                    hit_c = np.zeros(int(np.sum(cand)), dtype=bool)
                    if np.any(inb_c):
                        vc = ci[inb_c]
                        hit_c[inb_c] = vol[vc[:, 0], vc[:, 1], vc[:, 2]]
                    is_dyn[cand] = hit_c

        self.prev_bgr = bgr_cur.copy()
        self.prev_depth = cur_depth.copy()
        _st["dynmask"] = (_t() - _t0) * 1e3
        _t0 = _t()

        # 更新实时动态层（仅当前帧，完全无多帧残影！）
        self.dyn_pts = pts_w[is_dyn]
        self.dyn_nrms = nrm_w[is_dyn]
        self.dyn_crease = cr_loc[is_dyn]

        st_pts = pts_w[~is_dyn]
        st_nrms = nrm_w[~is_dyn]
        st_cr = cr_loc[~is_dyn]

        fuse_info = {"total_pts": int(len(self.static_cloud.pos)), "rot_corr_deg": rot_corr, "trans_corr_mm": trans_corr_mm}
        carved = 0
        self._pf_idx += 1
        # 收回上一个已完成的静态任务结果（结果只是日志数据，晚到一帧无碍）
        if self._static_fut is not None and self._static_fut.done():
            try:
                fuse_info, carved = self._static_fut.result()
            except Exception as exc:
                fuse_info["static_err"] = str(exc)
            self._static_fut = None
        run_static = (self._pf_idx % self.static_decim == 0)
        if not gate.is_disturbed and len(st_pts) > 200 and run_static:
            if self._static_pool is None:
                _ts = _t()
                fuse_info, carved = self._static_task(st_pts, st_nrms, st_cr, cam_pos.copy(), rot_corr, trans_corr_mm)
                _st["static_sync"] = (_t() - _ts) * 1e3
            elif self._static_fut is None:
                # 静态块整体进工人线程：控制关键路径零尖峰；busy 则丢帧（地图下轮补）
                self._static_fut = self._static_pool.submit(
                    self._static_task, st_pts, st_nrms, st_cr, cam_pos.copy(), rot_corr, trans_corr_mm
                )
                _st["static_submit"] = (_t() - _t0) * 1e3
            else:
                fuse_info["static_async"] = "busy_skip"

        rec = {
            "disturbed": bool(gate.is_disturbed),
            "reason": gate.reason,
            "n_static_frame": int(len(st_pts)),
            "n_dynamic_frame": int(len(self.dyn_pts)),
            "n_static_map": int(len(self.static_cloud.pos)),
            "carved_ghosts": int(carved),
            "rot_corr_deg": float(fuse_info["rot_corr_deg"]),
            "trans_corr_mm": float(fuse_info["trans_corr_mm"]),
            "slam_stage_ms": {k: round(v, 2) for k, v in _st.items()},
        }
        self.stats_log.append(rec)
        return rec

    def _static_task(
        self,
        st_pts: np.ndarray,
        st_nrms: np.ndarray,
        st_cr: np.ndarray,
        cam_pos: np.ndarray,
        rot_corr: float,
        trans_corr_mm: float,
    ) -> tuple[dict, int]:
        """工人线程执行的静态建图块：融合→ever-free 视线→体素保护→鬼影雕刻。

        安全论证：输入 st_* 是布尔索引产生的新数组（主线程不再引用）；
        点云写入走 _v 原子发表；主线程对它的读（ICP/重定位/展示）永远拿一致整代；
        原位加权写是 x86 对齐 float32 原子写，读者最坏拿到"半新"值，无崩溃路径。
        """
        fuse_info = self.static_cloud.fuse_frame(st_pts, st_nrms, st_cr, cam_pos=cam_pos, align=False)
        fuse_info["rot_corr_deg"] = round(rot_corr, 3)
        fuse_info["trans_corr_mm"] = round(trans_corr_mm, 2)
        ray_stride = max(4, len(st_pts) // 2400)
        self.ever_free.integrate_clean_rays(cam_pos, st_pts, safety_margin_m=0.045, stride=ray_stride, n_samples=48)
        conf_mask = self.static_cloud.wt >= 1.5
        if np.any(conf_mask):
            self.ever_free.protect_static_voxels(self.static_cloud.pos[conf_mask])
        carved = self._carve_tentative_ghosts()
        return fuse_info, carved

    def get_confirmed_static_cloud(self, min_wt: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """取出已确认（或全部）超密静态点云。"""
        _p, _n, _c, _w = self.static_cloud.get_view()  # 原子视图：工人线程融合中也不撕裂
        if len(_p) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        if min_wt is None or self.static_cloud.frames_fused <= 2:
            return _p, _n, _c
        m = _w >= float(min_wt)
        if int(np.sum(m)) < 500:
            return _p, _n, _c
        return _p[m], _n[m], _c[m]

    def rebuild_traversability_grid(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_yaw_deg: float = 0.0,
    ) -> TraversabilityGrid2D:
        """从 2.0 mm 超密静态地图实时构建 5 cm 导航通行栅格（忽略动态挥手，不锁死路径）。"""
        pos, _, _ = self.get_confirmed_static_cloud(min_wt=1.0)
        if len(pos) > 0:
            step = max(1, len(pos) // 60000)
            self.trav_grid.update(pos[::step], robot_x=robot_x, robot_y=robot_y, robot_yaw_deg=robot_yaw_deg)
        return self.trav_grid


# ---------------------------------------------------------------------------
# 4. 双层纯绿色激光渲染器（静态 2mm 翡翠绿骨架 + 动态无残影霓虹薄荷绿前景）
# ---------------------------------------------------------------------------


def render_hd_dual_layer_green(
    static_pts: np.ndarray,
    static_nrms: np.ndarray,
    static_crease: np.ndarray,
    dyn_pts: np.ndarray | None,
    dyn_nrms: np.ndarray | None,
    cam: OrbitCamera3D,
    w: int = 800,
    h: int = 560,
    point_radius_px: int = 1,
    draw_grid: bool = True,
    target_pin_xyz: tuple[float, float, float] | None = None,
    robot_pose: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    path_xy: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """渲染纯绿色双层激光 3D 轮廓：静态地图深浅翡翠绿 + 实时动态手部高亮薄荷绿（零残影、零 RGB 彩图）。"""
    img = render_hd_laser_green_cloud(
        static_pts,
        static_nrms,
        static_crease,
        cam=cam,
        w=w,
        h=h,
        point_radius_px=point_radius_px,
        draw_grid=draw_grid,
        target_pin_xyz=None,
        robot_pose=None,
    )

    basis = _PROJ.compute_camera_basis(cam)
    focal = _PROJ.compute_focal_px(cam, float(h))
    cx, cy = w * 0.5, h * 0.5

    # 叠加当前帧无残影实时动态前景（亮薄荷绿轮廓，只占当前 1 帧！）
    if dyn_pts is not None and len(dyn_pts) > 0:
        rx = dyn_pts[:, 0] - basis[0]
        ry = dyn_pts[:, 1] - basis[1]
        rz = dyn_pts[:, 2] - basis[2]
        z_cam = rx * basis[3] + ry * basis[4] + rz * basis[5]
        vis = z_cam > 0.05
        if np.any(vis):
            z_v = z_cam[vis]
            x_c = rx[vis] * basis[6] + ry[vis] * basis[7] + rz[vis] * basis[8]
            y_c = rx[vis] * basis[9] + ry[vis] * basis[10] + rz[vis] * basis[11]
            sx = np.rint(cx + (x_c / z_v) * focal).astype(np.int32)
            sy = np.rint(cy - (y_c / z_v) * focal).astype(np.int32)
            inb = (sx >= 1) & (sx < w - 1) & (sy >= 1) & (sy < h - 1)
            sx, sy, z_v = sx[inb], sy[inb], z_v[inb]
            if dyn_nrms is not None and len(dyn_nrms) == len(dyn_pts):
                nv = dyn_nrms[vis][inb]
                ld = np.array([-0.6, -0.5, 0.62], dtype=np.float32)
                ld /= np.linalg.norm(ld)
                sh = 0.55 + 0.45 * np.clip(np.sum(nv * ld[None, :], axis=-1), 0.0, 1.0)
            else:
                sh = np.ones(len(sx), dtype=np.float32) * 0.85

            order = np.argsort(-z_v)
            sx, sy, sh = sx[order], sy[order], sh[order]
            # 霓虹薄荷绿 (BGR: B=195..235, G=255, R=120..165) — 纯绿色系内高辨识动态层
            col_bgr = np.stack([
                np.clip(190.0 * sh, 90, 235),
                np.clip(255.0 * sh, 160, 255),
                np.clip(125.0 * sh, 55, 175),
            ], axis=-1).astype(np.uint8)
            for oy, ox in ((0, 0), (0, 1), (1, 0)):
                img[np.clip(sy + oy, 0, h - 1), np.clip(sx + ox, 0, w - 1)] = col_bgr

    # 叠加 A* 绿色导航路径（如传入）
    if path_xy and len(path_xy) >= 2:
        pts_2d = []
        for wx, wy in path_xy:
            pr = _PROJ.project_world_point(float(wx), float(wy), 0.008, cam, float(w), float(h))
            if pr is not None:
                pts_2d.append((int(pr[0]), int(pr[1])))
        for i in range(len(pts_2d) - 1):
            cv2.line(img, pts_2d[i], pts_2d[i + 1], (155, 255, 195), 3, cv2.LINE_AA)

    # 叠加点击目标标针与机器人朝向图标
    if target_pin_xyz is not None:
        p0 = _PROJ.project_world_point(target_pin_xyz[0], target_pin_xyz[1], target_pin_xyz[2], cam, float(w), float(h))
        p1 = _PROJ.project_world_point(target_pin_xyz[0], target_pin_xyz[1], target_pin_xyz[2] + 0.09, cam, float(w), float(h))
        if p0 is not None and p1 is not None:
            cv2.circle(img, (int(p0[0]), int(p0[1])), 9, (150, 255, 210), 2, cv2.LINE_AA)
            cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (200, 255, 230), 2, cv2.LINE_AA)
            cv2.circle(img, (int(p1[0]), int(p1[1])), 4, (235, 255, 245), -1, cv2.LINE_AA)

    if robot_pose is not None:
        rx0, ry0, ryaw = robot_pose
        pr0 = _PROJ.project_world_point(rx0, ry0, 0.0, cam, float(w), float(h))
        pr_top = _PROJ.project_world_point(rx0, ry0, 0.08, cam, float(w), float(h))
        pr_nose = _PROJ.project_world_point(
            rx0 + 0.12 * math.cos(math.radians(ryaw)),
            ry0 + 0.12 * math.sin(math.radians(ryaw)),
            0.04, cam, float(w), float(h),
        )
        if pr0 is not None and pr_top is not None:
            cv2.circle(img, (int(pr0[0]), int(pr0[1])), 8, (100, 245, 140), 2, cv2.LINE_AA)
            cv2.line(img, (int(pr0[0]), int(pr0[1])), (int(pr_top[0]), int(pr_top[1])), (150, 255, 190), 2, cv2.LINE_AA)
        if pr_top is not None and pr_nose is not None:
            cv2.arrowedLine(img, (int(pr_top[0]), int(pr_top[1])), (int(pr_nose[0]), int(pr_nose[1])), (200, 255, 225), 2, cv2.LINE_AA, tipLength=0.35)

    return img
