# -*- coding: utf-8 -*-
"""verify_balancing_rocking_survival.py — 基于真机实测双轮自平衡摇摆参数的抗晃动生存力压测与可视化对比。

全部物理扰动参数直接源自本台 RIG-Hover 真机实测（COM3 串口 + /status 遥测 + 3181 帧平衡日志 + 400 帧光流谱 + robot.cc 源码）：
- 前后平衡极限环频率：1.16 Hz（周期 0.86 s，由 robot.cc:280 的 |u|>20 死区 ±10 阶跃补偿与轮组静摩擦引起）；
- 静止自平衡俯仰摇摆：±3.2°（峰峰值 6.4°，实测 p05..p95 = [-0.33°, +6.21°]，中位偏置 +2.76°）；
- 轮子前后微幅拉扯位移：±1.8 cm（与俯仰角正交耦合，实测 wheel_v p95 = 11.4）；
- 机身静态横滚偏斜：-4.37° ± 0.80°（实测 COM3/CSV 均值 -4.37°）；
- 相机-IMU 传输与曝光相位延迟：40 ms（在 1.16 Hz 过零点产生 ~1.2°~1.6° 逐帧残余滞后误差）；
- 头部舵机机械回程差：±3.5°（实测 COM3 target=0° 时 q_head=+3.7°）；
- 过零点卷帘微模糊：与 |pit_gyro| 成正比（折返点角速度≈0 时图像最清晰），外加 1 帧 +9.5° 碰撞冲击。

对比三种方案（严格留出集 empty_d0.5 + cb8455，100% 端侧 .tflite）：
1. Naive（无 IMU 倾角补偿、无 KISS-ICP、无门控）
2. IMU-Only（有 40ms 滞后的 IMU 粗配平，无 KISS-ICP 残差/轮涌修正、无门控）
3. Full Open-Source Stack（IMU 粗配平 + KISS-ICP 消相位滞后与前后轮涌 + 折返点角速度加权 + Dynablox + 校准后 Unblur-SLAM 门控）
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dense_scene_pipeline import (  # noqa: E402
    TraversabilityGrid2D,
    reconstruct_dense_frame_3d,
    render_minimap_2d,
    run_student_tflite_frame,
    solve_table_plane_from_student,
)
from neural_gaussian_splat_model import OrbitCamera3D  # noqa: E402
from robust_contour_slam import (  # noqa: E402
    RobustDualLayerContourMap,
    render_green_contour_edl,
)

RUN_DIR = REPO / "runs_end_side" / "dense_splat_v3"
TFLITE_PATH = RUN_DIR / "model.tflite"
HELDOUT_CB8455 = REPO / "dataset" / "takes" / "robot_20261115_151733_cb8455" / "take.mp4"
EMPTY_SESSION_DIR = REPO / "evidence" / "sessions" / "empty_d0.5_front_night_182201" / "frames"

# 真机实测自平衡物理参数
REAL_ROCK_HZ = 1.16
REAL_PITCH_AMP_DEG = 3.2
REAL_ROLL_BIAS_DEG = -4.37
REAL_ROLL_AMP_DEG = 0.80
REAL_WHEEL_SURGE_M = 0.018
REAL_IMU_LAG_SEC = 0.040
REAL_HEAD_HYST_DEG = 3.5
CAPTURE_FPS = 12.0


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.46,
             color: tuple[int, int, int] = (150, 255, 185), thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (6, 12, 8), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def rot_y(deg: float) -> np.ndarray:
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_x(deg: float) -> np.ndarray:
    ph = math.radians(deg)
    c, s = math.cos(ph), math.sin(ph)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def vertical_motion_blur(bgr: np.ndarray, pitch_rate_dps: float) -> np.ndarray:
    """按实时俯仰角速度模拟 GC0308 ~25ms 曝光的垂直运动微模糊（折返点清晰，过零点略糊）。"""
    # 72 deg / 240 px -> 3.33 px/deg; exposure ~0.025s -> blur_px ≈ |rate| * 0.083
    blur_px = int(round(abs(pitch_rate_dps) * 0.085))
    if blur_px < 3:
        return bgr
    k = min(19, blur_px | 1)
    ker = np.zeros((k, k), dtype=np.float32)
    ker[:, k // 2] = 1.0 / k
    return cv2.filter2D(bgr, -1, ker)


def simulate_rocking_trajectory(n_frames: int = 24, shock_idx: int = 14) -> list[dict]:
    """生成 24 帧（2 秒，约 2.3 个完整前后平衡周期）真机实测摇摆轨迹。"""
    traj = []
    dt = 1.0 / CAPTURE_FPS
    omega = 2.0 * math.pi * REAL_ROCK_HZ
    for i in range(n_frames):
        t = i * dt
        # 前 3 帧近似平稳建立初图，随后连续 1.16 Hz 前后极限环摇摆 + 第 shock_idx 帧碰撞冲击
        amp = 0.25 if i < 3 else 1.0
        true_pitch = amp * REAL_PITCH_AMP_DEG * math.sin(omega * t)
        true_pitch_rate = amp * REAL_PITCH_AMP_DEG * omega * math.cos(omega * t)  # deg/s
        true_roll = REAL_ROLL_BIAS_DEG + amp * REAL_ROLL_AMP_DEG * math.sin(omega * t + 0.7)
        true_surge_x = amp * REAL_WHEEL_SURGE_M * math.cos(omega * t)  # 与俯仰正交的轮子前后拉扯
        true_yaw_hyst = amp * REAL_HEAD_HYST_DEG * math.sin(0.55 * omega * t)

        # IMU 存在 40ms 传输+曝光滞后，且包含真实静态横滚/噪声
        t_lag = max(0.0, t - REAL_IMU_LAG_SEC)
        imu_pitch = amp * REAL_PITCH_AMP_DEG * math.sin(omega * t_lag)
        imu_roll = REAL_ROLL_BIAS_DEG + amp * REAL_ROLL_AMP_DEG * math.sin(omega * t_lag + 0.7)

        is_shock = (i == shock_idx)
        if is_shock:
            true_pitch += 9.5
            true_pitch_rate = 115.0  # 严重碰撞瞬间剧烈角速度 -> 明显卷帘模糊
            imu_pitch += 9.5

        traj.append({
            "idx": i,
            "t_s": t,
            "true_pitch": true_pitch,
            "true_pitch_rate": true_pitch_rate,
            "true_roll": true_roll,
            "true_surge_x": true_surge_x,
            "true_yaw_hyst": true_yaw_hyst,
            "imu_pitch": imu_pitch,
            "imu_roll": imu_roll,
            "is_shock": is_shock,
        })
    return traj


def run_mode(frames: list[np.ndarray], traj: list[dict], mode: str) -> dict:
    """运行一种模式: 'naive' | 'imu_only' | 'full_stack'。"""
    use_imu = mode in ("imu_only", "full_stack")
    use_stack = (mode == "full_stack")

    cmap = RobustDualLayerContourMap(
        vox_size=0.0085,
        fov_h_deg=72.0,
        confirm_hits=2,
        enable_icp=use_stack,
        enable_ever_free=use_stack,
        enable_disturbance_gate=use_stack,
    )
    grid = TraversabilityGrid2D()
    r_base = None
    z_base = 0.0

    green_hist = []
    nav_ok_count = 0
    nav_trials = 0
    icp_corrs = []
    shock_blocked = False

    # 用首帧学生推理锁住理想桌面参考平面，再对后续每帧施加真实机身前后摆动误差
    for step_info, bgr in zip(traj, frames):
        bgr_obs = vertical_motion_blur(bgr, step_info["true_pitch_rate"]) if step_info["idx"] >= 3 else bgr
        splats = run_student_tflite_frame(TFLITE_PATH, bgr_obs)
        if r_base is None:
            r_base, z_base, _ = solve_table_plane_from_student(
                splats[..., 0].astype(np.float32),
                splats[..., 1].astype(np.float32),
                fov_h_deg=72.0,
            )

        if not use_imu:
            # Naive: 完全无视真实摇摆和 -4.37° 横滚偏斜，按平直相机反投 -> 全部真摆动变成地图误差
            err_pitch = step_info["true_pitch"]
            err_roll = step_info["true_roll"]
            err_yaw = step_info["true_yaw_hyst"]
            err_x = step_info["true_surge_x"]
        else:
            # IMU 粗配平抵消了绝大部分 1.16Hz 摇摆和 -4.37° 横滚，但仍留有 40ms 相位滞后 + 轮子前后涌动 + 舵机回程差
            err_pitch = step_info["true_pitch"] - step_info["imu_pitch"]
            err_roll = step_info["true_roll"] - step_info["imu_roll"]
            err_yaw = step_info["true_yaw_hyst"]
            err_x = step_info["true_surge_x"]
            # 碰撞瞬间 IMU 与曝光不同步，额外产生瞬态大错配
            if step_info["is_shock"]:
                err_pitch += 4.5

        r_used = (rot_y(err_pitch) @ rot_x(err_roll) @ r_base).astype(np.float32)
        recon = reconstruct_dense_frame_3d(
            bgr=bgr_obs,
            splats_96=splats,
            r_level=r_used,
            z_bias=z_base,
            fov_h_deg=72.0,
            yaw_deg=err_yaw,
            cam_x=err_x,
            cam_y=0.0,
            out_hw=(192, 192),
        )

        st = cmap.update(
            bgr=bgr_obs,
            pts=recon["pts"],
            nrms=recon["nrms"],
            depth_hr=recon["depth_hr"],
            r_level=r_used,
            yaw_deg=err_yaw,
            pitch_deg=step_info["imu_pitch"],
            roll_deg=step_info["imu_roll"],
            cam_x=err_x,
            cam_y=0.0,
            z_bias=z_base,
        )
        if step_info["is_shock"] and st.is_disturbed:
            shock_blocked = True
        if not st.is_disturbed and st.icp_rot_corr_deg > 0.01:
            icp_corrs.append(st.icp_rot_corr_deg)

        comb = cmap.get_combined_for_traversability()
        grid.update(comb, robot_x=0.0, robot_y=0.0, robot_yaw_deg=0.0)
        if step_info["idx"] >= 4:
            ng = int(grid.reach.sum())
            green_hist.append(ng)
            nav_trials += 1
            # 检验每帧是否都能成功点击远端绿格规划出无碰撞 A* 路径
            ry, rx = np.where(grid.reach)
            if len(ry) > 0:
                k = int(np.argmax(np.hypot(grid.xs[rx], grid.ys[ry])))
                wx = float(grid.xs[rx[k]] + 0.5 * grid.res)
                wy = float(grid.ys[ry[k]] + 0.5 * grid.res)
                plan = grid.click_and_plan(wx, wy, 0.0, 0.0, 0.0)
                if plan.get("accepted"):
                    nav_ok_count += 1

    flat = cmap.desk_flatness_metrics()
    s_pos, s_nrm = cmap.get_static_confirmed()
    # 末帧规划一条展示路径
    ry, rx = np.where(grid.reach)
    final_path = None
    if len(ry) > 0:
        k = int(np.argmax(np.hypot(grid.xs[rx], grid.ys[ry])))
        pl = grid.click_and_plan(
            float(grid.xs[rx[k]] + 0.5 * grid.res),
            float(grid.ys[ry[k]] + 0.5 * grid.res),
            0.0, 0.0, 0.0,
        )
        if pl.get("accepted"):
            final_path = pl["path"]

    return {
        "map": cmap,
        "grid": grid,
        "path": final_path,
        "metrics": {
            "mode": mode,
            "static_pts": int(len(s_pos)),
            "desk_z_std_mm": round(flat["z_std_mm"], 2),
            "desk_z_mad_mm": round(flat["z_mad_mm"], 2),
            "blocked_frames": int(cmap.disturbed_frames_blocked),
            "shock_blocked": bool(shock_blocked),
            "normal_rock_frames_accepted": int(len(traj) - 1 - max(0, cmap.disturbed_frames_blocked - (1 if shock_blocked else 0))),
            "icp_mean_rot_corr_deg": round(float(np.mean(icp_corrs)), 2) if icp_corrs else 0.0,
            "green_cells_mean": round(float(np.mean(green_hist)), 1) if green_hist else 0.0,
            "green_cells_min": int(np.min(green_hist)) if green_hist else 0,
            "astar_success_rate": round(nav_ok_count / max(1, nav_trials), 3),
        },
    }


def read_frames(kind: str, path: pathlib.Path, start: int, count: int) -> list[np.ndarray]:
    if kind == "video":
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out = []
        while len(out) < count:
            ok, bgr = cap.read()
            if not ok:
                break
            out.append(bgr)
        cap.release()
        return out
    jpgs = sorted(path.glob("*.jpg"))[start : start + count]
    return [cv2.imread(str(j)) for j in jpgs if cv2.imread(str(j)) is not None]


def main() -> None:
    t0 = time.perf_counter()
    traj = simulate_rocking_trajectory(n_frames=24, shock_idx=14)
    cases = [
        ("HELD-OUT empty_d0.5 (static desk+keyboard under real 1.16Hz +/-3.2deg rock + -4.4deg roll + 1.8cm surge + 40ms lag)",
         read_frames("frames", EMPTY_SESSION_DIR, 12, 24),
         OrbitCamera3D(azimuth_deg=-148.0, elevation_deg=36.0, distance_m=0.94, target_x=0.40, target_y=0.02, target_z=0.08)),
        ("HELD-OUT cb8455 (cup+notebook+arm under real 1.16Hz +/-3.2deg rock + -4.4deg roll + 1.8cm surge + 40ms lag)",
         read_frames("video", HELDOUT_CB8455, 595, 24),
         OrbitCamera3D(azimuth_deg=-145.0, elevation_deg=40.0, distance_m=0.98, target_x=0.42, target_y=0.05, target_z=0.10)),
    ]

    pw, ph = 560, 400
    header = np.full((42, pw * 3 + 8, 3), (10, 18, 12), dtype=np.uint8)
    put_text(
        header,
        "Real RIG-Hover 1.16Hz Balancing Rocking Survival Test (+/-3.2deg Pitch, -4.4deg Roll, +/-1.8cm Surge, 40ms Lag, +9.5deg Bump)",
        (10, 27),
        scale=0.50,
    )

    rows = [header]
    all_results = []
    mode_titles = [
        ("naive", "1) NAIVE (No IMU / No ICP / No Gate)"),
        ("imu_only", "2) IMU LEVELING ONLY (40ms lag + surge + hyst)"),
        ("full_stack", "3) FULL OPEN-SOURCE STACK (IMU+KISS-ICP+Dynablox+Gate)"),
    ]

    for title, frames, cam in cases:
        bar = np.full((30, pw * 3 + 8, 3), (18, 34, 22), dtype=np.uint8)
        put_text(bar, title, (10, 21), scale=0.45, color=(190, 255, 210))
        panels = []
        case_rec = {"case": title, "modes": {}}
        for m_key, m_label in mode_titles:
            res = run_mode(frames, traj, m_key)
            m = res["metrics"]
            case_rec["modes"][m_key] = m
            s_pos, s_nrm = res["map"].get_static_confirmed()
            view = render_green_contour_edl(
                static_pts=s_pos,
                static_nrms=s_nrm,
                cam=cam,
                grid=res["grid"],
                dyn_pts=res["map"].d_pos,
                dyn_nrms=res["map"].d_nrm,
                w=pw,
                h=ph,
                robot_pose=(0.0, 0.0, 0.0),
                path=res["path"],
                draw_ground_grid=True,
                draw_trav_overlay=True,
            )
            mm = render_minimap_2d(res["grid"], size_px=108, robot_pose=(0.0, 0.0, 0.0), path=res["path"])
            view[ph - 108 - 6 : ph - 6, pw - 108 - 6 : pw - 6] = mm

            hud = [
                m_label,
                f"desk MAD={m['desk_z_mad_mm']:.1f}mm  std={m['desk_z_std_mm']:.1f}mm  pts={m['static_pts']}",
                f"A* pass={m['astar_success_rate']*100:.0f}%  green={m['green_cells_mean']:.0f}  blocked={m['blocked_frames']}  ICP={m['icp_mean_rot_corr_deg']:.2f}deg",
            ]
            ov = view.copy()
            cv2.rectangle(ov, (0, 0), (pw, 66), (6, 12, 8), -1)
            view = cv2.addWeighted(ov, 0.74, view, 0.26, 0)
            for li, line in enumerate(hud):
                put_text(view, line, (8, 19 + 20 * li), scale=0.43)
            panels.append(view)
            print(f"  [{m_key:10s}] {m}", flush=True)

        vbar = np.full((ph, 4, 3), (38, 86, 52), dtype=np.uint8)
        rows.extend([bar, np.hstack([panels[0], vbar, panels[1], vbar, panels[2]])])
        all_results.append(case_rec)

    out_png = RUN_DIR / "showcase_balancing_survival_v5.png"
    cv2.imwrite(str(out_png), np.vstack(rows))
    out_json = RUN_DIR / "summary_balancing_survival_v5.json"
    out_json.write_text(
        json.dumps(
            {
                "real_hardware_params": {
                    "rock_hz": REAL_ROCK_HZ,
                    "pitch_amp_deg": REAL_PITCH_AMP_DEG,
                    "roll_bias_deg": REAL_ROLL_BIAS_DEG,
                    "roll_amp_deg": REAL_ROLL_AMP_DEG,
                    "wheel_surge_m": REAL_WHEEL_SURGE_M,
                    "imu_lag_ms": REAL_IMU_LAG_SEC * 1000.0,
                    "head_hyst_deg": REAL_HEAD_HYST_DEG,
                },
                "results": all_results,
                "elapsed_s": round(time.perf_counter() - t0, 1),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"SAVED {out_png} & {out_json} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
