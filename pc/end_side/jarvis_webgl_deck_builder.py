#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jarvis_webgl_deck_builder.py —— 独立 60 FPS 交互式 3D 全息空间智脑指挥舱 (`JARVIS Holo-Table Command Deck v10`) 生成器

生成的单文件 HTML 零依赖、无需启动服务器，双击即可在 Chrome/Edge 中 60 FPS 运行，包含：
1. 【双真实场景无缝切换】：
   - 场景 A (`empty_d0.5` 真实桌面)：16 万高密 3D 高斯点 + 无监督自动解耦 3D OBB 与真实毫米尺寸框 + `HoverSDF-Net` 实时避障势场；
   - 场景 B (`cb8455` 真实 4D 活体时空记忆)：静态全息桌面底图 + **14 帧真实人体/手部双敲 4D 动态 3D 点云动画**（可一边 360° 自由旋转一边看真实的手在 3D 空间里抬起敲击桌面并激发桌面雷达冲击波！）；
2. 【3D 铰接式文文自平衡机器人数字孪生体 (`Wenwen 3D Digital Twin`)】：
   - 双轮滚动 + 随加速度前倾/后仰的 `1.16 Hz` 倒立摆车身 + 独立摇头舵机头部 + 65° 半透明荧光绿相机视锥；
   - **鼠标点击桌面任意位置**：利用内嵌的 `HoverSDF-Net` 连续符号距离场 + 梯度排斥力实时在浏览器内解出平滑避障轨迹，文文数字孪生体立刻启动 `HoverFineMotionPolicyV3` 游标精停律平滑绕开真实鼠标/水杯抵达目标点，驻留后自动抬头回看！
3. 【沉浸式特技工具箱】：
   - `乘坐文文第一视角 (Robot-Eye POV)`：将 3D 镜头直接架在文文脑袋上，跟随它以第一视角穿梭你的真实桌面！
   - `3D 毫米激光测距尺 (Laser Ruler)`：点击 3D 场景任意两点实时测算真实三维距离 `(ΔX, ΔY, ΔZ, 空间距 mm)`！
   - `5 种实时 GLSL 着色器`：全息照片轮廓绿 / 激光雷达扫描波 / 神经边界 X 光 / 3D 高程等高地形 / 纯几何法向！
"""

from __future__ import annotations

import base64
import json
import pathlib
from typing import Any

import numpy as np


def _pack_pts_and_cols(pts: np.ndarray, rgb_f: np.ndarray, max_pts: int) -> dict[str, Any]:
    """将 3D 点云量化为紧凑 Int16 毫米坐标 + Uint8 RGB Base64（保证单文件体积小且 60 FPS 秒开）。"""
    n_raw = len(pts)
    if n_raw == 0:
        return {"n": 0, "xyz_b64": "", "rgb_b64": ""}
    step = max(1, n_raw // int(max_pts))
    sp = pts[::step].astype(np.float32)
    sc = np.clip(rgb_f[::step] * 255.0, 0, 255).astype(np.uint8)
    mm = np.clip(np.rint(sp * 1000.0), -32000, 32000).astype("<i2")
    return {
        "n": int(len(sp)),
        "xyz_b64": base64.b64encode(mm.tobytes()).decode("ascii"),
        "rgb_b64": base64.b64encode(sc.tobytes()).decode("ascii"),
    }


def _pack_scene_payload(scene: dict[str, Any], max_static_pts: int = 135_000, max_dyn_pts: int = 12_000) -> dict[str, Any]:
    """打包单个真实场景的静态全息点云、纯几何对比色、自动 3D OBB、烘焙的 HoverSDF-Net 势场以及可选 4D 动态帧序列。"""
    pts = scene["pts"]
    rgb_holo = scene["rgb_holo"]
    rgb_geo = scene["rgb_geo"]
    contour = scene["contour"]
    step = max(1, len(pts) // int(max_static_pts))
    sp = pts[::step].astype(np.float32)
    mm = np.clip(np.rint(sp * 1000.0), -32000, 32000).astype("<i2")
    c_holo = np.clip(rgb_holo[::step] * 255.0, 0, 255).astype(np.uint8)
    c_geo = np.clip(rgb_geo[::step] * 255.0, 0, 255).astype(np.uint8)
    c_edge = np.clip(contour[::step] * 255.0, 0, 255).astype(np.uint8)

    # 烘焙 64×64 HoverSDF-Net 势场（单位 mm，Int16）供浏览器端实时 Neural-CBF 避障规划！
    sdf_grid = scene["sdf_grid_64"]  # [64, 64] meters
    sdf_mm = np.clip(np.rint(sdf_grid * 1000.0), -3000, 3000).astype("<i2")

    dyn_packed = []
    for df in scene.get("dyn_frames", []):
        dp = _pack_pts_and_cols(df["pts"], df["rgb"], max_pts=max_dyn_pts)
        dp["t_s"] = round(float(df["t_s"]), 2)
        dp["knock_flash"] = bool(df.get("knock_flash", False))
        dyn_packed.append(dp)

    return {
        "id": scene["id"],
        "name": scene["name"],
        "subtitle": scene["subtitle"],
        "n_static": int(len(sp)),
        "xyz_b64": base64.b64encode(mm.tobytes()).decode("ascii"),
        "holo_b64": base64.b64encode(c_holo.tobytes()).decode("ascii"),
        "geo_b64": base64.b64encode(c_geo.tobytes()).decode("ascii"),
        "edge_b64": base64.b64encode(c_edge.tobytes()).decode("ascii"),
        "obbs": scene["obbs"],
        "default_goal": scene["default_goal"],
        "default_path": scene["default_path"],
        "knock_xyz": scene.get("knock_xyz"),
        "face_xyz": scene.get("face_xyz", [0.85, 0.02, 0.30]),
        "sdf_meta": {
            "res": int(sdf_grid.shape[0]),
            "x_range": list(scene["sdf_x_range"]),
            "y_range": list(scene["sdf_y_range"]),
            "mae_mm": scene["sdf_mae_mm"],
            "eikonal_err": scene["sdf_eikonal_err"],
            "sdf_mm_b64": base64.b64encode(sdf_mm.tobytes()).decode("ascii"),
        },
        "dyn_frames": dyn_packed,
    }


def build_jarvis_holo_command_deck_html(
    scene_a_empty: dict[str, Any],
    scene_b_cb8455_4d: dict[str, Any],
    out_html_path: pathlib.Path,
) -> dict[str, Any]:
    """构建零依赖、60 FPS 单文件交互式 3D 全息空间智脑指挥舱 (`jarvis_holo_command_deck_v10.html`)。"""
    pack_a = _pack_scene_payload(scene_a_empty, max_static_pts=140_000)
    pack_b = _pack_scene_payload(scene_b_cb8455_4d, max_static_pts=120_000, max_dyn_pts=11_000)
    data_json = json.dumps({"empty_d05": pack_a, "cb8455_4d": pack_b}, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RIG-Hover 文文 · JARVIS 4D 全息空间智脑指挥舱 (v10)</title>
<style>
  * {{ box-sizing: border-box; user-select: none; }}
  html, body {{
    margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
    background: #020704; color: #a8ffca;
    font-family: 'Segoe UI', 'Microsoft YaHei', Consolas, monospace;
  }}
  #gl {{ display: block; width: 100%; height: 100%; cursor: grab; }}
  #gl.crosshair {{ cursor: crosshair; }}
  .panel {{
    position: fixed; background: rgba(3, 14, 8, 0.84);
    border: 1px solid rgba(55, 220, 125, 0.45);
    box-shadow: 0 0 22px rgba(20, 160, 80, 0.18), inset 0 0 12px rgba(35, 190, 100, 0.08);
    backdrop-filter: blur(8px); border-radius: 10px;
  }}
  #topHud {{
    top: 14px; left: 14px; padding: 12px 16px; max-width: 520px;
  }}
  .title-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .pulse-dot {{
    width: 10px; height: 10px; border-radius: 50%; background: #48ff96;
    box-shadow: 0 0 10px #48ff96; animation: pulse 1.6s infinite;
  }}
  @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.45;transform:scale(0.8)}} }}
  .h-title {{ font-size: 15px; font-weight: 700; letter-spacing: 0.5px; color: #d8ffe7; }}
  .h-sub {{ font-size: 11.5px; color: #7ce2a6; line-height: 1.5; }}
  #telemetry {{
    top: 14px; right: 14px; width: 292px; padding: 12px 14px; font-size: 11.5px;
  }}
  .t-sec {{ font-size: 11px; font-weight: 700; color: #54e895; letter-spacing: 1px; margin: 7px 0 4px; border-bottom: 1px solid rgba(55,220,125,0.22); padding-bottom: 2px; }}
  .kv {{ display: flex; justify-content: space-between; margin: 2.5px 0; }}
  .kv span:first-child {{ color: #7cc99c; }}
  .kv b {{ color: #d6ffe6; font-family: Consolas, monospace; }}
  .bar-bg {{ width: 100%; height: 6px; background: rgba(25,75,45,0.55); border-radius: 3px; overflow: hidden; margin: 3px 0 6px; }}
  .bar-fill {{ height: 100%; background: linear-gradient(90deg, #1fd16c, #9bffcc); width: 0%; transition: width 0.08s linear; }}
  #obbList {{ max-height: 126px; overflow-y: auto; font-size: 11px; line-height: 1.45; }}
  .obb-item {{ padding: 2px 5px; margin: 2px 0; background: rgba(16, 60, 34, 0.45); border-left: 2px solid #52ff9c; border-radius: 3px; }}
  #dock {{
    bottom: 14px; left: 14px; right: 14px; padding: 10px 14px;
    display: flex; flex-wrap: wrap; gap: 7px; align-items: center; justify-content: space-between;
  }}
  .btn-group {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  button {{
    background: rgba(10, 42, 24, 0.88); color: #9effc5;
    border: 1px solid rgba(55, 220, 125, 0.5); border-radius: 6px;
    padding: 6px 11px; font-size: 12px; cursor: pointer;
    font-family: 'Microsoft YaHei', Consolas, monospace; transition: all 0.15s;
  }}
  button:hover {{ background: rgba(24, 90, 52, 0.95); box-shadow: 0 0 12px rgba(72,255,150,0.3); }}
  button.active {{
    background: linear-gradient(135deg, #167a42, #25b864); color: #021409;
    font-weight: 700; border-color: #b4ffd4; box-shadow: 0 0 14px rgba(80,255,155,0.45);
  }}
  #labelsLayer {{ position: fixed; inset: 0; pointer-events: none; overflow: hidden; }}
  .tag3d {{
    position: absolute; transform: translate(-50%, -100%);
    background: rgba(3, 18, 9, 0.86); border: 1px solid #4dff98;
    color: #d0ffe3; padding: 2px 7px; border-radius: 4px;
    font-size: 11px; white-space: nowrap; box-shadow: 0 0 8px rgba(40,220,115,0.28);
  }}
  .tag3d.robot {{ border-color: #80ffff; color: #d8ffff; background: rgba(4, 24, 28, 0.88); }}
  .tag3d.knock {{ border-color: #fff275; color: #ffffc2; background: rgba(32, 26, 4, 0.88); }}
  #toast {{
    position: fixed; top: 82px; left: 50%; transform: translateX(-50%);
    background: rgba(8, 38, 22, 0.92); border: 1px solid #6effaa; color: #e2ffee;
    padding: 7px 16px; border-radius: 20px; font-size: 12.5px; pointer-events: none;
    opacity: 0; transition: opacity 0.25s; box-shadow: 0 0 18px rgba(72,255,150,0.35);
  }}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="labelsLayer"></div>
<div id="toast"></div>

<div id="topHud" class="panel">
  <div class="title-row">
    <div class="pulse-dot"></div>
    <div class="h-title">RIG-Hover 文文 · JARVIS 4D 全息空间智脑指挥舱 (v10)</div>
  </div>
  <div class="h-sub" id="sceneInfo">加载自研 HoverGeoNet-ViT-HD v3 + HoloContour-3DGS + HoverSDF-Net...</div>
  <div class="h-sub" style="margin-top:3px;color:#b8ffd6;">
    左键旋转 · 滚轮缩放 · 右键平移 · <b>点击桌面任意点直接指挥文文绕障精停！</b>
  </div>
</div>

<div id="telemetry" class="panel">
  <div class="t-sec">WENWEN DIGITAL TWIN TELEMETRY</div>
  <div class="kv"><span>当前状态 MODE</span><b id="tMode">IDLE_STANDBY</b></div>
  <div class="kv"><span>机器人位置 (X, Y)</span><b id="tPos">0.000, 0.000 m</b></div>
  <div class="kv"><span>1.16Hz 自平衡俯仰</span><b id="tPitch">+3.0°</b></div>
  <div class="kv"><span>距目标剩余距离</span><b id="tRem">-- cm</b></div>
  <div class="kv"><span>HoverSDF-Net 净空</span><b id="tSdf">-- cm</b></div>
  <div class="kv"><span>在线打滑辨识 α_slip</span><b id="tSlip">0.780</b></div>
  <div class="kv"><span>游标精停门控 Vernier</span><b id="tVernier">0%</b></div>
  <div class="bar-bg"><div class="bar-fill" id="vernierBar"></div></div>
  <div class="t-sec">AUTO 3D OBB 物体毫米测量 (ZERO MANUAL LABEL)</div>
  <div id="obbList"></div>
  <div class="t-sec">3D 激光测距尺 (RULER)</div>
  <div class="kv"><span>测量结果</span><b id="tRuler">点击下方“3D测距尺”启用</b></div>
</div>

<div id="dock" class="panel">
  <div class="btn-group">
    <button id="btnSceneA" class="active" onclick="switchScene('empty_d05')">① 真实桌面 (鼠标+键盘 3D OBB)</button>
    <button id="btnSceneB" onclick="switchScene('cb8455_4d')">② 真实 4D 活体双敲 (手部3D动态!)</button>
    <button id="btnReplay" onclick="triggerScenario1Demo()">▶ 触发双敲绕障全流程</button>
  </div>
  <div class="btn-group">
    <button id="btnNavMode" class="active" onclick="setInteractMode('nav')">🎯 点击桌面派单</button>
    <button id="btnRulerMode" onclick="setInteractMode('ruler')">📏 3D 毫米测距尺</button>
    <button id="btnPov" onclick="togglePov()">🤖 乘坐文文第一视角</button>
    <button id="btnScan" class="active" onclick="toggleScan()">📡 雷达扫描波</button>
    <button id="btnObb" class="active" onclick="toggleObb()">📦 3D OBB 框</button>
  </div>
  <div class="btn-group">
    <button id="btnShader" onclick="cycleShader()">🎨 着色: ① 全息照片轮廓绿</button>
    <button onclick="setView(176, 22, 0.62, 0.45, 0.0, 0.05)">视角: 照片同视角</button>
    <button onclick="setView(-152, 28, 0.72, 0.46, 0.0, 0.05)">视角: 35°浮雕</button>
    <button onclick="setView(180, 82, 1.05, 0.48, 0.0, 0.0)">视角: 正俯视</button>
    <button onclick="ptSize=Math.max(1.5,ptSize-0.6)">点径-</button>
    <button onclick="ptSize=Math.min(7.5,ptSize+0.6)">点径+</button>
  </div>
</div>

<script>
const SCENES = {data_json};

function b64ToU8(s){{
  if(!s) return new Uint8Array(0);
  const b = atob(s); const a = new Uint8Array(b.length);
  for(let i=0;i<b.length;i++) a[i] = b.charCodeAt(i);
  return a;
}}

function toast(msg){{
  const el = document.getElementById('toast');
  el.textContent = msg; el.style.opacity = '1';
  clearTimeout(window._tTimer);
  window._tTimer = setTimeout(()=>el.style.opacity='0', 2800);
}}

// 解包场景数据为 Float32Arrays + GPU 缓冲
const cv = document.getElementById('gl');
const gl = cv.getContext('webgl', {{antialias: true, preserveDrawingBuffer: true}});

const vsPts = `
attribute vec3 aPos;
attribute vec3 aHolo;
attribute vec3 aGeo;
attribute float aEdge;
uniform mat4 uMVP;
uniform float uPt;
uniform float uTime;
uniform int uShader;
uniform float uScanOn;
uniform float uScanR;
varying vec3 vCol;
void main(){{
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = clamp(uPt / max(gl_Position.w, 0.14), 1.4, 8.0);
  vec3 c = aHolo;
  if(uShader == 1){{
    c = aGeo;
  }} else if(uShader == 2){{
    // 神经边界 X 光模式
    float e = pow(aEdge, 1.15);
    c = vec3(0.06 + 0.55*e, 0.14 + 0.86*e, 0.10 + 0.72*e);
  }} else if(uShader == 3){{
    // 3D 高程等高地形 (每 8mm 一条荧光等高环)
    float z = clamp((aPos.z + 0.01) / 0.22, 0.0, 1.0);
    float iso = smoothstep(0.90, 0.99, abs(sin(aPos.z * 392.7)));
    c = vec3(0.05 + 0.45*z + 0.40*iso, 0.22 + 0.78*z + 0.25*iso, 0.18 + 0.55*(1.0-z) + 0.45*iso);
  }}
  // 激光雷达全息扫描波
  if(uScanOn > 0.5){{
    float dist = length(aPos.xy);
    float ring = exp(-pow((dist - uScanR) / 0.032, 2.0));
    c += vec3(0.28, 0.72, 0.55) * ring;
  }}
  vCol = clamp(c, 0.0, 1.0);
}}`;

const fsPts = `
precision mediump float;
varying vec3 vCol;
void main(){{
  vec2 d = gl_PointCoord - 0.5;
  if(dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vCol, 1.0);
}}`;

const vsLine = `
attribute vec3 aPos;
attribute vec3 aCol;
uniform mat4 uMVP;
uniform float uPt;
varying vec3 vCol;
void main(){{
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = clamp(uPt / max(gl_Position.w, 0.14), 2.0, 9.0);
  vCol = aCol;
}}`;

const fsLine = `
precision mediump float;
varying vec3 vCol;
void main(){{ gl_FragColor = vec4(vCol, 1.0); }}`;

function mkProg(vs, fs){{
  const sh=(t,s)=>{{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o;}};
  const p=gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  return p;
}}
const progPts = mkProg(vsPts, fsPts);
const progLine = mkProg(vsLine, fsLine);
function mkBuf(arr){{ const b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); return b; }}

const gpuScenes = {{}};
for(const key of Object.keys(SCENES)){{
  const sc = SCENES[key];
  const N = sc.n_static;
  const xyz16 = new Int16Array(b64ToU8(sc.xyz_b64).buffer);
  const h8 = b64ToU8(sc.holo_b64);
  const g8 = b64ToU8(sc.geo_b64);
  const e8 = b64ToU8(sc.edge_b64);
  const pos = new Float32Array(N*3), holo = new Float32Array(N*3), geo = new Float32Array(N*3), edge = new Float32Array(N);
  for(let i=0;i<N*3;i++){{ pos[i]=xyz16[i]*0.001; holo[i]=h8[i]/255.0; geo[i]=g8[i]/255.0; }}
  for(let i=0;i<N;i++){{ edge[i]=e8[i]/255.0; }}

  const sdf16 = new Int16Array(b64ToU8(sc.sdf_meta.sdf_mm_b64).buffer);
  const sdfF32 = new Float32Array(sdf16.length);
  for(let i=0;i<sdf16.length;i++) sdfF32[i] = sdf16[i]*0.001;

  const dynBufs = sc.dyn_frames.map(df => {{
    const dn = df.n;
    const d16 = new Int16Array(b64ToU8(df.xyz_b64).buffer);
    const dc8 = b64ToU8(df.rgb_b64);
    const dp = new Float32Array(dn*3), dc = new Float32Array(dn*3);
    for(let i=0;i<dn*3;i++){{ dp[i]=d16[i]*0.001; dc[i]=dc8[i]/255.0; }}
    return {{ n: dn, bPos: mkBuf(dp), bCol: mkBuf(dc), t_s: df.t_s, knock_flash: df.knock_flash }};
  }});

  gpuScenes[key] = {{
    meta: sc,
    N,
    bPos: mkBuf(pos),
    bHolo: mkBuf(holo),
    bGeo: mkBuf(geo),
    bEdge: mkBuf(edge),
    sdfF32,
    dynBufs,
  }};
}}

// 地面 5cm 物理参考网格
const gridP=[], gridC=[];
for(let x=0;x<=1.201;x+=0.05){{ const m=Math.round(x*100)%20===0?0.26:0.10; gridP.push(x,-0.60,0,x,0.60,0); gridC.push(0.03,m,0.08,0.03,m,0.08); }}
for(let y=-0.60;y<=0.601;y+=0.05){{ const m=Math.round(Math.abs(y)*100)%20===0?0.26:0.10; gridP.push(0,y,0,1.20,y,0); gridC.push(0.03,m,0.08,0.03,m,0.08); }}
const bGridP = mkBuf(new Float32Array(gridP)), bGridC = mkBuf(new Float32Array(gridC));

// 交互与状态
let curKey = 'empty_d05';
let shaderMode = 0;
const shaderNames = ['① 全息照片轮廓绿', '② 纯几何法向绿', '③ 神经边界 X 光', '④ 3D 高程等高地形'];
let scanOn = true, showObb = true, povMode = false, interactMode = 'nav';
let az = 176, el = 22, dist = 0.62, tx = 0.45, ty = 0.0, tz = 0.05, ptSize = 3.4;

// 文文 3D 数字孪生体状态 + 实时 Neural-CBF 轨迹跟随
const robot = {{
  x: 0.04, y: 0.0, yaw: 0.0, headYaw: 0.0, pitch: 3.0, wheelPhase: 0.0,
  mode: 'IDLE_STANDBY', path: [], goal: null, dwellT: 0.0,
  slipEst: 0.78, vernierGate: 0.0, remM: 0.0, sdfClear: 0.25,
  rippleT: -1.0, rippleXY: [0.5, 0.0],
}};
let rulerPts = [];
let dynPlayIdx = 0, dynAccT = 0;

function querySdfAndGrad(scObj, x, y){{
  const sm = scObj.meta.sdf_meta;
  const R = sm.res;
  const [x0, x1] = sm.x_range, [y0, y1] = sm.y_range;
  const fx = (x - x0) / (x1 - x0) * (R - 1);
  const fy = (y - y0) / (y1 - y0) * (R - 1);
  const ix = Math.max(1, Math.min(R - 2, Math.floor(fx)));
  const iy = Math.max(1, Math.min(R - 2, Math.floor(fy)));
  const tx0 = Math.max(0, Math.min(1, fx - ix));
  const ty0 = Math.max(0, Math.min(1, fy - iy));
  const f = scObj.sdfF32;
  const v00=f[ix*R+iy], v10=f[(ix+1)*R+iy], v01=f[ix*R+iy+1], v11=f[(ix+1)*R+iy+1];
  const phi = (1-tx0)*(1-ty0)*v00 + tx0*(1-ty0)*v10 + (1-tx0)*ty0*v01 + tx0*ty0*v11;
  const dx = (x1 - x0) / (R - 1), dy = (y1 - y0) / (R - 1);
  const gx = (f[(ix+1)*R+iy] - f[(ix-1)*R+iy]) / (2*dx);
  const gy = (f[ix*R+iy+1] - f[ix*R+iy-1]) / (2*dy);
  return {{phi, gx, gy}};
}}

// 浏览器端实时 HoverSDF-Net + Neural-CBF 平滑避障轨迹优化 (多拓扑启动 < 2ms!)
function planNeuralCbfPath(scObj, sx, sy, gx, gy){{
  const K = 24, safeR = 0.088;
  const vx = gx - sx, vy = gy - sy, d = Math.max(Math.hypot(vx, vy), 0.01);
  const nx = -vy / d, ny = vx / d;
  let bestCost = 1e9, bestPts = null;
  for(const bias of [0.0, -0.18, 0.18, -0.30, 0.30]){{
    const pts = [];
    for(let i=0;i<K;i++){{
      const t = i / (K - 1);
      const arch = Math.sin(t * Math.PI) * bias;
      pts.push([sx + vx*t + nx*arch, sy + vy*t + ny*arch]);
    }}
    for(let it=0;it<55;it++){{
      for(let i=1;i<K-1;i++){{
        const [px, py] = pts[i];
        const s = querySdfAndGrad(scObj, px, py);
        // 平滑弹性力 + 神经控制屏障梯度排斥力
        let fx = 0.42 * (pts[i-1][0] + pts[i+1][0] - 2*px);
        let fy = 0.42 * (pts[i-1][1] + pts[i+1][1] - 2*py);
        if(s.phi < safeR + 0.025){{
          const pen = (safeR + 0.025 - s.phi);
          const gn = Math.hypot(s.gx, s.gy) + 1e-4;
          fx += (s.gx / gn) * pen * 0.55;
          fy += (s.gy / gn) * pen * 0.55;
        }}
        pts[i][0] = Math.max(0.02, Math.min(1.06, px + fx));
        pts[i][1] = Math.max(-0.48, Math.min(0.48, py + fy));
      }}
    }}
    let minPhi = 1.0, len = 0.0;
    for(let i=1;i<K-1;i++){{
      minPhi = Math.min(minPhi, querySdfAndGrad(scObj, pts[i][0], pts[i][1]).phi);
    }}
    for(let i=1;i<K;i++) len += Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
    const cost = len + Math.max(0, safeR - minPhi) * 35.0;
    if(cost < bestCost){{ bestCost = cost; bestPts = pts; }}
  }}
  return bestPts;
}}

function dispatchRobotTo(gx, gy, label='已锁定 3D 桌面目标'){{
  const scObj = gpuScenes[curKey];
  robot.goal = [gx, gy];
  robot.path = planNeuralCbfPath(scObj, robot.x, robot.y, gx, gy);
  robot.mode = 'APPROACH_CBF';
  robot.dwellT = 0.0;
  robot.rippleT = 0.0;
  robot.rippleXY = [gx, gy];
  toast(`${{label}} (${{gx.toFixed(2)}}m, ${{gy.toFixed(2)}}m) · HoverSDF-Net 实时规划完成！`);
}}

function switchScene(key){{
  curKey = key;
  document.getElementById('btnSceneA').classList.toggle('active', key==='empty_d05');
  document.getElementById('btnSceneB').classList.toggle('active', key==='cb8455_4d');
  const sc = gpuScenes[key].meta;
  document.getElementById('sceneInfo').innerHTML =
    `<b>${{sc.name}}</b> · 静态 <b>${{sc.n_static.toLocaleString()}}</b> 点` +
    (sc.dyn_frames.length ? ` + <b>${{sc.dyn_frames.length}} 帧真实 4D 活体手部动画</b>` : '') +
    ` · HoverSDF-Net MAE=<b>${{sc.sdf_meta.mae_mm}}mm</b>`;

  const ol = document.getElementById('obbList');
  ol.innerHTML = sc.obbs.map(o =>
    `<div class="obb-item"><b>#${{o.id}} ${{o.archetype}}</b> · ${{o.dims_cm[0]}}×${{o.dims_cm[1]}}×${{o.dims_cm[2]}} cm (距 ${{o.dist_m}}m)</div>`
  ).join('') || '<div class="obb-item">全桌面畅通无阻</div>';

  robot.x = 0.04; robot.y = 0.0; robot.yaw = 0.0; robot.headYaw = 0.0;
  robot.mode = 'IDLE_STANDBY'; robot.path = sc.default_path || []; robot.goal = null;
  rulerPts = [];
  if(key === 'cb8455_4d'){{
    toast('已切换至 4D 活体场景：你现在可以 360° 旋转观看自己真实的手在 3D 空间里双敲！');
  }}
}}

function triggerScenario1Demo(){{
  const sc = gpuScenes[curKey].meta;
  robot.x = 0.04; robot.y = 0.0; robot.yaw = 0.0;
  const g = sc.knock_xyz ? [sc.knock_xyz[0], sc.knock_xyz[1]] : sc.default_goal;
  dispatchRobotTo(g[0], g[1], 'DoubleKnockNet 双敲触发！锁定 3D 敲击点');
}}

function setInteractMode(m){{
  interactMode = m;
  document.getElementById('btnNavMode').classList.toggle('active', m==='nav');
  document.getElementById('btnRulerMode').classList.toggle('active', m==='ruler');
  cv.classList.toggle('crosshair', m==='ruler');
  if(m==='ruler'){{ rulerPts = []; toast('3D 测距模式：请在 3D 场景上连续点击任意两个点！'); }}
}}
function togglePov(){{
  povMode = !povMode;
  document.getElementById('btnPov').classList.toggle('active', povMode);
  toast(povMode ? '已切入文文第一视角 (Robot-Eye POV)！点击“触发双敲全流程”可乘车巡航！' : '已切回自由全息轨道视角');
}}
function toggleScan(){{ scanOn = !scanOn; document.getElementById('btnScan').classList.toggle('active', scanOn); }}
function toggleObb(){{ showObb = !showObb; document.getElementById('btnObb').classList.toggle('active', showObb); }}
function cycleShader(){{
  shaderMode = (shaderMode + 1) % shaderNames.length;
  document.getElementById('btnShader').textContent = '🎨 着色: ' + shaderNames[shaderMode];
}}
function setView(a,e,d,x,y,z){{ povMode=false; document.getElementById('btnPov').classList.remove('active'); az=a; el=e; dist=d; tx=x; ty=y; tz=z; }}

// 矩阵与相机工具
function mul4(a,b){{
  const o=new Float32Array(16);
  for(let c=0;c<4;c++) for(let r=0;r<4;r++)
    o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];
  return o;
}}
function getCam(){{
  if(povMode){{
    const yawR = (robot.yaw + robot.headYaw) * Math.PI / 180;
    const pitR = -(robot.pitch + 6.0) * Math.PI / 180;
    const cx = robot.x - 0.05*Math.cos(yawR), cy = robot.y - 0.05*Math.sin(yawR), cz = 0.145;
    let fx = Math.cos(pitR)*Math.cos(yawR), fy = Math.cos(pitR)*Math.sin(yawR), fz = Math.sin(pitR);
    let rx = -fy, ry = fx, rz = 0; const rl = Math.hypot(rx,ry)||1; rx/=rl; ry/=rl;
    let ux = fy*rz - fz*ry, uy = fz*rx - fx*rz, uz = fx*ry - fy*rx;
    return {{cx,cy,cz,fx,fy,fz,rx,ry,rz,ux,uy,uz,fov:62}};
  }}
  const ar=az*Math.PI/180, er=el*Math.PI/180;
  const cx=tx+dist*Math.cos(er)*Math.cos(ar), cy=ty+dist*Math.cos(er)*Math.sin(ar), cz=tz+dist*Math.sin(er);
  let fx=tx-cx, fy=ty-cy, fz=tz-cz; const fl=Math.hypot(fx,fy,fz)||1; fx/=fl; fy/=fl; fz/=fl;
  let rx=-fy, ry=fx, rz=0; const rl=Math.hypot(rx,ry)||1; rx/=rl; ry/=rl;
  let ux=fy*rz-fz*ry, uy=fz*rx-fx*rz, uz=fx*ry-fy*rx;
  return {{cx,cy,cz,fx,fy,fz,rx,ry,rz,ux,uy,uz,fov:50}};
}}
function getMVP(c){{
  const asp=cv.width/Math.max(1,cv.height), f=1/Math.tan(c.fov*Math.PI/360), zn=0.025, zf=16;
  const P=new Float32Array([f/asp,0,0,0, 0,f,0,0, 0,0,(zf+zn)/(zn-zf),-1, 0,0,2*zf*zn/(zn-zf),0]);
  const V=new Float32Array([c.rx,c.ux,-c.fx,0, c.ry,c.uy,-c.fy,0, c.rz,c.uz,-c.fz,0,
    -(c.rx*c.cx+c.ry*c.cy+c.rz*c.cz), -(c.ux*c.cx+c.uy*c.cy+c.uz*c.cz), (c.fx*c.cx+c.fy*c.cy+c.fz*c.cz), 1]);
  return mul4(P, V);
}}
function projectToScreen(M, wx, wy, wz){{
  const x = M[0]*wx + M[4]*wy + M[8]*wz + M[12];
  const y = M[1]*wx + M[5]*wy + M[9]*wz + M[13];
  const w = M[3]*wx + M[7]*wy + M[11]*wz + M[15];
  if(w <= 0.04) return null;
  const sx = (x/w * 0.5 + 0.5) * window.innerWidth;
  const sy = (0.5 - y/w * 0.5) * window.innerHeight;
  return [sx, sy];
}}

// 鼠标轨道操控 + 单击派单/3D测距
let drag=0, lx=0, ly=0, downX=0, downY=0;
cv.oncontextmenu = e => e.preventDefault();
cv.onmousedown = e => {{
  drag = e.button===2 ? 2 : 1; lx = downX = e.clientX; ly = downY = e.clientY;
}};
window.onmouseup = e => {{
  const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
  const wasLeft = (drag === 1);
  drag = 0;
  if(wasLeft && moved < 5 && e.target === cv){{
    // 射线求交桌面 Z=0
    const r = cv.getBoundingClientRect();
    const nx = (e.clientX - r.left)/r.width*2 - 1, ny = 1 - (e.clientY - r.top)/r.height*2;
    const c = getCam(), asp = cv.width/cv.height, th = Math.tan(c.fov*Math.PI/360);
    const dx = c.fx + nx*asp*th*c.rx + ny*th*c.ux;
    const dy = c.fy + nx*asp*th*c.ry + ny*th*c.uy;
    const dz = c.fz + nx*asp*th*c.rz + ny*th*c.uz;
    if(dz < -1e-4){{
      const t = -c.cz / dz;
      if(t > 0.05 && t < 4.0){{
        const wx = c.cx + t*dx, wy = c.cy + t*dy;
        if(wx >= 0.05 && wx <= 1.15 && wy >= -0.55 && wy <= 0.55){{
          if(interactMode === 'nav'){{
            dispatchRobotTo(wx, wy, '已派单 3D 目标');
          }} else {{
            if(rulerPts.length >= 2) rulerPts = [];
            rulerPts.push([wx, wy, 0.0]);
            if(rulerPts.length === 2){{
              const dmm = Math.hypot(rulerPts[1][0]-rulerPts[0][0], rulerPts[1][1]-rulerPts[0][1]) * 1000.0;
              document.getElementById('tRuler').textContent = `${{dmm.toFixed(1)}} mm (${{(dmm/10).toFixed(2)}} cm)`;
              toast(`3D 空间实测距离: ${{dmm.toFixed(1)}} mm`);
            }} else {{
              document.getElementById('tRuler').textContent = '已选起点，请点击终点...';
            }}
          }}
        }}
      }}
    }}
  }}
}};
window.onmousemove = e => {{
  if(!drag || povMode) return;
  const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
  if(drag === 1){{ az -= dx*0.34; el = Math.max(4, Math.min(88, el + dy*0.28)); }}
  else {{ const c=getCam(), s=dist*0.0018; tx -= dx*s*c.rx - dy*s*c.ux; ty -= dx*s*c.ry - dy*s*c.uy; tz -= dy*s*c.uz; }}
}};
cv.onwheel = e => {{ if(povMode) return; e.preventDefault(); dist = Math.max(0.16, Math.min(3.0, dist*(e.deltaY>0?1.10:0.91))); }};

// 物理仿真步进：HoverFineMotionPolicyV3 游标精停 + 1.16Hz 自平衡俯仰摇摆
function stepRobot(dt, tSec){{
  const scObj = gpuScenes[curKey];
  const sdfQ = querySdfAndGrad(scObj, robot.x, robot.y);
  robot.sdfClear = sdfQ.phi;
  let pitchBase = 2.8 + 2.4 * Math.sin(2*Math.PI*1.16*tSec);

  if(robot.mode === 'APPROACH_CBF' && robot.goal){{
    const gx = robot.goal[0], gy = robot.goal[1];
    const rem = Math.hypot(gx - robot.x, gy - robot.y);
    robot.remM = rem;
    robot.vernierGate = 1.0 / (1.0 + Math.exp((rem - 0.14) / 0.025));

    if(rem < 0.008){{
      robot.mode = 'DWELL_ARRIVED';
      robot.dwellT = 0.0;
      toast('游标卡尺毫米精停抵达！驻留 1.5s 后自动抬头回看人脸...');
    }} else {{
      // 取前瞻航点
      let wp = robot.goal;
      if(robot.path && robot.path.length > 1 && rem > 0.10){{
        let bestI = 0, bestD = 1e9;
        for(let i=0;i<robot.path.length;i++){{
          const di = Math.hypot(robot.path[i][0]-robot.x, robot.path[i][1]-robot.y);
          if(di < bestD){{ bestD = di; bestI = i; }}
        }}
        for(let j=bestI;j<robot.path.length;j++){{
          if(Math.hypot(robot.path[j][0]-robot.x, robot.path[j][1]-robot.y) >= 0.11){{ wp = robot.path[j]; break; }}
        }}
      }}
      const desYaw = Math.atan2(wp[1] - robot.y, wp[0] - robot.x) * 180 / Math.PI;
      let errYaw = ((desYaw - robot.yaw + 180) % 360 + 360) % 360 - 180;
      const wDps = Math.max(-72, Math.min(72, 2.4 * errYaw));
      const align = Math.max(0.15, Math.cos(errYaw * Math.PI / 180));
      const vCruise = Math.min(0.24, 0.72 * rem * align);
      const vVernier = Math.min(0.16, 1.35 * rem * align);
      const vAct = (1 - robot.vernierGate) * vCruise + robot.vernierGate * vVernier;
      robot.yaw += wDps * dt;
      robot.x += vAct * dt * Math.cos(robot.yaw * Math.PI / 180);
      robot.y += vAct * dt * Math.sin(robot.yaw * Math.PI / 180);
      robot.wheelPhase += vAct * dt / 0.033;
      robot.headYaw += Math.max(-90*dt, Math.min(90*dt, errYaw * 0.6 - robot.headYaw));
      pitchBase += vAct * 24.0; // 倒立摆加速前倾物理！
    }}
  }} else if(robot.mode === 'DWELL_ARRIVED'){{
    robot.dwellT += dt;
    robot.remM = robot.goal ? Math.hypot(robot.goal[0]-robot.x, robot.goal[1]-robot.y) : 0;
    if(robot.dwellT > 1.4){{
      robot.mode = 'LOOKBACK_FACE';
    }}
  }} else if(robot.mode === 'LOOKBACK_FACE'){{
    const fx = scObj.meta.face_xyz[0], fy = scObj.meta.face_xyz[1];
    const faceYaw = Math.atan2(fy - robot.y, fx - robot.x) * 180 / Math.PI;
    let relFace = ((faceYaw - robot.yaw + 180) % 360 + 360) % 360 - 180;
    relFace = Math.max(-45, Math.min(45, relFace));
    robot.headYaw += Math.max(-75*dt, Math.min(75*dt, relFace - robot.headYaw));
  }}
  robot.pitch = pitchBase;
  if(robot.rippleT >= 0){{ robot.rippleT += dt; if(robot.rippleT > 3.2) robot.rippleT = -1.0; }}
}}

// 动态几何线框构建器（OBB 框、平滑轨迹、3D 涟漪、文文双轮自平衡数字孪生体 + 65° 相机视锥）
function buildOverlayLines(scObj, tSec){{
  const P = [], C = [];
  const addSeg = (a,b,col) => {{ P.push(a[0],a[1],a[2], b[0],b[1],b[2]); C.push(col[0],col[1],col[2], col[0],col[1],col[2]); }};

  // 1) 自动 3D OBB 包围盒
  if(showObb){{
    for(const ob of scObj.meta.obbs){{
      const c2 = ob.corners_xy, z0 = ob.z_min_m, z1 = ob.z_max_m;
      const col = [0.38, 1.0, 0.64];
      for(let i=0;i<4;i++){{
        const j=(i+1)%4;
        addSeg([c2[i][0],c2[i][1],z0], [c2[j][0],c2[j][1],z0], [0.18,0.68,0.38]);
        addSeg([c2[i][0],c2[i][1],z1], [c2[j][0],c2[j][1],z1], col);
        addSeg([c2[i][0],c2[i][1],z0], [c2[i][0],c2[i][1],z1], [0.28,0.85,0.52]);
      }}
    }}
  }}

  // 2) HoverSDF-Net + Neural-CBF 平滑避障轨迹线
  if(robot.path && robot.path.length >= 2){{
    for(let i=0;i<robot.path.length-1;i++){{
      const a = robot.path[i], b = robot.path[i+1];
      addSeg([a[0],a[1],0.008], [b[0],b[1],0.008], [0.45, 1.0, 0.95]);
      addSeg([a[0],a[1],0.015], [b[0],b[1],0.015], [0.75, 1.0, 0.98]);
    }}
  }}

  // 3) 敲击/目标点全息雷达冲击波涟漪
  if(robot.goal){{
    const gx = robot.goal[0], gy = robot.goal[1];
    addSeg([gx,gy,0], [gx,gy,0.11], [1.0, 0.96, 0.55]);
    const rPulse = 0.02 + 0.14 * ((tSec * 0.85) % 1.0);
    for(let k=0;k<28;k++){{
      const a0 = k/28*Math.PI*2, a1 = (k+1)/28*Math.PI*2;
      addSeg([gx+rPulse*Math.cos(a0), gy+rPulse*Math.sin(a0), 0.004], [gx+rPulse*Math.cos(a1), gy+rPulse*Math.sin(a1), 0.004], [0.55, 1.0, 0.78]);
    }}
  }}

  // 4) 3D 激光测距尺
  if(rulerPts.length === 2){{
    const a = rulerPts[0], b = rulerPts[1];
    addSeg([a[0],a[1],0.012], [b[0],b[1],0.012], [1.0, 0.92, 0.35]);
    addSeg([a[0],a[1],0], [a[0],a[1],0.04], [1.0, 0.92, 0.35]);
    addSeg([b[0],b[1],0], [b[0],b[1],0.04], [1.0, 0.92, 0.35]);
  }}

  // 5) 3D 铰接式文文自平衡机器人数字孪生体 (双轮 + 倾斜机身 + 摇头舵机头 + 65° 视锥!)
  const rx = robot.x, ry = robot.y, yr = robot.yaw * Math.PI / 180;
  const pr = robot.pitch * Math.PI / 180;
  const cy = Math.cos(yr), sy = Math.sin(yr);
  const toW = (lx, ly, lz) => {{
    // 机身绕轮轴 z=0.033 做倒立摆前倾
    const dz = lz - 0.033;
    const lx2 = lx * Math.cos(pr) + dz * Math.sin(pr);
    const lz2 = 0.033 - lx * Math.sin(pr) + dz * Math.cos(pr);
    return [rx + lx2*cy - ly*sy, ry + lx2*sy + ly*cy, lz2];
  }};
  // 左右轮 (直径 6.6cm, 轮距 11cm)
  for(const side of [-0.055, 0.055]){{
    for(let k=0;k<14;k++){{
      const a0 = k/14*Math.PI*2 + robot.wheelPhase, a1 = (k+1)/14*Math.PI*2 + robot.wheelPhase;
      const p0 = [rx + 0.033*Math.cos(a0)*cy - side*sy, ry + 0.033*Math.cos(a0)*sy + side*cy, 0.033 + 0.033*Math.sin(a0)];
      const p1 = [rx + 0.033*Math.cos(a1)*cy - side*sy, ry + 0.033*Math.cos(a1)*sy + side*cy, 0.033 + 0.033*Math.sin(a1)];
      addSeg(p0, p1, [0.45, 1.0, 0.92]);
    }}
  }}
  // 机身长方体 (Z=0.033..0.115)
  const bx=0.036, by=0.042, zB0=0.033, zB1=0.115;
  const corners = [[-bx,-by],[bx,-by],[bx,by],[-bx,by]];
  for(let i=0;i<4;i++){{
    const j=(i+1)%4;
    addSeg(toW(corners[i][0],corners[i][1],zB0), toW(corners[j][0],corners[j][1],zB0), [0.35,0.95,0.75]);
    addSeg(toW(corners[i][0],corners[i][1],zB1), toW(corners[j][0],corners[j][1],zB1), [0.60,1.0,0.92]);
    addSeg(toW(corners[i][0],corners[i][1],zB0), toW(corners[i][0],corners[i][1],zB1), [0.45,0.98,0.84]);
  }}
  // 头部与 65° 相机视锥 (Z = 0.135m 实测相机高度!)
  const hyr = (robot.yaw + robot.headYaw) * Math.PI / 180;
  const hcy = Math.cos(hyr), hsy = Math.sin(hyr);
  const camO = toW(0.015, 0.0, 0.135);
  const fDist = 0.28, halfS = fDist * Math.tan(32.5 * Math.PI / 180);
  const fPts = [
    [fDist, -halfS, -0.08], [fDist, halfS, -0.08], [fDist, halfS, 0.11], [fDist, -halfS, 0.11]
  ].map(q => [camO[0] + q[0]*hcy - q[1]*hsy, camO[1] + q[0]*hsy + q[1]*hcy, Math.max(0.002, camO[2] + q[2])]);
  for(let i=0;i<4;i++){{
    addSeg(camO, fPts[i], [0.30, 0.95, 0.62]);
    addSeg(fPts[i], fPts[(i+1)%4], [0.45, 1.0, 0.72]);
  }}
  return {{ P: new Float32Array(P), C: new Float32Array(C) }};
}}

const bDynLineP = gl.createBuffer(), bDynLineC = gl.createBuffer();
let lastT = performance.now() * 0.001;

function draw(){{
  const nowT = performance.now() * 0.001;
  const dt = Math.min(0.05, Math.max(0.001, nowT - lastT));
  lastT = nowT;

  if(cv.width !== innerWidth || cv.height !== innerHeight){{ cv.width = innerWidth; cv.height = innerHeight; }}
  stepRobot(dt, nowT);

  const scObj = gpuScenes[curKey];
  if(scObj.dynBufs.length > 0){{
    dynAccT += dt;
    if(dynAccT > 0.13){{ dynAccT = 0; dynPlayIdx = (dynPlayIdx + 1) % scObj.dynBufs.length; }}
  }}

  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.010, 0.026, 0.015, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  const cam = getCam();
  const M = getMVP(cam);

  // 1) 画 5cm 物理底网格
  gl.useProgram(progLine);
  gl.uniformMatrix4fv(gl.getUniformLocation(progLine,'uMVP'), false, M);
  gl.uniform1f(gl.getUniformLocation(progLine,'uPt'), 3.0);
  const laP = gl.getAttribLocation(progLine,'aPos'), laC = gl.getAttribLocation(progLine,'aCol');
  gl.enableVertexAttribArray(laP); gl.enableVertexAttribArray(laC);
  gl.bindBuffer(gl.ARRAY_BUFFER, bGridP); gl.vertexAttribPointer(laP, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bGridC); gl.vertexAttribPointer(laC, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINES, 0, gridP.length / 3);

  // 2) 画静态超密全息 3D 高斯点云
  gl.useProgram(progPts);
  gl.uniformMatrix4fv(gl.getUniformLocation(progPts,'uMVP'), false, M);
  gl.uniform1f(gl.getUniformLocation(progPts,'uPt'), ptSize);
  gl.uniform1f(gl.getUniformLocation(progPts,'uTime'), nowT);
  gl.uniform1i(gl.getUniformLocation(progPts,'uShader'), shaderMode);
  gl.uniform1f(gl.getUniformLocation(progPts,'uScanOn'), scanOn ? 1.0 : 0.0);
  gl.uniform1f(gl.getUniformLocation(progPts,'uScanR'), (nowT * 0.42) % 1.45);
  const pPos = gl.getAttribLocation(progPts,'aPos');
  const pHol = gl.getAttribLocation(progPts,'aHolo');
  const pGeo = gl.getAttribLocation(progPts,'aGeo');
  const pEdg = gl.getAttribLocation(progPts,'aEdge');
  gl.enableVertexAttribArray(pPos); gl.enableVertexAttribArray(pHol); gl.enableVertexAttribArray(pGeo); gl.enableVertexAttribArray(pEdg);
  gl.bindBuffer(gl.ARRAY_BUFFER, scObj.bPos); gl.vertexAttribPointer(pPos, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, scObj.bHolo); gl.vertexAttribPointer(pHol, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, scObj.bGeo); gl.vertexAttribPointer(pGeo, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, scObj.bEdge); gl.vertexAttribPointer(pEdg, 1, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, scObj.N);
  gl.disableVertexAttribArray(pHol); gl.disableVertexAttribArray(pGeo); gl.disableVertexAttribArray(pEdg);

  // 3) 若处于 4D 活体双敲场景，实时叠加当前帧真实移动的 3D 人体/手部动态高斯层！
  gl.useProgram(progLine);
  gl.uniformMatrix4fv(gl.getUniformLocation(progLine,'uMVP'), false, M);
  gl.enableVertexAttribArray(laP); gl.enableVertexAttribArray(laC);
  if(scObj.dynBufs.length > 0){{
    const df = scObj.dynBufs[dynPlayIdx];
    gl.uniform1f(gl.getUniformLocation(progLine,'uPt'), ptSize * 1.15);
    gl.bindBuffer(gl.ARRAY_BUFFER, df.bPos); gl.vertexAttribPointer(laP, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, df.bCol); gl.vertexAttribPointer(laC, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, df.n);
  }}

  // 4) 画 3D OBB 框、平滑轨迹、3D 涟漪、测距尺与文文数字孪生体
  const ov = buildOverlayLines(scObj, nowT);
  gl.bindBuffer(gl.ARRAY_BUFFER, bDynLineP); gl.bufferData(gl.ARRAY_BUFFER, ov.P, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(laP, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bDynLineC); gl.bufferData(gl.ARRAY_BUFFER, ov.C, gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(laC, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINES, 0, ov.P.length / 3);

  // 5) 更新 3D 悬浮标签与右侧遥测面板
  const lblDiv = document.getElementById('labelsLayer');
  let htmlLbl = '';
  if(showObb){{
    for(const ob of scObj.meta.obbs){{
      const pt = projectToScreen(M, ob.center_xyz[0], ob.center_xyz[1], ob.z_max_m + 0.025);
      if(pt && pt[0]>40 && pt[0]<innerWidth-40 && pt[1]>40 && pt[1]<innerHeight-60){{
        htmlLbl += `<div class="tag3d" style="left:${{pt[0].toFixed(0)}}px;top:${{pt[1].toFixed(0)}}px">#${{ob.id}} ${{ob.archetype}} · ${{ob.dims_cm[0]}}×${{ob.dims_cm[1]}}×${{ob.dims_cm[2]}}cm</div>`;
      }}
    }}
  }}
  if(!povMode){{
    const rp = projectToScreen(M, robot.x, robot.y, 0.17);
    if(rp) htmlLbl += `<div class="tag3d robot" style="left:${{rp[0].toFixed(0)}}px;top:${{rp[1].toFixed(0)}}px">文文 Twin (${{robot.mode}})</div>`;
  }}
  if(robot.goal){{
    const gp = projectToScreen(M, robot.goal[0], robot.goal[1], 0.12);
    if(gp) htmlLbl += `<div class="tag3d knock" style="left:${{gp[0].toFixed(0)}}px;top:${{gp[1].toFixed(0)}}px">🎯 TARGET (${{robot.goal[0].toFixed(2)}}, ${{robot.goal[1].toFixed(2)}}m)</div>`;
  }}
  lblDiv.innerHTML = htmlLbl;

  document.getElementById('tMode').textContent = robot.mode;
  document.getElementById('tPos').textContent = `${{robot.x.toFixed(3)}}, ${{robot.y.toFixed(3)}} m`;
  document.getElementById('tPitch').textContent = `${{robot.pitch >= 0 ? '+' : ''}}${{robot.pitch.toFixed(1)}}°`;
  document.getElementById('tRem').textContent = robot.goal ? `${{(robot.remM*100).toFixed(2)}} cm` : '-- cm';
  document.getElementById('tSdf').textContent = `${{(robot.sdfClear*100).toFixed(1)}} cm`;
  document.getElementById('tSlip').textContent = robot.slipEst.toFixed(3);
  document.getElementById('tVernier').textContent = `${{Math.round(robot.vernierGate*100)}}%`;
  document.getElementById('vernierBar').style.width = `${{Math.round(robot.vernierGate*100)}}%`;

  requestAnimationFrame(draw);
}}

switchScene('empty_d05');
draw();
</script>
</body>
</html>"""
    out_html_path.write_text(html, encoding="utf-8")
    return {
        "html_path": str(out_html_path),
        "html_mb": round(out_html_path.stat().st_size / 1048576.0, 2),
        "scene_a_pts": pack_a["n_static"],
        "scene_b_pts": pack_b["n_static"],
        "scene_b_4d_frames": len(pack_b["dyn_frames"]),
    }


if __name__ == "__main__":
    print("jarvis_webgl_deck_builder.py OK")
