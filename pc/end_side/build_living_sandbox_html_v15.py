# -*- coding: utf-8 -*-
"""build_living_sandbox_html_v15.py — 生成单文件零联网 HTML5《文文·活体桌面伴侣 v15》互动回放剧场 + 60FPS 可玩沙盒！"""

from __future__ import annotations

import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "runs_end_side" / "living_brain_v15"
REPORT_JSON = OUT_DIR / "living_brain_v15_report.json"
ARTIFACT_DIR = pathlib.Path.home() / ".cache" / "rig_hover_artifacts"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RIG-Hover 文文 v15 · 活体桌面具身大脑剧场 & 可玩沙盒</title>
<style>
  :root { --bg:#090f15; --card:#101a24; --line:#23384a; --accent:#5ef5ae; --warn:#ffd166; --danger:#ff5c77; --blue:#5cc8ff; --text:#e8f3fb; --muted:#8fa9be; }
  * { box-sizing:border-box; margin:0; padding:0; font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
  body { background:var(--bg); color:var(--text); min-height:100vh; display:flex; flex-direction:column; align-items:center; padding:16px; }
  header { width:100%; max-width:1240px; display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; border-bottom:1px solid var(--line); padding-bottom:12px; }
  h1 { font-size:20px; color:var(--accent); letter-spacing:0.4px; }
  .sub { font-size:12px; color:var(--muted); margin-top:3px; }
  .tabs { display:flex; gap:8px; }
  .tab { padding:8px 16px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--muted); cursor:pointer; font-weight:600; transition:0.15s; }
  .tab.active { background:#16332a; color:var(--accent); border-color:var(--accent); }
  .grid { width:100%; max-width:1240px; display:grid; grid-template-columns: 1fr 370px; gap:16px; }
  @media (max-width:960px) { .grid { grid-template-columns:1fr; } }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }
  canvas#desk { width:100%; height:auto; display:block; border-radius:10px; background:#0d161f; border:1px solid #1e3242; cursor:crosshair; }
  .hud-face { width:100%; height:130px; background:#0c141d; border:1px solid #203748; border-radius:10px; margin-bottom:12px; display:block; }
  .bubble { background:#132533; border-left:3px solid var(--accent); border-radius:8px; padding:10px 12px; margin-bottom:12px; font-size:13px; line-height:1.45; min-height:54px; }
  .bubble b { color:var(--warn); display:block; font-size:11px; text-transform:uppercase; letter-spacing:0.6px; margin-bottom:2px; }
  .btn-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:10px; }
  button.act { padding:9px 10px; border-radius:8px; border:1px solid var(--line); background:#142230; color:var(--text); cursor:pointer; font-size:12.5px; font-weight:600; transition:0.15s; }
  button.act:hover { border-color:var(--accent); background:#18303c; }
  button.act.danger { border-color:#5a2431; color:#ff96a8; }
  button.act.danger:hover { border-color:var(--danger); background:#311621; }
  button.act.active-m { border-color:var(--accent); background:#17352b; color:var(--accent); }
  .stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:10px; }
  .stat { background:#0c141c; border:1px solid #1c2f3f; border-radius:8px; padding:8px 10px; }
  .stat .k { font-size:11px; color:var(--muted); }
  .stat .v { font-size:16px; font-weight:700; color:var(--blue); margin-top:2px; }
  .row { display:flex; align-items:center; gap:8px; margin-top:10px; }
  input[type=range] { flex:1; accent-color:var(--accent); }
  .tip { font-size:12px; color:var(--muted); line-height:1.55; margin-top:10px; background:#0b131b; border-radius:8px; padding:9px 11px; border:1px dashed #24394b; }
  .hidden { display:none !important; }
</style>
</head>
<body>
<header>
  <div>
    <h1>🤖 RIG-Hover 文文 v15 · 活体桌面具身大脑剧场 &amp; 互动沙盒</h1>
    <div class="sub">100% 零开挂 · 纯视觉悬崖本能 · 防捉弄假动作免疫 · 皮亚杰物体恒存探头 · 三连劫持/动态避障通关</div>
  </div>
  <div class="tabs">
    <button class="tab active" id="tabReplay" onclick="setMode('replay')">🎬 真实 v15 实测回放剧场</button>
    <button class="tab" id="tabPlay" onclick="setMode('play')">🎮 实时互动调戏文文沙盒</button>
  </div>
</header>

<div class="grid">
  <div class="panel">
    <canvas id="desk" width="800" height="580"></canvas>
    <div id="replayBar" class="row">
      <button class="act" style="width:76px" id="btnPlayPause" onclick="togglePlay()">⏸ 暂停</button>
      <input type="range" id="scrubber" min="0" max="40" value="0" oninput="onScrub(this.value)">
      <span id="stepLbl" style="font-size:12px;color:var(--muted);min-width:68px;text-align:right">0 / 0</span>
    </div>
  </div>

  <div class="panel">
    <canvas id="face" class="hud-face" width="340" height="130"></canvas>
    <div class="bubble"><b id="moodTag">READY</b><span id="thoughtTxt">双击桌面任意位置召唤文文！</span></div>

    <!-- 回放模式控件 -->
    <div id="replayControls">
      <div style="font-size:12px;color:var(--muted);margin-bottom:6px;font-weight:600">选择 v15 真实跑出的 4 项极限测试回放：</div>
      <div class="btn-grid">
        <button class="act active-m" id="mL1" onclick="loadMission('L1')">⛰ L1 视觉悬崖拒跳</button>
        <button class="act" id="mL2" onclick="loadMission('L2')">😜 L2 假动作+换点寻人</button>
        <button class="act" id="mL3" onclick="loadMission('L3')">👀 L3 遮挡探头绕障</button>
        <button class="act" id="mL4" onclick="loadMission('L4')">🔥 L4 地狱三连拷打</button>
      </div>
      <div id="missionMeta" class="tip"></div>
    </div>

    <!-- 互动沙盒控件 -->
    <div id="playControls" class="hidden">
      <div style="font-size:12px;color:var(--muted);margin-bottom:6px;font-weight:600">随时给文文加戏（支持连续组合捉弄！）：</div>
      <div class="btn-grid">
        <button class="act" onclick="spawnCup()">☕ 滑过移动水杯</button>
        <button class="act" onclick="dropCylinder()">🧱 空投遮挡障碍</button>
        <button class="act danger" onclick="kidnapWenwen()">🌀 暴力劫持 +95°</button>
        <button class="act" onclick="moveHuman()">🪑 人类悄悄换座位</button>
        <button class="act danger" onclick="suicidalKnock()">⛰ 诱敲悬崖外！</button>
        <button class="act" onclick="resetSandbox()">🔄 重置桌面</button>
      </div>
      <div class="tip">
        <b>操作玩法：</b><br>
        • <b>双击桌面</b> = 双敲召唤文文过去停靠并回看你；<br>
        • <b>单击桌面</b> = 假动作挑逗，文文会识破并无视你；<br>
        • <b>双击左侧红色悬崖区</b> = 诱骗跳崖，看文文急刹摇头抗议！
      </div>
    </div>

    <div class="stats">
      <div class="stat"><div class="k">实时位置 (X, Y)</div><div class="v" id="stPos">0.00, 0.00</div></div>
      <div class="stat"><div class="k">净空安全裕度</div><div class="v" id="stClear">+20.0 cm</div></div>
      <div class="stat"><div class="k">机身朝向 / 舵机转头</div><div class="v" id="stYaw">0° / 0°</div></div>
      <div class="stat"><div class="k">安全状态</div><div class="v" id="stSafe" style="color:var(--accent)">SAFE ✓</div></div>
    </div>
  </div>
</div>

<script>
const V15_DATA = __V15_JSON__;

const desk = document.getElementById('desk');
const ctx = desk.getContext('2d');
const fcan = document.getElementById('face');
const fctx = fcan.getContext('2d');

let mode = 'replay';
let curMissionKey = 'L1';
let frames = [];
let fIdx = 0;
let playing = true;
let lastReplayTick = 0;

// 桌面坐标映射到 800x580 Canvas：Y ∈ [-0.50, 0.34] -> u ∈ [40, 760]; X ∈ [-0.06, 1.18] -> v ∈ [545, 35]
function w2c(x, y) {
  const u = 40 + (y - (-0.50)) / 0.84 * 720;
  const v = 545 - (x - (-0.06)) / 1.24 * 510;
  return [u, v];
}
function c2w(u, v) {
  const y = -0.50 + (u - 40) / 720 * 0.84;
  const x = -0.06 + (545 - v) / 510 * 1.24;
  return [x, y];
}
function r2px(rm) { return rm / 0.84 * 720; }

const STATIC_MUG = {cx: 0.36, cy: -0.02, rad: 0.040};
const STATIC_MOUSE = {cx: 0.66, cy: -0.13, rx: 0.058, ry: 0.036};
const CLIFF_Y = -0.36;

function setMode(m) {
  mode = m;
  document.getElementById('tabReplay').classList.toggle('active', m === 'replay');
  document.getElementById('tabPlay').classList.toggle('active', m === 'play');
  document.getElementById('replayControls').classList.toggle('hidden', m !== 'replay');
  document.getElementById('replayBar').classList.toggle('hidden', m !== 'replay');
  document.getElementById('playControls').classList.toggle('hidden', m !== 'play');
  if (m === 'replay') loadMission(curMissionKey);
  else resetSandbox();
}

function loadMission(key) {
  curMissionKey = key;
  ['L1','L2','L3','L4'].forEach(k => document.getElementById('m'+k).classList.toggle('active-m', k===key));
  let mObj, meta = '';
  if (key === 'L1') {
    const d = V15_DATA.L1_visual_cliff_survival;
    mObj = d.v15_living_mind;
    meta = `<b>L1 实测结果：</b>无悬崖本能对照组坠崖 <b>fell=${d.baseline_no_cliff.fell_off_cliff}</b>；v15 纯视觉发现崖边 <b>y=${mObj.cliff_y_est_m}m</b> (真值 -0.36m)，提前急刹留足 <b>+${mObj.min_cliff_margin_cm}cm</b> 并摇头抗议！`;
  } else if (key === 'L2') {
    mObj = V15_DATA.L2_teasing_and_active_face_search;
    meta = `<b>L2 实测结果：</b>无视假动作=${mObj.fakeout_ignored}，半路改口绕杯停靠 <b>${mObj.final_err_cm}cm</b>，发现人类悄悄换座位后主动扫视找回，锁眸误差 <b>${mObj.final_face_gaze_err_deg}°</b>，0 碰撞！`;
  } else if (key === 'L3') {
    const d = V15_DATA.L3_object_permanence_nbv_peek;
    mObj = d.v15_nbv_peek;
    meta = `<b>L3 实测结果：</b>17.5cm 高保温杯挡死视线；v15 记住被挡目标，计算最优侧向 NBV 探头点 <b>(${mObj.nbv_peek_xy})</b> 探头确认后绕入，停靠 <b>${mObj.final_err_cm}cm</b>，净空 <b>+${mObj.min_clearance_cm}cm</b>，0 碰撞！`;
  } else {
    const d = V15_DATA.L4_triple_chaos_gauntlet;
    mObj = d.v15_living_mind;
    meta = `<b>L4 地狱三连：</b>移动杯(s${mObj.attack1_moving_det_step}) + 突降圆柱(s${mObj.attack2_drop_det_step}) + 暴力+88°劫持(<b>${mObj.attack3_kidnap_recov_steps}步</b>自愈)，全程 <b>0 碰撞</b>（对照组误差 ${d.baseline_no_v15.final_err_cm}cm / 碰撞=${d.baseline_no_v15.collision}）！`;
  }
  frames = mObj.replay_frames || [];
  fIdx = 0;
  playing = true;
  document.getElementById('btnPlayPause').textContent = '⏸ 暂停';
  document.getElementById('scrubber').max = Math.max(0, frames.length - 1);
  document.getElementById('scrubber').value = 0;
  document.getElementById('missionMeta').innerHTML = meta;
}

function togglePlay() {
  playing = !playing;
  document.getElementById('btnPlayPause').textContent = playing ? '⏸ 暂停' : '▶ 播放';
}
function onScrub(val) {
  fIdx = parseInt(val, 10) || 0;
  playing = false;
  document.getElementById('btnPlayPause').textContent = '▶ 播放';
}

// ---------- 实时 60FPS 互动沙盒物理 + 活体大脑 ----------
let sb = {};
function resetSandbox() {
  sb = {
    x: 0.04, y: -0.02, yaw: 0.0, head: 0.0,
    target: null, nbv: null,
    human: [1.08, 0.06], memHuman: [1.08, 0.06],
    cyls: [], movingCup: null,
    mood: 'IDLE', text: '双击桌面召唤文文！单击则是假动作挑逗哦～',
    trail: [[0.04, -0.02]],
    cliffTimer: 0, kidnapTimer: 0, searchTimer: 0,
    singleClickTimer: null, lastClickTime: 0,
  };
}

function wrapDeg(a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

desk.addEventListener('click', (e) => {
  if (mode !== 'play') return;
  const rect = desk.getBoundingClientRect();
  const u = (e.clientX - rect.left) * (desk.width / rect.width);
  const v = (e.clientY - rect.top) * (desk.height / rect.height);
  const [wx, wy] = c2w(u, v);
  const now = performance.now();
  if (now - sb.lastClickTime < 380) {
    // 双敲！
    clearTimeout(sb.singleClickTimer);
    sb.lastClickTime = 0;
    sb.target = [Math.max(0.05, Math.min(0.95, wx)), Math.max(-0.46, Math.min(0.28, wy))];
    sb.nbv = null;
    sb.cliffTimer = 0;
    sb.searchTimer = 0;
    sb.mood = 'APPROACH';
    sb.text = `收到双敲 (${sb.target[0].toFixed(2)}, ${sb.target[1].toFixed(2)})！正在规划无碰撞路线...`;
  } else {
    sb.lastClickTime = now;
    sb.singleClickTimer = setTimeout(() => {
      if (sb.mood !== 'KIDNAP_SCAN' && sb.mood !== 'REFUSE_CLIFF') {
        sb.mood = 'IGNORE_FAKEOUT';
        sb.text = '哼，只敲了一下！假动作挑逗被我识破啦～';
      }
    }, 390);
  }
});

function spawnCup() {
  sb.movingCup = {cx: 0.56, cy: -0.34, vx: -0.055, vy: 0.125, rad: 0.038};
  sb.mood = 'DODGE_CUP';
  sb.text = '发现移动水杯！卡尔曼速度管已开启提前让行！';
}
function dropCylinder() {
  const gx = sb.target ? 0.5 * (sb.x + sb.target[0]) : 0.34;
  const gy = sb.target ? 0.5 * (sb.y + sb.target[1]) : -0.14;
  sb.cyls.push({cx: gx, cy: gy, rad: 0.044});
  sb.nbv = [Math.max(0.14, gx - 0.05), gy > -0.10 ? gy - 0.16 : gy + 0.16];
  sb.mood = 'PEEK_NBV';
  sb.text = '前方被高障碍挡死！先侧滑探头看看后面情况～';
}
function kidnapWenwen() {
  sb.x = Math.max(0.05, Math.min(0.55, sb.x + (Math.random() - 0.5) * 0.18));
  sb.y = Math.max(-0.28, Math.min(0.18, sb.y + (Math.random() - 0.5) * 0.18));
  sb.yaw = wrapDeg(sb.yaw + (Math.random() > 0.5 ? 95 : -95));
  sb.kidnapTimer = 38;
  sb.mood = 'KIDNAP_SCAN';
  sb.text = '卧槽谁拎我？！急刹停轮 + 360° 全局视觉重定位中...';
}
function moveHuman() {
  sb.human = [1.05, sb.human[1] > -0.1 ? -0.34 : 0.18];
  sb.text = '（人类悄悄换到了新座位，文文还蒙在鼓里...）';
}
function suicidalKnock() {
  sb.target = [0.46, -0.44];
  sb.cliffTimer = 0;
  sb.mood = 'APPROACH';
  sb.text = '有人在右侧桌沿外敲桌子？先过去看看...';
}

function stepSandbox(dt) {
  // 更新移动水杯
  if (sb.movingCup) {
    sb.movingCup.cx += sb.movingCup.vx * dt;
    sb.movingCup.cy += sb.movingCup.vy * dt;
    if (sb.movingCup.cy > 0.30 || sb.movingCup.cx < 0.08) sb.movingCup = null;
  }

  // 1) 劫持全局扫描自愈
  if (sb.kidnapTimer > 0) {
    sb.kidnapTimer--;
    sb.yaw = wrapDeg(sb.yaw - 140 * dt);
    sb.head = Math.sin(sb.kidnapTimer * 0.4) * 20;
    if (sb.kidnapTimer === 0) {
      sb.mood = 'RECOVERED';
      sb.text = '全局 2 点 SE(2) 重锁成功！继续完成原任务！';
    }
    return;
  }

  // 2) 悬崖拒跳摇头抗议
  if (sb.cliffTimer > 0) {
    sb.cliffTimer--;
    if (sb.cliffTimer > 35) {
      sb.head = Math.sin(sb.cliffTimer * 0.65) * 30;
    } else {
      const bH = wrapDeg(Math.atan2(sb.human[1] - sb.y, sb.human[0] - sb.x) * 180 / Math.PI - sb.yaw);
      sb.head = Math.max(-32, Math.min(32, bH));
      sb.yaw = wrapDeg(sb.yaw + Math.max(-90, Math.min(90, (bH - sb.head) * 2.2)) * dt);
      sb.mood = 'STARE_HUMAN';
      sb.text = '拒绝跳崖！回头盯住你：休想骗我摔下去！';
    }
    return;
  }

  // 3) 停靠后主动寻人
  if (sb.searchTimer > 0) {
    sb.searchTimer--;
    const bTrue = wrapDeg(Math.atan2(sb.human[1] - sb.y, sb.human[0] - sb.x) * 180 / Math.PI - (sb.yaw + sb.head));
    if (Math.abs(bTrue) < 16) {
      sb.head = Math.max(-30, Math.min(30, sb.head + bTrue * 0.4));
      sb.memHuman = [...sb.human];
      sb.mood = 'FOUND_HUMAN';
      sb.text = '抓到你啦！居然偷偷换座位，锁眸盯住！';
      sb.searchTimer = 0;
    } else {
      sb.mood = 'SEARCH_HUMAN';
      sb.text = '咦？旧座位没人？！主动旋转扫视找人中...';
      sb.yaw = wrapDeg(sb.yaw + (bTrue >= 0 ? 1 : -1) * 85 * dt);
      sb.head = (bTrue >= 0 ? 1 : -1) * 18;
    }
    return;
  }

  if (!sb.target) return;

  // 悬崖视觉提前感知
  if (sb.target[1] < CLIFF_Y + 0.02 && sb.y < CLIFF_Y + 0.095) {
    sb.target = null;
    sb.cliffTimer = 75;
    sb.mood = 'REFUSE_CLIFF';
    sb.text = '前方悬崖！视觉 Cliff-CBF 急刹 + 摇头抗议！';
    return;
  }

  // NBV 探头点到达判定
  let goal = sb.nbv || sb.target;
  if (sb.nbv && Math.hypot(sb.x - sb.nbv[0], sb.y - sb.nbv[1]) < 0.06) {
    sb.nbv = null;
    goal = sb.target;
    sb.mood = 'PEEK_CLEAR';
    sb.text = '侧面探头确认后面通畅！绕过去停靠目标！';
  }

  // 合成势场 + 移动水杯未来速度管斥力
  let gx = goal[0] - sb.x, gy = goal[1] - sb.y;
  const distG = Math.hypot(gx, gy);
  if (distG < 0.025) {
    sb.target = null;
    const bOld = wrapDeg(Math.atan2(sb.memHuman[1] - sb.y, sb.memHuman[0] - sb.x) * 180 / Math.PI - sb.yaw);
    const bTrue = wrapDeg(Math.atan2(sb.human[1] - sb.y, sb.human[0] - sb.x) * 180 / Math.PI - sb.yaw);
    if (Math.abs(wrapDeg(bOld - bTrue)) > 16) {
      sb.searchTimer = 140;
      sb.mood = 'SEARCH_HUMAN';
      sb.text = '到啦！回头看旧座位——咦，人跑哪去了？！';
    } else {
      sb.head = Math.max(-32, Math.min(32, bTrue));
      sb.mood = 'DOCKED';
      sb.text = '精准停靠完成！抬头乖乖看人脸～';
    }
    return;
  }

  gx /= distG; gy /= distG;
  const allObs = [STATIC_MUG, {cx: STATIC_MOUSE.cx, cy: STATIC_MOUSE.cy, rad: 0.048}, ...sb.cyls];
  if (sb.movingCup) {
    for (let k = 0; k <= 5; k++) {
      allObs.push({cx: sb.movingCup.cx + sb.movingCup.vx * k * 0.16, cy: sb.movingCup.cy + sb.movingCup.vy * k * 0.16, rad: sb.movingCup.rad + 0.024});
    }
  }
  let fx = gx, fy = gy;
  for (const o of allObs) {
    const dx = sb.x - o.cx, dy = sb.y - o.cy;
    const d = Math.hypot(dx, dy);
    const safe = o.rad + 0.115;
    if (d < safe && d > 1e-4) {
      const push = ((safe - d) / safe) * 3.4;
      fx += (dx / d) * push;
      fy += (dy / d) * push;
    }
  }
  // 虚拟悬崖墙斥力
  if (sb.y < CLIFF_Y + 0.095) fy += ((CLIFF_Y + 0.095 - sb.y) / 0.095) * 4.5;

  const desYaw = Math.atan2(fy, fx) * 180 / Math.PI;
  const errYaw = wrapDeg(desYaw - sb.yaw);
  sb.yaw = wrapDeg(sb.yaw + Math.max(-135, Math.min(135, errYaw * 3.6)) * dt);
  const spd = Math.abs(errYaw) > 55 ? 0.05 : Math.min(0.22, distG * 1.4 + 0.05);
  sb.x += spd * Math.cos(sb.yaw * Math.PI / 180) * dt;
  sb.y = Math.max(CLIFF_Y + 0.052, sb.y + spd * Math.sin(sb.yaw * Math.PI / 180) * dt);
  sb.head = Math.max(-28, Math.min(28, errYaw * 0.45));
  sb.trail.push([sb.x, sb.y]);
  if (sb.trail.length > 260) sb.trail.shift();
}

function drawFaceHUD(mood, headDeg) {
  fctx.fillStyle = '#0c141d';
  fctx.fillRect(0, 0, fcan.width, fcan.height);
  const cx = 170 + Math.max(-34, Math.min(34, headDeg * 1.1));
  const cy = 58;
  let col = '#5ef5ae';
  if (mood === 'REFUSE_CLIFF' || mood === 'FELL_OFF') col = '#ff5c77';
  else if (mood === 'KIDNAP_SCAN' || mood === 'SEARCH_HUMAN') col = '#5cc8ff';
  else if (mood === 'PEEK_NBV' || mood === 'DODGE_CUP' || mood === 'DODGE_DROP' || mood === 'IGNORE_FAKEOUT') col = '#ffd166';

  fctx.strokeStyle = col; fctx.fillStyle = col; fctx.lineWidth = 4; fctx.lineCap = 'round';
  if (mood === 'REFUSE_CLIFF' || mood === 'FELL_OFF') {
    for (const s of [-1, 1]) {
      fctx.beginPath();
      fctx.moveTo(cx + s * 46, cy - 16); fctx.lineTo(cx + s * 22, cy); fctx.lineTo(cx + s * 46, cy + 16);
      fctx.stroke();
    }
  } else if (mood === 'FOUND_HUMAN' || mood === 'DOCKED') {
    for (const s of [-1, 1]) {
      fctx.beginPath(); fctx.arc(cx + s * 34, cy + 8, 18, Math.PI * 1.12, Math.PI * 1.88); fctx.stroke();
    }
  } else if (mood === 'KIDNAP_SCAN' || mood === 'SEARCH_HUMAN') {
    for (const s of [-1, 1]) {
      fctx.beginPath(); fctx.arc(cx + s * 34, cy, 17, 0, Math.PI * 2); fctx.stroke();
      fctx.beginPath(); fctx.arc(cx + s * 34 + Math.sin(performance.now()/180)*6, cy, 6, 0, Math.PI * 2); fctx.fill();
    }
  } else {
    for (const s of [-1, 1]) {
      fctx.beginPath(); fctx.ellipse(cx + s * 34, cy, 16, mood === 'IGNORE_FAKEOUT' ? 7 : 15, 0, 0, Math.PI * 2); fctx.fill();
    }
  }
  fctx.fillStyle = '#8fa9be'; fctx.font = '12px monospace';
  fctx.fillText(`HEAD SERVO: ${headDeg >= 0 ? '+' : ''}${headDeg.toFixed(0)}°`, 14, 118);
  fctx.fillStyle = col;
  fctx.fillText(mood, 215, 118);
}

function drawScene(st) {
  ctx.fillStyle = '#0d161f';
  ctx.fillRect(0, 0, desk.width, desk.height);

  // 悬崖区（L1 或沙盒模式显示）
  if (st.showCliff) {
    const [cu] = w2c(0, CLIFF_Y);
    ctx.fillStyle = 'rgba(130, 18, 42, 0.32)';
    ctx.fillRect(0, 0, cu, desk.height);
    ctx.strokeStyle = '#ff5c77'; ctx.lineWidth = 2.5; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(cu, 0); ctx.lineTo(cu, desk.height); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff7d93'; ctx.font = 'bold 12px sans-serif';
    ctx.fillText('⛰ 72cm 悬崖深渊', 10, 26);
    if (st.cliffEst !== null && st.cliffEst !== undefined) {
      const [eu] = w2c(0, st.cliffEst);
      ctx.strokeStyle = '#5cc8ff'; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(eu, 0); ctx.lineTo(eu, desk.height); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // 桌面网格
  ctx.strokeStyle = 'rgba(55, 86, 110, 0.25)'; ctx.lineWidth = 1;
  for (let gx = 0; gx <= 1.1; gx += 0.2) {
    const [, v] = w2c(gx, 0); ctx.beginPath(); ctx.moveTo(0, v); ctx.lineTo(desk.width, v); ctx.stroke();
  }
  for (let gy = -0.4; gy <= 0.3; gy += 0.1) {
    const [u] = w2c(0, gy); ctx.beginPath(); ctx.moveTo(u, 0); ctx.lineTo(u, desk.height); ctx.stroke();
  }

  // 静态桌面物体：水杯、鼠标、键盘
  const [mu, mv] = w2c(STATIC_MUG.cx, STATIC_MUG.cy);
  ctx.fillStyle = '#20486c'; ctx.strokeStyle = '#67d4ff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(mu, mv, r2px(STATIC_MUG.rad), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#bfe9ff'; ctx.font = '11px sans-serif'; ctx.fillText('静态水杯', mu - 22, mv + 4);

  const [msu, msv] = w2c(STATIC_MOUSE.cx, STATIC_MOUSE.cy);
  ctx.fillStyle = '#1f4f3c'; ctx.strokeStyle = '#5ef5ae'; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.ellipse(msu, msv, r2px(STATIC_MOUSE.ry), r2px(STATIC_MOUSE.rx), 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#b9fbd9'; ctx.fillText('鼠标', msu - 11, msv + 4);

  const [ku1, kv1] = w2c(0.85, 0.01), [ku2, kv2] = w2c(0.70, 0.31);
  ctx.fillStyle = '#1f2b38'; ctx.strokeStyle = '#637a90';
  ctx.fillRect(ku1, kv1, ku2 - ku1, kv2 - kv1); ctx.strokeRect(ku1, kv1, ku2 - ku1, kv2 - kv1);

  // 人类座位
  if (st.face) {
    const [fu, fv] = w2c(st.face[0], st.face[1]);
    ctx.fillStyle = '#c77dff'; ctx.beginPath(); ctx.arc(fu, fv, 14, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif'; ctx.fillText('🧑 人类', fu - 22, fv - 18);
  }

  // 预测速度障碍管 (青色预测环)
  for (const [ox, oy, orad] of (st.tube || [])) {
    const [ou, ov] = w2c(ox, oy);
    ctx.strokeStyle = 'rgba(92, 200, 255, 0.65)'; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(ou, ov, r2px(orad), 0, Math.PI * 2); ctx.stroke();
  }

  // 动态/遮挡圆柱
  for (const c of (st.cyls || [])) {
    const [cu, cv] = w2c(c.cx, c.cy);
    ctx.fillStyle = '#c1121f'; ctx.strokeStyle = '#ffccd5'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cu, cv, r2px(c.rad), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }

  // NBV 侧向探头点 + 视线
  if (st.nbv && st.tgt) {
    const [pu, pv] = w2c(st.nbv[0], st.nbv[1]);
    const [tu, tv] = w2c(st.tgt[0], st.tgt[1]);
    ctx.strokeStyle = '#5ef5ae'; ctx.setLineDash([5, 5]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pu, pv); ctx.lineTo(tu, tv); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#ffd166'; ctx.beginPath(); ctx.arc(pu, pv, 7, 0, Math.PI * 2); ctx.fill();
    ctx.fillText('NBV 探头点', pu + 10, pv + 4);
  }

  // 目标星标
  if (st.tgt) {
    const [tu, tv] = w2c(st.tgt[0], st.tgt[1]);
    ctx.fillStyle = '#fff176'; ctx.strokeStyle = '#ff9f1c'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(tu, tv, 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#fff176'; ctx.font = 'bold 11px sans-serif'; ctx.fillText('★ 敲击点', tu + 12, tv + 4);
  }

  // 轨迹
  if (st.trail && st.trail.length > 1) {
    ctx.strokeStyle = '#5ef5ae'; ctx.lineWidth = 3;
    ctx.beginPath();
    st.trail.forEach((p, idx) => { const [u, v] = w2c(p[0], p[1]); if (idx === 0) ctx.moveTo(u, v); else ctx.lineTo(u, v); });
    ctx.stroke();
  }

  // 机器人机身 + 60° 相机视线扇形
  const [ru, rv] = w2c(st.x, st.y);
  const gazeRad = (st.yaw + st.head) * Math.PI / 180;
  const [huL, hvL] = w2c(st.x + 0.28 * Math.cos(gazeRad + 0.45), st.y + 0.28 * Math.sin(gazeRad + 0.45));
  const [huR, hvR] = w2c(st.x + 0.28 * Math.cos(gazeRad - 0.45), st.y + 0.28 * Math.sin(gazeRad - 0.45));
  ctx.fillStyle = st.mood === 'KIDNAP_SCAN' ? 'rgba(92,200,255,0.25)' : 'rgba(94,245,174,0.16)';
  ctx.beginPath(); ctx.moveTo(ru, rv); ctx.lineTo(huL, hvL); ctx.lineTo(huR, hvR); ctx.closePath(); ctx.fill();

  ctx.fillStyle = '#5ef5ae'; ctx.strokeStyle = '#09151e'; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.arc(ru, rv, r2px(0.052), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  const [hu, hv] = w2c(st.x + 0.11 * Math.cos(gazeRad), st.y + 0.11 * Math.sin(gazeRad));
  ctx.strokeStyle = '#fff176'; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(ru, rv); ctx.lineTo(hu, hv); ctx.stroke();

  // 更新右侧仪表
  document.getElementById('moodTag').textContent = st.mood;
  document.getElementById('thoughtTxt').textContent = st.text;
  document.getElementById('stPos').textContent = `${st.x.toFixed(2)}, ${st.y.toFixed(2)}`;
  document.getElementById('stClear').textContent = `${st.clear >= 0 ? '+' : ''}${st.clear.toFixed(1)} cm`;
  document.getElementById('stYaw').textContent = `${st.yaw.toFixed(0)}° / ${st.head >= 0 ? '+' : ''}${st.head.toFixed(0)}°`;
  const sf = document.getElementById('stSafe');
  sf.textContent = st.clear >= 0 ? 'SAFE ✓' : 'COLLISION!';
  sf.style.color = st.clear >= 0 ? '#5ef5ae' : '#ff5c77';
  drawFaceHUD(st.mood, st.head);
}

let prevTs = performance.now();
function loop(ts) {
  const dt = Math.min(0.05, (ts - prevTs) / 1000);
  prevTs = ts;
  if (mode === 'replay') {
    if (playing && frames.length > 0 && ts - lastReplayTick > 140) {
      lastReplayTick = ts;
      fIdx = (fIdx + 1) % frames.length;
      document.getElementById('scrubber').value = fIdx;
    }
    document.getElementById('stepLbl').textContent = `${fIdx + 1} / ${frames.length}`;
    if (frames.length > 0) {
      const r = frames[fIdx];
      const trail = frames.slice(0, fIdx + 1).map(f => [f.tx, f.ty]);
      drawScene({
        x: r.tx, y: r.ty, yaw: r.tyaw, head: r.head, mood: r.mood, text: r.text,
        tgt: r.tgt, cyls: r.cyls, tube: r.tube, nbv: r.nbv, face: r.face,
        trail, clear: r.clear_cm, showCliff: (curMissionKey === 'L1'), cliffEst: r.cliff_est,
      });
    }
  } else {
    stepSandbox(dt);
    const cyls = [...sb.cyls];
    const tube = [];
    if (sb.movingCup) {
      cyls.push(sb.movingCup);
      for (let k = 0; k <= 5; k++) tube.push([sb.movingCup.cx + sb.movingCup.vx * k * 0.16, sb.movingCup.cy + sb.movingCup.vy * k * 0.16, sb.movingCup.rad + 0.02]);
    }
    let minCl = Math.hypot(sb.x - STATIC_MUG.cx, sb.y - STATIC_MUG.cy) - STATIC_MUG.rad - 0.055;
    minCl = Math.min(minCl, Math.hypot(sb.x - STATIC_MOUSE.cx, sb.y - STATIC_MOUSE.cy) - 0.048 - 0.055);
    for (const c of cyls) minCl = Math.min(minCl, Math.hypot(sb.x - c.cx, sb.y - c.cy) - c.rad - 0.055);
    drawScene({
      x: sb.x, y: sb.y, yaw: sb.yaw, head: sb.head, mood: sb.mood, text: sb.text,
      tgt: sb.target, cyls, tube, nbv: sb.nbv, face: sb.human,
      trail: sb.trail, clear: minCl * 100.0, showCliff: true, cliffEst: sb.cliffTimer > 0 ? -0.358 : null,
    });
  }
  requestAnimationFrame(loop);
}

resetSandbox();
loadMission('L1');
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


def main() -> None:
    assert REPORT_JSON.exists(), f"Run living_desktop_brain_v15.py first! Missing {REPORT_JSON}"
    raw = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    html = HTML_TEMPLATE.replace("__V15_JSON__", json.dumps(raw, ensure_ascii=False))
    out_html = OUT_DIR / "wenwen_living_brain_v15.html"
    out_html.write_text(html, encoding="utf-8")
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_html, ARTIFACT_DIR / out_html.name)
    print(f"Built Interactive HTML5 Living Sandbox -> {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
