"""RIG-Hover capture studio -- a tiny local web console for labelled data capture.

Why this exists
===============

Collecting a labelled dataset needed the operator and the recorder to agree on
"start now". Announcing a recording in chat raced the operator's reading of the
message, so bursts landed mid-action and had to be discarded. This server moves
the trigger to the operator: the task list is fixed here (so the labels are
canonical and never depend on timing), the operator presses Start, and the
session directory is named from the task plus the metadata they entered.

The task list is the scenario, not a menu
=========================================

``TASKS`` below holds exactly what SCENARIO-v2.md needs -- gaze at a face, hold
a fist, knock twice. Gesture classes such as ``wave`` / ``pointing`` /
``thumbs_up`` belong to the earlier ACK scenario ("wave to call me"), which
scenario v2 does not use; listing them here made the operator perform actions
the deployment never consumes, so they are deliberately absent.

Recording the *transition*, not the pose
========================================

Every task is a repeating action rather than a held pose. A held pose gives the
decision network one label for twenty identical frames; a repeating action gives
it the enter/exit transitions, which is what "the person made a fist" and "the
person knocked twice" actually are. ``hint`` says so on every task.

Countdown
=========

``/api/start`` enters a ``COUNTDOWN_S`` pause before the first frame is written,
so the operator can get into position after pressing the button. Frames captured
during the countdown are dropped: a frame written before the action starts is a
frame whose label is "nothing was happening yet", which is exactly the noise that
made the first collection unusable.

Design
======

One capture loop polls the robot's ``GET /capture`` and fans the frame out to
both consumers -- the MJPEG preview in the browser and, while a recording is
active, the on-disk session. Sharing one capture loop matters: two independent
pollers would contend for the camera and halve the effective frame rate.

Frame budget: one round trip is ~33 ms (see LATENCY-MAP.md), so the loop runs at
10 Hz with headroom. The preview of a 240x240 frame is 3x upscaled by CSS only;
the wire stays at 240x240.

Run:
    python tools/capture_studio.py
    python tools/capture_studio.py --port 8770 --robot 192.168.137.50
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "evidence" / "sessions_v2"

#: Seconds between pressing Start and the first saved frame.
COUNTDOWN_S = 3.0

#: Canonical task list. ``id`` becomes the directory prefix, so it must stay
#: stable -- downstream training enumerates sessions by this id, not by folder
#: name, so renaming an id orphans collected data.
#:
#: These three and only these three: SCENARIO-v2.md needs a face to gaze at, a
#: fist to switch to, and two knocks to trigger the approach. The ACK gestures
#: (wave / pointing / thumbs_up / open_palm) belong to scenario v1 and are
#: deliberately not offered.
TASKS = [
    ("gaze_face", "盯脸（正常坐着，缓慢转头）",
     "坐着面对机器人，偶尔向左/向右缓慢转头再回正。让脸在画面里移动，"
     "但身体别动——要的是头部带来的目标移动"),
    ("fist", "握拳（举到胸前，反复握紧和松开）",
     "握紧举到胸前停 3 秒 → 完全松开 → 再握紧，一直反复。"
     "要的是「握上」和「松开」的转换，不是保持一个姿势不动"),
    ("knock", "敲桌子两下（连续）",
     "每 3 秒敲两下：抬起→敲→抬起→敲，两下之间有明显停顿。"
     "重点是「两下」这个节奏，敲太连成一串就废了"),
]
TASK_IDS = {t[0] for t in TASKS}
TASK_LABELS = {t[0]: t[1] for t in TASKS}


class Studio:
    """Shared capture state: one polling loop, many readers."""

    def __init__(self, robot: str, fps: float) -> None:
        self.robot = robot
        self.period = 1.0 / max(fps, 1.0)
        self.lock = threading.Lock()
        self.latest: bytes | None = None
        self.latest_at = 0.0
        self.frames_total = 0
        self.failures = 0
        self.online = False
        #: ``None`` when idle, else a dict describing the active take. ``phase``
        #: is one of ``countdown`` / ``recording`` / ``done``.
        self.take: dict | None = None
        self._last_take: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- capture loop ------------------------------------------------------
    def _fetch(self) -> bytes | None:
        try:
            conn = http.client.HTTPConnection(self.robot, 80, timeout=6)
            conn.request("GET", "/capture")
            resp = conn.getresponse()
            data = resp.read()
            conn.close()
            return data if resp.status == 200 else None
        except Exception:  # noqa: BLE001 - the UI reports the outage
            return None

    def _loop(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            jpg = self._fetch()
            now = time.monotonic()
            with self.lock:
                if jpg is None:
                    self.failures += 1
                    self.online = False
                else:
                    self.latest = jpg
                    self.latest_at = now
                    self.frames_total += 1
                    self.online = True
                    self._advance_locked(now)
            next_at += self.period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()

    def _advance_locked(self, now: float) -> None:
        take = self.take
        if take is None:
            return
        if take["phase"] == "countdown":
            left = take["t_start"] + COUNTDOWN_S - now
            if left > 0:
                return
            take["phase"] = "recording"
            take["t0"] = now
            take["count"] = 0
            return
        if take["phase"] == "recording":
            frame_idx = take["count"]
            t_s = max(0.0, now - take["t0"])
            frame_path = take["dir"] / "frames" / f"{frame_idx:05d}.jpg"
            frame_path.write_bytes(self.latest or b"")
            with (take["dir"] / "frames.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "frame_index": frame_idx,
                    "t_s": round(t_s, 4),
                    "captured_at": datetime.now().isoformat(timespec="milliseconds"),
                    "file": f"frames/{frame_idx:05d}.jpg",
                }, ensure_ascii=False) + "\n")
            take["count"] += 1
            if t_s >= take["seconds"]:
                self._finalise_locked(now)

    def _finalise_locked(self, now: float) -> None:
        take = self.take
        assert take is not None
        if take["phase"] == "done":
            return
        take["phase"] = "done"
        take["finished_at"] = now
        if not (take["dir"] / "frames.jsonl").exists():
            (take["dir"] / "frames.jsonl").write_text("", encoding="utf-8")
        meta = {
            "task": take["task"],
            "label": take["behavior_name"] or TASK_LABELS.get(take["task"], "自定义行为"),
            "behavior_name": take["behavior_name"],
            "behavior_description": take["behavior_description"],
            "scenario": "custom" if take["task"] == "custom" else "v2",
            "seconds_requested": take["seconds"],
            "countdown_s": COUNTDOWN_S,
            "seconds_actual": round(max(0.0, now - take["t0"]), 2) if take["t0"] else 0.0,
            "frames": take["count"],
            "events": take["events"],
            "distance_m": take["distance_m"],
            "facing": take["facing"],
            "light": take["light"],
            "note": take["note"],
            "robot": self.robot,
            "started_at": take["started_at"],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "frame_dir": str(Path(take["dir"].name) / "frames"),
            "frame_index_file": str(Path(take["dir"].name) / "frames.jsonl"),
            "event_file": str(Path(take["dir"].name) / "events.jsonl"),
        }
        (take["dir"] / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        row = dict(meta)
        row["dir"] = take["dir"].name
        take["meta"] = row
        with (OUT_ROOT / "manifest.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._last_take = take

    # -- public API --------------------------------------------------------
    def start_recording(self, task: str, seconds: float,
                        distance_m: float | None, facing: str, light: str,
                        note: str, behavior_name: str = "",
                        behavior_description: str = "") -> dict:
        if task not in TASK_IDS and task != "custom":
            return {"ok": False, "err": f"unknown task {task!r}"}
        behavior_name = behavior_name.strip()[:80]
        behavior_description = behavior_description.strip()[:1000]
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return {"ok": False, "err": "录制时长必须是数字"}
        if task == "custom" and not behavior_name:
            return {"ok": False, "err": "请填写行为名称"}
        if not math.isfinite(seconds) or not 5 <= seconds <= 300:
            return {"ok": False, "err": "录制时长需在 5 到 300 秒之间"}
        if distance_m is not None:
            try:
                distance_m = float(distance_m)
            except (TypeError, ValueError):
                return {"ok": False, "err": "距离必须是数字"}
            if not math.isfinite(distance_m) or not 0 <= distance_m <= 5:
                return {"ok": False, "err": "距离需在 0 到 5 米之间"}
        with self.lock:
            if self.take is not None and self.take["phase"] != "done":
                return {"ok": False, "err": "already recording"}
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dist = "" if distance_m is None else f"_d{distance_m:g}"
            slug = "custom" if task == "custom" else task
            name = f"{slug}{dist}_{facing}_{light}_{stamp}"
            d = OUT_ROOT / name
            d.mkdir(parents=True, exist_ok=False)
            (d / "frames").mkdir()
            self.take = {
                "task": task, "behavior_name": behavior_name,
                "behavior_description": behavior_description,
                "seconds": float(seconds), "distance_m": distance_m,
                "facing": facing, "light": light, "note": note,
                "dir": d, "count": 0, "events": [],
                "phase": "countdown", "t_start": time.monotonic(),
                "t0": None,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            (d / "events.jsonl").write_text("", encoding="utf-8")
            return {"ok": True, "dir": name, "countdown_s": COUNTDOWN_S}

    def mark_event(self, label: str = "event") -> dict:
        label = label.strip()[:120] or "event"
        with self.lock:
            take = self.take
            if take is None or take["phase"] != "recording":
                return {"ok": False, "err": "录制进行中才能标记事件"}
            event = {
                "event_index": len(take["events"]),
                "label": label,
                "t_s": round(max(0.0, time.monotonic() - take["t0"]), 4),
                "frame_index": max(0, take["count"] - 1),
                "marked_at": datetime.now().isoformat(timespec="milliseconds"),
            }
            take["events"].append(event)
            with (take["dir"] / "events.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            return {"ok": True, "event": event}

    def stop_recording(self) -> dict:
        with self.lock:
            take = self.take
            if take is None or take["phase"] == "done":
                return {"ok": False, "err": "not recording"}
            self._finalise_locked(time.monotonic())
            return {"ok": True, "frames": take["count"]}

    def snapshot(self) -> dict:
        with self.lock:
            take = self.take or self._last_take
            out = {
                "online": self.online,
                "frames_total": self.frames_total,
                "failures": self.failures,
                "countdown_s": COUNTDOWN_S,
                "take": None,
            }
            if take is not None:
                now = time.monotonic()
                phase = take["phase"]
                if phase == "countdown":
                    remaining = max(0.0, take["t_start"] + COUNTDOWN_S - now)
                elif phase == "recording":
                    remaining = max(0.0, take["seconds"] - (now - take["t0"]))
                else:
                    remaining = 0.0
                elapsed = max(0.0, now - take["t0"]) if take["t0"] else 0.0
                out["take"] = {
                    "task": take["task"],
                    "label": take["behavior_name"] or TASK_LABELS.get(take["task"], "自定义行为"),
                    "behavior_name": take["behavior_name"],
                    "events": list(take["events"]),
                    "dir": take["dir"].name, "phase": phase,
                    "frames": take["count"],
                    "seconds": take["seconds"],
                    "remaining_s": round(remaining, 1),
                    "progress": (0.0 if phase == "countdown" else
                                 min(1.0, elapsed / max(take["seconds"], 0.1))),
                    "distance_m": take["distance_m"],
                    "facing": take["facing"], "light": take["light"],
                    "note": take["note"],
                }
            return out

    def frame(self) -> bytes | None:
        with self.lock:
            return self.latest


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RIG-Hover 采集台 v2</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--mut:#8d96a8;
      --acc:#4c8dff;--ok:#3ddc84;--warn:#ffb020;--bad:#ff5a5f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 "Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px}
h1{font-size:18px;margin:0 0 4px}
.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:18px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.preview{position:relative;background:#000;border-radius:10px;overflow:hidden;
         display:flex;align-items:center;justify-content:center;min-height:300px}
.preview img{width:100%;max-width:720px;aspect-ratio:1/1;object-fit:contain;display:block}
.badge{position:absolute;top:10px;left:10px;padding:4px 10px;border-radius:999px;
       font-size:12px;font-weight:600;background:#000a;backdrop-filter:blur(4px)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.on{background:var(--ok)}.off{background:var(--bad)}
.rec{background:var(--bad);animation:pulse 1s infinite}
@keyframes pulse{50%{opacity:.25}}
.count{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
       background:#000b;font-size:120px;font-weight:800;color:#fff;
       text-shadow:0 4px 24px #000;pointer-events:none}
.count.show{display:flex}
label{display:block;font-size:12px;color:var(--mut);margin:12px 0 4px}
select,input,textarea{width:100%;background:#0c0e13;border:1px solid var(--line);
    color:var(--fg);border-radius:8px;padding:9px 10px;font:inherit}
textarea{resize:vertical;min-height:52px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
button{border:0;border-radius:9px;padding:11px 16px;font:inherit;font-weight:600;
       cursor:pointer;transition:.15s}
.go{background:var(--acc);color:#fff;width:100%;margin-top:14px}
.go:hover{filter:brightness(1.12)}
.go:disabled{background:#2a3242;color:#66708a;cursor:not-allowed}
.stop{background:var(--bad);color:#fff;width:100%;margin-top:10px}
.stop:disabled{background:#2a3242;color:#66708a;cursor:not-allowed}
.bar{height:6px;background:#0c0e13;border-radius:999px;overflow:hidden;margin-top:12px}
.bar>i{display:block;height:100%;width:0;background:var(--ok);transition:width .2s}
.hint{font-size:12px;color:var(--warn);background:#2a2210;border-radius:8px;
      padding:8px 10px;margin-top:10px;line-height:1.6}
.stat{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-top:10px;
      flex-wrap:wrap}
.stat b{color:var(--fg);font-weight:600}
.list{max-height:230px;overflow:auto;margin-top:8px}
.item{display:flex;justify-content:space-between;gap:8px;padding:7px 0;
      border-bottom:1px solid var(--line);font-size:12px}
.item:last-child{border-bottom:0}
.item span:first-child{color:var(--mut)}
.pill{font-size:11px;padding:2px 8px;border-radius:999px;background:#222836;color:var(--mut)}
.hidden{display:none!important}
.mark-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:end}
.mark-btn{background:#39465c;color:#fff;padding:9px 12px}
</style></head><body><div class="wrap">
<h1>RIG-Hover 视频数据采集台</h1>
<div class="sub">可选内置行为，也可描述新的行为；每帧保留时间索引，录制中可标记关键事件。</div>
<div class="grid">
  <div class="card">
    <div class="preview">
      <img id="v" alt="等待画面">
      <div class="count" id="cd"></div>
      <div class="badge"><span id="dot" class="dot off"></span><span id="st">连接中…</span></div>
    </div>
    <div class="stat">
      <span>累计帧 <b id="ft">0</b></span>
      <span>失败 <b id="ff">0</b></span>
      <span>本次 <b id="fc">0</b></span>
      <span>剩余 <b id="rm">—</b></span>
    </div>
    <div class="bar"><i id="pb"></i></div>
    <div class="hint" id="hint"></div>
  </div>

  <div class="card">
    <label>任务</label>
    <select id="task"></select>
    <div class="hint" id="thint"></div>
    <div id="customFields" class="hidden">
      <label>自定义行为名称</label>
      <input id="behaviorName" maxlength="80" placeholder="例如：敲桌面后把手收回">
      <label>你要采集的动作过程</label>
      <textarea id="behaviorDescription" maxlength="1000" placeholder="描述从什么状态开始、连续动作的顺序、如何结束，以及需要覆盖的变化"></textarea>
    </div>

    <div class="row">
      <div><label>时长（秒）</label><input id="secs" type="number" value="60" min="5" max="300"></div>
      <div><label>距离（米）</label><input id="dist" type="number" value="0.5" step="0.1" min="0" max="5"></div>
    </div>
    <div class="row">
      <div><label>朝向</label><select id="facing">
        <option value="front">正对机器人</option>
        <option value="left">偏左 30°</option>
        <option value="right">偏右 30°</option></select></div>
      <div><label>光照</label><select id="light">
        <option value="day">白天窗光</option>
        <option value="night">夜间顶灯</option></select></div>
    </div>
    <label>备注（可选）</label>
    <textarea id="note" placeholder="例如：右手 / 手抬高一点 / 戴着耳机"></textarea>

    <button class="go" id="go">开始录制</button>
    <button class="stop" id="stop" disabled>停止并保存</button>
    <div id="eventFields" class="hidden">
      <div class="mark-row">
        <div><label>事件标记（录制中）</label><input id="eventLabel" maxlength="120" value="关键动作发生"></div>
        <button class="mark-btn" id="mark" type="button" disabled>标记时刻</button>
      </div>
      <div id="eventList" class="list"></div>
    </div>

    <label style="margin-top:18px">本次归档目录</label>
    <div class="item"><span id="dir">—</span></div>

    <label style="margin-top:14px">已采集</label>
    <div class="list" id="list"><div class="item"><span>尚无记录</span></div></div>
  </div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const escapeHtml=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let TASKS=[];
async function loadTasks(){
  const r=await fetch('/api/tasks'); const j=await r.json(); TASKS=j.tasks;
  $('task').innerHTML=TASKS.map(t=>`<option value="${t.id}">${t.label}</option>`).join('')+
    '<option value="custom">自定义行为…</option>';
  syncHint();
}
function syncHint(){
  const custom=$('task').value==='custom';
  $('customFields').classList.toggle('hidden',!custom);
  $('eventFields').classList.toggle('hidden',!custom);
  const t=TASKS.find(x=>x.id===$('task').value);
  $('thint').textContent=custom?'描述可观察的完整动作过程；标记按钮可记录关键时刻。':
    (t?'提示：'+t.hint:'');
  $('eventLabel').disabled=!custom;
}
$('task').addEventListener('change',syncHint);

let lastDoneDir=null;
async function tick(){
  try{
    const r=await fetch('/api/state'); const s=await r.json();
    const tk=s.take;
    if(tk){
      const busy = tk.phase!=='done';
      $('dot').className='dot '+(busy?'rec':(s.online?'on':'off'));
      if(tk.phase==='countdown'){
        const n=Math.ceil(tk.remaining_s);
        $('cd').textContent=n>0?n:'GO';
        $('cd').classList.add('show');
        $('st').textContent='倒计时';
        $('rm').textContent=Math.ceil(tk.remaining_s)+' 秒后开始';
        $('pb').style.width='0%';
        $('fc').textContent=0;
      }else if(tk.phase==='recording'){
        $('cd').classList.remove('show');
        $('st').textContent='录制中';
        $('rm').textContent=tk.remaining_s+' 秒';
        $('pb').style.width=(tk.progress*100)+'%';
        $('fc').textContent=tk.frames;
      }else{
        $('cd').classList.remove('show');
        $('st').textContent='已保存';
        $('dot').className='dot on';
        $('rm').textContent='0 秒';
        if(lastDoneDir!==tk.dir){
          lastDoneDir=tk.dir;
          $('hint').textContent='已保存 '+tk.frames+' 帧 → '+tk.dir;
          loadList();
        }
      }
      $('dir').textContent=tk.dir;
      $('go').disabled=busy; $('stop').disabled=!busy;
      $('task').disabled=busy;
      $('secs').disabled=busy; $('dist').disabled=busy;
      $('facing').disabled=busy; $('light').disabled=busy;
      $('note').disabled=busy; $('behaviorName').disabled=busy;
      $('behaviorDescription').disabled=busy;
      $('eventLabel').disabled=$('task').value!=='custom';
      $('mark').disabled=tk.phase!=='recording'||$('task').value!=='custom';
      if(tk.events){$('eventList').innerHTML=tk.events.map(e=>
        `<div class="item"><span>${escapeHtml(e.label)}</span><span>${e.t_s.toFixed(2)} s · 帧 ${e.frame_index}</span></div>`).join('');}
    }else{
      $('cd').classList.remove('show');
      $('go').disabled=false; $('stop').disabled=true;
      $('task').disabled=false; $('secs').disabled=false; $('dist').disabled=false;
      $('facing').disabled=false; $('light').disabled=false; $('note').disabled=false;
      $('behaviorName').disabled=false; $('behaviorDescription').disabled=false;
      $('eventLabel').disabled=$('task').value!=='custom'; $('mark').disabled=true;
      $('fc').textContent=0; $('rm').textContent='—'; $('pb').style.width='0%';
    }
    $('ft').textContent=s.frames_total; $('ff').textContent=s.failures;
  }catch(e){ $('st').textContent='采集台未响应'; $('dot').className='dot off'; }
}
async function loadList(){
  try{
    const r=await fetch('/api/sessions'); const j=await r.json();
    if(!j.sessions.length){ $('list').innerHTML='<div class="item"><span>尚无记录</span></div>'; return; }
    $('list').innerHTML=j.sessions.map(s=>
      `<div class="item"><span>${escapeHtml(s.label||s.task)}</span><span>${s.frames} 帧 · ${s.distance_m??'?'} m
       <span class="pill">${escapeHtml(s.facing||'')} ${escapeHtml(s.light||'')}</span></span></div>`).join('');
  }catch(e){}
}
$('go').onclick=async()=>{
  lastDoneDir=null;
  $('hint').textContent='';
  const body={task:$('task').value, seconds:parseFloat($('secs').value)||60,
    distance_m:parseFloat($('dist').value)||null, facing:$('facing').value,
    light:$('light').value, note:$('note').value,
    behavior_name:$('behaviorName').value, behavior_description:$('behaviorDescription').value};
  try{
    const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    const j=await r.json();
    if(!j.ok){ $('hint').textContent='启动失败：'+j.err; }
  }catch(e){ $('hint').textContent='启动失败：'+e; }
};
$('stop').onclick=async()=>{ await fetch('/api/stop',{method:'POST'}); loadList(); };
$('mark').onclick=async()=>{
  const r=await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({label:$('eventLabel').value})});
  const j=await r.json();
  if(!j.ok){$('hint').textContent='标记失败：'+j.err;}
  else { $('hint').textContent=`已标记 ${j.event.label} · ${j.event.t_s.toFixed(2)} 秒`; }
};

$('v').src='/stream?t='+Date.now();
loadTasks(); loadList();
setInterval(tick,250); setInterval(loadList,4000); tick();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    studio: Studio

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        pass  # the console log is the UI; silence per-request noise

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/tasks":
            self._json({"tasks": [{"id": i, "label": l, "hint": h}
                                  for (i, l, h) in TASKS]})
            return
        if path == "/api/state":
            self._json(self.studio.snapshot())
            return
        if path == "/api/sessions":
            man = OUT_ROOT / "manifest.jsonl"
            rows = []
            if man.exists():
                for line in man.read_text(encoding="utf-8").splitlines():
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
            self._json({"sessions": rows[-40:][::-1]})
            return
        if path == "/stream":
            self._stream()
            return
        if path == "/snapshot":
            jpg = self.studio.frame()
            if jpg is None:
                self._json({"err": "no frame"}, 503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
        self._json({"err": "not found"}, 404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpg = self.studio.frame()
                if jpg is None:
                    time.sleep(0.08)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode()
                                 + b"\r\n\r\n")
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._json({"ok": False, "err": "bad json"}, 400)
            return
        if path == "/api/start":
            distance = req.get("distance_m")
            self._json(self.studio.start_recording(
                task=str(req.get("task", "")),
                seconds=req.get("seconds", 60),
                distance_m=None if distance in (None, "") else distance,
                facing=str(req.get("facing") or "")[:32],
                light=str(req.get("light") or "")[:32],
                note=str(req.get("note") or "")[:500],
                behavior_name=str(req.get("behavior_name") or ""),
                behavior_description=str(req.get("behavior_description") or ""),
            ))
            return
        if path == "/api/mark":
            self._json(self.studio.mark_event(str(req.get("label") or "event")))
            return
        if path == "/api/stop":
            self._json(self.studio.stop_recording())
            return
        self._json({"err": "not found"}, 404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--robot", default="192.168.137.50")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    studio = Studio(args.robot, args.fps)
    studio.start()
    Handler.studio = studio
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"capture studio v2 on http://127.0.0.1:{args.port}  (robot {args.robot})")
    print(f"countdown {COUNTDOWN_S:.0f}s | sessions land under {OUT_ROOT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.stop()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
