# -*- coding: utf-8 -*-
"""bench.py — 真机 CfC 推理耗时实测（不依赖 /bench，那个端点有 bug）。

用法::

    set PY=D:\\smartcar\\.venv\\Scripts\\python.exe
    %PY% bench.py --label float32                     # 只测当前模型
    %PY% bench.py --label int8 --blob model_slot0_int8.blob --upload

为什么会话里要两种测法（``/status`` 的 ``us`` 是**最后一次**采样，会抖）：

* ``--poll N``：连续拉 N 次 ``/status``，取 ``cfc.us`` 的中位数/最大/最小；
* **饱和法**：把 ``cfc_hz`` 顶到 1000（固件钳到 200），读实际达到的 ``hz``。
  ``hz`` 掉到 200 以下说明单步推理把周期撑爆了，此时 ``当前周期`` 才是真实
  上界；``hz`` 稳定在 200 说明没饱和，``us`` 就是可信的单步耗时。

烧录后第一发请求可能失败，脚本会等 20 s 再试。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
DEFAULT_HOST = "http://192.168.137.50"

# 机器人只在局域网内：绝不能让系统代理（HTTP_PROXY）接手，否则 502 Bad Gateway
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _req(host, path, data=None, timeout=20):
    url = host + path
    req = urllib.request.Request(url, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def get_json(host, path, retries=3, delay=2.0):
    last = None
    for k in range(retries):
        try:
            return json.loads(_req(host, path))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if k + 1 < retries:
                time.sleep(delay)
    raise RuntimeError("GET %s 失败: %r" % (path, last))


def post(host, path, body=None, retries=3, delay=3.0):
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode()
    last = None
    for k in range(retries):
        try:
            return _req(host, path, data=body if body is not None else b"")
        except Exception as exc:  # noqa: BLE001
            last = exc
            if k + 1 < retries:
                time.sleep(delay)
    raise RuntimeError("POST %s 失败: %r" % (path, last))


def cfc_of(st):
    return st.get("cfc", {}) or {}


def snapshot(host, label=""):
    st = get_json(host, "/status")
    c = cfc_of(st)
    return {
        "label": label,
        "us": c.get("us", -1),
        "hz": c.get("hz", -1),
        "slot": c.get("slot", -1),
        "frames": c.get("frames", -1),
        "dropped": c.get("dropped", -1),
        "q": c.get("q", 0),
        "heap": st.get("heap", 0),
    }


def poll_us(host, n=15, gap=0.25):
    vals = []
    for _ in range(n):
        st = get_json(host, "/status")
        us = cfc_of(st).get("us", -1)
        if isinstance(us, (int, float)) and us > 0:
            vals.append(int(us))
        time.sleep(gap)
    return vals


def saturation(host, target_hz=1000, settle=6.0):
    """把 cfc_hz 顶高，看实际达到的 hz；返回 (achieved_hz, cfg_hz)。"""
    post(host, "/cfg", {"cfc_hz": target_hz})
    time.sleep(settle)
    st = get_json(host, "/status")
    return cfc_of(st).get("hz", -1), cfc_of(st).get("us", -1)


def restore_hz(host, hz=50):
    try:
        post(host, "/cfg", {"cfc_hz": hz})
    except Exception as exc:  # noqa: BLE001
        print("  恢复 cfc_hz 失败: %r" % (exc,))


def upload(host, blob_path, slot=0):
    data = Path(blob_path).read_bytes()
    r = post(host, "/model?slot=%d" % slot, data, retries=3, delay=4.0)
    return json.loads(r) if r.strip().startswith("{") else {"raw": r}


def main(argv=None):
    ap = argparse.ArgumentParser(description="真机 CfC 推理耗时实测")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--label", default="", help="本次测量标签（只用于打印）")
    ap.add_argument("--blob", default=None, help="要上传的 blob 路径")
    ap.add_argument("--upload", action="store_true", help="上传 --blob 到 slot 0")
    ap.add_argument("--select", action="store_true", help="切到 src=cfc&slot=0")
    ap.add_argument("--poll", type=int, default=15, help="轮询 /status 的次数")
    ap.add_argument("--no-sat", action="store_true", help="跳过饱和法")
    ap.add_argument("--json-out", default=None, help="把结果写到这个 json 文件")
    args = ap.parse_args(argv)

    host = args.host
    t0 = time.time()
    while True:
        try:
            get_json(host, "/status", retries=1)
            break
        except Exception as exc:  # noqa: BLE001
            if time.time() - t0 > 40:
                print("机器人不可达: %r" % (exc,))
                return 2
            time.sleep(2)

    if args.upload:
        if not args.blob:
            print("--upload 需要 --blob")
            return 1
        info = upload(host, args.blob)
        print("上传: %s" % json.dumps(info))
        if not info.get("ok"):
            return 3
    if args.select:
        print("切源: %s" % post(host, "/mode?src=cfc&slot=0"))

    before = snapshot(host, "before")
    print("模型槽位: %s" % json.dumps(get_json(host, "/model").get("slots")))
    print("测量前: %s" % json.dumps(before))

    us = poll_us(host, args.poll)
    res = {"label": args.label, "before": before, "us": us}
    if us:
        res["us_median"] = int(statistics.median(us))
        res["us_min"] = min(us)
        res["us_max"] = max(us)
        print("cfc.us  n=%d  median=%d  min=%d  max=%d"
              % (len(us), res["us_median"], res["us_min"], res["us_max"]))
    else:
        print("没有采到有效的 cfc.us")

    if not args.no_sat:
        hz, last_us = saturation(host)
        res["sat_hz"] = hz
        res["sat_us"] = last_us
        print("饱和法: 实际 hz=%s（<200 说明周期被推理撑爆）最后 us=%s"
              % (hz, last_us))
        us2 = poll_us(host, 6)
        if us2:
            res["sat_us_median"] = int(statistics.median(us2))
            print("饱和态 cfc.us median=%d（%s）"
                  % (res["sat_us_median"], us2))
        restore_hz(host)
        print("已把 cfc_hz 还原到 50")

    after = snapshot(host, "after")
    res["after"] = after
    print("测量后: %s" % json.dumps(after))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
