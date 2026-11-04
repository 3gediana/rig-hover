# -*- coding: utf-8 -*-
"""bench_pct.py — 真机单步推理耗时：p0/p10/p50 分位数 + /bench 同步计时。

和 ``bench.py`` 的区别：这边只做一件事——给出一份**可复现的分位数**，并且把
``/status`` 的分段耗时（norm/proj/cell/heads）和 ``/bench`` 的同步循环一起打印，
这样"变快了"能落到具体的段上，而不是只有一个总数。

用法::

    set PY=D:\\smartcar\\.venv\\Scripts\\python.exe
    %PY% bench_pct.py --label new --blob model_slot0_w8a8.blob --upload
    %PY% bench_pct.py --label f32                     :: 只测当前激活的模型

判据口径：``cfc.us`` 是 ``cfc_task.c`` 里 ``cfc_model_forward()`` 的一次真实
计时段（``s_last_us``），采样 60 次取分位数；``/bench?n=N`` 是同步循环，没有
任务调度抖动，用来交叉验证。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

DEFAULT_HOST = "http://192.168.137.50"
#: 机器人只在局域网内：绝不能让系统代理接手（否则 502）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _req(host, path, data=None, timeout=30):
    req = urllib.request.Request(host + path, data=data,
                                 method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/octet-stream")
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def get_json(host, path, retries=4, delay=3.0):
    last = None
    for k in range(retries):
        try:
            return json.loads(_req(host, path))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if k + 1 < retries:
                time.sleep(delay)
    raise RuntimeError("GET %s 失败: %r" % (path, last))


def post(host, path, body=b"", retries=4, delay=4.0):
    last = None
    for k in range(retries):
        try:
            return _req(host, path, data=body)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if k + 1 < retries:
                time.sleep(delay)
    raise RuntimeError("POST %s 失败: %r" % (path, last))


def pct(vals, q):
    """最近秩分位数（q 取 0..100），不插值——样本就 60 个，插值只会假装精确。"""
    if not vals:
        return -1
    v = sorted(vals)
    k = int(round((len(v) - 1) * q / 100.0))
    return v[min(max(k, 0), len(v) - 1)]


def wait_ready(host, timeout=45.0):
    t0 = time.time()
    while True:
        try:
            get_json(host, "/status", retries=1)
            return True
        except Exception as exc:  # noqa: BLE001
            if time.time() - t0 > timeout:
                print("机器人不可达: %r" % (exc,))
                return False
            time.sleep(2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="真机 cfc.us 分位数 + /bench 同步计时")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--label", default="bench")
    ap.add_argument("--blob", default=None, help="要上传到 slot 0 的 blob")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--select", action="store_true", help="切到 src=cfc&slot=0")
    ap.add_argument("--n", type=int, default=60, help="cfc.us 采样次数")
    ap.add_argument("--gap", type=float, default=0.15, help="采样间隔（秒）")
    ap.add_argument("--bench-n", type=int, default=500, help="/bench 的循环次数")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if not wait_ready(args.host):
        return 2

    res = {"label": args.label}
    if args.upload:
        if not args.blob:
            print("--upload 需要 --blob")
            return 1
        data = Path(args.blob).read_bytes()
        r = json.loads(post(args.host, "/model?slot=0", data))
        print("上传: %s（%d 字节）" % (json.dumps(r), len(data)))
        res["upload"] = r
        if not r.get("ok"):
            return 3
    if args.select:
        r = post(args.host, "/mode?src=cfc&slot=0")
        print("切源: %s" % r)
        res["select"] = r

    st = get_json(args.host, "/status")
    c = st.get("cfc", {}) or {}
    res["q"] = c.get("q", 0)
    res["slot"] = c.get("slot", -1)
    res["heap"] = st.get("heap", 0)
    print("模型: slot=%s q=%s heap=%s" % (c.get("slot"), c.get("q"), st.get("heap")))

    # 先把采样打满，确保 cfc_task 真在跑
    time.sleep(1.0)
    vals, splits = [], []
    for _ in range(args.n):
        c = (get_json(args.host, "/status").get("cfc", {}) or {})
        us = c.get("us", -1)
        if isinstance(us, (int, float)) and us > 0:
            vals.append(int(us))
            splits.append((c.get("n_us", 0), c.get("p_us", 0),
                           c.get("s_us", 0), c.get("h_us", 0)))
        time.sleep(args.gap)

    res["n"] = len(vals)
    res["p0"] = pct(vals, 0)
    res["p10"] = pct(vals, 10)
    res["p50"] = pct(vals, 50)
    res["p90"] = pct(vals, 90)
    res["max"] = pct(vals, 100)
    print("cfc.us  n=%d  p0=%d  p10=%d  p50=%d  p90=%d  max=%d"
          % (len(vals), res["p0"], res["p10"], res["p50"], res["p90"], res["max"]))
    if splits:
        for i, name in enumerate(("norm", "proj", "cell", "heads")):
            col = [s[i] for s in splits if s[i] >= 0]
            if col:
                res["split_" + name] = {"p0": pct(col, 0), "p50": pct(col, 50)}
                print("  分段 %-5s p0=%3d p50=%3d us" % (name, pct(col, 0), pct(col, 50)))

    try:
        b = get_json(args.host, "/bench?n=%d" % args.bench_n)
        res["bench"] = b
        print("/bench: %s" % json.dumps(b))
    except Exception as exc:  # noqa: BLE001
        print("/bench 失败: %r" % (exc,))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
