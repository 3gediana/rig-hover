# -*- coding: utf-8 -*-
"""ctools.py — 在 PC 上编 C 的小工具（MSVC 环境、编译、跑）。

``cl.exe`` 需要 ``INCLUDE``/``LIB`` 环境变量，普通 shell 里是空的，所以这里按
``cfc_host/test_parity.py`` 的做法跑一次 ``vcvars64.bat`` 把环境抓出来缓存。

用法::

    from ctools import compile_exe, run_exe, CL_EXE
    compile_exe(["dump_offsets.c"], "dump_offsets.exe")
    print(run_exe("dump_offsets.exe"))
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MSVC_ROOT = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
CL_DIR = MSVC_ROOT / r"VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
CL_EXE = CL_DIR / "cl.exe"
VCVARS = MSVC_ROOT / r"VC\Auxiliary\Build\vcvars64.bat"

#: /O2 /fp:precise —— 绝不用 /fp:fast，否则和 torch 的对拍没意义。
#: /utf-8：源文件里有中文注释，不加这条 MSVC 会按 GBK 解并报 C4819。
CFLAGS = ("/nologo", "/c", "/O2", "/fp:precise", "/W3", "/utf-8")
_ENV_CACHE = {}


def _ascii(text):
    """把编译器输出压成 ASCII（某些代码页会解出不可打印的私用区字符）。"""
    return text.encode("ascii", "replace").decode("ascii")


def msvc_env():
    """返回带 INCLUDE/LIB/PATH 的环境变量字典（缓存）。"""
    if "env" in _ENV_CACHE:
        return _ENV_CACHE["env"]
    env = dict(os.environ)
    if not CL_EXE.exists():
        raise RuntimeError("找不到 cl.exe: %s" % CL_EXE)
    if not (env.get("INCLUDE") and env.get("LIB")):
        if not VCVARS.exists():
            raise RuntimeError("找不到 vcvars64.bat: %s" % VCVARS)
        bat = HERE / "build" / "_vcvars_env.bat"
        bat.parent.mkdir(parents=True, exist_ok=True)
        bat.write_text('@echo off\r\ncall "%s" >nul 2>&1\r\nset\r\n' % VCVARS,
                       encoding="ascii")
        try:
            out = subprocess.run(["cmd.exe", "/c", str(bat)], capture_output=True,
                                 text=True, encoding="mbcs", errors="replace",
                                 cwd=str(bat.parent))
        finally:
            try:
                bat.unlink()
            except OSError:
                pass
        if out.returncode != 0 or "INCLUDE=" not in out.stdout:
            raise RuntimeError("vcvars64.bat 失败 rc=%d:\n%s\n%s"
                               % (out.returncode, out.stdout, out.stderr))
        parsed = {}
        for line in out.stdout.splitlines():
            if "=" in line and not line.startswith("="):
                k, v = line.split("=", 1)
                parsed[k.upper()] = v
        env.update(parsed)
        env["PATH"] = parsed.get("PATH", env.get("PATH", "")) + ";" + str(CL_DIR)
    _ENV_CACHE["env"] = env
    return env


def run(args, timeout=600, check=True):
    """跑一条命令（list 形式），返回 ``(stdout+stderr, returncode)``。"""
    p = subprocess.run([str(a) for a in args], cwd=str(HERE), env=msvc_env(),
                       capture_output=True, text=True,
                       encoding="mbcs" if os.name == "nt" else "utf-8",
                       errors="replace", timeout=timeout)
    text = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError("命令失败 rc=%d: %s\n%s" % (p.returncode, args, text))
    return text, p.returncode


def compile_exe(sources, exe_name, include_dirs=(), out_dir=None, extra_flags=()):
    """编译并链接出一个 exe，返回 ``(exe_path, warnings)``。

    Args:
        sources: .c 文件名（相对 :data:`HERE`）或绝对路径。
        exe_name: 产物名（放在 ``build/`` 下）。
        include_dirs: 额外的 ``/I`` 目录，**按顺序**排在前面（stub 头要抢先）。
        out_dir: 产物目录，默认 ``build/``。
        extra_flags: 追加的编译开关。

    Returns:
        ``(exe, warnings)``；``warnings`` 每条形如 ``"文件名: 编译器原文"``。
    """
    out_dir = Path(out_dir) if out_dir else (HERE / "build")
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / exe_name
    warnings = []
    objs = []
    for src in sources:
        src = Path(src)
        name = src.name
        args = [CL_EXE, *CFLAGS]
        for inc in include_dirs:
            args.append("/I" + str(inc))
        args += list(extra_flags)
        args += [str(src), "/Fo" + str(out_dir / (name + ".obj"))]
        text, _rc = run(args)
        warnings += ["%s: %s" % (name, _ascii(ln)) for ln in text.splitlines()
                     if "warning" in ln.lower()]
        objs.append(str(out_dir / (name + ".obj")))
    text, _rc = run([CL_EXE, "/nologo", *objs, "/Fe:" + str(exe)])
    warnings += ["link: %s" % _ascii(ln) for ln in text.splitlines()
                 if "warning" in ln.lower()]
    if not exe.exists():
        raise RuntimeError("没有生成 %s\n%s" % (exe, text))
    return exe, warnings


def run_exe(exe, args=(), timeout=600, check=True):
    """跑 exe，返回 ``(stdout, returncode)``。"""
    return run([str(exe), *[str(a) for a in args]], timeout=timeout, check=check)


if __name__ == "__main__":
    print("CL_EXE =", CL_EXE, CL_EXE.exists())
    print("VCVARS =", VCVARS, VCVARS.exists())
    env = msvc_env()
    print("INCLUDE ok:", bool(env.get("INCLUDE")), " LIB ok:", bool(env.get("LIB")))
    sys.exit(0)
