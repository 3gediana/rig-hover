@echo off
chcp 65001 >nul
rem === 第 3 步：点击导航。窗口里左键点桌面上一点（30cm 内），它会开过去停住；按 q 退出。===
"D:\smartcar\.venv\Scripts\python.exe" D:\robot_work\pc\s2r\live.py --mode click --secs 180
pause
