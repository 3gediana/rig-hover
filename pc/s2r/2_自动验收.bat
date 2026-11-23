@echo off
chcp 65001 >nul
rem === 第 2 步：机器人立在桌上、前方留 20cm 空桌面、你坐在它对面。倾斜会自动停。===
"D:\smartcar\.venv\Scripts\python.exe" D:\robot_work\pc\s2r\morning_check.py
for /f "delims=" %%d in ('dir /b /ad /o-d D:\robot_work\runs_end_side\s2r_morning') do ( start "" notepad "D:\robot_work\runs_end_side\s2r_morning\%%d\report.md" & goto :done )
:done
pause
