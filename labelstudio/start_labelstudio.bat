@echo off
REM ============================================================
REM RIG-Hover Label Studio annotation workbench
REM   Reuses the install at D:\smartcar\labelstudio, but the data dir
REM   and port are independent (3334). Fully isolated from the smart
REM   car instance (3333).
REM   Stop = Ctrl+C in this window
REM ============================================================
title RIG-Hover Label Studio (Ctrl+C stop)
set DJANGO_TELEMETRY=0
set LABEL_STUDIO_DATA_DIR=D:\robot_work\labelstudio\data
set LABEL_STUDIO_BASE_DATA_DIR=D:\robot_work\labelstudio\data
cd /d D:\robot_work\labelstudio

for /f %%P in ('powershell -NoProfile -Command "(Get-NetTCPConnection -State Listen -LocalPort 3334 -EA SilentlyContinue | Select-Object -First 1).OwningProcess"') do set LSPID=%%P
if defined LSPID (
    echo [note] 3334 is already listening (PID %LSPID%), platform already running.
    echo        to stop it: Ctrl+C in its window, or taskkill /PID %LSPID% /F
    pause
    exit /b 1
)

echo local address : http://localhost:3334
echo stop          : Ctrl+C in this window
echo.
start "" powershell -NoProfile -Command "1..60 | ForEach-Object { Start-Sleep 1; if (Get-NetTCPConnection -State Listen -LocalPort 3334 -EA SilentlyContinue) { Start-Process 'http://localhost:3334'; break } }"

echo ---------- service log (Ctrl+C to stop) ----------
"D:\smartcar\labelstudio\venv\Scripts\label-studio.exe" start --port 3334 --data-dir "D:\robot_work\labelstudio\data"
echo ---------- stopped, press any key ----------
pause