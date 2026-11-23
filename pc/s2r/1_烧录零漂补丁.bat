@echo off
chcp 65001 >nul
rem === 第 1 步：烧录陀螺零漂补丁（先插 USB）。按项目规矩先整片备份，再只烧 app。===
set IDF_TOOLS_PATH=G:\esp\.espressif
set BK=D:\robot_work\_fw_backup_%date:~0,4%%date:~5,2%%date:~8,2%_pre_s2r.bin
powershell -NoProfile -Command "if (-not ([System.IO.Ports.SerialPort]::GetPortNames() -contains 'COM3')) { Write-Host 'COM3 不存在：请插好 USB 再运行'; exit 1 }"
if errorlevel 1 ( pause & exit /b 1 )
call G:\esp\esp-idf\export.bat >nul
echo [1/3] 整片备份到 %BK% ...
python -m esptool --chip esp32s3 -p COM3 -b 460800 --before default_reset read_flash 0 0x1000000 "%BK%" || ( echo 备份失败，停止，不烧录 & pause & exit /b 1 )
echo [2/3] 烧录 app ...
cd /d D:\robot_work\rig_omni
idf.py -p COM3 -b 460800 app-flash || ( echo 烧录失败。回滚: python -m esptool -p COM3 write_flash 0 "%BK%" & pause & exit /b 1 )
echo [3/3] 等 40 秒让它开机并完成 IMU 标定，然后量 20 秒静止零漂（机器人放着别动）...
timeout /t 40 >nul
"D:\smartcar\.venv\Scripts\python.exe" -c "import time,requests;S=requests.Session();S.trust_env=False;r=[];[r.append(S.get('http://192.168.137.50/status',timeout=2).json()['imu']['yaw_rate']) or time.sleep(1) for _ in range(20)];m=sum(r)/len(r);print('静止 yaw_rate 均值 %%.2f deg/s -> %%s'%%(m,'通过 (<=0.3)' if abs(m)<=0.3 else '未通过，把这个数发给我'))"
pause
