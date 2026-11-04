import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import serial

p = serial.Serial()
p.port = "COM3"
p.baudrate = 115200
p.timeout = 0.2
p.dtr = False
p.rts = False
p.open()

# EN pulse to reset the robot (CH343 RTS drives EN).
p.dtr = False
p.rts = True
time.sleep(0.12)
p.rts = False
time.sleep(0.4)
p.close()
print("RESET_DONE")