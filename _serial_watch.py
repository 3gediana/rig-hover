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
p.open()  # dtr/rts already low: no reset

out = r"D:\robot_work\_serial_reconnect.txt"
deadline = time.time() + 120
buf = bytearray()
while time.time() < deadline:
    chunk = p.read(8192)
    if chunk:
        buf.extend(chunk)
p.close()
with open(out, "w", encoding="utf-8") as f:
    f.write(buf.decode("utf-8", errors="replace"))
print("BYTES", len(buf))