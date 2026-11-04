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

# Deliberate EN pulse: RTS drives EN on the CH343 auto-reset circuit.
p.dtr = False
p.rts = True
time.sleep(0.12)
p.rts = False
time.sleep(0.2)

out = r"D:\robot_work\_serial_boot.txt"
deadline = time.time() + 70
buf = bytearray()
while time.time() < deadline:
    chunk = p.read(8192)
    if chunk:
        buf.extend(chunk)
p.close()
text = buf.decode("utf-8", errors="replace")
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print("BYTES", len(buf))
print("WIFI_LINES")
for line in text.splitlines():
    low = line.lower()
    if any(k in low for k in ("wifi", "ssid", "ip:", "scan", "connect", "static-ip", "dhcp", "station")):
        print(line)