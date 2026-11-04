import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import serial
except ImportError:
    print("NO_PYSERIAL")
    sys.exit(2)

p = serial.Serial()
p.port = "COM3"
p.baudrate = 115200
p.timeout = 0.2
p.dtr = False
p.rts = False
p.open()
print("PORT_OPEN", p.port, p.baudrate)
deadline = time.time() + 12
buf = bytearray()
while time.time() < deadline:
    chunk = p.read(4096)
    if chunk:
        buf.extend(chunk)
p.close()
text = buf.decode("utf-8", errors="replace")
print("BYTES", len(buf))
print("--- RAW START ---")
print(text[-8000:])
print("--- RAW END ---")