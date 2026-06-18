import serial
import time

ser = serial.Serial('COM10', 115200, timeout=2)
time.sleep(2)

print("Checking data format:\n")
for _ in range(10):
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    parts = line.split(',')
    print(f"  raw: {repr(line):25s}  parts: {len(parts)}  values: {parts}")

ser.close()
