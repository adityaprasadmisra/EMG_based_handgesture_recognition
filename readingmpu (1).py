import serial
import csv
import time
import sys
import os

# ─── SETTINGS ───────────────────────────────────────────
PORT           = 'COM10'
BAUD           = 115200
TRIALS         = 15
HOLD_SECONDS   = 3
SETTLE_SECONDS = 1
REST_SECONDS   = 2
FILENAME       = 'data21' 
'.csv'

GESTURES = ['relax', 'fist', 'flexor', 'extensor', 'open']

GESTURE_INSTRUCTIONS = {
    'relax'    : 'Let your arm hang loose, no tension', 
    'fist'     : 'Curl all fingers into a tight fist',
    'flexor'   : 'Curl fingers slightly, bend wrist inward',
    'extensor' : 'Straighten and spread fingers, bend wrist outward',
    'open'     : 'Fully open hand, fingers straight and together',
}
# ────────────────────────────────────────────────────────

def countdown(message, seconds, symbol=''):
    for remaining in range(seconds, 0, -1):
        print(f'\r  {symbol}  {message}  [{remaining}s remaining]   ', end='')
        sys.stdout.flush()
        time.sleep(1)
    print()

def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = int(100 * current / total)
    print(f'\r  RECORDING  [{bar}] {percent}%  ', end='')
    sys.stdout.flush()

# ── CONNECT ─────────────────────────────────────────────
print('\n' + '='*50)
print(f'  EMG + IMU DATA COLLECTION — ALL 5 GESTURES')
print(f'  Channels : EMG1, EMG2, AX, AY, AZ, GX, GY, Roll, Pitch')
print(f'  Trials per gesture : {TRIALS}')
print(f'  Estimated time     : ~{int(len(GESTURES) * TRIALS * (HOLD_SECONDS + REST_SECONDS + SETTLE_SECONDS) / 60)} minutes')
print('='*50)

print('\nConnecting to ESP32...')
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)
print('Connected.\n')

# ── CSV SETUP ────────────────────────────────────────────
file_exists = os.path.exists(FILENAME) and os.path.getsize(FILENAME) > 0
csvfile = open(FILENAME, 'a', newline='')
writer  = csv.writer(csvfile)
if not file_exists:
    writer.writerow(['emg1', 'emg2', 'ax', 'ay', 'az',
                     'gx', 'gy', 'roll', 'pitch', 'gesture'])

total_collected = 0

# ── GESTURE LOOP ─────────────────────────────────────────
for g_index, gesture in enumerate(GESTURES):

    print(f'\n{"#"*50}')
    print(f'  GESTURE {g_index+1} of {len(GESTURES)}: {gesture.upper()}')
    print(f'  Instruction: {GESTURE_INSTRUCTIONS[gesture]}')
    print(f'{"#"*50}')

    countdown(f'GET READY FOR {gesture.upper()}', 5, '🕐')

    gesture_collected = 0

    for trial in range(1, TRIALS + 1):

        print(f'\n  {"─"*44}')
        print(f'  TRIAL {trial} of {TRIALS}  |  Gesture: {gesture.upper()}')
        print(f'  {"─"*44}')

        print(f'\n  👉  {GESTURE_INSTRUCTIONS[gesture]}')
        countdown('PREPARE — form your gesture', REST_SECONDS, '✋')
        countdown('SETTLE — hold still, not recording yet', SETTLE_SECONDS, '⏸')

        print(f'  Recording...')
        trial_start = time.time()
        trial_rows  = 0
        trial_data  = []

        while True:
            elapsed = time.time() - trial_start
            if elapsed >= HOLD_SECONDS:
                break

            progress_bar(elapsed, HOLD_SECONDS)

            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) == 9:             # ← 9 values now
                        emg1  = int(parts[0])
                        emg2  = int(parts[1])
                        ax    = float(parts[2])
                        ay    = float(parts[3])
                        az    = float(parts[4])
                        gx    = float(parts[5])
                        gy    = float(parts[6])
                        roll  = float(parts[7])
                        pitch = float(parts[8])
                        trial_data.append([emg1, emg2, ax, ay, az,
                                           gx, gy, roll, pitch, gesture])
                        trial_rows += 1
            except:
                pass

        writer.writerows(trial_data)
        csvfile.flush()

        gesture_collected += trial_rows
        total_collected   += trial_rows

        print(f'\r  RECORDING  [{"█"*30}] 100%  ')
        print(f'  ✓ {trial_rows} readings  |  Gesture total: {gesture_collected}  |  Overall: {total_collected}')

        if trial < TRIALS:
            countdown('RELAX — open your hand completely', REST_SECONDS, '🖐')

    print(f'\n  ✅  {gesture.upper()} complete — {gesture_collected} readings saved')

    if g_index < len(GESTURES) - 1:
        next_gesture = GESTURES[g_index + 1]
        print(f'\n  Next gesture: {next_gesture.upper()}')
        print(f'  {GESTURE_INSTRUCTIONS[next_gesture]}')
        countdown(f'BREAK before {next_gesture.upper()}', 5, '💤')

# ── DONE ─────────────────────────────────────────────────
csvfile.close()
ser.close()

print(f'\n{"="*50}')
print(f'  🎉  ALL GESTURES COMPLETE!')
print(f'  Total readings : {total_collected}')
print(f'  Saved to       : {FILENAME}')
print(f'{"="*50}\n')
