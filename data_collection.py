import serial
import csv
import time
import sys

# ─── SETTINGS ───────────────────────────────────────────
PORT          = 'COM3'        # change to your ESP32 port
                              # Mac/Linux: '/dev/ttyUSB0'
BAUD          = 115200
GESTURE_NAME  = 'fist'        # change for each gesture
TRIALS        = 15
HOLD_SECONDS  = 3
SETTLE_SECONDS = 1
REST_SECONDS  = 2
# ────────────────────────────────────────────────────────

def countdown(message, seconds, symbol=''):
    # Prints a live countdown on the SAME line
    for remaining in range(seconds, 0, -1):
        # \r moves cursor to start of line, end='' prevents newline
        print(f'\r  {symbol}  {message}  [{remaining}s remaining]   ', end='')
        sys.stdout.flush()
        time.sleep(1)
    print()  # move to next line when done

def progress_bar(current, total, width=30):
    # Shows a filling bar: [████████░░░░░░░░░░] 60%
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = int(100 * current / total)
    print(f'\r  RECORDING  [{bar}] {percent}%  ', end='')
    sys.stdout.flush()

# ── CONNECT ─────────────────────────────────────────────
print('\n' + '='*50)
print(f'  EMG DATA COLLECTION')
print(f'  Gesture  : {GESTURE_NAME.upper()}')
print(f'  Trials   : {TRIALS}')
print(f'  Hold time: {HOLD_SECONDS} seconds per trial')
print('='*50)

print('\nConnecting to ESP32...')
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)
print('Connected.\n')

print('Starting in 5 seconds — get your arm ready...')
countdown('GET READY', 5, '🕐')

all_rows = []

# ── MAIN LOOP ────────────────────────────────────────────
for trial in range(1, TRIALS + 1):

    print(f'\n{"─"*50}')
    print(f'  TRIAL {trial} of {TRIALS}')
    print(f'{"─"*50}')

    # ── PHASE 1: PREPARE ──────────────────────────────
    print(f'\n  Form the gesture: {GESTURE_NAME.upper()}')
    countdown('PREPARE — form your gesture', REST_SECONDS, '✋')

    # ── PHASE 2: SETTLE ───────────────────────────────
    print(f'  Hold position still...')
    countdown('SETTLE — hold still, not recording yet', SETTLE_SECONDS, '⏸')

    # ── PHASE 3: RECORD ───────────────────────────────
    print(f'  Recording...')
    trial_start = time.time()
    trial_rows = 0

    while True:
        elapsed = time.time() - trial_start

        # Stop after HOLD_SECONDS
        if elapsed >= HOLD_SECONDS:
            break

        # Show live progress bar
        progress_bar(elapsed, HOLD_SECONDS)

        # Read one line from ESP32
        try:
            line = ser.readline().decode().strip()
            if ',' in line:
                parts = line.split(',')
                if len(parts) == 3:
                    ch1  = int(parts[0])
                    ch2  = int(parts[1])
                    ch3  = int(parts[2])
                    all_rows.append([ch1, ch2, ch3, GESTURE_NAME])
                    trial_rows += 1
        except:
            pass  # skip corrupted lines

    print(f'\r  RECORDING  [{"█"*30}] 100%  ')
    print(f'  ✓ Captured {trial_rows} readings this trial')
    print(f'  Total saved so far: {len(all_rows)} readings')

    # ── PHASE 4: RELAX ────────────────────────────────
    if trial < TRIALS:  # no relax message after last trial
        print()
        countdown('RELAX — open your hand completely', REST_SECONDS, '🖐')

# ── SAVE TO CSV ──────────────────────────────────────────
print(f'\n{"="*50}')
print(f'  All trials complete!')
print(f'  Total readings collected: {len(all_rows)}')

filename = 'data.csv'
with open(filename, 'a', newline='') as f:
    csv.writer(f).writerows(all_rows)

print(f'  Saved to: {filename}')
print(f'  Expected training windows: ~{int((len(all_rows)/TRIALS - 50) / 25 * TRIALS)}')
print(f'{"="*50}\n')

ser.close()
