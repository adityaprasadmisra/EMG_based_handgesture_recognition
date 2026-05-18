import serial
import numpy as np
import pickle
import time

from collections import Counter


# ── LOAD MODEL ───────────────────────────────────────────
model  = pickle.load(open('gesture_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl',        'rb'))
le     = pickle.load(open('label_encoder.pkl', 'rb'))

print("═" * 40)
print("  MODEL LOADED")
print(f"  Type     : {type(model).__name__}")
print(f"  Gestures : {[str(g) for g in le.classes_]}")
print("═" * 40)

PORT        = 'COM10'
BAUD        = 115200
WINDOW_SIZE = 50
STEP        = 25
VOTE_BUFFER = 7
CONFIDENCE  = 0.70

GESTURE_DISPLAY = {
    'relax'    : '😶  RELAX',
    'fist'     : '✊  FIST',
    'flexor'   : '🤙  FLEXOR',
    'extensor' : '🖖  EXTENSOR',
    'open'     : '✋  OPEN HAND',
}

def extract_features(window):
    window = np.array(window)
    features = []
    for ch in [0, 1]:
        col = window[:, ch].astype(float)
        
        features.append(np.mean(np.abs(col)))
        features.append(np.sqrt(np.mean(col ** 2)))
        features.append(np.var(col))
        features.append(np.std(col))
        zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
        features.append(zc)
    roll      = window[:, 7]
    pitch     = window[:, 8]
    ax, ay, az = window[:, 2], window[:, 3], window[:, 4]
    features.append(np.mean(roll))
    features.append(np.mean(pitch))
    features.append(np.mean(np.sqrt(ax**2 + ay**2 + az**2)))
    return np.array(features)

def predict_from_buffer(data_buffer):
    """Run windowed prediction over entire recorded buffer, return majority vote."""
    all_predictions = []
    for start in range(0, len(data_buffer) - WINDOW_SIZE, STEP):
        window   = data_buffer[start : start + WINDOW_SIZE]
        features = extract_features(window)
        features_scaled = scaler.transform([features])
        pred     = model.predict(features_scaled)[0]
        gesture  = str(le.inverse_transform([pred])[0])
        all_predictions.append(gesture)

    if not all_predictions:
        return None, 0.0

    counts      = Counter(all_predictions)
    top_gesture, top_count = counts.most_common(1)[0]
    confidence  = top_count / len(all_predictions)
    return top_gesture, confidence

# ── CONNECT ──────────────────────────────────────────────
print(f"\nConnecting on {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(2)
print("Connected.\n")

session = 1

while True:
    print("─" * 40)
    print(f"  SESSION {session}")
    print("─" * 40)
    input("\n  Press ENTER to START recording your gesture...")

    # ── RECORDING PHASE ──────────────────────────────────
    print("  🔴 RECORDING — Hold your gesture now...")
    print("  Press ENTER to STOP\n")

    data_buffer  = []
    stop_flag    = [False]

    # Non-blocking stop — runs stop detection in a separate thread
    import threading
    def wait_for_stop():
        input()
        stop_flag[0] = True

    stop_thread = threading.Thread(target=wait_for_stop, daemon=True)
    stop_thread.start()

    sample_count = 0
    ser.reset_input_buffer()   # clear stale bytes before recording

    while not stop_flag[0]:
        try:
            line  = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) != 9:
                continue
            row = [
                int(parts[0]),   int(parts[1]),
                float(parts[2]), float(parts[3]), float(parts[4]),
                float(parts[5]), float(parts[6]),
                float(parts[7]), float(parts[8]),
            ]
            data_buffer.append(row)
            sample_count += 1

            # Show a live dot every 100 samples so user knows it's recording
            if sample_count % 100 == 0:
                print(f"  ● {sample_count} samples captured...", end='\r')

        except ValueError:
            continue
        except Exception:
            continue

    # ── DETECTION PHASE ──────────────────────────────────
    print(f"\n\n  ⏹  STOPPED — {len(data_buffer)} samples recorded")

    if len(data_buffer) < WINDOW_SIZE:
        print(f"  ⚠  Not enough data (need at least {WINDOW_SIZE} samples). Hold gesture longer.\n")
        session += 1
        continue

    print("  Analysing gesture...")
    time.sleep(0.3)

    gesture, confidence = predict_from_buffer(data_buffer)

    if gesture is None:
        print("  ⚠  Could not determine gesture. Try again.\n")
    elif confidence < CONFIDENCE:
        label = GESTURE_DISPLAY.get(gesture, gesture.upper())
        print(f"\n  ⚠  LOW CONFIDENCE — Best guess: {label}  ({int(confidence*100)}%)")
        print(f"  Hold the gesture more steadily next time.\n")
    else:
        label = GESTURE_DISPLAY.get(gesture, gesture.upper())
        print(f"\n  ╔══════════════════════════════╗")
        print(f"  ║  DETECTED → {label:<18}║")
        print(f"  ║  Confidence : {int(confidence*100)}%{'':13}║")
        print(f"  ║  Windows    : {len(data_buffer)//STEP:<14} ║")

        print(f"  ╚══════════════════════════════╝\n")

    again = input("  Run again? (y / n): ").strip().lower()
    if again != 'y':
        break

    session += 1

ser.close()
print("\n  Session ended. Port closed.")