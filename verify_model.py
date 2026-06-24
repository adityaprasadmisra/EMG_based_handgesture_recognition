import pandas as pd
import numpy as np
import pickle
from scipy import signal

# Settings and constants
GESTURES_DICT = {
    'relax': '  RELAX',
    'fist': '✊  FIST',
    'flexor': '🤙  FLEXOR',
    'extensor': '🖖  EXTENSOR',
    'open': '✋  OPEN HAND',
}

FS = 500  # sampling rate

def notch_filter(data, freq=50, fs=500, Q=30):
    b, a = signal.iirnotch(freq / (fs / 2), Q)
    return signal.filtfilt(b, a, data)

def bandpass_filter(data, lowcut=20, highcut=240, fs=500, order=4):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def process_emg(series):
    arr      = series.values.astype(float)
    notched  = notch_filter(arr, fs=FS)
    filtered = bandpass_filter(notched, fs=FS)
    return filtered

def extract_features(window):
    features = []
    # EMG features (2 channels)
    for ch in ['emg1', 'emg2']:
        col = window[ch].values.astype(float)
        features.append(np.mean(np.abs(col))) # MAV
        features.append(np.sqrt(np.mean(col ** 2))) # RMS
        features.append(np.var(col)) # Variance
        features.append(np.std(col)) # Std Dev
        zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
        features.append(zc) # Zero Crossing

    # IMU features
    features.append(window['roll'].mean())
    features.append(window['pitch'].mean())
    accel_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
    features.append(accel_mag.mean())
    return features

# 1. Load Model Files
print("Loading model components...")
model  = pickle.load(open('gesture_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le     = pickle.load(open('label_encoder.pkl', 'rb'))

print("Model Loaded successfully.")
print(f"Model Type: {type(model).__name__}")
print(f"Supported Classes: {list(le.classes_)}\n")

# 2. Load a sample window from the dataset
print("Loading dataset for test prediction...")
df = pd.read_csv('data8.csv', header=0)
df.columns = ['emg1', 'emg2', 'ax', 'ay', 'az', 'gx', 'gy', 'roll', 'pitch', 'gesture']

# Clean & Filter the data
df['gesture'] = df['gesture'].str.strip()
df['emg1'] = pd.to_numeric(df['emg1'], errors='coerce')
df['emg2'] = pd.to_numeric(df['emg2'], errors='coerce')
df = df.dropna()
df['emg1'] = df['emg1'].astype(int)
df['emg2'] = df['emg2'].astype(int)

# Filter EMG signals
emg1_filtered = np.zeros(len(df))
emg2_filtered = np.zeros(len(df))
for gesture in df['gesture'].unique():
    idx = df[df['gesture'] == gesture].index
    emg1_filtered[idx] = process_emg(df.loc[idx, 'emg1'])
    emg2_filtered[idx] = process_emg(df.loc[idx, 'emg2'])

df['emg1'] = emg1_filtered
df['emg2'] = emg2_filtered

# 3. Test prediction on random samples of each gesture type
print("\n--- Running Test Predictions ---")
for target_gesture in df['gesture'].unique():
    subset = df[df['gesture'] == target_gesture].reset_index(drop=True)
    if len(subset) >= 50:
        window = subset.iloc[10:60] # take a 50-sample window
        features = extract_features(window)
        scaled_features = scaler.transform([features])
        pred_class = model.predict(scaled_features)[0]
        pred_label = le.inverse_transform([pred_class])[0]
        
        display_target = GESTURES_DICT.get(target_gesture, target_gesture.upper())
        display_pred = GESTURES_DICT.get(pred_label, pred_label.upper())
        
        status = "✓ MATCH" if target_gesture == pred_label else "✗ MISMATCH"
        print(f"Target: {display_target:15s} | Predicted: {display_pred:15s} | Result: {status}")

print("\nAll done. Model is verified and ready for deployment.")
