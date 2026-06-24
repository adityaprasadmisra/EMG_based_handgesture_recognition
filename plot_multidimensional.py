import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Use Agg backend for non-interactive saving
import matplotlib
matplotlib.use('Agg')

# Settings and constants
CSV_FILES = {
    'person1': 'data4.csv',
    'person2': 'data5.csv',
    'person3': 'data6.csv',
    'person4': 'data7.csv',
    'person5': 'data8.csv',
}
COLUMNS = ['emg1', 'emg2', 'ax', 'ay', 'az', 'gx', 'gy', 'roll', 'pitch', 'gesture']
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
    for ch in ['emg1', 'emg2']:
        col = window[ch].values.astype(float)
        features.append(np.mean(np.abs(col))) # MAV
        features.append(np.sqrt(np.mean(col ** 2))) # RMS
        features.append(np.var(col)) # Variance
        features.append(np.std(col)) # Std Dev
        zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
        features.append(zc) # Zero Crossing

    features.append(window['roll'].mean())
    features.append(window['pitch'].mean())
    accel_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
    features.append(accel_mag.mean())
    return features

# 1. Load and clean
print("Loading data...")
frames = []
for person, filename in CSV_FILES.items():
    temp = pd.read_csv(filename, header=0)
    temp.columns = COLUMNS
    temp['person'] = person
    frames.append(temp)

df = pd.concat(frames, ignore_index=True)
df['emg1'] = pd.to_numeric(df['emg1'], errors='coerce')
df['emg2'] = pd.to_numeric(df['emg2'], errors='coerce')
df = df.dropna(subset=COLUMNS)
df['gesture'] = df['gesture'].str.strip()
df['emg1']    = df['emg1'].astype(int)
df['emg2']    = df['emg2'].astype(int)
df = df[(df['emg1'] >= 0) & (df['emg1'] <= 4095)]
df = df[(df['emg2'] >= 0) & (df['emg2'] <= 4095)]

# 2. Filter
print("Filtering EMG signals...")
emg1_filtered = np.zeros(len(df))
emg2_filtered = np.zeros(len(df))
for gesture in df['gesture'].unique():
    idx = df[df['gesture'] == gesture].index
    emg1_filtered[idx] = process_emg(df.loc[idx, 'emg1'])
    emg2_filtered[idx] = process_emg(df.loc[idx, 'emg2'])

df['emg1'] = emg1_filtered
df['emg2'] = emg2_filtered

# 3. Window & Feature Extraction
print("Extracting features...")
WINDOW_SIZE = 50
STEP        = 25
gestures = df['gesture'].unique()

X = []
y = []
for gesture in gestures:
    gesture_data = df[df['gesture'] == gesture].reset_index(drop=True)
    # Downsample slightly for faster PCA plotting (take every 2nd window)
    for start in range(0, len(gesture_data) - WINDOW_SIZE, STEP * 2):
        window = gesture_data.iloc[start : start + WINDOW_SIZE]
        X.append(extract_features(window))
        y.append(gesture)

X = np.array(X)
y = np.array(y)

# 4. Scale and PCA
print("Scaling and PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 5. Plotting 3D Scatter
print("Generating 3D plot...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = {
    'relax': '#94a3b8',       # slate gray
    'fist': '#ef4444',        # vibrant red
    'flexor': '#3b82f6',      # deep blue
    'extensor': '#f97316',    # orange
    'open': '#10b981'         # emerald green
}

for gesture in np.unique(y):
    idx = np.where(y == gesture)
    ax.scatter(
        X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2],
        label=gesture.upper(),
        color=colors.get(gesture, '#6b7280'),
        alpha=0.6,
        s=15,
        edgecolors='none'
    )

ax.set_title("Multidimensional Feature Space Projection (3D PCA)\nShowing Gesture Separation", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight='bold')
ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='bold')
ax.set_zlabel(f"PCA 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", fontweight='bold')

# Style adjustments
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
plt.tight_layout()

# Save output image
output_filename = 'multidimensional_pca.png'
plt.savefig(output_filename, dpi=150)
print(f"Saved multidimensional plot as {output_filename}")
