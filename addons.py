import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ====================== SETTINGS ======================
CSV_FILES = {
    'person1': 'data4.csv',
    'person2': 'data5.csv',
    'person3': 'data6.csv',
    'person4': 'data7.csv',
    'person5': 'data8.csv',
    # Add 'person6': 'data9.csv' when you collect it
}

COLUMNS = ['emg1', 'emg2', 'ax', 'ay', 'az', 'gx', 'gy', 'roll', 'pitch', 'gesture']

# ====================== LOAD ALL DATA ======================
frames = []
for person, filename in CSV_FILES.items():
    try:
        df = pd.read_csv(filename, header=0)
        df.columns = COLUMNS
        df['person'] = person
        frames.append(df)
        print(f"Loaded {filename:25s} → {len(df):,} rows")
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Skipping.")

data = pd.concat(frames, ignore_index=True)
print(f"\nTotal combined rows: {len(data):,}")

# ====================== BASIC STATISTICS ======================
print("\n" + "="*60)
print("GESTURE-WISE SAMPLE COUNT")
print("="*60)
print(data['gesture'].value_counts().sort_index())

# ====================== EXTRACT RELAX (Baseline Noise) ======================
relax_mask = data['gesture'] == 'relax'
relax = data.loc[relax_mask, 'emg1'].values

print(f"\nRelax (baseline) samples : {len(relax):,}")
print(f"Mean     : {np.mean(relax):.2f}")
print(f"Std Dev  : {np.std(relax):.2f}")
print(f"Noise Floor (±2σ) : ±{np.std(relax)*2:.2f} ADC counts")

# Plot baseline noise (first 10,000 samples to keep plot manageable)
plt.figure(figsize=(12, 3))
plt.plot(relax[:10000], color='tab:blue', linewidth=0.5)
plt.title('Baseline Noise — Relaxed Arm (EMG1)', fontweight='bold')
plt.xlabel('Sample')
plt.ylabel('ADC Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_noise.png', dpi=150)
plt.show()

# ====================== PSD: Relax vs Fist ======================
n_samples = 5000  # Use more samples for better PSD estimate

relax_data = data[data['gesture'] == 'relax']['emg1'].values[:n_samples]
fist_data  = data[data['gesture'] == 'fist']['emg1'].values[:n_samples]

fs = 500.0  # Sampling frequency in Hz

f_relax, Pxx_relax = signal.welch(relax_data, fs=fs, nperseg=512)
f_fist,  Pxx_fist  = signal.welch(fist_data,  fs=fs, nperseg=512)

plt.figure(figsize=(10, 5))
plt.semilogy(f_relax, Pxx_relax, label='Relax (Baseline)', color='tab:blue')
plt.semilogy(f_fist,  Pxx_fist,  label='Fist', color='tab:red')
plt.axvspan(20, 450, alpha=0.12, color='green', label='Useful EMG band (20–450 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V²/Hz)')
plt.title('Power Spectral Density — Relax vs Fist', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('psd_plot.png', dpi=150)
plt.show()

# ====================== SNR ======================
noise_rms = np.sqrt(np.mean(relax_data.astype(float)**2))
signal_rms = np.sqrt(np.mean(fist_data.astype(float)**2))

snr_db = 20 * np.log10(signal_rms / noise_rms) if noise_rms > 0 else float('inf')

print(f"\nNoise RMS  : {noise_rms:.2f}")
print(f"Signal RMS : {signal_rms:.2f}")
print(f"SNR        : {snr_db:.2f} dB")

# ====================== FILTERING FUNCTIONS ======================
def bandpass_filter(data, lowcut=20, highcut=450, fs=500, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, freq=50, fs=500, Q=30):
    b, a = signal.iirnotch(freq / (fs / 2), Q)
    return signal.filtfilt(b, a, data)

# ====================== APPLY FILTERS ON FIST ======================
emg_raw = fist_data.astype(float).copy()
emg_notched = notch_filter(emg_raw)
emg_filtered = bandpass_filter(emg_notched)

# Plot filtering stages
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(emg_raw,      color='tab:red',    linewidth=0.6)
axes[0].set_title('Raw EMG Signal (Fist)', fontweight='bold')

axes[1].plot(emg_notched,  color='tab:orange', linewidth=0.6)
axes[1].set_title('After 50Hz Notch Filter', fontweight='bold')

axes[2].plot(emg_filtered, color='tab:green',  linewidth=0.6)
axes[2].set_title('After Bandpass (20–450 Hz)', fontweight='bold')

for ax in axes:
    ax.set_ylabel('ADC Value')
    ax.grid(alpha=0.3)

axes[2].set_xlabel('Sample')
plt.suptitle('EMG Signal Filtering Pipeline', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('filtering.png', dpi=150)
plt.show()
