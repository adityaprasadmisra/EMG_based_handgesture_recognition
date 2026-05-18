import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# ═══════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD AND CLEAN DATA
# ═══════════════════════════════════════════════════════════════

print("═" * 50)
print("  SECTION 1 — LOAD AND CLEAN")
print("═" * 50)

CSV_FILES = {
    'person1': 'data4.csv',
    'person2': 'data5.csv',
    'person3': 'data6.csv',
    'person4': 'data7.csv',
    'person5': 'data8.csv',
}

COLUMNS = ['emg1', 'emg2', 'ax', 'ay', 'az', 'gx', 'gy', 'roll', 'pitch', 'gesture']

frames = []
for person, filename in CSV_FILES.items():
    temp = pd.read_csv(filename, header=0)
    temp.columns = COLUMNS
    temp['person'] = person
    frames.append(temp)
    print(f"  Loaded {filename:25s} → {len(temp)} rows")

df = pd.concat(frames, ignore_index=True)
print(f"\n  Combined total rows : {len(df)}")

# ── CLEAN ──────────────────────────────────────────────────────
df['emg1'] = pd.to_numeric(df['emg1'], errors='coerce')
df['emg2'] = pd.to_numeric(df['emg2'], errors='coerce')

before = len(df)
df = df.dropna(subset=COLUMNS)
after  = len(df)
if before != after:
    print(f"  Dropped {before - after} corrupted rows")

df['gesture'] = df['gesture'].str.strip()
df['emg1']    = df['emg1'].astype(int)
df['emg2']    = df['emg2'].astype(int)
df = df[(df['emg1'] >= 0) & (df['emg1'] <= 4095)]
df = df[(df['emg2'] >= 0) & (df['emg2'] <= 4095)]

print(f"  Rows after cleaning : {len(df)}")
print(f"\n  Samples per gesture :")
print(df['gesture'].value_counts().to_string())


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — SIGNAL PROCESSING  ← NEW LAYER
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 2 — SIGNAL PROCESSING")
print("═" * 50)

FS = 500  # sampling rate

def notch_filter(data, freq=50, fs=500, Q=30):
    """Remove 50Hz power line interference"""
    b, a = signal.iirnotch(freq / (fs / 2), Q)
    return signal.filtfilt(b, a, data)

def bandpass_filter(data, lowcut=20, highcut=240, fs=500, order=4):
    """Keep only the useful EMG band (20–240Hz at 500Hz sampling)"""
    nyq  = fs / 2
    low  = lowcut  / nyq   # 20/250  = 0.08
    high = highcut / nyq   # 240/250 = 0.96
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def process_emg(series):
    """Apply notch then bandpass to a pandas Series, return numpy array"""
    arr      = series.values.astype(float)
    notched  = notch_filter(arr, fs=FS)
    filtered = bandpass_filter(notched, fs=FS)
    return filtered

# ── Apply filtering per gesture group ─────────────────────────
# WHY per gesture group?
# filtfilt needs a continuous signal. If we filter the entire
# concatenated dataframe at once, the filter bleeds across
# gesture boundaries (a fist signal leaking into a relax signal).
# Filtering per gesture keeps each gesture's signal independent.

emg1_filtered = np.zeros(len(df))
emg2_filtered = np.zeros(len(df))

for gesture in df['gesture'].unique():
    idx = df[df['gesture'] == gesture].index

    emg1_filtered[df.index.get_indexer(idx)] = process_emg(df.loc[idx, 'emg1'])
    emg2_filtered[df.index.get_indexer(idx)] = process_emg(df.loc[idx, 'emg2'])

df['emg1'] = emg1_filtered
df['emg2'] = emg2_filtered

print("  Applied: 50Hz notch filter → bandpass filter (20–240Hz)")
print("  Filtered per gesture group to avoid boundary bleed")
print(f"  EMG1 post-filter stats: mean={df['emg1'].mean():.2f}, std={df['emg1'].std():.2f}")
print(f"  EMG2 post-filter stats: mean={df['emg2'].mean():.2f}, std={df['emg2'].std():.2f}")

# ── Quick visual check ─────────────────────────────────────────
# Shows raw vs filtered for one gesture so you can verify it worked
sample_gesture = 'fist'
sample_data    = df[df['gesture'] == sample_gesture].head(500)

raw_sample = pd.read_csv(list(CSV_FILES.values())[0])
raw_sample.columns = COLUMNS + ['person'] if 'person' in raw_sample.columns else COLUMNS
raw_emg1   = raw_sample[raw_sample['gesture'] == sample_gesture]['emg1'].values[:500].astype(float)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(raw_emg1,                        color='tab:red',   linewidth=0.5)
axes[0].set_title('Raw EMG1 — Fist',          fontweight='bold')
axes[1].plot(sample_data['emg1'].values,       color='tab:green', linewidth=0.5)
axes[1].set_title('Filtered EMG1 — Fist (Notch + Bandpass)', fontweight='bold')
for ax in axes:
    ax.set_ylabel('ADC Value')
axes[1].set_xlabel('Sample')
plt.tight_layout()
plt.savefig('filtering_check.png', dpi=150)
plt.show()
print("  Saved → filtering_check.png")

# ── SNR check ─────────────────────────────────────────────────
relax_rms  = np.sqrt(np.mean(df[df['gesture'] == 'relax' ]['emg1'].values ** 2))
fist_rms   = np.sqrt(np.mean(df[df['gesture'] == 'fist'  ]['emg1'].values ** 2))
snr_db     = 20 * np.log10(fist_rms / relax_rms)
print(f"\n  Post-filter SNR (fist vs relax) : {snr_db:.2f} dB")


# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 3 — FEATURE EXTRACTION")
print("═" * 50)

WINDOW_SIZE = 50   # 100ms at 500Hz
STEP        = 25   # 50% overlap

def extract_features(window):
    features = []
    for ch in ['emg1', 'emg2']:
        col = window[ch].values.astype(float)
        features.append(np.mean(np.abs(col)))
        features.append(np.sqrt(np.mean(col ** 2)))
        features.append(np.var(col))
        features.append(np.std(col))
        zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
        features.append(zc)

    features.append(window['roll'].mean())
    features.append(window['pitch'].mean())
    accel_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
    features.append(accel_mag.mean())

    return features  # 13 features

gestures = df['gesture'].unique()
X = []
y = []

for gesture in gestures:
    gesture_data = df[df['gesture'] == gesture].reset_index(drop=True)
    for start in range(0, len(gesture_data) - WINDOW_SIZE, STEP):
        window = gesture_data.iloc[start : start + WINDOW_SIZE]
        X.append(extract_features(window))
        y.append(gesture)

X = np.array(X)
y = np.array(y)

print(f"\n  Feature matrix shape : {X.shape}")
print(f"\n  Windows per gesture  :")
print(pd.Series(y).value_counts().to_string())

# ── rest of your sections 3–8 remain exactly the same ─────────
# ═══════════════════════════════════════════════════════════════
#  SECTION 3 — PREPARE FOR TRAINING
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 3 — PREPARE FOR TRAINING")
print("═" * 50)

le        = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n  Label mapping:")
for i, name in enumerate(le.classes_):
    print(f"    {name} → {i}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_encoded
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 4 — TRAIN AND COMPARE ALL 3 MODELS
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 4 — TRAIN AND COMPARE MODELS")
print("═" * 50)

models = {
    'KNN'          : KNeighborsClassifier(n_neighbors=3),
    'SVM'          : SVC(kernel='rbf', C=10, gamma='scale'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results     = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred            = model.predict(X_test)
    acc               = accuracy_score(y_test, y_pred)
    results[name]     = acc
    predictions[name] = y_pred

    print(f"\n{'─'*40}")
    print(f"  Model    : {name}")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"{'─'*40}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n  ── ACCURACY SUMMARY ──────────────────")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(acc * 20)
    print(f"  {name:15s}  {acc*100:5.1f}%  {bar}")


# ═══════════════════════════════════════════════════════════════
#  SECTION 5 — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 5 — CONFUSION MATRIX")
print("═" * 50)

best_name   = max(results, key=results.get)
best_model  = models[best_name]
y_pred_best = predictions[best_name]

print(f"\n  Best model: {best_name} ({results[best_name]*100:.1f}%)")

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best,
    display_labels = le.classes_,
    cmap           = 'Blues',
    ax             = ax
)
ax.set_title(f'Confusion Matrix — {best_name}  ({results[best_name]*100:.1f}%)',
             fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("  Saved → confusion_matrix.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions[name],
        display_labels = le.classes_,
        cmap           = 'Blues',
        ax             = ax
    )
    ax.set_title(f'{name}\n{results[name]*100:.1f}%', fontweight='bold')

plt.suptitle('Confusion Matrix — All Models', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix_all.png', dpi=150)
plt.show()
print("  Saved → confusion_matrix_all.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 6 — CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 6 — CROSS VALIDATION (5-fold)")
print("═" * 50)

cv_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler',     StandardScaler()),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y_encoded, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"  {name:15s}: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  {'':15s}  Per fold: {[f'{s*100:.1f}%' for s in scores]}")
    print()


# ═══════════════════════════════════════════════════════════════
#  SECTION 7 — FEATURE IMPORTANCE (Random Forest)
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 7 — FEATURE IMPORTANCE")
print("═" * 50)

feature_names = [
    'mav_emg1', 'rms_emg1', 'var_emg1', 'std_emg1', 'zc_emg1',
    'mav_emg2', 'rms_emg2', 'var_emg2', 'std_emg2', 'zc_emg2',
    'roll_mean', 'pitch_mean', 'accel_mag'
]

importances = models['Random Forest'].feature_importances_
sorted_idx  = np.argsort(importances)[::-1]

print("\n  Feature importance ranking:")
for rank, idx in enumerate(sorted_idx):
    bar = '█' * int(importances[idx] * 200)
    print(f"  {rank+1:2d}. {feature_names[idx]:12s}  {importances[idx]:.4f}  {bar}")

plt.figure(figsize=(11, 4))
colors = ['tab:blue'] * 5 + ['tab:orange'] * 5
plt.bar(feature_names, importances, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Importance score')
plt.title('Feature Importance — Random Forest\n(blue = EMG1 Flexor, orange = EMG2 Extensor)',
          fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("\n  Saved → feature_importance.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 8 — SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 8 — SAVE MODEL FILES")
print("═" * 50)

X_all_scaled = scaler.fit_transform(X)
best_model.fit(X_all_scaled, y_encoded)

pickle.dump(best_model, open('gesture_model.pkl', 'wb'))
pickle.dump(scaler,     open('scaler.pkl',         'wb'))
pickle.dump(le,         open('label_encoder.pkl',  'wb'))

print(f"""
  Saved 3 files:

    gesture_model.pkl   ← trained {best_name} model
    scaler.pkl          ← StandardScaler
    label_encoder.pkl   ← maps numbers back to gesture names
""")

print("═" * 50)
print("  ALL DONE")
print(f"  Best model    : {best_name}")
print(f"  Test accuracy : {results[best_name]*100:.1f}%")
cv_mean = cv_results[best_name].mean()
cv_std  = cv_results[best_name].std()
print(f"  CV accuracy   : {cv_mean*100:.1f}% ± {cv_std*100:.1f}%")
print("═" * 50)
