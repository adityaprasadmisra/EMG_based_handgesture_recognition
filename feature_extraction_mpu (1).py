import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    'person2': 'data7.csv',
    'person3': 'data8.csv',

}

COLUMNS = ['emg1', 'emg2', 'ax', 'ay', 'az', 'gx', 'gy', 'roll', 'pitch', 'gesture']

frames = []
for person, filename in CSV_FILES.items():
    temp = pd.read_csv(filename, header=0)
    temp.columns = COLUMNS
    temp['person'] = person          # optional: track which person each row came from
    frames.append(temp)
    print(f"  Loaded {filename:25s} → {len(temp)} rows")

# Combine all three into one dataframe
df = pd.concat(frames, ignore_index=True)
print(f"\n  Combined total rows : {len(df)}")

# ── CLEAN ────────────────────────────────────────────────────────
df['emg1'] = pd.to_numeric(df['emg1'], errors='coerce')
df['emg2'] = pd.to_numeric(df['emg2'], errors='coerce')

before = len(df)
df = df.dropna(subset=COLUMNS)      # only drop rows where sensor columns are NaN
after  = len(df)
if before != after:
    print(f"  Dropped {before - after} corrupted/empty rows")

# Clean label column
df['gesture'] = df['gesture'].str.strip()

# Convert to integer
df['emg1'] = df['emg1'].astype(int)
df['emg2'] = df['emg2'].astype(int)

# Remove readings outside valid 12-bit ADC range
df = df[(df['emg1'] >= 0) & (df['emg1'] <= 4095)]
df = df[(df['emg2'] >= 0) & (df['emg2'] <= 4095)]

print(f"\n  Rows after cleaning : {len(df)}")
print(f"  Column types        :\n{df.dtypes}")
print(f"\n  Samples per gesture :")
print(df['gesture'].value_counts().to_string())
print(f"\n  Signal statistics   :")
print(df[['emg1','emg2']].describe().to_string())
print(f"\n  Missing values      : {df.isnull().sum().sum()}")

# ── Plot each gesture ────────────────────────────────────────────
gestures = df['gesture'].unique()
n = len(gestures)

fig, axes = plt.subplots(n, 1, figsize=(12, n * 2))
if n == 1:
    axes = [axes]

for i, gesture in enumerate(gestures):
    subset = df[df['gesture'] == gesture].head(200)
    axes[i].plot(subset['emg1'].values, label='EMG1 (Flexor)',   color='tab:blue')
    axes[i].plot(subset['emg2'].values, label='EMG2 (Extensor)', color='tab:orange')
    axes[i].set_title(f'Gesture: {gesture.upper()}', fontweight='bold')
    axes[i].set_ylabel('ADC value (0–4095)')
    axes[i].legend(loc='upper right')
    axes[i].set_ylim(0, 4095)

plt.suptitle('Raw EMG Signal Per Gesture (first 200 samples)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('gesture_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved → gesture_plots.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

print("\n" + "═" * 50)
print("  SECTION 2 — FEATURE EXTRACTION")
print("═" * 50)

WINDOW_SIZE = 50    # 50 samples = 100ms at 500Hz
STEP        = 25    # 50% overlap

def extract_features(window):
    """
    2 EMG channels × 5 features = 10
    Roll + Pitch mean             =  2
    Accel magnitude mean          =  1
    Total                         = 13 features
    """
    features = []

    # EMG features
    for ch in ['emg1', 'emg2']:
        col = window[ch].values.astype(float)
        features.append(np.mean(np.abs(col)))               # MAV
        features.append(np.sqrt(np.mean(col ** 2)))         # RMS
        features.append(np.var(col))                        # Variance
        features.append(np.std(col))                        # Std Dev
        zc = np.sum(np.diff(np.sign(col - np.mean(col))) != 0)
        features.append(zc)                                 # Zero Crossing

    # IMU features — mean angle over the window
    features.append(window['roll'].mean())                  # Wrist roll
    features.append(window['pitch'].mean())                 # Wrist pitch

    # Resultant acceleration magnitude — captures overall wrist movement
    accel_mag = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
    features.append(accel_mag.mean())

    return features  # 13 features total
X = []
y = []

for gesture in gestures:
    gesture_data = df[df['gesture'] == gesture].reset_index(drop=True)
    windows_this_gesture = 0

    for start in range(0, len(gesture_data) - WINDOW_SIZE, STEP):
        window = gesture_data.iloc[start : start + WINDOW_SIZE]
        X.append(extract_features(window))
        y.append(gesture)
        windows_this_gesture += 1

X = np.array(X)
y = np.array(y)

print(f"\n  Feature matrix shape  : {X.shape}   ← (total_windows, 10_features)")
print(f"\n  Windows per gesture after windowing:")
print(pd.Series(y).value_counts().to_string())


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
