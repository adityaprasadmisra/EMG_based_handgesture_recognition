#taking fake data
# import numpy as np, pandas as pd

# gestures = ['fist', 'open', 'flex', 'extend', 'pinch']
# rows = []
# for g in gestures:
#     for _ in range(200):
#         # Fake different activation patterns per gesture
#         rows.append([np.random.randint(200,800),
#                      np.random.randint(100,400),
#                      np.random.randint(300,900), g])

# pd.DataFrame(rows, columns=['ch1','ch2','ch3','label']).to_csv('fake_data.csv', index=False)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('fake_data.csv', names=['ch1', 'ch2', 'ch3', 'label'])

# Clean labels first — removes accidental spaces
df['label'] = df['label'].str.strip()

print(df.shape)
print(df['label'].value_counts())
print(df.describe())

# Fix: don't hardcode 5, use actual number of gestures
gestures = df['label'].unique()
n = len(gestures)

fig, axes = plt.subplots(n, 1, figsize=(12, n * 2))

# Also fix: if n=1, axes is not a list — this handles that edge case
if n == 1:
    axes = [axes]

for i, gesture in enumerate(gestures):
    subset = df[df['label'] == gesture].head(200)
    axes[i].plot(subset['ch1'].values, label='ch1')
    axes[i].plot(subset['ch2'].values, label='ch2')
    axes[i].plot(subset['ch3'].values, label='ch3')
    axes[i].set_title(gesture)
    axes[i].legend()

plt.tight_layout()
plt.savefig('gesture_plots.png')
plt.show()
