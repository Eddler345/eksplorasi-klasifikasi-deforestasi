import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

# Menjaga konsistensi random seed
np.random.seed(42)
tf.random.set_seed(42)

print('=== EKSPLORASI KLASIFIKASI DEFORESTASI BERBASIS DATA SINTETIS ===')

# ================= ================= =================
# 1. PEMBANGKITAN DATA SINTETIS UNTUK DEMONSTRASI
# ================= ================= =================
n_samples = 2500
n_features = 36  # Fitur tiruan; bukan pengukuran satelit yang sebenarnya

# X berisi fitur acak sintetis untuk demonstrasi alur klasifikasi
X = np.random.randn(n_samples, n_features)

# y berisi label kelas yang dipilih secara acak; tidak berasal dari observasi lapangan
y = np.random.choice([0, 1, 2], size=n_samples, p=[0.50, 0.30, 0.20])

print(f'Jumlah sampel sintetis: {n_samples}')
print(f'Distribusi label sintetis (pertanian, pertambangan, penebangan): {np.bincount(y)}\n')

# ================= ================= =================
# 2. DATA SPLIT (TRAINING & TESTING)
# ================= ================= =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= ================= =================
# 3. MEMBANGUN ARSITEKTUR DEEP LEARNING (NEURAL NETWORK)
# ================= ================= =================
model = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Mencegah overfitting pada data citra
    layers.Dense(32, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 kelas pemicu deforestasi
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print('Melatih model demonstrasi pada data sintetis...')
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    verbose=1
)

# ================= ================= =================
# 4. EVALUASI DAN EKSPOR HASIL PREDIKSI
# ================= ================= =================
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print('\n--- LAPORAN EVALUASI PADA DATA SINTETIS ---')
print(classification_report(y_test, y_pred, target_names=['Pertanian (0)', 'Pertambangan (1)', 'Penebangan (2)']))

# Menyimpan ringkasan hasil prediksi ke CSV
df_output = pd.DataFrame({
    'actual_driver': y_test,
    'predicted_driver': y_pred,
    'confidence_score': np.max(y_pred_probs, axis=1)
})
df_output.to_csv('hasil_klasifikasi_deforestasi.csv', index=False)
print('[SELESAI] Hasil prediksi disimpan ke \"hasil_klasifikasi_deforestasi.csv\"')