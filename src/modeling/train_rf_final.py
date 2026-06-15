import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Import modul kita
from src.utils.config import FEATURES, TARGET, OPTIMIZED_MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.preprocessing.loader import load_raw_data
from src.preprocessing.feature_engineering import target_encode_routes
from src.modeling.evaluate import evaluate_model

def run_final_training():
    print("MEMULAI TRAINING FINAL (LOG-TRANSFORM + BEST PARAMS)\n")

    # Load & Siapkan Bahan
    train_df = load_raw_data(TRAIN_DATA_PATH)
    test_df = load_raw_data(TEST_DATA_PATH)
    if train_df is None or test_df is None: return
    
    train_df, test_df, route_target_mean = target_encode_routes(train_df, test_df)
    
    # Simpan data yang sudah bersih ke folder processed
    from src.utils.config import TRAIN_PROCESSED_PATH, TEST_PROCESSED_PATH
    os.makedirs(os.path.dirname(TRAIN_PROCESSED_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_PROCESSED_PATH, index=False)
    test_df.to_csv(TEST_PROCESSED_PATH, index=False)
    print("Salinan data bersih (Train & Test) berhasil disimpan di folder processed.")
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # LOGARITHMIC TRANSFORMATION
    print("Menerapkan transformasi Logaritmik (np.log1p) pada data target...")
    y_train_log = np.log1p(y_train)

    # Memasukkan Hyperparameter Pemenang dari Eksperimen Sebelumnya
    best_params = {
        'n_estimators': 300,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'random_state': 42,
        'n_jobs': -1
    }

    # Training Model
    print("\nMemulai proses training Random Forest...")
    rf_model = RandomForestRegressor(**best_params)
    
    # melatih model menggunakan y_train_log
    rf_model.fit(X_train, y_train_log)
    print("Training selesai!")

    # Prediksi & Kembalikan ke Skala Asli
    print("\n Mengembalikan skala prediksi ke angka asli (np.expm1)...")
    predictions_log = rf_model.predict(X_test)
    
    # Kembalikan log ke angka normal, lalu bulatkan ke bilangan bulat
    predictions = np.expm1(predictions_log)
    predictions_rounded = np.round(predictions).astype(int)

    # Cek Hasil Akhir
    evaluate_model(predictions_rounded, y_test)

    # Simpan Model Terbaik
    os.makedirs(os.path.dirname(OPTIMIZED_MODEL_PATH), exist_ok=True)
    saved_artifacts = {
        'model': rf_model,             # Model yang memprediksi dalam bentuk log
        'route_encoder': route_target_mean,
        'features': FEATURES,
        'is_log_transformed': True     # Penanda agar predict.py nanti tahu harus di-expm1
    }
    joblib.dump(saved_artifacts, OPTIMIZED_MODEL_PATH)
    print(f" Model AI Final berhasil disimpan di:\n   {OPTIMIZED_MODEL_PATH}")

if __name__ == "__main__":
    run_final_training()
