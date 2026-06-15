import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# Import modul-modul yang sudah kita buat sebelumnya
from src.utils.config import RF_PARAMS, FEATURES, TARGET, MODEL_SAVE_PATH, PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.preprocessing.loader import load_raw_data
from src.preprocessing.feature_engineering import target_encode_routes
from src.modeling.evaluate import evaluate_model

def run_training():
    print("MEMULAI PROSES PEMBUATAN MODEL AI 🍳\n")

    # Ambil bahan mentah dari gudang
    train_df = load_raw_data(TRAIN_DATA_PATH)
    test_df = load_raw_data(TEST_DATA_PATH)
    if train_df is None or test_df is None:
        return
        
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Salinan data bersih disimpan di folder processed.")
    
    # Target Encoding (AI belajar bobot rute HANYA dari data masa lalu)
    train_df, test_df, route_target_mean = target_encode_routes(train_df, test_df)
    
    # Siapkan Fitur dan Target
    print(f"\n Menggunakan fitur: {FEATURES}")
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    # Masukkan ke Oven (Training Model)
    print("\n Memulai proses training Random Forest...")
    rf_model = RandomForestRegressor(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    print("Training selesai!")
    
    # Evaluasi Performa
    predictions = rf_model.predict(X_test)
    evaluate_model(predictions, y_test)
    
    # Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Simpan model, kamus rute, dan daftar fitur agar mudah dipakai di tahap selanjutnya
    saved_artifacts = {
        'model': rf_model,
        'route_encoder': route_target_mean,
        'features': FEATURES
    }
    joblib.dump(saved_artifacts, MODEL_SAVE_PATH)
    print(f"Model AI berhasil disimpan di:\n   {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    run_training()