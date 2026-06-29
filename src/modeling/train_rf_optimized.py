import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Import modul
from src.utils.config import FEATURES, TARGET, OPTIMIZED_MODEL_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH
from src.preprocessing.loader import load_raw_data
from src.preprocessing.feature_engineering import target_encode_routes
from src.modeling.evaluate import evaluate_model

def run_optimization():
    print("MEMULAI PROSES HYPERPARAMETER TUNING\n")

    # Load & Siapkan Bahan
    train_df = load_raw_data(TRAIN_DATA_PATH)
    test_df = load_raw_data(TEST_DATA_PATH)
    if train_df is None or test_df is None: return
    
    train_df, test_df, route_target_mean = target_encode_routes(train_df, test_df)
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Tentukan Rentang Eksperimen 
    param_distributions = {
        'n_estimators': [100, 200, 300],          
        'max_depth': [10, 20, 30, None],          
        'min_samples_split': [2, 5, 10],          
        'min_samples_leaf': [1, 2, 4]             
    }

    print("\n Rentang Hyperparameter yang akan diuji:")
    for key, val in param_distributions.items():
        print(f"   - {key}: {val}")

    # TimeSeriesSplit agar saat proses Tuning, 
    # model tidak mengalami data leakage (tetap urut waktu).
    tscv = TimeSeriesSplit(n_splits=3)

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    rf_tuner = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=10,               # 10 kombinasi berbeda
        cv=tscv,                 # pemisahan waktu
        scoring='neg_mean_absolute_error', # MAE yang paling kecil
        verbose=2,               
        random_state=42,
        n_jobs=-1                
    )

     
    print("\n Memulai pencarian kombinasi terbaik (Ini akan memakan waktu beberapa menit)...")
    rf_tuner.fit(X_train, y_train)

    best_model = rf_tuner.best_estimator_
    print("\n HYPERPARAMETER TERBAIK DITEMUKAN:")
    for key, val in rf_tuner.best_params_.items():
        print(f"    {key}: {val}")

    print("\n Menguji Model Pemenang ke Data Test...")
    predictions = best_model.predict(X_test)
    evaluate_model(predictions, y_test)

    # Simpan Model Terbaik
    os.makedirs(os.path.dirname(OPTIMIZED_MODEL_PATH), exist_ok=True)
    saved_artifacts = {
        'model': best_model,
        'route_encoder': route_target_mean,
        'features': FEATURES,
        'best_params': rf_tuner.best_params_
    }
    joblib.dump(saved_artifacts, OPTIMIZED_MODEL_PATH)
    print(f"Model AI yang teroptimasi berhasil disimpan di:\n   {OPTIMIZED_MODEL_PATH}")

if __name__ == "__main__":
    run_optimization()