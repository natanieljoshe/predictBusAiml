import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mta_bus_all_routes_weekly.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mta_bus_all_routes_weekly_2024.csv")

TRAIN_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "train_2023_cleaned.csv")
TEST_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "test_2024_cleaned.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "rf_baseline_2324.pkl")
OPTIMIZED_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_final_optimized_2324.pkl")

# Hyperparameters 
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}

FEATURES = ['route_encoded', 'hour', 'day_of_week', 'month']
TARGET = 'total_ridership'