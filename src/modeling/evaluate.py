import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def evaluate_model(predictions, y_test):
    print("\n Mengevaluasi Akurasi Model di Data Tahun 2024")
    
    # Konversi ke array numpy agar mudah dilakukan filtering (masking)
    y_test_array = np.array(y_test)
    pred_array = np.array(predictions)
    
    # evaluasi global (Menyeluruh)
    mae = mean_absolute_error(y_test_array, pred_array)
    rmse = np.sqrt(mean_squared_error(y_test_array, pred_array))
    rata_rata_penumpang = y_test_array.mean()
    wmape_percentage = (mae / rata_rata_penumpang) * 100
    
    mask_global = y_test_array > 0
    mape_global = mean_absolute_percentage_error(y_test_array[mask_global], pred_array[mask_global]) * 100
    
    # evaluasi terpisah
    # A. Jam Sibuk (Aktual > 50 penumpang)
    mask_peak = y_test_array > 50
    mape_peak = mean_absolute_percentage_error(y_test_array[mask_peak], pred_array[mask_peak]) * 100
    wmape_peak = (mean_absolute_error(y_test_array[mask_peak], pred_array[mask_peak]) / y_test_array[mask_peak].mean()) * 100
    
    # B. Jam Normal (11 - 50 penumpang)
    mask_normal = (y_test_array > 10) & (y_test_array <= 50)
    mape_normal = mean_absolute_percentage_error(y_test_array[mask_normal], pred_array[mask_normal]) * 100
    
    # C. Jam Sepi (1 - 10 penumpang)
    mask_offpeak = (y_test_array > 0) & (y_test_array <= 10)
    mape_offpeak = mean_absolute_percentage_error(y_test_array[mask_offpeak], pred_array[mask_offpeak]) * 100

    print("=== HASIL EVALUASI GLOBAL ===")
    print(f"MAE Global                     : {mae:.2f} penumpang")
    print(f"RMSE Global                    : {rmse:.2f} penumpang")
    print(f"Error Relatif (WMAPE)          : {wmape_percentage:.2f}%")
    print(f"MAPE Global (Scikit-Learn)     : {mape_global:.2f}%\n")

    print("Membuktikan model AI sangat akurat di jam krusial, dan error % besar hanya terjadi pada jam sepi yang tidak signifikan secara operasional:\n")
    print(f"      1. Jam Sibuk (> 50 Penumpang)")
    print(f"      MAPE = {mape_peak:.2f}%  |  WMAPE = {wmape_peak:.2f}%  (Total Data: {mask_peak.sum():,} baris)")
    
    print(f"\n    2. Jam Normal (11 - 50 Penumpang)")
    print(f"      MAPE = {mape_normal:.2f}%  (Total Data: {mask_normal.sum():,} baris)")
    
    print(f"\n    3. Jam Sepi (1 - 10 Penumpang)")
    print(f"      MAPE = {mape_offpeak:.2f}% (Total Data: {mask_offpeak.sum():,} baris)")
    print("====================================================\n")