import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def target_encode_routes(train_df, test_df):
    """
    Mengubah nama rute bus menjadi angka berdasarkan rata-rata historis (Target Encoding)
    Mencegah bias urutan ordinal dari LabelEncoder.
    """
    print("Menerapkan Target Encoding pada rute bus...")
    
    # Hitung rata-rata murni hanya dari data Train (Mencegah Data Leakage)
    route_target_mean = train_df.groupby('bus_route')['total_ridership'].mean().to_dict()
    
    # Terapkan ke kedua set data
    train_df['route_encoded'] = train_df['bus_route'].map(route_target_mean)
    test_df['route_encoded'] = test_df['bus_route'].map(route_target_mean).fillna(0)
    
    return train_df, test_df, route_target_mean

