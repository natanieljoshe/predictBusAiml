import pandas as pd

def load_raw_data(filepath):
    print(f"Membaca dataset dari: {filepath.split('/')[-1]}...")
    try:
        df = pd.read_csv(filepath)
        if 'transit_timestamp' in df.columns:
            df['transit_timestamp'] = pd.to_datetime(df['transit_timestamp'])
        print(f"Berhasil memuat {df.shape[0]:,} baris dan {df.shape[1]} kolom.")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {filepath}")
        return None