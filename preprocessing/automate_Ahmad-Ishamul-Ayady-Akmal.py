"""
Automated Preprocessing Script
Dataset: Teen Social Media Usage & Mental Health
Author: Ahmad Ishamul Ayady Akmal
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import sys


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset dari path yang diberikan."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset berhasil dimuat. Shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing values dengan median (numerik) dan modus (kategorikal)."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    missing_before = df.isnull().sum().sum()

    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values: {missing_before} -> {missing_after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Duplikat dihapus: {before - after} baris. Shape: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan encoding pada kolom kategorikal."""
    df = df.copy()
    le = LabelEncoder()

    df['gender_encoded'] = le.fit_transform(df['gender'])
    df['platform_encoded'] = le.fit_transform(df['platform_usage'])
    df['social_interaction_encoded'] = le.fit_transform(df['social_interaction_level'])

    # Encode target dengan urutan yang konsisten
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    df['depression_risk_encoded'] = df['depression_risk'].map(label_map)

    print("[INFO] Encoding kategorikal selesai.")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi fitur numerik menggunakan MinMaxScaler."""
    df = df.copy()
    scaler = MinMaxScaler()

    cols_to_scale = [
        'age', 'daily_social_media_hours', 'sleep_hours',
        'screen_time_before_sleep', 'academic_performance',
        'physical_activity', 'stress_level', 'anxiety_level'
    ]

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    print("[INFO] Normalisasi fitur numerik selesai.")
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Memilih kolom fitur dan target untuk dataset final."""
    feature_cols = [
        'age', 'daily_social_media_hours', 'sleep_hours',
        'screen_time_before_sleep', 'academic_performance',
        'physical_activity', 'stress_level', 'anxiety_level',
        'gender_encoded', 'platform_encoded', 'social_interaction_encoded'
    ]
    target_col = 'depression_risk_encoded'

    df_final = df[feature_cols + [target_col]].copy()
    print(f"[INFO] Dataset final shape: {df_final.shape}")
    return df_final


def save_preprocessed(df: pd.DataFrame, output_path: str) -> None:
    """Menyimpan dataset yang sudah dipreprocessing ke file CSV."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset preprocessed disimpan ke: {output_path}")


def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """Menjalankan seluruh pipeline preprocessing."""
    print("=" * 50)
    print("Memulai proses preprocessing...")
    print("=" * 50)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categoricals(df)
    df = normalize_features(df)
    df_final = select_final_features(df)
    save_preprocessed(df_final, output_path)

    print("=" * 50)
    print("Preprocessing selesai!")
    print("=" * 50)
    return df_final


if __name__ == "__main__":
    # Path relatif (sesuaikan jika perlu)
    INPUT_PATH = "../Teen_Mental_Health_Dataset.csv"
    OUTPUT_PATH = "Teen_Mental_Health_preprocessed.csv"

    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_PATH = sys.argv[2]

    df_result = run_preprocessing(INPUT_PATH, OUTPUT_PATH)
    print(f"\nSample data preprocessed:\n{df_result.head()}")
