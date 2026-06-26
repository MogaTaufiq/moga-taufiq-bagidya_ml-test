import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.data import load_raw, add_gap_flag, clean_outliers
from src.features import add_time_features, TARGET

print("Loading data...")
raw = load_raw()
flagged = add_gap_flag(raw)
df, _ = clean_outliers(flagged)
df = add_time_features(df)

print("\n--- 1. Analisis Jam Rawan (Rush Hour) vs Luar Jam Rawan ---")
# Menghitung statistik waktu tempuh pada jam rawan vs luar jam rawan
rush_stats = df.groupby('is_rush_hour')[TARGET].agg(['mean', 'median', 'std', 'count'])
rush_stats.index = ['Luar Jam Rawan (0)', 'Jam Rawan (1)']
print(rush_stats)

print("\n--- 2. Deteksi Titik Kemacetan (Bottleneck Segments) ---")
# Baseline: Median waktu tempuh per segmen di LUAR jam rawan (kondisi lancar)
baseline_freeflow = df[df['is_rush_hour'] == 0].groupby('segment_id')[TARGET].median().rename('freeflow_median')

# Hitung 'congestion_index' (rasio waktu tempuh aktual thd kondisi lancar)
df = df.join(baseline_freeflow, on='segment_id')
df['congestion_index'] = df[TARGET] / (df['freeflow_median'] + 1) # +1 utk hindari div/0

# Segmen mana yang paling macet selama jam rawan?
rush_congestion = df[df['is_rush_hour'] == 1].groupby('segment_id').agg(
    avg_congestion_index=('congestion_index', 'mean'),
    median_travel_time=(TARGET, 'median'),
    freeflow_median=('freeflow_median', 'first'),
    count=('congestion_index', 'count')
).reset_index()
# Filter yang punya cukup data, lalu urutkan dari yang paling parah
bottlenecks = rush_congestion[rush_congestion['count'] > 100].sort_values('avg_congestion_index', ascending=False)
print("Top 5 Segmen Paling Macet (Rasio waktu tempuh aktual thd lancar tertinggi):")
print(bottlenecks.head(5))

print("\n--- 3. Deteksi Anomali Statistik Khusus per Segmen ---")
# Menghitung Z-Score (standarisasi) dari waktu tempuh khusus untuk tiap segment_id dan hour
# Ini lebih presisi dari sekadar "jam rawan" vs "tidak", karena menangkap anomali di jam tertentu
df['segment_hour_mean'] = df.groupby(['segment_id', 'hour'])[TARGET].transform('mean')
df['segment_hour_std'] = df.groupby(['segment_id', 'hour'])[TARGET].transform('std')
df['z_score_segment_hour'] = (df[TARGET] - df['segment_hour_mean']) / (df['segment_hour_std'] + 1)

# Identifikasi anomali: Z-score > 3 (3 standar deviasi lebih lambat dari biasanya di jam dan rute tsb)
df['is_anomaly_slow'] = (df['z_score_segment_hour'] > 3).astype(int)
print(f"Persentase anomali macet ekstrem (Z > 3): {df['is_anomaly_slow'].mean() * 100:.2f}%")

print("\n--- 4. Korelasi Fitur Baru terhadap Target (Log Traveling Time) ---")
df['log_target'] = np.log1p(df[TARGET])
corr_features = ['is_rush_hour', 'congestion_index', 'z_score_segment_hour', 'is_gap_suspected', 'is_anomaly_slow']
corrs = df[corr_features + ['log_target']].corr()['log_target'].drop('log_target').sort_values(ascending=False)
print(corrs)

