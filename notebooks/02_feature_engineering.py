# %% [markdown]
# # 02 — Cleaning & Feature Engineering
#
# Dari data mentah → frame siap-latih. Urutan: **gap flag → drop outlier → fitur
# leakage-free → split loop-aware → encoder fit-di-train**. Menghasilkan
# `outputs/training_ready.parquet` dengan kolom wajib `is_gap_suspected` + `deviation_ratio`.

# %%
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
from src.data import load_raw, add_gap_flag, clean_outliers, time_split, sequence_ready
from src import features as F, models as Mdl

raw = load_raw()
flagged = add_gap_flag(raw)                       # is_gap_suspected + loop_n_segments (RAW order)
clean, report = clean_outliers(flagged)          # drop > 3600s (D3)
print(report)
print(f"\nFitur dipakai model = {len(F.FEATURE_COLS)} fitur (termasuk 2 fitur tambahan: time_since_prev_arrival_sec + bus_encoded)")

# %% [markdown]
# ## Fitur leakage-free (dihitung di frame penuh, time-ordered, di-shift)
# `baseline_segment_hour` = expanding mean per (segment,hour) **shifted** = pengganti bersih
# `average_time_sec`. `rolling_mean_segment_5`, `trip_progress`, cyclic time, dll.

# %%
feat = F.build_base_features(clean)
new_cols = ["hour", "is_weekend", "is_rush_hour", "hour_sin", "trip_progress",
            "baseline_segment_hour", "rolling_mean_segment_5", "deviation_ratio"]
feat[new_cols].head()

# %% [markdown]
# ## Split loop-aware time-based (D6) + encoder fit-di-train (anti-leakage)
# Tiap `no_do` utuh di satu sisi (tak straddle) → Loop MAE tak rusak, tak bocor antar-split.

# %%
tr, te = time_split(feat)
tr, te = F.add_dwell_feature(tr, te)              # D12: time_since_prev_arrival_sec (leakage-free, combined-frame)
art = F.fit_feature_artifacts(tr)                 # volatility, kategori, baseline-model, bus_encoded (D13): TRAIN saja
tr = F.transform_with_artifacts(tr, art)
te = F.transform_with_artifacts(te, art)
print(f"train {len(tr)} ({tr['departure_time'].min().date()}..{tr['departure_time'].max().date()}) | "
      f"test {len(te)} ({te['departure_time'].min().date()}..{te['departure_time'].max().date()})")
print("overlap loop train∩test (harus 0):", len(set(tr['no_do']) & set(te['no_do'])))

# %% [markdown]
# ## Bukti TIDAK ada leakage
# XGBoost jujur (17 fitur, tanpa `average_time_sec`): **median AE ~12s** — kalau ada fitur
# yang membocorkan target, median AE jatuh ke ~0. `baseline_segment_hour` korelasi 0.69 (wajar);
# `time_since_prev_arrival_sec` korelasi linear hampir 0 (-0.001) tapi sinyal non-linear kuat (lihat permutation test di DECISIONS D12).

# %%
Xtr, ytr_log, _ = Mdl.make_xy(tr); Xte, _, yte = Mdl.make_xy(te)
m = Mdl.train_xgb(Xtr, ytr_log)
pred = Mdl.predict_sec(m, Xte)
print(f"median AE = {np.median(np.abs(yte - pred)):.2f}s  (jujur, bukan ~0)")
print(f"corr(baseline_segment_hour, target) = {tr[['baseline_segment_hour','traveling_time_sec']].corr().iloc[0,1]:.3f}")
print("FEATURE_COLS:", F.FEATURE_COLS)
print("output-only (mengandung target, BUKAN fitur):", F.OUTPUT_ONLY_COLS)

# %% [markdown]
# ## Kolom output wajib + simpan training-ready dataframe

# %%
tr["split"], te["split"] = "train", "test"
ready = pd.concat([tr, te]).sort_values(["no_do", "stop_sequence"]).reset_index(drop=True)
print("is_gap_suspected:", ready["is_gap_suspected"].dtype,
      "| deviation_ratio terisi:", f"{ready['deviation_ratio'].notna().mean()*100:.1f}%")
ready.to_parquet("../outputs/training_ready.parquet", index=False)
print("saved -> outputs/training_ready.parquet", ready.shape)
ready[["no_do", "stop_sequence", "segment_id", "traveling_time_sec",
       "is_gap_suspected", "deviation_ratio", "baseline_segment_hour"]].head()
