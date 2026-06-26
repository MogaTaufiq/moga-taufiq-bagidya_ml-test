# %% [markdown]
# # 01 — EDA: Bus Segment Travel-Time Prediction
#
# **Tujuan:** memahami struktur data, membongkar dua jebakan kritis
# (leakage `average_time_sec`; `no_do` vs `trip_id` sebagai unit loop), menetapkan
# threshold cleaning, dan menemukan identitas target. Semua angka **direproduksi dari
# parquet mentah**; logika reusable hidup di `src/`. Keputusan dicatat di `DECISIONS.md`.

# %%
import sys; sys.path.insert(0, "..")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_raw, validate_raw, add_gap_flag

raw = load_raw()
print("shape:", raw.shape)
validate_raw(raw)

# %% [markdown]
# ## A. Struktur & jebakan #1 — `no_do` adalah unit loop, BUKAN `trip_id`
# `trip_id` hanya 13 nilai untuk 351k baris (~27rb baris/trip → mustahil satu loop fisik):
# itu **varian rute**. `no_do` = 17.857 loop, median 19 segmen, `stop_sequence` urut rapi
# di dalamnya → **inilah satu putaran**. Implikasi: sequence key LSTM & Loop MAE pakai `no_do`.

# %%
for c in ["route_code", "trip_id", "no_do", "segment_id", "bus_body_no"]:
    print(f"{c:14s}: {raw[c].nunique()} unik")
seg = raw.groupby("no_do").size()
print(f"\nsegmen/loop: median={seg.median():.0f}, range={seg.min()}-{seg.max()}")
print(f"rata-rata baris/trip_id: {len(raw)/raw['trip_id'].nunique():.0f}")

# %% [markdown]
# ## B. Jebakan #2 — `average_time_sec` adalah TARGET LEAKAGE
# `deviation_ratio = traveling/average` ter-cap **persis di [0, 2]** dengan **0% di atas 2**.
# Itu mustahil untuk baseline historis nyata (macet/insiden → rasio >2×). Korelasi 0.84.
# Empiris (ablation di `03`): menambahkannya memangkas MAE ~44%. → **jangan dipakai mentah**;
# kita recompute baseline leakage-free (expanding mean per (segment,hour), di-shift).

# %%
ratio = raw["traveling_time_sec"] / raw["average_time_sec"].replace(0, np.nan)
print(f"deviation_ratio: min={ratio.min():.3f} median={ratio.median():.3f} "
      f"max={ratio.max():.3f} | %>2.0 = {100*(ratio>2).mean():.4f}%")
print(f"corr(traveling, average) = {raw[['traveling_time_sec','average_time_sec']].corr().iloc[0,1]:.3f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ratio.clip(0, 2.5).dropna(), bins=60, color="#c0392b")
ax.axvline(2.0, ls="--", c="k", lw=2, label="cap = 2.0")
ax.set(title="deviation_ratio caps EXACTLY at 2.0 (leakage signature)",
       xlabel="traveling / average", ylabel="count"); ax.legend(); plt.show()

# %% [markdown]
# ## C. Skewness ekstrem + outlier
# Skew mentah 58 → `log1p` 2.2. Max 958.259 dtk (~11 hari) = mustahil untuk satu segmen.
# Patahan tajam p99=32mnt (wajar) → p99.5=6.6jam (mustahil) → **threshold drop > 3600s (1 jam)**.

# %%
t = raw["traveling_time_sec"]
print(f"target: median={t.median():.1f}s mean={t.mean():.1f}s max={t.max():.1f}s")
print(f"skew raw={t.skew():.2f} -> log1p={np.log1p(t).skew():.2f}")
for q in [0.99, 0.995, 0.999]:
    print(f"  p{q*100:.1f} = {t.quantile(q):.0f}s ({t.quantile(q)/60:.1f} min)")
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(t.clip(0, t.quantile(.99)), bins=60, color="#2980b9"); axs[0].set_title(f"raw (skew {t.skew():.1f})")
axs[1].hist(np.log1p(t), bins=60, color="#27ae60"); axs[1].set_title(f"log1p (skew {np.log1p(t).skew():.2f})")
plt.show()

# %% [markdown]
# ## D. Incomplete trips (gap) & sparsity
# 47.4% loop punya gap (`stop_sequence.diff()>1`); 179 loop hanya 1 segmen.
# Sparsity ringan: hanya 3.3% kombinasi (segment,hour) punya <30 observasi.

# %%
flagged = add_gap_flag(raw)
loops_gap = int(flagged.groupby("no_do")["is_gap_suspected"].first().sum())
print(f"loop bergap: {loops_gap}/{flagged['no_do'].nunique()} = {100*loops_gap/flagged['no_do'].nunique():.1f}%")
raw_h = raw.assign(hour=raw["departure_time"].dt.hour)
sh = raw_h.groupby(["segment_id", "hour"]).size()
print(f"(segment,hour) combos={len(sh)} | <30 obs={100*(sh<30).mean():.1f}% | <10 obs={100*(sh<10).mean():.1f}%")
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(seg.clip(0, 40), bins=40, color="#8e44ad"); ax.axvline(seg.median(), ls="--", c="k")
ax.set(title="segmen per loop (range 1-72)", xlabel="n segmen", ylabel="count"); plt.show()

# %% [markdown]
# ## E. Temuan kunci: target = selisih timestamp → `arrival_time` post-hoc
# `traveling_time_sec == arrival_time − from_arrival_time_str` untuk **100% baris** (<1s).
# Artinya `arrival_time` baru ada **setelah** segmen selesai → memakainya = leakage.
# **Semua fitur waktu diambil dari `from_arrival_time_str` (departure).**

# %%
delta = (raw["arrival_time"] - raw["departure_time"]).dt.total_seconds()
print(f"max |delta - traveling| = {(delta - raw['traveling_time_sec']).abs().max():.3f}s")
print(f"baris arrival < departure: {(delta < 0).sum()}")
print(f"span DEPARTURE: {raw['departure_time'].min()} .. {raw['departure_time'].max()} "
      f"(= Februari, {raw['departure_time'].dt.normalize().nunique()} hari)")

# %% [markdown]
# ## Ringkasan keputusan (detail di `DECISIONS.md`)
# - **D2** `average_time_sec` leakage → recompute baseline bersih.
# - **D3** drop `traveling_time_sec > 3600s` (0.85% baris).
# - **D4** `is_gap_suspected = diff>1`; kebijakan drop spesifik-model (tabular simpan+flag).
# - **D6** split time-based loop-aware, cut 2026-02-22 (test = 7 hari terakhir).
# - **D11** fitur waktu dari departure; `arrival_time` di-drop.
