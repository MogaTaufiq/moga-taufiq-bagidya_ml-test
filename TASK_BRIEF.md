# TASK_BRIEF.md — Spec Tugas (Terdistilasi)

> Versi ringkas & rapi dari brief asli, supaya Claude Code punya spec lengkap tanpa dokumen
> mentah yang berantakan. Sumber kebenaran requirement tetap brief asli — ini turunannya.

---

## Tujuan

Membangun model prediksi **travel time per segmen** (antar dua halte berurutan) dari data
GPS historis. Granularitas: **1 baris = 1 segmen antar halte dalam satu perjalanan**.
Metodologi yang diusulkan tim: **ensemble LSTM + XGBoost** (LSTM menangkap pola sekuensial
antar segmen dalam satu loop; XGBoost menangani fitur tabular). Kandidat **boleh** memakai
model lain yang lebih efisien/cocok — dan memang **diminta membandingkan** (lihat aturan
perbandingan model di `CLAUDE.md` §4).

---

## Empat Tugas Fungsional

**Task 1 — EDA & Rekomendasi.**
EDA + feature engineering awal pada data mentah. Beri rekomendasi desain model dan cara
menangani masalah kualitas data (incomplete trips, skewness, sparsity).

**Task 2 — Data Generation (opsional tapi bernilai).**
Boleh membuat data sintetis tambahan yang meniru struktur asli agar analisis lebih
meyakinkan. Identifikasi syarat untuk meningkatkan akurasi prediksi.

**Task 3 — Feature Engineering.**
Bangun fitur baru dari `from_arrival_time_str` / `arrival_time` (mis. hour, weekday vs
weekend) dan dari relasi `traveling_time_sec` ↔ `average_time_sec`.

**Task 4 — Output Dataframe.**
Hasilkan dataframe siap-latih dengan **minimal 2 kolom baru**:
- **`is_gap_suspected`** — menandai perjalanan tidak lengkap (gap `stop_sequence`).
- **`deviation_ratio`** (atau metrik serupa) — menandai anomali (aktual vs baseline).

---

## Write-Up (MAKSIMAL 2 HALAMAN) — wajib mencakup semua poin ini

1. **Strategi incomplete trips** — drop vs interpolasi, dan **syarat** memilih strategi itu.
2. **Penanganan skewness** — **minimal 2 teknik**, plus dampaknya ke **loss function /
   metrik evaluasi** yang dipilih.
3. **Strategi sparsity** — mengatasi kombinasi segment/hour minim data **tanpa overfit ke noise**.
4. **Justifikasi fitur** — usulkan & justifikasi **minimal 5 fitur baru** (di luar kolom
   mentah) dengan alasan domain.
5. **Justifikasi model** — alasan teknis memilih ensemble LSTM + XGBoost (dan model lain yang
   dicoba). Bahas: **skenario di mana ensemble tidak memberi manfaat signifikan**, strategi
   menggabungkan output (**weighted average vs stacking**), dan metrik evaluasi yang dipilih
   mengingat skewness data.
6. **Pengukuran akurasi** — laporkan **MAE, RMSE, MAPE Segment, dan MAE putaran (Loop MAE)**.

> Karena hanya 2 halaman, **padat & tajam**. Alokasi disarankan ada di `METHODOLOGY.md`.

---

## Deliverables

1. **Source code** — `.py` atau `.ipynb` (proses jelas), **trained model**, dan output relevan.
2. **README.md** — cara menjalankan kode + asumsi threshold yang dipakai
   (+ **tabel perbandingan model & alasan pemilihan** — lihat `README_OUTLINE.md`).
3. **Write-up analitik** — PDF atau Markdown, **maks 2 halaman**.
4. **(Bonus)** EDA notebook visual tambahan → poin bonus.

---

## Stack teknis yang ditetapkan

- **Bahasa:** Python 3.10+.
- **Inti:** pandas, numpy (manipulasi data), matplotlib (visualisasi), scikit-learn (preprocessing).
- **Opsional:** seaborn, statsmodels, mlflow, dll.
- **Model ML:** LSTM, XGBoost (+ model efisien lain yang cocok, sesuai keputusan kandidat).

---

## Aturan submission (jangan keliru)

- **Nama folder/repo:** `moga-taufiq-bagidya_ml-test`
  *(brief menulis konvensi `nama-kandidat_ml-test`; gunakan nama lengkap Moga ber-hyphen).*
- **Subject email:** `[Tes Teknikal] AI/ML - Moga Taufiq Bagidya`
- **Pengiriman:** push ke GitHub (public, atau private + undang reviewer) atau kirim `.zip`.

---

## Metrik — definisi singkat (detail rumus di `METHODOLOGY.md`)

- **MAE** — rata-rata |error|; robust, tidak menghukum error besar berlebihan.
- **RMSE** — menghukum error besar; relevan bila salah besar = mahal secara operasi.
- **MAPE Segment** — error persentase rata-rata per segmen; **hati-hati** meledak saat
  aktual ≈ 0 (pertimbangkan sMAPE / filter segmen sangat pendek).
- **Loop MAE (MAE putaran)** — error pada **total waktu satu putaran** (`no_do`), bukan
  per-segmen. **Metrik paling relevan untuk operasi transit** (penjadwalan, headway, SLA).
