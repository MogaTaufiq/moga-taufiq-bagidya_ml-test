# README_OUTLINE.md — Struktur Wajib README Final

> README.md adalah **deliverable** (bukan file ini). File ini cuma cetakannya. Claude Code:
> isi `README.md` mengikuti struktur di bawah saat semua hasil sudah ada. Bahasa boleh
> Indonesia atau Inggris — reviewer Transjakarta kemungkinan Indonesia, jadi Indonesia aman.
> Yang **tidak boleh hilang**: bagian **Perbandingan Model** dan **Asumsi/Threshold**.

---

## Kerangka README.md

```
# [Nama Kandidat] — ML Test: Bus Segment Travel-Time Prediction

## 1. Ringkasan
- 2–3 kalimat: masalah, pendekatan, hasil utama (sebut Loop MAE final & lift vs baseline).

## 2. Cara Menjalankan (Reproducibility)
- Versi Python, cara buat venv, `pip install -r requirements.txt`.
- Urutan menjalankan notebook/script: 01_eda → 02_feature_engineering → 03_modeling → 04_evaluation.
- Lokasi dataset (data/AI_Engineer_dataset.parquet) & output (outputs/).
- Seed yang dipakai.

## 3. Struktur Repo
- Pohon folder + 1 baris penjelasan tiap bagian.

## 4. Temuan Data Kunci (singkat)
- `no_do` = unit loop (bukan trip_id).
- `average_time_sec` suspect leakage → baseline di-recompute.
- Skewness ekstrem + outlier; gap 47% loop; sparsity ringan.
  (Detail panjang ada di write-up; di sini cukup ringkas.)

## 5. Asumsi & Threshold yang Dipakai   ← WAJIB
- Threshold outlier traveling_time_sec = ... (alasan).
- Definisi gap: stop_sequence.diff() > 1; drop loop bila n<... atau gap>... .
- Definisi rush hour, smoothing m, dsb.
- Cara handle MAPE saat aktual ≈ 0.
  (Reviewer membaca bagian ini untuk menilai judgment — jangan dikosongkan.)

## 6. Feature Engineering
- Daftar fitur + 1 baris alasan tiap fitur (≥5).
- Kolom output wajib: is_gap_suspected, deviation_ratio.

## 7. Perbandingan Model   ← WAJIB, INI YANG DINILAI
- Jelaskan kandidat: Baseline, XGBoost/LightGBM, LSTM/GRU, [opsional].
- Setup fair comparison: split sama, metrik sama, (subset/full).
- TABEL hasil (contoh format):

  | Model            | MAE (s) | RMSE (s) | MAPE Segment | Loop MAE (s) | Catatan |
  |------------------|--------:|---------:|-------------:|-------------:|---------|
  | Baseline (mean)  |         |          |              |              | lantai  |
  | XGBoost          |         |          |              |              |         |
  | LightGBM         |         |          |              |              |         |
  | LSTM/GRU         |         |          |              |              |         |
  | Ensemble (stack) |         |          |              |              |         |

- **Model terpilih + ALASAN**: kenapa model ini (angka Loop MAE + kompleksitas/maintainability).
- **Apakah ensemble menang?** Jawab jujur dengan bukti. Kalau tidak menang signifikan,
  jelaskan kenapa memilih model tunggal (lihat METHODOLOGY §5c).

## 8. Hasil & Metrik Final
- Metrik model terpilih di test set (ruang asli/detik): MAE, RMSE, MAPE Segment, Loop MAE.
- Lift vs baseline (persen perbaikan).
- (Opsional) plot prediksi vs aktual, distribusi error.

## 9. Keterbatasan & Langkah Lanjut
- Apa yang belum sempat, asumsi yang perlu divalidasi (mis. konfirmasi leakage),
  data tambahan yang akan meningkatkan akurasi (Task 2).

## 10. Deliverables
- Link/lokasi: notebook, trained model, training_ready dataframe, write-up (PDF/MD).
```

---

## Pengingat untuk Claude Code

- Tabel Perbandingan Model **harus** terisi angka nyata dari `04_evaluation.ipynb` — jangan
  placeholder saat submit.
- Bagian "Asumsi & Threshold" mengambil isi final dari `DECISIONS.md`.
- README ringkas; pembahasan mendalam ada di `WRITEUP.md` (maks 2 halaman). Hindari duplikasi
  panjang antara keduanya — README = "cara pakai + ringkas hasil", write-up = "reasoning".
