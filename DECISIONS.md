# DECISIONS.md — Log Keputusan Teknis (Living Document)

> Setiap keputusan besar dicatat di sini **saat** diambil, bukan belakangan. Ini sekaligus
> draft mentah write-up dan amunisi interview. Claude Code wajib menambah entri tiap ada
> keputusan; Moga wajib bisa menjawab kolom "Jawaban kalau ditanya".
>
> Format tiap entri:
> ```
> ### [ID] Judul keputusan
> - Konteks      : kenapa keputusan ini muncul
> - Keputusan    : apa yang dipilih
> - Alasan       : kenapa ini, bukan yang lain
> - Alternatif ditolak: ... + alasan ditolak
> - Trade-off    : konsekuensi dari pilihan ini
> - Jawaban kalau ditanya "<pertanyaan>": ...
> - Status       : [DRAFT / FINAL]
> ```

---

## Keputusan yang SUDAH terindikasi dari EDA (isi & finalkan saat kerja)

### [D1] Unit "satu putaran" = `no_do`, bukan `trip_id`
- Konteks: brief menyarankan `no_do` bisa diabaikan; perlu kunci loop untuk LSTM & Loop MAE.
- Keputusan: gunakan `no_do` sebagai unit loop (sequence key) dan unit Loop MAE.
- Alasan: `trip_id` hanya 13 nilai (varian rute); `no_do` = 17.857 loop, median 19 segmen,
  `stop_sequence` urut rapi di dalamnya.
- Alternatif ditolak: `trip_id` sebagai loop → mustahil (27rb baris/trip_id).
- Trade-off: melawan saran eksplisit brief — siap jelaskan dengan bukti angka.
- Jawaban kalau ditanya "kenapa tidak pakai trip_id?": [isi ringkas dari DATA_FINDINGS #1]
- Status: DRAFT

### [D2] `average_time_sec` diperlakukan sebagai SUSPECT LEAKAGE
- Konteks: `deviation_ratio` ter-cap [0,2], 0% > 2.0; korelasi traveling-average 0.84.
- Keputusan: tidak pakai mentah; recompute baseline leakage-free (expanding mean per
  segment-hour, shift); uji leakage via ablation.
- Alasan: cap 2.0 mustahil untuk baseline historis nyata (macet → rasio >2).
- Alternatif ditolak: pakai langsung sebagai fitur → risiko metrik palsu.
- Trade-off: kerja ekstra recompute; tapi metrik jadi jujur.
- Jawaban kalau ditanya "bagaimana kamu tahu itu leakage?": [isi dari DATA_FINDINGS #2]
- Status: DRAFT

### [D3] Threshold outlier `traveling_time_sec`
- Konteks: max 958.259 dtk (~11 hari) jelas error sensor; skew 58.
- Keputusan: [tetapkan: capping p99 / buang > X dtk] → **isi nilai & alasan**
- Alasan: [isi]
- Trade-off: buang data vs simpan noise.
- Jawaban kalau ditanya "kenapa threshold-nya segitu?": [isi]
- Status: DRAFT

### [D4] Definisi gap / `is_gap_suspected`
- Konteks: 47,4% loop punya gap; perlu definisi operasional.
- Keputusan: `is_gap_suspected = True` bila `stop_sequence.diff() > 1` dalam `no_do`;
  drop loop bila [n<5 atau gap >2] → **isi & finalkan**
- Alasan: [isi]
- Trade-off: [isi]
- Jawaban kalau ditanya: [isi]
- Status: DRAFT

### [D5] Transformasi skewness
- Konteks: target sangat skewed.
- Keputusan: `log1p` pada target (balik via `expm1` saat evaluasi). Alternatif Yeo-Johnson.
- Alasan: `+1` aman untuk nilai kecil/0; MSE di ruang log = relative error (cocok skewed).
- Alternatif ditolak: Box-Cox (butuh strictly positive).
- Trade-off: prediksi di ruang log, harus ingat reverse transform.
- Jawaban kalau ditanya "kenapa log1p bukan log?": [isi]
- Status: DRAFT

### [D6] Strategi split = time-based
- Konteks: data time series 34 hari (1 Feb–7 Mar 2026).
- Keputusan: train = [periode awal], test = [periode akhir]; encoder/scaler fit di train saja.
- Alasan: cegah kebocoran masa depan ke train.
- Alternatif ditolak: random split → bocor temporal.
- Trade-off: [isi]
- Status: DRAFT

---

## Keputusan yang AKAN muncul saat modeling (kosong — isi nanti)

### [D7] Model final terpilih + alasan
- Kandidat dibanding: baseline, XGBoost/LightGBM, LSTM/GRU, [opsional RF/Ridge]
- Hasil (Loop MAE / MAE / RMSE / MAPE): [tabel]
- Keputusan: [model] karena [angka + kompleksitas]
- Apakah ensemble menang? [ya/tidak + bukti]
- Status: TODO

### [D8] Strategi ensemble (jika dipakai)
- Pilihan: weighted average vs stacking → [isi + alasan]
- Status: TODO

### [D9] Strategi sparsity yang diterapkan
- [hierarchical fallback / target encoding + smoothing + nilai m] → [isi]
- Status: TODO

### [D10] Penanganan MAPE saat aktual ≈ 0
- [filter segmen pendek / pakai sMAPE] → [isi]
- Status: TODO

---

> Tambah entri baru di bawah ini sesuai kebutuhan. Setiap entri DRAFT harus jadi FINAL
> sebelum submit, dan setiap "Jawaban kalau ditanya" harus benar-benar bisa dijawab Moga.
