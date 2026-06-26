# Laporan Analitis: Prediksi Waktu Tempuh Segmen Bus BRT
**Kandidat:** Moga Taufiq Bagidya  
**Peran:** AI/ML Engineer  

---

## 1. Pendahuluan & Temuan Kunci Sistemik
Laporan ini menyajikan solusi prediksi waktu tempuh bus BRT per segmen (antar-halte berurutan) dari 351.103 baris GPS tracking. Dua anomali struktural mendominasi seluruh keputusan arsitektural:

1. **Target Leakage pada `average_time_sec`**: rasio aktual/average ter-*capped* persis di $[0, 2]$ dengan 0% baris di atas 2,0 (87% terkonsentrasi di $[0,75, 1,25]$; lihat `outputs/plots/leakage_deviation_ratio_cap.png`). Uji *ablation* — memasukkan kolom ini memotong MAE 44,4% dan menyerap 85,1% importance — mengkonfirmasi sinyal target. Pola ini mustahil pada operasional nyata. Fitur **dihapus** dan diganti baseline bersih *expanding mean* per (segmen, jam) yang di-*shift*.
2. **Unit Putaran = `no_do`, bukan `trip_id`**: `trip_id` hanya 13 nilai (~27.000 baris/nilai) → varian rute. `no_do` memiliki 17.857 nilai, median 19 segmen berurutan → unit satu putaran fisik bus yang valid. Digunakan sebagai *sequence key* LSTM dan basis *Loop MAE*.

---

## 2. Strategi Perjalanan Tidak Lengkap (*Incomplete Trips*)
47,4% putaran memiliki celah (*gap*) `stop_sequence`. Strategi spesifik per konteks model:
* **Tabular (XGBoost/LightGBM)**: data ber-gap **dipertahankan + ditandai** via fitur biner `is_gap_suspected`. Tiap baris diproses independen — target segmen tetap valid meski segmen lain hilang. Membuang seluruh loop ber-gap akan mengurangi data latih 42% secara sia-sia.
* **Sekuensial (LSTM) & Loop MAE**: hanya **putaran lengkap** yang dipakai — gap merusak representasi spasio-temporal & akumulasi waktu loop.
* **Interpolasi**: **tidak diterapkan** pada target — menyuntikkan label artifisial (*noise*) yang menurunkan generalisasi. Fragmen 1 segmen dibuang dari analisis sekuensial.

---

## 3. Penanganan Skewness Ekstrem & Desain Loss
Skew mentah **58,33** dengan max anomali ~11 hari (sensor error/dwell terminal). Dua teknik:
1. **Outlier plausibilitas fisis**: drop `traveling_time_sec > 3600` detik. Patahan tajam p99=32 menit (wajar untuk macet Jakarta) → p99,5=6,6 jam (mustahil 1 segmen BRT). Batas 1 jam menyapu 0,85% data anomali tanpa memotong macet riil.
2. **Transformasi `log1p`** menekan skew dari 58,33 → 2,21 (lihat `outputs/plots/target_skew_log1p.png`). Evaluasi dikembalikan ke detik via `expm1`.

**Dampak pada loss & metrik**: di ruang log, MSE bertindak seperti *relative error* di ruang asli — error 30 detik pada segmen 60 detik dihukum lebih berat daripada pada 600 detik (sesuai konteks BRT). Saya menguji tiga objektif: L1 (MAE), Huber, L2 (MSE). **L1 (MAE-log) terpilih** karena Loop MAE terbaik dan lebih *robust* terhadap sisa ekor panjang. *Loop MAE* dipilih sebagai metrik bisnis utama karena merepresentasikan kesalahan satu putaran penuh — langsung relevan untuk *headway*/jadwal BRT.

---

## 4. Penanganan Kelangkaan Data (*Sparsity*)
Sparsity ringan (3,3% dari 1.029 kombinasi segmen-jam <30 observasi). Strategi tetap diterapkan untuk generalisasi:
1. **Hierarchical fallback**: (segmen, jam) → segmen → global.
2. **Bayesian smoothing**: $\mu_{smooth} = (n \cdot \bar{y}_{group} + m \cdot \bar{y}_{global})/(n + m)$, prior $m=20$, fit **hanya di train**. Kombinasi sampel sedikit otomatis tertarik ke rata-rata global.
3. **Regularisasi pohon**: `min_child_weight=5` (XGB) / `min_child_samples=30` (LGBM) — cegah daun lahir dari segelintir baris langka.

---

## 5. Justifikasi Fitur (Domain-Driven)
Total **17 fitur** untuk model tabular. Tujuh fitur dengan kontribusi terbesar:
1. **`rolling_mean_segment_5`**: rata-rata 5 observasi terakhir per segmen (di-*shift*, leakage-free). Menangkap kondisi kemacetan terkini → fitur terkuat di model final.
2. **`baseline_segment_hour`**: *expanding mean* per (segmen, jam) di-*shift* — pengganti bersih `average_time_sec` yang bocor.
3. **`time_since_prev_arrival_sec`** *(baru, D12)*: selisih *departure* sekarang − *arrival* segmen terakhir bus yang sama. Proksi *dwell* terminal & headway antar-loop. Korelasi linear dengan target = -0,001, namun sinyal **non-linear** kuat: uji permutasi membuat Loop MAE meledak 259 → 424 detik. Menyumbang mayoritas perbaikan ~70 detik.
4. **`bus_encoded`** *(baru, D13)*: *target encoding* rata-rata waktu per `bus_body_no` (*Bayesian smoothing* $m=20$, fit di train saja). Menangkap karakteristik bus (umur, supir, mesin).
5. **`hour_sin`/`hour_cos`** + **`is_rush_hour`** (06–09 & 16–19) + **`is_weekend`**: fitur waktu siklis & komuter Jakarta, dari *departure* saja (anti-leakage).
6. **`segment_volatility`**: std historis waktu per segmen (proksi kerentanan macet kronis).
7. **`trip_progress`**: posisi relatif bus dalam putaran (`stop_sequence / max`) — menangkap akumulasi keterlambatan di akhir loop.

---

## 5.5. Proses Iterasi Peningkatan (v1 15 fitur → v2 17 fitur)
Model awal dengan **15 fitur** menghasilkan Loop MAE **328,8 detik** (lift -35,6%). Mencari ruang perbaikan, saya menganalisis komposisi *outlier* yang dibuang (D3): **99,2% berada di `stop_sequence == 1`**, **99,9% di posisi terminal akhir loop**, dengan puncak jam 20–22. Pola ini **bukan *random sensor error*** (dugaan awal), melainkan **dwell terminal** — bus parkir menunggu jadwal. Sinyal operasional hilang ketika baris dwell di-drop.

**Tindakan**: (1) Fitur **`time_since_prev_arrival_sec` (D12)** — proksi dwell, selisih *departure* sekarang dengan *arrival* segmen terakhir bus yang sama (leakage-free). (2) Fitur **`bus_encoded` (D13)** — target encoding `bus_body_no` (Bayesian smoothing). (3) **Tuning Optuna 30-trial** diuji namun tidak memperbaiki (val LoopMAE 321,9 vs default 258,8) → default dipertahankan, hindari overtuning pada val kecil.

| Iterasi | Fitur | MAE | RMSE | MAPE | **Loop MAE** | Lift |
|---|--:|--:|--:|--:|--:|--:|
| v1 (awal) | 15 | 38,1 | 113,9 | 30,4% | 328,8 | -35,6% |
| **v2 (final)** | **17** | **31,8** | **94,0** | **23,6%** | **258,8** | **-49,3%** |

*(Visual: `outputs/plots/iteration_v1_vs_v2.png` & `feature_importance_final.png`).*

**Verifikasi anti-leakage** untuk `time_since_prev_arrival_sec`: (i) median AE tetap 12,57s (bukan ~0); (ii) korelasi linear dengan target = -0,001 (sinyal murni non-linear); (iii) uji permutasi mengacak fitur di test set membuat Loop MAE meledak **259 → 424 detik (Δ +165s)** — bukti fitur asli yang bertanggung jawab atas perbaikan, bukan artifact perhitungan.

---

## 6. Pemilihan Model & Kajian Ensemble
Split temporal *loop-aware*: train Feb 1–21 (258.590 baris) vs test Feb 22–28 (89.519 baris). Evaluasi fair pada set uji umum 2.325 putaran lengkap (ruang detik; visual: `outputs/plots/model_comparison_loop_mae.png`):

| Model | MAE (s) | RMSE (s) | MAPE Seg | **Loop MAE (s)** | Karakteristik Inferensi |
|---|---:|---:|---:|---:|---|
| *Baseline (seg, hour mean)* | 50.8 | 125.3 | 46.1% | 510.3 | Instan, tanpa pembelajaran |
| **XGBoost (MAE-log) — Terpilih** | **31.8** | **94.0** | 23.6% | **258.8** | **Cepat (1,47 ms/loop), 1 berkas `.json`** |
| *XGBoost (MSE-log)* | 32.4 | 94.3 | **23.6%** | 266.9 | Optimasi L2 di ruang log |
| *LightGBM (MSE-log)* | 32.3 | **93.1** | 23.6% | 267.4 | Implementasi cepat & efisien memori |
| *XGBoost (Huber-log)* | 32.3 | 96.1 | 23.7% | 267.8 | Gradien mulus untuk outlier |
| *LSTM (Huber-log)* | 41.0 | 125.1 | 30.5% | 368.5 | Lambat, rentang urutan pendek |
| *Ensemble (Weighted Convex)* | — | — | — | tidak menang | XGB+LSTM kalah dari XGB tunggal (diuji) |

### Ensemble & Strategi Penggabungan
Saya menguji dua strategi penggabungan output XGBoost + LSTM: **weighted convex** (bobot $w=0,65$ untuk XGBoost, di-fit di validasi) dan **stacking** (Ridge meta-learner). Kedua **tidak menang signifikan**: convex hanya tipis di atas base-nya & masih kalah dari XGBoost tunggal full-train (258,8 detik); stacking *overfit* pada val kecil (965 putaran). Alasan kegagalan: (i) error base berkorelasi tinggi karena keduanya pakai `baseline_segment_hour`, (ii) sekuens pendek (median 19) → XGBoost tabular sudah mengekstrak sinyal temporal maksimal via *rolling*/*expanding mean*. **Kapan ensemble tidak menguntungkan**: ketika error base berkorelasi, satu model mendominasi, atau panjang sekuens tidak cukup memberi sinyal tambahan ke LSTM — ketiganya berlaku di sini.

### Rekomendasi Produksi: XGBoost Tunggal (MAE-log)
1. **Akurasi**: Loop MAE 258,8 detik (**-49,3%** vs baseline).
2. **Latensi**: 1,47 ms/loop — 50–100× lebih cepat dari LSTM, siap real-time.
3. **Maintainability**: 1 berkas `.json`, tanpa runtime PyTorch, retrain hitungan detik.
