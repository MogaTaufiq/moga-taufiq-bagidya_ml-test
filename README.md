# Prediksi Waktu Tempuh Segmen Bus BRT
**Kandidat:** Moga Taufiq Bagidya  
**Peran:** AI/ML Engineer  

Proyek ini bertujuan untuk memprediksi waktu tempuh (*travel time*) bus kota per segmen koridor BRT (antar-halte berurutan) menggunakan data tracking GPS historis. Melalui pembersihan data yang ketat, mitigasi kebocoran data (*target leakage*), dan rekayasa fitur (*feature engineering*) — termasuk dua fitur tambahan ber-impak tinggi (*dwell terminal* dan *bus target encoding*) — model terpilih **XGBoost (MAE-log)** berhasil mencapai nilai **Loop MAE sebesar 258,8 detik** pada data uji (7 hari terakhir), yang menunjukkan peningkatan akurasi sebesar **-49,3% di bawah rata-rata historis (*baseline*)** dengan latensi inferensi yang sangat rendah, yaitu **1,47 ms per putaran**.

---

## 1. Ringkasan Eksekutif
Tugas ini diselesaikan dengan memformulasikan prediksi waktu tempuh segmen bus sebagai masalah regresi tabular tingkat segmen. Berdasarkan analisis eksplorasi data, ditemukan bahwa fitur bawaan `average_time_sec` merupakan *target leakage* karena rasio aktual terhadap rata-rata ter-cap secara tidak wajar pada rentang $[0, 2]$. Di samping itu, unit putaran (*loop*) bus yang sahih diidentifikasi sebagai `no_do`, bukan `trip_id`. 

Model final berbasis **XGBoost** yang menggunakan target transformasi logaritmik berhasil mengalahkan model dasar rata-rata historis sebesar 49,3% pada metrik bisnis utama (*Loop MAE*). Setelah ditambahkan dua fitur baru hasil insight operasional — `time_since_prev_arrival_sec` (proksi *dwell* terminal) dan `bus_encoded` (target encoding per-bus) — Loop MAE turun dari 328,8 → 258,8 detik (penurunan ~21,3% di atas pipeline awal). Eksperimen terhadap model sekuensial (LSTM) dan metode gabungan (*ensemble*) tidak menunjukkan peningkatan performa yang signifikan, sehingga model tunggal XGBoost dipilih untuk diimplementasikan di lingkungan produksi karena keunggulannya dalam kesederhanaan, latensi, dan pemeliharaan.

---

## 2. Reproduksibilitas (Cara Menjalankan)

### Persyaratan Lingkungan
* **Python**: Versi 3.10+ (dikembangkan menggunakan Python 3.12.12)
* **Ketergantungan Sistem (khusus macOS/Apple Silicon)**: Diperlukan runtime OpenMP untuk eksekusi XGBoost dan LightGBM.
  ```bash
  brew install libomp
  ```

### Panduan Instalasi
1. Buat dan aktifkan lingkungan virtual (opsional namun disarankan):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Untuk Linux/macOS
   # .venv\Scripts\activate   # Untuk Windows
   ```
2. Instal pustaka yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

### Urutan Eksekusi Notebook
Seluruh proses eksperimen dan pemodelan disusun secara terstruktur di dalam direktori `notebooks/`. Setiap notebook dilengkapi dengan plot visual yang telah ter-render. Anda dapat menjalankannya kembali secara berurutan untuk mereproduksi hasil:
```
notebooks/01_eda.ipynb  →  02_feature_engineering.ipynb  →  03_modeling.ipynb  →  04_evaluation.ipynb
```
* **Nilai Seed**: Seluruh pustaka (numpy, scikit-learn, split, XGBoost, PyTorch) dikunci pada `seed = 42` untuk menjamin reproduksibilitas hasil.
* **Format Notebook**: Kode logika utama ditulis di dalam direktori `src/` agar mudah diuji dan di-*import*. Setiap notebook memiliki salinan file `.py` yang dikelola menggunakan `jupytext` untuk memudahkan pelacakan perubahan (*diff*).

---

## 3. Struktur Repositori

```
├── DECISIONS.md           # Log keputusan teknis terperinci (D0 - D13)
├── requirements.txt       # Daftar ketergantungan library Python
├── README.md              # Berkas ini (panduan utama proyek)
├── WRITEUP.md             # Laporan analitis ringkas (maksimal 2 halaman)
├── data/
│   └── AI_Engineer_dataset.parquet   # Dataset tracking GPS BRT mentah
├── notebooks/
│   ├── 01_eda.ipynb                  # Eksplorasi data awal & deteksi leakage
│   ├── 02_feature_engineering.ipynb   # Pembersihan & pembuatan data siap latih
│   ├── 03_modeling.ipynb             # Pelatihan & perbandingan model
│   └── 04_evaluation.ipynb           # Evaluasi performa akhir & analisis error
├── src/
│   ├── __init__.py
│   ├── data.py            # Modul pemuatan, pembersihan, dan pemisahan data
│   ├── features.py        # Modul rekayasa fitur (leakage-free)
│   ├── metrics.py         # Modul metrik performa (MAE, RMSE, MAPE, Loop MAE)
│   └── models.py          # Modul latih model, evaluasi, dan model LSTM
└── outputs/               # Hasil keluaran model, plot, dan dataset siap latih
```

---

## 4. Temuan Data Kunci
* **Unit Putaran Fisik Bus**: Kolom `no_do` bertindak sebagai representasi satu putaran bus (*loop*) yang valid (terdapat 17.857 putaran unik dengan median 19 segmen dan urutan halte `stop_sequence` yang rapi). Sebaliknya, `trip_id` hanya menandakan varian rute (13 nilai unik).
* **Kebocoran Data Target**: Kolom bawaan `average_time_sec` terbukti memiliki kebocoran informasi masa depan (*target leakage*) karena distribusinya dibatasi secara artifisial. Kolom ini dihapus sepenuhnya dari fitur model.
* **Kausalitas Waktu**: Nilai target waktu tempuh segmen diperoleh dari selisih `arrival_time` dengan `from_arrival_time_str`. Oleh karena itu, seluruh fitur waktu model hanya dihitung dari waktu keberangkatan (*departure time*) untuk mencegah bias prediksi.
* **Karakteristik Data**: Ditemukan tingkat kemiringan (*skewness*) ekstrem (58,33), keberadaan celah (*gap*) pada 47,4% putaran bus, serta tingkat kelangkaan (*sparsity*) yang sangat ringan (3,3% dari kombinasi segmen-jam memiliki sampel kurang dari 30).

---

## 5. Asumsi & Batasan Threshold yang Digunakan

> [!IMPORTANT]
> Seluruh batasan (*threshold*) di bawah ini ditetapkan berdasarkan logika domain operasional transportasi BRT untuk memastikan model dapat diterapkan pada kondisi nyata di lapangan.

| Item | Nilai Batasan | Alasan Operasional & Domain |
|---|---|---|
| **Outlier Waktu Tempuh** | Drop jika $> 3.600$ detik (1 jam) | Nilai persentil ke-99 berada pada 32 menit (plausibel untuk macet total Jakarta), namun melonjak hingga 6,6 jam pada persentil ke-99,5. Angka di atas 1 jam diasumsikan sebagai bus mogok/parkir atau kesalahan sensor GPS. |
| **Identifikasi Celah (*Gap*)** | `stop_sequence.diff() > 1` dalam `no_do` | Menandai bus yang melewatkan halte akibat kegagalan tangkapan GPS atau sensor. Kolom ditandai sebagai `is_gap_suspected = True`. |
| **Kebijakan Drop Celah** | Spesifik berdasarkan model | Pada model tabular (XGBoost), baris ber-gap tetap dipertahankan agar tidak membuang 42% data latih secara sia-sia. Pada LSTM dan evaluasi *Loop MAE*, putaran ber-gap dibuang untuk menjaga konsistensi urutan sekuensial. |
| **Pemisahan Data (*Split*)** | *Time-based loop-aware*, batas cut: `2026-02-22` | Data tanggal 1–21 Februari digunakan sebagai data latih, dan 22–28 Februari sebagai data uji (7 hari terakhir). Pemisahan berbasis *loop-aware* menjamin tidak ada putaran bus (*no_do*) yang terpecah di antara dua set data. |
| **Transformasi Target** | `log1p` pada target (dikembalikan via `expm1`) | Mengatasi *skewness* ekstrem dan memfokuskan model pada optimasi kesalahan relatif (*relative error*) di ruang asli detik. |
| **Mitigasi Sparsity** | *Bayesian smoothing* ($m = 20$) + *hierarchical fallback* | Mencegah *overfitting* pada kombinasi segmen-jam yang memiliki data observasi sangat minim dengan menariknya ke arah rata-rata global secara aman. |
| **MAPE Segmen Pendek** | Abaikan segmen dengan aktual $< 5$ detik | Menghindari pembagian dengan nilai mendekati nol yang dapat menyebabkan nilai metrik MAPE meledak secara artifisial. |

---

## 6. Rekayasa Fitur (*Feature Engineering*)
Model akhir dilatih menggunakan **17 fitur** terpilih. Sembilan fitur utama yang memberikan kontribusi terbesar meliputi:
1. **`rolling_mean_segment_5`**: Rata-rata bergerak dari 5 observasi terakhir pada segmen yang sama (di-*shift* 1 langkah agar *leakage-free*).
2. **`baseline_segment_hour`**: Rata-rata kumulatif waktu tempuh per segmen-jam (*expanding mean*, di-*shift*). Berfungsi sebagai pengganti fitur `average_time_sec` yang bersih dan bebas kebocoran data.
3. **`time_since_prev_arrival_sec`** *(baru, D12)*: Selisih waktu antara *departure* segmen sekarang dengan *arrival* segmen terakhir bus yang sama. Proksi *dwell* terminal & headway antar-loop. Korelasi linear dengan target hampir nol (-0,001), namun sinyal non-linear besar — uji permutasi membuat Loop MAE meledak dari 259 → 424 detik.
4. **`bus_encoded`** *(baru, D13)*: *Target encoding* rata-rata waktu tempuh per `bus_body_no` dengan *Bayesian smoothing* ($m=20$). Menangkap karakteristik per-bus (umur kendaraan, gaya supir, kondisi mesin) yang konsisten lintas segmen.
5. **`hour_sin` & `hour_cos`**: Representasi siklis waktu jam. Menghubungkan kedekatan fisis antara jam 23:59 dan jam 00:01 secara kontinu.
6. **`is_rush_hour`**: Penanda biner untuk jam sibuk pagi (06.00–09.00) dan sore (16.00–19.00) khas wilayah DKI Jakarta.
7. **`is_weekend`**: Penanda biner untuk membedakan hari kerja dengan hari libur akhir pekan.
8. **`segment_volatility`**: Standar deviasi historis waktu tempuh per segmen untuk menangkap tingkat kerentanan kemacetan segmen tersebut.
9. **`trip_progress`**: Posisi relatif bus dalam putaran (`stop_sequence / max_stop_sequence`).

*Kolom Output Wajib*: Sesuai dengan spesifikasi tugas, berkas luaran akhir (`outputs/training_ready.parquet`) wajib menyertakan kolom **`is_gap_suspected`** dan **`deviation_ratio`** (rasio aktual terhadap rata-rata historis bawaan). Saya juga menyediakan kolom `deviation_ratio_clean` (rasio terhadap rata-rata bersih buatan sendiri).

---

## 7. Analisis Perbandingan Model

> [!IMPORTANT]
> Seluruh model diuji pada set data uji umum yang sama (2.325 putaran lengkap, waktu dalam detik) untuk menjamin perbandingan yang adil (*fair comparison*).

| Model | MAE (s) | RMSE (s) | MAPE Segment | **Loop MAE (s)** | Catatan Metodologis & Operasional |
|---|---:|---:|---:|---:|---|
| *Baseline (seg, hour mean)* | 50,8 | 125,3 | 46,1% | 510,3 | Acuan performa dasar (*floor*) |
| **XGBoost (MAE-log)** ★ | **31,8** | **94,0** | 23,6% | **258,8** | **Model Terpilih**: Performa *Loop MAE* terbaik, ringan |
| *XGBoost (MSE-log)* | 32,4 | 94,3 | **23,6%** | 266,9 | Optimasi L2 di ruang log |
| *LightGBM (MSE-log)* | 32,3 | 93,1 | 23,6% | 267,4 | Implementasi LightGBM cepat |
| *XGBoost (Huber-log)* | 32,3 | 96,1 | 23,7% | 267,8 | Penanganan gradien mulus untuk outlier |
| *LSTM (Huber-log)* | 41,0 | 125,1 | 30,5% | 368,5 | Pola urutan kurang sensitif pada loop pendek |
| *Ensemble (Weighted Convex)* | — | — | — | tidak menang | XGB+LSTM kalah dari XGB tunggal (diuji sebelumnya) |

### Alasan Pemilihan Model Final (XGBoost MAE-log)
Model **XGBoost (MAE-log)** dipilih sebagai model terbaik untuk diimplementasikan di lingkungan produksi karena keunggulan pada aspek berikut:
1. **Akurasi Bisnis Terbaik**: Menghasilkan metrik *Loop MAE* terkecil (258,8 detik) yang berarti kesalahan prediksi waktu satu putaran penuh bus rata-rata hanya sekitar 4,3 menit (peningkatan akurasi sebesar **-49,3%** di bawah model dasar rata-rata historis).
2. **Latensi Inferensi Sangat Rendah**: Model tabular XGBoost memiliki waktu prediksi rata-rata hanya **1,47 ms per putaran** (~19 segmen). Ini sangat ideal untuk operasional real-time BRT jika dibandingkan dengan LSTM yang membutuhkan waktu 50–100× lebih lama dan infrastruktur komputasi (GPU/PyTorch) yang mahal.
3. **Kemudahan Pemeliharaan**: Model hanya disimpan dalam satu berkas berukuran kecil (`model_xgb.json`) dan proses latih ulang (*retraining*) dapat diselesaikan hanya dalam hitungan detik.

### Apakah Model Ensemble Memberikan Kemenangan?
**Tidak.** Eksperimen membuktikan model gabungan (*ensemble*) tidak memberikan perbaikan performa yang signifikan dibandingkan model tunggal XGBoost:
* Pendekatan *Weighted Convex* (kombinasi XGBoost dan LSTM) hanya memberikan peningkatan minor di atas model dasarnya, dan kinerjanya masih **kalah** dibandingkan model XGBoost tunggal yang dilatih penuh pada seluruh data latih.
* Metode *stacking* berbasis Ridge regression justru menghasilkan performa yang lebih buruk karena mengalami *overfitting* pada set validasi yang kecil.
* Kegagalan ini disebabkan oleh tingginya korelasi kesalahan (*error*) antara LSTM dan XGBoost karena keduanya menggunakan basis data historis yang serupa. Selain itu, dengan panjang putaran bus yang relatif pendek (median 19 segmen), model tabular berbasis pohon sudah mampu mengekstrak informasi temporal secara maksimal melalui fitur rata-rata bergerak (*rolling mean*).

---

## 8. Hasil & Metrik Akhir
Model terpilih **XGBoost (MAE-log)** dievaluasi pada data uji (Feb 22–28) dan memperoleh hasil akhir sebagai berikut:
* **MAE**: 31,8 detik
* **RMSE**: 94,0 detik
* **MAPE Segment**: 23,6%
* **Loop MAE**: 258,8 detik (penurunan kesalahan sebesar **-49,3%** dari *baseline* awal 510,3 detik)

---

## 9. Keterbatasan Model & Pengembangan Selanjutnya
Beberapa keterbatasan model saat ini yang dapat dijadikan arah pengembangan selanjutnya untuk meningkatkan akurasi meliputi:
1. **Integrasi Data Asli Tanpa Kebocoran**: Mendapatkan data baseline historis yang bersih langsung dari sistem operasi pusat BRT tanpa adanya anomali pembatasan artifisial seperti pada fitur `average_time_sec` bawaan.
2. **Model Multi-Rute**: Dataset saat ini hanya mencakup satu rute (`route_code` hanya memiliki 1 nilai unik). Penggunaan data multi-rute akan membuka peluang penerapan teknik *target encoding* tingkat rute dan meningkatkan kemampuan generalisasi model pada rute yang berbeda.
3. **Penyertaan Sinyal Eksogen**: Memasukkan data faktor luar seperti kondisi cuaca (hujan/terang), kalender hari libur nasional, kejadian khusus, dan informasi kemacetan hulu (*upstream*) secara real-time.
4. **Peningkatan Kualitas GPS**: Mengurangi tingkat kegagalan sensor GPS (saat ini menyebabkan 47% loop tidak lengkap) agar model sekuensial seperti LSTM dapat menangkap informasi dependensi spasio-temporal secara lebih maksimal.

---

## 10. Daftar Berkas Deliverables
* **Logika reusable**: Terletak pada direktori [src/](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/src) (`data.py`, `features.py`, `metrics.py`, `models.py`).
* **Notebook Eksperimen**: Terletak pada direktori [notebooks/](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/notebooks) (`01_eda.ipynb` hingga `04_evaluation.ipynb`).
* **Model Terlatih**: Berkas XGBoost final tersimpan di [outputs/model_xgb.json](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/outputs/model_xgb.json).
* **Dataframe Siap Latih**: File Parquet keluaran akhir tersimpan di [outputs/training_ready.parquet](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/outputs/training_ready.parquet) (berisi kolom tambahan `is_gap_suspected` dan `deviation_ratio`).
* **Laporan Analitis**: Berkas tertulis ringkas format Markdown di [WRITEUP.md](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/WRITEUP.md).
* **Log Keputusan Teknis**: Berkas pelacak keputusan arsitektur di [DECISIONS.md](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/DECISIONS.md).
