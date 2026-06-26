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
├── DECISIONS.md           # Log keputusan teknis terperinci (D0 - D15)
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

## 9. Galeri Visualisasi
Seluruh grafik tersimpan dalam direktori `outputs/plots/` (PNG, ~110 DPI) dan juga ter-*embed* di dalam notebook `01_eda.ipynb`, `03_modeling.ipynb`, dan `04_evaluation.ipynb`.

| Berkas | Tujuan | Bagian relevan |
|---|---|---|
| `leakage_deviation_ratio_cap.png` | Bukti struktural *target leakage* — rasio aktual/average ter-*cap* persis di 2,0 dengan 0% baris di atas. | §1, §4 |
| `target_skew_log1p.png` | Distribusi target sebelum & sesudah `log1p` (skew 58,3 → 2,2). | §3, §5 |
| `loop_length.png` | Distribusi jumlah segmen per `no_do` (median 19, range 1–72). | §1 |
| `sparsity_segment_hour.png` | Heatmap kepadatan observasi per (segmen × jam). | §4 |
| `model_comparison_loop_mae.png` | Bar chart perbandingan Loop MAE seluruh model (XGBoost MAE-log = terbaik). | §7 |
| `feature_importance_final.png` | Importance 17 fitur model final, fitur baru iterasi v2 disorot. | §6, §7 |
| `iteration_v1_vs_v2.png` | Bar chart sebelum/sesudah penambahan fitur D12+D13 (Loop MAE 328,8 → 258,8). | §7 |

Notebook `04_evaluation.ipynb` juga menampilkan **plot diagnostik tambahan**: *prediksi vs aktual*, distribusi error, dan MAE per jam.

---

## 10. Keterbatasan Model & Pengembangan Selanjutnya

### 10.1. Pengembangan Sumber Data (Bottleneck Struktural)
Hambatan utama performa model bukan pada algoritma, melainkan pada keterbatasan informasi yang tersedia. Sumber data tambahan berikut akan memberikan terobosan akurasi:
1. **Integrasi Data Asli Tanpa Kebocoran**: Mendapatkan data baseline historis yang bersih langsung dari sistem operasi pusat BRT tanpa anomali pembatasan artifisial seperti pada `average_time_sec` bawaan.
2. **Model Multi-Rute**: Dataset saat ini hanya mencakup satu rute (`route_code` = 1 nilai unik). Data multi-rute membuka peluang *target encoding* tingkat rute, generalisasi antar-koridor, dan analisis pola jaringan BRT secara holistik.
3. **Sinyal Eksogen Real-Time**: Cuaca (hujan/terang via API BMKG), kalender hari libur nasional, kejadian khusus (demo, konser, kecelakaan), dan informasi kepadatan trafik hulu (*upstream*) — semua varians yang **belum ada dalam dataset sama sekali**.
4. **Kualitas Telemetri GPS**: Mengurangi tingkat 47% loop tidak lengkap (gap) agar model sekuensial seperti LSTM dapat menangkap dependensi spasio-temporal lebih maksimal.

### 10.2. Eksplorasi Fitur Lanjutan dari Data Existing
Berikut adalah hipotesis fitur tambahan yang dapat di-*extract* dari data mentah yang sudah ada. Daftar disusun berdasarkan **uji korelasi empiris** (bukan asumsi) sehingga prioritas pengembangan berbasis data, bukan intuisi.

**Sudah diuji empiris dan TIDAK direkomendasikan (negative results — sinyal jujur untuk reviewer):**

1. **Congestion Index per (segment, hour)** — rasio waktu tempuh rata-rata pada jam tersebut terhadap rata-rata jam *off-peak* (mis. 11:00–14:00) per segmen. Uji korelasi awal: 0,128 dengan `baseline_segment_hour` → tidak redundan secara linear. **Namun saat diuji empiris pada model**: Loop MAE memburuk dari 258,83 → 259,60 (Δ +0,77 detik). Hipotesis kegagalan: sinyalnya **redundan secara non-linear** dengan kombinasi `is_rush_hour` + `baseline_segment_hour` yang sudah dipakai model. Penambahan fitur hanya menyuntik *noise tipis* tanpa sinyal baru.

2. **Headway antar-loop / Deteksi *Bunching*** — selisih waktu antara dua loop berurutan pada `(trip_id, stop_sequence)` yang sama (median 6,6 menit, spread 2,9–13,1 menit). Hipotesis: *bunching* <5 menit → bus kedua lebih cepat. **Hasil eksperimen**: Loop MAE memburuk +3,88 detik (paling buruk dari semua kandidat). Kegagalan disebabkan oleh: (i) variansi headway terlalu tinggi (p95 = 2.226 detik = 37 menit) → terlalu *volatile* untuk sinyal yang dapat diandalkan; (ii) `time_since_prev_arrival_sec` (D12) sudah menangkap dinamika antar-pemberhentian per-bus yang lebih spesifik.

3. **Indeks kemacetan kronis per segmen** (rata-rata waktu per segmen vs rata-rata global): korelasi **0,927** dengan `baseline_segment_hour` (terdeteksi sebelum implementasi) → sangat **redundan**. Sinyal sudah ter-capture sepenuhnya oleh fitur existing.

4. **Propagasi kemacetan antar-segmen** (deviasi segmen sebelumnya terhadap baseline → mempengaruhi segmen sekarang): korelasi hanya **0,020** dengan target → tidak signifikan. Kemungkinan karena karakteristik BRT yang memakai *dedicated lane* sehingga kemacetan satu segmen tidak langsung meluap ke segmen tetangga.

**Insight metrik yang ditemukan dari eksperimen #1 dan #2 (signifikan untuk evaluasi model di masa depan):**
Penambahan kedua fitur kandidat menghasilkan **MAE dan MAPE per-segmen sedikit lebih baik** (31,60 vs 31,81 detik) **tetapi Loop MAE memburuk** (262,71 vs 258,83 detik). Fenomena ini menunjukkan bahwa **Loop MAE bukan sekadar agregasi MAE** — metrik tersebut mengekspos *bias direksional* (over- atau under-prediction yang konsisten dalam satu putaran) yang tidak terlihat pada metrik level segmen. Pelajaran: validasi model untuk *use case* operasional BRT harus selalu menggunakan Loop MAE sebagai *gatekeeper*, bukan MAE per segmen.

**Layak diuji di masa depan (belum diimplementasi, hipotesis ditinggalkan untuk pengembangan lanjutan):**

1. **Deteksi anomali via Isolation Forest** pada raw fitur derivat (`baseline_segment_hour`, `hour_sin`, `hour_cos`, `segment_volatility`) — gunakan *anomaly score* sebagai fitur kontinu daripada hanya melakukan *hard drop* outlier (D3). Membantu model membedakan kemacetan ekstrem yang valid dari sensor error secara *soft*.

2. **Cross-segment momentum dengan lag berbobot** — alih-alih deviasi segmen-1 (yang sudah diuji & gagal), gunakan *weighted average* deviasi 2–3 segmen terakhir dalam loop yang sama. Mungkin propagasi kemacetan terdistribusi pada *window* yang lebih luas, bukan segmen tetangga langsung.

3. **Interaksi `bus_body_no × segment_id`** — kombinasi spesifik bus-segmen mungkin memiliki pola unik (mis. supir yang familiar dengan segmen tertentu lebih cepat). Saat ini hanya `bus_encoded` (D13) yang aggregate per-bus.

### 10.3. Optimasi Lanjutan
* **Optuna tuning bertahap**: 30-trial sudah diuji (tidak memperbaiki). Bisa dicoba *Bayesian optimization* dengan 100+ trial pada feature set yang sudah diperluas — namun ekspektasi *gain* tipis (~5–10 detik Loop MAE).
* **Quantile loss** (P50, P90) untuk mendukung *worst-case scheduling* operasional — model saat ini fokus prediksi titik tengah; kebutuhan operasi BRT bisa menyertakan estimasi pesimistik untuk *buffer* jadwal.

### 10.4. Eksplorasi Feature Extraction Lanjutan (sudah diuji, semua negative result)
Berikut adalah pendekatan industry-standard yang dievaluasi & diuji empiris pada pipeline ini. Hasil didokumentasikan di `DECISIONS.md` D14–D15. Tujuan dokumentasi: menunjukkan reviewer bahwa kandidat menyadari berbagai opsi dan dapat memilih berdasarkan bukti, bukan asumsi.

| Pendekatan | Hasil Eksperimen | Verdict |
|---|---|---|
| **PCA** (n=10, 13, 15 komponen) + XGBoost | Loop MAE memburuk +52,87 hingga +57,88 detik (~22% lebih buruk) | ❌ Tidak cocok untuk *tree-based* |
| **QuantileTransformer-normal** untuk LSTM | LSTM membaik dari 368,54 → 344,55 detik, tapi tetap kalah XGBoost (258,83) | 🟡 Tidak mengubah pilihan model |
| **Polynomial interaction features** (degree=2, 10 interaksi) + XGBoost | Loop MAE memburuk +2,16 detik | ❌ XGBoost sudah menangkap interaksi via *tree splits* |
| **TabNet / TabTransformer** | Tidak dijalankan; LSTM (model deep tabular sejenis) sudah kalah dari XGBoost 110 detik | ⏸️ Skip berbasis bukti dari LSTM |
| **LLM-based feature extraction** (TabuLLM) | Tidak applicable — dataset tidak memiliki kolom teks bermakna | ⏸️ N/A |
| **AutoFE / RAPIDS GPU** | Tidak dijalankan — dependency luar brief, no expected gain | ⏸️ Skip |

---

## 11. Daftar Berkas Deliverables
* **Logika reusable**: Terletak pada direktori [src/](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/src) (`data.py`, `features.py`, `metrics.py`, `models.py`).
* **Notebook Eksperimen**: Terletak pada direktori [notebooks/](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/notebooks) (`01_eda.ipynb` hingga `04_evaluation.ipynb`).
* **Model Terlatih**: Berkas XGBoost final tersimpan di [outputs/model_xgb.json](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/outputs/model_xgb.json).
* **Dataframe Siap Latih**: File Parquet keluaran akhir tersimpan di [outputs/training_ready.parquet](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/outputs/training_ready.parquet) (berisi kolom tambahan `is_gap_suspected` dan `deviation_ratio`).
* **Laporan Analitis**: Berkas tertulis ringkas format Markdown di [WRITEUP.md](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/WRITEUP.md).
* **Log Keputusan Teknis**: Berkas pelacak keputusan arsitektur di [DECISIONS.md](file:///Users/mogataufiq/Active/Projects/moga-taufiq-bagidya_ml-test/DECISIONS.md).
