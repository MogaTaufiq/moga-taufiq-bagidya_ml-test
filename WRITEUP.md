# Laporan Analitis: Prediksi Waktu Tempuh Segmen Bus BRT
**Kandidat:** Moga Taufiq Bagidya  
**Peran:** AI/ML Engineer  

---

## 1. Pendahuluan & Temuan Kunci Sistemik
Laporan ini menyajikan solusi *machine learning* untuk memprediksi waktu tempuh (*travel time*) bus kota per segmen koridor BRT (antar-halte berurutan). Berdasarkan analisis eksplorasi data (*Exploratory Data Analysis* - EDA) pada dataset `AI_Engineer_dataset.parquet` yang berukuran 351.103 baris, ditemukan dua anomali struktural penting yang mendominasi seluruh keputusan arsitektural model:

1. **Kebocoran Data Target (*Target Leakage*) pada `average_time_sec`**:  
   Ditemukan bahwa rasio antara target aktual (`traveling_time_sec`) dengan fitur `average_time_sec` dibatasi (*capped*) secara artifisial persis pada rentang $[0, 2]$. Tidak ada satu pun observasi (0%) yang memiliki rasio di atas 2,0, dan 87% data terkonsentrasi sangat rapat pada rentang $[0.75, 1.25]$. Pada uji *ablation*, memasukkan kolom ini sebagai fitur memotong MAE model hingga 44,4% dan menyerap 85,1% tingkat kepentingan fitur (*feature importance*). Pola ini mustahil terjadi pada operasional nyata (di mana kemacetan parah atau insiden dapat membuat waktu tempuh melebihi 2× rata-rata historis). Oleh karena itu, fitur `average_time_sec` diidentifikasi sebagai *target leakage* dan **dihapus sepenuhnya** dari fitur latih. Kami menghitung ulang *baseline* historis yang bersih (*expanding mean* per segmen-jam yang di-*shift*) sebagai pengganti yang aman.
2. **Definisi Unit Putaran (Loop) yang Tepat**:  
   Spesifikasi awal menyarankan penggunaan `trip_id` sebagai representasi putaran bus dan mengabaikan `no_do`. Namun, data menunjukkan `trip_id` hanya memiliki 13 nilai unik (dengan rata-rata ~27.000 baris per nilai), yang menjadikannya sebagai penanda varian rute utama. Sebaliknya, `no_do` memiliki 17.857 nilai unik dengan median 19 segmen berurutan dan urutan `stop_sequence` yang teratur. Dengan demikian, **`no_do` diidentifikasi sebagai unit satu putaran fisik bus (loop)** yang valid, dan digunakan sebagai unit *sequence key* pada model sekuensial serta basis evaluasi metrik bisnis utama (*Loop MAE*).

---

## 2. Strategi Perjalanan Tidak Lengkap (*Incomplete Trips*)
Sebanyak 47,4% dari total putaran (*no_do*) memiliki celah (*gap*) pada `stop_sequence` akibat kegagalan tangkapan GPS atau halte yang terlewat. Penanganan masalah ini diselesaikan secara spesifik berdasarkan kebutuhan model (*context-specific strategy*):
* **Model Tabular (XGBoost/LightGBM)**: Data perjalanan tidak lengkap **dipertahankan dan ditandai** menggunakan fitur biner `is_gap_suspected`. Karena model tabular memproses setiap baris secara independen, nilai target waktu tempuh pada segmen tertentu tetap valid dan akurat meskipun segmen lain dalam putaran yang sama hilang. Membuang seluruh data putaran bergap akan mengurangi data latih sebesar 42% secara sia-sia.
* **Model Sekuensial (LSTM) & Evaluasi Loop MAE**: Untuk melatih LSTM dan menghitung metrik *Loop MAE* secara adil, kami **hanya menggunakan putaran yang lengkap** (tanpa gap). Putaran dengan gap akan mengacaukan representasi spasio-temporal sekuensial dan menghasilkan penjumlahan akumulasi waktu tempuh yang tidak sahih.
* **Kebijakan Interpolasi**: Kami memilih untuk **tidak menginterpolasi target** pada data latih karena tindakan ini menyuntikkan label artifisial (*noise*) yang dapat menurunkan kualitas generalisasi model. Putaran sangat pendek (fragmen 1 segmen) dibuang dari analisis sekuensial karena tidak memiliki konteks historis.

---

## 3. Penanganan Skewness Ekstrem & Desain Metrik Evaluasi
Dataset target memiliki tingkat kemiringan (*skewness*) ekstrem sebesar 58,33 dengan nilai maksimum anomali fisis mencapai ~11 hari akibat kegagalan sensor. Dua teknik diterapkan untuk mengatasi masalah ini:
1. **Pembersihan Outlier Plausibilitas Fisis**: Membuang baris data dengan target `traveling_time_sec > 3600` detik (1 jam). Persentil ke-99 menunjukkan angka 32 menit (plausibel untuk kemacetan parah di Jakarta), namun melompat ke 6,6 jam pada persentil ke-99,5. Batas 1 jam menyapu bersih anomali sensor (0,85% data) dengan aman tanpa memotong fenomena kemacetan riil.
2. **Transformasi Logaritmik Target**: Menggunakan transformasi $log(x + 1)$ atau `log1p` yang menekan tingkat kemiringan (*skewness*) dari 58,33 menjadi 2,21. Selama evaluasi, seluruh prediksi dikembalikan ke satuan detik asli menggunakan fungsi eksponensial `expm1`.

**Dampak pada *Loss Function* & Metrik**:  
Di dalam ruang logaritmik, fungsi optimasi *Mean Squared Error* (MSE) bertindak seperti *relative error* di ruang asli. Kesalahan prediksi sebesar 30 detik pada segmen pendek (misal 60 detik) akan dihukum lebih berat dibandingkan kesalahan 30 detik pada segmen panjang (misal 600 detik). Ini sangat sesuai dengan karakteristik operasional BRT. Kami menguji tiga objektif optimasi di ruang log: L1 (*Mean Absolute Error* - MAE), Huber (*pseudo-huber*), dan L2 (MSE). Objektif **L1 (MAE-log) dipilih sebagai model final** karena menghasilkan performa terbaik pada metrik *Loop MAE* dan lebih tangguh (*robust*) terhadap sisa distribusi ekor panjang (*long-tail*).

---

## 4. Penanganan Kelangkaan Data (*Sparsity*)
Meskipun secara akumulatif kelangkaan data kombinasi segmen-jam tergolong ringan (hanya 3,3% dari 1.029 kombinasi segmen-jam memiliki kurang dari 30 observasi), kami menerapkan strategi berlapis untuk mencegah terjadinya *overfitting* pada kebisingan (*noise*):
1. **Hierarchical Fallback (Mundur Berjenjang)**: Jika kombinasi (segmen, jam) tertentu memiliki observasi yang sangat sedikit, estimasi model akan didukung oleh rata-rata segmen secara keseluruhan, dan jika masih minim, akan mundur ke rata-rata global sistem.
2. **Bayesian Smoothing (Penghalusan Bayesian)**: Kami menghitung rata-rata historis menggunakan teknik *Bayesian target encoding*:
   $$\mu_{smooth} = \frac{n \cdot \bar{y}_{group} + m \cdot \bar{y}_{global}}{n + m}$$
   di mana $n$ adalah jumlah observasi kelompok, dan bobot prior $m = 20$. Kombinasi dengan sampel kecil secara otomatis akan ditarik mendekati rata-rata global. Estimasi ini **hanya dihitung pada data latih** untuk mencegah kebocoran temporal (*temporal leakage*).
3. **Regularisasi Model Pohon**: Mengatur parameter `min_child_weight=5` pada XGBoost untuk melarang pembentukan daun (*leaf node*) baru dari observasi yang terlalu sedikit.

---

## 5. Justifikasi Fitur Baru (Domain-Driven)
Kami merancang 15 fitur untuk model tabular. Tujuh fitur utama yang memberikan kontribusi terbesar (di luar kolom mentah) dijabarkan sebagai berikut:
1. **`rolling_mean_segment_5`**: Rata-rata bergerak dari 5 observasi terakhir pada segmen yang sama (di-*shift* 1 langkah agar *leakage-free*). Fitur ini menangkap kondisi kemacetan terkini (*real-time traffic trend*) dan menjadi fitur terkuat pada model final dengan nilai tingkat kepentingan (*importance*) sekitar 0,50 (XGBoost MAE-log).
2. **`baseline_segment_hour`**: Rata-rata kumulatif waktu tempuh per segmen-jam (*expanding mean*, di-*shift*). Berfungsi sebagai pengganti fitur `average_time_sec` yang bersih dan bebas kebocoran data.
3. **`hour_sin` & `hour_cos`**: Representasi siklis waktu jam. Menghubungkan kedekatan fisis antara jam 23:59 dan jam 00:01 secara kontinu yang tidak dapat ditangkap oleh representasi numerik linear.
4. **`is_rush_hour`**: Penanda biner untuk jam sibuk pagi (06.00–09.00) dan sore (16.00–19.00) khas wilayah DKI Jakarta untuk menangkap variasi lalu lintas komuter.
5. **`is_weekend`**: Penanda biner untuk membedakan pola pergerakan lalu lintas hari kerja dengan hari libur akhir pekan.
6. **`segment_volatility`**: Standar deviasi historis waktu tempuh per segmen. Berfungsi sebagai proksi tingkat ketidakpastian (*traffic volatility*) atau kerentanan kemacetan kronis segmen tersebut.
7. **`trip_progress`**: Posisi relatif bus dalam putaran (`stop_sequence / max_stop_sequence`). Berguna untuk menangkap akumulasi keterlambatan bus di akhir perjalanan.

---

## 6. Justifikasi Pemilihan Model & Kajian Ensemble
Kami menguji performa model dengan membagi data secara temporal (*time-based split*) berbasis *loop-aware*: periode Feb 1–21 (258.590 baris) sebagai data latih dan Feb 22–28 (89.519 baris) sebagai data uji. Evaluasi dilakukan secara adil pada set data uji umum berisi 2.325 putaran lengkap (ruang asli detik):

| Model | MAE (s) | RMSE (s) | MAPE Seg | **Loop MAE (s)** | Karakteristik Inferensi |
|---|---:|---:|---:|---:|---|
| *Baseline (seg, hour mean)* | 50.8 | 125.3 | 46.1% | 510.3 | Instan, tanpa pembelajaran |
| **XGBoost (MAE-log) — Terpilih** | **38.1** | **113.9** | 30.4% | **328.8** | **Cepat (0,85 ms/loop), 1 berkas `.json`** |
| *XGBoost (Huber-log)* | 38.7 | 115.3 | 29.7% | 333.9 | Cepat, gradien halus |
| *LightGBM (MSE-log)* | 38.9 | 115.6 | **28.9%** | 340.1 | Cepat, efisien memori |
| *LSTM (Huber-log)* | 40.2 | 119.6 | 31.1% | 353.9 | Lambat, rentang urutan pendek |
| *Ensemble (Weighted Convex)* | — | — | — | ~331.0 | Sangat kompleks, butuh 2 model |

### Analisis Kegagalan Ensemble
Kami menguji teknik *Ensemble* menggabungkan XGBoost dan LSTM melalui metode rata-rata tertimbang (*weighted convex* dengan bobot $w=0,65$ untuk XGBoost) dan metode *stacking* berbasis regresi Ridge. Hasil eksperimen membuktikan bahwa **model ensemble tidak menghasilkan perbaikan performa yang signifikan**:
* Performa *Weighted Convex* (~331.0s) hanya membaik tipis di atas model dasarnya, dan masih **kalah telak** dibandingkan jika model XGBoost tunggal dilatih pada data latih penuh (*full-training* set) yang menghasilkan *Loop MAE* terbaik sebesar **328,8 detik**.
* Metode *stacking* menghasilkan performa yang lebih buruk akibat *overfitting* pada set validasi yang berukuran kecil (965 putaran).

Ensemble gagal memberikan manfaat karena **korelasi kesalahan (*error*) antar model dasar sangat tinggi**. LSTM dan XGBoost menggunakan fitur penunjuk waktu historis yang sama (`baseline_segment_hour`). Selain itu, karena sekuens putaran relatif pendek (median 19 segmen), model tabular berbasis pohon keputusan (XGBoost) sudah mampu mengekstrak sinyal temporal secara maksimal melalui fitur *rolling mean* dan *expanding mean*.

### Rekomendasi Produksi
Kami memilih **XGBoost Tunggal (MAE-log)** sebagai model final dengan pertimbangan operasional produksi:
1. **Akurasi Bisnis Tertinggi**: Menghasilkan metrik *Loop MAE* terkecil (328,8 detik), memberikan peningkatan performa sebesar **-35,6%** di bawah *baseline* rata-rata historis.
2. **Latensi Inferensi Sangat Rendah**: Waktu prediksi per putaran (19 segmen) hanya **0,85 ms**, berbanding terbalik dengan model LSTM yang 50–100× lebih lambat dan membutuhkan lingkungan runtime PyTorch yang berat.
3. **Kemudahan Pemeliharaan (*Maintainability*)**: Model dapat disimpan dalam satu berkas berukuran kecil (`model_xgb.json`) tanpa ketergantungan pustaka eksternal yang kompleks, serta proses latih ulang (*retraining*) yang selesai hanya dalam hitungan detik ketika terjadi pergeseran data (*data drift*).
