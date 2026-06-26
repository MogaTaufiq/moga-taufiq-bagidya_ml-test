# Log Keputusan Teknis (Living Document)
**Kandidat:** Moga Taufiq Bagidya  
**Peran:** AI/ML Engineer  

Dokumen ini mencatat setiap keputusan arsitektural dan teknis utama selama pengerjaan proyek prediksi waktu tempuh bus per segmen. Format ini dirancang untuk menunjukkan rekam jejak penalaran teknis (*analytical reasoning*) kandidat serta persiapan menghadapi sesi wawancara teknis.

---

## Bagian 1: Keputusan Setup & Tooling

### [D0] Penentuan Lingkungan Kerja & Format Artefak
* **Konteks**: Lingkungan kerja dasar (*base environment*) mini-forge sudah terpasang pustaka besar seperti TensorFlow dan PyTorch (berukuran multi-GB), namun belum memiliki `pyarrow`, `xgboost`, dan `lightgbm`. Saya perlu memilih format penyampaian berkas (*deliverables*) yang paling bersih untuk evaluasi.
* **Keputusan**:
  1. Memasang pustaka yang kurang langsung ke lingkungan dasar (*base environment*) tanpa membuat *virtual environment* conda baru.
  2. Menggunakan tata letak hibrida (*hybrid layout*): logika utama yang dapat digunakan kembali (*reusable logic*) diekstrak ke dalam modul `src/*.py`, sedangkan visualisasi dan alur naratif tipis tetap disimpan di dalam Jupyter Notebook (`.ipynb`).
  3. Memilih PyTorch sebagai runtime untuk pemodelan LSTM (karena TensorFlow 2.16/Keras 3 mengalami konflik fungsionalitas dengan NumPy 2.x).
* **Alasan**: Pembuatan lingkungan baru memerlukan pengunduhan ulang PyTorch/TensorFlow sebesar beberapa gigabyte yang tidak efisien. Format hibrida memberikan keseimbangan antara kemudahan pengujian unit (*unit testing*) dan kemudahan dokumentasi visual bagi reviewer.
* **Alternatif Ditolak**:
  * *Membuat virtual environment baru*: Ditolak karena waktu tunggu pengunduhan ulang library deep learning yang terlalu lama.
  * *Pure Jupyter Notebook*: Ditolak karena menyulitkan pelacakan perubahan kode (*git diff*) dan membuat kode tidak dapat diuji ulang secara otomatis.
* **Trade-off**: Lingkungan dasar menjadi sedikit bercampur dengan pustaka eksternal tambahan. Hal ini dimitigasi dengan mendokumentasikan pustaka tersebut secara ketat di berkas `requirements.txt`.
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Mengapa Anda memisahkan logika ke src/ daripada menulis semua di notebook?"*  
  * *Jawaban*: *"Notebook sangat baik untuk eksperimen cepat dan visualisasi hasil. Namun, untuk menjaga kualitas kode produksi, logika pembersihan data, rekayasa fitur, metrik, dan pelatihan model harus modular dan dapat diuji. Dengan mengekstraknya ke modul `.py` di dalam `src/`, kode menjadi reusable, bersih, dan dapat diintegrasikan dengan pipeline CI/CD."*
* **Status**: FINAL

---

## Bagian 2: Keputusan Eksplorasi Data & Pembersihan

### [D1] Pendefinisian Unit Putaran (Loop) Berbasis `no_do`
* **Konteks**: Spesifikasi tugas menyarankan bahwa kolom `no_do` dapat diabaikan. Namun, saya membutuhkan kunci sekuensial yang tepat untuk model LSTM dan metrik akumulasi *Loop MAE*.
* **Keputusan**: Menetapkan kolom `no_do` sebagai representasi dari satu putaran bus (*loop*) fisik dan menggunakannya sebagai basis unit perhitungan sekuens serta metrik *Loop MAE*.
* **Alasan**: Analisis data membuktikan kolom `trip_id` hanya memiliki 13 nilai unik (dengan rata-rata ~27.000 baris per nilai) yang menandai varian rute utama. Sebaliknya, `no_do` memiliki 17.857 nilai unik dengan median 19 segmen berurutan dan nomor halte `stop_sequence` yang urut secara logis.
* **Alternatif Ditolak**: Menggunakan `trip_id` sebagai kunci urutan sekuensial (ditolak karena tidak merepresentasikan putaran fisik bus yang sebenarnya).
* **Trade-off**: Keputusan ini bertentangan dengan saran eksplisit di berkas petunjuk awal, tetapi sangat didukung oleh bukti statistik distribusi data.
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Mengapa Anda menggunakan no_do padahal instruksi menyarankan mengabaikannya?"*  
  * *Jawaban*: *"Saya melakukan verifikasi empiris pada data. Kolom `trip_id` hanya memiliki 13 nilai unik untuk 351 ribu baris, yang berarti itu adalah kode varian rute makro. Jika kita menggunakannya sebagai sekuens, model LSTM akan dipaksa mempelajari urutan 27 ribu segmen sekaligus, yang tidak masuk akal secara fisik. Kolom `no_do` memiliki median 19 segmen dengan `stop_sequence` yang berurutan secara rapi, menjadikannya satu-satunya representasi putaran bus fisik yang valid."*
* **Status**: FINAL

### [D2] Penanganan Fitur `average_time_sec` sebagai Kebocoran Target (*Target Leakage*)
* **Konteks**: Saya menemukan korelasi yang sangat tinggi (0,84) antara `traveling_time_sec` (aktual) dan `average_time_sec` (rata-rata historis bawaan).
* **Keputusan**: Menghapus kolom `average_time_sec` dari daftar fitur model latih dan menghitung ulang nilai baseline historis yang bersih (*expanding mean* per segmen-jam yang di-*shift*).
* **Alasan**: Ditemukan bukti struktural bahwa rasio aktual terhadap rata-rata (`deviation_ratio`) dibatasi persis pada rentang $[0, 2]$. Pola ini menunjukkan nilai rata-rata tersebut dihitung secara *post-hoc* dari target, sehingga akan menyebabkan kebocoran informasi masa depan (*target leakage*) yang membuat performa model terlihat sangat akurat pada data latih namun gagal di produksi.
* **Alternatif Ditolak**: Menggunakan `average_time_sec` langsung sebagai fitur model.
* **Trade-off**: Memerlukan komputasi tambahan untuk menghitung ulang rata-rata historis yang bersih (*expanding mean*).
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Bagaimana Anda membuktikan adanya leakage pada average_time_sec?"*  
  * *Jawaban*: *"Pertama, secara statistik, rasio aktual terhadap average ter-cap persis di 2,0000 tanpa ada satu pun baris di atasnya. Pada kondisi jalan raya nyata, kemacetan atau kecelakaan pasti akan membuat perjalanan melampaui 2× rata-rata historis. Kedua, analisis ablation menunjukkan memasukkan fitur ini memotong MAE model hingga 44% dan mendominasi kepentingan fitur hingga 85%. Ini adalah bukti kuat bahwa fitur tersebut dibuat menggunakan informasi target aktual."*
* **Status**: FINAL

### [D3] Batasan (*Threshold*) Pembersihan Outlier Waktu Tempuh
* **Konteks**: Nilai maksimum target mencapai ~11 hari yang secara fisik tidak mungkin terjadi untuk perjalanan antar-halte BRT.
* **Keputusan**: Menghapus observasi dengan nilai `traveling_time_sec > 3600` detik (1 jam). Sisa kemiringan distribusi ditangani menggunakan transformasi matematika.
* **Alasan**: Distribusi persentil ke-99 berada di angka 32 menit (masih wajar untuk kemacetan parah di kota metropolitan), tetapi melonjak secara drastis menjadi 6,6 jam pada persentil ke-99,5. Nilai di atas 1 jam diidentifikasi sebagai kesalahan sensor atau bus yang berhenti beroperasi (parkir/mogok).
* **Alternatif Ditolak**: Menggunakan teknik *Winsorization* pada persentil ke-99 (ditolak karena menumpuk data artifisial pada satu nilai dan mempertahankan data yang tidak masuk akal secara fisik).
* **Trade-off**: Membuang sekitar 0,85% baris data yang berpotensi memunculkan celah (*gap*) baru pada putaran bus terkait.
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Mengapa Anda memotong pada 1 jam, bukan persentil ke-99?"*  
  * *Jawaban*: *"Persentil ke-99 (32 menit) masih secara fisik mungkin terjadi pada kondisi macet total di Jakarta. Memotong tepat di persentil ke-99 akan membuang informasi kemacetan ekstrem yang sebenarnya penting dipelajari model. Batas 1 jam dipilih sebagai batas fisis logis antara kemacetan nyata dengan kondisi bus berhenti beroperasi atau kesalahan sensor GPS."*
* **Status**: FINAL

### [D4] Penanganan Celah Perjalanan (*Incomplete Trips*)
* **Konteks**: Sebanyak 47,4% putaran bus memiliki nomor halte yang melompat (*gap*). Saya perlu menentukan kebijakan penanganan data tidak lengkap ini.
* **Keputusan**:
  * Pada model tabular (XGBoost): Data putaran tidak lengkap tetap dipertahankan dan ditandai menggunakan kolom biner `is_gap_suspected`.
  * Pada model sekuensial (LSTM) & Evaluasi Loop MAE: Menghapus putaran yang memiliki gap.
  * Kebijakan Interpolasi: Tidak melakukan interpolasi pada target nilai perjalanan.
* **Alasan**: Model tabular memproses setiap segmen secara independen sehingga target segmen tetap sahih meski segmen lain dalam putaran tersebut hilang. Membuang seluruh data akan membuang 42% observasi berharga. Namun, untuk LSTM dan penjumlahan *Loop MAE*, keutuhan sekuensial adalah syarat mutlak.
* **Alternatif Ditolak**: Melakukan interpolasi target (ditolak karena berisiko memasukkan asumsi artifisial/noise ke dalam label latih).
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Kenapa tidak dibuang saja semua putaran yang memiliki gap?"*  
  * *Jawaban*: *"Karena untuk model tabular, setiap segmen diproses sebagai baris independen. Selisih timestamp keberangkatan dan kedatangan pada segmen tersebut tetap benar dan akurat meskipun halte setelahnya tidak terekam GPS. Membuang seluruh putaran akan menghilangkan hampir setengah dari total data latih secara sia-sia. Pembersihan hanya dilakukan pada model sekuensial (LSTM) karena gap merusak struktur transisi waktu antar-segmen."*
* **Status**: FINAL

### [D5] Transformasi Skewness Target
* **Konteks**: Target memiliki tingkat kemiringan (*skewness*) ekstrem (58,33).
* **Keputusan**: Menerapkan transformasi $log(x + 1)$ atau `log1p` pada target sebelum pelatihan, dan mengembalikannya menggunakan fungsi eksponensial `expm1` saat evaluasi.
* **Alasan**: Transformasi `log1p` menekan skewness menjadi 2,21 dan sangat aman digunakan untuk nilai yang mendekati nol.
* **Alternatif Ditolak**: Transformasi Box-Cox (ditolak karena mengharuskan data bernilai positif mutlak, tidak aman untuk nilai mendekati nol).
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Mengapa menggunakan log1p daripada log biasa?"*  
  * *Jawaban*: *"Beberapa segmen bus memiliki waktu tempuh yang sangat singkat mendekati nol detik. Fungsi log biasa akan menghasilkan nilai minus tak terhingga pada titik nol. Fungsi `log1p(x)` menghitung $log(x+1)$, yang menjamin nilai keluaran tetap aman dan stabil pada angka nol, sembari tetap berperilaku seperti fungsi logaritma biasa untuk nilai yang besar."*
* **Status**: FINAL

### [D6] Strategi Pemisahan Data Latih dan Uji (*Splitting*)
* **Konteks**: Data merupakan deret waktu (*time series*) beruntun dari tanggal 1 Februari hingga 7 Maret.
* **Keputusan**: Menerapkan pemisahan berbasis waktu (*time-based split*) yang memperhatikan unit putaran (*loop-aware*). Batas pemotongan (*cut-off*) ditetapkan pada tanggal 22 Februari 2026.
  * Data Latih: 1–21 Februari 2026 (258.590 baris).
  * Data Uji: 22–28 Februari 2026 (89.519 baris).
* **Alasan**: Menghindari kebocoran data temporal (*temporal leakage*) dan mensimulasikan skenario dunia nyata di mana model dilatih menggunakan data historis masa lalu untuk memprediksi waktu tempuh satu minggu ke depan. Batasan *loop-aware* memastikan bahwa tidak ada putaran bus (`no_do`) yang terpecah sebagian di data latih dan sebagian di data uji.
* **Alternatif Ditolak**: *Random split* tingkat baris (ditolak karena menyebabkan kebocoran informasi temporal yang parah antar-baris).
* **Status**: FINAL

---

## Bagian 3: Keputusan Pemodelan & Evaluasi

### [D7] Pemilihan Model Final XGBoost (MAE-log)
* **Konteks**: Saya menguji berbagai algoritma model (XGBoost, LightGBM, LSTM) dengan berbagai objektif loss function.
* **Keputusan**: Memilih model **XGBoost** dengan fungsi objektif optimasi L1 di ruang log (`objective='reg:absoluteerror'`) sebagai model final.
* **Alasan**: Model ini memberikan performa terbaik pada metrik bisnis utama dengan nilai *Loop MAE* sebesar **258,8 detik** (peningkatan akurasi sebesar **-49,3%** dibanding baseline). Model tabular terbukti lebih unggul dibanding LSTM karena fitur-fitur seperti *rolling mean*, *expanding mean*, dan dwell terminal (D12) telah berhasil mengekstrak informasi temporal & operasional secara maksimal.
* **Catatan iterasi**: Pipeline awal dengan 15 fitur menghasilkan Loop MAE 328,8 detik. Penambahan dua fitur baru hasil insight operasional — `time_since_prev_arrival_sec` (D12, dwell terminal) dan `bus_encoded` (D13, target encoding) — menurunkan Loop MAE ke 258,8 detik (penurunan ~21,3% incremental). Tuning Optuna 30-trial diuji tapi tidak memperbaiki (val LoopMAE 321,9 dengan 7 hyperparameter di-tune), sehingga parameter default dipertahankan.
* **Alternatif Ditolak**: Model LSTM (ditolak karena performa Loop MAE lebih rendah, yaitu 368,5 detik dengan 17 fitur, latensi tinggi, dan proses latih ulang yang lama).
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Mengapa XGBoost dengan objektif MAE lebih unggul daripada MSE di data ini?"*  
  * *Jawaban*: *"Data target kita memiliki distribusi ekor panjang (long-tail) karena sisa-sisa kemacetan kota. Objektif MSE (L2 loss) sangat sensitif terhadap kesalahan besar di ujung ekor distribusi, sehingga gradiennya akan terseret untuk mencocokkan nilai ekstrem tersebut. Objektif MAE (L1 loss) lebih robust terhadap pencilan, sehingga menghasilkan prediksi nilai tengah yang lebih akurat untuk mayoritas perjalanan bus normal."*
* **Status**: FINAL

### [D8] Keputusan Terkait Model Gabungan (*Ensemble*)
* **Konteks**: Brief menyarankan penggunaan arsitektur ensemble LSTM + XGBoost.
* **Keputusan**: Menguji pendekatan *ensemble* (*Weighted Convex* dan *Ridge Stacking*), namun memutuskan untuk **tidak menggunakannya** pada model produksi akhir.
* **Alasan**: Hasil eksperimen menunjukkan performa model ensemble tidak memberikan peningkatan yang signifikan dan kalah dari model tunggal XGBoost yang dilatih pada data latih penuh dengan 17 fitur (Loop MAE final 258,8 detik). Menggunakan ensemble akan meningkatkan latensi inferensi sebesar 50–100×, meningkatkan kompleksitas pustaka (Torch + XGBoost), dan mempersulit proses latih ulang di produksi.
* **Persiapan Interview**:  
  * *Pertanyaan*: *"Kapan model ensemble tidak memberikan manfaat signifikan?"*  
  * *Jawaban*: *"Model ensemble tidak akan memberikan manfaat jika kesalahan (error) dari model-model dasarnya saling berkorelasi erat (karena menggunakan basis fitur historis yang sama). Selain itu, jika salah satu model (XGBoost) sudah sangat mendominasi performa dibandingkan model lainnya (LSTM), penggabungan hanya akan menambah variansi tanpa memberikan peningkatan akurasi yang berarti."*
* **Status**: FINAL

### [D9] Strategi Penanganan Kelangkaan Data (*Sparsity*)
* **Konteks**: Terdapat beberapa kombinasi halte-jam yang memiliki jumlah sampel data sangat sedikit.
* **Keputusan**: Menerapkan *Bayesian target encoding* dengan prior smoothing $m=20$, dikombinasikan dengan regularisasi pohon `min_child_weight=5`.
* **Alasan**: Menghindari model membuat keputusan berdasarkan statistik dari sampel data yang terlalu sedikit (cegah overfit pada noise). Rata-rata dari sampel kecil secara otomatis ditarik mendekati rata-rata global.
* **Status**: FINAL

### [D10] Penanganan Ledakan Nilai MAPE
* **Konteks**: Pada segmen yang sangat pendek, nilai aktual waktu tempuh mendekati nol detik sehingga persentase kesalahan (MAPE) dapat meledak hingga ribuan persen.
* **Keputusan**: Menghapus observasi dengan waktu aktual `traveling_time_sec < 5` detik khusus pada perhitungan metrik `mape_segment`, serta melaporkan metrik alternatif `sMAPE` (Symmetric MAPE) yang dibatasi pada rentang $[0, 200\%]$.
* **Alasan**: Menjaga stabilitas pelaporan metrik akurasi segmen agar tidak terdistorsi oleh nilai pembagian yang sangat kecil.
* **Status**: FINAL

### [D11] Penghapusan Fitur `arrival_time` untuk Mencegah Kebocoran Data
* **Konteks**: Kolom `arrival_time` merekam waktu kedatangan bus di halte tujuan.
* **Keputusan**: Menghapus fitur `arrival_time` dari model dan menghitung semua fitur berbasis waktu (jam, hari, rush hour) hanya dari kolom keberangkatan (`from_arrival_time_str`).
* **Alasan**: Waktu kedatangan (*arrival time*) baru diketahui setelah bus menyelesaikan segmen tersebut. Menggunakannya sebagai fitur prediksi waktu tempuh segmen saat bus baru berangkat adalah bentuk kebocoran data (*data leakage*) yang fatal karena informasi masa depan digunakan sebagai input.
* **Status**: FINAL

### [D12] Penambahan Fitur Dwell Terminal (`time_since_prev_arrival_sec`)
* **Konteks**: Analisis komposisi outlier yang dibuang (D3) menunjukkan 99% outlier berada pada `stop_sequence == 1` atau pada posisi terminal akhir loop, dengan puncak pada jam 20–22 (malam). Pola ini bukan *random sensor error* seperti dugaan awal, melainkan **waktu dwell bus parkir di terminal** menunggu jadwal keberangkatan/pulang. Sinyal operasional ini sebelumnya hilang ketika baris dwell di-drop.
* **Keputusan**: Menambahkan fitur **`time_since_prev_arrival_sec`** = selisih antara *departure time* segmen sekarang dengan *arrival time* segmen terakhir bus yang sama (per `bus_body_no`, di-shift, dihitung pada frame gabungan train+test untuk menghindari NaN di batas split).
* **Alasan**: Bersifat *leakage-free* (hanya menggunakan arrival yang sudah terjadi sebelum departure sekarang). Menangkap proksi dwell, headway antar-loop, dan kondisi bus baru aktif. Setelah penambahan, Loop MAE turun **328,8 → 258,8 detik (−21,3%)**.
* **Bukti tanggung jawab fitur** (sanity tests):
  * Median AE = 12,57 detik (tetap jujur, bukan ~0 → tidak ada leakage).
  * Korelasi linear dengan target = **-0,001** (mendekati nol) → bukan leakage linear.
  * Uji permutasi: meng-shuffle fitur di test set membuat Loop MAE meledak **259 → 424 detik (Δ +165s)** → bukti kuat fitur asli yang bertanggung jawab atas perbaikan.
* **Alternatif Ditolak**: Membuat fitur kompleks "dwell duration" yang memerlukan kolom waktu mulai/berhenti eksplisit (data tidak tersedia secara mentah); pakai outlier yang sudah di-drop sebagai data tambahan (sirkular karena outlier = target tinggi).
* **Persiapan Interview**:
  * *Pertanyaan*: *"Bagaimana fitur dengan korelasi linear nol bisa memperbaiki model sebanyak 70 detik Loop MAE?"*
  * *Jawaban*: *"Model tree-based seperti XGBoost memanfaatkan **interaksi non-linear** antar fitur, bukan korelasi linear marginal. Fitur dwell rendah pada stop_sequence=1 menandai 'bus baru aktif' (cenderung lebih lambat di segmen pertama), sedangkan dwell tinggi di stop_seq tengah menandai segmen problematik. Tree melakukan split kondisional pada interaksi ini, sehingga sinyalnya jauh lebih kuat daripada yang terlihat di korelasi linear. Permutation test mengkonfirmasi ini: shuffle fitur membuat Loop MAE melonjak 165 detik."*
* **Status**: FINAL

### [D13] Penambahan Fitur Bus Target Encoding (`bus_encoded`)
* **Konteks**: Kolom `bus_body_no` (259 nilai unik) sebelumnya tidak dipakai sebagai fitur. Analisis variansi rata-rata waktu antar bus menunjukkan std=19s pada range 85–312s (coefisien variasi 0,129) — sinyal moderat yang konsisten lintas segmen, kemungkinan mencerminkan karakteristik per-bus (umur kendaraan, gaya supir, kondisi mesin).
* **Keputusan**: Menambahkan fitur **`bus_encoded`** = target encoding rata-rata waktu tempuh per `bus_body_no` dengan *Bayesian smoothing* ($m=20$). Encoding dihitung **hanya pada data train**, lalu di-map ke test. Bus yang tidak terlihat di train (cold-start) di-fallback ke rata-rata global target.
* **Alasan**: Bersifat *leakage-free* (encoding fit hanya di train, tidak menggunakan label test). Smoothing $m=20$ mencegah encoding untuk bus dengan sedikit observasi tertarik ke statistik yang noisy. Kontribusi marginal terhadap Loop MAE: ~3 detik di atas D12.
* **Alternatif Ditolak**: One-hot encoding 259 kategori (terlalu sparse, meningkatkan dimensi tanpa manfaat); target encoding tanpa smoothing (rentan overfit pada bus dengan obs sedikit).
* **Persiapan Interview**:
  * *Pertanyaan*: *"Mengapa target encoding aman dan tidak menyebabkan leakage?"*
  * *Jawaban*: *"Target encoding hanya melanggar leakage jika encoding dihitung dari data yang sama dengan yang dievaluasi. Saya menerapkannya dengan disiplin: encoding fit pada train saja, lalu di-map ke test sebagai lookup tabel. Bus yang muncul pertama kali di test akan di-fallback ke rata-rata global. Smoothing m=20 juga mencegah encoding bus dengan sedikit observasi (mis. 3 baris) mendapat encoding ekstrem yang tidak generalisir."*
* **Status**: FINAL
