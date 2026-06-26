# METHODOLOGY.md — Cara Menyelesaikan dengan Benar

> Ini "how-to" teknisnya. Setiap strategi di sini harus berakhir jadi keputusan tercatat di
> `DECISIONS.md` dan, kalau relevan, masuk write-up. Claude Code: ikuti urutan ini, dan tiap
> langkah keluarkan Insight Report (lihat `CLAUDE.md` §2).

---

## Fase pengerjaan (dan output tiap fase)

| Fase | Fokus | Output |
|---|---|---|
| 1. EDA terarah | Reproduksi temuan `DATA_FINDINGS.md`, pahami struktur loop & leakage | `01_eda.ipynb` + 5 temuan kunci + isi awal `DECISIONS.md` |
| 2. Cleaning | Threshold outlier, definisi gap, recompute baseline bersih | data ter-clean + fungsi di `src/data.py` |
| 3. Feature engineering | ≥5 fitur berjustifikasi + `is_gap_suspected` + `deviation_ratio` | `02_feature_engineering.ipynb`, `outputs/training_ready.parquet` |
| 4. Modeling | Baseline → 3+ model → ensemble, fair comparison di subset | `03_modeling.ipynb`, artifact model |
| 5. Evaluasi | MAE, RMSE, MAPE Segment, Loop MAE; banding ke baseline | `04_evaluation.ipynb`, tabel metrik |
| 6. Write-up + README | Rangkai reasoning jadi 2 halaman + README | `WRITEUP.md`, `README.md` |

Di antara fase, baca ulang `DECISIONS.md`. Kalau ada keputusan yang pas dibaca lagi terasa
"kok gw pilih ini ya?" — itu sinyal harus ditinjau ulang sebelum lanjut.

---

## 1. Strategi incomplete trips (drop vs interpolasi)

Konteks dari data: 47,4% loop punya gap; 179 loop hanya 1 segmen; panjang loop 1–72.

**Aturan keputusan yang disarankan (sesuaikan & catat threshold-nya):**
- **Drop** bila: loop terlalu pendek (mis. `n_segmen < 5`) **atau** gap besar berurutan
  (mis. `stop_sequence.diff() > 2`). Alasan: terlalu sedikit konteks sekuens / risiko
  menyuntik sinyal palsu.
- **Pertahankan + tandai** bila gap kecil (1 halte hilang) dan loop cukup panjang: jangan
  buang seluruh loop hanya karena 1 gap; cukup beri `is_gap_suspected = True` agar model /
  evaluasi sadar.
- **Interpolasi `traveling_time_sec`?** Hanya untuk gap kecil **dan** ada dasar historis
  yang kuat (mis. median segmen pada jam itu). **Default lebih aman: jangan interpolasi
  target** untuk training (menambah label artifisial = noise). Interpolasi lebih cocok untuk
  visualisasi/penyajian, bukan untuk membuat label latih.

**Untuk Loop MAE:** kalau sebuah loop tidak lengkap, jangan bandingkan total loop-nya secara
penuh — hitung Loop MAE hanya pada loop yang lengkap, atau normalisasi per jumlah segmen
yang ada. Catat pilihan ini.

**Inti yang harus dipahami:** *drop = lebih sedikit noise tapi kehilangan data; interpolasi
= mempertahankan data tapi menambah asumsi.* Pilih per kondisi, bukan satu aturan global.

---

## 2. Penanganan skewness (minimal 2 teknik)

Data: skew mentah 58.33; `log1p` → 2.21; ada outlier mustahil (958rb dtk).

**Teknik 1 — Transformasi target.** `log1p(traveling_time_sec)` (pakai `+1` agar aman untuk
nilai 0/kecil). Saat prediksi, balikkan dengan `expm1`. Alternatif: Yeo-Johnson (menangani
0/negatif; Box-Cox tidak, karena butuh strictly positive). Pilih satu, jelaskan alasannya.

**Teknik 2 — Penanganan outlier.** Winsorize/capping pada persentil atas (mis. p99) **atau**
buang baris fisik-mustahil (mis. `traveling_time_sec > X` detik yang tak masuk akal untuk
satu segmen). Tetapkan `X` dengan alasan domain dan **catat di README** sebagai asumsi.

**Dampak ke loss/metrik (wajib dibahas di write-up):**
- MSE/RMSE di **ruang log** ≈ menghukum **relative error**, cocok untuk target skewed —
  salah 30 detik pada segmen 60 detik lebih "berat" daripada pada segmen 600 detik.
- Tanpa transform, RMSE didominasi outlier; MAE lebih robust tapi mengabaikan ekor.
- **Huber Loss** sebagai jalan tengah: berperilaku seperti MSE untuk error kecil (smooth,
  gradien stabil saat training) dan seperti MAE untuk error besar (robust ke outlier).
  Cocok di data ini karena: (i) ada outlier sensor ekstrem yang akan menyeret MSE, tapi
  (ii) kita tetap mau gradien yang halus di mayoritas baris normal. Parameter `delta`
  mengontrol titik transisi — tune di validation set, jangan asal default. Di XGBoost
  pakai `objective='reg:pseudohubererror'` (dengan `huber_slope`); di Keras pakai
  `tf.keras.losses.Huber(delta=...)`.
- Pertimbangkan **quantile/pinball loss** bila ingin model sadar ekor secara eksplisit
  (mis. ingin prediksi P90 untuk worst-case scheduling) — opsional, lebih advance.

**Cara memilih:** mulai dari `log1p` + MAE atau `log1p` + Huber. Bandingkan di validation.
Jangan pilih loss "karena keren" — pilih yang turunkan Loop MAE di validation.

---

## 3. Strategi sparsity (tanpa overfit ke noise)

Data: sparsity ringan di `(segment, hour)` (3,3% kombinasi <30 obs). Relevan terutama di
granularitas halus & generalisasi.

**Hierarchical fallback / backoff.** Bila `(segment, hour)` minim observasi, mundur ke
`(segment)` lalu ke global. Bisa diimplementasi sebagai fitur agregat berjenjang.

**Target/mean encoding dengan Bayesian smoothing.** Blend mean kelompok dengan mean global
berbobot jumlah observasi:
`enc = (n*mean_group + m*mean_global) / (n + m)`, dengan `m` = kekuatan smoothing.
Kelompok dengan `n` kecil otomatis tertarik ke global → tahan overfit ke noise.
**Hitung encoding HANYA dari data train** (anti-leakage), lalu map ke test.

**Regularisasi.** Pada model tabular, jaga `min_child_weight` / `min_data_in_leaf` cukup
besar agar daun tidak terbentuk dari segelintir baris langka.

**Inti yang harus dipahami:** smoothing = "jangan percaya penuh rata-rata dari sedikit
sampel; tarik ke arah rata-rata global sampai cukup bukti".

---

## 4. Feature engineering (target ≥5–7 fitur berjustifikasi)

Wajib ada `is_gap_suspected` dan `deviation_ratio` (Task 4). Tambahkan, dengan alasan domain:

1. **Fitur waktu:** `hour`, `day_of_week`, `is_weekend`, `is_rush_hour`
   (peak Jakarta: pagi ±06–09, sore ±16–19). *Alasan:* travel time sangat bergantung jam.
2. **Cyclic encoding:** `hour_sin`, `hour_cos`. *Alasan:* jam 23 dan 00 berdekatan secara
   siklis tapi jauh secara numerik; encoding sin/cos memperbaiki ini.
3. **`deviation_ratio`** = `traveling_time_sec / baseline`. **Gunakan baseline leakage-free
   hasil recompute** (lihat `DATA_FINDINGS.md` Jebakan #2). Hitung **juga** versi terhadap
   `average_time_sec` bawaan karena diminta sebagai kolom output — tapi sadari ia ter-bound.
   *Guard:* jika baseline = 0, set rasio ke NaN/1.0 + flag (hindari div-by-zero / infinity).
4. **`rolling_mean_segment_k`** = rata-rata `traveling_time_sec` k-observasi terakhir pada
   segmen tsb (di-shift, leakage-free). *Alasan:* menangkap kondisi terkini/tren.
5. **`segment_volatility`** = std historis `traveling_time_sec` per segmen (dari train).
   *Alasan:* proksi ketidakstabilan/kemacetan kronis suatu segmen.
6. **`trip_progress`** = `stop_sequence / max_stop_sequence` dalam satu `no_do`.
   *Alasan:* posisi dalam loop (awal/tengah/akhir) berkorelasi dengan kondisi lalu lintas.
7. **`is_gap_suspected`** = True bila `stop_sequence.diff() > 1` dalam `no_do`.
   *Alasan:* menandai loop tidak lengkap agar model/evaluasi sadar.

Encoding kategorikal: `segment_id` (44) & `trip_id` (13) → one-hot/target encoding (target
encoding di-fit di train saja). `route_code` **di-drop** (zero-variance).

---

## 5. Protokol perbandingan model (inti penilaian)

### 5a. Selalu mulai dari baseline
**Baseline historical-mean:** prediksi = rata-rata `traveling_time_sec` per `(segment, hour)`
dari train. Ini **lantai** yang semua model harus kalahkan. Interviewer hampir pasti tanya
"berapa lift di atas baseline?". Tanpa baseline, angka model tidak bermakna.

### 5b. Bandingkan minimal 3 model, fair
Kandidat default:
- **XGBoost / LightGBM** — workhorse tabular: gradient boosting, menangkap interaksi fitur,
  relatif robust ke outlier, cepat. Backbone paling mungkin menang di data ini.
- **LSTM / GRU** — menangkap dependensi **antar segmen dalam satu `no_do`** (urut
  `stop_sequence`). Input: sekuens per loop. Hati-hati: loop pendek (~19 langkah) → sinyal
  sekuens mungkin tipis.
- **RandomForest / Ridge (opsional)** — pembanding/kontras.

**Aturan fair comparison:**
- Split **sama** (time-based) untuk semua model.
- Fitur tabular **sama** untuk model tabular; untuk LSTM siapkan representasi sekuens yang setara.
- Metrik **sama**, dilaporkan berdampingan (lihat §6).
- Boleh banding di **subset** (sampel **utuh per `no_do`**, jaga urutan waktu) demi kecepatan,
  lalu **retrain model terpilih di data penuh**.

### 5c. Uji ensemble — jangan asumsikan menang
Strategi gabung:
- **Weighted average** — sederhana, bobot bisa di-set/di-tune. Mudah dijelaskan tapi bobot
  agak arbitrer.
- **Stacking** (meta-learner linear di atas prediksi base) — bobot **dipelajari dari data**,
  lebih defensible. **Disarankan** kalau mau pakai ensemble.

**Kapan ensemble GAGAL memberi manfaat (wajib dibahas di write-up):**
- Saat base models punya **error berkorelasi tinggi** (mereka salah di tempat yang sama).
- Saat **satu model sudah mendominasi** dan yang lain hanya menambah varians.
- Saat **fitur tabular sudah menangkap mayoritas varians** dan sekuens pendek → LSTM
  menyumbang sedikit. **Ini sangat mungkin terjadi di dataset ini.** Kalau hasilnya begitu,
  **laporkan jujur**: "ensemble tidak mengalahkan XGBoost tunggal secara signifikan, jadi
  saya memilih model tunggal yang lebih simpel & mudah di-deploy." Itu jawaban ML engineer
  yang matang, bukan kegagalan.

### 5d. Pemilihan final
Pilih berdasarkan **Loop MAE** (utama) + MAE/RMSE, lalu pertimbangkan tiga dimensi
non-akurasi yang menentukan apakah model layak masuk production:

- **Latensi inferensi.** Ukur waktu prediksi per loop (`no_do`, ~19 segmen) di mesin
  pengembangan. XGBoost/LightGBM biasanya **<10 ms**; LSTM bisa **50–500 ms** tergantung
  arsitektur & batch. Untuk transit real-time (estimasi ETA, headway adjustment), 50ms vs
  500ms beda nasibnya. Bilang eksplisit angka latensi di README.
- **Kompleksitas deployment.** Tabular model = 1 artifact `.pkl`, 1 dependency. LSTM =
  artifact lebih besar, butuh TF/Torch runtime, plus pipeline preprocessing sekuens yang
  lebih rumit. Setiap titik kompleksitas = titik kegagalan potensial di production.
- **Maintainability & retrainability.** Seberapa mudah pipeline ini dilatih ulang saat data
  drift (kondisi lalu lintas berubah)? XGBoost retrain dalam menit; LSTM bisa jam.

**Aturan keputusan praktis:** jika model A mengalahkan model B sebesar X% di Loop MAE tapi
2× lebih lambat & lebih sulit di-deploy, pilih B kecuali X cukup besar untuk membenarkan
biaya tersebut. Tulis aturan ini eksplisit di README — itu yang membedakan ML engineer dari
researcher.

---

## 6. Metrik & evaluasi

Implementasi di `src/metrics.py`. Definisi:

- **MAE** = mean(|y − ŷ|). Robust, interpretasi langsung (detik).
- **RMSE** = sqrt(mean((y − ŷ)²)). Menghukum error besar.
- **MAPE Segment** = mean(|y − ŷ| / y) per segmen, lalu agregasi. **Guard:** singkirkan/handle
  segmen dengan `y` sangat kecil (≈0) agar tidak meledak; atau pakai **sMAPE**.
- **Loop MAE (MAE putaran)** — metrik bisnis utama:
  1. Untuk tiap `no_do`: `total_aktual = Σ y`, `total_prediksi = Σ ŷ` (atas segmen di loop itu).
  2. `error_loop = |total_aktual − total_prediksi|`.
  3. **Loop MAE = mean(error_loop)** atas semua loop (lengkap).
  *Kenapa penting:* operasi transit peduli akurasi **satu putaran penuh** (untuk headway &
  jadwal), bukan akurasi per-segmen. Model bisa bagus per-segmen tapi error-nya menumpuk
  searah sepanjang loop → Loop MAE membongkar itu.

**Selalu** laporkan metrik **berdampingan dengan baseline** dan antar model dalam satu tabel.
Evaluasi **di ruang asli** (detik), bukan ruang log — kembalikan prediksi via `expm1` dulu.

---

## 7. Alokasi write-up (maks 2 halaman — ketat)

| Bagian | Porsi kira-kira |
|---|---|
| Strategi incomplete trips | ¼ halaman |
| Penanganan skewness (≥2 teknik + dampak loss) | ¼ halaman |
| Strategi sparsity | ¼ halaman |
| Justifikasi ≥5 fitur | ½ halaman |
| Justifikasi model + kapan ensemble gagal + cara gabung | ½ halaman |
| Hasil akurasi (tabel MAE/RMSE/MAPE/Loop MAE) | ¼ halaman |

Selipkan **temuan leakage `average_time_sec`** dan **`no_do` sebagai unit loop** — dua hal
itu paling membedakan write-up Moga dari kandidat lain. Bahasa harus suara Moga sendiri:
setiap kalimat harus tahan saat ditanya "jelaskan ini lebih dalam".
