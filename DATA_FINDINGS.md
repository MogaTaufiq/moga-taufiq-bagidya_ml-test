# DATA_FINDINGS.md — Temuan Data Terverifikasi & Jebakan

> Ini hasil inspeksi langsung pada `AI_Engineer_dataset.parquet` (bukan asumsi dari brief).
> **Baca ini sebelum menulis kode apa pun.** Semua angka di bawah sudah diverifikasi dan
> bisa Moga klaim saat interview. Tetap **verifikasi ulang** angka-angka ini di notebook EDA
> milik sendiri — jangan copy buta; jadikan ini peta, bukan hasil akhir.

---

## 0. Ringkasan dataset

| Properti | Nilai |
|---|---|
| Baris × Kolom | **351.103 × 10** |
| Rentang waktu | **2026-02-01 → 2026-03-07** (~34 hari) |
| Sifat data | Sintetis / dummy (semua ID berprefiks `*-DUMMY-*`) |
| Missing values | **0** di semua kolom (tapi lihat jebakan: "tidak missing" ≠ "tidak bermasalah") |
| `route_code` | **1 nilai saja** (`ROUTE-DUMMY-0001`) → zero-variance |
| `trip_id` | **13 nilai** (pola/varian rute, sebagian punya >100rb baris) |
| `no_do` | **17.857 nilai** (median **19** segmen/loop) |
| `segment_id` | **44 nilai** |
| `bus_body_no` | **259 nilai** |

Cukup untuk fitur **hour / weekday / weekend**, **tidak cukup** untuk pola musiman/bulanan.
Cocok untuk **time-based split** (mis. train 27 hari awal, test 7 hari akhir).

---

## ⚠️ JEBAKAN #1 — "Satu putaran" adalah `no_do`, BUKAN `trip_id`

Brief menulis: *"trip_id terdiri dari beberapa segmen berurutan…"* dan *"no_do bisa
diabaikan."* **Data membuktikan sebaliknya.**

- `trip_id` hanya **13 nilai** untuk 351rb baris → rata-rata ~27.000 baris/`trip_id`.
  Itu mustahil jadi satu loop fisik. **`trip_id` = pola/varian rute**, bukan satu perjalanan.
- `no_do` punya **17.857 nilai**, **median 19 segmen** per `no_do`, dan di dalam satu `no_do`
  `stop_sequence` berjalan rapi `1,2,3,…,19`. **`no_do` = satu putaran fisik (loop).**
- `(bus_body_no, no_do)` menghasilkan 17.858 grup ≈ sama dengan `no_do` → praktis satu
  `no_do` milik satu bus. Aman pakai `no_do` saja sebagai kunci loop.

**Implikasi teknis (PENTING):**
- **Sequence key untuk LSTM = `no_do`** (urut `stop_sequence`), **bukan `trip_id`**.
- **Loop MAE (MAE putaran)** dihitung per `no_do`, bukan per `trip_id`.
- `trip_id` tetap berguna sebagai **fitur kategorikal** (varian rute → beda karakteristik).

**Talking point interview:**
> "Saya menemukan `trip_id` cuma 13 nilai untuk 351rb baris, jadi itu varian rute, bukan
> loop fisik. Loop sebenarnya adalah `no_do` (~17,8rb loop, ~19 segmen). Karena itu saya
> pakai `no_do` sebagai unit sekuens untuk LSTM dan unit perhitungan Loop MAE — meski brief
> menyarankan `no_do` bisa diabaikan."

---

## ⚠️ JEBAKAN #2 — `average_time_sec` kemungkinan besar TARGET LEAKAGE

Ini temuan paling kritis. Kalau salah tangani, seluruh metrik jadi palsu (over-optimistis).

**Bukti:**
- `deviation_ratio = traveling_time_sec / average_time_sec` ter-**cap persis di [0, 2]**:
  min `0.000`, median `0.999`, p95 `1.261`, p99 `1.538`, **max `2.000`**, dan **0,00% baris
  punya rasio > 2**.
- Distribusi rasio menumpuk di sekitar 1.0: **~87% baris ada di [0.75, 1.25)**.
- Korelasi `traveling_time_sec` vs `average_time_sec` = **0.84** (tinggi).
- Bahkan outlier ekstrem `traveling_time_sec = 958.259 dtk` (~11 hari!) tetap punya
  `average_time_sec = 523.009 dtk` yang **proporsional** (rasio ~1.83).

**Kenapa ini mencurigakan:** baseline historis **asli** tidak mungkin membatasi aktual ≤ 2×.
Macet/insiden nyata menghasilkan rasio 3×, 5×, bahkan lebih. Faktanya **tidak ada satu pun**
baris yang melebihi 2× "rata-rata historis"-nya. Ini justru **mengontradiksi klaim brief
sendiri** ("some values vastly exceeding their historical average"). Pola ini khas data
sintetis di mana `average` diturunkan **per-baris** dari `traveling_time`.

**Implikasi & strategi (wajib):**
1. **Jangan** pakai `average_time_sec` mentah sebagai fitur tanpa uji. Kalau ia membawa
   informasi target, model akan terlihat "akurat" padahal curang.
2. **Recompute baseline bersih sendiri:** *expanding mean* `traveling_time_sec` per
   `(segment_id, hour)`, di-`shift(1)` (atau pakai hanya data sebelum timestamp baris)
   sehingga baris saat ini **tidak** menghitung rata-ratanya sendiri. Ini baseline
   leakage-free yang sah dipakai sebagai fitur dan sebagai pembanding anomali.
3. **Uji empiris leakage:** latih model dengan vs tanpa `average_time_sec`. Jika dengan-nya
   tiba-tiba MAE jatuh drastis ke level "terlalu bagus", itu konfirmasi leakage.
4. `deviation_ratio` versi brief tetap **boleh dihitung** (memang diminta sebagai kolom
   output), tapi sadari ia ter-bound; hitung **juga** deviasi terhadap baseline recompute.

**Talking point interview:**
> "Saya curiga `average_time_sec` bocor karena `deviation_ratio` ter-cap persis di [0,2],
> 0% di atas 2 — mustahil untuk baseline historis nyata di kondisi macet. Jadi saya
> menghitung ulang baseline leakage-free (expanding mean per segment-hour, di-shift) dan
> menguji leakage secara empiris dengan ablation."

---

## ⚠️ JEBAKAN #3 — Kalibrasi ulang klaim brief: skewness NYATA, gap NYATA, sparsity RINGAN

Brief menyebut tiga masalah. Setelah dicek, porsinya berbeda — **jangan melebih-lebihkan**.

### (a) Skewness — NYATA & ekstrem
- `traveling_time_sec`: min `0.2`, median `89.1`, mean `467.3`, p95 `459.3`, p99 `1909.6`,
  **max `958.259,7`** (~11 hari).
- **Skewness mentah = 58.33**; setelah **`log1p` → 2.21** (jauh membaik, masih kanan).
- Outlier teratas (mis. `958.259 dtk`) jelas **error sensor/anomali**, bukan macet wajar.

**Strategi:** transform `log1p` pada target + tangani outlier ekstrem (capping/winsorize
atau buang yang fisik-mustahil, dengan threshold yang dijelaskan di README). Pilih
loss/metrik yang sadar-skew (MSE di ruang log = relative error; pertimbangkan MAE & metrik
robust).

### (b) Incomplete trips (gap `stop_sequence`) — NYATA & sering
- **47,4% loop** (8.458 dari 17.857) punya gap di `stop_sequence` (ada nomor halte yang lompat).
- **179 loop** hanya berisi 1 segmen (fragmen).
- Panjang loop: median 19, range **1–72** (72 = anomali, mungkin dua loop tergabung/varian panjang).

**Strategi:** fitur **`is_gap_suspected`** (True jika `stop_sequence.diff() > 1` dalam satu
`no_do`) memang bermakna. Untuk training: drop fragmen sangat pendek & gap besar; interpolasi
hanya untuk gap kecil bila ada dukungan historis (detail di `METHODOLOGY.md`).

### (c) Sparsity (segment, hour) — RINGAN di dataset ini
- Total kombinasi `(segment_id, hour)` = **1.029**.
- Hanya **3,3%** kombinasi punya **<30 observasi**; **0,9%** punya **<10**.
- Karena `route_code` cuma 1, granularitas `(route, segment, hour)` **tidak** menambah
  kelangkaan di sini.

**Strategi (jujur):** sparsity **bukan** masalah besar di bulk data ini. Ia jadi relevan saat
(i) granularitas diperhalus `(segment, hour, weekday)` atau `(segment, hour, is_weekend)`,
dan (ii) generalisasi ke kombinasi minim observasi. Tetap siapkan **hierarchical fallback**
+ **smoothing** (lihat `METHODOLOGY.md`) — tapi **jangan** mengklaim sparsity parah kalau
datanya tidak menunjukkan itu. Reviewer akan menghargai kalibrasi yang jujur.

---

## Catatan kolom (ringkas)

| Kolom | Peran | Catatan |
|---|---|---|
| `bus_body_no` | identitas bus | 259 unik; ~1:1 dengan `no_do` |
| `segment_id` | pasangan halte asal-tujuan | 44 unik; **fitur kategorikal kuat** |
| `route_code` | rute utama | **1 nilai → drop sebagai fitur** (catat: di data multi-rute akan penting) |
| `trip_id` | varian/pola rute | 13 unik; fitur kategorikal, **bukan** unit loop |
| `stop_sequence` | urutan halte dalam loop | sumber deteksi gap |
| `traveling_time_sec` | **TARGET** | sangat skewed; ada outlier mustahil |
| `from_arrival_time_str`, `arrival_time` | timestamp | sumber fitur waktu (hour/weekday/weekend/rush) |
| `average_time_sec` | "baseline historis" | **DICURIGAI LEAKAGE** — lihat Jebakan #2 |
| `no_do` | **unit satu putaran (loop)** | kunci sekuens LSTM & Loop MAE |

---

## Urutan kerja yang disarankan dari temuan ini

1. Validasi ulang semua angka di atas di `01_eda.ipynb` (reproduksi sendiri).
2. Putuskan & dokumentasikan threshold outlier + definisi gap (catat di `DECISIONS.md`).
3. Recompute baseline leakage-free → jadikan fondasi fitur deviasi.
4. Bangun fitur waktu + identitas (segment/trip) + sekuens (per `no_do`).
5. Baru masuk modeling & perbandingan ≥3 model.
