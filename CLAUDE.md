# CLAUDE.md: Fake News Detection v1

> File konteks untuk Claude Code. Baca ini dulu sebelum mengerjakan apa pun di repo ini.

## Tentang project & pemiliknya

- **Pemilik:** mahasiswa yang sedang belajar jadi AI/ML engineer. Saat ini di **Fase 1 roadmap (ML klasik)**, belajar dari StatQuest + scikit-learn.
- **Tujuan project:** klasifikasi berita **hoaks vs valid** pakai ML klasik (TF-IDF + Logistic Regression), dengan Naive Bayes sebagai pembanding.
- **Tujuan belajar (lebih penting dari akurasi):** paham alur ML end-to-end (data → fitur → model → evaluasi), bisa membaca dan mengkritisi model sendiri, dan punya project portfolio yang rapi.
- **Rencana ke depan:** model ini akan jadi **baseline** untuk "Fake News v2" (LLM + RAG) di Fase 3. Jadi test set dan hasil evaluasinya harus disimpan rapi dan tidak berubah.

## 🎓 Mode belajar (WAJIB diikuti Claude Code)

Pemilik repo sedang **belajar**, bukan sekadar ingin project jadi. Karena itu:

1. **Jangan langsung menulis seluruh kode untuk satu milestone.** Default-nya: jelaskan apa yang harus dilakukan, beri hint atau kerangka (signature fungsi, TODO), lalu biarkan pemilik mencoba.
2. Tulis kode lengkap **hanya kalau diminta eksplisit** ("tulisin aja", "kasih solusinya") atau untuk hal boilerplate (setup folder, `.gitignore`, `requirements.txt`).
3. Saat me-review kode pemilik: tunjukkan bug atau masalahnya, jelaskan **kenapa**, sarankan perbaikan. Jangan diam-diam menulis ulang semuanya.
4. Hubungkan penjelasan ke konsep StatQuest yang relevan (mis. "ini yang dibahas di video Confusion Matrix").
5. Setelah tiap milestone, ajukan **1–2 pertanyaan cek pemahaman**.
6. Bahasa: Indonesia santai. Komentar kode boleh Bahasa Indonesia.
7. Kalau pemilik bilang **"mode cepat"**, aturan 1–5 boleh dilonggarkan untuk sesi itu.

## Tech stack

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib` (opsional `seaborn`), `jupyter`, `joblib`
- Opsional untuk teks Bahasa Indonesia: daftar stopword Indonesia (mis. dari NLTK atau library Sastrawi). Putuskan setelah dataset dipilih.
- **Tanpa** deep learning, LLM, atau API eksternal di versi ini.

## Struktur folder target

```
fake-news-v1/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore              # data/raw, data/processed, .venv, models/*.joblib
├── data/
│   ├── raw/                # dataset asli (TIDAK di-commit)
│   └── processed/          # train.csv, val.csv, test.csv (test DIBEKUKAN)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_error_analysis.ipynb
├── src/
│   ├── preprocess.py       # pembersihan teks
│   ├── split.py            # dedupe + split stratified
│   ├── train.py            # pipeline TF-IDF + model
│   ├── evaluate.py         # metrik + plot
│   ├── logreg_scratch.py   # BONUS: logistic regression dari nol (NumPy)
│   └── predict.py          # prediksi 1 teks dari CLI
├── models/                 # model tersimpan (joblib)
└── reports/
    ├── figures/            # confusion matrix, ROC curve
    └── results.md          # tabel hasil semua eksperimen
```

## Aturan teknis (anti-bocor & reproducible)

- **Random seed = 42** di semua tempat (split, model, CV).
- **Hapus duplikat SEBELUM split.** Berita yang sama (atau hampir sama) tidak boleh ada di train dan test sekaligus.
- **Split stratified:** train 70% / val 15% / test 15%. Simpan ke `data/processed/`.
- **Test set dibekukan.** Tidak boleh dipakai untuk tuning atau eksplorasi. Hanya dipakai **sekali** di evaluasi akhir. Semua keputusan pakai val set atau cross-validation di train.
- **TF-IDF hanya di-fit di data train.** Selalu pakai `sklearn.pipeline.Pipeline` biar transform-nya otomatis benar dan tidak bocor.
- Setiap eksperimen dicatat di `reports/results.md` (model, parameter, metrik di val).

## Milestones

Centang `[x]` kalau selesai. Claude Code: perbarui checklist ini saat pemilik menyelesaikan milestone.

### M0: Setup
- [x] Virtual environment, `requirements.txt`, `.gitignore`, struktur folder, `git init`

### M1: Pilih dataset + EDA
- [x] Pilih dataset. Kandidat: dataset hoaks berbahasa Indonesia dari Kaggle (cari "indonesian hoax" / "berita hoaks"), atau dataset Inggris seperti LIAR / ISOT.
- [ ] Cek dan catat di README: jumlah data, distribusi label (seimbang atau tidak?), bahasa, kolom apa saja, sumber, lisensi _(sisa: link + lisensi Kaggle)_
- [x] EDA: distribusi panjang teks per label, jumlah duplikat, kata yang paling sering per label
- [x] **Cari "jejak sumber"**: string yang muncul hampir hanya di satu label (nama media, "(Reuters)", tag "[HOAKS]", pola URL, format tanggal). Ini calon kebocoran (lihat M7).

### M2: Split data
- [x] `src/split.py`: dedupe → stratified split 70/15/15 → simpan CSV
- [x] Verifikasi: distribusi label mirip di ketiga set, tidak ada teks yang sama di train & test

### M3: Preprocessing
- [x] `src/preprocess.py`: lowercase, buang URL, angka, tanda baca, spasi berlebih
- [x] Putuskan: pakai stopword removal atau tidak? (Coba dua-duanya nanti di M6, jangan diasumsikan) _(keputusan: tidak dibuang di preprocess; diuji on/off di M6 lewat `TfidfVectorizer(stop_words=...)`)_
- [x] Buang token "jejak sumber" yang ditemukan di M1 (atau simpan sebagai opsi on/off untuk eksperimen M7) _(opsi `buang_jejak=True/False`)_

### M4: Baseline
- [x] **Dummy baseline** (`DummyClassifier`, tebak kelas terbanyak). Patokan minimum.
- [x] **TF-IDF + Multinomial Naive Bayes**
- [x] **TF-IDF + Logistic Regression**
- [x] Catat semua di `reports/results.md`

### M5: Evaluasi yang benar
- [x] Confusion matrix (simpan gambarnya)
- [x] Precision, recall, F1 per kelas + ROC-AUC + ROC curve
- [x] **Diskusikan di README:** mana yang lebih berbahaya, hoaks lolos (false negative) atau berita valid dituduh hoaks (false positive)? Metrik mana yang jadi prioritas?

### M6: Tuning dengan cross-validation
- [ ] `GridSearchCV` (5-fold stratified, **di train saja**) untuk: `C` (kekuatan regularisasi), `ngram_range` (1,1) vs (1,2), `min_df`, `max_features`, stopword on/off
- [ ] Bandingkan skor CV vs skor val. Kalau beda jauh, curigai overfitting (StatQuest: Bias & Variance, Regularization)

### M7: Audit "model nyontek" ⭐ (bagian paling penting)
- [ ] Tampilkan **20 kata dengan koefisien paling positif & paling negatif** dari Logistic Regression
- [ ] Apakah kata-kata itu masuk akal secara isi, atau malah nama sumber, format, atau tag? Catat temuannya.
- [ ] Eksperimen: latih ulang **dengan dan tanpa** token "jejak sumber". Seberapa turun skornya? Tulis di README.
- [ ] Error analysis: ambil 20 prediksi salah dari val set, kelompokkan polanya

### M8: Evaluasi akhir
- [ ] Pilih model terbaik **berdasarkan val/CV**, latih ulang di train+val
- [ ] Evaluasi **sekali** di test set. Simpan semua metrik ke `reports/results.md` sebagai **"Hasil final v1"**. Angka ini yang nanti dibandingkan dengan v2 (LLM).
- [ ] Simpan model ke `models/` dengan `joblib`

### M9: BONUS, Logistic Regression dari nol
- [ ] `src/logreg_scratch.py` pakai NumPy: sigmoid, binary cross-entropy loss, gradient, update gradient descent, plus opsi regularisasi L2
- [ ] Pakai fitur TF-IDF yang sama (batasi `max_features` mis. 5000 biar ringan; matriks sparse dari scikit-learn bisa langsung dikali `@`)
- [ ] Plot loss per epoch
- [ ] Bandingkan akurasi & koefisien dengan `LogisticRegression` scikit-learn. Seberapa mirip? Kenapa bisa beda?

### M10: Finishing
- [ ] `src/predict.py`: `python src/predict.py "teks berita..."` → label + probabilitas
- [ ] README lengkap (lihat template di bawah)
- [ ] Push ke GitHub (cek `git status`: data mentah dan file model besar tidak ikut)

## Template README (isi di akhir)

1. **Masalah**: apa yang diklasifikasi dan kenapa penting
2. **Dataset**: sumber, ukuran, distribusi label, keterbatasan
3. **Pendekatan**: preprocessing, TF-IDF, model, alasan pemilihan
4. **Hasil**: tabel dummy vs NB vs LogReg vs scratch (val & test)
5. **Audit kebocoran**: temuan M7 ⭐
6. **Error analysis**: pola kesalahan model
7. **Keterbatasan**: model ini mendeteksi *gaya bahasa*, bukan *kebenaran fakta*
8. **Next step**: v2 dengan LLM + RAG (Fase 3)
9. **Cara menjalankan**

## Definition of done

- [ ] Semua milestone M0–M8 selesai (M9 bonus)
- [ ] Test set hanya dipakai sekali, dan hasilnya tercatat
- [ ] Ada analisis kebocoran/shortcut yang jujur di README
- [ ] Pemilik bisa menjelaskan tanpa contekan: kenapa TF-IDF di-fit di train saja, arti tiap angka di confusion matrix, dan apa yang dilakukan parameter `C`

## Catatan progres

> Claude Code: tambahkan catatan singkat di sini tiap akhir sesi (tanggal, apa yang selesai, keputusan penting, langkah berikutnya).

- **2026-10-31**: M0 selesai. Root project = folder ini (bukan subfolder `fake-news-v1/`). `.venv` Python 3.10.5; terpasang pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, matplotlib 3.10.9. `git init` di branch `main`, belum ada commit. Berikutnya: M1 (pilih dataset + EDA).
- **2026-11-01**: M1 (sebagian). Dataset Komdigi (`data/raw/komdigi/`) ditolak karena isinya 100% hoaks (cuma 1 kelas); disimpan sebagai calon knowledge base RAG v2. Dataset yang dipakai: `data/raw/hoax_id/Cleaned/` (CNN, Kompas, Tempo = valid; Turnbackhoax = hoaks). Keputusan: teks hoaks = `Narasi` dipotong di pembatas `===`/`Penjelasan:` (bukan `FullText`, karena itu artikel cek fakta); hanya judul berawalan SALAH/HOAX/DISINFORMASI/dll. (buang [BENAR], [KLARIFIKASI], dll.); buang teks < 3 kata. Hasil: 27.066 baris (77,4% valid / 22,6% hoaks) → `data/processed/dataset_gabungan.csv` (belum di-dedupe). Notebook `notebooks/01_eda.ipynb` dibuat (ditulis Claude atas permintaan pemilik). Sisa M1: pemilik isi ringkasan temuan di notebook + info dataset di README.
- **2026-11-01**: Ringkasan temuan notebook (bagian 11) + bagian Dataset README diisi Claude atas permintaan pemilik; sisa M1 = link & lisensi Kaggle di README. M2 selesai: `src/split.py` (ditulis Claude atas permintaan) → dedupe 14 baris → 27.052 baris diberi kolom `id` permanen → train 18.936 / val 4.058 / test 4.058, semua 22,6% hoaks, 0 overlap. Ada pengaman: script menolak menimpa `test.csv` kecuali `--force`. **Test set sudah dibekukan, jangan di-split ulang.** Berikutnya: M3 (preprocessing).
- **2026-11-02**: M3 selesai. `src/preprocess.py` (ditulis Claude atas permintaan): `bersihkan_teks(teks, buang_jejak=True)`. Jejak yang dibuang: dateline & nama media (CNN/Kompas/Tempo), ADVERTISEMENT/SCROLL TO RESUME CONTENT, `[Gambas:...]`, penanda "Baca/Lihat Juga" (judul ikut dibuang hanya kalau di baris sendiri, karena di CNN judul nempel ke kalimat berikut; versi pertama regex sempat memakan ~100 kata/artikel CNN), inisial wartawan `(abc/def)`, "Penjelasan & Fakta:", catatan terjemahan pemeriksa fakta. Pembersihan umum: lowercase, `[dot]`→`.`, URL, angka, apostrof (tanpa spasi), non-huruf, spasi. Kecepatan ~6 ms/dok → di M4+ bersihkan sekali lalu pakai ulang (aman karena stateless). Sisa shortcut yang tidak bisa dibuang: panjang teks (median valid ~300 vs hoaks ~29 kata), gaya informal, topik politik. Berikutnya: M4 (baseline).
- **2026-11-02**: M4 selesai. `src/train.py` (ditulis Claude atas permintaan): 3 Pipeline (TF-IDF default → Dummy / MultinomialNB(alpha=1) / LogReg(C=1, max_iter=1000)), latih di train, nilai di val; `--catat` menambah baris ke `reports/results.md`. Tambahan di `preprocess.py`: `muat_split_bersih(nama, buang_jejak)` dengan cache pickle di `data/processed/cache/` (nama file memuat hash md5 preprocess.py → otomatis invalid kalau aturan berubah). Hasil val: Dummy acc 0.774; NB acc 0.852, recall 0.346, AUC 0.997; LogReg acc 0.989, F1 0.975, AUC 0.998 (acc train 0.995). NB: recall rendah merata di semua panjang teks, median proba hoaks ~0.3 → masalah ambang/kalibrasi, bukan ranking; dugaan: alpha=1 terlalu besar dibanding "massa" TF-IDF kelas hoaks (78.700 kosakata) → uji alpha kecil di M6. Skor LogReg 98,9% walau jejak sudah dibuang → dicurigai shortcut gaya/panjang (M7). Berikutnya: M5 (evaluasi: confusion matrix, ROC).
- **2026-11-03**: M5 selesai. `src/evaluate.py` (ditulis Claude atas permintaan): latih ulang 3 baseline, cetak TN/FP/FN/TP + `classification_report` per kelas + ROC-AUC, simpan `reports/figures/confusion_matrix_val.png` (3 panel, warna = % per baris) dan `roc_curve_val.png` (utuh + zoom FPR 0–5%, titik kerja ambang 0,5). LogReg di val: FP 13, FN 32. README: bagian "Evaluasi di val set" + diskusi FN vs FP. **Keputusan: skenario "alat bantu penyaring untuk pemeriksa fakta" → prioritas recall hoaks, precision dijaga ≥ ~0,90; F1 sebagai pembanding.** Draf diskusi ditulis Claude, pemilik diminta merevisi dengan kata-katanya sendiri. Berikutnya: M6 (GridSearchCV).
