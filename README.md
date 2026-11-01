# Fake News Detection v1

Klasifikasi berita **hoaks vs valid** pakai ML klasik (TF-IDF + Logistic Regression, dengan Naive Bayes sebagai pembanding).

> 🚧 Masih dikerjakan. README lengkap diisi di M10.

## Dataset

- **Sumber:** Kaggle, _TODO: link halaman dataset_
- **Lisensi:** _TODO: cek di halaman Kaggle_
- **Bahasa:** Indonesia
- **Isi:** 4 file hasil scraping (folder `Cleaned/`)

| Sumber | Label | Kolom teks yang dipakai | Jumlah (setelah dibersihkan) |
|---|---|---|---|
| CNN Indonesia | valid (0) | `FullText` | 9.630 |
| Kompas | valid (0) | `FullText` | 4.723 |
| Tempo | valid (0) | `FullText` | 6.592 |
| Turnbackhoax | hoaks (1) | `Narasi` | 6.121 |
| **Total** | | | **27.066** (77,4% valid / 22,6% hoaks) |

**Pembersihan yang dilakukan** (detail di `notebooks/01_eda.ipynb`):
- Hoaks pakai kolom `Narasi` (isi hoaks yang beredar), bukan `FullText` (artikel bantahan pemeriksa fakta). `Narasi` dipotong di pembatas `===` / `Penjelasan:` supaya penjelasan pemeriksa fakta tidak ikut.
- Hanya artikel Turnbackhoax berjudul `[SALAH]`, `[HOAX]`, `[DISINFORMASI]`, dsb. Yang berjudul `[BENAR]`, `[KLARIFIKASI]`, `[BERITA]`, dll. dibuang (1.158 baris) karena bukan hoaks.
- Teks kosong / kurang dari 3 kata dibuang.

**Keterbatasan:** label ditentukan dari **sumber** (portal berita = valid, unggahan medsos yang dicek Turnbackhoax = hoaks). Akibatnya ada banyak "jejak sumber": dateline media, sisa scraping, perbedaan panjang teks (median 312 vs 28 kata), dan gaya bahasa formal vs informal. Lihat bagian Audit kebocoran (M7).

## Cara menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
