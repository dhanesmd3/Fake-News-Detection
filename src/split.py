"""M2: dedupe -> split stratified 70/15/15 -> simpan train/val/test.csv

Cara menjalankan (dari root project, venv aktif):
    python src/split.py           # split pertama kali
    python src/split.py --force   # paksa bikin ulang (HATI-HATI: test set ikut berubah)
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Path(__file__) = lokasi file ini (src/split.py); parents[1] = naik 1 folder ke root project.
# Jadi script ini tetap jalan benar walaupun dipanggil dari folder mana pun.
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "dataset_gabungan.csv"
OUT_DIR = ROOT / "data" / "processed"


def normalisasi(teks: pd.Series) -> pd.Series:
    """Versi teks untuk MENCARI duplikat (bukan untuk model)."""
    # astype(str)    -> jaga-jaga kalau ada nilai yang bukan teks
    # str.lower()    -> huruf kecil semua, jadi "Jokowi" dan "JOKOWI" dianggap sama
    # str.split()    -> pecah jadi daftar kata; spasi/enter/tab sebanyak apa pun dianggap 1 pemisah
    # str.join(" ")  -> gabung lagi pakai tepat 1 spasi
    return teks.astype(str).str.lower().str.split().str.join(" ")


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris yang teksnya duplikat, sisakan 1 salinan."""
    kunci = normalisasi(df["teks"])

    # duplicated() -> True untuk salinan ke-2, ke-3, dst.
    # Salinan PERTAMA tetap False, jadi tetap disimpan (nggak semua kembaran dibuang).
    is_dup = kunci.duplicated()
    print(f"Duplikat dibuang: {is_dup.sum()} baris")

    # ~is_dup -> kebalikannya (True jadi False), artinya "ambil baris yang BUKAN duplikat"
    # reset_index(drop=True) -> nomor baris diurutkan ulang 0, 1, 2, ... (nomor lama dibuang)
    return df[~is_dup].reset_index(drop=True)


def split_stratified(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi df jadi train 70% / val 15% / test 15%, proporsi label sama di ketiganya."""
    # train_test_split cuma bisa membelah jadi DUA, jadi kita belah dua kali.
    #
    # Tahap 1: 70% train, 30% sisa
    #   test_size=0.30          -> 30% data masuk bagian kedua ("sisa")
    #   stratify=df["label"]    -> proporsi hoaks:valid di kedua bagian dijaga sama
    #   random_state            -> acakannya selalu sama tiap dijalankan (reproducible)
    train, sisa = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_STATE
    )

    # Tahap 2: sisa (30% dari total) dibelah dua sama besar -> 15% val + 15% test
    #   test_size=0.50 karena 15% dari total = setengah dari 30%
    #   stratify pakai label milik "sisa" (bukan df), karena yang dibelah sekarang "sisa"
    val, test = train_test_split(
        sisa, test_size=0.50, stratify=sisa["label"], random_state=RANDOM_STATE
    )
    return train, val, test


def cek_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Verifikasi hasil split. Kalau ada yang salah, program BERHENTI (pakai assert)."""
    sets = {"train": train, "val": val, "test": test}
    total = sum(len(d) for d in sets.values())

    # --- Cek 1: ukuran & proporsi hoaks tiap set ---
    # Trik: label isinya 0/1, jadi rata-ratanya = proporsi angka 1 (= proporsi hoaks).
    # Contoh: [1, 0, 0, 1] -> rata-rata 0.5 -> 50% hoaks.
    print(f"\n{'set':6s} {'jumlah':>7s} {'% total':>8s} {'% hoaks':>8s}")
    for nama, d in sets.items():
        print(f"{nama:6s} {len(d):7d} {len(d) / total * 100:7.1f}% {d['label'].mean() * 100:7.1f}%")

    # assert kondisi, pesan -> kalau kondisinya False, program langsung error dengan pesan itu.
    # Di sini: selisih proporsi hoaks terbesar vs terkecil harus < 1 poin persen.
    proporsi = [d["label"].mean() for d in sets.values()]
    assert max(proporsi) - min(proporsi) < 0.01, "Proporsi label antar set terlalu beda!"

    # --- Cek 2: nggak boleh ada teks yang sama di dua set berbeda (anti-bocor) ---
    # set() = kumpulan nilai unik. Operator & = irisan, yaitu nilai yang ada di KEDUA set.
    teks_unik = {nama: set(normalisasi(d["teks"])) for nama, d in sets.items()}
    print()
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        bocor = teks_unik[a] & teks_unik[b]
        print(f"Teks sama di {a} & {b}: {len(bocor)}")
        assert len(bocor) == 0, f"BOCOR! {len(bocor)} teks muncul di {a} dan {b}"

    print("\nSemua cek lolos.")


def main() -> None:
    # --- Pengaman: test set itu DIBEKUKAN ---
    # Kalau test.csv sudah ada, jangan ditimpa diam-diam. Kalau dataset_gabungan.csv berubah
    # (misal setelah M3), split ulang akan menghasilkan test set yang BEDA, dan hasil
    # evaluasi lama jadi nggak bisa dibandingkan lagi (termasuk dengan v2 nanti).
    # sys.argv = daftar kata yang diketik setelah "python", mis. ["src/split.py", "--force"]
    if (OUT_DIR / "test.csv").exists() and "--force" not in sys.argv:
        print("test.csv sudah ada. Test set DIBEKUKAN, jadi split tidak diulang.")
        print("Kalau yakin mau bikin ulang: python src/split.py --force")
        return

    df = pd.read_csv(INPUT_PATH)
    print("Data awal:", df.shape)

    df = dedupe(df)

    # Kasih nomor ID permanen ke tiap berita SEBELUM diacak.
    # Gunanya: nanti di error analysis (M7) & perbandingan v2, kita bisa nunjuk "berita id 1234"
    # dan semua orang tahu berita yang mana.
    df.insert(0, "id", range(len(df)))

    train, val, test = split_stratified(df)
    cek_split(train, val, test)

    # Simpan ke CSV. index=False -> nomor baris pandas nggak ikut disimpan (kita sudah punya kolom id)
    print()
    for nama, d in [("train", train), ("val", val), ("test", test)]:
        path = OUT_DIR / f"{nama}.csv"
        d.to_csv(path, index=False)
        print(f"Tersimpan: {path.name:10s} {d.shape}")


# Blok ini cuma jalan kalau file dipanggil langsung (python src/split.py),
# bukan kalau file ini di-import dari file lain.
if __name__ == "__main__":
    main()
