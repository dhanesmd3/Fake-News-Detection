"""M3: pembersihan teks.

Fungsi utama: bersihkan_teks(teks, buang_jejak=True)

File CSV di data/processed/ TIDAK diubah. Teks asli tetap utuh, dibersihkan waktu dipakai.

Catatan kecepatan: fungsi ini butuh ~6 ms per dokumen (teksnya panjang-panjang).
Jadi di M4-M7, bersihkan SEKALI di awal lalu pakai ulang hasilnya, misalnya:
    train["teks_bersih"] = train["teks"].apply(bersihkan_teks)
Ini aman (nggak bocor), karena bersihkan_teks cuma mengolah 1 teks sendirian dan
nggak "belajar" apa pun dari data lain. Beda dengan TF-IDF yang WAJIB di-fit di train saja.

Cara menjalankan demo + audit (pakai train.csv saja):
    python src/preprocess.py
"""
import re
from pathlib import Path

import pandas as pd

# =====================================================================
# 1. POLA "JEJAK SUMBER"
# =====================================================================
# Pola = potongan teks yang muncul karena ASAL datanya (media / situs pemeriksa fakta),
# bukan karena isinya hoaks atau valid. Daftar ini berasal dari temuan EDA (M1).
#
# PENTING: semua pola diterapkan ke SEMUA teks, tanpa melihat label.
# Kenapa? Waktu model dipakai beneran (predict.py), kita nggak tahu labelnya.
# Pembersihan yang "ngintip" label = curang.
#
# re.compile(...) = "menerjemahkan" pola regex sekali di awal, biar cepat dipakai berulang
# (fungsi ini akan dipanggil puluhan ribu kali).
# re.IGNORECASE = huruf besar/kecil dianggap sama.
# re.DOTALL     = titik (.) juga boleh mencocokkan baris baru (\n).

POLA_JEJAK = [
    # --- Dari portal berita (valid) ---
    # Dateline di awal teks: "Jakarta, CNN Indonesia --", "JAKARTA, KOMPAS.com -", "TEMPO.CO, Jakarta -"
    #   ^\s*                       = di awal teks (boleh ada spasi dulu)
    #   (?:[A-Za-z .]{1,30},\s*)?  = nama kota + koma (opsional), mis. "Jakarta, "
    #   (?:CNN Indonesia|...)      = nama medianya
    #   (?:,\s*[A-Za-z .]{1,30})?  = koma + nama kota setelah media (opsional), mis. ", Jakarta"
    #   \s*-+                      = diakhiri tanda strip "-" atau "--"
    re.compile(
        r"^\s*(?:[A-Za-z .]{1,30},\s*)?(?:CNN Indonesia|KOMPAS\.com|TEMPO\.CO)(?:,\s*[A-Za-z .]{1,30})?\s*-+",
        re.IGNORECASE,
    ),
    # Sisa tampilan halaman CNN: tulisan iklan & penanda video
    re.compile(r"ADVERTISEMENT|SCROLL TO RESUME CONTENT"),
    re.compile(r"\[Gambas:[^\]]*\]"),  # mis. "[Gambas:Video CNN]"
    # Tautan ke artikel lain: "Baca Juga:", "Lihat Juga :", "Baca:", "Selengkapnya baca juga:".
    # (a) Format Tempo: penanda, lalu judul di BARIS SENDIRI -> penanda + judulnya dibuang.
    #     [ \t]* = spasi/tab saja (bukan enter), \n = wajib ganti baris, [^\n]* = 1 baris judul
    re.compile(r"(?:Baca|Lihat)(?:\s+juga)?[ \t]*:[ \t]*\n[^\n]*", re.IGNORECASE),
    # (b) Format CNN/Kompas: judul NEMPEL ke kalimat berikutnya ("...IKNSebagai informasi...").
    #     Batas judulnya nggak bisa ditebak, jadi cuma PENANDA-nya yang dibuang. Judulnya dibiarkan:
    #     lebih baik ada sedikit noise daripada ikut memotong isi berita.
    #     Wajib ada ":" supaya kalimat biasa seperti "saya baca dulu" nggak ikut kena.
    re.compile(r"\b(?:Selengkapnya\s+)?(?:Baca|Lihat)(?:\s+juga)?\s*:", re.IGNORECASE),
    # Inisial penulis/editor CNN di akhir artikel, mis. "(yoa/isn)"
    re.compile(r"\([a-z]{2,4}/[a-z]{2,4}\)"),
    # Nama media di mana pun dalam teks (mis. "ujarnya kepada CNNIndonesia.com")
    # Catatan: kata biasa "tempo" (mis. "dalam tempo singkat") ikut terbuang. Harga kecil, diterima.
    #   \b = batas kata, jadi "tempo" di dalam "sementempo" (misalnya) nggak ikut kena
    #   (?:\.com|\.co)? = boleh diikuti ".com" atau ".co" (mis. "KOMPAS.com", "TEMPO.CO")
    re.compile(r"\b(?:CNN\s*Indonesia|CNN|Kompas|Tempo)(?:\.com|\.co)?\b", re.IGNORECASE),

    # --- Dari pemeriksa fakta (hoaks) ---
    # Format lama "Penjelasan & Fakta :" yang lolos dari pemotongan di M1 -> buang sampai akhir teks
    re.compile(r"\bPenjelasan\s*(?:&\s*Fakta)?\s*:.*", re.IGNORECASE | re.DOTALL),
    # Catatan terjemahan yang ditambahkan pemeriksa fakta, bukan bagian dari hoaksnya.
    # Kurung "( )" / "[ ]" dan titik dua ":" di sekitarnya nggak perlu ditulis di pola,
    # karena nanti ikut terbuang di langkah "buang tanda baca". Pola yang pendek = lebih cepat.
    # "(Narasi diterjemahkan ke dalam bahasa Indonesia)", "[Diterjemahkan dalam Bahasa Indonesia]",
    # "(Diterjemahkan ke Bahasa Indonesia)"  ->  kata "ke" dan "dalam" sama-sama opsional
    re.compile(r"\b(?:narasi\s+)?diterjemahkan\s+(?:ke\s+)?(?:dalam\s+)?bahasa\s+indonesia", re.IGNORECASE),
    # "Terjemahan:", "(terjemahan)", "Terjemahan Narasi:"
    re.compile(r"\bterjemahan(?:\s+narasi)?\b", re.IGNORECASE),
    # "NARASI DALAM GAMBAR:"
    re.compile(r"\bnarasi dalam gambar\b", re.IGNORECASE),
]

# =====================================================================
# 2. POLA PEMBERSIHAN UMUM (berlaku untuk teks apa pun)
# =====================================================================
# Link yang sengaja "disamarkan" pemeriksa fakta: "bit[dot]ly" -> "bit.ly", biar ketangkap pola URL
POLA_DOT = re.compile(r"\[dot\]|\(dot\)")
# URL: yang diawali http(s):// atau www., ATAU yang berakhiran domain umum (mis. "bit.ly/xyz", "detik.com")
#   \S+ = semua karakter sampai ketemu spasi
POLA_URL = re.compile(
    r"https?\s*:\s*//\S+|www\.\S+|\b[\w.-]+\.(?:com|net|org|id|ly|me|vip|info)\b(?:/\S*)?",
    re.IGNORECASE,
)
POLA_APOSTROF = re.compile(r"['’‘`]")    # apostrof dihapus TANPA spasi: "syari'ah" -> "syariah"
POLA_ANGKA = re.compile(r"\d+")           # satu angka atau lebih
POLA_BUKAN_HURUF = re.compile(r"[^a-z\s]")  # apa pun selain huruf a-z & spasi (tanda baca, emoji, “ ”)
POLA_SPASI = re.compile(r"\s+")           # spasi/enter/tab beruntun


def hapus_jejak_sumber(teks: str) -> str:
    """Ganti setiap pola jejak sumber dengan spasi."""
    for pola in POLA_JEJAK:
        # pola.sub(pengganti, teks) = cari semua yang cocok dengan pola, ganti dengan pengganti
        teks = pola.sub(" ", teks)
    return teks


def bersihkan_teks(teks: str, buang_jejak: bool = True) -> str:
    """Bersihkan 1 teks: (opsional) buang jejak sumber, lalu lowercase, URL, angka, tanda baca, spasi.

    buang_jejak=False dipakai di eksperimen M7, untuk membandingkan skor model
    DENGAN vs TANPA jejak sumber.
    """
    teks = str(teks)  # jaga-jaga kalau isinya bukan string (mis. NaN)

    # Langkah 1: jejak sumber dibuang PALING AWAL, karena polanya butuh huruf besar & baris baru asli
    # (mis. "ADVERTISEMENT", atau "Baca Juga:" yang judulnya ada di baris berikutnya)
    if buang_jejak:
        teks = hapus_jejak_sumber(teks)

    teks = teks.lower()                      # 2. huruf kecil semua: "Jokowi" = "jokowi"
    teks = POLA_DOT.sub(".", teks)           # 3a. "[dot]" -> "."
    teks = POLA_URL.sub(" ", teks)           # 3b. buang URL
    teks = POLA_ANGKA.sub(" ", teks)         # 4. buang angka
    teks = POLA_APOSTROF.sub("", teks)       # 5a. apostrof dihapus (bukan jadi spasi), biar kata nggak terbelah
    teks = POLA_BUKAN_HURUF.sub(" ", teks)   # 5b. buang tanda baca, emoji, simbol (sisa huruf & spasi)
    teks = POLA_SPASI.sub(" ", teks).strip() # 6. spasi beruntun jadi 1, buang spasi di ujung
    return teks


# =====================================================================
# 3. STOPWORD: SENGAJA TIDAK DIBUANG DI SINI
# =====================================================================
# Keputusan M3: stopword ("yang", "dan", "di", ...) TIDAK dibuang di preprocessing.
# Nanti di M6 dicoba dua-duanya lewat parameter TfidfVectorizer(stop_words=...),
# lalu dipilih berdasarkan skor cross-validation, bukan berdasarkan feeling.


# =====================================================================
# 4. DEMO & AUDIT (hanya jalan kalau file ini dipanggil langsung)
# =====================================================================
TOKEN_JEJAK = [
    "cnn", "kompas", "tempo", "advertisement", "scroll", "resume", "gambas",
    "baca", "lihat", "terjemahan", "diterjemahkan", "penjelasan", "www", "http", "permalink", "fafhh",
]


def _persen_mengandung(teks_bersih: pd.Series, label: pd.Series) -> pd.DataFrame:
    """Untuk tiap token jejak: berapa % dokumen per label yang masih mengandungnya."""
    # Pecah tiap dokumen jadi set kata, biar cek "kata ada atau tidak" cepat
    kata_per_dok = teks_bersih.str.split().apply(set)
    hasil = {}
    for nilai, nama in [(0, "valid"), (1, "hoaks")]:
        subset = kata_per_dok[label == nilai]
        hasil[nama] = {t: round(subset.apply(lambda s: t in s).mean() * 100, 1) for t in TOKEN_JEJAK}
    return pd.DataFrame(hasil)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # HANYA train.csv. Test set dibekukan, jangan dipakai untuk eksplorasi.
    df = pd.read_csv(root / "data" / "processed" / "train.csv")

    print("=== Contoh sebelum vs sesudah ===")
    for label in [0, 1]:
        teks = df.loc[df["label"] == label, "teks"].iloc[0]
        print(f"\n[label {label}] SEBELUM:\n{teks[:300]!r}")
        print(f"[label {label}] SESUDAH:\n{bersihkan_teks(teks)[:300]!r}")

    # Audit pakai sampel 5.000 dokumen biar cepat (~1 menit), hasilnya sudah cukup mewakili
    sampel = df.sample(5000, random_state=42)
    print(f"\n(audit pakai sampel {len(sampel)} dokumen dari train)")

    # Bersihkan 2 versi SEKALI saja, lalu hasilnya dipakai ulang di bawah
    # .apply(fungsi, argumen) = jalankan fungsi ke setiap baris
    tanpa_buang = sampel["teks"].apply(bersihkan_teks, buang_jejak=False)
    dengan_buang = sampel["teks"].apply(bersihkan_teks, buang_jejak=True)

    print("\n=== Audit: % dokumen yang masih mengandung token jejak ===")
    sebelum = _persen_mengandung(tanpa_buang, sampel["label"]).add_suffix(" (sebelum)")
    sesudah = _persen_mengandung(dengan_buang, sampel["label"]).add_suffix(" (sesudah)")
    # pd.concat(axis=1) = tempel tabel ke samping; lalu urutkan kolom biar enak dibandingkan
    tabel = pd.concat([sebelum, sesudah], axis=1)
    print(tabel[["valid (sebelum)", "valid (sesudah)", "hoaks (sebelum)", "hoaks (sesudah)"]].to_string())

    print("\n=== Median jumlah kata: mentah vs bersih ===")
    print("(buang jejak harusnya cuma memotong sedikit kata; kalau banyak, ada pola yang kebablasan)")
    panjang = pd.DataFrame({
        "mentah": sampel["teks"].str.split().str.len(),
        "bersih (tanpa buang jejak)": tanpa_buang.str.split().str.len(),
        "bersih (buang jejak)": dengan_buang.str.split().str.len(),
    })
    print(panjang.groupby(sampel["sumber"]).median().to_string())
    print("\nPerbedaan panjang hoaks vs valid TIDAK bisa dihilangkan lewat preprocessing -> catat untuk M7.")


if __name__ == "__main__":
    main()
