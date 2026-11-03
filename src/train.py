"""M4: baseline. Dummy vs TF-IDF + Naive Bayes vs TF-IDF + Logistic Regression.

Model dilatih di TRAIN, dinilai di VAL. Test set TIDAK disentuh (itu jatah M8).

Cara menjalankan (dari root project, venv aktif):
    python src/train.py           # latih + tampilkan hasil
    python src/train.py --catat   # sama, plus tambahkan hasilnya ke reports/results.md
"""
import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocess import muat_split_bersih

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / "results.md"


def buat_model() -> dict[str, tuple[Pipeline, str]]:
    """Bikin 3 model baseline. Return: {nama_model: (pipeline, deskripsi_parameter)}.

    Semua model dibungkus Pipeline: [TF-IDF] -> [classifier].
    Pipeline menjamin TF-IDF cuma "belajar" (fit) dari data train. Waktu dipakai ke val,
    pipeline otomatis cuma melakukan transform (pakai kosakata & bobot hasil belajar dari train).
    """
    return {
        # 1. Dummy: SELALU menebak kelas terbanyak (valid). Nggak baca teks sama sekali.
        #    Gunanya cuma jadi patokan minimum: model lain WAJIB lebih bagus dari ini.
        #    (TF-IDF tetap dipasang biar bentuk semua pipeline seragam; Dummy mengabaikannya.)
        "Dummy": (
            Pipeline([("tfidf", TfidfVectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))]),
            "strategy=most_frequent",
        ),
        # 2. Naive Bayes: hitung peluang tiap kata muncul di hoaks vs valid, lalu kalikan semuanya.
        #    "Naive" karena menganggap tiap kata saling bebas (padahal nggak). Cepat & sering ampuh untuk teks.
        #    alpha=1.0 = smoothing, biar kata yang nggak pernah muncul di satu kelas nggak bikin peluangnya 0.
        "TF-IDF + Naive Bayes": (
            Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB(alpha=1.0))]),
            "TF-IDF default; MultinomialNB(alpha=1.0)",
        ),
        # 3. Logistic Regression: tiap kata dapat 1 bobot (koefisien). Bobot dijumlahkan, lalu
        #    dimasukkan ke fungsi sigmoid -> peluang hoaks 0..1.
        #    C=1.0 = kekuatan regularisasi (default). Makin KECIL C, makin "direm" bobotnya.
        #    max_iter=1000 = batas putaran optimasi; default (100) kadang belum cukup untuk data teks.
        "TF-IDF + LogReg": (
            Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "TF-IDF default; LogisticRegression(C=1.0)",
        ),
    }


def evaluasi(model: Pipeline, X: pd.Series, y: pd.Series) -> dict[str, float]:
    """Hitung metrik dasar. Precision/recall/F1 dihitung untuk kelas HOAKS (label 1)."""
    pred = model.predict(X)               # tebakan label: 0 atau 1
    proba = model.predict_proba(X)[:, 1]  # peluang hoaks (kolom ke-1), angka 0..1
    return {
        # Dari semua tebakan, berapa persen yang benar
        "accuracy": accuracy_score(y, pred),
        # Dari yang DITEBAK hoaks, berapa persen yang beneran hoaks
        # zero_division=0: kalau model nggak pernah nebak hoaks (Dummy), precision dianggap 0, bukan error
        "precision": precision_score(y, pred, zero_division=0),
        # Dari yang BENERAN hoaks, berapa persen yang berhasil ketangkap
        "recall": recall_score(y, pred),
        # Rata-rata "harmonis" precision & recall: tinggi cuma kalau DUA-DUANYA tinggi
        "f1": f1_score(y, pred, zero_division=0),
        # Seberapa bagus model MENGURUTKAN: hoaks dapat peluang lebih tinggi daripada valid?
        # 0.5 = nebak acak, 1.0 = sempurna. Nggak tergantung ambang 0.5.
        "roc_auc": roc_auc_score(y, proba),
    }


def catat_ke_results(hasil: list[dict]) -> None:
    """Tambahkan baris hasil ke tabel eksperimen di reports/results.md."""
    teks = RESULTS_PATH.read_text(encoding="utf-8")

    # partition = belah teks jadi 3: [bagian sebelum, pemisah, bagian sesudah].
    # Tabel eksperimen ada di bagian SEBELUM judul "## Hasil final".
    atas, pemisah, bawah = teks.partition("\n## Hasil final")
    baris = atas.rstrip("\n").split("\n")

    # Buang baris tabel kosong (placeholder "|   |   |...") kalau masih ada
    baris = [b for b in baris if not (b.startswith("|") and b.replace("|", "").strip() == "")]

    # Hitung sudah ada berapa eksperimen: baris tabel dikurangi 2 (header + garis pemisah)
    nomor = sum(1 for b in baris if b.startswith("|")) - 2

    tanggal = date.today().isoformat()
    for h in hasil:
        nomor += 1
        baris.append(
            f"| {nomor} | {tanggal} | {h['model']} | {h['parameter']} | {h['accuracy']:.4f} | "
            f"{h['precision']:.4f} | {h['recall']:.4f} | {h['f1']:.4f} | {h['roc_auc']:.4f} | {h['catatan']} |"
        )

    RESULTS_PATH.write_text("\n".join(baris) + "\n" + pemisah + bawah, encoding="utf-8")
    print(f"\n{len(hasil)} baris ditambahkan ke {RESULTS_PATH.relative_to(ROOT)}")


def main() -> None:
    # argparse = cara rapi membaca opsi dari terminal, mis. "--catat"
    parser = argparse.ArgumentParser(description="Latih & bandingkan model baseline (M4).")
    parser.add_argument("--catat", action="store_true", help="tambahkan hasil ke reports/results.md")
    args = parser.parse_args()

    # --- 1. Muat data (yang sudah dibersihkan; run pertama agak lama, berikutnya cepat) ---
    train = muat_split_bersih("train")
    val = muat_split_bersih("val")
    # X = input (teks bersih), y = jawaban (label). Konvensi nama ini dipakai di mana-mana di ML.
    X_train, y_train = train["teks_bersih"], train["label"]
    X_val, y_val = val["teks_bersih"], val["label"]
    print(f"Train: {len(train)} dok | Val: {len(val)} dok\n")

    # --- 2. Latih & nilai tiap model ---
    hasil = []
    for nama, (model, parameter) in buat_model().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)  # fit = BELAJAR. Cuma boleh pakai data train!
        durasi = time.perf_counter() - t0

        skor = evaluasi(model, X_val, y_val)
        # Akurasi di train juga dihitung, buat dibandingkan dengan val.
        # Kalau train jauh lebih tinggi dari val -> tanda overfitting (StatQuest: Bias & Variance)
        acc_train = accuracy_score(y_train, model.predict(X_train))

        hasil.append({
            "model": nama, "parameter": parameter, **skor,  # ** = "tuang" isi dict skor ke sini
            "acc_train": acc_train, "detik": durasi,
            "catatan": "baseline M4, buang_jejak=True",
        })
        print(f"[selesai] {nama:22s} ({durasi:.1f} detik)")

    # Ukuran kosakata hasil belajar TF-IDF (sama untuk semua model karena setting-nya sama)
    tfidf = buat_model()["TF-IDF + LogReg"][0].named_steps["tfidf"].fit(X_train)
    print(f"\nKosakata TF-IDF (dari train): {len(tfidf.vocabulary_):,} kata")

    # --- 3. Tampilkan tabel perbandingan ---
    kolom = ["accuracy", "precision", "recall", "f1", "roc_auc", "acc_train", "detik"]
    tabel = pd.DataFrame(hasil).set_index("model")[kolom]
    print("\n=== Hasil di VAL (precision/recall/F1 = kelas hoaks) ===")
    print(tabel.round(4).to_string())

    if args.catat:
        catat_ke_results(hasil)


if __name__ == "__main__":
    main()
