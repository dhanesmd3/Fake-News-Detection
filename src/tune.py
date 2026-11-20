"""M6: tuning dengan GridSearchCV (5-fold stratified, DI TRAIN SAJA), lalu cek ke val.

Yang dicari:
  - LogReg : C, ngram_range, min_df, max_features, stopword on/off
  - Naive Bayes : alpha (menguji dugaan M4: alpha=1 terlalu besar)

Cara menjalankan (dari root project, venv aktif):
    python src/tune.py --uji     # uji cepat (~2 menit): data & grid kecil, cuma cek kodenya jalan
    python src/tune.py           # beneran: ~30-60 menit di laptop 4 core
    python src/tune.py --catat   # sama, plus tambahkan model terbaik ke reports/results.md

Output:
    reports/tuning/cv_logreg.csv, cv_naive_bayes.csv  -> skor CV semua kombinasi
    reports/tuning/terbaik.json                        -> parameter terbaik (dipakai lagi di M7/M8)
"""
import argparse
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from preprocess import muat_split_bersih
from train import RANDOM_STATE, catat_ke_results, evaluasi

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "tuning"

# Skor yang dihitung di tiap fold. Model terbaik dipilih berdasarkan F1 (refit="f1"):
# F1 menyeimbangkan recall (prioritas kita, lihat M5) dengan precision, jadi model yang
# "asal tuduh semua hoaks" (recall 100%, precision jeblok) nggak akan menang.
SKOR = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc"}


def stopwords_indonesia() -> list[str]:
    """Daftar stopword Bahasa Indonesia dari Sastrawi (126 kata, mis. "yang", "dan", "di")."""
    kata = StopWordRemoverFactory().get_stop_words()
    # set() = buang duplikat (di Sastrawi ada yang dobel, mis. "jika").
    # Saring ke kata huruf a-z minimal 2 karakter, sama dengan aturan token TF-IDF,
    # biar scikit-learn nggak protes "stop words tidak konsisten dengan tokenisasi".
    return sorted(w for w in set(kata) if re.fullmatch(r"[a-z]{2,}", w))


def label_param(nilai) -> str:
    """Ubah nilai parameter jadi teks pendek yang enak dibaca di tabel."""
    if isinstance(nilai, list):   # daftar stopword -> cukup tulis "sastrawi"
        return "sastrawi"
    if nilai is None:
        return "-"
    return str(nilai)


def pipeline_terbaik(nama_model: str = "TF-IDF + LogReg") -> tuple[Pipeline, str]:
    """Bangun ulang pipeline dengan parameter terbaik hasil M6 (dari reports/tuning/terbaik.json).

    Dipakai di M7 & M8. Pipeline yang dikembalikan BELUM dilatih.
    Return: (pipeline, deskripsi_parameter)
    """
    params = json.loads((OUT_DIR / "terbaik.json").read_text(encoding="utf-8"))[nama_model]

    # JSON nggak kenal tuple & nggak menyimpan daftar stopword utuh, jadi kembalikan ke bentuk aslinya:
    # [1, 2] -> (1, 2), dan "sastrawi" -> daftar kata stopword
    if "tfidf__ngram_range" in params:
        params["tfidf__ngram_range"] = tuple(params["tfidf__ngram_range"])
    if params.get("tfidf__stop_words") == "sastrawi":
        params["tfidf__stop_words"] = stopwords_indonesia()

    if "LogReg" in nama_model:
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        clf = MultinomialNB()
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", clf)])
    # set_params = isi parameter pakai nama "<langkah>__<parameter>", sama persis dengan format grid
    pipe.set_params(**params)

    deskripsi = "; ".join(f"{k.split('__')[1]}={label_param(v)}" for k, v in params.items())
    return pipe, deskripsi


def jalankan_grid(nama: str, pipeline: Pipeline, grid: dict, X, y, n_jobs: int, out_dir: Path):
    """Jalankan GridSearchCV dan simpan tabel skornya ke CSV."""
    # StratifiedKFold = bagi train jadi 5 bagian dengan proporsi hoaks:valid yang sama di tiap bagian.
    # shuffle + random_state = diacak dulu, tapi acakannya selalu sama (reproducible).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    jumlah_kombinasi = 1
    for nilai in grid.values():
        jumlah_kombinasi *= len(nilai)
    print(f"\n### {nama}: {jumlah_kombinasi} kombinasi x 5 fold = {jumlah_kombinasi * 5} kali latih")

    search = GridSearchCV(
        pipeline, grid, scoring=SKOR,
        refit="f1",      # setelah selesai, latih ulang kombinasi terbaik (menurut F1) di SELURUH train
        cv=cv, n_jobs=n_jobs, verbose=1,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    print(f"Selesai dalam {(time.perf_counter() - t0) / 60:.1f} menit")

    # cv_results_ = semua skor semua kombinasi. Kita rapikan jadi tabel yang enak dibaca.
    hasil = pd.DataFrame(search.cv_results_)
    kolom_param = [k for k in hasil.columns if k.startswith("param_")]
    for k in kolom_param:
        hasil[k] = hasil[k].apply(label_param)
    kolom_skor = [f"mean_test_{s}" for s in SKOR] + ["std_test_f1", "rank_test_f1"]
    hasil = hasil[kolom_param + kolom_skor].sort_values("rank_test_f1")
    hasil.columns = [k.replace("param_", "").replace("mean_test_", "cv_") for k in hasil.columns]

    out_dir.mkdir(parents=True, exist_ok=True)
    hasil.to_csv(out_dir / f"cv_{nama}.csv", index=False)
    return search, hasil


def tampilkan_efek(hasil: pd.DataFrame, kolom_param: list[str]) -> None:
    """Untuk tiap parameter: rata-rata F1 CV per nilainya. Biar kelihatan parameter mana yang ngaruh."""
    print("\nEfek tiap parameter (rata-rata cv_f1 dari semua kombinasi yang memakai nilai itu):")
    for k in kolom_param:
        rata = hasil.groupby(k)["cv_f1"].mean().round(4)
        print(f"  {k:28s} " + " | ".join(f"{v}: {s}" for v, s in rata.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning hyperparameter dengan GridSearchCV (M6).")
    parser.add_argument("--catat", action="store_true", help="tambahkan model terbaik ke reports/results.md")
    parser.add_argument("--n-jobs", type=int, default=2, help="jumlah proses paralel (default 2, hemat RAM)")
    parser.add_argument("--uji", action="store_true", help="uji cepat: 3000 dokumen & grid kecil")
    args = parser.parse_args()

    train = muat_split_bersih("train")
    val = muat_split_bersih("val")
    out_dir = OUT_DIR
    if args.uji:
        # Mode uji: cuma buat ngecek kodenya jalan dari awal sampai akhir. Hasilnya JANGAN dipakai.
        train = train.sample(3000, random_state=RANDOM_STATE)
        out_dir = ROOT / "data" / "processed" / "cache" / "tuning_uji"  # folder yang nggak di-commit
        args.catat = False
        print("MODE UJI: 3000 dokumen, grid kecil, hasil disimpan terpisah & tidak dicatat")
    X_train, y_train = train["teks_bersih"], train["label"]
    X_val, y_val = val["teks_bersih"], val["label"]
    stop_id = stopwords_indonesia()
    print(f"Train {len(train)} | Val {len(val)} | stopword Sastrawi: {len(stop_id)} kata")

    # Pipeline(memory=...) = simpan hasil TF-IDF tiap fold ke folder sementara.
    # Jadi kalau yang beda cuma C (atau alpha), TF-IDF nggak dihitung ulang -> jauh lebih cepat.
    # Ini tetap aman: TF-IDF tetap di-fit HANYA di bagian latih tiap fold.
    cache_dir = tempfile.mkdtemp(prefix="tfidf_cache_")

    try:
        # ------------------------------------------------------------------
        # 1. LOGISTIC REGRESSION
        # ------------------------------------------------------------------
        pipe_lr = Pipeline(
            [("tfidf", TfidfVectorizer()),
             ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))],
            memory=cache_dir,
        )
        # Nama parameter: "<nama langkah>__<nama parameter>", mis. "clf__C" = parameter C milik langkah "clf"
        grid_lr = {
            "tfidf__ngram_range": [(1, 1), (1, 2)],   # 1 kata saja, atau 1 kata + pasangan 2 kata ("tidak benar")
            "tfidf__min_df": [2, 5],                  # buang kata yang muncul di < 2 (atau < 5) dokumen
            "tfidf__max_features": [None, 20_000],    # batasi kamus ke 20 ribu kata tersering (None = tanpa batas)
            "tfidf__stop_words": [None, stop_id],     # stopword dibuang atau tidak
            "clf__C": [0.1, 1.0, 10.0],               # kekuatan "rem" regularisasi: kecil = rem kuat
        }
        if args.uji:
            grid_lr = {"tfidf__min_df": [2], "tfidf__stop_words": [None, stop_id], "clf__C": [1.0]}
        search_lr, hasil_lr = jalankan_grid("logreg", pipe_lr, grid_lr, X_train, y_train, args.n_jobs, out_dir)
        print("\n10 kombinasi terbaik (LogReg):")
        print(hasil_lr.head(10).to_string(index=False))
        tampilkan_efek(hasil_lr, list(grid_lr))

        # ------------------------------------------------------------------
        # 2. NAIVE BAYES (menguji dugaan M4: alpha=1 terlalu besar)
        # ------------------------------------------------------------------
        pipe_nb = Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())], memory=cache_dir)
        grid_nb = {"clf__alpha": [1.0, 0.3, 0.1, 0.03, 0.01]}
        if args.uji:
            grid_nb = {"clf__alpha": [1.0, 0.1]}
        search_nb, hasil_nb = jalankan_grid("naive_bayes", pipe_nb, grid_nb, X_train, y_train, args.n_jobs, out_dir)
        print("\nSemua nilai alpha (Naive Bayes):")
        print(hasil_nb.to_string(index=False))
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)  # hapus cache sementara

    # ------------------------------------------------------------------
    # 3. BANDINGKAN SKOR CV vs SKOR VAL
    # ------------------------------------------------------------------
    # best_estimator_ = pipeline terbaik yang SUDAH dilatih ulang di seluruh train.
    # Sekarang baru dinilai di val, data yang sama sekali nggak dipakai waktu tuning.
    print("\n=== CV (di train) vs VAL ===")
    ringkasan, untuk_catat, terbaik_json = [], [], {}
    for nama, search in [("TF-IDF + LogReg", search_lr), ("TF-IDF + Naive Bayes", search_nb)]:
        model = search.best_estimator_
        skor_val = evaluasi(model, X_val, y_val)
        acc_train = accuracy_score(y_train, model.predict(X_train))
        f1_cv = search.best_score_
        ringkasan.append({
            "model": nama, "cv_f1": f1_cv, "val_f1": skor_val["f1"], "selisih": f1_cv - skor_val["f1"],
            "val_recall": skor_val["recall"], "val_precision": skor_val["precision"],
            "val_auc": skor_val["roc_auc"], "acc_train": acc_train, "acc_val": skor_val["accuracy"],
        })
        param_teks = "; ".join(f"{k.split('__')[1]}={label_param(v)}" for k, v in search.best_params_.items())
        terbaik_json[nama] = {k: label_param(v) if isinstance(v, list) else v for k, v in search.best_params_.items()}
        untuk_catat.append({
            "model": nama, "parameter": param_teks, **skor_val,
            "catatan": f"M6 GridSearchCV 5-fold, CV F1={f1_cv:.4f}, buang_jejak=True",
        })
        print(f"{nama}: {param_teks}")

    print()
    print(pd.DataFrame(ringkasan).set_index("model").round(4).to_string())

    # Skor CV adalah "ramalan" performa di data baru. Kalau val (data baru beneran) jauh lebih jelek,
    # artinya proses tuning kebablasan menyesuaikan diri ke train -> curigai overfitting.
    print()
    for r in ringkasan:
        if abs(r["selisih"]) < 0.02:
            vonis = "kecil -> skor CV bisa dipercaya sebagai ramalan performa di data baru"
        elif r["selisih"] > 0:
            vonis = "CV jauh LEBIH TINGGI dari val -> curigai overfitting saat tuning"
        else:
            vonis = "val lebih tinggi dari CV -> wajar kalau kecil; kalau besar, cek apakah val 'kebetulan gampang'"
        print(f"{r['model']}: selisih F1 CV - val = {r['selisih']:+.4f} ({vonis})")

    # Simpan parameter terbaik buat dipakai lagi di M7/M8.
    # ngram_range berupa tuple; JSON nggak kenal tuple, jadi otomatis jadi list [1, 2].
    with open(out_dir / "terbaik.json", "w", encoding="utf-8") as f:
        json.dump(terbaik_json, f, indent=2, ensure_ascii=False)
    print(f"Parameter terbaik disimpan ke {(out_dir / 'terbaik.json').relative_to(ROOT)}")

    if args.catat:
        catat_ke_results(untuk_catat)


if __name__ == "__main__":
    main()
