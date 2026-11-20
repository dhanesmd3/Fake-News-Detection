"""M7: audit "model nyontek". Semua dinilai di VAL (test tetap dikunci).

1. Koefisien: 20 kata paling condong ke hoaks (+) dan ke valid (-) dari LogReg terbaik (M6)
2. Eksperimen jejak: latih ulang TANPA buang jejak sumber -> seberapa naik/turun skornya?
3. Uji shortcut panjang: model yang CUMA tahu jumlah kata -> seberapa jauh dia bisa nebak?
4. Error analysis: semua prediksi salah di val disimpan ke CSV buat dibaca & dikelompokkan

Cara menjalankan (dari root project, venv aktif):
    python src/audit.py           # run pertama ~8 menit (perlu bersihkan versi "asli" dulu)
    python src/audit.py --catat   # sama, plus catat eksperimen ke reports/results.md

Output:
    reports/m7/koefisien_top.csv, koefisien_top_tanpa_buang_jejak.csv, kesalahan_val.csv
    reports/figures/koefisien_logreg.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluate import WARNA, atur_gaya
from preprocess import muat_split_bersih
from train import RANDOM_STATE, catat_ke_results, evaluasi
from tune import pipeline_terbaik

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "m7"
FIG_DIR = ROOT / "reports" / "figures"
# Pasangan warna "berlawanan" (diverging): biru = condong valid, merah = condong hoaks
WARNA_VALID, WARNA_HOAKS = "#2a78d6", "#e34948"


# =====================================================================
# 1. KOEFISIEN
# =====================================================================
def tabel_koefisien(pipe) -> pd.DataFrame:
    """Pasangkan tiap kata di kamus TF-IDF dengan bobotnya di LogReg."""
    kata = pipe.named_steps["tfidf"].get_feature_names_out()  # daftar kata (kolom ke-0, ke-1, ...)
    bobot = pipe.named_steps["clf"].coef_[0]                   # bobot untuk tiap kolom, urutannya sama
    return pd.DataFrame({"kata": kata, "bobot": bobot})


def top_koefisien(koef: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """n kata paling + (condong hoaks) dan n kata paling - (condong valid)."""
    hoaks = koef.nlargest(n, "bobot").assign(arah="hoaks")
    valid = koef.nsmallest(n, "bobot").assign(arah="valid")
    return pd.concat([hoaks, valid], ignore_index=True)


def plot_koefisien(top: pd.DataFrame, path: Path) -> None:
    """Dua panel batang horizontal: kiri condong valid, kanan condong hoaks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, arah, warna, judul in [
        (axes[0], "valid", WARNA_VALID, "20 kata paling condong ke VALID (bobot −)"),
        (axes[1], "hoaks", WARNA_HOAKS, "20 kata paling condong ke HOAKS (bobot +)"),
    ]:
        # Urutkan biar kata dengan bobot terbesar (nilai mutlak) ada di paling atas
        d = top[top["arah"] == arah].assign(besar=lambda x: x["bobot"].abs()).sort_values("besar")
        ax.barh(d["kata"], d["besar"], color=warna, height=0.65)
        ax.set_title(judul, loc="left")
        ax.set_xlabel("besar bobot (|koefisien|)")
        ax.tick_params(axis="y", length=0, labelcolor=WARNA["teks"])
        ax.grid(axis="x", color=WARNA["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sisi in ("top", "right", "left"):
            ax.spines[sisi].set_visible(False)
    fig.suptitle("Kata yang paling dipercaya LogReg terbaik (M6), dilatih di train",
                 fontweight="bold", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tersimpan: {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model nyontek (M7).")
    parser.add_argument("--catat", action="store_true", help="catat eksperimen ke reports/results.md")
    args = parser.parse_args()
    atur_gaya()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # A. Model utama: LogReg terbaik, jejak sumber DIBUANG (sama dengan M6)
    # -----------------------------------------------------------------
    train = muat_split_bersih("train", buang_jejak=True)
    val = muat_split_bersih("val", buang_jejak=True)
    pipe, deskripsi = pipeline_terbaik("TF-IDF + LogReg")
    print(f"Melatih LogReg terbaik ({deskripsi}) ...")
    pipe.fit(train["teks_bersih"], train["label"])
    skor_buang = evaluasi(pipe, val["teks_bersih"], val["label"])

    koef = tabel_koefisien(pipe)
    top = top_koefisien(koef)
    top.to_csv(OUT_DIR / "koefisien_top.csv", index=False)
    print("\n=== 1. Koefisien (jejak DIBUANG) ===")
    print("Condong HOAKS:", ", ".join(top.loc[top.arah == "hoaks", "kata"]))
    print("Condong VALID:", ", ".join(top.loc[top.arah == "valid", "kata"]))
    plot_koefisien(top, FIG_DIR / "koefisien_logreg.png")

    # -----------------------------------------------------------------
    # B. Eksperimen jejak: setting SAMA, tapi jejak sumber TIDAK dibuang
    # -----------------------------------------------------------------
    train_asli = muat_split_bersih("train", buang_jejak=False)
    val_asli = muat_split_bersih("val", buang_jejak=False)
    pipe_asli, _ = pipeline_terbaik("TF-IDF + LogReg")
    print("\nMelatih LogReg yang sama, tapi jejak sumber TIDAK dibuang ...")
    pipe_asli.fit(train_asli["teks_bersih"], train_asli["label"])
    skor_asli = evaluasi(pipe_asli, val_asli["teks_bersih"], val_asli["label"])

    top_asli = top_koefisien(tabel_koefisien(pipe_asli))
    top_asli.to_csv(OUT_DIR / "koefisien_top_tanpa_buang_jejak.csv", index=False)
    print("Condong VALID (tanpa buang jejak):", ", ".join(top_asli.loc[top_asli.arah == "valid", "kata"].head(10)))

    # -----------------------------------------------------------------
    # C. Uji shortcut panjang: model yang CUMA dikasih tahu jumlah kata
    # -----------------------------------------------------------------
    # log1p = log(1 + x). Dipakai karena panjang teks rentangnya jauh (3 sampai ribuan kata);
    # log "memampatkan" rentang itu biar LogReg lebih gampang memakainya.
    # .reshape(-1, 1) = ubah jadi tabel 1 kolom (scikit-learn butuh input 2 dimensi)
    def fitur_panjang(df):
        return np.log1p(df["teks_bersih"].str.split().str.len().to_numpy()).reshape(-1, 1)

    model_panjang = LogisticRegression(random_state=RANDOM_STATE)
    model_panjang.fit(fitur_panjang(train), train["label"])
    skor_panjang = evaluasi(model_panjang, fitur_panjang(val), val["label"])

    # -----------------------------------------------------------------
    # Ringkasan eksperimen
    # -----------------------------------------------------------------
    eksperimen = [
        ("LogReg terbaik, jejak DIBUANG (model utama)", skor_buang),
        ("LogReg terbaik, jejak TIDAK dibuang", skor_asli),
        ("Cuma panjang teks (1 fitur)", skor_panjang),
    ]
    print("\n=== 2 & 3. Perbandingan di VAL ===")
    tabel = pd.DataFrame({nama: s for nama, s in eksperimen}).T
    print(tabel.round(4).to_string())

    # -----------------------------------------------------------------
    # D. Error analysis: semua prediksi salah model utama di val
    # -----------------------------------------------------------------
    val = val.copy()
    val["proba_hoaks"] = pipe.predict_proba(val["teks_bersih"])[:, 1]
    val["tebakan"] = (val["proba_hoaks"] >= 0.5).astype(int)
    val["n_kata"] = val["teks_bersih"].str.split().str.len()
    salah = val[val["tebakan"] != val["label"]].copy()
    salah["jenis"] = np.where(salah["label"] == 1, "FN (hoaks lolos)", "FP (valid dituduh hoaks)")
    # Urutkan dari yang paling "yakin tapi salah" (paling jauh dari 0,5) biar yang paling parah di atas
    salah["keyakinan_salah"] = (salah["proba_hoaks"] - 0.5).abs()
    salah = salah.sort_values(["jenis", "keyakinan_salah"], ascending=[True, False])
    kolom = ["id", "jenis", "sumber", "proba_hoaks", "n_kata", "teks"]
    salah[kolom].assign(teks=salah["teks"].str.slice(0, 400)).to_csv(
        OUT_DIR / "kesalahan_val.csv", index=False, encoding="utf-8-sig"  # utf-8-sig biar rapi dibuka di Excel
    )
    print(f"\n=== 4. Error analysis ===\n{salah['jenis'].value_counts().to_string()}")
    print(f"Median jumlah kata: salah={salah['n_kata'].median():.0f}, "
          f"semua hoaks val={val.loc[val.label == 1, 'n_kata'].median():.0f}, "
          f"semua valid val={val.loc[val.label == 0, 'n_kata'].median():.0f}")
    print(f"Tersimpan: {(OUT_DIR / 'kesalahan_val.csv').relative_to(ROOT)}")

    if args.catat:
        catat_ke_results([
            {"model": "TF-IDF + LogReg", "parameter": deskripsi, **skor_asli,
             "catatan": "M7 eksperimen: jejak sumber TIDAK dibuang (buang_jejak=False)"},
            {"model": "LogReg cuma panjang teks", "parameter": "1 fitur: log(1+jumlah kata)", **skor_panjang,
             "catatan": "M7 uji shortcut panjang"},
        ])


if __name__ == "__main__":
    main()
