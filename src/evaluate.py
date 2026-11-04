"""M5: evaluasi yang benar di VAL. Confusion matrix, metrik per kelas, ROC curve.

Test set TIDAK disentuh (itu jatah M8).

Cara menjalankan (dari root project, venv aktif):
    python src/evaluate.py
Gambar disimpan ke reports/figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # mode tanpa jendela: gambar langsung disimpan ke file, nggak muncul popup
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from preprocess import muat_split_bersih
from train import buat_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
NAMA_KELAS = ["valid", "hoaks"]  # urutan sesuai label: 0 = valid, 1 = hoaks

# --- Warna grafik ---
# Diambil dari palet yang sudah dicek aman untuk buta warna. Teks pakai warna netral (bukan warna seri).
WARNA = {
    "latar": "#fcfcfb", "teks": "#0b0b0b", "teks2": "#52514e", "redup": "#898781",
    "grid": "#e1e0d9", "sumbu": "#c3c2b7",
}
WARNA_MODEL = {"TF-IDF + Naive Bayes": "#2a78d6", "TF-IDF + LogReg": "#eb6834"}  # biru, oranye
# Gradasi 1 warna (biru muda -> biru tua) untuk confusion matrix: makin gelap = makin besar
CMAP_BIRU = LinearSegmentedColormap.from_list(
    "biru", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)


def atur_gaya() -> None:
    """Gaya dasar semua grafik: latar terang, garis bantu tipis, teks netral."""
    plt.rcParams.update({
        "figure.facecolor": WARNA["latar"], "axes.facecolor": WARNA["latar"],
        "savefig.facecolor": WARNA["latar"],
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
        "text.color": WARNA["teks"], "axes.labelcolor": WARNA["teks2"],
        "xtick.color": WARNA["redup"], "ytick.color": WARNA["redup"],
        "axes.edgecolor": WARNA["sumbu"], "axes.titleweight": "bold", "axes.titlesize": 11,
    })


# =====================================================================
# 1. LATIH & PREDIKSI
# =====================================================================
def latih_dan_prediksi(train, val) -> dict[str, dict]:
    """Latih 3 model baseline (sama persis dengan M4), lalu prediksi val.

    Return: {nama_model: {"pred": label tebakan, "proba": peluang hoaks}}
    """
    hasil = {}
    for nama, (model, _) in buat_model().items():
        model.fit(train["teks_bersih"], train["label"])
        hasil[nama] = {
            "pred": model.predict(val["teks_bersih"]),
            "proba": model.predict_proba(val["teks_bersih"])[:, 1],
        }
        print(f"[selesai] {nama}")
    return hasil


# =====================================================================
# 2. LAPORAN TEKS
# =====================================================================
def cetak_laporan(nama: str, y, pred, proba) -> None:
    """Cetak confusion matrix dalam kata-kata + precision/recall/F1 untuk TIAP kelas."""
    # confusion_matrix -> tabel 2x2:
    #                 tebak valid   tebak hoaks
    #   asli valid  [[    TN     ,      FP    ],
    #   asli hoaks   [    FN     ,      TP    ]]
    # .ravel() = ratakan jadi 4 angka berurutan: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    print(f"\n{'=' * 64}\n{nama}\n{'=' * 64}")
    print(f"  TN (valid ditebak valid)       : {tn:5d}  <- benar")
    print(f"  TP (hoaks ketangkap)           : {tp:5d}  <- benar")
    print(f"  FP (valid DITUDUH hoaks)       : {fp:5d}  <- salah")
    print(f"  FN (hoaks LOLOS, dikira valid) : {fn:5d}  <- salah")

    # classification_report = precision, recall, F1 untuk SETIAP kelas sekaligus.
    # zero_division=0: Dummy nggak pernah nebak hoaks, jadi precision hoaks-nya dianggap 0 (bukan error)
    print()
    print(classification_report(y, pred, target_names=NAMA_KELAS, digits=4, zero_division=0))
    print(f"  ROC-AUC: {roc_auc_score(y, proba):.4f}")


# =====================================================================
# 3. GAMBAR: CONFUSION MATRIX
# =====================================================================
def plot_confusion(hasil: dict, y, path: Path) -> None:
    """Satu panel per model. Warna = % dari baris (label asli), biar kelas kecil (hoaks) tetap kelihatan."""
    nama_sel = [["TN · benar", "FP · valid dituduh hoaks"],
                ["FN · hoaks lolos", "TP · hoaks ketangkap"]]

    fig, axes = plt.subplots(1, len(hasil), figsize=(4.2 * len(hasil), 4.2))
    for ax, (nama, h) in zip(axes, hasil.items()):
        cm = confusion_matrix(y, h["pred"])
        # Persentase per BARIS: dari semua yang aslinya valid (atau hoaks), berapa % ke tiap tebakan
        persen = cm / cm.sum(axis=1, keepdims=True)

        ax.imshow(persen, cmap=CMAP_BIRU, vmin=0, vmax=1)
        # Garis warna latar di antara sel = celah tipis biar tiap kotak terpisah jelas
        for pos in (0.5,):
            ax.axhline(pos, color=WARNA["latar"], lw=3)
            ax.axvline(pos, color=WARNA["latar"], lw=3)

        for i in range(2):          # i = baris (label asli)
            for j in range(2):      # j = kolom (tebakan model)
                # Sel gelap -> teks putih; sel terang -> teks hitam (biar selalu terbaca)
                warna_teks = "#ffffff" if persen[i, j] > 0.55 else WARNA["teks"]
                jumlah = f"{cm[i, j]:,}".replace(",", ".")  # 3,142 -> 3.142 (format Indonesia)
                ax.text(j, i - 0.18, nama_sel[i][j], ha="center", va="center", fontsize=8.5, color=warna_teks)
                ax.text(j, i + 0.05, jumlah, ha="center", va="center", fontsize=15, fontweight="bold", color=warna_teks)
                ax.text(j, i + 0.27, f"{persen[i, j]:.1%} dari baris", ha="center", va="center", fontsize=8.5, color=warna_teks)

        ax.set_xticks([0, 1], NAMA_KELAS)
        ax.set_yticks([0, 1], NAMA_KELAS)
        ax.set_xlabel("Tebakan model")
        ax.set_ylabel("Label asli")
        ax.set_title(nama)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Confusion matrix di val set (ambang 0,5)", fontweight="bold", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    # bbox_inches="tight" = potong/longgarkan tepi gambar otomatis biar label sumbu nggak kepotong
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tersimpan: {path.relative_to(ROOT)}")


# =====================================================================
# 4. GAMBAR: ROC CURVE
# =====================================================================
def plot_roc(hasil: dict, y, path: Path) -> None:
    """Kiri: ROC utuh. Kanan: diperbesar di pojok kiri atas (kurvanya terlalu mepet buat dibedakan)."""
    # Dummy nggak digambar: ROC-nya persis garis diagonal "tebakan acak"
    model_roc = {n: h for n, h in hasil.items() if n in WARNA_MODEL}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    titik_kerja = {}
    for ax in axes:
        # Garis diagonal = model yang nebak acak (AUC 0,5). Kurva yang bagus melengkung jauh di atasnya.
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=WARNA["redup"], label="tebakan acak (AUC 0.500)")

        for nama, h in model_roc.items():
            # roc_curve mencoba SEMUA ambang (dari 1 turun ke 0). Untuk tiap ambang dihitung:
            #   FPR = FP / (FP + TN) -> % valid yang salah dituduh hoaks  (sumbu x, makin kecil makin bagus)
            #   TPR = TP / (TP + FN) -> % hoaks yang ketangkap = recall    (sumbu y, makin besar makin bagus)
            fpr, tpr, _ = roc_curve(y, h["proba"])
            auc = roc_auc_score(y, h["proba"])
            ax.plot(fpr, tpr, lw=2, color=WARNA_MODEL[nama], label=f"{nama} (AUC {auc:.3f})")

            # Titik kerja = posisi model di ambang 0,5 (ambang yang dipakai .predict())
            tn, fp, fn, tp = confusion_matrix(y, h["pred"]).ravel()
            titik_kerja[nama] = (fp / (fp + tn), tp / (tp + fn))
            ax.scatter(*titik_kerja[nama], s=70, color=WARNA_MODEL[nama],
                       edgecolor=WARNA["latar"], linewidth=2, zorder=3)

        ax.set_xlabel("False Positive Rate (valid dituduh hoaks)")
        ax.set_ylabel("True Positive Rate / recall (hoaks ketangkap)")
        ax.grid(color=WARNA["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sisi in ("top", "right"):
            ax.spines[sisi].set_visible(False)

    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.01)
    axes[0].set_title("ROC curve (utuh)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)

    # Panel kanan: zoom. Batas bawah y disesuaikan biar titik kerja semua model kelihatan.
    y_min = max(0.0, min(t[1] for t in titik_kerja.values()) - 0.08)
    axes[1].set_xlim(0, 0.05)
    axes[1].set_ylim(y_min, 1.005)
    axes[1].set_title("Diperbesar: FPR 0–5%")
    for nama, (x, y_titik) in titik_kerja.items():
        # Titik yang tinggi (dekat kurva-kurva lain) -> label ditaruh agak jauh di kanan bawah (area kosong)
        # pakai garis penunjuk; titik yang rendah -> label cukup di kanan atasnya.
        geser = (45, -55) if y_titik > 0.8 else (14, 16)
        axes[1].annotate(f"{nama}\nambang 0,5 → recall {y_titik:.1%}", (x, y_titik), xytext=geser,
                         textcoords="offset points", fontsize=8.5, color=WARNA["teks2"],
                         arrowprops={"arrowstyle": "-", "color": WARNA["redup"], "lw": 0.8})

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tersimpan: {path.relative_to(ROOT)}")


def main() -> None:
    atur_gaya()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = muat_split_bersih("train")
    val = muat_split_bersih("val")
    y_val = val["label"].to_numpy()

    hasil = latih_dan_prediksi(train, val)

    for nama, h in hasil.items():
        cetak_laporan(nama, y_val, h["pred"], h["proba"])

    print()
    plot_confusion(hasil, y_val, FIG_DIR / "confusion_matrix_val.png")
    plot_roc(hasil, y_val, FIG_DIR / "roc_curve_val.png")


if __name__ == "__main__":
    main()
