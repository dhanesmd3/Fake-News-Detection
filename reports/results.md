# Hasil Eksperimen

Semua metrik di bawah dihitung di **val set** (kecuali bagian "Hasil final v1").
Precision, Recall, dan F1 dihitung untuk kelas **hoaks (label 1)**.

| # | Tanggal | Model | Parameter | Accuracy | Precision | Recall | F1 | ROC-AUC | Catatan |
|---|---------|-------|-----------|----------|-----------|--------|----|---------|---------|
| 1 | 2026-11-02 | Dummy | strategy=most_frequent | 0.7743 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | baseline M4, buang_jejak=True |
| 2 | 2026-11-02 | TF-IDF + Naive Bayes | TF-IDF default; MultinomialNB(alpha=1.0) | 0.8521 | 0.9969 | 0.3461 | 0.5138 | 0.9972 | baseline M4, buang_jejak=True |
| 3 | 2026-11-02 | TF-IDF + LogReg | TF-IDF default; LogisticRegression(C=1.0) | 0.9889 | 0.9855 | 0.9651 | 0.9752 | 0.9978 | baseline M4, buang_jejak=True |
| 4 | 2026-11-04 | TF-IDF + LogReg | C=10.0; max_features=-; min_df=5; ngram_range=(1, 2); stop_words=- | 0.9936 | 0.9944 | 0.9771 | 0.9857 | 0.9986 | M6 GridSearchCV 5-fold, CV F1=0.9862, buang_jejak=True |
| 5 | 2026-11-04 | TF-IDF + Naive Bayes | alpha=0.03 | 0.9552 | 0.9842 | 0.8144 | 0.8913 | 0.9963 | M6 GridSearchCV 5-fold, CV F1=0.8863, buang_jejak=True |

## Hasil final v1 (test set, dipakai SEKALI)

_(belum ada, diisi di M8)_
