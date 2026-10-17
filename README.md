# 📚Scikit-learn-cookbook
Over 80 recipes for machine learning in Python with scikit-learn
# scikit-learn Cookbook — Bab 1–8

|           |                         |
| --------- | ----------------------- |
| **Nama**  | Zacky Yusup Hakim       |
| **NIM**   | 101032300183            |
| **Kelas** | BS1TK-47-REG-G13        |

Repositori ini berisi rangkuman dan implementasi kode Python untuk **Bab 1 sampai Bab 8** dari buku *scikit-learn Cookbook, Third Edition* (John Sukup, Packt Publishing, 2025). Setiap bab disajikan dalam satu Jupyter Notebook yang menggabungkan penjelasan konsep (Bahasa Indonesia) dengan contoh kode, visualisasi, ringkasan *Key Ideas*, dan latihan beserta solusinya (pada bab yang memiliki latihan).

Repositori ini dibuat untuk **Tugas 2 (Enrichment for Machine Learning Classes): Code Reproduction + Theoretical Deep-Dive**. Seluruh output (grafik dan hasil perhitungan) sudah tersimpan di notebook, sehingga isinya dapat dibaca tanpa dijalankan ulang.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Instalasi & Cara Menjalankan](#instalasi--cara-menjalankan)
3. [Bab 1 — Common Conventions and API Elements of scikit-learn](#bab-1--common-conventions-and-api-elements-of-scikit-learn)
4. [Bab 2 — Pre-Model Workflow and Data Preprocessing](#bab-2--pre-model-workflow-and-data-preprocessing)
5. [Bab 3 — Dimensionality Reduction Techniques](#bab-3--dimensionality-reduction-techniques)
6. [Bab 4 — Building Models with Distance Metrics and Nearest Neighbors](#bab-4--building-models-with-distance-metrics-and-nearest-neighbors)
7. [Bab 5 — Linear Models and Regularization](#bab-5--linear-models-and-regularization)
8. [Bab 6 — Advanced Logistic Regression and Extensions](#bab-6--advanced-logistic-regression-and-extensions)
9. [Bab 7 — Support Vector Machines and Kernel Methods](#bab-7--support-vector-machines-and-kernel-methods)
10. [Bab 8 — Tree-Based Algorithms and Ensemble Methods](#bab-8--tree-based-algorithms-and-ensemble-methods)
11. [Dataset](#dataset)
12. [Library yang Digunakan](#library-yang-digunakan)
13. [Referensi](#referensi)

---

## Struktur Proyek

```
scikit-learn-cookbook/
├── README.md
├── scikit_learn_Cookbook_Chapter1.ipynb
├── scikit_learn_Cookbook_Chapter2.ipynb
├── scikit_learn_Cookbook_Chapter3.ipynb
├── scikit_learn_Cookbook_Chapter4.ipynb
├── scikit_learn_Cookbook_Chapter5.ipynb
├── scikit_learn_Cookbook_Chapter6.ipynb
├── scikit_learn_Cookbook_Chapter7.ipynb
└── scikit_learn_Cookbook_Chapter8.ipynb
```

## Instalasi & Cara Menjalankan

```
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Notebook juga dapat dibuka langsung di **Google Colab** (*File → Open notebook → GitHub*). Semua dataset berasal dari modul `sklearn.datasets` atau dibuat secara sintetis di dalam notebook. Satu-satunya yang memerlukan internet adalah `fetch_california_housing()` pada latihan akhir Bab 2, karena datasetnya diunduh saat pertama kali dijalankan.

---

## Bab 1 — Common Conventions and API Elements of scikit-learn

📓 [`scikit-learn-Cookbook-Chapter1.ipynb`](scikit_learn_Cookbook_Chapter1.ipynb)

Bab fondasi: **konvensi dan pola API** yang dipakai hampir semua model scikit-learn. Bab ini tidak memiliki repositori kode terpisah di buku, sehingga seluruh contohnya diambil dari teks buku.

**Materi:**

1. Filosofi desain scikit-learn (konsistensi, kesederhanaan, modularitas, *reusability*)
2. *Estimator*: `fit()`, `predict()`, `fit_predict()` (`LinearRegression`, `KMeans`)
3. *Transformer*: `fit()`, `transform()`, `fit_transform()` (`StandardScaler`)
4. Estimator dan transformer kustom (`BaseEstimator`, *mixin class*)
5. Pipeline dan otomatisasi *workflow* (pengantar MLOps)
6. Atribut dan metode umum (`coef_`, `intercept_`, `score()`)
7. Penyetelan hyperparameter (`GridSearchCV`, `RandomizedSearchCV`, `set_params()`, `get_params()`)
8. Metadata: *tags* dan *metadata routing*
9. Praktik terbaik penggunaan API

**Rangkuman:**

- Hampir semua model mengikuti pola **`fit()` → `predict()`** (atau `transform()`), sehingga mengganti model cukup mengubah satu baris kode.
- **Data uji hanya di-`transform()`** dengan parameter dari data latih; memanggil `fit` ulang pada data uji menyebabkan inkonsistensi (*data leakage*).
- **Pipeline** membuat *workflow* ringkas, konsisten, dan dapat direproduksi.

---

## Bab 2 — Pre-Model Workflow and Data Preprocessing

📓 [`scikit-learn-Cookbook-Chapter2.ipynb`](scikit_learn_Cookbook_Chapter2.ipynb)

*Garbage in, garbage out*: menyiapkan data mentah sebelum pemodelan.

**Materi:**

1. Dampak data mentah terhadap performa model
2. Data hilang: `SimpleImputer`, `KNNImputer`, `IterativeImputer`
3. Scaling: `StandardScaler`, `MinMaxScaler`, `Normalizer`
4. Encoding kategorikal: `OneHotEncoder`, `LabelEncoder`, `ColumnTransformer`
5. Pengantar pipeline & visualisasi pipeline (`set_config(display="diagram")`)
6. *Feature engineering*: `PolynomialFeatures`, `KBinsDiscretizer`, `RFE`, `SelectFromModel`
7. Latihan: pipeline komprehensif pada California Housing

**Rangkuman:**

- Imputasi bergerak dari yang sederhana (rata-rata) ke berbasis tetangga (KNN) dan berbasis model antarfitur (iteratif).
- `StandardScaler` (z-score), `MinMaxScaler` (rentang 0–1), dan `Normalizer` (norma satuan per **baris**) punya kegunaan berbeda; scaling penting untuk model berbasis jarak/gradien.
- `OneHotEncoder` untuk kategori nominal; `LabelEncoder` hanya untuk label target.
- Preprocessing sebaiknya dibangun sebagai **pipeline** agar konsisten dan bebas *data leakage*.

---

## Bab 3 — Dimensionality Reduction Techniques

📓 [`scikit-learn-Cookbook-Chapter3.ipynb`](scikit_learn_Cookbook_Chapter3.ipynb)

Mengurangi jumlah fitur sambil mempertahankan informasi penting.

**Materi:**

1. Pengantar: *curse of dimensionality*, *feature selection* vs *feature extraction*
2. PCA: variansi maksimum, komponen ortogonal, *explained variance ratio*
3. LDA: separasi kelas (*supervised*)
4. Perbedaan PCA dan LDA
5. t-SNE untuk visualisasi dan pengaruh `perplexity`
6. Panduan memilih teknik reduksi dimensi
7. Latihan: PCA + regresi logistik (Digits) dan t-SNE + K-means

**Rangkuman:**

- **PCA** (*unsupervised*) pada dataset Wine: dua komponen pertama menjelaskan ≈ 55% variansi (PC1 ≈ 36,2%, PC2 ≈ 19,2%); butuh ≈ 10 komponen untuk 95%.
- **LDA** (*supervised*) memisahkan tiga kelas anggur lebih jelas dari PCA; komponen maksimum = jumlah kelas − 1.
- **t-SNE** bersifat non-linear dan hanya untuk visualisasi (tidak punya `transform()` untuk data baru).
- Pada Digits, PCA mereduksi 64 fitur menjadi 39 dengan akurasi regresi logistik turun tipis dari ≈ 97,5% ke ≈ 96,4%.

---

## Bab 4 — Building Models with Distance Metrics and Nearest Neighbors

📓 [`scikit-learn-Cookbook-Chapter4.ipynb`](scikit_learn_Cookbook_Chapter4.ipynb)

Model **KNN**: prediksi dari *k* tetangga terdekat, sehingga hasilnya bergantung pada definisi "jarak".

**Materi:**

1. Memahami KNN: cara kerja, pengaruh nilai *k*, kelemahan
2. Metrik jarak: Euclidean, Manhattan, Minkowski
3. Penyetelan hyperparameter dengan `GridSearchCV`
4. Evaluasi: *cross-validation*, *learning curve*, *confusion matrix*, precision/recall/F1
5. Tambahan: efek scaling, memilih *k*, KNN untuk regresi
6. Latihan: classifier KNN (Breast Cancer), *grid search*, dan evaluasi pada data sintetis

**Rangkuman:**

- *k* kecil → *overfitting*; *k* besar → *underfitting*. Nilai terbaik dipilih lewat *cross-validation*.
- Pada Iris, kombinasi terbaik hasil grid search: `n_neighbors=9`, `metric='euclidean'`, `weights='uniform'` (skor CV ≈ 0,99).
- **Scaling sangat penting**: pada Wine, akurasi KNN naik dari ≈ 63,9% (tanpa scaling) menjadi ≈ 94,4% (dengan scaling).
- Pada Breast Cancer, KNN yang di-*scale* mencapai akurasi ≈ 94,7%.

---

## Bab 5 — Linear Models and Regularization

📓 [`scikit-learn-Cookbook-Chapter5.ipynb`](scikit_learn_Cookbook_Chapter5.ipynb)

Regresi linear, masalahnya pada data berkorelasi, dan solusi **regularisasi**.

**Materi:**

1. Regresi linear (OLS, MSE, R²) dan asumsinya
2. Ridge (L2) dan Lasso (L1)
3. ElasticNet dan *coefficient path plot*
4. Regresi polinomial (`PolynomialFeatures`)
5. Spline (`SplineTransformer`)
6. Tambahan: memilih `alpha` dengan `RidgeCV` dan `LassoCV`
7. Latihan: Ridge, Lasso, dan ElasticNet

**Rangkuman:**

- **Ridge** mengecilkan koefisien; **Lasso** dapat menolkan koefisien (seleksi fitur otomatis); **ElasticNet** menggabungkan keduanya lewat `l1_ratio`.
- Pada data multikolinear, ElasticNet memberi R² terbaik (≈ 0,15) dibanding Ridge (≈ 0,08), Lasso (≈ 0,06), dan regresi linear (≈ 0,06). Nilai R² rendah karena sinyal pada dataset sintetis ini lemah.
- Dengan `alpha` dipilih lewat *cross-validation*, `LassoCV` hanya mempertahankan 14 dari 100 fitur dan menaikkan R² dari ≈ 0,065 menjadi ≈ 0,205.
- Polinomial derajat 5 mencapai R² ≈ 0,983 pada data bergelombang; pada latihan Ridge, penambahan fitur polinomial menaikkan R² uji dari ≈ −0,04 menjadi ≈ 0,89.

---

## Bab 6 — Advanced Logistic Regression and Extensions

📓 [`scikit_learn_Cookbook_Chapter6.ipynb`](scikit_learn_Cookbook_Chapter6.ipynb)

Memperluas **regresi logistik** dari klasifikasi biner ke multikelas, regularisasi, multilabel, dan evaluasi model klasifikasi.

**Materi:**

1. Ikhtisar regresi logistik: sigmoid, *odds*/log-odds, MLE (*log-loss*), ambang keputusan, *odds ratio*
2. Klasifikasi multikelas: *One-vs-Rest* (`OneVsRestClassifier`) vs multinomial/*softmax*
3. Regularisasi L1/L2 pada regresi logistik: parameter `C`, jalur koefisien, `GridSearchCV`
4. Klasifikasi multilabel: satu model biner per label, *co-occurrence* label, *subset accuracy*, *Hamming loss*
5. Metrik evaluasi: precision, recall, F1, kurva ROC/AUC, *precision-recall curve*
6. Latihan: Lasso/Ridge, multinomial pada Iris, visualisasi *decision boundary*

**Rangkuman:**

- Regresi logistik adalah model **linear dalam log-odds**; akurasi ≈ 94,7% pada Breast Cancer, dan menggeser ambang keputusan menukar jenis kesalahan (ganas lolos vs jinak salah dianggap ganas).
- Pada Iris, OvR (≈ 0,956) dan multinomial (≈ 0,933) hampir setara; dengan hanya dua fitur *petal*, multinomial (≈ 0,911) mengungguli OvR (≈ 0,844). Tidak ada strategi yang selalu menang.
- **`C` kecil = regularisasi kuat.** Dengan `C=0.1`, Ridge mencapai akurasi ≈ 0,982 dan Lasso ≈ 0,965 (menolkan 23 dari 30 koefisien); `GridSearchCV` memilih `C=0.1` dengan L2 (skor CV ≈ 0,975). *Scaling* menaikkan akurasi Ridge dari ≈ 0,942 menjadi ≈ 0,982.
- Pada multilabel, *subset accuracy* hanya ≈ 0,340 (sangat ketat), F1 *micro* ≈ 0,701 vs F1 *macro* ≈ 0,577 karena label jarang sulit dipelajari.
- ROC AUC ≈ 0,99 (CV 5-*fold* ≈ 0,995); regresi logistik pada data sintetis non-linear hanya ≈ 0,897 (dibandingkan dengan SVM pada Bab 7).

---

## Bab 7 — Support Vector Machines and Kernel Methods

📓 [`scikit_learn_Cookbook_Chapter7.ipynb`](scikit_learn_Cookbook_Chapter7.ipynb)

Model geometris yang mencari batas dengan **margin maksimum**, diperluas ke batas non-linear lewat **trik kernel**.

**Materi:**

1. Pengantar SVM: *hyperplane*, margin, *support vectors*, *soft margin* (`C`), `SVC` dan `SVR`
2. Fungsi kernel: linear, polinomial, RBF (`gamma`), data non-linear (`make_moons`, `make_circles`)
3. Penyetelan parameter: `GridSearchCV`, efek `C`, interaksi `C` × `gamma`
4. SVM pada ruang berdimensi tinggi: 1.000 fitur, proyeksi PCA, perbandingan kernel
5. Evaluasi SVM: ROC/AUC, `probability=True`, efek *scaling*
6. Latihan: klasifikasi Iris, *grid search* Breast Cancer, visualisasi *decision boundary*

**Rangkuman:**

- Pada Iris, kernel linear ≈ 0,93, polinomial ≈ 0,96, RBF ≈ 0,91 (selisih 1–2 sampel dari 45 data uji). Pada `make_circles`, kernel linear hanya ≈ 0,458 sedangkan RBF ≈ 1,000.
- Kurva `C` pada RBF memuncak pada `C ≈ 2,8` (skor CV ≈ 0,990) lalu menurun pelan (*diminishing returns*); `gamma` terlalu besar menyebabkan *overfitting* (latih 0,98 vs uji 0,87 pada data *moons*).
- Pada data 1.000 fitur, kernel linear dengan `C` bawaan *overfit* (latih 1,0, uji ≈ 0,747), sedangkan RBF mencapai ≈ 0,883–0,893. Proyeksi PCA 2D hanya menjelaskan ≈ 4,6% variansi.
- **Standardisasi fitur sangat penting:** akurasi SVM RBF pada Breast Cancer naik dari ≈ 0,918 menjadi ≈ 0,982 (AUC ≈ 0,999).
- Pada data sintetis 2 fitur, SVM RBF ≈ 0,977 vs regresi logistik ≈ 0,897.

---

## Bab 8 — Tree-Based Algorithms and Ensemble Methods

📓 [`scikit_learn_Cookbook_Chapter8.ipynb`](scikit_learn_Cookbook_Chapter8.ipynb)

Pohon keputusan sebagai blok bangunan, lalu ensemble: *bagging*, *boosting*, dan *stacking*.

**Materi:**

1. Pengantar pohon keputusan: Gini/entropi, hiperparameter pengendali kompleksitas, `plot_tree`, `export_text`
2. *Random forest* dan *bagging*: *bootstrap*, skor OOB, *feature importance* vs *permutation importance*
3. *Gradient boosting*: pohon berurutan pada residu, `learning_rate`, `staged_predict`, *early stopping*
4. Penyetelan hiperparameter: `GridSearchCV`, `RandomizedSearchCV`
5. Membandingkan ensemble: bagging vs boosting vs stacking pada beberapa dataset (5-*fold* CV)
6. Latihan: pohon pada Wine, *grid search* *random forest*, GB vs RF pada Digits

**Rangkuman:**

- Pohon tunggal pada Iris ≈ 0,867 (kedalaman 4); *random forest* ≈ 0,911. Pada Breast Cancer (CV 5-*fold*): pohon tunggal ≈ 0,940, *bagging* ≈ 0,952, *random forest* ≈ 0,961, skor OOB ≈ 0,965.
- Pada Iris, `petal length` dan `petal width` menyumbang ≈ 88,5% *feature importance*; *permutation importance* menunjukkan fitur *sepal* praktis tidak berpengaruh.
- *Gradient boosting* dengan `learning_rate` kecil memuncak lebih lambat tetapi sedikit lebih tinggi (≈ 0,982 pada pohon ke-199 vs ≈ 0,977 pada pohon ke-81 untuk `learning_rate=0.5`); *early stopping* perlu diatur agar tidak berhenti terlalu dini.
- Perbandingan 3 dataset: *random forest* terbaik atau setara terbaik (Wine ≈ 0,983), *stacking* hampir setara tetapi lebih mahal, *gradient boosting* tanpa penyetelan tidak unggul; semua ensemble mengalahkan pohon tunggal.
- Pada Digits, *random forest* ≈ 0,97 dan *gradient boosting* ≈ 0,96, dengan waktu latih *random forest* jauh lebih singkat.

---

## Dataset

| Bab | Dataset                                                                | Sumber                                       |
| --- | ---------------------------------------------------------------------- | -------------------------------------------- |
| 1   | Data mainan, Iris (pada contoh tambahan)                               | Didefinisikan di notebook / `sklearn.datasets` |
| 2   | Data sintetis acak; California Housing (latihan)                       | NumPy / `fetch_california_housing()`         |
| 3   | Wine, Digits                                                           | `sklearn.datasets`                           |
| 4   | Iris, Wine, Breast Cancer; `make_circles`, `make_classification`       | `sklearn.datasets`                           |
| 5   | `make_regression` (multikolinear), data polinomial/sinus sintetis      | `sklearn.datasets` / NumPy                   |
| 6   | Breast Cancer, Iris; `make_multilabel_classification`, `make_classification` | `sklearn.datasets`                     |
| 7   | Iris, Breast Cancer; `make_moons`, `make_circles`, `make_classification` (1.000 fitur); data sinus sintetis | `sklearn.datasets` / NumPy |
| 8   | Iris, Wine, Breast Cancer, Digits                                      | `sklearn.datasets`                           |

## Library yang Digunakan

`numpy` · `pandas` · `scipy` · `scikit-learn` · `matplotlib` · `seaborn`

## Referensi

- Sukup, J. (2025). *scikit-learn Cookbook, Third Edition*. Packt Publishing.
- Repositori kode resmi: <https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition>
- Dokumentasi scikit-learn: <https://scikit-learn.org/stable/>
