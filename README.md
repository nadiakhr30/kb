# Prediksi Lama Rawat Pasien DBD
**Kelompok 7 — KB A**

Aplikasi Streamlit untuk memprediksi lama rawat pasien Demam Berdarah Dengue (DBD) menggunakan **Stacking Regressor**.

## Fitur Aplikasi
| Menu | Deskripsi |
|------|-----------|
| Upload & Preprocessing | Upload dataset, cek missing value, encoding, normalisasi |
| EDA | Visualisasi distribusi, heatmap korelasi, chi-square, silhouette |
| Training Model | Latih Stacking Regressor (LR + KNN + SVR + MLP) |
| Prediksi Pasien | Input data pasien baru → estimasi lama dirawat |
| Evaluasi Model | MSE, RMSE, MAD, MAPE + visualisasi residual |

## Cara Menjalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
- Format: TSV (tab-separated), ekstensi `.csv.xls`
- Target: `lama_dirawat` (jumlah hari dirawat)
- Fitur: `jenis_kelamin`, `umur`, `jenis_demam`, `hemoglobin`, `hct`, `trombosit`, `tgl_masuk`, `tgl_keluar`

## Model
- **Base Regressors**: Linear Regression, KNN (k=5), SVR (RBF kernel), MLP (100 neurons)
- **Meta Regressor**: Linear Regression
- **Cross-validation**: 5-fold
