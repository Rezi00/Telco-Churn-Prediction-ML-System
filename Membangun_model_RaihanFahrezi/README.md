# Membangun Model - Telco Customer Churn

Folder ini berisi implementasi Kriteria 2 proyek akhir **Membangun Sistem Machine Learning**.

Dataset yang digunakan adalah hasil preprocessing dari Kriteria 1, yaitu:

```text
telco_churn_preprocessed.csv
```

Target prediksi:

```text
Churn
```

Jenis masalah:

```text
Binary Classification
```

## Struktur Folder

```text
Membangun_model
├── modelling.py
├── modelling_tuning.py
├── telco_churn_preprocessed.csv
├── requirements.txt
├── screenshot_dashboard.jpg
└── screenshot_artifact.jpg
```

## Cara Menjalankan

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan training basic dengan MLflow autolog:

```bash
python modelling.py
```

Jalankan training skilled dengan hyperparameter tuning dan manual logging:

```bash
python modelling_tuning.py
```

Buka MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Lalu buka browser:

```text
http://127.0.0.1:5000
```

## Checklist Kriteria 2 Skilled

- Melatih model machine learning menggunakan MLflow Tracking UI.
- Menggunakan dataset hasil preprocessing.
- `modelling.py` menggunakan MLflow autolog.
- `modelling_tuning.py` menggunakan hyperparameter tuning.
- `modelling_tuning.py` menggunakan manual logging.
- Mencatat metrics: accuracy, precision, recall, f1_score.
- Menyimpan artifact tambahan:
  - `confusion_matrix.png`
  - `classification_report.txt`
  - `best_random_forest_model.joblib`
