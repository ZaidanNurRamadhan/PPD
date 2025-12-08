# SIAGA Jantung Pro (Cloud Monolith Edition)

Aplikasi Analisis Risiko Penyakit Jantung berbasis *Machine Learning* yang terintegrasi penuh untuk *Telemedicine* dan *Clinical Decision Support System*.

## 1. Tech Stack & Environment
Dokumentasi teknis untuk pengembangan dan pemeliharaan fase selanjutnya.

### Core Environment
- **Bahasa**: Python 3.10+ (Recommended: 3.11/3.12)
- **Framework UI**: Streamlit (Cloud Compatible)
- **Database**: SQLite (Local/Dev) -> Bisa diganti PostgreSQL untuk Production.

### Libraries Utama (`requirements.txt`)
Berikut adalah stack teknologi yang digunakan saat ini:
1.  **Data Processing**:
    - `pandas`, `numpy`: Manipulasi data tabular.
    - `scikit-learn`: Preprocessing pipeline.
2.  **Machine Learning**:
    - `xgboost`: Model klasifikasi utama (Gradient Boosting).
    - `imbalanced-learn` (SMOTE): Penanganan data tidak seimbang saat training.
    - `shap`: *Explainable AI* untuk interpretasi model (Faktor Risiko).
3.  **Visualization**:
    - `plotly`: Grafik interaktif (Gauge, Radar, Trends).
    - `streamlit-option-menu`: Navigasi sidebar modern.
    - `Pillow`: Manipulasi gambar/icon.
4.  **Backend & Data**:
    - `sqlalchemy`: ORM untuk manajemen database.
    - `joblib`: Serialisasi model ML.

## 2. Struktur Proyek
Struktur direktori saat ini (Monolith Architecture):

```text
siaga_heart_app/
├── ml/
│   ├── cardio_model.py          # Wrapper Class untuk Load Model XGBoost & SHAP
│   └── best_xgb_pipeline.joblib # File Model ML (Binary)
├── streamlit_app/
│   └── app.py                   # MAIN APPLICATION (Frontend + Backend + Logic)
├── appheart/
│   ├── database.py              # Koneksi Database (SQLAlchemy)
│   ├── models.py                # Skema Tabel Database (User, Patient, Checkup)
│   ├── schemas.py               # Pydantic Schemas (Data Validation)
│   └── crud.py                  # Create/Read/Update/Delete Operations
├── assets/
│   └── heart.png                # Icon Aplikasi
├── requirements.txt             # Daftar Pustaka (Dependencies)
├── task.md                      # Log Pengerjaan (Dev)
├── walkthrough.md               # Dokumentasi Update (Dev)
└── README.md                    # Dokumentasi Proyek (File Ini)
```

## 3. Panduan Pembaruan (Maintenance)

### A. Environment Setup (Lokal)
1.  Clone Repository.
2.  Buat Virtual Environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    venv\Scripts\activate     # Windows
    ```
3.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### B. Menjalankan Aplikasi
```bash
streamlit run streamlit_app/app.py
```

### C. Deployment (Streamlit Cloud)
1.  Pastikan file `requirements.txt` selalu ter-update jika menambah library baru.
2.  Push perubahan ke GitHub:
    ```bash
    git add .
    git commit -m "Update fitur"
    git push
    ```
3.  Di Dashboard Streamlit Cloud, klik **Reboot App** untuk menarik perubahan terbaru.

## 4. Pengembangan Lanjutan (Next Phase)
Untuk fase selanjutnya, pertimbangkan hal berikut:
- **Database**: Migrasi dari SQLite ke PostgreSQL (Supabase/Neon) untuk data persisten di Cloud.
- **Auth**: Implementasi JWT atau Auth0 jika user bertambah banyak.
- **API**: Mengaktifkan kembali `fastapi` jika ingin memisahkan Backend dan Frontend (Microservices).

