# =========================================
# 🧠 MODELLING.PY — Final Version (Lokal + CI)
# =========================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import os
import argparse


# ===============================
# 1️⃣ Fungsi untuk Load Dataset
# ===============================
def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset hasil preprocessing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {path}")
    data = pd.read_csv(path)
    print(f"✅ Data berhasil dimuat. Jumlah data: {data.shape}")
    return data


# ===============================
# 2️⃣ Fungsi untuk Training Model
# ===============================
def train_model(data: pd.DataFrame):
    """
    Melatih model Logistic Regression dan mencatat hasil di MLflow.
    """
    # Pisahkan fitur dan target
    X = data.drop(columns=["pass_status"])
    y = data["pass_status"]

    # Split data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Konfigurasi MLflow
    # mlflow.set_experiment("student-performance")
    mlflow.sklearn.autolog()

    # Jalankan training dengan MLflow Tracking (nested run aman untuk CI)
    with mlflow.start_run(run_name="logistic_regression_CI"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\n=== HASIL EVALUASI MODEL ===")
        print(f"Akurasi     : {acc:.4f}")
        print(f"Presisi     : {prec:.4f}")
        print(f"Recall      : {rec:.4f}")
        print(f"F1-Score    : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Metrik tambahan sudah otomatis dicatat oleh autolog
        # Model juga otomatis tersimpan oleh autolog

    print("\n✅ Model berhasil dilatih dan dicatat di MLflow!")


# ===============================
# 3️⃣ Entry Point (Lokal + CI)
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression model for student performance")
    parser.add_argument(
        "--data_path",
        type=str,
        default="siswa_clean.csv",
        help="Path ke dataset hasil preprocessing"
    )
    args = parser.parse_args()

    # Load data dan training model
    data = load_data(args.data_path)
    train_model(data)
