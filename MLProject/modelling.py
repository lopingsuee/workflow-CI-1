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


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset hasil preprocessing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File tidak ditemukan: {path}")
    data = pd.read_csv(path)
    print(f"✅ Data berhasil dimuat. Jumlah data: {data.shape}")
    return data


def train_model(data: pd.DataFrame):
    """Melatih model Logistic Regression dan mencatat hasil di MLflow."""
    # Pisahkan fitur dan target
    X = data.drop(columns=["pass_status"])
    y = data["pass_status"]

    # Split data train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Konfigurasi MLflow
    mlflow.set_experiment("student-performance")
    mlflow.sklearn.autolog()

    # ✅ Cek apakah sudah ada run aktif dari MLflow (misalnya di CI)
    active_run = mlflow.active_run()
    if active_run:
        print(f"ℹ️ MLflow run sudah aktif (ID: {active_run.info.run_id}), lanjut tanpa membuat run baru.")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
    else:
        # Jalankan training dengan MLflow Tracking (lokal / manual)
        with mlflow.start_run(run_name="logistic_regression_CI", nested=True):
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

    # Log metrik manual (aman untuk CI & lokal)
    try:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
    except Exception as e:
        print(f"⚠️ Peringatan: Gagal mencatat metrik di MLflow ({e})")

    print("\n✅ Model berhasil dilatih dan dicatat di MLflow!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression model for student performance")
    parser.add_argument(
        "--data_path",
        type=str,
        default="siswa_clean.csv",
        help="Path ke dataset hasil preprocessing"
    )
    args = parser.parse_args()

    data = load_data(args.data_path)
    train_model(data)
