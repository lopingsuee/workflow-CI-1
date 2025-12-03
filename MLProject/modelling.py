import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import argparse
import joblib

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    data = pd.read_csv(path)
    print(f"Data berhasil dimuat. Jumlah data: {data.shape}")
    return data

def train_model(data: pd.DataFrame, model_path: str):
    X = data.drop(columns=["pass_status"])
    y = data["pass_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nHASIL EVALUASI MODEL")
    print(f"Akurasi     : {acc:.4f}")
    print(f"Presisi     : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"\nModel disimpan ke: {model_path}")

    mlflow.sklearn.save_model(model, "mlflow_model_ci")
    print("MLflow model disimpan ke folder: mlflow_model_ci")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression model for student performance"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="siswa_clean.csv",
        help="Path ke dataset hasil preprocessing"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_ci.joblib",
        help="Path file model yang disimpan"
    )
    args = parser.parse_args()
    data = load_data(args.data_path)
    train_model(data, args.model_path)
