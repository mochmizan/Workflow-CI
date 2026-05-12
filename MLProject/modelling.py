"""
modelling.py  –  Kriteria 2 (Basic)
Melatih model klasifikasi Iris menggunakan MLflow autolog.
Tracking UI disimpan lokal di ./mlruns

Cara menjalankan (dari folder Membangun_model/):
    python modelling.py

Lalu buka MLflow UI:
    mlflow ui --port 5000
    http://127.0.0.1:5000
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
TRAIN_PATH  = os.path.join('iris_preprocessing', 'train.csv')
TEST_PATH   = os.path.join('iris_preprocessing', 'test.csv')
TARGET_COL  = 'species'
EXPERIMENT  = 'Iris_Classification_Basic'
MLFLOW_URI  = 'mlruns'   # local

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# ─── Load Data ──────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    feature_cols = [c for c in train.columns if c != TARGET_COL]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COL]
    return X_train, X_test, y_train, y_test, feature_cols


# ─── Training ───────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n[1/2] Training Logistic Regression...")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="LogisticRegression_autolog", nested=True):
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

    mlflow.sklearn.autolog(disable=True)
    return model


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n[2/2] Training Random Forest...")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_autolog", nested=True):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec  = recall_score(y_test, y_pred, average='weighted')

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")

    mlflow.sklearn.autolog(disable=True)
    return model


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MODELLING - Moch Mizan Ghodafail (Basic)")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"Data dimuat. Train: {X_train.shape} | Test: {X_test.shape}")

    train_logistic_regression(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 55)
    print("  SELESAI! Buka MLflow UI:")
    print("  mlflow ui --port 5000")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
