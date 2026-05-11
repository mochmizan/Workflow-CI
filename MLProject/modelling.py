"""
modelling.py  –  Kriteria 2 + Kriteria 3 (CI-ready)
- MLflow autolog (local)
- Setelah training: export model ke model_artifact/
- Menulis run_id.txt
- Generate confusion_matrix.png & feature_importance.png

Cara menjalankan (dari folder MLProject/):
    python modelling.py
"""

import os
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
TRAIN_PATH  = os.path.join('iris_preprocessing', 'train.csv')
TEST_PATH   = os.path.join('iris_preprocessing', 'test.csv')
TARGET_COL  = 'species'
EXPERIMENT  = 'Iris_Classification_CI'
MLFLOW_URI  = 'mlruns'

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    feature_cols = [c for c in train.columns if c != TARGET_COL]

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COL]
    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, X_test, y_train, y_test, feature_cols):
    print("\nTraining Random Forest dengan MLflow autolog...")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_CI"):
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

        # ── Generate artifacts untuk CI ──
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title('Confusion Matrix - Random Forest', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("  [artifact] confusion_matrix.png saved")

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_cols[i] for i in indices]
        sorted_importance = importances[indices]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(range(len(sorted_features)), sorted_importance[::-1],
                       color='#2196F3', edgecolor='white')
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features[::-1])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance - Random Forest', fontsize=13, fontweight='bold')
        for bar, val in zip(bars, sorted_importance[::-1]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        print("  [artifact] feature_importance.png saved")

        # ── Export model ke folder model_artifact ──
        mlflow.sklearn.save_model(model, path='model_artifact')
        print("  [artifact] model_artifact/ saved")

        run_id = mlflow.active_run().info.run_id
        with open('run_id.txt', 'w') as f:
            f.write(run_id)
        print(f"  [artifact] run_id.txt saved ({run_id})")

    mlflow.sklearn.autolog(disable=True)
    return model, run_id


if __name__ == "__main__":
    print("=" * 55)
    print("  MODELLING CI - Moch Mizan Ghodafail")
    print("=" * 55)

    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"Data dimuat. Train: {X_train.shape} | Test: {X_test.shape}")

    model, run_id = train_random_forest(X_train, X_test, y_train, y_test, feature_cols)

    print("\n" + "=" * 55)
    print("  SELESAI! Model siap untuk Docker build.")
    print("=" * 55)