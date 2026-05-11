"""
modelling.py  –  Digunakan oleh MLflow Project (Kriteria 3)
Melatih Random Forest + GridSearchCV dan menyimpan model artifact.

File ini dipanggil oleh: mlflow run MLProject/ -e main
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(SCRIPT_DIR, 'iris_preprocessing', 'train.csv')
TEST_PATH   = os.path.join(SCRIPT_DIR, 'iris_preprocessing', 'test.csv')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'model_artifact')
TARGET_COL  = 'species'
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

EXPERIMENT  = 'Iris_CI_Pipeline'
mlflow.set_experiment(EXPERIMENT)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    feat  = [c for c in train.columns if c != TARGET_COL]
    return train[feat], test[feat], train[TARGET_COL], test[TARGET_COL], feat


def make_confusion_matrix_plot(y_true, y_pred) -> str:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def make_feature_importance_plot(model, feature_cols) -> str:
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feature_cols, importances, color='#2196F3')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Score')
    plt.tight_layout()
    path = os.path.join(SCRIPT_DIR, 'feature_importance.png')
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("=" * 50)
    print("  CI PIPELINE - iris_classifier")
    print("=" * 50)

    X_train, X_test, y_train, y_test, feature_cols = load_data()
    print(f"Data loaded. Train: {X_train.shape} | Test: {X_test.shape}")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth':    [None, 5, 10],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred     = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')

    print(f"Best params: {grid_search.best_params_}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    with mlflow.start_run(run_name="CI_RandomForest"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('accuracy',  acc)
        mlflow.log_metric('f1_score',  f1)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall',    rec)

        # Artifacts
        cm_path = make_confusion_matrix_plot(y_test, y_pred)
        fi_path = make_feature_importance_plot(best_model, feature_cols)
        mlflow.log_artifact(cm_path, 'plots')
        mlflow.log_artifact(fi_path, 'plots')

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path='model')

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")

    # Simpan model ke path tetap (untuk Docker build)
    mlflow.sklearn.save_model(best_model, path=MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    # Simpan run_id untuk digunakan CI
    with open(os.path.join(SCRIPT_DIR, 'run_id.txt'), 'w') as f:
        f.write(run_id)

    print("=" * 50)
    print("  TRAINING SELESAI!")
    print("=" * 50)