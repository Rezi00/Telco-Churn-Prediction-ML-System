import os
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore")


DATA_PATH = "telco_churn_preprocessed.csv"
TARGET_COLUMN = "Churn"
EXPERIMENT_NAME = "Telco Churn Classification - Manual Logging Tuning"
ARTIFACT_DIR = "artifacts"


def load_dataset(path: str = DATA_PATH):
    """Load preprocessed Telco Customer Churn dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan pada dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def create_confusion_matrix_artifact(y_true, y_pred, output_path):
    """Create and save confusion matrix plot as artifact."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix - Telco Churn")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_classification_report_artifact(y_true, y_pred, output_path):
    """Create and save classification report text file as artifact."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Classification Report - Telco Customer Churn\n")
        file.write("=" * 55)
        file.write("\n\n")
        file.write(report)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    report_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
    model_joblib_path = os.path.join(ARTIFACT_DIR, "best_random_forest_model.joblib")

    create_confusion_matrix_artifact(y_test, y_pred, cm_path)
    create_classification_report_artifact(y_test, y_pred, report_path)
    joblib.dump(best_model, model_joblib_path)

    input_example = X_train.head(5)

    with mlflow.start_run(run_name="random_forest_gridsearch_manual_logging"):
        # Manual logging: parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target_column", TARGET_COLUMN)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("cv", 3)

        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)

        # Manual logging: metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Manual logging: artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(model_joblib_path)

        # Manual logging: model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example
        )

    print("Hyperparameter tuning selesai.")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("Artifact tersimpan pada folder artifacts dan MLflow Tracking.")
    print("Jalankan: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()