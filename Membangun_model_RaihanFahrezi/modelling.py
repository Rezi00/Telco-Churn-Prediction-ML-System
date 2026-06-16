import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


DATA_PATH = "telco_churn_preprocessed.csv"
TARGET_COLUMN = "Churn"
EXPERIMENT_NAME = "Telco Churn Classification - Basic Autolog"


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


def main():
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

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    input_example = X_train.head(5)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="random_forest_basic_autolog"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy :", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, zero_division=0))
        print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
        print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))

        # Explicit model logging to make sure model artifact is available
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

    print("Training selesai. Jalankan: mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()