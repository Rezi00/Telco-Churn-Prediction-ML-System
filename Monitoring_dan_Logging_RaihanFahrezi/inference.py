import time
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import psutil


MODEL_PATH = "best_random_forest_model.joblib"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Model serving untuk prediksi churn pelanggan telekomunikasi.",
    version="1.0.0",
)

REQUEST_COUNT = Counter(
    "telco_churn_request_total",
    "Total request yang masuk ke API Telco Churn",
    ["endpoint", "method", "status"]
)

PREDICTION_COUNT = Counter(
    "telco_churn_prediction_total",
    "Total prediksi berdasarkan kelas hasil prediksi",
    ["prediction_label"]
)

REQUEST_LATENCY = Histogram(
    "telco_churn_request_latency_seconds",
    "Latency request API dalam detik",
    ["endpoint"]
)

CPU_USAGE = Gauge(
    "system_cpu_usage_percent",
    "Persentase penggunaan CPU"
)

MEMORY_USAGE = Gauge(
    "system_memory_usage_percent",
    "Persentase penggunaan RAM"
)


class TelcoCustomer(BaseModel):
    gender: int = Field(..., example=1)
    SeniorCitizen: int = Field(..., example=0)
    Partner: int = Field(..., example=1)
    Dependents: int = Field(..., example=0)
    tenure: int = Field(..., example=12)
    PhoneService: int = Field(..., example=1)
    MultipleLines: int = Field(..., example=0)
    InternetService: int = Field(..., example=1)
    OnlineSecurity: int = Field(..., example=0)
    OnlineBackup: int = Field(..., example=1)
    DeviceProtection: int = Field(..., example=0)
    TechSupport: int = Field(..., example=0)
    StreamingTV: int = Field(..., example=1)
    StreamingMovies: int = Field(..., example=1)
    Contract: int = Field(..., example=0)
    PaperlessBilling: int = Field(..., example=1)
    PaymentMethod: int = Field(..., example=2)
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=1397.47)


class BatchTelcoCustomer(BaseModel):
    data: List[TelcoCustomer]


def update_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEMORY_USAGE.set(psutil.virtual_memory().percent)


@app.get("/")
def root():
    REQUEST_COUNT.labels(endpoint="/", method="GET", status="success").inc()
    return {
        "message": "Telco Customer Churn Prediction API is running",
        "features": FEATURE_COLUMNS
    }


@app.get("/health")
def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="success").inc()
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: TelcoCustomer):
    start_time = time.time()
    endpoint = "/predict"

    try:
        input_df = pd.DataFrame([customer.dict()])[FEATURE_COLUMNS]
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][prediction])
        else:
            probability = None

        label = "Churn" if prediction == 1 else "No Churn"

        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="success").inc()
        PREDICTION_COUNT.labels(prediction_label=label).inc()

        return {
            "prediction": prediction,
            "label": label,
            "probability": probability
        }

    except Exception as error:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="error").inc()
        return {
            "error": str(error)
        }

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


@app.post("/predict_batch")
def predict_batch(payload: BatchTelcoCustomer):
    start_time = time.time()
    endpoint = "/predict_batch"

    try:
        records = [item.dict() for item in payload.data]
        input_df = pd.DataFrame(records)[FEATURE_COLUMNS]
        predictions = model.predict(input_df).astype(int).tolist()

        results = []
        for pred in predictions:
            label = "Churn" if pred == 1 else "No Churn"
            PREDICTION_COUNT.labels(prediction_label=label).inc()
            results.append({
                "prediction": pred,
                "label": label
            })

        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="success").inc()

        return {
            "total_data": len(results),
            "results": results
        }

    except Exception as error:
        REQUEST_COUNT.labels(endpoint=endpoint, method="POST", status="error").inc()
        return {
            "error": str(error)
        }

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


@app.get("/metrics")
def metrics():
    update_system_metrics()
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )