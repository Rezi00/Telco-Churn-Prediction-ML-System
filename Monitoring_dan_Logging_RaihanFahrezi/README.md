# Monitoring dan Logging - Telco Customer Churn

Folder ini berisi implementasi Kriteria 4 proyek akhir **Membangun Sistem Machine Learning**.

Model yang digunakan:

```text
best_random_forest_model.joblib
```

API dibuat menggunakan FastAPI dan expose metrics Prometheus pada endpoint:

```text
/metrics
```

## Struktur

```text
Monitoring dan Logging
├── 1.bukti_serving
├── 2.prometheus.yml
├── 3.prometheus_exporter.py
├── 4.bukti monitoring Prometheus
├── 5.bukti monitoring Grafana
├── 6.bukti alerting Grafana
├── 7.inference.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── best_random_forest_model.joblib
```

## Cara Menjalankan Tanpa Docker

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan API:

```bash
uvicorn inference:app --host 0.0.0.0 --port 8000
```

Buka:

```text
http://localhost:8000/docs
http://localhost:8000/metrics
```

## Cara Menjalankan Dengan Docker Compose

```bash
docker compose up --build
```

Akses:

```text
FastAPI     : http://localhost:8000/docs
Prometheus  : http://localhost:9090
Grafana     : http://localhost:3000
```

Login Grafana default:

```text
username: admin
password: admin
```

## Metrik Prometheus

Minimal metrik yang bisa dipakai:

```text
telco_churn_request_total
telco_churn_prediction_total
telco_churn_request_latency_seconds_count
telco_churn_request_latency_seconds_sum
system_cpu_usage_percent
system_memory_usage_percent
```

## Query Prometheus/Grafana

Total request:

```promql
sum(telco_churn_request_total)
```

Request error:

```promql
sum(telco_churn_request_total{status="error"})
```

Rata-rata latency:

```promql
rate(telco_churn_request_latency_seconds_sum[1m]) / rate(telco_churn_request_latency_seconds_count[1m])
```

CPU usage:

```promql
system_cpu_usage_percent
```

Memory usage:

```promql
system_memory_usage_percent
```

Prediction count:

```promql
sum by (prediction_label) (telco_churn_prediction_total)
```

## Contoh Payload

```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "MonthlyCharges": 70.35,
  "TotalCharges": 1397.47
}
```

## Bukti Screenshot yang Perlu Diambil

1. `1.bukti_serving`
   - FastAPI `/docs`
   - Hasil `/predict`
   - Endpoint `/metrics`

2. `4.bukti monitoring Prometheus`
   - Target Prometheus UP
   - Query total request
   - Query latency
   - Query CPU/RAM

3. `5.bukti monitoring Grafana`
   - Dashboard dengan minimal 5 metrik:
     - total request
     - error request
     - latency
     - CPU usage
     - memory usage

4. `6.bukti alerting Grafana`
   - Rule alerting
   - Bukti status alert/rule
