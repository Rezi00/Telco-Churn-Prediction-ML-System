# Telco Churn Prediction - Machine Learning System

This project is an end-to-end Machine Learning system for predicting customer churn using the Telco Customer Churn dataset. The project covers data preprocessing, model training, experiment tracking, deployment, monitoring, and alerting.

## Project Overview

Customer churn prediction helps identify customers who are likely to stop using a service. By detecting potential churn early, businesses can take preventive actions to improve customer retention.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-Learn
* MLflow
* FastAPI
* Prometheus
* Grafana
* Joblib
* GitHub Actions

## Machine Learning Workflow

* Data preprocessing and feature engineering
* Model training using Random Forest Classifier
* Model evaluation using Accuracy, Precision, Recall, and F1-Score
* Experiment tracking with MLflow
* Model artifact logging using Joblib
* Model deployment through API endpoint
* API monitoring using Prometheus
* Dashboard visualization using Grafana
* Alerting setup for service monitoring

## Model

The main model used in this project is:

* Random Forest Classifier

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

## API Endpoint

Prediction endpoint:

```text
POST /predict
```

## Monitoring

The system includes monitoring for API requests and service metrics using Prometheus. Metrics can be visualized through Grafana dashboards.

## Alerting

Alerting is configured to notify when specific monitoring conditions are met, such as abnormal request behavior or service-related issues.

## Author

Raihan Fahrezi


## Project Structure

```text
SMSML_RaihanFahrezi/
├── Membangun_model_RaihanFahrezi/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── telco_churn_preprocessed.csv
│   ├── artifacts/
│   └── README.md
│
├── Monitoring_dan_Logging_RaihanFahrezi/
│   ├── inference.py
│   ├── prometheus_exporter.py
│   ├── prometheus.yml
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
│
├── Eksperimen_SML_Raihan Fahrezi.txt
└── Workflow-CI.txt
