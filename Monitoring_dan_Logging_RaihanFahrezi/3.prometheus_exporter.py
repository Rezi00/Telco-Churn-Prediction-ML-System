import random
import time
import requests

API_URL = "http://localhost:8000/predict"

sample_payloads = [
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
    },
    {
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 1,
        "tenure": 48,
        "PhoneService": 1,
        "MultipleLines": 1,
        "InternetService": 0,
        "OnlineSecurity": 2,
        "OnlineBackup": 2,
        "DeviceProtection": 2,
        "TechSupport": 2,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "Contract": 2,
        "PaperlessBilling": 0,
        "PaymentMethod": 1,
        "MonthlyCharges": 55.20,
        "TotalCharges": 2600.80
    }
]


def send_requests(total_requests: int = 100, sleep_time: float = 0.2):
    for i in range(total_requests):
        payload = random.choice(sample_payloads)
        response = requests.post(API_URL, json=payload, timeout=10)
        print(i + 1, response.status_code, response.json())
        time.sleep(sleep_time)


if __name__ == "__main__":
    send_requests()