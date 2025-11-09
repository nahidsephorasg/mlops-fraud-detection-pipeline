# Fraud Detection API

REST API for real-time fraud detection using MLflow-managed machine learning models.

## Features

- **Real-time Predictions**: Single transaction fraud detection
- **Batch Processing**: Multiple transactions in one request
- **MLflow Integration**: Automatically loads latest model from Model Registry
- **Model Management**: Hot reload without downtime
- **Health Monitoring**: Built-in health checks and model info endpoints
- **Risk Levels**: Categorizes transactions as LOW, MEDIUM, HIGH, or CRITICAL risk

## Quick Start

### Start the API

```bash
# Build and start the API service
docker compose -f docker-compose.api.yaml up -d

# Check logs
docker logs fraud-detection-api

# Wait for "API ready to serve predictions" message
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)

## API Endpoints

### Health & Info

#### GET `/health`
Check API health and model status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "name": "fraud_detection_xgboost",
    "version": "5",
    "stage": "None",
    "run_id": "f4b283a0a90146ac9d5ea3328e8a0413"
  },
  "timestamp": "2025-11-08T10:30:00"
}
```

#### GET `/model/info`
Get current model details

**Response:**
```json
{
  "model_name": "fraud_detection_xgboost",
  "model_version": "5",
  "model_stage": "None",
  "run_id": "f4b283a0a90146ac9d5ea3328e8a0413",
  "mlflow_uri": "http://mlflow-server:5101"
}
```

#### POST `/model/reload`
Reload model from registry (useful after retraining)

### Predictions

#### POST `/predict`
Predict fraud for a single transaction

**Request Body:**
```json
{
  "TransactionAmt": 150.50,
  "card1": 13926,
  "card2": 150.0,
  "card3": 150.0,
  "card4": "visa",
  "card5": 226.0,
  "card6": "credit",
  "addr1": 315.0,
  "addr2": 87.0,
  "ProductCD": "W",
  "P_emaildomain": "gmail.com",
  "R_emaildomain": "gmail.com"
}
```

**Response:**
```json
{
  "transaction_id": "TXN_20251108103000123456",
  "is_fraud": false,
  "fraud_probability": 0.15,
  "risk_level": "LOW",
  "model_name": "fraud_detection_xgboost",
  "model_version": "5",
  "timestamp": "2025-11-08T10:30:00"
}
```

#### POST `/predict/batch`
Predict fraud for multiple transactions

**Request Body:**
```json
{
  "transactions": [
    {
      "TransactionAmt": 50.00,
      "card1": 13926,
      ...
    },
    {
      "TransactionAmt": 9999.99,
      "card1": 1234,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "transaction_id": "TXN_20251108103000123456",
      "is_fraud": false,
      "fraud_probability": 0.12,
      "risk_level": "LOW",
      ...
    },
    {
      "transaction_id": "TXN_20251108103000123457",
      "is_fraud": true,
      "fraud_probability": 0.89,
      "risk_level": "CRITICAL",
      ...
    }
  ],
  "total_transactions": 2,
  "fraud_count": 1,
  "model_info": {...}
}
```

## Testing

### Using the Test Script

```bash
# Run all tests
python test_api.py
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 150.50,
    "card1": 13926,
    "card2": 150.0,
    "card3": 150.0,
    "card4": "visa",
    "card5": 226.0,
    "card6": "credit",
    "addr1": 315.0,
    "addr2": 87.0,
    "ProductCD": "W",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com"
  }'
```

### Using Python

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "TransactionAmt": 150.50,
        "card1": 13926,
        "card2": 150.0,
        "card3": 150.0,
        "card4": "visa",
        "card5": 226.0,
        "card6": "credit",
        "addr1": 315.0,
        "addr2": 87.0,
        "ProductCD": "W",
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com"
    }
)

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

## Risk Levels

| Fraud Probability | Risk Level |
|------------------|------------|
| < 30% | LOW |
| 30% - 60% | MEDIUM |
| 60% - 80% | HIGH |
| > 80% | CRITICAL |

## Configuration

Environment variables in `docker-compose.api.yaml`:

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://mlflow-server:5101)
- `MODEL_NAME`: Model name in MLflow registry (default: fraud_detection_xgboost)
- `MODEL_STAGE`: Model stage to use - None, Staging, or Production (default: None = latest version)
- `PREPROCESSING_SERVICE_URL`: URL of the preprocessing service (default: http://preprocessing-service:8001)
- `SENTRY_DSN`: Sentry error tracking DSN (default: None = disabled)
- `ENVIRONMENT`: Environment name for Sentry (default: production)
- `SENTRY_TRACES_SAMPLE_RATE`: Trace sampling rate for Sentry (default: 1.0 = 100%)
- `SENTRY_PROFILES_SAMPLE_RATE`: Profiling sampling rate for Sentry (default: 1.0 = 100%)
- `APP_VERSION`: Application version for Sentry (default: 1.0.0)
- `LOG_LEVEL`: Logging level (default: info)
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OpenTelemetry exporter endpoint (default: http://opentelemetry-collector:4317)
- `OTEL_SERVICE_NAME`: Service name for OpenTelemetry (default: fraud-detection-api)


## Model Versioning

The API automatically loads the latest model version from MLflow. To use a specific stage:

```yaml
environment:
  MODEL_STAGE: Production 
```

After training a new model:
1. Transition model to desired stage in MLflow UI
2. Call `/model/reload` endpoint or restart the API

## Performance

- **Latency**: < 50ms for single prediction
- **Throughput**: ~200 requests/second
- **Batch Size**: Up to 1000 transactions per batch request

## Architecture

```
Client Request
     │
     ▼
┌──────────────────┐
│   FastAPI App    │
│   (Port 8000)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Service   │
│  (Load/Predict)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MLflow Server   │
│  (Port 5101)     │
│  Model Registry  │
└──────────────────┘
```

## Troubleshooting

### API won't start

```bash
# Check logs
docker logs fraud-detection-api

# Verify MLflow is running
curl http://localhost:5101/health

# Verify network
docker network inspect fraud-detection-network
```

### Model loading fails

```bash
# Check model exists in MLflow
curl http://localhost:5101/api/2.0/mlflow/registered-models/get?name=fraud_detection_xgboost

# Verify model has versions
curl http://localhost:5101/api/2.0/mlflow/registered-models/search
```

### Predictions fail

- Ensure all required features are provided
- Check feature types match training data
- Review API logs for detailed error messages

