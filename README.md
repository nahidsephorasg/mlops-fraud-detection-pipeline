# Real-Time Fraud Detection MLOps Pipeline

A complete end-to-end machine learning operations pipeline for detecting fraudulent transactions in real-time. This project demonstrates a production-ready MLOps setup with feature engineering, model training, experiment tracking, model serving, and comprehensive observability.

Note: This is a learning project to understand MLOps concepts and tools in practice.

## Project Overview

This system processes transaction data, engineers features, trains machine learning models, tracks experiments, serves predictions via REST API, and monitors the entire pipeline with distributed tracing and error tracking. The architecture follows microservices patterns with separate services for data storage, orchestration, feature engineering, model serving, and observability.

## Architecture

The pipeline consists of six main components running in separate Docker Compose files:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Docker Network                                  │
│                      (fraud-detection-network)                           │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │   MongoDB    │    │    Redis     │    │    Feast     │             │
│  │   (27017)    │◄───┤    (6379)    │◄───┤   Server     │             │
│  │              │    │              │    │   (6566)     │             │
│  └──────┬───────┘    └──────────────┘    └──────────────┘             │
│         │                                                               │
│         │ Read/Write Features                                          │
│         │                                                               │
│  ┌──────▼────────────────────────────────────────────┐                │
│  │           Apache Airflow (8080)                    │                │
│  │  ┌────────────────┐  ┌────────────────┐          │                │
│  │  │ Preprocessing  │  │    Training    │          │                │
│  │  │   Pipeline     │─►│    Pipeline    │          │                │
│  │  └────────────────┘  └────────┬───────┘          │                │
│  └────────────────────────────────┼───────────────────┘                │
│                                   │                                    │
│                                   │ Log Experiments                    │
│                                   │                                    │
│  ┌────────────────────────────────▼────────────────┐                  │
│  │         MLflow Server (5101)                     │                  │
│  │  ┌──────────────┐  ┌──────────────────┐         │                  │
│  │  │  Experiments │  │  Model Registry  │         │                  │
│  │  │   Tracking   │  │  - XGBoost       │         │                  │
│  │  │              │  │  - Isolation     │         │                  │
│  │  └──────────────┘  └────────┬─────────┘         │                  │
│  └───────────────────────────────┼───────────────────┘                  │
│                                  │                                     │
│                                  │ Load Model                          │
│                                  │                                     │
│  ┌───────────────────────────────▼──────────────────────────────┐    │
│  │              Microservices Layer                              │    │
│  │                                                                │    │
│  │  ┌─────────────────────┐    ┌──────────────────────┐        │    │
│  │  │  API Service        │───►│ Preprocessing        │        │    │
│  │  │  (8000)             │    │ Service (8031)       │        │    │
│  │  │                     │    │                      │        │    │
│  │  │ - Model Inference   │    │ - Feature Engineering│        │    │
│  │  │ - Request Handling  │    │ - Training Parity    │        │    │
│  │  │ - Risk Assessment   │    │ - 130 Features       │        │    │
│  │  └─────────┬───────────┘    └──────────────────────┘        │    │
│  └────────────┼───────────────────────────────────────────────────┘    │
│               │                                                       │
│               │ Send Traces & Errors                                  │
│               │                                                       │
│  ┌────────────▼──────────────────────────────────────────────────┐  │
│  │              Observability Stack                               │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │  │
│  │  │ Prometheus   │  │   Grafana    │  │     Loki     │        │  │
│  │  │   (9090)     │  │   (3000)     │  │    (3100)    │        │  │
│  │  │              │  │              │  │              │        │  │
│  │  │  - Metrics   │  │ - Dashboards │  │ - Logs       │        │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │  │
│  │                                                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐                           │  │
│  │  │    Tempo     │  │    Sentry    │                           │  │
│  │  │   (3200)     │  │   (Cloud)    │                           │  │
│  │  │              │  │              │                           │  │
│  │  │  - Traces    │  │ - Errors     │                           │  │
│  │  └──────────────┘  └──────────────┘                           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**Component Breakdown:**

1. **Data Infrastructure** (docker-compose.yaml)
   - MongoDB for raw and processed data storage
   - Redis for online feature serving
   - Feast feature server for feature management
   - Web UIs for MongoDB (port 8081) and Redis (port 8001)

2. **Workflow Orchestration** (docker-compose.airflow.yaml)
   - Apache Airflow 3.1.2 in standalone mode
   - PostgreSQL backend for Airflow metadata
   - LocalExecutor for task execution
   - Airflow UI on port 8080

3. **Experiment Tracking** (docker-compose.mlflow.yaml)
   - MLflow server for experiment tracking and model registry
   - PostgreSQL backend for MLflow metadata
   - Local artifact storage
   - MLflow UI on port 5101

4. **Microservices Layer**
   - **API Service** (docker-compose.api.yaml) - Model serving and prediction endpoint
   - **Preprocessing Service** (docker-compose.preprocessing.yaml) - Feature engineering service

5. **Monitoring Stack** (docker-compose.monitoring.yaml)
   - **Prometheus** (9090) - Metrics collection and storage
   - **Grafana** (3000) - Visualization and dashboards
   - **Loki** (3100) - Log aggregation and querying
   - **Tempo** (3200) - Distributed tracing backend
   - **Promtail** - Log shipping agent

6. **Error Monitoring**
   - **Sentry** (Cloud or Self-hosted) - Error tracking and performance monitoring

All services communicate through a shared Docker network called fraud-detection-network.

## Setup and Installation

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available
- Transaction dataset (IEEE-CIS Fraud Detection from Kaggle)

### Start the Services

Start each component in order:

```bash
# 1. Start data infrastructure
docker-compose up -d

# 2. Start Airflow
docker-compose -f docker-compose.airflow.yaml up -d

# 3. Start MLflow
docker-compose -f docker-compose.mlflow.yaml up -d

# 4. Start monitoring stack
docker-compose -f docker-compose.monitoring.yaml up -d

# 5. Start preprocessing service
docker-compose -f docker-compose.preprocessing.yaml up -d

# 6. Start API service (after training models)
docker-compose -f docker-compose.api.yaml up -d
```

### Access the UIs

**Core Services:**
- Airflow: http://localhost:8080 (admin/admin)
- MLflow: http://localhost:5101
- MongoDB Express: http://localhost:8081 (admin/admin123)
- Redis Insight: http://localhost:8001

**API Services:**
- Fraud Detection API: http://localhost:8000 (docs at /docs)
- Preprocessing Service: http://localhost:8031 (docs at /docs)

**Monitoring:**
- Grafana: http://localhost:3000 (admin/admin123)
- Prometheus: http://localhost:9090

## Data Pipeline

### 1. Preprocessing Pipeline

The preprocessing pipeline (`preprocessing_pipeline.py`) performs feature engineering on transaction data:

- Loads raw transaction and identity data from MongoDB
- Normalizes D-columns (time deltas)
- Creates frequency encodings for categorical variables
- Applies label encoding
- Generates aggregation features (count, sum, mean, std, min, max)
- Stores processed features back to MongoDB
- Saves feature metadata to Feast

The pipeline processes approximately 1000 transactions and creates 20+ features per transaction.

### 2. Training Pipeline

The training pipeline (`training.py`) trains multiple models and tracks experiments:

**Models Trained:**
- XGBoost Classifier
- Isolation Forest (anomaly detection)
- LightGBM (optional, skipped if unavailable)

**Pipeline Steps:**
1. Test connections to Feast and MLflow
2. Load processed features from MongoDB
3. Prepare training/validation split (80/20)
4. Train each model independently
5. Log parameters, metrics, and model artifacts to MLflow
6. Register models in MLflow Model Registry
7. Select best model based on F1-score
8. Save models locally

**Metrics Tracked:**
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1-Score

## MLflow Configuration

MLflow 3.6.0 includes security middleware that blocks requests from Docker internal hostnames. To allow Airflow to communicate with MLflow, we disable the security middleware:

```yaml
environment:
  MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE: "true"
```

This is appropriate for local development and internal networks. For production, configure proper host validation using `--allowed-hosts`.

## Model Artifacts

Model artifacts are stored in two locations:

1. **MLflow Artifact Store**: `./mlflow_artifacts/` directory
   - Organized by experiment ID and run ID
   - Contains model files, metadata, and requirements

2. **Airflow Container**: `/opt/airflow/fraud_detection_pipeline/models/`
   - Local copies of trained models
   - Used for deployment and inference

## Registered Models

Models are automatically registered in MLflow Model Registry with these names:
- `fraud_detection_xgboost`
- `fraud_detection_isolation_forest`

View registered models in MLflow UI under the "Models" tab. Each model version includes:
- Training metrics
- Model parameters
- Artifact location
- Source run information
- Stage (None/Staging/Production)

## Running the Pipeline

### Execute Preprocessing

1. Go to Airflow UI (http://localhost:8080)
2. Find `preprocessing_pipeline_simple_feast` DAG
3. Unpause the DAG (toggle switch)
4. Click the play button to trigger
5. Monitor task execution (12 tasks total)
6. Verify completion in logs and MongoDB

### Execute Training

1. Ensure preprocessing has completed successfully
2. Find `training_pipeline` DAG in Airflow UI
3. Unpause and trigger the DAG
4. Monitor training progress (8 tasks)
5. Check MLflow UI for experiment results
6. Review registered models

## Troubleshooting

### MLflow Connection Issues

If Airflow cannot connect to MLflow:
- Verify all containers are on `fraud-detection-network`
- Check MLflow logs: `docker logs mlflow-server`
- Ensure security middleware is disabled
- Test connectivity: `docker exec airflow-standalone curl http://mlflow-server:5101/health`

### Airflow DAG Not Loading

If DAGs don't appear in Airflow:
- Check DAG syntax for Python errors
- Review Airflow logs: `docker logs airflow-standalone`
- Verify DAG file is in `./dags/` directory
- Wait for Airflow to parse DAGs (30-60 seconds)

### Missing Dependencies

If tasks fail with import errors:
- Check `_PIP_ADDITIONAL_REQUIREMENTS` in docker-compose.airflow.yaml
- Restart Airflow after adding packages
- LightGBM is optional and gracefully skipped if unavailable

## Technology Stack

**Orchestration**: Apache Airflow 3.1.2  
**Experiment Tracking**: MLflow 3.6.0  
**Feature Store**: Feast 0.56.0  
**Data Storage**: MongoDB 8.2, Redis
**Databases**: PostgreSQL 16  
**API Framework**: FastAPI  
**ML Libraries**: XGBoost, Isolation Forest, scikit-learn  
**Monitoring**: Prometheus, Grafana, Loki, Tempo  
**Distributed Tracing**: OpenTelemetry  
**Error Monitoring**: Sentry  
**Containerization**: Docker, Docker Compose

## Project Structure

```
e2e-real-time-fraud-detection/
├── dags/
│   ├── preprocessing_pipeline.py
│   ├── training.py
│   └── sample_dag.py
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── feature_store/
│   └── feature_repo/
├── mlflow_artifacts/
├── logs/
├── docker-compose.yaml
├── docker-compose.airflow.yaml
├── docker-compose.mlflow.yaml
├── docker-compose.monitoring.yaml
├── docker-compose.preprocessing.yaml
├── docker-compose.api.yaml
└── README.md
```

## Model Deployment API

The fraud detection system uses a microservices architecture to ensure feature engineering consistency between training and serving.

### Architecture: Why Microservices?

**The Problem:**
- ML models were trained on 130 engineered features (frequency encodings, aggregations, V-columns, C-columns)
- Raw transaction data has only 12 input fields
- Direct prediction fails due to feature mismatch

**The Solution:**
We implemented a preprocessing microservice that applies the same feature engineering used during training, ensuring training-serving parity.

```
┌─────────────────────────────────────────────────────────────┐
│              MICROSERVICES ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Client (e.g., test_api.py, mobile app, web frontend)
    │
    │ POST /predict
    │ Raw Transaction Data (12 fields):
    │ {TransactionAmt, card1, card2, card3, card4, card5, 
    │  card6, addr1, addr2, ProductCD, P_emaildomain, R_emaildomain}
    │
    ▼
┌─────────────────────────────────┐
│  Fraud Detection API            │  http://localhost:8000
│  (Port 8000)                    │
│                                 │
│  Responsibilities:              │
│  • Load model from MLflow       │
│  • Manage predictions           │
│  • Handle API requests          │
│  • Return fraud risk assessment │
└─────────────┬───────────────────┘
              │
              │ HTTP POST /engineer
              │ Send raw transaction data
              │
              ▼
┌─────────────────────────────────┐
│  Preprocessing Service          │  http://localhost:8031
│  (Port 8031)                    │
│                                 │
│  Responsibilities:              │
│  • Frequency encoding           │
│  • Aggregation features         │
│  • Combination features         │
│  • Add V & C columns            │
│  • Return 130 features          │
└─────────────┬───────────────────┘
              │
              │ Response: 130 Engineered Features
              │ {TransactionID, TransactionAmt, card1, card3,
              │  C1-C14, V95-V137, V279-V321, card1_freq,
              │  card1_count, TransactionCents, ...}
              │
              ▼
┌─────────────────────────────────┐
│  Fraud Detection API            │
│                                 │
│  • Receives 130 features        │
│  • Passes to XGBoost model      │
│  • Gets prediction              │
└─────────────┬───────────────────┘
              │
              │ Prediction Response
              │
              ▼
┌─────────────────────────────────┐
│  Client                         │
│                                 │
│  Receives:                      │
│  {                              │
│    "is_fraud": false,           │
│    "fraud_probability": 0.15,   │
│    "risk_level": "LOW",         │
│    "transaction_id": "TXN_...", │
│    "model_version": "6"         │
│  }                              │
└─────────────────────────────────┘
```

### Benefits of This Architecture

**Training/Serving Parity**: Identical feature engineering in training and production  
**Separation of Concerns**: Feature logic isolated from prediction logic  
**Reusability**: Other services can use the same preprocessing endpoint  
**Independent Scaling**: Scale preprocessing and prediction services separately  
**Maintainability**: Update feature engineering without touching prediction code  
**MLOps Best Practice**: Industry-standard pattern for production ML systems

### Quick Start

```bash
# 1. Start preprocessing service (feature engineering)
docker-compose -f docker-compose.preprocessing.yaml up -d

# 2. Start prediction API (model serving)
docker-compose -f docker-compose.api.yaml up -d

# 3. Test the complete pipeline
python test_api.py

# 4. View API documentation
open http://localhost:8000/docs        # Prediction API
open http://localhost:8031/docs        # Preprocessing API
```

### Example Usage

```python
import requests

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

See `api/README.md` for complete API documentation.

## Monitoring and Observability

The system includes comprehensive observability with metrics, logs, traces, and error monitoring.

### Components

**1. Metrics (Prometheus + Grafana)**
- Request rate and latency (P50, P95, P99)
- Error rates (4xx, 5xx)
- Fraud detection rate
- Feature engineering performance
- Model inference timing

**2. Logs (Loki + Promtail + Grafana)**
- Structured application logs
- Request and response logging
- Trace ID correlation
- Log aggregation and search

**3. Distributed Tracing (OpenTelemetry + Tempo + Grafana)**
- End-to-end request flow visualization
- Service dependency mapping
- Performance bottleneck identification
- Automatic context propagation between services

**4. Error Monitoring (Sentry)**
- Real-time error tracking
- Stack trace capture
- Performance profiling
- Email and Slack alerts

### Quick Start

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yaml up -d

# Access dashboards
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:9090  # Prometheus
```

### Pre-configured Dashboards

**Fraud Detection API**
- Real-time request monitoring
- Latency percentiles
- Fraud rate tracking
- Risk distribution

**Distributed Tracing**
- Recent traces timeline
- Error traces filtered view
- Slow traces (over 500ms)
- Service dependency graph

**Logs**
- Live log streaming
- Log rate by level
- Error log filtering
- Trace-to-log correlation

### Key Features

**Trace-Log-Metric Correlation**
- Every request gets a unique trace ID
- Trace ID appears in all logs for that request
- Click any log to see full trace
- Click any trace to see related logs
- Link metrics spikes to specific traces

**Example: Debugging a Slow Request**
1. Notice latency spike in Prometheus metrics
2. Search Tempo for slow traces (duration over 500ms)
3. Click trace to see timing breakdown
4. Identify bottleneck (model inference took 380ms)
5. View correlated logs for that trace
6. Fix issue and verify with new traces

### Documentation

- LOGGING.md - Complete logging guide
- TRACING.md - Distributed tracing and Sentry setup
- OBSERVABILITY_SUMMARY.md - Quick reference

## What This Project Implements

This project demonstrates a complete MLOps pipeline with:

**Data Engineering**
- Feature engineering with frequency encodings, aggregations, and derived features
- Feature store integration with Feast
- MongoDB for data persistence
- Redis for online feature serving

**Model Training**
- Automated training pipelines with Apache Airflow
- Multiple model training (XGBoost, Isolation Forest)
- Experiment tracking with MLflow
- Model registry and versioning

**Model Serving**
- Microservices architecture for prediction and feature engineering
- REST API with FastAPI
- Training-serving parity through shared feature engineering
- Real-time fraud risk assessment

**Observability**
- Metrics collection with Prometheus
- Distributed tracing with OpenTelemetry and Tempo
- Log aggregation with Loki and Promtail
- Error monitoring with Sentry
- Unified visualization in Grafana
- Trace-log-metric correlation

**Infrastructure**
- Containerized deployment with Docker Compose
- Service orchestration across multiple compose files
- Shared networking for service communication
- Health checks and automated restarts

