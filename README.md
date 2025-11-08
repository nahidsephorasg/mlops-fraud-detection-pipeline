# Real-Time Fraud Detection MLOps Pipeline

A complete end-to-end machine learning operations pipeline for detecting fraudulent transactions in real-time. This project demonstrates a production-ready MLOps setup with feature engineering, model training, experiment tracking, and deployment capabilities.

**Note**: This is a learning project to understand MLOps concepts and tools in practice.

## Project Overview

This system processes transaction data, engineers features, trains multiple machine learning models, and tracks experiments using industry-standard MLOps tools. The entire pipeline runs in Docker containers with separate services for data storage, orchestration, feature store, and experiment tracking.

## Architecture

The pipeline consists of four main components running in separate Docker Compose files:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network                               │
│                 (fraud-detection-network)                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   MongoDB    │    │    Redis     │    │    Feast     │     │
│  │   (27017)    │◄───┤    (6379)    │◄───┤   Server     │     │
│  │              │    │              │    │   (6566)     │     │
│  └──────┬───────┘    └──────────────┘    └──────────────┘     │
│         │                                                       │
│         │ Read/Write Features                                  │
│         │                                                       │
│  ┌──────▼───────────────────────────────────────────┐         │
│  │           Apache Airflow (8080)                   │         │
│  │  ┌────────────────┐  ┌────────────────┐         │         │
│  │  │ Preprocessing  │  │    Training    │         │         │
│  │  │   Pipeline     │─►│    Pipeline    │         │         │
│  │  └────────────────┘  └────────┬───────┘         │         │
│  └────────────────────────────────┼──────────────────┘         │
│                                   │                            │
│                                   │ Log Experiments            │
│                                   │                            │
│  ┌────────────────────────────────▼──────────────────┐        │
│  │         MLflow Server (5101)                       │        │
│  │  ┌──────────────┐  ┌────────────────────┐        │        │
│  │  │  Experiments │  │  Model Registry    │        │        │
│  │  │   Tracking   │  │  - XGBoost         │        │        │
│  │  │              │  │  - Isolation Forest│        │        │
│  │  └──────────────┘  └────────────────────┘        │        │
│  └───────────────────────────────────────────────────┘        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
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

All services communicate through a shared Docker network called `fraud-detection-network`.

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
```

### Access the UIs

- Airflow: http://localhost:8080 (admin/admin)
- MLflow: http://localhost:5101
- MongoDB Express: http://localhost:8081 (admin/admin123)
- Redis Insight: http://localhost:8001

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

- **Orchestration**: Apache Airflow 3.1.2
- **Experiment Tracking**: MLflow 3.6.0
- **Feature Store**: Feast 0.56.0
- **Data Storage**: MongoDB 8.2, Redis 7.4
- **Databases**: PostgreSQL 16
- **ML Libraries**: XGBoost, scikit-learn, pandas, numpy
- **Containerization**: Docker, Docker Compose

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
└── README.md
```

## Next Steps

Future enhancements for this pipeline:

1. **Model Deployment**: Create REST API endpoint for real-time predictions
2. **Monitoring**: Add data drift detection and model performance tracking
3. **Automated Retraining**: Schedule periodic model updates
4. **Model Serving**: Deploy best model using MLflow Model Serving
5. **CI/CD Integration**: Automate testing and deployment
6. **Feature Engineering**: Add more sophisticated feature transformations
7. **Hyperparameter Tuning**: Implement automated hyperparameter optimization
