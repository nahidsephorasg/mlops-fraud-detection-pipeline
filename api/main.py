from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import requests
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
from logging_config import setup_logger
from tracing_config import setup_tracing
from sentry_config import setup_sentry
from opentelemetry import trace
from drift_monitor import DriftMonitor
from performance_monitor import PerformanceMonitor

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/api.log")  # Use local path for development
logger = setup_logger(
    name="fraud_detection_api", log_file=LOG_FILE, level=getattr(logging, LOG_LEVEL)
)

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Time spent processing prediction",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
)

fraud_predictions_total = Counter(
    "fraud_predictions_total",
    "Total fraud predictions made",
    ["prediction", "risk_level"],
)

fraud_probability_gauge = Gauge(
    "fraud_probability_latest", "Latest fraud probability prediction"
)

model_info_gauge = Gauge(
    "model_info", "Model information", ["model_name", "model_version"]
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5101")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_detection_xgboost")
MODEL_STAGE = os.getenv("MODEL_STAGE", "None")  # None, Staging, Production
PREPROCESSING_SERVICE_URL = os.getenv(
    "PREPROCESSING_SERVICE_URL", "http://preprocessing-service:8001"
)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using MLflow-managed models",
    version="1.0.0",
)

tracer = setup_tracing(app, service_name="fraud-detection-api")
setup_sentry(service_name="fraud-detection-api")

logger.info("OpenTelemetry tracing configured with Tempo backend")
logger.info("Sentry error monitoring initialized")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses"""
    start_time = time.time()

    current_span = trace.get_current_span()
    trace_id = (
        format(current_span.get_span_context().trace_id, "032x")
        if current_span
        else "no-trace"
    )

    logger.info(
        f"Request: {request.method} {request.url.path} from {request.client.host} [trace_id={trace_id}]"
    )

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Duration: {duration:.3f}s [trace_id={trace_id}]"
    )

    return response


model = None
model_info = {}

# Initialize monitoring
drift_monitor = DriftMonitor()
performance_monitor = PerformanceMonitor()


class TransactionFeatures(BaseModel):
    """Transaction features for prediction"""

    TransactionAmt: float = Field(..., description="Transaction amount")
    card1: Optional[float] = Field(None, description="Card feature 1")
    card2: Optional[float] = Field(None, description="Card feature 2")
    card3: Optional[float] = Field(None, description="Card feature 3")
    card4: Optional[str] = Field(None, description="Card feature 4")
    card5: Optional[float] = Field(None, description="Card feature 5")
    card6: Optional[str] = Field(None, description="Card feature 6")
    addr1: Optional[float] = Field(None, description="Address 1")
    addr2: Optional[float] = Field(None, description="Address 2")
    ProductCD: Optional[str] = Field(None, description="Product code")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")

    class Config:
        schema_extra = {
            "example": {
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
                "R_emaildomain": "gmail.com",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_name: str
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    transactions: List[TransactionFeatures]


class GroundTruthLabel(BaseModel):
    """Ground truth label for a prediction"""

    transaction_id: str = Field(..., description="Transaction ID from prediction")
    is_fraud: bool = Field(..., description="True fraud label")
    timestamp: Optional[str] = Field(None, description="Label timestamp (ISO format)")


class ReferenceDataRequest(BaseModel):
    """Request to set reference data for drift detection"""

    data: List[Dict[str, Any]] = Field(..., description="Reference data samples")


def load_model_from_registry():
    """Load the latest model from MLflow Model Registry"""
    global model, model_info

    try:
        logger.info(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        logger.info(f"Loading model: {MODEL_NAME}, stage: {MODEL_STAGE}")

        if MODEL_STAGE == "None":
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            if not versions:
                raise Exception(f"No versions found for model {MODEL_NAME}")

            latest_version = max(versions, key=lambda x: int(x.version))
            model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
            model_info = {
                "name": MODEL_NAME,
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
            }
        else:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            version = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
            model_info = {
                "name": MODEL_NAME,
                "version": version.version,
                "stage": MODEL_STAGE,
                "run_id": version.run_id,
            }

        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"✓ Model loaded successfully: {model_info}")
        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Fraud Detection API...")
    load_model_from_registry()

    if model_info:
        model_info_gauge.labels(
            model_name=model_info.get("name", "unknown"),
            model_version=model_info.get("version", "unknown"),
        ).set(1)

    logger.info("API ready to serve predictions")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "Fraud Detection API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": model_info,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_information():
    """Get current model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_info.get("name"),
        "model_version": model_info.get("version"),
        "model_stage": model_info.get("stage"),
        "run_id": model_info.get("run_id"),
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/sentry-debug")
async def trigger_error():
    """Debug endpoint to test Sentry error reporting"""
    division_by_zero = 1 / 0


@app.post("/model/reload")
async def reload_model():
    """Reload model from registry (useful after retraining)"""
    try:
        load_model_from_registry()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_info": model_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


def preprocess_features(transaction: TransactionFeatures) -> pd.DataFrame:
    """
    Preprocess transaction features by calling preprocessing service
    """
    try:
        raw_data = transaction.dict()

        logger.info(
            f"Calling preprocessing service at {PREPROCESSING_SERVICE_URL}/engineer"
        )
        response = requests.post(
            f"{PREPROCESSING_SERVICE_URL}/engineer", json=raw_data, timeout=10
        )

        if response.status_code != 200:
            raise Exception(
                f"Preprocessing service returned {response.status_code}: {response.text}"
            )

        result = response.json()
        features = result["features"]

        logger.info(
            f"Received {result['feature_count']} engineered features from preprocessing service"
        )

        features_df = pd.DataFrame([features])

        return features_df

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to preprocessing service: {e}")
        raise HTTPException(
            status_code=503, detail=f"Preprocessing service unavailable: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Feature preprocessing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Feature preprocessing failed: {str(e)}"
        )


def calculate_risk_level(probability: float) -> str:
    """Calculate risk level based on fraud probability"""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.8:
        return "HIGH"
    else:
        return "CRITICAL"


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionFeatures):
    """
    Predict fraud for a single transaction

    Returns:
    - is_fraud: Boolean indicating if transaction is fraudulent
    - fraud_probability: Probability of fraud (0-1)
    - risk_level: LOW, MEDIUM, HIGH, or CRITICAL
    """
    start_time = time.time()

    with tracer.start_as_current_span("predict_fraud") as span:
        span.set_attribute("transaction.amount", transaction.TransactionAmt)
        span.set_attribute("transaction.product", transaction.ProductCD or "unknown")

        if model is None:
            http_requests_total.labels(
                method="POST", endpoint="/predict", status="503"
            ).inc()
            span.set_attribute("error", True)
            span.add_event("Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            with tracer.start_as_current_span("preprocess_features"):
                features_df = preprocess_features(transaction)

            with tracer.start_as_current_span("model_inference") as pred_span:
                prediction = model.predict(features_df)
                pred_span.set_attribute("model.name", model_info.get("name", "unknown"))
                pred_span.set_attribute(
                    "model.version", model_info.get("version", "unknown")
                )

                try:
                    probability = model.predict_proba(features_df)[0][1]
                except:
                    probability = float(prediction[0])

                pred_span.set_attribute("prediction.probability", probability)

            transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            is_fraud = bool(prediction[0] >= 0.5)
            risk_level = calculate_risk_level(probability)

            duration = time.time() - start_time
            prediction_duration_seconds.observe(duration)
            http_requests_total.labels(
                method="POST", endpoint="/predict", status="200"
            ).inc()
            fraud_predictions_total.labels(
                prediction="fraud" if is_fraud else "legitimate", risk_level=risk_level
            ).inc()
            fraud_probability_gauge.set(probability)

            span.set_attribute("prediction.is_fraud", is_fraud)
            span.set_attribute("prediction.risk_level", risk_level)
            span.set_attribute("prediction.duration_ms", duration * 1000)

            logger.info(
                f"Prediction completed in {duration:.3f}s - Fraud: {is_fraud}, Probability: {probability:.2%}"
            )

            # Monitor drift
            try:
                drift_monitor.add_prediction(features_df.iloc[0].to_dict())
            except Exception as e:
                logger.warning(f"Error adding prediction to drift monitor: {e}")

            # Log prediction for performance tracking
            try:
                performance_monitor.log_prediction(
                    transaction_id=transaction_id,
                    features=features_df.iloc[0].to_dict(),
                    prediction=1 if is_fraud else 0,
                    probability=probability,
                    timestamp=datetime.now().isoformat(),
                )
            except Exception as e:
                logger.warning(
                    f"Error logging prediction for performance tracking: {e}"
                )

            return PredictionResponse(
                transaction_id=transaction_id,
                is_fraud=is_fraud,
                fraud_probability=float(probability),
                risk_level=risk_level,
                model_name=model_info.get("name"),
                model_version=model_info.get("version"),
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            duration = time.time() - start_time
            prediction_duration_seconds.observe(duration)
            http_requests_total.labels(
                method="POST", endpoint="/predict", status="500"
            ).inc()
            span.set_attribute("error", True)
            span.record_exception(e)
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/direct", response_model=PredictionResponse)
async def predict_direct(features: Dict[str, Any]):
    """
    Predict fraud using pre-engineered features (for testing with real dataset)

    This endpoint bypasses feature engineering and accepts features directly.
    Useful for testing with data that already has V-columns, C-columns, etc.
    """
    start_time = time.time()

    if model is None:
        http_requests_total.labels(
            method="POST", endpoint="/predict/direct", status="503"
        ).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features_df = pd.DataFrame([features])

        for col in features_df.columns:
            if features_df[col].dtype == "object":
                features_df[col] = pd.to_numeric(
                    features_df[col], errors="coerce"
                ).fillna(0)

        prediction = model.predict(features_df)

        try:
            probability = model.predict_proba(features_df)[0][1]
        except:
            probability = float(prediction[0])

        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        is_fraud = bool(prediction[0] >= 0.5)
        risk_level = calculate_risk_level(probability)

        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        http_requests_total.labels(
            method="POST", endpoint="/predict/direct", status="200"
        ).inc()
        fraud_predictions_total.labels(
            prediction="fraud" if is_fraud else "legitimate", risk_level=risk_level
        ).inc()
        fraud_probability_gauge.set(probability)

        logger.info(
            f"Direct prediction completed in {duration:.3f}s - Fraud: {is_fraud}, Probability: {probability:.2%}"
        )

        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_probability=float(probability),
            risk_level=risk_level,
            model_name=model_info.get("name"),
            model_version=model_info.get("version"),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        http_requests_total.labels(
            method="POST", endpoint="/predict/direct", status="500"
        ).inc()
        logger.error(f"Direct prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Direct prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions

    Returns list of predictions for each transaction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for transaction in request.transactions:
            features_df = preprocess_features(transaction)

            prediction = model.predict(features_df)

            try:
                probability = model.predict_proba(features_df)[0][1]
            except:
                probability = float(prediction[0])

            transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            predictions.append(
                {
                    "transaction_id": transaction_id,
                    "is_fraud": bool(prediction[0] >= 0.5),
                    "fraud_probability": float(probability),
                    "risk_level": calculate_risk_level(probability),
                    "model_name": model_info.get("name"),
                    "model_version": model_info.get("version"),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            "predictions": predictions,
            "total_transactions": len(predictions),
            "fraud_count": sum(1 for p in predictions if p["is_fraud"]),
            "model_info": model_info,
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


# --- Monitoring Endpoints ---


@app.post("/monitoring/ground-truth")
async def submit_ground_truth(label: GroundTruthLabel):
    """
    Submit ground truth label for a prediction

    This endpoint allows you to provide the actual fraud label after verification.
    The system will use this to calculate model performance metrics.
    """
    try:
        performance_monitor.submit_ground_truth(
            transaction_id=label.transaction_id,
            ground_truth=1 if label.is_fraud else 0,
            timestamp=label.timestamp,
        )

        return {
            "status": "success",
            "message": f"Ground truth label submitted for {label.transaction_id}",
            "transaction_id": label.transaction_id,
            "is_fraud": label.is_fraud,
        }

    except Exception as e:
        logger.error(f"Error submitting ground truth: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to submit ground truth: {str(e)}"
        )


@app.get("/monitoring/performance")
async def get_performance_metrics():
    """
    Get current model performance metrics

    Returns precision, recall, F1, accuracy, and confusion matrix
    based on predictions that have received ground truth labels.
    """
    try:
        summary = performance_monitor.get_metrics_summary()
        return summary

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance metrics: {str(e)}"
        )


@app.get("/monitoring/drift")
async def get_drift_status():
    """
    Get current data drift monitoring status

    Returns information about drift detection configuration and status.
    """
    try:
        summary = drift_monitor.get_drift_summary()
        return summary

    except Exception as e:
        logger.error(f"Error getting drift status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get drift status: {str(e)}"
        )


@app.post("/monitoring/drift/reference")
async def set_reference_data(request: ReferenceDataRequest):
    """
    Set reference data for drift detection

    The reference data should be a sample from your training dataset.
    This will be used as baseline to detect distribution shifts.
    """
    try:
        df = pd.DataFrame(request.data)

        drift_monitor.set_reference_data(df)

        return {
            "status": "success",
            "message": "Reference data set successfully",
            "samples": len(df),
            "features": len(df.columns),
        }

    except Exception as e:
        logger.error(f"Error setting reference data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to set reference data: {str(e)}"
        )


@app.post("/monitoring/performance/calculate")
async def force_performance_calculation():
    """
    Force recalculation of performance metrics

    Useful for immediately updating metrics without waiting for buffer to fill.
    """
    try:
        performance_monitor.force_metric_calculation()

        return {"status": "success", "message": "Performance metrics recalculated"}

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate metrics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
