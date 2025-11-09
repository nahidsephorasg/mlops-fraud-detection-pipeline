from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from logging_config import setup_logger
from tracing_config import setup_tracing
from sentry_config import setup_sentry
from opentelemetry import trace

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = setup_logger(
    name="preprocessing_service",
    log_file="/app/logs/preprocessing.log",
    level=getattr(logging, LOG_LEVEL),
)

# Prometheus metrics
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

feature_engineering_duration_seconds = Histogram(
    "feature_engineering_duration_seconds",
    "Time spent on feature engineering",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

features_generated_total = Counter(
    "features_generated_total", "Total number of features generated"
)

# Initialize FastAPI app
app = FastAPI(
    title="Feature Engineering Service",
    description="Preprocessing service for fraud detection - applies feature engineering",
    version="1.0.0",
)

# Setup tracing and error monitoring
tracer = setup_tracing(app, service_name="preprocessing-service")
setup_sentry(service_name="preprocessing-service")

logger.info("OpenTelemetry tracing configured with Tempo backend")
logger.info("Sentry error monitoring initialized")


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses"""
    start_time = time.time()

    # Get current span for trace correlation
    current_span = trace.get_current_span()
    trace_id = (
        format(current_span.get_span_context().trace_id, "032x")
        if current_span
        else "no-trace"
    )

    # Log request with trace ID
    logger.info(
        f"Request: {request.method} {request.url.path} from {request.client.host} [trace_id={trace_id}]"
    )

    # Process request
    response = await call_next(request)

    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} - Duration: {duration:.3f}s [trace_id={trace_id}]"
    )

    return response


class RawTransaction(BaseModel):
    """Raw transaction data"""

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


class BatchTransactionRequest(BaseModel):
    """Batch transaction processing request"""

    transactions: List[RawTransaction]


class EngineeredFeatures(BaseModel):
    """Response with engineered features"""

    features: Dict[str, float]
    feature_count: int
    timestamp: str


def create_transaction_cents_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cents from transaction amount"""
    df = df.copy()
    df["TransactionCents"] = (
        df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    ) * 100
    return df


def create_frequency_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create frequency encoding for categorical columns"""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
    return df


def create_aggregation_features(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Create aggregation features (count, nunique)"""
    df = df.copy()

    for col in group_cols:
        if col in df.columns:
            # Count occurrences
            count_map = df[col].value_counts().to_dict()
            df[f"{col}_count"] = df[col].map(count_map).fillna(0)

            # Number of unique values
            df[f"{col}_nunique"] = df[col].map(count_map).fillna(0)

    return df


def create_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create combination features from multiple columns"""
    df = df.copy()

    # card1_addr1 combination
    if "card1" in df.columns and "addr1" in df.columns:
        df["card1_addr1_encoded"] = (
            (
                df["card1"].fillna(0).astype(str)
                + "_"
                + df["addr1"].fillna(0).astype(str)
            )
            .astype("category")
            .cat.codes
        )

        # Frequency of card1_addr1 combination
        combo = (
            df["card1"].fillna(0).astype(str) + "_" + df["addr1"].fillna(0).astype(str)
        )
        freq_map = combo.value_counts(normalize=True).to_dict()
        df["card1_addr1_freq"] = combo.map(freq_map).fillna(0)

    # card1_addr1_P_emaildomain combination
    if all(col in df.columns for col in ["card1", "addr1", "P_emaildomain"]):
        df["card1_addr1_P_emaildomain_encoded"] = (
            (
                df["card1"].fillna(0).astype(str)
                + "_"
                + df["addr1"].fillna(0).astype(str)
                + "_"
                + df["P_emaildomain"].fillna("unknown").astype(str)
            )
            .astype("category")
            .cat.codes
        )

        # Frequency
        combo = (
            df["card1"].fillna(0).astype(str)
            + "_"
            + df["addr1"].fillna(0).astype(str)
            + "_"
            + df["P_emaildomain"].fillna("unknown").astype(str)
        )
        freq_map = combo.value_counts(normalize=True).to_dict()
        df["card1_addr1_P_emaildomain_freq"] = combo.map(freq_map).fillna(0)

    return df


def create_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features based on transaction amount"""
    df = df.copy()

    if "TransactionAmt" in df.columns:
        mean_amt = df["TransactionAmt"].mean()
        df["TransactionAmt_to_mean"] = (
            df["TransactionAmt"] / mean_amt if mean_amt > 0 else 0
        )

    return df


def engineer_features(raw_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Apply full feature engineering pipeline to raw transaction data

    Args:
        raw_data: Dictionary containing raw transaction fields

    Returns:
        Dictionary with all engineered features
    """
    # Convert to DataFrame
    df = pd.DataFrame([raw_data])

    logger.info(f"Starting feature engineering with {len(df.columns)} raw features")

    # Handle categorical columns - convert to numeric before processing
    categorical_mappings = {
        "card4": {"visa": 0, "mastercard": 1, "discover": 2, "american express": 3},
        "card6": {"credit": 0, "debit": 1},
        "ProductCD": {"W": 0, "C": 1, "R": 2, "H": 3, "S": 4},
    }

    for col, mapping in categorical_mappings.items():
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].map(mapping).fillna(-1)

    # Encode email domains
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # 1. Basic feature engineering
    df = create_transaction_cents_feature(df)
    df = create_transaction_amount_features(df)

    # 2. Frequency encoding for categorical features
    categorical_cols = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
    ]
    df = create_frequency_encoding(df, categorical_cols)

    # 3. Aggregation features
    group_cols = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
    ]
    df = create_aggregation_features(df, group_cols)

    # 4. Combination features
    df = create_combination_features(df)

    # 5. Add all V-columns and C-columns with default values
    v_columns = [f"V{i}" for i in range(95, 138)] + [  # V95-V137
        f"V{i}"
        for i in [
            279,
            280,
            284,
            285,
            286,
            287,
            290,
            291,
            292,
            293,
            294,
            295,
            297,
            298,
            299,
            302,
            303,
            304,
            305,
            306,
            307,
            308,
            309,
            310,
            311,
            312,
            316,
            317,
            318,
            319,
            320,
            321,
        ]
    ]

    c_columns = [f"C{i}" for i in range(1, 15)]

    # Add TransactionID if missing
    if "TransactionID" not in df.columns:
        df["TransactionID"] = 0

    # Add all missing V and C columns with 0
    for col in v_columns + c_columns:
        if col not in df.columns:
            df[col] = 0

    # 6. Define expected feature list (exact order from training)
    expected_features = [
        "TransactionID",
        "TransactionAmt",
        "card1",
        "card3",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
        "C10",
        "C11",
        "C12",
        "C13",
        "C14",
        "V95",
        "V96",
        "V97",
        "V98",
        "V99",
        "V100",
        "V101",
        "V102",
        "V103",
        "V104",
        "V105",
        "V106",
        "V107",
        "V108",
        "V109",
        "V110",
        "V111",
        "V112",
        "V113",
        "V114",
        "V115",
        "V116",
        "V117",
        "V118",
        "V119",
        "V120",
        "V121",
        "V122",
        "V123",
        "V124",
        "V125",
        "V126",
        "V127",
        "V128",
        "V129",
        "V130",
        "V131",
        "V132",
        "V133",
        "V134",
        "V135",
        "V136",
        "V137",
        "V279",
        "V280",
        "V284",
        "V285",
        "V286",
        "V287",
        "V290",
        "V291",
        "V292",
        "V293",
        "V294",
        "V295",
        "V297",
        "V298",
        "V299",
        "V302",
        "V303",
        "V304",
        "V305",
        "V306",
        "V307",
        "V308",
        "V309",
        "V310",
        "V311",
        "V312",
        "V316",
        "V317",
        "V318",
        "V319",
        "V320",
        "V321",
        "card1_freq",
        "card2_freq",
        "card3_freq",
        "card4_freq",
        "card5_freq",
        "card6_freq",
        "addr1_freq",
        "addr2_freq",
        "card1_addr1_encoded",
        "card1_addr1_P_emaildomain_encoded",
        "card1_count",
        "card2_count",
        "card3_count",
        "card4_count",
        "card5_count",
        "card6_count",
        "addr1_count",
        "addr2_count",
        "card1_nunique",
        "card2_nunique",
        "card3_nunique",
        "card4_nunique",
        "card5_nunique",
        "card6_nunique",
        "addr1_nunique",
        "addr2_nunique",
        "TransactionAmt_to_mean",
        "TransactionCents",
        "card1_addr1_freq",
        "card1_addr1_P_emaildomain_freq",
    ]

    # 7. Add any still-missing features
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    # 8. Select only expected features in correct order
    df = df[expected_features]

    # 9. Fill any remaining NaN values
    df = df.fillna(0)

    # 10. Convert to dictionary
    features_dict = df.iloc[0].to_dict()

    logger.info(
        f"Feature engineering complete: {len(features_dict)} features generated"
    )

    return features_dict


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Feature Engineering Service",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/sentry-debug")
async def trigger_error():
    """Debug endpoint to test Sentry error reporting"""
    division_by_zero = 1 / 0


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()
    return {
        "status": "healthy",
        "service": "feature_engineering",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/engineer", response_model=EngineeredFeatures)
async def engineer_transaction_features(transaction: RawTransaction):
    """
    Engineer features for a single transaction

    Takes raw transaction data and returns engineered features ready for model prediction
    """
    start_time = time.time()

    # Create span for feature engineering operation
    with tracer.start_as_current_span("engineer_features") as span:
        span.set_attribute("transaction.amount", transaction.TransactionAmt)
        span.set_attribute("transaction.product", transaction.ProductCD or "unknown")

        try:
            raw_data = transaction.dict()

            with tracer.start_as_current_span("feature_computation"):
                features = engineer_features(raw_data)

            # Update metrics
            duration = time.time() - start_time
            feature_engineering_duration_seconds.observe(duration)
            http_requests_total.labels(
                method="POST", endpoint="/engineer", status="200"
            ).inc()
            features_generated_total.inc(len(features))

            # Add span attributes
            span.set_attribute("features.count", len(features))
            span.set_attribute("processing.duration_ms", duration * 1000)

            logger.info(
                f"Feature engineering completed in {duration:.3f}s - Generated {len(features)} features"
            )

            return EngineeredFeatures(
                features=features,
                feature_count=len(features),
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            duration = time.time() - start_time
            feature_engineering_duration_seconds.observe(duration)
            http_requests_total.labels(
                method="POST", endpoint="/engineer", status="500"
            ).inc()
            span.set_attribute("error", True)
            span.record_exception(e)
            logger.error(f"Feature engineering error: {e}")
            raise HTTPException(
                status_code=500, detail=f"Feature engineering failed: {str(e)}"
            )


@app.post("/engineer/batch")
async def engineer_batch_features(request: BatchTransactionRequest):
    """
    Engineer features for multiple transactions

    Returns a list of engineered features for batch processing
    """
    try:
        results = []

        for transaction in request.transactions:
            raw_data = transaction.dict()
            features = engineer_features(raw_data)
            results.append({"features": features, "feature_count": len(features)})

        return {
            "results": results,
            "total_transactions": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch feature engineering error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8031)
