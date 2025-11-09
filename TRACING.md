# Distributed Tracing and Error Monitoring

Observability setup with OpenTelemetry, Tempo, and Sentry for the Fraud Detection MLOps pipeline.
![alt text](tracing.png)


## Overview

The system implements complete observability with four pillars:

1. Metrics (Prometheus/Grafana) - Request rates, latencies, throughput
2. Logs (Loki/Promtail/Grafana) - Application logs with trace correlation
3. Traces (OpenTelemetry/Tempo/Grafana) - Distributed request tracing
4. Errors (Sentry) - Error monitoring, alerting, and performance tracking

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Client    │────▶│  API Service     │────▶│ Preprocessing│
│   Request   │     │  (Port 8000)     │     │  (Port 8031) │
└─────────────┘     └──────────────────┘     └─────────────┘
       │                    │                        │
       │                    │                        │
       ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│              OpenTelemetry Instrumentation              │
│    (Automatic span creation, context propagation)       │
└─────────────────────────────────────────────────────────┘
       │                    │                        │
       ├────────────────────┴────────────────────────┤
       │                                              │
       ▼                                              ▼
┌─────────────┐                              ┌──────────────┐
│    Tempo    │                              │   Sentry     │
│  (Port 3200)│                              │   (Cloud)    │
│   Traces    │                              │   Errors     │
└─────────────┘                              └──────────────┘
       │
       ▼
┌─────────────┐
│   Grafana   │
│  (Port 3000)│
│   Unified   │
│    View     │
└─────────────┘
```

## Components

### 1. OpenTelemetry

**What it does:**
- Automatically instruments FastAPI applications
- Creates spans for each HTTP request
- Propagates trace context between services
- Tracks timing, errors, and custom attributes

**Configuration:**
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Tempo endpoint (default: `http://tempo:4317`)
- `ENVIRONMENT`: Deployment environment (production/staging/dev)

**Files:**
- `api/tracing_config.py` - API service tracing setup
- `preprocessing_service/tracing_config.py` - Preprocessing tracing setup

### 2. Grafana Tempo

**What it does:**
- Stores distributed traces
- Indexes traces by service, operation, attributes
- Provides TraceQL query language
- Correlates traces with logs and metrics

**Ports:**
- `3200` - HTTP API
- `4317` - OTLP gRPC (OpenTelemetry ingestion)
- `4318` - OTLP HTTP

**Configuration:**
- `monitoring/tempo/tempo-config.yaml` - Tempo configuration
- `monitoring/grafana/provisioning/datasources/tempo.yml` - Grafana datasource

### 3. Sentry

**What it does:**
- Error tracking and alerting
- Performance monitoring (APM)
- Release tracking
- User feedback and breadcrumbs
- Performance profiling

**Configuration:**
- `SENTRY_DSN`: Your Sentry project DSN (required)
- `SENTRY_TRACES_SAMPLE_RATE`: Percentage of traces to send (0.0-1.0, default: 1.0)
- `SENTRY_PROFILES_SAMPLE_RATE`: Percentage of profiles to send (0.0-1.0, default: 1.0)
- `APP_VERSION`: Application version for release tracking

**Files:**
- `api/sentry_config.py` - API service Sentry setup
- `preprocessing_service/sentry_config.py` - Preprocessing Sentry setup
- `dags/training.py` - Training pipeline Sentry setup
- `dags/preprocessing_pipeline.py` - Preprocessing DAG Sentry setup

## Setup Instructions

### Step 1: Get Sentry DSN (Optional but Recommended)

1. **Create Sentry Account**: https://sentry.io/signup/
2. **Create New Project**: Choose "Python" → "FastAPI"
3. **Copy DSN**: Format: `https://your-key@o123456.ingest.sentry.io/123456`
4. **Update docker-compose files**:
   ```yaml
   # In docker-compose.api.yaml and docker-compose.preprocessing.yaml
   environment:
     SENTRY_DSN: "https://your-key@o123456.ingest.sentry.io/123456"
   ```

> **Note**: Sentry is optional. If `SENTRY_DSN` is empty, error monitoring will be disabled but tracing will still work.

### Step 2: Rebuild Services

```bash
# Rebuild with new dependencies
docker compose -f docker-compose.preprocessing.yaml build
docker compose -f docker-compose.api.yaml build

# Restart monitoring stack with Tempo
docker compose -f docker-compose.monitoring.yaml down
docker compose -f docker-compose.monitoring.yaml up -d

# Restart services
docker compose -f docker-compose.preprocessing.yaml restart
docker compose -f docker-compose.api.yaml restart
```

### Step 3: Verify Setup

```bash
# Check Tempo is running
curl http://localhost:3200/ready

# Check services are sending traces
docker logs fraud-detection-api | grep "OpenTelemetry"
docker logs preprocessing-service | grep "OpenTelemetry"

# Expected output:
# OpenTelemetry tracing configured with Tempo backend
# Sentry error monitoring initialized
```

## Using Distributed Tracing in Grafana

### 1. Access Grafana Tracing Dashboard

1. **Open Grafana**: http://localhost:3000 (admin/admin123)
2. **Navigate**: Dashboards → "Fraud Detection Distributed Tracing"
3. **View**:
   - Total Request Rate (stat)
   - P95 Prediction Latency (stat)
   - P95 Feature Engineering Latency (stat)
   - Error Rate (stat)
   - Recent Traces - API Service (trace list)
   - Recent Traces - Preprocessing Service (trace list)
   - Error Traces (filtered)
   - Slow Traces >500ms (filtered)

### 2. Explore Traces

**Navigate to Explore:**
1. Click Explore icon in left sidebar
2. Select Tempo datasource
3. Choose Search query type

**Search Options:**
- Service Name: Filter by service (fraud-detection-api, preprocessing-service)
- Span Name: Filter by operation (predict_fraud, engineer_features, etc.)
- Duration: Find slow traces (over 500ms)
- Status: Find errors (status=error)
- Tags: Custom attributes (transaction.amount, risk_level)

### 3. TraceQL Queries

TraceQL is Tempo's query language for finding traces:

**Basic Queries:**
```traceql
# All traces from API service
{service.name="fraud-detection-api"}

# All traces from preprocessing service
{service.name="preprocessing-service"}

# All traces from either service
{service.name=~"fraud-detection-api|preprocessing-service"}

# Error traces
{status=error}

# Slow traces (>500ms)
{duration>500ms}

# Very slow traces (>1s)
{duration>1s}
```

**Advanced Queries:**
```traceql
# High-value transactions
{service.name="fraud-detection-api" && span.transaction.amount>1000}

# Fraud predictions with high probability
{span.prediction.is_fraud=true && span.prediction.risk_level="CRITICAL"}

# Feature engineering taking >100ms
{service.name="preprocessing-service" && duration>100ms}

# Errors in prediction operations
{service.name="fraud-detection-api" && span.name="predict_fraud" && status=error}

# Combine conditions
{service.name="fraud-detection-api" && duration>200ms && span.prediction.is_fraud=true}
```

**Aggregation Queries:**
```traceql
# Count traces by service
{} | count() by(service.name)

# Average duration by operation
{} | avg(duration) by(span.name)

# P95 latency by service
{} | quantile_over_time(duration, 0.95) by(service.name)
```

### 4. Understanding Trace Visualization

When you click on a trace, you see:

**Trace Timeline:**
- Each bar is a span (operation)
- Width indicates duration
- Color indicates service or status
- Parent-child relationships shown hierarchically

**Span Details:**
- Duration: How long the operation took
- Service: Which service executed it
- Operation: Function or endpoint name
- Attributes: Custom data (transaction amount, fraud probability)
- Events: Log messages within the span
- Status: OK, Error, Unset

**Example Trace:**
```
▼ POST /predict                                [500ms total]
  ├── predict_fraud                            [450ms]
  │   ├── preprocess_features                  [50ms]
  │   │   └── HTTP POST /engineer              [45ms]
  │   │       └── engineer_features            [40ms]
  │   │           └── feature_computation      [35ms]
  │   └── model_inference                      [380ms]
  └── Response                                 [5ms]
```

### 5. Correlating Traces with Logs

Grafana automatically links traces and logs:

**From Trace to Logs:**
1. Click on any span in the trace view
2. Click Logs for this span button
3. Grafana queries Loki for logs with matching trace_id
4. See all log messages during that request

**From Logs to Traces:**
1. In Explore, select Loki datasource
2. Query logs: `{service="api"} | json`
3. Click on any log line
4. Click Tempo button to see the full trace

**Trace ID in Logs:**
All logs include trace_id:
```
2025-11-09 03:22:54 - fraud_detection_api - INFO - Request: GET /metrics from 172.19.0.2 [trace_id=26d7f6f0903cecbd9f008563bf45dac3]
```

### 6. Service Map

View the service dependency graph:

1. Go to Explore → Tempo datasource
2. Click "Service Graph" tab
3. See visual representation:
   - Nodes = services
   - Edges = request flows
   - Width = request volume
   - Color = error rate

## Using Sentry Error Monitoring

### 1. Access Sentry Dashboard

1. **Open Sentry**: https://sentry.io
2. **Select Project**: Your fraud detection project
3. **View**:
   - Issues: All errors and exceptions
   - Performance: Transaction performance (APM)
   - Releases: Deploy tracking
   - Alerts: Configured alerts

### 2. Understanding Sentry Issues

**Issue Details:**
- Error Message: Exception type and message
- Stack Trace: Full Python stack trace
- Breadcrumbs: Events leading up to error (HTTP requests, logs)
- Tags: Service name, environment, transaction type
- Context: Request data, user info, custom data
- Release: Which version had the error

**Example Issue:**
```
ValueError: Model not loaded
  File "main.py", line 352, in predict_fraud
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

Tags:
  - service_type: mlops-microservice
  - environment: production
  - endpoint: /predict
  
Breadcrumbs:
  10:15:20 - HTTP Request: POST /predict
  10:15:21 - Log: Model loading failed
  10:15:22 - Exception raised
```

### 3. Sentry Alerts

**Create Alert Rules:**
1. Go to Alerts then Create Alert Rule
2. Choose conditions:
   - Error frequency (over 10 errors per hour)
   - New issue created
   - Issue frequency increase (50% spike)
   - Performance degradation (P95 over 500ms)

3. Set actions:
   - Email notification
   - Slack message
   - PagerDuty incident
   - Webhook

**Example Alert:**
```
Alert: High Error Rate on API Service
Condition: >10 errors in 5 minutes
Action: Send to #alerts Slack channel
```

### 4. Performance Monitoring

Sentry tracks transaction performance:

**Transaction Types:**
- HTTP requests (POST /predict)
- Background tasks
- Celery tasks

**Metrics:**
- P50, P75, P95, P99 latencies
- Throughput (requests per second)
- Failure rate
- Apdex score

**View Performance:**
1. Go to Performance then Transactions
2. Sort by P95 Duration to find slowest endpoints
3. Click transaction to see:
   - Latency distribution
   - Span operations breakdown
   - Suspect spans (slow operations)

### 5. Release Tracking

Track errors by deployment version:

**Set Release:**
```yaml
# In docker-compose files
environment:
  APP_VERSION: "1.2.0"
```

**Benefits:**
- See errors introduced in each release
- Compare error rates between releases
- Roll back if issues detected
- Track issue resolution across releases

## Span Attributes Reference

Custom attributes added to traces:

### API Service Spans

**predict_fraud span:**
- transaction.amount (float) - Transaction amount
- transaction.product (string) - Product code
- prediction.is_fraud (bool) - Fraud prediction
- prediction.risk_level (string) - LOW/MEDIUM/HIGH/CRITICAL
- prediction.probability (float) - Fraud probability (0-1)
- prediction.duration_ms (float) - Total prediction time
- model.name (string) - Model name
- model.version (string) - Model version

**model_inference span:**
- model.name (string)
- model.version (string)
- prediction.probability (float)

### Preprocessing Service Spans

**engineer_features span:**
- transaction.amount (float) - Transaction amount
- transaction.product (string) - Product code
- features.count (int) - Number of features generated
- processing.duration_ms (float) - Feature engineering time

## Troubleshooting

### Traces Not Appearing

**Check 1: Tempo is running**
```bash
curl http://localhost:3200/ready
# Should return: {"ready": "true"}
```

**Check 2: Services configured correctly**
```bash
docker logs fraud-detection-api | grep "OpenTelemetry"
# Should see: OpenTelemetry tracing configured with Tempo backend
```

**Check 3: Network connectivity**
```bash
docker exec fraud-detection-api ping -c 1 tempo
# Should succeed
```

**Check 4: Generate test traffic**
```bash
# Make predictions to generate traces
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0, "card1": 13926}'
```

### Sentry Not Receiving Errors

**Check 1: DSN configured**
```bash
docker exec fraud-detection-api env | grep SENTRY_DSN
# Should show your DSN
```

**Check 2: Test error**
```bash
# Trigger an error intentionally
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{}'  # Invalid request
```

**Check 3: Check logs**
```bash
docker logs fraud-detection-api | grep Sentry
# Should see: Sentry initialized
```

## Best Practices

### 1. Span Naming

**Good span names:**
- predict_fraud
- engineer_features
- model_inference
- load_model_from_registry

**Bad span names:**
- predict_fraud_TXN_20231108_123456 (includes unique ID)
- process_request (too generic)

### 2. Custom Attributes

**Good attributes:**
- Business metrics: transaction amount, product type
- Performance indicators: cache hit or miss, database queries
- Error context: error type, retry count

**Bad attributes:**
- High cardinality: user IDs, transaction IDs, timestamps
- Sensitive data: passwords, credit card numbers, personally identifiable information

### 3. Sampling

For high-traffic services, reduce sampling:

```python
# In tracing_config.py
tracer_provider = TracerProvider(
    resource=resource,
    sampler=TraceIdRatioBased(0.1)  # Sample 10% of traces
)
```

**Production Recommendations:**
- Development: 100% sampling
- Staging: 50-100% sampling
- Production (low traffic): 100% sampling
- Production (high traffic): 10-25% sampling

### 4. Error Handling

Always record exceptions in spans:

```python
try:
    result = some_operation()
except Exception as e:
    span.set_attribute("error", True)
    span.record_exception(e)  # Records full exception details
    raise
```

### 5. Context Propagation

OpenTelemetry automatically propagates context between services via HTTP headers:

```
traceparent: 00-a1b2c3d4e5f6...  # Trace ID + Span ID + Flags
tracestate: vendor=value         # Additional vendor-specific data
```

FastAPI instrumentation handles this automatically.

## Performance Impact

### Resource Overhead

**CPU:**
- OpenTelemetry: less than 1% overhead
- Span creation: approximately 50 microseconds per span
- OTLP export: Async, no blocking

**Memory:**
- Span objects: approximately 1KB per span
- Batch processor buffer: 5MB max (default)

**Network:**
- OTLP gRPC: approximately 500 bytes per span (compressed)
- Batch export: Every 30 seconds or 2048 spans

### Tempo Storage

**Retention:**
- Default: 744 hours (31 days)
- Configurable in tempo-config.yaml

**Storage Growth:**
- Approximately 1MB per 1000 traces
- Approximately 2.6GB per month for 100 requests per second

## Resources

- **OpenTelemetry Docs**: https://opentelemetry.io/docs/
- **Grafana Tempo Docs**: https://grafana.com/docs/tempo/
- **TraceQL Reference**: https://grafana.com/docs/tempo/latest/traceql/
- **Sentry Docs**: https://docs.sentry.io/
- **FastAPI OpenTelemetry**: https://opentelemetry-python-contrib.readthedocs.io/
