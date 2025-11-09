# Monitoring & Observability

Complete monitoring setup for the Fraud Detection MLOps pipeline using Prometheus and Grafana.

## Overview

This monitoring stack provides real-time observability for:
- **API Performance**: Request rates, latency, error rates
- **Feature Engineering**: Processing time and throughput
- **Model Predictions**: Fraud detection rates, risk levels, confidence scores
- **System Health**: Service availability and resource usage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│             Monitoring Architecture                      │
└─────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  Fraud Detection │         │  Preprocessing   │
│       API        │         │     Service      │
│   (Port 8000)    │         │   (Port 8001)    │
│                  │         │                  │
│  /metrics        │         │  /metrics        │
└────────┬─────────┘         └────────┬─────────┘
         │                            │
         │ Scrape every 10s           │ Scrape every 10s
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────┐
│           Prometheus (Port 9090)             │
│                                              │
│  • Collects metrics from both services      │
│  • Stores time-series data                  │
│  • Provides PromQL query interface          │
└─────────────────┬────────────────────────────┘
                  │
                  │ Data source
                  │
                  ▼
┌──────────────────────────────────────────────┐
│             Grafana (Port 3000)              │
│                                              │
│  • Visual dashboards                        │
│  • Real-time metric graphs                  │
│  • Alerting (optional)                      │
└──────────────────────────────────────────────┘
```

## Metrics Collected

### API Metrics (`fraud-detection-api`)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `http_requests_total` | Counter | Total HTTP requests | method, endpoint, status |
| `prediction_duration_seconds` | Histogram | Prediction latency | - |
| `fraud_predictions_total` | Counter | Total predictions made | prediction, risk_level |
| `fraud_probability_latest` | Gauge | Latest fraud probability | - |
| `model_info` | Gauge | Model information | model_name, model_version |

### Preprocessing Metrics (`preprocessing-service`)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `http_requests_total` | Counter | Total HTTP requests | method, endpoint, status |
| `feature_engineering_duration_seconds` | Histogram | Feature engineering time | - |
| `features_generated_total` | Counter | Total features generated | - |

## Setup Instructions

### 1. Start Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose -f docker-compose.monitoring.yaml up -d

# Verify containers are running
docker ps | grep -E "prometheus|grafana"
```

### 2. Start Application Services

```bash
# Start preprocessing service
docker-compose -f docker-compose.preprocessing.yaml up -d

# Start prediction API
docker-compose -f docker-compose.api.yaml up -d
```

### 3. Access Interfaces

- **Prometheus**: http://localhost:9090
  - Query metrics using PromQL
  - Check targets at http://localhost:9090/targets
  
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin123`
  - Pre-configured dashboard: "Fraud Detection API Metrics"

- **API Metrics**: http://localhost:8000/metrics
- **Preprocessing Metrics**: http://localhost:8031/metrics

## Grafana Dashboard

The pre-configured dashboard includes:

### 1. Request Rate Panel
- Shows requests/second for all endpoints
- Grouped by method and endpoint
- Helps identify traffic patterns

### 2. Prediction Latency Panel
- P50 and P95 latency percentiles
- Identifies performance bottlenecks
- Target: P95 < 500ms

### 3. Fraud Detection Rate
- Real-time fraud percentage
- Shows model behavior
- Useful for data drift detection

### 4. Total Predictions Counter
- Cumulative prediction count
- Tracks overall usage

### 5. API Error Rate
- 4xx (client errors) and 5xx (server errors)
- Helps identify issues quickly

### 6. Feature Engineering Latency
- Time spent on feature transformation
- P50 and P95 percentiles
- Most critical performance metric

### 7. Risk Level Distribution
- Pie chart of LOW/MEDIUM/HIGH/CRITICAL predictions
- Visualizes risk distribution

## Example PromQL Queries

### Request Rate
```promql
rate(http_requests_total{job="fraud-detection-api"}[5m])
```

### P95 Latency
```promql
histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))
```

### Fraud Detection Rate
```promql
sum(rate(fraud_predictions_total{prediction="fraud"}[5m])) / 
sum(rate(fraud_predictions_total[5m]))
```

### Error Rate
```promql
sum(rate(http_requests_total{status=~"5.."}[5m])) / 
sum(rate(http_requests_total[5m]))
```

### Feature Engineering P99 Latency
```promql
histogram_quantile(0.99, rate(feature_engineering_duration_seconds_bucket[5m]))
```

## Testing Metrics Collection

### Generate Test Traffic

```bash
# Run API tests to generate metrics
python test_api.py

# Check if metrics are being collected
curl http://localhost:8000/metrics
curl http://localhost:8031/metrics
```

### Verify in Prometheus

1. Go to http://localhost:9090/targets
2. Both targets should show "UP" status
3. Go to http://localhost:9090/graph
4. Try query: `http_requests_total`

### View Dashboard in Grafana

1. Login to http://localhost:3000
2. Navigate to Dashboards → Fraud Detection API Metrics
3. You should see real-time data updating

## Alerting (Optional)

Add alerting rules to `monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: fraud_detection_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) / 
          sum(rate(http_requests_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: SlowPredictions
        expr: |
          histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow predictions detected"
          description: "P95 latency is {{ $value }}s"
      
      - alert: HighFraudRate
        expr: |
          sum(rate(fraud_predictions_total{prediction="fraud"}[10m])) / 
          sum(rate(fraud_predictions_total[10m])) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Unusually high fraud rate"
          description: "Fraud rate is {{ $value | humanizePercentage }}"
```

## Troubleshooting

### Metrics Not Appearing

**Problem**: Grafana shows "No data"

**Solutions**:
1. Check if services are running: `docker ps`
2. Verify Prometheus targets: http://localhost:9090/targets (should show UP)
3. Check service metrics: `curl http://localhost:8000/metrics`
4. Verify network: All containers should be on `fraud-detection-network`

### Prometheus Can't Scrape Services

**Problem**: Targets show "DOWN" in Prometheus

**Solutions**:
1. Check service names in `prometheus.yml` match container names
2. Verify services expose `/metrics` endpoint
3. Check firewall/network: `docker network inspect fraud-detection-network`
4. Restart Prometheus: `docker restart prometheus`

### Grafana Dashboard Empty

**Problem**: Dashboard created but no data

**Solutions**:
1. Verify Prometheus datasource: Configuration → Data Sources
2. Check Prometheus URL: `http://prometheus:9090`
3. Test queries directly in Prometheus UI first
4. Refresh Grafana dashboard

## Performance Impact

Monitoring adds minimal overhead:
- **Prometheus scraping**: ~10ms per scrape (every 10s)
- **Metric collection**: <1ms per request
- **Storage**: ~100MB per day (configurable retention)

## Data Retention

Default retention: 15 days

To change retention, update `docker-compose.monitoring.yaml`:

```yaml
prometheus:
  command:
    - '--storage.tsdb.retention.time=30d'
```

## Cleanup

```bash
# Stop monitoring stack
docker-compose -f docker-compose.monitoring.yaml down

# Remove volumes (deletes all metrics data)
docker-compose -f docker-compose.monitoring.yaml down -v
```