"""
Model Performance Monitoring Module

Tracks model performance metrics in production by comparing predictions
with ground truth labels. Calculates precision, recall, F1, false positives, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from prometheus_client import Gauge, Counter, Histogram

logger = logging.getLogger("fraud_detection_api")


class PerformanceMonitor:
    """Monitor model performance in production"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize performance monitor

        Args:
            storage_path: Path to store prediction-label pairs
        """
        self.storage_path = storage_path or "./data/predictions.jsonl"
        self.predictions_buffer = []
        self.buffer_size = 100  # Compute metrics every 100 labeled predictions

        # Prometheus metrics
        self.precision_gauge = Gauge(
            "model_precision", "Model precision score (TP / (TP + FP))"
        )

        self.recall_gauge = Gauge("model_recall", "Model recall score (TP / (TP + FN))")

        self.f1_gauge = Gauge("model_f1_score", "Model F1 score")

        self.accuracy_gauge = Gauge("model_accuracy", "Model accuracy score")

        self.auc_gauge = Gauge("model_auc_roc", "Model AUC-ROC score")

        self.true_positives = Counter(
            "model_true_positives_total", "Total number of true positives"
        )

        self.false_positives = Counter(
            "model_false_positives_total", "Total number of false positives"
        )

        self.true_negatives = Counter(
            "model_true_negatives_total", "Total number of true negatives"
        )

        self.false_negatives = Counter(
            "model_false_negatives_total", "Total number of false negatives"
        )

        self.predictions_with_labels = Counter(
            "predictions_with_ground_truth_total",
            "Total number of predictions that received ground truth labels",
        )

        self.predictions_without_labels = Counter(
            "predictions_without_ground_truth_total",
            "Total number of predictions without ground truth labels",
        )

        self.fraud_detection_latency = Histogram(
            "fraud_detection_latency_hours",
            "Time between prediction and label arrival (in hours)",
            buckets=[1, 6, 12, 24, 48, 72, 168],  # 1h to 1 week
        )

        # Create storage directory
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        transaction_id: str,
        features: Dict,
        prediction: int,
        probability: float,
        timestamp: Optional[str] = None,
    ):
        """
        Log a prediction for future performance tracking

        Args:
            transaction_id: Unique transaction identifier
            features: Input features used for prediction
            prediction: Predicted class (0 or 1)
            probability: Prediction probability
            timestamp: Prediction timestamp (ISO format)
        """
        record = {
            "transaction_id": transaction_id,
            "prediction": int(prediction),
            "probability": float(probability),
            "timestamp": timestamp or datetime.now().isoformat(),
            "features": features,
            "ground_truth": None,
            "label_timestamp": None,
        }

        # Append to storage file
        try:
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")

    def submit_ground_truth(
        self, transaction_id: str, ground_truth: int, timestamp: Optional[str] = None
    ):
        """
        Submit ground truth label for a prediction

        Args:
            transaction_id: Transaction identifier
            ground_truth: True label (0 or 1)
            timestamp: Label timestamp (ISO format)
        """
        label_timestamp = timestamp or datetime.now().isoformat()

        # Load existing predictions
        predictions = self._load_predictions()

        # Find matching prediction
        updated = False
        for pred in predictions:
            if pred["transaction_id"] == transaction_id:
                pred["ground_truth"] = int(ground_truth)
                pred["label_timestamp"] = label_timestamp

                # Calculate latency
                pred_time = datetime.fromisoformat(pred["timestamp"])
                label_time = datetime.fromisoformat(label_timestamp)
                latency_hours = (label_time - pred_time).total_seconds() / 3600
                self.fraud_detection_latency.observe(latency_hours)

                updated = True
                break

        if not updated:
            logger.warning(f"Prediction not found for transaction_id: {transaction_id}")
            self.predictions_without_labels.inc()
            return

        # Save updated predictions
        self._save_predictions(predictions)
        self.predictions_with_labels.inc()

        # Add to buffer for metric calculation
        self.predictions_buffer.append(
            {
                "prediction": pred["prediction"],
                "ground_truth": int(ground_truth),
                "probability": pred["probability"],
            }
        )

        # Calculate metrics when buffer is full
        if len(self.predictions_buffer) >= self.buffer_size:
            self._calculate_metrics()
            self.predictions_buffer = []

    def _load_predictions(self) -> List[Dict]:
        """Load predictions from storage"""
        predictions = []
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    for line in f:
                        predictions.append(json.loads(line.strip()))
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
        return predictions

    def _save_predictions(self, predictions: List[Dict]):
        """Save predictions to storage"""
        try:
            with open(self.storage_path, "w") as f:
                for pred in predictions:
                    f.write(json.dumps(pred) + "\n")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

    def _calculate_metrics(self):
        """Calculate performance metrics from buffer"""
        if len(self.predictions_buffer) == 0:
            return

        try:
            # Extract predictions and ground truth
            y_true = [p["ground_truth"] for p in self.predictions_buffer]
            y_pred = [p["prediction"] for p in self.predictions_buffer]
            y_prob = [p["probability"] for p in self.predictions_buffer]

            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

            # Calculate AUC-ROC if possible
            try:
                auc = roc_auc_score(y_true, y_prob)
                self.auc_gauge.set(auc)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")

            # Update Prometheus gauges
            self.precision_gauge.set(precision)
            self.recall_gauge.set(recall)
            self.f1_gauge.set(f1)
            self.accuracy_gauge.set(accuracy)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Update counters
            self.true_positives.inc(int(tp))
            self.false_positives.inc(int(fp))
            self.true_negatives.inc(int(tn))
            self.false_negatives.inc(int(fn))

            logger.info(
                f"Performance metrics calculated: "
                f"Precision={precision:.3f}, Recall={recall:.3f}, "
                f"F1={f1:.3f}, Accuracy={accuracy:.3f}, "
                f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")

    def get_metrics_summary(self) -> Dict:
        """Get summary of current metrics"""
        predictions = self._load_predictions()
        labeled = [p for p in predictions if p.get("ground_truth") is not None]

        if len(labeled) == 0:
            return {
                "status": "no_labeled_data",
                "total_predictions": len(predictions),
                "labeled_predictions": 0,
            }

        y_true = [p["ground_truth"] for p in labeled]
        y_pred = [p["prediction"] for p in labeled]
        y_prob = [p["probability"] for p in labeled]

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "status": "ok",
            "total_predictions": len(predictions),
            "labeled_predictions": len(labeled),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "auc_roc": float(auc) if auc is not None else None,
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            },
        }

    def force_metric_calculation(self):
        """Force metric calculation on all labeled data"""
        predictions = self._load_predictions()
        labeled = [p for p in predictions if p.get("ground_truth") is not None]

        if len(labeled) > 0:
            self.predictions_buffer = [
                {
                    "prediction": p["prediction"],
                    "ground_truth": p["ground_truth"],
                    "probability": p["probability"],
                }
                for p in labeled
            ]
            self._calculate_metrics()
            self.predictions_buffer = []
