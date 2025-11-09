"""
Data Drift Monitoring Module

Tracks distribution changes in input features over time using Evidently AI.
Calculates drift metrics and exposes them to Prometheus for visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import os
from pathlib import Path
from evidently.presets import DataDriftPreset
from evidently import Report
from scipy import stats
from prometheus_client import Gauge, Counter

logger = logging.getLogger("fraud_detection_api")


class DriftMonitor:
    """Monitor data drift between reference (training) and current (production) data"""

    def __init__(self, reference_data_path: Optional[str] = None):
        """
        Initialize drift monitor

        Args:
            reference_data_path: Path to reference dataset (training data sample)
        """
        self.reference_data = None
        self.reference_data_path = (
            reference_data_path or "./data/reference_data.parquet"
        )
        self.current_data_buffer = []
        self.buffer_size = 100  # Number of predictions to accumulate before drift check

        # Prometheus metrics
        self.drift_detected_total = Counter(
            "data_drift_detected_total",
            "Number of times data drift was detected",
            ["feature"],
        )

        self.drift_score = Gauge(
            "data_drift_score",
            "Drift score for features (0-1, higher means more drift)",
            ["feature"],
        )

        self.dataset_drift = Gauge(
            "dataset_drift_detected", "Whether dataset-level drift is detected (0 or 1)"
        )

        self.feature_drift_count = Gauge(
            "feature_drift_count", "Number of features with detected drift"
        )

        self.kolmogorov_smirnov_statistic = Gauge(
            "kolmogorov_smirnov_statistic",
            "KS statistic for numerical features",
            ["feature"],
        )

        # Load reference data
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference dataset from training data"""
        try:
            if os.path.exists(self.reference_data_path):
                self.reference_data = pd.read_parquet(self.reference_data_path)
                logger.info(
                    f"Loaded reference data: {len(self.reference_data)} samples, {len(self.reference_data.columns)} features"
                )
            else:
                logger.warning(
                    f"Reference data not found at {self.reference_data_path}. Drift monitoring will be disabled until reference data is set."
                )
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")

    def set_reference_data(self, df: pd.DataFrame):
        """
        Set reference dataset manually

        Args:
            df: Reference DataFrame
        """
        self.reference_data = df.copy()

        # Save to disk
        Path(self.reference_data_path).parent.mkdir(parents=True, exist_ok=True)
        self.reference_data.to_parquet(self.reference_data_path)

        logger.info(
            f"Reference data set: {len(self.reference_data)} samples, {len(self.reference_data.columns)} features"
        )

    def add_prediction(self, features: Dict[str, float]):
        """
        Add a prediction to the current data buffer

        Args:
            features: Feature dictionary from a single prediction
        """
        self.current_data_buffer.append(features)

        # Check drift when buffer is full
        if len(self.current_data_buffer) >= self.buffer_size:
            self._check_drift()
            self.current_data_buffer = []  # Clear buffer

    def _check_drift(self):
        """Check for data drift between reference and current data"""
        if self.reference_data is None:
            logger.warning("Reference data not set. Skipping drift check.")
            return

        if len(self.current_data_buffer) == 0:
            return

        try:
            # Convert current buffer to DataFrame
            current_data = pd.DataFrame(self.current_data_buffer)

            # Ensure both datasets have the same columns
            common_cols = list(
                set(self.reference_data.columns) & set(current_data.columns)
            )

            if len(common_cols) == 0:
                logger.warning("No common columns between reference and current data")
                return

            reference_subset = self.reference_data[common_cols]
            current_subset = current_data[common_cols]

            # Run Evidently drift report
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(current_subset, reference_subset)

            # Extract drift metrics
            report_dict = drift_report.as_dict()

            # Update Prometheus metrics
            if "metrics" in report_dict:
                for metric in report_dict["metrics"]:
                    if metric.get("metric") == "DataDriftTable":
                        result = metric.get("result", {})

                        # Dataset-level drift
                        dataset_drift_detected = result.get("dataset_drift", False)
                        self.dataset_drift.set(1 if dataset_drift_detected else 0)

                        # Number of drifted features
                        drift_by_columns = result.get("drift_by_columns", {})
                        drifted_count = sum(
                            1
                            for v in drift_by_columns.values()
                            if v.get("drift_detected", False)
                        )
                        self.feature_drift_count.set(drifted_count)

                        # Per-feature drift metrics
                        for feature, drift_info in drift_by_columns.items():
                            if drift_info.get("drift_detected", False):
                                self.drift_detected_total.labels(feature=feature).inc()

                            # Drift score
                            drift_score_value = drift_info.get("drift_score", 0)
                            self.drift_score.labels(feature=feature).set(
                                drift_score_value
                            )

                            # KS statistic for numerical features
                            if (
                                "stattest_name" in drift_info
                                and "ks" in drift_info["stattest_name"].lower()
                            ):
                                ks_stat = drift_info.get("drift_score", 0)
                                self.kolmogorov_smirnov_statistic.labels(
                                    feature=feature
                                ).set(ks_stat)

            logger.info(
                f"Drift check completed. Dataset drift: {dataset_drift_detected}, Drifted features: {drifted_count}/{len(common_cols)}"
            )

        except Exception as e:
            logger.error(f"Error during drift check: {e}")

    def calculate_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI) manually

        PSI measures the shift in distribution between two datasets.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change

        Args:
            reference: Reference data array
            current: Current data array
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference, bins=bins)

            # Get frequency distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)

            # Convert to percentages
            ref_pct = ref_counts / len(reference)
            curr_pct = curr_counts / len(current)

            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0

    def get_drift_summary(self) -> Dict:
        """Get summary of current drift status"""
        if self.reference_data is None:
            return {"status": "no_reference_data"}

        return {
            "reference_samples": len(self.reference_data),
            "buffer_size": len(self.current_data_buffer),
            "buffer_capacity": self.buffer_size,
            "reference_features": len(self.reference_data.columns),
        }
