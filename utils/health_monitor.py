"""
Health monitoring and observability system.

Tracks errors, metrics, and system health status with CSV-based logging.
No database required — append-only CSV files for simplicity.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class HealthMonitor:
    """
    Central observability engine.

    Tracks:
    - Errors (component, message, context) → logs/errors.csv
    - Metrics (latency, counts) → logs/metrics.csv
    - System health (model, RAG status)
    """

    def __init__(self) -> None:
        """Initialize health monitor with status flags."""
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        self.errors_csv = self.logs_dir / "errors.csv"
        self.metrics_csv = self.logs_dir / "metrics.csv"

        self.model_healthy = True
        self.rag_healthy = True
        self.last_error: Optional[str] = None

        self._init_csv_files()

    def _init_csv_files(self) -> None:
        """Create CSV files with headers if they don't exist."""
        if not self.errors_csv.exists():
            with open(self.errors_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'component', 'error_message', 'context'])

        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'metric_name', 'value', 'tags'])

    def log_error(self, component: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error to CSV.

        Args:
            component: Where the error occurred (e.g., "forecast_tab", "rag_engine")
            error_message: Human-readable error description
            context: Optional dict with additional context (e.g., {"state": "TN"})

        Example:
            monitor.log_error("forecast_tab", "Model prediction failed", {"state": "TN"})
        """
        timestamp = datetime.now().isoformat()
        context_str = str(context) if context else ""

        with open(self.errors_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, component, error_message, context_str])

        self.last_error = f"{component}: {error_message}"

    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a performance metric to CSV.

        Args:
            metric_name: Metric identifier (e.g., "forecast_latency_seconds")
            value: Numeric value
            tags: Optional dict with additional tags (e.g., {"state": "TN"})

        Example:
            monitor.log_metric("forecast_latency_seconds", 1.23, {"state": "TN"})
        """
        timestamp = datetime.now().isoformat()
        tags_str = str(tags) if tags else ""

        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, metric_name, value, tags_str])

    def set_model_status(self, healthy: bool) -> None:
        """
        Update model health status.

        Args:
            healthy: True if model loads successfully, False otherwise
        """
        self.model_healthy = healthy

    def set_rag_status(self, healthy: bool) -> None:
        """
        Update RAG system health status.

        Args:
            healthy: True if RAG initializes successfully, False otherwise
        """
        self.rag_healthy = healthy

    def get_status_dict(self) -> Dict[str, Any]:
        """
        Get current health status as a dict.

        Returns:
            Dict with keys:
            - "model": "✅" or "❌"
            - "rag": "✅" or "❌"
            - "last_error": None or error string
        """
        return {
            "model": "✅" if self.model_healthy else "❌",
            "rag": "✅" if self.rag_healthy else "❌",
            "last_error": self.last_error
        }
