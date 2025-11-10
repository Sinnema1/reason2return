"""Logging utilities for R2R.

Provides structured logging with correlation IDs for tracing.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

from pythonjsonlogger import jsonlogger

# Context variable for correlation ID (thread-safe)
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id attribute to log record.

        Args:
            record: Log record to modify

        Returns:
            True to allow the record to be logged
        """
        record.correlation_id = correlation_id.get() or "N/A"
        return True


def setup_logging(
    level: str = "INFO",
    log_format: str = "structured",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: 'structured' for JSON, 'simple' for human-readable
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("r2r")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Remove any existing handlers

    # Add correlation ID filter to all handlers
    logger.addFilter(CorrelationIdFilter())

    structured = log_format == "structured"
    if structured:
        # JSON formatter for production
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(correlation_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        # Simple formatter for development
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )  # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "r2r") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for request tracing.

    Args:
        cid: Correlation ID string (generates UUID if None)

    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


def log_metrics(logger: logging.Logger, metrics: dict[str, Any], prefix: str = "") -> None:
    """Log metrics in a structured way.

    Args:
        logger: Logger instance
        metrics: Dictionary of metric name -> value
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        full_name = f"{prefix}.{name}" if prefix else name
        logger.info(
            "metric",
            extra={
                "metric_name": full_name,
                "metric_value": value,
                "event_type": "metric",
            },
        )


# Example usage functions for documentation
def _example_usage():
    """Example of how to use the logging utilities."""
    # Setup logging (typically done once at app startup)
    logger = setup_logging(level="INFO", log_format="structured")

    # Set correlation ID for a request/operation
    cid = set_correlation_id()
    logger.info("Starting operation", extra={"operation": "train_model", "correlation_id": cid})

    # Log structured data
    logger.info(
        "Training epoch completed",
        extra={"epoch": 1, "loss": 0.456, "accuracy": 0.789},
    )

    # Log metrics
    log_metrics(
        logger,
        {"sharpe": 1.23, "max_dd": -0.15, "accuracy": 0.85},
        prefix="backtest",
    )

    # Get logger in other modules
    module_logger = get_logger("r2r.training")
    module_logger.debug("Detailed training info")
