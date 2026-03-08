"""Structured logging configuration for the MAGI pipeline."""

import json
import logging
import uuid
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def __init__(self, trace_id: Optional[str] = None):
        super().__init__()
        self.trace_id = trace_id or str(uuid.uuid4())[:8]

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": self.trace_id,
        }
        if hasattr(record, "step_name"):
            log_entry["step_name"] = record.step_name
        if hasattr(record, "model_name"):
            log_entry["model_name"] = record.model_name
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "tokens_used"):
            log_entry["tokens_used"] = record.tokens_used
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def configure_logging(
    verbose: bool = False, trace_id: Optional[str] = None
) -> str:
    """Configure structured logging for the pipeline.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
        trace_id: Optional trace ID for request tracking.

    Returns:
        The trace ID being used.
    """
    _trace_id = trace_id or str(uuid.uuid4())[:8]

    logger = logging.getLogger("magi_system")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(trace_id=_trace_id))
    logger.addHandler(handler)

    return _trace_id
