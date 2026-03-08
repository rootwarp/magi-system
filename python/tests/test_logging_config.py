"""Tests for MAGI-034: Structured logging configuration."""

import json
import logging
from unittest.mock import patch


class TestStructuredFormatter:
    """Tests for the StructuredFormatter class."""

    def test_formatter_produces_valid_json(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="test123")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_trace_id_included_in_output(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="abc123")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert output["trace_id"] == "abc123"

    def test_auto_generated_trace_id_when_none_provided(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter()
        assert formatter.trace_id is not None
        assert len(formatter.trace_id) == 8

    def test_custom_trace_id_is_used(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="custom42")
        assert formatter.trace_id == "custom42"

    def test_output_contains_required_fields(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="test")
        record = logging.LogRecord(
            name="magi_system.pipeline",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="something happened",
            args=None,
            exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert "timestamp" in output
        assert output["level"] == "WARNING"
        assert output["logger"] == "magi_system.pipeline"
        assert output["message"] == "something happened"
        assert output["trace_id"] == "test"

    def test_extra_field_step_name(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="step",
            args=None,
            exc_info=None,
        )
        record.step_name = "clarification"
        output = json.loads(formatter.format(record))
        assert output["step_name"] == "clarification"

    def test_extra_field_model_name(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="model",
            args=None,
            exc_info=None,
        )
        record.model_name = "gemini-pro"
        output = json.loads(formatter.format(record))
        assert output["model_name"] == "gemini-pro"

    def test_extra_field_latency_ms(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="latency",
            args=None,
            exc_info=None,
        )
        record.latency_ms = 150.5
        output = json.loads(formatter.format(record))
        assert output["latency_ms"] == 150.5

    def test_extra_field_tokens_used(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="tokens",
            args=None,
            exc_info=None,
        )
        record.tokens_used = 1024
        output = json.loads(formatter.format(record))
        assert output["tokens_used"] == 1024

    def test_extra_fields_absent_when_not_set(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        record = logging.LogRecord(
            name="magi_system",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="plain",
            args=None,
            exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert "step_name" not in output
        assert "model_name" not in output
        assert "latency_ms" not in output
        assert "tokens_used" not in output

    def test_exception_info_included(self) -> None:
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="t1")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="magi_system",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=None,
            exc_info=exc_info,
        )
        output = json.loads(formatter.format(record))
        assert output["exception"] == "test error"


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_returns_trace_id(self) -> None:
        from magi_system.logging_config import configure_logging

        trace_id = configure_logging()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 8

    def test_custom_trace_id_returned(self) -> None:
        from magi_system.logging_config import configure_logging

        trace_id = configure_logging(trace_id="myid123")
        assert trace_id == "myid123"

    def test_verbose_sets_debug_level(self) -> None:
        from magi_system.logging_config import configure_logging

        configure_logging(verbose=True)
        logger = logging.getLogger("magi_system")
        assert logger.level == logging.DEBUG

    def test_non_verbose_sets_info_level(self) -> None:
        from magi_system.logging_config import configure_logging

        configure_logging(verbose=False)
        logger = logging.getLogger("magi_system")
        assert logger.level == logging.INFO

    def test_handler_uses_structured_formatter(self) -> None:
        from magi_system.logging_config import StructuredFormatter, configure_logging

        configure_logging(trace_id="fmt_test")
        logger = logging.getLogger("magi_system")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, StructuredFormatter)

    def test_clears_existing_handlers(self) -> None:
        from magi_system.logging_config import configure_logging

        logger = logging.getLogger("magi_system")
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) >= 2

        configure_logging()
        assert len(logger.handlers) == 1
