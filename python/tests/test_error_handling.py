"""Integration tests for error handling and retry with structured logging.

MAGI-038: Tests that retry decorator integrates correctly with
structured logging, mocked provider timeouts, and PipelineWarnings.
"""

import json
import logging
from unittest.mock import patch

from magi_system.logging_config import StructuredFormatter, configure_logging
from magi_system.tools.retry import TRANSIENT_ERRORS, retry_with_backoff


class TestRetryLogsAttempts:
    """Test that retry decorator emits structured log entries."""

    def test_retry_warning_is_valid_json(self) -> None:
        """Retry warnings produce valid JSON when structured formatter is active."""
        configure_logging(trace_id="retry-log-test")
        logger = logging.getLogger("magi_system.tools.retry")

        handler = logger.parent.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)

        # Capture log output
        records: list[str] = []
        original_emit = handler.emit

        def capture_emit(record):
            records.append(handler.format(record))
            original_emit(record)

        handler.emit = capture_emit
        try:

            @retry_with_backoff(max_retries=2, base_delay=0.001)
            def flaky():
                if len(records) < 2:
                    raise ConnectionError("provider timeout")
                return "ok"

            with patch("magi_system.tools.retry.time"):
                result = flaky()

            assert result == "ok"
            # At least one warning log should have been emitted
            warning_logs = [r for r in records if '"level": "WARNING"' in r]
            assert len(warning_logs) >= 1
            parsed = json.loads(warning_logs[0])
            assert parsed["trace_id"] == "retry-log-test"
            assert "Retry" in parsed["message"]
        finally:
            handler.emit = original_emit

    def test_retry_error_logged_on_max_retries(self) -> None:
        """Max retries exceeded produces an ERROR-level structured log."""
        configure_logging(trace_id="max-retry-test")
        logger = logging.getLogger("magi_system.tools.retry")
        handler = logger.parent.handlers[0]

        records: list[str] = []
        original_emit = handler.emit

        def capture_emit(record):
            records.append(handler.format(record))
            original_emit(record)

        handler.emit = capture_emit
        try:

            @retry_with_backoff(max_retries=1, base_delay=0.001)
            def always_fail():
                raise TimeoutError("provider timeout")

            with patch("magi_system.tools.retry.time"):
                try:
                    always_fail()
                except TimeoutError:
                    pass

            error_logs = [r for r in records if '"level": "ERROR"' in r]
            assert len(error_logs) >= 1
            parsed = json.loads(error_logs[0])
            assert "Max retries" in parsed["message"]
            assert parsed["trace_id"] == "max-retry-test"
        finally:
            handler.emit = original_emit


class TestRetryWithMockedProviderTimeouts:
    """Test retry behavior simulating LLM provider timeouts."""

    @patch("magi_system.tools.retry.time")
    def test_simulated_api_timeout_retries_and_recovers(self, mock_time) -> None:
        """Simulate a provider that times out twice then succeeds."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_provider():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("Request timed out after 30s")
            return {"response": "LLM answer", "tokens": 150}

        result = call_provider()
        assert result["response"] == "LLM answer"
        assert call_count == 3
        assert mock_time.sleep.call_count == 2

    @patch("magi_system.tools.retry.time")
    def test_connection_refused_is_transient(self, mock_time) -> None:
        """ConnectionError (e.g. provider down) triggers retry."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.5)
        def call_provider():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            return "success"

        result = call_provider()
        assert result == "success"
        assert call_count == 2

    @patch("magi_system.tools.retry.time")
    def test_auth_error_not_retried(self, mock_time) -> None:
        """Non-transient errors like auth failures are not retried."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_provider():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid API key")

        try:
            call_provider()
        except ValueError:
            pass
        assert call_count == 1
        mock_time.sleep.assert_not_called()


class TestRetryWithDegradation:
    """Test retry + PipelineWarnings integration."""

    @patch("magi_system.tools.retry.time")
    def test_failed_retry_feeds_degradation_warnings(self, mock_time) -> None:
        """When retry exhausts, the error can be fed into PipelineWarnings."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()

        @retry_with_backoff(max_retries=1, base_delay=0.001)
        def call_model():
            raise TimeoutError("provider timeout")

        try:
            call_model()
        except TimeoutError as e:
            pw.add_model_failure("gpt-4", str(e))

        assert pw.has_degradation
        assert "gpt-4" in pw.unavailable_models
        report = pw.format_for_report()
        assert "provider timeout" in report
        assert "Confidence scores may be affected" in report
