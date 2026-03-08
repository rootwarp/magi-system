"""Tests for MAGI-031: retry utility with exponential backoff."""

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch


class TestRetryWithBackoffSync:
    """Tests for synchronous retry_with_backoff decorator."""

    @patch("magi_system.tools.retry.time")
    def test_succeeds_without_retry(self, mock_time) -> None:
        """Test function that succeeds on first call is not retried."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def succeed():
            return "ok"

        result = succeed()
        assert result == "ok"
        mock_time.sleep.assert_not_called()

    @patch("magi_system.tools.retry.time")
    def test_retries_on_transient_error_then_succeeds(self, mock_time) -> None:
        """Test retry on transient error, then succeed."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection failed")
            return "ok"

        result = fail_then_succeed()
        assert result == "ok"
        assert call_count == 3
        assert mock_time.sleep.call_count == 2

    @patch("magi_system.tools.retry.time")
    def test_max_retries_exceeded_raises(self, mock_time) -> None:
        """Test that error is raised after max retries are exhausted."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff(max_retries=2, base_delay=1.0)
        def always_fail():
            raise TimeoutError("timed out")

        try:
            always_fail()
            assert False, "Should have raised TimeoutError"
        except TimeoutError as e:
            assert str(e) == "timed out"

    @patch("magi_system.tools.retry.time")
    def test_non_transient_error_not_retried(self, mock_time) -> None:
        """Test that non-transient errors (e.g. ValueError) are raised immediately."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad value")

        try:
            raise_value_error()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        assert call_count == 1
        mock_time.sleep.assert_not_called()

    @patch("magi_system.tools.retry.time")
    def test_exponential_backoff_timing(self, mock_time) -> None:
        """Test that backoff delays increase exponentially."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        try:
            always_fail()
        except ConnectionError:
            pass

        # Delays: 1*2^0=1, 1*2^1=2, 1*2^2=4
        assert mock_time.sleep.call_count == 3
        delays = [call.args[0] for call in mock_time.sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @patch("magi_system.tools.retry.time")
    def test_custom_transient_errors(self, mock_time) -> None:
        """Test custom transient error types."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(
            max_retries=2, base_delay=0.5, transient_errors=(ValueError,)
        )
        def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "ok"

        result = fail_with_value_error()
        assert result == "ok"
        assert call_count == 2

    @patch("magi_system.tools.retry.time")
    def test_preserves_function_name(self, mock_time) -> None:
        """Test that decorator preserves function metadata."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff()
        def my_function():
            """My docstring."""
            return "ok"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestRetryWithBackoffAsync:
    """Tests for async retry_with_backoff decorator."""

    def test_async_succeeds_without_retry(self) -> None:
        """Test async function that succeeds on first call."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def succeed():
            return "ok"

        with patch("magi_system.tools.retry.asyncio") as mock_asyncio:
            mock_asyncio.iscoroutinefunction.return_value = True
            # Re-apply decorator to pick up mocked asyncio
            pass

        # Run the async function
        result = asyncio.get_event_loop().run_until_complete(succeed())
        assert result == "ok"

    def test_async_retries_on_transient_error(self) -> None:
        """Test async retry on transient error."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = asyncio.get_event_loop().run_until_complete(fail_then_succeed())
        assert result == "ok"
        assert call_count == 3

    def test_async_max_retries_exceeded(self) -> None:
        """Test async max retries exceeded raises."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff(max_retries=1, base_delay=0.01)
        async def always_fail():
            raise OSError("os error")

        try:
            asyncio.get_event_loop().run_until_complete(always_fail())
            assert False, "Should have raised OSError"
        except OSError as e:
            assert str(e) == "os error"

    def test_async_non_transient_not_retried(self) -> None:
        """Test async non-transient errors not retried."""
        from magi_system.tools.retry import retry_with_backoff

        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def raise_runtime():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fatal")

        try:
            asyncio.get_event_loop().run_until_complete(raise_runtime())
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
        assert call_count == 1

    def test_async_preserves_function_name(self) -> None:
        """Test async decorator preserves function metadata."""
        from magi_system.tools.retry import retry_with_backoff

        @retry_with_backoff()
        async def my_async_func():
            """Async docstring."""
            return "ok"

        assert my_async_func.__name__ == "my_async_func"
        assert my_async_func.__doc__ == "Async docstring."


class TestTransientErrors:
    """Tests for TRANSIENT_ERRORS constant."""

    def test_transient_errors_defined(self) -> None:
        """Test TRANSIENT_ERRORS is a tuple of exception types."""
        from magi_system.tools.retry import TRANSIENT_ERRORS

        assert isinstance(TRANSIENT_ERRORS, tuple)
        assert TimeoutError in TRANSIENT_ERRORS
        assert ConnectionError in TRANSIENT_ERRORS
        assert OSError in TRANSIENT_ERRORS


class TestPipelineConfigRetryFields:
    """Tests for retry fields in PipelineConfig."""

    def test_pipeline_config_has_retry_fields(self) -> None:
        """Test PipelineConfig has retry-related fields."""
        from magi_system.config import PipelineConfig

        field_names = {f.name for f in fields(PipelineConfig)}
        assert "api_timeout_seconds" in field_names
        assert "max_retries" in field_names
        assert "retry_base_delay" in field_names

    def test_pipeline_config_retry_defaults(self) -> None:
        """Test PipelineConfig retry fields have correct defaults."""
        from magi_system.config import load_config

        config = load_config()
        assert config.api_timeout_seconds == 30
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0


class TestRetryExports:
    """Tests for retry module exports."""

    def test_retry_importable_from_tools(self) -> None:
        """Test retry_with_backoff is importable from tools package."""
        from magi_system.tools import retry_with_backoff

        assert callable(retry_with_backoff)

    def test_transient_errors_importable_from_tools(self) -> None:
        """Test TRANSIENT_ERRORS is importable from tools package."""
        from magi_system.tools import TRANSIENT_ERRORS

        assert isinstance(TRANSIENT_ERRORS, tuple)
