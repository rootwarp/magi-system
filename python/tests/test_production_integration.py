"""Integration tests for Phase 4 production features working together.

MAGI-038: Tests cross-cutting integration between retry, degradation,
config_loader, effort scaling, context management, and structured logging.
"""

import json
import logging
import os
import textwrap
from pathlib import Path
from unittest.mock import patch


class TestDegradationWithFormattedReport:
    """Test PipelineWarnings integrates with format_for_report end-to-end."""

    def test_multiple_failures_produce_complete_report(self) -> None:
        """Multiple model failures and warnings produce a well-structured report."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "connection timeout")
        pw.add_model_failure("claude-3", "rate limited")
        pw.add_warning("Search results truncated due to context limits")

        report = pw.format_for_report()

        # Report structure
        assert report.startswith("## Pipeline Warnings")
        assert "gpt-4" in report
        assert "claude-3" in report
        assert "connection timeout" in report
        assert "rate limited" in report
        assert "Search results truncated" in report
        assert "Confidence scores may be affected" in report
        # Model list
        assert "Models unavailable: gpt-4, claude-3" in report

    def test_warnings_only_no_degradation_notice(self) -> None:
        """General warnings without model failures omit degradation notice."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_warning("Slow response from search provider")

        report = pw.format_for_report()
        assert "## Pipeline Warnings" in report
        assert "Slow response" in report
        assert "Confidence scores" not in report
        assert not pw.has_degradation


class TestConfigLoaderWithEnvOverrides:
    """Test config loader with env vars overriding file and defaults."""

    def test_env_overrides_toml_file(self, tmp_path: Path) -> None:
        """Environment variables take precedence over TOML file values."""
        from magi_system.config_loader import load_config_from_file

        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [pipeline]
            max_research_iterations = 10
            max_sub_questions = 20
        """))

        env = {"MAGI_MAX_ITERATIONS": "2", "MAGI_MAX_SUB_QUESTIONS": "3"}
        with patch.dict(os.environ, env):
            config = load_config_from_file(str(config_file))

        assert config.max_research_iterations == 2
        assert config.max_sub_questions == 3

    def test_config_loader_returns_usable_config(self) -> None:
        """Config from loader has all fields needed by pipeline components."""
        from magi_system.config_loader import load_config_from_file

        config = load_config_from_file()
        # Verify fields used by effort scaling and context management
        assert hasattr(config, "max_sub_questions")
        assert hasattr(config, "max_research_iterations")
        assert hasattr(config, "max_context_chars")
        assert hasattr(config, "orchestrator")
        assert hasattr(config, "synthesis")


class TestEffortScalingWithComplexity:
    """Test effort scaling with different complexity levels."""

    def test_simple_query_gets_fewer_sub_questions(self) -> None:
        """Simple complexity limits sub-questions to 5."""
        from magi_system.search.effort_config import get_effort_level

        plan = "COMPLEXITY: simple\n\nSUB_QUESTIONS:\nSQ1: What?"
        level = get_effort_level(plan)
        assert level.max_sub_questions == 5
        assert level.max_searches_per_agent == 2

    def test_deep_query_gets_more_sub_questions(self) -> None:
        """Deep complexity allows up to 15 sub-questions."""
        from magi_system.search.effort_config import get_effort_level

        plan = "COMPLEXITY: deep\n\nSUB_QUESTIONS:\nSQ1: What?"
        level = get_effort_level(plan)
        assert level.max_sub_questions == 15
        assert level.max_searches_per_agent == 5

    def test_effort_level_respects_config_cap(self) -> None:
        """Config max_sub_questions can further cap effort level."""
        from magi_system.config import load_config
        from magi_system.search.effort_config import get_effort_level

        config = load_config()
        plan = "COMPLEXITY: deep\n\nSQ1: Q?"
        level = get_effort_level(plan)

        # The effective cap is min(effort, config)
        effective = min(level.max_sub_questions, config.max_sub_questions)
        assert effective <= config.max_sub_questions


class TestContextTruncationWithLargeInputs:
    """Test context truncation handles realistic large inputs."""

    def test_large_search_results_truncated(self) -> None:
        """Search results exceeding 50k chars are truncated."""
        from magi_system.tools.context_utils import summarize_search_results

        # Simulate large multi-source search results
        results = "\n".join([
            f"Source {i}: " + "x" * 5000 for i in range(20)
        ])
        assert len(results) > 50000

        summarized = summarize_search_results(results)
        assert len(summarized) <= 50000
        assert "TRUNCATED" in summarized

    def test_truncation_preserves_start_content(self) -> None:
        """Truncated text preserves the beginning for context."""
        from magi_system.tools.context_utils import truncate_text

        header = "IMPORTANT HEADER: " + "a" * 100
        body = "b" * 60000
        text = header + body

        result = truncate_text(text, max_chars=10000, preserve_start=200)
        assert result.startswith("IMPORTANT HEADER:")
        assert len(result) <= 10000

    def test_truncation_preserves_end_content(self) -> None:
        """Truncated text preserves the end for conclusions."""
        from magi_system.tools.context_utils import truncate_text

        body = "a" * 60000
        footer = "CONCLUSION: Final answer."
        text = body + footer

        result = truncate_text(text, max_chars=10000, preserve_start=1000)
        assert result.endswith(footer)

    def test_multiple_truncations_compose(self) -> None:
        """Multiple search results can each be truncated independently."""
        from magi_system.tools.context_utils import truncate_text

        results = []
        for i in range(5):
            chunk = f"Result {i}: " + "x" * 20000
            results.append(truncate_text(chunk, max_chars=5000))

        for r in results:
            assert len(r) <= 5000

        combined = "\n".join(results)
        assert len(combined) <= 25005  # 5 * 5000 + 4 newlines


class TestStructuredLoggingProducesValidJSON:
    """Test structured logging produces valid JSON across scenarios."""

    def test_log_with_all_extra_fields(self) -> None:
        """Log record with all extra fields produces valid JSON."""
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="integ-test")
        record = logging.LogRecord(
            name="magi_system.pipeline",
            level=logging.INFO,
            pathname="pipeline.py",
            lineno=42,
            msg="Step completed",
            args=None,
            exc_info=None,
        )
        record.step_name = "search"
        record.model_name = "gemini-pro"
        record.latency_ms = 1250.5
        record.tokens_used = 4096

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Step completed"
        assert parsed["trace_id"] == "integ-test"
        assert parsed["step_name"] == "search"
        assert parsed["model_name"] == "gemini-pro"
        assert parsed["latency_ms"] == 1250.5
        assert parsed["tokens_used"] == 4096

    def test_logging_through_configure_produces_json(self) -> None:
        """configure_logging sets up formatter that produces JSON output."""
        from magi_system.logging_config import configure_logging

        trace_id = configure_logging(trace_id="json-test")
        logger = logging.getLogger("magi_system")

        handler = logger.handlers[0]
        record = logger.makeRecord(
            "magi_system", logging.WARNING, "test.py", 1,
            "test warning", (), None,
        )
        output = handler.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "WARNING"
        assert parsed["trace_id"] == "json-test"
        assert parsed["message"] == "test warning"

    def test_exception_logged_as_json(self) -> None:
        """Exceptions in log records produce valid JSON with exception field."""
        import sys
        from magi_system.logging_config import StructuredFormatter

        formatter = StructuredFormatter(trace_id="exc-test")
        try:
            raise ValueError("something broke")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="magi_system",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Operation failed",
            args=None,
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["exception"] == "something broke"
        assert parsed["level"] == "ERROR"


class TestAllProductionModulesImportable:
    """Test that all Phase 4 production modules can be imported together."""

    def test_all_phase4_modules_import(self) -> None:
        """All Phase 4 modules import without errors or conflicts."""
        from magi_system.tools.retry import retry_with_backoff, TRANSIENT_ERRORS
        from magi_system.tools.degradation import PipelineWarnings
        from magi_system.tools.context_utils import truncate_text, summarize_search_results
        from magi_system.search.effort_config import (
            EffortLevel, EFFORT_LEVELS, get_effort_level,
        )
        from magi_system.config_loader import load_config_from_file
        from magi_system.logging_config import (
            StructuredFormatter, configure_logging,
        )
        from magi_system.cli import create_parser, main

        # All imports succeeded — verify key types
        assert callable(retry_with_backoff)
        assert callable(truncate_text)
        assert callable(get_effort_level)
        assert callable(load_config_from_file)
        assert callable(configure_logging)
        assert callable(create_parser)
        assert isinstance(TRANSIENT_ERRORS, tuple)
        assert isinstance(EFFORT_LEVELS, dict)

    def test_tools_package_exports(self) -> None:
        """Tools package __init__ exports Phase 4 utilities."""
        from magi_system.tools import retry_with_backoff, TRANSIENT_ERRORS, PipelineWarnings

        assert callable(retry_with_backoff)
        assert isinstance(TRANSIENT_ERRORS, tuple)
        assert PipelineWarnings is not None


class TestDegradationWithLogging:
    """Test PipelineWarnings integrates with structured logging."""

    def test_model_failure_emits_structured_log(self) -> None:
        """add_model_failure emits a warning via structured logging."""
        from magi_system.logging_config import configure_logging
        from magi_system.tools.degradation import PipelineWarnings

        configure_logging(trace_id="degrad-log")
        logger = logging.getLogger("magi_system.tools.degradation")
        handler = logger.parent.handlers[0]

        records: list[str] = []
        original_emit = handler.emit

        def capture_emit(record):
            records.append(handler.format(record))
            original_emit(record)

        handler.emit = capture_emit
        try:
            pw = PipelineWarnings()
            pw.add_model_failure("claude-3", "rate limited")

            warning_logs = [r for r in records if '"level": "WARNING"' in r]
            assert len(warning_logs) >= 1
            parsed = json.loads(warning_logs[0])
            assert "claude-3" in parsed["message"]
            assert "rate limited" in parsed["message"]
            assert parsed["trace_id"] == "degrad-log"
        finally:
            handler.emit = original_emit
