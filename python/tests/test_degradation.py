"""Tests for MAGI-032: graceful degradation utilities."""


class TestPipelineWarningsInit:
    """Tests for PipelineWarnings initial state."""

    def test_starts_empty(self) -> None:
        """Test PipelineWarnings starts with empty lists."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        assert pw.warnings == []
        assert pw.unavailable_models == []

    def test_has_degradation_false_when_empty(self) -> None:
        """Test has_degradation is False when no model failures recorded."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        assert pw.has_degradation is False


class TestAddModelFailure:
    """Tests for PipelineWarnings.add_model_failure."""

    def test_records_model_name(self) -> None:
        """Test add_model_failure records the model name."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "connection refused")
        assert "gpt-4" in pw.unavailable_models

    def test_records_warning_message(self) -> None:
        """Test add_model_failure records a warning string."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "connection refused")
        assert len(pw.warnings) == 1
        assert "gpt-4" in pw.warnings[0]
        assert "connection refused" in pw.warnings[0]

    def test_has_degradation_true_after_failure(self) -> None:
        """Test has_degradation is True after a model failure."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("claude-3", "timeout")
        assert pw.has_degradation is True

    def test_multiple_failures_tracked(self) -> None:
        """Test multiple model failures are all tracked."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "connection refused")
        pw.add_model_failure("claude-3", "timeout")
        pw.add_model_failure("gemini", "rate limited")
        assert len(pw.unavailable_models) == 3
        assert len(pw.warnings) == 3
        assert "gpt-4" in pw.unavailable_models
        assert "claude-3" in pw.unavailable_models
        assert "gemini" in pw.unavailable_models


class TestAddWarning:
    """Tests for PipelineWarnings.add_warning."""

    def test_records_general_warning(self) -> None:
        """Test add_warning records a general warning."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_warning("Search timed out, using cached results")
        assert len(pw.warnings) == 1
        assert pw.warnings[0] == "Search timed out, using cached results"

    def test_general_warning_does_not_affect_degradation(self) -> None:
        """Test add_warning does not mark as degraded."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_warning("minor issue")
        assert pw.has_degradation is False
        assert len(pw.unavailable_models) == 0


class TestFormatForReport:
    """Tests for PipelineWarnings.format_for_report."""

    def test_empty_when_no_warnings(self) -> None:
        """Test format_for_report returns empty string when no warnings."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        assert pw.format_for_report() == ""

    def test_includes_model_name_when_degraded(self) -> None:
        """Test format_for_report includes unavailable model names."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "connection refused")
        report = pw.format_for_report()
        assert "## Pipeline Warnings" in report
        assert "gpt-4" in report
        assert "connection refused" in report
        assert "Confidence scores may be affected" in report

    def test_includes_general_warnings(self) -> None:
        """Test format_for_report includes general warnings."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_warning("Search timeout occurred")
        report = pw.format_for_report()
        assert "## Pipeline Warnings" in report
        assert "Search timeout occurred" in report

    def test_multiple_models_listed(self) -> None:
        """Test format_for_report lists all unavailable models."""
        from magi_system.tools.degradation import PipelineWarnings

        pw = PipelineWarnings()
        pw.add_model_failure("gpt-4", "down")
        pw.add_model_failure("claude-3", "timeout")
        report = pw.format_for_report()
        assert "gpt-4" in report
        assert "claude-3" in report


class TestDegradationExports:
    """Tests for degradation module exports."""

    def test_importable_from_tools(self) -> None:
        """Test PipelineWarnings is importable from tools package."""
        from magi_system.tools import PipelineWarnings

        assert PipelineWarnings is not None
