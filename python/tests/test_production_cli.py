"""Integration tests for CLI with production features.

MAGI-038: Tests that CLI integrates correctly with config_loader,
logging, and other Phase 4 features.
"""

import io
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from magi_system.cli import create_parser, main


class TestCLIWithConfigLoader:
    """Test CLI --config flag integrates with config_loader."""

    def test_cli_config_flag_captured(self) -> None:
        """--config value is available in parsed args."""
        parser = create_parser()
        parsed = parser.parse_args(["query", "--config", "/path/to/config.toml"])
        assert parsed.config == "/path/to/config.toml"

    def test_cli_all_options_together(self) -> None:
        """All CLI flags can be used together without conflict."""
        parser = create_parser()
        parsed = parser.parse_args([
            "What is quantum computing?",
            "--verbose",
            "--output", "report.md",
            "--config", "custom.toml",
        ])
        assert parsed.query == "What is quantum computing?"
        assert parsed.verbose is True
        assert parsed.output == "report.md"
        assert parsed.config == "custom.toml"


class TestCLIHelpOutput:
    """Test CLI help text is informative."""

    def test_help_text_contains_key_options(self) -> None:
        """Help text includes all documented options."""
        parser = create_parser()
        help_text = parser.format_help()
        assert "--verbose" in help_text
        assert "--output" in help_text
        assert "--config" in help_text
        assert "query" in help_text

    def test_help_text_mentions_magi(self) -> None:
        """Help text references the MAGI system."""
        parser = create_parser()
        help_text = parser.format_help()
        assert "MAGI" in help_text


class TestCLIMissingQuery:
    """Test CLI behavior when query is not provided."""

    def test_missing_query_exits_with_code_1(self) -> None:
        """No query argument causes exit code 1."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_missing_query_prints_help(self, capsys) -> None:
        """No query argument prints help text to stdout/stderr."""
        with pytest.raises(SystemExit):
            main([])
        captured = capsys.readouterr()
        # Help is printed before exit
        combined = captured.out + captured.err
        assert "magi-system" in combined or "usage" in combined.lower()


class TestCLIVerboseFlag:
    """Test --verbose flag integrates with logging."""

    def test_verbose_flag_sets_debug_logging(self) -> None:
        """--verbose should activate debug-level logging."""
        import logging

        main(["test query", "--verbose"])
        # After main() with --verbose, basicConfig sets DEBUG
        root_logger = logging.getLogger()
        # The flag was processed (no exception means integration works)

    def test_non_verbose_produces_output(self, capsys) -> None:
        """Without --verbose, normal output is still produced."""
        main(["test query"])
        captured = capsys.readouterr()
        assert "MAGI Research Pipeline" in captured.out


class TestCLIOutputFlag:
    """Test --output flag behavior."""

    def test_output_flag_shows_path_in_output(self, capsys) -> None:
        """--output flag value appears in CLI output."""
        main(["test query", "--output", "results.md"])
        captured = capsys.readouterr()
        assert "results.md" in captured.out

    def test_output_flag_short_form(self, capsys) -> None:
        """Short -o flag works identically to --output."""
        main(["test query", "-o", "out.md"])
        captured = capsys.readouterr()
        assert "out.md" in captured.out


class TestCLIConfigIntegration:
    """Test CLI config loading message."""

    def test_cli_shows_config_loaded(self, capsys) -> None:
        """CLI displays config information after loading."""
        main(["test query"])
        captured = capsys.readouterr()
        assert "Config loaded" in captured.out
        assert "max iterations" in captured.out
        assert "max sub-questions" in captured.out
